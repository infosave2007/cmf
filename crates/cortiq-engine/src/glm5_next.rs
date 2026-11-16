//! Text-only GLM-5.3-Flash (`glm5_next`) runtime.
//!
//! GLM is not a conventional one-residual Transformer: every block is
//! wrapped by four-stream mHC, while the mixer alternates recurrent KDA and
//! NoPE MLA/DSA.  Keeping the stack here avoids silently flattening mHC in
//! the generic pipeline.  Large matrices remain mmap-backed `QTensor`s and
//! routed experts may use the same VRAM-sized dynamic cache as Qwen/DSV4.

use crate::dsv4::{Dsv4Cfg, HcScratch, hc_block};
use crate::linear_core::{KdaCfg, KdaOutGate, KdaWeights, kda_forward};
use crate::loader::{Overlay, build_ffn_at, load_f32, load_matrix};
use crate::pipeline::{Act, DenseFfn, FfnKind, MlaWeights, MoeFfn, moe_route_with_eps};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfError, CmfModel, Glm5NextConfig, LayerType, ModelArch, TensorDtype};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const INDEX_NORM_EPS: f64 = 1e-6;

/// Sequence ids are process-unique.  Model uid alone is not enough because a
/// Pipeline can reset and reuse the same mmap-backed model for many prompts.
static NEXT_GLM_SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn next_glm_sequence() -> u64 {
    NEXT_GLM_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

// A decode token visits many MoE layers, but only one layer at a time.  Keep
// the intermediate gate/up rows on the caller thread and resize in place so
// every token does not allocate one Vec per selected expert per layer.  The
// pool joins the caller before this scratch is touched again.
thread_local! {
    static GLM_MOE_ACTS: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

struct HcWeights {
    function: Vec<f32>,
    base: Vec<f32>,
    scale: [f32; 3],
}

enum Mixer {
    Kda(KdaWeights),
    Dsa(DsaWeights),
}

struct DsaWeights {
    mla: MlaWeights,
    index_q: QTensor,
    index_k: QTensor,
    index_k_norm: Vec<f32>,
    index_k_bias: Vec<f32>,
    index_weights: QTensor,
    pool_gate: QTensor,
    pool_ape: Vec<f32>,
    /// kv_b in f32 `[nh·(nope + v), lora]`, made on first use for the
    /// absorbed attention (`dsa_absorbed`).
    kvb_f32: std::sync::OnceLock<Vec<f32>>,
}

pub struct Layer {
    attn_hc: HcWeights,
    ffn_hc: HcWeights,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    ffn: FfnKind,
    expert_ids: Vec<(usize, usize, usize)>,
    shared_ids: Option<(usize, usize, usize)>,
}

pub struct Globals {
    embed: QTensor,
    norm: Vec<f32>,
    lm_head: QTensor,
}

#[derive(Clone)]
pub struct Cfg {
    hc: Dsv4Cfg,
    kda: KdaCfg,
    eps: f64,
    swiglu_limit: f32,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    index_kpool: usize,
    index_tail: bool,
}

enum MixerState {
    Kda(Vec<f32>),
    Dsa(DsaState),
}

#[derive(Default)]
struct DsaState {
    /// Token-major normalized MLA latent `[position, kv_lora_rank]`.
    /// Keeping the compressed cache (rather than expanding every token to
    /// all heads) is what makes a 1M-token context viable on 16 GB devices.
    latents: Vec<f32>,
    /// One normalized index key and one pool-gate vector per position.
    index_keys: Vec<f32>,
    pool_gates: Vec<f32>,
}

pub struct State {
    hyper: Vec<f32>,
    mixers: Vec<MixerState>,
    gpu_pool: Option<crate::qwen4_exp::QwenGpuPool>,
    /// Device state key.  Both values are stable for the life of this host
    /// state and are replaced on every fresh sequence reset.
    model_uid: u64,
    sequence_id: u64,
    gpu_ready: bool,
    pub pos: usize,
}

impl State {
    fn reset(&mut self, layers: &[Layer], _cfg: &Cfg) {
        crate::gpu::glm_sequence_abort(self.model_uid, self.sequence_id);
        self.sequence_id = next_glm_sequence();
        self.gpu_ready = false;
        self.hyper.clear();
        self.pos = 0;
        self.mixers = layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Kda(_) => MixerState::Kda(Vec::new()),
                Mixer::Dsa(_) => MixerState::Dsa(DsaState::default()),
            })
            .collect();
    }

    fn ensure_gpu_sequence(&mut self, layers: &[Layer], cfg: &Cfg, position: usize) {
        // The native state allocator and focused kernels are deliberately
        // opt-in until the complete multi-layer chain encoder is present.
        // Keeping this seam inactive by default guarantees that a partial
        // bring-up cannot consume VRAM or perturb the authoritative CPU/Metal
        // output path.
        if std::env::var_os("CMF_GLM_NATIVE_STATE").is_none() {
            return;
        }
        if self.gpu_ready {
            if !crate::gpu::glm_sequence_reserve(
                self.model_uid,
                self.sequence_id,
                position.saturating_add(1),
            ) {
                crate::gpu::glm_sequence_abort(self.model_uid, self.sequence_id);
                self.gpu_ready = false;
            }
            return;
        }
        let (mut dsa_rank, mut dsa_index) = (0usize, 0usize);
        for layer in layers {
            if let Mixer::Dsa(w) = &layer.mixer {
                dsa_rank = dsa_rank.max(w.mla.lora);
                dsa_index = dsa_index.max(
                    w.index_q
                        .rows()
                        .checked_div(cfg.index_n_heads.max(1))
                        .unwrap_or(w.index_q.rows()),
                );
            }
        }
        self.gpu_ready = crate::gpu::glm_sequence_prepare(
            self.model_uid,
            self.sequence_id,
            layers.len(),
            cfg.hc.dim,
            cfg.hc.hc_mult,
            cfg.kda.num_heads,
            cfg.kda.head_k_dim,
            cfg.kda.head_v_dim,
            cfg.kda.conv_kernel,
            dsa_rank,
            dsa_index,
            cfg.index_topk,
            cfg.index_kpool,
        );
        if self.gpu_ready
            && !crate::gpu::glm_sequence_reserve(
                self.model_uid,
                self.sequence_id,
                position.saturating_add(1),
            )
        {
            crate::gpu::glm_sequence_abort(self.model_uid, self.sequence_id);
            self.gpu_ready = false;
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        crate::gpu::glm_sequence_abort(self.model_uid, self.sequence_id);
    }
}

fn err(s: impl Into<String>) -> CmfError {
    CmfError::Parse(format!("glm5_next: {}", s.into()))
}

fn f(model: &CmfModel, name: &str) -> Result<Vec<f32>, CmfError> {
    load_f32(model, name, &Overlay::None).map_err(err)
}

fn t(model: &Arc<CmfModel>, name: &str) -> Result<QTensor, CmfError> {
    load_matrix(model, name, false, &Overlay::None)
}

fn load_hc(model: &CmfModel, prefix: &str, kind: &str) -> Result<HcWeights, CmfError> {
    let scale = f(model, &format!("{prefix}hc_{kind}_scale"))?;
    if scale.len() != 3 {
        return Err(err(format!(
            "{prefix}hc_{kind}_scale has {} values, expected 3",
            scale.len()
        )));
    }
    Ok(HcWeights {
        function: f(model, &format!("{prefix}hc_{kind}_fn"))?,
        base: f(model, &format!("{prefix}hc_{kind}_base"))?,
        scale: [scale[0], scale[1], scale[2]],
    })
}

fn load_kda(
    model: &Arc<CmfModel>,
    prefix: &str,
    lower: Option<f32>,
) -> Result<KdaWeights, CmfError> {
    let p = format!("{prefix}kda_attn.");
    Ok(KdaWeights {
        q_proj: t(model, &format!("{p}q_proj.weight"))?,
        k_proj: t(model, &format!("{p}k_proj.weight"))?,
        v_proj: t(model, &format!("{p}v_proj.weight"))?,
        conv_q: f(model, &format!("{p}q_conv1d.weight"))?,
        conv_k: f(model, &format!("{p}k_conv1d.weight"))?,
        conv_v: f(model, &format!("{p}v_conv1d.weight"))?,
        f_a: t(model, &format!("{p}f_a_proj.weight"))?,
        f_b: t(model, &format!("{p}f_b_proj.weight"))?,
        dt_bias: f(model, &format!("{p}dt_bias"))?,
        a_log: f(model, &format!("{p}A_log"))?,
        b_proj: t(model, &format!("{p}b_proj.weight"))?,
        gate: KdaOutGate::LowRank(
            t(model, &format!("{p}g_a_proj.weight"))?,
            t(model, &format!("{p}g_b_proj.weight"))?,
        ),
        o_norm: f(model, &format!("{p}o_norm.weight"))?,
        o_proj: t(model, &format!("{p}o_proj.weight"))?,
        gate_lower_bound: lower,
    })
}

fn load_dsa(model: &Arc<CmfModel>, arch: &ModelArch, prefix: &str) -> Result<DsaWeights, CmfError> {
    let m = arch
        .mla
        .as_ref()
        .ok_or_else(|| err("missing MLA descriptor"))?;
    let p = format!("{prefix}self_attn.");
    let q_proj = t(model, &format!("{p}q_b_proj.weight"))?;
    let hd = m.qk_rope_head_dim + m.qk_nope_head_dim;
    let nh = q_proj.rows() / hd.max(1);
    let mla = MlaWeights {
        q_proj,
        q_a: Some(t(model, &format!("{p}q_a_proj.weight"))?),
        q_a_norm: Some(f(model, &format!("{p}q_a_layernorm.weight"))?),
        kv_a: t(model, &format!("{p}kv_a_proj_with_mqa.weight"))?,
        kv_a_norm: f(model, &format!("{p}kv_a_layernorm.weight"))?,
        kv_b: t(model, &format!("{p}kv_b_proj.weight"))?,
        o_proj: t(model, &format!("{p}o_proj.weight"))?,
        nh,
        qk_rope: m.qk_rope_head_dim,
        qk_nope: m.qk_nope_head_dim,
        v_dim: m.v_head_dim,
        lora: m.kv_lora_rank,
        scale: (hd as f32).sqrt().recip(),
        nope: true,
    };
    let ix = format!("{p}indexer.");
    Ok(DsaWeights {
        mla,
        index_q: t(model, &format!("{ix}wq_b.weight"))?,
        index_k: t(model, &format!("{ix}wk.weight"))?,
        index_k_norm: f(model, &format!("{ix}k_norm.weight"))?,
        index_k_bias: f(model, &format!("{ix}k_norm.bias"))?,
        index_weights: t(model, &format!("{ix}weights_proj.weight"))?,
        pool_gate: t(model, &format!("{ix}index_kpool_compress_gate"))?,
        pool_ape: f(model, &format!("{ix}index_kpool_compress_ape"))?,
        kvb_f32: std::sync::OnceLock::new(),
    })
}

fn expert_ids(m: &MoeFfn) -> Result<Vec<(usize, usize, usize)>, CmfError> {
    m.experts
        .iter()
        .enumerate()
        .map(|(e, d)| {
            let gate = d
                .gate_proj
                .model_idx()
                .ok_or_else(|| err(format!("expert {e} gate is not mmap-backed")))?;
            let up = d
                .up_proj
                .model_idx()
                .ok_or_else(|| err(format!("expert {e} up is not mmap-backed")))?;
            let down = d
                .down_proj
                .model_idx()
                .ok_or_else(|| err(format!("expert {e} down is not mmap-backed")))?;
            Ok((gate, up, down))
        })
        .collect()
}

fn shared_ids(m: &MoeFfn) -> Option<(usize, usize, usize)> {
    let (d, _) = m.shared.as_ref()?;
    Some((
        d.gate_proj.model_idx()?,
        d.up_proj.model_idx()?,
        d.down_proj.model_idx()?,
    ))
}

fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f64) {
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len().max(1) as f64;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / x.len().max(1) as f64;
    let inv = (var + eps).sqrt().recip() as f32;
    for i in 0..x.len() {
        x[i] = (x[i] - mean as f32) * inv * weight[i] + bias[i];
    }
}

/// Deterministic DSA pool selection shared by the host fallback and the
/// native selector oracle.  `torch.topk`'s score order is retained, with the
/// pool index as a stable tie-break; the visible incomplete tail is appended
/// after the selected complete pools.
fn dsa_select_positions(
    mut scored: Vec<(usize, f32)>,
    npos: usize,
    kpool: usize,
    topk: usize,
    include_tail: bool,
) -> Vec<usize> {
    let keep = (topk / kpool).min(scored.len());
    // A total score/index order is required here.  `select_nth_unstable_by`
    // is allowed to choose any member of an equal-score partition, which
    // made a tied pool edge diverge from torch.topk and from the device
    // selector even though the final sort had a tie-breaker.
    scored.sort_unstable_by(|(pa, sa), (pb, sb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(pa.cmp(pb))
    });
    scored.truncate(keep);
    let mut selected = Vec::with_capacity(keep * kpool + kpool.saturating_sub(1));
    for (pi, _) in scored {
        selected.extend(pi * kpool..(pi + 1) * kpool);
    }
    if include_tail {
        let tail = (npos / kpool) * kpool;
        selected.extend(tail..npos);
    }
    selected
}

/// Exact single-token GLM DSA. The learned indexer compresses complete
/// k-pools, selects `index_topk / kpool` pools and always carries the visible
/// incomplete tail. At <= index_topk positions this reduces to full causal
/// attention, while longer contexts stay bounded at the trained 2048 tokens.
fn dsa_forward(
    x: &[f32],
    w: &DsaWeights,
    cfg: &Cfg,
    st: &mut DsaState,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let m = &w.mla;
    let (nh, hd, dv, lora) = (m.nh, m.qk_rope + m.qk_nope, m.v_dim, m.lora);

    let qa = m.q_a.as_ref().expect("GLM DSA always has q_a");
    let mut q_resid = vec![0.0f32; qa.rows()];
    qa.matvec(x, &mut q_resid, pool);
    q_resid = crate::inference::rms_norm(
        &q_resid,
        m.q_a_norm.as_ref().expect("GLM DSA q_a norm"),
        cfg.eps,
        cortiq_core::NormStyle::Qwen,
    );
    let id = cfg.index_head_dim;
    let ih = cfg.index_n_heads;
    // These projections share their respective activations.  The Q4TP
    // matvec-many kernel keeps each pair under one pool dispatch while
    // retaining the same per-row arithmetic and output buffers.  Q2TP's
    // mixed projection geometry is deliberately left on the established
    // separate path: its specialized tiled kernels are faster than the
    // generic batched fallback on the exact release checkpoint.
    let mut q = vec![0.0f32; nh * hd];
    let mut iq = vec![0.0f32; ih * id];
    let q4_fused = |a: &QTensor, b: &QTensor| {
        matches!(
            (a, b),
            (
                QTensor::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                },
                QTensor::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                }
            )
        )
    };
    if q4_fused(&m.q_proj, &w.index_q) {
        QTensor::matvec_many([&m.q_proj, &w.index_q], &q_resid, [&mut q, &mut iq], pool);
    } else {
        m.q_proj.matvec(&q_resid, &mut q, pool);
        w.index_q.matvec(&q_resid, &mut iq, pool);
    }

    let mut compressed = vec![0.0f32; lora + m.qk_rope];
    let mut ik = vec![0.0f32; id];
    let mut gate = vec![0.0f32; id];
    let mut head_weights = vec![0.0f32; ih];
    if q4_fused(&m.kv_a, &w.index_k)
        && q4_fused(&m.kv_a, &w.pool_gate)
        && q4_fused(&m.kv_a, &w.index_weights)
    {
        QTensor::matvec_many(
            [&m.kv_a, &w.index_k, &w.pool_gate, &w.index_weights],
            x,
            [&mut compressed, &mut ik, &mut gate, &mut head_weights],
            pool,
        );
    } else {
        m.kv_a.matvec(x, &mut compressed, pool);
        w.index_k.matvec(x, &mut ik, pool);
        w.pool_gate.matvec(x, &mut gate, pool);
        w.index_weights.matvec(x, &mut head_weights, pool);
    }
    let latent = crate::inference::rms_norm(
        &compressed[..lora],
        &m.kv_a_norm,
        cfg.eps,
        cortiq_core::NormStyle::Qwen,
    );
    st.latents.extend_from_slice(&latent);
    // qk_rope is validated as zero for GLM-5.3; retain the generic geometry
    // assertion so a future rotary checkpoint fails loudly.
    debug_assert_eq!(m.qk_rope, 0);

    layer_norm(
        &mut ik,
        &w.index_k_norm,
        &w.index_k_bias,
        // The indexer's LayerNorm is deliberately fixed at 1e-6 in the
        // reference, independent of the text RMSNorm epsilon (1e-5 in the
        // release config).  Reusing cfg.eps subtly changes every pool score.
        INDEX_NORM_EPS,
    );
    st.index_keys.extend_from_slice(&ik);
    st.pool_gates.extend_from_slice(&gate);

    let npos = st.index_keys.len() / id;
    let kp = cfg.index_kpool;
    let complete = npos / kp;
    let hscale = (ih as f32).sqrt().recip();
    for v in &mut head_weights {
        *v *= hscale;
    }
    let iscale = (id as f32).sqrt().recip();
    let mut scored = Vec::with_capacity(complete);
    let mut pooled = vec![0.0f32; id];
    for pi in 0..complete {
        for d in 0..id {
            let mut mx = f32::NEG_INFINITY;
            for t0 in 0..kp {
                let pos = pi * kp + t0;
                mx = mx.max(st.pool_gates[pos * id + d] + w.pool_ape[t0 * id + d]);
            }
            let mut z = 0.0f32;
            let mut v = 0.0f32;
            for t0 in 0..kp {
                let pos = pi * kp + t0;
                let p = (st.pool_gates[pos * id + d] + w.pool_ape[t0 * id + d] - mx).exp();
                z += p;
                v += p * st.index_keys[pos * id + d];
            }
            pooled[d] = v / z.max(f32::MIN_POSITIVE);
        }
        let mut score = 0.0f32;
        for h in 0..ih {
            let dot = iq[h * id..(h + 1) * id]
                .iter()
                .zip(&pooled)
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * iscale;
            score += head_weights[h] * dot.max(0.0);
        }
        scored.push((pi, score));
    }
    // `torch.topk` returns the selected pools in score order.  Keep that
    // order (with a deterministic pool-index tie break) so the subsequent
    // f32 softmax/reduction follows the reference accumulation order.
    let selected = dsa_select_positions(scored, npos, kp, cfg.index_topk, cfg.index_tail);
    if dsa_absorb_on() && !selected.is_empty() {
        let merged = dsa_absorbed(&q, w, st, &selected, pool);
        let mut out = vec![0.0f32; cfg.hc.dim];
        m.o_proj.matvec(&merged, &mut out, pool);
        return out;
    }
    // Expand only the selected latents.  The old implementation kept an
    // `[position, head, qk+v]` cache, which is ~128 KiB/token and exhausted
    // host memory long before GLM's trained 1M context.  A single matmat
    // streams the same kv_b rows once for the bounded selection (≤2048).
    let mut selected_latent = Vec::with_capacity(selected.len() * lora);
    for &pos in &selected {
        selected_latent.extend_from_slice(&st.latents[pos * lora..(pos + 1) * lora]);
    }
    let kv_width = nh * (m.qk_nope + dv);
    let mut selected_kv = vec![0.0f32; selected.len() * kv_width];
    if !selected.is_empty() {
        m.kv_b
            .matmat(&selected_latent, selected.len(), &mut selected_kv, pool);
    }

    let mut merged = vec![0.0f32; nh * dv];
    for h in 0..nh {
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = Vec::with_capacity(selected.len());
        let mut mx = f32::NEG_INFINITY;
        for (si, &_pos) in selected.iter().enumerate() {
            let off = si * kv_width + h * (m.qk_nope + dv);
            let s = qh
                .iter()
                .zip(&selected_kv[off..off + m.qk_nope])
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * m.scale;
            mx = mx.max(s);
            scores.push(s);
        }
        let z = scores.iter().map(|&s| (s - mx).exp()).sum::<f32>();
        let out = &mut merged[h * dv..(h + 1) * dv];
        for (si, &s) in scores.iter().enumerate() {
            let p = (s - mx).exp() / z.max(f32::MIN_POSITIVE);
            let off = si * kv_width + h * (m.qk_nope + dv) + m.qk_nope;
            for d in 0..dv {
                out[d] += p * selected_kv[off + d];
            }
        }
    }
    let mut out = vec![0.0f32; cfg.hc.dim];
    m.o_proj.matvec(&merged, &mut out, pool);
    out
}

/// `CMF_GLM_DSA_ABSORB=0`: the selected latents expanded through kv_b every
/// token (the reference's order) instead of the absorbed form.
fn dsa_absorb_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_GLM_DSA_ABSORB").as_deref() != Ok("0"))
}

/// DSA's attention with kv_b absorbed (MLA's inference form): per head the
/// query goes through K's half of kv_b once (`q' = W_kᵀ q`, lora wide), the
/// scores are `q' · latent` over the selected positions, the latents' softmax
/// mix goes through V's half. Exactly the expanded product up to f32
/// rounding, at ~nh·lora·(nope + v + 2·n) multiply-adds instead of
/// n·nh·(nope + v)·lora — at 512 positions 50 M instead of 8.6 G a layer.
fn dsa_absorbed(q: &[f32], w: &DsaWeights, st: &DsaState, selected: &[usize], pool: Option<&Pool>) -> Vec<f32> {
    let m = &w.mla;
    let (nh, hd, nope, dv, lora) = (m.nh, m.qk_rope + m.qk_nope, m.qk_nope, m.v_dim, m.lora);
    let stride = nope + dv;
    let wkv = w.kvb_f32.get_or_init(|| {
        let rows = m.kv_b.rows();
        let mut v = vec![0.0f32; rows * lora];
        for (r, row) in v.chunks_exact_mut(lora).enumerate() {
            m.kv_b.row_f32(r, row);
        }
        v
    });
    let mut merged = vec![0.0f32; nh * dv];
    let ptr = crate::audiovae::SendPtr(merged.as_mut_ptr());
    let head = |h: usize| {
        let qh = &q[h * hd + m.qk_rope..h * hd + m.qk_rope + nope];
        let wk = &wkv[h * stride * lora..(h * stride + nope) * lora];
        let wv = &wkv[(h * stride + nope) * lora..(h + 1) * stride * lora];
        let mut qp = vec![0.0f32; lora];
        for (d, &qd) in qh.iter().enumerate() {
            for (a, &b) in qp.iter_mut().zip(&wk[d * lora..(d + 1) * lora]) {
                *a += qd * b;
            }
        }
        let mut scores: Vec<f32> = selected
            .iter()
            .map(|&pos| qp.iter().zip(&st.latents[pos * lora..(pos + 1) * lora]).map(|(&a, &b)| a * b).sum::<f32>() * m.scale)
            .collect();
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0f32;
        for s in &mut scores {
            *s = (*s - mx).exp();
            z += *s;
        }
        let z = z.max(f32::MIN_POSITIVE);
        let mut mix = vec![0.0f32; lora];
        for (&p, &pos) in scores.iter().zip(selected) {
            let p = p / z;
            for (a, &b) in mix.iter_mut().zip(&st.latents[pos * lora..(pos + 1) * lora]) {
                *a += p * b;
            }
        }
        // SAFETY: each head writes its own dv-wide slice.
        let out = unsafe { ptr.row(h * dv, dv) };
        for (d, o) in out.iter_mut().enumerate() {
            *o = wv[d * lora..(d + 1) * lora].iter().zip(&mix).map(|(&a, &b)| a * b).sum();
        }
    };
    match pool {
        Some(p) => p.run_rows(nh, &|lo, hi| (lo..hi).for_each(&head)),
        None => (0..nh).for_each(head),
    }
    merged
}

/// Load the exact GLM text stack. The vision tower is intentionally omitted
/// by the converter; image/video tokens are therefore not accepted by CMF.
pub fn load(
    model: &Arc<CmfModel>,
    arch: &ModelArch,
) -> Result<(Globals, Vec<Layer>, Cfg, State), CmfError> {
    let gc: &Glm5NextConfig = arch
        .glm5_next
        .as_ref()
        .ok_or_else(|| err("missing glm5_next descriptor"))?;
    if arch.layer_types.len() != arch.num_layers {
        return Err(err(format!(
            "layer schedule has {} entries, expected {}",
            arch.layer_types.len(),
            arch.num_layers
        )));
    }
    let mla = arch
        .mla
        .as_ref()
        .ok_or_else(|| err("missing MLA geometry"))?;
    if gc.hc_mult == 0
        || gc.index_n_heads == 0
        || gc.index_head_dim == 0
        || gc.index_topk == 0
        || gc.index_kpool == 0
        || gc.index_topk % gc.index_kpool != 0
    {
        return Err(err("invalid mHC/DSA geometry"));
    }
    let kda = KdaCfg {
        num_heads: arch
            .linear_num_key_heads
            .ok_or_else(|| err("missing KDA heads"))?,
        head_k_dim: arch
            .linear_key_head_dim
            .ok_or_else(|| err("missing KDA key dim"))?,
        head_v_dim: arch
            .linear_value_head_dim
            .ok_or_else(|| err("missing KDA value dim"))?,
        conv_kernel: arch
            .linear_conv_kernel_dim
            .ok_or_else(|| err("missing KDA conv"))?,
        hidden_size: arch.hidden_size,
        rms_eps: arch.rms_norm_eps,
    };
    let qk = mla.qk_rope_head_dim + mla.qk_nope_head_dim;
    let hc = Dsv4Cfg {
        dim: arch.hidden_size,
        n_heads: arch.num_attention_heads,
        head_dim: qk,
        rope_head_dim: mla.qk_rope_head_dim,
        q_lora_rank: mla.q_lora_rank.unwrap_or(0),
        o_lora_rank: 0,
        o_groups: 1,
        hc_mult: gc.hc_mult,
        hc_sinkhorn_iters: gc.hc_sinkhorn_iters,
        hc_eps: gc.hc_eps,
        norm_eps: arch.rms_norm_eps as f32,
        n_routed_experts: arch.moe.as_ref().map_or(0, |m| m.num_experts),
        top_k: arch.moe.as_ref().map_or(0, |m| m.top_k),
        moe_inter: arch.moe.as_ref().map_or(0, |m| m.moe_intermediate_size),
        route_scale: arch
            .moe
            .as_ref()
            .and_then(|m| m.routed_scaling_factor)
            .unwrap_or(1.0),
        swiglu_limit: gc.swiglu_limit,
        window: 0,
        index_topk: gc.index_topk,
        vocab: arch.vocab_size,
    };
    let cfg = Cfg {
        hc,
        kda,
        eps: arch.rms_norm_eps,
        swiglu_limit: gc.swiglu_limit,
        index_n_heads: gc.index_n_heads,
        index_head_dim: gc.index_head_dim,
        index_topk: gc.index_topk,
        index_kpool: gc.index_kpool,
        index_tail: gc.index_kpool_always_select_tail,
    };
    let globals = Globals {
        embed: t(model, "model.embed_tokens.weight")?,
        norm: f(model, "model.norm.weight")?,
        lm_head: t(model, "lm_head.weight")?,
    };
    let mut layers = Vec::with_capacity(arch.num_layers);
    for li in 0..arch.num_layers {
        let p = format!("model.layers.{li}.");
        let mixer = match arch.layer_types.get(li) {
            Some(LayerType::Kda) => Mixer::Kda(load_kda(
                model,
                &p,
                arch.kda_gate_lower_bound.map(|v| v as f32),
            )?),
            _ => Mixer::Dsa(load_dsa(model, arch, &p)?),
        };
        let ffn = build_ffn_at(model, arch, &p, false, &Overlay::None)?;
        let (ids, shared) = match &ffn {
            FfnKind::Moe(m) => (expert_ids(m)?, shared_ids(m)),
            _ => (Vec::new(), None),
        };
        layers.push(Layer {
            attn_hc: load_hc(model, &p, "attn")?,
            ffn_hc: load_hc(model, &p, "ffn")?,
            input_norm: f(model, &format!("{p}input_layernorm.weight"))?,
            post_norm: f(model, &format!("{p}post_attention_layernorm.weight"))?,
            mixer,
            ffn,
            expert_ids: ids,
            shared_ids: shared,
        });
    }
    let mut state = State {
        hyper: Vec::new(),
        mixers: Vec::new(),
        gpu_pool: None,
        model_uid: model.uid(),
        sequence_id: next_glm_sequence(),
        gpu_ready: false,
        pos: 0,
    };
    state.reset(&layers, &cfg);
    Ok((globals, layers, cfg, state))
}

fn dense_clamped(d: &DenseFfn, x: &[f32], limit: f32, pool: Option<&Pool>) -> Vec<f32> {
    let mut out = vec![0.0f32; d.down_proj.rows()];
    expert_swiglu_qtensor(
        x,
        &d.gate_proj,
        &d.up_proj,
        &d.down_proj,
        d.gate_proj.rows(),
        1.0,
        limit,
        pool,
        &mut out,
    );
    out
}

/// SwiGLU with the two projections sharing one input.  `QTensor::matvec_many`
/// keeps their row kernels under one pool dispatch for the large Q4TP expert
/// matrices; serial and mixed-dtype callers retain the exact legacy kernels.
/// The down projection remains a separate pass because it consumes the gated
/// product, so this helper changes synchronization only, never arithmetic.
#[allow(clippy::too_many_arguments)]
fn expert_swiglu_qtensor(
    x: &[f32],
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    inter: usize,
    weight: f32,
    limit: f32,
    pool: Option<&Pool>,
    out: &mut [f32],
) {
    // Q2TP's established SwiGLU helper is the numerical oracle for GLM's
    // real checkpoint.  The fused row kernel is useful for synthetic/Q4
    // probes, but on Q2 the changed accumulation order can move the greedy
    // vocabulary edge even when its per-row dot product is exact.  Keep the
    // production Q2 path on the legacy temporary-and-accumulate helper until
    // a full-logit parity proof exists; this still benefits from the shared
    // Q2TP matvec kernel itself.
    if gate.model_dtype() == Some(TensorDtype::Q2TiledP) {
        crate::dsv4::expert_swiglu(
            x,
            &|x, y| gate.matvec(x, y, pool),
            &|x, y| up.matvec(x, y, pool),
            &|x, y| down.matvec(x, y, pool),
            inter,
            weight,
            limit,
            out,
        );
        return;
    }
    // Keep the accepted Q4TP serial path byte-for-byte on the established
    // helper.  Its three ordinary row dispatches are faster on the large
    // Q4 checkpoint than the generic pair dispatcher (and preserve the
    // repair-5 throughput/behavior envelope).  Q2's release path below uses
    // the dedicated batched gate/up + down kernels instead.
    if gate.model_dtype() == Some(TensorDtype::Q4TiledP) {
        crate::dsv4::expert_swiglu(
            x,
            &|x, y| gate.matvec(x, y, pool),
            &|x, y| up.matvec(x, y, pool),
            &|x, y| down.matvec(x, y, pool),
            inter,
            weight,
            limit,
            out,
        );
        return;
    }
    // A fused pair is still useful for synthetic/mixed-dtype probes, where
    // the legacy helper cannot share the activation rows.
    let mut gate_buf = vec![0.0f32; inter];
    if QTensor::matvec_silu_mul_limited(gate, up, x, &mut gate_buf, limit, pool) {
        if weight != 1.0 {
            for v in gate_buf.iter_mut() {
                *v *= weight;
            }
        }
        down.matvec(&gate_buf, out, pool);
        return;
    }
    let mut up_buf = vec![0.0f32; inter];
    let fused_q4 = matches!(
        (gate, up, down),
        (
            QTensor::Mapped {
                dtype: TensorDtype::Q4TiledP,
                ..
            },
            QTensor::Mapped {
                dtype: TensorDtype::Q4TiledP,
                ..
            },
            QTensor::Mapped {
                dtype: TensorDtype::Q4TiledP,
                ..
            }
        )
    );
    if fused_q4 {
        QTensor::matvec_many([gate, up], x, [&mut gate_buf, &mut up_buf], pool);
    } else {
        // Q2TP and mixed-dtype experts stay on their established individual
        // tiled kernels; the generic batched dispatcher is slower there.
        gate.matvec(x, &mut gate_buf, pool);
        up.matvec(x, &mut up_buf, pool);
    }
    if limit > 0.0 {
        for u in up_buf.iter_mut() {
            *u = u.clamp(-limit, limit);
        }
        for g in gate_buf.iter_mut() {
            *g = g.min(limit);
        }
    }
    for (g, u) in gate_buf.iter_mut().zip(&up_buf) {
        *g = (*g / (1.0 + (-*g).exp())) * *u * weight;
    }
    down.matvec(&gate_buf, out, pool);
}

/// Decide whether GLM's dynamic expert path is eligible for automatic use.
///
/// Q2 gate/up payloads have a bounded explicit Vulkan path, but the measured
/// RTX 3090 decode is slower than exact CPU after applying the physical-VRAM
/// reserve.  Automatic mode therefore stays on CPU for both profiles; an
/// operator can still force the bounded dynamic path with
/// `CMF_GLM_DYNAMIC_MOE=1`.
fn glm_dynamic_moe_enabled(mode: Option<&str>) -> bool {
    match mode {
        Some("1") => true,
        Some("0") => false,
        None | Some("auto") => false,
        Some(_) => false,
    }
}

/// Keep the host-side GLM trunk on the exact CPU kernels while the explicitly
/// forced dynamic path exercises only the resident-expert frame on Vulkan.
/// GLM's KDA/DSA and HyperConnection state are scalar, token-at-a-time
/// operations: sending each projection over PCIe costs more than its kernel.
/// Forced dynamic mode uses this split by default; the env switch is retained
/// as a narrow A/B escape hatch for a backend that has a faster static arm.
#[cfg(feature = "gpu")]
fn glm_gpu_moe_only_enabled(dynamic_enabled: bool) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("CMF_GLM_GPU_MOE_ONLY").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            Some("1") | Some("true") | Some("on") => true,
            _ => dynamic_enabled,
        },
    )
}

/// Allow a release run to disable the batched CPU MoE path for an A/B profile
/// without changing the default policy.  The switch is intentionally opt-in
/// and process-local: `CMF_GLM_MOE_BATCH=0` is evidence-only instrumentation,
/// while every normal invocation keeps the exact batched implementation.
fn glm_moe_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Batch gate/up and down rows by default; `CMF_GLM_MOE_BATCH=0` keeps
        // the old serial path available for an explicit parity A/B.  The
        // implementation retains the release Q2/Q4 oracle after the Q2
        // SIMD experiment was rejected for changing greedy output.
        !matches!(
            std::env::var("CMF_GLM_MOE_BATCH").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        )
    })
}

/// Optional per-layer wall-clock profile for the bounded Q2 CPU path.  It is
/// deliberately off by default so the release policy has no logging or
/// timing overhead; `CMF_GLM_CPU_PROF=1` is used only for evidence runs.
fn glm_cpu_profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_GLM_CPU_PROF").ok().as_deref() == Some("1"))
}

/// Exact CPU GLM MoE with one gate/up dispatch and one down dispatch for all
/// selected experts.  Q2TP gate/up and Q4TP down are the release profile, so
/// this reuses the shared batched kernels instead of creating a second
/// GLM-only execution subsystem.  `None` means the dtypes/shape are outside
/// the shared kernel contract and the caller must use the serial fallback.
fn moe_cpu_batched(
    m: &MoeFfn,
    x: &[f32],
    idx: &[usize],
    p: &[f32],
    wsum: f32,
    limit: f32,
    pool: Option<&Pool>,
) -> Option<Vec<f32>> {
    if idx.is_empty() {
        return None;
    }
    let n = idx.len() + usize::from(m.shared.is_some());
    let mut pairs = Vec::with_capacity(n);
    let mut downs = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for &e in idx {
        let d = m.experts.get(e)?;
        if d.act != Act::Silu {
            return None;
        }
        pairs.push((&d.gate_proj, &d.up_proj));
        downs.push(&d.down_proj);
        weights.push(p[e] / wsum * m.per_expert_scale.as_ref().map_or(1.0, |v| v[e]));
    }
    // Preserve the serial order: routed experts first, always-on shared last.
    if let Some((shared, gate)) = &m.shared {
        if shared.act != Act::Silu {
            return None;
        }
        let shared_weight = gate.as_ref().map_or(1.0, |g| {
            let mut y = [0.0f32; 1];
            g.matvec(x, &mut y, pool);
            1.0 / (1.0 + (-y[0]).exp())
        });
        pairs.push((&shared.gate_proj, &shared.up_proj));
        downs.push(&shared.down_proj);
        weights.push(shared_weight);
    }

    let inter = pairs.first()?.0.rows();
    GLM_MOE_ACTS.with(|scratch| {
        let mut activations = scratch.borrow_mut();
        activations.truncate(n);
        while activations.len() < n {
            activations.push(Vec::new());
        }
        for a in activations.iter_mut() {
            a.resize(inter, 0.0);
        }
        if !QTensor::moe_gate_up_many_limited(&pairs, x, &mut activations[..], limit, pool) {
            return None;
        }
        let mut out = vec![0.0f32; x.len()];
        if !QTensor::moe_down_many(&downs, &activations[..], &weights, &mut out, pool) {
            return None;
        }
        Some(out)
    })
}

/// Exact CPU completion for the routed GLM experts that are not resident in
/// the device arena.  This mirrors `moe_cold_experts_cpu` from the shared
/// pipeline, but keeps GLM's positive SwiGLU clamp in the fused path; using
/// the unclamped helper here would silently change the model on a cache miss.
fn glm_cold_experts_cpu(
    experts: &[(&DenseFfn, f32)],
    x: &[f32],
    limit: f32,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if experts.is_empty() {
        return out;
    }
    let pairs: Vec<_> = experts
        .iter()
        .map(|(e, _)| (&e.gate_proj, &e.up_proj))
        .collect();
    let downs: Vec<_> = experts.iter().map(|(e, _)| &e.down_proj).collect();
    let weights: Vec<_> = experts.iter().map(|(_, w)| *w).collect();
    let inter = experts[0].0.gate_proj.rows();
    let mut activations: Vec<Vec<f32>> = (0..experts.len()).map(|_| vec![0.0; inter]).collect();
    if QTensor::moe_gate_up_many_limited(&pairs, x, &mut activations, limit, pool)
        && QTensor::moe_down_many(&downs, &activations, &weights, &mut out, pool)
    {
        return out;
    }
    out.fill(0.0);
    for &(expert, weight) in experts {
        let mut one = vec![0.0f32; x.len()];
        expert_swiglu_qtensor(
            x,
            &expert.gate_proj,
            &expert.up_proj,
            &expert.down_proj,
            expert.gate_proj.rows(),
            weight,
            limit,
            pool,
            &mut one,
        );
        for (o, v) in out.iter_mut().zip(one) {
            *o += v;
        }
    }
    out
}

fn moe_cpu(m: &MoeFfn, x: &[f32], limit: f32, pool: Option<&Pool>) -> Vec<f32> {
    let mut logits = vec![0.0f32; m.experts.len()];
    m.router.matvec(x, &mut logits, pool);
    // GLM's noaux_tc router renormalizes selected sigmoid scores with a
    // 1e-20 floor (the generic Qwen/LFM2 helper intentionally keeps its
    // historical 1e-6 floor).
    let (idx, p, wsum) = moe_route_with_eps(&logits, m, None, 1e-20);
    // Keep the same trace format as the generic pipeline for cache/locality
    // profiling; with no `CMF_MOE_TRACE` the helper is a cheap no-op.
    crate::pipeline::moe_trace_at(crate::gpu::cur_layer() as i32, &idx);
    // The batched Q2TP path is the repair-7 target.  Keep Q4TP on the
    // accepted serial helper: its larger rows make one giant virtual dispatch
    // disproportionately expensive on this CPU/NFS-backed checkpoint, and
    // repair-5 already established its behavior and throughput.
    let q2_batch =
        m.experts.first().and_then(|d| d.gate_proj.model_dtype()) == Some(TensorDtype::Q2TiledP);
    if q2_batch && glm_moe_batch_enabled() {
        if let Some(out) = moe_cpu_batched(m, x, &idx, &p, wsum, limit, pool) {
            return out;
        }
    }
    let mut out = vec![0.0f32; x.len()];
    for &e in &idx {
        let d = &m.experts[e];
        let mut one = vec![0.0f32; x.len()];
        expert_swiglu_qtensor(
            x,
            &d.gate_proj,
            &d.up_proj,
            &d.down_proj,
            d.gate_proj.rows(),
            p[e] / wsum,
            limit,
            pool,
            &mut one,
        );
        for (o, v) in out.iter_mut().zip(one) {
            *o += v;
        }
    }
    if let Some((d, shared_gate)) = &m.shared {
        let weight = shared_gate.as_ref().map_or(1.0, |g| {
            let mut y = [0.0f32; 1];
            g.matvec(x, &mut y, pool);
            1.0 / (1.0 + (-y[0]).exp())
        });
        let mut one = vec![0.0f32; x.len()];
        expert_swiglu_qtensor(
            x,
            &d.gate_proj,
            &d.up_proj,
            &d.down_proj,
            d.gate_proj.rows(),
            weight,
            limit,
            pool,
            &mut one,
        );
        for (o, v) in out.iter_mut().zip(one) {
            *o += v;
        }
    }
    out
}

/// The routed FFN on a CUDA card (`gpu::moe_row`): the host routes as
/// `moe_cpu` does, the card computes the picks (built into its slots from
/// the mapped file) and the shared expert. `CMF_GLM_CUDA_MOE=0` keeps them on
/// the host; a refusal does too.
fn cuda_moe(layer: &Layer, x: &[f32], limit: f32, pool: Option<&Pool>) -> Option<Vec<f32>> {
    if !crate::gpu::cuda_active() || std::env::var("CMF_GLM_CUDA_MOE").as_deref() == Ok("0") {
        return None;
    }
    let FfnKind::Moe(m) = &layer.ffn else {
        return None;
    };
    if m.route_tau.is_some() || m.mask.is_some() {
        return None;
    }
    let first = m.experts.first()?;
    let model = first.gate_proj.model_arc()?;
    let inter = first.gate_proj.rows();
    let mut logits = vec![0.0f32; m.experts.len()];
    m.router.matvec(x, &mut logits, pool);
    let (idx, p, wsum) = moe_route_with_eps(&logits, m, None, 1e-20);
    crate::pipeline::moe_trace_at(crate::gpu::cur_layer() as i32, &idx);
    let picks: Vec<(usize, f32)> = idx.iter().map(|&e| (e, p[e] / wsum)).collect();
    let shared = match (&m.shared, layer.shared_ids) {
        (Some((d, gate)), Some(ids)) => {
            let w = gate.as_ref().map_or(1.0, |g| {
                let mut y = [0.0f32; 1];
                g.matvec(x, &mut y, pool);
                1.0 / (1.0 + (-y[0]).exp())
            });
            Some((ids, d.gate_proj.rows(), w))
        }
        (None, _) => None,
        (Some(_), None) => return None,
    };
    let mut out = vec![0.0f32; x.len()];
    crate::gpu::moe_row(&model, &layer.expert_ids, shared, inter, limit, x, &picks, &mut out).then_some(out)
}

fn ffn_cpu(ffn: &FfnKind, x: &[f32], limit: f32, pool: Option<&Pool>) -> Vec<f32> {
    match ffn {
        FfnKind::Dense(d) => dense_clamped(d, x, limit, pool),
        FfnKind::Moe(m) => moe_cpu(m, x, limit, pool),
        FfnKind::DenseMoe(_) => unreachable!("GLM has no DenseMoe layer"),
    }
}

#[cfg(feature = "gpu")]
fn dynamic_moe_gpu(
    layer: &Layer,
    li: usize,
    x: &[f32],
    gpu_pool: &mut Option<crate::qwen4_exp::QwenGpuPool>,
    cfg: &Cfg,
    n_layers: usize,
    pool: Option<&Pool>,
) -> Option<Vec<f32>> {
    let FfnKind::Moe(m) = &layer.ffn else {
        return None;
    };
    if !crate::gpu::enabled_here() || m.route_tau.is_some() || m.mask.is_some() {
        return None;
    }
    let model = m.experts.first()?.gate_proj.model_arc()?;
    let gu_q2 = m.experts.first()?.gate_proj.model_dtype() == Some(TensorDtype::Q2TiledP);
    let enabled = glm_dynamic_moe_enabled(std::env::var("CMF_GLM_DYNAMIC_MOE").ok().as_deref());
    if !enabled {
        return None;
    }
    if gpu_pool.is_none() {
        *gpu_pool = crate::qwen4_exp::QwenGpuPool::create_for_glm(
            &model,
            m.experts.first()?.gate_proj.rows(),
            x.len(),
            n_layers,
            m.experts.len(),
            gu_q2,
        );
    }
    let mut logits = vec![0.0f32; m.experts.len()];
    m.router.matvec(x, &mut logits, pool);
    // Keep the GPU and CPU branches on the same GLM router denominator.
    let (picks, probabilities, wsum) = moe_route_with_eps(&logits, m, None, 1e-20);
    let (remap, shared_slot) =
        gpu_pool
            .as_mut()?
            .ensure(&model, li, &picks, &layer.expert_ids, layer.shared_ids)?;
    let mut mix = vec![0.0f32; m.experts.len()];
    for &e in &picks {
        mix[e] = probabilities[e] / wsum;
    }
    let cold_ids: Vec<usize> = picks
        .iter()
        .copied()
        .filter(|&e| remap[e] == u32::MAX)
        .collect();
    let shared_weight = m.shared.as_ref().map_or(1.0, |(_, gate)| {
        gate.as_ref().map_or(1.0, |g| {
            let mut y = [0.0f32; 1];
            g.matvec(x, &mut y, pool);
            1.0 / (1.0 + (-y[0]).exp())
        })
    });
    let weights = crate::gpu_wgpu::Dsv4MoeW {
        router: &[],
        experts: &layer.expert_ids,
        logits: &logits,
        bias: Some(&mix),
        mask: None,
        forced: Some(&picks),
        remap: Some(&remap),
        global: Some(crate::gpu_wgpu::Dsv4GlobalMoe {
            pool_uid: model.uid(),
            shared_slot,
            segment_slots: gpu_pool.as_ref()?.segment_slots as u32,
        }),
        has_shared: m.shared.is_some(),
        shared_weight,
        preweighted: true,
        qwen_softmax: true,
    };
    let geom = crate::gpu_wgpu::Dsv4MoeGeom {
        hidden: x.len(),
        inter: m.experts.first()?.gate_proj.rows(),
        top_k: m.top_k,
        route_scale: 1.0,
        swiglu_limit: cfg.swiglu_limit,
        gu_q2,
        bf16: false,
    };
    let mut out = vec![0.0f32; x.len()];
    let mut cold = Vec::new();
    let mut cold_x = Vec::new();
    // The GPU frame and cache-miss experts are independent: resident experts
    // consume the device's arena while cold experts read immutable model
    // pages on the host.  Run both arms in parallel, as Qwen's proven path
    // does, so a miss no longer serializes a full gate/up/down completion
    // behind every one of the 42 layer submissions.
    let cold_jobs: Vec<_> = cold_ids
        .iter()
        .copied()
        .map(|e| (&m.experts[e], mix[e]))
        .collect();
    let (frame_ok, cold_cpu) = std::thread::scope(|scope| {
        let cpu = (!cold_jobs.is_empty())
            .then(|| scope.spawn(|| glm_cold_experts_cpu(&cold_jobs, x, cfg.swiglu_limit, pool)));
        let ok = crate::gpu_wgpu::dsv4_moe_frame(
            &model,
            &weights,
            geom,
            x,
            &mut cold,
            &mut cold_x,
            None,
            None,
            &mut out,
        );
        let early = cpu
            .map(|job| job.join().ok())
            .flatten()
            .unwrap_or_else(|| vec![0.0; x.len()]);
        (ok, early)
    });
    if !frame_ok
        || cold.len() != cold_jobs.len()
        || cold.iter().map(|&(e, _)| e).ne(cold_ids.iter().copied())
    {
        return None;
    }
    // Complete cache misses exactly on the CPU. The resident subset has
    // already been mixed with its final pre-normalized weights on device.
    for (o, c) in out.iter_mut().zip(&cold_cpu) {
        *o += c;
    }
    Some(out)
}

/// Decode one token. DSA is mathematically full attention until the 2048
/// token budget is exceeded; the standard MLA cache is therefore exact for
/// ordinary prompts and remains the portable CPU/Metal fallback.
pub fn forward_token(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    token_id: u32,
    position: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    let prof = std::env::var_os("CMF_GLM_PROF").is_some();
    #[cfg(feature = "gpu")]
    let gpu_before = if prof {
        use std::sync::atomic::Ordering;
        Some((
            crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_GPU_NS[0].load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_GPU_N.load(Ordering::Relaxed),
            crate::gpu_wgpu::DSV4_FILLS.load(Ordering::Relaxed),
            crate::gpu_wgpu::DSV4_FILL_BYTES.load(Ordering::Relaxed),
            crate::gpu_wgpu::SUBMITS.load(Ordering::Relaxed),
            crate::gpu_wgpu::PASSES.load(Ordering::Relaxed),
            crate::gpu_wgpu::residency_stats(),
        ))
    } else {
        None
    };
    let token_t0 = std::time::Instant::now();
    if position == 0 || state.pos != position {
        state.reset(layers, cfg);
    }
    // Bring up the native persistent state before any layer touches the
    // mixer.  This is a capability/allocation probe only: the exact host
    // implementation remains authoritative until a complete GLM chain is
    // enabled, and any later cache-growth refusal tears the state down.
    state.ensure_gpu_sequence(layers, cfg, position);
    let mut emb = vec![0.0f32; cfg.hc.dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut emb);
    }
    state.hyper.clear();
    for _ in 0..cfg.hc.hc_mult {
        state.hyper.extend_from_slice(&emb);
    }
    let mut scratch = HcScratch::new(&cfg.hc);
    let (hyper, mixers, gpu_pool) = (&mut state.hyper, &mut state.mixers, &mut state.gpu_pool);
    let profile = glm_cpu_profile_enabled();
    for (li, layer) in layers.iter().enumerate() {
        crate::gpu::set_layer(li as i64);
        let mixer_state = &mut mixers[li];
        let attn_t0 = profile.then(std::time::Instant::now);
        #[cfg(feature = "gpu")]
        let gpu_moe_only = glm_gpu_moe_only_enabled(glm_dynamic_moe_enabled(
            std::env::var("CMF_GLM_DYNAMIC_MOE").ok().as_deref(),
        ));
        #[cfg(not(feature = "gpu"))]
        let gpu_moe_only = false;
        let mut run_attn = || {
            hc_block(
                hyper,
                &layer.attn_hc.function,
                &layer.attn_hc.scale,
                &layer.attn_hc.base,
                &layer.input_norm,
                &cfg.hc,
                &mut scratch,
                pool,
                |x, out| {
                    let y = match (&layer.mixer, &mut *mixer_state) {
                        (Mixer::Kda(w), MixerState::Kda(st)) => {
                            kda_forward(x, w, &cfg.kda, st, pool)
                        }
                        (Mixer::Dsa(w), MixerState::Dsa(st)) => dsa_forward(x, w, cfg, st, pool),
                        _ => unreachable!("GLM mixer state mismatch"),
                    };
                    out.copy_from_slice(&y);
                },
            )
        };
        if gpu_moe_only {
            crate::gpu::cpu_scope(run_attn);
        } else {
            run_attn();
        }
        let attn_ms = attn_t0.map(|t| t.elapsed().as_secs_f64() * 1e3);
        let ffn_t0 = profile.then(std::time::Instant::now);
        let mut run_ffn = || {
            hc_block(
                hyper,
                &layer.ffn_hc.function,
                &layer.ffn_hc.scale,
                &layer.ffn_hc.base,
                &layer.post_norm,
                &cfg.hc,
                &mut scratch,
                pool,
                |x, out| {
                    #[cfg(feature = "gpu")]
                    let gpu = dynamic_moe_gpu(layer, li, x, gpu_pool, cfg, layers.len(), pool);
                    #[cfg(not(feature = "gpu"))]
                    let gpu: Option<Vec<f32>> = None;
                    let gpu = gpu.or_else(|| cuda_moe(layer, x, cfg.swiglu_limit, pool));
                    let y = gpu.unwrap_or_else(|| ffn_cpu(&layer.ffn, x, cfg.swiglu_limit, pool));
                    out.copy_from_slice(&y);
                },
            )
        };
        // Keep the MoE frame outside a CPU scope: dynamic_moe_gpu must see
        // enabled_here() for its router/projection dispatches.  The HC math
        // itself is already scalar CPU work and remains in the closure.
        run_ffn();
        if let (Some(attn_ms), Some(ffn_ms)) =
            (attn_ms, ffn_t0.map(|t| t.elapsed().as_secs_f64() * 1e3))
        {
            let mixer = match layer.mixer {
                Mixer::Kda(_) => "kda",
                Mixer::Dsa(_) => "dsa",
            };
            eprintln!("glm-prof layer={li} mixer={mixer} attn_ms={attn_ms:.3} ffn_ms={ffn_ms:.3}");
        }
    }
    crate::gpu::set_layer(-1);
    if want_logits {
        let mut hidden = vec![0.0f32; cfg.hc.dim];
        for stream in hyper.chunks(cfg.hc.dim) {
            for (h, &v) in hidden.iter_mut().zip(stream) {
                *h += v / cfg.hc.hc_mult as f32;
            }
        }
        hidden = crate::inference::rms_norm(
            &hidden,
            &globals.norm,
            cfg.eps,
            cortiq_core::NormStyle::Qwen,
        );
        logits.resize(globals.lm_head.rows(), 0.0);
        globals.lm_head.matvec(&hidden, logits, pool);
    } else {
        logits.clear();
    }
    state.pos = position + 1;
    if prof {
        eprintln!(
            "glm-prof pos={position} total={:.3}s",
            token_t0.elapsed().as_secs_f64()
        );
        #[cfg(feature = "gpu")]
        if let Some((enc0, wait0, card0, calls0, fills0, fill_bytes0, submits0, passes0, res0)) =
            gpu_before
        {
            use std::sync::atomic::Ordering;
            let enc = crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed) - enc0;
            let wait = crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed) - wait0;
            let card = crate::gpu_wgpu::MOE_GPU_NS[0].load(Ordering::Relaxed) - card0;
            let calls = crate::gpu_wgpu::MOE_GPU_N.load(Ordering::Relaxed) - calls0;
            let fills = crate::gpu_wgpu::DSV4_FILLS.load(Ordering::Relaxed) - fills0;
            let fill_bytes = crate::gpu_wgpu::DSV4_FILL_BYTES.load(Ordering::Relaxed) - fill_bytes0;
            let submits = crate::gpu_wgpu::SUBMITS.load(Ordering::Relaxed) - submits0;
            let passes = crate::gpu_wgpu::PASSES.load(Ordering::Relaxed) - passes0;
            let (hits0, misses0, bytes0, evicts0) = res0;
            let (hits, misses, bytes, evicts) = crate::gpu_wgpu::residency_stats();
            eprintln!(
                "glm-moe-gpu pos={position} calls={calls} fills={fills} fill={:.1}MB \
                 encode={:.1}ms wait={:.1}ms card={:.1}ms submits={submits} passes={passes} \
                 residency=+{}h/+{}m/{:.1}MB/+{}e",
                fill_bytes as f64 / 1e6,
                enc as f64 / 1e6,
                wait as f64 / 1e6,
                card as f64 / 1e6,
                hits.saturating_sub(hits0),
                misses.saturating_sub(misses0),
                bytes.saturating_sub(bytes0) as f64 / 1e6,
                evicts.saturating_sub(evicts0),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dsa_select_positions, glm_dynamic_moe_enabled, moe_cpu};
    use crate::pipeline::{Act, DenseFfn, MoeFfn};
    use crate::qtensor::QTensor;

    fn dense(gate: &[f32], up: &[f32], down: &[f32]) -> DenseFfn {
        DenseFfn {
            gate_proj: QTensor::from_f32(gate.to_vec(), 2, 2),
            up_proj: QTensor::from_f32(up.to_vec(), 2, 2),
            down_proj: QTensor::from_f32(down.to_vec(), 2, 2),
            act: Act::Silu,
            down_t: None,
            segs: Vec::new(),
        }
    }

    /// Each selected expert must contribute to the routed sum.  Passing the
    /// shared output buffer directly to `expert_swiglu` lets `QTensor::matvec`
    /// overwrite the prior expert, so this catches the regression with two
    /// deliberately different experts.
    #[test]
    fn glm_moe_accumulates_all_selected_experts() {
        let id = [1.0, 0.0, 0.0, 1.0];
        let twice = [2.0, 0.0, 0.0, 2.0];
        let expert0 = dense(&id, &id, &id);
        let expert1 = dense(&twice, &id, &id);
        let model = MoeFfn {
            router: QTensor::from_f32(vec![0.0; 4], 2, 2),
            experts: vec![expert0, expert1],
            top_k: 2,
            norm_topk_prob: true,
            router_sigmoid: true,
            sigmoid_denom_eps: 1e-20,
            expert_bias: None,
            routed_scaling: 1.0,
            route_tau: None,
            shared: None,
            stats: std::cell::RefCell::new(Vec::new()),
            act_sq: std::cell::RefCell::new(Vec::new()),
            act_rows: std::cell::RefCell::new(Vec::new()),
            mask: None,
            per_expert_scale: None,
            router_input_norm: false,
            resonance: None,
            grown: Vec::new(),
        };
        let x = [1.0f32, 2.0];
        let actual = moe_cpu(&model, &x, 0.0, None);

        let mut expected = vec![0.0f32; 2];
        for expert in &model.experts {
            let mut one = vec![0.0f32; 2];
            crate::dsv4::expert_swiglu(
                &x,
                &|x, y| expert.gate_proj.matvec(x, y, None),
                &|x, y| expert.up_proj.matvec(x, y, None),
                &|x, y| expert.down_proj.matvec(x, y, None),
                2,
                0.5,
                0.0,
                &mut one,
            );
            for (dst, src) in expected.iter_mut().zip(one) {
                *dst += src;
            }
        }
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn glm_dynamic_policy_is_dtype_aware() {
        // Automatic mode follows the measured faster exact CPU path.  The
        // bounded dynamic implementation remains available by explicit
        // force, including for Q4 and unsupported/unknown budgets.
        assert!(!glm_dynamic_moe_enabled(None));
        assert!(!glm_dynamic_moe_enabled(Some("auto")));
        assert!(glm_dynamic_moe_enabled(Some("1")));
        assert!(!glm_dynamic_moe_enabled(Some("0")));
        assert!(!glm_dynamic_moe_enabled(Some("unexpected")));
    }

    /// The absorbed DSA attention against the expanded one (kv_b over every
    /// selected latent, then the per-head softmax): the same heads' outputs.
    #[test]
    fn dsa_absorbed_matches_the_expanded_form() {
        use super::{DsaState, DsaWeights, dsa_absorbed};
        use crate::pipeline::MlaWeights;
        let (nh, nope, dv, lora, npos) = (3usize, 8usize, 4usize, 16usize, 7usize);
        let mut seed = 0x2545_f491u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        let kvb: Vec<f32> = (0..nh * (nope + dv) * lora).map(|_| rnd()).collect();
        let q: Vec<f32> = (0..nh * nope).map(|_| rnd()).collect();
        let latents: Vec<f32> = (0..npos * lora).map(|_| rnd()).collect();
        let z = |r: usize, c: usize| QTensor::from_f32(vec![0.0; r * c], r, c);
        let w = DsaWeights {
            mla: MlaWeights {
                q_proj: z(nh * nope, 4),
                q_a: None,
                q_a_norm: None,
                kv_a: z(lora, 4),
                kv_a_norm: vec![1.0; lora],
                kv_b: QTensor::from_f32(kvb.clone(), nh * (nope + dv), lora),
                o_proj: z(4, nh * dv),
                nh,
                qk_rope: 0,
                qk_nope: nope,
                v_dim: dv,
                lora,
                scale: 0.3,
                nope: true,
            },
            index_q: z(1, 4),
            index_k: z(1, 4),
            index_k_norm: vec![1.0],
            index_k_bias: vec![0.0],
            index_weights: z(1, 4),
            pool_gate: z(1, 4),
            pool_ape: vec![0.0],
            kvb_f32: std::sync::OnceLock::new(),
        };
        let st = DsaState { latents: latents.clone(), index_keys: Vec::new(), pool_gates: Vec::new() };
        let selected = [5usize, 0, 3, 6];
        let got = dsa_absorbed(&q, &w, &st, &selected, None);
        let stride = nope + dv;
        for h in 0..nh {
            let kv: Vec<Vec<f32>> = selected
                .iter()
                .map(|&p| (0..stride).map(|r| (0..lora).map(|l| kvb[(h * stride + r) * lora + l] * latents[p * lora + l]).sum()).collect())
                .collect();
            let sc: Vec<f32> = kv.iter().map(|k| (0..nope).map(|d| q[h * nope + d] * k[d]).sum::<f32>() * 0.3).collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let zs: f32 = sc.iter().map(|&s| (s - mx).exp()).sum();
            for d in 0..dv {
                let want: f32 = sc.iter().zip(&kv).map(|(&s, k)| (s - mx).exp() / zs * k[nope + d]).sum();
                assert!((got[h * dv + d] - want).abs() < 1e-5, "head {h} lane {d}: {} vs {want}", got[h * dv + d]);
            }
        }
    }

    #[test]
    fn dsa_selector_preserves_topk_order_and_tail() {
        // Three complete pools of four plus a visible two-token tail.  The
        // first two scores tie; lower pool index must win, and the tail must
        // remain after the selected complete pools.
        let got = dsa_select_positions(vec![(0, 0.5), (1, 0.5), (2, 0.9)], 14, 4, 8, true);
        assert_eq!(got, vec![8, 9, 10, 11, 0, 1, 2, 3, 12, 13]);
    }
}
