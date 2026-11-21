//! Inkling-2-Mini (`inkling2_loop`) text runtime.
//!
//! Transcribed from the reference `modeling_inkling2_loop.py`
//! (huggingface/transformers#51050) and checked against llama.cpp's graph
//! (`src/models/inkling2*.cpp`), not inferred from tensor names. The stack:
//!
//! * 24 layers; `[core_start, core_end)` (6..18) run `passes` times (3) with
//!   shared weights: 6 + 12·3 + 6 = 48 layer applications a token, each with
//!   its own keys, values and convolution rings (a "slot": the physical
//!   layer for the first pass, `layers + (pass − 1)·loop + i` after);
//! * block attention residuals instead of a plain residual stream: every
//!   `block` (6) layers the running sum closes into a block state; a layer's
//!   attention input, its MoE input and the final output are softmax
//!   mixtures, by learned queries, of the RMS-normed (no weight) block states
//!   and the running sum. The full-attention layers mix three inputs (q, k, v);
//! * no rotary embedding: a relative bias over the last `extent` (1024)
//!   distances, `r · table[d]`, `r` a 16-wide per-head projection of the
//!   query input (scaled, L2-clipped to 8, log-scaled on the full layers);
//!   queries scaled per channel and L2-clipped to 22.63, keys L2-clipped, values
//!   RMS-normed without weight; one sink logit per head and token; the output
//!   gated by `σ(gate · x)`; local layers see a window of 1024 positions;
//! * causal short convolutions (4 taps, tap 0 the current position) with a
//!   residual: on keys and values (`(x + conv) / √5`; the full layers add
//!   per-token taps projected from the layer input, `/ 3`), after the
//!   attention and after the MoE (`/ √5`);
//! * passes 2 and 3 of a loop layer read four messages (q, k, v, MLP) mixed
//!   from the previous pass's block states and output, and gate them in:
//!   `(1 − tanh f)·x + tanh g·m`, `f, g` from a rank-64 projection of
//!   `[norm x, norm m]`;
//! * MoE: σ router, top-8 of 128 by score + selection bias, weights the
//!   chosen scores renormalized times 8; experts `4·tanh(g/4)·σ(g)·25·tanh(u/25)`;
//! * the head over the final mixture, RMS-normed, softcapped at 30.
//!
//! The converter is `cortiq-cli/src/inklingpack.rs`.

use crate::loader::{Overlay, load_f32, load_matrix};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfError, CmfModel, ModelArch};
use std::sync::Arc;

pub const ARCH_NAME: &str = "inkling2_loop";

fn err(s: impl Into<String>) -> CmfError {
    CmfError::Parse(format!("inkling2_loop: {}", s.into()))
}

fn f(model: &CmfModel, name: &str) -> Result<Vec<f32>, CmfError> {
    load_f32(model, name, &Overlay::None).map_err(err)
}

fn t(model: &Arc<CmfModel>, name: &str) -> Result<QTensor, CmfError> {
    load_matrix(model, name, false, &Overlay::None)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x / rms(x)` (no weight), the reference's `UnweightedRMSNorm`.
fn unit_rms(x: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = (1.0 / (ms + eps).sqrt()) as f32;
    x.iter().map(|&v| v * r).collect()
}

/// `w · x / rms(x)`.
fn rms_w(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = (1.0 / (ms + eps).sqrt()) as f32;
    x.iter().zip(w).map(|(&v, &g)| g * (v * r)).collect()
}

/// Scales each `hd`-wide head down to an L2 norm of `threshold` when longer.
fn l2_clip(x: &mut [f32], hd: usize, threshold: f32) {
    for h in x.chunks_mut(hd) {
        let n = h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt() as f32;
        let s = (threshold / (n + 1e-6)).min(1.0);
        for v in h.iter_mut() {
            *v *= s;
        }
    }
}

fn mv(pool: Option<&Pool>, w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    crate::pool::matvec_rows(pool, w, x, &mut out);
    out
}

/// A softmax mixture of the RMS-normed `states` by each of `queries`
/// (`[nq, dim]`), scaled by `1/√dim`: the reference's block attention
/// residual (keys and values are the normed states).
pub fn mix(states: &[&[f32]], queries: &[f32], dim: usize) -> Vec<Vec<f32>> {
    let normed: Vec<Vec<f32>> = states.iter().map(|s| unit_rms(s, 1e-6)).collect();
    let scale = 1.0 / (dim as f64).sqrt();
    queries
        .chunks(dim)
        .map(|q| {
            let logits: Vec<f64> = normed.iter().map(|s| s.iter().zip(q).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale).collect();
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
            let den: f64 = e.iter().sum();
            let mut out = vec![0f32; dim];
            for (s, w) in normed.iter().zip(&e) {
                let w = (w / den) as f32;
                for (o, v) in out.iter_mut().zip(s) {
                    *o += w * v;
                }
            }
            out
        })
        .collect()
}

/// The log scale of the full layers' queries and relative states at `pos`.
pub fn tau(pos: usize, alpha: f64, floor: f64) -> f32 {
    (1.0 + alpha * (((pos + 1) as f64 / floor).max(1.0)).ln()) as f32
}

/// A short convolution: `[channels, kernel]` taps, tap 0 on the current position.
pub struct Conv {
    pub taps: Vec<f32>,
    pub channels: usize,
    pub kernel: usize,
    /// The full layers' per-token taps `[groups · kernel, hidden]` over the
    /// layer input; a group spans `channels / groups` channels.
    pub proj: Option<Vec<f32>>,
    pub groups: usize,
}

impl Conv {
    fn load(model: &CmfModel, stem: &str, kernel: usize, groups: usize) -> Result<Self, CmfError> {
        let taps = f(model, &format!("{stem}.taps"))?;
        if taps.len() % kernel != 0 {
            return Err(err(format!("{stem}.taps: {} values for {kernel} taps", taps.len())));
        }
        let proj = if model.tensor(&format!("{stem}.proj")).is_some() { Some(f(model, &format!("{stem}.proj"))?) } else { None };
        Ok(Self { channels: taps.len() / kernel, taps, kernel, proj, groups })
    }

    /// One position: `ring` holds the previous `kernel − 1` inputs (newest
    /// first); `dynw` the per-token taps `[groups · kernel]` of the full
    /// layers' convolutions.
    fn step_with(&self, x: &[f32], ring: &mut Vec<f32>, dynw: Option<&[f32]>) -> Vec<f32> {
        let (c, k) = (self.channels, self.kernel);
        if ring.len() != (k - 1) * c {
            *ring = vec![0f32; (k - 1) * c];
        }
        let per = c / self.groups.max(1);
        let scale = 1.0 / (((if dynw.is_some() { 2 } else { 1 }) * k + 1) as f32).sqrt();
        let mut out = vec![0f32; c];
        for ch in 0..c {
            let tp = &self.taps[ch * k..(ch + 1) * k];
            let mut acc = 0f32;
            for lag in 0..k {
                let xv = if lag == 0 { x[ch] } else { ring[(lag - 1) * c + ch] };
                let mut w = tp[lag];
                if let Some(d) = dynw {
                    w += d[(ch / per) * k + lag];
                }
                acc += w * xv;
            }
            out[ch] = (x[ch] + acc) * scale;
        }
        // shift the ring: newest first
        if k > 1 {
            ring.copy_within(0..(k - 2) * c, c);
            ring[..c].copy_from_slice(x);
        }
        out
    }
}

pub struct Attn {
    pub q: QTensor,
    pub k: QTensor,
    pub v: QTensor,
    pub o: QTensor,
    pub gate: QTensor,
    pub rel: QTensor,
    pub sink: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    /// `[d_rel, extent]`.
    pub rel_table: Vec<f32>,
    pub k_conv: Conv,
    pub v_conv: Conv,
}

pub struct Gate {
    /// `[rank, 2·hidden]`, `[hidden, rank]`, `[hidden, rank]`.
    pub down: Vec<f32>,
    pub forget: Vec<f32>,
    pub gain: Vec<f32>,
}

impl Gate {
    fn load(model: &CmfModel, stem: &str) -> Result<Self, CmfError> {
        Ok(Self { down: f(model, &format!("{stem}.down"))?, forget: f(model, &format!("{stem}.forget"))?, gain: f(model, &format!("{stem}.gain"))? })
    }

}

pub struct Moe {
    /// `[E, hidden]` f32 router and the selection bias.
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    pub experts: Vec<[QTensor; 3]>,
}

pub struct Layer {
    pub sliding: bool,
    /// `[1 | 3, hidden]`; None on layer 0.
    pub attn_query: Option<Vec<f32>>,
    pub mlp_query: Vec<f32>,
    /// `input_layernorm` (sliding) or the q, k, v input norms (full).
    pub in_norm: [Vec<f32>; 3],
    pub post_norm: Vec<f32>,
    pub attn: Attn,
    pub post_attn_conv: Conv,
    pub post_mlp_conv: Conv,
    pub moe: Moe,
    /// The q, k, v and MLP message gates of a loop layer.
    pub gates: Option<[Gate; 4]>,
}

/// The embedding table, a row a token (f16 / bf16 rows from the mapping).
pub enum Embed {
    Half { idx: usize, rows: usize, bf16: bool },
    Matrix(QTensor),
}

impl Embed {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, CmfError> {
        let idx = model.tensor_index(name).ok_or_else(|| err(format!("{name} missing")))?;
        let e = &model.tensors[idx];
        match e.dtype {
            cortiq_core::TensorDtype::F16 | cortiq_core::TensorDtype::Bf16 if e.shape.len() == 2 => Ok(Embed::Half { idx, rows: e.shape[0], bf16: e.dtype == cortiq_core::TensorDtype::Bf16 }),
            _ => Ok(Embed::Matrix(t(model, name)?)),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Embed::Half { rows, .. } => *rows,
            Embed::Matrix(m) => m.rows(),
        }
    }

    pub fn row_f32(&self, model: &CmfModel, r: usize, out: &mut [f32]) {
        match self {
            Embed::Half { idx, bf16, .. } => {
                let bytes = model.entry_bytes(&model.tensors[*idx]);
                let dim = out.len();
                for (o, b) in out.iter_mut().zip(bytes[r * dim * 2..(r + 1) * dim * 2].chunks_exact(2)) {
                    let h = u16::from_le_bytes([b[0], b[1]]);
                    *o = if *bf16 { cortiq_core::quant::bf16_to_f32(h) } else { cortiq_core::quant::f16_to_f32(h) };
                }
            }
            Embed::Matrix(m) => m.row_f32(r, out),
        }
    }
}

pub struct Globals {
    pub model: Arc<CmfModel>,
    pub embed: Embed,
    pub embed_norm: Vec<f32>,
    pub lm_head: QTensor,
    pub norm: Vec<f32>,
    pub final_query: Vec<f32>,
    /// `[loop, 4, hidden]`: q, k, v, MLP message queries of each loop layer.
    pub message_query: Vec<f32>,
    /// The DSpark drafter, when the file carries it (`mtp.*`).
    pub draft: Option<Draft>,
}

/// One dense layer of the DSpark drafter.
pub struct DraftLayer {
    pub q: QTensor,
    pub k: QTensor,
    pub v: QTensor,
    pub o: QTensor,
    pub k_ctx: QTensor,
    pub v_ctx: QTensor,
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    pub dflare_norm: Vec<f32>,
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_ctx_norm: Vec<f32>,
    /// A sliding-window layer (`mtp_config.local_layer_ids`).
    pub local: bool,
}

/// The DSpark drafter (the release's `mtp.safetensors`): dense layers over a
/// block of the anchor and mask rows, their context keys and values made from
/// the target's taps (`targets`: application indices, the last the final
/// normed state), and a recurrent Markov head over the target's head.
pub struct Draft {
    pub layers: Vec<DraftLayer>,
    /// softmax(fuse_weight) rows: `[mixes][targets]`, a mix per draft layer and
    /// the Markov head's last.
    pub alpha: Vec<f32>,
    pub mask: Vec<f32>,
    pub norm: Vec<f32>,
    pub markov_norm: Vec<f32>,
    /// The recurrence: `[3 rank][2 rank + 2 hidden]` (state, previous token,
    /// hidden, context columns) and its bias.
    pub rnn_w: Vec<f32>,
    pub rnn_b: Vec<f32>,
    /// The previous token's row `[vocab][rank]` (f16).
    pub w1: QTensor,
    /// head · hidden_out `[vocab][rank]` and head · hidden_out's bias.
    pub fold: QTensor,
    pub fold_b: Vec<f32>,
    pub targets: Vec<usize>,
    pub window: usize,
    pub block: usize,
    pub rank: usize,
    pub rope_theta: f64,
    pub eps: f32,
}

impl Draft {
    fn load(model: &Arc<CmfModel>, mc: &serde_json::Value, dim: usize) -> Result<Self, CmfError> {
        let u = |k: &str| -> Result<usize, CmfError> { mc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| err(format!("mtp_config.{k}"))) };
        let n = u("num_hidden_layers")?;
        let local: Vec<usize> = mc.get("local_layer_ids").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
        let targets: Vec<usize> = mc.get("target_layer_ids").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            let p = format!("mtp.layers.{l}.");
            layers.push(DraftLayer {
                q: t(model, &format!("{p}self_attn.q_proj.weight"))?,
                k: t(model, &format!("{p}self_attn.k_proj.weight"))?,
                v: t(model, &format!("{p}self_attn.v_proj.weight"))?,
                o: t(model, &format!("{p}self_attn.o_proj.weight"))?,
                k_ctx: t(model, &format!("{p}self_attn.k_proj_ctx.weight"))?,
                v_ctx: t(model, &format!("{p}self_attn.v_proj_ctx.weight"))?,
                gate: t(model, &format!("{p}mlp.gate_proj.weight"))?,
                up: t(model, &format!("{p}mlp.up_proj.weight"))?,
                down: t(model, &format!("{p}mlp.down_proj.weight"))?,
                dflare_norm: f(model, &format!("{p}dflare_norm.weight"))?,
                in_norm: f(model, &format!("{p}input_layernorm.weight"))?,
                post_norm: f(model, &format!("{p}post_attention_layernorm.weight"))?,
                q_norm: f(model, &format!("{p}self_attn.q_norm.weight"))?,
                k_norm: f(model, &format!("{p}self_attn.k_norm.weight"))?,
                k_ctx_norm: f(model, &format!("{p}self_attn.k_norm_ctx.weight"))?,
                local: local.contains(&l),
            });
        }
        // the mixes' weights, each row softmaxed over the targets
        let fuse = f(model, "mtp.fuse_weight")?;
        let nt = targets.len();
        if nt == 0 || fuse.len() != (n + 1) * nt {
            return Err(err(format!("mtp fuse_weight: {} values for {} mixes of {nt} targets", fuse.len(), n + 1)));
        }
        let mut alpha = fuse.clone();
        for row in alpha.chunks_mut(nt) {
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let s: f32 = row.iter().map(|&x| (x - m).exp()).sum();
            for x in row.iter_mut() {
                *x = (*x - m).exp() / s;
            }
        }
        let rank = u("markov_rank")?;
        let d = Self {
            layers,
            alpha,
            mask: f(model, "mtp.mask_embed")?,
            norm: f(model, "mtp.norm.weight")?,
            markov_norm: f(model, "mtp.markov_head.dflare_norm.weight")?,
            rnn_w: f(model, "mtp.markov_head.rnn_proj.weight")?,
            rnn_b: f(model, "mtp.markov_head.rnn_proj.bias")?,
            w1: t(model, "mtp.markov_head.markov_w1.weight")?,
            fold: t(model, "mtp.markov_head.fold")?,
            fold_b: f(model, "mtp.markov_head.fold_b")?,
            targets,
            window: u("sliding_window_size")?,
            block: u("block_size")?,
            rank,
            rope_theta: mc.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1e6),
            eps: mc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        };
        if d.rnn_w.len() != 3 * rank * (2 * rank + 2 * dim) || d.mask.len() != dim {
            return Err(err("mtp: the Markov head's or the mask's shape"));
        }
        Ok(d)
    }
}

pub struct Cfg {
    pub dim: usize,
    pub layers: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub d_rel: usize,
    pub eps: f64,
    pub window: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_alpha: f64,
    pub log_floor: f64,
    pub block: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub logit_cap: f32,
    pub max_seq: usize,
}

impl Cfg {
    pub fn loop_len(&self) -> usize {
        self.core_end - self.core_start
    }

    /// Layer applications a token (= cache slots).
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * self.loop_len()
    }

    /// The slot of physical layer `li` in pass `pass`.
    pub fn slot(&self, li: usize, pass: usize) -> usize {
        if pass == 0 { li } else { self.layers + (pass - 1) * self.loop_len() + (li - self.core_start) }
    }

    /// `(physical layer, pass)` of every application, in order.
    pub fn plan(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = (0..self.core_start).map(|l| (l, 0)).collect();
        for p in 0..self.passes {
            v.extend((self.core_start..self.core_end).map(|l| (l, p)));
        }
        v.extend((self.core_end..self.layers).map(|l| (l, 0)));
        v
    }
}

/// One slot's cache: keys and values (`nkv·hd` a position; a sliding slot
/// keeps its last `window` positions, `base` the position of row 0) and the
/// four convolution rings (k, v, after the attention, after the MoE).
#[derive(Default)]
struct SlotState {
    k: Vec<f32>,
    v: Vec<f32>,
    base: usize,
    rings: [Vec<f32>; 4],
}

pub struct State {
    pub pos: usize,
    pub tokens: Vec<u32>,
    slots: Vec<SlotState>,
    /// The sequence's device key (process-unique: a pipeline reuses one model
    /// for many prompts).
    sequence_id: u64,
    /// The sequence runs on the card: None not tried yet, Some(false) the
    /// host runs it, Some(true) the card holds its states.
    card: Option<bool>,
    /// The DSpark depth (kept across sequences: a device's prices and a
    /// drafter's holds outlive a request).
    pub spec: SpecPolicy,
}

static NEXT_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl State {
    pub fn reset(&mut self, n_slots: usize) {
        if self.card == Some(true) {
            crate::gpu::inkling_reset(self.sequence_id);
        }
        self.card = None;
        self.sequence_id = (3u64 << 60) | NEXT_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pos = 0;
        self.tokens.clear();
        self.slots = (0..n_slots).map(|_| SlotState::default()).collect();
        self.spec.forget_step();
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if self.card == Some(true) {
            crate::gpu::inkling_reset(self.sequence_id);
        }
    }
}

pub fn load(model: &Arc<CmfModel>, arch: &ModelArch) -> Result<(Globals, Vec<Layer>, Cfg, State), CmfError> {
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("inkling2_loop")).ok_or_else(|| err("header has no inkling2_loop record"))?;
    let c = prov.get("config").ok_or_else(|| err("no config"))?;
    let u = |k: &str| -> Result<usize, CmfError> { c.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| err(format!("config.{k}"))) };
    let fl = |k: &str| -> Result<f64, CmfError> { c.get(k).and_then(|v| v.as_f64()).ok_or_else(|| err(format!("config.{k}"))) };
    let lc = c.get("loop_config").ok_or_else(|| err("config.loop_config"))?;
    let lu = |k: &str| -> Result<usize, CmfError> { lc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| err(format!("loop_config.{k}"))) };
    let moe = arch.moe.as_ref().ok_or_else(|| err("missing MoE geometry"))?;
    let local: Vec<usize> = c.get("local_layer_ids").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
    let mut cfg = Cfg {
        dim: arch.hidden_size,
        layers: arch.num_layers,
        nh: arch.num_attention_heads,
        nkv: arch.num_kv_heads,
        hd: arch.head_dim,
        d_rel: u("d_rel")?,
        eps: arch.rms_norm_eps,
        window: u("sliding_window_size")?,
        rel_extent: u("rel_extent")?,
        qk_clip: fl("qk_l2_clip_threshold")? as f32,
        rel_clip: fl("rel_l2_clip_threshold")? as f32,
        log_alpha: fl("log_scaling_alpha")?,
        log_floor: u("log_scaling_n_floor")? as f64,
        block: u("n_layers_per_attn_res_block")?,
        core_start: lu("core_start")?,
        core_end: lu("core_end")?,
        passes: lu("num_passes")?,
        top_k: moe.top_k,
        route_scale: moe.routed_scaling_factor.unwrap_or(1.0),
        gate_cap: fl("mlp_gate_softcap")? as f32,
        up_cap: fl("mlp_up_softcap")? as f32,
        logit_cap: fl("final_logit_softcapping")? as f32,
        max_seq: std::env::var("CMF_MAX_SEQ").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32_768).min(model.header.arch.max_position_embeddings).max(1),
    };
    if cfg.nkv == 0 || cfg.nh % cfg.nkv != 0 || cfg.block == 0 || cfg.core_start == 0 || cfg.core_end <= cfg.core_start || cfg.core_end > cfg.layers || cfg.passes == 0 {
        return Err(err("geometry"));
    }
    // `--loops N` (CMF_INKLING_PASSES): the loop core run N times. A varloop file
    // (TML-d24-varloop-research) was trained at 1 to 5 passes; Inkling-2-Mini at its
    // file's count only
    if let Some(n) = passes_override() {
        let varloop = lc.get("is_varloop").and_then(|v| v.as_bool()) == Some(true);
        if n != cfg.passes {
            if varloop {
                tracing::info!("inkling2_loop: the loop runs {n} passes (the file's default {})", cfg.passes);
            } else {
                tracing::warn!("inkling2_loop: the loop runs {n} passes; this file was trained at {} (a varloop file is trained for other counts)", cfg.passes);
            }
        }
        cfg.passes = n;
    }
    let globals = Globals {
        model: model.clone(),
        embed: Embed::load(model, "model.embed_tokens.weight")?,
        embed_norm: f(model, "model.embed_tokens.norm.weight")?,
        lm_head: t(model, "lm_head.weight")?,
        norm: f(model, "model.norm.weight")?,
        final_query: f(model, "model.final_input.query")?,
        message_query: f(model, "model.message_input.query")?,
        // the drafter: kept off with CMF_INKLING_DRAFT=0
        draft: match prov.get("mtp_config") {
            Some(mc) if !mc.is_null() && std::env::var("CMF_INKLING_DRAFT").as_deref() != Ok("0") => match Draft::load(model, mc, cfg.dim) {
                Ok(d) => Some(d),
                Err(e) => {
                    tracing::warn!("inkling2_loop: the DSpark drafter does not load ({e}); plain decoding");
                    None
                }
            },
            _ => None,
        },
    };
    if globals.message_query.len() != cfg.loop_len() * 4 * cfg.dim || globals.final_query.len() != cfg.dim {
        return Err(err("residual queries"));
    }
    let kernel = arch.linear_conv_kernel_dim.unwrap_or(4).max(1);
    let mut layers = Vec::with_capacity(cfg.layers);
    for li in 0..cfg.layers {
        let p = format!("model.layers.{li}.");
        let a = format!("{p}self_attn.");
        let sliding = local.contains(&li);
        let attn_query = if li > 0 { Some(f(model, &format!("{p}attention_input.query"))?) } else { None };
        if attn_query.as_ref().is_some_and(|q| q.len() != (if sliding { 1 } else { 3 }) * cfg.dim) {
            return Err(err(format!("layer {li}: attention_input.query")));
        }
        let in_norm = if sliding {
            let n = f(model, &format!("{p}input_layernorm.weight"))?;
            [n.clone(), n.clone(), n]
        } else {
            [f(model, &format!("{p}q_input_layernorm.weight"))?, f(model, &format!("{p}k_input_layernorm.weight"))?, f(model, &format!("{p}v_input_layernorm.weight"))?]
        };
        let attn = Attn {
            q: t(model, &format!("{a}q_proj.weight"))?,
            k: t(model, &format!("{a}k_proj.weight"))?,
            v: t(model, &format!("{a}v_proj.weight"))?,
            o: t(model, &format!("{a}o_proj.weight"))?,
            gate: t(model, &format!("{a}gate_proj.weight"))?,
            rel: t(model, &format!("{a}rel_proj.weight"))?,
            sink: f(model, &format!("{a}sink_proj.weight"))?,
            q_scale: f(model, &format!("{a}q_scale"))?,
            rel_scale: f(model, &format!("{a}rel_scale"))?,
            rel_table: f(model, &format!("{a}rel_table"))?,
            k_conv: Conv::load(model, &format!("{a}k_conv"), kernel, cfg.nkv)?,
            v_conv: Conv::load(model, &format!("{a}v_conv"), kernel, cfg.nkv)?,
        };
        let extent = if sliding { cfg.window } else { cfg.rel_extent };
        if attn.q.rows() != cfg.nh * cfg.hd || attn.k.rows() != cfg.nkv * cfg.hd || attn.v.rows() != cfg.nkv * cfg.hd || attn.rel.rows() != cfg.nh * cfg.d_rel || attn.rel_table.len() != cfg.d_rel * extent || attn.sink.len() != cfg.nh * cfg.dim {
            return Err(err(format!("layer {li}: attention shapes")));
        }
        let m = format!("{p}mlp.");
        let mut experts = Vec::with_capacity(moe.num_experts);
        for e in 0..moe.num_experts {
            let x = format!("{m}experts.{e}.");
            experts.push([t(model, &format!("{x}gate_proj.weight"))?, t(model, &format!("{x}up_proj.weight"))?, t(model, &format!("{x}down_proj.weight"))?]);
        }
        let mo = Moe { router: f(model, &format!("{m}gate.weight"))?, bias: f(model, &format!("{m}gate.e_score_correction_bias"))?, experts };
        if mo.bias.len() != moe.num_experts || mo.router.len() != moe.num_experts * cfg.dim {
            return Err(err(format!("layer {li}: router shapes")));
        }
        let gates = if (cfg.core_start..cfg.core_end).contains(&li) && cfg.passes > 1 {
            Some([
                Gate::load(model, &format!("{p}q_message_gate"))?,
                Gate::load(model, &format!("{p}k_message_gate"))?,
                Gate::load(model, &format!("{p}v_message_gate"))?,
                Gate::load(model, &format!("{p}mlp_message_gate"))?,
            ])
        } else {
            None
        };
        layers.push(Layer {
            sliding,
            attn_query,
            mlp_query: f(model, &format!("{p}mlp_input.query"))?,
            in_norm,
            post_norm: f(model, &format!("{p}post_attention_layernorm.weight"))?,
            attn,
            post_attn_conv: Conv::load(model, &format!("{p}post_attention_conv"), kernel, 1)?,
            post_mlp_conv: Conv::load(model, &format!("{p}post_mlp_conv"), kernel, 1)?,
            moe: mo,
            gates,
        });
    }
    let mut state = State { pos: 0, tokens: Vec::new(), slots: Vec::new(), sequence_id: 0, card: None, spec: SpecPolicy::default() };
    state.reset(cfg.slots());
    Ok((globals, layers, cfg, state))
}

/// `rows` rows of `width` floats each, for the pool.
fn par_rows(pool: Option<&Pool>, rows: usize, width: usize, out: &mut [f32], f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    debug_assert_eq!(out.len(), rows * width);
    match pool {
        Some(p) if rows > 1 => {
            let addr = crate::pool::SendMut::new(out.as_mut_ptr());
            p.run_rows(rows, &|a, b| {
                for r in a..b {
                    f(r, unsafe { std::slice::from_raw_parts_mut(addr.at(r * width), width) });
                }
            });
        }
        _ => {
            for (r, o) in out.chunks_mut(width).enumerate() {
                f(r, o);
            }
        }
    }
}

/// `t · x` for `n` rows of `x` (`[n, cols]` → `[n, rows]`).
fn qmm(t: &QTensor, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
    let mut out = vec![0f32; n * t.rows()];
    if n == 1 {
        t.matvec(x, &mut out, pool);
    } else {
        t.matmat(x, n, &mut out, pool);
    }
    out
}

/// An f32 matrix `[rows, cols]` times `n` rows.
fn fmm(pool: Option<&Pool>, w: &[f32], x: &[f32], n: usize, rows: usize) -> Vec<f32> {
    let cols = x.len() / n.max(1);
    let mut out = vec![0f32; n * rows];
    if n == 1 {
        crate::pool::matvec_rows(pool, w, x, &mut out);
        return out;
    }
    // a row of the matrix against every input row: the matrix is read once
    let mut t = vec![0f32; rows * n];
    par_rows(pool, rows, n, &mut t, &|o, dst| {
        let wr = &w[o * cols..(o + 1) * cols];
        for (r, d) in dst.iter_mut().enumerate() {
            *d = wr.iter().zip(&x[r * cols..(r + 1) * cols]).map(|(a, b)| a * b).sum();
        }
    });
    for o in 0..rows {
        for r in 0..n {
            out[r * rows + o] = t[o * n + r];
        }
    }
    out
}

impl Conv {
    /// `n` consecutive positions (`x` `[n, channels]`); `dyn_in` `[n, hidden]`.
    fn rows(&self, x: &[f32], n: usize, ring: &mut Vec<f32>, dyn_in: Option<&[f32]>, pool: Option<&Pool>) -> Vec<f32> {
        let c = self.channels;
        let dynw = match (&self.proj, dyn_in) {
            (Some(p), Some(xi)) => Some(fmm(pool, p, xi, n, self.groups * self.kernel)),
            _ => None,
        };
        let gk = self.groups * self.kernel;
        let mut out = vec![0f32; n * c];
        for r in 0..n {
            let d = dynw.as_ref().map(|d| &d[r * gk..(r + 1) * gk]);
            let y = self.step_with(&x[r * c..(r + 1) * c], ring, d);
            out[r * c..(r + 1) * c].copy_from_slice(&y);
        }
        out
    }
}

/// Unit-RMS each `hd`-wide head of every row.
fn norm_heads(x: &mut [f32], hd: usize, eps: f64) {
    for h in x.chunks_mut(hd) {
        let n = unit_rms(h, eps);
        h.copy_from_slice(&n);
    }
}

impl Attn {
    /// `n` positions from `p0`: q from `xq`, k from `xk`, v from `xv` (normed
    /// inputs, `[n, hidden]`); their keys and values into the slot's cache;
    /// each row's attention over its past with the relative bias and the
    /// sink; the gated O projection.
    #[allow(clippy::too_many_arguments)]
    fn rows(&self, xq: &[f32], xk: &[f32], xv: &[f32], n: usize, ss: &mut SlotState, p0: usize, sliding: bool, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let (nh, nkv, hd, dr, dim) = (cfg.nh, cfg.nkv, cfg.hd, cfg.d_rel, cfg.dim);
        let per = nh / nkv;
        let mut q = qmm(&self.q, xq, n, pool);
        let mut r = qmm(&self.rel, xq, n, pool);
        for (row, (qr, rr)) in q.chunks_mut(nh * hd).zip(r.chunks_mut(nh * dr)).enumerate() {
            for (v, s) in qr.iter_mut().zip(&self.q_scale) {
                *v *= s;
            }
            l2_clip(qr, hd, cfg.qk_clip);
            for (v, s) in rr.iter_mut().zip(&self.rel_scale) {
                *v *= s;
            }
            l2_clip(rr, dr, cfg.rel_clip);
            if !sliding {
                let tu = tau(p0 + row, cfg.log_alpha, cfg.log_floor);
                for v in qr.iter_mut().chain(rr.iter_mut()) {
                    *v *= tu;
                }
            }
        }
        let sinks = fmm(pool, &self.sink, xq, n, nh);
        let gate = qmm(&self.gate, xq, n, pool);
        let kl = qmm(&self.k, xk, n, pool);
        let mut k = self.k_conv.rows(&kl, n, &mut ss.rings[0], Some(xk), pool);
        l2_clip(&mut k, hd, cfg.qk_clip);
        let vl = qmm(&self.v, xv, n, pool);
        let mut v = self.v_conv.rows(&vl, n, &mut ss.rings[1], Some(xv), pool);
        norm_heads(&mut v, hd, cfg.eps);
        let row = nkv * hd;
        ss.k.extend_from_slice(&k);
        ss.v.extend_from_slice(&v);
        let base = ss.base;
        debug_assert_eq!(base + ss.k.len() / row, p0 + n);

        let extent = self.rel_table.len() / dr;
        let scale = 1.0 / (hd as f32).sqrt();
        let (kc, vc) = (&ss.k, &ss.v);
        let mut out = vec![0f32; n * nh * hd];
        let (q, r) = (&q, &r);
        par_rows(pool, n * nh, hd, &mut out, &|i, o| {
            let (rw, h) = (i / nh, i % nh);
            let pos = p0 + rw;
            let g = h / per;
            let qh = &q[(rw * nh + h) * hd..(rw * nh + h + 1) * hd];
            let rh = &r[(rw * nh + h) * dr..(rw * nh + h + 1) * dr];
            let first = if sliding { base.max((pos + 1).saturating_sub(cfg.window)) } else { base };
            // the relative bias of this row and head over the distances it sees
            let near = extent.min(pos + 1 - first);
            let bias: Vec<f32> = (0..near).map(|d| (0..dr).map(|j| rh[j] * self.rel_table[j * extent + d]).sum::<f32>()).collect();
            let sk = sinks[rw * nh + h];
            let cnt = pos + 1 - first;
            let mut s = vec![0f32; cnt];
            let mut mx = sk;
            for (ii, sv) in s.iter_mut().enumerate() {
                let tpos = first + ii;
                let ci = tpos - base;
                let kr = &kc[(ci * nkv + g) * hd..(ci * nkv + g + 1) * hd];
                let mut x = kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale;
                let d = pos - tpos;
                if d < near {
                    x += bias[d];
                }
                *sv = x;
                mx = mx.max(x);
            }
            let mut den = ((sk - mx) as f64).exp();
            for sv in s.iter_mut() {
                *sv = (*sv - mx).exp();
                den += *sv as f64;
            }
            let inv = (1.0 / den) as f32;
            for (ii, &w) in s.iter().enumerate() {
                let ci = first + ii - base;
                let vr = &vc[(ci * nkv + g) * hd..(ci * nkv + g + 1) * hd];
                let w = w * inv;
                for (a, b) in o.iter_mut().zip(vr) {
                    *a += w * b;
                }
            }
        });
        // a sliding slot keeps its last `window` rows
        if sliding && ss.k.len() > cfg.window * row {
            let drop = ss.k.len() / row - cfg.window;
            ss.k.drain(..drop * row);
            ss.v.drain(..drop * row);
            ss.base += drop;
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        let y = qmm(&self.o, &out, n, pool);
        debug_assert_eq!(y.len(), n * dim);
        y
    }
}

impl Moe {
    /// The selected experts and their mixing weights: the choice by
    /// `σ(logit) + bias`, the weights the chosen scores over their sum.
    pub fn route(&self, x: &[f32], top_k: usize, pool: Option<&Pool>) -> Vec<(usize, f32)> {
        let ne = self.bias.len();
        let logits = mv(pool, &self.router, x, ne);
        route_logits(&logits, &self.bias, top_k)
    }

    /// `n` rows: each row's experts, the rows of an expert in one product.
    fn rows(&self, x: &[f32], n: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let dim = cfg.dim;
        let ne = self.bias.len();
        let logits = fmm(pool, &self.router, x, n, ne);
        let mut by: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for r in 0..n {
            for (e, w) in route_logits(&logits[r * ne..(r + 1) * ne], &self.bias, cfg.top_k) {
                by[e].push((r, w * cfg.route_scale));
            }
        }
        let mut y = vec![0f32; n * dim];
        for (e, rows) in by.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let mut xs = vec![0f32; m * dim];
            for (j, &(r, _)) in rows.iter().enumerate() {
                xs[j * dim..(j + 1) * dim].copy_from_slice(&x[r * dim..(r + 1) * dim]);
            }
            let [gt, up, dn] = &self.experts[e];
            let g = qmm(gt, &xs, m, pool);
            let u = qmm(up, &xs, m, pool);
            let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| softcap(g, cfg.gate_cap) * sigmoid(g) * softcap(u, cfg.up_cap)).collect();
            let ye = qmm(dn, &a, m, pool);
            for (j, &(r, w)) in rows.iter().enumerate() {
                for (o, v) in y[r * dim..(r + 1) * dim].iter_mut().zip(&ye[j * dim..(j + 1) * dim]) {
                    *o += w * v;
                }
            }
        }
        y
    }
}

#[inline]
pub fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// Top-k by `σ(logit) + bias` (ties to the lower expert), weights
/// `σ / (Σσ + 1e-20)` over the chosen ones.
pub fn route_logits(logits: &[f32], bias: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| (scores[b] + bias[b]).partial_cmp(&(scores[a] + bias[a])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    order.truncate(top_k);
    let sum: f32 = order.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
    order.into_iter().map(|e| (e, scores[e] / sum)).collect()
}

/// `CMF_INKLING_DUMP=<dir>,<pos>,<pos>…`: the host stack writes every
/// application's output at those positions (`<dir>/<kind>_s<slot>_p<pos>.f32`).
fn dump_rows(slot: usize, p0: usize, kind: &str, v: &[f32], dim: usize) {
    static CFG: std::sync::OnceLock<Option<(String, Vec<usize>)>> = std::sync::OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let s = std::env::var("CMF_INKLING_DUMP").ok()?;
        let mut it = s.split(',');
        let dir = it.next()?.to_string();
        Some((dir, it.filter_map(|p| p.parse().ok()).collect()))
    });
    if let Some((dir, ps)) = cfg {
        for (r, row) in v.chunks(dim).enumerate() {
            if ps.contains(&(p0 + r)) {
                let b: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/{kind}_s{slot}_p{}.f32", p0 + r), b);
            }
        }
    }
}

/// `x` and the closed blocks per row; the mixture of every row by `queries`.
fn mix_rows(blocks: &[Vec<f32>], x: &[f32], queries: &[f32], n: usize, dim: usize) -> Vec<Vec<f32>> {
    let nq = queries.len() / dim;
    let mut out = vec![vec![0f32; n * dim]; nq];
    for r in 0..n {
        let mut st: Vec<&[f32]> = blocks.iter().map(|b| &b[r * dim..(r + 1) * dim]).collect();
        st.push(&x[r * dim..(r + 1) * dim]);
        for (qi, m) in mix(&st, queries, dim).into_iter().enumerate() {
            out[qi][r * dim..(r + 1) * dim].copy_from_slice(&m);
        }
    }
    out
}

impl Gate {
    fn rows(&self, x: &[f32], m: &[f32], n: usize, eps: f64, pool: Option<&Pool>) -> Vec<f32> {
        let dim = x.len() / n;
        let rank = self.down.len() / (2 * dim);
        let mut cat = vec![0f32; n * 2 * dim];
        for r in 0..n {
            cat[r * 2 * dim..r * 2 * dim + dim].copy_from_slice(&unit_rms(&x[r * dim..(r + 1) * dim], eps));
            cat[r * 2 * dim + dim..(r + 1) * 2 * dim].copy_from_slice(&unit_rms(&m[r * dim..(r + 1) * dim], eps));
        }
        let d = fmm(pool, &self.down, &cat, n, rank);
        let fg = fmm(pool, &self.forget, &d, n, dim);
        let gn = fmm(pool, &self.gain, &d, n, dim);
        (0..n * dim).map(|i| (1.0 - fg[i].tanh()) * x[i] + gn[i].tanh() * m[i]).collect()
    }
}

fn rms_rows(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    x.chunks(w.len()).flat_map(|r| rms_w(r, w, eps)).collect()
}

/// One layer application for `n` positions from `p0`: `h` the running sums
/// `[n, hidden]`, `blocks` the closed block states (a boundary layer closes
/// `h` into them), `msgs` the four messages of a loop pass after the first.
#[allow(clippy::too_many_arguments)]
fn layer_rows(l: &Layer, li: usize, ss: &mut SlotState, h: &[f32], n: usize, blocks: &mut Vec<Vec<f32>>, msgs: Option<&[Vec<f32>]>, p0: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
    let dim = cfg.dim;
    let boundary = li % cfg.block == 0;
    let mut inputs: [Vec<f32>; 3] = match &l.attn_query {
        Some(q) => {
            let m = mix_rows(blocks, h, q, n, dim);
            if l.sliding { [m[0].clone(), m[0].clone(), m[0].clone()] } else { [m[0].clone(), m[1].clone(), m[2].clone()] }
        }
        None => [h.to_vec(), h.to_vec(), h.to_vec()],
    };
    if let (Some(ms), Some(g)) = (msgs, &l.gates) {
        for j in 0..3 {
            inputs[j] = g[j].rows(&inputs[j], &ms[j], n, cfg.eps, pool);
        }
    }
    let xq = rms_rows(&inputs[0], &l.in_norm[0], cfg.eps);
    let (xk, xv) = if l.sliding && msgs.is_none() {
        (xq.clone(), xq.clone())
    } else {
        (rms_rows(&inputs[1], &l.in_norm[1], cfg.eps), rms_rows(&inputs[2], &l.in_norm[2], cfg.eps))
    };
    let a = l.attn.rows(&xq, &xk, &xv, n, ss, p0, l.sliding, cfg, pool);
    let a = l.post_attn_conv.rows(&a, n, &mut ss.rings[2], None, pool);
    let r: Vec<f32> = if boundary {
        blocks.push(h.to_vec());
        a
    } else {
        h.iter().zip(&a).map(|(x, y)| x + y).collect()
    };
    let mut m = mix_rows(blocks, &r, &l.mlp_query, n, dim).swap_remove(0);
    if let (Some(ms), Some(g)) = (msgs, &l.gates) {
        m = g[3].rows(&m, &ms[3], n, cfg.eps, pool);
    }
    let x = rms_rows(&m, &l.post_norm, cfg.eps);
    let y = l.moe.rows(&x, n, cfg, pool);
    let y = l.post_mlp_conv.rows(&y, n, &mut ss.rings[3], None, pool);
    r.iter().zip(&y).map(|(a, b)| a + b).collect()
}

/// `n` positions from `p0` through the whole stack (host): the normed final
/// mixture of every row, ready for the head.
fn host_rows(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], p0: usize, pool: Option<&Pool>) -> Vec<f32> {
    let (dim, n) = (cfg.dim, ids.len());
    let mut h = vec![0f32; n * dim];
    for (r, &id) in ids.iter().enumerate() {
        let mut e = vec![0f32; dim];
        if (id as usize) < globals.embed.rows() {
            globals.embed.row_f32(&globals.model, id as usize, &mut e);
        }
        h[r * dim..(r + 1) * dim].copy_from_slice(&rms_w(&e, &globals.embed_norm, cfg.eps));
    }
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for li in 0..cfg.core_start {
        h = layer_rows(&layers[li], li, &mut state.slots[li], &h, n, &mut blocks, None, p0, cfg, pool);
        dump_rows(li, p0, "h", &h, dim);
    }
    let (pre_h, pre_blocks) = (h.clone(), blocks.clone());
    for pass in 0..cfg.passes {
        let mut bank: Vec<Vec<f32>> = Vec::new();
        if pass > 0 {
            bank = blocks[pre_blocks.len() + 1..].to_vec();
            bank.push(std::mem::replace(&mut h, pre_h.clone()));
            blocks = pre_blocks.clone();
        }
        for li in cfg.core_start..cfg.core_end {
            let slot = cfg.slot(li, pass);
            let msgs = (pass > 0).then(|| {
                let i = li - cfg.core_start;
                let last = bank.len() - 1;
                mix_rows(&bank[..last], &bank[last], &globals.message_query[i * 4 * dim..(i + 1) * 4 * dim], n, dim)
            });
            h = layer_rows(&layers[li], li, &mut state.slots[slot], &h, n, &mut blocks, msgs.as_deref(), p0, cfg, pool);
            dump_rows(slot, p0, "h", &h, dim);
        }
    }
    for li in cfg.core_end..cfg.layers {
        h = layer_rows(&layers[li], li, &mut state.slots[li], &h, n, &mut blocks, None, p0, cfg, pool);
        dump_rows(li, p0, "h", &h, dim);
    }
    let fin = mix_rows(&blocks, &h, &globals.final_query, n, dim).swap_remove(0);
    let x = rms_rows(&fin, &globals.norm, cfg.eps);
    dump_rows(cfg.slots(), p0, "final", &x, dim);
    x
}

/// The loop's pass count asked for at run time (`--loops N` sets
/// `CMF_INKLING_PASSES`), at least 1.
pub fn passes_override() -> Option<usize> {
    std::env::var("CMF_INKLING_PASSES").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n >= 1)
}

/// Rows of a prompt chunk on the host (`CMF_INKLING_CHUNK`, 128).
fn chunk_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_INKLING_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 1).unwrap_or(128))
}

/// The softcapped logits of `n` normed rows.
fn head_rows(globals: &Globals, cfg: &Cfg, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
    let mut lg = qmm(&globals.lm_head, x, n, pool);
    if cfg.logit_cap > 0.0 {
        for v in lg.iter_mut() {
            *v = softcap(*v, cfg.logit_cap);
        }
    }
    lg
}

/// `ids` from `pos` on the card while it holds the sequence (or takes it at
/// position 0); `false`: the host runs it. A card that held the sequence and
/// fails hands its rows to the host (`host_replay`).
#[allow(clippy::too_many_arguments)]
fn on_card(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], pos: usize, want: bool, all: bool, logits: &mut Vec<f32>, pool: Option<&Pool>) -> bool {
    let try_card = match state.card {
        Some(true) => true,
        None => pos == 0,
        Some(false) => false,
    };
    if !try_card || !(crate::gpu::cuda_active() || crate::gpu::metal_active() || crate::gpu::wgpu_active()) {
        if state.card.is_none() && pos == 0 {
            state.card = Some(false);
        }
        return false;
    }
    if crate::gpu::inkling_forward(state.sequence_id, globals, layers, cfg, ids, pos, want, all, logits) {
        if state.card.is_none() {
            tracing::info!("inkling2_loop: the sequence runs on the card");
        }
        state.card = Some(true);
        state.tokens.extend_from_slice(ids);
        state.pos = pos + ids.len();
        return true;
    }
    if state.card == Some(true) {
        tracing::warn!("inkling2_loop: the card dropped the sequence at {pos}; the host replays its {} rows", state.tokens.len());
        host_replay(globals, layers, cfg, state, pool);
    }
    state.card = Some(false);
    false
}

/// The host's caches rebuilt from the sequence's tokens (after the card let go).
fn host_replay(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, pool: Option<&Pool>) {
    let toks = std::mem::take(&mut state.tokens);
    state.slots = (0..cfg.slots()).map(|_| SlotState::default()).collect();
    for (ci, part) in toks.chunks(chunk_rows()).enumerate() {
        let _ = host_rows(globals, layers, cfg, state, part, ci * chunk_rows(), pool);
    }
    state.pos = toks.len();
    state.tokens = toks;
}

/// The prompt from `start` through the host stack in chunks; the last row's
/// logits. Returns the positions taken (all of them, or 0 when the sequence
/// is not at `start`).
#[allow(clippy::too_many_arguments)]
pub fn forward_prompt(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], start: usize, pool: Option<&Pool>, logits: &mut Vec<f32>) -> usize {
    if ids.is_empty() || std::env::var("CMF_INKLING_BATCH").as_deref() == Ok("0") {
        return 0;
    }
    if start == 0 {
        state.reset(cfg.slots());
    } else if state.pos != start {
        return 0;
    }
    let t0 = std::time::Instant::now();
    if on_card(globals, layers, cfg, state, ids, start, true, false, logits, pool) {
        tracing::info!("inkling2_loop: {} prompt rows on the card in {:.2} s", ids.len(), t0.elapsed().as_secs_f64());
        return ids.len();
    }
    let mut last = Vec::new();
    for (ci, part) in ids.chunks(chunk_rows()).enumerate() {
        let p0 = start + ci * chunk_rows();
        last = host_rows(globals, layers, cfg, state, part, p0, pool);
        state.tokens.extend_from_slice(part);
        state.pos = p0 + part.len();
        let n = part.len();
        last = last[(n - 1) * cfg.dim..].to_vec();
    }
    *logits = head_rows(globals, cfg, &last, 1, pool);
    tracing::info!("inkling2_loop: {} prompt rows on the host in {:.1} s", ids.len(), t0.elapsed().as_secs_f64());
    ids.len()
}

/// A window's negative log-likelihood through the host stack in chunks (the
/// predictions from `from` on).
pub fn nll(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], from: usize, pool: Option<&Pool>) -> Option<(f64, usize)> {
    if ids.len() < 2 || std::env::var("CMF_INKLING_BATCH").as_deref() == Ok("0") {
        return None;
    }
    let n = ids.len() - 1;
    let t0 = std::time::Instant::now();
    state.reset(cfg.slots());
    let (mut nll, mut cnt) = (0f64, 0usize);
    let vocab = globals.lm_head.rows();
    // rows a card call scores (`CMF_INKLING_NLL_ROWS`): 16 keeps every call on
    // the arms' decode path, more takes CUDA's prompt path
    let card_rows = std::env::var("CMF_INKLING_NLL_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&r| r >= 1).unwrap_or(16);
    let mut card = None;
    let mut p0 = 0usize;
    while p0 < n {
        let take = if card != Some(false) { card_rows } else { chunk_rows() }.min(n - p0);
        let part = &ids[p0..p0 + take];
        let mut lg = Vec::new();
        if card != Some(false) {
            let ok = on_card(globals, layers, cfg, state, part, p0, true, true, &mut lg, pool);
            card = Some(ok);
        }
        if card == Some(false) {
            let x = host_rows(globals, layers, cfg, state, part, p0, pool);
            state.pos = p0 + part.len();
            lg = head_rows(globals, cfg, &x, part.len(), pool);
        }
        for (j, row) in lg.chunks(vocab).enumerate() {
            let p = p0 + j;
            if p < from {
                continue;
            }
            let target = (ids[p + 1] as usize).min(vocab - 1);
            let best = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
            let den: f64 = row.iter().map(|&v| ((v as f64) - best).exp()).sum();
            nll += den.ln() - (row[target] as f64 - best);
            cnt += 1;
        }
        p0 += take;
    }
    state.reset(cfg.slots());
    tracing::info!("inkling2_loop nll: {n} rows on the {} in {:.1} s", if card == Some(true) { "card" } else { "host" }, t0.elapsed().as_secs_f64());
    Some((nll, cnt))
}

/// One token; the logits when asked.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, token_id: u32, position: usize, pool: Option<&Pool>, logits: &mut Vec<f32>, want_logits: bool) {
    if position == 0 || state.pos != position {
        if position != 0 && state.pos != position {
            tracing::warn!("inkling2_loop: position {position} after {}; the sequence restarts", state.pos);
        }
        state.reset(cfg.slots());
    }
    if on_card(globals, layers, cfg, state, &[token_id], position, want_logits, false, logits, pool) {
        return;
    }
    state.tokens.push(token_id);
    let x = host_rows(globals, layers, cfg, state, &[token_id], position, pool);
    state.pos = position + 1;
    if want_logits {
        *logits = head_rows(globals, cfg, &x, 1, pool);
    }
}

/// `CMF_INKLING_SPEC`: unset or `auto` lets [`SpecPolicy`] pick each round's
/// drafts (`usize::MAX` here), 0 is plain decoding, a number pins the drafts a
/// round offers (the block's 7 at most).
pub fn spec_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("CMF_INKLING_SPEC").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX))
}

/// A verify of `k` drafts against a plain token, NVIDIA L4 (calls of 1..8
/// rows: 18.4, 23.1, 25.8, 28.4, 30.7, 32.8, 34.9, 36.9 ms): the price of a
/// depth no round has timed yet, scaled from the nearest timed one.
const SPEC_REL: [f32; 8] = [1.0, 1.26, 1.40, 1.54, 1.67, 1.78, 1.90, 2.0];

/// How often draft `j` holds once the one before it held, before a round has
/// shown it (DSpark's own figures for Inkling 2: a half to two thirds of
/// drafts kept on prose and chat).
const SPEC_HOLD_PRIOR: [f32; 7] = [0.70, 0.65, 0.62, 0.60, 0.58, 0.56, 0.55];

/// The DSpark depth that lands the most tokens per millisecond: every round
/// and plain token is timed from one decode step to the next (the
/// pipeline's own work included), a depth's price is its running average
/// (one never timed is scaled from the nearest timed one by [`SPEC_REL`]),
/// and how often each draft holds is a running average too. 0 drafts, plain
/// decoding, is a depth like the others; while it wins, a round of 2 every
/// 32 tokens keeps the holds current.
#[derive(Clone, Debug)]
pub struct SpecPolicy {
    hold: [f32; 7],
    wall: [f32; 8],
    seen: [u8; 8],
    k: usize,
    dwell: u8,
    plain_run: u16,
    picked: Option<std::time::Instant>,
    last: Option<(std::time::Instant, usize, usize)>,
}

impl Default for SpecPolicy {
    fn default() -> Self {
        Self { hold: SPEC_HOLD_PRIOR, wall: [0.0; 8], seen: [0; 8], k: 3, dwell: 0, plain_run: 0, picked: None, last: None }
    }
}

impl SpecPolicy {
    /// The drafts for the step at `pos` (`room`: tokens the request still
    /// allows past the anchor). Times the previous step when it ended here.
    pub fn pick(&mut self, pos: usize, room: usize) -> usize {
        let now = std::time::Instant::now();
        self.picked = Some(now);
        if let Some((t, k, at)) = self.last.take() {
            if at == pos {
                let w = now.duration_since(t).as_secs_f32() * 1e3;
                // a depth's first step builds its call's graph and is not counted
                match self.seen[k] {
                    0 => self.seen[k] = 1,
                    1 => (self.wall[k], self.seen[k]) = (w, 2),
                    _ if w < self.wall[k] * 4.0 => self.wall[k] += 0.2 * (w - self.wall[k]),
                    _ => {}
                }
            }
        }
        let pinned = spec_k();
        let k = if pinned != usize::MAX {
            pinned.min(7)
        } else {
            self.plan()
        };
        let k = if k == 0 && self.plain_run >= 32 { 2 } else { k };
        k.min(room)
    }

    /// Record the step just taken at `pos`: `k` drafts offered, `accepted`
    /// kept (0 and 0 for a plain token).
    pub fn done(&mut self, pos: usize, k: usize, accepted: usize) {
        let k = k.min(7);
        self.last = self.picked.take().map(|t| (t, k, pos + 1 + accepted));
        if k == 0 {
            self.plain_run = self.plain_run.saturating_add(1);
            return;
        }
        self.plain_run = 0;
        for j in 1..=k {
            if accepted + 1 >= j {
                let hit = if accepted >= j { 1.0 } else { 0.0 };
                self.hold[j - 1] += 0.1 * (hit - self.hold[j - 1]);
            }
        }
        // text this predictable is worth a deeper round now and then
        if k < 7 && self.hold[k - 1] >= 0.9 {
            for j in k + 1..=7 {
                self.hold[j - 1] += 0.02 * (self.hold[k - 1] - self.hold[j - 1]);
            }
        }
    }

    /// The step after a reset or a new request is not timed against the last one.
    pub fn forget_step(&mut self) {
        self.last = None;
        self.picked = None;
    }

    fn price(&self, k: usize) -> Option<f32> {
        if self.seen[k] >= 2 {
            return Some(self.wall[k]);
        }
        let near = (0..8usize).filter(|&j| self.seen[j] >= 2).min_by_key(|&j| j.abs_diff(k))?;
        Some(self.wall[near] * SPEC_REL[k] / SPEC_REL[near])
    }

    fn plan(&mut self) -> usize {
        let mut rate = [0f32; 8];
        let (mut tokens, mut reach) = (1.0f32, 1.0f32);
        for k in 0..8usize {
            if k > 0 {
                reach *= self.hold[k - 1];
                tokens += reach;
            }
            match self.price(k) {
                Some(p) if p > 0.0 => rate[k] = tokens / p,
                // nothing timed yet: stay
                _ => return self.k,
            }
        }
        if self.dwell > 0 {
            self.dwell -= 1;
            return self.k;
        }
        let best = (0..8usize).max_by(|&a, &b| rate[a].total_cmp(&rate[b])).unwrap_or(self.k);
        if best != self.k && rate[best] > rate[self.k] * 1.04 {
            if std::env::var("CMF_INKLING_SPEC_TIME").is_ok() {
                eprintln!("dspark: {} → {best} drafts (a step costs {:.1?} ms, a draft holds {:.2?})", self.k, self.wall, self.hold);
            }
            self.k = best;
            self.dwell = 6;
        }
        self.k
    }
}

/// One DSpark round at the decode tip, on the card: `anchor` (the token the
/// sampler took for `pos`) and up to `k` drafts through the target in one
/// call; the drafts it agrees with (its greedy picks) are kept, the rest rolled
/// back. Returns the accepted drafts and leaves `logits` the last kept row's;
/// `None` when no round ran (no drafter, not on the card, a refusal): nothing
/// changed and the caller forwards the anchor itself.
pub fn spec_round(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, anchor: u32, pos: usize, k: usize, logits: &mut Vec<f32>) -> Option<Vec<u32>> {
    globals.draft.as_ref()?;
    if k == 0 || state.card != Some(true) || state.pos != pos || pos == 0 {
        return None;
    }
    let drafts = crate::gpu::inkling_draft(state.sequence_id, globals, cfg, anchor, pos, k)?;
    if drafts.is_empty() || !crate::gpu::inkling_ring_snap(state.sequence_id) {
        return None;
    }
    let mut ids = Vec::with_capacity(drafts.len() + 1);
    ids.push(anchor);
    ids.extend_from_slice(&drafts);
    // each row's greedy pick on the card; only the kept row's logits come over
    let mut picks = Vec::with_capacity(ids.len());
    if !crate::gpu::inkling_verify(state.sequence_id, globals, layers, cfg, &ids, pos, &mut picks) || picks.len() != ids.len() {
        // the card let go mid-round: the host replays what it ran
        state.card = Some(false);
        return None;
    }
    let vocab = globals.lm_head.rows();
    let mut acc = 0usize;
    while acc < drafts.len() && picks[acc] == drafts[acc] {
        acc += 1;
    }
    let keep = acc + 1;
    if keep < ids.len() && !crate::gpu::inkling_ring_back(state.sequence_id, keep) {
        state.card = Some(false);
        return None;
    }
    if !crate::gpu::inkling_row_logits(state.sequence_id, acc, vocab, logits) {
        state.card = Some(false);
        return None;
    }
    state.tokens.extend_from_slice(&ids[..keep]);
    state.pos = pos + keep;
    Some(drafts[..acc].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        Cfg {
            dim: 4,
            layers: 24,
            nh: 2,
            nkv: 1,
            hd: 2,
            d_rel: 2,
            eps: 1e-6,
            window: 4,
            rel_extent: 4,
            qk_clip: 22.6,
            rel_clip: 8.0,
            log_alpha: 0.1,
            log_floor: 8192.0,
            block: 6,
            core_start: 6,
            core_end: 18,
            passes: 3,
            top_k: 2,
            route_scale: 8.0,
            gate_cap: 4.0,
            up_cap: 25.0,
            logit_cap: 30.0,
            max_seq: 64,
        }
    }

    #[test]
    fn the_plan_visits_the_core_three_times_with_its_own_slots() {
        let c = cfg();
        let plan = c.plan();
        assert_eq!(plan.len(), 48);
        assert_eq!(c.slots(), 48);
        assert_eq!(plan[6], (6, 0));
        assert_eq!(plan[18], (6, 1));
        assert_eq!(plan[30], (6, 2));
        assert_eq!(plan[42], (18, 0));
        assert_eq!(c.slot(6, 1), 24);
        assert_eq!(c.slot(17, 2), 47);
        let mut slots: Vec<usize> = plan.iter().map(|&(l, p)| c.slot(l, p)).collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 48);
    }

    #[test]
    fn the_dspark_depth_follows_the_holds_and_the_prices() {
        // L4 prices: a plain token 18.4 ms, a verify of k drafts by SPEC_REL
        let timed = |hold: [f32; 7]| {
            let mut p = SpecPolicy { hold, ..SpecPolicy::default() };
            for k in 0..8 {
                p.wall[k] = 18.4 * SPEC_REL[k] + if k > 0 { 2.0 } else { 0.0 };
                p.seen[k] = 2;
            }
            p.plan()
        };
        // text that holds: a deep round
        assert!(timed([0.95; 7]) >= 6);
        // drafts that rarely hold: plain decoding
        assert_eq!(timed([0.15; 7]), 0);
        // chat-like holds: a short round
        let k = timed([0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5]);
        assert!((1..=4).contains(&k), "{k}");
        // nothing timed yet: the starting depth
        assert_eq!(SpecPolicy::default().plan(), 3);
        // one timed depth prices the rest from SPEC_REL
        let mut p = SpecPolicy::default();
        p.wall[3] = 28.4;
        p.seen[3] = 2;
        assert!((p.price(0).unwrap() - 28.4 / 1.54).abs() < 1e-3);
    }

    #[test]
    fn a_varloop_plan_runs_the_core_any_number_of_times() {
        for passes in [1usize, 2, 5] {
            let c = Cfg { passes, ..cfg() };
            let plan = c.plan();
            assert_eq!(plan.len(), 12 + 12 * passes);
            assert_eq!(c.slots(), 24 + 12 * (passes - 1));
            assert_eq!(plan.last(), Some(&(23, 0)));
            let mut slots: Vec<usize> = plan.iter().map(|&(l, p)| c.slot(l, p)).collect();
            slots.sort();
            slots.dedup();
            assert_eq!(slots.len(), c.slots());
            assert_eq!(slots.last(), Some(&(c.slots() - 1)));
        }
    }

    #[test]
    fn tau_is_one_below_the_floor() {
        assert_eq!(tau(0, 0.1, 8192.0), 1.0);
        assert_eq!(tau(8191, 0.1, 8192.0), 1.0);
        assert!((tau(16383, 0.1, 8192.0) - (1.0 + 0.1 * 2f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn the_mixture_of_equal_states_is_that_state_normed() {
        let s = [1.0f32, 2.0, 3.0, 4.0];
        let m = mix(&[&s, &s], &[0.3, -0.2, 0.1, 0.5], 4);
        let n = unit_rms(&s, 1e-6);
        for (a, b) in m[0].iter().zip(&n) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn a_static_conv_keeps_its_ring_newest_first() {
        // one channel, taps [current 1, lag1 0.5, lag2 0.25, lag3 0]
        let c = Conv { taps: vec![1.0, 0.5, 0.25, 0.0], channels: 1, kernel: 4, proj: None, groups: 1 };
        let mut ring = Vec::new();
        let s5 = 5f32.sqrt();
        let y0 = c.step_with(&[2.0], &mut ring, None);
        assert!((y0[0] - (2.0 + 2.0) / s5).abs() < 1e-6);
        let y1 = c.step_with(&[4.0], &mut ring, None);
        assert!((y1[0] - (4.0 + 4.0 + 1.0) / s5).abs() < 1e-6);
        let y2 = c.step_with(&[8.0], &mut ring, None);
        assert!((y2[0] - (8.0 + 8.0 + 2.0 + 0.5) / s5).abs() < 1e-6);
        assert_eq!(ring, vec![8.0, 4.0, 2.0]);
    }

    #[test]
    fn the_router_picks_by_the_biased_score_and_weighs_by_the_plain_one() {
        let r = route_logits(&[0.0, 2.0, -1.0], &[0.9, 0.0, 0.0], 2);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[1].0, 1);
        let (s0, s1) = (0.5f32, sigmoid(2.0));
        assert!((r[0].1 - s0 / (s0 + s1)).abs() < 1e-6);
    }
}
