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

fn qmv(t: &QTensor, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let mut out = vec![0f32; t.rows()];
    t.matvec(x, &mut out, pool);
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
    /// first); `dyn_in` the input the per-token taps project.
    fn step(&self, x: &[f32], ring: &mut Vec<f32>, dyn_in: Option<&[f32]>, pool: Option<&Pool>) -> Vec<f32> {
        let (c, k) = (self.channels, self.kernel);
        if ring.len() != (k - 1) * c {
            *ring = vec![0f32; (k - 1) * c];
        }
        let dynw = match (&self.proj, dyn_in) {
            (Some(p), Some(xi)) => Some(mv(pool, p, xi, self.groups * k)),
            _ => None,
        };
        let per = c / self.groups.max(1);
        let scale = 1.0 / (((if dynw.is_some() { 2 } else { 1 }) * k + 1) as f32).sqrt();
        let mut out = vec![0f32; c];
        for ch in 0..c {
            let tp = &self.taps[ch * k..(ch + 1) * k];
            let mut acc = 0f32;
            for lag in 0..k {
                let xv = if lag == 0 { x[ch] } else { ring[(lag - 1) * c + ch] };
                let mut w = tp[lag];
                if let Some(d) = &dynw {
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

    /// `(1 − tanh f)·x + tanh g·m`.
    fn apply(&self, x: &[f32], m: &[f32], eps: f64, pool: Option<&Pool>) -> Vec<f32> {
        let dim = x.len();
        let rank = self.down.len() / (2 * dim);
        let mut cat = unit_rms(x, eps);
        cat.extend(unit_rms(m, eps));
        let d = mv(pool, &self.down, &cat, rank);
        let fg = mv(pool, &self.forget, &d, dim);
        let gn = mv(pool, &self.gain, &d, dim);
        (0..dim).map(|i| (1.0 - fg[i].tanh()) * x[i] + gn[i].tanh() * m[i]).collect()
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
}

impl State {
    pub fn reset(&mut self, n_slots: usize) {
        self.pos = 0;
        self.tokens.clear();
        self.slots = (0..n_slots).map(|_| SlotState::default()).collect();
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
    let cfg = Cfg {
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
    let globals = Globals {
        model: model.clone(),
        embed: Embed::load(model, "model.embed_tokens.weight")?,
        embed_norm: f(model, "model.embed_tokens.norm.weight")?,
        lm_head: t(model, "lm_head.weight")?,
        norm: f(model, "model.norm.weight")?,
        final_query: f(model, "model.final_input.query")?,
        message_query: f(model, "model.message_input.query")?,
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
    let mut state = State { pos: 0, tokens: Vec::new(), slots: Vec::new() };
    state.reset(cfg.slots());
    Ok((globals, layers, cfg, state))
}

impl Attn {
    /// One position: q from `xq`, k from `xk`, v from `xv` (normed inputs);
    /// its key and value into the slot's cache; the attention with the
    /// relative bias and the sink; the gated O projection.
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, xq: &[f32], xk: &[f32], xv: &[f32], ss: &mut SlotState, pos: usize, sliding: bool, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let (nh, nkv, hd, dr) = (cfg.nh, cfg.nkv, cfg.hd, cfg.d_rel);
        let per = nh / nkv;
        let mut q = qmv(&self.q, xq, pool);
        for (v, s) in q.iter_mut().zip(&self.q_scale) {
            *v *= s;
        }
        l2_clip(&mut q, hd, cfg.qk_clip);
        let mut r = qmv(&self.rel, xq, pool);
        for (v, s) in r.iter_mut().zip(&self.rel_scale) {
            *v *= s;
        }
        l2_clip(&mut r, dr, cfg.rel_clip);
        if !sliding {
            let tu = tau(pos, cfg.log_alpha, cfg.log_floor);
            for v in q.iter_mut().chain(r.iter_mut()) {
                *v *= tu;
            }
        }
        let sinks = mv(pool, &self.sink, xq, nh);
        let gate = qmv(&self.gate, xq, pool);

        let kl = qmv(&self.k, xk, pool);
        let mut k = self.k_conv.step(&kl, &mut ss.rings[0], Some(xk), pool);
        l2_clip(&mut k, hd, cfg.qk_clip);
        let vl = qmv(&self.v, xv, pool);
        let mut v = self.v_conv.step(&vl, &mut ss.rings[1], Some(xv), pool);
        for h in v.chunks_mut(hd) {
            let n = unit_rms(h, cfg.eps);
            h.copy_from_slice(&n);
        }
        let row = nkv * hd;
        ss.k.extend_from_slice(&k);
        ss.v.extend_from_slice(&v);
        if sliding && ss.k.len() > cfg.window * row {
            let drop = ss.k.len() / row - cfg.window;
            ss.k.drain(..drop * row);
            ss.v.drain(..drop * row);
            ss.base += drop;
        }
        let first = ss.base;
        debug_assert_eq!(first + ss.k.len() / row, pos + 1);

        let extent = self.rel_table.len() / dr;
        let scale = 1.0 / (hd as f32).sqrt();
        let (kc, vc) = (&ss.k, &ss.v);
        let mut out = vec![0f32; nh * hd];
        let head = |h: usize, o: &mut [f32]| {
            let g = h / per;
            let qh = &q[h * hd..(h + 1) * hd];
            let rh = &r[h * dr..(h + 1) * dr];
            let n = pos + 1 - first;
            let mut s = vec![0f32; n];
            let mut mx = sinks[h];
            for (i, sv) in s.iter_mut().enumerate() {
                let tpos = first + i;
                let kr = &kc[(i * nkv + g) * hd..(i * nkv + g + 1) * hd];
                let mut x = kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale;
                let d = pos - tpos;
                if d < extent {
                    x += (0..dr).map(|j| rh[j] * self.rel_table[j * extent + d]).sum::<f32>();
                }
                *sv = x;
                mx = mx.max(x);
            }
            let mut den = ((sinks[h] - mx) as f64).exp();
            for sv in s.iter_mut() {
                *sv = (*sv - mx).exp();
                den += *sv as f64;
            }
            let inv = (1.0 / den) as f32;
            for (i, &w) in s.iter().enumerate() {
                let vr = &vc[(i * nkv + g) * hd..(i * nkv + g + 1) * hd];
                let w = w * inv;
                for (a, b) in o.iter_mut().zip(vr) {
                    *a += w * b;
                }
            }
        };
        match pool {
            Some(p) => {
                let addr = crate::pool::SendMut::new(out.as_mut_ptr());
                p.run_rows(nh, &|a, b| {
                    for h in a..b {
                        head(h, unsafe { std::slice::from_raw_parts_mut(addr.at(h * hd), hd) });
                    }
                });
            }
            None => {
                for (h, o) in out.chunks_mut(hd).enumerate() {
                    head(h, o);
                }
            }
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        qmv(&self.o, &out, pool)
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

    fn forward(&self, x: &[f32], cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for (e, w) in self.route(x, cfg.top_k, pool) {
            let [gt, up, dn] = &self.experts[e];
            let g = qmv(gt, x, pool);
            let u = qmv(up, x, pool);
            let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| softcap(g, cfg.gate_cap) * sigmoid(g) * softcap(u, cfg.up_cap)).collect();
            let ye = qmv(dn, &a, pool);
            let w = w * cfg.route_scale;
            for (o, v) in y.iter_mut().zip(&ye) {
                *o += w * v;
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
fn dump(slot: usize, pos: usize, kind: &str, v: &[f32]) {
    static CFG: std::sync::OnceLock<Option<(String, Vec<usize>)>> = std::sync::OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let s = std::env::var("CMF_INKLING_DUMP").ok()?;
        let mut it = s.split(',');
        let dir = it.next()?.to_string();
        Some((dir, it.filter_map(|p| p.parse().ok()).collect()))
    });
    if let Some((dir, ps)) = cfg
        && ps.contains(&pos)
    {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{kind}_s{slot}_p{pos}.f32"), b);
    }
}

/// One layer application at one position: `h` the running sum, `blocks` the
/// closed block states (a boundary layer closes `h` into them), `msgs` the
/// four messages of a loop pass after the first.
#[allow(clippy::too_many_arguments)]
fn layer_step(l: &Layer, li: usize, ss: &mut SlotState, h: &[f32], blocks: &mut Vec<Vec<f32>>, msgs: Option<&[Vec<f32>]>, pos: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
    let dim = cfg.dim;
    let boundary = li % cfg.block == 0;
    let mut inputs: [Vec<f32>; 3] = match &l.attn_query {
        Some(q) => {
            let mut st: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
            st.push(h);
            let m = mix(&st, q, dim);
            if l.sliding { [m[0].clone(), m[0].clone(), m[0].clone()] } else { [m[0].clone(), m[1].clone(), m[2].clone()] }
        }
        None => [h.to_vec(), h.to_vec(), h.to_vec()],
    };
    if let (Some(ms), Some(g)) = (msgs, &l.gates) {
        for j in 0..3 {
            inputs[j] = g[j].apply(&inputs[j], &ms[j], cfg.eps, pool);
        }
    }
    let xq = rms_w(&inputs[0], &l.in_norm[0], cfg.eps);
    let (xk, xv) = if l.sliding && msgs.is_none() {
        (xq.clone(), xq.clone())
    } else {
        (rms_w(&inputs[1], &l.in_norm[1], cfg.eps), rms_w(&inputs[2], &l.in_norm[2], cfg.eps))
    };
    let a = l.attn.forward(&xq, &xk, &xv, ss, pos, l.sliding, cfg, pool);
    let a = l.post_attn_conv.step(&a, &mut ss.rings[2], None, pool);
    let r: Vec<f32> = if boundary {
        blocks.push(h.to_vec());
        a
    } else {
        h.iter().zip(&a).map(|(x, y)| x + y).collect()
    };
    let mut st: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
    st.push(&r);
    let mut m = mix(&st, &l.mlp_query, dim).swap_remove(0);
    if let (Some(ms), Some(g)) = (msgs, &l.gates) {
        m = g[3].apply(&m, &ms[3], cfg.eps, pool);
    }
    let x = rms_w(&m, &l.post_norm, cfg.eps);
    let y = l.moe.forward(&x, cfg, pool);
    let y = l.post_mlp_conv.step(&y, &mut ss.rings[3], None, pool);
    r.iter().zip(&y).map(|(a, b)| a + b).collect()
}

/// The final mixture of one position after the whole stack (host): its
/// normed hidden, ready for the head.
fn host_hidden(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, token_id: u32, pos: usize, pool: Option<&Pool>) -> Vec<f32> {
    let dim = cfg.dim;
    let mut e = vec![0f32; dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(&globals.model, token_id as usize, &mut e);
    }
    let mut h = rms_w(&e, &globals.embed_norm, cfg.eps);
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for li in 0..cfg.core_start {
        h = layer_step(&layers[li], li, &mut state.slots[li], &h, &mut blocks, None, pos, cfg, pool);
        dump(li, pos, "h", &h);
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
                let st: Vec<&[f32]> = bank.iter().map(|b| b.as_slice()).collect();
                mix(&st, &globals.message_query[i * 4 * dim..(i + 1) * 4 * dim], dim)
            });
            h = layer_step(&layers[li], li, &mut state.slots[slot], &h, &mut blocks, msgs.as_deref(), pos, cfg, pool);
            dump(slot, pos, "h", &h);
        }
    }
    for li in cfg.core_end..cfg.layers {
        h = layer_step(&layers[li], li, &mut state.slots[li], &h, &mut blocks, None, pos, cfg, pool);
        dump(li, pos, "h", &h);
    }
    let mut st: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
    st.push(&h);
    let fin = mix(&st, &globals.final_query, dim).swap_remove(0);
    rms_w(&fin, &globals.norm, cfg.eps)
}

/// The softcapped logits of a normed hidden.
fn head(globals: &Globals, cfg: &Cfg, x: &[f32], logits: &mut Vec<f32>, pool: Option<&Pool>) {
    logits.resize(globals.lm_head.rows(), 0.0);
    globals.lm_head.matvec(x, logits, pool);
    if cfg.logit_cap > 0.0 {
        for v in logits.iter_mut() {
            *v = softcap(*v, cfg.logit_cap);
        }
    }
}

/// The prompt through a batched path; 0 = not taken (the caller runs it a
/// token at a time).
#[allow(clippy::too_many_arguments)]
pub fn forward_prompt(_globals: &Globals, _layers: &[Layer], _cfg: &Cfg, _state: &mut State, _ids: &[u32], _start: usize, _pool: Option<&Pool>, _logits: &mut Vec<f32>) -> usize {
    0
}

/// A window's negative log-likelihood through a batched path; None = the
/// caller scores it a token at a time.
pub fn nll(_globals: &Globals, _layers: &[Layer], _cfg: &Cfg, _state: &mut State, _ids: &[u32], _from: usize, _pool: Option<&Pool>) -> Option<(f64, usize)> {
    None
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
    state.tokens.push(token_id);
    let x = host_hidden(globals, layers, cfg, state, token_id, position, pool);
    state.pos = position + 1;
    if want_logits {
        dump(cfg.slots(), position, "final", &x);
        head(globals, cfg, &x, logits, pool);
    }
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
        let y0 = c.step(&[2.0], &mut ring, None, None);
        assert!((y0[0] - (2.0 + 2.0) / s5).abs() < 1e-6);
        let y1 = c.step(&[4.0], &mut ring, None, None);
        assert!((y1[0] - (4.0 + 4.0 + 1.0) / s5).abs() < 1e-6);
        let y2 = c.step(&[8.0], &mut ring, None, None);
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
