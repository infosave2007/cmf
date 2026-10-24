//! Gemma-4 E-series (E4B) text stack: the gemma4 backbone (plain x̂·w
//! norms, dual-geometry attention — sliding GQA at `head_dim` with a
//! local RoPE base, full layers at `global_head_dim` with proportional
//! RoPE, scale-less V norm, gelu-tanh MLP) plus the two E-series
//! extensions that the shared layer loop does not model:
//!
//! * PLE — per-layer input embeddings. One packed table
//!   `[L·ple_dim, vocab]` indexed by token id, projected from the input
//!   embedding through `per_layer_model_projection`, RMS-normalized,
//!   mixed with the table row (√ple_dim · row, × 1/√2) and injected
//!   after every block through a gelu gate + projection + norm.
//! * Shared KV — the LAST `num_kv_shared_layers` layers carry no k/v
//!   projections at all and attend over the KV cache of the last
//!   non-shared layer *of the same attention type* (sliding tail layers
//!   read layer `first_shared-2`, full tail layers `first_shared-1` —
//!   exactly the `shared_kv_states[layer_type]` semantics of
//!   transformers and the `n_layer_kv_from_start` reuse of llama.cpp).
//!
//! Formulas are ported 1:1 from transformers' modeling_gemma4.py
//! (decoder layer forward + `project_per_layer_inputs`); every norm is
//! plain x̂·w with eps inside the mean. Like the gemma-3n stack, this
//! arch runs its own per-token forward: batched/pair prefill and the
//! GPU graphs stay off (gates key on `Pipeline::per_token_stack`).

use crate::attention::{self};
use crate::pool::Pool;
use crate::qtensor::QTensor;

pub struct G4eGlobals {
    /// `[vocab, L·ple_dim]` — the packed per-layer embedding table
    /// (row = token id, columns = layer-major ple rows).
    pub ple_embed: QTensor,
    /// `[L·ple_dim, hidden]` — input-embedding → per-layer projection.
    pub ple_model_proj: QTensor,
    /// `[ple_dim]` — RMS norm applied to each layer's model projection.
    pub ple_norm: Vec<f32>,
    pub ple_vocab: usize,
    pub ple_dim: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub rms_eps: f64,
    /// RoPE inverse frequencies: sliding layers (`head_dim`, local
    /// theta) and full layers (`global_head_dim`, model theta with the
    /// proportional partial-rotary prefix — the tail is zero, which is
    /// identity under `rope_rotate_scaled`).
    pub inv_freq_local: Vec<f32>,
    pub inv_freq_global: Vec<f32>,
    /// Sliding window (tokens, current included).
    pub window: usize,
    pub nh: usize,
    pub nkv: usize,
}

pub struct G4eLayer {
    /// Pre-attention norm (HF `input_layernorm`).
    pub input_norm: Vec<f32>,
    /// Attention-branch output norm (HF `post_attention_layernorm`).
    pub attn_out_norm: Vec<f32>,
    /// Pre-FFN norm (HF `pre_feedforward_layernorm`).
    pub pre_ffn_norm: Vec<f32>,
    /// FFN-branch output norm (HF `post_feedforward_layernorm`).
    pub post_ffw_norm: Vec<f32>,
    pub wq: QTensor,
    /// Shared-KV layers carry no k/v projections and read the cache of
    /// `kv_share_src` instead of appending their own.
    pub wk: Option<QTensor>,
    pub wv: Option<QTensor>,
    pub wo: QTensor,
    pub q_norm: Vec<f32>,
    pub k_norm: Option<Vec<f32>>,
    pub kv_share_src: Option<usize>,
    pub sliding: bool,
    /// Per-layer head width: `head_dim` on sliding layers,
    /// `global_head_dim` on full layers.
    pub hd: usize,
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    /// PLE injection: gelu(gate·h) ⊙ ple → proj → norm.
    pub ple_gate: QTensor,
    pub ple_proj: QTensor,
    pub post_ple_norm: Vec<f32>,
    /// HF `layer_scalar` — per-block output scale (absent = 1.0).
    pub layer_scalar: Option<f32>,
}

/// Plain x̂·w RMS norm (eps inside the mean) into a fresh Vec.
fn rms(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (ms + eps).powf(-0.5);
    x.iter()
        .zip(w)
        .map(|(&v, &g)| (v as f64 * inv * g as f64) as f32)
        .collect()
}

/// Scale-less x̂ (HF `v_norm` is `Gemma4RMSNorm(with_scale=False)`).
fn rms_plain(x: &mut [f32], eps: f64) {
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (ms + eps).powf(-0.5) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// gelu_pytorch_tanh — `config.hidden_activation` of the text tower.
fn gelu_tanh(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())) as f32
}

fn matvec(w: &QTensor, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let mut o = vec![0.0f32; w.rows()];
    w.matvec(x, &mut o, pool);
    o
}

/// One decoded row of a (possibly quantized) embedding-style table.
fn matvec_row(t: &QTensor, row: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t.cols()];
    t.row_f32(row, &mut out);
    out
}

impl G4eGlobals {
    /// Extended embedding for one token: `[e(hidden) ++ per_layer_input
    /// (L·ple_dim)]` — the PLE half needs the token ID, so it rides
    /// with the embedding instead of threading ids through the forward.
    /// `e` is the already-scaled (√hidden) token embedding.
    ///
    /// HF: `project_per_layer_inputs` = rmsnorm(e · hidden^-0.5) +
    /// √ple_dim·table[id], all × 1/√2.
    pub fn extend_embedding(&self, id: u32, e: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let (l, d) = (self.num_layers, self.ple_dim);
        let mut proj = matvec(&self.ple_model_proj, e, pool);
        let scale = (self.hidden as f32).powf(-0.5);
        for v in proj.iter_mut() {
            *v *= scale;
        }
        let ple_scale = (d as f32).sqrt();
        let ple_row: Option<Vec<f32>> = if (id as usize) < self.ple_vocab {
            Some(matvec_row(&self.ple_embed, id as usize))
        } else {
            None
        };
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let mut out = Vec::with_capacity(self.hidden + l * d);
        out.extend_from_slice(e);
        for li in 0..l {
            let pn = rms(&proj[li * d..(li + 1) * d], &self.ple_norm, self.rms_eps);
            for j in 0..d {
                let ple = ple_row
                    .as_ref()
                    .map(|r| r[li * d + j] * ple_scale)
                    .unwrap_or(0.0);
                out.push((pn[j] + ple) * half);
            }
        }
        out
    }
}

/// Forward one token through the whole gemma-4-E stack. `ext` is the
/// extended embedding from `extend_embedding`; `caches` the per-layer
/// KV caches; returns the pre-final-norm hidden (the caller applies the
/// final norm + lm_head + final-logit softcap as usual).
pub fn g4e_forward(
    g: &G4eGlobals,
    layers: &[G4eLayer],
    ext: &[f32],
    position: usize,
    caches: &mut [crate::kv_cache::LayerKvCache],
    num_heads: usize,
    num_kv_heads: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let hs = g.hidden;
    let (e, ple) = ext.split_at(hs);
    let eps = g.rms_eps;
    let nh = g.nh.min(num_heads).max(num_kv_heads);
    let nkv = g.nkv.min(num_heads);
    let hpk = (nh / nkv).max(1);
    let mut h = e.to_vec();

    for (li, lw) in layers.iter().enumerate() {
        let hd = lw.hd;
        let inv_freq: &[f32] = if lw.sliding {
            &g.inv_freq_local
        } else {
            &g.inv_freq_global
        };

        // ── attention ──
        let xn = rms(&h, &lw.input_norm, eps);
        let mut q = matvec(&lw.wq, &xn, pool);
        for hi in 0..nh {
            let head = &mut q[hi * hd..(hi + 1) * hd];
            let n = rms(head, &lw.q_norm, eps);
            head.copy_from_slice(&n);
            attention::rope_rotate_scaled(head, position, inv_freq, 1.0);
        }
        let src = lw.kv_share_src.unwrap_or(li);
        if lw.kv_share_src.is_none() {
            let wk = lw.wk.as_ref().expect("non-shared layer has k_proj");
            let wv = lw.wv.as_ref().expect("non-shared layer has v_proj");
            let kn = lw.k_norm.as_ref().expect("non-shared layer has k_norm");
            let mut k = matvec(wk, &xn, pool);
            for gi in 0..nkv {
                let head = &mut k[gi * hd..(gi + 1) * hd];
                let n = rms(head, kn, eps);
                head.copy_from_slice(&n);
                attention::rope_rotate_scaled(head, position, inv_freq, 1.0);
            }
            let mut v = matvec(wv, &xn, pool);
            for gi in 0..nkv {
                rms_plain(&mut v[gi * hd..(gi + 1) * hd], eps);
            }
            caches[li].append(&k, &v, &[]);
        }
        let window = if lw.sliding { Some(g.window) } else { None };
        let (mut ao, mut imp) =
            attention::attend_all_heads(&q, &caches[src], nh, hpk, hd, 1.0, window, 0.0);
        attention::recycle_buf(&mut imp);
        let a = matvec(&lw.wo, &ao, pool);
        attention::recycle_buf(&mut ao);

        // sandwich: h += rms(a, post_attention_layernorm)
        let an = rms(&a, &lw.attn_out_norm, eps);
        for (hv, av) in h.iter_mut().zip(&an) {
            *hv += av;
        }

        // ── FFN (gelu-tanh gated) ──
        let xn2 = rms(&h, &lw.pre_ffn_norm, eps);
        let gate_v = matvec(&lw.gate, &xn2, pool);
        let up_v = matvec(&lw.up, &xn2, pool);
        let act: Vec<f32> = gate_v
            .iter()
            .zip(&up_v)
            .map(|(&gv, &uv)| gelu_tanh(gv) * uv)
            .collect();
        let ffn = matvec(&lw.down, &act, pool);
        let ffn_n = rms(&ffn, &lw.post_ffw_norm, eps);
        for (hv, fv) in h.iter_mut().zip(&ffn_n) {
            *hv += fv;
        }

        // ── per-layer input injection (PLE) ──
        let d = g.ple_dim;
        let pli = &ple[li * d..(li + 1) * d];
        let mut gated = matvec(&lw.ple_gate, &h, pool);
        for gv in gated.iter_mut() {
            *gv = gelu_tanh(*gv);
        }
        let mixed: Vec<f32> = gated
            .iter()
            .zip(pli)
            .map(|(&gv, &pv)| gv * pv)
            .collect();
        let proj = matvec(&lw.ple_proj, &mixed, pool);
        let pn = rms(&proj, &lw.post_ple_norm, eps);
        for (hv, pv) in h.iter_mut().zip(&pn) {
            *hv += pv;
        }

        // ── per-block output scalar (HF `layer_scalar`) ──
        if let Some(s) = lw.layer_scalar {
            for hv in h.iter_mut() {
                *hv *= s;
            }
        }
    }
    h
}