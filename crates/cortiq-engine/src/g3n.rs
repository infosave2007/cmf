//! Gemma-3n (E-series) text stack: AltUp (4 hidden replicas with
//! predict/correct), LAuReL low-rank residual, Per-Layer Embeddings,
//! KV-cache sharing across the last 15 layers, and gaussian-top-k
//! activation sparsity. Formulas ported 1:1 from transformers'
//! modular_gemma3n.py; every norm is plain x̂·w with eps inside the
//! mean. The stack runs its own per-token forward (the 4-replica flow
//! does not fit the shared layer loop); batched/pair prefill and the
//! GPU graphs stay off for this arch.

use crate::attention::{self, QwenAttnCfg};
use crate::inference;
use crate::pool::Pool;
use crate::qtensor::QTensor;

pub const ALTUP_N: usize = 4;

pub struct G3nAltUp {
    /// [hidden] — plain-w RMS norm for the router input.
    pub router_norm: Vec<f32>,
    /// [ALTUP_N, hidden]
    pub modality_router: QTensor,
    /// [ALTUP_N², ALTUP_N]
    pub prediction_coefs: QTensor,
    /// [ALTUP_N, ALTUP_N]
    pub correction_coefs: QTensor,
    /// [hidden] — elementwise scale of the active corrected replica.
    pub correct_output_scale: Vec<f32>,
}

pub struct G3nLaurel {
    /// [rank, hidden]
    pub left: QTensor,
    /// [hidden, rank]
    pub right: QTensor,
    /// [hidden]
    pub post_norm: Vec<f32>,
}

pub struct G3nLayer {
    pub altup: G3nAltUp,
    pub laurel: G3nLaurel,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_ffw_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    /// Attention: shared-KV layers carry no k/v projections and read
    /// the cache of `kv_share_src` instead of appending their own.
    pub wq: QTensor,
    pub wk: Option<QTensor>,
    pub wv: Option<QTensor>,
    pub wo: QTensor,
    pub q_norm: Vec<f32>,
    pub k_norm: Option<Vec<f32>>,
    pub kv_share_src: Option<usize>,
    pub sliding: bool,
    /// MLP with optional gaussian-top-k sparsity on the gate.
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    /// Φ⁻¹(p)·… threshold active when p > 0 (first 10 layers, p=0.95).
    pub sparsity: f32,
    /// Per-layer input injection (PLE).
    pub ple_gate: QTensor,
    pub ple_proj: QTensor,
    pub post_ple_norm: Vec<f32>,
}

pub struct G3nGlobals {
    /// [hidden, hidden] ×3 — replica 1..3 construction from replica 0.
    pub altup_proj: Vec<QTensor>,
    /// [hidden, hidden] ×3 — replica 1..3 unembedding at the top.
    pub altup_unembed: Vec<QTensor>,
    /// [ple_vocab, L·ple_dim] — the per-layer embedding table.
    pub ple_embed: QTensor,
    /// [L·ple_dim, hidden]
    pub ple_model_proj: QTensor,
    /// [ple_dim]
    pub ple_norm: Vec<f32>,
    pub ple_vocab: usize,
    pub ple_dim: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub rms_eps: f64,
    /// inv_freq for sliding (local theta) and full (global theta) layers.
    pub inv_freq_local: Vec<f32>,
    pub inv_freq_global: Vec<f32>,
    pub window: usize,
}

/// Plain x̂·w RMS norm (eps inside the mean) into a fresh Vec.
fn rms(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (ms + eps).powf(-0.5);
    x.iter().zip(w).map(|(&v, &g)| (v as f64 * inv * g as f64) as f32).collect()
}

fn rms_mag(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

fn gelu_tanh(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())) as f32
}

/// Acklam's inverse normal CDF (|ε| < 1.15e-9) — the sparsity cutoff
/// multiplier Φ⁻¹(p); only p ∈ (0.5, 1) occurs in the configs.
pub fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let (pl, ph) = (0.02425, 1.0 - 0.02425);
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= ph {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -inv_normal_cdf(1.0 - p)
    }
}

fn matvec(w: &QTensor, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let mut o = vec![0.0f32; w.rows()];
    w.matvec(x, &mut o, pool);
    o
}

impl G3nGlobals {
    /// Extended embedding for one token: [e(hidden) ++ per_layer_input
    /// (L·ple_dim)] — the PLE half needs the token ID, so it rides with
    /// the embedding instead of threading ids through the forward.
    /// `e` is the already-scaled (√hidden) token embedding.
    pub fn extend_embedding(&self, id: u32, e: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let (l, d) = (self.num_layers, self.ple_dim);
        // proj = per_layer_model_projection(e) · hidden^-0.5 → per-layer rms.
        let mut proj = matvec(&self.ple_model_proj, e, pool);
        let scale = (self.hidden as f32).powf(-0.5);
        for v in proj.iter_mut() {
            *v *= scale;
        }
        // ple rows: table[id] · √ple_dim; ids past the PLE vocab (the
        // multimodal placeholder tail) contribute zeros.
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
                let ple = ple_row.as_ref().map(|r| r[li * d + j] * ple_scale).unwrap_or(0.0);
                out.push((pn[j] + ple) * half);
            }
        }
        out
    }
}

/// One decoded row of a (possibly quantized) embedding-style table.
fn matvec_row(t: &QTensor, row: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t.cols()];
    t.row_f32(row, &mut out);
    out
}

/// AltUp modalities: tanh(router(router_norm(x) · hidden^-1)).
fn modalities(altup: &G3nAltUp, x: &[f32], eps: f64, pool: Option<&Pool>) -> [f32; ALTUP_N] {
    let mut n = rms(x, &altup.router_norm, eps);
    let s = 1.0 / x.len() as f32;
    for v in n.iter_mut() {
        *v *= s;
    }
    let r = matvec(&altup.modality_router, &n, pool);
    let mut out = [0.0f32; ALTUP_N];
    for (o, &v) in out.iter_mut().zip(&r) {
        *o = v.tanh();
    }
    out
}

/// Forward one token through the whole gemma-3n stack. `ext` is the
/// extended embedding from `extend_embedding`; `caches` the per-layer
/// KV caches; returns the pre-final-norm hidden (the caller applies
/// final norm + lm_head as usual).
#[allow(clippy::too_many_arguments)]
pub fn g3n_forward(
    g: &G3nGlobals,
    layers: &[G3nLayer],
    ext: &[f32],
    position: usize,
    caches: &mut [crate::kv_cache::LayerKvCache],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let hs = g.hidden;
    let (e, ple) = ext.split_at(hs);
    let eps = g.rms_eps;

    // Replicas: X[0] = e; X[i] = altup_proj[i-1]·e renormalized to |e|.
    let target = rms_mag(e);
    let mut x: Vec<Vec<f32>> = Vec::with_capacity(ALTUP_N);
    x.push(e.to_vec());
    for i in 1..ALTUP_N {
        let mut xi = matvec(&g.altup_proj[i - 1], e, pool);
        let mag = (rms_mag(&xi).powi(2).max(1e-5)).sqrt();
        let f = (target / mag) as f32;
        for v in xi.iter_mut() {
            *v *= f;
        }
        x.push(xi);
    }

    for (li, lw) in layers.iter().enumerate() {
        // ── AltUp predict ──
        let m = modalities(&lw.altup, &x[0], eps, pool);
        let pc = matvec(&lw.altup.prediction_coefs, &m, pool); // [16]
        let mut pred: Vec<Vec<f32>> = (0..ALTUP_N)
            .map(|j| {
                let mut p = x[j].clone();
                for (pi, xp) in x.iter().enumerate() {
                    let c = pc[j * ALTUP_N + pi];
                    for (pv, &xv) in p.iter_mut().zip(xp) {
                        *pv += c * xv;
                    }
                }
                p
            })
            .collect();

        // ── attention + laurel on the ACTIVE replica ──
        let an = rms(&pred[0], &lw.input_norm, eps);
        let laurel = {
            let low = matvec(&lw.laurel.left, &an, pool);
            let up = matvec(&lw.laurel.right, &low, pool);
            let pn = rms(&up, &lw.laurel.post_norm, eps);
            an.iter().zip(&pn).map(|(&a, &b)| a + b).collect::<Vec<f32>>()
        };

        let inv_freq = if lw.sliding { &g.inv_freq_local } else { &g.inv_freq_global };
        let attn = g3n_attention(
            lw, &an, position, caches, li, num_heads, num_kv_heads, head_dim, inv_freq,
            if lw.sliding { Some(g.window) } else { None },
            eps, pool,
        );
        let attn_n = rms(&attn, &lw.post_attn_norm, eps);

        let half = 0.5f32.sqrt();
        let attn_laurel: Vec<f32> = (0..hs)
            .map(|i| (pred[0][i] + attn_n[i] + laurel[i]) * half)
            .collect();

        // ── MLP (gaussian-top-k sparsity on the gate) ──
        let fin = rms(&attn_laurel, &lw.pre_ffw_norm, eps);
        let mut gate_v = matvec(&lw.gate, &fin, pool);
        if lw.sparsity > 0.0 {
            let n = gate_v.len() as f64;
            let mean: f64 = gate_v.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var: f64 =
                gate_v.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let cutoff = mean + var.sqrt() * inv_normal_cdf(lw.sparsity as f64);
            for v in gate_v.iter_mut() {
                *v = (*v as f64 - cutoff).max(0.0) as f32;
            }
        }
        let up_v = matvec(&lw.up, &fin, pool);
        let act: Vec<f32> = gate_v
            .iter()
            .zip(&up_v)
            .map(|(&gv, &uv)| gelu_tanh(gv) * uv)
            .collect();
        let ffw = matvec(&lw.down, &act, pool);
        let ffw_n = rms(&ffw, &lw.post_ffw_norm, eps);
        let h2: Vec<f32> = attn_laurel.iter().zip(&ffw_n).map(|(&a, &b)| a + b).collect();

        // ── AltUp correct ──
        let m2 = modalities(&lw.altup, &h2, eps, pool);
        let cc = matvec(&lw.altup.correction_coefs, &m2, pool); // [4]
        let innov: Vec<f32> = h2.iter().zip(&pred[0]).map(|(&a, &b)| a - b).collect();
        for (j, p) in pred.iter_mut().enumerate() {
            let c = cc[j] + 1.0;
            for (pv, &iv) in p.iter_mut().zip(&innov) {
                *pv += c * iv;
            }
        }

        // ── Per-layer input injection (into replicas 1.. only) ──
        let first: Vec<f32> = pred[0]
            .iter()
            .zip(&lw.altup.correct_output_scale)
            .map(|(&v, &s)| v * s)
            .collect();
        let gated = matvec(&lw.ple_gate, &first, pool);
        let d = g.ple_dim;
        let pli = &ple[li * d..(li + 1) * d];
        let mixed: Vec<f32> = gated
            .iter()
            .zip(pli)
            .map(|(&gv, &pv)| gelu_tanh(gv) * pv)
            .collect();
        let proj = matvec(&lw.ple_proj, &mixed, pool);
        let pn = rms(&proj, &lw.post_ple_norm, eps);
        for p in pred.iter_mut().skip(1) {
            for (pv, &nv) in p.iter_mut().zip(&pn) {
                *pv += nv;
            }
        }
        x = pred;
    }

    // ── replicas → one hidden: unembed 1.. renormalized to |X[0]|, mean ──
    let target = rms_mag(&x[0]);
    let mut acc = x[0].clone();
    for i in 1..ALTUP_N {
        let mut yi = matvec(&g.altup_unembed[i - 1], &x[i], pool);
        let mag = (rms_mag(&yi).powi(2).max(1e-5)).sqrt();
        let f = (target / mag) as f32;
        for (a, &v) in acc.iter_mut().zip(&yi) {
            *a += v * f;
        }
        yi.clear();
    }
    let inv_n = 1.0 / ALTUP_N as f32;
    for v in acc.iter_mut() {
        *v *= inv_n;
    }
    acc
}

/// Attention for one position: standard GQA with per-head q/k norms,
/// scale 1.0, optional sliding window; shared-KV layers attend over the
/// source layer's cache and never append.
#[allow(clippy::too_many_arguments)]
fn g3n_attention(
    lw: &G3nLayer,
    an: &[f32],
    position: usize,
    caches: &mut [crate::kv_cache::LayerKvCache],
    li: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    inv_freq: &[f32],
    window: Option<usize>,
    eps: f64,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let heads_per_kv = nh / nkv;
    let mut q = matvec(&lw.wq, an, pool);
    for h in 0..nh {
        let head = &mut q[h * hd..(h + 1) * hd];
        let n = rms(head, &lw.q_norm, eps);
        head.copy_from_slice(&n);
        attention::rope_rotate_scaled(&mut q[h * hd..(h + 1) * hd], position, inv_freq, 1.0);
    }
    let src = lw.kv_share_src.unwrap_or(li);
    if lw.kv_share_src.is_none() {
        let wk = lw.wk.as_ref().expect("non-shared layer has k_proj");
        let wv = lw.wv.as_ref().expect("non-shared layer has v_proj");
        let mut k = matvec(wk, an, pool);
        let kn = lw.k_norm.as_ref().expect("non-shared layer has k_norm");
        for g in 0..nkv {
            let head = &mut k[g * hd..(g + 1) * hd];
            let n = rms(head, kn, eps);
            head.copy_from_slice(&n);
            attention::rope_rotate_scaled(&mut k[g * hd..(g + 1) * hd], position, inv_freq, 1.0);
        }
        let mut v = matvec(wv, an, pool);
        for g in 0..nkv {
            // v_norm is scale-less: x̂ alone.
            let head = &mut v[g * hd..(g + 1) * hd];
            let ms: f64 =
                head.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / hd as f64;
            let inv = (ms + eps).powf(-0.5) as f32;
            for x in head.iter_mut() {
                *x *= inv;
            }
        }
        caches[li].append(&k, &v, &[]);
    }
    let (mut ao, mut imp) = attention::attend_all_heads(
        &q,
        &caches[src],
        nh,
        heads_per_kv,
        hd,
        1.0,
        window,
        0.0,
    );
    attention::recycle_buf(&mut imp);
    let out = matvec(&lw.wo, &ao, pool);
    attention::recycle_buf(&mut ao);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predict step's coefficient transpose, checked against a
    /// LITERAL index port of the reference (reshape [4,4] → permute
    /// (…,3,2) → matmul over the replica axis): the engine computes
    /// pred[j] += Σ_p X[p]·pc[j·4+p].
    #[test]
    fn altup_predict_transpose_matches_reference() {
        let n = ALTUP_N;
        let h = 3usize;
        let x: Vec<Vec<f32>> = (0..n)
            .map(|j| (0..h).map(|i| ((j * h + i) as f32 * 0.7).sin()).collect())
            .collect();
        let pc: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.31).cos()).collect();

        // Reference: raw[a][b] = pc[a·n+b]; all_coefs = raw.permute(-1,-2)
        // → ac[p][j] = raw[j][p]; predictions[·,h,j] = Σ_p X[p][h]·ac[p][j];
        // then += X.
        let mut want = vec![vec![0.0f32; h]; n];
        for j in 0..n {
            for hh in 0..h {
                let mut acc = x[j][hh];
                for p in 0..n {
                    let ac_pj = pc[j * n + p]; // raw[j][p]
                    acc += x[p][hh] * ac_pj;
                }
                want[j][hh] = acc;
            }
        }
        // Engine formula (the g3n_forward predict block).
        for j in 0..n {
            for hh in 0..h {
                let mut got = x[j][hh];
                for p in 0..n {
                    got += pc[j * ALTUP_N + p] * x[p][hh];
                }
                assert!((got - want[j][hh]).abs() < 1e-6, "j={j} h={hh}");
            }
        }
    }

    #[test]
    fn inv_normal_cdf_hits_the_sparsity_quantile() {
        // Φ⁻¹(0.95) = 1.6448536269514722 (the only value in E4B's
        // pattern); Acklam is good to ~1e-9.
        assert!((inv_normal_cdf(0.95) - 1.6448536269514722).abs() < 1e-7);
        assert!(inv_normal_cdf(0.5).abs() < 1e-9);
    }
}
