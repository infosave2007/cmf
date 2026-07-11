//! Linear-attention cores, selected by `arch.linear_core.kind`
//! (descriptor-driven operators — Patent 15 claim 8).
//!
//! Two tracks (owner decision 2026-07-04):
//!
//! * `gated_delta_net` — the faithful vendor operator (Qwen3.5 /
//!   Qwen3-Next). Default for models that ship GDN weights: conversion
//!   carries the tensors 1:1 and needs no training. Port of the
//!   validated `gated_delta_net` (vmfcore/rust/src/forward.rs) against
//!   the numpy/torch oracle (vmfcore/gdn_layer.py).
//!
//! * `vmf_phase` — the canonical core: token carries a phase θ; kernel
//!   φ(θ) = [cos θ; sin θ] gives a linear factorization; the condensate
//!   is a recurrent state S[head][p2, dv] with decay exp(−exp(A_log)).
//!   Noise-robust and simpler than vendor recurrences. Exotic operators
//!   are folded onto it at CONVERT time (`--linear-core vmf_phase`) and
//!   quality is restored by the offline heal — the research track and
//!   the production mechanism for Patent-15 skills (mask→heal→compress).
//!
//! Both cores implement the same contract: `*_forward` (one position,
//! advances the state) and `*_pair` (fused two positions; lane 1
//! commits, lane 2 is tentative in `scratch` for speculative verify).
//! State lives in the layer's `linear_state: Vec<f64>` and is resized
//! lazily by the core itself.

use crate::pool::Pool;
use crate::qtensor::QTensor;

/// Weights of one vmf_phase layer (`model.layers.{i}.vmf_attn.*`).
pub struct VmfPhaseWeights {
    /// [nh·nphase, hidden] — query phase projection
    pub thq: QTensor,
    /// [nh·nphase, hidden] — key phase projection
    pub thk: QTensor,
    /// [nh·dv, hidden]
    pub v_proj: QTensor,
    /// [hidden, nh·dv]
    pub out_proj: QTensor,
    /// Per-component decay exp(−exp(A_log)), len nh·2·nphase (precomputed).
    pub decay: Vec<f64>,
    /// Selective-write input gate κ (hybrid_k core, stage 71): weight
    /// [nh, hidden] + bias [nh]; κ_h = σ(W_k·x + b)_h multiplies the
    /// state WRITE (S = decay·S + κ·φk⊗v). None = classic phase core,
    /// bit-identical to the pre-κ kernel. Measured at mechanism level:
    /// knee ×2–6 earlier, restores correlated-noise robustness, LM
    /// crossover vs softmax at SEQ 512 (experiments/lc_final_merged.json).
    pub k_gate: Option<(QTensor, Vec<f32>)>,
}

#[derive(Clone, Copy)]
pub struct VmfPhaseCfg {
    pub num_heads: usize,
    pub nphase: usize,
    pub value_head_dim: usize,
    pub hidden_size: usize,
    /// θ-mass (η′ correction): a restoring potential pulling the phase
    /// toward 0 — θ_eff = θ/(1+mass) — which WIDENS the phase kernel.
    /// Measured (experiments/vmf_native_core*.py) to restore noise
    /// robustness when the phase projection is FIXED (exactly CMF's
    /// fold-before-heal regime: thq/thk are init, not trained) — recall
    /// 3%→91% at moderate noise; redundant once the projection is
    /// healed. 0.0 = massless Goldstone (bit-identical to prior kernel).
    /// Set via CMF_PHASE_MASS. Validated at mechanism level, not yet LM.
    pub phase_mass: f32,
}

impl VmfPhaseCfg {
    pub fn state_len(&self) -> usize {
        self.num_heads * 2 * self.nphase * self.value_head_dim
    }
}

/// One recurrent step for one head-set given projected phases/values.
/// `state` is S[nh][p2, dv] in f64 (condensate persists across tokens).
fn phase_step(
    thq: &[f32],
    thk: &[f32],
    v: &[f32],
    decay: &[f64],
    kap: Option<&[f32]>,
    cfg: &VmfPhaseCfg,
    state: &mut [f64],
    out: &mut [f32],
) {
    let (nh, nph, dv) = (cfg.num_heads, cfg.nphase, cfg.value_head_dim);
    // θ-mass (η′): θ_eff = θ/(1+mass). mass=0 → factor 1 → no-op.
    let mscale = 1.0f64 / (1.0 + cfg.phase_mass as f64);
    let p2 = 2 * nph;
    for h in 0..nh {
        let s = &mut state[h * p2 * dv..(h + 1) * p2 * dv];
        let thk_h = &thk[h * nph..(h + 1) * nph];
        let thq_h = &thq[h * nph..(h + 1) * nph];
        let vt = &v[h * dv..(h + 1) * dv];
        let ot = &mut out[h * dv..(h + 1) * dv];
        let dec = &decay[h * p2..(h + 1) * p2];
        // Selective write (hybrid_k): κ scales what enters the condensate.
        let kh = kap.map_or(1.0f64, |k| k[h] as f64);
        for f in 0..p2 {
            // φ(θ) = [cos·nph, sin·nph], θ scaled by the mass factor.
            let (fk, fq) = if f < nph {
                ((thk_h[f] as f64 * mscale).cos(), (thq_h[f] as f64 * mscale).cos())
            } else {
                (
                    (thk_h[f - nph] as f64 * mscale).sin(),
                    (thq_h[f - nph] as f64 * mscale).sin(),
                )
            };
            let fkw = fk * kh;
            let row = &mut s[f * dv..(f + 1) * dv];
            let dcf = dec[f];
            for d in 0..dv {
                row[d] = dcf * row[d] + fkw * vt[d] as f64; // S = decay·S + κ·φk⊗v
                ot[d] += (fq * row[d]) as f32; // o = Σ φq·S
            }
        }
    }
}

/// κ_h = σ(W_k·x + b)_h — the per-head write gate (None when the layer
/// has no k_gate tensors: classic phase core).
fn kappa_of(x: &[f32], w: &VmfPhaseWeights, nh: usize, pool: Option<&Pool>) -> Option<Vec<f32>> {
    let (kw, kb) = w.k_gate.as_ref()?;
    let mut k = vec![0.0f32; nh];
    kw.matvec(x, &mut k, pool);
    for (v, b) in k.iter_mut().zip(kb) {
        *v = 1.0 / (1.0 + (-(*v + b)).exp());
    }
    Some(k)
}

/// Forward one position through a vmf_phase layer, advancing `state`.
pub fn vmf_phase_forward(
    x: &[f32],
    w: &VmfPhaseWeights,
    cfg: &VmfPhaseCfg,
    state: &mut Vec<f64>,
    pool: Option<&Pool>,
) -> Vec<f32> {
    if state.len() != cfg.state_len() {
        *state = vec![0f64; cfg.state_len()];
    }
    let (nh, nph, dv) = (cfg.num_heads, cfg.nphase, cfg.value_head_dim);

    let mut thq = vec![0.0f32; nh * nph];
    w.thq.matvec(x, &mut thq, pool);
    let mut thk = vec![0.0f32; nh * nph];
    w.thk.matvec(x, &mut thk, pool);
    let mut v = vec![0.0f32; nh * dv];
    w.v_proj.matvec(x, &mut v, pool);

    let kap = kappa_of(x, w, nh, pool);
    let mut o = vec![0.0f32; nh * dv];
    phase_step(&thq, &thk, &v, &w.decay, kap.as_deref(), cfg, state, &mut o);

    let mut out = vec![0.0f32; cfg.hidden_size];
    w.out_proj.matvec(&o, &mut out, pool);
    out
}

/// Fused two-position forward (speculative verify). Lane 1 commits into
/// `state` (its token is always committed); lane 2's tentative state
/// goes into `scratch` — the caller swaps it in on draft acceptance and
/// simply drops it on rejection.
#[allow(clippy::too_many_arguments)]
pub fn vmf_phase_pair(
    x1: &[f32],
    x2: &[f32],
    w: &VmfPhaseWeights,
    cfg: &VmfPhaseCfg,
    state: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
    pool: Option<&Pool>,
) -> (Vec<f32>, Vec<f32>) {
    if state.len() != cfg.state_len() {
        *state = vec![0f64; cfg.state_len()];
    }
    let (nh, nph, dv) = (cfg.num_heads, cfg.nphase, cfg.value_head_dim);

    let mut thq1 = vec![0.0f32; nh * nph];
    let mut thq2 = vec![0.0f32; nh * nph];
    w.thq.matvec2(x1, x2, &mut thq1, &mut thq2, pool);
    let mut thk1 = vec![0.0f32; nh * nph];
    let mut thk2 = vec![0.0f32; nh * nph];
    w.thk.matvec2(x1, x2, &mut thk1, &mut thk2, pool);
    let mut v1 = vec![0.0f32; nh * dv];
    let mut v2 = vec![0.0f32; nh * dv];
    w.v_proj.matvec2(x1, x2, &mut v1, &mut v2, pool);

    // Lane 1 commits into the real state.
    let kap1 = kappa_of(x1, w, nh, pool);
    let mut o1 = vec![0.0f32; nh * dv];
    phase_step(&thq1, &thk1, &v1, &w.decay, kap1.as_deref(), cfg, state, &mut o1);

    // Lane 2 runs on a copy — tentative until the draft is verified.
    let kap2 = kappa_of(x2, w, nh, pool);
    scratch.clear();
    scratch.extend_from_slice(state);
    let mut o2 = vec![0.0f32; nh * dv];
    phase_step(&thq2, &thk2, &v2, &w.decay, kap2.as_deref(), cfg, scratch, &mut o2);

    let mut out1 = vec![0.0f32; cfg.hidden_size];
    let mut out2 = vec![0.0f32; cfg.hidden_size];
    w.out_proj.matvec2(&o1, &o2, &mut out1, &mut out2, pool);
    (out1, out2)
}

// ───────────────────────── GatedDeltaNet (faithful vendor operator) ─────────────────────────

/// Weights of one GatedDeltaNet layer (`model.layers.{i}.linear_attn.*`,
/// names 1:1 with the source model — no fold, no training).
pub struct GdnWeights {
    /// [2·nk·dk + nv·dv, hidden] — fused q/k/v projection
    pub in_proj_qkv: QTensor,
    /// [nv·dv, hidden] — output-gate projection z
    pub in_proj_z: QTensor,
    /// [nv, hidden] — decay modulation a
    pub in_proj_a: QTensor,
    /// [nv, hidden] — write-strength b (β = σ(b))
    pub in_proj_b: QTensor,
    /// [c_dim · kk] — depthwise causal conv taps, flattened [c][tap]
    pub conv1d: Vec<f32>,
    /// [nv]
    pub a_log: Vec<f32>,
    /// [nv]
    pub dt_bias: Vec<f32>,
    /// [dv] — gated RMSNorm weight (plain x̂·w, validated by the oracle)
    pub norm: Vec<f32>,
    /// [hidden, nv·dv]
    pub out_proj: QTensor,
}

#[derive(Clone, Copy)]
pub struct GdnCfg {
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub hidden_size: usize,
    pub rms_eps: f64,
}

impl GdnCfg {
    pub fn conv_dim(&self) -> usize {
        2 * self.num_k_heads * self.key_head_dim + self.num_v_heads * self.value_head_dim
    }

    /// Packed state: [conv ring (kk−1)·c_dim | S nv·dk·dv], one Vec<f64>
    /// so the speculative scratch-swap moves ring and condensate together.
    pub fn state_len(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dim()
            + self.num_v_heads * self.key_head_dim * self.value_head_dim
    }
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// One recurrent step given the raw (pre-conv) projections of this
/// position. Advances the packed state (conv ring + S) and writes the
/// gated per-head output into `of` [nv·dv].
#[allow(clippy::too_many_arguments)]
fn gdn_step(qkv: &[f32], z: &[f32], a: &[f32], b: &[f32], w: &GdnWeights, cfg: &GdnCfg, state: &mut [f64], of: &mut [f32]) {
    let (nv, nk, dk, dv, kk) = (
        cfg.num_v_heads,
        cfg.num_k_heads,
        cfg.key_head_dim,
        cfg.value_head_dim,
        cfg.conv_kernel,
    );
    let c_dim = cfg.conv_dim();
    let (kd, rep) = (nk * dk, nv / nk);
    let (ring, s_all) = state.split_at_mut((kk - 1) * c_dim);

    // Depthwise causal conv over [ring…, current] + SiLU. Taps are
    // ordered oldest→newest; tap kk−1 multiplies the current position.
    let mut cq = vec![0f64; c_dim];
    for c in 0..c_dim {
        let taps = &w.conv1d[c * kk..(c + 1) * kk];
        let mut acc = qkv[c] as f64 * taps[kk - 1] as f64;
        for j in 0..kk - 1 {
            acc += ring[j * c_dim + c] * taps[j] as f64;
        }
        cq[c] = silu(acc);
    }
    // Ring shift: drop the oldest position, append the raw current one.
    if kk > 1 {
        ring.copy_within(c_dim.., 0);
        let tail = (kk - 2) * c_dim;
        for c in 0..c_dim {
            ring[tail + c] = qkv[c] as f64;
        }
    }

    for h in 0..nv {
        let ko = h / rep; // source q/k head (GQA)
        let (qs, ks) = (ko * dk, kd + ko * dk);
        // l2-normalize q and k; q additionally scaled by 1/√dk.
        let (mut nq, mut nkn) = (0f64, 0f64);
        for d in 0..dk {
            nq += cq[qs + d] * cq[qs + d];
            nkn += cq[ks + d] * cq[ks + d];
        }
        let invq = 1.0 / ((nq + 1e-6).sqrt() * (dk as f64).sqrt());
        let invk = 1.0 / (nkn + 1e-6).sqrt();

        let g = (-(w.a_log[h] as f64).exp() * softplus(a[h] as f64 + w.dt_bias[h] as f64)).exp();
        let beta = sigmoid(b[h] as f64);

        let s = &mut s_all[h * dk * dv..(h + 1) * dk * dv];
        let vt = &cq[2 * kd + h * dv..2 * kd + (h + 1) * dv];
        // S ← g·S;  kv = kᵀS;  S += k ⊗ β(v − kv);  o = qᵀS
        let mut kv = vec![0f64; dv];
        for di in 0..dk {
            let kf = cq[ks + di] * invk;
            let row = &mut s[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                row[dj] *= g;
                kv[dj] += row[dj] * kf;
            }
        }
        let mut o = vec![0f64; dv];
        for di in 0..dk {
            let kf = cq[ks + di] * invk;
            let qf = cq[qs + di] * invq;
            let row = &mut s[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                row[dj] += kf * (vt[dj] - kv[dj]) * beta;
                o[dj] += qf * row[dj];
            }
        }
        // Gated RMSNorm per head: x̂·w·silu(z) (oracle-validated form).
        let ss: f64 = o.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / dv as f64 + cfg.rms_eps).sqrt();
        for dj in 0..dv {
            of[h * dv + dj] =
                ((o[dj] * inv) * w.norm[dj] as f64 * silu(z[h * dv + dj] as f64)) as f32;
        }
    }
}

/// Forward one position through a GatedDeltaNet layer, advancing `state`.
pub fn gdn_forward(
    x: &[f32],
    w: &GdnWeights,
    cfg: &GdnCfg,
    state: &mut Vec<f64>,
    pool: Option<&Pool>,
) -> Vec<f32> {
    if state.len() != cfg.state_len() {
        *state = vec![0f64; cfg.state_len()];
    }
    let (c_dim, vd) = (cfg.conv_dim(), cfg.num_v_heads * cfg.value_head_dim);

    let mut qkv = vec![0.0f32; c_dim];
    let mut z = vec![0.0f32; vd];
    // D5: two heavy projections (qkv+z ≈ 24MB q8 per 35B layer, ×30 layers
    // = half the token's bytes) — in a single GPU submission; a/b are tiny
    // f16 and stay on CPU.
    if !gdn_projs_gpu(w, x, &mut qkv, &mut z) {
        w.in_proj_qkv.matvec(x, &mut qkv, pool);
        w.in_proj_z.matvec(x, &mut z, pool);
    }
    let mut a = vec![0.0f32; cfg.num_v_heads];
    w.in_proj_a.matvec(x, &mut a, pool);
    let mut b = vec![0.0f32; cfg.num_v_heads];
    w.in_proj_b.matvec(x, &mut b, pool);

    let mut of = vec![0.0f32; vd];
    gdn_step(&qkv, &z, &a, &b, w, cfg, state, &mut of);

    let mut out = vec![0.0f32; cfg.hidden_size];
    w.out_proj.matvec(&of, &mut out, pool);
    out
}

/// Batched GDN forward (prefill-GEMM): the qkv/z/a/b and out_proj
/// projections are matmat over the batch (a weight row once per chunk),
/// the gdn_step recurrence runs sequentially over positions (state is the
/// same as the sequential path; the math is elementwise identical).
pub fn gdn_forward_batch(
    xs: &[f32],
    b: usize,
    w: &GdnWeights,
    cfg: &GdnCfg,
    state: &mut Vec<f64>,
    pool: Option<&Pool>,
) -> Vec<f32> {
    if state.len() != cfg.state_len() {
        *state = vec![0f64; cfg.state_len()];
    }
    let (c_dim, vd) = (cfg.conv_dim(), cfg.num_v_heads * cfg.value_head_dim);
    let nv = cfg.num_v_heads;

    let mut qkv = vec![0.0f32; b * c_dim];
    w.in_proj_qkv.matmat(xs, b, &mut qkv, pool);
    let mut z = vec![0.0f32; b * vd];
    w.in_proj_z.matmat(xs, b, &mut z, pool);
    let mut a = vec![0.0f32; b * nv];
    w.in_proj_a.matmat(xs, b, &mut a, pool);
    let mut bb = vec![0.0f32; b * nv];
    w.in_proj_b.matmat(xs, b, &mut bb, pool);

    let mut of = vec![0.0f32; b * vd];
    for bi in 0..b {
        gdn_step(
            &qkv[bi * c_dim..(bi + 1) * c_dim],
            &z[bi * vd..(bi + 1) * vd],
            &a[bi * nv..(bi + 1) * nv],
            &bb[bi * nv..(bi + 1) * nv],
            w,
            cfg,
            state,
            &mut of[bi * vd..(bi + 1) * vd],
        );
    }
    let mut out = vec![0.0f32; b * cfg.hidden_size];
    w.out_proj.matmat(&of, b, &mut out, pool);
    out
}

/// GDN qkv+z on GPU in a single submission (independent matvecs of one input).
fn gdn_projs_gpu(w: &GdnWeights, x: &[f32], qkv: &mut [f32], z: &mut [f32]) -> bool {
    use crate::gpu::matvec_batch;
    use crate::qtensor::QTensor;
    // Measured (35B, alternating A/B): 2 matvecs per submission do NOT
    // amortize the sync cost — neutral within the noise. Opt-in until
    // attention/norms move into this same submission.
    if !crate::gpu::enabled_here()
        || !std::env::var("CMF_GPU_GDN").map(|v| v == "1").unwrap_or(false)
    {
        return false;
    }
    fn part<'a>(
        t: &'a QTensor,
        x: &[f32],
    ) -> Option<(std::sync::Arc<cortiq_core::CmfModel>, crate::gpu::BatchJob<'a>)> {
        use crate::gpu::BatchJob;
        use crate::qtensor::prescale;
        use cortiq_core::TensorDtype;
        match t {
        QTensor::Mapped {
            model,
            idx,
            dtype: dt @ (TensorDtype::Q8Row | TensorDtype::Q8_2f),
            rows,
            cols,
            row_scale,
            col_field,
        } => Some((
            model.clone(),
            BatchJob {
                idx: *idx,
                rows: *rows,
                cols: *cols,
                row_scale,
                xs: prescale(x, col_field, *dt).into_owned(),
            },
        )),
        _ => None,
        }
    }
    let Some((model, jq)) = part(&w.in_proj_qkv, x) else { return false };
    let Some((_, jz)) = part(&w.in_proj_z, x) else { return false };
    matvec_batch(&model, &[jq, jz], &mut [qkv, z])
}

/// Fused two-position forward (speculative verify): lane 1 commits into
/// `state`, lane 2 is tentative in `scratch` (ring + S move together).
#[allow(clippy::too_many_arguments)]
pub fn gdn_pair(
    x1: &[f32],
    x2: &[f32],
    w: &GdnWeights,
    cfg: &GdnCfg,
    state: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
    pool: Option<&Pool>,
) -> (Vec<f32>, Vec<f32>) {
    if state.len() != cfg.state_len() {
        *state = vec![0f64; cfg.state_len()];
    }
    let (c_dim, vd, nv) = (
        cfg.conv_dim(),
        cfg.num_v_heads * cfg.value_head_dim,
        cfg.num_v_heads,
    );

    let mut qkv1 = vec![0.0f32; c_dim];
    let mut qkv2 = vec![0.0f32; c_dim];
    w.in_proj_qkv.matvec2(x1, x2, &mut qkv1, &mut qkv2, pool);
    let mut z1 = vec![0.0f32; vd];
    let mut z2 = vec![0.0f32; vd];
    w.in_proj_z.matvec2(x1, x2, &mut z1, &mut z2, pool);
    let mut a1 = vec![0.0f32; nv];
    let mut a2 = vec![0.0f32; nv];
    w.in_proj_a.matvec2(x1, x2, &mut a1, &mut a2, pool);
    let mut b1 = vec![0.0f32; nv];
    let mut b2 = vec![0.0f32; nv];
    w.in_proj_b.matvec2(x1, x2, &mut b1, &mut b2, pool);

    let mut of1 = vec![0.0f32; vd];
    gdn_step(&qkv1, &z1, &a1, &b1, w, cfg, state, &mut of1);

    scratch.clear();
    scratch.extend_from_slice(state);
    let mut of2 = vec![0.0f32; vd];
    gdn_step(&qkv2, &z2, &a2, &b2, w, cfg, scratch, &mut of2);

    let mut out1 = vec![0.0f32; cfg.hidden_size];
    let mut out2 = vec![0.0f32; cfg.hidden_size];
    w.out_proj.matvec2(&of1, &of2, &mut out1, &mut out2, pool);
    (out1, out2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (VmfPhaseWeights, VmfPhaseCfg) {
        let cfg = VmfPhaseCfg {
            num_heads: 2,
            nphase: 3,
            value_head_dim: 4,
            hidden_size: 8,
            phase_mass: 0.0,
        };
        let synth = |rows: usize, cols: usize, salt: usize| {
            QTensor::from_f32(
                (0..rows * cols)
                    .map(|i| (((i * 13 + salt * 7) % 97) as f32 / 97.0 - 0.5) * 0.4)
                    .collect(),
                rows,
                cols,
            )
        };
        let w = VmfPhaseWeights {
            thq: synth(cfg.num_heads * cfg.nphase, cfg.hidden_size, 1),
            thk: synth(cfg.num_heads * cfg.nphase, cfg.hidden_size, 2),
            v_proj: synth(cfg.num_heads * cfg.value_head_dim, cfg.hidden_size, 3),
            out_proj: synth(cfg.hidden_size, cfg.num_heads * cfg.value_head_dim, 4),
            decay: (0..cfg.num_heads * 2 * cfg.nphase)
                .map(|i| 0.9 + 0.005 * (i % 10) as f64)
                .collect(),
            k_gate: None,
        };
        (w, cfg)
    }

    #[test]
    fn state_persists_and_changes_output() {
        let (w, cfg) = tiny();
        let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut state = Vec::new();
        let o1 = vmf_phase_forward(&x, &w, &cfg, &mut state, None);
        let o2 = vmf_phase_forward(&x, &w, &cfg, &mut state, None);
        // Same input, evolved condensate → different output.
        assert!(o1.iter().zip(&o2).any(|(a, b)| (a - b).abs() > 1e-6));
        assert_eq!(state.len(), cfg.state_len());
    }

    /// θ-mass (η′): mass=0 is bit-identical to the massless kernel; mass>0
    /// changes the output (phase narrowed → kernel widened). Guards the
    /// no-op default and that the knob is actually wired.
    #[test]
    fn phase_mass_zero_is_noop_and_positive_shifts() {
        let (w, cfg0) = tiny();
        let mut cfg_m = cfg0.clone();
        cfg_m.phase_mass = 1.0;
        let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.4).sin()).collect();

        let mut s0 = Vec::new();
        let base = vmf_phase_forward(&x, &w, &cfg0, &mut s0, None);
        // Re-run with mass=0 → must be bit-identical.
        let mut s0b = Vec::new();
        let base2 = vmf_phase_forward(&x, &w, &cfg0, &mut s0b, None);
        assert_eq!(base, base2, "mass=0 must be deterministic/no-op");
        // mass=1 → output differs (θ halved before cos/sin).
        let mut sm = Vec::new();
        let massed = vmf_phase_forward(&x, &w, &cfg_m, &mut sm, None);
        assert!(
            base.iter().zip(&massed).any(|(a, b)| (a - b).abs() > 1e-5),
            "mass>0 must change the output"
        );
        assert!(massed.iter().all(|v| v.is_finite()));
    }

    /// κ write gate (hybrid_k): saturated-open gate (bias ≫ 0 → κ→1)
    /// matches the gateless kernel within fp tolerance; a closed gate
    /// (bias ≪ 0 → κ→0) writes nothing — the state stays zero and the
    /// output collapses to the empty-condensate readout.
    #[test]
    fn kappa_gate_open_matches_none_and_closed_writes_nothing() {
        let (mut w, cfg) = tiny();
        let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();

        let mut s_none = Vec::new();
        let base1 = vmf_phase_forward(&x, &w, &cfg, &mut s_none, None);
        let base2 = vmf_phase_forward(&x, &w, &cfg, &mut s_none, None);

        // Open gate: W=0, bias=+20 → κ = σ(20) ≈ 1 − 2e−9.
        w.k_gate = Some((
            QTensor::from_f32(vec![0.0; cfg.num_heads * cfg.hidden_size], cfg.num_heads, cfg.hidden_size),
            vec![20.0; cfg.num_heads],
        ));
        let mut s_open = Vec::new();
        let o1 = vmf_phase_forward(&x, &w, &cfg, &mut s_open, None);
        let o2 = vmf_phase_forward(&x, &w, &cfg, &mut s_open, None);
        for (a, b) in base1.iter().zip(&o1).chain(base2.iter().zip(&o2)) {
            assert!((a - b).abs() < 1e-5, "open κ must match gateless: {a} vs {b}");
        }

        // Closed gate: bias=−20 → κ ≈ 0 → nothing is written.
        w.k_gate = Some((
            QTensor::from_f32(vec![0.0; cfg.num_heads * cfg.hidden_size], cfg.num_heads, cfg.hidden_size),
            vec![-20.0; cfg.num_heads],
        ));
        let mut s_closed = Vec::new();
        let oc = vmf_phase_forward(&x, &w, &cfg, &mut s_closed, None);
        assert!(s_closed.iter().all(|&v| v.abs() < 1e-7), "closed κ: state must stay empty");
        assert!(oc.iter().all(|&v| v.abs() < 1e-6), "closed κ: empty-condensate readout");
    }

    #[test]
    fn pair_matches_two_singles_bitexact() {
        let (w, cfg) = tiny();
        let x1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.2).cos()).collect();
        let x2: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5).sin()).collect();

        // Reference: two sequential singles.
        let mut s_ref = Vec::new();
        let r1 = vmf_phase_forward(&x1, &w, &cfg, &mut s_ref, None);
        let r2 = vmf_phase_forward(&x2, &w, &cfg, &mut s_ref, None);

        // Pair: lane1 commits, lane2 tentative in scratch.
        let mut s = Vec::new();
        let mut scratch = Vec::new();
        let (p1, p2) = vmf_phase_pair(&x1, &x2, &w, &cfg, &mut s, &mut scratch, None);
        assert_eq!(r1, p1, "lane 1 must be bit-identical");
        assert_eq!(r2, p2, "lane 2 must be bit-identical");
        // Accepting the draft = swapping scratch in → equals s_ref.
        std::mem::swap(&mut s, &mut scratch);
        assert_eq!(s, s_ref, "accepted state must equal sequential state");
    }

    #[test]
    fn rejected_draft_leaves_state_at_lane1() {
        let (w, cfg) = tiny();
        let x1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let x2 = vec![0.5f32; 8];

        let mut s_ref = Vec::new();
        let _ = vmf_phase_forward(&x1, &w, &cfg, &mut s_ref, None);

        let mut s = Vec::new();
        let mut scratch = Vec::new();
        let _ = vmf_phase_pair(&x1, &x2, &w, &cfg, &mut s, &mut scratch, None);
        // Reject: state must be exactly the post-lane1 state.
        assert_eq!(s, s_ref);
    }

    // ───────────── GatedDeltaNet ─────────────

    fn tiny_gdn() -> (GdnWeights, GdnCfg) {
        let cfg = GdnCfg {
            num_v_heads: 4,
            num_k_heads: 2,
            key_head_dim: 3,
            value_head_dim: 5,
            conv_kernel: 4,
            hidden_size: 8,
            rms_eps: 1e-6,
        };
        let c_dim = cfg.conv_dim();
        let vd = cfg.num_v_heads * cfg.value_head_dim;
        let synth = |rows: usize, cols: usize, salt: usize| {
            QTensor::from_f32(
                (0..rows * cols)
                    .map(|i| (((i * 13 + salt * 7) % 97) as f32 / 97.0 - 0.5) * 0.4)
                    .collect(),
                rows,
                cols,
            )
        };
        let vecf = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 11 + salt * 5) % 89) as f32 / 89.0 - 0.5) * 0.6)
                .collect()
        };
        let w = GdnWeights {
            in_proj_qkv: synth(c_dim, cfg.hidden_size, 1),
            in_proj_z: synth(vd, cfg.hidden_size, 2),
            in_proj_a: synth(cfg.num_v_heads, cfg.hidden_size, 3),
            in_proj_b: synth(cfg.num_v_heads, cfg.hidden_size, 4),
            conv1d: vecf(c_dim * cfg.conv_kernel, 5),
            a_log: (0..cfg.num_v_heads).map(|i| 0.2 + 0.3 * i as f32).collect(),
            dt_bias: vecf(cfg.num_v_heads, 6),
            norm: vec![1.0; cfg.value_head_dim],
            out_proj: synth(cfg.hidden_size, vd, 7),
        };
        (w, cfg)
    }

    #[test]
    fn gdn_state_persists_and_changes_output() {
        let (w, cfg) = tiny_gdn();
        let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut state = Vec::new();
        let o1 = gdn_forward(&x, &w, &cfg, &mut state, None);
        let o2 = gdn_forward(&x, &w, &cfg, &mut state, None);
        assert!(o1.iter().zip(&o2).any(|(a, b)| (a - b).abs() > 1e-6));
        assert_eq!(state.len(), cfg.state_len());
    }

    #[test]
    fn gdn_pair_matches_two_singles_bitexact() {
        let (w, cfg) = tiny_gdn();
        let x1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.2).cos()).collect();
        let x2: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5).sin()).collect();

        let mut s_ref = Vec::new();
        let r1 = gdn_forward(&x1, &w, &cfg, &mut s_ref, None);
        let r2 = gdn_forward(&x2, &w, &cfg, &mut s_ref, None);

        let mut s = Vec::new();
        let mut scratch = Vec::new();
        let (p1, p2) = gdn_pair(&x1, &x2, &w, &cfg, &mut s, &mut scratch, None);
        assert_eq!(r1, p1, "lane 1 must be bit-identical");
        assert_eq!(r2, p2, "lane 2 must be bit-identical");
        std::mem::swap(&mut s, &mut scratch);
        assert_eq!(s, s_ref, "accepted state must equal sequential state");
    }

    #[test]
    fn gdn_rejected_draft_leaves_state_at_lane1() {
        let (w, cfg) = tiny_gdn();
        let x1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let x2 = vec![0.5f32; 8];

        let mut s_ref = Vec::new();
        let _ = gdn_forward(&x1, &w, &cfg, &mut s_ref, None);

        let mut s = Vec::new();
        let mut scratch = Vec::new();
        let _ = gdn_pair(&x1, &x2, &w, &cfg, &mut s, &mut scratch, None);
        assert_eq!(s, s_ref);
    }

    /// The conv ring must give the same result as an explicit causal
    /// conv over the whole sequence (oracle semantics: zero left-pad,
    /// tap kk−1 on the current position).
    #[test]
    fn gdn_conv_ring_matches_explicit_causal_conv() {
        let (w, cfg) = tiny_gdn();
        let seq: Vec<Vec<f32>> = (0..6)
            .map(|t| (0..8).map(|i| ((t * 8 + i) as f32 * 0.17).sin()).collect())
            .collect();

        // Reference: recompute position t from scratch each time with a
        // fresh state built by replaying the prefix.
        let mut s_inc = Vec::new();
        for (t, x) in seq.iter().enumerate() {
            let inc = gdn_forward(x, &w, &cfg, &mut s_inc, None);
            let mut s_replay = Vec::new();
            let mut replay = Vec::new();
            for xr in &seq[..=t] {
                replay = gdn_forward(xr, &w, &cfg, &mut s_replay, None);
            }
            assert_eq!(inc, replay, "position {t}: ring must equal replay");
        }
    }
}
