//! CPU reference ops (f64 accumulate) — the oracle every Metal kernel is
//! checked against, and the gradcheck substrate for the hand-rolled
//! backwards. Slow on purpose: clarity over speed.

/// C[M,N] = alpha·op(A)·op(B) + beta·C, same layout contract as the
/// Metal `gemm_f32` (see `metal::Op`).
#[allow(clippy::too_many_arguments)]
pub fn gemm_ref(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                let av = if ta { a[kk * lda + i] } else { a[i * lda + kk] } as f64;
                let bv = if tb { b[j * ldb + kk] } else { b[kk * ldb + j] } as f64;
                s += av * bv;
            }
            let idx = i * ldc + j;
            let prev = if beta != 0.0 {
                beta as f64 * c[idx] as f64
            } else {
                0.0
            };
            c[idx] = (alpha as f64 * s + prev) as f32;
        }
    }
}

/// Deterministic pseudo-random floats in [-1, 1) (splitmix64) — test and
/// init helper, no rand crate.
pub fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Causal four-position mean of per-expert routing scores.
///
/// `scores` is row-major `[rows, experts]`; rows are grouped into `batch`
/// sequences of length `seq`.  For position `t`, only the current score and
/// up to the preceding three positions in the same sequence contribute.  A
/// single-position sequence is therefore an identity transform.  The helper
/// is the CPU witness for the optional Metal routing kernel; it deliberately
/// accumulates in f64 before converting back to f32.
pub fn causal_k4_score_smooth(
    scores: &[f32],
    batch: usize,
    seq: usize,
    experts: usize,
) -> Vec<f32> {
    assert!(seq > 0, "causal score smoothing requires seq > 0");
    let rows = batch
        .checked_mul(seq)
        .expect("causal score smoothing rows overflow");
    assert_eq!(scores.len(), rows * experts, "score matrix shape mismatch");
    let mut out = vec![0.0f32; scores.len()];
    for b in 0..batch {
        for t in 0..seq {
            let row = b * seq + t;
            let first = t.saturating_sub(3);
            let count = (t - first + 1) as f64;
            for e in 0..experts {
                let mut sum = 0.0f64;
                for p in first..=t {
                    sum += scores[(b * seq + p) * experts + e] as f64;
                }
                out[row * experts + e] = (sum / count) as f32;
            }
        }
    }
    out
}

/// Deterministic top-1 over already-smoothed routing scores.  Ties retain
/// the lowest expert index, matching the Metal route kernels' strict `<`
/// comparison.
pub fn causal_k4_argmin(
    scores: &[f32],
    bias: &[f32],
    batch: usize,
    seq: usize,
    experts: usize,
) -> Vec<usize> {
    assert!(experts > 0, "argmin requires at least one expert");
    assert_eq!(bias.len(), experts, "bias shape mismatch");
    let smoothed = causal_k4_score_smooth(scores, batch, seq, experts);
    let mut out = vec![0usize; batch * seq];
    for row in 0..out.len() {
        let mut best = 0usize;
        let mut best_score = smoothed[row * experts] - bias[0];
        for e in 1..experts {
            let score = smoothed[row * experts + e] - bias[e];
            if score < best_score {
                best_score = score;
                best = e;
            }
        }
        out[row] = best;
    }
    out
}

/// Result of the optional ambiguity-triggered resonance router.
///
/// `assign` is the deterministic top-1 expert.  `runner_up` is the
/// deterministic second-best expert, or `usize::MAX` when the fixed margin
/// did not trigger (or there is only one expert).  `runner_weight` is the
/// residual weight assigned to the runner-up; the top-1 residual keeps the
/// complementary weight.  The current scaffold deliberately uses a fixed
/// 50/50 blend whenever the fallback is active: this adds no trainable gate
/// or persistent state and keeps the piecewise routing decision explicit.
#[derive(Clone, Debug, PartialEq)]
pub struct Top2Route {
    pub assign: Vec<usize>,
    pub runner_up: Vec<usize>,
    pub margin: Vec<f32>,
    pub runner_weight: Vec<f32>,
    pub fallback_count: usize,
}

impl Top2Route {
    /// Fraction of rows on which the runner-up was evaluated/blended.
    pub fn fallback_rate(&self) -> f32 {
        if self.assign.is_empty() {
            0.0
        } else {
            self.fallback_count as f32 / self.assign.len() as f32
        }
    }
}

/// Deterministic top-2 selection from bias-adjusted resonance costs.
///
/// `scores` is row-major `[rows, experts]`; `bias` is `[experts]`.  The
/// runner-up is considered only when `0 <= margin < threshold`, where
/// `margin = score₂ - score₁`.  A non-positive threshold therefore preserves
/// the exact top-1 route while still exposing the winning margin for
/// diagnostics.  Ties retain the lowest expert index (`<`, never `<=`).
/// `batch` and `seq` are accepted to mirror the causal helper and to make the
/// row/sequence contract explicit; raw routing itself is position-local.
pub fn top2_argmin(
    scores: &[f32],
    bias: &[f32],
    batch: usize,
    seq: usize,
    experts: usize,
    threshold: f32,
) -> Top2Route {
    assert!(seq > 0, "top2 routing requires seq > 0");
    let rows = batch.checked_mul(seq).expect("top2 routing rows overflow");
    assert!(experts > 0, "top2 routing requires at least one expert");
    assert_eq!(scores.len(), rows * experts, "score matrix shape mismatch");
    assert_eq!(bias.len(), experts, "bias shape mismatch");
    top2_from_adjusted_scores(scores, bias, rows, experts, threshold)
}

/// Causal-k4 variant of [`top2_argmin`].  Scores are averaged over the
/// current and preceding three positions in each sequence before selecting
/// the top-1/runner-up pair.  The helper is the CPU witness for the optional
/// Metal route kernel; no future row contributes to a decision.
pub fn causal_k4_top2_argmin(
    scores: &[f32],
    bias: &[f32],
    batch: usize,
    seq: usize,
    experts: usize,
    threshold: f32,
) -> Top2Route {
    let smoothed = causal_k4_score_smooth(scores, batch, seq, experts);
    top2_from_adjusted_scores(&smoothed, bias, batch * seq, experts, threshold)
}

fn top2_from_adjusted_scores(
    scores: &[f32],
    bias: &[f32],
    rows: usize,
    experts: usize,
    threshold: f32,
) -> Top2Route {
    let mut assign = vec![0usize; rows];
    let mut runner_up = vec![usize::MAX; rows];
    let mut margin = vec![f32::INFINITY; rows];
    let mut runner_weight = vec![0.0f32; rows];
    let mut fallback_count = 0usize;
    for row in 0..rows {
        let base = row * experts;
        let mut best = usize::MAX;
        let mut second = usize::MAX;
        let mut best_score = f32::INFINITY;
        let mut second_score = f32::INFINITY;
        for e in 0..experts {
            let score = scores[base + e] - bias[e];
            // Strict comparisons provide deterministic lowest-index ties.
            if best == usize::MAX || score < best_score {
                second = best;
                second_score = best_score;
                best = e;
                best_score = score;
            } else if second == usize::MAX || score < second_score {
                second = e;
                second_score = score;
            }
        }
        assign[row] = best;
        if second != usize::MAX {
            let m = second_score - best_score;
            margin[row] = m;
            if threshold.is_finite() && threshold > 0.0 && m < threshold {
                runner_up[row] = second;
                runner_weight[row] = 0.5;
                fallback_count += 1;
            }
        }
    }
    Top2Route {
        assign,
        runner_up,
        margin,
        runner_weight,
        fallback_count,
    }
}

/// Blend only routed-expert residual rows according to [`Top2Route`]'s
/// runner weights.  `primary` and `runner` are row-major `[rows, hidden]`;
/// the shared expert residual is intentionally not an argument and therefore
/// remains unchanged by this helper.  A zero runner weight is an exact copy
/// of the top-1 residual, which is the disabled/high-margin fast path.
pub fn blend_top2_residual(
    primary: &[f32],
    runner: &[f32],
    runner_weight: &[f32],
    hidden: usize,
) -> Vec<f32> {
    assert!(hidden > 0, "residual hidden width must be positive");
    assert_eq!(primary.len(), runner.len(), "residual shape mismatch");
    assert_eq!(
        primary.len() % hidden,
        0,
        "residual rows must divide hidden"
    );
    let rows = primary.len() / hidden;
    assert_eq!(runner_weight.len(), rows, "runner weight row mismatch");
    let mut out = vec![0.0f32; primary.len()];
    for row in 0..rows {
        let w = runner_weight[row].clamp(0.0, 1.0);
        let inv = 1.0 - w;
        for j in 0..hidden {
            let i = row * hidden + j;
            out[i] = inv * primary[i] + w * runner[i];
        }
    }
    out
}

// ---------------------------------------------------------------------
// hybrid_k mixer (the runtime's vmf_phase core + κ write gate,
// linear_core.rs::phase_step) — CPU reference forward/backward.
//
// Per head (h dropped), position t, feature f ∈ [0, p2), p2 = 2·nph:
//   φ(θ)[i] = cos θ_i, φ(θ)[nph+i] = sin θ_i
//   S_t = diag(γ)·S_{t−1} + κ_t·φk_t ⊗ v_t         S: [p2, dv]
//   o_t = φq_tᵀ·S_t                                   o: [dv]
// γ_f is a FIXED per-feature decay (log-spaced horizons, no gradient).
//
// Closed form used by the backward (and by the chunked GPU kernels):
//   A[t,s] = Σ_f φq_t[f]·φk_s[f]·γ_f^{t−s}   (s ≤ t)
//   o_t    = Σ_{s≤t} A[t,s]·κ_s·v_s
// Layouts (all row-major, matching the projection GEMM outputs):
//   thq, thk: [B·T, nh·nph]   v: [B·T, nh·dv]   kappa: [B·T, nh]
//   o: [B·T, nh·dv]           decay: [nh·p2]
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct HkDims {
    pub b: usize,
    pub t: usize,
    pub nh: usize,
    pub nph: usize,
    pub dv: usize,
}

impl HkDims {
    pub fn p2(&self) -> usize {
        2 * self.nph
    }
}

/// Fixed decays γ_f = exp(−1/H_f), H log-spaced in [h_min, h_max] over the
/// nph phase pairs (cos and sin of one θ_i share a horizon), same grid
/// for every head. Returns [nh·p2].
pub fn hk_decay_grid(nh: usize, nph: usize, h_min: f64, h_max: f64) -> Vec<f32> {
    let mut d = vec![0.0f32; nh * 2 * nph];
    for h in 0..nh {
        for i in 0..nph {
            let frac = if nph > 1 {
                i as f64 / (nph - 1) as f64
            } else {
                0.0
            };
            let horizon = h_min * (h_max / h_min).powf(frac);
            let g = (-1.0 / horizon).exp() as f32;
            d[h * 2 * nph + i] = g;
            d[h * 2 * nph + nph + i] = g;
        }
    }
    d
}

#[inline]
fn phi(theta: &[f64], nph: usize, f: usize) -> f64 {
    if f < nph {
        theta[f].cos()
    } else {
        theta[f - nph].sin()
    }
}

/// Reference forward by the literal recurrence (f64). Returns o.
pub fn hk_ref_fwd(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
) -> Vec<f64> {
    let (p2, nph, dv, nh, t_len) = (d.p2(), d.nph, d.dv, d.nh, d.t);
    let mut o = vec![0.0f64; d.b * t_len * nh * dv];
    for b in 0..d.b {
        for h in 0..nh {
            let mut s = vec![0.0f64; p2 * dv];
            for t in 0..t_len {
                let row = b * t_len + t;
                let tq = &thq[row * nh * nph + h * nph..row * nh * nph + (h + 1) * nph];
                let tk = &thk[row * nh * nph + h * nph..row * nh * nph + (h + 1) * nph];
                let vt = &v[row * nh * dv + h * dv..row * nh * dv + (h + 1) * dv];
                let kap = kappa[row * nh + h];
                let ot = &mut o[row * nh * dv + h * dv..row * nh * dv + (h + 1) * dv];
                for f in 0..p2 {
                    let g = decay[h * p2 + f];
                    let fk = phi(tk, nph, f) * kap;
                    let fq = phi(tq, nph, f);
                    for dd in 0..dv {
                        let cell = g * s[f * dv + dd] + fk * vt[dd];
                        s[f * dv + dd] = cell;
                        ot[dd] += fq * cell;
                    }
                }
            }
        }
    }
    o
}

/// `hk_ref_fwd` from a carried initial state `s0` `[B, nh, p2, dv]`
/// (state carry-over across windows, plan S6b).
#[allow(clippy::too_many_arguments)]
pub fn hk_ref_fwd_s0(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    s0: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (p2, nph, dv, nh, t_len) = (d.p2(), d.nph, d.dv, d.nh, d.t);
    let mut o = vec![0.0f64; d.b * t_len * nh * dv];
    let mut s_end = vec![0.0f64; d.b * nh * p2 * dv];
    for b in 0..d.b {
        for h in 0..nh {
            let bh = b * nh + h;
            let mut s = s0[bh * p2 * dv..(bh + 1) * p2 * dv].to_vec();
            for t in 0..t_len {
                let row = b * t_len + t;
                let tq = &thq[row * nh * nph + h * nph..row * nh * nph + (h + 1) * nph];
                let tk = &thk[row * nh * nph + h * nph..row * nh * nph + (h + 1) * nph];
                let vt = &v[row * nh * dv + h * dv..row * nh * dv + (h + 1) * dv];
                let kap = kappa[row * nh + h];
                let ot = &mut o[row * nh * dv + h * dv..row * nh * dv + (h + 1) * dv];
                for f in 0..p2 {
                    let g = decay[h * p2 + f];
                    let fk = phi(tk, nph, f) * kap;
                    let fq = phi(tq, nph, f);
                    for dd in 0..dv {
                        let cell = g * s[f * dv + dd] + fk * vt[dd];
                        s[f * dv + dd] = cell;
                        ot[dd] += fq * cell;
                    }
                }
            }
            s_end[bh * p2 * dv..(bh + 1) * p2 * dv].copy_from_slice(&s);
        }
    }
    (o, s_end)
}

/// `hk_ref_bwd` with a carried (constant) initial state: only dθq gains the
/// term through `o_t ∋ Σ_f φq_t[f]·γ_f^{t+1}·S_0[f]`; dθk/dv/dκ are unchanged.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn hk_ref_bwd_s0(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    s0: &[f64],
    dout: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut dthq, dthk, dvv, dkap) = hk_ref_bwd(d, thq, thk, v, kappa, decay, dout);
    let (p2, nph, dv, nh, t_len) = (d.p2(), d.nph, d.dv, d.nh, d.t);
    for b in 0..d.b {
        for h in 0..nh {
            let bh = b * nh + h;
            for t in 0..t_len {
                let row = b * t_len + t;
                let dout_t = &dout[row * nh * dv + h * dv..row * nh * dv + (h + 1) * dv];
                let mut e = vec![0.0f64; p2];
                for f in 0..p2 {
                    let g = decay[h * p2 + f].powi(t as i32 + 1);
                    let mut acc = 0.0;
                    for dd in 0..dv {
                        acc += dout_t[dd] * s0[bh * p2 * dv + f * dv + dd];
                    }
                    e[f] = g * acc;
                }
                for i in 0..nph {
                    let tq = thq[row * nh * nph + h * nph + i];
                    dthq[row * nh * nph + h * nph + i] += -tq.sin() * e[i] + tq.cos() * e[nph + i];
                }
            }
        }
    }
    (dthq, dthk, dvv, dkap)
}

/// Reference backward from the closed form (f64, O(T²) per head — a test
/// oracle, not a trainer path). Returns (dthq, dthk, dv, dkappa).
#[allow(clippy::type_complexity)]
pub fn hk_ref_bwd(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    dout: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (p2, nph, dv, nh, t_len) = (d.p2(), d.nph, d.dv, d.nh, d.t);
    let mut dthq = vec![0.0f64; thq.len()];
    let mut dthk = vec![0.0f64; thk.len()];
    let mut dvv = vec![0.0f64; v.len()];
    let mut dkap = vec![0.0f64; kappa.len()];
    for b in 0..d.b {
        for h in 0..nh {
            let idx_th = |t: usize| (b * t_len + t) * nh * nph + h * nph;
            let idx_v = |t: usize| (b * t_len + t) * nh * dv + h * dv;
            let idx_k = |t: usize| (b * t_len + t) * nh + h;
            // φq, φk tables [T][p2]
            let mut fq = vec![0.0f64; t_len * p2];
            let mut fk = vec![0.0f64; t_len * p2];
            for t in 0..t_len {
                for f in 0..p2 {
                    fq[t * p2 + f] = phi(&thq[idx_th(t)..idx_th(t) + nph], nph, f);
                    fk[t * p2 + f] = phi(&thk[idx_th(t)..idx_th(t) + nph], nph, f);
                }
            }
            // A[t,s] and dA[t,s] = Σ_d do_t[d]·κ_s·v_s[d]
            let mut a = vec![0.0f64; t_len * t_len];
            let mut da = vec![0.0f64; t_len * t_len];
            for t in 0..t_len {
                for s in 0..=t {
                    let mut acc = 0.0;
                    for f in 0..p2 {
                        acc += fq[t * p2 + f]
                            * fk[s * p2 + f]
                            * decay[h * p2 + f].powi((t - s) as i32);
                    }
                    a[t * t_len + s] = acc;
                    let mut dacc = 0.0;
                    for dd in 0..dv {
                        dacc += dout[idx_v(t) + dd] * kappa[idx_k(s)] * v[idx_v(s) + dd];
                    }
                    da[t * t_len + s] = dacc;
                }
            }
            // d(κv)_s = Σ_{t≥s} A[t,s]·do_t  → dv, dκ
            for s in 0..t_len {
                let mut dkv = vec![0.0f64; dv];
                for t in s..t_len {
                    let av = a[t * t_len + s];
                    for dd in 0..dv {
                        dkv[dd] += av * dout[idx_v(t) + dd];
                    }
                }
                let kap = kappa[idx_k(s)];
                let mut dk = 0.0;
                for dd in 0..dv {
                    dvv[idx_v(s) + dd] += kap * dkv[dd];
                    dk += dkv[dd] * v[idx_v(s) + dd];
                }
                dkap[idx_k(s)] += dk;
            }
            // dφq_t[f] = Σ_{s≤t} dA[t,s]·φk_s[f]·γ^{t−s};  dφk_s[f] = Σ_{t≥s} dA[t,s]·φq_t[f]·γ^{t−s}
            let mut dfq = vec![0.0f64; t_len * p2];
            let mut dfk = vec![0.0f64; t_len * p2];
            for t in 0..t_len {
                for s in 0..=t {
                    let dav = da[t * t_len + s];
                    for f in 0..p2 {
                        let g = decay[h * p2 + f].powi((t - s) as i32);
                        dfq[t * p2 + f] += dav * fk[s * p2 + f] * g;
                        dfk[s * p2 + f] += dav * fq[t * p2 + f] * g;
                    }
                }
            }
            // chain through φ: dθ_i = −sin θ_i·dφ[i] + cos θ_i·dφ[nph+i]
            for t in 0..t_len {
                for i in 0..nph {
                    let tq = thq[idx_th(t) + i];
                    let tk = thk[idx_th(t) + i];
                    dthq[idx_th(t) + i] +=
                        -tq.sin() * dfq[t * p2 + i] + tq.cos() * dfq[t * p2 + nph + i];
                    dthk[idx_th(t) + i] +=
                        -tk.sin() * dfk[t * p2 + i] + tk.cos() * dfk[t * p2 + nph + i];
                }
            }
        }
    }
    (dthq, dthk, dvv, dkap)
}

/// Reference gradient w.r.t. the per-(head, feature) decays γ (closed form,
/// f64): dγ_f = Σ_{t>s} dA[t,s]·φq_t[f]·φk_s[f]·(t−s)·γ_f^{t−s−1}.
/// Returns [nh·p2].
pub fn hk_ref_dgamma(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    dout: &[f64],
) -> Vec<f64> {
    let (p2, nph, dv, nh, t_len) = (d.p2(), d.nph, d.dv, d.nh, d.t);
    let mut dg = vec![0.0f64; nh * p2];
    for b in 0..d.b {
        for h in 0..nh {
            let idx_th = |t: usize| (b * t_len + t) * nh * nph + h * nph;
            let idx_v = |t: usize| (b * t_len + t) * nh * dv + h * dv;
            let idx_k = |t: usize| (b * t_len + t) * nh + h;
            let mut fq = vec![0.0f64; t_len * p2];
            let mut fk = vec![0.0f64; t_len * p2];
            for t in 0..t_len {
                for f in 0..p2 {
                    fq[t * p2 + f] = phi(&thq[idx_th(t)..idx_th(t) + nph], nph, f);
                    fk[t * p2 + f] = phi(&thk[idx_th(t)..idx_th(t) + nph], nph, f);
                }
            }
            for t in 0..t_len {
                for s in 0..t {
                    let mut da = 0.0;
                    for dd in 0..dv {
                        da += dout[idx_v(t) + dd] * kappa[idx_k(s)] * v[idx_v(s) + dd];
                    }
                    let delta = (t - s) as f64;
                    for f in 0..p2 {
                        let g = decay[h * p2 + f];
                        dg[h * p2 + f] += da
                            * fq[t * p2 + f]
                            * fk[s * p2 + f]
                            * delta
                            * g.powi((t - s - 1) as i32);
                    }
                }
            }
        }
    }
    dg
}

// ---------------------------------------------------------------------
// Parameter-neutral in-place Phase-Delta oracle.
//
// The phase features are deliberately normalized by 1/sqrt(nphase), while
// the recurrent state remains the legacy [2*nphase, dv] matrix.  This is a
// clean-room f64 implementation of the accepted write law and its exact
// reverse recurrence; Metal has a separate f32 implementation in
// `shaders.metal`.
// ---------------------------------------------------------------------

/// Normalized phase feature `[cos(theta), sin(theta)] / sqrt(nphase)`.
#[inline]
fn phase_delta_phi(theta: &[f64], nph: usize, f: usize) -> f64 {
    let c = 1.0 / (nph as f64).sqrt();
    if f < nph {
        c * theta[f].cos()
    } else {
        c * theta[f - nph].sin()
    }
}

/// Literal Phase-Delta forward recurrence (f64).  The initial recurrent
/// state is zero and the returned output is after the current-token write.
pub fn phase_delta_ref_fwd(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
) -> Vec<f64> {
    let (p2, nph, dv) = (d.p2(), d.nph, d.dv);
    let mut out = vec![0.0; d.b * d.t * d.nh * dv];
    for b in 0..d.b {
        for h in 0..d.nh {
            let mut s = vec![0.0; p2 * dv];
            for t in 0..d.t {
                let row = b * d.t + t;
                let q0 = row * d.nh * nph + h * nph;
                let k0 = q0;
                let v0 = row * d.nh * dv + h * dv;
                let kap = kappa[row * d.nh + h];
                let mut p = vec![0.0; p2 * dv];
                let mut r = vec![0.0; dv];
                for f in 0..p2 {
                    let g = decay[h * p2 + f];
                    for j in 0..dv {
                        let x = g * s[f * dv + j];
                        p[f * dv + j] = x;
                        r[j] += phase_delta_phi(&thk[k0..k0 + nph], nph, f) * x;
                    }
                }
                for f in 0..p2 {
                    let kf = phase_delta_phi(&thk[k0..k0 + nph], nph, f);
                    for j in 0..dv {
                        s[f * dv + j] = p[f * dv + j] + kap * kf * (v[v0 + j] - r[j]);
                    }
                }
                for j in 0..dv {
                    let mut x = 0.0;
                    for f in 0..p2 {
                        x += phase_delta_phi(&thq[q0..q0 + nph], nph, f) * s[f * dv + j];
                    }
                    out[row * d.nh * dv + h * dv + j] = x;
                }
            }
        }
    }
    out
}

/// One-token Phase-Delta step for continuation/replay tests. `state` is the
/// feature-major `[2*nphase, dv]` matrix and is updated in place; the output
/// is the post-write `qᵀS` value vector.
pub fn phase_delta_step(
    state: &mut [f64],
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: f64,
    decay: &[f64],
) -> Vec<f64> {
    let nph = thq.len();
    let p2 = 2 * nph;
    let dv = v.len();
    assert_eq!(thk.len(), nph);
    assert_eq!(decay.len(), p2);
    assert_eq!(state.len(), p2 * dv);
    let c = 1.0 / (nph as f64).sqrt();
    let mut p = vec![0.0; p2 * dv];
    let mut r = vec![0.0; dv];
    for f in 0..p2 {
        let kf = if f < nph {
            c * thk[f].cos()
        } else {
            c * thk[f - nph].sin()
        };
        for j in 0..dv {
            p[f * dv + j] = decay[f] * state[f * dv + j];
            r[j] += kf * p[f * dv + j];
        }
    }
    for f in 0..p2 {
        let kf = if f < nph {
            c * thk[f].cos()
        } else {
            c * thk[f - nph].sin()
        };
        for j in 0..dv {
            state[f * dv + j] = p[f * dv + j] + kappa * kf * (v[j] - r[j]);
        }
    }
    let mut out = vec![0.0; dv];
    for f in 0..p2 {
        let qf = if f < nph {
            c * thq[f].cos()
        } else {
            c * thq[f - nph].sin()
        };
        for j in 0..dv {
            out[j] += qf * state[f * dv + j];
        }
    }
    out
}

/// Continuation form of the literal oracle. `state` stores one matrix per
/// batch/head as `[B, nh, 2*nphase, dv]` and is advanced in place.
pub fn phase_delta_ref_fwd_state(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    state: &mut [f64],
) -> Vec<f64> {
    assert_eq!(state.len(), d.b * d.nh * d.p2() * d.dv);
    let mut out = vec![0.0; d.b * d.t * d.nh * d.dv];
    for b in 0..d.b {
        for t in 0..d.t {
            let row = b * d.t + t;
            for h in 0..d.nh {
                let s0 = (b * d.nh + h) * d.p2() * d.dv;
                let step = phase_delta_step(
                    &mut state[s0..s0 + d.p2() * d.dv],
                    &thq[row * d.nh * d.nph + h * d.nph..row * d.nh * d.nph + (h + 1) * d.nph],
                    &thk[row * d.nh * d.nph + h * d.nph..row * d.nh * d.nph + (h + 1) * d.nph],
                    &v[row * d.nh * d.dv + h * d.dv..row * d.nh * d.dv + (h + 1) * d.dv],
                    kappa[row * d.nh + h],
                    &decay[h * d.p2()..(h + 1) * d.p2()],
                );
                out[row * d.nh * d.dv + h * d.dv..row * d.nh * d.dv + (h + 1) * d.dv]
                    .copy_from_slice(&step);
            }
        }
    }
    out
}

/// Exact reverse recurrence for [`phase_delta_ref_fwd`].  Returns gradients
/// with respect to angles, values, and the already-sigmoided write gate for a
/// zero initial state (the historical API used by the gradcheck suite).
#[allow(clippy::type_complexity)]
pub fn phase_delta_ref_bwd(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    dout: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let init = vec![0.0; d.b * d.nh * d.p2() * d.dv];
    let (gq, gk, gv, gb, _) = phase_delta_ref_bwd_state(d, thq, thk, v, kappa, decay, dout, &init);
    (gq, gk, gv, gb)
}

/// Exact reverse recurrence with an explicit nonzero entry state.  The fifth
/// return value is the gradient with respect to that initial `[B,nh,P2,DV]`
/// state; this is the CPU f64 oracle for continuation/backward parity.
#[allow(clippy::type_complexity)]
pub fn phase_delta_ref_bwd_state(
    d: &HkDims,
    thq: &[f64],
    thk: &[f64],
    v: &[f64],
    kappa: &[f64],
    decay: &[f64],
    dout: &[f64],
    initial_state: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (p2, nph, dv) = (d.p2(), d.nph, d.dv);
    assert_eq!(initial_state.len(), d.b * d.nh * p2 * dv);
    let mut dthq = vec![0.0; thq.len()];
    let mut dthk = vec![0.0; thk.len()];
    let mut dv_out = vec![0.0; v.len()];
    let mut dkap = vec![0.0; kappa.len()];
    let mut dstate0 = vec![0.0; initial_state.len()];
    let c = 1.0 / (nph as f64).sqrt();
    for b in 0..d.b {
        for h in 0..d.nh {
            let mut states = vec![0.0; (d.t + 1) * p2 * dv];
            let init_off = (b * d.nh + h) * p2 * dv;
            states[..p2 * dv].copy_from_slice(&initial_state[init_off..init_off + p2 * dv]);
            for t in 0..d.t {
                let row = b * d.t + t;
                let q0 = row * d.nh * nph + h * nph;
                let v0 = row * d.nh * dv + h * dv;
                let kap = kappa[row * d.nh + h];
                let src = t * p2 * dv;
                let (before, after) = states.split_at_mut(src + p2 * dv);
                let prev = &before[src..src + p2 * dv];
                let cur = &mut after[..p2 * dv];
                let mut r = vec![0.0; dv];
                for f in 0..p2 {
                    let kf = phase_delta_phi(&thk[q0..q0 + nph], nph, f);
                    let g = decay[h * p2 + f];
                    for j in 0..dv {
                        r[j] += kf * g * prev[f * dv + j];
                    }
                }
                for f in 0..p2 {
                    let kf = phase_delta_phi(&thk[q0..q0 + nph], nph, f);
                    let g = decay[h * p2 + f];
                    for j in 0..dv {
                        cur[f * dv + j] = g * prev[f * dv + j] + kap * kf * (v[v0 + j] - r[j]);
                    }
                }
            }
            let mut dstate = vec![0.0; p2 * dv];
            for tr in 0..d.t {
                let t = d.t - 1 - tr;
                let row = b * d.t + t;
                let q0 = row * d.nh * nph + h * nph;
                let v0 = row * d.nh * dv + h * dv;
                let kap = kappa[row * d.nh + h];
                let mut qf = vec![0.0; p2];
                let mut kf = vec![0.0; p2];
                for f in 0..p2 {
                    qf[f] = phase_delta_phi(&thq[q0..q0 + nph], nph, f);
                    kf[f] = phase_delta_phi(&thk[q0..q0 + nph], nph, f);
                }
                let prev = &states[t * p2 * dv..(t + 1) * p2 * dv];
                let cur = &states[(t + 1) * p2 * dv..(t + 2) * p2 * dv];
                let mut p = vec![0.0; p2 * dv];
                let mut r = vec![0.0; dv];
                for f in 0..p2 {
                    let g = decay[h * p2 + f];
                    for j in 0..dv {
                        p[f * dv + j] = g * prev[f * dv + j];
                        r[j] += kf[f] * p[f * dv + j];
                    }
                }
                let mut e = vec![0.0; dv];
                for j in 0..dv {
                    e[j] = v[v0 + j] - r[j];
                }
                let mut gtot = vec![0.0; p2 * dv];
                for f in 0..p2 {
                    for j in 0..dv {
                        gtot[f * dv + j] =
                            dstate[f * dv + j] + qf[f] * dout[row * d.nh * dv + h * dv + j];
                    }
                }
                let mut u = vec![0.0; dv];
                for j in 0..dv {
                    for f in 0..p2 {
                        u[j] += gtot[f * dv + j] * kf[f];
                    }
                }
                let mut dbeta = 0.0;
                for j in 0..dv {
                    dbeta += u[j] * e[j];
                    dv_out[v0 + j] += kap * u[j];
                }
                dkap[row * d.nh + h] += dbeta;
                let mut dprev = vec![0.0; p2 * dv];
                for f in 0..p2 {
                    for j in 0..dv {
                        dprev[f * dv + j] =
                            decay[h * p2 + f] * (gtot[f * dv + j] - kf[f] * kap * u[j]);
                    }
                }
                // Chain the cosine/sine channels through each phase angle.
                for i in 0..nph {
                    let mut gq_c = 0.0;
                    let mut gq_s = 0.0;
                    let mut gk_c = 0.0;
                    let mut gk_s = 0.0;
                    for j in 0..dv {
                        gq_c += dout[row * d.nh * dv + h * dv + j] * cur[i * dv + j];
                        gq_s += dout[row * d.nh * dv + h * dv + j] * cur[(nph + i) * dv + j];
                        gk_c += kap * (gtot[i * dv + j] * e[j] - p[i * dv + j] * u[j]);
                        gk_s +=
                            kap * (gtot[(nph + i) * dv + j] * e[j] - p[(nph + i) * dv + j] * u[j]);
                    }
                    dthq[q0 + i] += c * (-thq[q0 + i].sin() * gq_c + thq[q0 + i].cos() * gq_s);
                    dthk[q0 + i] += c * (-thk[q0 + i].sin() * gk_c + thk[q0 + i].cos() * gk_s);
                }
                dstate = dprev;
            }
            dstate0[init_off..init_off + p2 * dv].copy_from_slice(&dstate);
        }
    }
    (dthq, dthk, dv_out, dkap, dstate0)
}

// ---------------------------------------------------------------------
// Appended one-head GDN correction-lane oracle.
// ---------------------------------------------------------------------

/// Geometry for the optional correction lane.  The trainer uses one head
/// with `dk=dv=64`; the oracle is generic over those dimensions so focused
/// tests can exercise tiny cases without a Metal device.
#[derive(Clone, Copy, Debug)]
pub struct GdnLaneDims {
    pub b: usize,
    pub t: usize,
    pub dk: usize,
    pub dv: usize,
}

#[inline]
fn gdn_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn gdn_softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Literal one-head GDN recurrence (f64 accumulate), matching the clean-room
/// equations used by the phase-2 correction lane.  `q/k/v/z` are row-major
/// `[B·T, dk|dk|dv|dv]`, `a/b` are `[B·T]`; `norm` is `[dv]`.
pub fn gdn_ref_fwd(
    d: &GdnLaneDims,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    z: &[f64],
    a: &[f64],
    b_gate: &[f64],
    a_log: f64,
    dt_bias: f64,
    norm: &[f64],
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let rows = d.b * d.t;
    assert!(q.len() >= rows * d.dk && k.len() >= rows * d.dk);
    assert!(v.len() >= rows * d.dv && z.len() >= rows * d.dv);
    let mut out = vec![0.0; rows * d.dv];
    let mut states = vec![0.0; d.b * (d.t + 1) * d.dk * d.dv];
    let ea = a_log.exp();
    for bi in 0..d.b {
        for ti in 0..d.t {
            let row = bi * d.t + ti;
            let prev = (bi * (d.t + 1) + ti) * d.dk * d.dv;
            let cur = (bi * (d.t + 1) + ti + 1) * d.dk * d.dv;
            let mut qn = 0.0;
            let mut kn = 0.0;
            for i in 0..d.dk {
                qn += q[row * d.dk + i] * q[row * d.dk + i];
                kn += k[row * d.dk + i] * k[row * d.dk + i];
            }
            let iq = 1.0 / ((qn + 1e-6).sqrt() * (d.dk as f64).sqrt());
            let ik = 1.0 / (kn + 1e-6).sqrt();
            let g = (-ea * gdn_softplus(a[row] + dt_bias)).exp();
            let beta = gdn_sigmoid(b_gate[row]);
            let mut kv = vec![0.0; d.dv];
            for i in 0..d.dk {
                let kf = k[row * d.dk + i] * ik;
                for j in 0..d.dv {
                    let s = states[prev + i * d.dv + j] * g;
                    states[cur + i * d.dv + j] = s;
                    kv[j] += s * kf;
                }
            }
            let mut oo = vec![0.0; d.dv];
            for i in 0..d.dk {
                let kf = k[row * d.dk + i] * ik;
                let qf = q[row * d.dk + i] * iq;
                for j in 0..d.dv {
                    let s = &mut states[cur + i * d.dv + j];
                    *s += kf * (v[row * d.dv + j] - kv[j]) * beta;
                    oo[j] += qf * *s;
                }
            }
            let mut ss = 0.0;
            for &x in &oo {
                ss += x * x;
            }
            let inv = 1.0 / (ss / d.dv as f64 + eps).sqrt();
            for j in 0..d.dv {
                let zv = z[row * d.dv + j];
                let silu = zv / (1.0 + (-zv).exp());
                out[row * d.dv + j] = oo[j] * inv * norm[j] * silu;
            }
        }
    }
    (out, states)
}

/// Numerical directional-gradient helper for the GDN oracle.  It deliberately
/// uses central differences and is intended for focused finite-difference
/// tests, not the Metal trainer's analytic backward path.
pub fn gdn_ref_directional(
    d: &GdnLaneDims,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    z: &[f64],
    a: &[f64],
    b_gate: &[f64],
    a_log: f64,
    dt_bias: f64,
    norm: &[f64],
    eps: f64,
    dout: &[f64],
    dq: &[f64],
    h: f64,
) -> f64 {
    let dot = |qq: &[f64]| {
        let (o, _) = gdn_ref_fwd(d, qq, k, v, z, a, b_gate, a_log, dt_bias, norm, eps);
        o.iter().zip(dout).map(|(x, y)| x * y).sum::<f64>()
    };
    let mut qp = q.to_vec();
    let mut qm = q.to_vec();
    for (x, d) in qp.iter_mut().zip(dq) {
        *x += h * d;
    }
    for (x, d) in qm.iter_mut().zip(dq) {
        *x -= h * d;
    }
    (dot(&qp) - dot(&qm)) / (2.0 * h)
}

// ---------------------------------------------------------------------
// GDN mixer token scan (plan S7, variant B): the f64 literal recurrence of
// the SCAN part of `linear_core::gdn_step` (inputs are the post-conv,
// post-SiLU q/k/v channels; the conv, the gated RMSNorm and the output gate
// are the shared conv1d / rmsnorm / swiglu ops around it), multi-head with
// an explicit initial state — the oracle of the Metal/WGSL scan kernels.
// ---------------------------------------------------------------------

/// Geometry of one GDN scan: `qkv_cv` rows are `[q_0..q_{nv-1} | k_0.. | v_0..]`
/// (head h's q at `h·dk`, k at `nv·dk + h·dk`, v at `2·nv·dk + h·dv`);
/// `a_pre`/`b_pre` rows are `ab_ld` wide with head h in column h.
#[derive(Clone, Copy, Debug)]
pub struct GdnScanDims {
    pub b: usize,
    pub t: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub c_dim: usize,
    pub ab_ld: usize,
}

impl GdnScanDims {
    pub fn rows(&self) -> usize {
        self.b * self.t
    }
    pub fn state(&self) -> usize {
        self.dk * self.dv
    }
    pub fn nch(&self) -> usize {
        self.t.div_ceil(64)
    }
}

/// Forward result of the f64 scan reference.
pub struct GdnScanRef {
    /// `[B·T, nv·dv]` raw recurrent output `Sᵀq̂`
    pub raw_o: Vec<f64>,
    /// `[B, nv, dk, dv]` final states
    pub s_end: Vec<f64>,
}

/// Through- and parameter-gradients of the f64 scan reference.
pub struct GdnScanGrads {
    /// `[B·T, c_dim]` d qkv_cv (q/k through the L2 norm, v direct)
    pub dcv: Vec<f64>,
    /// `[B·T, nv]` d a_pre
    pub da: Vec<f64>,
    /// `[B·T, nv]` d b_pre
    pub db: Vec<f64>,
    /// `[nv]`
    pub dalog: Vec<f64>,
    /// `[nv]`
    pub ddt: Vec<f64>,
    /// `[B, nv, dk, dv]` adjoint of the initial state
    pub ds0: Vec<f64>,
}

#[inline]
fn gdn_scalars(a: f64, dt: f64, ea: f64, b: f64) -> (f64, f64, f64, f64) {
    // (g, softplus, σ(a+dt), β) exactly as `gdn_step`
    let x = a + dt;
    let sp = gdn_softplus(x);
    let g = (-ea * sp).exp();
    let sig = gdn_sigmoid(x);
    let beta = gdn_sigmoid(b);
    (g, sp, sig, beta)
}

/// Literal f64 GDN scan: `S ← g·S; kv = Sᵀk̂; S += k̂ ⊗ β(v − kv); o = Sᵀq̂`
/// from `s0` (`[B, nv, dk, dv]`, zeros for a fresh state).
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_ref_fwd(
    d: &GdnScanDims,
    qkv_cv: &[f64],
    a_pre: &[f64],
    b_pre: &[f64],
    alog: &[f64],
    dt: &[f64],
    s0: &[f64],
) -> GdnScanRef {
    let (nv, dk, dv, cd) = (d.nv, d.dk, d.dv, d.c_dim);
    let ss = dk * dv;
    let mut raw_o = vec![0.0; d.rows() * nv * dv];
    let mut s_end = vec![0.0; d.b * nv * ss];
    let sdk = (dk as f64).sqrt();
    for bi in 0..d.b {
        for h in 0..nv {
            let bh = bi * nv + h;
            let mut s = s0[bh * ss..(bh + 1) * ss].to_vec();
            let ea = alog[h].exp();
            let mut kv = vec![0.0; dv];
            for ti in 0..d.t {
                let row = bi * d.t + ti;
                let q = &qkv_cv[row * cd + h * dk..row * cd + (h + 1) * dk];
                let k = &qkv_cv[row * cd + nv * dk + h * dk..row * cd + nv * dk + (h + 1) * dk];
                let v = &qkv_cv[row * cd + 2 * nv * dk + h * dv..row * cd + 2 * nv * dk + (h + 1) * dv];
                let qn: f64 = q.iter().map(|x| x * x).sum();
                let kn: f64 = k.iter().map(|x| x * x).sum();
                let iq = 1.0 / ((qn + 1e-6).sqrt() * sdk);
                let ik = 1.0 / (kn + 1e-6).sqrt();
                let (g, _, _, beta) = gdn_scalars(a_pre[row * d.ab_ld + h], dt[h], ea, b_pre[row * d.ab_ld + h]);
                for x in kv.iter_mut() {
                    *x = 0.0;
                }
                for i in 0..dk {
                    let kf = k[i] * ik;
                    for j in 0..dv {
                        s[i * dv + j] *= g;
                        kv[j] += s[i * dv + j] * kf;
                    }
                }
                let o = &mut raw_o[row * nv * dv + h * dv..row * nv * dv + (h + 1) * dv];
                for i in 0..dk {
                    let kf = k[i] * ik;
                    let qf = q[i] * iq;
                    for j in 0..dv {
                        s[i * dv + j] += kf * (v[j] - kv[j]) * beta;
                        o[j] += qf * s[i * dv + j];
                    }
                }
            }
            s_end[bh * ss..(bh + 1) * ss].copy_from_slice(&s);
        }
    }
    GdnScanRef { raw_o, s_end }
}

/// Hand-rolled f64 BPTT of [`gdn_scan_ref_fwd`] (keeps every state; the
/// kernels replay chunks instead). `doo` is `[B·T, nv·dv]`; `ds_end` (the
/// adjoint of the final state, `[B, nv, dk, dv]`) may be `None` for zero.
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_ref_bwd(
    d: &GdnScanDims,
    qkv_cv: &[f64],
    a_pre: &[f64],
    b_pre: &[f64],
    alog: &[f64],
    dt: &[f64],
    s0: &[f64],
    doo: &[f64],
    ds_end: Option<&[f64]>,
) -> GdnScanGrads {
    let (nv, dk, dv, cd) = (d.nv, d.dk, d.dv, d.c_dim);
    let ss = dk * dv;
    let rows = d.rows();
    let mut dcv = vec![0.0; rows * cd];
    let mut da = vec![0.0; rows * nv];
    let mut db = vec![0.0; rows * nv];
    let mut dalog = vec![0.0; nv];
    let mut ddt = vec![0.0; nv];
    let mut ds0 = vec![0.0; d.b * nv * ss];
    let sdk = (dk as f64).sqrt();
    for bi in 0..d.b {
        for h in 0..nv {
            let bh = bi * nv + h;
            let ea = alog[h].exp();
            // forward with every state kept: st[ti] = S before token ti
            let mut st = vec![0.0; (d.t + 1) * ss];
            st[..ss].copy_from_slice(&s0[bh * ss..(bh + 1) * ss]);
            let mut kvs = vec![0.0; d.t * dv];
            let mut sc = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); d.t];
            for ti in 0..d.t {
                let row = bi * d.t + ti;
                let q = &qkv_cv[row * cd + h * dk..row * cd + (h + 1) * dk];
                let k = &qkv_cv[row * cd + nv * dk + h * dk..row * cd + nv * dk + (h + 1) * dk];
                let v = &qkv_cv[row * cd + 2 * nv * dk + h * dv..row * cd + 2 * nv * dk + (h + 1) * dv];
                let qn: f64 = q.iter().map(|x| x * x).sum();
                let kn: f64 = k.iter().map(|x| x * x).sum();
                let iq = 1.0 / ((qn + 1e-6).sqrt() * sdk);
                let ik = 1.0 / (kn + 1e-6).sqrt();
                let (g, sp, sig, beta) = gdn_scalars(a_pre[row * d.ab_ld + h], dt[h], ea, b_pre[row * d.ab_ld + h]);
                sc[ti] = (iq, ik, g, sp, sig, beta);
                let (prev, cur) = st.split_at_mut((ti + 1) * ss);
                let prev = &prev[ti * ss..];
                let cur = &mut cur[..ss];
                let kv = &mut kvs[ti * dv..(ti + 1) * dv];
                for i in 0..dk {
                    let kf = k[i] * ik;
                    for j in 0..dv {
                        kv[j] += prev[i * dv + j] * kf; // undecayed Sᵀk̂
                    }
                }
                for i in 0..dk {
                    let kf = k[i] * ik;
                    for j in 0..dv {
                        cur[i * dv + j] = g * prev[i * dv + j] + kf * (v[j] - g * kv[j]) * beta;
                    }
                }
            }
            // reverse
            let mut ds = match ds_end {
                Some(x) => x[bh * ss..(bh + 1) * ss].to_vec(),
                None => vec![0.0; ss],
            };
            for ti in (0..d.t).rev() {
                let row = bi * d.t + ti;
                let q = &qkv_cv[row * cd + h * dk..row * cd + (h + 1) * dk];
                let k = &qkv_cv[row * cd + nv * dk + h * dk..row * cd + nv * dk + (h + 1) * dk];
                let v = &qkv_cv[row * cd + 2 * nv * dk + h * dv..row * cd + 2 * nv * dk + (h + 1) * dv];
                let dout = &doo[row * nv * dv + h * dv..row * nv * dv + (h + 1) * dv];
                let (iq, ik, g, sp, sig, beta) = sc[ti];
                let prev = &st[ti * ss..(ti + 1) * ss];
                let cur = &st[(ti + 1) * ss..(ti + 2) * ss];
                let kv = &kvs[ti * dv..(ti + 1) * dv];
                // o = S_tᵀ q̂
                let mut dqh = vec![0.0; dk];
                for i in 0..dk {
                    let qf = q[i] * iq;
                    for j in 0..dv {
                        dqh[i] += cur[i * dv + j] * dout[j];
                        ds[i * dv + j] += qf * dout[j];
                    }
                }
                // S_t = g S_{t-1} + k̂ ⊗ u, u = β (v − g kv)
                let mut du = vec![0.0; dv];
                let mut dkh = vec![0.0; dk];
                for i in 0..dk {
                    let kf = k[i] * ik;
                    for j in 0..dv {
                        let u = beta * (v[j] - g * kv[j]);
                        du[j] += ds[i * dv + j] * kf;
                        dkh[i] += ds[i * dv + j] * u;
                    }
                }
                let mut dbeta = 0.0;
                let mut dg = 0.0;
                let mut dkv = vec![0.0; dv];
                for j in 0..dv {
                    dbeta += du[j] * (v[j] - g * kv[j]);
                    dcv[row * cd + 2 * nv * dk + h * dv + j] = beta * du[j];
                    dkv[j] = -beta * g * du[j];
                    dg += -beta * du[j] * kv[j];
                }
                // kv = S_{t-1}ᵀ k̂ ; S_t ∋ g S_{t-1}
                let mut ds_prev = vec![0.0; ss];
                for i in 0..dk {
                    let kf = k[i] * ik;
                    for j in 0..dv {
                        dkh[i] += prev[i * dv + j] * dkv[j];
                        dg += ds[i * dv + j] * prev[i * dv + j];
                        ds_prev[i * dv + j] = g * ds[i * dv + j] + kf * dkv[j];
                    }
                }
                ds = ds_prev;
                // L2 norms
                let dqdot: f64 = (0..dk).map(|i| dqh[i] * q[i]).sum();
                let dkdot: f64 = (0..dk).map(|i| dkh[i] * k[i]).sum();
                for i in 0..dk {
                    dcv[row * cd + h * dk + i] = iq * dqh[i] - q[i] * iq * iq * iq * dk as f64 * dqdot;
                    dcv[row * cd + nv * dk + h * dk + i] = ik * dkh[i] - k[i] * ik * ik * ik * dkdot;
                }
                // g = exp(−e^A·softplus(a+dt)), β = σ(b)
                let d_a = dg * (-ea * sig * g);
                da[row * nv + h] = d_a;
                db[row * nv + h] = dbeta * beta * (1.0 - beta);
                dalog[h] += dg * (-ea * sp * g);
                ddt[h] += d_a;
            }
            ds0[bh * ss..(bh + 1) * ss].copy_from_slice(&ds);
        }
    }
    GdnScanGrads {
        dcv,
        da,
        db,
        dalog,
        ddt,
        ds0,
    }
}

// ---------------------------------------------------------------------
// Bounded anchor `swa_sink_v1` (docs/EMBRYO_BOUNDED_ANCHOR.md §1): the f64
// reference of the served operator used by the Metal and Vulkan gradchecks
// (absolute rope on q̂/k̂, band `j ∈ (t − W, t]`, S NoPE sinks, one softmax,
// then the output projection). Forward + hand-rolled backward.
// ---------------------------------------------------------------------

/// Geometry of one bounded-anchor reference evaluation.

pub struct BoundedAnchorDims {
    pub b: usize,
    pub t: usize,
    pub qh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub h: usize,
    pub s: usize,
    pub w: usize,
    pub base: f64,
}

/// The trainer's RoPE (`rope_f32`: pair (i, i + hd/2), angle pos·base^(−2i/hd)) in f64.
pub fn rope_ref(x: &[f64], pos: usize, hd: usize, base: f64, inverse: bool) -> Vec<f64> {
    let half = hd / 2;
    let mut y = vec![0.0; hd];
    for i in 0..half {
        let inv_freq = base.powf(-(2.0 * i as f64) / hd as f64);
        let ang = pos as f64 * inv_freq;
        let (c, mut s) = (ang.cos(), ang.sin());
        if inverse {
            s = -s;
        }
        y[i] = x[i] * c - x[i + half] * s;
        y[i + half] = x[i] * s + x[i + half] * c;
    }
    y
}

/// Band predicate: key `j` visible from query `t` (`w = 0`: full causal).
pub fn bounded_valid(d: &BoundedAnchorDims, t: usize, j: usize) -> bool {
    j <= t && (d.w == 0 || t - j < d.w)
}

pub struct BoundedAnchorRef {
    pub y: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub dsink_k: Vec<f64>,
    pub dsink_v: Vec<f64>,
    pub dwo: Vec<f64>,
}

/// Forward + analytic backward of the bounded attention core followed by
/// the output projection `y = o·Woᵀ`, in f64. Layouts mirror the trainer:
/// q̂ [b·t, qh·hd], k̂/v [b·t, kvh·hd], sinks [kvh, S, hd], Wo [H, qh·hd].
#[allow(clippy::too_many_arguments)]
pub fn bounded_anchor_ref(
    d: &BoundedAnchorDims,
    q_raw: &[f64],
    k_raw: &[f64],
    v: &[f64],
    sink_k: &[f64],
    sink_v: &[f64],
    wo: &[f64],
    dy: Option<&[f64]>,
) -> BoundedAnchorRef {
    let (qd, kd) = (d.qh * d.hd, d.kvh * d.hd);
    let group = d.qh / d.kvh;
    let scale = 1.0 / (d.hd as f64).sqrt();
    let m = d.b * d.t;
    let mut o = vec![0.0; m * qd];
    let mut dq = vec![0.0; m * qd];
    let mut dk = vec![0.0; m * kd];
    let mut dv = vec![0.0; m * kd];
    let mut dsk = vec![0.0; d.kvh * d.s * d.hd];
    let mut dsv = vec![0.0; d.kvh * d.s * d.hd];
    let mut dwo = vec![0.0; d.h * qd];
    // dO = dy·Wo ; dWo = dyᵀ·o (after the forward)
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut y = vec![0.0; m * d.h];
    for bi in 0..d.b {
        for g in 0..d.kvh {
            let krot: Vec<Vec<f64>> = (0..d.t)
                .map(|j| {
                    let r = (bi * d.t + j) * kd + g * d.hd;
                    rope_ref(&k_raw[r..r + d.hd], j, d.hd, d.base, false)
                })
                .collect();
            for jj in 0..group {
                let hh = g * group + jj;
                let mut p_all: Vec<Vec<f64>> = Vec::with_capacity(d.t);
                let mut qrots: Vec<Vec<f64>> = Vec::with_capacity(d.t);
                for t in 0..d.t {
                    let r = (bi * d.t + t) * qd + hh * d.hd;
                    let qr = &q_raw[r..r + d.hd];
                    let qrot = rope_ref(qr, t, d.hd, d.base, false);
                    let mut sc = vec![f64::NEG_INFINITY; d.s + d.t];
                    for s in 0..d.s {
                        let sk = &sink_k[(g * d.s + s) * d.hd..(g * d.s + s + 1) * d.hd];
                        sc[s] = dot(qr, sk) * scale;
                    }
                    for j in 0..d.t {
                        if bounded_valid(d, t, j) {
                            sc[d.s + j] = dot(&qrot, &krot[j]) * scale;
                        }
                    }
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut p: Vec<f64> = sc
                        .iter()
                        .map(|z| if z.is_finite() { (z - mx).exp() } else { 0.0 })
                        .collect();
                    let sum: f64 = p.iter().sum();
                    for x in &mut p {
                        *x /= sum;
                    }
                    let orow = &mut o[(bi * d.t + t) * qd + hh * d.hd..(bi * d.t + t) * qd + (hh + 1) * d.hd];
                    for s in 0..d.s {
                        let sv = &sink_v[(g * d.s + s) * d.hd..(g * d.s + s + 1) * d.hd];
                        for x in 0..d.hd {
                            orow[x] += p[s] * sv[x];
                        }
                    }
                    for j in 0..d.t {
                        if p[d.s + j] != 0.0 {
                            let vr = (bi * d.t + j) * kd + g * d.hd;
                            for x in 0..d.hd {
                                orow[x] += p[d.s + j] * v[vr + x];
                            }
                        }
                    }
                    p_all.push(p);
                    qrots.push(qrot);
                }
                let Some(dy) = dy else { continue };
                // dO for this head's rows
                for t in 0..d.t {
                    let row = bi * d.t + t;
                    let mut d_o = vec![0.0; d.hd];
                    for c in 0..d.h {
                        let g_c = dy[row * d.h + c];
                        for x in 0..d.hd {
                            d_o[x] += g_c * wo[c * qd + hh * d.hd + x];
                        }
                    }
                    let p = &p_all[t];
                    // dp over the valid entries
                    let mut dp = vec![0.0; d.s + d.t];
                    for s in 0..d.s {
                        let sv = &sink_v[(g * d.s + s) * d.hd..(g * d.s + s + 1) * d.hd];
                        dp[s] = dot(&d_o, sv);
                        for x in 0..d.hd {
                            dsv[(g * d.s + s) * d.hd + x] += p[s] * d_o[x];
                        }
                    }
                    for j in 0..d.t {
                        if bounded_valid(d, t, j) {
                            let vr = (bi * d.t + j) * kd + g * d.hd;
                            dp[d.s + j] = dot(&d_o, &v[vr..vr + d.hd]);
                            for x in 0..d.hd {
                                dv[vr + x] += p[d.s + j] * d_o[x];
                            }
                        }
                    }
                    let rowsum: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                    let ds: Vec<f64> = p.iter().zip(&dp).map(|(a, b)| a * (b - rowsum)).collect();
                    // window: dq_rot, dk_rot
                    let mut dq_rot = vec![0.0; d.hd];
                    for j in 0..d.t {
                        if bounded_valid(d, t, j) {
                            let c = ds[d.s + j] * scale;
                            for x in 0..d.hd {
                                dq_rot[x] += c * krot[j][x];
                            }
                            let kr = (bi * d.t + j) * kd + g * d.hd;
                            // accumulate dk in ROTATED space, inverse-rotate at the end
                            for x in 0..d.hd {
                                dk[kr + x] += c * qrots[t][x];
                            }
                        }
                    }
                    let mut dq_raw = rope_ref(&dq_rot, t, d.hd, d.base, true);
                    let qr = &q_raw[row * qd + hh * d.hd..row * qd + (hh + 1) * d.hd];
                    for s in 0..d.s {
                        let c = ds[s] * scale;
                        let sk = &sink_k[(g * d.s + s) * d.hd..(g * d.s + s + 1) * d.hd];
                        for x in 0..d.hd {
                            dq_raw[x] += c * sk[x];
                            dsk[(g * d.s + s) * d.hd + x] += c * qr[x];
                        }
                    }
                    dq[row * qd + hh * d.hd..row * qd + (hh + 1) * d.hd].copy_from_slice(&dq_raw);
                }
            }
            // dk: rotated accumulator → raw space (per key position)
            if dy.is_some() {
                for j in 0..d.t {
                    let kr = (bi * d.t + j) * kd + g * d.hd;
                    let raw = rope_ref(&dk[kr..kr + d.hd], j, d.hd, d.base, true);
                    dk[kr..kr + d.hd].copy_from_slice(&raw);
                }
            }
        }
    }
    for row in 0..m {
        for c in 0..d.h {
            let mut acc = 0.0;
            for i in 0..qd {
                acc += o[row * qd + i] * wo[c * qd + i];
            }
            y[row * d.h + c] = acc;
        }
    }
    if let Some(dy) = dy {
        for row in 0..m {
            for c in 0..d.h {
                let g_c = dy[row * d.h + c];
                for i in 0..qd {
                    dwo[c * qd + i] += g_c * o[row * qd + i];
                }
            }
        }
    }
    BoundedAnchorRef {
        y,
        dq,
        dk,
        dv,
        dsink_k: dsk,
        dsink_v: dsv,
        dwo,
    }
}

// ---------------------------------------------------------------------
// Kernel fixture shared by the Metal and Vulkan band+sink softmax tests
// (scores, upstream gradient, sampled positions, f64 reference).
// ---------------------------------------------------------------------

/// Shared fixture of the kernel tests (identical on both backends).
pub struct SoftmaxFixture {
    pub t: usize,
    pub ld: usize,
    pub sink: usize,
    pub sink_pad: usize,
    pub window: usize,
    pub blocks: usize,
    pub off: usize,
}

pub const SOFTMAX_FIXTURE: SoftmaxFixture = SoftmaxFixture {
    t: 64,
    ld: 128,
    sink: 3,
    sink_pad: 64,
    window: 5,
    blocks: 3,
    off: 4,
};

/// Sampled (block, row, col) triples of the fixture: sink columns, the
/// diagonal, the band edge and a masked column.
pub fn softmax_samples(f: &SoftmaxFixture) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for i in 0..16 {
        let block = i % f.blocks;
        let row = (i * 13 + 7) % f.t;
        let col = match i % 4 {
            0 => i % f.sink,                                   // sink column
            1 => f.sink_pad + row,                             // diagonal
            2 => f.sink_pad + row.saturating_sub(f.window - 1), // band edge
            _ => f.sink_pad + row.saturating_sub(f.window),    // just outside (0)
        };
        v.push((block, row, col));
    }
    v
}

pub fn softmax_scores(f: &SoftmaxFixture) -> Vec<f32> {
    lcg_vec(4242, f.off + f.blocks * f.t * f.ld).iter().map(|x| x * 3.0).collect()
}

pub fn softmax_upstream(f: &SoftmaxFixture) -> Vec<f32> {
    lcg_vec(4343, f.off + f.blocks * f.t * f.ld)
}

/// f64 reference of the band+sink softmax rows (P) and its backward (dS).
pub fn softmax_reference(f: &SoftmaxFixture, scores: &[f32], upstream: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut p = vec![0.0f64; scores.len()];
    let mut ds = vec![0.0f64; scores.len()];
    for block in 0..f.blocks {
        for row in 0..f.t {
            let base = f.off + block * f.t * f.ld + row * f.ld;
            let ok = |col: usize| -> bool {
                col < f.sink
                    || (col >= f.sink_pad
                        && col <= f.sink_pad + row
                        && (f.window == 0 || row - (col - f.sink_pad) < f.window))
            };
            let mx = (0..f.ld)
                .filter(|&c| ok(c))
                .map(|c| scores[base + c] as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = (0..f.ld)
                .filter(|&c| ok(c))
                .map(|c| (scores[base + c] as f64 - mx).exp())
                .sum();
            for c in 0..f.ld {
                p[base + c] = if ok(c) { (scores[base + c] as f64 - mx).exp() / den } else { 0.0 };
            }
            let dot: f64 = (0..f.ld).map(|c| p[base + c] * upstream[base + c] as f64).sum();
            for c in 0..f.ld {
                ds[base + c] = p[base + c] * (upstream[base + c] as f64 - dot);
            }
        }
    }
    (p, ds)
}
