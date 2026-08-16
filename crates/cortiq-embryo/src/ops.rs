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
            let prev = if beta != 0.0 { beta as f64 * c[idx] as f64 } else { 0.0 };
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
            let frac = if nph > 1 { i as f64 / (nph - 1) as f64 } else { 0.0 };
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
    if f < nph { theta[f].cos() } else { theta[f - nph].sin() }
}

/// Reference forward by the literal recurrence (f64). Returns o.
pub fn hk_ref_fwd(d: &HkDims, thq: &[f64], thk: &[f64], v: &[f64], kappa: &[f64], decay: &[f64]) -> Vec<f64> {
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
                        acc += fq[t * p2 + f] * fk[s * p2 + f] * decay[h * p2 + f].powi((t - s) as i32);
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
                    dthq[idx_th(t) + i] += -tq.sin() * dfq[t * p2 + i] + tq.cos() * dfq[t * p2 + nph + i];
                    dthk[idx_th(t) + i] += -tk.sin() * dfk[t * p2 + i] + tk.cos() * dfk[t * p2 + nph + i];
                }
            }
        }
    }
    (dthq, dthk, dvv, dkap)
}
