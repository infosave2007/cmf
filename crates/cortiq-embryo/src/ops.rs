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
