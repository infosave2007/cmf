//! Calibration-time Hessian capture for GPTQ / error-feedback transfer.
//!
//! While `begin()` is active, every `QTensor::matmat` (the batched prefill
//! path) folds its input activations into a per-tensor second-moment
//! `H = Σ_t x_t·x_tᵀ` and per-channel `Σ x²` (for the activation RMS field
//! of the two-field outlier score). One global hook covers every linear in
//! the model — attention projections, FFN, experts — with no per-layer
//! wiring. Zero cost when off (one relaxed atomic load). Single-threaded
//! accumulation from the caller's thread, so it never races the parallel
//! matmat that follows.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static ON: AtomicBool = AtomicBool::new(false);
/// Accumulate the full `H = X·Xᵀ` (needed only for the GPTQ fold). Off ⇒
/// diagonal-only (`Σx²`), which is all the ternary + per-row correction
/// path needs — and the only thing that fits for a 12B (full H would be
/// ~100 GB across the model).
static FULL_H: AtomicBool = AtomicBool::new(true);
static REG: Mutex<Option<HashMap<String, HessianAcc>>> = Mutex::new(None);

/// Accumulated input statistics of one linear layer over the calibration set.
pub struct HessianAcc {
    pub cols: usize,
    /// `H = Σ_t x_t·x_tᵀ`, dense `[cols·cols]`, f64.
    pub h: Vec<f64>,
    /// `Σ_t x_t²` per input channel `[cols]` — RMS = sqrt(sumsq / count).
    pub sumsq: Vec<f64>,
    /// Token-position samples folded in.
    pub count: usize,
}

impl HessianAcc {
    /// Per-input-channel activation RMS.
    pub fn rms(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.sumsq.iter().map(|&s| (s / n).sqrt() as f32).collect()
    }
}

/// Start capturing (clears any prior registry). `full_h` = accumulate the
/// dense Hessian (fold path); false = diagonal-only (ternary + correction).
pub fn begin(full_h: bool) {
    *REG.lock().unwrap() = Some(HashMap::new());
    FULL_H.store(full_h, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
}

/// Stop capturing and take the accumulated Hessians. `accumulate` fills
/// only the upper triangle of each dense `H`; it is mirrored here so every
/// consumer sees the full symmetric matrix.
pub fn end() -> HashMap<String, HessianAcc> {
    ON.store(false, Ordering::SeqCst);
    let mut map = REG.lock().unwrap().take().unwrap_or_default();
    for acc in map.values_mut() {
        let n = acc.cols;
        if acc.h.len() == n * n {
            for i in 0..n {
                for j in (i + 1)..n {
                    acc.h[j * n + i] = acc.h[i * n + j];
                }
            }
        }
    }
    map
}

#[inline]
pub fn capturing() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Fold a batch of `b` input vectors (`xs = [b·cols]`) into `name`'s Hessian.
pub fn accumulate(name: &str, xs: &[f32], b: usize, cols: usize) {
    let mut guard = REG.lock().unwrap();
    let Some(map) = guard.as_mut() else {
        return;
    };
    let full = FULL_H.load(Ordering::Relaxed);
    let acc = map.entry(name.to_string()).or_insert_with(|| HessianAcc {
        cols,
        h: if full {
            vec![0.0; cols * cols]
        } else {
            Vec::new()
        },
        sumsq: vec![0.0; cols],
        count: 0,
    });
    if acc.cols != cols {
        return; // shape mismatch — skip defensively
    }
    for bi in 0..b {
        let x = &xs[bi * cols..(bi + 1) * cols];
        for (i, &xi) in x.iter().enumerate() {
            acc.sumsq[i] += xi as f64 * xi as f64;
        }
        acc.count += 1;
    }
    if full {
        gram_upper_add(&mut acc.h, xs, b, cols);
    }
}

/// `H[i][j] += Σ_t x_t[i]·x_t[j]` for `j ≥ i` (upper triangle only; `end()`
/// mirrors it). The batch is transposed so each entry is one contiguous
/// length-`b` dot product, and rows are dealt round-robin over threads so
/// the triangle's uneven row lengths balance out. The former rank-1 update
/// per token streamed the whole `cols²` matrix once per token and made
/// calibrating even a 2B model take hours; this is compute-bound instead.
fn gram_upper_add(h: &mut [f64], xs: &[f32], b: usize, cols: usize) {
    if b == 0 || cols == 0 {
        return;
    }
    let mut xt = vec![0f32; cols * b];
    for t in 0..b {
        let row = &xs[t * cols..(t + 1) * cols];
        for (i, &v) in row.iter().enumerate() {
            xt[i * b + t] = v;
        }
    }
    let nth = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(cols.div_ceil(16))
        .max(1);
    struct SendPtr(*mut f64);
    unsafe impl Sync for SendPtr {}
    unsafe impl Send for SendPtr {}
    let hp = SendPtr(h.as_mut_ptr());
    let hp = &hp;
    let xt = &xt;
    std::thread::scope(|s| {
        for k in 0..nth {
            s.spawn(move || {
                // Rows in pairs; pair p goes to thread p % nth. Each (i, j)
                // belongs to exactly one thread (the owner of row i).
                let mut i = 2 * k;
                while i < cols {
                    // SAFETY: disjoint rows of H per thread (see above).
                    unsafe { gram_pair_dispatch(xt, b, cols, i, hp.0) };
                    i += 2 * nth;
                }
            });
        }
    });
}

unsafe fn gram_pair_dispatch(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { gram_pair_avx2(xt, b, cols, i, h) };
        }
    }
    unsafe { gram_pair(xt, b, cols, i, h) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gram_pair_avx2(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    unsafe { gram_pair(xt, b, cols, i, h) }
}

/// Upper-triangle entries of rows `i` and `i+1`: a 2×4 register block of
/// 8-lane accumulators over the batch axis (10 vector registers — fits the
/// 16 of AVX2), so each loaded activation lane feeds 4 or 2 products
/// instead of one.
#[inline(always)]
unsafe fn gram_pair(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    let two = i + 1 < cols;
    let a0 = &xt[i * b..(i + 1) * b];
    let a1 = if two { &xt[(i + 1) * b..(i + 2) * b] } else { a0 };
    let n8 = b / 8;
    let mut j = i;
    while j + 4 <= cols {
        let ys = [
            &xt[j * b..(j + 1) * b],
            &xt[(j + 1) * b..(j + 2) * b],
            &xt[(j + 2) * b..(j + 3) * b],
            &xt[(j + 3) * b..(j + 4) * b],
        ];
        let mut acc = [[[0f32; 8]; 4]; 2];
        for c in 0..n8 {
            let x0: &[f32; 8] = a0[c * 8..c * 8 + 8].try_into().unwrap();
            let x1: &[f32; 8] = a1[c * 8..c * 8 + 8].try_into().unwrap();
            for q in 0..4 {
                let y: &[f32; 8] = ys[q][c * 8..c * 8 + 8].try_into().unwrap();
                for l in 0..8 {
                    acc[0][q][l] += x0[l] * y[l];
                    acc[1][q][l] += x1[l] * y[l];
                }
            }
        }
        for q in 0..4 {
            let mut s0 = acc[0][q].iter().sum::<f32>();
            let mut s1 = acc[1][q].iter().sum::<f32>();
            for t in n8 * 8..b {
                s0 += a0[t] * ys[q][t];
                s1 += a1[t] * ys[q][t];
            }
            let jj = j + q;
            unsafe { *h.add(i * cols + jj) += s0 as f64 };
            if two && jj > i {
                unsafe { *h.add((i + 1) * cols + jj) += s1 as f64 };
            }
        }
        j += 4;
    }
    while j < cols {
        let y = &xt[j * b..(j + 1) * b];
        unsafe { *h.add(i * cols + j) += dot_f32(a0, y) as f64 };
        if two && j > i {
            unsafe { *h.add((i + 1) * cols + j) += dot_f32(a1, y) as f64 };
        }
        j += 1;
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] * b[c * 8 + l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for k in chunks * 8..a.len() {
        s += a[k] * b[k];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_matches_rank1_reference() {
        let (b, cols) = (13usize, 37usize);
        let xs: Vec<f32> = (0..b * cols)
            .map(|k| ((k * 7919 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let mut h = vec![0f64; cols * cols];
        gram_upper_add(&mut h, &xs, b, cols);
        for i in 0..cols {
            for j in i..cols {
                let r: f64 = (0..b)
                    .map(|t| xs[t * cols + i] as f64 * xs[t * cols + j] as f64)
                    .sum();
                assert!((h[i * cols + j] - r).abs() < 1e-3 * (1.0 + r.abs()), "{i},{j}");
            }
            for j in 0..i {
                assert_eq!(h[i * cols + j], 0.0);
            }
        }
    }
}
