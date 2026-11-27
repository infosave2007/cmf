//! Host kernels of the GigaAM encoder: the row-batch product against a
//! weight in the file's codec, vector `exp` and the activations built on
//! it, the front end's convolutions.
//!
//! The product never keeps a float copy of a weight: a block of rows is
//! widened into a scratch tile that stays in cache and multiplied by the
//! whole frame batch with one f32 GEMM, so a 240 MB eight-bit file runs at
//! float-GEMM speed with float-GEMM numerics and the file's own memory.

use crate::pool::Pool;
use crate::qtensor::sgemm_public;

/// A raw pointer that may cross to pool workers writing disjoint ranges.
#[derive(Clone, Copy)]
pub(super) struct Shared<T>(pub *mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    /// # Safety
    /// The caller guarantees `[at, at + len)` is in bounds and that no other
    /// thread touches it for the life of the slice.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn slice(&self, at: usize, len: usize) -> &mut [f32]
    where
        T: Sized,
    {
        unsafe { std::slice::from_raw_parts_mut(self.0.cast::<f32>().add(at), len) }
    }
}

/// Run `f(start, end)` over `0..n` on the pool (or inline without one, or
/// when the job is small).
pub(super) fn par(pool: Option<&Pool>, n: usize, min: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) if n >= min.max(2) => p.run_rows(n, f),
        _ => f(0, n),
    }
}

// ---------------------------------------------------------------------
// exp and the activations
// ---------------------------------------------------------------------

/// `exp(x)` to float precision (relative error under 2e-7), written so the
/// compiler vectorizes the loop it sits in: no branches, no library call.
#[inline(always)]
fn exp_f(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let x = x.clamp(-87.0, 88.0);
    let t = x * LOG2E;
    // round to nearest through a truncating cast (one instruction on every target)
    let n = (t + if t >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let nf = n as f32;
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    p * f32::from_bits(((n + 127) as u32) << 23)
}

#[inline(always)]
fn sigmoid_f(x: f32) -> f32 {
    1.0 / (1.0 + exp_f(-x))
}

macro_rules! simd_fn {
    ($(#[$doc:meta])* $name:ident, $body:ident, ($($arg:ident : $ty:ty),*)) => {
        $(#[$doc])*
        pub(super) fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn wide($($arg: $ty),*) {
                    $body($($arg),*)
                }
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    // SAFETY: the features were just detected.
                    return unsafe { wide($($arg),*) };
                }
            }
            $body($($arg),*)
        }
    };
}

#[inline(always)]
fn silu_body(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= sigmoid_f(*v);
    }
}
simd_fn!(
    /// `x ← x · σ(x)`
    silu, silu_body, (x: &mut [f32])
);

#[inline(always)]
fn mul_sigmoid_body(x: &mut [f32], g: &[f32]) {
    for (v, &g) in x.iter_mut().zip(g) {
        *v *= sigmoid_f(g);
    }
}
simd_fn!(
    /// `x ← x · σ(g)`
    mul_sigmoid, mul_sigmoid_body, (x: &mut [f32], g: &[f32])
);

#[inline(always)]
fn add_mul_sigmoid_silu_body(x: &mut [f32], add: &[f32], g: &[f32]) {
    for ((v, &a), &g) in x.iter_mut().zip(add).zip(g) {
        let t = (*v + a) * sigmoid_f(g);
        *v = t * sigmoid_f(t);
    }
}
simd_fn!(
    /// `x ← silu((x + add) · σ(g))` — the convolution module's tail
    add_mul_sigmoid_silu, add_mul_sigmoid_silu_body, (x: &mut [f32], add: &[f32], g: &[f32])
);

#[inline(always)]
fn softmax_scaled_body(x: &mut [f32], scale: f32) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        max = max.max(v);
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = exp_f((*v - max) * scale);
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}
simd_fn!(
    /// `x ← softmax(scale · x)` over one row
    softmax_scaled, softmax_scaled_body, (x: &mut [f32], scale: f32)
);

#[inline(always)]
fn axpy_body(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}
simd_fn!(
    /// `y ← y + a·x`
    axpy, axpy_body, (y: &mut [f32], a: f32, x: &[f32])
);

#[inline(always)]
fn widen_i8_body(dst: &mut [f32], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as i8 as f32;
    }
}
simd_fn!(
    /// int8 codes to f32
    widen_i8, widen_i8_body, (dst: &mut [f32], src: &[u8])
);

#[inline(always)]
fn mul_body(x: &mut [f32], s: &[f32]) {
    for (v, &s) in x.iter_mut().zip(s) {
        *v *= s;
    }
}
simd_fn!(
    /// `x ← x ⊙ s`
    mul, mul_body, (x: &mut [f32], s: &[f32])
);

#[inline(always)]
fn sumsq_body(x: &[f32], out: &mut f32) {
    // eight lanes, so the reduction vectorizes without reassociation flags
    let mut acc = [0.0f32; 8];
    let chunks = x.chunks_exact(8);
    let tail = chunks.remainder();
    for c in chunks {
        for (a, &v) in acc.iter_mut().zip(c) {
            *a += v * v;
        }
    }
    let mut s = 0.0f32;
    for a in acc {
        s += a;
    }
    for &v in tail {
        s += v * v;
    }
    *out = s;
}
simd_fn!(sumsq_into, sumsq_body, (x: &[f32], out: &mut f32));

/// Σ x²
pub(super) fn sumsq(x: &[f32]) -> f32 {
    let mut s = 0.0;
    sumsq_into(x, &mut s);
    s
}

// ---------------------------------------------------------------------
// the row-batch product
// ---------------------------------------------------------------------

/// A weight the host multiplies by.
pub(super) enum Weight<'a> {
    /// `[rows, cols]` floats
    Dense(&'a [f32]),
    /// int8 codes by row, a scale per row and a field per column
    /// (`w[r][c] = code · row_scale[r] · col_field[c]`; an empty field is 1)
    Q8 { codes: &'a [u8], row_scale: &'a [f32], col_field: &'a [f32] },
    /// any other codec: rows are dequantized by the tensor itself
    Rows(&'a crate::qtensor::QTensor),
}

/// Weight rows widened per tile: the tile (`TILE_FLOATS` floats) stays in
/// the second-level cache while the frame batch streams past it.
const TILE_FLOATS: usize = 1 << 19;

/// `y[n, rows] = x[n, cols] · Wᵀ` (no bias), `y` overwritten.
///
/// Off macOS the tiles are spread over the pool (one single-threaded GEMM
/// each); on macOS Accelerate threads the GEMM itself, so the tiles go one
/// after another and only the widening is spread.
pub(super) fn matmul(w: &Weight, rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(x.len(), n * cols);
    debug_assert_eq!(y.len(), n * rows);
    if n == 0 || rows == 0 {
        return;
    }
    // the column field belongs to the activation side of the product
    let scaled: Option<Vec<f32>> = match w {
        Weight::Q8 { col_field, .. } if !col_field.is_empty() => {
            let mut xs = x.to_vec();
            let xp = Shared(xs.as_mut_ptr());
            par(pool, n, 64, &|a, b| {
                // SAFETY: disjoint row ranges of `xs`.
                let part = unsafe { xp.slice(a * cols, (b - a) * cols) };
                for row in part.chunks_exact_mut(cols) {
                    mul(row, col_field);
                }
            });
            Some(xs)
        }
        _ => None,
    };
    let x = scaled.as_deref().unwrap_or(x);
    let yp = Shared(y.as_mut_ptr());
    let fill = |tile: &mut [f32], r0: usize, r1: usize| match w {
        Weight::Dense(_) => unreachable!(),
        Weight::Q8 { codes, .. } => {
            for (t, r) in tile.chunks_exact_mut(cols).zip(r0..r1) {
                widen_i8(t, &codes[r * cols..(r + 1) * cols]);
            }
        }
        Weight::Rows(q) => {
            for (t, r) in tile.chunks_exact_mut(cols).zip(r0..r1) {
                q.row_f32(r, t);
            }
        }
    };
    let finish = |r0: usize, r1: usize| {
        if let Weight::Q8 { row_scale, .. } = w {
            // SAFETY: columns r0..r1 of every row belong to this tile alone.
            for i in 0..n {
                let out = unsafe { yp.slice(i * rows + r0, r1 - r0) };
                mul(out, &row_scale[r0..r1]);
            }
        }
    };
    let gemm = |block: &[f32], r0: usize, r1: usize| {
        // SAFETY: the GEMM writes columns r0..r1 of rows 0..n with stride `rows`.
        let out = unsafe { yp.slice(r0, (n - 1) * rows + (r1 - r0)) };
        sgemm_public(n, r1 - r0, cols, 1.0, x, cols, block, cols, true, out, rows);
    };
    if let Weight::Dense(d) = w {
        if cfg!(target_os = "macos") {
            gemm(d, 0, rows);
        } else {
            let step = (TILE_FLOATS / cols).clamp(8, rows.max(8));
            par(pool, rows.div_ceil(step), 1, &|a, b| {
                for t in a..b {
                    let (r0, r1) = (t * step, ((t + 1) * step).min(rows));
                    gemm(&d[r0 * cols..r1 * cols], r0, r1);
                }
            });
        }
        return;
    }
    if cfg!(target_os = "macos") {
        // one tile at a time, widened on the pool, multiplied by Accelerate
        let step = (4 * TILE_FLOATS / cols).clamp(8, rows.max(8));
        let mut tile = vec![0f32; step.min(rows) * cols];
        let mut r0 = 0;
        while r0 < rows {
            let r1 = (r0 + step).min(rows);
            let tp = Shared(tile.as_mut_ptr());
            par(pool, r1 - r0, 32, &|a, b| {
                // SAFETY: disjoint row ranges of the tile.
                let part = unsafe { tp.slice(a * cols, (b - a) * cols) };
                fill(part, r0 + a, r0 + b);
            });
            gemm(&tile[..(r1 - r0) * cols], r0, r1);
            finish(r0, r1);
            r0 = r1;
        }
    } else {
        let step = (TILE_FLOATS / cols).clamp(8, rows.max(8));
        par(pool, rows.div_ceil(step), 1, &|a, b| {
            let mut tile = vec![0f32; step * cols];
            for t in a..b {
                let (r0, r1) = (t * step, ((t + 1) * step).min(rows));
                fill(&mut tile[..(r1 - r0) * cols], r0, r1);
                gemm(&tile[..(r1 - r0) * cols], r0, r1);
                finish(r0, r1);
            }
        });
    }
}

/// `c[m, n] = a[m, k] · b[n, k]ᵀ` on floats, rows of `c` spread over the
/// pool where the GEMM is single-threaded.
pub(super) fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, pool: Option<&Pool>) {
    if m == 0 || n == 0 {
        return;
    }
    if cfg!(target_os = "macos") || pool.is_none() || m * n * k < 1 << 18 {
        sgemm_public(m, n, k, 1.0, a, k, b, k, true, c, n);
        return;
    }
    let cp = Shared(c.as_mut_ptr());
    par(pool, m, 2, &|r0, r1| {
        // SAFETY: disjoint row ranges of `c`.
        let out = unsafe { cp.slice(r0 * n, (r1 - r0) * n) };
        sgemm_public(r1 - r0, n, k, 1.0, &a[r0 * k..r1 * k], k, b, k, true, out, n);
    });
}

/// `c[m, n] = a[m, k] · b[k, n]` on floats.
pub(super) fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, pool: Option<&Pool>) {
    if m == 0 || n == 0 {
        return;
    }
    if cfg!(target_os = "macos") || pool.is_none() || m * n * k < 1 << 18 {
        sgemm_public(m, n, k, 1.0, a, k, b, n, false, c, n);
        return;
    }
    let cp = Shared(c.as_mut_ptr());
    par(pool, m, 2, &|r0, r1| {
        // SAFETY: disjoint row ranges of `c`.
        let out = unsafe { cp.slice(r0 * n, (r1 - r0) * n) };
        sgemm_public(r1 - r0, n, k, 1.0, &a[r0 * k..r1 * k], k, b, n, false, out, n);
    });
}

// ---------------------------------------------------------------------
// the front end's convolutions
// ---------------------------------------------------------------------

/// A `[c, t, f]` volume (channel, time, frequency), row-major.
pub(super) struct Vol {
    pub c: usize,
    pub t: usize,
    pub f: usize,
    pub d: Vec<f32>,
}

/// 3×3 convolution as a product over patches: `out[co, to, fo]`, with
/// `pad_t` zero frames before the first (none after — the front end is
/// causal in time or pads the stride itself), `pad_f` zero bins on both
/// sides, stride `s` on both axes. `w` is `[co, ci, 3, 3]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn conv3x3(
    x: &Vol,
    w: &[f32],
    bias: &[f32],
    co: usize,
    stride: usize,
    pad_t: (usize, usize),
    pad_f: usize,
    pool: Option<&Pool>,
) -> Vol {
    let (ci, t, f) = (x.c, x.t, x.f);
    debug_assert_eq!(w.len(), co * ci * 9);
    let to = (t + pad_t.0 + pad_t.1 - 3) / stride + 1;
    let fo = (f + 2 * pad_f - 3) / stride + 1;
    let k = ci * 9;
    // patches[(to, fo), (ci, kt, kf)], a block of output frames at a time
    let mut out = vec![0f32; co * to * fo];
    let op = Shared(out.as_mut_ptr());
    let block_t = (1usize << 20).div_ceil(fo * k).max(1);
    par(pool, to.div_ceil(block_t), 1, &|a, b| {
        let mut patches = vec![0f32; block_t * fo * k];
        let mut y = vec![0f32; block_t * fo * co];
        for blk in a..b {
            let (t0, t1) = (blk * block_t, ((blk + 1) * block_t).min(to));
            let rows = (t1 - t0) * fo;
            patches[..rows * k].fill(0.0);
            for (ti, tt) in (t0..t1).enumerate() {
                for kt in 0..3 {
                    let st = (tt * stride + kt) as isize - pad_t.0 as isize;
                    if st < 0 || st >= t as isize {
                        continue;
                    }
                    for c in 0..ci {
                        let src = &x.d[(c * t + st as usize) * f..(c * t + st as usize + 1) * f];
                        for ff in 0..fo {
                            let row = &mut patches[(ti * fo + ff) * k + c * 9 + kt * 3..][..3];
                            for (kf, slot) in row.iter_mut().enumerate() {
                                let sf = (ff * stride + kf) as isize - pad_f as isize;
                                if sf >= 0 && (sf as usize) < f {
                                    *slot = src[sf as usize];
                                }
                            }
                        }
                    }
                }
            }
            sgemm_public(rows, co, k, 1.0, &patches[..rows * k], k, w, k, true, &mut y[..rows * co], co);
            for (ti, tt) in (t0..t1).enumerate() {
                for c in 0..co {
                    // SAFETY: output frame `tt` belongs to this block alone.
                    let dst = unsafe { op.slice((c * to + tt) * fo, fo) };
                    for (ff, d) in dst.iter_mut().enumerate() {
                        *d = y[(ti * fo + ff) * co + c] + bias[c];
                    }
                }
            }
        }
    });
    Vol { c: co, t: to, f: fo, d: out }
}

/// Depthwise 3×3 over `[c, t, f]`: two zero frames before the first, one
/// zero bin on both sides (the causal ConvNeXt block). `w` is `[c, 1, 3, 3]`.
pub(super) fn depthwise3x3_causal(x: &Vol, w: &[f32], bias: &[f32], pool: Option<&Pool>) -> Vol {
    let (c, t, f) = (x.c, x.t, x.f);
    let mut out = vec![0f32; c * t * f];
    let op = Shared(out.as_mut_ptr());
    par(pool, c, 8, &|a, b| {
        for ch in a..b {
            let k = &w[ch * 9..ch * 9 + 9];
            // SAFETY: channel `ch` belongs to this range alone.
            let dst = unsafe { op.slice(ch * t * f, t * f) };
            dst.fill(bias[ch]);
            for kt in 0..3 {
                for tt in 0..t {
                    let st = tt as isize + kt as isize - 2;
                    if st < 0 {
                        continue;
                    }
                    let src = &x.d[(ch * t + st as usize) * f..][..f];
                    let row = &mut dst[tt * f..(tt + 1) * f];
                    for kf in 0..3 {
                        let wv = k[kt * 3 + kf];
                        // out[ff] += w · src[ff + kf − 1]
                        let (lo, hi) = (1usize.saturating_sub(kf), (f + 1 - kf).min(f));
                        for ff in lo..hi {
                            row[ff] += wv * src[ff + kf - 1];
                        }
                    }
                }
            }
        }
    });
    Vol { c, t, f, d: out }
}

/// Depthwise 1-D convolution over time on `[n, c]` rows with `(k−1)/2` zero
/// frames on both sides. `w` is `[c, 1, k]`, the result replaces `x`.
pub(super) fn depthwise1d(x: &[f32], n: usize, c: usize, w: &[f32], bias: &[f32], k: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(x.len(), n * c);
    debug_assert_eq!(out.len(), n * c);
    // taps by offset: wt[j][c]
    let mut wt = vec![0f32; k * c];
    for ch in 0..c {
        for j in 0..k {
            wt[j * c + ch] = w[ch * k + j];
        }
    }
    let half = (k - 1) / 2;
    let op = Shared(out.as_mut_ptr());
    par(pool, n, 64, &|a, b| {
        for t in a..b {
            // SAFETY: row `t` belongs to this range alone.
            let dst = unsafe { op.slice(t * c, c) };
            dst.copy_from_slice(bias);
            for j in 0..k {
                let s = t as isize + j as isize - half as isize;
                if s < 0 || s >= n as isize {
                    continue;
                }
                let src = &x[s as usize * c..(s as usize + 1) * c];
                let taps = &wt[j * c..(j + 1) * c];
                for ((d, &v), &wv) in dst.iter_mut().zip(src).zip(taps) {
                    *d += v * wv;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_the_library() {
        let mut worst = 0f32;
        let mut x = -30.0f32;
        while x < 30.0 {
            let (a, b) = (exp_f(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst relative error {worst}");
        assert_eq!(exp_f(0.0), 1.0);
        assert!(exp_f(-200.0) > 0.0 && exp_f(-200.0) < 1e-37);
    }

    #[test]
    fn activations_match_scalar_definitions() {
        let x0: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.61).collect();
        let g: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
        let sg = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut x = x0.clone();
        silu(&mut x);
        for (a, &v) in x.iter().zip(&x0) {
            assert!((a - v * sg(v)).abs() < 1e-5);
        }
        let mut x = x0.clone();
        mul_sigmoid(&mut x, &g);
        for ((a, &v), &g) in x.iter().zip(&x0).zip(&g) {
            assert!((a - v * sg(g)).abs() < 1e-5);
        }
        let mut x = x0.clone();
        softmax_scaled(&mut x, 0.25);
        let z: f32 = x0.iter().map(|v| (v * 0.25).exp()).sum();
        for (a, &v) in x.iter().zip(&x0) {
            assert!((a - (v * 0.25).exp() / z).abs() < 1e-6);
        }
    }

    #[test]
    fn q8_product_matches_the_float_product() {
        let (rows, cols, n) = (37usize, 24usize, 5usize);
        let codes: Vec<u8> = (0..rows * cols).map(|i| ((i * 37 + 11) % 255) as u8).collect();
        let row_scale: Vec<f32> = (0..rows).map(|r| 0.01 + r as f32 * 1e-3).collect();
        let col_field: Vec<f32> = (0..cols).map(|c| 0.5 + c as f32 * 0.07).collect();
        let x: Vec<f32> = (0..n * cols).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.3).collect();
        let mut dense = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dense[r * cols + c] = codes[r * cols + c] as i8 as f32 * row_scale[r] * col_field[c];
            }
        }
        let mut want = vec![0f32; n * rows];
        for i in 0..n {
            for r in 0..rows {
                want[i * rows + r] = (0..cols).map(|c| x[i * cols + c] as f64 * dense[r * cols + c] as f64).sum::<f64>() as f32;
            }
        }
        let pool = Pool::new(3);
        for p in [None, Some(&pool)] {
            let mut got = vec![0f32; n * rows];
            matmul(&Weight::Q8 { codes: &codes, row_scale: &row_scale, col_field: &col_field }, rows, cols, &x, n, &mut got, p);
            let mut got_dense = vec![0f32; n * rows];
            matmul(&Weight::Dense(&dense), rows, cols, &x, n, &mut got_dense, p);
            for ((a, b), w) in got.iter().zip(&got_dense).zip(&want) {
                assert!((a - w).abs() < 1e-3 * (1.0 + w.abs()), "q8 {a} vs {w}");
                assert!((b - w).abs() < 1e-3 * (1.0 + w.abs()), "dense {b} vs {w}");
            }
        }
    }

    #[test]
    fn conv3x3_matches_the_direct_sum() {
        let (ci, t, f, co) = (2usize, 7usize, 9usize, 3usize);
        let x = Vol { c: ci, t, f, d: (0..ci * t * f).map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.1).collect() };
        let w: Vec<f32> = (0..co * ci * 9).map(|i| ((i * 13 % 11) as f32 - 5.0) * 0.05).collect();
        let bias = vec![0.1f32, -0.2, 0.3];
        for (stride, pad_t, pad_f) in [(1usize, (2usize, 0usize), 1usize), (2, (1, 1), 0)] {
            let y = conv3x3(&x, &w, &bias, co, stride, pad_t, pad_f, None);
            let to = (t + pad_t.0 + pad_t.1 - 3) / stride + 1;
            let fo = (f + 2 * pad_f - 3) / stride + 1;
            assert_eq!((y.t, y.f), (to, fo));
            for c in 0..co {
                for tt in 0..to {
                    for ff in 0..fo {
                        let mut s = bias[c];
                        for i in 0..ci {
                            for kt in 0..3 {
                                for kf in 0..3 {
                                    let st = (tt * stride + kt) as isize - pad_t.0 as isize;
                                    let sf = (ff * stride + kf) as isize - pad_f as isize;
                                    if st >= 0 && (st as usize) < t && sf >= 0 && (sf as usize) < f {
                                        s += w[((c * ci + i) * 3 + kt) * 3 + kf] * x.d[(i * t + st as usize) * f + sf as usize];
                                    }
                                }
                            }
                        }
                        assert!((y.d[(c * to + tt) * fo + ff] - s).abs() < 1e-5);
                    }
                }
            }
        }
    }
}
