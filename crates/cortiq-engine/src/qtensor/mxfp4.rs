//! CPU kernels for `mxfp4` (dtype 21): E2M1 code plane + E8M0 scale plane
//! (`cortiq_core::mxfp4`). Two contracts, the same split every 4-bit codec
//! here has:
//!
//! - **exact** (`CMF_SDOT=0`, float-activation scopes, no dot-product ISA):
//!   every weight is dequantized EXACTLY (`E2M1_LUT[code] · 2^(k−127)`) and
//!   accumulated as `sum += w·x[j]` over `j = 0..cols` in order — the
//!   reference F32 matvec's (`pool::matvec_rows`) reduction, so an `mxfp4`
//!   matrix computes BIT FOR BIT what its dequantized f32 matrix computes.
//! - **A8W8** (default on ARM dotprod / x86 AVX2): the activation goes
//!   through the shared int8 split (`split_act`, outliers exact), each code
//!   becomes the exact int8 `2·E2M1` (`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`) by
//!   one table lookup (`tbl` / `pshufb`), a group is one 32-wide int dot,
//!   and the factor two folds into the scale (`2^(k−128)`). The only
//!   approximation is the activation's int8 rounding, as for q4tp.
//!
//! `matmat` keeps per-token semantics identical to `matvec` (each token is
//! one matvec, bit for bit), with the row's codes decoded once per batch.

use super::*;
use cortiq_core::mxfp4::{
    E2M1_LUT, E2M1_X2_I8, GROUP, GROUP_BYTES, e8m0_half_to_f32, e8m0_to_f32, scale_offset,
};

/// Borrowed view of an `mxfp4` payload.
#[derive(Clone, Copy)]
pub(super) struct Mx<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    gpr: usize,
}

impl<'a> Mx<'a> {
    pub(super) fn new(bytes: &'a [u8], rows: usize, cols: usize) -> Self {
        let off = scale_offset(rows, cols);
        let gpr = cols / GROUP;
        assert!(
            bytes.len() >= off + rows * gpr,
            "mxfp4 payload {} < {} for [{rows}, {cols}]",
            bytes.len(),
            off + rows * gpr
        );
        Self {
            codes: &bytes[..off],
            scales: &bytes[off..off + rows * gpr],
            gpr,
        }
    }

    #[inline]
    fn row_codes(&self, r: usize) -> &'a [u8] {
        &self.codes[r * self.gpr * GROUP_BYTES..(r + 1) * self.gpr * GROUP_BYTES]
    }

    #[inline]
    fn row_scales(&self, r: usize) -> &'a [u8] {
        &self.scales[r * self.gpr..(r + 1) * self.gpr]
    }

    /// Dequantize row `r` exactly into `dst[..cols]`.
    pub(super) fn row_f32(&self, r: usize, dst: &mut [f32]) {
        let (codes, scales) = (self.row_codes(r), self.row_scales(r));
        for g in 0..self.gpr {
            let s = e8m0_to_f32(scales[g]);
            for j in 0..GROUP_BYTES {
                let b = codes[g * GROUP_BYTES + j];
                dst[g * GROUP + 2 * j] = E2M1_LUT[(b & 0x0F) as usize] * s;
                dst[g * GROUP + 2 * j + 1] = E2M1_LUT[(b >> 4) as usize] * s;
            }
        }
    }

    /// Exact dot of row `r` with `x` in `pool::matvec_rows`'s order.
    #[inline]
    fn row_exact(&self, r: usize, x: &[f32]) -> f32 {
        let (codes, scales) = (self.row_codes(r), self.row_scales(r));
        let mut sum = 0.0f32;
        for g in 0..self.gpr {
            let s = e8m0_to_f32(scales[g]);
            let xg = &x[g * GROUP..(g + 1) * GROUP];
            for j in 0..GROUP_BYTES {
                let b = codes[g * GROUP_BYTES + j];
                sum += (E2M1_LUT[(b & 0x0F) as usize] * s) * xg[2 * j];
                sum += (E2M1_LUT[(b >> 4) as usize] * s) * xg[2 * j + 1];
            }
        }
        sum
    }

    /// Four rows at once: four independent chains (latency), each the
    /// exact sequential sum of `row_exact`.
    #[inline]
    fn rows4_exact(&self, r: usize, x: &[f32]) -> [f32; 4] {
        let c = [
            self.row_codes(r),
            self.row_codes(r + 1),
            self.row_codes(r + 2),
            self.row_codes(r + 3),
        ];
        let sc = [
            self.row_scales(r),
            self.row_scales(r + 1),
            self.row_scales(r + 2),
            self.row_scales(r + 3),
        ];
        let mut a = [0.0f32; 4];
        for g in 0..self.gpr {
            let s = [
                e8m0_to_f32(sc[0][g]),
                e8m0_to_f32(sc[1][g]),
                e8m0_to_f32(sc[2][g]),
                e8m0_to_f32(sc[3][g]),
            ];
            let xg = &x[g * GROUP..(g + 1) * GROUP];
            for j in 0..GROUP_BYTES {
                let (x0, x1) = (xg[2 * j], xg[2 * j + 1]);
                for k in 0..4 {
                    let b = c[k][g * GROUP_BYTES + j];
                    a[k] += (E2M1_LUT[(b & 0x0F) as usize] * s[k]) * x0;
                    a[k] += (E2M1_LUT[(b >> 4) as usize] * s[k]) * x1;
                }
            }
        }
        a
    }

    /// One weight and its scale — the A8W8 outlier term.
    #[inline]
    fn weight(&self, r: usize, j: usize) -> (f32, f32) {
        let b = self.codes[r * self.gpr * GROUP_BYTES + j / 2];
        let code = if j % 2 == 0 { b & 0x0F } else { b >> 4 };
        (
            E2M1_LUT[code as usize],
            e8m0_to_f32(self.scales[r * self.gpr + j / GROUP]),
        )
    }

    /// Row `r` against int8 activations: Σ_g (Σ 2·E2M1·xq) · 2^(k−128).
    #[inline]
    fn dot_row_i8(&self, r: usize, xq: &[i8]) -> f32 {
        dot_i8(self.row_codes(r), self.row_scales(r), xq)
    }
}

/// Group-scaled int8 dot of one row's codes. Every implementation forms the
/// same exact i32 per group and accumulates `acc += isum · scale` in group
/// order, so all of them (and the decoded-row twin below) agree bit for bit.
#[inline]
#[allow(unreachable_code)]
fn dot_i8(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_i8_sdot(codes, scales, xq);
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        return unsafe { dot_i8_avx2(codes, scales, xq) };
    }
    dot_i8_scalar(codes, scales, xq)
}

fn dot_i8_scalar(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    let mut acc = 0f32;
    for (g, &k) in scales.iter().enumerate() {
        let mut d = 0i32;
        for j in 0..GROUP_BYTES {
            let b = codes[g * GROUP_BYTES + j];
            d += E2M1_X2_I8[(b & 0x0F) as usize] as i32 * xq[g * GROUP + 2 * j] as i32
                + E2M1_X2_I8[(b >> 4) as usize] as i32 * xq[g * GROUP + 2 * j + 1] as i32;
        }
        acc += d as f32 * e8m0_half_to_f32(k);
    }
    acc
}

/// NEON: `tbl` turns 16 nibbles into 16 int8 weights, `zip` restores the
/// element order, two `sdot`s per group. Gated by `a8w8_enabled()` (which
/// checks dotprod) like every other SDOT kernel.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_sdot(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    // SAFETY: codes.len() == 16·gpr, xq.len() >= 32·gpr (callers).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let table = vld1q_s8(E2M1_X2_I8.as_ptr());
        let lomask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        for (g, &k) in scales.iter().enumerate() {
            let b = vld1q_u8(codes.as_ptr().add(g * GROUP_BYTES));
            let lo = vqtbl1q_s8(table, vandq_u8(b, lomask));
            let hi = vqtbl1q_s8(table, vshrq_n_u8::<4>(b));
            let e0 = vzip1q_s8(lo, hi);
            let e1 = vzip2q_s8(lo, hi);
            let x0 = vld1q_s8(xq.as_ptr().add(g * GROUP));
            let x1 = vld1q_s8(xq.as_ptr().add(g * GROUP + 16));
            let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
            asm!(
                "sdot {a0:v}.4s, {e0:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {e1:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                e0 = in(vreg) e0, x0 = in(vreg) x0, e1 = in(vreg) e1, x1 = in(vreg) x1,
                options(pure, nomem, nostack),
            );
            acc += vaddvq_s32(vaddq_s32(a0, a1)) as f32 * e8m0_half_to_f32(k);
        }
        acc
    }
}

/// AVX2: `pshufb` table lookup, then the `maddubs(|w|, sign(x, w))` trick.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    // SAFETY: see dot_i8_sdot.
    unsafe {
        use core::arch::x86_64::*;
        let table = _mm_loadu_si128(E2M1_X2_I8.as_ptr() as *const __m128i);
        let lomask = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        for (g, &k) in scales.iter().enumerate() {
            let b = _mm_loadu_si128(codes.as_ptr().add(g * GROUP_BYTES) as *const __m128i);
            let lo = _mm_shuffle_epi8(table, _mm_and_si128(b, lomask));
            let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(b), lomask));
            let w = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            let x = _mm256_loadu_si256(xq.as_ptr().add(g * GROUP) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let d = _mm256_madd_epi16(p16, ones);
            let hi128 = _mm256_extracti128_si256::<1>(d);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            acc += _mm_cvtsi128_si32(s32) as f32 * e8m0_half_to_f32(k);
        }
        acc
    }
}

/// Decode row `r`'s codes into int8 weights `2·E2M1` (`dst[..cols]`).
#[inline]
fn decode_row_i8(m: &Mx<'_>, r: usize, dst: &mut [i8]) {
    for (j, &b) in m.row_codes(r).iter().enumerate() {
        dst[2 * j] = E2M1_X2_I8[(b & 0x0F) as usize];
        dst[2 * j + 1] = E2M1_X2_I8[(b >> 4) as usize];
    }
}

/// `dot_i8` over an already-decoded int8 row: the same per-group i32 and
/// the same `acc += isum · scale` order, so it equals `dot_i8` bit for bit.
#[inline]
fn dot_decoded_i8(w: &[i8], scales: &[u8], xq: &[i8]) -> f32 {
    let wu: &[u8] = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) };
    // The ISA test of `dot_i8` (sdot on ARM, AVX2 on x86), not the
    // thread-local contract switch: this runs on pool workers.
    #[cfg(target_arch = "aarch64")]
    let simd = true;
    #[cfg(target_arch = "x86_64")]
    let simd = avx2_enabled();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd = false;
    let mut acc = 0f32;
    if simd {
        for (g, &k) in scales.iter().enumerate() {
            let d = dot_i8_i8(
                &wu[g * GROUP..(g + 1) * GROUP],
                &xq[g * GROUP..(g + 1) * GROUP],
            );
            acc += d as f32 * e8m0_half_to_f32(k);
        }
    } else {
        for (g, &k) in scales.iter().enumerate() {
            let mut d = 0i32;
            for j in 0..GROUP {
                d += w[g * GROUP + j] as i32 * xq[g * GROUP + j] as i32;
            }
            acc += d as f32 * e8m0_half_to_f32(k);
        }
    }
    acc
}

/// Whether this call takes the A8W8 contract. Decided ONCE on the calling
/// thread (float-activation scopes are thread-local), never per worker.
#[inline]
fn use_a8w8() -> bool {
    a8w8_enabled()
}

/// `out[r] = W[r,:] · x` for every row.
pub(super) fn matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert!(out.len() >= rows && x.len() >= cols);
    let m = Mx::new(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    if use_a8w8() {
        let act = split_act(&x[..cols]);
        let run = |start: usize, end: usize| {
            for r in start..end {
                let mut acc = m.dot_row_i8(r, &act.xq) * act.sx;
                for &(j, xv) in &act.outliers {
                    let (w, s) = m.weight(r, j);
                    acc += w * s * xv;
                }
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = acc };
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        let mut r = start;
        while end - r >= 4 {
            let a = m.rows4_exact(r, x);
            for (k, v) in a.into_iter().enumerate() {
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r + k) = v };
            }
            r += 4;
        }
        while r < end {
            // SAFETY: disjoint row ranges per worker.
            unsafe { *out_addr.at(r) = m.row_exact(r, x) };
            r += 1;
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Two activations against the same weights (MTP verify pair): each output
/// equals its own `matvec` bit for bit; the row is read once for both.
#[allow(clippy::too_many_arguments)]
pub(super) fn matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    let m = Mx::new(bytes, rows, cols);
    let (p1, p2) = (SendMut(o1.as_mut_ptr()), SendMut(o2.as_mut_ptr()));
    if use_a8w8() {
        let (a1, a2) = (split_act(&x1[..cols]), split_act(&x2[..cols]));
        let run = |start: usize, end: usize| {
            for r in start..end {
                for (act, p) in [(&a1, p1), (&a2, p2)] {
                    let mut acc = m.dot_row_i8(r, &act.xq) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (w, s) = m.weight(r, j);
                        acc += w * s * xv;
                    }
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *p.at(r) = acc };
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        for r in start..end {
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = m.row_exact(r, x1);
                *p2.at(r) = m.row_exact(r, x2);
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Batched matvec: `xs` `[b, cols]` row-major → `out` `[b, rows]`. Token
/// `t`'s outputs are `matvec(xs[t])`'s bit for bit; a row's codes are
/// decoded once and spent on the whole batch.
#[allow(clippy::too_many_arguments)]
pub(super) fn matmat(
    bytes: &[u8],
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert!(xs.len() >= b * cols && out.len() >= b * rows);
    let m = Mx::new(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    if use_a8w8() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|t| split_act(&xs[t * cols..(t + 1) * cols]))
            .collect();
        let run = |start: usize, end: usize| {
            let mut w = vec![0i8; cols];
            for r in start..end {
                decode_row_i8(&m, r, &mut w);
                let sc = m.row_scales(r);
                for (t, act) in acts.iter().enumerate() {
                    let mut acc = dot_decoded_i8(&w, sc, &act.xq) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (wv, s) = m.weight(r, j);
                        acc += wv * s * xv;
                    }
                    // SAFETY: disjoint (t, r) cells per worker range.
                    unsafe { *out_addr.at(t * rows + r) = acc };
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        let mut w = vec![0f32; cols];
        for r in start..end {
            m.row_f32(r, &mut w);
            for t in 0..b {
                let x = &xs[t * cols..(t + 1) * cols];
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += w[j] * x[j];
                }
                // SAFETY: disjoint (t, r) cells per worker range.
                unsafe { *out_addr.at(t * rows + r) = sum };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let codes: Vec<u8> = (0..rows * cols / 2)
            .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed * 977) >> 7) as u8)
            .collect();
        let scales: Vec<u8> = (0..rows * cols / GROUP)
            .map(|i| 118 + ((i * 7 + seed) % 13) as u8)
            .collect();
        cortiq_core::mxfp4::repack(&codes, &scales, rows, cols).unwrap()
    }

    fn act(cols: usize, seed: usize) -> Vec<f32> {
        (0..cols)
            .map(|j| {
                let v = ((j * 37 + seed * 101) % 211) as f32 / 211.0 - 0.5;
                // On wide rows, sparse large activations (past split_act's
                // 8·rms threshold) exercise the exact outlier term; on a
                // short row one such value would BE the rms and stay in
                // the int8 bulk, which only tests int8 rounding.
                if cols >= 1024 && j % 97 == 13 {
                    30.0f32.copysign(v)
                } else {
                    v
                }
            })
            .collect()
    }

    fn dequant(p: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0f32; rows * cols];
        cortiq_core::mxfp4::dequant(p, rows, cols, &mut w);
        w
    }

    /// f64 reference of the dequantized matrix.
    fn reference(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|j| w[r * cols + j] as f64 * x[j] as f64)
                    .sum()
            })
            .collect()
    }

    /// The exact contract: bit for bit the reference F32 matvec
    /// (`matvec_rows`) over the dequantized matrix, with and without a pool,
    /// and within f32 summation noise of an f64 reference.
    #[test]
    fn exact_matvec_equals_f32_reference_bit_for_bit() {
        let pool = Pool::new(4);
        for (rows, cols) in [(1usize, 32usize), (7, 64), (300, 160), (259, 2304)] {
            let p = payload(rows, cols, rows);
            let w = dequant(&p, rows, cols);
            let x = act(cols, cols);
            let mut want = vec![0f32; rows];
            matvec_rows(None, &w, &x, &mut want);
            for pl in [None, Some(&pool)] {
                let mut got = vec![0f32; rows];
                float_activations_scope(|| matvec(&p, &x, rows, cols, &mut got, pl));
                assert!(
                    want.iter()
                        .zip(&got)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{rows}x{cols} pool={}",
                    pl.is_some()
                );
            }
            let r64 = reference(&w, &x, rows, cols);
            let norm: f64 = x.iter().map(|v| (*v as f64).abs()).sum::<f64>()
                * w.iter().fold(0f64, |m, v| m.max((*v as f64).abs()));
            for (g, r) in want.iter().zip(&r64) {
                assert!(
                    (*g as f64 - r).abs() <= norm * cols as f64 * 1.2e-7,
                    "{g} vs {r}"
                );
            }
            // matvec2 and matmat keep per-token equality in the exact mode.
            let x2 = act(cols, cols + 3);
            let (mut a, mut b2) = (vec![0f32; rows], vec![0f32; rows]);
            float_activations_scope(|| {
                matvec2(&p, &x, &x2, rows, cols, &mut a, &mut b2, Some(&pool))
            });
            let mut want2 = vec![0f32; rows];
            matvec_rows(None, &w, &x2, &mut want2);
            assert_eq!(a, want);
            assert_eq!(b2, want2);
            let bt = 3usize;
            let xs: Vec<f32> = (0..bt).flat_map(|t| act(cols, t * 5 + 1)).collect();
            let mut mm = vec![0f32; bt * rows];
            float_activations_scope(|| matmat(&p, &xs, bt, rows, cols, &mut mm, Some(&pool)));
            for t in 0..bt {
                let mut one = vec![0f32; rows];
                matvec_rows(None, &w, &xs[t * cols..(t + 1) * cols], &mut one);
                assert!(
                    one.iter()
                        .zip(&mm[t * rows..(t + 1) * rows])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    /// A memory-mapped `mxfp4` tensor from a real CMF file: it maps (no f32
    /// copy), declines every device-graph accessor, and under the exact
    /// contract every QTensor entry point equals its dequantized F32 twin
    /// bit for bit (matvec, matvec2, matmat, matvec_many's per-tensor
    /// fallback, the embedding row).
    #[test]
    fn mapped_tensor_paths_equal_the_dequantized_f32_twin() {
        use cortiq_core::{CmfHeader, TensorSpec};
        let (rows, cols) = (300usize, 512usize);
        let p = payload(rows, cols, 9);
        let p2 = payload(rows, cols, 10);
        let w = dequant(&p, rows, cols);
        let w2 = dequant(&p2, rows, cols);
        let header: CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf", "version": cortiq_core::CMF_VERSION,
            "arch": {"arch_name": "tiny-mxfp4", "hidden_size": cols,
                "intermediate_size": rows, "num_layers": 1, "num_attention_heads": 2,
                "num_kv_heads": 1, "head_dim": 128, "vocab_size": rows,
                "layer_types": ["FullAttention"], "rms_norm_eps": 1e-6,
                "max_position_embeddings": 8},
            "quant_type": "VBIT"
        }))
        .unwrap();
        let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let spec = |name: &str, dtype, data: Vec<u8>| TensorSpec {
            name: name.into(),
            dtype,
            shape: vec![rows, cols],
            data,
        };
        let specs = [
            spec("q", TensorDtype::Mxfp4, p.clone()),
            spec("q2", TensorDtype::Mxfp4, p2.clone()),
            spec("f", TensorDtype::F32, f32b(&w)),
            spec("f2", TensorDtype::F32, f32b(&w2)),
        ];
        let dir = std::env::temp_dir().join(format!("cmf-mxfp4-mapped-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.cmf");
        CmfModel::write(&path, &header, &specs, None, None).unwrap();
        let model = Arc::new(CmfModel::open(&path).unwrap());
        assert!(model.verify().is_empty(), "{:?}", model.verify());
        let t = |n: &str| QTensor::from_model(&model, n).unwrap();
        let (q, q2, f, f2) = (t("q"), t("q2"), t("f"), t("f2"));
        assert_eq!(q.model_dtype(), Some(TensorDtype::Mxfp4));
        assert!(
            q.is_mxfp4() && q.as_f32().is_none(),
            "mapped, not dequantized"
        );
        assert!(q.graph_weight().is_none() && q.q1_parts().is_none());
        assert!(q.mapped_q4tp().is_none() && q.mapped_device_gemm().is_none());
        let pool = Pool::new(3);
        let x = act(cols, 4);
        let x2 = act(cols, 5);
        float_activations_scope(|| {
            let mv = |a: &QTensor, x: &[f32]| {
                let mut o = vec![0f32; rows];
                a.matvec(x, &mut o, Some(&pool));
                o
            };
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(mv(&q, &x)), bits(mv(&f, &x)), "matvec");
            let (mut a1, mut a2) = (vec![0f32; rows], vec![0f32; rows]);
            q.matvec2(&x, &x2, &mut a1, &mut a2, Some(&pool));
            assert_eq!(bits(a1), bits(mv(&f, &x)), "matvec2 first");
            assert_eq!(bits(a2), bits(mv(&f, &x2)), "matvec2 second");
            let (mut m1, mut m2) = (vec![0f32; rows], vec![0f32; rows]);
            QTensor::matvec_many([&q, &q2], &x, [&mut m1, &mut m2], Some(&pool));
            assert_eq!(bits(m1), bits(mv(&f, &x)), "matvec_many first");
            assert_eq!(bits(m2), bits(mv(&f2, &x)), "matvec_many second");
            let b = 5usize;
            let xs: Vec<f32> = (0..b).flat_map(|k| act(cols, k + 20)).collect();
            let (mut mq, mut mf) = (vec![0f32; b * rows], vec![0f32; b * rows]);
            q.matmat(&xs, b, &mut mq, Some(&pool));
            f.matmat(&xs, b, &mut mf, Some(&pool));
            assert_eq!(bits(mq), bits(mf), "matmat");
        });
        // The fused SwiGLU kernel refuses mxfp4 (per-tensor fallback).
        let mut o = vec![0f32; rows];
        assert!(!QTensor::matvec_silu_mul(&q, &q2, &x, &mut o, Some(&pool)));
        for r in [0, 1, rows / 2, rows - 1] {
            let mut row = vec![0f32; cols];
            q.row_f32(r, &mut row);
            assert_eq!(row, w[r * cols..(r + 1) * cols], "embedding row {r}");
        }
        drop((q, q2, f, f2, model));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CPU speed of the mxfp4 kernels against q4tp on DeepSeek-V4.1 expert
    /// shapes (2304×5120 gate/up, 5120×2304 down). Measurement only:
    /// `cargo test -p cortiq-engine --release --lib
    ///  qtensor::mxfp4::tests::cpu_matvec_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cpu_matvec_speed() {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let pool = Pool::new(threads);
        let best = |f: &mut dyn FnMut()| {
            for _ in 0..3 {
                f();
            }
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    f();
                }
                best = best.min(t.elapsed().as_secs_f64() / 20.0);
            }
            best * 1e6
        };
        for (rows, cols) in [(2304usize, 5120usize), (5120, 2304)] {
            let p = payload(rows, cols, 1);
            // q4tp of the same shape: random nibbles and codes, a valid
            // ladder (kernel speed does not depend on the values).
            let gpr = cols / GROUP;
            let stride = cortiq_core::quant::q4tp_code_stride(gpr);
            let mut q4 = bytes_of(rows * gpr * 16, 3);
            for _ in 0..rows {
                q4.extend_from_slice(&cortiq_core::quant::f32_to_f16(-6.0).to_le_bytes());
                q4.extend_from_slice(&cortiq_core::quant::f32_to_f16(0.1).to_le_bytes());
            }
            q4.extend(bytes_of(rows * stride, 4).iter().map(|b| b & 0x7F));
            let x = act(cols, 2);
            let mut out = vec![0f32; rows];
            let mx_a8 = best(&mut || matvec(&p, &x, rows, cols, &mut out, Some(&pool)));
            let mx_ex = best(&mut || {
                float_activations_scope(|| matvec(&p, &x, rows, cols, &mut out, Some(&pool)))
            });
            let q4_a8 = best(&mut || q4tp_matvec(&q4, &x, rows, cols, &mut out, Some(&pool)));
            let mx_1t = best(&mut || matvec(&p, &x, rows, cols, &mut out, None));
            let q4_1t = best(&mut || q4tp_matvec(&q4, &x, rows, cols, &mut out, None));
            let gb = |us: f64, len: usize| len as f64 / us / 1e3;
            eprintln!(
                "cpu x{threads} a8w8={} {rows}x{cols}: mxfp4 a8w8 {mx_a8:7.1} us ({:5.1} GB/s) | mxfp4 exact {mx_ex:7.1} us | q4tp a8w8 {q4_a8:7.1} us ({:5.1} GB/s) | 1 thread: mxfp4 {mx_1t:7.1} us, q4tp {q4_1t:7.1} us",
                a8w8_enabled(),
                gb(mx_a8, p.len()),
                gb(q4_a8, q4.len()),
            );
        }
    }

    fn bytes_of(n: usize, seed: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed * 7919) >> 11) as u8)
            .collect()
    }

    /// The A8W8 contract: every kernel (SIMD, scalar, decoded-row) gives the
    /// same f32 bit for bit, matmat equals matvec per token, and the result
    /// stays within the int8 activation rounding of the f64 reference.
    #[test]
    fn a8w8_kernels_agree_and_track_the_reference() {
        let pool = Pool::new(4);
        for (rows, cols) in [(5usize, 64usize), (300, 2304), (64, 5120)] {
            let p = payload(rows, cols, cols);
            let m = Mx::new(&p, rows, cols);
            let x = act(cols, rows);
            let a = split_act(&x);
            let mut w = vec![0i8; cols];
            for r in 0..rows {
                let simd = m.dot_row_i8(r, &a.xq);
                let scalar = dot_i8_scalar(m.row_codes(r), m.row_scales(r), &a.xq);
                decode_row_i8(&m, r, &mut w);
                let dec = dot_decoded_i8(&w, m.row_scales(r), &a.xq);
                assert_eq!(simd.to_bits(), scalar.to_bits(), "row {r}");
                assert_eq!(simd.to_bits(), dec.to_bits(), "row {r}");
            }
            // f64 emulation of the A8W8 contract itself (int8 activations,
            // exact outliers): the kernel must match it to f32 rounding.
            let wf0 = dequant(&p, rows, cols);
            for r in 0..rows {
                let mut e = 0f64;
                for j in 0..cols {
                    e += wf0[r * cols + j] as f64 * a.xq[j] as f64 * a.sx as f64;
                }
                for &(j, xv) in &a.outliers {
                    e += wf0[r * cols + j] as f64 * xv as f64;
                }
                let mut acc = m.dot_row_i8(r, &a.xq) * a.sx;
                for &(j, xv) in &a.outliers {
                    let (w, s) = m.weight(r, j);
                    acc += w * s * xv;
                }
                let mag: f64 = (0..cols)
                    .map(|j| (wf0[r * cols + j] as f64 * x[j] as f64).abs())
                    .sum();
                assert!(
                    (acc as f64 - e).abs() <= mag * 1e-5,
                    "row {r}: {acc} vs emulated {e}"
                );
            }
            if !a8w8_enabled() {
                continue; // no dot-product ISA here: matvec takes the exact path
            }
            let mut mv = vec![0f32; rows];
            matvec(&p, &x, rows, cols, &mut mv, Some(&pool));
            let bt = 5usize;
            let xs: Vec<f32> = (0..bt)
                .flat_map(|t| if t == 2 { x.clone() } else { act(cols, t + 11) })
                .collect();
            let mut mm = vec![0f32; bt * rows];
            matmat(&p, &xs, bt, rows, cols, &mut mm, Some(&pool));
            assert!(
                mv.iter()
                    .zip(&mm[2 * rows..3 * rows])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let wf = dequant(&p, rows, cols);
            let r64 = reference(&wf, &x, rows, cols);
            let scale = r64.iter().fold(0f64, |m, v| m.max(v.abs()));
            for (g, r) in mv.iter().zip(&r64) {
                assert!(
                    (*g as f64 - r).abs() <= 0.02 * scale,
                    "{g} vs {r} (scale {scale})"
                );
            }
        }
    }
}
