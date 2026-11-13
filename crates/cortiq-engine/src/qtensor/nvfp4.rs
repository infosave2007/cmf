//! CPU kernels for `nvfp4` (dtype 22): E2M1 code plane + UE4M3 scale plane
//! per 16 elements + one f32 tensor factor (`cortiq_core::nvfp4`). The same
//! two contracts as `mxfp4`:
//!
//! - **exact** (`CMF_SDOT=0`, float-activation scopes, no dot-product ISA):
//!   a 16-group's partial sum `Σ E2M1·x` in element order, times the group's
//!   UE4M3 scale, accumulated in group order; the tensor factor multiplies
//!   the row's sum once.
//! - **A8W8** (default with a dot-product ISA): the activation goes through
//!   the shared int8 split (`split_act`, outliers exact), each code becomes
//!   the exact int8 `2·E2M1`, a 16-group is one int dot, and the factor two
//!   folds into the scale (`UE4M3 / 2`).
//!
//! `matmat` keeps per-token semantics identical to `matvec` (each token is
//! one matvec, bit for bit), with the row's codes decoded once per batch.

use super::*;
use cortiq_core::nvfp4::{E2M1_LUT, E2M1_X2_I8, GROUP, GROUP_BYTES, UE4M3, UE4M3_HALF, global, scale_offset};

/// Borrowed view of an `nvfp4` payload.
#[derive(Clone, Copy)]
pub(super) struct Nv<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    gpr: usize,
    g: f32,
}

impl<'a> Nv<'a> {
    pub(super) fn new(bytes: &'a [u8], rows: usize, cols: usize) -> Self {
        let off = scale_offset(rows, cols);
        let gpr = cols / GROUP;
        assert!(
            bytes.len() >= off + rows * gpr + 4,
            "nvfp4 payload {} < {} for [{rows}, {cols}]",
            bytes.len(),
            off + rows * gpr + 4
        );
        Self {
            codes: &bytes[..off],
            scales: &bytes[off..off + rows * gpr],
            gpr,
            g: global(bytes, rows, cols),
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

    /// Dequantize row `r` into `dst[..cols]` (`cortiq_core::nvfp4::dequant`'s values).
    pub(super) fn row_f32(&self, r: usize, dst: &mut [f32]) {
        let (codes, scales) = (self.row_codes(r), self.row_scales(r));
        for k in 0..self.gpr {
            let s = UE4M3[scales[k] as usize];
            for j in 0..GROUP_BYTES {
                let b = codes[k * GROUP_BYTES + j];
                dst[k * GROUP + 2 * j] = (E2M1_LUT[(b & 0x0F) as usize] * s) * self.g;
                dst[k * GROUP + 2 * j + 1] = (E2M1_LUT[(b >> 4) as usize] * s) * self.g;
            }
        }
    }

    /// Exact contract: per-group float partials, scaled, tensor factor last.
    #[inline]
    fn row_exact(&self, r: usize, x: &[f32]) -> f32 {
        let (codes, scales) = (self.row_codes(r), self.row_scales(r));
        let mut sum = 0.0f32;
        for k in 0..self.gpr {
            let xg = &x[k * GROUP..(k + 1) * GROUP];
            let cg = &codes[k * GROUP_BYTES..(k + 1) * GROUP_BYTES];
            let mut p = 0.0f32;
            for j in 0..GROUP_BYTES {
                let b = cg[j];
                p += E2M1_LUT[(b & 0x0F) as usize] * xg[2 * j];
                p += E2M1_LUT[(b >> 4) as usize] * xg[2 * j + 1];
            }
            sum += p * UE4M3[scales[k] as usize];
        }
        sum * self.g
    }

    /// One weight without the tensor factor — the A8W8 outlier term.
    #[inline]
    fn weight(&self, r: usize, j: usize) -> f32 {
        let b = self.codes[r * self.gpr * GROUP_BYTES + j / 2];
        let code = if j % 2 == 0 { b & 0x0F } else { b >> 4 };
        E2M1_LUT[code as usize] * UE4M3[self.scales[r * self.gpr + j / GROUP] as usize]
    }

    /// Row `r` against int8 activations: Σ_g (Σ 2·E2M1·xq) · UE4M3/2.
    #[inline]
    fn dot_row_i8(&self, r: usize, xq: &[i8]) -> f32 {
        dot_i8(self.row_codes(r), self.row_scales(r), xq)
    }
}

/// Group-scaled int8 dot of one row's codes; every implementation forms the
/// same exact i32 per 16-group and accumulates `acc += isum · scale` in group
/// order, so all of them agree bit for bit.
#[inline]
#[allow(unreachable_code)]
fn dot_i8(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() && scales.len() % 2 == 0 {
        return unsafe { dot_i8_avx2(codes, scales, xq) };
    }
    dot_i8_scalar(codes, scales, xq)
}

fn dot_i8_scalar(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    let mut acc = 0f32;
    for (k, &s) in scales.iter().enumerate() {
        let mut d = 0i32;
        for j in 0..GROUP_BYTES {
            let b = codes[k * GROUP_BYTES + j];
            d += E2M1_X2_I8[(b & 0x0F) as usize] as i32 * xq[k * GROUP + 2 * j] as i32
                + E2M1_X2_I8[(b >> 4) as usize] as i32 * xq[k * GROUP + 2 * j + 1] as i32;
        }
        acc += d as f32 * UE4M3_HALF[s as usize];
    }
    acc
}

/// AVX2: two 16-groups per step — `pshufb` decodes 16 code bytes into 32
/// int8 weights, `maddubs(|w|, sign(x, w))` forms the products, and the low
/// and high 128-bit halves are the two groups' integer sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(codes: &[u8], scales: &[u8], xq: &[i8]) -> f32 {
    // SAFETY: codes.len() == 8·gpr, xq.len() >= 16·gpr, gpr even (caller).
    unsafe {
        use core::arch::x86_64::*;
        let table = _mm_loadu_si128(E2M1_X2_I8.as_ptr() as *const __m128i);
        let lomask = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        let mut k = 0;
        while k < scales.len() {
            let b = _mm_loadu_si128(codes.as_ptr().add(k * GROUP_BYTES) as *const __m128i);
            let lo = _mm_shuffle_epi8(table, _mm_and_si128(b, lomask));
            let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(b), lomask));
            let w = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            let x = _mm256_loadu_si256(xq.as_ptr().add(k * GROUP) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let d = _mm256_madd_epi16(p16, ones);
            let h = |v: __m128i| -> i32 {
                let s64 = _mm_add_epi32(v, _mm_srli_si128::<8>(v));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                _mm_cvtsi128_si32(s32)
            };
            let d0 = h(_mm256_castsi256_si128(d));
            let d1 = h(_mm256_extracti128_si256::<1>(d));
            acc += d0 as f32 * UE4M3_HALF[scales[k] as usize];
            acc += d1 as f32 * UE4M3_HALF[scales[k + 1] as usize];
            k += 2;
        }
        acc
    }
}

/// Decode row `r`'s codes into int8 weights `2·E2M1` (`dst[..cols]`).
#[inline]
fn decode_row_i8(m: &Nv<'_>, r: usize, dst: &mut [i8]) {
    for (j, &b) in m.row_codes(r).iter().enumerate() {
        dst[2 * j] = E2M1_X2_I8[(b & 0x0F) as usize];
        dst[2 * j + 1] = E2M1_X2_I8[(b >> 4) as usize];
    }
}

/// `dot_i8` over an already-decoded int8 row (same i32 per group, same order).
#[inline]
fn dot_decoded_i8(w: &[i8], scales: &[u8], xq: &[i8]) -> f32 {
    let mut acc = 0f32;
    for (k, &s) in scales.iter().enumerate() {
        let mut d = 0i32;
        for j in 0..GROUP {
            d += w[k * GROUP + j] as i32 * xq[k * GROUP + j] as i32;
        }
        acc += d as f32 * UE4M3_HALF[s as usize];
    }
    acc
}

#[inline]
fn use_a8w8() -> bool {
    a8w8_enabled()
}

#[inline]
fn row_a8w8(m: &Nv<'_>, r: usize, act: &SplitAct) -> f32 {
    let mut acc = m.dot_row_i8(r, &act.xq) * act.sx;
    for &(j, xv) in &act.outliers {
        acc += m.weight(r, j) * xv;
    }
    acc * m.g
}

/// `out[r] = W[r,:] · x` for every row.
pub(super) fn matvec(bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert!(out.len() >= rows && x.len() >= cols);
    let m = Nv::new(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    if use_a8w8() {
        let act = split_act(&x[..cols]);
        let run = |start: usize, end: usize| {
            for r in start..end {
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = row_a8w8(&m, r, &act) };
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        for r in start..end {
            // SAFETY: disjoint row ranges per worker.
            unsafe { *out_addr.at(r) = m.row_exact(r, x) };
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Two activations against the same weights: each output equals its own
/// `matvec` bit for bit.
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
    let m = Nv::new(bytes, rows, cols);
    let (p1, p2) = (SendMut(o1.as_mut_ptr()), SendMut(o2.as_mut_ptr()));
    if use_a8w8() {
        let (a1, a2) = (split_act(&x1[..cols]), split_act(&x2[..cols]));
        let run = |start: usize, end: usize| {
            for r in start..end {
                // SAFETY: disjoint row ranges per worker.
                unsafe {
                    *p1.at(r) = row_a8w8(&m, r, &a1);
                    *p2.at(r) = row_a8w8(&m, r, &a2);
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

/// `xs` `[b, cols]` → `out` `[b, rows]`, per token identical to `matvec`.
pub(super) fn matmat(bytes: &[u8], xs: &[f32], b: usize, rows: usize, cols: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert!(xs.len() >= b * cols && out.len() >= b * rows);
    let m = Nv::new(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    if use_a8w8() {
        let acts: Vec<SplitAct> = (0..b).map(|t| split_act(&xs[t * cols..(t + 1) * cols])).collect();
        let run = |start: usize, end: usize| {
            let mut w = vec![0i8; cols];
            for r in start..end {
                decode_row_i8(&m, r, &mut w);
                let sc = m.row_scales(r);
                for (t, act) in acts.iter().enumerate() {
                    let mut acc = dot_decoded_i8(&w, sc, &act.xq) * act.sx;
                    for &(j, xv) in &act.outliers {
                        acc += m.weight(r, j) * xv;
                    }
                    // SAFETY: disjoint (t, r) cells per worker range.
                    unsafe { *out_addr.at(t * rows + r) = acc * m.g };
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        for r in start..end {
            for t in 0..b {
                let v = m.row_exact(r, &xs[t * cols..(t + 1) * cols]);
                // SAFETY: disjoint (t, r) cells per worker range.
                unsafe { *out_addr.at(t * rows + r) = v };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let n = rows * cols;
        let mut p: Vec<u8> = (0..n / 2 + n / GROUP)
            .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed * 977) >> 7) as u8)
            .collect();
        for s in &mut p[n / 2..] {
            *s = (*s & 0x3F) | 0x20; // realistic scale bytes, no NaN
        }
        p.extend_from_slice(&1.5e-3f32.to_le_bytes());
        p
    }

    #[test]
    fn exact_and_a8w8_track_the_dequantized_matrix() {
        let (rows, cols) = (37usize, 256usize);
        let p = payload(rows, cols, 3);
        let mut w = vec![0f32; rows * cols];
        cortiq_core::nvfp4::dequant(&p, rows, cols, &mut w);
        let x: Vec<f32> = (0..cols).map(|j| ((j * 7919 % 113) as f32 - 56.0) / 17.0).collect();
        let want: Vec<f64> =
            (0..rows).map(|r| (0..cols).map(|j| w[r * cols + j] as f64 * x[j] as f64).sum()).collect();
        let m = Nv::new(&p, rows, cols);
        let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt() / (rows as f64).sqrt();
        for r in 0..rows {
            let e = m.row_exact(r, &x) as f64;
            assert!((e - want[r]).abs() <= 1e-4 * norm + 1e-9, "exact row {r}: {e} vs {}", want[r]);
        }
        let act = split_act(&x);
        for r in 0..rows {
            let a = row_a8w8(&m, r, &act) as f64;
            assert!((a - want[r]).abs() <= 2e-2 * norm, "a8w8 row {r}: {a} vs {}", want[r]);
            let s = dot_i8_scalar(m.row_codes(r), m.row_scales(r), &act.xq);
            assert_eq!(dot_i8(m.row_codes(r), m.row_scales(r), &act.xq), s, "simd == scalar, row {r}");
        }
    }
}
