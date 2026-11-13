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
    if avx2_enabled() && scales.len().is_multiple_of(2) {
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

/// AVX-512 VNNI: one row against up to eight int8 activations, 64 elements (four scale groups) a step: the
/// codes become `2·E2M1 + 12` bytes by `vpshufb`, `vpdpbusd` sums four products a lane, the offset comes off
/// by the activation's quad sums (`xs4`, times 12), and the lanes stay floats (times their group's scale)
/// until the row ends. `out[b]` = Σ_g (Σ 2·E2M1·xq) · UE4M3/2, as [`dot_i8`] up to float rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2")]
unsafe fn rows_dot_v512(codes: &[u8], scales: &[u8], xq: &[&[i8]], xs4: &[&[i32]], out: &mut [f32]) {
    use core::arch::x86_64::*;
    const T: [i8; 16] = [12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0];
    // SAFETY: codes holds 32 bytes, scales 4, every xq 64 and every xs4 16 per step (the caller's shapes).
    unsafe {
        let tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(T.as_ptr() as *const __m128i));
        let m4 = _mm256_set1_epi8(0x0F);
        let gidx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        let nb = xq.len().min(8);
        let mut acc = [_mm512_setzero_ps(); 8];
        for st in 0..scales.len() / 4 {
            let c = _mm256_loadu_si256(codes.as_ptr().add(st * 32) as *const __m256i);
            let lo = _mm256_shuffle_epi8(tab, _mm256_and_si256(c, m4));
            let hi = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16::<4>(c), m4));
            let e0 = _mm256_unpacklo_epi8(lo, hi);
            let e1 = _mm256_unpackhi_epi8(lo, hi);
            let w = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_permute2x128_si256::<0x20>(e0, e1)),
                _mm256_permute2x128_si256::<0x31>(e0, e1),
            );
            let sc = scales.as_ptr().add(st * 4);
            let sv = _mm_set_ps(
                UE4M3_HALF[*sc.add(3) as usize],
                UE4M3_HALF[*sc.add(2) as usize],
                UE4M3_HALF[*sc.add(1) as usize],
                UE4M3_HALF[*sc as usize],
            );
            let scv = _mm512_permutexvar_ps(gidx, _mm512_castps128_ps512(sv));
            for b in 0..nb {
                let x = _mm512_loadu_si512(xq[b].as_ptr().add(st * 64) as *const _);
                let d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x);
                let off = _mm512_loadu_si512(xs4[b].as_ptr().add(st * 16) as *const _);
                acc[b] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d, off)), scv, acc[b]);
            }
        }
        for b in 0..nb {
            out[b] = _mm512_reduce_add_ps(acc[b]);
        }
    }
}

/// `12 · Σ` of every four consecutive int8 activations (the offset [`rows_dot_v512`] takes off).
fn quad_sums12(xq: &[i8]) -> Vec<i32> {
    xq.as_chunks::<4>().0.iter().map(|q| 12 * q.iter().map(|&v| v as i32).sum::<i32>()).collect()
}

/// Whether [`rows_dot_v512`] runs here (AVX-512 VNNI and the int8 activation path).
fn v512_on() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            use_a8w8()
                && avx512vnni_enabled()
                && !std::env::var("CMF_NVFP4_V512").is_ok_and(|v| v.trim() == "0")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Rows `start..end` of `m` against the activations `ps` (pairs' indices into `acts`), `put(pair, row, value)`
/// with the full value (activation scale, outliers, the tensor's global). The VNNI kernel for 64-wide rows.
fn rows_against(m: &Nv<'_>, start: usize, end: usize, ps: std::ops::Range<usize>, acts: &[SplitAct], xs4: &[Vec<i32>], put: &mut dyn FnMut(usize, usize, f32)) {
    #[cfg(target_arch = "x86_64")]
    if !xs4.is_empty() && m.gpr.is_multiple_of(4) {
        // eight activations a pass over the rows; nothing is allocated per row
        let mut p0 = ps.start;
        while p0 < ps.end {
            let p1 = (p0 + 8).min(ps.end);
            let nb = p1 - p0;
            let mut xq: [&[i8]; 8] = [&[]; 8];
            let mut x4: [&[i32]; 8] = [&[]; 8];
            for (k, (a, x)) in acts[p0..p1].iter().zip(&xs4[p0..p1]).enumerate() {
                xq[k] = &a.xq[..];
                x4[k] = &x[..];
            }
            let mut tmp = [0f32; 8];
            for r in start..end {
                // SAFETY: the feature check of `v512_on` (xs4 is empty without it); 64-wide rows
                unsafe { rows_dot_v512(m.row_codes(r), m.row_scales(r), &xq[..nb], &x4[..nb], &mut tmp) };
                for (k, a) in acts[p0..p1].iter().enumerate() {
                    let mut v = tmp[k] * a.sx;
                    for &(j, xv) in &a.outliers {
                        v += m.weight(r, j) * xv;
                    }
                    put(p0 + k, r, v * m.g);
                }
            }
            p0 = p1;
        }
        return;
    }
    let _ = xs4;
    for r in start..end {
        for p in ps.clone() {
            put(p, r, row_a8w8(m, r, &acts[p]));
        }
    }
}

/// Latent experts on the host (Edge0's experts a card does not hold): expert `i` takes the input rows
/// `groups[i]` of `xs` (`[row][cols]`), and `out` gets, pair after pair in that order, `[cols]` =
/// `down · (silu(gate · x) ⊙ (up · x))` — gate and up `[ff, cols]`, down `[cols, ff]`, all `nvfp4`. Two
/// dispatches over every expert's rows (gate with up, then down), each row read once for all its pairs.
#[allow(clippy::type_complexity)]
pub(super) fn latent_rows(
    experts: &[(&[u8], &[u8], &[u8])],
    ff: usize,
    cols: usize,
    groups: &[Vec<usize>],
    xs: &[f32],
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    let pairs: usize = groups.iter().map(|g| g.len()).sum();
    debug_assert!(out.len() >= pairs * cols);
    let mut base = Vec::with_capacity(groups.len());
    let mut at = 0usize;
    for g in groups {
        base.push(at);
        at += g.len();
    }
    let gate: Vec<(Nv<'_>, Nv<'_>, Nv<'_>)> =
        experts.iter().map(|&(g, u, d)| (Nv::new(g, ff, cols), Nv::new(u, ff, cols), Nv::new(d, cols, ff))).collect();
    let a8 = use_a8w8();
    let row_of = |p: usize| -> &[f32] {
        let mut k = 0usize;
        for (i, g) in groups.iter().enumerate() {
            if p < base[i] + g.len() {
                k = g[p - base[i]];
                break;
            }
        }
        &xs[k * cols..(k + 1) * cols]
    };
    let acts: Vec<SplitAct> = if a8 { (0..pairs).map(|p| split_act(row_of(p))).collect() } else { Vec::new() };
    let v512 = a8 && v512_on() && cols.is_multiple_of(64) && ff.is_multiple_of(64);
    let xs4: Vec<Vec<i32>> = if v512 { acts.iter().map(|a| quad_sums12(&a.xq)).collect() } else { Vec::new() };
    let mut gu = vec![0f32; 2 * pairs * ff];
    let gu_at = SendMut(gu.as_mut_ptr());
    let run_gu = |i: usize, start: usize, end: usize| {
        let (g, u, _) = &gate[i];
        if v512 {
            let ps = base[i]..base[i] + groups[i].len();
            // SAFETY: (pair, row) cells are disjoint across the workers' row ranges.
            rows_against(g, start, end, ps.clone(), &acts, &xs4, &mut |p, r, v| unsafe { *gu_at.at(2 * p * ff + r) = v });
            rows_against(u, start, end, ps, &acts, &xs4, &mut |p, r, v| unsafe { *gu_at.at((2 * p + 1) * ff + r) = v });
            return;
        }
        for r in start..end {
            for p in base[i]..base[i] + groups[i].len() {
                let (a, b) = if a8 {
                    (row_a8w8(g, r, &acts[p]), row_a8w8(u, r, &acts[p]))
                } else {
                    let x = row_of(p);
                    (g.row_exact(r, x), u.row_exact(r, x))
                };
                // SAFETY: (pair, row) cells are disjoint across the workers' row ranges.
                unsafe {
                    *gu_at.at(2 * p * ff + r) = a;
                    *gu_at.at((2 * p + 1) * ff + r) = b;
                }
            }
        }
    };
    let fs: Vec<Box<dyn Fn(usize, usize) + Sync + '_>> =
        (0..experts.len()).map(|i| Box::new(move |s: usize, e: usize| run_gu(i, s, e)) as Box<dyn Fn(usize, usize) + Sync>).collect();
    let parts: Vec<(usize, &(dyn Fn(usize, usize) + Sync))> = fs.iter().map(|f| (ff, f.as_ref())).collect();
    match pool {
        Some(p) => p.run_many(&parts),
        None => parts.iter().for_each(|(n, f)| f(0, *n)),
    }
    let mut act = vec![0f32; pairs * ff];
    for p in 0..pairs {
        for r in 0..ff {
            let (g, u) = (gu[2 * p * ff + r], gu[(2 * p + 1) * ff + r]);
            act[p * ff + r] = g / (1.0 + (-g).exp()) * u;
        }
    }
    let acts2: Vec<SplitAct> = if a8 { (0..pairs).map(|p| split_act(&act[p * ff..(p + 1) * ff])).collect() } else { Vec::new() };
    let xs42: Vec<Vec<i32>> = if v512 { acts2.iter().map(|a| quad_sums12(&a.xq)).collect() } else { Vec::new() };
    let out_at = SendMut(out.as_mut_ptr());
    let run_d = |i: usize, start: usize, end: usize| {
        let d = &gate[i].2;
        if v512 {
            let ps = base[i]..base[i] + groups[i].len();
            // SAFETY: (pair, row) cells are disjoint across the workers' row ranges.
            rows_against(d, start, end, ps, &acts2, &xs42, &mut |p, r, v| unsafe { *out_at.at(p * cols + r) = v });
            return;
        }
        for r in start..end {
            for p in base[i]..base[i] + groups[i].len() {
                let v = if a8 { row_a8w8(d, r, &acts2[p]) } else { d.row_exact(r, &act[p * ff..(p + 1) * ff]) };
                // SAFETY: (pair, row) cells are disjoint across the workers' row ranges.
                unsafe { *out_at.at(p * cols + r) = v };
            }
        }
    };
    let fs: Vec<Box<dyn Fn(usize, usize) + Sync + '_>> =
        (0..experts.len()).map(|i| Box::new(move |s: usize, e: usize| run_d(i, s, e)) as Box<dyn Fn(usize, usize) + Sync>).collect();
    let parts: Vec<(usize, &(dyn Fn(usize, usize) + Sync))> = fs.iter().map(|f| (cols, f.as_ref())).collect();
    match pool {
        Some(p) => p.run_many(&parts),
        None => parts.iter().for_each(|(n, f)| f(0, *n)),
    }
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

    #[test]
    fn latent_rows_match_the_dequantized_expert() {
        let (ff, l) = (64usize, 128usize);
        let ex: Vec<[Vec<u8>; 3]> = (0..2).map(|i| [payload(ff, l, 10 + i), payload(ff, l, 20 + i), payload(l, ff, 30 + i)]).collect();
        let xs: Vec<f32> = (0..3 * l).map(|j| ((j * 7919 % 113) as f32 - 56.0) / 97.0).collect();
        let groups = vec![vec![2usize, 0], vec![1usize]];
        let views: Vec<(&[u8], &[u8], &[u8])> = ex.iter().map(|e| (&e[0][..], &e[1][..], &e[2][..])).collect();
        let mut out = vec![0f32; 3 * l];
        let pool = Pool::new(3);
        latent_rows(&views, ff, l, &groups, &xs, &mut out, Some(&pool));
        let mut p = 0usize;
        for (i, g) in groups.iter().enumerate() {
            let deq = |b: &[u8], rows: usize, cols: usize| {
                let mut w = vec![0f32; rows * cols];
                cortiq_core::nvfp4::dequant(b, rows, cols, &mut w);
                w
            };
            let (wg, wu, wd) = (deq(&ex[i][0], ff, l), deq(&ex[i][1], ff, l), deq(&ex[i][2], l, ff));
            for &k in g {
                let x = &xs[k * l..(k + 1) * l];
                let a: Vec<f64> = (0..ff)
                    .map(|r| {
                        let g: f64 = (0..l).map(|j| wg[r * l + j] as f64 * x[j] as f64).sum();
                        let u: f64 = (0..l).map(|j| wu[r * l + j] as f64 * x[j] as f64).sum();
                        g / (1.0 + (-g).exp()) * u
                    })
                    .collect();
                let want: Vec<f64> = (0..l).map(|r| (0..ff).map(|j| wd[r * ff + j] as f64 * a[j]).sum()).collect();
                let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
                let err = (0..l).map(|r| (out[p * l + r] as f64 - want[r]).powi(2)).sum::<f64>().sqrt();
                assert!(err <= 3e-2 * norm, "expert {i} input {k}: relative error {}", err / norm);
                p += 1;
            }
        }
    }
}
