//! §4.3 experiment: does an interleaved q4 tile layout
//! (`repeat { f16 scale; 16B nibbles }`, 18-byte stride) beat the
//! current split layout (`all nibbles, then all scales`) end-to-end on
//! a decode-shaped matvec? The roadmap accepts a new on-disk layout
//! ONLY on a measured win — this prints the honest ratio.
//!
//! Run: cargo run --release -p cortiq-engine --example q4_layout_ab

use std::time::Instant;

const GROUP: usize = 32;

fn f16_bits(x: f32) -> u16 {
    // Quick f32→f16 (round-to-nearest is irrelevant for the timing A/B).
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}

fn f16_val(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Split layout: packed nibbles for all groups, then f16 scales.
struct Split {
    packed: Vec<u8>,
    scales: Vec<u8>,
}

/// Interleaved layout: per group `[f16 scale][16B nibbles]` (18B tiles).
struct Tiles {
    tiles: Vec<u8>,
}

fn synth(rows: usize, cols: usize) -> (Split, Tiles, Vec<i8>) {
    let gpr = cols / GROUP;
    let groups = rows * gpr;
    let mut packed = vec![0u8; groups * 16];
    let mut scales = vec![0u8; groups * 2];
    let mut tiles = vec![0u8; groups * 18];
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for g in 0..groups {
        let s = 0.002 + (rnd() % 1000) as f32 * 1e-5;
        let sb = f16_bits(s).to_le_bytes();
        scales[g * 2..g * 2 + 2].copy_from_slice(&sb);
        tiles[g * 18..g * 18 + 2].copy_from_slice(&sb);
        for k in 0..16 {
            let byte = (rnd() & 0xFF) as u8;
            packed[g * 16 + k] = byte;
            tiles[g * 18 + 2 + k] = byte;
        }
    }
    let xq: Vec<i8> = (0..cols)
        .map(|_| ((rnd() % 255) as i32 - 127) as i8)
        .collect();
    (Split { packed, scales }, Tiles { tiles }, xq)
}

#[inline(always)]
fn dot_group_scalar(pk: &[u8], xq: &[i8]) -> i32 {
    let mut d = 0i32;
    for (k, &b) in pk.iter().enumerate() {
        d += ((b & 0x0F) as i32 - 8) * xq[k * 2] as i32
            + (((b >> 4) & 0x0F) as i32 - 8) * xq[k * 2 + 1] as i32;
    }
    d
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_group(pk: &[u8], xq: &[i8]) -> i32 {
    // NEON nibble unpack + sdot, same structure as the shipped kernel.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let b = vld1q_u8(pk.as_ptr());
        let lo = vandq_u8(b, lomask);
        let hi = vshrq_n_u8::<4>(b);
        let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
        let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
        let x0 = vld1q_s8(xq.as_ptr());
        let x1 = vld1q_s8(xq.as_ptr().add(16));
        let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        asm!(
            "sdot {a0:v}.4s, {e0:v}.16b, {x0:v}.16b",
            "sdot {a1:v}.4s, {e1:v}.16b, {x1:v}.16b",
            a0 = inout(vreg) a0, a1 = inout(vreg) a1,
            e0 = in(vreg) e0, x0 = in(vreg) x0, e1 = in(vreg) e1, x1 = in(vreg) x1,
            options(pure, nomem, nostack),
        );
        vaddvq_s32(vaddq_s32(a0, a1))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dot_group(pk: &[u8], xq: &[i8]) -> i32 {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { dot_group_avx2(pk, xq) }
    } else {
        dot_group_scalar(pk, xq)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_group_avx2(pk: &[u8], xq: &[i8]) -> i32 {
    // Same structure as the shipped dot_q4_row_avx2 group body.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let b = _mm_loadu_si128(pk.as_ptr() as *const __m128i);
        let lo = _mm_and_si128(b, lomask);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
        let w = _mm256_sub_epi8(
            _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
            eight,
        );
        let x = _mm256_loadu_si256(xq.as_ptr() as *const __m256i);
        let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
        let d = _mm256_madd_epi16(p16, ones);
        let hi128 = _mm256_extracti128_si256::<1>(d);
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
        let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
        let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
        _mm_cvtsi128_si32(s32)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn dot_group(pk: &[u8], xq: &[i8]) -> i32 {
    dot_group_scalar(pk, xq)
}

fn matvec_split(s: &Split, rows: usize, cols: usize, xq: &[i8], out: &mut [f32]) {
    let gpr = cols / GROUP;
    for r in 0..rows {
        let mut acc = 0f32;
        for gi in 0..gpr {
            let g = r * gpr + gi;
            let sc = f16_val(u16::from_le_bytes([s.scales[g * 2], s.scales[g * 2 + 1]]));
            let d = dot_group(
                &s.packed[g * 16..g * 16 + 16],
                &xq[gi * GROUP..gi * GROUP + GROUP],
            );
            acc += d as f32 * sc;
        }
        out[r] = acc;
    }
}

fn matvec_tiles(t: &Tiles, rows: usize, cols: usize, xq: &[i8], out: &mut [f32]) {
    let gpr = cols / GROUP;
    for r in 0..rows {
        let mut acc = 0f32;
        for gi in 0..gpr {
            let g = r * gpr + gi;
            let tile = &t.tiles[g * 18..g * 18 + 18];
            let sc = f16_val(u16::from_le_bytes([tile[0], tile[1]]));
            let d = dot_group(&tile[2..18], &xq[gi * GROUP..gi * GROUP + GROUP]);
            acc += d as f32 * sc;
        }
        out[r] = acc;
    }
}

fn main() {
    // Decode-shaped: FFN matvec of a ~0.5–1B class model.
    let (rows, cols) = (4864usize, 896usize);
    let reps = 60;
    let (split, tiles, xq) = synth(rows, cols);
    let bytes_per_pass = rows * cols / 2 + rows * (cols / GROUP) * 2;

    let mut o1 = vec![0f32; rows];
    let mut o2 = vec![0f32; rows];
    // Warmup + correctness cross-check.
    matvec_split(&split, rows, cols, &xq, &mut o1);
    matvec_tiles(&tiles, rows, cols, &xq, &mut o2);
    assert_eq!(o1, o2, "layouts must produce identical results");

    let t0 = Instant::now();
    for _ in 0..reps {
        matvec_split(&split, rows, cols, &xq, &mut o1);
    }
    let split_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..reps {
        matvec_tiles(&tiles, rows, cols, &xq, &mut o2);
    }
    let tiles_s = t1.elapsed().as_secs_f64();

    let gbs = |secs: f64| bytes_per_pass as f64 * reps as f64 / secs / 1e9;
    println!("q4 layout A/B  ({rows}x{cols}, {reps} reps, single thread)");
    println!(
        "  split (nibbles.., scales..): {:7.2} ms/pass  {:5.2} GB/s",
        split_s * 1e3 / reps as f64,
        gbs(split_s)
    );
    println!(
        "  tiles (scale+nibbles, 18B):  {:7.2} ms/pass  {:5.2} GB/s",
        tiles_s * 1e3 / reps as f64,
        gbs(tiles_s)
    );
    println!("  tiles/split speed ratio: {:.3}x", split_s / tiles_s);
}
