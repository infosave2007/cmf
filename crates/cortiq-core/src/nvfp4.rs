//! `nvfp4` (dtype 22): NVIDIA NVFP4 — E2M1 elements with one unsigned E4M3
//! scale per 16 consecutive elements of a row and one f32 scale for the whole
//! tensor.
//!
//! Payload of a `[rows, cols]` tensor (`cols % 16 == 0`):
//!
//! ```text
//! [codes : rows · cols/2 bytes]   byte j of row r = elements 2j (low nibble), 2j+1 (high)
//! [scales: rows · cols/16 bytes]  byte g of row r = UE4M3 scale of elements 16g..16g+16
//! [global: 4 bytes]               f32, little-endian
//! ```
//!
//! Element value = `E2M1(code) · UE4M3(scale) · global`.
//!
//! The UE4M3 decode is ggml's (`ggml_ue4m3_to_fp32`): bit 7 is ignored,
//! exponent bits 6..3, mantissa bits 2..0, bias 7, a zero exponent is the
//! subnormal `m · 2^-9`, and `0x7F` (E4M3's NaN) reads as zero. That is the
//! contract the NVFP4 checkpoints this codec carries were produced and served
//! under, so a re-pack from ggml's `block_nvfp4` keeps every value.
//!
//! The code plane is the `mxfp4` one (`crate::mxfp4`), so the E2M1 tables
//! are shared; only the scale plane (16-element groups, E4M3) and the
//! per-tensor factor differ.

pub use crate::mxfp4::{E2M1_LUT, E2M1_X2_I8};

/// Elements per UE4M3 scale.
pub const GROUP: usize = 16;
/// Code bytes per 16-element group.
pub const GROUP_BYTES: usize = GROUP / 2;
/// Elements of one ggml `block_nvfp4` (four 16-element sub-blocks).
pub const GGML_BLOCK: usize = 64;
/// Bytes of one ggml `block_nvfp4`: four UE4M3 scales, then 32 code bytes.
pub const GGML_BLOCK_BYTES: usize = 4 + GGML_BLOCK / 2;

const fn ue4m3_table(half: bool) -> [f32; 256] {
    let mut t = [0f32; 256];
    let mut x = 0usize;
    while x < 256 {
        let b = x & 0x7F;
        if b != 0 && b != 0x7F {
            let e = (b >> 3) & 0xF;
            let m = (b & 7) as u32;
            // value = m·2^-9 (e = 0) or (8 + m)·2^(e−10): an exact f32 built
            // from its integer significand and a power of two.
            let (sig, exp) = if e == 0 { (m, -9i32) } else { (8 + m, e as i32 - 10) };
            let mut v = sig as f32;
            let mut k = exp + if half { -1 } else { 0 };
            while k < 0 {
                v *= 0.5;
                k += 1;
            }
            while k > 0 {
                v *= 2.0;
                k -= 1;
            }
            t[x] = v;
        }
        x += 1;
    }
    t
}

/// UE4M3 byte → f32 (exact; see the module docs for the bit contract).
pub static UE4M3: [f32; 256] = ue4m3_table(false);
/// UE4M3 byte → f32 / 2: the scale of the doubled int8 weights
/// [`E2M1_X2_I8`] (exact: every entry is a dyadic rational far from f32's
/// subnormal range).
pub static UE4M3_HALF: [f32; 256] = ue4m3_table(true);

/// Byte offset of the scale plane (behind every row's codes).
#[inline]
pub fn scale_offset(rows: usize, cols: usize) -> usize {
    rows * cols / 2
}

/// Byte offset of the f32 global scale.
#[inline]
pub fn global_offset(rows: usize, cols: usize) -> usize {
    rows * cols / 2 + rows * cols / GROUP
}

/// Payload length of a `[rows, cols]` tensor (None for a shape the format
/// does not allow: cols not a positive multiple of 16, or overflow).
pub fn nbytes(rows: usize, cols: usize) -> Option<usize> {
    if cols == 0 || cols % GROUP != 0 {
        return None;
    }
    let n = rows.checked_mul(cols)?;
    (n / 2).checked_add(n / GROUP)?.checked_add(4)
}

/// The f32 global scale of a payload.
#[inline]
pub fn global(bytes: &[u8], rows: usize, cols: usize) -> f32 {
    let o = global_offset(rows, cols);
    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

/// Split a payload into its (codes, scales, global) parts.
#[inline]
pub fn planes(bytes: &[u8], rows: usize, cols: usize) -> (&[u8], &[u8], f32) {
    let off = scale_offset(rows, cols);
    let n = rows * cols / GROUP;
    (&bytes[..off], &bytes[off..off + n], global(bytes, rows, cols))
}

/// Dequantize a full tensor into f32: `(E2M1 · UE4M3) · global`. The first
/// product is exact (a ≤2-bit by a ≤4-bit significand); only the global
/// factor rounds.
pub fn dequant(bytes: &[u8], rows: usize, cols: usize, dst: &mut [f32]) {
    debug_assert_eq!(cols % GROUP, 0);
    debug_assert!(dst.len() >= rows * cols);
    let (codes, scales, g) = planes(bytes, rows, cols);
    let gpr = cols / GROUP;
    for r in 0..rows {
        for k in 0..gpr {
            let s = UE4M3[scales[r * gpr + k] as usize];
            let src = &codes[(r * gpr + k) * GROUP_BYTES..(r * gpr + k + 1) * GROUP_BYTES];
            let out = &mut dst[r * cols + k * GROUP..r * cols + (k + 1) * GROUP];
            for (j, &b) in src.iter().enumerate() {
                out[2 * j] = (E2M1_LUT[(b & 0x0F) as usize] * s) * g;
                out[2 * j + 1] = (E2M1_LUT[(b >> 4) as usize] * s) * g;
            }
        }
    }
}

/// Lossless re-pack of a row-major ggml `block_nvfp4` matrix (`rows` rows of
/// `cols / 64` blocks, each `{u8 d[4]; u8 qs[32]}`; sub-block `s` keeps its
/// element `j` in the low nibble of `qs[8s + j]` and element `j + 8` in the
/// high one) and its per-tensor f32 scale into the `nvfp4` payload. Only bits
/// move: codes and scale bytes are copied unchanged.
pub fn repack_ggml(blocks: &[u8], rows: usize, cols: usize, global: f32) -> Result<Vec<u8>, String> {
    if cols == 0 || cols % GGML_BLOCK != 0 {
        return Err(format!("nvfp4: cols {cols} not a positive multiple of {GGML_BLOCK}"));
    }
    let bpr = cols / GGML_BLOCK;
    if blocks.len() != rows * bpr * GGML_BLOCK_BYTES {
        return Err(format!(
            "nvfp4: {} bytes of ggml blocks do not fit [{rows}, {cols}] (want {})",
            blocks.len(),
            rows * bpr * GGML_BLOCK_BYTES
        ));
    }
    if !global.is_finite() {
        return Err(format!("nvfp4: global scale {global} is not finite"));
    }
    let n = rows * cols;
    let mut out = vec![0u8; n / 2 + n / GROUP + 4];
    let (codes, rest) = out.split_at_mut(n / 2);
    let (scales, tail) = rest.split_at_mut(n / GROUP);
    for r in 0..rows {
        for b in 0..bpr {
            let blk = &blocks[(r * bpr + b) * GGML_BLOCK_BYTES..(r * bpr + b + 1) * GGML_BLOCK_BYTES];
            let (d, qs) = blk.split_at(4);
            for s in 0..4 {
                let gi = r * (cols / GROUP) + b * 4 + s;
                scales[gi] = d[s];
                // elements 16s + j, j in 0..16, of this block
                let mut v = [0u8; GROUP];
                for j in 0..8 {
                    let q = qs[8 * s + j];
                    v[j] = q & 0x0F;
                    v[j + 8] = q >> 4;
                }
                let dst = &mut codes[gi * GROUP_BYTES..(gi + 1) * GROUP_BYTES];
                for k in 0..GROUP_BYTES {
                    dst[k] = v[2 * k] | (v[2 * k + 1] << 4);
                }
            }
        }
    }
    tail.copy_from_slice(&global.to_le_bytes());
    Ok(out)
}

/// The value ggml's reference dequantizer gives element `i` of a row of
/// `block_nvfp4` blocks, times the tensor scale: `kvalues_fp4[code] ·
/// (UE4M3/2) · global` — the contract [`repack_ggml`] preserves.
pub fn ggml_element(blocks_row: &[u8], i: usize, global: f32) -> f32 {
    let blk = &blocks_row[(i / GGML_BLOCK) * GGML_BLOCK_BYTES..];
    let within = i % GGML_BLOCK;
    let (s, j) = (within / GROUP, within % GROUP);
    let q = blk[4 + 8 * s + (j % 8)];
    let code = if j < 8 { q & 0x0F } else { q >> 4 };
    (E2M1_X2_I8[code as usize] as f32 * UE4M3_HALF[blk[s] as usize]) * global
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue4m3_matches_ggml_formula() {
        for x in 0..256usize {
            let b = x & 0x7F;
            let want = if b == 0 || b == 0x7F {
                0.0
            } else {
                let e = ((b >> 3) & 0xF) as i32;
                let m = (b & 7) as f32;
                if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
            };
            assert_eq!(UE4M3[x], want, "byte {x:#x}");
            assert_eq!(UE4M3_HALF[x], want * 0.5, "byte {x:#x}");
        }
    }

    #[test]
    fn repack_keeps_every_ggml_value() {
        let (rows, cols) = (3usize, 128usize);
        let bpr = cols / GGML_BLOCK;
        let mut blocks = vec![0u8; rows * bpr * GGML_BLOCK_BYTES];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = (i.wrapping_mul(2654435761) >> 9) as u8;
        }
        // keep scale bytes away from 0x7F / bit 7 noise, as real files do
        for r in 0..rows * bpr {
            for s in 0..4 {
                blocks[r * GGML_BLOCK_BYTES + s] &= 0x76;
            }
        }
        let g = 3.0e-4f32;
        let p = repack_ggml(&blocks, rows, cols, g).unwrap();
        assert_eq!(p.len(), nbytes(rows, cols).unwrap());
        let mut d = vec![0f32; rows * cols];
        dequant(&p, rows, cols, &mut d);
        for r in 0..rows {
            let row = &blocks[r * bpr * GGML_BLOCK_BYTES..(r + 1) * bpr * GGML_BLOCK_BYTES];
            for i in 0..cols {
                assert_eq!(d[r * cols + i], ggml_element(row, i, g), "row {r} elem {i}");
            }
        }
    }
}
