//! Canonical quantization layouts and scalar dequantization.
//!
//! Layouts are byte-identical to `.vmfc` ("quants first, then scales"):
//! - `q8_row`  (2-D `[out, in]`): `[int8: out·in][f16: out]`,
//!   `w = q[o,i] · scale[o]`, `scale[o] = absmax(row_o) / 127`.
//! - `q4_block`: groups of 32 over the flattened tensor (zero-padded),
//!   `[u8: ceil(n/32)·16][f16: ceil(n/32)]`, nibbles low-first,
//!   `w = (q − 8) · scale`, `scale = absmax(group) / 7`.
//!
//! 1-D tensors (norms) are always stored `f16`.

use crate::format::TensorEntry;
use crate::types::TensorDtype;

pub const GROUP_SIZE: usize = 32;

/// IEEE half → f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal: normalize. A subnormal half equals mant·2^-24; shifting
            // its MSB up to bit 10 takes e = 10-b shifts (b = MSB position), so
            // the true exponent is b-24 and the f32 biased exponent is b+103 =
            // 113-e. (The old `127-15-e` form was off by one — it halved every
            // subnormal, which corrupts K-quant super-block scales.)
            let mut e = 0u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            (sign << 31) | ((113 - e) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 → IEEE half (round-to-nearest-even). Used by the Rust writer.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    exp -= 127 - 15;
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow → Inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow → 0
        }
        // subnormal
        let m = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = m >> shift;
        let round = (m >> (shift - 1)) & 1;
        return sign | ((half + round) as u16);
    }
    let half = ((exp as u32) << 10) | (mant >> 13);
    let round = (mant >> 12) & 1;
    // round-to-nearest-even: bump if round bit set and (sticky or odd)
    let sticky = (mant & 0xFFF) != 0;
    let bump = round & (sticky as u32 | (half & 1));
    sign | ((half + bump) as u16)
}

/// bfloat16 → f32.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Dequantize a full `q8_row` tensor: `[int8: out·in][f16: out]`.
pub fn dequant_q8_row(bytes: &[u8], out_dim: usize, in_dim: usize, dst: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out_dim * in_dim + out_dim * 2);
    debug_assert_eq!(dst.len(), out_dim * in_dim);
    let (q, scales) = bytes.split_at(out_dim * in_dim);
    for o in 0..out_dim {
        let s = f16_to_f32(u16::from_le_bytes([scales[o * 2], scales[o * 2 + 1]]));
        let row = &q[o * in_dim..(o + 1) * in_dim];
        let out = &mut dst[o * in_dim..(o + 1) * in_dim];
        for (d, &b) in out.iter_mut().zip(row) {
            *d = (b as i8) as f32 * s;
        }
    }
}

/// Dequantize a full `q8_2f` tensor (two-field 𝒲×θ, dtype 9):
/// `[int8: out·in][f16 row_scale: out][f16 col: in]`,
/// `w[o,i] = q[o,i] · row_scale[o] · col[i]`. The column field absorbs
/// outlier input channels — validated in vmfcore (+37% at equal size
/// for the two-field family; q8_2f recovers ~75% of the q8→f16 gap).
pub fn dequant_q8_2f(bytes: &[u8], out_dim: usize, in_dim: usize, dst: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out_dim * in_dim + out_dim * 2 + in_dim * 2);
    debug_assert_eq!(dst.len(), out_dim * in_dim);
    let (q, rest) = bytes.split_at(out_dim * in_dim);
    let (scales, cols) = rest.split_at(out_dim * 2);
    let col: Vec<f32> = (0..in_dim)
        .map(|i| f16_to_f32(u16::from_le_bytes([cols[i * 2], cols[i * 2 + 1]])))
        .collect();
    for o in 0..out_dim {
        let s = f16_to_f32(u16::from_le_bytes([scales[o * 2], scales[o * 2 + 1]]));
        let row = &q[o * in_dim..(o + 1) * in_dim];
        let out = &mut dst[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            out[i] = (row[i] as i8) as f32 * s * col[i];
        }
    }
}

/// Dequantize a full `q4_block` tensor into `dst` (`dst.len()` = real
/// element count; the trailing pad group elements are discarded).
pub fn dequant_q4_block(bytes: &[u8], dst: &mut [f32]) {
    let n_groups = (dst.len() + GROUP_SIZE - 1) / GROUP_SIZE;
    let packed_len = n_groups * GROUP_SIZE / 2;
    debug_assert_eq!(bytes.len(), packed_len + n_groups * 2);
    let (packed, scales) = bytes.split_at(packed_len);
    for g in 0..n_groups {
        let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
        let base = g * GROUP_SIZE;
        let pk = &packed[g * 16..(g + 1) * 16];
        for (k, &byte) in pk.iter().enumerate() {
            let i0 = base + k * 2;
            let i1 = i0 + 1;
            if i0 < dst.len() {
                dst[i0] = ((byte & 0x0F) as f32 - 8.0) * s;
            }
            if i1 < dst.len() {
                dst[i1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * s;
            }
        }
    }
}

/// Dequantize a full `vbit` tensor (P13 FIG.3, grouped variant):
/// [u8 bits: rows][f16 scales: rows·cols/32][bit-packed rows MSB-first,
/// each row padded to a whole byte]. w = (u − L)·scale, L = 2^{b−1}−1.
pub fn dequant_vbit(bytes: &[u8], rows: usize, cols: usize, dst: &mut [f32]) -> Result<(), String> {
    if cols % GROUP_SIZE != 0 {
        return Err(format!("vbit: cols {cols} not a multiple of {GROUP_SIZE}"));
    }
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    if let Some(&b) = bits.iter().find(|&&b| !(3..=8).contains(&b)) {
        return Err(format!("vbit: bit-width {b} outside safe range 3..=8"));
    }
    let sc_off = rows;
    let data_off = sc_off + rows * ng * 2;
    let mut off = data_off;
    for r in 0..rows {
        let b = bits[r] as usize;
        let l = ((1usize << (b - 1)) - 1) as f32;
        let rowlen = (cols * b + 7) / 8;
        let data = &bytes[off..off + rowlen];
        let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
        for i in 0..cols {
            while nbits < b {
                acc = (acc << 8) | data[idx] as u64;
                idx += 1;
                nbits += 8;
            }
            let u = ((acc >> (nbits - b)) & ((1u64 << b) - 1)) as f32;
            nbits -= b;
            let so = (r * ng + i / GROUP_SIZE) * 2;
            let s = f16_to_f32(u16::from_le_bytes([bytes[sc_off + so], bytes[sc_off + so + 1]]));
            dst[r * cols + i] = (u - l) * s;
        }
        off += rowlen;
    }
    Ok(())
}


/// Bytes per q4_tiled group tile: 2 (f16 scale) + 16 (nibbles).
pub const Q4_TILE: usize = 18;

/// Dequantize a full `q4_tiled` tensor: per 32-group
/// `[f16 scale][16B nibbles]`, nibbles low-first inside each byte —
/// the same values/order as `q4_block`, only the placement of the
/// scale differs.
pub fn dequant_q4_tiled(bytes: &[u8], dst: &mut [f32]) {
    let n_groups = (dst.len() + GROUP_SIZE - 1) / GROUP_SIZE;
    debug_assert_eq!(bytes.len(), n_groups * Q4_TILE);
    for g in 0..n_groups {
        let tile = &bytes[g * Q4_TILE..(g + 1) * Q4_TILE];
        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
        let base = g * GROUP_SIZE;
        for (k, &byte) in tile[2..].iter().enumerate() {
            let i0 = base + k * 2;
            let i1 = i0 + 1;
            if i0 < dst.len() {
                dst[i0] = ((byte & 0x0F) as f32 - 8.0) * s;
            }
            if i1 < dst.len() {
                dst[i1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * s;
            }
        }
    }
}

/// Byte layout of a `vbit_ro` payload (roadmap §4.2):
/// `[u8 bits: rows][f16 scales: rows·cols/32][u32 row_offsets: rows+1]
///  [bit-packed rows]` — offsets are relative to the packed area, so
/// `offsets[r]..offsets[r+1]` is row r without any prefix scan.
/// Returns (scales_off, offsets_off, packed_off).
pub fn vbit_ro_sections(rows: usize, cols: usize) -> (usize, usize, usize) {
    let ng = cols / GROUP_SIZE;
    let scales_off = rows;
    let offsets_off = scales_off + rows * ng * 2;
    let packed_off = offsets_off + (rows + 1) * 4;
    (scales_off, offsets_off, packed_off)
}

/// Read one u32 row offset from a `vbit_ro` offsets table.
#[inline]
pub fn vbit_ro_offset(bytes: &[u8], offsets_off: usize, r: usize) -> usize {
    let o = offsets_off + r * 4;
    u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
}

/// Dequantize a full `vbit_ro` tensor — same math as `dequant_vbit`,
/// rows addressed through the stored offset table.
pub fn dequant_vbit_ro(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    dst: &mut [f32],
) -> Result<(), String> {
    if cols % GROUP_SIZE != 0 {
        return Err(format!("vbit_ro: cols {cols} not a multiple of {GROUP_SIZE}"));
    }
    let ng = cols / GROUP_SIZE;
    let (sc_off, off_off, packed_off) = vbit_ro_sections(rows, cols);
    let bits = &bytes[..rows];
    for r in 0..rows {
        let b = bits[r] as usize;
        if !matches!(b, 3..=6 | 8) {
            return Err(format!("vbit_ro row {r}: bit width {b} outside {{3,4,5,6,8}}"));
        }
        let l = ((1usize << (b - 1)) - 1) as f32;
        let start = packed_off + vbit_ro_offset(bytes, off_off, r);
        let end = packed_off + vbit_ro_offset(bytes, off_off, r + 1);
        let data = &bytes[start..end];
        let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
        for i in 0..cols {
            while nbits < b {
                acc = (acc << 8) | data[idx] as u64;
                idx += 1;
                nbits += 8;
            }
            let u = ((acc >> (nbits - b)) & ((1u64 << b) - 1)) as f32;
            nbits -= b;
            let so = (r * ng + i / GROUP_SIZE) * 2;
            let sc = f16_to_f32(u16::from_le_bytes([bytes[sc_off + so], bytes[sc_off + so + 1]]));
            dst[r * cols + i] = (u - l) * sc;
        }
    }
    Ok(())
}

/// Expected byte length of a tensor given dtype and element count.
pub fn expected_nbytes(dtype: TensorDtype, shape: &[usize]) -> Option<usize> {
    let n: usize = shape.iter().product();
    Some(match dtype {
        TensorDtype::F32 => n * 4,
        TensorDtype::F16 | TensorDtype::Bf16 => n * 2,
        TensorDtype::Q8Row => {
            let out = *shape.first()?;
            n + out * 2
        }
        TensorDtype::Q4Block => {
            let groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
            groups * 16 + groups * 2
        }
        TensorDtype::Q4Tiled => {
            // Interleaved tiles: [f16 scale][16B nibbles] per 32-group.
            let groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
            groups * 18
        }
        TensorDtype::Q8_2f => {
            let out = *shape.first()?;
            let inn = n / out.max(1);
            n + out * 2 + inn * 2
        }
        _ => return None, // reserved dtypes: size not defined by this reader
    })
}

/// Validate a tensor payload against its directory entry (roadmap
/// §4.9): every length is checked BEFORE any slice is taken, so a
/// corrupted or truncated file fails loudly at `open()` instead of
/// panicking in a kernel. For fixed-size dtypes this is the
/// `expected_nbytes` equality; for vbit — whose payload length depends
/// on the per-row bit widths stored in the payload itself — the exact
/// length is computed from the (validated) width header.
pub fn validate_payload(
    dtype: TensorDtype,
    shape: &[usize],
    bytes: &[u8],
) -> Result<(), String> {
    if dtype == TensorDtype::VbitRo {
        if shape.len() != 2 {
            return Err(format!("vbit_ro tensor must be 2-D, got {shape:?}"));
        }
        let (rows, cols) = (shape[0], shape[1]);
        if cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(format!(
                "vbit_ro cols {cols} not a positive multiple of {GROUP_SIZE}"
            ));
        }
        let (_, off_off, packed_off) = vbit_ro_sections(rows, cols);
        if bytes.len() < packed_off {
            return Err(format!(
                "vbit_ro payload {} bytes cannot hold headers ({packed_off})",
                bytes.len()
            ));
        }
        if vbit_ro_offset(bytes, off_off, 0) != 0 {
            return Err("vbit_ro offsets[0] must be 0".to_string());
        }
        for r in 0..rows {
            let b = bytes[r];
            if !matches!(b, 3..=6 | 8) {
                return Err(format!(
                    "vbit_ro row {r}: bit width {b} outside {{3,4,5,6,8}}"
                ));
            }
            let want = (cols * b as usize).div_ceil(8);
            let got = vbit_ro_offset(bytes, off_off, r + 1)
                .checked_sub(vbit_ro_offset(bytes, off_off, r))
                .ok_or_else(|| format!("vbit_ro offsets not monotonic at row {r}"))?;
            if want != got {
                return Err(format!(
                    "vbit_ro row {r}: offset span {got} != {want} for width {b}"
                ));
            }
        }
        let total = packed_off + vbit_ro_offset(bytes, off_off, rows);
        if total != bytes.len() {
            return Err(format!(
                "vbit_ro payload length {} != computed {total}",
                bytes.len()
            ));
        }
        return Ok(());
    }
    if dtype == TensorDtype::Vbit {
        if shape.len() != 2 {
            return Err(format!("vbit tensor must be 2-D, got {shape:?}"));
        }
        let (rows, cols) = (shape[0], shape[1]);
        if cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(format!(
                "vbit cols {cols} not a positive multiple of {GROUP_SIZE}"
            ));
        }
        // bits header: bounds BEFORE slicing.
        if bytes.len() < rows {
            return Err(format!(
                "vbit payload {} bytes cannot hold the {rows}-byte width header",
                bytes.len()
            ));
        }
        let ng = cols / GROUP_SIZE;
        let mut total = rows + rows * ng * 2;
        for (r, &b) in bytes[..rows].iter().enumerate() {
            if !matches!(b, 3..=6 | 8) {
                return Err(format!("vbit row {r}: bit width {b} outside {{3,4,5,6,8}}"));
            }
            total += (cols * b as usize).div_ceil(8);
        }
        if total != bytes.len() {
            return Err(format!(
                "vbit payload length {} != computed {} (rows {rows}, cols {cols})",
                bytes.len(),
                total
            ));
        }
        return Ok(());
    }
    if let Some(expect) = expected_nbytes(dtype, shape) {
        if expect != bytes.len() {
            return Err(format!(
                "payload length {} != expected {expect} for {dtype:?}{shape:?}",
                bytes.len()
            ));
        }
    }
    Ok(())
}

/// Dequantize any supported tensor into f32.
pub fn dequant_tensor(entry: &TensorEntry, bytes: &[u8], dst: &mut [f32]) -> Result<(), String> {
    let n: usize = entry.shape.iter().product();
    if dst.len() != n {
        return Err(format!(
            "dst len {} != tensor elems {} for '{}'",
            dst.len(),
            n,
            entry.name
        ));
    }
    match entry.dtype {
        TensorDtype::F32 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        TensorDtype::F16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
            }
        }
        TensorDtype::Bf16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = bf16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
            }
        }
        TensorDtype::Q8Row => {
            if entry.shape.len() != 2 {
                return Err(format!("q8_row tensor '{}' must be 2-D", entry.name));
            }
            dequant_q8_row(bytes, entry.shape[0], entry.shape[1], dst);
        }
        TensorDtype::Q4Block => dequant_q4_block(bytes, dst),
        TensorDtype::Q4Tiled => dequant_q4_tiled(bytes, dst),
        TensorDtype::Vbit => {
            if entry.shape.len() != 2 {
                return Err(format!("vbit tensor '{}' must be 2-D", entry.name));
            }
            dequant_vbit(bytes, entry.shape[0], entry.shape[1], dst)?;
        }
        TensorDtype::VbitRo => {
            if entry.shape.len() != 2 {
                return Err(format!("vbit_ro tensor '{}' must be 2-D", entry.name));
            }
            dequant_vbit_ro(bytes, entry.shape[0], entry.shape[1], dst)?;
        }
        TensorDtype::Q8_2f => {
            if entry.shape.len() != 2 {
                return Err(format!("q8_2f tensor '{}' must be 2-D", entry.name));
            }
            dequant_q8_2f(bytes, entry.shape[0], entry.shape[1], dst);
        }
        other => {
            return Err(format!(
                "dtype {} of '{}' is reserved — not decodable by this runtime",
                other.name(),
                entry.name
            ))
        }
    }
    Ok(())
}

/// Approximate stored bytes per weight for a dtype (informational).
pub fn bytes_per_weight(dtype: TensorDtype) -> f32 {
    match dtype {
        TensorDtype::F32 => 4.0,
        TensorDtype::F16 | TensorDtype::Bf16 => 2.0,
        TensorDtype::Q8Row | TensorDtype::Q8_2f => 1.0,
        TensorDtype::Q4Block | TensorDtype::Q4Col | TensorDtype::Mix84
        | TensorDtype::Q4Tiled => 0.5625,
        TensorDtype::Vbit | TensorDtype::VbitRo => 0.5,
        TensorDtype::U8 => 1.0,
    }
}

#[cfg(test)]
mod f16_tests {
    use super::{f16_to_f32, f32_to_f16, validate_payload, GROUP_SIZE};

    #[test]
    fn f16_subnormals_decode_correctly() {
        // Smallest positive subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-12);
        // Largest subnormal: 1023 * 2^-24.
        assert!((f16_to_f32(0x03FF) - 6.0975552e-5).abs() < 1e-9);
        // The value that exposed the halving bug (mant=299, subnormal).
        assert!((f16_to_f32(0x812b) - -1.7821789e-5).abs() < 1e-9);
        // Smallest normal (boundary) still correct: 2^-14.
        assert!((f16_to_f32(0x0400) - 6.1035156e-5).abs() < 1e-9);
    }

    #[test]
    fn f16_roundtrip_including_subnormals() {
        for &v in &[0.0f32, 1.0, -2.5, 6.097e-5, 3.0e-5, 5.96e-8, -1.782e-5, 65504.0] {
            let back = f16_to_f32(f32_to_f16(v));
            let tol = (v.abs() * 1e-3).max(1e-9);
            assert!((back - v).abs() <= tol, "roundtrip {v} -> {back}");
        }
    }
    /// §4.9: vbit payload validation — exact length from the width
    /// header, bounds before any slice, width whitelist.
    #[test]
    fn validate_payload_vbit_contract() {
        use crate::types::TensorDtype as D;
        let (rows, cols) = (3usize, 64usize);
        let ng = cols / GROUP_SIZE;
        let bits = [4u8, 3, 8];
        let mut good = bits.to_vec();
        good.extend(std::iter::repeat(0u8).take(rows * ng * 2)); // scales
        for &b in &bits {
            good.extend(std::iter::repeat(0u8).take((cols * b as usize).div_ceil(8)));
        }
        assert!(validate_payload(D::Vbit, &[rows, cols], &good).is_ok());

        // Truncated: shorter than the width header itself.
        assert!(validate_payload(D::Vbit, &[rows, cols], &good[..2]).is_err());
        // One byte short / one byte long.
        assert!(validate_payload(D::Vbit, &[rows, cols], &good[..good.len() - 1]).is_err());
        let mut long = good.clone();
        long.push(0);
        assert!(validate_payload(D::Vbit, &[rows, cols], &long).is_err());
        // Forbidden width (7).
        let mut bad = good.clone();
        bad[0] = 7;
        assert!(validate_payload(D::Vbit, &[rows, cols], &bad).is_err());
        // Non-2D / non-multiple-of-group cols.
        assert!(validate_payload(D::Vbit, &[rows * cols], &good).is_err());
        assert!(validate_payload(D::Vbit, &[rows, 33], &good).is_err());

        // Fixed-size dtype goes through expected_nbytes.
        let q8 = vec![0u8; 2 * 8 + 2 * 2];
        assert!(validate_payload(D::Q8Row, &[2, 8], &q8).is_ok());
        assert!(validate_payload(D::Q8Row, &[2, 8], &q8[..q8.len() - 1]).is_err());
    }

    /// §4.9: vbit is a first-class supported dtype.
    #[test]
    fn vbit_is_supported() {
        assert!(crate::types::TensorDtype::Vbit.is_supported());
    }

}
