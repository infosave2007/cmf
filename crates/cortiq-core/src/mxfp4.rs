//! `mxfp4` (dtype 21): OCP Microscaling FP4 — E2M1 elements with one E8M0
//! scale per 32 consecutive elements of a row.
//!
//! Payload of a `[rows, cols]` tensor (`cols % 32 == 0`):
//!
//! ```text
//! [codes : rows · cols/2 bytes]   byte j of row r = elements 2j (low nibble), 2j+1 (high)
//! [scales: rows · cols/32 bytes]  byte g of row r = E8M0 exponent of elements 32g..32g+32
//! ```
//!
//! Element value = `E2M1(code) · 2^(scale − 127)`; scale byte 255 is NaN (OCP).
//! E2M1 code: bit 3 sign, bits 2..0 index `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
//!
//! This is exactly the in-memory layout of OCP MXFP4 checkpoints (DeepSeek-V4
//! `I8` expert weights + `F8_E8M0` 1×32 scales, Kimi-K3 / MiMo-V2 `U8`
//! weights + `U8` scales): the source's two tensors ARE the two planes, so
//! [`repack`] is a checked concatenation and never touches a float.

/// Elements per E8M0 scale (OCP MX block size).
pub const GROUP: usize = 32;
/// Code bytes per 32-element group.
pub const GROUP_BYTES: usize = GROUP / 2;
/// Largest E2M1 magnitude (`emax = 2`, `1.5 · 2^2`).
pub const E2M1_MAX: f32 = 6.0;

/// E2M1 value of every 4-bit code (low three bits = magnitude index, bit 3 =
/// sign). Code 8 is −0.0: `−mag` of the zero magnitude, bit for bit what the
/// converter's historical `unpack_mxfp4` produced.
pub const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// `2 · E2M1(code)` as an exact integer (`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`):
/// the int8 weight the A8W8 kernels feed to `sdot`/`maddubs`; the factor of
/// two moves into the group scale (`2^(e−128)`).
pub const E2M1_X2_I8: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// E8M0 scale byte → f32, exactly `2^(k−127)`: built from the exponent bits,
/// no `exp2` (byte 0 is the subnormal 2^-127; 255 is NaN per OCP).
#[inline]
pub fn e8m0_to_f32(k: u8) -> f32 {
    match k {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((k as u32) << 23),
    }
}

/// `2^(k−128)` — the scale of the doubled int8 weights [`E2M1_X2_I8`].
/// Exact for every finite scale byte (byte 0 gives the subnormal 2^-128,
/// byte 1 the subnormal 2^-127).
#[inline]
pub fn e8m0_half_to_f32(k: u8) -> f32 {
    match k {
        255 => f32::NAN,
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits(((k - 1) as u32) << 23),
    }
}

/// Byte offset of the scale plane (behind every row's codes).
#[inline]
pub fn scale_offset(rows: usize, cols: usize) -> usize {
    rows * cols / 2
}

/// Payload length of a `[rows, cols]` tensor (None for a shape the format
/// does not allow: cols not a positive multiple of 32, or overflow).
pub fn nbytes(rows: usize, cols: usize) -> Option<usize> {
    if cols == 0 || cols % GROUP != 0 {
        return None;
    }
    let n = rows.checked_mul(cols)?;
    (n / 2).checked_add(n / GROUP)
}

/// Split a payload into its (codes, scales) planes.
#[inline]
pub fn planes(bytes: &[u8], rows: usize, cols: usize) -> (&[u8], &[u8]) {
    let off = scale_offset(rows, cols);
    (&bytes[..off], &bytes[off..off + rows * cols / GROUP])
}

/// Dequantize a full tensor into f32, exactly: every element is
/// `E2M1_LUT[code] · 2^(k−127)`, a product of a ≤2-bit and a 1-bit mantissa
/// (no rounding; only an exponent past f32's range — scale bytes ≥ 253 with
/// a large code — saturates to ±inf, and a 255 scale yields NaN).
pub fn dequant(bytes: &[u8], rows: usize, cols: usize, dst: &mut [f32]) {
    debug_assert_eq!(cols % GROUP, 0);
    debug_assert!(dst.len() >= rows * cols);
    let (codes, scales) = planes(bytes, rows, cols);
    let gpr = cols / GROUP;
    for r in 0..rows {
        for g in 0..gpr {
            let s = e8m0_to_f32(scales[r * gpr + g]);
            let src = &codes[(r * gpr + g) * GROUP_BYTES..(r * gpr + g + 1) * GROUP_BYTES];
            let out = &mut dst[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            for (j, &b) in src.iter().enumerate() {
                out[2 * j] = E2M1_LUT[(b & 0x0F) as usize] * s;
                out[2 * j + 1] = E2M1_LUT[(b >> 4) as usize] * s;
            }
        }
    }
}

/// Lossless re-pack of a source MXFP4 tensor — `codes` `[rows, cols/2]`
/// (low nibble = even column) and E8M0 `scales` `[rows, cols/32]`, both
/// row-major — into the `mxfp4` payload. No float is formed: the result is
/// the two planes back to back. A NaN scale (255) is refused, as the
/// float decoders refuse it, rather than carried into a model.
pub fn repack(codes: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if cols == 0 || cols % GROUP != 0 {
        return Err(format!("mxfp4: cols {cols} not a positive multiple of {GROUP}"));
    }
    let gpr = cols / GROUP;
    if codes.len() != rows * cols / 2 || scales.len() != rows * gpr {
        return Err(format!(
            "mxfp4: codes {} / scales {} bytes do not fit [{rows}, {cols}] (want {} / {})",
            codes.len(),
            scales.len(),
            rows * cols / 2,
            rows * gpr
        ));
    }
    if let Some(i) = scales.iter().position(|&k| k == 255) {
        return Err(format!(
            "mxfp4: NaN scale (E8M0 255) at row {} group {}",
            i / gpr,
            i % gpr
        ));
    }
    let mut out = Vec::with_capacity(codes.len() + scales.len());
    out.extend_from_slice(codes);
    out.extend_from_slice(scales);
    Ok(out)
}

/// `floor(log2(a))` of a positive finite f32, exactly (from the bits, so
/// subnormals and exact powers of two are right).
#[inline]
fn floor_log2(a: f32) -> i32 {
    let bits = a.to_bits() & 0x7FFF_FFFF;
    let e = (bits >> 23) as i32;
    if e > 0 {
        e - 127
    } else {
        let m = bits & 0x007F_FFFF;
        (31 - m.leading_zeros() as i32) - 149
    }
}

/// Round a non-negative magnitude (already divided by the group scale) to
/// the nearest E2M1 magnitude index, ties to the even index (even mantissa),
/// saturating at 6.
#[inline]
fn round_e2m1(a: f64) -> u8 {
    const GRID: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if a >= 6.0 {
        return 7;
    }
    let mut i = 0usize;
    while i < 7 && GRID[i + 1] <= a {
        i += 1;
    }
    if i == 7 || a == GRID[i] {
        return i as u8;
    }
    let mid = (GRID[i] + GRID[i + 1]) * 0.5;
    if a < mid {
        i as u8
    } else if a > mid {
        (i + 1) as u8
    } else if i % 2 == 0 {
        i as u8
    } else {
        (i + 1) as u8
    }
}

/// Encode one 32-element group per the OCP MX rule: shared exponent
/// `floor(log2(amax)) − 2` (`emax` of E2M1), clamped to E8M0's
/// `[−127, 127]`; each element divided by `2^e` (exact) and rounded to
/// nearest-even on the E2M1 grid, saturating to ±6. An all-zero group gets
/// scale byte 0 (the reference's `log2(FP32_MIN_NORMAL) − 2` clamped);
/// a group holding a NaN or ±inf gets the NaN scale 255 (OCP) with zero
/// codes. Zero always encodes as code 0 (never −0).
pub fn encode_group(group: &[f32]) -> ([u8; GROUP_BYTES], u8) {
    debug_assert_eq!(group.len(), GROUP);
    let mut codes = [0u8; GROUP_BYTES];
    if group.iter().any(|v| !v.is_finite()) {
        return (codes, 255);
    }
    let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return (codes, 0);
    }
    let e = (floor_log2(amax) - 2).clamp(-127, 127);
    let inv = (-(e as f64)).exp2(); // exact power of two in f64
    for (j, &v) in group.iter().enumerate() {
        let q = v as f64 * inv;
        let mag = round_e2m1(q.abs());
        let code = if mag != 0 && q < 0.0 { mag | 8 } else { mag };
        codes[j / 2] |= if j % 2 == 0 { code } else { code << 4 };
    }
    (codes, (e + 127) as u8)
}

/// Quantize a row-major `[rows, cols]` f32 matrix into the `mxfp4` payload
/// (OCP rule, see [`encode_group`]). `cols % 32 == 0`.
pub fn quantize(vals: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % GROUP, 0, "mxfp4 needs cols % 32 == 0");
    assert_eq!(vals.len(), rows * cols, "mxfp4: value count");
    let gpr = cols / GROUP;
    let mut out = vec![0u8; nbytes(rows, cols).expect("mxfp4 shape")];
    let (codes, scales) = out.split_at_mut(scale_offset(rows, cols));
    for g in 0..rows * gpr {
        let (c, s) = encode_group(&vals[g * GROUP..(g + 1) * GROUP]);
        codes[g * GROUP_BYTES..(g + 1) * GROUP_BYTES].copy_from_slice(&c);
        scales[g] = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent decode of one code: sign bit, then E2M1 with bias 1
    /// (e = 0 subnormal m·0.5, else (1 + m/2)·2^(e−1)).
    fn e2m1_ref(code: u8) -> f64 {
        let e = ((code >> 1) & 3) as i32;
        let m = (code & 1) as f64;
        let mag = if e == 0 {
            m * 0.5
        } else {
            (1.0 + m / 2.0) * 2f64.powi(e - 1)
        };
        if code & 8 != 0 { -mag } else { mag }
    }

    #[test]
    fn tables_match_the_ocp_definition() {
        for code in 0..16u8 {
            assert_eq!(E2M1_LUT[code as usize] as f64, e2m1_ref(code), "code {code}");
            assert_eq!(E2M1_X2_I8[code as usize] as f64, 2.0 * e2m1_ref(code));
        }
        assert_eq!(E2M1_LUT[8].to_bits(), (-0.0f32).to_bits());
        for k in 0..255u8 {
            let want = 2f64.powi(k as i32 - 127);
            assert_eq!(e8m0_to_f32(k) as f64, want, "e8m0 {k}");
            assert_eq!(e8m0_half_to_f32(k) as f64, want / 2.0, "e8m0/2 {k}");
        }
        assert!(e8m0_to_f32(255).is_nan() && e8m0_half_to_f32(255).is_nan());
    }

    #[test]
    fn dtype_identity_size_validation_and_generic_dequant() {
        use crate::format::TensorEntry;
        use crate::quant::{bytes_per_weight, dequant_tensor, expected_nbytes, validate_payload};
        use crate::types::TensorDtype;
        let d = TensorDtype::Mxfp4;
        assert_eq!(d.id(), 21);
        assert_eq!(TensorDtype::from_id(21), Some(d));
        assert_eq!(d.name(), "mxfp4");
        assert!(d.is_supported() && !d.is_raw());
        assert_eq!(serde_json::to_string(&d).unwrap(), "\"mxfp4\"");
        assert_eq!(serde_json::from_str::<TensorDtype>("\"mxfp4\"").unwrap(), d);
        assert_eq!(bytes_per_weight(d) * 8.0, 4.25);
        let (rows, cols) = (5usize, 96usize);
        let n = rows * cols;
        assert_eq!(expected_nbytes(d, &[rows, cols]), Some(n * 17 / 32));
        assert_eq!(expected_nbytes(d, &[rows, 48]), None);
        assert_eq!(expected_nbytes(d, &[n]), None);
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 - 200.0) * 0.01).collect();
        let good = quantize(&vals, rows, cols);
        assert!(validate_payload(d, &[rows, cols], &good).is_ok());
        assert!(validate_payload(d, &[rows, cols], &good[1..]).is_err());
        let e = validate_payload(d, &[rows, 48], &good).unwrap_err();
        assert!(e.contains("multiple of 32"), "{e}");
        let e = validate_payload(d, &[n], &good).unwrap_err();
        assert!(e.contains("2-D"), "{e}");
        let entry = TensorEntry {
            name: "w".into(),
            dtype: d,
            shape: vec![rows, cols],
            off: 0,
            nbytes: good.len() as u64,
            shard: 0,
            hash: 0,
        };
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        dequant_tensor(&entry, &good, &mut a).unwrap();
        dequant(&good, rows, cols, &mut b);
        assert_eq!(a, b);
        assert!(dequant_tensor(&entry, &good[..10], &mut a).is_err());
    }

    #[test]
    fn size_is_4_25_bits_per_weight() {
        assert_eq!(nbytes(2304, 5120), Some(2304 * 5120 * 17 / 32));
        assert_eq!(nbytes(3, 64).unwrap() * 8, 3 * 64 * 17 / 4);
        assert_eq!(nbytes(4, 48), None);
        assert_eq!(nbytes(4, 0), None);
    }

    #[test]
    fn repack_is_the_two_source_planes_and_refuses_nan() {
        let (rows, cols) = (3usize, 64usize);
        let codes: Vec<u8> = (0..rows * cols / 2).map(|i| (i * 37 + 11) as u8).collect();
        let scales: Vec<u8> = (0..rows * cols / 32).map(|i| (120 + i) as u8).collect();
        let p = repack(&codes, &scales, rows, cols).unwrap();
        let (c, s) = planes(&p, rows, cols);
        assert_eq!(c, &codes[..]);
        assert_eq!(s, &scales[..]);
        assert_eq!(p.len(), nbytes(rows, cols).unwrap());
        let mut bad = scales.clone();
        bad[4] = 255;
        let e = repack(&codes, &bad, rows, cols).unwrap_err();
        assert!(e.contains("row 2 group 0"), "{e}");
        assert!(repack(&codes[1..], &scales, rows, cols).is_err());
        assert!(repack(&codes, &scales, rows, 48).is_err());
    }

    #[test]
    fn dequant_is_lut_times_power_of_two_for_every_code_and_scale() {
        // One row per scale byte 0..=254, every code twice per group.
        let (rows, cols) = (255usize, 32usize);
        let mut codes = Vec::new();
        for _ in 0..rows {
            for j in 0..16u8 {
                codes.push(j | (((j + 5) & 15) << 4));
            }
        }
        let scales: Vec<u8> = (0..rows as u8).collect();
        let p = repack(&codes, &scales, rows, cols).unwrap();
        let mut d = vec![0f32; rows * cols];
        dequant(&p, rows, cols, &mut d);
        for r in 0..rows {
            for c in 0..cols {
                let b = codes[r * 16 + c / 2];
                let code = if c % 2 == 0 { b & 15 } else { b >> 4 };
                let want = e2m1_ref(code) * 2f64.powi(r as i32 - 127);
                let got = d[r * cols + c];
                if want.abs() <= f32::MAX as f64 {
                    // Exact: the f32 product equals the real value.
                    assert_eq!(got as f64, want, "row {r} col {c}");
                } else {
                    assert!(got.is_infinite() && (got < 0.0) == (want < 0.0));
                }
            }
        }
        // NaN scale (only reachable through a hand-built payload) → NaN group.
        let mut p2 = p.clone();
        let off = scale_offset(rows, cols);
        p2[off + 3] = 255;
        dequant(&p2, rows, cols, &mut d);
        assert!(d[3 * cols..4 * cols].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn rounding_is_nearest_even_on_the_e2m1_grid_and_saturates() {
        // Ties: 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2, 3.5→4, 5→4.
        let cases = [
            (0.25, 0.0),
            (0.26, 0.5),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.01, 6.0),
            (6.0, 6.0),
            (7.99, 6.0),
        ];
        for (a, want) in cases {
            assert_eq!(E2M1_LUT[round_e2m1(a) as usize] as f64, want, "{a}");
        }
    }

    #[test]
    fn quantize_follows_the_ocp_shared_exponent_rule() {
        // amax = 7.5 → floor(log2) = 2 → e = 0 → scale byte 127; 7.5 clips to 6.
        let mut g = [0f32; 32];
        g[0] = 7.5;
        g[1] = -1.25; // tie 1.0/1.5 → 1.0
        g[2] = 0.1; // → 0
        g[3] = -0.3; // → −0.5
        let (c, s) = encode_group(&g);
        assert_eq!(s, 127);
        assert_eq!(c[0] & 15, 7);
        assert_eq!(c[0] >> 4, 8 | 2);
        assert_eq!(c[1] & 15, 0);
        assert_eq!(c[1] >> 4, 8 | 1);
        // amax exactly 4 → e = 0; amax 3.99 → e = −1 (values ×2).
        g = [0f32; 32];
        g[5] = 4.0;
        assert_eq!(encode_group(&g).1, 127);
        g[5] = 3.99;
        assert_eq!(encode_group(&g).1, 126);
        // All-zero group → scale byte 0, zero codes; −0 encodes as +0.
        g = [0f32; 32];
        g[7] = -0.0;
        assert_eq!(encode_group(&g), ([0u8; 16], 0));
        // Non-finite → NaN scale (OCP), zero codes.
        g[7] = f32::NAN;
        assert_eq!(encode_group(&g).1, 255);
        g[7] = f32::INFINITY;
        assert_eq!(encode_group(&g).1, 255);
        // Tiny and huge groups clamp to E8M0's range.
        g = [0f32; 32];
        g[0] = f32::from_bits(1); // 2^-149 → e = −151 → clamp −127
        assert_eq!(encode_group(&g).1, 0);
        g[0] = f32::MAX; // floor(log2) = 127 → e = 125
        assert_eq!(encode_group(&g).1, 252);
    }

    #[test]
    fn quantize_dequant_error_is_bounded_and_grid_values_round_trip() {
        let (rows, cols) = (8usize, 96usize);
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let x = ((i * 2654435761usize) % 10007) as f32 / 10007.0 - 0.5;
                x * (1.0 + (i % 7) as f32) * 0.03
            })
            .collect();
        let p = quantize(&vals, rows, cols);
        assert_eq!(p.len(), nbytes(rows, cols).unwrap());
        let mut d = vec![0f32; rows * cols];
        dequant(&p, rows, cols, &mut d);
        for g in 0..rows * cols / GROUP {
            let src = &vals[g * GROUP..(g + 1) * GROUP];
            let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
            let e = floor_log2(amax) - 2;
            let s = 2f64.powi(e);
            for (j, &v) in src.iter().enumerate() {
                let err = (d[g * GROUP + j] as f64 - v as f64).abs();
                let q = (v as f64 / s).abs();
                // Inside the grid: half the local step (≤ 1·s between 4 and
                // 6); in the clipped band [6, 8): the distance to 6·s.
                let bound = if q >= 6.0 { q * s - 6.0 * s } else { s };
                assert!(err <= bound + 1e-12, "group {g} elem {j}: err {err} bound {bound}");
            }
        }
        // Values already on the grid survive a quantize → dequant unchanged,
        // and the payload re-quantizes to itself byte for byte.
        let p2 = quantize(&d, rows, cols);
        let mut d2 = vec![0f32; rows * cols];
        dequant(&p2, rows, cols, &mut d2);
        assert_eq!(d, d2);
        assert_eq!(p, p2);
    }
}
