//! Signed, normalized Sylvester Walsh-Hadamard transforms used by the
//! Prism/Bonsai ternary extension.

/// Convert one Prism activation value to binary16 using IEEE-754
/// round-to-nearest-even.  This is deliberately Prism-local: the CMF
/// quantization codec has its own historical conversion contract, while the
/// source Prism path crosses a real `f16` activation boundary.  In
/// particular, subnormal halfway ties (`±2^-25`) must round to signed zero,
/// not away from zero.
pub fn prism_f32_to_f16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        return if frac == 0 {
            sign | 0x7c00
        } else {
            // Preserve a useful payload while forcing a quiet NaN.
            sign | 0x7c00 | 0x0200 | ((frac >> 13) as u16 & 0x01ff)
        };
    }

    let unbiased = exp - 127;
    if unbiased < -14 {
        // A binary16 subnormal has a 2^-24 quantum.  Every finite f32 is
        // exactly representable in f64, and multiplying by 2^24 is exact,
        // so this branch can perform the tie/parity decision without a
        // second, approximate floating-point oracle.
        let scaled = (x.abs() as f64) * 16_777_216.0;
        let floor = scaled.floor();
        let frac_part = scaled - floor;
        let mut q = floor as u64;
        if frac_part > 0.5 || (frac_part == 0.5 && (q & 1) != 0) {
            q += 1;
        }
        return if q >= 1024 {
            sign | 0x0400 // carry into the smallest normal
        } else {
            sign | q as u16
        };
    }

    if unbiased > 15 {
        return sign | 0x7c00;
    }

    // Normal binary16 result.  Round the 24-bit f32 significand after the
    // 13 discarded low bits, including the even-parity tie rule.
    let significand = frac | 0x0080_0000;
    // Drop the hidden f32 bit before storing the ten-bit binary16 fraction.
    // Keeping it here would make every exact power of two look like a carry
    // and incorrectly increment the half exponent.
    let mut half_frac = (significand >> 13) & 0x03ff;
    let discarded = significand & 0x1fff;
    if discarded > 0x1000 || (discarded == 0x1000 && (half_frac & 1) != 0) {
        half_frac += 1;
    }
    let mut half_exp = unbiased + 15;
    if half_frac == 0x0400 {
        half_frac = 0;
        half_exp += 1;
    }
    if half_exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((half_exp as u16) << 10) | (half_frac as u16 & 0x03ff)
    }
}

/// Apply the normalized Sylvester FWHT to every `block`-wide chunk in place.
/// `block` must be a power of two and divide `values.len()`.  The butterfly
/// is deliberately f32 and deterministic; SIMD/GPU implementations are
/// checked against this oracle rather than defining a second math contract.
pub fn fwht_f32(values: &mut [f32], block: usize) -> Result<(), String> {
    if block == 0 || !block.is_power_of_two() || values.len() % block != 0 {
        return Err(format!(
            "FWHT block {block} must be a power of two dividing {}",
            values.len()
        ));
    }
    let mut width = 1usize;
    while width < block {
        let stride = width * 2;
        for base in (0..values.len()).step_by(block) {
            for i in (0..block).step_by(stride) {
                for j in 0..width {
                    let a = base + i + j;
                    let b = a + width;
                    let x = values[a];
                    let y = values[b];
                    values[a] = x + y;
                    values[b] = x - y;
                }
            }
        }
        width = stride;
    }
    let inv = (block as f32).sqrt().recip();
    for x in values {
        *x *= inv;
    }
    Ok(())
}

/// Apply the Prism forward activation transform `x * D * H` to a row-vector
/// represented in Rust's natural order.  The sign vector is full-width and
/// is not repeated per FWHT block.
pub fn signed_fwht_forward(values: &mut [f32], signs: &[f32], block: usize) -> Result<(), String> {
    if signs.len() != values.len() {
        return Err(format!(
            "FWHT signs length {} != activation width {}",
            signs.len(),
            values.len()
        ));
    }
    for (x, &s) in values.iter_mut().zip(signs) {
        *x *= s;
    }
    fwht_f32(values, block)
}

/// Apply the inverse embedding transform `z * H * D`.  H is self-inverse
/// after normalization, so the same butterfly is used with the sign vector
/// applied after it.
pub fn signed_fwht_inverse(values: &mut [f32], signs: &[f32], block: usize) -> Result<(), String> {
    if signs.len() != values.len() {
        return Err(format!(
            "FWHT signs length {} != embedding width {}",
            signs.len(),
            values.len()
        ));
    }
    fwht_f32(values, block)?;
    for (x, &s) in values.iter_mut().zip(signs) {
        *x *= s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow, independent reference for a normalized Sylvester block.  The
    /// production oracle above is an in-place butterfly; using the direct
    /// Walsh sign matrix here catches a shared butterfly/indexing mistake in
    /// both CPU and device tests.  Accumulate in f64 so this also separates
    /// FWHT arithmetic drift from the declared f32 boundary.
    fn reference_fwht(values: &[f32], block: usize) -> Vec<f32> {
        assert!(block.is_power_of_two() && values.len() % block == 0);
        let inv = 1.0f64 / (block as f64).sqrt();
        let mut out = vec![0.0f32; values.len()];
        for base in (0..values.len()).step_by(block) {
            for row in 0..block {
                let mut sum = 0.0f64;
                for col in 0..block {
                    let sign = if (row & col).count_ones() & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sum += sign * values[base + col] as f64;
                }
                out[base + row] = (sum * inv) as f32;
            }
        }
        out
    }

    fn patterned(width: usize, salt: usize) -> Vec<f32> {
        (0..width)
            .map(|i| {
                let a = ((i.wrapping_mul(7919) + salt * 313) % 4093) as f32;
                let b = ((i.wrapping_mul(3571) + salt * 97) % 1021) as f32;
                (a - 2046.0) / 257.0 + (b - 510.0) / 4096.0
            })
            .collect()
    }

    fn signed_reference(values: &[f32], signs: &[f32], block: usize) -> Vec<f32> {
        let signed: Vec<f32> = values
            .iter()
            .zip(signs)
            .map(|(&x, &s)| x * s)
            .collect();
        reference_fwht(&signed, block)
    }

    #[test]
    fn normalized_fwht_is_its_own_inverse() {
        let signs = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let mut x = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>();
        let original = x.clone();
        signed_fwht_forward(&mut x, &signs, 8).unwrap();
        signed_fwht_inverse(&mut x, &signs, 8).unwrap();
        for (a, b) in x.iter().zip(original) {
            assert!((a - b).abs() < 2e-6, "{a} != {b}");
        }
    }

    #[test]
    fn block_boundaries_are_independent() {
        let mut x = vec![0.0; 16];
        x[3] = 1.0;
        fwht_f32(&mut x, 8).unwrap();
        assert!(x[8..].iter().all(|v| *v == 0.0));
        assert!((x[..8].iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f32_fwht_matches_independent_reference_at_prism_widths() {
        for &(width, salt) in &[(1024usize, 3usize), (5120, 5), (6144, 7), (17408, 11)] {
            let input = patterned(width, salt);
            let signs: Vec<f32> = (0..width)
                .map(|i| if (i * 17 + i / 7 + salt) & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            let want = signed_reference(&input, &signs, 1024);
            let mut got = input.clone();
            signed_fwht_forward(&mut got, &signs, 1024).unwrap();
            let (mut num, mut den, mut max_rel) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in got.iter().zip(&want) {
                let d = a as f64 - b as f64;
                num += d * d;
                den += (b as f64) * (b as f64);
                max_rel = max_rel.max(d.abs() / (b.abs() as f64).max(1e-6));
            }
            let rel_rms = (num / den.max(1e-30)).sqrt();
            assert!(
                rel_rms <= 1e-5 && max_rel <= 1e-3,
                "width {width}: independent FWHT drift rel_rms={rel_rms:.3e} max_rel={max_rel:.3e}"
            );

            signed_fwht_inverse(&mut got, &signs, 1024).unwrap();
            let roundtrip = input
                .iter()
                .zip(&got)
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(roundtrip <= 2e-3, "width {width}: FWHT roundtrip rms={roundtrip:.3e}");
        }
    }

    #[test]
    fn signed_fwht_reference_covers_impulses_and_half_boundaries() {
        let width = 1024usize;
        let signs: Vec<f32> = (0..width)
            .map(|i| if (i * 13) & 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut input = vec![0.0f32; width];
        for &i in &[0usize, 1, 31, 511, 1023] {
            input[i] = if i & 1 == 0 { 1.0 } else { -0.5 };
        }
        // Include values at both normal and subnormal f16 boundaries.  The
        // direct f64 oracle is computed before the declared f16 cast, so a
        // later GPU test can compare the raw f32 and rounded output halves
        // separately rather than hiding boundary errors in one tolerance.
        input[127] = f32::from_bits(1);
        input[255] = f32::from_bits(0x3380_0000); // 2^-24, an f16 halfway case
        input[767] = 65504.0;
        let want = signed_reference(&input, &signs, 1024);
        let mut got = input.clone();
        signed_fwht_forward(&mut got, &signs, 1024).unwrap();
        let rms = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(rms <= 2e-4, "FWHT impulse/boundary rms={rms:.3e}");
        for &v in &want {
            let h = crate::quant::f32_to_f16(v);
            let back = crate::quant::f16_to_f32(h);
            assert!(back.is_finite(), "f16 boundary produced non-finite {v}");
        }
    }

    #[test]
    fn prism_f16_rne_covers_zero_ties_subnormals_and_overflow() {
        assert_eq!(prism_f32_to_f16_rne(0.0), 0x0000);
        assert_eq!(prism_f32_to_f16_rne(-0.0), 0x8000);
        // Exactly halfway between binary16 zero and its least subnormal.
        assert_eq!(prism_f32_to_f16_rne(2f32.powi(-25)), 0x0000);
        assert_eq!(prism_f32_to_f16_rne(-2f32.powi(-25)), 0x8000);
        // The next tie is resolved to the even subnormal payload (2).
        assert_eq!(prism_f32_to_f16_rne(3f32 * 2f32.powi(-25)), 0x0002);
        assert_eq!(prism_f32_to_f16_rne(2f32.powi(-24)), 0x0001);
        assert_eq!(prism_f32_to_f16_rne(f32::from_bits(1)), 0x0000);
        assert_eq!(prism_f32_to_f16_rne(2f32.powi(-14)), 0x0400);
        assert_eq!(prism_f32_to_f16_rne(65504.0), 0x7bff);
        assert_eq!(prism_f32_to_f16_rne(65520.0), 0x7c00);
        assert_eq!(prism_f32_to_f16_rne(-65520.0), 0xfc00);
        assert_eq!(prism_f32_to_f16_rne(f32::INFINITY), 0x7c00);
        assert_eq!(prism_f32_to_f16_rne(f32::NEG_INFINITY), 0xfc00);
        assert!((prism_f32_to_f16_rne(f32::NAN) & 0x7c00) == 0x7c00);
    }
}
