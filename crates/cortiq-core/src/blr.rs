//! Binary low-rank weights (`blr`, dtype 19) — docs/BLR_NVG.md §2.
//!
//! A linear layer `W [d_out, d_in]` is stored as `P ∈ {1, 2}` paths, each
//! `diag(h) · S_U · diag(ℓ) · S_Vᵀ · diag(g)` with `S_U ∈ {±1}^{d_out×r}`,
//! `S_V ∈ {±1}^{d_in×r}` and f16 scales. The factorization layout is the one
//! of LittleBit (Lee et al., NeurIPS 2025); here it is only a storage
//! container — which signs and scales go into it is the encoder's business
//! (`cortiq convert --quant blr`, NVG or the dual-SVID baseline).
//!
//! Payload (little-endian):
//!
//! ```text
//! header 16 B: u8 magic 0x4C, u8 version 1, u8 P, u8 0, u32 r, u32 d_out, u32 d_in
//! per path p: h f16[d_out] | g f16[d_in] | ℓ f16[r]
//!             U: d_out rows × r/8 bytes   bit (i,k) = byte[i·r/8 + k/8] >> (k%8) & 1
//!             V: r rows × d_in/8 bytes    bit (k,j) = byte[k·d_in/8 + j/8] >> (j%8) & 1
//! ```
//!
//! Bit 1 → +1, bit 0 → −1; sign(0) encodes as 1. `V` is stored transposed (one
//! row per latent) so both kernel stages stream contiguous rows. The magic
//! byte and the layout are byte-identical to the research branch's files, so
//! those stay readable.

use crate::quant::{f16_to_f32, f32_to_f16};

pub const BLR_MAGIC: u8 = 0x4C;
pub const BLR_VERSION: u8 = 1;
pub const BLR_HEADER: usize = 16;

/// Parsed `blr` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlrHeader {
    pub paths: usize,
    pub rank: usize,
    pub d_out: usize,
    pub d_in: usize,
}

/// Byte offsets of one path's sections inside the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlrPathOffsets {
    pub h: usize,
    pub g: usize,
    pub l: usize,
    pub u: usize,
    pub v: usize,
}

/// Bytes of one path: `2(d_out + d_in + r) + r(d_out + d_in)/8`.
pub fn blr_path_bytes(d_out: usize, d_in: usize, r: usize) -> usize {
    2 * (d_out + d_in + r) + d_out * (r / 8) + r * (d_in / 8)
}

/// Exact payload length for `paths` paths of rank `r`.
pub fn blr_payload_len(paths: usize, r: usize, d_out: usize, d_in: usize) -> usize {
    BLR_HEADER + paths * blr_path_bytes(d_out, d_in, r)
}

/// Strict bit count of the container (scales at 16 bits; the 16-byte header
/// is not counted): `P · [r(d_out + d_in) + 16(d_out + d_in) + 16 r]`.
pub fn blr_bits(d_out: usize, d_in: usize, r: usize, paths: usize) -> u64 {
    paths as u64 * (r as u64 * (d_out + d_in) as u64 + 16 * (d_out + d_in) as u64 + 16 * r as u64)
}

/// Rank rule (docs/BLR_NVG.md §1): float `r*` for the budget `b` bits per
/// weight, times `mult` (the K/V factor) before rounding, floored to a
/// multiple of 8, at least 8 and at most `min(d_out, d_in)` rounded down to 8.
pub fn blr_rank(d_out: usize, d_in: usize, b: f64, paths: usize, mult: f64) -> usize {
    let (m, n) = (d_out as f64, d_in as f64);
    let s = if paths == 2 {
        (b * m * n - 32.0 * (m + n)) / (2.0 * (m + n + 16.0))
    } else {
        (b * m * n - 16.0 * (m + n)) / (m + n + 16.0)
    };
    let r = ((s * mult / 8.0).floor() * 8.0).max(8.0) as usize;
    let cap = (d_out.min(d_in) / 8) * 8;
    r.min(cap.max(8))
}

/// Parse and sanity-check the 16-byte header (no length check).
pub fn blr_parse_header(bytes: &[u8]) -> Result<BlrHeader, String> {
    if bytes.len() < BLR_HEADER {
        return Err(format!("blr payload {} bytes < 16-byte header", bytes.len()));
    }
    if bytes[0] != BLR_MAGIC {
        return Err(format!("blr magic {:#04x} != 0x4c", bytes[0]));
    }
    if bytes[1] != BLR_VERSION {
        return Err(format!("blr version {} unsupported (want 1)", bytes[1]));
    }
    let paths = bytes[2] as usize;
    if !(1..=2).contains(&paths) {
        return Err(format!("blr paths {paths} not in {{1,2}}"));
    }
    if bytes[3] != 0 {
        return Err("blr reserved byte must be 0".into());
    }
    let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    Ok(BlrHeader {
        paths,
        rank: rd(4),
        d_out: rd(8),
        d_in: rd(12),
    })
}

/// Section offsets of path `p`.
pub fn blr_path_offsets(h: &BlrHeader, p: usize) -> BlrPathOffsets {
    let base = BLR_HEADER + p * blr_path_bytes(h.d_out, h.d_in, h.rank);
    let hh = base;
    let g = hh + 2 * h.d_out;
    let l = g + 2 * h.d_in;
    let u = l + 2 * h.rank;
    let v = u + h.d_out * (h.rank / 8);
    BlrPathOffsets { h: hh, g, l, u, v }
}

/// Full validation against the directory shape: header fields, constraints,
/// exact length — before any slice is taken.
pub fn validate_blr(shape: &[usize], bytes: &[u8]) -> Result<BlrHeader, String> {
    if shape.len() != 2 {
        return Err(format!("blr tensor must be 2-D, got {shape:?}"));
    }
    let h = blr_parse_header(bytes)?;
    if h.d_out != shape[0] || h.d_in != shape[1] {
        return Err(format!(
            "blr header dims {}x{} != shape {shape:?}",
            h.d_out, h.d_in
        ));
    }
    if h.d_out == 0 || h.d_in == 0 || h.d_in % 8 != 0 {
        return Err(format!(
            "blr d_in {} must be a positive multiple of 8",
            h.d_in
        ));
    }
    if h.rank < 8 || h.rank % 8 != 0 || h.rank > h.d_out.min(h.d_in) {
        return Err(format!(
            "blr rank {} must be a multiple of 8 in [8, min({}, {})]",
            h.rank, h.d_out, h.d_in
        ));
    }
    let want = blr_payload_len(h.paths, h.rank, h.d_out, h.d_in);
    if bytes.len() != want {
        return Err(format!(
            "blr payload length {} != computed {want} (P {} r {} {}x{})",
            bytes.len(),
            h.paths,
            h.rank,
            h.d_out,
            h.d_in
        ));
    }
    Ok(h)
}

/// Read `n` f16 values at `off` as f32.
pub fn blr_f16s(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            f16_to_f32(u16::from_le_bytes([
                bytes[off + 2 * i],
                bytes[off + 2 * i + 1],
            ]))
        })
        .collect()
}

#[inline]
fn bit(bytes: &[u8], off: usize, idx: usize) -> bool {
    (bytes[off + idx / 8] >> (idx % 8)) & 1 == 1
}

/// Reference dequant: `W[i,:] = Σ_p h_p[i] Σ_k ℓ_p[k] U_p(i,k) (V_p(k,:) ⊙ g_p)`.
pub fn dequant_blr(bytes: &[u8], rows: usize, cols: usize, dst: &mut [f32]) -> Result<(), String> {
    let h = validate_blr(&[rows, cols], bytes)?;
    if dst.len() != rows * cols {
        return Err(format!("blr dst len {} != {}", dst.len(), rows * cols));
    }
    dst.iter_mut().for_each(|v| *v = 0.0);
    let r = h.rank;
    let mut vg = vec![0f32; r * cols];
    for p in 0..h.paths {
        let o = blr_path_offsets(&h, p);
        let hv = blr_f16s(bytes, o.h, rows);
        let gv = blr_f16s(bytes, o.g, cols);
        let lv = blr_f16s(bytes, o.l, r);
        for k in 0..r {
            let row = &mut vg[k * cols..(k + 1) * cols];
            for j in 0..cols {
                let s = if bit(bytes, o.v + k * (cols / 8), j) {
                    1.0
                } else {
                    -1.0
                };
                row[j] = s * gv[j];
            }
        }
        let mut acc = vec![0f32; cols];
        for i in 0..rows {
            acc.iter_mut().for_each(|v| *v = 0.0);
            for k in 0..r {
                let s = if bit(bytes, o.u + i * (r / 8), k) {
                    lv[k]
                } else {
                    -lv[k]
                };
                let src = &vg[k * cols..(k + 1) * cols];
                for j in 0..cols {
                    acc[j] += s * src[j];
                }
            }
            let d = &mut dst[i * cols..(i + 1) * cols];
            for j in 0..cols {
                d[j] += hv[i] * acc[j];
            }
        }
    }
    Ok(())
}

/// One path to encode. Signs as bools (`true` = +1), `su[i·r + k]` and
/// `sv[j·r + k]` (both row-major over their own rows, latent index fastest).
pub struct BlrPathIn<'a> {
    pub h: &'a [f32],
    pub g: &'a [f32],
    pub l: &'a [f32],
    pub su: &'a [bool],
    pub sv: &'a [bool],
}

/// Pack paths into a `blr` payload. Scales are written as given (the encoder
/// rebalances them before rounding).
pub fn encode_blr(d_out: usize, d_in: usize, r: usize, paths: &[BlrPathIn]) -> Vec<u8> {
    assert!(matches!(paths.len(), 1 | 2));
    assert!(r % 8 == 0 && r >= 8 && d_in % 8 == 0);
    let mut out = Vec::with_capacity(blr_payload_len(paths.len(), r, d_out, d_in));
    out.push(BLR_MAGIC);
    out.push(BLR_VERSION);
    out.push(paths.len() as u8);
    out.push(0);
    out.extend_from_slice(&(r as u32).to_le_bytes());
    out.extend_from_slice(&(d_out as u32).to_le_bytes());
    out.extend_from_slice(&(d_in as u32).to_le_bytes());
    for p in paths {
        assert_eq!(p.h.len(), d_out);
        assert_eq!(p.g.len(), d_in);
        assert_eq!(p.l.len(), r);
        assert_eq!(p.su.len(), d_out * r);
        assert_eq!(p.sv.len(), d_in * r);
        for v in p.h.iter().chain(p.g).chain(p.l) {
            out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
        for i in 0..d_out {
            for kb in 0..r / 8 {
                let mut b = 0u8;
                for t in 0..8 {
                    if p.su[i * r + kb * 8 + t] {
                        b |= 1 << t;
                    }
                }
                out.push(b);
            }
        }
        for k in 0..r {
            for jb in 0..d_in / 8 {
                let mut b = 0u8;
                for t in 0..8 {
                    if p.sv[(jb * 8 + t) * r + k] {
                        b |= 1 << t;
                    }
                }
                out.push(b);
            }
        }
    }
    debug_assert_eq!(out.len(), blr_payload_len(paths.len(), r, d_out, d_in));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    #[test]
    fn rank_rule_anchors() {
        assert_eq!(blr_rank(4096, 4096, 0.55, 2, 1.0), 544);
        assert_eq!(blr_rank(4096, 11008, 0.10, 2, 1.0), 128);
        // Qwen3-0.6B shapes
        assert_eq!(blr_rank(2048, 1024, 1.0, 2, 1.0), 320);
        assert_eq!(blr_rank(1024, 1024, 1.0, 2, 1.0), 232);
        assert_eq!(blr_rank(3072, 1024, 1.0, 2, 1.0), 360);
        assert_eq!(blr_rank(2048, 1024, 0.55, 2, 1.0), 168);
        assert_eq!(blr_rank(1024, 1024, 0.55, 2, 1.0), 120);
        assert_eq!(blr_rank(1024, 3072, 0.55, 2, 1.0), 192);
        assert_eq!(blr_rank(1024, 1024, 0.55, 2, 4.0), 488);
        assert_eq!(blr_rank(1024, 1024, 0.1, 2, 1.0), 8);
        // clamp to min(d_out, d_in)
        assert_eq!(blr_rank(64, 64, 8.0, 2, 1.0), 64);
        let b = blr_bits(4096, 4096, 544, 2) as f64 / (4096.0 * 4096.0);
        assert!((b - 0.5479).abs() < 1e-4, "{b}");
    }

    fn random_paths(
        rng: &mut Rng,
        d_out: usize,
        d_in: usize,
        r: usize,
        p: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>, Vec<bool>)> {
        (0..p)
            .map(|_| {
                let h: Vec<f32> = (0..d_out).map(|_| 0.5 + rng.f()).collect();
                let g: Vec<f32> = (0..d_in).map(|_| 0.5 + rng.f()).collect();
                let l: Vec<f32> = (0..r).map(|_| 0.01 + 0.1 * rng.f()).collect();
                let su: Vec<bool> = (0..d_out * r).map(|_| rng.next() & 1 == 1).collect();
                let sv: Vec<bool> = (0..d_in * r).map(|_| rng.next() & 1 == 1).collect();
                (h, g, l, su, sv)
            })
            .collect()
    }

    #[test]
    fn encode_dequant_roundtrip() {
        let mut rng = Rng(7);
        for &(d_out, d_in, r, p) in &[
            (24usize, 16usize, 8usize, 1usize),
            (40, 32, 16, 2),
            (16, 48, 16, 2),
        ] {
            let ps = random_paths(&mut rng, d_out, d_in, r, p);
            let ins: Vec<BlrPathIn> = ps
                .iter()
                .map(|(h, g, l, su, sv)| BlrPathIn { h, g, l, su, sv })
                .collect();
            let bytes = encode_blr(d_out, d_in, r, &ins);
            assert_eq!(bytes.len(), blr_payload_len(p, r, d_out, d_in));
            let mut w = vec![0f32; d_out * d_in];
            dequant_blr(&bytes, d_out, d_in, &mut w).unwrap();
            let q = |x: f32| f16_to_f32(f32_to_f16(x));
            for i in 0..d_out {
                for j in 0..d_in {
                    let mut want = 0f64;
                    for (h, g, l, su, sv) in &ps {
                        let mut s = 0f64;
                        for k in 0..r {
                            let a = if su[i * r + k] { 1.0 } else { -1.0 };
                            let b = if sv[j * r + k] { 1.0 } else { -1.0 };
                            s += a * b * q(l[k]) as f64;
                        }
                        want += q(h[i]) as f64 * q(g[j]) as f64 * s;
                    }
                    let got = w[i * d_in + j] as f64;
                    assert!(
                        (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "{i},{j}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn validate_rejects_each_field() {
        let mut rng = Rng(3);
        let (d_out, d_in, r) = (32usize, 32usize, 16usize);
        let ps = random_paths(&mut rng, d_out, d_in, r, 2);
        let ins: Vec<BlrPathIn> = ps
            .iter()
            .map(|(h, g, l, su, sv)| BlrPathIn { h, g, l, su, sv })
            .collect();
        let good = encode_blr(d_out, d_in, r, &ins);
        assert!(validate_blr(&[d_out, d_in], &good).is_ok());
        assert!(validate_blr(&[d_out, d_in + 8], &good).is_err());
        assert!(validate_blr(&[d_out * d_in], &good).is_err());
        assert!(validate_blr(&[d_out, d_in], &good[..good.len() - 1]).is_err());
        let mut long = good.clone();
        long.push(0);
        assert!(validate_blr(&[d_out, d_in], &long).is_err());
        assert!(validate_blr(&[d_out, d_in], &good[..10]).is_err());
        let set = |at: usize, v: u8| {
            let mut b = good.clone();
            b[at] = v;
            b
        };
        assert!(validate_blr(&[d_out, d_in], &set(0, 0x4D)).is_err()); // magic
        assert!(validate_blr(&[d_out, d_in], &set(1, 2)).is_err()); // version
        assert!(validate_blr(&[d_out, d_in], &set(2, 3)).is_err()); // paths
        assert!(validate_blr(&[d_out, d_in], &set(2, 1)).is_err()); // length no longer matches
        assert!(validate_blr(&[d_out, d_in], &set(3, 1)).is_err()); // reserved
        assert!(validate_blr(&[d_out, d_in], &set(4, 12)).is_err()); // r % 8
        assert!(validate_blr(&[d_out, d_in], &set(4, 40)).is_err()); // r > min
        assert!(validate_blr(&[d_out, d_in], &set(8, 33)).is_err()); // d_out
        assert!(validate_blr(&[d_out, d_in], &set(12, 33)).is_err()); // d_in
    }

    /// The generic payload gate and the generic decoder both route dtype 19
    /// here: `validate_payload` is what `CmfModel::open`/`verify` run, and
    /// `dequant_tensor` is the dense fallback of every engine path.
    #[test]
    fn generic_codec_entry_points_handle_blr() {
        use crate::format::TensorEntry;
        use crate::types::TensorDtype;
        let mut rng = Rng(5);
        let (d_out, d_in, r) = (48usize, 64usize, 16usize);
        let ps = random_paths(&mut rng, d_out, d_in, r, 2);
        let ins: Vec<BlrPathIn> = ps
            .iter()
            .map(|(h, g, l, su, sv)| BlrPathIn { h, g, l, su, sv })
            .collect();
        let bytes = encode_blr(d_out, d_in, r, &ins);
        let d = TensorDtype::BinaryLowRank;
        assert_eq!(TensorDtype::from_id(19), Some(d));
        assert_eq!(d.id(), 19);
        assert_eq!(d.name(), "blr");
        assert!(d.is_supported());
        assert_eq!(crate::quant::expected_nbytes(d, &[d_out, d_in]), None);
        assert!(crate::quant::validate_payload(d, &[d_out, d_in], &bytes).is_ok());
        assert!(crate::quant::validate_payload(d, &[d_out, d_in], &bytes[1..]).is_err());
        let entry = TensorEntry {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: d,
            shape: vec![d_out, d_in],
            off: 0,
            nbytes: bytes.len() as u64,
            shard: 0,
            hash: 0,
        };
        let mut a = vec![0f32; d_out * d_in];
        let mut b = vec![0f32; d_out * d_in];
        crate::quant::dequant_tensor(&entry, &bytes, &mut a).unwrap();
        dequant_blr(&bytes, d_out, d_in, &mut b).unwrap();
        assert_eq!(a, b);
    }
}
