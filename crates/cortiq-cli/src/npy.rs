//! Minimal NumPy `.npy` reader — converter-side only, for the defrag
//! keep-set (`ffn_keep.npy`) and baked FFN overlays (`tensors/*.npy`)
//! emitted by the pruning pipeline. Supports C-order float32/float64/bool
//! arrays (versions 1.0 and 2.0/3.0); anything else is a loud error.

use std::path::Path;

pub enum NpyData {
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

pub fn read(path: &Path) -> anyhow::Result<Npy> {
    let buf =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    if buf.len() < 12 || &buf[0..6] != b"\x93NUMPY" {
        anyhow::bail!("{}: not a .npy file", path.display());
    }
    // v1.0 uses a u16 header length; v2.0+ a u32.
    let (header, data_start) = if buf[6] >= 2 {
        let hlen = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        (std::str::from_utf8(&buf[12..12 + hlen])?, 12 + hlen)
    } else {
        let hlen = u16::from_le_bytes(buf[8..10].try_into().unwrap()) as usize;
        (std::str::from_utf8(&buf[10..10 + hlen])?, 10 + hlen)
    };
    if header_bool(header, "fortran_order") {
        anyhow::bail!("{}: fortran_order arrays not supported", path.display());
    }
    let descr = field(header, "descr")
        .ok_or_else(|| anyhow::anyhow!("{}: npy header has no descr", path.display()))?;
    let shape = parse_shape(header)
        .ok_or_else(|| anyhow::anyhow!("{}: npy header has no shape", path.display()))?;
    let numel: usize = shape.iter().product();
    let raw = &buf[data_start..];

    let data = match descr.trim_start_matches(['<', '=', '|']) {
        "f4" => {
            need(raw, numel * 4, path)?;
            NpyData::F32(
                (0..numel)
                    .map(|i| f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect(),
            )
        }
        "f8" => {
            need(raw, numel * 8, path)?;
            NpyData::F32(
                (0..numel)
                    .map(|i| {
                        f64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()) as f32
                    })
                    .collect(),
            )
        }
        "b1" => {
            need(raw, numel, path)?;
            NpyData::Bool((0..numel).map(|i| raw[i] != 0).collect())
        }
        other => anyhow::bail!("{}: unsupported npy dtype '{other}'", path.display()),
    };
    Ok(Npy { shape, data })
}

fn need(raw: &[u8], n: usize, path: &Path) -> anyhow::Result<()> {
    if raw.len() < n {
        anyhow::bail!("{}: truncated npy data ({} < {n})", path.display(), raw.len());
    }
    Ok(())
}

/// Value of a `'key': '<value>'` string field in the header dict literal.
fn field(h: &str, key: &str) -> Option<String> {
    let k = format!("'{key}':");
    let after = &h[h.find(&k)? + k.len()..];
    let q0 = after.find('\'')?;
    let rest = &after[q0 + 1..];
    let q1 = rest.find('\'')?;
    Some(rest[..q1].to_string())
}

/// `'key': True|False` (defaults to false if absent).
fn header_bool(h: &str, key: &str) -> bool {
    let k = format!("'{key}':");
    match h.find(&k) {
        Some(i) => h[i + k.len()..]
            .split(',')
            .next()
            .unwrap_or("")
            .contains("True"),
        None => false,
    }
}

/// `'shape': (a, b, ...)` → dims. Handles the 1-D trailing-comma form `(n,)`.
fn parse_shape(h: &str) -> Option<Vec<usize>> {
    let after = &h[h.find("'shape':")? + "'shape':".len()..];
    let lp = after.find('(')?;
    let rp = after[lp..].find(')')? + lp;
    Some(
        after[lp + 1..rp]
            .split(',')
            .filter_map(|p| {
                let t = p.trim();
                (!t.is_empty()).then(|| t.parse::<usize>().ok())
            })
            .collect::<Option<Vec<usize>>>()?,
    )
}
