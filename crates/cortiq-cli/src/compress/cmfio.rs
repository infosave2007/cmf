//! CMF → CMF rewrites for `compress`: copy a container tensor by tensor,
//! replacing some tensors with new f32 values (a narrowed FFN), and either
//! keep every other payload byte for byte or re-encode the float 2-D
//! weights into a converter profile (the q8_2f calibration sibling, or a
//! final codec that needs no GPTQ).

use crate::convert::{Quant, effective_quant, keeps_float, quantize_2d};
use anyhow::{Result, anyhow};
use cortiq_core::format::{CmfHeader, CmfModel, CmfStreamWriter, TensorEntry};
use cortiq_core::quant::dequant_tensor;
use cortiq_core::types::TensorDtype;
use std::path::Path;

/// A replacement tensor: its new shape and f32 values.
pub type Replacement = (Vec<usize>, Vec<f32>);

/// How the payloads of the new file are encoded.
#[derive(Clone, Copy, Debug)]
pub enum Encoding {
    /// Copy every tensor that is not replaced; replacements are stored as
    /// f16 (the working precision of the compress stages).
    KeepF16,
    /// Re-encode every float 2-D weight (replaced or copied) by the
    /// converter's per-tensor policy for this profile; norms and the
    /// names the converter keeps in float are copied.
    Profile(Quant),
}

pub fn tensor_f32(model: &CmfModel, e: &TensorEntry) -> Result<Vec<f32>> {
    let mut v = vec![0f32; e.n_elems()];
    dequant_tensor(e, model.entry_bytes(e), &mut v)
        .map_err(|err| anyhow!("dequantizing {}: {err}", e.name))?;
    Ok(v)
}

fn is_float(d: TensorDtype) -> bool {
    matches!(d, TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
}

/// One output tensor before it is pushed.
enum Out {
    Copy,
    Encoded(TensorDtype, Vec<usize>, Vec<u8>),
}

/// Write `out` from `src`. `replace(name)` returns the new tensor for names
/// it rewrites; `edit` adjusts the cloned header (arch, provenance).
/// Section hashes are recomputed by the writer; task masks are dropped (a
/// mask over a narrowed FFN would gate the wrong neurons). Returns the
/// written file's size.
pub fn rewrite(
    src: &CmfModel,
    out: &Path,
    enc: Encoding,
    replace: &(dyn Fn(&str) -> Result<Option<Replacement>> + Sync),
    edit: &dyn Fn(&mut CmfHeader),
    threads: usize,
) -> Result<u64> {
    let arch = src.arch().clone();
    let avg = src.tensors.iter().map(|e| e.name.len()).sum::<usize>() / src.tensors.len().max(1);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = out.with_extension("cmf.partial");
    let _ = std::fs::remove_file(&tmp);
    let mut w = CmfStreamWriter::new(&tmp, CmfStreamWriter::head_reserve_for(src.tensors.len(), avg))?;
    let encode_one = |e: &TensorEntry| -> Result<Out> {
        let rep = replace(&e.name)?;
        let reencode = match enc {
            Encoding::KeepF16 => false,
            Encoding::Profile(_) => {
                e.shape.len() == 2 && !keeps_float(&e.name) && (is_float(e.dtype) || rep.is_some())
            }
        };
        match (rep, enc) {
            (None, _) if !reencode => Ok(Out::Copy),
            (Some((shape, vals)), Encoding::KeepF16) => {
                let (dt, bytes) = if shape.len() == 2 {
                    quantize_2d(Quant::F16, &vals, shape[0], shape[1])
                } else {
                    // 1-D replacements (biases, norms) keep the source dtype
                    // when it is float, else f32.
                    match e.dtype {
                        TensorDtype::F16 => quantize_2d(Quant::F16, &vals, 1, vals.len()),
                        _ => (TensorDtype::F32, vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
                    }
                };
                Ok(Out::Encoded(dt, shape, bytes))
            }
            (rep, Encoding::Profile(q)) => {
                let (shape, vals) = match rep {
                    Some(r) => r,
                    None => (e.shape.clone(), tensor_f32(src, e)?),
                };
                if shape.len() != 2 {
                    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    return Ok(Out::Encoded(TensorDtype::F32, shape, bytes));
                }
                let codec = effective_quant(&arch, q, &e.name);
                let (dt, bytes) = quantize_2d(codec, &vals, shape[0], shape[1]);
                Ok(Out::Encoded(dt, shape, bytes))
            }
            (None, Encoding::KeepF16) => Ok(Out::Copy),
        }
    };
    // Encode a batch of tensors in parallel, push them in directory order.
    let threads = threads.max(1);
    for batch in src.tensors.chunks(threads) {
        let outs: Vec<Result<Out>> = std::thread::scope(|s| {
            let hs: Vec<_> = batch.iter().map(|e| s.spawn(|| encode_one(e))).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("encoder panicked")))).collect()
        });
        for (e, o) in batch.iter().zip(outs) {
            match o? {
                Out::Copy => w.push_bounded(&e.name, e.dtype, &e.shape, src.entry_bytes(e), 1 << 28)?,
                Out::Encoded(dt, shape, bytes) => w.push(&e.name, dt, &shape, &bytes)?,
            }
        }
    }
    let mut header = src.header.clone();
    header.section_hashes = None;
    header.segments.clear();
    edit(&mut header);
    w.finish(&header, None, src.vocab.as_deref())?;
    std::fs::rename(&tmp, out)?;
    Ok(std::fs::metadata(out)?.len())
}

/// Copy `src` to `out` with an edited header only (provenance stamp).
pub fn restamp(src: &CmfModel, out: &Path, edit: &dyn Fn(&mut CmfHeader)) -> Result<u64> {
    rewrite(src, out, Encoding::KeepF16, &|_| Ok(None), edit, 1)
}

/// Merge `entry` into `header.provenance[key]`.
pub fn provenance_set(header: &mut CmfHeader, key: &str, entry: serde_json::Value) {
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    if !prov.is_object() {
        prov = serde_json::json!({ "previous": prov });
    }
    prov[key] = entry;
    header.provenance = Some(prov);
}

/// The rows `keep` of a binary low-rank tensor (dtype 19): each path's
/// output scales `h` and the rows of its `U` bits; `g`, `ℓ`, `V` stay.
/// Layout: 16-byte head (`np` at byte 2, `r`, `m`, `n` as u32 at 4, 8, 12),
/// then per path `h[m]`, `g[n]`, `ℓ[r]` (f16), `U` (m rows of r bits),
/// `V` (r rows of n bits).
pub fn blr_rows(bytes: &[u8], keep: &[usize]) -> Result<Vec<u8>> {
    blr_subset(bytes, Some(keep), None)
}

/// The columns `keep` of a binary low-rank tensor: each path's input scales
/// `g` and the matching bit of every `V` row.
pub fn blr_cols(bytes: &[u8], keep: &[usize]) -> Result<Vec<u8>> {
    blr_subset(bytes, None, Some(keep))
}

fn blr_subset(bytes: &[u8], rows: Option<&[usize]>, cols: Option<&[usize]>) -> Result<Vec<u8>> {
    anyhow::ensure!(bytes.len() >= 16, "blr: {} bytes", bytes.len());
    let u = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let (np, r, m, n) = (bytes[2] as usize, u(4), u(8), u(12));
    anyhow::ensure!(r % 8 == 0 && n % 8 == 0, "blr: rank {r} / columns {n} not byte-aligned");
    let per = 2 * m + 2 * n + 2 * r + m * r / 8 + r * n / 8;
    anyhow::ensure!(bytes.len() == 16 + np * per, "blr: {} bytes, expected {} ({np} paths, r {r}, {m}×{n})", bytes.len(), 16 + np * per);
    let rk: Vec<usize> = rows.map_or_else(|| (0..m).collect(), <[usize]>::to_vec);
    let ck: Vec<usize> = cols.map_or_else(|| (0..n).collect(), <[usize]>::to_vec);
    anyhow::ensure!(rk.iter().all(|&i| i < m) && ck.iter().all(|&j| j < n), "blr: kept index out of range");
    let (m2, n2) = (rk.len(), ck.len());
    anyhow::ensure!(n2 % 8 == 0, "blr: {n2} kept columns are not byte-aligned");
    let mut out = bytes[..16].to_vec();
    out[8..12].copy_from_slice(&(m2 as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(n2 as u32).to_le_bytes());
    let mut off = 16;
    for _ in 0..np {
        let (h, g, l) = (&bytes[off..off + 2 * m], &bytes[off + 2 * m..off + 2 * m + 2 * n], &bytes[off + 2 * m + 2 * n..off + 2 * m + 2 * n + 2 * r]);
        let ub = &bytes[off + 2 * (m + n + r)..off + 2 * (m + n + r) + m * r / 8];
        let vb = &bytes[off + 2 * (m + n + r) + m * r / 8..off + per];
        for &i in &rk {
            out.extend_from_slice(&h[2 * i..2 * i + 2]);
        }
        for &j in &ck {
            out.extend_from_slice(&g[2 * j..2 * j + 2]);
        }
        out.extend_from_slice(l);
        for &i in &rk {
            out.extend_from_slice(&ub[i * r / 8..(i + 1) * r / 8]);
        }
        for k in 0..r {
            let row = &vb[k * n / 8..(k + 1) * n / 8];
            let mut packed = vec![0u8; n2 / 8];
            for (q, &j) in ck.iter().enumerate() {
                if (row[j / 8] >> (j % 8)) & 1 == 1 {
                    packed[q / 8] |= 1 << (q % 8);
                }
            }
            out.extend_from_slice(&packed);
        }
        off += per;
    }
    Ok(out)
}

/// Drop FFN neurons from every layer: `keep[l]` (ascending) are the
/// intermediate neurons layer `l` keeps — rows of gate/up (and their
/// biases), columns of down. Binary low-rank tensors are cut in place (no
/// re-encoding); others are decoded, cut and re-encoded in their own
/// codec. The header's `intermediate_size` becomes the widest layer.
pub fn subset_ffn(src: &CmfModel, out: &Path, keep: &[Vec<usize>], edit: &dyn Fn(&mut CmfHeader)) -> Result<u64> {
    use crate::convert::encode_for;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = out.with_extension("cmf.partial");
    let _ = std::fs::remove_file(&tmp);
    let avg = src.tensors.iter().map(|e| e.name.len()).sum::<usize>() / src.tensors.len().max(1);
    let mut w = CmfStreamWriter::new(&tmp, CmfStreamWriter::head_reserve_for(src.tensors.len(), avg))?;
    fn layer_of(name: &str) -> Option<(usize, &str)> {
        let t = name.strip_prefix("model.layers.")?;
        let (l, rest) = t.split_once('.')?;
        Some((l.parse().ok()?, rest))
    }
    let reencode = |e: &TensorEntry, vals: &[f32], shape: &[usize]| -> (TensorDtype, Vec<u8>) {
        match e.dtype {
            TensorDtype::F32 => (TensorDtype::F32, vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            TensorDtype::Q4Tiled | TensorDtype::Q4TiledP | TensorDtype::Q4Block | TensorDtype::Q8Row | TensorDtype::Q8_2f | TensorDtype::F16 => {
                (e.dtype, encode_for(e.dtype, vals, shape[0], shape.get(1).copied().unwrap_or(1)))
            }
            _ => (TensorDtype::F16, encode_for(TensorDtype::F16, vals, shape[0], shape.get(1).copied().unwrap_or(1))),
        }
    };
    for e in &src.tensors {
        let cut = layer_of(&e.name).and_then(|(l, rest)| {
            let k = keep.get(l)?;
            match rest {
                "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.gate_proj.bias" | "mlp.up_proj.bias" => Some((k, true)),
                "mlp.down_proj.weight" => Some((k, false)),
                _ => None,
            }
        });
        let Some((k, rows)) = cut else {
            w.push_bounded(&e.name, e.dtype, &e.shape, src.entry_bytes(e), 1 << 28)?;
            continue;
        };
        if e.dtype == TensorDtype::BinaryLowRank {
            let b = if rows { blr_rows(src.entry_bytes(e), k)? } else { blr_cols(src.entry_bytes(e), k)? };
            let shape = if rows { vec![k.len(), e.shape[1]] } else { vec![e.shape[0], k.len()] };
            w.push(&e.name, e.dtype, &shape, &b)?;
            continue;
        }
        let v = tensor_f32(src, e)?;
        let (shape, vals): (Vec<usize>, Vec<f32>) = if e.shape.len() == 1 {
            (vec![k.len()], k.iter().map(|&i| v[i]).collect())
        } else if rows {
            let cols = e.shape[1];
            (vec![k.len(), cols], k.iter().flat_map(|&i| v[i * cols..(i + 1) * cols].iter().copied()).collect())
        } else {
            let (m, n) = (e.shape[0], e.shape[1]);
            (vec![m, k.len()], (0..m).flat_map(|i| k.iter().map(move |&j| (i, j))).map(|(i, j)| v[i * n + j]).collect())
        };
        let (dt, bytes) = reencode(e, &vals, &shape);
        w.push(&e.name, dt, &shape, &bytes)?;
    }
    let mut header = src.header.clone();
    header.section_hashes = None;
    header.segments.clear();
    header.arch.intermediate_size = keep.iter().map(Vec::len).max().unwrap_or(header.arch.intermediate_size);
    edit(&mut header);
    w.finish(&header, None, src.vocab.as_deref())?;
    std::fs::rename(&tmp, out)?;
    Ok(std::fs::metadata(out)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random binary low-rank blob with `np` paths.
    fn blob(np: usize, r: usize, m: usize, n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut b = vec![1u8, 0, np as u8, 0];
        for v in [r, m, n] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for _ in 0..np {
            for _ in 0..(m + n + r) {
                let x = cortiq_core::quant::f32_to_f16(0.5 + (next() % 1000) as f32 / 1000.0);
                b.extend_from_slice(&x.to_le_bytes());
            }
            for _ in 0..(m * r / 8 + r * n / 8) {
                b.push(next() as u8);
            }
        }
        b
    }

    /// The dense matrix of a blob: Σ_p diag(h)·U·diag(ℓ)·V·diag(g), bits ±1.
    fn dense(b: &[u8]) -> (usize, usize, Vec<f64>) {
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (np, r, m, n) = (b[2] as usize, u(4), u(8), u(12));
        let f = |o: usize| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]])) as f64;
        let mut w = vec![0f64; m * n];
        let mut off = 16;
        for _ in 0..np {
            let (ho, go, lo) = (off, off + 2 * m, off + 2 * m + 2 * n);
            let uo = lo + 2 * r;
            let vo = uo + m * r / 8;
            let bit = |base: usize, row: usize, width: usize, k: usize| if (b[base + row * width / 8 + k / 8] >> (k % 8)) & 1 == 1 { 1.0 } else { -1.0 };
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..r {
                        acc += bit(uo, i, r, k) * f(lo + 2 * k) * bit(vo, k, n, j);
                    }
                    w[i * n + j] += f(ho + 2 * i) * acc * f(go + 2 * j);
                }
            }
            off = vo + r * n / 8;
        }
        (m, n, w)
    }

    #[test]
    fn blr_cuts_match_the_dense_cuts() {
        let b = blob(2, 16, 24, 32, 7);
        let (m, n, w) = dense(&b);
        let rows = [1usize, 4, 5, 9, 23];
        let (m2, n2, wr) = dense(&blr_rows(&b, &rows).unwrap());
        assert_eq!((m2, n2), (rows.len(), n));
        for (q, &i) in rows.iter().enumerate() {
            for j in 0..n {
                assert_eq!(wr[q * n + j], w[i * n + j]);
            }
        }
        let cols: Vec<usize> = (0..n).filter(|j| j % 4 != 1).collect();
        let (m3, n3, wc) = dense(&blr_cols(&b, &cols).unwrap());
        assert_eq!((m3, n3), (m, cols.len()));
        for i in 0..m {
            for (q, &j) in cols.iter().enumerate() {
                assert_eq!(wc[i * n3 + q], w[i * n + j]);
            }
        }
        assert!(blr_cols(&b, &[0, 1, 2]).is_err(), "unaligned column count");
    }
}
