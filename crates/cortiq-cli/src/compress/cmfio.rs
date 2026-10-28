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
