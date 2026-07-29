//! In-place recoding of an existing `.cmf` between quantization layouts.
//!
//! The motivating case is `q4_tiled → q4tp`: the nibbles are unchanged in
//! meaning, only the per-tile scale is re-expressed as a rung on a per-row
//! ladder, so a published model can shed ~7% of its bytes without anyone
//! re-downloading the original checkpoint. That matters — the checkpoints
//! behind the published CMF files are tens of gigabytes, and re-converting
//! them is the expensive path this command exists to avoid.
//!
//! Everything the recoder cannot improve is copied verbatim, so the output
//! is byte-identical outside the tensors it deliberately rewrites.

use anyhow::{Context, bail};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_core::quant::{GROUP_SIZE, dequant_tensor};
use cortiq_core::types::TensorDtype;
use std::sync::Arc;

use crate::convert::{Quant, encode_q4tp, parse_quant};

/// Which dtypes a target quant can profitably recode, and into what.
fn target_dtype(quant: Quant, src: TensorDtype, shape: &[usize]) -> Option<TensorDtype> {
    let two_d = shape.len() == 2 && shape[1] % GROUP_SIZE == 0;
    match quant {
        // q4tp reads the same 4-bit grid as q4_tiled/q4_block, so recoding
        // either is lossless in the grid and only re-expresses the scale.
        Quant::Q4TiledP if two_d && matches!(src, TensorDtype::Q4Tiled | TensorDtype::Q4Block) => {
            Some(TensorDtype::Q4TiledP)
        }
        _ => None,
    }
}

pub fn cmd_requant(model_path: &str, output: &str, quant: &str) -> anyhow::Result<()> {
    let q = parse_quant(quant)?;
    if !matches!(q, Quant::Q4TiledP) {
        bail!(
            "requant currently targets only q4tp (got '{quant}'); other layouts \
             change the value grid, which needs the original checkpoint"
        );
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);

    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
    let (mut recoded, mut copied, mut before, mut after) = (0usize, 0usize, 0u64, 0u64);
    let mut buf: Vec<f32> = Vec::new();

    for entry in &model.tensors {
        let shape: Vec<usize> = entry.shape.clone();
        let src = model.entry_bytes(entry);
        before += src.len() as u64;

        let data = match target_dtype(q, entry.dtype, &shape) {
            Some(TensorDtype::Q4TiledP) => {
                let (rows, cols) = (shape[0], shape[1]);
                buf.clear();
                buf.resize(rows * cols, 0.0);
                dequant_tensor(entry, src, &mut buf)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("dequantizing '{}'", entry.name))?;
                recoded += 1;
                encode_q4tp(&buf, rows, cols)
            }
            _ => {
                copied += 1;
                src.to_vec()
            }
        };
        after += data.len() as u64;
        specs.push(TensorSpec {
            name: entry.name.clone(),
            dtype: target_dtype(q, entry.dtype, &shape).unwrap_or(entry.dtype),
            shape,
            data,
        });
    }

    CmfModel::write(
        output,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;

    let (a, b) = (before as f64, after as f64);
    println!(
        "requant → {quant}: {recoded} tensors recoded, {copied} copied verbatim\n\
         payload {:.2} GB → {:.2} GB ({:+.1}%)",
        a / 1e9,
        b / 1e9,
        (b / a - 1.0) * 100.0
    );
    let (in_sz, out_sz) = (
        std::fs::metadata(model_path)?.len() as f64 / 1e9,
        std::fs::metadata(output)?.len() as f64 / 1e9,
    );
    println!(
        "file {in_sz:.2} GB → {out_sz:.2} GB ({:+.1}%)",
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}
