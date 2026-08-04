//! In-place recoding of an existing `.cmf` between quantization layouts.
//!
//! The motivating case is `q4_tiled → q4tp`: the nibbles are unchanged in
//! meaning, only the per-tile scale is re-expressed as a rung on a per-row
//! ladder, so a published model can shed ~7% of its bytes without anyone
//! re-downloading the original checkpoint. That matters — the checkpoints
//! behind the published CMF files are tens of gigabytes, and re-converting
//! them is the expensive path this command exists to avoid.
//!
//! The second case is `q2tp-draft`: the MTP draft stack's expert gate/up
//! planes to q2tp — the layout a fresh conversion produces since the
//! converter's q2tp profile covered them. These weights only ever draft
//! (a trunk pass verifies every token), so the grid change is a fidelity
//! trade confined to acceptance rate, never to output. The trunk's own
//! experts are deliberately NOT eligible: recoding them 4→2 bit without
//! the original checkpoint would degrade the model itself.
//!
//! Everything the recoder cannot improve is copied verbatim, so the output
//! is byte-identical outside the tensors it deliberately rewrites. With
//! `--in-place` the rewrite happens inside the source file itself (new
//! payload ≤ old slot; the slack goes dark) — for machines whose disk
//! cannot hold the model twice.

use anyhow::{Context, bail};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_core::quant::{GROUP_SIZE, dequant_tensor};
use cortiq_core::types::TensorDtype;
use std::sync::Arc;

use crate::convert::{Quant, encode_q2tp, encode_q4tp, parse_quant, q2tp_expert_gate_or_up};

enum Mode {
    /// Whole-file scale re-expression, lossless in the 4-bit grid.
    FullQ4tp,
    /// Draft-only expert inputs to the 2-bit grid.
    DraftQ2tp,
}

/// Which dtypes this mode recodes, and into what.
fn target_dtype(mode: &Mode, name: &str, src: TensorDtype, shape: &[usize]) -> Option<TensorDtype> {
    let two_d = shape.len() == 2 && shape[1] % GROUP_SIZE == 0;
    match mode {
        // q4tp reads the same 4-bit grid as q4_tiled/q4_block, so recoding
        // either is lossless in the grid and only re-expresses the scale.
        Mode::FullQ4tp if two_d && matches!(src, TensorDtype::Q4Tiled | TensorDtype::Q4Block) => {
            Some(TensorDtype::Q4TiledP)
        }
        // The draft's SwiGLU inputs, and only the draft's: `.mtp.` scopes
        // the match to the speculative stack.
        Mode::DraftQ2tp
            if two_d
                && src == TensorDtype::Q4TiledP
                && name.contains(".mtp.")
                && q2tp_expert_gate_or_up(name) =>
        {
            Some(TensorDtype::Q2TiledP)
        }
        _ => None,
    }
}

pub fn cmd_requant(
    model_path: &str,
    output: Option<&str>,
    quant: &str,
    in_place: bool,
) -> anyhow::Result<()> {
    let mode = if quant == "q2tp-draft" {
        Mode::DraftQ2tp
    } else if matches!(parse_quant(quant)?, Quant::Q4TiledP) {
        Mode::FullQ4tp
    } else {
        bail!(
            "requant targets q4tp or q2tp-draft (got '{quant}'); other layouts \
             change the value grid, which needs the original checkpoint"
        );
    };
    if !in_place && output.is_none() {
        bail!("either --output or --in-place is required");
    }

    let model = Arc::new(CmfModel::open_sharded(model_path)?);

    let recode = |entry: &cortiq_core::format::TensorEntry,
                  dt: TensorDtype|
     -> anyhow::Result<Vec<u8>> {
        let (rows, cols) = (entry.shape[0], entry.shape[1]);
        let mut buf = vec![0.0f32; rows * cols];
        dequant_tensor(entry, model.entry_bytes(entry), &mut buf)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("dequantizing '{}'", entry.name))?;
        Ok(match dt {
            TensorDtype::Q2TiledP => encode_q2tp(&buf, rows, cols),
            _ => encode_q4tp(&buf, rows, cols),
        })
    };

    if in_place {
        // The new payload rides into the old slot; the directory entry keeps
        // its offset. Only shard-0 tensors are reachable this way.
        let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::new();
        let (mut before, mut after) = (0u64, 0u64);
        for (i, entry) in model.tensors.iter().enumerate() {
            let Some(dt) = target_dtype(&mode, &entry.name, entry.dtype, &entry.shape) else {
                continue;
            };
            if entry.shard != 0 {
                bail!("'{}' lives in a sibling shard; in-place needs shard 0", entry.name);
            }
            let data = recode(entry, dt)?;
            if data.len() as u64 > entry.nbytes {
                bail!(
                    "'{}' would grow {} → {} bytes; in-place cannot move tensors — \
                     use --output",
                    entry.name,
                    entry.nbytes,
                    data.len()
                );
            }
            before += entry.nbytes;
            after += data.len() as u64;
            patches.push((i, dt, data));
        }
        let n = patches.len();
        drop(model); // release the mmap before rewriting under it
        CmfModel::recode_entries_in_place(model_path, &patches)?;
        println!(
            "requant → {quant} (in place): {n} tensors recoded, \
             {:.2} GB → {:.2} GB payload (file length unchanged; slack goes dark)",
            before as f64 / 1e9,
            after as f64 / 1e9,
        );
        return Ok(());
    }

    let output = output.unwrap();
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
    let (mut recoded, mut copied, mut before, mut after) = (0usize, 0usize, 0u64, 0u64);

    for entry in &model.tensors {
        let src = model.entry_bytes(entry);
        before += src.len() as u64;

        let (dtype, data) = match target_dtype(&mode, &entry.name, entry.dtype, &entry.shape) {
            Some(dt) => {
                recoded += 1;
                (dt, recode(entry, dt)?)
            }
            None => {
                copied += 1;
                (entry.dtype, src.to_vec())
            }
        };
        after += data.len() as u64;
        specs.push(TensorSpec {
            name: entry.name.clone(),
            dtype,
            shape: entry.shape.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end through the real command: a mini CMF with a draft expert,
    /// a trunk expert and a draft down-projection — only the draft's w1
    /// changes layout, in place, and the file stays verifiable.
    #[test]
    fn in_place_q2tp_draft_recodes_the_file_it_was_pointed_at() {
        let dir = std::env::temp_dir().join(format!("cmf-rq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.cmf");
        let path_s = path.to_str().unwrap();

        let (rows, cols) = (8usize, 256usize);
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.377).sin()) * 0.5)
            .collect();
        let q4 = encode_q4tp(&vals, rows, cols);
        let spec = |name: &str| TensorSpec {
            name: name.into(),
            dtype: TensorDtype::Q4TiledP,
            shape: vec![rows, cols],
            data: q4.clone(),
        };
        let tensors = vec![
            spec("model.mtp.0.ffn.experts.0.w1.weight"),
            spec("model.mtp.0.ffn.experts.0.w2.weight"),
            spec("model.layers.0.mlp.experts.0.gate_proj.weight"),
        ];
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf",
            "version": cortiq_core::CMF_VERSION,
            "arch": {
                "arch_name": "tiny-test",
                "hidden_size": cols,
                "intermediate_size": rows,
                "num_layers": 1,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "layer_types": ["FullAttention"],
                "rms_norm_eps": 1e-6,
                "norm_style": "qwen",
                "hidden_act": "silu",
                "max_position_embeddings": 64
            },
            "quant_type": "Q4_BLOCK"
        }))
        .expect("header json");
        CmfModel::write(path_s, &header, &tensors, None, None).unwrap();

        cmd_requant(path_s, None, "q2tp-draft", true).unwrap();

        let m = CmfModel::open(path_s).unwrap();
        assert!(m.verify().is_empty(), "verify(): {:?}", m.verify());
        let by = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap();
        assert_eq!(by("model.mtp.0.ffn.experts.0.w1.weight").dtype, TensorDtype::Q2TiledP);
        assert_eq!(by("model.mtp.0.ffn.experts.0.w2.weight").dtype, TensorDtype::Q4TiledP);
        let trunk = by("model.layers.0.mlp.experts.0.gate_proj.weight");
        assert_eq!(trunk.dtype, TensorDtype::Q4TiledP);
        assert_eq!(m.entry_bytes(trunk), &q4[..], "trunk bytes changed");

        // The recoded plane must dequant close to the source values — the
        // exact bytes the pure-conversion path would produce.
        let e = by("model.mtp.0.ffn.experts.0.w1.weight");
        let mut back = vec![0.0f32; rows * cols];
        dequant_tensor(e, m.entry_bytes(e), &mut back).unwrap();
        let q2_direct = encode_q2tp(&vals_dequant(&q4, rows, cols), rows, cols);
        assert_eq!(m.entry_bytes(e), &q2_direct[..], "not the conversion-path bytes");
    }

    fn vals_dequant(q4: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let entry = cortiq_core::format::TensorEntry {
            name: "x".into(),
            dtype: TensorDtype::Q4TiledP,
            shape: vec![rows, cols],
            off: 0,
            nbytes: q4.len() as u64,
            hash: 0,
            shard: 0,
        };
        let mut out = vec![0.0f32; rows * cols];
        dequant_tensor(&entry, q4, &mut out).unwrap();
        out
    }

    /// The 2-bit recode must reach ONLY the draft stack's SwiGLU inputs:
    /// trunk experts came from the original checkpoint and a 4→2 bit hop
    /// through dequant would degrade the model itself, not just the draft.
    #[test]
    fn q2tp_draft_recode_is_scoped_to_the_mtp_stack() {
        let t = |name: &str| {
            target_dtype(&Mode::DraftQ2tp, name, TensorDtype::Q4TiledP, &[64, 256])
        };
        assert_eq!(t("model.mtp.0.ffn.experts.42.w1.weight"), Some(TensorDtype::Q2TiledP));
        assert_eq!(t("model.mtp.2.ffn.experts.7.w3.weight"), Some(TensorDtype::Q2TiledP));
        assert_eq!(t("model.mtp.1.ffn.shared_experts.w1.weight"), Some(TensorDtype::Q2TiledP));
        // the draft's own down/skeleton stay 4-bit
        assert_eq!(t("model.mtp.0.ffn.experts.42.w2.weight"), None);
        assert_eq!(t("model.mtp.0.self_attn.wkv.weight"), None);
        // the trunk is untouchable, expert or not
        assert_eq!(t("model.layers.7.mlp.experts.42.gate_proj.weight"), None);
        assert_eq!(t("model.layers.7.mlp.experts.42.up_proj.weight"), None);
        // already 2-bit, or the wrong rank — nothing to do
        assert_eq!(
            target_dtype(&Mode::DraftQ2tp, "model.mtp.0.ffn.experts.1.w1.weight",
                TensorDtype::Q2TiledP, &[64, 256]),
            None
        );
        assert_eq!(
            target_dtype(&Mode::DraftQ2tp, "model.mtp.0.ffn.experts.1.w1.weight",
                TensorDtype::Q4TiledP, &[64]),
            None
        );
    }
}
