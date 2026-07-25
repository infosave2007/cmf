//! Pack a Lumina-Image 2.0 diffusers directory into ONE quantized
//! .cmf: `te.*` (Gemma-2 prompt encoder), `dit.*` (Next-DiT denoiser),
//! `vae.*` (FLUX-VAE decoder), the tokenizer.json in the VOCAB
//! section, and each component's config as a `*.config_json` U8
//! tensor. Big projections carry the chosen codec (q4t/q8), AdaLN
//! modulation and the embedding table always q8 (a single-vector
//! input and per-token lookups — cheap to keep at 8 bit), VAE conv
//! kernels f16, every norm/bias exact f32. The imagegen runtime
//! (`cortiq imagine file.cmf`) runs the quantized projections
//! straight off the mmap through the engine's batched dot kernels.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::{Path, PathBuf};

use crate::convert::{encode_f16, encode_q4_tiled, encode_q8_row};

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Q8,
    Q4t,
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode one 2-D projection at `level` (falls back q4t→q8→f32 when a
/// codec's column constraint isn't met).
fn encode_2d(name: String, vals: &[f32], rows: usize, cols: usize, level: Level) -> TensorSpec {
    let (dtype, data) = if level == Level::Q4t && cols % 32 == 0 {
        (TensorDtype::Q4Tiled, encode_q4_tiled(vals, rows, cols))
    } else {
        (TensorDtype::Q8Row, encode_q8_row(vals, rows, cols))
    };
    TensorSpec {
        name,
        dtype,
        shape: vec![rows, cols],
        data,
    }
}

fn spec_f32(name: String, shape: Vec<usize>, vals: &[f32]) -> TensorSpec {
    TensorSpec {
        name,
        dtype: TensorDtype::F32,
        shape,
        data: f32_bytes(vals),
    }
}

fn config_spec(prefix: &str, dir: &Path) -> anyhow::Result<TensorSpec> {
    let raw = std::fs::read(dir.join("config.json"))
        .with_context(|| format!("{}/config.json", dir.display()))?;
    Ok(TensorSpec {
        name: format!("{prefix}.config_json"),
        dtype: TensorDtype::U8,
        shape: vec![raw.len()],
        data: raw,
    })
}

/// Every shard of a sharded (or single-file) safetensors component.
fn shard_paths(dir: &Path, index_name: &str, single_name: &str) -> anyhow::Result<Vec<PathBuf>> {
    let idx_path = dir.join(index_name);
    if idx_path.exists() {
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path)?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("weight_map"))?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        Ok(shards.into_iter().map(|s| dir.join(s)).collect())
    } else {
        Ok(vec![dir.join(single_name)])
    }
}

/// Codec decision for one tensor: output name + Some(level) to
/// quantize a 2-D projection, None to store exact f32.
type Pick<'a> = &'a dyn Fn(&str, &[usize]) -> Option<(String, Option<Level>)>;

/// Stream a component's tensors into specs. `pick` maps a source name
/// to Some(codec decision); None drops the tensor.
fn pack_component(
    dir: &Path,
    index_name: &str,
    single_name: &str,
    specs: &mut Vec<TensorSpec>,
    pick: Pick<'_>,
) -> anyhow::Result<()> {
    for shard in shard_paths(dir, index_name, single_name)? {
        cortiq_engine::vae::read_safetensors_each(&shard, &mut |name, shape, data| {
            if let Some((out_name, level)) = pick(name, &shape) {
                specs.push(match level {
                    Some(l) if shape.len() == 2 => {
                        encode_2d(out_name, &data, shape[0], shape[1], l)
                    }
                    _ => spec_f32(out_name, shape, &data),
                });
            }
            Ok(())
        })
        .map_err(|e| anyhow!("{}: {e}", shard.display()))?;
    }
    Ok(())
}

pub fn cmd_imagine_pack(root: &str, quant: &str, out: &str) -> anyhow::Result<()> {
    let root = Path::new(root);
    let level = match quant {
        "q8" | "q8_row" => Level::Q8,
        "q4t" | "q4_tiled" | "q4" => Level::Q4t,
        other => return Err(anyhow!("--quant {other}: expected q4t or q8")),
    };
    let t0 = std::time::Instant::now();
    let mut specs: Vec<TensorSpec> = Vec::new();

    // ── Gemma-2 text encoder → te.* ──
    let te_dir = root.join("text_encoder");
    specs.push(config_spec("te", &te_dir)?);
    pack_component(
        &te_dir,
        "model.safetensors.index.json",
        "model.safetensors",
        &mut specs,
        &|name, _shape| {
            // The Lumina export drops the "model." prefix; normalize.
            let n = name.strip_prefix("model.").unwrap_or(name);
            if n == "lm_head.weight" {
                return None; // tied — the encoder never projects logits
            }
            let big = n.ends_with("_proj.weight");
            let lvl = if n == "embed_tokens.weight" {
                Some(Level::Q8)
            } else if big {
                Some(level)
            } else {
                None
            };
            Some((format!("te.{n}"), lvl))
        },
    )?;
    eprintln!("te packed ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Next-DiT → dit.* ──
    let dit_dir = root.join("transformer");
    specs.push(config_spec("dit", &dit_dir)?);
    pack_component(
        &dit_dir,
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        &mut specs,
        &|name, shape| {
            let lvl = if name.ends_with("norm1.linear.weight") {
                Some(Level::Q8) // AdaLN modulation: keep 8-bit
            } else if name.ends_with("caption_embedder.1.weight")
                || (shape.len() == 2
                    && shape[1] >= 1024
                    && (name.contains(".attn.to_") || name.contains(".feed_forward.")))
            {
                Some(level)
            } else {
                None
            };
            Some((format!("dit.{name}"), lvl))
        },
    )?;
    eprintln!("dit packed ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── FLUX-VAE decoder → vae.* (f16 conv kernels, f32 the rest) ──
    let vae_dir = root.join("vae");
    specs.push(config_spec("vae", &vae_dir)?);
    pack_component(
        &vae_dir,
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        &mut specs,
        &|name, _shape| {
            if !name.starts_with("decoder.") {
                return None; // the encoder half is not needed for t2i
            }
            Some((format!("vae.{name}"), None))
        },
    )?;
    // f16 for the 4-D conv kernels, after the fact (pack_component's
    // Level path is 2-D only).
    for s in specs.iter_mut() {
        if s.name.starts_with("vae.") && s.shape.len() == 4 && s.dtype == TensorDtype::F32 {
            let vals: Vec<f32> = s
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            s.data = encode_f16(&vals);
            s.dtype = TensorDtype::F16;
        }
    }
    eprintln!("vae packed ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── header: honest Gemma numbers, a distinct arch tag ──
    let te_cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(te_dir.join("config.json"))?)?;
    let nl = te_cfg["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "lumina2-image",
        "hidden_size": te_cfg["hidden_size"],
        "intermediate_size": te_cfg["intermediate_size"],
        "num_layers": nl,
        "num_attention_heads": te_cfg["num_attention_heads"],
        "num_kv_heads": te_cfg["num_key_value_heads"],
        "head_dim": te_cfg["head_dim"],
        "vocab_size": te_cfg["vocab_size"],
        "layer_types": vec!["FullAttention"; nl],
        "rms_norm_eps": te_cfg["rms_norm_eps"],
        "max_position_embeddings": te_cfg["max_position_embeddings"],
        // linear-attention fields are inert for this arch
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match level {
            Level::Q8 => QuantType::Q8Row,
            Level::Q4t => QuantType::Q4Block, // file-level label; per-tensor truth is the directory
        },
        provenance: Some(serde_json::json!({
            "pipeline": "lumina-image-2.0",
            "components": {"te": "gemma-2 encoder", "dit": "next-dit", "vae": "flux-vae decoder"},
            "packed_from": root.display().to_string(),
            "quant": quant,
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };
    let vocab = std::fs::read(root.join("tokenizer").join("tokenizer.json"))
        .with_context(|| "tokenizer/tokenizer.json")?;
    CmfModel::write(out, &header, &specs, None, Some(&vocab))
        .map_err(|e| anyhow!("write {out}: {e}"))?;
    let size = std::fs::metadata(out)?.len();
    println!(
        "{out}: {} tensors, {:.2} GB in {:.1}s",
        specs.len(),
        size as f64 / (1 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
