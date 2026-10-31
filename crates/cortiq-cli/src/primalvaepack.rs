//! Pack a Primal VAE checkpoint (`ostris/primal_vae`, `primal_vae_2x1` /
//! `primal_vae_2x2`) into one CMF that `cortiq vae` runs.
//!
//! The source is an ai-toolkit training checkpoint, fp32: `encoder.*` and
//! `decoder.*` with the module names of `modeling_primal_vae.py`, plus an
//! `arch_spec` buffer the reference ignores (so do we). The architecture is
//! read back from the tensor shapes at load time, exactly as the reference
//! `config_from_shapes` does, so the names are kept unchanged.
//!
//! Precision: `f16` (default) stores the 3×3 kernels in half precision —
//! the model card puts bf16 inference at ~0.05 dB — with the biases and
//! norm weights exact; `f32` keeps the checkpoint bit for bit.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::{Path, PathBuf};

/// Whether `model` (a directory or a weight file in one) is a Primal VAE
/// checkpoint: its `config.json` names the `PrimalVAE` class.
pub fn is_primal_vae_source(model: &str) -> bool {
    let p = Path::new(model);
    let dir = if p.is_file() { p.parent().unwrap_or(Path::new(".")) } else { p };
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|c| c.get("_class_name").and_then(|v| v.as_str()).map(|s| s == "PrimalVAE"))
        .unwrap_or(false)
}

fn weight_file(model: &str) -> anyhow::Result<PathBuf> {
    let p = Path::new(model);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(p)
        .with_context(|| model.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(anyhow!("{model}: no .safetensors weight file")),
        _ => Err(anyhow!(
            "{model} holds {} weight files ({}); pass one as --model DIR/<file>.safetensors",
            found.len(),
            found.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Every F32 tensor of a safetensors file, in header order. The
/// checkpoint's `arch_spec` (I32, ai-toolkit's own buffer) is skipped; any
/// other dtype is an error.
fn read_f32_tensors(
    path: &Path,
    f: &mut dyn FnMut(&str, Vec<usize>, Vec<f32>) -> Result<(), String>,
) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 8 {
        return Err("truncated safetensors header".into());
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..8 + hlen).ok_or("truncated safetensors header")?).map_err(|e| e.to_string())?;
    let data = &bytes[8 + hlen..];
    let mut entries: Vec<(&String, &serde_json::Value)> =
        header.as_object().ok_or("safetensors header is not an object")?.iter().filter(|(k, _)| *k != "__metadata__").collect();
    entries.sort_by_key(|(_, v)| v["data_offsets"][0].as_u64().unwrap_or(0));
    for (name, v) in entries {
        let dtype = v["dtype"].as_str().unwrap_or("");
        if name == "arch_spec" {
            continue;
        }
        if dtype != "F32" {
            return Err(format!("tensor '{name}' is {dtype}, expected F32"));
        }
        let shape: Vec<usize> = v["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
        let (a, b) = (v["data_offsets"][0].as_u64().unwrap_or(0) as usize, v["data_offsets"][1].as_u64().unwrap_or(0) as usize);
        let raw = data.get(a..b).ok_or_else(|| format!("tensor '{name}' lies outside the file"))?;
        if raw.len() != shape.iter().product::<usize>() * 4 {
            return Err(format!("tensor '{name}' size disagrees with its shape"));
        }
        let vals: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        f(name, shape, vals)?;
    }
    Ok(())
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn pack(model: &str, quant: Option<&str>, out: &str) -> anyhow::Result<()> {
    let half = match quant.unwrap_or("f16") {
        "f16" | "auto" => true,
        "f32" => false,
        other => return Err(anyhow!("--quant {other}: a Primal VAE packs as f16 (default) or f32")),
    };
    let file = weight_file(model)?;
    let t0 = std::time::Instant::now();
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut params = 0usize;
    read_f32_tensors(&file, &mut |name, shape, data| {
        if !(name.starts_with("encoder.") || name.starts_with("decoder.")) {
            return Err(format!("unexpected tensor '{name}' (a Primal VAE has encoder.* and decoder.*)"));
        }
        params += data.len();
        let (dtype, bytes) = if half && shape.len() == 4 {
            (TensorDtype::F16, crate::convert::encode_f16(&data))
        } else {
            (TensorDtype::F32, f32_bytes(&data))
        };
        specs.push(TensorSpec { name: name.to_string(), dtype, shape, data: bytes });
        Ok(())
    })
    .map_err(|e| anyhow!("{}: {e}", file.display()))?;
    anyhow::ensure!(
        specs.iter().any(|s| s.name == "encoder.conv_in.weight") && specs.iter().any(|s| s.name == "decoder.conv_out.weight"),
        "{}: not a Primal VAE checkpoint (no encoder.conv_in / decoder.conv_out)",
        file.display()
    );
    let shape_of = |n: &str| specs.iter().find(|s| s.name == n).map(|s| s.shape.clone()).unwrap_or_default();
    let cin = shape_of("encoder.conv_in.weight");
    let latent = shape_of("encoder.conv_out.weight");
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "primal_vae",
        "hidden_size": cin.first().copied().unwrap_or(0),
        "intermediate_size": 0,
        "num_layers": 0,
        "num_attention_heads": 0,
        "num_kv_heads": 0,
        "head_dim": 0,
        "vocab_size": 0,
        "layer_types": [],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 0,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    let source_name = file.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: if half { QuantType::F16 } else { QuantType::F32 },
        provenance: Some(serde_json::json!({
            "tool": "cortiq convert",
            "pipeline": "primal_vae",
            "source_model": model,
            "source_file": source_name,
            "compression": ((cin.get(1).copied().unwrap_or(3) / 3) as f64).sqrt().round() as usize,
            "latent_channels": latent.first().copied().unwrap_or(0),
            "weight_quant": if half { "f16" } else { "f32" },
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
        genome: None,
        lineage: Vec::new(),
        router: None,
        segments: Vec::new(),
    };
    CmfModel::write(out, &header, &specs, None, None).map_err(|e| anyhow!("write {out}: {e}"))?;
    let size = std::fs::metadata(out)?.len();
    println!(
        "{out}: Primal VAE ({source_name}), {} tensors, {:.1} M parameters, {:.1} MB in {:.1}s",
        specs.len(),
        params as f64 / 1e6,
        size as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
