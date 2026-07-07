//! Native Rust converter: a Hugging Face checkpoint (config.json +
//! *.safetensors + tokenizer.json) → a `.cmf` container. No Python, numpy, or
//! torch — reads safetensors and quantizes in Rust, then writes with
//! `cortiq_core::CmfModel::write`.
//!
//! Scope: standard dense transformers (qwen2 / qwen3 / llama / mistral-style,
//! RMSNorm + RoPE + SwiGLU, optional attention biases). Tensor handling is
//! arch-agnostic — 1-D tensors are stored f16, 2-D weights are quantized — so
//! it works by tensor presence without a hard-coded tensor set. Exotic layers
//! (GatedDeltaNet, MoE) are out of scope here and still use the Python path.

use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec, TokenizerBundle, CMF_VERSION};
use cortiq_core::quant::{bf16_to_f32, f16_to_f32, f32_to_f16};
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType, TensorDtype};
use std::fs;
use std::path::Path;

const GROUP_SIZE: usize = 32;
/// Smallest normal f16 — floor for degenerate (all-zero) rows so the stored
/// scale never underflows to a subnormal the reader would read back as 0.
const F16_TINY: f32 = 6.103_515_625e-5;

/// Round a scale to f16 precision (the reader stores/uses it as f16), so the
/// quantized values are computed against the *same* scale the reader dequantizes
/// with. This is what the reference converter does; without it `q` and the
/// stored scale disagree and inference degrades to garbage.
fn f16_scale(raw: f32) -> f32 {
    f16_to_f32(f32_to_f16(raw)).max(F16_TINY)
}

/// Quantization choice for 2-D weight matrices.
#[derive(Clone, Copy, PartialEq)]
enum Quant {
    Q8Row,
    Q4Block,
    F16,
}

fn parse_quant(s: &str) -> anyhow::Result<Quant> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q8" | "q8_row" | "q8row" => Quant::Q8Row,
        "q4" | "q4_block" | "q4block" => Quant::Q4Block,
        "f16" | "fp16" => Quant::F16,
        other => anyhow::bail!("unknown quant '{other}' (use q8, q4, or f16)"),
    })
}

/// q8_row: `[int8 : out·in][f16 : out]` (validated layout, matches the reader).
fn encode_q8_row(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let mut q = Vec::with_capacity(out_dim * in_dim);
    let mut scales = Vec::with_capacity(out_dim * 2);
    for o in 0..out_dim {
        let row = &vals[o * in_dim..(o + 1) * in_dim];
        let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = f16_scale(absmax / 127.0);
        for &v in row {
            q.push((v / scale).round().clamp(-128.0, 127.0) as i8 as u8);
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    q.extend_from_slice(&scales);
    q
}

/// q4_block: groups of 32 over the flattened tensor, `[u8 packed][f16 scales]`.
fn encode_q4_block(vals: &[f32]) -> Vec<u8> {
    let n_groups = vals.len().div_ceil(GROUP_SIZE);
    let mut padded = vals.to_vec();
    padded.resize(n_groups * GROUP_SIZE, 0.0);
    let mut packed = Vec::with_capacity(n_groups * 16);
    let mut scales = Vec::with_capacity(n_groups * 2);
    for g in 0..n_groups {
        let group = &padded[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let absmax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = f16_scale(absmax / 7.0);
        for k in 0..16 {
            let q0 = ((group[k * 2] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((group[k * 2 + 1] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            packed.push((q0 & 0x0F) | (q1 << 4));
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    packed.extend_from_slice(&scales);
    packed
}

/// f16 blob for a 1-D / small tensor.
fn encode_f16(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

/// Decode a safetensors dtype blob into f32 values.
fn to_f32(dtype: &str, raw: &[u8]) -> anyhow::Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        "BF16" => raw.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        other => anyhow::bail!("unsupported safetensors dtype '{other}' (need F32/F16/BF16)"),
    })
}

/// One safetensors file → (name, dtype, shape, raw-bytes) per tensor.
#[allow(clippy::type_complexity)]
fn read_safetensors(path: &Path) -> anyhow::Result<Vec<(String, String, Vec<usize>, Vec<u8>)>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        anyhow::bail!("{}: too small to be safetensors", path.display());
    }
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])?;
    let data_start = 8 + hlen;
    let obj = header.as_object().ok_or_else(|| anyhow::anyhow!("bad safetensors header"))?;
    let mut out = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> =
            v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow::anyhow!("tensor '{name}': no data_offsets"))?;
        let s = offs[0].as_u64().unwrap_or(0) as usize;
        let e = offs[1].as_u64().unwrap_or(0) as usize;
        out.push((name.clone(), dtype, shape, bytes[data_start + s..data_start + e].to_vec()));
    }
    Ok(out)
}

/// Gather all tensors from a model dir (single file or sharded index).
#[allow(clippy::type_complexity)]
fn read_model_tensors(dir: &Path) -> anyhow::Result<Vec<(String, String, Vec<usize>, Vec<u8>)>> {
    let index = dir.join("model.safetensors.index.json");
    let single = dir.join("model.safetensors");
    if single.exists() {
        return read_safetensors(&single);
    }
    if index.exists() {
        let idx: serde_json::Value = serde_json::from_slice(&fs::read(&index)?)?;
        let map = idx["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("bad index json"))?;
        let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        let mut all = Vec::new();
        for f in files {
            all.extend(read_safetensors(&dir.join(f))?);
        }
        return Ok(all);
    }
    anyhow::bail!("no model.safetensors or model.safetensors.index.json in {}", dir.display())
}

fn cfg_usize(c: &serde_json::Value, key: &str) -> Option<usize> {
    c.get(key).and_then(|v| v.as_u64()).map(|x| x as usize)
}

/// Build ModelArch from a HF config.json (dense transformer families).
fn build_arch(config: &serde_json::Value) -> anyhow::Result<ModelArch> {
    // Vision/multimodal configs nest the text model under "text_config".
    let tc = config.get("text_config").unwrap_or(config);
    let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let hidden = cfg_usize(tc, "hidden_size").ok_or_else(|| anyhow::anyhow!("config: missing hidden_size"))?;
    let n_heads = cfg_usize(tc, "num_attention_heads").ok_or_else(|| anyhow::anyhow!("config: missing num_attention_heads"))?;
    let n_layers = cfg_usize(tc, "num_hidden_layers").ok_or_else(|| anyhow::anyhow!("config: missing num_hidden_layers"))?;
    if tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) > 0
        || tc.get("linear_num_value_heads").is_some()
    {
        anyhow::bail!("this model uses MoE / linear-attention layers — not supported by the native converter yet (use the Python converter)");
    }
    let head_dim = cfg_usize(tc, "head_dim").unwrap_or(hidden / n_heads.max(1));
    let norm_style = if model_type.to_lowercase().contains("gemma") { NormStyle::Gemma } else { NormStyle::Qwen };
    Ok(ModelArch {
        arch_name: model_type,
        hidden_size: hidden,
        intermediate_size: cfg_usize(tc, "intermediate_size").ok_or_else(|| anyhow::anyhow!("config: missing intermediate_size"))?,
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads),
        head_dim,
        vocab_size: cfg_usize(tc, "vocab_size").ok_or_else(|| anyhow::anyhow!("config: missing vocab_size"))?,
        layer_types: vec![LayerType::FullAttention; n_layers],
        rms_norm_eps: tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        norm_style,
        rope_theta: tc.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10_000.0),
        tie_word_embeddings: config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false),
        partial_rotary_factor: tc.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: cfg_usize(tc, "max_position_embeddings").unwrap_or(32_768),
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
    })
}

/// Collect eos ids from generation_config.json / config.json (int or array).
fn eos_ids(gen_cfg: &serde_json::Value, config: &serde_json::Value) -> Vec<u32> {
    for src in [gen_cfg.get("eos_token_id"), config.get("eos_token_id")] {
        if let Some(v) = src {
            if let Some(n) = v.as_u64() {
                return vec![n as u32];
            }
            if let Some(a) = v.as_array() {
                return a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
            }
        }
    }
    Vec::new()
}

/// Convert a HF model directory to a `.cmf` file. `progress` receives fraction
/// 0..1 (used to stream `@PROGRESS` markers for a UI).
pub fn run_convert(
    model_dir: &str,
    quant: &str,
    output: &str,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    let dir = Path::new(model_dir);
    let quant = parse_quant(quant)?;

    let config: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("read config.json: {e}"))?)?;
    let arch = build_arch(&config)?;

    let raw = read_model_tensors(dir)?;
    let total = raw.len().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(raw.len());
    for (i, (name, dtype, shape, bytes)) in raw.into_iter().enumerate() {
        let vals = to_f32(&dtype, &bytes)?;
        let numel: usize = shape.iter().product();
        if numel != vals.len() {
            anyhow::bail!("tensor '{name}': {} values for shape {:?}", vals.len(), shape);
        }
        // 1-D tensors, tiny tensors, and non-2-D always go f16 (norms, biases).
        let two_d = shape.len() == 2 && numel >= GROUP_SIZE;
        let (dt, data) = if !two_d {
            (TensorDtype::F16, encode_f16(&vals))
        } else {
            match quant {
                Quant::Q8Row => (TensorDtype::Q8Row, encode_q8_row(&vals, shape[0], shape[1])),
                Quant::Q4Block => (TensorDtype::Q4Block, encode_q4_block(&vals)),
                Quant::F16 => (TensorDtype::F16, encode_f16(&vals)),
            }
        };
        tensors.push(TensorSpec { name, dtype: dt, shape, data });
        progress((i + 1) as f32 / total as f32);
    }

    // Tokenizer + chat bundle (optional but recommended).
    let vocab = fs::read(dir.join("tokenizer.json")).ok();
    let tok_cfg: serde_json::Value =
        fs::read(dir.join("tokenizer_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::Value::Null);
    let gen_cfg: serde_json::Value =
        fs::read(dir.join("generation_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::Value::Null);
    let chat_template = fs::read_to_string(dir.join("chat_template.jinja")).ok()
        .or_else(|| tok_cfg.get("chat_template").and_then(|v| v.as_str().map(String::from)));
    let bundle = TokenizerBundle {
        chat_template,
        eos_token_ids: eos_ids(&gen_cfg, &config),
        bos_token_id: config.get("bos_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
        pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
    };

    let quant_type = match quant {
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type,
        provenance: Some(serde_json::json!({ "tool": "cortiq convert", "source_model": model_dir })),
        tokenizer_config: Some(bundle),
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };

    CmfModel::write(output, &header, &tensors, None, vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    progress(1.0);
    Ok(())
}
