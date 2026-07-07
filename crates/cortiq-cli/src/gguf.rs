//! Native GGUF → `.cmf` importer. Parses a GGUF file, dequantizes F32/F16/Q8_0
//! tensors, maps ggml tensor names to HF names, reconstructs a Hugging Face
//! tokenizer.json from the embedded ggml metadata, and writes a `.cmf`. No
//! Python. K-quants (Q4_K/Q5_K/Q6_K) are not supported yet — the Python
//! importer handles those.

use crate::convert::{self, Quant};
use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec, TokenizerBundle, CMF_VERSION};
use cortiq_core::quant::f16_to_f32;
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType, TensorDtype};
use std::collections::BTreeMap;
use std::fs;

// GGUF metadata value types.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

// ggml tensor dtypes we can read.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q8_0: u32 = 8;

/// A parsed GGUF metadata value (only the parts we need are typed richly).
#[derive(Clone)]
enum Val {
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
    /// Array of strings (tokens / merges).
    StrArr(Vec<String>),
    /// Array of ints (token_type).
    IntArr(Vec<i64>),
    Other,
}

impl Val {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Val::U64(v) => Some(*v),
            Val::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Val::F64(v) => Some(*v),
            Val::U64(v) => Some(*v as f64),
            Val::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            anyhow::bail!("gguf: truncated");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn gstr(&mut self) -> anyhow::Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> anyhow::Result<Val> {
        Ok(match t {
            T_U8 | T_BOOL => Val::U64(self.take(1)?[0] as u64),
            T_I8 => Val::I64(self.take(1)?[0] as i8 as i64),
            T_U16 => Val::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            T_I16 => Val::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            T_U32 => Val::U64(self.u32()? as u64),
            T_I32 => Val::I64(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            T_F32 => Val::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            T_U64 => Val::U64(self.u64()?),
            T_I64 => Val::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_F64 => Val::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_STR => Val::Str(self.gstr()?),
            other => anyhow::bail!("gguf: bad value type {other}"),
        })
    }
    fn value(&mut self, t: u32) -> anyhow::Result<Val> {
        if t != T_ARR {
            return self.scalar(t);
        }
        let et = self.u32()?;
        let n = self.u64()? as usize;
        if et == T_STR {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.gstr()?);
            }
            Ok(Val::StrArr(v))
        } else if matches!(et, T_U8 | T_I8 | T_U16 | T_I16 | T_U32 | T_I32 | T_U64 | T_I64 | T_BOOL) {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.scalar(et)?.as_u64().map(|x| x as i64).unwrap_or(0));
            }
            Ok(Val::IntArr(v))
        } else {
            // arrays of floats etc. — consume and ignore
            for _ in 0..n {
                let _ = self.scalar(et)?;
            }
            Ok(Val::Other)
        }
    }
}

struct GgufTensor {
    name: String,
    dims: Vec<u64>, // ggml order (ne[0] fastest)
    ggml_type: u32,
    offset: u64, // relative to data section
}

struct Gguf {
    md: BTreeMap<String, Val>,
    tensors: Vec<GgufTensor>,
    bytes: Vec<u8>,
    data_start: usize,
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

fn parse(path: &std::path::Path) -> anyhow::Result<Gguf> {
    let bytes = fs::read(path)?;
    let mut c = Cursor { b: &bytes, p: 0 };
    if c.take(4)? != b"GGUF" {
        anyhow::bail!("not a GGUF file");
    }
    let _ver = c.u32()?;
    let n_tensors = c.u64()? as usize;
    let n_kv = c.u64()? as usize;
    let mut md = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.gstr()?;
        let t = c.u32()?;
        md.insert(key, c.value(t)?);
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.gstr()?;
        let nd = c.u32()? as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(c.u64()?);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(GgufTensor { name, dims, ggml_type, offset });
    }
    let align = md.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
    let data_start = align_up(c.p, align.max(1));
    Ok(Gguf { md, tensors, bytes, data_start })
}

/// Dequantize `n` elements of a ggml block into f32.
fn dequant(ggml_type: u32, raw: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    match ggml_type {
        GGML_F32 => Ok(raw.chunks_exact(4).take(n).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        GGML_F16 => Ok(raw.chunks_exact(2).take(n).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        GGML_Q8_0 => {
            // blocks of 34 bytes: [f16 scale][32 × int8]
            let mut out = Vec::with_capacity(n);
            for blk in raw.chunks_exact(34) {
                let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    out.push(q as i8 as f32 * scale);
                }
            }
            out.truncate(n);
            Ok(out)
        }
        other => anyhow::bail!(
            "ggml tensor type {other} not supported by the native importer (F32/F16/Q8_0 only — use the Python importer for K-quants)"
        ),
    }
}

fn nbytes(ggml_type: u32, n: usize) -> anyhow::Result<usize> {
    Ok(match ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 => n * 2,
        GGML_Q8_0 => n / 32 * 34,
        other => anyhow::bail!("ggml type {other} unsupported"),
    })
}

/// ggml tensor name → HF name (`None` = skip, e.g. rope freqs).
fn map_name(g: &str) -> Option<String> {
    match g {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = g.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let mapped = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{mapped}"))
}

fn arch_from_md(md: &BTreeMap<String, Val>) -> anyhow::Result<ModelArch> {
    let arch = md.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string();
    let g = |k: &str| md.get(&format!("{arch}.{k}"));
    let gu = |k: &str| g(k).and_then(|v| v.as_u64()).map(|x| x as usize);
    let n_layers = gu("block_count").ok_or_else(|| anyhow::anyhow!("gguf: no block_count"))?;
    let hidden = gu("embedding_length").ok_or_else(|| anyhow::anyhow!("gguf: no embedding_length"))?;
    let n_heads = gu("attention.head_count").ok_or_else(|| anyhow::anyhow!("gguf: no head_count"))?;
    let vocab = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| if let Val::StrArr(a) = v { Some(a.len()) } else { None })
        .unwrap_or(0);
    let norm_style = if arch.contains("gemma") { NormStyle::Gemma } else { NormStyle::Qwen };
    Ok(ModelArch {
        arch_name: arch.clone(),
        hidden_size: hidden,
        intermediate_size: gu("feed_forward_length").unwrap_or(0),
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: gu("attention.head_count_kv").unwrap_or(n_heads),
        head_dim: gu("attention.key_length").unwrap_or(hidden / n_heads.max(1)),
        vocab_size: vocab,
        layer_types: vec![LayerType::FullAttention; n_layers],
        rms_norm_eps: g("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        norm_style,
        rope_theta: g("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(10_000.0),
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: gu("context_length").unwrap_or(32_768),
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
    })
}

/// Reconstruct a HF byte-level-BPE tokenizer.json + chat bundle from ggml metadata.
fn tokenizer(md: &BTreeMap<String, Val>) -> (Option<Vec<u8>>, TokenizerBundle) {
    let empty = TokenizerBundle { chat_template: None, eos_token_ids: Vec::new(), bos_token_id: None, pad_token_id: None };
    let tokens = match md.get("tokenizer.ggml.tokens") {
        Some(Val::StrArr(a)) => a,
        _ => return (None, empty),
    };
    let types: &[i64] = match md.get("tokenizer.ggml.token_type") {
        Some(Val::IntArr(a)) => a,
        _ => &[],
    };
    let merges = match md.get("tokenizer.ggml.merges") {
        Some(Val::StrArr(a)) => a.clone(),
        _ => Vec::new(),
    };
    // vocab: token -> id
    let vocab: serde_json::Map<String, serde_json::Value> =
        tokens.iter().enumerate().map(|(i, t)| (t.clone(), serde_json::json!(i))).collect();
    // added/special tokens: ggml token_type CONTROL(3) / USER_DEFINED(4).
    let added: Vec<serde_json::Value> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| matches!(types.get(*i).copied(), Some(3) | Some(4)))
        .map(|(i, t)| {
            serde_json::json!({
                "id": i, "content": t, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    let tj = serde_json::json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges
        }
    });
    let eos = md.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let bos = md.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let pad = md.get("tokenizer.ggml.padding_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let bundle = TokenizerBundle {
        chat_template: md.get("tokenizer.chat_template").and_then(|v| v.as_str().map(String::from)),
        eos_token_ids: eos.into_iter().collect(),
        bos_token_id: bos,
        pad_token_id: pad,
    };
    (Some(serde_json::to_vec(&tj).unwrap()), bundle)
}

/// Import a GGUF file into a `.cmf` (quantized with `quant`).
pub fn run_import_gguf(gguf: &str, quant: &str, output: &str, mut progress: impl FnMut(f32)) -> anyhow::Result<()> {
    let quant = convert::parse_quant(quant)?;
    let g = parse(std::path::Path::new(gguf))?;
    let arch = arch_from_md(&g.md)?;
    let is_llama = arch.arch_name == "llama";
    let n_heads = arch.num_attention_heads;
    let n_kv = arch.num_kv_heads;

    let total = g.tensors.len().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    for (idx, t) in g.tensors.iter().enumerate() {
        progress((idx + 1) as f32 / total as f32);
        let Some(name) = map_name(&t.name) else { continue };
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        let nb = nbytes(t.ggml_type, numel)?;
        let raw = &g.bytes[g.data_start + t.offset as usize..g.data_start + t.offset as usize + nb];
        let mut vals = dequant(t.ggml_type, raw, numel)?;
        // HF shape = ggml dims reversed (ne[0] is fastest / the input dim).
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();

        // llama.cpp permutes q/k weights for its rope; undo it for HF layout.
        if is_llama && shape.len() == 2 {
            if name.ends_with("self_attn.q_proj.weight") {
                vals = unpermute(&vals, shape[0], shape[1], n_heads);
            } else if name.ends_with("self_attn.k_proj.weight") {
                vals = unpermute(&vals, shape[0], shape[1], n_kv);
            }
        }

        let two_d = shape.len() == 2 && numel >= 32;
        let (dt, data) = if two_d {
            convert::quantize_2d(quant, &vals, shape[0], shape[1])
        } else {
            (TensorDtype::F16, convert::encode_f16(&vals))
        };
        tensors.push(TensorSpec { name, dtype: dt, shape, data });
    }

    let (vocab, bundle) = tokenizer(&g.md);
    let quant_type = match quant {
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type,
        provenance: Some(serde_json::json!({ "tool": "cortiq import-gguf", "source": gguf })),
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

/// Undo llama.cpp's q/k rope permutation: rows are interleaved (d/2, 2) → (2, d/2).
fn unpermute(vals: &[f32], out_dim: usize, in_dim: usize, n_heads: usize) -> Vec<f32> {
    if n_heads == 0 || out_dim % n_heads != 0 {
        return vals.to_vec();
    }
    let hd = out_dim / n_heads; // head dim
    if hd % 2 != 0 {
        return vals.to_vec();
    }
    let half = hd / 2;
    let mut out = vec![0f32; vals.len()];
    for h in 0..n_heads {
        for r in 0..hd {
            // permuted row r ← original row: interleave halves
            let src_r = if r < half { r * 2 } else { (r - half) * 2 + 1 };
            let dst = (h * hd + r) * in_dim;
            let srco = (h * hd + src_r) * in_dim;
            out[dst..dst + in_dim].copy_from_slice(&vals[srco..srco + in_dim]);
        }
    }
    out
}
