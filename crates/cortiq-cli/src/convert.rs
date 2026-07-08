//! Native Rust converter: a Hugging Face checkpoint (config.json +
//! *.safetensors + tokenizer.json) → a `.cmf` container. No Python, numpy, or
//! torch — reads safetensors and quantizes in Rust, then writes with
//! `cortiq_core::CmfModel::write`.
//!
//! Scope: standard dense transformers (qwen2 / qwen3 / llama / mistral-style,
//! RMSNorm + RoPE + SwiGLU, optional attention biases). Tensor handling is
//! arch-agnostic — 1-D tensors are stored f16, 2-D weights are quantized — so
//! it works by tensor presence without a hard-coded tensor set. Mixture-of-experts
//! is supported (router + per-expert matrices), as is GatedDeltaNet linear
//! attention in the Qwen3.5 hub layout (separate in_proj_qkv/z/a/b). Fused
//! qwen3_next / AgentWorld checkpoints still use the Python path.

use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec, TokenizerBundle, CMF_VERSION};
use cortiq_core::quant::{bf16_to_f32, f16_to_f32, f32_to_f16};
use cortiq_core::types::{LayerType, LinearCoreConfig, ModelArch, MoeConfig, NormStyle, QuantType, TensorDtype};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

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

/// Canonicalize a source tensor name to the CMF layout the runtime expects, or
/// `None` to skip it. Multimodal wrappers (Qwen3.5) nest the text model under
/// `model.language_model.*`; vision (`*.visual.*`) and the MTP head (`mtp.*`) are
/// dropped — plain greedy decoding is correct without MTP.
fn canon_name(raw: &str) -> Option<String> {
    if raw.contains(".visual.") || raw.starts_with("visual.") || raw.starts_with("mtp.") || raw.contains(".mtp.") {
        return None;
    }
    for pfx in ["model.language_model.", "language_model.model.", "language_model."] {
        if let Some(rest) = raw.strip_prefix(pfx) {
            return Some(format!("model.{rest}"));
        }
    }
    Some(raw.to_string())
}

/// Small, noise-sensitive 2-D projections the reference converter keeps at f16
/// (a bit-flip there is costly): the GDN a/b gate projections and MoE routers.
fn force_f16(name: &str) -> bool {
    name.ends_with("linear_attn.in_proj_a.weight")
        || name.ends_with("linear_attn.in_proj_b.weight")
        || name.ends_with("mlp.gate.weight")
        || name.ends_with("shared_expert_gate.weight")
}

/// Quantization choice for 2-D weight matrices.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Quant {
    Q8Row,
    Q8_2f,
    Q4Block,
    F16,
    /// Grouped variable-bit (per-row 3–8 bit, water-filled by row amplitude).
    Vbit,
}

/// Quantize a 2-D matrix `[out_dim, in_dim]` per the chosen scheme.
pub(crate) fn quantize_2d(quant: Quant, vals: &[f32], out_dim: usize, in_dim: usize) -> (TensorDtype, Vec<u8>) {
    match quant {
        Quant::Q8Row => (TensorDtype::Q8Row, encode_q8_row(vals, out_dim, in_dim)),
        Quant::Q8_2f => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q4Block => (TensorDtype::Q4Block, encode_q4_block(vals)),
        Quant::F16 => (TensorDtype::F16, encode_f16(vals)),
        // v-bit needs the input dim to be a multiple of the group size; other
        // shapes fall back to the two-field q8_2f (best equal-size alternative).
        Quant::Vbit if in_dim % GROUP_SIZE == 0 => (TensorDtype::Vbit, encode_vbit(vals, out_dim, in_dim)),
        Quant::Vbit => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
    }
}

pub(crate) fn parse_quant(s: &str) -> anyhow::Result<Quant> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q8" | "q8_row" | "q8row" => Quant::Q8Row,
        "q8_2f" | "q82f" | "q8f" => Quant::Q8_2f,
        "q4" | "q4_block" | "q4block" => Quant::Q4Block,
        "f16" | "fp16" => Quant::F16,
        "vbit" | "v_bit" => Quant::Vbit,
        other => anyhow::bail!("unknown quant '{other}' (use q8, q8_2f, q4, f16, or vbit)"),
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
            // round-half-to-even matches numpy's np.round → byte-identical weights.
            q.push((v / scale).round_ties_even().clamp(-128.0, 127.0) as i8 as u8);
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
            let q0 = ((group[k * 2] / scale).round_ties_even().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((group[k * 2 + 1] / scale).round_ties_even().clamp(-8.0, 7.0) as i8 + 8) as u8;
            packed.push((q0 & 0x0F) | (q1 << 4));
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    packed.extend_from_slice(&scales);
    packed
}

/// q8_2f (two-field 𝒲×θ): `[int8: out·in][f16 row_scale: out][f16 col: in]`.
/// `col[i]` = RMS over rows (absorbs outlier input channels); each row is int8
/// over the residual normalized by col. Dequant: `w = q·scale[o]·col[i]`.
/// Recovers most of the q8→f16 quality gap at the same size.
fn encode_q8_2f(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    // Column field: RMS over rows, f16-rounded (the decoder multiplies by these).
    let mut col = vec![0f32; in_dim];
    for (i, c) in col.iter_mut().enumerate() {
        let mut acc = 0f64;
        for o in 0..out_dim {
            let v = vals[o * in_dim + i] as f64;
            acc += v * v;
        }
        let rms = (acc / out_dim as f64).sqrt().max(1e-12) as f32;
        *c = f16_to_f32(f32_to_f16(rms)).max(F16_TINY);
    }
    let mut q = Vec::with_capacity(out_dim * in_dim);
    let mut scales = Vec::with_capacity(out_dim * 2);
    for o in 0..out_dim {
        let mut absmax = 0f32;
        for i in 0..in_dim {
            absmax = absmax.max((vals[o * in_dim + i] / col[i]).abs());
        }
        let scale = f16_scale(absmax.max(1e-12) / 127.0);
        for i in 0..in_dim {
            let wn = vals[o * in_dim + i] / col[i];
            q.push((wn / scale).round_ties_even().clamp(-127.0, 127.0) as i8 as u8);
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    let mut out = q;
    out.extend_from_slice(&scales);
    for &c in &col {
        out.extend_from_slice(&f32_to_f16(c).to_le_bytes());
    }
    out
}

// Grouped variable-bit (v-bit) encoder — the weight-only (round-to-nearest) path
// of the reference converter. On-disk layout read by `cortiq_core::dequant_vbit`:
//   [u8 bits: rows][f16 scales: rows·(in/32)][per row: ceil(in·b/8) bytes,
//    MSB-first b-bit codes, zero-padded]. w = (u − L)·scale, L = 2^(b−1)−1.
// The GPTQ / calibrated variant (needs a Hessian) stays in the Python converter.
const VBIT_LEVELS: [u8; 5] = [3, 4, 5, 6, 8];
/// Target mean bit-width for VBIT water-filling. Default 4.25; overridable via
/// `cortiq convert --mean-bits` (stored ×1000 in a static to avoid signature churn).
static VBIT_MEAN_BITS_MILLI: AtomicU32 = AtomicU32::new(4250);
/// Set the VBIT target mean bit-width (converter CLI knob). Clamped to [3.0, 8.0].
pub fn set_vbit_mean_bits(bits: f32) {
    VBIT_MEAN_BITS_MILLI.store((bits.clamp(3.0, 8.0) * 1000.0) as u32, Ordering::Relaxed);
}
fn vbit_mean_bits() -> f32 {
    VBIT_MEAN_BITS_MILLI.load(Ordering::Relaxed) as f32 / 1000.0
}

/// Snap `x` to the nearest allowed bit-width (first wins on a tie, like argmin).
fn vbit_snap_level(x: f32) -> u8 {
    let mut best = VBIT_LEVELS[0];
    let mut bestd = (x - best as f32).abs();
    for &lv in &VBIT_LEVELS[1..] {
        let d = (x - lv as f32).abs();
        if d < bestd {
            bestd = d;
            best = lv;
        }
    }
    best
}

/// Per-row bit-width via water-filling over log2 row amplitude (floor 3 bits).
fn vbit_bits(vals: &[f32], out_dim: usize, in_dim: usize, mean_bits: f32) -> Vec<u8> {
    let a: Vec<f32> = (0..out_dim)
        .map(|o| {
            let mx = vals[o * in_dim..(o + 1) * in_dim].iter().fold(0f32, |m, v| m.max(v.abs()));
            mx.max(1e-12).log2()
        })
        .collect();
    let amean = a.iter().sum::<f32>() / out_dim as f32;
    a.iter().map(|&ar| vbit_snap_level(mean_bits + (ar - amean)).max(3)).collect()
}

/// Big-endian (MSB-first) bit packer; the last byte of each row is zero-padded.
struct BitWriter {
    buf: Vec<u8>,
    cur: u8,
    nbits: u8,
}
impl BitWriter {
    fn with_capacity(n: usize) -> Self {
        Self { buf: Vec::with_capacity(n), cur: 0, nbits: 0 }
    }
    fn push(&mut self, v: u32, b: u32) {
        for i in (0..b).rev() {
            self.cur = (self.cur << 1) | ((v >> i) & 1) as u8;
            self.nbits += 1;
            if self.nbits == 8 {
                self.buf.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }
    fn flush_row(&mut self) {
        if self.nbits > 0 {
            self.buf.push(self.cur << (8 - self.nbits));
            self.cur = 0;
            self.nbits = 0;
        }
    }
}

fn encode_vbit(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let ng = in_dim / GROUP_SIZE;
    let bits = vbit_bits(vals, out_dim, in_dim, vbit_mean_bits());

    // Per-(row, group) scale = group absmax / L, f16-rounded and floored.
    let mut scale = vec![0f32; out_dim * ng];
    let mut sc_bytes = Vec::with_capacity(out_dim * ng * 2);
    for o in 0..out_dim {
        let l = (2f32.powi(bits[o] as i32 - 1) - 1.0).max(1.0);
        for g in 0..ng {
            let base = o * in_dim + g * GROUP_SIZE;
            let mx = vals[base..base + GROUP_SIZE].iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = f16_scale(mx / l);
            scale[o * ng + g] = s;
            sc_bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        }
    }

    let mut out = Vec::with_capacity(out_dim + sc_bytes.len() + out_dim * in_dim);
    out.extend_from_slice(&bits);
    out.extend_from_slice(&sc_bytes);
    let mut bw = BitWriter::with_capacity(out_dim * in_dim);
    for o in 0..out_dim {
        let b = bits[o] as u32;
        let l = 2f32.powi(bits[o] as i32 - 1) - 1.0;
        let maxq = 2f32.powi(bits[o] as i32) - 1.0;
        for c in 0..in_dim {
            let s = scale[o * ng + c / GROUP_SIZE];
            let q = ((vals[o * in_dim + c] / s).round_ties_even() + l).clamp(0.0, maxq) as u32;
            bw.push(q, b);
        }
        bw.flush_row();
    }
    out.extend_from_slice(&bw.buf);
    out
}

/// f16 blob for a 1-D / small tensor.
pub(crate) fn encode_f16(vals: &[f32]) -> Vec<u8> {
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

/// A tensor's metadata within a safetensors file (bytes are read lazily from mmap).
struct TensorMeta {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A memory-mapped safetensors file — tensor bytes are borrowed from the mmap, so
/// the raw weights are never fully loaded into RAM (peak stays ~one tensor).
struct SafeTensors {
    mmap: memmap2::Mmap,
    data_start: usize,
    tensors: Vec<TensorMeta>,
}

impl SafeTensors {
    fn bytes(&self, m: &TensorMeta) -> &[u8] {
        &self.mmap[self.data_start + m.start..self.data_start + m.end]
    }
}

fn open_safetensors(path: &Path) -> anyhow::Result<SafeTensors> {
    let file = fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    if mmap.len() < 8 {
        anyhow::bail!("{}: too small to be safetensors", path.display());
    }
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])?;
    let data_start = 8 + hlen;
    let obj = header.as_object().ok_or_else(|| anyhow::anyhow!("bad safetensors header"))?;
    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> =
            v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow::anyhow!("tensor '{name}': no data_offsets"))?;
        let start = offs[0].as_u64().unwrap_or(0) as usize;
        let end = offs[1].as_u64().unwrap_or(0) as usize;
        tensors.push(TensorMeta { name: name.clone(), dtype, shape, start, end });
    }
    Ok(SafeTensors { mmap, data_start, tensors })
}

/// Memory-map a model dir's weights (single file or sharded index).
fn open_model(dir: &Path) -> anyhow::Result<Vec<SafeTensors>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![open_safetensors(&single)?]);
    }
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: serde_json::Value = serde_json::from_slice(&fs::read(&index)?)?;
        let map = idx["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("bad index json"))?;
        let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        return files.iter().map(|f| open_safetensors(&dir.join(f))).collect();
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
    // Linear-attention (GatedDeltaNet, Qwen3.5): the per-layer schedule comes
    // from config.layer_types; the vendor operator is carried 1:1 and we declare
    // the canonical core so the runtime dispatches it.
    let layer_types: Vec<LayerType> = match tc.get("layer_types").and_then(|v| v.as_array()) {
        Some(a) => a
            .iter()
            .map(|v| {
                if v.as_str() == Some("linear_attention") {
                    LayerType::LinearAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect(),
        None => vec![LayerType::FullAttention; n_layers],
    };
    let has_linear = layer_types.iter().any(|t| matches!(t, LayerType::LinearAttention));
    let lnv = cfg_usize(tc, "linear_num_value_heads");
    let lvd = cfg_usize(tc, "linear_value_head_dim");
    let linear_core = if has_linear {
        Some(LinearCoreConfig {
            kind: "gated_delta_net".into(),
            num_heads: lnv.unwrap_or(0),
            nphase: None,
            value_head_dim: lvd.unwrap_or(0),
        })
    } else {
        None
    };
    // Qwen3.5 nests rope params under `rope_parameters`.
    let rope = tc.get("rope_parameters");
    let rope_theta = tc
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| rope.and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64()))
        .unwrap_or(10_000.0);
    let prf = tc
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| rope.and_then(|r| r.get("partial_rotary_factor")).and_then(|v| v.as_f64()))
        .unwrap_or(1.0) as f32;
    // Mixture-of-experts: the FFN becomes a router + per-expert matrices. Tensor
    // handling is unchanged (experts are ordinary 2-D matrices); we just declare
    // the MoE config so the runtime dispatches it. Router presence per layer
    // (in the directory) decides which layers are sparse.
    let moe = tc.get("num_experts").and_then(|v| v.as_u64()).filter(|&n| n > 0).map(|ne| {
        let mt = model_type.to_lowercase();
        let ntp_default = mt.starts_with("qwen3_5") || mt.contains("qwen3_next");
        MoeConfig {
            num_experts: ne as usize,
            top_k: cfg_usize(tc, "num_experts_per_tok").unwrap_or(2),
            moe_intermediate_size: cfg_usize(tc, "moe_intermediate_size").unwrap_or(0),
            norm_topk_prob: tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(ntp_default),
            shared_expert_intermediate_size: cfg_usize(tc, "shared_expert_intermediate_size"),
        }
    });
    let head_dim = cfg_usize(tc, "head_dim").unwrap_or(hidden / n_heads.max(1));
    // Zero-centered RMSNorm x̂·(1+w): Gemma family and Qwen3.5 / Qwen3-Next.
    let mt = model_type.to_lowercase();
    let norm_style = if mt.contains("gemma") || mt.starts_with("qwen3_5") || mt.contains("qwen3_next") {
        NormStyle::Gemma
    } else {
        NormStyle::Qwen
    };
    Ok(ModelArch {
        arch_name: model_type,
        hidden_size: hidden,
        intermediate_size: cfg_usize(tc, "intermediate_size")
            .or_else(|| cfg_usize(tc, "moe_intermediate_size"))
            .ok_or_else(|| anyhow::anyhow!("config: missing intermediate_size"))?,
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads),
        head_dim,
        vocab_size: cfg_usize(tc, "vocab_size").ok_or_else(|| anyhow::anyhow!("config: missing vocab_size"))?,
        layer_types,
        rms_norm_eps: tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        norm_style,
        rope_theta,
        tie_word_embeddings: config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false),
        partial_rotary_factor: prf,
        mtp: None,
        moe,
        linear_core,
        max_position_embeddings: cfg_usize(tc, "max_position_embeddings").unwrap_or(32_768),
        linear_conv_kernel_dim: cfg_usize(tc, "linear_conv_kernel_dim"),
        linear_num_key_heads: cfg_usize(tc, "linear_num_key_heads"),
        linear_num_value_heads: lnv,
        linear_key_head_dim: cfg_usize(tc, "linear_key_head_dim"),
        linear_value_head_dim: lvd,
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

/// `owner/name` HF repo id (not an existing local path).
pub(crate) fn looks_like_repo(s: &str) -> bool {
    let s = s.trim_matches('/');
    s.split('/').count() == 2 && !s.contains(char::is_whitespace) && !Path::new(s).exists()
}

/// A fresh ureq agent with the same timeouts the downloader uses.
fn hf_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(300))
        .build()
}

/// List a repo's files via the HF API (best-effort; empty on failure). Reused by
/// the GGUF importer to pick a `.gguf` from a repo.
pub(crate) fn hf_repo_files(repo: &str, token: Option<&str>) -> Vec<String> {
    repo_files(&hf_agent(), repo, token)
}

/// Download a single named file from an HF repo into the cache (parallel chunks
/// for large files); returns its local path. Used to fetch one `.gguf`.
pub(crate) fn hf_fetch_file(
    repo: &str,
    filename: &str,
    token: Option<&str>,
) -> anyhow::Result<std::path::PathBuf> {
    let dir = hf_cache_dir(repo)?;
    let dest = dir.join(filename.replace('/', "__"));
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    fetch(&hf_agent(), &url, &dest, token, true, hf_threads())?;
    Ok(dest)
}

/// Local cache dir for a downloaded HF repo (`~/.cache/cortiq/hf/owner--name`).
fn hf_cache_dir(repo: &str) -> anyhow::Result<std::path::PathBuf> {
    let base = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".cache/cortiq/hf"))
        .unwrap_or_else(|| std::path::PathBuf::from(".cortiq-hf"));
    let dir = base.join(repo.replace('/', "--"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Parallel range chunk size (32 MiB) and default connection count.
const HF_CHUNK: u64 = 32 * 1024 * 1024;

fn hf_threads() -> usize {
    std::env::var("CORTIQ_HF_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(8)
        .min(16)
}

fn cached(dest: &Path) -> bool {
    dest.exists() && fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false)
}

fn auth<'a>(mut req: ureq::Request, token: Option<&'a str>) -> ureq::Request {
    req = req.set("User-Agent", "cortiq-convert");
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    req
}

/// Total size of a remote file via a `Range: bytes=0-0` probe (Content-Range),
/// or None if the server doesn't support/report ranges (→ single stream).
fn probe_size(agent: &ureq::Agent, url: &str, token: Option<&str>) -> Option<u64> {
    let resp = auth(agent.get(url).set("Range", "bytes=0-0"), token).call().ok()?;
    resp.header("Content-Range")?.rsplit('/').next()?.trim().parse::<u64>().ok()
}

fn get_range(agent: &ureq::Agent, url: &str, token: Option<&str>, start: u64, end: u64) -> anyhow::Result<Vec<u8>> {
    let resp = auth(agent.get(url).set("Range", &format!("bytes={}-{}", start, end - 1)), token)
        .call()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut buf = Vec::with_capacity((end - start) as usize);
    resp.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn write_at(path: &Path, offset: u64, data: &[u8]) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut f = fs::OpenOptions::new().write(true).open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(data)
}

/// Retry `f` with exponential backoff — smooths over transient network errors.
fn with_retry<T>(attempts: u32, mut f: impl FnMut() -> anyhow::Result<T>) -> anyhow::Result<T> {
    let mut delay = Duration::from_millis(400);
    let mut last: Option<anyhow::Error> = None;
    for a in 0..attempts {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                last = Some(e);
                if a + 1 < attempts {
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(Duration::from_secs(8));
                }
            }
        }
    }
    Err(last.unwrap())
}

/// Fetch one file into `dest` (cached). Large range-capable files are pulled in
/// parallel 32 MiB chunks over `threads` reused connections; otherwise a single
/// stream. Returns false on 404 when `required` is false.
fn fetch(agent: &ureq::Agent, url: &str, dest: &Path, token: Option<&str>, required: bool, threads: usize) -> anyhow::Result<bool> {
    if cached(dest) {
        return Ok(true);
    }
    let tmp = dest.with_extension("part");
    let size = probe_size(agent, url, token);
    if let Some(sz) = size {
        if sz > HF_CHUNK && threads > 1 {
            {
                let f = fs::File::create(&tmp)?;
                f.set_len(sz)?;
            }
            let chunks: Vec<(u64, u64)> =
                (0..sz).step_by(HF_CHUNK as usize).map(|s| (s, (s + HF_CHUNK).min(sz))).collect();
            let total = chunks.len();
            let queue = Mutex::new(chunks);
            let err: Mutex<Option<String>> = Mutex::new(None);
            let done = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| loop {
                        if err.lock().unwrap().is_some() {
                            break;
                        }
                        let Some((start, end)) = queue.lock().unwrap().pop() else { break };
                        // Each chunk retries on a transient failure before aborting.
                        let r = with_retry(4, || get_range(agent, url, token, start, end))
                            .and_then(|buf| write_at(&tmp, start, &buf).map_err(Into::into));
                        match r {
                            Ok(()) => {
                                let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                eprint!("\r    downloading: {:>3}% ({d}/{total} chunks)", d * 100 / total);
                            }
                            Err(e) => {
                                *err.lock().unwrap() = Some(e.to_string());
                                break;
                            }
                        }
                    });
                }
            });
            eprintln!();
            if let Some(e) = err.into_inner().unwrap() {
                anyhow::bail!("download {url}: {e}");
            }
            fs::rename(&tmp, dest)?;
            return Ok(true);
        }
    }
    // Small file / no range support → single stream (with retry). Returns
    // Some(()) on success, None on an allowed 404 (optional file).
    let got = with_retry(4, || match auth(agent.get(url), token).call() {
        Ok(resp) => {
            let mut r = resp.into_reader();
            let mut f = fs::File::create(&tmp)?;
            std::io::copy(&mut r, &mut f)?;
            Ok(Some(()))
        }
        Err(ureq::Error::Status(404, _)) if !required => Ok(None),
        Err(e) => Err(anyhow::anyhow!("download {url}: {e}")),
    })?;
    match got {
        Some(()) => {
            fs::rename(&tmp, dest)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// List a repo's file names via the HF API (best-effort; empty on any failure).
fn repo_files(agent: &ureq::Agent, repo: &str, token: Option<&str>) -> Vec<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    match auth(agent.get(&url), token).call() {
        Ok(resp) => resp
            .into_json::<serde_json::Value>()
            .ok()
            .and_then(|j| {
                j["siblings"].as_array().map(|a| {
                    a.iter().filter_map(|s| s["rfilename"].as_str().map(String::from)).collect()
                })
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Fetch a HF repo's convertible files (config, tokenizer, weights) into the
/// cache, with parallel chunked downloads for the weight shards.
fn hf_download(repo: &str, token: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    let dir = hf_cache_dir(repo)?;
    let base = format!("https://huggingface.co/{repo}/resolve/main");
    let threads = hf_threads();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(300))
        .build();
    // config.json is mandatory for the safetensors path. If it is absent, give an
    // actionable message rather than a raw 404 — most often the repo is a GGUF-only
    // distribution (has `*.gguf`, no `config.json`), which needs a different tool.
    if !fetch(&agent, &format!("{base}/config.json"), &dir.join("config.json"), token, false, threads)? {
        let files = repo_files(&agent, repo, token);
        let ggufs = files.iter().filter(|f| f.to_lowercase().ends_with(".gguf")).count();
        if ggufs > 0 {
            let src = repo
                .strip_suffix("-GGUF")
                .or_else(|| repo.strip_suffix("-gguf"))
                .filter(|s| !s.is_empty());
            anyhow::bail!(
                "'{repo}' is a GGUF repository ({ggufs} .gguf file(s), no config.json); \
                 `cortiq convert` needs a safetensors checkpoint. Either import a GGUF file \
                 directly with `cortiq import-gguf <file.gguf>` (dense llama/qwen2/qwen3, F32/F16/Q8_0), \
                 or convert the source safetensors repo instead{}.",
                match src {
                    Some(s) => format!(" — try `--model {s}`"),
                    None => String::new(),
                }
            );
        }
        anyhow::bail!("'{repo}': no config.json — not a Hugging Face safetensors checkpoint");
    }
    for (f, required) in [
        ("tokenizer.json", true),
        ("tokenizer_config.json", false),
        ("generation_config.json", false),
    ] {
        fetch(&agent, &format!("{base}/{f}"), &dir.join(f), token, required, threads)?;
    }
    let idx = dir.join("model.safetensors.index.json");
    if fetch(&agent, &format!("{base}/model.safetensors.index.json"), &idx, token, false, 1)? {
        let j: serde_json::Value = serde_json::from_slice(&fs::read(&idx)?)?;
        let map = j["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("bad safetensors index"))?;
        let mut shards: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        shards.sort();
        shards.dedup();
        for (i, s) in shards.iter().enumerate() {
            eprintln!("  shard {}/{} ({threads}× parallel): {s}", i + 1, shards.len());
            fetch(&agent, &format!("{base}/{s}"), &dir.join(s), token, true, threads)?;
        }
    } else {
        eprintln!("  model.safetensors ({threads}× parallel)");
        fetch(&agent, &format!("{base}/model.safetensors"), &dir.join("model.safetensors"), token, true, threads)?;
    }
    Ok(dir)
}

/// Split a fused GDN projection (`in_proj_qkvz` or `in_proj_ba`) into the
/// canonical hub tensors. The fused weight is `[nk · group_width, hid]`; rows
/// are grouped by k-head. This mirrors transformers' `fix_query_key_value_ordering`
/// inverse — a pure row permutation, no value changes. Returns `(name, values,
/// out_rows)` for each produced tensor.
fn split_fused_gdn(
    name: &str,
    w: &[f32],
    hid: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> anyhow::Result<Vec<(String, Vec<f32>, usize)>> {
    if nk == 0 || nv % nk != 0 {
        anyhow::bail!("fused GDN: bad head config nk={nk} nv={nv}");
    }
    let r = nv / nk;
    // Row `g·gw + gr` of the source (group g, within-group row gr).
    let row = |w: &[f32], gw: usize, g: usize, gr: usize| -> Vec<f32> {
        let base = (g * gw + gr) * hid;
        w[base..base + hid].to_vec()
    };

    if name.contains("in_proj_qkvz") {
        let gw = 2 * dk + 2 * r * dv;
        if w.len() != nk * gw * hid {
            anyhow::bail!("fused GDN qkvz: {} values, expected {}", w.len(), nk * gw * hid);
        }
        // qkv = [q: nk·dk][k: nk·dk][v: nv·dv]
        let mut qkv = Vec::with_capacity((2 * nk * dk + nv * dv) * hid);
        for g in 0..nk {
            for rr in 0..dk {
                qkv.extend_from_slice(&row(w, gw, g, rr));
            }
        }
        for g in 0..nk {
            for rr in 0..dk {
                qkv.extend_from_slice(&row(w, gw, g, dk + rr));
            }
        }
        for g in 0..nk {
            for rr in 0..r * dv {
                qkv.extend_from_slice(&row(w, gw, g, 2 * dk + rr));
            }
        }
        // z = nv·dv
        let mut z = Vec::with_capacity(nv * dv * hid);
        for g in 0..nk {
            for rr in 0..r * dv {
                z.extend_from_slice(&row(w, gw, g, 2 * dk + r * dv + rr));
            }
        }
        let p = name.strip_suffix("in_proj_qkvz.weight").unwrap_or(name);
        Ok(vec![
            (format!("{p}in_proj_qkv.weight"), qkv, 2 * nk * dk + nv * dv),
            (format!("{p}in_proj_z.weight"), z, nv * dv),
        ])
    } else {
        // in_proj_ba: group width 2·r → b (first r per group), a (next r) → nv rows each.
        let gw = 2 * r;
        if w.len() != nk * gw * hid {
            anyhow::bail!("fused GDN ba: {} values, expected {}", w.len(), nk * gw * hid);
        }
        let mut b = Vec::with_capacity(nv * hid);
        let mut a = Vec::with_capacity(nv * hid);
        for g in 0..nk {
            for rr in 0..r {
                b.extend_from_slice(&row(w, gw, g, rr));
            }
        }
        for g in 0..nk {
            for rr in 0..r {
                a.extend_from_slice(&row(w, gw, g, r + rr));
            }
        }
        let p = name.strip_suffix("in_proj_ba.weight").unwrap_or(name);
        Ok(vec![
            (format!("{p}in_proj_b.weight"), b, nv),
            (format!("{p}in_proj_a.weight"), a, nv),
        ])
    }
}

/// Convert a HF model (local directory or `owner/name` repo id) to a `.cmf`
/// file. `progress` receives fraction 0..1 (streamed as `@PROGRESS` markers).
pub fn run_convert(
    model: &str,
    quant: &str,
    output: &str,
    hf_token: Option<&str>,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    let quant = parse_quant(quant)?;

    // Source: a local HF directory, or an HF repo id to download.
    let downloaded;
    let dir: &Path = if Path::new(model).join("config.json").exists() {
        Path::new(model)
    } else if looks_like_repo(model) {
        eprintln!("downloading {model} from Hugging Face…");
        downloaded = hf_download(model, hf_token)?;
        downloaded.as_path()
    } else {
        anyhow::bail!("'{model}': not a local model dir (no config.json) and not an HF repo id (owner/name)");
    };

    let config: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("read config.json: {e}"))?)?;
    let arch = build_arch(&config)?;

    // Memory-map the weights and process one tensor at a time — the raw model is
    // never fully loaded into RAM (peak ≈ the .cmf output + one tensor).
    let files = open_model(dir)?;
    let total: usize = files.iter().map(|f| f.tensors.len()).sum::<usize>().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    let mut done = 0usize;
    for file in &files {
        for m in &file.tensors {
            done += 1;
            progress(done as f32 / total as f32);
            let Some(name) = canon_name(&m.name) else { continue };
            // qwen3_next / AgentWorld fuse the GDN projections (in_proj_qkvz /
            // in_proj_ba) with a group-interleaved layout; split them natively
            // into the canonical hub tensors (in_proj_qkv/z/a/b). Pure row
            // permutation — no value is changed.
            if name.contains(".linear_attn.in_proj_qkvz") || name.contains(".linear_attn.in_proj_ba") {
                if m.shape.len() != 2 {
                    anyhow::bail!("fused GDN tensor '{name}': expected 2-D, got {:?}", m.shape);
                }
                let w = to_f32(&m.dtype, file.bytes(m))?;
                let hid = m.shape[1];
                let miss = |k: &str| anyhow::anyhow!("fused GDN needs {k} in config");
                let nk = arch.linear_num_key_heads.ok_or_else(|| miss("linear_num_key_heads"))?;
                let dk = arch.linear_key_head_dim.ok_or_else(|| miss("linear_key_head_dim"))?;
                let nv = arch.linear_num_value_heads.ok_or_else(|| miss("linear_num_value_heads"))?;
                let dv = arch.linear_value_head_dim.ok_or_else(|| miss("linear_value_head_dim"))?;
                for (out_name, out_vals, out_rows) in split_fused_gdn(&name, &w, hid, nk, dk, nv, dv)? {
                    let two_d = out_rows * hid >= GROUP_SIZE && !force_f16(&out_name);
                    let (dt, data) = if two_d {
                        quantize_2d(quant, &out_vals, out_rows, hid)
                    } else {
                        (TensorDtype::F16, encode_f16(&out_vals))
                    };
                    tensors.push(TensorSpec { name: out_name, dtype: dt, shape: vec![out_rows, hid], data });
                }
                continue;
            }
            let vals = to_f32(&m.dtype, file.bytes(m))?;
            let numel: usize = m.shape.iter().product();
            if numel != vals.len() {
                anyhow::bail!("tensor '{name}': {} values for shape {:?}", vals.len(), m.shape);
            }
            // 1-D tensors, tiny tensors, non-2-D, and gate-critical projections go f16.
            let two_d = m.shape.len() == 2 && numel >= GROUP_SIZE && !force_f16(&name);
            let (dt, data) = if two_d {
                quantize_2d(quant, &vals, m.shape[0], m.shape[1])
            } else {
                (TensorDtype::F16, encode_f16(&vals))
            };
            tensors.push(TensorSpec { name, dtype: dt, shape: m.shape.clone(), data });
        }
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
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
        Quant::Vbit => QuantType::Vbit,
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type,
        provenance: Some(serde_json::json!({ "tool": "cortiq convert", "source_model": model })),
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

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::quant::{dequant_q4_block, dequant_q8_2f, dequant_q8_row, dequant_vbit};

    #[test]
    fn vbit_roundtrip_within_quant_error() {
        // rows with distinct amplitudes → distinct bit-widths; 2 groups per row.
        let (rows, cols) = (5usize, 64usize);
        let mut vals = vec![0f32; rows * cols];
        for o in 0..rows {
            for i in 0..cols {
                vals[o * cols + i] = (o as f32 + 1.0) * 0.13 * (i as f32 * 0.27).sin();
            }
        }
        let enc = encode_vbit(&vals, rows, cols);
        // header sizes match the decoder's expectation.
        let bits = &enc[..rows];
        assert!(bits.iter().all(|&b| (3..=8).contains(&b)));
        let mut dec = vec![0f32; rows * cols];
        dequant_vbit(&enc, rows, cols, &mut dec).unwrap();
        for o in 0..rows {
            let amp = vals[o * cols..(o + 1) * cols].iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            for i in 0..cols {
                let e = (dec[o * cols + i] - vals[o * cols + i]).abs();
                assert!(e <= amp * 0.2, "row {o} col {i}: err {e} vs amp {amp} (bits {})", bits[o]);
            }
        }
    }

    #[test]
    fn fused_gdn_split_is_correct_permutation() {
        // nk=2, dk=3, nv=4 (r=2), dv=2, hid=1. Each source row's value = its flat
        // row index, so we can trace exactly where each row lands after the split.
        let (nk, dk, nv, dv, hid) = (2usize, 3usize, 4usize, 2usize, 1usize);
        let r = nv / nk; // 2
        let gw = 2 * dk + 2 * r * dv; // 6 + 8 = 14
        let w: Vec<f32> = (0..nk * gw * hid).map(|i| i as f32).collect();
        let out = split_fused_gdn("m.linear_attn.in_proj_qkvz.weight", &w, hid, nk, dk, nv, dv).unwrap();
        let qkv = &out[0];
        assert_eq!(qkv.0, "m.linear_attn.in_proj_qkv.weight");
        assert_eq!(qkv.2, 2 * nk * dk + nv * dv); // 12 + 8 = 20
        // q rows: group g row rr -> source flat row g*gw+rr.
        // g=0 -> rows 0,1,2 ; g=1 -> rows 14,15,16
        assert_eq!(qkv.1[0..3], [0.0, 1.0, 2.0]);
        assert_eq!(qkv.1[3..6], [14.0, 15.0, 16.0]);
        // k rows: source g*gw + dk+rr. g=0 -> 3,4,5 ; g=1 -> 17,18,19
        assert_eq!(qkv.1[6..9], [3.0, 4.0, 5.0]);
        assert_eq!(qkv.1[9..12], [17.0, 18.0, 19.0]);
        // v rows: source g*gw + 2dk+rr (rr 0..4). g=0 -> 6,7,8,9 ; g=1 -> 20,21,22,23
        assert_eq!(qkv.1[12..16], [6.0, 7.0, 8.0, 9.0]);
        assert_eq!(qkv.1[16..20], [20.0, 21.0, 22.0, 23.0]);
        let z = &out[1];
        assert_eq!(z.0, "m.linear_attn.in_proj_z.weight");
        assert_eq!(z.2, nv * dv); // 8
        // z rows: source g*gw + 2dk+r*dv+rr. g=0 -> 10,11,12,13 ; g=1 -> 24,25,26,27
        assert_eq!(z.1, [10.0, 11.0, 12.0, 13.0, 24.0, 25.0, 26.0, 27.0]);

        // in_proj_ba: group width 2r=4. rows nk*4 = 8.
        let wb: Vec<f32> = (0..nk * 2 * r * hid).map(|i| i as f32).collect();
        let outb = split_fused_gdn("m.linear_attn.in_proj_ba.weight", &wb, hid, nk, dk, nv, dv).unwrap();
        // b = first r per group: g=0 -> 0,1 ; g=1 -> 4,5
        assert_eq!(outb[0].0, "m.linear_attn.in_proj_b.weight");
        assert_eq!(outb[0].1, [0.0, 1.0, 4.0, 5.0]);
        // a = next r per group: g=0 -> 2,3 ; g=1 -> 6,7
        assert_eq!(outb[1].0, "m.linear_attn.in_proj_a.weight");
        assert_eq!(outb[1].1, [2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn parse_quant_variants() {
        for q in ["q8", "q8_row", "q8_2f", "q4", "q4_block", "f16"] {
            assert!(parse_quant(q).is_ok(), "{q}");
        }
        assert!(parse_quant("nope").is_err());
    }

    #[test]
    fn q8_row_roundtrips() {
        let (o, i) = (4usize, 64usize);
        let vals: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.017).sin() * 2.5).collect();
        let bytes = encode_q8_row(&vals, o, i);
        assert_eq!(bytes.len(), o * i + o * 2);
        let mut back = vec![0f32; o * i];
        dequant_q8_row(&bytes, o, i, &mut back);
        for (a, b) in vals.iter().zip(&back) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn q8_2f_roundtrips() {
        let (o, i) = (8usize, 48usize);
        let vals: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.023).cos() * 1.7).collect();
        let bytes = encode_q8_2f(&vals, o, i);
        assert_eq!(bytes.len(), o * i + o * 2 + i * 2);
        let mut back = vec![0f32; o * i];
        dequant_q8_2f(&bytes, o, i, &mut back);
        for (a, b) in vals.iter().zip(&back) {
            assert!((a - b).abs() < 0.1, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_block_roundtrips() {
        let vals: Vec<f32> = (0..128).map(|k| (k as f32 * 0.05).sin()).collect();
        let bytes = encode_q4_block(&vals);
        let mut back = vec![0f32; 128];
        dequant_q4_block(&bytes, &mut back);
        for (a, b) in vals.iter().zip(&back) {
            assert!((a - b).abs() < 0.2, "{a} vs {b}");
        }
    }

    /// A raw safetensors blob from F32 tensors, for the end-to-end test.
    fn tiny_safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, vals) in tensors {
            let start = data.len();
            for &v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start, data.len()]}),
            );
        }
        let hjson = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hjson);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn convert_tiny_model_end_to_end() {
        let dir = std::env::temp_dir().join(format!("cortiq-convtest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"intermediate_size":128,"vocab_size":32,"rms_norm_eps":0.000001,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        let st = tiny_safetensors(&[
            ("model.embed_tokens.weight", vec![32, 64], (0..32 * 64).map(|k| (k as f32 * 0.01).sin()).collect()),
            ("model.norm.weight", vec![64], vec![1.0f32; 64]),
        ]);
        fs::write(dir.join("model.safetensors"), &st).unwrap();
        let out = dir.join("m.cmf");
        run_convert(dir.to_str().unwrap(), "q8", out.to_str().unwrap(), None, |_| {}).unwrap();

        let model = CmfModel::open(&out).unwrap();
        assert_eq!(model.arch().vocab_size, 32);
        assert_eq!(model.arch().num_layers, 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
