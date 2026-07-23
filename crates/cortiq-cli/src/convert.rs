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
//! attention in the Qwen3.5 hub layout (separate in_proj_qkv/z/a/b) and the
//! fused qwen3_next / AgentWorld layout, whose group-interleaved `in_proj_qkvz`
//! / `in_proj_ba` projections are split natively (`split_fused_gdn`).
//!
//! Not in scope: per-skill delta tensors and task masks — this writes backbones.
//! Those come from the DTG-MA path in `converter/`.

use crate::npy;
use cortiq_core::format::{CMF_VERSION, CmfHeader, CmfModel, TensorSpec, TokenizerBundle};
use cortiq_core::quant::{bf16_to_f32, f16_to_f32, f32_to_f16};
use cortiq_core::types::{LayerType, LinearCoreConfig, ModelArch, MoeConfig, NormStyle, QuantType, TensorDtype, YarnConfig};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

const GROUP_SIZE: usize = 32;
/// Smallest normal f16 — floor for degenerate (all-zero) rows so the stored
/// scale never underflows to a subnormal the reader would read back as 0.
const F16_TINY: f32 = 6.103_515_6e-5;

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
pub(crate) fn canon_name(raw: &str) -> Option<String> {
    if raw.contains(".visual.") || raw.starts_with("visual.") || raw.starts_with("mtp.") || raw.contains(".mtp.") {
        return None;
    }
    // Gemma-4 multimodal towers (text tower converts alone).
    for pfx in ["model.vision_embedder.", "model.embed_audio.", "model.embed_vision."] {
        if raw.starts_with(pfx) {
            return None;
        }
    }
    for pfx in ["model.language_model.", "language_model.model.", "language_model."] {
        if let Some(rest) = raw.strip_prefix(pfx) {
            return Some(lfm2_canon(&format!("model.{rest}")));
        }
    }
    // Laguna stores the router's auxiliary-loss-free selection bias under
    // `experts`, although it belongs to the router mathematically. CMF keeps
    // the canonical bias beside `mlp.gate.weight`.
    if raw.ends_with(".mlp.experts.e_score_correction_bias") {
        return Some(raw.replace(".mlp.experts.e_score_correction_bias", ".mlp.expert_bias"));
    }
    Some(lfm2_canon(raw))
}

/// Map LFM2 / LFM2-MoE vendor tensor names onto CMF's canonical (Qwen2)
/// layout so the standard loader reads them unchanged. Every substring
/// below is LFM2-exclusive among the supported architectures, so the
/// rewrite never touches another model's tensors. Returns the name
/// verbatim for non-LFM2 checkpoints.
///
///   operator_norm → input_layernorm      ffn_norm → post_attention_layernorm
///   embedding_norm → norm                 self_attn.out_proj → self_attn.o_proj
///   self_attn.{q,k}_layernorm → {q,k}_norm
///   conv.{in_proj,conv,out_proj} → short_conv.*
///   feed_forward.gate/expert_bias/experts.N → mlp.*
///   feed_forward.w1/w3/w2 → mlp.{gate,up,down}_proj (dense + per expert)
fn lfm2_canon(name: &str) -> String {
    let is_lfm2 =
        name == "model.embedding_norm.weight" || name.contains(".operator_norm") || name.contains(".ffn_norm") || name.contains(".feed_forward.") || name.contains(".conv.") || name.contains(".self_attn.out_proj") || name.contains(".self_attn.q_layernorm") || name.contains(".self_attn.k_layernorm");
    if !is_lfm2 {
        return name.to_string();
    }
    if name == "model.embedding_norm.weight" {
        return "model.norm.weight".to_string();
    }
    let mut n = name.to_string();
    n = n.replace(".operator_norm.", ".input_layernorm.");
    n = n.replace(".ffn_norm.", ".post_attention_layernorm.");
    n = n.replace(".self_attn.out_proj.", ".self_attn.o_proj.");
    n = n.replace(".self_attn.q_layernorm.", ".self_attn.q_norm.");
    n = n.replace(".self_attn.k_layernorm.", ".self_attn.k_norm.");
    n = n.replace(".conv.in_proj.", ".short_conv.in_proj.");
    n = n.replace(".conv.out_proj.", ".short_conv.out_proj.");
    n = n.replace(".conv.conv.", ".short_conv.conv.");
    // FFN: router/bias/experts first, then the dense fallback, then the
    // w1/w3/w2 → gate/up/down rename (applies to both mlp.wK and
    // mlp.experts.N.wK). Order matters — the experts substring carries
    // `.feed_forward.` so it must run before the bare `.feed_forward.`.
    n = n.replace(".feed_forward.gate.weight", ".mlp.gate.weight");
    n = n.replace(".feed_forward.expert_bias", ".mlp.expert_bias");
    n = n.replace(".feed_forward.experts.", ".mlp.experts.");
    n = n.replace(".feed_forward.", ".mlp.");
    n = n.replace(".w1.weight", ".gate_proj.weight");
    n = n.replace(".w3.weight", ".up_proj.weight");
    n = n.replace(".w2.weight", ".down_proj.weight");
    n
}

/// Small, noise-sensitive 2-D projections the reference converter keeps at f16
/// (a bit-flip there is costly): the GDN a/b gate projections and MoE routers.
fn force_f16(name: &str) -> bool {
    name.ends_with("linear_attn.in_proj_a.weight") || name.ends_with("linear_attn.in_proj_b.weight") || name.ends_with("mlp.gate.weight") || name.ends_with("shared_expert_gate.weight") || name.ends_with("self_attn.g_proj.weight")
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
    Q4Tiled,
    /// 1-bit binary (explicit opt-in): for 1-bit-TRAINED models
    /// (Bonsai / BitNet class), where per-group weights already sit on
    /// two levels ±s and the encoding is (near-)lossless. As PTQ of a
    /// normal checkpoint this destroys quality — never a default.
    Q1,
    /// 1-bit PTQ of a NORMAL checkpoint via error diffusion (перетекание):
    /// same on-disk `Q1` tile, but the encoder carries each weight's sign
    /// residual forward so the row sum survives binarization. Training-free;
    /// pair with `cortiq skill bake` (FCD) on the tail layers to recover
    /// quality. Bit-identical to `q1` on a genuinely 1-bit model.
    Q1p,
    /// 1-bit PTQ with an outlier mask (`Q1S` dtype): keeps the heavy tail
    /// (`CMF_Q1S_KEEP` of weights by |value|, default 1%) at full f16 in a
    /// sparse overlay, binarizes the rest with error diffusion. The mask
    /// lever of the holographic-transfer path — what lets a NORMAL
    /// checkpoint survive 1-bit.
    Q1s,
    Q1t,
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
        Quant::Q4Tiled if in_dim % GROUP_SIZE == 0 => (TensorDtype::Q4Tiled, encode_q4_tiled(vals, out_dim, in_dim)),
        Quant::Q4Tiled => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Vbit if in_dim % GROUP_SIZE == 0 => (TensorDtype::VbitRo, encode_vbit_ro(vals, out_dim, in_dim)),
        Quant::Vbit => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1 if in_dim % GROUP_SIZE == 0 => (TensorDtype::Q1, encode_q1(vals, out_dim, in_dim)),
        Quant::Q1 => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1p if in_dim % GROUP_SIZE == 0 => (TensorDtype::Q1, encode_q1_ef(vals, out_dim, in_dim)),
        Quant::Q1p => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1s if in_dim % GROUP_SIZE == 0 => (TensorDtype::Q1S, encode_q1s(vals, out_dim, in_dim, q1s_keep_frac())),
        Quant::Q1s => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1t if in_dim % GROUP_SIZE == 0 => (TensorDtype::Q1T, crate::gptq::quantize_q1t(vals, out_dim, in_dim, &vec![1.0; in_dim], 0.0)),
        Quant::Q1t => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
    }
}

pub(crate) fn parse_quant(s: &str) -> anyhow::Result<Quant> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q8" | "q8_row" | "q8row" => Quant::Q8Row,
        "q8_2f" | "q82f" | "q8f" => Quant::Q8_2f,
        "q4" | "q4_block" | "q4block" => Quant::Q4Block,
        "f16" | "fp16" => Quant::F16,
        "vbit" | "v_bit" => Quant::Vbit,
        "q4t" | "q4_tiled" => Quant::Q4Tiled,
        "q1" => Quant::Q1,
        "q1p" | "q1_ptq" => Quant::Q1p,
        "q1s" | "q1_mask" => Quant::Q1s,
        "q1t" | "q1_ternary" => Quant::Q1t,
        other => anyhow::bail!("unknown quant '{other}' (use q8, q8_2f, q4, q4t, f16, vbit, q1, q1p, q1s, or q1t)"),
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
/// q4_tiled (§4.3): the same 4-bit values/scales as q4_block, laid
/// out as one sequential stream of 18-byte tiles
/// `[f16 scale][16B nibbles]` per 32-group — measured x1.66 (ARM) /
/// x1.13 (AVX2) at kernel level over the split layout.
fn encode_q4_tiled(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let legacy = encode_q4_block(vals);
    let n_groups = vals.len() / GROUP_SIZE;
    let (packed, scales) = legacy.split_at(n_groups * 16);
    let mut out = Vec::with_capacity(n_groups * 18);
    for g in 0..n_groups {
        out.extend_from_slice(&scales[g * 2..g * 2 + 2]);
        out.extend_from_slice(&packed[g * 16..(g + 1) * 16]);
    }
    out
}

/// q1 (dtype 12): per 32-group tile `[f16 scale][4B sign bits]`,
/// bit k of byte j (LSB-first) = weight j·8+k; value = s·(2·bit−1).
/// Scale = group mean |v| — the L2-optimal binary level; for a
/// 1-bit-TRAINED model whose group weights already sit on ±s this
/// recovers the level exactly (encoding is lossless up to f16 range).
fn encode_q1(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let n_groups = vals.len() / GROUP_SIZE;
    let mut out = Vec::with_capacity(n_groups * 6);
    for g in 0..n_groups {
        let grp = &vals[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let mean_abs = grp.iter().map(|v| v.abs()).sum::<f32>() / GROUP_SIZE as f32;
        let s = f16_scale(mean_abs);
        out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        for j in 0..GROUP_SIZE / 8 {
            let mut byte = 0u8;
            for k in 0..8 {
                if grp[j * 8 + k] >= 0.0 {
                    byte |= 1 << k;
                }
            }
            out.push(byte);
        }
    }
    out
}

/// Error-diffusion ("перетекание") q1 encoder — the training-free PTQ path
/// for a NON-1-bit-trained model. Naïve q1 throws away every weight's
/// magnitude, keeping only its sign against a shared group scale; for a
/// normal checkpoint that is catastrophic. Here the per-weight rounding
/// residual `w − ŵ` is carried FORWARD along the row's input dimension and
/// folded into the next sign decision (`sign(w + carry)`), so the row's
/// running sum — hence its contribution to the dot product for the
/// slowly-varying part of the activation — is preserved rather than
/// discarded. Same on-disk `Q1` tile as `encode_q1` (reuses the kernel and
/// GPU path unchanged), and bit-identical to it on a genuinely 1-bit model
/// (near-constant |w| per group ⇒ `carry ≈ 0` ⇒ the sign never flips).
/// The carry resets at each row start (each output is an independent sum).
fn encode_q1_ef(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let groups_per_row = in_dim / GROUP_SIZE;
    let n_groups = vals.len() / GROUP_SIZE;
    let mut out = Vec::with_capacity(n_groups * 6);
    let mut carry = 0.0f32;
    for g in 0..n_groups {
        if g % groups_per_row == 0 {
            carry = 0.0; // new output row: its dot product starts fresh
        }
        let grp = &vals[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let mean_abs = grp.iter().map(|v| v.abs()).sum::<f32>() / GROUP_SIZE as f32;
        let s = f16_scale(mean_abs);
        out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        for j in 0..GROUP_SIZE / 8 {
            let mut byte = 0u8;
            for k in 0..8 {
                let w = grp[j * 8 + k];
                let v = w + carry;
                let bit = v >= 0.0;
                if bit {
                    byte |= 1 << k;
                }
                carry = v - if bit { s } else { -s };
            }
            out.push(byte);
        }
    }
    out
}

/// Fraction of weights the `q1s` mask keeps at full precision (the outlier
/// budget). `CMF_Q1S_KEEP` overrides; default 1%. Clamped to [0, 25%].
fn q1s_keep_frac() -> f32 {
    std::env::var("CMF_Q1S_KEEP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.01).clamp(0.0, 0.25)
}

/// 1-bit PTQ with an outlier mask (Stage 2a of the holographic-transfer
/// path). Keeps the top `keep_frac` of weights by |value| — the heavy tail
/// a normal checkpoint carries — at full f16 precision in a sparse overlay,
/// and binarizes the rest with the error-diffusion base, EXCLUDING the
/// outliers from each group's shared ±s scale (an outlier must not inflate
/// the level the bulk is quantized against). This is the |W| field of the
/// two-field mask; the activation field (`|W|·RMS(x)`) and the covariance
/// fold `Σ_PS·Σ_SS⁻¹` come from the calibration path on top of this.
fn encode_q1s(vals: &[f32], out_dim: usize, in_dim: usize, keep_frac: f32) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let n = vals.len();
    let n_out = (((n as f32) * keep_frac).round() as usize).min(n);
    // Outlier threshold via nth_element (O(n)): the (n − n_out)-th smallest
    // |w| — weights at or above it are the kept heavy tail.
    let threshold = if n_out == 0 {
        f32::INFINITY
    } else {
        let mut absv: Vec<f32> = vals.iter().map(|v| v.abs()).collect();
        let k = n - n_out;
        absv.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        absv[k]
    };
    let is_out: Vec<bool> = (0..n).map(|i| n_out > 0 && vals[i].abs() >= threshold).collect();

    let groups_per_row = in_dim / GROUP_SIZE;
    let n_groups = n / GROUP_SIZE;
    let n_out_actual = is_out.iter().filter(|&&o| o).count();
    let mut out = Vec::with_capacity(n_groups * 6 + 4 + n_out_actual * 6);
    let mut carry = 0.0f32;
    for g in 0..n_groups {
        if g % groups_per_row == 0 {
            carry = 0.0;
        }
        let base = g * GROUP_SIZE;
        // Scale = mean |w| over the NON-outlier weights of the group.
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for j in 0..GROUP_SIZE {
            if !is_out[base + j] {
                sum += vals[base + j].abs();
                cnt += 1;
            }
        }
        let s = f16_scale(if cnt > 0 { sum / cnt as f32 } else { 0.0 });
        out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        for jb in 0..GROUP_SIZE / 8 {
            let mut byte = 0u8;
            for k in 0..8 {
                let i = base + jb * 8 + k;
                if is_out[i] {
                    // Outlier: bit is only a sign hint (the overlay restores
                    // the exact value); it carries no error forward.
                    if vals[i] >= 0.0 {
                        byte |= 1 << k;
                    }
                } else {
                    let v = vals[i] + carry;
                    let bit = v >= 0.0;
                    if bit {
                        byte |= 1 << k;
                    }
                    carry = v - if bit { s } else { -s };
                }
            }
            out.push(byte);
        }
    }
    // Sparse outlier section: [u32 count][count × (u32 index, f16 value)].
    out.extend_from_slice(&(n_out_actual as u32).to_le_bytes());
    for (i, &o) in is_out.iter().enumerate() {
        if o {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            out.extend_from_slice(&f32_to_f16(vals[i]).to_le_bytes());
        }
    }
    out
}

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

/// `vbit_ro` (§4.2): the same bits/scales/packed encoding as
/// `encode_vbit`, plus `u32 row_offsets[rows+1]` (relative to the
/// packed area) between the scales and the packed rows — readers get
/// O(1) row access without a prefix scan. New dtype id; the byte
/// semantics of legacy `vbit` are untouched.
fn encode_vbit_ro(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let legacy = encode_vbit(vals, out_dim, in_dim);
    let ng = in_dim / GROUP_SIZE;
    let sc_len = out_dim * ng * 2;
    let (head, packed) = legacy.split_at(out_dim + sc_len);
    let bits = &head[..out_dim];
    let mut out = Vec::with_capacity(legacy.len() + (out_dim + 1) * 4);
    out.extend_from_slice(head);
    let mut off = 0u32;
    for &b in bits {
        out.extend_from_slice(&off.to_le_bytes());
        off += ((in_dim * b as usize).div_ceil(8)) as u32;
    }
    out.extend_from_slice(&off.to_le_bytes());
    debug_assert_eq!(off as usize, packed.len());
    out.extend_from_slice(packed);
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
pub(crate) fn to_f32(dtype: &str, raw: &[u8]) -> anyhow::Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        "BF16" => raw.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        other => anyhow::bail!("unsupported safetensors dtype '{other}' (need F32/F16/BF16)"),
    })
}

pub(crate) fn unpack_mlx(w_raw: &[u8], s_raw: &[u8], b_raw: Option<&[u8]>, out_dim: usize, in_dim: usize, bits: usize) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0f32; out_dim * in_dim];
    let num_groups = s_raw.len() / 2 / out_dim;
    let group_size = in_dim / num_groups;

    let w_u32: Vec<u32> = w_raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let s_f16: Vec<u16> = s_raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let b_f16: Option<Vec<u16>> = b_raw.map(|r| r.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect());

    let vals_per_u32 = 32 / bits;
    let mask = (1 << bits) - 1;

    for row in 0..out_dim {
        for col in 0..in_dim {
            let group = col / group_size;
            let scale = f16_to_f32(s_f16[row * num_groups + group]);
            let bias = b_f16.as_ref().map(|b| f16_to_f32(b[row * num_groups + group])).unwrap_or(0.0);

            let u32_idx = (row * in_dim + col) / vals_per_u32;
            let shift = (col % vals_per_u32) * bits;
            let val = (w_u32[u32_idx] >> shift) & mask;

            // For 1-bit, MLX might map 0->-1 and 1->1, but wait!
            // In 2-bit, MLX maps 0,1,2,3 directly to value * scale + bias.
            // If the model is 1-bit, is the value 0 and 1, or is it sign bits?
            // Actually, bias handles the shift. If it's a 1-bit scale+bias model, `val * scale + bias` works.
            out[row * in_dim + col] = (val as f32) * scale + bias;
        }
    }
    Ok(out)
}

/// Blob-layout sort key that puts tensors in decode-traversal order:
/// `(phase, layer, group, expert, projection)`. Phase orders embed → layers →
/// final-norm → lm_head → MTP → tail; within a layer, attention precedes the
/// FFN and MoE experts are grouped per expert (each expert's gate/up/down
/// contiguous). A stable name tiebreak keeps it deterministic. Layout only —
/// no effect on decoding (the directory is the offset authority).
pub(crate) fn exec_order_key(name: &str) -> (u32, u32, u32, u32, u32) {
    let num_after = |marker: &str| name.split(marker).nth(1).and_then(|s| s.split('.').next()).and_then(|s| s.parse::<u32>().ok());
    let expert = num_after(".experts.").unwrap_or(0);
    // Projection order within a block: q/gate, k/up, v/down, o, else.
    let proj = if name.contains("q_proj") || name.contains("gate_proj") {
        0
    } else if name.contains("k_proj") || name.contains("up_proj") {
        1
    } else if name.contains("v_proj") || name.contains("down_proj") {
        2
    } else if name.contains("o_proj") {
        3
    } else {
        4
    };
    if name.contains("embed_tokens") {
        (0, 0, 0, 0, 0)
    } else if let Some(l) = num_after(".layers.") {
        let group = if name.contains("input_layernorm") {
            0
        } else if name.contains("self_attn") || name.contains("linear_attn") || name.contains("short_conv") {
            1
        } else if name.contains("post_attention_layernorm") {
            2
        } else if name.ends_with("mlp.gate.weight") || name.contains("shared_expert") || name.contains("expert_bias") {
            3 // MoE router / shared expert (before the routed experts)
        } else if name.contains(".experts.") {
            4
        } else {
            5 // dense FFN (gate/up/down_proj) and anything else in the layer
        };
        (1, l, group, expert, proj)
    } else if name.contains("model.mtp") {
        (4, 0, 0, 0, 0)
    } else if name.contains("lm_head") {
        (3, 0, 0, 0, 0)
    } else if name.contains("model.norm") || name.ends_with("norm.weight") {
        (2, 0, 0, 0, 0)
    } else {
        (5, 0, 0, 0, 0)
    }
}

/// A tensor's metadata within a safetensors file (bytes are read lazily from mmap).
pub(crate) struct TensorMeta {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// A memory-mapped safetensors file — tensor bytes are borrowed from the mmap, so
/// the raw weights are never fully loaded into RAM (peak stays ~one tensor).
pub(crate) struct SafeTensors {
    mmap: memmap2::Mmap,
    data_start: usize,
    pub(crate) tensors: Vec<TensorMeta>,
}

impl SafeTensors {
    pub(crate) fn bytes(&self, m: &TensorMeta) -> &[u8] {
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
        let shape: Vec<usize> = v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow::anyhow!("tensor '{name}': no data_offsets"))?;
        let start = offs[0].as_u64().unwrap_or(0) as usize;
        let end = offs[1].as_u64().unwrap_or(0) as usize;
        tensors.push(TensorMeta { name: name.clone(), dtype, shape, start, end });
    }
    Ok(SafeTensors { mmap, data_start, tensors })
}

/// Memory-map a model dir's weights (single file or sharded index).
pub(crate) fn open_model(dir: &Path) -> anyhow::Result<Vec<SafeTensors>> {
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
            .map(|v| match v.as_str() {
                Some("linear_attention") => LayerType::LinearAttention,
                // LFM2 gated short convolution mixer.
                Some("conv") | Some("short_conv") => LayerType::ShortConv,
                Some("sliding_attention") => LayerType::SlidingAttention,
                _ => LayerType::FullAttention,
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
    // Qwen3.5 nests rope params under `rope_parameters`. Laguna goes one
    // level deeper and carries independent full/sliding profiles.
    let rope_root = tc.get("rope_parameters");
    let is_laguna_config = model_type.eq_ignore_ascii_case("laguna");
    let rope = if is_laguna_config { rope_root.and_then(|r| r.get("full_attention")) } else { rope_root };
    let local_rope = if is_laguna_config { rope_root.and_then(|r| r.get("sliding_attention")) } else { None };
    let rope_theta = tc.get("rope_theta").and_then(|v| v.as_f64()).or_else(|| rope.and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64())).unwrap_or(10_000.0);
    let prf = tc.get("partial_rotary_factor").and_then(|v| v.as_f64()).or_else(|| rope.and_then(|r| r.get("partial_rotary_factor")).and_then(|v| v.as_f64())).unwrap_or(1.0) as f32;
    let local_prf = local_rope.and_then(|r| r.get("partial_rotary_factor")).and_then(|v| v.as_f64()).map(|v| v as f32);
    let attention_heads_per_layer = tc
        .get("num_attention_heads_per_layer")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| anyhow::anyhow!("num_attention_heads_per_layer must contain integers"))).collect::<anyhow::Result<Vec<_>>>())
        .transpose()?;
    if let Some(heads) = &attention_heads_per_layer {
        anyhow::ensure!(heads.len() == n_layers, "num_attention_heads_per_layer has {} entries, expected {n_layers}", heads.len());
        let nkv = cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads);
        anyhow::ensure!(heads.iter().all(|&nh| nh > 0 && nh % nkv == 0), "every per-layer attention head count must be positive and divisible by num_key_value_heads={nkv}");
    }
    // Mixture-of-experts: the FFN becomes a router + per-expert matrices. Tensor
    // handling is unchanged (experts are ordinary 2-D matrices); we just declare
    // the MoE config so the runtime dispatches it. Router presence per layer
    // (in the directory) decides which layers are sparse.
    let moe = tc.get("num_experts").and_then(|v| v.as_u64()).filter(|&n| n > 0).map(|ne| {
        let mt = model_type.to_lowercase();
        let ntp_default = mt.starts_with("qwen3_5") || mt.contains("qwen3_next");
        // LFM2-MoE routes with a sigmoid gate + selection bias (DeepSeek-V3
        // noaux_tc); Qwen keeps the softmax-over-all default.
        let is_lfm2 = mt.starts_with("lfm2");
        let is_laguna = mt == "laguna";
        MoeConfig {
            num_experts: ne as usize,
            top_k: cfg_usize(tc, "num_experts_per_tok").unwrap_or(2),
            moe_intermediate_size: cfg_usize(tc, "moe_intermediate_size").unwrap_or(0),
            norm_topk_prob: tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(ntp_default),
            shared_expert_intermediate_size: cfg_usize(tc, "shared_expert_intermediate_size"),
            router_sigmoid: is_lfm2 || is_laguna,
            // A stored scale of 1.0 is the no-op default; only non-trivial
            // scales need to ride in the header.
            routed_scaling_factor: tc.get("routed_scaling_factor").or_else(|| tc.get("moe_routed_scaling_factor")).and_then(|v| v.as_f64()).map(|v| v as f32).filter(|&v| (v - 1.0).abs() > 1e-9),
        }
    });
    let head_dim = cfg_usize(tc, "head_dim").unwrap_or(hidden / n_heads.max(1));
    // Zero-centered RMSNorm x̂·(1+w): Gemma family and Qwen3.5 / Qwen3-Next.
    let mt = model_type.to_lowercase();
    let is_laguna = mt == "laguna";
    if is_laguna {
        anyhow::ensure!(!tc.get("swa_attention_sink_enabled").and_then(|v| v.as_bool()).unwrap_or(false), "laguna: learned SWA attention sinks are not supported yet");
        anyhow::ensure!(tc.get("moe_router_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) == 0.0, "laguna: non-zero MoE router logit soft-capping is not supported");
        anyhow::ensure!(!tc.get("moe_apply_router_weight_on_input").and_then(|v| v.as_bool()).unwrap_or(false), "laguna: moe_apply_router_weight_on_input=true is not supported");
    }
    let norm_style = if (mt.contains("gemma") && !mt.contains("gemma4")) || mt.starts_with("qwen3_5") || mt.contains("qwen3_next") {
        NormStyle::Gemma
    } else {
        // Gemma-4 went back to plain x̂·w (Gemma3nRMSNorm lineage).
        NormStyle::Qwen
    };
    // Gemma family: GeGLU FFN, √hidden embedding scale, an attention
    // scale of its own, and (Gemma-3) interleaved sliding-window layers
    // with a separate local RoPE base. Gemma-2's ATTENTION soft-capping
    // is not implemented — refuse it loudly rather than emit a wrong
    // file. (Gemma-4's FINAL-logit capping is supported.)
    let is_gemma = mt.contains("gemma");
    let is_gemma4 = mt.contains("gemma4");
    if tc.get("attn_logit_softcapping").and_then(|v| v.as_f64()).is_some() || (!is_gemma4 && tc.get("final_logit_softcapping").and_then(|v| v.as_f64()).is_some()) {
        anyhow::bail!(
            "{model_type}: attention logit soft-capping (Gemma-2) is not supported yet — \
             Gemma-1/Gemma-3/Gemma-4 convert natively"
        );
    }
    // Gemma-4 (text tower): plain x̂·w norms (unlike gemma-3), dual-geometry
    // attention (sliding GQA at head_dim + global MQA at global_head_dim
    // with proportional partial rotary), scale-less V-norm, per-layer
    // output scalars and final-logit capping. The dense 12B/31B variants
    // convert; the MoE / E-series machinery is refused honestly.
    if is_gemma4 {
        if tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false) {
            anyhow::bail!("{model_type}: gemma-4 MoE block (26B-A4B) is not supported yet");
        }
        if cfg_usize(tc, "hidden_size_per_layer_input").unwrap_or(0) > 0 {
            anyhow::bail!(
                "{model_type}: gemma-4 E-series per-layer inputs are not supported yet — \
                 the dense 12B/31B variants convert natively"
            );
        }
        if cfg_usize(tc, "num_kv_shared_layers").unwrap_or(0) > 0 {
            anyhow::bail!("{model_type}: gemma-4 KV-shared layers are not supported yet");
        }
    }
    // Gemma-4 keys rope_parameters by layer type: the global layers'
    // theta is the model theta, the sliding layers' theta is the local
    // base (same split gemma-3 spells with flat keys).
    let (g4_rope_theta, g4_local_theta, g4_global_prf) = match rope {
        Some(r) if is_gemma4 => {
            let full = r.get("full_attention");
            let slide = r.get("sliding_attention");
            (
                full.and_then(|f| f.get("rope_theta")).and_then(|v| v.as_f64()),
                slide.and_then(|f| f.get("rope_theta")).and_then(|v| v.as_f64()),
                full.and_then(|f| f.get("partial_rotary_factor")).and_then(|v| v.as_f64()).map(|v| v as f32),
            )
        }
        _ => (None, None, None),
    };
    let rope_theta = g4_rope_theta.unwrap_or(rope_theta);
    let yarn = rope
        .filter(|r| r.get("rope_type").and_then(|v| v.as_str()) == Some("yarn"))
        .map(|r| {
            Ok::<YarnConfig, anyhow::Error>(YarnConfig {
                factor: r.get("factor").and_then(|v| v.as_f64()).ok_or_else(|| anyhow::anyhow!("YaRN rope profile is missing factor"))? as f32,
                original_max_position_embeddings: r.get("original_max_position_embeddings").and_then(|v| v.as_u64()).ok_or_else(|| anyhow::anyhow!("YaRN rope profile is missing original_max_position_embeddings"))? as usize,
                beta_fast: r.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32,
                beta_slow: r.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                attention_factor: r.get("attention_factor").and_then(|v| v.as_f64()).unwrap_or_else(|| {
                    let factor = r.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    0.1 * factor.ln() + 1.0
                }) as f32,
            })
        })
        .transpose()?;
    // Sliding pattern from the explicit layer-type list: full layers must
    // sit at every P-th position ((i+1) % P == 0), which is how the
    // runtime models the cadence.
    let g4_pattern: Option<usize> = if is_gemma4 {
        let fulls: Vec<usize> = tc.get("layer_types").and_then(|v| v.as_array()).map(|a| a.iter().enumerate().filter(|(_, v)| v.as_str() == Some("full_attention")).map(|(i, _)| i).collect()).unwrap_or_default();
        let p = fulls.first().map(|f| f + 1).unwrap_or(0);
        if p == 0 || fulls.iter().enumerate().any(|(k, &i)| i != p * (k + 1) - 1) || (n_layers / p) != fulls.len() {
            anyhow::bail!("{model_type}: irregular full/sliding layer schedule not supported");
        }
        Some(p)
    } else {
        None
    };
    let hidden_act = match tc.get("hidden_activation").or_else(|| tc.get("hidden_act")).and_then(|v| v.as_str()).unwrap_or("silu") {
        "gelu_pytorch_tanh" | "gelu_tanh" | "gelu_new" => "gelu_tanh".to_string(),
        "silu" | "swish" => "silu".to_string(),
        other => anyhow::bail!("unsupported hidden_act '{other}'"),
    };
    let embed_multiplier = if is_gemma { (hidden as f32).sqrt() } else { 1.0 };
    // Phi-3 longrope: exact only within the ORIGINAL context — cap the
    // declared max honestly instead of serving stretched positions.
    let mut max_pos = cfg_usize(tc, "max_position_embeddings").unwrap_or(32768);
    if let Some(rs) = tc.get("rope_scaling").filter(|v| !v.is_null()) {
        let kind = rs.get("type").or_else(|| rs.get("rope_type")).and_then(|v| v.as_str());
        match kind {
            Some("longrope") | Some("su") | Some("yarn") | Some("linear") | Some("dynamic") | Some("mrope") => {
                let orig = cfg_usize(tc, "original_max_position_embeddings").unwrap_or(4096);
                eprintln!("  note: rope scaling '{:?}' — serving the exact {orig}-token native window", kind.unwrap());
                max_pos = orig;
            }
            Some(other) => anyhow::bail!("rope_scaling '{other}' not supported yet"),
            None => {}
        }
    }
    Ok(ModelArch {
        arch_name: model_type,
        hidden_size: hidden,
        intermediate_size: cfg_usize(tc, "intermediate_size").or_else(|| cfg_usize(tc, "moe_intermediate_size")).ok_or_else(|| anyhow::anyhow!("config: missing intermediate_size"))?,
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads),
        head_dim,
        vocab_size: cfg_usize(tc, "vocab_size").ok_or_else(|| anyhow::anyhow!("config: missing vocab_size"))?,
        layer_types,
        // LFM2 spells the RMSNorm epsilon `norm_eps`.
        rms_norm_eps: tc.get("rms_norm_eps").or_else(|| tc.get("norm_eps")).and_then(|v| v.as_f64()).unwrap_or(1e-6),
        norm_style,
        rope_theta,
        // Gemma ties embeddings by default and its configs omit the key.
        tie_word_embeddings: config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(is_gemma),
        partial_rotary_factor: prf,
        yarn,
        attention_heads_per_layer,
        mtp: None,
        moe,
        linear_core,
        max_position_embeddings: max_pos,
        // GDN spells it `linear_conv_kernel_dim`; LFM2 spells it `conv_L_cache`.
        linear_conv_kernel_dim: cfg_usize(tc, "linear_conv_kernel_dim").or_else(|| cfg_usize(tc, "conv_L_cache")),
        linear_num_key_heads: cfg_usize(tc, "linear_num_key_heads"),
        linear_num_value_heads: lnv,
        linear_key_head_dim: cfg_usize(tc, "linear_key_head_dim"),
        linear_value_head_dim: lvd,
        hidden_act,
        embed_multiplier,
        // Gemma-4 attends with scaling = 1.0 (q-norm carries the scale).
        query_pre_attn_scalar: tc.get("query_pre_attn_scalar").and_then(|v| v.as_f64()).or(if is_gemma4 { Some(1.0) } else { None }),
        sliding_window: cfg_usize(tc, "sliding_window").filter(|_| is_laguna || tc.get("sliding_window_pattern").is_some() || g4_pattern.is_some()),
        sliding_window_pattern: cfg_usize(tc, "sliding_window_pattern").or(g4_pattern),
        rope_local_base_freq: tc.get("rope_local_base_freq").and_then(|v| v.as_f64()).or(g4_local_theta).or_else(|| local_rope.and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64())),
        local_partial_rotary_factor: local_prf,
        global_head_dim: cfg_usize(tc, "global_head_dim").filter(|_| is_gemma4),
        num_global_kv_heads: cfg_usize(tc, "num_global_key_value_heads").filter(|_| is_gemma4),
        global_partial_rotary_factor: g4_global_prf,
        final_logit_softcapping: if is_gemma4 { tc.get("final_logit_softcapping").and_then(|v| v.as_f64()) } else { None },
        attn_v_norm: is_gemma4,
        // Looped Transformer (Nanbeige 4.2): re-apply the layer stack num_loops times.
        num_loops: cfg_usize(tc, "num_loops").unwrap_or(1),
        // skip_loop_final_norm=false means loop_final_norm=true (apply norm after each loop).
        loop_final_norm: !tc.get("skip_loop_final_norm").and_then(|v| v.as_bool()).unwrap_or(true),
    })
}

/// Collect eos ids from generation_config.json / config.json (int or array).
fn eos_ids(gen_cfg: &serde_json::Value, config: &serde_json::Value) -> Vec<u32> {
    for v in [gen_cfg.get("eos_token_id"), config.get("eos_token_id")].into_iter().flatten() {
        if let Some(n) = v.as_u64() {
            return vec![n as u32];
        }
        if let Some(a) = v.as_array() {
            return a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
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
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(300)).build()
}

/// List a repo's files via the HF API (best-effort; empty on failure). Reused by
/// the GGUF importer to pick a `.gguf` from a repo.
pub(crate) fn hf_repo_files(repo: &str, token: Option<&str>) -> Vec<String> {
    repo_files(&hf_agent(), repo, token)
}

/// Download a single named file from an HF repo into the cache (parallel chunks
/// for large files); returns its local path. Used to fetch one `.gguf`.
pub(crate) fn hf_fetch_file(repo: &str, filename: &str, token: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    let dir = hf_cache_dir(repo)?;
    let dest = dir.join(filename.replace('/', "__"));
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    fetch(&hf_agent(), &url, &dest, token, true, hf_threads())?;
    Ok(dest)
}

/// Local cache dir for a downloaded HF repo (`~/.cache/cortiq/hf/owner--name`).
fn hf_cache_dir(repo: &str) -> anyhow::Result<std::path::PathBuf> {
    let base = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache/cortiq/hf")).unwrap_or_else(|| std::path::PathBuf::from(".cortiq-hf"));
    let dir = base.join(repo.replace('/', "--"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Parallel range chunk size (32 MiB) and default connection count.
const HF_CHUNK: u64 = 32 * 1024 * 1024;

fn hf_threads() -> usize {
    std::env::var("CORTIQ_HF_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 1).unwrap_or(8).min(16)
}

fn cached(dest: &Path) -> bool {
    dest.exists() && fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false)
}

fn auth(mut req: ureq::Request, token: Option<&str>) -> ureq::Request {
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
    let resp = auth(agent.get(url).set("Range", &format!("bytes={}-{}", start, end - 1)), token).call().map_err(|e| anyhow::anyhow!("{e}"))?;
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
            let chunks: Vec<(u64, u64)> = (0..sz).step_by(HF_CHUNK as usize).map(|s| (s, (s + HF_CHUNK).min(sz))).collect();
            let total = chunks.len();
            let queue = Mutex::new(chunks);
            let err: Mutex<Option<String>> = Mutex::new(None);
            let done = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        loop {
                            if err.lock().unwrap().is_some() {
                                break;
                            }
                            let Some((start, end)) = queue.lock().unwrap().pop() else {
                                break;
                            };
                            // Each chunk retries on a transient failure before aborting.
                            let r = with_retry(4, || get_range(agent, url, token, start, end)).and_then(|buf| write_at(&tmp, start, &buf).map_err(Into::into));
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
            .and_then(|j| j["siblings"].as_array().map(|a| a.iter().filter_map(|s| s["rfilename"].as_str().map(String::from)).collect()))
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Fetch a HF repo's convertible files (config, tokenizer, weights) into the
/// cache, with parallel chunked downloads for the weight shards.
pub(crate) fn hf_download(repo: &str, token: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    let dir = hf_cache_dir(repo)?;
    let base = format!("https://huggingface.co/{repo}/resolve/main");
    let threads = hf_threads();
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(300)).build();
    // config.json is mandatory for the safetensors path. If it is absent, give an
    // actionable message rather than a raw 404 — most often the repo is a GGUF-only
    // distribution (has `*.gguf`, no `config.json`), which needs a different tool.
    if !fetch(&agent, &format!("{base}/config.json"), &dir.join("config.json"), token, false, threads)? {
        let files = repo_files(&agent, repo, token);
        let ggufs = files.iter().filter(|f| f.to_lowercase().ends_with(".gguf")).count();
        if ggufs > 0 {
            let src = repo.strip_suffix("-GGUF").or_else(|| repo.strip_suffix("-gguf")).filter(|s| !s.is_empty());
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
        // Newer HF checkpoints (LFM2, Qwen3, …) ship the chat template as a
        // sidecar `chat_template.jinja` instead of embedding it in
        // tokenizer_config.json — without it `run` falls back to a generic
        // ChatML default that does not match the model's real format.
        ("chat_template.jinja", false),
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
fn split_fused_gdn(name: &str, w: &[f32], hid: usize, nk: usize, dk: usize, nv: usize, dv: usize) -> anyhow::Result<Vec<(String, Vec<f32>, usize)>> {
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
        Ok(vec![(format!("{p}in_proj_qkv.weight"), qkv, 2 * nk * dk + nv * dv), (format!("{p}in_proj_z.weight"), z, nv * dv)])
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
        Ok(vec![(format!("{p}in_proj_b.weight"), b, nv), (format!("{p}in_proj_a.weight"), a, nv)])
    }
}

/// Convert a HF model (local directory or `owner/name` repo id) to a `.cmf`
/// file. `progress` receives fraction 0..1 (streamed as `@PROGRESS` markers).
// ───────────────────────── defrag (spec §11, Patent 2 claims 9/10) ─────────────────────────

enum FfnKind {
    Gate,
    Up,
    Down,
}

/// Match `model.layers.{li}.mlp.{gate|up|down}_proj.weight` → (layer, kind).
fn ffn_kind(name: &str) -> Option<(usize, FfnKind)> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let li: usize = rest[..dot].parse().ok()?;
    let kind = match &rest[dot + 1..] {
        "mlp.gate_proj.weight" => FfnKind::Gate,
        "mlp.up_proj.weight" => FfnKind::Up,
        "mlp.down_proj.weight" => FfnKind::Down,
        _ => return None,
    };
    Some((li, kind))
}

/// Drop dead neurons: gate/up keep ROWS (axis 0), down keeps COLUMNS
/// (axis 1). `keep` indexes the intermediate dim. Returns (reduced shape,
/// reduced f32 values).
fn slice_ffn(kind: &FfnKind, shape: &[usize], vals: &[f32], keep: &[bool]) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    let k = keep.iter().filter(|&&b| b).count();
    match kind {
        FfnKind::Gate | FfnKind::Up => {
            let (inter, hidden) = (shape[0], shape[1]);
            if keep.len() != inter {
                anyhow::bail!("defrag: keep len {} != gate/up rows {inter}", keep.len());
            }
            let mut out = Vec::with_capacity(k * hidden);
            for r in 0..inter {
                if keep[r] {
                    out.extend_from_slice(&vals[r * hidden..(r + 1) * hidden]);
                }
            }
            Ok((vec![k, hidden], out))
        }
        FfnKind::Down => {
            let (hidden, inter) = (shape[0], shape[1]);
            if keep.len() != inter {
                anyhow::bail!("defrag: keep len {} != down cols {inter}", keep.len());
            }
            let mut out = Vec::with_capacity(hidden * k);
            for r in 0..hidden {
                for c in 0..inter {
                    if keep[c] {
                        out.push(vals[r * inter + c]);
                    }
                }
            }
            Ok((vec![hidden, k], out))
        }
    }
}

/// Effective f32 for a canonical tensor: the baked overlay if present,
/// otherwise the backbone tensor from the safetensors files.
fn effective_tensor(overlay: &HashMap<String, (Vec<usize>, Vec<f32>)>, files: &[SafeTensors], name: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    if let Some((s, v)) = overlay.get(name) {
        return Ok((s.clone(), v.clone()));
    }
    for f in files {
        for m in &f.tensors {
            if canon_name(&m.name).as_deref() == Some(name) {
                return Ok((m.shape.clone(), to_f32(&m.dtype, f.bytes(m))?));
            }
        }
    }
    anyhow::bail!("defrag: tensor '{name}' not in overlay or base model")
}

struct DefragPlan {
    /// Baked FFN replacements (canonical name → shape + f32), overriding
    /// the backbone before pruning (carries FCD-retrained weights).
    overlay: HashMap<String, (Vec<usize>, Vec<f32>)>,
    /// Per-layer live-neuron mask over the intermediate dim.
    keep: HashMap<usize, Vec<bool>>,
}

/// Build the defrag plan from a skill dir: baked overlays (`tensors/*.npy`)
/// and a keep-set — explicit `ffn_keep.npy` if present, else autodetected
/// from zeroed down_proj columns (the Factory-Hard bake).
fn build_defrag_plan(dir: &Path, arch: &ModelArch, files: &[SafeTensors]) -> anyhow::Result<DefragPlan> {
    let mut overlay: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let tdir = dir.join("tensors");
    if tdir.is_dir() {
        for entry in fs::read_dir(&tdir)? {
            let p = entry?.path();
            if p.extension().and_then(|e| e.to_str()) != Some("npy") {
                continue;
            }
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
            let a = npy::read(&p)?;
            let vals = match a.data {
                npy::NpyData::F32(v) => v,
                npy::NpyData::Bool(_) => {
                    anyhow::bail!("defrag overlay {stem}: expected float, got bool")
                }
            };
            overlay.insert(stem, (a.shape, vals));
        }
    }
    println!("  Defrag overlay: {} baked tensors from {}", overlay.len(), dir.display());

    let (nl, inter) = (arch.num_layers, arch.intermediate_size);
    let mut keep: HashMap<usize, Vec<bool>> = HashMap::new();
    let keep_path = dir.join("ffn_keep.npy");
    if keep_path.exists() {
        let a = npy::read(&keep_path)?;
        if a.shape != [nl, inter] {
            anyhow::bail!("ffn_keep.npy shape {:?} != model ({nl}, {inter})", a.shape);
        }
        let flags: Vec<bool> = match a.data {
            npy::NpyData::Bool(v) => v,
            npy::NpyData::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
        };
        for li in 0..nl {
            let row = flags[li * inter..(li + 1) * inter].to_vec();
            if !row.iter().any(|&b| b) {
                anyhow::bail!("defrag: layer {li} has 0 live neurons");
            }
            keep.insert(li, row);
        }
    } else {
        // Producer-free: a neuron is dead iff its down_proj INPUT column is
        // all-zero (Factory-Hard bake). Reads each layer's effective down.
        println!("  Defrag: no ffn_keep.npy — autodetecting from zero down_proj columns");
        for li in 0..nl {
            let name = format!("model.layers.{li}.mlp.down_proj.weight");
            let (shape, vals) = effective_tensor(&overlay, files, &name)?;
            let (hidden, cols) = (shape[0], shape[1]);
            let mut alive = vec![false; cols];
            for r in 0..hidden {
                for c in 0..cols {
                    if vals[r * cols + c] != 0.0 {
                        alive[c] = true;
                    }
                }
            }
            if !alive.iter().any(|&b| b) {
                anyhow::bail!("defrag: layer {li} autodetected 0 live neurons");
            }
            keep.insert(li, alive);
        }
    }
    Ok(DefragPlan { overlay, keep })
}

pub fn run_convert(
    model: &str,
    quant: &str,
    output: &str,
    hf_token: Option<&str>,
    defrag: Option<&str>,
    // O(1) Nyström runtime hint (`--o1`): recorded in header provenance,
    // weights untouched — the runtime resolves it at load (loader.rs).
    o1_hint: Option<serde_json::Value>,
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

    let config: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("config.json")).map_err(|e| anyhow::anyhow!("read config.json: {e}"))?)?;
    let mut arch = build_arch(&config)?;

    // Memory-map the weights and process one tensor at a time — the raw model is
    // never fully loaded into RAM (peak ≈ the .cmf output + one tensor).
    let files = open_model(dir)?;

    // Physical defragmentation plan (spec §11): drop pruned FFN neurons so
    // they are neither stored nor computed. arch.intermediate_size becomes
    // nominal/max (per-layer truth lives in the reduced tensor shapes).
    let orig_inter = arch.intermediate_size;
    let defrag_plan = match defrag {
        Some(d) => Some(build_defrag_plan(Path::new(d), &arch, &files)?),
        None => None,
    };
    if let Some(plan) = &defrag_plan {
        let max_kept = (0..arch.num_layers).filter_map(|li| plan.keep.get(&li).map(|k| k.iter().filter(|&&b| b).count())).max().unwrap_or(orig_inter);
        arch.intermediate_size = max_kept;
    }
    let total: usize = files.iter().map(|f| f.tensors.len()).sum::<usize>().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    let mut done = 0usize;
    for file in &files {
        for m in &file.tensors {
            done += 1;
            progress(done as f32 / total as f32);
            let Some(name) = canon_name(&m.name) else {
                continue;
            };

            // Skip MLX scales and biases as they are processed with the weight.
            if m.dtype == "F16" && (name.ends_with(".scales") || name.ends_with(".biases")) {
                continue;
            }

            let (m_shape, m_vals) = if m.dtype == "U32" && m.name.ends_with(".weight") {
                let scales_name = m.name.replace(".weight", ".scales");
                let biases_name = m.name.replace(".weight", ".biases");
                let mut scales_blob = None;
                let mut biases_blob = None;
                for f in &files {
                    if let Some(t) = f.tensors.iter().find(|t| t.name == scales_name) {
                        scales_blob = Some(f.bytes(t));
                    }
                    if let Some(t) = f.tensors.iter().find(|t| t.name == biases_name) {
                        biases_blob = Some(f.bytes(t));
                    }
                }
                let scales = scales_blob.ok_or_else(|| anyhow::anyhow!("missing {} for MLX unpacking", scales_name))?;
                let out_dim = m.shape[0];
                let w_cols = m.shape[1];
                let num_groups = scales.len() / 2 / out_dim;

                let mut bits = 0;
                let mut in_dim = 0;
                for b in [1, 2, 3, 4, 8] {
                    let possible_in_dim = w_cols * 32 / b;
                    if possible_in_dim % num_groups == 0 {
                        let gs = possible_in_dim / num_groups;
                        if gs == 32 || gs == 64 || gs == 128 {
                            bits = b;
                            in_dim = possible_in_dim;
                            break;
                        }
                    }
                }
                if bits == 0 {
                    anyhow::bail!("Could not deduce MLX bit width for shape {:?} and {} scale groups", m.shape, num_groups);
                }
                (vec![out_dim, in_dim], unpack_mlx(file.bytes(m), scales, biases_blob, out_dim, in_dim, bits)?)
            } else {
                (m.shape.clone(), to_f32(&m.dtype, file.bytes(m))?)
            };

            // qwen3_next / AgentWorld fuse the GDN projections (in_proj_qkvz /
            // in_proj_ba) with a group-interleaved layout; split them natively
            // into the canonical hub tensors (in_proj_qkv/z/a/b). Pure row
            // permutation — no value is changed.
            if name.contains(".linear_attn.in_proj_qkvz") || name.contains(".linear_attn.in_proj_ba") {
                if m_shape.len() != 2 {
                    anyhow::bail!("fused GDN tensor '{name}': expected 2-D, got {:?}", m_shape);
                }
                let w = &m_vals;
                let hid = m_shape[1];
                let miss = |k: &str| anyhow::anyhow!("fused GDN needs {k} in config");
                let nk = arch.linear_num_key_heads.ok_or_else(|| miss("linear_num_key_heads"))?;
                let dk = arch.linear_key_head_dim.ok_or_else(|| miss("linear_key_head_dim"))?;
                let nv = arch.linear_num_value_heads.ok_or_else(|| miss("linear_num_value_heads"))?;
                let dv = arch.linear_value_head_dim.ok_or_else(|| miss("linear_value_head_dim"))?;
                for (out_name, out_vals, out_rows) in split_fused_gdn(&name, w, hid, nk, dk, nv, dv)? {
                    let two_d = out_rows * hid >= GROUP_SIZE && !force_f16(&out_name);
                    let (dt, data) = if two_d { quantize_2d(quant, &out_vals, out_rows, hid) } else { (TensorDtype::F16, encode_f16(&out_vals)) };
                    tensors.push(TensorSpec {
                        name: out_name,
                        dtype: dt,
                        shape: vec![out_rows, hid],
                        data,
                    });
                }
                continue;
            }
            // Phi-3 family fuses QKV (`qkv_proj`) and gate/up
            // (`gate_up_proj`): split into the canonical tensors — a
            // pure row slice, no value changes.
            if name.ends_with(".self_attn.qkv_proj.weight") || name.ends_with(".mlp.gate_up_proj.weight") {
                anyhow::ensure!(m_shape.len() == 2, "fused '{name}': expected 2-D");
                let w = &m_vals;
                let (rows, cols) = (m_shape[0], m_shape[1]);
                let parts: Vec<(String, usize, usize)> = if name.contains("qkv_proj") {
                    let q = arch.num_attention_heads * arch.head_dim;
                    let kv = arch.num_kv_heads * arch.head_dim;
                    anyhow::ensure!(q + 2 * kv == rows, "'{name}': {rows} rows != q({q}) + 2·kv({kv})");
                    vec![(name.replace("qkv_proj", "q_proj"), 0, q), (name.replace("qkv_proj", "k_proj"), q, kv), (name.replace("qkv_proj", "v_proj"), q + kv, kv)]
                } else {
                    anyhow::ensure!(rows % 2 == 0, "'{name}': odd row count {rows}");
                    vec![(name.replace("gate_up_proj", "gate_proj"), 0, rows / 2), (name.replace("gate_up_proj", "up_proj"), rows / 2, rows / 2)]
                };
                for (out_name, r0, nr) in parts {
                    let vals = &w[r0 * cols..(r0 + nr) * cols];
                    let (dt, data) = if nr * cols >= GROUP_SIZE && !force_f16(&out_name) { quantize_2d(quant, vals, nr, cols) } else { (TensorDtype::F16, encode_f16(vals)) };
                    tensors.push(TensorSpec { name: out_name, dtype: dt, shape: vec![nr, cols], data });
                }
                continue;
            }
            // Gemma-4 global (full-attention) layers carry no v_proj —
            // V is the K projection, normalized separately at runtime
            // (attention_k_eq_v). Materialize the duplicate so the
            // runtime keeps its uniform Q/K/V/O contract; the overlay
            // costs one MQA-sized tensor per global layer.
            if arch.global_head_dim.is_some() && name.ends_with(".self_attn.k_proj.weight") {
                let li: Option<usize> = name.split(".layers.").nth(1).and_then(|r| r.split('.').next()).and_then(|n| n.parse().ok());
                let pat = arch.sliding_window_pattern.unwrap_or(usize::MAX);
                if let Some(li) = li {
                    if (li + 1) % pat == 0 {
                        anyhow::ensure!(m_shape.len() == 2, "'{name}': expected 2-D");
                        let w = &m_vals;
                        let (rows, cols) = (m_shape[0], m_shape[1]);
                        for out_name in [name.clone(), name.replace("k_proj", "v_proj")] {
                            let (dt, data) = if rows * cols >= GROUP_SIZE && !force_f16(&out_name) { quantize_2d(quant, w, rows, cols) } else { (TensorDtype::F16, encode_f16(w)) };
                            tensors.push(TensorSpec {
                                name: out_name,
                                dtype: dt,
                                shape: vec![rows, cols],
                                data,
                            });
                        }
                        continue;
                    }
                }
            }
            // Defrag: for an FFN weight of a pruned layer, take the baked
            // overlay (if any) else the backbone value, drop dead neurons
            // (gate/up rows, down columns), then quantize the reduced shape.
            // The neuron never enters the blob — nor the runtime's math.
            if let Some(plan) = defrag_plan.as_ref() {
                if let Some((li, kind)) = ffn_kind(&name) {
                    if let Some(keep) = plan.keep.get(&li) {
                        let (shape, vals) = match plan.overlay.get(&name) {
                            Some((s, v)) => (s.clone(), v.clone()),
                            None => (m_shape.clone(), m_vals.clone()),
                        };
                        let (out_shape, out_vals) = slice_ffn(&kind, &shape, &vals, keep)?;
                        let numel = out_shape[0] * out_shape[1];
                        let two_d = numel >= GROUP_SIZE && !force_f16(&name);
                        let (dt, data) = if two_d { quantize_2d(quant, &out_vals, out_shape[0], out_shape[1]) } else { (TensorDtype::F16, encode_f16(&out_vals)) };
                        tensors.push(TensorSpec { name, dtype: dt, shape: out_shape, data });
                        continue;
                    }
                }
            }
            let vals = m_vals;
            let numel: usize = m_shape.iter().product();
            if numel != vals.len() {
                anyhow::bail!("tensor '{name}': {} values for shape {:?}", vals.len(), m_shape);
            }
            // 1-D tensors, tiny tensors, non-2-D, and gate-critical projections go f16.
            let two_d = m_shape.len() == 2 && numel >= GROUP_SIZE && !force_f16(&name);
            let (dt, data) = if two_d { quantize_2d(quant, &vals, m_shape[0], m_shape[1]) } else { (TensorDtype::F16, encode_f16(&vals)) };
            tensors.push(TensorSpec { name, dtype: dt, shape: m_shape.clone(), data });
        }
    }

    // Tokenizer + chat bundle (optional but recommended).
    let vocab = fs::read(dir.join("tokenizer.json")).ok();
    let tok_cfg: serde_json::Value = fs::read(dir.join("tokenizer_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::Value::Null);
    let gen_cfg: serde_json::Value = fs::read(dir.join("generation_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::Value::Null);
    // Sidecar `chat_template.jinja` first, then the tokenizer_config field;
    // ignore an empty/blank file so we correctly fall through to the config.
    let chat_template = fs::read_to_string(dir.join("chat_template.jinja")).ok().filter(|s| !s.trim().is_empty()).or_else(|| tok_cfg.get("chat_template").and_then(|v| v.as_str().map(String::from)));
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
        Quant::Q4Tiled => QuantType::Q4Block,
        // File-level label only (per-tensor truth is in the directory);
        // Vbit is the closest existing informational bucket for q1.
        Quant::Q1 | Quant::Q1p | Quant::Q1s | Quant::Q1t => QuantType::Vbit,
    };
    let provenance = match &defrag_plan {
        Some(plan) => {
            let kept: Vec<usize> = (0..arch.num_layers).map(|li| plan.keep.get(&li).map(|k| k.iter().filter(|&&b| b).count()).unwrap_or(orig_inter)).collect();
            let live: usize = kept.iter().sum();
            let ratio = 1.0 - live as f64 / (arch.num_layers as f64 * orig_inter as f64);
            eprintln!(
                "defrag: FFN pruned per-layer, {live}/{} live ({:.0}% pruned), inter {orig_inter} -> max {} (per-layer variable); masks dropped",
                arch.num_layers * orig_inter,
                ratio * 100.0,
                arch.intermediate_size
            );
            serde_json::json!({
                "tool": "cortiq convert",
                "source_model": model,
                "defrag": {
                    "source_skill": defrag,
                    "pre_intermediate": orig_inter,
                    "post_intermediate_max": arch.intermediate_size,
                    "kept_per_layer": kept,
                    "pruned_ratio": (ratio * 10000.0).round() / 10000.0,
                }
            })
        }
        None => serde_json::json!({ "tool": "cortiq convert", "source_model": model }),
    };
    let provenance = match o1_hint {
        Some(h) => {
            let mut p = provenance;
            p["o1_attn"] = h;
            p
        }
        None => provenance,
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type,
        provenance: Some(provenance),
        tokenizer_config: Some(bundle),
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };

    // Lay the blob out in EXECUTION order — embed, then each layer's tensors
    // contiguously (attention, then FFN, with MoE experts grouped per expert),
    // then final norm, lm_head, MTP, then any tail. HF safetensors are often
    // alphabetical (`layers.10` before `layers.2`), which scatters the decode
    // traversal across the file; sequential layer layout streams cold-start
    // reads at disk rate and lets a per-layer `madvise(WILLNEED)` cover one
    // contiguous range. Pure layout — the directory carries offsets, so the
    // reader (which addresses tensors by name/offset) is unaffected.
    tensors.sort_by(|a, b| exec_order_key(&a.name).cmp(&exec_order_key(&b.name)).then_with(|| a.name.cmp(&b.name)));

    CmfModel::write(output, &header, &tensors, None, vocab.as_deref()).map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    progress(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::quant::{dequant_q4_block, dequant_q8_2f, dequant_q8_row, dequant_vbit};

    #[test]
    fn laguna_config_maps_to_exact_cmf_contract() {
        let config = serde_json::json!({
            "model_type": "laguna",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_attention_heads_per_layer": [4, 6, 6, 6],
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 100,
            "max_position_embeddings": 1048576,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "norm_topk_prob": true,
            "moe_routed_scaling_factor": 2.5,
            "sliding_window": 512,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "yarn", "rope_theta": 500000.0,
                    "factor": 128.0, "original_max_position_embeddings": 8192,
                    "beta_fast": 32.0, "beta_slow": 1.0,
                    "attention_factor": 1.485203, "partial_rotary_factor": 0.5
                },
                "sliding_attention": {
                    "rope_type": "default", "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0
                }
            }
        });
        let arch = build_arch(&config).unwrap();
        assert_eq!(arch.arch_name, "laguna");
        assert_eq!(arch.attention_heads_per_layer, Some(vec![4, 6, 6, 6]));
        assert_eq!(arch.layer_types, vec![LayerType::FullAttention, LayerType::SlidingAttention, LayerType::SlidingAttention, LayerType::SlidingAttention,]);
        assert_eq!(arch.sliding_window, Some(512));
        assert_eq!(arch.rope_theta, 500000.0);
        assert_eq!(arch.rope_local_base_freq, Some(10000.0));
        assert_eq!(arch.partial_rotary_factor, 0.5);
        assert_eq!(arch.local_partial_rotary_factor, Some(1.0));
        let yarn = arch.yarn.unwrap();
        assert_eq!(yarn.factor, 128.0);
        let moe = arch.moe.unwrap();
        assert!(moe.router_sigmoid);
        assert_eq!(moe.routed_scaling_factor, Some(2.5));
        assert_eq!(canon_name("model.layers.1.mlp.experts.e_score_correction_bias").as_deref(), Some("model.layers.1.mlp.expert_bias"));
    }

    #[test]
    fn exec_order_lays_out_by_layer_then_block() {
        // Alphabetical order (the safetensors default) would put layer 10 before
        // layer 2 and the router after the experts — the exec-order key fixes both.
        let mut names: Vec<&str> = vec![
            "lm_head.weight",
            "model.layers.10.mlp.experts.1.up_proj.weight",
            "model.embed_tokens.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.norm.weight",
            "model.layers.2.mlp.gate.weight", // MoE router
            "model.layers.2.mlp.experts.0.down_proj.weight",
            "model.layers.2.input_layernorm.weight",
            "model.layers.2.mlp.experts.0.gate_proj.weight",
            "model.layers.10.self_attn.o_proj.weight",
        ];
        names.sort_by(|a, b| exec_order_key(a).cmp(&exec_order_key(b)).then_with(|| a.cmp(b)));
        assert_eq!(
            names,
            vec![
                "model.embed_tokens.weight",
                "model.layers.2.input_layernorm.weight",
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.2.mlp.gate.weight",
                "model.layers.2.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.0.down_proj.weight",
                "model.layers.10.self_attn.o_proj.weight",
                "model.layers.10.mlp.experts.1.up_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ]
        );
    }

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
    fn vbit_ro_roundtrip_and_validation() {
        use cortiq_core::TensorDtype;
        use cortiq_core::quant::{dequant_vbit_ro, validate_payload};
        let (rows, cols) = (5usize, 64usize);
        let mut vals = vec![0f32; rows * cols];
        for o in 0..rows {
            for i in 0..cols {
                vals[o * cols + i] = (o as f32 + 1.0) * 0.13 * (i as f32 * 0.27).sin();
            }
        }
        let enc = encode_vbit_ro(&vals, rows, cols);
        validate_payload(TensorDtype::VbitRo, &[rows, cols], &enc).unwrap();
        let mut dec = vec![0f32; rows * cols];
        dequant_vbit_ro(&enc, rows, cols, &mut dec).unwrap();
        // Must be BYTE-identical in reconstruction to the legacy layout.
        let legacy = encode_vbit(&vals, rows, cols);
        let mut dec_legacy = vec![0f32; rows * cols];
        dequant_vbit(&legacy, rows, cols, &mut dec_legacy).unwrap();
        assert_eq!(dec, dec_legacy, "vbit_ro must reconstruct exactly like vbit");
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
    fn lfm2_names_map_to_canonical_layout() {
        // Conv (dense) layer 0.
        let c = |s: &str| canon_name(s).unwrap();
        assert_eq!(c("model.embedding_norm.weight"), "model.norm.weight");
        assert_eq!(c("model.layers.0.operator_norm.weight"), "model.layers.0.input_layernorm.weight");
        assert_eq!(c("model.layers.0.ffn_norm.weight"), "model.layers.0.post_attention_layernorm.weight");
        assert_eq!(c("model.layers.0.conv.in_proj.weight"), "model.layers.0.short_conv.in_proj.weight");
        assert_eq!(c("model.layers.0.conv.conv.weight"), "model.layers.0.short_conv.conv.weight");
        assert_eq!(c("model.layers.0.conv.out_proj.weight"), "model.layers.0.short_conv.out_proj.weight");
        // Dense FFN: w1/w3/w2 → gate/up/down.
        assert_eq!(c("model.layers.0.feed_forward.w1.weight"), "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(c("model.layers.0.feed_forward.w3.weight"), "model.layers.0.mlp.up_proj.weight");
        assert_eq!(c("model.layers.0.feed_forward.w2.weight"), "model.layers.0.mlp.down_proj.weight");
        // Attention (full_attention layer 2): out_proj → o_proj, q/k layernorm.
        assert_eq!(c("model.layers.2.self_attn.out_proj.weight"), "model.layers.2.self_attn.o_proj.weight");
        assert_eq!(c("model.layers.2.self_attn.q_layernorm.weight"), "model.layers.2.self_attn.q_norm.weight");
        assert_eq!(c("model.layers.2.self_attn.k_layernorm.weight"), "model.layers.2.self_attn.k_norm.weight");
        // MoE router / bias / experts.
        assert_eq!(c("model.layers.2.feed_forward.gate.weight"), "model.layers.2.mlp.gate.weight");
        assert_eq!(c("model.layers.2.feed_forward.expert_bias"), "model.layers.2.mlp.expert_bias");
        assert_eq!(c("model.layers.2.feed_forward.experts.7.w1.weight"), "model.layers.2.mlp.experts.7.gate_proj.weight");
        assert_eq!(c("model.layers.2.feed_forward.experts.7.w2.weight"), "model.layers.2.mlp.experts.7.down_proj.weight");
        // Q/K/V projections already canonical — must pass through untouched.
        assert_eq!(c("model.layers.2.self_attn.q_proj.weight"), "model.layers.2.self_attn.q_proj.weight");
        // A Qwen tensor must be untouched by the LFM2 rewrite.
        assert_eq!(c("model.layers.3.mlp.gate_proj.weight"), "model.layers.3.mlp.gate_proj.weight");
    }

    #[test]
    fn lfm2_moe_arch_routing_and_layers() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":4,
                "num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":7168,
                "moe_intermediate_size":1792,"vocab_size":128000,"norm_eps":1e-5,
                "conv_L_cache":3,"num_experts":32,"num_experts_per_tok":4,
                "norm_topk_prob":true,"use_expert_bias":true,"routed_scaling_factor":1.0,
                "tie_word_embeddings":true,"rope_parameters":{"rope_theta":5000000},
                "layer_types":["conv","conv","full_attention","conv"]}"#,
        )
        .unwrap();
        let arch = build_arch(&cfg).unwrap();
        assert_eq!(arch.layer_types[0], LayerType::ShortConv);
        assert_eq!(arch.layer_types[2], LayerType::FullAttention);
        assert_eq!(arch.head_dim, 64);
        assert_eq!(arch.linear_conv_kernel_dim, Some(3));
        assert!((arch.rms_norm_eps - 1e-5).abs() < 1e-12);
        let moe = arch.moe.as_ref().unwrap();
        assert!(moe.router_sigmoid, "lfm2_moe must route with a sigmoid gate");
        assert_eq!(moe.top_k, 4);
        assert!(moe.norm_topk_prob);
        // Scale 1.0 stores as None (no-op).
        assert_eq!(moe.routed_scaling_factor, None);
    }

    /// The safety invariant that lets `q1p` be an unconditional replacement
    /// for `q1` on models that ARE 1-bit-trained: when every group weight
    /// already sits on ±s, the carry stays zero and no sign flips, so the
    /// encoder is bit-identical to the plain sign quantizer. (For a NORMAL
    /// checkpoint the two differ — that difference is the training-free PTQ,
    /// judged by end-to-end PPL, not by any single closed-form proxy.)
    #[test]
    fn q1_ef_bit_identical_on_a_1bit_tensor() {
        let (rows, cols) = (4usize, 96usize);
        let onebit: Vec<f32> = (0..rows * cols).map(|i| if (i * 7 + 3) % 5 < 2 { 0.25 } else { -0.25 }).collect();
        assert_eq!(encode_q1(&onebit, rows, cols), encode_q1_ef(&onebit, rows, cols), "error diffusion must be a no-op on a genuinely 1-bit tensor");
    }

    /// Q1S roundtrip: kept outliers come back at f16 precision and the bulk
    /// decodes to the per-group ±s level. Guards the format the holographic
    /// fold will populate.
    #[test]
    fn q1s_roundtrip_restores_outliers_and_binarizes_the_rest() {
        use cortiq_core::quant::dequant_q1s;
        let (rows, cols) = (2usize, 64usize);
        let mut vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.017).sin() * 0.1).collect();
        let spikes = [5usize, 40, 70, 120];
        for &i in &spikes {
            vals[i] = if i % 2 == 0 { 3.0 } else { -3.0 };
        }
        let keep = spikes.len() as f32 / (rows * cols) as f32;
        let bytes = encode_q1s(&vals, rows, cols, keep);
        let mut dec = vec![0f32; rows * cols];
        dequant_q1s(&bytes, &mut dec);
        for &i in &spikes {
            assert!((dec[i] - vals[i]).abs() < 0.02, "outlier {i}: {} vs {}", dec[i], vals[i]);
        }
        for i in 0..rows * cols {
            if !spikes.contains(&i) {
                assert!(dec[i].abs() < 2.0, "bulk {i} should be a small ±s, got {}", dec[i]);
            }
        }
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
            header.insert(name.to_string(), serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start, data.len()]}));
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
        let st = tiny_safetensors(&[("model.embed_tokens.weight", vec![32, 64], (0..32 * 64).map(|k| (k as f32 * 0.01).sin()).collect()), ("model.norm.weight", vec![64], vec![1.0f32; 64])]);
        fs::write(dir.join("model.safetensors"), &st).unwrap();
        let out = dir.join("m.cmf");
        run_convert(dir.to_str().unwrap(), "q8", out.to_str().unwrap(), None, None, None, |_| {}).unwrap();

        let model = CmfModel::open(&out).unwrap();
        assert_eq!(model.arch().vocab_size, 32);
        assert_eq!(model.arch().num_layers, 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
