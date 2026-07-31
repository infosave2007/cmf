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
use cortiq_core::format::{CMF_VERSION, CmfHeader, TensorSpec, TokenizerBundle};
use cortiq_core::quant::{
    Q2TP_CHUNK, Q2TP_LMAX, Q4TP_LMAX, Q4TP_NIB, bf16_to_f32, f16_to_f32, f32_to_f16,
    q2tp_ladder, q4tp_code_stride, q4tp_ladder, q4tp_put_code,
};
use cortiq_core::types::{
    LayerType, LinearCoreConfig, ModelArch, MoeConfig, NormStyle, QuantType, TensorDtype,
    YarnConfig,
};
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

/// tiktoken pre-tokenization pattern (Kimi family, o200k lineage) —
/// verbatim from tokenization_kimi.py; tiktoken itself compiles this
/// with fancy-regex, the same engine the runtime tokenizer uses.
const TIKTOKEN_KIMI_PAT: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// GPT-2 byte-level alphabet: raw byte → printable char (identity for
/// the printable ranges, 256+n in first-free order for the rest).
fn byte_level_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

/// Build an HF tokenizer.json from a tiktoken rank table (Kimi family).
///
/// tiktoken stores `base64(token_bytes) rank` per line and no merge
/// list — the merges ARE the ranks. Recovery (transformers' tiktoken
/// converter): BPE-split every multi-byte token using only merges of
/// strictly lower rank; the two parts it stops at are that token's
/// merge rule. Specials come from tokenizer_config.added_tokens_decoder.
fn tiktoken_to_tokenizer_json(
    model: &str,
    tok_cfg: &serde_json::Value,
) -> anyhow::Result<String> {
    use base64::Engine as _;
    use std::collections::HashMap;
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut ordered: Vec<(Vec<u8>, u32)> = Vec::new();
    for line in model.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (tok, rank) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow::anyhow!("tiktoken: bad line {line:?}"))?;
        let bytes = b64
            .decode(tok)
            .map_err(|e| anyhow::anyhow!("tiktoken: bad base64 {tok:?}: {e}"))?;
        let rank: u32 = rank.parse()?;
        ranks.insert(bytes.clone(), rank);
        ordered.push((bytes, rank));
    }
    anyhow::ensure!(!ordered.is_empty(), "tiktoken: empty rank table");
    ordered.sort_by_key(|(_, r)| *r);

    let table = byte_level_table();
    let to_bl = |bytes: &[u8]| -> String { bytes.iter().map(|&b| table[b as usize]).collect() };

    // Merge recovery: split `token` greedily by lowest-rank pair, using
    // only merges of rank < the token's own.
    let bpe_parts = |token: &[u8], max_rank: u32| -> Vec<Vec<u8>> {
        let mut parts: Vec<Vec<u8>> = token.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                let cat = [parts[i].as_slice(), parts[i + 1].as_slice()].concat();
                if let Some(&r) = ranks.get(cat.as_slice()) {
                    if r < max_rank && best.map(|(_, br)| r < br).unwrap_or(true) {
                        best = Some((i, r));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    let b = parts.remove(i + 1);
                    parts[i].extend(b);
                }
                None => break,
            }
        }
        parts
    };

    let mut vocab = serde_json::Map::new();
    let mut merges: Vec<serde_json::Value> = Vec::new();
    let mut skipped = 0usize;
    for (bytes, rank) in &ordered {
        vocab.insert(to_bl(bytes), serde_json::json!(rank));
        if bytes.len() > 1 {
            let parts = bpe_parts(bytes, *rank);
            if parts.len() == 2 {
                merges.push(serde_json::json!([to_bl(&parts[0]), to_bl(&parts[1])]));
            } else {
                // Unreachable-by-merges token (rare): encodable only as
                // a whole pre-token piece; keep it in the vocab.
                skipped += 1;
            }
        }
    }
    if skipped > 0 {
        tracing::warn!("tiktoken: {skipped} tokens yielded no merge rule (kept vocab-only)");
    }

    let mut added: Vec<serde_json::Value> = Vec::new();
    if let Some(atd) = tok_cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) {
        for (id, tok) in atd {
            if let (Ok(id), Some(content)) = (
                id.parse::<u32>(),
                tok.get("content").and_then(|c| c.as_str()),
            ) {
                added.push(serde_json::json!({
                    "id": id, "content": content, "special": true
                }));
            }
        }
    }

    let json = serde_json::json!({
        "version": "1.0",
        "added_tokens": added,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": { "Regex": TIKTOKEN_KIMI_PAT },
            "behavior": "Isolated",
            "invert": false
        },
        "decoder": { "type": "ByteLevel" },
        "model": {
            "type": "BPE",
            "vocab": serde_json::Value::Object(vocab),
            "merges": merges
        }
    });
    Ok(serde_json::to_string(&json)?)
}

/// Canonicalize a source tensor name to the CMF layout the runtime expects, or
/// `None` to skip it. Multimodal wrappers (Qwen3.5) nest the text model under
/// `model.language_model.*`; vision (`*.visual.*`) and the MTP head (`mtp.*`) are
/// dropped — plain greedy decoding is correct without MTP.
pub(crate) fn canon_name(raw: &str) -> Option<String> {
    if raw.contains(".visual.") || raw.starts_with("visual.") {
        return None;
    }
    // MTP head: kept, and renamed to the layout the loader reads. Qwen3.6
    // spells the projection `fc` and its two input norms
    // `pre_fc_norm_{embedding,hidden}`; the loader (written against the
    // DeepSeek spelling) wants `eh_proj`, `enorm`, `hnorm`. Everything
    // under `layers.0.` passes through — including the MoE mlp, which the
    // block carries here rather than a dense one.
    if let Some(rest) = raw.strip_prefix("mtp.").or_else(|| {
        raw.strip_prefix("model.mtp.")
            .or_else(|| raw.split_once(".mtp.").map(|(_, r)| r))
    }) {
        let mapped = match rest {
            "fc.weight" => "eh_proj.weight".to_string(),
            "pre_fc_norm_embedding.weight" => "enorm.weight".to_string(),
            "pre_fc_norm_hidden.weight" => "hnorm.weight".to_string(),
            other => other.to_string(),
        };
        return Some(format!("model.mtp.{mapped}"));
    }
    // Gemma-4 multimodal towers (text tower converts alone).
    for pfx in [
        "model.vision_embedder.",
        "model.embed_audio.",
        "model.embed_vision.",
        // Kimi-K3 vision tower + projector (text tower converts alone).
        "vision_tower.",
        "mm_projector.",
        // Gemma-3n multimodal towers.
        "model.audio_tower.",
        "model.vision_tower.",
    ] {
        if raw.starts_with(pfx) {
            return None;
        }
    }
    for pfx in [
        "model.language_model.",
        "language_model.model.",
        "language_model.",
    ] {
        if let Some(rest) = raw.strip_prefix(pfx) {
            // Re-enter with the wrapper stripped so every later rewrite
            // (Kimi block_sparse_moe, shared_experts, expert_bias) still
            // applies to nested checkpoints.
            return canon_name(&format!("model.{rest}"));
        }
    }
    // ── DeepSeek-V4 (`deepseek_v4`) ────────────────────────────────────
    // Its checkpoint drops the `model.` wrapper entirely and spells the
    // blocks `attn`/`ffn`, so the names arrive at top level: `embed`,
    // `head`, `norm`, `layers.N.attn.*`, `layers.N.ffn.*`. Everything
    // below is a rename into the layout the loader already reads; the
    // architecture-specific tensors (compressor, indexer, hyper-
    // connections) keep their spelling under the layer prefix so the
    // engine can find them once those nodes exist.
    if raw == "embed.weight" {
        return Some("model.embed_tokens.weight".into());
    }
    if raw == "head.weight" {
        return Some("lm_head.weight".into());
    }
    if raw == "norm.weight" {
        return Some("model.norm.weight".into());
    }
    // The hyper-connection head tensors are model-global (hc_head_base /
    // _fn / _scale) — pass them through under `model.`.
    if raw.starts_with("hc_head_") {
        return Some(format!("model.{raw}"));
    }
    if let Some(rest) = raw.strip_prefix("layers.") {
        // `layers.N.` + the rest; N is the layer index.
        if let Some((li, tail)) = rest.split_once('.') {
            if li.chars().all(|c| c.is_ascii_digit()) {
                let mapped = match tail {
                    "attn_norm.weight" => "input_layernorm.weight".to_string(),
                    "ffn_norm.weight" => "post_attention_layernorm.weight".to_string(),
                    // Router: weight + the noaux_tc selection bias, which
                    // the loader reads as `mlp.expert_bias`.
                    "ffn.gate.weight" => "mlp.gate.weight".to_string(),
                    "ffn.gate.bias" => "mlp.expert_bias".to_string(),
                    // Hash-routed layers carry a token-id → expert table
                    // instead of a learned gate.
                    "ffn.gate.tid2eid" => "mlp.tid2eid".to_string(),
                    other => {
                        // experts.E.w{1,3,2} → experts.E.{gate,up,down}_proj
                        // (DeepSeek's w1 = gate, w3 = up, w2 = down), and
                        // the same for the single shared expert.
                        let mut t = other
                            .replace("ffn.shared_experts.", "mlp.shared_expert.")
                            .replace("ffn.experts.", "mlp.experts.");
                        if t.starts_with("mlp.experts.") || t.starts_with("mlp.shared_expert.") {
                            t = t
                                .replace(".w1.weight", ".gate_proj.weight")
                                .replace(".w3.weight", ".up_proj.weight")
                                .replace(".w2.weight", ".down_proj.weight")
                                .replace(".w1.scale", ".gate_proj.scale")
                                .replace(".w3.scale", ".up_proj.scale")
                                .replace(".w2.scale", ".down_proj.scale");
                        }
                        // Attention keeps DeepSeek's own spelling: the
                        // double-LoRA q/o, the compressed KV, the sink,
                        // the compressor and the sparse indexer have no
                        // equivalent in any arch already supported, so
                        // renaming them would invent a layout nothing
                        // reads. `self_attn.` is the prefix every loader
                        // looks under.
                        t = t.replace("attn.", "self_attn.");
                        t
                    }
                };
                return Some(format!("model.layers.{li}.{mapped}"));
            }
        }
    }

    // Kimi (Kimi Linear / Kimi-K3) MoE block: mixtral-style w1/w3/w2
    // experts and a router with the noaux_tc selection bias.
    // `block_sparse_moe` is Kimi-exclusive among supported archs.
    if raw.contains(".block_sparse_moe.") {
        let mut n = raw
            .replace(
                ".block_sparse_moe.gate.e_score_correction_bias",
                ".mlp.expert_bias",
            )
            .replace(".block_sparse_moe.shared_experts.", ".mlp.shared_expert.")
            .replace(".block_sparse_moe.", ".mlp.");
        if n.contains(".mlp.experts.") || n.contains(".mlp.shared_expert.") {
            n = n
                .replace(".w1.weight", ".gate_proj.weight")
                .replace(".w3.weight", ".up_proj.weight")
                .replace(".w2.weight", ".down_proj.weight");
        }
        return Some(n);
    }
    // Laguna stores the router's auxiliary-loss-free selection bias under
    // `experts`, although it belongs to the router mathematically. CMF keeps
    // the canonical bias beside `mlp.gate.weight`.
    if raw.contains(".mlp.shared_experts.") {
        return Some(raw.replace(".mlp.shared_experts.", ".mlp.shared_expert."));
    }
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
    let is_lfm2 = name == "model.embedding_norm.weight"
        || name.contains(".operator_norm")
        || name.contains(".ffn_norm")
        || name.contains(".feed_forward.")
        || name.contains(".conv.")
        || name.contains(".self_attn.out_proj")
        || name.contains(".self_attn.q_layernorm")
        || name.contains(".self_attn.k_layernorm");
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
/// Tensors that must not be quantized OR narrowed: lookup tables whose
/// values are indices, not magnitudes. f16 is exact only to 2048, and
/// DeepSeek-V4's table holds expert ids per vocabulary id (129 280 rows).
fn force_f32(name: &str) -> bool {
    name.ends_with(".tid2eid")
}

fn force_f16(name: &str) -> bool {
    name.ends_with("linear_attn.in_proj_a.weight")
        || name.ends_with("linear_attn.in_proj_b.weight")
        // KDA (Kimi): decay/β/gate low-rank stages and conv taps are tiny
        // and sit on exp/σ paths — keep them exact.
        || name.ends_with("kda_attn.f_a_proj.weight")
        || name.ends_with("kda_attn.f_b_proj.weight")
        || name.ends_with("kda_attn.b_proj.weight")
        || name.ends_with("kda_attn.g_a_proj.weight")
        || name.ends_with("kda_attn.g_b_proj.weight")
        || name.ends_with("_conv1d.weight")
        // Gemma-3n: 4-wide AltUp coefficient mats and the low-rank /
        // per-layer-input projections sit on tanh/gelu gates.
        || name.ends_with("altup.correction_coefs.weight")
        || name.ends_with("altup.prediction_coefs.weight")
        || name.ends_with("altup.modality_router.weight")
        || name.ends_with("laurel.linear_left.weight")
        || name.ends_with("laurel.linear_right.weight")
        || name.ends_with("per_layer_input_gate.weight")
        || name.ends_with(".per_layer_projection.weight")
        || name.ends_with("mlp.gate.weight")
        || name.ends_with("shared_expert_gate.weight")
        || name.ends_with("self_attn.g_proj.weight")
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
    /// q4 tiled with PREDICTED scales: same nibbles, but the per-tile f16
    /// scale becomes a 5-bit rung on a per-row ladder — 7.3% off a q4t file
    /// for 1.14% RMS weight perturbation (q4t's own error vs fp16 is ~10%).
    Q4TiledP,
    /// 2-bit tiles on the q4tp ladder (Escha-W2 size class). As a whole-model
    /// profile the converter applies it to MoE `gate_up` experts only and
    /// keeps `down` experts and the skeleton at q4tp — the 2/4 split that
    /// mirrors Escha's 2/3-bit choice.
    Q2TiledP,
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
pub(crate) fn quantize_2d(
    quant: Quant,
    vals: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (TensorDtype, Vec<u8>) {
    match quant {
        Quant::Q8Row => (TensorDtype::Q8Row, encode_q8_row(vals, out_dim, in_dim)),
        Quant::Q8_2f => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q4Block => (TensorDtype::Q4Block, encode_q4_block(vals)),
        Quant::F16 => (TensorDtype::F16, encode_f16(vals)),
        // v-bit needs the input dim to be a multiple of the group size; other
        // shapes fall back to the two-field q8_2f (best equal-size alternative).
        Quant::Q4Tiled if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::Q4Tiled, encode_q4_tiled(vals, out_dim, in_dim))
        }
        Quant::Q4Tiled => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q4TiledP if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::Q4TiledP, encode_q4tp(vals, out_dim, in_dim))
        }
        Quant::Q4TiledP => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q2TiledP if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::Q2TiledP, encode_q2tp(vals, out_dim, in_dim))
        }
        Quant::Q2TiledP => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Vbit if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::VbitRo, encode_vbit_ro(vals, out_dim, in_dim))
        }
        Quant::Vbit => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1 if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::Q1, encode_q1(vals, out_dim, in_dim))
        }
        Quant::Q1 => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1p if in_dim % GROUP_SIZE == 0 => {
            (TensorDtype::Q1, encode_q1_ef(vals, out_dim, in_dim))
        }
        Quant::Q1p => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1s if in_dim % GROUP_SIZE == 0 => (
            TensorDtype::Q1S,
            encode_q1s(vals, out_dim, in_dim, q1s_keep_frac()),
        ),
        Quant::Q1s => (TensorDtype::Q8_2f, encode_q8_2f(vals, out_dim, in_dim)),
        Quant::Q1t if in_dim % GROUP_SIZE == 0 => (
            TensorDtype::Q1T,
            crate::gptq::quantize_q1t(vals, out_dim, in_dim, &vec![1.0; in_dim], 0.0),
        ),
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
        "q4tp" | "q4t_pred" => Quant::Q4TiledP,
        "q2tp" | "q2t_pred" => Quant::Q2TiledP,
        "q1" => Quant::Q1,
        "q1p" | "q1_ptq" => Quant::Q1p,
        "q1s" | "q1_mask" => Quant::Q1s,
        "q1t" | "q1_ternary" => Quant::Q1t,
        other => anyhow::bail!(
            "unknown quant '{other}' (use q8, q8_2f, q4, q4t, q4tp, q2tp, f16, vbit, q1, q1p, q1s, or q1t)"
        ),
    })
}

/// q8_row: `[int8 : out·in][f16 : out]` (validated layout, matches the reader).
pub(crate) fn encode_q8_row(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
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
            let q1 = ((group[k * 2 + 1] / scale)
                .round_ties_even()
                .clamp(-8.0, 7.0) as i8
                + 8) as u8;
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
pub(crate) fn encode_q4_tiled(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
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

/// q4tp (§4.10): the same nibbles as `q4_tiled`, but each tile's scale is a
/// 5-bit rung on a per-row geometric ladder — `[nibbles][row (f16 lo, f16
/// step)][5-bit codes]`, 4.17 bits/weight against 4.50.
///
/// Two details carry the accuracy. The row's `lo`/`step` are rounded to f16
/// FIRST and the codes are then chosen against those rounded values, so the
/// rounding of `lo` is absorbed by the code instead of stacking on top of it.
/// And the nibbles are quantized against the RECONSTRUCTED scale, not the
/// exact per-tile absmax — otherwise encoder and reader disagree, which is
/// the same trap `f16_scale` exists to avoid.
///
/// Quantizing the same fp32 weights both ways, q4tp's error against the source
/// is within 0.1% of q4t's at the median within-row scale spread measured on
/// KAT-Coder-V2.5 (1.27 in log2), 0.3% at its 90th percentile — a coarser scale
/// costs almost nothing because the nibbles re-round against it. Rounding the
/// code to nearest beats rounding up (1.19% vs 1.60% RMS against q4t's output):
/// letting the top tile clip a nibble costs less than coarsening the whole row.
/// `q2tp`: the `q4tp` ladder with a 2-bit weight plane (8 B per 32-group).
/// Levels are (c − 1.5)·s, so the group scale fits absmax/1.5. Built for
/// transcoding 2-bit-class checkpoints (Escha-W2) where the source already
/// paid the big quantization cost — and for MoE experts generally.
/// How many threads the row encoders use. `CMF_ENCODE_THREADS` overrides it,
/// which is also how the parity test pins a single thread.
pub(crate) fn encode_threads() -> usize {
    std::env::var("CMF_ENCODE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1)
        })
}

/// Run a per-row encoder across the machine's cores.
///
/// Every tiled layout here writes three planes indexed by row at a fixed
/// stride, and no row reads another, so splitting on rows is exact — the
/// bytes are identical to the serial loop, which the parity test pins.
///
/// It is worth the plumbing: quantizing a 300B MoE is hours of arithmetic,
/// and on a 48-core box the serial loop used one of them.
fn encode_rows_parallel(
    out_dim: usize,
    strides: [usize; 3],
    planes: [&mut [u8]; 3],
    row: &(dyn Fn(usize, &mut [u8], &mut [u8], &mut [u8]) + Sync),
) {
    let [s0, s1, s2] = strides;
    let [p0, p1, p2] = planes;
    let threads = encode_threads().min(out_dim.max(1)).max(1);
    if threads == 1 {
        for (i, ((a, b), c)) in p0
            .chunks_mut(s0)
            .zip(p1.chunks_mut(s1))
            .zip(p2.chunks_mut(s2))
            .enumerate()
        {
            row(i, a, b, c);
        }
        return;
    }
    let per = out_dim.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ti, ((a, b), c)) in p0
            .chunks_mut(per * s0)
            .zip(p1.chunks_mut(per * s1))
            .zip(p2.chunks_mut(per * s2))
            .enumerate()
        {
            sc.spawn(move || {
                let r0 = ti * per;
                for (i, ((aa, bb), cc)) in a
                    .chunks_mut(s0)
                    .zip(b.chunks_mut(s1))
                    .zip(c.chunks_mut(s2))
                    .enumerate()
                {
                    row(r0 + i, aa, bb, cc);
                }
            });
        }
    });
}

pub(crate) fn encode_q2tp(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let gpr = in_dim / GROUP_SIZE;
    let stride = q4tp_code_stride(gpr);
    let mut chunks = vec![0u8; out_dim * gpr * Q2TP_CHUNK];
    let mut params = vec![0u8; out_dim * 4];
    let mut codes = vec![0u8; out_dim * stride];

    encode_rows_parallel(
        out_dim,
        [gpr * Q2TP_CHUNK, 4, stride],
        [&mut chunks, &mut params, &mut codes],
        &|r, chunks_row, params_row, codes_row| {
        let mut lg = vec![0f32; gpr];
        let mut dead = vec![false; gpr];

        let row = &vals[r * in_dim..(r + 1) * in_dim];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for g in 0..gpr {
            let tile = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let absmax = tile.iter().fold(0f32, |m, v| m.max(v.abs()));
            dead[g] = absmax == 0.0;
            lg[g] = f16_scale(absmax / 1.5).log2();
            if !dead[g] {
                lo = lo.min(lg[g]);
                hi = hi.max(lg[g]);
            }
        }
        if !lo.is_finite() {
            lo = lg[0];
            hi = lo;
        }
        let lo_h = f32_to_f16(lo);
        let lo_r = f16_to_f32(lo_h);
        let span = (hi - lo_r).max(0.0);
        let mut st_h = f32_to_f16(span / Q2TP_LMAX as f32);
        for _ in 0..64 {
            let st = f16_to_f32(st_h);
            if st > 0.0 && lo_r + Q2TP_LMAX as f32 * st >= hi {
                break;
            }
            st_h += 1;
        }
        params_row[0..2].copy_from_slice(&lo_h.to_le_bytes());
        params_row[2..4].copy_from_slice(&st_h.to_le_bytes());

        let st = f16_to_f32(st_h);
        let tab = q2tp_ladder(params_row, 0);
        let crow = &mut *codes_row;
        for g in 0..gpr {
            // Rung 0 is the exact zero; live groups start at 1.
            let c = if dead[g] {
                0
            } else if st <= 0.0 {
                1
            } else {
                1 + ((lg[g] - lo_r) / st).round_ties_even().clamp(0.0, Q2TP_LMAX as f32) as usize
            };
            q4tp_put_code(crow, g, c);
            let inv = if tab[c] > 0.0 { 1.0 / tab[c] } else { 0.0 };
            let tile = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let dst = &mut chunks_row[g * Q2TP_CHUNK..(g + 1) * Q2TP_CHUNK];
            for (k, d) in dst.iter_mut().enumerate() {
                let mut b = 0u8;
                for j in 0..4 {
                    // (c − 1.5)·s: quantize w/s + 1.5 onto 0..=3.
                    let q = (tile[k * 4 + j] * inv + 1.5)
                        .round_ties_even()
                        .clamp(0.0, 3.0) as u8;
                    b |= q << (2 * j);
                }
                *d = b;
            }
        }
            },
    );
    chunks.extend_from_slice(&params);
    chunks.extend_from_slice(&codes);
    chunks
}

pub(crate) fn encode_q4tp(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    debug_assert_eq!(vals.len(), out_dim * in_dim);
    debug_assert_eq!(in_dim % GROUP_SIZE, 0);
    let gpr = in_dim / GROUP_SIZE;
    let stride = q4tp_code_stride(gpr);
    let mut nib = vec![0u8; out_dim * gpr * Q4TP_NIB];
    let mut params = vec![0u8; out_dim * 4];
    let mut codes = vec![0u8; out_dim * stride];

    encode_rows_parallel(
        out_dim,
        [gpr * Q4TP_NIB, 4, stride],
        [&mut nib, &mut params, &mut codes],
        &|r, nib_row, params_row, codes_row| {
        let mut lg = vec![0f32; gpr];
        let mut dead = vec![false; gpr];

        let row = &vals[r * in_dim..(r + 1) * in_dim];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for g in 0..gpr {
            let tile = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let absmax = tile.iter().fold(0f32, |m, v| m.max(v.abs()));
            // An all-zero tile dequantizes to zero whatever its scale, so keep
            // it out of the row's range — otherwise `f16_scale`'s tiny floor
            // stretches the ladder and coarsens every live tile in the row.
            dead[g] = absmax == 0.0;
            lg[g] = f16_scale(absmax / 7.0).log2();
            if !dead[g] {
                lo = lo.min(lg[g]);
                hi = hi.max(lg[g]);
            }
        }
        if !lo.is_finite() {
            lo = lg[0];
            hi = lo;
        }

        let lo_h = f32_to_f16(lo);
        let lo_r = f16_to_f32(lo_h);
        let span = (hi - lo_r).max(0.0);
        let mut st_h = f32_to_f16(span / Q4TP_LMAX as f32);
        // Round-to-nearest on `step` can land just short of `hi`; walk up by
        // single ULPs until rung 31 provably covers the row's top scale.
        for _ in 0..64 {
            let st = f16_to_f32(st_h);
            if st > 0.0 && lo_r + Q4TP_LMAX as f32 * st >= hi {
                break;
            }
            st_h += 1;
        }
        params_row[0..2].copy_from_slice(&lo_h.to_le_bytes());
        params_row[2..4].copy_from_slice(&st_h.to_le_bytes());

        let st = f16_to_f32(st_h);
        let tab = q4tp_ladder(params_row, 0);
        let crow = &mut *codes_row;
        for g in 0..gpr {
            let c = if dead[g] || st <= 0.0 {
                0
            } else {
                ((lg[g] - lo_r) / st).round_ties_even().clamp(0.0, Q4TP_LMAX as f32) as usize
            };
            q4tp_put_code(crow, g, c);
            let inv = if tab[c] > 0.0 { 1.0 / tab[c] } else { 0.0 };
            let tile = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let dst = &mut nib_row[g * Q4TP_NIB..(g + 1) * Q4TP_NIB];
            for k in 0..16 {
                let q0 = ((tile[k * 2] * inv).round_ties_even().clamp(-8.0, 7.0) as i8 + 8) as u8;
                let q1 =
                    ((tile[k * 2 + 1] * inv).round_ties_even().clamp(-8.0, 7.0) as i8 + 8) as u8;
                dst[k] = (q0 & 0x0F) | (q1 << 4);
            }
        }
            },
    );

    nib.extend_from_slice(&params);
    nib.extend_from_slice(&codes);
    nib
}

/// Re-encode f32 values into an EXISTING tensor's dtype — the offline passes
/// (AWNP) rewrite weights and must put them back in the layout the file
/// already uses, not pick a new one.
pub(crate) fn encode_for(dtype: TensorDtype, vals: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    match dtype {
        TensorDtype::Q4Tiled => encode_q4_tiled(vals, rows, cols),
        TensorDtype::Q4TiledP => encode_q4tp(vals, rows, cols),
        TensorDtype::Q4Block => encode_q4_block(vals),
        TensorDtype::Q8Row => encode_q8_row(vals, rows, cols),
        TensorDtype::Q8_2f => encode_q8_2f(vals, rows, cols),
        _ => encode_f16(vals),
    }
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
    std::env::var("CMF_Q1S_KEEP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.01)
        .clamp(0.0, 0.25)
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
    let is_out: Vec<bool> = (0..n)
        .map(|i| n_out > 0 && vals[i].abs() >= threshold)
        .collect();

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
            let mx = vals[o * in_dim..(o + 1) * in_dim]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            mx.max(1e-12).log2()
        })
        .collect();
    let amean = a.iter().sum::<f32>() / out_dim as f32;
    a.iter()
        .map(|&ar| vbit_snap_level(mean_bits + (ar - amean)).max(3))
        .collect()
}

/// Big-endian (MSB-first) bit packer; the last byte of each row is zero-padded.
struct BitWriter {
    buf: Vec<u8>,
    cur: u8,
    nbits: u8,
}
impl BitWriter {
    fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
            cur: 0,
            nbits: 0,
        }
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
            let mx = vals[base..base + GROUP_SIZE]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
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
/// Move every produced payload out of RAM and straight into the output file,
/// then drop it. Called after each source shard, so peak residency is one
/// shard's tensors rather than the model's — and because the writer streams
/// into the final file, peak DISK is the finished model rather than twice it.
fn drain_to_writer(
    tensors: &mut Vec<TensorSpec>,
    writer: &mut cortiq_core::format::CmfStreamWriter,
) -> anyhow::Result<()> {
    for t in tensors.drain(..) {
        writer
            .push(&t.name, t.dtype, &t.shape, &t.data)
            .map_err(|e| anyhow::anyhow!("write tensor '{}': {e}", t.name))?;
    }
    Ok(())
}

pub(crate) fn to_f32(dtype: &str, raw: &[u8]) -> anyhow::Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        // DeepSeek-V4's hash-routing table is I64 token→expert indices, and
        // I32 shows up in the same role elsewhere. They are LOOKUP TABLES,
        // not weights: widen to f32 so the pipeline can carry them, and let
        // the writer's force_f16 rule keep them exact.
        "I64" => raw
            .chunks_exact(8)
            .map(|b| {
                i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            })
            .collect(),
        "I32" => raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        other => anyhow::bail!("unsupported safetensors dtype '{other}' (need F32/F16/BF16)"),
    })
}

/// OCP MXFP4 (compressed-tensors "mxfp4-pack-quantized", Kimi-K3):
/// two FP4-E2M1 values per byte (LOW nibble = even element), groups of
/// 32 along the last axis, one E8M0 scale byte (2^(k−127)) per group.
pub(crate) fn unpack_mxfp4(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols_packed: usize,
) -> anyhow::Result<Vec<f32>> {
    // E2M1, bias 1: e=0 subnormal m·0.5; e≥1 normal (1+m/2)·2^(e−1).
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let cols = cols_packed * 2;
    anyhow::ensure!(cols % 32 == 0, "mxfp4: cols {cols} not a multiple of 32");
    let gpr = cols / 32;
    anyhow::ensure!(
        packed.len() == rows * cols_packed && scales.len() == rows * gpr,
        "mxfp4: packed {} / scales {} vs rows {rows} cols {cols}",
        packed.len(),
        scales.len()
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for g in 0..gpr {
            let k = scales[r * gpr + g];
            // E8M0: value = 2^(k−127); 255 = NaN per OCP — refuse loudly.
            anyhow::ensure!(k != 255, "mxfp4: NaN scale at row {r} group {g}");
            let scale = (k as f32 - 127.0).exp2();
            for b in 0..16 {
                let byte = packed[r * cols_packed + g * 16 + b];
                for (half, nib) in [(0usize, byte & 0x0F), (1usize, byte >> 4)] {
                    let mag = LUT[(nib & 0x7) as usize];
                    let v = if nib & 0x8 != 0 { -mag } else { mag };
                    out[r * cols + g * 32 + b * 2 + half] = v * scale;
                }
            }
        }
    }
    Ok(out)
}

/// OCP FP8 E4M3 → f32. 1 sign, 4 exp (bias 7), 3 mantissa; no infinities,
/// and exp=15,mant=7 is the only NaN. Subnormals are m·2^-9.
#[inline]
pub(crate) fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    let mag = if exp == 0 {
        man * (2f32).powi(-9)
    } else {
        (1.0 + man / 8.0) * (2f32).powi(exp - 7)
    };
    sign * mag
}

/// DeepSeek-V4 style FP8 weights: E4M3 values with one E8M0 scale per
/// `block`×`block` tile (`quantization_config.weight_block_size`). The
/// scale plane is `ceil(rows/block) × ceil(cols/block)`, row-major, and
/// the tiles at the edges are partial — the model's own layout, not a
/// padded one.
pub(crate) fn unpack_fp8_blocks(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block: usize,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(block > 0, "fp8 blocks: block size 0");
    let sr = rows.div_ceil(block);
    let sc = cols.div_ceil(block);
    anyhow::ensure!(
        packed.len() == rows * cols && scales.len() == sr * sc,
        "fp8 blocks: weight {} (want {}) / scales {} (want {}) for {rows}x{cols} block {block}",
        packed.len(),
        rows * cols,
        scales.len(),
        sr * sc
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let br = r / block;
        for c in 0..cols {
            let k = scales[br * sc + c / block];
            // E8M0: 2^(k−127); 255 is NaN per OCP — refuse loudly rather
            // than fold a NaN through the whole row.
            anyhow::ensure!(k != 255, "fp8 blocks: NaN scale at ({r},{c})");
            out[r * cols + c] = fp8_e4m3_to_f32(packed[r * cols + c]) * (k as f32 - 127.0).exp2();
        }
    }
    Ok(out)
}

pub(crate) fn unpack_mlx(
    w_raw: &[u8],
    s_raw: &[u8],
    b_raw: Option<&[u8]>,
    out_dim: usize,
    in_dim: usize,
    bits: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0f32; out_dim * in_dim];
    let num_groups = s_raw.len() / 2 / out_dim;
    let group_size = in_dim / num_groups;

    let w_u32: Vec<u32> = w_raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let s_f16: Vec<u16> = s_raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let b_f16: Option<Vec<u16>> = b_raw.map(|r| {
        r.chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    });

    let vals_per_u32 = 32 / bits;
    let mask = (1 << bits) - 1;

    for row in 0..out_dim {
        for col in 0..in_dim {
            let group = col / group_size;
            let scale = f16_to_f32(s_f16[row * num_groups + group]);
            let bias = b_f16
                .as_ref()
                .map(|b| f16_to_f32(b[row * num_groups + group]))
                .unwrap_or(0.0);

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
    let num_after = |marker: &str| {
        name.split(marker)
            .nth(1)
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<u32>().ok())
    };
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
        } else if name.contains("self_attn")
            || name.contains("linear_attn")
            || name.contains("short_conv")
        {
            1
        } else if name.contains("post_attention_layernorm") {
            2
        } else if name.ends_with("mlp.gate.weight")
            || name.contains("shared_expert")
            || name.contains("expert_bias")
        {
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
    let obj = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("bad safetensors header"))?;
    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let offs = v["data_offsets"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}': no data_offsets"))?;
        let start = offs[0].as_u64().unwrap_or(0) as usize;
        let end = offs[1].as_u64().unwrap_or(0) as usize;
        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
        });
    }
    Ok(SafeTensors {
        mmap,
        data_start,
        tensors,
    })
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
        let map = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("bad index json"))?;
        let mut files: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        return files
            .iter()
            .map(|f| open_safetensors(&dir.join(f)))
            .collect();
    }
    anyhow::bail!(
        "no model.safetensors or model.safetensors.index.json in {}",
        dir.display()
    )
}

fn cfg_usize(c: &serde_json::Value, key: &str) -> Option<usize> {
    c.get(key).and_then(|v| v.as_u64()).map(|x| x as usize)
}

/// Build ModelArch from a HF config.json (dense transformer families).
fn build_arch(config: &serde_json::Value) -> anyhow::Result<ModelArch> {
    // Vision/multimodal configs nest the text model under "text_config".
    let tc = config.get("text_config").unwrap_or(config);
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let is_dsv4 = model_type == "deepseek_v4";
    // DeepSeek-V4: the name mapping and both source quantizations (FP8
    // E4M3 with 128x128 block scales, MXFP4 experts) are in place, but
    // five of its blocks have no runtime yet — and without them the file
    // would convert and then decode noise. Say exactly what is missing
    // here, at the config, rather than after a 167 GB download.
    if model_type == "deepseek_v4" {
        tracing::info!(
            "deepseek_v4: converting. The engine now carries all five of its \
             blocks — double-LoRA attention with a compressed KV, the per-layer \
             KV compressor, the sparse indexer, hyper-connections (Sinkhorn) and \
             hash routing — so this file is meant to DECODE, not merely to pack. \
             What has not been established is numerical parity against the \
             reference implementation: treat the first coherent generation as \
             the gate, not this message."
        );
    }
    let hidden = cfg_usize(tc, "hidden_size")
        .ok_or_else(|| anyhow::anyhow!("config: missing hidden_size"))?;
    let n_heads = cfg_usize(tc, "num_attention_heads")
        .ok_or_else(|| anyhow::anyhow!("config: missing num_attention_heads"))?;
    let n_layers = cfg_usize(tc, "num_hidden_layers")
        .ok_or_else(|| anyhow::anyhow!("config: missing num_hidden_layers"))?;
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
    // Kimi Linear / Kimi-K3: the per-layer schedule lives in
    // linear_attn_config.full_attn_layers (1-BASED layer numbers);
    // everything else is a KDA layer.
    let tc_model_type = tc.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let is_kimi = model_type == "kimi_linear"
        || model_type == "kimi_k3"
        || tc_model_type == "kimi_linear";
    let layer_types = if let Some(lac) = tc.get("linear_attn_config").filter(|_| is_kimi) {
        anyhow::ensure!(
            tc.get("attn_res_block_size")
                .map(|v| v.is_null())
                .unwrap_or(true),
            "kimi: attn_res_block_size residual streams (Kimi-K3) are not supported yet"
        );
        anyhow::ensure!(
            !tc.get("latent_moe_use_norm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "kimi: latent MoE (routed_expert_up/down_proj, Kimi-K3) is not supported yet"
        );
        anyhow::ensure!(
            !tc.get("mla_use_output_gate")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "kimi: MLA output gate (Kimi-K3) is not supported yet"
        );
        let full: std::collections::HashSet<usize> = lac
            .get("full_attn_layers")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
            .unwrap_or_default();
        anyhow::ensure!(!full.is_empty(), "kimi: linear_attn_config.full_attn_layers is empty");
        (1..=n_layers)
            .map(|i| {
                if full.contains(&i) {
                    LayerType::FullAttention
                } else {
                    LayerType::Kda
                }
            })
            .collect()
    } else {
        layer_types
    };
    let kimi_lac = tc.get("linear_attn_config").filter(|_| is_kimi);
    let has_linear = layer_types
        .iter()
        .any(|t| matches!(t, LayerType::LinearAttention));
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
    let rope = if is_laguna_config {
        rope_root.and_then(|r| r.get("full_attention"))
    } else {
        rope_root
    };
    let local_rope = if is_laguna_config {
        rope_root.and_then(|r| r.get("sliding_attention"))
    } else {
        None
    };
    let rope_theta = tc
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            rope.and_then(|r| r.get("rope_theta"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(10_000.0);
    let prf = tc
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            rope.and_then(|r| r.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(1.0) as f32;
    let local_prf = local_rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    let attention_heads_per_layer = tc
        .get("num_attention_heads_per_layer")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| {
                    v.as_u64().map(|n| n as usize).ok_or_else(|| {
                        anyhow::anyhow!("num_attention_heads_per_layer must contain integers")
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .transpose()?;
    if let Some(heads) = &attention_heads_per_layer {
        anyhow::ensure!(
            heads.len() == n_layers,
            "num_attention_heads_per_layer has {} entries, expected {n_layers}",
            heads.len()
        );
        let nkv = cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads);
        anyhow::ensure!(
            heads.iter().all(|&nh| nh > 0 && nh % nkv == 0),
            "every per-layer attention head count must be positive and divisible by num_key_value_heads={nkv}"
        );
    }
    // Mixture-of-experts: the FFN becomes a router + per-expert matrices. Tensor
    // handling is unchanged (experts are ordinary 2-D matrices); we just declare
    // the MoE config so the runtime dispatches it. Router presence per layer
    // (in the directory) decides which layers are sparse.
    // DeepSeek-V2 MLA geometry (kv_lora_rank marks the family).
    let mla = cfg_usize(tc, "kv_lora_rank").map(|lora| cortiq_core::MlaConfig {
        kv_lora_rank: lora,
        qk_rope_head_dim: cfg_usize(tc, "qk_rope_head_dim").unwrap_or(64),
        qk_nope_head_dim: cfg_usize(tc, "qk_nope_head_dim").unwrap_or(128),
        v_head_dim: cfg_usize(tc, "v_head_dim").unwrap_or(128),
        q_lora_rank: cfg_usize(tc, "q_lora_rank"),
        // Kimi Linear: full-attention layers run NoPE (KDA carries
        // position) — the rotation is skipped, the layout is kept.
        nope: tc
            .get("mla_use_nope")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    });
    let moe = tc
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .or_else(|| tc.get("n_routed_experts").and_then(|v| v.as_u64()))
        .filter(|&n| n > 0)
        .map(|ne| {
            let mt = model_type.to_lowercase();
            let ntp_default =
                mt.starts_with("qwen3_5") || mt.contains("qwen3_next") || mt.contains("gemma4");
            // LFM2-MoE routes with a sigmoid gate + selection bias (DeepSeek-V3
            // noaux_tc); Qwen keeps the softmax-over-all default.
            let is_lfm2 = mt.starts_with("lfm2");
            let is_laguna = mt == "laguna";
            MoeConfig {
                num_experts: ne as usize,
                top_k: cfg_usize(tc, "num_experts_per_tok")
                    .or_else(|| cfg_usize(tc, "top_k_experts"))
                    // Kimi spells it with the full word.
                    .or_else(|| cfg_usize(tc, "num_experts_per_token"))
                    .unwrap_or(2),
                moe_intermediate_size: cfg_usize(tc, "moe_intermediate_size").unwrap_or(0),
                norm_topk_prob: tc
                    .get("norm_topk_prob")
                    .and_then(|v| v.as_bool())
                    // Kimi: moe_renormalize is the same switch.
                    .or_else(|| tc.get("moe_renormalize").and_then(|v| v.as_bool()))
                    .unwrap_or(ntp_default),
                shared_expert_intermediate_size: cfg_usize(tc, "shared_expert_intermediate_size")
                    .or_else(|| {
                        // DeepSeek fuses its n shared experts into one
                        // MLP of n·moe_intermediate_size (Kimi spells the
                        // count num_shared_experts).
                        Some(
                            cfg_usize(tc, "n_shared_experts")
                                .or_else(|| cfg_usize(tc, "num_shared_experts"))?
                                * cfg_usize(tc, "moe_intermediate_size")?,
                        )
                    }),
                router_sigmoid: is_lfm2
                    || is_laguna
                    || tc
                        .get("moe_router_activation_func")
                        .and_then(|v| v.as_str())
                        == Some("sigmoid"),
                // A stored scale of 1.0 is the no-op default; only non-trivial
                // scales need to ride in the header.
                routed_scaling_factor: tc
                    .get("routed_scaling_factor")
                    .or_else(|| tc.get("moe_routed_scaling_factor"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .filter(|&v| (v - 1.0).abs() > 1e-9),
            }
        });
    let head_dim = cfg_usize(tc, "head_dim").unwrap_or(hidden / n_heads.max(1));
    // Zero-centered RMSNorm x̂·(1+w): Gemma family and Qwen3.5 / Qwen3-Next.
    let mt = model_type.to_lowercase();
    let is_laguna = mt == "laguna";
    if is_laguna {
        anyhow::ensure!(
            !tc.get("swa_attention_sink_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "laguna: learned SWA attention sinks are not supported yet"
        );
        anyhow::ensure!(
            tc.get("moe_router_logit_softcapping")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                == 0.0,
            "laguna: non-zero MoE router logit soft-capping is not supported"
        );
        anyhow::ensure!(
            !tc.get("moe_apply_router_weight_on_input")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "laguna: moe_apply_router_weight_on_input=true is not supported"
        );
    }
    let norm_style = if (mt.contains("gemma") && !mt.contains("gemma4") && !mt.contains("gemma3n"))
        || mt.starts_with("qwen3_5")
        || mt.contains("qwen3_next")
    {
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
    // Gemma-2: attention-logit soft-capping is a supported operator
    // (tanh(s/c)·c before the causal softmax); its every-other-layer
    // sliding schedule maps onto sliding_window_pattern = 2 (full
    // attention at odd indices), same machinery as gemma-3/Laguna.
    let is_gemma2 = mt.contains("gemma2");
    // Gemma-4 (text tower): plain x̂·w norms (unlike gemma-3), dual-geometry
    // attention (sliding GQA at head_dim + global MQA at global_head_dim
    // with proportional partial rotary), scale-less V-norm, per-layer
    // output scalars and final-logit capping. The dense 12B/31B variants
    // convert; the MoE / E-series machinery is refused honestly.
    if is_gemma4 {
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
                full.and_then(|f| f.get("rope_theta"))
                    .and_then(|v| v.as_f64()),
                slide
                    .and_then(|f| f.get("rope_theta"))
                    .and_then(|v| v.as_f64()),
                full.and_then(|f| f.get("partial_rotary_factor"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
            )
        }
        _ => (None, None, None),
    };
    let rope_theta = g4_rope_theta.unwrap_or(rope_theta);
    // Both spellings are in the wild: newer configs say `rope_type`, older
    // ones (DeepSeek-V4 among them) say `type`. Reading only one of them
    // drops the profile silently and the model decodes with unscaled
    // frequencies — which looks like a bad conversion, not a missing field.
    let yarn = rope
        .filter(|r| {
            r.get("rope_type")
                .or_else(|| r.get("type"))
                .and_then(|v| v.as_str())
                == Some("yarn")
        })
        .map(|r| {
            Ok::<YarnConfig, anyhow::Error>(YarnConfig {
                factor: r
                    .get("factor")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("YaRN rope profile is missing factor"))?
                    as f32,
                original_max_position_embeddings: r
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "YaRN rope profile is missing original_max_position_embeddings"
                        )
                    })? as usize,
                beta_fast: r.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32,
                beta_slow: r.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                mscale_all_dim: r
                    .get("mscale_all_dim")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                attention_factor: r
                    .get("attention_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or_else(|| {
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
        let fulls: Vec<usize> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter(|(_, v)| v.as_str() == Some("full_attention"))
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default();
        let p = fulls.first().map(|f| f + 1).unwrap_or(0);
        if p == 0
            || fulls.iter().enumerate().any(|(k, &i)| i != p * (k + 1) - 1)
            || (n_layers / p) != fulls.len()
        {
            anyhow::bail!("{model_type}: irregular full/sliding layer schedule not supported");
        }
        Some(p)
    } else {
        None
    };
    let hidden_act = match tc
        .get("hidden_activation")
        .or_else(|| tc.get("hidden_act"))
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
    {
        "gelu_pytorch_tanh" | "gelu_tanh" | "gelu_new" => "gelu_tanh".to_string(),
        "silu" | "swish" => "silu".to_string(),
        other => anyhow::bail!("unsupported hidden_act '{other}'"),
    };
    let is_minicpm3 = model_type == "minicpm3";
    let embed_multiplier = if is_gemma {
        (hidden as f32).sqrt()
    } else if is_minicpm3 {
        // MiniCPM: embeddings enter the stack ×scale_emb.
        tc.get("scale_emb").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32
    } else {
        1.0
    };
    // MiniCPM: logits = lm_head(h) · dim_model_base/hidden; the head is
    // tied to the embedding, so the divisor rides in the header.
    let logit_multiplier = if is_minicpm3 {
        cfg_usize(tc, "dim_model_base").map(|d| d as f32 / hidden as f32)
    } else {
        None
    };
    // Gemma-3n (E-series): AltUp/LAuReL/PLE/KV-sharing geometry rides
    // in the header; the runtime switches to the dedicated stack.
    let is_gemma3n =
        model_type.starts_with("gemma3n") || tc_model_type.starts_with("gemma3n");
    let g3n_cfg: Option<cortiq_core::G3nConfig> = if is_gemma3n {
        let need = |k: &str| {
            cfg_usize(tc, k).ok_or_else(|| anyhow::anyhow!("gemma3n: config missing {k}"))
        };
        Some(cortiq_core::G3nConfig {
            altup_num_inputs: need("altup_num_inputs")?,
            laurel_rank: need("laurel_rank")?,
            ple_dim: need("hidden_size_per_layer_input")?,
            ple_vocab: need("vocab_size_per_layer_input")?,
            num_kv_shared_layers: need("num_kv_shared_layers")?,
            activation_sparsity: tc
                .get("activation_sparsity_pattern")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
                .unwrap_or_default(),
        })
    } else {
        None
    };
    let mut rope_freq_factors: Option<Vec<f64>> = None;
    // Phi-3 longrope: exact only within the ORIGINAL context — cap the
    // declared max honestly instead of serving stretched positions.
    let mut max_pos = cfg_usize(tc, "max_position_embeddings")
        .unwrap_or(if is_kimi { 1_048_576 } else { 32768 });
    if let Some(rs) = tc.get("rope_scaling").filter(|v| !v.is_null()) {
        let kind = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str());
        match kind {
            Some("longrope") | Some("su") | Some("yarn") | Some("linear") | Some("dynamic")
            | Some("mrope") => {
                let orig = cfg_usize(tc, "original_max_position_embeddings")
                    .or_else(|| {
                        rs.get("original_max_position_embeddings")
                            .and_then(|v| v.as_u64().map(|v| v as usize))
                    })
                    .unwrap_or(4096);
                eprintln!(
                    "  note: rope scaling '{:?}' — serving the exact {orig}-token native window",
                    kind.unwrap()
                );
                max_pos = orig;
                // MiniCPM3: the short factors are the TRAINED rope inside
                // the native window (1.06–16.9 per dim — not ≈1 like phi);
                // carry them so inv_freq divides per frequency at load.
                if is_minicpm3 {
                    if let Some(f) = rs.get("short_factor").and_then(|v| v.as_array()) {
                        let fac: Vec<f64> = f.iter().filter_map(|v| v.as_f64()).collect();
                        anyhow::ensure!(!fac.is_empty(), "minicpm3: empty longrope short_factor");
                        rope_freq_factors = Some(fac);
                    }
                }
            }
            Some(other) => anyhow::bail!("rope_scaling '{other}' not supported yet"),
            None => {}
        }
    }
    Ok(ModelArch {
        arch_name: model_type,
        hidden_size: hidden,
        intermediate_size: cfg_usize(tc, "intermediate_size")
            .or_else(|| {
                // Gemma-3n stores a per-layer LIST; E-models keep it
                // uniform — take the head and insist on uniformity.
                tc.get("intermediate_size").and_then(|v| v.as_array()).and_then(|a| {
                    let first = a.first()?.as_u64()?;
                    a.iter().all(|v| v.as_u64() == Some(first)).then_some(first as usize)
                })
            })
            .or_else(|| cfg_usize(tc, "moe_intermediate_size"))
            .ok_or_else(|| anyhow::anyhow!("config: missing intermediate_size"))?,
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: cfg_usize(tc, "num_key_value_heads").unwrap_or(n_heads),
        head_dim,
        vocab_size: cfg_usize(tc, "vocab_size")
            .ok_or_else(|| anyhow::anyhow!("config: missing vocab_size"))?,
        layer_types,
        // LFM2 spells the RMSNorm epsilon `norm_eps`.
        rms_norm_eps: tc
            .get("rms_norm_eps")
            .or_else(|| tc.get("norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6),
        norm_style,
        rope_theta,
        // Gemma ties embeddings by default and its configs omit the key.
        tie_word_embeddings: config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            // MiniCPM3 ships no lm_head tensor — the head is the embedding.
            .unwrap_or(is_gemma || is_minicpm3),
        // DeepSeek-V4 rotates only `qk_rope_head_dim` of each head's 512,
        // and its config states that directly rather than as a fraction.
        // Carrying it here is what lets a retuned checkpoint load without
        // the loader guessing.
        partial_rotary_factor: if is_dsv4 {
            match (cfg_usize(tc, "qk_rope_head_dim"), cfg_usize(tc, "head_dim")) {
                (Some(rd), Some(hd)) if hd > 0 => rd as f32 / hd as f32,
                _ => prf,
            }
        } else {
            prf
        },
        yarn,
        attention_heads_per_layer,
        // MTP head, when the checkpoint ships one. Qwen3.6 spells the count
        // `mtp_num_hidden_layers`; DeepSeek-lineage configs say
        // `num_nextn_predict_layers`. Absent → no speculative head, which is
        // the honest default for every model that has none.
        mtp: ["mtp_num_hidden_layers", "num_nextn_predict_layers"]
            .iter()
            .find_map(|k| cfg_usize(tc, k))
            .filter(|n| *n > 0)
            .map(|n| cortiq_core::MtpConfig {
                num_layers: n,
                share_lm_head: true,
                share_embed: true,
            }),
        moe,
        linear_core,
        max_position_embeddings: max_pos,
        // GDN spells it `linear_conv_kernel_dim`; LFM2 spells it
        // `conv_L_cache`; Kimi nests KDA geometry in linear_attn_config.
        linear_conv_kernel_dim: cfg_usize(tc, "linear_conv_kernel_dim")
            .or_else(|| cfg_usize(tc, "conv_L_cache"))
            .or_else(|| kimi_lac.and_then(|l| cfg_usize(l, "short_conv_kernel_size"))),
        linear_num_key_heads: cfg_usize(tc, "linear_num_key_heads")
            .or_else(|| kimi_lac.and_then(|l| cfg_usize(l, "num_heads"))),
        linear_num_value_heads: lnv,
        linear_key_head_dim: cfg_usize(tc, "linear_key_head_dim")
            .or_else(|| kimi_lac.and_then(|l| cfg_usize(l, "head_dim"))),
        linear_value_head_dim: lvd
            .or_else(|| kimi_lac.and_then(|l| cfg_usize(l, "head_dim"))),
        hidden_act,
        embed_multiplier,
        // Gemma-4 attends with scaling = 1.0 (q-norm carries the scale).
        query_pre_attn_scalar: tc
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            // Gemma-4 and Gemma-3n attend with scale 1.0 (q-norm carries it).
            .or(if is_gemma4 || is_gemma3n { Some(1.0) } else { None }),
        sliding_window: cfg_usize(tc, "sliding_window").filter(|_| {
            is_laguna
                || is_gemma2
                || is_gemma3n
                || tc.get("sliding_window_pattern").is_some()
                || g4_pattern.is_some()
        }),
        sliding_window_pattern: cfg_usize(tc, "sliding_window_pattern")
            .or(g4_pattern)
            .or(if is_gemma2 { Some(2) } else { None }),
        rope_local_base_freq: tc
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .or(g4_local_theta)
            .or_else(|| {
                local_rope
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            }),
        local_partial_rotary_factor: local_prf,
        global_head_dim: cfg_usize(tc, "global_head_dim").filter(|_| is_gemma4),
        num_global_kv_heads: cfg_usize(tc, "num_global_key_value_heads").filter(|_| is_gemma4),
        global_partial_rotary_factor: g4_global_prf,
        final_logit_softcapping: tc.get("final_logit_softcapping").and_then(|v| v.as_f64()),
        attn_logit_softcapping: tc.get("attn_logit_softcapping").and_then(|v| v.as_f64()),
        mla,
        activation_situ_beta: tc.get("activation_situ_beta").and_then(|v| v.as_f64()),
        activation_situ_linear_beta: tc
            .get("activation_situ_linear_beta")
            .and_then(|v| v.as_f64()),
        attn_v_norm: is_gemma4,
        rope_freq_factors,
        logit_multiplier,
        g3n: g3n_cfg,
        // KDA (Kimi-K3): lower-bound decay-gate variant; absent = standard.
        kda_gate_lower_bound: tc
            .get("linear_attn_config")
            .and_then(|c| c.get("gate_lower_bound"))
            .and_then(|v| v.as_f64()),
        // Looped Transformer (Nanbeige 4.2): re-apply the layer stack num_loops times.
        num_loops: cfg_usize(tc, "num_loops").unwrap_or(1),
        // skip_loop_final_norm=false means loop_final_norm=true (apply norm after each loop).
        loop_final_norm: !tc
            .get("skip_loop_final_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
    })
}

/// Collect eos ids from generation_config.json / config.json (int or array).
fn eos_ids(gen_cfg: &serde_json::Value, config: &serde_json::Value) -> Vec<u32> {
    for v in [gen_cfg.get("eos_token_id"), config.get("eos_token_id")]
        .into_iter()
        .flatten()
    {
        if let Some(n) = v.as_u64() {
            return vec![n as u32];
        }
        if let Some(a) = v.as_array() {
            return a
                .iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect();
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
    let resp = auth(agent.get(url).set("Range", "bytes=0-0"), token)
        .call()
        .ok()?;
    resp.header("Content-Range")?
        .rsplit('/')
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

fn get_range(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
) -> anyhow::Result<Vec<u8>> {
    let resp = auth(
        agent
            .get(url)
            .set("Range", &format!("bytes={}-{}", start, end - 1)),
        token,
    )
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
fn fetch(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    token: Option<&str>,
    required: bool,
    threads: usize,
) -> anyhow::Result<bool> {
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
            let chunks: Vec<(u64, u64)> = (0..sz)
                .step_by(HF_CHUNK as usize)
                .map(|s| (s, (s + HF_CHUNK).min(sz)))
                .collect();
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
                            let r = with_retry(4, || get_range(agent, url, token, start, end))
                                .and_then(|buf| write_at(&tmp, start, &buf).map_err(Into::into));
                            match r {
                                Ok(()) => {
                                    let d =
                                        done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                    eprint!(
                                        "\r    downloading: {:>3}% ({d}/{total} chunks)",
                                        d * 100 / total
                                    );
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
            .and_then(|j| {
                j["siblings"].as_array().map(|a| {
                    a.iter()
                        .filter_map(|s| s["rfilename"].as_str().map(String::from))
                        .collect()
                })
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Fetch a HF repo's convertible files (config, tokenizer, weights) into the
/// cache, with parallel chunked downloads for the weight shards.
pub(crate) fn hf_download(repo: &str, token: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    Ok(hf_download_opts(repo, token, true)?.0)
}

/// `weights=false`: fetch only config/tokenizer/index and RETURN the
/// weight-shard names without downloading them — the streaming convert
/// pulls them one at a time (peak disk = one shard + the output).
pub(crate) fn hf_download_opts(
    repo: &str,
    token: Option<&str>,
    weights: bool,
) -> anyhow::Result<(std::path::PathBuf, Vec<String>)> {
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
    if !fetch(
        &agent,
        &format!("{base}/config.json"),
        &dir.join("config.json"),
        token,
        false,
        threads,
    )? {
        let files = repo_files(&agent, repo, token);
        let ggufs = files
            .iter()
            .filter(|f| f.to_lowercase().ends_with(".gguf"))
            .count();
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
        // Kimi ships tiktoken.model instead of tokenizer.json — either
        // satisfies the bundle (checked after conversion).
        ("tokenizer.json", false),
        ("tiktoken.model", false),
        ("tokenizer_config.json", false),
        ("generation_config.json", false),
        // Newer HF checkpoints (LFM2, Qwen3, …) ship the chat template as a
        // sidecar `chat_template.jinja` instead of embedding it in
        // tokenizer_config.json — without it `run` falls back to a generic
        // ChatML default that does not match the model's real format.
        ("chat_template.jinja", false),
    ] {
        fetch(
            &agent,
            &format!("{base}/{f}"),
            &dir.join(f),
            token,
            required,
            threads,
        )?;
    }
    let idx = dir.join("model.safetensors.index.json");
    if fetch(
        &agent,
        &format!("{base}/model.safetensors.index.json"),
        &idx,
        token,
        false,
        1,
    )? {
        let j: serde_json::Value = serde_json::from_slice(&fs::read(&idx)?)?;
        let map = j["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("bad safetensors index"))?;
        let mut shards: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        if weights {
            for (i, s) in shards.iter().enumerate() {
                eprintln!(
                    "  shard {}/{} ({threads}× parallel): {s}",
                    i + 1,
                    shards.len()
                );
                fetch(
                    &agent,
                    &format!("{base}/{s}"),
                    &dir.join(s),
                    token,
                    true,
                    threads,
                )?;
            }
        }
        return Ok((dir, shards));
    } else if weights {
        eprintln!("  model.safetensors ({threads}× parallel)");
        fetch(
            &agent,
            &format!("{base}/model.safetensors"),
            &dir.join("model.safetensors"),
            token,
            true,
            threads,
        )?;
    }
    Ok((dir, vec!["model.safetensors".to_string()]))
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
            anyhow::bail!(
                "fused GDN qkvz: {} values, expected {}",
                w.len(),
                nk * gw * hid
            );
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
            anyhow::bail!(
                "fused GDN ba: {} values, expected {}",
                w.len(),
                nk * gw * hid
            );
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
fn slice_ffn(
    kind: &FfnKind,
    shape: &[usize],
    vals: &[f32],
    keep: &[bool],
) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
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
fn effective_tensor(
    overlay: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    files: &[SafeTensors],
    name: &str,
) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
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
fn build_defrag_plan(
    dir: &Path,
    arch: &ModelArch,
    files: &[SafeTensors],
) -> anyhow::Result<DefragPlan> {
    let mut overlay: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let tdir = dir.join("tensors");
    if tdir.is_dir() {
        for entry in fs::read_dir(&tdir)? {
            let p = entry?.path();
            if p.extension().and_then(|e| e.to_str()) != Some("npy") {
                continue;
            }
            let stem = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
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
    println!(
        "  Defrag overlay: {} baked tensors from {}",
        overlay.len(),
        dir.display()
    );

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
    // The q2tp PROFILE: 2-bit tiles go to the MoE gate/up experts only —
    // `down` experts and the whole skeleton stay q4tp (the 2/4 split that
    // mirrors Escha's 2/3-bit choice). `gu_quant` is what gate/up get.
    let gu_quant = quant;
    let quant = if matches!(quant, Quant::Q2TiledP) {
        Quant::Q4TiledP
    } else {
        quant
    };

    // Source: a local HF directory, or an HF repo id — hub checkpoints
    // convert STREAMED: one weight shard on disk at a time.
    let downloaded;
    let mut stream_shards: Vec<String> = Vec::new();
    let mut stream_repo: Option<String> = None;
    let dir: &Path = if Path::new(model).join("config.json").exists() {
        Path::new(model)
    } else if looks_like_repo(model) {
        eprintln!("downloading {model} from Hugging Face (streamed)…");
        let (d, shards) = hf_download_opts(model, hf_token, false)?;
        stream_shards = shards;
        stream_repo = Some(model.to_string());
        downloaded = d;
        downloaded.as_path()
    } else {
        anyhow::bail!(
            "'{model}': not a local model dir (no config.json) and not an HF repo id (owner/name)"
        );
    };

    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(dir.join("config.json")).map_err(|e| anyhow::anyhow!("read config.json: {e}"))?,
    )?;
    let mut arch = build_arch(&config)?;

    // Memory-map the weights and process one tensor at a time — the raw model is
    // never fully loaded into RAM (peak ≈ the .cmf output + one tensor).
    let files = if stream_repo.is_some() {
        Vec::new() // shards arrive one at a time below
    } else {
        open_model(dir)?
    };
    if stream_repo.is_some() && defrag.is_some() {
        anyhow::bail!("--defrag needs a local checkpoint dir (streaming hub convert)");
    }

    // Physical defragmentation plan (spec §11): drop pruned FFN neurons so
    // they are neither stored nor computed. arch.intermediate_size becomes
    // nominal/max (per-layer truth lives in the reduced tensor shapes).
    let orig_inter = arch.intermediate_size;
    let defrag_plan = match defrag {
        Some(d) => Some(build_defrag_plan(Path::new(d), &arch, &files)?),
        None => None,
    };
    if let Some(plan) = &defrag_plan {
        let max_kept = (0..arch.num_layers)
            .filter_map(|li| plan.keep.get(&li).map(|k| k.iter().filter(|&&b| b).count()))
            .max()
            .unwrap_or(orig_inter);
        arch.intermediate_size = max_kept;
    }
    let total: usize = files.iter().map(|f| f.tensors.len()).sum::<usize>().max(1);
    // A 300B-class MoE encodes to ~100 GB, and holding that in
    // `Vec<TensorSpec>` until the writer runs OOMs any machine (measured:
    // +1.8 GB/min, a 176 GB box exhausted mid-model). Each encoded tensor
    // therefore goes straight into the output file and is dropped; the head
    // is patched into a reserved gap once the last tensor lands, so the
    // payloads are never held twice — not in RAM and not on disk.
    let head_reserve = cortiq_core::format::CmfStreamWriter::head_reserve_for(2 * total.max(4096), 96);
    let manifest_path = format!("{output}.manifest");
    let mut writer = cortiq_core::format::CmfStreamWriter::new(output, head_reserve)
        .and_then(|w| w.with_manifest(&manifest_path))
        .map_err(|e| anyhow::anyhow!("create {output}: {e}"))?;
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    let mut done = 0usize;
    // Tiny cross-shard tensors (gemma-4 router.scale, ~128 f32 each):
    // stashed as shards stream by, so a projection in shard 2 can fold
    // a scale that lived in the already-deleted shard 1.
    let mut small_stash: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    // Per-file conversion body, shared by the resident local path and
    // the streaming hub path (shard downloaded → converted → DELETED;
    // peak disk = one shard + the growing output).
    // MiniCPM residual-branch scale (folded at write).
    let resid_fold: Option<f32> = if arch.arch_name == "minicpm3" {
        let tcv = config.get("text_config").unwrap_or(&config);
        let sd = tcv.get("scale_depth").and_then(|v| v.as_f64()).unwrap_or(1.0);
        Some((sd / (arch.num_layers as f64).sqrt()) as f32)
    } else {
        None
    };
    // Kimi: which layer indices are KDA (canon retag inside the loop).
    // Kimi: which layer indices are KDA (canon retag inside the loop).
    let kda_layers: std::collections::HashSet<usize> = arch
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| matches!(t, cortiq_core::LayerType::Kda))
        .map(|(i, _)| i)
        .collect();

    let mut process_file = |file: &SafeTensors,
                            files: &[SafeTensors],
                            tensors: &mut Vec<TensorSpec>,
                            done: &mut usize,
                            total: usize,
                            progress: &mut dyn FnMut(f32)|
     -> anyhow::Result<()> {
        for m in &file.tensors {
            if m.name.ends_with(".router.scale") {
                small_stash.insert(m.name.clone(), to_f32(&m.dtype, file.bytes(m))?);
            }
        }
        for m in &file.tensors {
            *done += 1;
            progress(*done as f32 / total as f32);
            let Some(name) = canon_name(&m.name) else {
                continue;
            };
            // Kimi: KDA layers share the `self_attn.` vendor prefix with
            // the MLA full-attention layers — retag them by the layer
            // schedule so the loader dispatches unambiguously.
            let name = if kda_layers.is_empty() {
                name
            } else {
                match name
                    .strip_prefix("model.layers.")
                    .and_then(|r| r.split_once('.'))
                    .and_then(|(li, rest)| Some((li.parse::<usize>().ok()?, rest)))
                {
                    Some((li, rest)) if kda_layers.contains(&li) && rest.starts_with("self_attn.") => {
                        format!(
                            "model.layers.{li}.kda_attn.{}",
                            &rest["self_attn.".len()..]
                        )
                    }
                    _ => name,
                }
            };

            // Skip MLX scales and biases as they are processed with the weight.
            if m.dtype == "F16" && (name.ends_with(".scales") || name.ends_with(".biases")) {
                continue;
            }
            // MXFP4 group scales ride with their .weight_packed twin.
            if m.dtype == "U8" && name.ends_with(".weight_scale") {
                continue;
            }
            // DeepSeek-V4 keeps every quantized tensor's E8M0 scale plane in
            // a sibling `.scale`; it rides with the weight below.
            if m.dtype == "F8_E8M0" && name.ends_with(".scale") {
                continue;
            }
            // Its own two quantizations, decoded to f32 so the rest of the
            // pipeline (defrag, requant to q4tp/q2tp) is layout-agnostic:
            //   * experts  — I8 holding two FP4 (E2M1) values per byte with
            //     one E8M0 scale per 32 values: OCP MXFP4 exactly.
            //   * skeleton — F8_E4M3 with one E8M0 scale per 128x128 tile.
            let dsv4 = if matches!(m.dtype.as_str(), "I8" | "F8_E4M3")
                && !name.ends_with(".scale")
            {
                let scale_name = format!("{}.scale", m.name.trim_end_matches(".weight"));
                let mut sb = None;
                for f in files {
                    if let Some(t) = f.tensors.iter().find(|t| t.name == scale_name) {
                        sb = Some(f.bytes(t));
                    }
                }
                match sb {
                    Some(scales) if m.shape.len() == 2 => {
                        let (rows, cols) = (m.shape[0], m.shape[1]);
                        if m.dtype == "I8" {
                            // packed FP4: `cols` bytes = 2*cols values
                            Some((
                                vec![rows, cols * 2],
                                unpack_mxfp4(file.bytes(m), scales, rows, cols)?,
                            ))
                        } else {
                            Some((
                                vec![rows, cols],
                                unpack_fp8_blocks(file.bytes(m), scales, rows, cols, 128)?,
                            ))
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            let mxfp4 = if m.dtype == "U8" && m.name.ends_with(".weight_packed") {
                // MXFP4 (Kimi-K3 experts): decode to f32 and continue the
                // normal path under the plain `.weight` name.
                let scale_name = m.name.replace(".weight_packed", ".weight_scale");
                let mut scales_blob = None;
                for f in files {
                    if let Some(t) = f.tensors.iter().find(|t| t.name == scale_name) {
                        scales_blob = Some(f.bytes(t));
                    }
                }
                let scales = scales_blob
                    .ok_or_else(|| anyhow::anyhow!("missing {scale_name} for mxfp4 unpacking"))?;
                anyhow::ensure!(m.shape.len() == 2, "{name}: mxfp4 expects 2-D");
                let (rows, cp) = (m.shape[0], m.shape[1]);
                Some((vec![rows, cp * 2], unpack_mxfp4(file.bytes(m), scales, rows, cp)?))
            } else {
                None
            };
            let name = if mxfp4.is_some() {
                name.strip_suffix("_packed").expect("suffix checked").to_string()
            } else {
                name
            };
            let mxfp4 = mxfp4.or(dsv4);
            let (m_shape, m_vals) = if let Some(v) = mxfp4 {
                v
            } else if m.dtype == "U32" && m.name.ends_with(".weight") {
                let scales_name = m.name.replace(".weight", ".scales");
                let biases_name = m.name.replace(".weight", ".biases");
                let mut scales_blob = None;
                let mut biases_blob = None;
                for f in files {
                    if let Some(t) = f.tensors.iter().find(|t| t.name == scales_name) {
                        scales_blob = Some(f.bytes(t));
                    }
                    if let Some(t) = f.tensors.iter().find(|t| t.name == biases_name) {
                        biases_blob = Some(f.bytes(t));
                    }
                }
                let scales = scales_blob
                    .ok_or_else(|| anyhow::anyhow!("missing {} for MLX unpacking", scales_name))?;
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
                    anyhow::bail!(
                        "Could not deduce MLX bit width for shape {:?} and {} scale groups",
                        m.shape,
                        num_groups
                    );
                }
                (
                    vec![out_dim, in_dim],
                    unpack_mlx(file.bytes(m), scales, biases_blob, out_dim, in_dim, bits)?,
                )
            } else {
                (m.shape.clone(), to_f32(&m.dtype, file.bytes(m))?)
            };

            let mut m_vals = m_vals;
            // MiniCPM: every residual add is h += branch·(scale_depth/√L)
            // — fold the factor into the branch output rows (o_proj and
            // the dense FFN down_proj) so the runtime stays standard.
            if let Some(fs) = resid_fold {
                if name.ends_with("self_attn.o_proj.weight")
                    || name.ends_with(".mlp.down_proj.weight")
                {
                    for v in m_vals.iter_mut() {
                        *v *= fs;
                    }
                }
            }

            // DeepSeek-V2 MLA: permute each q head to rope-first
            // ([nope|rope] → [rope|nope]) so the runtime's standard
            // partial rotary (first rotary_dim dims) covers it.
            if arch.mla.is_some()
                && (name.ends_with("self_attn.q_proj.weight")
                    || name.ends_with("self_attn.q_b_proj.weight"))
            {
                let mla = arch.mla.as_ref().unwrap();
                let (dr, dn) = (mla.qk_rope_head_dim, mla.qk_nope_head_dim);
                let hd = dr + dn;
                anyhow::ensure!(
                    m_shape.len() == 2 && m_shape[0] % hd == 0,
                    "{name}: rows {:?} not a multiple of qk head dim {hd}",
                    m_shape
                );
                let cols = m_shape[1];
                let nh = m_shape[0] / hd;
                let mut w = vec![0.0f32; m_vals.len()];
                for h in 0..nh {
                    for r in 0..dr {
                        // DeepSeek rotary interleaves pairs (view d/2,2 →
                        // transpose): store rope dims even-first so our
                        // half-split rotation reproduces their math.
                        // Kimi NoPE (no rotation) and MiniCPM3 (plain
                        // half-split rotate_half) keep the rope block
                        // order — only the move in front of nope.
                        let interleaved = !mla.nope && arch.arch_name.contains("deepseek");
                        let src = if !interleaved {
                            r
                        } else if r < dr / 2 {
                            2 * r
                        } else {
                            2 * (r - dr / 2) + 1
                        };
                        w[(h * hd + r) * cols..(h * hd + r + 1) * cols].copy_from_slice(
                            &m_vals[(h * hd + dn + src) * cols..(h * hd + dn + src + 1) * cols],
                        );
                    }
                    for r in 0..dn {
                        w[(h * hd + dr + r) * cols..(h * hd + dr + r + 1) * cols].copy_from_slice(
                            &m_vals[(h * hd + r) * cols..(h * hd + r + 1) * cols],
                        );
                    }
                }
                let (dt, data) = quantize_2d(quant, &w, m_shape[0], cols);
                tensors.push(TensorSpec {
                    name,
                    dtype: dt,
                    shape: m_shape.clone(),
                    data,
                });
                continue;
            }

            // DeepSeek MLA: the shared rope key rows (tail of kv_a) get
            // the same even-first interleave fix. Kimi NoPE skips it —
            // no rotation, the tail rows are already where the loader
            // expects them.
            if arch.mla.as_ref().is_some_and(|m| !m.nope)
                && arch.arch_name.contains("deepseek")
                && name.ends_with("self_attn.kv_a_proj_with_mqa.weight")
            {
                let mla = arch.mla.as_ref().unwrap();
                let (lora, dr) = (mla.kv_lora_rank, mla.qk_rope_head_dim);
                anyhow::ensure!(
                    m_shape.len() == 2 && m_shape[0] == lora + dr,
                    "{name}: rows {:?} != lora+rope",
                    m_shape
                );
                let cols = m_shape[1];
                let mut w = m_vals.clone();
                for r in 0..dr {
                    let src = if r < dr / 2 { 2 * r } else { 2 * (r - dr / 2) + 1 };
                    w[(lora + r) * cols..(lora + r + 1) * cols]
                        .copy_from_slice(&m_vals[(lora + src) * cols..(lora + src + 1) * cols]);
                }
                let (dt, data) = quantize_2d(quant, &w, m_shape[0], cols);
                tensors.push(TensorSpec {
                    name,
                    dtype: dt,
                    shape: m_shape.clone(),
                    data,
                });
                continue;
            }

            // Gemma-4 MoE: packed 3-D expert tensors + a router whose
            // input gain (scale-less rms ⊙ router.scale · hidden^-1/2)
            // folds entirely into the projection columns. Experts split
            // into the canonical per-expert 2-D matrices; the fused
            // gate_up halves are [gate | up] along the last axis.
            if name.ends_with(".experts.gate_up_proj") || name.ends_with(".experts.down_proj") {
                anyhow::ensure!(m_shape.len() == 3, "{name}: expected 3-D, got {:?}", m_shape);
                let base = name
                    .strip_suffix(".experts.gate_up_proj")
                    .or_else(|| name.strip_suffix(".experts.down_proj"))
                    .unwrap();
                // Two lineages name the packed tensor differently: gemma-4
                // has `layers.N.experts.*`, qwen3.5/3.6 `layers.N.mlp.experts.*`.
                // The output name below appends `.mlp.experts`, so a base that
                // already ends in `.mlp` would double it — and a doubled prefix
                // converts silently, then fails at load with "router present but
                // no expert tensors", which points nowhere near here.
                let base = base.strip_suffix(".mlp").unwrap_or(base);
                let moe = arch.moe.as_ref().ok_or_else(|| anyhow::anyhow!("no moe cfg"))?;
                let (mi, hid) = (moe.moe_intermediate_size, arch.hidden_size);
                let (ne, d1, d2) = (m_shape[0], m_shape[1], m_shape[2]);
                let is_gu = name.ends_with("gate_up_proj");
                // Both orientations occur in the wild: standard per-expert
                // nn.Linear [out, in] rows (gemma-4 QAT) or the packed
                // [in, out] (Mixtral lineage). Detect by dims.
                let row_major = if is_gu {
                    if (d1, d2) == (2 * mi, hid) {
                        true
                    } else if (d1, d2) == (hid, 2 * mi) {
                        false
                    } else {
                        anyhow::bail!("{name}: dims {:?} match neither orientation", m_shape)
                    }
                } else if (d1, d2) == (hid, mi) {
                    true
                } else if (d1, d2) == (mi, hid) {
                    false
                } else {
                    anyhow::bail!("{name}: dims {:?} match neither orientation", m_shape)
                };
                for e in 0..ne {
                    let ev = &m_vals[e * d1 * d2..(e + 1) * d1 * d2];
                    let emit = |tensors: &mut Vec<TensorSpec>,
                                nm: String,
                                vals: &[f32],
                                rows: usize,
                                cols: usize,
                                q: Quant| {
                        let (dt, data) = if rows * cols >= GROUP_SIZE && !force_f16(&nm) {
                            quantize_2d(q, vals, rows, cols)
                        } else {
                            (TensorDtype::F16, encode_f16(vals))
                        };
                        tensors.push(TensorSpec {
                            name: nm,
                            dtype: dt,
                            shape: vec![rows, cols],
                            data,
                        });
                    };
                    if is_gu {
                        let mut gate = vec![0.0f32; mi * hid];
                        let mut up = vec![0.0f32; mi * hid];
                        if row_major {
                            gate.copy_from_slice(&ev[..mi * hid]);
                            up.copy_from_slice(&ev[mi * hid..]);
                        } else {
                            for h in 0..hid {
                                for r in 0..mi {
                                    gate[r * hid + h] = ev[h * d2 + r];
                                    up[r * hid + h] = ev[h * d2 + mi + r];
                                }
                            }
                        }
                        emit(
                            &mut *tensors,
                            format!("{base}.mlp.experts.{e}.gate_proj.weight"),
                            &gate,
                            mi,
                            hid,
                            gu_quant,
                        );
                        emit(
                            &mut *tensors,
                            format!("{base}.mlp.experts.{e}.up_proj.weight"),
                            &up,
                            mi,
                            hid,
                            gu_quant,
                        );
                    } else {
                        let mut down = vec![0.0f32; hid * mi];
                        if row_major {
                            down.copy_from_slice(ev);
                        } else {
                            for r in 0..mi {
                                for h in 0..hid {
                                    down[h * mi + r] = ev[r * d2 + h];
                                }
                            }
                        }
                        emit(
                            &mut *tensors,
                            format!("{base}.mlp.experts.{e}.down_proj.weight"),
                            &down,
                            hid,
                            mi,
                            quant,
                        );
                    }
                }
                continue;
            }
            if name.ends_with(".router.proj.weight") {
                anyhow::ensure!(m_shape.len() == 2, "{name}: expected 2-D, got {:?}", m_shape);
                let (ne, hid) = (m_shape[0], m_shape[1]);
                // Fold router.scale ⊙ hidden^-1/2 into the columns.
                let scale_name = m.name.replace(".proj.weight", ".scale");
                let mut scale_blob = small_stash.get(&scale_name).cloned();
                if scale_blob.is_none() {
                    for f in files {
                        if let Some(t) = f.tensors.iter().find(|t| t.name == scale_name) {
                            scale_blob = Some(to_f32(&t.dtype, f.bytes(t))?);
                        }
                    }
                }
                let scale =
                    scale_blob.ok_or_else(|| anyhow::anyhow!("missing {scale_name} for fold"))?;
                anyhow::ensure!(scale.len() == hid, "{scale_name}: len != hidden");
                let c = 1.0 / (hid as f32).sqrt();
                let mut w = m_vals.clone();
                for r in 0..ne {
                    for j in 0..hid {
                        w[r * hid + j] *= scale[j] * c;
                    }
                }
                let base = name.strip_suffix(".router.proj.weight").unwrap();
                tensors.push(TensorSpec {
                    name: format!("{base}.mlp.gate.weight"),
                    dtype: TensorDtype::F16,
                    shape: vec![ne, hid],
                    data: encode_f16(&w),
                });
                continue;
            }
            if name.ends_with(".router.scale") {
                continue; // folded into the router projection above
            }
            if name.ends_with(".router.per_expert_scale") {
                let base = name.strip_suffix(".router.per_expert_scale").unwrap();
                tensors.push(TensorSpec {
                    name: format!("{base}.mlp.per_expert_scale"),
                    dtype: TensorDtype::F16,
                    shape: vec![m_vals.len()],
                    data: encode_f16(&m_vals),
                });
                continue;
            }

            // qwen3_next / AgentWorld fuse the GDN projections (in_proj_qkvz /
            // in_proj_ba) with a group-interleaved layout; split them natively
            // into the canonical hub tensors (in_proj_qkv/z/a/b). Pure row
            // permutation — no value is changed.
            if name.contains(".linear_attn.in_proj_qkvz")
                || name.contains(".linear_attn.in_proj_ba")
            {
                if m_shape.len() != 2 {
                    anyhow::bail!("fused GDN tensor '{name}': expected 2-D, got {:?}", m_shape);
                }
                let w = &m_vals;
                let hid = m_shape[1];
                let miss = |k: &str| anyhow::anyhow!("fused GDN needs {k} in config");
                let nk = arch
                    .linear_num_key_heads
                    .ok_or_else(|| miss("linear_num_key_heads"))?;
                let dk = arch
                    .linear_key_head_dim
                    .ok_or_else(|| miss("linear_key_head_dim"))?;
                let nv = arch
                    .linear_num_value_heads
                    .ok_or_else(|| miss("linear_num_value_heads"))?;
                let dv = arch
                    .linear_value_head_dim
                    .ok_or_else(|| miss("linear_value_head_dim"))?;
                for (out_name, out_vals, out_rows) in
                    split_fused_gdn(&name, w, hid, nk, dk, nv, dv)?
                {
                    let two_d = out_rows * hid >= GROUP_SIZE && !force_f16(&out_name);
                    let (dt, data) = if two_d {
                        quantize_2d(quant, &out_vals, out_rows, hid)
                    } else {
                        (TensorDtype::F16, encode_f16(&out_vals))
                    };
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
            if name.ends_with(".self_attn.qkv_proj.weight")
                || name.ends_with(".mlp.gate_up_proj.weight")
            {
                anyhow::ensure!(m_shape.len() == 2, "fused '{name}': expected 2-D");
                let w = &m_vals;
                let (rows, cols) = (m_shape[0], m_shape[1]);
                let parts: Vec<(String, usize, usize)> = if name.contains("qkv_proj") {
                    let q = arch.num_attention_heads * arch.head_dim;
                    let kv = arch.num_kv_heads * arch.head_dim;
                    anyhow::ensure!(
                        q + 2 * kv == rows,
                        "'{name}': {rows} rows != q({q}) + 2·kv({kv})"
                    );
                    vec![
                        (name.replace("qkv_proj", "q_proj"), 0, q),
                        (name.replace("qkv_proj", "k_proj"), q, kv),
                        (name.replace("qkv_proj", "v_proj"), q + kv, kv),
                    ]
                } else {
                    anyhow::ensure!(rows % 2 == 0, "'{name}': odd row count {rows}");
                    vec![
                        (name.replace("gate_up_proj", "gate_proj"), 0, rows / 2),
                        (name.replace("gate_up_proj", "up_proj"), rows / 2, rows / 2),
                    ]
                };
                for (out_name, r0, nr) in parts {
                    let vals = &w[r0 * cols..(r0 + nr) * cols];
                    let (dt, data) = if nr * cols >= GROUP_SIZE && !force_f16(&out_name) {
                        quantize_2d(quant, vals, nr, cols)
                    } else {
                        (TensorDtype::F16, encode_f16(vals))
                    };
                    tensors.push(TensorSpec {
                        name: out_name,
                        dtype: dt,
                        shape: vec![nr, cols],
                        data,
                    });
                }
                continue;
            }
            // Gemma-4 global (full-attention) layers carry no v_proj —
            // V is the K projection, normalized separately at runtime
            // (attention_k_eq_v). Materialize the duplicate so the
            // runtime keeps its uniform Q/K/V/O contract; the overlay
            // costs one MQA-sized tensor per global layer.
            if arch.global_head_dim.is_some() && name.ends_with(".self_attn.k_proj.weight") {
                let li: Option<usize> = name
                    .split(".layers.")
                    .nth(1)
                    .and_then(|r| r.split('.').next())
                    .and_then(|n| n.parse().ok());
                let pat = arch.sliding_window_pattern.unwrap_or(usize::MAX);
                if let Some(li) = li {
                    if (li + 1) % pat == 0 {
                        anyhow::ensure!(m_shape.len() == 2, "'{name}': expected 2-D");
                        let w = &m_vals;
                        let (rows, cols) = (m_shape[0], m_shape[1]);
                        for out_name in [name.clone(), name.replace("k_proj", "v_proj")] {
                            let (dt, data) = if rows * cols >= GROUP_SIZE && !force_f16(&out_name) {
                                quantize_2d(quant, w, rows, cols)
                            } else {
                                (TensorDtype::F16, encode_f16(w))
                            };
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
                        let (dt, data) = if two_d {
                            quantize_2d(quant, &out_vals, out_shape[0], out_shape[1])
                        } else {
                            (TensorDtype::F16, encode_f16(&out_vals))
                        };
                        tensors.push(TensorSpec {
                            name,
                            dtype: dt,
                            shape: out_shape,
                            data,
                        });
                        continue;
                    }
                }
            }
            let vals = m_vals;
            let numel: usize = m_shape.iter().product();
            if numel != vals.len() {
                anyhow::bail!(
                    "tensor '{name}': {} values for shape {:?}",
                    vals.len(),
                    m_shape
                );
            }
            // 1-D tensors, tiny tensors, non-2-D, and gate-critical projections go f16.
            // Index tables ride as raw f32: quantizing them would round
            // expert ids, and f16 cannot even hold a vocabulary id exactly.
            if force_f32(&name) {
                tensors.push(TensorSpec {
                    name,
                    dtype: TensorDtype::F32,
                    shape: m_shape.clone(),
                    data: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                });
                continue;
            }
            let two_d = m_shape.len() == 2 && numel >= GROUP_SIZE && !force_f16(&name);
            // The q2tp profile covers the gate/up planes of EVERY expert.
            // Checkpoints that pack their experts into one 3-D tensor are
            // handled above; the ones that ship a tensor per expert (DeepSeek
            // among them) arrive here, and reading this condition as
            // "shared expert only" left the routed experts — which are
            // essentially the whole model — at 4 bits. That is a 50% size
            // miss on a 300B MoE, and it looks like nothing but a large file.
            //
            // The shared expert additionally MUST match the routed layout: it
            // rides in the same packed buffer (last slot) and the MoE kernels
            // index that buffer with one per-expert stride, so a mismatch
            // makes the whole graph decline, silently, into a CPU MoE.
            let expert_gu = (name.contains(".experts.") || name.contains(".shared_expert."))
                && (name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight"));
            let q_here = if expert_gu { gu_quant } else { quant };
            let (dt, data) = if two_d {
                quantize_2d(q_here, &vals, m_shape[0], m_shape[1])
            } else {
                (TensorDtype::F16, encode_f16(&vals))
            };
            tensors.push(TensorSpec {
                name,
                dtype: dt,
                shape: m_shape.clone(),
                data,
            });
        }
    
        Ok(())
    };
    if let Some(repo) = stream_repo.clone() {
        let base = format!("https://huggingface.co/{repo}/resolve/main");
        let threads = hf_threads();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(20))
            .timeout_read(Duration::from_secs(300))
            .build();
        let ns = stream_shards.len().max(1);
        for (si, sname) in stream_shards.iter().enumerate() {
            eprintln!("  [stream {}/{}] {sname}", si + 1, ns);
            fetch(
                &agent,
                &format!("{base}/{sname}"),
                &dir.join(sname),
                hf_token,
                true,
                threads,
            )?;
            let f = open_safetensors(&dir.join(sname))?;
            let one = [f];
            let ft = one[0].tensors.len().max(1);
            let mut fd = 0usize;
            let mut sub = |p: f32| progress((si as f32 + p) / ns as f32);
            process_file(&one[0], &one, &mut tensors, &mut fd, ft, &mut sub)?;
            // Spill this shard's payloads before touching the next one:
            // that is what keeps residency at one shard instead of the
            // whole model.
            drain_to_writer(&mut tensors, &mut writer)?;
            drop(one);
            let _ = fs::remove_file(dir.join(sname));
        }
    } else {
        for file in &files {
            process_file(file, &files, &mut tensors, &mut done, total, &mut progress)?;
        }
    }

    // Tokenizer + chat bundle (optional but recommended).
    let tok_cfg: serde_json::Value = fs::read(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    let vocab = fs::read(dir.join("tokenizer.json")).ok().or_else(|| {
        // Kimi (tiktoken): synthesize a standard tokenizer.json from the
        // rank table so the runtime tokenizer stays unchanged.
        let tk = fs::read_to_string(dir.join("tiktoken.model")).ok()?;
        match tiktoken_to_tokenizer_json(&tk, &tok_cfg) {
            Ok(j) => {
                tracing::info!("tiktoken.model → tokenizer.json ({} bytes)", j.len());
                Some(j.into_bytes())
            }
            Err(e) => {
                eprintln!("  tiktoken.model found but conversion failed: {e}");
                None
            }
        }
    });
    let gen_cfg: serde_json::Value = fs::read(dir.join("generation_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    // Sidecar `chat_template.jinja` first, then the tokenizer_config field;
    // ignore an empty/blank file so we correctly fall through to the config.
    let chat_template = fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            tok_cfg
                .get("chat_template")
                .and_then(|v| v.as_str().map(String::from))
        });
    let bundle = TokenizerBundle {
        chat_template,
        eos_token_ids: eos_ids(&gen_cfg, &config),
        bos_token_id: config
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        pad_token_id: config
            .get("pad_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
    };

    let quant_type = match quant {
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
        Quant::Vbit => QuantType::Vbit,
        Quant::Q4Tiled | Quant::Q4TiledP | Quant::Q2TiledP => QuantType::Q4Block,
        // File-level label only (per-tensor truth is in the directory);
        // Vbit is the closest existing informational bucket for q1.
        Quant::Q1 | Quant::Q1p | Quant::Q1s | Quant::Q1t => QuantType::Vbit,
    };
    let provenance = match &defrag_plan {
        Some(plan) => {
            let kept: Vec<usize> = (0..arch.num_layers)
                .map(|li| {
                    plan.keep
                        .get(&li)
                        .map(|k| k.iter().filter(|&&b| b).count())
                        .unwrap_or(orig_inter)
                })
                .collect();
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
    tensors.sort_by(|a, b| {
        exec_order_key(&a.name)
            .cmp(&exec_order_key(&b.name))
            .then_with(|| a.name.cmp(&b.name))
    });

    // Anything still resident (small tensors produced outside the main
    // loop) is appended last.
    drain_to_writer(&mut tensors, &mut writer)?;
    writer
        .finish(&header, None, vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    // The manifest only exists to rescue an interrupted run.
    let _ = std::fs::remove_file(&manifest_path);
    progress(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::format::CmfModel;
    use cortiq_core::quant::{
        dequant_q4_block, dequant_q4_tiled, dequant_q4tp, dequant_q8_2f, dequant_q8_row,
        dequant_vbit, expected_nbytes,
    };

    /// Deterministic pseudo-random weights whose per-TILE amplitude spreads
    /// over `spread` powers of two within each row — that spread is the only
    /// thing a per-row scale ladder has to cope with, so it is the axis worth
    /// parameterizing. Measured on KAT-Coder-V2.5, real rows sit at a median
    /// spread of 1.27 and a 90th percentile of 1.89; 3.7 is past the tail.
    fn synth_rows(rows: usize, cols: usize, spread: f32) -> Vec<f32> {
        let mut v = vec![0f32; rows * cols];
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8388608.0 - 1.0
        };
        for r in 0..rows {
            let amp = 10f32.powi((r % 7) as i32 - 3); // rows span 10^-3..10^3
            for g in 0..cols / GROUP_SIZE {
                let tilt = (next().abs() * spread).exp2();
                for k in 0..GROUP_SIZE {
                    v[r * cols + g * GROUP_SIZE + k] = next() * amp * tilt;
                }
            }
        }
        v
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            num += ((x - y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
        (num / den).sqrt() as f32
    }

    /// Rows spanning 10^-3..10^3 with a per-group tilt, so the row ladder
    /// has to work for the scales to come back at all.
    fn q2tp_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut v = vec![0f32; rows * cols];
        let mut s = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8388608.0 - 1.0
        };
        for r in 0..rows {
            let amp = 10f32.powi((r % 7) as i32 - 3);
            for g in 0..cols / GROUP_SIZE {
                let tilt = (next().abs() * 2.0).exp2();
                for k in 0..GROUP_SIZE {
                    v[r * cols + g * GROUP_SIZE + k] = next() * amp * tilt;
                }
            }
        }
        v
    }

    fn q2tp_rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            num += ((x - y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
        (num / den).sqrt() as f32
    }

    /// The 2-bit plane must be exactly half of q4tp's, with the params and
    /// code planes untouched — a wrong section offset reads scales out of
    /// weight bytes and still produces fluent-looking numbers, so pin the
    /// length AND the reader's own view of it.
    /// E4M3 has no infinities and one NaN; the values below are the
    /// bit patterns that pin its exponent bias and its subnormal step.
    /// DeepSeek-V4 names arrive with no `model.` wrapper and spell the
    /// blocks `attn`/`ffn`. Pin the whole map: a silent miss here means a
    /// tensor lands under a name nothing reads, and the model loads with
    /// a hole instead of failing.
    /// The 2-bit profile is worth 50 GB on a 300B MoE, and getting it wrong
    /// produces a file that is merely large — no error, no warning. These are
    /// the names it has to recognize, in the spelling `canon_name` emits.
    /// The row encoders were made parallel because a 300B MoE otherwise
    /// quantizes on one core. Rows are independent, so the bytes must be
    /// identical however many threads produce them — not merely close.
    #[test]
    fn parallel_row_encoding_is_byte_identical_to_serial() {
        // Wide enough to split across threads, with a dead row and a row of
        // wildly different magnitudes so the scale ladder actually varies.
        let (rows, cols) = (97usize, 256usize);
        let mut vals = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                vals[r * cols + c] = match r {
                    7 => 0.0, // a dead row: rung 0 everywhere
                    _ => ((r * 31 + c * 7) as f32 * 0.017).sin() * (1 << (r % 9)) as f32,
                };
            }
        }
        for (name, enc) in [
            ("q2tp", encode_q2tp as fn(&[f32], usize, usize) -> Vec<u8>),
            ("q4tp", encode_q4tp as fn(&[f32], usize, usize) -> Vec<u8>),
        ] {
            // SAFETY-of-test: the env var is read per call, and these run
            // sequentially within one test.
            unsafe { std::env::set_var("CMF_ENCODE_THREADS", "1") };
            let serial = enc(&vals, rows, cols);
            unsafe { std::env::set_var("CMF_ENCODE_THREADS", "8") };
            let parallel = enc(&vals, rows, cols);
            unsafe { std::env::remove_var("CMF_ENCODE_THREADS") };
            assert_eq!(
                serial.len(),
                parallel.len(),
                "{name}: length differs between 1 and 8 threads"
            );
            let diff = serial
                .iter()
                .zip(&parallel)
                .position(|(a, b)| a != b);
            assert!(
                diff.is_none(),
                "{name}: byte {} differs between 1 and 8 threads",
                diff.unwrap()
            );
        }
    }

    #[test]
    fn the_two_bit_profile_covers_every_experts_gate_and_up() {
        let gu = |name: &str| {
            (name.contains(".experts.") || name.contains(".shared_expert."))
                && (name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight"))
        };
        // routed experts — the bulk of the model
        assert!(gu("model.layers.7.mlp.experts.42.gate_proj.weight"));
        assert!(gu("model.layers.7.mlp.experts.42.up_proj.weight"));
        // the shared expert, which must match the routed layout
        assert!(gu("model.layers.7.mlp.shared_expert.gate_proj.weight"));
        assert!(gu("model.layers.7.mlp.shared_expert.up_proj.weight"));
        // down stays 4-bit, and nothing outside the experts is touched
        assert!(!gu("model.layers.7.mlp.experts.42.down_proj.weight"));
        assert!(!gu("model.layers.7.mlp.shared_expert.down_proj.weight"));
        assert!(!gu("model.layers.7.mlp.gate.weight"));
        assert!(!gu("model.layers.7.self_attn.wq_b.weight"));
        assert!(!gu("model.embed_tokens.weight"));
    }

    #[test]
    fn deepseek_v4_names_map_onto_the_loader_layout() {
        let m = |s: &str| canon_name(s).unwrap();
        assert_eq!(m("embed.weight"), "model.embed_tokens.weight");
        assert_eq!(m("head.weight"), "lm_head.weight");
        assert_eq!(m("norm.weight"), "model.norm.weight");
        assert_eq!(m("hc_head_scale"), "model.hc_head_scale");
        assert_eq!(
            m("layers.7.attn_norm.weight"),
            "model.layers.7.input_layernorm.weight"
        );
        assert_eq!(
            m("layers.7.ffn_norm.weight"),
            "model.layers.7.post_attention_layernorm.weight"
        );
        // router + the noaux_tc bias + the hash table
        assert_eq!(m("layers.7.ffn.gate.weight"), "model.layers.7.mlp.gate.weight");
        assert_eq!(m("layers.7.ffn.gate.bias"), "model.layers.7.mlp.expert_bias");
        assert_eq!(m("layers.7.ffn.gate.tid2eid"), "model.layers.7.mlp.tid2eid");
        // w1 = gate, w3 = up, w2 = down — for routed AND shared experts
        assert_eq!(
            m("layers.7.ffn.experts.42.w1.weight"),
            "model.layers.7.mlp.experts.42.gate_proj.weight"
        );
        assert_eq!(
            m("layers.7.ffn.experts.42.w3.weight"),
            "model.layers.7.mlp.experts.42.up_proj.weight"
        );
        assert_eq!(
            m("layers.7.ffn.experts.42.w2.scale"),
            "model.layers.7.mlp.experts.42.down_proj.scale"
        );
        assert_eq!(
            m("layers.7.ffn.shared_experts.w1.weight"),
            "model.layers.7.mlp.shared_expert.gate_proj.weight"
        );
        // attention keeps its own spelling under self_attn — the double
        // LoRA, the compressed KV, the sink, the compressor and the
        // indexer have no equivalent in a supported arch yet
        assert_eq!(
            m("layers.7.attn.wq_a.weight"),
            "model.layers.7.self_attn.wq_a.weight"
        );
        assert_eq!(
            m("layers.7.attn.indexer.weights_proj.weight"),
            "model.layers.7.self_attn.indexer.weights_proj.weight"
        );
        assert_eq!(
            m("layers.7.attn.compressor.wkv.weight"),
            "model.layers.7.self_attn.compressor.wkv.weight"
        );
        assert_eq!(m("layers.7.hc_attn_scale"), "model.layers.7.hc_attn_scale");
    }

    #[test]
    fn fp8_e4m3_decodes_the_ocp_reference_points() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0); // exp 7 → 2^0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5); // 1 + 4/8
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), (2f32).powi(-9)); // smallest subnormal
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0); // largest finite
    }

    /// The block plane must be indexed by TILE, not by element: a wrong
    /// stride still produces plausible numbers, which is how a silently
    /// wrong conversion happens.
    #[test]
    fn fp8_block_scales_apply_per_tile() {
        let (rows, cols, block) = (4usize, 4usize, 2usize);
        // every weight is 1.0; scales differ per 2x2 tile: 2^0, 2^1 / 2^2, 2^3
        let packed = vec![0x38u8; rows * cols];
        let scales = vec![127u8, 128, 129, 130];
        let out = unpack_fp8_blocks(&packed, &scales, rows, cols, block).unwrap();
        assert_eq!(out[0], 1.0); // tile (0,0)
        assert_eq!(out[2], 2.0); // tile (0,1)
        assert_eq!(out[2 * cols], 4.0); // tile (1,0)
        assert_eq!(out[2 * cols + 2], 8.0); // tile (1,1)
        // a NaN scale must fail rather than poison the tensor
        let bad = vec![255u8, 128, 129, 130];
        assert!(unpack_fp8_blocks(&packed, &bad, rows, cols, block).is_err());
    }

    #[test]
    fn q2tp_payload_length_and_sections_match_the_reader() {
        use cortiq_core::quant::{expected_nbytes, q2tp_sections, validate_payload};
        let (rows, cols) = (7usize, 96usize);
        let v = q2tp_test_matrix(rows, cols);
        let blob = encode_q2tp(&v, rows, cols);
        let want = expected_nbytes(TensorDtype::Q2TiledP, &[rows, cols]).unwrap();
        assert_eq!(blob.len(), want, "payload length");
        let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
        assert_eq!(params_off, rows * (cols / GROUP_SIZE) * 8, "weight plane");
        assert_eq!(codes_off, params_off + rows * 4, "params plane");
        assert_eq!(codes_off + rows * stride, blob.len(), "code plane");
        assert!(
            validate_payload(TensorDtype::Q2TiledP, &[rows, cols], &blob).is_ok(),
            "reader rejected its own encoder's output"
        );
    }

    /// 2 bits carry 4 levels, so the error is large by construction — what
    /// must hold is that it stays in the 2-bit BAND (a broken decode lands
    /// at ~100%) and that an all-zero row survives exactly.
    #[test]
    fn q2tp_roundtrip_stays_in_the_two_bit_error_band() {
        use cortiq_core::quant::dequant_q2tp;
        let (rows, cols) = (9usize, 128usize);
        let mut v = q2tp_test_matrix(rows, cols);
        for k in 0..cols {
            v[3 * cols + k] = 0.0;
        }
        let blob = encode_q2tp(&v, rows, cols);
        let mut back = vec![0f32; rows * cols];
        dequant_q2tp(&blob, rows, cols, &mut back);
        let e = q2tp_rel_rms(&back, &v);
        assert!(e < 0.55, "q2tp rel-RMS {e} — decode is wrong, not just coarse");
        assert!(e > 0.05, "q2tp rel-RMS {e} — suspiciously exact for 2 bits");
        assert!(
            back[3 * cols..4 * cols].iter().all(|x| *x == 0.0),
            "an all-zero row must dequantize to exact zeros"
        );
    }

    #[test]
    fn q4tp_payload_length_matches_expected_nbytes() {
        for &(rows, cols) in &[(1usize, 32usize), (7, 64), (16, 2048), (2048, 512)] {
            let v = synth_rows(rows, cols, 1.27);
            let bytes = encode_q4tp(&v, rows, cols);
            assert_eq!(
                Some(bytes.len()),
                expected_nbytes(TensorDtype::Q4TiledP, &[rows, cols]),
                "shape {rows}x{cols}"
            );
        }
    }

    #[test]
    fn q4tp_codes_survive_bit_packing_at_every_offset() {
        // 5 bits per code means the byte boundary lands differently for each
        // tile mod 8; walk a full period plus change.
        let gpr = 37;
        let mut buf = vec![0u8; q4tp_code_stride(gpr)];
        let want: Vec<usize> = (0..gpr).map(|g| (g * 7 + 3) % 32).collect();
        for (g, &c) in want.iter().enumerate() {
            q4tp_put_code(&mut buf, g, c);
        }
        let got: Vec<usize> = (0..gpr)
            .map(|g| cortiq_core::quant::q4tp_code(&buf, g))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn q4tp_costs_almost_nothing_against_q4t_from_the_same_source() {
        // The honest comparison is BOTH codecs against the same fp32 weights.
        // Comparing q4tp against q4t's output instead measures the distance
        // between two representations, which reads far worse than the truth:
        // when the scale shifts a little the nibbles simply re-round, so the
        // total error stays pinned to the 4-bit grid either way.
        let (rows, cols) = (256usize, 2048usize);
        for &(spread, budget) in &[(0.0f32, 1.02f32), (1.27, 1.02), (1.89, 1.03), (3.70, 1.06)] {
            let v = synth_rows(rows, cols, spread);
            let mut a = vec![0f32; rows * cols];
            dequant_q4_tiled(&encode_q4_tiled(&v, rows, cols), &mut a);
            let mut b = vec![0f32; rows * cols];
            dequant_q4tp(&encode_q4tp(&v, rows, cols), rows, cols, &mut b);

            let (e_q4t, e_q4tp) = (rel_rms(&v, &a), rel_rms(&v, &b));
            assert!(
                e_q4tp <= e_q4t * budget,
                "spread {spread}: q4tp {e_q4tp:.4} over budget {budget}× of q4t {e_q4t:.4}"
            );
        }
    }

    #[test]
    fn q4tp_keeps_an_all_zero_tile_exactly_zero() {
        // A dead tile must not drag the row's ladder down (that would coarsen
        // every live tile) and must still dequantize to exact zeros.
        let (rows, cols) = (4usize, 128usize);
        let mut v = synth_rows(rows, cols, 1.27);
        for c in 0..GROUP_SIZE {
            v[1 * cols + c] = 0.0;
        }
        let mut out = vec![0f32; rows * cols];
        dequant_q4tp(&encode_q4tp(&v, rows, cols), rows, cols, &mut out);
        assert!(out[cols..cols + GROUP_SIZE].iter().all(|&x| x == 0.0));

        let live = &v[cols + GROUP_SIZE..2 * cols];
        let got = &out[cols + GROUP_SIZE..2 * cols];
        assert!(
            rel_rms(live, got) < 0.15,
            "dead tile stretched the row's ladder"
        );
    }

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
        assert_eq!(
            arch.layer_types,
            vec![
                LayerType::FullAttention,
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
            ]
        );
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
        assert_eq!(
            canon_name("model.layers.1.mlp.experts.e_score_correction_bias").as_deref(),
            Some("model.layers.1.mlp.expert_bias")
        );
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
        names.sort_by(|a, b| {
            exec_order_key(a)
                .cmp(&exec_order_key(b))
                .then_with(|| a.cmp(b))
        });
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
            let amp = vals[o * cols..(o + 1) * cols]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-6);
            for i in 0..cols {
                let e = (dec[o * cols + i] - vals[o * cols + i]).abs();
                assert!(
                    e <= amp * 0.2,
                    "row {o} col {i}: err {e} vs amp {amp} (bits {})",
                    bits[o]
                );
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
        assert_eq!(
            dec, dec_legacy,
            "vbit_ro must reconstruct exactly like vbit"
        );
    }

    #[test]
    fn tiktoken_roundtrip_synthetic() {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        // Mini rank table: single bytes first, then pairs — exactly the
        // shape tiktoken ships (ranks are the merge order).
        let toks: Vec<&[u8]> = vec![
            b"h", b"e", b"l", b"o", b" ", b"w", b"r", b"d", b"he", b"ll", b"lo",
            b"hell", b"hello", b" w", b"or", b" wor", b" world",
        ];
        let model: String = toks
            .iter()
            .enumerate()
            .map(|(r, t)| format!("{} {}\n", b64.encode(t), r))
            .collect();
        let cfg = serde_json::json!({
            "added_tokens_decoder": {
                "100": {"content": "[BOS]"},
                "101": {"content": "[EOS]"}
            }
        });
        let json = tiktoken_to_tokenizer_json(&model, &cfg).unwrap();
        let tok = cortiq_engine::tokenizer::Tokenizer::from_json(&json).unwrap();
        let ids = tok.encode("hello world");
        // "hello" merges to one token (rank 12). " world"(16) is BPE-
        // UNREACHABLE with this table (merge path stops at " wor"+l+d) —
        // real tiktoken tokenizes it identically, which is exactly the
        // semantics the recovered merge list must reproduce.
        let strip: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&i| i != 100)
            .collect(); // drop a possible auto-BOS
        assert_eq!(strip, vec![12, 15, 2, 7], "ids: {ids:?}");
        assert_eq!(tok.decode(&strip), "hello world");
    }

    #[test]
    fn tiktoken_real_model_roundtrip() {
        // Gated: point CMF_TIKTOKEN at a real tiktoken.model (+ optional
        // CMF_TIKTOKEN_CFG tokenizer_config.json) to validate against the
        // shipped table. Skipped otherwise.
        let Ok(path) = std::env::var("CMF_TIKTOKEN") else {
            return;
        };
        let model = std::fs::read_to_string(&path).unwrap();
        let cfg: serde_json::Value = std::env::var("CMF_TIKTOKEN_CFG")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let t0 = std::time::Instant::now();
        let json = tiktoken_to_tokenizer_json(&model, &cfg).unwrap();
        eprintln!("converted in {:?}, json {} MB", t0.elapsed(), json.len() / 1_000_000);
        let tok = cortiq_engine::tokenizer::Tokenizer::from_json(&json).unwrap();
        for text in [
            "Hello, world!",
            "Kimi Linear is a hybrid attention architecture.",
            "Пример текста на русском языке 123",
            "你好，世界！混合注意力架构。",
            "def main():\n    print(\"ok\")  # comment\n",
            "   spaces    and\ttabs\n\n",
        ] {
            let ids = tok.encode(text);
            let back = tok.decode(&ids.iter().copied().filter(|&i| Some(i) != tok.bos_token_id).collect::<Vec<_>>());
            assert_eq!(back, text, "roundtrip failed for {text:?} → {ids:?}");
            eprintln!("{:3} toks | {text:?}", ids.len());
        }
    }

    #[test]
    fn mxfp4_unpack_roundtrip() {
        // Values ON the E2M1 grid times power-of-two scales roundtrip
        // exactly; low nibble = even element (compressed-tensors pack).
        let grid = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let rows = 2usize;
        let cols = 64usize; // two groups of 32 per row
        let scales_exp = [[127u8, 130u8], [124u8, 127u8]]; // ×1, ×8, ×0.125, ×1
        let mut want = vec![0.0f32; rows * cols];
        let mut packed = vec![0u8; rows * cols / 2];
        let mut scales = vec![0u8; rows * 2];
        for r in 0..rows {
            for g in 0..2 {
                let k = scales_exp[r][g];
                scales[r * 2 + g] = k;
                let sc = (k as f32 - 127.0).exp2();
                for i in 0..32 {
                    let nib = ((r * 31 + g * 7 + i * 3) % 16) as u8;
                    let mag = grid[(nib & 7) as usize];
                    let v = if nib & 8 != 0 { -mag } else { mag };
                    want[r * cols + g * 32 + i] = v * sc;
                    let byte = &mut packed[r * cols / 2 + g * 16 + i / 2];
                    if i % 2 == 0 {
                        *byte |= nib;
                    } else {
                        *byte |= nib << 4;
                    }
                }
            }
        }
        let got = unpack_mxfp4(&packed, &scales, rows, cols / 2).unwrap();
        assert_eq!(got, want);
        // NaN scale (E8M0 255) must refuse, not propagate.
        let mut bad = scales.clone();
        bad[0] = 255;
        assert!(unpack_mxfp4(&packed, &bad, rows, cols / 2).is_err());
    }

    #[test]
    fn minicpm3_arch_scales_and_factors() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{
            "model_type": "minicpm3",
            "hidden_size": 2560, "num_hidden_layers": 62,
            "num_attention_heads": 40, "intermediate_size": 6400,
            "vocab_size": 73448, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "q_lora_rank": 768, "kv_lora_rank": 256,
            "qk_rope_head_dim": 32, "qk_nope_head_dim": 64, "v_head_dim": 64,
            "scale_emb": 12, "scale_depth": 1.4, "dim_model_base": 256,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.05, 1.12, 1.25, 1.53, 2.09, 3.14, 4.93, 7.52,
                                 10.47, 13.06, 14.85, 15.90, 16.46, 16.74, 16.87, 16.94],
                "long_factor":  [1.05, 1.12, 1.25, 1.53, 2.09, 3.14, 4.93, 7.52,
                                 10.47, 13.06, 14.85, 15.90, 16.46, 16.74, 16.87, 16.94],
                "original_max_position_embeddings": 32768
            }
        }"#,
        )
        .unwrap();
        let arch = build_arch(&cfg).unwrap();
        assert_eq!(arch.embed_multiplier, 12.0, "scale_emb");
        assert_eq!(arch.logit_multiplier, Some(0.1), "dim_model_base/hidden");
        let fac = arch.rope_freq_factors.as_ref().expect("short factors carried");
        assert_eq!(fac.len(), 16);
        assert!((fac[15] - 16.94).abs() < 1e-9);
        assert!(arch.tie_word_embeddings, "no lm_head tensor — tied");
        let mla = arch.mla.as_ref().expect("MLA");
        assert_eq!(mla.q_lora_rank, Some(768), "compressed q");
        assert!(!mla.nope);
        assert_eq!(arch.max_position_embeddings, 32768, "native window");
    }

    #[test]
    fn kimi_canon_names() {
        let c = |s: &str| canon_name(s).unwrap();
        // K3 wrapper prefix + mixtral-style expert names → canonical MoE.
        assert_eq!(
            c("language_model.model.layers.3.block_sparse_moe.experts.5.w1.weight"),
            "model.layers.3.mlp.experts.5.gate_proj.weight"
        );
        assert_eq!(
            c("model.layers.2.block_sparse_moe.experts.0.w2.weight"),
            "model.layers.2.mlp.experts.0.down_proj.weight"
        );
        assert_eq!(
            c("model.layers.2.block_sparse_moe.experts.0.w3.weight"),
            "model.layers.2.mlp.experts.0.up_proj.weight"
        );
        assert_eq!(
            c("model.layers.2.block_sparse_moe.gate.weight"),
            "model.layers.2.mlp.gate.weight"
        );
        assert_eq!(
            c("model.layers.2.block_sparse_moe.gate.e_score_correction_bias"),
            "model.layers.2.mlp.expert_bias"
        );
        assert_eq!(
            c("model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight"),
            "model.layers.2.mlp.shared_expert.gate_proj.weight"
        );
        // Vision tower dropped (text tower converts alone).
        assert_eq!(canon_name("vision_tower.encoder.blocks.0.wqkv.weight"), None);
        assert_eq!(canon_name("mm_projector.proj.0.weight"), None);
    }

    #[test]
    fn kimi_linear_arch_layers_and_geometry() {
        // Kimi-Linear-48B shape in miniature: 8 layers, full attention on
        // the 1-BASED layers [4, 8], KDA elsewhere; MLA NoPE; sigmoid MoE.
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{
            "model_type": "kimi_linear",
            "hidden_size": 2304, "num_hidden_layers": 8,
            "num_attention_heads": 32, "num_key_value_heads": 32,
            "intermediate_size": 9216, "vocab_size": 1000,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128, "v_head_dim": 128,
            "mla_use_nope": true,
            "hidden_act": "silu",
            "num_experts": 256, "num_experts_per_token": 8,
            "moe_intermediate_size": 1024, "num_shared_experts": 1,
            "moe_renormalize": true, "moe_router_activation_func": "sigmoid",
            "first_k_dense_replace": 1,
            "linear_attn_config": {
                "full_attn_layers": [4, 8],
                "kda_layers": [1, 2, 3, 5, 6, 7],
                "head_dim": 128, "num_heads": 32,
                "short_conv_kernel_size": 4
            }
        }"#,
        )
        .unwrap();
        let arch = build_arch(&cfg).unwrap();
        assert_eq!(arch.num_layers, 8);
        // 1-based full_attn_layers [4,8] → 0-based indices 3 and 7.
        for (i, t) in arch.layer_types.iter().enumerate() {
            let want_full = i == 3 || i == 7;
            assert_eq!(
                matches!(t, cortiq_core::LayerType::FullAttention),
                want_full,
                "layer {i}"
            );
            assert_eq!(
                matches!(t, cortiq_core::LayerType::Kda),
                !want_full,
                "layer {i}"
            );
        }
        assert_eq!(arch.linear_num_key_heads, Some(32));
        assert_eq!(arch.linear_key_head_dim, Some(128));
        assert_eq!(arch.linear_value_head_dim, Some(128));
        assert_eq!(arch.linear_conv_kernel_dim, Some(4));
        let mla = arch.mla.as_ref().expect("kv_lora_rank → MLA");
        assert!(mla.nope, "mla_use_nope must carry into the header");
        assert_eq!(mla.q_lora_rank, None);
        let moe = arch.moe.as_ref().expect("num_experts → MoE");
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.top_k, 8);
        assert!(moe.router_sigmoid, "sigmoid router");
        assert!(moe.norm_topk_prob, "moe_renormalize");
        assert_eq!(moe.shared_expert_intermediate_size, Some(1024));
        assert!(arch.kda_gate_lower_bound.is_none());
    }

    #[test]
    fn kimi_k3_unsupported_features_bail_honestly() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{
            "model_type": "kimi_k3",
            "text_config": {
                "model_type": "kimi_linear",
                "hidden_size": 7168, "num_hidden_layers": 4,
                "num_attention_heads": 96, "intermediate_size": 33792,
                "vocab_size": 1000, "rms_norm_eps": 1e-5,
                "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128, "v_head_dim": 128,
                "attn_res_block_size": 12,
                "linear_attn_config": {
                    "full_attn_layers": [4],
                    "head_dim": 128, "num_heads": 96,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0
                }
            }
        }"#,
        )
        .unwrap();
        let err = build_arch(&cfg).unwrap_err().to_string();
        assert!(err.contains("attn_res_block_size"), "got: {err}");
    }

    #[test]
    fn fused_gdn_split_is_correct_permutation() {
        // nk=2, dk=3, nv=4 (r=2), dv=2, hid=1. Each source row's value = its flat
        // row index, so we can trace exactly where each row lands after the split.
        let (nk, dk, nv, dv, hid) = (2usize, 3usize, 4usize, 2usize, 1usize);
        let r = nv / nk; // 2
        let gw = 2 * dk + 2 * r * dv; // 6 + 8 = 14
        let w: Vec<f32> = (0..nk * gw * hid).map(|i| i as f32).collect();
        let out =
            split_fused_gdn("m.linear_attn.in_proj_qkvz.weight", &w, hid, nk, dk, nv, dv).unwrap();
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
        let outb =
            split_fused_gdn("m.linear_attn.in_proj_ba.weight", &wb, hid, nk, dk, nv, dv).unwrap();
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
        assert_eq!(
            c("model.layers.0.operator_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            c("model.layers.0.ffn_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            c("model.layers.0.conv.in_proj.weight"),
            "model.layers.0.short_conv.in_proj.weight"
        );
        assert_eq!(
            c("model.layers.0.conv.conv.weight"),
            "model.layers.0.short_conv.conv.weight"
        );
        assert_eq!(
            c("model.layers.0.conv.out_proj.weight"),
            "model.layers.0.short_conv.out_proj.weight"
        );
        // Dense FFN: w1/w3/w2 → gate/up/down.
        assert_eq!(
            c("model.layers.0.feed_forward.w1.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            c("model.layers.0.feed_forward.w3.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            c("model.layers.0.feed_forward.w2.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
        // Attention (full_attention layer 2): out_proj → o_proj, q/k layernorm.
        assert_eq!(
            c("model.layers.2.self_attn.out_proj.weight"),
            "model.layers.2.self_attn.o_proj.weight"
        );
        assert_eq!(
            c("model.layers.2.self_attn.q_layernorm.weight"),
            "model.layers.2.self_attn.q_norm.weight"
        );
        assert_eq!(
            c("model.layers.2.self_attn.k_layernorm.weight"),
            "model.layers.2.self_attn.k_norm.weight"
        );
        // MoE router / bias / experts.
        assert_eq!(
            c("model.layers.2.feed_forward.gate.weight"),
            "model.layers.2.mlp.gate.weight"
        );
        assert_eq!(
            c("model.layers.2.feed_forward.expert_bias"),
            "model.layers.2.mlp.expert_bias"
        );
        assert_eq!(
            c("model.layers.2.feed_forward.experts.7.w1.weight"),
            "model.layers.2.mlp.experts.7.gate_proj.weight"
        );
        assert_eq!(
            c("model.layers.2.feed_forward.experts.7.w2.weight"),
            "model.layers.2.mlp.experts.7.down_proj.weight"
        );
        // Q/K/V projections already canonical — must pass through untouched.
        assert_eq!(
            c("model.layers.2.self_attn.q_proj.weight"),
            "model.layers.2.self_attn.q_proj.weight"
        );
        // A Qwen tensor must be untouched by the LFM2 rewrite.
        assert_eq!(
            c("model.layers.3.mlp.gate_proj.weight"),
            "model.layers.3.mlp.gate_proj.weight"
        );
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
        assert!(
            moe.router_sigmoid,
            "lfm2_moe must route with a sigmoid gate"
        );
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
        let onebit: Vec<f32> = (0..rows * cols)
            .map(|i| if (i * 7 + 3) % 5 < 2 { 0.25 } else { -0.25 })
            .collect();
        assert_eq!(
            encode_q1(&onebit, rows, cols),
            encode_q1_ef(&onebit, rows, cols),
            "error diffusion must be a no-op on a genuinely 1-bit tensor"
        );
    }

    /// Q1S roundtrip: kept outliers come back at f16 precision and the bulk
    /// decodes to the per-group ±s level. Guards the format the holographic
    /// fold will populate.
    #[test]
    fn q1s_roundtrip_restores_outliers_and_binarizes_the_rest() {
        use cortiq_core::quant::dequant_q1s;
        let (rows, cols) = (2usize, 64usize);
        let mut vals: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.017).sin() * 0.1)
            .collect();
        let spikes = [5usize, 40, 70, 120];
        for &i in &spikes {
            vals[i] = if i % 2 == 0 { 3.0 } else { -3.0 };
        }
        let keep = spikes.len() as f32 / (rows * cols) as f32;
        let bytes = encode_q1s(&vals, rows, cols, keep);
        let mut dec = vec![0f32; rows * cols];
        dequant_q1s(&bytes, &mut dec);
        for &i in &spikes {
            assert!(
                (dec[i] - vals[i]).abs() < 0.02,
                "outlier {i}: {} vs {}",
                dec[i],
                vals[i]
            );
        }
        for i in 0..rows * cols {
            if !spikes.contains(&i) {
                assert!(
                    dec[i].abs() < 2.0,
                    "bulk {i} should be a small ±s, got {}",
                    dec[i]
                );
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
            (
                "model.embed_tokens.weight",
                vec![32, 64],
                (0..32 * 64).map(|k| (k as f32 * 0.01).sin()).collect(),
            ),
            ("model.norm.weight", vec![64], vec![1.0f32; 64]),
        ]);
        fs::write(dir.join("model.safetensors"), &st).unwrap();
        let out = dir.join("m.cmf");
        run_convert(
            dir.to_str().unwrap(),
            "q8",
            out.to_str().unwrap(),
            None,
            None,
            None,
            |_| {},
        )
        .unwrap();

        let model = CmfModel::open(&out).unwrap();
        assert_eq!(model.arch().vocab_size, 32);
        assert_eq!(model.arch().num_layers, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Qwen3.6 ships an MTP head the converter used to drop on the floor.
    /// It must survive, and land under the names the loader reads.
    #[test]
    fn mtp_head_is_kept_and_renamed_to_the_loader_layout() {
        let m = |s: &str| canon_name(s);
        assert_eq!(m("mtp.fc.weight").as_deref(), Some("model.mtp.eh_proj.weight"));
        assert_eq!(
            m("mtp.pre_fc_norm_embedding.weight").as_deref(),
            Some("model.mtp.enorm.weight")
        );
        assert_eq!(
            m("mtp.pre_fc_norm_hidden.weight").as_deref(),
            Some("model.mtp.hnorm.weight")
        );
        assert_eq!(m("mtp.norm.weight").as_deref(), Some("model.mtp.norm.weight"));
        // The block's own tensors pass through untouched — including the MoE
        // mlp, which is what this head actually carries.
        assert_eq!(
            m("mtp.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("model.mtp.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            m("mtp.layers.0.mlp.experts.gate_up_proj").as_deref(),
            Some("model.mtp.layers.0.mlp.experts.gate_up_proj")
        );
        assert_eq!(
            m("mtp.layers.0.mlp.shared_expert_gate.weight").as_deref(),
            Some("model.mtp.layers.0.mlp.shared_expert_gate.weight")
        );
        // Vision towers are still dropped.
        assert!(m("visual.blocks.0.attn.qkv.weight").is_none());
    }
}
