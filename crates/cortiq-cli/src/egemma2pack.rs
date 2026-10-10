//! Pack google/embeddinggemma-2 (`model_type: embedding_gemma2`) into one
//! `.cmf` — the text encoder, and beside it the vision and audio towers, so
//! one file serves every modality the model embeds.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here; the generic LLM walk is never
//! touched. Tensor names stay the checkpoint's own (`language_model.*`,
//! `vision_tower.*`, `audio_tower.*`, `embed_vision.*`, `embed_audio.*`) —
//! the runtime (`cortiq_engine::egemma2`) reads them by those names.
//!
//! ## Profiles (`--quant`)
//!
//! * `bf16` (alias `exact`) — every tensor copied byte for byte from the
//!   bf16 release. The f32 reference is computed from these same values, so
//!   this is the parity file.
//! * `f32` — the same values widened to f32 (twice the bytes, no change in
//!   any number; for tooling that wants plain floats).
//! * `q8_2f` (default, also `auto`/`q8`) and `q4tp` — the text encoder's 2-D
//!   matrices on the requested codec. What stays exact, and why:
//!   - every 1-D tensor (the sandwich norms carry weights near 6-8, the
//!     per-layer `layer_scalar`, q/k norms, clip bounds): a few KB;
//!   - `language_model.embedding_projection` (512→768, the head every
//!     output vector goes through, 0.4 M weights);
//!   - the towers' non-matrix planes: the patch position table, the audio
//!     convolutions, `per_dim_scale`, the clip scalars.
//!
//!   Under `q4tp` the token table and the per-layer-input projection are
//!   kept at q8_2f: a token row *is* the residual stream at layer 0, and the
//!   per-layer projection feeds all 24 layers at once — neither error is
//!   averaged away downstream. The towers' matrices take q8_2f under both
//!   quantized profiles. Measured against the float32 reference: the vision
//!   tower at q8_2f keeps every image / video / interleaved embedding at
//!   cosine >= 0.9997 (`tests/egemma2_vision_parity.rs`); a q8_2f audio
//!   tower under the exact text encoder costs 2e-6…1.4e-5 of `1 − cos`
//!   (10 clips and interleaved inputs), the q8_2f text encoder ~1e-4 — the
//!   towers are not where a q8_2f file loses accuracy. The audio
//!   convolutions (4-D subsampling, 3-D depthwise), `per_dim_scale`, the
//!   clip bounds and the output bias stay exact as non-matrices.
//!   `--tensor-quant PATTERN=QUANT` overrides any 2-D matrix (`f16`/`bf16`
//!   there means the exact copy).
//!
//! The tokenizer (`tokenizer.json`) is the VOCAB section. The configs the
//! runtime needs — `config.json`, the sentence-transformers prompt table
//! (`config_sentence_transformers.json`), the pooling config and the
//! image/video/audio processor configs — ride in the header provenance
//! under `embedding_gemma2`, so the file needs no sidecar.
//!
//! `CMF_EGEMMA2_TEXT_ONLY=1` leaves the towers out (a 270 M text encoder).

use crate::convert::{
    Quant, SafeTensors, open_model, quant_name, quantize_2d, tensor_quant_override, to_f32,
};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpecRef};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::Path;

/// The architecture name written into the header (`cortiq info` shows it;
/// the runtime dispatches on it).
pub const ARCH_NAME: &str = cortiq_engine::egemma2::ARCH_NAME;

/// The codec profile of a pack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Profile {
    /// byte-exact bf16 copy
    Bf16,
    /// bf16 widened to f32
    F32,
    /// a quantized text encoder
    Quant(Quant),
}

pub(crate) fn parse_profile(s: &str) -> anyhow::Result<Profile> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "q8_2f" | "q82f" | "q8f" | "q8" => Profile::Quant(Quant::Q8_2f),
        "q4tp" | "q4" => Profile::Quant(Quant::Q4TiledP),
        "bf16" | "exact" => Profile::Bf16,
        "f32" | "fp32" => Profile::F32,
        "f16" | "fp16" => anyhow::bail!(
            "embedding_gemma2: f16 is not offered — the model's activations overflow half \
             precision; use bf16 (exact), f32, q8_2f or q4tp"
        ),
        other => anyhow::bail!(
            "embedding_gemma2: unknown --quant '{other}' (bf16 | f32 | q8_2f | q4tp | auto)"
        ),
    })
}

fn profile_name(p: Profile) -> &'static str {
    match p {
        Profile::Bf16 => "bf16",
        Profile::F32 => "f32",
        Profile::Quant(q) => quant_name(q),
    }
}

/// How one tensor is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Codec {
    /// the source bytes, verbatim
    Exact,
    /// widened to f32
    F32,
    /// a 2-D matrix on a quantizing codec
    Matrix(Quant),
}

fn is_tower(name: &str) -> bool {
    name.starts_with("vision_tower.")
        || name.starts_with("audio_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("embed_audio.")
}

/// The codec of one tensor under `profile` (first rule that applies).
fn codec_for(name: &str, shape: &[usize], profile: Profile) -> Codec {
    let base = match profile {
        Profile::Bf16 => return Codec::Exact,
        Profile::F32 => return Codec::F32,
        Profile::Quant(q) => q,
    };
    // only true matrices quantize; norms, scalars, convs, tables stay exact
    if shape.len() != 2 || shape[0] < 2 {
        return Codec::Exact;
    }
    if let Some(q) = tensor_quant_override(name) {
        return match q {
            Quant::F16 => Codec::Exact,
            q => Codec::Matrix(q),
        };
    }
    if name == "language_model.embedding_projection.weight"
        || name.ends_with(".embedding_projection.weight") && is_tower(name)
    {
        return Codec::Exact;
    }
    if is_tower(name) {
        return Codec::Matrix(Quant::Q8_2f);
    }
    let sensitive = name == "language_model.embed_tokens.weight"
        || name == "language_model.ple.per_layer_model_projection.weight";
    if sensitive && !matches!(base, Quant::Q8_2f | Quant::Q8Row) {
        return Codec::Matrix(Quant::Q8_2f);
    }
    Codec::Matrix(base)
}

fn exact_dtype(src: &str) -> anyhow::Result<TensorDtype> {
    Ok(match src {
        "BF16" => TensorDtype::Bf16,
        "F16" => TensorDtype::F16,
        "F32" => TensorDtype::F32,
        other => anyhow::bail!("embedding_gemma2: unexpected source dtype {other}"),
    })
}

fn read_json(dir: &Path, rel: &str) -> Option<serde_json::Value> {
    std::fs::read(dir.join(rel))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}

/// Convert the checkpoint in `dir` into `output` under the `quant` profile.
pub fn convert(dir: &Path, quant: &str, output: &str) -> anyhow::Result<()> {
    let profile = parse_profile(quant)?;
    let t0 = std::time::Instant::now();
    let config = read_json(dir, "config.json")
        .ok_or_else(|| anyhow::anyhow!("{}: no readable config.json", dir.display()))?;
    let tc = config
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("embedding_gemma2 config: no text_config"))?;
    let st_cfg = read_json(dir, "config_sentence_transformers.json").ok_or_else(|| {
        anyhow::anyhow!(
            "{}: config_sentence_transformers.json is required (the task prompt table)",
            dir.display()
        )
    })?;
    let vocab = std::fs::read(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
    let text_only = std::env::var("CMF_EGEMMA2_TEXT_ONLY").is_ok_and(|v| v == "1");

    let files: Vec<SafeTensors> = open_model(dir)?;
    // (file index, tensor index) sorted by name: a deterministic file
    let mut order: Vec<(usize, usize)> = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        for (ti, _) in f.tensors.iter().enumerate() {
            order.push((fi, ti));
        }
    }
    order.sort_by(|a, b| {
        files[a.0].tensors[a.1]
            .name
            .cmp(&files[b.0].tensors[b.1].name)
    });

    // Encoded payloads live here; exact copies borrow the mmap directly.
    enum Payload {
        Borrowed(usize, usize),
        Owned(Vec<u8>),
    }
    let mut planned: Vec<(String, TensorDtype, Vec<usize>, Payload)> = Vec::new();
    let mut census: std::collections::BTreeMap<String, (usize, u64, u64)> = Default::default();
    let mut skipped = 0usize;
    for &(fi, ti) in &order {
        let m = &files[fi].tensors[ti];
        if text_only && is_tower(&m.name) {
            skipped += 1;
            continue;
        }
        let numel: usize = m.shape.iter().product::<usize>().max(1);
        let codec = codec_for(&m.name, &m.shape, profile);
        let raw = files[fi].bytes(m);
        let (dtype, payload) = match codec {
            Codec::Exact => (exact_dtype(&m.dtype)?, Payload::Borrowed(fi, ti)),
            Codec::F32 => {
                let vals = to_f32(&m.dtype, raw)?;
                let mut out = Vec::with_capacity(vals.len() * 4);
                for v in &vals {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (TensorDtype::F32, Payload::Owned(out))
            }
            Codec::Matrix(q) => {
                let vals = to_f32(&m.dtype, raw)?;
                let (dt, bytes) = quantize_2d(q, &vals, m.shape[0], m.shape[1]);
                (dt, Payload::Owned(bytes))
            }
        };
        let nbytes = match &payload {
            Payload::Borrowed(..) => raw.len(),
            Payload::Owned(b) => b.len(),
        } as u64;
        let group = m.name.split('.').next().unwrap_or("").to_string();
        let e = census.entry(group).or_default();
        e.0 += 1;
        e.1 += numel as u64;
        e.2 += nbytes;
        planned.push((m.name.clone(), dtype, m.shape.clone(), payload));
    }
    anyhow::ensure!(
        planned
            .iter()
            .any(|(n, ..)| n == "language_model.embed_tokens.weight"),
        "embedding_gemma2: language_model.embed_tokens.weight missing — not an EmbeddingGemma 2 checkpoint"
    );
    for (g, (n, w, b)) in &census {
        eprintln!(
            "  {g}: {n} tensors, {:.1} M weights → {:.1} MB",
            *w as f64 / 1e6,
            *b as f64 / 1e6
        );
    }
    if skipped > 0 {
        eprintln!("  towers left out (CMF_EGEMMA2_TEXT_ONLY=1): {skipped} tensors");
    }

    let refs: Vec<TensorSpecRef> = planned
        .iter()
        .map(|(name, dtype, shape, p)| TensorSpecRef {
            name: name.clone(),
            dtype: *dtype,
            shape: shape.clone(),
            data: match p {
                Payload::Borrowed(fi, ti) => files[*fi].bytes(&files[*fi].tensors[*ti]),
                Payload::Owned(b) => b.as_slice(),
            },
        })
        .collect();

    let g = |k: &str, d: u64| tc.get(k).and_then(|v| v.as_u64()).unwrap_or(d) as usize;
    let layers = g("num_hidden_layers", 24);
    let layer_types: Vec<&str> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|s| {
                    if s.as_str() == Some("full_attention") {
                        "FullAttention"
                    } else {
                        "SlidingAttention"
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| vec!["SlidingAttention"; layers]);
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": g("hidden_size", 512),
        "intermediate_size": g("intermediate_size", 2048),
        "num_layers": layers,
        "num_attention_heads": g("num_attention_heads", 4),
        "num_kv_heads": g("num_key_value_heads", 2),
        "head_dim": g("head_dim", 256),
        "vocab_size": g("vocab_size", 262144),
        "layer_types": layer_types,
        "rms_norm_eps": tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        "rope_theta": 10000.0,
        "max_position_embeddings": cortiq_engine::egemma2::MAX_TOKENS,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    let provenance = serde_json::json!({
        "source": dir.display().to_string(),
        "base_model": "google/embeddinggemma-2",
        "quant": profile_name(profile),
        "text_only": text_only,
        "embedding_gemma2": {
            "config": config,
            "sentence_transformers": st_cfg,
            "pooling": read_json(dir, "1_Pooling/config.json"),
            "processor": read_json(dir, "processor_config.json"),
            "preprocessor": read_json(dir, "preprocessor_config.json"),
            "tokenizer_config": read_json(dir, "tokenizer_config.json"),
            "embedding_dim": tc.get("embedding_dim").and_then(|v| v.as_u64()).unwrap_or(768),
            "matryoshka_dims": [768, 512, 256, 128],
            "max_tokens": cortiq_engine::egemma2::MAX_TOKENS,
        },
    });
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match profile {
            Profile::Bf16 => QuantType::BF16,
            Profile::F32 => QuantType::F32,
            Profile::Quant(Quant::Q4TiledP) => QuantType::Q4Block,
            Profile::Quant(_) => QuantType::Q8_2f,
        },
        provenance: Some(provenance),
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
    CmfModel::write_ref(output, &header, &refs, None, Some(&vocab))
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    let size = std::fs::metadata(output)?.len();
    eprintln!(
        "{output}: {} tensors, {:.1} MB ({}) in {:.1}s",
        refs.len(),
        size as f64 / 1e6,
        profile_name(profile),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_profiles_never_quantize() {
        for p in [Profile::Bf16, Profile::F32] {
            let c = codec_for(
                "language_model.layers.0.mlp.up_proj.weight",
                &[2048, 512],
                p,
            );
            assert_ne!(c, Codec::Matrix(Quant::Q8_2f));
        }
    }

    #[test]
    fn quantized_profiles_keep_the_sensitive_planes() {
        let q4 = Profile::Quant(Quant::Q4TiledP);
        assert_eq!(
            codec_for(
                "language_model.layers.3.mlp.gate_proj.weight",
                &[2048, 512],
                q4
            ),
            Codec::Matrix(Quant::Q4TiledP)
        );
        assert_eq!(
            codec_for("language_model.embed_tokens.weight", &[262144, 512], q4),
            Codec::Matrix(Quant::Q8_2f)
        );
        assert_eq!(
            codec_for(
                "language_model.embedding_projection.weight",
                &[768, 512],
                q4
            ),
            Codec::Exact
        );
        assert_eq!(
            codec_for("language_model.layers.3.layer_scalar", &[1], q4),
            Codec::Exact
        );
        assert_eq!(
            codec_for("language_model.norm.weight", &[512], q4),
            Codec::Exact
        );
        assert_eq!(
            codec_for(
                "vision_tower.patch_embedder.position_embedding_table",
                &[2, 10240, 768],
                q4
            ),
            Codec::Exact
        );
        assert_eq!(
            codec_for(
                "vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
                &[3072, 768],
                q4
            ),
            Codec::Matrix(Quant::Q8_2f)
        );
        assert_eq!(
            codec_for(
                "audio_tower.layers.0.feed_forward1.ffw_layer_1.input_min",
                &[],
                q4
            ),
            Codec::Exact
        );
    }

    #[test]
    fn audio_tower_codecs() {
        let q8 = Profile::Quant(Quant::Q8_2f);
        let q4 = Profile::Quant(Quant::Q4TiledP);
        for p in [q8, q4] {
            for (name, shape) in [
                (
                    "audio_tower.layers.3.feed_forward2.ffw_layer_1.linear.weight",
                    &[4096usize, 1024][..],
                ),
                (
                    "audio_tower.layers.3.self_attn.relative_k_proj.weight",
                    &[1024, 1024],
                ),
                (
                    "audio_tower.subsample_conv_projection.input_proj_linear.weight",
                    &[1024, 1024],
                ),
                ("audio_tower.output_proj.weight", &[1536, 1024]),
            ] {
                assert_eq!(
                    codec_for(name, shape, p),
                    Codec::Matrix(Quant::Q8_2f),
                    "{name}"
                );
            }
            for (name, shape) in [
                (
                    "audio_tower.subsample_conv_projection.layer1.conv.weight",
                    &[32usize, 128, 3, 3][..],
                ),
                (
                    "audio_tower.layers.3.lconv1d.depthwise_conv1d.weight",
                    &[1024, 1, 5],
                ),
                ("audio_tower.layers.3.self_attn.per_dim_scale", &[128]),
                ("audio_tower.output_proj.bias", &[1536]),
                ("embed_audio.embedding_projection.weight", &[512, 1536]),
            ] {
                assert_eq!(codec_for(name, shape, p), Codec::Exact, "{name}");
            }
        }
    }

    #[test]
    fn f16_profile_is_refused() {
        assert!(parse_profile("f16").is_err());
        assert_eq!(parse_profile("auto").unwrap(), Profile::Quant(Quant::Q8_2f));
        assert_eq!(parse_profile("bf16").unwrap(), Profile::Bf16);
    }
}
