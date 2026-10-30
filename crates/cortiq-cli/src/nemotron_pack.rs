//! Pack nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (`model_type:
//! nemo_speechlm`, NeMo key layout) into one `.cmf`.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here; the generic LLM walk is never
//! touched. The runtime is `cortiq_engine::nemotron_asr`.
//!
//! ## Layout
//!
//! * Language model (Nemotron-H: 52 blocks, each one mixer — Mamba-2,
//!   attention or MoE — behind its own RMSNorm):
//!   `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`,
//!   `model.layers.{i}.norm.weight`, and per block
//!   - Mamba-2: `model.layers.{i}.mixer.{in_proj,out_proj}.weight`,
//!     `conv1d.weight` `[channels, kernel]`, `conv1d.bias`, `A_log`, `D`,
//!     `dt_bias`, `norm.weight` (the gated group RMSNorm);
//!   - attention: `model.layers.{i}.mixer.{q,k,v,o}_proj.weight`;
//!   - MoE: `model.layers.{i}.mlp.gate.weight`, `…gate.e_score_correction_bias`,
//!     `model.layers.{i}.mlp.experts.{e}.{up,down}_proj.weight` (the packed
//!     `(experts, in, out)` banks split and transposed to `[out, in]`),
//!     `model.layers.{i}.mlp.shared_experts.{up,down}_proj.weight`.
//!   The multi-token-prediction head keeps the same names under `mtp.layers.{j}`.
//! * Audio: the NeMo names under `perception.` — the ASR transformer
//!   encoder, the Sortformer diarization model, the two norms that join them
//!   and the projector; the ASR front end's mel filter bank and window
//!   (`perception.preprocessor.featurizer.{fb,window}`).
//! * Left out: the word-timestamp CTC head (`ctc_timestamp.*`), the
//!   diarization model's duplicate front end, its unused two-speaker-cache
//!   head (`hidden_to_spks`) and the empty `_extra_state` entries.
//!
//! ## Profiles (`--quant`)
//!
//! * `q8_2f` (default, `auto`): every language-model matrix q8_2f.
//! * `q4tp`: the routed experts q4tp; the always-active skeleton (Mamba-2
//!   and attention projections, shared experts, vocabulary edges) q8_2f.
//! * `q2tp`: the routed experts' up projection q2tp, down q4tp; the rest as
//!   `q4tp`.
//!
//! In every profile the audio towers, the router, the norms, the Mamba-2
//! scalars and the convolution stay exact (bf16/f32 as in the release).
//! `--tensor-quant PATTERN=QUANT` overrides any language-model matrix.
//!
//! The tokenizer (`tokenizer.json`) is the VOCAB section; the configs the
//! runtime needs ride in the header provenance under `nemotron_transcribe`.

use crate::convert::{
    Quant, open_safetensors, quant_name, quantize_2d, tensor_quant_override, to_f32,
};
use cortiq_core::format::{CmfHeader, CmfStreamWriter};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::Path;

pub const ARCH_NAME: &str = cortiq_engine::nemotron_asr::ARCH_NAME;

/// Is `config` a Nemotron 3.5 Transcribe checkpoint?
pub fn detect(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(|v| v.as_str()) == Some("nemo_speechlm")
        && config
            .get("architectures")
            .and_then(|v| v.as_array())
            .is_some_and(|a| {
                a.iter().any(|s| {
                    s.as_str()
                        .is_some_and(|s| s.starts_with("Nemotron3_5Transcribe"))
                })
            })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Profile {
    Q8,
    Q4,
    Q2,
}

fn parse_profile(s: &str) -> anyhow::Result<Profile> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "q8_2f" | "q82f" | "q8" => Profile::Q8,
        "q4tp" | "q4" => Profile::Q4,
        "q2tp" | "q2" => Profile::Q2,
        other => anyhow::bail!(
            "nemotron_transcribe: unknown --quant '{other}' (q8_2f | q4tp | q2tp | auto)"
        ),
    })
}

fn profile_name(p: Profile) -> &'static str {
    match p {
        Profile::Q8 => "q8_2f",
        Profile::Q4 => "q4tp",
        Profile::Q2 => "q2tp",
    }
}

/// How one output tensor is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Codec {
    /// the source bytes, verbatim
    Exact,
    /// a 2-D matrix on a quantizing codec
    Matrix(Quant),
}

fn is_routed_expert(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

/// The codec of an output tensor (canonical name) under `profile`.
fn codec_for(name: &str, shape: &[usize], profile: Profile) -> Codec {
    let llm =
        name.starts_with("model.") || name.starts_with("lm_head.") || name.starts_with("mtp.");
    if !llm || shape.len() != 2 || shape[0] < 2 || name.ends_with(".mlp.gate.weight") {
        return Codec::Exact;
    }
    // Mamba-2 depthwise taps are `[channels, kernel]`, not a matrix.
    if name.ends_with(".conv1d.weight") {
        return Codec::Exact;
    }
    if let Some(q) = tensor_quant_override(name) {
        return match q {
            Quant::F16 => Codec::Exact,
            q => Codec::Matrix(q),
        };
    }
    if is_routed_expert(name) {
        return Codec::Matrix(match profile {
            Profile::Q8 => Quant::Q8_2f,
            Profile::Q4 => Quant::Q4TiledP,
            Profile::Q2 if name.ends_with(".up_proj.weight") => Quant::Q2TiledP,
            Profile::Q2 => Quant::Q4TiledP,
        });
    }
    Codec::Matrix(Quant::Q8_2f)
}

fn exact_dtype(src: &str) -> anyhow::Result<TensorDtype> {
    Ok(match src {
        "BF16" => TensorDtype::Bf16,
        "F16" => TensorDtype::F16,
        "F32" => TensorDtype::F32,
        other => anyhow::bail!("nemotron_transcribe: unexpected source dtype {other}"),
    })
}

/// Canonical name of a source tensor; None = not packed.
fn canon(raw: &str) -> Option<String> {
    if raw.ends_with("._extra_state")
        || raw.starts_with("ctc_timestamp.")
        || raw.starts_with("perception.encoder.diarization_model.preprocessor.")
        || raw.contains(".sortformer_modules.hidden_to_spks.")
    {
        return None;
    }
    if raw.starts_with("perception.") {
        return Some(raw.to_string());
    }
    let (prefix, rest) = if let Some(r) = raw.strip_prefix("llm.model.layers.") {
        ("model.layers.", r)
    } else if let Some(r) = raw.strip_prefix("llm.mtp.layers.") {
        ("mtp.layers.", r)
    } else {
        return match raw {
            "llm.model.embed_tokens.weight" => Some("model.embed_tokens.weight".into()),
            "llm.model.norm.weight" => Some("model.norm.weight".into()),
            "llm.lm_head.weight" => Some("lm_head.weight".into()),
            _ => None,
        };
    };
    let (idx, tail) = rest.split_once('.')?;
    let tail = tail.replace("mixer._fp32_params.", "mixer.");
    // MoE blocks live under `mlp.` (router `gate`, experts, shared experts)
    let tail = if tail.starts_with("mixer.gate.")
        || tail.starts_with("mixer.experts.")
        || tail.starts_with("mixer.shared_experts.")
    {
        tail.replacen("mixer.", "mlp.", 1)
    } else {
        tail
    };
    Some(format!("{prefix}{idx}.{tail}"))
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
    anyhow::ensure!(
        detect(&config),
        "{}: not a Nemotron 3.5 Transcribe checkpoint",
        dir.display()
    );
    let llm = read_json(dir, "llm_backbone/config.json")
        .or_else(|| config.get("llm_config").cloned())
        .ok_or_else(|| anyhow::anyhow!("nemotron_transcribe: no llm_backbone/config.json"))?;
    let processor = read_json(dir, "processor_config.json")
        .ok_or_else(|| anyhow::anyhow!("nemotron_transcribe: no processor_config.json"))?;
    let tokenizer_config = read_json(dir, "tokenizer_config.json")
        .ok_or_else(|| anyhow::anyhow!("nemotron_transcribe: no tokenizer_config.json"))?;
    let generation = read_json(dir, "generation_config.json");
    let vocab = std::fs::read(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
    let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .or_else(|| {
            tokenizer_config
                .get("chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        });

    let block_types: Vec<String> = llm
        .get("layers_block_type")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("nemotron_transcribe: llm config has no layers_block_type"))?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    for t in &block_types {
        anyhow::ensure!(
            matches!(t.as_str(), "mamba" | "attention" | "moe"),
            "nemotron_transcribe: unsupported block type '{t}'"
        );
    }
    let g = |k: &str| -> anyhow::Result<usize> {
        llm.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("nemotron_transcribe: llm config has no {k}"))
    };
    let hidden = g("hidden_size")?;
    let n_exp = g("n_routed_experts")?;
    let moe_inter = g("moe_intermediate_size")?;
    anyhow::ensure!(
        llm.get("moe_latent_size").is_none_or(|v| v.is_null()),
        "nemotron_transcribe: latent MoE projections are not supported"
    );
    anyhow::ensure!(
        llm.get("mlp_hidden_act").and_then(|v| v.as_str()) == Some("relu2"),
        "nemotron_transcribe: only relu2 experts are supported"
    );

    let st = open_safetensors(&dir.join("model.safetensors"))?;
    let mut order: Vec<usize> = (0..st.tensors.len()).collect();
    order.sort_by(|&a, &b| st.tensors[a].name.cmp(&st.tensors[b].name));

    let n_out_estimate = st.tensors.len() + 2 * n_exp * (block_types.len() + 2);
    let gap = CmfStreamWriter::head_reserve_for(n_out_estimate, 64);
    let tmp = format!("{output}.part");
    let mut w =
        CmfStreamWriter::new(&tmp, gap).map_err(|e| anyhow::anyhow!("create {tmp}: {e}"))?;
    let mut census: std::collections::BTreeMap<&'static str, (usize, u64, u64)> =
        Default::default();
    let mut note = |group: &'static str, numel: usize, bytes: usize| {
        let e = census.entry(group).or_default();
        e.0 += 1;
        e.1 += numel as u64;
        e.2 += bytes as u64;
    };
    let group_of = |name: &str| -> &'static str {
        if name.starts_with("perception.") {
            "audio"
        } else if is_routed_expert(name) {
            "experts"
        } else if name.starts_with("mtp.") {
            "mtp"
        } else {
            "skeleton"
        }
    };
    let mut emitted = 0usize;
    for &ti in &order {
        let m = &st.tensors[ti];
        let Some(name) = canon(&m.name) else { continue };
        let raw = st.bytes(m);
        // Packed expert banks: (experts, in, out) → per-expert [out, in].
        if name.ends_with(".mlp.experts.gate_and_up_projs")
            || name.ends_with(".mlp.experts.down_projs")
        {
            anyhow::ensure!(
                m.shape.len() == 3 && m.shape[0] == n_exp,
                "{}: shape {:?}",
                m.name,
                m.shape
            );
            let (din, dout) = (m.shape[1], m.shape[2]);
            let is_up = name.ends_with("gate_and_up_projs");
            anyhow::ensure!(
                if is_up {
                    (din, dout) == (hidden, moe_inter)
                } else {
                    (din, dout) == (moe_inter, hidden)
                },
                "{}: dims {:?}",
                m.name,
                m.shape
            );
            let base = name
                .strip_suffix(".gate_and_up_projs")
                .or_else(|| name.strip_suffix(".down_projs"))
                .unwrap();
            let per = din * dout * 2;
            for e in 0..n_exp {
                let vals = to_f32(&m.dtype, &raw[e * per..(e + 1) * per])?;
                let mut t = vec![0f32; din * dout];
                for i in 0..din {
                    let src = &vals[i * dout..(i + 1) * dout];
                    for (o, &v) in src.iter().enumerate() {
                        t[o * din + i] = v;
                    }
                }
                let ename = format!(
                    "{base}.{e}.{}_proj.weight",
                    if is_up { "up" } else { "down" }
                );
                let shape = [dout, din];
                let (dt, bytes) = match codec_for(&ename, &shape, profile) {
                    Codec::Matrix(q) => quantize_2d(q, &t, dout, din),
                    // bf16 values widen to f32 losslessly
                    Codec::Exact => (
                        TensorDtype::F32,
                        t.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ),
                };
                note(group_of(&ename), t.len(), bytes.len());
                w.push(&ename, dt, &shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("write {ename}: {e}"))?;
                emitted += 1;
            }
            continue;
        }
        // Depthwise conv taps `[channels, 1, kernel]` → `[channels, kernel]`.
        let shape: Vec<usize> =
            if name.ends_with(".mixer.conv1d.weight") && m.shape.len() == 3 && m.shape[1] == 1 {
                vec![m.shape[0], m.shape[2]]
            } else {
                m.shape.clone()
            };
        let numel = shape.iter().product::<usize>().max(1);
        match codec_for(&name, &shape, profile) {
            Codec::Exact => {
                w.push(&name, exact_dtype(&m.dtype)?, &shape, raw)
                    .map_err(|e| anyhow::anyhow!("write {name}: {e}"))?;
                note(group_of(&name), numel, raw.len());
            }
            Codec::Matrix(q) => {
                let vals = to_f32(&m.dtype, raw)?;
                let (dt, bytes) = quantize_2d(q, &vals, shape[0], shape[1]);
                w.push(&name, dt, &shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("write {name}: {e}"))?;
                note(group_of(&name), numel, bytes.len());
            }
        }
        emitted += 1;
        if emitted % 200 == 0 {
            eprintln!("  {emitted} tensors, {:.0}s", t0.elapsed().as_secs_f64());
        }
    }
    for (g, (n, wts, b)) in &census {
        eprintln!(
            "  {g}: {n} tensors, {:.2} B weights → {:.2} GB",
            *wts as f64 / 1e9,
            *b as f64 / 1e9
        );
    }

    let n_attn = block_types.iter().filter(|t| *t == "attention").count();
    let layer_types: Vec<&str> = block_types
        .iter()
        .filter(|t| *t != "moe")
        .map(|t| {
            if t == "mamba" {
                "LinearAttention"
            } else {
                "FullAttention"
            }
        })
        .collect();
    let audio_config = config
        .get("audio_config")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": hidden,
        "intermediate_size": g("moe_shared_expert_intermediate_size")?,
        "num_layers": layer_types.len(),
        "num_attention_heads": g("num_attention_heads")?,
        "num_kv_heads": g("num_key_value_heads")?,
        "head_dim": g("head_dim")?,
        "vocab_size": g("vocab_size")?,
        "layer_types": layer_types,
        "rms_norm_eps": llm.get("layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-5),
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.0,
        "hidden_act": "relu2",
        "max_position_embeddings": 65536,
        "linear_conv_kernel_dim": g("conv_kernel")?,
        "linear_num_key_heads": g("n_groups")?,
        "linear_num_value_heads": g("mamba_num_heads")?,
        "linear_key_head_dim": g("ssm_state_size")?,
        "linear_value_head_dim": g("mamba_head_dim")?,
        "moe": {
            "num_experts": n_exp,
            "top_k": g("num_experts_per_tok")?,
            "moe_intermediate_size": moe_inter,
            "norm_topk_prob": llm.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true),
            "shared_expert_intermediate_size": g("moe_shared_expert_intermediate_size")?,
            "router_sigmoid": true,
            "routed_scaling_factor": llm.get("routed_scaling_factor").and_then(|v| v.as_f64()).unwrap_or(1.0),
        },
    }))?;
    let _ = n_attn;
    let provenance = serde_json::json!({
        "source": dir.display().to_string(),
        "base_model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        "quant": profile_name(profile),
        "license": "OpenMDW-1.1",
        "nemotron_transcribe": {
            "llm_config": llm,
            "audio_config": audio_config,
            "audio_token_id": config.get("audio_token_id").cloned().unwrap_or(serde_json::json!(131072)),
            "processor": processor,
            "generation_config": generation,
            "chat_template": chat_template,
            "audio_locator_tag": config.get("audio_locator_tag").cloned(),
            "bos_token": tokenizer_config.get("bos_token").cloned(),
            "eos_token": tokenizer_config.get("eos_token").cloned(),
        },
    });
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match profile {
            Profile::Q8 => QuantType::Q8_2f,
            Profile::Q4 | Profile::Q2 => QuantType::Q4Block,
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
    let count = w.tensor_count();
    w.finish(&header, None, Some(&vocab))
        .map_err(|e| anyhow::anyhow!("finish {tmp}: {e}"))?;
    std::fs::rename(&tmp, output)?;
    let size = std::fs::metadata(output)?.len();
    eprintln!(
        "{output}: {count} tensors, {:.2} GB ({}) in {:.0}s",
        size as f64 / 1e9,
        profile_name(profile),
        t0.elapsed().as_secs_f64()
    );
    let _ = quant_name;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names() {
        assert_eq!(
            canon("llm.model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            canon("llm.model.layers.0.mixer._fp32_params.A_log").as_deref(),
            Some("model.layers.0.mixer.A_log")
        );
        assert_eq!(
            canon("llm.model.layers.1.mixer.experts.down_projs").as_deref(),
            Some("model.layers.1.mlp.experts.down_projs")
        );
        assert_eq!(
            canon("llm.model.layers.1.mixer.gate.e_score_correction_bias").as_deref(),
            Some("model.layers.1.mlp.gate.e_score_correction_bias")
        );
        assert_eq!(
            canon("llm.mtp.layers.0.mixer.q_proj.weight").as_deref(),
            Some("mtp.layers.0.mixer.q_proj.weight")
        );
        assert_eq!(canon("llm.model.layers.5.mixer.q_proj._extra_state"), None);
        assert_eq!(canon("ctc_timestamp.decoder.upsample.weight"), None);
        assert_eq!(
            canon("perception.encoder.diarization_model.sortformer_modules.hidden_to_spks.weight"),
            None
        );
        assert_eq!(
            canon("perception.encoder.diarization_model.preprocessor.featurizer.fb"),
            None
        );
        assert_eq!(
            canon("perception.preprocessor.featurizer.fb").as_deref(),
            Some("perception.preprocessor.featurizer.fb")
        );
    }

    #[test]
    fn profile_codecs() {
        let e = "model.layers.1.mlp.experts.7.up_proj.weight";
        let d = "model.layers.1.mlp.experts.7.down_proj.weight";
        assert_eq!(
            codec_for(e, &[1856, 2688], Profile::Q8),
            Codec::Matrix(Quant::Q8_2f)
        );
        assert_eq!(
            codec_for(e, &[1856, 2688], Profile::Q4),
            Codec::Matrix(Quant::Q4TiledP)
        );
        assert_eq!(
            codec_for(e, &[1856, 2688], Profile::Q2),
            Codec::Matrix(Quant::Q2TiledP)
        );
        assert_eq!(
            codec_for(d, &[2688, 1856], Profile::Q2),
            Codec::Matrix(Quant::Q4TiledP)
        );
        for p in [Profile::Q4, Profile::Q2] {
            for n in [
                "model.layers.0.mixer.in_proj.weight",
                "model.layers.5.mixer.q_proj.weight",
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "lm_head.weight",
                "model.embed_tokens.weight",
            ] {
                assert_eq!(
                    codec_for(n, &[4096, 2688], p),
                    Codec::Matrix(Quant::Q8_2f),
                    "{n}"
                );
            }
        }
        assert_eq!(
            codec_for("model.layers.1.mlp.gate.weight", &[128, 2688], Profile::Q4),
            Codec::Exact
        );
        assert_eq!(
            codec_for(
                "model.layers.0.mixer.conv1d.weight",
                &[6144, 4],
                Profile::Q4
            ),
            Codec::Exact
        );
        assert_eq!(
            codec_for("model.layers.0.norm.weight", &[2688], Profile::Q4),
            Codec::Exact
        );
        assert_eq!(
            codec_for(
                "perception.encoder.asr_encoder.layers.0.attn.w_qkv.weight",
                &[3840, 1280],
                Profile::Q4
            ),
            Codec::Exact
        );
    }
}
