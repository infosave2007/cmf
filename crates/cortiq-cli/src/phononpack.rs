//! Package Gradium Phonon TTS checkpoints as opaque, self-contained CMF
//! artifacts.
//!
//! Phonon's upstream runtime owns the execution semantics of its custom
//! transformer, flow head, and Mimi decoder.  This packer deliberately does
//! not pretend that those weights are generic CMF LLM tensors: it carries the
//! official Q8 GGUF unchanged in a U8 tensor, alongside the exact JSON and
//! voice prompt files that the bridge needs.  The CMF envelope supplies
//! integrity, a portable vocabulary section, and explicit provenance.

use anyhow::{Context, anyhow, ensure};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use sha2::{Digest, Sha256};
use std::path::Path;

const UPSTREAM_MODEL: &str = "gradium/phonon-40m-multilingual";
const WEIGHTS_FILE: &str = "model.q8.gguf";
const WEIGHTS_TENSOR: &str = "phonon.model_q8_gguf";

/// Configuration values used both to verify that the input is a Phonon
/// checkpoint and to describe the text-transformer portion in the CMF
/// header.  The rest of the model remains opaque by design.
struct PhononConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    norm_eps: f64,
    max_position_embeddings: usize,
    rope_theta: f64,
    voice_dim: usize,
    latent_dim: usize,
    sample_rate: usize,
    frame_rate: f64,
    revision_hint: Option<String>,
}

fn raw_spec(name: impl Into<String>, data: Vec<u8>) -> TensorSpec {
    let len = data.len();
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::U8,
        shape: vec![len],
        data,
    }
}

fn required_bytes(root: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let path = root.join(name);
    let metadata = std::fs::metadata(&path)
        .with_context(|| format!("required Phonon file {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "required Phonon file {} is not a regular file",
        path.display()
    );
    let bytes = std::fs::read(&path).with_context(|| path.display().to_string())?;
    ensure!(
        !bytes.is_empty(),
        "required Phonon file {} is empty",
        path.display()
    );
    Ok(bytes)
}

fn required_object<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<&'a serde_json::Map<String, serde_json::Value>> {
    object
        .get(key)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow!("{}: {key} must be a JSON object", path.display()))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<&'a str> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("{}: {key} must be a non-empty string", path.display()))
}

fn positive_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<usize> {
    let value = object
        .get(key)
        .ok_or_else(|| anyhow!("{}: missing {key}", path.display()))?;
    positive_usize_value(value, key, path)
}

fn positive_usize_value(
    value: &serde_json::Value,
    description: &str,
    path: &Path,
) -> anyhow::Result<usize> {
    let value = value.as_u64().ok_or_else(|| {
        anyhow!(
            "{}: {description} must be a positive integer",
            path.display()
        )
    })?;
    let value = usize::try_from(value)
        .map_err(|_| anyhow!("{}: {description} does not fit usize", path.display()))?;
    ensure!(
        value > 0,
        "{}: {description} must be a positive integer",
        path.display()
    );
    Ok(value)
}

fn optional_positive_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<Option<usize>> {
    match object.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => positive_usize_value(value, key, path).map(Some),
    }
}

fn positive_f64(value: &serde_json::Value, description: &str, path: &Path) -> anyhow::Result<f64> {
    let value = value
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| {
            anyhow!(
                "{}: {description} must be a positive number",
                path.display()
            )
        })?;
    Ok(value)
}

fn positive_f64_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<f64> {
    let value = object
        .get(key)
        .ok_or_else(|| anyhow!("{}: missing {key}", path.display()))?;
    positive_f64(value, key, path)
}

fn optional_positive_f64(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> anyhow::Result<Option<f64>> {
    match object.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => positive_f64(value, key, path).map(Some),
    }
}

fn set_or_check(
    slot: &mut Option<usize>,
    value: usize,
    description: &str,
    path: &Path,
) -> anyhow::Result<()> {
    if let Some(existing) = *slot {
        ensure!(
            existing == value,
            "{}: {description} disagrees with the rest of the Phonon config",
            path.display()
        );
    } else {
        *slot = Some(value);
    }
    Ok(())
}

fn optional_revision_hint(root: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    ["_commit_hash", "commit_hash", "revision"]
        .into_iter()
        .find_map(|key| {
            root.get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

/// The Q8 file executed by the `ptts` bridge uses this native config layout.
/// It predates the Transformers export and has no `model_type` field, so it
/// must be recognized from its flow LM and Mimi codec records instead.
fn parse_native_ptts_config(
    root: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> anyhow::Result<PhononConfig> {
    let flow = required_object(root, "flow_lm", path)?;
    let hidden_size = positive_usize(flow, "d_model", path)?;
    let num_attention_heads = positive_usize(flow, "num_heads", path)?;
    ensure!(
        hidden_size % num_attention_heads == 0,
        "{}: flow_lm.d_model must be divisible by flow_lm.num_heads",
        path.display()
    );
    let latent_dim = positive_usize(flow, "ldim", path)?;
    let mimi = required_object(root, "mimi", path)?;
    ensure!(
        positive_usize(mimi, "channels", path)? == 1,
        "{}: Phonon CMF wrapper supports mono Mimi output only",
        path.display()
    );
    ensure!(
        latent_dim == positive_usize(mimi, "quantizer_dimension", path)?,
        "{}: flow_lm.ldim must equal mimi.quantizer_dimension",
        path.display()
    );
    let ratios = mimi
        .get("ratios")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("{}: mimi.ratios must be an array", path.display()))?;
    ensure!(
        !ratios.is_empty()
            && ratios
                .iter()
                .all(|ratio| ratio.as_u64().is_some_and(|value| value > 0)),
        "{}: mimi.ratios must contain positive integers",
        path.display()
    );
    let rope_theta = flow.get("max_period").map_or(Ok(10_000.0), |value| {
        positive_f64(value, "flow_lm.max_period", path)
    })?;

    Ok(PhononConfig {
        hidden_size,
        intermediate_size: positive_usize(flow, "dim_feedforward", path)?,
        num_layers: positive_usize(flow, "num_layers", path)?,
        num_attention_heads,
        num_kv_heads: num_attention_heads,
        head_dim: hidden_size / num_attention_heads,
        vocab_size: positive_usize(flow, "n_bins", path)?,
        // These fields are not represented in ptts' native config.  They
        // describe the fixed Phonon transformer defaults in the CMF header;
        // the bridge still executes the opaque upstream GGUF.
        norm_eps: 1e-5,
        max_position_embeddings: 8192,
        rope_theta,
        // The default voice is a speaker-Mimi embedding, whose width is the
        // Mimi dimension rather than necessarily the flow LM width.
        voice_dim: positive_usize(mimi, "dimension", path)?,
        latent_dim,
        sample_rate: positive_usize(mimi, "sample_rate", path)?,
        frame_rate: positive_f64_field(mimi, "frame_rate", path)?,
        revision_hint: optional_revision_hint(root),
    })
}

fn parse_config(raw: &[u8], path: &Path) -> anyhow::Result<PhononConfig> {
    let value: serde_json::Value =
        serde_json::from_slice(raw).with_context(|| format!("{}: invalid JSON", path.display()))?;
    let root = value
        .as_object()
        .ok_or_else(|| anyhow!("{}: config root must be a JSON object", path.display()))?;

    if !root.contains_key("model_type") {
        return parse_native_ptts_config(root, path);
    }
    ensure!(
        required_string(root, "model_type", path)? == "gradium_phonon_tts",
        "{}: model_type must be gradium_phonon_tts",
        path.display()
    );
    if let Some(architectures) = root.get("architectures") {
        let architectures = architectures
            .as_array()
            .ok_or_else(|| anyhow!("{}: architectures must be an array", path.display()))?;
        ensure!(
            architectures.iter().any(|value| {
                value.as_str() == Some("GradiumPhononTTSForConditionalGeneration")
            }),
            "{}: architectures must include GradiumPhononTTSForConditionalGeneration",
            path.display()
        );
    }

    let hidden_size = positive_usize(root, "hidden_size", path)?;
    let intermediate_size = positive_usize(root, "intermediate_size", path)?;
    let num_layers = positive_usize(root, "num_hidden_layers", path)?;
    let num_attention_heads = positive_usize(root, "num_attention_heads", path)?;
    ensure!(
        hidden_size % num_attention_heads == 0,
        "{}: hidden_size must be divisible by num_attention_heads",
        path.display()
    );
    let num_kv_heads =
        optional_positive_usize(root, "num_key_value_heads", path)?.unwrap_or(num_attention_heads);
    let head_dim = optional_positive_usize(root, "head_dim", path)?
        .unwrap_or(hidden_size / num_attention_heads);
    let vocab_size = positive_usize(root, "vocab_size", path)?;
    let norm_eps = positive_f64_field(root, "layer_norm_eps", path)?;
    let max_position_embeddings = positive_usize(root, "max_position_embeddings", path)?;
    ensure!(
        num_kv_heads <= num_attention_heads,
        "{}: num_key_value_heads exceeds num_attention_heads",
        path.display()
    );
    ensure!(
        hidden_size / num_attention_heads == head_dim,
        "{}: hidden_size / num_attention_heads must equal head_dim",
        path.display()
    );

    let rope_theta = root
        .get("rope_parameters")
        .and_then(serde_json::Value::as_object)
        .and_then(|rope| rope.get("rope_theta"))
        .or_else(|| root.get("rope_theta"))
        .map_or(Ok(10_000.0), |value| {
            positive_f64(value, "rope_theta", path)
        })?;

    // The original ptts config describes the head/codec as `flow_lm` and
    // `mimi`; the current Transformers config calls the head
    // `flow_head_config` and puts the prompt/latent dimensions at the root.
    // Both point to the same released Q8 GGUF runtime, so accept either
    // official schema while checking every overlapping value for agreement.
    let mut voice_dim = optional_positive_usize(root, "audio_prompt_dim", path)?;
    let mut latent_dim = optional_positive_usize(root, "latent_dim", path)?;
    let mut sample_rate = optional_positive_usize(root, "sample_rate", path)?;
    let mut frame_rate = optional_positive_f64(root, "frame_rate", path)?;
    let mut found_flow_config = false;
    if let Some(flow) = root.get("flow_head_config") {
        let flow = flow
            .as_object()
            .ok_or_else(|| anyhow!("{}: flow_head_config must be a JSON object", path.display()))?;
        found_flow_config = true;
        let condition_dim = positive_usize(flow, "condition_dim", path)?;
        ensure!(
            condition_dim == hidden_size,
            "{}: flow_head_config.condition_dim must equal hidden_size",
            path.display()
        );
        set_or_check(
            &mut latent_dim,
            positive_usize(flow, "latent_dim", path)?,
            "flow_head_config.latent_dim",
            path,
        )?;
    }
    if let Some(flow) = root.get("flow_lm") {
        let flow = flow
            .as_object()
            .ok_or_else(|| anyhow!("{}: flow_lm must be a JSON object", path.display()))?;
        found_flow_config = true;
        set_or_check(
            &mut voice_dim,
            positive_usize(flow, "d_model", path)?,
            "flow_lm.d_model",
            path,
        )?;
        set_or_check(
            &mut latent_dim,
            positive_usize(flow, "ldim", path)?,
            "flow_lm.ldim",
            path,
        )?;
    }
    ensure!(
        found_flow_config,
        "{}: missing flow_head_config or flow_lm",
        path.display()
    );
    if let Some(mimi) = root.get("mimi") {
        let mimi = mimi
            .as_object()
            .ok_or_else(|| anyhow!("{}: mimi must be a JSON object", path.display()))?;
        if let Some(channels) = optional_positive_usize(mimi, "channels", path)? {
            ensure!(
                channels == 1,
                "{}: Phonon CMF wrapper supports mono Mimi output only",
                path.display()
            );
        }
        if let Some(value) = optional_positive_usize(mimi, "sample_rate", path)? {
            set_or_check(&mut sample_rate, value, "mimi.sample_rate", path)?;
        }
        if let Some(value) = optional_positive_f64(mimi, "frame_rate", path)? {
            if let Some(existing) = frame_rate {
                ensure!(
                    (existing - value).abs() < f64::EPSILON,
                    "{}: mimi.frame_rate disagrees with the rest of the Phonon config",
                    path.display()
                );
            } else {
                frame_rate = Some(value);
            }
        }
        if let Some(value) = optional_positive_usize(mimi, "quantizer_dimension", path)? {
            set_or_check(&mut latent_dim, value, "mimi.quantizer_dimension", path)?;
        }
        if let Some(ratios) = mimi.get("ratios") {
            let ratios = ratios
                .as_array()
                .ok_or_else(|| anyhow!("{}: mimi.ratios must be an array", path.display()))?;
            ensure!(
                !ratios.is_empty()
                    && ratios
                        .iter()
                        .all(|ratio| ratio.as_u64().is_some_and(|n| n > 0)),
                "{}: mimi.ratios must contain positive integers",
                path.display()
            );
        }
    }
    let voice_dim = voice_dim.unwrap_or(hidden_size);
    let latent_dim = latent_dim.ok_or_else(|| {
        anyhow!(
            "{}: missing latent_dim in the Phonon config or flow head",
            path.display()
        )
    })?;
    // Phonon releases are fixed at 24 kHz and 12.5 latent frames per second;
    // legacy config carries those in `mimi`, while current HF configs carry
    // only `frame_rate` and keep the codec setting in the processor.
    let sample_rate = sample_rate.unwrap_or(24_000);
    let frame_rate = frame_rate.unwrap_or(12.5);

    Ok(PhononConfig {
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        norm_eps,
        max_position_embeddings,
        rope_theta,
        voice_dim,
        latent_dim,
        sample_rate,
        frame_rate,
        revision_hint: optional_revision_hint(root),
    })
}

fn validate_tokenizer(raw: &[u8], path: &Path) -> anyhow::Result<()> {
    let value: serde_json::Value = serde_json::from_slice(raw)
        .with_context(|| format!("{}: invalid tokenizer JSON", path.display()))?;
    let root = value
        .as_object()
        .ok_or_else(|| anyhow!("{}: tokenizer root must be a JSON object", path.display()))?;
    let model = required_object(root, "model", path)?;
    ensure!(
        required_string(model, "type", path)? == "Unigram",
        "{}: tokenizer model.type must be Unigram",
        path.display()
    );
    let vocab = model
        .get("vocab")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("{}: tokenizer model.vocab must be an array", path.display()))?;
    ensure!(
        !vocab.is_empty(),
        "{}: tokenizer model.vocab must not be empty",
        path.display()
    );
    for (index, entry) in vocab.iter().enumerate() {
        let Some(entry) = entry.as_array() else {
            return Err(anyhow!(
                "{}: tokenizer model.vocab[{index}] must be an array",
                path.display()
            ));
        };
        ensure!(
            entry.len() >= 2 && entry[0].as_str().is_some() && entry[1].as_f64().is_some(),
            "{}: tokenizer model.vocab[{index}] must be [token, score]",
            path.display()
        );
    }
    Ok(())
}

/// Validate the only speaker prompt distributed with the official model.
/// The upstream loader reads `speaker_wavs[0].T`, so a different geometry
/// would produce an artifact that cannot be executed faithfully.
fn validate_default_voice(raw: &[u8], path: &Path, expected_dim: usize) -> anyhow::Result<()> {
    ensure!(
        raw.len() >= 8,
        "{}: truncated safetensors header",
        path.display()
    );
    let header_len_u64 = u64::from_le_bytes(raw[..8].try_into().unwrap());
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| anyhow!("{}: safetensors header is too large", path.display()))?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| anyhow!("{}: safetensors header overflows", path.display()))?;
    let header = raw
        .get(8..header_end)
        .ok_or_else(|| anyhow!("{}: safetensors header past EOF", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(header)
        .with_context(|| format!("{}: invalid safetensors header", path.display()))?;
    let root = header
        .as_object()
        .ok_or_else(|| anyhow!("{}: safetensors header must be an object", path.display()))?;
    let speaker = required_object(root, "speaker_wavs", path)?;
    let dtype = required_string(speaker, "dtype", path)?;
    let bytes_per_element = match dtype {
        "F32" => 4usize,
        "F16" | "BF16" => 2usize,
        other => {
            return Err(anyhow!(
                "{}: speaker_wavs has unsupported dtype {other}",
                path.display()
            ));
        }
    };
    let shape = speaker
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("{}: speaker_wavs.shape must be an array", path.display()))?;
    ensure!(
        shape.len() == 3,
        "{}: speaker_wavs must have shape [1, {expected_dim}, frames]",
        path.display()
    );
    let dim = |index: usize| -> anyhow::Result<usize> {
        let value = shape[index].as_u64().ok_or_else(|| {
            anyhow!(
                "{}: speaker_wavs.shape[{index}] must be a positive integer",
                path.display()
            )
        })?;
        let value = usize::try_from(value).map_err(|_| {
            anyhow!(
                "{}: speaker_wavs.shape[{index}] does not fit usize",
                path.display()
            )
        })?;
        ensure!(
            value > 0,
            "{}: speaker_wavs.shape[{index}] must be a positive integer",
            path.display()
        );
        Ok(value)
    };
    let batch = dim(0)?;
    let channels = dim(1)?;
    let frames = dim(2)?;
    ensure!(
        batch == 1 && channels == expected_dim,
        "{}: speaker_wavs must have shape [1, {expected_dim}, frames]",
        path.display()
    );
    let offsets = speaker
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            anyhow!(
                "{}: speaker_wavs.data_offsets must be an array",
                path.display()
            )
        })?;
    ensure!(
        offsets.len() == 2,
        "{}: speaker_wavs.data_offsets must have two elements",
        path.display()
    );
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| anyhow!("{}: invalid speaker_wavs.data_offsets[0]", path.display()))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| anyhow!("{}: invalid speaker_wavs.data_offsets[1]", path.display()))?;
    ensure!(
        start <= end,
        "{}: speaker_wavs data offsets are inverted",
        path.display()
    );
    let payload_len = raw.len() - header_end;
    let payload_len = u64::try_from(payload_len).unwrap();
    ensure!(
        end <= payload_len,
        "{}: speaker_wavs data offsets extend past EOF",
        path.display()
    );
    let elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(frames))
        .ok_or_else(|| anyhow!("{}: speaker_wavs shape overflows", path.display()))?;
    let expected_len = elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| anyhow!("{}: speaker_wavs byte length overflows", path.display()))?;
    ensure!(
        end - start == u64::try_from(expected_len).unwrap(),
        "{}: speaker_wavs data length does not match its shape and dtype",
        path.display()
    );
    Ok(())
}

fn source_revision(root: &Path, hint: Option<&str>) -> String {
    if let Some(hint) = hint.filter(|hint| !hint.is_empty()) {
        return hint.into();
    }
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut current: Option<&Path> = Some(canonical.as_path());
    while let Some(path) = current {
        if path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "snapshots")
        {
            if let Some(revision) = path.file_name().and_then(|name| name.to_str()) {
                return revision.into();
            }
        }
        current = path.parent();
    }
    "unresolved-local-source".into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn build_arch(config: &PhononConfig) -> anyhow::Result<ModelArch> {
    serde_json::from_value(serde_json::json!({
        "arch_name": "gradium-phonon-tts",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "layer_types": vec!["FullAttention"; config.num_layers],
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "hidden_act": "gelu_erf",
        "max_position_embeddings": config.max_position_embeddings,
        "linear_conv_kernel_dim": null,
        "linear_num_key_heads": null,
        "linear_num_value_heads": null,
    }))
    .context("construct Gradium Phonon CMF architecture")
}

/// Wrap an official local `gradium/phonon-40m-multilingual` checkout in CMF.
///
/// Required source files are `config.json`, `model.q8.gguf`, `tokenizer.json`,
/// and `default-voice.safetensors`.  The Q8 GGUF and non-JSON voice prompt
/// remain byte-for-byte U8 blobs; `tokenizer.json` is stored both as the raw
/// backend asset and as the CMF VOCAB section.
pub fn cmd_phonon_pack(source_dir: &str, output: &str) -> anyhow::Result<()> {
    let source = Path::new(source_dir);
    ensure!(
        source.is_dir(),
        "Phonon source directory {} does not exist or is not a directory",
        source.display()
    );

    let config_path = source.join("config.json");
    let config_bytes = required_bytes(source, "config.json")?;
    let config = parse_config(&config_bytes, &config_path)?;

    let weights = required_bytes(source, WEIGHTS_FILE)?;
    let tokenizer_path = source.join("tokenizer.json");
    let tokenizer = required_bytes(source, "tokenizer.json")?;
    validate_tokenizer(&tokenizer, &tokenizer_path)?;
    let voice_path = source.join("default-voice.safetensors");
    let voice = required_bytes(source, "default-voice.safetensors")?;
    validate_default_voice(&voice, &voice_path, config.voice_dim)?;

    let source_files = vec![
        "config.json".to_string(),
        WEIGHTS_FILE.to_string(),
        "tokenizer.json".to_string(),
        "default-voice.safetensors".to_string(),
    ];
    let specs = vec![
        raw_spec("phonon.config_json", config_bytes.clone()),
        raw_spec(WEIGHTS_TENSOR, weights.clone()),
        raw_spec("phonon.tokenizer_json", tokenizer.clone()),
        raw_spec("phonon.default_voice_safetensors", voice.clone()),
    ];
    let model_sha256 = sha256_hex(&weights);
    let config_sha256 = sha256_hex(&config_bytes);
    let tokenizer_sha256 = sha256_hex(&tokenizer);
    let voice_sha256 = sha256_hex(&voice);
    let revision = source_revision(source, config.revision_hint.as_deref());
    let arch = build_arch(&config)?;
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        // Informational only: the real source encoding is declared below and
        // carried unchanged in `phonon.model_q8_gguf`.
        quant_type: QuantType::Q8Row,
        provenance: Some(serde_json::json!({
            "source": format!("https://huggingface.co/{UPSTREAM_MODEL}"),
            "source_model": UPSTREAM_MODEL,
            "source_revision": revision,
            "source_model_sha256": model_sha256,
            "source_hashes": {
                "algorithm": "sha256",
                "config.json": config_sha256,
                "model.q8.gguf": model_sha256,
                "tokenizer.json": tokenizer_sha256,
                "default-voice.safetensors": voice_sha256,
            },
            "license": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "source_format": "Gradium Phonon Hugging Face checkpoint + upstream Q8 GGUF",
            "source_files": source_files,
            "packing": {
                "tool": "cortiq phonon-pack",
                "weights_tensor": WEIGHTS_TENSOR,
                "weights_encoding": "opaque GGUF Q8",
                "artifact_encoding": "raw U8 tensors",
                "runtime": "gradium-phonon-tts bridge",
            },
            "voice": {
                "tensor": "speaker_wavs",
                "stored_as": "phonon.default_voice_safetensors",
                "embedding_dim": config.voice_dim,
            },
            "audio": {
                "sample_rate": config.sample_rate,
                "frame_rate": config.frame_rate,
                "latent_dim": config.latent_dim,
            },
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
    CmfModel::write(output, &header, &specs, None, Some(&tokenizer))
        .map_err(|error| anyhow!("write {output}: {error}"))?;

    let size = std::fs::metadata(output)
        .with_context(|| format!("read written CMF {output}"))?
        .len();
    println!(
        "{output}: {} raw Phonon tensors, {:.2} MB",
        specs.len(),
        size as f64 / (1 << 20) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let nonce = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cortiq-phonon-pack-{label}-{}-{nanos}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_default_voice(path: &Path) {
        let header = serde_json::json!({
            "speaker_wavs": {
                "dtype": "F32",
                "shape": [1, 4, 2],
                "data_offsets": [0, 32],
            }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0; 32]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_checkpoint(dir: &Path, model_type: &str) -> Vec<u8> {
        let config = serde_json::json!({
            "model_type": model_type,
            "architectures": ["GradiumPhononTTSForConditionalGeneration"],
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 5,
            "layer_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "rope_parameters": {"rope_theta": 10000.0},
            "flow_lm": {"d_model": 4, "ldim": 2},
            "mimi": {
                "channels": 1,
                "sample_rate": 24000,
                "frame_rate": 12.5,
                "quantizer_dimension": 2,
                "ratios": [6, 5, 4],
            },
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let weights = b"tiny opaque Q8 GGUF payload".to_vec();
        std::fs::write(dir.join(WEIGHTS_FILE), &weights).unwrap();
        let tokenizer = serde_json::json!({
            "model": {
                "type": "Unigram",
                "vocab": [["<unk>", 0.0], ["a", -1.0]],
            },
        });
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        write_default_voice(&dir.join("default-voice.safetensors"));
        weights
    }

    #[test]
    fn packs_opaque_q8_weights_vocab_and_voice() {
        let source = TempDir::new("source");
        let output = TempDir::new("output");
        let weights = write_checkpoint(source.path(), "gradium_phonon_tts");
        let out = output.path().join("phonon.cmf");

        cmd_phonon_pack(source.path().to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let model = CmfModel::open(&out).unwrap();
        assert_eq!(model.header.arch.arch_name, "gradium-phonon-tts");
        assert_eq!(model.header.arch.num_layers, 2);
        assert_eq!(
            model.tensor_bytes(WEIGHTS_TENSOR).unwrap(),
            weights.as_slice()
        );
        assert_eq!(model.tensor(WEIGHTS_TENSOR).unwrap().dtype, TensorDtype::U8);
        let voice = std::fs::read(source.path().join("default-voice.safetensors")).unwrap();
        assert_eq!(
            model
                .tensor_bytes("phonon.default_voice_safetensors")
                .unwrap(),
            voice.as_slice()
        );
        let tokenizer = std::fs::read(source.path().join("tokenizer.json")).unwrap();
        assert_eq!(
            model.tensor_bytes("phonon.tokenizer_json").unwrap(),
            tokenizer.as_slice()
        );
        assert_eq!(model.vocab.as_deref(), Some(tokenizer.as_slice()));
        let provenance = model.header.provenance.as_ref().unwrap();
        assert_eq!(provenance["license"], "CC-BY-NC-4.0");
        assert_eq!(provenance["source_model"], UPSTREAM_MODEL);
        assert_eq!(
            provenance["source_model_sha256"],
            serde_json::Value::String(sha256_hex(&weights))
        );
        assert_eq!(provenance["packing"]["weights_tensor"], WEIGHTS_TENSOR);
    }

    #[test]
    fn rejects_a_non_phonon_config_before_writing() {
        let source = TempDir::new("bad-config-source");
        let output = TempDir::new("bad-config-output");
        write_checkpoint(source.path(), "llama");
        let out = output.path().join("phonon.cmf");

        let error =
            cmd_phonon_pack(source.path().to_str().unwrap(), out.to_str().unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("gradium_phonon_tts"));
        assert!(!out.exists());
    }

    #[test]
    fn requires_the_q8_gguf_and_default_voice() {
        let source = TempDir::new("missing-source");
        let output = TempDir::new("missing-output");
        write_checkpoint(source.path(), "gradium_phonon_tts");
        std::fs::remove_file(source.path().join(WEIGHTS_FILE)).unwrap();
        let out = output.path().join("phonon.cmf");

        let error =
            cmd_phonon_pack(source.path().to_str().unwrap(), out.to_str().unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains(WEIGHTS_FILE));
        assert!(!out.exists());
    }

    #[test]
    fn accepts_the_current_transformers_config_layout() {
        let config = serde_json::json!({
            "model_type": "gradium_phonon_tts",
            "architectures": ["GradiumPhononTTSForConditionalGeneration"],
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": null,
            "head_dim": null,
            "vocab_size": 5,
            "layer_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "audio_prompt_dim": 4,
            "latent_dim": 2,
            "frame_rate": 12.5,
            "flow_head_config": {"condition_dim": 4, "latent_dim": 2},
        });
        let raw = serde_json::to_vec(&config).unwrap();
        let parsed = parse_config(&raw, Path::new("modern-config.json")).unwrap();
        assert_eq!(parsed.voice_dim, 4);
        assert_eq!(parsed.latent_dim, 2);
        assert_eq!(parsed.sample_rate, 24_000);
        assert_eq!(parsed.frame_rate, 12.5);
    }

    #[test]
    fn accepts_the_native_ptts_q8_config_layout() {
        let config = serde_json::json!({
            "flow_lm": {
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 2,
                "dim_feedforward": 8,
                "max_period": 10000.0,
                "n_bins": 5,
                "ldim": 2,
            },
            "mimi": {
                "channels": 1,
                "sample_rate": 24000,
                "frame_rate": 12.5,
                "dimension": 4,
                "quantizer_dimension": 2,
                "ratios": [6, 5, 4],
            },
        });
        let raw = serde_json::to_vec(&config).unwrap();
        let parsed = parse_config(&raw, Path::new("ptts-config.json")).unwrap();
        assert_eq!(parsed.hidden_size, 4);
        assert_eq!(parsed.voice_dim, 4);
        assert_eq!(parsed.vocab_size, 5);
    }
}
