//! Gradium Phonon text-to-speech from a self-contained CMF package.
//!
//! Phonon is not a decoder-only language model: it is a causal flow model
//! followed by a Mimi audio decoder.  The published Q8 GGUF is therefore kept
//! intact in a CMF `U8` record and executed by Gradium's maintained `ptts`
//! runtime.  CMF remains the distribution boundary: its directory and hashes
//! protect every asset that is materialized for the backend.

use anyhow::{Context, Result, anyhow, bail, ensure};
use cortiq_core::CmfModel;
use ptts::synth::{Quant, Synth};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Architecture tag written by `cortiq phonon-pack`.
pub const ARCH_NAME: &str = "gradium-phonon-tts";

const CONFIG: (&str, &str) = ("phonon.config_json", "config.json");
const WEIGHTS: (&str, &str) = ("phonon.model_q8_gguf", "model.q8.gguf");
const TOKENIZER: (&str, &str) = ("phonon.tokenizer_json", "tokenizer.json");
const DEFAULT_VOICE: (&str, &str) = (
    "phonon.default_voice_safetensors",
    "default-voice.safetensors",
);

/// Controls exposed by the CMF TTS command.
///
/// `duration_delta` is the model's native rate conditioner: negative values
/// ask for faster speech and positive values ask for slower speech.  It is not
/// a post-processing resampler, so pitch remains model-controlled.
#[derive(Clone, Debug)]
pub struct SynthesisOptions<'a> {
    pub language: &'a str,
    /// Precomputed voice latents.  A CMF contains its source checkpoint's
    /// default voice; another file must be a Phonon-compatible safetensors
    /// embedding, not an audio recording.
    pub voice_file: Option<&'a Path>,
    pub voice_name: &'a str,
    pub temperature: f32,
    pub seed: u64,
    pub guidance: Option<f32>,
    pub duration_delta: f32,
    pub eos_threshold: Option<f32>,
    pub max_tokens_per_chunk: Option<usize>,
}

impl Default for SynthesisOptions<'static> {
    fn default() -> Self {
        Self {
            language: "en",
            voice_file: None,
            voice_name: "default",
            temperature: 0.3,
            seed: 4_242_424_242_424_242,
            guidance: None,
            duration_delta: 0.0,
            eos_threshold: None,
            max_tokens_per_chunk: None,
        }
    }
}

/// Synthesis result useful to frontends without parsing the written WAV.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisResult {
    pub sample_rate: u32,
    pub samples: usize,
    pub seconds: f64,
}

/// Materialize verified Phonon payloads from `model_path`, synthesize mono
/// 24 kHz WAV audio, then remove the temporary backend directory.
pub fn synthesize_to_wav(
    model_path: &Path,
    text: &str,
    output_path: &Path,
    options: &SynthesisOptions<'_>,
) -> Result<SynthesisResult> {
    ensure!(!text.trim().is_empty(), "text must not be empty");
    ensure!(
        options.temperature.is_finite() && options.temperature >= 0.0,
        "--temperature must be a finite value >= 0"
    );
    ensure!(
        options.duration_delta.is_finite(),
        "--duration-delta must be finite"
    );
    if let Some(guidance) = options.guidance {
        ensure!(
            guidance.is_finite() && guidance >= 0.0,
            "--guidance must be finite and >= 0"
        );
    }
    if let Some(eos_threshold) = options.eos_threshold {
        ensure!(eos_threshold.is_finite(), "--eos-threshold must be finite");
    }
    if let Some(max) = options.max_tokens_per_chunk {
        ensure!(max > 0, "--max-tokens-per-chunk must be greater than zero");
    }

    let language = ptts::preprocess::Lang::from_str(options.language).map_err(|e| {
        anyhow!(
            "unsupported --lang {:?}: {e}; Phonon multilingual supports en, fr, de, es, pt",
            options.language
        )
    })?;
    let model = CmfModel::open(model_path)
        .with_context(|| format!("opening Phonon CMF {}", model_path.display()))?;
    ensure!(
        model.header.arch.arch_name == ARCH_NAME,
        "{} is not a {ARCH_NAME} CMF (found {:?})",
        model_path.display(),
        model.header.arch.arch_name
    );
    let errors = model.verify();
    if !errors.is_empty() {
        bail!(
            "Phonon CMF integrity check failed:\n{}",
            errors
                .into_iter()
                .map(|e| format!("- {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let temp = MaterializedModel::new(&model)?;
    let voice_path = options
        .voice_file
        .map(Path::to_path_buf)
        .unwrap_or_else(|| temp.default_voice());
    if options.voice_file.is_none() && options.voice_name != "default" {
        bail!(
            "this CMF only embeds the default voice; use --voice-file with precomputed Phonon voice latents to select {:?}",
            options.voice_name
        );
    }
    let mut builder = Synth::builder(
        ptts::checkpoint::read_config(temp.config()).context("reading embedded Phonon config")?,
        temp.weights(),
        language,
    )
    .tokenizer_file(temp.tokenizer())
    // The package contains upstream's `model.q8.gguf`; this tells `ptts`
    // which CPU tensor representation to instantiate.
    .quant(Quant::Q80)
    .temperature(options.temperature)
    .seed(options.seed)
    .condition("duration_delta", options.duration_delta.to_string())
    .add_voice(options.voice_name, &voice_path)
    .voice(options.voice_name);
    if let Some(guidance) = options.guidance {
        builder = builder.cfg_coef(guidance);
    }
    if let Some(eos_threshold) = options.eos_threshold {
        builder = builder.eos_threshold(eos_threshold);
    }
    if let Some(max) = options.max_tokens_per_chunk {
        builder = builder.max_tokens_per_chunk(max);
    }
    let synthesizer = builder
        .build()
        .context("loading embedded Phonon Q8 model")?;
    let pcm = synthesizer.say(text).context("synthesizing speech")?;
    ensure!(!pcm.is_empty(), "Phonon produced no audio");
    let sample_rate = synthesizer.sample_rate();
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    ptts::wav::write_wav_file(output_path, &pcm, sample_rate)
        .with_context(|| format!("writing {}", output_path.display()))?;
    Ok(SynthesisResult {
        sample_rate,
        samples: pcm.len(),
        seconds: pcm.len() as f64 / sample_rate as f64,
    })
}

/// Every `cortiq tts` call gets an isolated 0700 directory.  It is removed
/// after the inference threads join, so source payloads never become a hidden
/// persistent cache outside the CMF the user explicitly selected.
struct MaterializedModel {
    root: PathBuf,
}

impl MaterializedModel {
    fn new(model: &CmfModel) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cortiq-phonon-{}-{}-{}-{nonce}",
            std::process::id(),
            serial,
            model.dir_hash()
        ));
        let mut builder = fs::DirBuilder::new();
        builder.recursive(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&root)
            .with_context(|| format!("creating temporary directory {}", root.display()))?;
        let materialized = Self { root };
        for (tensor, filename) in [CONFIG, WEIGHTS, TOKENIZER, DEFAULT_VOICE] {
            let entry = model
                .tensor(tensor)
                .ok_or_else(|| anyhow!("Phonon CMF is missing required payload {tensor}"))?;
            let length = usize::try_from(entry.nbytes).map_err(|_| {
                anyhow!("Phonon payload {tensor} length does not fit this platform")
            })?;
            ensure!(
                entry.dtype == cortiq_core::TensorDtype::U8 && entry.shape == [length],
                "Phonon payload {tensor} must be a one-dimensional U8 record"
            );
            fs::write(materialized.root.join(filename), model.entry_bytes(entry))
                .with_context(|| format!("materializing {tensor}"))?;
        }
        Ok(materialized)
    }

    fn config(&self) -> PathBuf {
        self.root.join(CONFIG.1)
    }

    fn weights(&self) -> PathBuf {
        self.root.join(WEIGHTS.1)
    }

    fn tokenizer(&self) -> PathBuf {
        self.root.join(TOKENIZER.1)
    }

    fn default_voice(&self) -> PathBuf {
        self.root.join(DEFAULT_VOICE.1)
    }
}

impl Drop for MaterializedModel {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            tracing::warn!(path = %self.root.display(), %error, "could not remove temporary Phonon payloads");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::format::{CMF_VERSION, CmfHeader, TensorSpec};
    use cortiq_core::types::{ModelArch, QuantType, TensorDtype};

    fn header() -> CmfHeader {
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": ARCH_NAME,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_layers": 9,
            "num_attention_heads": 8,
            "num_kv_heads": 8,
            "head_dim": 64,
            "vocab_size": 4001,
            "layer_types": vec!["FullAttention"; 9],
            "rms_norm_eps": 0.00001,
            "max_position_embeddings": 8192,
            "linear_conv_kernel_dim": 0,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
        }))
        .unwrap();
        CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Q8Row,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: vec![],
            router: None,
            segments: vec![],
        }
    }

    #[test]
    fn materializes_only_required_raw_assets() {
        let dir = std::env::temp_dir().join(format!("cortiq-phonon-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.cmf");
        let specs: Vec<_> = [CONFIG, WEIGHTS, TOKENIZER, DEFAULT_VOICE]
            .iter()
            .map(|(name, _)| TensorSpec {
                name: (*name).into(),
                dtype: TensorDtype::U8,
                shape: vec![3],
                data: b"abc".to_vec(),
            })
            .collect();
        CmfModel::write(&path, &header(), &specs, None, None).unwrap();
        let cmf = CmfModel::open(&path).unwrap();
        let temp = MaterializedModel::new(&cmf).unwrap();
        assert_eq!(fs::read(temp.config()).unwrap(), b"abc");
        assert_eq!(fs::read(temp.weights()).unwrap(), b"abc");
        assert_eq!(fs::read(temp.tokenizer()).unwrap(), b"abc");
        assert_eq!(fs::read(temp.default_voice()).unwrap(), b"abc");
        let root = temp.root.clone();
        drop(temp);
        assert!(!root.exists());
        let _ = fs::remove_dir_all(dir);
    }
}
