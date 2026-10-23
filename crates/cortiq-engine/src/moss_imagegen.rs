//! Staged native MOSS-Image generation.
//!
//! A MOSS CMF is intentionally one portable file, but it is *not* resident
//! as one GPU model: Qwen3.5 conditioning is loaded/run/dropped first, then
//! the MOSS DiT owns the device through the Euler loop, then Flux2 VAE is
//! loaded for decoding.  This preserves the official component-offload
//! contract and makes the q4tp profile viable on 24 GB Vulkan and Metal.

use crate::Pipeline;
use crate::moss_image::MossDit;
use crate::sampler::{SamplerConfig, SplitMix64};
use crate::vae::Flux2VaeDecoder;
use cortiq_core::CmfModel;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// `header.arch.arch_name` written by `imagine-pack` for MOSS-Image.
pub const ARCH_NAME: &str = "moss_image";
pub const MAX_TOKENS: usize = 4096;

/// The portable defaults serialized by the MOSS packer.  The source release
/// uses 1024², 50 flow-matching Euler steps, CFG 3, shift 5, and skips main
/// layer 22 only for the unconditional pass.
#[derive(Clone, Debug)]
pub struct MossDefaults {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub cfg: f32,
    pub shift: f32,
    pub negative_prompt: String,
    pub max_sequence_length: usize,
    pub uncond_skip_layers: Vec<usize>,
}

impl Default for MossDefaults {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 50,
            cfg: 3.0,
            shift: 5.0,
            negative_prompt: String::new(),
            max_sequence_length: MAX_TOKENS,
            uncond_skip_layers: vec![22],
        }
    }
}

impl MossDefaults {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        let usize_or =
            |key: &str, fallback| v[key].as_u64().map(|x| x as usize).unwrap_or(fallback);
        let layers = v["uncond_skip_layers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|i| i as usize)
                    .collect()
            })
            .unwrap_or(d.uncond_skip_layers.clone());
        Self {
            height: usize_or("height", d.height),
            width: usize_or("width", d.width),
            steps: usize_or("steps", d.steps),
            cfg: v["cfg"].as_f64().map(|x| x as f32).unwrap_or(d.cfg),
            shift: v["shift"].as_f64().map(|x| x as f32).unwrap_or(d.shift),
            negative_prompt: v["negative_prompt"]
                .as_str()
                .unwrap_or(&d.negative_prompt)
                .to_string(),
            max_sequence_length: usize_or("max_sequence_length", d.max_sequence_length),
            uncond_skip_layers: layers,
        }
    }

    pub fn of(model: &CmfModel) -> Self {
        model
            .tensor_bytes("moss.defaults_json")
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .map(|v| Self::from_json(&v))
            .unwrap_or_default()
    }
}

/// Scheduler facts that alter the numerical Euler grid. They live in the
/// container so a future MOSS variant cannot silently inherit this release's
/// 1000-step / training-shift-1 assumptions.
#[derive(Clone, Copy, Debug)]
struct MossScheduler {
    num_train_timesteps: usize,
    training_shift: f32,
}

impl MossScheduler {
    fn of(model: &CmfModel) -> Result<Self, String> {
        let json: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("moss.scheduler_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("moss.scheduler_json: {e}"))?;
        let num_train_timesteps = json["num_train_timesteps"]
            .as_u64()
            .ok_or("MOSS scheduler missing num_train_timesteps")?
            as usize;
        let training_shift = json["training_shift"]
            .as_f64()
            .ok_or("MOSS scheduler missing training_shift")? as f32;
        if num_train_timesteps < 2 || !training_shift.is_finite() || training_shift <= 0.0 {
            return Err("MOSS scheduler has invalid training grid/shift".into());
        }
        Ok(Self {
            num_train_timesteps,
            training_shift,
        })
    }
}

/// User-overridable MOSS image recipe.
#[derive(Clone, Debug)]
pub struct MossParams {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub cfg: f32,
    pub shift: f32,
    pub max_tokens: usize,
    pub seed: u64,
    pub num_images: usize,
    pub negative_prompt: Option<String>,
    /// `None` means the source release default `[22]`; `Some(vec![])`
    /// intentionally disables the negative branch skip.
    pub uncond_skip_layers: Option<Vec<usize>>,
}

impl MossParams {
    pub fn from_defaults(d: &MossDefaults) -> Self {
        Self {
            height: d.height,
            width: d.width,
            steps: d.steps,
            cfg: d.cfg,
            shift: d.shift,
            max_tokens: d.max_sequence_length,
            seed: 42,
            num_images: 1,
            negative_prompt: None,
            uncond_skip_layers: None,
        }
    }
}

impl Default for MossParams {
    fn default() -> Self {
        Self::from_defaults(&MossDefaults::default())
    }
}

/// Generated RGB image, row-major interleaved u8.
pub struct MossImage {
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
}

impl MossImage {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("ppm"))
        {
            let mut ppm = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
            ppm.extend_from_slice(&self.rgb);
            return std::fs::write(path, ppm).map_err(|e| format!("{}: {e}", path.display()));
        }
        image::RgbImage::from_raw(self.width as u32, self.height as u32, self.rgb.clone())
            .ok_or("MOSS output dimensions overflow")?
            .save(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct MossTimings {
    pub text_encode: f64,
    pub dit_load: f64,
    pub text_prepare: f64,
    pub steps: Vec<f64>,
    pub vae: f64,
    pub total: f64,
}

impl MossTimings {
    pub fn median_step(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let mut v = self.steps.clone();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
}

/// Source MOSS V2 grid: shifted inference samples are snapped to the nearest
/// unshifted training sigma grid before Euler integration.  The returned
/// vector includes terminal zero; its denoising length can be lower than the
/// requested count only when nearest-grid snaps coalesce adjacent values.
pub fn moss_sigmas(steps: usize, shift: f32, training_shift: f32) -> Result<Vec<f32>, String> {
    moss_sigmas_for_training_grid(steps, shift, training_shift, 1000)
}

fn moss_sigmas_for_training_grid(
    steps: usize,
    shift: f32,
    training_shift: f32,
    total: usize,
) -> Result<Vec<f32>, String> {
    if steps == 0
        || total < 2
        || !shift.is_finite()
        || !training_shift.is_finite()
        || shift <= 0.0
        || training_shift <= 0.0
    {
        return Err("MOSS steps and shifts must be finite positive values".into());
    }
    let grid = |count: usize, s: f32| -> Vec<f32> {
        let lower = s / (total as f32 + s - 1.0);
        if count == 1 {
            return vec![1.0];
        }
        (0..count)
            .map(|i| {
                let v = 1.0 + (lower - 1.0) * (i as f32 / (count - 1) as f32);
                s * v / (1.0 + (s - 1.0) * v)
            })
            .collect()
    };
    let proposed = grid(steps, shift);
    let mut training = grid(total, training_shift);
    training.reverse();
    let mut snapped = Vec::with_capacity(steps + 1);
    for sigma in proposed {
        let upper = training.partition_point(|v| *v < sigma).min(total - 1);
        let lower = upper.saturating_sub(1);
        let choose = if (sigma - training[lower]).abs() <= (training[upper] - sigma).abs() {
            lower
        } else {
            upper
        };
        let value = training[choose];
        if snapped.last().copied() != Some(value) {
            snapped.push(value);
        }
    }
    if snapped.is_empty()
        || snapped
            .windows(2)
            .any(|s| !s[0].is_finite() || s[0] <= s[1])
    {
        return Err("MOSS scheduler produced an invalid non-descending sigma grid".into());
    }
    snapped.push(0.0);
    Ok(snapped)
}

fn prompt_ids(pipeline: &Pipeline, prompt: &str, max: usize) -> Result<Vec<u32>, String> {
    if max == 0 || max > MAX_TOKENS {
        return Err(format!("--max-sequence-length must be in 1..={MAX_TOKENS}"));
    }
    let messages = [("user".to_string(), prompt.to_string())];
    // The stored official Jinja template receives add_generation_prompt=true
    // and enable_thinking=true, exactly as MOSSImagePipeline.encode_prompt.
    let mut ids = pipeline
        .tokenizer
        .apply_chat_template_opts(&messages, Some(true));
    ids.truncate(max);
    if ids.is_empty() {
        return Err("MOSS template/tokenizer produced no tokens".into());
    }
    Ok(ids)
}

fn gauss(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut uniform = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let a = uniform().max(1e-300);
        let b = uniform();
        let r = (-2.0 * a.ln()).sqrt();
        let phase = 2.0 * std::f64::consts::PI * b;
        out.push((r * phase.cos()) as f32);
        if out.len() < n {
            out.push((r * phase.sin()) as f32);
        }
    }
    out
}

fn pixels_to_u8(rgb: &[f32], height: usize, width: usize) -> Result<Vec<u8>, String> {
    let plane = height
        .checked_mul(width)
        .ok_or("MOSS image dimensions overflow")?;
    if rgb.len() != plane * 3 {
        return Err(format!(
            "Flux2 RGB has {} values, expected {}",
            rgb.len(),
            plane * 3
        ));
    }
    let mut out = vec![0u8; plane * 3];
    for px in 0..plane {
        for c in 0..3 {
            out[px * 3 + c] =
                ((rgb[c * plane + px] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
        }
    }
    Ok(out)
}

/// Generate MOSS images from a complete single-file CMF.  Stages deliberately
/// own independent device scopes, so mapped Qwen3.5 weights are released
/// before the DiT, and DiT planes are released before the VAE decoder.
pub fn generate_images(
    model_path: &Path,
    prompt: &str,
    p: &MossParams,
    mut progress: impl FnMut(usize, usize, usize),
) -> Result<(Vec<MossImage>, MossTimings), String> {
    if p.height == 0 || p.width == 0 || p.height % 16 != 0 || p.width % 16 != 0 {
        return Err(format!(
            "MOSS --height/--width must be positive multiples of 16 (got {}x{})",
            p.width, p.height
        ));
    }
    if p.height > 4096 || p.width > 4096 {
        return Err("MOSS dimensions may not exceed 4096".into());
    }
    if p.num_images == 0
        || p.steps == 0
        || !p.cfg.is_finite()
        || p.cfg < 0.0
        || !p.shift.is_finite()
        || p.shift <= 0.0
    {
        return Err("invalid MOSS steps, cfg or shift".into());
    }
    let all_started = Instant::now();
    let model = CmfModel::open(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?;
    if model.header.arch.arch_name != ARCH_NAME {
        return Err(format!(
            "{} has architecture {:?}, expected {ARCH_NAME}",
            model_path.display(),
            model.header.arch.arch_name
        ));
    }
    let defaults = MossDefaults::of(&model);
    let scheduler = MossScheduler::of(&model)?;
    let neg_prompt = p
        .negative_prompt
        .clone()
        .unwrap_or_else(|| defaults.negative_prompt.clone());
    // The official path performs CFG whenever cfg != 1 (including cfg=0).
    let do_cfg = p.cfg != 1.0;
    let mut timing = MossTimings::default();

    // Every component gets a separately opened CMF mapping below.  This is
    // intentional rather than redundant: backend weight caches are keyed by
    // a model UID.  Dropping each stage's mapping before the next one makes
    // text → DiT → VAE *actual* staged GPU residency on Metal as well as
    // WGPU/Vulkan, while the OS page cache still shares the file contents.
    drop(model);

    // ── Qwen3.5 conditioning (then release all text weights/device maps) ──
    let text_started = Instant::now();
    let (positive, positive_tokens, negative, negative_tokens) = {
        // Register this file with the stage guard even when the selected
        // backend ultimately declines text kernels: a future opt-in
        // `CMF_MOSS_TE_GPU=1` must not leave its mapped Qwen planes resident
        // when the much larger DiT begins.
        let mut stage = crate::gpu::image_stage_scope();
        let stage_model = Arc::new(
            CmfModel::open(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?,
        );
        stage.track_model(stage_model.uid());
        // Hybrid GDN text encoding is normally faster/more stable on the
        // host; opt in to its device experiments with CMF_MOSS_TE_GPU=1.
        let _cpu =
            (std::env::var("CMF_MOSS_TE_GPU").as_deref() != Ok("1")).then(crate::gpu::pause_gpu);
        let mut encoder = Pipeline::from_encoder_model(&stage_model, SamplerConfig::default())
            .map_err(|e| format!("MOSS Qwen3.5 encoder: {e}"))?;
        let pos_ids = prompt_ids(&encoder, prompt, p.max_tokens)?;
        let pos_tokens = pos_ids.len();
        let pos = encoder.encode_hidden_states(&pos_ids)?;
        let (neg, neg_tokens) = if do_cfg {
            let ids = prompt_ids(&encoder, &neg_prompt, p.max_tokens)?;
            let n = ids.len();
            (Some(encoder.encode_hidden_states(&ids)?), Some(n))
        } else {
            (None, None)
        };
        (pos, pos_tokens, neg, neg_tokens)
    };
    timing.text_encode = text_started.elapsed().as_secs_f64();

    // Loading/dequant metadata of Flux2 is cheap compared with the 42-block
    // DiT and can happen on a CPU thread while denoising uses the GPU.  Do
    // not prewarm its device chain here: Metal/WGPU contexts are shared, and
    // concurrent queue use would defeat the staged-residency contract (and
    // has historically been a source of driver hangs).
    let vae_loader = {
        let path = model_path.to_path_buf();
        std::thread::spawn(move || -> Result<Flux2VaeDecoder, String> {
            let stage_model =
                Arc::new(CmfModel::open(&path).map_err(|e| format!("{}: {e}", path.display()))?);
            Flux2VaeDecoder::from_cmf(&stage_model)
        })
    };

    // ── MOSS DiT ──
    let (latents, latent_h, latent_w) = {
        let mut stage = crate::gpu::image_stage_scope();
        let stage_model = Arc::new(
            CmfModel::open(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?,
        );
        stage.track_model(stage_model.uid());
        let started = Instant::now();
        let dit = MossDit::from_cmf(&stage_model)?;
        timing.dit_load = started.elapsed().as_secs_f64();
        let positive = dit.prepare_text(&positive, positive_tokens)?;
        let negative = match (&negative, negative_tokens) {
            (Some(v), Some(n)) => Some((dit.prepare_text(v, n)?, n)),
            _ => None,
        };
        timing.text_prepare = started.elapsed().as_secs_f64() - timing.dit_load;
        let sigmas = moss_sigmas_for_training_grid(
            p.steps,
            p.shift,
            scheduler.training_shift,
            scheduler.num_train_timesteps,
        )?;
        let actual_steps = sigmas.len() - 1;
        let skips = p
            .uncond_skip_layers
            .as_deref()
            .unwrap_or(&defaults.uncond_skip_layers);
        if skips.iter().any(|&i| i >= dit.cfg.num_layers) {
            return Err(format!(
                "MOSS uncond skip layer outside 0..{}",
                dit.cfg.num_layers
            ));
        }
        let (lh, lw) = (p.height / 16, p.width / 16);
        let mut completed = Vec::with_capacity(p.num_images);
        for image in 0..p.num_images {
            let mut latent = gauss(
                dit.cfg.latent_channels * lh * lw,
                p.seed.wrapping_add(image as u64),
            );
            for step in 0..actual_steps {
                let clock = Instant::now();
                // The source FlowMatch scheduler exposes timestep = sigma ×
                // its recorded training-grid length.
                let t = sigmas[step] * scheduler.num_train_timesteps as f32;
                let mut prediction =
                    dit.forward(&latent, lh, lw, t, &positive, positive_tokens, &[])?;
                if let Some((negative, neg_len)) = &negative {
                    let uncond = dit.forward(&latent, lh, lw, t, negative, *neg_len, skips)?;
                    for (v, &u) in prediction.iter_mut().zip(&uncond) {
                        *v = u + p.cfg * (*v - u);
                    }
                }
                let dt = sigmas[step + 1] - sigmas[step];
                for (x, v) in latent.iter_mut().zip(&prediction) {
                    *x += dt * v;
                }
                if latent.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "MOSS image {image}: non-finite latent at step {}",
                        step + 1
                    ));
                }
                timing.steps.push(clock.elapsed().as_secs_f64());
                progress(image, step + 1, actual_steps);
            }
            completed.push(latent);
        }
        (completed, lh, lw)
    };

    // ── Flux2 VAE ──
    let vae_started = Instant::now();
    let images = {
        let _stage = crate::gpu::image_stage_scope();
        let vae = vae_loader
            .join()
            .map_err(|_| "MOSS VAE loader panicked".to_string())??;
        // Prewarm only after the DiT stage and its owner buffers are gone.
        // This keeps the VAE's reusable chain hot without overlapping two
        // independent GPU command streams.
        if crate::gpu::enabled_here() {
            let _ = vae.prewarm();
        }
        let mut images = Vec::with_capacity(latents.len());
        for (image, latent) in latents.iter().enumerate() {
            let rgb = vae.decode_fast(latent, latent_h, latent_w)?;
            images.push(MossImage {
                rgb: pixels_to_u8(&rgb, p.height, p.width)?,
                height: p.height,
                width: p.width,
                seed: p.seed.wrapping_add(image as u64),
            });
        }
        images
    };
    timing.vae = vae_started.elapsed().as_secs_f64();
    timing.total = all_started.elapsed().as_secs_f64();
    Ok((images, timing))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> serde_json::Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/moss_image_contract.json"
        )))
        .expect("valid MOSS contract fixture")
    }

    fn f32_le(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{} is f32-le", path.display());
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect()
    }

    #[test]
    fn scheduler_is_descending_and_ends_at_zero() {
        let s = moss_sigmas(50, 5.0, 1.0).expect("valid scheduler");
        assert_eq!(s.first().copied(), Some(1.0));
        assert_eq!(s.last().copied(), Some(0.0));
        assert!(s.windows(2).all(|p| p[0] > p[1]));
    }

    /// Keep the portable scheduler tied to values emitted by the pinned
    /// official `MOSSImageScheduler`, not just its qualitative shape.  The
    /// source's `torch.linspace` operates in f32, hence a small tolerance
    /// for values such as 0.46300003 that occur after shift/snap arithmetic.
    #[test]
    fn scheduler_matches_official_contract() {
        let fixture = contract();
        for schedule in fixture["scheduler"]["schedules"]
            .as_array()
            .expect("scheduler cases")
        {
            let requested = schedule["steps_requested"].as_u64().expect("step count") as usize;
            let want = schedule["sigmas"].as_array().expect("sigmas");
            let got = moss_sigmas(requested, 5.0, 1.0).expect("official schedule");
            assert_eq!(got.len(), want.len(), "{requested} steps");
            for (index, (actual, expected)) in got.iter().zip(want).enumerate() {
                let expected = expected.as_f64().expect("numeric sigma") as f32;
                assert!(
                    (actual - expected).abs() <= 2e-6,
                    "{requested} steps, sigma {index}: got {actual:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn fixed_rng_is_repeatable() {
        assert_eq!(gauss(11, 42), gauss(11, 42));
        assert_ne!(gauss(11, 42), gauss(11, 43));
    }

    /// This opt-in integration gate proves that the embedded tokenizer and
    /// stored Jinja template preserve MOSS's nonstandard thinking suffix.
    /// It deliberately needs a real packed CMF, so ordinary source-only CI
    /// remains lightweight; the release pod sets `CMF_MOSS_TEST_MODEL`.
    #[test]
    fn packed_template_matches_source_contract_when_model_supplied() {
        let Ok(path) = std::env::var("CMF_MOSS_TEST_MODEL") else {
            return;
        };
        let model = Arc::new(CmfModel::open(&path).expect("open supplied MOSS CMF"));
        assert_eq!(model.header.arch.arch_name, ARCH_NAME);
        let encoder = Pipeline::from_encoder_model(&model, SamplerConfig::default())
            .expect("open packed MOSS text encoder");
        for case in contract()["tokenizer"]["cases"]
            .as_array()
            .expect("tokenizer cases")
        {
            let got = prompt_ids(
                &encoder,
                case["prompt"].as_str().expect("prompt"),
                MAX_TOKENS,
            )
            .expect("apply stored MOSS template");
            let want: Vec<u32> = case["ids"]
                .as_array()
                .expect("token ids")
                .iter()
                .map(|id| id.as_u64().expect("integer token") as u32)
                .collect();
            assert_eq!(got, want, "prompt {:?}", case["prompt"]);
        }
    }

    /// End-to-end Flux2 decoder oracle.  The raw tensor fixtures are emitted
    /// by the pinned official pipeline and intentionally live outside Git;
    /// release validation supplies their directory on the pod.  Run it with
    /// `CMF_GPU=0` for an architecture-independent f32 comparison.
    #[test]
    fn packed_flux2_decoder_matches_source_oracle_when_supplied() {
        let (Ok(model_path), Ok(oracle_dir)) = (
            std::env::var("CMF_MOSS_TEST_MODEL"),
            std::env::var("CMF_MOSS_ORACLE_DIR"),
        ) else {
            return;
        };
        let model = CmfModel::open(&model_path).expect("open supplied MOSS CMF");
        let vae = Flux2VaeDecoder::from_cmf(&model).expect("open packed Flux2 decoder");
        let root = std::path::Path::new(&oracle_dir);
        let packed = f32_le(&root.join("vae_packed_latent.f32"));
        let want = f32_le(&root.join("vae_decode_ref.f32"));
        let got = vae
            .decode_fast(&packed, 3, 5)
            .expect("decode source packed latent");
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f64;
        for (&actual, &expected) in got.iter().zip(&want) {
            let delta = (actual - expected).abs();
            max_abs = max_abs.max(delta);
            mean_abs += delta as f64;
        }
        mean_abs /= got.len() as f64;
        assert!(
            max_abs <= 5e-3 && mean_abs <= 2e-4,
            "Flux2 oracle drift: max_abs={max_abs:.6}, mean_abs={mean_abs:.6}"
        );
    }
}
