//! Native Qwen-Image-Edit-2509 orchestration over independent CMF components.
//! Model stages are scoped separately; only conditioning and latents survive.

use crate::qwen_image::QwenImageTransformer;
use crate::qwen_image_encoder::{Conditioning, QwenImageEncoder};
use crate::qwen_image_vae::QwenImageVae;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use image::{RgbImage, imageops::FilterType};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub struct QwenImagePaths {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    pub scheduler: Option<PathBuf>,
}

pub struct QwenImageParams {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub true_cfg_scale: f32,
    pub negative_prompt: Option<String>,
    pub seed: u64,
    /// Square-root of reference VAE pixel area; the official default is 1024.
    pub reference_size: usize,
    /// Optional raw NCHW f32 noise for comparisons using identical latents.
    pub initial_latents: Option<PathBuf>,
}

impl Default for QwenImageParams {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            steps: 40,
            true_cfg_scale: 4.0,
            negative_prompt: Some(" ".into()),
            seed: 0,
            reference_size: 1024,
            initial_latents: None,
        }
    }
}

pub struct QwenImageOutput {
    /// Channels-first RGB in [0, 1].
    pub pixels: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

impl QwenImageOutput {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let plane = self.width * self.height;
        if self.pixels.len() != plane * 3 {
            return Err("invalid output RGB plane lengths".into());
        }
        let mut bytes = Vec::with_capacity(plane * 3);
        for p in 0..plane {
            for c in 0..3 {
                bytes.push((self.pixels[c * plane + p].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("ppm"))
        {
            let mut ppm = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
            ppm.extend(bytes);
            return std::fs::write(path, ppm).map_err(|e| e.to_string());
        }
        RgbImage::from_raw(self.width as u32, self.height as u32, bytes)
            .ok_or("output image dimensions overflow")?
            .save(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

pub fn edit_files(
    paths: &QwenImagePaths,
    prompt: &str,
    images: &[PathBuf],
    params: &QwenImageParams,
    progress: impl FnMut(&str, usize, usize),
) -> Result<QwenImageOutput, String> {
    let images = images
        .iter()
        .map(|path| {
            image::open(path)
                .map(|image| image.into_rgb8())
                .map_err(|e| format!("{}: {e}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    edit(paths, prompt, &images, params, progress)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct FlowMatchConfig {
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f32,
    pub max_shift: f32,
    pub shift_terminal: f32,
    pub num_train_timesteps: usize,
    pub use_dynamic_shifting: bool,
    pub time_shift_type: String,
    pub invert_sigmas: bool,
    pub stochastic_sampling: bool,
    pub use_karras_sigmas: bool,
    pub use_exponential_sigmas: bool,
    pub use_beta_sigmas: bool,
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        Self {
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: 0.02,
            num_train_timesteps: 1000,
            use_dynamic_shifting: true,
            time_shift_type: "exponential".into(),
            invert_sigmas: false,
            stochastic_sampling: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        }
    }
}

/// Diffusers FlowMatchEulerDiscreteScheduler, including terminal stretching.
/// Resolution shifting uses generated tokens, excluding reference tokens.
pub fn flow_match_sigmas(
    steps: usize,
    generated_tokens: usize,
    config: &FlowMatchConfig,
) -> Result<Vec<f32>, String> {
    if steps < 2 || steps > 1000 || generated_tokens == 0 {
        return Err("Qwen Image needs 2..=1000 steps and a nonempty latent grid".into());
    }
    if !config.use_dynamic_shifting
        || config.time_shift_type != "exponential"
        || config.invert_sigmas
        || config.stochastic_sampling
        || config.use_karras_sigmas
        || config.use_exponential_sigmas
        || config.use_beta_sigmas
        || config.num_train_timesteps != 1000
        || config.max_image_seq_len <= config.base_image_seq_len
        || !config.base_shift.is_finite()
        || !config.max_shift.is_finite()
        || !(0.0..1.0).contains(&config.shift_terminal)
    {
        return Err(
            "scheduler is not the supported Qwen Image exponential FlowMatch Euler contract".into(),
        );
    }
    let slope = (config.max_shift as f64 - config.base_shift as f64)
        / (config.max_image_seq_len - config.base_image_seq_len) as f64;
    let mu = config.base_shift as f64
        + slope * (generated_tokens as f64 - config.base_image_seq_len as f64);
    let exp_mu = mu.exp() as f32;
    let mut sigmas = (0..steps)
        .map(|i| {
            let sigma = (1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1) as f64) as f32;
            exp_mu / (exp_mu + (1.0 / sigma - 1.0))
        })
        .collect::<Vec<_>>();
    if config.shift_terminal != 0.0 {
        let scale = (1.0 - sigmas[steps - 1]) / (1.0 - config.shift_terminal);
        if !scale.is_finite() || scale <= 0.0 {
            return Err("invalid terminal-stretch scale".into());
        }
        for sigma in &mut sigmas {
            *sigma = 1.0 - (1.0 - *sigma) / scale;
        }
    }
    sigmas.push(0.0);
    if sigmas.iter().any(|v| !v.is_finite()) || sigmas.windows(2).any(|w| w[0] <= w[1]) {
        return Err("Qwen Image scheduler did not produce finite descending sigmas".into());
    }
    Ok(sigmas)
}

fn latent_len(channels: usize, height: usize, width: usize) -> Result<usize, String> {
    channels
        .checked_mul(height)
        .and_then(|v| v.checked_mul(width))
        .filter(|&v| v > 0)
        .ok_or_else(|| "empty or overflowing latent dimensions".into())
}

/// NCHW -> token-major [h/2*w/2, channels*4], with channel,dy,dx order.
pub fn pack_latents(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    if height % 2 != 0 || width % 2 != 0 || input.len() != latent_len(channels, height, width)? {
        return Err("latent packing needs even spatial dimensions and exact NCHW data".into());
    }
    let mut out = Vec::with_capacity(input.len());
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            for channel in 0..channels {
                for dy in 0..2 {
                    for dx in 0..2 {
                        out.push(input[(channel * height + y + dy) * width + x + dx]);
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn unpack_latents(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    if height % 2 != 0 || width % 2 != 0 || input.len() != latent_len(channels, height, width)? {
        return Err("latent unpacking needs even spatial dimensions and exact packed data".into());
    }
    let mut out = vec![0.0; input.len()];
    let mut i = 0;
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            for channel in 0..channels {
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[(channel * height + y + dy) * width + x + dx] = input[i];
                        i += 1;
                    }
                }
            }
        }
    }
    Ok(out)
}

#[derive(Deserialize)]
struct VaeScale {
    z_dim: usize,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

fn vae_scale(path: &Path) -> Result<VaeScale, String> {
    let model = CmfModel::open(path).map_err(|e| e.to_string())?;
    let config_name = if model.tensor("image.vae.config_json").is_some() {
        "image.vae.config_json"
    } else {
        "image.config_json"
    };
    let config: VaeScale = serde_json::from_slice(
        model
            .tensor_bytes(config_name)
            .map_err(|e| format!("VAE CMF image.config_json: {e}"))?,
    )
    .map_err(|e| format!("VAE configuration: {e}"))?;
    if config.z_dim != 16
        || config.latents_mean.len() != 16
        || config.latents_std.len() != 16
        || config.latents_mean.iter().any(|v| !v.is_finite())
        || config
            .latents_std
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(
            "Qwen Image requires 16 finite VAE means and positive standard deviations".into(),
        );
    }
    Ok(config)
}

fn embedded_scheduler(path: &Path) -> Result<Option<FlowMatchConfig>, String> {
    let model = CmfModel::open(path).map_err(|e| e.to_string())?;
    let Some(entry) = model.tensor("image.scheduler_config_json") else {
        return Ok(None);
    };
    if entry.dtype != cortiq_core::TensorDtype::U8
        || entry.shape.len() != 1
        || entry.shape[0] != entry.n_elems()
    {
        return Err("embedded Qwen scheduler must be a one-dimensional U8 blob".into());
    }
    serde_json::from_slice(model.entry_bytes(entry))
        .map(Some)
        .map_err(|e| format!("embedded scheduler configuration: {e}"))
}

fn normalize_latents(data: &mut [f32], config: &VaeScale, decode: bool) -> Result<(), String> {
    if data.is_empty() || config.z_dim == 0 || data.len() % config.z_dim != 0 {
        return Err("invalid VAE channel planes".into());
    }
    let plane = data.len() / config.z_dim;
    for (channel, row) in data.chunks_exact_mut(plane).enumerate() {
        for x in row {
            *x = if decode {
                *x * config.latents_std[channel] + config.latents_mean[channel]
            } else {
                (*x - config.latents_mean[channel]) / config.latents_std[channel]
            };
        }
    }
    Ok(())
}

fn gaussian_noise(count: usize, seed: u64, path: Option<&Path>) -> Result<Vec<f32>, String> {
    if let Some(path) = path {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() != count * 4 {
            return Err("initial latent file has the wrong size".into());
        }
        let data = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if data.iter().any(|v| !v.is_finite()) {
            return Err("initial latents contain nonfinite values".into());
        }
        return Ok(data);
    }
    let mut rng = SplitMix64::new(seed);
    let mut data = Vec::with_capacity(count);
    while data.len() < count {
        let a = ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
        let b = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let radius = (-2.0 * a.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * b;
        data.push((radius * angle.cos()) as f32);
        if data.len() < count {
            data.push((radius * angle.sin()) as f32);
        }
    }
    Ok(data)
}

fn validate_conditioning(value: &Conditioning) -> Result<(), String> {
    if value.seq_len == 0
        || value.hidden_size == 0
        || value.hidden.len()
            != value
                .seq_len
                .checked_mul(value.hidden_size)
                .ok_or("conditioning size overflow")?
        || value.hidden.iter().any(|v| !v.is_finite())
    {
        return Err("text encoder returned invalid or nonfinite conditioning".into());
    }
    Ok(())
}

/// True CFG and Qwen's per-token vector-norm correction.
fn rescale_cfg(conditional: &mut [f32], unconditional: &[f32], scale: f32) -> Result<(), String> {
    if conditional.len() != unconditional.len() || conditional.len() % 64 != 0 || !scale.is_finite()
    {
        return Err("invalid packed CFG prediction shape or scale".into());
    }
    for (cond, uncond) in conditional
        .chunks_exact_mut(64)
        .zip(unconditional.chunks_exact(64))
    {
        let cond_norm = cond.iter().map(|v| v * v).sum::<f32>().sqrt();
        for (c, u) in cond.iter_mut().zip(uncond) {
            *c = *u + scale * (*c - *u);
        }
        let combined_norm = cond.iter().map(|v| v * v).sum::<f32>().sqrt();
        if combined_norm > 0.0 {
            for v in cond {
                *v *= cond_norm / combined_norm;
            }
        }
    }
    Ok(())
}

/// Native image editing; events are stage name, completed steps, total steps.
pub fn edit(
    paths: &QwenImagePaths,
    prompt: &str,
    images: &[RgbImage],
    params: &QwenImageParams,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<QwenImageOutput, String> {
    let (height, width) = (params.height / 16 * 16, params.width / 16 * 16);
    if images.is_empty() || images.iter().any(|i| i.width() == 0 || i.height() == 0) {
        return Err("Qwen Image Edit requires at least one reference image".into());
    }
    if height == 0
        || width == 0
        || height > 2048
        || width > 2048
        || params.reference_size < 32
        || params.reference_size > 2048
        || !params.true_cfg_scale.is_finite()
        || params.true_cfg_scale < 0.0
    {
        return Err("invalid Qwen Image dimensions, reference size, or CFG scale".into());
    }
    for path in [&paths.transformer, &paths.text_encoder, &paths.vae] {
        if !path.is_file() {
            return Err(format!("missing Qwen Image component: {}", path.display()));
        }
    }
    let scheduler = match &paths.scheduler {
        Some(path) => serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("scheduler configuration: {e}"))?,
        None => embedded_scheduler(&paths.transformer)?.unwrap_or_default(),
    };
    let (lh, lw) = (height / 8, width / 8);
    let sigmas = flow_match_sigmas(params.steps, lh * lw / 4, &scheduler)?;
    let scale = vae_scale(&paths.vae)?;
    progress("text encoder", 0, params.steps);
    let (positive, negative) = {
        let mut stage = crate::gpu::image_stage_scope();
        let encoder = QwenImageEncoder::open(&paths.text_encoder)?;
        stage.track_model(encoder.model_uid());
        let positive = encoder.encode(prompt, images)?;
        validate_conditioning(&positive)?;
        let negative = if params.true_cfg_scale > 1.0 {
            params
                .negative_prompt
                .as_deref()
                .map(|p| encoder.encode(p, images))
                .transpose()?
        } else {
            None
        };
        if let Some(value) = &negative {
            validate_conditioning(value)?;
        }
        (positive, negative)
    };
    progress("reference VAE", 0, params.steps);
    let mut shapes = vec![[1, lh / 2, lw / 2]];
    let reference = {
        let _stage = crate::gpu::image_stage_scope();
        let vae = QwenImageVae::open(&paths.vae)?;
        let mut packed = Vec::new();
        for image in images {
            let ratio = image.width() as f64 / image.height() as f64;
            let w = (params.reference_size as f64 * ratio.sqrt() / 32.0)
                .round_ties_even()
                .max(1.0) as usize
                * 32;
            let h = (params.reference_size as f64 / ratio.sqrt() / 32.0)
                .round_ties_even()
                .max(1.0) as usize
                * 32;
            if w > 32768 || h > 32768 || packed.len() / 64 + w / 16 * (h / 16) + lh * lw / 4 > 16384
            {
                return Err("reference images exceed the Qwen Image token budget".into());
            }
            let resized = image::imageops::resize(image, w as u32, h as u32, FilterType::Lanczos3);
            let mut pixels = vec![0.0; w * h * 3];
            for (i, pixel) in resized.pixels().enumerate() {
                for channel in 0..3 {
                    pixels[channel * w * h + i] = pixel[channel] as f32 / 127.5 - 1.0;
                }
            }
            let mut latent = vae.encode_mean(&pixels, h, w)?;
            if latent.len() != latent_len(16, h / 8, w / 8)? {
                return Err("VAE encoder returned the wrong latent shape".into());
            }
            normalize_latents(&mut latent, &scale, false)?;
            packed.extend(pack_latents(&latent, 16, h / 8, w / 8)?);
            shapes.push([1, h / 16, w / 16]);
        }
        packed
    };
    let initial_path = params
        .initial_latents
        .clone()
        .or_else(|| std::env::var_os("CMF_INIT_LATENT").map(PathBuf::from));
    let noise = gaussian_noise(
        latent_len(16, lh, lw)?,
        params.seed,
        initial_path.as_deref(),
    )?;
    let mut latents = pack_latents(&noise, 16, lh, lw)?;
    drop(noise);
    progress("denoiser", 0, params.steps);
    {
        let mut stage = crate::gpu::image_stage_scope();
        let transformer = QwenImageTransformer::open(&paths.transformer)?;
        stage.track_model(transformer.model_uid());
        let mut input = Vec::with_capacity(latents.len() + reference.len());
        for step in 0..params.steps {
            input.clear();
            input.extend_from_slice(&latents);
            input.extend_from_slice(&reference);
            let mut prediction = transformer.forward(
                &input,
                &positive.hidden,
                &shapes,
                positive.seq_len,
                sigmas[step],
            )?;
            if prediction.len() != input.len() {
                return Err("transformer output shape does not match packed image tokens".into());
            }
            prediction.truncate(latents.len());
            if let Some(negative) = &negative {
                let unconditional = transformer.forward(
                    &input,
                    &negative.hidden,
                    &shapes,
                    negative.seq_len,
                    sigmas[step],
                )?;
                if unconditional.len() != input.len() {
                    return Err("negative transformer output has the wrong shape".into());
                }
                rescale_cfg(
                    &mut prediction,
                    &unconditional[..latents.len()],
                    params.true_cfg_scale,
                )?;
            }
            let dt = sigmas[step + 1] - sigmas[step];
            for (x, dx) in latents.iter_mut().zip(prediction) {
                *x += dt * dx;
            }
            if latents.iter().any(|v| !v.is_finite()) {
                return Err(format!("nonfinite latent at step {}", step + 1));
            }
            progress("denoiser", step + 1, params.steps);
        }
    }
    drop(reference);
    drop(positive);
    drop(negative);
    progress("decode VAE", params.steps, params.steps);
    let mut raw = unpack_latents(&latents, 16, lh, lw)?;
    drop(latents);
    normalize_latents(&mut raw, &scale, true)?;
    let mut pixels = {
        let _stage = crate::gpu::image_stage_scope();
        QwenImageVae::open(&paths.vae)?.decode(&raw, lh, lw)?
    };
    if pixels.len() != latent_len(3, height, width)? || pixels.iter().any(|v| !v.is_finite()) {
        return Err("VAE decoder returned invalid RGB pixels".into());
    }
    for value in &mut pixels {
        *value = (*value * 0.5 + 0.5).clamp(0.0, 1.0);
    }
    Ok(QwenImageOutput {
        pixels,
        height,
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/qwen_image_pipeline.json")).unwrap()
    }
    fn values(v: &serde_json::Value) -> Vec<f32> {
        serde_json::from_value(v.clone()).unwrap()
    }
    fn close(got: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= tolerance,
                "element {i}: {a} != {b}"
            );
        }
    }
    #[test]
    fn official_diffusers_flowmatch_schedules() {
        let reference = fixture();
        for case in reference["schedules"].as_array().unwrap() {
            let result = flow_match_sigmas(
                case["steps"].as_u64().unwrap() as usize,
                case["tokens"].as_u64().unwrap() as usize,
                &FlowMatchConfig::default(),
            )
            .unwrap();
            close(&result, &values(&case["sigmas"]), 3e-7);
        }
    }
    #[test]
    fn official_diffusers_non_square_latent_layout() {
        let reference = fixture();
        close(
            &pack_latents(&values(&reference["nchw"]), 2, 4, 6).unwrap(),
            &values(&reference["packed"]),
            0.0,
        );
        close(
            &unpack_latents(&values(&reference["packed"]), 2, 4, 6).unwrap(),
            &values(&reference["unpacked"]),
            0.0,
        );
        assert!(pack_latents(&[0.0; 6], 1, 3, 2).is_err());
    }
    #[test]
    fn official_diffusers_cfg_rescales_each_image_token() {
        let reference = fixture();
        let mut prediction = values(&reference["cond"]);
        rescale_cfg(&mut prediction, &values(&reference["uncond"]), 4.0).unwrap();
        close(&prediction, &values(&reference["cfg"]), 5e-7);
    }
}
