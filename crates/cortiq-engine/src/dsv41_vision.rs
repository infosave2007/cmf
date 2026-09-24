//! DeepSeek-V4.1 vision tower, image preprocessing, and image-span layout.
//!
//! This is a direct CPU port of the pinned model's `inference/vision.py` and
//! `inference/image_processor.py`.  The model's vision weights stay in the
//! CMF mapping through [`QTensor`]; only the activations are materialised.
//! The public preparation types are deliberately independent of the text
//! pipeline so an OpenAI server, CLI, or another caller can build the same
//! image spans before handing them to the runtime owner.

use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use image::imageops::{self, FilterType};
use image::{Rgb, RgbImage};
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of vision attention calls that completed through the selected GPU
/// backend in this process.  The component probe reports this so a Vulkan
/// timing cannot be mistaken for a CPU fallback.
static GPU_ATTENTION_DISPATCHES: AtomicUsize = AtomicUsize::new(0);
static GPU_DENSE_GEMM_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

pub fn gpu_attention_dispatches() -> usize {
    GPU_ATTENTION_DISPATCHES.load(Ordering::Relaxed)
}

pub fn gpu_dense_gemm_dispatches() -> usize {
    GPU_DENSE_GEMM_DISPATCHES.load(Ordering::Relaxed)
}

/// Text positions carry this type; non-negative values identify image-span
/// positions.  The values match the reference processor exactly.
pub const TEXT: i8 = -1;
pub const IMAGE_START: i8 = 0;
pub const IMAGE: i8 = 1;
pub const IMAGE_NEW_LINE: i8 = 2;
pub const IMAGE_END: i8 = 3;

/// Minimal source configuration needed by the V4.1 vision tower and image
/// processor.  `from_source` accepts the complete checkpoint config and
/// reads its nested `vision_config`/`text_config` objects.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub vision_n_layers: usize,
    pub vision_dim: usize,
    pub vision_n_heads: usize,
    pub vision_inter_dim: usize,
    pub vision_patch_size: usize,
    pub vision_rope_theta: f32,
    pub vision_downsample_ratio: usize,
    pub vision_max_n_token: usize,
    pub vision_min_pixels: usize,
    pub vision_max_wh_ratio: Option<f64>,
    pub text_dim: usize,
    pub image_token_id: u32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            vision_n_layers: 0,
            vision_dim: 1024,
            vision_n_heads: 16,
            vision_inter_dim: 2816,
            vision_patch_size: 14,
            vision_rope_theta: 10_000.0,
            vision_downsample_ratio: 3,
            vision_max_n_token: 1024,
            vision_min_pixels: 544 * 544,
            vision_max_wh_ratio: None,
            text_dim: 5120,
            image_token_id: 129_264,
        }
    }
}

impl VisionConfig {
    /// Read a complete DeepSeek-V4.1 config, or a vision-only object.
    pub fn from_source(source: &Value) -> Result<Self, String> {
        let mut out = Self::default();
        let vision = source.get("vision_config").unwrap_or(source);
        let empty = Value::Null;
        let text = source.get("text_config").unwrap_or(&empty);
        let usize_field = |object: &Value, key: &str, old: usize| {
            object
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(old)
        };
        let f32_field = |object: &Value, key: &str, old: f32| {
            object
                .get(key)
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .unwrap_or(old)
        };
        out.vision_n_layers = usize_field(vision, "num_hidden_layers", out.vision_n_layers);
        out.vision_dim = usize_field(vision, "hidden_size", out.vision_dim);
        out.vision_n_heads = usize_field(vision, "num_attention_heads", out.vision_n_heads);
        out.vision_inter_dim = usize_field(vision, "intermediate_size", out.vision_inter_dim);
        out.vision_patch_size = usize_field(vision, "patch_size", out.vision_patch_size);
        out.vision_rope_theta = f32_field(vision, "rope_theta", out.vision_rope_theta);
        out.vision_downsample_ratio =
            usize_field(vision, "downsample_ratio", out.vision_downsample_ratio);
        out.vision_max_n_token = usize_field(vision, "max_image_tokens", out.vision_max_n_token);
        out.vision_min_pixels = usize_field(vision, "min_pixels", out.vision_min_pixels);
        out.vision_max_wh_ratio = vision.get("max_wh_ratio").and_then(Value::as_f64);
        out.text_dim = usize_field(text, "hidden_size", out.text_dim);
        out.image_token_id = source
            .get("image_token_id")
            .and_then(Value::as_u64)
            .or_else(|| vision.get("image_token_id").and_then(Value::as_u64))
            .map(|v| v as u32)
            .unwrap_or(out.image_token_id);
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vision_n_layers == 0 {
            return Ok(());
        }
        if self.vision_dim == 0
            || self.vision_n_heads == 0
            || self.vision_dim % self.vision_n_heads != 0
            || self.vision_dim / self.vision_n_heads % 2 != 0
        {
            return Err(format!(
                "invalid vision attention geometry dim={} heads={}",
                self.vision_dim, self.vision_n_heads
            ));
        }
        if self.vision_patch_size == 0 || self.vision_downsample_ratio == 0 {
            return Err("vision patch_size and downsample_ratio must be non-zero".to_string());
        }
        if self.vision_max_n_token < 4 {
            return Err("vision max_image_tokens must be at least 4".to_string());
        }
        if self.text_dim == 0 {
            return Err("text hidden_size must be non-zero for the aligner".to_string());
        }
        Ok(())
    }

    pub fn vision_enabled(&self) -> bool {
        self.vision_n_layers > 0
    }

    pub fn head_dim(&self) -> usize {
        self.vision_dim / self.vision_n_heads
    }

    pub fn rope_dim(&self) -> usize {
        self.head_dim() / 2
    }
}

/// An image record after prompt encoding.  Patches use the same contiguous
/// `[n_vit_h*n_vit_w, 3, patch, patch]` order as PyTorch's reference.
#[derive(Clone, Debug)]
pub struct ImageInput {
    pub start: usize,
    pub patches: Vec<f32>,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub types: Vec<i8>,
}

impl ImageInput {
    pub fn image_positions(&self) -> usize {
        self.types.iter().filter(|&&kind| kind == IMAGE).count()
    }

    pub fn span_len(&self) -> usize {
        self.types.len()
    }
}

/// Token IDs plus per-position image type metadata consumed by the text
/// runtime.  `images` is empty for text-only prompts.
#[derive(Clone, Debug)]
pub struct PreparedVlInputs {
    pub token_ids: Vec<u32>,
    pub token_types: Vec<i8>,
    pub images: Vec<ImageInput>,
}

/// `num_image_tokens` from the pinned processor.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h.saturating_mul(n_llm_w + 1).saturating_add(2)
}

/// Number of LLM rows/columns after the aligner's `r×r` downsample.
pub fn llm_grid(
    best_height: usize,
    best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
) -> (usize, usize) {
    (
        (best_height / patch_size).div_ceil(downsample_ratio),
        (best_width / patch_size).div_ceil(downsample_ratio),
    )
}

/// Largest aspect-preserving pixel size whose token grid fits the cap.
pub fn solve_resize_ratio(
    height: usize,
    width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    solve_resize_ratio_f64(
        height.max(1) as f64,
        width.max(1) as f64,
        patch_size,
        downsample_ratio,
        max_n_token,
    )
}

fn solve_resize_ratio_f64(
    height_f: f64,
    width_f: f64,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let aspect = height_f / width_f;
    let max_w_float = (((max_n_token.saturating_sub(2)) as f64 / aspect) + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * aspect;
    let cell = patch_size.saturating_mul(downsample_ratio).max(1);
    if max_w_float < 1.0 {
        return (max_n_token.saturating_sub(2) / 2 * cell, cell);
    }
    if max_h_float < 1.0 {
        return (cell, max_n_token.saturating_sub(3) * cell);
    }
    let beta = (max_w_float.floor() * cell as f64 / width_f)
        .min(max_h_float.floor() * cell as f64 / height_f);
    let best_h =
        ((height_f * beta / patch_size.max(1) as f64).floor() as usize).saturating_mul(patch_size);
    let best_w =
        ((width_f * beta / patch_size.max(1) as f64).floor() as usize).saturating_mul(patch_size);
    (best_h.max(patch_size), best_w.max(patch_size))
}

/// Apply the official token-cap correction to a planned pixel grid.
pub fn safe_resize(
    height: usize,
    width: usize,
    mut best_height: usize,
    mut best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let (mut n_llm_h, mut n_llm_w) =
        llm_grid(best_height, best_width, patch_size, downsample_ratio);
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token {
        let (h, w) = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token);
        best_height = h;
        best_width = w;
        (n_llm_h, n_llm_w) = llm_grid(best_height, best_width, patch_size, downsample_ratio);
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token {
            return Err(format!(
                "image grid {}x{} costs {} tokens, cap {}",
                n_llm_h,
                n_llm_w,
                num_image_tokens(n_llm_h, n_llm_w),
                max_n_token
            ));
        }
    }
    Ok((n_llm_h, n_llm_w, best_height, best_width))
}

/// Compute the pixel/token plan before decoding the image payload.
pub fn plan_image_grid(
    width: usize,
    height: usize,
    config: &VisionConfig,
) -> Result<(usize, usize, usize, usize), String> {
    config.validate()?;
    if width == 0 || height == 0 {
        return Err("image dimensions must be non-zero".to_string());
    }
    let mut width_f = width as f64;
    let mut height_f = height as f64;
    if let Some(max_ratio) = config.vision_max_wh_ratio.filter(|v| *v > 0.0) {
        if width_f > height_f * max_ratio {
            width_f = height_f * max_ratio;
        }
    }
    if width_f * height_f < config.vision_min_pixels as f64 {
        let scale = (config.vision_min_pixels as f64 / (width_f * height_f)).sqrt();
        width_f = (width_f * scale) as usize as f64;
        height_f = (height_f * scale) as usize as f64;
    }
    let p = config.vision_patch_size;
    let mut best_width = (width_f.ceil() as usize)
        .max(1)
        .div_ceil(p)
        .saturating_mul(p);
    let mut best_height = (height_f.ceil() as usize)
        .max(1)
        .div_ceil(p)
        .saturating_mul(p);
    let (mut n_h, mut n_w) = llm_grid(best_height, best_width, p, config.vision_downsample_ratio);
    if num_image_tokens(n_h, n_w) > config.vision_max_n_token {
        let (h, w) = solve_resize_ratio_f64(
            height_f,
            width_f,
            p,
            config.vision_downsample_ratio,
            config.vision_max_n_token,
        );
        best_height = h;
        best_width = w;
        (n_h, n_w) = llm_grid(best_height, best_width, p, config.vision_downsample_ratio);
        if num_image_tokens(n_h, n_w) > config.vision_max_n_token {
            return Err(format!(
                "image grid {}x{} costs {} tokens, cap {}",
                n_h,
                n_w,
                num_image_tokens(n_h, n_w),
                config.vision_max_n_token
            ));
        }
    }
    Ok((n_h, n_w, best_height, best_width))
}

/// The exact span type order: start, each row's image tokens plus newline,
/// end.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i8> {
    let mut types = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    types.push(IMAGE_START);
    for _ in 0..n_llm_h {
        types.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        types.push(IMAGE_NEW_LINE);
    }
    types.push(IMAGE_END);
    types
}

/// Moved to [`crate::media`] so every vision front end shares one fetcher.
pub use crate::media::load_image_bytes;

fn resize_fit(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let scale = (width as f64 / image.width() as f64).min(height as f64 / image.height() as f64);
    let resized_width = (image.width() as f64 * scale).round().max(1.0) as u32;
    let resized_height = (image.height() as f64 * scale).round().max(1.0) as u32;
    imageops::resize(image, resized_width, resized_height, FilterType::CatmullRom)
}

fn pad_to(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let resized = resize_fit(image, width, height);
    let mut output = RgbImage::from_pixel(width, height, Rgb([127, 127, 127]));
    let left = (width.saturating_sub(resized.width())) / 2;
    let top = (height.saturating_sub(resized.height())) / 2;
    imageops::overlay(&mut output, &resized, i64::from(left), i64::from(top));
    output
}

/// PyTorch converts the normalized image tensor to bfloat16 before the ViT
/// sees it.  The rest of the CPU engine works in f32, so round-trip each
/// patch value through BF16 at this boundary instead of silently retaining
/// extra image precision.
#[inline]
fn bf16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(round) & 0xffff_0000))
}

/// Decode, resize/pad, normalize, and patchify one image record.
pub fn load_image(
    record: &Value,
    config: &VisionConfig,
) -> Result<(Vec<f32>, usize, usize, usize, usize), String> {
    config.validate()?;
    if !config.vision_enabled() {
        return Err("image input requires a model with vision_n_layers > 0".to_string());
    }
    let bytes = load_image_bytes(record)?;
    let decoded = image::load_from_memory(&bytes)
        .map_err(|e| format!("image decode failed: {e}"))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let (n_llm_h, n_llm_w, best_height, best_width) =
        plan_image_grid(width as usize, height as usize, config)?;
    let p = config.vision_patch_size as u32;
    let target_width = best_width as u32;
    let target_height = best_height as u32;
    let transformed = if config
        .vision_max_wh_ratio
        .is_some_and(|ratio| width as f64 >= ratio * height as f64)
    {
        imageops::resize(
            &decoded,
            target_width,
            target_height,
            FilterType::CatmullRom,
        )
    } else {
        pad_to(&decoded, target_width, target_height)
    };
    let n_vit_h = best_height / config.vision_patch_size;
    let n_vit_w = best_width / config.vision_patch_size;
    let patch_values = config
        .vision_patch_size
        .saturating_mul(config.vision_patch_size)
        .saturating_mul(3);
    let mut patches =
        Vec::with_capacity(n_vit_h.saturating_mul(n_vit_w).saturating_mul(patch_values));
    // Equivalent to PyTorch:
    // x.reshape(3,n_h,p,n_w,p).permute(1,3,0,2,4).reshape(n_h*n_w,3,p,p)
    for patch_y in 0..n_vit_h {
        for patch_x in 0..n_vit_w {
            for channel in 0..3 {
                for dy in 0..config.vision_patch_size {
                    for dx in 0..config.vision_patch_size {
                        let pixel = transformed.get_pixel(
                            (patch_x * config.vision_patch_size + dx) as u32,
                            (patch_y * config.vision_patch_size + dy) as u32,
                        );
                        let value = (pixel[channel] as f32 / 255.0 - 0.5) / 0.5;
                        patches.push(bf16_roundtrip(value));
                    }
                }
            }
        }
    }
    debug_assert_eq!(patches.len(), n_vit_h * n_vit_w * patch_values);
    Ok((patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w))
}

/// Tokenize a prompt and replace each image placeholder with its official
/// span.  The tokenizer must expose `<｜deepseek_image｜>` as one added token;
/// a mismatched known id is rejected rather than silently feeding text tokens.
pub fn prepare_vl_inputs(
    prompt: &str,
    images: &[Value],
    tokenizer: &Tokenizer,
    config: &VisionConfig,
) -> Result<PreparedVlInputs, String> {
    config.validate()?;
    let image_token_id = config.image_token_id;
    if let Some(placeholder_id) = tokenizer.token_to_id(crate::dsv41_encoding::IMAGE_PLACEHOLDER) {
        if placeholder_id != image_token_id {
            return Err(format!(
                "tokenizer image placeholder id {} != config image_token_id {}",
                placeholder_id, image_token_id
            ));
        }
    }
    let prompt_tokens = tokenizer.encode(prompt);
    let placeholders = prompt_tokens
        .iter()
        .filter(|&&token| token == image_token_id)
        .count();
    if placeholders != images.len() {
        return Err(format!(
            "found {placeholders} image tokens but received {} images",
            images.len()
        ));
    }
    if placeholders > 0 && !config.vision_enabled() {
        return Err("prompt contains images but the model has no vision tower".to_string());
    }
    let mut token_ids = Vec::with_capacity(prompt_tokens.len());
    let mut token_types = Vec::with_capacity(prompt_tokens.len());
    let mut image_inputs = Vec::with_capacity(images.len());
    let mut image_index = 0;
    for token in prompt_tokens {
        if token != image_token_id {
            token_ids.push(token);
            token_types.push(TEXT);
            continue;
        }
        let (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w) =
            load_image(&images[image_index], config)?;
        let types = image_token_types(n_llm_h, n_llm_w);
        image_inputs.push(ImageInput {
            start: token_ids.len(),
            patches,
            n_vit_h,
            n_vit_w,
            n_llm_h,
            n_llm_w,
            types: types.clone(),
        });
        token_ids.extend(std::iter::repeat_n(image_token_id, types.len()));
        token_types.extend(types);
        image_index += 1;
    }
    Ok(PreparedVlInputs {
        token_ids,
        token_types,
        images: image_inputs,
    })
}

pub struct VisionLinear {
    pub weight: QTensor,
    pub bias: Option<Vec<f32>>,
}

impl fmt::Debug for VisionLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VisionLinear")
            .field("rows", &self.weight.rows())
            .field("cols", &self.weight.cols())
            .field("has_bias", &self.bias.is_some())
            .finish()
    }
}

impl VisionLinear {
    fn new(weight: QTensor, bias: Option<Vec<f32>>) -> Result<Self, String> {
        if let Some(values) = &bias {
            if values.len() != weight.rows() {
                return Err(format!(
                    "linear bias length {} != output rows {}",
                    values.len(),
                    weight.rows()
                ));
            }
        }
        Ok(Self { weight, bias })
    }

    fn apply_many(&self, input: &[f32], batch: usize, output: &mut [f32], pool: Option<&Pool>) {
        assert_eq!(input.len(), batch * self.weight.cols());
        assert!(output.len() >= batch * self.weight.rows());

        // F16 vision matrices are owned as f32 by QTensor after the one-time
        // CMF decode.  Reuse the existing dense f32 NT GEMM on an explicitly
        // enabled discrete backend for the large image batch; small calls
        // and every backend refusal retain the portable QTensor path.
        let rows = self.weight.rows();
        let cols = self.weight.cols();
        if batch >= 8
            && batch.saturating_mul(rows).saturating_mul(cols) >= (1 << 22)
            && crate::gpu::enabled_here()
            && let Some(weight) = self.weight.as_f32()
            && crate::gpu::gemm_nt_f32(input, weight, output, batch, cols, rows)
        {
            GPU_DENSE_GEMM_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            if let Some(bias) = &self.bias {
                for row in output[..batch * rows].chunks_exact_mut(rows) {
                    for (value, &b) in row.iter_mut().zip(bias) {
                        *value += b;
                    }
                }
            }
            return;
        }
        self.weight.matmat(input, batch, output, pool);
        if let Some(bias) = &self.bias {
            for row in output[..batch * self.weight.rows()].chunks_exact_mut(self.weight.rows()) {
                for (value, &b) in row.iter_mut().zip(bias) {
                    *value += b;
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct VisionAttention {
    pub wqkv: VisionLinear,
    pub wo: VisionLinear,
}

#[derive(Debug)]
pub struct VisionMlp {
    pub w1: VisionLinear,
    pub w2: VisionLinear,
}

#[derive(Debug)]
pub struct VisionBlock {
    pub norm1: Vec<f32>,
    pub attn: VisionAttention,
    pub norm2: Vec<f32>,
    pub mlp: VisionMlp,
}

/// Loaded V4.1 vision tower plus projector/marker embeddings.
#[derive(Debug)]
pub struct VisionModel {
    pub config: VisionConfig,
    pub patch_embed: VisionLinear,
    pub blocks: Vec<VisionBlock>,
    pub norm: Vec<f32>,
    pub aligner_w1: VisionLinear,
    pub aligner_w2: VisionLinear,
    pub image_start: Vec<f32>,
    pub image_end: Vec<f32>,
    pub image_newline: Vec<f32>,
}

fn load_tensor(model: &Arc<CmfModel>, name: &str) -> Result<QTensor, String> {
    QTensor::from_model(model, name)
}

fn load_vector(model: &Arc<CmfModel>, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?;
    if entry.shape.iter().product::<usize>() != expected {
        return Err(format!(
            "tensor '{name}' has shape {:?}, expected {} elements",
            entry.shape, expected
        ));
    }
    let mut data = vec![0.0f32; expected];
    cortiq_core::quant::dequant_tensor(entry, model.entry_bytes(entry), &mut data)?;
    Ok(data)
}

fn load_optional_vector(
    model: &Arc<CmfModel>,
    name: &str,
    expected: usize,
) -> Result<Option<Vec<f32>>, String> {
    model
        .tensor(name)
        .map(|_| load_vector(model, name, expected))
        .transpose()
}

fn required_norm(model: &Arc<CmfModel>, name: &str, dim: usize) -> Result<Vec<f32>, String> {
    load_vector(model, name, dim)
}

/// The pinned aligner uses `F.pad(x, (0, -n_w % r, 0, -n_h % r))`.
/// Python's modulo is non-negative, so the apparently negative arguments add
/// zero padding up to the next multiple rather than cropping a partial cell.
fn padded_vit_grid(n_vit_h: usize, n_vit_w: usize, ratio: usize) -> (usize, usize) {
    (
        n_vit_h.div_ceil(ratio) * ratio,
        n_vit_w.div_ceil(ratio) * ratio,
    )
}

/// Materialize the channel-major windows consumed by the reference
/// `F.unfold`.  The source first pads the `[dim, n_vit_h, n_vit_w]` feature
/// map on the bottom/right, then unfolds non-overlapping `ratio×ratio`
/// windows.  Keeping this operation separate makes its boundary behavior
/// directly testable without loading the 1 GB vision component.
fn unfold_padded(
    x: &[f32],
    n_vit_h: usize,
    n_vit_w: usize,
    vision_dim: usize,
    ratio: usize,
) -> (Vec<f32>, usize, usize) {
    let (h, w) = padded_vit_grid(n_vit_h, n_vit_w, ratio);
    let rows = (h / ratio) * (w / ratio);
    let in_dim = vision_dim * ratio * ratio;
    debug_assert_eq!(x.len(), n_vit_h * n_vit_w * vision_dim);
    let mut input = vec![0.0f32; rows * in_dim];
    for block_y in 0..h / ratio {
        for block_x in 0..w / ratio {
            let row = block_y * (w / ratio) + block_x;
            let mut at = 0;
            // F.unfold after CHW permutation is channel-major, then the
            // ratio×ratio values in row-major order.
            for channel in 0..vision_dim {
                for dy in 0..ratio {
                    for dx in 0..ratio {
                        let patch_y = block_y * ratio + dy;
                        let patch_x = block_x * ratio + dx;
                        input[row * in_dim + at] = if patch_y < n_vit_h && patch_x < n_vit_w {
                            let patch = patch_y * n_vit_w + patch_x;
                            x[patch * vision_dim + channel]
                        } else {
                            0.0
                        };
                        at += 1;
                    }
                }
            }
        }
    }
    (input, h / ratio, w / ratio)
}

impl VisionModel {
    /// Load canonical V4.1 names from a CMF model.  The marker vectors are
    /// required whenever the tower is enabled because they replace the
    /// image-start/end/newline token embeddings in the text hidden state.
    pub fn from_model(model: &Arc<CmfModel>, config: VisionConfig) -> Result<Self, String> {
        config.validate()?;
        if !config.vision_enabled() {
            return Err("cannot load a disabled vision tower".to_string());
        }
        let patch_in = 3 * config.vision_patch_size * config.vision_patch_size;
        let patch_embed = VisionLinear::new(
            load_tensor(model, "vision.patch_embed.proj.weight")?,
            load_optional_vector(model, "vision.patch_embed.proj.bias", config.vision_dim)?,
        )?;
        if patch_embed.weight.rows() != config.vision_dim || patch_embed.weight.cols() != patch_in {
            return Err(format!(
                "patch embedding shape {}x{}, expected {}x{}",
                patch_embed.weight.rows(),
                patch_embed.weight.cols(),
                config.vision_dim,
                patch_in
            ));
        }
        let mut blocks = Vec::with_capacity(config.vision_n_layers);
        for layer in 0..config.vision_n_layers {
            let prefix = format!("vision.blocks.{layer}");
            let norm1 = required_norm(model, &format!("{prefix}.norm1.weight"), config.vision_dim)?;
            let wqkv = VisionLinear::new(
                load_tensor(model, &format!("{prefix}.attn.wqkv.weight"))?,
                load_optional_vector(
                    model,
                    &format!("{prefix}.attn.wqkv.bias"),
                    3 * config.vision_dim,
                )?,
            )?;
            let wo = VisionLinear::new(
                load_tensor(model, &format!("{prefix}.attn.wo.weight"))?,
                load_optional_vector(model, &format!("{prefix}.attn.wo.bias"), config.vision_dim)?,
            )?;
            let norm2 = required_norm(model, &format!("{prefix}.norm2.weight"), config.vision_dim)?;
            let w1 = VisionLinear::new(
                load_tensor(model, &format!("{prefix}.mlp.w1.weight"))?,
                load_optional_vector(
                    model,
                    &format!("{prefix}.mlp.w1.bias"),
                    2 * config.vision_inter_dim,
                )?,
            )?;
            let w2 = VisionLinear::new(
                load_tensor(model, &format!("{prefix}.mlp.w2.weight"))?,
                load_optional_vector(model, &format!("{prefix}.mlp.w2.bias"), config.vision_dim)?,
            )?;
            if wqkv.weight.rows() != 3 * config.vision_dim
                || wqkv.weight.cols() != config.vision_dim
                || wo.weight.rows() != config.vision_dim
                || wo.weight.cols() != config.vision_dim
                || w1.weight.rows() != 2 * config.vision_inter_dim
                || w1.weight.cols() != config.vision_dim
                || w2.weight.rows() != config.vision_dim
                || w2.weight.cols() != config.vision_inter_dim
            {
                return Err(format!(
                    "vision block {layer} has a non-reference linear shape"
                ));
            }
            blocks.push(VisionBlock {
                norm1,
                attn: VisionAttention { wqkv, wo },
                norm2,
                mlp: VisionMlp { w1, w2 },
            });
        }
        let norm = required_norm(model, "vision.norm.weight", config.vision_dim)?;
        let aligner_in = config.vision_dim * config.vision_downsample_ratio.pow(2);
        let aligner_w1 = VisionLinear::new(
            load_tensor(model, "aligner.w1.weight")?,
            load_optional_vector(model, "aligner.w1.bias", config.text_dim)?,
        )?;
        let aligner_w2 = VisionLinear::new(
            load_tensor(model, "aligner.w2.weight")?,
            load_optional_vector(model, "aligner.w2.bias", config.text_dim)?,
        )?;
        if aligner_w1.weight.rows() != config.text_dim
            || aligner_w1.weight.cols() != aligner_in
            || aligner_w2.weight.rows() != config.text_dim
            || aligner_w2.weight.cols() != config.text_dim
        {
            return Err("aligner linear shapes do not match vision config".to_string());
        }
        let image_start = load_vector(model, "image_start", config.text_dim)?;
        let image_end = load_vector(model, "image_end", config.text_dim)?;
        let image_newline = load_vector(model, "image_newline", config.text_dim)?;
        Ok(Self {
            config,
            patch_embed,
            blocks,
            norm,
            aligner_w1,
            aligner_w2,
            image_start,
            image_end,
            image_newline,
        })
    }

    /// Compute a vision image embedding with the official ViT and aligner.
    /// The returned rows correspond only to `IMAGE` positions, in reading
    /// order; marker vectors remain separate for the text runtime to insert.
    pub fn encode_image(
        &self,
        image: &ImageInput,
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>, String> {
        if image.n_vit_h == 0 || image.n_vit_w == 0 {
            return Err("image patch grid must be non-empty".to_string());
        }
        let patch_dim = 3 * self.config.vision_patch_size * self.config.vision_patch_size;
        let n = image.n_vit_h * image.n_vit_w;
        if image.patches.len() != n * patch_dim {
            return Err(format!(
                "patch payload has {} values, expected {}",
                image.patches.len(),
                n * patch_dim
            ));
        }
        let mut x = vec![0.0f32; n * self.config.vision_dim];
        self.patch_embed.apply_many(&image.patches, n, &mut x, pool);
        let (cos, sin) = get_vision_cos_sin(
            image.n_vit_h,
            image.n_vit_w,
            self.config.rope_dim(),
            self.config.vision_rope_theta,
        );
        for block in &self.blocks {
            let mut normed = vec![0.0f32; x.len()];
            for (src, dst) in x
                .chunks_exact(self.config.vision_dim)
                .zip(normed.chunks_exact_mut(self.config.vision_dim))
            {
                rms_norm(src, &block.norm1, 1e-6, dst);
            }
            let attention = attention_forward(
                &normed,
                &block.attn,
                &cos,
                &sin,
                self.config.vision_dim,
                self.config.vision_n_heads,
                pool,
            );
            for (dst, update) in x
                .chunks_exact_mut(self.config.vision_dim)
                .zip(attention.chunks_exact(self.config.vision_dim))
            {
                for (v, &u) in dst.iter_mut().zip(update) {
                    *v += u;
                }
            }
            let mut normed = vec![0.0f32; x.len()];
            for (src, dst) in x
                .chunks_exact(self.config.vision_dim)
                .zip(normed.chunks_exact_mut(self.config.vision_dim))
            {
                rms_norm(src, &block.norm2, 1e-6, dst);
            }
            let inter = block.mlp.w1.weight.rows() / 2;
            let mut hidden = vec![0.0f32; n * 2 * inter];
            block.mlp.w1.apply_many(&normed, n, &mut hidden, pool);
            let mut activated = vec![0.0f32; n * inter];
            for (source, output) in hidden
                .chunks_exact(2 * inter)
                .zip(activated.chunks_exact_mut(inter))
            {
                for i in 0..inter {
                    let gate = source[i];
                    output[i] = gate / (1.0 + (-gate).exp()) * source[inter + i];
                }
            }
            let mut mlp_out = vec![0.0f32; x.len()];
            block.mlp.w2.apply_many(&activated, n, &mut mlp_out, pool);
            for (dst, update) in x
                .chunks_exact_mut(self.config.vision_dim)
                .zip(mlp_out.chunks_exact(self.config.vision_dim))
            {
                for (v, &u) in dst.iter_mut().zip(update) {
                    *v += u;
                }
            }
        }
        for src in x.chunks_exact_mut(self.config.vision_dim).take(n) {
            let copy = src.to_vec();
            rms_norm(&copy, &self.norm, 1e-6, src);
        }
        self.align(image, &x, pool)
    }

    /// Insert image marker/projector rows into a text hidden-state span.
    /// `span` must have the exact `ImageInput::types` length and be laid out
    /// with the text hidden size in its last dimension.
    pub fn fill_image_span(
        &self,
        image: &ImageInput,
        span: &mut [f32],
        pool: Option<&Pool>,
    ) -> Result<(), String> {
        let dim = self.config.text_dim;
        if span.len() != image.types.len() * dim {
            return Err(format!(
                "image span has {} values, expected {}",
                span.len(),
                image.types.len() * dim
            ));
        }
        let embeds = self.encode_image(image, pool)?;
        let mut image_row = 0;
        for (kind, row) in image.types.iter().zip(span.chunks_exact_mut(dim)) {
            match *kind {
                IMAGE_START => row.copy_from_slice(&self.image_start),
                IMAGE_END => row.copy_from_slice(&self.image_end),
                IMAGE_NEW_LINE => row.copy_from_slice(&self.image_newline),
                IMAGE => {
                    let source = embeds
                        .get(image_row * dim..(image_row + 1) * dim)
                        .ok_or_else(|| "aligner/image token count mismatch".to_string())?;
                    row.copy_from_slice(source);
                    image_row += 1;
                }
                TEXT => return Err("TEXT type cannot occur inside an image span".to_string()),
                other => return Err(format!("unknown image token type {other}")),
            }
        }
        if image_row != embeds.len() / dim {
            return Err("aligner produced a different number of image rows".to_string());
        }
        Ok(())
    }

    /// Return one complete text-hidden image span (markers plus projected
    /// image rows) for callers that do not already own a preallocated hidden
    /// buffer.  The runtime's batched prefill can copy these rows directly
    /// over the corresponding placeholder positions.
    pub fn image_span(&self, image: &ImageInput, pool: Option<&Pool>) -> Result<Vec<f32>, String> {
        let mut span = vec![0.0f32; image.types.len() * self.config.text_dim];
        self.fill_image_span(image, &mut span, pool)?;
        Ok(span)
    }

    fn align(
        &self,
        image: &ImageInput,
        x: &[f32],
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>, String> {
        let r = self.config.vision_downsample_ratio;
        let (input, out_h, out_w) =
            unfold_padded(x, image.n_vit_h, image.n_vit_w, self.config.vision_dim, r);
        let rows = out_h * out_w;
        let mut out = vec![0.0f32; rows * self.config.text_dim];
        let mut hidden = vec![0.0f32; rows * self.config.text_dim];
        self.aligner_w1.apply_many(&input, rows, &mut hidden, pool);
        for value in &mut hidden {
            *value = gelu_exact(*value);
        }
        self.aligner_w2.apply_many(&hidden, rows, &mut out, pool);
        if out.len() != image.image_positions() * self.config.text_dim {
            return Err(format!(
                "aligner produced {} rows but span requests {} image positions",
                rows,
                image.image_positions()
            ));
        }
        Ok(out)
    }
}

/// Official 2-D RoPE grid.  Returned vectors are `[tokens, rope_dim]`, where
/// `rope_dim` is half a head and is broadcast over each q/k half.
pub fn get_vision_cos_sin(n_h: usize, n_w: usize, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut inv_freq = Vec::with_capacity(dim / 2);
    for i in (0..dim).step_by(2) {
        inv_freq.push(1.0 / theta.powf(i as f32 / dim.max(1) as f32));
    }
    let mut cos = Vec::with_capacity(n_h * n_w * dim);
    let mut sin = Vec::with_capacity(n_h * n_w * dim);
    for h in 0..n_h {
        for w in 0..n_w {
            for &position in &[h as f32, w as f32] {
                for &frequency in &inv_freq {
                    let angle = position * frequency;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
    }
    debug_assert_eq!(cos.len(), n_h * n_w * dim);
    debug_assert_eq!(sin.len(), cos.len());
    (cos, sin)
}

pub fn apply_rotary(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    assert_eq!(x.len(), cos.len() * 2);
    let half = x.len() / 2;
    let left = x[..half].to_vec();
    let right = x[half..].to_vec();
    for i in 0..half {
        x[i] = left[i] * cos[i] - right[i] * sin[i];
        x[half + i] = right[i] * cos[i] + left[i] * sin[i];
    }
}

pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    assert_eq!(input.len(), weight.len());
    assert!(output.len() >= input.len());
    let mean = input.iter().map(|v| v * v).sum::<f32>() / input.len().max(1) as f32;
    let scale = (mean + eps).sqrt().recip();
    for ((dst, &value), &factor) in output.iter_mut().zip(input).zip(weight) {
        *dst = value * scale * factor;
    }
}

fn gelu_exact(value: f32) -> f32 {
    // Abramowitz-Stegun erf approximation (maximum error ~1.5e-7), matching
    // torch.nn.functional.gelu(..., approximate="none") closely in f32.
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / std::f32::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * value * (1.0 + erf)
}

fn attention_forward(
    input: &[f32],
    attention: &VisionAttention,
    cos: &[f32],
    sin: &[f32],
    dim: usize,
    heads: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let n = input.len() / dim;
    let head_dim = dim / heads;
    let rope_dim = head_dim / 2;
    // The reference applies qkv and output projections token by token.  The
    // mathematical result is the same when the existing QTensor GEMM path
    // handles the complete image batch, while the weight stream is read once
    // per projection (and Q4TP can use its GPU matmat kernel).
    let mut qkv = vec![0.0f32; n * 3 * dim];
    attention.wqkv.apply_many(input, n, &mut qkv, pool);
    let mut q = vec![0.0f32; n * dim];
    let mut k = vec![0.0f32; n * dim];
    let mut v = vec![0.0f32; n * dim];
    for (token, source) in qkv.chunks_exact(3 * dim).enumerate() {
        q[token * dim..(token + 1) * dim].copy_from_slice(&source[..dim]);
        k[token * dim..(token + 1) * dim].copy_from_slice(&source[dim..2 * dim]);
        v[token * dim..(token + 1) * dim].copy_from_slice(&source[2 * dim..]);
        let cos_row = &cos[token * rope_dim..(token + 1) * rope_dim];
        let sin_row = &sin[token * rope_dim..(token + 1) * rope_dim];
        for head in 0..heads {
            let offset = token * dim + head * head_dim;
            apply_rotary(&mut q[offset..offset + head_dim], cos_row, sin_row);
            apply_rotary(&mut k[offset..offset + head_dim], cos_row, sin_row);
        }
    }
    let scale = (head_dim as f32).sqrt().recip();

    // The CPU reference below is intentionally retained as the portable
    // fallback, but its O(heads * n² * head_dim) loop is not viable for the
    // 3072-token V4.1 image grid.  The existing backend attention kernel
    // consumes head-major panels and returns the original token-major
    // layout, so transpose only when the caller has explicitly enabled a
    // live backend.  A refusal falls through to the exact same CPU path.
    if n >= 128 && crate::gpu::enabled_here() {
        let panel_len = n * heads * head_dim;
        let mut qh = vec![0.0f32; panel_len];
        let mut kh = vec![0.0f32; panel_len];
        let mut vh = vec![0.0f32; panel_len];
        for token in 0..n {
            for head in 0..heads {
                let src = token * dim + head * head_dim;
                let dst = head * n * head_dim + token * head_dim;
                qh[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                kh[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                vh[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
        }
        let mut context = vec![0.0f32; panel_len];
        if crate::gpu::dit_attention(
            &qh,
            &kh,
            &vh,
            heads,
            heads,
            n,
            head_dim,
            scale,
            &mut context,
        ) {
            GPU_ATTENTION_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            let mut output = vec![0.0f32; n * dim];
            attention.wo.apply_many(&context, n, &mut output, pool);
            return output;
        }
    }

    let mut context = vec![0.0f32; n * dim];
    let mut scores = vec![0.0f32; n];
    for head in 0..heads {
        for query in 0..n {
            let qrow = &q[query * dim + head * head_dim..query * dim + (head + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for key in 0..n {
                let krow = &k[key * dim + head * head_dim..key * dim + (head + 1) * head_dim];
                let score = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores[key] = score;
                max_score = max_score.max(score);
            }
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            let inv = denominator.recip();
            let out =
                &mut context[query * dim + head * head_dim..query * dim + (head + 1) * head_dim];
            for key in 0..n {
                let probability = scores[key] * inv;
                let vrow = &v[key * dim + head * head_dim..key * dim + (head + 1) * head_dim];
                for (dst, &value) in out.iter_mut().zip(vrow) {
                    *dst += probability * value;
                }
            }
        }
    }
    let mut output = vec![0.0f32; n * dim];
    attention.wo.apply_many(&context, n, &mut output, pool);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_grid_and_types_match_reference() {
        let config = VisionConfig {
            vision_n_layers: 1,
            ..VisionConfig::default()
        };
        let (h, w, best_h, best_w) = plan_image_grid(544, 544, &config).unwrap();
        assert_eq!((h, w), (13, 13));
        assert_eq!((best_h, best_w), (546, 546));
        assert_eq!(num_image_tokens(h, w), 184);
        let types = image_token_types(h, w);
        assert_eq!(types.first(), Some(&IMAGE_START));
        assert_eq!(types.last(), Some(&IMAGE_END));
        assert_eq!(types.iter().filter(|&&v| v == IMAGE_NEW_LINE).count(), h);
        assert_eq!(types.iter().filter(|&&v| v == IMAGE).count(), h * w);
    }

    #[test]
    fn rope_grid_has_reference_first_rows() {
        let (cos, sin) = get_vision_cos_sin(2, 2, 4, 10_000.0);
        assert_eq!(&cos[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&sin[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert!((cos[6] - 0.5403023).abs() < 1e-6);
        assert!((sin[6] - 0.8414710).abs() < 1e-6);
    }

    #[test]
    fn exact_gelu_and_rms_are_finite() {
        assert!((gelu_exact(1.0) - 0.8413447).abs() < 2e-6);
        let mut out = [0.0; 2];
        rms_norm(&[3.0, 4.0], &[1.0, 2.0], 1e-6, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let scale = (12.5_f32 + 1e-6).sqrt().recip();
        assert!((out[0] - 3.0 * scale).abs() < 1e-4);
        assert!((out[1] - 8.0 * scale).abs() < 1e-4);
    }

    #[test]
    fn aligner_zero_pads_nonmultiple_vit_grid() {
        // Python's `-n % r` is positive for a nonmultiple n.  Thus a 35x46
        // ViT grid becomes 36x48 before F.unfold and yields the same 12x16
        // image rows advertised by the processor, rather than dropping the
        // final partial cells.
        assert_eq!(padded_vit_grid(35, 46, 3), (36, 48));
        assert_eq!(llm_grid(36 * 14, 48 * 14, 14, 3), (12, 16));
        assert_eq!(
            12 * 16,
            image_token_types(12, 16)
                .iter()
                .filter(|&&v| v == IMAGE)
                .count()
        );

        // Also pin the actual channel-major unfold at both right and bottom
        // boundaries.  The second window contains the final real column and
        // two zero columns; its bottom two rows are zero as well.
        let x: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let (windows, out_h, out_w) = unfold_padded(&x, 2, 4, 1, 3);
        assert_eq!((out_h, out_w), (1, 2));
        assert_eq!(
            &windows[..9],
            &[0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            &windows[9..],
            &[3.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
