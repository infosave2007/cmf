//! Qwen2.5-VL vision tower and the exact image processor path used by
//! Qwen-Image-Edit-2509.
//!
//! The image processor is intentionally kept beside the tower.  The first
//! resize is the Qwen Image pipeline's 384²-area `VaeImageProcessor.resize`
//! (Lanczos); the second is Qwen2VL's smart resize (BICUBIC, patch 14,
//! temporal patch 2, merge 2).  Keeping both stages here prevents a caller
//! from accidentally feeding the VAE-sized reference image to the prompt
//! encoder.

use crate::pool::Pool;
use crate::qwen_image_ops::Linear;
use cortiq_core::CmfModel;
use cortiq_core::TensorDtype;
use image::{imageops::FilterType, RgbImage};
use serde_json::Value;
use std::sync::Arc;

const EPS: f64 = 1e-6;
const CONDITION_AREA: f64 = 384.0 * 384.0;
const CONDITION_ROUND: f64 = 32.0;

/// Processor values copied from the pinned `processor/preprocessor_config`
/// file.  The JSON is embedded in the encoder CMF and is required at load;
/// these defaults are only the official values used by tiny fixtures that
/// omit an optional field.
#[derive(Clone, Debug)]
pub(crate) struct ProcessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resample: u32,
}

impl ProcessorConfig {
    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let cfg: Value = serde_json::from_slice(bytes)
            .map_err(|e| format!("image.processor_config_json: {e}"))?;
        let size = cfg.get("size").unwrap_or(&cfg);
        let min_pixels = usize_field(&cfg, "min_pixels")
            .or_else(|| usize_field(size, "shortest_edge"))
            .unwrap_or(3136);
        let max_pixels = usize_field(&cfg, "max_pixels")
            .or_else(|| usize_field(size, "longest_edge"))
            .unwrap_or(12_845_056);
        let patch_size = usize_field(&cfg, "patch_size").unwrap_or(14);
        let temporal_patch_size = usize_field(&cfg, "temporal_patch_size").unwrap_or(2);
        let merge_size = usize_field(&cfg, "merge_size").unwrap_or(2);
        let mean = vec3_field(&cfg, "image_mean").unwrap_or([0.48145466, 0.4578275, 0.40821073]);
        let std = vec3_field(&cfg, "image_std").unwrap_or([0.26862954, 0.26130258, 0.27577711]);
        let resample = cfg.get("resample").and_then(Value::as_u64).unwrap_or(3) as u32;

        if min_pixels == 0 || max_pixels < min_pixels {
            return Err(format!(
                "processor pixel bounds are invalid: min_pixels={min_pixels}, max_pixels={max_pixels}"
            ));
        }
        if patch_size == 0 || temporal_patch_size == 0 || merge_size == 0 {
            return Err("processor patch, temporal_patch, and merge sizes must be positive".into());
        }
        if mean.iter().any(|v| !v.is_finite()) || std.iter().any(|v| !v.is_finite() || *v == 0.0) {
            return Err("processor mean/std contains a non-finite or zero value".into());
        }
        Ok(Self {
            min_pixels,
            max_pixels,
            patch_size,
            temporal_patch_size,
            merge_size,
            mean,
            std,
            resample,
        })
    }
}

fn usize_field(v: &Value, name: &str) -> Option<usize> {
    v.get(name).and_then(Value::as_u64).map(|n| n as usize)
}

fn vec3_field(v: &Value, name: &str) -> Option<[f32; 3]> {
    let a = v.get(name)?.as_array()?;
    if a.len() != 3 {
        return None;
    }
    Some([
        a[0].as_f64()? as f32,
        a[1].as_f64()? as f32,
        a[2].as_f64()? as f32,
    ])
}

/// A processor output for one image.  `patches` is the row-major matrix fed
/// to `visual.patch_embed.proj`; the rows are in Qwen2VL's merged-grid order.
#[derive(Clone, Debug)]
pub(crate) struct PreparedImage {
    pub patches: Vec<f32>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
}

impl PreparedImage {
    pub(crate) fn image_tokens(&self, merge: usize) -> Result<usize, String> {
        self.grid_t
            .checked_mul(self.grid_h)
            .and_then(|n| n.checked_mul(self.grid_w))
            .and_then(|n| n.checked_div(merge.saturating_mul(merge)))
            .ok_or_else(|| "image grid token count overflow or invalid merge size".into())
    }
}

/// Python's `round` for the positive values used by the official dimensions.
/// Rust's `f64::round` breaks exact `.5` ties away from zero, while Python
/// (and the pinned Diffusers/Transformers code) uses ties-to-even.
fn round_positive(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    let rounded = if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else if (floor as u64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded.max(1.0) as usize
}

/// Diffusers' Qwen Image `calculate_dimensions(384², aspect)`.
pub(crate) fn condition_dimensions(width: u32, height: u32) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 {
        return Err("image dimensions must be non-zero".into());
    }
    let ratio = width as f64 / height as f64;
    let w = round_positive((CONDITION_AREA * ratio).sqrt() / CONDITION_ROUND) * 32;
    let h = round_positive((CONDITION_AREA / ratio).sqrt() / CONDITION_ROUND) * 32;
    Ok((w, h))
}

/// Exact Qwen2VL smart-resize dimensions.  The returned tuple is
/// `(height,width)`, matching the processor and Diffusers call sites.
pub(crate) fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err("smart_resize requires positive dimensions and factor".into());
    }
    let aspect = height.max(width) as f64 / height.min(width) as f64;
    if aspect > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {aspect}"
        ));
    }
    let mut h = round_positive(height as f64 / factor as f64) * factor;
    let mut w = round_positive(width as f64 / factor as f64) * factor;
    h = h.max(factor);
    w = w.max(factor);
    if h.saturating_mul(w) > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h.saturating_mul(w) < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h = ((height as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
        w = ((width as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
    }
    if h == 0 || w == 0 || h % factor != 0 || w % factor != 0 {
        return Err(format!(
            "smart_resize produced invalid grid {h}x{w} for factor {factor}"
        ));
    }
    Ok((h, w))
}

fn resize_filter(code: u32) -> FilterType {
    // PIL's enum value 3 is BICUBIC, the pinned Qwen processor default.  The
    // image crate's CatmullRom is its cubic kernel and is the closest native
    // equivalent; the first (Diffusers) stage is always Lanczos3 below.
    match code {
        0 => FilterType::Nearest,
        2 => FilterType::Triangle,
        4 => FilterType::Lanczos3,
        _ => FilterType::CatmullRom,
    }
}

/// Prepare one RGB image for the Qwen2.5-VL vision tower.
pub(crate) fn prepare_image(
    image: &RgbImage,
    cfg: &ProcessorConfig,
) -> Result<PreparedImage, String> {
    let (condition_w, condition_h) = condition_dimensions(image.width(), image.height())?;
    let first = image::imageops::resize(
        image,
        condition_w as u32,
        condition_h as u32,
        FilterType::Lanczos3,
    );
    let (height, width) = smart_resize(
        first.height() as usize,
        first.width() as usize,
        cfg.patch_size
            .checked_mul(cfg.merge_size)
            .ok_or_else(|| "processor patch × merge overflow".to_string())?,
        cfg.min_pixels,
        cfg.max_pixels,
    )?;
    let second = image::imageops::resize(
        &first,
        width as u32,
        height as u32,
        resize_filter(cfg.resample),
    );
    let grid_h = height / cfg.patch_size;
    let grid_w = width / cfg.patch_size;
    if grid_h % cfg.merge_size != 0 || grid_w % cfg.merge_size != 0 {
        return Err(format!(
            "processor grid {grid_h}x{grid_w} is not divisible by merge {}",
            cfg.merge_size
        ));
    }
    let grid_t: usize = 1;
    let patch_dim = 3usize
        .checked_mul(cfg.temporal_patch_size)
        .and_then(|n| n.checked_mul(cfg.patch_size))
        .and_then(|n| n.checked_mul(cfg.patch_size))
        .ok_or_else(|| "processor patch dimension overflow".to_string())?;
    let rows = grid_t
        .checked_mul(grid_h)
        .and_then(|n| n.checked_mul(grid_w))
        .ok_or_else(|| "processor patch row count overflow".to_string())?;
    let mut patches = Vec::with_capacity(rows * patch_dim);
    let m = cfg.merge_size;
    // The single input frame is repeated to temporal_patch_size exactly as
    // Qwen2VLImageProcessorFast does before its view/permute/reshape.
    for _gt in 0..grid_t {
        for gh in 0..grid_h / m {
            for gw in 0..grid_w / m {
                for mh in 0..m {
                    for mw in 0..m {
                        for c in 0..3 {
                            for _t in 0..cfg.temporal_patch_size {
                                for py in 0..cfg.patch_size {
                                    for px in 0..cfg.patch_size {
                                        let x = gw * m * cfg.patch_size + mw * cfg.patch_size + px;
                                        let y = gh * m * cfg.patch_size + mh * cfg.patch_size + py;
                                        let raw =
                                            second.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                                        patches.push((raw - cfg.mean[c]) / cfg.std[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if patches.len() != rows * patch_dim {
        return Err(format!(
            "processor produced {} patch values, expected {}",
            patches.len(),
            rows * patch_dim
        ));
    }
    Ok(PreparedImage {
        patches,
        grid_t,
        grid_h,
        grid_w,
        patch_dim,
    })
}

struct VisionBlock {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Linear,
    qkv_bias: Vec<f32>,
    proj: Linear,
    proj_bias: Vec<f32>,
    gate: Linear,
    gate_bias: Vec<f32>,
    up: Linear,
    up_bias: Vec<f32>,
    down: Linear,
    down_bias: Vec<f32>,
}

struct Merger {
    norm: Vec<f32>,
    fc1: Linear,
    fc1_bias: Vec<f32>,
    fc2: Linear,
    fc2_bias: Vec<f32>,
}

/// The source patch projection is a rank-5 Conv3d kernel.  `Linear` is
/// intentionally rank-2-only, so this small first layer keeps the semantic
/// `[out, channel, temporal, height, width]` shape while flattening each
/// output kernel for the exact patch rows emitted by the processor.  It is
/// only 1,505,280 values in the canonical tower; the large text/vision
/// projections remain mmap/quantized `Linear`s.
struct PatchProjection {
    weights: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// Native Qwen2.5-VL vision tower.  It returns one merged feature matrix per
/// input image, in the same order as the processor's `<|image_pad|>` tokens.
pub(crate) struct VisionTower {
    patch: PatchProjection,
    blocks: Vec<VisionBlock>,
    merger: Merger,
    pub(crate) hidden: usize,
    pub(crate) out_hidden: usize,
    pub(crate) heads: usize,
    pub(crate) patch_size: usize,
    pub(crate) temporal_patch_size: usize,
    pub(crate) merge_size: usize,
    pub(crate) window_size: usize,
    fullatt: Vec<usize>,
    pool: Option<Arc<Pool>>,
}

fn cfg_usize(cfg: &Value, key: &str) -> Result<usize, String> {
    cfg.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("vision config missing integer '{key}'"))
}

fn cfg_usize_default(cfg: &Value, key: &str, default: usize) -> usize {
    cfg.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn expect_linear(
    model: &Arc<CmfModel>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Linear, String> {
    let p = Linear::load(model, name)?;
    if p.rows() != rows || p.cols() != cols {
        return Err(format!(
            "tensor '{name}' shape [{},{}] != expected [{rows},{cols}]",
            p.rows(),
            p.cols()
        ));
    }
    Ok(p)
}

fn patch_projection(
    model: &CmfModel,
    cfg: &Value,
    hidden: usize,
) -> Result<PatchProjection, String> {
    let entry = model
        .tensor("visual.patch_embed.proj.weight")
        .ok_or_else(|| "missing tensor 'visual.patch_embed.proj.weight'".to_string())?;
    if entry.shape.len() != 5 {
        return Err(format!(
            "tensor 'visual.patch_embed.proj.weight' must be rank 5, got shape {:?}",
            entry.shape
        ));
    }
    let in_channels = cfg_usize_default(cfg, "in_channels", 3);
    let temporal = cfg_usize(cfg, "temporal_patch_size")?;
    let patch = cfg_usize(cfg, "patch_size")?;
    let expected = [hidden, in_channels, temporal, patch, patch];
    if entry.shape.as_slice() != expected.as_slice() {
        return Err(format!(
            "tensor 'visual.patch_embed.proj.weight' shape {:?} != expected {:?}",
            entry.shape, expected
        ));
    }
    let cols = in_channels
        .checked_mul(temporal)
        .and_then(|n| n.checked_mul(patch))
        .and_then(|n| n.checked_mul(patch))
        .ok_or_else(|| "vision patch projection width overflow".to_string())?;
    let weights = crate::dit::cmf_f32(model, "visual.patch_embed.proj.weight")?;
    if weights.len() != hidden * cols {
        return Err(format!(
            "patch projection has {} values, expected {}",
            weights.len(),
            hidden * cols
        ));
    }
    Ok(PatchProjection {
        weights,
        rows: hidden,
        cols,
    })
}

impl PatchProjection {
    fn forward(&self, x: &[f32], batch: usize, out: &mut [f32]) -> Result<(), String> {
        if x.len() != batch.saturating_mul(self.cols) {
            return Err(format!(
                "patch input length {} != batch {batch} × cols {}",
                x.len(),
                self.cols
            ));
        }
        if out.len() != batch.saturating_mul(self.rows) {
            return Err(format!(
                "patch output length {} != batch {batch} × rows {}",
                out.len(),
                self.rows
            ));
        }
        for b in 0..batch {
            let input = &x[b * self.cols..(b + 1) * self.cols];
            let output = &mut out[b * self.rows..(b + 1) * self.rows];
            for (r, dst) in output.iter_mut().enumerate() {
                *dst = crate::attention::dot_f32(
                    input,
                    &self.weights[r * self.cols..(r + 1) * self.cols],
                );
            }
        }
        Ok(())
    }
}

fn vector(model: &CmfModel, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!(
            "tensor '{name}' has {} values, expected {len}",
            v.len()
        ));
    }
    Ok(v)
}

impl VisionTower {
    pub(crate) fn from_cmf(model: &Arc<CmfModel>, cfg: &Value) -> Result<Self, String> {
        let depth = cfg_usize(cfg, "depth")?;
        let hidden = cfg_usize(cfg, "hidden_size")?;
        let intermediate = cfg_usize(cfg, "intermediate_size")?;
        let heads = cfg_usize(cfg, "num_heads")?;
        let in_channels = cfg_usize_default(cfg, "in_channels", 3);
        let patch_size = cfg_usize(cfg, "patch_size")?;
        let temporal_patch_size = cfg_usize(cfg, "temporal_patch_size")?;
        let merge_size = cfg_usize(cfg, "spatial_merge_size")?;
        let window_size = cfg_usize(cfg, "window_size")?;
        let out_hidden = cfg_usize(cfg, "out_hidden_size")?;
        if hidden == 0 || heads == 0 || hidden % heads != 0 {
            return Err(format!(
                "vision hidden size {hidden} is not divisible by heads {heads}"
            ));
        }
        if patch_size == 0 || temporal_patch_size == 0 || merge_size == 0 || window_size == 0 {
            return Err("vision patch, temporal, merge, and window sizes must be positive".into());
        }
        if hidden % heads != 0 || (hidden / heads) % 4 != 0 {
            return Err(
                "vision head dimension must be divisible by four for Qwen rotary embedding".into(),
            );
        }
        if cfg
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu")
            != "silu"
        {
            return Err("Qwen2.5-VL vision MLP requires hidden_act='silu'".into());
        }
        let fullatt = match cfg.get("fullatt_block_indexes") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_u64().map(|x| x as usize).ok_or_else(|| {
                        "vision fullatt_block_indexes contains a non-integer".to_string()
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        if fullatt.iter().any(|&i| i >= depth) {
            return Err(format!(
                "vision full-attention block index exceeds depth {depth}"
            ));
        }
        let patch = patch_projection(model, cfg, hidden)?;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let p = format!("visual.blocks.{i}");
            blocks.push(VisionBlock {
                norm1: vector(model, &format!("{p}.norm1.weight"), hidden)?,
                norm2: vector(model, &format!("{p}.norm2.weight"), hidden)?,
                qkv: expect_linear(model, &format!("{p}.attn.qkv.weight"), hidden * 3, hidden)?,
                qkv_bias: vector(model, &format!("{p}.attn.qkv.bias"), hidden * 3)?,
                proj: expect_linear(model, &format!("{p}.attn.proj.weight"), hidden, hidden)?,
                proj_bias: vector(model, &format!("{p}.attn.proj.bias"), hidden)?,
                gate: expect_linear(
                    model,
                    &format!("{p}.mlp.gate_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                gate_bias: vector(model, &format!("{p}.mlp.gate_proj.bias"), intermediate)?,
                up: expect_linear(
                    model,
                    &format!("{p}.mlp.up_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                up_bias: vector(model, &format!("{p}.mlp.up_proj.bias"), intermediate)?,
                down: expect_linear(
                    model,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    intermediate,
                )?,
                down_bias: vector(model, &format!("{p}.mlp.down_proj.bias"), hidden)?,
            });
        }
        let merger_in = hidden
            .checked_mul(merge_size)
            .and_then(|n| n.checked_mul(merge_size))
            .ok_or_else(|| "vision merger width overflow".to_string())?;
        let merger = Merger {
            norm: vector(model, "visual.merger.ln_q.weight", hidden)?,
            fc1: expect_linear(model, "visual.merger.mlp.0.weight", merger_in, merger_in)?,
            fc1_bias: vector(model, "visual.merger.mlp.0.bias", merger_in)?,
            fc2: expect_linear(model, "visual.merger.mlp.2.weight", out_hidden, merger_in)?,
            fc2_bias: vector(model, "visual.merger.mlp.2.bias", out_hidden)?,
        };
        Ok(Self {
            patch,
            blocks,
            merger,
            hidden,
            out_hidden,
            heads,
            patch_size,
            temporal_patch_size,
            merge_size,
            window_size,
            fullatt,
            pool: Pool::from_env(),
        })
    }

    fn rms_norm(x: &[f32], w: &[f32], dst: &mut [f32]) {
        let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ss + EPS).sqrt();
        for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
            *d = (v as f64 * inv) as f32 * g;
        }
    }

    fn gelu_exact(v: f32) -> f32 {
        0.5 * v * (1.0 + erf(v as f64 / std::f64::consts::SQRT_2) as f32)
    }

    fn vision_rope(&self, raw_id: usize, grid_w: usize, cos: &mut [f32], sin: &mut [f32]) {
        let m = self.merge_size;
        let inside = raw_id % (m * m);
        let group = raw_id / (m * m);
        let mh = inside / m;
        let mw = inside % m;
        let gh = group / (grid_w / m);
        let gw = group % (grid_w / m);
        let h = gh * m + mh;
        let w = gw * m + mw;
        let hd = self.hidden / self.heads;
        // Qwen2.5-VL first builds rotary frequencies for each spatial axis,
        // flattens `[h-frequencies, w-frequencies]`, then concatenates that
        // vector with itself.  The resulting `[h,w,h,w]` layout is paired by
        // rotate-half; assigning `[h,h,w,w]` here silently changes the model.
        let freq_count = hd / 4;
        for d in 0..hd {
            let (axis, j) = if d < 2 * freq_count {
                if d < freq_count {
                    (h, d)
                } else {
                    (w, d - freq_count)
                }
            } else {
                let local = d - 2 * freq_count;
                if local < freq_count {
                    (h, local)
                } else {
                    (w, local - freq_count)
                }
            };
            let freq = 1.0 / 10_000f64.powf(2.0 * j as f64 / (hd / 2) as f64);
            let (s, c) = (axis as f64 * freq).sin_cos();
            cos[d] = c as f32;
            sin[d] = s as f32;
        }
    }

    fn build_window_order(
        &self,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(Vec<usize>, Vec<(usize, usize)>), String> {
        let m = self.merge_size;
        if grid_t == 0 || grid_h == 0 || grid_w == 0 || grid_h % m != 0 || grid_w % m != 0 {
            return Err(format!(
                "vision grid {grid_t}×{grid_h}×{grid_w} is empty or not merge-aligned ({m})"
            ));
        }
        let mh = grid_h / m;
        let mw = grid_w / m;
        let window = self.window_size / m / self.patch_size;
        if window == 0 {
            return Err(format!(
                "vision window {} is smaller than merge×patch {}",
                self.window_size,
                m * self.patch_size
            ));
        }
        let nw_h = (mh + window - 1) / window;
        let nw_w = (mw + window - 1) / window;
        let mut order = Vec::with_capacity(grid_t * mh * mw * m * m);
        let mut ranges = Vec::new();
        for t in 0..grid_t {
            for wh in 0..nw_h {
                for ww in 0..nw_w {
                    let start = order.len();
                    for ih in 0..window {
                        for iw in 0..window {
                            let h = wh * window + ih;
                            let w = ww * window + iw;
                            if h < mh && w < mw {
                                let group = t * mh * mw + h * mw + w;
                                for k in 0..m * m {
                                    order.push(group * m * m + k);
                                }
                            }
                        }
                    }
                    if order.len() > start {
                        ranges.push((start, order.len()));
                    }
                }
            }
        }
        Ok((order, ranges))
    }

    fn forward_one(&self, prepared: &PreparedImage) -> Result<Vec<f32>, String> {
        if prepared.patch_dim != self.patch_size * self.patch_size * 3 * self.temporal_patch_size {
            return Err(format!(
                "prepared patch width {} != vision expected {}",
                prepared.patch_dim,
                self.patch_size * self.patch_size * 3 * self.temporal_patch_size
            ));
        }
        let raw = prepared.grid_t * prepared.grid_h * prepared.grid_w;
        if prepared.patches.len() != raw * prepared.patch_dim {
            return Err("prepared patch matrix has an inconsistent shape".into());
        }
        let (order, window_ranges) =
            self.build_window_order(prepared.grid_t, prepared.grid_h, prepared.grid_w)?;
        if order.len() != raw {
            return Err(format!(
                "vision window permutation has {} rows, expected {raw}",
                order.len()
            ));
        }
        let pool = self.pool.as_deref();
        let mut original = vec![0f32; raw * self.hidden];
        self.patch.forward(&prepared.patches, raw, &mut original)?;
        let mut x = vec![0f32; raw * self.hidden];
        for (dst, &src) in order.iter().enumerate() {
            x[dst * self.hidden..(dst + 1) * self.hidden]
                .copy_from_slice(&original[src * self.hidden..(src + 1) * self.hidden]);
        }
        let hd = self.hidden / self.heads;
        let mut rope_cos = vec![0f32; raw * hd];
        let mut rope_sin = vec![0f32; raw * hd];
        for (dst, &src) in order.iter().enumerate() {
            self.vision_rope(
                src,
                prepared.grid_w,
                &mut rope_cos[dst * hd..(dst + 1) * hd],
                &mut rope_sin[dst * hd..(dst + 1) * hd],
            );
        }
        let mut norm = vec![0f32; raw * self.hidden];
        let mut qkv = vec![0f32; raw * self.hidden * 3];
        let mut q = vec![0f32; raw * self.hidden];
        let mut k = vec![0f32; raw * self.hidden];
        let mut v = vec![0f32; raw * self.hidden];
        let mut attn = vec![0f32; raw * self.hidden];
        let mut proj = vec![0f32; raw * self.hidden];
        let mut mlp_gate =
            vec![0f32; raw * self.blocks.first().map_or(self.hidden, |b| b.gate.rows())];
        let mut mlp_up = mlp_gate.clone();
        let mut mlp_down = vec![0f32; raw * self.hidden];
        for (block_idx, block) in self.blocks.iter().enumerate() {
            for i in 0..raw {
                Self::rms_norm(
                    &x[i * self.hidden..(i + 1) * self.hidden],
                    &block.norm1,
                    &mut norm[i * self.hidden..(i + 1) * self.hidden],
                );
            }
            block.qkv.forward(&norm, raw, &mut qkv, pool)?;
            crate::qwen_image_ops::add_bias(&mut qkv, raw, &block.qkv_bias)?;
            for i in 0..raw {
                q[i * self.hidden..(i + 1) * self.hidden]
                    .copy_from_slice(&qkv[i * self.hidden * 3..i * self.hidden * 3 + self.hidden]);
                k[i * self.hidden..(i + 1) * self.hidden].copy_from_slice(
                    &qkv[i * self.hidden * 3 + self.hidden..i * self.hidden * 3 + self.hidden * 2],
                );
                v[i * self.hidden..(i + 1) * self.hidden].copy_from_slice(
                    &qkv[i * self.hidden * 3 + self.hidden * 2..(i + 1) * self.hidden * 3],
                );
                for h in 0..self.heads {
                    let qs = &mut q[i * self.hidden + h * hd..i * self.hidden + (h + 1) * hd];
                    let ks = &mut k[i * self.hidden + h * hd..i * self.hidden + (h + 1) * hd];
                    let cs = &rope_cos[i * hd..(i + 1) * hd];
                    let ss = &rope_sin[i * hd..(i + 1) * hd];
                    let half = hd / 2;
                    for d in 0..half {
                        let (qa, qb) = (qs[d], qs[d + half]);
                        qs[d] = qa * cs[d] - qb * ss[d];
                        qs[d + half] = qa * ss[d] + qb * cs[d];
                        let (ka, kb) = (ks[d], ks[d + half]);
                        ks[d] = ka * cs[d] - kb * ss[d];
                        ks[d + half] = ka * ss[d] + kb * cs[d];
                    }
                }
            }
            attn.fill(0.0);
            let full = self.fullatt.contains(&block_idx);
            let full_range = (0usize, raw);
            let segments: &[(usize, usize)] = if full {
                std::slice::from_ref(&full_range)
            } else {
                &window_ranges
            };
            for &(lo, hi) in segments {
                for head in 0..self.heads {
                    let mut scores = vec![0f32; hi - lo];
                    for i in lo..hi {
                        let qi = &q[i * self.hidden + head * hd..i * self.hidden + (head + 1) * hd];
                        let mut mx = f32::NEG_INFINITY;
                        for (offset, j) in (lo..hi).enumerate() {
                            let kj =
                                &k[j * self.hidden + head * hd..j * self.hidden + (head + 1) * hd];
                            let s = crate::attention::dot_f32(qi, kj) / (hd as f32).sqrt();
                            scores[offset] = s;
                            mx = mx.max(s);
                        }
                        let mut den = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - mx).exp();
                            den += *s;
                        }
                        if den == 0.0 {
                            return Err("vision attention softmax underflowed".into());
                        }
                        let inv = 1.0 / den;
                        let out = &mut attn
                            [i * self.hidden + head * hd..i * self.hidden + (head + 1) * hd];
                        for (offset, j) in (lo..hi).enumerate() {
                            let vv =
                                &v[j * self.hidden + head * hd..j * self.hidden + (head + 1) * hd];
                            let weight = scores[offset] * inv;
                            for (o, &vv) in out.iter_mut().zip(vv) {
                                *o += weight * vv;
                            }
                        }
                    }
                }
            }
            block.proj.forward(&attn, raw, &mut proj, pool)?;
            crate::qwen_image_ops::add_bias(&mut proj, raw, &block.proj_bias)?;
            for (a, &b) in x.iter_mut().zip(&proj) {
                *a += b;
            }
            for i in 0..raw {
                Self::rms_norm(
                    &x[i * self.hidden..(i + 1) * self.hidden],
                    &block.norm2,
                    &mut norm[i * self.hidden..(i + 1) * self.hidden],
                );
            }
            block.gate.forward(&norm, raw, &mut mlp_gate, pool)?;
            block.up.forward(&norm, raw, &mut mlp_up, pool)?;
            crate::qwen_image_ops::add_bias(&mut mlp_gate, raw, &block.gate_bias)?;
            crate::qwen_image_ops::add_bias(&mut mlp_up, raw, &block.up_bias)?;
            for (g, &u) in mlp_gate.iter_mut().zip(&mlp_up) {
                *g = (*g / (1.0 + (-*g).exp())) * u;
            }
            block.down.forward(&mlp_gate, raw, &mut mlp_down, pool)?;
            crate::qwen_image_ops::add_bias(&mut mlp_down, raw, &block.down_bias)?;
            for (a, &b) in x.iter_mut().zip(&mlp_down) {
                *a += b;
            }
        }

        let groups = raw / (self.merge_size * self.merge_size);
        let merger_in = self.hidden * self.merge_size * self.merge_size;
        let mut merged_window = vec![0f32; groups * self.out_hidden];
        let mut merger_input = vec![0f32; groups * merger_in];
        for g in 0..groups {
            for k in 0..self.merge_size * self.merge_size {
                let src = &x[(g * self.merge_size * self.merge_size + k) * self.hidden
                    ..(g * self.merge_size * self.merge_size + k + 1) * self.hidden];
                Self::rms_norm(
                    src,
                    &self.merger.norm,
                    &mut merger_input
                        [g * merger_in + k * self.hidden..g * merger_in + (k + 1) * self.hidden],
                );
            }
        }
        let mut h = vec![0f32; groups * merger_in];
        self.merger
            .fc1
            .forward(&merger_input, groups, &mut h, pool)?;
        crate::qwen_image_ops::add_bias(&mut h, groups, &self.merger.fc1_bias)?;
        for v in &mut h {
            *v = Self::gelu_exact(*v);
        }
        self.merger
            .fc2
            .forward(&h, groups, &mut merged_window, pool)?;
        crate::qwen_image_ops::add_bias(&mut merged_window, groups, &self.merger.fc2_bias)?;

        // `window_index` is indexed by reordered group, while the text image
        // placeholders remain in the original merged-grid order.
        let mut out = vec![0f32; groups * self.out_hidden];
        for (window_group, &raw_token) in order
            .iter()
            .step_by(self.merge_size * self.merge_size)
            .enumerate()
        {
            let original_group = raw_token / (self.merge_size * self.merge_size);
            out[original_group * self.out_hidden..(original_group + 1) * self.out_hidden]
                .copy_from_slice(
                    &merged_window
                        [window_group * self.out_hidden..(window_group + 1) * self.out_hidden],
                );
        }
        Ok(out)
    }

    pub(crate) fn forward(&self, images: &[PreparedImage]) -> Result<Vec<Vec<f32>>, String> {
        images.iter().map(|image| self.forward_one(image)).collect()
    }
}

/// Numerical Recipes/Abramowitz-style erf used by torch's exact GELU path.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.5 * x);
    let tau = t
        * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    sign * (1.0 - tau)
}

#[cfg(test)]
mod tests {
    use super::{condition_dimensions, round_positive, smart_resize, ProcessorConfig};
    use image::{Rgb, RgbImage};

    #[test]
    fn python_round_uses_ties_to_even() {
        assert_eq!(round_positive(2.5), 2);
        assert_eq!(round_positive(3.5), 4);
    }

    #[test]
    fn official_condition_and_smart_resize_are_area_bounded() {
        let (w, h) = condition_dimensions(640, 480).unwrap();
        assert_eq!((w, h), (448, 320));
        let (h2, w2) = smart_resize(h, w, 28, 3136, 12_845_056).unwrap();
        assert_eq!((h2 % 28, w2 % 28), (0, 0));
        assert!(h2 * w2 >= 3136);
    }

    #[test]
    fn processor_config_accepts_official_shape() {
        let p = ProcessorConfig::from_json(
            br#"{"min_pixels":3136,"max_pixels":12845056,"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711],"resample":3}"#,
        )
        .unwrap();
        assert_eq!(
            (p.patch_size, p.temporal_patch_size, p.merge_size),
            (14, 2, 2)
        );
    }

    #[test]
    fn non_square_processor_preserves_patch_and_token_geometry() {
        let cfg = ProcessorConfig {
            min_pixels: 16,
            max_pixels: 64,
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
            mean: [0.0; 3],
            std: [1.0; 3],
            resample: 0,
        };
        let image = RgbImage::from_fn(3, 5, |x, y| Rgb([x as u8, y as u8, 17]));
        let prepared = super::prepare_image(&image, &cfg).unwrap();
        assert_eq!(prepared.grid_t, 1);
        assert_eq!(prepared.grid_h % 2, 0);
        assert_eq!(prepared.grid_w % 2, 0);
        assert_eq!(
            prepared.image_tokens(2).unwrap(),
            prepared.grid_h * prepared.grid_w / 4
        );
        assert_eq!(
            prepared.patches.len(),
            prepared.grid_h * prepared.grid_w * prepared.patch_dim
        );
    }
}
