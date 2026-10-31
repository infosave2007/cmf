//! LFM3-VL (Liquid AI `lfm3_vl`): the vision tower, its projector into the
//! LFM3 text model, the image preprocessing of `Lfm3VlImageProcessor`, and
//! the prompt assembly that replaces the `<|image|>` placeholders with the
//! tower's rows.
//!
//! The text model is plain LFM3 (`arch_name = "lfm3"`); a file is
//! vision-capable when it carries `vision_tower.*` and the checkpoint and
//! processor configs under `provenance.lfm3_vl`.
//!
//! ## Preprocessing (`Lfm3VlImageProcessor`)
//!
//! 1. **Resize** to `smart_resize(h, w, factor 32, min_pixels, max_pixels)`
//!    — Python `round`, so halves go to even — with torchvision's uint8
//!    antialiased bicubic (`egemma2_vision::resize_bicubic_aa`, byte for
//!    byte).
//! 2. **Standardize** `(x − 127.5) / 127.5` (rescale 1/255 fused into mean
//!    0.5 / std 0.5, as the fast processor does).
//! 3. **Patchify** like Qwen2-VL: the image twice (temporal patch 2), 16×16
//!    patches, rows in 2×2 merge-block order, each row `(c, t, py, px)`.
//!
//! ## The tower (`Lfm3VlVisionModel`)
//!
//! * `patchifier` (a bias-free linear over the 1536-wide patch row), then
//!   RMSNorm;
//! * LFM-style pre-norm layers, `h += mixer(N(h)); h += SwiGLU(N(h))`, the
//!   mixer alternating between a gated depthwise 3×3 convolution over the
//!   frame's patch grid (`in_proj → B·x → conv → C·y → out_proj`, zero
//!   padding, in SPATIAL order — the rows are unpermuted from merge-block
//!   order for it) and bidirectional GQA over the frame's patches with
//!   per-head Q/K RMSNorm and axial RoPE: 16 frequencies of θ^(-2i/32)
//!   (head_dim 64), angles `[row, col]` concatenated and repeated, applied
//!   `rotate_half` over the whole head;
//! * no final norm. The projector LayerNorms each row (with bias), joins the
//!   four rows of a merge block (consecutive) into one 4096-wide row and
//!   runs `linear_1 → exact GELU → linear_2` to the text width.
//!
//! Parity: `scripts/lfm3/lfm3_vl_ref.py` dumps the processor's pixels, the
//! tower rows and the image-prompt logits from Transformers.

use crate::Pipeline;
use crate::media::{self, RgbFrame};
use crate::mimo_vision::{MimoProcessorConfig, VisualInput, VisualKind};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::sync::Arc;

/// Tensor whose presence marks a vision-capable LFM3 file.
pub const PATCHIFIER: &str = "vision_tower.patchifier.weight";

/// Everything the vision path reads from `provenance.lfm3_vl`.
#[derive(Clone, Debug)]
pub struct Lfm3VlConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    /// Per layer: true = attention, false = gated 2-D convolution.
    pub attention: Vec<bool>,
    pub conv_kernel: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub projector_hidden: usize,
    pub projector_eps: f64,
    pub text_hidden: usize,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
    /// Video sampling (`Lfm3VlVideoProcessor`): frames per second of
    /// source time, the frame-count bounds, and the pixel bounds over all
    /// frames together.
    pub video_fps: f64,
    pub video_min_frames: usize,
    pub video_max_frames: usize,
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
}

/// Whether `model` carries the LFM3-VL vision tower.
pub fn has_vision(model: &CmfModel) -> bool {
    model.header.arch.arch_name == "lfm3" && model.tensor(PATCHIFIER).is_some()
}

fn need_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("lfm3_vl config: missing '{key}'"))
}

fn need_f64(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("lfm3_vl config: missing '{key}'"))
}

fn triple(v: &Value, key: &str) -> Result<[f32; 3], String> {
    let a = v
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("lfm3_vl processor: '{key}' must hold 3 numbers"))?;
    let mut out = [0f32; 3];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x
            .as_f64()
            .ok_or_else(|| format!("lfm3_vl processor: '{key}' must hold numbers"))? as f32;
    }
    Ok(out)
}

impl Lfm3VlConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let prov = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("lfm3_vl"))
            .ok_or("this LFM3 file carries no vision config (provenance.lfm3_vl)")?;
        Self::from_json(&prov["config"], &prov["processor"])
    }

    pub fn from_json(cfg: &Value, processor: &Value) -> Result<Self, String> {
        let vc = &cfg["vision_config"];
        let tc = &cfg["text_config"];
        let ip = &processor["image_processor"];
        let hidden = need_usize(vc, "hidden_size")?;
        let heads = need_usize(vc, "num_attention_heads")?;
        let attention: Vec<bool> = vc
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("lfm3_vl config: vision_config.layer_types missing")?
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("conv") => Ok(false),
                other => Err(format!("lfm3_vl: unknown vision layer type {other:?}")),
            })
            .collect::<Result<_, _>>()?;
        if attention.len() != need_usize(vc, "num_hidden_layers")? {
            return Err("lfm3_vl: layer_types does not match num_hidden_layers".into());
        }
        let rope = &vc["rope_parameters"];
        if rope.get("rope_type").and_then(Value::as_str) != Some("axial") {
            return Err("lfm3_vl: only the axial vision RoPE is supported".into());
        }
        if cfg.get("projector_hidden_act").and_then(Value::as_str) != Some("gelu") {
            return Err("lfm3_vl: only the exact-GELU projector is supported".into());
        }
        let size = &ip["size"];
        let vp = &processor["video_processor"];
        if vp.get("cap_pixels_per_frame").and_then(Value::as_bool) == Some(true) {
            return Err("lfm3_vl: cap_pixels_per_frame=true is not supported".into());
        }
        let c = Self {
            hidden,
            heads,
            kv_heads: need_usize(vc, "num_key_value_heads")?,
            head_dim: hidden / heads,
            intermediate: need_usize(vc, "intermediate_size")?,
            attention,
            conv_kernel: need_usize(vc, "conv_kernel_size")?,
            patch_size: need_usize(vc, "patch_size")?,
            temporal_patch_size: need_usize(vc, "temporal_patch_size")?,
            merge_size: need_usize(vc, "spatial_merge_size")?,
            norm_eps: need_f64(vc, "norm_eps")?,
            rope_theta: need_f64(rope, "rope_theta")? as f32,
            projector_hidden: need_usize(cfg, "projector_hidden_size")?,
            projector_eps: need_f64(cfg, "projector_layer_norm_eps")?,
            text_hidden: need_usize(tc, "hidden_size")?,
            image_token_id: need_usize(cfg, "image_token_id")? as u32,
            video_token_id: need_usize(cfg, "video_token_id")? as u32,
            min_pixels: need_usize(size, "shortest_edge")?,
            max_pixels: need_usize(size, "longest_edge")?,
            mean: triple(ip, "image_mean")?,
            std: triple(ip, "image_std")?,
            rescale: need_f64(ip, "rescale_factor")? as f32,
            video_fps: need_f64(vp, "fps")?,
            video_min_frames: need_usize(vp, "min_frames")?,
            video_max_frames: need_usize(vp, "max_frames")?,
            video_min_pixels: need_usize(&vp["size"], "shortest_edge")?,
            video_max_pixels: need_usize(&vp["size"], "longest_edge")?,
        };
        if cfg.get("downsample_factor").and_then(Value::as_u64) != Some(c.merge_size as u64) {
            return Err("lfm3_vl: downsample_factor must equal spatial_merge_size".into());
        }
        if c.head_dim % 4 != 0 || c.heads % c.kv_heads != 0 || c.conv_kernel % 2 == 0 {
            return Err("lfm3_vl: unsupported vision geometry".into());
        }
        if need_usize(ip, "patch_size")? != c.patch_size
            || need_usize(ip, "merge_size")? != c.merge_size
            || need_usize(ip, "temporal_patch_size")? != c.temporal_patch_size
        {
            return Err("lfm3_vl: processor and tower patch geometry disagree".into());
        }
        Ok(c)
    }

    /// The patch geometry in the form `mimo_vision::patchify` reads.
    fn patch_cfg(&self) -> MimoProcessorConfig {
        MimoProcessorConfig {
            patch_size: self.patch_size,
            merge_size: self.merge_size,
            temporal_patch_size: self.temporal_patch_size,
            ..Default::default()
        }
    }

    fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// Python's `round` (halves to even).
fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `smart_resize` of `image_processing_lfm3_vl.py`: both sides multiples of
/// `factor`, the area inside `[min_pixels, max_pixels]`, the aspect kept.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("image has a zero side".into());
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let mut hb = py_round(h / f) * f;
    let mut wb = py_round(w / f) * f;
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    if hb < f || wb < f {
        return Err(format!("image {height}x{width} resizes to an empty grid"));
    }
    Ok((hb as usize, wb as usize))
}

/// Preprocess one image into tower rows. `max_pixels` lowers the
/// processor's ceiling (`--image-max-pixels`).
pub fn prepare_image(
    frame: &RgbFrame,
    cfg: &Lfm3VlConfig,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    let factor = cfg.patch_size * cfg.merge_size;
    let max_px = max_pixels.map_or(cfg.max_pixels, |m| m.min(cfg.max_pixels));
    let (hb, wb) = smart_resize(frame.height, frame.width, factor, cfg.min_pixels, max_px)?;
    let chw = standardize(frame, hb, wb, cfg);
    let frames: Vec<&[f32]> = (0..cfg.temporal_patch_size).map(|_| chw.as_slice()).collect();
    let (rows, grid_t, grid_h, grid_w) =
        crate::mimo_vision::patchify(&frames, hb, wb, &cfg.patch_cfg())?;
    Ok(VisualInput {
        kind: VisualKind::Image,
        rows,
        grid_t,
        grid_h,
        grid_w,
        patch_dim: cfg.patch_dim(),
        merge_size: cfg.merge_size,
        timestamps: Vec::new(),
        frame_indices: Vec::new(),
        resized: (hb, wb),
    })
}

/// numpy's `round` (halves to even) on a non-negative value.
fn np_rint(x: f64) -> f64 {
    py_round(x)
}

/// `Lfm3VlVideoProcessor.sample_frames` with `fps`: `int(total / src_fps ·
/// fps)` frames clamped to `[min_frames, max_frames]` and the clip, at
/// `linspace(0, total − 1, n).round()`.
pub fn sample_indices(total: usize, src_fps: f64, cfg: &Lfm3VlConfig) -> Result<Vec<usize>, String> {
    if total == 0 {
        return Err("video has no frames".into());
    }
    if !(src_fps > 0.0 && src_fps.is_finite()) {
        return Err(format!("video fps must be positive, got {src_fps}"));
    }
    let n = (total as f64 / src_fps * cfg.video_fps) as usize;
    let n = n.max(cfg.video_min_frames).min(cfg.video_max_frames).min(total);
    Ok((0..n)
        .map(|i| {
            let x = if n == 1 { 0.0 } else { (total - 1) as f64 * i as f64 / (n - 1) as f64 };
            np_rint(x) as usize
        })
        .collect())
}

/// The video `smart_resize` of `video_processing_lfm3_vl.py`: the pixel
/// bounds hold for all `num_frames` frames together.
pub fn video_smart_resize(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if num_frames < temporal {
        return Err(format!("{num_frames} frames is fewer than the temporal patch {temporal}"));
    }
    let (mut h, mut w, f) = (height as f64, width as f64, factor as f64);
    if h < f || w < f {
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!("absolute aspect ratio must be smaller than 200, got {}", h.max(w) / h.min(w)));
    }
    let nf = num_frames as f64;
    let mut hb = py_round(h / f) * f;
    let mut wb = py_round(w / f) * f;
    let tb = py_round(nf / temporal as f64) * temporal as f64;
    if tb * hb * wb > max_pixels as f64 {
        let beta = (nf * h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if tb * hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (nf * h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Resize to `(hb, wb)` with the processor's bicubic and standardize to
/// planar `[3][hb·wb]`.
fn standardize(frame: &RgbFrame, hb: usize, wb: usize, cfg: &Lfm3VlConfig) -> Vec<f32> {
    let resized = if (hb, wb) == (frame.height, frame.width) {
        frame.clone()
    } else {
        crate::egemma2_vision::resize_bicubic_aa(frame, hb, wb)
    };
    // The fast processor folds the 1/255 rescale into mean and std.
    let plane = hb * wb;
    let mut chw = vec![0f32; 3 * plane];
    for c in 0..3 {
        let m = cfg.mean[c] / cfg.rescale;
        let s = cfg.std[c] / cfg.rescale;
        for (i, px) in resized.data.chunks_exact(3).enumerate() {
            chw[c * plane + i] = (px[c] as f32 - m) / s;
        }
    }
    chw
}

/// Preprocess a video: sample at the processor's rate, resize every frame
/// to one size, pad to whole temporal patches with the last frame, patchify.
/// `timestamps` carry one entry per temporal patch (the mean of its first
/// and last frame time); `max_pixels` lowers the all-frames pixel ceiling.
pub fn prepare_video(
    src: &crate::mimo_vision::VideoSource,
    cfg: &Lfm3VlConfig,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    let indices = sample_indices(src.frame_count(), src.fps(), cfg)?;
    let first = src.read_frame(indices[0])?;
    let max_px = max_pixels.map_or(cfg.video_max_pixels, |m| m.min(cfg.video_max_pixels));
    let (hb, wb) = video_smart_resize(
        indices.len(),
        first.height,
        first.width,
        cfg.temporal_patch_size,
        cfg.patch_size * cfg.merge_size,
        cfg.video_min_pixels,
        max_px,
    )?;
    let mut planes = Vec::with_capacity(indices.len() + 1);
    for (k, &idx) in indices.iter().enumerate() {
        let frame = if k == 0 { first.clone() } else { src.read_frame(idx)? };
        if (frame.height, frame.width) != (first.height, first.width) {
            return Err("video frames differ in size".into());
        }
        planes.push(standardize(&frame, hb, wb, cfg));
    }
    let tp = cfg.temporal_patch_size;
    let mut padded = indices.clone();
    while planes.len() % tp != 0 {
        planes.push(planes.last().unwrap().clone());
        padded.push(*padded.last().unwrap());
    }
    let frames: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let (rows, grid_t, grid_h, grid_w) =
        crate::mimo_vision::patchify(&frames, hb, wb, &cfg.patch_cfg())?;
    let times: Vec<f64> = padded.iter().map(|&i| i as f64 / src.fps()).collect();
    let timestamps = times
        .chunks(tp)
        .map(|c| ((c[0] + c[tp - 1]) / 2.0) as f32)
        .collect();
    Ok(VisualInput {
        kind: VisualKind::Video,
        rows,
        grid_t,
        grid_h,
        grid_w,
        patch_dim: cfg.patch_dim(),
        merge_size: cfg.merge_size,
        timestamps,
        frame_indices: padded,
        resized: (hb, wb),
    })
}

pub(crate) fn load_vec(model: &CmfModel, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
    }
    Ok(v)
}

/// A tower matrix as an exact f32 operand of the host GEMM. The tower is
/// stored f16 and run in f32 on purpose: through the device GEMMs (f16 /
/// tf32-class operands) a few rows of a 4320-patch image drift to a cosine
/// of 0.97 against Transformers float32, on the host every row stays above
/// 0.99999.
pub(crate) struct Dense {
    w: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Dense {
    /// `y[n, rows] = x[n, cols] · Wᵀ`.
    pub(crate) fn matmat(&self, x: &[f32], n: usize, y: &mut [f32], pool: Option<&Pool>) {
        crate::fcd_ops::gemm_nt_host(x, &self.w, y, n, self.cols, self.rows, pool);
    }
}

pub(crate) fn load_proj(model: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Result<Dense, String> {
    let w = crate::dit::cmf_f32(model, name)?;
    if w.len() != rows * cols {
        return Err(format!(
            "tensor '{name}' has {} values, expected [{rows}, {cols}]",
            w.len()
        ));
    }
    Ok(Dense { w, rows, cols })
}

enum Mixer {
    Conv {
        in_proj: Dense,
        /// `[channel][ky][kx]`, the source `[C, 1, k, k]` flattened.
        taps: Vec<f32>,
        out_proj: Dense,
    },
    Attn {
        q: Dense,
        k: Dense,
        v: Dense,
        o: Dense,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

struct Layer {
    op_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: Mixer,
    w1: Dense,
    w3: Dense,
    w2: Dense,
}

/// The LFM3-VL vision tower and projector.
pub struct Lfm3VlTower {
    pub cfg: Lfm3VlConfig,
    patchifier: Dense,
    emb_norm: Vec<f32>,
    layers: Vec<Layer>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    l1: Dense,
    l1_b: Vec<f32>,
    l2: Dense,
    l2_b: Vec<f32>,
    pool: Option<Arc<Pool>>,
    gpu_attention: bool,
}

pub(crate) fn rms_rows(x: &[f32], w: &[f32], eps: f64, out: &mut [f32], pool: Option<&Pool>) {
    let d = w.len();
    let rows = x.len() / d;
    let dst = crate::pool::SendMut::new(out.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for r in lo..hi {
            let xr = &x[r * d..(r + 1) * d];
            let ss = xr.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64;
            let inv = 1.0 / (ss + eps).sqrt();
            for (j, (&v, &g)) in xr.iter().zip(w).enumerate() {
                // SAFETY: rows are disjoint across workers.
                unsafe { *dst.at(r * d + j) = (v as f64 * inv) as f32 * g };
            }
        }
    };
    match pool {
        Some(p) => p.run_rows(rows, &f),
        None => f(0, rows),
    }
}

/// `erf` by the Numerical Recipes rational form (1.2e-7 relative).
pub(crate) fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { 1.0 - ans } else { ans - 1.0 }
}

pub(crate) fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        for (v, &bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

impl Lfm3VlTower {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cfg = Lfm3VlConfig::from_model(model)?;
        let (d, inter, hd) = (cfg.hidden, cfg.intermediate, cfg.head_dim);
        let (qd, kvd) = (cfg.heads * hd, cfg.kv_heads * hd);
        let kk = cfg.conv_kernel * cfg.conv_kernel;
        let mut layers = Vec::with_capacity(cfg.attention.len());
        for (i, &attn) in cfg.attention.iter().enumerate() {
            let p = |leaf: &str| format!("vision_tower.layers.{i}.{leaf}");
            let mixer = if attn {
                Mixer::Attn {
                    q: load_proj(model, &p("self_attn.q_proj.weight"), qd, d)?,
                    k: load_proj(model, &p("self_attn.k_proj.weight"), kvd, d)?,
                    v: load_proj(model, &p("self_attn.v_proj.weight"), kvd, d)?,
                    o: load_proj(model, &p("self_attn.out_proj.weight"), d, qd)?,
                    q_norm: load_vec(model, &p("self_attn.q_layernorm.weight"), hd)?,
                    k_norm: load_vec(model, &p("self_attn.k_layernorm.weight"), hd)?,
                }
            } else {
                Mixer::Conv {
                    in_proj: load_proj(model, &p("conv.in_proj.weight"), 3 * d, d)?,
                    taps: load_vec(model, &p("conv.spatial_conv.weight"), d * kk)?,
                    out_proj: load_proj(model, &p("conv.out_proj.weight"), d, d)?,
                }
            };
            layers.push(Layer {
                op_norm: load_vec(model, &p("operator_norm.weight"), d)?,
                ffn_norm: load_vec(model, &p("ffn_norm.weight"), d)?,
                mixer,
                w1: load_proj(model, &p("feed_forward.w1.weight"), inter, d)?,
                w3: load_proj(model, &p("feed_forward.w3.weight"), inter, d)?,
                w2: load_proj(model, &p("feed_forward.w2.weight"), d, inter)?,
            });
        }
        let merged = d * cfg.merge_size * cfg.merge_size;
        let mp = |leaf: &str| format!("multi_modal_projector.{leaf}");
        Ok(Self {
            patchifier: load_proj(model, PATCHIFIER, d, cfg.patch_dim())?,
            emb_norm: load_vec(model, "vision_tower.embedding_norm.weight", d)?,
            layers,
            ln_w: load_vec(model, &mp("layer_norm.weight"), d)?,
            ln_b: load_vec(model, &mp("layer_norm.bias"), d)?,
            l1: load_proj(model, &mp("linear_1.weight"), cfg.projector_hidden, merged)?,
            l1_b: load_vec(model, &mp("linear_1.bias"), cfg.projector_hidden)?,
            l2: load_proj(model, &mp("linear_2.weight"), cfg.text_hidden, cfg.projector_hidden)?,
            l2_b: load_vec(model, &mp("linear_2.bias"), cfg.text_hidden)?,
            pool,
            gpu_attention: std::env::var("CMF_LFM3_VL_GPU_ATTN").as_deref() != Ok("0"),
            cfg,
        })
    }

    /// `(y, x)` of every row of one frame, rows in merge-block order.
    fn frame_coords(&self, gh: usize, gw: usize) -> Vec<(usize, usize)> {
        let m = self.cfg.merge_size;
        let mut out = Vec::with_capacity(gh * gw);
        for a in 0..gh / m {
            for b in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        out.push((a * m + mh, b * m + mw));
                    }
                }
            }
        }
        out
    }

    /// Axial RoPE `(cos, sin)` for one frame, `[rows, head_dim]` each,
    /// laid out `[row, col, row, col]` in quarters.
    fn rope_tables(&self, coords: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.head_dim;
        let quarter = hd / 4;
        let dim = (hd / 2) as f32;
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0f32 / self.cfg.rope_theta.powf((2 * i) as f32 / dim))
            .collect();
        let n = coords.len();
        let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
        for (r, &(y, x)) in coords.iter().enumerate() {
            for d in 0..hd {
                let q = d % (hd / 2);
                let ang = if q < quarter {
                    y as f32 * inv[q]
                } else {
                    x as f32 * inv[q - quarter]
                };
                cos[r * hd + d] = ang.cos();
                sin[r * hd + d] = ang.sin();
            }
        }
        (cos, sin)
    }

    /// Per-head RMSNorm then RoPE, in place, on `[rows, heads·hd]`.
    fn norm_rope(&self, x: &mut [f32], heads: usize, w: &[f32], cos: &[f32], sin: &[f32]) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let eps = self.cfg.norm_eps;
        let rows = x.len() / (heads * hd);
        let dst = crate::pool::SendMut::new(x.as_mut_ptr());
        let f = |lo: usize, hi: usize| {
            let mut tmp = vec![0f32; hd];
            for r in lo..hi {
                for h in 0..heads {
                    // SAFETY: rows are disjoint across workers.
                    let v = unsafe { std::slice::from_raw_parts_mut(dst.at((r * heads + h) * hd), hd) };
                    let ss = v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / hd as f64;
                    let inv = 1.0 / (ss + eps).sqrt();
                    for (t, (&a, &g)) in tmp.iter_mut().zip(v.iter().zip(w)) {
                        *t = (a as f64 * inv) as f32 * g;
                    }
                    let (c, s) = (&cos[r * hd..(r + 1) * hd], &sin[r * hd..(r + 1) * hd]);
                    for d in 0..hd {
                        let rot = if d < half { -tmp[d + half] } else { tmp[d - half] };
                        v[d] = tmp[d] * c[d] + rot * s[d];
                    }
                }
            }
        };
        match self.pool.as_deref() {
            Some(p) => p.run_rows(rows, &f),
            None => f(0, rows),
        }
    }

    /// Bidirectional GQA over one frame's `l` rows starting at row `c0`;
    /// writes token-major `[l, heads·hd]` into `out`.
    fn attend(&self, q: &[f32], k: &[f32], v: &[f32], c0: usize, l: usize, out: &mut [f32]) {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let scale = (hd as f32).powf(-0.5);
        if self.gpu_attention && l >= 64 && l * l * 4 <= (1usize << 30) && crate::gpu::enabled_here() {
            let mut qh = vec![0f32; nh * l * hd];
            let mut kh = vec![0f32; nkv * l * hd];
            let mut vh = vec![0f32; nkv * l * hd];
            for p in 0..l {
                let r = c0 + p;
                for h in 0..nh {
                    qh[(h * l + p) * hd..(h * l + p + 1) * hd]
                        .copy_from_slice(&q[r * qd + h * hd..r * qd + (h + 1) * hd]);
                }
                for h in 0..nkv {
                    kh[(h * l + p) * hd..(h * l + p + 1) * hd]
                        .copy_from_slice(&k[r * kvd + h * hd..r * kvd + (h + 1) * hd]);
                    vh[(h * l + p) * hd..(h * l + p + 1) * hd]
                        .copy_from_slice(&v[r * kvd + h * hd..r * kvd + (h + 1) * hd]);
                }
            }
            if crate::gpu::dit_attention(&qh, &kh, &vh, nh, nkv, l, hd, scale, &mut out[c0 * qd..(c0 + l) * qd]) {
                return;
            }
        }
        let pool = self.pool.as_deref();
        let group = nh / nkv;
        let tile = l.min(1024);
        let mut kh = vec![0f32; l * hd];
        let mut vt = vec![0f32; hd * l];
        let mut qh = vec![0f32; tile * hd];
        let mut sc = vec![0f32; tile * l];
        let mut oh = vec![0f32; tile * hd];
        for g in 0..nkv {
            for j in 0..l {
                let r = c0 + j;
                kh[j * hd..(j + 1) * hd].copy_from_slice(&k[r * kvd + g * hd..r * kvd + (g + 1) * hd]);
                for d in 0..hd {
                    vt[d * l + j] = v[r * kvd + g * hd + d];
                }
            }
            for h in g * group..(g + 1) * group {
                for t0 in (0..l).step_by(tile) {
                    let tq = tile.min(l - t0);
                    for i in 0..tq {
                        let r = c0 + t0 + i;
                        for d in 0..hd {
                            qh[i * hd + d] = q[r * qd + h * hd + d] * scale;
                        }
                    }
                    crate::fcd_ops::gemm_nt_host(&qh[..tq * hd], &kh, &mut sc[..tq * l], tq, hd, l, pool);
                    {
                        let sp = crate::pool::SendMut::new(sc.as_mut_ptr());
                        let f = |lo: usize, hi: usize| {
                            for i in lo..hi {
                                // SAFETY: disjoint score rows per worker.
                                let row = unsafe { std::slice::from_raw_parts_mut(sp.at(i * l), l) };
                                let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                                let mut sum = 0f32;
                                for x in row.iter_mut() {
                                    *x = (*x - mx).exp();
                                    sum += *x;
                                }
                                let inv = 1.0 / sum;
                                for x in row.iter_mut() {
                                    *x *= inv;
                                }
                            }
                        };
                        match pool {
                            Some(p) => p.run_rows(tq, &f),
                            None => f(0, tq),
                        }
                    }
                    crate::fcd_ops::gemm_nt_host(&sc[..tq * l], &vt, &mut oh[..tq * hd], tq, l, hd, pool);
                    for i in 0..tq {
                        let r = c0 + t0 + i;
                        out[r * qd + h * hd..r * qd + (h + 1) * hd].copy_from_slice(&oh[i * hd..(i + 1) * hd]);
                    }
                }
            }
        }
    }

    /// Depthwise `k×k` convolution with zero padding over one frame, rows in
    /// merge-block order (`coords`), `[l, d]` in and out.
    fn spatial_conv(&self, x: &[f32], taps: &[f32], coords: &[(usize, usize)], gh: usize, gw: usize, out: &mut [f32]) {
        let d = self.cfg.hidden;
        let k = self.cfg.conv_kernel;
        let pad = (k / 2) as isize;
        let mut row_at = vec![usize::MAX; gh * gw];
        for (r, &(y, xx)) in coords.iter().enumerate() {
            row_at[y * gw + xx] = r;
        }
        let dst = crate::pool::SendMut::new(out.as_mut_ptr());
        let f = |lo: usize, hi: usize| {
            for r in lo..hi {
                let (y, xx) = coords[r];
                // SAFETY: output rows are disjoint across workers.
                let o = unsafe { std::slice::from_raw_parts_mut(dst.at(r * d), d) };
                o.fill(0.0);
                for ky in 0..k {
                    let sy = y as isize + ky as isize - pad;
                    if sy < 0 || sy >= gh as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let sx = xx as isize + kx as isize - pad;
                        if sx < 0 || sx >= gw as isize {
                            continue;
                        }
                        let src = row_at[sy as usize * gw + sx as usize];
                        let xs = &x[src * d..(src + 1) * d];
                        let t = ky * k + kx;
                        for c in 0..d {
                            o[c] += taps[c * k * k + t] * xs[c];
                        }
                    }
                }
            }
        };
        match self.pool.as_deref() {
            Some(p) => p.run_rows(coords.len(), &f),
            None => f(0, coords.len()),
        }
    }

    /// The tower and projector over one image or video: `[tokens, text_hidden]`.
    pub fn forward(&self, input: &VisualInput) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let pool = self.pool.as_deref();
        let (d, hd) = (c.hidden, c.head_dim);
        let (gt, gh, gw) = (input.grid_t, input.grid_h, input.grid_w);
        let l = gh * gw;
        let n = gt * l;
        if input.patch_dim != c.patch_dim() || input.rows.len() != n * c.patch_dim() {
            return Err("lfm3_vl: patch rows do not match the tower".into());
        }
        let mut x = vec![0f32; n * d];
        self.patchifier.matmat(&input.rows, n, &mut x, pool);
        let mut xn = vec![0f32; n * d];
        rms_rows(&x, &self.emb_norm, c.norm_eps, &mut xn, pool);
        std::mem::swap(&mut x, &mut xn);
        let coords = self.frame_coords(gh, gw);
        let (cos1, sin1) = self.rope_tables(&coords);
        let (cos, sin): (Vec<f32>, Vec<f32>) = (cos1.repeat(gt), sin1.repeat(gt));
        let (qd, kvd) = (c.heads * hd, c.kv_heads * hd);
        let mut mix = vec![0f32; n * d];
        for layer in &self.layers {
            rms_rows(&x, &layer.op_norm, c.norm_eps, &mut xn, pool);
            match &layer.mixer {
                Mixer::Conv { in_proj, taps, out_proj } => {
                    let mut bcx = vec![0f32; n * 3 * d];
                    in_proj.matmat(&xn, n, &mut bcx, pool);
                    let mut gated = vec![0f32; n * d];
                    for r in 0..n {
                        let row = &bcx[r * 3 * d..(r + 1) * 3 * d];
                        for j in 0..d {
                            gated[r * d + j] = row[j] * row[2 * d + j];
                        }
                    }
                    let mut conv = vec![0f32; n * d];
                    for t in 0..gt {
                        self.spatial_conv(
                            &gated[t * l * d..(t + 1) * l * d],
                            taps,
                            &coords,
                            gh,
                            gw,
                            &mut conv[t * l * d..(t + 1) * l * d],
                        );
                    }
                    for r in 0..n {
                        for j in 0..d {
                            conv[r * d + j] *= bcx[r * 3 * d + d + j];
                        }
                    }
                    out_proj.matmat(&conv, n, &mut mix, pool);
                }
                Mixer::Attn { q, k, v, o, q_norm, k_norm } => {
                    let mut qb = vec![0f32; n * qd];
                    let mut kb = vec![0f32; n * kvd];
                    let mut vb = vec![0f32; n * kvd];
                    q.matmat(&xn, n, &mut qb, pool);
                    k.matmat(&xn, n, &mut kb, pool);
                    v.matmat(&xn, n, &mut vb, pool);
                    self.norm_rope(&mut qb, c.heads, q_norm, &cos, &sin);
                    self.norm_rope(&mut kb, c.kv_heads, k_norm, &cos, &sin);
                    let mut ao = vec![0f32; n * qd];
                    for t in 0..gt {
                        self.attend(&qb, &kb, &vb, t * l, l, &mut ao);
                    }
                    o.matmat(&ao, n, &mut mix, pool);
                }
            }
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
            rms_rows(&x, &layer.ffn_norm, c.norm_eps, &mut xn, pool);
            let inter = c.intermediate;
            let mut g = vec![0f32; n * inter];
            let mut u = vec![0f32; n * inter];
            layer.w1.matmat(&xn, n, &mut g, pool);
            layer.w3.matmat(&xn, n, &mut u, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = *a / (1.0 + (-*a).exp()) * b;
            }
            layer.w2.matmat(&g, n, &mut mix, pool);
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
        }
        // Projector: LayerNorm per row, 2×2 merge blocks (consecutive rows)
        // into one row, linear → exact GELU → linear.
        for r in 0..n {
            let row = &mut x[r * d..(r + 1) * d];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + c.projector_eps).sqrt();
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((*v as f64 - mean) * inv) as f32 * self.ln_w[j] + self.ln_b[j];
            }
        }
        let m2 = c.merge_size * c.merge_size;
        let tokens = n / m2;
        let mut h1 = vec![0f32; tokens * c.projector_hidden];
        self.l1.matmat(&x, tokens, &mut h1, pool);
        add_bias(&mut h1, &self.l1_b);
        for v in h1.iter_mut() {
            *v = (0.5 * *v as f64 * (1.0 + erf(*v as f64 / std::f64::consts::SQRT_2))) as f32;
        }
        let mut out = vec![0f32; tokens * c.text_hidden];
        self.l2.matmat(&h1, tokens, &mut out, pool);
        add_bias(&mut out, &self.l2_b);
        if !out.iter().all(|v| v.is_finite()) {
            return Err("lfm3_vl: the tower produced non-finite rows".into());
        }
        Ok(out)
    }
}

/// A chat prompt with its media rows: `rows` is the full prompt embedding
/// `[ids, hidden]` (text rows from the embedding table), None for text.
pub struct PreparedVlPrompt {
    pub token_ids: Vec<u32>,
    pub rows: Option<Vec<f32>>,
}

/// Options of the media path.
#[derive(Default, Clone)]
pub struct VlOptions {
    /// Lowers the processor's pixel ceiling (per image, over all frames of
    /// a video).
    pub image_max_pixels: Option<usize>,
    /// Frame rate of a frame directory (a Y4M file carries its own).
    pub video_fps: Option<f64>,
}

#[derive(Debug, PartialEq)]
enum Item {
    Image(Value),
    Video(Value),
}

/// Replace every image or video part of `messages` with `{"type": "image"}`
/// / `{"type": "video"}` (what the template renders as
/// `<|vision_start|><|image|><|vision_end|>`, likewise `<|video|>`) and
/// collect the sources in order. Other non-text parts are refused.
fn normalize(messages: &[Value]) -> Result<(Vec<Value>, Vec<Item>), String> {
    let mut messages = messages.to_vec();
    let mut items = Vec::new();
    for message in &mut messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.is_string() {
                continue;
            }
            let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
            let record = |v: &Value| match v.as_str() {
                Some(url) => json!({"url": url}),
                None => v.clone(),
            };
            if typ == "image" || typ == "image_url" || part.get("image").is_some() || part.get("image_url").is_some() {
                let source = part.get("image_url").or_else(|| part.get("image")).unwrap_or(part);
                items.push(Item::Image(record(source)));
                *part = json!({"type": "image"});
            } else if typ == "video" || typ == "video_url" || part.get("video").is_some() || part.get("video_url").is_some() {
                let source = part.get("video_url").or_else(|| part.get("video")).unwrap_or(part);
                let mut rec = record(source);
                if let (Some(fps), Some(obj)) = (part.get("fps").cloned(), rec.as_object_mut()) {
                    obj.entry("fps").or_insert(fps);
                }
                items.push(Item::Video(rec));
                *part = json!({"type": "video"});
            } else if matches!(typ, "audio" | "input_audio" | "audio_url") {
                // Speech: LFM3-Omni's prepare collects it (`lfm3_audio`);
                // a vision-only file refuses it there.
            } else if typ != "text" && part.get("text").is_none() {
                return Err(format!(
                    "LFM3-VL: unsupported content part '{typ}' (text, images and video only)"
                ));
            }
        }
    }
    Ok((messages, items))
}

/// Whether any message carries an image or video part.
pub fn has_media(messages: &[Value]) -> Result<bool, String> {
    Ok(!normalize(messages)?.1.is_empty())
}

/// [`prepare`] on the pipeline's own file.
pub fn prepare_for_pipeline(
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    options: &VlOptions,
) -> Result<PreparedVlPrompt, String> {
    let model = pipeline
        .model
        .as_ref()
        .ok_or("LFM3-VL media require a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking, options)
}

/// The processor's video placeholder: per temporal patch
/// `<T.T seconds><|vision_start|><|video|>×N<|vision_end|>`.
fn video_placeholder(labels: &[String], per_step: usize) -> String {
    let mut out = String::new();
    for label in labels {
        out.push_str(label);
        out.push_str("<|vision_start|>");
        for _ in 0..per_step {
            out.push_str("<|video|>");
        }
        out.push_str("<|vision_end|>");
    }
    out
}

/// Render `messages` with the file's template, run every image and video
/// through the tower, expand their placeholders the way `Lfm3VlProcessor`
/// does, and assemble the prompt rows.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    options: &VlOptions,
) -> Result<PreparedVlPrompt, String> {
    // LFM3-Omni: its speech clips first (each part becomes {"type":"audio"}).
    let speech = crate::lfm3_audio::has_audio(model);
    if !speech && crate::lfm3_audio::has_audio_parts(messages) {
        return Err("this LFM3 file takes no audio (LFM3-Omni and LFM3-Audio do)".into());
    }
    let (messages, clips) = if speech {
        crate::lfm3_audio::normalize(messages)
    } else {
        (messages.to_vec(), Vec::new())
    };
    let (messages, items) = normalize(&messages)?;
    let tok = &pipeline.tokenizer;
    if items.is_empty() && clips.is_empty() {
        let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
        return Ok(PreparedVlPrompt { token_ids: ids, rows: None });
    }
    let text = tok
        .render_chat_json(&messages, tools, thinking)
        .ok_or("LFM3-VL: the chat template failed to render")?;
    let tower = Lfm3VlTower::from_model(model, crate::pool::Pool::from_env())?;
    let cfg = &tower.cfg;
    let audio_enc = if clips.is_empty() {
        None
    } else {
        Some(crate::lfm3_audio::Lfm3AudioEncoder::from_model(model, crate::pool::Pool::from_env())?)
    };
    let audio_id = tok.token_to_id("<|audio|>");
    let mut clips = clips.into_iter();
    if cfg.text_hidden != pipeline.hidden_size {
        return Err("LFM3-VL: projector width differs from the text model".into());
    }
    // CMF_LFM3_VL_DUMP=DIR: each item's patch rows and tower rows as raw
    // little-endian f32 (`pixels_I.f32`, `rows_I.f32`) for parity checks.
    let dump = std::env::var_os("CMF_LFM3_VL_DUMP").map(std::path::PathBuf::from);
    let (mut images, mut videos) = (Vec::new(), Vec::new());
    for item in items {
        match item {
            Item::Image(src) => images.push(src),
            Item::Video(src) => videos.push(src),
        }
    }
    let (mut images, mut videos) = (images.into_iter(), videos.into_iter());
    let mut expanded = String::with_capacity(text.len());
    let mut media_rows: Vec<f32> = Vec::new();
    let mut rest = text.as_str();
    let mut k = 0usize;
    loop {
        let next_img = rest.find("<|image|>");
        let next_vid = rest.find("<|video|>");
        let next_aud = audio_enc.as_ref().and_then(|_| rest.find("<|audio|>"));
        // A speech clip (LFM3-Omni) before the next picture or video.
        if let Some(a) = next_aud.filter(|&a| next_img.is_none_or(|i| a < i) && next_vid.is_none_or(|v| a < v)) {
            let enc = audio_enc.as_ref().expect("checked above");
            expanded.push_str(&rest[..a]);
            rest = &rest[a + "<|audio|>".len()..];
            let src = clips.next().ok_or("LFM3: more <|audio|> placeholders than clips")?;
            let rows = enc.rows(&crate::egemma2_audio::load_audio_source(&src)?)?;
            for _ in 0..rows.len() / enc.text_hidden {
                expanded.push_str("<|audio|>");
            }
            media_rows.extend(rows);
            continue;
        }
        let (at, video) = match (next_img, next_vid) {
            (None, None) => break,
            (Some(i), None) => (i, false),
            (None, Some(v)) => (v, true),
            (Some(i), Some(v)) => (i.min(v), v < i),
        };
        expanded.push_str(&rest[..at]);
        let tag = if video { "<|video|>" } else { "<|image|>" };
        rest = &rest[at + tag.len()..];
        let t0 = std::time::Instant::now();
        let (input, placeholder) = if video {
            let source = videos.next().ok_or("LFM3-VL: more <|video|> placeholders than videos")?;
            let path = source
                .get("url")
                .or_else(|| source.get("path"))
                .and_then(Value::as_str)
                .ok_or("LFM3-VL video needs a local path (Y4M file or frame directory)")?;
            if path.starts_with("http:") || path.starts_with("https:") || path.starts_with("data:") {
                return Err("LFM3-VL video supports local Y4M files or frame directories only".into());
            }
            let fps = source.get("fps").and_then(Value::as_f64).or(options.video_fps);
            let src = crate::mimo_vision::VideoSource::open(
                std::path::Path::new(path.strip_prefix("file://").unwrap_or(path)),
                fps,
            )?;
            let input = prepare_video(&src, cfg, options.image_max_pixels)?;
            let tp = cfg.temporal_patch_size;
            let labels: Vec<String> = input
                .frame_indices
                .chunks(tp)
                .map(|c| {
                    let t = (c[0] as f64 / src.fps() + c[tp - 1] as f64 / src.fps()) / 2.0;
                    format!("<{t:.1} seconds>")
                })
                .collect();
            let ph = video_placeholder(&labels, input.tokens_per_step());
            (input, ph)
        } else {
            let source = images.next().ok_or("LFM3-VL: more <|image|> placeholders than images")?;
            let frame = media::decode_rgb(&media::load_image_bytes(&source)?)?;
            let input = prepare_image(&frame, cfg, options.image_max_pixels)?;
            let ph = "<|image|>".repeat(input.tokens());
            (input, ph)
        };
        let rows = tower.forward(&input)?;
        if let Some(dir) = &dump {
            let write = |name: String, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_LFM3_VL_DUMP: {e}"))
            };
            write(format!("pixels_{k}.f32"), &input.rows)?;
            write(format!("rows_{k}.f32"), &rows)?;
            eprintln!(
                "lfm3-vl: item {k} resized to {}x{}, grid {}x{}x{}, {} tokens, tower {:.2} s",
                input.resized.0,
                input.resized.1,
                input.grid_t,
                input.grid_h,
                input.grid_w,
                input.tokens(),
                t0.elapsed().as_secs_f64()
            );
        }
        expanded.push_str(&placeholder);
        media_rows.extend(rows);
        k += 1;
    }
    expanded.push_str(rest);
    if images.next().is_some() || videos.next().is_some() || clips.next().is_some() {
        return Err("LFM3-VL: the template rendered fewer placeholders than media".into());
    }
    let ids = tok.with_bos(tok.encode(&expanded));
    let h = pipeline.hidden_size;
    let (image_id, video_id) = (cfg.image_token_id, cfg.video_token_id);
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut at = 0usize;
    for &id in &ids {
        if id == image_id || id == video_id || (Some(id) == audio_id && audio_enc.is_some()) {
            let row = media_rows
                .get(at..at + h)
                .ok_or("LFM3-VL: placeholder and media row counts differ")?;
            rows.extend_from_slice(row);
            at += h;
        } else {
            rows.extend(pipeline.embed_id(id));
        }
    }
    if at != media_rows.len() {
        return Err("LFM3-VL: unused media rows".into());
    }
    Ok(PreparedVlPrompt { token_ids: ids, rows: Some(rows) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The processor's resize arithmetic, with Python's banker's rounding.
    #[test]
    fn smart_resize_matches_the_processor() {
        // 640x480 is already a multiple of 32 and inside the bounds.
        assert_eq!(smart_resize(480, 640, 32, 65536, 16_777_216).unwrap(), (480, 640));
        // 100x100 is below min_pixels: scaled up to cover 65536.
        assert_eq!(smart_resize(100, 100, 32, 65536, 16_777_216).unwrap(), (256, 256));
        assert_eq!(smart_resize(48 * 8, 80 * 8, 32, 0, usize::MAX).unwrap(), (384, 640));
        // 48/32 = 1.5 rounds to 2 (even), 80/32 = 2.5 rounds to 2 (even).
        assert_eq!(smart_resize(48, 80, 32, 0, usize::MAX).unwrap(), (64, 64));
        assert!(smart_resize(1, 300, 32, 0, usize::MAX).is_err());
    }

    /// `sample_frames` with fps 2: 16 frames at 8 fps → 4 frames.
    #[test]
    fn video_sampling_matches_the_processor() {
        let cfg = Lfm3VlConfig {
            hidden: 1024, heads: 16, kv_heads: 8, head_dim: 64, intermediate: 2560,
            attention: vec![], conv_kernel: 3, patch_size: 16, temporal_patch_size: 2,
            merge_size: 2, norm_eps: 1e-5, rope_theta: 1e6, projector_hidden: 4096,
            projector_eps: 1e-6, text_hidden: 2048, image_token_id: 0, video_token_id: 1,
            min_pixels: 65536, max_pixels: 16_777_216, mean: [0.5; 3], std: [0.5; 3],
            rescale: 1.0 / 255.0, video_fps: 2.0, video_min_frames: 4, video_max_frames: 768,
            video_min_pixels: 4096, video_max_pixels: 25_165_824,
        };
        assert_eq!(sample_indices(16, 8.0, &cfg).unwrap(), vec![0, 5, 10, 15]);
        // 100 frames at 25 fps = 4 s → 8 frames; linspace(0, 99, 8) rounds
        // 14.142857 → 14, 28.285714 → 28, ...
        assert_eq!(
            sample_indices(100, 25.0, &cfg).unwrap(),
            vec![0, 14, 28, 42, 57, 71, 85, 99]
        );
        // 4 frames at 640x480: well inside the all-frames budget.
        assert_eq!(video_smart_resize(4, 480, 640, 2, 32, 4096, 25_165_824).unwrap(), (480, 640));
    }

    #[test]
    fn image_parts_become_template_images_in_order() {
        let msgs = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "x.png"}},
            {"type": "image", "image": "y.png"}]})];
        let (m, items) = normalize(&msgs).unwrap();
        assert_eq!(
            items,
            vec![Item::Image(json!({"url": "x.png"})), Item::Image(json!({"url": "y.png"}))]
        );
        assert_eq!(m[0]["content"][1], json!({"type": "image"}));
        let video = vec![json!({"role": "user", "content": [
            {"type": "video", "video": {"path": "frames"}, "fps": 8.0}]})];
        let (m, items) = normalize(&video).unwrap();
        assert_eq!(items, vec![Item::Video(json!({"path": "frames", "fps": 8.0}))]);
        assert_eq!(m[0]["content"][0], json!({"type": "video"}));
        // Speech parts pass through (LFM3-Omni's prepare collects them; a
        // vision-only file refuses them there); other kinds are refused.
        let audio = vec![json!({"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]})];
        let (_, items) = normalize(&audio).unwrap();
        assert!(items.is_empty());
        let other = vec![json!({"role": "user", "content": [{"type": "file", "file": {}}]})];
        assert!(normalize(&other).is_err());
    }
}
