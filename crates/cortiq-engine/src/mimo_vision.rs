//! MiMo-V2.6 vision: image and video preprocessing, prompt-id expansion for
//! the image/video placeholders, and the MiMo ViT (`visual.*`).
//!
//! Reference semantics are the upstream processor (`MiMoProcessor`) and the
//! HF `MiMoVisionTransformer`, with the merger taken as vLLM/sglang build it
//! (RMSNorm, no biases — the checkpoint carries only `ln_q.weight` and the
//! two merger matrices). Conventions that are easy to get wrong:
//!
//! * **Resize is `F.interpolate(bilinear, align_corners=False)` on 0..255**,
//!   with no antialias and no rounding back to 8 bit, then ImageNet
//!   mean/std *in 0..255 units*. `smart_resize` uses factor 32 and has an
//!   upscale branch for a side below 32 that skips the aspect check.
//! * **Every image is two identical frames** (T = 2), and each patch row is
//!   the Conv3d kernel flattened as (c, tt, py, px).
//! * **Rows are in merge-block order** (t, block row, block col, mh, mw).
//!   Window blocks of type 1 run in *column* order: whole merge units are
//!   listed by (t, block col, block row), the RoPE tables are permuted the
//!   same way, and the band |i−j| ≤ 64 is taken over chunk-local indices in
//!   the CURRENT order. Frames are independent chunks in every block.
//! * **Sinks add to key 0's logit** (HF, vLLM `sinks_bias_key0`), they are
//!   not an extra softmax column like the text model's sinks. Beyond query
//!   64 key 0 is masked and the sink does nothing. `CMF_MIMO_VIT_SINK`
//!   (`key0` | `column` | `off`) switches it for A/B only.
//! * **GQA 32/8, head_dim 64 (`qk_channels`), not hidden/heads = 40.**
//!
//! Weights are read by source name from any CMF that carries them — the
//! `<stem>.mm.cmf` companion or a single-file multimodal CMF — through
//! [`MimoVit::from_model`]. Dense (F32/F16/BF16) matrices become exact f32
//! GEMM operands; quantized ones (q4tp, q8_2f) stay mapped on the engine's
//! kernels, CPU or GPU.

use crate::dit::Proj;
use crate::media::RgbFrame;
use crate::pool::Pool;
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const VISION_START_ID: u32 = 151652;
pub const VISION_END_ID: u32 = 151653;
pub const IMAGE_PAD_ID: u32 = 151655;
pub const VIDEO_PAD_ID: u32 = 151656;
pub const AUDIO_PAD_ID: u32 = 151669;
pub const VIDEO_START_ID: u32 = 151670;
pub const VIDEO_END_ID: u32 = 151671;
pub const AUDIO_START_ID: u32 = 151673;
pub const AUDIO_END_ID: u32 = 151674;

/// Name of the U8 blob that carries the checkpoint's full `config.json`.
pub const MM_CONFIG_TENSOR: &str = "mm.config_json";

const PATCH_EMBED: &str = "visual.patch_embed.proj.weight";

static GPU_ATTENTION_DISPATCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Full-attention frame chunks this process ran through `gpu::dit_attention`
/// (so a timing on a live backend cannot be mistaken for the host path).
pub fn gpu_attention_dispatches() -> usize {
    GPU_ATTENTION_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

// ─────────────────────────────── processor ───────────────────────────────

/// The processor knobs, read from `config.json` (`processor_config` plus the
/// vision geometry). Missing fields take the upstream processor's defaults.
#[derive(Clone, Debug)]
pub struct MimoProcessorConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
    pub video_total_max_pixels: usize,
    /// Sampling rate of video frames, frames per second of source time.
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for MimoProcessorConfig {
    /// The values pinned by MiMo-V2.6-Flash's `config.json`.
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            image_min_pixels: 8192,
            image_max_pixels: 8_388_608,
            video_min_pixels: 8192,
            video_max_pixels: 8_388_608,
            video_total_max_pixels: 268_435_456,
            fps: 1.0,
            min_frames: 8,
            max_frames: 3600,
            mean: [123.675, 116.28, 103.53],
            std: [58.395, 57.12, 57.375],
        }
    }
}

fn get_usize(v: Option<&Value>, key: &str) -> Option<usize> {
    v?.get(key)?.as_u64().map(|n| n as usize)
}

impl MimoProcessorConfig {
    /// Build from the full `config.json` value.
    pub fn from_config(cfg: &Value) -> Result<Self, String> {
        let pc = cfg.get("processor_config");
        let vc = cfg.get("vision_config");
        let patch_size = get_usize(pc, "patch_size")
            .or_else(|| get_usize(vc, "patch_size"))
            .unwrap_or(16);
        let merge_size = get_usize(pc, "merge_size")
            .or_else(|| get_usize(vc, "spatial_merge_size"))
            .unwrap_or(2);
        let temporal_patch_size = get_usize(pc, "temporal_patch_size")
            .or_else(|| get_usize(vc, "temporal_patch_size"))
            .unwrap_or(2);
        if let Some(r) = get_usize(pc, "temporal_compression_ratio") {
            if r != 1 {
                return Err(format!(
                    "temporal_compression_ratio {r} is not supported (MiMo pins 1)"
                ));
            }
        }
        let unit = patch_size * merge_size;
        // `x or default` in the upstream processor: a null/0 takes the default.
        let nz = |k: &str, d: usize| get_usize(pc, k).filter(|&v| v > 0).unwrap_or(d);
        let fps = pc
            .and_then(|p| p.get("fps"))
            .and_then(Value::as_f64)
            .filter(|&v| v > 0.0)
            .unwrap_or(2.0);
        let out = Self {
            patch_size,
            merge_size,
            temporal_patch_size,
            image_min_pixels: nz("image_min_pixels", 4 * unit * unit),
            image_max_pixels: nz("image_max_pixels", 4096 * unit * unit),
            video_min_pixels: nz("video_min_pixels", 4 * unit * unit),
            video_max_pixels: nz("video_max_pixels", 4096 * unit * unit),
            video_total_max_pixels: nz("video_total_max_pixels", 16384 * unit * unit),
            fps,
            min_frames: nz("min_frames", 8),
            max_frames: nz("max_frames", 256),
            ..Self::default()
        };
        if out.patch_size == 0 || out.merge_size == 0 || out.temporal_patch_size == 0 {
            return Err("processor patch/merge/temporal sizes must be positive".into());
        }
        Ok(out)
    }

    /// `patch_size · merge_size`: every resized side is a multiple of it.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// Python's `round` (ties to even) for the non-negative values used here.
fn py_round(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if f % 2.0 == 0.0 {
        f
    } else {
        f + 1.0
    }
}

/// MiMo's `smart_resize` (processor `MiMoProcessor.smart_resize`), returning
/// `(height, width)`. Note the upscale branch: when the short side is below
/// `factor` both sides are scaled up first and the aspect check is skipped.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(format!(
            "smart_resize needs positive sizes, got {height}x{width} factor {factor}"
        ));
    }
    let (mut h, mut w) = (height, width);
    let short = h.min(w);
    if short < factor {
        let scale = factor as f64 / short as f64;
        h = py_round(h as f64 * scale) as usize;
        w = py_round(w as f64 * scale) as usize;
    } else {
        let aspect = h.max(w) as f64 / h.min(w) as f64;
        if aspect > 200.0 {
            return Err(format!(
                "absolute aspect ratio must be smaller than 200, got {aspect}"
            ));
        }
    }
    let f = factor as f64;
    let mut hb = py_round(h as f64 / f) as usize * factor;
    let mut wb = py_round(w as f64 / f) as usize * factor;
    let area = (h as f64) * (w as f64);
    if hb * wb > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        hb = (h as f64 / beta / f).floor() as usize * factor;
        wb = (w as f64 / beta / f).floor() as usize * factor;
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / area).sqrt();
        hb = (h as f64 * beta / f).ceil() as usize * factor;
        wb = (w as f64 * beta / f).ceil() as usize * factor;
    }
    if hb == 0 || wb == 0 {
        return Err(format!(
            "smart_resize of {height}x{width} collapsed to {hb}x{wb} (pixel bounds {min_pixels}..{max_pixels})"
        ));
    }
    Ok((hb, wb))
}

/// Per-output-index taps of torch's CPU `upsample_bilinear2d` along one axis
/// (`align_corners=False`, scale from sizes, float opmath): `(i0, i1, w0, w1)`.
fn linear_taps(in_size: usize, out_size: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale = in_size as f32 / out_size as f32;
    (0..out_size)
        .map(|i| {
            let real = scale * (i as f32 + 0.5) - 0.5;
            let real = if real < 0.0 { 0.0 } else { real };
            let i0 = (real.floor() as usize).min(in_size - 1);
            let lambda = (real - i0 as f32).clamp(0.0, 1.0);
            let i1 = if i0 < in_size - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - lambda, lambda)
        })
        .collect()
}

/// `F.interpolate(x, size=(out_h,out_w), mode="bilinear",
/// align_corners=False)` on a planar `[c][h][w]` f32 image, no antialias.
/// Taps nest as torch does: interpolate along W on the two source rows,
/// then along H.
pub fn resize_bilinear(
    src: &[f32],
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    assert_eq!(
        src.len(),
        channels * in_h * in_w,
        "resize_bilinear: bad input size"
    );
    let ty = linear_taps(in_h, out_h);
    let tx = linear_taps(in_w, out_w);
    let mut out = vec![0f32; channels * out_h * out_w];
    for c in 0..channels {
        let plane = &src[c * in_h * in_w..(c + 1) * in_h * in_w];
        let dst = &mut out[c * out_h * out_w..(c + 1) * out_h * out_w];
        for (y, &(y0, y1, wy0, wy1)) in ty.iter().enumerate() {
            let r0 = &plane[y0 * in_w..(y0 + 1) * in_w];
            let r1 = &plane[y1 * in_w..(y1 + 1) * in_w];
            let row = &mut dst[y * out_w..(y + 1) * out_w];
            for (x, &(x0, x1, wx0, wx1)) in tx.iter().enumerate() {
                let t0 = r0[x0] * wx0 + r0[x1] * wx1;
                let t1 = r1[x0] * wx0 + r1[x1] * wx1;
                row[x] = t0 * wy0 + t1 * wy1;
            }
        }
    }
    out
}

/// HWC u8 → planar CHW f32 in 0..255.
fn frame_to_chw(frame: &RgbFrame) -> Vec<f32> {
    let n = frame.width * frame.height;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in frame.data.chunks_exact(3).enumerate() {
        out[i] = px[0] as f32;
        out[n + i] = px[1] as f32;
        out[2 * n + i] = px[2] as f32;
    }
    out
}

/// Resize one frame to `(hb, wb)` and standardize it: `(x − mean) / std`.
fn resize_normalize(frame: &RgbFrame, hb: usize, wb: usize, cfg: &MimoProcessorConfig) -> Vec<f32> {
    let chw = frame_to_chw(frame);
    let mut r = resize_bilinear(&chw, 3, frame.height, frame.width, hb, wb);
    let plane = hb * wb;
    for c in 0..3 {
        let (m, s) = (cfg.mean[c], cfg.std[c]);
        for v in &mut r[c * plane..(c + 1) * plane] {
            *v = (*v - m) / s;
        }
    }
    r
}

/// What a visual item is; it decides the placeholder and its expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualKind {
    Image,
    Video,
}

/// One preprocessed image or video: the patch rows fed to the ViT, the
/// patch grid, and (video) the per-frame timestamps in seconds.
#[derive(Clone, Debug)]
pub struct VisualInput {
    pub kind: VisualKind,
    /// `[grid_t · grid_h · grid_w, 3 · T · P · P]`, merge-block row order.
    pub rows: Vec<f32>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
    pub merge_size: usize,
    /// Video: one timestamp per frame after even padding (`2 · grid_t`).
    pub timestamps: Vec<f32>,
    /// Video: the source frame index of each entry of `timestamps`.
    pub frame_indices: Vec<usize>,
    /// The resized frame size `(height, width)`.
    pub resized: (usize, usize),
}

impl VisualInput {
    /// ViT rows (patches).
    pub fn patches(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }

    /// LLM placeholder tokens for one temporal step.
    pub fn tokens_per_step(&self) -> usize {
        self.grid_h * self.grid_w / (self.merge_size * self.merge_size)
    }

    /// LLM placeholder tokens for the whole item.
    pub fn tokens(&self) -> usize {
        self.grid_t * self.tokens_per_step()
    }

    /// The "MM:SS" label of every temporal step (video only): the timestamp
    /// of the first frame of each pair.
    pub fn timestamp_labels(&self) -> Vec<String> {
        (0..self.grid_t)
            .filter_map(|t| self.timestamps.get(2 * t).map(|&ts| format_timestamp(ts)))
            .collect()
    }
}

/// Patchify `frames` (each planar CHW at `hb × wb`, count a multiple of T)
/// exactly as `view(gt,T,C,gh/m,m,P,gw/m,m,P).permute(0,3,6,4,7,2,1,5,8)`.
fn patchify(
    frames: &[&[f32]],
    hb: usize,
    wb: usize,
    cfg: &MimoProcessorConfig,
) -> Result<(Vec<f32>, usize, usize, usize), String> {
    let (p, m, tp) = (cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size);
    if frames.is_empty() || frames.len() % tp != 0 {
        return Err(format!(
            "{} frames is not a positive multiple of temporal_patch_size {tp}",
            frames.len()
        ));
    }
    if hb % (p * m) != 0 || wb % (p * m) != 0 {
        return Err(format!("frame {hb}x{wb} is not a multiple of {}", p * m));
    }
    let (gt, gh, gw) = (frames.len() / tp, hb / p, wb / p);
    let dim = cfg.patch_dim();
    let mut rows = Vec::with_capacity(gt * gh * gw * dim);
    let plane = hb * wb;
    for t in 0..gt {
        for a in 0..gh / m {
            for b in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let y0 = (a * m + mh) * p;
                        let x0 = (b * m + mw) * p;
                        for c in 0..3 {
                            for tt in 0..tp {
                                let f = frames[t * tp + tt];
                                for py in 0..p {
                                    let base = c * plane + (y0 + py) * wb + x0;
                                    rows.extend_from_slice(&f[base..base + p]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((rows, gt, gh, gw))
}

/// Preprocess one image: smart_resize (factor 32), bilinear resize on
/// 0..255, standardize, duplicate to T frames, patchify. `max_pixels`
/// overrides `image_max_pixels` (the `--image-max-pixels` knob).
pub fn prepare_image(
    frame: &RgbFrame,
    cfg: &MimoProcessorConfig,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    let max_px = max_pixels.unwrap_or(cfg.image_max_pixels);
    let (hb, wb) = smart_resize(
        frame.height,
        frame.width,
        cfg.factor(),
        cfg.image_min_pixels,
        max_px,
    )?;
    let norm = resize_normalize(frame, hb, wb, cfg);
    let frames: Vec<&[f32]> = (0..cfg.temporal_patch_size)
        .map(|_| norm.as_slice())
        .collect();
    let (rows, grid_t, grid_h, grid_w) = patchify(&frames, hb, wb, cfg)?;
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

// ───────────────────────────────── video ─────────────────────────────────

/// Frame count for `total` source frames at `video_fps` (sglang
/// `smart_nframes` with the processor defaults): `total / fps_src · fps`,
/// clamped to `[min_frames, max_frames]` and `total`, floored to even.
pub fn smart_nframes(
    total: usize,
    video_fps: f64,
    cfg: &MimoProcessorConfig,
) -> Result<usize, String> {
    const FRAME_FACTOR: f64 = 2.0;
    if !(video_fps > 0.0) || !video_fps.is_finite() {
        return Err(format!("video fps must be positive, got {video_fps}"));
    }
    let min_frames = (cfg.min_frames as f64 / FRAME_FACTOR).ceil() * FRAME_FACTOR;
    let max_frames = (cfg.max_frames as f64 / FRAME_FACTOR).floor() * FRAME_FACTOR;
    let n = total as f64 / video_fps * cfg.fps;
    let n = n.max(min_frames).min(max_frames).min(total as f64);
    let n = (n / FRAME_FACTOR).floor() * FRAME_FACTOR;
    if !(FRAME_FACTOR <= n && n <= total as f64) {
        return Err(format!(
            "nframes should in interval [2, {total}], but got {n} (video has {total} frames)"
        ));
    }
    Ok(n as usize)
}

/// The sampled frame indices and their timestamps:
/// `unique(int64(linspace(0, total−1, n)))` and `float32(idx) / fps_src`.
pub fn sample_frames(
    total: usize,
    video_fps: f64,
    cfg: &MimoProcessorConfig,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let n = smart_nframes(total, video_fps, cfg)?;
    let stop = (total - 1) as f64;
    let step = stop / (n - 1) as f64;
    let mut idx: Vec<usize> = (0..n)
        .map(|i| {
            if i == n - 1 {
                stop
            } else {
                (i as f64 * step).floor()
            }
        })
        .map(|v| v as usize)
        .collect();
    idx.dedup();
    let fps32 = video_fps as f32;
    let ts = idx.iter().map(|&i| i as f32 / fps32).collect();
    Ok((idx, ts))
}

/// Per-frame pixel ceiling for `n_sampled` frames:
/// `max(min_pixels, min(total_max · T // n, max_pixels))`.
pub fn video_max_pixels(n_sampled: usize, cfg: &MimoProcessorConfig) -> usize {
    let per_frame = cfg.video_total_max_pixels * cfg.temporal_patch_size / n_sampled.max(1);
    cfg.video_min_pixels
        .max(per_frame.min(cfg.video_max_pixels))
}

/// `f"{int(ts // 60):02d}:{int(ts % 60):02d}"` on the float32 timestamp,
/// with torch's float floor-division/remainder. Minutes are not wrapped.
pub fn format_timestamp(ts: f32) -> String {
    let rem = ts % 60.0;
    let minutes = ((ts - rem) / 60.0).floor();
    format!("{:02}:{:02}", minutes as i64, rem as i64)
}

/// The Y4M stream layout needed to seek to a frame.
#[derive(Clone, Debug)]
pub struct Y4mInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    chroma: Chroma,
    /// Byte offset of every frame's payload.
    offsets: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chroma {
    C420,
    C422,
    C444,
    Mono,
}

impl Chroma {
    fn plane_dims(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Chroma::C420 => (w.div_ceil(2), h.div_ceil(2)),
            Chroma::C422 => (w.div_ceil(2), h),
            Chroma::C444 => (w, h),
            Chroma::Mono => (0, 0),
        }
    }
}

/// A silent video given as decoded frames: a directory of images with an
/// explicit frame rate, or a Y4M stream (`ffmpeg -i in.mp4 -pix_fmt yuv420p
/// out.y4m`). mp4 decoding is out of scope for v1.
#[derive(Clone, Debug)]
pub enum VideoSource {
    FrameDir { frames: Vec<PathBuf>, fps: f64 },
    Y4m { path: PathBuf, info: Y4mInfo },
}

const FRAME_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "ppm"];

/// Natural-order key: digit runs compare numerically, so `f2` < `f10`.
fn natural_key(s: &str) -> Vec<(u8, u128, String)> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        let mut run = String::new();
        if c.is_ascii_digit() {
            while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                run.push(d);
                chars.next();
            }
            let v = run.parse::<u128>().unwrap_or(u128::MAX);
            out.push((0, v, run));
        } else {
            while let Some(&d) = chars.peek().filter(|d| !d.is_ascii_digit()) {
                run.push(d);
                chars.next();
            }
            out.push((1, 0, run));
        }
    }
    out
}

impl VideoSource {
    /// Every image file of `dir` (png/jpg/jpeg/webp/gif/ppm), in natural
    /// name order, played at `fps`.
    pub fn frame_dir(dir: &Path, fps: f64) -> Result<Self, String> {
        if !(fps > 0.0) || !fps.is_finite() {
            return Err(format!(
                "frame directory needs a positive --video-fps, got {fps}"
            ));
        }
        let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| FRAME_EXTS.contains(&e.to_ascii_lowercase().as_str()))
            })
            .collect();
        frames.sort_by_cached_key(|p| {
            natural_key(&p.file_name().unwrap_or_default().to_string_lossy())
        });
        if frames.is_empty() {
            return Err(format!("{}: no image frames found", dir.display()));
        }
        Ok(VideoSource::FrameDir { frames, fps })
    }

    /// Index a Y4M file (8-bit 4:2:0 / 4:2:2 / 4:4:4 / mono).
    pub fn y4m(path: &Path) -> Result<Self, String> {
        use std::io::{BufRead, BufReader, Seek, SeekFrom};
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut rd = BufReader::new(file);
        let mut line = Vec::new();
        rd.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        let header = std::str::from_utf8(&line)
            .map_err(|_| "Y4M header is not text".to_string())?
            .trim_end();
        let mut parts = header.split(' ');
        if parts.next() != Some("YUV4MPEG2") {
            return Err(format!("{}: not a YUV4MPEG2 stream", path.display()));
        }
        let (mut w, mut h, mut fps, mut chroma) = (0usize, 0usize, 0f64, Chroma::C420);
        for p in parts {
            let (tag, val) = p.split_at(1.min(p.len()));
            match tag {
                "W" => w = val.parse().map_err(|_| format!("bad Y4M width '{val}'"))?,
                "H" => h = val.parse().map_err(|_| format!("bad Y4M height '{val}'"))?,
                "F" => {
                    let (n, d) = val
                        .split_once(':')
                        .ok_or_else(|| format!("bad Y4M rate '{val}'"))?;
                    let n: f64 = n.parse().map_err(|_| format!("bad Y4M rate '{val}'"))?;
                    let d: f64 = d.parse().map_err(|_| format!("bad Y4M rate '{val}'"))?;
                    fps = n / d;
                }
                "C" => {
                    chroma = match val {
                        "420" | "420jpeg" | "420paldv" | "420mpeg2" => Chroma::C420,
                        "422" => Chroma::C422,
                        "444" => Chroma::C444,
                        "mono" => Chroma::Mono,
                        other => {
                            return Err(format!(
                                "Y4M colorspace C{other} is not supported (8-bit 420/422/444/mono)"
                            ));
                        }
                    }
                }
                "I" => {
                    if val != "p" && val != "?" {
                        return Err(format!("interlaced Y4M (I{val}) is not supported"));
                    }
                }
                _ => {}
            }
        }
        if w == 0 || h == 0 || !(fps > 0.0) || !fps.is_finite() {
            return Err(format!("{}: Y4M header lacks W/H/F", path.display()));
        }
        let (cw, ch) = chroma.plane_dims(w, h);
        let payload = (w * h + 2 * cw * ch) as u64;
        let mut offsets = Vec::new();
        let mut pos = line.len() as u64;
        loop {
            line.clear();
            let got = rd.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
            if got == 0 {
                break;
            }
            if !line.starts_with(b"FRAME") {
                return Err(format!(
                    "{}: bad frame marker at byte {pos}",
                    path.display()
                ));
            }
            pos += got as u64;
            if pos + payload > file_len {
                return Err(format!(
                    "{}: truncated frame {}",
                    path.display(),
                    offsets.len()
                ));
            }
            offsets.push(pos);
            pos += payload;
            rd.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
        }
        if offsets.is_empty() {
            return Err(format!("{}: Y4M stream has no frames", path.display()));
        }
        Ok(VideoSource::Y4m {
            path: path.to_path_buf(),
            info: Y4mInfo {
                width: w,
                height: h,
                fps,
                chroma,
                offsets,
            },
        })
    }

    /// A directory (needs `fps`) or a `.y4m` file (its own rate; an explicit
    /// `fps` overrides it).
    pub fn open(path: &Path, fps: Option<f64>) -> Result<Self, String> {
        if path.is_dir() {
            let fps = fps.ok_or_else(|| {
                format!("{}: a frame directory needs --video-fps", path.display())
            })?;
            return Self::frame_dir(path, fps);
        }
        let mut src = Self::y4m(path)?;
        if let (Some(f), VideoSource::Y4m { info, .. }) = (fps, &mut src) {
            if !(f > 0.0) {
                return Err(format!("--video-fps must be positive, got {f}"));
            }
            info.fps = f;
        }
        Ok(src)
    }

    pub fn frame_count(&self) -> usize {
        match self {
            VideoSource::FrameDir { frames, .. } => frames.len(),
            VideoSource::Y4m { info, .. } => info.offsets.len(),
        }
    }

    pub fn fps(&self) -> f64 {
        match self {
            VideoSource::FrameDir { fps, .. } => *fps,
            VideoSource::Y4m { info, .. } => info.fps,
        }
    }

    /// Decode frame `idx` to RGB.
    pub fn read_frame(&self, idx: usize) -> Result<RgbFrame, String> {
        match self {
            VideoSource::FrameDir { frames, .. } => {
                let p = frames
                    .get(idx)
                    .ok_or_else(|| format!("frame {idx} out of range"))?;
                crate::media::read_rgb(p)
            }
            VideoSource::Y4m { path, info } => read_y4m_frame(path, info, idx),
        }
    }
}

/// BT.601 limited-range YUV → RGB with nearest-neighbour chroma — the
/// swscale default for untagged 8-bit video. (The upstream processor gets
/// frames from its own decoder; Y4M has no reference path to match.)
fn read_y4m_frame(path: &Path, info: &Y4mInfo, idx: usize) -> Result<RgbFrame, String> {
    use std::io::{Read, Seek, SeekFrom};
    let off = *info
        .offsets
        .get(idx)
        .ok_or_else(|| format!("Y4M frame {idx} out of range"))?;
    let (w, h) = (info.width, info.height);
    let (cw, ch) = info.chroma.plane_dims(w, h);
    let mut buf = vec![0u8; w * h + 2 * cw * ch];
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let (yp, rest) = buf.split_at(w * h);
    let (up, vp) = rest.split_at(cw * ch);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let yy = yp[y * w + x] as f32 - 16.0;
            let (u, v) = match info.chroma {
                Chroma::Mono => (0.0, 0.0),
                Chroma::C444 => (up[y * cw + x] as f32 - 128.0, vp[y * cw + x] as f32 - 128.0),
                Chroma::C422 => (
                    up[y * cw + x / 2] as f32 - 128.0,
                    vp[y * cw + x / 2] as f32 - 128.0,
                ),
                Chroma::C420 => (
                    up[(y / 2) * cw + x / 2] as f32 - 128.0,
                    vp[(y / 2) * cw + x / 2] as f32 - 128.0,
                ),
            };
            let r = 1.164_383 * yy + 1.596_027 * v;
            let g = 1.164_383 * yy - 0.391_762 * u - 0.812_968 * v;
            let b = 1.164_383 * yy + 2.017_232 * u;
            let o = (y * w + x) * 3;
            rgb[o] = r.round().clamp(0.0, 255.0) as u8;
            rgb[o + 1] = g.round().clamp(0.0, 255.0) as u8;
            rgb[o + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }
    RgbFrame::new(w, h, rgb)
}

/// Preprocess already-sampled frames with their timestamps: per-frame pixel
/// budget from the sampled count, even padding (repeat the last frame and
/// its timestamp), one smart_resize for all frames, standardize, and pair
/// consecutive frames as temporal patches.
pub fn prepare_video_frames(
    frames: &[RgbFrame],
    timestamps: &[f32],
    frame_indices: &[usize],
    cfg: &MimoProcessorConfig,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    if frames.is_empty() || frames.len() != timestamps.len() {
        return Err(format!(
            "video has {} frames but {} timestamps",
            frames.len(),
            timestamps.len()
        ));
    }
    let (h, w) = (frames[0].height, frames[0].width);
    if let Some(bad) = frames.iter().position(|f| f.height != h || f.width != w) {
        return Err(format!(
            "video frame {bad} is {}x{}, frame 0 is {w}x{h}: all frames must share one size",
            frames[bad].width, frames[bad].height
        ));
    }
    let n = frames.len();
    let budget = video_max_pixels(n, cfg);
    let max_px = max_pixels.map_or(budget, |m| m.min(budget).max(cfg.video_min_pixels));
    let (hb, wb) = smart_resize(h, w, cfg.factor(), cfg.video_min_pixels, max_px)?;
    let tp = cfg.temporal_patch_size;
    let padded = n.div_ceil(tp) * tp;
    let mut ts = timestamps.to_vec();
    let mut fi = frame_indices.to_vec();
    let mut normed: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| resize_normalize(f, hb, wb, cfg))
        .collect();
    while normed.len() < padded {
        normed.push(normed[n - 1].clone());
        ts.push(timestamps[n - 1]);
        if let Some(&last) = frame_indices.last() {
            fi.push(last);
        }
    }
    let refs: Vec<&[f32]> = normed.iter().map(|v| v.as_slice()).collect();
    let (rows, grid_t, grid_h, grid_w) = patchify(&refs, hb, wb, cfg)?;
    Ok(VisualInput {
        kind: VisualKind::Video,
        rows,
        grid_t,
        grid_h,
        grid_w,
        patch_dim: cfg.patch_dim(),
        merge_size: cfg.merge_size,
        timestamps: ts,
        frame_indices: fi,
        resized: (hb, wb),
    })
}

/// Sample, decode and preprocess a video source.
pub fn prepare_video(
    src: &VideoSource,
    cfg: &MimoProcessorConfig,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    let (idx, ts) = sample_frames(src.frame_count(), src.fps(), cfg)?;
    let frames = idx
        .iter()
        .map(|&i| src.read_frame(i))
        .collect::<Result<Vec<_>, _>>()?;
    prepare_video_frames(&frames, &ts, &idx, cfg, max_pixels)
}

// ─────────────────────────────── prompt ids ──────────────────────────────

/// Expand the rendered prompt's media placeholders.
///
/// The chat template renders one `<|vision_start|><|image_pad|><|vision_end|>`
/// per image part, `…<|video_pad|>…` per video part and
/// `<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>` per audio part.
/// Like the upstream regex (`(?:pad)+`) a run of pads between the markers
/// is one placeholder. Expansion, in order within each modality:
/// * image → `[vs] + N×[image_pad] + [ve]`, N = the item's tokens;
/// * video → `[video_start] + Σₜ(encode("MM:SS") + [vs] + n×[video_pad] +
///   [ve]) + [video_end]`, the WHOLE triple replaced;
/// * audio → `[as] + K×[audio_pad] + [ae]`, K from `audio_tokens`.
///
/// A placeholder/item count mismatch, or a pad token outside a
/// placeholder, is an error.
pub fn expand_prompt_ids(
    ids: &[u32],
    images: &[&VisualInput],
    videos: &[&VisualInput],
    audio_tokens: &[usize],
    tok: &Tokenizer,
) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len());
    let (mut ni, mut nv, mut na) = (0usize, 0usize, 0usize);
    let mut p = 0usize;
    // Length of a `start pad+ end` run at `p`, if one starts there.
    let run = |p: usize, start: u32, pad: u32, end: u32| -> Option<usize> {
        if ids.get(p) != Some(&start) || ids.get(p + 1) != Some(&pad) {
            return None;
        }
        let mut q = p + 1;
        while ids.get(q) == Some(&pad) {
            q += 1;
        }
        (ids.get(q) == Some(&end)).then_some(q + 1 - p)
    };
    while p < ids.len() {
        if let Some(len) = run(p, VISION_START_ID, IMAGE_PAD_ID, VISION_END_ID) {
            let item = images.get(ni).ok_or_else(|| {
                format!(
                    "prompt has more image placeholders than the {} images given",
                    images.len()
                )
            })?;
            ni += 1;
            out.push(VISION_START_ID);
            out.extend(std::iter::repeat_n(IMAGE_PAD_ID, item.tokens()));
            out.push(VISION_END_ID);
            p += len;
        } else if let Some(len) = run(p, VISION_START_ID, VIDEO_PAD_ID, VISION_END_ID) {
            let item = videos.get(nv).ok_or_else(|| {
                format!(
                    "prompt has more video placeholders than the {} videos given",
                    videos.len()
                )
            })?;
            nv += 1;
            let labels = item.timestamp_labels();
            if labels.len() != item.grid_t {
                return Err(format!(
                    "video has {} timestamps for {} temporal steps",
                    item.timestamps.len(),
                    item.grid_t
                ));
            }
            out.push(VIDEO_START_ID);
            for label in &labels {
                out.extend(tok.encode(label));
                out.push(VISION_START_ID);
                out.extend(std::iter::repeat_n(VIDEO_PAD_ID, item.tokens_per_step()));
                out.push(VISION_END_ID);
            }
            out.push(VIDEO_END_ID);
            p += len;
        } else if let Some(len) = run(p, AUDIO_START_ID, AUDIO_PAD_ID, AUDIO_END_ID) {
            let k = *audio_tokens.get(na).ok_or_else(|| {
                format!(
                    "prompt has more audio placeholders than the {} audios given",
                    audio_tokens.len()
                )
            })?;
            na += 1;
            out.push(AUDIO_START_ID);
            out.extend(std::iter::repeat_n(AUDIO_PAD_ID, k));
            out.push(AUDIO_END_ID);
            p += len;
        } else {
            let id = ids[p];
            if id == IMAGE_PAD_ID || id == VIDEO_PAD_ID || id == AUDIO_PAD_ID {
                return Err(format!(
                    "media pad token {id} at position {p} is outside a placeholder"
                ));
            }
            out.push(id);
            p += 1;
        }
    }
    if ni != images.len() || nv != videos.len() || na != audio_tokens.len() {
        return Err(format!(
            "placeholder/data mismatch: prompt has {ni} image, {nv} video, {na} audio \
             placeholders; request has {} images, {} videos, {} audios",
            images.len(),
            videos.len(),
            audio_tokens.len()
        ));
    }
    Ok(out)
}

// ────────────────────────────────── ViT ──────────────────────────────────

/// `vision_config` geometry.
#[derive(Clone, Debug)]
pub struct MimoVisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub in_channels: usize,
    pub out_hidden: usize,
    pub fullatt: Vec<usize>,
    /// `vit_window_attn_types`: 1 = column order, anything else = row order.
    pub window_types: Vec<i64>,
    /// `visual_token_window_size`; `None` = no band (`≤ 0` upstream).
    pub window: Option<usize>,
    pub use_sink: bool,
    pub eps: f64,
    pub rope_theta: f32,
}

impl MimoVisionConfig {
    /// From the full `config.json` (reads `vision_config`) or from the
    /// `vision_config` object itself.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let vc = v.get("vision_config").unwrap_or(v);
        let req = |k: &str| {
            vc.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("vision_config lacks integer '{k}'"))
        };
        let opt = |k: &str, d: usize| vc.get(k).and_then(Value::as_u64).map_or(d, |n| n as usize);
        let depth = req("depth")?;
        let heads = req("num_heads")?;
        let act = vc
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu");
        if act != "silu" {
            return Err(format!("MiMo ViT MLP needs hidden_act 'silu', got '{act}'"));
        }
        let fullatt = match vc.get("fullatt_block_indexes") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or("non-integer fullatt index")
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        let window_types = match vc.get("vit_window_attn_types") {
            Some(Value::Array(a)) if !a.is_empty() => a
                .iter()
                .map(|x| x.as_i64().ok_or("non-integer vit_window_attn_types entry"))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![-1; depth],
        };
        if window_types.len() != depth {
            return Err(format!(
                "vit_window_attn_types has {} entries for depth {depth}",
                window_types.len()
            ));
        }
        let window = vc
            .get("visual_token_window_size")
            .and_then(Value::as_i64)
            .filter(|&w| w > 0)
            .map(|w| w as usize);
        let cfg = Self {
            depth,
            hidden: req("hidden_size")?,
            intermediate: req("intermediate_size")?,
            heads,
            kv_heads: opt("num_key_value_heads", heads),
            head_dim: opt("qk_channels", 64),
            patch_size: req("patch_size")?,
            temporal_patch_size: req("temporal_patch_size")?,
            merge_size: opt("spatial_merge_size", 2),
            in_channels: vc
                .get("in_channels")
                .or_else(|| vc.get("in_chans"))
                .and_then(Value::as_u64)
                .map_or(3, |n| n as usize),
            out_hidden: req("out_hidden_size")?,
            fullatt,
            window_types,
            window,
            use_sink: vc.get("use_sink").and_then(Value::as_bool).unwrap_or(false),
            eps: vc
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6),
            rope_theta: 10_000.0,
        };
        if cfg.kv_heads == 0 || cfg.heads % cfg.kv_heads != 0 {
            return Err(format!(
                "heads {} not divisible by kv heads {}",
                cfg.heads, cfg.kv_heads
            ));
        }
        if cfg.head_dim % 4 != 0 || cfg.head_dim == 0 {
            return Err(format!(
                "head_dim {} must be a positive multiple of 4",
                cfg.head_dim
            ));
        }
        if cfg.in_channels != 3 {
            return Err(format!(
                "in_channels {} (only RGB is supported)",
                cfg.in_channels
            ));
        }
        if cfg.fullatt.iter().any(|&i| i >= depth) {
            return Err("fullatt_block_indexes exceeds depth".into());
        }
        Ok(cfg)
    }
}

/// How the per-head sinks enter the windowed blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkMode {
    /// HF / vLLM: `logit[h, i, key 0] += sinks[h]` (the default).
    Key0,
    /// sglang FA3: an extra softmax column carrying no value.
    Column,
    /// No sinks (A/B only).
    Off,
}

impl SinkMode {
    /// `CMF_MIMO_VIT_SINK` = `key0` (default) | `column` | `off`.
    pub fn from_env() -> Self {
        match std::env::var("CMF_MIMO_VIT_SINK").as_deref() {
            Ok("column") => SinkMode::Column,
            Ok("off") => SinkMode::Off,
            _ => SinkMode::Key0,
        }
    }
}

/// A named projection. The name keys the GPTQ Hessian capture: dense
/// weights run as f32 GEMM operands, which the `QTensor` hook never sees,
/// so the tower folds its own inputs while `gptq_capture` is active.
struct Lin {
    name: String,
    p: Proj,
}

/// Activations above this are scaled down by a power of two before a GEMM.
/// The device GEMMs on matrix units stage their operands as f16 (max 65504),
/// and the last ViT block's down_proj input reaches ~1.1e5 (448² image) —
/// an inf, then NaN rows, on the default NVIDIA path. A power-of-two scale
/// is exact in f32, so the host result is bit-identical either way.
const F16_HEADROOM: f32 = 16384.0;

impl Lin {
    fn mm(&self, x: &[f32], n: usize, out: &mut [f32], pool: Option<&Pool>) {
        if crate::gptq_capture::capturing() && matches!(self.p, Proj::F32 { .. }) {
            crate::gptq_capture::accumulate(&self.name, x, n, self.p.cols());
        }
        let peak = max_abs(x);
        if peak > F16_HEADROOM && peak.is_finite() {
            let k = (peak / F16_HEADROOM).log2().ceil() as i32;
            let (down, up) = (2f32.powi(-k), 2f32.powi(k));
            let xs: Vec<f32> = x.iter().map(|v| v * down).collect();
            self.p.matmat(&xs, n, out, pool);
            for v in out.iter_mut() {
                *v *= up;
            }
        } else {
            self.p.matmat(x, n, out, pool);
        }
    }
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Lin,
    qkv_b: Vec<f32>,
    proj: Lin,
    proj_b: Vec<f32>,
    gate: Lin,
    gate_b: Vec<f32>,
    up: Lin,
    up_b: Vec<f32>,
    down: Lin,
    down_b: Vec<f32>,
    sinks: Option<Vec<f32>>,
}

/// The MiMo vision transformer and its patch merger.
pub struct MimoVit {
    pub cfg: MimoVisionConfig,
    patch: Proj,
    blocks: Vec<Block>,
    ln_q: Vec<f32>,
    mlp0: Lin,
    mlp0_b: Option<Vec<f32>>,
    mlp2: Lin,
    mlp2_b: Option<Vec<f32>>,
    sink_mode: SinkMode,
    pool: Option<Arc<Pool>>,
    /// Use `gpu::dit_attention` for the full blocks when a backend is live.
    gpu_attention: bool,
    /// `CMF_MIMO_VIT_TRACE=1`: per-block activation ranges on stderr.
    trace: bool,
}

fn max_abs(x: &[f32]) -> f32 {
    x.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// Whether `model` carries the vision tower.
pub fn has_vision(model: &CmfModel) -> bool {
    model.tensor(PATCH_EMBED).is_some()
}

/// The checkpoint `config.json` stored in the file as [`MM_CONFIG_TENSOR`].
pub fn read_mm_config(model: &CmfModel) -> Result<Value, String> {
    let bytes = model
        .tensor_bytes(MM_CONFIG_TENSOR)
        .map_err(|_| format!("file lacks the '{MM_CONFIG_TENSOR}' config blob"))?;
    serde_json::from_slice(bytes).map_err(|e| format!("{MM_CONFIG_TENSOR}: {e}"))
}

fn load_vec(model: &CmfModel, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!(
            "tensor '{name}' has {} values, expected {len}",
            v.len()
        ));
    }
    Ok(v)
}

fn load_opt_vec(model: &CmfModel, name: &str, len: usize) -> Result<Option<Vec<f32>>, String> {
    if model.tensor(name).is_none() {
        return Ok(None);
    }
    load_vec(model, name, len).map(Some)
}

fn load_proj(model: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Result<Lin, String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("missing tensor '{name}'"))?;
    if entry.shape.as_slice() != [rows, cols] {
        return Err(format!(
            "tensor '{name}' shape {:?} != expected [{rows}, {cols}]",
            entry.shape
        ));
    }
    Ok(Lin {
        name: name.to_string(),
        p: Proj::from_model(model, name)?,
    })
}

fn rms_norm_rows(x: &[f32], w: &[f32], eps: f64, out: &mut [f32], pool: Option<&Pool>) {
    let d = w.len();
    let rows = x.len() / d;
    let dst = crate::pool::SendMut::new(out.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for r in lo..hi {
            let xr = &x[r * d..(r + 1) * d];
            let ss = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
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

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        for (v, &bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

/// `erf` by the Numerical Recipes rational form (1.2e-7 relative).
fn erf(x: f64) -> f64 {
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

fn gelu_erf(v: f32) -> f32 {
    (0.5 * v as f64 * (1.0 + erf(v as f64 / std::f64::consts::SQRT_2))) as f32
}

/// Row permutation into column order: merge units listed by (t, b, a),
/// each unit's `m²` rows kept together. `perm[dst] = src`.
pub fn column_permutation(grid_t: usize, grid_h: usize, grid_w: usize, merge: usize) -> Vec<usize> {
    let (ua, ub, mm) = (grid_h / merge, grid_w / merge, merge * merge);
    let mut perm = Vec::with_capacity(grid_t * grid_h * grid_w);
    for t in 0..grid_t {
        for b in 0..ub {
            for a in 0..ua {
                let unit = t * ua * ub + a * ub + b;
                perm.extend(unit * mm..(unit + 1) * mm);
            }
        }
    }
    perm
}

fn gather_rows(x: &[f32], perm: &[usize], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (dst, &src) in perm.iter().enumerate() {
        out[dst * d..(dst + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
    }
    out
}

fn scatter_rows(x: &[f32], perm: &[usize], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (src, &dst) in perm.iter().enumerate() {
        out[dst * d..(dst + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
    }
    out
}

impl MimoVit {
    /// Load from a CMF that carries `visual.*` and the `mm.config_json` blob.
    pub fn from_model(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg = read_mm_config(model)?;
        Self::from_model_with_config(model, &cfg)
    }

    /// Load `visual.*` from `model` with an explicit config (`config.json`
    /// or its `vision_config`).
    pub fn from_model_with_config(model: &Arc<CmfModel>, config: &Value) -> Result<Self, String> {
        let c = MimoVisionConfig::from_json(config)?;
        let (hid, inter, hd) = (c.hidden, c.intermediate, c.head_dim);
        let (qd, kvd) = (c.heads * hd, c.kv_heads * hd);
        let patch_dim = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        let pe = model
            .tensor(PATCH_EMBED)
            .ok_or_else(|| format!("missing tensor '{PATCH_EMBED}'"))?;
        let want = [
            hid,
            c.in_channels,
            c.temporal_patch_size,
            c.patch_size,
            c.patch_size,
        ];
        if pe.shape.as_slice() != want && pe.shape.as_slice() != [hid, patch_dim] {
            return Err(format!(
                "'{PATCH_EMBED}' shape {:?} != {:?}",
                pe.shape, want
            ));
        }
        let patch = Proj::f32(load_vec(model, PATCH_EMBED, hid * patch_dim)?, patch_dim);
        let mut blocks = Vec::with_capacity(c.depth);
        for i in 0..c.depth {
            let p = format!("visual.blocks.{i}");
            let full = c.fullatt.contains(&i);
            let sink_name = format!("{p}.attn.sinks");
            let sinks = if c.use_sink && !full {
                Some(load_vec(model, &sink_name, c.heads)?)
            } else {
                None
            };
            blocks.push(Block {
                norm1: load_vec(model, &format!("{p}.norm1.weight"), hid)?,
                norm2: load_vec(model, &format!("{p}.norm2.weight"), hid)?,
                qkv: load_proj(model, &format!("{p}.attn.qkv.weight"), qd + 2 * kvd, hid)?,
                qkv_b: load_vec(model, &format!("{p}.attn.qkv.bias"), qd + 2 * kvd)?,
                proj: load_proj(model, &format!("{p}.attn.proj.weight"), hid, qd)?,
                proj_b: load_vec(model, &format!("{p}.attn.proj.bias"), hid)?,
                gate: load_proj(model, &format!("{p}.mlp.gate_proj.weight"), inter, hid)?,
                gate_b: load_vec(model, &format!("{p}.mlp.gate_proj.bias"), inter)?,
                up: load_proj(model, &format!("{p}.mlp.up_proj.weight"), inter, hid)?,
                up_b: load_vec(model, &format!("{p}.mlp.up_proj.bias"), inter)?,
                down: load_proj(model, &format!("{p}.mlp.down_proj.weight"), hid, inter)?,
                down_b: load_vec(model, &format!("{p}.mlp.down_proj.bias"), hid)?,
                sinks,
            });
        }
        let mw = hid * c.merge_size * c.merge_size;
        if model.tensor("visual.merger.ln_q.bias").is_some() {
            return Err(
                "visual.merger.ln_q.bias present: the MiMo merger norm is an RMSNorm".into(),
            );
        }
        let vit = Self {
            ln_q: load_vec(model, "visual.merger.ln_q.weight", hid)?,
            mlp0: load_proj(model, "visual.merger.mlp.0.weight", mw, mw)?,
            mlp0_b: load_opt_vec(model, "visual.merger.mlp.0.bias", mw)?,
            mlp2: load_proj(model, "visual.merger.mlp.2.weight", c.out_hidden, mw)?,
            mlp2_b: load_opt_vec(model, "visual.merger.mlp.2.bias", c.out_hidden)?,
            cfg: c,
            patch,
            blocks,
            sink_mode: SinkMode::from_env(),
            pool: Pool::from_env(),
            gpu_attention: std::env::var("CMF_MIMO_VIT_GPU_ATTN").as_deref() != Ok("0"),
            trace: std::env::var("CMF_MIMO_VIT_TRACE").is_ok_and(|v| v != "0"),
        };
        Ok(vit)
    }

    pub fn set_sink_mode(&mut self, mode: SinkMode) {
        self.sink_mode = mode;
    }

    pub fn sink_mode(&self) -> SinkMode {
        self.sink_mode
    }

    /// Allow (default) or forbid the device attention for the full blocks.
    pub fn set_gpu_attention(&mut self, on: bool) {
        self.gpu_attention = on;
    }

    /// The row-order RoPE tables `(cos, sin)`, `[rows, head_dim]` each,
    /// laid out `[h, w, h, w]` in quarters.
    fn rope_tables(&self, gt: usize, gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.head_dim;
        let quarter = hd / 4;
        let m = self.cfg.merge_size;
        // torch: 1 / theta ** (arange(0, dim, 2, f32) / dim), dim = hd/2.
        let dim = (hd / 2) as f32;
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0f32 / self.cfg.rope_theta.powf((2 * i) as f32 / dim))
            .collect();
        let n = gt * gh * gw;
        let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
        let mut r = 0usize;
        for _t in 0..gt {
            for a in 0..gh / m {
                for b in 0..gw / m {
                    for mh in 0..m {
                        for mw in 0..m {
                            let (hp, wp) = ((a * m + mh) as f32, (b * m + mw) as f32);
                            for d in 0..hd {
                                let q = d % (hd / 2);
                                let ang = if q < quarter {
                                    hp * inv[q]
                                } else {
                                    wp * inv[q - quarter]
                                };
                                let (s, c) = (ang as f64).sin_cos();
                                cos[r * hd + d] = c as f32;
                                sin[r * hd + d] = s as f32;
                            }
                            r += 1;
                        }
                    }
                }
            }
        }
        (cos, sin)
    }

    /// Encode one image or video: `[tokens, out_hidden]`, in the order the
    /// item's placeholders take them (raster (t, block row, block col)).
    pub fn forward(&self, input: &VisualInput) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let m = c.merge_size;
        let (gt, gh, gw) = (input.grid_t, input.grid_h, input.grid_w);
        if gt == 0 || gh == 0 || gw == 0 || gh % m != 0 || gw % m != 0 {
            return Err(format!(
                "vision grid {gt}x{gh}x{gw} is empty or not {m}-aligned"
            ));
        }
        let patch_dim = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        let n = gt * gh * gw;
        if input.rows.len() != n * patch_dim {
            return Err(format!(
                "vision input has {} values, expected {n} rows × {patch_dim}",
                input.rows.len()
            ));
        }
        let hid = c.hidden;
        let pool = self.pool.as_deref();
        let mut x = vec![0f32; n * hid];
        self.patch.matmat(&input.rows, n, &mut x, pool);

        let hd = c.head_dim;
        let (cos_r, sin_r) = self.rope_tables(gt, gh, gw);
        let perm = column_permutation(gt, gh, gw, m);
        let cos_c = gather_rows(&cos_r, &perm, hd);
        let sin_c = gather_rows(&sin_r, &perm, hd);
        let chunk = gh * gw;
        let mut scratch = Scratch::new(n, c);
        let t_blocks = std::time::Instant::now();
        for (i, blk) in self.blocks.iter().enumerate() {
            let ty = c.window_types[i];
            let prev = if i > 0 {
                c.window_types[i - 1]
            } else {
                i64::MIN
            };
            if ty == 1 && (i == 0 || prev != 1) {
                x = gather_rows(&x, &perm, hid);
            }
            if i > 0 && ty != 1 && prev == 1 {
                x = scatter_rows(&x, &perm, hid);
            }
            let (cs, sn) = if ty == 1 {
                (&cos_c, &sin_c)
            } else {
                (&cos_r, &sin_r)
            };
            let full = c.fullatt.contains(&i);
            self.block(blk, &mut x, n, chunk, cs, sn, full, &mut scratch)?;
            if self.trace {
                eprintln!(
                    "mimo-vit block {i:2} ({}): max|x| {:.1}, max|down in| {:.1}",
                    if full {
                        "full"
                    } else if ty == 1 {
                        "col "
                    } else {
                        "row "
                    },
                    max_abs(&x),
                    max_abs(&scratch.gate)
                );
            }
        }

        if self.trace {
            let total = t_blocks.elapsed().as_secs_f64();
            eprintln!(
                "mimo-vit {n} rows: blocks {total:.2} s = full attention {:.2} s + windowed attention {:.2} s + projections/norms {:.2} s",
                scratch.t_full,
                scratch.t_band,
                total - scratch.t_full - scratch.t_band
            );
        }
        // Merger: RMSNorm per row, four consecutive rows per merge unit.
        let groups = n / (m * m);
        let mw = hid * m * m;
        let mut xn = vec![0f32; n * hid];
        rms_norm_rows(&x, &self.ln_q, c.eps, &mut xn, pool);
        drop(x);
        let mut h1 = vec![0f32; groups * mw];
        self.mlp0.mm(&xn, groups, &mut h1, pool);
        if let Some(b) = &self.mlp0_b {
            add_bias(&mut h1, b);
        }
        for v in &mut h1 {
            *v = gelu_erf(*v);
        }
        let mut out = vec![0f32; groups * c.out_hidden];
        self.mlp2.mm(&h1, groups, &mut out, pool);
        if let Some(b) = &self.mlp2_b {
            add_bias(&mut out, b);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        b: &Block,
        x: &mut [f32],
        n: usize,
        chunk: usize,
        cos: &[f32],
        sin: &[f32],
        full: bool,
        s: &mut Scratch,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let pool = self.pool.as_deref();
        let (hid, hd) = (c.hidden, c.head_dim);
        let (qd, kvd) = (c.heads * hd, c.kv_heads * hd);
        let width = qd + 2 * kvd;
        rms_norm_rows(x, &b.norm1, c.eps, &mut s.norm, pool);
        b.qkv.mm(&s.norm, n, &mut s.qkv, pool);
        add_bias(&mut s.qkv, &b.qkv_b);
        // Split and rotate: q [n, heads·hd], k/v [n, kv_heads·hd].
        {
            let (qp, kp, vp) = (
                crate::pool::SendMut::new(s.q.as_mut_ptr()),
                crate::pool::SendMut::new(s.k.as_mut_ptr()),
                crate::pool::SendMut::new(s.v.as_mut_ptr()),
            );
            let qkv = &s.qkv;
            let half = hd / 2;
            let f = |lo: usize, hi: usize| {
                for r in lo..hi {
                    let src = &qkv[r * width..(r + 1) * width];
                    let (cr, sr) = (&cos[r * hd..(r + 1) * hd], &sin[r * hd..(r + 1) * hd]);
                    let rot = |head: &[f32], dst: crate::pool::SendMut, off: usize| {
                        for d in 0..hd {
                            let rh = if d < half {
                                -head[d + half]
                            } else {
                                head[d - half]
                            };
                            // SAFETY: each row writes only its own slice.
                            unsafe { *dst.at(off + d) = head[d] * cr[d] + rh * sr[d] };
                        }
                    };
                    for h in 0..c.heads {
                        rot(&src[h * hd..(h + 1) * hd], qp, r * qd + h * hd);
                    }
                    for h in 0..c.kv_heads {
                        rot(&src[qd + h * hd..qd + (h + 1) * hd], kp, r * kvd + h * hd);
                    }
                    for j in 0..kvd {
                        unsafe { *vp.at(r * kvd + j) = src[qd + kvd + j] };
                    }
                }
            };
            match pool {
                Some(p) => p.run_rows(n, &f),
                None => f(0, n),
            }
        }
        let sinks = match self.sink_mode {
            SinkMode::Off => None,
            _ => b.sinks.as_deref(),
        };
        let window = if full { None } else { c.window };
        s.attn.fill(0.0);
        let t_attn = std::time::Instant::now();
        for c0 in (0..n).step_by(chunk) {
            if window.is_none() && sinks.is_none() {
                if self.full_attention_gpu(&s.q, &s.k, &s.v, c0, chunk, &mut s.attn) {
                    continue;
                }
                self.full_attention_cpu(&s.q, &s.k, &s.v, c0, chunk, &mut s.attn);
            } else {
                self.band_attention_cpu(&s.q, &s.k, &s.v, c0, chunk, window, sinks, &mut s.attn);
            }
        }
        let dt = t_attn.elapsed().as_secs_f64();
        if window.is_none() && sinks.is_none() {
            s.t_full += dt;
        } else {
            s.t_band += dt;
        }
        b.proj.mm(&s.attn, n, &mut s.proj, pool);
        add_bias(&mut s.proj, &b.proj_b);
        for (a, &p) in x.iter_mut().zip(&s.proj) {
            *a += p;
        }
        rms_norm_rows(x, &b.norm2, c.eps, &mut s.norm, pool);
        b.gate.mm(&s.norm, n, &mut s.gate, pool);
        b.up.mm(&s.norm, n, &mut s.up, pool);
        add_bias(&mut s.gate, &b.gate_b);
        add_bias(&mut s.up, &b.up_b);
        for (g, &u) in s.gate.iter_mut().zip(&s.up) {
            *g = (*g / (1.0 + (-*g).exp())) * u;
        }
        b.down.mm(&s.gate, n, &mut s.proj, pool);
        add_bias(&mut s.proj, &b.down_b);
        for (a, &p) in x.iter_mut().zip(&s.proj) {
            *a += p;
        }
        debug_assert_eq!(s.proj.len(), n * hid);
        Ok(())
    }

    /// One frame chunk of a full block on the device, head-major panels in,
    /// token-major `[l, heads·hd]` out. `false` = not taken (no backend,
    /// refused, or too large for the n² score scratch).
    fn full_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        c0: usize,
        l: usize,
        out: &mut [f32],
    ) -> bool {
        let c = &self.cfg;
        if !self.gpu_attention
            || l < 128
            || l * l * 4 > (1usize << 30)
            || !crate::gpu::enabled_here()
        {
            return false;
        }
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
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
        let scale = (hd as f32).powf(-0.5);
        let dst = &mut out[c0 * qd..(c0 + l) * qd];
        let ok = crate::gpu::dit_attention(&qh, &kh, &vh, nh, nkv, l, hd, scale, dst);
        if ok {
            GPU_ATTENTION_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        ok
    }

    /// Exact full attention over one frame chunk: per head, query tiles
    /// through the engine GEMM, row softmax, P·V.
    fn full_attention_cpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        c0: usize,
        l: usize,
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let pool = self.pool.as_deref();
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let group = nh / nkv;
        let scale = (hd as f32).powf(-0.5);
        let tile = l.min(1024);
        let mut kh = vec![0f32; l * hd];
        let mut vt = vec![0f32; hd * l];
        let mut qh = vec![0f32; tile * hd];
        let mut sc = vec![0f32; tile * l];
        let mut oh = vec![0f32; tile * hd];
        for g in 0..nkv {
            for j in 0..l {
                let r = c0 + j;
                kh[j * hd..(j + 1) * hd]
                    .copy_from_slice(&k[r * kvd + g * hd..r * kvd + (g + 1) * hd]);
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
                    crate::fcd_ops::gemm_nt(
                        &qh[..tq * hd],
                        &kh,
                        &mut sc[..tq * l],
                        tq,
                        hd,
                        l,
                        pool,
                    );
                    {
                        let sp = crate::pool::SendMut::new(sc.as_mut_ptr());
                        let f = |lo: usize, hi: usize| {
                            for i in lo..hi {
                                // SAFETY: disjoint score rows per worker.
                                let row =
                                    unsafe { std::slice::from_raw_parts_mut(sp.at(i * l), l) };
                                softmax_row(row, None);
                            }
                        };
                        match pool {
                            Some(p) => p.run_rows(tq, &f),
                            None => f(0, tq),
                        }
                    }
                    crate::fcd_ops::gemm_nt(
                        &sc[..tq * l],
                        &vt,
                        &mut oh[..tq * hd],
                        tq,
                        l,
                        hd,
                        pool,
                    );
                    for i in 0..tq {
                        let r = c0 + t0 + i;
                        out[r * qd + h * hd..r * qd + (h + 1) * hd]
                            .copy_from_slice(&oh[i * hd..(i + 1) * hd]);
                    }
                }
            }
        }
    }

    /// Windowed attention over one frame chunk (|i−j| ≤ window in the
    /// current order), with the per-head sinks as `sink_mode` says.
    /// `window = None` means every key (a sink-carrying block without band).
    #[allow(clippy::too_many_arguments)]
    fn band_attention_cpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        c0: usize,
        l: usize,
        window: Option<usize>,
        sinks: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let group = nh / nkv;
        let scale = (hd as f32).powf(-0.5);
        let w = window.unwrap_or(l);
        let mode = self.sink_mode;
        let op = crate::pool::SendMut::new(out.as_mut_ptr());
        let f = |lo: usize, hi: usize| {
            let mut sc = vec![0f32; (2 * w + 1).min(l)];
            let mut acc = vec![0f32; hd];
            for li in lo..hi {
                let r = c0 + li;
                let j0 = li.saturating_sub(w);
                let j1 = (li + w).min(l - 1);
                for h in 0..nh {
                    let g = h / group;
                    let qi = &q[r * qd + h * hd..r * qd + (h + 1) * hd];
                    let row = &mut sc[..j1 - j0 + 1];
                    for (s, j) in row.iter_mut().zip(j0..=j1) {
                        let kr = c0 + j;
                        let kj = &k[kr * kvd + g * hd..kr * kvd + (g + 1) * hd];
                        *s = crate::attention::dot_f32(qi, kj) * scale;
                    }
                    let column = match (mode, sinks) {
                        (SinkMode::Key0, Some(sk)) => {
                            if j0 == 0 {
                                row[0] += sk[h];
                            }
                            None
                        }
                        (SinkMode::Column, Some(sk)) => Some(sk[h]),
                        _ => None,
                    };
                    softmax_row(row, column);
                    acc.fill(0.0);
                    for (&p, j) in row.iter().zip(j0..=j1) {
                        let kr = c0 + j;
                        let vj = &v[kr * kvd + g * hd..kr * kvd + (g + 1) * hd];
                        for (a, &vv) in acc.iter_mut().zip(vj) {
                            *a += p * vv;
                        }
                    }
                    for (d, &a) in acc.iter().enumerate() {
                        // SAFETY: each query row writes only its own slice.
                        unsafe { *op.at(r * qd + h * hd + d) = a };
                    }
                }
            }
        };
        match self.pool.as_deref() {
            Some(p) => p.run_rows(l, &f),
            None => f(0, l),
        }
    }
}

/// In-place softmax; `column` is an extra logit that joins the max and the
/// denominator but carries no value (sglang-style sink).
fn softmax_row(row: &mut [f32], column: Option<f32>) {
    let mut mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if let Some(s) = column {
        mx = mx.max(s);
    }
    let mut den = 0f32;
    for v in row.iter_mut() {
        *v = (*v - mx).exp();
        den += *v;
    }
    if let Some(s) = column {
        den += (s - mx).exp();
    }
    let inv = 1.0 / den;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Activation buffers reused across blocks.
struct Scratch {
    /// Seconds spent in full / windowed attention (trace only).
    t_full: f64,
    t_band: f64,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl Scratch {
    fn new(n: usize, c: &MimoVisionConfig) -> Self {
        let (qd, kvd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        Self {
            t_full: 0.0,
            t_band: 0.0,
            norm: vec![0f32; n * c.hidden],
            qkv: vec![0f32; n * (qd + 2 * kvd)],
            q: vec![0f32; n * qd],
            k: vec![0f32; n * kvd],
            v: vec![0f32; n * kvd],
            attn: vec![0f32; n * qd],
            proj: vec![0f32; n * c.hidden],
            gate: vec![0f32; n * c.intermediate],
            up: vec![0f32; n * c.intermediate],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_worked_examples() {
        let (lo, hi) = (8192, 8_388_608);
        assert_eq!(smart_resize(448, 448, 32, lo, hi).unwrap(), (448, 448));
        assert_eq!(smart_resize(352, 640, 32, lo, hi).unwrap(), (352, 640));
        assert_eq!(smart_resize(768, 1024, 32, lo, hi).unwrap(), (768, 1024));
        // Upscale branch: 20 → 32, 300 → 480, no aspect check.
        assert_eq!(smart_resize(20, 300, 32, lo, hi).unwrap(), (32, 480));
        // Ties to even: 48/32 = 1.5 → 2, 80/32 = 2.5 → 2.
        assert_eq!(smart_resize(48, 80, 32, 0, hi).unwrap(), (64, 64));
        assert!(smart_resize(32, 32 * 201, 32, lo, hi).is_err());
    }

    #[test]
    fn identity_resize_is_exact() {
        let src: Vec<f32> = (0..3 * 5 * 7).map(|v| v as f32).collect();
        assert_eq!(resize_bilinear(&src, 3, 5, 7, 5, 7), src);
    }

    #[test]
    fn frame_counts_follow_smart_nframes() {
        let cfg = MimoProcessorConfig::default();
        // 5 frames: max(5, 8) clamps to 5 → floor even 4.
        assert_eq!(smart_nframes(5, 30.0, &cfg).unwrap(), 4);
        // 10 s at 30 fps → 10 → 10.
        assert_eq!(smart_nframes(300, 30.0, &cfg).unwrap(), 10);
        assert!(smart_nframes(1, 1.0, &cfg).is_err());
        let (idx, ts) = sample_frames(8, 1.0, &cfg).unwrap();
        assert_eq!(idx, (0..8).collect::<Vec<_>>());
        assert_eq!(ts[7], 7.0);
    }

    #[test]
    fn timestamps_format_like_python() {
        assert_eq!(format_timestamp(7.0), "00:07");
        assert_eq!(format_timestamp(65.9), "01:05");
        assert_eq!(format_timestamp(6000.0), "100:00");
    }

    #[test]
    fn column_permutation_lists_units_by_column() {
        // gh = 2, gw = 4 → one unit row, two unit columns.
        let p = column_permutation(1, 4, 4, 2);
        // units (a,b): (0,0)=0 (0,1)=1 (1,0)=2 (1,1)=3; column order 0,2,1,3.
        let units: Vec<usize> = p.chunks(4).map(|c| c[0] / 4).collect();
        assert_eq!(units, vec![0, 2, 1, 3]);
    }

    #[test]
    fn y4m_source_indexes_and_decodes_frames() {
        let dir = std::env::temp_dir().join(format!("mimo-y4m-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.y4m");
        // 4x2 4:2:0 at 30000/1001 fps, 3 frames; frame k has luma 16+100k,
        // neutral chroma except frame 2 (U=V=200 → reddish-magenta).
        let mut bytes = b"YUV4MPEG2 W4 H2 F30000:1001 Ip A1:1 C420jpeg XYSCSS=420JPEG\n".to_vec();
        for k in 0..3u8 {
            bytes.extend_from_slice(b"FRAME\n");
            bytes.extend(std::iter::repeat_n(16 + 100 * k.min(2), 8));
            let c = if k == 2 { 200 } else { 128 };
            bytes.extend(std::iter::repeat_n(c, 2 * 2));
        }
        std::fs::write(&path, &bytes).unwrap();
        let src = VideoSource::open(&path, None).unwrap();
        assert_eq!(src.frame_count(), 3);
        assert!((src.fps() - 30000.0 / 1001.0).abs() < 1e-12);
        let f0 = src.read_frame(0).unwrap();
        assert_eq!((f0.width, f0.height), (4, 2));
        assert!(f0.data.iter().all(|&v| v == 0), "Y=16 is black");
        let f1 = src.read_frame(1).unwrap();
        // 1.164383 · 100 = 116.4 → 116 on every channel.
        assert!(f1.data.iter().all(|&v| v == 116), "{:?}", &f1.data[..3]);
        let f2 = src.read_frame(2).unwrap();
        assert!(
            f2.data[0] > f2.data[1] && f2.data[2] > f2.data[1],
            "{:?}",
            &f2.data[..3]
        );
        // An explicit rate overrides the header.
        assert_eq!(VideoSource::open(&path, Some(2.0)).unwrap().fps(), 2.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_directory_sorts_naturally() {
        let dir = std::env::temp_dir().join(format!("mimo-frames-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (i, name) in ["f10.ppm", "f2.ppm", "f1.ppm", "notes.txt"]
            .iter()
            .enumerate()
        {
            let mut b = b"P6\n1 1\n255\n".to_vec();
            b.extend_from_slice(&[i as u8, 0, 0]);
            std::fs::write(dir.join(name), b).unwrap();
        }
        let src = VideoSource::open(&dir, Some(1.0)).unwrap();
        assert_eq!(src.frame_count(), 3);
        let order: Vec<u8> = (0..3).map(|i| src.read_frame(i).unwrap().data[0]).collect();
        assert_eq!(order, vec![2, 1, 0], "f1, f2, f10");
        assert!(
            VideoSource::open(&dir, None).is_err(),
            "a directory needs a rate"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expansion_counts_and_rejects_mismatches() {
        let tok = Tokenizer::byte_level();
        let img = VisualInput {
            kind: VisualKind::Image,
            rows: Vec::new(),
            grid_t: 1,
            grid_h: 4,
            grid_w: 6,
            patch_dim: 1536,
            merge_size: 2,
            timestamps: Vec::new(),
            frame_indices: Vec::new(),
            resized: (64, 96),
        };
        let ids = [1, VISION_START_ID, IMAGE_PAD_ID, VISION_END_ID, 2];
        let out = expand_prompt_ids(&ids, &[&img], &[], &[], &tok).unwrap();
        assert_eq!(out.len(), 2 + 2 + 6);
        assert_eq!(out.iter().filter(|&&t| t == IMAGE_PAD_ID).count(), 6);
        // Already-expanded runs count as one placeholder, like the regex.
        let again = expand_prompt_ids(&out, &[&img], &[], &[], &tok).unwrap();
        assert_eq!(again, out);
        assert!(expand_prompt_ids(&ids, &[], &[], &[], &tok).is_err());
        assert!(expand_prompt_ids(&ids, &[&img, &img], &[], &[], &tok).is_err());
        assert!(expand_prompt_ids(&[IMAGE_PAD_ID], &[], &[], &[], &tok).is_err());
        let audio = [AUDIO_START_ID, AUDIO_PAD_ID, AUDIO_END_ID];
        let a = expand_prompt_ids(&audio, &[], &[], &[32], &tok).unwrap();
        assert_eq!(a.len(), 34);
    }

    #[test]
    fn patchify_orders_rows_by_merge_block() {
        let cfg = MimoProcessorConfig {
            patch_size: 1,
            merge_size: 2,
            temporal_patch_size: 2,
            ..MimoProcessorConfig::default()
        };
        // 2x4 frame, value = y*4+x in channel 0.
        let mut f = vec![0f32; 3 * 8];
        for i in 0..8 {
            f[i] = i as f32;
        }
        let (rows, gt, gh, gw) = patchify(&[&f, &f], 2, 4, &cfg).unwrap();
        assert_eq!((gt, gh, gw), (1, 2, 4));
        // row dim = 3*2*1*1 = 6; channel 0 is values [0..2) of each row.
        let ch0: Vec<f32> = rows.chunks(6).map(|r| r[0]).collect();
        assert_eq!(ch0, vec![0., 1., 4., 5., 2., 3., 6., 7.]);
    }
}
