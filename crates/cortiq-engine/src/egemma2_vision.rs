//! EmbeddingGemma 2 vision: the `gemma4_vision` tower, its projection into
//! the text model's input space, and the image / video preprocessing of
//! `EmbeddingGemma2Processor` — so an image, a video or an interleaved
//! text-and-media input lands in the same 768-d space as text.
//!
//! ## Preprocessing (`Gemma4ImageProcessor`, `EmbeddingGemma2VideoProcessor`)
//!
//! 1. **RGB.** Alpha is dropped (PIL's `convert("RGB")`), grey replicated.
//! 2. **Resize** to the largest size whose sides are multiples of 48 px
//!    (`patch 16 × pooling 3`) holding at most `budget · 9` patches, the
//!    aspect ratio kept (`get_aspect_ratio_preserving_size`); an image that
//!    already has that size is left alone. The resampler is torch's uint8
//!    antialiased bicubic (`_upsample_bicubic2d_aa`, Keys a = -0.5, int16
//!    fixed-point weights, horizontal pass then vertical, each rounded to
//!    uint8) — what torchvision runs on a uint8 tensor, byte for byte.
//! 3. **Patches.** `v/255`, then the tower's own `2·(v - 0.5)`; 16×16
//!    patches in row-major order, each flattened `(py, px, rgb)`; patch
//!    `(x, y)` is its column and row.
//!
//! The soft-token budget is 70, 140, 280 (default), 560 or 1120 per image;
//! the image yields `(W/48)·(H/48)` tokens. Video frames are sampled at
//! 1 fps (`int(i · fps_src)` for `i < int(frames / fps_src)`, at least one),
//! cut to 32 frames evenly when longer, and each frame is an image at a
//! budget of 140.
//!
//! ## The tower (`Gemma4VisionModel`, 16 layers, width 768, 12 heads of 64)
//!
//! * patch embedding: `x·W_inᵀ + P_x[x] + P_y[y]` (two 10240-row tables);
//! * every layer is the Gemma sandwich with gelu-tanh gated FFN:
//!   `h += N(attn(N(h)))`, `h += N(FFN(N(h)))`, RMS norms multiplying by
//!   their weight;
//! * attention is bidirectional over the image's own patches, unscaled
//!   (the q/k RMS norms carry the scale), values RMS-normalized without a
//!   weight, and 2-D RoPE (θ = 100): the first 32 channels of a head
//!   rotate with the patch column, the last 32 with its row, each half
//!   `rotate_half` over 16 frequencies;
//! * pooling: the mean of each 3×3 block of patches, times `sqrt(768)`;
//! * `embed_vision`: RMS norm without weight, then 768→512 — the rows that
//!   replace the `<|image|>` / `<|video|>` placeholders in the text model's
//!   input (not multiplied by the token embedding scale).
//!
//! Padding patches of the reference processor are masked out of attention
//! and pooling there, so the tower here simply runs each image on its real
//! patches. Images are packed back to back for the projections (one GEMM
//! per weight over every patch of a batch); attention runs per image on the
//! host in f32 (scores, softmax and value mix pipelined so the matrix unit
//! and the vector units overlap), and on Metal's flash kernel for the
//! largest images (4096+ patches: budgets 560, 1120), where the n² term
//! dominates
//! (`CMF_EGEMMA2_VISION_GPU=0` keeps everything on the host, `=1` sends
//! every image to the device).

use crate::egemma2::{Mat, add_normed, gelu_mul, rms_into, rms_rows, softmax, vecf};
use crate::ltxdit::{Shared, rows};
use crate::media::RgbFrame;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The soft-token budgets the processor accepts.
pub const BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// Patches are packed into one tower forward up to this many rows.
const PACK_PATCHES: usize = 10_080;
/// Scores per host attention block (32 MB of f32), and the fewest query
/// rows a block takes.
const SCORE_BLOCK: usize = 8 << 20;
const QBLOCK_MIN: usize = 256;
/// Images with at least this many patches attend on Metal when it is up
/// (budgets 560 and 1120 of a square-ish image).
const GPU_MIN_PATCHES: usize = 4096;

/// `CMF_EGEMMA2_PROF=1`: per-forward time split of the tower on stderr.
mod vprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub const QKV: usize = 0;
    pub const NORM_ROPE: usize = 1;
    pub const CORE: usize = 2;
    pub const O: usize = 3;
    pub const MLP: usize = 4;
    pub const GELU: usize = 5;
    pub const NORMS: usize = 6;
    const NAMES: [&str; 7] = [
        "q/k/v",
        "norm+rope",
        "attention core",
        "o",
        "mlp",
        "gelu",
        "norms",
    ];
    static T: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_EGEMMA2_PROF").is_ok_and(|v| v == "1"))
    }
    pub fn add(i: usize, t: std::time::Instant) {
        if on() {
            T[i].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }
    pub fn take() -> String {
        NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{n} {:.1}", T[i].swap(0, Ordering::Relaxed) as f64 / 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ------------------------------------------------------------ processor

/// The processor settings (`processor_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionProcessor {
    pub patch: usize,
    pub pool_k: usize,
    /// soft tokens per image (`image_processor.max_soft_tokens`)
    pub image_budget: usize,
    /// soft tokens per video frame (`video_processor.max_soft_tokens`)
    pub video_budget: usize,
    /// frames sampled per second of video
    pub video_fps: f64,
    /// sampled frames are cut to this many, evenly (`overflow_strategy`)
    pub max_frames: usize,
    /// `uniform` (default) or `truncate`
    pub overflow_truncate: bool,
}

impl Default for VisionProcessor {
    fn default() -> Self {
        VisionProcessor {
            patch: 16,
            pool_k: 3,
            image_budget: 280,
            video_budget: 140,
            video_fps: 1.0,
            max_frames: 32,
            overflow_truncate: false,
        }
    }
}

impl VisionProcessor {
    /// From `processor_config.json` (missing fields keep the defaults).
    pub fn from_config(cfg: &Value) -> Self {
        let mut p = VisionProcessor::default();
        let u = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let ip = &cfg["image_processor"];
        let vp = &cfg["video_processor"];
        if let Some(x) = u(ip, "patch_size") {
            p.patch = x;
        }
        if let Some(x) = u(ip, "pooling_kernel_size") {
            p.pool_k = x;
        }
        if let Some(x) = u(ip, "max_soft_tokens").or_else(|| u(cfg, "image_seq_length")) {
            p.image_budget = x;
        }
        if let Some(x) = u(vp, "max_soft_tokens") {
            p.video_budget = x;
        }
        if let Some(x) = vp.get("fps").and_then(|x| x.as_f64()) {
            p.video_fps = x;
        }
        if let Some(x) = u(vp, "max_frames") {
            p.max_frames = x;
        }
        p.overflow_truncate =
            vp.get("overflow_strategy").and_then(|x| x.as_str()) == Some("truncate");
        p
    }

    /// `budget` checked against the five the model was trained for.
    pub fn check_budget(budget: usize) -> Result<usize, String> {
        if BUDGETS.contains(&budget) {
            Ok(budget)
        } else {
            Err(format!(
                "soft-token budget {budget}: use 70, 140, 280, 560 or 1120"
            ))
        }
    }
}

/// `get_aspect_ratio_preserving_size`: the `(height, width)` an image of
/// `height × width` is resized to so it holds at most `max_patches` patches
/// with both sides multiples of `patch · k`.
pub fn target_size(
    height: usize,
    width: usize,
    patch: usize,
    max_patches: usize,
    k: usize,
) -> Result<(usize, usize), String> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / total_px).sqrt();
    let (ih, iw) = (factor * height as f64, factor * width as f64);
    let side = patch * k;
    let mut th = (ih / side as f64).floor() as usize * side;
    let mut tw = (iw / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err(format!(
            "a {width}x{height} image cannot be resized to a non-empty patch grid"
        ));
    }
    let max_side = (max_patches / (k * k)) * side;
    if th == 0 {
        th = side;
        tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        return Err(format!(
            "resizing {width}x{height} to {tw}x{th} exceeds {max_patches} patches"
        ));
    }
    Ok((th, tw))
}

// ------------------------------------------------------------ resize

/// Keys' cubic, a = -0.5 (torch `HelperInterpCubic::aa_filter`).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One axis of torch's antialiased bicubic: per output index the first
/// source index, the tap count and the int16 weights, plus the weights'
/// fixed-point precision (`_compute_index_ranges_int16_weights`).
struct AxisTaps {
    xmin: Vec<usize>,
    xsize: Vec<usize>,
    w: Vec<i16>,
    stride: usize,
    precision: u32,
}

fn axis_taps(in_size: usize, out_size: usize) -> AxisTaps {
    let scale = in_size as f64 / out_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_interp = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut wf = vec![0f64; out_size * max_interp];
    let mut xmin = vec![0usize; out_size];
    let mut xsize = vec![0usize; out_size];
    let mut wt_max = 0f64;
    for i in 0..out_size {
        let center = scale * (i as f64 + 0.5);
        let lo = ((center - support + 0.5) as i64).max(0);
        let hi = ((center + support + 0.5) as i64).min(in_size as i64);
        let n = (hi - lo).clamp(0, max_interp as i64) as usize;
        let row = &mut wf[i * max_interp..(i + 1) * max_interp];
        let mut total = 0f64;
        for (j, w) in row.iter_mut().enumerate().take(n) {
            *w = cubic((j as f64 + lo as f64 - center + 0.5) * invscale);
            total += *w;
        }
        if total != 0.0 {
            for w in row.iter_mut().take(n) {
                *w /= total;
            }
        }
        for &w in row.iter() {
            wt_max = wt_max.max(w);
        }
        xmin[i] = lo as usize;
        xsize[i] = n;
    }
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * (1u64 << (precision + 1)) as f64) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let mul = (1u64 << precision) as f64;
    let w = wf
        .iter()
        .map(|&v| {
            let v = v * mul;
            (if v < 0.0 {
                (-0.5 + v) as i32
            } else {
                (0.5 + v) as i32
            }) as i16
        })
        .collect();
    AxisTaps {
        xmin,
        xsize,
        w,
        stride: max_interp,
        precision,
    }
}

/// One resampling pass over `src` (`[outer][len_in][inner]`, uint8), along
/// the middle axis, into `[outer][len_out][inner]`.
fn resample_axis(
    src: &[u8],
    outer: usize,
    len_in: usize,
    inner: usize,
    taps: &AxisTaps,
    len_out: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; outer * len_out * inner];
    let round = 1i64 << (taps.precision.max(1) - 1);
    for o in 0..outer {
        let s = &src[o * len_in * inner..(o + 1) * len_in * inner];
        let d = &mut out[o * len_out * inner..(o + 1) * len_out * inner];
        for i in 0..len_out {
            let (x0, n) = (taps.xmin[i], taps.xsize[i]);
            let w = &taps.w[i * taps.stride..i * taps.stride + n];
            for c in 0..inner {
                let mut acc = round;
                for (j, &wj) in w.iter().enumerate() {
                    acc += s[(x0 + j) * inner + c] as i64 * wj as i64;
                }
                d[i * inner + c] = (acc >> taps.precision).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// torch / torchvision's uint8 antialiased bicubic resize (`antialias =
/// True`, `align_corners = False`): horizontal pass, then vertical, each
/// rounded to uint8; an unchanged axis is skipped.
pub fn resize_bicubic_aa(img: &RgbFrame, height: usize, width: usize) -> RgbFrame {
    let (h0, w0) = (img.height, img.width);
    let mut cur = img.data.clone();
    if width != w0 {
        cur = resample_axis(&cur, h0, w0, 3, &axis_taps(w0, width), width);
    }
    if height != h0 {
        cur = resample_axis(&cur, 1, h0, width * 3, &axis_taps(h0, height), height);
    }
    RgbFrame {
        width,
        height,
        data: cur,
    }
}

// ------------------------------------------------------------ patches

/// One image (or video frame) ready for the tower: its patch grid and the
/// patch rows, `[pw·ph, 3·patch²]`, already in the tower's `[-1, 1]`.
#[derive(Clone, Debug)]
pub struct VisionInput {
    pub pw: usize,
    pub ph: usize,
    pub pixels: Vec<f32>,
}

impl VisionInput {
    pub fn n_patches(&self) -> usize {
        self.pw * self.ph
    }

    /// Soft tokens this input becomes (`(pw/k)·(ph/k)`).
    pub fn n_soft(&self, k: usize) -> usize {
        (self.pw / k) * (self.ph / k)
    }
}

/// Patchify an image whose sides are multiples of `patch`: `v/255` (f32,
/// the processor's `rescale`), then `2·(v - 0.5)` (the patch embedder).
pub fn patchify(img: &RgbFrame, patch: usize) -> VisionInput {
    let (pw, ph) = (img.width / patch, img.height / patch);
    let dim = 3 * patch * patch;
    // the processor's rescale_factor (1/255 as Python's float), as f32
    let scale = (1.0f64 / 255.0) as f32;
    let mut pixels = vec![0f32; pw * ph * dim];
    for py in 0..ph {
        for px in 0..pw {
            let dst = &mut pixels[(py * pw + px) * dim..(py * pw + px + 1) * dim];
            for yy in 0..patch {
                let row = (py * patch + yy) * img.width + px * patch;
                let src = &img.data[row * 3..(row + patch) * 3];
                for (d, &v) in dst[yy * patch * 3..(yy + 1) * patch * 3]
                    .iter_mut()
                    .zip(src)
                {
                    *d = 2.0 * (v as f32 * scale - 0.5);
                }
            }
        }
    }
    VisionInput { pw, ph, pixels }
}

impl VisionProcessor {
    /// The size an image of `w × h` is resized to at `budget` soft tokens.
    pub fn resized_size(
        &self,
        w: usize,
        h: usize,
        budget: usize,
    ) -> Result<(usize, usize), String> {
        let (th, tw) = target_size(
            h,
            w,
            self.patch,
            budget * self.pool_k * self.pool_k,
            self.pool_k,
        )?;
        Ok((tw, th))
    }

    /// Resize (when the size changes) and patchify one image.
    pub fn prepare_image(&self, img: &RgbFrame, budget: usize) -> Result<VisionInput, String> {
        VisionProcessor::check_budget(budget)?;
        let (tw, th) = self.resized_size(img.width, img.height, budget)?;
        if (tw, th) == (img.width, img.height) {
            return Ok(patchify(img, self.patch));
        }
        Ok(patchify(&resize_bicubic_aa(img, th, tw), self.patch))
    }

    /// Video frames (already sampled, one size): each resized to the size
    /// frame 0 takes at `budget`, then patchified.
    pub fn prepare_frames(
        &self,
        frames: &[RgbFrame],
        budget: usize,
    ) -> Result<Vec<VisionInput>, String> {
        let first = frames.first().ok_or("a video needs at least one frame")?;
        if let Some((i, f)) = frames
            .iter()
            .enumerate()
            .find(|(_, f)| (f.width, f.height) != (first.width, first.height))
        {
            return Err(format!(
                "video frame {i} is {}x{}, frame 0 is {}x{}: all frames must share one size",
                f.width, f.height, first.width, first.height
            ));
        }
        VisionProcessor::check_budget(budget)?;
        let (tw, th) = self.resized_size(first.width, first.height, budget)?;
        Ok(frames
            .iter()
            .map(|f| {
                if (tw, th) == (f.width, f.height) {
                    patchify(f, self.patch)
                } else {
                    patchify(&resize_bicubic_aa(f, th, tw), self.patch)
                }
            })
            .collect())
    }

    /// The frames of a `total`-frame video at `src_fps` the processor keeps:
    /// `int(i · src_fps / fps)` for `i < max(1, int(total / src_fps · fps))`,
    /// then cut to `max_frames` (evenly, `np.linspace(..., dtype=int)`, or
    /// the first ones under `truncate`). `src_fps = None` (frames with no
    /// rate) keeps every frame and applies only the cap.
    pub fn sample_indices(&self, total: usize, src_fps: Option<f64>) -> Vec<usize> {
        if total == 0 {
            return Vec::new();
        }
        let mut idx: Vec<usize> = match src_fps.filter(|f| *f > 0.0 && f.is_finite()) {
            None => (0..total).collect(),
            Some(f) => {
                let step = f / self.video_fps;
                let duration = total as f64 / f;
                let n = ((duration * self.video_fps) as usize).max(1);
                (0..n)
                    .map(|i| ((i as f64 * step) as usize).min(total - 1))
                    .collect()
            }
        };
        let m = self.max_frames;
        if m > 0 && idx.len() > m {
            if self.overflow_truncate {
                idx.truncate(m);
            } else {
                // np.linspace(0, len-1, m, dtype=int): i·step, the last
                // exactly the end, floored
                let last = idx.len() - 1;
                let step = if m > 1 {
                    last as f64 / (m - 1) as f64
                } else {
                    0.0
                };
                idx = (0..m)
                    .map(|i| {
                        let t = if i + 1 == m && m > 1 {
                            last
                        } else {
                            (i as f64 * step).floor() as usize
                        };
                        idx[t.min(last)]
                    })
                    .collect();
            }
        }
        idx
    }
}

// ------------------------------------------------------------ video

/// A decoded video: the sampled frames and where they came from.
#[derive(Clone, Debug)]
pub struct DecodedVideo {
    pub frames: Vec<RgbFrame>,
    /// source frame index of each kept frame
    pub indices: Vec<usize>,
    pub total_frames: usize,
    pub src_fps: Option<f64>,
    /// `frames-dir`, `y4m` or `ffmpeg`
    pub decoder: &'static str,
}

/// Decode and sample a video the way the processor does.
///
/// * a **directory** of images (png/jpg/webp/gif/ppm, natural name order)
///   is a list of frames: with `fps` they play at that rate and are sampled
///   to 1 fps; without it they are taken as already sampled (every frame
///   kept, cut to 32);
/// * a **`.y4m`** stream is read natively (its own rate, BT.601 YUV → RGB);
/// * anything else (**mp4, webm, mov, mkv, …**) goes through the `ffmpeg`
///   and `ffprobe` executables — decoded to rgb24 and sampled exactly as the
///   reference's PyAV path (`frames / average_rate` is the duration; frames
///   as stored, no rotation applied). ffmpeg's YUV → RGB rounding differs
///   from PyAV's by up to 2 levels (cosine 0.9993 against the reference
///   computed from PyAV's frames). There is no in-process codec: without
///   ffmpeg on `PATH`, pass a frame directory or a y4m.
pub fn decode_video(
    path: &Path,
    fps: Option<f64>,
    proc: &VisionProcessor,
) -> Result<DecodedVideo, String> {
    use crate::mimo_vision::VideoSource;
    if path.is_dir() {
        let src = VideoSource::frame_dir(path, fps.unwrap_or(1.0))?;
        let total = src.frame_count();
        let indices = proc.sample_indices(total, fps);
        let frames = indices
            .iter()
            .map(|&i| src.read_frame(i))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(DecodedVideo {
            frames,
            indices,
            total_frames: total,
            src_fps: fps,
            decoder: "frames-dir",
        });
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "y4m" {
        let src = VideoSource::open(path, fps)?;
        let total = src.frame_count();
        let f = src.fps();
        let indices = proc.sample_indices(total, Some(f));
        let frames = indices
            .iter()
            .map(|&i| src.read_frame(i))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(DecodedVideo {
            frames,
            indices,
            total_frames: total,
            src_fps: Some(f),
            decoder: "y4m",
        });
    }
    decode_with_ffmpeg(path, fps, proc)
}

/// What ffprobe says about a video's first stream.
struct Probe {
    width: usize,
    height: usize,
    fps: f64,
    /// the container's frame count, when it keeps one
    frames: Option<usize>,
    duration: Option<f64>,
}

fn ffprobe_stream(path: &Path) -> Result<Probe, String> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| {
            format!(
                "{}: decoding this video needs ffmpeg/ffprobe on PATH ({e}); \
                 or pass a directory of frames or a .y4m",
                path.display()
            )
        })?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("ffprobe json: {e}"))?;
    let s = &v["streams"][0];
    let w = s["width"].as_u64().ok_or("ffprobe: no video stream")? as usize;
    let h = s["height"].as_u64().ok_or("ffprobe: no video stream")? as usize;
    let rate = |k: &str| -> Option<f64> {
        let r = s[k].as_str()?;
        let (n, d) = r.split_once('/').unwrap_or((r, "1"));
        let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
        (d > 0.0 && n > 0.0).then(|| n / d)
    };
    let fps = rate("avg_frame_rate")
        .or_else(|| rate("r_frame_rate"))
        .ok_or("ffprobe: the video has no frame rate")?;
    let nb = s["nb_frames"]
        .as_str()
        .and_then(|x| x.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let dur = s["duration"]
        .as_str()
        .or_else(|| v["format"]["duration"].as_str())
        .and_then(|x| x.parse::<f64>().ok());
    Ok(Probe {
        width: w,
        height: h,
        fps,
        frames: nb,
        duration: dur,
    })
}

fn decode_with_ffmpeg(
    path: &Path,
    fps_override: Option<f64>,
    proc: &VisionProcessor,
) -> Result<DecodedVideo, String> {
    use std::io::Read;
    let Probe {
        width: w,
        height: h,
        fps: probed_fps,
        frames: nb,
        duration: dur,
    } = ffprobe_stream(path)?;
    let fps = fps_override.unwrap_or(probed_fps);
    // PyAV: total = stream.frames, duration = total / average_rate; a
    // container that does not count its frames falls back to its duration.
    let total = nb
        .or_else(|| dur.map(|d| (d * probed_fps).round() as usize))
        .ok_or_else(|| format!("{}: frame count unknown", path.display()))?;
    let indices = proc.sample_indices(total, Some(fps));
    let last = *indices.last().ok_or("video has no frames")?;
    let mut child = std::process::Command::new("ffmpeg")
        // -noautorotate: frames as stored, as PyAV (the reference decoder)
        // hands them over — a rotation tag is not applied
        .args(["-v", "error", "-nostdin", "-noautorotate", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-frames:v",
            &(last + 1).to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut stdout = child.stdout.take().ok_or("ffmpeg: no stdout")?;
    let frame_bytes = w * h * 3;
    let mut buf = vec![0u8; frame_bytes];
    let mut frames = Vec::with_capacity(indices.len());
    let mut kept = Vec::with_capacity(indices.len());
    let mut want = indices.iter().peekable();
    let mut i = 0usize;
    while want.peek().is_some() {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("ffmpeg read: {e}")),
        }
        while want.peek() == Some(&&i) {
            frames.push(RgbFrame::new(w, h, buf.clone())?);
            kept.push(i);
            want.next();
        }
        i += 1;
    }
    drop(stdout);
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    if frames.is_empty() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        return Err(format!(
            "ffmpeg decoded no frames from {} ({status}): {}",
            path.display(),
            err.trim()
        ));
    }
    Ok(DecodedVideo {
        frames,
        indices: kept,
        total_frames: total,
        src_fps: Some(fps),
        decoder: "ffmpeg",
    })
}

// ------------------------------------------------------------ the tower

struct VLayer {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    in_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ff_norm: Vec<f32>,
    post_ff_norm: Vec<f32>,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// The `gemma4_vision` tower and `embed_vision` of an EmbeddingGemma 2 pack.
pub struct VisionTower {
    input_proj: Mat,
    /// `[2, rows, hidden]`: column table, then row table
    pos: Vec<f32>,
    pos_rows: usize,
    layers: Vec<VLayer>,
    embed_proj: Mat,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    /// RoPE inverse frequencies, `head_dim / 4` of them
    inv_freq: Vec<f32>,
    pub pool_k: usize,
    /// the text model's width (the soft tokens' size)
    pub out_dim: usize,
    pool: Option<Arc<Pool>>,
    busy: Mutex<()>,
    /// images of at least this many patches attend on the device
    gpu_min_patches: usize,
}

/// Per-image segment of a packed batch.
#[derive(Clone, Copy)]
struct Seg {
    off: usize,
    pw: usize,
    ph: usize,
}

impl VisionTower {
    /// Does this pack carry the vision tower?
    pub fn present(model: &CmfModel) -> bool {
        model
            .tensor("vision_tower.patch_embedder.input_proj.weight")
            .is_some()
    }

    pub fn load(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        if !Self::present(model) {
            return Err(
                "this EmbeddingGemma 2 file has no vision tower (packed with CMF_EGEMMA2_TEXT_ONLY=1?)"
                    .into(),
            );
        }
        let prov = model.header.provenance.clone().unwrap_or(Value::Null);
        let vc = &prov["embedding_gemma2"]["config"]["vision_config"];
        let g = |k: &str, d: u64| vc.get(k).and_then(|v| v.as_u64()).unwrap_or(d) as usize;
        let hidden = g("hidden_size", 768);
        let heads = g("num_attention_heads", 12);
        let head_dim = g("head_dim", 64);
        let n_layers = g("num_hidden_layers", 16);
        let pool_k = g("pooling_kernel_size", 3);
        let theta = vc["rope_parameters"]["rope_theta"]
            .as_f64()
            .unwrap_or(100.0) as f32;
        if vc.get("use_clipped_linears").and_then(|v| v.as_bool()) == Some(true) {
            return Err("vision tower with clipped linears is not supported".into());
        }
        if vc.get("standardize").and_then(|v| v.as_bool()) == Some(true) {
            return Err("vision tower with standardization is not supported".into());
        }
        // Like the text stack: quantized matrices are widened to f32 once
        // (Accelerate GEMMs); CMF_EGEMMA2_LOWMEM=1 keeps the quantized kernels.
        let dequant = !std::env::var("CMF_EGEMMA2_LOWMEM").is_ok_and(|v| v == "1");
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("vision_tower.encoder.layers.{i}");
            let ld = |s: &str| Mat::load(model, &format!("{p}.{s}.linear.weight"), dequant);
            let nv = |s: &str| vecf(model, &format!("{p}.{s}.weight"));
            layers.push(VLayer {
                q: ld("self_attn.q_proj")?,
                k: ld("self_attn.k_proj")?,
                v: ld("self_attn.v_proj")?,
                o: ld("self_attn.o_proj")?,
                q_norm: nv("self_attn.q_norm")?,
                k_norm: nv("self_attn.k_norm")?,
                in_norm: nv("input_layernorm")?,
                post_attn_norm: nv("post_attention_layernorm")?,
                pre_ff_norm: nv("pre_feedforward_layernorm")?,
                post_ff_norm: nv("post_feedforward_layernorm")?,
                gate: ld("mlp.gate_proj")?,
                up: ld("mlp.up_proj")?,
                down: ld("mlp.down_proj")?,
            });
        }
        let pos_name = "vision_tower.patch_embedder.position_embedding_table";
        let pos_shape = model
            .tensor(pos_name)
            .ok_or_else(|| format!("missing tensor {pos_name}"))?
            .shape
            .clone();
        if pos_shape.len() != 3 || pos_shape[0] != 2 || pos_shape[2] != hidden {
            return Err(format!("{pos_name}: unexpected shape {pos_shape:?}"));
        }
        let pos = vecf(model, pos_name)?;
        let embed_proj = Mat::load(model, "embed_vision.embedding_projection.weight", true)?;
        let quarter = head_dim / 4;
        // torch: 1 / theta ** (arange(0, head_dim/2, 2) / (head_dim/2)), f32
        let inv_freq = (0..quarter)
            .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / (head_dim / 2) as f32))
            .collect();
        // Metal's flash attention (f16 operands, f32 sums) wins only where the
        // n² scores dominate. Attention core on an M4, device vs the
        // pipelined host path: 4.3 s vs 8.2 s at 9801 patches (budget 1120),
        // 1.05 s vs 1.68 s at 4761 (560), 0.43 s vs 0.35 s at 2304 (280) —
        // where the host path, exact f32, stays.
        // CMF_EGEMMA2_VISION_GPU=0 never, =1 from 256 patches.
        let gpu_min_patches = match std::env::var("CMF_EGEMMA2_VISION_GPU").as_deref() {
            Ok("0") => usize::MAX,
            Ok("1") => 256,
            _ => GPU_MIN_PATCHES,
        };
        Ok(VisionTower {
            input_proj: Mat::load(
                model,
                "vision_tower.patch_embedder.input_proj.weight",
                dequant,
            )?,
            pos,
            pos_rows: pos_shape[1],
            layers,
            out_dim: embed_proj.rows(),
            embed_proj,
            hidden,
            heads,
            head_dim,
            inv_freq,
            pool_k,
            pool,
            busy: Mutex::new(()),
            gpu_min_patches,
        })
    }

    /// Soft tokens of each input, `[n_soft, out_dim]` row-major: the rows
    /// that replace the input's placeholders in the text model.
    pub fn encode(&self, inputs: &[VisionInput]) -> Result<Vec<Vec<f32>>, String> {
        let k = self.pool_k;
        for (i, x) in inputs.iter().enumerate() {
            if x.pw % k != 0 || x.ph % k != 0 || x.pw == 0 || x.ph == 0 {
                return Err(format!(
                    "image {i}: a {}x{} patch grid does not pool by {k}",
                    x.pw, x.ph
                ));
            }
            if x.pw > self.pos_rows || x.ph > self.pos_rows {
                return Err(format!(
                    "image {i}: {}x{} patches exceed the {}-entry position table",
                    x.pw, x.ph, self.pos_rows
                ));
            }
        }
        let _g = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(inputs.len());
        let mut start = 0usize;
        while start < inputs.len() {
            let mut end = start;
            let mut rows_n = 0usize;
            while end < inputs.len()
                && (end == start || rows_n + inputs[end].n_patches() <= PACK_PATCHES)
            {
                rows_n += inputs[end].n_patches();
                end += 1;
            }
            out.extend(self.forward_packed(&inputs[start..end]));
            start = end;
        }
        Ok(out)
    }

    fn forward_packed(&self, inputs: &[VisionInput]) -> Vec<Vec<f32>> {
        let t_fwd = std::time::Instant::now();
        let pool = self.pool.as_deref();
        let d = self.hidden;
        let mut segs = Vec::with_capacity(inputs.len());
        let mut off = 0usize;
        for x in inputs {
            segs.push(Seg {
                off,
                pw: x.pw,
                ph: x.ph,
            });
            off += x.n_patches();
        }
        let n = off;
        // patch embedding + the two position tables
        let pix: Vec<f32> = inputs
            .iter()
            .flat_map(|x| x.pixels.iter().copied())
            .collect();
        let mut h = self.input_proj.apply(&pix, n, pool);
        drop(pix);
        let (tx, ty) = self.pos.split_at(self.pos_rows * d);
        for s in &segs {
            for py in 0..s.ph {
                for px in 0..s.pw {
                    let r = &mut h[(s.off + py * s.pw + px) * d..][..d];
                    let (ex, ey) = (&tx[px * d..(px + 1) * d], &ty[py * d..(py + 1) * d]);
                    for ((v, &a), &b) in r.iter_mut().zip(ex).zip(ey) {
                        *v += a + b;
                    }
                }
            }
        }
        // per-row RoPE angles: (column, row) of every patch
        let quarter = self.head_dim / 4;
        let mut cos = vec![0f32; n * 2 * quarter];
        let mut sin = vec![0f32; n * 2 * quarter];
        for s in &segs {
            for py in 0..s.ph {
                for px in 0..s.pw {
                    let t = s.off + py * s.pw + px;
                    for (axis, p) in [(0usize, px), (1, py)] {
                        for (j, &f) in self.inv_freq.iter().enumerate() {
                            let a = (p as f32 * f) as f64;
                            cos[(t * 2 + axis) * quarter + j] = a.cos() as f32;
                            sin[(t * 2 + axis) * quarter + j] = a.sin() as f32;
                        }
                    }
                }
            }
        }
        for l in &self.layers {
            let t0 = std::time::Instant::now();
            let a = rms_rows(&h, Some(&l.in_norm), d, pool);
            vprof::add(vprof::NORMS, t0);
            let attn = self.attention(l, &a, n, &segs, &cos, &sin, pool);
            let t0 = std::time::Instant::now();
            add_normed(&mut h, &attn, &l.post_attn_norm, d, pool);
            let m = rms_rows(&h, Some(&l.pre_ff_norm), d, pool);
            vprof::add(vprof::NORMS, t0);
            let t0 = std::time::Instant::now();
            let mut g = l.gate.apply(&m, n, pool);
            let u = l.up.apply(&m, n, pool);
            vprof::add(vprof::MLP, t0);
            let t0 = std::time::Instant::now();
            gelu_mul(&mut g, &u, pool);
            drop(u);
            vprof::add(vprof::GELU, t0);
            let t0 = std::time::Instant::now();
            let f = l.down.apply(&g, n, pool);
            vprof::add(vprof::MLP, t0);
            let t0 = std::time::Instant::now();
            add_normed(&mut h, &f, &l.post_ff_norm, d, pool);
            vprof::add(vprof::NORMS, t0);
        }
        // 3x3 average pooling, sqrt(hidden), then embed_vision
        let k = self.pool_k;
        let root = (d as f32).sqrt();
        let inv_k2 = 1.0f32 / (k * k) as f32;
        if vprof::on() {
            eprintln!(
                "egemma2 vision: {n} patches / {} images in {:.1} ms — {}",
                segs.len(),
                t_fwd.elapsed().as_secs_f64() * 1e3,
                vprof::take()
            );
        }
        segs.iter()
            .map(|s| {
                let (cw, ch) = (s.pw / k, s.ph / k);
                let mut pooled = vec![0f32; cw * ch * d];
                for py in 0..s.ph {
                    for px in 0..s.pw {
                        let cell = (py / k) * cw + px / k;
                        let src = &h[(s.off + py * s.pw + px) * d..][..d];
                        for (o, &v) in pooled[cell * d..(cell + 1) * d].iter_mut().zip(src) {
                            *o += v * inv_k2;
                        }
                    }
                }
                for v in pooled.iter_mut() {
                    *v *= root;
                }
                let normed = rms_rows(&pooled, None, d, pool);
                self.embed_proj.apply(&normed, cw * ch, pool)
            })
            .collect()
    }

    /// Bidirectional attention of one layer, per image. Returns `o_proj`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: &VLayer,
        a: &[f32],
        n: usize,
        segs: &[Seg],
        cos: &[f32],
        sin: &[f32],
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let (hd, nh) = (self.head_dim, self.heads);
        let w = nh * hd;
        let t0 = std::time::Instant::now();
        let mut q = l.q.apply(a, n, pool);
        let mut k = l.k.apply(a, n, pool);
        let mut v = l.v.apply(a, n, pool);
        vprof::add(vprof::QKV, t0);
        let t0 = std::time::Instant::now();
        let quarter = hd / 4;
        {
            let (qp, kp, vp) = (
                Shared(q.as_mut_ptr()),
                Shared(k.as_mut_ptr()),
                Shared(v.as_mut_ptr()),
            );
            rows(pool, n, &|s, e| {
                let mut tmp = vec![0f32; hd];
                for t in s..e {
                    let c = &cos[t * 2 * quarter..(t + 1) * 2 * quarter];
                    let sn = &sin[t * 2 * quarter..(t + 1) * 2 * quarter];
                    let rope = |x: &mut [f32]| {
                        // two halves: columns then rows, rotate_half in each
                        for axis in 0..2 {
                            let part = &mut x[axis * 2 * quarter..(axis + 1) * 2 * quarter];
                            let (cc, ss) = (
                                &c[axis * quarter..(axis + 1) * quarter],
                                &sn[axis * quarter..(axis + 1) * quarter],
                            );
                            for i in 0..quarter {
                                let (x1, x2) = (part[i], part[i + quarter]);
                                part[i] = x1 * cc[i] - x2 * ss[i];
                                part[i + quarter] = x2 * cc[i] + x1 * ss[i];
                            }
                        }
                    };
                    let qr = unsafe { qp.at(t * w, w) };
                    for hh in qr.chunks_exact_mut(hd) {
                        rms_into(hh, Some(&l.q_norm), &mut tmp);
                        hh.copy_from_slice(&tmp);
                        rope(hh);
                    }
                    let kr = unsafe { kp.at(t * w, w) };
                    for hh in kr.chunks_exact_mut(hd) {
                        rms_into(hh, Some(&l.k_norm), &mut tmp);
                        hh.copy_from_slice(&tmp);
                        rope(hh);
                    }
                    let vr = unsafe { vp.at(t * w, w) };
                    for hh in vr.chunks_exact_mut(hd) {
                        rms_into(hh, None, &mut tmp);
                        hh.copy_from_slice(&tmp);
                    }
                }
            });
        }
        vprof::add(vprof::NORM_ROPE, t0);
        let t0 = std::time::Instant::now();
        let mut out = vec![0f32; n * w];
        for s in segs {
            let len = s.pw * s.ph;
            let rng = s.off * w..(s.off + len) * w;
            if len >= self.gpu_min_patches
                && self.attend_gpu(
                    &q[rng.clone()],
                    &k[rng.clone()],
                    &v[rng.clone()],
                    len,
                    &mut out[rng.clone()],
                )
            {
                continue;
            }
            attend_host(
                &q[rng.clone()],
                &k[rng.clone()],
                &v[rng.clone()],
                len,
                nh,
                hd,
                &mut out[rng],
                pool,
            );
        }
        vprof::add(vprof::CORE, t0);
        let t0 = std::time::Instant::now();
        let r = l.o.apply(&out, n, pool);
        vprof::add(vprof::O, t0);
        r
    }

    /// All heads of one image on the device (`gpu::dit_attention`, flash
    /// path, scale 1). `false` = refused; the host path runs.
    fn attend_gpu(&self, q: &[f32], k: &[f32], v: &[f32], len: usize, out: &mut [f32]) -> bool {
        if !crate::gpu::enabled_here() {
            return false;
        }
        let (hd, nh) = (self.head_dim, self.heads);
        let w = nh * hd;
        let pack = |x: &[f32]| {
            let mut p = vec![0f32; len * w];
            for t in 0..len {
                for hh in 0..nh {
                    p[hh * len * hd + t * hd..][..hd].copy_from_slice(&x[t * w + hh * hd..][..hd]);
                }
            }
            p
        };
        let (qh, kh, vh) = (pack(q), pack(k), pack(v));
        crate::gpu::dit_attention(&qh, &kh, &vh, nh, nh, len, hd, 1.0, out)
    }
}

/// `C[m×n] = A[m×k] · op(B)` on strided row-major views: `bt` = B is
/// `[n×k]` (used transposed), else `[k×n]`. Accelerate reads the views in
/// place; elsewhere they are gathered for the portable GEMM.
#[allow(clippy::too_many_arguments)]
fn sgemm_view(
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    bt: bool,
    c: &mut [f32],
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_sgemm(
                order: i32,
                ta: i32,
                tb: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
        assert!(a.len() >= (m - 1) * lda + k);
        assert!(if bt {
            b.len() >= (n - 1) * ldb + k
        } else {
            b.len() >= (k - 1) * ldb + n
        });
        assert!(c.len() >= (m - 1) * ldc + n);
        // SAFETY: the views were bounds-checked above; row-major (101).
        unsafe {
            cblas_sgemm(
                101,
                111,
                if bt { 112 } else { 111 },
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                0.0,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    sgemm_view_portable(a, lda, b, ldb, bt, c, ldc, m, n, k);
}

/// [`sgemm_view`] without Accelerate: the views gathered for the portable
/// GEMM (compiled everywhere, so the macOS tests cover it).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn sgemm_view_portable(
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    bt: bool,
    c: &mut [f32],
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let mut ac = vec![0f32; m * k];
    for r in 0..m {
        ac[r * k..(r + 1) * k].copy_from_slice(&a[r * lda..r * lda + k]);
    }
    // the portable GEMM takes B as [n×k]
    let mut bc = vec![0f32; n * k];
    for r in 0..n {
        for j in 0..k {
            bc[r * k + j] = if bt { b[r * ldb + j] } else { b[j * ldb + r] };
        }
    }
    let mut cc = vec![0f32; m * n];
    crate::fcd_ops::gemm_nt_host(&ac, &bc, &mut cc, m, k, n, None);
    for r in 0..m {
        c[r * ldc..r * ldc + n].copy_from_slice(&cc[r * n..(r + 1) * n]);
    }
}

/// Host attention of one image: per head, blocks of queries against every
/// key — one GEMM for the scores, the softmax rows across the pool, one
/// GEMM for the value mix, on the heads' strided views of q/k/v (no
/// gathers). A block holds up to ~8 M scores (32 MB), so a 280-token
/// image's 2304 patches go in one block a head.
///
/// The steps are pipelined: while the pool runs the softmax of block `t`,
/// a second thread computes the scores of block `t + 1` and the value mix
/// of block `t - 1` — the GEMMs (the matrix unit) and the exponentials
/// (the vector units) overlap instead of taking turns.
#[allow(clippy::too_many_arguments)]
fn attend_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    len: usize,
    nh: usize,
    hd: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    let w = nh * hd;
    let qblock = (SCORE_BLOCK / len).clamp(QBLOCK_MIN, len.max(1));
    let nblk = len.div_ceil(qblock);
    let steps = nh * nblk;
    // step -> (head, first row, rows)
    let step = |t: usize| {
        let (h, b) = (t / nblk, t % nblk);
        let i0 = b * qblock;
        (h, i0, (i0 + qblock).min(len) - i0)
    };
    let mut bufs: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; qblock * len]);
    let scores = |t: usize, sc: &mut [f32]| {
        let (h, i0, nb) = step(t);
        sgemm_view(
            &q[i0 * w + h * hd..],
            w,
            &k[h * hd..],
            w,
            true,
            &mut sc[..nb * len],
            len,
            nb,
            len,
            hd,
        );
    };
    let outp = Shared(out.as_mut_ptr());
    let out_len = out.len();
    let mix = |t: usize, sc: &[f32]| {
        let (h, i0, nb) = step(t);
        // SAFETY: steps write disjoint (rows, head) blocks of `out`
        let o = unsafe { outp.at(0, out_len) };
        sgemm_view(
            &sc[..nb * len],
            len,
            &v[h * hd..],
            w,
            false,
            &mut o[i0 * w + h * hd..],
            w,
            nb,
            hd,
            len,
        );
    };
    let soft = |t: usize, sc: &mut [f32]| {
        let (_, _, nb) = step(t);
        let sp = Shared(sc.as_mut_ptr());
        rows(pool, nb, &|s, e| {
            for i in s..e {
                softmax(unsafe { sp.at(i * len, len) });
            }
        });
    };
    if pool.is_none() || steps < 2 {
        let sc = &mut bufs[0];
        for t in 0..steps {
            scores(t, sc);
            soft(t, sc);
            mix(t, sc);
        }
        return;
    }
    scores(0, &mut bufs[0]);
    for t in 0..steps {
        // bufs[t % 3] holds the scores of t; t + 1 goes to the next, t - 1
        // is in the previous
        let [b0, b1, b2] = &mut bufs;
        let (cur, next, prev) = match t % 3 {
            0 => (b0, b1, b2),
            1 => (b1, b2, b0),
            _ => (b2, b0, b1),
        };
        std::thread::scope(|sc| {
            sc.spawn(|| {
                if t + 1 < steps {
                    scores(t + 1, next);
                }
                if t >= 1 {
                    mix(t - 1, prev);
                }
            });
            soft(t, cur);
        });
    }
    mix(steps - 1, &bufs[(steps - 1) % 3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes_match_the_processor() {
        // (h, w, budget) -> (h, w), from the reference manifest
        let cases = [
            ((512, 512, 280), (768, 768)),
            ((756, 1980, 280), (480, 1296)),
            ((941, 530, 280), (1056, 576)),
            ((48, 64, 280), (672, 912)),
            ((512, 512, 70), (384, 384)),
            ((512, 512, 140), (528, 528)),
            ((512, 512, 560), (1104, 1104)),
            ((512, 512, 1120), (1584, 1584)),
            ((360, 640, 140), (384, 720)),
        ];
        for ((h, w, b), want) in cases {
            assert_eq!(
                target_size(h, w, 16, b * 9, 3).unwrap(),
                want,
                "{w}x{h} @ {b}"
            );
        }
        // extreme aspect ratios clamp to one 48-px strip
        let (h, w) = target_size(10, 10_000, 16, 280 * 9, 3).unwrap();
        assert_eq!(h, 48);
        assert!(w <= 280 * 48 && w % 48 == 0);
    }

    #[test]
    fn resize_identity_and_constant() {
        let img = RgbFrame::new(4, 3, (0..36).map(|i| (i * 7) as u8).collect()).unwrap();
        assert_eq!(resize_bicubic_aa(&img, 3, 4), img);
        // a constant image stays constant under any resize
        let c = RgbFrame::new(37, 23, vec![131; 37 * 23 * 3]).unwrap();
        for (h, w) in [(48, 96), (10, 7), (100, 3)] {
            let r = resize_bicubic_aa(&c, h, w);
            assert!(r.data.iter().all(|&v| v == 131), "{w}x{h}");
        }
    }

    #[test]
    fn patchify_orders_rows_then_columns() {
        // 32x16 image: two patches side by side
        let mut data = vec![0u8; 32 * 16 * 3];
        for y in 0..16 {
            for x in 0..32 {
                data[(y * 32 + x) * 3] = if x < 16 { 0 } else { 255 };
            }
        }
        let p = patchify(&RgbFrame::new(32, 16, data).unwrap(), 16);
        assert_eq!((p.pw, p.ph), (2, 1));
        assert_eq!(p.pixels[0], -1.0);
        assert_eq!(p.pixels[768], 1.0);
        assert_eq!(p.pixels[1], -1.0); // green of the first pixel
    }

    #[test]
    fn frames_sample_at_one_fps() {
        let p = VisionProcessor::default();
        // the reference video: 115 frames at 25 fps
        assert_eq!(p.sample_indices(115, Some(25.0)), vec![0, 25, 50, 75]);
        // short clip: one frame
        assert_eq!(p.sample_indices(10, Some(25.0)), vec![0]);
        // frames with no rate: all of them, capped evenly at 32
        let all = p.sample_indices(100, None);
        assert_eq!(all.len(), 32);
        assert_eq!(all[0], 0);
        assert_eq!(*all.last().unwrap(), 99);
        // np.linspace(0, 99, 32, dtype=int)
        assert_eq!(all[1], 3);
        assert_eq!(all[2], 6);
        // a long video at 1 fps is cut to 32 evenly
        let long = p.sample_indices(30 * 100, Some(30.0));
        assert_eq!(long.len(), 32);
        assert_eq!(long[0], 0);
        assert_eq!(*long.last().unwrap(), 99 * 30);
    }

    #[test]
    fn strided_gemm_views_agree() {
        // C = A·Bᵀ and C = A·B on strided views, Accelerate vs the portable
        // gather path vs a scalar reference
        let (m, n, k, lda, ldb, ldc) = (5usize, 7usize, 9usize, 13usize, 11usize, 10usize);
        let a: Vec<f32> = (0..m * lda)
            .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
            .collect();
        for bt in [true, false] {
            let blen = if bt { n * ldb } else { k * ldb };
            let b: Vec<f32> = (0..blen)
                .map(|i| ((i * 5 % 13) as f32 - 6.0) / 6.0)
                .collect();
            let mut want = vec![0f32; m * ldc];
            for r in 0..m {
                for c in 0..n {
                    want[r * ldc + c] = (0..k)
                        .map(|j| a[r * lda + j] * if bt { b[c * ldb + j] } else { b[j * ldb + c] })
                        .sum();
                }
            }
            let mut c1 = vec![0f32; m * ldc];
            sgemm_view(&a, lda, &b, ldb, bt, &mut c1, ldc, m, n, k);
            let mut c2 = vec![0f32; m * ldc];
            sgemm_view_portable(&a, lda, &b, ldb, bt, &mut c2, ldc, m, n, k);
            for r in 0..m {
                for c in 0..n {
                    let w = want[r * ldc + c];
                    assert!((c1[r * ldc + c] - w).abs() < 1e-4, "bt={bt} accel");
                    assert!((c2[r * ldc + c] - w).abs() < 1e-4, "bt={bt} portable");
                }
            }
        }
    }

    #[test]
    fn host_attention_matches_naive() {
        // two heads of 8 over 300 rows: the pipelined block path (with and
        // without a pool) against a direct softmax(q·kᵀ)·v
        let (len, nh, hd) = (300usize, 2usize, 8usize);
        let w = nh * hd;
        let f = |i: usize, s: usize| (((i * s) % 29) as f32 - 14.0) / 20.0;
        let q: Vec<f32> = (0..len * w).map(|i| f(i, 7)).collect();
        let k: Vec<f32> = (0..len * w).map(|i| f(i, 11)).collect();
        let v: Vec<f32> = (0..len * w).map(|i| f(i, 13)).collect();
        let mut want = vec![0f32; len * w];
        for h in 0..nh {
            for i in 0..len {
                let sc: Vec<f64> = (0..len)
                    .map(|j| {
                        (0..hd)
                            .map(|c| (q[i * w + h * hd + c] * k[j * w + h * hd + c]) as f64)
                            .sum()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for c in 0..hd {
                    want[i * w + h * hd + c] = (0..len)
                        .map(|j| e[j] / z * v[j * w + h * hd + c] as f64)
                        .sum::<f64>() as f32;
                }
            }
        }
        let pool = crate::pool::Pool::new(3);
        for p in [None, Some(&pool)] {
            let mut out = vec![0f32; len * w];
            attend_host(&q, &k, &v, len, nh, hd, &mut out, p);
            let err = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "max err {err}");
        }
    }

    #[test]
    fn processor_config_is_read() {
        let cfg = serde_json::json!({
            "image_processor": {"max_soft_tokens": 560, "patch_size": 16, "pooling_kernel_size": 3},
            "video_processor": {"max_soft_tokens": 70, "fps": 2, "max_frames": 8, "overflow_strategy": "truncate"},
        });
        let p = VisionProcessor::from_config(&cfg);
        assert_eq!(p.image_budget, 560);
        assert_eq!(p.video_budget, 70);
        assert_eq!(p.video_fps, 2.0);
        assert_eq!(p.max_frames, 8);
        assert!(p.overflow_truncate);
        assert!(VisionProcessor::check_budget(300).is_err());
    }
}
