//! End-to-end MiniMax-H3 text→(video + synchronized stereo audio):
//! Qwen3-VL prompt encode → 4-step dual-schedule flow sampling over the
//! packed DiT → ViT3D video decode and BigVGAN audio decode.
//!
//! Stages load and drop one at a time, as the image pipeline next door
//! does: peak resident is one component, not their sum.
//!
//! ## Two clocks, four steps
//!
//! The sampler walks the VIDEO sigma grid — `simple` at shift 12, which
//! at four steps is 1, 0.973, 0.923, 0.8, 0 — and the audio stream is
//! integrated on its own remap of that grid (shift 3). Stepping both on
//! the video grid is what a stock sampler does, and it is fine at
//! twenty steps and audibly wrong at four: `Δσ_a` and `Δσ_v` differ by
//! a factor of three over the last interval, and no per-step slope
//! correction fixes a step that large. Hence `--stock-sampler`, which
//! reproduces the broken behaviour on purpose, for comparison.

use crate::audiovae::AudioVae;
use crate::mmh3::{Layout, MiniMaxH3, time_shift_sigma};
use crate::qwen3te::{ImageSpan, Qwen3Encoder};
use crate::qwen3vis::{self, VisionTower};
use crate::vae3d::VideoVaeEncoder;
use crate::sampler::SplitMix64;
use crate::tokenizer::Tokenizer;
use crate::vae3d::VideoVae;
use std::path::Path;
use std::sync::Arc;

pub const FPS: usize = 24;
pub const AUDIO_LATENT_FPS: usize = 40;

pub struct AnimParams {
    pub width: usize,
    pub height: usize,
    /// Frames at 24 fps; snapped up to the model's 17k+5 grid.
    pub frames: usize,
    pub steps: usize,
    pub seed: u64,
    /// Integrate the audio on the video's grid, as a stock sampler
    /// would. Wrong at four steps; kept for A/B.
    pub stock_sampler: bool,
    pub max_tokens: usize,
    /// RGB in [0, 1] as `[3, h, w]` with its size — the clip's first
    /// frame, and/or its last.
    pub first_frame: Option<(Vec<f32>, usize, usize)>,
    pub last_frame: Option<(Vec<f32>, usize, usize)>,
}

/// The vision-block token ids the H3 presentation flanks a picture with.
const VISION_START: u32 = 151_652;
const VISION_END: u32 = 151_653;

/// Resize `[3, h, w]` RGB to the canvas. The first frame is a geometry
/// anchor and is stretched; the last one follows and is cover-cropped,
/// which is what the reference node does with each.
pub fn fit_to_canvas(
    rgb: &[f32],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    crop: bool,
) -> Vec<f32> {
    // Cover-crop picks the largest centred rectangle of the source with
    // the target's aspect; a stretch takes the whole thing.
    let (sx0, sy0, sw, sh) = if crop {
        let (tw, th) = (out_w as f64, out_h as f64);
        let scale = (w as f64 / tw).min(h as f64 / th);
        let (cw, ch) = ((tw * scale).round() as usize, (th * scale).round() as usize);
        ((w - cw) / 2, (h - ch) / 2, cw.max(1), ch.max(1))
    } else {
        (0, 0, w, h)
    };
    let mut out = vec![0f32; 3 * out_h * out_w];
    for c in 0..3 {
        for y in 0..out_h {
            let sy = ((y as f64 + 0.5) * sh as f64 / out_h as f64 - 0.5).max(0.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let fy = (sy - y0 as f64) as f32;
            for x in 0..out_w {
                let sx = ((x as f64 + 0.5) * sw as f64 / out_w as f64 - 0.5).max(0.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let fx = (sx - x0 as f64) as f32;
                let p = |yy: usize, xx: usize| rgb[(c * h + sy0 + yy) * w + sx0 + xx];
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                out[(c * out_h + y) * out_w + x] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    out
}

impl Default for AnimParams {
    fn default() -> Self {
        Self {
            width: 512,
            height: 288,
            frames: 39,
            steps: 4,
            seed: 42,
            stock_sampler: false,
            max_tokens: 512,
            first_frame: None,
            last_frame: None,
        }
    }
}

/// The rendered result: RGB in [0, 1] as `[3, frames, h, w]`, and
/// stereo f32 in [-1, 1] as `[2, samples]`.
pub struct Anim {
    pub rgb: Vec<f32>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub audio: Vec<f32>,
    pub samples: usize,
    pub sample_rate: usize,
}

/// Frame counts snap UP to 17k+5 — the grid the temporal VAE and the
/// DiT's frame-span pattern agree on.
pub fn align_frames(n: usize) -> usize {
    let mut n = n.max(5);
    while n % 17 != 5 {
        n += 1;
    }
    n
}

pub fn video_latent_t(frames: usize) -> usize {
    if frames <= 5 {
        2
    } else {
        (frames - 5) / 17 * 5 + 2
    }
}

/// `(frames, latent_t, audio_t)` for a requested length.
pub fn temporal_shape(len: usize) -> (usize, usize, usize) {
    let frames = align_frames(len);
    let audio_t =
        ((frames as f64 / FPS as f64) * AUDIO_LATENT_FPS as f64).round() as usize;
    (frames, video_latent_t(frames), audio_t)
}

/// The `simple` scheduler over `ModelSamplingDiscreteFlow(shift)`: the
/// 1000-entry sigma table sampled at even strides, terminal 0 appended.
pub fn sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let table = 1000usize;
    let mut out: Vec<f64> = (0..steps)
        .map(|x| {
            let idx = table - 1 - x * table / steps;
            let t = (idx + 1) as f64 / table as f64;
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect();
    out.push(0.0);
    out
}

/// Shared with the DiT's condition-row noise blend.
pub fn gauss_pub(n: usize, seed: u64) -> Vec<f32> {
    gauss(n, seed)
}

fn gauss(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut u = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let (a, b) = (u().max(1e-300), u());
        let r = (-2.0 * a.ln()).sqrt();
        let ang = 2.0 * std::f64::consts::PI * b;
        out.push((r * ang.cos()) as f32);
        if out.len() < n {
            out.push((r * ang.sin()) as f32);
        }
    }
    out
}

/// Text→(video, audio) from a packaged `.cmf`.
pub fn generate(
    path: &Path,
    prompt: &str,
    p: &AnimParams,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<Anim, String> {
    if p.width % 32 != 0 || p.height % 32 != 0 {
        return Err("width/height must be multiples of 32".into());
    }
    // The wgpu wide-GEMM arm was measured WRONG on one driver stack
    // (RTX PRO 6000: step-1 velocity rms off, step-2 NaN) and byte-
    // healthy on another (2×RTX 5090, coop and plain arms within 0.5%
    // of each other and 3.5% of the host render). Trust is therefore
    // PER-STACK, decided by a parity probe on this file's own first
    // qkv weight at DiT-scale activations — not by a hardcoded verdict
    // either way. CMF_MMH3_GPU=1/0 still forces.
    let use_gpu = match std::env::var("CMF_MMH3_GPU").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => mmh3_gpu_parity_probe(path).unwrap_or(false),
    };
    if use_gpu {
        generate_inner(path, prompt, p, &mut progress)
    } else {
        crate::gpu::cpu_scope(|| generate_inner(path, prompt, p, &mut progress))
    }
}

/// GPU-vs-host parity on the packed DiT's first attention projection:
/// the real q4tp bytes, activations spanning the modulation range
/// (±2000 mixed with ±2), rms gate at 1e-2 — the measured failure was
/// ~24%, honest drift is ~1e-5, so the gate has a decade of margin on
/// each side. Any refusal (no adapter, dtype outside the kernel) is a
/// clean "no": the host path is never wrong, only slower.
fn mmh3_gpu_parity_probe(path: &Path) -> Result<bool, String> {
    let model = Arc::new(
        cortiq_core::CmfModel::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let Some(idx) = model.tensors.iter().position(|t| {
        t.name.starts_with("dit.")
            && t.name.ends_with("attn.qkv_proj.weight")
            && t.dtype == cortiq_core::TensorDtype::Q4TiledP
    }) else {
        tracing::info!("mmh3 GPU parity probe: no q4tp qkv tensor — host path");
        return Ok(false);
    };
    let entry = &model.tensors[idx];
    let (rows, cols) = (entry.shape[0], entry.shape[1]);
    let b = 64usize;
    let mut xs = vec![0f32; b * cols];
    for (i, v) in xs.iter_mut().enumerate() {
        let base = ((i * 37 + 11) % 1009) as f32 / 1009.0 - 0.5;
        *v = base * if i % 7 == 0 { 2000.0 } else { 2.0 };
    }
    let mut gpu = vec![0f32; b * rows];
    if std::env::var("CMF_GPU_DEBUG").is_ok() {
        eprintln!(
            "mmh3 probe env: CMF_GPU={:?} enabled={} avail={}",
            std::env::var("CMF_GPU").ok(),
            crate::gpu::enabled(),
            crate::gpu::backend_available(),
        );
    }
    if !crate::gpu::q4tp_matmat(&model, idx, &xs, b, rows, cols, &mut gpu) {
        tracing::info!(
            "mmh3 GPU parity probe: q4tp_matmat refused ({rows}x{cols}) — host path"
        );
        if std::env::var("CMF_GPU_DEBUG").is_ok() {
            eprintln!("mmh3 probe: q4tp_matmat refused {rows}x{cols}");
        }
        return Ok(false);
    }
    let host = {
        let name = entry.name.clone();
        let proj = crate::dit::Proj::from_model(&model, &name)?;
        let mut out = vec![0f32; b * rows];
        crate::gpu::cpu_scope(|| proj.matmat(&xs, b, &mut out, None));
        out
    };
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, h) in gpu.iter().zip(&host) {
        num += ((g - h) as f64).powi(2);
        den += (*h as f64).powi(2);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    let ok = rel < 1e-2;
    tracing::info!(
        "mmh3 GPU parity probe: rel rms {rel:.2e} → {}",
        if ok { "device" } else { "host" }
    );
    if std::env::var("CMF_GPU_DEBUG").is_ok() {
        eprintln!("mmh3 GPU parity probe: rel rms {rel:.2e}");
    }
    Ok(ok)
}

fn generate_inner(
    path: &Path,
    prompt: &str,
    p: &AnimParams,
    progress: &mut dyn FnMut(&str, usize, usize),
) -> Result<Anim, String> {
    let model = Arc::new(
        cortiq_core::CmfModel::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let (frames_total, latent_t, audio_t) = temporal_shape(p.frames);
    let (lat_h, lat_w) = (p.height / 16, p.width / 16);

    // ── prompt ──
    // The H3 presentation is raw text: no chat template, no BOS, no
    // special tokens at all.
    let vocab = model
        .vocab
        .as_deref()
        .ok_or("packaged .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    // fl2va: every keyframe is presented as "<Picture i>: " and a
    // vision block BEFORE the prompt, and separately conditions the DiT
    // as a latent. Both halves come from the same picture.
    let keyframes: Vec<(&(Vec<f32>, usize, usize), usize)> = p
        .first_frame
        .iter()
        .map(|f| (f, 0usize))
        .chain(p.last_frame.iter().map(|f| (f, frames_total - 1)))
        .collect();
    let mut ids: Vec<u32> = Vec::new();
    let mut spans: Vec<ImageSpan> = Vec::new();
    let mut embeds: Vec<Vec<f32>> = Vec::new();
    let mut deepstack: Vec<Vec<f32>> = Vec::new();
    let mut cond: Vec<Vec<f32>> = Vec::new();
    let mut tags: Vec<u8> = Vec::new();

    if !keyframes.is_empty() {
        let tower = VisionTower::from_cmf(&model)?;
        let venc = VideoVaeEncoder::from_cmf(&model)?;
        for (i, (frame, _)) in keyframes.iter().enumerate() {
            let (src, sh, sw) = *frame;
            // The picture the DiT sees is on the generation canvas; the
            // one Qwen sees keeps its own resolution policy.
            let fitted = fit_to_canvas(src, *sh, *sw, p.height, p.width, i > 0);
            let (z, _, _) = venc.encode_frame(
                &fitted.iter().map(|&v| v * 2.0 - 1.0).collect::<Vec<_>>(),
                p.height,
                p.width,
            );
            cond.push(z);

            for t in tok.encode(&format!("<Picture {}>: ", i + 1)) {
                ids.push(t);
                tags.push(1);
            }
            let (patches, gh, gw) = qwen3vis::preprocess(
                &fitted, p.height, p.width,
                tower.patch_size, tower.temporal_patch, tower.merge,
            );
            let (merged, deep) = tower.forward(&patches, gh, gw);
            let n_img = merged.len() / tower.out_hidden;
            // The whole block carries the VIDEO tag, the flanking
            // markers included.
            ids.push(VISION_START);
            tags.push(0);
            let start = ids.len();
            for _ in 0..n_img {
                ids.push(VISION_START); // a placeholder the embed replaces
                tags.push(0);
            }
            ids.push(VISION_END);
            tags.push(0);
            spans.push(ImageSpan { start, len: n_img, merged_h: gh / tower.merge, merged_w: gw / tower.merge });
            embeds.push(merged);
            if deepstack.is_empty() {
                deepstack = deep;
            } else {
                for (a, b) in deepstack.iter_mut().zip(deep) {
                    a.extend_from_slice(&b);
                }
            }
        }
    }
    for t in tok.encode(prompt) {
        ids.push(t);
        tags.push(1);
    }
    if ids.is_empty() {
        ids.push(151643); // the pad id, as the reference does for ""
        tags.push(1);
    }
    ids.truncate(p.max_tokens);
    tags.truncate(ids.len());
    progress("encode", 0, 1);
    let states = {
        let enc = Qwen3Encoder::from_cmf(&model)?;
        enc.encode_with_images(&ids, &spans, &embeds, &deepstack)
    };
    progress("encode", 1, 1);

    // ── denoise ──
    let (video, audio) = {
        let dit = MiniMaxH3::from_cmf(&model)?;
        let kf: Vec<(usize, usize)> = keyframes.iter().map(|&(_, idx)| (idx, frames_total)).collect();
        let layout = if kf.is_empty() {
            Layout::t2va(ids.len(), latent_t, lat_h, lat_w, audio_t)
        } else {
            Layout::fl2va(ids.len(), latent_t, lat_h, lat_w, audio_t, &kf, &tags)
        };
        let text = dit.refine_text(&states, ids.len());
        let mut v = gauss(dit.latents_dim * latent_t * lat_h * lat_w, p.seed);
        let mut a = gauss(dit.audio_dim * 2 * audio_t, p.seed ^ 0x9E37_79B9_7F4A_7C15);
        let sg = sigmas(p.steps, dit.shift_video);
        // `CMF_ANIM_PROF=1`: the per-step rms of both streams and of
        // their velocities. A run that is not denoising shows it here
        // long before anything is written out.
        let prof = std::env::var_os("CMF_ANIM_PROF").is_some();
        let rms = |x: &[f32]| {
            (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
        };
        if prof {
            eprintln!(
                "  text {} tok, refined rms {:.4}, sigmas {:?}",
                ids.len(),
                rms(&text),
                sg.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
            );
        }
        for i in 0..p.steps {
            let (sv, sv_n) = (sg[i], sg[i + 1]);
            let (dv, da) = dit.forward(&layout, &text, &v, &a, sv, &cond);
            let step_v = (sv_n - sv) as f32;
            for (x, &d) in v.iter_mut().zip(&dv) {
                *x += step_v * d;
            }
            let step_a = if p.stock_sampler {
                step_v
            } else {
                (time_shift_sigma(sv_n, dit.shift_video, dit.shift_audio)
                    - time_shift_sigma(sv.max(1e-6), dit.shift_video, dit.shift_audio))
                    as f32
            };
            for (x, &d) in a.iter_mut().zip(&da) {
                *x += step_a * d;
            }
            if prof {
                eprintln!(
                    "  step {i}: sv {sv:.4}->{sv_n:.4} v_vel {:.4} a_vel {:.4} | video {:.4} audio {:.4}",
                    rms(&dv), rms(&da), rms(&v), rms(&a)
                );
            }
            progress("denoise", i + 1, p.steps);
        }
        (v, a)
    };

    // ── decode ──
    progress("video vae", 0, 1);
    let (rgb, out_frames) = {
        let vae = VideoVae::from_cmf(&model)?;
        vae.decode(&video, latent_t, lat_h, lat_w)
    };
    progress("video vae", 1, 1);
    progress("audio vae", 0, 1);
    let (wave, samples, sr) = {
        let vae = AudioVae::from_cmf(&model)?;
        let c = audio.len() / (2 * audio_t);
        let (w, n) = vae.decode(&audio, c, audio_t);
        (w, n, vae.sample_rate)
    };
    progress("audio vae", 1, 1);

    // The VAE emits latent_t·4 frames; the request snapped to 17k+5,
    // which is one fewer than a multiple of four plus the leading key
    // frame, so trim rather than pad.
    let keep = out_frames.min(frames_total);
    Ok(Anim {
        rgb: trim_frames(&rgb, out_frames, keep, p.height, p.width),
        frames: keep,
        height: p.height,
        width: p.width,
        audio: wave,
        samples,
        sample_rate: sr,
    })
}

fn trim_frames(rgb: &[f32], have: usize, keep: usize, h: usize, w: usize) -> Vec<f32> {
    if keep == have {
        return rgb.to_vec();
    }
    let mut out = vec![0f32; 3 * keep * h * w];
    for c in 0..3 {
        let s = c * have * h * w;
        let d = c * keep * h * w;
        out[d..d + keep * h * w].copy_from_slice(&rgb[s..s + keep * h * w]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_step_schedule_is_the_references() {
        let s = sigmas(4, 12.0);
        let want = [1.0, 0.972_973, 0.923_077, 0.8, 0.0];
        assert_eq!(s.len(), want.len());
        for (g, w) in s.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{s:?}");
        }
    }

    #[test]
    fn a_stretch_keeps_the_corners_and_a_crop_takes_the_middle() {
        // A 4x2 ramp: value rises left to right, so the corners name
        // themselves.
        let (h, w) = (2usize, 4usize);
        let mut rgb = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    rgb[(c * h + y) * w + x] = x as f32 / (w - 1) as f32;
                }
            }
        }
        // Stretch to a square: the far edges survive.
        let s = fit_to_canvas(&rgb, h, w, 4, 4, false);
        assert!((s[0] - 0.0).abs() < 1e-6, "left edge");
        assert!((s[3] - 1.0).abs() < 1e-6, "right edge");
        // Cover-crop to a square takes the centre 2x2, so the extremes
        // are gone and the span is narrower.
        let c = fit_to_canvas(&rgb, h, w, 4, 4, true);
        let (lo, hi) = c[..16]
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(lo > 0.05, "crop kept the left edge: {lo}");
        assert!(hi < 0.95, "crop kept the right edge: {hi}");
    }

    #[test]
    fn frame_counts_snap_to_the_models_grid() {
        // The grid is 5 + 17k: 5, 22, 39, 56, … 124.
        assert_eq!(align_frames(1), 5);
        assert_eq!(align_frames(39), 39);
        assert_eq!(align_frames(41), 56);
        assert_eq!(align_frames(124), 124);
        assert_eq!(video_latent_t(124), 37);
        assert_eq!(video_latent_t(39), 12);
        let (f, lt, at) = temporal_shape(124);
        assert_eq!((f, lt, at), (124, 37, 207));
    }
}
