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
use crate::qwen3te::Qwen3Encoder;
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
    // The wgpu wide-GEMM arm is WRONG on this stack: measured on an RTX
    // PRO 6000, the DiT's first step already disagrees with the host
    // (video velocity rms 1.32 against 1.73, audio 0.16 against 1.01)
    // and the second returns NaN. The op probe picks that arm on its
    // own because it is three times faster, so this pipeline runs pure
    // host unless the caller says otherwise — a wrong answer produced
    // quickly is not a faster answer. `CMF_MMH3_GPU=1` opts in.
    let force_gpu = std::env::var("CMF_MMH3_GPU").ok().as_deref() == Some("1");
    if force_gpu {
        generate_inner(path, prompt, p, &mut progress)
    } else {
        crate::gpu::cpu_scope(|| generate_inner(path, prompt, p, &mut progress))
    }
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
    let (frames, latent_t, audio_t) = temporal_shape(p.frames);
    let (lat_h, lat_w) = (p.height / 16, p.width / 16);

    // ── prompt ──
    // The H3 presentation is raw text: no chat template, no BOS, no
    // special tokens at all.
    let vocab = model
        .vocab
        .as_deref()
        .ok_or("packaged .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    let mut ids = tok.encode(prompt);
    if ids.is_empty() {
        ids.push(151643); // the pad id, as the reference does for ""
    }
    ids.truncate(p.max_tokens);
    progress("encode", 0, 1);
    let states = {
        let enc = Qwen3Encoder::from_cmf(&model)?;
        enc.encode(&ids)
    };
    progress("encode", 1, 1);

    // ── denoise ──
    let (video, audio) = {
        let dit = MiniMaxH3::from_cmf(&model)?;
        let layout = Layout::t2va(ids.len(), latent_t, lat_h, lat_w, audio_t);
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
            let (dv, da) = dit.forward(&layout, &text, &v, &a, sv);
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
    let keep = out_frames.min(frames);
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
