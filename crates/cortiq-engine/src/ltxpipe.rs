//! The LTX-2.5 sampler: latent geometry, the noise schedule and the Euler
//! loops that drive [`crate::ltxdit::LtxDit`] from noise to a clean latent.
//!
//! The reference pipeline is two-stage — eight ancestral steps at half
//! resolution, a latent upsample, then three deterministic steps at full
//! resolution. Both stages run the same loop; they differ only in the sigma
//! schedule, whether noise is re-injected, and what the latent starts from.
//!
//! Everything positional is derived here, because the DiT reads positions
//! rather than shapes: video tokens carry `(seconds, pixel row, pixel
//! column)` patch midpoints — the temporal axis divided by the frame rate so
//! it shares a unit with audio — and the first latent frame is shifted by the
//! causal correction, since a causal video encoder gives it one pixel frame
//! where every later latent frame gets eight.

use crate::ltxdit::{LtxDit, StreamInput};
use crate::pool::Pool;

/// Video VAE downscaling: 8 frames, 32 rows, 32 columns per latent step.
pub const SCALE_TIME: usize = 8;
pub const SCALE_SPACE: usize = 32;
/// Audio latents per second: 16000 / 160 / 4.
pub const AUDIO_LATENTS_PER_SEC: f64 = 25.0;
const AUDIO_HOP: f64 = 160.0;
const AUDIO_RATE: f64 = 16000.0;
const AUDIO_DOWNSAMPLE: f64 = 4.0;

/// The distilled schedules the release ships with.
pub const STAGE1_SIGMAS: [f32; 9] =
    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];
pub const STAGE2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// Counter-based RNG with a Box-Muller normal — reproducible from a seed,
/// and independent of any host library.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// One standard normal sample.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
    pub fn fill_normal(&mut self, dst: &mut [f32]) {
        for v in dst.iter_mut() {
            *v = self.normal();
        }
    }
}

/// Latent geometry of one render.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: f64,
    /// Latent frames / rows / columns.
    pub lf: usize,
    pub lh: usize,
    pub lw: usize,
    /// Audio latent frames.
    pub af: usize,
}

impl Geometry {
    pub fn new(frames: usize, height: usize, width: usize, fps: f64) -> Geometry {
        let lf = (frames - 1) / SCALE_TIME + 1;
        let duration = frames as f64 / fps;
        Geometry {
            frames,
            height,
            width,
            fps,
            lf,
            lh: height / SCALE_SPACE,
            lw: width / SCALE_SPACE,
            af: (duration * AUDIO_LATENTS_PER_SEC).round() as usize,
        }
    }

    pub fn video_tokens(&self) -> usize {
        self.lf * self.lh * self.lw
    }

    pub fn tokens_per_frame(&self) -> usize {
        self.lh * self.lw
    }

    /// Patch midpoints in `(seconds, pixel row, pixel column)`, in the
    /// frame-major order `patchify` produces.
    pub fn video_positions(&self) -> Vec<Vec<f64>> {
        let causal = |v: f64| (v + 1.0 - SCALE_TIME as f64).max(0.0);
        let mut out = Vec::with_capacity(self.video_tokens());
        for f in 0..self.lf {
            let t0 = causal((f * SCALE_TIME) as f64) / self.fps;
            let t1 = causal(((f + 1) * SCALE_TIME) as f64) / self.fps;
            let t = (t0 + t1) / 2.0;
            for h in 0..self.lh {
                let y = ((h * SCALE_SPACE) as f64 + ((h + 1) * SCALE_SPACE) as f64) / 2.0;
                for w in 0..self.lw {
                    let x = ((w * SCALE_SPACE) as f64 + ((w + 1) * SCALE_SPACE) as f64) / 2.0;
                    out.push(vec![t, y, x]);
                }
            }
        }
        out
    }

    /// Audio patch midpoints, in seconds.
    pub fn audio_positions(&self) -> Vec<Vec<f64>> {
        let sec = |i: usize| {
            let mel = (i as f64 * AUDIO_DOWNSAMPLE + 1.0 - AUDIO_DOWNSAMPLE).max(0.0);
            mel * AUDIO_HOP / AUDIO_RATE
        };
        (0..self.af).map(|i| vec![(sec(i) + sec(i + 1)) / 2.0]).collect()
    }

    /// Non-zero on the first latent frame, whose latent encodes a single
    /// standalone pixel frame.
    pub fn keyframes_mask(&self) -> Vec<f32> {
        let mut m = vec![0f32; self.video_tokens()];
        for v in m.iter_mut().take(self.tokens_per_frame()) {
            *v = 1.0;
        }
        m
    }
}

/// A denoising stage: which schedule, and whether it re-injects noise.
pub struct Stage {
    pub sigmas: Vec<f32>,
    pub ancestral: bool,
}

impl Stage {
    pub fn stage1() -> Stage {
        Stage { sigmas: STAGE1_SIGMAS.to_vec(), ancestral: true }
    }
    pub fn stage2() -> Stage {
        Stage { sigmas: STAGE2_SIGMAS.to_vec(), ancestral: false }
    }
}

/// One ancestral Euler step in the rectified-flow parameterization
/// (`alpha = 1 - sigma`): advance to `sigma_down`, then renoise back up to
/// `sigma_next` with the variance-preserving rescale. `eta = 0` reduces it to
/// a plain Euler step and ignores `noise`.
fn euler_step(x: &mut [f32], denoised: &[f32], sigma: f32, sigma_next: f32, eta: f32, noise: Option<&[f32]>) {
    if sigma_next == 0.0 {
        x.copy_from_slice(denoised);
        return;
    }
    let down_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta;
    let sigma_down = sigma_next * down_ratio;
    let r = sigma_down / sigma;
    for (v, &d) in x.iter_mut().zip(denoised) {
        *v = r * *v + (1.0 - r) * d;
    }
    if eta > 0.0 {
        let alpha_next = 1.0 - sigma_next;
        let alpha_down = 1.0 - sigma_down;
        let coeff = (sigma_next * sigma_next
            - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down))
            .max(0.0)
            .sqrt();
        let scale = alpha_next / alpha_down;
        let n = noise.expect("ancestral step needs noise");
        for (v, &e) in x.iter_mut().zip(n) {
            *v = scale * *v + e * coeff;
        }
    }
}

/// The state a stage carries: patchified latents for both streams.
pub struct Latents {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// Progress callback: `(step, total, seconds for that step)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, f64);

#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    dit: &LtxDit,
    geo: &Geometry,
    stage: &Stage,
    video_ctx: &[f32],
    audio_ctx: &[f32],
    ctx_len: usize,
    init: Option<Latents>,
    rng: &mut Rng,
    pool: Option<&Pool>,
    progress: Progress<'_>,
) -> Latents {
    let vt = geo.video_tokens();
    let at = geo.af;
    let vch = 128usize;
    let ach = 128usize;
    let s0 = stage.sigmas[0];

    // A fresh stage starts from pure noise; a refinement stage lerps the
    // incoming latent toward noise by the first sigma, exactly as the
    // reference's noiser does.
    let mut v = vec![0f32; vt * vch];
    let mut a = vec![0f32; at * ach];
    rng.fill_normal(&mut v);
    rng.fill_normal(&mut a);
    if let Some(prev) = init {
        for (x, &p) in v.iter_mut().zip(&prev.video) {
            *x = p + (*x - p) * s0;
        }
        for (x, &p) in a.iter_mut().zip(&prev.audio) {
            *x = p + (*x - p) * s0;
        }
    }

    let vpos = geo.video_positions();
    let apos = geo.audio_positions();
    let kf = geo.keyframes_mask();
    let steps = stage.sigmas.len() - 1;
    let eta = if stage.ancestral { 1.0 } else { 0.0 };

    for i in 0..steps {
        let t0 = std::time::Instant::now();
        let sigma = stage.sigmas[i];
        let sigma_next = stage.sigmas[i + 1];
        let vin = StreamInput {
            latent: v.clone(),
            tokens: vt,
            timesteps: vec![sigma; vt],
            positions: vpos.clone(),
            context: video_ctx.to_vec(),
            ctx_len,
            context_mask: Vec::new(),
            keyframes: kf.clone(),
            sigma,
        };
        let ain = StreamInput {
            latent: a.clone(),
            tokens: at,
            timesteps: vec![sigma; at],
            positions: apos.clone(),
            context: audio_ctx.to_vec(),
            ctx_len,
            context_mask: Vec::new(),
            keyframes: Vec::new(),
            sigma,
        };
        let (vv, av) = dit.forward(&vin, &ain, pool);
        // velocity → denoised, at the token's own timestep
        let vd: Vec<f32> = v.iter().zip(&vv).map(|(&x, &g)| x - g * sigma).collect();
        let ad: Vec<f32> = a.iter().zip(&av).map(|(&x, &g)| x - g * sigma).collect();
        let (vn, an) = if eta > 0.0 && sigma_next > 0.0 {
            let mut vn = vec![0f32; v.len()];
            let mut an = vec![0f32; a.len()];
            rng.fill_normal(&mut vn);
            rng.fill_normal(&mut an);
            (Some(vn), Some(an))
        } else {
            (None, None)
        };
        euler_step(&mut v, &vd, sigma, sigma_next, eta, vn.as_deref());
        euler_step(&mut a, &ad, sigma, sigma_next, eta, an.as_deref());
        progress(i + 1, steps, t0.elapsed().as_secs_f64());
    }
    Latents { video: v, audio: a }
}

/// Patchified video tokens `[T, 128]` back to a `[128, F, H, W]` volume.
pub fn unpatchify_video(tokens: &[f32], geo: &Geometry) -> Vec<f32> {
    let (lf, lh, lw) = (geo.lf, geo.lh, geo.lw);
    let c = 128usize;
    let mut out = vec![0f32; c * lf * lh * lw];
    for f in 0..lf {
        for h in 0..lh {
            for w in 0..lw {
                let t = (f * lh + h) * lw + w;
                for ch in 0..c {
                    out[((ch * lf + f) * lh + h) * lw + w] = tokens[t * c + ch];
                }
            }
        }
    }
    out
}

/// Patchified audio tokens `[T, 128]` back to `[8, T, 16]` (channels, time,
/// mel bins) — the layout the audio VAE decodes.
pub fn unpatchify_audio(tokens: &[f32], frames: usize) -> Vec<f32> {
    let (c, mel) = (8usize, 16usize);
    let mut out = vec![0f32; c * frames * mel];
    for t in 0..frames {
        for ch in 0..c {
            for m in 0..mel {
                out[(ch * frames + t) * mel + m] = tokens[t * c * mel + ch * mel + m];
            }
        }
    }
    out
}
