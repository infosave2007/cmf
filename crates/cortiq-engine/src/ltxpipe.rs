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
    /// Patch midpoints for a *guide* block of `gf` latent frames placed at
    /// `frame_offset` pixel frames — negative for a reference slot, which is
    /// what puts it before the clip on the time axis. The causal correction
    /// of the first frame is the same one `video_positions` applies; the
    /// offset is added after it, in pixel frames, exactly as the reference
    /// implementation shifts a keyframe's coordinates.
    pub fn guide_positions(&self, gf: usize, frame_offset: i64) -> Vec<Vec<f64>> {
        let causal = |v: f64| (v + 1.0 - SCALE_TIME as f64).max(0.0);
        let off = frame_offset as f64;
        let mut out = Vec::with_capacity(gf * self.tokens_per_frame());
        for f in 0..gf {
            let t0 = (causal((f * SCALE_TIME) as f64) + off) / self.fps;
            let t1 = (causal(((f + 1) * SCALE_TIME) as f64) + off) / self.fps;
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

    /// The same ladder resampled to `steps` rungs. The distilled schedule is
    /// not a discretization of a continuous curve that more steps approximate
    /// better — it is four near-zero moves at the top and three large jumps,
    /// which is what the model was distilled to take. Asking for more steps
    /// puts it on sigmas it never saw, and the usual result is a softer frame,
    /// not a sharper one. The dial exists so that can be measured rather than
    /// argued about; `steps == 8` returns the distilled ladder unchanged, bit
    /// for bit.
    pub fn stage1_steps(steps: usize) -> Stage {
        let base = &STAGE1_SIGMAS;
        let n = steps.max(1);
        if n + 1 == base.len() {
            return Stage::stage1();
        }
        let last = base.len() - 1;
        let sigmas: Vec<f32> = (0..=n)
            .map(|i| {
                let t = i as f64 * last as f64 / n as f64;
                let lo = (t.floor() as usize).min(last);
                let hi = (lo + 1).min(last);
                let f = (t - lo as f64) as f32;
                base[lo] + (base[hi] - base[lo]) * f
            })
            .collect();
        Stage { sigmas, ancestral: true }
    }

    /// `stage2` resampled the same way, for the refinement pass.
    pub fn stage2_steps(steps: usize) -> Stage {
        let base = &STAGE2_SIGMAS;
        let n = steps.max(1);
        if n + 1 == base.len() {
            return Stage::stage2();
        }
        let last = base.len() - 1;
        let sigmas: Vec<f32> = (0..=n)
            .map(|i| {
                let t = i as f64 * last as f64 / n as f64;
                let lo = (t.floor() as usize).min(last);
                let hi = (lo + 1).min(last);
                let f = (t - lo as f64) as f32;
                base[lo] + (base[hi] - base[lo]) * f
            })
            .collect();
        Stage { sigmas, ancestral: false }
    }

    /// The tail of the schedule that starts at or below `strength` — the
    /// video-to-video dial. The clip is re-noised to that level and denoised
    /// from there, so 1.0 keeps only the composition and 0.2 barely touches
    /// it. The first sigma of the returned schedule *is* the noise scale the
    /// starting latent is mixed to, which is the same pairing the reference
    /// uses between its second stage and the latent it upsampled.
    pub fn from_strength(strength: f32) -> Stage {
        let s0 = strength.clamp(0.02, 1.0);
        // Start *at* the level asked for, then follow the distilled ladder
        // down. Filtering the ladder alone would silently start lower than
        // requested — at 0.72 the nearest rung below is 0.42, which is a
        // different edit than the one the caller asked for.
        let mut sigmas = vec![s0];
        sigmas.extend(STAGE1_SIGMAS.iter().copied().filter(|&s| s < s0 && s > 0.0));
        sigmas.push(0.0);
        Stage { sigmas, ancestral: true }
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

/// What is held fixed while the rest is denoised. `mask[t] = 0` freezes
/// token `t` at `clean[t]` and hands the transformer a timestep of zero for
/// it — which is how one encoded image becomes the first frame of a
/// generated shot, and how a whole encoded clip becomes the picture a
/// soundtrack is written for.
#[derive(Clone, Default)]
pub struct Conditioning {
    pub video_mask: Vec<f32>,
    pub video_clean: Vec<f32>,
    pub audio_mask: Vec<f32>,
    pub audio_clean: Vec<f32>,
    /// Extra clean video tokens carried alongside the clip — reference
    /// images, at their own positions on the time axis. They are denoised
    /// with the sequence and dropped from the result.
    pub refs: Option<RefTokens>,
}

/// Reference tokens: already patchified `[count, 128]`, with one position
/// triple each.
#[derive(Clone, Default)]
pub struct RefTokens {
    pub latent: Vec<f32>,
    pub positions: Vec<Vec<f64>>,
    pub count: usize,
}

impl Conditioning {
    /// Freeze the first `frames` latent frames at `clean` (patchified).
    pub fn video_prefix(geo: &Geometry, clean: &[f32], frames: usize) -> Conditioning {
        let per = geo.tokens_per_frame();
        let mut mask = vec![1f32; geo.video_tokens()];
        let mut full = vec![0f32; geo.video_tokens() * 128];
        let n = (frames * per).min(geo.video_tokens());
        for (t, m) in mask.iter_mut().enumerate().take(n) {
            *m = 0.0;
            full[t * 128..(t + 1) * 128].copy_from_slice(&clean[t * 128..(t + 1) * 128]);
        }
        Conditioning { video_mask: mask, video_clean: full, ..Default::default() }
    }

    /// Freeze the whole video stream — the picture is given, the sound is
    /// what is being generated.
    pub fn video_all(geo: &Geometry, clean: &[f32]) -> Conditioning {
        Conditioning {
            video_mask: vec![0f32; geo.video_tokens()],
            video_clean: clean.to_vec(),
            ..Default::default()
        }
    }

    /// Carry reference images beside the clip. They are frozen (a guide is
    /// given, not generated) and cropped off the result, so the render comes
    /// back the size the caller asked for.
    pub fn with_references(mut self, refs: RefTokens) -> Conditioning {
        self.refs = Some(refs);
        self
    }

    /// Freeze the whole soundtrack — the sound is given, the picture is what
    /// is being generated.
    pub fn with_audio_all(mut self, geo: &Geometry, clean: &[f32]) -> Conditioning {
        self.audio_mask = vec![0f32; geo.af];
        self.audio_clean = clean.to_vec();
        self
    }
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
    run_stage_cond(dit, geo, stage, video_ctx, audio_ctx, ctx_len, init, None, rng, pool, progress)
}

#[allow(clippy::too_many_arguments)]
pub fn run_stage_cond(
    dit: &LtxDit,
    geo: &Geometry,
    stage: &Stage,
    video_ctx: &[f32],
    audio_ctx: &[f32],
    ctx_len: usize,
    init: Option<Latents>,
    cond: Option<&Conditioning>,
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

    // conditioning: a frozen token starts clean, stays clean, and is handed
    // a timestep of zero so the modulation treats it as already denoised
    let vmask: Vec<f32> = cond
        .map(|c| c.video_mask.clone())
        .filter(|m| m.len() == vt)
        .unwrap_or_else(|| vec![1f32; vt]);
    let amask: Vec<f32> = cond
        .map(|c| c.audio_mask.clone())
        .filter(|m| m.len() == at)
        .unwrap_or_else(|| vec![1f32; at]);
    let vclean: Vec<f32> = cond
        .map(|c| c.video_clean.clone())
        .filter(|c| c.len() == v.len())
        .unwrap_or_else(|| vec![0f32; v.len()]);
    let aclean: Vec<f32> = cond
        .map(|c| c.audio_clean.clone())
        .filter(|c| c.len() == a.len())
        .unwrap_or_else(|| vec![0f32; a.len()]);
    let blend = |x: &mut [f32], clean: &[f32], mask: &[f32], ch: usize| {
        for (t, &m) in mask.iter().enumerate() {
            if m >= 1.0 {
                continue;
            }
            for d in 0..ch {
                let i = t * ch + d;
                x[i] = clean[i] + (x[i] - clean[i]) * m;
            }
        }
    };
    blend(&mut v, &vclean, &vmask, vch);
    blend(&mut a, &aclean, &amask, ach);

    let mut vpos = geo.video_positions();
    let apos = geo.audio_positions();
    let mut kf = geo.keyframes_mask();

    // Reference tokens ride in the same sequence: clean, frozen, at their own
    // coordinates. The transformer's attention is permutation-invariant apart
    // from RoPE, so appending them is the same operation the reference
    // implementation calls prepending — the position is what carries the
    // meaning, not the index. `vt` grows here and the result is cropped back
    // to `clip_tokens` at the end.
    let clip_tokens = vt;
    let mut vt = vt;
    let mut vmask = vmask;
    let mut vclean = vclean;
    let mut v = v;
    if let Some(r) = cond.and_then(|c| c.refs.as_ref()) {
        if r.count > 0 && r.latent.len() == r.count * vch && r.positions.len() == r.count {
            v.extend_from_slice(&r.latent);
            vclean.extend_from_slice(&r.latent);
            vmask.extend(std::iter::repeat_n(0f32, r.count));
            kf.extend(std::iter::repeat_n(0f32, r.count));
            vpos.extend(r.positions.iter().cloned());
            vt += r.count;
            tracing::info!(
                "reference conditioning: {} tokens beside {clip_tokens} of clip",
                r.count
            );
        } else if r.count > 0 {
            tracing::warn!(
                "reference conditioning ignored: {} tokens, {} latents, {} positions",
                r.count,
                r.latent.len(),
                r.positions.len()
            );
        }
    }
    // The frozen-stream rule is decided on the clip, not on the guides: a
    // render that carries references is still generating its picture, and
    // reading the whole extended mask would call it clean and close the
    // fusion gate on it.
    let v_frozen_src: Vec<f32> = vmask[..clip_tokens].to_vec();
    let steps = stage.sigmas.len() - 1;
    let eta = if stage.ancestral { 1.0 } else { 0.0 };

    // A stream that is frozen everywhere is *clean*, and both its own
    // prompt-adaLN and the other stream's fusion gate must be told so: the
    // gate reads the other side's sigma and closes on noise, so leaving the
    // schedule's sigma there makes the transformer discount a picture it was
    // handed intact. The reference sets it to zero for a frozen modality.
    let v_frozen = v_frozen_src.iter().all(|&m| m == 0.0);
    let a_frozen = amask.iter().all(|&m| m == 0.0);

    for i in 0..steps {
        let t0 = std::time::Instant::now();
        let sigma = stage.sigmas[i];
        let sigma_next = stage.sigmas[i + 1];
        let vin = StreamInput {
            latent: v.clone(),
            tokens: vt,
            timesteps: vmask.iter().map(|m| sigma * m).collect(),
            positions: vpos.clone(),
            context: video_ctx.to_vec(),
            ctx_len,
            context_mask: Vec::new(),
            keyframes: kf.clone(),
            sigma: if v_frozen { 0.0 } else { sigma },
        };
        let ain = StreamInput {
            latent: a.clone(),
            tokens: at,
            timesteps: amask.iter().map(|m| sigma * m).collect(),
            positions: apos.clone(),
            context: audio_ctx.to_vec(),
            ctx_len,
            context_mask: Vec::new(),
            keyframes: Vec::new(),
            sigma: if a_frozen { 0.0 } else { sigma },
        };
        let (vv, av) = dit.forward(&vin, &ain, pool);
        // velocity → denoised, at the token's own timestep
        // velocity → denoised at each token's own timestep, then the frozen
        // tokens are put back exactly as they were
        let mut vd: Vec<f32> = v
            .iter()
            .zip(&vv)
            .enumerate()
            .map(|(i, (&x, &g))| x - g * sigma * vmask[i / vch])
            .collect();
        let mut ad: Vec<f32> = a
            .iter()
            .zip(&av)
            .enumerate()
            .map(|(i, (&x, &g))| x - g * sigma * amask[i / ach])
            .collect();
        blend(&mut vd, &vclean, &vmask, vch);
        blend(&mut ad, &aclean, &amask, ach);
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
        blend(&mut v, &vclean, &vmask, vch);
        blend(&mut a, &aclean, &amask, ach);
        progress(i + 1, steps, t0.elapsed().as_secs_f64());
    }
    v.truncate(clip_tokens * vch);
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

/// A `[128, F, H, W]` volume back to patchified tokens `[T, 128]` — the
/// inverse of [`unpatchify_video`], for feeding a stage its starting latent.
pub fn patchify_video(vol: &[f32], geo: &Geometry) -> Vec<f32> {
    let (lf, lh, lw) = (geo.lf, geo.lh, geo.lw);
    let c = 128usize;
    let mut out = vec![0f32; c * lf * lh * lw];
    for f in 0..lf {
        for h in 0..lh {
            for w in 0..lw {
                let t = (f * lh + h) * lw + w;
                for ch in 0..c {
                    out[t * c + ch] = vol[((ch * lf + f) * lh + h) * lw + w];
                }
            }
        }
    }
    out
}
