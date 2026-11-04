//! Kandinsky S2V: reference frame + speech track → avatar video.
//!
//! The pipeline of `KandinskyS2VDistilledAutoBlocks`: Thinker text and audio
//! features plus CLIP's pooled embedding → the reference frame through the VAE
//! encoder → per chunk, fresh noise, 16 pi-Flow segments over the DiT, the
//! chunk's tail rolled into FramePack memory → the temporally tiled VAE
//! decoder, the chunks joined and cut to the length of the track.
//!
//! pi-Flow (the distilled head): one forward predicts `n_grid` x0 values across
//! a segment of raw time and the ODE is integrated across that segment with no
//! further network evaluations, so one segment costs one forward.

use crate::ltxvae::Vol;
use crate::s2v::{S2vConfig, S2vDit, S2vGeom, S2vInput};
use crate::s2v_enc::{ClipText, OmniAudio, OmniText, omni_mel};
use crate::s2v_vae::{S2vVae, S2vVaeEncoder, SCALING_FACTOR, TEMPORAL};
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub use crate::s2v::ARCH_NAME;

pub const SAMPLE_RATE: usize = 16000;
pub const AUDIO_TOKENS_PER_SECOND: usize = 25;

/// The recipe stored in the container (`s2v.config_json`).
#[derive(Clone, Debug)]
pub struct S2vDefaults {
    pub steps: usize,
    pub fps: usize,
    pub latent_frames: usize,
    pub seed: u64,
    pub buckets: Vec<(usize, usize)>,
    pub scale_factor: [f32; 3],
    pub prompt: String,
    pub system_prompt: String,
    pub shift: f32,
    pub n_grid: usize,
    pub eps: f32,
    pub final_scale: f32,
    pub substeps: usize,
}

impl S2vDefaults {
    pub fn of(model: &CmfModel) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_slice(model.tensor_bytes("s2v.config_json").map_err(|e| e.to_string())?)
            .map_err(|e| format!("s2v.config_json: {e}"))?;
        let u = |k: &str, d: u64| v[k].as_u64().unwrap_or(d) as usize;
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d) as f32;
        let sf = v["scale_factor"].as_array().ok_or("s2v.config_json: no scale_factor")?;
        Ok(Self {
            steps: u("steps", 16),
            fps: u("fps", 25),
            latent_frames: u("latent_frames", 32),
            seed: v["seed"].as_u64().unwrap_or(5),
            buckets: v["buckets"]
                .as_array()
                .ok_or("s2v.config_json: no buckets")?
                .iter()
                .filter_map(|b| Some((b[0].as_u64()? as usize, b[1].as_u64()? as usize)))
                .collect(),
            scale_factor: [
                sf[0].as_f64().unwrap_or(1.0) as f32,
                sf[1].as_f64().unwrap_or(2.0) as f32,
                sf[2].as_f64().unwrap_or(2.0) as f32,
            ],
            prompt: v["prompt"].as_str().unwrap_or("").to_string(),
            system_prompt: v["system_prompt"].as_str().unwrap_or("").to_string(),
            shift: f("shift", 10.0),
            n_grid: u("n_grid", 5),
            eps: f("eps", 1e-6),
            final_scale: f("final_step_size_scale", 0.5),
            substeps: u("num_policy_substeps", 128),
        })
    }
}

/// `(height, width)` of the bucket closest in aspect ratio to the image.
pub fn select_bucket(buckets: &[(usize, usize)], w: usize, h: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let mut best = buckets[0];
    let mut d = f64::MAX;
    for &b in buckets {
        let e = (aspect - b.1 as f64 / b.0 as f64).abs();
        if e < d {
            d = e;
            best = b;
        }
    }
    best
}

fn shift_t(t: f32, shift: f32) -> f32 {
    shift * t / (1.0 + (shift - 1.0) * t)
}

/// The segment schedule of `PiflowScheduler.set_timesteps`, walked one segment
/// at a time in f32, as the sampler the students were trained against does.
pub struct Piflow {
    pub shift: f32,
    pub eps: f32,
    pub n_grid: usize,
    pub substeps: usize,
    pub raw: Vec<f32>,
    pub segments: Vec<f32>,
    pub sigmas: Vec<f32>,
}

impl Piflow {
    pub fn new(d: &S2vDefaults, steps: usize) -> Self {
        let base = 1.0f64 / (steps as f64 - (1.0 - d.final_scale as f64));
        let mut raw = Vec::with_capacity(steps);
        let mut segments = Vec::with_capacity(steps);
        let mut r = 1.0f32;
        for s in 0..steps {
            let seg = base * if s == steps - 1 { d.final_scale as f64 } else { 1.0 };
            raw.push(r);
            segments.push(seg as f32);
            r = (r - seg as f32).max(d.eps);
        }
        let sigmas = raw.iter().map(|&t| shift_t(t, d.shift)).collect();
        Self { shift: d.shift, eps: d.eps, n_grid: d.n_grid, substeps: d.substeps, raw, segments, sigmas }
    }

    pub fn timesteps(&self) -> Vec<f32> {
        self.sigmas.iter().map(|&s| s * 1000.0).collect()
    }

    /// One segment: `pred` is `[T, H, W, n_grid·C]` grid-major, `x` `[T, H, W, C]`.
    pub fn step(&self, i: usize, pred: &[f32], x: &mut [f32], c: usize, pool: Option<&crate::pool::Pool>) {
        let g = self.n_grid;
        let sigma_src = self.sigmas[i];
        let unwarp = |s: f32| s / (self.shift + (1.0 - self.shift) * s);
        let raw_src_p = unwarp(sigma_src);
        let raw_dst_p = (raw_src_p - self.segments[i]).max(0.0);
        let seg_p = (raw_src_p - raw_dst_p).max(self.eps);
        let raw_end = (self.raw[i] - self.segments[i]).max(self.eps);
        let delta = self.raw[i] - raw_end;
        let n = ((delta * self.substeps as f32).round() as i64).max(1) as usize;
        let sub = delta / n as f32;
        // Every voxel shares the schedule, so each substep is two scalars and
        // one pair of grid rows: x += (x - x0)·dσ/σ.
        let mut plan = Vec::with_capacity(n);
        let (mut raw, mut sigma) = (self.raw[i], sigma_src);
        for _ in 0..n {
            let r = unwarp(sigma);
            let t = ((r - raw_dst_p) / seg_p).clamp(0.0, 1.0) * (g - 1) as f32;
            let t0 = (t.floor() as i64).clamp(0, g as i64 - 2) as usize;
            let t1 = t0 + 1;
            let raw_m = (raw - sub).max(0.0);
            let sigma_m = shift_t(raw_m, self.shift);
            plan.push((t0, t1 as f32 - t, t - t0 as f32, sigma.max(self.eps), sigma_m - sigma));
            raw = raw_m;
            sigma = sigma_m;
        }
        let rows = x.len() / c;
        let xp = crate::s2v::Rows(x.as_mut_ptr());
        crate::s2v::pool_rows(pool, rows, &|lo, hi| {
            let mut x0 = vec![0f32; g];
            for r in lo..hi {
                // SAFETY: disjoint rows.
                let xr = unsafe { xp.at(r * c, c) };
                let pr = &pred[r * g * c..(r + 1) * g * c];
                for ch in 0..c {
                    let x_src = xr[ch];
                    for (gi, v) in x0.iter_mut().enumerate() {
                        *v = x_src - sigma_src * pr[gi * c + ch];
                    }
                    let mut v = x_src;
                    for &(t0, a, b, sig, ds) in &plan {
                        let target = a * x0[t0] + b * x0[t0 + 1];
                        v += (v - target) / sig * ds;
                    }
                    xr[ch] = v;
                }
            }
        });
    }
}

/// PIL's BILINEAR `resize` (antialiased triangle filter, 8-bit fixed point),
/// which `torchvision.transforms.functional.resize` calls on a PIL image.
pub fn pil_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    const PREC: i64 = 22;
    let coeffs = |ins: usize, outs: usize| -> (Vec<(usize, usize)>, Vec<Vec<i64>>) {
        let scale = ins as f64 / outs as f64;
        let filterscale = scale.max(1.0);
        let support = filterscale;
        let mut bounds = Vec::with_capacity(outs);
        let mut kk = Vec::with_capacity(outs);
        for xx in 0..outs {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
            let xmax = ((center + support + 0.5).floor() as i64).min(ins as i64) as usize;
            let mut w = Vec::with_capacity(xmax - xmin);
            let mut sum = 0.0;
            for x in xmin..xmax {
                let d = (x as f64 - center + 0.5) / filterscale;
                let v = if d.abs() < 1.0 { 1.0 - d.abs() } else { 0.0 };
                w.push(v);
                sum += v;
            }
            let fixed: Vec<i64> = w
                .iter()
                .map(|v| {
                    let n = v / sum * (1i64 << PREC) as f64;
                    (if n < 0.0 { n - 0.5 } else { n + 0.5 }) as i64
                })
                .collect();
            bounds.push((xmin, xmax - xmin));
            kk.push(fixed);
        }
        (bounds, kk)
    };
    let clip = |v: i64| ((v >> PREC).clamp(0, 255)) as u8;
    let round = 1i64 << (PREC - 1);
    let (bx, kx) = coeffs(sw, dw);
    let mut mid = vec![0u8; dw * sh * 3];
    for y in 0..sh {
        for x in 0..dw {
            let (x0, n) = bx[x];
            for c in 0..3 {
                let mut ss = round;
                for j in 0..n {
                    ss += src[(y * sw + x0 + j) * 3 + c] as i64 * kx[x][j];
                }
                mid[(y * dw + x) * 3 + c] = clip(ss);
            }
        }
    }
    let (by, ky) = coeffs(sh, dh);
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let (y0, n) = by[y];
        for x in 0..dw * 3 {
            let mut ss = round;
            for j in 0..n {
                ss += mid[(y0 + j) * dw * 3 + x] as i64 * ky[y][j];
            }
            out[y * dw * 3 + x] = clip(ss);
        }
    }
    out
}

/// The reference frame as `[3, 1, H, W]` in [-1, 1]: bilinear to cover the
/// bucket, then a centered crop.
pub fn prepare_image(rgb: &[u8], w: usize, h: usize, height: usize, width: usize) -> Vol {
    let scale = (width as f64 / w as f64).max(height as f64 / h as f64);
    let (rw, rh) = (
        ((w as f64 * scale).round_ties_even() as usize),
        ((h as f64 * scale).round_ties_even() as usize),
    );
    let resized = pil_bilinear(rgb, w, h, rw, rh);
    let top = ((rh - height) as f64 / 2.0).round_ties_even() as usize;
    let left = ((rw - width) as f64 / 2.0).round_ties_even() as usize;
    let mut v = Vol::zeros(3, 1, height, width);
    for c in 0..3 {
        for r in 0..height {
            for col in 0..width {
                let s = resized[((r + top) * rw + col + left) * 3 + c] as f32 / 255.0;
                v.data[c * height * width + r * width + col] = s * 2.0 - 1.0;
            }
        }
    }
    v
}

/// Everything that conditions the DiT, whichever path computes the latents.
pub struct S2vCond {
    pub text: Vec<f32>,
    pub pooled: Vec<f32>,
    pub audio: Vec<Vec<f32>>,
    pub reference: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// The chat string of `encode_qwen_omni`, cut before the audio.
pub fn prompt_ids(tok: &Tokenizer, system: &str, prompt: &str) -> (Vec<u32>, usize) {
    let head = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n");
    let head_ids = tok.encode(&head);
    let body = tok.encode(prompt);
    let start = head_ids.len();
    let mut ids = head_ids;
    ids.extend(body);
    (ids, start)
}

/// `FramePack` memory: zero for the first chunk, `[1×, 2×, 4×]` order.
pub fn init_memory(cfg: &S2vConfig, lh: usize, lw: usize) -> [Vec<f32>; 3] {
    let c = cfg.in_visual;
    let dim = |m: usize| (lh - lh % m, lw - lw % m);
    let b = cfg.frame_buckets;
    let (h1, w1) = dim(2);
    let (h2, w2) = dim(4);
    let (h4, w4) = dim(8);
    [
        vec![0f32; b[0] * h1 * w1 * c],
        vec![0f32; b[1] * h2 * w2 * c],
        vec![0f32; b[2] * h4 * w4 * c],
    ]
}

/// The tail of a chunk becomes the next chunk's memory: newest two frames at
/// 1×, the four before them at 2×, the 24 before those at 4×.
pub fn roll_memory(cfg: &S2vConfig, g: &S2vGeom, x: &[f32]) -> [Vec<f32>; 3] {
    let c = cfg.in_visual;
    let b = cfg.frame_buckets;
    let t = g.lt;
    let mut off = 0usize;
    let mut out: [Vec<f32>; 3] = Default::default();
    for (i, slot) in out.iter_mut().enumerate() {
        let m = 2usize << i;
        let (hh, ww) = (g.lh - g.lh % m, g.lw - g.lw % m);
        let (a, e) = (t - off - b[i], t - off);
        let mut v = Vec::with_capacity((e - a) * hh * ww * c);
        for f in a..e {
            for r in 0..hh {
                let s = ((f * g.lh + r) * g.lw) * c;
                v.extend_from_slice(&x[s..s + ww * c]);
            }
        }
        *slot = v;
        off += b[i];
    }
    out
}

/// A progress line, stage + counters.
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize);

pub struct S2vParams {
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub steps: Option<usize>,
    pub chunks: Option<usize>,
    pub seed: u64,
    pub prompt: Option<String>,
    /// Raw f32 reference posterior noise and per-chunk noise, parity work only.
    pub oracle_noise: Option<(Vec<f32>, Vec<Vec<f32>>)>,
    /// Fix the conditioning to oracle tensors (text, audio, pooled), parity only.
    pub oracle_cond: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub max_steps: Option<usize>,
}

pub struct S2vOutput {
    /// `[3, F, H, W]` in [-1, 1], trimmed to the track.
    pub frames: Vol,
    pub fps: usize,
    pub latents: Vec<Vec<f32>>,
    pub lt: usize,
    pub lh: usize,
    pub lw: usize,
}

struct SplitMix(crate::ltxpipe::Rng);
impl SplitMix {
    fn normal_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        self.0.fill_normal(&mut v);
        v
    }
}

/// Read a speech track as mono 16 kHz: WAV here, anything else through ffmpeg.
pub fn load_speech(path: &Path) -> Result<Vec<f32>, String> {
    crate::egemma2_audio::read_audio(path)
}

/// `reference` latent `[H, W, C]` from the posterior: mean + std·ε, scaled.
pub fn reference_latent(mean: &Vol, logvar: &Vol, eps: &[f32]) -> Vec<f32> {
    let (h, w, c) = (mean.h, mean.w, mean.c);
    let mut out = vec![0f32; h * w * c];
    for ch in 0..c {
        for p in 0..h * w {
            let m = mean.data[ch * h * w + p];
            let s = (0.5 * logvar.data[ch * h * w + p]).exp();
            out[p * c + ch] = (m + s * eps[ch * h * w + p]) * SCALING_FACTOR;
        }
    }
    out
}

/// `[T, H, W, C]` latents → `[C, T, H, W]`, unscaled, for the decoder.
pub fn to_vol(x: &[f32], t: usize, h: usize, w: usize, c: usize) -> Vol {
    let mut v = Vol::zeros(c, t, h, w);
    let n = t * h * w;
    for p in 0..n {
        for ch in 0..c {
            v.data[ch * n + p] = x[p * c + ch] / SCALING_FACTOR;
        }
    }
    v
}

/// Thinker hidden states of the user's text and CLIP's pooled embedding.
pub fn encode_prompt(model: &Arc<CmfModel>, d: &S2vDefaults, prompt: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    let vocab = model.vocab.as_deref().ok_or("S2V .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    let (ids, start) = prompt_ids(&tok, &d.system_prompt, prompt);
    let text = {
        let te = OmniText::from_cmf(model)?;
        let h = te.encode(&ids);
        h[start * te.hidden..].to_vec()
    };
    let clip_vocab = model.tensor_bytes("clip.tokenizer_json").map_err(|e| e.to_string())?;
    let clip_tok = Tokenizer::from_bytes(clip_vocab).map_err(|e| format!("clip tokenizer: {e}"))?;
    let ct = ClipText::from_cmf(model)?;
    let mut cids = clip_tok.encode(prompt);
    const BOS: u32 = 49406;
    const EOS: u32 = 49407;
    if cids.first() != Some(&BOS) {
        cids.insert(0, BOS);
    }
    cids.truncate(76);
    cids.push(EOS);
    cids.resize(77, EOS);
    Ok((text, ct.pooled(&cids, EOS)))
}

/// Text, speech and reference latent. Each model loads and is dropped.
pub fn encode_conditions(
    model: &Arc<CmfModel>,
    d: &S2vDefaults,
    image: &crate::media::RgbFrame,
    wav: &[f32],
    chunks: usize,
    p: &S2vParams,
    rng_eps: &mut dyn FnMut(usize) -> Vec<f32>,
    progress: Progress<'_>,
) -> Result<S2vCond, String> {
    let chunk_seconds = ((d.latent_frames - 1) * TEMPORAL + 1) as f64 / d.fps as f64;
    let target = chunks as f64 * chunk_seconds;
    let samples = (target * SAMPLE_RATE as f64).round() as usize;
    let mut speech = wav.to_vec();
    speech.resize(samples, 0.0);

    let (text, audio, pooled) = if let Some((t, a, po)) = &p.oracle_cond {
        (t.clone(), a.clone(), po.clone())
    } else {
        let t0 = Instant::now();
        progress("audio tower", 0, 1);
        let (mel, frames) = omni_mel(&speech);
        let audio = {
            let at = OmniAudio::from_cmf(model)?;
            at.features(&mel, frames)
        };
        prof("audio", t0);
        let t0 = Instant::now();
        // The canned prompt's conditions are in the file; the 7B Thinker is only
        // opened for another prompt.
        let canned = model.tensor("s2v.default_text").is_some() && p.prompt.as_ref().is_none_or(|q| *q == d.prompt);
        let (text, pooled) = if canned {
            (crate::dit::cmf_f32(model, "s2v.default_text")?, crate::dit::cmf_f32(model, "s2v.default_pooled")?)
        } else {
            progress("text encoder", 0, 1);
            encode_prompt(model, d, p.prompt.as_deref().unwrap_or(&d.prompt))?
        };
        prof("prompt", t0);
        (text, audio, pooled)
    };
    let tokens = audio.len() / 3584;
    if tokens < (target * AUDIO_TOKENS_PER_SECOND as f64 * 0.95) as usize {
        return Err(format!("the speech encoder returned {tokens} tokens for {target:.1} s"));
    }
    let per = tokens / chunks;
    let audio_chunks: Vec<Vec<f32>> = (0..chunks).map(|k| audio[k * per * 3584..(k + 1) * per * 3584].to_vec()).collect();

    let (height, width) = match (p.height, p.width) {
        (Some(h), Some(w)) => (h, w),
        _ => select_bucket(&d.buckets, image.width, image.height),
    };
    let t0 = Instant::now();
    progress("reference frame", 0, 1);
    let img = prepare_image(&image.data, image.width, image.height, height, width);
    let (mean, logvar) = {
        let enc = S2vVaeEncoder::from_cmf(model)?;
        crate::gpu::s2v_vae_encode(&enc, &img).unwrap_or_else(|| enc.encode(&img))
    };
    prof("vae encode", t0);
    let eps = rng_eps(mean.data.len());
    Ok(S2vCond {
        text,
        pooled,
        audio: audio_chunks,
        reference: reference_latent(&mean, &logvar, &eps),
        height,
        width,
    })
}

pub(crate) fn prof(what: &str, t0: Instant) {
    if std::env::var("CMF_S2V_PROF").is_ok() {
        eprintln!("s2v {what}: {:.2} s", t0.elapsed().as_secs_f64());
    }
}

/// One whole render.
pub fn generate(
    model_path: &Path,
    image: &crate::media::RgbFrame,
    wav: &[f32],
    p: &S2vParams,
    mut progress: Progress<'_>,
) -> Result<S2vOutput, String> {
    let model = Arc::new(CmfModel::open_sharded(model_path).map_err(|e| e.to_string())?);
    if model.header.arch.arch_name != ARCH_NAME {
        return Err(format!(
            "{}: architecture '{}' is not {ARCH_NAME}",
            model_path.display(),
            model.header.arch.arch_name
        ));
    }
    let d = S2vDefaults::of(&model)?;
    let chunk_seconds = ((d.latent_frames - 1) * TEMPORAL + 1) as f64 / d.fps as f64;
    let track = wav.len() as f64 / SAMPLE_RATE as f64;
    let chunks = p.chunks.unwrap_or_else(|| (track / chunk_seconds).ceil().max(1.0) as usize);
    let target = chunks as f64 * chunk_seconds;
    if target > 300.0 + 0.01 {
        return Err(format!("{chunks} chunks need {target} s of speech, the encoder truncates at 300 s"));
    }
    let mut rng = SplitMix(crate::ltxpipe::Rng::new(p.seed));
    let oracle = p.oracle_noise.as_ref();
    let cond = {
        let mut eps = |n: usize| -> Vec<f32> {
            match oracle {
                Some((e, _)) => e.clone(),
                None => rng.normal_vec(n),
            }
        };
        encode_conditions(&model, &d, image, wav, chunks, p, &mut eps, &mut progress)?
    };
    let (lh, lw) = (cond.height / 8, cond.width / 8);
    let lt = d.latent_frames;
    let steps = p.steps.unwrap_or(d.steps);
    let flow = Piflow::new(&d, steps);
    let ts = flow.timesteps();
    let t_load = Instant::now();
    let dit = S2vDit::from_cmf(&model)?;
    prof("dit load", t_load);
    let cfg = dit.cfg.clone();
    let g = S2vGeom::new(&cfg, lt, lh, lw);
    let c = cfg.in_visual;
    let mut memory = init_memory(&cfg, lh, lw);
    let mut latents = Vec::with_capacity(chunks);
    let reference = cond.reference.clone();
    let mut dev = crate::gpu::s2v_prepare(&dit, &g, &cond, &ts, d.scale_factor);
    for k in 0..chunks {
        let mut x = match oracle {
            Some((_, n)) => n[k].clone(),
            None => rng.normal_vec(lt * lh * lw * c),
        };
        for (i, &t) in ts.iter().enumerate() {
            if p.max_steps.is_some_and(|m| i >= m) {
                break;
            }
            let t0 = Instant::now();
            let pred = match dev.as_mut() {
                Some(dv) => dv.step(i, k, &x, &memory)?,
                None => dit.forward(
                    &S2vInput {
                        x: &x,
                        reference: &reference,
                        mem1: &memory[0],
                        mem2: &memory[1],
                        mem4: &memory[2],
                        t,
                        text_feats: &cond.text,
                        audio_feats: &cond.audio[k],
                        pooled: &cond.pooled,
                        geom: g,
                        scale_factor: d.scale_factor,
                    },
                    None,
                )?,
            };
            flow.step(i, &pred, &mut x, c, dit.pool());
            progress("step", i + 1, steps);
            prof(&format!("chunk {k} step {i}"), t0);
        }
        if k + 1 < chunks {
            memory = roll_memory(&cfg, &g, &x);
        }
        latents.push(x);
    }
    drop(dev);
    drop(dit);
    crate::gpu::s2v_release();

    let t0 = Instant::now();
    progress("vae decode", 0, chunks);
    let vae = S2vVae::from_cmf(&model)?;
    let frames_per = (lt - 1) * TEMPORAL + 1;
    let keep = ((track * d.fps as f64) as usize).min(chunks * frames_per);
    let px = (cond.height, cond.width);
    let mut out = Vol::zeros(3, keep, px.0, px.1);
    let per = px.0 * px.1;
    let mut done = 0usize;
    for (k, lat) in latents.iter().enumerate() {
        let z = to_vol(lat, lt, lh, lw, c);
        let dec = crate::gpu::s2v_vae_decode(&vae, &z).unwrap_or_else(|| vae.decode(&z));
        let take = dec.f.min(keep - done);
        for ch in 0..3 {
            let src = &dec.data[ch * dec.f * per..ch * dec.f * per + take * per];
            out.data[ch * keep * per + done * per..ch * keep * per + (done + take) * per].copy_from_slice(src);
        }
        done += take;
        progress("vae decode", k + 1, chunks);
    }
    prof("vae decode", t0);
    Ok(S2vOutput { frames: out, fps: d.fps, latents, lt, lh, lw })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> S2vDefaults {
        S2vDefaults {
            steps: 16,
            fps: 25,
            latent_frames: 32,
            seed: 5,
            buckets: vec![(512, 512), (768, 512), (512, 768)],
            scale_factor: [1.0, 2.0, 2.0],
            prompt: String::new(),
            system_prompt: String::new(),
            shift: 10.0,
            n_grid: 5,
            eps: 1e-6,
            final_scale: 0.5,
            substeps: 128,
        }
    }

    #[test]
    fn piflow_schedule_matches_the_reference() {
        let f = Piflow::new(&defaults(), 16);
        let t = f.timesteps();
        assert_eq!(t.len(), 16);
        // `PiflowScheduler.set_timesteps(16)` of diffusers.
        assert!((t[0] - 1000.0).abs() < 1e-3);
        assert!((t[1] - 993.1507).abs() < 1e-2);
        assert!((t[2] - 985.4014).abs() < 1e-2);
        assert!((t[15] - 250.0).abs() < 2e-1);
    }

    #[test]
    fn piflow_step_on_a_constant_grid_lands_on_the_constant() {
        // Every grid point predicts x0 = 0, so u = x/σ and the rollout scales x by σ_dst/σ_src.
        let d = defaults();
        let f = Piflow::new(&d, 16);
        let sigma = f.sigmas[0];
        let mut x = vec![2.0f32; 16];
        let pred = vec![2.0 / sigma; 16 * 5];
        f.step(0, &pred, &mut x, 16, None);
        let want = 2.0 * f.sigmas[1] / sigma;
        assert!(x.iter().all(|v| (v - want).abs() < 1e-3), "{} vs {want}", x[0]);
    }

    #[test]
    fn bucket_follows_the_aspect_ratio() {
        let b = defaults().buckets;
        assert_eq!(select_bucket(&b, 960, 960), (512, 512));
        assert_eq!(select_bucket(&b, 960, 1431), (768, 512));
        assert_eq!(select_bucket(&b, 1431, 960), (512, 768));
    }

    #[test]
    fn pil_bilinear_keeps_a_flat_image() {
        let src = vec![90u8; 8 * 6 * 3];
        let out = pil_bilinear(&src, 8, 6, 4, 3);
        assert_eq!(out.len(), 4 * 3 * 3);
        assert!(out.iter().all(|&v| v == 90));
    }

    #[test]
    fn memory_rolls_the_newest_frames_to_the_finest_scale() {
        let v: serde_json::Value = serde_json::json!({
            "in_visual_dim": 1, "out_visual_dim": 5, "in_text_dim": 8, "in_text_dim2": 8, "in_audio_dim": 8,
            "time_dim": 8, "patch_size": [1, 2, 2], "frame_buckets": [2, 4, 24], "index_rope_ref_image": 40,
            "model_dim": 128, "ff_dim": 256, "num_text_blocks": 1, "num_audio_blocks": 1, "num_visual_blocks": 1,
            "axes_dims": [32, 48, 48], "audio_inject_layers": []
        });
        let cfg = S2vConfig::from_json(&v).unwrap();
        let g = S2vGeom::new(&cfg, 32, 8, 8);
        // One value per frame.
        let mut x = vec![0f32; 32 * 8 * 8];
        for f in 0..32 {
            x[f * 64..(f + 1) * 64].fill(f as f32);
        }
        let m = roll_memory(&cfg, &g, &x);
        assert_eq!(m[0].len(), 2 * 64);
        assert_eq!(m[0][0], 30.0);
        assert_eq!(m[1][0], 26.0);
        assert_eq!(m[2][0], 2.0);
        assert_eq!(g.total(), g.n4 + g.n2 + g.n1 + g.nref + g.nvis);
        let (c, _) = crate::s2v::rope_visual(&cfg, &g, [1.0, 2.0, 2.0]);
        assert_eq!(c.len(), g.total() * crate::s2v::ROPE_PAIRS);
    }

    #[test]
    fn speech_groups_cover_the_tokens_once() {
        let g = crate::s2v::audio_groups(125, 32);
        assert_eq!(g.len(), 32);
        assert_eq!(g[0], (0, 1));
        assert_eq!(g.iter().map(|x| x.1).sum::<usize>(), 125);
        assert_eq!(g[31].0 + g[31].1, 125);
    }
}
