//! Haru (higurashi-ai/Haru-4B): full-duplex spoken dialogue at 7.5 Hz.
//!
//! One language-model position is one 133 ms frame, and its input is the sum
//! of three streams at that frame: the model's text token, the model's audio
//! latent and the user's audio latent, each latent through its projector
//! (`haru.{model,user}_audio_proj`). The final-normed hidden state at frame t
//! predicts frame t + 1: the text token through the tied embedding, the audio
//! latent through the MeanFlow head (`haru.head`, a four-layer adaLN MLP, one
//! sampling step). The VibeVoice codec (`haru.codec`, causal convolution
//! stacks) turns 3200 samples of 24 kHz audio into one 64-dim latent and back.
//! The language model is a Qwen3.5 text stack driven row by row through
//! `Pipeline::speech_*`; everything else runs in f32, its big products on the
//! device through `gemm_nt`.
//!
//! The prompt (character card as a ChatML system turn, the voice reference as
//! audio on the model's stream), the sampling (DRY, PAD / EOP in float32,
//! top-k) and the offline loop follow higurashi-ai/Haru's `haru` package
//! (`DuplexSession`, `haru-inference`). The codec is causal, so encoding a
//! whole recording and decoding the whole reply give the frame-by-frame
//! stream's results exactly.

use crate::inference::{gelu_erf, silu};
use crate::pipeline::Pipeline;
use crate::pool::Pool;
use crate::qwen3_tts::{Lin, SendMut, par_rows, rms, vec_n};
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::Arc;

/// Whether the file is a Haru conversion.
pub fn is_haru(model: &CmfModel) -> bool {
    model.header.provenance.as_ref().is_some_and(|p| p.get("haru").is_some())
}

fn ju(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("haru: config lacks {k}"))
}

fn jf(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn jv(v: &Value, k: &str) -> Result<Vec<usize>, String> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
        .ok_or_else(|| format!("haru: config lacks {k}"))
}

/// What the loop needs from `provenance.haru.config`.
#[derive(Clone, Debug)]
pub struct HaruConfig {
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub latent_dim: usize,
    pub pad: u32,
    pub eop: u32,
    pub eos: Option<u32>,
    pub hidden: usize,
    base: usize,
    ratios: Vec<usize>,
    enc_depths: Vec<usize>,
    dec_depths: Vec<usize>,
    codec_eps: f32,
    latent_std: f32,
    head_layers: usize,
    head_ffn: usize,
    head_eps: f32,
    interval_embed: f32,
    pub num_steps: usize,
    pub guidance: f32,
}

impl HaruConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let c = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("haru"))
            .and_then(|p| p.get("config"))
            .ok_or("haru: the file has no provenance.haru.config")?;
        let codec = c.get("codec").ok_or("haru: no codec config")?;
        let head = c.get("head").ok_or("haru: no head config")?;
        let lm = c.get("lm").ok_or("haru: no lm config")?;
        let hidden = ju(head, "hidden_size")?;
        let cfg = Self {
            sample_rate: ju(c, "sample_rate")? as u32,
            frame_samples: ju(c, "frame_samples")?,
            latent_dim: ju(c, "latent_dim")?,
            pad: ju(c, "pad_token_id")? as u32,
            eop: ju(c, "eop_token_id")? as u32,
            eos: lm.get("eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32),
            hidden,
            base: ju(codec, "base_channels")?,
            ratios: jv(codec, "ratios")?,
            enc_depths: jv(codec, "encoder_depths")?,
            dec_depths: jv(codec, "decoder_depths")?,
            codec_eps: jf(codec, "norm_eps", 1e-5) as f32,
            latent_std: jf(codec, "latent_std", 0.5) as f32,
            head_layers: ju(head, "num_layers")?,
            head_ffn: (hidden as f64 * jf(head, "ffn_ratio", 3.0)) as usize,
            head_eps: jf(head, "norm_eps", 1e-5) as f32,
            interval_embed: jf(head, "interval_embed", 7.0) as f32,
            num_steps: ju(head, "num_steps").unwrap_or(1),
            guidance: jf(head, "guidance_scale", 1.5) as f32,
        };
        if cfg.ratios.iter().product::<usize>() != cfg.frame_samples {
            return Err("haru: codec ratios do not match frame_samples".into());
        }
        Ok(cfg)
    }
}

/// A standard normal draw (Box–Muller).
fn normal(rng: &mut SplitMix64) -> f32 {
    let u1 = (rng.next_f32() as f64).max(1e-12);
    let u2 = rng.next_f32() as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

// ───────────────────────────── codec ─────────────────────────────

/// A causal convolution over `[T][cin]` (channels last), stride `s`, the
/// left context `k - s` zero (a whole clip from a zero state); a partial
/// final hop is padded on the right.
struct Conv {
    /// `[cout][k][cin]` from the source `[cout][cin][k]`.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    s: usize,
}

impl Conv {
    fn load(model: &CmfModel, name: &str, cin: usize, cout: usize, k: usize, s: usize) -> Result<Self, String> {
        let src = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        if src.len() != cout * cin * k {
            return Err(format!("{name}.weight: {} values, expected [{cout}, {cin}, {k}]", src.len()));
        }
        let mut w = vec![0f32; src.len()];
        for o in 0..cout {
            for i in 0..cin {
                for t in 0..k {
                    w[(o * k + t) * cin + i] = src[(o * cin + i) * k + t];
                }
            }
        }
        Ok(Self { w, b: vec_n(model, &format!("{name}.bias"), cout)?, cin, cout, k, s })
    }

    /// `x [t][cin]` → `[ceil(t / s)][cout]`.
    fn apply(&self, x: &[f32], t: usize, pool: Option<&Pool>) -> (Vec<f32>, usize) {
        let (cin, k, s) = (self.cin, self.k, self.s);
        let ctx = k - s;
        let tout = t.div_ceil(s);
        let mut y = vec![0f32; tout * self.cout];
        let block = ((1usize << 22) / (k * cin).max(1)).clamp(1, tout.max(1));
        let mut col = vec![0f32; block * k * cin];
        let mut t0 = 0;
        while t0 < tout {
            let nb = block.min(tout - t0);
            let cp = SendMut(col.as_mut_ptr());
            par_rows(pool, nb, &|lo, hi| {
                for r in lo..hi {
                    let start = (t0 + r) * s;
                    for j in 0..k {
                        let src = start as isize + j as isize - ctx as isize;
                        // SAFETY: row r of `col` belongs to this worker alone
                        let dst = unsafe { std::slice::from_raw_parts_mut(cp.get().add((r * k + j) * cin), cin) };
                        if src < 0 || src as usize >= t {
                            dst.fill(0.0);
                        } else {
                            dst.copy_from_slice(&x[src as usize * cin..(src as usize + 1) * cin]);
                        }
                    }
                }
            });
            crate::fcd_ops::gemm_nt(&col[..nb * k * cin], &self.w, &mut y[t0 * self.cout..(t0 + nb) * self.cout], nb, k * cin, self.cout, pool);
            t0 += nb;
        }
        add_bias(&mut y, &self.b, pool);
        (y, tout)
    }
}

fn add_bias(y: &mut [f32], b: &[f32], pool: Option<&Pool>) {
    let c = b.len();
    let rows = y.len() / c;
    let yp = SendMut(y.as_mut_ptr());
    par_rows(pool, rows, &|lo, hi| {
        // SAFETY: rows lo..hi of `y` belong to this worker alone
        let part = unsafe { std::slice::from_raw_parts_mut(yp.get().add(lo * c), (hi - lo) * c) };
        for r in part.chunks_mut(c) {
            for (v, &bb) in r.iter_mut().zip(b) {
                *v += bb;
            }
        }
    });
}

/// `CausalConvTranspose1d`: kernel `2s`, stride `s`, the overlap tail of a
/// whole sequence dropped: `[t][cin]` → `[t·s][cout]`.
struct ConvT {
    /// `[cout·k][cin]` (rows `(o, j)`), from the source `[cin][cout][k]`.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    s: usize,
}

impl ConvT {
    fn load(model: &CmfModel, name: &str, cin: usize, cout: usize, s: usize) -> Result<Self, String> {
        let k = 2 * s;
        let src = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        if src.len() != cin * cout * k {
            return Err(format!("{name}.weight: {} values, expected [{cin}, {cout}, {k}]", src.len()));
        }
        let mut w = vec![0f32; src.len()];
        for i in 0..cin {
            for o in 0..cout {
                for j in 0..k {
                    w[(o * k + j) * cin + i] = src[(i * cout + o) * k + j];
                }
            }
        }
        Ok(Self { w, b: vec_n(model, &format!("{name}.bias"), cout)?, cin, cout, k, s })
    }

    fn apply(&self, x: &[f32], t: usize, pool: Option<&Pool>) -> Vec<f32> {
        let (cout, k, s) = (self.cout, self.k, self.s);
        let keep = t * s;
        let mut y = vec![0f32; keep * cout];
        let block = ((1usize << 22) / (cout * k)).clamp(1, t.max(1));
        let mut part = vec![0f32; block * cout * k];
        let mut t0 = 0;
        while t0 < t {
            let nb = block.min(t - t0);
            crate::fcd_ops::gemm_nt(&x[t0 * self.cin..(t0 + nb) * self.cin], &self.w, &mut part[..nb * cout * k], nb, self.cin, cout * k, pool);
            let (yp, part) = (SendMut(y.as_mut_ptr()), &part);
            par_rows(pool, cout, &|o0, o1| {
                for r in 0..nb {
                    let tt = t0 + r;
                    for o in o0..o1 {
                        for j in 0..k {
                            let at = tt * s + j;
                            if at < keep {
                                // SAFETY: column o of `y` belongs to this worker alone
                                unsafe { *yp.get().add(at * cout + o) += part[(r * cout + o) * k + j] };
                            }
                        }
                    }
                }
            });
            t0 += nb;
        }
        add_bias(&mut y, &self.b, pool);
        y
    }
}

/// `Block`: a depthwise causal k7 mixer and an MLP, each with a scaled
/// residual, RMS norms in front.
struct Block {
    norm: Vec<f32>,
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    gamma: Vec<f32>,
    ffn_norm: Vec<f32>,
    fc1: Lin,
    fc2: Lin,
    ffn_gamma: Vec<f32>,
    dim: usize,
}

impl Block {
    fn load(model: &CmfModel, p: &str, dim: usize) -> Result<Self, String> {
        let dw = crate::dit::cmf_f32(model, &format!("{p}.conv.weight"))?;
        if dw.len() != dim * 7 {
            return Err(format!("{p}.conv.weight: {} values", dw.len()));
        }
        Ok(Self {
            norm: vec_n(model, &format!("{p}.norm.weight"), dim)?,
            dw,
            dw_b: vec_n(model, &format!("{p}.conv.bias"), dim)?,
            gamma: vec_n(model, &format!("{p}.gamma"), dim)?,
            ffn_norm: vec_n(model, &format!("{p}.ffn_norm.weight"), dim)?,
            fc1: Lin::load(model, &format!("{p}.fc1"), 4 * dim, dim, true)?,
            fc2: Lin::load(model, &format!("{p}.fc2"), dim, 4 * dim, true)?,
            ffn_gamma: vec_n(model, &format!("{p}.ffn_gamma"), dim)?,
            dim,
        })
    }

    fn apply(&self, x: &mut [f32], t: usize, eps: f32, pool: Option<&Pool>) {
        let c = self.dim;
        let xn: Vec<f32> = x.chunks(c).flat_map(|r| rms(r, &self.norm, eps)).collect();
        let xp = SendMut(x.as_mut_ptr());
        let xn = &xn;
        par_rows(pool, t, &|lo, hi| {
            // rows lo..hi read only rows ≤ their own of `xn`, write their own of `x`
            for r in lo..hi {
                // SAFETY: row r of `x` belongs to this worker alone
                let row = unsafe { std::slice::from_raw_parts_mut(xp.get().add(r * c), c) };
                for (ch, out) in row.iter_mut().enumerate() {
                    let mut acc = self.dw_b[ch];
                    for j in 0..7 {
                        let src = r as isize + j as isize - 6;
                        if src >= 0 {
                            acc += self.dw[ch * 7 + j] * xn[src as usize * c + ch];
                        }
                    }
                    *out += acc * self.gamma[ch];
                }
            }
        });
        let hn: Vec<f32> = x.chunks(c).flat_map(|r| rms(r, &self.ffn_norm, eps)).collect();
        let mut h = self.fc1.apply(&hn, t, pool);
        let hp = SendMut(h.as_mut_ptr());
        let n = h.len();
        par_rows(pool, t, &|lo, hi| {
            let w = n / t;
            // SAFETY: rows lo..hi of `h` belong to this worker alone
            let part = unsafe { std::slice::from_raw_parts_mut(hp.get().add(lo * w), (hi - lo) * w) };
            for v in part {
                *v = gelu_erf(*v);
            }
        });
        let h = self.fc2.apply(&h, t, pool);
        for (row, hr) in x.chunks_mut(c).zip(h.chunks(c)) {
            for ((v, &a), &g) in row.iter_mut().zip(hr).zip(&self.ffn_gamma) {
                *v += a * g;
            }
        }
    }
}

/// The VibeVoice acoustic codec.
pub struct Codec {
    enc_stem: Conv,
    enc_down: Vec<Conv>,
    enc_stages: Vec<Vec<Block>>,
    enc_head: Conv,
    dec_stem: Conv,
    dec_up: Vec<ConvT>,
    dec_stages: Vec<Vec<Block>>,
    dec_head: Conv,
    eps: f32,
    latent_dim: usize,
    frame: usize,
    latent_std: f32,
    pool: Option<Arc<Pool>>,
}

impl Codec {
    pub fn from_model(model: &CmfModel, c: &HaruConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let (b, n) = (c.base, c.ratios.len());
        let p = "haru.codec";
        // encoder: strides are the ratios reversed, the channels double
        let strides: Vec<usize> = c.ratios.iter().rev().copied().collect();
        let mut enc_down = Vec::new();
        for (i, &s) in strides.iter().enumerate() {
            enc_down.push(Conv::load(model, &format!("{p}.encoder.down.{i}"), b << i, b << (i + 1), 2 * s, s)?);
        }
        let mut enc_stages = Vec::new();
        for (i, &d) in c.enc_depths.iter().enumerate() {
            let mut st = Vec::new();
            for j in 0..d {
                st.push(Block::load(model, &format!("{p}.encoder.stages.{i}.{j}"), b << i)?);
            }
            enc_stages.push(st);
        }
        let mut dec_up = Vec::new();
        for (i, &s) in c.ratios.iter().enumerate() {
            dec_up.push(ConvT::load(model, &format!("{p}.decoder.up.{i}"), b << (n - i), b << (n - i - 1), s)?);
        }
        let mut dec_stages = Vec::new();
        for (i, &d) in c.dec_depths.iter().enumerate() {
            let mut st = Vec::new();
            for j in 0..d {
                st.push(Block::load(model, &format!("{p}.decoder.stages.{i}.{j}"), b << (n - i))?);
            }
            dec_stages.push(st);
        }
        Ok(Self {
            enc_stem: Conv::load(model, &format!("{p}.encoder.stem"), 1, b, 7, 1)?,
            enc_down,
            enc_stages,
            enc_head: Conv::load(model, &format!("{p}.encoder.head"), b << n, c.latent_dim, 7, 1)?,
            dec_stem: Conv::load(model, &format!("{p}.decoder.stem"), c.latent_dim, b << n, 7, 1)?,
            dec_up,
            dec_stages,
            dec_head: Conv::load(model, &format!("{p}.decoder.head"), b, 1, 7, 1)?,
            eps: c.codec_eps,
            latent_dim: c.latent_dim,
            frame: c.frame_samples,
            latent_std: c.latent_std,
            pool,
        })
    }

    /// Latent means `[ceil(len / 3200)][64]` of a whole clip.
    pub fn encode(&self, audio: &[f32]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut t = audio.len();
        let mut x = audio.to_vec();
        for (i, stage) in self.enc_stages.iter().enumerate() {
            let conv = if i == 0 { &self.enc_stem } else { &self.enc_down[i - 1] };
            let (y, tn) = conv.apply(&x, t, pool);
            x = y;
            t = tn;
            for blk in stage {
                blk.apply(&mut x, t, self.eps, pool);
            }
        }
        self.enc_head.apply(&x, t, pool).0
    }

    /// The waveform of `latents [t][64]` (raw codec space), 3200 samples a latent.
    pub fn decode(&self, latents: &[f32]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut t = latents.len() / self.latent_dim;
        let mut x = latents.to_vec();
        for (i, stage) in self.dec_stages.iter().enumerate() {
            if i == 0 {
                x = self.dec_stem.apply(&x, t, pool).0;
            } else {
                let up = &self.dec_up[i - 1];
                x = up.apply(&x, t, pool);
                t *= up.s;
            }
            for blk in stage {
                blk.apply(&mut x, t, self.eps, pool);
            }
        }
        let mut wav = self.dec_head.apply(&x, t, pool).0;
        for v in &mut wav {
            *v = v.clamp(-1.0, 1.0);
        }
        wav
    }

    /// Latents drawn around `means [n][64]` with one noise scale for all
    /// of them (`VibeVoiceCodec.sample` over one call's rows).
    pub fn sample(&self, means: &[f32], rng: &mut SplitMix64) -> Vec<f32> {
        let scale = normal(rng) * self.latent_std / 0.8;
        means.iter().map(|&m| m + scale * normal(rng)).collect()
    }

    /// The latent of digital silence (the last of sixteen silent frames).
    pub fn silence(&self) -> Vec<f32> {
        let m = self.encode(&vec![0f32; 16 * self.frame]);
        m[m.len() - self.latent_dim..].to_vec()
    }
}

// ───────────────────────────── projectors and head ─────────────────────────────

/// Audio latent → embedding space: `fc2(rms(fc1(x)))`.
struct Projector {
    fc1: Lin,
    norm: Vec<f32>,
    fc2: Lin,
}

impl Projector {
    fn load(model: &CmfModel, p: &str, latent: usize, hidden: usize) -> Result<Self, String> {
        Ok(Self {
            fc1: Lin::load(model, &format!("{p}.fc1"), hidden, latent, true)?,
            norm: vec_n(model, &format!("{p}.norm.weight"), hidden)?,
            fc2: Lin::load(model, &format!("{p}.fc2"), hidden, hidden, true)?,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let h = self.fc1.apply(x, n, pool);
        let hidden = h.len() / n.max(1);
        let hn: Vec<f32> = h.chunks(hidden).flat_map(|r| rms(r, &self.norm, 1e-6)).collect();
        self.fc2.apply(&hn, n, pool)
    }
}

struct ScalarEmbed {
    fc1: Lin,
    fc2: Lin,
}

impl ScalarEmbed {
    fn apply(&self, v: f32) -> Vec<f32> {
        const HALF: usize = 128;
        let mut f = vec![0f32; 2 * HALF];
        for i in 0..HALF {
            let freq = (-(10000f64.ln()) * i as f64 / HALF as f64).exp() as f32;
            let a = v * freq;
            f[i] = a.cos();
            f[HALF + i] = a.sin();
        }
        let mut h = self.fc1.apply(&f, 1, None);
        for x in &mut h {
            *x = silu(*x);
        }
        self.fc2.apply(&h, 1, None)
    }
}

struct HeadLayer {
    ada: Lin,
    norm: Vec<f32>,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// The MeanFlow head: a latent from a hidden state and noise.
pub struct Head {
    cond_proj: Lin,
    latent_proj: Lin,
    t_embed: ScalarEmbed,
    dt_embed: ScalarEmbed,
    h_embed: ScalarEmbed,
    omega_embed: ScalarEmbed,
    layers: Vec<HeadLayer>,
    final_ada: Lin,
    final_proj: Lin,
    hidden: usize,
    eps: f32,
    interval: f32,
    pool: Option<Arc<Pool>>,
}

/// RMS norm without a weight.
fn rms_plain(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps as f64).sqrt();
    x.iter().map(|&v| (v as f64 * r) as f32).collect()
}

impl Head {
    pub fn from_model(model: &CmfModel, c: &HaruConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let (hd, ld, ffn) = (c.hidden, c.latent_dim, c.head_ffn);
        let p = "haru.head";
        let se = |n: &str| -> Result<ScalarEmbed, String> {
            Ok(ScalarEmbed { fc1: Lin::load(model, &format!("{p}.{n}.fc1"), hd, 256, false)?, fc2: Lin::load(model, &format!("{p}.{n}.fc2"), hd, hd, false)? })
        };
        let mut layers = Vec::new();
        for i in 0..c.head_layers {
            let b = format!("{p}.layers.{i}");
            layers.push(HeadLayer {
                ada: Lin::load(model, &format!("{b}.ada"), 3 * hd, hd, false)?,
                norm: vec_n(model, &format!("{b}.norm.weight"), hd)?,
                gate: Lin::load(model, &format!("{b}.ffn.gate_proj"), ffn, hd, false)?,
                up: Lin::load(model, &format!("{b}.ffn.up_proj"), ffn, hd, false)?,
                down: Lin::load(model, &format!("{b}.ffn.down_proj"), hd, ffn, false)?,
            });
        }
        Ok(Self {
            cond_proj: Lin::load(model, &format!("{p}.cond_proj"), hd, hd, false)?,
            latent_proj: Lin::load(model, &format!("{p}.latent_proj"), hd, ld, false)?,
            t_embed: se("t_embed")?,
            dt_embed: se("dt_embed")?,
            h_embed: se("h_embed")?,
            omega_embed: se("omega_embed")?,
            layers,
            final_ada: Lin::load(model, &format!("{p}.final.ada"), 2 * hd, hd, false)?,
            final_proj: Lin::load(model, &format!("{p}.final.proj"), ld, hd, false)?,
            hidden: hd,
            eps: c.head_eps,
            interval: c.interval_embed,
            pool,
        })
    }

    /// The scalar terms of the condition at step time `t`, interval `h`
    /// and guidance `w` (constant across frames).
    pub fn scalars(&self, t: f32, h: f32, w: f32) -> Vec<f32> {
        let mut c = self.t_embed.apply(t);
        for e in [self.dt_embed.apply(self.interval), self.h_embed.apply(h), self.omega_embed.apply(1.0 - 1.0 / w)] {
            for (a, b) in c.iter_mut().zip(e) {
                *a += b;
            }
        }
        c
    }

    /// Normalized latent from `hidden` (final-normed) and `noise`, `steps`
    /// steps of the mean flow; `scalars[i]` from [`Self::scalars`].
    pub fn sample(&self, hidden: &[f32], noise: &[f32], scalars: &[Vec<f32>]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let hd = self.hidden;
        let steps = scalars.len();
        let cond = self.cond_proj.apply(hidden, 1, pool);
        let mut z = noise.to_vec();
        let dh = 1.0 / steps as f32;
        for sc in scalars {
            let c: Vec<f32> = cond.iter().zip(sc).map(|(&a, &b)| silu(a + b)).collect();
            let mut x = self.latent_proj.apply(&z, 1, pool);
            for l in &self.layers {
                let ada = l.ada.apply(&c, 1, pool);
                let (shift, scale, gate) = (&ada[..hd], &ada[hd..2 * hd], &ada[2 * hd..]);
                let h: Vec<f32> = rms_plain(&x, self.eps).iter().zip(&l.norm).map(|(&v, &w)| v * w).collect();
                let m: Vec<f32> = h.iter().zip(scale).zip(shift).map(|((&v, &s), &b)| v * (1.0 + s) + b).collect();
                let g = l.gate.apply(&m, 1, pool);
                let u = l.up.apply(&m, 1, pool);
                let mid: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
                let f = l.down.apply(&mid, 1, pool);
                for ((xv, &fv), &gv) in x.iter_mut().zip(&f).zip(gate) {
                    *xv += gv * fv;
                }
            }
            let ada = self.final_ada.apply(&c, 1, pool);
            let (shift, scale) = (&ada[..hd], &ada[hd..]);
            let m: Vec<f32> = rms_plain(&x, self.eps).iter().zip(scale).zip(shift).map(|((&v, &s), &b)| v * (1.0 + s) + b).collect();
            let v = self.final_proj.apply(&m, 1, pool);
            for (zv, &vv) in z.iter_mut().zip(&v) {
                *zv += dh * vv;
            }
        }
        z
    }
}

// ───────────────────────────── sampling ─────────────────────────────

/// DRY: discourage extending a token sequence that already occurred.
#[derive(Clone, Copy, Debug)]
pub struct Dry {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub lookback: usize,
    pub max_match_length: usize,
    pub max_penalty: f32,
}

impl Default for Dry {
    fn default() -> Self {
        Self { multiplier: 0.8, base: 1.75, allowed_length: 1, lookback: 1024, max_match_length: 32, max_penalty: 32.0 }
    }
}

/// `{token: penalty}` for the tokens that would extend an earlier repeat.
fn dry_penalties(history: &[u32], d: &Dry) -> Vec<(u32, f32)> {
    let n = history.len();
    if n < d.allowed_length.max(1) {
        return Vec::new();
    }
    let mut longest: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let last = history[n - 1];
    let lo = (n as isize - 1 - d.lookback as isize).max(0) as usize;
    let mut j = n as isize - 2;
    while j >= lo as isize {
        let ju = j as usize;
        if history[ju] == last {
            let mut length = 1;
            while length < d.max_match_length && ju >= length && history[ju - length] == history[n - 1 - length] {
                length += 1;
            }
            if length >= d.allowed_length {
                let cand = history[ju + 1];
                let e = longest.entry(cand).or_insert(0);
                if length > *e {
                    *e = length;
                }
            }
        }
        j -= 1;
    }
    longest
        .into_iter()
        .map(|(t, l)| (t, (d.multiplier * d.base.powi((l - d.allowed_length) as i32)).min(d.max_penalty)))
        .collect()
}

/// `torch.topk` + softmax + multinomial over `logits / temperature`
/// (greedy at temperature 0).
fn sample_token(logits: &[f32], temperature: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    }
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    if top_k > 0 && top_k < idx.len() {
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(top_k);
    }
    let mx = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = idx.iter().map(|&i| (((logits[i] - mx) / temperature) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.next_f32() as f64 * total;
    for (&i, &wi) in idx.iter().zip(&w) {
        r -= wi;
        if r <= 0.0 {
            return i as u32;
        }
    }
    *idx.last().unwrap() as u32
}

// ───────────────────────────── the conversation ─────────────────────────────

/// How a conversation is sampled (the `DuplexConfig` defaults).
#[derive(Clone, Debug)]
pub struct Options {
    pub guidance: Option<f32>,
    pub temperature: f32,
    pub top_k: usize,
    pub dry: Option<Dry>,
    pub eop_bias: f32,
    pub speak_first: bool,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self { guidance: None, temperature: 0.7, top_k: 25, dry: Some(Dry::default()), eop_bias: 0.0, speak_first: false, seed: 0 }
    }
}

/// The model's side of a conversation.
pub struct Reply {
    pub audio: Vec<f32>,
    /// Per frame: the new text (`" "` for a run of silent frames).
    pub text: Vec<String>,
    pub frames: usize,
    pub step_ms: Vec<f64>,
}

/// Everything besides the language model.
pub struct Haru {
    pub cfg: HaruConfig,
    pub codec: Codec,
    head: Head,
    model_proj: Projector,
    user_proj: Projector,
    latent_scale: f32,
    latent_bias: f32,
    pool: Option<Arc<Pool>>,
}

impl Haru {
    pub fn from_model(model: &CmfModel, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cfg = HaruConfig::from_model(model)?;
        let one = |n: &str| -> Result<f32, String> {
            let v = crate::dit::cmf_f32(model, n)?;
            v.first().copied().ok_or_else(|| format!("{n}: empty"))
        };
        Ok(Self {
            codec: Codec::from_model(model, &cfg, pool.clone())?,
            head: Head::from_model(model, &cfg, pool.clone())?,
            model_proj: Projector::load(model, "haru.model_audio_proj", cfg.latent_dim, cfg.hidden)?,
            user_proj: Projector::load(model, "haru.user_audio_proj", cfg.latent_dim, cfg.hidden)?,
            latent_scale: one("haru.latent_scale")?,
            latent_bias: one("haru.latent_bias")?,
            cfg,
            pool,
        })
    }

    fn scaled(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter().map(|&v| (v + self.latent_bias) * self.latent_scale).collect()
    }

    fn unscaled(&self, n: &[f32]) -> Vec<f32> {
        n.iter().map(|&v| v / self.latent_scale - self.latent_bias).collect()
    }

    /// A position's input: the token's embedding plus each projected stream.
    fn row(&self, pipe: &Pipeline, token: u32, model_latent: Option<&[f32]>, user_latent: Option<&[f32]>) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut x = pipe.embed_id(token);
        for (proj, lat) in [(&self.model_proj, model_latent), (&self.user_proj, user_latent)] {
            if let Some(l) = lat {
                let y = proj.apply(&self.scaled(l), 1, pool);
                for (a, b) in x.iter_mut().zip(&y) {
                    *a += b;
                }
            }
        }
        x
    }

    /// Play `user` (mono 24 kHz) to the model frame by frame, as in a live
    /// call, and return what it says over the same span (`haru-inference`).
    pub fn converse(
        &self,
        pipe: &mut Pipeline,
        system_prompt: &str,
        reference: Option<&[f32]>,
        user: &[f32],
        o: &Options,
    ) -> Result<Reply, String> {
        let cfg = &self.cfg;
        let (fs, ld) = (cfg.frame_samples, cfg.latent_dim);
        let mut rng = SplitMix64::new(o.seed);
        let guidance = o.guidance.unwrap_or(cfg.guidance);
        let steps = cfg.num_steps.max(1);
        let scalars: Vec<Vec<f32>> = (0..steps).map(|i| self.head.scalars(i as f32 / steps as f32, 1.0 / steps as f32, guidance)).collect();
        // ── the prompt: the card, the reference on the model's stream, the grid
        let card: Vec<u32> = if system_prompt.is_empty() {
            Vec::new()
        } else {
            pipe.tokenizer.encode(&format!("<|im_start|>system\n{system_prompt}<|im_end|>\n"))
        };
        let (ref_lat, ref_raw) = match reference.filter(|r| !r.is_empty()) {
            Some(r) => (Some(self.codec.sample(&self.codec.encode(r), &mut rng)), Some(r)),
            None => (None, None),
        };
        let n_ref = ref_lat.as_ref().map_or(0, |l| l.len() / ld);
        let mut grid: Vec<u32> = if o.speak_first { vec![cfg.pad, cfg.eop] } else { Vec::new() };
        if card.is_empty() && n_ref == 0 && grid.is_empty() {
            grid.push(cfg.pad);
        }
        let mut tokens = card.clone();
        if n_ref > 0 {
            tokens.extend(std::iter::repeat_n(cfg.pad, n_ref - 1));
            tokens.push(cfg.eop);
        }
        tokens.extend(&grid);
        let (n, n_card, n_grid) = (tokens.len(), card.len(), grid.len());
        let silence = self.codec.silence();
        // the speaker's noise floor: the reference's quietest whole frame
        let room = match (&ref_lat, ref_raw) {
            (Some(l), Some(r)) if r.len() >= fs => {
                let whole = r.len() / fs;
                let quiet = (0..whole)
                    .min_by(|&a, &b| {
                        let e = |i: usize| r[i * fs..(i + 1) * fs].iter().map(|v| v * v).sum::<f32>();
                        e(a).total_cmp(&e(b))
                    })
                    .unwrap_or(0);
                l[quiet * ld..(quiet + 1) * ld].to_vec()
            }
            _ => silence.clone(),
        };
        let mut rows = Vec::with_capacity(n * cfg.hidden);
        for (i, &tok) in tokens.iter().enumerate() {
            let model_lat: Option<Vec<f32>> = if i < n_card {
                None
            } else if i < n_card + n_ref {
                ref_lat.as_ref().map(|l| l[(i - n_card) * ld..(i - n_card + 1) * ld].to_vec())
            } else if i >= n - n_grid {
                Some(room.clone())
            } else {
                Some(vec![0f32; ld])
            };
            let user_lat = (i >= n - n_grid).then(|| silence.clone());
            rows.extend(self.row(pipe, tok, model_lat.as_deref(), user_lat.as_deref()));
        }
        let dummy = vec![0u32; n];
        let t0 = std::time::Instant::now();
        let h = pipe.speech_prefill(&dummy, Some(&rows))?;
        let logits = pipe.speech_logits(&h);
        // ── the user's latents: the whole recording encoded once (the codec
        // is causal), each frame drawn with its own noise scale as the
        // stream's per-frame calls do
        let count = user.len().div_ceil(fs).max(1);
        let mut padded = user.to_vec();
        padded.resize(count * fs, 0.0);
        let user_means = self.codec.encode(&padded);
        let mut state = Emit { history: Vec::new(), force_text: usize::from(o.speak_first), pending: Vec::new(), in_gap: false };
        let mut reply = Reply { audio: Vec::new(), text: Vec::new(), frames: 0, step_ms: Vec::new() };
        let mut latents: Vec<f32> = Vec::with_capacity(count * ld);
        let (mut token, mut latent) = self.emit(pipe, &h, logits, &scalars, o, &mut state, &mut rng, &mut reply);
        reply.step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        latents.extend(&latent);
        let mut pos = n;
        for f in 0..count - 1 {
            let t = std::time::Instant::now();
            let ul = self.codec.sample(&user_means[f * ld..(f + 1) * ld], &mut rng);
            let row = self.row(pipe, token, Some(&latent), Some(&ul));
            let (h, logits) = pipe.speech_step_logits(&row, pos);
            pos += 1;
            (token, latent) = self.emit(pipe, &h, logits, &scalars, o, &mut state, &mut rng, &mut reply);
            latents.extend(&latent);
            reply.step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        reply.frames = count;
        reply.audio = self.codec.decode(&latents);
        Ok(reply)
    }

    /// The next frame's text token and audio latent (raw codec space).
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &self,
        pipe: &Pipeline,
        hidden: &[f32],
        mut logits: Vec<f32>,
        scalars: &[Vec<f32>],
        o: &Options,
        st: &mut Emit,
        rng: &mut SplitMix64,
        reply: &mut Reply,
    ) -> (u32, Vec<f32>) {
        let cfg = &self.cfg;
        let silent = [cfg.pad, cfg.eop];
        if let Some(d) = &o.dry {
            for (t, p) in dry_penalties(&st.history, d) {
                if !silent.contains(&t) && Some(t) != cfg.eos {
                    logits[t as usize] -= p;
                }
            }
        }
        // PAD / EOP (stay silent or start speaking) in float32 against the
        // exact embedding rows: the decision is often closer than the
        // device head's resolution
        for &t in &silent {
            let e = pipe.embed_id(t);
            logits[t as usize] = hidden.iter().zip(&e).map(|(a, b)| a * b).sum();
        }
        logits[cfg.eop as usize] += o.eop_bias;
        if st.force_text > 0 {
            for t in silent.iter().copied().chain(cfg.eos) {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            st.force_text -= 1;
        }
        let token = sample_token(&logits, o.temperature, o.top_k, rng);
        let noise: Vec<f32> = (0..cfg.latent_dim).map(|_| normal(rng)).collect();
        let latent = self.unscaled(&self.head.sample(hidden, &noise, scalars));
        if !silent.contains(&token) && Some(token) != cfg.eos {
            st.history.push(token);
            let keep = o.dry.map_or(usize::MAX, |d| d.lookback + d.max_match_length + 1);
            if st.history.len() > keep {
                let drop = st.history.len() - keep;
                st.history.drain(..drop);
            }
        }
        // the frame's text: a run of silent frames is one space, a character
        // waits for all of its tokens
        let text = if silent.contains(&token) {
            if st.in_gap {
                String::new()
            } else {
                st.in_gap = true;
                " ".into()
            }
        } else {
            st.in_gap = false;
            // the token's own bytes (its leading space kept), a character
            // split over tokens held back until it is whole
            pipe.tokenizer.decode_token_stream(token, &mut st.pending)
        };
        reply.text.push(text);
        (token, latent)
    }
}

struct Emit {
    history: Vec<u32>,
    force_text: usize,
    pending: Vec<u8>,
    in_gap: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DRY penalizes the token that would extend a repeat, harder for a
    /// longer one.
    #[test]
    fn dry_penalizes_repeats() {
        let d = Dry::default();
        // … 5 6 7 … 5 6: 7 would repeat a run of two
        let h = [1u32, 5, 6, 7, 2, 3, 5, 6];
        let p = dry_penalties(&h, &d);
        let p7 = p.iter().find(|(t, _)| *t == 7).map(|x| x.1).unwrap();
        assert!((p7 - 0.8 * 1.75).abs() < 1e-5, "{p7}");
    }

    /// A temperature of zero is the argmax.
    #[test]
    fn greedy_token() {
        let mut rng = SplitMix64::new(3);
        assert_eq!(sample_token(&[0.1, 2.0, 1.0], 0.0, 25, &mut rng), 1);
    }
}
