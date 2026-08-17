//! The LTX-2.5 audio path: the spectrogram VAE decoder and the BigVGAN
//! vocoder that turns its output into a waveform.
//!
//! The transformer denoises sound in the same 48 blocks as the picture, so
//! the soundtrack arrives as a `[8, T, 16]` latent. Turning that into audio
//! is two models:
//!
//! 1. **The audio VAE decoder** — 2-D convolutions over (time, mel bin) with
//!    PixelNorm and *height-causal* padding, so a frame never sees the
//!    future. Two mid blocks, then three levels of three residual blocks
//!    with a nearest ×2 between them; the first row after each upsample is
//!    dropped, because the causal padding on the following convolution
//!    already accounts for it. Out comes a 64-bin log-mel spectrogram.
//! 2. **BigVGAN v2 with bandwidth extension** — `conv_pre`, six transposed
//!    convolutions each followed by three anti-aliased multi-receptive-field
//!    blocks whose outputs are averaged, and `conv_post`. Every activation is
//!    a SnakeBeta sandwiched between a ×2 sinc upsample and a ×2 sinc
//!    downsample, which is what keeps the harmonics it generates from
//!    aliasing. That gives 16 kHz stereo; a second generator predicts a
//!    residual from its mel spectrogram and adds it to a sinc-resampled copy
//!    at 48 kHz.
//!
//! Everything is f32 throughout: the reference notes that bf16 accumulation
//! through 108 sequential convolutions costs 40-90 % on spectral metrics.

use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

fn tensor_f32(model: &Arc<CmfModel>, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = model.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let mut out = vec![0.0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, model.entry_bytes(e), &mut out)?;
    Ok((out, e.shape.clone()))
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// A `[C, H, W]` plane: channels, time, mel bin.
#[derive(Clone)]
pub struct Grid {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Grid {
    fn zeros(c: usize, h: usize, w: usize) -> Grid {
        Grid { c, h, w, data: vec![0.0; c * h * w] }
    }
    fn n(&self) -> usize {
        self.h * self.w
    }
}

/// 2-D convolution, padded symmetrically on the mel axis and **causally on
/// the time axis** — the whole kernel extent on the left, nothing on the
/// right.
struct Conv2d {
    w: Vec<f32>,
    b: Vec<f32>,
    c_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
}

impl Conv2d {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Conv2d, String> {
        let (w, s) = tensor_f32(model, &format!("{name}.weight"))?;
        let (b, _) = tensor_f32(model, &format!("{name}.bias"))?;
        Ok(Conv2d { w, b, c_out: s[0], c_in: s[1], kh: s[2], kw: s[3] })
    }

    fn forward(&self, x: &Grid, pool: Option<&Pool>) -> Grid {
        let (h, w) = (x.h, x.w);
        let npos = h * w;
        let k = self.c_in * self.kh * self.kw;
        let mut out = Grid::zeros(self.c_out, h, w);
        let pad_h = self.kh - 1; // causal: all of it on the left
        let pad_w = (self.kw - 1) / 2;
        const CHUNK: usize = 8192;
        let mut patches = vec![0f32; CHUNK.min(npos) * k];
        let mut ys = vec![0f32; CHUNK.min(npos) * self.c_out];
        let mut p0 = 0usize;
        while p0 < npos {
            let n = CHUNK.min(npos - p0);
            patches[..n * k].fill(0.0);
            for i in 0..n {
                let p = p0 + i;
                let (pwi, phi) = (p % w, p / w);
                for ci in 0..self.c_in {
                    for a in 0..self.kh {
                        let sh = phi as isize + a as isize - pad_h as isize;
                        if sh < 0 || sh >= h as isize {
                            continue;
                        }
                        for bb in 0..self.kw {
                            let sw = pwi as isize + bb as isize - pad_w as isize;
                            if sw < 0 || sw >= w as isize {
                                continue;
                            }
                            patches[i * k + (ci * self.kh + a) * self.kw + bb] =
                                x.data[(ci * h + sh as usize) * w + sw as usize];
                        }
                    }
                }
            }
            crate::fcd_ops::gemm_nt(
                &patches[..n * k],
                &self.w,
                &mut ys[..n * self.c_out],
                n,
                k,
                self.c_out,
                pool,
            );
            for i in 0..n {
                for co in 0..self.c_out {
                    out.data[co * npos + p0 + i] = ys[i * self.c_out + co] + self.b[co];
                }
            }
            p0 += n;
        }
        out
    }
}

/// RMS across channels at each (time, mel) location — no learned weight.
fn pixel_norm(x: &mut Grid) {
    let n = x.n();
    for p in 0..n {
        let mut ss = 0f64;
        for c in 0..x.c {
            let v = x.data[c * n + p] as f64;
            ss += v * v;
        }
        let inv = 1.0 / (ss / x.c as f64 + 1e-6).sqrt();
        for c in 0..x.c {
            x.data[c * n + p] = (x.data[c * n + p] as f64 * inv) as f32;
        }
    }
}

struct ResnetBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<ResnetBlock, String> {
        Ok(ResnetBlock {
            conv1: Conv2d::load(model, &format!("{p}.conv1.conv"))?,
            conv2: Conv2d::load(model, &format!("{p}.conv2.conv"))?,
            shortcut: match model.tensor(&format!("{p}.nin_shortcut.conv.weight")) {
                Some(_) => Some(Conv2d::load(model, &format!("{p}.nin_shortcut.conv"))?),
                None => None,
            },
        })
    }

    fn forward(&self, x: &Grid, pool: Option<&Pool>) -> Grid {
        let mut h = x.clone();
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.conv1.forward(&h, pool);
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.conv2.forward(&h, pool);
        let res = match &self.shortcut {
            Some(c) => c.forward(x, pool),
            None => x.clone(),
        };
        for (v, &r) in h.data.iter_mut().zip(&res.data) {
            *v += r;
        }
        h
    }
}

/// Nearest ×2 on both axes, a convolution, then the first time row dropped
/// — the causal padding on the convolution has already reproduced it.
fn upsample2(x: &Grid, conv: &Conv2d, pool: Option<&Pool>) -> Grid {
    let (h2, w2) = (x.h * 2, x.w * 2);
    let mut up = Grid::zeros(x.c, h2, w2);
    for c in 0..x.c {
        for y in 0..h2 {
            for z in 0..w2 {
                up.data[(c * h2 + y) * w2 + z] = x.data[(c * x.h + y / 2) * x.w + z / 2];
            }
        }
    }
    let conved = conv.forward(&up, pool);
    let mut out = Grid::zeros(conved.c, h2 - 1, w2);
    for c in 0..conved.c {
        for y in 1..h2 {
            for z in 0..w2 {
                out.data[(c * (h2 - 1) + y - 1) * w2 + z] = conved.data[(c * h2 + y) * w2 + z];
            }
        }
    }
    out
}

pub struct AudioVaeDecoder {
    conv_in: Conv2d,
    mid: Vec<ResnetBlock>,
    levels: Vec<(Vec<ResnetBlock>, Option<Conv2d>)>,
    conv_out: Conv2d,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl AudioVaeDecoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<AudioVaeDecoder, String> {
        let mut levels = Vec::new();
        let mut lv = 0usize;
        while model
            .tensor(&format!("avae.decoder.up.{lv}.block.0.conv1.conv.weight"))
            .is_some()
        {
            let mut blocks = Vec::new();
            let mut bi = 0usize;
            while model
                .tensor(&format!("avae.decoder.up.{lv}.block.{bi}.conv1.conv.weight"))
                .is_some()
            {
                blocks.push(ResnetBlock::load(model, &format!("avae.decoder.up.{lv}.block.{bi}"))?);
                bi += 1;
            }
            let up = match model.tensor(&format!("avae.decoder.up.{lv}.upsample.conv.conv.weight")) {
                Some(_) => Some(Conv2d::load(model, &format!("avae.decoder.up.{lv}.upsample.conv.conv"))?),
                None => None,
            };
            levels.push((blocks, up));
            lv += 1;
        }
        Ok(AudioVaeDecoder {
            conv_in: Conv2d::load(model, "avae.decoder.conv_in.conv")?,
            mid: vec![
                ResnetBlock::load(model, "avae.decoder.mid.block_1")?,
                ResnetBlock::load(model, "avae.decoder.mid.block_2")?,
            ],
            levels,
            conv_out: Conv2d::load(model, "avae.decoder.conv_out.conv")?,
            mean: tensor_f32(model, "avae.per_channel_statistics.mean-of-means")?.0,
            std: tensor_f32(model, "avae.per_channel_statistics.std-of-means")?.0,
        })
    }

    /// `[8, T, 16]` latent → `[2, 4T-3, 64]` log-mel spectrogram.
    pub fn decode(&self, latent: &Grid, pool: Option<&Pool>) -> Grid {
        // The statistics are per *patchified* channel (channel-major over mel
        // bins), which is how the transformer sees them.
        let mut x = latent.clone();
        let n = x.n();
        for c in 0..x.c {
            for wi in 0..x.w {
                let idx = c * x.w + wi;
                let (m, s) = (self.mean[idx], self.std[idx]);
                for hi in 0..x.h {
                    let o = (c * x.h + hi) * x.w + wi;
                    x.data[o] = x.data[o] * s + m;
                }
            }
        }
        let _ = n;
        let mut h = self.conv_in.forward(&x, pool);
        for b in &self.mid {
            h = b.forward(&h, pool);
        }
        for (blocks, up) in self.levels.iter().rev() {
            for b in blocks {
                h = b.forward(&h, pool);
            }
            if let Some(c) = up {
                h = upsample2(&h, c, pool);
            }
        }
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let out = self.conv_out.forward(&h, pool);
        // the causal decoder produces 4T-3 frames; crop to that
        let target = (latent.h * 4).saturating_sub(3).max(1);
        if out.h == target {
            return out;
        }
        let mut cropped = Grid::zeros(out.c, target.min(out.h), out.w);
        for c in 0..out.c {
            for y in 0..cropped.h {
                for z in 0..out.w {
                    cropped.data[(c * cropped.h + y) * out.w + z] = out.data[(c * out.h + y) * out.w + z];
                }
            }
        }
        cropped
    }
}

// ------------------------------------------------------------- 1-D layers

/// A multi-channel signal `[C, T]`.
#[derive(Clone)]
pub struct Sig {
    pub c: usize,
    pub t: usize,
    pub data: Vec<f32>,
}

impl Sig {
    fn zeros(c: usize, t: usize) -> Sig {
        Sig { c, t, data: vec![0.0; c * t] }
    }
}

struct Conv1d {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    c_out: usize,
    c_in: usize,
    k: usize,
    dilation: usize,
    pad: usize,
}

impl Conv1d {
    fn load(model: &Arc<CmfModel>, name: &str, dilation: usize) -> Result<Conv1d, String> {
        let (w, s) = tensor_f32(model, &format!("{name}.weight"))?;
        let b = tensor_f32(model, &format!("{name}.bias")).ok().map(|x| x.0);
        let k = s[2];
        Ok(Conv1d { w, b, c_out: s[0], c_in: s[1], k, dilation, pad: (k - 1) * dilation / 2 })
    }

    fn forward(&self, x: &Sig, pool: Option<&Pool>) -> Sig {
        let t = x.t;
        let kk = self.c_in * self.k;
        let mut patches = vec![0f32; t * kk];
        for p in 0..t {
            for ci in 0..self.c_in {
                for a in 0..self.k {
                    let s = p as isize + (a * self.dilation) as isize - self.pad as isize;
                    if s >= 0 && s < t as isize {
                        patches[p * kk + ci * self.k + a] = x.data[ci * t + s as usize];
                    }
                }
            }
        }
        let mut ys = vec![0f32; t * self.c_out];
        crate::fcd_ops::gemm_nt(&patches, &self.w, &mut ys, t, kk, self.c_out, pool);
        let mut out = Sig::zeros(self.c_out, t);
        for p in 0..t {
            for co in 0..self.c_out {
                out.data[co * t + p] = ys[p * self.c_out + co] + self.b.as_ref().map_or(0.0, |b| b[co]);
            }
        }
        out
    }
}

/// `ConvTranspose1d(in, out, k, stride, padding)`, weights `[in, out, k]`.
struct ConvT1d {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl ConvT1d {
    fn load(model: &Arc<CmfModel>, name: &str, stride: usize) -> Result<ConvT1d, String> {
        let (w, s) = tensor_f32(model, &format!("{name}.weight"))?;
        let b = tensor_f32(model, &format!("{name}.bias")).ok().map(|x| x.0);
        let k = s[2];
        Ok(ConvT1d { w, b, c_in: s[0], c_out: s[1], k, stride, pad: (k - stride) / 2 })
    }

    fn forward(&self, x: &Sig) -> Sig {
        let t_out = (x.t - 1) * self.stride + self.k - 2 * self.pad;
        let mut out = Sig::zeros(self.c_out, t_out);
        for ci in 0..self.c_in {
            for p in 0..x.t {
                let v = &x.data[ci * x.t + p];
                if *v == 0.0 {
                    continue;
                }
                let base = p * self.stride;
                for a in 0..self.k {
                    let o = base + a;
                    if o < self.pad || o - self.pad >= t_out {
                        continue;
                    }
                    let oo = o - self.pad;
                    for co in 0..self.c_out {
                        out.data[co * t_out + oo] += v * self.w[(ci * self.c_out + co) * self.k + a];
                    }
                }
            }
        }
        if let Some(b) = &self.b {
            for co in 0..self.c_out {
                for v in out.data[co * t_out..(co + 1) * t_out].iter_mut() {
                    *v += b[co];
                }
            }
        }
        out
    }
}

/// The anti-aliasing pair around every activation: a ×2 sinc upsample, the
/// nonlinearity, a ×2 sinc downsample. Both filters ship in the checkpoint.
struct Aliasing {
    up: Vec<f32>,
    down: Vec<f32>,
    ratio: usize,
}

impl Aliasing {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<Aliasing, String> {
        Ok(Aliasing {
            up: tensor_f32(model, &format!("{p}.upsample.filter"))?.0,
            down: tensor_f32(model, &format!("{p}.downsample.lowpass.filter"))?.0,
            ratio: 2,
        })
    }

    fn upsample(&self, x: &Sig) -> Sig {
        let k = self.up.len();
        let stride = self.ratio;
        let pad = k / stride - 1;
        let pad_left = pad * stride + (k - stride) / 2;
        let pad_right = pad * stride + (k - stride + 1) / 2;
        // replicate-pad, transposed convolution, then the same trim the
        // reference takes
        let tp = x.t + 2 * pad;
        let full = (tp - 1) * stride + k;
        let mut out = Sig::zeros(x.c, full);
        for c in 0..x.c {
            for p in 0..tp {
                let src = (p as isize - pad as isize).clamp(0, x.t as isize - 1) as usize;
                let v = x.data[c * x.t + src] * self.ratio as f32;
                if v == 0.0 {
                    continue;
                }
                for a in 0..k {
                    out.data[c * full + p * stride + a] += v * self.up[a];
                }
            }
        }
        let (lo, hi) = (pad_left, full - pad_right);
        let t2 = hi - lo;
        let mut trimmed = Sig::zeros(x.c, t2);
        for c in 0..x.c {
            trimmed.data[c * t2..(c + 1) * t2].copy_from_slice(&out.data[c * full + lo..c * full + hi]);
        }
        trimmed
    }

    fn downsample(&self, x: &Sig) -> Sig {
        let k = self.down.len();
        let pad_left = k / 2 - if k % 2 == 0 { 1 } else { 0 };
        let pad_right = k / 2;
        let tp = x.t + pad_left + pad_right;
        let t2 = (tp - k) / self.ratio + 1;
        let mut out = Sig::zeros(x.c, t2);
        for c in 0..x.c {
            for p in 0..t2 {
                let mut acc = 0f32;
                for a in 0..k {
                    let s = (p * self.ratio + a) as isize - pad_left as isize;
                    let s = s.clamp(0, x.t as isize - 1) as usize;
                    acc += x.data[c * x.t + s] * self.down[a];
                }
                out.data[c * t2 + p] = acc;
            }
        }
        out
    }
}

/// `x + sin(αx)² / β`, with α and β kept in log space.
struct SnakeBeta {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    aa: Aliasing,
}

impl SnakeBeta {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<SnakeBeta, String> {
        Ok(SnakeBeta {
            alpha: tensor_f32(model, &format!("{p}.act.alpha"))?.0,
            beta: tensor_f32(model, &format!("{p}.act.beta"))?.0,
            aa: Aliasing::load(model, p)?,
        })
    }

    fn forward(&self, x: &Sig) -> Sig {
        let mut up = self.aa.upsample(x);
        for c in 0..up.c {
            let a = self.alpha[c].exp();
            let b = self.beta[c].exp();
            for v in up.data[c * up.t..(c + 1) * up.t].iter_mut() {
                let s = (*v * a).sin();
                *v += s * s / (b + 1e-9);
            }
        }
        self.aa.downsample(&up)
    }
}

/// One multi-receptive-field block: three dilated conv pairs, each wrapped
/// in its own anti-aliased activation, summed into the residual.
struct AmpBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acts1: Vec<SnakeBeta>,
    acts2: Vec<SnakeBeta>,
}

impl AmpBlock {
    fn load(model: &Arc<CmfModel>, p: &str, dil: &[usize]) -> Result<AmpBlock, String> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut acts1 = Vec::new();
        let mut acts2 = Vec::new();
        for (i, &d) in dil.iter().enumerate() {
            convs1.push(Conv1d::load(model, &format!("{p}.convs1.{i}"), d)?);
            convs2.push(Conv1d::load(model, &format!("{p}.convs2.{i}"), 1)?);
            acts1.push(SnakeBeta::load(model, &format!("{p}.acts1.{i}"))?);
            acts2.push(SnakeBeta::load(model, &format!("{p}.acts2.{i}"))?);
        }
        Ok(AmpBlock { convs1, convs2, acts1, acts2 })
    }

    fn forward(&self, x: &Sig, pool: Option<&Pool>) -> Sig {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let h = self.acts1[i].forward(&x);
            let h = self.convs1[i].forward(&h, pool);
            let h = self.acts2[i].forward(&h);
            let h = self.convs2[i].forward(&h, pool);
            for (v, &y) in x.data.iter_mut().zip(&h.data) {
                *v += y;
            }
        }
        x
    }
}

/// BigVGAN v2: `conv_pre`, then per level a transposed convolution and three
/// receptive-field blocks whose outputs are averaged, then `conv_post`.
pub struct Vocoder {
    conv_pre: Conv1d,
    ups: Vec<ConvT1d>,
    blocks: Vec<AmpBlock>,
    per_level: usize,
    act_post: SnakeBeta,
    conv_post: Conv1d,
    tanh_final: bool,
    apply_final: bool,
}

impl Vocoder {
    fn from_cmf(
        model: &Arc<CmfModel>,
        p: &str,
        rates: &[usize],
        dils: &[Vec<usize>],
        apply_final: bool,
    ) -> Result<Vocoder, String> {
        let mut ups = Vec::new();
        for (i, &r) in rates.iter().enumerate() {
            ups.push(ConvT1d::load(model, &format!("{p}.ups.{i}"), r)?);
        }
        let mut blocks = Vec::new();
        let mut i = 0usize;
        while model.tensor(&format!("{p}.resblocks.{i}.convs1.0.weight")).is_some() {
            blocks.push(AmpBlock::load(model, &format!("{p}.resblocks.{i}"), &dils[i % dils.len()])?);
            i += 1;
        }
        let per_level = blocks.len() / rates.len().max(1);
        Ok(Vocoder {
            conv_pre: Conv1d::load(model, &format!("{p}.conv_pre"), 1)?,
            ups,
            blocks,
            per_level,
            act_post: SnakeBeta::load(model, &format!("{p}.act_post"))?,
            conv_post: Conv1d::load(model, &format!("{p}.conv_post"), 1)?,
            tanh_final: false,
            apply_final,
        })
    }

    /// `[2, T, mel]` log-mel → `[2, T·∏rates]` waveform.
    fn forward(&self, mel: &Grid, pool: Option<&Pool>) -> Sig {
        // (channels, time, mel) → (channels·mel, time)
        let mut x = Sig::zeros(mel.c * mel.w, mel.h);
        for s in 0..mel.c {
            for m in 0..mel.w {
                let c = s * mel.w + m;
                for t in 0..mel.h {
                    x.data[c * mel.h + t] = mel.data[(s * mel.h + t) * mel.w + m];
                }
            }
        }
        let mut h = self.conv_pre.forward(&x, pool);
        for (i, up) in self.ups.iter().enumerate() {
            h = up.forward(&h);
            let mut acc: Option<Sig> = None;
            for j in 0..self.per_level {
                let b = &self.blocks[i * self.per_level + j];
                let o = b.forward(&h, pool);
                match &mut acc {
                    None => acc = Some(o),
                    Some(a) => {
                        for (v, &y) in a.data.iter_mut().zip(&o.data) {
                            *v += y;
                        }
                    }
                }
            }
            h = acc.unwrap();
            let inv = 1.0 / self.per_level as f32;
            h.data.iter_mut().for_each(|v| *v *= inv);
        }
        let h = self.act_post.forward(&h);
        let mut out = self.conv_post.forward(&h, pool);
        if self.apply_final {
            out.data.iter_mut().for_each(|v| {
                *v = if self.tanh_final { v.tanh() } else { v.clamp(-1.0, 1.0) }
            });
        }
        out
    }
}

/// The causal log-mel the bandwidth extender is conditioned on: an STFT
/// carried out as a convolution with the checkpoint's own DFT × Hann bases,
/// so the numbers match what the extender was trained against.
struct MelStft {
    forward_basis: Vec<f32>,
    mel_basis: Vec<f32>,
    n_freqs: usize,
    filter_len: usize,
    hop: usize,
    n_mels: usize,
}

impl MelStft {
    fn load(model: &Arc<CmfModel>, p: &str, hop: usize) -> Result<MelStft, String> {
        let (fb, fs) = tensor_f32(model, &format!("{p}.stft_fn.forward_basis"))?;
        let (mb, ms) = tensor_f32(model, &format!("{p}.mel_basis"))?;
        Ok(MelStft {
            n_freqs: fs[0] / 2,
            filter_len: fs[2],
            forward_basis: fb,
            mel_basis: mb,
            n_mels: ms[0],
            hop,
        })
    }

    /// `[C, T]` waveform → `[C, frames, n_mels]` log-mel.
    fn forward(&self, x: &Sig) -> Grid {
        let left = self.filter_len.saturating_sub(self.hop);
        let padded = x.t + left;
        let frames = if padded >= self.filter_len {
            (padded - self.filter_len) / self.hop + 1
        } else {
            0
        };
        let mut out = Grid::zeros(x.c, frames, self.n_mels);
        let rows = 2 * self.n_freqs;
        let mut mag = vec![0f32; self.n_freqs];
        for c in 0..x.c {
            for f in 0..frames {
                let start = f * self.hop;
                for r in 0..self.n_freqs {
                    let mut re = 0f32;
                    let mut im = 0f32;
                    for j in 0..self.filter_len {
                        let idx = start + j;
                        let v = if idx < left {
                            0.0
                        } else {
                            let s = idx - left;
                            if s < x.t { x.data[c * x.t + s] } else { 0.0 }
                        };
                        re += v * self.forward_basis[r * self.filter_len + j];
                        im += v * self.forward_basis[(self.n_freqs + r) * self.filter_len + j];
                    }
                    mag[r] = (re * re + im * im).sqrt();
                }
                for m in 0..self.n_mels {
                    let mut acc = 0f32;
                    for r in 0..self.n_freqs {
                        acc += self.mel_basis[m * self.n_freqs + r] * mag[r];
                    }
                    out.data[(c * frames + f) * self.n_mels + m] = acc.max(1e-5).ln();
                }
            }
        }
        let _ = rows;
        out
    }
}

/// A Hann-windowed sinc resampler, the ×3 skip path from 16 kHz to 48 kHz.
/// The reference does not store this filter, so it is rebuilt here.
fn hann_sinc_upsample(x: &Sig, ratio: usize) -> Sig {
    let rolloff = 0.99f64;
    let lpw = 6f64;
    let width = (lpw / rolloff).ceil() as usize;
    let k = 2 * width * ratio + 1;
    let pad = width;
    let pad_left = 2 * width * ratio;
    let pad_right = k - ratio;
    let filt: Vec<f32> = (0..k)
        .map(|i| {
            let ta = (i as f64 / ratio as f64 - width as f64) * rolloff;
            let tc = ta.clamp(-lpw, lpw);
            let win = (tc * std::f64::consts::PI / lpw / 2.0).cos().powi(2);
            let s = if ta == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * ta).sin() / (std::f64::consts::PI * ta)
            };
            (s * win * rolloff / ratio as f64) as f32
        })
        .collect();
    let tp = x.t + 2 * pad;
    let full = (tp - 1) * ratio + k;
    let mut acc = Sig::zeros(x.c, full);
    for c in 0..x.c {
        for p in 0..tp {
            let src = (p as isize - pad as isize).clamp(0, x.t as isize - 1) as usize;
            let v = x.data[c * x.t + src] * ratio as f32;
            if v == 0.0 {
                continue;
            }
            for a in 0..k {
                acc.data[c * full + p * ratio + a] += v * filt[a];
            }
        }
    }
    let (lo, hi) = (pad_left, full - pad_right);
    let t2 = hi - lo;
    let mut out = Sig::zeros(x.c, t2);
    for c in 0..x.c {
        out.data[c * t2..(c + 1) * t2].copy_from_slice(&acc.data[c * full + lo..c * full + hi]);
    }
    out
}

/// The whole audio tail: latent → spectrogram → 16 kHz → 48 kHz.
pub struct AudioStack {
    pub decoder: AudioVaeDecoder,
    vocoder: Vocoder,
    bwe: Vocoder,
    mel: MelStft,
    hop: usize,
    in_rate: usize,
    pub out_rate: usize,
}

impl AudioStack {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<AudioStack, String> {
        let cfg: serde_json::Value = ["avae.config_json"]
            .iter()
            .filter_map(|n| model.tensor(n).map(|e| model.entry_bytes(e)))
            .filter_map(|b| serde_json::from_slice(b).ok())
            .next()
            .unwrap_or(serde_json::Value::Null);
        let voc = cfg.pointer("/vocoder/vocoder").cloned().unwrap_or_default();
        let bwe = cfg.pointer("/vocoder/bwe").cloned().unwrap_or_default();
        let rates = |v: &serde_json::Value, d: Vec<usize>| -> Vec<usize> {
            v.get("upsample_rates")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or(d)
        };
        let dils = vec![vec![1usize, 3, 5], vec![1, 3, 5], vec![1, 3, 5]];
        let hop = bwe.get("hop_length").and_then(|v| v.as_u64()).unwrap_or(80) as usize;
        Ok(AudioStack {
            decoder: AudioVaeDecoder::from_cmf(model)?,
            vocoder: Vocoder::from_cmf(
                model,
                "avae.vocoder.vocoder",
                &rates(&voc, vec![5, 2, 2, 2, 2, 2]),
                &dils,
                false,
            )?,
            bwe: Vocoder::from_cmf(
                model,
                "avae.vocoder.bwe_generator",
                &rates(&bwe, vec![6, 5, 2, 2, 2]),
                &dils,
                false,
            )?,
            mel: MelStft::load(model, "avae.vocoder.mel_stft", hop)?,
            hop,
            in_rate: bwe.get("input_sampling_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as usize,
            out_rate: bwe.get("output_sampling_rate").and_then(|v| v.as_u64()).unwrap_or(48000) as usize,
        })
    }

    /// `[8, T, 16]` latent → stereo waveform at `out_rate`.
    pub fn decode(&self, latent: &Grid, pool: Option<&Pool>) -> Sig {
        let mel = self.decoder.decode(latent, pool);
        let low = self.vocoder.forward(&mel, pool);
        let out_len = low.t * self.out_rate / self.in_rate;
        // pad to a whole number of hops so the mel frame count is exact
        let rem = low.t % self.hop;
        let padded = if rem == 0 {
            low.clone()
        } else {
            let t2 = low.t + self.hop - rem;
            let mut p = Sig::zeros(low.c, t2);
            for c in 0..low.c {
                p.data[c * t2..c * t2 + low.t].copy_from_slice(&low.data[c * low.t..(c + 1) * low.t]);
            }
            p
        };
        let m = self.mel.forward(&padded);
        let residual = self.bwe.forward(&m, pool);
        let skip = hann_sinc_upsample(&padded, self.out_rate / self.in_rate);
        let t = residual.t.min(skip.t).min(out_len);
        let mut out = Sig::zeros(skip.c, t);
        for c in 0..skip.c {
            for i in 0..t {
                out.data[c * t + i] =
                    (residual.data[c * residual.t + i] + skip.data[c * skip.t + i]).clamp(-1.0, 1.0);
            }
        }
        out
    }
}

/// 16-bit PCM WAV — the one container every tool and browser reads.
pub fn write_wav(path: &std::path::Path, sig: &Sig, rate: usize) -> std::io::Result<()> {
    use std::io::Write;
    let n = sig.t;
    let ch = sig.c as u16;
    let bytes = (n * sig.c * 2) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + bytes).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&ch.to_le_bytes())?;
    f.write_all(&(rate as u32).to_le_bytes())?;
    f.write_all(&((rate * sig.c * 2) as u32).to_le_bytes())?;
    f.write_all(&((sig.c * 2) as u16).to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&bytes.to_le_bytes())?;
    for i in 0..n {
        for c in 0..sig.c {
            let v = (sig.data[c * n + i].clamp(-1.0, 1.0) * 32767.0) as i16;
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}
