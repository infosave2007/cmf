//! The LTX-2.5 spatial latent upscaler — the ×2 between the two denoising
//! stages.
//!
//! Structurally simple and computationally not: 3-D convolutions at 1024
//! channels with ordinary zero padding, GroupNorm(32) and SiLU, four
//! residual blocks before a pixel-shuffle upsample and four after. It runs
//! on the *un-normalized* latent — the video VAE's per-channel statistics
//! are undone going in and reapplied coming out — because the upscaler was
//! trained in the VAE's own units, not the diffusion model's.

use crate::ltxvae::Vol;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

fn tensor_f32(model: &Arc<CmfModel>, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = model
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let mut out = vec![0.0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, model.entry_bytes(e), &mut out)?;
    Ok((out, e.shape.clone()))
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `[out, in, kf, kh, kw]` convolution with zero padding of one on every
/// axis, as im2col into a single GEMM per position chunk.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    c_out: usize,
    c_in: usize,
    kf: usize,
    kh: usize,
    kw: usize,
}

impl Conv {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Conv, String> {
        let (w, s) = tensor_f32(model, &format!("{name}.weight"))?;
        let (b, _) = tensor_f32(model, &format!("{name}.bias"))?;
        let (kf, kh, kw) = match s.len() {
            5 => (s[2], s[3], s[4]),
            4 => (1, s[2], s[3]),
            _ => return Err(format!("{name}: rank {}", s.len())),
        };
        Ok(Conv {
            w,
            b,
            c_out: s[0],
            c_in: s[1],
            kf,
            kh,
            kw,
        })
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let (f, h, w) = (x.f, x.h, x.w);
        let npos = f * h * w;
        let k = self.c_in * self.kf * self.kh * self.kw;
        let mut out = Vol::zeros(self.c_out, f, h, w);
        const CHUNK: usize = 4096;
        let mut patches = vec![0f32; CHUNK * k];
        let mut ys = vec![0f32; CHUNK * self.c_out];
        let (pf, ph, pw) = (self.kf / 2, self.kh / 2, self.kw / 2);
        let mut p0 = 0usize;
        while p0 < npos {
            let n = CHUNK.min(npos - p0);
            // the buffer is reused across chunks — a stale row would become
            // the padding value
            patches[..n * k].fill(0.0);
            for i in 0..n {
                let p = p0 + i;
                let (pw_, rest) = (p % w, p / w);
                let (ph_, pf_) = (rest % h, rest / h);
                for ci in 0..self.c_in {
                    for a in 0..self.kf {
                        let sf = pf_ as isize + a as isize - pf as isize;
                        if sf < 0 || sf >= f as isize {
                            continue;
                        }
                        for bb in 0..self.kh {
                            let sh = ph_ as isize + bb as isize - ph as isize;
                            if sh < 0 || sh >= h as isize {
                                continue;
                            }
                            for c in 0..self.kw {
                                let sw = pw_ as isize + c as isize - pw as isize;
                                if sw < 0 || sw >= w as isize {
                                    continue;
                                }
                                let v = x.data
                                    [((ci * f + sf as usize) * h + sh as usize) * w + sw as usize];
                                patches
                                    [i * k + ((ci * self.kf + a) * self.kh + bb) * self.kw + c] = v;
                            }
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
                let p = p0 + i;
                for co in 0..self.c_out {
                    out.data[co * npos + p] = ys[i * self.c_out + co] + self.b[co];
                }
            }
            p0 += n;
        }
        out
    }
}

/// GroupNorm(32) with affine, in place.
fn group_norm(x: &mut Vol, w: &[f32], b: &[f32]) {
    let groups = 32usize;
    let per = x.c / groups;
    let npos = x.positions();
    for g in 0..groups {
        let (lo, hi) = (g * per * npos, (g + 1) * per * npos);
        let s = &x.data[lo..hi];
        let n = s.len() as f64;
        let mean = s.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = s
            .iter()
            .map(|&v| (v as f64 - mean) * (v as f64 - mean))
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for c in 0..per {
            let ch = g * per + c;
            let (gw, gb) = (w[ch], b[ch]);
            for v in x.data[ch * npos..(ch + 1) * npos].iter_mut() {
                *v = ((*v as f64 - mean) * inv) as f32 * gw + gb;
            }
        }
    }
}

struct ResBlock {
    conv1: Conv,
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    conv2: Conv,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
}

impl ResBlock {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<ResBlock, String> {
        Ok(ResBlock {
            conv1: Conv::load(model, &format!("{p}.conv1"))?,
            n1w: tensor_f32(model, &format!("{p}.norm1.weight"))?.0,
            n1b: tensor_f32(model, &format!("{p}.norm1.bias"))?.0,
            conv2: Conv::load(model, &format!("{p}.conv2"))?,
            n2w: tensor_f32(model, &format!("{p}.norm2.weight"))?.0,
            n2b: tensor_f32(model, &format!("{p}.norm2.bias"))?.0,
        })
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let mut h = self.conv1.forward(x, pool);
        group_norm(&mut h, &self.n1w, &self.n1b);
        for v in h.data.iter_mut() {
            *v = silu(*v);
        }
        let mut h2 = self.conv2.forward(&h, pool);
        group_norm(&mut h2, &self.n2w, &self.n2b);
        // the activation comes *after* the residual add, not before
        for (v, &r) in h2.data.iter_mut().zip(&x.data) {
            *v = silu(*v + r);
        }
        h2
    }
}

pub struct LatentUpscaler {
    initial: Conv,
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    pre: Vec<ResBlock>,
    up: Conv,
    post: Vec<ResBlock>,
    final_conv: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl LatentUpscaler {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<LatentUpscaler, String> {
        let load_blocks = |p: &str| -> Result<Vec<ResBlock>, String> {
            let mut v = Vec::new();
            let mut i = 0;
            while model.tensor(&format!("{p}.{i}.conv1.weight")).is_some() {
                v.push(ResBlock::load(model, &format!("{p}.{i}"))?);
                i += 1;
            }
            Ok(v)
        };
        Ok(LatentUpscaler {
            initial: Conv::load(model, "ups.initial_conv")?,
            in_w: tensor_f32(model, "ups.initial_norm.weight")?.0,
            in_b: tensor_f32(model, "ups.initial_norm.bias")?.0,
            pre: load_blocks("ups.res_blocks")?,
            up: Conv::load(model, "ups.upsampler.0")?,
            post: load_blocks("ups.post_upsample_res_blocks")?,
            final_conv: Conv::load(model, "ups.final_conv")?,
            mean: tensor_f32(model, "vvae.per_channel_statistics.mean-of-means")?.0,
            std: tensor_f32(model, "vvae.per_channel_statistics.std-of-means")?.0,
        })
    }

    /// `[128, F, H, W]` → `[128, F, 2H, 2W]`, in the diffusion model's units
    /// on both sides.
    pub fn upscale(&self, latent: &Vol, pool: Option<&Pool>) -> Vol {
        let npos = latent.positions();
        let mut x = latent.clone();
        // into the VAE's units
        for c in 0..x.c {
            for v in x.data[c * npos..(c + 1) * npos].iter_mut() {
                *v = *v * self.std[c] + self.mean[c];
            }
        }
        let mut h = self.initial.forward(&x, pool);
        group_norm(&mut h, &self.in_w, &self.in_b);
        for v in h.data.iter_mut() {
            *v = silu(*v);
        }
        for b in &self.pre {
            h = b.forward(&h, pool);
        }
        h = self.pixel_shuffle(&self.up.forward(&h, pool));
        for b in &self.post {
            h = b.forward(&h, pool);
        }
        let mut out = self.final_conv.forward(&h, pool);
        let npos2 = out.positions();
        for c in 0..out.c {
            for v in out.data[c * npos2..(c + 1) * npos2].iter_mut() {
                *v = (*v - self.mean[c]) / self.std[c];
            }
        }
        out
    }

    /// `b (c p1 p2) f h w -> b c f (h p1) (w p2)`.
    fn pixel_shuffle(&self, x: &Vol) -> Vol {
        let c = x.c / 4;
        let (h2, w2) = (x.h * 2, x.w * 2);
        let mut out = Vol::zeros(c, x.f, h2, w2);
        for ch in 0..c {
            for p1 in 0..2 {
                for p2 in 0..2 {
                    let src = (ch * 2 + p1) * 2 + p2;
                    for f in 0..x.f {
                        for y in 0..x.h {
                            for z in 0..x.w {
                                let v = x.data[((src * x.f + f) * x.h + y) * x.w + z];
                                out.data[((ch * x.f + f) * h2 + y * 2 + p1) * w2 + z * 2 + p2] = v;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}
