//! The LTX-2.5 video VAE **encoder** — pixels into the latent space the
//! transformer denoises in, which is what every conditioned mode needs:
//! image-to-video, video-to-video, and any of the audio-video pairs that
//! start from a picture.
//!
//! It mirrors the decoder and runs it backwards: `patchify(4)` trades
//! spatial resolution for channel depth, a causal 3-D convolution lifts it
//! to 128 channels, then the block ladder from the checkpoint's own
//! `encoder_blocks` — `res_x` stacks that keep the shape and `compress_all`
//! convolutions with stride 2 in every axis that do not. PixelNorm and SiLU,
//! a final convolution to 129 channels (128 means and one shared
//! log-variance the sampler ignores), and the per-channel statistics
//! *normalize* the means on the way out.
//!
//! Causal in time, like the decoder: the first latent frame sees one pixel
//! frame and every later one sees eight, which is why a single image encodes
//! to exactly one latent frame.

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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn pixel_norm(x: &mut Vol) {
    let npos = x.positions();
    for p in 0..npos {
        let mut s = 0f32;
        for c in 0..x.c {
            let v = x.data[c * npos + p];
            s += v * v;
        }
        let inv = 1.0 / (s / x.c as f32 + 1e-6).sqrt();
        for c in 0..x.c {
            x.data[c * npos + p] *= inv;
        }
    }
}

/// A 3×3×3 convolution, causal in time (the kernel reaches backwards only,
/// with the first frame replicated) and zero-padded in space, at stride 1
/// or 2 on any axis.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    c_out: usize,
    c_in: usize,
    stride: (usize, usize, usize),
}

impl Conv {
    fn load(model: &Arc<CmfModel>, p: &str, stride: (usize, usize, usize)) -> Result<Conv, String> {
        let (w, s) = tensor_f32(model, &format!("{p}.conv.weight"))?;
        let b = match model.tensor(&format!("{p}.conv.bias")) {
            Some(_) => tensor_f32(model, &format!("{p}.conv.bias"))?.0,
            None => vec![0.0; s[0]],
        };
        Ok(Conv {
            w,
            b,
            c_out: s[0],
            c_in: s[1],
            stride,
        })
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let (sf, sh, sw) = self.stride;
        let (of, oh, ow) = (x.f.div_ceil(sf), x.h.div_ceil(sh), x.w.div_ceil(sw));
        let mut out = Vol::zeros(self.c_out, of, oh, ow);
        let k = self.c_in * 27;
        let npos = of * oh * ow;
        const CHUNK: usize = 8192;
        let mut patches = vec![0f32; CHUNK.min(npos) * k];
        let mut ys = vec![0f32; CHUNK.min(npos) * self.c_out];
        let mut p0 = 0usize;
        while p0 < npos {
            let n = CHUNK.min(npos - p0);
            patches[..n * k].fill(0.0);
            for i in 0..n {
                let p = p0 + i;
                let (pw, rest) = (p % ow, p / ow);
                let (ph, pf) = (rest % oh, rest / oh);
                let (bf, bh, bw) = (pf * sf, ph * sh, pw * sw);
                let row = &mut patches[i * k..(i + 1) * k];
                for ci in 0..self.c_in {
                    for kf in 0..3usize {
                        // causal in time: taps reach back, the first frame
                        // stands in for anything before it
                        let src_f = (bf + kf).saturating_sub(2).min(x.f - 1);
                        for kh in 0..3usize {
                            let s = bh + kh;
                            if s == 0 || s > x.h {
                                continue;
                            }
                            let src_h = s - 1;
                            for kw in 0..3usize {
                                let s = bw + kw;
                                if s == 0 || s > x.w {
                                    continue;
                                }
                                row[(ci * 3 + kf) * 9 + kh * 3 + kw] =
                                    x.at(ci, src_f, src_h, s - 1);
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
                for co in 0..self.c_out {
                    out.data[co * npos + p0 + i] = ys[i * self.c_out + co] + self.b[co];
                }
            }
            p0 += n;
        }
        out
    }
}

struct Res {
    c1: Conv,
    c2: Conv,
}

impl Res {
    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let mut h = x.clone();
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.c1.forward(&h, pool);
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.c2.forward(&h, pool);
        for (v, &r) in h.data.iter_mut().zip(&x.data) {
            *v += r;
        }
        h
    }
}

/// `SpaceToDepthDownsample`: a stride-1 convolution whose output is folded
/// into the channel axis, plus a skip that folds the *input* the same way
/// and averages each group of channels down to the same width. The temporal
/// stride prepends a copy of the first frame, which is what keeps a
/// `1 + 8k` frame count landing on `1 + k` latent frames.
struct Down {
    conv: Conv,
    stride: (usize, usize, usize),
    out_channels: usize,
}

impl Down {
    fn load(
        model: &Arc<CmfModel>,
        p: &str,
        stride: (usize, usize, usize),
        multiplier: usize,
    ) -> Result<Down, String> {
        let conv = Conv::load(model, &format!("{p}.conv"), (1, 1, 1))?;
        let out_channels = conv.c_in * multiplier;
        Ok(Down {
            conv,
            stride,
            out_channels,
        })
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let (sf, sh, sw) = self.stride;
        let padded;
        let src = if sf == 2 {
            let mut v = Vol::zeros(x.c, x.f + 1, x.h, x.w);
            for c in 0..x.c {
                for f in 0..=x.f {
                    let from = f.saturating_sub(1);
                    for y in 0..x.h {
                        for z in 0..x.w {
                            v.data[((c * (x.f + 1) + f) * x.h + y) * x.w + z] = x.at(c, from, y, z);
                        }
                    }
                }
            }
            padded = v;
            &padded
        } else {
            x
        };
        let folded = space_to_depth(src, self.stride);
        // the skip: average each consecutive group of folded channels down
        let group = folded.c / self.out_channels;
        let np = folded.positions();
        let mut skip = Vol::zeros(self.out_channels, folded.f, folded.h, folded.w);
        for c in 0..self.out_channels {
            for g in 0..group {
                let sc = c * group + g;
                for i in 0..np {
                    skip.data[c * np + i] += folded.data[sc * np + i];
                }
            }
            let inv = 1.0 / group as f32;
            for i in 0..np {
                skip.data[c * np + i] *= inv;
            }
        }
        let mut out = space_to_depth(&self.conv.forward(src, pool), self.stride);
        for (v, &r) in out.data.iter_mut().zip(&skip.data) {
            *v += r;
        }
        out
    }
}

/// `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`.
fn space_to_depth(x: &Vol, stride: (usize, usize, usize)) -> Vol {
    let (p1, p2, p3) = stride;
    let (df, dh, dw) = (x.f / p1, x.h / p2, x.w / p3);
    let mut out = Vol::zeros(x.c * p1 * p2 * p3, df, dh, dw);
    let np = df * dh * dw;
    for c in 0..x.c {
        for i1 in 0..p1 {
            for i2 in 0..p2 {
                for i3 in 0..p3 {
                    let dc = ((c * p1 + i1) * p2 + i2) * p3 + i3;
                    for f in 0..df {
                        for y in 0..dh {
                            for z in 0..dw {
                                out.data[dc * np + (f * dh + y) * dw + z] =
                                    x.at(c, f * p1 + i1, y * p2 + i2, z * p3 + i3);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

enum Block {
    Res(Vec<Res>),
    Down(Down),
}

pub struct VideoEncoder {
    patch: usize,
    conv_in: Conv,
    blocks: Vec<Block>,
    conv_out: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
    latent_channels: usize,
}

impl VideoEncoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<VideoEncoder, String> {
        let cfg: serde_json::Value = ["vvae.config_json", "ltx.config_json"]
            .iter()
            .filter_map(|n| model.tensor(n).map(|e| model.entry_bytes(e)))
            .filter_map(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .find(|c| c.get("vae").and_then(|v| v.get("encoder_blocks")).is_some())
            .ok_or("no config in this container carries vae.encoder_blocks")?;
        let vae = &cfg["vae"];
        let patch = vae["patch_size"].as_u64().unwrap_or(4) as usize;
        let list = vae["encoder_blocks"]
            .as_array()
            .ok_or("vae.encoder_blocks")?;
        let mut blocks = Vec::new();
        for (i, entry) in list.iter().enumerate() {
            let name = entry[0].as_str().unwrap_or("");
            let p = format!("vvae.encoder.down_blocks.{i}");
            match name {
                "res_x" => {
                    let mut r = Vec::new();
                    let mut j = 0usize;
                    while model
                        .tensor(&format!("{p}.res_blocks.{j}.conv1.conv.weight"))
                        .is_some()
                    {
                        r.push(Res {
                            c1: Conv::load(model, &format!("{p}.res_blocks.{j}.conv1"), (1, 1, 1))?,
                            c2: Conv::load(model, &format!("{p}.res_blocks.{j}.conv2"), (1, 1, 1))?,
                        });
                        j += 1;
                    }
                    blocks.push(Block::Res(r));
                }
                "compress_all_res" | "compress_time_res" | "compress_space_res" => {
                    let stride = match name {
                        "compress_time_res" => (2, 1, 1),
                        "compress_space_res" => (1, 2, 2),
                        _ => (2, 2, 2),
                    };
                    let mult = entry[1]
                        .get("multiplier")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2) as usize;
                    blocks.push(Block::Down(Down::load(model, &p, stride, mult)?));
                }
                other => return Err(format!("encoder block '{other}' is not ported")),
            }
        }
        Ok(VideoEncoder {
            patch,
            conv_in: Conv::load(model, "vvae.encoder.conv_in", (1, 1, 1))?,
            blocks,
            conv_out: Conv::load(model, "vvae.encoder.conv_out", (1, 1, 1))?,
            mean: tensor_f32(model, "vvae.per_channel_statistics.mean-of-means")?.0,
            std: tensor_f32(model, "vvae.per_channel_statistics.std-of-means")?.0,
            latent_channels: vae["latent_channels"].as_u64().unwrap_or(128) as usize,
        })
    }

    /// `[3, F, H, W]` in `[-1, 1]` → `[128, 1+(F-1)/8, H/32, W/32]`, in the
    /// units the transformer works in.
    pub fn encode(&self, frames: &Vol, pool: Option<&Pool>) -> Vol {
        let p = self.patch;
        let (h2, w2) = (frames.h / p, frames.w / p);
        // patchify is `b c (f p) (h q) (w r) -> b (c p r q) f h w`: the
        // *width* offset varies slower than the height one, which is the
        // opposite of the obvious order and the difference between a picture
        // and a grid of shuffled tiles.
        let mut x = Vol::zeros(frames.c * p * p, frames.f, h2, w2);
        let np = x.positions();
        for c in 0..frames.c {
            for a in 0..p {
                for b in 0..p {
                    let dc = (c * p + b) * p + a;
                    for f in 0..frames.f {
                        for y in 0..h2 {
                            for z in 0..w2 {
                                x.data[dc * np + (f * h2 + y) * w2 + z] =
                                    frames.at(c, f, y * p + a, z * p + b);
                            }
                        }
                    }
                }
            }
        }
        let mut h = self.conv_in.forward(&x, pool);
        for b in &self.blocks {
            h = match b {
                Block::Res(rs) => {
                    let mut cur = h;
                    for r in rs {
                        cur = r.forward(&cur, pool);
                    }
                    cur
                }
                Block::Down(c) => c.forward(&h, pool),
            };
        }
        pixel_norm(&mut h);
        h.data.iter_mut().for_each(|v| *v = silu(*v));
        let out = self.conv_out.forward(&h, pool);
        // the means are the first `latent_channels`; the trailing channel is
        // the shared log-variance, which sampling ignores
        let npos = out.positions();
        let mut lat = Vol::zeros(self.latent_channels, out.f, out.h, out.w);
        for c in 0..self.latent_channels {
            for i in 0..npos {
                lat.data[c * npos + i] = (out.data[c * npos + i] - self.mean[c]) / self.std[c];
            }
        }
        lat
    }
}
