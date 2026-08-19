//! The MiniMax-H3 latent upscaler: a 3-D conv net that resizes the DiT's
//! own latent instead of decoding, scaling pixels and encoding again.
//!
//! Why it exists. A 5 B-parameter video VAE round trip is the expensive way
//! to change resolution, and interpolating the latent directly is the cheap
//! way that ghosts. This net — 345 M parameters, published by LBH-123-AI —
//! learns the map: twelve residual blocks at the source size, a trilinear
//! resize in the middle, twelve more at the target size, with a scalar
//! "how much am I scaling" embedding modulating every block.
//!
//! ```text
//! z ── (z − mean)/std ── conv_in ── [Res × 12, Temporal × 6] ──┐
//!                                                              trilinear
//!   z' ── ·std + mean ── conv_out ── silu ── norm ── [same again]
//! ```
//!
//! Three details that are easy to get wrong and are gated by
//! `examples/mmh3_ups_parity.rs` against a torch reference:
//!
//! * **The block list is a flat `ModuleList`.** `temporal_every = 2` means a
//!   `TemporalConv` follows blocks 0, 2, 4 …, so the sequence is Res, Temp,
//!   Res, Res, Temp, Res, … and the state dict indexes THAT, not the
//!   residual blocks. We dispatch on which keys an index carries rather
//!   than assuming a pattern.
//! * **The modulation is `out_norm(h)·(1 + scale) + shift`,** applied to the
//!   *second* norm, after the first convolution — not the adaLN order the
//!   DiT next door uses.
//! * **The scale embedding takes `scale − 1`**, so an identity resize feeds
//!   zero, and the network was trained with that offset.
//!
//! The latent statistics are the release's own 24-channel mean/std; they are
//! constants of the VAE, not of this net, and they are applied outside it.

use crate::pool::Pool;
use std::collections::HashMap;
use std::path::Path;

/// Per-channel latent statistics of the H3 video VAE — the values the
/// upscaler was trained against, from the release's training code.
pub const LATENT_MEAN: [f32; 24] = [
    0.858_090_34,
    -0.960_659_15,
    1.066_164,
    -0.509_032_55,
    -0.272_758_19,
    -1.367_541_4,
    -0.255_325_5,
    -0.269_075_54,
    -0.537_684_08,
    -0.046_409_73,
    0.665_737_03,
    0.196_901_28,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_341_99,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_44,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];
pub const LATENT_STD: [f32; 24] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_8,
    0.799_680_95,
    0.449_889,
    0.719_739_97,
    0.693_629_3,
    2.961_095_1,
    2.769_419_9,
    3.049_618_5,
    2.108_805_4,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];

/// Read a `.safetensors` of f32 tensors — the parity oracle's own format.
pub fn read_oracle(
    path: &Path,
) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    crate::ltxlora::read_safetensors(path)
}

/// A latent volume, channel-major: `[c][t][h][w]`.
#[derive(Clone)]
pub struct Vol {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vol {
    fn zeros(c: usize, t: usize, h: usize, w: usize) -> Vol {
        Vol { c, t, h, w, data: vec![0f32; c * t * h * w] }
    }
    #[inline]
    fn at(&self, c: usize, t: usize, h: usize, w: usize) -> f32 {
        self.data[((c * self.t + t) * self.h + h) * self.w + w]
    }
}

/// A dense 3-D convolution with a 3×3×3 or 1×1×1 kernel and "same" padding.
struct Conv3 {
    w: Vec<f32>, // [c_out, c_in, kt, kh, kw]
    b: Vec<f32>,
    c_out: usize,
    c_in: usize,
    k: usize, // 1 or 3, cubic
}

impl Conv3 {
    /// `dst = conv(x)`. Implemented as im2col + `gemm_nt` one temporal
    /// slice at a time: the patch matrix for a whole volume at 512
    /// channels is gigabytes, and for one slice it is a hundred megabytes.
    fn apply(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let (t, h, w) = (x.t, x.h, x.w);
        let mut out = Vol::zeros(self.c_out, t, h, w);
        let k = self.k;
        let pad = k / 2;
        let patch = self.c_in * k * k * k;
        let n = h * w;
        let mut col = vec![0f32; n * patch];
        let mut acc = vec![0f32; n * self.c_out];
        for ti in 0..t {
            // im2col for this slice: row p is the neighbourhood of voxel p.
            col.iter_mut().for_each(|v| *v = 0.0);
            for hi in 0..h {
                for wi in 0..w {
                    let row = (hi * w + wi) * patch;
                    for ci in 0..self.c_in {
                        for kt in 0..k {
                            let tt = ti as isize + kt as isize - pad as isize;
                            if tt < 0 || tt >= t as isize {
                                continue;
                            }
                            for kh in 0..k {
                                let hh = hi as isize + kh as isize - pad as isize;
                                if hh < 0 || hh >= h as isize {
                                    continue;
                                }
                                for kw in 0..k {
                                    let ww = wi as isize + kw as isize - pad as isize;
                                    if ww < 0 || ww >= w as isize {
                                        continue;
                                    }
                                    let idx = ((ci * k + kt) * k + kh) * k + kw;
                                    col[row + idx] =
                                        x.at(ci, tt as usize, hh as usize, ww as usize);
                                }
                            }
                        }
                    }
                }
            }
            crate::gpu::cpu_scope(|| {
                crate::fcd_ops::gemm_nt(&col, &self.w, &mut acc, n, patch, self.c_out, pool);
            });
            for p in 0..n {
                let (hi, wi) = (p / w, p % w);
                for co in 0..self.c_out {
                    out.data[((co * t + ti) * h + hi) * w + wi] = acc[p * self.c_out + co] + self.b[co];
                }
            }
        }
        out
    }
}

/// The temporal depthwise convolution: one channel, one `[k,1,1]` kernel.
struct DepthwiseT {
    w: Vec<f32>, // [c, 1, k, 1, 1]
    b: Vec<f32>,
    k: usize,
}

impl DepthwiseT {
    fn apply(&self, x: &Vol) -> Vol {
        let mut out = Vol::zeros(x.c, x.t, x.h, x.w);
        let pad = self.k / 2;
        for c in 0..x.c {
            for ti in 0..x.t {
                for hi in 0..x.h {
                    for wi in 0..x.w {
                        let mut acc = self.b[c];
                        for kt in 0..self.k {
                            let tt = ti as isize + kt as isize - pad as isize;
                            if tt < 0 || tt >= x.t as isize {
                                continue;
                            }
                            acc += self.w[c * self.k + kt] * x.at(c, tt as usize, hi, wi);
                        }
                        out.data[((c * x.t + ti) * x.h + hi) * x.w + wi] = acc;
                    }
                }
            }
        }
        out
    }
}

/// `GroupNorm(32)` over the channel axis, per (t, h, w) volume.
struct GroupNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    groups: usize,
}

impl GroupNorm {
    fn apply(&self, x: &Vol) -> Vol {
        let mut out = x.clone();
        let per = x.c / self.groups;
        let n = x.t * x.h * x.w;
        for g in 0..self.groups {
            let (lo, hi) = (g * per, (g + 1) * per);
            let mut sum = 0f64;
            let mut sq = 0f64;
            for c in lo..hi {
                for i in 0..n {
                    let v = x.data[c * n + i] as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let cnt = (per * n) as f64;
            let mean = sum / cnt;
            let var = (sq / cnt - mean * mean).max(0.0);
            let inv = 1.0 / (var + 1e-5).sqrt();
            for c in lo..hi {
                for i in 0..n {
                    let v = (x.data[c * n + i] as f64 - mean) * inv;
                    out.data[c * n + i] = v as f32 * self.w[c] + self.b[c];
                }
            }
        }
        out
    }
}

fn silu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x /= 1.0 + (-*x).exp();
    }
}

struct ResBlock {
    in_norm: GroupNorm,
    in_conv: Conv3,
    emb: (Vec<f32>, Vec<f32>), // [2C, E] and [2C]
    out_norm: GroupNorm,
    out_conv: Conv3,
    skip: Option<Conv3>,
}

impl ResBlock {
    fn apply(&self, x: &Vol, emb: &[f32], pool: Option<&Pool>) -> Vol {
        let mut h = self.in_norm.apply(x);
        silu_in_place(&mut h.data);
        let h = self.in_conv.apply(&h, pool);

        // emb_layers: SiLU then Linear, giving [scale | shift] over channels.
        let e_in = emb.len();
        let two_c = self.emb.1.len();
        let mut es = emb.to_vec();
        silu_in_place(&mut es);
        let mut mod_v = vec![0f32; two_c];
        for (o, m) in mod_v.iter_mut().enumerate() {
            let row = &self.emb.0[o * e_in..(o + 1) * e_in];
            *m = self.emb.1[o] + row.iter().zip(&es).map(|(a, b)| a * b).sum::<f32>();
        }

        let mut h2 = self.out_norm.apply(&h);
        let n = h2.t * h2.h * h2.w;
        let c = h2.c;
        for ci in 0..c {
            let (s, sh) = (mod_v[ci], mod_v[c + ci]);
            for i in 0..n {
                h2.data[ci * n + i] = h2.data[ci * n + i] * (1.0 + s) + sh;
            }
        }
        silu_in_place(&mut h2.data);
        let h2 = self.out_conv.apply(&h2, pool);

        let mut out = match &self.skip {
            Some(cv) => cv.apply(x, pool),
            None => x.clone(),
        };
        for (o, v) in out.data.iter_mut().zip(&h2.data) {
            *o += *v;
        }
        out
    }
}

struct TemporalConv {
    norm: GroupNorm,
    dw: DepthwiseT,
    pw: Conv3, // 1×1×1
}

impl TemporalConv {
    fn apply(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let mut h = self.norm.apply(x);
        silu_in_place(&mut h.data);
        let h = self.dw.apply(&h);
        let h = self.pw.apply(&h, pool);
        let mut out = x.clone();
        for (o, v) in out.data.iter_mut().zip(&h.data) {
            *o += *v;
        }
        out
    }
}

enum Block {
    Res(ResBlock),
    Temporal(TemporalConv),
}

pub struct LatentUpscaler {
    conv_in: Conv3,
    embed: ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)),
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: GroupNorm,
    conv_out: Conv3,
    pub channels: usize,
}

type St = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn take(st: &St, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    st.get(name)
        .cloned()
        .ok_or_else(|| format!("upscaler: missing {name}"))
}

fn conv(st: &St, prefix: &str) -> Result<Conv3, String> {
    let (shape, w) = take(st, &format!("{prefix}.weight"))?;
    let (_, b) = take(st, &format!("{prefix}.bias"))?;
    if shape.len() != 5 || shape[2] != shape[3] || shape[3] != shape[4] {
        return Err(format!("{prefix}: expected a cubic 3-D kernel, got {shape:?}"));
    }
    Ok(Conv3 { w, b, c_out: shape[0], c_in: shape[1], k: shape[2] })
}

fn gnorm(st: &St, prefix: &str) -> Result<GroupNorm, String> {
    let (_, w) = take(st, &format!("{prefix}.weight"))?;
    let (_, b) = take(st, &format!("{prefix}.bias"))?;
    Ok(GroupNorm { w, b, groups: 32 })
}

impl LatentUpscaler {
    /// Read the published `.safetensors` — the net is a third-party model
    /// with its own licence, so it is loaded beside the container rather
    /// than packed into it, the way an adapter is.
    pub fn load(path: &Path) -> Result<LatentUpscaler, String> {
        let st = crate::ltxlora::read_safetensors(path)?;
        let conv_in = conv(&st, "conv_in")?;
        let channels = conv_in.c_out;
        let e0 = (take(&st, "embed.0.weight")?.1, take(&st, "embed.0.bias")?.1);
        let e2 = (take(&st, "embed.2.weight")?.1, take(&st, "embed.2.bias")?.1);

        let load_blocks = |side: &str| -> Result<Vec<Block>, String> {
            let mut out = Vec::new();
            for i in 0.. {
                let p = format!("{side}.{i}");
                if st.contains_key(&format!("{p}.dwconv.weight")) {
                    let (shape, w) = take(&st, &format!("{p}.dwconv.weight"))?;
                    let (_, b) = take(&st, &format!("{p}.dwconv.bias"))?;
                    out.push(Block::Temporal(TemporalConv {
                        norm: gnorm(&st, &format!("{p}.norm"))?,
                        dw: DepthwiseT { w, b, k: shape[2] },
                        pw: conv(&st, &format!("{p}.pwconv"))?,
                    }));
                } else if st.contains_key(&format!("{p}.in_layers.0.weight")) {
                    let (eshape, ew) = take(&st, &format!("{p}.emb_layers.1.weight"))?;
                    let (_, eb) = take(&st, &format!("{p}.emb_layers.1.bias"))?;
                    let _ = eshape;
                    out.push(Block::Res(ResBlock {
                        in_norm: gnorm(&st, &format!("{p}.in_layers.0"))?,
                        in_conv: conv(&st, &format!("{p}.in_layers.2"))?,
                        emb: (ew, eb),
                        out_norm: gnorm(&st, &format!("{p}.out_norm"))?,
                        out_conv: conv(&st, &format!("{p}.out_layers.2"))?,
                        skip: match st.contains_key(&format!("{p}.skip.weight")) {
                            true => Some(conv(&st, &format!("{p}.skip"))?),
                            false => None,
                        },
                    }));
                } else {
                    break;
                }
            }
            if out.is_empty() {
                return Err(format!("upscaler: no {side} found"));
            }
            Ok(out)
        };

        Ok(LatentUpscaler {
            conv_in,
            embed: (e0, e2),
            in_blocks: load_blocks("in_blocks")?,
            out_blocks: load_blocks("out_blocks")?,
            norm_out: gnorm(&st, "norm_out")?,
            conv_out: conv(&st, "conv_out")?,
            channels,
        })
    }

    /// `embed(scale − 1)`: Linear, SiLU, Linear.
    fn embedding(&self, scale: f32) -> Vec<f32> {
        let ((w0, b0), (w2, b2)) = &self.embed;
        let mut h: Vec<f32> = b0.iter().zip(w0).map(|(b, w)| b + w * (scale - 1.0)).collect();
        silu_in_place(&mut h);
        let e = b2.len();
        let mut out = vec![0f32; e];
        for (o, v) in out.iter_mut().enumerate() {
            let row = &w2[o * h.len()..(o + 1) * h.len()];
            *v = b2[o] + row.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
        }
        out
    }

    /// Trilinear resize, `align_corners=false` — torch's own convention, and
    /// the temporal axis is left alone by every caller here.
    fn resize(x: &Vol, t: usize, h: usize, w: usize) -> Vol {
        let mut out = Vol::zeros(x.c, t, h, w);
        let map = |o: usize, n_out: usize, n_in: usize| -> (usize, usize, f32) {
            if n_out == n_in {
                return (o, o, 0.0);
            }
            let s = ((o as f32 + 0.5) * n_in as f32 / n_out as f32 - 0.5).max(0.0);
            let i0 = s.floor() as usize;
            let i1 = (i0 + 1).min(n_in - 1);
            (i0, i1, s - i0 as f32)
        };
        for c in 0..x.c {
            for ti in 0..t {
                let (t0, t1, ft) = map(ti, t, x.t);
                for hi in 0..h {
                    let (h0, h1, fh) = map(hi, h, x.h);
                    for wi in 0..w {
                        let (w0, w1, fw) = map(wi, w, x.w);
                        let p = |tt: usize, hh: usize, ww: usize| x.at(c, tt, hh, ww);
                        let lerp = |a: f32, b: f32, f: f32| a + (b - a) * f;
                        let v00 = lerp(p(t0, h0, w0), p(t0, h0, w1), fw);
                        let v01 = lerp(p(t0, h1, w0), p(t0, h1, w1), fw);
                        let v10 = lerp(p(t1, h0, w0), p(t1, h0, w1), fw);
                        let v11 = lerp(p(t1, h1, w0), p(t1, h1, w1), fw);
                        let v0 = lerp(v00, v01, fh);
                        let v1 = lerp(v10, v11, fh);
                        out.data[((c * t + ti) * h + hi) * w + wi] = lerp(v0, v1, ft);
                    }
                }
            }
        }
        out
    }

    /// Upscale a *raw* latent (the sampler's own scale) to `(h_out, w_out)`.
    /// Normalization by the VAE's channel statistics happens here, because
    /// the network was trained on the normalized latent and every caller
    /// holds the raw one.
    pub fn upscale(&self, z: &Vol, h_out: usize, w_out: usize, pool: Option<&Pool>) -> Vol {
        assert_eq!(z.c, LATENT_MEAN.len(), "upscaler expects a 24-channel latent");
        let n = z.t * z.h * z.w;
        let mut x = z.clone();
        for c in 0..z.c {
            for i in 0..n {
                x.data[c * n + i] = (x.data[c * n + i] - LATENT_MEAN[c]) / LATENT_STD[c];
            }
        }
        // The node feeds the SPATIAL ratio; temporal length never changes.
        let scale = h_out as f32 / z.h as f32;
        let emb = self.embedding(scale);

        let mut v = self.conv_in.apply(&x, pool);
        for b in &self.in_blocks {
            v = match b {
                Block::Res(r) => r.apply(&v, &emb, pool),
                Block::Temporal(t) => t.apply(&v, pool),
            };
        }
        v = Self::resize(&v, v.t, h_out, w_out);
        for b in &self.out_blocks {
            v = match b {
                Block::Res(r) => r.apply(&v, &emb, pool),
                Block::Temporal(t) => t.apply(&v, pool),
            };
        }
        let mut v = self.norm_out.apply(&v);
        silu_in_place(&mut v.data);
        let mut out = self.conv_out.apply(&v, pool);
        let n = out.t * out.h * out.w;
        for c in 0..out.c {
            for i in 0..n {
                out.data[c * n + i] = out.data[c * n + i] * LATENT_STD[c] + LATENT_MEAN[c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `align_corners=false` maps output centres onto input centres — the
    /// one convention difference that would show as a half-pixel shift over
    /// a whole clip.
    #[test]
    fn resize_matches_torch_convention() {
        let x = Vol { c: 1, t: 1, h: 1, w: 2, data: vec![0.0, 1.0] };
        let up = LatentUpscaler::resize(&x, 1, 1, 4);
        // torch: F.interpolate([0,1], size=4, mode='linear', align_corners=False)
        // → [0, 0.25, 0.75, 1]
        let want = [0.0f32, 0.25, 0.75, 1.0];
        for (g, w) in up.data.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{:?} vs {want:?}", up.data);
        }
    }

    /// A resize to the same size is the identity, not a blur.
    #[test]
    fn resize_identity() {
        let x = Vol { c: 1, t: 2, h: 2, w: 2, data: (0..8).map(|i| i as f32).collect() };
        let up = LatentUpscaler::resize(&x, 2, 2, 2);
        assert_eq!(up.data, x.data);
    }
}
