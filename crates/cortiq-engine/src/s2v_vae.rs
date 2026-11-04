//! HunyuanVideo VAE (`AutoencoderKLHunyuanVideo`): 3-D causal convolutions,
//! 8× space and 4× time, 16 latent channels.
//!
//! What the reference does and this reproduces:
//! - every conv replicates one frame's worth in space and pads two frames in
//!   front (causal), never behind;
//! - `GroupNorm` statistics run over a whole tile, every frame of it;
//! - upsampling is nearest, with the first frame kept to one in time;
//! - the mid attention is single-head over every (t, h, w) token, and a token
//!   only sees tokens of its own frame or an earlier one;
//! - decode ALWAYS tiles in time: 5 latent frames every 3, blending 4 pixel
//!   frames across each seam. That is the default of the diffusers class and
//!   not a memory strategy to drop: the output depends on it.
//!
//! The latents here are `[C, F, H, W]`, in [`crate::ltxvae::Vol`].

use crate::dit::cmf_f32;
use crate::ltxvae::Vol;
use crate::pool::Pool;
use crate::s2v::{Rows, pool_rows, silu};
use crate::zimage::host_gemm;
use cortiq_core::CmfModel;
use std::sync::Arc;

pub const SPATIAL: usize = 8;
pub const TEMPORAL: usize = 4;
pub const SCALING_FACTOR: f32 = 0.476986;
const CHUNK: usize = 4096;
const GN_EPS: f64 = 1e-6;

pub(crate) struct Conv {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
    pub(crate) cin: usize,
    pub(crate) cout: usize,
    pub(crate) k: [usize; 3],
    pub(crate) stride: [usize; 3],
}

impl Conv {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str, stride: [usize; 3]) -> Result<Conv, String> {
        let name = format!("{p}.weight");
        let e = m.tensor(&name).ok_or_else(|| format!("missing tensor {name}"))?;
        if e.shape.len() != 5 {
            return Err(format!("{name}: not a 3-D conv"));
        }
        let (cout, cin, k) = (e.shape[0], e.shape[1], [e.shape[2], e.shape[3], e.shape[4]]);
        Ok(Conv { w: cmf_f32(m, &name)?, b: cmf_f32(m, &format!("{p}.bias"))?, cin, cout, k, stride })
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        assert_eq!(x.c, self.cin);
        let [kt, kh, kw] = self.k;
        let [st, sh, sw] = self.stride;
        // Causal replicate padding; `k/2` in space, `k - 1` in front in time.
        let (pt, ph, pw) = (kt - 1, kh / 2, kw / 2);
        let out_f = (x.f + pt - (kt - 1) - 1) / st + 1;
        let out_h = (x.h + 2 * ph - (kh - 1) - 1) / sh + 1;
        let out_w = (x.w + 2 * pw - (kw - 1) - 1) / sw + 1;
        let mut out = Vol::zeros(self.cout, out_f, out_h, out_w);
        let npos = out_f * out_h * out_w;
        let taps = kt * kh * kw;
        let k = self.cin * taps;
        let mut pat = vec![0f32; CHUNK.min(npos) * k];
        let mut ybuf = vec![0f32; CHUNK.min(npos) * self.cout];
        let mut p0 = 0;
        while p0 < npos {
            let n = CHUNK.min(npos - p0);
            let pr = Rows(pat.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for i in lo..hi {
                    let p = p0 + i;
                    let (oc, rest) = (p % out_w, p / out_w);
                    let (or, of) = (rest % out_h, rest / out_h);
                    // SAFETY: disjoint rows.
                    let row = unsafe { pr.at(i * k, k) };
                    for c in 0..self.cin {
                        let mut o = c * taps;
                        for dt in 0..kt {
                            let f = (of * st + dt).saturating_sub(pt).min(x.f - 1);
                            for dh in 0..kh {
                                let h = (or * sh + dh).saturating_sub(ph).min(x.h - 1);
                                for dw in 0..kw {
                                    let w = (oc * sw + dw).saturating_sub(pw).min(x.w - 1);
                                    row[o] = x.at(c, f, h, w);
                                    o += 1;
                                }
                            }
                        }
                    }
                }
            });
            host_gemm::gemm_nt(&pat[..n * k], &self.w, &mut ybuf[..n * self.cout], n, k, self.cout, pool);
            for co in 0..self.cout {
                let bias = self.b[co];
                let dst = &mut out.data[co * npos + p0..co * npos + p0 + n];
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = ybuf[i * self.cout + co] + bias;
                }
            }
            p0 += n;
        }
        out
    }
}

pub(crate) struct GroupNorm {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl GroupNorm {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self { w: cmf_f32(m, &format!("{p}.weight"))?, b: cmf_f32(m, &format!("{p}.bias"))? })
    }
    /// 32 groups over `[C, F, H, W]`, in place.
    fn apply(&self, x: &mut Vol, pool: Option<&Pool>) {
        let npos = x.positions();
        let per = x.c / 32;
        let sr = Rows(x.data.as_mut_ptr());
        pool_rows(pool, 32, &|lo, hi| {
            for g in lo..hi {
                // SAFETY: groups own disjoint channel ranges.
                let d = unsafe { sr.at(g * per * npos, per * npos) };
                let cnt = d.len() as f64;
                let mean = d.iter().map(|&v| v as f64).sum::<f64>() / cnt;
                let var = d.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / cnt;
                let inv = 1.0 / (var + GN_EPS).sqrt();
                for c in 0..per {
                    let (w, b) = (self.w[g * per + c], self.b[g * per + c]);
                    for v in &mut d[c * npos..(c + 1) * npos] {
                        *v = ((*v as f64 - mean) * inv) as f32 * w + b;
                    }
                }
            }
        });
    }
}

fn silu_all(x: &mut Vol, pool: Option<&Pool>) {
    let n = x.data.len();
    let sr = Rows(x.data.as_mut_ptr());
    pool_rows(pool, n / 4096 + 1, &|lo, hi| {
        for i in lo..hi {
            let (a, b) = (i * 4096, ((i + 1) * 4096).min(n));
            if a >= b {
                continue;
            }
            // SAFETY: disjoint spans.
            for v in unsafe { sr.at(a, b - a) } {
                *v = silu(*v);
            }
        }
    });
}

pub(crate) struct Res {
    pub(crate) n1: GroupNorm,
    pub(crate) c1: Conv,
    pub(crate) n2: GroupNorm,
    pub(crate) c2: Conv,
    pub(crate) short: Option<Conv>,
}

impl Res {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str) -> Result<Res, String> {
        let s = format!("{p}.conv_shortcut.conv.weight");
        Ok(Res {
            n1: GroupNorm::load(m, &format!("{p}.norm1"))?,
            c1: Conv::load(m, &format!("{p}.conv1.conv"), [1, 1, 1])?,
            n2: GroupNorm::load(m, &format!("{p}.norm2"))?,
            c2: Conv::load(m, &format!("{p}.conv2.conv"), [1, 1, 1])?,
            short: if m.tensor(&s).is_some() {
                Some(Conv::load(m, &format!("{p}.conv_shortcut.conv"), [1, 1, 1])?)
            } else {
                None
            },
        })
    }
    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let mut h = x.clone();
        self.n1.apply(&mut h, pool);
        silu_all(&mut h, pool);
        let mut h = self.c1.forward(&h, pool);
        self.n2.apply(&mut h, pool);
        silu_all(&mut h, pool);
        let mut h = self.c2.forward(&h, pool);
        match &self.short {
            Some(s) => {
                let r = s.forward(x, pool);
                for (a, &b) in h.data.iter_mut().zip(&r.data) {
                    *a += b;
                }
            }
            None => {
                for (a, &b) in h.data.iter_mut().zip(&x.data) {
                    *a += b;
                }
            }
        }
        h
    }
}

pub(crate) struct Mid {
    pub(crate) r0: Res,
    pub(crate) gn: GroupNorm,
    pub(crate) q: (Vec<f32>, Vec<f32>),
    pub(crate) k: (Vec<f32>, Vec<f32>),
    pub(crate) v: (Vec<f32>, Vec<f32>),
    pub(crate) o: (Vec<f32>, Vec<f32>),
    pub(crate) r1: Res,
}

impl Mid {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str) -> Result<Mid, String> {
        let a = format!("{p}.attentions.0");
        let lin = |n: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((cmf_f32(m, &format!("{a}.{n}.weight"))?, cmf_f32(m, &format!("{a}.{n}.bias"))?))
        };
        Ok(Mid {
            r0: Res::load(m, &format!("{p}.resnets.0"))?,
            gn: GroupNorm::load(m, &format!("{a}.group_norm"))?,
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            o: lin("to_out.0")?,
            r1: Res::load(m, &format!("{p}.resnets.1"))?,
        })
    }

    fn attention(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let c = x.c;
        let n = x.positions();
        let per_frame = x.h * x.w;
        let mut y = x.clone();
        self.gn.apply(&mut y, pool);
        // [C, n] → [n, C]
        let mut rows = vec![0f32; n * c];
        for ch in 0..c {
            for p in 0..n {
                rows[p * c + ch] = y.data[ch * n + p];
            }
        }
        let lin = |l: &(Vec<f32>, Vec<f32>), s: &[f32], m: usize| -> Vec<f32> {
            let mut out = vec![0f32; m * c];
            host_gemm::gemm_nt(s, &l.0, &mut out, m, c, c, pool);
            for r in out.chunks_exact_mut(c) {
                for (v, &b) in r.iter_mut().zip(&l.1) {
                    *v += b;
                }
            }
            out
        };
        let q = lin(&self.q, &rows, n);
        let k = lin(&self.k, &rows, n);
        let v = lin(&self.v, &rows, n);
        drop(rows);
        let scale = 1.0 / (c as f32).sqrt();
        // A query frame sees frames ≤ its own, so one frame of queries runs
        // over a growing key prefix and the N² matrix never exists.
        let mut att = vec![0f32; n * c];
        for f in 0..x.f {
            let live = (f + 1) * per_frame;
            let mut s = vec![0f32; per_frame * live];
            let qh: Vec<f32> = q[f * per_frame * c..(f + 1) * per_frame * c].iter().map(|&a| a * scale).collect();
            host_gemm::gemm_nt(&qh, &k[..live * c], &mut s, per_frame, c, live, pool);
            let sr = Rows(s.as_mut_ptr());
            pool_rows(pool, per_frame, &|lo, hi| {
                for r in lo..hi {
                    // SAFETY: disjoint rows.
                    let row = unsafe { sr.at(r * live, live) };
                    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f64;
                    for v in row.iter_mut() {
                        *v = (*v - mx).exp();
                        den += *v as f64;
                    }
                    let inv = (1.0 / den) as f32;
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }
            });
            let mut vt = vec![0f32; c * live];
            for d in 0..c {
                for r in 0..live {
                    vt[d * live + r] = v[r * c + d];
                }
            }
            host_gemm::gemm_nt(&s, &vt, &mut att[f * per_frame * c..(f + 1) * per_frame * c], per_frame, live, c, pool);
        }
        let o = lin(&self.o, &att, n);
        // Residual connection, back to [C, n].
        let mut out = x.clone();
        for ch in 0..c {
            for p in 0..n {
                out.data[ch * n + p] += o[p * c + ch];
            }
        }
        out
    }

    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let h = self.r0.forward(x, pool);
        let h = self.attention(&h, pool);
        self.r1.forward(&h, pool)
    }
}

pub(crate) struct Up {
    pub(crate) res: Vec<Res>,
    pub(crate) conv: Option<(Conv, [usize; 3])>,
}

/// Nearest upsample by `(ft, fs, fs)`; the first frame stays one frame.
fn nearest_up(x: &Vol, ft: usize, fs: usize) -> Vol {
    let f_out = if ft == 2 { 1 + (x.f - 1) * 2 } else { x.f };
    let (h_out, w_out) = (x.h * fs, x.w * fs);
    let mut out = Vol::zeros(x.c, f_out, h_out, w_out);
    let (ni, no) = (x.positions(), f_out * h_out * w_out);
    let src_f = |of: usize| if ft == 2 && of > 0 { 1 + (of - 1) / 2 } else { of };
    for c in 0..x.c {
        for of in 0..f_out {
            let sf = src_f(of);
            for oh in 0..h_out {
                let sh = oh / fs;
                for ow in 0..w_out {
                    let sw = ow / fs;
                    out.data[c * no + ((of * h_out + oh) * w_out + ow)] =
                        x.data[c * ni + ((sf * x.h + sh) * x.w + sw)];
                }
            }
        }
    }
    out
}

pub struct S2vVae {
    pub(crate) post_quant: Conv,
    pub(crate) conv_in: Conv,
    pub(crate) mid: Mid,
    pub(crate) ups: Vec<Up>,
    pub(crate) norm_out: GroupNorm,
    pub(crate) conv_out: Conv,
    pub(crate) pool: Option<Arc<Pool>>,
}

pub struct S2vVaeEncoder {
    pub(crate) quant: Conv,
    pub(crate) conv_in: Conv,
    pub(crate) downs: Vec<(Vec<Res>, Option<Conv>)>,
    pub(crate) mid: Mid,
    pub(crate) norm_out: GroupNorm,
    pub(crate) conv_out: Conv,
    pub(crate) pool: Option<Arc<Pool>>,
}

impl S2vVae {
    pub fn from_cmf(m: &Arc<CmfModel>) -> Result<S2vVae, String> {
        let mut ups = Vec::new();
        for (i, factor) in [[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]].into_iter().enumerate() {
            let p = format!("vae.decoder.up_blocks.{i}");
            let mut res = Vec::new();
            for r in 0..3 {
                res.push(Res::load(m, &format!("{p}.resnets.{r}"))?);
            }
            let conv = if i < 3 {
                Some((Conv::load(m, &format!("{p}.upsamplers.0.conv.conv"), [1, 1, 1])?, factor))
            } else {
                None
            };
            ups.push(Up { res, conv });
        }
        Ok(S2vVae {
            post_quant: Conv::load(m, "vae.post_quant_conv", [1, 1, 1])?,
            conv_in: Conv::load(m, "vae.decoder.conv_in.conv", [1, 1, 1])?,
            mid: Mid::load(m, "vae.decoder.mid_block")?,
            ups,
            norm_out: GroupNorm::load(m, "vae.decoder.conv_norm_out")?,
            conv_out: Conv::load(m, "vae.decoder.conv_out.conv", [1, 1, 1])?,
            pool: Pool::from_env(),
        })
    }

    /// The post-quant conv alone, on the host (16 channels into a 1×1 conv).
    pub(crate) fn post_quant_host(&self, z: &Vol) -> Vol {
        self.post_quant.forward(z, self.pool.as_deref())
    }

    /// One temporal tile through the decoder (post-quant conv included).
    fn decode_tile(&self, z: &Vol) -> Vol {
        let pool = self.pool.as_deref();
        let h = self.post_quant.forward(z, pool);
        let h = self.conv_in.forward(&h, pool);
        let mut h = self.mid.forward(&h, pool);
        for up in &self.ups {
            for r in &up.res {
                h = r.forward(&h, pool);
            }
            if let Some((conv, f)) = &up.conv {
                let u = nearest_up(&h, f[0], f[1]);
                h = conv.forward(&u, pool);
            }
        }
        self.norm_out.apply(&mut h, pool);
        silu_all(&mut h, pool);
        self.conv_out.forward(&h, pool)
    }

    /// `_temporal_tiled_decode` of the diffusers class, on latents already
    /// divided by the scaling factor. `[16, F, H, W]` → `[3, 4(F-1)+1, 8H, 8W]`.
    pub fn decode_with(&self, z: &Vol, tile: &mut dyn FnMut(&Vol) -> Vol) -> Vol {
        tiled(z, tile)
    }

    pub fn decode(&self, z: &Vol) -> Vol {
        tiled(z, &mut |t| self.decode_tile(t))
    }
}

/// The schedule shared by every backend: 5 latent frames every 3.
pub(crate) fn tiled(z: &Vol, tile: &mut dyn FnMut(&Vol) -> Vol) -> Vol {
    const MIN_LAT: usize = 4;
    const STRIDE_LAT: usize = 3;
    const STRIDE_PX: usize = 12;
    const BLEND: usize = 4;
    let frames_out = (z.f - 1) * TEMPORAL + 1;
    let mut row: Vec<Vol> = Vec::new();
    let mut i = 0;
    while i < z.f {
        let take = (MIN_LAT + 1).min(z.f - i);
        let mut t = Vol::zeros(z.c, take, z.h, z.w);
        let per = z.h * z.w;
        for c in 0..z.c {
            let src = &z.data[c * z.positions() + i * per..c * z.positions() + (i + take) * per];
            t.data[c * take * per..(c + 1) * take * per].copy_from_slice(src);
        }
        let mut d = tile(&t);
        if i > 0 {
            d = drop_first_frame(&d);
        }
        row.push(d);
        i += STRIDE_LAT;
    }
    let first = &row[0];
    let (h, w) = (first.h, first.w);
    let per = h * w;
    let mut out = Vol::zeros(first.c, frames_out, h, w);
    let mut at = 0usize;
    for ti in 0..row.len() {
        let keep = if ti == 0 { STRIDE_PX + 1 } else { STRIDE_PX };
        let tl = &row[ti];
        let take = keep.min(tl.f).min(frames_out - at);
        for fr in 0..take {
            for c in 0..tl.c {
                let cb = c * tl.f * per + fr * per;
                let dst = &mut out.data[c * frames_out * per + (at + fr) * per..][..per];
                if ti > 0 && fr < BLEND.min(row[ti - 1].f).min(tl.f) {
                    // `blend_t` over the previous tile's already-blended frames
                    // is what the reference writes back into `row`, but it reads
                    // the previous tile from its tail only, which was never
                    // blended unless the tile is very short.
                    let prev = &row[ti - 1];
                    let ext = BLEND.min(prev.f).min(tl.f);
                    let pa = c * prev.f * per + (prev.f - ext + fr) * per;
                    let wgt = fr as f32 / ext as f32;
                    for j in 0..per {
                        dst[j] = prev.data[pa + j] * (1.0 - wgt) + tl.data[cb + j] * wgt;
                    }
                } else {
                    dst.copy_from_slice(&tl.data[cb..cb + per]);
                }
            }
        }
        at += take;
    }
    out
}

fn drop_first_frame(v: &Vol) -> Vol {
    let per = v.h * v.w;
    let mut o = Vol::zeros(v.c, v.f - 1, v.h, v.w);
    for c in 0..v.c {
        let s = c * v.f * per + per;
        o.data[c * (v.f - 1) * per..(c + 1) * (v.f - 1) * per].copy_from_slice(&v.data[s..s + (v.f - 1) * per]);
    }
    o
}

impl S2vVaeEncoder {
    pub fn from_cmf(m: &Arc<CmfModel>) -> Result<Self, String> {
        let mut downs = Vec::new();
        for (i, st) in [[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]].into_iter().enumerate() {
            let p = format!("vae.encoder.down_blocks.{i}");
            let mut res = Vec::new();
            for r in 0..2 {
                res.push(Res::load(m, &format!("{p}.resnets.{r}"))?);
            }
            let d = if i < 3 {
                Some(Conv::load(m, &format!("{p}.downsamplers.0.conv.conv"), st)?)
            } else {
                None
            };
            downs.push((res, d));
        }
        Ok(Self {
            quant: Conv::load(m, "vae.quant_conv", [1, 1, 1])?,
            conv_in: Conv::load(m, "vae.encoder.conv_in.conv", [1, 1, 1])?,
            downs,
            mid: Mid::load(m, "vae.encoder.mid_block")?,
            norm_out: GroupNorm::load(m, "vae.encoder.conv_norm_out")?,
            conv_out: Conv::load(m, "vae.encoder.conv_out.conv", [1, 1, 1])?,
            pool: Pool::from_env(),
        })
    }

    /// `[3, F, H, W]` in [-1, 1] → (mean, logvar), each `[16, F', H/8, W/8]`.
    pub fn encode(&self, img: &Vol) -> (Vol, Vol) {
        let pool = self.pool.as_deref();
        let mut h = self.conv_in.forward(img, pool);
        for (res, d) in &self.downs {
            for r in res {
                h = r.forward(&h, pool);
            }
            if let Some(d) = d {
                h = d.forward(&h, pool);
            }
        }
        let mut h = self.mid.forward(&h, pool);
        self.norm_out.apply(&mut h, pool);
        silu_all(&mut h, pool);
        let h = self.conv_out.forward(&h, pool);
        self.finish_host(&h)
    }

    /// The quant conv and the posterior's split, after `conv_out`.
    pub(crate) fn finish_host(&self, h: &Vol) -> (Vol, Vol) {
        let h = self.quant.forward(h, self.pool.as_deref());
        let n = h.positions();
        let mut mean = Vol::zeros(16, h.f, h.h, h.w);
        let mut lv = Vol::zeros(16, h.f, h.h, h.w);
        mean.data.copy_from_slice(&h.data[..16 * n]);
        lv.data.copy_from_slice(&h.data[16 * n..32 * n]);
        for v in &mut lv.data {
            *v = v.clamp(-30.0, 20.0);
        }
        (mean, lv)
    }
}
