//! LTX-2.5 convolutional video VAE decoder — the `vvae.*` half of an
//! `ltx-2.5-av` container, in Rust with no ML framework underneath.
//!
//! Latents are `[128, F, H, W]`; the decoder returns `[3, 8·(F−1)+1, 32·H,
//! 32·W]` in `[-1, 1]`. The stack mirrors the encoder, and the checkpoint
//! stores its blocks in the encoder's order, so `decoder_blocks` from the
//! config is walked **backwards**:
//!
//! ```text
//! un_normalize → conv_in(128→1024)
//!   res×2 @1024 │ up(2,2,2) 1024→512 │ res×2 @512 │ up(2,2,2) 512→512
//!   res×4 @512  │ up(2,1,1) 512→256  │ res×6 @256 │ up(1,2,2) 256→128
//!   res×4 @128
//! → PixelNorm → SiLU → conv_out(128→48) → unpatchify(4) → [3, …]
//! ```
//!
//! Every convolution is 3×3×3 with zero spatial padding and **replicated
//! frames** at both ends of the time axis (`causal_decoder: false`), and
//! every normalization is `PixelNorm` — RMS over channels at each
//! location, no learned parameters, which is why the checkpoint carries no
//! norm weights. A residual path that changes width normalizes with a
//! 1-group GroupNorm and a 1×1×1 projection; at equal width both are
//! identity, and this decoder is equal-width everywhere.
//!
//! Convolutions run as im2col + GEMM over the worker pool, in position
//! chunks so the patch buffer stays bounded regardless of resolution.

use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Positions per im2col chunk (patch buffer ≈ CHUNK · Cin · 27 floats).
const CHUNK: usize = 8192;

/// A 3×3×3 convolution with zero spatial padding and replicate padding in
/// time. Weights are `[Cout, Cin, 3, 3, 3]` flattened to `[Cout, Cin·27]`.
pub struct Conv3d {
    w: Vec<f32>,
    b: Vec<f32>,
    c_out: usize,
    c_in: usize,
}

/// One `[C, F, H, W]` volume.
#[derive(Clone)]
pub struct Vol {
    pub c: usize,
    pub f: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vol {
    pub fn zeros(c: usize, f: usize, h: usize, w: usize) -> Vol {
        Vol { c, f, h, w, data: vec![0.0; c * f * h * w] }
    }
    #[inline]
    pub fn at(&self, c: usize, f: usize, h: usize, w: usize) -> f32 {
        self.data[((c * self.f + f) * self.h + h) * self.w + w]
    }
    pub fn positions(&self) -> usize {
        self.f * self.h * self.w
    }
}

/// Dequantize one tensor of the container to f32 (the VAE ships f16, so
/// this is a widening read, not a decode).
fn tensor_f32(model: &Arc<CmfModel>, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = model
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let mut out = vec![0.0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, model.entry_bytes(e), &mut out)?;
    Ok((out, e.shape.clone()))
}

impl Conv3d {
    fn load(model: &Arc<CmfModel>, prefix: &str, _pool: Option<&Pool>) -> Result<Conv3d, String> {
        let (w, shape) = tensor_f32(model, &format!("{prefix}.conv.weight"))?;
        let (c_out, c_in) = (shape[0], shape[1]);
        let b = match model.tensor(&format!("{prefix}.conv.bias")) {
            Some(_) => tensor_f32(model, &format!("{prefix}.conv.bias"))?.0,
            None => vec![0.0; c_out],
        };
        Ok(Conv3d { w, b, c_out, c_in })
    }

    /// `[Cin, F, H, W]` → `[Cout, F, H, W]`.
    pub fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        assert_eq!(x.c, self.c_in, "conv3d channels");
        let (f, h, w) = (x.f, x.h, x.w);
        let mut out = Vol::zeros(self.c_out, f, h, w);
        let k = self.c_in * 27;
        let npos = f * h * w;
        let mut patches = vec![0.0f32; CHUNK.min(npos) * k];
        let mut ybuf = vec![0.0f32; CHUNK.min(npos) * self.c_out];
        let mut p0 = 0usize;
        while p0 < npos {
            let n = CHUNK.min(npos - p0);
            // im2col: one row per output position, Cin·27 columns. The
            // spatial border taps are LEFT OUT below (zero padding), so the
            // buffer must start clean — a stale row from the previous chunk
            // would silently become the padding value.
            patches[..n * k].fill(0.0);
            for i in 0..n {
                let p = p0 + i;
                let (pw, rest) = (p % w, p / w);
                let (ph, pf) = (rest % h, rest / h);
                let row = &mut patches[i * k..(i + 1) * k];
                for ci in 0..self.c_in {
                    for kf in 0..3usize {
                        // replicate padding in time (non-causal decoder)
                        let sf = (pf + kf).saturating_sub(1).min(f - 1);
                        for kh in 0..3usize {
                            let sh = ph + kh;
                            if sh == 0 || sh > h {
                                continue; // zero padding
                            }
                            let sh = sh - 1;
                            for kw in 0..3usize {
                                let sw = pw + kw;
                                if sw == 0 || sw > w {
                                    continue;
                                }
                                let sw = sw - 1;
                                row[(ci * 3 + kf) * 9 + kh * 3 + kw] = x.at(ci, sf, sh, sw);
                            }
                        }
                    }
                }
            }
            // y[n, Cout] = patches[n, k] · wᵀ
            crate::fcd_ops::gemm_nt(&patches[..n * k], &self.w, &mut ybuf[..n * self.c_out], n, k, self.c_out, pool);
            for i in 0..n {
                let p = p0 + i;
                for co in 0..self.c_out {
                    out.data[co * npos + p] = ybuf[i * self.c_out + co] + self.b[co];
                }
            }
            p0 += n;
        }
        out
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMS over the channel axis at every (f, h, w) — no learned parameters.
fn pixel_norm(x: &mut Vol, eps: f32) {
    let npos = x.positions();
    for p in 0..npos {
        let mut s = 0.0f32;
        for c in 0..x.c {
            let v = x.data[c * npos + p];
            s += v * v;
        }
        let inv = 1.0 / (s / x.c as f32 + eps).sqrt();
        for c in 0..x.c {
            x.data[c * npos + p] *= inv;
        }
    }
}

fn silu_inplace(x: &mut Vol) {
    for v in x.data.iter_mut() {
        *v = silu(*v);
    }
}

/// PixelNorm → SiLU → conv1 → PixelNorm → SiLU → conv2, plus the residual.
pub struct ResBlock {
    conv1: Conv3d,
    conv2: Conv3d,
}

impl ResBlock {
    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let mut h = x.clone();
        pixel_norm(&mut h, 1e-8);
        silu_inplace(&mut h);
        let mut h = self.conv1.forward(&h, pool);
        pixel_norm(&mut h, 1e-8);
        silu_inplace(&mut h);
        let h = self.conv2.forward(&h, pool);
        let mut out = x.clone();
        for (o, v) in out.data.iter_mut().zip(&h.data) {
            *o += v;
        }
        out
    }
}

/// conv → depth-to-space over `(p1, p2, p3)` → drop the first frame when
/// the time stride is 2. Channels are laid out `(c p1 p2 p3)`.
pub struct DepthToSpaceUp {
    conv: Conv3d,
    stride: (usize, usize, usize),
}

impl DepthToSpaceUp {
    fn forward(&self, x: &Vol, pool: Option<&Pool>) -> Vol {
        let y = self.conv.forward(x, pool);
        let (p1, p2, p3) = self.stride;
        let cout = y.c / (p1 * p2 * p3);
        let (f2, h2, w2) = (y.f * p1, y.h * p2, y.w * p3);
        let mut out = Vol::zeros(cout, f2, h2, w2);
        for c in 0..cout {
            for a in 0..p1 {
                for b in 0..p2 {
                    for d in 0..p3 {
                        let src_c = ((c * p1 + a) * p2 + b) * p3 + d;
                        for f in 0..y.f {
                            for hh in 0..y.h {
                                for ww in 0..y.w {
                                    let v = y.at(src_c, f, hh, ww);
                                    let (of, oh, ow) = (f * p1 + a, hh * p2 + b, ww * p3 + d);
                                    out.data[((c * f2 + of) * h2 + oh) * w2 + ow] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
        if p1 == 2 {
            // the temporal upsample emits one frame too many at the head
            let f3 = f2 - 1;
            let mut trimmed = Vol::zeros(cout, f3, h2, w2);
            for c in 0..cout {
                for f in 0..f3 {
                    let src = ((c * f2 + f + 1) * h2) * w2;
                    let dst = ((c * f3 + f) * h2) * w2;
                    trimmed.data[dst..dst + h2 * w2].copy_from_slice(&out.data[src..src + h2 * w2]);
                }
            }
            return trimmed;
        }
        out
    }
}

enum Block {
    Res(Vec<ResBlock>),
    Up(DepthToSpaceUp),
}

/// The whole decoder, read from a packed `ltx-2.5-av` container.
pub struct ConvVaeDecoder {
    conv_in: Conv3d,
    blocks: Vec<Block>,
    conv_out: Conv3d,
    mean: Vec<f32>,
    std: Vec<f32>,
    patch: usize,
}

impl ConvVaeDecoder {
    /// Build from `vvae.*` tensors. The block schedule comes from
    /// `ltx.config_json`'s `vae.decoder_blocks`, walked in reverse.
    pub fn from_cmf(model: &Arc<CmfModel>, pool: Option<&Pool>) -> Result<ConvVaeDecoder, String> {
        // the pipeline config (from the DiT pass) or the VAE file's own
        // Take whichever config actually describes the VAE: a container
        // packed from the transformer first carries `ltx.config_json`, whose
        // `transformer` section says nothing about decoder blocks.
        let cfg: serde_json::Value = ["vvae.config_json", "ltx.config_json"]
            .iter()
            .filter_map(|n| model.tensor(n).map(|e| model.entry_bytes(e)))
            .filter_map(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .find(|c| c.get("vae").and_then(|v| v.get("decoder_blocks")).is_some())
            .ok_or("no config in this container carries vae.decoder_blocks")?;
        let vae = &cfg["vae"];
        let patch = vae["patch_size"].as_u64().unwrap_or(4) as usize;
        let blocks_cfg = vae["decoder_blocks"]
            .as_array()
            .ok_or("vae.decoder_blocks missing")?;
        let conv_in = Conv3d::load(model, "vvae.decoder.conv_in", pool)?;
        let conv_out = Conv3d::load(model, "vvae.decoder.conv_out", pool)?;
        let mut blocks = Vec::new();
        // the checkpoint numbers up_blocks in decode order; the config lists
        // them in the encoder's order
        for (i, entry) in blocks_cfg.iter().rev().enumerate() {
            let name = entry[0].as_str().unwrap_or("");
            let params = &entry[1];
            let prefix = format!("vvae.decoder.up_blocks.{i}");
            match name {
                "res_x" => {
                    let n = params["num_layers"].as_u64().unwrap_or(1) as usize;
                    let mut res = Vec::new();
                    for j in 0..n {
                        res.push(ResBlock {
                            conv1: Conv3d::load(model, &format!("{prefix}.res_blocks.{j}.conv1"), pool)?,
                            conv2: Conv3d::load(model, &format!("{prefix}.res_blocks.{j}.conv2"), pool)?,
                        });
                    }
                    blocks.push(Block::Res(res));
                }
                "compress_time" | "compress_space" | "compress_all" => {
                    let stride = match name {
                        "compress_time" => (2, 1, 1),
                        "compress_space" => (1, 2, 2),
                        _ => (2, 2, 2),
                    };
                    blocks.push(Block::Up(DepthToSpaceUp {
                        conv: Conv3d::load(model, &format!("{prefix}.conv"), pool)?,
                        stride,
                    }));
                }
                other => return Err(format!("unknown decoder block '{other}'")),
            }
        }
        let mean = tensor_f32(model, "vvae.per_channel_statistics.mean-of-means")?.0;
        let std = tensor_f32(model, "vvae.per_channel_statistics.std-of-means")?.0;
        Ok(ConvVaeDecoder {
            conv_in,
            blocks,
            conv_out,
            mean,
            std,
            patch,
        })
    }

    /// Latent `[128, F, H, W]` → frames `[3, 8·(F−1)+1, 32·H, 32·W]`.
    pub fn decode(&self, latent: &Vol, pool: Option<&Pool>) -> Vol {
        self.decode_traced(latent, pool, &mut |_, _| {})
    }

    /// `decode` with a callback after every stage — the parity gate walks
    /// the same names the reference's forward hooks produce.
    pub fn decode_traced(
        &self,
        latent: &Vol,
        pool: Option<&Pool>,
        trace: &mut dyn FnMut(&str, &Vol),
    ) -> Vol {
        let mut x = latent.clone();
        // un_normalize: x·std + mean, per latent channel
        let npos = x.positions();
        for c in 0..x.c {
            let (s, m) = (self.std[c], self.mean[c]);
            for p in 0..npos {
                x.data[c * npos + p] = x.data[c * npos + p] * s + m;
            }
        }
        let mut h = self.conv_in.forward(&x, pool);
        trace("after_conv_in", &h);
        for (i, b) in self.blocks.iter().enumerate() {
            h = match b {
                Block::Res(res) => {
                    let mut cur = h;
                    for r in res {
                        cur = r.forward(&cur, pool);
                    }
                    cur
                }
                Block::Up(u) => u.forward(&h, pool),
            };
            trace(&format!("after_block_{i}"), &h);
        }
        pixel_norm(&mut h, 1e-8);
        silu_inplace(&mut h);
        let y = self.conv_out.forward(&h, pool);
        trace("after_conv_out", &y);
        let out = unpatchify(&y, self.patch);
        trace("frames", &out);
        out
    }

}

/// `[C·q·r, F, H, W]` → `[C, F, H·q, W·r]` (channels laid out `(c r q)`).
pub fn unpatchify(x: &Vol, patch: usize) -> Vol {
    if patch == 1 {
        return x.clone();
    }
    let c = x.c / (patch * patch);
    let (h2, w2) = (x.h * patch, x.w * patch);
    let mut out = Vol::zeros(c, x.f, h2, w2);
    for cc in 0..c {
        for r in 0..patch {
            for q in 0..patch {
                let src_c = (cc * patch + r) * patch + q;
                for f in 0..x.f {
                    for hh in 0..x.h {
                        for ww in 0..x.w {
                            let v = x.at(src_c, f, hh, ww);
                            out.data[((cc * x.f + f) * h2 + hh * patch + q) * w2 + ww * patch + r] = v;
                        }
                    }
                }
            }
        }
    }
    out
}
