//! MiniMax-H3's video VAE decoder: a ViT3D, not a conv stack.
//!
//! 36 transformer blocks over the latent grid, each latent cell one
//! token, and a single linear that expands every token into a
//! 4×16×16×3 block of pixels. The encoder half is a 3-D causal CNN and
//! is not packed — text-to-video never runs it.
//!
//! ## Tiling is not an optimization here
//!
//! The reference decodes in 256-pixel spatial tiles and 17-frame
//! temporal clips ALWAYS — `tiling=True` is the constructor default and
//! `decode_tiled` just forwards to `decode`. Because the decoder is
//! global attention, a tile sees a different context than the whole
//! frame would, so the tiling is part of the model's output, not a
//! memory strategy layered on top of it. Both schedules are reproduced
//! exactly: `split_tiles` down to the last overlap unit, and the
//! clip/token-drop bookkeeping that makes a chunk emit 17 frames and
//! carry 5 over.

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Pixel tile edge and the smallest overlap `split_tiles` will accept.
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
/// Temporal clip in frames, and the tokens dropped off the encoder's
/// tail that the decoder must therefore re-manufacture.
const CLIP_LENGTH: usize = 17;
const TOKEN_DROP: usize = 3;

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    scale1: Vec<f32>,
    scale2: Vec<f32>,
    qkv: Proj, // [3·dim, dim], PER HEAD interleaved
    qkv_b: Vec<f32>,
    out: Proj,
    out_b: Vec<f32>,
    w1: Proj, // [2·4·dim, dim]
    w1_b: Vec<f32>,
    w2: Proj, // [dim, 4·dim]
    w2_b: Vec<f32>,
}

pub struct VideoVae {
    post_quant: Proj,
    post_quant_b: Vec<f32>,
    x_embed: Proj,
    x_embed_b: Vec<f32>,
    registers: Vec<f32>, // [n_reg, dim]
    blocks: Vec<Block>,
    norm_out_w: Vec<f32>,
    norm_out_b: Vec<f32>,
    proj_out: Proj,
    proj_out_b: Vec<f32>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pool: Option<Arc<Pool>>,
    dim: usize,
    heads: usize,
    head_dim: usize,
    z_channels: usize,
    patch: usize,
    patch_t: usize,
    n_reg: usize,
    rope_theta: f32,
    rope_dim: usize,
    eps: f64,
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f64, dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for (((d, &v), &g), &bb) in dst.iter_mut().zip(x).zip(w).zip(b) {
        *d = ((v as f64 - mean) * inv) as f32 * g + bb;
    }
}

/// RMSNorm with a weight, no bias.
fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

/// RMSNorm with NO affine — the decoder's q/k norms are weightless.
fn rms_norm_plain(x: &mut [f32], eps: f64) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v = (*v as f64 * inv) as f32;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

fn softmax_inplace(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
    let mut den = 0f32;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r;
    }
    if den > 0.0 {
        let inv = 1.0 / den;
        for r in row.iter_mut() {
            *r *= inv;
        }
    }
}

/// `(starts, lens, overlaps)` for one axis — the reference's schedule,
/// which grows the overlaps rather than the tile count so every tile is
/// exactly `TILE` wide.
fn split_tiles(input_len: usize, ratio: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if TILE >= input_len {
        return (vec![0], vec![input_len], Vec::new());
    }
    let mut n = input_len.div_ceil(TILE);
    let (mut overlaps, mut remaining);
    loop {
        overlaps = vec![TILE_OVERLAP_MIN; n - 1];
        let total: usize = overlaps.iter().sum();
        if TILE * n < total + input_len {
            n += 1;
            continue;
        }
        remaining = TILE * n - total - input_len;
        break;
    }
    for i in 0..remaining / ratio {
        overlaps[i % (n - 1)] += ratio;
    }
    let mut starts = vec![0usize];
    for i in 0..n - 1 {
        starts.push(starts[i] + TILE - overlaps[i]);
    }
    (starts, vec![TILE; n], overlaps)
}

impl VideoVae {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model.tensor_bytes("vvae.config_json").map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("vvae.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);
        let n = u("num_layers", 36);
        let mut blocks = Vec::with_capacity(n);
        for l in 0..n {
            let p = format!("vvae.blocks.{l}");
            blocks.push(Block {
                norm1: f32v(&format!("{p}.norm1"))?,
                norm2: f32v(&format!("{p}.norm2"))?,
                scale1: f32v(&format!("{p}.scale1"))?,
                scale2: f32v(&format!("{p}.scale2"))?,
                qkv: Proj::from_model(model, &format!("{p}.attn.to_qkv.weight"))?,
                qkv_b: f32v(&format!("{p}.attn.to_qkv.bias"))?,
                out: Proj::from_model(model, &format!("{p}.attn.to_out.weight"))?,
                out_b: f32v(&format!("{p}.attn.to_out.bias"))?,
                w1: Proj::from_model(model, &format!("{p}.ff.w1.weight"))?,
                w1_b: f32v(&format!("{p}.ff.w1.bias"))?,
                w2: Proj::from_model(model, &format!("{p}.ff.w2.weight"))?,
                w2_b: f32v(&format!("{p}.ff.w2.bias"))?,
            });
        }
        let dim = u("dim", 2048);
        let heads = u("heads", 32);
        Ok(Self {
            post_quant: Proj::from_model(model, "vvae.post_quant_conv.weight")?,
            post_quant_b: f32v("vvae.post_quant_conv.bias")?,
            x_embed: Proj::from_model(model, "vvae.x_embedder.weight")?,
            x_embed_b: f32v("vvae.x_embedder.bias")?,
            registers: f32v("vvae.register_tokens")?,
            blocks,
            norm_out_w: f32v("vvae.norm_out.weight")?,
            norm_out_b: f32v("vvae.norm_out.bias")?,
            proj_out: Proj::from_model(model, "vvae.proj_out.weight")?,
            proj_out_b: f32v("vvae.proj_out.bias")?,
            latents_mean: f32v("vvae.latents_mean")?,
            latents_std: f32v("vvae.latents_std")?,
            pool: Pool::from_env(),
            dim,
            heads,
            head_dim: dim / heads,
            z_channels: u("z_channels", 24),
            patch: u("patch_size", 16),
            patch_t: u("patch_size_t", 4),
            n_reg: u("num_register_tokens", 4),
            rope_theta: cfg["rope_theta"].as_f64().unwrap_or(100.0) as f32,
            rope_dim: ((dim / heads) as f64 * cfg["rope_dim_ratio"].as_f64().unwrap_or(0.75))
                as usize,
            eps: cfg["eps"].as_f64().unwrap_or(1e-5),
        })
    }

    /// `arange(0.5, n)/n · 2 − 1` — the normalized cell centre of one
    /// axis, so a tile's coordinates depend on the TILE's extent and
    /// not on the frame's.
    fn axis(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
            .collect()
    }

    /// `[S, rope_dim/2]` angles: three axes × `rope_dim/6` frequencies,
    /// scaled by 2π. Suffix tokens sit at the origin and get zeros.
    fn rope_angles(&self, t: usize, h: usize, w: usize, suffix: usize) -> Vec<f32> {
        let k = self.rope_dim / 6; // frequencies per axis
        let inv: Vec<f32> = (0..k)
            .map(|i| 1.0 / self.rope_theta.powf(i as f32 * 2.0 * 3.0 / self.rope_dim as f32))
            .collect();
        let (ta, ha, wa) = (Self::axis(t), Self::axis(h), Self::axis(w));
        let tau = 2.0 * std::f32::consts::PI;
        let mut out = Vec::with_capacity((t * h * w + suffix) * 3 * k);
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    for &c in &[ta[ti], ha[hi], wa[wi]] {
                        for &f in &inv {
                            out.push(tau * c * f);
                        }
                    }
                }
            }
        }
        out.extend(std::iter::repeat_n(0.0, suffix * 3 * k));
        out
    }

    fn attention(&self, qkv: &[f32], n: usize, attn: &mut [f32], angles: &[f32]) {
        let (nh, hd, dim) = (self.heads, self.head_dim, self.dim);
        let pairs = angles.len() / n; // = rope_dim / 2
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = self.pool.as_deref();
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                // to_qkv is viewed [.., heads, 3·hd] then chunked, so a
                // head's q, k and v are adjacent — not three planes.
                let base = p * 3 * dim + h * 3 * hd;
                qh[p * hd..(p + 1) * hd].copy_from_slice(&qkv[base..base + hd]);
                kh[p * hd..(p + 1) * hd].copy_from_slice(&qkv[base + hd..base + 2 * hd]);
                for d in 0..hd {
                    vt[d * n + p] = qkv[base + 2 * hd + d];
                }
            }
            for (buf, _) in [(&mut qh, 0), (&mut kh, 1)] {
                for p in 0..n {
                    let x = &mut buf[p * hd..(p + 1) * hd];
                    rms_norm_plain(x, self.eps);
                    for j in 0..pairs {
                        let (s, c) = angles[p * pairs + j].sin_cos();
                        let (a, b) = (x[j], x[j + pairs]);
                        x[j] = a * c - b * s;
                        x[j + pairs] = a * s + b * c;
                    }
                }
            }
            for v in qh.iter_mut() {
                *v *= scale;
            }
            crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
            let sp = SendPtr(scores.as_mut_ptr());
            let soft = |lo: usize, hi: usize| {
                for r in lo..hi {
                    // SAFETY: workers own disjoint score rows.
                    softmax_inplace(unsafe { sp.row(r * n, n) });
                }
            };
            match pool {
                Some(pl) => pl.run_rows(n, &soft),
                None => soft(0, n),
            }
            crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            for p in 0..n {
                attn[p * dim + h * hd..p * dim + (h + 1) * hd]
                    .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// One tile-clip: latents `[z_channels, t, h, w]` → pixels
    /// `[3, t·4, h·16, w·16]`, both channel-major.
    fn decode_tile(&self, z: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let dim = self.dim;
        let np = t * h * w;
        let suffix = 1 + self.n_reg;
        let n = np + suffix;

        // [C, T, H, W] → [T·H·W, C], then the 1×1×1 post-quant conv and
        // the embedder, which are both plain linears at this point.
        let zc = self.z_channels;
        let mut rows = vec![0f32; np * zc];
        for c in 0..zc {
            for i in 0..np {
                rows[i * zc + c] = z[c * np + i];
            }
        }
        let mut pq = vec![0f32; np * zc];
        self.post_quant.matmat(&rows, np, &mut pq, pool);
        for r in pq.chunks_exact_mut(zc) {
            for (v, &b) in r.iter_mut().zip(&self.post_quant_b) {
                *v += b;
            }
        }
        let mut x = vec![0f32; n * dim];
        self.x_embed.matmat(&pq, np, &mut x[..np * dim], pool);
        for r in x[..np * dim].chunks_exact_mut(dim) {
            for (v, &b) in r.iter_mut().zip(&self.x_embed_b) {
                *v += b;
            }
        }
        // register tokens, then one all-zero token
        x[np * dim..(np + self.n_reg) * dim].copy_from_slice(&self.registers);

        let angles = self.rope_angles(t, h, w, suffix);
        let mut xn = vec![0f32; n * dim];
        let mut qkv = vec![0f32; n * 3 * dim];
        let mut attn = vec![0f32; n * dim];
        let mut proj = vec![0f32; n * dim];
        let inner = 4 * dim;
        for blk in &self.blocks {
            for (o, src) in xn.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
                rms_norm_into(src, &blk.norm1, self.eps, o);
            }
            blk.qkv.matmat(&xn, n, &mut qkv, pool);
            for r in qkv.chunks_exact_mut(3 * dim) {
                for (v, &b) in r.iter_mut().zip(&blk.qkv_b) {
                    *v += b;
                }
            }
            self.attention(&qkv, n, &mut attn, &angles);
            blk.out.matmat(&attn, n, &mut proj, pool);
            for (p, r) in proj.chunks_exact_mut(dim).enumerate() {
                for (i, v) in r.iter_mut().enumerate() {
                    *v += blk.out_b[i];
                    x[p * dim + i] += *v * blk.scale1[i];
                }
            }
            for (o, src) in xn.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
                rms_norm_into(src, &blk.norm2, self.eps, o);
            }
            let mut gu = vec![0f32; n * 2 * inner];
            blk.w1.matmat(&xn, n, &mut gu, pool);
            let mut act = vec![0f32; n * inner];
            for p in 0..n {
                let r = &gu[p * 2 * inner..(p + 1) * 2 * inner];
                for i in 0..inner {
                    act[p * inner + i] =
                        silu(r[i] + blk.w1_b[i]) * (r[inner + i] + blk.w1_b[inner + i]);
                }
            }
            blk.w2.matmat(&act, n, &mut proj, pool);
            for (p, r) in proj.chunks_exact_mut(dim).enumerate() {
                for (i, v) in r.iter_mut().enumerate() {
                    *v += blk.w2_b[i];
                    x[p * dim + i] += *v * blk.scale2[i];
                }
            }
        }

        let (pt, ps) = (self.patch_t, self.patch);
        let od = 3 * pt * ps * ps;
        let mut head = vec![0f32; np * dim];
        for (o, src) in head.chunks_exact_mut(dim).zip(x[..np * dim].chunks_exact(dim)) {
            layer_norm(src, &self.norm_out_w, &self.norm_out_b, self.eps, o);
        }
        let mut out = vec![0f32; np * od];
        self.proj_out.matmat(&head, np, &mut out, pool);
        for r in out.chunks_exact_mut(od) {
            for (v, &b) in r.iter_mut().zip(&self.proj_out_b) {
                *v += b;
            }
        }

        // [T,H,W, 3,pt,ps,ps] → [3, T·pt, H·ps, W·ps]
        let (ot, oh, ow) = (t * pt, h * ps, w * ps);
        let mut px = vec![0f32; 3 * ot * oh * ow];
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    let src = &out[((ti * h + hi) * w + wi) * od..((ti * h + hi) * w + wi + 1) * od];
                    for c in 0..3 {
                        for a in 0..pt {
                            for b in 0..ps {
                                for d in 0..ps {
                                    px[((c * ot + ti * pt + a) * oh + hi * ps + b) * ow
                                        + wi * ps + d] =
                                        src[((c * pt + a) * ps + b) * ps + d];
                                }
                            }
                        }
                    }
                }
            }
        }
        px
    }

    /// Spatially tiled decode of one temporal clip. `z` is
    /// `[z_channels, t, zh, zw]`; the result is `[3, t·4, zh·16, zw·16]`.
    fn decode_clip(&self, z: &[f32], t: usize, zh: usize, zw: usize) -> Vec<f32> {
        let r = self.patch;
        let (height, width) = (zh * r, zw * r);
        let (ys, yl, yo) = split_tiles(height, r);
        let (xs, xl, xo) = split_tiles(width, r);
        let frames = t * self.patch_t;
        let mut canvas = vec![0f32; 3 * frames * height * width];

        // The reference blends each tile into its predecessor's tail and
        // then trims the overlap off, so a pixel is written once.
        let mut row_tails: Vec<Vec<f32>> = Vec::new();
        let mut out_y = 0usize;
        for (i, (&ip, &il)) in ys.iter().zip(&yl).enumerate() {
            let (zi, zl) = (ip / r, il / r);
            let mut new_tails: Vec<Vec<f32>> = Vec::new();
            let mut left_tail: Option<Vec<f32>> = None;
            let mut out_x = 0usize;
            let mut row_h = 0usize;
            for (j, (&jp, &jl)) in xs.iter().zip(&xl).enumerate() {
                let (zj, zw_t) = (jp / r, jl / r);
                let mut sub = vec![0f32; self.z_channels * t * zl * zw_t];
                for c in 0..self.z_channels {
                    for ti in 0..t {
                        for hh in 0..zl {
                            for ww in 0..zw_t {
                                sub[((c * t + ti) * zl + hh) * zw_t + ww] =
                                    z[((c * t + ti) * zh + zi + hh) * zw + zj + ww];
                            }
                        }
                    }
                }
                let mut tile = self.decode_tile(&sub, t, zl, zw_t);
                let (mut th, mut tw) = (zl * r, zw_t * r);
                if i + 1 < ys.len() {
                    new_tails.push(crop(&tile, frames, th, tw, th - yo[i], th, 0, tw));
                }
                let next_left = if j + 1 < xs.len() {
                    Some(crop(&tile, frames, th, tw, 0, th, tw - xo[j], tw))
                } else {
                    None
                };
                if i > 0 {
                    tile = blend(&row_tails[j], &tile, frames, th, tw, yo[i - 1], 2);
                }
                if j > 0 {
                    let lt = left_tail.as_ref().unwrap();
                    tile = blend(lt, &tile, frames, th, tw, xo[j - 1], 3);
                }
                left_tail = next_left;
                if i + 1 < ys.len() {
                    tile = crop(&tile, frames, th, tw, 0, th - yo[i], 0, tw);
                    th -= yo[i];
                }
                if j + 1 < xs.len() {
                    tile = crop(&tile, frames, th, tw, 0, th, 0, tw - xo[j]);
                    tw -= xo[j];
                }
                for c in 0..3 {
                    for f in 0..frames {
                        for hh in 0..th {
                            let dst = ((c * frames + f) * height + out_y + hh) * width + out_x;
                            let src = ((c * frames + f) * th + hh) * tw;
                            canvas[dst..dst + tw].copy_from_slice(&tile[src..src + tw]);
                        }
                    }
                }
                out_x += tw;
                row_h = th;
            }
            row_tails = new_tails;
            out_y += row_h;
        }
        canvas
    }

    /// Normalized latents `[z_channels, t_lat, zh, zw]` → RGB in [0, 1],
    /// `[3, frames, zh·16, zw·16]`.
    pub fn decode(&self, z: &[f32], t_lat: usize, zh: usize, zw: usize) -> (Vec<f32>, usize) {
        let zc = self.z_channels;
        let np = t_lat * zh * zw;
        let mut zz = vec![0f32; zc * np];
        for c in 0..zc {
            let (m, s) = (self.latents_mean[c], self.latents_std[c]);
            for i in 0..np {
                zz[c * np + i] = z[c * np + i] * s + m;
            }
        }

        let ratio_t = self.patch_t;
        let chunk_tokens = CLIP_LENGTH.div_ceil(ratio_t); // 5
        let token_overlap = (chunk_tokens - TOKEN_DROP % chunk_tokens) % chunk_tokens; // 2
        let frame_pre_pad = (ratio_t - CLIP_LENGTH % ratio_t) % ratio_t; // 3
        let frame_overlap = (token_overlap * ratio_t).saturating_sub(frame_pre_pad); // 5
        let chunk_dec = chunk_tokens * ratio_t; // 20

        // The encoder dropped TOKEN_DROP tokens off the tail, so the
        // decoder plans against a longer pseudo-sequence and pads the
        // real one out with a repeat of its last token.
        let mut pseudo = t_lat + TOKEN_DROP;
        let mut pad = 0usize;
        if pseudo % chunk_tokens != 0 {
            pad = chunk_tokens - pseudo % chunk_tokens;
            pseudo += pad;
        }
        let mut chunks = pseudo / chunk_tokens - usize::from(TOKEN_DROP > 0);
        if chunks < 1 {
            pad += chunk_tokens;
            chunks += 1;
        }
        let t_pad = t_lat + pad;
        if pad > 0 {
            let mut grown = vec![0f32; zc * t_pad * zh * zw];
            for c in 0..zc {
                for ti in 0..t_pad {
                    let src = ti.min(t_lat - 1);
                    let a = (c * t_lat + src) * zh * zw;
                    let b = (c * t_pad + ti) * zh * zw;
                    grown[b..b + zh * zw].copy_from_slice(&zz[a..a + zh * zw]);
                }
            }
            zz = grown;
        }

        let (h, w) = (zh * self.patch, zw * self.patch);
        let mut out: Vec<f32> = Vec::new();
        let mut carry: Option<(Vec<f32>, usize)> = None;
        for i in 0..chunks {
            let a = i * chunk_tokens;
            let b = (a + chunk_tokens + token_overlap).min(t_pad);
            let n = b.saturating_sub(a.min(t_pad));
            if n == 0 {
                continue;
            }
            let mut sub = vec![0f32; zc * n * zh * zw];
            for c in 0..zc {
                let src = (c * t_pad + a) * zh * zw;
                let dst = c * n * zh * zw;
                sub[dst..dst + n * zh * zw].copy_from_slice(&zz[src..src + n * zh * zw]);
            }
            let dec = self.decode_clip(&sub, n, zh, zw);
            let dec_frames = n * ratio_t;
            for j in 0..2 {
                let fa = j * chunk_dec;
                let fb = (fa + chunk_dec).min(dec_frames);
                if fb <= fa + frame_pre_pad {
                    continue;
                }
                let mut part = frames_of(&dec, dec_frames, h, w, fa + frame_pre_pad, fb);
                let mut pn = fb - fa - frame_pre_pad;
                if j == 0 {
                    if let Some((tail, tn)) = carry.take() {
                        part = blend_frames(&tail, tn, &part, pn, h, w, frame_overlap);
                        pn = part.len() / (3 * h * w);
                    }
                    append_frames(&mut out, &part, pn, h, w);
                } else {
                    carry = Some((part, pn));
                }
            }
            if i + 1 == chunks {
                if let Some((tail, tn)) = carry.take() {
                    append_frames(&mut out, &tail, tn, h, w);
                }
            }
        }

        // Undo the ImageNet pixel normalization and clamp; the reference
        // then maps to [-1, 1] and every caller maps straight back, so
        // stop at [0, 1].
        let frames = out.len() / (3 * h * w);
        let want = t_lat * ratio_t - pad_frames(t_lat, pad, chunk_tokens, ratio_t);
        for c in 0..3 {
            let base = c * frames * h * w;
            for v in out[base..base + frames * h * w].iter_mut() {
                *v = (*v * IMAGENET_STD[c] + IMAGENET_MEAN[c]).clamp(0.0, 1.0);
            }
        }
        let keep = want.min(frames);
        if keep < frames {
            let mut trimmed = vec![0f32; 3 * keep * h * w];
            for c in 0..3 {
                let src = c * frames * h * w;
                let dst = c * keep * h * w;
                trimmed[dst..dst + keep * h * w].copy_from_slice(&out[src..src + keep * h * w]);
            }
            out = trimmed;
        }
        (out, keep)
    }

    pub fn spatial_ratio(&self) -> usize {
        self.patch
    }
    pub fn temporal_ratio(&self) -> usize {
        self.patch_t
    }
}

/// Frames the padding tokens manufactured, which the reference trims.
fn pad_frames(t_lat: usize, pad: usize, chunk_tokens: usize, ratio_t: usize) -> usize {
    if pad == 0 {
        return 0;
    }
    let intra = CLIP_LENGTH % ratio_t;
    if intra == 0 {
        return pad * ratio_t;
    }
    (0..pad)
        .map(|k| {
            if (t_lat + k) % chunk_tokens == 0 {
                intra
            } else {
                ratio_t
            }
        })
        .sum()
}

/// `[3, f, h, w]` sub-rectangle.
#[allow(clippy::too_many_arguments)]
fn crop(x: &[f32], f: usize, h: usize, w: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> Vec<f32> {
    let (nh, nw) = (y1 - y0, x1 - x0);
    let mut out = vec![0f32; 3 * f * nh * nw];
    for c in 0..3 {
        for fi in 0..f {
            for yy in 0..nh {
                let s = ((c * f + fi) * h + y0 + yy) * w + x0;
                let d = ((c * f + fi) * nh + yy) * nw;
                out[d..d + nw].copy_from_slice(&x[s..s + nw]);
            }
        }
    }
    out
}

/// Linear cross-fade of `a`'s tail into `b`'s head along `dim` (2 = y,
/// 3 = x); the result is `b` with its first `extent` rows/columns
/// replaced by the blend.
#[allow(clippy::too_many_arguments)]
fn blend(a: &[f32], b: &[f32], f: usize, h: usize, w: usize, extent: usize, dim: usize) -> Vec<f32> {
    let ah = a.len() / (3 * f * w);
    let aw = a.len() / (3 * f * h);
    let mut out = b.to_vec();
    let e = if dim == 2 {
        extent.min(ah).min(h)
    } else {
        extent.min(aw).min(w)
    };
    for c in 0..3 {
        for fi in 0..f {
            for k in 0..e {
                let wb = k as f32 / e as f32;
                let wa = 1.0 - wb;
                if dim == 2 {
                    let sa = ((c * f + fi) * ah + ah - e + k) * w;
                    let sb = ((c * f + fi) * h + k) * w;
                    for x in 0..w {
                        out[sb + x] = a[sa + x] * wa + b[sb + x] * wb;
                    }
                } else {
                    for y in 0..h {
                        let sa = ((c * f + fi) * h + y) * aw + aw - e + k;
                        let sb = ((c * f + fi) * h + y) * w + k;
                        out[sb] = a[sa] * wa + b[sb] * wb;
                    }
                }
            }
        }
    }
    out
}

/// Frames `[a, b)` of a `[3, f, h, w]` clip.
fn frames_of(x: &[f32], f: usize, h: usize, w: usize, a: usize, b: usize) -> Vec<f32> {
    let n = b - a;
    let mut out = vec![0f32; 3 * n * h * w];
    for c in 0..3 {
        let s = (c * f + a) * h * w;
        let d = c * n * h * w;
        out[d..d + n * h * w].copy_from_slice(&x[s..s + n * h * w]);
    }
    out
}

/// Cross-fade `tail` into the head of `part` along the frame axis.
#[allow(clippy::too_many_arguments)]
fn blend_frames(
    tail: &[f32],
    tn: usize,
    part: &[f32],
    pn: usize,
    h: usize,
    w: usize,
    extent: usize,
) -> Vec<f32> {
    let e = extent.min(tn).min(pn);
    let mut out = part.to_vec();
    for c in 0..3 {
        for k in 0..e {
            let wb = k as f32 / e as f32;
            let wa = 1.0 - wb;
            let s = ((c * tn) + tn - e + k) * h * w;
            let d = ((c * pn) + k) * h * w;
            for i in 0..h * w {
                out[d + i] = tail[s + i] * wa + part[d + i] * wb;
            }
        }
    }
    out
}

/// Append a `[3, n, h, w]` clip to a growing `[3, ?, h, w]` buffer.
fn append_frames(out: &mut Vec<f32>, part: &[f32], n: usize, h: usize, w: usize) {
    let old = out.len() / (3 * h * w);
    let total = old + n;
    let mut grown = vec![0f32; 3 * total * h * w];
    for c in 0..3 {
        if old > 0 {
            let s = c * old * h * w;
            let d = c * total * h * w;
            grown[d..d + old * h * w].copy_from_slice(&out[s..s + old * h * w]);
        }
        let s = c * n * h * w;
        let d = (c * total + old) * h * w;
        grown[d..d + n * h * w].copy_from_slice(&part[s..s + n * h * w]);
    }
    *out = grown;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_schedule_matches_the_reference() {
        // 288 tall: two 256 tiles, the overlap grown to swallow the slack.
        let (s, l, o) = split_tiles(288, 16);
        assert_eq!(s, vec![0, 32]);
        assert_eq!(l, vec![256, 256]);
        assert_eq!(o, vec![224]);
        // 512 wide does not fit in two tiles at the minimum overlap.
        let (s, l, o) = split_tiles(512, 16);
        assert_eq!(s, vec![0, 128, 256]);
        assert_eq!(l, vec![256; 3]);
        assert_eq!(o, vec![128, 128]);
        // Anything at or under one tile is one tile.
        assert_eq!(split_tiles(256, 16).0, vec![0]);
        assert_eq!(split_tiles(128, 16).1, vec![128]);
    }

    #[test]
    fn temporal_constants_come_out_as_the_reference_computes_them() {
        let ratio_t = 4usize;
        let chunk = CLIP_LENGTH.div_ceil(ratio_t);
        assert_eq!(chunk, 5);
        assert_eq!((chunk - TOKEN_DROP % chunk) % chunk, 2);
        assert_eq!((ratio_t - CLIP_LENGTH % ratio_t) % ratio_t, 3);
    }
}
