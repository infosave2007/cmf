//! Qwen-Image-2.1 VAE (diffusers `AutoencoderKLQwenImage21`): the
//! image specialisation of Wan-2.2's residual VAE — every "causal 3-D"
//! convolution is a 2-D one over the single frame — with 4 image
//! channels (RGBA), 64 latent channels and 16× spatial compression
//! (four 2× stages over `dim_mult` [1, 2, 4, 8, 8]).
//!
//! Single-frame semantics reproduced exactly:
//! - the temporal convolutions (`time_conv`) never run for one frame
//!   (the feature cache is empty on the first chunk), so they are not
//!   packed;
//! - `DupUp3D` on the first chunk keeps the LAST temporal slot of its
//!   channel shuffle, `AvgDown3D` zero-pads the time axis in front and
//!   averages that zero frame in;
//! - `RMS_norm` = L2-normalise over channels (ε 1e-12) · √C · γ;
//! - upsampling is nearest 2× then a 3×3 conv; downsampling pads
//!   (0, 1, 0, 1) and runs a stride-2 3×3 conv;
//! - the mid block is resnet → single-head attention → resnet;
//! - decode clamps to [−1, 1].
//!
//! Tensors live channel-first, `[C, H·W]`. The decoder runs resident on
//! the device when a backend implements `gpu::qi21_vae_decode` (wgpu:
//! `gpu_wgpu/qi21_vae.rs`; `CMF_QI21_VAE_CHAIN=0` turns it off). Otherwise
//! same-padding convolutions go through the device (`gpu::vae_conv2d`,
//! `gpu::vae_upsample_conv`) when one is up; everything else, and every
//! conv without a device, runs the host GEMM.

use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

type R<T> = Result<T, String>;

/// Below this many MACs a conv stays on the host (upload costs more).
const GPU_CONV_WORK: usize = 1 << 26;

#[derive(Clone, Debug)]
pub struct Qi21VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Qi21VaeConfig {
    pub fn from_json(v: &serde_json::Value) -> R<Self> {
        let u = |k: &str, d: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(d);
        let fl = |k: &str| -> Vec<f32> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect())
                .unwrap_or_default()
        };
        let base_dim = u("base_dim", 96);
        let c = Self {
            base_dim,
            decoder_base_dim: v["decoder_base_dim"].as_u64().map(|x| x as usize).unwrap_or(base_dim),
            z_dim: u("z_dim", 64),
            dim_mult: v["dim_mult"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_else(|| vec![1, 2, 4, 8, 8]),
            num_res_blocks: u("num_res_blocks", 2),
            temperal_downsample: v["temperal_downsample"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_bool()).collect())
                .unwrap_or_else(|| vec![false, true, true, true]),
            in_channels: u("in_channels", 4),
            out_channels: u("out_channels", 4),
            latents_mean: fl("latents_mean"),
            latents_std: fl("latents_std"),
        };
        if !v["is_residual"].as_bool().unwrap_or(true) {
            return Err("Qwen-Image-2.1 VAE: only the residual layout is supported".into());
        }
        if v["patch_size"].as_u64().is_some_and(|p| p > 1) {
            return Err("Qwen-Image-2.1 VAE: patchified variants are not supported".into());
        }
        if c.latents_mean.len() != c.z_dim || c.latents_std.len() != c.z_dim {
            return Err("Qwen-Image-2.1 VAE: latents_mean/std do not match z_dim".into());
        }
        if c.temperal_downsample.len() + 1 != c.dim_mult.len() {
            return Err("Qwen-Image-2.1 VAE: temperal_downsample must have len(dim_mult) - 1 entries".into());
        }
        Ok(c)
    }

    /// Spatial compression (2 per downsampling stage).
    pub fn scale(&self) -> usize {
        1 << (self.dim_mult.len() - 1)
    }
}

// ───────────────────────────── ops ─────────────────────────────

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

fn gpu_ok() -> bool {
    std::env::var("CMF_QI21_VAE_GPU").as_deref() != Ok("0") && crate::gpu::enabled_here()
}

struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    ci: usize,
    co: usize,
    k: usize,
}

impl Conv {
    fn load(m: &CmfModel, p: &str) -> R<Self> {
        let w = crate::dit::cmf_f32(m, &format!("{p}.weight"))?;
        let b = crate::dit::cmf_f32(m, &format!("{p}.bias"))?;
        let shape = m
            .tensor(&format!("{p}.weight"))
            .map(|t| t.shape.clone())
            .ok_or_else(|| format!("{p}.weight: missing"))?;
        let (co, ci, k) = match shape.as_slice() {
            [co, ci, k, k2] if k == k2 => (*co, *ci, *k),
            [co, ci, 1, k, k2] if k == k2 => (*co, *ci, *k),
            [co, ci] => (*co, *ci, 1),
            s => return Err(format!("{p}.weight: unexpected shape {s:?}")),
        };
        if w.len() != co * ci * k * k || b.len() != co {
            return Err(format!("{p}: weight/bias length mismatch"));
        }
        Ok(Self { w, b, ci, co, k })
    }

    /// Same-padding stride-1 conv (k = 1 or 3).
    fn same(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> Vec<f32> {
        let work = h * w * self.ci * self.co * self.k * self.k;
        if work >= GPU_CONV_WORK && gpu_ok() {
            let mut out = vec![0f32; self.co * h * w];
            if crate::gpu::vae_conv2d(&self.w, &self.b, x, self.ci, self.co, h, w, self.k, &mut out) {
                return out;
            }
        }
        let p = self.k / 2;
        conv2d_cpu(x, self.ci, self.co, h, w, self.k, 1, (p, p, p, p), &self.w, &self.b, pool).0
    }

    /// Nearest 2× upsample, then this (3×3 same) conv.
    fn up_same(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> Vec<f32> {
        let work = 4 * h * w * self.ci * self.co * self.k * self.k;
        if work >= GPU_CONV_WORK && gpu_ok() {
            let mut out = vec![0f32; self.co * 4 * h * w];
            if crate::gpu::vae_upsample_conv(&self.w, &self.b, x, self.ci, self.co, h, w, self.k, &mut out) {
                return out;
            }
        }
        let up = upsample2x(x, self.ci, h, w);
        self.same(&up, 2 * h, 2 * w, pool)
    }

    /// Pad (0, 1, 0, 1), stride-2 3×3 conv (the downsampler).
    fn down(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> (Vec<f32>, usize, usize) {
        let (o, oh, ow) =
            conv2d_cpu(x, self.ci, self.co, h, w, self.k, 2, (0, 1, 0, 1), &self.w, &self.b, pool);
        (o, oh, ow)
    }
}

/// Host conv through a bounded im2col tile and the GEMM. `pad` =
/// (top, bottom, left, right).
#[allow(clippy::too_many_arguments)]
fn conv2d_cpu(
    x: &[f32],
    ci: usize,
    co: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    pad: (usize, usize, usize, usize),
    weight: &[f32],
    bias: &[f32],
    pool: Option<&Pool>,
) -> (Vec<f32>, usize, usize) {
    let (pt, pb, pl, pr) = pad;
    let oh = (h + pt + pb - k) / s + 1;
    let ow = (w + pl + pr - k) / s + 1;
    let patch_k = ci * k * k;
    let npos = oh * ow;
    const TILE_BYTES: usize = 64 << 20;
    let tile = (TILE_BYTES / (4 * (patch_k + co))).clamp(1, npos.max(1));
    let mut patches = vec![0f32; tile * patch_k];
    let mut ybuf = vec![0f32; tile * co];
    let mut out = vec![0f32; co * npos];
    let mut p0 = 0usize;
    while p0 < npos {
        let n = tile.min(npos - p0);
        {
            let pp = SendPtr(patches.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for row in lo..hi {
                    let p = p0 + row;
                    let (yo, xo) = (p / ow, p % ow);
                    // SAFETY: disjoint rows.
                    let patch = unsafe { pp.slice(row * patch_k, patch_k) };
                    let mut i = 0usize;
                    for c in 0..ci {
                        let img = &x[c * h * w..(c + 1) * h * w];
                        for ky in 0..k {
                            let sy = (yo * s + ky) as isize - pt as isize;
                            for kx in 0..k {
                                let sx = (xo * s + kx) as isize - pl as isize;
                                patch[i] = if sy >= 0 && (sy as usize) < h && sx >= 0 && (sx as usize) < w {
                                    img[sy as usize * w + sx as usize]
                                } else {
                                    0.0
                                };
                                i += 1;
                            }
                        }
                    }
                }
            });
        }
        crate::zimage::host_gemm::gemm_nt(&patches[..n * patch_k], weight, &mut ybuf[..n * co], n, patch_k, co, pool);
        for row in 0..n {
            let p = p0 + row;
            for c in 0..co {
                out[c * npos + p] = ybuf[row * co + c] + bias[c];
            }
        }
        p0 += n;
    }
    (out, oh, ow)
}

fn upsample2x(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (2 * h, 2 * w);
    let mut out = vec![0f32; c * h2 * w2];
    for ch in 0..c {
        let src = &x[ch * h * w..(ch + 1) * h * w];
        let dst = &mut out[ch * h2 * w2..(ch + 1) * h2 * w2];
        for y in 0..h2 {
            for xx in 0..w2 {
                dst[y * w2 + xx] = src[(y / 2) * w + xx / 2];
            }
        }
    }
    out
}

/// RMS_norm over channels: x / max(‖x‖₂, 1e-12) · √C · γ, optionally
/// followed by SiLU. `x` is `[c, hw]`.
fn rms_channels(x: &[f32], c: usize, hw: usize, gamma: &[f32], silu: bool, pool: Option<&Pool>) -> Vec<f32> {
    let mut out = vec![0f32; c * hw];
    let scale = (c as f64).sqrt();
    let op = SendPtr(out.as_mut_ptr());
    pool_rows(pool, hw, &|lo, hi| {
        for p in lo..hi {
            let mut ss = 0f64;
            for ch in 0..c {
                let v = x[ch * hw + p] as f64;
                ss += v * v;
            }
            let inv = scale / ss.sqrt().max(1e-12);
            for ch in 0..c {
                let mut v = (x[ch * hw + p] as f64 * inv) as f32 * gamma[ch];
                if silu {
                    v /= 1.0 + (-v).exp();
                }
                // SAFETY: disjoint positions.
                unsafe { op.slice(ch * hw + p, 1)[0] = v };
            }
        }
    });
    out
}

struct Resnet {
    g1: Vec<f32>,
    c1: Conv,
    g2: Vec<f32>,
    c2: Conv,
    shortcut: Option<Conv>,
}

impl Resnet {
    fn load(m: &CmfModel, p: &str) -> R<Self> {
        let g = |n: &str| crate::dit::cmf_f32(m, &format!("{p}.{n}.gamma"));
        Ok(Self {
            g1: g("norm1")?,
            c1: Conv::load(m, &format!("{p}.conv1"))?,
            g2: g("norm2")?,
            c2: Conv::load(m, &format!("{p}.conv2"))?,
            shortcut: if m.tensor(&format!("{p}.conv_shortcut.weight")).is_some() {
                Some(Conv::load(m, &format!("{p}.conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> Vec<f32> {
        let hw = h * w;
        let ci = self.c1.ci;
        let a = rms_channels(x, ci, hw, &self.g1, true, pool);
        let a = self.c1.same(&a, h, w, pool);
        let a = rms_channels(&a, self.c1.co, hw, &self.g2, true, pool);
        let mut a = self.c2.same(&a, h, w, pool);
        match &self.shortcut {
            Some(s) => {
                let sc = s.same(x, h, w, pool);
                for (o, v) in a.iter_mut().zip(sc) {
                    *o += v;
                }
            }
            None => {
                for (o, &v) in a.iter_mut().zip(x) {
                    *o += v;
                }
            }
        }
        a
    }
}

struct Attn {
    gamma: Vec<f32>,
    qkv: Conv,
    proj: Conv,
}

impl Attn {
    fn load(m: &CmfModel, p: &str) -> R<Self> {
        Ok(Self {
            gamma: crate::dit::cmf_f32(m, &format!("{p}.norm.gamma"))?,
            qkv: Conv::load(m, &format!("{p}.to_qkv"))?,
            proj: Conv::load(m, &format!("{p}.proj"))?,
        })
    }

    /// Single-head attention over the h·w positions, with the residual.
    fn forward(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> Vec<f32> {
        use crate::zimage::host_gemm;
        let c = self.qkv.ci;
        let hw = h * w;
        let xn = rms_channels(x, c, hw, &self.gamma, false, pool);
        let qkv = self.qkv.same(&xn, h, w, pool); // [3c, hw]
        // token-major q (pre-scaled), k, and v^T = the channel planes
        let scale = 1.0 / (c as f32).sqrt();
        let mut q = vec![0f32; hw * c];
        let mut k = vec![0f32; hw * c];
        for ch in 0..c {
            for p in 0..hw {
                q[p * c + ch] = qkv[ch * hw + p] * scale;
                k[p * c + ch] = qkv[(c + ch) * hw + p];
            }
        }
        let vt = &qkv[2 * c * hw..3 * c * hw]; // [c, hw]
        let mut o = vec![0f32; hw * c];
        const QCH: usize = 1024;
        let mut scores = vec![0f32; QCH.min(hw) * hw];
        let mut q0 = 0;
        while q0 < hw {
            let nq = QCH.min(hw - q0);
            host_gemm::gemm_nt(&q[q0 * c..(q0 + nq) * c], &k, &mut scores[..nq * hw], nq, c, hw, pool);
            {
                let sp = SendPtr(scores.as_mut_ptr());
                pool_rows(pool, nq, &|lo, hi| {
                    for r in lo..hi {
                        // SAFETY: disjoint rows.
                        let row = unsafe { sp.slice(r * hw, hw) };
                        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
            }
            host_gemm::gemm_nt(&scores[..nq * hw], vt, &mut o[q0 * c..(q0 + nq) * c], nq, hw, c, pool);
            q0 += nq;
        }
        // back to channel planes
        let mut oc = vec![0f32; c * hw];
        for p in 0..hw {
            for ch in 0..c {
                oc[ch * hw + p] = o[p * c + ch];
            }
        }
        let mut out = self.proj.same(&oc, h, w, pool);
        for (d, &v) in out.iter_mut().zip(x) {
            *d += v;
        }
        out
    }
}

struct Mid {
    r0: Resnet,
    attn: Attn,
    r1: Resnet,
}

impl Mid {
    fn load(m: &CmfModel, p: &str) -> R<Self> {
        Ok(Self {
            r0: Resnet::load(m, &format!("{p}.resnets.0"))?,
            attn: Attn::load(m, &format!("{p}.attentions.0"))?,
            r1: Resnet::load(m, &format!("{p}.resnets.1"))?,
        })
    }
    fn forward(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> Vec<f32> {
        let x = self.r0.forward(x, h, w, pool);
        let x = self.attn.forward(&x, h, w, pool);
        self.r1.forward(&x, h, w, pool)
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    /// (upsample conv, DupUp temporal factor) when this block upsamples.
    up: Option<(Conv, usize)>,
    in_dim: usize,
    out_dim: usize,
}

/// DupUp3D on the first chunk: `out[o, 2y+sy, 2x+sx] =
/// x[(o·f·4 + (f−1)·4 + 2·sy + sx) / repeats, y, x]`, f = temporal factor.
fn dup_up(x: &[f32], cin: usize, cout: usize, ft: usize, h: usize, w: usize) -> Vec<f32> {
    let factor = ft * 4;
    let repeats = cout * factor / cin;
    let (h2, w2) = (2 * h, 2 * w);
    let mut out = vec![0f32; cout * h2 * w2];
    for o in 0..cout {
        for sy in 0..2 {
            for sx in 0..2 {
                let j = o * factor + (ft - 1) * 4 + 2 * sy + sx;
                let src = &x[(j / repeats) * h * w..(j / repeats + 1) * h * w];
                let dst = &mut out[o * h2 * w2..(o + 1) * h2 * w2];
                for y in 0..h {
                    for xx in 0..w {
                        dst[(2 * y + sy) * w2 + 2 * xx + sx] = src[y * w + xx];
                    }
                }
            }
        }
    }
    out
}

/// AvgDown3D on a single frame: the time axis is zero-padded in front
/// (to `ft` frames), every `ft·fs²` sub-sample becomes a channel and
/// groups of `cin·ft·fs²/cout` consecutive channels are averaged.
fn avg_down(x: &[f32], cin: usize, cout: usize, ft: usize, fs: usize, h: usize, w: usize) -> Vec<f32> {
    let factor = ft * fs * fs;
    let group = cin * factor / cout;
    let (oh, ow) = (h / fs, w / fs);
    let mut out = vec![0f32; cout * oh * ow];
    for o in 0..cout {
        let dst = &mut out[o * oh * ow..(o + 1) * oh * ow];
        for g in 0..group {
            let j = o * group + g;
            let c = j / factor;
            let r = j % factor;
            let t = r / (fs * fs);
            if t != ft - 1 {
                continue; // a zero-padded frame
            }
            let (sy, sx) = ((r % (fs * fs)) / fs, r % fs);
            let src = &x[c * h * w..(c + 1) * h * w];
            for y in 0..oh {
                for xx in 0..ow {
                    dst[y * ow + xx] += src[(y * fs + sy) * w + xx * fs + sx];
                }
            }
        }
        let inv = 1.0 / group as f32;
        for v in dst.iter_mut() {
            *v *= inv;
        }
    }
    out
}

struct Decoder {
    conv_in: Conv,
    mid: Mid,
    ups: Vec<UpBlock>,
    norm_out: Vec<f32>,
    conv_out: Conv,
}

struct DownBlock {
    resnets: Vec<Resnet>,
    /// The downsampler conv when this block halves the resolution.
    down: Option<Conv>,
    ft: usize,
    in_dim: usize,
    out_dim: usize,
}

struct Encoder {
    conv_in: Conv,
    downs: Vec<DownBlock>,
    mid: Mid,
    norm_out: Vec<f32>,
    conv_out: Conv,
    quant: Conv,
}

pub struct Qi21Vae {
    pub cfg: Qi21VaeConfig,
    dec: Option<Decoder>,
    enc: Option<Encoder>,
    post_quant: Option<Conv>,
    pool: Option<Arc<Pool>>,
    /// The container's identity (the device decoder caches its weights by it).
    uid: u64,
}

fn conv_ref(c: &Conv) -> crate::gpu::Qi21VaeConvRef<'_> {
    crate::gpu::Qi21VaeConvRef {
        w: &c.w,
        b: &c.b,
        ci: c.ci,
        co: c.co,
        k: c.k,
    }
}

fn res_ref(r: &Resnet) -> crate::gpu::Qi21VaeResRef<'_> {
    crate::gpu::Qi21VaeResRef {
        g1: &r.g1,
        c1: conv_ref(&r.c1),
        g2: &r.g2,
        c2: conv_ref(&r.c2),
        shortcut: r.shortcut.as_ref().map(conv_ref),
    }
}

impl Qi21Vae {
    /// Load from a container (`vae.*`); the encoder is optional (a
    /// text-to-image-only pack leaves it out).
    pub fn from_cmf(m: &Arc<CmfModel>, want_encoder: bool, want_decoder: bool) -> R<Self> {
        let raw = m
            .tensor_bytes("vae.config_json")
            .map_err(|e| format!("vae.config_json: {e}"))?;
        let v: serde_json::Value = serde_json::from_slice(raw).map_err(|e| format!("vae.config_json: {e}"))?;
        let cfg = Qi21VaeConfig::from_json(&v)?;
        let nb = cfg.num_res_blocks;
        let stages = cfg.dim_mult.len();
        let dec = if want_decoder {
            let d = cfg.decoder_base_dim;
            let mut dims = vec![d * cfg.dim_mult[stages - 1]];
            dims.extend(cfg.dim_mult.iter().rev().map(|&u| d * u));
            let tup: Vec<bool> = cfg.temperal_downsample.iter().rev().cloned().collect();
            let mut ups = Vec::with_capacity(stages);
            for i in 0..stages {
                let p = format!("vae.decoder.up_blocks.{i}");
                let resnets = (0..=nb)
                    .map(|r| Resnet::load(m, &format!("{p}.resnets.{r}")))
                    .collect::<R<Vec<_>>>()?;
                let up = if i != stages - 1 {
                    let ft = if tup[i] { 2 } else { 1 };
                    Some((Conv::load(m, &format!("{p}.upsampler.resample.1"))?, ft))
                } else {
                    None
                };
                ups.push(UpBlock {
                    resnets,
                    up,
                    in_dim: dims[i],
                    out_dim: dims[i + 1],
                });
            }
            Some(Decoder {
                conv_in: Conv::load(m, "vae.decoder.conv_in")?,
                mid: Mid::load(m, "vae.decoder.mid_block")?,
                ups,
                norm_out: crate::dit::cmf_f32(m, "vae.decoder.norm_out.gamma")?,
                conv_out: Conv::load(m, "vae.decoder.conv_out")?,
            })
        } else {
            None
        };
        let enc = if want_encoder {
            let d = cfg.base_dim;
            let mut dims = vec![d];
            dims.extend(cfg.dim_mult.iter().map(|&u| d * u));
            let mut downs = Vec::with_capacity(stages);
            for i in 0..stages {
                let p = format!("vae.encoder.down_blocks.{i}");
                let resnets = (0..nb)
                    .map(|r| Resnet::load(m, &format!("{p}.resnets.{r}")))
                    .collect::<R<Vec<_>>>()?;
                let last = i == stages - 1;
                let down = if !last {
                    Some(Conv::load(m, &format!("{p}.downsampler.resample.1"))?)
                } else {
                    None
                };
                let ft = if !last && cfg.temperal_downsample[i] { 2 } else { 1 };
                downs.push(DownBlock {
                    resnets,
                    down,
                    ft,
                    in_dim: dims[i],
                    out_dim: dims[i + 1],
                });
            }
            Some(Encoder {
                conv_in: Conv::load(m, "vae.encoder.conv_in")?,
                downs,
                mid: Mid::load(m, "vae.encoder.mid_block")?,
                norm_out: crate::dit::cmf_f32(m, "vae.encoder.norm_out.gamma")?,
                conv_out: Conv::load(m, "vae.encoder.conv_out")?,
                quant: Conv::load(m, "vae.quant_conv")?,
            })
        } else {
            None
        };
        let post_quant = if want_decoder {
            Some(Conv::load(m, "vae.post_quant_conv")?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            dec,
            enc,
            post_quant,
            pool: Pool::from_env(),
            uid: m.uid(),
        })
    }

    pub fn has_encoder(&self) -> bool {
        self.enc.is_some()
    }

    /// Normalised latent tokens `[h·w, z]` (the transformer's layout) →
    /// raw channel planes `[z, h·w]` (× std + mean).
    pub fn denormalize_tokens(&self, tok: &[f32], hw: usize) -> Vec<f32> {
        let z = self.cfg.z_dim;
        let mut out = vec![0f32; z * hw];
        for p in 0..hw {
            for c in 0..z {
                out[c * hw + p] = tok[p * z + c] * self.cfg.latents_std[c] + self.cfg.latents_mean[c];
            }
        }
        out
    }

    /// Raw planes `[z, h·w]` → normalised tokens `[h·w, z]`.
    pub fn normalize_to_tokens(&self, planes: &[f32], hw: usize) -> Vec<f32> {
        let z = self.cfg.z_dim;
        let mut out = vec![0f32; hw * z];
        for p in 0..hw {
            for c in 0..z {
                out[p * z + c] = (planes[c * hw + p] - self.cfg.latents_mean[c]) / self.cfg.latents_std[c];
            }
        }
        out
    }

    /// Raw latent planes `[z, h, w]` → image `[out_channels, 16h, 16w]` in [−1, 1].
    pub fn decode(&self, z: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let dec = self.dec.as_ref().ok_or("this container carries no VAE decoder")?;
        let pool = self.pool.as_deref();
        if z.len() != self.cfg.z_dim * h * w {
            return Err("VAE decode: latent size mismatch".into());
        }
        if let Some(mut img) = self.decode_device(dec, z, h, w) {
            for v in img.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
            return Ok(img);
        }
        let x = self.post_quant.as_ref().unwrap().same(z, h, w, pool);
        let mut x = dec.conv_in.same(&x, h, w, pool);
        x = dec.mid.forward(&x, h, w, pool);
        let (mut ch, mut cw) = (h, w);
        for ub in &dec.ups {
            let copy = x.clone();
            for r in &ub.resnets {
                x = r.forward(&x, ch, cw, pool);
            }
            if let Some((conv, ft)) = &ub.up {
                x = conv.up_same(&x, ch, cw, pool);
                let sc = dup_up(&copy, ub.in_dim, ub.out_dim, *ft, ch, cw);
                ch *= 2;
                cw *= 2;
                for (o, v) in x.iter_mut().zip(sc) {
                    *o += v;
                }
            }
        }
        let c_last = dec.conv_out.ci;
        let a = rms_channels(&x, c_last, ch * cw, &dec.norm_out, true, pool);
        let mut img = dec.conv_out.same(&a, ch, cw, pool);
        for v in img.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(img)
    }

    /// The whole decoder resident on the device (`gpu::qi21_vae_decode`):
    /// `None` = the per-conv path runs. `CMF_QI21_VAE_CHAIN=0` turns it off.
    fn decode_device(&self, dec: &Decoder, z: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        if std::env::var("CMF_QI21_VAE_CHAIN").as_deref() == Ok("0") || !gpu_ok() {
            return None;
        }
        let args = crate::gpu::Qi21VaeDecodeArgs {
            key: self.uid,
            post_quant: conv_ref(self.post_quant.as_ref()?),
            conv_in: conv_ref(&dec.conv_in),
            mid_res: [res_ref(&dec.mid.r0), res_ref(&dec.mid.r1)],
            attn_gamma: &dec.mid.attn.gamma,
            attn_qkv: conv_ref(&dec.mid.attn.qkv),
            attn_proj: conv_ref(&dec.mid.attn.proj),
            ups: dec
                .ups
                .iter()
                .map(|u| crate::gpu::Qi21VaeUpRef {
                    resnets: u.resnets.iter().map(res_ref).collect(),
                    up: u.up.as_ref().map(|(c, ft)| (conv_ref(c), *ft)),
                    in_dim: u.in_dim,
                    out_dim: u.out_dim,
                })
                .collect(),
            norm_out: &dec.norm_out,
            conv_out: conv_ref(&dec.conv_out),
        };
        let s = self.cfg.scale();
        let mut out = vec![0f32; dec.conv_out.co * h * s * w * s];
        crate::gpu::qi21_vae_decode(&args, z, h, w, &mut out).then_some(out)
    }

    /// Image `[in_channels, H, W]` in [−1, 1] (H, W multiples of the
    /// scale) → raw latent mean planes `[z, H/s, W/s]`.
    pub fn encode_mean(&self, img: &[f32], hh: usize, ww: usize) -> R<Vec<f32>> {
        let enc = self.enc.as_ref().ok_or("this container carries no VAE encoder")?;
        let pool = self.pool.as_deref();
        let s = self.cfg.scale();
        if hh % s != 0 || ww % s != 0 || img.len() != self.cfg.in_channels * hh * ww {
            return Err(format!("VAE encode: the image must be {}×(H, W) with H, W multiples of {s}", self.cfg.in_channels));
        }
        let mut x = enc.conv_in.same(img, hh, ww, pool);
        let (mut h, mut w) = (hh, ww);
        for db in &enc.downs {
            let copy = x.clone();
            for r in &db.resnets {
                x = r.forward(&x, h, w, pool);
            }
            if let Some(conv) = &db.down {
                let (y, oh, ow) = conv.down(&x, h, w, pool);
                let sc = avg_down(&copy, db.in_dim, db.out_dim, db.ft, 2, h, w);
                x = y;
                h = oh;
                w = ow;
                for (o, v) in x.iter_mut().zip(sc) {
                    *o += v;
                }
            } else {
                let sc = avg_down(&copy, db.in_dim, db.out_dim, db.ft, 1, h, w);
                for (o, v) in x.iter_mut().zip(sc) {
                    *o += v;
                }
            }
        }
        x = enc.mid.forward(&x, h, w, pool);
        let a = rms_channels(&x, enc.conv_out.ci, h * w, &enc.norm_out, true, pool);
        let y = enc.conv_out.same(&a, h, w, pool);
        let q = enc.quant.same(&y, h, w, pool);
        Ok(q[..self.cfg.z_dim * h * w].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dup_up_keeps_the_last_temporal_slot() {
        // cin 2 → cout 2, ft 2: factor 8, repeats 8 → every output reads
        // input channel (o·8 + 4 + …)/8 = o
        let x = vec![1.0, 2.0];
        let y = dup_up(&x, 2, 2, 2, 1, 1);
        assert_eq!(y, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        // ft 1, cin 4 → cout 2: factor 4, repeats 2 → o=0 reads 0,0,1,1
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = dup_up(&x, 4, 2, 1, 1, 1);
        assert_eq!(y, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn avg_down_averages_the_zero_frame_in() {
        // one channel 2×2, cin 1 → cout 1, ft 2, fs 2: factor 8, group 8;
        // the real frame's four pixels sum to 10 → mean 10/8
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = avg_down(&x, 1, 1, 2, 2, 2, 2);
        assert!((y[0] - 1.25).abs() < 1e-7);
        // ft 1 → a plain 2×2 mean
        let y = avg_down(&x, 1, 1, 1, 2, 2, 2);
        assert!((y[0] - 2.5).abs() < 1e-7);
    }

    #[test]
    fn stride_two_conv_pads_right_and_bottom() {
        // identity-ish 3×3 kernel picking the centre, 4×4 input → 2×2
        let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut wk = vec![0f32; 9];
        wk[4] = 1.0;
        let (y, oh, ow) = conv2d_cpu(&x, 1, 1, 4, 4, 3, 2, (0, 1, 0, 1), &wk, &[0.0], None);
        assert_eq!((oh, ow), (2, 2));
        // centre of the window at (1,1),(1,3),(3,1),(3,3)
        assert_eq!(y, vec![5.0, 7.0, 13.0, 15.0]);
    }
}
