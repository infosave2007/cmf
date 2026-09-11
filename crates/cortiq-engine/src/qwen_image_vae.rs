//! Native Qwen-Image VAE (Diffusers `AutoencoderKLQwenImage`).
//!
//! The Qwen image VAE is a small causal video VAE used with one image frame
//! at a time by the image pipeline.  It is deliberately kept separate from
//! the older FLUX VAE in [`crate::vae`]: the block layout, RMS normalisation,
//! causal padding and first-frame cache path are different.  This module
//! keeps the CMF mapping borrowed and widens only the tensors needed by one
//! convolution or normalisation at a time.
//!
//! The public methods expose the raw VAE space.  The image pipeline owns the
//! per-channel `latents_mean`/`latents_std` normalisation and 2x2 latent
//! packing used by the denoiser.

use cortiq_core::{CmfModel, TensorDtype};
use std::path::Path;
use std::sync::Arc;

/// Scratch budget for one lowered convolution.  The activation itself is
/// owned by the caller; this cap covers the patch and GEMM output buffers.
const MAX_WORK_BYTES: usize = 64 * 1024 * 1024;
const EPS_NORMALIZE: f32 = 1.0e-12;

type VaeResult<T> = Result<T, String>;

/// A lazily decoded tensor.  CMF F16/BF16 and quantized payloads remain in
/// the mmap; `values` widens one tensor only for the duration of its op.
#[derive(Clone)]
struct TensorRef {
    model: Arc<CmfModel>,
    idx: usize,
    name: String,
}

impl TensorRef {
    fn load(model: &Arc<CmfModel>, name: impl Into<String>, shape: &[usize]) -> VaeResult<Self> {
        let name = name.into();
        let idx = model
            .tensor_index(&name)
            .ok_or_else(|| format!("qwen image VAE: missing tensor '{name}'"))?;
        let entry = &model.tensors[idx];
        if entry.shape != shape {
            return Err(format!(
                "qwen image VAE: tensor '{}' has shape {:?}, expected {:?}",
                name, entry.shape, shape
            ));
        }
        let expected = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if expected != Some(entry.n_elems()) {
            return Err(format!(
                "qwen image VAE: tensor '{}' element count does not match shape {:?}",
                name, shape
            ));
        }
        match entry.dtype {
            TensorDtype::F32
            | TensorDtype::F16
            | TensorDtype::Bf16
            | TensorDtype::Q8Row
            | TensorDtype::Q8_2f
            | TensorDtype::Q4Block
            | TensorDtype::Q4Tiled
            | TensorDtype::Q4TiledP
            | TensorDtype::Q2TiledP
            | TensorDtype::Vbit
            | TensorDtype::VbitRo
            | TensorDtype::Q1
            | TensorDtype::Q1S
            | TensorDtype::Q1T => {}
            other => {
                return Err(format!(
                    "qwen image VAE: tensor '{}' has unsupported dtype {}",
                    name,
                    other.name()
                ));
            }
        }
        Ok(Self {
            model: model.clone(),
            idx,
            name,
        })
    }

    fn values(&self) -> VaeResult<Vec<f32>> {
        let entry = &self.model.tensors[self.idx];
        let mut out = vec![0.0f32; entry.n_elems()];
        cortiq_core::quant::dequant_tensor(entry, self.model.entry_bytes(entry), &mut out)
            .map_err(|e| format!("qwen image VAE tensor '{}': {e}", self.name))?;
        Ok(out)
    }
}

/// Channel-major `[C, T, H, W]` activation.  The public image path always
/// uses `T == 1`, but retaining the temporal dimension makes causal padding
/// explicit and prevents accidental 2-D reinterpretation of a 3-D weight.
#[derive(Clone)]
struct Volume {
    c: usize,
    t: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Volume {
    fn from_frame(data: &[f32], c: usize, h: usize, w: usize) -> VaeResult<Self> {
        let n = c
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .ok_or_else(|| "qwen image VAE: activation shape overflow".to_string())?;
        if data.len() != n {
            return Err(format!(
                "qwen image VAE: expected frame with {n} values, got {}",
                data.len()
            ));
        }
        Ok(Self {
            c,
            t: 1,
            h,
            w,
            data: data.to_vec(),
        })
    }

    fn zeros(c: usize, t: usize, h: usize, w: usize) -> VaeResult<Self> {
        let n = c
            .checked_mul(t)
            .and_then(|n| n.checked_mul(h))
            .and_then(|n| n.checked_mul(w))
            .ok_or_else(|| "qwen image VAE: activation shape overflow".to_string())?;
        Ok(Self {
            c,
            t,
            h,
            w,
            data: vec![0.0; n],
        })
    }

    #[inline]
    fn offset(&self, c: usize, t: usize, y: usize, x: usize) -> usize {
        ((c * self.t + t) * self.h + y) * self.w + x
    }

    #[inline]
    fn get(&self, c: usize, t: usize, y: usize, x: usize) -> f32 {
        self.data[self.offset(c, t, y, x)]
    }

    #[inline]
    fn set(&mut self, c: usize, t: usize, y: usize, x: usize, value: f32) {
        let i = self.offset(c, t, y, x);
        self.data[i] = value;
    }

    fn frame(&self, t: usize) -> VaeResult<Vec<f32>> {
        if t >= self.t {
            return Err(format!(
                "qwen image VAE: frame index {t} outside temporal length {}",
                self.t
            ));
        }
        let mut out = vec![0.0; self.c * self.h * self.w];
        for c in 0..self.c {
            let src = self.offset(c, t, 0, 0);
            let dst = c * self.h * self.w;
            for y in 0..self.h {
                let s = src + y * self.w;
                let d = dst + y * self.w;
                out[d..d + self.w].copy_from_slice(&self.data[s..s + self.w]);
            }
        }
        Ok(out)
    }

    fn from_frames(frames: &[Vec<f32>], c: usize, h: usize, w: usize) -> VaeResult<Self> {
        let t = frames.len();
        let mut out = Self::zeros(c, t, h, w)?;
        for (ti, frame) in frames.iter().enumerate() {
            if frame.len() != c * h * w {
                return Err("qwen image VAE: frame size mismatch".into());
            }
            for ch in 0..c {
                let src = ch * h * w;
                let dst = out.offset(ch, ti, 0, 0);
                out.data[dst..dst + h * w].copy_from_slice(&frame[src..src + h * w]);
            }
        }
        Ok(out)
    }
}

/// Causal 3-D convolution.  Diffusers constructs the layer with ordinary
/// `(time,height,width)` padding and then changes it to left-only temporal
/// padding of `2 * padding[0]`, with symmetric spatial padding.  This is the
/// exact convention used here.
struct Conv3dRef {
    weight: TensorRef,
    bias: TensorRef,
    ci: usize,
    co: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    st: usize,
    sh: usize,
    sw: usize,
    left_t: usize,
    pad_h: usize,
    pad_w: usize,
}

impl Conv3dRef {
    fn load(
        model: &Arc<CmfModel>,
        prefix: &str,
        shape: [usize; 5],
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> VaeResult<Self> {
        let weight = TensorRef::load(model, format!("{prefix}.weight"), &shape)?;
        let bias = TensorRef::load(model, format!("{prefix}.bias"), &[shape[0]])?;
        Ok(Self {
            weight,
            bias,
            co: shape[0],
            ci: shape[1],
            kt: shape[2],
            kh: shape[3],
            kw: shape[4],
            st: stride[0],
            sh: stride[1],
            sw: stride[2],
            left_t: padding[0] * 2,
            pad_h: padding[1],
            pad_w: padding[2],
        })
    }

    fn output_dim(
        input: usize,
        before: usize,
        after: usize,
        kernel: usize,
        stride: usize,
    ) -> VaeResult<usize> {
        if stride == 0 {
            return Err("qwen image VAE: zero convolution stride".into());
        }
        let padded = input
            .checked_add(before)
            .and_then(|n| n.checked_add(after))
            .ok_or_else(|| "qwen image VAE: convolution shape overflow".to_string())?;
        if padded < kernel {
            return Err(format!(
                "qwen image VAE: convolution input {padded} smaller than kernel {kernel}"
            ));
        }
        Ok((padded - kernel) / stride + 1)
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        if x.c != self.ci {
            return Err(format!(
                "qwen image VAE: Conv3d input channels {} != {}",
                x.c, self.ci
            ));
        }
        let ot = Self::output_dim(x.t, self.left_t, 0, self.kt, self.st)?;
        let oh = Self::output_dim(x.h, self.pad_h, self.pad_h, self.kh, self.sh)?;
        let ow = Self::output_dim(x.w, self.pad_w, self.pad_w, self.kw, self.sw)?;
        let npos = ot
            .checked_mul(oh)
            .and_then(|n| n.checked_mul(ow))
            .ok_or_else(|| "qwen image VAE: convolution position overflow".to_string())?;
        let patch_k = self.ci * self.kt * self.kh * self.kw;
        let weight = self.weight.values()?;
        let bias = self.bias.values()?;
        let mut out = Volume::zeros(self.co, ot, oh, ow)?;
        let bytes_per_position = 4usize
            .checked_mul(patch_k.saturating_add(self.co).max(1))
            .unwrap_or(MAX_WORK_BYTES + 1);
        let tile = (MAX_WORK_BYTES / bytes_per_position)
            .max(1)
            .min(npos.max(1));
        let mut patches = vec![0.0f32; tile * patch_k];
        let mut ybuf = vec![0.0f32; tile * self.co];
        let mut p0 = 0usize;
        while p0 < npos {
            let n = tile.min(npos - p0);
            patches[..n * patch_k].fill(0.0);
            for row in 0..n {
                let p = p0 + row;
                let xw = p % ow;
                let rest = p / ow;
                let xh = rest % oh;
                let xt = rest / oh;
                let patch = &mut patches[row * patch_k..(row + 1) * patch_k];
                let mut i = 0usize;
                for ci in 0..self.ci {
                    for kt in 0..self.kt {
                        let st =
                            xt as isize * self.st as isize + kt as isize - self.left_t as isize;
                        for kh in 0..self.kh {
                            let sh =
                                xh as isize * self.sh as isize + kh as isize - self.pad_h as isize;
                            for kw in 0..self.kw {
                                let sw = xw as isize * self.sw as isize + kw as isize
                                    - self.pad_w as isize;
                                patch[i] = if st >= 0
                                    && st < x.t as isize
                                    && sh >= 0
                                    && sh < x.h as isize
                                    && sw >= 0
                                    && sw < x.w as isize
                                {
                                    x.get(ci, st as usize, sh as usize, sw as usize)
                                } else {
                                    0.0
                                };
                                i += 1;
                            }
                        }
                    }
                }
            }
            crate::fcd_ops::gemm_nt(
                &patches[..n * patch_k],
                &weight,
                &mut ybuf[..n * self.co],
                n,
                patch_k,
                self.co,
                None,
            );
            for row in 0..n {
                let p = p0 + row;
                let xw = p % ow;
                let rest = p / ow;
                let xh = rest % oh;
                let xt = rest / oh;
                for co in 0..self.co {
                    out.set(co, xt, xh, xw, ybuf[row * self.co + co] + bias[co]);
                }
            }
            p0 += n;
        }
        Ok(out)
    }
}

/// A spatial convolution used by the resampling blocks and mid attention.
struct Conv2dRef {
    weight: TensorRef,
    bias: TensorRef,
    ci: usize,
    co: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
}

impl Conv2dRef {
    fn load(
        model: &Arc<CmfModel>,
        prefix: &str,
        shape: [usize; 4],
        stride: [usize; 2],
        pads: [usize; 4],
    ) -> VaeResult<Self> {
        let weight = TensorRef::load(model, format!("{prefix}.weight"), &shape)?;
        let bias = TensorRef::load(model, format!("{prefix}.bias"), &[shape[0]])?;
        Ok(Self {
            weight,
            bias,
            co: shape[0],
            ci: shape[1],
            kh: shape[2],
            kw: shape[3],
            sh: stride[0],
            sw: stride[1],
            pad_top: pads[0],
            pad_bottom: pads[1],
            pad_left: pads[2],
            pad_right: pads[3],
        })
    }

    fn forward_frame(&self, x: &[f32], h: usize, w: usize) -> VaeResult<(Vec<f32>, usize, usize)> {
        if x.len() != self.ci * h * w {
            return Err(format!(
                "qwen image VAE: Conv2d input length {} != {}",
                x.len(),
                self.ci * h * w
            ));
        }
        let oh = Conv3dRef::output_dim(h, self.pad_top, self.pad_bottom, self.kh, self.sh)?;
        let ow = Conv3dRef::output_dim(w, self.pad_left, self.pad_right, self.kw, self.sw)?;
        let patch_k = self.ci * self.kh * self.kw;
        let npos = oh * ow;
        let weight = self.weight.values()?;
        let bias = self.bias.values()?;
        let bytes_per_position = 4usize
            .checked_mul(patch_k.saturating_add(self.co).max(1))
            .unwrap_or(MAX_WORK_BYTES + 1);
        let tile = (MAX_WORK_BYTES / bytes_per_position)
            .max(1)
            .min(npos.max(1));
        let mut patches = vec![0.0f32; tile * patch_k];
        let mut ybuf = vec![0.0f32; tile * self.co];
        let mut out = vec![0.0f32; self.co * npos];
        let mut p0 = 0usize;
        while p0 < npos {
            let n = tile.min(npos - p0);
            patches[..n * patch_k].fill(0.0);
            for row in 0..n {
                let p = p0 + row;
                let xw = p % ow;
                let xh = p / ow;
                let patch = &mut patches[row * patch_k..(row + 1) * patch_k];
                let mut i = 0usize;
                for ci in 0..self.ci {
                    let img = &x[ci * h * w..(ci + 1) * h * w];
                    for kh in 0..self.kh {
                        let sy =
                            xh as isize * self.sh as isize + kh as isize - self.pad_top as isize;
                        for kw in 0..self.kw {
                            let sx = xw as isize * self.sw as isize + kw as isize
                                - self.pad_left as isize;
                            patch[i] = if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                img[sy as usize * w + sx as usize]
                            } else {
                                0.0
                            };
                            i += 1;
                        }
                    }
                }
            }
            crate::fcd_ops::gemm_nt(
                &patches[..n * patch_k],
                &weight,
                &mut ybuf[..n * self.co],
                n,
                patch_k,
                self.co,
                None,
            );
            for row in 0..n {
                let p = p0 + row;
                for co in 0..self.co {
                    out[co * npos + p] = ybuf[row * self.co + co] + bias[co];
                }
            }
            p0 += n;
        }
        Ok((out, oh, ow))
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        if x.c != self.ci {
            return Err(format!(
                "qwen image VAE: Conv2d input channels {} != {}",
                x.c, self.ci
            ));
        }
        let mut frames = Vec::with_capacity(x.t);
        let mut oh = 0;
        let mut ow = 0;
        for t in 0..x.t {
            let frame = x.frame(t)?;
            let (out, h, w) = self.forward_frame(&frame, x.h, x.w)?;
            oh = h;
            ow = w;
            frames.push(out);
        }
        Volume::from_frames(&frames, self.co, oh, ow)
    }
}

struct RmsRef {
    gamma: TensorRef,
    channels: usize,
}

impl RmsRef {
    fn load(model: &Arc<CmfModel>, prefix: &str, channels: usize, images: bool) -> VaeResult<Self> {
        let shape = if images {
            vec![channels, 1, 1]
        } else {
            vec![channels, 1, 1, 1]
        };
        Ok(Self {
            gamma: TensorRef::load(model, format!("{prefix}.gamma"), &shape)?,
            channels,
        })
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        if x.c != self.channels {
            return Err(format!(
                "qwen image VAE: RMS channels {} != {}",
                x.c, self.channels
            ));
        }
        let gamma = self.gamma.values()?;
        let mut out = x.clone();
        let scale = (self.channels as f32).sqrt();
        for t in 0..x.t {
            for y in 0..x.h {
                for xx in 0..x.w {
                    let mut norm2 = 0.0f32;
                    for c in 0..x.c {
                        let v = x.get(c, t, y, xx);
                        norm2 += v * v;
                    }
                    let inv = 1.0 / norm2.sqrt().max(EPS_NORMALIZE);
                    for c in 0..x.c {
                        out.set(c, t, y, xx, x.get(c, t, y, xx) * inv * scale * gamma[c]);
                    }
                }
            }
        }
        Ok(out)
    }
}

fn silu_inplace(data: &mut [f32]) {
    for v in data {
        *v /= 1.0 + (-*v).exp();
    }
}

struct ResidualRef {
    norm1: RmsRef,
    conv1: Conv3dRef,
    norm2: RmsRef,
    conv2: Conv3dRef,
    shortcut: Option<Conv3dRef>,
}

impl ResidualRef {
    fn load(model: &Arc<CmfModel>, prefix: &str, ci: usize, co: usize) -> VaeResult<Self> {
        Ok(Self {
            norm1: RmsRef::load(model, &format!("{prefix}.norm1"), ci, false)?,
            conv1: Conv3dRef::load(
                model,
                &format!("{prefix}.conv1"),
                [co, ci, 3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            norm2: RmsRef::load(model, &format!("{prefix}.norm2"), co, false)?,
            conv2: Conv3dRef::load(
                model,
                &format!("{prefix}.conv2"),
                [co, co, 3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            shortcut: if ci != co {
                Some(Conv3dRef::load(
                    model,
                    &format!("{prefix}.conv_shortcut"),
                    [co, ci, 1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let mut h = self.norm1.forward(x)?;
        silu_inplace(&mut h.data);
        h = self.conv1.forward(&h)?;
        h = self.norm2.forward(&h)?;
        silu_inplace(&mut h.data);
        h = self.conv2.forward(&h)?;
        if h.data.len() != shortcut.data.len() {
            return Err("qwen image VAE: residual shape mismatch".into());
        }
        for (dst, skip) in h.data.iter_mut().zip(shortcut.data) {
            *dst += skip;
        }
        Ok(h)
    }
}

struct AttentionRef {
    norm: RmsRef,
    to_qkv: Conv2dRef,
    proj: Conv2dRef,
    channels: usize,
}

impl AttentionRef {
    fn load(model: &Arc<CmfModel>, prefix: &str, channels: usize) -> VaeResult<Self> {
        Ok(Self {
            norm: RmsRef::load(model, &format!("{prefix}.norm"), channels, true)?,
            to_qkv: Conv2dRef::load(
                model,
                &format!("{prefix}.to_qkv"),
                [channels * 3, channels, 1, 1],
                [1, 1],
                [0, 0, 0, 0],
            )?,
            proj: Conv2dRef::load(
                model,
                &format!("{prefix}.proj"),
                [channels, channels, 1, 1],
                [1, 1],
                [0, 0, 0, 0],
            )?,
            channels,
        })
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        // The public VAE is intentionally single-frame.  Diffusers applies
        // this block independently per frame, but accepting a multi-frame
        // call here would require reintroducing cache-index state into the
        // public API and would make the first-frame guarantee ambiguous.
        if x.t != 1 {
            return Err("qwen image VAE: attention currently accepts one frame".into());
        }
        let n = x.h * x.w;
        let normalized = self.norm.forward(x)?;
        let qkv = self.to_qkv.forward(&normalized)?;
        let mut q = vec![0.0f32; n * self.channels];
        let mut k = vec![0.0f32; n * self.channels];
        let mut v = vec![0.0f32; n * self.channels];
        for p in 0..n {
            for c in 0..self.channels {
                q[p * self.channels + c] = qkv.get(c, 0, p / x.w, p % x.w);
                k[p * self.channels + c] = qkv.get(self.channels + c, 0, p / x.w, p % x.w);
                v[p * self.channels + c] = qkv.get(2 * self.channels + c, 0, p / x.w, p % x.w);
            }
        }
        let scale = 1.0 / (self.channels as f32).sqrt();
        let attended = if crate::gpu::enabled_here() && n >= 256 {
            let mut got = vec![0.0f32; n * self.channels];
            if crate::gpu::dit_attention(&q, &k, &v, 1, 1, n, self.channels, scale, &mut got) {
                got
            } else {
                attention_cpu(&q, &k, &v, n, self.channels, scale)
            }
        } else {
            attention_cpu(&q, &k, &v, n, self.channels, scale)
        };
        let attended_volume = Volume::from_frame(
            &token_to_channels(&attended, n, self.channels),
            self.channels,
            x.h,
            x.w,
        )?;
        let projected = self.proj.forward(&attended_volume)?;
        let mut out = x.clone();
        for (dst, value) in out.data.iter_mut().zip(projected.data) {
            *dst += value;
        }
        Ok(out)
    }
}

fn token_to_channels(tokens: &[f32], n: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * n];
    for p in 0..n {
        for c in 0..channels {
            out[c * n + p] = tokens[p * channels + c];
        }
    }
    out
}

/// CPU attention with one score row in flight.  This has the same
/// scaled-dot-product and row-softmax order as the reference while avoiding
/// the `N × N` score allocation at 128x128 mid-blocks.
fn attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    channels: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * channels];
    let mut scores = vec![0.0f32; n];
    for qi in 0..n {
        let qrow = &q[qi * channels..(qi + 1) * channels];
        let mut max_score = f32::NEG_INFINITY;
        for kj in 0..n {
            let krow = &k[kj * channels..(kj + 1) * channels];
            let mut dot = 0.0f32;
            for c in 0..channels {
                dot += qrow[c] * krow[c];
            }
            let score = dot * scale;
            scores[kj] = score;
            max_score = max_score.max(score);
        }
        let mut denom = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        let inv = 1.0 / denom.max(EPS_NORMALIZE);
        let outrow = &mut out[qi * channels..(qi + 1) * channels];
        for kj in 0..n {
            let a = scores[kj] * inv;
            let vrow = &v[kj * channels..(kj + 1) * channels];
            for c in 0..channels {
                outrow[c] += a * vrow[c];
            }
        }
    }
    out
}

enum ResampleMode {
    Down2d,
    Down3d,
    Up2d,
    Up3d,
}

struct ResampleRef {
    mode: ResampleMode,
    spatial: Conv2dRef,
    // Loaded and shape-checked even though the public single-frame path
    // follows Diffusers' first cache call and intentionally skips this
    // temporal convolution.  Retaining the reference makes the cache-first
    // reduction explicit and prevents a silent 2-D-only implementation.
    time_conv: Option<Conv3dRef>,
}

impl ResampleRef {
    fn load(
        model: &Arc<CmfModel>,
        prefix: &str,
        channels: usize,
        temporal: bool,
        up: bool,
    ) -> VaeResult<Self> {
        let (mode, spatial, time_conv) = if up {
            let mode = if temporal {
                ResampleMode::Up3d
            } else {
                ResampleMode::Up2d
            };
            let spatial = Conv2dRef::load(
                model,
                &format!("{prefix}.resample.1"),
                [channels / 2, channels, 3, 3],
                [1, 1],
                [1, 1, 1, 1],
            )?;
            let time_conv = if temporal {
                Some(Conv3dRef::load(
                    model,
                    &format!("{prefix}.time_conv"),
                    [channels * 2, channels, 3, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                )?)
            } else {
                None
            };
            (mode, spatial, time_conv)
        } else {
            let mode = if temporal {
                ResampleMode::Down3d
            } else {
                ResampleMode::Down2d
            };
            let spatial = Conv2dRef::load(
                model,
                &format!("{prefix}.resample.1"),
                [channels, channels, 3, 3],
                [2, 2],
                [0, 1, 0, 1],
            )?;
            let time_conv = if temporal {
                Some(Conv3dRef::load(
                    model,
                    &format!("{prefix}.time_conv"),
                    [channels, channels, 3, 1, 1],
                    [2, 1, 1],
                    [0, 0, 0],
                )?)
            } else {
                None
            };
            (mode, spatial, time_conv)
        };
        Ok(Self {
            mode,
            spatial,
            time_conv,
        })
    }

    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        if x.t != 1 {
            return Err("qwen image VAE: resampling currently accepts one frame".into());
        }
        let y = match self.mode {
            ResampleMode::Up2d | ResampleMode::Up3d => self.spatial.forward(&upsample2x(x)?)?,
            ResampleMode::Down2d | ResampleMode::Down3d => self.spatial.forward(x)?,
        };
        // Diffusers' first-frame calls have a None cache entry.  For
        // upsample3d that skips time_conv before spatial resampling; for
        // downsample3d it stores the spatial result and skips time_conv after
        // it.  The resulting single-frame tensor is exactly the spatial path
        // above.  Do not call time_conv here merely because it is loaded.
        let _first_frame_cache_temporal_conv = &self.time_conv;
        Ok(y)
    }
}

fn upsample2x(x: &Volume) -> VaeResult<Volume> {
    let mut out = Volume::zeros(x.c, x.t, x.h * 2, x.w * 2)?;
    for c in 0..x.c {
        for t in 0..x.t {
            for y in 0..out.h {
                for xx in 0..out.w {
                    out.set(c, t, y, xx, x.get(c, t, y / 2, xx / 2));
                }
            }
        }
    }
    Ok(out)
}

struct MidBlock {
    res0: ResidualRef,
    attention: AttentionRef,
    res1: ResidualRef,
}

impl MidBlock {
    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        let x = self.res0.forward(x)?;
        let x = self.attention.forward(&x)?;
        self.res1.forward(&x)
    }
}

enum EncoderLayer {
    Residual(ResidualRef),
    Attention(AttentionRef),
    Resample(ResampleRef),
}

impl EncoderLayer {
    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        match self {
            Self::Residual(layer) => layer.forward(x),
            Self::Attention(layer) => layer.forward(x),
            Self::Resample(layer) => layer.forward(x),
        }
    }
}

struct EncoderRef {
    conv_in: Conv3dRef,
    layers: Vec<EncoderLayer>,
    mid: MidBlock,
    norm_out: RmsRef,
    conv_out: Conv3dRef,
}

impl EncoderRef {
    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        let mut x = self.conv_in.forward(x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        x = self.mid.forward(&x)?;
        x = self.norm_out.forward(&x)?;
        silu_inplace(&mut x.data);
        self.conv_out.forward(&x)
    }
}

struct UpBlock {
    resnets: Vec<ResidualRef>,
    upsample: Option<ResampleRef>,
}

impl UpBlock {
    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(resample) = &self.upsample {
            x = resample.forward(&x)?;
        }
        Ok(x)
    }
}

struct DecoderRef {
    conv_in: Conv3dRef,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: RmsRef,
    conv_out: Conv3dRef,
}

impl DecoderRef {
    fn forward(&self, x: &Volume) -> VaeResult<Volume> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid.forward(&x)?;
        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }
        x = self.norm_out.forward(&x)?;
        silu_inplace(&mut x.data);
        self.conv_out.forward(&x)
    }
}

/// Native Qwen-Image VAE component.
pub struct QwenImageVae {
    // TensorRef values borrow this mapping; retaining the root Arc makes the
    // ownership obvious and prevents accidental future eager extraction.
    _model: Arc<CmfModel>,
    encoder: EncoderRef,
    quant_conv: Conv3dRef,
    decoder: DecoderRef,
    post_quant_conv: Conv3dRef,
    pub z_dim: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub spatial_compression_ratio: usize,
}

impl QwenImageVae {
    /// Open a VAE-only CMF containing official state-dict tensor names and
    /// the Diffusers config in `image.config_json`.
    pub fn open(path: &Path) -> VaeResult<Self> {
        let model = Arc::new(CmfModel::open(path).map_err(|e| format!("qwen image VAE CMF: {e}"))?);
        let config_entry = model
            .tensor("image.config_json")
            .ok_or_else(|| "qwen image VAE: missing image.config_json".to_string())?;
        if config_entry.dtype != TensorDtype::U8 {
            return Err("qwen image VAE: image.config_json must be U8".into());
        }
        let cfg: serde_json::Value = serde_json::from_slice(model.entry_bytes(config_entry))
            .map_err(|e| format!("qwen image VAE image.config_json: {e}"))?;
        if let Some(class_name) = cfg.get("_class_name").and_then(|v| v.as_str()) {
            if class_name != "AutoencoderKLQwenImage" {
                return Err(format!(
                    "qwen image VAE: image.config_json _class_name is '{class_name}'"
                ));
            }
        }
        let base_dim = cfg_usize(&cfg, "base_dim")?;
        let z_dim = cfg_usize(&cfg, "z_dim")?;
        let dim_mult = cfg_usize_array(&cfg, "dim_mult")?;
        let num_res_blocks = cfg_usize(&cfg, "num_res_blocks")?;
        let attn_scales = cfg_f64_array(&cfg, "attn_scales")?;
        let temporal_downsample = cfg_bool_array(&cfg, "temperal_downsample")?;
        let latents_mean = cfg_f32_array(&cfg, "latents_mean")?;
        let latents_std = cfg_f32_array(&cfg, "latents_std")?;
        if base_dim == 0 || z_dim == 0 || dim_mult.is_empty() || num_res_blocks == 0 {
            return Err("qwen image VAE: configuration contains a zero-sized component".into());
        }
        if temporal_downsample.len() + 1 != dim_mult.len() {
            return Err(format!(
                "qwen image VAE: temperal_downsample length {} must equal dim_mult length {} minus one",
                temporal_downsample.len(),
                dim_mult.len()
            ));
        }
        if latents_mean.len() != z_dim || latents_std.len() != z_dim {
            return Err(format!(
                "qwen image VAE: latent statistics lengths ({}, {}) must equal z_dim {z_dim}",
                latents_mean.len(),
                latents_std.len()
            ));
        }
        if latents_std.iter().any(|v| !v.is_finite() || *v == 0.0) {
            return Err("qwen image VAE: latents_std must be finite and nonzero".into());
        }

        let encoder = load_encoder(
            &model,
            base_dim,
            z_dim,
            &dim_mult,
            num_res_blocks,
            &attn_scales,
            &temporal_downsample,
        )?;
        let quant_conv = Conv3dRef::load(
            &model,
            "quant_conv",
            [z_dim * 2, z_dim * 2, 1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        )?;
        let decoder = load_decoder(
            &model,
            base_dim,
            z_dim,
            &dim_mult,
            num_res_blocks,
            &attn_scales,
            &temporal_downsample,
        )?;
        let post_quant_conv = Conv3dRef::load(
            &model,
            "post_quant_conv",
            [z_dim, z_dim, 1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        )?;
        let spatial_compression_ratio = 1usize
            .checked_shl(temporal_downsample.len() as u32)
            .ok_or_else(|| "qwen image VAE: compression ratio overflow".to_string())?;
        Ok(Self {
            _model: model,
            encoder,
            quant_conv,
            decoder,
            post_quant_conv,
            z_dim,
            latents_mean,
            latents_std,
            spatial_compression_ratio,
        })
    }

    pub fn latent_channels(&self) -> usize {
        self.z_dim
    }

    pub fn spatial_compression(&self) -> usize {
        self.spatial_compression_ratio
    }

    /// Encode one RGB frame in `[-1, 1]` to the posterior mode, in raw VAE
    /// space `[z_dim, latent_height, latent_width]`.  The second half of the
    /// encoder's `2*z_dim` output is the log-variance and is intentionally
    /// ignored, matching `DiagonalGaussianDistribution.mode()`.
    pub fn encode_mean(
        &self,
        nchw_rgb: &[f32],
        height: usize,
        width: usize,
    ) -> VaeResult<Vec<f32>> {
        if height == 0 || width == 0 {
            return Err("qwen image VAE: encode dimensions must be nonzero".into());
        }
        let input = Volume::from_frame(nchw_rgb, 3, height, width)?;
        let encoded = self.encoder.forward(&input)?;
        let quantized = self.quant_conv.forward(&encoded)?;
        if quantized.c < self.z_dim || quantized.t != 1 {
            return Err("qwen image VAE: encoder produced invalid latent shape".into());
        }
        let plane = quantized.h * quantized.w;
        let mut out = vec![0.0f32; self.z_dim * plane];
        for c in 0..self.z_dim {
            for y in 0..quantized.h {
                let src = quantized.offset(c, 0, y, 0);
                let dst = c * plane + y * quantized.w;
                out[dst..dst + quantized.w]
                    .copy_from_slice(&quantized.data[src..src + quantized.w]);
            }
        }
        Ok(out)
    }

    /// Decode raw VAE latents `[z_dim, latent_height, latent_width]` to RGB
    /// NCHW in `[-1, 1]`.  Latent mean/std normalisation belongs to the image
    /// pipeline and is therefore intentionally absent here.
    pub fn decode(
        &self,
        raw_latents: &[f32],
        latent_height: usize,
        latent_width: usize,
    ) -> VaeResult<Vec<f32>> {
        if latent_height == 0 || latent_width == 0 {
            return Err("qwen image VAE: decode dimensions must be nonzero".into());
        }
        let input = Volume::from_frame(raw_latents, self.z_dim, latent_height, latent_width)?;
        let post_quantized = self.post_quant_conv.forward(&input)?;
        let decoded = self.decoder.forward(&post_quantized)?;
        if decoded.c != 3 || decoded.t != 1 {
            return Err("qwen image VAE: decoder produced invalid RGB shape".into());
        }
        // Keep the raw decoder values here.  The integration layer maps
        // [-1, 1] to display RGB and owns its final clamp, so this component
        // does not silently discard an out-of-range diagnostic value.
        decoded.frame(0)
    }
}

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> VaeResult<usize> {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("qwen image VAE: image.config_json missing integer '{key}'"))
}

fn cfg_usize_array(cfg: &serde_json::Value, key: &str) -> VaeResult<Vec<usize>> {
    let values = cfg
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("qwen image VAE: image.config_json missing array '{key}'"))?;
    values
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("qwen image VAE: invalid integer in '{key}'"))
        })
        .collect()
}

fn cfg_f64_array(cfg: &serde_json::Value, key: &str) -> VaeResult<Vec<f64>> {
    let values = cfg
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("qwen image VAE: image.config_json missing array '{key}'"))?;
    values
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| format!("qwen image VAE: invalid float in '{key}'"))
        })
        .collect()
}

fn cfg_f32_array(cfg: &serde_json::Value, key: &str) -> VaeResult<Vec<f32>> {
    cfg_f64_array(cfg, key)?
        .into_iter()
        .map(|v| Ok(v as f32))
        .collect()
}

fn cfg_bool_array(cfg: &serde_json::Value, key: &str) -> VaeResult<Vec<bool>> {
    let values = cfg
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("qwen image VAE: image.config_json missing array '{key}'"))?;
    values
        .iter()
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| format!("qwen image VAE: invalid boolean in '{key}'"))
        })
        .collect()
}

fn has_attention(scales: &[f64], scale: f64) -> bool {
    scales.iter().any(|v| *v == scale)
}

fn load_encoder(
    model: &Arc<CmfModel>,
    base_dim: usize,
    z_dim: usize,
    dim_mult: &[usize],
    num_res_blocks: usize,
    attn_scales: &[f64],
    temporal_downsample: &[bool],
) -> VaeResult<EncoderRef> {
    let conv_in = Conv3dRef::load(
        model,
        "encoder.conv_in",
        [base_dim, 3, 3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    let mut layers = Vec::new();
    let mut current = base_dim;
    let mut scale = 1.0f64;
    let mut block_index = 0usize;
    for (i, &mult) in dim_mult.iter().enumerate() {
        let out = base_dim
            .checked_mul(mult)
            .ok_or_else(|| "qwen image VAE: encoder channel overflow".to_string())?;
        for _ in 0..num_res_blocks {
            let prefix = format!("encoder.down_blocks.{block_index}");
            layers.push(EncoderLayer::Residual(ResidualRef::load(
                model, &prefix, current, out,
            )?));
            current = out;
            block_index += 1;
            if has_attention(attn_scales, scale) {
                let prefix = format!("encoder.down_blocks.{block_index}");
                layers.push(EncoderLayer::Attention(AttentionRef::load(
                    model, &prefix, current,
                )?));
                block_index += 1;
            }
        }
        if i + 1 < dim_mult.len() {
            let prefix = format!("encoder.down_blocks.{block_index}");
            layers.push(EncoderLayer::Resample(ResampleRef::load(
                model,
                &prefix,
                current,
                temporal_downsample[i],
                false,
            )?));
            block_index += 1;
            scale /= 2.0;
        }
    }
    let mid = MidBlock {
        res0: ResidualRef::load(model, "encoder.mid_block.resnets.0", current, current)?,
        attention: AttentionRef::load(model, "encoder.mid_block.attentions.0", current)?,
        res1: ResidualRef::load(model, "encoder.mid_block.resnets.1", current, current)?,
    };
    let norm_out = RmsRef::load(model, "encoder.norm_out", current, false)?;
    let conv_out = Conv3dRef::load(
        model,
        "encoder.conv_out",
        [z_dim * 2, current, 3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    Ok(EncoderRef {
        conv_in,
        layers,
        mid,
        norm_out,
        conv_out,
    })
}

fn load_decoder(
    model: &Arc<CmfModel>,
    base_dim: usize,
    z_dim: usize,
    dim_mult: &[usize],
    num_res_blocks: usize,
    _attn_scales: &[f64],
    temporal_downsample: &[bool],
) -> VaeResult<DecoderRef> {
    let highest = base_dim
        .checked_mul(*dim_mult.last().ok_or("qwen image VAE: empty dim_mult")?)
        .ok_or_else(|| "qwen image VAE: decoder channel overflow".to_string())?;
    let conv_in = Conv3dRef::load(
        model,
        "decoder.conv_in",
        [highest, z_dim, 3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    let mid = MidBlock {
        res0: ResidualRef::load(model, "decoder.mid_block.resnets.0", highest, highest)?,
        attention: AttentionRef::load(model, "decoder.mid_block.attentions.0", highest)?,
        res1: ResidualRef::load(model, "decoder.mid_block.resnets.1", highest, highest)?,
    };
    let mut dims = Vec::with_capacity(dim_mult.len() + 1);
    dims.push(highest);
    for &mult in dim_mult.iter().rev() {
        dims.push(
            base_dim
                .checked_mul(mult)
                .ok_or_else(|| "qwen image VAE: decoder channel overflow".to_string())?,
        );
    }
    let temporal_up: Vec<bool> = temporal_downsample.iter().rev().copied().collect();
    let mut up_blocks = Vec::with_capacity(dim_mult.len());
    for i in 0..dim_mult.len() {
        let mut input = dims[i];
        if i > 0 {
            input /= 2;
        }
        let output = dims[i + 1];
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        for r in 0..=num_res_blocks {
            let prefix = format!("decoder.up_blocks.{i}.resnets.{r}");
            resnets.push(ResidualRef::load(model, &prefix, input, output)?);
            input = output;
        }
        let upsample = if i + 1 < dim_mult.len() {
            Some(ResampleRef::load(
                model,
                &format!("decoder.up_blocks.{i}.upsamplers.0"),
                output,
                temporal_up[i],
                true,
            )?)
        } else {
            None
        };
        up_blocks.push(UpBlock { resnets, upsample });
    }
    let final_dim = *dims.last().unwrap();
    let norm_out = RmsRef::load(model, "decoder.norm_out", final_dim, false)?;
    let conv_out = Conv3dRef::load(
        model,
        "decoder.conv_out",
        [3, final_dim, 3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    Ok(DecoderRef {
        conv_in,
        mid,
        up_blocks,
        norm_out,
        conv_out,
    })
}
