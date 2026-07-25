//! FLUX-class VAE decoder (diffusers `AutoencoderKL`, the Lumina-Image
//! 2.0 latent decoder): latents `[16, h, w]` → RGB `[3, 8h, 8w]`.
//!
//! First increment of the image-generation runtime (docs/GENERATIVE.ru.md):
//! a plain-Rust NCHW decoder — conv2d, GroupNorm(32), SiLU, spatial
//! self-attention, nearest ×2 upsampling — loaded straight from a
//! diffusers `vae/` directory (config.json + safetensors). The CMF
//! packaging comes with the Lumina converter; parity is pinned by a
//! numpy reference (`python/vae_ref.py` + `tests/vae_parity.rs`).
//!
//! Convs run parallel over output channels via scoped threads — naive
//! kernels, good enough to validate the pipeline end-to-end; the im2col
//! + GEMM path rides later on the existing matmul kernels.

use std::path::Path;

// ── Minimal safetensors reader (f32/f16/bf16 → f32) ──────────────────

/// One tensor from a .safetensors file, dequantized to f32.
pub struct StTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Stream every tensor of one .safetensors file through `f` as f32,
/// one at a time — a 9 GB shard costs one raw blob plus the single
/// tensor in flight, not a second full-file f32 copy.
pub fn read_safetensors_each(
    path: &Path,
    f: &mut dyn FnMut(&str, Vec<usize>, Vec<f32>) -> Result<(), String>,
) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err("safetensors: truncated header".into());
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
        .map_err(|e| format!("safetensors header: {e}"))?;
    let base = 8 + hlen;
    let obj = header
        .as_object()
        .ok_or("safetensors: header not an object")?;
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().ok_or("dtype")?;
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or("shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = meta["data_offsets"].as_array().ok_or("offsets")?;
        let (s, e) = (
            offs[0].as_u64().unwrap_or(0) as usize + base,
            offs[1].as_u64().unwrap_or(0) as usize + base,
        );
        let raw = bytes.get(s..e).ok_or("safetensors: span out of file")?;
        let n: usize = shape.iter().product::<usize>().max(1);
        let mut data = Vec::with_capacity(n);
        match dtype {
            "F32" => {
                for c in raw.chunks_exact(4) {
                    data.push(f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            "F16" => {
                for c in raw.chunks_exact(2) {
                    data.push(cortiq_core::quant::f16_to_f32(u16::from_le_bytes(
                        c.try_into().unwrap(),
                    )));
                }
            }
            "BF16" => {
                for c in raw.chunks_exact(2) {
                    let b = u16::from_le_bytes(c.try_into().unwrap());
                    data.push(f32::from_bits((b as u32) << 16));
                }
            }
            other => return Err(format!("safetensors: unsupported dtype {other}")),
        }
        f(name, shape, data)?;
    }
    Ok(())
}

/// All tensors of one .safetensors file, keyed by name.
pub fn read_safetensors(
    path: &Path,
) -> Result<std::collections::HashMap<String, StTensor>, String> {
    let mut out = std::collections::HashMap::new();
    read_safetensors_each(path, &mut |name, shape, data| {
        out.insert(name.to_string(), StTensor { shape, data });
        Ok(())
    })?;
    Ok(out)
}

// ── NCHW ops ─────────────────────────────────────────────────────────

/// 2-D convolution, stride 1, square kernel, symmetric padding
/// (pad = k/2). Parallel over output channels.
pub struct Conv2d {
    pub w: Vec<f32>, // [oc, ic, k, k]
    pub b: Vec<f32>, // [oc]
    pub oc: usize,
    pub ic: usize,
    pub k: usize,
}

impl Conv2d {
    fn from(t: &StTensor, bias: &StTensor) -> Self {
        let (oc, ic, k) = (t.shape[0], t.shape[1], t.shape[2]);
        Self {
            w: t.data.clone(),
            b: bias.data.clone(),
            oc,
            ic,
            k,
        }
    }

    /// `x`: [ic, h, w] → [oc, h, w]. Banded im2col + GEMM: bands of
    /// output rows are lowered to a [rows·w, ic·k²] patch matrix and hit
    /// `fcd_ops::gemm_nt` (Accelerate/AMX on macOS, the portable blocked
    /// kernel elsewhere) — the band cap keeps the patch matrix ≤ ~128 MB
    /// at any image size.
    pub fn apply(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.ic * h * w);
        let pad = self.k / 2;
        let ick2 = self.ic * self.k * self.k;
        let mut out = vec![0f32; self.oc * h * w];
        let band = (128 << 20) / (ick2 * w * 4).max(1);
        let band = band.clamp(1, h);
        let mut cols = vec![0f32; band * w * ick2];
        let mut yt = vec![0f32; band * w * self.oc];
        let mut y0 = 0usize;
        while y0 < h {
            let rows = band.min(h - y0);
            let hw_band = rows * w;
            // im2col: row p of `cols` = the receptive field of output
            // position p (zero-padded at the borders).
            for (dy, colrow) in cols[..hw_band * ick2].chunks_mut(w * ick2).enumerate() {
                let y = y0 + dy;
                for (xx, patch) in colrow.chunks_mut(ick2).enumerate() {
                    let mut i = 0;
                    for c in 0..self.ic {
                        let img = &x[c * h * w..(c + 1) * h * w];
                        for ky in 0..self.k {
                            let sy = y as isize + ky as isize - pad as isize;
                            for kx in 0..self.k {
                                let sx = xx as isize + kx as isize - pad as isize;
                                patch[i] =
                                    if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                        img[sy as usize * w + sx as usize]
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
                &cols[..hw_band * ick2],
                &self.w,
                &mut yt[..hw_band * self.oc],
                hw_band,
                ick2,
                self.oc,
                None,
            );
            // [hw, oc] → NCHW [oc, hw] + bias.
            for o in 0..self.oc {
                let b = self.b[o];
                let dst = &mut out[o * h * w + y0 * w..][..hw_band];
                for (p, d) in dst.iter_mut().enumerate() {
                    *d = yt[p * self.oc + o] + b;
                }
            }
            y0 += rows;
        }
        out
    }
}

/// GroupNorm over channel groups (eps 1e-6, affine), NCHW in place.
pub struct GroupNorm {
    pub g: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl GroupNorm {
    fn from(w: &StTensor, b: &StTensor, groups: usize) -> Self {
        Self {
            g: groups,
            w: w.data.clone(),
            b: b.data.clone(),
        }
    }

    pub fn apply(&self, x: &mut [f32], h: usize, w: usize) {
        let c = self.w.len();
        let per = c / self.g;
        let hw = h * w;
        // One thread per group (32 groups saturate the cores; the pass
        // is memory-bound, f64 accumulation kept for parity).
        std::thread::scope(|s| {
            for (gi, span) in x.chunks_mut(per * hw).enumerate() {
                let (wref, bref) = (&self.w, &self.b);
                s.spawn(move || {
                    let n = span.len() as f64;
                    let mean = span.iter().map(|&v| v as f64).sum::<f64>() / n;
                    let var = span.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
                    let inv = 1.0 / (var + 1e-6).sqrt();
                    for (ci, ch) in span.chunks_mut(hw).enumerate() {
                        let cc = gi * per + ci;
                        let (sw, sb) = (wref[cc], bref[cc]);
                        for v in ch.iter_mut() {
                            *v = ((*v as f64 - mean) * inv) as f32 * sw + sb;
                        }
                    }
                });
            }
        });
    }
}

fn silu(x: &mut [f32]) {
    let nt = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk = x.len().div_ceil(nt).max(1 << 14);
    std::thread::scope(|s| {
        for part in x.chunks_mut(chunk) {
            s.spawn(move || {
                for v in part.iter_mut() {
                    *v /= 1.0 + (-*v).exp();
                }
            });
        }
    });
}

/// Nearest-neighbour ×2 upsample, NCHW.
fn upsample2x(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * 4 * h * w];
    for ci in 0..c {
        let src = &x[ci * h * w..(ci + 1) * h * w];
        let dst = &mut out[ci * 4 * h * w..(ci + 1) * 4 * h * w];
        for y in 0..2 * h {
            for xx in 0..2 * w {
                dst[y * 2 * w + xx] = src[(y / 2) * w + xx / 2];
            }
        }
    }
    out
}

struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn apply(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut t = x.to_vec();
        self.norm1.apply(&mut t, h, w);
        silu(&mut t);
        let mut t = self.conv1.apply(&t, h, w);
        self.norm2.apply(&mut t, h, w);
        silu(&mut t);
        let t = self.conv2.apply(&t, h, w);
        let skip = match &self.shortcut {
            Some(sc) => sc.apply(x, h, w),
            None => x.to_vec(),
        };
        skip.iter().zip(&t).map(|(a, b)| a + b).collect()
    }
}

/// Single-head self-attention over the spatial grid (mid-block).
struct AttnBlock {
    norm: GroupNorm,
    q: (Vec<f32>, Vec<f32>), // [c, c] weight (row-major out×in), bias
    k: (Vec<f32>, Vec<f32>),
    v: (Vec<f32>, Vec<f32>),
    out: (Vec<f32>, Vec<f32>),
    c: usize,
}

impl AttnBlock {
    /// Token-major dense projection: `x` [hw, c] → [hw, c] via one
    /// `gemm_nt` (weight rows are output channels), bias fused after.
    fn proj(w: &[f32], b: &[f32], x: &[f32], hw: usize, c: usize) -> Vec<f32> {
        let mut y = vec![0f32; hw * c];
        crate::fcd_ops::gemm_nt(x, w, &mut y, hw, c, c, None);
        for row in y.chunks_mut(c) {
            for (v, bb) in row.iter_mut().zip(b) {
                *v += bb;
            }
        }
        y
    }

    fn apply(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (c, hw) = (self.c, h * w);
        let mut n = x.to_vec();
        self.norm.apply(&mut n, h, w);
        // Channel-major → token-major once; everything below is GEMM.
        let mut nt = vec![0f32; hw * c];
        for ci in 0..c {
            for p in 0..hw {
                nt[p * c + ci] = n[ci * hw + p];
            }
        }
        let mut q = Self::proj(&self.q.0, &self.q.1, &nt, hw, c);
        let k = Self::proj(&self.k.0, &self.k.1, &nt, hw, c);
        let v = Self::proj(&self.v.0, &self.v.1, &nt, hw, c);
        let scale = 1.0 / (c as f32).sqrt();
        for qv in q.iter_mut() {
            *qv *= scale;
        }
        // scores[i, j] = q_i · k_j; row softmax; out = attn · v.
        let mut scores = vec![0f32; hw * hw];
        crate::fcd_ops::gemm_nt(&q, &k, &mut scores, hw, c, hw, None);
        for row in scores.chunks_mut(hw) {
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0f32;
            for r in row.iter_mut() {
                *r = (*r - mx).exp();
                den += *r;
            }
            let inv = 1.0 / den;
            for r in row.iter_mut() {
                *r *= inv;
            }
        }
        // gemm_nt wants the right operand transposed: v as [c, hw].
        let mut vt = vec![0f32; c * hw];
        for p in 0..hw {
            for ci in 0..c {
                vt[ci * hw + p] = v[p * c + ci];
            }
        }
        let mut ot = vec![0f32; hw * c];
        crate::fcd_ops::gemm_nt(&scores, &vt, &mut ot, hw, hw, c, None);
        let o = Self::proj(&self.out.0, &self.out.1, &ot, hw, c);
        // Token-major → channel-major + residual.
        let mut y = x.to_vec();
        for p in 0..hw {
            for ci in 0..c {
                y[ci * hw + p] += o[p * c + ci];
            }
        }
        y
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

/// The full decoder: `conv_in → mid(res, attn, res) → up-blocks →
/// GroupNorm + SiLU → conv_out`, plus the diffusers latent
/// de-normalization `z/scaling_factor + shift_factor`.
pub struct VaeDecoder {
    conv_in: Conv2d,
    mid_res1: ResnetBlock,
    mid_attn: AttnBlock,
    mid_res2: ResnetBlock,
    ups: Vec<UpBlock>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    pub latent_channels: usize,
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl VaeDecoder {
    /// Load from a diffusers `vae/` directory (config.json +
    /// diffusion_pytorch_model.safetensors).
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let t = read_safetensors(&dir.join("diffusion_pytorch_model.safetensors"))?;
        Self::from_tensors(t, &cfg)
    }

    /// Load from a packaged imagegen .cmf (`vae.*` tensors, stored
    /// f16/f32, + `vae.config_json`).
    pub fn from_cmf(model: &cortiq_core::CmfModel) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("vae.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("vae.config_json: {e}"))?;
        let mut t = std::collections::HashMap::new();
        for entry in model
            .tensors
            .iter()
            .filter(|e| e.name.starts_with("vae.") && e.name != "vae.config_json")
        {
            let data = crate::dit::cmf_f32(model, &entry.name)?;
            t.insert(
                entry.name["vae.".len()..].to_string(),
                StTensor {
                    shape: entry.shape.clone(),
                    data,
                },
            );
        }
        Self::from_tensors(t, &cfg)
    }

    fn from_tensors(
        t: std::collections::HashMap<String, StTensor>,
        cfg: &serde_json::Value,
    ) -> Result<Self, String> {
        let groups = cfg["norm_num_groups"].as_u64().unwrap_or(32) as usize;
        let get = |n: &str| -> Result<&StTensor, String> {
            t.get(n).ok_or_else(|| format!("missing tensor {n}"))
        };
        let conv = |n: &str| -> Result<Conv2d, String> {
            Ok(Conv2d::from(
                get(&format!("{n}.weight"))?,
                get(&format!("{n}.bias"))?,
            ))
        };
        let gnorm = |n: &str| -> Result<GroupNorm, String> {
            Ok(GroupNorm::from(
                get(&format!("{n}.weight"))?,
                get(&format!("{n}.bias"))?,
                groups,
            ))
        };
        let resnet = |n: &str| -> Result<ResnetBlock, String> {
            Ok(ResnetBlock {
                norm1: gnorm(&format!("{n}.norm1"))?,
                conv1: conv(&format!("{n}.conv1"))?,
                norm2: gnorm(&format!("{n}.norm2"))?,
                conv2: conv(&format!("{n}.conv2"))?,
                shortcut: if t.contains_key(&format!("{n}.conv_shortcut.weight")) {
                    Some(conv(&format!("{n}.conv_shortcut"))?)
                } else {
                    None
                },
            })
        };
        // Attention projections are [c, c] linears (diffusers stores
        // to_q/... as Linear weights).
        let lin = |n: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((
                get(&format!("{n}.weight"))?.data.clone(),
                get(&format!("{n}.bias"))?.data.clone(),
            ))
        };
        let attn_c = get("decoder.mid_block.attentions.0.to_q.weight")?.shape[0];
        let mid_attn = AttnBlock {
            norm: gnorm("decoder.mid_block.attentions.0.group_norm")?,
            q: lin("decoder.mid_block.attentions.0.to_q")?,
            k: lin("decoder.mid_block.attentions.0.to_k")?,
            v: lin("decoder.mid_block.attentions.0.to_v")?,
            out: lin("decoder.mid_block.attentions.0.to_out.0")?,
            c: attn_c,
        };
        let mut ups = Vec::new();
        for b in 0.. {
            if !t.contains_key(&format!("decoder.up_blocks.{b}.resnets.0.conv1.weight")) {
                break;
            }
            let mut resnets = Vec::new();
            for r in 0.. {
                let n = format!("decoder.up_blocks.{b}.resnets.{r}");
                if !t.contains_key(&format!("{n}.conv1.weight")) {
                    break;
                }
                resnets.push(resnet(&n)?);
            }
            let upsample =
                if t.contains_key(&format!("decoder.up_blocks.{b}.upsamplers.0.conv.weight")) {
                    Some(conv(&format!("decoder.up_blocks.{b}.upsamplers.0.conv"))?)
                } else {
                    None
                };
            ups.push(UpBlock { resnets, upsample });
        }
        Ok(Self {
            conv_in: conv("decoder.conv_in")?,
            mid_res1: resnet("decoder.mid_block.resnets.0")?,
            mid_attn,
            mid_res2: resnet("decoder.mid_block.resnets.1")?,
            ups,
            norm_out: gnorm("decoder.conv_norm_out")?,
            conv_out: conv("decoder.conv_out")?,
            latent_channels: cfg["latent_channels"].as_u64().unwrap_or(16) as usize,
            scaling_factor: cfg["scaling_factor"].as_f64().unwrap_or(1.0) as f32,
            shift_factor: cfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
        })
    }

    /// Decode latents `[latent_channels, h, w]` (model scale, i.e. as
    /// produced by the diffusion loop) into RGB `[3, 8h, 8w]` in [-1, 1].
    pub fn decode(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        let prof = std::env::var("CMF_VAE_PROF").is_ok();
        macro_rules! stage {
            ($name:expr, $e:expr) => {{
                let t = std::time::Instant::now();
                let r = $e;
                if prof {
                    eprintln!("vae {}: {:.2}s", $name, t.elapsed().as_secs_f64());
                }
                r
            }};
        }
        let z: Vec<f32> = z
            .iter()
            .map(|&v| v / self.scaling_factor + self.shift_factor)
            .collect();
        let mut x = stage!("conv_in", self.conv_in.apply(&z, h, w));
        x = stage!("mid_res1", self.mid_res1.apply(&x, h, w));
        x = stage!("mid_attn", self.mid_attn.apply(&x, h, w));
        x = stage!("mid_res2", self.mid_res2.apply(&x, h, w));
        let (mut h, mut w) = (h, w);
        for (ui, up) in self.ups.iter().enumerate() {
            for (ri, r) in up.resnets.iter().enumerate() {
                x = stage!(format!("up{ui}.res{ri} ({h}x{w})"), r.apply(&x, h, w));
            }
            if let Some(upc) = &up.upsample {
                let c = upc.ic;
                x = upsample2x(&x, c, h, w);
                h *= 2;
                w *= 2;
                x = stage!(format!("up{ui}.conv ({h}x{w})"), upc.apply(&x, h, w));
            }
        }
        self.norm_out.apply(&mut x, h, w);
        silu(&mut x);
        stage!("conv_out", self.conv_out.apply(&x, h, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// conv2d against a hand-computed 3×3 case with padding.
    #[test]
    fn conv2d_matches_hand_reference() {
        // 1 in-channel 3×3 image, 1 out-channel 3×3 kernel of ones,
        // bias 0.5: each output = sum of the 3×3 neighbourhood + 0.5.
        let c = Conv2d {
            w: vec![1.0; 9],
            b: vec![0.5],
            oc: 1,
            ic: 1,
            k: 3,
        };
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = c.apply(&x, 3, 3);
        // centre: 1+2+..+9 + 0.5 = 45.5; corner (0,0): 1+2+4+5 + 0.5 = 12.5.
        assert_eq!(y[4], 45.5);
        assert_eq!(y[0], 12.5);
        assert_eq!(y[8], 5. + 6. + 8. + 9. + 0.5);
    }

    /// GroupNorm: two groups of one channel each — plain per-channel
    /// standardization times affine.
    #[test]
    fn group_norm_standardizes() {
        let gn = GroupNorm {
            g: 2,
            w: vec![2.0, 1.0],
            b: vec![0.0, 3.0],
        };
        let mut x = vec![1., 3., 5., 7., 10., 10., 10., 10.];
        gn.apply(&mut x, 2, 2);
        // ch0: mean 4, std sqrt(5) → (1-4)/√5*2 …
        let s = 5f64.sqrt();
        assert!((x[0] as f64 - (-3.0 / s * 2.0)).abs() < 1e-5);
        // ch1: constant → normalized 0 → bias 3.
        assert!((x[4] - 3.0).abs() < 1e-4);
    }

    /// Upsample doubles both dimensions with nearest fill.
    #[test]
    fn upsample_nearest() {
        let x = vec![1., 2., 3., 4.];
        let y = upsample2x(&x, 1, 2, 2);
        assert_eq!(
            y,
            vec![
                1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.,
            ]
        );
    }
}
