//! Qwen3-VL's vision tower — the half of `fl2va` that rides in the
//! TEXT stream.
//!
//! MiniMax-H3 conditions on a keyframe twice: the VAE latent goes to the
//! DiT as a condition row, and the picture itself goes to the prompt
//! encoder as `"<Picture 1>: "` followed by a vision block. This is the
//! second path — 27 ViT blocks at hidden 1152 over 16×16 patches, a
//! patch merger down to the language model's 5120, and three deepstack
//! mergers whose output is added into the LM's residual stream at layers
//! 8, 16 and 24.
//!
//! Three conventions here are easy to get wrong and are therefore
//! spelled out:
//!
//! * **The position grid is interpolated, not looked up.** The table is
//!   48×48 and an image is whatever it is, so each patch reads four
//!   neighbours bilinearly — and the result is then reordered into
//!   2×2 merge blocks, because that is the order the merger consumes.
//! * **The rotation is 2-D.** Every patch carries a (row, column) pair,
//!   each contributing 18 angles; the 36 are concatenated with
//!   themselves to 72 and applied split-half over the whole head.
//! * **Two different GELUs.** The per-block MLP uses the tanh
//!   approximation; both mergers use the exact one. They are not
//!   interchangeable at this width.
//!
//! Parity: `tools/mk_qwen3vis_toy.py` against ComfyUI's own
//! `Qwen3VLVisionModel`.

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Side of the learned position grid (2304 entries).
const GRID_SIDE: usize = 48;
const EPS: f64 = 1e-6;

struct Block {
    n1_w: Vec<f32>,
    n1_b: Vec<f32>,
    n2_w: Vec<f32>,
    n2_b: Vec<f32>,
    qkv: Proj,
    qkv_b: Vec<f32>,
    proj: Proj,
    proj_b: Vec<f32>,
    fc1: Proj,
    fc1_b: Vec<f32>,
    fc2: Proj,
    fc2_b: Vec<f32>,
}

/// A merger: LayerNorm, then two linears with an exact GELU between.
/// The main one norms BEFORE the 2×2 shuffle, the deepstack ones after —
/// which is the only thing that distinguishes them.
struct Merger {
    n_w: Vec<f32>,
    n_b: Vec<f32>,
    fc1: Proj,
    fc1_b: Vec<f32>,
    fc2: Proj,
    fc2_b: Vec<f32>,
    post_shuffle: bool,
}

pub struct VisionTower {
    patch: Proj, // [hidden, in·t·p·p] — the conv is a linear per patch
    patch_b: Vec<f32>,
    pos: Vec<f32>, // [GRID_SIDE², hidden]
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    deepstack_at: Vec<usize>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    pub out_hidden: usize,
    heads: usize,
    head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    /// GEMMs on the device (`gpu::vision_gemm_nt`, f32 matrices) and the
    /// attention on `gpu::dit_attention` — SenseNova-Loop's tower
    /// ([`VisionTower::from_cmf_device`]); MiniMax-H3 keeps the host path.
    device: bool,
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + EPS).sqrt();
    for (((d, &v), &g), &bb) in dst.iter_mut().zip(x).zip(w).zip(b) {
        *d = ((v as f64 - mean) * inv) as f32 * g + bb;
    }
}

/// `x·Φ(x)` with the error function — what `F.gelu` does by default.
fn gelu_exact(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v as f64 / std::f64::consts::SQRT_2) as f32)
}

/// The tanh approximation — what the per-block MLP asks for by name.
fn gelu_tanh(v: f32) -> f32 {
    const C: f32 = 0.797_884_6; // √(2/π)
    0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh())
}

/// `erf` by the Numerical Recipes rational form — 1.2e-7 relative
/// everywhere, which is comfortably under what an f32 activation can
/// carry. Written out because the alternatives here are a series that
/// loses digits to cancellation past |x| = 3 and a continued fraction
/// that is easy to get subtly wrong; this one is checked against known
/// values below.
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    // `ans` is erfc(|x|); erfc is 2 − that on the negative side.
    if x >= 0.0 { 1.0 - ans } else { ans - 1.0 }
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

impl Merger {
    fn load(model: &Arc<CmfModel>, prefix: &str, post_shuffle: bool) -> Result<Self, String> {
        let f = |n: &str| crate::dit::cmf_f32(model, n);
        Ok(Self {
            n_w: f(&format!("{prefix}.norm.weight"))?,
            n_b: f(&format!("{prefix}.norm.bias"))?,
            fc1: Proj::from_model(model, &format!("{prefix}.linear_fc1.weight"))?,
            fc1_b: f(&format!("{prefix}.linear_fc1.bias"))?,
            fc2: Proj::from_model(model, &format!("{prefix}.linear_fc2.weight"))?,
            fc2_b: f(&format!("{prefix}.linear_fc2.bias"))?,
            post_shuffle,
        })
    }

    /// `[n, hidden]` → `[n/merge², out]`. `merge_dim = hidden·merge²`.
    fn apply(
        &self,
        x: &[f32],
        n: usize,
        hidden: usize,
        merge_dim: usize,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let groups = n * hidden / merge_dim;
        let mut buf = vec![0f32; n * hidden];
        if self.post_shuffle {
            // norm over the merged 4608, after the shuffle
            for g in 0..groups {
                layer_norm(
                    &x[g * merge_dim..(g + 1) * merge_dim],
                    &self.n_w,
                    &self.n_b,
                    &mut buf[g * merge_dim..(g + 1) * merge_dim],
                );
            }
        } else {
            // norm over the 1152 of each patch, before the shuffle
            for p in 0..n {
                layer_norm(
                    &x[p * hidden..(p + 1) * hidden],
                    &self.n_w,
                    &self.n_b,
                    &mut buf[p * hidden..(p + 1) * hidden],
                );
            }
        }
        let mut h = vec![0f32; groups * merge_dim];
        self.fc1.matmat(&buf, groups, &mut h, pool);
        for r in h.chunks_exact_mut(merge_dim) {
            for (v, &b) in r.iter_mut().zip(&self.fc1_b) {
                *v = gelu_exact(*v + b);
            }
        }
        let out_dim = self.fc2.rows();
        let mut out = vec![0f32; groups * out_dim];
        self.fc2.matmat(&h, groups, &mut out, pool);
        for r in out.chunks_exact_mut(out_dim) {
            for (v, &b) in r.iter_mut().zip(&self.fc2_b) {
                *v += b;
            }
        }
        out
    }
}

impl VisionTower {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("vis.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("vis.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let n = u("depth", 27);
        let f = |s: &str| crate::dit::cmf_f32(model, s);
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("vis.blocks.{i}");
            blocks.push(Block {
                n1_w: f(&format!("{p}.norm1.weight"))?,
                n1_b: f(&format!("{p}.norm1.bias"))?,
                n2_w: f(&format!("{p}.norm2.weight"))?,
                n2_b: f(&format!("{p}.norm2.bias"))?,
                qkv: Proj::from_model(model, &format!("{p}.attn.qkv.weight"))?,
                qkv_b: f(&format!("{p}.attn.qkv.bias"))?,
                proj: Proj::from_model(model, &format!("{p}.attn.proj.weight"))?,
                proj_b: f(&format!("{p}.attn.proj.bias"))?,
                fc1: Proj::from_model(model, &format!("{p}.mlp.linear_fc1.weight"))?,
                fc1_b: f(&format!("{p}.mlp.linear_fc1.bias"))?,
                fc2: Proj::from_model(model, &format!("{p}.mlp.linear_fc2.weight"))?,
                fc2_b: f(&format!("{p}.mlp.linear_fc2.bias"))?,
            });
        }
        let deepstack_at: Vec<usize> = cfg["deepstack_visual_indexes"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let deepstack = (0..deepstack_at.len())
            .map(|i| Merger::load(model, &format!("vis.deepstack.{i}"), true))
            .collect::<Result<Vec<_>, _>>()?;
        let hidden = u("hidden_size", 1152);
        Ok(Self {
            // The conv kernel may be stored as its source 5-D
            // `[hidden, in, t, p, p]`: a linear over the flattened patch.
            patch: match Proj::from_model(model, "vis.patch_embed.weight") {
                Ok(p) => p,
                Err(_) => Proj::f32(f("vis.patch_embed.weight")?, {
                    let t = model.tensor("vis.patch_embed.weight").ok_or("vis.patch_embed.weight")?;
                    t.shape.iter().skip(1).product()
                }),
            },
            patch_b: f("vis.patch_embed.bias")?,
            pos: f("vis.pos_embed.weight")?,
            blocks,
            merger: Merger::load(model, "vis.merger", false)?,
            deepstack,
            deepstack_at,
            pool: Pool::from_env(),
            hidden,
            out_hidden: u("out_hidden_size", 5120),
            heads: u("num_heads", 16),
            head_dim: hidden / u("num_heads", 16),
            patch_size: u("patch_size", 16),
            temporal_patch: u("temporal_patch_size", 2),
            merge: u("spatial_merge_size", 2),
            device: false,
        })
    }

    /// The tower with f32 matrices for the device GEMM and the device
    /// attention (each falls back to the host when refused).
    pub fn from_cmf_device(model: &Arc<CmfModel>) -> Result<Self, String> {
        let mut t = Self::from_cmf(model)?;
        let f32p = |name: &str, p: &Proj| -> Result<Proj, String> {
            Ok(match p {
                Proj::F32 { .. } => Proj::f32(crate::dit::cmf_f32(model, name)?, p.cols()),
                Proj::Q(_) => Proj::f32(crate::dit::cmf_f32(model, name)?, p.cols()),
            })
        };
        t.patch = f32p("vis.patch_embed.weight", &t.patch)?;
        for (i, b) in t.blocks.iter_mut().enumerate() {
            let p = format!("vis.blocks.{i}");
            b.qkv = f32p(&format!("{p}.attn.qkv.weight"), &b.qkv)?;
            b.proj = f32p(&format!("{p}.attn.proj.weight"), &b.proj)?;
            b.fc1 = f32p(&format!("{p}.mlp.linear_fc1.weight"), &b.fc1)?;
            b.fc2 = f32p(&format!("{p}.mlp.linear_fc2.weight"), &b.fc2)?;
        }
        t.merger.fc1 = f32p("vis.merger.linear_fc1.weight", &t.merger.fc1)?;
        t.merger.fc2 = f32p("vis.merger.linear_fc2.weight", &t.merger.fc2)?;
        t.device = true;
        Ok(t)
    }

    /// `y = x · Wᵀ` on the device GEMM when this tower runs there.
    fn mm(&self, p: &Proj, x: &[f32], n: usize, out: &mut [f32], pool: Option<&Pool>) {
        if self.device && crate::gpu::enabled_here() {
            if let Proj::F32 { w, rows, cols } = p {
                if crate::gpu::vision_gemm_nt(x, w, out, n, *cols, *rows) {
                    return;
                }
            }
        }
        p.matmat(x, n, out, pool);
    }

    /// Bilinear read of the 48×48 grid at `h`×`w`, then reordered into
    /// 2×2 merge blocks: `[h·w, hidden]` in the order the blocks see.
    fn positions(&self, h: usize, w: usize) -> Vec<f32> {
        let hd = self.hidden;
        let m = self.merge;
        let lin = |n: usize, i: usize| -> f64 {
            if n == 1 {
                0.0
            } else {
                i as f64 * (GRID_SIDE - 1) as f64 / (n - 1) as f64
            }
        };
        let mut flat = vec![0f32; h * w * hd];
        for y in 0..h {
            let fy = lin(h, y);
            let y0 = fy as usize;
            let y1 = (y0 + 1).min(GRID_SIDE - 1);
            let dy = (fy - y0 as f64) as f32;
            for x in 0..w {
                let fx = lin(w, x);
                let x0 = fx as usize;
                let x1 = (x0 + 1).min(GRID_SIDE - 1);
                let dx = (fx - x0 as f64) as f32;
                let dst = &mut flat[(y * w + x) * hd..(y * w + x + 1) * hd];
                for (idx, wt) in [
                    (y0 * GRID_SIDE + x0, (1.0 - dy) * (1.0 - dx)),
                    (y0 * GRID_SIDE + x1, (1.0 - dy) * dx),
                    (y1 * GRID_SIDE + x0, dy * (1.0 - dx)),
                    (y1 * GRID_SIDE + x1, dy * dx),
                ] {
                    let src = &self.pos[idx * hd..(idx + 1) * hd];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d += s * wt;
                    }
                }
            }
        }
        // (h/m, m, w/m, m) → (h/m, w/m, m, m)
        let mut out = vec![0f32; h * w * hd];
        let mut k = 0usize;
        for by in 0..h / m {
            for bx in 0..w / m {
                for iy in 0..m {
                    for ix in 0..m {
                        let s = ((by * m + iy) * w + bx * m + ix) * hd;
                        out[k * hd..(k + 1) * hd].copy_from_slice(&flat[s..s + hd]);
                        k += 1;
                    }
                }
            }
        }
        out
    }

    /// 18 angles for the row index and 18 for the column, in the same
    /// merge-block order: `[n, 36]`.
    fn rope_angles(&self, h: usize, w: usize) -> Vec<f32> {
        let half = self.head_dim / 2; // 36
        let k = half / 2; // 18 frequencies per axis
        let inv: Vec<f64> = (0..k)
            .map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / half as f64))
            .collect();
        let m = self.merge;
        let mut out = Vec::with_capacity(h * w * half);
        for by in 0..h / m {
            for bx in 0..w / m {
                for iy in 0..m {
                    for ix in 0..m {
                        let (r, c) = ((by * m + iy) as f64, (bx * m + ix) as f64);
                        for &f in &inv {
                            out.push((r * f) as f32);
                        }
                        for &f in &inv {
                            out.push((c * f) as f32);
                        }
                    }
                }
            }
        }
        out
    }

    fn attention(&self, qkv: &[f32], n: usize, angles: &[f32], out: &mut [f32]) {
        let (nh, hd) = (self.heads, self.head_dim);
        let inner = nh * hd;
        let half = hd / 2;
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = self.pool.as_deref();
        if self.device && n >= 64 && crate::gpu::enabled_here() {
            // Head-major planes, q and k rotated (unscaled), for the device:
            // one sin/cos per (patch, angle), the heads across the pool.
            let mut cs = vec![(0f32, 0f32); n * half];
            for (i, a) in angles[..n * half].iter().enumerate() {
                cs[i] = a.sin_cos();
            }
            let (mut qa, mut ka, mut va) = (vec![0f32; nh * n * hd], vec![0f32; nh * n * hd], vec![0f32; nh * n * hd]);
            let (qp, kp, vp) = (SendPtr(qa.as_mut_ptr()), SendPtr(ka.as_mut_ptr()), SendPtr(va.as_mut_ptr()));
            let fill = |lo: usize, hi: usize| {
                for h in lo..hi {
                    // SAFETY: each worker owns whole head planes.
                    let (q, k, v) = unsafe { (qp.row(h * n * hd, n * hd), kp.row(h * n * hd, n * hd), vp.row(h * n * hd, n * hd)) };
                    for p in 0..n {
                        let base = p * 3 * inner;
                        let (qs, ks, vs) = (
                            &qkv[base + h * hd..base + (h + 1) * hd],
                            &qkv[base + inner + h * hd..base + inner + (h + 1) * hd],
                            &qkv[base + 2 * inner + h * hd..base + 2 * inner + (h + 1) * hd],
                        );
                        let o = p * hd;
                        for j in 0..half {
                            let (s, c) = cs[p * half + j];
                            q[o + j] = qs[j] * c - qs[j + half] * s;
                            q[o + j + half] = qs[j] * s + qs[j + half] * c;
                            k[o + j] = ks[j] * c - ks[j + half] * s;
                            k[o + j + half] = ks[j] * s + ks[j + half] * c;
                        }
                        v[o..o + hd].copy_from_slice(vs);
                    }
                }
            };
            match pool {
                Some(p) => p.run_rows(nh, &fill),
                None => fill(0, nh),
            }
            if crate::gpu::dit_attention(&qa, &ka, &va, nh, nh, n, hd, scale, out) {
                return;
            }
        }
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                // qkv is [n, 3, heads, hd] — the head axis is innermost
                // of the three, not three separate planes.
                let base = p * 3 * inner;
                let (qs, ks, vs) = (
                    &qkv[base + h * hd..base + (h + 1) * hd],
                    &qkv[base + inner + h * hd..base + inner + (h + 1) * hd],
                    &qkv[base + 2 * inner + h * hd..base + 2 * inner + (h + 1) * hd],
                );
                let ang = &angles[p * half..(p + 1) * half];
                // split-half over the whole head: angle j serves dims
                // j and j+half, and the angle list is the 36 duplicated.
                for j in 0..half {
                    let (s, c) = ang[j].sin_cos();
                    qh[p * hd + j] = (qs[j] * c - qs[j + half] * s) * scale;
                    qh[p * hd + j + half] = (qs[j] * s + qs[j + half] * c) * scale;
                    kh[p * hd + j] = ks[j] * c - ks[j + half] * s;
                    kh[p * hd + j + half] = ks[j] * s + ks[j + half] * c;
                }
                for (d, &val) in vs.iter().enumerate() {
                    vt[d * n + p] = val;
                }
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
                Some(p) => p.run_rows(n, &soft),
                None => soft(0, n),
            }
            crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            for p in 0..n {
                out[p * inner + h * hd..p * inner + (h + 1) * hd]
                    .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// Flattened patches `[n, in·t·p·p]` on a `h`×`w` patch grid →
    /// `(merged [n/4, out_hidden], deepstack [k][n/4, out_hidden])`.
    pub fn forward(&self, patches: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n = h * w;
        let hd = self.hidden;
        let pool = self.pool.as_deref();
        let prof = std::env::var_os("CMF_SN_PROF").is_some();
        let t0 = std::time::Instant::now();
        let mut x = vec![0f32; n * hd];
        self.mm(&self.patch, patches, n, &mut x, pool);
        let pos = self.positions(h, w);
        par_rows(&mut x, hd, pool, |p, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v += self.patch_b[i] + pos[p * hd + i];
            }
        });
        let angles = self.rope_angles(h, w);

        let inner = self.heads * self.head_dim;
        let inter = self.blocks[0].fc1.rows();
        let mut xn = vec![0f32; n * hd];
        let mut qkv = vec![0f32; n * 3 * inner];
        let mut attn = vec![0f32; n * inner];
        let mut proj = vec![0f32; n * hd];
        let mut ff = vec![0f32; n * inter];
        let merge_dim = hd * self.merge * self.merge;
        let mut deep = Vec::new();
        let (mut t_mm, mut t_att) = (0f64, 0f64);
        // The device tower: the whole encoder stack resident when the
        // backend takes it (f32 matrices, no deepstack taps).
        let resident = self.device && self.deepstack_at.is_empty() && crate::gpu::enabled_here() && {
            let f32w = |p: &'_ Proj| -> Option<*const Vec<f32>> {
                match p {
                    Proj::F32 { w, .. } => Some(w as *const Vec<f32>),
                    Proj::Q(_) => None,
                }
            };
            let half = self.head_dim / 2;
            let mut cos = vec![0f32; n * self.head_dim];
            let mut sin = vec![0f32; n * self.head_dim];
            for p in 0..n {
                for j in 0..self.head_dim {
                    let (s, c) = angles[p * half + j % half].sin_cos();
                    cos[p * self.head_dim + j] = c;
                    sin[p * self.head_dim + j] = s;
                }
            }
            let layers: Option<Vec<crate::gpu::VitLayer>> = self
                .blocks
                .iter()
                .map(|b| {
                    // SAFETY: the pointers come from `b`, alive for this call.
                    let (q, pr, f1, f2) = unsafe {
                        (&*f32w(&b.qkv)?, &*f32w(&b.proj)?, &*f32w(&b.fc1)?, &*f32w(&b.fc2)?)
                    };
                    Some(crate::gpu::VitLayer {
                        ln1_w: &b.n1_w,
                        ln1_b: &b.n1_b,
                        qkv: q,
                        qkv_b: &b.qkv_b,
                        proj: pr,
                        proj_b: &b.proj_b,
                        ln2_w: &b.n2_w,
                        ln2_b: &b.n2_b,
                        fc1: f1,
                        fc1_b: &b.fc1_b,
                        fc2: f2,
                        fc2_b: &b.fc2_b,
                    })
                })
                .collect();
            match layers {
                Some(layers) => {
                    let args = crate::gpu::VitArgs {
                        key: self as *const Self as u64,
                        layers,
                        n,
                        d: hd,
                        heads: self.heads,
                        head_dim: self.head_dim,
                        inter,
                        eps: EPS,
                        gelu_tanh: true,
                        rope_halves: true,
                        cos: &cos,
                        sin: &sin,
                        prof,
                    };
                    crate::gpu::vit_encode(&args, &mut x)
                }
                None => false,
            }
        };
        for (li, blk) in self.blocks.iter().enumerate().filter(|_| !resident) {
            norm_rows(&x, &blk.n1_w, &blk.n1_b, &mut xn, hd, pool);
            let tm = std::time::Instant::now();
            self.mm(&blk.qkv, &xn, n, &mut qkv, pool);
            t_mm += tm.elapsed().as_secs_f64();
            par_rows(&mut qkv, 3 * inner, pool, |_, r| {
                for (v, &b) in r.iter_mut().zip(&blk.qkv_b) {
                    *v += b;
                }
            });
            let ta = std::time::Instant::now();
            self.attention(&qkv, n, &angles, &mut attn);
            t_att += ta.elapsed().as_secs_f64();
            let tm = std::time::Instant::now();
            self.mm(&blk.proj, &attn, n, &mut proj, pool);
            t_mm += tm.elapsed().as_secs_f64();
            add_rows(&mut x, &proj, &blk.proj_b, hd, pool);
            norm_rows(&x, &blk.n2_w, &blk.n2_b, &mut xn, hd, pool);
            let tm = std::time::Instant::now();
            self.mm(&blk.fc1, &xn, n, &mut ff, pool);
            t_mm += tm.elapsed().as_secs_f64();
            par_rows(&mut ff, inter, pool, |_, r| {
                for (v, &b) in r.iter_mut().zip(&blk.fc1_b) {
                    *v = gelu_tanh(*v + b);
                }
            });
            let tm = std::time::Instant::now();
            self.mm(&blk.fc2, &ff, n, &mut proj, pool);
            t_mm += tm.elapsed().as_secs_f64();
            add_rows(&mut x, &proj, &blk.fc2_b, hd, pool);
            if let Some(k) = self.deepstack_at.iter().position(|&d| d == li) {
                deep.push(self.deepstack[k].apply(&x, n, hd, merge_dim, pool));
            }
        }
        let out = self.merger.apply(&x, n, hd, merge_dim, pool);
        if prof {
            eprintln!(
                "qwen3vis: {n} patches, tower {:.3} s (gemm {:.3} s, attention {:.3} s)",
                t0.elapsed().as_secs_f64(),
                t_mm,
                t_att
            );
        }
        (out, deep)
    }

}

/// `f(row_index, row)` over the `width`-wide rows of `x`, across the pool.
fn par_rows(x: &mut [f32], width: usize, pool: Option<&Pool>, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = x.len() / width;
    let sp = SendPtr(x.as_mut_ptr());
    let run = |lo: usize, hi: usize| {
        for r in lo..hi {
            // SAFETY: workers own disjoint rows.
            f(r, unsafe { sp.row(r * width, width) });
        }
    };
    match pool {
        Some(p) => p.run_rows(rows, &run),
        None => run(0, rows),
    }
}

/// LayerNorm of every `width`-wide row of `x` into `out`.
fn norm_rows(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], width: usize, pool: Option<&Pool>) {
    par_rows(out, width, pool, |r, o| layer_norm(&x[r * width..(r + 1) * width], w, b, o));
}

/// `x += y + bias` row by row.
fn add_rows(x: &mut [f32], y: &[f32], bias: &[f32], width: usize, pool: Option<&Pool>) {
    par_rows(x, width, pool, |r, row| {
        for (i, v) in row.iter_mut().enumerate() {
            *v += y[r * width + i] + bias[i];
        }
    });
}

/// An RGB image in [0, 1], `[3, h, w]` → the flattened patches the tower
/// takes and its `(grid_h, grid_w)`.
///
/// The reference resizes to a multiple of `patch·merge`, normalizes to
/// [-1, 1] — Qwen3-VL uses mean and std 0.5, not CLIP's — and lays each
/// patch out as `[in, t, ph, pw]` with the two temporal slots holding
/// the SAME frame for a still image.
pub fn preprocess(
    rgb: &[f32],
    h: usize,
    w: usize,
    patch: usize,
    temporal: usize,
    merge: usize,
) -> (Vec<f32>, usize, usize) {
    let factor = patch * merge;
    let hb = ((h as f64 / factor as f64).round() as usize).max(1) * factor;
    let wb = ((w as f64 / factor as f64).round() as usize).max(1) * factor;
    // Bilinear resize, align_corners=false, as `F.interpolate` does.
    let mut img = vec![0f32; 3 * hb * wb];
    for c in 0..3 {
        for y in 0..hb {
            let sy = ((y as f64 + 0.5) * h as f64 / hb as f64 - 0.5).max(0.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = (sy - y0 as f64) as f32;
            for x in 0..wb {
                let sx = ((x as f64 + 0.5) * w as f64 / wb as f64 - 0.5).max(0.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = (sx - x0 as f64) as f32;
                let p = |yy: usize, xx: usize| rgb[(c * h + yy) * w + xx];
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                img[(c * hb + y) * wb + x] = (top * (1.0 - fy) + bot * fy - 0.5) / 0.5;
            }
        }
    }
    let (gh, gw) = (hb / patch, wb / patch);
    let per = 3 * temporal * patch * patch;
    let mut out = vec![0f32; gh * gw * per];
    // The patch ORDER is the 2x2 merge-block one, not row-major: the
    // reference's `permute(0, 3, 6, 4, 7, ...)` walks block row, block
    // column, then the two intra-block indices. The position table and
    // the rotation are built in the same order, and the merger consumes
    // four consecutive patches as one cell — row-major here would
    // scramble all three at once.
    let mut k = 0usize;
    for bby in 0..gh / merge {
        for bbx in 0..gw / merge {
            for iy in 0..merge {
                for ix in 0..merge {
                    let (by, bx) = (bby * merge + iy, bbx * merge + ix);
                    let dst = &mut out[k * per..(k + 1) * per];
                    k += 1;
                    let mut j = 0;
                    for c in 0..3 {
                        for _t in 0..temporal {
                            for y in 0..patch {
                                for x in 0..patch {
                                    dst[j] = img[(c * hb + by * patch + y) * wb + bx * patch + x];
                                    j += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (out, gh, gw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_gelus_are_not_the_same_function() {
        // Both agree to ~1e-3 in the middle and diverge in the tails;
        // the point is that the code picks deliberately.
        assert!((gelu_exact(0.0)).abs() < 1e-12);
        assert!((gelu_exact(1.0) - 0.841_344_8).abs() < 2e-7);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu_exact(1.0) - gelu_tanh(1.0)).abs() > 1e-5);
        assert!((gelu_exact(-3.0) - -0.004_049_5).abs() < 2e-6);
    }

    #[test]
    fn erf_is_accurate_across_the_range() {
        for (x, want) in [
            (0.0, 0.0),
            (0.25, 0.276_326_390_168_236_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
        ] {
            let got = erf(x);
            assert!((got - want).abs() < 2e-7, "erf({x}) = {got}, want {want}");
            assert!((erf(-x) + want).abs() < 2e-7, "erf is odd");
        }
    }

    #[test]
    fn preprocess_rounds_the_canvas_to_the_merge_factor() {
        let (h, w) = (70usize, 100usize);
        let rgb = vec![0.5f32; 3 * h * w];
        let (p, gh, gw) = preprocess(&rgb, h, w, 16, 2, 2);
        // 70 → 64 (round(70/32)=2 → 64), 100 → 96
        assert_eq!((gh, gw), (4, 6));
        assert_eq!(p.len(), gh * gw * 3 * 2 * 16 * 16);
        // A flat 0.5 image normalizes to exactly zero.
        assert!(p.iter().all(|&v| v.abs() < 1e-6));
    }
}
