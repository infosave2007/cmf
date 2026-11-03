//! Kandinsky S2V `KandinskyS2VTransformer3DModel`: the speech-to-video DiT,
//! read from a `kandinsky_s2v` container. This module is the exact host path
//! and the geometry every device path shares.
//!
//! The visual stream is one packed sequence `[memory | ref | noisy]`:
//! FramePack memory of the previous chunk at 4×, 2× and 1× temporal
//! downsampling (negative temporal RoPE positions), the reference frame
//! (pinned to temporal RoPE index 40), then the latent being denoised. Clean
//! rows are modulated at t = 0 and noisy rows at t, so each visual block's
//! modulation holds two rows and every row op selects one per token.
//!
//! Semantics reproduced (diffusers `transformer_kandinsky_s2v.py`):
//! - adaLN: `LayerNorm(x)·(1 + scale) + shift`, residual `x + gate·out`;
//! - attention: RMS-normalize q and k per head (ε 1e-5), rotate them when the
//!   block has RoPE (never in cross-attention), 1/√128 softmax;
//! - speech cross-attention: every 18th-ish block (`audio_inject_layers`),
//!   noisy rows only, frame i attends only to its own audio tokens, residual
//!   ungated, LayerNorm with learned weight and bias;
//! - GELU is the erf form, not the tanh approximation;
//! - the time embedding and every modulation stay fp32 in the reference, so
//!   they are computed once for the whole render here.

use crate::dit::{Proj, cmf_f32};
use crate::pool::Pool;
use crate::zimage::host_gemm;
use cortiq_core::CmfModel;
use std::sync::Arc;

pub const ARCH_NAME: &str = "kandinsky_s2v";
pub const HEAD_DIM: usize = 128;
pub const ROPE_PAIRS: usize = HEAD_DIM / 2;
const NORM_EPS: f64 = 1e-5;
/// Rows of the temporal RoPE table on each side of zero.
const ROPE_T_MAX: i32 = 128;

#[derive(Clone, Debug)]
pub struct S2vConfig {
    pub in_visual: usize,
    pub out_visual: usize,
    pub in_text: usize,
    pub in_text2: usize,
    pub in_audio: usize,
    pub time_dim: usize,
    pub patch: [usize; 3],
    pub frame_buckets: [usize; 3],
    pub ref_rope_index: i32,
    pub dim: usize,
    pub ff: usize,
    pub n_text: usize,
    pub n_audio: usize,
    pub n_visual: usize,
    pub axes: [usize; 3],
    pub audio_layers: Vec<usize>,
}

impl S2vConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |k: &str| -> Result<usize, String> {
            v[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("dit.config_json: no {k}"))
        };
        let tri = |k: &str| -> Result<[usize; 3], String> {
            let a = v[k].as_array().ok_or_else(|| format!("dit.config_json: no {k}"))?;
            if a.len() != 3 {
                return Err(format!("dit.config_json: {k} is not a triple"));
            }
            Ok([
                a[0].as_u64().unwrap_or(0) as usize,
                a[1].as_u64().unwrap_or(0) as usize,
                a[2].as_u64().unwrap_or(0) as usize,
            ])
        };
        let cfg = S2vConfig {
            in_visual: u("in_visual_dim")?,
            out_visual: u("out_visual_dim")?,
            in_text: u("in_text_dim")?,
            in_text2: u("in_text_dim2")?,
            in_audio: u("in_audio_dim")?,
            time_dim: u("time_dim")?,
            patch: tri("patch_size")?,
            frame_buckets: tri("frame_buckets")?,
            ref_rope_index: u("index_rope_ref_image")? as i32,
            dim: u("model_dim")?,
            ff: u("ff_dim")?,
            n_text: u("num_text_blocks")?,
            n_audio: u("num_audio_blocks")?,
            n_visual: u("num_visual_blocks")?,
            axes: tri("axes_dims")?,
            audio_layers: v["audio_inject_layers"]
                .as_array()
                .ok_or("dit.config_json: no audio_inject_layers")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect(),
        };
        if cfg.patch != [1, 2, 2] || cfg.axes.iter().sum::<usize>() != HEAD_DIM || cfg.dim % HEAD_DIM != 0 {
            return Err(format!("unsupported S2V geometry: {cfg:?}"));
        }
        Ok(cfg)
    }
    pub fn heads(&self) -> usize {
        self.dim / HEAD_DIM
    }
}

/// Token layout of one forward. Every count derives from the latent shape.
#[derive(Clone, Copy, Debug)]
pub struct S2vGeom {
    pub lt: usize,
    pub lh: usize,
    pub lw: usize,
    /// Rows of memory 4×, 2×, 1×, the reference and the noisy tail.
    pub n4: usize,
    pub n2: usize,
    pub n1: usize,
    pub nref: usize,
    pub nvis: usize,
}

impl S2vGeom {
    pub fn new(cfg: &S2vConfig, lt: usize, lh: usize, lw: usize) -> Self {
        let per = |c: usize| (lh / (2 * c)) * (lw / (2 * c));
        let b = cfg.frame_buckets;
        S2vGeom {
            lt,
            lh,
            lw,
            n4: (b[2] / 4) * per(4),
            n2: (b[1] / 2) * per(2),
            n1: b[0] * per(1),
            nref: per(1),
            nvis: lt * per(1),
        }
    }
    pub fn clean(&self) -> usize {
        self.n4 + self.n2 + self.n1 + self.nref
    }
    pub fn total(&self) -> usize {
        self.clean() + self.nvis
    }
    pub fn per_frame(&self) -> usize {
        self.nvis / self.lt
    }
}

/// `torch.exp(-log(max_period)·arange(d)/d)` in f32, in the same order.
fn freqs(d: usize) -> Vec<f32> {
    let ln = (10000f64).ln() as f32;
    (0..d).map(|j| (-ln * j as f32 / d as f32).exp()).collect()
}

/// RoPE of an explicit position list on one flat frequency ladder (the 1-D
/// text and audio tables).
pub fn rope_1d(pos: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let f = freqs(ROPE_PAIRS);
    let mut c = Vec::with_capacity(pos.len() * ROPE_PAIRS);
    let mut s = Vec::with_capacity(pos.len() * ROPE_PAIRS);
    for &p in pos {
        for &fr in &f {
            let a = p as f32 * fr;
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    (c, s)
}

/// The 3-D RoPE of the packed visual sequence `[4× | 2× | 1× | ref | noisy]`,
/// `[n][64]` cos and sin rows. The temporal table is offset by its half size
/// so memory can sit at negative positions.
pub fn rope_visual(cfg: &S2vConfig, g: &S2vGeom, sf: [f32; 3]) -> (Vec<f32>, Vec<f32>) {
    let ladders = [freqs(cfg.axes[0] / 2), freqs(cfg.axes[1] / 2), freqs(cfg.axes[2] / 2)];
    let (ph, pw) = (g.lh / 2, g.lw / 2);
    let b = cfg.frame_buckets;
    let bounds = [0i32, b[0] as i32, (b[0] + b[1]) as i32, (b[0] + b[1] + b[2]) as i32];
    // (temporal positions, row step, column step) in sequence order.
    let mut parts: Vec<(Vec<i32>, usize)> = Vec::new();
    let span = |lo: i32, hi: i32, st: i32| -> Vec<i32> { (lo..hi).step_by(st as usize).collect() };
    parts.push((span(-bounds[3], -bounds[2], 4), 4));
    parts.push((span(-bounds[2], -bounds[1], 2), 2));
    parts.push((span(-bounds[1], -bounds[0], 1), 1));
    parts.push((vec![cfg.ref_rope_index], 1));
    parts.push(((0..g.lt as i32).collect(), 1));
    let mut cos = Vec::with_capacity(g.total() * ROPE_PAIRS);
    let mut sin = Vec::with_capacity(g.total() * ROPE_PAIRS);
    for (ts, st) in parts {
        // The reference's memory buffers are cropped to the coarser patch, so
        // both counts floor exactly as `memory_rope_pos` does.
        let rows = (0..(ph - ph % st)).step_by(st);
        for &t in &ts {
            debug_assert!(t >= -ROPE_T_MAX && t < ROPE_T_MAX);
            for r in rows.clone() {
                for col in (0..(pw - pw % st)).step_by(st) {
                    for (ax, lad) in ladders.iter().enumerate() {
                        let p = [t, r as i32, col as i32][ax] as f32;
                        for &fr in lad {
                            let a = p * fr / sf[ax];
                            cos.push(a.cos());
                            sin.push(a.sin());
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(cos.len(), g.total() * ROPE_PAIRS);
    (cos, sin)
}

/// Group boundaries of the frame-aligned speech attention: the first latent
/// frame takes one token, the rest split the remainder evenly.
pub fn audio_groups(tokens: usize, frames: usize) -> Vec<(usize, usize)> {
    let rest = tokens - 1;
    let bounds: Vec<usize> = (0..frames).map(|i| rest * i / (frames - 1)).collect();
    let mut v = vec![(0usize, 1usize)];
    for w in bounds.windows(2) {
        v.push((1 + w[0], w[1] - w[0]));
    }
    v
}

/// The latent frame each speech token belongs to (audio RoPE positions).
pub fn audio_rope_pos(tokens: usize, frames: usize) -> Vec<i32> {
    let mut p = vec![0i32; tokens];
    for (fr, (st, len)) in audio_groups(tokens, frames).into_iter().enumerate() {
        for v in &mut p[st..st + len] {
            *v = fr as i32;
        }
    }
    p
}

pub(crate) struct Rows(pub(crate) *mut f32);
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}
impl Rows {
    /// SAFETY: callers take disjoint ranges.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn at(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

pub(crate) fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

/// `nn.GELU()` — the erf form (A&S 7.1.26, |ε| < 1.5e-7).
#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let e = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t + 0.254_829_6)
            * t
            * (-a * a).exp();
    0.5 * x * (1.0 + if z < 0.0 { -e } else { e })
}

#[inline]
pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// LayerNorm over one row (biased variance, torch's form), no affine.
#[inline]
pub(crate) fn layer_norm_row(x: &[f32], eps: f64, dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for (d, &v) in dst.iter_mut().zip(x) {
        *d = ((v as f64 - mean) * inv) as f32;
    }
}

fn rms_row(v: &mut [f32], w: &[f32], eps: f64) {
    let ss = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for (x, &g) in v.iter_mut().zip(w) {
        *x = (*x as f64 * inv) as f32 * g;
    }
}

fn softmax(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0f64;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r as f64;
    }
    let inv = (1.0 / den) as f32;
    for r in row.iter_mut() {
        *r *= inv;
    }
}

/// A linear layer read from the container: quantized weights stay mapped.
pub(crate) struct Lin {
    pub(crate) w: Proj,
    pub(crate) b: Option<Vec<f32>>,
}

impl Lin {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str) -> Result<Lin, String> {
        let name = format!("{p}.weight");
        let e = m.tensor(&name).ok_or_else(|| format!("missing tensor {name}"))?;
        let w = match e.dtype {
            cortiq_core::TensorDtype::F16 | cortiq_core::TensorDtype::Bf16 => {
                let cols = *e.shape.last().unwrap_or(&1);
                Proj::f32(cmf_f32(m, &name)?, cols)
            }
            _ => Proj::from_model(m, &name)?,
        };
        let bn = format!("{p}.bias");
        let b = if m.tensor(&bn).is_some() { Some(cmf_f32(m, &bn)?) } else { None };
        Ok(Lin { w, b })
    }
    pub(crate) fn rows(&self) -> usize {
        self.w.rows()
    }
    /// y[n, rows] = x[n, cols]·Wᵀ + b.
    pub(crate) fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let rows = self.w.rows();
        let mut y = vec![0f32; n * rows];
        host_lin(&self.w, x, n, &mut y, pool);
        if let Some(b) = &self.b {
            let sr = Rows(y.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for r in lo..hi {
                    // SAFETY: disjoint rows.
                    for (d, &v) in unsafe { sr.at(r * rows, rows) }.iter_mut().zip(b) {
                        *d += v;
                    }
                }
            });
        }
        y
    }
}

/// The host GEMM of the exact path; device offload is the chain's business.
pub(crate) fn host_lin(p: &Proj, x: &[f32], n: usize, y: &mut [f32], pool: Option<&Pool>) {
    let (rows, cols) = (p.rows(), p.cols());
    match p {
        Proj::F32 { w, .. } => host_gemm::gemm_nt(x, w, y, n, cols, rows, pool),
        Proj::Q(q) => {
            const CH: usize = 768;
            let mut wbuf = vec![0f32; CH.min(rows) * cols];
            let mut r0 = 0;
            while r0 < rows {
                let rc = CH.min(rows - r0);
                {
                    let wp = Rows(wbuf.as_mut_ptr());
                    pool_rows(pool, rc, &|lo, hi| {
                        for r in lo..hi {
                            // SAFETY: disjoint rows.
                            q.row_f32(r0 + r, unsafe { wp.at(r * cols, cols) });
                        }
                    });
                }
                host_gemm::gemm_nt_ld(x, &wbuf[..rc * cols], &mut y[r0..], rows, n, cols, rc, pool);
                r0 += rc;
            }
        }
    }
}

/// A LayerNorm's learned weight and bias.
pub(crate) struct Affine {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl Affine {
    pub(crate) fn load(m: &Arc<CmfModel>, p: &str) -> Result<Affine, String> {
        Ok(Affine { w: cmf_f32(m, &format!("{p}.weight"))?, b: cmf_f32(m, &format!("{p}.bias"))? })
    }
    pub(crate) fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let d = self.w.len();
        let mut y = vec![0f32; n * d];
        let sr = Rows(y.as_mut_ptr());
        pool_rows(pool, n, &|lo, hi| {
            for r in lo..hi {
                // SAFETY: disjoint rows.
                let row = unsafe { sr.at(r * d, d) };
                layer_norm_row(&x[r * d..(r + 1) * d], NORM_EPS, row);
                for ((v, &w), &b) in row.iter_mut().zip(&self.w).zip(&self.b) {
                    *v = *v * w + b;
                }
            }
        });
        y
    }
}

pub(crate) struct Attn {
    pub(crate) q: Lin,
    pub(crate) k: Lin,
    pub(crate) v: Lin,
    pub(crate) o: Lin,
    pub(crate) qn: Vec<f32>,
    pub(crate) kn: Vec<f32>,
}

impl Attn {
    fn load(m: &Arc<CmfModel>, p: &str) -> Result<Attn, String> {
        Ok(Attn {
            q: Lin::load(m, &format!("{p}.to_query"))?,
            k: Lin::load(m, &format!("{p}.to_key"))?,
            v: Lin::load(m, &format!("{p}.to_value"))?,
            o: Lin::load(m, &format!("{p}.out_layer"))?,
            qn: cmf_f32(m, &format!("{p}.query_norm.weight"))?,
            kn: cmf_f32(m, &format!("{p}.key_norm.weight"))?,
        })
    }

    /// `x` [n, h] queries, `ctx` [m, h] keys and values. `rope` rotates both
    /// (same table: self-attention only). `groups` makes the attention
    /// block-diagonal: per group `(q0, qn, k0, kn)`.
    fn forward(
        &self,
        x: &[f32],
        n: usize,
        ctx: &[f32],
        m: usize,
        rope: Option<(&[f32], &[f32])>,
        groups: Option<&[(usize, usize, usize, usize)]>,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let h = self.q.rows();
        let nh = h / HEAD_DIM;
        let mut q = self.q.apply(x, n, pool);
        let mut k = self.k.apply(ctx, m, pool);
        let v = self.v.apply(ctx, m, pool);
        let norm_rot = |buf: &mut [f32], rows: usize, w: &[f32]| {
            let sr = Rows(buf.as_mut_ptr());
            pool_rows(pool, rows, &|lo, hi| {
                for r in lo..hi {
                    for hh in 0..nh {
                        // SAFETY: disjoint rows.
                        let row = unsafe { sr.at((r * nh + hh) * HEAD_DIM, HEAD_DIM) };
                        rms_row(row, w, NORM_EPS);
                        if let Some((cos, sin)) = rope {
                            for j in 0..ROPE_PAIRS {
                                let (c, s) = (cos[r * ROPE_PAIRS + j], sin[r * ROPE_PAIRS + j]);
                                let (a, b) = (row[2 * j], row[2 * j + 1]);
                                row[2 * j] = a * c - b * s;
                                row[2 * j + 1] = a * s + b * c;
                            }
                        }
                    }
                }
            });
        };
        norm_rot(&mut q, n, &self.qn);
        norm_rot(&mut k, m, &self.kn);
        let whole = [(0usize, n, 0usize, m)];
        let gs: &[(usize, usize, usize, usize)] = groups.unwrap_or(&whole);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut att = vec![0f32; n * h];
        for &(q0, qn, k0, kn) in gs {
            let mut qh = vec![0f32; qn * HEAD_DIM];
            let mut kh = vec![0f32; kn * HEAD_DIM];
            let mut vt = vec![0f32; HEAD_DIM * kn];
            let mut sc = vec![0f32; qn * kn];
            let mut oh = vec![0f32; qn * HEAD_DIM];
            for hh in 0..nh {
                for r in 0..qn {
                    for d in 0..HEAD_DIM {
                        qh[r * HEAD_DIM + d] = q[((q0 + r) * nh + hh) * HEAD_DIM + d] * scale;
                    }
                }
                for r in 0..kn {
                    let base = ((k0 + r) * nh + hh) * HEAD_DIM;
                    kh[r * HEAD_DIM..(r + 1) * HEAD_DIM].copy_from_slice(&k[base..base + HEAD_DIM]);
                    for d in 0..HEAD_DIM {
                        vt[d * kn + r] = v[base + d];
                    }
                }
                host_gemm::gemm_nt(&qh, &kh, &mut sc, qn, HEAD_DIM, kn, pool);
                let sp = Rows(sc.as_mut_ptr());
                pool_rows(pool, qn, &|lo, hi| {
                    for r in lo..hi {
                        // SAFETY: disjoint rows.
                        softmax(unsafe { sp.at(r * kn, kn) });
                    }
                });
                host_gemm::gemm_nt(&sc, &vt, &mut oh, qn, kn, HEAD_DIM, pool);
                for r in 0..qn {
                    let o = ((q0 + r) * nh + hh) * HEAD_DIM;
                    att[o..o + HEAD_DIM].copy_from_slice(&oh[r * HEAD_DIM..(r + 1) * HEAD_DIM]);
                }
            }
        }
        self.o.apply(&att, n, pool)
    }
}

pub(crate) struct Ffn {
    pub(crate) w1: Lin,
    pub(crate) w2: Lin,
}

impl Ffn {
    fn load(m: &Arc<CmfModel>, p: &str) -> Result<Ffn, String> {
        Ok(Ffn {
            w1: Lin::load(m, &format!("{p}.in_layer"))?,
            w2: Lin::load(m, &format!("{p}.out_layer"))?,
        })
    }
    fn forward(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut hid = self.w1.apply(x, n, pool);
        let inner = self.w1.rows();
        let sr = Rows(hid.as_mut_ptr());
        pool_rows(pool, n, &|lo, hi| {
            for r in lo..hi {
                // SAFETY: disjoint rows.
                for v in unsafe { sr.at(r * inner, inner) } {
                    *v = gelu(*v);
                }
            }
        });
        self.w2.apply(&hid, n, pool)
    }
}

pub(crate) struct VisualBlock {
    pub(crate) sa: Attn,
    pub(crate) ca: Attn,
    pub(crate) v2a: Option<(Affine, Attn)>,
    pub(crate) ffn: Ffn,
}

pub(crate) struct TextBlock {
    pub(crate) attn: Attn,
    pub(crate) ffn: Ffn,
}

pub(crate) struct AudioBlock {
    pub(crate) sa_norm: Affine,
    pub(crate) attn: Attn,
    pub(crate) ff_norm: Affine,
    pub(crate) ffn: Ffn,
}

pub struct S2vDit {
    pub(crate) model: Arc<CmfModel>,
    pub cfg: S2vConfig,
    pub(crate) pool: Option<Arc<Pool>>,
    pub(crate) time_in: Lin,
    pub(crate) time_out: Lin,
    pub(crate) time_freqs: Vec<f32>,
    pub(crate) text_in: Lin,
    pub(crate) text_norm: Affine,
    pub(crate) pooled_in: Lin,
    pub(crate) pooled_norm: Affine,
    pub(crate) audio_in: Lin,
    pub(crate) audio_norm: Affine,
    pub(crate) emb1: Lin,
    pub(crate) emb_p1: Lin,
    pub(crate) emb_p2: Lin,
    pub(crate) emb_p4: Lin,
    pub(crate) mask: Vec<f32>,
    pub(crate) text: Vec<TextBlock>,
    pub(crate) audio: Vec<AudioBlock>,
    pub(crate) visual: Vec<VisualBlock>,
    pub(crate) out_w: Lin,
}

/// Inputs of one forward, in the reference's layouts: latents `[T, H, W, C]`.
pub struct S2vInput<'a> {
    pub x: &'a [f32],
    pub reference: &'a [f32],
    pub mem1: &'a [f32],
    pub mem2: &'a [f32],
    pub mem4: &'a [f32],
    pub t: f32,
    pub text_feats: &'a [f32],
    pub audio_feats: &'a [f32],
    pub pooled: &'a [f32],
    pub geom: S2vGeom,
    pub scale_factor: [f32; 3],
}

/// `[T', H', W', C]` patches → `[n, pt·ph·pw·C]` in the `(pt, ph, pw, c)` order
/// the embedders were trained on.
pub fn patchify(x: &[f32], c: usize, pt: usize, ps: usize, h: usize, w: usize) -> Vec<f32> {
    let frames = x.len() / (h * w * c);
    let (th, tw, tt) = (h / ps, w / ps, frames / pt);
    let dim = pt * ps * ps * c;
    let mut out = vec![0f32; tt * th * tw * dim];
    for it in 0..tt {
        for ih in 0..th {
            for iw in 0..tw {
                let row = (it * th + ih) * tw + iw;
                let mut o = row * dim;
                for dt in 0..pt {
                    for dh in 0..ps {
                        for dw in 0..ps {
                            let src = (((it * pt + dt) * h + ih * ps + dh) * w + iw * ps + dw) * c;
                            out[o..o + c].copy_from_slice(&x[src..src + c]);
                            o += c;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `[n, ph·pw·C]` head output `[C][ph][pw]` → `[T, H, W, C]`.
pub fn unpatchify_out(tok: &[f32], g: &S2vGeom, c: usize) -> Vec<f32> {
    let (ph, pw) = (g.lh / 2, g.lw / 2);
    let mut out = vec![0f32; g.lt * g.lh * g.lw * c];
    for t in 0..g.lt {
        for r in 0..ph {
            for col in 0..pw {
                let row = (t * ph + r) * pw + col;
                for dh in 0..2 {
                    for dw in 0..2 {
                        let dst = ((t * g.lh + r * 2 + dh) * g.lw + col * 2 + dw) * c;
                        for ch in 0..c {
                            out[dst + ch] = tok[row * 4 * c + ch * 4 + dh * 2 + dw];
                        }
                    }
                }
            }
        }
    }
    out
}

impl S2vDit {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<S2vDit, String> {
        let raw = model.tensor_bytes("dit.config_json").map_err(|e| e.to_string())?;
        let v: serde_json::Value = serde_json::from_slice(raw).map_err(|e| format!("dit.config_json: {e}"))?;
        let cfg = S2vConfig::from_json(&v)?;
        let m = model;
        let mut text = Vec::with_capacity(cfg.n_text);
        for i in 0..cfg.n_text {
            let p = format!("dit.text_transformer_blocks.{i}");
            text.push(TextBlock {
                attn: Attn::load(m, &format!("{p}.self_attention"))?,
                ffn: Ffn::load(m, &format!("{p}.feed_forward"))?,
            });
        }
        let mut audio = Vec::with_capacity(cfg.n_audio);
        for i in 0..cfg.n_audio {
            let p = format!("dit.audio_transformer_blocks.{i}");
            audio.push(AudioBlock {
                sa_norm: Affine::load(m, &format!("{p}.self_attention_norm"))?,
                attn: Attn::load(m, &format!("{p}.self_attention"))?,
                ff_norm: Affine::load(m, &format!("{p}.feed_forward_norm"))?,
                ffn: Ffn::load(m, &format!("{p}.feed_forward"))?,
            });
        }
        let mut visual = Vec::with_capacity(cfg.n_visual);
        for i in 0..cfg.n_visual {
            let p = format!("dit.visual_transformer_blocks.{i}");
            let v2a = if cfg.audio_layers.contains(&i) {
                Some((
                    Affine::load(m, &format!("{p}.cross_attention_norm_v2a"))?,
                    Attn::load(m, &format!("{p}.cross_attention_v2a"))?,
                ))
            } else {
                None
            };
            visual.push(VisualBlock {
                sa: Attn::load(m, &format!("{p}.self_attention"))?,
                ca: Attn::load(m, &format!("{p}.cross_attention"))?,
                v2a,
                ffn: Ffn::load(m, &format!("{p}.feed_forward"))?,
            });
        }
        Ok(S2vDit {
            model: m.clone(),
            pool: Pool::from_env(),
            time_in: Lin::load(m, "dit.time_embeddings.in_layer")?,
            time_out: Lin::load(m, "dit.time_embeddings.out_layer")?,
            time_freqs: freqs(cfg.dim / 2),
            text_in: Lin::load(m, "dit.text_embeddings.in_layer")?,
            text_norm: Affine::load(m, "dit.text_embeddings.norm")?,
            pooled_in: Lin::load(m, "dit.pooled_text_embeddings.in_layer")?,
            pooled_norm: Affine::load(m, "dit.pooled_text_embeddings.norm")?,
            audio_in: Lin::load(m, "dit.audio_embeddings.in_layer")?,
            audio_norm: Affine::load(m, "dit.audio_embeddings.norm")?,
            emb1: Lin::load(m, "dit.visual_embeddings.in_layer")?,
            emb_p1: Lin::load(m, "dit.visual_embeddings_proj1x.in_layer")?,
            emb_p2: Lin::load(m, "dit.visual_embeddings_proj2x.in_layer")?,
            emb_p4: Lin::load(m, "dit.visual_embeddings_proj4x.in_layer")?,
            mask: cmf_f32(m, "dit.mask_embeddings.weight")?,
            text,
            audio,
            visual,
            out_w: Lin::load(m, "dit.out_layer.out_layer")?,
            cfg,
        })
    }

    pub(crate) fn pool(&self) -> Option<&Pool> {
        self.pool.as_deref()
    }

    /// The linear part of a modulation: `Y[n, 9h] = silu(te)[n, time]·Wᵀ + b`.
    /// Read from the file per block and dropped, so 2.3 B parameters of them
    /// never sit in RAM.
    pub(crate) fn modulate(&self, name: &str, te_silu: &[f32], n: usize) -> Result<Vec<f32>, String> {
        let l = Lin::load(&self.model, name)?;
        Ok(l.apply(te_silu, n, self.pool()))
    }

    /// `modulate` over many layers at once, a thread per layer: each read is a
    /// bf16 widening and a small GEMM, so they scale and the pool would not help.
    pub(crate) fn modulate_all(&self, names: &[String], te_silu: &[f32], n: usize) -> Result<Vec<Vec<f32>>, String> {
        let threads = Pool::effective_threads().clamp(1, 16);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let out: Vec<std::sync::Mutex<Option<Result<Vec<f32>, String>>>> = names.iter().map(|_| std::sync::Mutex::new(None)).collect();
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= names.len() {
                            break;
                        }
                        let r = Lin::load(&self.model, &names[i]).map(|l| l.apply(te_silu, n, None));
                        *out[i].lock().unwrap() = Some(r);
                    }
                });
            }
        });
        out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
    }

    /// `time_embeddings(t) + pooled_text_embeddings(pooled)`.
    pub fn time_embed(&self, t: f32, pooled_emb: &[f32]) -> Vec<f32> {
        let d = self.cfg.dim / 2;
        let mut e = vec![0f32; self.cfg.dim];
        for j in 0..d {
            let a = t * self.time_freqs[j];
            e[j] = a.cos();
            e[d + j] = a.sin();
        }
        let mut h = self.time_in.apply(&e, 1, None);
        for v in &mut h {
            *v = silu(*v);
        }
        let mut out = self.time_out.apply(&h, 1, None);
        for (o, &p) in out.iter_mut().zip(pooled_emb) {
            *o += p;
        }
        out
    }

    pub fn pooled_embed(&self, pooled: &[f32]) -> Vec<f32> {
        let y = self.pooled_in.apply(pooled, 1, None);
        self.pooled_norm.apply(&y, 1, None)
    }

    pub fn embed_text(&self, feats: &[f32]) -> Vec<f32> {
        let n = feats.len() / self.cfg.in_text;
        let y = self.text_in.apply(feats, n, self.pool());
        self.text_norm.apply(&y, n, self.pool())
    }

    pub fn embed_audio(&self, feats: &[f32]) -> Vec<f32> {
        let n = feats.len() / self.cfg.in_audio;
        let y = self.audio_in.apply(feats, n, self.pool());
        self.audio_norm.apply(&y, n, self.pool())
    }

    /// The four speech blocks: unmodulated, so once per chunk.
    pub fn audio_blocks(&self, x: &mut Vec<f32>, frames: usize) {
        let h = self.cfg.dim;
        let n = x.len() / h;
        let pool = self.pool();
        let (cos, sin) = rope_1d(&audio_rope_pos(n, frames));
        for b in &self.audio {
            let xn = b.sa_norm.apply(x, n, pool);
            let o = b.attn.forward(&xn, n, &xn, n, Some((&cos, &sin)), None, pool);
            add_assign(x, &o);
            let xn = b.ff_norm.apply(x, n, pool);
            let o = b.ffn.forward(&xn, n, pool);
            add_assign(x, &o);
        }
    }

    /// One text block at the modulation row `m` (`[6h]`).
    fn text_block(&self, b: &TextBlock, x: &mut [f32], m: &[f32], rope: (&[f32], &[f32])) {
        let h = self.cfg.dim;
        let n = x.len() / h;
        let pool = self.pool();
        let xn = modulated_norm(x, n, h, &[&m[..3 * h]], &|_| 0, pool);
        let o = b.attn.forward(&xn, n, &xn, n, Some(rope), None, pool);
        gated_add(x, &o, n, h, &[&m[2 * h..3 * h]], &|_| 0, pool);
        let xn = modulated_norm(x, n, h, &[&m[3 * h..6 * h]], &|_| 0, pool);
        let o = b.ffn.forward(&xn, n, pool);
        gated_add(x, &o, n, h, &[&m[5 * h..6 * h]], &|_| 0, pool);
    }

    /// `visual_embeddings_*` of every part, the mask embeddings added.
    pub fn embed_visual(&self, i: &S2vInput<'_>) -> Vec<f32> {
        let (c, g) = (self.cfg.in_visual, &i.geom);
        let h = self.cfg.dim;
        let pool = self.pool();
        let part = |lin: &Lin, data: &[f32], pt: usize, ps: usize, mask: usize| -> Vec<f32> {
            let (hh, ww) = (g.lh - g.lh % ps, g.lw - g.lw % ps);
            let tok = patchify(data, c, pt, ps, hh, ww);
            let n = tok.len() / lin.w.cols();
            let mut y = lin.apply(&tok, n, pool);
            for row in y.chunks_exact_mut(h) {
                for (v, &m) in row.iter_mut().zip(&self.mask[mask * h..(mask + 1) * h]) {
                    *v += m;
                }
            }
            y
        };
        let mut out = Vec::with_capacity(g.total() * h);
        out.extend(part(&self.emb_p4, i.mem4, 4, 8, 2));
        out.extend(part(&self.emb_p2, i.mem2, 2, 4, 2));
        out.extend(part(&self.emb_p1, i.mem1, 1, 2, 2));
        out.extend(part(&self.emb1, i.reference, 1, 2, 1));
        out.extend(part(&self.emb1, i.x, 1, 2, 0));
        out
    }

    /// The clean rows of the visual stream, `[memory | ref]` embedded.
    pub fn embed_clean(&self, reference: &[f32], memory: &[Vec<f32>; 3], g: &S2vGeom) -> Vec<f32> {
        let (c, h) = (self.cfg.in_visual, self.cfg.dim);
        let pool = self.pool();
        let part = |lin: &Lin, data: &[f32], pt: usize, ps: usize, mask: usize| -> Vec<f32> {
            let (hh, ww) = (g.lh - g.lh % ps, g.lw - g.lw % ps);
            let tok = patchify(data, c, pt, ps, hh, ww);
            let n = tok.len() / lin.w.cols();
            let mut y = lin.apply(&tok, n, pool);
            for row in y.chunks_exact_mut(h) {
                for (v, &m) in row.iter_mut().zip(&self.mask[mask * h..(mask + 1) * h]) {
                    *v += m;
                }
            }
            y
        };
        let mut out = Vec::with_capacity(g.clean() * h);
        out.extend(part(&self.emb_p4, &memory[2], 4, 8, 2));
        out.extend(part(&self.emb_p2, &memory[1], 2, 4, 2));
        out.extend(part(&self.emb_p1, &memory[0], 1, 2, 2));
        out.extend(part(&self.emb1, reference, 1, 2, 1));
        out
    }

    /// One whole forward on the host. Returns `[T, H, W, out_visual]`.
    /// `tap` receives each block's output under the oracle's name.
    pub fn forward(
        &self,
        i: &S2vInput<'_>,
        mut tap: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<Vec<f32>, String> {
        let (h, g) = (self.cfg.dim, &i.geom);
        let pool = self.pool();
        let pe = self.pooled_embed(i.pooled);
        let te = self.time_embed(i.t, &pe);
        let te0 = self.time_embed(0.0, &pe);
        let silu_all = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| silu(x)).collect() };
        let (s, s0) = (silu_all(&te), silu_all(&te0));

        let mut text = self.embed_text(i.text_feats);
        let (tc, ts) = rope_1d(&(0..(text.len() / h) as i32).collect::<Vec<_>>());
        for (bi, b) in self.text.iter().enumerate() {
            let name = format!("dit.text_transformer_blocks.{bi}.text_modulation.out_layer");
            let m = self.modulate(&name, &s, 1)?;
            self.text_block(b, &mut text, &m, (&tc, &ts));
            if let Some(t) = tap.as_mut() {
                t(&format!("text_transformer_blocks.{bi}"), &text);
            }
        }
        let mut audio = self.embed_audio(i.audio_feats);
        {
            let n = audio.len() / h;
            let (ac, as_) = rope_1d(&audio_rope_pos(n, g.lt));
            for (bi, b) in self.audio.iter().enumerate() {
                let xn = b.sa_norm.apply(&audio, n, pool);
                let o = b.attn.forward(&xn, n, &xn, n, Some((&ac, &as_)), None, pool);
                add_assign(&mut audio, &o);
                let xn = b.ff_norm.apply(&audio, n, pool);
                let o = b.ffn.forward(&xn, n, pool);
                add_assign(&mut audio, &o);
                if let Some(t) = tap.as_mut() {
                    t(&format!("audio_transformer_blocks.{bi}"), &audio);
                }
            }
        }
        let mut x = self.embed_visual(i);
        let (vc, vs) = rope_visual(&self.cfg, g, i.scale_factor);
        let n = g.total();
        let clean = g.clean();
        let sel = move |r: usize| usize::from(r >= clean);
        let na = audio.len() / h;
        let nt = text.len() / h;
        let groups: Vec<(usize, usize, usize, usize)> = audio_groups(na, g.lt)
            .into_iter()
            .enumerate()
            .map(|(f, (k0, kn))| (f * g.per_frame(), g.per_frame(), k0, kn))
            .collect();
        let mut st = [s0.clone(), s.clone()].concat();
        for (bi, b) in self.visual.iter().enumerate() {
            let name = format!("dit.visual_transformer_blocks.{bi}.visual_modulation.out_layer");
            let m = self.modulate(&name, &st, 2)?;
            let row = |r: usize, a: usize, e: usize| &m[r * 9 * h + a * h..r * 9 * h + e * h];
            // self-attention
            let xn = modulated_norm(&x, n, h, &[row(0, 0, 3), row(1, 0, 3)], &sel, pool);
            let o = b.sa.forward(&xn, n, &xn, n, Some((&vc, &vs)), None, pool);
            gated_add(&mut x, &o, n, h, &[row(0, 2, 3), row(1, 2, 3)], &sel, pool);
            // prompt cross-attention
            let xn = modulated_norm(&x, n, h, &[row(0, 3, 6), row(1, 3, 6)], &sel, pool);
            let o = b.ca.forward(&xn, n, &text, nt, None, None, pool);
            gated_add(&mut x, &o, n, h, &[row(0, 5, 6), row(1, 5, 6)], &sel, pool);
            // speech cross-attention
            if let Some((norm, att)) = &b.v2a {
                let tail = &x[clean * h..];
                let xn = norm.apply(tail, g.nvis, pool);
                let o = att.forward(&xn, g.nvis, &audio, na, None, Some(&groups), pool);
                add_assign(&mut x[clean * h..], &o);
            }
            // feed-forward
            let xn = modulated_norm(&x, n, h, &[row(0, 6, 9), row(1, 6, 9)], &sel, pool);
            let o = b.ffn.forward(&xn, n, pool);
            gated_add(&mut x, &o, n, h, &[row(0, 8, 9), row(1, 8, 9)], &sel, pool);
            if let Some(t) = tap.as_mut() {
                t(&format!("visual_transformer_blocks.{bi}"), &x);
            }
        }
        st.clear();
        // Final layer: noisy rows, the t row only.
        let m = self.modulate("dit.out_layer.modulation.out_layer", &s, 1)?;
        let tail = &x[clean * h..];
        let sc: &[&[f32]] = &[&m[..2 * h]];
        let xn = modulated_norm_out(tail, g.nvis, h, sc, pool);
        let y = self.out_w.apply(&xn, g.nvis, pool);
        Ok(unpatchify_out(&y, g, self.cfg.out_visual))
    }
}

fn add_assign(x: &mut [f32], y: &[f32]) {
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// `LayerNorm(x)·(1 + scale) + shift` where each `chunks[k]` is `[shift, scale, …]`.
fn modulated_norm(
    x: &[f32],
    n: usize,
    h: usize,
    chunks: &[&[f32]],
    sel: &(dyn Fn(usize) -> usize + Sync),
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut y = vec![0f32; n * h];
    let sr = Rows(y.as_mut_ptr());
    pool_rows(pool, n, &|lo, hi| {
        for r in lo..hi {
            // SAFETY: disjoint rows.
            let row = unsafe { sr.at(r * h, h) };
            layer_norm_row(&x[r * h..(r + 1) * h], NORM_EPS, row);
            let c = chunks[sel(r)];
            let (shift, scale) = (&c[..h], &c[h..2 * h]);
            for ((v, &sc), &sh) in row.iter_mut().zip(scale).zip(shift) {
                *v = *v * (sc + 1.0) + sh;
            }
        }
    });
    y
}

/// The out layer's chunk order is `[shift, scale]` too.
fn modulated_norm_out(x: &[f32], n: usize, h: usize, chunks: &[&[f32]], pool: Option<&Pool>) -> Vec<f32> {
    modulated_norm(x, n, h, chunks, &|_| 0, pool)
}

fn gated_add(
    x: &mut [f32],
    o: &[f32],
    n: usize,
    h: usize,
    gates: &[&[f32]],
    sel: &(dyn Fn(usize) -> usize + Sync),
    pool: Option<&Pool>,
) {
    let sr = Rows(x.as_mut_ptr());
    pool_rows(pool, n, &|lo, hi| {
        for r in lo..hi {
            // SAFETY: disjoint rows.
            let row = unsafe { sr.at(r * h, h) };
            let g = gates[sel(r)];
            for ((v, &gg), &b) in row.iter_mut().zip(g).zip(&o[r * h..(r + 1) * h]) {
                *v += gg * b;
            }
        }
    });
}

/// Rows that decide the size of a device program (exposed for the planners).
pub fn tokens_for(cfg: &S2vConfig, lt: usize, lh: usize, lw: usize) -> usize {
    S2vGeom::new(cfg, lt, lh, lw).total()
}
