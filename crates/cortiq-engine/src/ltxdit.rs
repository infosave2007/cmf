//! LTX-2.5 `AVTransformer3DModel` — the audio-video diffusion transformer,
//! read straight from an `ltx-2.5-av` CMF container.
//!
//! One forward is one denoising step for both streams at once: 48 blocks
//! that each run video self-attention, video↔prompt cross-attention,
//! audio self-attention, audio↔prompt cross-attention and the two
//! directions of audio↔video cross-attention, every one of them modulated
//! by adaLN values that come from the timestep — per token, not per sample.
//!
//! What the reference does and this reproduces:
//!
//! * **adaLN-single**: a sinusoidal timestep embedding (256) → SiLU MLP →
//!   one `[9·dim]` vector per distinct timestep, added to the block's own
//!   `scale_shift_table`. Rows 0..3 modulate self-attention, 3..6 the
//!   feed-forward, 6..9 the prompt cross-attention.
//! * **ada-zero**: `rms_norm(x) · (1 + scale) + shift`, no learned weight.
//! * **post-SA**: `x + y·gate`, then a second `rms_norm` whose output is
//!   what cross-attention reads — the block never normalizes twice.
//! * **Split RoPE** over three video axes (frame, row, column) and one
//!   audio axis, evaluated at the *middle* of each patch's `[start, end)`
//!   bounds, with the frequency ladder built in f64 (the checkpoint's
//!   `frequencies_precision: float64`).
//! * **Gated attention**: `2·sigmoid(to_gate_logits(x))`, per head.
//! * **q/k RMS-norm across the whole inner dimension**, not per head.
//! * The A↔V pair reads the *pre-fusion* state of both streams, so the
//!   order the two directions run in cannot bias the result.
//!
//! Gated against reference forward-hook dumps by `cortiq ltx-dit`.

use crate::dit::{Proj, cmf_f32};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

const EPS: f64 = 1e-6;

// ---------------------------------------------------------------- helpers

/// Rows of `n` items split across pool workers (serial without a pool).
pub(crate) fn rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

/// Row handout for pool workers over one flat buffer.
pub(crate) struct Shared(pub(crate) *mut f32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    /// SAFETY: callers take disjoint `[off, off+len)` ranges.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn at(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// RMS normalization with no learned weight (`ada_zero`, `post_sa`).
pub(crate) fn rms_plain(x: &[f32], dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + EPS).sqrt();
    for (d, &v) in dst.iter_mut().zip(x) {
        *d = (v as f64 * inv) as f32;
    }
}

/// RMS normalization with a learned weight (q/k-norm).
fn rms_w(x: &mut [f32], w: &[f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + EPS).sqrt();
    for (v, &g) in x.iter_mut().zip(w) {
        *v = (*v as f64 * inv) as f32 * g;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `gelu(x, approximate="tanh")`, the feed-forward's projection activation.
///
/// In f32 and, where it matters, across the pool. A step puts half a billion
/// values through this — 672 tokens × 16384 wide × 48 blocks — so computing
/// it in f64 on one thread was worth several seconds of every step on its
/// own. The f32 form agrees with the f64 one to ~1e-7 relative, which is far
/// inside what the 4-bit weights around it carry.
#[inline]
pub(crate) fn gelu_tanh(v: f32) -> f32 {
    const K: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * v * (1.0 + (K * (v + 0.044715 * v * v * v)).tanh())
}

/// The same activation over a whole buffer, split across the pool.
pub(crate) fn gelu_tanh_rows(x: &mut [f32], pool: Option<&Pool>) {
    let dst = Shared(x.as_mut_ptr());
    let n = x.len();
    let grain = 4096usize;
    let chunks = n.div_ceil(grain);
    rows(pool, chunks, &|s, e| {
        let (lo, hi) = (s * grain, (e * grain).min(n));
        let r = unsafe { dst.at(lo, hi - lo) };
        for v in r.iter_mut() {
            *v = gelu_tanh(*v);
        }
    });
}

/// One attention row, normalized. A step softmaxes on the order of a
/// billion scores — forty-eight blocks × thirty-two heads × every query
/// against every key — so the scalar `exp()` here was not a detail: it was
/// the arithmetic. The NEON path evaluates four at a time.
pub(crate) fn softmax(row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::attention::softmax_row(row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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
}

/// LayerNorm with no affine — the output head's only normalization.
fn layer_norm(x: &[f32], dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + EPS).sqrt();
    for (d, &v) in dst.iter_mut().zip(x) {
        *d = ((v as f64 - mean) * inv) as f32;
    }
}

// ---------------------------------------------------------------- linear

/// `y = x·Wᵀ + b`, the weight read in place when the container quantized it.
pub(crate) struct Lin {
    w: Proj,
    b: Option<Vec<f32>>,
}

impl Lin {
    pub(crate) fn load(model: &Arc<CmfModel>, name: &str, bias: bool) -> Result<Lin, String> {
        let w = Proj::from_model(model, &format!("{name}.weight"))?;
        let b = if bias {
            Some(cmf_f32(model, &format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Lin { w, b })
    }

    /// The container-mapped q4tp weight behind this projection, when there
    /// is one: (model, tensor index, rows, cols).
    #[cfg(target_os = "macos")]
    fn mapped(&self) -> Option<(&Arc<CmfModel>, usize, usize, usize)> {
        let (model, idx) = self.w.q4tp_mapped()?;
        Some((model, idx, self.w.rows(), self.w.cols()))
    }

    /// The bias half of `apply`, for callers that got the product elsewhere.
    pub(crate) fn add_bias(&self, out: &mut [f32], n: usize, pool: Option<&Pool>) {
        let Some(b) = &self.b else { return };
        let m = self.w.rows();
        let dst = Shared(out.as_mut_ptr());
        rows(pool, n, &|s, e| {
            let r = unsafe { dst.at(s * m, (e - s) * m) };
            for row in r.chunks_exact_mut(m) {
                for (v, &bb) in row.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        });
    }

    pub(crate) fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let m = self.w.rows();
        let cols = self.w.cols();
        let t_alloc = std::time::Instant::now();
        let mut out = vec![0f32; n * m];
        attn_prof::ALLOC.fetch_add(
            t_alloc.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_mm = std::time::Instant::now();
        // A GPU binding is capped near 2 GiB, and the prompt encoder's
        // aggregate projection reads 188160 numbers per token — 1024 of them
        // at once is past the cap. Chunk the batch so no single dispatch
        // binds more than a quarter of a gigabyte of activations.
        let per_row = cols * 4;
        let chunk = (0x1000_0000usize / per_row.max(1)).max(1);
        let mut done = 0usize;
        while done < n {
            let take = chunk.min(n - done);
            self.w.matmat(
                &x[done * cols..(done + take) * cols],
                take,
                &mut out[done * m..(done + take) * m],
                pool,
            );
            done += take;
        }
        attn_prof::MATMAT.fetch_add(
            t_mm.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(b) = &self.b {
            let dst = Shared(out.as_mut_ptr());
            rows(pool, n, &|s, e| {
                let r = unsafe { dst.at(s * m, (e - s) * m) };
                for row in r.chunks_exact_mut(m) {
                    for (v, &bb) in row.iter_mut().zip(b) {
                        *v += bb;
                    }
                }
            });
        }
        out
    }
}

// ------------------------------------------------------------------ rope

/// Split-RoPE tables: `cos`/`sin` laid out as `[tokens, heads·dh/2]`.
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    heads: usize,
    half: usize,
}

impl Rope {
    /// `positions[t][d]` are patch midpoints and `max_pos[d]` the axis
    /// extent they are divided by. `dim` is the *inner* dimension
    /// (heads·dh) the frequency ladder is sized from.
    pub fn build(positions: &[Vec<f64>], max_pos: &[f64], dim: usize, heads: usize, theta: f64) -> Rope {
        let ndim = max_pos.len();
        let count = dim / (2 * ndim);
        // indices = theta^linspace(0, 1, count) · π/2, in f64
        let idx: Vec<f64> = (0..count)
            .map(|j| {
                let e = if count > 1 { j as f64 / (count - 1) as f64 } else { 0.0 };
                theta.powf(e) * std::f64::consts::PI / 2.0
            })
            .collect();
        let n = positions.len();
        let half = dim / 2;
        let pad = half - count * ndim;
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for (t, p) in positions.iter().enumerate() {
            let base = t * half;
            for i in 0..pad {
                cos[base + i] = 1.0;
            }
            for (j, &ind) in idx.iter().enumerate() {
                for (d, &mp) in max_pos.iter().enumerate() {
                    let f = ind * (p[d] / mp * 2.0 - 1.0);
                    let o = base + pad + j * ndim + d;
                    cos[o] = f.cos() as f32;
                    sin[o] = f.sin() as f32;
                }
            }
        }
        Rope { cos, sin, heads, half: half / heads }
    }

    /// In-place split rotation of one token's `[heads·dh]` projection.
    fn apply_row(&self, t: usize, row: &mut [f32]) {
        let dh = self.half * 2;
        let stride = self.heads * self.half;
        for h in 0..self.heads {
            let off = t * stride + h * self.half;
            let (c, s) = (&self.cos[off..off + self.half], &self.sin[off..off + self.half]);
            let v = &mut row[h * dh..(h + 1) * dh];
            for i in 0..self.half {
                let (a, b) = (v[i], v[i + self.half]);
                v[i] = a * c[i] - b * s[i];
                v[i + self.half] = b * c[i] + a * s[i];
            }
        }
    }
}


/// Sub-phase microseconds inside attention, summed across every call in a
/// step. `CMF_LTX_PROF=1` prints them: the phase timers above say *which*
/// attention is slow, these say *what part of it* is.
pub(crate) mod attn_prof {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static PROJ: AtomicU64 = AtomicU64::new(0);
    pub static NORM: AtomicU64 = AtomicU64::new(0);
    pub static GATHER: AtomicU64 = AtomicU64::new(0);
    pub static SCORE: AtomicU64 = AtomicU64::new(0);
    pub static SOFT: AtomicU64 = AtomicU64::new(0);
    pub static VALUE: AtomicU64 = AtomicU64::new(0);
    pub static OUT: AtomicU64 = AtomicU64::new(0);
    pub static ALLOC: AtomicU64 = AtomicU64::new(0);
    pub static MATMAT: AtomicU64 = AtomicU64::new(0);

    pub fn add(c: &AtomicU64, t: std::time::Instant) -> std::time::Instant {
        c.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
        std::time::Instant::now()
    }

    pub fn report() -> String {
        let s = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / 1e6;
        format!(
            "proj {:.2}s  qk-norm+rope {:.2}s  gather {:.2}s  scores {:.2}s  softmax {:.2}s  values {:.2}s  out {:.2}s  [linear: alloc {:.2}s  matmat {:.2}s]",
            s(&PROJ), s(&NORM), s(&GATHER), s(&SCORE), s(&SOFT), s(&VALUE), s(&OUT),
            s(&ALLOC), s(&MATMAT)
        )
    }
}

// ------------------------------------------------------------- attention

pub(crate) struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Option<Lin>,
    heads: usize,
    dh: usize,
}

impl Attn {
    pub(crate) fn load(model: &Arc<CmfModel>, p: &str, heads: usize, dh: usize) -> Result<Attn, String> {
        Ok(Attn {
            q: Lin::load(model, &format!("{p}.to_q"), true)?,
            k: Lin::load(model, &format!("{p}.to_k"), true)?,
            v: Lin::load(model, &format!("{p}.to_v"), true)?,
            o: Lin::load(model, &format!("{p}.to_out.0"), true)?,
            q_norm: cmf_f32(model, &format!("{p}.q_norm.weight"))?,
            k_norm: cmf_f32(model, &format!("{p}.k_norm.weight"))?,
            gate: match model.tensor(&format!("{p}.to_gate_logits.weight")) {
                Some(_) => Some(Lin::load(model, &format!("{p}.to_gate_logits"), true)?),
                None => None,
            },
            heads,
            dh,
        })
    }

    /// The three projections in one submission, when the platform has a
    /// batched entry and all three read the same activation. Returns `None`
    /// when they do not — cross-attention's query comes from the tokens and
    /// its keys from the prompt — and the caller falls back to three calls.
    #[cfg(target_os = "macos")]
    fn fused_qkv(
        &self,
        x: &[f32],
        n: usize,
        ctx: &[f32],
        m: usize,
        pool: Option<&Pool>,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if !std::ptr::eq(x.as_ptr(), ctx.as_ptr()) || n != m {
            return None;
        }
        if !crate::gpu::enabled_here() || crate::gpu::mm_killed() {
            return None;
        }
        let (qw, kw, vw) = (self.q.mapped()?, self.k.mapped()?, self.v.mapped()?);
        if !Arc::ptr_eq(qw.0, kw.0) || !Arc::ptr_eq(qw.0, vw.0) {
            return None;
        }
        let jobs = [
            crate::gpu_metal::MmJob { idx: qw.1, rows: qw.2, cols: qw.3 },
            crate::gpu_metal::MmJob { idx: kw.1, rows: kw.2, cols: kw.3 },
            crate::gpu_metal::MmJob { idx: vw.1, rows: vw.2, cols: vw.3 },
        ];
        if n * jobs[0].rows * jobs[0].cols < 128_000_000 || n < 32 {
            return None;
        }
        let mut oq = vec![0f32; n * jobs[0].rows];
        let mut ok = vec![0f32; n * jobs[1].rows];
        let mut ov = vec![0f32; n * jobs[2].rows];
        let done = {
            let mut outs: [&mut [f32]; 3] = [&mut oq, &mut ok, &mut ov];
            crate::gpu_metal::q4tp_matmat_many(qw.0, &jobs, x, n, &mut outs)
        };
        if !done {
            return None;
        }
        self.q.add_bias(&mut oq, n, pool);
        self.k.add_bias(&mut ok, n, pool);
        self.v.add_bias(&mut ov, n, pool);
        Some((oq, ok, ov))
    }

    /// The key and value projections in one submission — they always read
    /// the same buffer, whatever the query does.
    #[cfg(target_os = "macos")]
    fn fused_kv(&self, ctx: &[f32], m: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if !crate::gpu::enabled_here() || crate::gpu::mm_killed() {
            return None;
        }
        let (kw, vw) = (self.k.mapped()?, self.v.mapped()?);
        if !Arc::ptr_eq(kw.0, vw.0) {
            return None;
        }
        let jobs = [
            crate::gpu_metal::MmJob { idx: kw.1, rows: kw.2, cols: kw.3 },
            crate::gpu_metal::MmJob { idx: vw.1, rows: vw.2, cols: vw.3 },
        ];
        if m < 32 || m * jobs[0].rows * jobs[0].cols < 128_000_000 {
            return None;
        }
        let mut ok = vec![0f32; m * jobs[0].rows];
        let mut ov = vec![0f32; m * jobs[1].rows];
        let done = {
            let mut outs: [&mut [f32]; 2] = [&mut ok, &mut ov];
            crate::gpu_metal::q4tp_matmat_many(kw.0, &jobs, ctx, m, &mut outs)
        };
        if !done {
            return None;
        }
        self.k.add_bias(&mut ok, m, None);
        self.v.add_bias(&mut ov, m, None);
        Some((ok, ov))
    }

    #[cfg(not(target_os = "macos"))]
    fn fused_kv(&self, _ctx: &[f32], _m: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    #[cfg(not(target_os = "macos"))]
    fn fused_qkv(
        &self,
        _x: &[f32],
        _n: usize,
        _ctx: &[f32],
        _m: usize,
        _pool: Option<&Pool>,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        None
    }

    /// `x` is `[n, query_dim]`, `ctx` is `[m, context_dim]` (self-attention
    /// passes the same buffer twice). `mask` is an additive per-key bias.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        n: usize,
        ctx: &[f32],
        m: usize,
        pe_q: Option<&Rope>,
        pe_k: Option<&Rope>,
        mask: Option<&[f32]>,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let inner = self.heads * self.dh;
        let prof = std::env::var("CMF_LTX_PROF").is_ok();
        let mut t = std::time::Instant::now();
        // q, k and v do not depend on each other. When they also read the
        // same buffer — self-attention, and the A↔V pair on the context
        // side — they go to the device as one command buffer instead of
        // three, which is three times less of the ~1.3 ms a completion
        // costs whatever it contains.
        let (mut q, mut k, v) = match self.fused_qkv(x, n, ctx, m, pool) {
            Some(t) => t,
            // Cross-attention's query reads the tokens and its keys read the
            // prompt, so those two cannot share a submission — but the keys
            // and the values still can.
            None => match self.fused_kv(ctx, m) {
                Some((k, v)) => (self.q.apply(x, n, pool), k, v),
                None => (
                    self.q.apply(x, n, pool),
                    self.k.apply(ctx, m, pool),
                    self.v.apply(ctx, m, pool),
                ),
            },
        };
        if prof {
            t = attn_prof::add(&attn_prof::PROJ, t);
        }

        let qn = Shared(q.as_mut_ptr());
        rows(pool, n, &|s, e| {
            let r = unsafe { qn.at(s * inner, (e - s) * inner) };
            for (i, row) in r.chunks_exact_mut(inner).enumerate() {
                rms_w(row, &self.q_norm);
                if let Some(pe) = pe_q {
                    pe.apply_row(s + i, row);
                }
            }
        });
        let kn = Shared(k.as_mut_ptr());
        rows(pool, m, &|s, e| {
            let r = unsafe { kn.at(s * inner, (e - s) * inner) };
            for (i, row) in r.chunks_exact_mut(inner).enumerate() {
                rms_w(row, &self.k_norm);
                if let Some(pe) = pe_k {
                    pe.apply_row(s + i, row);
                }
            }
        });

        // Per head, both halves of attention are GEMMs: scores are
        // q·kᵀ and the value product is p·v. Gathering each head into a
        // contiguous `[tokens, dh]` block costs one copy and buys the
        // engine's blocked/BLAS/GPU kernels instead of a scalar loop —
        // this is most of a step's arithmetic.
        if prof {
            t = attn_prof::add(&attn_prof::NORM, t);
        }
        let mut out = vec![0f32; n * inner];
        let scale = 1.0 / (self.dh as f32).sqrt();
        let dh = self.dh;
        let mut qh = vec![0f32; n * dh];
        let mut kh = vec![0f32; m * dh];
        let mut vh = vec![0f32; m * dh];
        let mut sc = vec![0f32; n * m];
        let mut oh = vec![0f32; n * dh];
        for h in 0..self.heads {
            for i in 0..n {
                qh[i * dh..(i + 1) * dh].copy_from_slice(&q[i * inner + h * dh..][..dh]);
            }
            for j in 0..m {
                kh[j * dh..(j + 1) * dh].copy_from_slice(&k[j * inner + h * dh..][..dh]);
                vh[j * dh..(j + 1) * dh].copy_from_slice(&v[j * inner + h * dh..][..dh]);
            }
            if prof {
                t = attn_prof::add(&attn_prof::GATHER, t);
            }
            crate::fcd_ops::gemm_nt(&qh, &kh, &mut sc, n, dh, m, pool);
            if prof {
                t = attn_prof::add(&attn_prof::SCORE, t);
            }
            let sp = Shared(sc.as_mut_ptr());
            rows(pool, n, &|s, e| {
                let r = unsafe { sp.at(s * m, (e - s) * m) };
                for row in r.chunks_exact_mut(m) {
                    for (x, j) in row.iter_mut().zip(0..m) {
                        *x = *x * scale + mask.map_or(0.0, |mk| mk[j]);
                    }
                    softmax(row);
                }
            });
            if prof {
                t = attn_prof::add(&attn_prof::SOFT, t);
            }
            oh.iter_mut().for_each(|x| *x = 0.0);
            crate::fcd_ops::gemm_dx(&sc, &vh, &mut oh, n, dh, m, pool);
            if prof {
                t = attn_prof::add(&attn_prof::VALUE, t);
            }
            for i in 0..n {
                out[i * inner + h * dh..i * inner + (h + 1) * dh]
                    .copy_from_slice(&oh[i * dh..(i + 1) * dh]);
            }
            if prof {
                t = attn_prof::add(&attn_prof::GATHER, t);
            }
        }

        if let Some(g) = &self.gate {
            let logits = g.apply(x, n, pool);
            let h = self.heads;
            let dst = Shared(out.as_mut_ptr());
            rows(pool, n, &|s, e| {
                let r = unsafe { dst.at(s * inner, (e - s) * inner) };
                for (i, row) in r.chunks_exact_mut(inner).enumerate() {
                    for hh in 0..h {
                        let gate = 2.0 / (1.0 + (-logits[(s + i) * h + hh]).exp());
                        for d in row[hh * self.dh..(hh + 1) * self.dh].iter_mut() {
                            *d *= gate;
                        }
                    }
                }
            });
        }
        let r = self.o.apply(&out, n, pool);
        if prof {
            attn_prof::add(&attn_prof::OUT, t);
        }
        r
    }
}

// ---------------------------------------------------------- adaLN single

/// `AdaLayerNormSingle`: sinusoidal timestep → SiLU MLP → `coeff·dim`
/// modulation values, plus the embedding the output head reuses.
struct AdaLn {
    l1: Lin,
    l2: Lin,
    lin: Lin,
    dim: usize,
}

impl AdaLn {
    fn load(model: &Arc<CmfModel>, p: &str, dim: usize) -> Result<AdaLn, String> {
        Ok(AdaLn {
            l1: Lin::load(model, &format!("{p}.emb.timestep_embedder.linear_1"), true)?,
            l2: Lin::load(model, &format!("{p}.emb.timestep_embedder.linear_2"), true)?,
            lin: Lin::load(model, &format!("{p}.linear"), true)?,
            dim,
        })
    }

    /// `(values [n, coeff·dim], embedded [n, dim])` for `n` timesteps.
    fn forward(&self, t: &[f32], pool: Option<&Pool>) -> (Vec<f32>, Vec<f32>) {
        let n = t.len();
        // get_timestep_embedding(256, flip_sin_to_cos=True, shift=0): the
        // flip puts cosine first, so the halves are [cos, sin].
        let half = 128usize;
        let mut proj = vec![0f32; n * 256];
        let ws: Vec<f64> = (0..half)
            .map(|j| (-(10000f64).ln() * j as f64 / half as f64).exp())
            .collect();
        for (i, &tv) in t.iter().enumerate() {
            for (j, &w) in ws.iter().enumerate() {
                let a = tv as f64 * w;
                proj[i * 256 + j] = a.cos() as f32;
                proj[i * 256 + half + j] = a.sin() as f32;
            }
        }
        let mut h = self.l1.apply(&proj, n, pool);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        let embedded = self.l2.apply(&h, n, pool);
        let mut act = embedded.clone();
        for v in act.iter_mut() {
            *v = silu(*v);
        }
        (self.lin.apply(&act, n, pool), embedded)
    }
}

/// adaLN values for the *distinct* timesteps of a stream, plus each token's
/// index into them. Per-token timesteps take only a handful of values here
/// (a conditioning token sits at 0 while the rest sit at the current
/// sigma), so the `[36864, 4096]` projection runs a few times, not `T`.
struct TsTable {
    vals: Vec<f32>,
    emb: Vec<f32>,
    idx: Vec<usize>,
    width: usize,
    edim: usize,
}

impl TsTable {
    fn build(a: &AdaLn, ts: &[f32], scale: f64, pool: Option<&Pool>) -> TsTable {
        let mut vals: Vec<f32> = Vec::new();
        let mut idx = Vec::with_capacity(ts.len());
        for &t in ts {
            match vals.iter().position(|&v| v.to_bits() == t.to_bits()) {
                Some(i) => idx.push(i),
                None => {
                    vals.push(t);
                    idx.push(vals.len() - 1);
                }
            }
        }
        let scaled: Vec<f32> = vals.iter().map(|&v| (v as f64 * scale) as f32).collect();
        let (v, e) = a.forward(&scaled, pool);
        let width = v.len() / scaled.len().max(1);
        TsTable { vals: v, emb: e, idx, width, edim: a.dim }
    }

    fn distinct(&self) -> usize {
        self.vals.len() / self.width.max(1)
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.vals[r * self.width..(r + 1) * self.width]
    }

    fn emb_row(&self, r: usize) -> &[f32] {
        &self.emb[r * self.edim..(r + 1) * self.edim]
    }

    /// `(shift, scale, gate)` per distinct timestep for the adaLN triple at
    /// table row `off`: the block's static table plus the timestep's own
    /// contribution, summed once instead of once per token.
    fn triples(&self, table: &[f32], dim: usize, off: usize) -> Vec<[Vec<f32>; 3]> {
        (0..self.distinct())
            .map(|r| {
                let v = self.row(r);
                std::array::from_fn(|j| {
                    let o = (off + j) * dim;
                    (0..dim).map(|d| table[o + d] + v[o + d]).collect()
                })
            })
            .collect()
    }

    /// `(scale, shift)` per distinct timestep for an A↔V table, which — unlike
    /// the self-attention rows — comes out scale-first.
    fn pairs(&self, table: &[f32], dim: usize, off: usize) -> Vec<[Vec<f32>; 2]> {
        (0..self.distinct())
            .map(|r| {
                let v = self.row(r);
                std::array::from_fn(|j| {
                    let o = (off + j) * dim;
                    (0..dim).map(|d| table[o + d] + v[o + d]).collect()
                })
            })
            .collect()
    }
}


/// `rms_norm(x) · (1 + scale) + shift` over every token, in parallel.
/// `mods[r]` is the `(shift, scale, gate)` triple of the r-th distinct
/// timestep and `idx[t]` says which one token `t` uses.
fn ada_zero_rows(
    x: &[f32],
    out: &mut [f32],
    n: usize,
    dim: usize,
    mods: &[[Vec<f32>; 3]],
    idx: &[usize],
    pool: Option<&Pool>,
) {
    let dst = Shared(out.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            let md = &mods[idx[i]];
            rms_plain(&x[i * dim..(i + 1) * dim], row);
            for d in 0..dim {
                row[d] = row[d] * (1.0 + md[1][d]) + md[0][d];
            }
        }
    });
}

/// `x += y · gate`, the residual every sub-layer writes back through.
fn add_gated(
    x: &mut [f32],
    y: &[f32],
    n: usize,
    dim: usize,
    mods: &[[Vec<f32>; 3]],
    idx: &[usize],
    pool: Option<&Pool>,
) {
    let dst = Shared(x.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            let g = &mods[idx[i]][2];
            for d in 0..dim {
                row[d] += y[i * dim + d] * g[d];
            }
        }
    });
}

/// The post-SA pair: fold the gated update into the residual and hand the
/// re-normalized result to cross-attention, in one pass over the tokens.
fn post_sa_rows(
    x: &mut [f32],
    y: &[f32],
    normed: &mut [f32],
    n: usize,
    dim: usize,
    mods: &[[Vec<f32>; 3]],
    idx: &[usize],
    pool: Option<&Pool>,
) {
    let a = Shared(x.as_mut_ptr());
    let b = Shared(normed.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let xr = unsafe { a.at(s * dim, (e - s) * dim) };
        let nr = unsafe { b.at(s * dim, (e - s) * dim) };
        for ((row, nrow), i) in xr.chunks_exact_mut(dim).zip(nr.chunks_exact_mut(dim)).zip(s..e) {
            let g = &mods[idx[i]][2];
            for d in 0..dim {
                row[d] += y[i * dim + d] * g[d];
            }
            rms_plain(row, nrow);
        }
    });
}

/// `x += y · g` with one shared per-channel gate (the A↔V fusion).
fn add_scaled(x: &mut [f32], y: &[f32], n: usize, dim: usize, g: &[f32], pool: Option<&Pool>) {
    let dst = Shared(x.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            for d in 0..dim {
                row[d] += y[i * dim + d] * g[d];
            }
        }
    });
}

/// The cross-attention query: an affine on an already-normalized row.
fn affine_rows(
    x: &[f32],
    out: &mut [f32],
    n: usize,
    dim: usize,
    mods: &[[Vec<f32>; 3]],
    idx: &[usize],
    pool: Option<&Pool>,
) {
    let dst = Shared(out.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            let md = &mods[idx[i]];
            for d in 0..dim {
                row[d] = x[i * dim + d] * (1.0 + md[1][d]) + md[0][d];
            }
        }
    });
}


/// Where a denoising step actually goes. `CMF_LTX_PROF=1` prints the split
/// once per forward: guessing at this is how ports stay slow.
#[derive(Default)]
struct Prof {
    on: bool,
    t: [f64; 6],
}

const P_ADALN: usize = 0;
const P_SELF: usize = 1;
const P_CROSS: usize = 2;
const P_FUSE: usize = 3;
const P_FF: usize = 4;
const P_MOD: usize = 5;

impl Prof {
    fn new() -> Prof {
        Prof { on: std::env::var("CMF_LTX_PROF").is_ok(), t: [0.0; 6] }
    }
    #[inline]
    fn tick(&mut self, slot: usize, at: std::time::Instant) -> std::time::Instant {
        if self.on {
            self.t[slot] += at.elapsed().as_secs_f64();
            return std::time::Instant::now();
        }
        at
    }
    fn report(&self) {
        if !self.on {
            return;
        }
        // The q4tp GEMM's own split, when Metal is keeping it: a copy-bound
        // step wants fused blocks, a kernel-bound one wants a better kernel.
        #[cfg(target_os = "macos")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let n = crate::gpu_metal::MM_N.swap(0, Relaxed);
            if n > 0 {
                let us = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / 1e6;
                println!(
                    "  q4tp on device: {n} calls, upload {:.2}s  kernel {:.2}s  download {:.2}s",
                    us(&crate::gpu_metal::MM_UP),
                    us(&crate::gpu_metal::MM_GPU),
                    us(&crate::gpu_metal::MM_DN),
                );
            }
        }
        println!("  attention: {}", attn_prof::report());
        let names = ["adaln", "self-attn", "cross-attn", "a<->v", "ffn", "modulate"];
        let total: f64 = self.t.iter().sum();
        let parts: Vec<String> = names
            .iter()
            .zip(&self.t)
            .map(|(n, v)| format!("{n} {v:.1}s ({:.0}%)", 100.0 * v / total.max(1e-9)))
            .collect();
        println!("  profile: {}", parts.join("  "));
    }
}

// ----------------------------------------------------------------- block

struct Stream {
    attn1: Attn,
    attn2: Attn,
    ff_in: Lin,
    ff_out: Lin,
    sst: Vec<f32>,        // [9, dim]
    prompt_sst: Vec<f32>, // [2, dim]
}

impl Stream {
    fn load(
        model: &Arc<CmfModel>,
        p: &str,
        prefix: &str,
        heads: usize,
        dh: usize,
        ff_bias: bool,
    ) -> Result<Stream, String> {
        let a = |n: &str| format!("{p}.{prefix}{n}");
        Ok(Stream {
            attn1: Attn::load(model, &a("attn1"), heads, dh)?,
            attn2: Attn::load(model, &a("attn2"), heads, dh)?,
            ff_in: Lin::load(model, &a("ff.net.0.proj"), ff_bias)?,
            ff_out: Lin::load(model, &a("ff.net.2"), ff_bias)?,
            sst: cmf_f32(model, &a("scale_shift_table"))?,
            prompt_sst: cmf_f32(model, &a("prompt_scale_shift_table"))?,
        })
    }

    fn ff(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut h = self.ff_in.apply(x, n, pool);
        gelu_tanh_rows(&mut h, pool);
        self.ff_out.apply(&h, n, pool)
    }
}

struct Block {
    video: Stream,
    audio: Stream,
    a2v: Attn,
    v2a: Attn,
    sst_a2v_video: Vec<f32>, // [5, 4096]
    sst_a2v_audio: Vec<f32>, // [5, 2048]
}

// ----------------------------------------------------------------- model

/// One modality's per-step conditioning: everything the blocks read that is
/// not a weight.
pub struct StreamInput {
    /// Patchified latent, `[tokens, in_channels]`.
    pub latent: Vec<f32>,
    pub tokens: usize,
    /// Per-token timestep (the sigma·denoising-mask product).
    pub timesteps: Vec<f32>,
    /// Per-token patch midpoints, one entry per RoPE axis.
    pub positions: Vec<Vec<f64>>,
    /// Prompt embeddings out of the connector, `[ctx_len, cross_dim]`.
    pub context: Vec<f32>,
    pub ctx_len: usize,
    /// Additive per-key prompt mask, or empty for "attend to all".
    pub context_mask: Vec<f32>,
    /// Non-zero for tokens holding a standalone pixel frame (video only).
    pub keyframes: Vec<f32>,
    /// This stream's sigma — the *other* stream's fusion gate reads it.
    pub sigma: f32,
}

pub struct LtxDit {
    model: Arc<CmfModel>,
    blocks: Vec<Block>,
    patchify: Lin,
    a_patchify: Lin,
    keyframes_emb: Option<Vec<f32>>,
    adaln: AdaLn,
    a_adaln: AdaLn,
    prompt_adaln: AdaLn,
    a_prompt_adaln: AdaLn,
    av_v_ss: AdaLn,
    av_a_ss: AdaLn,
    av_a2v_gate: AdaLn,
    av_v2a_gate: AdaLn,
    proj_out: Lin,
    a_proj_out: Lin,
    sst_out: Vec<f32>,
    a_sst_out: Vec<f32>,
    pub heads: usize,
    pub dh: usize,
    pub a_heads: usize,
    pub a_dh: usize,
    pub max_pos: Vec<f64>,
    pub a_max_pos: Vec<f64>,
    pub cross_max_pos: f64,
    pub theta: f64,
    pub t_scale: f64,
    pub av_t_scale: f64,
    pub audio_cross_dim: usize,
}

impl LtxDit {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<LtxDit, String> {
        let cfg_bytes = ["ltx.config_json", "dit.config_json"]
            .iter()
            .find_map(|n| model.tensor(n).map(|e| model.entry_bytes(e)))
            .ok_or("container carries no ltx.config_json")?;
        let cfg: serde_json::Value =
            serde_json::from_slice(cfg_bytes).map_err(|e| format!("ltx.config_json: {e}"))?;
        let t = cfg.get("transformer").unwrap_or(&cfg).clone();
        let g = |k: &str, d: f64| t.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let heads = g("num_attention_heads", 32.0) as usize;
        let dh = g("attention_head_dim", 128.0) as usize;
        let a_heads = g("audio_num_attention_heads", 32.0) as usize;
        let a_dh = g("audio_attention_head_dim", 64.0) as usize;
        let n_layers = g("num_layers", 48.0) as usize;
        let ff_bias = t.get("ff_bias").and_then(|v| v.as_bool()).unwrap_or(true);
        let a_ff_bias = t.get("audio_ff_bias").and_then(|v| v.as_bool()).unwrap_or(true);
        let arr = |k: &str, d: Vec<f64>| -> Vec<f64> {
            t.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or(d)
        };
        let max_pos = arr("positional_embedding_max_pos", vec![20.0, 2048.0, 2048.0]);
        let a_max_pos = arr("audio_positional_embedding_max_pos", vec![20.0]);
        let cross_max_pos = max_pos[0].max(a_max_pos[0]);
        let dim = heads * dh;
        let a_dim = a_heads * a_dh;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("dit.transformer_blocks.{i}");
            blocks.push(Block {
                video: Stream::load(model, &p, "", heads, dh, ff_bias)?,
                audio: Stream::load(model, &p, "audio_", a_heads, a_dh, a_ff_bias)?,
                a2v: Attn::load(model, &format!("{p}.audio_to_video_attn"), a_heads, a_dh)?,
                v2a: Attn::load(model, &format!("{p}.video_to_audio_attn"), a_heads, a_dh)?,
                sst_a2v_video: cmf_f32(model, &format!("{p}.scale_shift_table_a2v_ca_video"))?,
                sst_a2v_audio: cmf_f32(model, &format!("{p}.scale_shift_table_a2v_ca_audio"))?,
            });
        }
        let _ = (dim, a_dim);
        Ok(LtxDit {
            blocks,
            patchify: Lin::load(model, "dit.patchify_proj", true)?,
            a_patchify: Lin::load(model, "dit.audio_patchify_proj", true)?,
            keyframes_emb: match model.tensor("dit.keyframes_abs_pos_embedding") {
                Some(_) => Some(cmf_f32(model, "dit.keyframes_abs_pos_embedding")?),
                None => None,
            },
            adaln: AdaLn::load(model, "dit.adaln_single", dim)?,
            a_adaln: AdaLn::load(model, "dit.audio_adaln_single", a_dim)?,
            prompt_adaln: AdaLn::load(model, "dit.prompt_adaln_single", dim)?,
            a_prompt_adaln: AdaLn::load(model, "dit.audio_prompt_adaln_single", a_dim)?,
            av_v_ss: AdaLn::load(model, "dit.av_ca_video_scale_shift_adaln_single", dim)?,
            av_a_ss: AdaLn::load(model, "dit.av_ca_audio_scale_shift_adaln_single", a_dim)?,
            av_a2v_gate: AdaLn::load(model, "dit.av_ca_a2v_gate_adaln_single", dim)?,
            av_v2a_gate: AdaLn::load(model, "dit.av_ca_v2a_gate_adaln_single", a_dim)?,
            proj_out: Lin::load(model, "dit.proj_out", true)?,
            a_proj_out: Lin::load(model, "dit.audio_proj_out", true)?,
            sst_out: cmf_f32(model, "dit.scale_shift_table")?,
            a_sst_out: cmf_f32(model, "dit.audio_scale_shift_table")?,
            heads,
            dh,
            a_heads,
            a_dh,
            max_pos,
            a_max_pos,
            cross_max_pos,
            theta: g("positional_embedding_theta", 10000.0),
            t_scale: g("timestep_scale_multiplier", 1000.0),
            av_t_scale: g("av_ca_timestep_scale_multiplier", 1.0),
            audio_cross_dim: g("audio_cross_attention_dim", 2048.0) as usize,
            model: model.clone(),
        })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn container(&self) -> &Arc<CmfModel> {
        &self.model
    }

    /// One denoising step: `(video velocity [T, C], audio velocity [T, C])`.
    pub fn forward(
        &self,
        video: &StreamInput,
        audio: &StreamInput,
        pool: Option<&Pool>,
    ) -> (Vec<f32>, Vec<f32>) {
        self.forward_traced(video, audio, pool, &mut |_, _| {})
    }

    pub fn forward_traced(
        &self,
        video: &StreamInput,
        audio: &StreamInput,
        pool: Option<&Pool>,
        trace: &mut dyn FnMut(&str, &[f32]),
    ) -> (Vec<f32>, Vec<f32>) {
        // A denoising step is the opposite of what the per-op probe is built
        // for: forty-eight identical blocks, the same shapes every time, the
        // device warm throughout. Take the probe out of it.
        let _trust = crate::gpu::trust_gpu();
        let dim = self.heads * self.dh;
        let a_dim = self.a_heads * self.a_dh;
        let (n, m) = (video.tokens, audio.tokens);

        // --- patchify -----------------------------------------------------
        let mut vx = self.patchify.apply(&video.latent, n, pool);
        if let Some(emb) = &self.keyframes_emb {
            for i in 0..n {
                if video.keyframes.get(i).copied().unwrap_or(0.0) > 0.0 {
                    for (d, &e) in vx[i * dim..(i + 1) * dim].iter_mut().zip(emb) {
                        *d += e;
                    }
                }
            }
        }
        let mut ax = self.a_patchify.apply(&audio.latent, m, pool);
        trace("v.args.x", &vx);
        trace("a.args.x", &ax);

        // --- adaLN tables, one row per distinct timestep -------------------
        let vt = TsTable::build(&self.adaln, &video.timesteps, self.t_scale, pool);
        let at = TsTable::build(&self.a_adaln, &audio.timesteps, self.t_scale, pool);
        let vpt = TsTable::build(&self.prompt_adaln, &[video.sigma], self.t_scale, pool);
        let apt = TsTable::build(&self.a_prompt_adaln, &[audio.sigma], self.t_scale, pool);
        let vxs = TsTable::build(&self.av_v_ss, &video.timesteps, self.t_scale, pool);
        let axs = TsTable::build(&self.av_a_ss, &audio.timesteps, self.t_scale, pool);
        // The fusion gate reads the *other* stream's sigma — the noise level
        // it is being asked to trust — at the A-V multiplier.
        let vgt = TsTable::build(&self.av_a2v_gate, &[audio.sigma], self.av_t_scale, pool);
        let agt = TsTable::build(&self.av_v2a_gate, &[video.sigma], self.av_t_scale, pool);

        // --- RoPE ---------------------------------------------------------
        let v_pe = Rope::build(&video.positions, &self.max_pos, dim, self.heads, self.theta);
        let a_pe = Rope::build(&audio.positions, &self.a_max_pos, a_dim, self.a_heads, self.theta);
        let time_only = |p: &[Vec<f64>]| p.iter().map(|r| vec![r[0]]).collect::<Vec<_>>();
        let v_xpe = Rope::build(
            &time_only(&video.positions),
            &[self.cross_max_pos],
            self.audio_cross_dim,
            self.heads,
            self.theta,
        );
        let a_xpe = Rope::build(
            &time_only(&audio.positions),
            &[self.cross_max_pos],
            self.audio_cross_dim,
            self.a_heads,
            self.theta,
        );

        let vmask = (!video.context_mask.is_empty()).then_some(&video.context_mask[..]);
        let amask = (!audio.context_mask.is_empty()).then_some(&audio.context_mask[..]);

        let mut prof = Prof::new();
        for (bi, blk) in self.blocks.iter().enumerate() {
            let mut pt = std::time::Instant::now();
            let v_msa = vt.triples(&blk.video.sst, dim, 0);
            let v_ca = vt.triples(&blk.video.sst, dim, 6);
            let v_mlp = vt.triples(&blk.video.sst, dim, 3);
            let a_msa = at.triples(&blk.audio.sst, a_dim, 0);
            let a_ca = at.triples(&blk.audio.sst, a_dim, 6);
            let a_mlp = at.triples(&blk.audio.sst, a_dim, 3);

            // ---- video: self-attention, then prompt cross-attention ----
            pt = prof.tick(P_ADALN, pt);
            let mut vnorm = vec![0f32; n * dim];
            ada_zero_rows(&vx, &mut vnorm, n, dim, &v_msa, &vt.idx, pool);
            pt = prof.tick(P_MOD, pt);
            if bi == 0 {
                trace("v.b0.sa.in", &vnorm);
            }
            let vsa = blk
                .video
                .attn1
                .forward(&vnorm, n, &vnorm, n, Some(&v_pe), Some(&v_pe), None, pool);
            if bi == 0 {
                trace("v.b0.sa.out", &vsa);
            }
            pt = prof.tick(P_SELF, pt);
            let mut vnormed = vec![0f32; n * dim];
            post_sa_rows(&mut vx, &vsa, &mut vnormed, n, dim, &v_msa, &vt.idx, pool);
            let mut vq = vec![0f32; n * dim];
            affine_rows(&vnormed, &mut vq, n, dim, &v_ca, &vt.idx, pool);
            let vctx = modulate_kv(&video.context, video.ctx_len, dim, &blk.video.prompt_sst, vpt.row(0), pool);
            let vca = blk.video.attn2.forward(&vq, n, &vctx, video.ctx_len, None, None, vmask, pool);
            if bi == 0 {
                trace("v.b0.ca.in", &vq);
                trace("v.b0.ca.ctx", &vctx);
                trace("v.b0.ca.out", &vca);
            }
            add_gated(&mut vx, &vca, n, dim, &v_ca, &vt.idx, pool);
            pt = prof.tick(P_CROSS, pt);

            // ---- audio: the same two steps ----
            let mut anorm = vec![0f32; m * a_dim];
            ada_zero_rows(&ax, &mut anorm, m, a_dim, &a_msa, &at.idx, pool);
            if bi == 0 {
                trace("a.b0.sa.in", &anorm);
            }
            let asa = blk
                .audio
                .attn1
                .forward(&anorm, m, &anorm, m, Some(&a_pe), Some(&a_pe), None, pool);
            if bi == 0 {
                trace("a.b0.sa.out", &asa);
            }
            let mut anormed = vec![0f32; m * a_dim];
            post_sa_rows(&mut ax, &asa, &mut anormed, m, a_dim, &a_msa, &at.idx, pool);
            let mut aq = vec![0f32; m * a_dim];
            affine_rows(&anormed, &mut aq, m, a_dim, &a_ca, &at.idx, pool);
            let actx = modulate_kv(&audio.context, audio.ctx_len, a_dim, &blk.audio.prompt_sst, apt.row(0), pool);
            let aca = blk.audio.attn2.forward(&aq, m, &actx, audio.ctx_len, None, None, amask, pool);
            if bi == 0 {
                trace("a.b0.ca.in", &aq);
                trace("a.b0.ca.ctx", &actx);
                trace("a.b0.ca.out", &aca);
            }
            add_gated(&mut ax, &aca, m, a_dim, &a_ca, &at.idx, pool);
            pt = prof.tick(P_CROSS, pt);

            // ---- audio ↔ video, both directions off the pre-fusion state ----
            let vx_pre = vx.clone();
            let ax_pre = ax.clone();
            let a2v_vp = vxs.pairs(&blk.sst_a2v_video, dim, 0);
            let a2v_ap = axs.pairs(&blk.sst_a2v_audio, a_dim, 0);
            let a2v_v = ada_pair(&vx_pre, n, dim, &a2v_vp, &vxs.idx, pool);
            let a2v_a = ada_pair(&ax_pre, m, a_dim, &a2v_ap, &axs.idx, pool);
            let a2v = blk
                .a2v
                .forward(&a2v_v, n, &a2v_a, m, Some(&v_xpe), Some(&a_xpe), None, pool);
            if bi == 0 {
                trace("v.b0.a2v.in", &a2v_v);
                trace("v.b0.a2v.ctx", &a2v_a);
                trace("v.b0.a2v.out", &a2v);
            }
            let gate_a2v = gate_row(&blk.sst_a2v_video, dim, vgt.row(0));
            add_scaled(&mut vx, &a2v, n, dim, &gate_a2v, pool);
            let v2a_ap = axs.pairs(&blk.sst_a2v_audio, a_dim, 2);
            let v2a_vp = vxs.pairs(&blk.sst_a2v_video, dim, 2);
            let v2a_a = ada_pair(&ax_pre, m, a_dim, &v2a_ap, &axs.idx, pool);
            let v2a_v = ada_pair(&vx_pre, n, dim, &v2a_vp, &vxs.idx, pool);
            let v2a = blk
                .v2a
                .forward(&v2a_a, m, &v2a_v, n, Some(&a_xpe), Some(&v_xpe), None, pool);
            if bi == 0 {
                trace("a.b0.v2a.in", &v2a_a);
                trace("a.b0.v2a.ctx", &v2a_v);
                trace("a.b0.v2a.out", &v2a);
            }
            let gate_v2a = gate_row(&blk.sst_a2v_audio, a_dim, agt.row(0));
            add_scaled(&mut ax, &v2a, m, a_dim, &gate_v2a, pool);
            pt = prof.tick(P_FUSE, pt);

            // ---- feed-forward ----
            let mut vsc = vec![0f32; n * dim];
            ada_zero_rows(&vx, &mut vsc, n, dim, &v_mlp, &vt.idx, pool);
            let vff = blk.video.ff(&vsc, n, pool);
            if bi == 0 {
                trace("v.b0.ff.in", &vsc);
                trace("v.b0.ff.out", &vff);
            }
            add_gated(&mut vx, &vff, n, dim, &v_mlp, &vt.idx, pool);
            let mut asc = vec![0f32; m * a_dim];
            ada_zero_rows(&ax, &mut asc, m, a_dim, &a_mlp, &at.idx, pool);
            let aff = blk.audio.ff(&asc, m, pool);
            if bi == 0 {
                trace("a.b0.ff.in", &asc);
                trace("a.b0.ff.out", &aff);
            }
            add_gated(&mut ax, &aff, m, a_dim, &a_mlp, &at.idx, pool);
            pt = prof.tick(P_FF, pt);
            trace(&format!("v.block{bi}"), &vx);
            trace(&format!("a.block{bi}"), &ax);
        }

        prof.report();

        // --- output head: LayerNorm (no affine), adaLN, projection --------
        let vout = head(&vx, n, dim, &self.sst_out, &vt, &self.proj_out, pool);
        let aout = head(&ax, m, a_dim, &self.a_sst_out, &at, &self.a_proj_out, pool);
        trace("v.out", &vout);
        trace("a.out", &aout);
        (vout, aout)
    }
}

/// `prompt_scale_shift_table` plus the prompt adaLN row, modulating the
/// cross-attention K/V — the same modulation for every context token.
fn modulate_kv(
    ctx: &[f32],
    len: usize,
    dim: usize,
    table: &[f32],
    extra: &[f32],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = vec![0f32; len * dim];
    let shift: Vec<f32> = (0..dim).map(|d| table[d] + extra[d]).collect();
    let scale: Vec<f32> = (0..dim).map(|d| table[dim + d] + extra[dim + d]).collect();
    // A thousand prompt tokens by four thousand channels, rebuilt in every
    // one of forty-eight blocks: three hundred million writes a step, which
    // is not something to do on one thread.
    let dst = Shared(out.as_mut_ptr());
    rows(pool, len, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            for d in 0..dim {
                row[d] = ctx[i * dim + d] * (1.0 + scale[d]) + shift[d];
            }
        }
    });
    out
}


/// `ada_zero` with an A↔V `(scale, shift)` pair per distinct timestep.
fn ada_pair(
    x: &[f32],
    n: usize,
    dim: usize,
    pairs: &[[Vec<f32>; 2]],
    idx: &[usize],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = vec![0f32; n * dim];
    let dst = Shared(out.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            let p = &pairs[idx[i]];
            rms_plain(&x[i * dim..(i + 1) * dim], row);
            for d in 0..dim {
                row[d] = row[d] * (1.0 + p[0][d]) + p[1][d];
            }
        }
    });
    out
}

/// The single gate row of an A↔V table — row 4 of `[5, dim]`, plus the
/// gate adaLN's own output.
fn gate_row(table: &[f32], dim: usize, extra: &[f32]) -> Vec<f32> {
    (0..dim).map(|d| table[4 * dim + d] + extra[d]).collect()
}

/// The output head: LayerNorm without affine, the final scale/shift pair
/// (both offset by the same embedded timestep), then the projection.
fn head(
    x: &[f32],
    n: usize,
    dim: usize,
    sst: &[f32],
    ts: &TsTable,
    proj: &Lin,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut y = vec![0f32; n * dim];
    let dst = Shared(y.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let r = unsafe { dst.at(s * dim, (e - s) * dim) };
        let mut ln = vec![0f32; dim];
        for (row, i) in r.chunks_exact_mut(dim).zip(s..e) {
            let emb = ts.emb_row(ts.idx[i.min(ts.idx.len() - 1)]);
            layer_norm(&x[i * dim..(i + 1) * dim], &mut ln);
            for d in 0..dim {
                row[d] = ln[d] * (1.0 + sst[dim + d] + emb[d]) + sst[d] + emb[d];
            }
        }
    });
    proj.apply(&y, n, pool)
}
