//! Z-Image DiT (Tongyi-MAI Z-Image and Z-Image-Turbo: one 6.15B
//! single-stream `ZImageTransformer2DModel`) — the CPU-exact reference
//! path and the host glue around the `gpu::zimage_*` contract.
//!
//! New code, NOT a mode of Lumina (`dit.rs` stays bit-identical): helpers
//! are copied, never shared. Sequence order is diffusers' [img, cap], so
//! every intermediate lines up row-for-row with the oracle dumps
//! (`python/zimage_oracle.py`).
//!
//! Semantics (diffusers 0.40 `transformer_z_image.py`): adaLN has NO SiLU
//! before it (the final layer does); t is scaled by 1000; RoPE θ = 256,
//! axes [32, 48, 48], complex-interleaved; caption ids (1+j, 0, 0) over
//! the padded L_p, image ids (L_p+1, r, c), image pad ids (0, 0, 0); pad
//! rows are real keys/queries (no mask at batch 1); caption pad rows :=
//! cap_pad_token after the embedder, image pad rows := x_pad_token after
//! the embed.
//!
//! Precision: every f32 op that diffusers runs in f32 on the host side
//! (sinusoid args, σ schedule, RoPE angles) is reproduced in f32 in the
//! same order; reductions (norms, the small linears) accumulate in f64.
//!
//! Per image the host precomputes everything latent-independent: caption
//! embed + context refiner (per prompt), the modulation of all steps, the
//! rope tables (per prompt and resolution). Per step the device (or
//! `step_cpu`) runs exactly x_embed → 2 noise refiners → 30 layers → final.

use crate::dit::Proj;
use crate::gpu::{ZBlockRef, ZGeom};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Sequence padding multiple (diffusers `SEQ_MULTI_OF`).
pub const SEQ_MULTI_OF: usize = 32;
/// Turbo scheduler shift (static, `use_dynamic_shifting: false`). The
/// base model's scheduler uses 6.0; the container stores its own.
pub const DEFAULT_SHIFT: f32 = 3.0;
/// Default number of DiT forwards (Turbo).
pub const DEFAULT_STEPS: usize = 8;

/// The Turbo geometry — for random-weight kernel tests in WP2/WP3 that
/// must not wait for a container.
pub const ZGEOM_TURBO: ZGeom = ZGeom {
    hidden: 3840,
    nh: 30,
    hd: 128,
    inter: 10240,
    eps: 1e-5,
    final_eps: 1e-6,
    patch_dim: 64,
};

/// `dit.config_json` / transformer `config.json` fields the runtime uses.
#[derive(Clone, Debug)]
pub struct ZConfig {
    /// 3840
    pub dim: usize,
    /// 30 main layers
    pub n_layers: usize,
    /// 2 noise-refiner and 2 context-refiner blocks
    pub n_refiner: usize,
    /// 30 heads, 30 kv heads, head_dim 128
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// int(dim / 3 · 8) = 10240
    pub ffn_dim: usize,
    /// 2560 (Qwen3-4B hidden)
    pub cap_feat_dim: usize,
    /// 256 (`ADALN_EMBED_DIM`) and the t-embedder MLP width 1024
    pub t_embed_dim: usize,
    pub t_hidden: usize,
    /// patch 2, latent channels 16 → patch vector 64
    pub patch: usize,
    pub in_channels: usize,
    /// 1e-5 (all RMSNorms), 1e-6 (final LayerNorm)
    pub norm_eps: f32,
    pub final_eps: f32,
    /// 256.0 and [32, 48, 48]
    pub rope_theta: f64,
    pub axes_dims: [usize; 3],
    /// 1000.0
    pub t_scale: f32,
}

impl ZConfig {
    /// Parse the diffusers transformer `config.json` (the same JSON is
    /// stored as `dit.config_json` in the container).
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |k: &str| -> Result<usize, String> {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("dit config: missing {k}"))
        };
        let dim = u("dim")?;
        let n_heads = u("n_heads")?;
        let axes: Vec<usize> = v["axes_dims"]
            .as_array()
            .ok_or("dit config: axes_dims")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        if axes.len() != 3 {
            return Err("dit config: axes_dims must have 3 entries".into());
        }
        let patch = v["all_patch_size"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|x| x.as_u64())
            .unwrap_or(2) as usize;
        let head_dim = dim / n_heads;
        if axes.iter().sum::<usize>() != head_dim {
            return Err(format!(
                "dit config: axes_dims {axes:?} do not sum to head_dim {head_dim}"
            ));
        }
        Ok(Self {
            dim,
            n_layers: u("n_layers")?,
            n_refiner: v["n_refiner_layers"].as_u64().unwrap_or(2) as usize,
            n_heads,
            n_kv_heads: v["n_kv_heads"].as_u64().unwrap_or(n_heads as u64) as usize,
            head_dim,
            ffn_dim: (dim as f64 / 3.0 * 8.0) as usize,
            cap_feat_dim: u("cap_feat_dim")?,
            t_embed_dim: dim.min(256),
            t_hidden: 1024,
            patch,
            in_channels: v["in_channels"].as_u64().unwrap_or(16) as usize,
            norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            final_eps: 1e-6,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(256.0),
            axes_dims: [axes[0], axes[1], axes[2]],
            t_scale: v["t_scale"].as_f64().unwrap_or(1000.0) as f32,
        })
    }

    pub fn geom(&self) -> ZGeom {
        ZGeom {
            hidden: self.dim,
            nh: self.n_heads,
            hd: self.head_dim,
            inter: self.ffn_dim,
            eps: self.norm_eps,
            final_eps: self.final_eps,
            patch_dim: self.patch * self.patch * self.in_channels,
        }
    }

    /// Blocks carrying adaLN: the noise refiner, then the main layers.
    pub fn n_mod_blocks(&self) -> usize {
        self.n_refiner + self.n_layers
    }
}

/// Token geometry of one (resolution, prompt length).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZShape {
    /// Latent size (H/8, W/8).
    pub h_lat: usize,
    pub w_lat: usize,
    /// Patch grid (H/16, W/16).
    pub grid: (usize, usize),
    /// n_img = grid.0 · grid.1, n_img_p = ceil32(n_img).
    pub n_img: usize,
    pub n_img_p: usize,
    /// Caption tokens L and ceil32(L).
    pub l: usize,
    pub l_p: usize,
}

impl ZShape {
    /// From the image size in pixels (multiples of 16) and the caption length.
    pub fn new(height: usize, width: usize, l: usize) -> Self {
        let grid = (height / 16, width / 16);
        let n_img = grid.0 * grid.1;
        Self {
            h_lat: height / 8,
            w_lat: width / 8,
            grid,
            n_img,
            n_img_p: ceil32(n_img),
            l,
            l_p: ceil32(l),
        }
    }

    /// S = n_img_p + l_p.
    pub fn seq(&self) -> usize {
        self.n_img_p + self.l_p
    }
}

/// RoPE tables, each `(cos, sin)` of [rows · hd/2] f32.
pub struct ZRope {
    /// Noise refiner, [n_img_p] rows (image ids).
    pub img: (Vec<f32>, Vec<f32>),
    /// Main layers, [n_img_p + l_p] rows ordered [img, cap].
    pub joint: (Vec<f32>, Vec<f32>),
    /// Context refiner, [l_p] rows (caption ids).
    pub cap: (Vec<f32>, Vec<f32>),
}

/// Which block of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZBlockId {
    NoiseRefiner(usize),
    ContextRefiner(usize),
    Layer(usize),
}

/// Device views of every block (for `ZPrepareArgs` and
/// `gpu::zimage_refine_caption`). Borrowed from a `ZImageDit::from_cmf`.
pub struct ZBlockRefs<'a> {
    pub noise_refiner: Vec<ZBlockRef<'a>>,
    pub context_refiner: Vec<ZBlockRef<'a>>,
    pub layers: Vec<ZBlockRef<'a>>,
}

/// Per-(prompt, resolution) state: the refined caption and the rope
/// tables, plus whether the device accepted `gpu::zimage_prepare`.
pub struct ZPrepared {
    pub key: u64,
    pub shape: ZShape,
    /// [l_p, dim], after cap_embedder, pad-token rows and the context refiner.
    pub cap: Vec<f32>,
    pub rope: ZRope,
    pub device: bool,
}

/// One transformer block. `adaln` is None for the context refiner.
struct ZBlock {
    /// `adaLN_modulation.0`: weight [4·dim, 256] f32 (host precompute only),
    /// bias [4·dim].
    adaln: Option<(Vec<f32>, Vec<f32>)>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    ffn_norm1: Vec<f32>,
    ffn_norm2: Vec<f32>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    w1: Proj,
    w2: Proj,
    w3: Proj,
    /// Tensor indices [q, k, v, o, w1, w3, w2] in the container.
    idx: Option<[usize; 7]>,
}

/// The Z-Image transformer. Tensor names are the ORIGINAL diffusers names
/// under `dit.` (no rename map).
pub struct ZImageDit {
    pub cfg: ZConfig,
    /// The container this was loaded from (None for `load_dir`) — device
    /// paths need tensor indices into it.
    pub model: Option<Arc<CmfModel>>,
    x_emb_w: Vec<f32>, // [dim, 64]
    x_emb_b: Vec<f32>,
    x_pad: Vec<f32>,   // [dim]
    cap_pad: Vec<f32>, // [dim]
    t_w0: Vec<f32>,    // [1024, 256]
    t_b0: Vec<f32>,
    t_w2: Vec<f32>, // [256, 1024]
    t_b2: Vec<f32>,
    cap_norm: Vec<f32>, // [cap_feat]
    cap_w: Proj,        // [dim, cap_feat]
    cap_b: Vec<f32>,
    final_mod_w: Vec<f32>, // [dim, 256]
    final_mod_b: Vec<f32>,
    final_w: Vec<f32>, // [64, dim]
    final_b: Vec<f32>,
    noise_refiner: Vec<ZBlock>,
    context_refiner: Vec<ZBlock>,
    layers: Vec<ZBlock>,
    pool: Option<Arc<Pool>>,
}

// ───────────────────────────── small helpers (copied, not shared) ─────

struct SendRows(*mut f32);
unsafe impl Send for SendRows {}
unsafe impl Sync for SendRows {}
impl SendRows {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
    unsafe fn set(&self, off: usize, v: f32) {
        unsafe { *self.0.add(off) = v }
    }
}

fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// diffusers RMSNorm (plain w), f64 accumulation, into `dst`.
fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

fn rms_norm_inplace(v: &mut [f32], w: &[f32], eps: f64) {
    let ss = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for (x, &g) in v.iter_mut().zip(w) {
        *x = (*x as f64 * inv) as f32 * g;
    }
}

/// y = x·Wᵀ + b for ONE row, f64 accumulation, pool-parallel over outputs.
/// `linear_row` for several inputs at once (each weight row read once);
/// bit-identical per input to `linear_row`.
fn linear_rows_multi(xs: &[Vec<f32>], w: &[f32], b: &[f32], pool: Option<&Pool>) -> Vec<Vec<f32>> {
    let rows = b.len();
    let n = xs.len();
    let mut out = vec![0f32; n * rows];
    let op = SendRows(out.as_mut_ptr());
    pool_rows(pool, rows, &|lo, hi| {
        for o in lo..hi {
            for (si, x) in xs.iter().enumerate() {
                let k = x.len();
                let row = &w[o * k..(o + 1) * k];
                let s: f64 = row.iter().zip(x).map(|(&a, &c)| a as f64 * c as f64).sum();
                // SAFETY: disjoint output indices per worker.
                unsafe { op.set(si * rows + o, (s + b[o] as f64) as f32) };
            }
        }
    });
    out.chunks(rows.max(1)).map(|c| c.to_vec()).collect()
}

fn linear_row(x: &[f32], w: &[f32], b: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let k = x.len();
    let rows = b.len();
    debug_assert_eq!(w.len(), rows * k);
    let mut out = vec![0f32; rows];
    let op = SendRows(out.as_mut_ptr());
    pool_rows(pool, rows, &|lo, hi| {
        for o in lo..hi {
            let row = &w[o * k..(o + 1) * k];
            let s: f64 = row
                .iter()
                .zip(x)
                .map(|(&a, &c)| a as f64 * c as f64)
                .sum();
            // SAFETY: disjoint output indices per worker.
            unsafe { op.set(o, (s + b[o] as f64) as f32) };
        }
    });
    out
}

/// Row softmax, in place (f32, max-subtracted; f64 denominator).
fn softmax_inplace(row: &mut [f32]) {
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

/// The host GEMM of the reference path: y[n, m] = x[n, k] · w[m, k]ᵀ.
///
/// On x86-64 with AVX2+FMA a register-blocked kernel (3×3 micro-tiles of
/// 8-wide FMAs, K blocked by 256, tiles pool-parallel); the engine's
/// `fcd_ops::gemm_nt` only blocks under AVX-512, and on an AVX2 EPYC its
/// per-output dot ran the 6B DiT at ~0.17 TFLOP/s on 14 threads. Elsewhere
/// (aarch64: Accelerate/NEON) it defers to `fcd_ops::gemm_nt`. f32
/// accumulation; the summation order differs from torch only by rounding.
/// `CMF_ZIMAGE_HOST_GEMM=engine` forces `fcd_ops::gemm_nt` everywhere.
pub(crate) mod host_gemm {
    use crate::pool::Pool;

    const KC: usize = 256;
    const MB: usize = 48;
    const NB: usize = 48;

    fn native() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            if matches!(std::env::var("CMF_ZIMAGE_HOST_GEMM").as_deref(), Ok("engine")) {
                return false;
            }
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        })
    }

    struct YPtr(*mut f32);
    unsafe impl Send for YPtr {}
    unsafe impl Sync for YPtr {}

    /// y[i·ldy + j] = Σ_k x[i·k + ..]·w[j·k + ..] for i < n, j < m.
    /// `y` rows are `ldy` apart (≥ m): a column slice of a wider output.
    pub(crate) fn gemm_nt_ld(
        x: &[f32],
        w: &[f32],
        y: &mut [f32],
        ldy: usize,
        n: usize,
        k: usize,
        m: usize,
        pool: Option<&Pool>,
    ) {
        debug_assert!(x.len() >= n * k && w.len() >= m * k);
        debug_assert!(n == 0 || y.len() >= (n - 1) * ldy + m);
        #[cfg(target_arch = "x86_64")]
        if native() && k % 8 == 0 {
            for i in 0..n {
                y[i * ldy..i * ldy + m].fill(0.0);
            }
            let (ti, tj) = (n.div_ceil(MB), m.div_ceil(NB));
            let yp = YPtr(y.as_mut_ptr());
            let work = |lo: usize, hi: usize| {
                let yp = &yp;
                for t in lo..hi {
                    let (bi, bj) = (t / tj, t % tj);
                    let (i0, i1) = (bi * MB, ((bi + 1) * MB).min(n));
                    let (j0, j1) = (bj * NB, ((bj + 1) * NB).min(m));
                    // SAFETY: ISA checked; tiles write disjoint y blocks.
                    unsafe { tile(x, w, yp.0, ldy, k, i0, i1, j0, j1) };
                }
            };
            match pool {
                Some(p) => p.run_rows(ti * tj, &work),
                None => work(0, ti * tj),
            }
            return;
        }
        if ldy == m {
            crate::fcd_ops::gemm_nt(&x[..n * k], &w[..m * k], &mut y[..n * m], n, k, m, pool);
        } else {
            let mut tmp = vec![0f32; n * m];
            crate::fcd_ops::gemm_nt(&x[..n * k], &w[..m * k], &mut tmp, n, k, m, pool);
            for i in 0..n {
                y[i * ldy..i * ldy + m].copy_from_slice(&tmp[i * m..(i + 1) * m]);
            }
        }
    }

    pub(crate) fn gemm_nt(
        x: &[f32],
        w: &[f32],
        y: &mut [f32],
        n: usize,
        k: usize,
        m: usize,
        pool: Option<&Pool>,
    ) {
        gemm_nt_ld(x, w, y, m, n, k, m, pool)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile(
        x: &[f32],
        w: &[f32],
        y: *mut f32,
        ldy: usize,
        k: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        use std::arch::x86_64::*;
        let xp = x.as_ptr();
        let wp = w.as_ptr();
        let mut kb = 0;
        while kb < k {
            let kl = (k - kb).min(KC);
            let mut i = i0;
            while i < i1 {
                let ih = (i1 - i).min(3);
                let mut j = j0;
                while j < j1 {
                    let jh = (j1 - j).min(3);
                    if ih == 3 && jh == 3 {
                        let (x0, x1, x2) = (
                            xp.add(i * k + kb),
                            xp.add((i + 1) * k + kb),
                            xp.add((i + 2) * k + kb),
                        );
                        let (w0, w1, w2) = (
                            wp.add(j * k + kb),
                            wp.add((j + 1) * k + kb),
                            wp.add((j + 2) * k + kb),
                        );
                        let mut a = [_mm256_setzero_ps(); 9];
                        let mut kk = 0;
                        while kk < kl {
                            let b0 = _mm256_loadu_ps(w0.add(kk));
                            let b1 = _mm256_loadu_ps(w1.add(kk));
                            let b2 = _mm256_loadu_ps(w2.add(kk));
                            let r0 = _mm256_loadu_ps(x0.add(kk));
                            a[0] = _mm256_fmadd_ps(r0, b0, a[0]);
                            a[1] = _mm256_fmadd_ps(r0, b1, a[1]);
                            a[2] = _mm256_fmadd_ps(r0, b2, a[2]);
                            let r1 = _mm256_loadu_ps(x1.add(kk));
                            a[3] = _mm256_fmadd_ps(r1, b0, a[3]);
                            a[4] = _mm256_fmadd_ps(r1, b1, a[4]);
                            a[5] = _mm256_fmadd_ps(r1, b2, a[5]);
                            let r2 = _mm256_loadu_ps(x2.add(kk));
                            a[6] = _mm256_fmadd_ps(r2, b0, a[6]);
                            a[7] = _mm256_fmadd_ps(r2, b1, a[7]);
                            a[8] = _mm256_fmadd_ps(r2, b2, a[8]);
                            kk += 8;
                        }
                        for r in 0..3 {
                            for c in 0..3 {
                                *y.add((i + r) * ldy + j + c) += hsum(a[r * 3 + c]);
                            }
                        }
                    } else {
                        for r in 0..ih {
                            for c in 0..jh {
                                let (xr, wr) = (xp.add((i + r) * k + kb), wp.add((j + c) * k + kb));
                                let mut acc = _mm256_setzero_ps();
                                let mut kk = 0;
                                while kk < kl {
                                    acc = _mm256_fmadd_ps(
                                        _mm256_loadu_ps(xr.add(kk)),
                                        _mm256_loadu_ps(wr.add(kk)),
                                        acc,
                                    );
                                    kk += 8;
                                }
                                *y.add((i + r) * ldy + j + c) += hsum(acc);
                            }
                        }
                    }
                    j += jh;
                }
                i += ih;
            }
            kb += kl;
        }
    }
}

/// y[n, rows] = x[n, cols] · Wᵀ for a projection of any codec, on the host
/// reference GEMM: f32 weights directly; a quantized weight is dequantized
/// to f32 in row chunks (weight-only error — the activations stay f32, as
/// on the device paths). `CMF_ZIMAGE_HOST_GEMM=engine` uses the engine's
/// own `Proj::matmat` (its quantized CPU kernels and per-op device arms).
fn lin(p: &Proj, x: &[f32], n: usize, y: &mut [f32], pool: Option<&Pool>) {
    if matches!(std::env::var("CMF_ZIMAGE_HOST_GEMM").as_deref(), Ok("engine")) {
        return p.matmat(x, n, y, pool);
    }
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
                    let wp = SendRows(wbuf.as_mut_ptr());
                    pool_rows(pool, rc, &|lo, hi| {
                        for r in lo..hi {
                            // SAFETY: disjoint rows.
                            q.row_f32(r0 + r, unsafe { wp.row(r * cols, cols) });
                        }
                    });
                }
                host_gemm::gemm_nt_ld(x, &wbuf[..rc * cols], &mut y[r0..], rows, n, cols, rc, pool);
                r0 += rc;
            }
        }
    }
}

/// Any CMF entry → f32.
fn cmf_f32(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    crate::dit::cmf_f32(model, name)
}

/// ceil to a multiple of `SEQ_MULTI_OF`.
pub fn ceil32(n: usize) -> usize {
    n.div_ceil(SEQ_MULTI_OF) * SEQ_MULTI_OF
}

/// torch.linspace(start, end, n) in f32, the CPU kernel's order: `start +
/// step·i` for i < n/2, `end − step·(n−1−i)` otherwise.
fn linspace_f32(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

/// σ schedule in torch/numpy f32 order: linspace(1, 1/n, n), then
/// shift·s/(1+(shift−1)·s), then a terminal 0. Length n + 1.
/// N=4, shift 3 → [1, .9, .75, .5, 0].
pub fn sigmas_torch_f32(n: usize, shift: f32) -> Vec<f32> {
    // `1 / num_inference_steps` is a Python double rounded to f32 by torch.
    let end = (1.0f64 / n as f64) as f32;
    let mut s: Vec<f32> = linspace_f32(1.0, end, n)
        .into_iter()
        .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
        .collect();
    s.push(0.0);
    s
}

/// The pipeline's timestep: (1000 − σ·1000)/1000 in f32.
pub fn t_model(sigma: f32) -> f32 {
    let t = sigma * 1000.0;
    (1000.0 - t) / 1000.0
}

/// Position ids and RoPE tables for a patch grid and a caption of `l`
/// tokens (θ, axes from the config; f64 freqs, f32 angles, f32 cos/sin).
pub fn ids_and_rope(grid: (usize, usize), l: usize, theta: f64, axes: [usize; 3]) -> ZRope {
    let l_p = ceil32(l);
    let n_img = grid.0 * grid.1;
    let n_img_p = ceil32(n_img);
    let mut img_ids: Vec<[usize; 3]> = Vec::with_capacity(n_img_p);
    for r in 0..grid.0 {
        for c in 0..grid.1 {
            img_ids.push([l_p + 1, r, c]);
        }
    }
    img_ids.resize(n_img_p, [0, 0, 0]);
    let cap_ids: Vec<[usize; 3]> = (0..l_p).map(|j| [1 + j, 0, 0]).collect();
    // freq tables per axis (f64, torch: 1 / theta ** (arange(0,d,2)/d))
    let freqs: Vec<Vec<f64>> = axes
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|j| 1.0 / theta.powf((2 * j) as f64 / d as f64))
                .collect()
        })
        .collect();
    let table = |ids: &[[usize; 3]]| -> (Vec<f32>, Vec<f32>) {
        let pairs: usize = axes.iter().sum::<usize>() / 2;
        let mut cos = Vec::with_capacity(ids.len() * pairs);
        let mut sin = Vec::with_capacity(ids.len() * pairs);
        for id in ids {
            for (a, f) in freqs.iter().enumerate() {
                for &fr in f {
                    let ang = (id[a] as f64 * fr) as f32;
                    cos.push(ang.cos());
                    sin.push(ang.sin());
                }
            }
        }
        (cos, sin)
    };
    let img = table(&img_ids);
    let cap = table(&cap_ids);
    let joint = (
        [img.0.as_slice(), cap.0.as_slice()].concat(),
        [img.1.as_slice(), cap.1.as_slice()].concat(),
    );
    ZRope { img, joint, cap }
}

/// latent [c, h, w] → tokens [(h/2)·(w/2), 4c], feature (dy·2+dx)·c + ch.
pub fn patchify(latent: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / 2, w / 2);
    let pd = 4 * c;
    let mut out = vec![0f32; hp * wp * pd];
    for r in 0..hp {
        for q in 0..wp {
            let t = r * wp + q;
            for dy in 0..2 {
                for dx in 0..2 {
                    for ch in 0..c {
                        out[t * pd + (dy * 2 + dx) * c + ch] =
                            latent[(ch * h + 2 * r + dy) * w + 2 * q + dx];
                    }
                }
            }
        }
    }
    out
}

/// Exact inverse of `patchify`.
pub fn unpatchify(tok: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / 2, w / 2);
    let pd = 4 * c;
    let mut out = vec![0f32; c * h * w];
    for r in 0..hp {
        for q in 0..wp {
            let t = r * wp + q;
            for dy in 0..2 {
                for dx in 0..2 {
                    for ch in 0..c {
                        out[(ch * h + 2 * r + dy) * w + 2 * q + dx] =
                            tok[t * pd + (dy * 2 + dx) * c + ch];
                    }
                }
            }
        }
    }
    out
}

/// Pad `rows` rows of `width` to `rows_p` by repeating the last row.
pub fn pad_rows_repeat_last(x: &[f32], rows: usize, rows_p: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows_p * width);
    out.extend_from_slice(&x[..rows * width]);
    for _ in rows..rows_p {
        out.extend_from_within((rows - 1) * width..rows * width);
    }
    out
}

/// Weight source for the loaders: a container or a diffusers directory
/// already read into f32.
enum Src<'a> {
    Cmf(&'a Arc<CmfModel>),
    Dir(std::cell::RefCell<HashMap<String, crate::vae::StTensor>>),
}

impl Src<'_> {
    fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        match self {
            Src::Cmf(m) => cmf_f32(m, &format!("dit.{name}")),
            Src::Dir(t) => t
                .borrow_mut()
                .remove(name)
                .map(|v| v.data)
                .ok_or_else(|| format!("missing tensor {name}")),
        }
    }
    fn proj(&self, name: &str) -> Result<(Proj, Option<usize>), String> {
        match self {
            Src::Cmf(m) => {
                let full = format!("dit.{name}");
                let idx = m.tensor_index(&full);
                Ok((Proj::from_model(m, &full)?, idx))
            }
            Src::Dir(t) => {
                let st = t
                    .borrow_mut()
                    .remove(name)
                    .ok_or_else(|| format!("missing tensor {name}"))?;
                if st.shape.len() != 2 {
                    return Err(format!("{name}: expected 2-D"));
                }
                Ok((Proj::f32(st.data, st.shape[1]), None))
            }
        }
    }
}

impl ZImageDit {
    /// Load from a diffusers `transformer/` directory (bf16/fp32
    /// safetensors, read whole and widened to f32 — a dev/parity path).
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let cfg = ZConfig::from_json(&cfg)?;
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("diffusion_pytorch_model.safetensors.index.json"))
                .map_err(|e| format!("index: {e}"))?,
        )
        .map_err(|e| format!("index: {e}"))?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .ok_or("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        let mut t = HashMap::new();
        for sh in &shards {
            t.extend(crate::vae::read_safetensors(&dir.join(sh))?);
        }
        Self::build(cfg, &Src::Dir(std::cell::RefCell::new(t)), None)
    }

    /// Load from a packaged `.cmf` (`dit.*` tensors + `dit.config_json`).
    /// Quantized projections stay mmap-resident.
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("dit.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("dit.config_json: {e}"))?;
        let cfg = ZConfig::from_json(&cfg)?;
        Self::build(cfg, &Src::Cmf(model), Some(model.clone()))
    }

    fn build(cfg: ZConfig, src: &Src, model: Option<Arc<CmfModel>>) -> Result<Self, String> {
        fn block(src: &Src, pfx: &str, modulated: bool) -> Result<ZBlock, String> {
            let (q, iq) = src.proj(&format!("{pfx}.attention.to_q.weight"))?;
            let (k, ik) = src.proj(&format!("{pfx}.attention.to_k.weight"))?;
            let (v, iv) = src.proj(&format!("{pfx}.attention.to_v.weight"))?;
            let (o, io) = src.proj(&format!("{pfx}.attention.to_out.0.weight"))?;
            let (w1, i1) = src.proj(&format!("{pfx}.feed_forward.w1.weight"))?;
            let (w2, i2) = src.proj(&format!("{pfx}.feed_forward.w2.weight"))?;
            let (w3, i3) = src.proj(&format!("{pfx}.feed_forward.w3.weight"))?;
            let idx = match (iq, ik, iv, io, i1, i3, i2) {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g)) => {
                    Some([a, b, c, d, e, f, g])
                }
                _ => None,
            };
            Ok(ZBlock {
                adaln: if modulated {
                    Some((
                        src.f32(&format!("{pfx}.adaLN_modulation.0.weight"))?,
                        src.f32(&format!("{pfx}.adaLN_modulation.0.bias"))?,
                    ))
                } else {
                    None
                },
                norm1: src.f32(&format!("{pfx}.attention_norm1.weight"))?,
                norm2: src.f32(&format!("{pfx}.attention_norm2.weight"))?,
                ffn_norm1: src.f32(&format!("{pfx}.ffn_norm1.weight"))?,
                ffn_norm2: src.f32(&format!("{pfx}.ffn_norm2.weight"))?,
                norm_q: src.f32(&format!("{pfx}.attention.norm_q.weight"))?,
                norm_k: src.f32(&format!("{pfx}.attention.norm_k.weight"))?,
                q,
                k,
                v,
                o,
                w1,
                w2,
                w3,
                idx,
            })
        }
        // A container's blocks load in parallel (B2: the adaLN widening of
        // 34 blocks was a serial 0.5 s of every cold start).
        let blocks = |names: Vec<(String, bool)>| -> Result<Vec<ZBlock>, String> {
            match src {
                Src::Cmf(m) => std::thread::scope(|sc| {
                    let hs: Vec<_> = names
                        .iter()
                        .map(|(pfx, md)| sc.spawn(move || block(&Src::Cmf(m), pfx, *md)))
                        .collect();
                    hs.into_iter()
                        .map(|h| h.join().map_err(|_| "block loader panicked".to_string())?)
                        .collect()
                }),
                _ => names.iter().map(|(pfx, md)| block(src, pfx, *md)).collect(),
            }
        };
        let noise_refiner = blocks((0..cfg.n_refiner).map(|i| (format!("noise_refiner.{i}"), true)).collect())?;
        let context_refiner =
            blocks((0..cfg.n_refiner).map(|i| (format!("context_refiner.{i}"), false)).collect())?;
        let layers = blocks((0..cfg.n_layers).map(|i| (format!("layers.{i}"), true)).collect())?;
        let pk = format!("{}-1", cfg.patch);
        let (cap_w, _) = src.proj("cap_embedder.1.weight")?;
        Ok(Self {
            x_emb_w: src.f32(&format!("all_x_embedder.{pk}.weight"))?,
            x_emb_b: src.f32(&format!("all_x_embedder.{pk}.bias"))?,
            x_pad: src.f32("x_pad_token")?,
            cap_pad: src.f32("cap_pad_token")?,
            t_w0: src.f32("t_embedder.mlp.0.weight")?,
            t_b0: src.f32("t_embedder.mlp.0.bias")?,
            t_w2: src.f32("t_embedder.mlp.2.weight")?,
            t_b2: src.f32("t_embedder.mlp.2.bias")?,
            cap_norm: src.f32("cap_embedder.0.weight")?,
            cap_w,
            cap_b: src.f32("cap_embedder.1.bias")?,
            final_mod_w: src.f32(&format!("all_final_layer.{pk}.adaLN_modulation.1.weight"))?,
            final_mod_b: src.f32(&format!("all_final_layer.{pk}.adaLN_modulation.1.bias"))?,
            final_w: src.f32(&format!("all_final_layer.{pk}.linear.weight"))?,
            final_b: src.f32(&format!("all_final_layer.{pk}.linear.bias"))?,
            noise_refiner,
            context_refiner,
            layers,
            pool: Pool::from_env(),
            cfg,
            model,
        })
    }

    pub fn geom(&self) -> ZGeom {
        self.cfg.geom()
    }

    fn pool(&self) -> Option<&Pool> {
        self.pool.as_deref()
    }

    fn blk(&self, id: ZBlockId) -> &ZBlock {
        match id {
            ZBlockId::NoiseRefiner(i) => &self.noise_refiner[i],
            ZBlockId::ContextRefiner(i) => &self.context_refiner[i],
            ZBlockId::Layer(i) => &self.layers[i],
        }
    }

    /// Timestep embedding for `t_model` = (1000 − 1000σ)/1000: sinusoid of
    /// t·1000 (cos first, f32 args) → mlp.0 → SiLU → mlp.2. Returns [256].
    pub fn temb(&self, t_model: f32) -> Vec<f32> {
        const HALF: usize = 128;
        let t = t_model * self.cfg.t_scale;
        // torch: exp((-ln(1e4) as f32) * arange_f32 / 128) in f32
        let c = -(10000f64.ln()) as f32;
        let mut freq = vec![0f32; 2 * HALF];
        for i in 0..HALF {
            let f = (c * i as f32 / HALF as f32).exp();
            let arg = t * f;
            freq[i] = arg.cos();
            freq[HALF + i] = arg.sin();
        }
        let mut h = linear_row(&freq, &self.t_w0, &self.t_b0, None);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        linear_row(&h, &self.t_w2, &self.t_b2, None)
    }

    /// Raw `adaLN_modulation.0(temb)` for every step and block:
    /// [steps][2 + n_layers][4 · dim], chunks [scale_msa, gate_msa,
    /// scale_mlp, gate_mlp] (no +1, no tanh). f64 accumulation.
    pub fn mods_for_steps(&self, t_models: &[f32]) -> Vec<f32> {
        let per = self.cfg.n_mod_blocks() * 4 * self.cfg.dim;
        let mut out = vec![0f32; t_models.len() * per];
        // Every step's temb against each weight row while the row is in
        // cache (B2: the per-step loop streamed the 0.5 GB of adaLN
        // weights once per step — ~1 s for the base model's 28 steps).
        // Same f64 dot per (row, step), so the values are unchanged.
        let tembs: Vec<Vec<f32>> = t_models.iter().map(|&t| self.temb(t)).collect();
        let mut off = 0;
        for b in self.noise_refiner.iter().chain(&self.layers) {
            let (w, bias) = b.adaln.as_ref().expect("modulated block");
            let rows = bias.len();
            let ys = linear_rows_multi(&tembs, w, bias, self.pool());
            for (si, y) in ys.iter().enumerate() {
                out[si * per + off..si * per + off + rows].copy_from_slice(y);
            }
            off += rows;
        }
        out
    }

    /// 1 + `all_final_layer.2-1.adaLN_modulation.1`(SiLU(temb)) per step:
    /// [steps][dim].
    pub fn final_scale_for_steps(&self, t_models: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(t_models.len() * self.cfg.dim);
        for &t in t_models {
            let te: Vec<f32> = self.temb(t).into_iter().map(silu).collect();
            let m = linear_row(&te, &self.final_mod_w, &self.final_mod_b, self.pool());
            out.extend(m.into_iter().map(|v| 1.0 + v));
        }
        out
    }

    /// Caption features [l, cap_feat_dim] → [l_p, dim]: pad rows are copies
    /// of the last row, RMSNorm(w, 1e-5) → Linear + b, rows ≥ l :=
    /// cap_pad_token.
    pub fn embed_caption(&self, cap_feats: &[f32], l: usize) -> Vec<f32> {
        let (cf, dim) = (self.cfg.cap_feat_dim, self.cfg.dim);
        let l_p = ceil32(l);
        // Pad rows are replaced by cap_pad_token after the embedder, so
        // only the l real rows need the linear.
        let mut xn = vec![0f32; l * cf];
        for (o, src) in xn.chunks_exact_mut(cf).zip(cap_feats.chunks_exact(cf)) {
            rms_norm_into(src, &self.cap_norm, self.cfg.norm_eps as f64, o);
        }
        let mut out = vec![0f32; l_p * dim];
        lin(&self.cap_w, &xn, l, &mut out[..l * dim], self.pool());
        for row in out[..l * dim].chunks_exact_mut(dim) {
            for (v, &b) in row.iter_mut().zip(&self.cap_b) {
                *v += b;
            }
        }
        for row in out[l * dim..].chunks_exact_mut(dim) {
            row.copy_from_slice(&self.cap_pad);
        }
        out
    }

    /// The two unmodulated context-refiner blocks on the host, in place on
    /// `cap` [l_p, dim].
    pub fn refine_caption_cpu(&self, cap: &mut [f32], rope_cap: (&[f32], &[f32])) {
        let n = cap.len() / self.cfg.dim;
        for i in 0..self.context_refiner.len() {
            self.block_cpu(ZBlockId::ContextRefiner(i), cap, n, rope_cap, None);
        }
    }

    /// Per-head attention over [n, nh·hd] q/k/v (qk already normed and
    /// rotated), full bidirectional softmax at 1/√hd, into `attn`.
    fn attention(&self, q_all: &[f32], k_all: &[f32], v_all: &[f32], n: usize, attn: &mut [f32]) {
        let (nh, nkv, hd) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim);
        let hpk = nh / nkv;
        let pool = self.pool();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for hh in 0..nh {
            let kv = hh / hpk;
            {
                let (sq, sk, sv) = (
                    SendRows(qh.as_mut_ptr()),
                    SendRows(kh.as_mut_ptr()),
                    SendRows(vt.as_mut_ptr()),
                );
                pool_rows(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        let qsrc = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                        // SAFETY: workers cover disjoint token ranges.
                        let qd = unsafe { sq.row(p * hd, hd) };
                        for (d, &v) in qsrc.iter().enumerate() {
                            qd[d] = v * scale;
                        }
                        unsafe { sk.row(p * hd, hd) }.copy_from_slice(
                            &k_all[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd],
                        );
                        let vv = &v_all[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd];
                        for (d, &val) in vv.iter().enumerate() {
                            unsafe { sv.set(d * n + p, val) };
                        }
                    }
                });
            }
            host_gemm::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
            {
                let sp = SendRows(scores.as_mut_ptr());
                pool_rows(pool, n, &|lo, hi| {
                    for r in lo..hi {
                        // SAFETY: disjoint rows.
                        softmax_inplace(unsafe { sp.row(r * n, n) });
                    }
                });
            }
            host_gemm::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            let sa = SendRows(attn.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    // SAFETY: disjoint tokens.
                    unsafe { sa.row((p * nh + hh) * hd, hd) }
                        .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
                }
            });
        }
    }

    /// One block on the host, in place on `x` [n, dim]. `m` = the block's
    /// raw modulation [4 · dim] (None = unmodulated: s = 0, gate = 1).
    pub fn block_cpu(
        &self,
        blk: ZBlockId,
        x: &mut [f32],
        n: usize,
        rope: (&[f32], &[f32]),
        m: Option<&[f32]>,
    ) {
        let b = self.blk(blk);
        let (hs, nh, nkv, hd) = (
            self.cfg.dim,
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            self.cfg.head_dim,
        );
        let eps = self.cfg.norm_eps as f64;
        let pool = self.pool();
        let (s_msa, g_msa, s_mlp, g_mlp) = match m {
            Some(m) => (
                Some(&m[..hs]),
                Some(m[hs..2 * hs].iter().map(|v| v.tanh()).collect::<Vec<f32>>()),
                Some(&m[2 * hs..3 * hs]),
                Some(m[3 * hs..4 * hs].iter().map(|v| v.tanh()).collect::<Vec<f32>>()),
            ),
            None => (None, None, None, None),
        };
        // dst = rms(src)·w · (1+s)
        let norm_scaled = |src: &[f32], w: &[f32], s: Option<&[f32]>, dst: &mut [f32]| {
            let sr = SendRows(dst.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    // SAFETY: disjoint rows.
                    let row = unsafe { sr.row(p * hs, hs) };
                    rms_norm_into(&src[p * hs..(p + 1) * hs], w, eps, row);
                    if let Some(s) = s {
                        for (r, &sc) in row.iter_mut().zip(s) {
                            *r *= 1.0 + sc;
                        }
                    }
                }
            });
        };
        // x += gate ⊙ rms(src)·w
        let residual = |src: &[f32], w: &[f32], gate: Option<&[f32]>, x: &mut [f32]| {
            let sr = SendRows(x.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                let mut tmp = vec![0f32; hs];
                for p in lo..hi {
                    rms_norm_into(&src[p * hs..(p + 1) * hs], w, eps, &mut tmp);
                    // SAFETY: disjoint rows.
                    let dst = unsafe { sr.row(p * hs, hs) };
                    match gate {
                        Some(g) => {
                            for ((d, &v), &gt) in dst.iter_mut().zip(&tmp).zip(g) {
                                *d += gt * v;
                            }
                        }
                        None => {
                            for (d, &v) in dst.iter_mut().zip(&tmp) {
                                *d += v;
                            }
                        }
                    }
                }
            });
        };
        // ── attention ──
        let mut xn = vec![0f32; n * hs];
        norm_scaled(x, &b.norm1, s_msa, &mut xn);
        let mut q_all = vec![0f32; n * nh * hd];
        let mut k_all = vec![0f32; n * nkv * hd];
        let mut v_all = vec![0f32; n * nkv * hd];
        lin(&b.q, &xn, n, &mut q_all, pool);
        lin(&b.k, &xn, n, &mut k_all, pool);
        lin(&b.v, &xn, n, &mut v_all, pool);
        let (cos, sin) = rope;
        let pairs = hd / 2;
        for (all, heads, w) in [(&mut q_all, nh, &b.norm_q), (&mut k_all, nkv, &b.norm_k)] {
            let sr = SendRows(all.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    for h in 0..heads {
                        // SAFETY: disjoint tokens.
                        let v = unsafe { sr.row((p * heads + h) * hd, hd) };
                        rms_norm_inplace(v, w, eps);
                        for j in 0..pairs {
                            let (c, s) = (cos[p * pairs + j], sin[p * pairs + j]);
                            let (a, bb) = (v[2 * j], v[2 * j + 1]);
                            v[2 * j] = a * c - bb * s;
                            v[2 * j + 1] = a * s + bb * c;
                        }
                    }
                }
            });
        }
        let mut attn = vec![0f32; n * nh * hd];
        self.attention(&q_all, &k_all, &v_all, n, &mut attn);
        drop((q_all, k_all, v_all));
        let mut proj = vec![0f32; n * hs];
        lin(&b.o, &attn, n, &mut proj, pool);
        drop(attn);
        residual(&proj, &b.norm2, g_msa.as_deref(), x);
        // ── SwiGLU FFN ──
        norm_scaled(x, &b.ffn_norm1, s_mlp, &mut xn);
        let inter = b.w1.rows();
        let mut g_all = vec![0f32; n * inter];
        let mut u_all = vec![0f32; n * inter];
        lin(&b.w1, &xn, n, &mut g_all, pool);
        lin(&b.w3, &xn, n, &mut u_all, pool);
        {
            let sg = SendRows(g_all.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    // SAFETY: disjoint tokens.
                    let g = unsafe { sg.row(p * inter, inter) };
                    for (gv, &uv) in g.iter_mut().zip(&u_all[p * inter..(p + 1) * inter]) {
                        *gv = silu(*gv) * uv;
                    }
                }
            });
        }
        drop(u_all);
        lin(&b.w2, &g_all, n, &mut proj, pool);
        residual(&proj, &b.ffn_norm2, g_mlp.as_deref(), x);
    }

    /// Device views of all blocks (requires `from_cmf`).
    pub fn block_refs(&self) -> Option<ZBlockRefs<'_>> {
        fn r(b: &ZBlock) -> Option<ZBlockRef<'_>> {
            let [wq, wk, wv, wo, w1, w3, w2] = b.idx?;
            Some(ZBlockRef {
                wq,
                wk,
                wv,
                wo,
                w1,
                w3,
                w2,
                norm1: &b.norm1,
                norm2: &b.norm2,
                ffn_norm1: &b.ffn_norm1,
                ffn_norm2: &b.ffn_norm2,
                norm_q: &b.norm_q,
                norm_k: &b.norm_k,
            })
        }
        self.model.as_ref()?;
        Some(ZBlockRefs {
            noise_refiner: self.noise_refiner.iter().map(r).collect::<Option<_>>()?,
            context_refiner: self.context_refiner.iter().map(r).collect::<Option<_>>()?,
            layers: self.layers.iter().map(r).collect::<Option<_>>()?,
        })
    }

    /// Once per (prompt, resolution): caption embed → context refiner
    /// (`gpu::zimage_refine_caption`, else CPU) → rope tables →
    /// `gpu::zimage_prepare` (sets `ZPrepared::device`). `mods_all` =
    /// (mods_for_steps, final_scale_for_steps) forwarded to the backend.
    pub fn prepare(
        &self,
        cap_feats: &[f32],
        shape: ZShape,
        key: u64,
        mods_all: Option<(&[f32], &[f32])>,
    ) -> Result<ZPrepared, String> {
        self.prepare_with(cap_feats, shape, key, mods_all, gpu_allowed())
    }

    /// `prepare` with the device use explicit: `device = false` builds a
    /// pure host state (CPU context refiner, no `gpu::zimage_prepare`) —
    /// the reference a device test diffs `step` against.
    pub fn prepare_with(
        &self,
        cap_feats: &[f32],
        shape: ZShape,
        key: u64,
        mods_all: Option<(&[f32], &[f32])>,
        device: bool,
    ) -> Result<ZPrepared, String> {
        let mut p = self.prepare_host(cap_feats, shape, key, device)?;
        if device {
            self.attach_device(&mut p, mods_all);
        }
        Ok(p)
    }

    /// The host half of `prepare`: caption embed → context refiner
    /// (`gpu::zimage_refine_caption` when `device_refine`, else CPU) →
    /// rope tables. `device` is false until `attach_device`.
    pub fn prepare_host(
        &self,
        cap_feats: &[f32],
        shape: ZShape,
        key: u64,
        device_refine: bool,
    ) -> Result<ZPrepared, String> {
        if cap_feats.len() != shape.l * self.cfg.cap_feat_dim || shape.l == 0 {
            return Err(format!(
                "caption features: {} floats for {} tokens of {}",
                cap_feats.len(),
                shape.l,
                self.cfg.cap_feat_dim
            ));
        }
        let rope = ids_and_rope(shape.grid, shape.l, self.cfg.rope_theta, self.cfg.axes_dims);
        let mut cap = self.embed_caption(cap_feats, shape.l);
        let refs = if device_refine { self.block_refs() } else { None };
        let geom = self.geom();
        let dev_refined = match (&refs, &self.model) {
            (Some(r), Some(m)) => crate::gpu::zimage_refine_caption(
                m,
                &geom,
                &r.context_refiner,
                (&rope.cap.0, &rope.cap.1),
                &mut cap,
            ),
            _ => false,
        };
        if !dev_refined {
            self.refine_caption_cpu(&mut cap, (&rope.cap.0, &rope.cap.1));
        }
        Ok(ZPrepared {
            key,
            shape,
            cap,
            rope,
            device: false,
        })
    }

    fn prepare_args<'a>(
        &'a self,
        m: &'a Arc<CmfModel>,
        r: &'a ZBlockRefs<'a>,
        p: &'a ZPrepared,
        key: u64,
        mods_all: Option<(&'a [f32], &'a [f32])>,
        neg: Option<&'a ZPrepared>,
    ) -> crate::gpu::ZPrepareArgs<'a> {
        let shape = p.shape;
        crate::gpu::ZPrepareArgs {
            model: m,
            geom: self.geom(),
            key,
            n_img: shape.n_img,
            n_img_p: shape.n_img_p,
            n_cap_p: shape.l_p,
            grid: shape.grid,
            cap: &p.cap,
            rope_img: (&p.rope.img.0, &p.rope.img.1),
            rope_joint: (&p.rope.joint.0, &p.rope.joint.1),
            x_emb_w: &self.x_emb_w,
            x_emb_b: &self.x_emb_b,
            x_pad: &self.x_pad,
            final_w: &self.final_w,
            final_b: &self.final_b,
            noise_refiner: &r.noise_refiner,
            layers: &r.layers,
            mods_all: mods_all.map(|m| m.0),
            final_scale_all: mods_all.map(|m| m.1),
            neg: neg.map(|n| crate::gpu::ZNegArgs {
                cap: &n.cap,
                n_cap_p: n.shape.l_p,
                rope_img: (&n.rope.img.0, &n.rope.img.1),
                rope_joint: (&n.rope.joint.0, &n.rope.joint.1),
            }),
        }
    }

    /// Upload every device plane now (`gpu::zimage_preload`); independent
    /// of the caption, so it can run beside the text encoder.
    pub fn preload_device(&self) -> bool {
        match (self.block_refs(), &self.model) {
            (Some(r), Some(m)) => crate::gpu::zimage_preload(
                m,
                &self.geom(),
                &r.noise_refiner,
                &r.layers,
                &r.context_refiner,
            ),
            _ => false,
        }
    }

    /// `gpu::zimage_prepare` for a host state (sets `p.device`).
    pub fn attach_device(&self, p: &mut ZPrepared, mods_all: Option<(&[f32], &[f32])>) -> bool {
        let refs = self.block_refs();
        p.device = match (&refs, &self.model) {
            (Some(r), Some(m)) => {
                crate::gpu::zimage_prepare(&self.prepare_args(m, r, p, p.key, mods_all, None))
            }
            _ => false,
        };
        p.device
    }

    /// One batch-2 device program for a CFG pair (item 0 = `pos`, item 1 =
    /// `neg`, both at the same resolution) under `key`. `false` = the
    /// backend has no batch 2 here; the caller steps the items one by one.
    pub fn attach_device_pair(
        &self,
        pos: &ZPrepared,
        neg: &ZPrepared,
        key: u64,
        mods_all: Option<(&[f32], &[f32])>,
    ) -> bool {
        if pos.shape.grid != neg.shape.grid {
            return false;
        }
        let refs = self.block_refs();
        match (&refs, &self.model) {
            (Some(r), Some(m)) => {
                crate::gpu::zimage_prepare(&self.prepare_args(m, r, pos, key, mods_all, Some(neg)))
            }
            _ => false,
        }
    }

    /// Both items of a CFG pair prepared by `attach_device_pair` under
    /// `key`, in one device forward: returns (v_pos, v_neg), or None when
    /// the backend declined (the caller steps the items separately).
    pub fn step_pair_device(
        &self,
        key: u64,
        n_img: usize,
        step: usize,
        x_tok: &[f32],
        mods: &[f32],
        final_scale: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let pd = self.geom().patch_dim;
        let mut out = vec![0f32; n_img * pd];
        let mut out_neg = vec![0f32; n_img * pd];
        crate::gpu::zimage_step(&mut crate::gpu::ZStepArgs {
            key,
            step,
            x_tok,
            mods,
            final_scale,
            out: &mut out,
            out_neg: Some(&mut out_neg),
        })
        .then_some((out, out_neg))
    }

    /// One DiT forward: `gpu::zimage_step` if `p.device`, else `step_cpu`.
    /// `x_tok` [n_img_p, 64] (pad rows = copies of the last row), `mods`
    /// [(2 + n_layers) · 4 · dim] of this step, `final_scale` [dim].
    /// Returns v [n_img, 64] (before the pipeline's negation).
    pub fn step(
        &self,
        p: &ZPrepared,
        step: usize,
        x_tok: &[f32],
        mods: &[f32],
        final_scale: &[f32],
    ) -> Vec<f32> {
        if p.device {
            let mut out = vec![0f32; p.shape.n_img * self.geom().patch_dim];
            if crate::gpu::zimage_step(&mut crate::gpu::ZStepArgs {
                key: p.key,
                step,
                x_tok,
                mods,
                final_scale,
                out: &mut out,
                out_neg: None,
            }) {
                return out;
            }
        }
        self.step_cpu(p, x_tok, mods, final_scale)
    }

    /// The step input for a latent [16, h_lat, w_lat]: patchify, then pad
    /// to n_img_p rows by repeating the last row.
    pub fn tokens(&self, latent: &[f32], shape: &ZShape) -> Vec<f32> {
        pad_rows_repeat_last(
            &patchify(latent, self.cfg.in_channels, shape.h_lat, shape.w_lat),
            shape.n_img,
            shape.n_img_p,
            self.geom().patch_dim,
        )
    }

    /// x_tok [n_img_p, 64] → [n_img_p, dim]: Linear + b, rows ≥ n_img :=
    /// x_pad_token.
    pub fn embed_image(&self, x_tok: &[f32], n_img: usize, n_img_p: usize) -> Vec<f32> {
        let (dim, pd) = (self.cfg.dim, self.geom().patch_dim);
        let mut x = vec![0f32; n_img_p * dim];
        host_gemm::gemm_nt(
            &x_tok[..n_img * pd],
            &self.x_emb_w,
            &mut x[..n_img * dim],
            n_img,
            pd,
            dim,
            self.pool(),
        );
        for row in x[..n_img * dim].chunks_exact_mut(dim) {
            for (v, &b) in row.iter_mut().zip(&self.x_emb_b) {
                *v += b;
            }
        }
        for row in x[n_img * dim..].chunks_exact_mut(dim) {
            row.copy_from_slice(&self.x_pad);
        }
        x
    }

    /// Final layer on the first `n` rows of `u`: LayerNorm(eps 1e-6, no
    /// affine) · final_scale → Linear(dim → 64) + b.
    pub fn final_layer(&self, u: &[f32], n: usize, final_scale: &[f32]) -> Vec<f32> {
        let (dim, pd) = (self.cfg.dim, self.geom().patch_dim);
        let eps = self.cfg.final_eps as f64;
        let mut y = vec![0f32; n * dim];
        {
            let sy = SendRows(y.as_mut_ptr());
            pool_rows(self.pool(), n, &|lo, hi| {
                for p in lo..hi {
                    let x = &u[p * dim..(p + 1) * dim];
                    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
                    let var =
                        x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
                    let inv = 1.0 / (var + eps).sqrt();
                    // SAFETY: disjoint rows.
                    let d = unsafe { sy.row(p * dim, dim) };
                    for ((o, &v), &s) in d.iter_mut().zip(x).zip(final_scale) {
                        *o = ((v as f64 - mean) * inv) as f32 * s;
                    }
                }
            });
        }
        let mut out = vec![0f32; n * pd];
        host_gemm::gemm_nt(&y, &self.final_w, &mut out, n, dim, pd, self.pool());
        for row in out.chunks_exact_mut(pd) {
            for (v, &b) in row.iter_mut().zip(&self.final_b) {
                *v += b;
            }
        }
        out
    }

    /// The host reference forward (WP2/WP3 gate against this):
    /// embed → pad rows := x_pad_token → noise refiner (img only) →
    /// concat [img, cap] → layers → LayerNorm(1e-6)·final_scale → Linear →
    /// image rows.
    pub fn step_cpu(&self, p: &ZPrepared, x_tok: &[f32], mods: &[f32], final_scale: &[f32]) -> Vec<f32> {
        self.step_cpu_taps(p, x_tok, mods, final_scale, &mut |_, _| {})
    }

    /// `step_cpu` with a tap callback: (name, tensor) at the oracle's tap
    /// points (`x_seq`, `nr{i}_out`, `u_in`, `l{i}_out`, `final_out`).
    pub fn step_cpu_taps(
        &self,
        p: &ZPrepared,
        x_tok: &[f32],
        mods: &[f32],
        final_scale: &[f32],
        tap: &mut dyn FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let s = p.shape;
        let dim = self.cfg.dim;
        let md = 4 * dim;
        let mut x = self.embed_image(x_tok, s.n_img, s.n_img_p);
        tap("x_seq", &x);
        for i in 0..self.noise_refiner.len() {
            let m = &mods[i * md..(i + 1) * md];
            self.block_cpu(
                ZBlockId::NoiseRefiner(i),
                &mut x,
                s.n_img_p,
                (&p.rope.img.0, &p.rope.img.1),
                Some(m),
            );
            tap(&format!("nr{i}_out"), &x);
        }
        x.extend_from_slice(&p.cap);
        let n = s.seq();
        tap("u_in", &x);
        let nr = self.noise_refiner.len();
        for i in 0..self.layers.len() {
            let m = &mods[(nr + i) * md..(nr + i + 1) * md];
            self.block_cpu(
                ZBlockId::Layer(i),
                &mut x,
                n,
                (&p.rope.joint.0, &p.rope.joint.1),
                Some(m),
            );
            tap(&format!("l{i}_out"), &x);
        }
        let out = self.final_layer(&x, s.n_img, final_scale);
        tap("final_out", &out);
        out
    }
}

/// Device paths are off under `CMF_GPU=0` or `CMF_ZIMAGE_GPU=0` (the
/// CPU-exact reference run).
/// The Z-Image device path may run (`CMF_ZIMAGE_GPU=0` / `CMF_GPU=0` off).
pub fn gpu_allowed() -> bool {
    !matches!(std::env::var("CMF_ZIMAGE_GPU").as_deref(), Ok("0"))
        && !matches!(std::env::var("CMF_GPU").as_deref(), Ok("0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_match_the_spec_table() {
        let s = sigmas_torch_f32(8, 3.0);
        let spec = [
            1.0f32,
            0.954545438,
            0.899999976,
            0.833333313,
            0.75,
            0.642857134,
            0.5,
            0.300000012,
            0.0,
        ];
        assert_eq!(s.len(), 9);
        for (a, b) in s.iter().zip(spec) {
            assert!((a - b).abs() <= 1e-7, "{s:?}");
        }
        let s4 = sigmas_torch_f32(4, 3.0);
        for (a, b) in s4.iter().zip([1.0f32, 0.9, 0.75, 0.5, 0.0]) {
            assert!((a - b).abs() <= 1e-6, "{s4:?}");
        }
        assert_eq!(t_model(1.0), 0.0);
        assert_eq!(t_model(0.5), 0.5);
    }

    #[test]
    fn patchify_roundtrip_and_order() {
        let (c, h, w) = (16, 6, 4);
        let lat: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let t = patchify(&lat, c, h, w);
        // token (1, 0), feature (dy=1, dx=0, ch=3) = latent[3, 3, 0]
        assert_eq!(t[(1 * 2) * 64 + (1 * 2) * 16 + 3], lat[(3 * h + 3) * w]);
        assert_eq!(unpatchify(&t, c, h, w), lat);
    }

    #[test]
    fn ids_and_pads() {
        assert_eq!(ceil32(22), 32);
        assert_eq!(ceil32(32), 32);
        assert_eq!(ceil32(33), 64);
        let r = ids_and_rope((2, 3), 5, 256.0, [32, 48, 48]);
        // 6 image tokens → 32 rows, caption 5 → 32 rows, 64 pairs each
        assert_eq!(r.img.0.len(), 32 * 64);
        assert_eq!(r.cap.0.len(), 32 * 64);
        assert_eq!(r.joint.0.len(), 64 * 64);
        // image pad row (id 0,0,0): all angles 0
        assert!(r.img.0[31 * 64..].iter().all(|&c| c == 1.0));
        // first caption row: axis-0 id 1, pair 0 → angle 1
        assert!((r.cap.0[0] - 1f32.cos()).abs() < 1e-7);
        // first image row: axis-0 id l_p+1 = 33
        assert!((r.img.0[0] - 33f32.cos()).abs() < 1e-6);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(
            pad_rows_repeat_last(&x, 2, 4, 2),
            vec![1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }
}
