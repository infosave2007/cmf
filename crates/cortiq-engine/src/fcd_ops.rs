//! FCD polish — hand-rolled forward/backward operators.
//!
//! The training graph is FIXED (docs/RUST_FCD.md), so there is no
//! autograd and no tape: every operator here is a (forward, backward)
//! pair written out by hand, llm.c style. Ops are generic over the
//! minimal `Fp` float trait so the SAME code that trains in f32 is
//! gradchecked in f64 against central finite differences
//! (tests/fcd_gradcheck.rs, rel err < 1e-3).
//!
//! Conventions:
//! - all matrices are row-major; weight matrices use the runtime layout
//!   `[out_dim, in_dim]` (a matvec is out[o] = dot(w_row_o, x));
//! - every backward ACCUMULATES into its output gradient buffers
//!   (`+=`) — callers zero them once per step, residual branches then
//!   just add up naturally;
//! - the attention-weight ops (`attn_head_*`, `nystrom_head_*`) are
//!   meant to run in f64: the certified CPU probe computed the T×T
//!   weight matrices in f64, which also makes raw `exp(±40)` safe with
//!   no flash-shift bookkeeping.

use crate::pool::Pool;

// ─────────────────────────── float trait ───────────────────────────

/// Minimal float abstraction: just enough for the fixed graph. Not a
/// general numeric tower — two impls (f32/f64), no external crates.
pub trait Fp:
    Copy
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::MulAssign
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn maxf(self, o: Self) -> Self;
    fn fromf(x: f64) -> Self;
    fn f64(self) -> f64;
}

impl Fp for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn maxf(self, o: Self) -> Self {
        f32::max(self, o)
    }
    #[inline]
    fn fromf(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn f64(self) -> f64 {
        self as f64
    }
}

impl Fp for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn maxf(self, o: Self) -> Self {
        f64::max(self, o)
    }
    #[inline]
    fn fromf(x: f64) -> Self {
        x
    }
    #[inline]
    fn f64(self) -> f64 {
        self
    }
}

#[inline]
fn dot<F: Fp>(a: &[F], b: &[F]) -> F {
    let mut s = F::ZERO;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

// ─────────────────────── generic matmul (nt) ───────────────────────

/// y[n,m] = x[n,k] · w[m,k]ᵀ (serial reference; the f32 hot path is
/// `gemm_nt` below — same math, blocked + pooled).
pub fn matmul_nt<F: Fp>(x: &[F], w: &[F], y: &mut [F], n: usize, k: usize, m: usize) {
    for i in 0..n {
        let xr = &x[i * k..(i + 1) * k];
        for o in 0..m {
            y[i * m + o] = dot(xr, &w[o * k..(o + 1) * k]);
        }
    }
}

/// dX += dY[n,m] · W[m,k] — the input-gradient half of matmul_nt.
pub fn matmul_nt_dx<F: Fp>(dy: &[F], w: &[F], dx: &mut [F], n: usize, k: usize, m: usize) {
    for i in 0..n {
        let dxr = &mut dx[i * k..(i + 1) * k];
        for o in 0..m {
            let g = dy[i * m + o];
            for (d, wv) in dxr.iter_mut().zip(&w[o * k..(o + 1) * k]) {
                *d += g * *wv;
            }
        }
    }
}

/// dW += dYᵀ[m,n] · X[n,k] — the weight-gradient half of matmul_nt.
pub fn matmul_nt_dw<F: Fp>(dy: &[F], x: &[F], dw: &mut [F], n: usize, k: usize, m: usize) {
    for i in 0..n {
        let xr = &x[i * k..(i + 1) * k];
        for o in 0..m {
            let g = dy[i * m + o];
            for (d, xv) in dw[o * k..(o + 1) * k].iter_mut().zip(xr) {
                *d += g * *xv;
            }
        }
    }
}

// ───────────────────── pooled f32 GEMM hot path ─────────────────────

/// Row block: the block's X stays cache-resident while W streams once
/// per block (a per-row W stream would move gigabytes per matmul).
const GEMM_BLOCK: usize = 128;

/// Pointer wrapper for disjoint parallel writes (same pattern as the
/// pipeline's scatter — workers touch disjoint index ranges).
struct SendMut<T>(*mut T);
unsafe impl<T> Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}
impl<T> SendMut<T> {
    #[inline]
    // Deliberate unsynchronized scatter: pool workers write disjoint
    // ranges in parallel, so returning `&mut` from `&self` is
    // intentional here (same pattern as the pipeline's SendMut).
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// y[n,m] = x[n,k] · w[m,k]ᵀ — parallel over row blocks; bit-identical
/// to `matmul_nt` (disjoint rows, same dot kernel regrouped by NEON).

/// Accelerate CBLAS fast path for the training GEMMs (macOS): the
/// naive pooled kernels below stay as the portable/reference path.
#[cfg(target_os = "macos")]
mod accel {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_ACCEL").map(|v| v != "0").unwrap_or(true))
    }
}

pub fn gemm_nt(x: &[f32], w: &[f32], y: &mut [f32], n: usize, k: usize, m: usize, pool: Option<&Pool>) {
    debug_assert_eq!(x.len(), n * k);
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(y.len(), n * m);
    #[cfg(target_os = "macos")]
    if accel::on() && n * k * m >= 1 << 18 {
        // Y = X · Wᵀ (row-major).
        unsafe {
            accel::cblas_sgemm(
                101, 111, 112, n as i32, m as i32, k as i32, 1.0, x.as_ptr(), k as i32,
                w.as_ptr(), k as i32, 0.0, y.as_mut_ptr(), m as i32,
            );
        }
        return;
    }
    let nb = n.div_ceil(GEMM_BLOCK);
    let block = |r0: usize, r1: usize, y: &mut [f32]| {
        for o in 0..m {
            let wr = &w[o * k..(o + 1) * k];
            for i in r0..r1 {
                y[(i - r0) * m + o] = crate::attention::dot_f32(&x[i * k..(i + 1) * k], wr);
            }
        }
    };
    match pool {
        Some(p) if nb > 1 => {
            let yp = SendMut(y.as_mut_ptr());
            p.run(&|widx, nw| {
                for bi in (widx..nb).step_by(nw) {
                    let (r0, r1) = (bi * GEMM_BLOCK, ((bi + 1) * GEMM_BLOCK).min(n));
                    // SAFETY: blocks are disjoint row ranges of y.
                    let ys = unsafe { yp.slice(r0 * m, (r1 - r0) * m) };
                    block(r0, r1, ys);
                }
            });
        }
        _ => {
            for bi in 0..nb {
                let (r0, r1) = (bi * GEMM_BLOCK, ((bi + 1) * GEMM_BLOCK).min(n));
                block(r0, r1, &mut y[r0 * m..r1 * m]);
            }
        }
    }
}

/// dX += dY[n,m] · W[m,k] — parallel over row blocks (disjoint dX rows).
pub fn gemm_dx(dy: &[f32], w: &[f32], dx: &mut [f32], n: usize, k: usize, m: usize, pool: Option<&Pool>) {
    debug_assert_eq!(dy.len(), n * m);
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(dx.len(), n * k);
    #[cfg(target_os = "macos")]
    if accel::on() && n * k * m >= 1 << 18 {
        // dX += dY · W (row-major, beta = 1 accumulates).
        unsafe {
            accel::cblas_sgemm(
                101, 111, 111, n as i32, k as i32, m as i32, 1.0, dy.as_ptr(), m as i32,
                w.as_ptr(), k as i32, 1.0, dx.as_mut_ptr(), k as i32,
            );
        }
        return;
    }
    let nb = n.div_ceil(GEMM_BLOCK);
    let block = |r0: usize, r1: usize, dxs: &mut [f32]| {
        for o in 0..m {
            let wr = &w[o * k..(o + 1) * k];
            for i in r0..r1 {
                let g = dy[i * m + o];
                if g != 0.0 {
                    crate::attention::axpy_f32(&mut dxs[(i - r0) * k..(i - r0 + 1) * k], wr, g);
                }
            }
        }
    };
    match pool {
        Some(p) if nb > 1 => {
            let dxp = SendMut(dx.as_mut_ptr());
            p.run(&|widx, nw| {
                for bi in (widx..nb).step_by(nw) {
                    let (r0, r1) = (bi * GEMM_BLOCK, ((bi + 1) * GEMM_BLOCK).min(n));
                    // SAFETY: blocks are disjoint row ranges of dx.
                    let dxs = unsafe { dxp.slice(r0 * k, (r1 - r0) * k) };
                    block(r0, r1, dxs);
                }
            });
        }
        _ => {
            for bi in 0..nb {
                let (r0, r1) = (bi * GEMM_BLOCK, ((bi + 1) * GEMM_BLOCK).min(n));
                block(r0, r1, &mut dx[r0 * k..r1 * k]);
            }
        }
    }
}

/// dW += dYᵀ · X — parallel over dW ROW ranges (each worker owns a
/// disjoint slice of output neurons; X is shared read-only).
pub fn gemm_dw(dy: &[f32], x: &[f32], dw: &mut [f32], n: usize, k: usize, m: usize, pool: Option<&Pool>) {
    debug_assert_eq!(dy.len(), n * m);
    debug_assert_eq!(x.len(), n * k);
    debug_assert_eq!(dw.len(), m * k);
    #[cfg(target_os = "macos")]
    if accel::on() && n * k * m >= 1 << 18 {
        // dW += dYᵀ · X (row-major, beta = 1 accumulates).
        unsafe {
            accel::cblas_sgemm(
                101, 112, 111, m as i32, k as i32, n as i32, 1.0, dy.as_ptr(), m as i32,
                x.as_ptr(), k as i32, 1.0, dw.as_mut_ptr(), k as i32,
            );
        }
        return;
    }
    let range = |o0: usize, o1: usize, dws: &mut [f32]| {
        // i-blocked so the X block stays in cache across the o loop.
        let nb = n.div_ceil(GEMM_BLOCK);
        for bi in 0..nb {
            let (r0, r1) = (bi * GEMM_BLOCK, ((bi + 1) * GEMM_BLOCK).min(n));
            for o in o0..o1 {
                let dwr = &mut dws[(o - o0) * k..(o - o0 + 1) * k];
                for i in r0..r1 {
                    let g = dy[i * m + o];
                    if g != 0.0 {
                        crate::attention::axpy_f32(dwr, &x[i * k..(i + 1) * k], g);
                    }
                }
            }
        }
    };
    match pool {
        Some(p) if m >= 8 => {
            let dwp = SendMut(dw.as_mut_ptr());
            p.run(&|widx, nw| {
                let (o0, o1) = (widx * m / nw, (widx + 1) * m / nw);
                if o0 < o1 {
                    // SAFETY: workers own disjoint dW row ranges.
                    let dws = unsafe { dwp.slice(o0 * k, (o1 - o0) * k) };
                    range(o0, o1, dws);
                }
            });
        }
        _ => range(0, m, dw),
    }
}

// ───────────────────────────── SiLU ─────────────────────────────

#[inline]
pub fn silu<F: Fp>(x: F) -> F {
    x / (F::ONE + (-x).exp())
}

/// d silu / dx = σ(x)·(1 + x·(1−σ(x))).
#[inline]
pub fn silu_bwd<F: Fp>(x: F) -> F {
    let s = F::ONE / (F::ONE + (-x).exp());
    s * (F::ONE + x * (F::ONE - s))
}

// ──────────────────────────── RMSNorm ────────────────────────────

/// RMSNorm over `n` rows of width `d = w.len()`:
/// Qwen style y = x̂·w, Gemma style y = x̂·(1+w), x̂ = x/√(mean x² + eps).
/// Sum-of-squares accumulates in f64 (runtime discipline). `inv_out`
/// stores the per-row 1/rms for the backward.
pub fn rmsnorm_fwd<F: Fp>(x: &[F], w: &[F], eps: f64, gemma: bool, y: &mut [F], inv_out: &mut [F]) {
    let d = w.len();
    let n = x.len() / d;
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        let mut ss = 0f64;
        for v in xr {
            ss += v.f64() * v.f64();
        }
        let inv = F::fromf(1.0 / (ss / d as f64 + eps).sqrt());
        inv_out[r] = inv;
        let yr = &mut y[r * d..(r + 1) * d];
        for j in 0..d {
            let weff = if gemma { F::ONE + w[j] } else { w[j] };
            yr[j] = xr[j] * inv * weff;
        }
    }
}

/// RMSNorm backward: through-grad into `dx` (+=) and, when the gain is
/// trainable, gain grad into `dw` (+=). `inv` is the saved 1/rms.
///
/// dx_j = inv·w_j·dy_j − x_j·inv³/d · Σ_i dy_i·w_i·x_i
/// dw_j += dy_j·x_j·inv (identical for both styles: ∂y/∂w = x̂).
pub fn rmsnorm_bwd<F: Fp>(
    x: &[F],
    w: &[F],
    inv: &[F],
    dy: &[F],
    gemma: bool,
    dx: &mut [F],
    mut dw: Option<&mut [F]>,
) {
    let d = w.len();
    let n = x.len() / d;
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        let dyr = &dy[r * d..(r + 1) * d];
        let iv = inv[r];
        let mut s = 0f64;
        for j in 0..d {
            let weff = if gemma { F::ONE + w[j] } else { w[j] };
            s += (dyr[j] * weff * xr[j]).f64();
        }
        let coef = F::fromf(s / d as f64) * iv * iv * iv;
        let dxr = &mut dx[r * d..(r + 1) * d];
        for j in 0..d {
            let weff = if gemma { F::ONE + w[j] } else { w[j] };
            dxr[j] += iv * weff * dyr[j] - xr[j] * coef;
        }
        if let Some(dwv) = dw.as_deref_mut() {
            for j in 0..d {
                dwv[j] += dyr[j] * xr[j] * iv;
            }
        }
    }
}

// ───────────────────────────── RoPE ─────────────────────────────

/// Rotate the first `2·inv_freq.len()` dims of one head vector in place
/// (half-split pairing, same convention as `attention::rope_rotate`).
pub fn rope_fwd<F: Fp>(x: &mut [F], position: usize, inv_freq: &[f64]) {
    let half = inv_freq.len();
    for (i, &freq) in inv_freq.iter().enumerate() {
        let angle = position as f64 * freq;
        let (sin, cos) = (F::fromf(angle.sin()), F::fromf(angle.cos()));
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

/// RoPE through-grad: a rotation's Jacobian is the rotation itself, so
/// dL/dx = R(−θ)·dL/dy — the inverse rotation, in place.
pub fn rope_bwd<F: Fp>(dy: &mut [F], position: usize, inv_freq: &[f64]) {
    let half = inv_freq.len();
    for (i, &freq) in inv_freq.iter().enumerate() {
        let angle = position as f64 * freq;
        let (sin, cos) = (F::fromf(angle.sin()), F::fromf(angle.cos()));
        let g0 = dy[i];
        let g1 = dy[i + half];
        dy[i] = g0 * cos + g1 * sin;
        dy[i + half] = -g0 * sin + g1 * cos;
    }
}

// ─────────────────────── landmark segment means ───────────────────────

/// Contiguous segment means (Nyströmformer landmark recipe), the same
/// integer split (i·t)/m as `nystrom::seg_means` and the torch probes.
pub fn seg_means<F: Fp>(x: &[F], t: usize, d: usize, m: usize, out: &mut [F]) {
    for i in 0..m {
        let (lo, hi) = (i * t / m, (i + 1) * t / m);
        let or = &mut out[i * d..(i + 1) * d];
        for v in or.iter_mut() {
            *v = F::ZERO;
        }
        for j in lo..hi {
            for c in 0..d {
                or[c] += x[j * d + c];
            }
        }
        let inv = F::fromf(1.0 / (hi - lo) as f64);
        for v in or.iter_mut() {
            *v *= inv;
        }
    }
}

/// Segment-mean backward: the mean is linear, so the landmark grad
/// scatters back uniformly over its segment (dx += dl/seg_len).
pub fn seg_means_bwd<F: Fp>(dl: &[F], t: usize, d: usize, m: usize, dx: &mut [F]) {
    for i in 0..m {
        let (lo, hi) = (i * t / m, (i + 1) * t / m);
        let inv = F::fromf(1.0 / (hi - lo) as f64);
        let dlr = &dl[i * d..(i + 1) * d];
        for j in lo..hi {
            for c in 0..d {
                dx[j * d + c] += dlr[c] * inv;
            }
        }
    }
}

// ────────────────── exact causal softmax attention ──────────────────

/// Exact per-head causal attention: out[t] = softmax(q_t·Kᵀ/√d)·V over
/// j ≤ t. `q`,`k` are `[t,d]`, `v` is `[t,dv]`, `out` is `[t,dv]`.
#[allow(clippy::needless_range_loop)] // row[j] pairs with k-row j — indices are the clearer form
pub fn attn_head_fwd<F: Fp>(q: &[F], k: &[F], v: &[F], t: usize, d: usize, dv: usize, out: &mut [F]) {
    let scale = F::fromf(1.0 / (d as f64).sqrt());
    let mut row = vec![F::ZERO; t];
    for ti in 0..t {
        let qr = &q[ti * d..(ti + 1) * d];
        let mut mx = F::fromf(f64::NEG_INFINITY);
        for j in 0..=ti {
            let s = dot(qr, &k[j * d..(j + 1) * d]) * scale;
            row[j] = s;
            mx = mx.maxf(s);
        }
        let mut den = F::ZERO;
        for j in 0..=ti {
            row[j] = (row[j] - mx).exp();
            den += row[j];
        }
        let or = &mut out[ti * dv..(ti + 1) * dv];
        for o in or.iter_mut() {
            *o = F::ZERO;
        }
        for j in 0..=ti {
            let p = row[j] / den;
            for (o, vv) in or.iter_mut().zip(&v[j * dv..(j + 1) * dv]) {
                *o += p * *vv;
            }
        }
    }
}

/// Exact-attention backward (probs recomputed row by row — the trainer
/// is layer-checkpointed, nothing is stored). Standard softmax chain:
/// ds_j = p_j·(dp_j − Σ p·dp), dp_j = dout·v_j.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attn_head_bwd<F: Fp>(
    q: &[F],
    k: &[F],
    v: &[F],
    dout: &[F],
    t: usize,
    d: usize,
    dv: usize,
    dq: &mut [F],
    dk: &mut [F],
    dvv: &mut [F],
) {
    let scale = F::fromf(1.0 / (d as f64).sqrt());
    let mut row = vec![F::ZERO; t];
    for ti in 0..t {
        let qr = &q[ti * d..(ti + 1) * d];
        let mut mx = F::fromf(f64::NEG_INFINITY);
        for j in 0..=ti {
            let s = dot(qr, &k[j * d..(j + 1) * d]) * scale;
            row[j] = s;
            mx = mx.maxf(s);
        }
        let mut den = F::ZERO;
        for j in 0..=ti {
            row[j] = (row[j] - mx).exp();
            den += row[j];
        }
        let dor = &dout[ti * dv..(ti + 1) * dv];
        // dp_j and the softmax dot Σ p·dp in one pass.
        let mut pdp = F::ZERO;
        let mut dp = vec![F::ZERO; ti + 1];
        for j in 0..=ti {
            let p = row[j] / den;
            row[j] = p; // row now holds probabilities
            dp[j] = dot(dor, &v[j * dv..(j + 1) * dv]);
            pdp += p * dp[j];
        }
        let dqr = &mut dq[ti * d..(ti + 1) * d];
        for j in 0..=ti {
            let p = row[j];
            // dV
            for (dvo, o) in dvv[j * dv..(j + 1) * dv].iter_mut().zip(dor) {
                *dvo += p * *o;
            }
            // d logits → dq, dk
            let ds = p * (dp[j] - pdp) * scale;
            let kr = &k[j * d..(j + 1) * d];
            for c in 0..d {
                dqr[c] += ds * kr[c];
            }
            let dkr = &mut dk[j * d..(j + 1) * d];
            for c in 0..d {
                dkr[c] += ds * qr[c];
            }
        }
    }
}

// ─────────────────── Nyström joint attention (matrix form) ───────────────────

/// Nyström joint kernel geometry — matches `nystrom::O1Cfg` semantics.
#[derive(Clone, Copy, Debug)]
pub struct NysCfg {
    /// Landmark budget (m_eff = clamp(t_prefill/8, 4, m), as sealed).
    pub m: usize,
    /// Exact-window width.
    pub w: usize,
    /// Permanent exact sink keys.
    pub sink: usize,
    /// Tokens of each training window that run the EXACT prompt pass
    /// before the O(1) seal. `None` = half the window.
    ///
    /// This is not a free knob: it is the train/serve contract. The
    /// runtime (`nystrom::NystromState::prefill`) freezes landmarks from
    /// the PROMPT ONLY and then serves every later position off that
    /// frozen skeleton, so a trainer that seals from the whole window is
    /// optimizing a model that is never served. The default mirrors
    /// `cortiq ppl --o1`'s own default (`o1_prefill = len/2`), which is
    /// the discipline every published o1 quality number was measured
    /// with — keep them equal or the polish and the gate disagree.
    pub prefill: Option<usize>,
}

impl NysCfg {
    /// Prompt length for a `t`-token window: the seal point. Clamped to
    /// [1, t] so a caller cannot ask for a prefix longer than the window.
    #[inline]
    pub fn prefill_len(&self, t: usize) -> usize {
        self.prefill.unwrap_or(t / 2).clamp(1, t)
    }
}

/// Joint-denominator floor (mirrors the reference probe / runtime).
const NYS_DEN_EPS: f64 = 1e-30;

/// Everything the forward computes that the backward reuses. All T×T
/// buffers — call this with F = f64 (raw exp of real logits overflows
/// f32; the certified CPU probe ran the skeleton in f64).
struct NysGraph<F: Fp> {
    m_eff: usize,
    /// Seal point: rows < tp are exact (the prompt pass), rows ≥ tp are
    /// served off the frozen skeleton.
    tp: usize,
    q_l: Vec<F>,
    k_l: Vec<F>,
    mu: Vec<F>,
    fu: Vec<F>,
    e: Vec<F>,
    fumu: Vec<F>,
    /// Per row: did the AGGREGATE far denominator survive the guard?
    /// False ⇒ this row's far field is dropped entirely (weights and
    /// gradient alike) — see `nys_graph`.
    ///
    /// This replaces the raw skeleton estimate `a`, which the backward
    /// used to carry ONLY to evaluate the per-(t,j) clamp's `[a>0]`
    /// subgradient mask. The shipped operator gates per ROW, so the mask
    /// — and the whole T×T buffer behind it — is gone.
    far_keep: Vec<bool>,
    /// Final joint weights: exp(lg−c) on near, a·e^{−c} on a kept far
    /// field, 0 on a dropped one.
    wmat: Vec<F>,
    c_row: Vec<F>,
    den: Vec<F>,
}

/// Is this row served exactly? Positions before the seal are the
/// runtime's PROMPT PASS: `Pipeline::nll_ids_o1` runs them through
/// ordinary full-KV attention and only then calls `o1_seal`, so the
/// skeleton must not touch them.
#[inline]
fn nys_exact_row(ti: usize, tp: usize) -> bool {
    ti < tp
}

/// near mask (spec §5b): (t−j < W) OR (j < sink), causal.
#[inline]
fn nys_near(ti: usize, j: usize, w: usize, sink: usize) -> bool {
    ti - j < w || j < sink
}

/// Build the joint weight matrix (docs/RUST_FCD.md §2.2). M comes from
/// the RUNTIME's ridge pseudo-inverse and is CONSTANT in backward.
/// `mu_override` freezes M explicitly (gradcheck of the constant-M
/// convention); None recomputes it from the landmarks.
fn nys_graph<F: Fp>(
    q: &[F],
    k: &[F],
    t: usize,
    d: usize,
    cfg: &NysCfg,
    mu_override: Option<&[F]>,
) -> NysGraph<F> {
    let scale = 1.0 / (d as f64).sqrt();
    let fscale = F::fromf(scale);
    // Landmarks are sealed from the PROMPT PREFIX, exactly as
    // `NystromState::prefill` does — including m_eff, which the runtime
    // derives from the prompt length it sealed at, not from how far the
    // sequence later runs.
    let tp = cfg.prefill_len(t);
    let m_eff = (tp / 8).clamp(4, cfg.m);
    let mut q_l = vec![F::ZERO; m_eff * d];
    let mut k_l = vec![F::ZERO; m_eff * d];
    seg_means(&q[..tp * d], tp, d, m_eff, &mut q_l);
    seg_means(&k[..tp * d], tp, d, m_eff, &mut k_l);

    // Au and its ridge pinv in f64 — one m×m solve, constant in backward.
    let mut au = vec![0f64; m_eff * m_eff];
    for i in 0..m_eff {
        for j in 0..m_eff {
            let mut s = 0f64;
            for c in 0..d {
                s += q_l[i * d + c].f64() * k_l[j * d + c].f64();
            }
            au[i * m_eff + j] = (s * scale).exp();
        }
    }
    let mu: Vec<F> = match mu_override {
        Some(m) => m.to_vec(),
        None => crate::nystrom::ridge_pinv(&au, m_eff)
            .iter()
            .map(|&x| F::fromf(x))
            .collect(),
    };

    // Landmark score factors.
    let mut fu = vec![F::ZERO; t * m_eff];
    for ti in 0..t {
        for i in 0..m_eff {
            fu[ti * m_eff + i] =
                (dot(&q[ti * d..(ti + 1) * d], &k_l[i * d..(i + 1) * d]) * fscale).exp();
        }
    }
    let mut e = vec![F::ZERO; m_eff * t];
    for i in 0..m_eff {
        for j in 0..t {
            e[i * t + j] =
                (dot(&q_l[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]) * fscale).exp();
        }
    }
    let mut fumu = vec![F::ZERO; t * m_eff];
    matmul_nt(
        &fu,
        // mu is [m,m] row-major; matmul_nt wants W[m_out, k] rows = muᵀ
        // columns… avoid transposition juggling: do it directly.
        &transpose(&mu, m_eff, m_eff),
        &mut fumu,
        t,
        m_eff,
        m_eff,
    );
    // a[t,j] = Σ_i fumu[t,i]·e[i,j] — e is [m,t], so column j of e is
    // strided; loop with accumulation over i keeps rows contiguous.
    // Rows before the seal are exact, so their skeleton is never
    // evaluated (and stays ZERO — the backward relies on that).
    let mut a = vec![F::ZERO; t * t];
    for ti in tp..t {
        let fr = &fumu[ti * m_eff..(ti + 1) * m_eff];
        let ar = &mut a[ti * t..ti * t + ti + 1]; // causal: j ≤ ti
        for (i, &f) in fr.iter().enumerate() {
            let er = &e[i * t..i * t + ti + 1];
            for (av, ev) in ar.iter_mut().zip(er) {
                *av += f * *ev;
            }
        }
    }

    // Joint weights with the per-row shift c (constant in backward —
    // it multiplies numerator and denominator identically).
    let mut wmat = vec![F::ZERO; t * t];
    let mut c_row = vec![F::ZERO; t];
    let mut den = vec![F::ZERO; t];
    let mut far_keep = vec![false; t];
    let mut lg_row = vec![F::ZERO; t];
    for ti in 0..t {
        let qr = &q[ti * d..(ti + 1) * d];
        let mut c = F::fromf(f64::NEG_INFINITY);
        for j in 0..=ti {
            let s = dot(qr, &k[j * d..(j + 1) * d]) * fscale;
            lg_row[j] = s;
            c = c.maxf(s);
        }
        c_row[ti] = c;
        let emc = (-c).exp();
        let exact_row = nys_exact_row(ti, tp);

        // AGGREGATE guard (`O1Rect::Aggregate`, the shipped default).
        // The runtime never materializes a per-(t,j) weight — it only
        // ever holds the far field already contracted into accumulators,
        // so the ONLY quantity it can test is the row's total far
        // denominator:
        //   far_den = Σ_b u[b]·ẑ[b] = e^{−c_all}·Σ_{j far} a[t,j]
        // (`nystrom.rs::step`). e^{−c_all} > 0, so the runtime's
        // `far_den >= 0` predicate is exactly `Σ_{j far} a[t,j] >= 0`
        // here. A row that fails drops its far field ENTIRELY; a row
        // that passes keeps every far weight RAW — negative per-key mass
        // included. That is not an oversight: clamping per key (what the
        // torch probe did, and what this trainer used to do) is a
        // strictly different, measurably worse operator (`O1Rect::Fm`,
        // ×1.414 vs ×1.296) that no streaming kernel can execute.
        let mut far_sum = F::ZERO;
        if !exact_row {
            for j in 0..=ti {
                if !nys_near(ti, j, cfg.w, cfg.sink) {
                    far_sum += a[ti * t + j];
                }
            }
        }
        let keep = !exact_row && far_sum.f64() >= 0.0;
        far_keep[ti] = keep;

        let wr = &mut wmat[ti * t..(ti + 1) * t];
        let mut dsum = F::ZERO;
        for j in 0..=ti {
            let wv = if exact_row || nys_near(ti, j, cfg.w, cfg.sink) {
                (lg_row[j] - c).exp()
            } else if keep {
                a[ti * t + j] * emc
            } else {
                F::ZERO
            };
            wr[j] = wv;
            dsum += wv;
        }
        den[ti] = dsum.maxf(F::fromf(NYS_DEN_EPS));
    }
    NysGraph { m_eff, tp, q_l, k_l, mu, fu, e, fumu, far_keep, wmat, c_row, den }
}

fn transpose<F: Fp>(x: &[F], rows: usize, cols: usize) -> Vec<F> {
    let mut out = vec![F::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// Nyström joint forward (teacher-forced matrix form). Falls back to
/// exact attention for short windows (runtime guard: t ≤ W+sink+8).
#[allow(clippy::too_many_arguments)]
pub fn nystrom_head_fwd<F: Fp>(
    q: &[F],
    k: &[F],
    v: &[F],
    t: usize,
    d: usize,
    dv: usize,
    cfg: &NysCfg,
    out: &mut [F],
) {
    if nys_degenerate(t, cfg) {
        attn_head_fwd(q, k, v, t, d, dv, out);
        return;
    }
    nystrom_head_fwd_mu(q, k, v, t, d, dv, cfg, None, out);
}

/// Does the runtime skip the skeleton for this window entirely?
///
/// `NystromState::prefill` decides `exact_only` from the PROMPT length
/// (`t <= w + sink + EXACT_SLACK`), not from how long the sequence
/// eventually grows: a short prompt seals no skeleton, so every later
/// key stays a permanent exact key. Keying this off the full window `t`
/// (as this did before) would build a skeleton for windows the runtime
/// serves exactly.
#[inline]
fn nys_degenerate(t: usize, cfg: &NysCfg) -> bool {
    cfg.prefill_len(t) <= cfg.w + cfg.sink + 8
}

/// The landmark mixing matrix M of this (q, k) — gradcheck hook for
/// freezing M across finite-difference perturbations.
#[doc(hidden)]
pub fn nystrom_mu_for_test<F: Fp>(q: &[F], k: &[F], t: usize, d: usize, cfg: &NysCfg) -> Vec<F> {
    nys_graph(q, k, t, d, cfg, None).mu
}

/// Forward with an explicitly frozen M (gradcheck of the constant-M
/// convention; the trainer always passes through `nystrom_head_fwd`).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn nystrom_head_fwd_mu<F: Fp>(
    q: &[F],
    k: &[F],
    v: &[F],
    t: usize,
    d: usize,
    dv: usize,
    cfg: &NysCfg,
    mu_override: Option<&[F]>,
    out: &mut [F],
) {
    let g = nys_graph(q, k, t, d, cfg, mu_override);
    for ti in 0..t {
        let wr = &g.wmat[ti * t..(ti + 1) * t];
        let den = g.den[ti];
        let or = &mut out[ti * dv..(ti + 1) * dv];
        for o in or.iter_mut() {
            *o = F::ZERO;
        }
        for j in 0..=ti {
            let p = wr[j] / den;
            if p.f64() != 0.0 {
                for (o, vv) in or.iter_mut().zip(&v[j * dv..(j + 1) * dv]) {
                    *o += p * *vv;
                }
            }
        }
    }
}

/// Nyström joint backward: dq/dk/dv (+=) with M constant. The whole
/// graph is recomputed (layer checkpointing) — see docs/RUST_FCD.md
/// §2.2 for the derivation. Chain summary:
///   dw[t,j]   = dout_t·(v_j − out_t)/den_t
///   near:  dlg = dw·w                     → dq, dk (±scale)
///   far:   da  = dw·e^{−c}·[row kept]     → dFu, dE through the two
///          matmuls with M constant        → dq, dk, dQ̃, dK̃
///   landmarks: segment-mean scatter back into the PREFIX rows of
///          dq, dk (the seal only saw those).
/// The far gate is per ROW (the aggregate guard), not per key.
#[allow(clippy::too_many_arguments)]
pub fn nystrom_head_bwd<F: Fp>(
    q: &[F],
    k: &[F],
    v: &[F],
    dout: &[F],
    t: usize,
    d: usize,
    dv: usize,
    cfg: &NysCfg,
    dq: &mut [F],
    dk: &mut [F],
    dvv: &mut [F],
) {
    if nys_degenerate(t, cfg) {
        attn_head_bwd(q, k, v, dout, t, d, dv, dq, dk, dvv);
        return;
    }
    nystrom_head_bwd_mu(q, k, v, dout, t, d, dv, cfg, None, dq, dk, dvv);
}

/// Backward with an explicitly frozen M (see `nystrom_head_fwd_mu`).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn nystrom_head_bwd_mu<F: Fp>(
    q: &[F],
    k: &[F],
    v: &[F],
    dout: &[F],
    t: usize,
    d: usize,
    dv: usize,
    cfg: &NysCfg,
    mu_override: Option<&[F]>,
    dq: &mut [F],
    dk: &mut [F],
    dvv: &mut [F],
) {
    let scale = F::fromf(1.0 / (d as f64).sqrt());
    let g = nys_graph(q, k, t, d, cfg, mu_override);
    let m_eff = g.m_eff;

    // Recompute out rows (needed inside dw), then dv and dw in one pass.
    let mut dwmat = vec![F::ZERO; t * t];
    let mut out_row = vec![F::ZERO; dv];
    for ti in 0..t {
        let wr = &g.wmat[ti * t..(ti + 1) * t];
        let den = g.den[ti];
        for o in out_row.iter_mut() {
            *o = F::ZERO;
        }
        for j in 0..=ti {
            let p = wr[j] / den;
            if p.f64() != 0.0 {
                for (o, vv) in out_row.iter_mut().zip(&v[j * dv..(j + 1) * dv]) {
                    *o += p * *vv;
                }
            }
        }
        let dor = &dout[ti * dv..(ti + 1) * dv];
        let dwr = &mut dwmat[ti * t..(ti + 1) * t];
        for j in 0..=ti {
            // dV: p·dout
            let p = wr[j] / den;
            if p.f64() != 0.0 {
                for (dvo, o) in dvv[j * dv..(j + 1) * dv].iter_mut().zip(dor) {
                    *dvo += p * *o;
                }
            }
            // dw = dout·(v_j − out)/den
            let mut s = F::ZERO;
            for c in 0..dv {
                s += dor[c] * (v[j * dv + c] - out_row[c]);
            }
            dwr[j] = s / den;
        }
    }

    // Near half: dlg = dw·w, straight to dq/dk. A pre-seal row is
    // exact, so EVERY causal j is a near key for it.
    for ti in 0..t {
        let qr = &q[ti * d..(ti + 1) * d];
        let dqr_base = ti * d;
        for j in 0..=ti {
            if !(nys_exact_row(ti, g.tp) || nys_near(ti, j, cfg.w, cfg.sink)) {
                continue;
            }
            let dlg = dwmat[ti * t + j] * g.wmat[ti * t + j] * scale;
            if dlg.f64() == 0.0 {
                continue;
            }
            let kr = &k[j * d..(j + 1) * d];
            for c in 0..d {
                dq[dqr_base + c] += dlg * kr[c];
            }
            let dkr = &mut dk[j * d..(j + 1) * d];
            for c in 0..d {
                dkr[c] += dlg * qr[c];
            }
        }
    }

    // Far half: da gated by the AGGREGATE guard, then back through the
    // skeleton.
    //
    // Gradient of the guard: the far field enters as the piecewise
    // function  far(row) = [Σ_far a ≥ 0] · a·e^{−c}, i.e. a per-ROW
    // switch, not a per-key clamp. On the kept branch it is the identity
    // in a — so the far grad flows RAW, with no [a>0] mask (that mask
    // belonged to the per-key clamp this replaces and would now zero the
    // gradient of exactly the negative-mass keys the shipped operator
    // keeps). On the dropped branch the far field is identically 0 in a
    // neighbourhood of a, so its gradient is 0. The switching boundary
    // itself (Σ_far a = 0) is measure-zero and, like any subgradient
    // convention, is not differentiated — at that point far ≡ 0 anyway,
    // so the two branches agree in value and only the derivative jumps.
    let mut da = vec![F::ZERO; t * t];
    for ti in g.tp..t {
        if !g.far_keep[ti] {
            continue; // dropped row: zero gradient through its far field
        }
        let emc = (-g.c_row[ti]).exp();
        for j in 0..=ti {
            if nys_near(ti, j, cfg.w, cfg.sink) {
                continue;
            }
            da[ti * t + j] = dwmat[ti * t + j] * emc;
        }
    }
    // dFuMu[t,i] = Σ_j da[t,j]·e[i,j]
    let mut dfumu = vec![F::ZERO; t * m_eff];
    for ti in 0..t {
        let dar = &da[ti * t..ti * t + ti + 1];
        let dfr = &mut dfumu[ti * m_eff..(ti + 1) * m_eff];
        for (i, df) in dfr.iter_mut().enumerate() {
            let er = &g.e[i * t..i * t + ti + 1];
            let mut s = F::ZERO;
            for (av, ev) in dar.iter().zip(er) {
                s += *av * *ev;
            }
            *df = s;
        }
    }
    // dFu = dFuMu·Muᵀ  (M constant)
    let mut dfu = vec![F::ZERO; t * m_eff];
    matmul_nt(&dfumu, &g.mu, &mut dfu, t, m_eff, m_eff);
    // dE[i,j] = Σ_t fumu[t,i]·da[t,j]
    let mut de = vec![F::ZERO; m_eff * t];
    for ti in 0..t {
        let dar = &da[ti * t..ti * t + ti + 1];
        let fr = &g.fumu[ti * m_eff..(ti + 1) * m_eff];
        for (i, &f) in fr.iter().enumerate() {
            if f.f64() == 0.0 {
                continue;
            }
            let der = &mut de[i * t..i * t + ti + 1];
            for (dev, av) in der.iter_mut().zip(dar) {
                *dev += f * *av;
            }
        }
    }
    // Chain through the two exponentials into dq/dk and the landmark
    // grads dQ̃/dK̃.
    let mut dq_l = vec![F::ZERO; m_eff * d];
    let mut dk_l = vec![F::ZERO; m_eff * d];
    for ti in 0..t {
        let qr = &q[ti * d..(ti + 1) * d];
        for i in 0..m_eff {
            let dlg = dfu[ti * m_eff + i] * g.fu[ti * m_eff + i] * scale;
            if dlg.f64() == 0.0 {
                continue;
            }
            let klr = &g.k_l[i * d..(i + 1) * d];
            for c in 0..d {
                dq[ti * d + c] += dlg * klr[c];
            }
            let dklr = &mut dk_l[i * d..(i + 1) * d];
            for c in 0..d {
                dklr[c] += dlg * qr[c];
            }
        }
    }
    for i in 0..m_eff {
        let qlr = &g.q_l[i * d..(i + 1) * d];
        for j in 0..t {
            let dlg = de[i * t + j] * g.e[i * t + j] * scale;
            if dlg.f64() == 0.0 {
                continue;
            }
            let kr = &k[j * d..(j + 1) * d];
            let dqlr = &mut dq_l[i * d..(i + 1) * d];
            for c in 0..d {
                dqlr[c] += dlg * kr[c];
            }
            for c in 0..d {
                dk[j * d + c] += dlg * qlr[c];
            }
        }
    }
    // Landmarks were sealed from the prompt prefix, so their gradient
    // scatters back into those rows ONLY — a post-seal token cannot move
    // a landmark it never contributed to.
    let tp = g.tp;
    seg_means_bwd(&dq_l, tp, d, m_eff, &mut dq[..tp * d]);
    seg_means_bwd(&dk_l, tp, d, m_eff, &mut dk[..tp * d]);
}

// ───────────────────────────── losses ─────────────────────────────

/// One position of the polish loss (docs/RUST_FCD.md §2.4):
/// L = (1−klw)·CE(student, target) + klw·KL(teacher‖student), both
/// per-position; `inv_n` = 1/(B·T) folds the batch mean into dlogits.
/// Returns (ce, kl) UNWEIGHTED for logging; dlogits gets the combined
/// gradient (+= is NOT used — each position owns its slice).
pub fn ce_kl_position<F: Fp>(
    s_logits: &[F],
    t_logits: &[F],
    target: usize,
    kl_w: f64,
    inv_n: f64,
    dlogits: &mut [F],
) -> (f64, f64) {
    let vsz = s_logits.len();
    debug_assert_eq!(t_logits.len(), vsz);
    // Student log-softmax in f64.
    let mut smax = f64::NEG_INFINITY;
    let mut tmax = f64::NEG_INFINITY;
    for i in 0..vsz {
        smax = smax.max(s_logits[i].f64());
        tmax = tmax.max(t_logits[i].f64());
    }
    let mut ssum = 0f64;
    let mut tsum = 0f64;
    for i in 0..vsz {
        ssum += (s_logits[i].f64() - smax).exp();
        tsum += (t_logits[i].f64() - tmax).exp();
    }
    let slz = smax + ssum.ln();
    let tlz = tmax + tsum.ln();
    let ce = slz - s_logits[target].f64();
    let mut kl = 0f64;
    for i in 0..vsz {
        let ls = s_logits[i].f64() - slz;
        let lt = t_logits[i].f64() - tlz;
        let pt = lt.exp();
        let ps = ls.exp();
        if pt > 0.0 {
            kl += pt * (lt - ls);
        }
        let mut gd = (1.0 - kl_w) * ps + kl_w * (ps - pt);
        if i == target {
            gd -= 1.0 - kl_w;
        }
        dlogits[i] = F::fromf(gd * inv_n);
    }
    (ce, kl)
}

// ─────────────────── GatedDeltaNet through-backward ───────────────────
//
// Faithful BPTT through `linear_core::gdn_step` semantics (docs/
// RUST_FCD.md §3): depthwise causal conv + SiLU, per-group l2-normalized
// q/k, gates g = exp(−exp(A_log)·softplus(a+dt_bias)) and β = σ(b), the
// delta-rule recurrence S ← g·S; kv = Sᵀk̂; S += k̂⊗β(v−kv); o = Sᵀq̂, and
// the gated per-head RMSNorm output x̂·w·silu(z). Through-grad ONLY —
// every GDN weight stays frozen (the FCD policy: attention operators
// are closed-form/frozen, training touches LN+FFN of converted layers).
//
// The backward stores the full per-head state history S_0..S_T (one
// head at a time: T·dk·dv floats — ~67 MB f64 at Qwen3.5-0.8B geometry,
// freed per head). Larger models would switch to segment checkpoints;
// the entry points below don't change for that.

/// GDN geometry + frozen elementwise weights for the sequence ops.
pub struct GdnSeqCfg<'a> {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub rms_eps: f64,
    /// Depthwise conv taps `[c_dim × kk]`, oldest→newest (tap kk−1
    /// multiplies the current position) — `GdnWeights::conv1d` layout.
    pub conv: &'a [f32],
    /// Per-v-head decay parameter A_log `[nv]`.
    pub a_log: &'a [f32],
    /// Per-v-head dt bias `[nv]`.
    pub dt_bias: &'a [f32],
    /// Gated-RMSNorm gain `[dv]` (plain x̂·w, norm-before-gate).
    pub norm: &'a [f32],
}

impl GdnSeqCfg<'_> {
    pub fn c_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
}

#[inline]
fn softplus_f<F: Fp>(x: F) -> F {
    // Same threshold as linear_core::softplus — parity matters more
    // than elegance (σ(20) ≈ 1 − 2e-9, consistent with the cutoff).
    if x.f64() > 20.0 {
        x
    } else {
        F::fromf(x.f64().exp().ln_1p())
    }
}

#[inline]
fn sigmoid_f<F: Fp>(x: F) -> F {
    F::ONE / (F::ONE + (-x).exp())
}

/// Depthwise causal conv + SiLU over the whole window: raw `[t, c_dim]`
/// → (pre-activation `[t, c_dim]`, cq = silu(pre)). Ring semantics of
/// `gdn_step` from a fresh state: positions before 0 are zeros.
pub fn gdn_conv_fwd<F: Fp>(
    raw: &[F],
    t: usize,
    c_dim: usize,
    kk: usize,
    conv: &[f32],
    pre: &mut [F],
    cq: &mut [F],
) {
    for ti in 0..t {
        for c in 0..c_dim {
            let taps = &conv[c * kk..(c + 1) * kk];
            let mut acc = F::ZERO;
            for (j, &tap) in taps.iter().enumerate() {
                // tap j multiplies raw position ti − (kk−1) + j.
                let p = ti as isize - (kk as isize - 1) + j as isize;
                if p >= 0 {
                    acc += raw[p as usize * c_dim + c] * F::fromf(tap as f64);
                }
            }
            pre[ti * c_dim + c] = acc;
            cq[ti * c_dim + c] = silu(acc);
        }
    }
}

/// Conv+SiLU backward: dcq → draw (+=). Taps frozen (through-grad only).
pub fn gdn_conv_bwd<F: Fp>(
    pre: &[F],
    t: usize,
    c_dim: usize,
    kk: usize,
    conv: &[f32],
    dcq: &[F],
    draw: &mut [F],
) {
    for ti in 0..t {
        for c in 0..c_dim {
            let g = dcq[ti * c_dim + c];
            if g.f64() == 0.0 {
                continue;
            }
            let dp = g * silu_bwd(pre[ti * c_dim + c]);
            let taps = &conv[c * kk..(c + 1) * kk];
            for (j, &tap) in taps.iter().enumerate() {
                let p = ti as isize - (kk as isize - 1) + j as isize;
                if p >= 0 {
                    draw[p as usize * c_dim + c] += dp * F::fromf(tap as f64);
                }
            }
        }
    }
}

/// l2-normalization factors of one q/k vector, matching gdn_step:
/// invq = 1/(√(Σq²+1e-6)·√dk), invk = 1/√(Σk²+1e-6).
#[inline]
fn gdn_inv<F: Fp>(x: &[F], extra_scale: f64) -> (F, F) {
    let mut n2 = F::ZERO;
    for v in x {
        n2 += *v * *v;
    }
    let n2e = n2 + F::fromf(1e-6);
    let inv = F::ONE / (n2e.sqrt() * F::fromf(extra_scale));
    (inv, n2e)
}

/// One GQA group (k-head `ko`, its rep = nv/nk v-heads) forward over
/// the window. Writes `out[t, nv·dv]` slices of its v-heads only.
#[allow(clippy::too_many_arguments)]
pub fn gdn_group_fwd<F: Fp>(
    cq: &[F],
    z: &[F],
    a: &[F],
    b: &[F],
    t: usize,
    cfg: &GdnSeqCfg,
    ko: usize,
    out: &mut [F],
) {
    let (nv, nk, dk, dv) = (cfg.nv, cfg.nk, cfg.dk, cfg.dv);
    let c_dim = cfg.c_dim();
    let kd = nk * dk;
    let rep = nv / nk;
    let vd = nv * dv;
    let sqdk = (dk as f64).sqrt();
    for hh in 0..rep {
        let h = ko * rep + hh;
        let ea = F::fromf((cfg.a_log[h] as f64).exp());
        let mut s = vec![F::ZERO; dk * dv];
        let mut kv = vec![F::ZERO; dv];
        let mut o = vec![F::ZERO; dv];
        for ti in 0..t {
            let qrow = &cq[ti * c_dim + ko * dk..ti * c_dim + (ko + 1) * dk];
            let krow = &cq[ti * c_dim + kd + ko * dk..ti * c_dim + kd + (ko + 1) * dk];
            let vrow = &cq[ti * c_dim + 2 * kd + h * dv..ti * c_dim + 2 * kd + (h + 1) * dv];
            let (invq, _) = gdn_inv(qrow, sqdk);
            let (invk, _) = gdn_inv(krow, 1.0);
            let g = (-ea * softplus_f(a[ti * nv + h] + F::fromf(cfg.dt_bias[h] as f64))).exp();
            let beta = sigmoid_f(b[ti * nv + h]);
            // S ← g·S; kv = Sᵀk̂; S += k̂ ⊗ β(v − kv); o = Sᵀq̂.
            for x in kv.iter_mut() {
                *x = F::ZERO;
            }
            for di in 0..dk {
                let kf = krow[di] * invk;
                let row = &mut s[di * dv..(di + 1) * dv];
                for dj in 0..dv {
                    row[dj] *= g;
                    kv[dj] += row[dj] * kf;
                }
            }
            for x in o.iter_mut() {
                *x = F::ZERO;
            }
            for di in 0..dk {
                let kf = krow[di] * invk;
                let qf = qrow[di] * invq;
                let row = &mut s[di * dv..(di + 1) * dv];
                for dj in 0..dv {
                    row[dj] += kf * (vrow[dj] - kv[dj]) * beta;
                    o[dj] += qf * row[dj];
                }
            }
            // Gated per-head RMSNorm: x̂·w·silu(z), norm BEFORE gate.
            let mut ss = 0f64;
            for v in &o {
                ss += v.f64() * v.f64();
            }
            let inv = F::fromf(1.0 / (ss / dv as f64 + cfg.rms_eps).sqrt());
            for dj in 0..dv {
                let zv = z[ti * vd + h * dv + dj];
                out[ti * vd + h * dv + dj] =
                    o[dj] * inv * F::fromf(cfg.norm[dj] as f64) * silu(zv);
            }
        }
    }
}

/// One GQA group backward: BPTT over the window given `dout` rows of
/// this group's v-heads. Accumulates (+=) into dcq (q/k channels of
/// `ko`, v channels of its v-heads), dz, da, db. All weights frozen.
#[allow(clippy::too_many_arguments)]
pub fn gdn_group_bwd<F: Fp>(
    cq: &[F],
    z: &[F],
    a: &[F],
    b: &[F],
    t: usize,
    cfg: &GdnSeqCfg,
    ko: usize,
    dout: &[F],
    dcq: &mut [F],
    dz: &mut [F],
    da: &mut [F],
    db: &mut [F],
) {
    let (nv, nk, dk, dv) = (cfg.nv, cfg.nk, cfg.dk, cfg.dv);
    let c_dim = cfg.c_dim();
    let kd = nk * dk;
    let rep = nv / nk;
    let vd = nv * dv;
    let sqdk = (dk as f64).sqrt();
    for hh in 0..rep {
        let h = ko * rep + hh;
        let ea = F::fromf((cfg.a_log[h] as f64).exp());

        // ── replay forward, keeping the state history + per-step scalars ──
        let mut s_hist = vec![F::ZERO; (t + 1) * dk * dv]; // S_0 = 0
        let mut kv_hist = vec![F::ZERO; t * dv];
        let mut o_hist = vec![F::ZERO; t * dv];
        let mut g_v = vec![F::ZERO; t];
        let mut beta_v = vec![F::ZERO; t];
        let mut sp_arg = vec![F::ZERO; t]; // a + dt_bias (for σ in the chain)
        for ti in 0..t {
            let qrow = &cq[ti * c_dim + ko * dk..ti * c_dim + (ko + 1) * dk];
            let krow = &cq[ti * c_dim + kd + ko * dk..ti * c_dim + kd + (ko + 1) * dk];
            let vrow = &cq[ti * c_dim + 2 * kd + h * dv..ti * c_dim + 2 * kd + (h + 1) * dv];
            let (invq, _) = gdn_inv(qrow, sqdk);
            let (invk, _) = gdn_inv(krow, 1.0);
            let arg = a[ti * nv + h] + F::fromf(cfg.dt_bias[h] as f64);
            let g = (-ea * softplus_f(arg)).exp();
            let beta = sigmoid_f(b[ti * nv + h]);
            sp_arg[ti] = arg;
            g_v[ti] = g;
            beta_v[ti] = beta;
            let (prev, cur) = s_hist.split_at_mut((ti + 1) * dk * dv);
            let sp = &prev[ti * dk * dv..];
            let sn = &mut cur[..dk * dv];
            let kvr = &mut kv_hist[ti * dv..(ti + 1) * dv];
            for di in 0..dk {
                let kf = krow[di] * invk;
                for dj in 0..dv {
                    let dec = sp[di * dv + dj] * g;
                    sn[di * dv + dj] = dec;
                    kvr[dj] += dec * kf;
                }
            }
            let or = &mut o_hist[ti * dv..(ti + 1) * dv];
            for di in 0..dk {
                let kf = krow[di] * invk;
                let qf = qrow[di] * invq;
                for dj in 0..dv {
                    let sv = sn[di * dv + dj] + kf * (vrow[dj] - kvr[dj]) * beta_v[ti];
                    sn[di * dv + dj] = sv;
                    or[dj] += qf * sv;
                }
            }
        }

        // ── reverse sweep ──
        let mut ds = vec![F::ZERO; dk * dv];
        let mut do_o = vec![F::ZERO; dv];
        let mut du = vec![F::ZERO; dv];
        let mut dkv = vec![F::ZERO; dv];
        let mut dqh = vec![F::ZERO; dk];
        let mut dkh = vec![F::ZERO; dk];
        for ti in (0..t).rev() {
            let qrow = &cq[ti * c_dim + ko * dk..ti * c_dim + (ko + 1) * dk];
            let krow = &cq[ti * c_dim + kd + ko * dk..ti * c_dim + kd + (ko + 1) * dk];
            let vrow = &cq[ti * c_dim + 2 * kd + h * dv..ti * c_dim + 2 * kd + (h + 1) * dv];
            let (invq, nq2) = gdn_inv(qrow, sqdk);
            let (invk, nk2) = gdn_inv(krow, 1.0);
            let g = g_v[ti];
            let beta = beta_v[ti];
            let s_t = &s_hist[(ti + 1) * dk * dv..(ti + 2) * dk * dv];
            let s_prev = &s_hist[ti * dk * dv..(ti + 1) * dk * dv];
            let kvr = &kv_hist[ti * dv..(ti + 1) * dv];
            let or = &o_hist[ti * dv..(ti + 1) * dv];

            // 1. Gated RMSNorm output: of = (o·inv)·w·silu(z).
            let mut ss = 0f64;
            for v in or {
                ss += v.f64() * v.f64();
            }
            let inv = F::fromf(1.0 / (ss / dv as f64 + cfg.rms_eps).sqrt());
            let dofr = &dout[ti * vd + h * dv..ti * vd + (h + 1) * dv];
            // dz and the effective-gain through-grad in one pass.
            let mut sdot = 0f64; // Σ dof·weff·o (for the rms through term)
            for dj in 0..dv {
                let zv = z[ti * vd + h * dv + dj];
                let w = F::fromf(cfg.norm[dj] as f64);
                let weff = w * silu(zv);
                sdot += (dofr[dj] * weff * or[dj]).f64();
                dz[ti * vd + h * dv + dj] += dofr[dj] * or[dj] * inv * w * silu_bwd(zv);
            }
            let coef = F::fromf(sdot / dv as f64) * inv * inv * inv;
            for dj in 0..dv {
                let zv = z[ti * vd + h * dv + dj];
                let weff = F::fromf(cfg.norm[dj] as f64) * silu(zv);
                do_o[dj] = inv * weff * dofr[dj] - or[dj] * coef;
            }

            // 2. o = S_tᵀ q̂ → dS += q̂ ⊗ do, dq̂ = S_t·do.
            for x in dqh.iter_mut() {
                *x = F::ZERO;
            }
            for di in 0..dk {
                let qf = qrow[di] * invq;
                let row = &s_t[di * dv..(di + 1) * dv];
                let dsr = &mut ds[di * dv..(di + 1) * dv];
                let mut acc = F::ZERO;
                for dj in 0..dv {
                    dsr[dj] += qf * do_o[dj];
                    acc += row[dj] * do_o[dj];
                }
                dqh[di] = acc;
            }

            // 3. S_t = S_pre + k̂ ⊗ u, u = β(v − kv):
            //    du = dSᵀk̂; dk̂ = dS·u; dβ = du·(v−kv); dv = β·du; dkv = −β·du.
            for x in du.iter_mut() {
                *x = F::ZERO;
            }
            for x in dkh.iter_mut() {
                *x = F::ZERO;
            }
            for di in 0..dk {
                let kf = krow[di] * invk;
                let dsr = &ds[di * dv..(di + 1) * dv];
                let mut acc = F::ZERO;
                for dj in 0..dv {
                    du[dj] += dsr[dj] * kf;
                    acc += dsr[dj] * (vrow[dj] - kvr[dj]) * beta;
                }
                dkh[di] = acc;
            }
            let mut dbeta = F::ZERO;
            for dj in 0..dv {
                dbeta += du[dj] * (vrow[dj] - kvr[dj]);
                // v channels of cq
                dcq[ti * c_dim + 2 * kd + h * dv + dj] += beta * du[dj];
                dkv[dj] = -(beta * du[dj]);
            }

            // 4. kv = S_preᵀk̂ → dS_pre = dS + k̂ ⊗ dkv; dk̂ += S_pre·dkv.
            //    5. S_pre = g·S_{t−1} → dg = ⟨dS_pre, S_{t−1}⟩; carry
            //    dS = g·dS_pre to the previous step. (S_pre = g·s_prev
            //    is rebuilt on the fly.)
            let mut dg = F::ZERO;
            for di in 0..dk {
                let kf = krow[di] * invk;
                let spr = &s_prev[di * dv..(di + 1) * dv];
                let dsr = &mut ds[di * dv..(di + 1) * dv];
                let mut acc = F::ZERO;
                for dj in 0..dv {
                    let dspre = dsr[dj] + kf * dkv[dj];
                    acc += (spr[dj] * g) * dkv[dj];
                    dg += dspre * spr[dj];
                    dsr[dj] = g * dspre;
                }
                dkh[di] += acc;
            }

            // 6. Gates: g = exp(−e_A·softplus(arg)), β = σ(b).
            let sig = sigmoid_f(sp_arg[ti]);
            da[ti * nv + h] += dg * g * (-ea) * sig;
            db[ti * nv + h] += dbeta * beta * (F::ONE - beta);

            // 7. l2-norm through-grads into the shared q/k channels.
            //    q̂ = q·invq with invq = 1/(√(Σq²+eps)·√dk):
            //    dq = invq·dq̂ − q·(dq̂·q)·invq/(Σq²+eps).
            let mut qdot = F::ZERO;
            let mut kdot = F::ZERO;
            for di in 0..dk {
                qdot += dqh[di] * qrow[di];
                kdot += dkh[di] * krow[di];
            }
            for di in 0..dk {
                dcq[ti * c_dim + ko * dk + di] +=
                    invq * dqh[di] - qrow[di] * qdot * invq / nq2;
                dcq[ti * c_dim + kd + ko * dk + di] +=
                    invk * dkh[di] - krow[di] * kdot * invk / nk2;
            }
        }
    }
}

/// Whole-layer GDN sequence forward (serial over groups): raw
/// projections → out `[t, nv·dv]`. The trainer parallelizes groups on
/// the pool with the same `gdn_group_*` entry points.
pub fn gdn_seq_fwd<F: Fp>(
    qkv: &[F],
    z: &[F],
    a: &[F],
    b: &[F],
    t: usize,
    cfg: &GdnSeqCfg,
    out: &mut [F],
) {
    let c_dim = cfg.c_dim();
    let mut pre = vec![F::ZERO; t * c_dim];
    let mut cq = vec![F::ZERO; t * c_dim];
    gdn_conv_fwd(qkv, t, c_dim, cfg.kk, cfg.conv, &mut pre, &mut cq);
    for ko in 0..cfg.nk {
        gdn_group_fwd(&cq, z, a, b, t, cfg, ko, out);
    }
}

/// Whole-layer GDN sequence backward: through-grads (+=) into the four
/// projection streams.
#[allow(clippy::too_many_arguments)]
pub fn gdn_seq_bwd<F: Fp>(
    qkv: &[F],
    z: &[F],
    a: &[F],
    b: &[F],
    t: usize,
    cfg: &GdnSeqCfg,
    dout: &[F],
    dqkv: &mut [F],
    dz: &mut [F],
    da: &mut [F],
    db: &mut [F],
) {
    let c_dim = cfg.c_dim();
    let mut pre = vec![F::ZERO; t * c_dim];
    let mut cq = vec![F::ZERO; t * c_dim];
    gdn_conv_fwd(qkv, t, c_dim, cfg.kk, cfg.conv, &mut pre, &mut cq);
    let mut dcq = vec![F::ZERO; t * c_dim];
    for ko in 0..cfg.nk {
        gdn_group_bwd(&cq, z, a, b, t, cfg, ko, dout, &mut dcq, dz, da, db);
    }
    gdn_conv_bwd(&pre, t, c_dim, cfg.kk, cfg.conv, &dcq, dqkv);
}
