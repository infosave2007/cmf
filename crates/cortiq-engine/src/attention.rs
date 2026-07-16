//! Attention forward pass — GQA with RoPE, head masking, head-major KV.
//!
//! Two entry points:
//! - `multi_head_attention` — the historical f32-slice path with per-head
//!   masking (task masks); untouched, exercised by masked models.
//! - `qwen_attention` / `qwen_attention_pair` — the dense QTensor path:
//!   quantized-from-mmap weights, optional Qwen3.5 extras (per-head
//!   qk-norm, output gate, partial rotary). With extras off and f32
//!   weights the math is identical to the historical path.

use crate::kv_cache::LayerKvCache;
use crate::pool::Pool;
use crate::qtensor::QTensor;

/// Precompute RoPE inverse frequencies for a head dimension — powf is
/// paid once per model, not per (head × position × dim) in the hot loop.
pub fn rope_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// Rotate one vector in place (RoPE, half-split pairing as in Llama/Qwen).
pub fn rope_rotate(x: &mut [f32], position: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    for (i, &freq) in inv_freq.iter().enumerate() {
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

/// Single-head attention: softmax(Q·Kᵀ/√d)·V over a contiguous cache.
/// `k_cache`/`v_cache`: `[seq_len × head_dim]`.
/// Returns `([head_dim] output, [seq_len] attention probabilities)` —
/// the probabilities feed Born-importance accumulation for eviction.
pub fn attention_head(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // These two loops are the decode-attention hot path: cost grows
    // linearly with the stored context, so they are NEON-vectorized
    // (regrouped summation only — same products).
    let mut scores = vec![0.0f32; seq_len];
    for s in 0..seq_len {
        let k = &k_cache[s * head_dim..(s + 1) * head_dim];
        scores[s] = dot_f32(q, k) * scale;
    }

    // Numerically stable softmax.
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }

    let mut output = vec![0.0f32; head_dim];
    for s in 0..seq_len {
        let w = scores[s];
        if w.abs() < 1e-12 {
            continue;
        }
        let v = &v_cache[s * head_dim..(s + 1) * head_dim];
        axpy_f32(&mut output, v, w);
    }
    (output, scores)
}

/// f32 dot with 4 independent accumulators — NEON on aarch64, scalar
/// elsewhere. Same products as the sequential loop, regrouped sums.
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_f32_neon(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::qtensor::avx2_enabled() {
        return unsafe { dot_f32_avx2(a, b) };
    }
    #[allow(unreachable_code)]
    {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

/// f32 dot via AVX2/FMA (x86 mirror of `dot_f32_neon`; regrouped sums).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: callers pass equal-length slices.
    unsafe {
        use core::arch::x86_64::*;
        let n = a.len().min(b.len());
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut j = 0usize;
        while j + 16 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)), _mm256_loadu_ps(bp.add(j)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(j + 8)),
                _mm256_loadu_ps(bp.add(j + 8)),
                s1,
            );
            j += 16;
        }
        let acc = _mm256_add_ps(s0, s1);
        let hi = _mm256_extractf128_ps::<1>(acc);
        let q = _mm_add_ps(_mm256_castps256_ps128(acc), hi);
        let d = _mm_add_ps(q, _mm_movehl_ps(q, q));
        let s = _mm_add_ss(d, _mm_shuffle_ps::<1>(d, d));
        let mut sum = _mm_cvtss_f32(s);
        while j < n {
            sum += *ap.add(j) * *bp.add(j);
            j += 1;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: callers pass equal-length slices (head_dim rows).
    unsafe {
        use core::arch::aarch64::*;
        let n = a.len().min(b.len());
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) =
            (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut j = 0usize;
        while j + 16 <= n {
            a0 = vfmaq_f32(a0, vld1q_f32(ap.add(j)), vld1q_f32(bp.add(j)));
            a1 = vfmaq_f32(a1, vld1q_f32(ap.add(j + 4)), vld1q_f32(bp.add(j + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(ap.add(j + 8)), vld1q_f32(bp.add(j + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(ap.add(j + 12)), vld1q_f32(bp.add(j + 12)));
            j += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while j < n {
            sum += *ap.add(j) * *bp.add(j);
            j += 1;
        }
        sum
}
}

/// `acc += w · row` (f32 axpy) — NEON on aarch64, scalar elsewhere.
#[inline]
pub(crate) fn axpy_f32(acc: &mut [f32], row: &[f32], w: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return axpy_f32_neon(acc, row, w);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::qtensor::avx2_enabled() {
        return unsafe { axpy_f32_avx2(acc, row, w) };
    }
    #[allow(unreachable_code)]
    {
        for (a, &r) in acc.iter_mut().zip(row) {
            *a += w * r;
        }
    }
}

/// f32 axpy via AVX2/FMA (x86 mirror of `axpy_f32_neon`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(acc: &mut [f32], row: &[f32], w: f32) {
    // SAFETY: callers pass equal-length slices (head_dim rows).
    unsafe {
        use core::arch::x86_64::*;
        let n = acc.len().min(row.len());
        let ap = acc.as_mut_ptr();
        let rp = row.as_ptr();
        let wv = _mm256_set1_ps(w);
        let mut j = 0usize;
        while j + 8 <= n {
            let v = _mm256_fmadd_ps(wv, _mm256_loadu_ps(rp.add(j)), _mm256_loadu_ps(ap.add(j)));
            _mm256_storeu_ps(ap.add(j), v);
            j += 8;
        }
        while j < n {
            *ap.add(j) += w * *rp.add(j);
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_f32_neon(acc: &mut [f32], row: &[f32], w: f32) {
    // SAFETY: callers pass equal-length slices (head_dim rows).
    unsafe {
        use core::arch::aarch64::*;
        let n = acc.len().min(row.len());
        let ap = acc.as_mut_ptr();
        let rp = row.as_ptr();
        let wv = vdupq_n_f32(w);
        let mut j = 0usize;
        while j + 4 <= n {
            let v = vfmaq_f32(vld1q_f32(ap.add(j)), wv, vld1q_f32(rp.add(j)));
            vst1q_f32(ap.add(j), v);
            j += 4;
        }
        while j < n {
            *ap.add(j) += w * *rp.add(j);
            j += 1;
        }
}
}

/// Multi-head GQA attention for one position.
///
/// - `active_heads[h]` — Q-head mask; a KV group whose Q heads are ALL
///   dead is neither projected nor cached (GQA-skip: no FLOPs, no memory).
/// - KV cache is head-major: per-head reads are contiguous slices,
///   no per-head gather copies.
///
/// Weights: `wq [num_heads·head_dim, hidden]`, `wk/wv [num_kv·head_dim, hidden]`,
/// `wo [hidden, num_heads·head_dim]`. Returns `[hidden_size]`.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    hidden: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cache: &mut LayerKvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    position: usize,
    active_heads: &[bool],
    inv_freq: &[f32],
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let head_alive =
        |h: usize| -> bool { active_heads.get(h).copied().unwrap_or(true) };
    // A KV group lives while at least one of its Q heads lives.
    let group_alive: Vec<bool> = (0..num_kv_heads)
        .map(|g| (0..heads_per_kv).any(|i| head_alive(g * heads_per_kv + i)))
        .collect();

    // ── Q projection (live heads only) ──
    let mut q_all = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        if !head_alive(h) {
            continue;
        }
        for d in 0..head_dim {
            let row = (h * head_dim + d) * hidden_size;
            let mut sum = 0.0f32;
            for j in 0..hidden_size {
                sum += wq[row + j] * hidden[j];
            }
            q_all[h * head_dim + d] = sum;
        }
        rope_rotate(&mut q_all[h * head_dim..(h + 1) * head_dim], position, inv_freq);
    }

    // ── K/V projection (live groups only) ──
    let mut k_new = vec![0.0f32; num_kv_heads * head_dim];
    let mut v_new = vec![0.0f32; num_kv_heads * head_dim];
    for g in 0..num_kv_heads {
        if !group_alive[g] {
            continue;
        }
        for d in 0..head_dim {
            let row = (g * head_dim + d) * hidden_size;
            let (mut ks, mut vs) = (0.0f32, 0.0f32);
            for j in 0..hidden_size {
                ks += wk[row + j] * hidden[j];
                vs += wv[row + j] * hidden[j];
            }
            k_new[g * head_dim + d] = ks;
            v_new[g * head_dim + d] = vs;
        }
        rope_rotate(&mut k_new[g * head_dim..(g + 1) * head_dim], position, inv_freq);
    }

    cache.append(&k_new, &v_new, &group_alive);

    // ── Per-head attention over contiguous head-major slices ──
    let mut attn_out = vec![0.0f32; num_heads * head_dim];
    let mut imp = vec![0.0f32; cache.seq_len];
    for h in 0..num_heads {
        if !head_alive(h) {
            continue; // dead head contributes zeros
        }
        let g = h / heads_per_kv;
        let stored = cache.head_len(g);
        if stored == 0 {
            continue;
        }
        let _ = stored;
        let (out, probs) = cache.attend(&q_all[h * head_dim..(h + 1) * head_dim], g);
        attn_out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
        for (dst, &p) in imp.iter_mut().zip(&probs) {
            *dst += p;
        }
    }
    // Born rule: a position's importance is the probability mass that
    // reads it — accumulated for importance-aware eviction.
    cache.accumulate_imp(&imp);

    // ── Output projection ──
    let mut output = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        let mut sum = 0.0f32;
        let row = i * num_heads * head_dim;
        for j in 0..(num_heads * head_dim) {
            sum += wo[row + j] * attn_out[j];
        }
        output[i] = sum;
    }
    output
}

/// Fused two-position GQA attention: Q/K/V/O weight rows are streamed
/// from memory once for both positions; the attention itself runs
/// sequentially (position p first — its K/V must be in the cache before
/// position p+1 attends). Dense-only (no head mask): the speculative
/// path uses it for draft verification. Bit-identical to two calls of
/// `multi_head_attention`.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_pair(
    hidden1: &[f32],
    hidden2: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cache: &mut LayerKvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    position: usize,
    inv_freq: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let heads_per_kv = num_heads / num_kv_heads;

    // ── Fused projections: each weight row read once, two dots ──
    let qk_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mut q1 = vec![0.0f32; qk_dim];
    let mut q2 = vec![0.0f32; qk_dim];
    let mut k1 = vec![0.0f32; kv_dim];
    let mut k2 = vec![0.0f32; kv_dim];
    let mut v1 = vec![0.0f32; kv_dim];
    let mut v2 = vec![0.0f32; kv_dim];
    let proj2 = |w: &[f32], o1: &mut [f32], o2: &mut [f32]| {
        for (o, (d1, d2)) in o1.iter_mut().zip(o2.iter_mut()).enumerate() {
            let row = &w[o * hidden_size..(o + 1) * hidden_size];
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for j in 0..hidden_size {
                s1 += row[j] * hidden1[j];
                s2 += row[j] * hidden2[j];
            }
            *d1 = s1;
            *d2 = s2;
        }
    };
    proj2(wq, &mut q1, &mut q2);
    proj2(wk, &mut k1, &mut k2);
    proj2(wv, &mut v1, &mut v2);

    for h in 0..num_heads {
        rope_rotate(&mut q1[h * head_dim..(h + 1) * head_dim], position, inv_freq);
        rope_rotate(&mut q2[h * head_dim..(h + 1) * head_dim], position + 1, inv_freq);
    }
    for g in 0..num_kv_heads {
        rope_rotate(&mut k1[g * head_dim..(g + 1) * head_dim], position, inv_freq);
        rope_rotate(&mut k2[g * head_dim..(g + 1) * head_dim], position + 1, inv_freq);
    }

    // ── Sequential attention: p, then p+1 (causal dependency) ──
    let alive = vec![true; num_kv_heads];
    let attend = |q_all: &[f32], cache: &LayerKvCache| -> Vec<f32> {
        let mut attn_out = vec![0.0f32; qk_dim];
        let mut imp = vec![0.0f32; cache.seq_len];
        for h in 0..num_heads {
            let g = h / heads_per_kv;
            let stored = cache.head_len(g);
            if stored == 0 {
                continue;
            }
            let _ = stored;
            let (out, probs) =
                cache.attend(&q_all[h * head_dim..(h + 1) * head_dim], g);
            attn_out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
            for (dst, &p) in imp.iter_mut().zip(&probs) {
                *dst += p;
            }
        }
        attn_out.extend_from_slice(&imp); // carry imp back to the caller
        attn_out
    };

    cache.append(&k1, &v1, &alive);
    let mut a1 = attend(&q1, cache);
    let imp1 = a1.split_off(qk_dim);
    cache.accumulate_imp(&imp1);

    cache.append(&k2, &v2, &alive);
    let mut a2 = attend(&q2, cache);
    let imp2 = a2.split_off(qk_dim);
    cache.accumulate_imp(&imp2);

    // ── Fused output projection ──
    let mut out1 = vec![0.0f32; hidden_size];
    let mut out2 = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        let row = &wo[i * qk_dim..(i + 1) * qk_dim];
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for j in 0..qk_dim {
            s1 += row[j] * a1[j];
            s2 += row[j] * a2[j];
        }
        out1[i] = s1;
        out2[i] = s2;
    }
    (out1, out2)
}

/// Per-head RMS norm with weight (qk-norm). Follows the model's norm
/// style: Qwen3.5/Qwen3-Next are zero-centered `x̂·(1+w)` (gemma-style),
/// classic Qwen/Llama are `x̂·w` — same authority as the layer norms.
#[inline]
fn rmsnorm_head(x: &mut [f32], w: &[f32], eps: f64, style: cortiq_core::NormStyle) {
    let mut ss = 0f64;
    for &v in x.iter() {
        ss += (v as f64) * (v as f64);
    }
    let inv = (1.0 / (ss / x.len() as f64 + eps).sqrt()) as f32;
    match style {
        cortiq_core::NormStyle::Qwen => {
            for (v, &wi) in x.iter_mut().zip(w) {
                *v = *v * inv * wi;
            }
        }
        cortiq_core::NormStyle::Gemma => {
            for (v, &wi) in x.iter_mut().zip(w) {
                *v = *v * inv * (1.0 + wi);
            }
        }
    }
}

/// Dense attention configuration (no head masks — masked execution uses
/// the historical path).
pub struct QwenAttnCfg<'a> {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub position: usize,
    /// len = rotary_dim / 2
    pub inv_freq: &'a [f32],
    /// ≤ head_dim; RoPE rotates only the first `rotary_dim` dims.
    pub rotary_dim: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// Qwen3.5: wq rows = 2·nh·hd, per-head [q(hd); gate(hd)];
    /// attention output is multiplied by sigmoid(gate) before o_proj.
    pub output_gate: bool,
    pub rms_eps: f64,
    /// Norm-weight semantics for qk-norm (same as the layer norms).
    pub norm_style: cortiq_core::NormStyle,
    /// Qwen2-family q/k/v projection biases (added after the matvecs).
    pub bias: Option<(&'a [f32], &'a [f32], &'a [f32])>,
    pub pool: Option<&'a Pool>,
}

thread_local! {
    /// Recycled projection/attention buffers: the dense attention path
    /// consumed ~7 fresh Vecs per layer per token (roadmap §3 P0).
    static PROJ_FREE: std::cell::RefCell<Vec<Vec<f32>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Take a zeroed buffer of length `n` from the freelist (or allocate).
pub(crate) fn take_buf(n: usize) -> Vec<f32> {
    let mut b = PROJ_FREE.with(|f| f.borrow_mut().pop()).unwrap_or_default();
    b.clear();
    b.resize(n, 0.0);
    b
}

/// Return a buffer to the freelist (leaves an empty Vec behind).
pub(crate) fn recycle_buf(b: &mut Vec<f32>) {
    let b = std::mem::take(b);
    if b.capacity() > 0 {
        PROJ_FREE.with(|f| {
            let mut f = f.borrow_mut();
            if f.len() < 16 {
                f.push(b);
            }
        });
    }
}

struct Projected {
    q: Vec<f32>,
    gate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Drop for Projected {
    fn drop(&mut self) {
        recycle_buf(&mut self.q);
        recycle_buf(&mut self.gate);
        recycle_buf(&mut self.k);
        recycle_buf(&mut self.v);
    }
}

/// Project + split gate + qk-norm + partial RoPE for one position.
fn project_position(
    hidden: &[f32],
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    cfg: &QwenAttnCfg,
    position: usize,
) -> Projected {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let mut q_raw = take_buf(wq.rows());
    let mut k = take_buf(nkv * hd);
    let mut v = take_buf(nkv * hd);
    // Multi-matrix job: Q, K and V projections under one pool dispatch.
    QTensor::matvec_many(
        [wq, wk, wv],
        hidden,
        [q_raw.as_mut_slice(), k.as_mut_slice(), v.as_mut_slice()],
        cfg.pool,
    );
    if let Some((bq, bk, bv)) = cfg.bias {
        for (x, b) in q_raw.iter_mut().zip(bq) {
            *x += b;
        }
        for (x, b) in k.iter_mut().zip(bk) {
            *x += b;
        }
        for (x, b) in v.iter_mut().zip(bv) {
            *x += b;
        }
    }

    // Gate split: per-head [q(hd); gate(hd)] (vmfcore/HF convention).
    let (mut q, gate) = if cfg.output_gate {
        let mut qn = take_buf(nh * hd);
        let mut g = take_buf(nh * hd);
        for h in 0..nh {
            let src = h * hd * 2;
            let dst = h * hd;
            qn[dst..dst + hd].copy_from_slice(&q_raw[src..src + hd]);
            g[dst..dst + hd].copy_from_slice(&q_raw[src + hd..src + 2 * hd]);
        }
        recycle_buf(&mut q_raw);
        (qn, g)
    } else {
        (q_raw, Vec::new())
    };

    // qk-norm before RoPE.
    if let Some(qw) = cfg.q_norm {
        for h in 0..nh {
            rmsnorm_head(&mut q[h * hd..h * hd + hd], qw, cfg.rms_eps, cfg.norm_style);
        }
    }
    if let Some(kw) = cfg.k_norm {
        for g in 0..nkv {
            rmsnorm_head(&mut k[g * hd..g * hd + hd], kw, cfg.rms_eps, cfg.norm_style);
        }
    }

    // Partial RoPE: rotate only the first rotary_dim dims of each head.
    let rd = cfg.rotary_dim.min(hd);
    for h in 0..nh {
        rope_rotate(&mut q[h * hd..h * hd + rd], position, cfg.inv_freq);
    }
    for g in 0..nkv {
        rope_rotate(&mut k[g * hd..g * hd + rd], position, cfg.inv_freq);
    }
    Projected { q, gate, k, v }
}

fn attend_all_heads(
    q: &[f32],
    cache: &LayerKvCache,
    nh: usize,
    heads_per_kv: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut attn_out = take_buf(nh * hd);
    let mut imp = take_buf(cache.seq_len);
    // Grouped GQA kernel: the group's shared K/V storage is streamed
    // once for all its Q-heads (per-head attend re-read it
    // heads_per_kv times). Bit-identical per head — see attend_group.
    let nkv = nh / heads_per_kv;
    for g in 0..nkv {
        if cache.head_len(g) == 0 {
            continue;
        }
        let span = g * heads_per_kv * hd..(g + 1) * heads_per_kv * hd;
        cache.attend_group(&q[span.clone()], g, &mut attn_out[span], &mut imp);
    }
    (attn_out, imp)
}

#[inline]
fn apply_gate(ao: &mut [f32], gate: &[f32]) {
    for (a, &g) in ao.iter_mut().zip(gate) {
        *a *= 1.0 / (1.0 + (-g).exp());
    }
}

/// Dense GQA attention for one position (QTensor weights, Qwen3.5 extras).
#[allow(clippy::too_many_arguments)]
pub fn qwen_attention(
    hidden: &[f32],
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    wo: &QTensor,
    cache: &mut LayerKvCache,
    cfg: &QwenAttnCfg,
) -> Vec<f32> {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let heads_per_kv = nh / nkv;
    let p = project_position(hidden, wq, wk, wv, cfg, cfg.position);
    // O(1) prefill trace: while a nystrom layer is collecting, the exact
    // prompt pass also records this position's queries for the seal
    // (no-op on plain layers).
    cache.o1_push_q(&p.q);
    // Empty alive slice = every head alive (append's get().unwrap_or(true))
    // — the vec![true; nkv] here was one allocation per layer per token.
    cache.append(&p.k, &p.v, &[]);

    let (mut ao, mut imp) = attend_all_heads(&p.q, cache, nh, heads_per_kv, hd);
    cache.accumulate_imp(&imp);
    if cfg.output_gate {
        apply_gate(&mut ao, &p.gate);
    }
    let mut out = take_buf(cfg.hidden_size);
    wo.matvec(&ao, &mut out, cfg.pool);
    recycle_buf(&mut ao);
    recycle_buf(&mut imp);
    out
}

/// Batched-chunk exact attention (roadmap §3 P0 «prefill»): Q/K/V and O
/// projections run as chunk-GEMMs — each weight row streams from memory
/// ONCE per chunk instead of once per position — while the attention
/// core stays per-position (causal append order). Per-position math is
/// identical to `qwen_attention` (matmat ≡ per-position matvec by the
/// existing parity tests), so the results match the sequential prefill.
/// `cfg.position` is the chunk's FIRST absolute position.
#[allow(clippy::too_many_arguments)]
pub fn qwen_attention_batch(
    normed_all: &[f32],
    b: usize,
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    wo: &QTensor,
    cache: &mut LayerKvCache,
    cfg: &QwenAttnCfg,
) -> Vec<f32> {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let heads_per_kv = nh / nkv;
    let qrows = wq.rows();
    debug_assert_eq!(normed_all.len(), b * cfg.hidden_size);

    // ── chunk-GEMM projections ──
    let mut q_all = take_buf(b * qrows);
    let mut k_all = take_buf(b * nkv * hd);
    let mut v_all = take_buf(b * nkv * hd);
    wq.matmat(normed_all, b, &mut q_all, cfg.pool);
    wk.matmat(normed_all, b, &mut k_all, cfg.pool);
    wv.matmat(normed_all, b, &mut v_all, cfg.pool);

    // ── per-position: bias, gate split, qk-norm, partial RoPE, causal
    //    attention over the growing cache ──
    let mut ao_all = take_buf(b * nh * hd);
    let rd = cfg.rotary_dim.min(hd);
    for bi in 0..b {
        let pos = cfg.position + bi;
        let q_raw = &mut q_all[bi * qrows..(bi + 1) * qrows];
        let k = &mut k_all[bi * nkv * hd..(bi + 1) * nkv * hd];
        let v = &mut v_all[bi * nkv * hd..(bi + 1) * nkv * hd];
        if let Some((bq, bk, bv)) = cfg.bias {
            for (x, bb) in q_raw.iter_mut().zip(bq) {
                *x += bb;
            }
            for (x, bb) in k.iter_mut().zip(bk) {
                *x += bb;
            }
            for (x, bb) in v.iter_mut().zip(bv) {
                *x += bb;
            }
        }
        // Gate split: per-head [q(hd); gate(hd)] (see project_position).
        let (mut q, mut gate) = if cfg.output_gate {
            let mut qn = take_buf(nh * hd);
            let mut g = take_buf(nh * hd);
            for hh in 0..nh {
                let src = hh * hd * 2;
                let dst = hh * hd;
                qn[dst..dst + hd].copy_from_slice(&q_raw[src..src + hd]);
                g[dst..dst + hd].copy_from_slice(&q_raw[src + hd..src + 2 * hd]);
            }
            (qn, g)
        } else {
            (take_buf(nh * hd), Vec::new())
        };
        if !cfg.output_gate {
            q.copy_from_slice(&q_raw[..nh * hd]);
        }
        if let Some(qw) = cfg.q_norm {
            for hh in 0..nh {
                rmsnorm_head(&mut q[hh * hd..hh * hd + hd], qw, cfg.rms_eps, cfg.norm_style);
            }
        }
        if let Some(kw) = cfg.k_norm {
            for g in 0..nkv {
                rmsnorm_head(&mut k[g * hd..g * hd + hd], kw, cfg.rms_eps, cfg.norm_style);
            }
        }
        for hh in 0..nh {
            rope_rotate(&mut q[hh * hd..hh * hd + rd], pos, cfg.inv_freq);
        }
        for g in 0..nkv {
            rope_rotate(&mut k[g * hd..g * hd + rd], pos, cfg.inv_freq);
        }

        cache.o1_push_q(&q);
        cache.append(k, v, &[]);
        let (mut ao, mut imp) = attend_all_heads(&q, cache, nh, heads_per_kv, hd);
        cache.accumulate_imp(&imp);
        if cfg.output_gate {
            apply_gate(&mut ao, &gate);
        }
        ao_all[bi * nh * hd..(bi + 1) * nh * hd].copy_from_slice(&ao);
        recycle_buf(&mut ao);
        recycle_buf(&mut imp);
        recycle_buf(&mut q);
        recycle_buf(&mut gate);
    }

    // ── chunk-GEMM output projection ──
    let mut out = vec![0.0f32; b * cfg.hidden_size];
    wo.matmat(&ao_all, b, &mut out, cfg.pool);
    recycle_buf(&mut q_all);
    recycle_buf(&mut k_all);
    recycle_buf(&mut v_all);
    recycle_buf(&mut ao_all);
    out
}

/// Dense GQA attention for one DECODE position on a SEALED O(1) layer:
/// projection / qk-norm / partial RoPE / output gate are identical to
/// `qwen_attention`, but the KV cache is replaced by per-KV-group
/// streaming Nyström states (exact window + permanent sinks + landmark
/// skeleton, shared across the group's Q heads). Head masks don't
/// apply — the o1 path is dense, like the masked-on-quantized fallback.
#[allow(clippy::too_many_arguments)]
pub fn qwen_attention_nystrom(
    hidden: &[f32],
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    wo: &QTensor,
    cache: &mut LayerKvCache,
    cfg: &QwenAttnCfg,
) -> Vec<f32> {
    let p = project_position(hidden, wq, wk, wv, cfg, cfg.position);
    let mut ao = cache.o1_step(&p.q, &p.k, &p.v, cfg.num_heads);
    if cfg.output_gate {
        apply_gate(&mut ao, &p.gate);
    }
    let mut out = vec![0.0f32; cfg.hidden_size];
    wo.matvec(&ao, &mut out, cfg.pool);
    out
}

/// Fused two-position dense attention (speculative verify): projections
/// stream the weights once via `matvec2`; attention runs sequentially
/// (causal dependency through the cache).
#[allow(clippy::too_many_arguments)]
pub fn qwen_attention_pair(
    h1: &[f32],
    h2: &[f32],
    wq: &QTensor,
    wk: &QTensor,
    wv: &QTensor,
    wo: &QTensor,
    cache: &mut LayerKvCache,
    cfg: &QwenAttnCfg,
) -> (Vec<f32>, Vec<f32>) {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let heads_per_kv = nh / nkv;

    // Fused projections (one weight pass for both positions) — Q, K
    // and V under a single pool dispatch (multi-matrix pair job).
    let mut q1r = take_buf(wq.rows());
    let mut q2r = take_buf(wq.rows());
    let mut k1 = take_buf(nkv * hd);
    let mut k2 = take_buf(nkv * hd);
    let mut v1 = take_buf(nkv * hd);
    let mut v2 = take_buf(nkv * hd);
    QTensor::matvec2_many(
        [wq, wk, wv],
        h1,
        h2,
        [q1r.as_mut_slice(), k1.as_mut_slice(), v1.as_mut_slice()],
        [q2r.as_mut_slice(), k2.as_mut_slice(), v2.as_mut_slice()],
        cfg.pool,
    );
    if let Some((bq, bk, bv)) = cfg.bias {
        for lane in [(&mut q1r, &mut k1, &mut v1), (&mut q2r, &mut k2, &mut v2)] {
            for (x, b) in lane.0.iter_mut().zip(bq) {
                *x += b;
            }
            for (x, b) in lane.1.iter_mut().zip(bk) {
                *x += b;
            }
            for (x, b) in lane.2.iter_mut().zip(bv) {
                *x += b;
            }
        }
    }

    let finish = |mut q_raw: Vec<f32>, k: &mut [f32], pos: usize| -> (Vec<f32>, Vec<f32>) {
        // split + norms + rope, reusing the single-position logic shape
        let (mut q, mut gate) = if cfg.output_gate {
            let mut qn = take_buf(nh * hd);
            let mut g = take_buf(nh * hd);
            for h in 0..nh {
                let src = h * hd * 2;
                let dst = h * hd;
                qn[dst..dst + hd].copy_from_slice(&q_raw[src..src + hd]);
                g[dst..dst + hd].copy_from_slice(&q_raw[src + hd..src + 2 * hd]);
            }
            recycle_buf(&mut q_raw);
            (qn, g)
        } else {
            (q_raw, Vec::new())
        };
        if let Some(qw) = cfg.q_norm {
            for h in 0..nh {
                rmsnorm_head(&mut q[h * hd..h * hd + hd], qw, cfg.rms_eps, cfg.norm_style);
            }
        }
        if let Some(kw) = cfg.k_norm {
            for g in 0..nkv {
                rmsnorm_head(&mut k[g * hd..g * hd + hd], kw, cfg.rms_eps, cfg.norm_style);
            }
        }
        let rd = cfg.rotary_dim.min(hd);
        for h in 0..nh {
            rope_rotate(&mut q[h * hd..h * hd + rd], pos, cfg.inv_freq);
        }
        for g in 0..nkv {
            rope_rotate(&mut k[g * hd..g * hd + rd], pos, cfg.inv_freq);
        }
        let _ = &mut gate;
        (q, gate)
    };

    let (mut qa, mut gate1) = finish(q1r, &mut k1, cfg.position);
    let (mut qb, mut gate2) = finish(q2r, &mut k2, cfg.position + 1);

    // O(1) prefill trace (see qwen_attention): lane order = position
    // order, so the collected buffer stays position-major.
    cache.o1_push_q(&qa);
    cache.o1_push_q(&qb);
    // Empty alive slice = every head alive (see qwen_attention).
    cache.append(&k1, &v1, &[]);
    let (mut a1, mut imp1) = attend_all_heads(&qa, cache, nh, heads_per_kv, hd);
    cache.accumulate_imp(&imp1);

    cache.append(&k2, &v2, &[]);
    let (mut a2, mut imp2) = attend_all_heads(&qb, cache, nh, heads_per_kv, hd);
    cache.accumulate_imp(&imp2);

    if cfg.output_gate {
        apply_gate(&mut a1, &gate1);
        apply_gate(&mut a2, &gate2);
    }

    let mut o1 = take_buf(cfg.hidden_size);
    let mut o2 = take_buf(cfg.hidden_size);
    wo.matvec2(&a1, &a2, &mut o1, &mut o2, cfg.pool);
    for b in [
        &mut qa, &mut qb, &mut gate1, &mut gate2, &mut k1, &mut k2, &mut v1, &mut v2,
        &mut a1, &mut a2, &mut imp1, &mut imp2,
    ] {
        recycle_buf(b);
    }
    (o1, o2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::LayerKvCache;

    fn synth(rows: usize, cols: usize, salt: usize) -> QTensor {
        QTensor::from_f32(
            (0..rows * cols)
                .map(|i| (((i * 13 + salt * 7) % 97) as f32 / 97.0 - 0.5) * 0.4)
                .collect(),
            rows,
            cols,
        )
    }

    /// Every projection path must apply identical semantics: the pair
    /// path (prefill / speculative verify) once missed the q/k/v bias
    /// while singles had it — healthy PPL, garbage generation.
    #[test]
    fn pair_with_bias_matches_two_singles() {
        let (nh, nkv, hd, hs) = (2usize, 1usize, 4usize, 8usize);
        let wq = synth(nh * hd, hs, 1);
        let wk = synth(nkv * hd, hs, 2);
        let wv = synth(nkv * hd, hs, 3);
        let wo = synth(hs, nh * hd, 4);
        let bq: Vec<f32> = (0..nh * hd).map(|i| 0.1 + 0.01 * i as f32).collect();
        let bk: Vec<f32> = (0..nkv * hd).map(|i| -0.2 + 0.02 * i as f32).collect();
        let bv: Vec<f32> = (0..nkv * hd).map(|i| 0.05 * i as f32).collect();
        let inv = rope_inv_freq(hd, 10_000.0);
        let cfg = |position| QwenAttnCfg {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            hidden_size: hs,
            position,
            inv_freq: &inv,
            rotary_dim: hd,
            q_norm: None,
            k_norm: None,
            output_gate: false,
            bias: Some((&bq, &bk, &bv)),
            rms_eps: 1e-6,
            norm_style: cortiq_core::NormStyle::Qwen,
            pool: None,
        };
        let h1: Vec<f32> = (0..hs).map(|i| (i as f32 * 0.3).sin()).collect();
        let h2: Vec<f32> = (0..hs).map(|i| (i as f32 * 0.7).cos()).collect();

        let mut c_ref = LayerKvCache::new(nkv, hd);
        let r1 = qwen_attention(&h1, &wq, &wk, &wv, &wo, &mut c_ref, &cfg(0));
        let r2 = qwen_attention(&h2, &wq, &wk, &wv, &wo, &mut c_ref, &cfg(1));

        let mut c = LayerKvCache::new(nkv, hd);
        let (p1, p2) = qwen_attention_pair(&h1, &h2, &wq, &wk, &wv, &wo, &mut c, &cfg(0));
        for (a, b) in r1.iter().zip(&p1) {
            assert!((a - b).abs() < 1e-5, "lane1 {a} vs {b}");
        }
        for (a, b) in r2.iter().zip(&p2) {
            assert!((a - b).abs() < 1e-5, "lane2 {a} vs {b}");
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let mut q = vec![1.0, 0.0, 0.5, 0.5];
        let before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope_rotate(&mut q, 7, &rope_inv_freq(4, 10000.0));
        let after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut q = vec![0.3, -0.7, 1.1, 0.2];
        let orig = q.clone();
        rope_rotate(&mut q, 0, &rope_inv_freq(4, 10000.0));
        for (a, b) in q.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn attention_head_uniform() {
        let head_dim = 4;
        let seq_len = 3;
        let q = vec![1.0; head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        let (out, probs) = attention_head(&q, &k, &v, head_dim, seq_len);
        for d in 0..3 {
            assert!((out[d] - 1.0 / 3.0).abs() < 0.1);
        }
        let mass: f32 = probs.iter().sum();
        assert!((mass - 1.0).abs() < 1e-5, "probs must sum to 1");
    }

    #[test]
    fn dead_group_skips_projection_and_cache() {
        let (heads, kv, hd, hidden) = (4usize, 2usize, 4usize, 8usize);
        let mut cache = LayerKvCache::new(kv, hd);
        let h_in = vec![0.5f32; hidden];
        let wq = vec![0.1f32; heads * hd * hidden];
        let wk = vec![0.1f32; kv * hd * hidden];
        let wv = vec![0.1f32; kv * hd * hidden];
        let wo = vec![0.1f32; hidden * heads * hd];

        // Kill group 1 (Q heads 2 and 3).
        let active = vec![true, true, false, false];
        let inv_freq = rope_inv_freq(hd, 1e4);
        let out = multi_head_attention(
            &h_in, &wq, &wk, &wv, &wo, &mut cache, heads, kv, hd, hidden, 0, &active, &inv_freq,
        );
        assert_eq!(cache.head_len(0), 1, "live group cached");
        assert_eq!(cache.head_len(1), 0, "dead group must not be cached");
        assert!(out.iter().any(|&x| x.abs() > 1e-9), "live heads still produce output");
    }

    #[test]
    fn attention_pair_equals_two_sequential_calls() {
        let (heads, kv, hd, hidden) = (4usize, 2usize, 4usize, 8usize);
        let mk = |salt: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * 7 + salt * 13) % 89) as f32 / 89.0 - 0.5).collect()
        };
        let h1 = mk(1, hidden);
        let h2 = mk(2, hidden);
        let wq = mk(3, heads * hd * hidden);
        let wk = mk(4, kv * hd * hidden);
        let wv = mk(5, kv * hd * hidden);
        let wo = mk(6, hidden * heads * hd);
        let inv_freq = rope_inv_freq(hd, 1e4);

        // Reference: two sequential single-position calls.
        let mut c_ref = LayerKvCache::new(kv, hd);
        let r1 = multi_head_attention(
            &h1, &wq, &wk, &wv, &wo, &mut c_ref, heads, kv, hd, hidden, 5, &[true; 4], &inv_freq,
        );
        let r2 = multi_head_attention(
            &h2, &wq, &wk, &wv, &wo, &mut c_ref, heads, kv, hd, hidden, 6, &[true; 4], &inv_freq,
        );

        // Fused pair.
        let mut c_pair = LayerKvCache::new(kv, hd);
        let (p1, p2) = multi_head_attention_pair(
            &h1, &h2, &wq, &wk, &wv, &wo, &mut c_pair, heads, kv, hd, hidden, 5, &inv_freq,
        );

        assert_eq!(r1, p1, "pair lane 1 must be bit-identical");
        assert_eq!(r2, p2, "pair lane 2 must be bit-identical");
        assert_eq!(c_ref.seq_len, c_pair.seq_len);
        assert_eq!(c_ref.head_keys(0), c_pair.head_keys(0));
    }

    #[test]
    fn masked_equals_dense_when_all_heads_alive() {
        let (heads, kv, hd, hidden) = (2usize, 1usize, 4usize, 8usize);
        let h_in: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.3).sin()).collect();
        let wq: Vec<f32> = (0..heads * hd * hidden).map(|i| (i as f32 * 0.01).cos() * 0.1).collect();
        let wk: Vec<f32> = (0..kv * hd * hidden).map(|i| (i as f32 * 0.02).sin() * 0.1).collect();
        let wv: Vec<f32> = (0..kv * hd * hidden).map(|i| (i as f32 * 0.03).cos() * 0.1).collect();
        let wo: Vec<f32> = (0..hidden * heads * hd).map(|i| (i as f32 * 0.04).sin() * 0.1).collect();

        let mut c1 = LayerKvCache::new(kv, hd);
        let mut c2 = LayerKvCache::new(kv, hd);
        let inv_freq = rope_inv_freq(hd, 1e4);
        let dense = multi_head_attention(
            &h_in, &wq, &wk, &wv, &wo, &mut c1, heads, kv, hd, hidden, 0, &[true, true], &inv_freq,
        );
        let masked = multi_head_attention(
            &h_in, &wq, &wk, &wv, &wo, &mut c2, heads, kv, hd, hidden, 0, &[true; 2], &inv_freq,
        );
        for (a, b) in dense.iter().zip(&masked) {
            assert_eq!(a, b, "full mask must be bit-identical to dense");
        }
    }
}
