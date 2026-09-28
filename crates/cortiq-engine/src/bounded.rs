//! Natively bounded softmax anchor `swa_sink_v1` (Embryo-O1).
//!
//! Contract: `docs/EMBRYO_BOUNDED_ANCHOR.md`. The operator is a TRAINED
//! bounded attention, not a masked full attention: per token `t`
//!
//! ```text
//! window keys:  j ∈ (t − W, t]                (W keys INCLUDING the token)
//! window score: s_j = ( R(t − j) q̂_t ) · k̂_j / √hd
//! sink score:   s_s = q̂_t · k̂ˢ_s / √hd,   s ∈ [0, S)      (NoPE)
//! one softmax over {s_s} ∪ {s_j};   out_t = Σ_s p_s v̂ˢ_s + Σ_j p_j v_j
//! ```
//!
//! The ring stores RAW (unrotated) keys; the query is rotated by the
//! distance `Δ = t − j ∈ [0, W)` through a `[W][rd/2]` cos/sin table.
//! With absolute RoPE `q_rot(t) = R(t) q̂`, `k_rot(j) = R(j) k̂` one has
//! `q_rot(t)·k_rot(j) = q̂ᵀ R(t)ᵀ R(j) k̂ = (R(t−j) q̂)·k̂` (every 2-D block
//! of `R` is a plane rotation, so `R(t)ᵀ R(j) = R(j − t)`), which is why
//! the trainer may keep rotating by absolute positions and only mask,
//! while the served operator never sees an absolute position at all.
//!
//! State per layer: `ring_k, ring_v [kvh][W][hd]` + the insert counter —
//! a record of fixed size derived from the header, identical in prefill,
//! decode and across turns. Nothing here grows with the context.

use crate::qtensor::QTensor;

/// Rows of rollback history kept beside the ring so a speculative reject
/// (`truncate_last`) can restore the slots the rejected tokens overwrote.
/// Bounded by construction; not part of the wire state.
pub const UNDO_DEPTH: usize = 64;

/// Weights of one bounded-anchor layer.
pub struct BoundedWeights {
    pub wq: QTensor,
    pub wk: QTensor,
    pub wv: QTensor,
    pub wo: QTensor,
    /// Trained NoPE sink keys `[kvh][sink][hd]` (weights, not positions).
    pub sink_k: Vec<f32>,
    /// Trained sink values `[kvh][sink][hd]`.
    pub sink_v: Vec<f32>,
    pub sink: usize,
    pub window: usize,
}

/// Relative-rotation table: `cos/sin[Δ][i] = cos/sin(Δ · inv_freq[i])`
/// for `Δ ∈ [0, W)`, built once per model from the layer's `inv_freq`
/// (same convention as `attention::rope_rotate_scaled`, angles computed
/// in f64 so the table carries only the final f32 rounding).
#[derive(Debug, Clone)]
pub struct BoundedRope {
    pub window: usize,
    /// `rotary_dim / 2`: dims `[0, half)` pair with `[half, 2·half)`;
    /// dims past `2·half` are copied unrotated (partial rotary).
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl BoundedRope {
    /// `rope_scale` is the YaRN attention factor the absolute path applies
    /// to BOTH q and k; the relative path rotates only q, so the window
    /// score carries `scale²` (`(s·R(t)q)·(s·R(j)k) = s²·(R(t−j)q)·k`).
    pub fn new(window: usize, inv_freq: &[f32], rope_scale: f32) -> Self {
        let half = inv_freq.len();
        let s2 = (rope_scale as f64) * (rope_scale as f64);
        let mut cos = Vec::with_capacity(window * half);
        let mut sin = Vec::with_capacity(window * half);
        for delta in 0..window {
            for &f in inv_freq {
                let (sn, cs) = ((delta as f64) * (f as f64)).sin_cos();
                cos.push((cs * s2) as f32);
                sin.push((sn * s2) as f32);
            }
        }
        Self {
            window,
            half,
            cos,
            sin,
        }
    }

    /// `out = R(Δ) x` (first `2·half` dims rotated, the rest copied).
    #[inline]
    pub fn rotate(&self, delta: usize, x: &[f32], out: &mut [f32]) {
        let half = self.half;
        let c = &self.cos[delta * half..(delta + 1) * half];
        let s = &self.sin[delta * half..(delta + 1) * half];
        for i in 0..half {
            let x0 = x[i];
            let x1 = x[i + half];
            out[i] = x0 * c[i] - x1 * s[i];
            out[i + half] = x0 * s[i] + x1 * c[i];
        }
        let r = 2 * half;
        if x.len() > r {
            out[r..x.len()].copy_from_slice(&x[r..]);
        }
    }
}

/// Bit-for-bit copy of everything the operator mutates (speculation).
#[derive(Debug, Clone)]
pub struct BoundedSnapshot {
    ring_k: Vec<f32>,
    ring_v: Vec<f32>,
    seen: usize,
}

/// Per-layer bounded state: the ring of the last `W` raw keys/values per
/// KV head plus the insert counter. `len = min(seen, W)`, next write slot
/// `head = seen mod W`, and the key of position `j` lives in slot
/// `j mod W` — so no absolute position is ever stored.
#[derive(Debug, Clone)]
pub struct BoundedState {
    pub window: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `[kvh][W][hd]`, raw (unrotated) keys.
    pub ring_k: Vec<f32>,
    /// `[kvh][W][hd]`.
    pub ring_v: Vec<f32>,
    /// Tokens inserted since the last clear (= the next position).
    pub seen: usize,
    /// Rollback rows: the slot contents each of the last `UNDO_DEPTH`
    /// inserts overwrote (`[UNDO_DEPTH][kvh][hd]` for k and v).
    undo_k: Vec<f32>,
    undo_v: Vec<f32>,
    undo_len: usize,
    undo_head: usize,
}

impl BoundedState {
    pub fn new(num_kv_heads: usize, head_dim: usize, window: usize) -> Self {
        let n = num_kv_heads * window * head_dim;
        let u = UNDO_DEPTH * num_kv_heads * head_dim;
        Self {
            window,
            num_kv_heads,
            head_dim,
            ring_k: vec![0.0; n],
            ring_v: vec![0.0; n],
            seen: 0,
            undo_k: vec![0.0; u],
            undo_v: vec![0.0; u],
            undo_len: 0,
            undo_head: 0,
        }
    }

    /// Filled slots, `min(seen, W)`.
    #[inline]
    pub fn len(&self) -> usize {
        self.seen.min(self.window)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.seen == 0
    }

    /// Next write slot, `seen mod W`.
    #[inline]
    pub fn head(&self) -> usize {
        self.seen % self.window
    }

    /// Bytes of the wire-visible state (ring + counter). The undo rows
    /// are scratch of fixed size and are not state.
    pub fn state_bytes(&self) -> usize {
        (self.ring_k.len() + self.ring_v.len()) * std::mem::size_of::<f32>()
            + std::mem::size_of::<u64>()
    }

    /// Zero the record (fresh sequence). Capacity never changes.
    pub fn clear(&mut self) {
        self.ring_k.fill(0.0);
        self.ring_v.fill(0.0);
        self.seen = 0;
        self.undo_len = 0;
        self.undo_head = 0;
    }

    /// Write `k, v` (`[kvh][hd]`, raw) into slot `seen mod W`, saving
    /// the overwritten rows for rollback.
    pub fn insert(&mut self, k: &[f32], v: &[f32]) {
        let (kvh, hd, w) = (self.num_kv_heads, self.head_dim, self.window);
        debug_assert_eq!(k.len(), kvh * hd);
        debug_assert_eq!(v.len(), kvh * hd);
        let slot = self.head();
        let u = self.undo_head;
        for h in 0..kvh {
            let r = (h * w + slot) * hd;
            let uo = (u * kvh + h) * hd;
            self.undo_k[uo..uo + hd].copy_from_slice(&self.ring_k[r..r + hd]);
            self.undo_v[uo..uo + hd].copy_from_slice(&self.ring_v[r..r + hd]);
            self.ring_k[r..r + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
            self.ring_v[r..r + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
        }
        self.undo_head = (u + 1) % UNDO_DEPTH;
        self.undo_len = (self.undo_len + 1).min(UNDO_DEPTH);
        self.seen += 1;
    }

    /// Undo the last `n` inserts exactly (restores the overwritten slots).
    /// Returns how many were rolled back — fewer than `n` only when the
    /// rollback history (`UNDO_DEPTH`) or the inserted count is shorter.
    pub fn rollback(&mut self, n: usize) -> usize {
        let (kvh, hd, w) = (self.num_kv_heads, self.head_dim, self.window);
        let n = n.min(self.undo_len).min(self.seen);
        for _ in 0..n {
            self.seen -= 1;
            let slot = self.seen % w;
            let u = (self.undo_head + UNDO_DEPTH - 1) % UNDO_DEPTH;
            for h in 0..kvh {
                let r = (h * w + slot) * hd;
                let uo = (u * kvh + h) * hd;
                self.ring_k[r..r + hd].copy_from_slice(&self.undo_k[uo..uo + hd]);
                self.ring_v[r..r + hd].copy_from_slice(&self.undo_v[uo..uo + hd]);
            }
            self.undo_head = u;
            self.undo_len -= 1;
        }
        n
    }

    pub fn snapshot(&self) -> BoundedSnapshot {
        BoundedSnapshot {
            ring_k: self.ring_k.clone(),
            ring_v: self.ring_v.clone(),
            seen: self.seen,
        }
    }

    /// Restore a snapshot taken on THIS state (same geometry). The undo
    /// history is discarded: it described inserts that no longer exist.
    pub fn restore(&mut self, s: &BoundedSnapshot) {
        debug_assert_eq!(s.ring_k.len(), self.ring_k.len());
        self.ring_k.copy_from_slice(&s.ring_k);
        self.ring_v.copy_from_slice(&s.ring_v);
        self.seen = s.seen;
        self.undo_len = 0;
        self.undo_head = 0;
    }

    /// Bit-for-bit equality of the wire-visible state.
    pub fn same_state(&self, other: &BoundedState) -> bool {
        self.window == other.window
            && self.seen == other.seen
            && self.ring_k == other.ring_k
            && self.ring_v == other.ring_v
    }

    /// Attend every Q head of every KV group over `sink ∪ window` and
    /// write `[nh][hd]` into `out`. `q` is `[nh][hd]` RAW (unrotated);
    /// the current token must already be inserted (the window includes
    /// it, matching the trainer's band `col == S + row`).
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        q: &[f32],
        num_heads: usize,
        sink_k: &[f32],
        sink_v: &[f32],
        sink: usize,
        rope: &BoundedRope,
        scale: f32,
        out: &mut [f32],
    ) {
        let (kvh, hd, w) = (self.num_kv_heads, self.head_dim, self.window);
        let nh = num_heads;
        let hpk = nh / kvh.max(1);
        debug_assert_eq!(hpk * kvh, nh);
        debug_assert_eq!(q.len(), nh * hd);
        debug_assert_eq!(out.len(), nh * hd);
        debug_assert_eq!(sink_k.len(), kvh * sink * hd);
        debug_assert_eq!(rope.window, w);
        let m = self.len();
        let n = sink + m;
        let head = self.head();
        thread_local! {
            static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            let (scores, qrot) = &mut *s;
            scores.clear();
            scores.resize(nh * n, 0.0);
            qrot.clear();
            qrot.resize(nh * hd, 0.0);
            // Sink scores on the raw query (NoPE).
            for h in 0..nh {
                let g = h / hpk;
                let qh = &q[h * hd..(h + 1) * hd];
                for s in 0..sink {
                    let kr = &sink_k[(g * sink + s) * hd..(g * sink + s + 1) * hd];
                    scores[h * n + s] = crate::attention::dot_f32(qh, kr) * scale;
                }
            }
            // Window scores: Δ = 0 is the token itself; slot(Δ) walks the
            // ring backwards from the last written slot.
            for d in 0..m {
                let slot = (head + w - 1 - d) % w;
                for h in 0..nh {
                    rope.rotate(d, &q[h * hd..(h + 1) * hd], &mut qrot[h * hd..(h + 1) * hd]);
                }
                for h in 0..nh {
                    let g = h / hpk;
                    let kr = &self.ring_k[(g * w + slot) * hd..(g * w + slot + 1) * hd];
                    scores[h * n + sink + d] =
                        crate::attention::dot_f32(&qrot[h * hd..(h + 1) * hd], kr) * scale;
                }
            }
            // One softmax per head over sinks ∪ window (attention_head order).
            for h in 0..nh {
                let sc = &mut scores[h * n..(h + 1) * n];
                let max = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in sc.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in sc.iter_mut() {
                        *v /= sum;
                    }
                }
            }
            out.fill(0.0);
            for h in 0..nh {
                let g = h / hpk;
                let oh = &mut out[h * hd..(h + 1) * hd];
                let p = &scores[h * n..(h + 1) * n];
                for s in 0..sink {
                    let vr = &sink_v[(g * sink + s) * hd..(g * sink + s + 1) * hd];
                    if p[s].abs() >= 1e-12 {
                        crate::attention::axpy_f32(oh, vr, p[s]);
                    }
                }
                for d in 0..m {
                    let slot = (head + w - 1 - d) % w;
                    let vr = &self.ring_v[(g * w + slot) * hd..(g * w + slot + 1) * hd];
                    let pw = p[sink + d];
                    if pw.abs() >= 1e-12 {
                        crate::attention::axpy_f32(oh, vr, pw);
                    }
                }
            }
        });
    }
}

/// Per-token configuration of a bounded layer (geometry + the shared
/// rotation table). No position: the operator has none.
pub struct BoundedAttnCfg<'a> {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    /// Score scale (1/√hd unless the arch overrides).
    pub scale: f32,
    pub rope: &'a BoundedRope,
    pub pool: Option<&'a crate::pool::Pool>,
}

/// One position: `q̂ k̂ v = W x`, insert, attend, `W_o`. The cache's
/// `bounded` record must exist (installed from the header at load).
pub fn bounded_attention(
    hidden: &[f32],
    w: &BoundedWeights,
    cache: &mut crate::kv_cache::LayerKvCache,
    cfg: &BoundedAttnCfg,
) -> Vec<f32> {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let mut q = crate::attention::take_buf(nh * hd);
    let mut k = crate::attention::take_buf(nkv * hd);
    let mut v = crate::attention::take_buf(nkv * hd);
    w.wq.matvec(hidden, &mut q, cfg.pool);
    w.wk.matvec(hidden, &mut k, cfg.pool);
    w.wv.matvec(hidden, &mut v, cfg.pool);
    let mut ao = crate::attention::take_buf(nh * hd);
    cache.bounded_step(&q, &k, &v, w, cfg.rope, cfg.scale, nh, &mut ao);
    let mut out = crate::attention::take_buf(cfg.hidden_size);
    w.wo.matvec(&ao, &mut out, cfg.pool);
    crate::attention::recycle_buf(&mut q);
    crate::attention::recycle_buf(&mut k);
    crate::attention::recycle_buf(&mut v);
    crate::attention::recycle_buf(&mut ao);
    out
}

/// A prefill chunk of `b` positions: the projections run as chunk
/// GEMMs (each weight row streams once per chunk), the operator runs
/// per position over ring + chunk — the scores of a position are
/// against at most `S + W` keys whatever the chunk or the context, and
/// the per-position arithmetic is the same code as `bounded_attention`.
pub fn bounded_attention_batch(
    normed_all: &[f32],
    b: usize,
    w: &BoundedWeights,
    cache: &mut crate::kv_cache::LayerKvCache,
    cfg: &BoundedAttnCfg,
) -> Vec<f32> {
    let (nh, nkv, hd, hs) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size);
    debug_assert_eq!(normed_all.len(), b * hs);
    let mut q_all = crate::attention::take_buf(b * nh * hd);
    let mut k_all = crate::attention::take_buf(b * nkv * hd);
    let mut v_all = crate::attention::take_buf(b * nkv * hd);
    w.wq.matmat(normed_all, b, &mut q_all, cfg.pool);
    w.wk.matmat(normed_all, b, &mut k_all, cfg.pool);
    w.wv.matmat(normed_all, b, &mut v_all, cfg.pool);
    let mut ao_all = crate::attention::take_buf(b * nh * hd);
    for bi in 0..b {
        let q = &q_all[bi * nh * hd..(bi + 1) * nh * hd];
        let k = &k_all[bi * nkv * hd..(bi + 1) * nkv * hd];
        let v = &v_all[bi * nkv * hd..(bi + 1) * nkv * hd];
        let ao = &mut ao_all[bi * nh * hd..(bi + 1) * nh * hd];
        cache.bounded_step(q, k, v, w, cfg.rope, cfg.scale, nh, ao);
    }
    let mut out = crate::attention::take_buf(b * hs);
    w.wo.matmat(&ao_all, b, &mut out, cfg.pool);
    crate::attention::recycle_buf(&mut q_all);
    crate::attention::recycle_buf(&mut k_all);
    crate::attention::recycle_buf(&mut v_all);
    crate::attention::recycle_buf(&mut ao_all);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, salt: u64, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
                let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
                (((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32) * scale
            })
            .collect()
    }

    /// `(R(t−j) q)·k == (R(t) q)·(R(j) k)` — the relative table against
    /// the absolute `rope_rotate_scaled` of the full-attention path, so
    /// the sign convention is the runtime's own, not assumed.
    #[test]
    fn relative_rotation_equals_absolute_pair() {
        let hd = 16;
        let inv = crate::attention::rope_inv_freq(hd, 10_000.0);
        let rope = BoundedRope::new(64, &inv, 1.0);
        for (t, j) in [(0usize, 0usize), (5, 5), (7, 3), (63, 0), (300, 250), (1000, 990)] {
            let q = synth(hd, t as u64 + 1, 1.0);
            let k = synth(hd, j as u64 + 77, 1.0);
            let mut qa = q.clone();
            let mut ka = k.clone();
            crate::attention::rope_rotate_scaled(&mut qa, t, &inv, 1.0);
            crate::attention::rope_rotate_scaled(&mut ka, j, &inv, 1.0);
            let absolute: f64 = qa.iter().zip(&ka).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let mut qr = vec![0.0; hd];
            rope.rotate(t - j, &q, &mut qr);
            let relative: f64 = qr.iter().zip(&k).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            assert!(
                (absolute - relative).abs() < 2e-4,
                "t={t} j={j}: absolute {absolute} vs relative {relative}"
            );
        }
    }

    #[test]
    fn rollback_restores_overwritten_slots_bit_for_bit() {
        let (kvh, hd, w) = (2, 4, 8);
        let mut st = BoundedState::new(kvh, hd, w);
        for p in 0..20 {
            st.insert(&synth(kvh * hd, p, 1.0), &synth(kvh * hd, 100 + p, 1.0));
        }
        let snap = st.snapshot();
        for p in 20..25 {
            st.insert(&synth(kvh * hd, p, 1.0), &synth(kvh * hd, 100 + p, 1.0));
        }
        assert_eq!(st.rollback(5), 5);
        assert!(st.same_state(&{
            let mut s = BoundedState::new(kvh, hd, w);
            s.restore(&snap);
            s
        }));
        assert_eq!(st.seen, 20);
        assert_eq!(st.len(), w);
    }
}
