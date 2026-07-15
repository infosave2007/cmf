//! KV cache — per-layer, head-major storage.
//!
//! Layout: one contiguous `Vec<f32>` per KV head (`[pos × head_dim]`),
//! so per-head attention reads a straight slice — no per-head gather
//! copies per token. Dead GQA groups (all Q heads masked) store
//! nothing at all: masked heads cost neither FLOPs nor memory.

/// KV storage mode. `CMF_KV=q8` enables the q8_2f cache: an int8 row per
/// (position, head) + an f32 scale per row + a per-channel field 𝒲×θ,
/// frozen after WARMUP positions with retroactive requantization
/// (D4: "KV-quant 2f"). Memory ×~3.7 smaller than f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    F32,
    /// Quantized components: (K, V) — sensitivity diagnostics.
    Q8 { k: bool, v: bool },
}

impl KvMode {
    pub fn from_env() -> Self {
        match std::env::var("CMF_KV").as_deref() {
            Ok("q8") | Ok("q8_2f") => KvMode::Q8 { k: true, v: true },
            Ok("q8k") => KvMode::Q8 { k: true, v: false },
            Ok("q8v") => KvMode::Q8 { k: false, v: true },
            _ => KvMode::F32,
        }
    }

    fn quant_k(self) -> bool {
        matches!(self, KvMode::Q8 { k: true, .. })
    }

    fn quant_v(self) -> bool {
        matches!(self, KvMode::Q8 { v: true, .. })
    }
}

/// Positions before freezing the per-channel field (2f): before — col ≡ 1,
/// after — col = RMS over channels of the stored rows, old rows are requantized.
const KV_COL_WARMUP: usize = 64;

/// K-rows are quantized in groups of 32 channels (scale per group):
/// attention logits are sensitive to the dot-product error, per-group scales
/// localize it along RoPE bands (35B: +4.6% PPL with a per-row scale
/// → target <1% with a per-group one). V — per-row scale (measured +0.56%).
const KV_K_GROUP: usize = 32;

/// Per-layer O(1) Nyström attention state (runtime `attn_type`
/// override — spec §7 presence-driven pattern, no format change).
///
/// Collecting: the prompt pass still runs EXACT cache attention (the
/// prefill outputs feed the residual stream, so they cannot be
/// deferred) while the per-position rotated queries are buffered;
/// `o1_seal()` then freezes landmarks + M from the full prompt, replays
/// it into per-Q-head streaming states, and DROPS the full KV. Sealed:
/// decode replaces cache attention with `NystromState::step()`.
#[derive(Debug, Clone)]
pub enum O1State {
    Collecting {
        m: usize,
        w: usize,
        sink: usize,
        rect: crate::nystrom::O1Rect,
        /// Rotated post-norm queries, `[pos × num_heads × head_dim]`.
        q_buf: Vec<f32>,
    },
    /// One state per Q head: the far field T̂/Ẑ depends on that head's
    /// own query landmarks, so GQA groups cannot share it. The window
    /// K/V is duplicated across the group's Q heads (×heads_per_kv) —
    /// the price of keeping the golden-parity kernel intact; still O(1)
    /// in context and far below full KV at depth.
    Sealed { states: Vec<crate::nystrom::NystromState> },
}

/// KV cache for a single layer, head-major.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    pub mode: KvMode,
    /// Per-KV-head keys: `k[h]` is `[seq_len × head_dim]` (empty if head is dead).
    k: Vec<Vec<f32>>,
    /// Per-KV-head values, same layout.
    v: Vec<Vec<f32>>,
    /// q8 storage (mode == Q8_2F): int8 rows + f32 scale per row.
    kq: Vec<Vec<i8>>,
    ks: Vec<Vec<f32>>,
    vq: Vec<Vec<i8>>,
    vs: Vec<Vec<f32>>,
    /// Per-channel fields 𝒲×θ per head [head_dim]; empty until frozen.
    kcol: Vec<Vec<f32>>,
    vcol: Vec<Vec<f32>>,
    /// Accumulated attention mass per stored position (Born rule:
    /// importance of a position = how much probability mass reads it).
    imp: Vec<f32>,
    /// Positions appended so far (grows once per token, dead heads included).
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Linear-core condensate S (vmf_phase), f64; empty on full layers.
    pub linear_state: Vec<f64>,
    /// Tentative lane-2 state during speculative verify.
    pub linear_scratch: Vec<f64>,
    /// O(1) Nyström override (None = plain cache attention).
    pub o1: Option<O1State>,
}

impl LayerKvCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            mode: KvMode::from_env(),
            k: vec![Vec::new(); num_kv_heads],
            v: vec![Vec::new(); num_kv_heads],
            kq: vec![Vec::new(); num_kv_heads],
            ks: vec![Vec::new(); num_kv_heads],
            vq: vec![Vec::new(); num_kv_heads],
            vs: vec![Vec::new(); num_kv_heads],
            kcol: vec![Vec::new(); num_kv_heads],
            vcol: vec![Vec::new(); num_kv_heads],
            imp: Vec::new(),
            seq_len: 0,
            num_kv_heads,
            head_dim,
            linear_state: Vec::new(),
            linear_scratch: Vec::new(),
            o1: None,
        }
    }

    // ── O(1) Nyström override ──

    /// Arm query collection for a fresh prompt pass (a cleared cache).
    pub fn o1_begin(&mut self, m: usize, w: usize, sink: usize, rect: crate::nystrom::O1Rect) {
        self.o1 = Some(O1State::Collecting { m, w, sink, rect, q_buf: Vec::new() });
    }

    /// Record one position's rotated queries (`[num_heads × head_dim]`)
    /// during the exact prompt pass. No-op unless collecting — the hook
    /// sits inside qwen_attention so every prefill flavor (sequential,
    /// batched) feeds the same trace.
    pub fn o1_push_q(&mut self, q_all: &[f32]) {
        if let Some(O1State::Collecting { q_buf, .. }) = &mut self.o1 {
            q_buf.extend_from_slice(q_all);
        }
    }

    pub fn o1_sealed(&self) -> bool {
        matches!(self.o1, Some(O1State::Sealed { .. }))
    }

    /// Freeze the prompt into per-Q-head Nyström states and drop this
    /// layer's full KV. Returns false (layer stays exact, KV kept) when
    /// the preconditions fail: the seal needs f32 KV rows (`CMF_KV=q8`
    /// stores int8), every group densely stored, and a full q trace.
    pub fn o1_seal(&mut self, num_heads: usize) -> bool {
        // Idempotent: sealing a sealed (or plain) layer must not
        // disturb its state — check before take().
        if !matches!(self.o1, Some(O1State::Collecting { .. })) {
            return self.o1_sealed();
        }
        let Some(O1State::Collecting { m, w, sink, rect, q_buf }) = self.o1.take() else {
            unreachable!("checked above");
        };
        let (hd, t) = (self.head_dim, self.seq_len);
        let hpk = num_heads / self.num_kv_heads.max(1);
        let ok = t > 0
            && self.mode == KvMode::F32
            && q_buf.len() == t * num_heads * hd
            && (0..self.num_kv_heads).all(|g| self.head_len(g) == t);
        if !ok {
            tracing::warn!(
                "o1: cannot seal (needs f32 KV mode, dense heads, full query \
                 trace) — layer keeps exact attention"
            );
            return false;
        }
        let mut states = Vec::with_capacity(num_heads);
        let mut qh = vec![0.0f32; t * hd];
        for h in 0..num_heads {
            let g = h / hpk;
            for p in 0..t {
                let src = (p * num_heads + h) * hd;
                qh[p * hd..(p + 1) * hd].copy_from_slice(&q_buf[src..src + hd]);
            }
            let mut st = crate::nystrom::NystromState::new(m, w, sink).with_rect(rect);
            st.prefill(&qh, &self.k[g], &self.v[g], t, hd, hd);
            states.push(st);
        }
        // The states now carry everything decode needs — release the
        // O(context) storage (this is the memory claim, not a cosmetic).
        for h in 0..self.num_kv_heads {
            self.k[h] = Vec::new();
            self.v[h] = Vec::new();
        }
        self.imp = Vec::new();
        self.o1 = Some(O1State::Sealed { states });
        true
    }

    /// One decode step on a sealed layer: per Q head, insert the GQA
    /// group's fresh (k, v) and read the attention output. Returns
    /// `[num_heads × head_dim]`. The same (k, v) is inserted into each
    /// of the group's per-Q-head states — same math as the shared KV
    /// row the exact path would have appended once.
    pub fn o1_step(
        &mut self,
        q_all: &[f32],
        k_new: &[f32],
        v_new: &[f32],
        num_heads: usize,
    ) -> Vec<f32> {
        let hd = self.head_dim;
        let hpk = num_heads / self.num_kv_heads.max(1);
        let mut out = vec![0.0f32; num_heads * hd];
        let Some(O1State::Sealed { states }) = &mut self.o1 else {
            debug_assert!(false, "o1_step on an unsealed layer");
            return out;
        };
        for h in 0..num_heads {
            let g = h / hpk;
            states[h].step(
                &q_all[h * hd..(h + 1) * hd],
                &k_new[g * hd..(g + 1) * hd],
                &v_new[g * hd..(g + 1) * hd],
                &mut out[h * hd..(h + 1) * hd],
            );
        }
        // Track the true context depth for the honest memory/seq report
        // (nothing is stored per position — the state is O(1)).
        self.seq_len += 1;
        out
    }

    /// Bytes held by the O(1) override (query trace while collecting,
    /// per-head states once sealed).
    pub fn o1_memory_bytes(&self) -> usize {
        match &self.o1 {
            Some(O1State::Collecting { q_buf, .. }) => {
                q_buf.len() * std::mem::size_of::<f32>()
            }
            Some(O1State::Sealed { states }) => {
                states.iter().map(|s| s.memory_bytes()).sum()
            }
            None => 0,
        }
    }

    /// Quantize one row against the per-channel field (empty col = 1);
    /// `group` — elements per scale (the whole row or KV_K_GROUP).
    fn quant_row(row: &[f32], col: &[f32], q: &mut Vec<i8>, sc: &mut Vec<f32>,
                 group: usize) {
        let mut resid = vec![0.0f32; row.len()];
        for (d, &x) in row.iter().enumerate() {
            resid[d] = if col.is_empty() { x } else { x / col[d] };
        }
        for g0 in (0..row.len()).step_by(group) {
            let g1 = (g0 + group).min(row.len());
            let mut absmax = 0.0f32;
            for &r in &resid[g0..g1] {
                absmax = absmax.max(r.abs());
            }
            let s = (absmax / 127.0).max(1e-12);
            sc.push(s);
            for &r in &resid[g0..g1] {
                q.push((r / s).round().clamp(-127.0, 127.0) as i8);
            }
        }
    }

    /// Freeze the 2f field: col = RMS of channels over stored rows, old
    /// rows are requantized against the new field (once per conversation).
    fn freeze_cols(&mut self) {
        let hd = self.head_dim;
        let ngk = hd.div_ceil(KV_K_GROUP);
        for h in 0..self.num_kv_heads {
            for (qv, sv, colv, group) in [
                (&mut self.kq[h], &mut self.ks[h], &mut self.kcol[h], KV_K_GROUP),
                (&mut self.vq[h], &mut self.vs[h], &mut self.vcol[h], hd),
            ] {
                let spp = if group == hd { 1 } else { ngk }; // scales per position
                let n = sv.len() / spp;
                if n == 0 {
                    continue;
                }
                // Dequantize to f32, RMS over channels, requantize.
                let mut rows = vec![0.0f32; n * hd];
                for p in 0..n {
                    for d in 0..hd {
                        rows[p * hd + d] =
                            qv[p * hd + d] as f32 * sv[p * spp + d / group];
                    }
                }
                let mut col = vec![0.0f32; hd];
                for p in 0..n {
                    for d in 0..hd {
                        col[d] += rows[p * hd + d] * rows[p * hd + d];
                    }
                }
                for c in col.iter_mut() {
                    *c = (*c / n as f32).sqrt().max(1e-6);
                }
                qv.clear();
                sv.clear();
                for p in 0..n {
                    Self::quant_row(&rows[p * hd..(p + 1) * hd], &col, qv, sv, group);
                }
                *colv = col;
            }
        }
    }

    /// Append K/V for one position. `k_new`/`v_new` are
    /// `[num_kv_heads × head_dim]`; heads with `alive[h] == false` are
    /// skipped (their slices stay empty).
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32], alive: &[bool]) {
        debug_assert_eq!(k_new.len(), self.num_kv_heads * self.head_dim);
        debug_assert_eq!(v_new.len(), self.num_kv_heads * self.head_dim);
        // Freeze the 2f field AT THE START of append: only rows that
        // survived verify are visible (a rejected lane-2 draft does not
        // pollute the field — found in review), and the threshold uses >=
        // rather than strict equality (in small windows eviction may
        // oscillate across 64).
        if matches!(self.mode, KvMode::Q8 { .. })
            && self.seq_len >= KV_COL_WARMUP
            && self.kcol.iter().all(Vec::is_empty)
            && self.vcol.iter().all(Vec::is_empty)
        {
            self.freeze_cols();
        }
        for h in 0..self.num_kv_heads {
            if !alive.get(h).copied().unwrap_or(true) {
                continue;
            }
            let s = h * self.head_dim;
            if self.mode.quant_k() {
                Self::quant_row(&k_new[s..s + self.head_dim],
                                &self.kcol[h], &mut self.kq[h], &mut self.ks[h],
                                KV_K_GROUP);
            } else {
                self.k[h].extend_from_slice(&k_new[s..s + self.head_dim]);
            }
            if self.mode.quant_v() {
                Self::quant_row(&v_new[s..s + self.head_dim],
                                &self.vcol[h], &mut self.vq[h], &mut self.vs[h],
                                self.head_dim);
            } else {
                self.v[h].extend_from_slice(&v_new[s..s + self.head_dim]);
            }
        }
        self.imp.push(0.0);
        self.seq_len += 1;
    }

    /// Per-head attention over its own storage: the f32 branch is
    /// bit-for-bit equal to attention_head() over slices; the q8 branch
    /// computes score = s_k·⟨q⊙col_k, k_q⟩ and the weighted sum of V in i8
    /// with f32 accumulation. Returns (output [head_dim], probs [stored]).
    pub fn attend(&self, q: &[f32], kv_head: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim;
        if self.mode == KvMode::F32 {
            let stored = self.k[kv_head].len() / hd;
            return crate::attention::attention_head(
                q, &self.k[kv_head], &self.v[kv_head], hd, stored);
        }
        let stored = self.head_len(kv_head);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; stored];
        if self.mode.quant_k() {
            let (kq, ks) = (&self.kq[kv_head], &self.ks[kv_head]);
            // q ⊙ col_k — once per call.
            let kcol = &self.kcol[kv_head];
            let mut qc = vec![0.0f32; hd];
            for d in 0..hd {
                qc[d] = if kcol.is_empty() { q[d] } else { q[d] * kcol[d] };
            }
            let ng = hd.div_ceil(KV_K_GROUP);
            for p in 0..stored {
                let row = &kq[p * hd..(p + 1) * hd];
                // SAFETY: i8 and u8 share layout; dot_i8_f32 reads the
                // bytes back as i8.
                let row_u8 = unsafe {
                    std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len())
                };
                let mut dot = 0.0f32;
                for g in 0..ng {
                    let g0 = g * KV_K_GROUP;
                    let g1 = (g0 + KV_K_GROUP).min(hd);
                    dot += crate::qtensor::dot_i8_f32(&row_u8[g0..g1], &qc[g0..g1])
                        * ks[p * ng + g];
                }
                scores[p] = dot * scale;
            }
        } else {
            let k = &self.k[kv_head];
            for p in 0..stored {
                let row = &k[p * hd..(p + 1) * hd];
                scores[p] = crate::attention::dot_f32(q, row) * scale;
            }
        }
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
        let mut acc = vec![0.0f32; hd];
        if self.mode.quant_v() {
            let (vq, vs) = (&self.vq[kv_head], &self.vs[kv_head]);
            for p in 0..stored {
                let w = scores[p] * vs[p];
                if w.abs() < 1e-12 {
                    continue;
                }
                crate::qtensor::axpy_i8_f32(&mut acc, &vq[p * hd..(p + 1) * hd], w);
            }
            let vcol = &self.vcol[kv_head];
            if !vcol.is_empty() {
                for d in 0..hd {
                    acc[d] *= vcol[d];
                }
            }
        } else {
            let v = &self.v[kv_head];
            for p in 0..stored {
                let w = scores[p];
                if w.abs() < 1e-12 {
                    continue;
                }
                crate::attention::axpy_f32(&mut acc, &v[p * hd..(p + 1) * hd], w);
            }
        }
        (acc, scores)
    }

    /// Roll back the last `n_drop` positions (speculative-decode reject).
    pub fn truncate_last(&mut self, n_drop: usize) {
        let d = n_drop.min(self.seq_len);
        for h in 0..self.num_kv_heads {
            let keep = self.k[h].len().saturating_sub(d * self.head_dim);
            self.k[h].truncate(keep);
            self.v[h].truncate(keep);
            let ngk = self.head_dim.div_ceil(KV_K_GROUP);
            let keep_q = self.kq[h].len().saturating_sub(d * self.head_dim);
            self.kq[h].truncate(keep_q);
            let keep_vq = self.vq[h].len().saturating_sub(d * self.head_dim);
            self.vq[h].truncate(keep_vq);
            let keep_ks = self.ks[h].len().saturating_sub(d * ngk);
            self.ks[h].truncate(keep_ks);
            let keep_vs = self.vs[h].len().saturating_sub(d);
            self.vs[h].truncate(keep_vs);
        }
        self.imp.truncate(self.imp.len().saturating_sub(d));
        self.seq_len -= d;
    }

    /// Accumulate attention mass per stored position (summed over heads).
    pub fn accumulate_imp(&mut self, probs: &[f32]) {
        for (dst, &p) in self.imp.iter_mut().zip(probs) {
            *dst += p;
        }
    }

    /// Contiguous keys of one head: `[stored_len × head_dim]`.
    pub fn head_keys(&self, kv_head: usize) -> &[f32] {
        &self.k[kv_head]
    }

    pub fn head_values(&self, kv_head: usize) -> &[f32] {
        &self.v[kv_head]
    }

    /// Number of positions actually stored for a head (0 for dead heads).
    pub fn head_len(&self, kv_head: usize) -> usize {
        let ng = self.head_dim.div_ceil(KV_K_GROUP);
        (self.k[kv_head].len() / self.head_dim)
            .max(self.ks[kv_head].len() / ng)
            .max(self.vs[kv_head].len())
    }

    /// Clear cache (e.g. on new conversation or task switch).
    pub fn clear(&mut self) {
        for h in 0..self.num_kv_heads {
            self.k[h].clear();
            self.v[h].clear();
            self.kq[h].clear();
            self.ks[h].clear();
            self.vq[h].clear();
            self.vs[h].clear();
            self.kcol[h].clear();
            self.vcol[h].clear();
        }
        self.imp.clear();
        self.linear_state.clear();
        self.linear_scratch.clear();
        // Fresh conversation → the pipeline re-arms collection if the
        // layer is o1-flagged (landmarks are per-prompt, never reused).
        self.o1 = None;
        self.seq_len = 0;
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let floats: usize = self.k.iter().map(Vec::len).sum::<usize>()
            + self.v.iter().map(Vec::len).sum::<usize>()
            + self.ks.iter().map(Vec::len).sum::<usize>()
            + self.vs.iter().map(Vec::len).sum::<usize>()
            + self.kcol.iter().map(Vec::len).sum::<usize>()
            + self.vcol.iter().map(Vec::len).sum::<usize>();
        let bytes: usize = self.kq.iter().map(Vec::len).sum::<usize>()
            + self.vq.iter().map(Vec::len).sum::<usize>();
        floats * std::mem::size_of::<f32>()
            + bytes
            // O(1) recurrent state of linear-core layers (vmf_phase/GDN):
            // constant in context, but real memory — the honest "KV+state"
            // line must count it (a pure-linear model reported 0 before).
            + self.linear_state.len() * std::mem::size_of::<f64>()
            // O(1) Nyström state (window + sinks + skeleton) — same
            // discipline: constant in context, but real memory.
            + self.o1_memory_bytes()
    }

    /// Drop oldest positions, keeping the last `keep_last`.
    fn evict(&mut self, keep_last: usize) {
        // A sealed o1 layer stores nothing per position — the Nyström
        // state IS the eviction policy; resetting seq_len here would lie
        // about the context depth.
        if self.o1_sealed() || self.seq_len <= keep_last {
            return;
        }
        let drop = self.seq_len - keep_last;
        for h in 0..self.num_kv_heads {
            // Dead heads store fewer positions; drop proportionally.
            let stored = self.head_len(h);
            let d = drop.min(stored);
            let hd = self.head_dim;
            fn drop_front<T>(v: &mut Vec<T>, n: usize) {
                let n = n.min(v.len());
                v.drain(..n);
            }
            drop_front(&mut self.k[h], d * hd);
            drop_front(&mut self.v[h], d * hd);
            drop_front(&mut self.kq[h], d * hd);
            drop_front(&mut self.vq[h], d * hd);
            drop_front(&mut self.ks[h], d * hd.div_ceil(KV_K_GROUP));
            drop_front(&mut self.vs[h], d);
        }
        let d = drop.min(self.imp.len());
        self.imp.drain(..d);
        self.seq_len = keep_last;
    }

    /// Born eviction: keep `sink` earliest positions (attention sinks),
    /// the `recent` latest, and fill the rest of the `keep_last` budget
    /// with the positions carrying the highest accumulated attention
    /// mass (vmfcore: PPL 8.342 vs 8.687 for recency-only, full 8.295).
    fn evict_born(&mut self, keep_last: usize, sink: usize, recent: usize) {
        if self.o1_sealed() {
            return; // see evict(): the o1 state is its own eviction
        }
        let stored = self.imp.len();
        if stored <= keep_last {
            return;
        }
        // Budget discipline: sinks first, recents next, both clamped so
        // the total never exceeds keep_last.
        let sink_n = sink.min(keep_last);
        let recent_n = recent.min(keep_last - sink_n);
        let mut keep = vec![false; stored];
        for k in keep.iter_mut().take(sink_n) {
            *k = true;
        }
        for k in keep.iter_mut().skip(stored.saturating_sub(recent_n)) {
            *k = true;
        }
        let mut budget = keep_last.saturating_sub(keep.iter().filter(|&&x| x).count());
        // Highest accumulated mass first among the middle positions.
        let mut order: Vec<usize> = (0..stored).filter(|&i| !keep[i]).collect();
        order.sort_by(|&a, &b| {
            self.imp[b].partial_cmp(&self.imp[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in order {
            if budget == 0 {
                break;
            }
            keep[i] = true;
            budget -= 1;
        }

        let kept: Vec<usize> = (0..stored).filter(|&i| keep[i]).collect();
        let hd = self.head_dim;
        fn gather<T: Copy>(src: &[T], kept: &[usize], step: usize) -> Vec<T> {
            let mut out = Vec::with_capacity(kept.len() * step);
            for &i in kept {
                out.extend_from_slice(&src[i * step..(i + 1) * step]);
            }
            out
        }
        // Each storage is gathered INDEPENDENTLY: in mixed modes
        // (q8k/q8v) K and V live in different storages — the paired branch
        // panicked (q8v) or silently left V uncompressed (q8k);
        // found by adversarial review, closed by regression tests.
        for h in 0..self.num_kv_heads {
            if !self.k[h].is_empty() {
                self.k[h] = gather(&self.k[h], &kept, hd);
            }
            if !self.v[h].is_empty() {
                self.v[h] = gather(&self.v[h], &kept, hd);
            }
            if !self.kq[h].is_empty() {
                self.kq[h] = gather(&self.kq[h], &kept, hd);
                self.ks[h] = gather(&self.ks[h], &kept, hd.div_ceil(KV_K_GROUP));
            }
            if !self.vq[h].is_empty() {
                self.vq[h] = gather(&self.vq[h], &kept, hd);
                self.vs[h] = gather(&self.vs[h], &kept, 1);
            }
        }
        self.imp = kept.iter().map(|&i| self.imp[i]).collect();
        self.seq_len = kept.len();
    }
}

/// Eviction policy for a bounded cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Sliding window: keep only the most recent positions.
    Recent,
    /// Born rule: sinks + recents + top accumulated attention mass.
    Born { sink: usize },
}

/// Full KV cache for all layers.
#[derive(Debug)]
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub max_seq_len: usize,
    pub policy: EvictionPolicy,
}

impl KvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(num_kv_heads, head_dim))
            .collect();
        Self {
            layers,
            max_seq_len,
            policy: EvictionPolicy::Born { sink: 4 },
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Current sequence length (max across layers — dead layers may lag).
    pub fn seq_len(&self) -> usize {
        self.layers.iter().map(|l| l.seq_len).max().unwrap_or(0)
    }

    pub fn needs_eviction(&self) -> bool {
        self.seq_len() >= self.max_seq_len
    }

    /// Evict down to `keep_last` positions according to the policy.
    pub fn evict(&mut self, keep_last: usize) {
        match self.policy {
            EvictionPolicy::Recent => {
                for layer in &mut self.layers {
                    layer.evict(keep_last);
                }
            }
            EvictionPolicy::Born { sink } => {
                let recent = (keep_last / 2).max(1);
                for layer in &mut self.layers {
                    layer.evict_born(keep_last, sink, recent);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_tracks_seq_len_and_layout() {
        let mut cache = LayerKvCache::new(4, 8);
        cache.mode = KvMode::F32;
        assert_eq!(cache.seq_len, 0);

        let k: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let v = vec![2.0f32; 32];
        cache.append(&k, &v, &[true; 4]);

        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.head_len(0), 1);
        // head 1 slice is contiguous and equals its part of k_new
        assert_eq!(cache.head_keys(1), &k[8..16]);
        assert_eq!(cache.memory_bytes(), 256);
    }

    #[test]
    fn dead_head_stores_nothing() {
        let mut cache = LayerKvCache::new(2, 4);
        cache.mode = KvMode::F32;
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(&k, &v, &[true, false]);
        cache.append(&k, &v, &[true, false]);

        assert_eq!(cache.seq_len, 2);
        assert_eq!(cache.head_len(0), 2);
        assert_eq!(cache.head_len(1), 0, "dead head must not store KV");
        assert_eq!(cache.memory_bytes(), 2 * 2 * 4 * 4);
    }

    #[test]
    fn eviction_keeps_recent() {
        let mut cache = KvCache::new(2, 4, 8, 10);
        cache.policy = EvictionPolicy::Recent;
        for l in &mut cache.layers { l.mode = KvMode::F32; }
        let k = vec![1.0f32; 32];
        let v = vec![2.0f32; 32];
        for _ in 0..8 {
            for layer in &mut cache.layers {
                layer.append(&k, &v, &[true; 4]);
            }
        }
        assert_eq!(cache.seq_len(), 8);
        assert!(!cache.needs_eviction());

        cache.evict(4);
        assert_eq!(cache.seq_len(), 4);
        assert_eq!(cache.layers[0].head_len(0), 4);
    }

    #[test]
    fn truncate_rolls_back_speculative_positions() {
        let mut cache = LayerKvCache::new(2, 4);
        cache.mode = KvMode::F32;
        for pos in 0..5 {
            let k = vec![pos as f32; 8];
            let v = vec![pos as f32; 8];
            cache.append(&k, &v, &[true; 2]);
        }
        cache.truncate_last(2);
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.head_len(0), 3);
        assert_eq!(cache.head_keys(0)[2 * 4], 2.0, "position 2 survives");
    }

    /// q8_2f-attend ≈ f32-attend: 100 positions (crosses the field freeze
    /// at the 64th), pseudo-random vectors, relative tolerance of the
    /// int8 grid. Plus rollback and Born eviction on the q8 storage.
    #[test]
    fn q8_attend_matches_f32_within_grid() {
        let (heads, hd) = (2, 32);
        let mut f = LayerKvCache::new(heads, hd);
        f.mode = KvMode::F32;
        let mut q8 = LayerKvCache::new(heads, hd);
        q8.mode = KvMode::Q8 { k: true, v: true };

        let synth = |p: usize, salt: usize| -> Vec<f32> {
            (0..heads * hd)
                .map(|i| {
                    let x = ((i * 31 + p * 17 + salt * 7 + 3) % 97) as f32 / 97.0 - 0.5;
                    // channel structure: even channels ×4 (checks the 2f field)
                    if i % 2 == 0 { x * 4.0 } else { x * 0.25 }
                })
                .collect()
        };
        for p in 0..100 {
            let k = synth(p, 1);
            let v = synth(p, 2);
            f.append(&k, &v, &[true; 2]);
            q8.append(&k, &v, &[true; 2]);
        }
        let q: Vec<f32> = (0..hd).map(|i| ((i * 13 + 5) % 89) as f32 / 89.0 - 0.5).collect();
        for g in 0..heads {
            let (of, pf) = f.attend(&q, g);
            let (o8, p8) = q8.attend(&q, g);
            let scale = of.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
            for d in 0..hd {
                assert!(
                    (of[d] - o8[d]).abs() <= scale * 0.03 + 1e-3,
                    "g{g} d{d}: f32 {} vs q8 {}", of[d], o8[d]
                );
            }
            for p in 0..100 {
                assert!((pf[p] - p8[p]).abs() < 0.02, "prob p{p}");
            }
        }
        // rollback + eviction live on the q8 storage
        q8.truncate_last(30);
        assert_eq!(q8.head_len(0), 70);
        let imp: Vec<f32> = (0..70).map(|i| i as f32).collect();
        q8.accumulate_imp(&imp);
        q8.evict_born(20, 2, 8);
        assert_eq!(q8.head_len(0), 20);
        let (o, _) = q8.attend(&q, 0);
        assert!(o.iter().all(|x| x.is_finite()));
        // memory: q8 ≈ 1 byte/element + scale per row (vs 4 for f32)
        assert!(q8.memory_bytes() * 3 < f.memory_bytes());
    }

    /// Review regression: Born eviction in MIXED modes. q8v used to
    /// panic (gather over an empty v[h]), q8k silently left raw V
    /// uncompressed (stale rows under kept keys + memory leak).
    #[test]
    fn born_eviction_mixed_modes_stay_consistent() {
        for (mk, mv) in [(false, true), (true, false)] {
            let mut c = LayerKvCache::new(1, 4);
            c.mode = KvMode::Q8 { k: mk, v: mv };
            for p in 0..80 {
                let k = vec![p as f32 * 0.01; 4];
                let v = vec![p as f32; 4];
                c.append(&k, &v, &[true]);
            }
            let imp: Vec<f32> = (0..80).map(|i| i as f32).collect();
            c.accumulate_imp(&imp);
            let before = c.memory_bytes();
            c.evict_born(20, 4, 8); // q8v: used to panic here
            assert_eq!(c.head_len(0), 20, "k={mk} v={mv}");
            assert!(c.memory_bytes() < before / 2,
                    "memory must shrink (k={mk} v={mv})");
            // V rows match the kept set: the heaviest positions
            // (tail 60..79) must be present in the attend output.
            let (out, _) = c.attend(&[1.0, 1.0, 1.0, 1.0], 0);
            assert!(out[0] > 30.0,
                    "V from the kept tail, not the stale head (k={mk} v={mv}, out {})",
                    out[0]);
        }
    }

    #[test]
    fn born_eviction_keeps_high_mass_position() {
        let mut cache = KvCache::new(1, 1, 2, 16);
        cache.policy = EvictionPolicy::Born { sink: 1 };
        for l in &mut cache.layers { l.mode = KvMode::F32; }
        let layer = &mut cache.layers[0];
        // 8 positions; keys carry the position index so we can verify
        // exactly which positions survive the gather.
        for pos in 0..8 {
            let k = vec![pos as f32; 2];
            let v = vec![pos as f32 + 100.0; 2];
            layer.append(&k, &v, &[true]);
        }
        // Position 3 carries the most attention mass (Born importance).
        let mut imp = vec![0.05f32; 8];
        imp[3] = 5.0;
        layer.accumulate_imp(&imp);

        cache.evict(4); // sink 1 + recent 2 + 1 top-mass slot
        let layer = &cache.layers[0];
        assert_eq!(layer.seq_len, 4);
        let kept_keys: Vec<f32> = (0..4).map(|i| layer.head_keys(0)[i * 2]).collect();
        assert_eq!(
            kept_keys,
            vec![0.0, 3.0, 6.0, 7.0],
            "kept = sink(0) + Born-top(3) + recent(6,7)"
        );
        // imp stays aligned with the gathered positions.
        assert_eq!(layer.head_len(0), 4);
    }
}
