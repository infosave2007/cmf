//! KV cache — per-layer, head-major storage.
//!
//! Layout: one contiguous `Vec<f32>` per KV head (`[pos × head_dim]`),
//! so per-head attention reads a straight slice — no per-head gather
//! copies per token. Dead GQA groups (all Q heads masked) store
//! nothing at all: masked heads cost neither FLOPs nor memory.

/// KV storage mode. `CMF_KV=q8` enables the q8_2f cache: an int8 row per
/// (position, head) + an f32 scale per row + a per-channel scale field,
/// frozen after WARMUP positions with retroactive requantization
/// (D4: "KV-quant 2f"). Memory ×~3.7 smaller than f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    F32,
    /// Quantized components: (K, V) — sensitivity diagnostics.
    Q8 {
        k: bool,
        v: bool,
    },
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

/// Source of `LayerKvCache::generation`. Process-wide so two caches (or a
/// cache and a restored clone of itself from another moment) never share a
/// value by accident: a device mirror keyed by (generation, rows) can then tell
/// "the rows I hold" from "the same number of different rows".
static KV_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_gen() -> u64 {
    KV_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Per-layer O(1) Nyström attention state (runtime `attn_type`
/// override — spec §7 presence-driven pattern, no format change).
///
/// Collecting: the prompt pass still runs EXACT cache attention (the
/// prefill outputs feed the residual stream, so they cannot be
/// deferred) while the per-position rotated queries are buffered;
/// `o1_seal()` then freezes landmarks + M from the full prompt, replays
/// it into per-KV-group streaming states, and DROPS the full KV.
/// Sealed: decode replaces cache attention with
/// `NystromState::step_group()`.
#[derive(Debug, Clone)]
pub enum O1State {
    Collecting {
        m: usize,
        w: usize,
        sink: usize,
        rect: crate::nystrom::O1Rect,
        /// Optional completed-row barrier. `None` means the caller will
        /// request a full-prompt seal; `Some(B)` keeps a short prompt exact
        /// until the first skeleton-safe boundary B.
        seal_at: Option<usize>,
        /// Rotated post-norm queries, `[pos × num_heads × head_dim]`.
        q_buf: Vec<f32>,
    },
    /// One state per KV GROUP, each holding its group's Q heads. The
    /// exact window / sinks / K̃ are stored once per group (every Q head
    /// of the group reads the same k/v rows); only the far field, Q̃ and
    /// M — the query-dependent pieces — stay per Q head. See
    /// `NystromState` for which piece is which and why.
    Sealed {
        groups: Vec<crate::nystrom::NystromState>,
    },
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
    /// Per-channel scale fields per head [head_dim]; empty until frozen.
    kcol: Vec<Vec<f32>>,
    vcol: Vec<Vec<f32>>,
    /// Accumulated attention mass per stored position: importance of a
    /// position is how much probability mass reads it.
    imp: Vec<f32>,
    /// Rows stored (grows once per token, dead heads included). Stored
    /// row 0 is absolute position `base`, so this equals the context depth
    /// only while `base == 0`; [`Self::pos_len`] is the depth. Every attend
    /// indexes rows, never positions (eviction made this a row count long
    /// before `trim_window` did).
    pub seq_len: usize,
    /// Absolute position of stored row 0. Only `trim_window` advances it;
    /// 0 on every layer that was never trimmed.
    base: usize,
    /// The attend window of a layer that keeps only its tail
    /// (`trim_window`), set by the first call. A property of the layer,
    /// not of the conversation: `clear()` keeps it. Such a layer bounds
    /// itself, so the cache-wide eviction leaves it alone.
    tail: Option<usize>,
    /// Storage generation: a fresh value on every mutation other than
    /// `append` and `truncate_last` (clear, trim, eviction, wire import,
    /// resets). A device mirror that copies these rows keys its validity on
    /// (generation, rows): a trimmed layer's row count is periodic (576..639 for a
    /// 512 window), so the count alone can match a different set of rows.
    /// Appends and rollbacks show up in the count, and the speculative
    /// paths re-point their mirrors themselves, so they keep the value.
    generation: u64,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Linear-core recurrent state S (vmf_phase), f64; empty on full layers.
    pub linear_state: Vec<f32>,
    /// Tentative lane-2 state during speculative verify.
    pub linear_scratch: Vec<f32>,
    /// The legacy cache wire has no operator tag. Delta operators therefore
    /// bind this layer to a fail-closed boundary until a versioned wire
    /// schema can carry the linear-core identity.
    linear_wire_allowed: bool,
    /// O(1) Nyström override (None = plain cache attention).
    pub o1: Option<O1State>,
    /// A deferred seal failure is terminal for the current request. The
    /// attention functions return only a hidden vector, so the pipeline
    /// consumes this side channel at its next forward boundary.
    o1_error: Option<String>,
    /// Set when a collecting state actually becomes sealed. Pipeline owns
    /// the epoch bump and consumes this bit after a complete forward.
    o1_transitioned: bool,
    /// Learned per-Q-head attention-sink logits of this layer (gpt-oss /
    /// MiMo-V2 `self_attn.sinks`, one f32 per Q head). The sink is an
    /// extra softmax column with no value: it joins the max and the
    /// denominator of every head's softmax and so lets a head attend to
    /// "nothing". These are WEIGHTS, not sequence state — `clear()` and
    /// the wire import keep them. None = an ordinary softmax.
    pub sinks: Option<Vec<f32>>,
    /// Natively bounded anchor (`swa_sink_v1`): a fixed-size ring
    /// installed from the header at load, never per prompt. A layer that
    /// carries it stores NOTHING per position (`k`/`v` stay empty).
    pub bounded: Option<crate::bounded::BoundedState>,
    /// Which state record this layer exchanges on the wire (v2).
    pub wire_kind: WireKind,
    /// This layer's index in the stack (the wire header names it).
    pub wire_layer: u32,
    /// hash64 of the model's operator identity
    /// (`ModelArch::linear_core_identity` JSON); 0 = no operator record.
    pub wire_identity: u64,
    /// `kv_abs_max`'s running answer: max |x| over the f32 K rows and over
    /// the V rows `[0, kv_amax_rows)` of every head (+inf once one was inf
    /// or NaN).
    kv_amax: [f32; 2],
    kv_amax_rows: usize,
}

/// State-record kind of the versioned cache wire (`export_wire` v2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireKind {
    /// Per-position K/V (+ importance) — the legacy body.
    Full = 0,
    /// Recurrent state vector (S + conv ring), f32.
    Linear = 1,
    /// Bounded anchor: insert counter + ring K/V.
    Bounded = 2,
    /// A full layer that stores only its tail (`trim_window`): `base` as
    /// u64, then the `Full` body of the stored rows. Sent only when
    /// `base > 0`, so an untrimmed record is byte-identical to `Full`; a
    /// peer that predates it refuses it as an unknown kind instead of
    /// reading the tail as a whole history.
    FullTail = 3,
}

impl WireKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(WireKind::Full),
            1 => Some(WireKind::Linear),
            2 => Some(WireKind::Bounded),
            3 => Some(WireKind::FullTail),
            _ => None,
        }
    }
}

/// Magic of the versioned state wire.
pub const WIRE_MAGIC: &[u8; 4] = b"CMFS";
/// Current wire version.
pub const WIRE_VERSION: u32 = 2;

impl LayerKvCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            sinks: None,
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
            base: 0,
            tail: None,
            generation: next_gen(),
            num_kv_heads,
            head_dim,
            linear_state: Vec::new(),
            linear_scratch: Vec::new(),
            linear_wire_allowed: true,
            o1: None,
            o1_error: None,
            o1_transitioned: false,
            bounded: None,
            wire_kind: WireKind::Full,
            wire_layer: 0,
            wire_identity: 0,
            kv_amax: [0.0; 2],
            kv_amax_rows: 0,
        }
    }

    /// (max |k|, max |v|) over every K and over every V entry the cache
    /// stores in f32, +inf when one is inf or NaN: what the wgpu prefill
    /// compares with f16's range before its matrix-unit attention casts
    /// these rows to f16. Lazy — it scans only the rows appended since the
    /// last call; every other change to the rows (truncate, evict, clear,
    /// import, an o1 seal) moves the scanned mark back
    /// (`kv_rows_changed`), and a scan from the first row starts afresh.
    pub(crate) fn kv_abs_max(&mut self) -> (f32, f32) {
        let hd = self.head_dim.max(1);
        let rows = self.kv_rows();
        if rows < self.kv_amax_rows {
            self.kv_amax_rows = 0;
        }
        if self.kv_amax_rows == 0 {
            self.kv_amax = [0.0; 2];
        }
        let from = self.kv_amax_rows * hd;
        for (m, heads) in self.kv_amax.iter_mut().zip([&self.k, &self.v]) {
            for x in heads.iter().filter(|x| x.len() > from) {
                *m = m.max(crate::gpu::abs_max_or_inf(&x[from..]));
            }
        }
        self.kv_amax_rows = rows;
        (self.kv_amax[0], self.kv_amax[1])
    }

    /// `kv_abs_max` for a caller that took the maxima of the rows it just
    /// appended itself: `chunk` = (from, upto, max |k|, max |v|) over the
    /// rows `[from, upto)`. Folded in without a scan when the cache holds
    /// exactly `upto` rows and every row before `from` was scanned; else
    /// the plain scan.
    pub(crate) fn kv_abs_max_after(&mut self, chunk: (usize, usize, f32, f32)) -> (f32, f32) {
        let (from, upto, km, vm) = chunk;
        if self.kv_amax_rows != from || self.kv_rows() != upto || from > upto {
            return self.kv_abs_max();
        }
        if from == 0 {
            self.kv_amax = [0.0; 2];
        }
        self.kv_amax = [self.kv_amax[0].max(km), self.kv_amax[1].max(vm)];
        self.kv_amax_rows = upto;
        (self.kv_amax[0], self.kv_amax[1])
    }

    /// Rows any head stores.
    fn kv_rows(&self) -> usize {
        let hd = self.head_dim.max(1);
        self.k
            .iter()
            .chain(&self.v)
            .map(|x| x.len() / hd)
            .max()
            .unwrap_or(0)
    }

    /// Rows from `first` on may no longer be the ones `kv_abs_max` scanned.
    fn kv_rows_changed(&mut self, first: usize) {
        self.kv_amax_rows = self.kv_amax_rows.min(first);
    }

    // ── Natively bounded anchor (swa_sink_v1) ──

    /// Give this layer its fixed-size ring (`[kvh][window][hd]` K and V).
    /// Called once at load from the header; the record is zeroed on
    /// `clear()` and never reallocated.
    pub fn install_bounded(&mut self, window: usize) {
        self.bounded = Some(crate::bounded::BoundedState::new(
            self.num_kv_heads,
            self.head_dim,
            window,
        ));
        self.wire_kind = WireKind::Bounded;
    }

    /// One position of the bounded operator: insert the raw `k, v`
    /// (`[kvh][hd]`) into slot `t mod W`, then attend every Q head of
    /// `q` (`[nh][hd]`, raw) over sinks ∪ window into `out` (`[nh][hd]`).
    /// Nothing is appended per position.
    #[allow(clippy::too_many_arguments)]
    pub fn bounded_step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        w: &crate::bounded::BoundedWeights,
        rope: &crate::bounded::BoundedRope,
        scale: f32,
        num_heads: usize,
        out: &mut [f32],
    ) {
        let st = self
            .bounded
            .as_mut()
            .expect("bounded_step on a layer without an installed ring");
        st.insert(k, v);
        st.attend(q, num_heads, &w.sink_k, &w.sink_v, w.sink, rope, scale, out);
        // Honest context depth for the memory/seq report — nothing is
        // stored per position.
        self.seq_len += 1;
    }

    /// Bytes of the bounded ring (0 on other layers).
    pub fn bounded_state_bytes(&self) -> usize {
        self.bounded.as_ref().map(|b| b.state_bytes()).unwrap_or(0)
    }

    /// Bit-for-bit copy of the ring for speculation (None on other layers).
    pub fn bounded_snapshot(&self) -> Option<crate::bounded::BoundedSnapshot> {
        self.bounded.as_ref().map(|b| b.snapshot())
    }

    /// Restore a ring snapshot taken on this layer; `seq_len` follows the
    /// restored insert counter.
    pub fn bounded_restore(&mut self, s: &crate::bounded::BoundedSnapshot) {
        if let Some(b) = self.bounded.as_mut() {
            b.restore(s);
            self.seq_len = b.seen;
            self.generation = next_gen();
        }
    }

    /// Bind this layer to the legacy cache-wire policy. Delta layers must
    /// refuse untagged state exchange rather than risk a plausible additive
    /// interpretation on the peer.
    pub fn set_linear_wire_allowed(&mut self, allowed: bool) {
        self.linear_wire_allowed = allowed;
    }

    /// Discard tentative recurrent state after a speculative rejection or
    /// any other path that abandons the lane-2 result.
    pub fn discard_linear_scratch(&mut self) {
        self.linear_scratch.clear();
    }

    /// Per-KV-head stored keys `[seq_len × head_dim]` (GPU token graph
    /// sync): the stored tail, row 0 = position `base()`.
    pub fn k_heads(&self) -> &[Vec<f32>] {
        &self.k
    }
    /// Per-KV-head stored values `[seq_len × head_dim]`, row 0 = `base()`.
    pub fn v_heads(&self) -> &[Vec<f32>] {
        &self.v
    }

    // ── Sliding-window tail (trim_window) ──

    /// Absolute position of stored row 0 (0 unless `trim_window` dropped
    /// a prefix).
    pub fn base(&self) -> usize {
        self.base
    }

    /// Absolute context depth: the position the next append will hold.
    pub fn pos_len(&self) -> usize {
        self.base + self.seq_len
    }

    /// Storage generation (see the field): with the row count, the key a
    /// device copy of these rows is valid under.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The window this layer trims to, once `trim_window` ran on it.
    pub fn tail_window(&self) -> Option<usize> {
        self.tail
    }

    /// Drop the rows a sliding-window layer can never read again. A query
    /// at position p reads p−w+1..=p only, so once more than `2w` rows are
    /// stored the front is drained down to `w + slack` rows, the cut
    /// rounded DOWN to an absolute position that is a multiple of `align`
    /// (so `w + slack ..= w + slack + align − 1` rows stay). Returns the
    /// rows dropped.
    ///
    /// The rows that stay keep their order and values (`Vec::drain` of the
    /// front), and every attend indexes rows relative to the newest one
    /// (`first = rows − w`), so each later query sees the very rows it saw
    /// untrimmed — bit for bit. `slack` rows past the window are what a
    /// rollback (`truncate_last`) may take back without losing a row the
    /// window needs. `align` keeps device GEMMs that reduce over the rows
    /// from row 0 in fixed tiles on the tile grid they had untrimmed (the
    /// dropped rows were whole all-zero tiles there).
    ///
    /// Callers trim only at a safe point: never between a call's appends
    /// and its attend or importance accumulation (a prefill chunk appends
    /// all its rows first and indexes them from the count it started at).
    ///
    /// No-op on an O(1) or bounded layer (they own their bound), and on a
    /// q8 layer whose per-channel field is not frozen yet: the freeze
    /// reads every stored row, so dropping one before it would change it.
    pub fn trim_window(&mut self, w: usize, slack: usize, align: usize) -> usize {
        if self.o1.is_some() || self.bounded.is_some() || w == 0 {
            return 0;
        }
        self.tail = Some(w);
        if self.seq_len <= 2 * w {
            return 0;
        }
        if matches!(self.mode, KvMode::Q8 { .. })
            && self.kcol.iter().all(Vec::is_empty)
            && self.vcol.iter().all(Vec::is_empty)
        {
            return 0;
        }
        let align = align.max(1);
        let keep_from = self.pos_len().saturating_sub(w + slack);
        let new_base = keep_from / align * align;
        if new_base <= self.base {
            return 0;
        }
        let d = new_base - self.base;
        let hd = self.head_dim;
        fn drop_front<T>(v: &mut Vec<T>, n: usize) {
            let n = n.min(v.len());
            v.drain(..n);
        }
        for h in 0..self.num_kv_heads {
            drop_front(&mut self.k[h], d * hd);
            drop_front(&mut self.v[h], d * hd);
            drop_front(&mut self.kq[h], d * hd);
            drop_front(&mut self.vq[h], d * hd);
            drop_front(&mut self.ks[h], d * hd.div_ceil(KV_K_GROUP));
            drop_front(&mut self.vs[h], d);
        }
        drop_front(&mut self.imp, d);
        self.base = new_base;
        self.seq_len -= d;
        self.generation = next_gen();
        // `kv_abs_max` counts its scanned mark from the old row 0. Moved
        // with the rows, the next scan still starts at the first row it has
        // not seen; left in place, rows appended after the trim up to the
        // old mark would never be scanned. The maxima may keep a dropped
        // row's value, which only over-states.
        self.kv_amax_rows = self.kv_amax_rows.saturating_sub(d);
        d
    }

    // ── O(1) Nyström override ──

    /// Arm query collection for a fresh prompt pass (a cleared cache).
    pub fn o1_begin(&mut self, m: usize, w: usize, sink: usize, rect: crate::nystrom::O1Rect) {
        self.o1_begin_with_boundary(m, w, sink, rect, None);
    }

    /// Arm query collection with an optional completed-row seal barrier.
    /// The barrier is deliberately part of the existing collecting state:
    /// no second history or scheduler is introduced for short prompts.
    pub(crate) fn o1_begin_with_boundary(
        &mut self,
        m: usize,
        w: usize,
        sink: usize,
        rect: crate::nystrom::O1Rect,
        seal_at: Option<usize>,
    ) {
        self.o1 = Some(O1State::Collecting {
            m,
            w,
            sink,
            rect,
            seal_at,
            q_buf: Vec::new(),
        });
        self.o1_error = None;
        self.o1_transitioned = false;
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

    /// Pending completed-row barrier, if any. A plain full-prompt seal has
    /// no barrier until the caller asks to seal.
    pub(crate) fn o1_pending_boundary(&self) -> Option<usize> {
        match &self.o1 {
            Some(O1State::Collecting { seal_at, .. }) => *seal_at,
            _ => None,
        }
    }

    /// Whether a batch of `count` exact rows would cross the deferred
    /// boundary. This lets batched/pair callers split before row B rather
    /// than appending exact KV past the point where conversion is required.
    pub(crate) fn o1_boundary_crossed_by(&self, count: usize) -> bool {
        let Some(target) = self.o1_pending_boundary() else {
            return false;
        };
        target <= self.seq_len
            || self
                .seq_len
                .checked_add(count)
                .map_or(true, |next| next >= target)
    }

    pub(crate) fn take_o1_transition(&mut self) -> bool {
        std::mem::take(&mut self.o1_transitioned)
    }

    pub(crate) fn take_o1_error(&self) -> Option<String> {
        // Error observation is deliberately non-consuming.  The error is the
        // append guard for this request; removing it would let a caller that
        // ignored the returned Err resume ordinary KV growth after the
        // bounded transition dropped its overlay.  `clear()` is the explicit
        // reset boundary that clears the latch.
        self.o1_error.clone()
    }

    /// Abort a malformed deferred transition after attention has already
    /// produced its current row. Clearing the exact storage and dropping
    /// the overlay makes the state unrecoverable by continued decode; the
    /// pipeline then routes through its normal graph/cancel cleanup.
    pub(crate) fn o1_abort(&mut self, err: String) {
        self.k.iter_mut().for_each(Vec::clear);
        self.v.iter_mut().for_each(Vec::clear);
        self.kv_rows_changed(0);
        self.kq.iter_mut().for_each(Vec::clear);
        self.ks.iter_mut().for_each(Vec::clear);
        self.vq.iter_mut().for_each(Vec::clear);
        self.vs.iter_mut().for_each(Vec::clear);
        self.kcol.iter_mut().for_each(Vec::clear);
        self.vcol.iter_mut().for_each(Vec::clear);
        self.imp.clear();
        self.o1 = None;
        self.seq_len = 0;
        self.base = 0;
        self.generation = next_gen();
        self.o1_transitioned = false;
        self.o1_error = Some(err);
    }

    /// Freeze the prompt into per-KV-group Nyström states and drop this
    /// layer's full KV. Returns false while a short collecting layer is
    /// below its deferred boundary; malformed prerequisites abort the
    /// layer instead of silently resuming exact KV growth. The seal needs
    /// f32 KV rows (`CMF_KV=q8` stores int8), every group densely stored, a
    /// full q trace, and a GQA fan-out that actually divides.
    pub fn o1_seal(&mut self, num_heads: usize) -> bool {
        match self.o1_seal_checked(num_heads) {
            Ok(sealed) => sealed,
            Err(err) => {
                tracing::error!("o1: seal aborted: {err}");
                self.o1_abort(err);
                false
            }
        }
    }

    /// Checked seal implementation. Validation happens while the collecting
    /// state and full KV are still intact; only a valid completed boundary
    /// is allowed to destructively convert them.
    pub(crate) fn o1_seal_checked(&mut self, num_heads: usize) -> Result<bool, String> {
        if let Some(err) = self.o1_error.clone() {
            return Err(err);
        }
        // Idempotent: sealing a sealed (or plain) layer must not disturb its
        // state, and a plain layer is not an O(1) participant.
        if !matches!(self.o1, Some(O1State::Collecting { .. })) {
            return Ok(self.o1_sealed());
        }
        let (m, w, sink, requested_boundary, q_len) = match &self.o1 {
            Some(O1State::Collecting {
                m,
                w,
                sink,
                rect: _,
                seal_at,
                q_buf,
            }) => (*m, *w, *sink, *seal_at, q_buf.len()),
            _ => unreachable!("checked above"),
        };
        let floor = crate::nystrom::o1_deferred_boundary(w, sink)
            .ok_or_else(|| "o1 seal: w + sink + slack + 1 overflow".to_string())?;
        let target = requested_boundary.unwrap_or(floor).max(floor);
        let t = self.seq_len;
        if t < target {
            if let Some(O1State::Collecting { seal_at, .. }) = &mut self.o1 {
                if *seal_at != Some(target) {
                    *seal_at = Some(target);
                    tracing::info!(
                        "o1 deferred seal: current rows={t}, boundary={target} (floor={floor})"
                    );
                }
            }
            return Ok(false);
        }

        let hd = self.head_dim;
        if t == 0 {
            return Err("o1 seal: cannot seal an empty layer".into());
        }
        if self.mode != KvMode::F32 {
            return Err("o1 seal: requires dense F32 KV storage".into());
        }
        if self.num_kv_heads == 0 || num_heads == 0 || num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "o1 seal: invalid GQA geometry num_heads={num_heads} num_kv_heads={}",
                self.num_kv_heads
            ));
        }
        let hpk = num_heads / self.num_kv_heads;
        let expected_k = t
            .checked_mul(hd)
            .ok_or_else(|| "o1 seal: KV row length overflow".to_string())?;
        let expected_q = expected_k
            .checked_mul(num_heads)
            .ok_or_else(|| "o1 seal: query trace length overflow".to_string())?;
        if q_len != expected_q {
            return Err(format!(
                "o1 seal: query trace has {q_len} values, expected {expected_q}"
            ));
        }
        if (0..self.num_kv_heads)
            .any(|g| self.k[g].len() != expected_k || self.v[g].len() != expected_k)
        {
            return Err("o1 seal: KV heads are not densely populated".into());
        }
        if m < 4 || w == 0 {
            return Err(format!("o1 seal: invalid geometry m={m} w={w}"));
        }

        let Some(O1State::Collecting {
            m,
            w,
            sink,
            rect,
            q_buf,
            ..
        }) = self.o1.take()
        else {
            unreachable!("collecting state disappeared after validation");
        };
        let mut groups = Vec::with_capacity(self.num_kv_heads);
        // Query trace is position-major; the state wants each head's
        // queries contiguous, so transpose one group at a time.
        let mut qh = vec![0.0f32; hpk * t * hd];
        for g in 0..self.num_kv_heads {
            for hh in 0..hpk {
                let h = g * hpk + hh;
                for p in 0..t {
                    let src = (p * num_heads + h) * hd;
                    let dst = (hh * t + p) * hd;
                    qh[dst..dst + hd].copy_from_slice(&q_buf[src..src + hd]);
                }
            }
            let qs: Vec<&[f32]> = (0..hpk)
                .map(|hh| &qh[hh * t * hd..(hh + 1) * t * hd])
                .collect();
            let mut st = crate::nystrom::NystromState::new_group(m, w, sink, hpk).with_rect(rect);
            st.prefill_group(&qs, &self.k[g], &self.v[g], t, hd, hd);
            groups.push(st);
        }
        // The states now carry everything decode needs — release the
        // O(context) storage (this is the memory claim, not a cosmetic).
        for h in 0..self.num_kv_heads {
            self.k[h] = Vec::new();
            self.v[h] = Vec::new();
        }
        self.kv_rows_changed(0);
        self.imp = Vec::new();
        self.o1 = Some(O1State::Sealed { groups });
        self.o1_transitioned = true;
        Ok(true)
    }

    /// One decode step on a sealed layer: per KV group, insert the
    /// group's fresh (k, v) ONCE and read every Q head's attention
    /// output. Returns `[num_heads × head_dim]`. Head h belongs to group
    /// h/hpk, so a group's Q heads are contiguous in `q_all`/`out` —
    /// same math as the shared KV row the exact path appends once.
    /// Device views of the sealed o1 groups, or None when o1 is not
    /// sealed on this layer (or any group is in the degenerate
    /// exact-only mode the GPU path does not carry).
    pub fn o1_views(&self) -> Option<Vec<crate::nystrom::O1DeviceView<'_>>> {
        let Some(O1State::Sealed { groups }) = &self.o1 else {
            return None;
        };
        let views: Vec<_> = groups.iter().map(|g| g.device_view()).collect();
        if views.iter().any(|v| v.exact_only) {
            return None;
        }
        Some(views)
    }

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
        let Some(O1State::Sealed { groups }) = &mut self.o1 else {
            debug_assert!(false, "o1_step on an unsealed layer");
            return out;
        };
        for (g, st) in groups.iter_mut().enumerate() {
            let (lo, hi) = (g * hpk * hd, (g + 1) * hpk * hd);
            st.step_group(
                &q_all[lo..hi],
                &k_new[g * hd..(g + 1) * hd],
                &v_new[g * hd..(g + 1) * hd],
                &mut out[lo..hi],
            );
        }
        // Track the true context depth for the honest memory/seq report
        // (nothing is stored per position — the state is O(1)).
        self.seq_len += 1;
        out
    }

    /// Bytes held by the O(1) override (query trace while collecting,
    /// per-KV-group states once sealed).
    pub fn o1_memory_bytes(&self) -> usize {
        match &self.o1 {
            Some(O1State::Collecting { q_buf, .. }) => q_buf.len() * std::mem::size_of::<f32>(),
            Some(O1State::Sealed { groups }) => groups.iter().map(|s| s.memory_bytes()).sum(),
            None => 0,
        }
    }

    /// Quantize one row against the per-channel field (empty col = 1);
    /// `group` — elements per scale (the whole row or KV_K_GROUP).
    fn quant_row(row: &[f32], col: &[f32], q: &mut Vec<i8>, sc: &mut Vec<f32>, group: usize) {
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
                (
                    &mut self.kq[h],
                    &mut self.ks[h],
                    &mut self.kcol[h],
                    KV_K_GROUP,
                ),
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
                        rows[p * hd + d] = qv[p * hd + d] as f32 * sv[p * spp + d / group];
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
        // A failed bounded transition is terminal until the sequence is
        // cleared. Do not let an ignored boolean/result resume plain KV
        // growth after the O(1) collector has aborted.
        if self.o1_error.is_some() {
            return;
        }
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
                Self::quant_row(
                    &k_new[s..s + self.head_dim],
                    &self.kcol[h],
                    &mut self.kq[h],
                    &mut self.ks[h],
                    KV_K_GROUP,
                );
            } else {
                self.k[h].extend_from_slice(&k_new[s..s + self.head_dim]);
            }
            if self.mode.quant_v() {
                Self::quant_row(
                    &v_new[s..s + self.head_dim],
                    &self.vcol[h],
                    &mut self.vq[h],
                    &mut self.vs[h],
                    self.head_dim,
                );
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
        // Full-context layers only: a trimmed tail is not the whole row.
        debug_assert!(self.tail.is_none(), "attend() on a sliding-window tail");
        let hd = self.head_dim;
        if self.mode == KvMode::F32 {
            let stored = self.k[kv_head].len() / hd;
            return crate::attention::attention_head(
                q,
                &self.k[kv_head],
                &self.v[kv_head],
                hd,
                stored,
            );
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
                qc[d] = if kcol.is_empty() {
                    q[d]
                } else {
                    q[d] * kcol[d]
                };
            }
            let ng = hd.div_ceil(KV_K_GROUP);
            for p in 0..stored {
                let row = &kq[p * hd..(p + 1) * hd];
                // SAFETY: i8 and u8 share layout; dot_i8_f32 reads the
                // bytes back as i8.
                let row_u8 =
                    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len()) };
                let mut dot = 0.0f32;
                for g in 0..ng {
                    let g0 = g * KV_K_GROUP;
                    let g1 = (g0 + KV_K_GROUP).min(hd);
                    dot +=
                        crate::qtensor::dot_i8_f32(&row_u8[g0..g1], &qc[g0..g1]) * ks[p * ng + g];
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

    /// Grouped GQA attention: all Q-heads of one KV group in a single
    /// pass over the stored K rows and a single pass over the V rows
    /// (per-head `attend` re-read the shared group storage
    /// heads_per_kv times — roadmap §3 P1). Per-head score order,
    /// softmax and V accumulation are IDENTICAL to `attend`, so each
    /// head's output is bit-for-bit the same.
    ///
    /// `q_group`: `[n_heads_in_group × head_dim]` (global head order);
    /// `out`: same shape; `imp_acc[0..stored]` accumulates the probabilities
    /// of every head (attention importance), matching the caller's former loop.
    /// `scale` is the score scale (1/√hd unless the arch overrides);
    /// `first` is the earliest visible position — sliding-window layers
    /// pass `stored − window` so older rows get zero probability.
    /// `sinks` holds one learned sink logit per head of `q_group` (the
    /// caller slices `self.sinks` to the group's heads) or is empty for
    /// an ordinary softmax.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_group(
        &self,
        q_group: &[f32],
        kv_head: usize,
        out: &mut [f32],
        imp_acc: &mut [f32],
        scale: f32,
        first: usize,
        softcap: f32,
        sinks: &[f32],
    ) {
        self.attend_group_upto(
            q_group,
            kv_head,
            out,
            imp_acc,
            scale,
            first,
            softcap,
            usize::MAX,
            sinks,
        )
    }

    /// `attend_group` over the first `upto` stored rows only — what the
    /// same call saw when the cache held exactly `upto` rows. A prefill
    /// chunk appends all its rows first and then attends every position
    /// in parallel; position `i` passes `upto = s0 + i + 1`, which makes
    /// its result bit-identical to the sequential append-then-attend.
    ///
    /// Only the visible rows `[first, stored)` are scored, so a
    /// sliding-window decode step costs O(window), not O(context). The
    /// rows before `first` get probability exactly 0 — what the former
    /// −inf-filled score row produced (exp(−inf) = 0 adds nothing to the
    /// max, the sum, V or the importance), so the result is bit-identical
    /// to scoring the whole row.
    ///
    /// Learned sinks (`sinks` non-empty, one per head): the sink logit
    /// joins the softmax max and adds exp(sink − max) to the denominator;
    /// it has no value row. Identical to appending a value-less column,
    /// which is how gpt-oss and MiMo-V2 define it.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_group_upto(
        &self,
        q_group: &[f32],
        kv_head: usize,
        out: &mut [f32],
        imp_acc: &mut [f32],
        scale: f32,
        first: usize,
        softcap: f32,
        upto: usize,
        sinks: &[f32],
    ) {
        let hd = self.head_dim;
        let nheads = q_group.len() / hd;
        debug_assert_eq!(out.len(), nheads * hd);
        assert!(
            sinks.is_empty() || sinks.len() == nheads,
            "attend_group: {} sink logits for {nheads} heads",
            sinks.len()
        );
        let stored = if self.mode == KvMode::F32 {
            self.k[kv_head].len() / hd
        } else {
            self.head_len(kv_head)
        }
        .min(upto);
        if stored == 0 {
            out.fill(0.0);
            return;
        }
        let first = first.min(stored.saturating_sub(1));
        // Visible span: row p ∈ [first, stored) lives at column p − first.
        let span = stored - first;

        thread_local! {
            /// scores [nheads × span] — reused across layers/tokens.
            static GQA_SCORES: std::cell::RefCell<Vec<f32>> =
                const { std::cell::RefCell::new(Vec::new()) };
            /// q ⊙ col_k per head (q8 K mode).
            static GQA_QC: std::cell::RefCell<Vec<f32>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }

        GQA_SCORES.with(|sc| {
            let mut scores = sc.borrow_mut();
            scores.resize(nheads * span, 0.0);

            // ── score pass: each stored K row is read ONCE for all heads.
            if self.mode.quant_k() {
                let (kq, ks) = (&self.kq[kv_head], &self.ks[kv_head]);
                let kcol = &self.kcol[kv_head];
                let ng = hd.div_ceil(KV_K_GROUP);
                GQA_QC.with(|qc| {
                    let mut qcb = qc.borrow_mut();
                    qcb.resize(nheads * hd, 0.0);
                    for h in 0..nheads {
                        for d in 0..hd {
                            let qv = q_group[h * hd + d];
                            qcb[h * hd + d] = if kcol.is_empty() { qv } else { qv * kcol[d] };
                        }
                    }
                    for p in first..stored {
                        let row = &kq[p * hd..(p + 1) * hd];
                        // SAFETY: i8 and u8 share layout; dot_i8_f32 reads
                        // the bytes back as i8.
                        let row_u8 = unsafe {
                            std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len())
                        };
                        for h in 0..nheads {
                            let qch = &qcb[h * hd..(h + 1) * hd];
                            let mut dot = 0.0f32;
                            for g in 0..ng {
                                let g0 = g * KV_K_GROUP;
                                let g1 = (g0 + KV_K_GROUP).min(hd);
                                dot += crate::qtensor::dot_i8_f32(&row_u8[g0..g1], &qch[g0..g1])
                                    * ks[p * ng + g];
                            }
                            scores[h * span + (p - first)] = dot * scale;
                        }
                    }
                });
            } else {
                let k = &self.k[kv_head];
                for p in first..stored {
                    let row = &k[p * hd..(p + 1) * hd];
                    for h in 0..nheads {
                        scores[h * span + (p - first)] =
                            crate::attention::dot_f32(&q_group[h * hd..(h + 1) * hd], row) * scale;
                    }
                }
            }

            // Gemma-2 attention-logit soft-capping: tanh-squash the
            // COMPUTED scores before the softmax (every scored row is in
            // the window; a learned sink is not a score and is not capped).
            if softcap > 0.0 {
                for v in scores.iter_mut() {
                    *v = softcap * (*v / softcap).tanh();
                }
            }

            // ── per-head softmax (identical to attend / attention_head;
            // a sink joins the max and the denominator, never the rows).
            for h in 0..nheads {
                let s = &mut scores[h * span..(h + 1) * span];
                let row_max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sink = sinks.get(h).copied();
                let max_score = match sink {
                    Some(z) => row_max.max(z),
                    None => row_max,
                };
                let mut sum = 0.0f32;
                for v in s.iter_mut() {
                    *v = (*v - max_score).exp();
                    sum += *v;
                }
                if let Some(z) = sink {
                    sum += (z - max_score).exp();
                }
                if sum > 0.0 {
                    for v in s.iter_mut() {
                        *v /= sum;
                    }
                }
            }

            // ── value pass: each stored V row is read ONCE for all heads.
            out.fill(0.0);
            if self.mode.quant_v() {
                let (vq, vs) = (&self.vq[kv_head], &self.vs[kv_head]);
                for p in first..stored {
                    let row = &vq[p * hd..(p + 1) * hd];
                    for h in 0..nheads {
                        let w = scores[h * span + (p - first)] * vs[p];
                        if w.abs() < 1e-12 {
                            continue;
                        }
                        crate::qtensor::axpy_i8_f32(&mut out[h * hd..(h + 1) * hd], row, w);
                    }
                }
                let vcol = &self.vcol[kv_head];
                if !vcol.is_empty() {
                    for h in 0..nheads {
                        for d in 0..hd {
                            out[h * hd + d] *= vcol[d];
                        }
                    }
                }
            } else {
                let v = &self.v[kv_head];
                for p in first..stored {
                    let row = &v[p * hd..(p + 1) * hd];
                    for h in 0..nheads {
                        let w = scores[h * span + (p - first)];
                        if w.abs() < 1e-12 {
                            continue;
                        }
                        crate::attention::axpy_f32(&mut out[h * hd..(h + 1) * hd], row, w);
                    }
                }
            }

            // ── Attention-importance accumulation (Σ probs over heads), same
            // head order as the caller's former per-head loop. Rows before
            // `first` carry probability 0 and are left untouched.
            let n = imp_acc.len().min(stored);
            if n > first {
                for h in 0..nheads {
                    let s = &scores[h * span..(h + 1) * span];
                    for (dst, &p) in imp_acc[first..n].iter_mut().zip(s) {
                        *dst += p;
                    }
                }
            }
        });
    }

    /// Batched causal attend for a prefill chunk (macOS/AArch64): the
    /// cache already holds every chunk row (`s0` old + `b` new). Per
    /// Q-head the scores GEMM `Q·Kᵀ` rides the AMX, the causal softmax
    /// zeroes the not-yet-visible tail so the `P·V` GEMM needs no
    /// mask, and attention importance takes the masked column sums. Same
    /// math as the per-position attend; summation order differs
    /// (tolerance-class, like the projection GEMMs).
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    pub fn attend_chunk(
        &mut self,
        q_all: &[f32],
        b: usize,
        s0: usize,
        nh: usize,
        heads_per_kv: usize,
        hd: usize,
        out: &mut [f32],
        pool: Option<&crate::pool::Pool>,
        scale: f32,
        window: Option<usize>,
    ) {
        let n = s0 + b;
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        impl SendPtr {
            fn at(&self, i: usize) -> *mut f32 {
                // Method receiver keeps the closure capturing &SendPtr
                // (2021 disjoint capture would grab the raw field).
                unsafe { self.0.add(i) }
            }
        }
        thread_local! {
            static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
        }
        // The portable NEON GEMM pays dearly for the gathered Bᵀ loads
        // of the scores multiply — pack Kᵀ once per (group, chunk) and
        // hand it the sequential-B fast path instead. Accelerate keeps
        // the no-copy transposed call.
        let neon_gemm = cfg!(not(target_os = "macos"))
            || std::env::var("CMF_FORCE_NEON_GEMM")
                .map(|v| v == "1")
                .unwrap_or(false);
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            let (qpanel, scores, aopanel, ktpack) = &mut *s;
            // The whole KV-group attends in one GEMM pair: the group's
            // Q-heads stack head-major into one tall panel [hpk·b, hd]
            // (row hl·b + bi), so each layer costs 2 sgemm calls per
            // group instead of 2 per head — fat M keeps the AMX fed.
            let m = heads_per_kv * b;
            qpanel.resize(m * hd, 0.0);
            scores.resize(m * n, 0.0);
            aopanel.resize(m * hd, 0.0);
            for g in 0..self.num_kv_heads {
                let kmat = &self.k[g];
                let vmat = &self.v[g];
                debug_assert_eq!(kmat.len(), n * hd);
                for hl in 0..heads_per_kv {
                    let hh = g * heads_per_kv + hl;
                    for bi in 0..b {
                        qpanel[(hl * b + bi) * hd..(hl * b + bi + 1) * hd]
                            .copy_from_slice(&q_all[bi * nh * hd + hh * hd..][..hd]);
                    }
                }
                if neon_gemm {
                    ktpack.resize(hd * n, 0.0);
                    for p in 0..n {
                        let row = &kmat[p * hd..(p + 1) * hd];
                        for (d, &v) in row.iter().enumerate() {
                            ktpack[d * n + p] = v;
                        }
                    }
                    // Accelerate threads its own GEMM; the NEON kernel
                    // splits the m rows across the pool instead.
                    let sp_q = SendPtr(qpanel.as_ptr() as *mut f32);
                    let sp_s = SendPtr(scores.as_mut_ptr());
                    let kt = &*ktpack;
                    let run = |start: usize, end: usize| {
                        if end > start {
                            // SAFETY: workers write disjoint score rows.
                            let a = unsafe {
                                std::slice::from_raw_parts(sp_q.at(start * hd), (end - start) * hd)
                            };
                            let c = unsafe {
                                std::slice::from_raw_parts_mut(
                                    sp_s.at(start * n),
                                    (end - start) * n,
                                )
                            };
                            crate::qtensor::neon_gemm_rm(
                                end - start,
                                n,
                                hd,
                                scale,
                                a,
                                hd,
                                kt,
                                n,
                                false,
                                c,
                                n,
                            );
                        }
                    };
                    match pool {
                        Some(p) if m >= 64 => p.run_rows(m, &run),
                        _ => run(0, m),
                    }
                } else {
                    crate::qtensor::sgemm_rm(
                        m, n, hd, scale, qpanel, hd, kmat, hd, true, scores, n,
                    );
                }
                // Causal softmax, row-parallel (rows are disjoint).
                let sp = SendPtr(scores.as_mut_ptr());
                let run = |start: usize, end: usize| {
                    for r in start..end {
                        let allowed = s0 + (r % b) + 1;
                        // Sliding-window layers see only the last W of
                        // the causal range; the zeroed head contributes
                        // nothing to P·V or attention importance.
                        let lo = window.map(|w| allowed.saturating_sub(w)).unwrap_or(0);
                        // SAFETY: workers cover disjoint row ranges.
                        let row = unsafe { std::slice::from_raw_parts_mut(sp.at(r * n), n) };
                        crate::attention::softmax_row(&mut row[lo..allowed]);
                        row[..lo].fill(0.0);
                        row[allowed..].fill(0.0);
                    }
                };
                match pool {
                    Some(p) if m >= 64 => p.run_rows(m, &run),
                    _ => run(0, m),
                }
                // Attention importance: masked column sums (probs of the
                // zeroed tail contribute nothing, same as the CPU
                // per-position accumulate).
                let ni = self.imp.len().min(n);
                for r in 0..m {
                    let al = (s0 + (r % b) + 1).min(ni);
                    for (dst, &p) in self.imp[..al].iter_mut().zip(&scores[r * n..r * n + al]) {
                        *dst += p;
                    }
                }
                if neon_gemm {
                    let sp_s = SendPtr(scores.as_mut_ptr());
                    let sp_o = SendPtr(aopanel.as_mut_ptr());
                    let run = |start: usize, end: usize| {
                        if end > start {
                            // SAFETY: workers write disjoint output rows.
                            let a = unsafe {
                                std::slice::from_raw_parts(sp_s.at(start * n), (end - start) * n)
                            };
                            let c = unsafe {
                                std::slice::from_raw_parts_mut(
                                    sp_o.at(start * hd),
                                    (end - start) * hd,
                                )
                            };
                            crate::qtensor::neon_gemm_rm(
                                end - start,
                                hd,
                                n,
                                1.0,
                                a,
                                n,
                                vmat,
                                hd,
                                false,
                                c,
                                hd,
                            );
                        }
                    };
                    match pool {
                        Some(p) if m >= 64 => p.run_rows(m, &run),
                        _ => run(0, m),
                    }
                } else {
                    crate::qtensor::sgemm_rm(
                        m, hd, n, 1.0, scores, n, vmat, hd, false, aopanel, hd,
                    );
                }
                for hl in 0..heads_per_kv {
                    let hh = g * heads_per_kv + hl;
                    for bi in 0..b {
                        out[bi * nh * hd + hh * hd..][..hd]
                            .copy_from_slice(&aopanel[(hl * b + bi) * hd..(hl * b + bi + 1) * hd]);
                    }
                }
            }
        });
    }

    /// Roll back the last `n_drop` positions (speculative-decode reject).
    pub fn truncate_last(&mut self, n_drop: usize) {
        self.discard_linear_scratch();
        let d = n_drop.min(self.seq_len);
        if let Some(b) = self.bounded.as_mut() {
            // The ring rolls back through its undo rows; nothing is
            // stored per position, so there is nothing else to drop.
            let rolled = b.rollback(d);
            if rolled < d {
                tracing::warn!(
                    "bounded anchor: rollback of {d} exceeds the undo depth ({rolled} restored)"
                );
            }
            self.seq_len = b.seen;
            return;
        }
        // A trimmed tail keeps `slack` rows past its window for exactly
        // this; a deeper rollback leaves the next query short of rows its
        // window reads (no Spark path rolls back more than 2).
        if let Some(w) = self.tail
            && self.base > 0
            && self.seq_len - d < w.saturating_sub(1)
        {
            tracing::warn!(
                "sliding tail: rollback of {d} leaves {} rows under the {w}-row window",
                self.seq_len - d
            );
        }
        let mut first_changed = usize::MAX;
        for h in 0..self.num_kv_heads {
            let keep = self.k[h].len().saturating_sub(d * self.head_dim);
            if !self.k[h].is_empty() || !self.v[h].is_empty() {
                first_changed = first_changed.min(keep / self.head_dim.max(1));
            }
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
        self.kv_rows_changed(first_changed);
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
        self.kv_rows_changed(0);
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
        self.discard_linear_scratch();
        // Fresh conversation → the pipeline re-arms collection if the
        // layer is o1-flagged (landmarks are per-prompt, never reused).
        self.o1 = None;
        self.o1_error = None;
        self.o1_transitioned = false;
        // The bounded ring is zeroed in place: its size is a property of
        // the file, not of the conversation.
        if let Some(b) = self.bounded.as_mut() {
            b.clear();
        }
        self.seq_len = 0;
        // `tail` stays: the window is the layer's, not the conversation's.
        self.base = 0;
        self.generation = next_gen();
    }

    /// Serialize this layer's state for the wire (versioned, v2): a
    /// fixed header `{magic "CMFS", version, operator identity hash64,
    /// layer, kind, f16 flag, position}` followed by one record whose
    /// shape the kind fixes — per-position K/V (+ importance) for a full
    /// layer, the recurrent vector for a linear layer, the insert counter
    /// + ring K/V for a bounded anchor. `f16` halves the K/V payloads and
    /// is the caller's explicit choice, exactly like the hidden-state
    /// wire; recurrent vectors stay f32 whatever the wire dtype (they are
    /// the ONLY state a linear layer has — rounding them rounds the whole
    /// history).
    ///
    /// REFUSES rather than travelling half-complete. A cache carrying
    /// frozen columns, a Nyström overlay or q8 storage holds state this
    /// format does not describe, and shipping the rest would land a
    /// plausible-looking cache that answers differently — the failure
    /// mode this whole format exists to avoid.
    pub fn export_wire(&self, f16: bool) -> Result<Vec<u8>, String> {
        if !matches!(self.mode, KvMode::F32) {
            return Err("kv export: only the F32 cache is described by this                         format (CMF_KV=q8 stores int8 rows and per-row scales)"
                .into());
        }
        if self.o1.is_some() {
            return Err("kv export: an O(1) Nyström overlay is not part of                         this format — the skeletons are irreversible and                         would have to travel with it"
                .into());
        }
        // Frozen columns only exist under q8 storage, which is refused
        // above. If one shows up under an F32 cache the format is lying
        // about something and the transfer must not proceed.
        if self.kcol.iter().any(|c| !c.is_empty()) || self.vcol.iter().any(|c| !c.is_empty()) {
            return Err(
                "kv export: frozen columns under an F32 cache — refusing to ship \
                        a state this format does not describe"
                    .into(),
            );
        }
        let mut out = Vec::with_capacity(self.memory_bytes() / if f16 { 2 } else { 1 } + 64);
        let u = |v: u32, o: &mut Vec<u8>| o.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(WIRE_MAGIC);
        u(WIRE_VERSION, &mut out);
        out.extend_from_slice(&self.wire_identity.to_le_bytes());
        u(self.wire_layer, &mut out);
        // A trimmed full layer travels as its tail and says where it starts.
        let kind = match self.wire_kind {
            WireKind::Full if self.base > 0 => WireKind::FullTail,
            k => k,
        };
        out.push(kind as u8);
        out.push(u8::from(f16));
        out.extend_from_slice(&0u16.to_le_bytes());
        // The header carries the absolute depth (== seq_len untrimmed).
        out.extend_from_slice(&(self.pos_len() as u64).to_le_bytes());
        let push = |xs: &[f32], o: &mut Vec<u8>| {
            if f16 {
                for &x in xs {
                    o.extend_from_slice(&cortiq_core::quant::f32_to_f16(x).to_le_bytes());
                }
            } else {
                for &x in xs {
                    o.extend_from_slice(&x.to_le_bytes());
                }
            }
        };
        match kind {
            WireKind::Full => self.export_full_body(f16, &mut out),
            WireKind::FullTail => {
                out.extend_from_slice(&(self.base as u64).to_le_bytes());
                self.export_full_body(f16, &mut out);
            }
            WireKind::Linear => {
                u(self.num_kv_heads as u32, &mut out);
                u(self.head_dim as u32, &mut out);
                u(self.linear_state.len() as u32, &mut out);
                for &x in &self.linear_state {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            WireKind::Bounded => {
                let b = self
                    .bounded
                    .as_ref()
                    .ok_or("kv export: bounded wire kind without an installed ring")?;
                u(self.num_kv_heads as u32, &mut out);
                u(self.head_dim as u32, &mut out);
                u(b.window as u32, &mut out);
                u(b.len() as u32, &mut out);
                u(b.head() as u32, &mut out);
                push(&b.ring_k, &mut out);
                push(&b.ring_v, &mut out);
            }
        }
        Ok(out)
    }

    /// The per-position record (the whole legacy wire): f16 flag,
    /// seq_len, geometry, recurrent vector, importance, per-head K/V.
    fn export_full_body(&self, f16: bool, out: &mut Vec<u8>) {
        let u = |v: u32, o: &mut Vec<u8>| o.extend_from_slice(&v.to_le_bytes());
        u(u8::from(f16) as u32, out);
        u(self.seq_len as u32, out);
        u(self.num_kv_heads as u32, out);
        u(self.head_dim as u32, out);
        u(self.linear_state.len() as u32, out);
        // Attention importance is ordinary state: every attention call
        // accumulates it and eviction reads it. Leaving it behind would
        // hand the far side a cache that forgets the RIGHT positions
        // later — a divergence that shows up only under pressure.
        u(self.imp.len() as u32, out);
        let push = |xs: &[f32], o: &mut Vec<u8>| {
            if f16 {
                for &x in xs {
                    o.extend_from_slice(&cortiq_core::quant::f32_to_f16(x).to_le_bytes());
                }
            } else {
                for &x in xs {
                    o.extend_from_slice(&x.to_le_bytes());
                }
            }
        };
        for &x in &self.linear_state {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &self.imp {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for h in 0..self.num_kv_heads {
            u(self.k[h].len() as u32, out);
            push(&self.k[h], out);
            u(self.v[h].len() as u32, out);
            push(&self.v[h], out);
        }
    }

    /// Install a peer's state over this layer. The geometry must match the
    /// model both sides hold — it is checked, not assumed. Accepts the
    /// versioned wire (magic "CMFS") and, for full/linear layers, the old
    /// unversioned per-position body.
    pub fn import_wire(&mut self, buf: &[u8]) -> Result<(), String> {
        if buf.len() >= 4 && &buf[..4] == WIRE_MAGIC {
            return self.import_wire_v2(buf);
        }
        // Legacy (unversioned) wire: no operator tag travels with it.
        if !self.linear_wire_allowed {
            return Err(
                "kv import: Delta linear state cannot use the unversioned cache wire; refusing until the wire carries operator identity".into(),
            );
        }
        if self.bounded.is_some() {
            return Err(
                "kv import: a bounded anchor takes only the versioned wire (v2) — the \
                 unversioned body has no ring record"
                    .into(),
            );
        }
        let n = self.import_full_body(buf)?;
        if n != buf.len() {
            return Err(format!(
                "kv import: {} trailing byte(s) after the record",
                buf.len() - n
            ));
        }
        Ok(())
    }

    fn import_wire_v2(&mut self, buf: &[u8]) -> Result<(), String> {
        let need = |n: usize, o: usize| -> Result<(), String> {
            if o + n > buf.len() {
                Err("kv import: truncated header".into())
            } else {
                Ok(())
            }
        };
        need(28, 0)?;
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != WIRE_VERSION {
            return Err(format!(
                "kv import: wire version {version}, this runtime speaks {WIRE_VERSION}"
            ));
        }
        let identity = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let layer = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let kind = WireKind::from_u8(buf[20])
            .ok_or_else(|| format!("kv import: unknown state kind {}", buf[20]))?;
        let f16 = buf[21] != 0;
        let position = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
        if identity != self.wire_identity {
            return Err(format!(
                "kv import: peer operator identity {identity:016x} != mine {:016x} — \
                 the two sides do not hold the same operator",
                self.wire_identity
            ));
        }
        if layer != self.wire_layer {
            return Err(format!(
                "kv import: record is for layer {layer}, this is layer {}",
                self.wire_layer
            ));
        }
        // A full layer takes its trimmed tail too.
        let fits = kind == self.wire_kind
            || (kind == WireKind::FullTail && self.wire_kind == WireKind::Full);
        if !fits {
            return Err(format!(
                "kv import: record kind {kind:?} does not match this layer's {:?}",
                self.wire_kind
            ));
        }
        let mut o = 32usize;
        let u32_at = |o: &mut usize| -> Result<u32, String> {
            if *o + 4 > buf.len() {
                return Err("kv import: truncated record".into());
            }
            let v = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap());
            *o += 4;
            Ok(v)
        };
        let need_payload = |n: usize, o: usize| -> Result<(), String> {
            if o + n > buf.len() {
                Err("kv import: truncated payload".into())
            } else {
                Ok(())
            }
        };
        match kind {
            WireKind::Full => {
                let n = self.import_full_body(&buf[o..])?;
                o += n;
                if self.seq_len != position {
                    return Err(format!(
                        "kv import: header position {position} != record seq_len {}",
                        self.seq_len
                    ));
                }
            }
            WireKind::FullTail => {
                need_payload(8, o)?;
                let base = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()) as usize;
                o += 8;
                let n = self.import_full_body(&buf[o..])?;
                o += n;
                if base.checked_add(self.seq_len) != Some(position) {
                    let rows = self.seq_len;
                    self.reset_per_position_storage();
                    self.seq_len = 0;
                    return Err(format!(
                        "kv import: tail base {base} + {rows} rows != header position {position}"
                    ));
                }
                self.base = base;
            }
            WireKind::Linear => {
                let heads = u32_at(&mut o)? as usize;
                let hd = u32_at(&mut o)? as usize;
                if heads != self.num_kv_heads || hd != self.head_dim {
                    return Err(format!(
                        "kv import: peer sent {heads}×{hd} per position, this layer is {}×{}",
                        self.num_kv_heads, self.head_dim
                    ));
                }
                let lin = u32_at(&mut o)? as usize;
                need_payload(lin * 4, o)?;
                self.linear_state = (0..lin)
                    .map(|i| f32::from_le_bytes(buf[o + i * 4..o + i * 4 + 4].try_into().unwrap()))
                    .collect();
                o += lin * 4;
                self.reset_per_position_storage();
                self.seq_len = position;
            }
            WireKind::Bounded => {
                let heads = u32_at(&mut o)? as usize;
                let hd = u32_at(&mut o)? as usize;
                let window = u32_at(&mut o)? as usize;
                let len = u32_at(&mut o)? as usize;
                let head = u32_at(&mut o)? as usize;
                let b = self
                    .bounded
                    .as_mut()
                    .ok_or("kv import: bounded record for a layer without a ring")?;
                if heads != b.num_kv_heads || hd != b.head_dim || window != b.window {
                    return Err(format!(
                        "kv import: bounded record {heads}×{window}×{hd} does not fit this \
                         layer's ring {}×{}×{}",
                        b.num_kv_heads, b.window, b.head_dim
                    ));
                }
                if len != position.min(window) || head != position % window {
                    return Err(format!(
                        "kv import: bounded record len/head {len}/{head} inconsistent with \
                         position {position} (window {window})"
                    ));
                }
                let n = b.ring_k.len();
                let w = if f16 { 2 } else { 4 };
                need_payload(2 * n * w, o)?;
                let read = |o: usize, dst: &mut [f32]| {
                    for (i, d) in dst.iter_mut().enumerate() {
                        let at = o + i * w;
                        *d = if f16 {
                            cortiq_core::quant::f16_to_f32(u16::from_le_bytes(
                                buf[at..at + 2].try_into().unwrap(),
                            ))
                        } else {
                            f32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
                        };
                    }
                };
                read(o, &mut b.ring_k);
                o += n * w;
                read(o, &mut b.ring_v);
                o += n * w;
                b.seen = position;
                // The undo rows describe inserts this side never made.
                let snap = b.snapshot();
                b.restore(&snap);
                self.linear_state = Vec::new();
                self.reset_per_position_storage();
                self.seq_len = position;
            }
        }
        if o != buf.len() {
            return Err(format!(
                "kv import: {} trailing byte(s) after the record",
                buf.len() - o
            ));
        }
        Ok(())
    }

    /// Empty every per-position store (keeps the recurrent vector and
    /// the bounded ring untouched).
    fn reset_per_position_storage(&mut self) {
        let heads = self.num_kv_heads;
        self.mode = KvMode::F32;
        self.k = vec![Vec::new(); heads];
        self.v = vec![Vec::new(); heads];
        self.kv_rows_changed(0);
        self.kq = vec![Vec::new(); heads];
        self.ks = vec![Vec::new(); heads];
        self.vq = vec![Vec::new(); heads];
        self.vs = vec![Vec::new(); heads];
        self.kcol = vec![Vec::new(); heads];
        self.vcol = vec![Vec::new(); heads];
        self.imp = Vec::new();
        self.discard_linear_scratch();
        self.o1 = None;
        self.o1_error = None;
        self.o1_transitioned = false;
        self.base = 0;
        self.generation = next_gen();
    }

    /// Parse the per-position record (legacy wire body); returns the
    /// bytes consumed.
    fn import_full_body(&mut self, buf: &[u8]) -> Result<usize, String> {
        let mut o = 0usize;
        let u32_at = |o: &mut usize| -> Result<u32, String> {
            if *o + 4 > buf.len() {
                return Err("kv import: truncated header".into());
            }
            let v = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap());
            *o += 4;
            Ok(v)
        };
        let f16 = u32_at(&mut o)? != 0;
        let seq_len = u32_at(&mut o)? as usize;
        let heads = u32_at(&mut o)? as usize;
        let hd = u32_at(&mut o)? as usize;
        let lin = u32_at(&mut o)? as usize;
        let nimp = u32_at(&mut o)? as usize;
        if heads != self.num_kv_heads || hd != self.head_dim {
            return Err(format!(
                "kv import: peer sent {heads}×{hd} per position, this layer is {}×{}",
                self.num_kv_heads, self.head_dim
            ));
        }
        let w = if f16 { 2 } else { 4 };
        let need = |n: usize, o: usize| -> Result<(), String> {
            if o + n > buf.len() {
                Err("kv import: truncated payload".into())
            } else {
                Ok(())
            }
        };
        need(lin * 4, o)?;
        self.linear_state = (0..lin)
            .map(|i| f32::from_le_bytes(buf[o + i * 4..o + i * 4 + 4].try_into().unwrap()))
            .collect();
        o += lin * 4;
        need(nimp * 4, o)?;
        let imp: Vec<f32> = (0..nimp)
            .map(|i| f32::from_le_bytes(buf[o + i * 4..o + i * 4 + 4].try_into().unwrap()))
            .collect();
        o += nimp * 4;
        let mut k: Vec<Vec<f32>> = Vec::with_capacity(heads);
        let mut v: Vec<Vec<f32>> = Vec::with_capacity(heads);
        for _ in 0..heads {
            for which in 0..2 {
                let n = u32_at(&mut o)? as usize;
                need(n * w, o)?;
                let xs: Vec<f32> = (0..n)
                    .map(|i| {
                        let at = o + i * w;
                        if f16 {
                            cortiq_core::quant::f16_to_f32(u16::from_le_bytes(
                                buf[at..at + 2].try_into().unwrap(),
                            ))
                        } else {
                            f32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
                        }
                    })
                    .collect();
                o += n * w;
                if which == 0 { k.push(xs) } else { v.push(xs) }
            }
        }
        self.reset_per_position_storage();
        self.k = k;
        self.v = v;
        self.imp = imp;
        self.seq_len = seq_len;
        Ok(o)
    }

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
            + self.linear_state.len() * std::mem::size_of::<f32>()
            // O(1) Nyström state (window + sinks + skeleton) — same
            // discipline: constant in context, but real memory.
            + self.o1_memory_bytes()
            // Natively bounded anchor: the ring is the whole state of the
            // layer, fixed by the header.
            + self.bounded_state_bytes()
    }

    /// Drop oldest positions, keeping the last `keep_last`.
    fn evict(&mut self, keep_last: usize) {
        // A collecting o1 layer owns a still-needed exact prefix and query
        // trace.  Evicting it would lower the effective seal boundary while
        // leaving q_buf untouched, so conversion could never match its KV
        // rows.  A sealed layer stores nothing per position — the Nyström
        // state IS the eviction policy; resetting seq_len here would lie
        // about the context depth.  Both states therefore bypass ordinary
        // eviction until the transition or explicit reset completes.  A
        // bounded anchor likewise: the ring evicts itself every token.
        // A sliding-window tail (`trim_window`) bounds itself and must stay
        // the exact contiguous window its attends index.
        if self.o1.is_some()
            || self.bounded.is_some()
            || self.tail.is_some()
            || self.seq_len <= keep_last
        {
            return;
        }
        self.generation = next_gen();
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
            self.kv_rows_changed(0);
            drop_front(&mut self.kq[h], d * hd);
            drop_front(&mut self.vq[h], d * hd);
            drop_front(&mut self.ks[h], d * hd.div_ceil(KV_K_GROUP));
            drop_front(&mut self.vs[h], d);
        }
        let d = drop.min(self.imp.len());
        self.imp.drain(..d);
        self.seq_len = keep_last;
    }

    /// Mass-based eviction: keep `sink` earliest positions (attention sinks),
    /// the `recent` latest, and fill the rest of the `keep_last` budget
    /// with the positions carrying the highest accumulated attention
    /// mass (vmfcore: PPL 8.342 vs 8.687 for recency-only, full 8.295).
    fn evict_born(&mut self, keep_last: usize, sink: usize, recent: usize) {
        if self.o1.is_some() || self.bounded.is_some() || self.tail.is_some() {
            // See evict(): collecting must retain the exact prefix as well as
            // sealed O(1) state must retain its own bounded representation;
            // a bounded anchor's ring is its own eviction, and a sliding
            // tail its own (a gather would make its rows non-contiguous).
            return;
        }
        let stored = self.imp.len();
        if stored <= keep_last {
            return;
        }
        self.generation = next_gen();
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
            self.imp[b]
                .partial_cmp(&self.imp[a])
                .unwrap_or(std::cmp::Ordering::Equal)
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
                self.kv_rows_changed(0);
            }
            if !self.v[h].is_empty() {
                self.v[h] = gather(&self.v[h], &kept, hd);
                self.kv_rows_changed(0);
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
    /// Mass-based eviction: sinks + recents + top accumulated attention mass.
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
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|li| {
                let mut l = LayerKvCache::new(num_kv_heads, head_dim);
                l.wire_layer = li as u32;
                l
            })
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

    /// Bytes owned by linear-core recurrent state (including the tentative
    /// speculative scratch).  This is reported separately from attention KV
    /// so a serving slot's O(1) capacity can be compared with its context
    /// cache without guessing from model geometry.
    pub fn recurrent_state_bytes(&self) -> usize {
        let floats: usize = self
            .layers
            .iter()
            .map(|l| l.linear_state.len() + l.linear_scratch.len())
            .sum();
        floats * std::mem::size_of::<f32>()
    }

    /// Attention KV (or sealed O(1) attention state) bytes, excluding the
    /// linear recurrent vectors returned by [`recurrent_state_bytes`].
    pub fn attention_state_bytes(&self) -> usize {
        self.total_memory_bytes()
            .saturating_sub(self.recurrent_state_bytes())
    }

    /// Current sequence length (max across layers — dead layers may lag):
    /// the absolute depth, which a trimmed sliding layer keeps in
    /// `pos_len()` while it stores only its tail.
    pub fn seq_len(&self) -> usize {
        self.layers.iter().map(|l| l.pos_len()).max().unwrap_or(0)
    }

    /// Bytes owned by bounded-anchor rings (constant in context).
    pub fn bounded_state_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.bounded_state_bytes()).sum()
    }

    /// True when some layer with PER-POSITION storage reached the cap.
    /// Bounded anchors hold nothing per position and never need it: a
    /// model whose every layer is O(1) has no eviction cliff at all. A
    /// sliding tail (`trim_window`) bounds itself and is not counted.
    pub fn needs_eviction(&self) -> bool {
        self.layers
            .iter()
            .filter(|l| l.bounded.is_none() && l.tail.is_none())
            .map(|l| l.seq_len)
            .max()
            .unwrap_or(0)
            >= self.max_seq_len
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

    /// The lazy K/V maximum the wgpu prefill's f16 guard reads: rows a
    /// truncation drops and later appends rewrite are scanned again, an
    /// inf or NaN reports +inf, `clear` starts afresh, and a caller's own
    /// maxima of the rows it just appended fold in only when they continue
    /// the scanned prefix.
    #[test]
    fn kv_abs_max_follows_appends_truncation_and_clear() {
        let (heads, hd) = (2usize, 4usize);
        let mut c = LayerKvCache::new(heads, hd);
        c.mode = KvMode::F32;
        let row = |x: f32| vec![x; heads * hd];
        for _ in 0..4 {
            c.append(&row(0.5), &row(-0.25), &[]);
        }
        assert_eq!(c.kv_abs_max(), (0.5, 0.25));
        c.truncate_last(2);
        c.append(&row(-9.0), &row(3.0), &[]);
        c.append(&row(0.5), &row(0.25), &[]);
        assert_eq!(c.kv_abs_max(), (9.0, 3.0), "rewritten rows were not rescanned");
        c.append(&row(f32::NAN), &row(0.0), &[]);
        assert_eq!(c.kv_abs_max().0, f32::INFINITY);
        c.clear();
        c.append(&row(1.0), &row(2.0), &[]);
        assert_eq!(c.kv_abs_max(), (1.0, 2.0));
        c.append(&row(4.0), &row(1.0), &[]);
        // Rows [1, 2) continue the scanned prefix: the caller's maxima.
        assert_eq!(c.kv_abs_max_after((1, 2, 4.0, 1.0)), (4.0, 2.0));
        c.append(&row(8.0), &row(1.0), &[]);
        // A stale `from` falls back to the scan.
        assert_eq!(c.kv_abs_max_after((0, 3, 0.0, 0.0)), (8.0, 2.0));
    }

    /// A trim drops rows from the front: `kv_abs_max` must still scan every
    /// row appended after it, also when more are appended than were dropped
    /// before the next call.
    #[test]
    fn kv_abs_max_scans_rows_appended_after_a_trim() {
        let (heads, hd, w) = (1usize, 4usize, 8usize);
        let mut c = LayerKvCache::new(heads, hd);
        c.mode = KvMode::F32;
        let row = |x: f32| vec![x; heads * hd];
        for _ in 0..(2 * w + 4) {
            c.append(&row(0.5), &row(0.5), &[]);
        }
        assert_eq!(c.kv_abs_max(), (0.5, 0.5));
        let d = c.trim_window(w, 2, 2);
        assert!(d > 0, "the trim must drop rows for this test");
        // The first row after the trim is large, then more rows than were
        // dropped: without moving the mark, rows [stored - d, old mark) —
        // the large one among them — would go unscanned.
        let first_new = c.kv_rows();
        c.append(&row(-7.0), &row(6.0), &[]);
        for _ in 0..(d + 3) {
            c.append(&row(0.5), &row(0.5), &[]);
        }
        assert!(c.kv_rows() > 2 * w + 4 && first_new < 2 * w + 4);
        assert_eq!(c.kv_abs_max(), (7.0, 6.0));
    }

    #[test]
    fn memory_breakdown_separates_recurrent_and_attention_state() {
        let mut cache = KvCache::new(1, 1, 4, 16);
        cache.layers[0].linear_state = vec![0.0; 8];
        cache.layers[0].linear_scratch = vec![0.0; 4];
        cache.layers[0].append(&[0.0; 4], &[1.0; 4], &[true]);
        let recurrent = cache.recurrent_state_bytes();
        assert_eq!(recurrent, 12 * std::mem::size_of::<f32>());
        assert_eq!(
            cache.attention_state_bytes() + recurrent,
            cache.total_memory_bytes()
        );
        assert!(cache.attention_state_bytes() > 0);
    }

    #[test]
    fn wire_round_trip_reproduces_attention() {
        // The state has to arrive as state, not as something that looks
        // like it: the oracle is what the layer ANSWERS, not what it
        // stores. Same query, same output, bit for bit.
        let (heads, hd) = (2usize, 4usize);
        let mut a = LayerKvCache::new(heads, hd);
        for p in 0..5 {
            let k: Vec<f32> = (0..heads * hd)
                .map(|i| (p * 10 + i) as f32 * 0.031)
                .collect();
            let v: Vec<f32> = (0..heads * hd)
                .map(|i| (p * 7 + i) as f32 * -0.017)
                .collect();
            a.append(&k, &v, &[true, true]);
        }
        a.linear_state = vec![0.5, -0.25, 1.0];
        let q: Vec<f32> = (0..hd).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        let bytes = a.export_wire(false).expect("f32 cache exports");
        let mut b = LayerKvCache::new(heads, hd);
        b.linear_scratch = vec![9.0; 3];
        b.import_wire(&bytes).expect("import");
        assert!(
            b.linear_scratch.is_empty(),
            "import must discard tentative state"
        );

        assert_eq!(b.seq_len, a.seq_len);
        assert_eq!(b.linear_state, a.linear_state);
        for h in 0..heads {
            let (oa, sa) = a.attend(&q, h);
            let (ob, sb) = b.attend(&q, h);
            assert_eq!(oa, ob, "head {h} attention output diverged");
            assert_eq!(sa, sb, "head {h} attention scores diverged");
        }
    }

    #[test]
    fn wire_refuses_what_it_cannot_describe() {
        // A refusal is the feature: a cache whose extra state this format
        // does not carry must not travel looking complete.
        let mut c = LayerKvCache::new(1, 4);
        c.mode = KvMode::Q8 { k: true, v: true };
        let err = c.export_wire(false).unwrap_err();
        assert!(err.contains("F32"), "{err}");
    }

    #[test]
    fn wire_refuses_unversioned_delta_state() {
        // The versioned wire carries the operator identity, so a delta
        // layer exports; only the OLD unversioned body is refused.
        let mut c = LayerKvCache::new(1, 4);
        c.set_linear_wire_allowed(false);
        let bytes = c.export_wire(false).expect("v2 export carries identity");
        assert_eq!(&bytes[..4], WIRE_MAGIC);
        let err = c.import_wire(&[0, 0, 0, 0]).unwrap_err();
        assert!(err.contains("operator identity"), "{err}");
    }

    #[test]
    fn wire_v2_round_trips_linear_and_bounded_records() {
        // Linear record: the recurrent vector travels f32 whatever the
        // wire dtype and the per-position stores come back empty.
        let mut a = LayerKvCache::new(1, 4);
        a.wire_kind = WireKind::Linear;
        a.wire_identity = 0xC0FFEE;
        a.linear_state = vec![0.5, -0.25, 1.0, 3.5];
        a.seq_len = 9;
        let bytes = a.export_wire(true).unwrap();
        let mut b = LayerKvCache::new(1, 4);
        b.wire_kind = WireKind::Linear;
        b.wire_identity = 0xC0FFEE;
        b.import_wire(&bytes).unwrap();
        assert_eq!(b.linear_state, a.linear_state);
        assert_eq!(b.seq_len, 9);
        // Identity mismatch is a refusal, not a warning.
        let mut c = LayerKvCache::new(1, 4);
        c.wire_kind = WireKind::Linear;
        let err = c.import_wire(&bytes).unwrap_err();
        assert!(err.contains("operator identity"), "{err}");

        // Bounded record, f32 and f16 rings.
        let (kvh, hd, w) = (2, 4, 8);
        let mut a = LayerKvCache::new(kvh, hd);
        a.install_bounded(w);
        let k: Vec<f32> = (0..kvh * hd).map(|i| i as f32 * 0.125).collect();
        for p in 0..11 {
            a.bounded.as_mut().unwrap().insert(&k, &k);
            a.seq_len = p + 1;
        }
        for f16 in [false, true] {
            let bytes = a.export_wire(f16).unwrap();
            let mut b = LayerKvCache::new(kvh, hd);
            b.install_bounded(w);
            b.import_wire(&bytes).unwrap();
            let (ra, rb) = (a.bounded.as_ref().unwrap(), b.bounded.as_ref().unwrap());
            assert_eq!(rb.seen, 11);
            assert_eq!(b.seq_len, 11);
            // 0.125 multiples are exact in f16, so both dtypes round-trip bit for bit.
            assert!(ra.same_state(rb), "f16={f16}");
            // A ring of another width refuses the record.
            let mut c = LayerKvCache::new(kvh, hd);
            c.install_bounded(w * 2);
            assert!(c.import_wire(&bytes).is_err());
        }
    }

    #[test]
    fn wire_import_checks_geometry() {
        let a = LayerKvCache::new(2, 4);
        let bytes = a.export_wire(false).unwrap();
        let mut wrong = LayerKvCache::new(2, 8);
        let err = wrong.import_wire(&bytes).unwrap_err();
        assert!(err.contains("2×4"), "{err}");
    }

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
        for l in &mut cache.layers {
            l.mode = KvMode::F32;
        }
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
    fn collecting_o1_eviction_retains_exact_storage_until_boundary() {
        const B: usize = 19;
        let q = vec![0.1f32; 8];
        let k = vec![0.2f32; 4];
        let v = vec![0.3f32; 4];

        for policy in [EvictionPolicy::Recent, EvictionPolicy::Born { sink: 2 }] {
            let mut cache = KvCache::new(1, 1, 4, 6);
            cache.policy = policy;
            cache.layers[0].mode = KvMode::F32;
            cache.layers[0].o1_begin_with_boundary(
                4,
                8,
                2,
                crate::nystrom::O1Rect::Aggregate,
                Some(B),
            );

            for pos in 0..B {
                {
                    let layer = &mut cache.layers[0];
                    layer.o1_push_q(&q);
                    layer.append(&k, &v, &[]);
                }
                if pos + 1 < B {
                    cache.evict(3);
                }
            }

            let layer = &cache.layers[0];
            let rows = B * layer.head_dim;
            assert_eq!(layer.seq_len, B, "policy {policy:?} retained depth");
            assert_eq!(layer.k[0].len(), rows, "policy {policy:?} K rows");
            assert_eq!(layer.v[0].len(), rows, "policy {policy:?} V rows");
            assert!(
                layer.k[0].capacity() >= rows,
                "policy {policy:?} K capacity"
            );
            assert!(
                layer.v[0].capacity() >= rows,
                "policy {policy:?} V capacity"
            );
            let q_capacity = match layer.o1.as_ref() {
                Some(O1State::Collecting { q_buf, .. }) => q_buf.capacity(),
                other => panic!("policy {policy:?} changed state early: {other:?}"),
            };
            assert!(
                q_capacity >= B * 8,
                "policy {policy:?} Q capacity must cover the exact prefix"
            );

            assert!(cache.layers[0].o1_seal_checked(2).unwrap());
            assert_eq!(cache.layers[0].k[0].capacity(), 0, "K released after seal");
            assert_eq!(cache.layers[0].v[0].capacity(), 0, "V released after seal");
        }
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
    /// int8 grid. Plus rollback and mass-based eviction on the q8 storage.
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
        let q: Vec<f32> = (0..hd)
            .map(|i| ((i * 13 + 5) % 89) as f32 / 89.0 - 0.5)
            .collect();
        for g in 0..heads {
            let (of, pf) = f.attend(&q, g);
            let (o8, p8) = q8.attend(&q, g);
            let scale = of.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
            for d in 0..hd {
                assert!(
                    (of[d] - o8[d]).abs() <= scale * 0.03 + 1e-3,
                    "g{g} d{d}: f32 {} vs q8 {}",
                    of[d],
                    o8[d]
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

    /// Grouped GQA attend must be bit-identical to per-head attend in
    /// every KV mode (it is the same math with rows streamed once).
    #[test]
    fn attend_group_equals_per_head_attend_bitexact() {
        let (kv_heads, hd, hpk) = (2usize, 32usize, 3usize); // 6 Q-heads
        for mode in [KvMode::F32, KvMode::Q8 { k: true, v: true }] {
            let mut c = LayerKvCache::new(kv_heads, hd);
            c.mode = mode;
            for p in 0..70 {
                let k: Vec<f32> = (0..kv_heads * hd)
                    .map(|i| ((i * 31 + p * 17 + 3) % 97) as f32 / 97.0 - 0.5)
                    .collect();
                let v: Vec<f32> = (0..kv_heads * hd)
                    .map(|i| ((i * 13 + p * 29 + 7) % 89) as f32 / 89.0 - 0.5)
                    .collect();
                c.append(&k, &v, &[true; 2]);
            }
            let q: Vec<f32> = (0..kv_heads * hpk * hd)
                .map(|i| ((i * 11 + 5) % 83) as f32 / 83.0 - 0.5)
                .collect();
            for g in 0..kv_heads {
                let span = g * hpk * hd..(g + 1) * hpk * hd;
                let mut out = vec![0f32; hpk * hd];
                let mut imp = vec![0f32; 70];
                c.attend_group(
                    &q[span.clone()],
                    g,
                    &mut out,
                    &mut imp,
                    1.0 / (hd as f32).sqrt(),
                    0,
                    0.0,
                    &[],
                );
                let mut imp_ref = vec![0f32; 70];
                for h in 0..hpk {
                    let qh = &q[span.start + h * hd..span.start + (h + 1) * hd];
                    let (o, probs) = c.attend(qh, g);
                    assert_eq!(
                        &out[h * hd..(h + 1) * hd],
                        &o[..],
                        "mode {mode:?} g{g} h{h}: grouped attend must be bit-identical"
                    );
                    for (dst, &p) in imp_ref.iter_mut().zip(&probs) {
                        *dst += p;
                    }
                }
                assert_eq!(
                    imp, imp_ref,
                    "mode {mode:?} g{g}: attention mass must match"
                );
            }
        }
    }

    /// Learned sinks (gpt-oss / MiMo-V2): the grouped softmax must equal a
    /// softmax over the visible rows PLUS an explicit value-less sink
    /// column, computed independently in f64 — at position 0 (one stored
    /// row), mid-sequence, and with the window truncating the rows. The
    /// importance row must be the row probabilities (the sink's share is
    /// nobody's importance).
    #[test]
    fn sink_attend_matches_explicit_sink_column() {
        let (nkv, hd, hpk) = (2usize, 8usize, 3usize);
        let rows = 9usize;
        let mut c = LayerKvCache::new(nkv, hd);
        c.mode = KvMode::F32;
        let kv = |r: usize, i: usize, a: usize, m: usize| {
            (((r * a + i * 7 + 3) % m) as f32 / m as f32 - 0.5) * 2.0
        };
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for r in 0..rows {
            let k: Vec<f32> = (0..nkv * hd).map(|i| kv(r, i, 31, 97)).collect();
            let v: Vec<f32> = (0..nkv * hd).map(|i| kv(r, i, 17, 89)).collect();
            c.append(&k, &v, &[]);
            ks.push(k);
            vs.push(v);
        }
        let q: Vec<f32> = (0..nkv * hpk * hd)
            .map(|i| (((i * 11 + 5) % 83) as f32 / 83.0 - 0.5) * 3.0)
            .collect();
        // Mixed signs and one sink that dominates its head.
        let sinks = [0.7f32, -1.3, 2.5, 0.0, -4.0, 6.0];
        let scale = 1.0 / (hd as f32).sqrt();
        let mut checked = 0usize;
        for upto in [1usize, 2, 5, 9] {
            for window in [None, Some(3usize), Some(1)] {
                let first = window.map(|w| upto.saturating_sub(w)).unwrap_or(0);
                for g in 0..nkv {
                    let qg = &q[g * hpk * hd..(g + 1) * hpk * hd];
                    let sg = &sinks[g * hpk..(g + 1) * hpk];
                    let mut out = vec![0f32; hpk * hd];
                    let mut imp = vec![0f32; upto];
                    c.attend_group_upto(qg, g, &mut out, &mut imp, scale, first, 0.0, upto, sg);
                    let mut imp_ref = vec![0f64; upto];
                    for h in 0..hpk {
                        let qh = &qg[h * hd..(h + 1) * hd];
                        // Logits of the visible rows, then the sink column.
                        let mut z: Vec<f64> = (first..upto)
                            .map(|p| {
                                let k = &ks[p][g * hd..(g + 1) * hd];
                                qh.iter()
                                    .zip(k)
                                    .map(|(&a, &b)| a as f64 * b as f64)
                                    .sum::<f64>()
                                    * scale as f64
                            })
                            .collect();
                        z.push(sg[h] as f64);
                        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
                        let s: f64 = e.iter().sum();
                        let p: Vec<f64> = e.iter().map(|&x| x / s).collect();
                        // The sink column carries a zero value vector.
                        for d in 0..hd {
                            let want: f64 = (first..upto)
                                .map(|r| p[r - first] * vs[r][g * hd + d] as f64)
                                .sum();
                            let got = out[h * hd + d] as f64;
                            assert!(
                                (got - want).abs() < 1e-6,
                                "upto {upto} window {window:?} g{g} h{h} d{d}: {got} vs {want}"
                            );
                        }
                        for r in first..upto {
                            imp_ref[r] += p[r - first];
                        }
                        checked += 1;
                    }
                    for r in 0..upto {
                        assert!(
                            (imp[r] as f64 - imp_ref[r]).abs() < 1e-6,
                            "imp upto {upto} window {window:?} g{g} row {r}: {} vs {}",
                            imp[r],
                            imp_ref[r]
                        );
                    }
                    // The sink takes real mass: rows sum below 1.
                    let row_mass: f32 = imp.iter().sum();
                    assert!(row_mass < hpk as f32, "sinks must absorb some mass");
                }
            }
        }
        assert_eq!(checked, 4 * 3 * nkv * hpk);
    }

    /// Scoring only the window's rows must be bit-identical to the former
    /// whole-row scoring (−inf outside the window) — with and without a
    /// sink, output and importance. The reference is the plain per-head
    /// softmax over the same rows written out here.
    #[test]
    fn windowed_attend_equals_masked_full_row() {
        let (nkv, hd, hpk) = (1usize, 16usize, 2usize);
        let rows = 40usize;
        let mut c = LayerKvCache::new(nkv, hd);
        c.mode = KvMode::F32;
        for r in 0..rows {
            let k: Vec<f32> = (0..hd)
                .map(|i| ((r * 13 + i * 5) % 29) as f32 / 29.0 - 0.5)
                .collect();
            let v: Vec<f32> = (0..hd)
                .map(|i| ((r * 7 + i * 3) % 31) as f32 / 31.0 - 0.5)
                .collect();
            c.append(&k, &v, &[]);
        }
        let q: Vec<f32> = (0..hpk * hd)
            .map(|i| ((i * 19) % 23) as f32 / 23.0 - 0.5)
            .collect();
        let scale = 0.25f32;
        for w in [1usize, 7, 39, 40, 100] {
            let first = rows.saturating_sub(w);
            let mut out = vec![0f32; hpk * hd];
            let mut imp = vec![0f32; rows];
            c.attend_group(&q, 0, &mut out, &mut imp, scale, first, 0.0, &[]);
            // Reference: the historical full-row −inf-masked kernel.
            let mut out_ref = vec![0f32; hpk * hd];
            let mut imp_ref = vec![0f32; rows];
            for h in 0..hpk {
                let mut s = vec![f32::NEG_INFINITY; rows];
                for p in first..rows {
                    s[p] = crate::attention::dot_f32(
                        &q[h * hd..(h + 1) * hd],
                        &c.head_keys(0)[p * hd..(p + 1) * hd],
                    ) * scale;
                }
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in s.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v;
                }
                for v in s.iter_mut() {
                    *v /= sum;
                }
                for p in first..rows {
                    if s[p].abs() < 1e-12 {
                        continue;
                    }
                    crate::attention::axpy_f32(
                        &mut out_ref[h * hd..(h + 1) * hd],
                        &c.head_values(0)[p * hd..(p + 1) * hd],
                        s[p],
                    );
                }
                for (d, &p) in imp_ref.iter_mut().zip(&s) {
                    *d += p;
                }
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&out), bits(&out_ref), "out window {w}");
            assert_eq!(bits(&imp), bits(&imp_ref), "imp window {w}");
        }
    }

    /// Sinks are weights: a new conversation (clear) and a wire import
    /// must keep them.
    #[test]
    fn sinks_survive_clear_and_wire_import() {
        let mut c = LayerKvCache::new(1, 4);
        c.mode = KvMode::F32;
        c.sinks = Some(vec![0.5, -0.5]);
        c.append(&[1.0; 4], &[2.0; 4], &[]);
        let wire = c.export_wire(false).unwrap();
        c.clear();
        assert_eq!(c.sinks.as_deref(), Some(&[0.5f32, -0.5][..]));
        c.import_wire(&wire).unwrap();
        assert_eq!(c.sinks.as_deref(), Some(&[0.5f32, -0.5][..]));
        assert_eq!(c.seq_len, 1);
    }

    /// Review regression: mass-based eviction in MIXED modes. q8v used to
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
            assert!(
                c.memory_bytes() < before / 2,
                "memory must shrink (k={mk} v={mv})"
            );
            // V rows match the kept set: the heaviest positions
            // (tail 60..79) must be present in the attend output.
            let (out, _) = c.attend(&[1.0, 1.0, 1.0, 1.0], 0);
            assert!(
                out[0] > 30.0,
                "V from the kept tail, not the stale head (k={mk} v={mv}, out {})",
                out[0]
            );
        }
    }

    #[test]
    fn born_eviction_keeps_high_mass_position() {
        let mut cache = KvCache::new(1, 1, 2, 16);
        cache.policy = EvictionPolicy::Born { sink: 1 };
        for l in &mut cache.layers {
            l.mode = KvMode::F32;
        }
        let layer = &mut cache.layers[0];
        // 8 positions; keys carry the position index so we can verify
        // exactly which positions survive the gather.
        for pos in 0..8 {
            let k = vec![pos as f32; 2];
            let v = vec![pos as f32 + 100.0; 2];
            layer.append(&k, &v, &[true]);
        }
        // Position 3 carries the most attention mass.
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
            "kept = sink(0) + mass-top(3) + recent(6,7)"
        );
        // imp stays aligned with the gathered positions.
        assert_eq!(layer.head_len(0), 4);
    }

    // ── Sliding-window tail (trim_window) ──

    /// Deterministic row of `n` values in [-1, 1) for position `p`.
    fn swa_row(p: usize, n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((p * 7919 + i * 104_729 + salt * 1_299_709) % 2003) as f32 / 1001.5 - 1.0)
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The stored tail is the untrimmed cache's rows from `base` on, in
    /// order, with the bounds the trigger promises — a dead head included.
    #[test]
    fn swa_trim_keeps_contiguous_tail() {
        let (nkv, hd, w, slack, align) = (2usize, 8usize, 6usize, 2usize, 4usize);
        let alive = [true, false];
        let mut full = LayerKvCache::new(nkv, hd);
        full.mode = KvMode::F32;
        let mut t = full.clone();
        let mut trims = 0;
        for p in 0..100 {
            let (k, v) = (swa_row(p, nkv * hd, 1), swa_row(p, nkv * hd, 2));
            full.append(&k, &v, &alive);
            t.append(&k, &v, &alive);
            if t.trim_window(w, slack, align) > 0 {
                trims += 1;
                assert!(
                    (w + slack..w + slack + align).contains(&t.seq_len),
                    "rows after a trim: {}",
                    t.seq_len
                );
            }
            assert!(t.seq_len <= 2 * w, "rows {} past the trigger", t.seq_len);
            assert_eq!(t.pos_len(), p + 1);
            assert_eq!(t.base() % align, 0);
            assert_eq!(t.base() + t.seq_len, full.seq_len);
            let b = t.base();
            assert_eq!(t.head_keys(0), &full.head_keys(0)[b * hd..]);
            assert_eq!(t.head_values(0), &full.head_values(0)[b * hd..]);
            assert!(t.head_keys(1).is_empty() && t.head_values(1).is_empty());
            assert_eq!(t.imp.len(), t.seq_len);
        }
        assert!(trims > 5, "only {trims} trims in 100 positions");
        assert_eq!(t.tail_window(), Some(w));
        assert_eq!(full.tail_window(), None);
    }

    /// Two caches fed the same rows, one trimmed: every windowed attend
    /// (`first = rows − w`) gives the same output and importance, bit for
    /// bit, under F32 and the q8 cache — the q8 field freezes at 64 rows,
    /// before the first trim.
    #[test]
    fn swa_trimmed_attend_equals_untrimmed() {
        let (nkv, hd, hpk) = (2usize, 16usize, 2usize);
        for mode in [KvMode::F32, KvMode::Q8 { k: true, v: true }] {
            for (w, slack, align) in [(40usize, 2usize, 4usize), (37, 5, 16), (64, 64, 64)] {
                let mut full = LayerKvCache::new(nkv, hd);
                full.mode = mode;
                let mut t = full.clone();
                let scale = 0.3f32;
                let mut trims = 0;
                for p in 0..400 {
                    let (k, v) = (swa_row(p, nkv * hd, 3), swa_row(p, nkv * hd, 4));
                    full.append(&k, &v, &[]);
                    t.append(&k, &v, &[]);
                    let q = swa_row(p, nkv * hpk * hd, 5);
                    for g in 0..nkv {
                        let qg = &q[g * hpk * hd..(g + 1) * hpk * hd];
                        let (nf, nt) = (full.head_len(g), t.head_len(g));
                        let (mut of, mut ot) = (vec![0f32; hpk * hd], vec![0f32; hpk * hd]);
                        let (mut imf, mut imt) = (vec![0f32; nf], vec![0f32; nt]);
                        full.attend_group(
                            qg,
                            g,
                            &mut of,
                            &mut imf,
                            scale,
                            nf.saturating_sub(w),
                            0.0,
                            &[],
                        );
                        t.attend_group(
                            qg,
                            g,
                            &mut ot,
                            &mut imt,
                            scale,
                            nt.saturating_sub(w),
                            0.0,
                            &[],
                        );
                        assert_eq!(bits(&of), bits(&ot), "{mode:?} w {w} pos {p} group {g}");
                        assert_eq!(bits(&imf[t.base()..]), bits(&imt), "{mode:?} w {w} pos {p}");
                        full.accumulate_imp(&imf);
                        t.accumulate_imp(&imt);
                    }
                    if t.trim_window(w, slack, align) > 0 {
                        trims += 1;
                    }
                    assert_eq!(bits(&full.imp[t.base()..]), bits(&t.imp));
                }
                assert!(trims > 0, "{mode:?} w {w}: never trimmed");
            }
        }
    }

    /// A rollback no deeper than the slack, then new rows: still the
    /// untrimmed answer.
    #[test]
    fn swa_truncate_after_trim() {
        let (nkv, hd, w) = (1usize, 8usize, 10usize);
        let mut full = LayerKvCache::new(nkv, hd);
        full.mode = KvMode::F32;
        let mut t = full.clone();
        let mut p = 0usize;
        for round in 0..60 {
            for _ in 0..3 {
                let (k, v) = (swa_row(p, hd, 6), swa_row(p, hd, 7));
                full.append(&k, &v, &[]);
                t.append(&k, &v, &[]);
                p += 1;
            }
            t.trim_window(w, 2, 4);
            // Speculative-style rollback of 2, then replacement rows.
            full.truncate_last(2);
            t.truncate_last(2);
            p -= 2;
            assert_eq!(t.pos_len(), full.seq_len);
            for _ in 0..2 {
                let (k, v) = (swa_row(p, hd, 8 + round), swa_row(p, hd, 9 + round));
                full.append(&k, &v, &[]);
                t.append(&k, &v, &[]);
                p += 1;
            }
            let q = swa_row(p, hd, 10);
            let (nf, nt) = (full.head_len(0), t.head_len(0));
            let (mut of, mut ot) = (vec![0f32; hd], vec![0f32; hd]);
            let (mut imf, mut imt) = (vec![0f32; nf], vec![0f32; nt]);
            full.attend_group(&q, 0, &mut of, &mut imf, 0.5, nf - w.min(nf), 0.0, &[]);
            t.attend_group(&q, 0, &mut ot, &mut imt, 0.5, nt - w.min(nt), 0.0, &[]);
            assert_eq!(bits(&of), bits(&ot), "round {round}");
        }
        assert!(t.base() > 0);
    }

    /// The cache-wide eviction leaves a sliding tail alone, and a tail
    /// does not count toward the eviction trigger.
    #[test]
    fn swa_evict_skips_tail() {
        for policy in [EvictionPolicy::Recent, EvictionPolicy::Born { sink: 1 }] {
            let mut c = KvCache::new(2, 1, 4, 30);
            c.policy = policy;
            for p in 0..25 {
                for l in &mut c.layers {
                    l.append(&swa_row(p, 4, 1), &swa_row(p, 4, 2), &[]);
                }
                c.layers[0].trim_window(8, 2, 4);
            }
            let t_rows = c.layers[0].seq_len;
            let t_base = c.layers[0].base();
            assert!(t_base > 0);
            assert!(!c.needs_eviction(), "25 rows < cap 30");
            for p in 25..30 {
                for l in &mut c.layers {
                    l.append(&swa_row(p, 4, 1), &swa_row(p, 4, 2), &[]);
                }
            }
            // Layer 0 (tail, 5 rows past its last trim) is not what triggers.
            assert!(c.needs_eviction());
            let gen0 = c.layers[0].generation();
            c.evict(10);
            assert_eq!(c.layers[0].seq_len, t_rows + 5, "{policy:?}: tail evicted");
            assert_eq!(c.layers[0].base(), t_base);
            assert_eq!(c.layers[0].generation(), gen0);
            assert_eq!(
                c.layers[1].seq_len, 10,
                "{policy:?}: full layer not evicted"
            );
            assert_eq!(c.seq_len(), 30, "absolute depth from the tail");
        }
    }

    /// `generation` moves on every mutation a device copy cannot follow by its
    /// row count, and stays on appends and rollbacks.
    #[test]
    fn swa_gen_bumps() {
        let mut l = LayerKvCache::new(1, 4);
        l.mode = KvMode::F32;
        let g0 = l.generation();
        assert_ne!(
            LayerKvCache::new(1, 4).generation(),
            g0,
            "fresh caches differ"
        );
        for p in 0..20 {
            l.append(&swa_row(p, 4, 1), &swa_row(p, 4, 2), &[]);
        }
        assert_eq!(l.generation(), g0, "append");
        l.truncate_last(2);
        assert_eq!(l.generation(), g0, "truncate_last");
        assert_eq!(l.trim_window(10, 2, 4), 0, "18 rows ≤ 2w: no trim");
        assert_eq!(l.generation(), g0, "a no-op trim");
        for p in 18..21 {
            l.append(&swa_row(p, 4, 1), &swa_row(p, 4, 2), &[]);
        }
        assert!(l.trim_window(10, 2, 4) > 0);
        let g1 = l.generation();
        assert_ne!(g1, g0, "trim");
        let snap = l.clone();
        l.clear();
        let g2 = l.generation();
        assert_ne!(g2, g1, "clear");
        assert_eq!((l.base(), l.pos_len(), l.tail_window()), (0, 0, Some(10)));
        let mut e = LayerKvCache::new(1, 4);
        e.mode = KvMode::F32;
        for p in 0..12 {
            e.append(&swa_row(p, 4, 1), &swa_row(p, 4, 2), &[]);
        }
        let ge = e.generation();
        e.evict(20);
        assert_eq!(e.generation(), ge, "an eviction that does nothing");
        e.evict(6);
        assert_ne!(e.generation(), ge, "evict");
        let ge = e.generation();
        e.evict_born(3, 1, 1);
        assert_ne!(e.generation(), ge, "evict_born");
        let mut i = LayerKvCache::new(1, 4);
        let gi = i.generation();
        i.import_wire(&snap.export_wire(false).unwrap()).unwrap();
        assert_ne!(i.generation(), gi, "import");
        assert_ne!(
            i.generation(),
            snap.generation(),
            "an import is new storage"
        );
    }

    /// A trimmed layer travels as `FullTail` and answers the same on the
    /// far side; an untrimmed one is still byte-for-byte `Full`; a layer
    /// with another kind refuses the tail.
    #[test]
    fn wire_full_tail_roundtrip() {
        let (nkv, hd, w) = (2usize, 4usize, 6usize);
        let mut a = LayerKvCache::new(nkv, hd);
        a.mode = KvMode::F32;
        for p in 0..5 {
            a.append(&swa_row(p, nkv * hd, 1), &swa_row(p, nkv * hd, 2), &[]);
        }
        let plain = a.export_wire(false).unwrap();
        assert_eq!(plain[20], WireKind::Full as u8, "untrimmed stays Full");
        assert_eq!(u64::from_le_bytes(plain[24..32].try_into().unwrap()), 5);
        for p in 5..40 {
            a.append(&swa_row(p, nkv * hd, 1), &swa_row(p, nkv * hd, 2), &[]);
            a.trim_window(w, 2, 4);
        }
        assert!(a.base() > 0);
        for f16 in [false, true] {
            let bytes = a.export_wire(f16).unwrap();
            assert_eq!(bytes[20], WireKind::FullTail as u8);
            assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 40);
            let mut b = LayerKvCache::new(nkv, hd);
            b.import_wire(&bytes).expect("tail import");
            assert_eq!(
                (b.base(), b.seq_len, b.pos_len()),
                (a.base(), a.seq_len, 40)
            );
            if !f16 {
                let q = swa_row(99, 2 * hd, 3);
                for g in 0..nkv {
                    let n = a.head_len(g);
                    let (mut oa, mut ob) = (vec![0f32; 2 * hd], vec![0f32; 2 * hd]);
                    let (mut ia, mut ib) = (vec![0f32; n], vec![0f32; n]);
                    a.attend_group(&q, g, &mut oa, &mut ia, 0.5, n - w, 0.0, &[]);
                    b.attend_group(&q, g, &mut ob, &mut ib, 0.5, n - w, 0.0, &[]);
                    assert_eq!(bits(&oa), bits(&ob));
                }
                // ... and a Full import over it puts row 0 back at 0.
                b.import_wire(&plain).unwrap();
                assert_eq!((b.base(), b.pos_len()), (0, 5));
            }
        }
        // A position that disagrees with base + rows is refused.
        let mut bad = a.export_wire(false).unwrap();
        bad[24..32].copy_from_slice(&41u64.to_le_bytes());
        assert!(LayerKvCache::new(nkv, hd).import_wire(&bad).is_err());
        // A bounded layer does not take a tail.
        let mut r = LayerKvCache::new(nkv, hd);
        r.install_bounded(8);
        let err = r.import_wire(&a.export_wire(false).unwrap()).unwrap_err();
        assert!(err.contains("does not match"), "{err}");
    }
}
