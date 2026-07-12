//! Nyström (landmark) attention kernel — streaming per-head runtime for
//! long-context `attn_type: nystrom` layers.
//!
//! Attention splits into an EXACT sliding window (last `w` keys) and a
//! landmark-skeleton far field sharing ONE joint denominator:
//!
//! ```text
//! out(q_t) = (Σ_{j>t-w} e_j·v_j + F·M·T_far) / (Σ_{j>t-w} e_j + F·M·Z_far)
//! e_j   = exp(q_t·k_j/√d)                      exact near weights
//! F_i   = exp(q_t·k̃_i/√d)                      scores vs landmark keys
//! M     = pinv_reg(exp(Q̃·K̃ᵀ/√d))               fixed after prefill
//! T_far = Σ_{j≤t-w} exp(Q̃·k_j/√d)·v_jᵀ         [m × dv]
//! Z_far = Σ_{j≤t-w} exp(Q̃·k_j/√d)              [m]
//! ```
//!
//! exp(q·k) is a PSD kernel, so the UNNORMALIZED skeleton (classic
//! Nyström/CUR) is legal.  Do NOT row-softmax the factors and do NOT
//! normalize the key scores over landmarks — both "simplifications"
//! measurably collapse quality (validated in the torch matrix probes).
//!
//! Boundary discipline: key j enters T/Z at the exact step it LEAVES
//! the window (t = j+w) — delayed insertion, no overlap, no hole; the
//! near mass stays exact rather than Nyström-estimated.
//!
//! Sink tokens (spec §5b, StreamingLLM discipline): the first `sink`
//! keys of the sequence are PERMANENT exact keys — the near mask is
//! (t-j < w) OR (j < sink) — and must never enter the far accumulators.
//! Here they never enter the ring window in the first place (they live
//! in a dedicated buffer), so delayed insertion cannot see them: no
//! double count, no gap.  Measured: sinks make the full 28/28-layer
//! O(1) conversion viable (×1.177 zero-shot) — the default mode.
//!
//! fp32 numerics: raw exp overflows on real logits, so shifts are
//! absorbed into diagonals.  T̂[i]/Ẑ[i] live at scale e^{-m_i} with a
//! per-landmark running max m_i (flash-style rescale on growth); each
//! token's landmark row uses its own shift f; near and far are brought
//! to one common scale before the single joint division.

/// Ridge factor for the regularized pseudo-inverse of the landmark
/// kernel: λ = RIDGE_REL · mean(diag(AᵀA)).
const RIDGE_REL: f64 = 1e-6;
/// Floor for the joint denominator (mirrors the reference probe).
const DEN_EPS: f32 = 1e-30;
/// Prompts of length ≤ w + EXACT_SLACK skip the skeleton entirely:
/// tiny prefills duplicate segment-mean landmarks (singular Au).
const EXACT_SLACK: usize = 8;

/// Streaming per-head Nyström attention state.
///
/// Lifecycle: `new(m, w, sink)` → `prefill(prompt)` once → `step()` per
/// decode token.  All buffers are flat `Vec<f32>`, row-major; the
/// skeleton path performs no allocations inside `step()`.
#[derive(Clone, Debug)]
pub struct NystromState {
    /// Landmark budget (m) — effective count may be lower (`m_eff`).
    m: usize,
    /// Exact-window width in keys.
    w: usize,
    /// Permanent exact sink keys at positions 0..sink (spec §5b).
    sink: usize,
    d: usize,
    dv: usize,
    /// Effective landmark count: clamp(t/8, 4, m) at prefill.
    m_eff: usize,
    /// Short-prompt mode: window holds ALL keys, no skeleton.  The
    /// buffer grows on decode, so this mode may allocate in `step()` —
    /// acceptable for the ≤ w+8-token degenerate case.
    exact_only: bool,
    scale: f32,
    /// Window keys `[cap][d]` — ring buffer in skeleton mode (cap = w),
    /// append-only in exact-only mode.
    win_k: Vec<f32>,
    /// Window values `[cap][dv]`.
    win_v: Vec<f32>,
    win_len: usize,
    /// Ring slot of the OLDEST window entry (0 while not yet full).
    win_head: usize,
    /// Sink keys `[sink_len][d]` — filled once at prefill, immutable.
    sink_k: Vec<f32>,
    /// Sink values `[sink_len][dv]`.
    sink_v: Vec<f32>,
    /// Number of stored sink tokens (0 in exact-only mode, where every
    /// key is permanent-exact anyway).
    sink_len: usize,
    /// Far numerator `[m_eff][dv]`, stored at scale e^{-m_max[i]}.
    t_hat: Vec<f32>,
    /// Far denominator `[m_eff]`, same scale.
    z_hat: Vec<f32>,
    /// Per-landmark running max of far logits q̃_i·k_j/√d.
    m_max: Vec<f32>,
    /// Number of keys absorbed into the far field.
    far_len: usize,
    /// Query landmarks `[m_eff][d]` (segment means of the prefill).
    q_tilde: Vec<f32>,
    /// Key landmarks `[m_eff][d]`.
    k_tilde: Vec<f32>,
    /// Regularized pseudo-inverse of Au = exp(Q̃·K̃ᵀ/√d), `[m_eff][m_eff]`.
    mu: Vec<f32>,
    // Scratch preallocated at prefill so skeleton-mode step() is
    // allocation-free.
    scr_s: Vec<f32>,
    scr_fh: Vec<f32>,
    scr_u: Vec<f32>,
    scr_l: Vec<f32>,
}

impl NystromState {
    /// `m` — landmark budget (≥ 4; validated setting is 32),
    /// `w` — exact window width (validated setting is 128),
    /// `sink` — permanent exact sink keys (validated default is 4;
    /// 0 reproduces the sink-free kernel bit-for-bit).
    pub fn new(m: usize, w: usize, sink: usize) -> Self {
        assert!(m >= 4, "landmark budget must be at least 4");
        assert!(w >= 1, "window must hold at least one key");
        NystromState {
            m,
            w,
            sink,
            d: 0,
            dv: 0,
            m_eff: 0,
            exact_only: true,
            scale: 0.0,
            win_k: Vec::new(),
            win_v: Vec::new(),
            win_len: 0,
            win_head: 0,
            sink_k: Vec::new(),
            sink_v: Vec::new(),
            sink_len: 0,
            t_hat: Vec::new(),
            z_hat: Vec::new(),
            m_max: Vec::new(),
            far_len: 0,
            q_tilde: Vec::new(),
            k_tilde: Vec::new(),
            mu: Vec::new(),
            scr_s: Vec::new(),
            scr_fh: Vec::new(),
            scr_u: Vec::new(),
            scr_l: Vec::new(),
        }
    }

    /// Absorb the whole prompt for this head: freeze landmarks and M,
    /// then replay the prompt through the step() state semantics
    /// (window fill + delayed far insertion).  `qs`/`ks` are `[t][d]`,
    /// `vs` is `[t][dv]`, all row-major.
    pub fn prefill(&mut self, qs: &[f32], ks: &[f32], vs: &[f32], t: usize, d: usize, dv: usize) {
        assert_eq!(qs.len(), t * d);
        assert_eq!(ks.len(), t * d);
        assert_eq!(vs.len(), t * dv);
        self.d = d;
        self.dv = dv;
        self.scale = 1.0 / (d as f32).sqrt();
        self.win_len = 0;
        self.win_head = 0;
        self.far_len = 0;
        self.sink_len = 0;
        self.exact_only = t <= self.w + self.sink + EXACT_SLACK;

        if self.exact_only {
            // Everything fits in the exact window (plus slack for a few
            // decode steps before Vec growth); no skeleton is built and
            // no separate sink buffer is needed — every key is already
            // a permanent exact key in this mode.
            self.win_k = Vec::with_capacity((t + 64) * d);
            self.win_v = Vec::with_capacity((t + 64) * dv);
            self.win_k.extend_from_slice(ks);
            self.win_v.extend_from_slice(vs);
            self.win_len = t;
            self.scr_s = Vec::with_capacity(t + 64);
            return;
        }

        // Sink tokens: positions 0..sink become permanent exact keys.
        // They bypass the ring window entirely, so the delayed-insertion
        // path below can never move them into the far accumulators.
        self.sink_len = self.sink; // skeleton mode guarantees t > sink
        self.sink_k = ks[..self.sink * d].to_vec();
        self.sink_v = vs[..self.sink * dv].to_vec();

        // Landmarks: contiguous segment means of the prompt (per-head).
        // The integer split (i·t)/m matches the reference probe; the
        // clamp keeps tiny prompts from producing duplicate landmarks.
        let m_eff = (t / 8).clamp(4, self.m);
        self.m_eff = m_eff;
        let q_tilde64 = seg_means(qs, t, d, m_eff);
        let k_tilde64 = seg_means(ks, t, d, m_eff);
        self.q_tilde = q_tilde64.iter().map(|&x| x as f32).collect();
        self.k_tilde = k_tilde64.iter().map(|&x| x as f32).collect();

        // Au and its regularized pseudo-inverse in f64 — one-off m×m
        // work at prefill only; the hot path stays f32.
        let mut au = vec![0.0f64; m_eff * m_eff];
        for i in 0..m_eff {
            for j in 0..m_eff {
                let mut s = 0.0f64;
                for c in 0..d {
                    s += q_tilde64[i * d + c] * k_tilde64[j * d + c];
                }
                au[i * m_eff + j] = (s * self.scale as f64).exp();
            }
        }
        let mu64 = ridge_pinv(&au, m_eff);
        self.mu = mu64.iter().map(|&x| x as f32).collect();

        self.t_hat = vec![0.0; m_eff * dv];
        self.z_hat = vec![0.0; m_eff];
        self.m_max = vec![f32::NEG_INFINITY; m_eff];
        self.win_k = vec![0.0; self.w * d];
        self.win_v = vec![0.0; self.w * dv];
        self.scr_s = vec![0.0; self.sink + self.w];
        self.scr_fh = vec![0.0; m_eff];
        self.scr_u = vec![0.0; m_eff];
        self.scr_l = vec![0.0; m_eff];

        // Replay the post-sink prompt through step() state semantics:
        // each key enters the window, evicting the (j-w)-th into the
        // far field.
        for j in self.sink..t {
            self.advance_window(&ks[j * d..(j + 1) * d], &vs[j * dv..(j + 1) * dv]);
        }
    }

    /// One decode step.  Inserts (k, v), evicting the oldest window key
    /// into the far accumulators, then writes attention output for `q`
    /// into `out` (`[dv]`).
    pub fn step(&mut self, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) {
        let (d, dv) = (self.d, self.dv);
        assert!(d > 0, "prefill() must run before step()");
        assert_eq!(q.len(), d);
        assert_eq!(k.len(), d);
        assert_eq!(v.len(), dv);
        assert_eq!(out.len(), dv);
        // The current token is part of its own near window (t-j = 0),
        // so insertion happens BEFORE the output is computed.
        self.advance_window(k, v);

        // Near field: exact logits over sinks + window, one shared
        // shift.  Sinks are permanent exact keys (near mask §5b:
        // t-j < w OR j < sink); sink_len = 0 in exact-only mode.
        let ns = self.sink_len;
        let n = ns + self.win_len;
        self.scr_s.resize(n, 0.0);
        let mut c = f32::NEG_INFINITY;
        for s in 0..ns {
            let lg = dot(q, &self.sink_k[s * d..(s + 1) * d]) * self.scale;
            self.scr_s[s] = lg;
            c = c.max(lg);
        }
        // Window scores are the decode hot loop — NEON dot (same
        // products, regrouped sums; parity-gated by the golden tests).
        for s in 0..self.win_len {
            let lg =
                crate::attention::dot_f32(q, &self.win_k[s * d..(s + 1) * d]) * self.scale;
            self.scr_s[ns + s] = lg;
            c = c.max(lg);
        }

        // Far field: shifted skeleton (spec §3).  All exp arguments are
        // ≤ 0 relative to the joint shift c_all, so nothing overflows.
        let mut far_den = 0.0f32;
        let mut c_all = c;
        let mut have_far = false;
        if self.far_len > 0 {
            // Per-token row shift f over landmark scores.
            let mut f = f32::NEG_INFINITY;
            for a in 0..self.m_eff {
                let s = crate::attention::dot_f32(q, &self.k_tilde[a * d..(a + 1) * d])
                    * self.scale;
                self.scr_fh[a] = s;
                f = f.max(s);
            }
            for a in 0..self.m_eff {
                self.scr_fh[a] = (self.scr_fh[a] - f).exp();
            }
            // u = (F·e^{-f}) · M — the landmark mixing row.
            for b in 0..self.m_eff {
                let mut s = 0.0f32;
                for a in 0..self.m_eff {
                    s += self.scr_fh[a] * self.mu[a * self.m_eff + b];
                }
                self.scr_u[b] = s;
            }
            // Joint scale: the far term b carries e^{f + m_max[b]}, the
            // near term e^{c}; take the max so every factor is ≤ 1.
            for b in 0..self.m_eff {
                c_all = c_all.max(f + self.m_max[b]);
            }
            for b in 0..self.m_eff {
                let g = self.scr_u[b] * (f + self.m_max[b] - c_all).exp();
                self.scr_u[b] = g;
                far_den += g * self.z_hat[b];
            }
            // Guard: M is indefinite, so the aggregated far mass can go
            // negative; a negative denominator means the skeleton
            // estimate is unusable for this row — drop the far field.
            // (The matrix probe clamps per-(t,j); per-key weights no
            // longer exist after aggregation, so the guard is coarser.)
            if far_den >= 0.0 {
                have_far = true;
            } else {
                far_den = 0.0;
            }
        }

        for o in out.iter_mut() {
            *o = 0.0;
        }
        if have_far {
            for b in 0..self.m_eff {
                crate::attention::axpy_f32(
                    out,
                    &self.t_hat[b * dv..(b + 1) * dv],
                    self.scr_u[b],
                );
            }
        }
        let mut den = far_den;
        for s in 0..n {
            let p = (self.scr_s[s] - c_all).exp();
            den += p;
            // scr_s rows 0..ns are sinks, the rest are window entries.
            let vv = if s < ns {
                &self.sink_v[s * dv..(s + 1) * dv]
            } else {
                &self.win_v[(s - ns) * dv..(s - ns + 1) * dv]
            };
            crate::attention::axpy_f32(out, vv, p);
        }
        let den = den.max(DEN_EPS);
        for o in out.iter_mut() {
            *o /= den;
        }
    }

    /// Heap bytes held by this state (window + sinks + skeleton +
    /// scratch) — feeds the honest "KV+state" memory line, same
    /// discipline as counting `linear_state` for the linear core.
    pub fn memory_bytes(&self) -> usize {
        (self.win_k.len()
            + self.win_v.len()
            + self.sink_k.len()
            + self.sink_v.len()
            + self.t_hat.len()
            + self.z_hat.len()
            + self.m_max.len()
            + self.q_tilde.len()
            + self.k_tilde.len()
            + self.mu.len()
            + self.scr_s.len()
            + self.scr_fh.len()
            + self.scr_u.len()
            + self.scr_l.len())
            * std::mem::size_of::<f32>()
    }

    /// Push (k, v) into the window; in skeleton mode a full ring first
    /// evicts its oldest key into the far accumulators (delayed
    /// insertion — the key leaves the exact window at this very step).
    fn advance_window(&mut self, k: &[f32], v: &[f32]) {
        let (d, dv) = (self.d, self.dv);
        if !self.exact_only && self.win_len == self.w {
            let slot = self.win_head;
            self.far_insert(slot);
            self.win_k[slot * d..(slot + 1) * d].copy_from_slice(k);
            self.win_v[slot * dv..(slot + 1) * dv].copy_from_slice(v);
            self.win_head = (self.win_head + 1) % self.w;
        } else if self.exact_only {
            self.win_k.extend_from_slice(k);
            self.win_v.extend_from_slice(v);
            self.win_len += 1;
        } else {
            self.win_k[self.win_len * d..(self.win_len + 1) * d].copy_from_slice(k);
            self.win_v[self.win_len * dv..(self.win_len + 1) * dv].copy_from_slice(v);
            self.win_len += 1;
        }
    }

    /// Absorb the window slot into the far accumulators with the
    /// per-landmark flash shift: T̂[i]/Ẑ[i] live at scale e^{-m_max[i]};
    /// when a new logit raises the max, existing mass is rescaled by
    /// e^{old-new} (exactly 0 on first insertion, since m_max = -inf).
    fn far_insert(&mut self, slot: usize) {
        let (d, dv) = (self.d, self.dv);
        // Runs once per evicted key per head — NEON dot/axpy like the
        // decode loop (same products, regrouped sums).
        for i in 0..self.m_eff {
            self.scr_l[i] = crate::attention::dot_f32(
                &self.q_tilde[i * d..(i + 1) * d],
                &self.win_k[slot * d..(slot + 1) * d],
            ) * self.scale;
        }
        for i in 0..self.m_eff {
            let l = self.scr_l[i];
            if l > self.m_max[i] {
                let r = (self.m_max[i] - l).exp();
                self.z_hat[i] *= r;
                for e in self.t_hat[i * dv..(i + 1) * dv].iter_mut() {
                    *e *= r;
                }
                self.m_max[i] = l;
            }
            let e = (l - self.m_max[i]).exp();
            self.z_hat[i] += e;
            crate::attention::axpy_f32(
                &mut self.t_hat[i * dv..(i + 1) * dv],
                &self.win_v[slot * dv..(slot + 1) * dv],
                e,
            );
        }
        self.far_len += 1;
    }
}

/// Contiguous segment means (the Nyströmformer landmark recipe), f64
/// accumulation.  The split (i·t)/m matches the Python reference.
fn seg_means(xs: &[f32], t: usize, d: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * d];
    for i in 0..m {
        let lo = i * t / m;
        let hi = (i + 1) * t / m;
        for j in lo..hi {
            for c in 0..d {
                out[i * d + c] += xs[j * d + c] as f64;
            }
        }
        let inv = 1.0 / (hi - lo) as f64;
        for c in 0..d {
            out[i * d + c] *= inv;
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// Regularized pseudo-inverse M = (AᵀA + λI)⁻¹ Aᵀ of a square matrix,
/// λ = RIDGE_REL·mean(diag(AᵀA)), solved via Cholesky.  f64 internal —
/// this runs once per prefill on an m×m matrix (m ≤ 32).  If Cholesky
/// fails (Au numerically singular despite the m_eff clamp), λ grows
/// tenfold — the jitter fallback of the reference probe.
fn ridge_pinv(a: &[f64], n: usize) -> Vec<f64> {
    let mut ata = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = s;
        }
    }
    let mean_diag: f64 = (0..n).map(|i| ata[i * n + i]).sum::<f64>() / n as f64;
    let mut lambda = RIDGE_REL * mean_diag.max(f64::MIN_POSITIVE);
    for _ in 0..12 {
        let mut g = ata.clone();
        for i in 0..n {
            g[i * n + i] += lambda;
        }
        if let Some(l) = cholesky(&mut g, n) {
            // Solve G·M = Aᵀ column by column; column j of Aᵀ is row j
            // of A.
            let mut m_out = vec![0.0f64; n * n];
            let mut x = vec![0.0f64; n];
            for j in 0..n {
                let rhs = &a[j * n..(j + 1) * n];
                // Forward: L·y = rhs.
                for i in 0..n {
                    let mut s = rhs[i];
                    for k in 0..i {
                        s -= l[i * n + k] * x[k];
                    }
                    x[i] = s / l[i * n + i];
                }
                // Backward: Lᵀ·x = y.
                for i in (0..n).rev() {
                    let mut s = x[i];
                    for k in i + 1..n {
                        s -= l[k * n + i] * x[k];
                    }
                    x[i] = s / l[i * n + i];
                }
                for i in 0..n {
                    m_out[i * n + j] = x[i];
                }
            }
            return m_out;
        }
        lambda *= 10.0;
    }
    // Unreachable in practice: λ eventually dominates the diagonal.
    // Degrade to a scaled identity rather than poison the output.
    let mut fallback = vec![0.0f64; n * n];
    for i in 0..n {
        fallback[i * n + i] = 1.0 / mean_diag.max(f64::MIN_POSITIVE);
    }
    fallback
}

// ── Runtime configuration (v1: runtime-level, NOT a format change) ──
//
// A layer set + {m, w, sink}, resolved in priority order:
//   1. CLI flag (`--o1` on run/serve/bench) — explicit user intent;
//   2. env `CMF_O1` (all | deepN | i,j,k | off) with CMF_O1_M /
//      CMF_O1_WINDOW / CMF_O1_SINK parameter overrides;
//   3. converter hint in the header JSON (`provenance.o1_attn`,
//      written by `cortiq convert --o1`) — additive metadata, the
//      binary envelope is untouched.

/// Validated defaults (spec: m=32, W=128, sink=4; sink ablation ×2.39).
pub const O1_DEFAULT_M: usize = 32;
pub const O1_DEFAULT_W: usize = 128;
pub const O1_DEFAULT_SINK: usize = 4;

/// Which layers run the O(1) kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum O1Layers {
    All,
    /// The N deepest layers (deep-N ladder of the price map; the
    /// early stack is the most sink-dependent, depth converts best).
    Deep(usize),
    /// Explicit layer indices.
    List(Vec<usize>),
}

/// Per-model O(1)-attention setting.
#[derive(Clone, Debug)]
pub struct O1Cfg {
    pub layers: O1Layers,
    /// Landmark budget (≥ 4; m=64 measured WORSE — collinear segment
    /// means poison the pinv, so don't "help" by raising it).
    pub m: usize,
    /// Exact-window width — the main quality lever.
    pub w: usize,
    /// Permanent exact sink keys (StreamingLLM discipline, spec §5b).
    pub sink: usize,
}

/// Three-state env reading: unset falls through to the header hint,
/// `off`/`0` force-disables even a header hint (the escape hatch).
pub enum O1Env {
    Unset,
    Off,
    On(O1Cfg),
}

impl O1Cfg {
    /// Parse a layer spec: `all` | `deepN` | `i,j,k`. None = not a spec
    /// (also used for `off`/`0`/empty).
    pub fn parse_layers(spec: &str) -> Option<O1Layers> {
        let s = spec.trim();
        match s {
            "" | "off" | "0" | "none" => None,
            "all" => Some(O1Layers::All),
            _ => {
                if let Some(n) = s.strip_prefix("deep") {
                    return n.parse::<usize>().ok().filter(|&n| n > 0).map(O1Layers::Deep);
                }
                let idx: Result<Vec<usize>, _> =
                    s.split(',').map(|p| p.trim().parse::<usize>()).collect();
                idx.ok().filter(|v| !v.is_empty()).map(O1Layers::List)
            }
        }
    }

    /// Build from an explicit spec (CLI path). None = `off` or malformed.
    /// Explicit m/w/sink beat env overrides beat validated defaults.
    pub fn from_spec(
        spec: &str,
        m: Option<usize>,
        w: Option<usize>,
        sink: Option<usize>,
    ) -> Option<O1Cfg> {
        let layers = Self::parse_layers(spec)?;
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        Some(O1Cfg {
            layers,
            // NystromState asserts m ≥ 4 and w ≥ 1 — clamp rather than
            // panic deep in the first prefill.
            m: m.or_else(|| env("CMF_O1_M")).unwrap_or(O1_DEFAULT_M).max(4),
            w: w.or_else(|| env("CMF_O1_WINDOW")).unwrap_or(O1_DEFAULT_W).max(1),
            sink: sink.or_else(|| env("CMF_O1_SINK")).unwrap_or(O1_DEFAULT_SINK),
        })
    }

    /// Converter hint from the header JSON: `{"layers": "all"|[i,…],
    /// "m": …, "w": …, "sink": …}`. Env parameter overrides still apply
    /// (the operator's knob wins over the file's suggestion).
    pub fn from_json(v: &serde_json::Value) -> Option<O1Cfg> {
        let layers = match v.get("layers") {
            Some(serde_json::Value::String(s)) => Self::parse_layers(s)?,
            Some(serde_json::Value::Array(a)) => O1Layers::List(
                a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect(),
            ),
            _ => return None,
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize);
        let env = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
        Some(O1Cfg {
            layers,
            m: env("CMF_O1_M").or_else(|| f("m")).unwrap_or(O1_DEFAULT_M).max(4),
            w: env("CMF_O1_WINDOW").or_else(|| f("w")).unwrap_or(O1_DEFAULT_W).max(1),
            sink: env("CMF_O1_SINK").or_else(|| f("sink")).unwrap_or(O1_DEFAULT_SINK),
        })
    }

    /// Per-layer flags over `num_layers` (indices past the end are
    /// silently dropped; the pipeline additionally filters non-Full
    /// layers — a linear layer keeps its own operator).
    pub fn layer_flags(&self, num_layers: usize) -> Vec<bool> {
        let mut flags = vec![false; num_layers];
        match &self.layers {
            O1Layers::All => flags.iter_mut().for_each(|f| *f = true),
            O1Layers::Deep(n) => {
                for f in flags.iter_mut().skip(num_layers.saturating_sub(*n)) {
                    *f = true;
                }
            }
            O1Layers::List(idx) => {
                for &i in idx {
                    if i < num_layers {
                        flags[i] = true;
                    }
                }
            }
        }
        flags
    }
}

/// Read `CMF_O1` (+ parameter overrides) — the embedding-friendly path
/// for hosts that don't go through the CLI flags.
pub fn o1_from_env() -> O1Env {
    match std::env::var("CMF_O1") {
        Err(_) => O1Env::Unset,
        Ok(s) => match O1Cfg::from_spec(&s, None, None, None) {
            Some(cfg) => O1Env::On(cfg),
            None => O1Env::Off,
        },
    }
}

/// In-place lower Cholesky of an SPD matrix; None if a pivot fails.
fn cholesky(g: &mut [f64], n: usize) -> Option<&[f64]> {
    for i in 0..n {
        for j in 0..=i {
            let mut s = g[i * n + j];
            for k in 0..j {
                s -= g[i * n + k] * g[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                g[i * n + i] = s.sqrt();
            } else {
                g[i * n + j] = s / g[j * n + j];
            }
        }
    }
    Some(g)
}
