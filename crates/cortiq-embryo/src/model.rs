//! Embryo genome: configuration, the flat parameter arena, and the fixed
//! training graph (forward + hand-rolled backward) on our Metal kernels.
//! See docs/NATIVE_MODEL_TECH.ru.md §3.
//!
//! One layer:
//!   x ─► RMSNorm ─► MIXER (hybrid_k, or the softmax anchor every 8th) ─► +res
//!     ─► RMSNorm ─► FFN (SwiGLU; the shared expert — routed experts are the
//!        growth slots, next) ─► +res
//! Head: tied embedding, full softmax (the hierarchical 128×256 head is next).
//!
//! No autograd: the graph is fixed, every block's backward is written out
//! (llm.c style, same discipline as the runtime's `fcd_ops`).

/// Embryo-0 as born (§3.2). All matrix dims are multiples of 64 so the
/// GEMM tile contract holds without edge paths.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbryoCfg {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// every `anchor_every`-th layer is a softmax anchor (o1-ready)
    pub anchor_every: usize,
    // hybrid_k mixer
    pub heads: usize,
    pub nphase: usize,
    pub dv: usize,
    /// decay horizons: log grid [h_min, h_max] over the phase pairs
    pub horizon_min: f64,
    pub horizon_max: f64,
    /// κ = σ(W_κ x + kappa_bias)
    pub kappa_bias: f32,
    // anchor (GQA softmax)
    pub anchor_q_heads: usize,
    pub anchor_kv_heads: usize,
    pub anchor_hd: usize,
    pub rope_base: f32,
    // experts
    pub experts: usize,
    pub inter: usize,
    // hierarchical head
    pub head_clusters: usize,
    pub mtp_heads: usize,
    pub seq: usize,
    pub norm_eps: f32,
    /// decays γ = exp(−exp(A_log)) are trained (A_log per head·feature in the
    /// arena, initialised from the horizon grid) instead of fixed
    #[serde(default)]
    pub learn_decay: bool,
    /// Causal depthwise conv over the mixer's normed input, k taps (0 = off).
    /// The per-token-diagnosis fix: the hybrid's gap to the softmax twin is
    /// FLAT by position and biggest on repeats — short-range mixing, which a
    /// 4-tap conv supplies for ~h·k parameters and O(1) per token. Identity
    /// init (last tap 1): the layer starts as a pass-through.
    #[serde(default)]
    pub conv_k: usize,
    /// Enable the appended one-head GDN correction lane on each hybrid_k
    /// mixer.  The lane is an optional checkpoint tail; when disabled the
    /// original arena and execution graph are bit-identical.
    #[serde(default)]
    pub gdn_lane: bool,
    /// Enable the layer-4 additive GQA donor lane.  This is an append-only
    /// experiment: the original hybrid mixer remains active and the new
    /// output projection is initialized to exact zero.
    #[serde(default)]
    pub gqa_lane: bool,
    /// Replace the legacy hybrid_k accumulation with the parameter-neutral
    /// normalized in-place Phase-Delta overwrite rule.  This discriminator
    /// carries no arena state or trainable tensors.
    #[serde(default)]
    pub phase_delta: bool,
    /// Optional zero-based hybrid layer on which to run Phase-Delta.  When
    /// present this selector is itself enabling and takes precedence over the
    /// legacy all-layer `phase_delta` switch.  It is deliberately serialized
    /// so a checkpoint cannot silently change operator semantics on resume.
    #[serde(default)]
    pub phase_delta_layer: Option<usize>,
    /// Optional parameter-neutral Phase-Delta placement on exactly two
    /// hybrid layers.  The entries are zero-based layer indices; anchors,
    /// duplicates, and out-of-range indices are rejected by
    /// [`EmbryoCfg::validate_phase_delta_layer`].  This is deliberately
    /// serialized so checkpoint resume cannot silently change operator
    /// semantics.  The current dual experiment uses `[3, 6]`.
    #[serde(default)]
    pub phase_delta_layers: Option<Vec<usize>>,
    /// Apply a causal four-position mean to each expert's reconstruction
    /// score before the custom argmin router.  The raw winning resonance is
    /// retained for diagnostics and descriptor balancing; this flag changes
    /// only expert selection and is default-off for legacy identity.
    #[serde(default)]
    pub router_smooth_k4: bool,
    /// Optional fixed reconstruction-error margin for the conditional top-2
    /// router.  A row whose runner-up score is less than this margin above
    /// the winner is marked for a deterministic 50/50 routed-residual blend.
    /// `None` is the safe legacy default. When enabled, the trainer allocates
    /// a bounded second routed stream and blends only routed residuals.
    #[serde(default)]
    pub router_top2_margin: Option<f32>,
    /// Bounded anchor `swa_sink_v1` (docs/EMBRYO_BOUNDED_ANCHOR.md): the
    /// served exact-window width in keys, INCLUDING the current token.
    /// 0 = the legacy full-causal anchor (old checkpoints deserialize to 0
    /// and stay bit-identical; the field is skipped on serialization so a
    /// legacy config still serializes byte-for-byte as before).
    #[serde(default, skip_serializing_if = "usize_is_zero")]
    pub anchor_window: usize,
    /// Trained NoPE sink vectors per KV head (`sink_k`/`sink_v` weights,
    /// `[kvh, S, hd]`); requires `anchor_window > 0`. `S + W ≤ 160`.
    #[serde(default, skip_serializing_if = "usize_is_zero")]
    pub anchor_sink: usize,
    /// SWAX-style stochastic training window: every optimizer step samples
    /// `W_t` from this list (deterministically from the step index); empty =
    /// the fixed `anchor_window`. Entries must lie in `1..=anchor_window`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchor_train_windows: Vec<usize>,
    /// Explicit zero-based anchor schedule. `None` = the legacy cadence
    /// `(l + 1) % anchor_every == 0`. Serialized so a checkpoint cannot
    /// silently change which layers are anchors on resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_layers: Option<Vec<usize>>,
    /// The mixer of every non-anchor layer (plan S7, variant B): the legacy
    /// `hybrid_k` phase core or the gated delta-rule core `gated_delta_net`
    /// (per-token learned decay, L2 keys, gated RMSNorm + SiLU output gate —
    /// exactly the runtime's `linear_core::gdn_step`). Skipped on
    /// serialization when legacy so old configs stay byte-identical.
    #[serde(default, skip_serializing_if = "Mixer::is_hybrid_k")]
    pub mixer: Mixer,
    /// GDN heads (`nk == nv`; default 4). Only read when `mixer == Gdn`.
    #[serde(default = "default_gdn_heads", skip_serializing_if = "is_default_gdn_heads")]
    pub gdn_heads: usize,
    /// GDN key head dim (default 128, ≤ 128, multiple of 4).
    #[serde(default = "default_gdn_dim", skip_serializing_if = "is_default_gdn_dim")]
    pub gdn_dk: usize,
    /// GDN value head dim (default 128, ≤ 128, multiple of 4).
    #[serde(default = "default_gdn_dim", skip_serializing_if = "is_default_gdn_dim")]
    pub gdn_dv: usize,
    /// Control arm (plan S7 (d)): pin β ≡ 1 in the delta rule (the erase
    /// term always fully overwrites; `in_proj_b` receives no gradient). A
    /// diagnostic-only switch: such a genome cannot be exported faithfully
    /// (the runtime computes β = σ(b)), so export refuses it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gdn_beta_one: bool,
}

fn usize_is_zero(x: &usize) -> bool {
    *x == 0
}

/// Which operator the non-anchor layers run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mixer {
    /// The legacy phase core (`vmf_phase` in the runtime).
    #[default]
    HybridK,
    /// Gated DeltaNet (`gated_delta_net` in the runtime).
    Gdn,
}

impl Mixer {
    pub fn is_hybrid_k(&self) -> bool {
        *self == Mixer::HybridK
    }
}

fn default_gdn_heads() -> usize {
    4
}
fn is_default_gdn_heads(x: &usize) -> bool {
    *x == 4
}
fn default_gdn_dim() -> usize {
    128
}
fn is_default_gdn_dim(x: &usize) -> bool {
    *x == 128
}

/// Rows of the padded `in_proj_a` / `in_proj_b` control projections in the
/// arena (only `gdn_heads` rows are live; the pad rows are zero and never
/// trained — export slices them off).
pub const GDN_AB_PAD: usize = 64;
/// Causal depthwise conv taps of the GDN core (`conv1d.weight [c_dim, 1, 4]`).
pub const GDN_CONV_K: usize = 4;
/// Largest head dim the token-scan kernels serve (one lane per state row /
/// output column in a 128-lane group).
pub const GDN_MAX_DIM: usize = 128;

/// Sink columns of the trainer's score rows are padded to one GEMM tile so
/// the sink block (`q̂·sink_kᵀ`, `P_sink·sink_v`, and their backward) runs on
/// the ordinary 64×64×32 tile GEMM: score rows are `[SINK_PAD + T]` wide
/// when `anchor_sink > 0`, columns `S..SINK_PAD` are masked (P = 0) and the
/// causal/band block starts at column `SINK_PAD`. With `anchor_sink == 0`
/// the rows are `[T]` wide exactly as in the legacy layout.
pub const SINK_PAD: usize = 64;
/// Format ceiling on `sink + window` (the runtime's bounded-attend scratch):
/// the value of `cortiq_core::types::AnchorCoreConfig::MAX_SINK_PLUS_WINDOW`
/// (spelled out so the trainer crate also builds against a core without
/// the bounded types; `tests/anchor_bounded.rs` pins the two together).
pub const MAX_SINK_PLUS_WINDOW: usize = 160;

impl EmbryoCfg {
    pub fn embryo0() -> Self {
        EmbryoCfg {
            vocab: 32768,
            hidden: 384,
            layers: 8,
            anchor_every: 8,
            conv_k: 0,
            heads: 8,
            nphase: 32,
            dv: 128,
            horizon_min: 8.0,
            horizon_max: 2048.0,
            kappa_bias: 2.0,
            anchor_q_heads: 8,
            anchor_kv_heads: 2,
            anchor_hd: 128,
            rope_base: 10000.0,
            experts: 4,
            inter: 768,
            head_clusters: 128,
            mtp_heads: 2,
            seq: 1024,
            norm_eps: 1e-6,
            learn_decay: false,
            gdn_lane: false,
            gqa_lane: false,
            phase_delta: false,
            phase_delta_layer: None,
            phase_delta_layers: None,
            router_smooth_k4: false,
            router_top2_margin: None,
            anchor_window: 0,
            anchor_sink: 0,
            anchor_train_windows: Vec::new(),
            anchor_layers: None,
            mixer: Mixer::HybridK,
            gdn_heads: 4,
            gdn_dk: 128,
            gdn_dv: 128,
            gdn_beta_one: false,
        }
    }
    /// A tiny genome for smoke tests and gradchecks (same shape family,
    /// every dim still a multiple of the GEMM tile).
    pub fn tiny() -> Self {
        EmbryoCfg {
            vocab: 4096,
            hidden: 64,
            layers: 2,
            anchor_every: 2,
            conv_k: 0,
            heads: 2,
            nphase: 32,
            dv: 64,
            horizon_min: 4.0,
            horizon_max: 128.0,
            kappa_bias: 2.0,
            anchor_q_heads: 2,
            anchor_kv_heads: 1,
            anchor_hd: 64,
            rope_base: 10000.0,
            experts: 0,
            inter: 128,
            head_clusters: 64,
            mtp_heads: 0,
            seq: 64,
            norm_eps: 1e-6,
            learn_decay: false,
            gdn_lane: false,
            gqa_lane: false,
            phase_delta: false,
            phase_delta_layer: None,
            phase_delta_layers: None,
            router_smooth_k4: false,
            router_top2_margin: None,
            anchor_window: 0,
            anchor_sink: 0,
            anchor_train_windows: Vec::new(),
            anchor_layers: None,
            mixer: Mixer::HybridK,
            gdn_heads: 4,
            gdn_dk: 128,
            gdn_dv: 128,
            gdn_beta_one: false,
        }
    }

    /// Whether the parameter-neutral Phase-Delta scratch/operator is needed
    /// for any layer in this configuration.  A selected layer enables the
    /// operator even when the historical boolean is false; this makes the
    /// explicit single/dual CLI selectors unambiguous.
    pub fn phase_delta_active(&self) -> bool {
        self.phase_delta || self.phase_delta_layer.is_some() || self.phase_delta_layers.is_some()
    }

    /// Resolve the semantic Phase-Delta placement that must be written into
    /// an exported operator record.  Unlike `validate_phase_delta_layer`, this
    /// is a fallible preflight for user-facing export: malformed metadata must
    /// return an error before the CMF writer creates/truncates its output.
    pub fn phase_delta_layers_for_export(&self) -> Result<Vec<usize>, String> {
        if self.phase_delta_layer.is_some() && self.phase_delta_layers.is_some() {
            return Err("phase-delta single and dual selectors cannot be combined".into());
        }
        let mut selected = if let Some(layers) = &self.phase_delta_layers {
            layers.clone()
        } else if let Some(layer) = self.phase_delta_layer {
            vec![layer]
        } else if self.phase_delta {
            (0..self.layers).filter(|&l| !self.is_anchor(l)).collect()
        } else {
            Vec::new()
        };
        if !self.phase_delta_active() {
            return Ok(selected);
        }
        if selected.is_empty() {
            return Err("Phase-Delta is enabled but its selector is empty".into());
        }
        for &layer in &selected {
            if layer >= self.layers {
                return Err(format!(
                    "phase-delta layer {layer} out of range for {} layers",
                    self.layers
                ));
            }
            if self.is_anchor(layer) {
                return Err(format!("phase-delta layer {layer} is an anchor layer"));
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(selected.len());
        if let Some(&duplicate) = selected.iter().find(|&&layer| !seen.insert(layer)) {
            return Err(format!(
                "phase-delta selector contains duplicate layer {duplicate}"
            ));
        }
        // The record is a set, not an execution schedule.  Canonical ordering
        // keeps equivalent configs byte-stable and makes malformed-file
        // diagnostics deterministic.
        selected.sort_unstable();
        Ok(selected)
    }

    /// Whether layer `layer` uses the Phase-Delta recurrence.  An explicit
    /// selector wins over the historical all-layer research switch, and
    /// anchors always remain on their attention path.  A dual selector takes
    /// precedence over the legacy singleton (the two cannot be combined).
    pub fn phase_delta_for_layer(&self, layer: usize) -> bool {
        if self.is_anchor(layer) {
            return false;
        }
        if let Some(selected) = &self.phase_delta_layers {
            return selected.contains(&layer);
        }
        match self.phase_delta_layer {
            Some(selected) => selected == layer,
            None => self.phase_delta,
        }
    }

    /// Reject malformed explicit selectors before allocating a model or
    /// constructing a checkpoint layout.  Layer indices are zero-based and
    /// must name a hybrid (non-anchor) layer.
    pub fn validate_phase_delta_layer(&self) {
        assert!(
            !(self.phase_delta_layer.is_some() && self.phase_delta_layers.is_some()),
            "phase-delta single and dual selectors cannot be combined"
        );
        if let Some(layer) = self.phase_delta_layer {
            assert!(
                layer < self.layers,
                "phase-delta layer {layer} out of range for {} layers",
                self.layers
            );
            assert!(
                !self.is_anchor(layer),
                "phase-delta layer {layer} is an anchor layer"
            );
        }
        if let Some(layers) = &self.phase_delta_layers {
            assert_eq!(
                layers.len(),
                2,
                "phase-delta dual selector requires exactly two layers"
            );
            assert_ne!(
                layers[0], layers[1],
                "phase-delta dual selector contains duplicate layer {}",
                layers[0]
            );
            for &layer in layers {
                assert!(
                    layer < self.layers,
                    "phase-delta layer {layer} out of range for {} layers",
                    self.layers
                );
                assert!(
                    !self.is_anchor(layer),
                    "phase-delta layer {layer} is an anchor layer"
                );
            }
        }
    }
    pub fn is_anchor(&self, layer: usize) -> bool {
        match &self.anchor_layers {
            Some(sel) => sel.contains(&layer),
            None => (layer + 1) % self.anchor_every == 0,
        }
    }

    /// Whether the anchors run the bounded operator (band + sinks) instead
    /// of the legacy full-causal softmax.
    pub fn anchor_bounded(&self) -> bool {
        self.anchor_window > 0
    }

    /// Width of the padded sink block in every score row (0 without sinks).
    pub fn sink_pad(&self) -> usize {
        if self.anchor_bounded() && self.anchor_sink > 0 {
            SINK_PAD
        } else {
            0
        }
    }

    /// Score-row length of an anchor for sequence length `t`.
    pub fn anchor_ld(&self, t: usize) -> usize {
        self.sink_pad() + t
    }

    /// The training window of optimizer step `step`: a deterministic draw
    /// from `anchor_train_windows` (splitmix over the step index, so a
    /// resumed birth replays the same schedule), or the fixed window.
    pub fn anchor_window_at_step(&self, step: u64) -> usize {
        if self.anchor_train_windows.is_empty() {
            return self.anchor_window;
        }
        let mut z = step
            .wrapping_add(0x5357_4158_0000_0000)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        self.anchor_train_windows[(z % self.anchor_train_windows.len() as u64) as usize]
    }

    /// Fallible validation of the bounded-anchor fields (export preflight
    /// and CLI); `validate_anchor` is the asserting twin used by `Layout`.
    pub fn check_anchor(&self) -> Result<(), String> {
        if let Some(sel) = &self.anchor_layers {
            let mut seen = std::collections::HashSet::with_capacity(sel.len());
            for &l in sel {
                if l >= self.layers {
                    return Err(format!(
                        "anchor layer {l} out of range for {} layers",
                        self.layers
                    ));
                }
                if !seen.insert(l) {
                    return Err(format!("anchor_layers contains duplicate layer {l}"));
                }
            }
        }
        if self.anchor_window == 0 {
            if self.anchor_sink > 0 {
                return Err("anchor_sink > 0 requires anchor_window > 0 (sinks are part of the bounded operator)".into());
            }
            if !self.anchor_train_windows.is_empty() {
                return Err("anchor_train_windows requires anchor_window > 0".into());
            }
            return Ok(());
        }
        if self.anchor_sink + self.anchor_window > MAX_SINK_PLUS_WINDOW {
            return Err(format!(
                "anchor sink + window = {} exceeds the kernel ceiling {}",
                self.anchor_sink + self.anchor_window,
                MAX_SINK_PLUS_WINDOW
            ));
        }
        if self.anchor_sink > SINK_PAD {
            return Err(format!(
                "anchor_sink = {} exceeds the trainer's sink tile {}",
                self.anchor_sink, SINK_PAD
            ));
        }
        if let Some(&w) = self
            .anchor_train_windows
            .iter()
            .find(|&&w| w == 0 || w > self.anchor_window)
        {
            return Err(format!(
                "anchor_train_windows entry {w} is outside 1..={}",
                self.anchor_window
            ));
        }
        Ok(())
    }

    /// Asserting twin of [`EmbryoCfg::check_anchor`].
    pub fn validate_anchor(&self) {
        if let Err(e) = self.check_anchor() {
            panic!("{e}");
        }
    }

    /// Whether the non-anchor layers run the gated delta-rule core.
    pub fn is_gdn_mixer(&self) -> bool {
        self.mixer == Mixer::Gdn
    }
    /// GDN projection width `2·nk·dk + nv·dv` (nk == nv).
    pub fn gdn_c_dim(&self) -> usize {
        2 * self.gdn_heads * self.gdn_dk + self.gdn_heads * self.gdn_dv
    }
    /// Length of the padded per-head scalar vectors (`A_log`, `dt_bias`) in
    /// the arena: `gdn_heads` rounded up to 4 floats so every tensor after
    /// them keeps the 4-float GEMM alignment.
    pub fn gdn_head_pad(&self) -> usize {
        self.gdn_heads.div_ceil(4) * 4
    }
    /// Fallible validation of the GDN mixer geometry (export preflight and
    /// CLI); `validate_gdn` is the asserting twin used by `Layout`.
    pub fn check_gdn(&self) -> Result<(), String> {
        if !self.is_gdn_mixer() {
            return Ok(());
        }
        let (nv, dk, dv) = (self.gdn_heads, self.gdn_dk, self.gdn_dv);
        if nv == 0 || nv > GDN_AB_PAD {
            return Err(format!("gdn_heads = {nv} must lie in 1..={GDN_AB_PAD}"));
        }
        if dk == 0 || dk > GDN_MAX_DIM || dk % 4 != 0 || dv == 0 || dv > GDN_MAX_DIM || dv % 4 != 0
        {
            return Err(format!(
                "gdn_dk = {dk}, gdn_dv = {dv} must be multiples of 4 in 4..={GDN_MAX_DIM}"
            ));
        }
        if self.gdn_c_dim() % 64 != 0 || (nv * dv) % 64 != 0 {
            return Err(format!(
                "GDN projection widths must be GEMM tiles: c_dim = {} and nv·dv = {} must be multiples of 64",
                self.gdn_c_dim(),
                nv * dv
            ));
        }
        if self.conv_k > 0 {
            return Err("the GDN mixer carries its own conv1d (k = 4); conv_k must be 0".into());
        }
        if self.gdn_lane || self.gqa_lane {
            return Err("the GDN mixer cannot be combined with the appended gdn_lane / gqa_lane tails".into());
        }
        if self.phase_delta_active() {
            return Err("Phase-Delta selects hybrid_k layers; it has no meaning with the GDN mixer".into());
        }
        if self.learn_decay {
            return Err("learn_decay is a hybrid_k switch; GDN decays are always learned (A_log)".into());
        }
        if self.router_smooth_k4 || self.router_top2_margin.is_some() {
            return Err("router_smooth_k4 / router_top2_margin are not supported with the GDN mixer".into());
        }
        Ok(())
    }
    /// Asserting twin of [`EmbryoCfg::check_gdn`].
    pub fn validate_gdn(&self) {
        if let Err(e) = self.check_gdn() {
            panic!("{e}");
        }
    }

    /// Whether the optional conditional top-2 router is requested.  The
    /// threshold is deliberately an `Option` so legacy configs and binaries
    /// remain bit-identical when the feature is absent.
    pub fn router_top2_enabled(&self) -> bool {
        self.router_top2_margin
            .is_some_and(|margin| margin.is_finite() && margin > 0.0)
    }
    /// κ pre-activation columns the projection GEMM produces (padded to
    /// the GEMM tile; only `heads` are real).
    pub fn kappa_ld(&self) -> usize {
        self.heads.div_ceil(64) * 64
    }
    /// Parameter count (total, active per token) INCLUDING the routed
    /// experts of §3 (which the trainer does not instantiate yet).
    pub fn params(&self) -> (usize, usize) {
        let h = self.hidden;
        let embed = self.vocab * h; // tied lm_head
        let mixer = 2 * (self.heads * self.nphase * h) // thq, thk
            + self.heads * self.dv * h                  // v_proj
            + h * self.heads * self.dv                  // out_proj
            + self.heads * h                            // κ gate
            + h * self.conv_k; // short conv (0 = off)
        let anchor = self.anchor_q_heads * self.anchor_hd * h
            + 2 * self.anchor_kv_heads * self.anchor_hd * h
            + h * self.anchor_q_heads * self.anchor_hd
            // trained sink key/value vectors of the bounded anchor
            + 2 * self.anchor_kv_heads * self.anchor_sink * self.anchor_hd;
        // gated delta-rule core: in_proj_qkv [c_dim,H], in_proj_z [nv·dv,H],
        // in_proj_a/b [nv,H] (live rows only), out_proj [H,nv·dv], conv1d
        // [c_dim·4], A_log/dt_bias [nv], norm [dv]
        let (nv, dk, dv) = (self.gdn_heads, self.gdn_dk, self.gdn_dv);
        let c_dim = 2 * nv * dk + nv * dv;
        let gdn = c_dim * h + nv * dv * h + 2 * nv * h + h * nv * dv + c_dim * GDN_CONV_K + 2 * nv + dv;
        let mixer = if self.is_gdn_mixer() { gdn } else { mixer };
        let ffn_one = 3 * h * self.inter;
        let ffn_total = ffn_one * (self.experts + 1);
        let ffn_active = ffn_one * (1 + self.experts.min(1)); // top-1 + shared
        let norms = 2 * h;
        let mut total = embed + h;
        let mut active = embed + h;
        for l in 0..self.layers {
            let mix = if self.is_anchor(l) { anchor } else { mixer };
            total += mix + ffn_total + norms;
            active += mix + ffn_active + norms;
        }
        if self.gdn_lane {
            // qkvz 256×H, depthwise conv (192×4), padded a/b 64×H,
            // gated norm 64, output H×64, two scalar controls and one
            // residual gain per hybrid layer.
            let lane = 256 * h + 192 * 4 + 64 * h + 64 + h * 64 + 2;
            let n = (0..self.layers).filter(|&l| !self.is_anchor(l)).count();
            total += n * lane + n;
            active += n * lane + n;
        }
        if self.gqa_lane {
            // One layer-4 GQA q/k/v/output tail; output starts at exact zero.
            let lane = self.anchor_q_heads * self.anchor_hd * h
                + 2 * self.anchor_kv_heads * self.anchor_hd * h
                + h * self.anchor_q_heads * self.anchor_hd;
            total += lane;
            active += lane;
        }
        (total, active)
    }
}

// ---------------------------------------------------------------------
// Parameter arena layout
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FfnOffs {
    pub wg: usize, // [I, H]  shared expert
    pub wu: usize, // [I, H]
    pub wd: usize, // [H, I]
    /// routed experts: expert e's gate at `experts + e·3·H·I`, up at +H·I,
    /// down at +2·H·I (usize::MAX when cfg.experts == 0)
    pub experts: usize,
}

/// Flat offsets for the optional appended GDN correction lane.  These
/// records are allocated in a separate tail after the complete legacy arena
/// so an old checkpoint remains an exact prefix of a candidate checkpoint.
#[derive(Clone, Debug)]
pub struct GdnOffs {
    pub qkvz: usize,    // [256,H] fused q/k/v/z training projection
    pub conv: usize,    // [192,4] depthwise causal taps
    pub ab: usize,      // [64,H] padded a/b control projection
    pub norm: usize,    // [64] gated output norm (reserved for parity/export)
    pub wo: usize,      // [H,64] correction output projection
    pub alog: usize,    // scalar decay parameter
    pub dt_bias: usize, // scalar beta/decay bias
    pub gain: usize,    // scalar residual gain (unit gain with zero-output init)
}

/// Flat offsets for the optional layer-4 additive GQA donor lane.
#[derive(Clone, Debug)]
pub struct GqaOffs {
    pub q: usize,  // [qh·hd,H]
    pub k: usize,  // [kvh·hd,H]
    pub v: usize,  // [kvh·hd,H]
    pub wo: usize, // [H,qh·hd] (exact-zero initialization)
}

#[derive(Clone, Debug)]
pub enum LayerOffs {
    Mixer {
        ln1: usize,
        wq: usize,   // [nh·nph, H]
        wk: usize,   // [nh·nph, H]
        wv: usize,   // [nh·dv, H]
        wkap: usize, // [kappa_ld, H] (rows ≥ nh are zero, never trained)
        wo: usize,   // [H, nh·dv]
        /// [nh·2nph] A_log (usize::MAX when the decay grid is fixed)
        alog: usize,
        /// [H·conv_k] short-conv taps (usize::MAX when conv_k = 0)
        conv: usize,
        ln2: usize,
        ffn: FfnOffs,
    },
    Anchor {
        ln1: usize,
        wq: usize, // [qh·hd, H]
        wk: usize, // [kvh·hd, H]
        wv: usize, // [kvh·hd, H]
        wo: usize, // [H, qh·hd]
        /// [kvh·S·hd] trained sink keys (usize::MAX when anchor_sink == 0)
        sink_k: usize,
        /// [kvh·S·hd] trained sink values (usize::MAX when anchor_sink == 0)
        sink_v: usize,
        ln2: usize,
        ffn: FfnOffs,
    },
    /// Gated DeltaNet mixer (`cfg.mixer == Mixer::Gdn`): the runtime's
    /// `gated_delta_net` operator, tensor for tensor. Matrices first, then
    /// the small tensors (conv taps, padded per-head scalars, norm) so every
    /// GEMM operand keeps its 4-float alignment.
    Gdn {
        ln1: usize,
        in_qkv: usize, // [c_dim, H]   c_dim = 2·nv·dk + nv·dv
        in_z: usize,   // [nv·dv, H]
        in_a: usize,   // [GDN_AB_PAD, H] rows ≥ nv are zero, never trained
        in_b: usize,   // [GDN_AB_PAD, H]
        wo: usize,     // [H, nv·dv]
        conv: usize,   // [c_dim, 4] depthwise causal taps, tap 3 = current
        alog: usize,   // [gdn_head_pad] A_log per head (α = exp(−e^{A_log}·softplus(a+dt)))
        dt_bias: usize, // [gdn_head_pad]
        norm: usize,   // [dv] gated-RMSNorm gain shared by the heads
        ln2: usize,
        ffn: FfnOffs,
    },
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub total: usize,
    pub embed: usize, // [V, H]
    pub final_norm: usize,
    /// [head_clusters, H] cluster embeddings of the hierarchical head
    /// (usize::MAX when the head is flat)
    pub head_clusters: usize,
    pub layers: Vec<LayerOffs>,
    /// Optional appended lane offsets, indexed by layer (None for anchors).
    pub gdn: Vec<Option<GdnOffs>>,
    /// Optional layer-4 additive GQA lane (None except layer index 3).
    pub gqa: Vec<Option<GqaOffs>>,
    /// (name, offset, len) of every tensor — checkpoints and the CMF export
    pub names: Vec<(String, usize, usize)>,
}

impl Layout {
    pub fn new(cfg: &EmbryoCfg) -> Layout {
        cfg.validate_anchor();
        cfg.validate_phase_delta_layer();
        cfg.validate_gdn();
        let h = cfg.hidden;
        let mut off = 0usize;
        let mut names = Vec::new();
        let mut take = |name: String, n: usize| -> usize {
            let o = off;
            names.push((name, o, n));
            off += n;
            o
        };
        let embed = take("embed".into(), cfg.vocab * h);
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let ffn = |take: &mut dyn FnMut(String, usize) -> usize| {
                let wg = take(format!("layers.{l}.ffn.gate"), cfg.inter * h);
                let wu = take(format!("layers.{l}.ffn.up"), cfg.inter * h);
                let wd = take(format!("layers.{l}.ffn.down"), h * cfg.inter);
                let mut experts = usize::MAX;
                for e in 0..cfg.experts {
                    let g = take(format!("layers.{l}.experts.{e}.gate"), cfg.inter * h);
                    take(format!("layers.{l}.experts.{e}.up"), cfg.inter * h);
                    take(format!("layers.{l}.experts.{e}.down"), h * cfg.inter);
                    if e == 0 {
                        experts = g;
                    }
                }
                FfnOffs {
                    wg,
                    wu,
                    wd,
                    experts,
                }
            };
            if cfg.is_anchor(l) {
                let ln1 = take(format!("layers.{l}.ln1"), h);
                let wq = take(
                    format!("layers.{l}.attn.q"),
                    cfg.anchor_q_heads * cfg.anchor_hd * h,
                );
                let wk = take(
                    format!("layers.{l}.attn.k"),
                    cfg.anchor_kv_heads * cfg.anchor_hd * h,
                );
                let wv = take(
                    format!("layers.{l}.attn.v"),
                    cfg.anchor_kv_heads * cfg.anchor_hd * h,
                );
                let wo = take(
                    format!("layers.{l}.attn.o"),
                    h * cfg.anchor_q_heads * cfg.anchor_hd,
                );
                // Bounded anchor: the sink vectors follow the matrices
                // (kvh·S·hd floats each; hd is a multiple of 4, so the
                // 4-float GEMM alignment of everything after them holds).
                // Absent for legacy configs, whose offsets stay unchanged.
                let (sink_k, sink_v) = if cfg.anchor_sink > 0 {
                    let n = cfg.anchor_kv_heads * cfg.anchor_sink * cfg.anchor_hd;
                    (
                        take(format!("layers.{l}.attn.sink_k"), n),
                        take(format!("layers.{l}.attn.sink_v"), n),
                    )
                } else {
                    (usize::MAX, usize::MAX)
                };
                let ln2 = take(format!("layers.{l}.ln2"), h);
                let f = ffn(&mut take);
                layers.push(LayerOffs::Anchor {
                    ln1,
                    wq,
                    wk,
                    wv,
                    wo,
                    sink_k,
                    sink_v,
                    ln2,
                    ffn: f,
                });
            } else if cfg.is_gdn_mixer() {
                let (nv, dk, dv) = (cfg.gdn_heads, cfg.gdn_dk, cfg.gdn_dv);
                let c_dim = cfg.gdn_c_dim();
                let ln1 = take(format!("layers.{l}.ln1"), h);
                let in_qkv = take(format!("layers.{l}.gdn.in_qkv"), c_dim * h);
                let in_z = take(format!("layers.{l}.gdn.in_z"), nv * dv * h);
                let in_a = take(format!("layers.{l}.gdn.in_a"), GDN_AB_PAD * h);
                let in_b = take(format!("layers.{l}.gdn.in_b"), GDN_AB_PAD * h);
                let wo = take(format!("layers.{l}.gdn.o"), h * nv * dv);
                let conv = take(format!("layers.{l}.gdn.conv"), c_dim * GDN_CONV_K);
                let alog = take(format!("layers.{l}.gdn.alog"), cfg.gdn_head_pad());
                let dt_bias = take(format!("layers.{l}.gdn.dt_bias"), cfg.gdn_head_pad());
                let norm = take(format!("layers.{l}.gdn.norm"), dv);
                let _ = dk;
                let ln2 = take(format!("layers.{l}.ln2"), h);
                let f = ffn(&mut take);
                layers.push(LayerOffs::Gdn {
                    ln1,
                    in_qkv,
                    in_z,
                    in_a,
                    in_b,
                    wo,
                    conv,
                    alog,
                    dt_bias,
                    norm,
                    ln2,
                    ffn: f,
                });
            } else {
                let ln1 = take(format!("layers.{l}.ln1"), h);
                let wq = take(format!("layers.{l}.hk.thq"), cfg.heads * cfg.nphase * h);
                let wk = take(format!("layers.{l}.hk.thk"), cfg.heads * cfg.nphase * h);
                let wv = take(format!("layers.{l}.hk.v"), cfg.heads * cfg.dv * h);
                let wkap = take(format!("layers.{l}.hk.kappa"), cfg.kappa_ld() * h);
                let wo = take(format!("layers.{l}.hk.o"), h * cfg.heads * cfg.dv);
                let alog = if cfg.learn_decay {
                    take(format!("layers.{l}.hk.alog"), cfg.heads * 2 * cfg.nphase)
                } else {
                    usize::MAX
                };
                let conv = if cfg.conv_k > 0 {
                    take(format!("layers.{l}.hk.conv"), h * cfg.conv_k)
                } else {
                    usize::MAX
                };
                let ln2 = take(format!("layers.{l}.ln2"), h);
                let f = ffn(&mut take);
                layers.push(LayerOffs::Mixer {
                    ln1,
                    wq,
                    wk,
                    wv,
                    wkap,
                    wo,
                    alog,
                    conv,
                    ln2,
                    ffn: f,
                });
            }
        }
        let final_norm = take("final_norm".into(), h);
        let head_clusters = if cfg.head_clusters > 0 {
            take("head.clusters".into(), cfg.head_clusters * h)
        } else {
            usize::MAX
        };
        let mut gdn = vec![None; cfg.layers];
        if cfg.gdn_lane {
            // Keep every old offset unchanged: append the lane tail only
            // after the legacy embedding/layer/head records.
            for l in 0..cfg.layers {
                if !cfg.is_anchor(l) {
                    let qkvz = take(format!("layers.{l}.gdn.qkvz"), 256 * h);
                    let conv = take(format!("layers.{l}.gdn.conv"), 192 * 4);
                    let ab = take(format!("layers.{l}.gdn.ab"), 64 * h);
                    let norm = take(format!("layers.{l}.gdn.norm"), 64);
                    let wo = take(format!("layers.{l}.gdn.o"), h * 64);
                    gdn[l] = Some(GdnOffs {
                        qkvz,
                        conv,
                        ab,
                        norm,
                        wo,
                        alog: usize::MAX,
                        dt_bias: usize::MAX,
                        gain: usize::MAX,
                    });
                }
            }
            // Keep scalar controls out of the per-lane matrix blocks: the
            // latter are all multiples of four floats, so every qkvz/conv/
            // ab/norm/o start remains GEMM-aligned.  Scalars are collected in
            // the tail (alongside gains) and do not participate in GEMMs.
            for l in 0..cfg.layers {
                if let Some(go) = gdn[l].as_mut() {
                    go.alog = take(format!("layers.{l}.gdn.alog"), 1);
                    go.dt_bias = take(format!("layers.{l}.gdn.dt_bias"), 1);
                }
            }
            // Gains are last and contiguous, as required by the experiment
            // contract and convenient for quick host-side statistics.
            for l in 0..cfg.layers {
                if let Some(go) = gdn[l].as_mut() {
                    go.gain = take(format!("layers.{l}.gdn.slow_gain"), 1);
                }
            }
        }
        let mut gqa = vec![None; cfg.layers];
        if cfg.gqa_lane {
            assert!(cfg.layers > 3, "gqa_lane requires a layer 4");
            assert!(
                !cfg.is_anchor(3),
                "gqa_lane layer 4 must preserve hybrid mixer path"
            );
            let l = 3usize;
            let q = take(
                format!("layers.{l}.gqa.q"),
                cfg.anchor_q_heads * cfg.anchor_hd * h,
            );
            let k = take(
                format!("layers.{l}.gqa.k"),
                cfg.anchor_kv_heads * cfg.anchor_hd * h,
            );
            let v = take(
                format!("layers.{l}.gqa.v"),
                cfg.anchor_kv_heads * cfg.anchor_hd * h,
            );
            let wo = take(
                format!("layers.{l}.gqa.o"),
                h * cfg.anchor_q_heads * cfg.anchor_hd,
            );
            gqa[l] = Some(GqaOffs { q, k, v, wo });
        }
        Layout {
            total: off,
            embed,
            final_norm,
            head_clusters,
            layers,
            gdn,
            gqa,
            names,
        }
    }
}

/// Standard normal samples (Box–Muller over the splitmix stream).
pub fn gauss_vec(seed: u64, n: usize) -> Vec<f32> {
    let u = crate::ops::lcg_vec(seed, 2 * n + 2);
    (0..n)
        .map(|i| {
            let a = (u[2 * i] as f64 * 0.5 + 0.5).clamp(1e-12, 1.0);
            let b = u[2 * i + 1] as f64 * 0.5 + 0.5;
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        })
        .collect()
}

/// Initialise a parameter arena on the host: N(0, 0.02) matrices, output
/// projections scaled by 1/√(2·layers), norms 1, κ pad rows 0.
pub fn init_params(cfg: &EmbryoCfg, lay: &Layout, seed: u64) -> Vec<f32> {
    let mut p = vec![0.0f32; lay.total];
    let std = 0.02f32;
    let out_scale = 1.0 / (2.0 * cfg.layers as f32).sqrt();
    let mut seed_i = seed;
    let mut fill = |p: &mut [f32], off: usize, n: usize, s: f32| {
        seed_i += 1;
        let g = gauss_vec(seed_i, n);
        for i in 0..n {
            p[off + i] = g[i] * s;
        }
    };
    let h = cfg.hidden;
    fill(&mut p, lay.embed, cfg.vocab * h, std);
    for (l, lo) in lay.layers.iter().enumerate() {
        let _ = l;
        match lo {
            LayerOffs::Mixer {
                ln1,
                wq,
                wk,
                wv,
                wkap,
                wo,
                alog,
                conv,
                ln2,
                ffn,
            } => {
                if *conv != usize::MAX {
                    // identity: only the current-token tap is 1 — training
                    // starts from exactly the conv-less model
                    for c in 0..h {
                        p[*conv + c * cfg.conv_k + (cfg.conv_k - 1)] = 1.0;
                    }
                }
                p[*ln1..*ln1 + h].fill(1.0);
                p[*ln2..*ln2 + h].fill(1.0);
                // phase projections: θ = W·x̂ with x̂ RMS-normed → θ std ≈ s·√H;
                // s = 0.05 gives θ ≈ N(0, 1) — a full turn of phase spread.
                let s_theta = 1.0 / (h as f32).sqrt();
                fill(&mut p, *wq, cfg.heads * cfg.nphase * h, s_theta);
                fill(&mut p, *wk, cfg.heads * cfg.nphase * h, s_theta);
                fill(&mut p, *wv, cfg.heads * cfg.dv * h, std);
                fill(&mut p, *wkap, cfg.heads * h, std); // pad rows stay 0
                fill(&mut p, *wo, h * cfg.heads * cfg.dv, std * out_scale);
                if *alog != usize::MAX {
                    // A_log = ln(−ln γ) of the horizon grid: the learned decays start
                    // exactly where the fixed ones were
                    let g = crate::ops::hk_decay_grid(
                        cfg.heads,
                        cfg.nphase,
                        cfg.horizon_min,
                        cfg.horizon_max,
                    );
                    for (i, gv) in g.iter().enumerate() {
                        p[*alog + i] = (-(*gv as f64).ln()).ln() as f32;
                    }
                }
                fill(&mut p, ffn.wg, cfg.inter * h, std);
                fill(&mut p, ffn.wu, cfg.inter * h, std);
                fill(&mut p, ffn.wd, h * cfg.inter, std * out_scale);
                for e in 0..cfg.experts {
                    let base = ffn.experts + e * 3 * h * cfg.inter;
                    fill(&mut p, base, cfg.inter * h, std);
                    fill(&mut p, base + cfg.inter * h, cfg.inter * h, std);
                    fill(
                        &mut p,
                        base + 2 * cfg.inter * h,
                        h * cfg.inter,
                        std * out_scale,
                    );
                }
            }
            LayerOffs::Gdn {
                ln1,
                in_qkv,
                in_z,
                in_a,
                in_b,
                wo,
                conv,
                alog,
                dt_bias,
                norm,
                ln2,
                ffn,
            } => {
                let (nv, dv) = (cfg.gdn_heads, cfg.gdn_dv);
                let c_dim = cfg.gdn_c_dim();
                p[*ln1..*ln1 + h].fill(1.0);
                p[*ln2..*ln2 + h].fill(1.0);
                fill(&mut p, *in_qkv, c_dim * h, std);
                fill(&mut p, *in_z, nv * dv * h, std);
                // a = b = 0 at init: β = σ(0) = ½ and the decay sits exactly on
                // its horizon (softplus(dt_bias) = 1); pad rows stay 0 forever.
                p[*in_a..*in_a + GDN_AB_PAD * h].fill(0.0);
                p[*in_b..*in_b + GDN_AB_PAD * h].fill(0.0);
                fill(&mut p, *wo, h * nv * dv, std * out_scale);
                // identity conv: only the current-token tap (the last) is 1
                for c in 0..c_dim {
                    p[*conv + c * GDN_CONV_K + (GDN_CONV_K - 1)] = 1.0;
                }
                // A_log_h = ln(1/H_h), H_h on the log grid [8, 2048] over the
                // heads (`hk_decay_grid` with the heads as the "phase" axis:
                // γ_h = e^{−1/H_h} ⇒ ln(−ln γ_h) = −ln H_h); dt_bias = ln(e−1)
                // ⇒ softplus(dt_bias) = 1 ⇒ α_h = γ_h at a = 0.
                let g = crate::ops::hk_decay_grid(1, nv, 8.0, 2048.0);
                for hh in 0..nv {
                    p[*alog + hh] = (-(g[hh] as f64).ln()).ln() as f32;
                    p[*dt_bias + hh] = (std::f32::consts::E - 1.0).ln();
                }
                p[*norm..*norm + dv].fill(1.0);
                fill(&mut p, ffn.wg, cfg.inter * h, std);
                fill(&mut p, ffn.wu, cfg.inter * h, std);
                fill(&mut p, ffn.wd, h * cfg.inter, std * out_scale);
                for e in 0..cfg.experts {
                    let base = ffn.experts + e * 3 * h * cfg.inter;
                    fill(&mut p, base, cfg.inter * h, std);
                    fill(&mut p, base + cfg.inter * h, cfg.inter * h, std);
                    fill(
                        &mut p,
                        base + 2 * cfg.inter * h,
                        h * cfg.inter,
                        std * out_scale,
                    );
                }
            }
            LayerOffs::Anchor {
                ln1,
                wq,
                wk,
                wv,
                wo,
                sink_k: _,
                sink_v: _,
                ln2,
                ffn,
            } => {
                p[*ln1..*ln1 + h].fill(1.0);
                p[*ln2..*ln2 + h].fill(1.0);
                fill(&mut p, *wq, cfg.anchor_q_heads * cfg.anchor_hd * h, std);
                fill(&mut p, *wk, cfg.anchor_kv_heads * cfg.anchor_hd * h, std);
                fill(&mut p, *wv, cfg.anchor_kv_heads * cfg.anchor_hd * h, std);
                fill(
                    &mut p,
                    *wo,
                    h * cfg.anchor_q_heads * cfg.anchor_hd,
                    std * out_scale,
                );
                fill(&mut p, ffn.wg, cfg.inter * h, std);
                fill(&mut p, ffn.wu, cfg.inter * h, std);
                fill(&mut p, ffn.wd, h * cfg.inter, std * out_scale);
                for e in 0..cfg.experts {
                    let base = ffn.experts + e * 3 * h * cfg.inter;
                    fill(&mut p, base, cfg.inter * h, std);
                    fill(&mut p, base + cfg.inter * h, cfg.inter * h, std);
                    fill(
                        &mut p,
                        base + 2 * cfg.inter * h,
                        h * cfg.inter,
                        std * out_scale,
                    );
                }
            }
        }
    }
    p[lay.final_norm..lay.final_norm + h].fill(1.0);
    if cfg.head_clusters > 0 {
        fill(&mut p, lay.head_clusters, cfg.head_clusters * h, std);
    }
    if cfg.anchor_sink > 0 {
        // Sink vectors draw from their own seed stream: appending them to
        // a legacy checkpoint (`append_anchor_sinks_checkpoint`) and being
        // born with the genome give the same sinks, and the main stream —
        // hence every legacy tensor — is untouched. sink_k ~ N(0, 0.02),
        // sink_v = 0 (the "zero value-sink"; a zero sink_k would still enter
        // the softmax denominator, so this init is not function-preserving
        // and a short re-adaptation is expected — see the plan, S2).
        let n = cfg.anchor_kv_heads * cfg.anchor_sink * cfg.anchor_hd;
        for (l, lo) in lay.layers.iter().enumerate() {
            if let LayerOffs::Anchor { sink_k, sink_v, .. } = lo {
                let g = gauss_vec(0x5349_4E4B_0000_0000u64 ^ seed ^ (l as u64 + 1), n);
                for i in 0..n {
                    p[*sink_k + i] = g[i] * std;
                }
                p[*sink_v..*sink_v + n].fill(0.0);
            }
        }
    }
    if cfg.gdn_lane {
        // The extension has its own fixed seed so appending the lane never
        // perturbs legacy initialization.  All control/state seams are
        // explicit: zero output projection and unit residual gains make the
        // candidate exactly identity at step zero while leaving the output
        // projection with an unsuppressed first-step gradient.
        let mut ext_seed = 0x4744_4E31u64;
        let mut ext_fill = |p: &mut [f32], off: usize, n: usize, s: f32| {
            ext_seed = ext_seed.wrapping_add(1);
            let g = gauss_vec(ext_seed, n);
            for i in 0..n {
                p[off + i] = g[i] * s;
            }
        };
        for go in lay.gdn.iter().flatten() {
            ext_fill(&mut p, go.qkvz, 256 * h, 0.02);
            // Identity depthwise taps (oldest → newest; current token is the
            // final tap), preserving a stable checkpoint-visible conv seam.
            for c in 0..192 {
                p[go.conv + c * 4 + 3] = 1.0;
            }
            // W_ab is padded to a GEMM-friendly 64 rows; only rows 0/1 are
            // live and remain zero so beta starts at sigmoid(0)=0.5.
            p[go.ab..go.ab + 64 * h].fill(0.0);
            p[go.norm..go.norm + 64].fill(1.0);
            // Zero output is the load-bearing identity seam: unlike a random
            // output plus near-zero gain, this keeps initial logits bit-identical
            // while allowing Wo to learn on the first backward pass.
            p[go.wo..go.wo + h * 64].fill(0.0);
            p[go.alog] = -(1024.0f32).ln();
            p[go.dt_bias] = (std::f32::consts::E - 1.0).ln();
            p[go.gain] = 1.0;
        }
    }
    if cfg.gqa_lane {
        let mut ext_seed = 0x4751_4131u64;
        let mut ext_fill = |p: &mut [f32], off: usize, n: usize, s: f32| {
            ext_seed = ext_seed.wrapping_add(1);
            let g = gauss_vec(ext_seed, n);
            for i in 0..n {
                p[off + i] = g[i] * s;
            }
        };
        if let Some(go) = lay.gqa.get(3).and_then(|x| x.as_ref()) {
            let qn = cfg.anchor_q_heads * cfg.anchor_hd * h;
            let kvn = cfg.anchor_kv_heads * cfg.anchor_hd * h;
            ext_fill(&mut p, go.q, qn, std);
            ext_fill(&mut p, go.k, kvn, std);
            ext_fill(&mut p, go.v, kvn, std);
            p[go.wo..go.wo + h * cfg.anchor_q_heads * cfg.anchor_hd].fill(0.0);
        }
    }
    p
}

/// principal directions per expert descriptor (reserved; filled by the
/// periodic PCA of the routed inputs). Shape constants, not device ones:
/// the exporter and the growth records read them on every platform, so they
/// live outside the Metal-only module that consumes them.
pub const MOE_K: usize = 16;
/// EMA rate of the descriptor means and the balancing-bias step
pub const MOE_ALPHA: f32 = 0.02;
pub const MOE_ETA: f32 = 0.05;

#[cfg(any(target_os = "macos", feature = "vulkan"))]
pub use gpu::*;

#[cfg(any(target_os = "macos", feature = "vulkan"))]
mod gpu {
    use super::*;
    use crate::metal::{
        ctx, hk_pow_table, Cmd, Ctx, GBuf, GdnScanDims, GemmBatch, GemmDyn, HkDims, HkGrads,
        HkScratch, HkWork, Op, RouteDims,
    };
    use crate::ops::hk_decay_grid;

    /// Per-layer activation buffers kept for the backward (M = B·T rows).
    pub enum LayerActs {
        Mixer {
            x_in: GBuf, // [M,H] residual stream entering the layer
            x1: GBuf,   // [M,H] normed
            x1c: GBuf,
            inv1: GBuf, // [M]
            thq: GBuf,  // [M, nh·nph]
            thk: GBuf,
            v: GBuf,     // [M, nh·dv]
            kpre: GBuf,  // [M, kappa_ld]
            kappa: GBuf, // [M, nh]
            phq: GBuf,   // [M, nh·2nph]
            phk: GBuf,
            kv: GBuf,     // [M, nh·dv]
            states: GBuf, // [B·nh·(T/64+1)·2nph·dv]
            o: GBuf,      // [M, nh·dv]
            // Optional appended GDN correction lane (one head, dk=dv=64).
            gdn_thq: GBuf,   // [M,32]
            gdn_thk: GBuf,   // [M,32]
            gdn_qfull: GBuf, // [M,64] GEMM-tile staging
            gdn_kfull: GBuf, // [M,64] GEMM-tile staging
            gdn_v: GBuf,     // [M,64]
            gdn_ab: GBuf,    // [M,64] padded a/b projection
            gdn_kappa: GBuf, // [M,1] beta gate
            gdn_phq: GBuf,   // [M,64]
            gdn_phk: GBuf,
            gdn_kv: GBuf,
            gdn_states: GBuf,
            gdn_raw_o: GBuf, // [M,64] pre-RMSNorm recurrent output
            gdn_inv: GBuf,   // [M] recurrent output RMS inverse
            gdn_o: GBuf,     // [M,64]
            gdn_z: GBuf,     // [M,64]
            gdn_gated: GBuf, // [M,64]
            gdn_dz: GBuf,    // [M,64]
            gdn_do: GBuf,    // [M,64]
            gdn_dq: GBuf,    // [M,64] d conv-q / projection q
            gdn_dk: GBuf,    // [M,64] d conv-k / projection k
            gdn_dv: GBuf,    // [M,64] d conv-v / projection v
            gdn_dab: GBuf,   // [M,64] d control projection
            gdn_d: GBuf,     // [M,H]
            gdn_grad: GBuf,  // [M,H] dL/d correction for gain/dot
            // Optional layer-4 additive GQA donor lane (q/k/v/o attention).
            gqa_q: GBuf, // [M,qh·hd]
            gqa_k: GBuf, // [M,kvh·hd]
            gqa_v: GBuf, // [M,kvh·hd]
            gqa_p: GBuf, // [B,qh,T,T]
            gqa_o: GBuf, // [M,qh·hd]
            x_mid: GBuf, // [M,H]
            x2: GBuf,    // [M,H]
            inv2: GBuf,
            gte: GBuf, // [M,I]
            up: GBuf,
            hh: GBuf,
        },
        Anchor {
            x_in: GBuf,
            x1: GBuf,
            inv1: GBuf,
            q: GBuf, // [M, qh·hd] (after RoPE)
            k: GBuf, // [M, kvh·hd]
            v: GBuf, // [M, kvh·hd]
            /// [B, qh, T, LD] softmax probabilities, LD = sink_pad + T:
            /// columns 0..S are the sink block (padded to `SINK_PAD`, the
            /// pad columns are exactly 0), the causal/band block starts at
            /// column `sink_pad` (legacy: LD = T, the old [B, qh, T, T]).
            p: GBuf,
            o: GBuf, // [M, qh·hd]
            /// [M, qh·hd] the UNROTATED q̂ (sink scores are NoPE); len 1
            /// without sinks
            q_raw: GBuf,
            /// [kvh, SINK_PAD, hd] the layer's sink keys/values padded with
            /// zero rows to the GEMM tile (rows < S copied from the arena
            /// every forward); len 1 without sinks
            sink_k_pad: GBuf,
            sink_v_pad: GBuf,
            x_mid: GBuf,
            x2: GBuf,
            inv2: GBuf,
            gte: GBuf,
            up: GBuf,
            hh: GBuf,
        },
        /// Gated DeltaNet mixer (`Mixer::Gdn`). No per-token state history:
        /// the scan keeps one state checkpoint every 64 tokens and replays
        /// the chunk in the backward (`Scratch::gdn_chunk`).
        Gdn {
            x_in: GBuf,   // [M,H]
            x1: GBuf,     // [M,H] normed
            inv1: GBuf,   // [M]
            qkv: GBuf,    // [M, c_dim] raw projections (pre conv)
            qkv_cv: GBuf, // [M, c_dim] SiLU(conv1d(qkv))
            z: GBuf,      // [M, nv·dv] output-gate pre-activation
            a_pre: GBuf,  // [M, GDN_AB_PAD] decay pre-activation (column h = head h)
            b_pre: GBuf,  // [M, GDN_AB_PAD] β pre-activation
            /// [B, nv, T/64 + 1, dk, dv] state checkpoints (slot 0 = S_0)
            states: GBuf,
            raw_o: GBuf,  // [M, nv·dv] Sᵀq̂ before the gated norm
            inv_o: GBuf,  // [M·nv] per-head RMS inverse
            o_norm: GBuf, // [M, nv·dv] RMSNorm(raw_o)·w
            gated: GBuf,  // [M, nv·dv] SiLU(z)·o_norm
            x_mid: GBuf,
            x2: GBuf,
            inv2: GBuf,
            gte: GBuf,
            up: GBuf,
            hh: GBuf,
        },
    }

    /// Result of [`EmbryoGpu::anchor_core_probe`] (host copies).
    #[derive(Clone, Debug)]
    pub struct AnchorProbe {
        /// diagnostics: score block 0 (sequence 0, head 0: [T, ld]) and the
        /// attention output of sequence 0 ([T, qd]) as left by the forward
        /// (or by the stage `CMF_ANCHOR_STAGE` stops at: 1 = scores, 2 =
        /// softmax, 3 = P·V)
        pub p0: Vec<f32>,
        pub o0: Vec<f32>,
        pub y: Vec<f32>,       // [M,H]  o·Woᵀ
        pub dq: Vec<f32>,      // [M,qd] dL/dq̂ (raw, before rope)
        pub dk: Vec<f32>,      // [M,kd] dL/dk̂ (raw)
        pub dv: Vec<f32>,      // [M,kd]
        pub dsink_k: Vec<f32>, // [kvh·S·hd] (empty without sinks)
        pub dsink_v: Vec<f32>, // [kvh·S·hd]
        pub dwo: Vec<f32>,     // [H, qd]
    }

    /// Routed-expert activations of one layer (kept for the backward).
    pub struct MoeActs {
        pub assign: GBuf, // [M] u32 expert of each token
        pub slot: GBuf,   // [M] u32 rank within its expert (≥ cap: dropped)
        pub res: GBuf,    // [M] raw current-row resonance of the chosen expert
        pub hg: GBuf,     // [E, cap, H] gathered inputs
        pub gte: GBuf,    // [E, cap, I]
        pub up: GBuf,
        pub hh: GBuf,
        pub yh: GBuf, // [E, cap, H] expert outputs
        /// Optional conditional top-2 runner-up stream.  These are activation
        /// buffers only (never checkpointed/optimized) and remain one-float
        /// dummies when the feature is disabled so the legacy arena is
        /// unchanged.
        pub runner: GBuf, // [M] u32 expert or UINT_MAX
        pub slot2: GBuf, // [M] u32 rank within runner expert
        pub margin: GBuf, // [M] adjusted score margin (diagnostics)
        pub runner_weight: GBuf, // [M] 0.5 for active runner, otherwise 0
        pub fallback_count: GBuf, // [1] per-dispatch telemetry
        pub count2: GBuf, // [E] runner stream counts
        pub indir2: GBuf, // [2, E, 3] runner GEMM dispatch grids
        pub hg2: GBuf, // [E, cap, H]
        pub gte2: GBuf, // [E, cap, I]
        pub up2: GBuf,
        pub hh2: GBuf,
        pub yh2: GBuf, // [E, cap, H]
    }

    /// Resonance descriptors of all layers' experts (not gradient-trained:
    /// online statistics — μ EMA, balancing bias; U by periodic PCA).
    pub struct Desc {
        pub mu: GBuf,    // [L, E, H]
        pub u: GBuf,     // [L, E, K, H]
        pub bias: GBuf,  // [L, E]
        pub count: GBuf, // [L, E] u32 tokens routed last step
        pub sums: GBuf,  // [L, E, H] Σ inputs last step
        pub indir: GBuf, // [L, 2, E, 3] u32 indirect dispatch grids
    }

    /// Initial descriptor means: unit-RMS Gaussian points (the FFN input is
    /// RMS-normed) — a k-means-style random init the EMA then moves.
    pub fn desc_init(cfg: &EmbryoCfg, seed: u64) -> Vec<f32> {
        let n = cfg.layers * cfg.experts * cfg.hidden;
        if n == 0 {
            return vec![0.0; 1];
        }
        gauss_vec(seed, n)
    }

    /// Scratch for the backward, shared by all layers.
    pub struct Scratch {
        pub dx: GBuf,    // [M,H] the gradient flowing down the residual stream
        pub dx1: GBuf,   // [M,H]
        pub dxc: GBuf,   // [M,H] conv-input grad (the short-conv backward)
        pub dx2: GBuf,   // [M,H]
        pub dbig: GBuf,  // [M, max(nh·dv, qh·hd)]  do / dq
        pub dk: GBuf,    // [M, max(nh·nph, kvh·hd)]
        pub dk2: GBuf,   // [M, nh·nph]  dthk
        pub dv: GBuf,    // [M, max(nh·dv, kvh·hd)]
        pub dkap: GBuf,  // [M, nh]
        pub dkpre: GBuf, // [M, kappa_ld]
        pub dstates: GBuf,
        /// Shared bounded activation scratch for the Phase-Delta reverse
        /// scan: `[B, heads, 65, 2*nphase, dv]` f32 values.
        pub phase_chunk: GBuf,
        /// Conditional Phase-Delta block partials:
        /// `[B, heads, ceil(dv/32), T, 1+2*nphase]` f32 values.
        pub phase_partial: GBuf,
        pub dkv: GBuf,  // [M, nh·dv]
        pub dq: GBuf,   // [M, qh·hd]  anchor dQ
        pub dphq: GBuf, // [M, nh·2nph]
        pub dphk: GBuf,
        pub dp: GBuf,     // [B, qh, T, LD] score-gradient blocks (LD = sink_pad + T)
        pub dkh: GBuf,    // [B, qh, T, hd] per-head dK partials
        pub dvh: GBuf,    // [B, qh, T, hd]
        /// [B, qh, SINK_PAD, hd] per-(sequence, head) dsink_k / dsink_v
        /// partial tiles of the bounded anchor (len 1 without sinks)
        pub dsk_part: GBuf,
        pub dsv_part: GBuf,
        pub dffn: GBuf,   // [M,I] dh
        pub dgte: GBuf,   // [M,I]
        pub dup: GBuf,    // [M,I]
        pub logits: GBuf, // [R, V] head row chunk
        // routed experts backward scratch (per layer reuse)
        pub moe_dyh: GBuf,  // [E, cap, H]
        pub moe_dhg: GBuf,  // [E, cap, H]
        pub moe_dffn: GBuf, // [E, cap, I]
        pub moe_dgte: GBuf, // [E, cap, I]
        pub moe_dup: GBuf,  // [E, cap, I]
        pub hh_pre: GBuf,   // [M, I] pre-mask FFN activation (skill bake)
        // hybrid_k GEMM-formulation scratch (chunk-major tables + A)
        pub hk_qt: GBuf,
        pub hk_kt: GBuf,
        pub hk_qp: GBuf,
        pub hk_kh: GBuf,
        pub hk_dqt: GBuf,
        pub hk_dkt: GBuf,
        pub hk_dqi: GBuf,
        pub hk_dki: GBuf,
        pub hk_a: GBuf,
        // GDN mixer scratch (one-float dummies unless `cfg.mixer == Gdn`)
        pub gdn_live: GBuf,   // [B, nv, dk, dv] running state of the forward scan
        pub gdn_dlive: GBuf,  // [B, nv, dk, dv] running state adjoint of the backward
        pub gdn_chunk: GBuf,  // [B, nv, 65, dk, dv] replayed states of one chunk
        pub gdn_pre: GBuf,    // [M, c_dim] conv pre-activation (recomputed)
        pub gdn_dcv: GBuf,    // [M, c_dim] d qkv_cv
        pub gdn_dpre: GBuf,   // [M, c_dim] d conv pre-activation
        pub gdn_dqkv: GBuf,   // [M, c_dim] d raw projections
        pub gdn_dgated: GBuf, // [M, nv·dv]
        pub gdn_dz: GBuf,     // [M, nv·dv]
        pub gdn_don: GBuf,    // [M, nv·dv] d o_norm
        pub gdn_doo: GBuf,    // [M, nv·dv] d raw_o
        pub gdn_da: GBuf,     // [M, GDN_AB_PAD]
        pub gdn_db: GBuf,     // [M, GDN_AB_PAD]
        pub gdn_part: GBuf,   // [B·nv, 2] per-(b, head) dA_log / d dt_bias partials
        pub loss: GBuf, // [M]
        pub partial: GBuf,
    }

    /// State carried across consecutive windows of one document stream
    /// (plan S6b / TBPTT-lite, `birth --carry`): every recurrent layer's S_0
    /// lives in checkpoint slot 0 of its `states` (copied from the last
    /// boundary slot by `carry_commit`), the short convs keep their last
    /// k−1 inputs, the bounded anchors the last `cp` raw keys/values of the
    /// previous window (rotated at positions 0..cp, the window at cp..cp+T,
    /// so every relative angle is exact). Everything is detached: no
    /// gradient crosses a window boundary.
    pub struct CarryState {
        /// carried-anchor columns (anchor_window rounded up to 64; 0 = none)
        pub cp: usize,
        /// [B] u32: 1 = the sequence continues (its carried inputs are valid)
        pub mask: GBuf,
        pub mask_host: Vec<bool>,
        /// [B] zeros: the mask of every pass outside `carry_begin`/`commit`
        /// (evaluation windows never see carried keys)
        pub mask_zero: GBuf,
        /// per layer: (hist_cur, hist_next, k, width) of the layer's short conv
        pub conv: Vec<Option<(GBuf, GBuf, usize, usize)>>,
        /// per anchor layer: (k_cur, v_cur, k_next, v_next, k_rot), each [B, cp, kd]
        pub tails: Vec<Option<(GBuf, GBuf, GBuf, GBuf, GBuf)>>,
        /// per recurrent layer: the carried S_0 store [B·heads, state] — a
        /// copy outside `states` (which every forward, including the
        /// held-out evaluation, rewrites), plus the per-(b, head) state size
        pub s0: Vec<Option<(GBuf, usize)>>,
        /// zeros for the per-row resets (≥ the largest slot copied)
        pub zeros: GBuf,
    }

    pub struct EmbryoGpu {
        pub cfg: EmbryoCfg,
        pub lay: Layout,
        pub b: usize,
        pub t: usize,
        pub p: GBuf,
        pub g: GBuf,
        pub m: GBuf,
        pub v: GBuf,
        pub pow: GBuf,
        /// Decay table for the optional one-head correction lane.
        pub gdn_pow: GBuf,
        /// Number of appended correction lanes (one per non-anchor layer).
        pub gdn_lane_count: usize,
        pub acts: Vec<LayerActs>,
        pub x_out: GBuf, // [M,H] last layer output (pre final norm)
        pub xf: GBuf,    // [M,H] final-normed
        pub invf: GBuf,
        pub dxf: GBuf, // [M,H]
        pub xft: GBuf, // [M,H] teacher's final-normed hidden (distillation)
        /// (offset, len) grad ranges zeroed after the backward — donor
        /// tensors held still while the fresh mixers learn to fit them.
        pub freeze: Vec<(usize, usize)>,
        pub tok: GBuf, // [M] u32 inputs
        pub tgt: GBuf, // [M] u32 targets
        // hierarchical head (cfg.head_clusters > 0)
        pub tgt_cluster: GBuf, // [M] u32 target cluster ids
        pub head_idx: GBuf,    // [Mpad] i32 grouped row → token index (−1 pad)
        /// (cluster, row offset, padded rows) of the grouped rows
        pub head_groups: std::cell::RefCell<Vec<(usize, usize, usize)>>,
        /// positions with a target (u32::MAX = ignored) in the last prepared batch
        pub head_valid: std::cell::Cell<usize>,
        pub hg: GBuf,    // [Mpad, H] gathered rows
        pub dhg: GBuf,   // [Mpad, H]
        pub lw: GBuf,    // [Mpad, S] within-cluster logits
        pub lc: GBuf,    // [M, C] cluster logits
        pub loss2: GBuf, // [M]
        pub mpad: usize,
        /// routed experts: per-layer activations, descriptors, capacity
        pub moe: Vec<MoeActs>,
        pub desc: Desc,
        pub moe_cap: usize,
        /// online descriptor updates during training forwards (tests turn it off)
        pub desc_updates: std::cell::Cell<bool>,
        /// reuse the previous step's expert assignments instead of routing (the
        /// gradcheck freezes the piecewise-linear region; never set in training)
        pub route_frozen: std::cell::Cell<bool>,
        /// skill bake state (masks over the shared FFN of selected layers)
        pub skill: Option<SkillState>,
        /// descriptors seeded from data (first training forward)
        pub desc_seeded: std::cell::Cell<bool>,
        /// experts below this index keep their descriptors frozen (grown genome:
        /// old records never move; only the new expert's μ/bias adapt)
        pub desc_frozen_below: std::cell::Cell<usize>,
        /// experts from this index on keep their balancing bias (growth
        /// records: a grown expert's `desc.bias` is written as its frozen
        /// value — 0, or the source's under `--bias-mode source` — and must
        /// train under that same value, so the record routes at runtime
        /// exactly as it trained); `usize::MAX` = every bias moves (birth)
        pub bias_frozen_from: std::cell::Cell<usize>,
        pub desc_seed_rows: GBuf,
        pub scratch: Scratch,
        pub step: u32,
        /// rows per head chunk (logits [head_rows, V] materialised at a time)
        pub head_rows: usize,
        /// The anchor window of the training pass being encoded (sampled
        /// from `cfg.anchor_train_windows` at the start of every training
        /// forward; the backward reads the same value). Evaluation forwards
        /// always use the served `cfg.anchor_window`.
        pub anchor_window_t: std::cell::Cell<usize>,
        /// Force the served window during training too (the last ~10% of a
        /// birth trains on the served window; tests use it for FD checks).
        pub anchor_fixed_window: std::cell::Cell<bool>,
        /// GDN mixer: read the initial state S_0 of every (sequence, head)
        /// from checkpoint slot 0 of the layer's `states` buffer (written by
        /// the caller — state carried across windows) instead of zero.
        pub gdn_s0_from_ckpt: std::cell::Cell<bool>,
        /// Carry-over state (`--carry`); None = every window starts fresh.
        pub carry: Option<CarryState>,
        /// Set by `carry_begin`, cleared by `carry_commit`: the forwards and
        /// backwards encoded in between read the carried inputs and write the
        /// next window's tails.
        pub carry_active: std::cell::Cell<bool>,
        /// GDN mixer on Vulkan: the chunked WY/UT form (`gdn_wy.rs`) replaces
        /// the token scan (`CMF_GDN_WY=0` selects the scan for A/B): the
        /// shared backward scratch plus one kept-intermediates arena per
        /// GDN layer (indexed by layer; None for anchors).
        #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
        pub gdn_wy: Option<(crate::gdn_wy::GdnWyScratch, Vec<Option<crate::gdn_wy::GdnWyKeep>>)>,
    }

    /// Terminal, host-side witness for the parameter-neutral Phase-Delta
    /// dynamics.  This is deliberately a plain snapshot: callers opt in to
    /// the readback after a timed step, so the legacy/control path pays no
    /// allocation or synchronization cost.
    #[derive(Clone, Debug)]
    pub struct PhaseDeltaLayerTelemetry {
        pub layer: usize,
        pub beta_p01: f32,
        pub beta_p50: f32,
        pub beta_p99: f32,
        pub r_rms: f32,
        pub e_rms: f32,
        pub correction_rms: f32,
        pub v_rms: f32,
        pub state_rms: f32,
        pub state_max: f32,
        pub wk_grad_l2: f32,
        pub wv_grad_l2: f32,
        pub wkappa_grad_l2: f32,
        pub finite: bool,
    }

    impl EmbryoGpu {
        pub fn new(cfg: EmbryoCfg, b: usize, t: usize, params: &[f32]) -> Option<EmbryoGpu> {
            Self::new_with_moe_capacity(cfg, b, t, params, false, false)
        }

        /// `new` / `new_eval_dropless` with the carry-over state allocated
        /// (`carry_begin` / `carry_commit` around every window).
        pub fn new_carry(
            cfg: EmbryoCfg,
            b: usize,
            t: usize,
            params: &[f32],
            dropless_eval: bool,
        ) -> Option<EmbryoGpu> {
            Self::new_with_moe_capacity(cfg, b, t, params, dropless_eval, true)
        }

        /// Evaluation-only constructor with enough resident expert capacity
        /// for every row to select the same expert.  It preserves the normal
        /// capacity-2 training allocation while making route/parity traces
        /// dropless by construction; no softmax router is introduced.
        pub fn new_eval_dropless(
            cfg: EmbryoCfg,
            b: usize,
            t: usize,
            params: &[f32],
        ) -> Option<EmbryoGpu> {
            Self::new_with_moe_capacity(cfg, b, t, params, true, false)
        }

        fn new_with_moe_capacity(
            cfg: EmbryoCfg,
            b: usize,
            t: usize,
            params: &[f32],
            dropless_eval: bool,
            carry: bool,
        ) -> Option<EmbryoGpu> {
            cfg.validate_phase_delta_layer();
            // The resident Vulkan graph deliberately exposes only the fixed
            // hybrid/Phase-Delta + top-1 resonance contract.  Optional tail
            // experiments have different state/gradient laws and must fail
            // closed rather than silently taking a legacy no-op path.
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            if cfg.gdn_lane
                || cfg.learn_decay
                || cfg.router_smooth_k4
                || cfg.router_top2_margin.is_some()
            {
                return None;
            }
            let c = ctx()?;
            assert!(
                !cfg.phase_delta_active() || !cfg.learn_decay,
                "Phase-Delta requires fixed decay (learn_decay=false)"
            );
            let lay = Layout::new(&cfg);
            assert_eq!(params.len(), lay.total);
            assert!(
                t % 64 == 0 && (b * t) % 64 == 0,
                "B·T and T must be multiples of 64"
            );
            let m = b * t;
            let h = cfg.hidden;
            let z = |n: usize| GBuf::zeros(c, n);
            // GDN scratch is allocated only for the GDN mixer (dummies otherwise)
            let gdn_sz = |n: usize| if cfg.is_gdn_mixer() { n } else { 1 };
            let nhd = cfg.heads * cfg.dv;
            let nhp = cfg.heads * cfg.nphase;
            let qhd = cfg.anchor_q_heads * cfg.anchor_hd;
            let kvhd = cfg.anchor_kv_heads * cfg.anchor_hd;
            let nst = b * cfg.heads * (t / 64 + 1) * 2 * cfg.nphase * cfg.dv;
            // Exact ordinary GDN keeps one 64×64 state snapshot per token for
            // the reverse causal scan (the legacy HK lane remains chunked).
            let gdn_nst = b * (t + 1) * 64 * 64;
            // carried anchor keys: the served window rounded up to a tile
            let cp = if carry && cfg.anchor_bounded() {
                cfg.anchor_window.div_ceil(64) * 64
            } else {
                0
            };
            let mut acts = Vec::new();
            for l in 0..cfg.layers {
                if cfg.is_anchor(l) {
                    let sinks = cfg.sink_pad() > 0;
                    acts.push(LayerActs::Anchor {
                        x_in: z(m * h),
                        x1: z(m * h),
                        inv1: z(m),
                        q: z(m * qhd),
                        k: z(m * kvhd),
                        v: z(m * kvhd),
                        p: z(b * cfg.anchor_q_heads * t * (cfg.anchor_ld(t) + cp)),
                        o: z(m * qhd),
                        q_raw: z(if sinks { m * qhd } else { 1 }),
                        sink_k_pad: z(if sinks {
                            cfg.anchor_kv_heads * SINK_PAD * cfg.anchor_hd
                        } else {
                            1
                        }),
                        sink_v_pad: z(if sinks {
                            cfg.anchor_kv_heads * SINK_PAD * cfg.anchor_hd
                        } else {
                            1
                        }),
                        x_mid: z(m * h),
                        x2: z(m * h),
                        inv2: z(m),
                        gte: z(m * cfg.inter),
                        up: z(m * cfg.inter),
                        hh: z(m * cfg.inter),
                    });
                } else if cfg.is_gdn_mixer() {
                    let (nv, dk, dv) = (cfg.gdn_heads, cfg.gdn_dk, cfg.gdn_dv);
                    let c_dim = cfg.gdn_c_dim();
                    acts.push(LayerActs::Gdn {
                        x_in: z(m * h),
                        x1: z(m * h),
                        inv1: z(m),
                        qkv: z(m * c_dim),
                        qkv_cv: z(m * c_dim),
                        z: z(m * nv * dv),
                        a_pre: z(m * GDN_AB_PAD),
                        b_pre: z(m * GDN_AB_PAD),
                        states: z(b * nv * (t / 64 + 1) * dk * dv),
                        raw_o: z(m * nv * dv),
                        inv_o: z(m * nv),
                        o_norm: z(m * nv * dv),
                        gated: z(m * nv * dv),
                        x_mid: z(m * h),
                        x2: z(m * h),
                        inv2: z(m),
                        gte: z(m * cfg.inter),
                        up: z(m * cfg.inter),
                        hh: z(m * cfg.inter),
                    });
                } else {
                    acts.push(LayerActs::Mixer {
                        x_in: z(m * h),
                        x1: z(m * h),
                        x1c: z(if cfg.conv_k > 0 { m * h } else { 1 }),
                        inv1: z(m),
                        thq: z(m * nhp),
                        thk: z(m * nhp),
                        v: z(m * nhd),
                        kpre: z(m * cfg.kappa_ld()),
                        kappa: z(m * cfg.heads),
                        phq: z(m * 2 * nhp),
                        phk: z(m * 2 * nhp),
                        kv: z(m * nhd),
                        states: z(nst),
                        o: z(m * nhd),
                        gdn_thq: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 32
                        } else {
                            1
                        }),
                        gdn_thk: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 32
                        } else {
                            1
                        }),
                        gdn_qfull: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_kfull: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_v: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_ab: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_kappa: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m
                        } else {
                            1
                        }),
                        gdn_phq: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_phk: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_kv: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_states: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            gdn_nst
                        } else {
                            1
                        }),
                        gdn_raw_o: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_inv: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m
                        } else {
                            1
                        }),
                        gdn_o: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_z: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_gated: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_dz: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_do: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_dq: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_dk: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_dv: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_dab: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * 64
                        } else {
                            1
                        }),
                        gdn_d: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * h
                        } else {
                            1
                        }),
                        gdn_grad: z(if cfg.gdn_lane && !cfg.is_anchor(l) {
                            m * h
                        } else {
                            1
                        }),
                        gqa_q: z(if cfg.gqa_lane && l == 3 { m * qhd } else { 1 }),
                        gqa_k: z(if cfg.gqa_lane && l == 3 { m * kvhd } else { 1 }),
                        gqa_v: z(if cfg.gqa_lane && l == 3 { m * kvhd } else { 1 }),
                        gqa_p: z(if cfg.gqa_lane && l == 3 {
                            b * cfg.anchor_q_heads * t * t
                        } else {
                            1
                        }),
                        gqa_o: z(if cfg.gqa_lane && l == 3 { m * qhd } else { 1 }),
                        x_mid: z(m * h),
                        x2: z(m * h),
                        inv2: z(m),
                        gte: z(m * cfg.inter),
                        up: z(m * cfg.inter),
                        hh: z(m * cfg.inter),
                    });
                }
            }
            // logits are materialised `head_rows` rows at a time: the largest
            // tile-multiple divisor of M up to 1024 (1024 for every historical
            // shape; e.g. M = 1088 → 64 rows × 17 chunks instead of a refusal)
            let head_rows = (1..=16)
                .rev()
                .map(|k| k * 64)
                .find(|&r| m % r == 0)
                .unwrap_or(64);
            assert!(m % head_rows == 0);
            // routed experts: capacity factor 2 per expert, rows padded to the tile
            let ne = cfg.experts;
            let moe_cap = if ne > 0 {
                if dropless_eval {
                    m.div_ceil(64) * 64
                } else {
                    (2 * m / ne).div_ceil(64) * 64
                }
            } else {
                0
            };
            let mut moe = Vec::new();
            let top2 = cfg.router_top2_enabled();
            for _ in 0..cfg.layers {
                let stream_h = if top2 { ne * moe_cap * h } else { 1 };
                let stream_i = if top2 { ne * moe_cap * cfg.inter } else { 1 };
                moe.push(MoeActs {
                    assign: GBuf::from_u32(c, &vec![0u32; m.max(1)]),
                    slot: GBuf::from_u32(c, &vec![0u32; m.max(1)]),
                    res: z(m),
                    hg: z(ne * moe_cap * h),
                    gte: z(ne * moe_cap * cfg.inter),
                    up: z(ne * moe_cap * cfg.inter),
                    hh: z(ne * moe_cap * cfg.inter),
                    yh: z(ne * moe_cap * h),
                    runner: GBuf::from_u32(c, &vec![u32::MAX; if top2 { m.max(1) } else { 1 }]),
                    slot2: GBuf::from_u32(c, &vec![u32::MAX; if top2 { m.max(1) } else { 1 }]),
                    margin: z(if top2 { m.max(1) } else { 1 }),
                    runner_weight: z(if top2 { m.max(1) } else { 1 }),
                    fallback_count: z(1),
                    count2: GBuf::from_u32(c, &vec![0u32; if top2 { ne.max(1) } else { 1 }]),
                    indir2: GBuf::from_u32(c, &vec![0u32; if top2 { 2 * ne * 3 } else { 1 }]),
                    hg2: z(stream_h),
                    gte2: z(stream_i),
                    up2: z(stream_i),
                    hh2: z(stream_i),
                    yh2: z(stream_h),
                });
            }
            let desc = Desc {
                mu: GBuf::from_slice(c, &desc_init(&cfg, 12345)),
                u: z(cfg.layers * ne * MOE_K * h),
                bias: z(cfg.layers * ne),
                count: GBuf::from_u32(c, &vec![0u32; (cfg.layers * ne).max(1)]),
                sums: z(cfg.layers * ne * h),
                indir: GBuf::from_u32(c, &vec![0u32; (cfg.layers * 2 * ne * 3).max(1)]),
            };
            let ncl = cfg.head_clusters;
            let mpad = m + ncl * 64;
            let cs = if ncl > 0 { cfg.vocab / ncl } else { 0 };
            if ncl > 0 {
                assert!(
                    cfg.vocab % ncl == 0 && cs % 64 == 0 && ncl % 64 == 0,
                    "hierarchical head: vocab = C·S with C, S multiples of 64"
                );
            }
            let scratch = Scratch {
                dx: z(m * h),
                dx1: z(m * h),
                dxc: z(m * h),
                dx2: z(m * h),
                dbig: z(m * nhd.max(qhd)),
                dk: z(m * nhp.max(kvhd)),
                dk2: z(m * nhp),
                dv: z(m * nhd.max(kvhd)),
                dkap: z(m * cfg.heads),
                dkpre: z(m * cfg.kappa_ld()),
                dstates: z(nst),
                // Bounded split-SIMD Phase-Delta scan scratch is conditional;
                // controls retain one-float dummies.
                phase_chunk: z(if cfg!(all(feature = "vulkan", not(target_os = "macos"))) {
                    b * cfg.heads * 65 * (2 * cfg.nphase + 1) * cfg.dv
                } else if cfg.phase_delta_active() {
                    b * cfg.heads * 65 * 2 * cfg.nphase * cfg.dv
                } else {
                    1
                }),
                phase_partial: z(
                    if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                        || cfg.phase_delta_active()
                    {
                        b * cfg.heads * cfg.dv.div_ceil(32) * t * (1 + 2 * cfg.nphase)
                    } else {
                        1
                    },
                ),
                dkv: z(m * nhd),
                dq: z(m * qhd),
                dphq: z(m * 2 * nhp),
                dphk: z(m * 2 * nhp),
                // The anchor backward is batched over (b, head): dP holds
                // every sequence's blocks (the same size as P).
                dp: z(b * cfg.anchor_q_heads * t * (cfg.anchor_ld(t) + cp)),
                dkh: z(m * qhd),
                dvh: z(m * qhd),
                dsk_part: z(if cfg.sink_pad() > 0 {
                    b * cfg.anchor_q_heads * SINK_PAD * cfg.anchor_hd
                } else {
                    1
                }),
                dsv_part: z(if cfg.sink_pad() > 0 {
                    b * cfg.anchor_q_heads * SINK_PAD * cfg.anchor_hd
                } else {
                    1
                }),
                dffn: z(m * cfg.inter),
                dgte: z(m * cfg.inter),
                dup: z(m * cfg.inter),
                logits: z(head_rows * cfg.vocab),
                moe_dyh: z(ne * moe_cap * h),
                moe_dhg: z(ne * moe_cap * h),
                moe_dffn: z(ne * moe_cap * cfg.inter),
                moe_dgte: z(ne * moe_cap * cfg.inter),
                moe_dup: z(ne * moe_cap * cfg.inter),
                hh_pre: z(m * cfg.inter),
                hk_qt: z(m * 2 * nhp),
                hk_kt: z(m * 2 * nhp),
                hk_qp: z(m * 2 * nhp),
                hk_kh: z(m * 2 * nhp),
                hk_dqt: z(m * 2 * nhp),
                hk_dkt: z(m * 2 * nhp),
                hk_dqi: z(m * 2 * nhp),
                hk_dki: z(m * 2 * nhp),
                hk_a: z(b * cfg.heads * (t / 64) * 4096),
                gdn_live: z(gdn_sz(b * cfg.gdn_heads * cfg.gdn_dk * cfg.gdn_dv)),
                gdn_dlive: z(gdn_sz(b * cfg.gdn_heads * cfg.gdn_dk * cfg.gdn_dv)),
                gdn_chunk: z(gdn_sz(b * cfg.gdn_heads * 65 * cfg.gdn_dk * cfg.gdn_dv)),
                gdn_pre: z(gdn_sz(m * cfg.gdn_c_dim())),
                gdn_dcv: z(gdn_sz(m * cfg.gdn_c_dim())),
                gdn_dpre: z(gdn_sz(m * cfg.gdn_c_dim())),
                gdn_dqkv: z(gdn_sz(m * cfg.gdn_c_dim())),
                gdn_dgated: z(gdn_sz(m * cfg.gdn_heads * cfg.gdn_dv)),
                gdn_dz: z(gdn_sz(m * cfg.gdn_heads * cfg.gdn_dv)),
                gdn_don: z(gdn_sz(m * cfg.gdn_heads * cfg.gdn_dv)),
                gdn_doo: z(gdn_sz(m * cfg.gdn_heads * cfg.gdn_dv)),
                gdn_da: z(gdn_sz(m * GDN_AB_PAD)),
                gdn_db: z(gdn_sz(m * GDN_AB_PAD)),
                gdn_part: z(gdn_sz(b * cfg.gdn_heads * 2)),
                loss: z(m),
                partial: z(4096),
            };
            let decay = hk_decay_grid(cfg.heads, cfg.nphase, cfg.horizon_min, cfg.horizon_max);
            let pow = GBuf::from_slice(c, &hk_pow_table(&decay, cfg.heads, cfg.nphase));
            let gdn_decay = hk_decay_grid(1, 32, 8.0, 2048.0);
            let gdn_pow = GBuf::from_slice(c, &hk_pow_table(&gdn_decay, 1, 32));
            let gdn_lane_count = if cfg.gdn_lane {
                (0..cfg.layers).filter(|&l| !cfg.is_anchor(l)).count()
            } else {
                0
            };
            let carry_state = carry.then(|| {
                let kd = cfg.anchor_kv_heads * cfg.anchor_hd;
                let mut conv = Vec::new();
                let mut tails = Vec::new();
                let mut s0 = Vec::new();
                let mut zmax = 1usize;
                for l in 0..cfg.layers {
                    if cfg.is_anchor(l) {
                        conv.push(None);
                        s0.push(None);
                        tails.push((cp > 0).then(|| {
                            let n = b * cp * kd;
                            (z(n), z(n), z(n), z(n), z(n))
                        }));
                    } else if cfg.is_gdn_mixer() {
                        let w = cfg.gdn_c_dim();
                        let ss = cfg.gdn_dk * cfg.gdn_dv;
                        conv.push(Some((
                            z(b * (GDN_CONV_K - 1) * w),
                            z(b * (GDN_CONV_K - 1) * w),
                            GDN_CONV_K,
                            w,
                        )));
                        s0.push(Some((z(b * cfg.gdn_heads * ss), ss)));
                        tails.push(None);
                        zmax = zmax.max((GDN_CONV_K - 1) * w).max(ss);
                    } else {
                        let ps = 2 * cfg.nphase * cfg.dv;
                        conv.push((cfg.conv_k > 0).then(|| {
                            (z(b * (cfg.conv_k - 1) * h), z(b * (cfg.conv_k - 1) * h), cfg.conv_k, h)
                        }));
                        s0.push(Some((z(b * cfg.heads * ps), ps)));
                        tails.push(None);
                        zmax = zmax.max((cfg.conv_k.max(1) - 1) * h).max(ps);
                    }
                }
                CarryState {
                    cp,
                    mask: GBuf::from_u32(c, &vec![0u32; b]),
                    mask_host: vec![false; b],
                    mask_zero: GBuf::from_u32(c, &vec![0u32; b]),
                    conv,
                    tails,
                    s0,
                    zeros: z(zmax),
                }
            });
            Some(EmbryoGpu {
                carry: carry_state,
                carry_active: std::cell::Cell::new(false),
                p: GBuf::from_slice(c, params),
                g: z(lay.total),
                m: z(lay.total),
                v: z(lay.total),
                pow,
                gdn_pow,
                gdn_lane_count,
                acts,
                x_out: z(m * h),
                xf: z(m * h),
                invf: z(m),
                dxf: z(m * h),
                xft: z(m * h),
                freeze: Vec::new(),
                tok: GBuf::from_u32(c, &vec![0u32; m]),
                tgt: GBuf::from_u32(c, &vec![0u32; m]),
                tgt_cluster: GBuf::from_u32(c, &vec![0u32; m]),
                head_idx: GBuf::from_u32(c, &vec![u32::MAX; mpad.max(1)]),
                head_groups: std::cell::RefCell::new(Vec::new()),
                head_valid: std::cell::Cell::new(0),
                hg: z(mpad * h),
                dhg: z(mpad * h),
                lw: z(mpad * cs.max(1)),
                lc: z(m * ncl.max(1)),
                loss2: z(m),
                mpad,
                moe,
                desc,
                moe_cap,
                desc_updates: std::cell::Cell::new(true),
                route_frozen: std::cell::Cell::new(false),
                skill: None,
                desc_seeded: std::cell::Cell::new(false),
                desc_frozen_below: std::cell::Cell::new(0),
                bias_frozen_from: std::cell::Cell::new(usize::MAX),
                desc_seed_rows: GBuf::from_u32(
                    c,
                    &(0..ne.max(1))
                        .map(|e| ((e * 7919 + 13) % (b * t)) as u32)
                        .collect::<Vec<u32>>(),
                ),
                scratch,
                step: 0,
                head_rows,
                anchor_window_t: std::cell::Cell::new(cfg.anchor_window),
                anchor_fixed_window: std::cell::Cell::new(false),
                gdn_s0_from_ckpt: std::cell::Cell::new(false),
                #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
                gdn_wy: if cfg.is_gdn_mixer()
                    && std::env::var("CMF_GDN_WY").ok().as_deref() != Some("0")
                {
                    let d = GdnScanDims {
                        b,
                        t,
                        nv: cfg.gdn_heads,
                        dk: cfg.gdn_dk,
                        dv: cfg.gdn_dv,
                        c_dim: cfg.gdn_c_dim(),
                        ab_ld: GDN_AB_PAD,
                    };
                    let keep = (0..cfg.layers)
                        .map(|l| (!cfg.is_anchor(l)).then(|| crate::gdn_wy::GdnWyKeep::new(c, &d)))
                        .collect();
                    Some((crate::gdn_wy::GdnWyScratch::new(c, &d), keep))
                } else {
                    None
                },
                cfg,
                lay,
                b,
                t,
            })
        }

        /// Offset of layer `l`'s γ^δ table in `self.pow` (learnable decay: one
        /// table per layer, rebuilt from A_log each forward; fixed grid: one
        /// shared table at 0).
        pub fn pow_off(&self, l: usize) -> usize {
            if self.cfg.learn_decay {
                l * self.cfg.heads * 65 * 2 * self.cfg.nphase
            } else {
                0
            }
        }

        pub fn hk_scratch(&self) -> HkScratch<'_> {
            let s = &self.scratch;
            HkScratch {
                qt: &s.hk_qt,
                kt: &s.hk_kt,
                qp: &s.hk_qp,
                kh: &s.hk_kh,
                dqt: &s.hk_dqt,
                dkt: &s.hk_dkt,
                dqi: &s.hk_dqi,
                dki: &s.hk_dki,
                a: &s.hk_a,
            }
        }

        pub fn ctx(&self) -> &'static Ctx {
            ctx().expect("Metal context")
        }

        pub(crate) fn route_dims(&self) -> RouteDims {
            RouteDims {
                rows: self.b * self.t,
                h: self.cfg.hidden,
                e: self.cfg.experts,
                k: MOE_K,
                cap: self.moe_cap,
            }
        }

        /// Shared expert + routed experts (top-1 by resonance) forward;
        /// `l` indexes the layer's MoE activations and descriptors.
        #[allow(clippy::too_many_arguments)]
        fn ffn_fwd(
            &self,
            cmd: &Cmd,
            l: usize,
            m: usize,
            ffn: &FfnOffs,
            x2: &GBuf,
            gte: &GBuf,
            up: &GBuf,
            hh: &GBuf,
            x_mid: &GBuf,
            x_out: &GBuf,
            train: bool,
        ) {
            let (h, i) = (self.cfg.hidden, self.cfg.inter);
            // gate/up: [M,H]·[I,H]ᵀ
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                i,
                h,
                1.0,
                x2,
                0,
                h,
                &self.p,
                ffn.wg,
                h,
                0.0,
                gte,
                0,
                i,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                i,
                h,
                1.0,
                x2,
                0,
                h,
                &self.p,
                ffn.wu,
                h,
                0.0,
                up,
                0,
                i,
            );
            cmd.swiglu_fwd(gte, up, hh, m * i);
            if let Some(sk) = self.skill.as_ref() {
                if let Some(mi) = sk.slot(l) {
                    cmd.mask_fwd(hh, &sk.logits, mi * i, m, i, sk.hard.get(), sk.tau);
                }
            }
            // x_out = x_mid + hh·Wdᵀ  ([M,I]·[H,I]ᵀ)
            cmd.copy(x_mid, 0, x_out, 0, m * h);
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                h,
                i,
                1.0,
                hh,
                0,
                i,
                &self.p,
                ffn.wd,
                i,
                1.0,
                x_out,
                0,
                h,
            );
            let ne = self.cfg.experts;
            if ne == 0 {
                return;
            }
            // ---- routed experts ----
            let r = self.route_dims();
            let mo = &self.moe[l];
            let d = &self.desc;
            let (mu_off, u_off, e_off) = (l * ne * h, l * ne * MOE_K * h, l * ne);
            if train && self.desc_updates.get() && !self.desc_seeded.get() {
                // first training forward: seed the descriptors from this batch's
                // own rows (spread across the batch), instead of random points
                cmd.moe_init_mu(&r, x2, &self.desc_seed_rows, &d.mu, mu_off);
            }
            if !self.route_frozen.get() {
                if self.cfg.router_top2_enabled() {
                    let seq = if self.cfg.router_smooth_k4 { self.t } else { 0 };
                    let threshold = self.cfg.router_top2_margin.unwrap_or(0.0);
                    cmd.route_top2(
                        &r,
                        seq,
                        x2,
                        &d.mu,
                        mu_off,
                        &d.u,
                        u_off,
                        &d.bias,
                        e_off,
                        threshold,
                        &mo.assign,
                        &mo.runner,
                        &mo.margin,
                        &mo.runner_weight,
                        &mo.res,
                        &mo.fallback_count,
                    );
                } else if self.cfg.router_smooth_k4 {
                    cmd.route_smooth_k4(
                        &r, self.t, x2, &d.mu, mu_off, &d.u, u_off, &d.bias, e_off, &mo.assign,
                        &mo.res,
                    );
                } else {
                    cmd.route(
                        &r, x2, &d.mu, mu_off, &d.u, u_off, &d.bias, e_off, &mo.assign, &mo.res,
                    );
                }
                cmd.route_group(&r, &mo.assign, &mo.slot, &d.count, e_off);
                if self.cfg.router_top2_enabled() {
                    cmd.route_group(&r, &mo.runner, &mo.slot2, &mo.count2, 0);
                }
            }
            if self.cfg.router_top2_enabled() {
                // Both experts receive the original normalized token. The
                // fixed blend weights apply only to their residual outputs;
                // keeping `mo.hg` unscaled also preserves top1-only
                // descriptor statistics exactly.
                cmd.moe_gather(&r, x2, &mo.assign, &mo.slot, &mo.hg);
                cmd.moe_gather(&r, x2, &mo.runner, &mo.slot2, &mo.hg2);
            } else {
                cmd.moe_gather(&r, x2, &mo.assign, &mo.slot, &mo.hg);
            }
            let cap = self.moe_cap;
            let ew = 3 * h * i; // expert weight stride
                                // per-expert GEMMs over the FILLED rows only: grids come from
                                // count[e] on the GPU (indirect dispatch), no host readback
            let ind_off = l * 2 * ne * 3;
            cmd.moe_indirect_args(&d.count, e_off, &d.indir, ind_off, ne, cap, i, h);
            for e in 0..ne {
                let ind_i = GemmDyn {
                    indirect: Some((&d.indir, (ind_off + e * 3) * 4)),
                    kcount: None,
                };
                let (wg, wu) = (ffn.experts + e * ew, ffn.experts + e * ew + h * i);
                let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                cmd.gemm_dyn(
                    Op::N,
                    Op::T,
                    cap,
                    i,
                    h,
                    1.0,
                    &mo.hg,
                    hg_o,
                    h,
                    &self.p,
                    wg,
                    h,
                    0.0,
                    &mo.gte,
                    gi_o,
                    i,
                    &GemmBatch::none(),
                    false,
                    &ind_i,
                );
                cmd.gemm_dyn(
                    Op::N,
                    Op::T,
                    cap,
                    i,
                    h,
                    1.0,
                    &mo.hg,
                    hg_o,
                    h,
                    &self.p,
                    wu,
                    h,
                    0.0,
                    &mo.up,
                    gi_o,
                    i,
                    &GemmBatch::none(),
                    false,
                    &ind_i,
                );
            }
            if self.cfg.router_top2_enabled() {
                // Runner-up stream: same expert weights, separate bounded
                // activation/dispatch buffers. Its fixed 0.5 row weight is
                // applied during the final scatter below.
                let ind2_off = 0usize;
                cmd.moe_indirect_args(&mo.count2, 0, &mo.indir2, ind2_off, ne, cap, i, h);
                for e in 0..ne {
                    let ind_i = GemmDyn {
                        indirect: Some((&mo.indir2, (ind2_off + e * 3) * 4)),
                        kcount: None,
                    };
                    let (wg, wu) = (ffn.experts + e * ew, ffn.experts + e * ew + h * i);
                    let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                    cmd.gemm_dyn(
                        Op::N,
                        Op::T,
                        cap,
                        i,
                        h,
                        1.0,
                        &mo.hg2,
                        hg_o,
                        h,
                        &self.p,
                        wg,
                        h,
                        0.0,
                        &mo.gte2,
                        gi_o,
                        i,
                        &GemmBatch::none(),
                        false,
                        &ind_i,
                    );
                    cmd.gemm_dyn(
                        Op::N,
                        Op::T,
                        cap,
                        i,
                        h,
                        1.0,
                        &mo.hg2,
                        hg_o,
                        h,
                        &self.p,
                        wu,
                        h,
                        0.0,
                        &mo.up2,
                        gi_o,
                        i,
                        &GemmBatch::none(),
                        false,
                        &ind_i,
                    );
                }
                cmd.swiglu_fwd(&mo.gte2, &mo.up2, &mo.hh2, ne * cap * i);
                for e in 0..ne {
                    let ind_h = GemmDyn {
                        indirect: Some((&mo.indir2, (ind2_off + (ne + e) * 3) * 4)),
                        kcount: None,
                    };
                    let wd = ffn.experts + e * ew + 2 * h * i;
                    let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                    cmd.gemm_dyn(
                        Op::N,
                        Op::T,
                        cap,
                        h,
                        i,
                        1.0,
                        &mo.hh2,
                        gi_o,
                        i,
                        &self.p,
                        wd,
                        i,
                        0.0,
                        &mo.yh2,
                        hg_o,
                        h,
                        &GemmBatch::none(),
                        false,
                        &ind_h,
                    );
                }
            }
            cmd.swiglu_fwd(&mo.gte, &mo.up, &mo.hh, ne * cap * i);
            for e in 0..ne {
                let ind_h = GemmDyn {
                    indirect: Some((&d.indir, (ind_off + (ne + e) * 3) * 4)),
                    kcount: None,
                };
                let wd = ffn.experts + e * ew + 2 * h * i;
                let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                cmd.gemm_dyn(
                    Op::N,
                    Op::T,
                    cap,
                    h,
                    i,
                    1.0,
                    &mo.hh,
                    gi_o,
                    i,
                    &self.p,
                    wd,
                    i,
                    0.0,
                    &mo.yh,
                    hg_o,
                    h,
                    &GemmBatch::none(),
                    false,
                    &ind_h,
                );
            }
            if self.cfg.router_top2_enabled() {
                cmd.moe_scatter_add_weighted(
                    &r,
                    x_out,
                    &mo.assign,
                    &mo.slot,
                    &mo.runner_weight,
                    &mo.yh,
                    true,
                );
                cmd.moe_scatter_add_weighted(
                    &r,
                    x_out,
                    &mo.runner,
                    &mo.slot2,
                    &mo.runner_weight,
                    &mo.yh2,
                    false,
                );
            } else {
                cmd.moe_scatter_add(&r, x_out, &mo.assign, &mo.slot, &mo.yh);
            }
            if train && self.desc_updates.get() {
                // descriptor statistics → μ EMA + balancing bias
                cmd.moe_stats(&r, &mo.hg, &d.count, e_off, &d.sums, mu_off);
                cmd.moe_update(
                    &r,
                    &d.mu,
                    mu_off,
                    &d.bias,
                    e_off,
                    &d.sums,
                    mu_off,
                    &d.count,
                    e_off,
                    &mo.res,
                    MOE_ALPHA,
                    MOE_ETA,
                    self.desc_frozen_below.get(),
                    self.bias_frozen_from.get(),
                );
            }
        }

        /// FFN backward: dx_out (in s.dx) → accumulates dx_mid into s.dx via
        /// the ln2 backward; weight grads into g.
        #[allow(clippy::too_many_arguments)]
        fn ffn_bwd(
            &self,
            cmd: &Cmd,
            l: usize,
            m: usize,
            ffn: &FfnOffs,
            ln2: usize,
            x_mid: &GBuf,
            x2: &GBuf,
            inv2: &GBuf,
            gte: &GBuf,
            up: &GBuf,
            hh: &GBuf,
        ) {
            let s = &self.scratch;
            let (h, i) = (self.cfg.hidden, self.cfg.inter);
            // dhh = dx·Wd  ([M,H]·[H,I])
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                i,
                h,
                1.0,
                &s.dx,
                0,
                h,
                &self.p,
                ffn.wd,
                i,
                0.0,
                &s.dffn,
                0,
                i,
            );
            // dWd += dxᵀ·hh  ([H,M]·[M,I])
            cmd.gemm(
                Op::T,
                Op::N,
                h,
                i,
                m,
                1.0,
                &s.dx,
                0,
                h,
                hh,
                0,
                i,
                1.0,
                &self.g,
                ffn.wd,
                i,
            );
            if let Some(sk) = self.skill.as_ref() {
                if let Some(mi) = sk.slot(l) {
                    // pre-mask activation recomputed; dm from the masked-input grad
                    cmd.swiglu_fwd(gte, up, &s.hh_pre, m * i);
                    cmd.mask_bwd(
                        &s.dffn,
                        &s.hh_pre,
                        &sk.logits,
                        mi * i,
                        &sk.g,
                        m,
                        i,
                        sk.hard.get(),
                        sk.tau,
                        sk.l1.get(),
                    );
                }
            }
            cmd.swiglu_bwd(gte, up, &s.dffn, &s.dgte, &s.dup, m * i);
            // dx2 = dgte·Wg + dup·Wu
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                i,
                1.0,
                &s.dgte,
                0,
                i,
                &self.p,
                ffn.wg,
                h,
                0.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                i,
                1.0,
                &s.dup,
                0,
                i,
                &self.p,
                ffn.wu,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            // dWg += dgteᵀ·x2 ; dWu += dupᵀ·x2
            cmd.gemm(
                Op::T,
                Op::N,
                i,
                h,
                m,
                1.0,
                &s.dgte,
                0,
                i,
                x2,
                0,
                h,
                1.0,
                &self.g,
                ffn.wg,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                i,
                h,
                m,
                1.0,
                &s.dup,
                0,
                i,
                x2,
                0,
                h,
                1.0,
                &self.g,
                ffn.wu,
                h,
            );
            let ne = self.cfg.experts;
            if ne > 0 {
                // ---- routed experts: same graph on the gathered slots ----
                let r = self.route_dims();
                let mo = &self.moe[l];
                let cap = self.moe_cap;
                let ew = 3 * h * i;
                // dyh[e][slot] = weighted dx_out[row].  The legacy path uses
                // the exact unweighted gather; top-2 splits the residual
                // gradient between the primary and runner streams.
                if self.cfg.router_top2_enabled() {
                    cmd.moe_gather_weighted(
                        &r,
                        &s.dx,
                        &mo.assign,
                        &mo.slot,
                        &mo.runner_weight,
                        &s.moe_dyh,
                        true,
                    );
                } else {
                    cmd.moe_gather(&r, &s.dx, &mo.assign, &mo.slot, &s.moe_dyh);
                }
                let ind_off = l * 2 * ne * 3;
                let d = &self.desc;
                for e in 0..ne {
                    let ind_i = GemmDyn {
                        indirect: Some((&d.indir, (ind_off + e * 3) * 4)),
                        kcount: None,
                    };
                    let kdyn = GemmDyn {
                        indirect: None,
                        kcount: Some((&d.count, l * ne + e)),
                    };
                    let wd = ffn.experts + e * ew + 2 * h * i;
                    let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                    // dhh = dyh·Wd_e ([rows,H]·[H,I]);  dWd_e += dyhᵀ·hh (K = rows)
                    cmd.gemm_dyn(
                        Op::N,
                        Op::N,
                        cap,
                        i,
                        h,
                        1.0,
                        &s.moe_dyh,
                        hg_o,
                        h,
                        &self.p,
                        wd,
                        i,
                        0.0,
                        &s.moe_dffn,
                        gi_o,
                        i,
                        &GemmBatch::none(),
                        false,
                        &ind_i,
                    );
                    cmd.gemm_dyn(
                        Op::T,
                        Op::N,
                        h,
                        i,
                        cap,
                        1.0,
                        &s.moe_dyh,
                        hg_o,
                        h,
                        &mo.hh,
                        gi_o,
                        i,
                        1.0,
                        &self.g,
                        wd,
                        i,
                        &GemmBatch::none(),
                        false,
                        &kdyn,
                    );
                }
                cmd.swiglu_bwd(
                    &mo.gte,
                    &mo.up,
                    &s.moe_dffn,
                    &s.moe_dgte,
                    &s.moe_dup,
                    ne * cap * i,
                );
                for e in 0..ne {
                    let ind_h = GemmDyn {
                        indirect: Some((&d.indir, (ind_off + (ne + e) * 3) * 4)),
                        kcount: None,
                    };
                    let kdyn = GemmDyn {
                        indirect: None,
                        kcount: Some((&d.count, l * ne + e)),
                    };
                    let (wg, wu) = (ffn.experts + e * ew, ffn.experts + e * ew + h * i);
                    let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                    // dhg = dgte·Wg_e + dup·Wu_e
                    cmd.gemm_dyn(
                        Op::N,
                        Op::N,
                        cap,
                        h,
                        i,
                        1.0,
                        &s.moe_dgte,
                        gi_o,
                        i,
                        &self.p,
                        wg,
                        h,
                        0.0,
                        &s.moe_dhg,
                        hg_o,
                        h,
                        &GemmBatch::none(),
                        false,
                        &ind_h,
                    );
                    cmd.gemm_dyn(
                        Op::N,
                        Op::N,
                        cap,
                        h,
                        i,
                        1.0,
                        &s.moe_dup,
                        gi_o,
                        i,
                        &self.p,
                        wu,
                        h,
                        1.0,
                        &s.moe_dhg,
                        hg_o,
                        h,
                        &GemmBatch::none(),
                        false,
                        &ind_h,
                    );
                    // dWg_e += dgteᵀ·hg ; dWu_e += dupᵀ·hg  (K = rows)
                    cmd.gemm_dyn(
                        Op::T,
                        Op::N,
                        i,
                        h,
                        cap,
                        1.0,
                        &s.moe_dgte,
                        gi_o,
                        i,
                        &mo.hg,
                        hg_o,
                        h,
                        1.0,
                        &self.g,
                        wg,
                        h,
                        &GemmBatch::none(),
                        false,
                        &kdyn,
                    );
                    cmd.gemm_dyn(
                        Op::T,
                        Op::N,
                        i,
                        h,
                        cap,
                        1.0,
                        &s.moe_dup,
                        gi_o,
                        i,
                        &mo.hg,
                        hg_o,
                        h,
                        1.0,
                        &self.g,
                        wu,
                        h,
                        &GemmBatch::none(),
                        false,
                        &kdyn,
                    );
                }
                // dx2 += scatter(dhg)
                cmd.moe_scatter_add(&r, &s.dx2, &mo.assign, &mo.slot, &s.moe_dhg);

                if self.cfg.router_top2_enabled() {
                    // Runner-up stream. Scratch is intentionally reused only
                    // after the primary stream has accumulated its parameter
                    // and input gradients; activations remain in `mo.*2` for
                    // the whole backward.
                    cmd.moe_gather_weighted(
                        &r,
                        &s.dx,
                        &mo.runner,
                        &mo.slot2,
                        &mo.runner_weight,
                        &s.moe_dyh,
                        false,
                    );
                    for e in 0..ne {
                        let ind_i = GemmDyn {
                            indirect: Some((&mo.indir2, e * 3 * 4)),
                            kcount: None,
                        };
                        let kdyn = GemmDyn {
                            indirect: None,
                            kcount: Some((&mo.count2, e)),
                        };
                        let wd = ffn.experts + e * ew + 2 * h * i;
                        let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                        cmd.gemm_dyn(
                            Op::N,
                            Op::N,
                            cap,
                            i,
                            h,
                            1.0,
                            &s.moe_dyh,
                            hg_o,
                            h,
                            &self.p,
                            wd,
                            i,
                            0.0,
                            &s.moe_dffn,
                            gi_o,
                            i,
                            &GemmBatch::none(),
                            false,
                            &ind_i,
                        );
                        cmd.gemm_dyn(
                            Op::T,
                            Op::N,
                            h,
                            i,
                            cap,
                            1.0,
                            &s.moe_dyh,
                            hg_o,
                            h,
                            &mo.hh2,
                            gi_o,
                            i,
                            1.0,
                            &self.g,
                            wd,
                            i,
                            &GemmBatch::none(),
                            false,
                            &kdyn,
                        );
                    }
                    cmd.swiglu_bwd(
                        &mo.gte2,
                        &mo.up2,
                        &s.moe_dffn,
                        &s.moe_dgte,
                        &s.moe_dup,
                        ne * cap * i,
                    );
                    for e in 0..ne {
                        let ind_h = GemmDyn {
                            indirect: Some((&mo.indir2, (ne + e) * 3 * 4)),
                            kcount: None,
                        };
                        let kdyn = GemmDyn {
                            indirect: None,
                            kcount: Some((&mo.count2, e)),
                        };
                        let (wg, wu) = (ffn.experts + e * ew, ffn.experts + e * ew + h * i);
                        let (hg_o, gi_o) = (e * cap * h, e * cap * i);
                        cmd.gemm_dyn(
                            Op::N,
                            Op::N,
                            cap,
                            h,
                            i,
                            1.0,
                            &s.moe_dgte,
                            gi_o,
                            i,
                            &self.p,
                            wg,
                            h,
                            0.0,
                            &s.moe_dhg,
                            hg_o,
                            h,
                            &GemmBatch::none(),
                            false,
                            &ind_h,
                        );
                        cmd.gemm_dyn(
                            Op::N,
                            Op::N,
                            cap,
                            h,
                            i,
                            1.0,
                            &s.moe_dup,
                            gi_o,
                            i,
                            &self.p,
                            wu,
                            h,
                            1.0,
                            &s.moe_dhg,
                            hg_o,
                            h,
                            &GemmBatch::none(),
                            false,
                            &ind_h,
                        );
                        cmd.gemm_dyn(
                            Op::T,
                            Op::N,
                            i,
                            h,
                            cap,
                            1.0,
                            &s.moe_dgte,
                            gi_o,
                            i,
                            &mo.hg2,
                            hg_o,
                            h,
                            1.0,
                            &self.g,
                            wg,
                            h,
                            &GemmBatch::none(),
                            false,
                            &kdyn,
                        );
                        cmd.gemm_dyn(
                            Op::T,
                            Op::N,
                            i,
                            h,
                            cap,
                            1.0,
                            &s.moe_dup,
                            gi_o,
                            i,
                            &mo.hg2,
                            hg_o,
                            h,
                            1.0,
                            &self.g,
                            wu,
                            h,
                            &GemmBatch::none(),
                            false,
                            &kdyn,
                        );
                    }
                    cmd.moe_scatter_add(&r, &s.dx2, &mo.runner, &mo.slot2, &s.moe_dhg);
                }
            }
            // dx_mid = dx_out + rmsnorm_bwd(x_mid; dx2)  (accumulate into s.dx)
            cmd.rmsnorm_bwd_at(
                x_mid, &self.p, ln2, &s.dx2, inv2, &s.dx, 1.0, &self.g, ln2, m, h,
            );
        }

        /// Forward of the optional one-head ordinary GDN correction lane.
        #[allow(clippy::too_many_arguments)]
        fn gdn_lane_fwd(
            &self,
            cmd: &Cmd,
            l: usize,
            xs: &GBuf,
            x_mid: &GBuf,
            gdn_qfull: &GBuf,
            gdn_kfull: &GBuf,
            gdn_v: &GBuf,
            gdn_ab: &GBuf,
            gdn_kappa: &GBuf,
            gdn_phq: &GBuf,
            gdn_phk: &GBuf,
            gdn_kv: &GBuf,
            gdn_states: &GBuf,
            gdn_raw_o: &GBuf,
            gdn_inv: &GBuf,
            gdn_o: &GBuf,
            gdn_z: &GBuf,
            gdn_gated: &GBuf,
            gdn_d: &GBuf,
        ) {
            let Some(go) = self.lay.gdn.get(l).and_then(|x| x.as_ref()) else {
                return;
            };
            let (m, h) = (self.b * self.t, self.cfg.hidden);
            // Fused qkvz rows: q[0..64], k[64..128], v[128..192], z[192..256].
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                64,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.qkvz,
                h,
                0.0,
                gdn_qfull,
                0,
                64,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                64,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.qkvz + 64 * h,
                h,
                0.0,
                gdn_kfull,
                0,
                64,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                64,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.qkvz + 128 * h,
                h,
                0.0,
                gdn_v,
                0,
                64,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                64,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.qkvz + 192 * h,
                h,
                0.0,
                gdn_z,
                0,
                64,
            );
            // Padded [64,H] control projection; rows 0/1 are decay and beta.
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                64,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.ab,
                h,
                0.0,
                gdn_ab,
                0,
                64,
            );
            if std::env::var_os("CMF_GDN_SERIAL").is_some() {
                cmd.gdn_forward(
                    gdn_qfull,
                    gdn_kfull,
                    gdn_v,
                    gdn_z,
                    gdn_ab,
                    &self.p,
                    go.conv,
                    go.norm,
                    go.alog,
                    go.dt_bias,
                    gdn_phq,
                    gdn_phk,
                    gdn_kv,
                    gdn_kappa,
                    gdn_raw_o,
                    gdn_inv,
                    gdn_o,
                    gdn_states,
                    self.b,
                    self.t,
                    self.cfg.norm_eps,
                );
            } else {
                cmd.gdn_forward_parallel(
                    gdn_qfull,
                    gdn_kfull,
                    gdn_v,
                    gdn_z,
                    gdn_ab,
                    &self.p,
                    go.conv,
                    go.norm,
                    go.alog,
                    go.dt_bias,
                    gdn_phq,
                    gdn_phk,
                    gdn_kv,
                    gdn_kappa,
                    gdn_raw_o,
                    gdn_inv,
                    gdn_o,
                    gdn_states,
                    self.b,
                    self.t,
                    self.cfg.norm_eps,
                );
            }
            // Gated output (`silu(z)·RMSNorm(o)`) and q4tp-eligible Wo.
            cmd.swiglu_fwd(gdn_z, gdn_o, gdn_gated, m * 64);
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                h,
                64,
                1.0,
                gdn_gated,
                0,
                64,
                &self.p,
                go.wo,
                64,
                0.0,
                gdn_d,
                0,
                h,
            );
            let gain = unsafe { host_slice(&self.p) }[go.gain];
            if gain != 0.0 {
                cmd.axpby(gain, gdn_d, 1.0, x_mid, m * h);
            }
        }

        /// Forward of the optional layer-4 additive GQA donor lane.  The
        /// original hybrid mixer output is already present in `x_mid`; this
        /// lane computes ordinary causal GQA from the same normalized input
        /// and adds it through its append-only output projection.
        fn gqa_lane_fwd(
            &self,
            cmd: &Cmd,
            l: usize,
            xs: &GBuf,
            x_mid: &GBuf,
            q: &GBuf,
            k: &GBuf,
            v: &GBuf,
            p: &GBuf,
            o: &GBuf,
        ) {
            let Some(go) = self.lay.gqa.get(l).and_then(|x| x.as_ref()) else {
                return;
            };
            let (b, t, h) = (self.b, self.t, self.cfg.hidden);
            let (qh, kvh, hd) = (
                self.cfg.anchor_q_heads,
                self.cfg.anchor_kv_heads,
                self.cfg.anchor_hd,
            );
            let (qd, kd) = (qh * hd, kvh * hd);
            cmd.gemm(
                Op::N,
                Op::T,
                b * t,
                qd,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.q,
                h,
                0.0,
                q,
                0,
                qd,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                b * t,
                kd,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.k,
                h,
                0.0,
                k,
                0,
                kd,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                b * t,
                kd,
                h,
                1.0,
                xs,
                0,
                h,
                &self.p,
                go.v,
                h,
                0.0,
                v,
                0,
                kd,
            );
            cmd.rope(q, 0, b * t, t, qh, hd, self.cfg.rope_base, false);
            cmd.rope(k, 0, b * t, t, kvh, hd, self.cfg.rope_base, false);
            let group = qh / kvh;
            let scale = 1.0 / (hd as f32).sqrt();
            let sq = [t * qd, group * hd, hd];
            let sk = [t * kd, hd, 0];
            let sp = [qh * t * t, group * t * t, t * t];
            let bt = GemmBatch {
                nb: b,
                nh: kvh,
                nc: group,
                sa: sq,
                sb: sk,
                sc: sp,
            };
            cmd.gemm_ex(
                Op::N,
                Op::T,
                t,
                t,
                hd,
                scale,
                q,
                0,
                qd,
                k,
                0,
                kd,
                0.0,
                p,
                0,
                t,
                &bt,
                false,
            );
            cmd.causal_softmax_blocks(p, 0, t, b * qh);
            let bt = GemmBatch {
                nb: b,
                nh: kvh,
                nc: group,
                sa: sp,
                sb: sk,
                sc: sq,
            };
            cmd.gemm_ex(
                Op::N,
                Op::N,
                t,
                hd,
                t,
                1.0,
                p,
                0,
                t,
                v,
                0,
                kd,
                0.0,
                o,
                0,
                qd,
                &bt,
                false,
            );
            cmd.gemm(
                Op::N,
                Op::T,
                b * t,
                h,
                qd,
                1.0,
                o,
                0,
                qd,
                &self.p,
                go.wo,
                qd,
                1.0,
                x_mid,
                0,
                h,
            );
        }

        /// Backward of the layer-4 additive GQA donor lane.  Gradients are
        /// accumulated into the append-only q/k/v/o tail and into the
        /// existing mixer input gradient; the legacy hybrid path remains in
        /// `s.dx1` and is never replaced.
        fn gqa_lane_bwd(
            &self,
            cmd: &Cmd,
            l: usize,
            xs: &GBuf,
            q: &GBuf,
            k: &GBuf,
            v: &GBuf,
            p: &GBuf,
            o: &GBuf,
        ) {
            let Some(go) = self.lay.gqa.get(l).and_then(|x| x.as_ref()) else {
                return;
            };
            let (b, t, h) = (self.b, self.t, self.cfg.hidden);
            let (qh, kvh, hd) = (
                self.cfg.anchor_q_heads,
                self.cfg.anchor_kv_heads,
                self.cfg.anchor_hd,
            );
            let (qd, kd) = (qh * hd, kvh * hd);
            let s = &self.scratch;
            // do = dX_mid·Wo; dWo += dX_midᵀ·O.  With Wo=0, q/k/v grads are
            // intentionally zero while the output projection receives an
            // unsuppressed first-step gradient.
            cmd.gemm(
                Op::N,
                Op::N,
                b * t,
                qd,
                h,
                1.0,
                &s.dx,
                0,
                h,
                &self.p,
                go.wo,
                qd,
                0.0,
                &s.dbig,
                0,
                qd,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                h,
                qd,
                b * t,
                1.0,
                &s.dx,
                0,
                h,
                o,
                0,
                qd,
                1.0,
                &self.g,
                go.wo,
                qd,
            );
            let scale = 1.0 / (hd as f32).sqrt();
            let group = qh / kvh;
            let sq = [0, group * hd, hd];
            let sk = [0, hd, 0];
            let sp = [0, group * t * t, t * t];
            let sh = [0, group * t * hd, t * hd];
            for bi in 0..b {
                let q_off = bi * t * qd;
                let kv_off = bi * t * kd;
                let p_off = bi * qh * t * t;
                let h_off = bi * qh * t * hd;
                let bt = GemmBatch {
                    nb: 1,
                    nh: kvh,
                    nc: group,
                    sa: sq,
                    sb: sk,
                    sc: sp,
                };
                cmd.gemm_ex(
                    Op::N,
                    Op::T,
                    t,
                    t,
                    hd,
                    1.0,
                    &s.dbig,
                    q_off,
                    qd,
                    v,
                    kv_off,
                    kd,
                    0.0,
                    &s.dp,
                    0,
                    t,
                    &bt,
                    false,
                );
                let bt = GemmBatch {
                    nb: 1,
                    nh: kvh,
                    nc: group,
                    sa: sp,
                    sb: sq,
                    sc: sh,
                };
                cmd.gemm_ex(
                    Op::T,
                    Op::N,
                    t,
                    hd,
                    t,
                    1.0,
                    p,
                    p_off,
                    t,
                    &s.dbig,
                    q_off,
                    qd,
                    0.0,
                    &s.dvh,
                    h_off,
                    hd,
                    &bt,
                    false,
                );
                cmd.softmax_bwd_blocks(p, p_off, &s.dp, 0, t, qh);
                let bt = GemmBatch {
                    nb: 1,
                    nh: kvh,
                    nc: group,
                    sa: sp,
                    sb: sk,
                    sc: sq,
                };
                cmd.gemm_ex(
                    Op::N,
                    Op::N,
                    t,
                    hd,
                    t,
                    scale,
                    &s.dp,
                    0,
                    t,
                    k,
                    kv_off,
                    kd,
                    0.0,
                    &s.dq,
                    q_off,
                    qd,
                    &bt,
                    false,
                );
                let bt = GemmBatch {
                    nb: 1,
                    nh: kvh,
                    nc: group,
                    sa: sp,
                    sb: sq,
                    sc: sh,
                };
                cmd.gemm_ex(
                    Op::T,
                    Op::N,
                    t,
                    hd,
                    t,
                    scale,
                    &s.dp,
                    0,
                    t,
                    q,
                    q_off,
                    qd,
                    0.0,
                    &s.dkh,
                    h_off,
                    hd,
                    &bt,
                    false,
                );
            }
            cmd.group_sum_heads(&s.dkh, &s.dk, b, t, qh, kvh, hd);
            cmd.group_sum_heads(&s.dvh, &s.dv, b, t, qh, kvh, hd);
            cmd.rope(&s.dq, 0, b * t, t, qh, hd, self.cfg.rope_base, true);
            cmd.rope(&s.dk, 0, b * t, t, kvh, hd, self.cfg.rope_base, true);
            // Input gradient of this lane is accumulated into the existing
            // mixer projection gradient, preserving the original hybrid path.
            cmd.gemm(
                Op::N,
                Op::N,
                b * t,
                h,
                qd,
                1.0,
                &s.dq,
                0,
                qd,
                &self.p,
                go.q,
                h,
                0.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                b * t,
                h,
                kd,
                1.0,
                &s.dk,
                0,
                kd,
                &self.p,
                go.k,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                b * t,
                h,
                kd,
                1.0,
                &s.dv,
                0,
                kd,
                &self.p,
                go.v,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                qd,
                h,
                b * t,
                1.0,
                &s.dq,
                0,
                qd,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.q,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                kd,
                h,
                b * t,
                1.0,
                &s.dk,
                0,
                kd,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.k,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                kd,
                h,
                b * t,
                1.0,
                &s.dv,
                0,
                kd,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.v,
                h,
            );
            cmd.axpby(1.0, &s.dx2, 1.0, &s.dx1, b * t * h);
        }

        /// The anchor window of a forward: evaluation always serves
        /// `cfg.anchor_window`; a training forward uses the window sampled
        /// by [`EmbryoGpu::begin_train_pass`] (0 = legacy full causal).
        /// Carried-anchor columns of the score rows (0 without carry).
        pub fn carry_pad(&self) -> usize {
            self.carry.as_ref().map_or(0, |c| c.cp)
        }

        /// Start a carried window: `reset[b]` = this sequence begins a new
        /// stream (its S_0, conv history and anchor keys are cleared and the
        /// carried block is masked); the other rows continue from the state
        /// left by the previous `carry_commit`.
        pub fn carry_begin(&mut self, reset: &[bool]) {
            let b = self.b;
            assert_eq!(reset.len(), b);
            let Some(cs) = self.carry.as_mut() else {
                panic!("carry_begin on a model built without carry (EmbryoGpu::new_carry)")
            };
            for (m, r) in cs.mask_host.iter_mut().zip(reset) {
                *m = !*r;
            }
            let mask: Vec<u32> = cs.mask_host.iter().map(|&m| m as u32).collect();
            unsafe {
                std::ptr::copy_nonoverlapping(mask.as_ptr(), cs.mask.buf.contents() as *mut u32, b);
            }
            let cs = self.carry.as_ref().unwrap();
            let cmd = Cmd::new(self.ctx());
            let nch = self.t / 64;
            for l in 0..self.cfg.layers {
                // checkpoint slot 0 of every (b, head) ← the carried store,
                // zeros for the rows that start a fresh stream (mask = 0)
                let (states, heads) = match &self.acts[l] {
                    LayerActs::Mixer { states, .. } => (states, self.cfg.heads),
                    LayerActs::Gdn { states, .. } => (states, self.cfg.gdn_heads),
                    LayerActs::Anchor { .. } => continue,
                };
                let Some((store, ps)) = &cs.s0[l] else { continue };
                cmd.block_copy(store, 0, *ps, states, 0, (nch + 1) * ps, b * heads, *ps, Some((&cs.mask, heads)));
            }
            for l in 0..self.cfg.layers {
                // conv history: keep for continuing rows, zero for fresh ones
                if let Some((cur, _, k, w)) = &cs.conv[l] {
                    let n = (k - 1) * w;
                    cmd.block_copy(cur, 0, n, cur, 0, n, b, n, Some((&cs.mask, 1)));
                }
            }
            cmd.commit();
            self.carry_active.set(true);
        }

        /// Recurrent-state statistics of the carried store (after
        /// `carry_commit`): per recurrent layer `(rms over all (b, head),
        /// max |s|, per-row rms [B])`; anchors give None.
        pub fn carry_state_stats(&self) -> Vec<Option<(f32, f32, Vec<f32>)>> {
            let Some(cs) = self.carry.as_ref() else { return Vec::new() };
            let mut out = Vec::with_capacity(self.cfg.layers);
            for l in 0..self.cfg.layers {
                let Some((store, ps)) = &cs.s0[l] else {
                    out.push(None);
                    continue;
                };
                let heads = match &self.acts[l] {
                    LayerActs::Mixer { .. } => self.cfg.heads,
                    LayerActs::Gdn { .. } => self.cfg.gdn_heads,
                    LayerActs::Anchor { .. } => 0,
                };
                let x = store.to_vec();
                let n = self.b * heads * ps;
                let mut ss = 0.0f64;
                let mut mx = 0.0f32;
                let mut rows = vec![0.0f32; self.b];
                for bi in 0..self.b {
                    let seg = &x[bi * heads * ps..(bi + 1) * heads * ps];
                    let s2: f64 = seg.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                    rows[bi] = (s2 / seg.len().max(1) as f64).sqrt() as f32;
                    ss += s2;
                    mx = mx.max(seg.iter().fold(0.0f32, |m, v| m.max(v.abs())));
                }
                out.push(Some(((ss / n.max(1) as f64).sqrt() as f32, mx, rows)));
            }
            out
        }

        /// End a carried window: the final recurrent states, conv tails and
        /// anchor tails written during the pass become the next window's
        /// carried inputs. Idempotent-safe: call exactly once per window.
        pub fn carry_commit(&self) {
            let Some(cs) = self.carry.as_ref() else {
                panic!("carry_commit on a model built without carry")
            };
            assert!(self.carry_active.get(), "carry_commit without carry_begin");
            let cmd = Cmd::new(self.ctx());
            let nch = self.t / 64;
            for l in 0..self.cfg.layers {
                // final boundary state (slot nch) → the carried store; the
                // store survives the held-out evaluations that rewrite `states`
                let (states, heads) = match &self.acts[l] {
                    LayerActs::Mixer { states, .. } => (states, self.cfg.heads),
                    LayerActs::Gdn { states, .. } => (states, self.cfg.gdn_heads),
                    LayerActs::Anchor { .. } => (&self.x_out, 0),
                };
                if let Some((store, ps)) = &cs.s0[l] {
                    cmd.block_copy(states, nch * ps, (nch + 1) * ps, store, 0, *ps, self.b * heads, *ps, None);
                }
                if let Some((cur, next, k, w)) = &cs.conv[l] {
                    cmd.copy(next, 0, cur, 0, self.b * (k - 1) * w);
                }
                if let Some((kc, vc, kn, vn, _)) = &cs.tails[l] {
                    let n = self.b * cs.cp * self.cfg.anchor_kv_heads * self.cfg.anchor_hd;
                    cmd.copy(kn, 0, kc, 0, n);
                    cmd.copy(vn, 0, vc, 0, n);
                }
            }
            cmd.commit();
            self.carry_active.set(false);
        }

        /// Copy the last `rows` rows of every sequence of `src` [B·T, w]
        /// into `dst` [B, rows, w] (the next window's carried history).
        fn carry_tail_copy(&self, cmd: &Cmd, src: &GBuf, dst: &GBuf, rows: usize, w: usize) {
            for bi in 0..self.b {
                cmd.copy(src, (bi * self.t + self.t - rows) * w, dst, bi * rows * w, rows * w);
            }
        }

        pub fn anchor_window_for(&self, train: bool) -> usize {
            if train {
                self.anchor_window_t.get()
            } else {
                self.cfg.anchor_window
            }
        }

        /// Draw this training pass's anchor window `W_t` (SWAX): a
        /// deterministic function of the optimizer step, unless the served
        /// window is forced. Called once at the start of every training
        /// forward; the backward of the same pass reads the same value.
        pub fn begin_train_pass(&self) {
            let w = if self.anchor_fixed_window.get() {
                self.cfg.anchor_window
            } else {
                self.cfg.anchor_window_at_step(self.step as u64)
            };
            self.anchor_window_t.set(w);
        }

        /// Attention core of an anchor (legacy full-causal or bounded
        /// `swa_sink_v1`), from the RAW projections in `q`/`k`/`v` to `o`:
        /// keeps q̂ in `q_raw` (NoPE sink scores), fills the padded sink
        /// tiles from the arena, rotates q/k in place by their absolute
        /// positions (the band makes the served operator relative, see
        /// docs/EMBRYO_BOUNDED_ANCHOR.md §1), then
        ///   P[:, 0..S]       = q̂·sink_kᵀ·scale        (columns S..SINK_PAD: 0)
        ///   P[:, SP..SP+T]   = Q_rot·K_rotᵀ·scale
        ///   P = band+sink softmax over the [T, SP+T] rows
        ///   O = P[:, SP..]·V + P[:, 0..SP]·sink_v_pad
        /// With `anchor_sink == 0` the row length is T and every call below
        /// is exactly the legacy dispatch sequence.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn anchor_attn_fwd(
            &self,
            cmd: &Cmd,
            layer: usize,
            q: &GBuf,
            k: &GBuf,
            v: &GBuf,
            p: &GBuf,
            o: &GBuf,
            q_raw: &GBuf,
            sink_k_pad: &GBuf,
            sink_v_pad: &GBuf,
            sink_k: usize,
            sink_v: usize,
            window: usize,
        ) {
            let cfg = &self.cfg;
            let (b, t) = (self.b, self.t);
            let m = b * t;
            let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
            let (qd, kd) = (qh * hd, kvh * hd);
            let sk = if cfg.anchor_bounded() { cfg.anchor_sink } else { 0 };
            let sp = cfg.sink_pad();
            let cp = self.carry_pad();
            let ld = sp + cp + t;
            let scale = 1.0 / (hd as f32).sqrt();
            let group = qh / kvh;
            if sk > 0 {
                cmd.copy(q, 0, q_raw, 0, m * qd);
                let n = sk * hd;
                for g in 0..kvh {
                    cmd.copy(&self.p, sink_k + g * n, sink_k_pad, g * SINK_PAD * hd, n);
                    cmd.copy(&self.p, sink_v + g * n, sink_v_pad, g * SINK_PAD * hd, n);
                }
            }
            // carried anchor keys: the raw tails of THIS window go to the next
            // window (before the in-place rope); the previous window's raw
            // tail is rotated at positions 0..cp for the carried score block
            let tails = self.carry.as_ref().and_then(|cs| cs.tails[layer].as_ref());
            if let Some((kc, vc, kn, vn, krot)) = tails {
                if self.carry_active.get() {
                    self.carry_tail_copy(cmd, k, kn, cp, kd);
                    self.carry_tail_copy(cmd, v, vn, cp, kd);
                }
                cmd.copy(kc, 0, krot, 0, b * cp * kd);
                cmd.rope_at(krot, 0, b * cp, cp, kvh, hd, cfg.rope_base, false, 0);
                let _ = vc;
            }
            cmd.rope_at(q, 0, m, t, qh, hd, cfg.rope_base, false, cp);
            cmd.rope_at(k, 0, m, t, kvh, hd, cfg.rope_base, false, cp);
            // batched over z = (b, kv-group g, head-in-group j), head i = g·group + j
            let sq = [t * qd, group * hd, hd]; // q / o column blocks
            let skv = [t * kd, hd, 0]; // k / v (shared across the group)
            let sct = [cp * kd, hd, 0]; // carried tails [B, cp, kd]
            let spb = [qh * t * ld, group * t * ld, t * ld]; // P blocks
            let ssk = [0, SINK_PAD * hd, 0]; // sink tiles (per kv head, shared over b and j)
            let bt = |sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]| GemmBatch {
                nb: b,
                nh: kvh,
                nc: group,
                sa,
                sb,
                sc,
            };
            if sk > 0 {
                // sink block: q̂·sink_kᵀ·scale into columns 0..SINK_PAD
                cmd.gemm_ex(
                    Op::N,
                    Op::T,
                    t,
                    SINK_PAD,
                    hd,
                    scale,
                    q_raw,
                    0,
                    qd,
                    sink_k_pad,
                    0,
                    hd,
                    0.0,
                    p,
                    0,
                    ld,
                    &bt(sq, ssk, spb),
                    false,
                );
            }
            if let Some((_, _, _, _, krot)) = tails {
                // carried block: Q_rot·K_tail_rotᵀ·scale at column offset sp
                cmd.gemm_ex(
                    Op::N, Op::T, t, cp, hd, scale, q, 0, qd, krot, 0, kd, 0.0, p, sp, ld,
                    &bt(sq, sct, spb), false,
                );
            }
            // S = Q_i·K_gᵀ·scale (all heads, all sequences) at column offset sp + cp
            cmd.gemm_ex(
                Op::N,
                Op::T,
                t,
                t,
                hd,
                scale,
                q,
                0,
                qd,
                k,
                0,
                kd,
                0.0,
                p,
                sp + cp,
                ld,
                &bt(sq, skv, spb),
                false,
            );
            // diagnostics: CMF_ANCHOR_STAGE=1|2|3 stops the forward after the
            // scores / the softmax / P·V (anchor_core_probe reads the buffers)
            let stage: usize = std::env::var("CMF_ANCHOR_STAGE")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
            if stage == 1 {
                return;
            }
            // outside a carried pass (evaluation windows) the carried block is
            // masked for every sequence
            cmd.banded_softmax_carry(
                p, 0, t, ld, sk, sp, window, b * qh, cp, qh,
                self.carry.as_ref().map(|cs| if self.carry_active.get() { &cs.mask } else { &cs.mask_zero }),
            );
            if stage == 2 {
                return;
            }
            // O_i = P·V_g (+ P_sink·sink_v) (+ P_carried·V_tail)
            cmd.gemm_ex(
                Op::N,
                Op::N,
                t,
                hd,
                t,
                1.0,
                p,
                sp + cp,
                ld,
                v,
                0,
                kd,
                0.0,
                o,
                0,
                qd,
                &bt(spb, skv, sq),
                false,
            );
            if stage == 3 {
                return;
            }
            if let Some((_, vc, _, _, _)) = tails {
                cmd.gemm_ex(
                    Op::N, Op::N, t, hd, cp, 1.0, p, sp, ld, vc, 0, kd, 1.0, o, 0, qd,
                    &bt(spb, sct, sq), false,
                );
            }
            if sk > 0 {
                cmd.gemm_ex(
                    Op::N,
                    Op::N,
                    t,
                    hd,
                    SINK_PAD,
                    1.0,
                    p,
                    0,
                    ld,
                    sink_v_pad,
                    0,
                    hd,
                    1.0,
                    o,
                    0,
                    qd,
                    &bt(spb, ssk, sq),
                    false,
                );
            }
        }

        /// Backward of [`EmbryoGpu::anchor_attn_fwd`]: `scratch.dbig` holds
        /// dO on entry; on exit `scratch.dq` = dq̂ (raw space: window term
        /// through the inverse rope, sink term added after it), `scratch.dk`
        /// = dk̂, `scratch.dv` = dv, and `g[sink_k]`/`g[sink_v]` gained
        /// dS_sinkᵀ·q̂·scale / P_sinkᵀ·dO. One batched dispatch per GEMM
        /// over z = (b, g, j) (dP holds every sequence's blocks); per-head
        /// dK/dV partials land head-major in dkh/dvh and are group-summed,
        /// per-(b, head) sink partial tiles in dsk_part/dsv_part are folded
        /// over (b, j) into the arena grad by `sink_grad_accum`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn anchor_attn_bwd(
            &self,
            cmd: &Cmd,
            layer: usize,
            q: &GBuf,
            k: &GBuf,
            v: &GBuf,
            p: &GBuf,
            q_raw: &GBuf,
            sink_k_pad: &GBuf,
            sink_v_pad: &GBuf,
            sink_k: usize,
            sink_v: usize,
            _window: usize,
        ) {
            let cfg = &self.cfg;
            let s = &self.scratch;
            let (b, t) = (self.b, self.t);
            let m = b * t;
            let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
            let (qd, kd) = (qh * hd, kvh * hd);
            let sk = if cfg.anchor_bounded() { cfg.anchor_sink } else { 0 };
            let sp = cfg.sink_pad();
            let cp = self.carry_pad();
            let ld = sp + cp + t;
            // carried keys/values of the previous window (detached: only dQ
            // receives their term); `krot` was rotated by the forward
            let tails = self.carry.as_ref().and_then(|cs| cs.tails[layer].as_ref());
            let sct = [cp * kd, hd, 0];
            let dq = &s.dq;
            assert!(dq.len >= m * qd && s.dk.len >= m * kd && s.dv.len >= m * kd);
            let scale = 1.0 / (hd as f32).sqrt();
            let group = qh / kvh;
            // Batched over z = (b, kv-group g, head-in-group j) within one
            // range per sequence (the default), or one range over all the
            // sequences (`CMF_ANCHOR_BWD_BATCH=1` / `ANCHOR_BWD_BATCH`, kept
            // for A/B timing; measured slower). Per-head dK/dV partials
            // land head-major in dkh/dvh and are group-summed; per-(b, head)
            // sink partial tiles are written at tile block 0 of the range and
            // folded by `sink_grad_accum` (which adds into the arena grad, so
            // per-range folds accumulate exactly like one fold over all b).
            let ranges: Vec<(usize, usize)> = if anchor_bwd_seq() {
                (0..b).map(|bi| (bi, 1)).collect()
            } else {
                vec![(0, b)]
            };
            let sq = [t * qd, group * hd, hd];
            let skv = [t * kd, hd, 0];
            let spb = [qh * t * ld, group * t * ld, t * ld];
            let sh = [qh * t * hd, group * t * hd, t * hd]; // head-major [B][qh][T][hd]
            let ssk = [0, SINK_PAD * hd, 0];
            let sps = [qh * SINK_PAD * hd, group * SINK_PAD * hd, SINK_PAD * hd];
            for &(b0, nb) in &ranges {
                let bt = |sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]| GemmBatch {
                    nb,
                    nh: kvh,
                    nc: group,
                    sa,
                    sb,
                    sc,
                };
                let q_off = b0 * t * qd;
                let kv_off = b0 * t * kd;
                let p_off = b0 * qh * t * ld;
                let h_off = b0 * qh * t * hd;
                let tail_off = b0 * cp * kd;
                // dP_win = dO_i·V_gᵀ → dp[:, sp+cp..]
                cmd.gemm_ex(
                    Op::N,
                    Op::T,
                    t,
                    t,
                    hd,
                    1.0,
                    &s.dbig,
                    q_off,
                    qd,
                    v,
                    kv_off,
                    kd,
                    0.0,
                    &s.dp,
                    p_off + sp + cp,
                    ld,
                    &bt(sq, skv, spb),
                    false,
                );
                if let Some((_, vc, _, _, _)) = tails {
                    // dP_carried = dO_i·V_tailᵀ → dp[:, sp..sp+cp]
                    cmd.gemm_ex(
                        Op::N, Op::T, t, cp, hd, 1.0, &s.dbig, q_off, qd, vc, tail_off, kd, 0.0,
                        &s.dp, p_off + sp, ld, &bt(sq, sct, spb), false,
                    );
                }
                if sk > 0 {
                    // dP_sink = dO·sink_v_padᵀ → dp[:, 0..SINK_PAD]
                    cmd.gemm_ex(
                        Op::N,
                        Op::T,
                        t,
                        SINK_PAD,
                        hd,
                        1.0,
                        &s.dbig,
                        q_off,
                        qd,
                        sink_v_pad,
                        0,
                        hd,
                        0.0,
                        &s.dp,
                        p_off,
                        ld,
                        &bt(sq, ssk, spb),
                        false,
                    );
                    // dsink_v(partial, per (b, head)) = P_sinkᵀ·dO
                    cmd.gemm_ex(
                        Op::T,
                        Op::N,
                        SINK_PAD,
                        hd,
                        t,
                        1.0,
                        p,
                        p_off,
                        ld,
                        &s.dbig,
                        q_off,
                        qd,
                        0.0,
                        &s.dsv_part,
                        0,
                        hd,
                        &bt(spb, sq, sps),
                        false,
                    );
                    cmd.sink_grad_accum(&s.dsv_part, &self.g, sink_v, nb, kvh, group, sk, hd, 1.0);
                }
                // dV_i(partial) = P_winᵀ·dO_i
                cmd.gemm_ex(
                    Op::T,
                    Op::N,
                    t,
                    hd,
                    t,
                    1.0,
                    p,
                    p_off + sp + cp,
                    ld,
                    &s.dbig,
                    q_off,
                    qd,
                    0.0,
                    &s.dvh,
                    h_off,
                    hd,
                    &bt(spb, sq, sh),
                    false,
                );
                // dS = P⊙(dP − rowsum) over the whole [T, LD] row (P = 0 off band)
                cmd.softmax_bwd_blocks_ld(p, p_off, &s.dp, p_off, t, ld, nb * qh);
                // dQ_i = dS_win·K_g·scale (+ dS_carried·K_tail_rot·scale)
                cmd.gemm_ex(
                    Op::N,
                    Op::N,
                    t,
                    hd,
                    t,
                    scale,
                    &s.dp,
                    p_off + sp + cp,
                    ld,
                    k,
                    kv_off,
                    kd,
                    0.0,
                    dq,
                    q_off,
                    qd,
                    &bt(spb, skv, sq),
                    false,
                );
                if let Some((_, _, _, _, krot)) = tails {
                    cmd.gemm_ex(
                        Op::N, Op::N, t, hd, cp, scale, &s.dp, p_off + sp, ld, krot, tail_off, kd,
                        1.0, dq, q_off, qd, &bt(spb, sct, sq), false,
                    );
                }
                // dK_i(partial) = dS_winᵀ·Q_i·scale
                cmd.gemm_ex(
                    Op::T,
                    Op::N,
                    t,
                    hd,
                    t,
                    scale,
                    &s.dp,
                    p_off + sp + cp,
                    ld,
                    q,
                    q_off,
                    qd,
                    0.0,
                    &s.dkh,
                    h_off,
                    hd,
                    &bt(spb, sq, sh),
                    false,
                );
            }
            cmd.group_sum_heads(&s.dkh, &s.dk, b, t, qh, kvh, hd);
            cmd.group_sum_heads(&s.dvh, &s.dv, b, t, qh, kvh, hd);
            cmd.rope_at(dq, 0, m, t, qh, hd, cfg.rope_base, true, cp);
            cmd.rope_at(&s.dk, 0, m, t, kvh, hd, cfg.rope_base, true, cp);
            if sk > 0 {
                for &(b0, nb) in &ranges {
                    let bt = |sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]| GemmBatch {
                        nb,
                        nh: kvh,
                        nc: group,
                        sa,
                        sb,
                        sc,
                    };
                    let q_off = b0 * t * qd;
                    let p_off = b0 * qh * t * ld;
                    // the NoPE sink term dS_sink·sink_k·scale on top of the raw-space dq
                    cmd.gemm_ex(
                        Op::N,
                        Op::N,
                        t,
                        hd,
                        SINK_PAD,
                        scale,
                        &s.dp,
                        p_off,
                        ld,
                        sink_k_pad,
                        0,
                        hd,
                        1.0,
                        dq,
                        q_off,
                        qd,
                        &bt(spb, ssk, sq),
                        false,
                    );
                    // dsink_k(partial, per (b, head)) = dS_sinkᵀ·q̂·scale
                    cmd.gemm_ex(
                        Op::T,
                        Op::N,
                        SINK_PAD,
                        hd,
                        t,
                        scale,
                        &s.dp,
                        p_off,
                        ld,
                        q_raw,
                        q_off,
                        qd,
                        0.0,
                        &s.dsk_part,
                        0,
                        hd,
                        &bt(spb, sq, sps),
                        false,
                    );
                    cmd.sink_grad_accum(&s.dsk_part, &self.g, sink_k, nb, kvh, group, sk, hd, 1.0);
                }
            }
        }

        /// Oracle entry for tests: the attention core + output projection of
        /// anchor layer `l` on host inputs — raw projections `q_raw` [M,qd],
        /// `k_raw` [M,kd], `v` [M,kd] and the upstream gradient `dy` [M,H] of
        /// `y = o·Woᵀ` — with the given window. Returns y and every gradient
        /// the core produces (dq̂, dk̂, dv, dsink_k, dsink_v, dWo). The
        /// activation buffers of layer `l` are clobbered.
        pub fn anchor_core_probe(
            &self,
            l: usize,
            q_raw_in: &[f32],
            k_raw_in: &[f32],
            v_in: &[f32],
            dy: &[f32],
            window: usize,
        ) -> AnchorProbe {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
            let (qd, kd) = (qh * hd, kvh * hd);
            assert_eq!(q_raw_in.len(), m * qd);
            assert_eq!(k_raw_in.len(), m * kd);
            assert_eq!(v_in.len(), m * kd);
            assert_eq!(dy.len(), m * h);
            let (LayerOffs::Anchor { wo, sink_k, sink_v, .. }, LayerActs::Anchor {
                q,
                k,
                v,
                p,
                o,
                q_raw,
                sink_k_pad,
                sink_v_pad,
                ..
            }) = (&self.lay.layers[l], &self.acts[l])
            else {
                panic!("layer {l} is not an anchor");
            };
            let s = &self.scratch;
            q.write_from(q_raw_in);
            k.write_from(k_raw_in);
            v.write_from(v_in);
            s.dx.write_from(dy);
            let cmd = Cmd::new(self.ctx());
            cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total);
            self.anchor_attn_fwd(
                &cmd, l, q, k, v, p, o, q_raw, sink_k_pad, sink_v_pad, *sink_k, *sink_v, window,
            );
            // y = o·Woᵀ → s.dx2 (a free [M,H] scratch)
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                h,
                qd,
                1.0,
                o,
                0,
                qd,
                &self.p,
                *wo,
                qd,
                0.0,
                &s.dx2,
                0,
                h,
            );
            // dO = dy·Wo ; dWo += dyᵀ·o
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                qd,
                h,
                1.0,
                &s.dx,
                0,
                h,
                &self.p,
                *wo,
                qd,
                0.0,
                &s.dbig,
                0,
                qd,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                h,
                qd,
                m,
                1.0,
                &s.dx,
                0,
                h,
                o,
                0,
                qd,
                1.0,
                &self.g,
                *wo,
                qd,
            );
            self.anchor_attn_bwd(
                &cmd, l, q, k, v, p, q_raw, sink_k_pad, sink_v_pad, *sink_k, *sink_v, window,
            );
            cmd.commit();
            let g = self.g.to_vec();
            let ns = kvh * cfg.anchor_sink * hd;
            let ld = SINK_PAD + self.carry_pad() + self.t;
            AnchorProbe {
                p0: p.to_vec()[..self.t * ld].to_vec(),
                o0: o.to_vec()[..self.t * qd].to_vec(),
                y: s.dx2.to_vec()[..m * h].to_vec(),
                dq: s.dq.to_vec()[..m * qd].to_vec(),
                dk: s.dk.to_vec()[..m * kd].to_vec(),
                dv: s.dv.to_vec()[..m * kd].to_vec(),
                dsink_k: if ns > 0 { g[*sink_k..*sink_k + ns].to_vec() } else { Vec::new() },
                dsink_v: if ns > 0 { g[*sink_v..*sink_v + ns].to_vec() } else { Vec::new() },
                dwo: g[*wo..*wo + h * qd].to_vec(),
            }
        }

        /// Forward through layer `l` from acts[l].x_in into `x_out`.
        pub(crate) fn layer_fwd(&self, cmd: &Cmd, l: usize, x_out: &GBuf, train: bool) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            match (&self.lay.layers[l], &self.acts[l]) {
                (
                    LayerOffs::Mixer {
                        ln1,
                        wq,
                        wk,
                        wv,
                        wkap,
                        wo,
                        alog: _,
                        conv,
                        ln2,
                        ffn,
                    },
                    LayerActs::Mixer {
                        x_in,
                        x1,
                        x1c,
                        inv1,
                        thq,
                        thk,
                        v,
                        kpre,
                        kappa,
                        phq,
                        phk,
                        kv,
                        states,
                        o,
                        gdn_thq: _,
                        gdn_thk: _,
                        gdn_qfull,
                        gdn_kfull,
                        gdn_v,
                        gdn_ab,
                        gdn_kappa,
                        gdn_phq,
                        gdn_phk,
                        gdn_kv,
                        gdn_states,
                        gdn_raw_o,
                        gdn_inv,
                        gdn_o,
                        gdn_z,
                        gdn_gated,
                        gdn_dz: _,
                        gdn_do: _,
                        gdn_dq: _,
                        gdn_dk: _,
                        gdn_dv: _,
                        gdn_dab: _,
                        gdn_d,
                        gdn_grad: _,
                        gqa_q,
                        gqa_k,
                        gqa_v,
                        gqa_p,
                        gqa_o,
                        x_mid,
                        x2,
                        inv2,
                        gte,
                        up,
                        hh,
                    },
                ) => {
                    let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                    cmd.rmsnorm_fwd_at(x_in, &self.p, *ln1, x1, inv1, m, h, cfg.norm_eps);
                    let active = self.carry_active.get();
                    let chist = self.carry.as_ref().and_then(|cs| cs.conv[l].as_ref());
                    let xs = if *conv != usize::MAX {
                        cmd.conv1d_fwd_hist(
                            x1, &self.p, *conv, x1c,
                            chist.filter(|_| active).map(|(cur, _, _, _)| cur),
                            b, t, h, cfg.conv_k,
                        );
                        if let (true, Some((_, next, k, w))) = (active, chist) {
                            self.carry_tail_copy(cmd, x1, next, k - 1, *w);
                        }
                        x1c
                    } else {
                        x1
                    };
                    cmd.set_hk_carry(active);
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        nh * nph,
                        h,
                        1.0,
                        xs,
                        0,
                        h,
                        &self.p,
                        *wq,
                        h,
                        0.0,
                        thq,
                        0,
                        nh * nph,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        nh * nph,
                        h,
                        1.0,
                        xs,
                        0,
                        h,
                        &self.p,
                        *wk,
                        h,
                        0.0,
                        thk,
                        0,
                        nh * nph,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        nh * dv,
                        h,
                        1.0,
                        xs,
                        0,
                        h,
                        &self.p,
                        *wv,
                        h,
                        0.0,
                        v,
                        0,
                        nh * dv,
                    );
                    let kld = cfg.kappa_ld();
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        kld,
                        h,
                        1.0,
                        xs,
                        0,
                        h,
                        &self.p,
                        *wkap,
                        h,
                        0.0,
                        kpre,
                        0,
                        kld,
                    );
                    cmd.kappa_fwd(kpre, kappa, m, nh, kld, cfg.kappa_bias);
                    let d = HkDims { b, t, nh, nph, dv };
                    let use_phase_delta = cfg.phase_delta_for_layer(l);
                    let w = HkWork {
                        thq,
                        thk,
                        v,
                        kappa,
                        pow: &self.pow,
                        pow_off: self.pow_off(l),
                        phq,
                        phk,
                        kv,
                        states,
                        out: o,
                        phase_chunk: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                            || use_phase_delta
                        {
                            Some(&self.scratch.phase_chunk)
                        } else {
                            None
                        },
                        phase_partial: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                            || use_phase_delta
                        {
                            Some(&self.scratch.phase_partial)
                        } else {
                            None
                        },
                    };
                    if use_phase_delta {
                        if active {
                            // continuation from the carried S_0 in slot 0
                            cmd.phase_delta_forward(&d, &w);
                        } else {
                            cmd.phase_delta_forward_reset(&d, &w);
                        }
                    } else if hk_simt() {
                        cmd.hk_forward(&d, &w)
                    } else {
                        cmd.hk_forward_gemm(&d, &w, &self.hk_scratch())
                    }
                    cmd.set_hk_carry(false);
                    // x_mid = x_in + o·Woᵀ   ([M, nh·dv]·[H, nh·dv]ᵀ)
                    cmd.copy(x_in, 0, x_mid, 0, m * h);
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        h,
                        nh * dv,
                        1.0,
                        o,
                        0,
                        nh * dv,
                        &self.p,
                        *wo,
                        nh * dv,
                        1.0,
                        x_mid,
                        0,
                        h,
                    );
                    self.gdn_lane_fwd(
                        cmd, l, xs, x_mid, gdn_qfull, gdn_kfull, gdn_v, gdn_ab, gdn_kappa, gdn_phq,
                        gdn_phk, gdn_kv, gdn_states, gdn_raw_o, gdn_inv, gdn_o, gdn_z, gdn_gated,
                        gdn_d,
                    );
                    self.gqa_lane_fwd(cmd, l, xs, x_mid, gqa_q, gqa_k, gqa_v, gqa_p, gqa_o);
                    cmd.rmsnorm_fwd_at(x_mid, &self.p, *ln2, x2, inv2, m, h, cfg.norm_eps);
                    self.ffn_fwd(cmd, l, m, ffn, x2, gte, up, hh, x_mid, x_out, train);
                }
                (
                    LayerOffs::Anchor {
                        ln1,
                        wq,
                        wk,
                        wv,
                        wo,
                        sink_k,
                        sink_v,
                        ln2,
                        ffn,
                    },
                    LayerActs::Anchor {
                        x_in,
                        x1,
                        inv1,
                        q,
                        k,
                        v,
                        p,
                        o,
                        q_raw,
                        sink_k_pad,
                        sink_v_pad,
                        x_mid,
                        x2,
                        inv2,
                        gte,
                        up,
                        hh,
                    },
                ) => {
                    let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                    let (qd, kd) = (qh * hd, kvh * hd);
                    cmd.rmsnorm_fwd_at(x_in, &self.p, *ln1, x1, inv1, m, h, cfg.norm_eps);
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        qd,
                        h,
                        1.0,
                        x1,
                        0,
                        h,
                        &self.p,
                        *wq,
                        h,
                        0.0,
                        q,
                        0,
                        qd,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        kd,
                        h,
                        1.0,
                        x1,
                        0,
                        h,
                        &self.p,
                        *wk,
                        h,
                        0.0,
                        k,
                        0,
                        kd,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        kd,
                        h,
                        1.0,
                        x1,
                        0,
                        h,
                        &self.p,
                        *wv,
                        h,
                        0.0,
                        v,
                        0,
                        kd,
                    );
                    let window = self.anchor_window_for(train);
                    self.anchor_attn_fwd(
                        cmd, l, q, k, v, p, o, q_raw, sink_k_pad, sink_v_pad, *sink_k, *sink_v,
                        window,
                    );
                    cmd.copy(x_in, 0, x_mid, 0, m * h);
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        m,
                        h,
                        qd,
                        1.0,
                        o,
                        0,
                        qd,
                        &self.p,
                        *wo,
                        qd,
                        1.0,
                        x_mid,
                        0,
                        h,
                    );
                    cmd.rmsnorm_fwd_at(x_mid, &self.p, *ln2, x2, inv2, m, h, cfg.norm_eps);
                    self.ffn_fwd(cmd, l, m, ffn, x2, gte, up, hh, x_mid, x_out, train);
                }
                (LayerOffs::Gdn { .. }, LayerActs::Gdn { .. }) => {
                    self.gdn_layer_fwd(cmd, l, x_out, train);
                }
                _ => unreachable!("layout/acts kind mismatch"),
            }
        }

        /// Forward of a GDN mixer layer: the runtime's `gated_delta_net`
        /// (`linear_core::gdn_step`) operator on the trainer's kernels —
        ///   qkv = x̂·W_qkvᵀ, z = x̂·W_zᵀ, a = x̂·W_aᵀ, b = x̂·W_bᵀ
        ///   qkv_cv = SiLU(conv1d_4(qkv))            (conv1d_fwd_at + silu)
        ///   raw_o  = token scan (L2 q/k, α, β, S)   (gdn_scan_fwd)
        ///   o_norm = RMSNorm_dv(raw_o)·w            (rmsnorm over M·nv rows)
        ///   gated  = SiLU(z)·o_norm                 (swiglu_fwd)
        ///   x_mid  = x_in + gated·W_oᵀ
        /// then the usual post-norm + FFN.
        fn gdn_layer_fwd(&self, cmd: &Cmd, l: usize, x_out: &GBuf, train: bool) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            let s = &self.scratch;
            let (
                LayerOffs::Gdn {
                    ln1,
                    in_qkv,
                    in_z,
                    in_a,
                    in_b,
                    wo,
                    conv,
                    alog,
                    dt_bias,
                    norm,
                    ln2,
                    ffn,
                },
                LayerActs::Gdn {
                    x_in,
                    x1,
                    inv1,
                    qkv,
                    qkv_cv,
                    z,
                    a_pre,
                    b_pre,
                    states,
                    raw_o,
                    inv_o,
                    o_norm,
                    gated,
                    x_mid,
                    x2,
                    inv2,
                    gte,
                    up,
                    hh,
                },
            ) = (&self.lay.layers[l], &self.acts[l])
            else {
                unreachable!("gdn_layer_fwd on a non-GDN layer");
            };
            let (nv, dk, dv) = (cfg.gdn_heads, cfg.gdn_dk, cfg.gdn_dv);
            let c_dim = cfg.gdn_c_dim();
            let vd = nv * dv;
            cmd.rmsnorm_fwd_at(x_in, &self.p, *ln1, x1, inv1, m, h, cfg.norm_eps);
            let proj = |w: usize, n: usize, out: &GBuf| {
                cmd.gemm(
                    Op::N, Op::T, m, n, h, 1.0, x1, 0, h, &self.p, w, h, 0.0, out, 0, n,
                );
            };
            proj(*in_qkv, c_dim, qkv);
            proj(*in_z, vd, z);
            proj(*in_a, GDN_AB_PAD, a_pre);
            proj(*in_b, GDN_AB_PAD, b_pre);
            // causal depthwise conv (tap 3 = current token) + SiLU on every channel;
            // with carry the previous window's last 3 raw rows replace the zero pad
            let active = self.carry_active.get();
            let chist = self.carry.as_ref().and_then(|cs| cs.conv[l].as_ref());
            cmd.conv1d_fwd_hist(
                qkv, &self.p, *conv, &s.gdn_pre,
                chist.filter(|_| active).map(|(cur, _, _, _)| cur),
                b, t, c_dim, GDN_CONV_K,
            );
            if let (true, Some((_, next, k, w))) = (active, chist) {
                self.carry_tail_copy(cmd, qkv, next, k - 1, *w);
            }
            cmd.silu_fwd(&s.gdn_pre, qkv_cv, m * c_dim);
            let d = GdnScanDims {
                b,
                t,
                nv,
                dk,
                dv,
                c_dim,
                ab_ld: GDN_AB_PAD,
            };
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            let wy = self.gdn_wy.as_ref().and_then(|(_, keep)| keep[l].as_ref());
            #[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
            let wy: Option<&()> = None;
            match wy {
                #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
                Some(w) => crate::gdn_wy::wy_fwd(
                    cmd,
                    &d,
                    w,
                    qkv_cv,
                    a_pre,
                    b_pre,
                    &self.p,
                    *alog,
                    *dt_bias,
                    raw_o,
                    states,
                    self.gdn_s0_from_ckpt.get() || active,
                    cfg.gdn_beta_one,
                ),
                _ => cmd.gdn_scan_fwd(
                    &d,
                    qkv_cv,
                    a_pre,
                    b_pre,
                    &self.p,
                    *alog,
                    *dt_bias,
                    raw_o,
                    states,
                    &s.gdn_live,
                    self.gdn_s0_from_ckpt.get() || active,
                    cfg.gdn_beta_one,
                ),
            }
            // gated RMSNorm per head (rows of dv, one shared gain) · SiLU(z)
            cmd.rmsnorm_fwd_at(raw_o, &self.p, *norm, o_norm, inv_o, m * nv, dv, cfg.norm_eps);
            cmd.swiglu_fwd(z, o_norm, gated, m * vd);
            cmd.copy(x_in, 0, x_mid, 0, m * h);
            cmd.gemm(
                Op::N, Op::T, m, h, vd, 1.0, gated, 0, vd, &self.p, *wo, vd, 1.0, x_mid, 0, h,
            );
            cmd.rmsnorm_fwd_at(x_mid, &self.p, *ln2, x2, inv2, m, h, cfg.norm_eps);
            self.ffn_fwd(cmd, l, m, ffn, x2, gte, up, hh, x_mid, x_out, train);
        }

        /// Backward of a GDN mixer layer (mirror of `gdn_layer_fwd`): the
        /// scan's reverse pass replays each 64-token chunk from its
        /// checkpoint; dA_log / d dt_bias arrive as per-(b, head) partials and
        /// are folded in a fixed order (deterministic, no float atomics).
        fn gdn_layer_bwd(&self, cmd: &Cmd, l: usize) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            let s = &self.scratch;
            let (
                LayerOffs::Gdn {
                    ln1,
                    in_qkv,
                    in_z,
                    in_a,
                    in_b,
                    wo,
                    conv,
                    alog,
                    dt_bias,
                    norm,
                    ln2,
                    ffn,
                },
                LayerActs::Gdn {
                    x_in,
                    x1,
                    inv1,
                    qkv,
                    qkv_cv,
                    z,
                    a_pre,
                    b_pre,
                    states,
                    raw_o,
                    inv_o,
                    o_norm,
                    gated,
                    x_mid,
                    x2,
                    inv2,
                    gte,
                    up,
                    hh,
                },
            ) = (&self.lay.layers[l], &self.acts[l])
            else {
                unreachable!("gdn_layer_bwd on a non-GDN layer");
            };
            let (nv, dk, dv) = (cfg.gdn_heads, cfg.gdn_dk, cfg.gdn_dv);
            let c_dim = cfg.gdn_c_dim();
            let vd = nv * dv;
            self.ffn_bwd(cmd, l, m, ffn, *ln2, x_mid, x2, inv2, gte, up, hh);
            // dgated = dx_mid·Wo ; dWo += dx_midᵀ·gated
            cmd.gemm(
                Op::N, Op::N, m, vd, h, 1.0, &s.dx, 0, h, &self.p, *wo, vd, 0.0, &s.gdn_dgated, 0,
                vd,
            );
            cmd.gemm(
                Op::T, Op::N, h, vd, m, 1.0, &s.dx, 0, h, gated, 0, vd, 1.0, &self.g, *wo, vd,
            );
            // gated = SiLU(z)·o_norm → dz, d o_norm
            cmd.swiglu_bwd(z, o_norm, &s.gdn_dgated, &s.gdn_dz, &s.gdn_don, m * vd);
            // o_norm = RMSNorm(raw_o)·w → d raw_o, dw
            cmd.rmsnorm_bwd_at(
                raw_o, &self.p, *norm, &s.gdn_don, inv_o, &s.gdn_doo, 0.0, &self.g, *norm, m * nv,
                dv,
            );
            // reverse scan: d qkv_cv, da (col h), db (col h), per-(b,h) dA_log/ddt partials
            cmd.axpby(0.0, &s.gdn_da, 0.0, &s.gdn_da, m * GDN_AB_PAD);
            cmd.axpby(0.0, &s.gdn_db, 0.0, &s.gdn_db, m * GDN_AB_PAD);
            let d = GdnScanDims {
                b,
                t,
                nv,
                dk,
                dv,
                c_dim,
                ab_ld: GDN_AB_PAD,
            };
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            let wy = self
                .gdn_wy
                .as_ref()
                .and_then(|(x, keep)| keep[l].as_ref().map(|w| (w, x)));
            #[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
            let wy: Option<&()> = None;
            match wy {
                #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
                Some((w, x)) => crate::gdn_wy::wy_bwd(
                    cmd,
                    &d,
                    w,
                    x,
                    qkv_cv,
                    a_pre,
                    &self.p,
                    *alog,
                    *dt_bias,
                    states,
                    &s.gdn_doo,
                    &s.gdn_dlive,
                    false,
                    cfg.gdn_beta_one,
                    &s.gdn_dcv,
                    &s.gdn_da,
                    &s.gdn_db,
                    &s.gdn_part,
                ),
                _ => cmd.gdn_scan_bwd(
                    &d,
                    qkv_cv,
                    a_pre,
                    b_pre,
                    &self.p,
                    *alog,
                    *dt_bias,
                    states,
                    &s.gdn_doo,
                    &s.gdn_chunk,
                    &s.gdn_dlive,
                    false,
                    cfg.gdn_beta_one,
                    &s.gdn_dcv,
                    &s.gdn_da,
                    &s.gdn_db,
                    &s.gdn_part,
                ),
            }
            cmd.gdn_scan_fold(&s.gdn_part, b, nv, &self.g, *alog, *dt_bias);
            // SiLU(conv1d(qkv)) backward: recompute the pre-activation (with the
            // carried history when active; the history itself is detached)
            let chist = self
                .carry
                .as_ref()
                .filter(|_| self.carry_active.get())
                .and_then(|cs| cs.conv[l].as_ref())
                .map(|(cur, _, _, _)| cur);
            cmd.conv1d_fwd_hist(qkv, &self.p, *conv, &s.gdn_pre, chist, b, t, c_dim, GDN_CONV_K);
            cmd.silu_bwd(&s.gdn_pre, &s.gdn_dcv, &s.gdn_dpre, m * c_dim);
            cmd.conv1d_bwd_hist(
                qkv, &self.p, *conv, &s.gdn_dpre, &s.gdn_dqkv, &self.g, *conv, chist, b, t, c_dim,
                GDN_CONV_K,
            );
            // weight grads of the four projections (x1 is what they saw)
            let wgrad = |dy: &GBuf, n: usize, w: usize| {
                cmd.gemm(
                    Op::T, Op::N, n, h, m, 1.0, dy, 0, n, x1, 0, h, 1.0, &self.g, w, h,
                );
            };
            wgrad(&s.gdn_dqkv, c_dim, *in_qkv);
            wgrad(&s.gdn_dz, vd, *in_z);
            wgrad(&s.gdn_da, GDN_AB_PAD, *in_a);
            wgrad(&s.gdn_db, GDN_AB_PAD, *in_b);
            // dx1 = dqkv·W_qkv + dz·W_z + da·W_a + db·W_b
            let mut beta = 0.0;
            for (dy, n, w) in [
                (&s.gdn_dqkv, c_dim, *in_qkv),
                (&s.gdn_dz, vd, *in_z),
                (&s.gdn_da, GDN_AB_PAD, *in_a),
                (&s.gdn_db, GDN_AB_PAD, *in_b),
            ] {
                cmd.gemm(
                    Op::N, Op::N, m, h, n, 1.0, dy, 0, n, &self.p, w, h, beta, &s.dx1, 0, h,
                );
                beta = 1.0;
            }
            // dx_in = dx_mid + rmsnorm_bwd(x_in; dx1)
            cmd.rmsnorm_bwd_at(
                x_in, &self.p, *ln1, &s.dx1, inv1, &s.dx, 1.0, &self.g, *ln1, m, h,
            );
        }

        /// Backward of the optional correction lane.  The exact reverse GDN
        /// scan accumulates projection/output gradients into the appended
        /// tail. `s.dx` is the gradient at the mixer residual boundary (after
        /// FFN backward); a copy is kept for the scalar gain dot.
        #[allow(clippy::too_many_arguments)]
        fn gdn_lane_bwd(
            &self,
            cmd: &Cmd,
            l: usize,
            xs: &GBuf,
            _x1: &GBuf,
            gdn_qfull: &GBuf,
            gdn_kfull: &GBuf,
            gdn_v: &GBuf,
            gdn_ab: &GBuf,
            gdn_kappa: &GBuf,
            gdn_phq: &GBuf,
            gdn_phk: &GBuf,
            gdn_kv: &GBuf,
            gdn_states: &GBuf,
            gdn_raw_o: &GBuf,
            gdn_inv: &GBuf,
            gdn_o: &GBuf,
            gdn_z: &GBuf,
            gdn_gated: &GBuf,
            gdn_dz: &GBuf,
            gdn_do: &GBuf,
            gdn_dq: &GBuf,
            gdn_dk: &GBuf,
            gdn_dv: &GBuf,
            gdn_dab: &GBuf,
            gdn_d: &GBuf,
            gdn_grad: &GBuf,
        ) {
            let Some(go) = self.lay.gdn.get(l).and_then(|x| x.as_ref()) else {
                return;
            };
            let (m, h) = (self.b * self.t, self.cfg.hidden);
            let s = &self.scratch;
            let gain = unsafe { host_slice(&self.p) }[go.gain];
            // Preserve the upstream mixer gradient for the gain derivative.
            cmd.copy(&s.dx, 0, gdn_grad, 0, m * h);
            if gain == 0.0 {
                // The lane is an exact identity seam at initialization.  No
                // parameter gradient is needed when its scalar gain is zero,
                // but emit the dot so a finite-difference gain check remains
                // meaningful for manually activated candidates.
                cmd.dot_accum(gdn_grad, gdn_d, &self.g, go.gain, m * h);
                return;
            }
            // dD = gain·dX and dWo += dXᵀ·gated.
            cmd.axpby(gain, &s.dx, 0.0, &s.dx1, m * h);
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                64,
                h,
                1.0,
                &s.dx1,
                0,
                h,
                &self.p,
                go.wo,
                64,
                0.0,
                &s.dbig,
                0,
                64,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                h,
                64,
                m,
                gain,
                &s.dx,
                0,
                h,
                gdn_gated,
                0,
                64,
                1.0,
                &self.g,
                go.wo,
                64,
            );
            // gated = silu(z)·o; its two stream gradients occupy scratch dk/dv.
            cmd.swiglu_bwd(gdn_z, gdn_o, &s.dbig, gdn_dz, gdn_do, m * 64);
            // Exact reverse GDN scan.  It writes projected q/k/v gradients,
            // control gradients, and direct conv/norm/decay parameter grads.
            if std::env::var_os("CMF_GDN_SERIAL").is_some() {
                cmd.gdn_backward(
                    gdn_qfull, gdn_kfull, gdn_v, gdn_phq, gdn_phk, gdn_kv, gdn_ab, gdn_kappa,
                    gdn_raw_o, gdn_inv, &self.p, gdn_states, gdn_do, gdn_dz, gdn_dq, gdn_dk,
                    gdn_dv, gdn_dab, &self.p, go.conv, go.norm, go.alog, go.dt_bias, &self.g,
                    go.conv, go.norm, go.alog, go.dt_bias, self.b, self.t,
                );
            } else {
                cmd.gdn_backward_parallel(
                    gdn_qfull, gdn_kfull, gdn_v, gdn_phq, gdn_phk, gdn_kv, gdn_ab, gdn_kappa,
                    gdn_raw_o, gdn_inv, &self.p, gdn_states, gdn_do, gdn_dz, gdn_dq, gdn_dk,
                    gdn_dv, gdn_dab, &self.p, go.conv, go.norm, go.alog, go.dt_bias, &self.g,
                    go.conv, go.norm, go.alog, go.dt_bias, self.b, self.t,
                );
            }
            cmd.gemm(
                Op::T,
                Op::N,
                64,
                h,
                m,
                1.0,
                gdn_dq,
                0,
                64,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.qkvz,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                64,
                h,
                m,
                1.0,
                gdn_dk,
                0,
                64,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.qkvz + 64 * h,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                64,
                h,
                m,
                1.0,
                gdn_dv,
                0,
                64,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.qkvz + 128 * h,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                64,
                h,
                m,
                1.0,
                gdn_dz,
                0,
                64,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.qkvz + 192 * h,
                h,
            );
            cmd.axpby(0.0, &s.dx2, 0.0, &s.dx2, m * h);
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dq,
                0,
                64,
                &self.p,
                go.qkvz,
                h,
                0.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dk,
                0,
                64,
                &self.p,
                go.qkvz + 64 * h,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dv,
                0,
                64,
                &self.p,
                go.qkvz + 128 * h,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dz,
                0,
                64,
                &self.p,
                go.qkvz + 192 * h,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::T,
                Op::N,
                64,
                h,
                m,
                1.0,
                gdn_dab,
                0,
                64,
                xs,
                0,
                h,
                1.0,
                &self.g,
                go.ab,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dab,
                0,
                64,
                &self.p,
                go.ab,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            cmd.gemm(
                Op::N,
                Op::N,
                m,
                h,
                64,
                1.0,
                gdn_dz,
                0,
                64,
                &self.p,
                go.qkvz + 192 * h,
                h,
                1.0,
                &s.dx2,
                0,
                h,
            );
            // Accumulate lane input gradient into the legacy projection grad
            // and one scalar gain derivative directly in the arena.
            cmd.axpby(1.0, &s.dx2, 1.0, &s.dx1, m * h);
            cmd.dot_accum(gdn_grad, gdn_d, &self.g, go.gain, m * h);
        }

        /// Backward through layer `l`: s.dx holds dL/dx_out on entry and
        /// dL/dx_in on exit.
        pub(crate) fn layer_bwd(&self, cmd: &Cmd, l: usize) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            let s = &self.scratch;
            match (&self.lay.layers[l], &self.acts[l]) {
                (
                    LayerOffs::Mixer {
                        ln1,
                        wq,
                        wk,
                        wv,
                        wkap,
                        wo,
                        alog: _,
                        conv,
                        ln2,
                        ffn,
                    },
                    LayerActs::Mixer {
                        x_in,
                        x1,
                        x1c,
                        inv1,
                        thq,
                        thk,
                        v,
                        kpre: _,
                        kappa,
                        phq,
                        phk,
                        kv,
                        states,
                        o,
                        gdn_thq: _,
                        gdn_thk: _,
                        gdn_qfull,
                        gdn_kfull,
                        gdn_v,
                        gdn_ab,
                        gdn_kappa,
                        gdn_phq,
                        gdn_phk,
                        gdn_kv,
                        gdn_states,
                        gdn_raw_o,
                        gdn_inv,
                        gdn_o,
                        gdn_z,
                        gdn_gated,
                        gdn_dz,
                        gdn_do,
                        gdn_dq,
                        gdn_dk,
                        gdn_dv,
                        gdn_dab,
                        gdn_d,
                        gdn_grad,
                        gqa_q,
                        gqa_k,
                        gqa_v,
                        gqa_p,
                        gqa_o,
                        x_mid,
                        x2,
                        inv2,
                        gte,
                        up,
                        hh,
                    },
                ) => {
                    let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                    self.ffn_bwd(cmd, l, m, ffn, *ln2, x_mid, x2, inv2, gte, up, hh);
                    // do = dx_mid·Wo  ([M,H]·[H, nh·dv]);  dWo += dx_midᵀ·o
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        nh * dv,
                        h,
                        1.0,
                        &s.dx,
                        0,
                        h,
                        &self.p,
                        *wo,
                        nh * dv,
                        0.0,
                        &s.dbig,
                        0,
                        nh * dv,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        h,
                        nh * dv,
                        m,
                        1.0,
                        &s.dx,
                        0,
                        h,
                        o,
                        0,
                        nh * dv,
                        1.0,
                        &self.g,
                        *wo,
                        nh * dv,
                    );
                    let d = HkDims { b, t, nh, nph, dv };
                    let use_phase_delta = cfg.phase_delta_for_layer(l);
                    let w = HkWork {
                        thq,
                        thk,
                        v,
                        kappa,
                        pow: &self.pow,
                        pow_off: self.pow_off(l),
                        phq,
                        phk,
                        kv,
                        states,
                        out: o,
                        phase_chunk: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                            || use_phase_delta
                        {
                            Some(&s.phase_chunk)
                        } else {
                            None
                        },
                        phase_partial: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                            || use_phase_delta
                        {
                            Some(&s.phase_partial)
                        } else {
                            None
                        },
                    };
                    let gr = HkGrads {
                        dout: &s.dbig,
                        dstates: &s.dstates,
                        dkv: &s.dkv,
                        dphq: &s.dphq,
                        dphk: &s.dphk,
                        dthq: &s.dk,
                        dthk: &s.dk2,
                        dv: &s.dv,
                        dkappa: &s.dkap,
                    };
                    if use_phase_delta {
                        cmd.phase_delta_backward(&d, &w, &gr);
                    } else if hk_simt() {
                        cmd.hk_backward(&d, &w, &gr, 0.0)
                    } else {
                        cmd.hk_backward_gemm(&d, &w, &gr, &self.hk_scratch(), 0.0)
                    }
                    let kld = cfg.kappa_ld();
                    cmd.kappa_bwd(kappa, &s.dkap, &s.dkpre, m, nh, kld);
                    // dx1 = dthq·Wq + dthk·Wk + dv·Wv + dkpre·Wκ (wrt the
                    // PROJECTION input — the conv output when conv is on)
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        nh * nph,
                        1.0,
                        &s.dk,
                        0,
                        nh * nph,
                        &self.p,
                        *wq,
                        h,
                        0.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        nh * nph,
                        1.0,
                        &s.dk2,
                        0,
                        nh * nph,
                        &self.p,
                        *wk,
                        h,
                        1.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        nh * dv,
                        1.0,
                        &s.dv,
                        0,
                        nh * dv,
                        &self.p,
                        *wv,
                        h,
                        1.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        kld,
                        1.0,
                        &s.dkpre,
                        0,
                        kld,
                        &self.p,
                        *wkap,
                        h,
                        1.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    // weight grads read what the projections actually saw
                    let xs = if *conv != usize::MAX { x1c } else { x1 };
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        nh * nph,
                        h,
                        m,
                        1.0,
                        &s.dk,
                        0,
                        nh * nph,
                        xs,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wq,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        nh * nph,
                        h,
                        m,
                        1.0,
                        &s.dk2,
                        0,
                        nh * nph,
                        xs,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wk,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        nh * dv,
                        h,
                        m,
                        1.0,
                        &s.dv,
                        0,
                        nh * dv,
                        xs,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wv,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        kld,
                        h,
                        m,
                        1.0,
                        &s.dkpre,
                        0,
                        kld,
                        xs,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wkap,
                        h,
                    );
                    self.gdn_lane_bwd(
                        cmd, l, xs, x1, gdn_qfull, gdn_kfull, gdn_v, gdn_ab, gdn_kappa, gdn_phq,
                        gdn_phk, gdn_kv, gdn_states, gdn_raw_o, gdn_inv, gdn_o, gdn_z, gdn_gated,
                        gdn_dz, gdn_do, gdn_dq, gdn_dk, gdn_dv, gdn_dab, gdn_d, gdn_grad,
                    );
                    self.gqa_lane_bwd(cmd, l, xs, gqa_q, gqa_k, gqa_v, gqa_p, gqa_o);
                    let dnorm = if *conv != usize::MAX {
                        // through the conv: dW += correlate(x1, dx1c); dx1 = w ⋆ dx1c
                        // (the carried history only enters dW; it is detached)
                        let chist = self
                            .carry
                            .as_ref()
                            .filter(|_| self.carry_active.get())
                            .and_then(|cs| cs.conv[l].as_ref())
                            .map(|(cur, _, _, _)| cur);
                        cmd.conv1d_bwd_hist(
                            x1, &self.p, *conv, &s.dx1, &s.dxc, &self.g, *conv, chist, b, t, h,
                            cfg.conv_k,
                        );
                        &s.dxc
                    } else {
                        &s.dx1
                    };
                    // dx_in = dx_mid + rmsnorm_bwd(x_in; dnorm)
                    cmd.rmsnorm_bwd_at(
                        x_in, &self.p, *ln1, dnorm, inv1, &s.dx, 1.0, &self.g, *ln1, m, h,
                    );
                }
                (
                    LayerOffs::Anchor {
                        ln1,
                        wq,
                        wk,
                        wv,
                        wo,
                        sink_k,
                        sink_v,
                        ln2,
                        ffn,
                    },
                    LayerActs::Anchor {
                        x_in,
                        x1,
                        inv1,
                        q,
                        k,
                        v,
                        p,
                        o,
                        q_raw,
                        sink_k_pad,
                        sink_v_pad,
                        x_mid,
                        x2,
                        inv2,
                        gte,
                        up,
                        hh,
                    },
                ) => {
                    let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                    let (qd, kd) = (qh * hd, kvh * hd);
                    self.ffn_bwd(cmd, l, m, ffn, *ln2, x_mid, x2, inv2, gte, up, hh);
                    // do = dx_mid·Wo ; dWo += dx_midᵀ·o
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        qd,
                        h,
                        1.0,
                        &s.dx,
                        0,
                        h,
                        &self.p,
                        *wo,
                        qd,
                        0.0,
                        &s.dbig,
                        0,
                        qd,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        h,
                        qd,
                        m,
                        1.0,
                        &s.dx,
                        0,
                        h,
                        o,
                        0,
                        qd,
                        1.0,
                        &self.g,
                        *wo,
                        qd,
                    );
                    // attention backward (band + sinks): dq̂ in s.dq, dk̂/dv in
                    // s.dk/s.dv, the sink gradients straight into the arena grad
                    let window = self.anchor_window_t.get();
                    self.anchor_attn_bwd(
                        cmd, l, q, k, v, p, q_raw, sink_k_pad, sink_v_pad, *sink_k, *sink_v, window,
                    );
                    let dq = &s.dq;
                    // dx1 = dq·Wq + dk·Wk + dv·Wv
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        qd,
                        1.0,
                        dq,
                        0,
                        qd,
                        &self.p,
                        *wq,
                        h,
                        0.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        kd,
                        1.0,
                        &s.dk,
                        0,
                        kd,
                        &self.p,
                        *wk,
                        h,
                        1.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::N,
                        Op::N,
                        m,
                        h,
                        kd,
                        1.0,
                        &s.dv,
                        0,
                        kd,
                        &self.p,
                        *wv,
                        h,
                        1.0,
                        &s.dx1,
                        0,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        qd,
                        h,
                        m,
                        1.0,
                        dq,
                        0,
                        qd,
                        x1,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wq,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        kd,
                        h,
                        m,
                        1.0,
                        &s.dk,
                        0,
                        kd,
                        x1,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wk,
                        h,
                    );
                    cmd.gemm(
                        Op::T,
                        Op::N,
                        kd,
                        h,
                        m,
                        1.0,
                        &s.dv,
                        0,
                        kd,
                        x1,
                        0,
                        h,
                        1.0,
                        &self.g,
                        *wv,
                        h,
                    );
                    cmd.rmsnorm_bwd_at(
                        x_in, &self.p, *ln1, &s.dx1, inv1, &s.dx, 1.0, &self.g, *ln1, m, h,
                    );
                }
                (LayerOffs::Gdn { .. }, LayerActs::Gdn { .. }) => {
                    self.gdn_layer_bwd(cmd, l);
                }
                _ => unreachable!(),
            }
        }

        pub(crate) fn x_in(&self, l: usize) -> &GBuf {
            match &self.acts[l] {
                LayerActs::Mixer { x_in, .. }
                | LayerActs::Anchor { x_in, .. }
                | LayerActs::Gdn { x_in, .. } => x_in,
            }
        }

        /// Host-side prep of the hierarchical head for the batch already in
        /// `tgt`: target cluster ids and the rows grouped by cluster (each
        /// group padded to a multiple of 64 with −1). No-op for the flat head.
        pub fn prepare_head(&self, targets: &[u32]) {
            let ncl = self.cfg.head_clusters;
            if ncl == 0 {
                // The flat and hierarchical heads share the same masked
                // response-only contract.  Keep the count on the host so the
                // CE scale and loss denominator are valid-answer based.
                self.head_valid
                    .set(targets.iter().filter(|&&t| t != u32::MAX).count());
                return;
            }
            let m = self.b * self.t;
            let cs = self.cfg.vocab / ncl;
            let mut tc = vec![0u32; m];
            let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); ncl];
            for (i, &t) in targets.iter().enumerate() {
                if t == u32::MAX {
                    tc[i] = u32::MAX; // ignored position (no target)
                    continue;
                }
                let c = (t as usize / cs).min(ncl - 1);
                tc[i] = c as u32;
                buckets[c].push(i as i32);
            }
            let mut idx: Vec<i32> = Vec::with_capacity(self.mpad);
            let mut groups = Vec::new();
            for (c, bk) in buckets.iter().enumerate() {
                if bk.is_empty() {
                    continue;
                }
                let off = idx.len();
                idx.extend_from_slice(bk);
                // Metal keeps 64-row bucket tiles for its SIMD GEMM path.
                // The Vulkan scalar path has no tile requirement; retaining
                // those pads multiplies sparse hierarchical-head work by up
                // to 64×, so keep only valid rows there.
                let pad = if cfg!(all(feature = "vulkan", not(target_os = "macos"))) {
                    bk.len()
                } else {
                    bk.len().div_ceil(64) * 64
                };
                idx.resize(off + pad, -1);
                groups.push((c, off, pad));
            }
            assert!(idx.len() <= self.mpad);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tc.as_ptr(),
                    self.tgt_cluster.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    idx.as_ptr(),
                    self.head_idx.buf.contents() as *mut i32,
                    idx.len(),
                );
            }
            *self.head_groups.borrow_mut() = groups;
            self.head_valid
                .set(targets.iter().filter(|&&t| t != u32::MAX).count());
        }

        /// Head on the given hidden/targets (the MTP heads reuse the tied head).
        pub(crate) fn encode_head_on(
            &self,
            cmd: &Cmd,
            train: bool,
            x: &GBuf,
            dx: &GBuf,
            tgt: &GBuf,
        ) {
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, self.b * self.t);
            let s = &self.scratch;
            // Prompt/padding rows use u32::MAX and contribute neither loss
            // nor gradient.  Normalise by the number of valid answer
            // targets, not B*T; zero is a legal all-masked probe and must
            // produce a finite zero objective/update.
            let valid = self.head_valid.get();
            let scale = if valid == 0 { 0.0 } else { 1.0 / valid as f32 };
            let ncl = cfg.head_clusters;
            if ncl == 0 {
                let r = self.head_rows;
                for c0 in (0..m).step_by(r) {
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        r,
                        cfg.vocab,
                        h,
                        1.0,
                        x,
                        c0 * h,
                        h,
                        &self.p,
                        self.lay.embed,
                        h,
                        0.0,
                        &s.logits,
                        0,
                        cfg.vocab,
                    );
                    cmd.softmax_ce_at(&s.logits, 0, tgt, c0, &s.loss, c0, r, cfg.vocab, scale);
                    if train {
                        cmd.gemm(
                            Op::N,
                            Op::N,
                            r,
                            h,
                            cfg.vocab,
                            1.0,
                            &s.logits,
                            0,
                            cfg.vocab,
                            &self.p,
                            self.lay.embed,
                            h,
                            0.0,
                            dx,
                            c0 * h,
                            h,
                        );
                        cmd.gemm(
                            Op::T,
                            Op::N,
                            cfg.vocab,
                            h,
                            r,
                            1.0,
                            &s.logits,
                            0,
                            cfg.vocab,
                            x,
                            c0 * h,
                            h,
                            1.0,
                            &self.g,
                            self.lay.embed,
                            h,
                        );
                    }
                }
                return;
            }
            let cs = cfg.vocab / ncl;
            let hc = self.lay.head_clusters;
            // level 1: clusters
            cmd.gemm(
                Op::N,
                Op::T,
                m,
                ncl,
                h,
                1.0,
                x,
                0,
                h,
                &self.p,
                hc,
                h,
                0.0,
                &self.lc,
                0,
                ncl,
            );
            cmd.softmax_ce_at(&self.lc, 0, &self.tgt_cluster, 0, &s.loss, 0, m, ncl, scale);
            if train {
                cmd.gemm(
                    Op::N,
                    Op::N,
                    m,
                    h,
                    ncl,
                    1.0,
                    &self.lc,
                    0,
                    ncl,
                    &self.p,
                    hc,
                    h,
                    0.0,
                    dx,
                    0,
                    h,
                );
                cmd.gemm(
                    Op::T,
                    Op::N,
                    ncl,
                    h,
                    m,
                    1.0,
                    &self.lc,
                    0,
                    ncl,
                    x,
                    0,
                    h,
                    1.0,
                    &self.g,
                    hc,
                    h,
                );
            }
            // level 2: within the target cluster, rows grouped by cluster
            let groups = self.head_groups.borrow();
            let rows_total = groups.last().map(|(_, off, pad)| off + pad).unwrap_or(0);
            let n_valid = self.head_valid.get();
            assert!(
                rows_total >= n_valid,
                "hierarchical head: prepare_head(targets) must precede the encode (grouped {rows_total} rows for {m} tokens)"
            );
            // Vulkan training head: the per-cluster GEMM loops (up to
            // `head_clusters` dispatches each) become one table-batched
            // dispatch per GEMM (`Cmd::gemm_table`): batch z = cluster
            // group z with its own (row offset, cluster offset, rows).
            // `CMF_VULKAN_HEAD_TABLE=0` keeps the loops for A/B.
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            let head_table = if train && head_table_enabled() && !groups.is_empty() {
                let pad_max = groups.iter().map(|g| g.2).max().unwrap_or(0);
                let mut tab: Vec<u32> = Vec::with_capacity(12 * groups.len());
                // section 0: forward  lw[off..] = hg[off..]·E[c]ᵀ
                for &(c, off, pad) in groups.iter() {
                    tab.extend_from_slice(&[
                        (off * h) as u32,
                        (c * cs * h) as u32,
                        (off * cs) as u32,
                        pad as u32,
                    ]);
                }
                // section 1: dhg[off..] = lw[off..]·E[c]
                for &(c, off, pad) in groups.iter() {
                    tab.extend_from_slice(&[
                        (off * cs) as u32,
                        (c * cs * h) as u32,
                        (off * h) as u32,
                        pad as u32,
                    ]);
                }
                // section 2: dE[c] += lw[off..]ᵀ·hg[off..]  (rows bound K)
                for &(c, off, pad) in groups.iter() {
                    tab.extend_from_slice(&[
                        (off * cs) as u32,
                        (off * h) as u32,
                        (c * cs * h) as u32,
                        pad as u32,
                    ]);
                }
                Some((GBuf::from_u32(self.ctx(), &tab), pad_max, groups.len()))
            } else {
                None
            };
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            if !train {
                // Vulkan's scalar WGSL GEMM must not multiply each sparse
                // target bucket by a 64-row padding factor.  The resident
                // grouped-head kernel reads the source token and cluster
                // directly and computes only `rows_total × cs` outputs.
                cmd.head_group_fwd(
                    x,
                    &self.p,
                    self.lay.embed,
                    &self.head_idx,
                    &self.tgt_cluster,
                    &self.lw,
                    rows_total,
                    cs,
                    h,
                );
            } else {
                cmd.gather_rows(x, &self.head_idx, &self.hg, rows_total, h);
                if let Some((tab, pad_max, ng)) = head_table.as_ref() {
                    cmd.gemm_table(
                        Op::N,
                        Op::T,
                        *pad_max,
                        cs,
                        h,
                        1.0,
                        &self.hg,
                        0,
                        h,
                        &self.p,
                        self.lay.embed,
                        h,
                        0.0,
                        &self.lw,
                        0,
                        cs,
                        tab,
                        0,
                        *ng,
                        false,
                    );
                } else {
                    for &(c, off, pad) in groups.iter() {
                        cmd.gemm(
                            Op::N,
                            Op::T,
                            pad,
                            cs,
                            h,
                            1.0,
                            &self.hg,
                            off * h,
                            h,
                            &self.p,
                            self.lay.embed + c * cs * h,
                            h,
                            0.0,
                            &self.lw,
                            off * cs,
                            cs,
                        );
                    }
                }
            }
            #[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
            {
                cmd.gather_rows(x, &self.head_idx, &self.hg, rows_total, h);
                for &(c, off, pad) in groups.iter() {
                    cmd.gemm(
                        Op::N,
                        Op::T,
                        pad,
                        cs,
                        h,
                        1.0,
                        &self.hg,
                        off * h,
                        h,
                        &self.p,
                        self.lay.embed + c * cs * h,
                        h,
                        0.0,
                        &self.lw,
                        off * cs,
                        cs,
                    );
                }
            }
            cmd.softmax_ce_idx(
                &self.lw,
                &self.head_idx,
                tgt,
                &self.loss2,
                rows_total,
                cs,
                scale,
            );
            if train {
                #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
                let grouped = if let Some((tab, pad_max, ng)) = head_table.as_ref() {
                    cmd.gemm_table(
                        Op::N,
                        Op::N,
                        *pad_max,
                        h,
                        cs,
                        1.0,
                        &self.lw,
                        0,
                        cs,
                        &self.p,
                        self.lay.embed,
                        h,
                        0.0,
                        &self.dhg,
                        0,
                        h,
                        tab,
                        4 * ng,
                        *ng,
                        false,
                    );
                    cmd.gemm_table(
                        Op::T,
                        Op::N,
                        cs,
                        h,
                        *pad_max,
                        1.0,
                        &self.lw,
                        0,
                        cs,
                        &self.hg,
                        0,
                        h,
                        1.0,
                        &self.g,
                        self.lay.embed,
                        h,
                        tab,
                        8 * ng,
                        *ng,
                        true,
                    );
                    true
                } else {
                    false
                };
                #[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
                let grouped = false;
                if !grouped {
                    for &(c, off, pad) in groups.iter() {
                        cmd.gemm(
                            Op::N,
                            Op::N,
                            pad,
                            h,
                            cs,
                            1.0,
                            &self.lw,
                            off * cs,
                            cs,
                            &self.p,
                            self.lay.embed + c * cs * h,
                            h,
                            0.0,
                            &self.dhg,
                            off * h,
                            h,
                        );
                        cmd.gemm(
                            Op::T,
                            Op::N,
                            cs,
                            h,
                            pad,
                            1.0,
                            &self.lw,
                            off * cs,
                            cs,
                            &self.hg,
                            off * h,
                            h,
                            1.0,
                            &self.g,
                            self.lay.embed + c * cs * h,
                            h,
                        );
                    }
                }
                cmd.scatter_add_rows(dx, &self.head_idx, &self.dhg, rows_total, h);
            }
        }

        /// Head: loss (+ dxf and dE/dC when `train`) on xf/tgt.
        pub fn encode_head(&self, cmd: &Cmd, train: bool) {
            self.encode_head_on(cmd, train, &self.xf, &self.dxf, &self.tgt);
        }

        /// Mean loss of the batch just run (both levels of the head).
        pub fn read_loss_f64(&self) -> f64 {
            let m = self.b * self.t;
            let l1: f64 = unsafe { host_slice(&self.scratch.loss) }[..m]
                .iter()
                .map(|x| *x as f64)
                .sum();
            let valid = self.head_valid.get();
            if valid == 0 {
                0.0
            } else {
                // Hierarchical-head CE writes loss2 by the original token
                // index, while head_idx is a compact valid-row list (with
                // optional -1 padding on Metal).  Sum only those valid
                // indices; summing the whole [B*T] buffer would retain
                // stale values whenever a masked batch has fewer grouped
                // rows than tokens.
                let l2: f64 = if self.cfg.head_clusters > 0 {
                    let rows_total = self
                        .head_groups
                        .borrow()
                        .last()
                        .map(|(_, off, pad)| off + pad)
                        .unwrap_or(0);
                    let all_idx = unsafe { host_u32_slice(&self.head_idx) };
                    let idx = &all_idx[..rows_total];
                    let losses = unsafe { host_slice(&self.loss2) };
                    idx.iter()
                        .filter_map(|&raw| {
                            let ix = raw as i32;
                            (ix >= 0).then(|| losses[ix as usize] as f64)
                        })
                        .sum()
                } else {
                    0.0
                };
                (l1 + l2) / valid as f64
            }
        }

        /// Per-position NLL of the last evaluated batch (`eval_loss` /
        /// `train_step`): cluster CE + within-cluster CE of the hierarchical
        /// head (or the flat CE), 0 at masked positions.
        pub fn per_position_loss(&self) -> Vec<f32> {
            let m = self.b * self.t;
            let mut out = unsafe { host_slice(&self.scratch.loss) }[..m].to_vec();
            if self.cfg.head_clusters > 0 {
                let l2 = unsafe { host_slice(&self.loss2) };
                for (o, x) in out.iter_mut().zip(&l2[..m]) {
                    *o += *x;
                }
            }
            out
        }

        /// Mean loss as the historical f32 API.  The f64 accessor above is
        /// used by numerical witnesses so the final average does not erase
        /// the small finite-difference signal after the GPU has already
        /// produced its f32 per-position losses.
        pub fn read_loss(&self) -> f32 {
            self.read_loss_f64() as f32
        }

        /// Encode one full training step's forward + backward (grads into
        /// `g`, zeroed first) + the grad-norm partials. Tokens/targets must
        /// already be in `tok`/`tgt`. Loss per position lands in scratch.loss.
        pub fn encode_fwd_bwd(&self, cmd: &Cmd) -> usize {
            self.encode_fwd_bwd_d(cmd, None)
        }

        /// As `encode_fwd_bwd`, plus feature distillation: with
        /// `distill = Some(w)` the teacher's final-normed hidden must sit
        /// in `scratch.xft`, and the head's dxf gains the MSE term
        /// (2w/M)·(xf − xft) — two axpby dispatches, no new kernels.
        pub fn encode_fwd_bwd_d(&self, cmd: &Cmd, distill: Option<f32>) -> usize {
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, self.b * self.t);
            let s = &self.scratch;
            cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total);
            self.begin_train_pass();
            // embed
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers {
                    self.x_in(l + 1)
                } else {
                    &self.x_out
                };
                self.layer_fwd(cmd, l, out, true);
            }
            self.desc_seeded.set(true);
            cmd.rmsnorm_fwd_at(
                &self.x_out,
                &self.p,
                self.lay.final_norm,
                &self.xf,
                &self.invf,
                m,
                h,
                cfg.norm_eps,
            );
            self.encode_head(cmd, true);
            if let Some(w) = distill {
                let c = 2.0 * w / (m * h) as f32;
                cmd.axpby(c, &self.xf, 1.0, &self.dxf, m * h);
                cmd.axpby(-c, &self.xft, 1.0, &self.dxf, m * h);
            }
            // final norm backward → s.dx
            cmd.rmsnorm_bwd_at(
                &self.x_out,
                &self.p,
                self.lay.final_norm,
                &self.dxf,
                &self.invf,
                &s.dx,
                0.0,
                &self.g,
                self.lay.final_norm,
                m,
                h,
            );
            for l in (0..cfg.layers).rev() {
                self.layer_bwd(cmd, l);
            }
            // embedding backward (tied: adds to the same rows the head wrote)
            cmd.embed_scatter_add(&self.g, self.lay.embed, &self.tok, &s.dx, m, h);
            cmd.sumsq(&self.g, self.lay.total, &s.partial)
        }

        /// Forward/backward for the lane-only final-hidden residual objective.
        /// The teacher hidden is supplied in `xft`; unlike feature
        /// distillation this path deliberately skips the vocabulary head and
        /// emits only the gradient of mean `||xf - xft||²`.  The complete
        /// graph is still traversed in the reverse pass so the appended lane
        /// can learn through the frozen trunk.  Callers select the trainable
        /// tail ranges when computing the norm and applying AdamW.
        pub fn encode_fwd_bwd_residual(&self, cmd: &Cmd) {
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, self.b * self.t);
            let s = &self.scratch;
            cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total);
            self.begin_train_pass();
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers {
                    self.x_in(l + 1)
                } else {
                    &self.x_out
                };
                self.layer_fwd(cmd, l, out, true);
            }
            self.desc_seeded.set(true);
            cmd.rmsnorm_fwd_at(
                &self.x_out,
                &self.p,
                self.lay.final_norm,
                &self.xf,
                &self.invf,
                m,
                h,
                cfg.norm_eps,
            );
            // d/dxf mean((xf-xft)^2), with the teacher buffer treated as
            // constant.  dxf is consumed by the existing final-norm reverse.
            let c = 2.0 / (m * h) as f32;
            cmd.axpby(c, &self.xf, 0.0, &self.dxf, m * h);
            cmd.axpby(-c, &self.xft, 1.0, &self.dxf, m * h);
            cmd.rmsnorm_bwd_at(
                &self.x_out,
                &self.p,
                self.lay.final_norm,
                &self.dxf,
                &self.invf,
                &s.dx,
                0.0,
                &self.g,
                self.lay.final_norm,
                m,
                h,
            );
            for l in (0..cfg.layers).rev() {
                self.layer_bwd(cmd, l);
            }
            cmd.embed_scatter_add(&self.g, self.lay.embed, &self.tok, &s.dx, m, h);
        }

        /// One optimiser step: AdamW over the arena with global-norm clip.
        /// `gnorm` is the pre-clip gradient norm (already computed).
        pub fn encode_adamw(&self, cmd: &Cmd, lr: f32, wd: f32, clip: f32, gnorm: f32, step: u32) {
            let gscale = if gnorm > clip { clip / gnorm } else { 1.0 };
            cmd.adamw(
                &self.p,
                &self.g,
                &self.m,
                &self.v,
                self.lay.total,
                lr,
                0.9,
                0.95,
                1e-8,
                wd,
                step,
                gscale,
            );
        }

        /// Full step: upload batch, fwd+bwd, clip, AdamW. Returns
        /// (mean loss, grad norm, gpu ms).
        pub fn train_step(
            &mut self,
            tokens: &[u32],
            targets: &[u32],
            lr: f32,
            wd: f32,
            clip: f32,
        ) -> (f32, f32, f64) {
            let m = self.b * self.t;
            assert!(tokens.len() == m && targets.len() == m);
            let c = self.ctx();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr(),
                    self.tok.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    targets.as_ptr(),
                    self.tgt.buf.contents() as *mut u32,
                    m,
                );
            }
            self.prepare_head(targets);
            let cmd = Cmd::new(c);
            let groups = self.encode_fwd_bwd(&cmd);
            let ms1 = cmd.commit();
            let mut gnorm = unsafe { host_slice(&self.scratch.partial) }[..groups]
                .iter()
                .map(|x| *x as f64)
                .sum::<f64>()
                .sqrt() as f32;
            if !self.freeze.is_empty() {
                let g = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.g.buf.contents() as *mut f32,
                        self.lay.total,
                    )
                };
                for &(o, l) in &self.freeze {
                    g[o..o + l].fill(0.0);
                }
                gnorm = g
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt() as f32;
            }
            let loss = self.read_loss();
            // SUMSQ covers every gradient element, including the final
            // partial tile.  Refuse the optimizer clock/update before any
            // AdamW dispatch if either the objective or complete reduction is
            // non-finite; a sparse host witness is not needed on the hot path.
            assert!(loss.is_finite(), "non-finite loss before AdamW: {loss:?}");
            assert!(
                gnorm.is_finite(),
                "non-finite gradient norm before AdamW: {gnorm:?}"
            );
            // Response-only batches may legally contain no answer token (for
            // example a fully padded/filtered microbatch).  The CE kernels
            // deliberately produce a finite zero objective and gradient for
            // that case.  Do not advance AdamW or apply decoupled weight
            // decay: an all-masked batch is a no-op, not a regularisation
            // step.
            if self.head_valid.get() == 0 {
                return (loss, gnorm, ms1);
            }
            self.step += 1;
            let cmd = Cmd::new(c);
            self.encode_adamw(&cmd, lr, wd, clip, gnorm, self.step);
            let ms2 = cmd.commit();
            (loss, gnorm, ms1 + ms2)
        }

        /// `train_step` with feature distillation from a teacher hidden:
        /// `xft` is the teacher's final-normed [M,H] for the same tokens.
        /// Returns (CE loss, distill MSE·w, grad norm, gpu ms).
        pub fn train_step_distill(
            &mut self,
            tokens: &[u32],
            targets: &[u32],
            lr: f32,
            wd: f32,
            clip: f32,
            xft: &[f32],
            w: f32,
        ) -> (f32, f32, f32, f64) {
            let m = self.b * self.t;
            let h = self.cfg.hidden;
            assert!(tokens.len() == m && targets.len() == m && xft.len() == m * h);
            let c = self.ctx();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr(),
                    self.tok.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    targets.as_ptr(),
                    self.tgt.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    xft.as_ptr(),
                    self.xft.buf.contents() as *mut f32,
                    m * h,
                );
            }
            self.prepare_head(targets);
            let cmd = Cmd::new(c);
            let groups = self.encode_fwd_bwd_d(&cmd, Some(w));
            let ms1 = cmd.commit();
            let mut gnorm = unsafe { host_slice(&self.scratch.partial) }[..groups]
                .iter()
                .map(|x| *x as f64)
                .sum::<f64>()
                .sqrt() as f32;
            if !self.freeze.is_empty() {
                // Unified memory: zero the frozen grads on the host, then
                // recompute the norm so the clip sees only the live params.
                let g = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.g.buf.contents() as *mut f32,
                        self.lay.total,
                    )
                };
                for &(o, l) in &self.freeze {
                    g[o..o + l].fill(0.0);
                }
                gnorm = g
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt() as f32;
            }
            let loss = self.read_loss();
            assert!(loss.is_finite(), "non-finite loss before AdamW: {loss:?}");
            assert!(
                gnorm.is_finite(),
                "non-finite gradient norm before AdamW: {gnorm:?}"
            );
            let dl = {
                let a = unsafe { host_slice(&self.xf) };
                let b = unsafe { host_slice(&self.xft) };
                let ss: f64 = a[..m * h]
                    .iter()
                    .zip(&b[..m * h])
                    .map(|(x, y)| ((x - y) as f64).powi(2))
                    .sum();
                (w as f64 * ss / (m * h) as f64) as f32
            };
            self.step += 1;
            let cmd = Cmd::new(c);
            self.encode_adamw(&cmd, lr, wd, clip, gnorm, self.step);
            let ms2 = cmd.commit();
            (loss, dl, gnorm, ms1 + ms2)
        }

        /// One lane-only final-hidden teacher-residual step.  Only the
        /// appended `Layout::gdn` ranges receive AdamW updates; all legacy
        /// parameters, moments, and descriptor records stay untouched.
        /// Returns `(residual_mse, tail_grad_norm, gpu_ms)`.
        pub fn train_step_residual(
            &mut self,
            tokens: &[u32],
            xft: &[f32],
            lr: f32,
            wd: f32,
            clip: f32,
        ) -> (f32, f32, f64) {
            use crate::metal::Cmd;
            let m = self.b * self.t;
            let h = self.cfg.hidden;
            assert!(tokens.len() == m && xft.len() == m * h);
            let c = self.ctx();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr(),
                    self.tok.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    xft.as_ptr(),
                    self.xft.buf.contents() as *mut f32,
                    m * h,
                );
            }
            let ranges = self.gdn_tail_ranges();
            assert!(!ranges.is_empty(), "residual pretrain requires gdn_lane");
            let cmd = Cmd::new(c);
            self.encode_fwd_bwd_residual(&cmd);
            let mut groups = Vec::with_capacity(ranges.len());
            let mut part_off = 0usize;
            for &(off, n) in &ranges {
                let g = cmd.sumsq_at(&self.g, off, n, &self.scratch.partial, part_off);
                groups.push((part_off, g));
                part_off += g;
            }
            let ms1 = cmd.commit();
            let part = unsafe { host_slice(&self.scratch.partial) };
            let gnorm = groups
                .iter()
                .map(|&(o, n)| part[o..o + n].iter().map(|x| *x as f64).sum::<f64>())
                .sum::<f64>()
                .sqrt() as f32;
            let xf = unsafe { host_slice(&self.xf) };
            let target = unsafe { host_slice(&self.xft) };
            let mse = xf[..m * h]
                .iter()
                .zip(&target[..m * h])
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum::<f64>() as f32
                / (m * h) as f32;
            assert!(mse.is_finite(), "non-finite residual loss before AdamW");
            assert!(
                gnorm.is_finite(),
                "non-finite residual gradient norm before AdamW: {gnorm:?}"
            );
            self.step += 1;
            let gscale = if gnorm > clip { clip / gnorm } else { 1.0 };
            let cmd = Cmd::new(c);
            for &(off, n) in &ranges {
                cmd.adamw_at(
                    &self.p, &self.g, &self.m, &self.v, off, n, lr, 0.9, 0.95, 1e-8, wd, self.step,
                    gscale,
                );
            }
            let ms2 = cmd.commit();
            (mse, gnorm, ms1 + ms2)
        }

        /// All appended GDN tensor ranges, one range per lane tensor.  The
        /// matrix blocks, scalar controls, and residual gain are intentionally
        /// listed separately because the layout keeps the scalar tail
        /// contiguous and matrix starts GEMM-aligned.
        pub fn gdn_tail_ranges(&self) -> Vec<(usize, usize)> {
            let h = self.cfg.hidden;
            let mut out = Vec::new();
            for go in self.lay.gdn.iter().flatten() {
                out.push((go.qkvz, 256 * h));
                out.push((go.conv, 192 * 4));
                out.push((go.ab, 64 * h));
                out.push((go.norm, 64));
                out.push((go.wo, h * 64));
                if go.alog != usize::MAX {
                    out.push((go.alog, 1));
                }
                if go.dt_bias != usize::MAX {
                    out.push((go.dt_bias, 1));
                }
                if go.gain != usize::MAX {
                    out.push((go.gain, 1));
                }
            }
            out
        }

        /// The four append-only layer-4 GQA lane tensors, suitable for
        /// lane-only gradient norm/update accounting.
        pub fn gqa_tail_ranges(&self) -> Vec<(usize, usize)> {
            let qn = self.cfg.anchor_q_heads * self.cfg.anchor_hd * self.cfg.hidden;
            let kn = self.cfg.anchor_kv_heads * self.cfg.anchor_hd * self.cfg.hidden;
            self.lay
                .gqa
                .iter()
                .flatten()
                .flat_map(|go| {
                    [
                        (go.q, qn),
                        (go.k, kn),
                        (go.v, kn),
                        (
                            go.wo,
                            self.cfg.hidden * self.cfg.anchor_q_heads * self.cfg.anchor_hd,
                        ),
                    ]
                })
                .collect()
        }

        /// Check finiteness of all appended GQA lane params and moments.
        pub fn gqa_tail_finite(&self) -> bool {
            let p = unsafe { host_slice(&self.p) };
            let m = unsafe { host_slice(&self.m) };
            let v = unsafe { host_slice(&self.v) };
            self.gqa_tail_ranges().iter().all(|&(off, n)| {
                p[off..off + n]
                    .iter()
                    .chain(&m[off..off + n])
                    .chain(&v[off..off + n])
                    .all(|x| x.is_finite())
            })
        }

        /// One ordinary CE step while updating only the additive GQA lane.
        /// The complete graph is differentiated, but every legacy arena
        /// range is excluded from AdamW, preserving the frozen student bytes.
        pub fn train_step_gqa_lane(
            &mut self,
            tokens: &[u32],
            targets: &[u32],
            lr: f32,
            wd: f32,
            clip: f32,
        ) -> (f32, f32, f64) {
            let ranges = self.gqa_tail_ranges();
            assert!(!ranges.is_empty(), "gqa lane training requires gqa_lane");
            let t0 = std::time::Instant::now();
            let (loss, gnorm) = self.train_step_ranges(tokens, targets, lr, wd, clip, &ranges);
            (loss, gnorm, t0.elapsed().as_secs_f64() * 1e3)
        }

        /// Mean squared final-hidden error against a host teacher buffer after
        /// `forward_hidden` has populated the candidate output.
        pub fn hidden_mse(hidden: &[f32], teacher: &[f32]) -> f32 {
            assert_eq!(hidden.len(), teacher.len());
            (hidden
                .iter()
                .zip(teacher)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum::<f64>()
                / hidden.len() as f64) as f32
        }

        /// Forward-only final-hidden MSE for a token batch and host teacher
        /// output.  This is intentionally separate from the CE evaluation
        /// path so residual gates cannot accidentally include head gradients.
        pub fn eval_hidden_mse(&self, tokens: &[u32], teacher: &[f32]) -> f32 {
            let hidden = self.forward_hidden(tokens);
            Self::hidden_mse(&hidden, teacher)
        }

        /// Check finiteness of every appended parameter/moment range.
        pub fn gdn_tail_finite(&self) -> bool {
            let p = unsafe { host_slice(&self.p) };
            let m = unsafe { host_slice(&self.m) };
            let v = unsafe { host_slice(&self.v) };
            self.gdn_tail_ranges().iter().all(|&(off, n)| {
                p[off..off + n]
                    .iter()
                    .chain(&m[off..off + n])
                    .chain(&v[off..off + n])
                    .all(|x| x.is_finite())
            })
        }

        /// Check finiteness of the recurrent state snapshots from every
        /// non-anchor lane.  This catches unstable scans even when the tail
        /// parameters and moments themselves remain finite.
        pub fn gdn_state_finite(&self) -> bool {
            self.acts.iter().all(|a| match a {
                LayerActs::Mixer { gdn_states, .. } => unsafe { host_slice(gdn_states) }
                    .iter()
                    .all(|x| x.is_finite()),
                LayerActs::Gdn { states, .. } => unsafe { host_slice(states) }
                    .iter()
                    .all(|x| x.is_finite()),
                LayerActs::Anchor { .. } => true,
            })
        }

        /// Forward only (no grads): mean loss on a batch already in tok/tgt.
        pub fn eval_loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
            let m = self.b * self.t;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr(),
                    self.tok.buf.contents() as *mut u32,
                    m,
                );
                std::ptr::copy_nonoverlapping(
                    targets.as_ptr(),
                    self.tgt.buf.contents() as *mut u32,
                    m,
                );
            }
            self.prepare_head(targets);
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, m);
            let cmd = Cmd::new(self.ctx());
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers {
                    self.x_in(l + 1)
                } else {
                    &self.x_out
                };
                self.layer_fwd(&cmd, l, out, false);
            }
            cmd.rmsnorm_fwd_at(
                &self.x_out,
                &self.p,
                self.lay.final_norm,
                &self.xf,
                &self.invf,
                m,
                h,
                cfg.norm_eps,
            );
            self.encode_head(&cmd, false);
            cmd.commit();
            self.read_loss()
        }

        pub fn params_host(&self) -> Vec<f32> {
            self.p.to_vec()
        }
        pub fn grads_host(&self) -> Vec<f32> {
            self.g.to_vec()
        }
        pub fn set_params(&self, p: &[f32]) {
            self.p.write_from(p);
        }

        /// Read the accepted Phase-Delta recurrence back on the host for a
        /// terminal dynamics witness.  The scan intentionally samples every
        /// 16th value channel: it is sufficient to catch a dead correction,
        /// exploding state, or pinned gate while avoiding a second full
        /// training step.  State boundaries and all projection gradients are
        /// read only after the caller's timed command buffer has completed.
        pub fn phase_delta_telemetry(&self) -> Vec<PhaseDeltaLayerTelemetry> {
            if !self.cfg.phase_delta_active() {
                return Vec::new();
            }
            let cfg = &self.cfg;
            let p2 = 2 * cfg.nphase;
            let nch = self.t.div_ceil(64);
            let mut rows = Vec::new();
            let quantile = |x: &mut Vec<f32>, q: f32| -> f32 {
                if x.is_empty() {
                    return f32::NAN;
                }
                x.sort_by(|a, b| a.total_cmp(b));
                let k = ((x.len() - 1) as f32 * q).round() as usize;
                x[k.min(x.len() - 1)]
            };
            let l2 = |buf: &[f32], off: usize, n: usize| -> f32 {
                buf[off..off + n]
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt() as f32
            };
            let grads = unsafe { host_slice(&self.g) };
            for (l, act) in self.acts.iter().enumerate() {
                let LayerActs::Mixer {
                    phk,
                    v,
                    kappa,
                    states,
                    ..
                } = act
                else {
                    continue;
                };
                // Report only the layers actually running Phase-Delta.  In
                // selected-layer candidates this intentionally excludes all
                // unselected legacy hybrid layers (including the dual mode).
                if !cfg.phase_delta_for_layer(l) {
                    continue;
                }
                let mut beta = unsafe { host_slice(kappa) }.to_vec();
                let finite_beta = beta.iter().all(|x| x.is_finite());
                let beta_p01 = quantile(&mut beta, 0.01);
                let beta_p50 = quantile(&mut beta, 0.50);
                let beta_p99 = quantile(&mut beta, 0.99);
                let phk = unsafe { host_slice(phk) };
                let vals = unsafe { host_slice(v) };
                let kap = unsafe { host_slice(kappa) };
                let st = unsafe { host_slice(states) };
                let mut r2 = 0.0f64;
                let mut e2 = 0.0f64;
                let mut corr2 = 0.0f64;
                let mut v2 = 0.0f64;
                let mut state2 = 0.0f64;
                let mut state_max = 0.0f32;
                let mut n_scalar = 0usize;
                let mut n_state = 0usize;
                let mut finite = finite_beta;
                // A deterministic, bounded channel sample for the expensive
                // recurrence witness. The state itself remains the exact
                // shared boundary buffer produced by Metal.
                for b in 0..self.b {
                    for h in 0..cfg.heads {
                        let bh = b * cfg.heads + h;
                        let gam_base = self.pow_off(l) + (h * 65 + 1) * p2;
                        for d in (0..cfg.dv).step_by(16) {
                            let mut s = vec![0.0f32; p2];
                            let st_base = bh * (nch + 1) * p2 * cfg.dv;
                            for f in 0..p2 {
                                s[f] = st[st_base + f * cfg.dv + d];
                                finite &= s[f].is_finite();
                            }
                            for t in 0..self.t {
                                if t % 64 == 0 && t != 0 {
                                    let c = t / 64;
                                    for f in 0..p2 {
                                        s[f] = st[st_base + (c * p2 + f) * cfg.dv + d];
                                        finite &= s[f].is_finite();
                                    }
                                }
                                let row = (b * self.t + t) * cfg.heads + h;
                                let ph_base = row * p2;
                                let mut r = 0.0f32;
                                for f in 0..p2 {
                                    let g = unsafe { host_slice(&self.pow) }[gam_base + f];
                                    r += phk[ph_base + f] * (g * s[f]);
                                }
                                let vv = vals[row * cfg.dv + d];
                                let bb = kap[row];
                                let e = vv - r;
                                let mut cc2 = 0.0f64;
                                for f in 0..p2 {
                                    let corr = bb * phk[ph_base + f] * e;
                                    cc2 += (corr as f64) * (corr as f64);
                                    s[f] = unsafe { host_slice(&self.pow) }[gam_base + f] * s[f]
                                        + corr;
                                    state2 += (s[f] as f64) * (s[f] as f64);
                                    state_max = state_max.max(s[f].abs());
                                    finite &= s[f].is_finite();
                                }
                                r2 += (r as f64) * (r as f64);
                                e2 += (e as f64) * (e as f64);
                                corr2 += cc2;
                                v2 += (vv as f64) * (vv as f64);
                                n_scalar += 1;
                                n_state += p2;
                                finite &= r.is_finite() && e.is_finite() && vv.is_finite();
                            }
                        }
                    }
                }
                let lo = match &self.lay.layers[l] {
                    LayerOffs::Mixer { wk, wv, wkap, .. } => (*wk, *wv, *wkap),
                    LayerOffs::Anchor { .. } | LayerOffs::Gdn { .. } => unreachable!(),
                };
                let wk_grad_l2 = l2(grads, lo.0, cfg.heads * cfg.nphase * cfg.hidden);
                let wv_grad_l2 = l2(grads, lo.1, cfg.heads * cfg.dv * cfg.hidden);
                let wkappa_grad_l2 = l2(grads, lo.2, cfg.kappa_ld() * cfg.hidden);
                finite &=
                    wk_grad_l2.is_finite() && wv_grad_l2.is_finite() && wkappa_grad_l2.is_finite();
                rows.push(PhaseDeltaLayerTelemetry {
                    layer: l,
                    beta_p01,
                    beta_p50,
                    beta_p99,
                    r_rms: (r2 / n_scalar.max(1) as f64).sqrt() as f32,
                    e_rms: (e2 / n_scalar.max(1) as f64).sqrt() as f32,
                    correction_rms: (corr2 / n_state.max(1) as f64).sqrt() as f32,
                    v_rms: (v2 / n_scalar.max(1) as f64).sqrt() as f32,
                    state_rms: (state2 / n_state.max(1) as f64).sqrt() as f32,
                    state_max,
                    wk_grad_l2,
                    wv_grad_l2,
                    wkappa_grad_l2,
                    finite,
                });
            }
            rows
        }
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// Per-phase GPU time of one step (each phase its own command buffer):
    /// the profile the kernel work is prioritised from.
    pub fn profile_step(&self) -> Vec<(String, f64)> {
        use crate::metal::Cmd;
        let cfg = &self.cfg;
        let (h, m) = (cfg.hidden, self.b * self.t);
        let s = &self.scratch;
        let c = self.ctx();
        let mut out = Vec::new();
        let time = |_name: String, f: &dyn Fn(&Cmd)| -> f64 {
            let cmd = Cmd::new(c);
            f(&cmd);
            cmd.commit()
        };
        self.begin_train_pass();
        out.push((
            "zero grads".into(),
            time("z".into(), &|cmd| {
                cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total)
            }),
        ));
        out.push((
            "embed".into(),
            time("e".into(), &|cmd| {
                cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h)
            }),
        ));
        for l in 0..cfg.layers {
            let ms = time(format!("fwd {l}"), &|cmd| {
                let o: &crate::metal::GBuf = if l + 1 < cfg.layers {
                    self.x_in(l + 1)
                } else {
                    &self.x_out
                };
                self.layer_fwd(cmd, l, o, true);
            });
            out.push((
                format!(
                    "layer {l} fwd{}",
                    if cfg.is_anchor(l) { " (anchor)" } else { "" }
                ),
                ms,
            ));
        }
        out.push((
            "final norm".into(),
            time("f".into(), &|cmd| {
                cmd.rmsnorm_fwd_at(
                    &self.x_out,
                    &self.p,
                    self.lay.final_norm,
                    &self.xf,
                    &self.invf,
                    m,
                    h,
                    cfg.norm_eps,
                )
            }),
        ));
        out.push((
            "head fwd+bwd".into(),
            time("h".into(), &|cmd| self.encode_head(cmd, true)),
        ));
        out.push((
            "final norm bwd".into(),
            time("fb".into(), &|cmd| {
                cmd.rmsnorm_bwd_at(
                    &self.x_out,
                    &self.p,
                    self.lay.final_norm,
                    &self.dxf,
                    &self.invf,
                    &s.dx,
                    0.0,
                    &self.g,
                    self.lay.final_norm,
                    m,
                    h,
                )
            }),
        ));
        for l in (0..cfg.layers).rev() {
            let ms = time(format!("bwd {l}"), &|cmd| self.layer_bwd(cmd, l));
            out.push((
                format!(
                    "layer {l} bwd{}",
                    if cfg.is_anchor(l) { " (anchor)" } else { "" }
                ),
                ms,
            ));
        }
        out.push((
            "embed bwd".into(),
            time("eb".into(), &|cmd| {
                cmd.embed_scatter_add(&self.g, self.lay.embed, &self.tok, &s.dx, m, h)
            }),
        ));
        out.push((
            "grad norm".into(),
            time("gn".into(), &|cmd| {
                cmd.sumsq(&self.g, self.lay.total, &s.partial);
            }),
        ));
        out.push((
            "adamw".into(),
            time("a".into(), &|cmd| {
                self.encode_adamw(cmd, 1e-4, 0.1, 1.0, 1.0, 1)
            }),
        ));
        out
    }

    /// Per-kernel profile of one hybrid_k mixer layer's forward + backward
    /// (kernel by kernel, each its own command buffer).
    pub fn profile_hk_layer(&self, l: usize) -> Vec<(String, f64)> {
        use crate::metal::{Cmd, HkDims, HkGrads, HkWork};
        let cfg = &self.cfg;
        let (b, t) = (self.b, self.t);
        let s = &self.scratch;
        let c = self.ctx();
        let LayerActs::Mixer {
            thq,
            thk,
            v,
            kappa,
            phq,
            phk,
            kv,
            states,
            o,
            ..
        } = &self.acts[l]
        else {
            return vec![];
        };
        let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
        let d = HkDims { b, t, nh, nph, dv };
        let use_phase_delta = cfg.phase_delta_for_layer(l);
        let w = HkWork {
            thq,
            thk,
            v,
            kappa,
            pow: &self.pow,
            pow_off: self.pow_off(l),
            phq,
            phk,
            kv,
            states,
            out: o,
            phase_chunk: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                || use_phase_delta
            {
                Some(&s.phase_chunk)
            } else {
                None
            },
            phase_partial: if cfg!(all(feature = "vulkan", not(target_os = "macos")))
                || use_phase_delta
            {
                Some(&s.phase_partial)
            } else {
                None
            },
        };
        let gr = HkGrads {
            dout: &s.dbig,
            dstates: &s.dstates,
            dkv: &s.dkv,
            dphq: &s.dphq,
            dphk: &s.dphk,
            dthq: &s.dk,
            dthk: &s.dk2,
            dv: &s.dv,
            dkappa: &s.dkap,
        };
        let mut out = Vec::new();
        if use_phase_delta {
            let cmd = Cmd::new(c);
            cmd.phase_delta_forward_reset(&d, &w);
            out.push((
                "phase_delta scan forward (32-lane blocks)".into(),
                cmd.commit(),
            ));
            let cmd = Cmd::new(c);
            cmd.phase_delta_backward(&d, &w, &gr);
            out.push((
                "phase_delta scan backward (blocks/fold)".into(),
                cmd.commit(),
            ));
            return out;
        }
        let cmd = Cmd::new(c);
        cmd.hk_forward(&d, &w);
        out.push((
            "hk forward SIMT (φ, kv, states, chunks)".into(),
            cmd.commit(),
        ));
        let cmd = Cmd::new(c);
        cmd.hk_backward(&d, &w, &gr, 0.0);
        out.push((
            "hk backward SIMT (dstates, chunks, split, dθ)".into(),
            cmd.commit(),
        ));
        let sc = self.hk_scratch();
        let cmd = Cmd::new(c);
        cmd.hk_forward_gemm(&d, &w, &sc);
        out.push(("hk forward GEMM".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_backward_gemm(&d, &w, &gr, &sc, 0.0);
        out.push(("hk backward GEMM".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_states_only(&d, &w);
        out.push(("  states scan SIMT (fwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_dstates_only(&d, &w, &gr);
        out.push(("  dstates scan SIMT (bwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_states_par(&d, &w);
        out.push(("  states scan cell-parallel (fwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_dstates_par(&d, &w, &gr);
        out.push(("  dstates scan cell-parallel (bwd)".into(), cmd.commit()));
        out
    }
}

/// `EMBRYO_HK_SIMT=1` selects the reference SIMT chunk kernels instead of
/// the batched-GEMM formulation (A/B and debugging).
#[cfg(any(target_os = "macos", feature = "vulkan"))]
/// Vulkan training head: table-batched cluster GEMMs (default; three
/// dispatches instead of up to 3·head_clusters) vs the per-cluster loops
/// (`CMF_VULKAN_HEAD_TABLE=0` at startup, or [`HEAD_TABLE`] at runtime for
/// in-process A/B). Bit-identical to the loops (same per-output sums);
/// every group is dispatched with the widest group's tile rows and rows
/// past a group's own count return early. Measured on the RTX PRO 4000:
/// head pass 29.4 → 22.9 ms (balanced targets), full B8/T512 step on the
/// natural train mix 1433 → 1339 ms (same batch, idle GPU).
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
pub static HEAD_TABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
fn head_table_enabled() -> bool {
    static ENV_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env_off =
        *ENV_OFF.get_or_init(|| std::env::var("CMF_VULKAN_HEAD_TABLE").ok().as_deref() == Some("0"));
    !env_off && HEAD_TABLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Anchor backward dispatch shape: one range per sequence (default — the
/// measured faster shape on both backends: Metal B8/T512 probe 32.0 vs
/// 34.0 ms, RTX PRO 4000 303.6 vs 326.5 ms, full step 1069 vs 1103 ms) or
/// one batched range over all sequences (`CMF_ANCHOR_BWD_BATCH=1` at
/// startup, or [`ANCHOR_BWD_BATCH`] at runtime for in-process A/B). The
/// gradients agree to ≤ 3e-7 rel (identical without sinks).
pub static ANCHOR_BWD_BATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn anchor_bwd_seq() -> bool {
    static ENV_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env_on = *ENV_ON
        .get_or_init(|| std::env::var("CMF_ANCHOR_BWD_BATCH").ok().as_deref() == Some("1"));
    !(env_on || ANCHOR_BWD_BATCH.load(std::sync::atomic::Ordering::Relaxed))
}

fn hk_simt() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("EMBRYO_HK_SIMT").is_ok_and(|v| v != "0"))
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// Routing statistics of the last step: per layer, tokens per expert.
    pub fn routing_counts(&self) -> Vec<Vec<u32>> {
        let ne = self.cfg.experts;
        if ne == 0 {
            return vec![];
        }
        let c = unsafe {
            std::slice::from_raw_parts(
                self.desc.count.buf.contents() as *const u32,
                self.cfg.layers * ne,
            )
        };
        (0..self.cfg.layers)
            .map(|l| c[l * ne..(l + 1) * ne].to_vec())
            .collect()
    }

    /// Conditional top-2 fallback counts from the last routed forward. This
    /// is activation telemetry only (one transient counter per layer); an
    /// empty vector is returned for legacy/top-1 configurations.
    pub fn routing_top2_fallbacks(&self) -> Vec<u32> {
        if !self.cfg.router_top2_enabled() {
            return vec![];
        }
        self.moe
            .iter()
            .map(|mo| {
                unsafe { host_u32_slice(&mo.fallback_count) }
                    .first()
                    .copied()
                    .unwrap_or(0)
            })
            .collect()
    }

    /// `(fallbacks, runner_capacity_drops)` per layer from the last routed
    /// forward. Counts are host-read activation telemetry; no state is saved
    /// in checkpoints and legacy configurations return an empty vector.
    pub fn routing_top2_telemetry(&self) -> Vec<(u32, u32)> {
        if !self.cfg.router_top2_enabled() {
            return vec![];
        }
        self.moe
            .iter()
            .map(|mo| {
                let fallback = unsafe { host_u32_slice(&mo.fallback_count) }
                    .first()
                    .copied()
                    .unwrap_or(0);
                let drops = unsafe { host_u32_slice(&mo.count2) }
                    .iter()
                    .map(|&n| n.saturating_sub(self.moe_cap as u32))
                    .sum();
                (fallback, drops)
            })
            .collect()
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// The expert descriptors (routing state that is part of the model).
    pub fn desc_host(&self) -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("desc.mu", self.desc.mu.to_vec()),
            ("desc.u", self.desc.u.to_vec()),
            ("desc.bias", self.desc.bias.to_vec()),
        ]
    }
    pub fn set_desc(&self, extras: &[(String, Vec<f32>)]) {
        if !extras.is_empty() {
            self.desc_seeded.set(true);
        }
        for (name, x) in extras {
            match name.as_str() {
                "desc.mu" if x.len() == self.desc.mu.len => self.desc.mu.write_from(x),
                "desc.u" if x.len() == self.desc.u.len => self.desc.u.write_from(x),
                "desc.bias" if x.len() == self.desc.bias.len => self.desc.bias.write_from(x),
                _ => {}
            }
        }
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// Forward (no head, no grads): the final-normed hidden xf [M, H] for the
    /// tokens — the runtime-parity test's reference.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Vec<f32> {
        use crate::metal::{Cmd, GBuf};
        let m = self.b * self.t;
        assert_eq!(tokens.len(), m);
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m)
        };
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let cmd = Cmd::new(self.ctx());
        cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
        for l in 0..cfg.layers {
            let out: &GBuf = if l + 1 < cfg.layers {
                self.x_in(l + 1)
            } else {
                &self.x_out
            };
            self.layer_fwd(&cmd, l, out, false);
        }
        cmd.rmsnorm_fwd_at(
            &self.x_out,
            &self.p,
            self.lay.final_norm,
            &self.xf,
            &self.invf,
            m,
            h,
            cfg.norm_eps,
        );
        cmd.commit();
        self.xf.to_vec()
    }

    /// The FFN / routed-expert input of layer `l` (`x2 = rmsnorm(x_mid)`,
    /// `[M, H]`) as left by the last forward — the vector the resonance
    /// router scores (trainer `route_f32`, runtime `Resonance::scores`).
    /// Growth reads it after an eval forward to recompute the routing on the
    /// host with the runtime formula (shells, routing shift, coverage).
    pub fn ffn_input_host(&self, l: usize) -> Vec<f32> {
        match &self.acts[l] {
            LayerActs::Mixer { x2, .. }
            | LayerActs::Anchor { x2, .. }
            | LayerActs::Gdn { x2, .. } => x2.to_vec(),
        }
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// Descriptor subspaces (P1): per (layer, expert) covariance of the
    /// centred routed inputs of the LAST forward (GEMM on the GPU, K = the
    /// expert's row count), EMA'd on the host, top-K eigenvectors by block
    /// power iteration → `desc.u` (orthonormal rows). Call every N steps.
    pub fn update_subspaces(&mut self, cov_ema: &mut Vec<f32>, ema: f32) {
        use crate::metal::{Cmd, GemmBatch, GemmDyn, Op};
        let cfg = &self.cfg;
        let (ne, h, l_n) = (cfg.experts, cfg.hidden, cfg.layers);
        if ne == 0 {
            return;
        }
        let k = MOE_K;
        let cap = self.moe_cap;
        let r = self.route_dims();
        let c = self.ctx();
        // one covariance buffer for all (layer, expert): [L·E, H, H]
        let cov = crate::metal::GBuf::zeros(c, l_n * ne * h * h);
        let cmd = Cmd::new(c);
        for l in 0..l_n {
            let mo = &self.moe[l];
            let d = &self.desc;
            let (mu_off, e_off) = (l * ne * h, l * ne);
            cmd.moe_center(
                &r,
                &mo.hg,
                &d.mu,
                mu_off,
                &d.count,
                e_off,
                &self.scratch.moe_dyh,
            );
            for e in 0..ne {
                let kdyn = GemmDyn {
                    indirect: None,
                    kcount: Some((&d.count, e_off + e)),
                };
                let hg_o = e * cap * h;
                // cov_e = hgcᵀ·hgc  ([H, rows]·[rows, H])
                cmd.gemm_dyn(
                    Op::T,
                    Op::N,
                    h,
                    h,
                    cap,
                    1.0,
                    &self.scratch.moe_dyh,
                    hg_o,
                    h,
                    &self.scratch.moe_dyh,
                    hg_o,
                    h,
                    0.0,
                    &cov,
                    (l * ne + e) * h * h,
                    h,
                    &GemmBatch::none(),
                    false,
                    &kdyn,
                );
            }
        }
        cmd.commit();
        let cov_h = cov.to_vec();
        let counts: Vec<u32> = unsafe {
            std::slice::from_raw_parts(self.desc.count.buf.contents() as *const u32, l_n * ne)
                .to_vec()
        };
        if cov_ema.len() != cov_h.len() {
            *cov_ema = vec![0.0; cov_h.len()];
        }
        let mut u_all = vec![0.0f32; l_n * ne * k * h];
        for le in 0..l_n * ne {
            let n = counts[le].min(cap as u32) as f32;
            let blk = &cov_h[le * h * h..(le + 1) * h * h];
            let ema_blk = &mut cov_ema[le * h * h..(le + 1) * h * h];
            if n >= 2.0 {
                for i in 0..h * h {
                    ema_blk[i] = ema * ema_blk[i] + (1.0 - ema) * blk[i] / n;
                }
            }
            let u = top_eigenvectors(ema_blk, h, k, 24, le as u64);
            u_all[le * k * h..(le + 1) * k * h].copy_from_slice(&u);
        }
        self.desc.u.write_from(&u_all);
    }
}

/// Top-k eigenvectors of a symmetric [n×n] matrix by block power iteration
/// with Gram–Schmidt re-orthonormalisation; rows of the result are the
/// orthonormal directions (largest eigenvalues first, approximately).
pub fn top_eigenvectors(a: &[f32], n: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut q: Vec<f32> = gauss_vec(seed.wrapping_add(777), k * n);
    let mut tmp = vec![0.0f32; k * n];
    let orth = |q: &mut [f32]| {
        for i in 0..k {
            for j in 0..i {
                let dot: f32 = (0..n).map(|t| q[i * n + t] * q[j * n + t]).sum();
                for t in 0..n {
                    q[i * n + t] -= dot * q[j * n + t];
                }
            }
            let nrm: f32 = (0..n)
                .map(|t| q[i * n + t] * q[i * n + t])
                .sum::<f32>()
                .sqrt();
            if nrm > 1e-12 {
                for t in 0..n {
                    q[i * n + t] /= nrm;
                }
            }
        }
    };
    orth(&mut q);
    for _ in 0..iters {
        // tmp = Q·A (A symmetric)
        for i in 0..k {
            for c in 0..n {
                let mut s = 0.0f32;
                for t in 0..n {
                    s += q[i * n + t] * a[t * n + c];
                }
                tmp[i * n + c] = s;
            }
        }
        q.copy_from_slice(&tmp);
        orth(&mut q);
    }
    q
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
use crate::metal::GBuf;

/// Borrow a device buffer's host mirror only at an explicitly unsafe
/// boundary. Vulkan uses a private readback mirror rather than Metal's shared
/// allocation, so a borrowed slice must not outlive the sequenced host/device
/// interval documented by `GBuf::as_slice`; keeping this helper unsafe
/// prevents a safe wrapper from reintroducing that alias.
#[cfg(any(target_os = "macos", feature = "vulkan"))]
#[inline]
unsafe fn host_slice<'a>(buf: &'a GBuf) -> &'a [f32] {
    unsafe { buf.as_slice() }
}

/// u32 counterpart of [`host_slice`].
#[cfg(any(target_os = "macos", feature = "vulkan"))]
#[inline]
unsafe fn host_u32_slice<'a>(buf: &'a GBuf) -> &'a [u32] {
    unsafe { buf.as_u32_slice() }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
/// Skill-bake state: DTG-MA neuron masks over the shared FFN of the
/// selected layers (one logit per neuron), with their AdamW state.
pub struct SkillState {
    pub layers: Vec<usize>,
    pub inter: usize,
    /// [layers.len(), I] mask logits
    pub logits: GBuf,
    pub g: GBuf,
    pub m: GBuf,
    pub v: GBuf,
    /// apply the binarized mask 1[σ>τ] instead of σ
    pub hard: std::cell::Cell<bool>,
    pub tau: f32,
    /// L1 pressure on σ(m) (phase A), added to the mask gradient
    pub l1: std::cell::Cell<f32>,
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl SkillState {
    pub fn new(
        c: &crate::metal::Ctx,
        layers: Vec<usize>,
        inter: usize,
        init_logit: f32,
        tau: f32,
    ) -> SkillState {
        let n = layers.len() * inter;
        SkillState {
            layers,
            inter,
            logits: GBuf::from_slice(c, &vec![init_logit; n.max(1)]),
            g: GBuf::zeros(c, n),
            m: GBuf::zeros(c, n),
            v: GBuf::zeros(c, n),
            hard: std::cell::Cell::new(false),
            tau,
            l1: std::cell::Cell::new(0.0),
        }
    }
    pub fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }
    /// Kept-neuron mask per selected layer from the current logits.
    pub fn hard_masks(&self) -> Vec<Vec<bool>> {
        let lg = self.logits.to_vec();
        self.layers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                lg[i * self.inter..(i + 1) * self.inter]
                    .iter()
                    .map(|&m| 1.0 / (1.0 + (-m).exp()) > self.tau)
                    .collect()
            })
            .collect()
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// Σx² over `ranges` (buffer, offset, len), encoded after the work
    /// already in `cmd`, which is committed here. The grad-norm partial
    /// buffer holds a fixed number of per-threadgroup sums; a trainable set
    /// needing more (e.g. the six FFN ranges of a two-layer fam-a skill:
    /// 6 × 1152 groups > 4096) continues in further command buffers, so no
    /// range is dropped. Before, every range past the last partial got 0
    /// groups: its gradient was missing from the norm, the clip and the
    /// non-finite guard. Both backends' sumsq kernels grid-stride, so any
    /// group count ≥ 1 covers the whole range.
    pub(crate) fn sumsq_ranges(
        &self,
        cmd: crate::metal::Cmd<'_>,
        ranges: &[(&GBuf, usize, usize)],
    ) -> f64 {
        use crate::metal::Cmd;
        let part = &self.scratch.partial;
        let cap = part.len;
        let mut total = 0f64;
        let mut first = Some(cmd);
        let mut idx = 0usize;
        loop {
            let cmd = first.take().unwrap_or_else(|| Cmd::new(self.ctx()));
            let mut groups: Vec<(usize, usize)> = Vec::new();
            let mut poff = 0usize;
            while idx < ranges.len() {
                let (buf, off, n) = ranges[idx];
                let need = n.div_ceil(256).clamp(1, 4096).min(cap);
                if poff > 0 && poff + need > cap {
                    break;
                }
                let g = cmd.sumsq_at(buf, off, n, part, poff);
                groups.push((poff, g));
                poff += g;
                idx += 1;
            }
            cmd.commit();
            let host = part.to_vec();
            total += groups
                .iter()
                .map(|&(o, g)| host[o..o + g].iter().map(|x| *x as f64).sum::<f64>())
                .sum::<f64>();
            if idx >= ranges.len() {
                return total;
            }
        }
    }

    /// One skill-bake step: full fwd/bwd with the masks active; then AdamW
    /// over the trainable set only — phase A: the mask logits; phase B: the
    /// shared-FFN tensors of the selected layers. Everything else stays
    /// byte-identical. Returns (loss, grad norm of the trainable set).
    pub fn train_step_skill(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        lr: f32,
        wd: f32,
        clip: f32,
        phase_b: bool,
    ) -> (f32, f32) {
        use crate::metal::Cmd;
        let m = self.b * self.t;
        assert!(tokens.len() == m && targets.len() == m);
        let c = self.ctx();
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m);
            std::ptr::copy_nonoverlapping(targets.as_ptr(), self.tgt.buf.contents() as *mut u32, m);
        }
        self.prepare_head(targets);
        let sk = self.skill.as_ref().expect("skill state");
        // trainable ranges of the arena (phase B): shared FFN of the selected layers
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        if phase_b {
            let (h, i) = (self.cfg.hidden, self.cfg.inter);
            for &l in &sk.layers {
                let ffn = match &self.lay.layers[l] {
                    LayerOffs::Mixer { ffn, .. }
                | LayerOffs::Anchor { ffn, .. }
                | LayerOffs::Gdn { ffn, .. } => ffn,
                };
                ranges.push((ffn.wg, i * h));
                ranges.push((ffn.wu, i * h));
                ranges.push((ffn.wd, h * i));
            }
        }
        let cmd = Cmd::new(c);
        cmd.axpby(0.0, &sk.g, 0.0, &sk.g, sk.g.len);
        self.encode_fwd_bwd(&cmd);
        // grad norm over the trainable set
        let norm_ranges: Vec<(&GBuf, usize, usize)> = if phase_b {
            ranges.iter().map(|&(off, n)| (&self.g, off, n)).collect()
        } else {
            vec![(&sk.g, 0, sk.g.len)]
        };
        let gnorm = self.sumsq_ranges(cmd, &norm_ranges).sqrt() as f32;
        let loss = self.read_loss();
        assert!(loss.is_finite(), "non-finite loss before AdamW: {loss:?}");
        assert!(
            gnorm.is_finite(),
            "non-finite range gradient norm before AdamW: {gnorm:?}"
        );
        let gscale = if gnorm > clip { clip / gnorm } else { 1.0 };
        self.step += 1;
        let cmd = Cmd::new(c);
        if phase_b {
            for &(off, n) in &ranges {
                cmd.adamw_at(
                    &self.p, &self.g, &self.m, &self.v, off, n, lr, 0.9, 0.95, 1e-8, wd, self.step,
                    gscale,
                );
            }
        } else {
            cmd.adamw(
                &sk.logits, &sk.g, &sk.m, &sk.v, sk.g.len, lr, 0.9, 0.95, 1e-8, 0.0, self.step,
                gscale,
            );
        }
        cmd.commit();
        (loss, gnorm)
    }

    /// Mean-pooled hidden state AFTER layer `phi_layer` (the runtime's
    /// probe_phi semantics: layers 0..=phi_layer) for each sequence of the
    /// batch — the P1 routing descriptor's φ(x), [B, H].
    pub fn probe_phi(&self, tokens: &[u32], phi_layer: usize, phi_len: usize) -> Vec<Vec<f32>> {
        let n = phi_len.clamp(1, self.t);
        let spans = vec![0..n; self.b];
        self.probe_phi_span(tokens, phi_layer, &spans)
    }

    /// The residual stream AFTER layer `layer` (layers `0..=layer`, the
    /// evaluation forward: served anchor window, no descriptor updates) for
    /// every position of the batch, `[B·T, H]` row-major, as an owned host
    /// copy.
    pub fn hidden_after_layer(&self, tokens: &[u32], layer: usize) -> Vec<f32> {
        use crate::metal::Cmd;
        let m = self.b * self.t;
        assert_eq!(tokens.len(), m);
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m)
        };
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let cmd = Cmd::new(self.ctx());
        cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
        let last = layer.min(cfg.layers - 1);
        for l in 0..=last {
            let out: &GBuf = if l + 1 < cfg.layers {
                self.x_in(l + 1)
            } else {
                &self.x_out
            };
            self.layer_fwd(&cmd, l, out, false);
        }
        cmd.commit();
        let x: &GBuf = if last + 1 < cfg.layers {
            self.x_in(last + 1)
        } else {
            &self.x_out
        };
        let mut out = x.to_vec();
        out.truncate(m * h);
        out
    }

    /// Span-mean φ (router v2, `pool = "span_mean"`): one prompt per batch
    /// row, right-padded to T; row r's φ is the mean of the hidden AFTER
    /// `phi_layer` over the positions `spans[r]` (the user-text tokens of
    /// the canonical `prefix ++ q ++ suffix` rendering). Every operator of
    /// the genome is causal (GDN scan, conv, bounded anchor), so the padding
    /// after a prompt never changes its span. Expert capacity is the one
    /// batch-coupled effect: probe on a `new_eval_dropless` instance so no
    /// row is dropped (the runtime routes every token). Returns the raw
    /// means (unnormalized), `[B][H]`; an empty span gives zeros.
    pub fn probe_phi_span(
        &self,
        tokens: &[u32],
        phi_layer: usize,
        spans: &[std::ops::Range<usize>],
    ) -> Vec<Vec<f32>> {
        assert_eq!(spans.len(), self.b, "one span per batch row");
        let h = self.cfg.hidden;
        let xs = self.hidden_after_layer(tokens, phi_layer);
        spans
            .iter()
            .enumerate()
            .map(|(bi, span)| {
                let (a, e) = (span.start.min(self.t), span.end.min(self.t));
                let mut acc = vec![0.0f32; h];
                if e <= a {
                    return acc;
                }
                for t in a..e {
                    let row = &xs[(bi * self.t + t) * h..(bi * self.t + t + 1) * h];
                    for (s, x) in acc.iter_mut().zip(row) {
                        *s += x;
                    }
                }
                let n = (e - a) as f32;
                for v in &mut acc {
                    *v /= n;
                }
                acc
            })
            .collect()
    }
}

#[cfg(any(target_os = "macos", feature = "vulkan"))]
impl EmbryoGpu {
    /// One step training ONLY the given arena ranges (everything else
    /// frozen; grad clip over the ranges) — growth of new experts, FCD of a
    /// subset, any append-only record. Returns (loss, grad norm).
    pub fn train_step_ranges(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        lr: f32,
        wd: f32,
        clip: f32,
        ranges: &[(usize, usize)],
    ) -> (f32, f32) {
        use crate::metal::Cmd;
        let m = self.b * self.t;
        assert!(tokens.len() == m && targets.len() == m);
        let c = self.ctx();
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m);
            std::ptr::copy_nonoverlapping(targets.as_ptr(), self.tgt.buf.contents() as *mut u32, m);
        }
        self.prepare_head(targets);
        let cmd = Cmd::new(c);
        self.encode_fwd_bwd(&cmd);
        let norm_ranges: Vec<(&GBuf, usize, usize)> =
            ranges.iter().map(|&(off, n)| (&self.g, off, n)).collect();
        let gnorm = self.sumsq_ranges(cmd, &norm_ranges).sqrt() as f32;
        let loss = self.read_loss();
        assert!(loss.is_finite(), "non-finite loss before AdamW: {loss:?}");
        assert!(
            gnorm.is_finite(),
            "non-finite range gradient norm before AdamW: {gnorm:?}"
        );
        let gscale = if gnorm > clip { clip / gnorm } else { 1.0 };
        self.step += 1;
        let cmd = Cmd::new(c);
        for &(off, n) in ranges {
            cmd.adamw_at(
                &self.p, &self.g, &self.m, &self.v, off, n, lr, 0.9, 0.95, 1e-8, wd, self.step,
                gscale,
            );
        }
        cmd.commit();
        (loss, gnorm)
    }
}
