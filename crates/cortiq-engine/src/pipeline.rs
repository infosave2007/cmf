//! Full inference pipeline: tokenize → embed → layers → lm_head → sample → decode.
//!
//! Prefill/decode contract: every token is forwarded exactly once and
//! enters the KV cache exactly once. Logits for the next token are
//! computed from the hidden state of the LAST forwarded token — the
//! decode loop forwards the freshly sampled token, never re-embeds the
//! prompt tail (v1 duplicated the last prompt token in the cache).

use crate::attention::{self, QwenAttnCfg};
use crate::inference;
use crate::kv_cache::KvCache;
use crate::linear_core::{
    GdnCfg, GdnWeights, ShortConvCfg, ShortConvWeights, VmfPhaseCfg, VmfPhaseWeights, gdn_forward,
    gdn_pair, short_conv_forward, short_conv_forward_batch, short_conv_pair, vmf_phase_forward,
    vmf_phase_pair,
};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::sampler::{self, SamplerConfig, SamplerScratch, SplitMix64};
use crate::tokenizer::Tokenizer;
use cortiq_core::mask::TaskMask;
use cortiq_core::types::NormStyle;

pub static GLOBAL_USE_GPU: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Reusable per-pipeline forward scratch: the four norm outputs the
/// decode paths recompute every layer (single: n1/p1; pair: all four).
/// Plain buffers, resized once — steady-state decode reuses them.
struct ForwardScratch {
    n1: Vec<f32>,
    n2: Vec<f32>,
    p1: Vec<f32>,
    p2: Vec<f32>,
}

impl ForwardScratch {
    fn new(hidden: usize) -> Self {
        Self {
            n1: vec![0.0; hidden],
            n2: vec![0.0; hidden],
            p1: vec![0.0; hidden],
            p2: vec![0.0; hidden],
        }
    }
}

/// Complete inference pipeline state.
pub struct Pipeline {
    /// In-process layer split across local GPUs: (device, first layer,
    /// last layer) per segment, in execution order. `None` = one device.
    /// Arc so cloning the plan out of `&mut self` does not fight the
    /// borrow checker on the hot path.
    gpu_plan: Option<std::sync::Arc<Vec<(usize, usize, usize)>>>,
    /// Arc: the server shares one tokenizer handle across request
    /// handlers without borrowing a pipeline slot.
    pub tokenizer: std::sync::Arc<Tokenizer>,
    pub kv_cache: KvCache,
    pub sampler_config: SamplerConfig,
    pub weights: PipelineWeights,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Total virtual layers (num_layers × num_loops for looped models).
    pub num_layers: usize,
    /// Physical layers in weights.layers (≤ num_layers for looped models).
    pub physical_layers: usize,
    /// Looped Transformer: apply final norm after each loop iteration.
    pub loop_final_norm: bool,
    pub vocab_size: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    pub norm_style: NormStyle,
    /// RoPE dims actually rotated (≤ head_dim; Qwen3.5 uses head_dim/4).
    pub rotary_dim: usize,
    /// Optional Q-head count override for each attention layer (Laguna).
    pub attention_heads_per_layer: Option<Vec<usize>>,
    /// Linear-core geometry (present when the model has linear layers).
    pub vmf_cfg: Option<VmfPhaseCfg>,
    /// GatedDeltaNet geometry (faithful vendor operator).
    pub gdn_cfg: Option<GdnCfg>,
    /// MiniCPM-class logit scale (tied lm_head → cannot fold into weights).
    pub logit_multiplier: Option<f32>,
    /// Cooperative cancel: set from any thread (FFI `cortiq_cancel`,
    /// a dropped server connection); the generate loop checks it at
    /// every prefill chunk and decode step and finishes with
    /// `finish_reason: "cancelled"`. Auto-cleared when honoured.
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// A GPU graph failure is distinct from a user/request cancellation.
    /// Graph code sets this before raising the cooperative cancel flag so the
    /// generation API can return an error instead of reporting a successful
    /// `finish_reason: cancelled` result.
    graph_failed: std::sync::atomic::AtomicBool,
    /// Token ids currently materialized in the KV cache (the forwarded
    /// prompt + all generated tokens except the last, which is sampled
    /// but not yet forwarded). Lets the next generate call prefill only
    /// the suffix when a chat app resends the whole history.
    pub kv_history: Vec<u32>,
    /// KDA geometry (Kimi Linear / Kimi-K3) — shared by every Kda layer.
    pub kda_cfg: Option<crate::linear_core::KdaCfg>,
    /// Gemma-3n stack (AltUp/LAuReL/PLE/KV-sharing): its own forward —
    /// weights.layers stays empty, the KV caches are the shared ones.
    pub g3n: Option<Box<(crate::g3n::G3nGlobals, Vec<crate::g3n::G3nLayer>)>>,
    /// DeepSeek-V4 runs its own stack too: its hidden state is `hc_mult`
    /// copies of a vector, so no loop written for a single residual
    /// stream can carry it.
    pub dsv4: Option<
        Box<(
            crate::dsv4::Dsv4Globals,
            Vec<crate::dsv4::Dsv4Layer>,
            crate::dsv4::Dsv4Cfg,
            crate::dsv4::Dsv4State,
        )>,
    >,
    /// DeepSeek-V4.1 owns the shared CED/CSA2 attention state, raw Engram
    /// lookup and four-stream mHC handoff. It cannot use the V4 cache
    /// layout, so it has a dedicated executor and state tuple.
    pub dsv41: Option<
        Box<(
            crate::dsv41::Dsv41Globals,
            Vec<crate::dsv41::Dsv41Layer>,
            crate::dsv41::Dsv41Cfg,
            crate::dsv41::Dsv41State,
        )>,
    >,
    /// Optional V4.1 vision tower. Text-only files leave this unset.
    pub dsv41_vision: Option<crate::dsv41_vision::VisionModel>,
    /// Prepared image rows consumed by the next V4.1 prefill.
    dsv41_prefill: Option<(Vec<Option<Vec<f32>>>, Vec<bool>)>,
    /// Qwen3.8-Flash-Next owns four residual streams plus QSA/PLE state;
    /// the generic single-residual layer loop cannot represent it.
    pub qwen4_exp: Option<
        Box<(
            crate::qwen4_exp::Globals,
            Vec<crate::qwen4_exp::Layer>,
            crate::qwen4_exp::Cfg,
            crate::qwen4_exp::State,
        )>,
    >,
    /// DeepSeek-V4's own speculation stack: three draft modules, each a full
    /// layer, plus a confidence head on the last. Empty when the file has
    /// none, which is the only signal the decode path needs.
    pub dsv4_mtp: Vec<crate::dsv4::Dsv4Mtp>,
    /// The draft's per-sequence state (KV rings, captured trunk hidden).
    pub dspark: Option<crate::dsv4::DsparkState>,
    /// Drafts awaiting their verdict: (position, proposals, still matching,
    /// accepted so far).
    pub dspark_pending: Vec<(usize, Vec<u32>, bool, usize)>,
    /// Accepted prefix length of every graded draft.
    pub dspark_hist: Vec<usize>,
    /// The real tokens the drafts were graded against — a degenerate,
    /// repeating output would make any acceptance number meaningless, and
    /// the cheapest guard against believing one is to count them.
    pub dspark_real: Vec<u32>,
    /// The trunk's expert picks for the last few tokens, per layer. The
    /// union over a window of them is what a batched verify would have to
    /// read, and the ratio to the pick count is all it could save.
    pub dspark_trunk_picks: Vec<Vec<(usize, Vec<usize>)>>,
    /// (unique, total) expert picks per draft, trunk side and draft side.
    pub dspark_exp: Vec<(usize, usize, usize, usize)>,
    /// Wall time spent in the deliberately out-of-core draft. Kept separate
    /// from trunk decode so block batching can be judged without conflating
    /// it with GPU chain variance.
    pub dspark_draft_ns: u128,
    /// LFM2 short-convolution geometry (present when the model has
    /// `ShortConv` mixer layers).
    pub short_conv_cfg: Option<ShortConvCfg>,
    /// Multi-token-prediction head (None = absent).
    pub mtp: Option<MtpModule>,
    /// Speculative decode via MTP (greedy only; `CMF_MTP=0` disables).
    pub speculative: bool,
    /// Keep generating past end-of-sequence ids (the llama-bench contract
    /// for a timed run). A loop flag, deliberately NOT a sampler
    /// suppression: suppressed ids count as a penalty and switch the
    /// speculative round and the greedy burst off, so a benchmark that
    /// suppressed EOS never measured either.
    pub ignore_eos: bool,
    /// Draft-head shortlist guard: tokens left during which the draft
    /// uses the FULL head because a recently committed id lay past the
    /// `CMF_DRAFT_VOCAB` cut (Cyrillic and CJK ids sit above 131072 in
    /// Qwen's table, so a prefix shortlist would draft nothing usable
    /// there — measured on Russian prose: 2.9 → 1.6 accepted a round).
    pub draft_full_streak: u32,
    /// Adaptive draft depth for the speculative round (None until the
    /// first round): grows while nearly every draft is accepted, shrinks
    /// when fewer than half are. The verify's cost climbs with the rows on
    /// a discrete card (RTX PRO 4000: 52 ms at 2 rows, 74 at 5, 80 at 6),
    /// so prose wants k≈3 and code or the repetitive bench k≈5 — measured
    /// 33.6 vs 27.6 tok/s on an essay at k=3 vs 5, 45.6 vs 38 on code.
    /// `CMF_GRAPH_SPEC_K` pins it.
    pub spec_k_adapt: Option<usize>,
    /// EWMA of the accepted fraction that drives `spec_k_adapt`.
    pub spec_acc_ewma: f32,
    rng: SplitMix64,
    sampler_scratch: SamplerScratch,
    /// Speculative SAMPLING state (graph_spec_step, temperature > 0): the
    /// correction token a rejected draft produced — committed by the loop
    /// top in place of a fresh draw — and the per-round draft
    /// distributions / target scratch, reused so a round allocates
    /// nothing at the vocab size.
    spec_forced: Option<u32>,
    spec_q: Vec<Vec<f32>>,
    spec_p: Vec<f32>,
    spec_res: Vec<f32>,
    /// The same three for the sparse chain (top-k configs).
    spec_qs: Vec<sampler::Sparse>,
    spec_ps: sampler::Sparse,
    spec_ress: sampler::Sparse,
    /// Which arm the MTP draft block runs on this generation: Some(true)
    /// = the whole-token graph (device attention, one submit a step),
    /// Some(false) = the per-op path; None = not decided yet. Decided
    /// on the first draft and held, because the two arms keep the MTP
    /// KV in different places (device mirror vs the CPU cache) and a
    /// mid-run switch would read the wrong one.
    mtp_graph_mode: Option<bool>,
    /// The Metal verify graph of the round in flight, between its sync
    /// (logits read) and the commit that replays the accepted prefix.
    #[cfg(target_os = "macos")]
    metal_verify: Option<MetalVerifyPending>,
    /// Precomputed RoPE inverse frequencies [head_dim/2]. Arc: the
    /// forward path clones a handle to escape the &mut self borrow —
    /// cloning the table itself was a per-forward allocation.
    pub(crate) inv_freq: std::sync::Arc<Vec<f32>>,
    /// Reusable norm buffers for the decode hot path (roadmap §3 P0:
    /// steady-state forward should not heap-allocate). Disjoint field
    /// from `weights`/`kv_cache`, so split borrows keep working.
    ws: ForwardScratch,
    /// Persistent worker pool (None = serial; see CMF_THREADS).
    pool: Option<std::sync::Arc<Pool>>,
    // ── Dynamic per-token skill routing (spec §9, claim 14/16) ──
    /// Source model, retained so a skill switch can re-resolve the
    /// touched layers' FFN tensors (Mapped = mmap pointers, cheap).
    pub(crate) model: Option<std::sync::Arc<cortiq_core::CmfModel>>,
    /// Masks present → weights are dequantized f32 (rebuild path).
    pub(crate) dyn_force_f32: bool,
    /// Per-skill FFN layers actually replaced (derived from tensors, not
    /// the meta `layers` field — ru2 replaces down_proj in 0..23 while
    /// its meta says [20..23]). None = skill touches non-FFN tensors →
    /// ineligible for cheap dynamic switching (honest refusal).
    pub(crate) dyn_skill_layers: Vec<Option<Vec<usize>>>,
    /// Currently overlaid skill (index into model.header.skills); None =
    /// backbone. Set at load time to the statically-overlaid skill so
    /// `set_active_skill(None)` correctly reverts it (else a static
    /// skill would silently persist — the union-diff assumes dyn_active
    /// always mirrors the live overlay). Switched by `set_active_skill`.
    pub(crate) dyn_active: Option<usize>,
    /// Pipeline was loaded with a soft blend (materialized working
    /// tensors, not a single skill index) → dynamic routing refuses:
    /// there is no single index to revert the blend from.
    pub(crate) dyn_blend_loaded: bool,
    /// Layer whose post-residual hidden feeds the router φ (shared by
    /// swarm skills). None = φ capture off.
    pub(crate) dyn_phi_layer: Option<usize>,
    /// EMA of φ at `dyn_phi_layer` over the decode window (on-policy).
    dyn_phi_ema: Vec<f32>,
    dyn_phi_seen: usize,
    /// Hysteresis router driving per-token skill switches during decode
    /// (None = static/no dynamic routing). Taken out during generation.
    pub dyn_router: Option<crate::swarm::DynRouter>,
    /// O(1) Nyström attention setting (CLI/env/header-hint resolved by
    /// the caller; None = plain cache attention everywhere).
    o1_cfg: Option<crate::nystrom::O1Cfg>,
    /// Bumped once per collecting→sealed transition — the GPU state mirror
    /// re-uploads when it sees a new epoch (each fresh sealed state).
    o1_epoch: u64,
    /// Per-layer o1 flags derived from `o1_cfg` (Full layers only).
    o1_flags: Vec<bool>,
    /// Emit a structured per-token trace (B4 telemetry channel). Off by
    /// default — the runtime is silent unless observation is requested.
    trace: bool,
    /// Confidence-calibration temperature (B1): reported probability is
    /// softmax(logits / calib_temp). 1.0 = raw. Set from header.calibration.
    calib_temp: f32,
    /// Process-unique id keying this pipeline's device KV mirrors.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    graph_kv_id: u64,
    /// Decode asks the token graph to also run final-norm + lm_head on
    /// the device (drops the separate per-op lm_head round trip).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    graph_want_logits: bool,
    /// NLL quality gates require the graph's fused head rather than silently
    /// accepting a CPU head fallback. Generation keeps the historical
    /// best-effort `graph_want_logits` behavior.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    graph_head_required: bool,
    /// Logits the graph produced for the token just forwarded (taken by
    /// the decode loop; None = compute on the CPU path).
    graph_logits: Option<Vec<f32>>,
    /// Token embeddings are multiplied by this at input (Gemma: √hidden).
    pub embed_multiplier: f32,
    /// Attention score scale (1/√head_dim unless the arch overrides —
    /// Gemma's query_pre_attn_scalar).
    pub attn_scale: f32,
    /// Sliding-window attention: (window, every-Nth-layer-is-global
    /// pattern) — Gemma-3.
    pub swa: Option<(usize, usize)>,
    /// Explicit local/global schedule for architectures that cannot be
    /// represented by Gemma's every-Nth-global convention.
    pub sliding_layers: Option<Vec<bool>>,
    /// RoPE table of the sliding (local) layers, when they use their
    /// own base frequency (Gemma-3: 10k local vs 1M global).
    pub inv_freq_local: Option<std::sync::Arc<Vec<f32>>>,
    pub rotary_dim_local: Option<usize>,
    pub rope_scale: f32,
    pub rope_scale_local: f32,
    /// Gemma-4: global layers run their own geometry — (head_dim,
    /// num_kv_heads); sliding layers keep the base fields.
    pub global_attn: Option<(usize, usize)>,
    /// Gemma-4: the global layers' proportional RoPE table (len
    /// global_head_dim/2, zero-padded tail = identity rotation).
    pub inv_freq_global: Option<std::sync::Arc<Vec<f32>>>,
    /// Scale-less RMS normalization of V heads before caching (Gemma-4).
    pub attn_v_norm: bool,
    /// HunYuan dense: per-head q/k norm runs after RoPE (see the arch flag).
    pub qk_norm_after_rope: bool,
    /// Final-logit soft-capping C: logits = C·tanh(logits/C) (Gemma-4).
    pub final_softcap: Option<f32>,
    /// Cortiq Embryo hierarchical head: cluster matrix [C, hidden]. The
    /// flat logits h·Eᵀ are turned into the two-level log-probabilities
    /// log softmax_c(h·Cᵀ)[c(v)] + log softmax_{s∈c(v)}(h·E_c(v)ᵀ)[v].
    pub head_clusters: Option<std::sync::Arc<Vec<f32>>>,
    /// Gemma-2 attention-logit soft-capping (0.0 = off).
    pub attn_softcap: f32,
    /// Compute per-token confidence (a full-vocab softmax each
    /// token). On by default; `bench --core` turns it off to match
    /// llama-bench's core timing.
    confidence_on: bool,
    /// Test-only one-shot forward failure, scoped to this pipeline so
    /// parallel scoring tests cannot consume one another's injection.
    #[cfg(test)]
    nll_test_fail_at: Option<usize>,
    /// Test-only route override; avoids mutating the process-wide
    /// `CMF_PREFILL` environment variable while forcing the serial path.
    #[cfg(test)]
    nll_test_force_serial: bool,
}

#[cfg(target_os = "macos")]
impl Drop for Pipeline {
    fn drop(&mut self) {
        // the async replay writes into `kv_cache` Vecs about to be freed
        let _ = crate::gpu_metal::wait_replay();
        crate::gpu::kv_mirror_drop(self.graph_kv_id);
    }
}

/// Model weights. Matrices are `QTensor` (owned f32 for small models
/// and tests — bit-identical to the historical paths — or quantized
/// bytes zero-copy from the CMF mmap for big models). 1-D norms are
/// always small and stay f32.
pub struct PipelineWeights {
    /// Embedding table: [vocab_size, hidden_size]
    pub embed_tokens: QTensor,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// LM head: [vocab_size, hidden_size]
    pub lm_head: QTensor,
    /// Final norm: [hidden_size]
    pub final_norm: Vec<f32>,
}

/// One transformer layer: shared norms + MLP, attention by kind.
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    /// The pre-FFN norm (`post_attention_layernorm` classically;
    /// `pre_feedforward_layernorm` on Gemma-2/3 sandwich layers).
    pub post_norm: Vec<f32>,
    /// Gemma-2/3 sandwich: norm applied to the ATTENTION OUTPUT before
    /// its residual add (`post_attention_layernorm` there).
    pub attn_out_norm: Option<Vec<f32>>,
    /// Gemma-4: the whole layer output is multiplied by this scalar.
    pub layer_scale: Option<f32>,
    /// Gemma-2/3 sandwich: norm applied to the FFN OUTPUT before its
    /// residual add (`post_feedforward_layernorm`).
    pub ffn_out_norm: Option<Vec<f32>>,
    pub ffn: FfnKind,
    pub attn: AttnKind,
}

/// FFN gate activation: SiLU (SwiGLU family) or tanh-GELU (Gemma's
/// GeGLU). A property of the model, carried on every FFN triple.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Act {
    #[default]
    Silu,
    GeluTanh,
    /// Kimi-K3 SituAndMul: BOTH halves transform —
    /// a = β·tanh(g/β)·σ(g), up' = linβ·tanh(u/linβ) (linβ>0), out = a·up'.
    Situ {
        beta: f32,
        linear_beta: f32,
    },
}

impl Act {
    pub fn from_arch(name: &str) -> Self {
        if name == "gelu_tanh" {
            Self::GeluTanh
        } else {
            Self::Silu
        }
    }

    /// Arch-driven constructor (activation name + situ betas).
    pub fn from_arch_full(arch: &cortiq_core::ModelArch) -> Self {
        match arch.hidden_act.as_str() {
            "situ" => Self::Situ {
                beta: arch.activation_situ_beta.unwrap_or(1.0) as f32,
                linear_beta: arch.activation_situ_linear_beta.unwrap_or(0.0) as f32,
            },
            other => Self::from_arch(other),
        }
    }

    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Silu => inference::silu(x),
            Self::GeluTanh => inference::gelu_tanh(x),
            Self::Situ { beta, .. } => beta * (x / beta).tanh() * (1.0 / (1.0 + (-x).exp())),
        }
    }

    /// Gated combine — the FFN contract. Situ transforms the UP half
    /// too, so callers must use this instead of apply(g)·u.
    #[inline]
    pub fn combine(self, g: f32, u: f32) -> f32 {
        match self {
            Self::Situ { linear_beta, .. } if linear_beta > 0.0 => {
                self.apply(g) * (linear_beta * (u / linear_beta).tanh())
            }
            _ => self.apply(g) * u,
        }
    }
}

/// Dense gated triple — the FFN of a dense layer or of one expert.
pub struct DenseFfn {
    pub gate_proj: QTensor,
    pub up_proj: QTensor,
    pub down_proj: QTensor,
    /// Gate activation (SiLU default; Gemma: tanh-GELU).
    pub act: Act,
    /// `down_proj` stored transposed (`[inter, hidden]`), when the file
    /// carries it. Only the per-token sparse path reads it: a neuron's
    /// down weights are a contiguous ROW there, so the token's chosen
    /// neurons are the only bytes touched. `None` = the ordinary layout,
    /// and the sparse path stays off.
    pub down_t: Option<QTensor>,
    /// Task tubes (spec: defragged task-conditional width). The three
    /// matrices above are the CORE — the neurons every task computes;
    /// each tube is an independently quantized slice of the SAME layer
    /// holding the neurons only some tasks need. A tube is a normal
    /// tensor triple, so every kernel runs it unchanged, and the bytes
    /// of an inactive tube are never read. Empty = ordinary dense FFN.
    pub segs: Vec<FfnSeg>,
}

/// One task tube: a contiguous slice of a layer's FFN neurons, stored
/// as its own `[w, hidden]` / `[hidden, w]` triple. `start` is the
/// neuron's index in the layer's FULL space (core first, then tubes in
/// order) — the bit a task mask sets to switch this tube on.
pub struct FfnSeg {
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    pub start: usize,
    pub width: usize,
}

/// FFN operator of a layer, decided by tensor presence at load time
/// (router `mlp.gate.weight` in the directory = MoE layer).
pub enum FfnKind {
    Dense(DenseFfn),
    /// Mixture-of-Experts (Qwen2-MoE / Qwen3-MoE): softmax over ALL
    /// expert logits → top-k, optional renorm; experts stay quantized
    /// in mmap — only the selected ones are touched per token.
    Moe(MoeFfn),
    /// Gemma-4 MoE: a dense MLP branch AND a routed-expert branch in
    /// the SAME layer, each with its own norm sandwich. The dense
    /// branch reads the pre-FFN-normed input; the expert branch (and
    /// the router) read the RAW residual through `pre_norm_2`:
    ///   d = post_norm_1(dense(x̂));  m = post_norm_2(Σwₑ·FFNₑ(pre_norm_2(h)))
    ///   ffn_out = d + m   (the caller's ffn_out_norm + residual follow)
    DenseMoe(Box<DenseMoeFfn>),
}

/// Gemma-4 dual-branch FFN (see `FfnKind::DenseMoe`).
pub struct DenseMoeFfn {
    pub dense: DenseFfn,
    pub moe: MoeFfn,
    /// post_feedforward_layernorm_1 — dense-branch output norm.
    pub post_norm_1: Vec<f32>,
    /// pre_feedforward_layernorm_2 — expert-branch input norm (applied
    /// to the RAW residual, not the pre-FFN-normed activation).
    pub pre_norm_2: Vec<f32>,
    /// post_feedforward_layernorm_2 — expert-branch output norm.
    pub post_norm_2: Vec<f32>,
}

pub struct MoeFfn {
    /// Router `mlp.gate.weight` [num_experts, hidden].
    pub router: QTensor,
    pub experts: Vec<DenseFfn>,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    /// Router scores per-expert with a sigmoid (LFM2-MoE / DeepSeek-V3
    /// `noaux_tc`) instead of a softmax over all experts (Qwen).
    pub router_sigmoid: bool,
    /// Per-expert selection bias `mlp.expert_bias` [num_experts]
    /// (LFM2-MoE): added to the sigmoid scores for the top-k CHOICE only;
    /// the gathered weights use the unbiased scores. None = no bias.
    pub expert_bias: Option<Vec<f32>>,
    /// Top-k weights are multiplied by this after the optional renorm
    /// (LFM2-MoE `routed_scaling_factor`; 1.0 = off).
    pub routed_scaling: f32,
    /// Adaptive routing (CMF_MOE_TAU, opt-in): keep the smallest
    /// prefix of the top-k whose renormalized mass reaches τ —
    /// confident tokens touch 1–2 experts, flat ones keep all k.
    /// MoE decode is memory-bound, so skipped experts are skipped
    /// weight traffic. None = classic fixed top-k (bit-identical).
    pub route_tau: Option<f32>,
    /// Always-on shared expert. Qwen2-MoE carries an additional sigmoid
    /// gate; Laguna adds the shared expert unconditionally (`None`).
    pub shared: Option<(DenseFfn, Option<QTensor>)>,
    /// Expert-selection counters (truncated Fisher B-field of claim 12:
    /// routing frequency during calibration). Filled by every forward,
    /// read by the CLI via CMF_MOE_STATS. RefCell: decode is single-threaded.
    pub stats: std::cell::RefCell<Vec<u64>>,
    /// Per-CHANNEL sum of squares of this FFN's input, accumulated over a
    /// calibration run (`CMF_RMS_TRACE`). These are the RMS activation
    /// traces AWNP needs: raw weight magnitude says every channel matters
    /// equally, and the question AWNP asks is whether the ACTIVATIONS
    /// disagree. Off unless the env var is set — an f64 add per channel
    /// per token is cheap, but not free.
    pub act_sq: std::cell::RefCell<Vec<f64>>,
    /// Raw FFN-input rows captured for the layers named by `CMF_ACT_DUMP`
    /// (`"9,19"`). AWNP is nullspace PROJECTION: after dropping channels the
    /// survivors are refitted to absorb what was removed, and how much they
    /// can absorb depends on the activation COVARIANCE, not on per-channel
    /// RMS. Per-channel numbers can only bound the cost from above.
    pub act_rows: std::cell::RefCell<Vec<f32>>,
    /// Task mask over routed experts (DTG-MA over MoE, claim-12 B-field
    /// applied): `false` experts are excluded from selection, the
    /// softmax renormalizes over the allowed set. Built by the loader
    /// from CMF_MOE_MASK=<stats.json> + CMF_MOE_MASK_COVER. None = all.
    pub mask: Option<Vec<bool>>,
    /// Gemma-4: per-expert weight scale applied AFTER the top-k renorm
    /// (`router.per_expert_scale`). None = 1.0 everywhere.
    pub per_expert_scale: Option<Vec<f32>>,
    /// Gemma-4: the router reads a SCALE-LESS rms-norm of its input
    /// (the constant gain router.scale·√hidden is folded into the
    /// router weights at convert time).
    pub router_input_norm: bool,
    /// Cortiq Embryo: resonance routing (P1) — the "logits" are
    /// bias_e − ‖(x−μ_e) − U_eᵀU_e(x−μ_e)‖², argmax = the expert whose
    /// descriptor reconstructs the input best. `router` is a placeholder.
    pub resonance: Option<Resonance>,
}

/// Per-expert resonance descriptors of one MoE layer (`mlp.desc.*`).
pub struct Resonance {
    /// [E, hidden]
    pub mu: Vec<f32>,
    /// [E, k, hidden] orthonormal directions (k may be 0)
    pub u: Vec<f32>,
    pub k: usize,
    /// [E] selection bias (loss-free balancing, trained online)
    pub bias: Vec<f32>,
}

impl Resonance {
    /// Routing scores for one input row (higher = better).
    pub fn scores(&self, x: &[f32], out: &mut [f32]) {
        let h = x.len();
        let ne = out.len();
        for e in 0..ne {
            let mu = &self.mu[e * h..(e + 1) * h];
            let mut d2 = 0.0f32;
            for j in 0..h {
                let d = x[j] - mu[j];
                d2 += d * d;
            }
            let mut proj = 0.0f32;
            for i in 0..self.k {
                let u = &self.u[(e * self.k + i) * h..(e * self.k + i + 1) * h];
                let mut p = 0.0f32;
                for j in 0..h {
                    p += (x[j] - mu[j]) * u[j];
                }
                proj += p * p;
            }
            out[e] = self.bias.get(e).copied().unwrap_or(0.0) - (d2 - proj);
        }
    }
}

/// Attention operator of a layer. Extension point: new operators are
/// new variants here + a forward in their own module.
pub enum AttnKind {
    /// GQA softmax attention (+ optional Qwen3.5 qk-norm / output gate).
    Full {
        wq: QTensor,
        wk: QTensor,
        wv: QTensor,
        wo: QTensor,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        output_gate: bool,
        /// Laguna: a separate softplus projection applied to the attention
        /// output before O. The bool means one scalar per head (broadcast
        /// across head_dim); false means one scalar per element.
        softplus_gate: Option<(QTensor, bool)>,
        /// Qwen2-family projection biases (q, k, v).
        bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    },
    /// Canonical linear core (VMF phase attention).
    Linear(VmfPhaseWeights),
    /// Faithful vendor linear operator (Qwen3.5 GatedDeltaNet).
    LinearGdn(GdnWeights),
    /// LFM2 gated short-convolution mixer (no KV cache; conv ring state
    /// lives in the layer's `linear_state`).
    ShortConv(ShortConvWeights),
    /// DeepSeek-V2 Multi-head Latent Attention. v1 executes it as
    /// expand-to-MHA: the latent is projected per token, K/V expand to
    /// every head and live in the ordinary cache (K head layout
    /// [rope | nope] so the standard partial rotary covers the shared
    /// rope key; V rows are zero-padded to the K head_dim and the pad
    /// is sliced off before O). Latent-resident cache is a later
    /// optimization, not a semantic change.
    Mla(Box<MlaWeights>),
    /// Kimi Delta Attention (Kimi Linear / Kimi-K3): per-channel decayed
    /// delta rule, separate q/k/v short convs, sigmoid-gated output norm.
    /// State lives in the layer's `linear_state` (no KV cache).
    Kda(Box<crate::linear_core::KdaWeights>),
}

/// DeepSeek-V2 MLA projections (see `AttnKind::Mla`).
pub struct MlaWeights {
    /// `[nh·(rope+nope), hidden]` (or `[…, q_lora]` when compressed) —
    /// the converter permutes each head rope-first so rotary_dim =
    /// qk_rope works unchanged.
    pub q_proj: QTensor,
    /// Compressed q (K3/V3 class): x → q_a `[q_lora, hidden]` →
    /// rms(q_a_norm) → q_proj (= q_b). None = direct q (V2-Lite).
    pub q_a: Option<QTensor>,
    pub q_a_norm: Option<Vec<f32>>,
    /// `kv_a_proj_with_mqa` `[lora + rope, hidden]` (latent first).
    pub kv_a: QTensor,
    /// RMS-norm weights over the latent (`kv_a_layernorm`, [lora]).
    pub kv_a_norm: Vec<f32>,
    /// `[nh·(nope+v), lora]` — per head [k_nope | v].
    pub kv_b: QTensor,
    /// `[hidden, nh·v]`.
    pub o_proj: QTensor,
    pub nh: usize,
    pub qk_rope: usize,
    pub qk_nope: usize,
    pub v_dim: usize,
    pub lora: usize,
    /// Softmax scale (1/√(rope+nope), YaRN-mscale-corrected at load).
    pub scale: f32,
    /// Kimi Linear NoPE: skip the rotary entirely (layout unchanged).
    pub nope: bool,
}

/// Multi-token-prediction head (DeepSeek/Qwen style, spec §2.1):
/// `x = eh_proj·[enorm(embed(next)); hnorm(hidden)]` → one transformer
/// block over its own KV → shared lm_head. Drafts the token after next;
/// the main model verifies, so output is exact — MTP only buys speed.
pub struct MtpModule {
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    /// [hidden, 2·hidden]
    pub eh_proj: QTensor,
    pub layer: LayerWeights,
    pub final_norm: Vec<f32>,
    pub kv: crate::kv_cache::LayerKvCache,
}

/// A Metal verify graph after its sync: what the commit needs — the
/// graph (per-layer replay scratch), the GDN layers in encode order (their
/// CPU states receive the replay), and the attention layers with the CPU
/// row count they were encoded against (the accepted rows are pulled from
/// the mirror from there).
/// One item of the Metal rows-graph plan.
#[cfg(target_os = "macos")]
enum MetalRowsItem<'a> {
    Gdn {
        run: Vec<crate::gpu_metal::GdnGpuLayer<'a>>,
        first: usize,
    },
    Attn {
        l: crate::gpu_metal::AttnGpuLayer<'a>,
        li: usize,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        output_gate: bool,
    },
}

#[cfg(target_os = "macos")]
struct MetalVerifyPending {
    graph: crate::gpu_metal::VerifyGraph,
    gdn_layers: Vec<usize>,
    attn_layers: Vec<(usize, usize)>,
}

/// A round's batched MTP warm-up, submitted but not yet waited
/// (`mtp_warm_batch_submit` → `mtp_warm_batch_finish`): the trunk commit's
/// GDN replay is queued between the two.
#[cfg(target_os = "macos")]
struct MetalWarmPending {
    graph: crate::gpu_metal::VerifyGraph,
    cpu_stored: usize,
    b: usize,
}

#[cfg(target_os = "macos")]
enum MetalRowsRun {
    /// Capability/preflight refusal before a command buffer was committed.
    Declined,
    /// A graph was admitted and then failed; callers must clear the sequence
    /// rather than replaying it through CPU/serial state.
    Failed,
    Completed(MetalVerifyPending),
}

#[cfg(target_os = "macos")]
enum MetalPrefillOutcome {
    Declined,
    Failed,
    Completed(Vec<f32>),
}

#[cfg(target_os = "macos")]
enum MetalBatchNllOutcome {
    Declined,
    Failed(String),
    Completed(f64, usize),
}

/// The speculation trial's phases (see the decode loop): four timed
/// speculative rounds, eight timed plain tokens, then the faster arm
/// until a re-check.
#[derive(Clone, Copy)]
enum SpecTrial {
    Spec {
        t0: std::time::Instant,
        gen0: usize,
        rounds: usize,
    },
    Plain {
        t0: std::time::Instant,
        gen0: usize,
    },
    Decided {
        spec: bool,
        recheck_at: usize,
    },
}

/// `CMF_GRAPH_SPEC_TIME`: 0 = off, 1 = one line per speculative round
/// plus the host stamps of any OUTLIER round (wall > 1.4× the running
/// median), 2 = the host stamps of every round.
pub(crate) fn spec_time_level() -> u8 {
    static L: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *L.get_or_init(|| match std::env::var("CMF_GRAPH_SPEC_TIME") {
        Ok(v) => v.trim().parse::<u8>().map(|n| n.max(1)).unwrap_or(1),
        Err(_) => 0,
    })
}

/// The round's host stamps: `spec_stamp(name)` records the time since
/// the previous stamp (the section that just ended) — from anywhere on
/// the round's call chain (the Metal verify, the draft step, the commit),
/// no plumbing. Off (a single atomic load) unless `CMF_GRAPH_SPEC_TIME`
/// is set; one decode thread at a time is assumed (diagnostics).
struct SpecStampLog {
    t_last: std::time::Instant,
    items: Vec<(&'static str, f32)>,
}

static SPEC_STAMPS: std::sync::Mutex<Option<SpecStampLog>> = std::sync::Mutex::new(None);

pub(crate) fn spec_stamp(name: &'static str) {
    if spec_time_level() == 0 {
        return;
    }
    if let Ok(mut g) = SPEC_STAMPS.lock() {
        if let Some(log) = g.as_mut() {
            let now = std::time::Instant::now();
            log.items
                .push((name, (now - log.t_last).as_secs_f32() * 1e3));
            log.t_last = now;
        }
    }
}

fn spec_stamps_begin() {
    if spec_time_level() == 0 {
        return;
    }
    if let Ok(mut g) = SPEC_STAMPS.lock() {
        *g = Some(SpecStampLog {
            t_last: std::time::Instant::now(),
            items: Vec::with_capacity(64),
        });
    }
}

fn spec_stamps_take() -> Vec<(&'static str, f32)> {
    SPEC_STAMPS
        .lock()
        .ok()
        .and_then(|mut g| g.take())
        .map(|l| l.items)
        .unwrap_or_default()
}

/// One line: every stamp name in first-seen order with its total over the
/// round and, when it fired more than once (the draft steps), the count.
fn spec_stamps_format(items: &[(&'static str, f32)]) -> String {
    let mut agg: Vec<(&'static str, f32, u32)> = Vec::with_capacity(items.len());
    for &(n, ms) in items {
        match agg.iter_mut().find(|e| e.0 == n) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => agg.push((n, ms, 1)),
        }
    }
    let mut s = String::with_capacity(agg.len() * 16);
    for (n, ms, k) in agg {
        if k > 1 {
            s.push_str(&format!("{n} {ms:.1}/{k} "));
        } else {
            s.push_str(&format!("{n} {ms:.1} "));
        }
    }
    s
}

/// The speculation monitor: exponential averages of a round's wall time
/// and of the tokens it produced, and the plain token's wall time — the
/// three numbers the keep/stop rule needs. A round pays when
/// `tokens_per_round · plain_ms > round_ms · 1.03`. The one-shot trial
/// (four rounds against eight tokens) mis-called prose: the first rounds
/// after a prompt are formulaic and accept well, the body does not (an
/// essay measured 39 against a plain 44.8 with the trial saying
/// "speculate"), so the rule now runs on EVERY round and stops after four
/// consecutive losing rounds; a stopped speculation is retried 128 tokens
/// later.
///
/// Native Metal (`metal: true`) does not pay the eight plain tokens up
/// front: on the 27B a plain token is ~150 ms, so the trial alone cost
/// ~1.2 s of every answer. There the plain phase is (a) skipped while the
/// rounds land at least `SPEC_PROXY_TOKENS` tokens each — a k=7 round on
/// Metal costs ~1.9 plain tokens (286 against 148 ms measured on the M4),
/// so 3.5 tokens/round cannot lose on any Metal round/plain ratio seen —
/// and (b) otherwise bounded to the fewest tokens that time it: two, or
/// as many as fit in `SPEC_PLAIN_MIN_MS` (a 150-ms token measures itself;
/// a 10-ms one needs the eight). The keep/stop rule itself is unchanged:
/// the moment a plain rate exists, it decides.
#[derive(Default, Clone, Copy)]
struct SpecMon {
    round_ms: f64,
    tokens: f64,
    plain_ms: f64,
    n: u32,
    fails: u32,
    metal: bool,
}

/// Tokens per round at or above which a Metal round pays without a plain
/// measurement (see `SpecMon`).
const SPEC_PROXY_TOKENS: f64 = 3.5;
/// The Metal plain phase: at least two tokens, and more until this much
/// wall time has been timed (up to the eight the other backends time).
const SPEC_PLAIN_MIN_MS: f64 = 200.0;

impl SpecMon {
    fn round(&mut self, dt_ms: f64, produced: usize) {
        self.n += 1;
        if self.n == 1 {
            return; // round 1 pays the batch scratch and the draft mirror
        }
        let a = if self.n == 2 { 1.0 } else { 0.3 };
        self.round_ms += a * (dt_ms - self.round_ms);
        self.tokens += a * (produced as f64 - self.tokens);
    }
    fn pays(&self) -> bool {
        if self.plain_ms > 0.0 {
            self.tokens * self.plain_ms > self.round_ms * 1.03
        } else {
            self.metal && self.tokens >= SPEC_PROXY_TOKENS
        }
    }
    /// Has the plain phase timed enough tokens to decide?
    fn plain_done(&self, t0: std::time::Instant, gen0: usize, generated: usize) -> bool {
        let n = generated.saturating_sub(gen0);
        if n >= 8 {
            return true;
        }
        self.metal && n >= 2 && t0.elapsed().as_secs_f64() * 1e3 >= SPEC_PLAIN_MIN_MS
    }
}

/// Result of a generation call.
pub struct GenerateResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub finish_reason: String,
    /// Speculative-decode stats (0/0 when MTP is absent or inactive).
    pub mtp_drafted: usize,
    pub mtp_accepted: usize,
    /// Per-generated-token confidence = softmax probability of the token
    /// that was actually emitted (softmax probability on the chosen state). High =
    /// the model was sure; low = it was guessing. Same length as the
    /// generated slice of `token_ids`.
    pub token_confidence: Vec<f32>,
    /// Structured per-token telemetry (B4 channel). Empty unless
    /// `set_trace(true)`; otherwise same length as the generated slice.
    pub traces: Vec<TokenTrace>,
}

/// One row of the structured telemetry trace (B4): the model's internal
/// routing state at the moment a token was emitted. Every field is a
/// quantity the runtime already computes — nothing is inferred or
/// estimated (anti-principle: only measured bytes).
#[derive(Clone, Debug)]
pub struct TokenTrace {
    /// 0-based index within the generated slice.
    pub t: usize,
    /// The emitted token id.
    pub token_id: u32,
    /// Softmax probability on the emitted token — how sure the model was.
    pub confidence: f32,
    /// Skill in force while this token was generated (None = backbone).
    pub active_skill: Option<String>,
    /// Recon error E = ‖r−BBᵀr‖²/‖φ‖² at the last routing eval — coherence
    /// with the active skill's subspace (low = coherent). None = no router
    /// or not yet evaluated.
    pub recon: Option<f32>,
    /// The router changed the active skill right after this token (a
    /// domain boundary crossed under the hysteresis barrier).
    pub switched: bool,
}

/// Calibrated softmax probability of `id` under `logits` (the confidence on
/// the emitted token) — the confidence signal, cheap from logits already
/// computed for sampling. `temp` is the calibration temperature (B1):
/// softmax(logits / temp); 1.0 = raw.
#[cfg_attr(not(test), allow(dead_code))]
fn top1_prob_t(logits: &[f32], id: u32, temp: f32) -> f32 {
    let t = if temp > 1e-3 { temp } else { 1.0 };
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum: f32 = logits.iter().map(|&v| ((v - max) / t).exp()).sum();
    if sum > 0.0 {
        (((logits[id as usize] - max) / t).exp()) / sum
    } else {
        0.0
    }
}

/// prefill-GEMM enabled? (CMF_PREFILL=seq — emergency fallback to the
/// sequential path.)
fn prefill_batched() -> bool {
    std::env::var("CMF_PREFILL")
        .map(|v| v != "seq")
        .unwrap_or(true)
}

/// Decide the graph NLL route without conflating graph quality with the
/// optional native-Metal fused head. A hidden-state graph remains a valid
/// quality route on Vulkan/Wgpu; only native Metal requires graph logits.
#[inline]
fn nll_graph_policy(
    unmasked: bool,
    prefer_graph: bool,
    native_metal: bool,
) -> (bool, bool) {
    let graph_quality = unmasked && prefer_graph;
    let fused_head_quality = graph_quality && native_metal;
    (graph_quality, fused_head_quality)
}

/// Input to the layer-major batched span walk: token ids (embeds itself,
/// full-stack and coordinator prefill) or ready boundary hiddens (the
/// network worker's side of a split).
#[derive(Clone, Copy)]
enum PrefillIn<'a> {
    Ids(&'a [u32]),
    Hidden(&'a [f32]),
}

/// The batched prefill walks `weights.layers`. Architectures that load
/// their own stack (gemma-3n's AltUp replicas, DeepSeek-V4's hyper-
/// connections) leave that empty and must go position by position — asking
/// otherwise indexes an empty vector, which is a panic rather than a
/// fallback. Every call site goes through here so the next such
/// architecture is one line, not four.
impl Pipeline {
    fn can_prefill_batched(&self) -> bool {
        #[cfg(test)]
        let force_serial = self.nll_test_force_serial;
        #[cfg(not(test))]
        let force_serial = false;
        prefill_batched() && !force_serial && !self.weights.layers.is_empty()
    }

    /// The backend's automatic capacity split for a mapped transformer.
    /// Kept as a method so prefill and decode use the exact same boundary.
    fn automatic_gpu_prefix(&self) -> Option<usize> {
        let (model, _, _, _) = self.weights.embed_tokens.graph_weight()?;
        crate::gpu::automatic_layer_prefix(&model, self.num_layers, self.physical_layers)
    }

    /// Positions per batched pass of the layer-stack prefill for THIS
    /// model on THIS backend (see [`prefill_chunk_rule`]). Pub: the network
    /// split must chunk exactly like the local path to reproduce it.
    pub fn prefill_chunk(&self) -> usize {
        let env = env_prefill_chunk();
        if env.is_some() || ChunkHost::here() != ChunkHost::Other {
            return prefill_chunk_rule(env, ChunkHost::here(), false);
        }
        prefill_chunk_rule(None, ChunkHost::Other, self.chunk_stack_facts().dense_on_discrete())
    }

    fn chunk_stack_facts(&self) -> ChunkStackFacts {
        let plain_dense = !self.weights.layers.is_empty()
            && self.g3n.is_none()
            && self.dsv4.is_none()
            && self.dsv41.is_none()
            && self.qwen4_exp.is_none()
            && self.weights.layers.iter().all(|lw| {
                matches!(lw.attn, AttnKind::Full { .. }) && matches!(lw.ffn, FfnKind::Dense(_))
            });
        let gpu_on = crate::gpu::enabled();
        ChunkStackFacts {
            plain_dense,
            discrete: gpu_on && crate::gpu::discrete(),
            gpu_on,
            // Only asked when the rest already qualifies: it opens the
            // backend's capacity plan.
            capacity_split: std::env::var_os("CMF_GPU_LAYERS").is_some()
                || (plain_dense && gpu_on && self.automatic_gpu_prefix().is_some()),
            multi_gpu: self.gpu_plan.is_some(),
            o1: self.o1_active(),
        }
    }
}

/// Prefill chunk (positions per batched pass), model-agnostic form. On
/// macOS the AMX GEMM path wants tall panels — M=48 starves the matrix
/// units (ggml uses ubatch 512); elsewhere the historical 48 stays.
/// CMF_PREFILL_CHUNK overrides. The architectures with their own stacks
/// (DeepSeek-V4/V4.1) chunk with this; the layer-stack prefill asks
/// [`Pipeline::prefill_chunk`], which also knows the model and the card.
/// A different chunk is a different (equally valid) generation: panel
/// width reorders float accumulation.
pub fn prefill_chunk() -> usize {
    prefill_chunk_rule(env_prefill_chunk(), ChunkHost::here(), false)
}

fn env_prefill_chunk() -> Option<usize> {
    std::env::var("CMF_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// The host classes the chunk width distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkHost {
    Macos,
    /// Linux/Android aarch64 (phones, SBCs).
    Aarch64,
    /// Everything else: x86-64 Linux/Windows, CPU or Vulkan/DX12.
    Other,
}

impl ChunkHost {
    fn here() -> Self {
        if cfg!(target_os = "macos") {
            ChunkHost::Macos
        } else if cfg!(target_arch = "aarch64") {
            ChunkHost::Aarch64
        } else {
            ChunkHost::Other
        }
    }
}

/// Chunk for a plain dense stack whose every layer lives on a discrete
/// card. On x86 the layer-stack prefill is host-driven: each GEMM and the
/// chunk attention (which re-uploads the whole KV prefix per layer) is a
/// separate submit + readback, so 48 positions a pass left the card idle
/// between them. Measured in-process on an RTX 3090 (Vulkan), 2048-token
/// prompt — see CHANGELOG 0.7.6 for the table.
const DISCRETE_DENSE_PREFILL_CHUNK: usize = 512;

/// The chunk-width rule. `dense_on_discrete` is true only for a plain
/// dense transformer (full attention, dense FFN, no special stack) that
/// is entirely resident on one discrete card — the one case measured
/// here. GDN hybrids, MoE, DeepSeek stacks, capacity-split and CPU-only
/// runs keep the width they were tuned with.
fn prefill_chunk_rule(env: Option<usize>, host: ChunkHost, dense_on_discrete: bool) -> usize {
    if let Some(n) = env {
        return n.max(1);
    }
    match host {
        ChunkHost::Macos => 512,
        // Mobile: big enough to feed the batched attend (gate b ≥ 32)
        // and the blocked SDOT GEMM without the memory of 512.
        ChunkHost::Aarch64 => 256,
        ChunkHost::Other if dense_on_discrete => DISCRETE_DENSE_PREFILL_CHUNK,
        ChunkHost::Other => 48,
    }
}

/// What the chunk rule needs to know about a loaded stack.
#[derive(Clone, Copy, Debug, Default)]
struct ChunkStackFacts {
    /// Every layer is `AttnKind::Full` + `FfnKind::Dense`, and no
    /// architecture-owned stack (g3n, DeepSeek-V4/V4.1, qwen4-exp) is set.
    plain_dense: bool,
    /// The active GPU backend is a discrete card.
    discrete: bool,
    /// The backend is up and not paused.
    gpu_on: bool,
    /// A capacity-derived device prefix: some layers run on the host.
    capacity_split: bool,
    /// An in-process multi-GPU plan is set.
    multi_gpu: bool,
    /// O(1) layers (their Q trace is recorded by the prefill).
    o1: bool,
}

impl ChunkStackFacts {
    fn dense_on_discrete(self) -> bool {
        self.plain_dense
            && self.discrete
            && self.gpu_on
            && !self.capacity_split
            && !self.multi_gpu
            && !self.o1
    }
}

/// Number of prompt rows that have a real teacher-forced next-token pair in a
/// prefill span.  The final prompt row has no successor token, so it must not
/// be handed to the MTP warm-up.  Keeping this arithmetic in one helper makes
/// the full-chunk and tail-chunk boundaries explicit for both the graph and
/// CPU implementations.
#[inline]
fn mtp_prefill_pair_count(start: usize, end: usize, input_len: usize) -> usize {
    if end <= start || start >= input_len {
        return 0;
    }
    let rows = (end.min(input_len) - start).min(input_len - start);
    if end < input_len {
        rows
    } else {
        rows.saturating_sub(1)
    }
}

/// Callback for streaming tokens. Return `false` to cancel.
pub type TokenCallback = Box<dyn FnMut(&str) -> bool + Send>;

impl Pipeline {
    /// Clear all per-sequence state, including backend device mirrors.
    ///
    /// The host KV/history buffers are only half of the request lifecycle on
    /// wgpu: GDN/O(1) state and cached graph bind groups are keyed by the
    /// pipeline id and otherwise survive a pooled request.  Keep every fresh
    /// sequence entry point on this one reset path so a new request cannot
    /// inherit the prior request's device state.
    fn clear_sequence_state(&mut self) {
        // a replay still writing the GDN owners must land before they are
        // cleared or reallocated (the device holds raw pointers to them)
        #[cfg(target_os = "macos")]
        let _ = crate::gpu_metal::wait_replay();
        self.kv_cache.clear();
        self.kv_history.clear();
        if let Some(b) = &mut self.dsv41 {
            b.3.clear();
        }
        crate::gpu::graph_kv_reset(self.graph_kv_id);
        // MTP is detached from `self` for the duration of generation, so its
        // device mirror is not covered by the trunk reset above.  Reset the
        // derived id as well: a failed/aborted warm-up must never leave a
        // mirror that a later request can mistake for a current MTP cache.
        crate::gpu::graph_kv_reset(self.mtp_kv_id());
    }

    /// Finish a generation lifecycle after the MTP/router owners were
    /// detached.  Every terminal path must put those owners back before the
    /// pooled pipeline can serve another request.  Graph side channels and
    /// device mirrors are cleared on errors and cancellations; a successful
    /// generation keeps its decode-ready host cache for KV reuse.
    fn finish_generation(
        &mut self,
        mtp: &mut Option<MtpModule>,
        router: &mut Option<crate::swarm::DynRouter>,
        clear_sequence: bool,
    ) {
        // A dynamic route may have switched the overlay before the terminal
        // path. Restore the backbone while the detached router is still
        // available, because set_active_skill also owns the overlay reset.
        if router.is_some() {
            let _ = self.set_active_skill(None);
        }
        // The last speculative round's replay may still be in flight on
        // the second queue: whoever reads the host cache after generate()
        // returns (session export, the network split's KV wire, a KV
        // reuse) must see the final states.
        // A replay that failed leaves the GDN owners half-written: fail
        // closed and drop the sequence instead of handing the cache on.
        #[cfg(target_os = "macos")]
        let clear_sequence = clear_sequence || !crate::gpu_metal::wait_replay();
        if clear_sequence {
            self.clear_sequence_state();
            if let Some(m) = mtp.as_mut() {
                // The MTP owner is detached while generation runs, so the
                // trunk reset above cannot clear its host cache.  Drop its
                // partial rows before reattaching it to the pooled pipeline;
                // the next request must start from the same empty anchor on
                // CPU and on the device mirror.
                m.kv.clear();
            }
            if let Some(m) = self.mtp.as_mut() {
                // A non-speculative request leaves the configured MTP owner
                // attached.  Clear that dormant cache too when a shared
                // generation failure/cancellation resets the sequence.
                m.kv.clear();
            }
        }
        self.graph_want_logits = false;
        self.graph_head_required = false;
        self.graph_logits = None;
        self.graph_failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.cancel
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.dyn_router = router.take().or(self.dyn_router.take());
        self.mtp = mtp.take().or(self.mtp.take());
        self.mtp_graph_mode = None;
        self.spec_forced = None;
    }

    /// Consume a graph failure reported by a forward that returns only a
    /// hidden vector.  `forward_ids` is a public Result API, so it must not
    /// turn the graph's zero hidden sentinel into a valid lm_head result.
    fn check_forward_graph(&mut self, phase: &str, pos: usize) -> Result<(), String> {
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.cancel
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.clear_sequence_state();
            self.graph_logits = None;
            self.graph_want_logits = false;
            self.graph_head_required = false;
            return Err(format!("GPU graph failed during {phase} at position {pos}"));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn fail_metal_graph(&mut self, reason: &str) {
        crate::pipeline::METAL_GRAPH_ERRORS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.clear_sequence_state();
        self.graph_logits = None;
        self.graph_failed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        tracing::error!("native Metal TokenGraph failed closed: {reason}");
    }

    /// Start an NLL/PPL request with all graph side channels in a known
    /// state.  A graph failure also raises the cooperative cancel bit; it is
    /// consumed here and that graph-induced bit is cleared so an independent
    /// request can be reused.  A caller-owned cancellation remains intact.
    fn nll_begin(&mut self) -> Result<(), String> {
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.cancel
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.clear_sequence_state();
            self.graph_logits = None;
            self.graph_want_logits = false;
            self.graph_head_required = false;
            return Err("GPU graph failed before NLL scoring".to_string());
        }
        self.clear_sequence_state();
        self.graph_logits = None;
        self.graph_want_logits = false;
        self.graph_head_required = false;
        Ok(())
    }

    /// End an NLL/PPL request, including the side channels that are not part
    /// of the host KV cache.  This is intentionally explicit instead of
    /// relying on a tuple/sentinel return: callers must see every failure.
    fn nll_end(&mut self) {
        self.clear_sequence_state();
        self.graph_logits = None;
        self.graph_want_logits = false;
        self.graph_head_required = false;
        self.graph_failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check the graph failure channel at a scoring boundary and leave the
    /// pipeline reusable when the device path failed.
    fn nll_check_graph(&mut self, phase: &str, pos: usize) -> Result<(), String> {
        #[cfg(test)]
        if self.nll_test_fail_at == Some(pos) {
            self.nll_test_fail_at = None;
            self.graph_failed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.cancel
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.clear_sequence_state();
            self.graph_logits = None;
            self.graph_want_logits = false;
            return Err(format!(
                "GPU graph failed during NLL {phase} at position {pos}"
            ));
        }
        Ok(())
    }

    /// Map a virtual layer index to its physical weight index.
    /// Looped Transformer (Nanbeige 4.2): 22 physical layers × 2 loops = 44 virtual;
    /// virtual layer 23 maps back to physical layer 1 (23 % 22 = 1).
    #[inline]
    pub fn phys_layer(&self, virtual_idx: usize) -> usize {
        virtual_idx % self.physical_layers
    }

    /// True when `virtual_idx` is the last layer of a loop iteration
    /// (used for loop_final_norm insertion).
    #[inline]
    pub fn is_loop_end(&self, virtual_idx: usize) -> bool {
        self.loop_final_norm && (virtual_idx + 1) % self.physical_layers == 0
    }

    /// Build a pipeline from parts (used by the loader and tests).
    #[allow(clippy::too_many_arguments)]

    /// Whole-block q1 token graph on the GPU (macOS/Metal): the run of
    /// consecutive q1 layers — GDN *and* full attention — starting at
    /// `start` executes as few command buffers as the CPU truly needs.
    /// Hidden stays device-resident across every layer; the only syncs
    /// are before each CPU attend (it needs q/k/v and owns the KV
    /// cache) and the final hidden readback. Recurrent states
    /// round-trip through shared memory (the CPU stays their owner, so
    /// every other path remains coherent). Returns the first layer
    /// index NOT covered (== `start` → refused, caller falls through
    /// to the per-layer CPU path).
    /// Should prefill run position-by-position through the GPU token
    /// graph instead of the batched CPU chunk-GEMM? True for q1 GDN
    /// hybrids on native Metal: their chunk prefill is walled by the
    /// sequential scalar recurrence, so the graph's decode rate wins.
    /// NOT for Looped Transformers, despite the per-chunk loop_final_norm
    /// sync: the chunk-GEMM amortizes each weight over the whole chunk,
    /// which the per-position graph cannot (Nanbeige 4.2 on M4, 512-token
    /// prompt: 85 tok/s chunked vs 14 through the graph).
    #[cfg(target_os = "macos")]
    fn graph_prefill_preferred(&self) -> bool {
        let graph_force = crate::gpu::q1_force() || crate::gpu::q2tp_gpu_opt_in();
        if !crate::gpu::enabled_here()
            || !graph_force
            || std::env::var("CMF_GPU_BLOCK")
                .map(|v| v == "0")
                .unwrap_or(false)
            // CMF_PREFILL_GRAPH=0: the chunked prefill (GEMM projections,
            // CPU recurrence) instead of the per-position token graph.
            || std::env::var("CMF_PREFILL_GRAPH").as_deref() == Ok("0")
        {
            return false;
        }
        self.weights
            .layers
            .iter()
            .any(|lw| {
                matches!(&lw.attn, AttnKind::LinearGdn(w) if w.in_proj_qkv.metal_graph_parts().is_some())
            })
    }

    #[cfg(not(target_os = "macos"))]
    fn graph_prefill_preferred(&self) -> bool {
        // Discrete-GPU wgpu whole-token graph: GDN layers carry recurrent state
        // (conv ring + delta-rule S) resident on the GPU. A batched CPU prefill
        // builds that state on the CPU only, leaving the GPU buffers zeroed at
        // decode → garbage. Route GDN-hybrid prefill through the graph one
        // position at a time so the resident state is seeded exactly as decode
        // will read it. Pure-attention models keep the batched CPU prefill (its
        // KV mirror re-syncs from the CPU cache, so no seeding gap).
        let graph_on = crate::gpu::wgpu_graph_on(crate::gpu::GraphPhase::Prefill);
        if !graph_on || !crate::gpu::enabled_here() {
            return false;
        }
        // The descriptor-aware Prism graph now carries both the FWHT/affine
        // transforms and resident GDN state, so it is also the exact prefill
        // path for this model.  Keeping it here (rather than falling through
        // to the CPU chunk walk) is required for a long prompt to seed the
        // same device state that decode consumes.
        // O(1) needs the CPU prefill: the q-trace that seals the Nyström
        // skeleton is recorded there and nowhere else. The GDN half of
        // the hybrid loses nothing — the graph's first decode creates
        // its (ring, S) entries seeded from `cpu_state`, the same
        // handoff every graph run relies on when the entry is fresh.
        // Without this line the two designs collide on hybrids and o1
        // never becomes graph-portable: prefill through the graph
        // records no trace, so views stay None forever.
        if self.o1_active() {
            return false;
        }
        if self
            .weights
            .layers
            .iter()
            .any(|lw| matches!(&lw.attn, AttnKind::LinearGdn(_)))
        {
            return true;
        }
        // MoE models too: the chunked CPU prefill runs every expert on the
        // host (Hy-MT2-30B-A3B on a Xeon: 8 tok/s of ingest against 53 of
        // graph decode), while the token graph — and the batched graph under
        // CMF_BATCH_K — keep the experts resident. Full attention in the
        // graph writes the KV mirror that decode reads, exactly as it does
        // for the hybrids' attention layers. Only when the whole stack is
        // resident: with a device prefix the per-position walk finishes
        // every token on the host, and the chunked prefill (GEMMs on the
        // card, the expert loop batched on the host) is the faster ingest
        // (the 8 GB ladder point: 7 tok/s chunked against ~1 walked).
        self.weights
            .layers
            .iter()
            .any(|lw| matches!(&lw.ffn, FfnKind::Moe(_)))
            && self.automatic_gpu_prefix().is_none()
    }

    #[cfg(target_os = "macos")]
    fn q1_graph_gpu(
        &mut self,
        start: usize,
        upto: Option<usize>,
        position: usize,
        h: &mut [f32],
    ) -> usize {
        let _mt0 = std::time::Instant::now(); // CMF_METAL_HOSTPROF
        use crate::gpu::{AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, GraphDims, MetalFfn, TokenGraph};
        let graph_force = crate::gpu::q1_force() || crate::gpu::q2tp_gpu_opt_in();
        if self.attn_softcap > 0.0 // capped scores: no graph kernel — CPU path
            || !crate::gpu::enabled_here()
            || !graph_force
            || std::env::var("CMF_GPU_BLOCK")
                .map(|v| v == "0")
                .unwrap_or(false)
        {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!(
                    "block-graph: front gate (softcap={} enabled_here={} graph_force={})",
                    self.attn_softcap > 0.0,
                    crate::gpu::enabled_here(),
                    graph_force,
                );
            }
            if self.graph_head_required {
                self.fail_metal_graph("native graph front gate refused");
            }
            return start;
        }
        // The graph encodes SiLU FFN and full-context attention with an
        // explicit model scale. Architectures with sliding windows,
        // sandwich norms or non-SiLU FFNs still fall back to the CPU path.
        if self.swa.is_some()
            || self.global_attn.is_some()
            || self.attention_heads_per_layer.is_some()
            || self.attn_v_norm
            || self.weights.layers.iter().any(|lw| {
                lw.attn_out_norm.is_some()
                    || lw.ffn_out_norm.is_some()
                    || lw.layer_scale.is_some()
                    || matches!(&lw.ffn, FfnKind::Dense(d) if d.act != Act::Silu)
            })
        {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!(
                    "block-graph: arch ineligible (swa={} gattn={} hpl={} vnorm={} scale_delta={:.2e})",
                    self.swa.is_some(),
                    self.global_attn.is_some(),
                    self.attention_heads_per_layer.is_some(),
                    self.attn_v_norm,
                    (self.attn_scale - 1.0 / (self.head_dim as f32).sqrt()).abs(),
                );
            }
            if self.graph_head_required {
                self.fail_metal_graph("native graph architecture gate refused");
            }
            return start;
        }
        // Looped Transformer: the graph covers ALL loop iterations;
        // encode_loop_norm is inserted on-device at each boundary.
        let limit = upto
            .map(|u| u + 1)
            .unwrap_or(self.num_layers)
            .min(self.num_layers);

        enum Item<'a> {
            Gdn {
                run: Vec<GdnGpuLayer<'a>>,
                first: usize,
            },
            Attn {
                l: AttnGpuLayer<'a>,
                li: usize,
                q_norm: Option<&'a [f32]>,
                k_norm: Option<&'a [f32]>,
                output_gate: bool,
                bias: Option<(&'a [f32], &'a [f32], &'a [f32])>,
                /// Attend on the device too (no sync): F32 KV, no
                /// o1/bias, dims inside the kernels' contract.
                full_gpu: bool,
            },
        }

        // Device-attend KERNEL contract, shared by every Full layer. The
        // hd>128 default-off POLICY is applied after the scan: it was
        // measured on dense models, and a MoE plan inverts it — with the
        // experts on device each CPU-attend sandwich costs a
        // commit+wait, ~30 submits/token (W2 on M4: 14.7 tok/s
        // sandwiched vs 27.1 device-attend vs 18.8 pure CPU).
        let attend_mode = std::env::var("CMF_GPU_ATTEND").unwrap_or_else(|_| "auto".into());
        let attend_contract = attend_mode != "0"
            && attend_mode != "off"
            && self.head_dim % 4 == 0
            && self.head_dim <= 256
            && self.rotary_dim >= 2
            && self.rotary_dim <= self.head_dim
            && (self.rotary_dim / 2) % 32 == 0
            && self.num_kv_heads > 0
            && self.num_heads % self.num_kv_heads == 0;

        let mut plan: Vec<Item> = Vec::new();
        let mut model_ref: Option<std::sync::Arc<cortiq_core::CmfModel>> = None;
        // Break-reason diagnostics ride the same env as the plan summary.
        let block_diag = std::env::var("CMF_GRAPH_DBG").is_ok();
        let mut scan = start;
        while scan < limit {
            let lw = &self.weights.layers[self.phys_layer(scan)];
            let ffn = match &lw.ffn {
                FfnKind::Dense(d) if d.segs.is_empty() => {
                    let (Some(g), Some(u), Some(dn)) = (
                        d.gate_proj.metal_graph_parts(),
                        d.up_proj.metal_graph_parts(),
                        d.down_proj.metal_graph_parts(),
                    ) else {
                        if block_diag {
                            eprintln!(
                                "block-graph: L{scan} FFN trio not graph-mappable — run ends"
                            );
                        }
                        break;
                    };
                    MetalFfn::Dense {
                        gate: g,
                        up: u,
                        down: dn,
                    }
                }
                FfnKind::Moe(m) => {
                    let Some(moe) = metal_moe_graph_parts(m, self.hidden_size) else {
                        if block_diag {
                            eprintln!(
                                "block-graph: L{scan} MoE outside the graph contract — run ends"
                            );
                        }
                        break;
                    };
                    if let QTensor::Mapped { model, .. } = &m.experts[0].gate_proj {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    MetalFfn::Moe(moe)
                }
                _ => {
                    if block_diag {
                        eprintln!("block-graph: L{scan} non-graph FFN — run ends");
                    }
                    break;
                }
            };
            match &lw.attn {
                AttnKind::LinearGdn(w) if self.gdn_cfg.is_some() => {
                    let parts = (
                        w.in_proj_qkv.metal_graph_parts(),
                        w.in_proj_z.metal_graph_parts(),
                        w.in_proj_a.f32_parts(),
                        w.in_proj_b.f32_parts(),
                        w.out_proj.metal_graph_parts(),
                    );
                    let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = parts else {
                        if block_diag {
                            eprintln!(
                                "block-graph: L{scan} GDN parts refused (qkv={} z={} a_f32={} b_f32={} out={})",
                                w.in_proj_qkv.metal_graph_parts().is_some(),
                                w.in_proj_z.metal_graph_parts().is_some(),
                                w.in_proj_a.f32_parts().is_some(),
                                w.in_proj_b.f32_parts().is_some(),
                                w.out_proj.metal_graph_parts().is_some(),
                            );
                        }
                        break;
                    };
                    if let QTensor::Mapped { model, .. } = &w.in_proj_qkv {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let gl = GdnGpuLayer {
                        attn_norm: &lw.input_norm,
                        post_norm: &lw.post_norm,
                        qkv,
                        z,
                        a,
                        b,
                        out,
                        ffn,
                        conv1d: &w.conv1d,
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                        gnorm: &w.norm,
                    };
                    match plan.last_mut() {
                        Some(Item::Gdn { run, .. }) => run.push(gl),
                        _ => plan.push(Item::Gdn {
                            run: vec![gl],
                            first: scan,
                        }),
                    }
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate: None,
                    bias,
                } if !self.kv_cache.layers[scan].o1_sealed()
                    // Sealed o1 stays plannable when the Metal o1 port
                    // is on: full_gpu attends through the device state,
                    // and any refusal falls to the sandwich, whose CPU
                    // core routes sealed layers through the nystrom step.
                    || std::env::var("CMF_O1_METAL").as_deref() == Ok("1") =>
                {
                    let parts = (
                        wq.metal_graph_parts(),
                        wk.metal_graph_parts(),
                        wv.metal_graph_parts(),
                        wo.metal_graph_parts(),
                    );
                    let (Some(pq), Some(pk), Some(pv), Some(po)) = parts else {
                        break;
                    };
                    if let QTensor::Mapped { model, .. } = wq {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let cache = &self.kv_cache.layers[scan];
                    // O(1) layer on Metal: the device attends through the
                    // sealed Nystrom state (opt-in while the port proves
                    // itself). Unsealed -> sandwich path = the CPU o1 step.
                    let o1_metal = cache.o1.is_some()
                        && std::env::var("CMF_O1_METAL").as_deref() == Ok("1")
                        && cache.o1_views().is_some();
                    let full_gpu = attend_contract
                        && cache.mode == crate::kv_cache::KvMode::F32
                        && (cache.o1.is_none() || o1_metal)
                        && bias.is_none()
                        && pq.1 == self.num_heads * self.head_dim * (1 + *output_gate as usize)
                        && pk.1 == self.num_kv_heads * self.head_dim
                        && pv.1 == self.num_kv_heads * self.head_dim
                        && po.2 == self.num_heads * self.head_dim;
                    plan.push(Item::Attn {
                        l: AttnGpuLayer {
                            attn_norm: &lw.input_norm,
                            post_norm: &lw.post_norm,
                            wq: pq,
                            wk: pk,
                            wv: pv,
                            wo: po,
                            ffn,
                        },
                        li: scan,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        bias: bias
                            .as_ref()
                            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        full_gpu,
                    });
                }
                _ => break,
            }
            scan += 1;
        }
        let Some(model) = model_ref else {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!("q1-graph: no model ref (start {start}, scanned to {scan})");
            }
            if self.graph_head_required {
                self.fail_metal_graph("native graph has no mapped model reference");
            }
            return start;
        };
        if plan.is_empty() {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!("q1-graph: empty plan at layer {start}");
            }
            if self.graph_head_required {
                self.fail_metal_graph("native graph plan is empty");
            }
            return start;
        }
        let has_moe = plan.iter().any(|it| match it {
            Item::Gdn { run, .. } => run.iter().any(|l| matches!(l.ffn, MetalFfn::Moe(_))),
            Item::Attn { l, .. } => matches!(l.ffn, MetalFfn::Moe(_)),
        });
        let has_gdn = plan.iter().any(|it| matches!(it, Item::Gdn { .. }));
        let dev_attend = attend_contract
            && (self.head_dim <= 128
                || has_moe
                // A GDN hybrid attends on a quarter of its layers: the
                // hd>128 caution was measured on pure-dense models where
                // gqa_attend dominates, and on Qwen3.8-27B (hd 256, 48
                // GDN + 16 attn) the sandwich costs 2x the whole decode
                // (1.2 vs 2.21 tok/s measured before the arena fix).
                || (self.head_dim <= 256 && has_gdn)
                || attend_mode == "force"
                || attend_mode == "256");
        if !dev_attend {
            for it in &mut plan {
                if let Item::Attn { li, full_gpu, .. } = it {
                    // The hd>128 policy is about gqa_attend; an o1 layer
                    // attends through its own kernel set.
                    let keep_o1 = self.kv_cache.layers[*li].o1.is_some()
                        && std::env::var("CMF_O1_METAL").as_deref() == Ok("1");
                    if !keep_o1 {
                        *full_gpu = false;
                    }
                }
            }
        }
        if std::env::var("CMF_GRAPH_DBG").is_ok() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                let fg = plan
                    .iter()
                    .filter(|it| matches!(it, Item::Attn { full_gpu: true, .. }))
                    .count();
                let att = plan
                    .iter()
                    .filter(|it| matches!(it, Item::Attn { .. }))
                    .count();
                eprintln!(
                    "q1-graph: plan of {} items from layer {start} to {scan} | dev_attend={dev_attend} full_gpu {fg}/{att} | hd={} rd={} nkv={} nh={}",
                    plan.len(),
                    self.head_dim,
                    self.rotary_dim,
                    self.num_kv_heads,
                    self.num_heads,
                );
            }
        }
        let dims = GraphDims {
            hidden: self.hidden_size,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        let Some(mut graph) = TokenGraph::new(&model, dims, h) else {
            if self.graph_head_required {
                self.fail_metal_graph("native TokenGraph allocation refused");
            }
            return start;
        };
        let gcfg = self.gdn_cfg.map(|cfg| GdnGpuCfg {
            nv: cfg.num_v_heads,
            nk: cfg.num_k_heads,
            dk: cfg.key_head_dim,
            dv: cfg.value_head_dim,
            kk: cfg.conv_kernel,
            hidden: self.hidden_size,
            inter: self.intermediate_size,
            c_dim: cfg.conv_dim(),
            eps: cfg.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        });
        // Validate the whole plan BEFORE encoding anything: after the
        // first sync a refused layer would leave the token
        // half-executed, so truncate to the provably encodable prefix.
        let mut valid = 0usize;
        let mut end = start;
        crate::gpu::stageprof(1, _mt0.elapsed()); // конец планирования
        if std::env::var("CMF_PLAN_DUMP").is_ok() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                for it in &plan {
                    match it {
                        Item::Gdn { first, run } => {
                            eprintln!("plan: Gdn first={first} len={}", run.len())
                        }
                        Item::Attn { li, full_gpu, .. } => {
                            eprintln!("plan: Attn li={li} full_gpu={full_gpu}")
                        }
                    }
                }
            });
        }
        for item in &plan {
            let ok = match item {
                Item::Gdn { run, .. } => gcfg
                    .as_ref()
                    .map(|gc| run.iter().all(|l| graph.gdn_ok(l, gc)))
                    .unwrap_or(false),
                Item::Attn { l, .. } => graph.attn_ok(l),
            };
            if !ok {
                if block_diag {
                    eprintln!(
                        "block-graph: plan item {} ({}) failed graph preflight",
                        valid,
                        match item {
                            Item::Gdn { run, first } => format!("GDN run L{first}+{}", run.len()),
                            Item::Attn { li, .. } => format!("Attn L{li}"),
                        }
                    );
                }
                break;
            }
            valid += 1;
            end += match item {
                Item::Gdn { run, .. } => run.len(),
                Item::Attn { .. } => 1,
            };
        }
        plan.truncate(valid);
        if plan.is_empty() {
            if self.graph_head_required {
                self.fail_metal_graph("native graph preflight produced no valid items");
            }
            return start;
        }

        if self.graph_head_required && (upto.is_some() || end != self.num_layers) {
            self.fail_metal_graph("fused-head NLL requires a complete 64-layer graph");
            return start;
        }

        let inv_freq = self.inv_freq.clone();
        let pool = self.pool.clone();
        let (nh, nkv, hd, hs, rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let norm_style = self.norm_style;
        let gemma = norm_style == cortiq_core::NormStyle::Gemma;
        let want = self.gdn_cfg.map(|c| c.state_len()).unwrap_or(0);
        let kv_id = self.graph_kv_id;
        // GDN runs whose states await readback after the next sync
        // (device-attended layers add no sync, so several may stack).
        let mut pending: Vec<(usize, usize)> = Vec::new();
        // Device-attended layers: their K/V/imp are pulled from the
        // mirror after the final sync.
        let mut dev_attn: Vec<usize> = Vec::new();
        for item in &plan {
            let _xt0 = std::time::Instant::now();
            let _xkind: u32 = match item {
                Item::Gdn { .. } => 2,
                Item::Attn { .. } => 3,
            };
            // Looped Transformer: insert on-device norm at loop boundaries.
            if self.loop_final_norm {
                let item_start = match item {
                    Item::Gdn { first, .. } => *first,
                    Item::Attn { li, .. } => *li,
                };
                if item_start > start && self.is_loop_end(item_start - 1) {
                    graph.encode_loop_norm(&self.weights.final_norm);
                }
            }
            match item {
                Item::Gdn { run, first } => {
                    for l in &mut self.kv_cache.layers[*first..*first + run.len()] {
                        if l.linear_state.len() != want {
                            l.linear_state = vec![0f32; want];
                        }
                    }
                    let ro: Vec<&[f32]> = self.kv_cache.layers[*first..*first + run.len()]
                        .iter()
                        .map(|l| l.linear_state.as_slice())
                        .collect();
                    let _ig = std::time::Instant::now();
                    if !graph.encode_gdn_run(run, &ro, gcfg.as_ref().unwrap()) {
                        // Unreachable: the plan was validated above.
                        tracing::error!("q1 graph: GDN run refused after validation");
                        return start;
                    }
                    // Early commit: the GPU starts the run while the
                    // CPU encodes the next layer (nothing to wait on).
                    graph.commit_kind = 2;
                    graph.commit();
                    crate::gpu::stageprof(0, _ig.elapsed());
                    pending.push((*first, run.len()));
                }
                Item::Attn {
                    l,
                    li,
                    q_norm,
                    k_norm,
                    output_gate,
                    bias,
                    full_gpu,
                } => {
                    let _ia = std::time::Instant::now();
                    // ── Fully device-resident attention: no sync at all.
                    if *full_gpu {
                        let cache = &self.kv_cache.layers[*li];
                        let o1p = if cache.o1.is_some() {
                            match cache.o1_views() {
                                Some(views) => Some(crate::gpu::O1AttnParams {
                                    views,
                                    epoch: self.o1_epoch,
                                }),
                                // Sealed state gone mid-run: sandwich.
                                None => None,
                            }
                        } else {
                            None
                        };
                        let o1_layer = cache.o1.is_some();
                        if o1_layer && o1p.is_none() {
                            // fall to the sandwich (CPU o1 step)
                        }
                        let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
                        let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
                        let cpu_stored = if o1_layer { 0 } else { cpu_k[0].len() / hd };
                        let p = crate::gpu::AttnDeviceParams {
                            kv_id,
                            layer: *li,
                            nh,
                            nkv,
                            hd,
                            rd,
                            position,
                            scale: self.attn_scale,
                            eps: eps as f32,
                            gemma,
                            late_qk_norm: self.qk_norm_after_rope,
                            output_gate: *output_gate,
                            q_norm: *q_norm,
                            k_norm: *k_norm,
                            inv_freq: &inv_freq,
                            cpu_k,
                            cpu_v,
                            cpu_stored,
                            o1: o1p,
                        };
                        let o1_bad = o1_layer && p.o1.is_none();
                        if !o1_bad && graph.attn_device_ok(l, &p) && graph.encode_attn_device(l, &p)
                        {
                            // o1 layers leave no mirror row to pull.
                            if p.o1.is_none() {
                                dev_attn.push(*li);
                            }
                            graph.commit_kind = 3;
                            graph.commit();
                            // The footer below is skipped by `continue`:
                            // account the device-attn item here or its
                            // cost hides from the stage profile entirely.
                            crate::gpu::stageprof(_xkind, _xt0.elapsed());
                            continue;
                        }
                        // Mirror refused (nothing encoded) → sandwich.
                    }
                    graph.encode_attn_prefix(l);
                    if let Err(err) = graph.sync_checked() {
                        self.fail_metal_graph(&err);
                        return start;
                    }
                    if !pending.is_empty() {
                        let idxs: Vec<usize> =
                            pending.drain(..).flat_map(|(f, n)| f..f + n).collect();
                        let mut outs: Vec<&mut [f32]> = self
                            .kv_cache
                            .layers
                            .iter_mut()
                            .enumerate()
                            .filter(|(i, _)| idxs.binary_search(i).is_ok())
                            .map(|(_, s)| s.linear_state.as_mut_slice())
                            .collect();
                        graph.read_states(&mut outs);
                    }
                    let mut q_raw = attention::take_buf(l.wq.1);
                    let mut k = attention::take_buf(l.wk.1);
                    let mut v = attention::take_buf(l.wv.1);
                    graph.read_qkv(&mut q_raw, &mut k, &mut v);
                    let cfg = QwenAttnCfg {
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq,
                        rotary_dim: rd,
                        scale: self.attn_scale,
                        softcap: self.attn_softcap,
                        window: None,
                        v_norm: false,
                        qk_norm_after_rope: self.qk_norm_after_rope,
                        q_norm: *q_norm,
                        k_norm: *k_norm,
                        output_gate: *output_gate,
                        softplus_gate: None,
                        rope_scale: 1.0,
                        bias: *bias,
                        rms_eps: eps,
                        norm_style,
                        pool: pool.as_deref(),
                    };
                    // CMF_ATTN_ORACLE=1: diff the device attend against
                    // this CPU attend on identical inputs (bring-up).
                    let oracle = std::env::var("CMF_ATTN_ORACLE").as_deref() == Ok("1")
                        || std::env::var("CMF_ATTN_DUMP").is_ok();
                    let _ = full_gpu;
                    let oracle_in = oracle.then(|| (q_raw.clone(), k.clone(), v.clone()));
                    let mut ao = attention::qwen_attention_core(
                        q_raw,
                        k,
                        v,
                        &mut self.kv_cache.layers[*li],
                        &cfg,
                    );
                    // CMF_ATTN_DUMP=<dir>: this token's rope'd Q and the layer's whole
                    // K/V cache as raw f32 (offline attention-statistics probes:
                    // block bounds, mass concentration). Needs CMF_GPU_ATTEND=0.
                    if let Ok(dir) = std::env::var("CMF_ATTN_DUMP") {
                        if let Some((qr0, k0, v0)) = oracle_in.clone() {
                            let (cq, _cg, _ck, _cv) =
                                attention::finish_projection_debug(qr0, k0, v0, &cfg, position);
                            let cache = &self.kv_cache.layers[*li];
                            let n = cache.head_keys(0).len() / hd;
                            let mut bytes: Vec<u8> = Vec::new();
                            for v in [nh as u32, nkv as u32, hd as u32, n as u32, position as u32] {
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                            for v in &cq {
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                            for g in 0..nkv {
                                for v in cache.head_keys(g) {
                                    bytes.extend_from_slice(&v.to_le_bytes());
                                }
                            }
                            for g in 0..nkv {
                                for v in cache.head_values(g) {
                                    bytes.extend_from_slice(&v.to_le_bytes());
                                }
                            }
                            let _ =
                                std::fs::write(format!("{dir}/L{li}_pos{position}.bin"), &bytes);
                        }
                    }
                    if let Some((qr0, k0, v0)) =
                        oracle_in.filter(|_| std::env::var("CMF_ATTN_ORACLE").as_deref() == Ok("1"))
                    {
                        let (cq, _cg, ck, cv) =
                            attention::finish_projection_debug(qr0, k0, v0, &cfg, position);
                        let mut h_now = vec![0f32; hs];
                        graph.read_h(&mut h_now);
                        let cache = &self.kv_cache.layers[*li];
                        let n_after = cache.head_keys(0).len() / hd;
                        // A sealed O(1) cache may have no dense current-row
                        // entry. The oracle is a debug probe, so let it see
                        // zero stored exact rows instead of underflowing.
                        let stored = n_after.saturating_sub(1);
                        let cpu_k: Vec<&[f32]> = (0..nkv)
                            .map(|g| &cache.head_keys(g)[..stored * hd])
                            .collect();
                        let cpu_v: Vec<&[f32]> = (0..nkv)
                            .map(|g| &cache.head_values(g)[..stored * hd])
                            .collect();
                        let p = crate::gpu::AttnDeviceParams {
                            kv_id,
                            layer: *li,
                            nh,
                            nkv,
                            hd,
                            rd,
                            position,
                            scale: self.attn_scale,
                            eps: eps as f32,
                            gemma,
                            late_qk_norm: self.qk_norm_after_rope,
                            output_gate: *output_gate,
                            q_norm: *q_norm,
                            k_norm: *k_norm,
                            inv_freq: &inv_freq,
                            cpu_k,
                            cpu_v,
                            cpu_stored: stored,
                            o1: None,
                        };
                        if let Some((dq, dk, dv, dao)) = graph.debug_attn_device(l, &p, &h_now) {
                            let md = |a: &[f32], b: &[f32]| {
                                a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
                            };
                            let nn = |a: &[f32]| a.iter().map(|x| x * x).sum::<f32>().sqrt();
                            eprintln!(
                                "attn-oracle L{li} pos {position}: |q| {:.2} max|dq| {:.4} | |k| {:.2} max|dk| {:.4} | |v| {:.2} max|dv| {:.4} | |ao| {:.2} max|dao| {:.4}",
                                nn(&cq),
                                md(&cq, &dq),
                                nn(&ck),
                                md(&ck, &dk),
                                nn(&cv),
                                md(&cv, &dv),
                                nn(&ao),
                                md(&ao, &dao)
                            );
                        } else {
                            eprintln!("attn-oracle L{li}: device probe declined");
                        }
                    }
                    graph.encode_attn_suffix(l, &ao);
                    // Early commit: the GPU starts O+FFN while the CPU
                    // encodes the following GDN run / attention prefix.
                    graph.commit();
                    attention::recycle_buf(&mut ao);
                }
            }

            crate::gpu::stageprof(_xkind, _xt0.elapsed());
        }
        // Ride the final norm + lm_head in the same command buffer when
        // this run reaches the model's end and the caller wants logits:
        // the separate per-op lm_head submit (a full round trip) folds
        // into the sync that already happens here.
        let mut lm_rows = None;
        if self.graph_want_logits
            && upto.is_none()
            && end == self.num_layers
            && std::env::var("CMF_GPU_LMHEAD")
                .map(|v| v != "0")
                .unwrap_or(true)
        {
            if let Some(lm) = self.weights.lm_head.metal_graph_parts() {
                if graph.lm_head_ok(lm) {
                    graph.encode_lm_head(&self.weights.final_norm, lm);
                    lm_rows = Some(lm.1);
                }
            }
        }
        if self.graph_head_required && lm_rows.is_none() {
            METAL_GRAPH_HEAD_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.fail_metal_graph("fused graph head was requested but not encodable");
            return start;
        }
        let _sy0 = std::time::Instant::now();
        if let Err(err) = graph.sync_checked() {
            self.fail_metal_graph(&err);
            return start;
        }
        let _rs0 = std::time::Instant::now();
        if !pending.is_empty() {
            let idxs: Vec<usize> = pending.drain(..).flat_map(|(f, n)| f..f + n).collect();
            let mut outs: Vec<&mut [f32]> = self
                .kv_cache
                .layers
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| idxs.binary_search(i).is_ok())
                .map(|(_, s)| s.linear_state.as_mut_slice())
                .collect();
            graph.read_states(&mut outs);
        }
        if std::env::var("CMF_GRAPH_HOSTPROF").as_deref() == Ok("1") {
            use std::sync::atomic::{AtomicU64, Ordering};
            static SY: AtomicU64 = AtomicU64::new(0);
            static RS: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            SY.fetch_add((_rs0 - _sy0).as_nanos() as u64, Ordering::Relaxed);
            RS.fetch_add(_rs0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let n = N.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 {
                eprintln!(
                    "postprof: sync-wait {:.1} ms/ток | read_states {:.1} ms/ток ({n})",
                    SY.load(Ordering::Relaxed) as f64 / n as f64 / 1e6,
                    RS.load(Ordering::Relaxed) as f64 / n as f64 / 1e6
                );
            }
        }
        if let Some(rows) = lm_rows {
            crate::gpu::hostprof_encode_done(_mt0);
            let mut lg = attention::take_buf(rows.min(self.vocab_size));
            graph.read_logits(&mut lg);
            crate::gpu::hostprof_total(_mt0);
            lg.resize(self.vocab_size, 0.0);
            if let Some(c) = self.final_softcap {
                for l in lg.iter_mut() {
                    *l = c * (*l / c).tanh();
                }
            }
            self.graph_logits = Some(lg);
        }
        graph.read_h(h);
        if self.graph_head_required && self.graph_logits.is_none() {
            METAL_GRAPH_HEAD_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.fail_metal_graph("fused graph head completed without logits readback");
            return start;
        }
        METAL_GRAPH_TOK_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        METAL_GRAPH_LAYERS.fetch_add(
            end.saturating_sub(start) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if self.graph_head_required {
            METAL_GRAPH_HEAD_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // Device-attended layers: replay the CPU bookkeeping — append
        // the mirror's new K/V row (rope'd on the GPU) into the owner
        // cache, then bank this token's attention-importance mass.
        for li in dev_attn {
            let mut krow = attention::take_buf(nkv * hd);
            let mut vrow = attention::take_buf(nkv * hd);
            if crate::gpu::kv_mirror_read_last(kv_id, li, nkv, hd, &mut krow, &mut vrow) {
                let cache = &mut self.kv_cache.layers[li];
                cache.append(&krow, &vrow, &[]);
                let n = cache.seq_len;
                let mut imp = attention::take_buf(n);
                crate::gpu::kv_mirror_take_imp(kv_id, li, &mut imp);
                cache.accumulate_imp(&imp);
                attention::recycle_buf(&mut imp);
            }
            attention::recycle_buf(&mut krow);
            attention::recycle_buf(&mut vrow);
        }
        end
    }

    pub fn new(
        tokenizer: Tokenizer,
        weights: PipelineWeights,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        physical_layers: usize,
        loop_final_norm: bool,
        vocab_size: usize,
        rms_eps: f64,
        rope_base: f32,
        norm_style: NormStyle,
        max_seq_len: usize,
        sampler_config: SamplerConfig,
    ) -> Self {
        let rng = match sampler_config.seed {
            Some(s) => SplitMix64::new(s),
            None => SplitMix64::from_entropy(),
        };
        let inv_freq = std::sync::Arc::new(attention::rope_inv_freq(head_dim, rope_base));
        let pool = Pool::from_env();
        if let Some(p) = &pool {
            tracing::info!("worker pool: {} threads", p.n_workers());
        }
        Self {
            gpu_plan: None,
            tokenizer: std::sync::Arc::new(tokenizer),
            kv_cache: KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len),
            sampler_config,
            weights,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            physical_layers,
            loop_final_norm,
            vocab_size,
            rms_eps,
            rope_base,
            norm_style,
            rotary_dim: head_dim,
            attention_heads_per_layer: None,
            vmf_cfg: None,
            gdn_cfg: None,
            kda_cfg: None,
            g3n: None,
            dsv4: None,
            dsv41: None,
            dsv41_vision: None,
            dsv41_prefill: None,
            qwen4_exp: None,
            dsv4_mtp: Vec::new(),
            dspark: None,
            dspark_pending: Vec::new(),
            dspark_hist: Vec::new(),
            dspark_real: Vec::new(),
            dspark_trunk_picks: Vec::new(),
            dspark_exp: Vec::new(),
            dspark_draft_ns: 0,
            logit_multiplier: None,
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            graph_failed: std::sync::atomic::AtomicBool::new(false),
            kv_history: Vec::new(),
            short_conv_cfg: None,
            mtp: None,
            speculative: std::env::var("CMF_MTP").map(|v| v != "0").unwrap_or(true),
            ignore_eos: false,
            draft_full_streak: 0,
            spec_k_adapt: None,
            spec_acc_ewma: 0.7,
            rng,
            sampler_scratch: SamplerScratch::default(),
            spec_forced: None,
            spec_q: Vec::new(),
            spec_p: Vec::new(),
            spec_res: Vec::new(),
            spec_qs: Vec::new(),
            spec_ps: Vec::new(),
            spec_ress: Vec::new(),
            mtp_graph_mode: None,
            #[cfg(target_os = "macos")]
            metal_verify: None,
            inv_freq,
            ws: ForwardScratch::new(hidden_size),
            pool,
            model: None,
            dyn_force_f32: false,
            dyn_skill_layers: Vec::new(),
            dyn_active: None,
            dyn_blend_loaded: false,
            dyn_phi_layer: None,
            dyn_phi_ema: Vec::new(),
            dyn_phi_seen: 0,
            dyn_router: None,
            o1_cfg: None,
            o1_epoch: 0,
            o1_flags: Vec::new(),
            trace: false,
            calib_temp: 1.0,
            confidence_on: true,
            embed_multiplier: 1.0,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            swa: None,
            sliding_layers: None,
            inv_freq_local: None,
            rotary_dim_local: None,
            rope_scale: 1.0,
            rope_scale_local: 1.0,
            global_attn: None,
            inv_freq_global: None,
            attn_v_norm: false,
            qk_norm_after_rope: false,
            final_softcap: None,
            head_clusters: None,
            attn_softcap: 0.0,
            graph_want_logits: false,
            graph_head_required: false,
            graph_logits: None,
            graph_kv_id: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            #[cfg(test)]
            nll_test_fail_at: None,
            #[cfg(test)]
            nll_test_force_serial: false,
        }
    }

    /// Enable/disable per-layer O(1) Nyström attention. Only Full
    /// layers are eligible (a linear layer keeps its own operator).
    /// Applies to generation (`generate*`/`forward_ids`): the prompt
    /// pass stays exact, then the state seals after prefill or at the
    /// deferred skeleton-safe boundary for short prompts; decode runs on
    /// the O(1) state. Teacher-forced scoring (`ppl_ids`) intentionally
    /// stays exact.
    pub fn set_o1(&mut self, cfg: Option<crate::nystrom::O1Cfg>) {
        if let Some(c) = &cfg {
            if crate::nystrom::o1_deferred_boundary(c.w, c.sink).is_none() {
                tracing::error!(
                    "o1 disabled: w + sink + slack + 1 overflows usize (w={}, sink={})",
                    c.w,
                    c.sink
                );
                self.o1_flags.clear();
                self.o1_cfg = None;
                return;
            }
        }
        self.o1_flags = match &cfg {
            Some(c) => {
                let mut flags = c.layer_flags(self.num_layers);
                for (li, f) in flags.iter_mut().enumerate() {
                    if *f
                        && !matches!(
                            self.weights.layers[self.phys_layer(li)].attn,
                            AttnKind::Full { .. }
                        )
                    {
                        *f = false;
                    }
                }
                flags
            }
            None => Vec::new(),
        };
        if let Some(c) = &cfg {
            let n = self.o1_flags.iter().filter(|&&f| f).count();
            tracing::info!(
                "o1 nystrom attention: {n}/{} layer(s), m={} w={} sink={} rect={:?}",
                self.num_layers,
                c.m,
                c.w,
                c.sink,
                c.rect
            );
        }
        self.o1_cfg = cfg;
    }

    /// True when at least one layer runs the O(1) kernel.
    pub fn o1_active(&self) -> bool {
        self.o1_cfg.is_some() && self.o1_flags.iter().any(|&f| f)
    }

    /// Whether generation's prompt ingest is routed through the whole-token
    /// graph.  The bench uses this to label the measured generation prefill
    /// honestly; keep the predicate in Pipeline so CLI labels cannot drift
    /// from the production route.
    /// Positions per batched-graph submit for the prompt: `CMF_BATCH_K`
    /// when set (0 = one position at a time through the token graph),
    /// otherwise 32 on a discrete card whose prompt takes the graph route.
    /// The batched graph read a 2048-token prompt at 53 tok/s against 28.5
    /// one position at a time on an RTX PRO 4000 (Qwen3.8-27B q4tp: TTFT
    /// 39 s against 72), and its states are the speculative verify's,
    /// measured identical to the plain path. macOS keeps its own arm.
    pub fn generation_batch_k(&self) -> usize {
        if let Some(k) = std::env::var("CMF_BATCH_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return k;
        }
        #[cfg(not(target_os = "macos"))]
        if self.graph_prefill_preferred() && !self.o1_active() {
            return 32;
        }
        0
    }

    pub fn generation_graph_prefill(&self) -> bool {
        let graph = self.graph_prefill_preferred();
        // On wgpu, an active MTP head now consumes the trunk's graph batches
        // and warms its own block from those returned rows.  The selected
        // generation measurement is therefore the batched path, even though
        // the underlying GDN model still satisfies the graph-prefill
        // predicate.  Keep the CLI label tied to the actual route.  Native
        // Metal has a separate prefill-batch arm and retains its historical
        // label here.
        // A batched prompt (`generation_batch_k` > 0) is the batched graph
        // for every model on the graph route, not only those with an MTP
        // head — the label follows the route.
        #[cfg(not(target_os = "macos"))]
        if graph
            && self.generation_batch_k() > 0
            && std::env::var("CMF_MTP_CHAIN_PROBE").is_err()
        {
            return false;
        }
        graph
    }

    /// Device-side O(1) mirrors currently uploaded for this pipeline's
    /// sequence.  The count/bytes are zero before seal or after a fresh
    /// reset; callers use this to distinguish logical host state from the
    /// GPU allocation that actually serves decode.
    pub fn o1_device_stats(&self) -> (usize, u64) {
        crate::gpu::o1_device_stats(self.graph_kv_id)
    }

    /// Arm query collection on the o1 layers (fresh prompt pass).
    /// Reset the o1 layers to Collecting for a fresh sequence. Pub for the
    /// network split: each side runs the o1 lifecycle over ITS OWN layers
    /// (begin before prefill, seal at the prefill barrier).
    pub fn o1_begin(&mut self) {
        self.o1_begin_with_prefix(None);
    }

    /// Arm collection and optionally request a positive calibration prefix.
    /// The effective barrier is always at least the skeleton-safe floor, so
    /// a short requested prefix cannot create an exact-only runtime state.
    pub fn o1_begin_with_prefix(&mut self, requested_prefix: Option<usize>) {
        if let Some(c) = &self.o1_cfg {
            let (m, w, sink, rect) = (c.m, c.w, c.sink, c.rect);
            let boundary = requested_prefix.map(|p| {
                p.max(
                    crate::nystrom::o1_deferred_boundary(w, sink)
                        .expect("o1 config boundary validated in set_o1"),
                )
            });
            for (li, &f) in self.o1_flags.iter().enumerate() {
                if f {
                    self.kv_cache.layers[li].o1_begin_with_boundary(m, w, sink, rect, boundary);
                }
            }
        }
    }

    /// Effective deferred boundary for a positive prefix request.
    fn o1_effective_boundary(&self, requested_prefix: usize) -> Option<usize> {
        self.o1_cfg.as_ref().and_then(|c| {
            crate::nystrom::o1_deferred_boundary(c.w, c.sink)
                .map(|floor| requested_prefix.max(floor))
        })
    }

    fn o1_note_transition(&mut self) {
        // Drain every layer's one-shot bit before publishing one pipeline
        // epoch. `any()` would short-circuit on the first layer and leak the
        // remaining bits into later forwards, causing one epoch per layer.
        let mut transitioned = false;
        for (li, &flagged) in self.o1_flags.iter().enumerate() {
            if flagged {
                transitioned |= self.kv_cache.layers[li].take_o1_transition();
            }
        }
        if transitioned {
            self.o1_epoch = self.o1_epoch.wrapping_add(1);
        }
    }

    fn o1_pending(&self) -> bool {
        self.o1_flags.iter().enumerate().any(|(li, &f)| {
            f && self.kv_cache.layers[li].seq_len > 0
                && self.kv_cache.layers[li].o1_pending_boundary().is_some()
        })
    }

    fn o1_fail(&mut self, err: String) {
        tracing::error!("o1 deferred seal failed; terminating sequence: {err}");
        self.clear_sequence_state();
        self.graph_failed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Seal participating layers while retaining the exact state when the
    /// prompt is below the deferred boundary. A split worker may have
    /// collecting layers outside its owned span; zero-depth layers remain
    /// armed and are intentionally skipped until their peer runs them.
    pub fn o1_seal_checked(&mut self) -> Result<bool, String> {
        if self.o1_cfg.is_none() {
            return Ok(false);
        }
        let mut participating = false;
        for li in 0..self.num_layers {
            if !self.o1_flags.get(li).copied().unwrap_or(false) {
                continue;
            }
            if let Some(err) = self.kv_cache.layers[li].take_o1_error() {
                return Err(err);
            }
            if self.kv_cache.layers[li].seq_len == 0 {
                continue;
            }
            participating = true;
            let num_heads = self.layer_num_heads(li);
            self.kv_cache.layers[li].o1_seal_checked(num_heads)?;
        }
        self.o1_note_transition();
        for li in 0..self.num_layers {
            if self.o1_flags.get(li).copied().unwrap_or(false) {
                if let Some(err) = self.kv_cache.layers[li].take_o1_error() {
                    return Err(err);
                }
            }
        }
        Ok(participating
            && (0..self.num_layers).all(|li| {
                !self.o1_flags.get(li).copied().unwrap_or(false)
                    || self.kv_cache.layers[li].seq_len == 0
                    || self.kv_cache.layers[li].o1_sealed()
            }))
    }

    /// Complete a deferred boundary after a full position/span forward.
    /// This is the pipeline owner for epoch publication and failure cleanup.
    fn o1_progress(&mut self) {
        if !self.o1_active() {
            return;
        }
        for li in 0..self.num_layers {
            if self.o1_flags.get(li).copied().unwrap_or(false) {
                if let Some(err) = self.kv_cache.layers[li].take_o1_error() {
                    self.o1_fail(err);
                    return;
                }
            }
        }
        // A qwen_attention row can seal in the middle of a complete layer
        // walk. Consume its transition even though the pending boundary has
        // already disappeared from the cache.
        self.o1_note_transition();
        if !self.o1_pending() {
            return;
        }
        if let Err(err) = self.o1_seal_checked() {
            self.o1_fail(err);
        }
    }

    /// Turn a deferred O(1) failure raised by a hidden-only forward into the
    /// Result error its public batch/span caller must return. The failure
    /// path already cleared host/device sequence state; consume only the
    /// side-channel marker here and leave the pipeline reusable.
    fn check_o1_progress_failure(&mut self, phase: &str) -> Result<(), String> {
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.cancel
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.clear_sequence_state();
            return Err(format!("{phase}: deferred O(1) transition failed"));
        }
        Ok(())
    }

    /// Freeze landmarks + skeleton state after the prompt pass and drop
    /// the o1 layers' full KV; decode then runs `step()` per token.
    /// Pub for the network split (see `o1_begin`).
    pub fn o1_seal(&mut self) {
        if let Err(err) = self.o1_seal_checked() {
            self.o1_fail(err);
        }
    }

    /// Enable/disable the structured per-token telemetry trace (B4).
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on;
    }

    /// Replace all request-scoped sampler options and reset the random stream.
    /// This is required for deterministic `seed` semantics in pooled servers.
    pub fn set_sampler_config(&mut self, config: SamplerConfig) {
        self.rng = match config.seed {
            Some(seed) => SplitMix64::new(seed),
            None => SplitMix64::from_entropy(),
        };
        self.sampler_config = config;
    }

    /// Toggle the per-token confidence reduction (a full-vocab
    /// softmax each token). `bench --core` turns it off so the timed
    /// loop matches llama-bench's core contract; the result's
    /// `confidence` vec is empty while off.
    pub fn set_confidence(&mut self, on: bool) {
        self.confidence_on = on;
    }

    /// Set the confidence-calibration temperature (B1). Values ≤0 are
    /// clamped to raw (1.0).
    pub fn set_calib_temp(&mut self, t: f32) {
        self.calib_temp = if t > 1e-3 { t } else { 1.0 };
    }

    /// The active calibration temperature (1.0 = raw probability).
    pub fn calib_temp(&self) -> f32 {
        self.calib_temp
    }

    /// Partial rotary (Qwen3.5): rotate only the first `rotary_dim` dims;
    /// the frequency table is rebuilt over the rotary dims.
    pub fn set_rotary(&mut self, rotary_dim: usize, base: f32) {
        self.rotary_dim = rotary_dim.min(self.head_dim);
        self.inv_freq = std::sync::Arc::new(attention::rope_inv_freq(self.rotary_dim, base));
    }

    fn attn_cfg(&self, position: usize) -> QwenAttnCfg<'_> {
        QwenAttnCfg {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            hidden_size: self.hidden_size,
            position,
            inv_freq: &self.inv_freq,
            rotary_dim: self.rotary_dim,
            scale: self.attn_scale,
            softcap: self.attn_softcap,
            window: None,
            v_norm: false,
            qk_norm_after_rope: self.qk_norm_after_rope,
            q_norm: None,
            k_norm: None,
            output_gate: false,
            softplus_gate: None,
            rope_scale: self.rope_scale,
            bias: None,
            rms_eps: self.rms_eps,
            norm_style: self.norm_style,
            pool: self.pool.as_deref(),
        }
    }

    /// Generate text from a plain-text prompt. Streams tokens via `on_token`.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        task_mask: Option<&TaskMask>,
        on_token: Option<TokenCallback>,
    ) -> Result<GenerateResult, String> {
        let input_ids = self.tokenizer.with_bos(self.tokenizer.encode(prompt));
        self.generate_from_ids(&input_ids, max_tokens, task_mask, on_token)
    }

    /// Generate from a V4.1 multimodal prompt prepared by the vision module.
    /// Vision rows are encoded once and fed through the same bounded token walk as text.
    pub fn generate_from_vl(
        &mut self,
        input: &crate::dsv41_vision::PreparedVlInputs,
        max_tokens: usize,
        task_mask: Option<&TaskMask>,
        on_token: Option<TokenCallback>,
    ) -> Result<GenerateResult, String> {
        let Some(dsv41) = &self.dsv41 else {
            return Err("V4.1 multimodal input requires a DeepSeek-V4.1 pipeline".into());
        };
        if input.token_ids.is_empty() {
            return Err("empty V4.1 multimodal prompt".into());
        }
        if input.token_types.len() != input.token_ids.len() {
            return Err(format!(
                "V4.1 token type count {} != token count {}",
                input.token_types.len(),
                input.token_ids.len()
            ));
        }
        let dim = dsv41.2.dim;
        let mut embeddings = vec![None; input.token_ids.len()];
        let mut participates = vec![true; input.token_ids.len()];
        if !input.images.is_empty() {
            let vision = self
                .dsv41_vision
                .as_ref()
                .ok_or_else(|| "V4.1 image prompt has no loaded vision tower".to_string())?;
            for image in &input.images {
                let end = image.start.saturating_add(image.types.len());
                if end > input.token_ids.len() {
                    return Err(format!(
                        "V4.1 image span {}..{} exceeds prompt length {}",
                        image.start,
                        end,
                        input.token_ids.len()
                    ));
                }
                let mut span = vec![0.0f32; image.types.len() * dim];
                vision.fill_image_span(image, &mut span, self.pool.as_deref())?;
                for (offset, &kind) in image.types.iter().enumerate() {
                    let pos = image.start + offset;
                    if input.token_types[pos] != kind {
                        return Err(format!(
                            "V4.1 image type mismatch at position {pos}: {} != {kind}",
                            input.token_types[pos]
                        ));
                    }
                    embeddings[pos] = Some(span[offset * dim..(offset + 1) * dim].to_vec());
                    participates[pos] = false;
                }
            }
        }
        for (pos, &kind) in input.token_types.iter().enumerate() {
            if kind == crate::dsv41_vision::TEXT && embeddings[pos].is_some() {
                return Err(format!("V4.1 text position {pos} has an image embedding"));
            }
            if kind != crate::dsv41_vision::TEXT && embeddings[pos].is_none() {
                return Err(format!("V4.1 image position {pos} has no image embedding"));
            }
        }
        self.dsv41_prefill = Some((embeddings, participates));
        let result = self.generate_from_ids(&input.token_ids, max_tokens, task_mask, on_token);
        self.dsv41_prefill = None;
        result
    }

    /// `None` when the mask forbids nothing (see `TaskMask::fully_open`).
    fn drop_open_mask<'m>(&self, m: Option<&'m TaskMask>) -> Option<&'m TaskMask> {
        m.filter(|m| !m.fully_open(self.intermediate_size, self.num_heads))
    }

    /// Generate from prepared token ids (e.g. a chat template).
    ///
    /// With an MTP head, greedy generation without a task mask takes the
    /// speculative path: the MTP module drafts the token after next and
    /// the main model verifies both in one fused two-position forward
    /// (weights streamed once). The output is EXACTLY the vanilla greedy
    /// sequence — a rejected draft is rolled back — MTP only buys speed.
    pub fn generate_from_ids(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        task_mask: Option<&TaskMask>,
        mut on_token: Option<TokenCallback>,
    ) -> Result<GenerateResult, String> {
        if std::env::var("CMF_TRACE_H").is_ok() {
            eprintln!("input_ids: {input_ids:?}");
        }
        if input_ids.is_empty() {
            return Err("empty prompt: nothing to generate from".to_string());
        }
        // A prior graph failure is terminal for that sequence but must not
        // poison the next independent request.  Keep this flag separate from
        // the externally-owned cooperative cancel bit.
        self.graph_failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        // A mask that forbids nothing still costs every fused path and
        // whole-token graph, all of which are gated on `is_none()`. A
        // narrowed file whose one segment is always on carries exactly
        // such a mask — drop it here rather than pay 5x for a no-op.
        let task_mask = self.drop_open_mask(task_mask);

        // Cross-turn KV reuse: a chat app resends the whole history
        // every turn; when the new ids strictly EXTEND what the cache
        // already holds, prefill only the tail — turn latency stays
        // proportional to the new text instead of the whole session.
        // Extension-only (no rollback), so it is exact for every layer
        // kind including recurrent state; MTP/o1/task-mask runs keep
        // the fresh-sequence path. CMF_KV_REUSE=0 disables.
        let reuse_from = {
            let on = !std::env::var("CMF_KV_REUSE").is_ok_and(|v| v == "0");
            let h = &self.kv_history;
            if on
                && task_mask.is_none()
                && self.mtp.is_none()
                && self.o1_cfg.is_none()
                && self.dsv41.is_none()
                && !h.is_empty()
                && h.len() < input_ids.len()
                && input_ids[..h.len()] == h[..]
            {
                h.len()
            } else {
                0
            }
        };
        if reuse_from == 0 {
            // Fresh sequence — the cache holds absolute positions.
            self.clear_sequence_state();
        } else if std::env::var("CMF_PREFILL_PROF").is_ok() {
            eprintln!(
                "kv-reuse: {} of {} prompt positions already cached",
                reuse_from,
                input_ids.len()
            );
        }
        crate::gpu::graph_race_begin_generation();
        // Optional bounded calibration prefix. Keep the requested value
        // even when it is longer than the prompt; the collecting layer will
        // defer at the effective boundary and remain exact for short input.
        let o1_prefill = if self.o1_active() && task_mask.is_none() {
            std::env::var("CMF_O1_PREFILL")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&p| p > 0)
        } else {
            None
        };
        if task_mask.is_none() {
            self.o1_begin_with_prefix(o1_prefill);
        }

        // Speculative decode is off under o1: a rejected draft can't be
        // rolled back out of the far accumulators / ring window (the
        // Nyström insertion is irreversible by design).
        // The wgpu token graph owns a device K/V mirror that speculative
        // rollback would desync — the two are mutually exclusive.
        let graph_on = crate::gpu::wgpu_graph_on(crate::gpu::GraphPhase::Decode);
        // Graph speculative decode (`CMF_GRAPH_SPEC=1`): the MTP head
        // drafts, ONE batched graph submit verifies the whole chain.
        //
        // It now PAYS on Qwen3.6-27B / RTX 5090 — 51.1 tok/s against a
        // plain 49.4 at k=3, medians of three, 89% of drafts accepted,
        // and the greedy continuation is byte-identical to the plain
        // path. That took the batch matvec sharing its nibble unpack
        // across the batch (`CMF_MV_BK=2`); before it, the same round
        // measured 43.6, an 11% LOSS, which is what the earlier note
        // here described.
        //
        // Still opt-in. One model's win is not a default: the verify
        // rides `gdn_spec_restore` and a batched frame whose numerics
        // are the batch kernels', and that has to be shown on more than
        // one architecture before every greedy decode takes it.
        // Greedy (with or without penalties) verifies by argmax equality.
        // Sampling (temperature > 0) can go through speculative SAMPLING —
        // draft from the MTP head's own post-chain distribution, accept
        // with min(1, p/q), correct from max(0, p − q); the emitted stream
        // is distributed exactly as the plain sampler's — but it is
        // OPT-IN (`CMF_GRAPH_SPEC_SAMPLE=1`): measured on Qwen3.8-27B /
        // RTX 5090 at the instruct row (0.7 / 0.80 / 20 / presence 1.5)
        // it decoded 19-22 tok/s against a plain 40 — nine post-chain
        // distributions a round plus a lower acceptance than greedy's,
        // against a verify that costs 2.7 single tokens. The greedy arms
        // pay +10%; the sampling arm needs a cheaper verify first.
        // Native Metal HAS that verify: its eight-row tile is flat in b,
        // so a round costs ~1.9 plain tokens and the sampling arm pays at
        // 2.3 accepted per round — measured on Qwen3.8-27B q4tp / M4 at
        // the CLI defaults (0.7 / rep 1.1 / top-k 40, seed 42), a code
        // prompt: 9.0 tok/s against a plain 5.4 in the same window, and
        // the per-round watchdog turns it off where prose loses. So on
        // Metal the sampling arm is ON (`CMF_GRAPH_SPEC_SAMPLE=0` opts out)
        // — but only for a config the SPARSE chain serves (a top-k within
        // `sparse_ok`): without it a round builds nine 248k-float
        // distributions on the host, which is the 5090's measured loss and
        // not a cost the round-token proxy below can see. A top-k-less
        // sampling config keeps the plain path unless asked for by name.
        #[cfg(target_os = "macos")]
        let metal_graph = crate::gpu::q1_force()
            && crate::gpu::enabled_here()
            && std::env::var("CMF_GPU_BLOCK")
                .map(|v| v != "0")
                .unwrap_or(true);
        #[cfg(not(target_os = "macos"))]
        let metal_graph = false;
        let spec_sample_env = std::env::var("CMF_GRAPH_SPEC_SAMPLE").ok();
        // A round whose cost is the MEASURED one: greedy (argmax rows), or
        // sampling through the sparse chain. Anything else pays the dense
        // chain's host time, which no proxy can price.
        let spec_cheap_round = self.sampler_config.temperature < 1e-6
            || sampler::sparse_ok(&self.sampler_config);
        let spec_sampling_ok = self.sampler_config.temperature < 1e-6
            || match spec_sample_env.as_deref() {
                Some("1") => true,
                Some(_) => false,
                None => metal_graph && spec_cheap_round,
            };
        // ON by default for greedy on the wgpu graph: with the draft on
        // the graph and the verify bit-exact, it measured 58.7 tok/s
        // against a plain 48.1 on Qwen3.8-27B q4tp / RTX 5090 (k=4) and
        // 51.1 against 49.4 on Qwen3.6-27B, and a round that stops
        // paying turns itself off below (acceptance watchdog).
        // `CMF_GRAPH_SPEC=0` disables; `=1` was the old opt-in spelling.
        // …but only where the batched verify has its register-blocked
        // kernel: q4tp dense FFNs (graph kind 6). q4t and q8_2f verify
        // through tile GEMMs today and measured a LOSS (q8_2f 22 against
        // 29 tok/s), the 2-bit plane the same; those stay opt-in
        // (`CMF_GRAPH_SPEC=1`).
        // …at least in nine dense FFNs of ten: a healed file carries its
        // last two layers at q8_2f, and two tile-GEMM verifies among 64 do
        // not change the arithmetic (measured: the healed q4tp file
        // decodes at the plain file's rate and would otherwise sit out).
        let (mut dense_n, mut dense_q4tp) = (0usize, 0usize);
        for lw in &self.weights.layers {
            if let FfnKind::Dense(d) = &lw.ffn {
                dense_n += 1;
                if matches!(d.gate_proj.graph_weight(), Some((_, _, 6, _)))
                    && matches!(d.up_proj.graph_weight(), Some((_, _, 6, _)))
                    && matches!(d.down_proj.graph_weight(), Some((_, _, 6, _)))
                {
                    dense_q4tp += 1;
                }
            }
        }
        let spec_default_ok = dense_n == 0 || dense_q4tp * 10 >= dense_n * 9;
        // Penalties break the draft head's agreement with the trunk (a
        // 1.1 repetition penalty measured 2 of 16 accepted): not by
        // default there either — off Metal that rule is untouched, and
        // suppressed ids keep counting as a penalty there, because no
        // measurement on a discrete card says otherwise.
        //
        // On native Metal the penalized arms DO pay: the draft applies
        // the same penalty and the verify scores the penalized rows
        // exactly (`greedy_pen`, the plain loop's arithmetic), so the
        // text is the plain path's and only the round's shape changes.
        // Measured on this M4 — see the report for the interleaved run.
        let penalized = !metal_graph
            && (self.sampler_config.repetition_penalty != 1.0
                || self.sampler_config.presence_penalty != 0.0
                || !self.sampler_config.suppress_tokens.is_empty());
        // …and not on wgpu-over-Metal: the batched verify graph there
        // returned 0 accepted drafts and garbage text on a GDN hybrid
        // (16.08, Qwen3.5-0.8B) while Vulkan is bit-exact; the Mac's
        // default backend is native Metal without a batch graph anyway.
        #[cfg(feature = "gpu")]
        let metal_wgpu = graph_on && crate::gpu_wgpu::wgpu_backend_is_metal();
        #[cfg(not(feature = "gpu"))]
        let metal_wgpu = false;
        let spec_env = std::env::var("CMF_GRAPH_SPEC").ok();
        let spec_wanted = match spec_env.as_deref() {
            Some("0") => false,
            Some(_) => {
                if metal_wgpu {
                    tracing::warn!(
                        "CMF_GRAPH_SPEC forced on wgpu/Metal: the batched verify graph is not \
                         verified on this backend (garbage measured on Qwen3.5-0.8B)"
                    );
                }
                true
            }
            None => spec_default_ok && !penalized && !metal_wgpu,
        };
        // Native Metal: the b-row verify graph (`try_batch_graph_metal`)
        // stands where the wgpu batch graph stands on discrete cards
        // (`metal_graph`, above).
        let graph_spec = self.speculative
            && (graph_on || metal_graph)
            && self.mtp.is_some()
            && task_mask.is_none()
            && !self.o1_active()
            && spec_sampling_ok
            && spec_wanted;
        // Native Metal: say the route ONCE (RUST_LOG=info), so a user can
        // confirm the fast path without setting a single flag — every
        // knob below defaults to the measured-best value on the M4.
        #[cfg(target_os = "macos")]
        if metal_graph {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                let spec = if graph_spec {
                    let k = std::env::var("CMF_GRAPH_SPEC_K")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| (1..=8).contains(&v))
                        .unwrap_or(7);
                    let arm = if self.sampler_config.temperature < 1e-6 {
                        "greedy"
                    } else {
                        "sampling"
                    };
                    format!(
                        "spec k={k} {arm} (batched verify, draft shortlist {}, trial: proxy)",
                        Self::draft_vocab_rows(usize::MAX)
                    )
                } else if !self.speculative {
                    "spec off (CMF_MTP=0)".to_string()
                } else if self.mtp.is_none() {
                    "spec off (no MTP head)".to_string()
                } else if !spec_sampling_ok {
                    if spec_cheap_round {
                        "spec off (CMF_GRAPH_SPEC_SAMPLE=0)".to_string()
                    } else {
                        "spec off (sampling without a top-k: the dense chain \
                         costs more than it saves)"
                            .to_string()
                    }
                } else if !spec_wanted {
                    "spec off (CMF_GRAPH_SPEC=0 or non-q4tp FFNs)".to_string()
                } else if task_mask.is_some() {
                    "spec off (task mask)".to_string()
                } else {
                    "spec off (O(1) attention)".to_string()
                };
                let on = |var: &str| {
                    if std::env::var(var).as_deref() == Ok("0") {
                        "off"
                    } else {
                        "on"
                    }
                };
                tracing::info!(
                    "metal native: {spec}, state4 {}, async replay {}, prefill graph {}, \
                     MTP graph {}, attend {}, probe {}",
                    if crate::gpu_metal::state4_on() { "on" } else { "off" },
                    if crate::gpu_metal::async_replay_on() { "on" } else { "off" },
                    on("CMF_METAL_PREFILL"),
                    on("CMF_MTP_GRAPH"),
                    std::env::var("CMF_GPU_ATTEND").unwrap_or_else(|_| "auto".into()),
                    if crate::gpu::probe_enabled() { "bypassed (q1 force)" } else { "off" },
                );
            });
        }
        // GDN hybrids sit the fused-pair speculation out by default: the
        // recurrence is sequential, so the pair lane cannot parallelize
        // (the bench's own Pair line reads fused 1.28x TWO singles on the
        // 35B) and the draft's full-vocab head rides on top — measured 2x
        // SLOWER end to end (16.1 vs 32.4 tok/s on the 48-core stand).
        // CMF_MTP=1 forces it back for study.
        let pair_pays = self.gdn_cfg.is_none() || std::env::var("CMF_MTP").as_deref() == Ok("1");
        let spec_active = self.speculative
            && self.mtp.is_some()
            && task_mask.is_none()
            && !self.o1_active()
            && ((!graph_on && pair_pays && self.sampler_config.temperature < 1e-6) || graph_spec);
        // The MTP module is detached during generation so its mutable
        // state does not fight the borrow on `self`.
        let mut mtp = if spec_active { self.mtp.take() } else { None };
        if std::env::var("CMF_MTP_CHAIN_PROBE").is_ok() {
            eprintln!(
                "mtp-probe gate: spec_active={spec_active} mtp={} speculative={} graph_on={graph_on} temp_ok={}",
                mtp.is_some(),
                self.speculative,
                self.sampler_config.temperature < 1e-6,
            );
        }
        if let Some(m) = &mut mtp {
            m.kv.clear();
            // The MTP block's own device mirror starts over with its cache.
            crate::gpu::graph_kv_reset(self.mtp_kv_id());
            self.mtp_graph_mode = None;
        }
        // Dynamic router detached during decode (same borrow trick as MTP).
        // Speculative decode and dynamic routing are mutually exclusive
        // for now — the fused-pair path doesn't carry per-token φ.
        let mut router = if mtp.is_none() {
            self.dyn_router.take()
        } else {
            None
        };
        if let Some(r) = &mut router {
            r.reset(); // active=backbone, matching a fresh overlay
            self.dyn_phi_seen = 0; // fresh φ EMA per generation
            let _ = self.set_active_skill(None);
        }

        let mut all_ids = input_ids.to_vec();
        let mut generated = 0usize;
        let mut finish_reason = "max_tokens".to_string();
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        // DeepSeek-V4's draft quality is strongly content-dependent.  Two
        // consecutive paid rounds with no extra token put it on a bounded
        // cooldown; predictable text keeps batching, ordinary prose falls
        // back to the exact walk instead of paying a slow draft forever.
        // Local to one generation so one difficult request cannot poison the
        // next one, and deliberately automatic — this is not a user knob.
        let mut dsv4_spec_bad = 0usize;
        let mut dsv4_spec_retry_at = 0usize;
        let mut confidence: Vec<f32> = Vec::new();
        let trace_on = self.trace;
        let calib_temp = self.calib_temp;
        let mut traces: Vec<TokenTrace> = Vec::new();

        // ── Prefill: forward each prompt token once, KEEP the last hidden.
        //    Dense prefill runs in fused pairs (weights streamed once per
        //    two positions — bit-identical to sequential, proven by the
        //    pair tests). With MTP: warm the draft head on
        //    (hidden_p, token_{p+1}) pairs.
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut pos = reuse_from;
        // lm_head-in-graph is only sound when the very next logits
        // consumer is this loop's own (MTP and skill routing interleave
        // other forwards / can swap lm_head between forward and sample).
        // CMF_GPU_LMHEAD=0 keeps lm_head off the graph: the token reads back
        // the 8 KB hidden instead of ~1 MB of logits, and the head runs on
        // the host. A probe for how much of the graph's fixed per-token cost
        // is the logits readback (the layer sweep puts that fixed part at
        // 3.88 ms of an 18.5 ms frame).
        let fuse_lm = mtp.is_none()
            && router.is_none()
            && std::env::var("CMF_GPU_LMHEAD").as_deref() != Ok("0");
        self.graph_logits = None;
        self.graph_want_logits = false;
        let _tpf = std::time::Instant::now();
        let batch_k = self.generation_batch_k();
        // DeepSeek-V4 owns a separate hyper-connection stack. Route it
        // before the generic prefill choices: those correctly reject an
        // empty `weights.layers`, but their final per-position fallback used
        // to consume the whole prompt before `dsv4::forward_chunk` could see
        // it. The batch implementation therefore existed without a live
        // production entry point.
        //
        // Bounded chunks preserve cancellation responsiveness. Only the
        // prompt's final chunk asks for logits; every earlier head projection
        // would produce 129 280 values that no caller reads.
        while self.qwen4_exp.is_some()
            && mtp.is_none()
            && pos < input_ids.len()
            && !self.cancel.load(std::sync::atomic::Ordering::Relaxed)
        {
            let token_id = input_ids[pos];
            let want_logits = pos + 1 == input_ids.len();
            let mut lg = Vec::new();
            if let Some(b) = &mut self.qwen4_exp {
                crate::qwen4_exp::forward_token(
                    &b.0,
                    &b.1,
                    &b.2,
                    &mut b.3,
                    token_id,
                    pos,
                    &self.inv_freq,
                    self.pool.as_deref(),
                    &mut lg,
                    want_logits,
                );
            }
            if want_logits {
                self.graph_logits = Some(lg);
            }
            pos += 1;
            hidden.fill(0.0);
        }
        while self.dsv4.is_some()
            && mtp.is_none()
            && pos < input_ids.len()
            && !self.cancel.load(std::sync::atomic::Ordering::Relaxed)
        {
            let end = (pos + prefill_chunk()).min(input_ids.len());
            let ids: Vec<u32> = input_ids[pos..end].to_vec();
            let mut lg = Vec::new();
            if let Some(b) = &mut self.dsv4 {
                let (g, layers, cfg, st) = (&b.0, &b.1, b.2, &mut b.3);
                crate::dsv4::forward_chunk(
                    g,
                    layers,
                    &cfg,
                    st,
                    &ids,
                    pos,
                    &self.inv_freq,
                    self.pool.as_deref(),
                    &mut lg,
                    end == input_ids.len(),
                );
            }
            if end == input_ids.len() {
                self.graph_logits = Some(lg);
            }
            pos = end;
            hidden = vec![0.0; self.hidden_size];
        }
        let dsv41_prefill = self.dsv41_prefill.take();
        while self.dsv41.is_some()
            && mtp.is_none()
            && pos < input_ids.len()
            && !self.cancel.load(std::sync::atomic::Ordering::Relaxed)
        {
            let end = (pos + prefill_chunk()).min(input_ids.len());
            let ids: Vec<u32> = input_ids[pos..end].to_vec();
            let mut lg = Vec::new();
            if let Some(b) = &mut self.dsv41 {
                let (g, layers, cfg, st) = (&b.0, &b.1, &b.2, &mut b.3);
                if let Some((embeddings, participates)) = dsv41_prefill.as_ref() {
                    crate::dsv41::forward_chunk_masked_with_embeddings(
                        g,
                        layers,
                        cfg,
                        st,
                        &ids,
                        pos,
                        &embeddings[pos..end],
                        &participates[pos..end],
                        self.pool.as_deref(),
                        &mut lg,
                    );
                } else {
                    crate::dsv41::forward_chunk(
                        g,
                        layers,
                        cfg,
                        st,
                        &ids,
                        pos,
                        self.pool.as_deref(),
                        &mut lg,
                    );
                }
            }
            if end == input_ids.len() {
                self.graph_logits = Some(lg);
            }
            pos = end;
            hidden = vec![0.0; self.hidden_size];
        }
        // With dynamic routing, prefill sequentially so the φ hook fires
        // over the PROMPT — the router enters decode with a warm φ (the
        // fused-pair path skips the per-layer φ capture). o1 layers
        // collect their query trace in both the single and pair paths.
        let dyn_prefill = router.is_some();
        // Optional bounded calibration prefix for generation.  The normal
        // O(1) path seals after the full prompt; this explicit knob instead
        // runs only the requested prefix through exact attention, seals the
        // Nyström state, and streams the rest of the prompt through the same
        // O(1) step used by decode.  It keeps the O(1) layers' Q trace and
        // temporary full KV bounded by the prefix while leaving the default
        // full-prompt quality profile untouched.
        let o1_prefill_limit = o1_prefill
            .and_then(|requested| self.o1_effective_boundary(requested))
            .map(|boundary| boundary.min(input_ids.len()));
        let mut o1_sealed = false;
        if let Some(limit) = o1_prefill_limit {
            // Reuse the exact batched prefix machinery when available; it
            // records the same per-position Q trace as the full prefill.
            if self.can_prefill_batched() && limit > 2 {
                let chunk = self.prefill_chunk();
                let hs = self.hidden_size;
                while pos < limit && !self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    let end = (pos + chunk).min(limit);
                    let hb = self.prefill_batch(&input_ids[pos..end], pos);
                    hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                    pos = end;
                }
            } else {
                while pos < limit && !self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    hidden = self.forward_layers(&self.embed_single(input_ids[pos]), pos, None);
                    pos += 1;
                }
            }
            if pos >= limit {
                o1_sealed = match self.o1_seal_checked() {
                    Ok(sealed) => sealed,
                    Err(err) => {
                        self.finish_generation(&mut mtp, &mut router, true);
                        return Err(err);
                    }
                };
                tracing::info!(
                    "o1 bounded prompt prefix: requested={} effective={} processed={} of {} token(s)",
                    o1_prefill.unwrap_or(0),
                    self.o1_effective_boundary(o1_prefill.unwrap_or(0))
                        .unwrap_or(limit),
                    limit,
                    input_ids.len()
                );
            }
        }
        // q1 hybrids on Metal: the per-position GPU token graph beats
        // the CPU chunk-GEMM (whose wall is the sequential scalar GDN
        // recurrence), so prefill goes position-by-position through the
        // same graph as decode. Pure-attention models keep the batched
        // path — there the chunk-GEMM amortization wins.
        let graph_prefill = self.graph_prefill_preferred();
        // Native Metal, q4tp GDN hybrids: the prompt through the b-row
        // rows graph — projections as GEMMs over up to 512 positions, the
        // GDN recurrence in registers on the device, K/V rows appended by
        // the chunk — instead of one token-graph submit per position (the
        // 27B: 8 tok/s → GEMM-bound). The MTP warm-up rows come out of one
        // batched run of the block per chunk. Any refusal leaves the rest
        // of the prompt to the sequential paths below.
        #[cfg(target_os = "macos")]
        if task_mask.is_none()
            && !dyn_prefill
            && (crate::gpu::q1_force() || crate::gpu::q2tp_gpu_opt_in())
            && crate::gpu::enabled_here()
            && self.gdn_cfg.is_some()
            && self.g3n.is_none()
            && input_ids.len() > 8
            && std::env::var("CMF_MTP_CHAIN_PROBE").is_err()
            && std::env::var("CMF_METAL_PREFILL").as_deref() != Ok("0")
        {
            let chunk: usize = std::env::var("CMF_METAL_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| (16..=512).contains(&v))
                .unwrap_or(256);
            let hs = self.hidden_size;
            let _tp = std::time::Instant::now();
            while pos < input_ids.len() && !self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let end = (pos + chunk).min(input_ids.len());
                let hb = match self.prefill_batch_metal(&input_ids[pos..end], pos) {
                    MetalPrefillOutcome::Completed(hb) => hb,
                    MetalPrefillOutcome::Declined => break,
                    MetalPrefillOutcome::Failed => {
                        self.finish_generation(&mut mtp, &mut router, true);
                        return Err("ordinary Metal prefill failed after admission".into());
                    }
                };
                if let Some(m) = &mut mtp {
                    let n_pairs = if end < input_ids.len() {
                        end - pos
                    } else {
                        end - pos - 1
                    };
                    if n_pairs > 0 {
                        let pairs: Vec<(&[f32], u32)> = (0..n_pairs)
                            .map(|j| (&hb[j * hs..(j + 1) * hs], input_ids[pos + j + 1]))
                            .collect();
                        if !self.mtp_warm_batch_metal(m, &pairs, pos) {
                            for (j, (h, t)) in pairs.iter().enumerate() {
                                let h = h.to_vec();
                                let _ = self.mtp_step(m, &h, *t, pos + j);
                            }
                        }
                    }
                }
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
            if std::env::var("CMF_PREFILL_PROF").is_ok() {
                eprintln!(
                    "metal-prefill: {} of {} tokens in {:.1} ms",
                    pos,
                    input_ids.len(),
                    _tp.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        if task_mask.is_none()
            && !dyn_prefill
            && !graph_prefill
            && self.can_prefill_batched()
            && self.g3n.is_none()
            && o1_prefill.is_none()
            && input_ids.len() > 2
        {
            // Production prefill = the same chunked prefill-GEMM that
            // bench/PPL measure (roadmap §3 P0: generation used to warm
            // the prompt with the slower pair path — the published
            // prefill number didn't match real TTFT). MTP warm-up reads
            // each position's hidden straight from the chunk result.
            let chunk = self.prefill_chunk();
            let hs = self.hidden_size;
            while pos < input_ids.len() && !self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let end = (pos + chunk).min(input_ids.len());
                let hb = self.prefill_batch(&input_ids[pos..end], pos);
                if let Some(m) = &mut mtp {
                    let probe: usize = std::env::var("CMF_MTP_CHAIN_PROBE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    for p in pos..end {
                        if p + 1 < input_ids.len() {
                            if probe >= 1 && p + 2 < input_ids.len() {
                                // Teacher-forced chain acceptance (see the
                                // tail loop's twin): the warm-up row stays,
                                // the chain's rows roll back.
                                let (d1, mut hx) = self.mtp_step_h(
                                    m,
                                    &hb[(p - pos) * hs..(p - pos + 1) * hs],
                                    input_ids[p + 1],
                                    p,
                                );
                                let mut ok = d1 == input_ids[p + 2];
                                Self::chain_probe_note(0, ok);
                                let mut d_prev = d1;
                                let mut extra = 0usize;
                                for j in 1..probe {
                                    if p + 2 + j >= input_ids.len() {
                                        break;
                                    }
                                    let (dj, hj) = self.mtp_step_h(m, &hx, d_prev, p + 1 + j);
                                    extra += 1;
                                    ok = ok && dj == input_ids[p + 2 + j];
                                    Self::chain_probe_note(j, ok);
                                    d_prev = dj;
                                    hx = hj;
                                }
                                m.kv.truncate_last(extra);
                            } else {
                                let _ = self.mtp_step(
                                    m,
                                    &hb[(p - pos) * hs..(p - pos + 1) * hs],
                                    input_ids[p + 1],
                                    p,
                                );
                            }
                        }
                    }
                }
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
        }
        let pair_off = std::env::var("CMF_PAIR").is_ok_and(|v| v == "0");
        if task_mask.is_none()
            && !dyn_prefill
            && !graph_prefill
            && !pair_off
            && self.pair_supported()
            && o1_prefill.is_none()
        {
            while pos + 1 < input_ids.len()
                && !self.cancel.load(std::sync::atomic::Ordering::Relaxed)
            {
                let e1 = self.embed_single(input_ids[pos]);
                let e2 = self.embed_single(input_ids[pos + 1]);
                let (h1, h2) = self.forward_pair(&e1, &e2, pos);
                // Both prefill tokens are real → commit lane-2 states.
                self.commit_linear_scratch();
                if let Some(m) = &mut mtp {
                    let _ = self.mtp_step(m, &h1, input_ids[pos + 1], pos);
                    if pos + 2 < input_ids.len() {
                        let probe: usize = std::env::var("CMF_MTP_CHAIN_PROBE")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        if probe >= 1 && pos + 3 < input_ids.len() {
                            // Same teacher-forced chain table as the tail
                            // loop below, fed from the pair path that owns
                            // most prefill positions.
                            let (d1, mut hx) = self.mtp_step_h(m, &h2, input_ids[pos + 2], pos + 1);
                            let mut ok = d1 == input_ids[pos + 3];
                            Self::chain_probe_note(0, ok);
                            let mut d_prev = d1;
                            let mut extra = 0usize;
                            for j in 1..probe {
                                if pos + 3 + j >= input_ids.len() {
                                    break;
                                }
                                let (dj, hj) = self.mtp_step_h(m, &hx, d_prev, pos + 2 + j);
                                extra += 1;
                                ok = ok && dj == input_ids[pos + 3 + j];
                                Self::chain_probe_note(j, ok);
                                d_prev = dj;
                                hx = hj;
                            }
                            m.kv.truncate_last(extra);
                        } else {
                            let _ = self.mtp_step(m, &h2, input_ids[pos + 2], pos + 1);
                        }
                    }
                }
                hidden = h2;
                pos += 2;
            }
        }
        // Batched GPU prefill for the wgpu decode graph (GDN hybrids): K prompt
        // positions per submit — projections/FFN as GEMMs (weight once per K),
        // attention/GDN looped inside — instead of one whole-graph submit per
        // position. Falls through to the per-position graph on any refusal.
        // Batched prefill is opt-in (CMF_BATCH_K>0). Default 0 = per-position
        // graph prefill. (Steady-state decode is provably identical either way —
        // token-graph submit and lm_head both unchanged — so this only trades
        // prefill wall.)
        // A bounded O(1) prefix is the one post-seal prompt interval: only
        // admit its batch when the device O(1) route is explicitly enabled and
        // every sealed layer exposes a portable view. The same batch size and
        // refusal behavior remain the ordinary controls/comparator.
        let o1_batch_ready = o1_sealed
            && o1_prefill.is_some()
            && mtp.is_none()
            && std::env::var("CMF_O1_GPU").as_deref() == Ok("1")
            && (0..self.num_layers).all(|li| {
                let cache = &self.kv_cache.layers[self.phys_layer(li)];
                cache.o1.is_none() || cache.o1_views().is_some()
            });
        // The ordinary graph-prefill route can share each completed trunk
        // chunk with an attached MTP head.  Keep chain probing on its
        // established per-position path: the probe deliberately needs every
        // teacher-forced draft row and its rollback table.
        let mtp_batch_prefill = mtp.is_some()
            && graph_prefill
            && task_mask.is_none()
            && !dyn_prefill
            && !self.o1_active()
            && std::env::var("CMF_MTP_CHAIN_PROBE").is_err();
        if batch_k > 0
            && (graph_prefill || o1_batch_ready)
            && task_mask.is_none()
            && (!self.o1_active() || o1_batch_ready)
            && (mtp.is_none() || mtp_batch_prefill)
            && !dyn_prefill
            && pos + 1 < input_ids.len()
        {
            let hs = self.hidden_size;
            let chunk = batch_k;
            while pos < input_ids.len() {
                let end = (pos + chunk).min(input_ids.len());
                let bk = end - pos;
                let mut hiddens = vec![0f32; bk * hs];
                for (j, &id) in input_ids[pos..end].iter().enumerate() {
                    hiddens[j * hs..(j + 1) * hs].copy_from_slice(&self.embed_single(id));
                }
                let positions: Vec<usize> = (pos..end).collect();
                let t_chunk = std::time::Instant::now();
                let outcome = self.try_batch_graph_wgpu(&mut hiddens, &positions, bk, None);
                let ok_b = outcome == crate::gpu::BatchGraphOutcome::Completed;
                if std::env::var("CMF_GRAPH_PROF").is_ok() {
                    let ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "batch-chunk: phase=prompt mode={} k={bk} outcome={outcome:?} {ms:.1} ms ({:.1} tok/s)",
                        if o1_batch_ready {
                            "o1"
                        } else if mtp_batch_prefill {
                            "ordinary_mtp"
                        } else {
                            "ordinary"
                        },
                        bk as f64 / (ms / 1000.0)
                    );
                }
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static SAID: AtomicBool = AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        if ok_b {
                            tracing::info!(
                                "batched prefill: ACTIVE mode={} (k={bk})",
                                if o1_batch_ready {
                                    "o1"
                                } else if mtp_batch_prefill {
                                    "ordinary_mtp"
                                } else {
                                    "ordinary"
                                }
                            );
                        } else {
                            tracing::warn!("batched prefill {:?} — per-position graph", outcome);
                        }
                    }
                }
                if ok_b {
                    if mtp_batch_prefill {
                        let n_pairs = mtp_prefill_pair_count(pos, end, input_ids.len());
                        if n_pairs > 0 {
                            // `hiddens` is owned by this chunk, so materialize
                            // row slices before borrowing the detached MTP
                            // module.  The last prompt row has no successor;
                            // the helper above is the single source of that
                            // boundary rule.
                            let rows: Vec<Vec<f32>> = (0..n_pairs)
                                .map(|j| hiddens[j * hs..(j + 1) * hs].to_vec())
                                .collect();
                            let pairs: Vec<(&[f32], u32)> = rows
                                .iter()
                                .enumerate()
                                .map(|(j, row)| (row.as_slice(), input_ids[pos + j + 1]))
                                .collect();
                            if std::env::var("CMF_GRAPH_PROF").is_ok() {
                                eprintln!(
                                    "mtp-warm: phase=prompt mode=ordinary_mtp first_pos={} pairs={} last_pos={}",
                                    pos,
                                    n_pairs,
                                    pos + n_pairs - 1,
                                );
                            }
                            let warm_error = if let Some(m) = mtp.as_mut() {
                                self.mtp_warm_prefill_pairs(m, &pairs, pos).err()
                            } else {
                                None
                            };
                            if let Some(err) = warm_error {
                                // The trunk batch was already admitted.  A
                                // failed MTP warm-up therefore clears both
                                // mirrors and exits; continuing would pair a
                                // current trunk state with a stale MTP cache.
                                self.finish_generation(&mut mtp, &mut router, true);
                                return Err(err.to_string());
                            }
                        }
                    }
                    hidden.copy_from_slice(&hiddens[(bk - 1) * hs..]);
                    pos = end;
                } else if outcome == crate::gpu::BatchGraphOutcome::Failed {
                    // A failed batch may have advanced a device recurrent
                    // state (ordinary GDN or sealed O(1)). A CPU fallback
                    // would then observe stale accumulators, so clear the
                    // request state and make the failure explicit.
                    self.finish_generation(&mut mtp, &mut router, true);
                    return Err(if o1_batch_ready {
                        "sealed O(1) batch graph failed after admission".to_string()
                    } else {
                        "ordinary recurrent batch graph failed after admission".to_string()
                    });
                } else {
                    break; // unsupported → per-position graph handles the rest
                }
            }
        }
        while pos < input_ids.len() && !self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            self.graph_want_logits = fuse_lm && pos + 1 == input_ids.len();
            hidden = self.forward_layers(&self.embed_single(input_ids[pos]), pos, task_mask);
            if let Some(m) = &mut mtp {
                if pos + 1 < input_ids.len() {
                    // `CMF_MTP_CHAIN_PROBE=k`: teacher-forced acceptance of a
                    // CHAINED draft — iterate the head on its own hidden k
                    // deep and score every depth against the prompt's real
                    // continuation. The economics of a k-token speculative
                    // round stand or fall on this table.
                    let probe: usize = std::env::var("CMF_MTP_CHAIN_PROBE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if probe >= 1 && pos + 2 < input_ids.len() {
                        let (d1, mut hx) = self.mtp_step_h(m, &hidden, input_ids[pos + 1], pos);
                        let mut ok = d1 == input_ids[pos + 2];
                        Self::chain_probe_note(0, ok);
                        let mut d_prev = d1;
                        let mut extra = 0usize;
                        for j in 1..probe {
                            if pos + 2 + j >= input_ids.len() {
                                break;
                            }
                            let (dj, hj) = self.mtp_step_h(m, &hx, d_prev, pos + 1 + j);
                            extra += 1;
                            ok = ok && dj == input_ids[pos + 2 + j];
                            Self::chain_probe_note(j, ok);
                            d_prev = dj;
                            hx = hj;
                        }
                        // The chain's rows are speculation, not the prompt —
                        // keep only the warmup row the plain path would add.
                        m.kv.truncate_last(extra);
                    } else {
                        let _ = self.mtp_step(m, &hidden, input_ids[pos + 1], pos);
                    }
                }
            }
            pos += 1;
        }
        if std::env::var("CMF_PREFILL_PROF").is_ok() {
            eprintln!(
                "prefill: {} tokens in {:.1} ms (batch_k={batch_k})",
                input_ids.len(),
                _tpf.elapsed().as_secs_f64() * 1000.0
            );
        }
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            // MTP is detached for speculative generation.  Restore the
            // module before returning the terminal graph error; otherwise a
            // failed request would silently remove the head from a pooled
            // pipeline and the next request would lose its configured route.
            self.finish_generation(&mut mtp, &mut router, true);
            return Err("GPU token graph failed during prefill".to_string());
        }
        // Cancelled mid-prefill: the cache holds a partial prompt —
        // drop the reuse history and return an empty generation.
        if self
            .cancel
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            // A cancelled prefill can already have advanced the device
            // mirror. Drop the whole partial sequence so a pooled pipeline
            // cannot carry that state into its next request.
            self.finish_generation(&mut mtp, &mut router, true);
            return Ok(GenerateResult {
                text: String::new(),
                token_ids: Vec::new(),
                prompt_tokens: input_ids.len(),
                tokens_generated: 0,
                finish_reason: "cancelled".to_string(),
                mtp_drafted: 0,
                mtp_accepted: 0,
                token_confidence: Vec::new(),
                traces: Vec::new(),
            });
        }

        // Prompt absorbed → freeze the o1 layers' skeletons; from here
        // every decode step on those layers is O(W + m·dv + m²).
        if !o1_sealed {
            match self.o1_seal_checked() {
                Ok(_) => {}
                Err(err) => {
                    self.finish_generation(&mut mtp, &mut router, true);
                    return Err(err);
                }
            }
        }

        // Commit one token: push, check EOS, stream. Returns false = stop.
        macro_rules! commit {
            ($id:expr) => {{
                all_ids.push($id);
                generated += 1;
                self.note_draft_id($id);
                if self.tokenizer.is_eos($id) && !self.ignore_eos {
                    finish_reason = "stop".to_string();
                    false
                } else {
                    let token_text = self.tokenizer.decode_token($id);
                    let mut go = true;
                    if let Some(ref mut cb) = on_token {
                        if !cb(&token_text) {
                            finish_reason = "cancelled".to_string();
                            go = false;
                        }
                    }
                    go
                }
            }};
        }

        // Speculation is decided by MEASUREMENT, not by an acceptance
        // model. A k=4 round costs ~3.8 plain tokens on the 5090 (draft
        // 6.6 + verify 66.6 + commit 4.8 ms against a 20.6 ms token), so it
        // pays only when the head lands ~2.8 of 4 — predictable text (code,
        // structured output) does, free prose often does not, and the
        // ratio at which the two cross depends on the card and the context
        // depth. So: four speculative rounds timed, then eight plain
        // tokens timed, and the faster arm runs until a re-check 256
        // tokens later (context growth moves the balance). The trial
        // costs at most a few tokens of the slower arm per 256.
        let mut spec_trial = SpecTrial::Spec {
            t0: std::time::Instant::now(),
            gen0: generated,
            rounds: 0,
        };
        // The token-count proxy prices a round at ~1.9 plain tokens. That
        // holds for the Metal rounds whose cost was measured — greedy and
        // the sparse sampling chain — so an expensive round (the dense
        // chain, reachable only by `CMF_GRAPH_SPEC_SAMPLE=1`) still times
        // the plain path before it decides.
        let mut spec_mon = SpecMon {
            metal: graph_spec && crate::gpu::q1_force() && spec_cheap_round,
            ..SpecMon::default()
        };
        let mut spec_watchdog_off = false;
        // CMF_GRAPH_SPEC_TIME: the round walls so far (round 1 excluded —
        // it pays the scratch), for the outlier test on each new one
        let mut spec_walls: Vec<f32> = Vec::new();
        // ... and the end of the last round: the host time between rounds
        // (token commits, streaming, the loop top) is printed at level 2
        let mut spec_round_end: Option<std::time::Instant> = None;
        // ── Decode ──
        let mut next_pos = input_ids.len();
        'decode: while generated < max_tokens {
            if self
                .graph_failed
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                // Keep the detached MTP module attached after a terminal
                // graph error so the pipeline can be reused for a fresh
                // sequence.  `clear_sequence_state` only clears mirrors and
                // host KV; it cannot recover a module dropped here.
                self.finish_generation(&mut mtp, &mut router, true);
                return Err("GPU token graph failed during decode".to_string());
            }
            if self
                .cancel
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                finish_reason = "cancelled".to_string();
                break 'decode;
            }
            // A rejected speculative draft already drew this position's
            // token from the residual distribution (graph_spec_step); it
            // is committed as-is — sampling again from the row's logits
            // would bias the stream toward the target's mode.
            let forced = self.spec_forced.take();
            let mut logits = match (forced, self.graph_logits.take()) {
                (Some(_), _) => Vec::new(),
                (None, Some(lg)) => lg,
                (None, None) => {
                    inference::rms_norm_into(
                        &hidden,
                        &self.weights.final_norm,
                        self.rms_eps,
                        self.norm_style,
                        &mut self.ws.n1,
                    );
                    self.lm_head_forward(&self.ws.n1)
                }
            };
            // CMF_LOGIT_DUMP=<path>: the first decode step's hidden + logits
            // as raw f32 (hidden first) — cross-backend numerics diffing.
            if generated
                == std::env::var("CMF_LOGIT_DUMP_STEP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            {
                if let Ok(path) = std::env::var("CMF_LOGIT_DUMP") {
                    let mut bytes: Vec<u8> = Vec::with_capacity((hidden.len() + logits.len()) * 4);
                    for v in hidden.iter().chain(logits.iter()) {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    if let Err(e) = std::fs::write(&path, &bytes) {
                        eprintln!("logit dump: failed to write {path}: {e}");
                        self.finish_generation(&mut mtp, &mut router, true);
                        return Err(format!("logit dump write failed: {e}"));
                    }
                }
            }
            let t_next = match forced {
                Some(c) => c,
                None => sampler::sample_with_scratch_pool(
                    &logits,
                    &self.sampler_config,
                    &all_ids,
                    &mut self.rng,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                ),
            };
            if self.confidence_on {
                confidence.push(if logits.is_empty() {
                    0.0
                } else {
                    sampler::top1_prob_pool(
                        self.pool.as_deref(),
                        &mut self.sampler_scratch,
                        &logits,
                        t_next,
                        calib_temp,
                    )
                });
            }
            if !logits.is_empty() {
                attention::recycle_buf(&mut logits);
            }
            if trace_on {
                // active_skill = the overlay in force while this token was
                // generated; recon/switched are filled after the post-emit
                // routing eval below (freshest coherence for this token).
                let skill = router.as_ref().and_then(|r| r.active_id());
                traces.push(TokenTrace {
                    t: generated,
                    token_id: t_next,
                    confidence: confidence.last().copied().unwrap_or(0.0),
                    active_skill: skill,
                    recon: None,
                    switched: false,
                });
            }
            if !commit!(t_next) {
                break 'decode;
            }
            if generated >= max_tokens {
                break 'decode;
            }

            if self.dsv41.is_none() && self.kv_cache.needs_eviction() {
                // Say it ONCE, loudly: past this point the model keeps
                // talking but has lost half its context, and on a GDN
                // hybrid the graph's device state goes stale on top. The
                // Qwen3.8 bring-up spent a day reading this cliff as
                // three different model bugs.
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        "KV cache full at {} positions — evicting half; quality \
                         will degrade. Raise CMF_MAX_SEQ.",
                        self.kv_cache.max_seq_len,
                    );
                });
                let keep = (self.kv_cache.max_seq_len / 2).max(1);
                self.kv_cache.evict(keep);
            }

            // Advance the speculation trial: plain-phase accounting and
            // the periodic re-check happen here, on every token.
            if graph_spec {
                match spec_trial {
                    SpecTrial::Plain { t0, gen0 } if spec_mon.plain_done(t0, gen0, generated) => {
                        spec_mon.plain_ms =
                            t0.elapsed().as_secs_f64() * 1e3 / (generated - gen0) as f64;
                        let keep = spec_mon.pays();
                        tracing::info!(
                            "speculation trial: {:.2} tok/round in {:.1} ms vs plain {:.1} ms/tok — {}",
                            spec_mon.tokens,
                            spec_mon.round_ms,
                            spec_mon.plain_ms,
                            if keep { "speculating" } else { "plain" }
                        );
                        spec_mon.fails = 0;
                        spec_trial = SpecTrial::Decided {
                            spec: keep,
                            recheck_at: if keep { usize::MAX } else { generated + 128 },
                        };
                    }
                    SpecTrial::Decided { recheck_at, .. } if generated >= recheck_at => {
                        spec_mon.n = 0;
                        spec_trial = SpecTrial::Spec {
                            t0: std::time::Instant::now(),
                            gen0: generated,
                            rounds: 0,
                        };
                    }
                    _ => {}
                }
                spec_watchdog_off = matches!(
                    spec_trial,
                    SpecTrial::Plain { .. } | SpecTrial::Decided { spec: false, .. }
                );
            }
            match &mut mtp {
                // ── Graph speculation: chain-draft, batch-verify on device ──
                #[cfg(feature = "gpu")]
                Some(m)
                    if graph_spec
                        && !spec_watchdog_off
                        && generated + 1 < max_tokens
                        && next_pos > 0 =>
                {
                    let t_round = std::time::Instant::now();
                    if spec_time_level() >= 2 {
                        if let Some(t) = spec_round_end.take() {
                            eprintln!(
                                "spec-gap {:.2} ms (host between rounds)",
                                t.elapsed().as_secs_f64() * 1e3
                            );
                        }
                    }
                    spec_stamps_begin();
                    // device buffers allocated during this round: a
                    // first-touch Shared allocation is zero-filled inside
                    // the command buffer that uses it, which is what the
                    // long outlier rounds were
                    #[cfg(target_os = "macos")]
                    let allocs0 = crate::gpu_metal::IO_BUF_ALLOCS
                        .load(std::sync::atomic::Ordering::Relaxed);
                    #[cfg(not(target_os = "macos"))]
                    let allocs0 = 0u64;
                    if let Some((extra, n_pos, new_h)) = self.graph_spec_step(
                        m,
                        &hidden,
                        t_next,
                        next_pos,
                        &mut drafted,
                        &mut accepted,
                        &mut all_ids,
                        max_tokens - generated,
                    ) {
                        next_pos = n_pos;
                        hidden = new_h;
                        let level = spec_time_level();
                        if level > 0 {
                            let wall = t_round.elapsed().as_secs_f32() * 1e3;
                            let stamps = spec_stamps_take();
                            // the running median of the rounds before this
                            // one (round 1 pays the scratch: not a sample)
                            let median = if spec_walls.len() >= 3 {
                                let mut s = spec_walls.clone();
                                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                Some(s[s.len() / 2])
                            } else {
                                None
                            };
                            let outlier = median.is_some_and(|m| wall > 1.4 * m);
                            #[cfg(target_os = "macos")]
                            let allocs = crate::gpu_metal::IO_BUF_ALLOCS
                                .load(std::sync::atomic::Ordering::Relaxed)
                                - allocs0;
                            #[cfg(not(target_os = "macos"))]
                            let allocs = allocs0;
                            eprintln!(
                                "spec-round wall {wall:.1} ms → {} tokens{}{}",
                                extra.len() + 1,
                                if allocs > 0 {
                                    format!(" [{allocs} new device buffers]")
                                } else {
                                    String::new()
                                },
                                match (outlier, median) {
                                    (true, Some(m)) => format!(" OUTLIER (median {m:.1})"),
                                    _ => String::new(),
                                }
                            );
                            if level >= 2 || outlier {
                                let sum: f32 = stamps.iter().map(|s| s.1).sum();
                                eprintln!(
                                    "spec-stamps: {}| untracked {:.1}",
                                    spec_stamps_format(&stamps),
                                    wall - sum
                                );
                            }
                            if spec_mon.n >= 1 {
                                spec_walls.push(wall);
                            }
                        }
                        // One speculative round done: the monitor counts it
                        // (round 1 untimed — it pays the batch scratch and
                        // the draft mirror), and the trial advances.
                        spec_mon.round(t_round.elapsed().as_secs_f64() * 1e3, extra.len() + 1);
                        // the round's tokens land in `generated` below; the
                        // plain phase must start counting AFTER them
                        spec_trial = Self::spec_trial_round(
                            spec_trial,
                            &mut spec_mon,
                            generated + extra.len() + 1,
                        );
                        let mut stopped = false;
                        for &id in &extra {
                            if self.confidence_on {
                                confidence.push(0.0);
                            }
                            if !commit!(id) {
                                stopped = true;
                                break;
                            }
                        }
                        if stopped {
                            break 'decode;
                        }
                        if spec_time_level() >= 2 {
                            spec_round_end = Some(std::time::Instant::now());
                        }
                        continue 'decode;
                    }
                    if self
                        .graph_failed
                        .swap(false, std::sync::atomic::Ordering::Relaxed)
                    {
                        // `graph_spec_step` may have detached MTP while a
                        // warm-up was in flight.  Do not reinterpret its
                        // terminal device failure as a plain decode step;
                        // restore the head, clear both mirrors, and surface
                        // one explicit error to the caller.
                        self.finish_generation(&mut mtp, &mut router, true);
                        return Err("GPU MTP graph failed during speculative decode".to_string());
                    }
                    // Declined (batch graph refused): plain forward below —
                    // and a round that produced one token for the trial's
                    // ledger, so a graph that keeps refusing is measured out
                    // like a head that keeps missing (it was spinning
                    // forever on a file whose batch graph declines).
                    // A declined round is not a cheap one-token round — it
                    // is a verify that does not exist for this file (a
                    // healed q8_2f tail measured 760 drafts, 0 accepted, 33
                    // against 48.8 tok/s while the monitor called the draft
                    // alone "paying"). Count it as the losing streak in one.
                    spec_mon.round(t_round.elapsed().as_secs_f64() * 1e3, 1);
                    spec_mon.tokens = 0.0;
                    spec_mon.fails = 3;
                    spec_trial = Self::spec_trial_round(spec_trial, &mut spec_mon, generated + 1);
                    hidden = self.forward_layers(&self.embed_single(t_next), next_pos, task_mask);
                    next_pos += 1;
                    continue 'decode;
                }
                // ── Speculative: draft t+2, verify in a fused pair ──
                Some(m) if !graph_spec && generated + 1 < max_tokens => {
                    let draft = self.mtp_step(m, &hidden, t_next, next_pos - 1);
                    drafted += 1;
                    let emb1 = self.embed_single(t_next);
                    let emb2 = self.embed_single(draft);
                    let (h1, h2) = self.forward_pair(&emb1, &emb2, next_pos);

                    inference::rms_norm_into(
                        &h1,
                        &self.weights.final_norm,
                        self.rms_eps,
                        self.norm_style,
                        &mut self.ws.n1,
                    );
                    let mut logits1 = self.lm_head_forward(&self.ws.n1);
                    let t_after = sampler::sample_with_scratch_pool(
                        &logits1,
                        &self.sampler_config,
                        &all_ids,
                        &mut self.rng,
                        &mut self.sampler_scratch,
                        self.pool.as_deref(),
                    );
                    if self.confidence_on {
                        confidence.push(sampler::top1_prob_pool(
                            self.pool.as_deref(),
                            &mut self.sampler_scratch,
                            &logits1,
                            t_after,
                            calib_temp,
                        ));
                    }
                    attention::recycle_buf(&mut logits1);
                    if trace_on {
                        // Speculative decode is mutually exclusive with
                        // dynamic routing (router is None here) — no skill.
                        traces.push(TokenTrace {
                            t: generated,
                            token_id: t_after,
                            confidence: confidence.last().copied().unwrap_or(0.0),
                            active_skill: None,
                            recon: None,
                            switched: false,
                        });
                    }
                    let stop = !commit!(t_after);

                    if t_after == draft {
                        accepted += 1;
                        self.commit_linear_scratch();
                        let _ = self.mtp_step(m, &h1, t_after, next_pos);
                        hidden = h2;
                        next_pos += 2;
                    } else {
                        // The draft lane is wrong: roll its KV entry back.
                        for layer in &mut self.kv_cache.layers {
                            layer.truncate_last(1);
                        }
                        if !stop {
                            let _ = self.mtp_step(m, &h1, t_after, next_pos);
                            hidden = self.forward_layers(
                                &self.embed_single(t_after),
                                next_pos + 1,
                                None,
                            );
                        }
                        next_pos += 2;
                    }
                    if stop {
                        break 'decode;
                    }
                }
                // ── Vanilla: forward the sampled token ──
                _ => {
                    // ── DeepSeek-V4 speculative decode (CMF_DSV4_SPEC=1):
                    // draft five on the card, verify batched, commit the
                    // accepted prefix. Greedy only; a rejected token's state
                    // is restored and replayed, so output equals the walk. ──
                    #[cfg(feature = "gpu")]
                    if Self::dsv4_spec_on() && self.dsv4.is_some() {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            eprintln!(
                                "dsv4-spec гейт: mtp={} mask={} router={} trace={} temp={} rep={} ",
                                !self.dsv4_mtp.is_empty(),
                                task_mask.is_none(),
                                router.is_none(),
                                !trace_on,
                                self.sampler_config.temperature < 1e-6,
                                self.sampler_config.repetition_penalty == 1.0,
                            );
                        });
                    }
                    #[cfg(feature = "gpu")]
                    if Self::dsv4_spec_on()
                        && self.dsv4.is_some()
                        && !self.dsv4_mtp.is_empty()
                        && task_mask.is_none()
                        && router.is_none()
                        && !trace_on
                        && self.sampler_config.temperature < 1e-6
                        && self.sampler_config.repetition_penalty == 1.0
                        && generated + 1 < max_tokens
                        && all_ids.len() >= 2
                        && generated >= dsv4_spec_retry_at
                    {
                        let tip_token = all_ids[all_ids.len() - 2];
                        let drafted0 = drafted;
                        let round = self.dsv4_spec_step(
                            tip_token,
                            t_next,
                            next_pos,
                            max_tokens.saturating_sub(generated),
                            &mut drafted,
                            &mut accepted,
                        );
                        if drafted > drafted0 {
                            let useful = round.as_ref().is_some_and(|(extra, _)| !extra.is_empty());
                            if useful {
                                dsv4_spec_bad = 0;
                            } else {
                                dsv4_spec_bad += 1;
                                if dsv4_spec_bad >= 2 {
                                    dsv4_spec_bad = 0;
                                    dsv4_spec_retry_at = generated.saturating_add(32);
                                    tracing::info!(
                                        "dsv4: draft не окупился дважды — точный walk на 32 токена"
                                    );
                                }
                            }
                        }
                        if let Some((extra, n_pos)) = round {
                            next_pos = n_pos;
                            let mut stopped = false;
                            for &id in &extra {
                                if self.confidence_on {
                                    confidence.push(0.0);
                                }
                                if !commit!(id) {
                                    stopped = true;
                                    break;
                                }
                            }
                            if stopped {
                                break 'decode;
                            }
                            continue 'decode;
                        }
                    }
                    self.graph_want_logits = fuse_lm;
                    // Greedy burst (CMF_MULTISTEP, default 8, 1 = off): while
                    // nothing observes per-token state — pure argmax sampling,
                    // no router/trace/confidence/mask — decode k tokens per
                    // submit and commit them wholesale. The trailing normal
                    // forward leaves logits for the loop top, as always.
                    let mut t_fwd = t_next;
                    let pure_greedy = self.sampler_config.temperature < 1e-6
                        && self.sampler_config.repetition_penalty == 1.0
                        && self.sampler_config.suppress_tokens.is_empty();
                    // Off by default: at every k the burst measured at or
                    // below the plain path on this graph shape (k=1 loses
                    // the argmax dispatches vs a 1 MB readback, k>=8 loses
                    // inter-step drains vs the saved sync). Experimental.
                    let burst_k = std::env::var("CMF_MULTISTEP")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    if pure_greedy
                        && burst_k >= 1
                        && fuse_lm
                        && task_mask.is_none()
                        && router.is_none()
                        && !trace_on
                        && !self.confidence_on
                    {
                        let mut stopped = false;
                        loop {
                            let room = max_tokens.saturating_sub(generated);
                            if room <= 2 {
                                break;
                            }
                            let k = burst_k.min(room - 1);
                            if k < 1 {
                                break;
                            }
                            let Some(ids) = self.try_multi_burst(t_fwd, next_pos, k) else {
                                if self
                                    .graph_failed
                                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                                {
                                    self.finish_generation(&mut mtp, &mut router, true);
                                    return Err(
                                        "GPU token graph failed during greedy burst".to_string()
                                    );
                                }
                                break;
                            };
                            next_pos += k;
                            for &id in &ids {
                                if !commit!(id) {
                                    stopped = true;
                                    break;
                                }
                            }
                            if stopped {
                                break;
                            }
                            t_fwd = *ids.last().unwrap();
                        }
                        if stopped {
                            break 'decode;
                        }
                    }
                    // Metal: keep the draft head's cache in step through
                    // the trial's plain phase and a paused speculation —
                    // the pair (hidden, t_fwd) at next_pos−1, the step the
                    // round's draft 0 would take. Without it the head's
                    // cache lagged the trunk by every plain token for the
                    // rest of the generation: the batched warm-up declined
                    // every later round and its rows went one by one (a
                    // whole MTP step per accepted token), and the drafts
                    // attended a context with those tokens missing.
                    #[cfg(target_os = "macos")]
                    if graph_spec
                        && spec_watchdog_off
                        && next_pos > 0
                        && self.mtp_graph_mode == Some(true)
                        && crate::gpu::q1_force()
                    {
                        if let Some(m) = mtp.as_mut() {
                            let _ = self.mtp_step_metal(m, &hidden, t_fwd, next_pos - 1, false);
                        }
                    }
                    hidden = self.forward_layers(&self.embed_single(t_fwd), next_pos, task_mask);
                    next_pos += 1;
                    // Dynamic routing: the forward updated φ; ask the
                    // router whether to switch skills before the next token.
                    if let Some(r) = &mut router {
                        let phi = self.dyn_phi_ema.clone();
                        let decision = r.step(&phi, generated);
                        if let Some(new_active) = decision {
                            let _ = self.set_active_skill(new_active);
                        }
                        // Backfill this token's coherence + switch flag from
                        // the just-run eval (freshest measured values).
                        if trace_on {
                            if let Some(last) = traces.last_mut() {
                                let e = r.last_best_e();
                                last.recon = e.is_finite().then_some(e);
                                last.switched = decision.is_some();
                            }
                        }
                    }
                }
            }
        }

        let cancelled = finish_reason == "cancelled";
        self.finish_generation(&mut mtp, &mut router, cancelled);

        let output_ids = &all_ids[input_ids.len()..];
        // Forwarded = prompt + all generated but the LAST sampled token
        // (emitted without being fed back). Exact only without MTP —
        // reuse is gated off when MTP is active.
        let forwarded = input_ids.len() + output_ids.len().saturating_sub(1);
        if cancelled {
            self.kv_history.clear();
        } else {
            self.kv_history = all_ids[..forwarded.min(all_ids.len())].to_vec();
        }
        confidence.truncate(output_ids.len()); // guard against any overshoot
        traces.truncate(output_ids.len());
        Ok(GenerateResult {
            text: self.tokenizer.decode(output_ids),
            token_ids: output_ids.to_vec(),
            prompt_tokens: input_ids.len(),
            tokens_generated: generated,
            finish_reason,
            mtp_drafted: drafted,
            mtp_accepted: accepted,
            token_confidence: confidence,
            traces,
        })
    }

    /// One MTP step: feed `(hidden_p, token_{p+1})` into the draft head,
    /// advance its KV cache at position `p`, return the drafted token
    /// for position `p+2`.
    fn mtp_step(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> u32 {
        self.mtp_step_h(m, hidden, next_token, position).0
    }

    /// Tally for `CMF_MTP_CHAIN_PROBE`: per depth, how often the CHAIN is
    /// still an exact prefix of the real continuation. Printed every 128
    /// depth-0 samples so a killed run still shows its table.
    fn chain_probe_note(depth: usize, prefix_ok: bool) {
        use std::sync::Mutex;
        static T: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
        let mut t = T.lock().unwrap();
        if t.len() <= depth {
            t.resize(depth + 1, (0, 0));
        }
        t[depth].0 += 1;
        t[depth].1 += prefix_ok as u64;
        if depth == 0 && t[0].0 % 128 == 0 {
            let line: Vec<String> = t
                .iter()
                .enumerate()
                .map(|(d, (n, k))| {
                    format!(
                        "d{}={:.0}%({n})",
                        d + 1,
                        100.0 * *k as f64 / (*n).max(1) as f64
                    )
                })
                .collect();
            eprintln!("mtp-chain: {}", line.join(" "));
        }
    }

    /// `mtp_step` that also hands back the block's own output hidden — the
    /// state a CHAINED draft feeds the next step, the way a multi-token
    /// speculative round iterates the head on itself.
    /// One MTP block step from (trunk hidden, token): the head's LOGITS
    /// and the block's own hidden for chaining. The draft is argmax of the
    /// logits on the greedy path and a draw from their post-chain
    /// distribution on the sampling path.
    fn mtp_step_hl(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // The graph arm: the MTP block as a one-layer token graph with the
        // head fused — device attention over the block's own KV mirror,
        // one submit for block + head, hidden and logits back together.
        // Decided once per generation (see `mtp_graph_mode`).
        #[cfg(target_os = "macos")]
        if self.mtp_graph_mode != Some(false) && crate::gpu::q1_force() {
            if let Some(r) = self.mtp_step_metal(m, hidden, next_token, position, true) {
                self.mtp_graph_mode = Some(true);
                return r;
            }
            if self.mtp_graph_mode == Some(true) {
                tracing::error!("mtp Metal graph failed after admission");
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return (Vec::new(), Vec::new());
            }
            self.mtp_graph_mode = Some(false);
        }
        #[cfg(feature = "gpu")]
        if self.mtp_graph_mode != Some(false) {
            if !self.mtp_graph_ok(m) {
                if self.mtp_graph_mode == Some(true) {
                    // A mirror was already admitted, so a capability change
                    // cannot safely switch this request to the stale CPU
                    // cache.  Keep the same terminal contract as a failed
                    // token graph.
                    tracing::error!("mtp graph became unavailable after admission");
                    self.clear_sequence_state();
                    self.graph_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    return (Vec::new(), Vec::new());
                }
                self.mtp_graph_mode = Some(false);
            } else {
                if let Some(r) = self.mtp_step_graph(m, hidden, next_token, position) {
                    self.mtp_graph_mode = Some(true);
                    return r;
                }
                if self.graph_failed.load(std::sync::atomic::Ordering::Relaxed) {
                    // A token graph can have admitted a persistent MTP/GDN
                    // mirror before its readback failed.  The CPU MTP cache
                    // is not a valid continuation in that state; leave the
                    // flag set so the generation caller returns through its
                    // terminal error path instead of silently switching
                    // arithmetic.
                    return (Vec::new(), Vec::new());
                }
                // `mtp_graph_ok` was true, so a None here means a refusal or
                // failure after graph admission.  Do not fall through to a
                // CPU cache whose rows may lag the device mirror.
                tracing::error!("mtp graph failed or declined after admission");
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return (Vec::new(), Vec::new());
            }
        }
        // fc concat order is [enorm(embed); hnorm(hidden)] — EMBEDDING
        // FIRST. Verified by the oracle (converter/mtp_oracle.py):
        // [emb;hid] → 45.8% acceptance, [hid;emb] → 0.00%.
        let e = self.embed_single(next_token);
        let mut cat = vec![0.0f32; 2 * self.hidden_size];
        let (cat_e, cat_h) = cat.split_at_mut(self.hidden_size);
        inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, cat_e);
        inference::rms_norm_into(hidden, &m.hnorm, self.rms_eps, self.norm_style, cat_h);
        let mut x = vec![0.0f32; self.hidden_size];
        m.eh_proj.matvec(&cat, &mut x, self.pool.as_deref());

        // One standard transformer block over the MTP's own cache.
        let lw = &m.layer;
        inference::rms_norm_into(
            &x,
            &lw.input_norm,
            self.rms_eps,
            self.norm_style,
            &mut self.ws.n1,
        );
        let attn = match &lw.attn {
            // MLA models carry no MTP head; this path cannot see them.
            AttnKind::Mla(_) => unreachable!("MLA has no MTP/pair path"),
            AttnKind::Kda(_) => unreachable!("KDA has no MTP/pair path"),
            AttnKind::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                output_gate,
                softplus_gate,
                bias,
            } => {
                let mut cfg = self.attn_cfg(position);
                cfg.q_norm = q_norm.as_deref();
                cfg.k_norm = k_norm.as_deref();
                cfg.output_gate = *output_gate;
                cfg.softplus_gate = softplus_gate
                    .as_ref()
                    .map(|(gate, per_head)| (gate, *per_head));
                cfg.bias = bias
                    .as_ref()
                    .map(|(q, k, v)| (q.as_slice(), k.as_slice(), v.as_slice()));
                attention::qwen_attention(&self.ws.n1, wq, wk, wv, wo, &mut m.kv, &cfg)
            }
            AttnKind::Linear(_) | AttnKind::LinearGdn(_) | AttnKind::ShortConv(_) => {
                unreachable!("MTP block is full attention")
            }
        };
        for (i, &a) in attn.iter().enumerate() {
            x[i] += a;
        }
        inference::rms_norm_into(
            &x,
            &lw.post_norm,
            self.rms_eps,
            self.norm_style,
            &mut self.ws.p1,
        );
        let ffn = ffn_forward(&lw.ffn, &self.ws.p1, self.pool.as_deref(), None);
        for (i, &f) in ffn.iter().enumerate() {
            x[i] += f;
        }

        inference::rms_norm_into(
            &x,
            &m.final_norm,
            self.rms_eps,
            self.norm_style,
            &mut self.ws.n1,
        );
        let lg = self.lm_head_forward(&self.ws.n1);
        (lg, x)
    }

    /// `mtp_step_hl` reduced to the greedy draft: argmax of the head.
    fn mtp_step_h(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> (u32, Vec<f32>) {
        let (mut lg, x) = self.mtp_step_hl(m, hidden, next_token, position);
        let draft = sampler::argmax(&lg);
        attention::recycle_buf(&mut lg);
        (draft, x)
    }

    /// One speculative round for the trial: rounds 1..5 of a `Spec` phase
    /// advance it (the monitor already averaged this round); after five,
    /// the plain phase runs (once — a known plain rate decides at once);
    /// a decided speculation keeps re-checking the rule every round and
    /// stops after four losing rounds in a row.
    fn spec_trial_round(trial: SpecTrial, mon: &mut SpecMon, generated: usize) -> SpecTrial {
        match trial {
            SpecTrial::Spec { t0, gen0, rounds } => {
                let rounds = rounds + 1;
                if rounds >= 5 {
                    if mon.plain_ms > 0.0 {
                        let keep = mon.pays();
                        mon.fails = 0;
                        tracing::info!(
                            "speculation re-check: {:.2} tok/round in {:.1} ms vs plain {:.1} ms/tok — {}",
                            mon.tokens,
                            mon.round_ms,
                            mon.plain_ms,
                            if keep { "speculating" } else { "plain" }
                        );
                        SpecTrial::Decided {
                            spec: keep,
                            recheck_at: if keep { usize::MAX } else { generated + 128 },
                        }
                    } else if mon.pays() {
                        // Metal: the rounds land enough tokens each that no
                        // plain measurement is needed — keep speculating,
                        // and re-check every round (a losing streak sends
                        // the loop to the plain phase, below).
                        mon.fails = 0;
                        tracing::info!(
                            "speculation trial: {:.2} tok/round in {:.1} ms — speculating (plain not timed)",
                            mon.tokens,
                            mon.round_ms,
                        );
                        SpecTrial::Decided {
                            spec: true,
                            recheck_at: usize::MAX,
                        }
                    } else {
                        SpecTrial::Plain {
                            t0: std::time::Instant::now(),
                            gen0: generated,
                        }
                    }
                } else {
                    SpecTrial::Spec { t0, gen0, rounds }
                }
            }
            SpecTrial::Decided { spec: true, .. } => {
                if mon.pays() {
                    mon.fails = 0;
                    trial
                } else {
                    mon.fails += 1;
                    if mon.fails >= 4 {
                        if mon.plain_ms <= 0.0 {
                            // Metal, plain never timed: four doubtful rounds
                            // buy the (bounded) plain measurement, and the
                            // exact rule decides from it.
                            tracing::info!(
                                "speculation doubtful: {:.2} tok/round in {:.1} ms — timing plain",
                                mon.tokens,
                                mon.round_ms,
                            );
                            return SpecTrial::Plain {
                                t0: std::time::Instant::now(),
                                gen0: generated,
                            };
                        }
                        tracing::info!(
                            "speculation stopped: {:.2} tok/round in {:.1} ms vs plain {:.1} ms/tok",
                            mon.tokens,
                            mon.round_ms,
                            mon.plain_ms
                        );
                        SpecTrial::Decided {
                            spec: false,
                            recheck_at: generated + 128,
                        }
                    } else {
                        trial
                    }
                }
            }
            other => other,
        }
    }

    /// The MTP block's device-mirror id: the trunk's id with a high bit,
    /// so the (kv_id, layer) mirror keys never collide.
    fn mtp_kv_id(&self) -> u64 {
        self.graph_kv_id | (1u64 << 40)
    }

    /// The MTP block's mirror layer index: 0 — its own kv_id keeps it
    /// apart from the trunk, and the BATCH graph (the warm-up path) keys
    /// its mirrors at layer 0 with no base of its own, so the draft's
    /// token graph must key the same slot.
    const MTP_LAYER_BASE: usize = 0;

    /// The wgpu MTP draft writes speculative rows straight into its device
    /// mirror while the CPU owner retains only the real prompt/decode anchor.
    /// After verification, move that mirror cursor back to the anchor before
    /// replaying accepted pairs.  The next graph append then sees the same
    /// contiguous position as the CPU/Metal path without uploading stale
    /// speculative rows.
    #[cfg(feature = "gpu")]
    fn rewind_mtp_graph_mirror(&self, stored: usize) -> bool {
        self.mtp_graph_mode != Some(true)
            || crate::gpu::graph_kv_set_stored(self.mtp_kv_id(), Self::MTP_LAYER_BASE, stored)
    }

    /// A speculative verify graph appends the full `k+1` trunk rows before
    /// the acceptance count is known.  GDN state already has a snapshot
    /// restore; Full-attention mirrors need the matching logical cursor
    /// rewind so the next graph call does not reject an ahead-of-position KV
    /// cache after a partial acceptance.
    #[cfg(feature = "gpu")]
    fn rewind_trunk_graph_mirrors(&self, stored: usize) -> bool {
        let mut ok = true;
        let mut expected = false;
        for li in 0..self.num_layers {
            if matches!(
                self.weights.layers[self.phys_layer(li)].attn,
                AttnKind::Full { .. }
            ) {
                expected = true;
                ok &= crate::gpu::graph_kv_set_stored(self.graph_kv_id, li, stored);
            }
        }
        !expected || ok
    }

    /// Count the recurrent layers participating in the trunk verify graph.
    /// Snapshot restore is all-or-nothing across that set; deriving the count
    /// from the model keeps the restore contract valid for looped models too.
    fn graph_gdn_layer_count(&self) -> usize {
        (0..self.num_layers)
            .filter(|&li| {
                matches!(
                    &self.weights.layers[self.phys_layer(li)].attn,
                    AttnKind::LinearGdn(_)
                )
            })
            .count()
    }

    /// The block's input from (trunk hidden, token): eh_proj · [enorm(e);
    /// hnorm(h)] — the same arithmetic the per-op path starts with.
    fn mtp_block_input(&mut self, m: &MtpModule, hidden: &[f32], next_token: u32) -> Vec<f32> {
        let e = self.embed_single(next_token);
        let mut cat = vec![0.0f32; 2 * self.hidden_size];
        let (cat_e, cat_h) = cat.split_at_mut(self.hidden_size);
        inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, cat_e);
        inference::rms_norm_into(hidden, &m.hnorm, self.rms_eps, self.norm_style, cat_h);
        let mut x = vec![0.0f32; self.hidden_size];
        m.eh_proj.matvec(&cat, &mut x, self.pool.as_deref());
        x
    }

    /// Is the MTP block graphable at all (device up, full attention
    /// without softplus, dense FFN)? The plan itself is built per call.
    #[cfg(feature = "gpu")]
    fn mtp_block_graph_ok(&self, m: &MtpModule) -> bool {
        if std::env::var("CMF_MTP_GRAPH").as_deref() == Ok("0") {
            return false;
        }
        if !crate::gpu::wgpu_graph_on(crate::gpu::GraphPhase::Decode)
            || !crate::gpu::enabled_here()
            || self.attn_softcap > 0.0
            || self.attention_heads_per_layer.is_some()
        {
            return false;
        }
        matches!(
            &m.layer.attn,
            AttnKind::Full {
                softplus_gate: None,
                ..
            }
        ) && matches!(&m.layer.ffn, FfnKind::Dense(_))
    }

    /// Full MTP token-graph eligibility, including the fused lm-head and all
    /// block projection weights.  Keep this distinct from the block-only
    /// check: prompt warm-up does not need the head, while a draft step does.
    #[cfg(feature = "gpu")]
    fn mtp_graph_ok(&self, m: &MtpModule) -> bool {
        if !self.mtp_block_graph_ok(m) {
            return false;
        }
        let AttnKind::Full { wq, wk, wv, wo, .. } = &m.layer.attn else {
            return false;
        };
        let FfnKind::Dense(d) = &m.layer.ffn else {
            return false;
        };
        d.segs.is_empty()
            && wq.graph_weight().is_some()
            && wk.graph_weight().is_some()
            && wv.graph_weight().is_some()
            && wo.graph_weight().is_some()
            && d.gate_proj.graph_weight().is_some()
            && d.up_proj.graph_weight().is_some()
            && d.down_proj.graph_weight().is_some()
            && self.weights.lm_head.graph_weight().is_some()
    }

    /// One MTP block step on the wgpu token graph: block + fused head in
    /// one submit, the block hidden and the logits read back together.
    /// None = the graph cannot take this block (softplus gate, non-dense
    /// FFN, unquantized head, no device) — the caller keeps the per-op
    /// path for the whole generation.
    #[cfg(feature = "gpu")]
    fn mtp_step_graph(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if !self.mtp_graph_ok(m) {
            return None;
        }
        let lw = &m.layer;
        let AttnKind::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            output_gate,
            softplus_gate,
            bias,
        } = &lw.attn
        else {
            return None;
        };
        if softplus_gate.is_some() {
            return None;
        }
        let FfnKind::Dense(d) = &lw.ffn else {
            return None;
        };
        if !d.segs.is_empty() {
            return None; // tube layers run on the segmented path
        }
        // The block's input first: it borrows `self` mutably (embed scratch,
        // pool), the plan below borrows the weights immutably.
        let mut x = self.mtp_block_input(m, hidden, next_token);
        fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
            let (_, i, kind, rs) = t.graph_weight()?;
            Some(crate::gpu::GraphW {
                idx: i,
                kind,
                row_scale: rs,
                data: &[],
                prism: crate::gpu::GraphPrismOp::None,
                affine: false,
            })
        }
        let (model, _, _, _) = wq.graph_weight()?;
        let model = model.clone();
        let (lm_gw, lm_rows) = {
            let (_, i, kind, rs) = self.weights.lm_head.graph_weight()?;
            // The draft's head over the CMF_DRAFT_VOCAB shortlist (the same
            // cut the native Metal draft takes): 662 MB a step on Qwen3.8
            // becomes 170 MB at 65536; the verify keeps the full head.
            let rows = if kind == 6 {
                self.draft_head_rows(self.weights.lm_head.rows())
            } else {
                self.weights.lm_head.rows()
            };
            (
                crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                    prism: crate::gpu::GraphPrismOp::None,
                    affine: false,
                },
                rows,
            )
        };
        let layer = crate::gpu::GraphLayer {
            input_norm: &lw.input_norm,
            attn: crate::gpu::GraphAttn::Full {
                wq: gw(wq)?,
                wk: gw(wk)?,
                wv: gw(wv)?,
                wo: gw(wo)?,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                late_qk_norm: self.qk_norm_after_rope,
                bias: bias
                    .as_ref()
                    .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                output_gate: *output_gate,
                cpu_k: m.kv.k_heads(),
                cpu_v: m.kv.v_heads(),
            },
            post_norm: &lw.post_norm,
            ffn: crate::gpu::GraphFfn::Dense {
                gate: gw(&d.gate_proj)?,
                up: gw(&d.up_proj)?,
                down: gw(&d.down_proj)?,
            },
        };
        let nh = self.num_heads;
        let (nkv, hd, rd) = self.layer_geom(0);
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        let mut logits = Vec::new();
        let ok = crate::gpu::forward_token_graph(
            &model,
            self.mtp_kv_id(),
            std::slice::from_ref(&layer),
            &[None],
            self.o1_epoch,
            &self.inv_freq,
            &mut x,
            nh,
            nkv,
            hd,
            self.attn_scale,
            rd,
            self.hidden_size,
            self.intermediate_size,
            position,
            self.kv_cache.max_seq_len,
            gemma,
            self.rms_eps as f32,
            Some((&lm_gw, lm_rows)),
            &m.final_norm,
            &mut logits,
            &[],
            1,
            None,
            None,
            None,
            Self::MTP_LAYER_BASE,
            true,
        );
        match ok {
            crate::gpu::TokenGraphOutcome::Completed => {}
            crate::gpu::TokenGraphOutcome::Declined => return None,
            crate::gpu::TokenGraphOutcome::Failed => {
                // The backend has already admitted persistent state.  Keep
                // this distinct from a capability refusal so the caller
                // cannot switch to the stale CPU MTP cache.
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
        }
        logits.resize(self.vocab_size, 0.0);
        Some((logits, x))
    }

    /// The warm-ups of one speculative round on the device: every accepted
    /// (hidden, token) pair as ONE batched graph run over the MTP block
    /// (no head) — its kv_append lands the pairs in the block's mirror.
    /// `pairs` are consecutive positions from `first_pos`.  The tri-state
    /// result is intentional: a refusal before admission may use the
    /// per-row/CPU route, while a failure after admission must terminate the
    /// sequence rather than fall through to a stale CPU cache.
    #[cfg(feature = "gpu")]
    fn mtp_warm_graph(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> crate::gpu::BatchGraphOutcome {
        if pairs.is_empty() {
            return crate::gpu::BatchGraphOutcome::Completed;
        }
        if !self.mtp_block_graph_ok(m) {
            return crate::gpu::BatchGraphOutcome::Declined;
        }
        let hs = self.hidden_size;
        // Block inputs for every pair (eh_proj on the per-op path, one
        // matvec each — the plan's own prologue).
        let mut hiddens = Vec::with_capacity(pairs.len() * hs);
        for (h, t) in pairs {
            hiddens.extend_from_slice(&self.mtp_block_input(m, h, *t));
        }
        let lw = &m.layer;
        let AttnKind::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            output_gate,
            bias,
            ..
        } = &lw.attn
        else {
            return crate::gpu::BatchGraphOutcome::Declined;
        };
        let FfnKind::Dense(d) = &lw.ffn else {
            return crate::gpu::BatchGraphOutcome::Declined;
        };
        if !d.segs.is_empty() {
            return crate::gpu::BatchGraphOutcome::Declined; // tube layers run on the segmented path
        }
        fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
            let (_, i, kind, rs) = t.graph_weight()?;
            Some(crate::gpu::GraphW {
                idx: i,
                kind,
                row_scale: rs,
                data: &[],
                prism: crate::gpu::GraphPrismOp::None,
                affine: false,
            })
        }
        let Some((model, _, _, _)) = wq.graph_weight() else {
            return crate::gpu::BatchGraphOutcome::Declined;
        };
        let model = model.clone();
        let (Some(gwq), Some(gwk), Some(gwv), Some(gwo), Some(gg), Some(gu), Some(gd)) = (
            gw(wq),
            gw(wk),
            gw(wv),
            gw(wo),
            gw(&d.gate_proj),
            gw(&d.up_proj),
            gw(&d.down_proj),
        ) else {
            return crate::gpu::BatchGraphOutcome::Declined;
        };
        let layer = crate::gpu::GraphLayer {
            input_norm: &lw.input_norm,
            attn: crate::gpu::GraphAttn::Full {
                wq: gwq,
                wk: gwk,
                wv: gwv,
                wo: gwo,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                late_qk_norm: self.qk_norm_after_rope,
                bias: bias
                    .as_ref()
                    .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                output_gate: *output_gate,
                cpu_k: m.kv.k_heads(),
                cpu_v: m.kv.v_heads(),
            },
            post_norm: &lw.post_norm,
            ffn: crate::gpu::GraphFfn::Dense {
                gate: gg,
                up: gu,
                down: gd,
            },
        };
        let positions: Vec<usize> = (first_pos..first_pos + pairs.len()).collect();
        let nh = self.num_heads;
        let (nkv, hd, rd) = self.layer_geom(0);
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        crate::gpu::forward_batch_graph(
            &model,
            self.mtp_kv_id(),
            std::slice::from_ref(&layer),
            &self.inv_freq,
            &mut hiddens,
            nh,
            nkv,
            hd,
            rd,
            hs,
            self.intermediate_size,
            &positions,
            self.kv_cache.max_seq_len,
            gemma,
            self.rms_eps as f32,
            self.attn_scale,
            pairs.len(),
            &[],
            0,
            None,
        )
    }

    /// Complete an MTP warm-up after the batched graph has refused.  A
    /// graphable block is retried one row at a time; once any device row has
    /// been admitted, a CPU fallback would observe a stale mirror, so every
    /// token-graph refusal is terminal.  If the block is not graphable and no
    /// mirror exists yet, warming on the CPU is safe and records the CPU mode
    /// for the rest of the generation.
    #[cfg(feature = "gpu")]
    fn mtp_warm_graph_fallback(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> bool {
        if pairs.is_empty() {
            return true;
        }
        let graphable = self.mtp_block_graph_ok(m);
        if !graphable {
            // A previously admitted mirror cannot be made coherent by
            // appending to the host cache.  The caller turns this into a
            // terminal generation error and clears both mirrors.
            if self.mtp_graph_mode == Some(true) {
                return false;
            }
            self.mtp_graph_mode = Some(false);
            for (j, (h, t)) in pairs.iter().enumerate() {
                self.mtp_warm(m, h, *t, first_pos + j);
            }
            return true;
        }

        // The batch refusal is recoverable only through the same device
        // state.  Keep rows owned until each token graph has completed; a
        // None is treated as unsafe because the token-graph API deliberately
        // collapses its backend refusal/failure into that result.
        for (j, (h, t)) in pairs.iter().enumerate() {
            if self.mtp_step_graph(m, h, *t, first_pos + j).is_none() {
                return false;
            }
        }
        self.mtp_graph_mode = Some(true);
        true
    }

    /// Warm a contiguous set of MTP pairs using the existing graph seam, with
    /// an all-or-nothing error contract for callers that already admitted the
    /// trunk batch.  The non-GPU build keeps the same pair accounting while
    /// using the established CPU warm path.
    #[cfg(feature = "gpu")]
    fn mtp_warm_prefill_pairs(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> Result<(), &'static str> {
        // Keep unsupported token-graph heads on the established CPU MTP
        // route before admitting any block mirror.  Once a device mirror is
        // active, the same condition is terminal because CPU rows cannot
        // repair its state.
        if self.mtp_graph_mode == Some(false) || !self.mtp_graph_ok(m) {
            if self.mtp_graph_mode == Some(true) {
                return Err("MTP token graph became unavailable after admission");
            }
            self.mtp_graph_mode = Some(false);
            for (j, (h, t)) in pairs.iter().enumerate() {
                self.mtp_warm(m, h, *t, first_pos + j);
            }
            return Ok(());
        }
        match self.mtp_warm_graph(m, pairs, first_pos) {
            crate::gpu::BatchGraphOutcome::Completed => {
                if !pairs.is_empty() {
                    self.mtp_graph_mode = Some(true);
                }
                Ok(())
            }
            crate::gpu::BatchGraphOutcome::Declined => {
                if self.mtp_warm_graph_fallback(m, pairs, first_pos) {
                    Ok(())
                } else {
                    Err("MTP warm-up fallback failed after device admission")
                }
            }
            crate::gpu::BatchGraphOutcome::Failed => {
                Err("MTP warm batch graph failed after admission")
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn mtp_warm_prefill_pairs(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> Result<(), &'static str> {
        for (j, (h, t)) in pairs.iter().enumerate() {
            self.mtp_warm(m, h, *t, first_pos + j);
        }
        Ok(())
    }

    /// The MTP block alone — advance its KV with a (hidden, token) pair the
    /// verify just proved, without paying the head. What keeps the draft's
    /// attention context warm between speculative rounds.
    fn mtp_warm(&mut self, m: &mut MtpModule, hidden: &[f32], next_token: u32, position: usize) {
        let e = self.embed_single(next_token);
        let mut cat = vec![0.0f32; 2 * self.hidden_size];
        let (cat_e, cat_h) = cat.split_at_mut(self.hidden_size);
        inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, cat_e);
        inference::rms_norm_into(hidden, &m.hnorm, self.rms_eps, self.norm_style, cat_h);
        let mut x = vec![0.0f32; self.hidden_size];
        m.eh_proj.matvec(&cat, &mut x, self.pool.as_deref());
        inference::rms_norm_into(
            &x,
            &m.layer.input_norm,
            self.rms_eps,
            self.norm_style,
            &mut self.ws.n1,
        );
        let attn = match &m.layer.attn {
            AttnKind::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                output_gate,
                softplus_gate,
                bias,
            } => {
                let mut cfg = self.attn_cfg(position);
                cfg.q_norm = q_norm.as_deref();
                cfg.k_norm = k_norm.as_deref();
                cfg.output_gate = *output_gate;
                cfg.softplus_gate = softplus_gate.as_ref().map(|(g, p)| (g, *p));
                cfg.bias = bias
                    .as_ref()
                    .map(|(q, k, v)| (q.as_slice(), k.as_slice(), v.as_slice()));
                attention::qwen_attention(&self.ws.n1, wq, wk, wv, wo, &mut m.kv, &cfg)
            }
            _ => return,
        };
        let _ = attn;
    }

    /// Speculative decode ON the wgpu whole-token graph: draft k with the
    /// MTP head, verify all of them plus the tip in ONE batched graph
    /// submit whose tail folds the head, commit the accepted prefix and
    /// roll the GDN state back to the last real position. Greedy only —
    /// output equals the plain graph's token for token, the way the DSV4
    /// verify equals the walk.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn graph_spec_step(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        t_next: u32,
        next_pos: usize,
        drafted: &mut usize,
        accepted: &mut usize,
        // The committed stream (prompt + generated so far, `t_next`
        // included): the sampler chain's penalties read it, and the
        // sampling arm extends it with the drafts position by position.
        all_ids: &mut Vec<u32>,
        // Tokens left before `max_tokens`. A round commits up to k
        // accepted drafts, and those positions are already in the cache,
        // so the depth is capped here — trimming the output afterwards
        // would leave cache rows the committed stream does not have.
        room: usize,
    ) -> Option<(Vec<u32>, usize, Vec<f32>)> {
        // 3 is the measured optimum on Qwen3.6-27B / RTX 5090 (medians
        // of three, greedy): 51.1 tok/s against a plain 49.4, where k=2
        // gives 46.1, k=4 50.0, k=5 47.4, k=6 45.2. Acceptance is 89-91%
        // throughout — what turns the curve over is the verify, which
        // costs ~7.4 ms per extra position, and the draft ~3 ms a step.
        // 4 since the draft moved onto the graph (Qwen3.8-27B / 5090:
        // k=3 51.2, k=4 51.8 with the per-op draft; the graph draft
        // halves the draft cost, so the extra draft is cheaper still).
        // 5 with the int8 verify (the default: measured 76.5 against
        // k=4's 72-74 and k=6's 74 on the 5090), 4 with the f32 one.
        #[cfg(target_os = "macos")]
        let metal_native = crate::gpu::q1_force();
        #[cfg(not(target_os = "macos"))]
        let metal_native = false;
        #[cfg(feature = "gpu")]
        let k_default = if metal_native {
            // the Metal verify's GEMM tile is 8 rows wide and flat in b:
            // seven drafts + the tip fill it for free
            7
        } else if crate::gpu_wgpu::verify_i8_on() {
            5
        } else {
            4
        };
        #[cfg(not(feature = "gpu"))]
        let k_default = 4;
        let k_env: Option<usize> = std::env::var("CMF_GRAPH_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| (1..=8).contains(&v));
        // Adaptive depth: start below the card's flat-verify optimum and
        // let the accepted fraction move it — predictable text climbs to
        // the old default within a few rounds, prose settles at 2-3 where
        // the shorter verify pays.
        let (k_start, k_max) = if metal_native { (7, 7) } else { (3, k_default.max(5)) };
        let k_full: usize = k_env.unwrap_or_else(|| self.spec_k_adapt.unwrap_or(k_start));
        let k_spec = k_full.min(room).max(1);
        // a tail round cut short by `room` says nothing about the text:
        // it must not move the adaptive depth the next request starts at
        let k_capped = k_spec < k_full;
        if next_pos == 0 {
            return None;
        }
        let t_round = std::time::Instant::now();
        // Submissions per phase — and they say where the round's money is.
        // Qwen3.6-27B on an RTX 5090, k=3:
        //
        //   draft   9.3 ms / 12 submissions   (four per MTP step)
        //   verify 52.8 ms /  1               (the batched graph)
        //   commit  5.4 ms /  6               (two per warm)
        //
        // The verify is already one submit. The draft's own work is 834 MB
        // a step — 0.8 ms at this card's measured 1056 GB/s — against 3.1
        // ms measured, so ~0.58 ms of every step is round trip, not
        // arithmetic, and the same holds for the warms. Eighteen round
        // trips a round at roughly half a millisecond each is ~11 ms of a
        // 68 ms round: fusing the MTP block into ONE submit the way the
        // trunk already is projects to ~64 tok/s against today's 50.9.
        // That is the largest measured item left on this path.
        let subs = || crate::gpu_wgpu::SUBMITS.load(std::sync::atomic::Ordering::Relaxed);
        let sub0 = subs();
        // Greedy without penalties verifies by argmax equality (bit-exact
        // against the plain path). Anything else is speculative SAMPLING:
        // each draft is a DRAW from the MTP head's post-chain distribution
        // q_j, kept for the accept test; the verify's rows give p_j.
        let cfg = self.sampler_config.clone();
        let penalized = !(cfg.repetition_penalty == 1.0
            && cfg.presence_penalty == 0.0
            && cfg.suppress_tokens.is_empty());
        // Three verify regimes: plain greedy (argmax of the raw rows),
        // greedy WITH penalties (argmax of the penalized rows — a single
        // pass each, no distributions), and sampling (draw / accept /
        // correct on post-chain distributions).
        let greedy_pen = cfg.temperature < 1e-6 && penalized;
        let sampling = cfg.temperature >= 1e-6;
        // Sampling with a top-k goes through the SPARSE chain: the dense
        // one builds nine 248k-float distributions a round (four drafts,
        // five verify rows) and measured 19-22 tok/s against a plain 40 —
        // the host, not the card. Sparse, the same nine cost tens of
        // microseconds each.
        let sparse = sampling && sampler::sparse_ok(&cfg);
        let base_len = all_ids.len();
        if sampling && !sparse && self.spec_q.len() < k_spec {
            self.spec_q.resize_with(k_spec, Vec::new);
        }
        if sparse && self.spec_qs.len() < k_spec {
            self.spec_qs.resize_with(k_spec, Vec::new);
        }
        // Draft the chain: first from the trunk's tip hidden, then the head
        // iterating on itself. Rows land in the MTP KV; the chain rows past
        // the first are speculation over speculative state and roll back
        // below, replaced by verified pairs.
        let mut drafts = Vec::with_capacity(k_spec);
        let mut hx = hidden.to_vec();
        // CMF_SPEC_DBG=1: draft 0 through BOTH MTP arms (graph and per-op)
        // from the same inputs — are the arms the difference, or the inputs?
        let spec_dbg = std::env::var("CMF_SPEC_DBG").is_ok();
        spec_stamp("pro");
        // Plain greedy on native Metal: the whole chain as one command
        // buffer (device argmax + embedding gather between the steps).
        // A decline before commit hands the round to the per-step loop
        // below; a failure after commit is terminal, like any graph
        // failure after admission.
        #[cfg(target_os = "macos")]
        if metal_native && !sampling && !greedy_pen && self.mtp_graph_mode != Some(false) {
            match self.mtp_draft_chain_metal(m, hidden, t_next, next_pos - 1, k_spec) {
                Ok(ids) => {
                    self.mtp_graph_mode = Some(true);
                    drafts = ids;
                }
                Err(true) => {
                    tracing::error!("mtp Metal draft chain failed after commit");
                    self.clear_sequence_state();
                    self.graph_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
                Err(false) => {}
            }
        }
        for j in drafts.len()..k_spec {
            let tok_in = if j == 0 { t_next } else { drafts[j - 1] };
            let mut dbg_ref: Option<(Vec<f32>, Vec<f32>)> = None;
            if spec_dbg {
                let saved = self.mtp_graph_mode;
                self.mtp_graph_mode = Some(false);
                let r = self.mtp_step_hl(m, &hx, tok_in, next_pos - 1 + j);
                self.mtp_graph_mode = saved;
                if self.graph_failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return None;
                }
                m.kv.truncate_last(1);
                dbg_ref = Some(r);
            }
            let (mut lg, hj) = self.mtp_step_hl(m, &hx, tok_in, next_pos - 1 + j);
            if self.graph_failed.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            if let Some((lg_cpu, h_cpu)) = dbg_ref {
                let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let dl = lg
                    .iter()
                    .zip(&lg_cpu)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                let dh = hj
                    .iter()
                    .zip(&h_cpu)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                eprintln!(
                    "spec-dbg j={j} pos {} tok_in {tok_in}: per-op draft {} graph draft {} | max|dlogit| {dl:.3} | |h_cpu| {:.2} |h_graph| {:.2} max|dh| {dh:.3} | kv rows {}",
                    next_pos - 1 + j,
                    sampler::argmax(&lg_cpu),
                    sampler::argmax(&lg),
                    n(&h_cpu),
                    n(&hj),
                    m.kv.seq_len
                );
            }
            let dj = if sparse {
                let mut q = std::mem::take(&mut self.spec_qs[j]);
                let ok = sampler::sparse_distribution_into(
                    &lg,
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                    &mut q,
                );
                let d = if ok {
                    sampler::draw_sparse(&q, &mut self.rng)
                } else {
                    // everything filtered: the dense chain's greedy fallback
                    let t = sampler::argmax(&lg);
                    q.clear();
                    q.push((t, 1.0));
                    t
                };
                self.spec_qs[j] = q;
                all_ids.push(d);
                d
            } else if sampling {
                let mut q = std::mem::take(&mut self.spec_q[j]);
                sampler::distribution_into(
                    &lg,
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                    &mut q,
                );
                let d = sampler::draw(&q, &mut self.rng);
                self.spec_q[j] = q;
                all_ids.push(d); // the next draft's penalties see this one
                d
            } else if greedy_pen {
                let d = sampler::argmax_penalized(
                    &lg,
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                );
                all_ids.push(d);
                d
            } else {
                sampler::argmax(&lg)
            };
            attention::recycle_buf(&mut lg);
            drafts.push(dj);
            hx = hj;
            spec_stamp("d.pick");
        }
        all_ids.truncate(base_len);
        *drafted += k_spec;
        let t_draft = t_round.elapsed();
        let sub_draft = subs();
        // Verify batch: [t_next, d1 .. d_{k-1}] at next_pos.. — every row's
        // logits come back from the graph's own head.
        let b = k_spec + 1;
        let mut hiddens = vec![0.0f32; b * self.hidden_size];
        for (i, &t) in std::iter::once(&t_next).chain(drafts.iter()).enumerate() {
            let e = self.embed_single(t);
            hiddens[i * self.hidden_size..(i + 1) * self.hidden_size].copy_from_slice(&e);
        }
        let positions: Vec<usize> = (next_pos..next_pos + b).collect();
        spec_stamp("v.emb");
        let (lm_gw, lm_rows) = {
            let (_, i, kind, rs) = self.weights.lm_head.graph_weight()?;
            (
                crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                    prism: crate::gpu::GraphPrismOp::None,
                    affine: false,
                },
                self.weights.lm_head.rows(),
            )
        };
        let mut logits = Vec::new();
        let final_norm = self.weights.final_norm.clone();
        // Plain greedy on Metal: the b argmaxes come from the device
        // (`argmax_rows` after the head) and the 7.9 MB logits plane is
        // never read back — the round's decision needs only the ids, and
        // the loop top takes the last verified id as `spec_forced`, which
        // is exactly what its argmax of the row would give. The full rows
        // stay for anything that reads them: sampling, penalties,
        // confidence, the verify oracle, the logit dump.
        // `CMF_METAL_DEV_ARGMAX=0` keeps the host path.
        #[cfg(target_os = "macos")]
        let greedy_dev = metal_native
            && !sampling
            && !greedy_pen
            && !self.confidence_on
            && self.final_softcap.is_none()
            // The host acceptance argmax scans the WHOLE head row
            // (`lm_rows`), the sampler's own row only `vocab_size`: they
            // coincide exactly when the head has no padding rows, and
            // only then is the device argmax (which scores `vocab_size`)
            // bit-identical to both.
            && self.vocab_size == lm_rows
            && std::env::var_os("CMF_METAL_VERIFY_CHECK").is_none()
            && std::env::var_os("CMF_LOGIT_DUMP").is_none()
            && std::env::var("CMF_METAL_DEV_ARGMAX").as_deref() != Ok("0");
        #[cfg(not(target_os = "macos"))]
        let greedy_dev = false;
        let mut dev_ids: Vec<u32> = Vec::new();
        #[cfg(target_os = "macos")]
        let verify_outcome = if metal_native {
            let lm = self.weights.lm_head.q1_parts()?;
            let n_score = self.vocab_size.min(lm_rows);
            self.try_batch_graph_metal(
                &mut hiddens,
                &positions,
                b,
                Some((lm, &final_norm, &mut logits)),
                if greedy_dev {
                    Some((n_score, &mut dev_ids))
                } else {
                    None
                },
            )
        } else {
            self.try_batch_graph_wgpu(
                &mut hiddens,
                &positions,
                b,
                Some(crate::gpu::SpecTail {
                    lm: lm_gw,
                    lm_rows,
                    final_norm: &final_norm,
                    logits_out: &mut logits,
                }),
            )
        };
        #[cfg(not(target_os = "macos"))]
        let verify_outcome = self.try_batch_graph_wgpu(
            &mut hiddens,
            &positions,
            b,
            Some(crate::gpu::SpecTail {
                lm: lm_gw,
                lm_rows,
                final_norm: &final_norm,
                logits_out: &mut logits,
            }),
        );
        match verify_outcome {
            crate::gpu::BatchGraphOutcome::Completed => {}
            crate::gpu::BatchGraphOutcome::Declined => {
                // The verifier refused before admission.  Its draft MTP
                // rows are still device-resident, so rewind the separate
                // mirror before the caller takes the exact one-token path.
                m.kv.truncate_last(k_spec);
                if !metal_native && !self.rewind_mtp_graph_mirror(next_pos) {
                    self.clear_sequence_state();
                    self.graph_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::error!("MTP graph mirror rewind failed after verify decline");
                }
                return None;
            }
            crate::gpu::BatchGraphOutcome::Failed => {
                // A failed batch may have advanced trunk/GDN state.  Clear
                // both mirrors and preserve the terminal outcome rather than
                // falling through to stale CPU state.
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::error!("MTP verify batch graph failed after admission");
                return None;
            }
        }
        // `CMF_METAL_VERIFY_CHECK=1`: run the same b tokens through the
        // plain per-token path and compare each row's argmax + logits with
        // the verify's — the bring-up oracle for the batched graph. The
        // plain forwards mutate the CPU state; it is snapshotted and put
        // back, and the K/V mirrors re-pointed, before the round goes on.
        #[cfg(target_os = "macos")]
        if metal_native && std::env::var("CMF_METAL_VERIFY_CHECK").as_deref() == Ok("1") {
            let snap: Vec<Vec<f32>> = self
                .kv_cache
                .layers
                .iter()
                .map(|l| l.linear_state.clone())
                .collect();
            let attn_lens: Vec<usize> = self.kv_cache.layers.iter().map(|l| l.seq_len).collect();
            let toks: Vec<u32> = std::iter::once(t_next)
                .chain(drafts.iter().copied())
                .collect();
            let want_save = self.graph_want_logits;
            self.graph_want_logits = false;
            for (i, &t) in toks.iter().enumerate() {
                let hi = self.forward_layers(&self.embed_single(t), next_pos + i, None);
                let _ = self.graph_logits.take();
                // CMF_SPEC_PLAIN_HIDDEN=1: the next round drafts from the
                // plain path's hidden instead of the verify's (an experiment
                // on the chain's sensitivity to the half-GEMM noise)
                if std::env::var("CMF_SPEC_PLAIN_HIDDEN").as_deref() == Ok("1") {
                    hiddens[i * self.hidden_size..(i + 1) * self.hidden_size].copy_from_slice(&hi);
                }
                let ref_lg = self.logits_from_hidden(&hi);
                let row = &logits[i * lm_rows..(i + 1) * lm_rows];
                let ra = sampler::argmax(&ref_lg);
                let va = sampler::argmax(row);
                let mut md = 0f32;
                let mut rms = 0f64;
                for j in 0..lm_rows.min(ref_lg.len()) {
                    let d = (ref_lg[j] - row[j]).abs();
                    md = md.max(d);
                    rms += (d as f64) * (d as f64);
                }
                let mut hd = 0f32;
                for j in 0..self.hidden_size {
                    hd = hd.max((hi[j] - hiddens[i * self.hidden_size + j]).abs());
                }
                eprintln!(
                    "verify-check row {i} tok {t} pos {}: ref argmax {ra} verify argmax {va} {} | max|dlogit| {md:.3} rms {:.4} | max|dhidden| {hd:.4}",
                    next_pos + i,
                    if ra == va { "OK" } else { "MISMATCH" },
                    (rms / lm_rows as f64).sqrt()
                );
            }
            self.graph_want_logits = want_save;
            // restore IN PLACE: the pending verify graph wraps these very
            // allocations (zero-copy) — replacing the Vec would strand it
            for (l, st) in self.kv_cache.layers.iter_mut().zip(snap) {
                if l.linear_state.len() == st.len() {
                    l.linear_state.copy_from_slice(&st);
                } else {
                    l.linear_state = st;
                }
            }
            for (li, (l, n0)) in self.kv_cache.layers.iter_mut().zip(attn_lens).enumerate() {
                let extra = l.seq_len.saturating_sub(n0);
                if extra > 0 {
                    l.truncate_last(extra);
                    crate::gpu_metal::kv_mirror_set_stored(self.graph_kv_id, li, n0);
                }
            }
        }
        let t_verify = t_round.elapsed();
        let sub_verify = subs();
        // Acceptance. Greedy: row i's argmax is the trunk's token after
        // input i. Sampling: accept draft i with min(1, p_i/q_i), and on
        // the first rejection draw the correction from max(0, p_i − q_i)
        // — that token is committed by the loop top as-is (spec_forced).
        let mut a = 0usize;
        let mut forced: Option<u32> = None;
        let ids: Vec<u32> = if sparse {
            let mut p = std::mem::take(&mut self.spec_ps);
            let mut res = std::mem::take(&mut self.spec_ress);
            while a < k_spec {
                let ok = sampler::sparse_distribution_into(
                    &logits[a * lm_rows..(a + 1) * lm_rows],
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                    &mut p,
                );
                if !ok {
                    let t = sampler::argmax(&logits[a * lm_rows..(a + 1) * lm_rows]);
                    p.clear();
                    p.push((t, 1.0));
                }
                match sampler::spec_accept_or_correct_sparse(
                    &p,
                    &self.spec_qs[a],
                    drafts[a],
                    &mut self.rng,
                    &mut res,
                ) {
                    None => {
                        all_ids.push(drafts[a]);
                        a += 1;
                    }
                    Some(c) => {
                        forced = Some(c);
                        break;
                    }
                }
            }
            all_ids.truncate(base_len);
            self.spec_ps = p;
            self.spec_ress = res;
            drafts.clone()
        } else if sampling {
            let mut p = std::mem::take(&mut self.spec_p);
            let mut res = std::mem::take(&mut self.spec_res);
            while a < k_spec {
                sampler::distribution_into(
                    &logits[a * lm_rows..(a + 1) * lm_rows],
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                    &mut p,
                );
                match sampler::spec_accept_or_correct(
                    &p,
                    &self.spec_q[a],
                    drafts[a],
                    &mut self.rng,
                    &mut res,
                    self.pool.as_deref(),
                ) {
                    None => {
                        all_ids.push(drafts[a]);
                        a += 1;
                    }
                    Some(c) => {
                        forced = Some(c);
                        break;
                    }
                }
            }
            all_ids.truncate(base_len);
            self.spec_p = p;
            self.spec_res = res;
            // the accepted drafts ARE the verified tokens after inputs 0..a
            drafts.clone()
        } else if greedy_pen {
            // Row i's penalized argmax, penalties over the stream that
            // includes the accepted drafts before it — the plain loop's
            // exact arithmetic, one pass per row, no working copy.
            let mut ids: Vec<u32> = Vec::with_capacity(b);
            for i in 0..b {
                let t = sampler::argmax_penalized(
                    &logits[i * lm_rows..(i + 1) * lm_rows],
                    &cfg,
                    all_ids,
                    &mut self.sampler_scratch,
                    self.pool.as_deref(),
                );
                ids.push(t);
                if i < k_spec && t == drafts[i] {
                    all_ids.push(t);
                } else {
                    break;
                }
            }
            all_ids.truncate(base_len);
            while a < k_spec && a < ids.len() && ids[a] == drafts[a] {
                a += 1;
            }
            // rows past the first mismatch were never scored; the loop
            // top re-samples the last verified row itself.
            ids
        } else if greedy_dev && dev_ids.len() == b {
            let ids = std::mem::take(&mut dev_ids);
            while a < k_spec && ids[a] == drafts[a] {
                a += 1;
            }
            ids
        } else {
            if logits.len() < b * lm_rows {
                // the device argmax was asked for and came back short:
                // no rows to fall back on — terminal like a failed batch
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::error!("Metal verify returned neither logits nor argmax ids");
                return None;
            }
            let ids: Vec<u32> = (0..b)
                .map(|i| sampler::argmax(&logits[i * lm_rows..(i + 1) * lm_rows]))
                .collect();
            while a < k_spec && ids[a] == drafts[a] {
                a += 1;
            }
            ids
        };
        spec_stamp("acc");
        if spec_dbg {
            eprintln!(
                "spec-dbg round: t_next {t_next} drafts {:?} verified {:?} accepted {a}",
                drafts, ids
            );
        }
        // CMF_METAL_VERIFY_CHECK=2: the commit oracle — plain-forward the
        // a+1 accepted tokens from a snapshot, then diff the replayed GDN
        // states and the appended K/V rows against that.
        #[cfg(target_os = "macos")]
        let commit_ref: Option<(Vec<Vec<f32>>, Vec<(usize, Vec<f32>, Vec<f32>)>)> = if metal_native
            && std::env::var("CMF_METAL_VERIFY_CHECK").as_deref() == Ok("2")
        {
            let snap: Vec<Vec<f32>> = self
                .kv_cache
                .layers
                .iter()
                .map(|l| l.linear_state.clone())
                .collect();
            let attn_lens: Vec<usize> = self.kv_cache.layers.iter().map(|l| l.seq_len).collect();
            let toks: Vec<u32> = std::iter::once(t_next)
                .chain(drafts.iter().copied())
                .collect();
            let want_save = self.graph_want_logits;
            self.graph_want_logits = false;
            for (i, &t) in toks.iter().take(a + 1).enumerate() {
                let _ = self.forward_layers(&self.embed_single(t), next_pos + i, None);
                let _ = self.graph_logits.take();
            }
            self.graph_want_logits = want_save;
            let plain_states: Vec<Vec<f32>> = self
                .kv_cache
                .layers
                .iter()
                .map(|l| l.linear_state.clone())
                .collect();
            let (nkv, hd) = (self.num_kv_heads, self.head_dim);
            let mut rows = Vec::new();
            for (li, (l, n0)) in self
                .kv_cache
                .layers
                .iter_mut()
                .zip(attn_lens.iter())
                .enumerate()
            {
                let extra = l.seq_len.saturating_sub(*n0);
                if extra > 0 {
                    let mut kk = Vec::new();
                    let mut vv = Vec::new();
                    for g in 0..nkv {
                        kk.extend_from_slice(&l.head_keys(g)[n0 * hd..]);
                        vv.extend_from_slice(&l.head_values(g)[n0 * hd..]);
                    }
                    rows.push((li, kk, vv));
                    l.truncate_last(extra);
                    crate::gpu_metal::kv_mirror_set_stored(self.graph_kv_id, li, *n0);
                }
            }
            for (l, st) in self.kv_cache.layers.iter_mut().zip(snap) {
                if l.linear_state.len() == st.len() {
                    l.linear_state.copy_from_slice(&st);
                } else {
                    l.linear_state = st;
                }
            }
            Some((plain_states, rows))
        } else {
            None
        };
        let warm_off = std::env::var("CMF_SPEC_WARM").is_ok_and(|v| v == "0");
        // Metal: the MTP cache cut and the round's warm-up SUBMIT come
        // BEFORE the trunk commit, so the warm-up's command buffer is
        // queued ahead of the GDN replay (second queue) and its wait
        // below no longer sits behind the replay — measured: the warm-up's
        // wait grew with the accepted count exactly like the replay does
        // (8 ms at a=1, 17 ms at a=3, 25 ms at a=5 for ~2 ms of its own
        // work). The replay now overlaps the warm-up's readback, the
        // round's return and the next draft chain.
        #[cfg(target_os = "macos")]
        let mut warm_pending: Option<MetalWarmPending> = None;
        #[cfg(target_os = "macos")]
        if metal_native {
            m.kv.truncate_last(k_spec.saturating_sub(1));
            if self.mtp_graph_mode == Some(true) {
                // the mirror rows below the cut are the CPU rows: re-point,
                // no re-upload
                crate::gpu_metal::kv_mirror_set_stored(
                    self.mtp_kv_id(),
                    Self::MTP_LAYER_BASE,
                    m.kv.seq_len,
                );
                if !warm_off && a > 0 {
                    let pairs: Vec<(&[f32], u32)> = (0..a)
                        .map(|j| {
                            (
                                &hiddens[j * self.hidden_size..(j + 1) * self.hidden_size],
                                ids[j],
                            )
                        })
                        .collect();
                    warm_pending = self.mtp_warm_batch_submit(m, &pairs, next_pos);
                }
            }
            spec_stamp("c.wsub");
        }
        // a fully-accepted round needs no restore: every input was real.
        #[cfg(target_os = "macos")]
        if metal_native {
            // the Metal verify never wrote its states: the commit replays the
            // accepted prefix into the CPU owners and appends the K/V rows
            if !self.metal_verify_commit(a) {
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::error!("Metal verify state/KV handoff failed after admission");
                return None;
            }
            if let Some((plain_states, rows)) = commit_ref {
                crate::gpu_metal::queue_fence();
                // the commit's replay runs on the second queue: collect it
                // before the oracle reads the CPU owners it writes into
                let _ = crate::gpu_metal::wait_replay();
                let (nkv, hd) = (self.num_kv_heads, self.head_dim);
                let mut worst_s = 0f32;
                let mut worst_li = 0usize;
                for (li, (l, ps)) in self.kv_cache.layers.iter().zip(&plain_states).enumerate() {
                    if l.linear_state.len() != ps.len() || ps.is_empty() {
                        continue;
                    }
                    let d = l
                        .linear_state
                        .iter()
                        .zip(ps)
                        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                    let n = ps.iter().fold(0f32, |m, y| m.max(y.abs()));
                    let rel = d / n.max(1e-6);
                    if rel > worst_s {
                        worst_s = rel;
                        worst_li = li;
                    }
                }
                let mut worst_k = 0f32;
                for (li, kk, vv) in &rows {
                    let l = &self.kv_cache.layers[*li];
                    let n0 = l.seq_len - (kk.len() / (nkv * hd));
                    let mut ck = Vec::new();
                    let mut cv = Vec::new();
                    for g in 0..nkv {
                        ck.extend_from_slice(&l.head_keys(g)[n0 * hd..]);
                        cv.extend_from_slice(&l.head_values(g)[n0 * hd..]);
                    }
                    if ck.len() == kk.len() {
                        let dk = ck
                            .iter()
                            .zip(kk)
                            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                        let dv = cv
                            .iter()
                            .zip(vv)
                            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                        worst_k = worst_k.max(dk).max(dv);
                    } else {
                        eprintln!(
                            "commit-check L{li}: kv row count mismatch {} vs {}",
                            ck.len(),
                            kk.len()
                        );
                    }
                }
                eprintln!(
                    "commit-check a={a}: worst GDN state rel-max diff {worst_s:.2e} (L{worst_li}) | worst K/V row abs diff {worst_k:.4}"
                );
            }
        }
        if !metal_native && a + 1 < b {
            let expected_gdn_layers = self.graph_gdn_layer_count();
            if expected_gdn_layers > 0
                && !crate::gpu::gdn_spec_restore(self.graph_kv_id, a, next_pos, expected_gdn_layers)
            {
                self.clear_sequence_state();
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::error!("GDN speculative restore failed after verify");
                return None;
            }
        }
        if !metal_native && !self.rewind_trunk_graph_mirrors(next_pos + a + 1) {
            // The verify graph committed the full batch, but one of its
            // persistent Full-attention mirrors could not be re-pointed to
            // the accepted prefix.  Treat that as terminal state failure;
            // an exact CPU fallback would otherwise consume stale GDN/KV.
            self.clear_sequence_state();
            self.graph_failed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::error!("trunk graph KV rewind failed after speculative verify");
            return None;
        }
        *accepted += a;
        // MTP cache: keep the first draft row (its inputs were real), drop
        // the chain's, then append the verified pairs the round produced.
        // Each of those is a whole MTP block on the per-op path and they
        // cost 5.8 ms of a 69 ms round at k=3 — a third of what the
        // round's own draft costs. PRICED, and they earn it: skipping
        // them (`CMF_SPEC_WARM=0`) drops acceptance from 89% to 81% at
        // k=3 and 85% to 74% at k=4, and the tok/s goes nowhere at k=3
        // (50.3 against 50.5) and backwards at k=4 (48.1 against 50.1).
        // The knob stays so the next person can re-price it after the
        // warms are batched instead of assuming either way.
        if !metal_native {
            // (Metal cut its MTP cache before the trunk commit, above)
            m.kv.truncate_last(k_spec.saturating_sub(1));
        }
        spec_stamp("c.trunc");
        if !metal_native
            && self.mtp_graph_mode == Some(true)
            && !self.rewind_mtp_graph_mirror(next_pos)
        {
            // The graph draft was admitted, so inability to move its cursor
            // back to the real anchor is a state failure, not a capability
            // refusal.  Do not warm or continue with a stale mirror.
            self.clear_sequence_state();
            self.graph_failed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::error!("MTP graph mirror rewind failed after verify commit");
            return None;
        }
        if !warm_off && a > 0 {
            // Graph arm: all accepted pairs in ONE batched run over the
            // MTP block; the token graph one by one if the batch declines.
            let mut warmed = false;
            #[cfg(target_os = "macos")]
            if metal_native && self.mtp_graph_mode == Some(true) {
                // the batched warm-up was submitted before the trunk
                // commit: collect it here; one by one on the token graph
                // if it declined (or failed)
                warmed = match warm_pending.take() {
                    Some(p) => self.mtp_warm_batch_finish(m, p),
                    None => false,
                };
                if !warmed {
                    warmed = true;
                    for j in 0..a {
                        let row =
                            hiddens[j * self.hidden_size..(j + 1) * self.hidden_size].to_vec();
                        if self
                            .mtp_step_metal(m, &row, ids[j], next_pos + j, false)
                            .is_none()
                        {
                            warmed = false;
                            break;
                        }
                    }
                }
            }
            if !warmed && self.mtp_graph_mode != Some(false) && !metal_native {
                let rows: Vec<Vec<f32>> = (0..a)
                    .map(|j| hiddens[j * self.hidden_size..(j + 1) * self.hidden_size].to_vec())
                    .collect();
                let pairs: Vec<(&[f32], u32)> = rows
                    .iter()
                    .zip(ids.iter())
                    .map(|(r, &t)| (r.as_slice(), t))
                    .collect();
                match self.mtp_warm_prefill_pairs(m, &pairs, next_pos) {
                    Ok(()) => warmed = true,
                    Err(err) => {
                        // A warm-up failure after graph admission cannot
                        // fall back to `mtp_warm`: the detached CPU cache is
                        // not authoritative for the device mirror.  Mark it
                        // terminal so the generation caller clears state and
                        // returns instead of drafting from stale attention.
                        tracing::error!("{err}");
                        self.clear_sequence_state();
                        self.graph_failed
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        self.cancel
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        return None;
                    }
                }
            }
            if !warmed {
                for j in 0..a {
                    let row = &hiddens[j * self.hidden_size..(j + 1) * self.hidden_size];
                    let row = row.to_vec();
                    self.mtp_warm(m, &row, ids[j], next_pos + j);
                }
            }
        }
        // The sampler's contract: logits of the LAST verified position —
        // unless a rejected draft already drew the correction, in which
        // case the loop top commits that token and samples nothing.
        spec_stamp("c.warm");
        if let Some(c) = forced {
            self.spec_forced = Some(c);
            self.graph_logits = None;
        } else if greedy_dev && logits.is_empty() {
            // the row's argmax IS the token the loop top would pick from
            // it (plain greedy, no penalties): commit it as forced
            self.spec_forced = Some(ids[a]);
            self.graph_logits = None;
        } else {
            let mut row = logits[a * lm_rows..(a + 1) * lm_rows].to_vec();
            row.resize(self.vocab_size, 0.0);
            if let Some(c) = self.final_softcap {
                for l in row.iter_mut() {
                    *l = c * (*l / c).tanh();
                }
            }
            self.graph_logits = Some(row);
        }
        let new_hidden = hiddens[a * self.hidden_size..(a + 1) * self.hidden_size].to_vec();
        spec_stamp("c.row");
        // Three phases, not two. The round's wall clock was 4 ms longer
        // than draft+verify and the difference had nowhere to be seen:
        // the accepted prefix re-runs the MTP block once per token to
        // keep the draft head's attention cache warm, and the GDN state
        // rolls back on any rejection. Both live here, after the verify.
        if std::env::var("CMF_GRAPH_SPEC_TIME").is_ok() {
            let end = subs();
            eprintln!(
                "spec-round: draft {:.1} ms/{} sub | verify {:.1} ms/{} sub | \
                 commit {:.1} ms/{} sub (accepted {a} of {k_spec}, full-head streak {})",
                t_draft.as_secs_f64() * 1e3,
                sub_draft - sub0,
                (t_verify - t_draft).as_secs_f64() * 1e3,
                sub_verify - sub_draft,
                (t_round.elapsed() - t_verify).as_secs_f64() * 1e3,
                end - sub_verify,
                self.draft_full_streak,
            );
        }
        // Native Metal's verify tile is flat in b (eight rows for the price
        // of one), so a shorter round only forfeits tokens — measured on
        // the M4: an essay round at k=2 still verified in 260 ms. The
        // adaptation is for cards whose verify grows with the rows.
        if k_env.is_none() && !metal_native && !k_capped {
            // Slow average and a wide band: a fast one oscillated 2↔3 on
            // an essay every other round (measured), which forfeits the
            // draft it just paid for.
            let f = a as f32 / k_spec.max(1) as f32;
            self.spec_acc_ewma += 0.2 * (f - self.spec_acc_ewma);
            let mut k_next = k_spec;
            if self.spec_acc_ewma >= 0.75 && k_spec < k_max {
                k_next = k_spec + 1;
            } else if self.spec_acc_ewma < 0.4 && k_spec > 2 {
                k_next = k_spec - 1;
            }
            if k_next != k_spec {
                self.spec_acc_ewma = 0.6;
                if std::env::var("CMF_GRAPH_SPEC_TIME").is_ok() {
                    eprintln!("spec-k: {k_spec} → {k_next}");
                }
            }
            self.spec_k_adapt = Some(k_next);
        }
        spec_stamp("end");
        Some((drafts[..a].to_vec(), next_pos + a + 1, new_hidden))
    }

    /// Micro-benchmark: two single-position forwards vs one fused pair
    /// from the current cache state (KV rewound after each probe).
    /// Returns (two_singles_ms, fused_pair_ms) per probe, or the (0, 0)
    /// sentinel when this model has no pair path to measure — the same
    /// answer the o1 arm gives, and the bench prints it the same way.
    /// (An architecture that loads its own layers leaves `weights.layers`
    /// empty; walking it here was an index panic, found by `bench` on
    /// deepseek_v4.)
    pub fn measure_pair_fusion(&mut self, iters: usize) -> (f64, f64) {
        if !self.pair_supported() {
            return (0.0, 0.0);
        }
        // This is a host-side pair micro-benchmark. It truncates the host KV
        // after every probe, so letting the whole-token graph participate
        // would leave its device GDN/KV mirror ahead of the next probe and
        // poison the process-wide graph verdict before the real generation
        // benchmark starts. Keep the existing per-op/GPU arithmetic while
        // suppressing only the stateful token graph for this measurement.
        let graph_env = std::env::var_os("CMF_GPU_WGPU_GRAPH");
        unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", "0") };
        let emb1 = self.embed_single(1);
        let emb2 = self.embed_single(2);
        let pos = self.kv_cache.seq_len();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.forward_layers(&emb1, pos, None);
            let _ = self.forward_layers(&emb2, pos + 1, None);
            for l in &mut self.kv_cache.layers {
                l.truncate_last(2);
            }
        }
        let singles_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.forward_pair(&emb1, &emb2, pos);
            for l in &mut self.kv_cache.layers {
                l.truncate_last(2);
            }
        }
        let pair_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        match graph_env {
            Some(value) => unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", value) },
            None => unsafe { std::env::remove_var("CMF_GPU_WGPU_GRAPH") },
        }
        (singles_ms, pair_ms)
    }

    /// Fused two-position forward: weight rows are streamed from memory
    /// once per layer for both positions. Full layers → fused GQA pair;
    /// linear layers → vmf_phase pair (lane 2 state is tentative in the
    /// per-layer scratch until the draft is accepted).
    /// Whether the fused two-position path covers every layer kind in
    /// this model. MLA and KDA run per position (their pair arms are
    /// unreachable); the seq prefill falls back to singles for them.
    fn pair_supported(&self) -> bool {
        // An EMPTY layer stack means the architecture loaded its own and
        // this path has nothing to walk. Checking that directly, rather
        // than naming each such architecture, is what makes the guard hold
        // for the next one: `any()` over no layers is false, so a
        // feature-by-feature test says "supported" for a model that has no
        // layers here at all.
        !self.weights.layers.is_empty()
            && self.g3n.is_none()
            && !self
                .weights
                .layers
                .iter()
                .any(|lw| matches!(&lw.attn, AttnKind::Mla(_) | AttnKind::Kda(_)))
    }

    fn forward_pair(
        &mut self,
        emb1: &[f32],
        emb2: &[f32],
        position: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut h1 = emb1.to_vec();
        let mut h2 = emb2.to_vec();
        let (_nkv, _hd, hs, _rd, eps) = (
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let pool = self.pool.clone();

        for li in 0..self.num_layers {
            let lw = &self.weights.layers[self.phys_layer(li)];
            // Norms into pipeline scratch (4 allocs/layer on the MTP
            // decode hot path before this).
            inference::rms_norm_into(
                &h1,
                &lw.input_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.n1,
            );
            inference::rms_norm_into(
                &h2,
                &lw.input_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.n2,
            );

            let (a1, a2) = match &lw.attn {
                AttnKind::Mla(_) => unreachable!("MLA has no MTP/pair path"),
                AttnKind::Kda(_) => unreachable!("KDA has no MTP/pair path"),
                AttnKind::Linear(w) => {
                    let cfg = self.vmf_cfg.expect("linear layer without vmf_cfg");
                    let layer = &mut self.kv_cache.layers[li];
                    let (state, scratch) = (&mut layer.linear_state, &mut layer.linear_scratch);
                    vmf_phase_pair(
                        &self.ws.n1,
                        &self.ws.n2,
                        w,
                        &cfg,
                        state,
                        scratch,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::LinearGdn(w) => {
                    let cfg = self.gdn_cfg.expect("gdn layer without gdn_cfg");
                    let layer = &mut self.kv_cache.layers[li];
                    let (state, scratch) = (&mut layer.linear_state, &mut layer.linear_scratch);
                    gdn_pair(
                        &self.ws.n1,
                        &self.ws.n2,
                        w,
                        &cfg,
                        state,
                        scratch,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::ShortConv(w) => {
                    let cfg = self
                        .short_conv_cfg
                        .expect("short-conv layer without short_conv_cfg");
                    let layer = &mut self.kv_cache.layers[li];
                    let (state, scratch) = (&mut layer.linear_state, &mut layer.linear_scratch);
                    short_conv_pair(
                        &self.ws.n1,
                        &self.ws.n2,
                        w,
                        &cfg,
                        state,
                        scratch,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate,
                    bias,
                } => {
                    let inv_freq_l = self.layer_inv_freq(li);
                    let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                    let cfg = QwenAttnCfg {
                        num_heads: self.layer_num_heads(li),
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        softcap: self.attn_softcap,
                        window: self.layer_window(li),
                        v_norm: self.attn_v_norm,
                        qk_norm_after_rope: self.qk_norm_after_rope,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        softplus_gate: softplus_gate
                            .as_ref()
                            .map(|(gate, per_head)| (gate, *per_head)),
                        rope_scale: self.layer_rope_scale(li),
                        bias: bias
                            .as_ref()
                            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        rms_eps: eps,
                        norm_style: self.norm_style,
                        pool: pool.as_deref(),
                    };
                    attention::qwen_attention_pair(
                        &self.ws.n1,
                        &self.ws.n2,
                        wq,
                        wk,
                        wv,
                        wo,
                        &mut self.kv_cache.layers[li],
                        &cfg,
                    )
                }
            };
            let (a1, a2) = match &self.weights.layers[self.phys_layer(li)].attn_out_norm {
                Some(w) => (
                    inference::rms_norm(&a1, w, self.rms_eps, self.norm_style),
                    inference::rms_norm(&a2, w, self.rms_eps, self.norm_style),
                ),
                None => (a1, a2),
            };
            for i in 0..self.hidden_size {
                h1[i] += a1[i];
                h2[i] += a2[i];
            }
            let (mut a1, mut a2) = (a1, a2);
            attention::recycle_buf(&mut a1);
            attention::recycle_buf(&mut a2);

            let lw = &self.weights.layers[self.phys_layer(li)];
            inference::rms_norm_into(
                &h1,
                &lw.post_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.p1,
            );
            inference::rms_norm_into(
                &h2,
                &lw.post_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.p2,
            );
            let (f1, f2) = match &lw.ffn {
                // Dual-branch layers need the raw residuals — run the
                // two positions through the same fn decode uses.
                FfnKind::DenseMoe(dm) => (
                    dense_moe_ffn(
                        dm,
                        &self.ws.p1,
                        &h1,
                        self.rms_eps,
                        self.norm_style,
                        self.pool.as_deref(),
                    ),
                    dense_moe_ffn(
                        dm,
                        &self.ws.p2,
                        &h2,
                        self.rms_eps,
                        self.norm_style,
                        self.pool.as_deref(),
                    ),
                ),
                _ => ffn_forward_pair(
                    &lw.ffn,
                    &self.ws.p1,
                    &self.ws.p2,
                    self.pool.as_deref(),
                    None,
                ),
            };
            let (f1, f2) = match &self.weights.layers[self.phys_layer(li)].ffn_out_norm {
                Some(w) => (
                    inference::rms_norm(&f1, w, self.rms_eps, self.norm_style),
                    inference::rms_norm(&f2, w, self.rms_eps, self.norm_style),
                ),
                None => (f1, f2),
            };
            for i in 0..self.hidden_size {
                h1[i] += f1[i];
                h2[i] += f2[i];
            }
            let (mut f1, mut f2) = (f1, f2);
            attention::recycle_buf(&mut f1);
            attention::recycle_buf(&mut f2);
            if let Some(sc) = self.weights.layers[self.phys_layer(li)].layer_scale {
                for i in 0..self.hidden_size {
                    h1[i] *= sc;
                    h2[i] *= sc;
                }
            }
            // Looped Transformer: apply final norm at the end of each loop iteration.
            if self.is_loop_end(li) && li + 1 < self.num_layers {
                h1 = inference::rms_norm(
                    &h1,
                    &self.weights.final_norm,
                    self.rms_eps,
                    self.norm_style,
                );
                h2 = inference::rms_norm(
                    &h2,
                    &self.weights.final_norm,
                    self.rms_eps,
                    self.norm_style,
                );
            }
        }
        // Real O(1) prefill pairs may also carry tentative lane-2 recurrent
        // state. Commit it before publishing the transition epoch so the
        // next serial/device row cannot observe a new attention epoch with an
        // old GDN state. Speculative pairs run only when O(1) is inactive and
        // retain their existing caller-controlled commit/rollback semantics.
        if self.o1_active() {
            self.commit_linear_scratch();
        }
        self.o1_progress();
        (h1, h2)
    }

    /// Commit lane-2 linear states after an accepted draft.
    fn commit_linear_scratch(&mut self) {
        for layer in &mut self.kv_cache.layers {
            if !layer.linear_scratch.is_empty() {
                std::mem::swap(&mut layer.linear_state, &mut layer.linear_scratch);
                layer.linear_scratch.clear();
            }
        }
    }

    /// Forward a full id sequence from a fresh cache and return the
    /// logits after the last position (golden-parity harness, bench).
    pub fn forward_ids(
        &mut self,
        ids: &[u32],
        task_mask: Option<&TaskMask>,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("empty id sequence".to_string());
        }
        self.clear_sequence_state();
        self.check_forward_graph("forward_ids setup", 0)?;
        if task_mask.is_none() {
            self.o1_begin();
        }
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut pos = 0usize;
        if let Some(b) = &mut self.dsv41 {
            let pool = self.pool.clone();
            let mut logits = Vec::new();
            crate::dsv41::forward_chunk(
                &b.0,
                &b.1,
                &b.2,
                &mut b.3,
                ids,
                0,
                pool.as_deref(),
                &mut logits,
            );
            if let Err(err) = self.o1_seal_checked() {
                self.clear_sequence_state();
                return Err(err);
            }
            return Ok(logits);
        }
        // Same routing predicate generation uses. Two reasons it must be
        // the same one: (1) a GDN hybrid's recurrent state is GPU-
        // resident, and a batched CPU prefill would build it on the host
        // only — decode then reads buffers the prefill never wrote;
        // (2) bench times THIS function and calls the result "prefill",
        // so a different path here reports a number production never
        // sees (W2 on 2×5090: 8.7 tok/s reported against 125 real).
        if self.can_prefill_batched() && !self.graph_prefill_preferred() && ids.len() > 2 {
            // prefill-GEMM in chunks; only the last position's hidden is
            // needed. (o1-compatible: the batch path attends per position
            // through qwen_attention, which carries the collection hook.)
            let chunk = self.prefill_chunk();
            let hs = self.hidden_size;
            while pos < ids.len() {
                let end = (pos + chunk).min(ids.len());
                let hb = self.prefill_batch_masked(&ids[pos..end], pos, task_mask);
                self.check_forward_graph("forward_ids batched prefill", end - 1)?;
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
        }
        // Same guards as generation's prefill — INCLUDING the graph one.
        // The CPU pair walk was intercepting positions that the resident
        // token graph would have run itself: on a GDN hybrid over wgpu
        // that is 89 ms of host forward against 7 ms of device submit,
        // and it made prefill look 12× slower than it is (W2 on an RTX
        // 5090, ctx 512: 11.2 tok/s with the walk, 136.6 without).
        // CMF_PAIR=0 opts out; a model whose layers live outside
        // `weights.layers` has no pair walk to take.
        if task_mask.is_none()
            && !self.graph_prefill_preferred()
            && !std::env::var("CMF_PAIR").is_ok_and(|v| v == "0")
            && self.pair_supported()
        {
            while pos + 1 < ids.len() {
                let e1 = self.embed_single(ids[pos]);
                let e2 = self.embed_single(ids[pos + 1]);
                let (_, h2) = self.forward_pair(&e1, &e2, pos);
                self.check_forward_graph("forward_ids pair", pos + 1)?;
                self.commit_linear_scratch();
                hidden = h2;
                pos += 2;
            }
        }
        while pos < ids.len() {
            hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, task_mask);
            self.check_forward_graph("forward_ids", pos)?;
            pos += 1;
        }
        // Harness contract: after forward_ids the cache is decode-ready —
        // under o1 that means sealed (bench measures the seal as part of
        // prefill, honestly).
        if let Err(err) = self.o1_seal_checked() {
            self.clear_sequence_state();
            return Err(err);
        }
        let normed = inference::rms_norm(
            &hidden,
            &self.weights.final_norm,
            self.rms_eps,
            self.norm_style,
        );
        Ok(self.lm_head_forward(&normed))
    }

    /// Run the V4.1 stack one token at a time and retain logits for every
    /// position. This is a diagnostic surface for comparing a converted
    /// checkpoint with a tokenwise reference implementation.
    #[doc(hidden)]
    pub fn dsv41_serial_logits(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        #[cfg(target_os = "macos")]
        crate::gpu_metal::set_io_namespace(self.graph_kv_id);
        if ids.is_empty() {
            return Err("empty id sequence".to_string());
        }
        self.clear_sequence_state();
        self.dsv41
            .as_ref()
            .ok_or_else(|| "dsv41 serial logits require a DeepSeek-V4.1 model".to_string())?;
        self.o1_begin();
        let rows = {
            let pool = self.pool.clone();
            let b = self
                .dsv41
                .as_mut()
                .expect("dsv41 checked above; state cannot change during forward");
            let mut rows = Vec::with_capacity(ids.len());
            for (position, &id) in ids.iter().enumerate() {
                let mut logits = Vec::new();
                crate::dsv41::forward_token(
                    &b.0,
                    &b.1,
                    &b.2,
                    &mut b.3,
                    id,
                    position,
                    pool.as_deref(),
                    &mut logits,
                );
                rows.push(logits);
            }
            rows
        };
        self.o1_seal();
        Ok(rows)
    }

    /// Teacher-forced perplexity over a token sequence (phase-C gate:
    /// honest quant comparisons instead of prompt vibes).
    ///
    /// Attention is EXACT even on a model whose layers are flagged for
    /// the O(1) kernel — scoring the backbone is the default on purpose
    /// (it is the yardstick). `nll_ids_o1` scores the CONVERTED model.
    pub fn ppl_ids(&mut self, ids: &[u32]) -> Result<f64, String> {
        let (nll, cnt) = self.nll_ids_from(ids, 0)?;
        Ok((nll / cnt.max(1) as f64).exp())
    }

    /// DTG-MA calibration pass (Patent 2): run `ids` through the model
    /// (CPU path, per position) and return each layer's per-neuron
    /// activation mass Σ|silu(gate)·up| — the statistic the task-guided
    /// FFN mask is derived from.
    pub fn probe_ffn_mass(&mut self, ids: &[u32]) -> Vec<Vec<f64>> {
        self.clear_sequence_state();
        FFN_PROBE.with(|p| {
            *p.borrow_mut() = Some(vec![vec![0f64; self.intermediate_size]; self.num_layers]);
        });
        crate::gpu::cpu_scope(|| {
            for (pos, &id) in ids.iter().enumerate() {
                let emb = self.embed_single(id);
                let _ = self.forward_layers(&emb, pos, None);
            }
        });
        self.clear_sequence_state();
        FFN_PROBE
            .with(|p| p.borrow_mut().take())
            .unwrap_or_default()
    }

    /// `probe_ffn_mass` over the BATCHED prefill: same accumulator, one
    /// sweep instead of one forward per token. What makes the statistic
    /// affordable on a 27B.
    pub fn probe_ffn_mass_batch(&mut self, ids: &[u32]) -> Result<Vec<Vec<f64>>, String> {
        if let Err(err) = self.nll_begin() {
            // A recorder can be left by a caller that was interrupted before
            // this request entered its scoring block.  Consume it even when
            // the preflight failure prevents initialization of a new one.
            let _ = FFN_PROBE.with(|p| p.borrow_mut().take());
            self.nll_end();
            return Err(err);
        }
        FFN_PROBE.with(|p| {
            *p.borrow_mut() = Some(vec![vec![0f64; self.intermediate_size]; self.num_layers]);
        });
        let result: Result<(), String> = (|| {
            for chunk in ids.chunks(256) {
                if chunk.len() < 2 {
                    continue;
                }
                self.nll_ids_masked(chunk, 0, None)?;
            }
            Ok(())
        })();
        self.nll_end();
        let probe = FFN_PROBE
            .with(|p| p.borrow_mut().take())
            .unwrap_or_default();
        match result {
            Ok(()) => Ok(probe),
            Err(err) => {
                drop(probe);
                Err(err)
            }
        }
    }

    /// Teacher-forced PPL with a task mask active (sparse execution) —
    /// the quality gate for a DTG-MA-masked skill. Sequential per
    /// position: the batched prefill path is dense-only.
    pub fn ppl_ids_masked(&mut self, ids: &[u32], mask: &TaskMask) -> Result<f64, String> {
        self.nll_begin()?;
        let result: Result<f64, String> = (|| {
            let mut nll = 0f64;
            let mut cnt = 0usize;
            let mut hidden = vec![0f32; self.hidden_size];
            for (pos, &id) in ids.iter().enumerate() {
                if pos > 0 {
                    inference::rms_norm_into(
                        &hidden,
                        &self.weights.final_norm,
                        self.rms_eps,
                        self.norm_style,
                        &mut self.ws.n1,
                    );
                    let mut logits = self.lm_head_forward(&self.ws.n1);
                    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum();
                    let p = ((logits[id as usize] - max) as f64).exp() / sum.max(1e-300);
                    nll -= p.max(1e-300).ln();
                    cnt += 1;
                    attention::recycle_buf(&mut logits);
                }
                let emb = self.embed_single(id);
                hidden = self.forward_layers(&emb, pos, Some(mask));
                self.nll_check_graph("masked serial forward", pos)?;
                // Consume a possible graph logits side channel before the
                // next row.  Masked scoring normally disables that route,
                // but stale channel state must never survive a request.
                let _ = self.graph_logits.take();
            }
            Ok((nll / cnt.max(1) as f64).exp())
        })();
        self.nll_end();
        result
    }

    /// Teacher-forced NLL sum + scored-token count over positions
    /// `start..len-1`, attention EXACT. Positions below `start` still
    /// run — they are the context — they are just not scored, so this
    /// pairs with `nll_ids_o1(ids, start)` over the very same tokens.
    ///
    /// Returning (nll, cnt) rather than a ppl is what lets a windowed
    /// caller combine windows before the exp, so every scored token
    /// weighs the same regardless of how the windows are cut.
    /// `nll_ids_from` with a task mask held active at every position.
    ///
    /// The batched prefill path does not thread masks, so this walks the
    /// per-position forward — slower, but it scores the file exactly the
    /// way `run --task` will serve it, which is the point of the gate
    /// that calls it. With `None` it defers to the fast path.
    /// Masked scoring rides the SAME batched sweep as unmasked scoring —
    /// the masked-inference fast path: `prefill_batch_masked` lands the
    /// per-visit FFN rows on the activations inside the fused arms. The
    /// per-position loop below remains only as the no-batch fallback.
    pub fn nll_ids_masked(
        &mut self,
        ids: &[u32],
        start: usize,
        task_mask: Option<&TaskMask>,
    ) -> Result<(f64, usize), String> {
        let task_mask = self.drop_open_mask(task_mask);
        self.nll_ids_inner(ids, start, task_mask)
    }

    pub fn nll_ids_from(&mut self, ids: &[u32], start: usize) -> Result<(f64, usize), String> {
        self.nll_ids_inner(ids, start, None)
    }

    fn nll_ids_inner(
        &mut self,
        ids: &[u32],
        start: usize,
        task_mask: Option<&TaskMask>,
    ) -> Result<(f64, usize), String> {
        self.nll_begin()?;
        let result: Result<(f64, usize), String> = (|| {
            let mut nll = 0f64;
            let mut cnt = 0usize;
            // An unmasked quality run with the resident wgpu graph must score
            // the same stateful path used by generation.  The layer-major
            // GEMM prefill below is a valid CPU/GEMM oracle, but it seeds
            // neither the graph's device GDN state nor its device KV mirrors;
            // using it here would silently score a different execution.  Keep
            // masked scoring on the exact per-position path as before, and
            // let the serial arm below drive the graph-aware scorer.
            // Only native Metal has a fused graph lm_head contract.  Vulkan
            // and other graph backends may expose hidden state without the
            // optional logits side channel; preserve their established CPU
            // norm/head fallback instead of turning that valid route into a
            // hard missing-logits error.
            let (graph_quality, fused_head_quality) = nll_graph_policy(
                task_mask.is_none(),
                self.graph_prefill_preferred(),
                crate::gpu::q1_force(),
            );
            self.graph_head_required = fused_head_quality;
            self.graph_want_logits = fused_head_quality;
            #[cfg(target_os = "macos")]
            if graph_quality && std::env::var("CMF_METAL_BATCH_NLL").as_deref() != Ok("0") {
                match self.nll_batch_metal(ids, start) {
                    MetalBatchNllOutcome::Completed(nll, count) => {
                        return Ok((nll, count));
                    }
                    MetalBatchNllOutcome::Declined => {}
                    MetalBatchNllOutcome::Failed(err) => return Err(err),
                }
            }
            if self.can_prefill_batched() && !graph_quality {
                // prefill-GEMM: layer-major position chunks, lm_head batched
                // (254MB lm_head read once per chunk, not per position).
                // The layer chunk is large (grouping positions by MoE experts
                // wins with size), lm_head in sub-blocks (logit buffer
                // 32×vocab ≈ 32MB instead of 128×).
                const CHUNK: usize = 128;
                const LM_SUB: usize = 32;
                let n = ids.len().saturating_sub(1);
                let hs = self.hidden_size;
                let rows = self.weights.lm_head.rows();
                let mut pos = 0usize;
                while pos < n {
                    let end = (pos + CHUNK).min(n);
                    let bsz = end - pos;
                    let hb = self.prefill_batch_masked(&ids[pos..end], pos, task_mask);
                    self.nll_check_graph("batched prefill", pos)?;
                    let mut k0 = 0usize;
                    while k0 < bsz {
                        let k1 = (k0 + LM_SUB).min(bsz);
                        let sb = k1 - k0;
                        // Sub-block entirely below the scored range: the KV
                        // it just built is all this pass needed from it.
                        if pos + k1 <= start {
                            k0 = k1;
                            continue;
                        }
                        let mut normed = vec![0.0f32; sb * hs];
                        for k in 0..sb {
                            let r = inference::rms_norm(
                                &hb[(k0 + k) * hs..(k0 + k + 1) * hs],
                                &self.weights.final_norm,
                                self.rms_eps,
                                self.norm_style,
                            );
                            normed[k * hs..(k + 1) * hs].copy_from_slice(&r);
                        }
                        let mut logits = vec![0.0f32; sb * rows];
                        self.weights
                            .lm_head
                            .matmat(&normed, sb, &mut logits, self.pool.as_deref());
                        for k in 0..sb {
                            if pos + k0 + k < start {
                                continue;
                            }
                            self.nll_check_graph("batched score row", pos + k0 + k)?;
                            let lg = &mut logits[k * rows..k * rows + self.vocab_size.min(rows)];
                            if let Some(mu) = self.logit_multiplier {
                                for v in lg.iter_mut() {
                                    *v *= mu;
                                }
                            }
                            // Gemma-class final-logit soft-capping: the
                            // decode paths apply it; scoring must too, or
                            // the uncapped softmax misprices every token.
                            if let Some(c) = self.final_softcap {
                                for v in lg.iter_mut() {
                                    *v = c * (*v / c).tanh();
                                }
                            }
                            // Cortiq Embryo hierarchical head: same correction
                            // the decode path applies (lm_head_forward).
                            if let Some(cm) = self.head_clusters.clone() {
                                self.hierarchical_head_logprobs(
                                    &normed[k * hs..(k + 1) * hs],
                                    &cm,
                                    lg,
                                );
                            }
                            let lg = &logits[k * rows..k * rows + self.vocab_size.min(rows)];
                            let target = ids[pos + k0 + k + 1] as usize;
                            let max = lg.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                            let lse: f64 = lg
                                .iter()
                                .map(|&v| ((v - max) as f64).exp())
                                .sum::<f64>()
                                .ln()
                                + max as f64;
                            nll += lse - lg[target] as f64;
                            cnt += 1;
                            if std::env::var("CMF_PPL_TRACE").is_ok() {
                                let top = lg
                                    .iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);
                                eprintln!(
                                    "BTRACE pos {} target {} nll {:.4} top {} lg_t {:.3} lg_top {:.3}",
                                    pos + k0 + k,
                                    target,
                                    lse - lg[target] as f64,
                                    top,
                                    lg[target],
                                    lg[top]
                                );
                            }
                        }
                        k0 = k1;
                    }
                    pos = end;
                }
                return Ok((nll, cnt));
            }
            for pos in 0..ids.len().saturating_sub(1) {
                let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, task_mask);
                self.nll_check_graph("serial forward", pos)?;
                // Architectures whose head lives inside their own stack return
                // the logits out of band and a zero hidden — DeepSeek-V4 folds
                // its hyper-connection copies between the last layer and the
                // norm, so it cannot hand back a vector this loop could use.
                // Scoring the zeros gave a perplexity of exactly the vocabulary
                // size, which is a uniform distribution reported as a
                // measurement. `generate` already reads this channel.
                let out_of_band = self.graph_logits.take();
                if self.graph_head_required && out_of_band.is_none() {
                    METAL_GRAPH_HEAD_MISS.fetch_add(
                        1,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    return Err(format!(
                        "fused Metal graph head did not complete at NLL position {pos}"
                    ));
                }
                if pos < start {
                    continue;
                }
                let logits = match out_of_band {
                    Some(lg) => lg,
                    None => {
                        let normed = inference::rms_norm(
                            &hidden,
                            &self.weights.final_norm,
                            self.rms_eps,
                            self.norm_style,
                        );
                        // lm_head_forward applies the final-logit softcap itself
                        // — capping again here double-squashed gemma-class
                        // logits (tanh∘tanh) and reported a flattered ppl.
                        self.lm_head_forward(&normed)
                    }
                };
                let target = ids[pos + 1] as usize;
                let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let lse: f64 = logits
                    .iter()
                    .map(|&v| ((v - max) as f64).exp())
                    .sum::<f64>()
                    .ln()
                    + max as f64;
                let tok_nll = lse - logits[target] as f64;
                if std::env::var("CMF_PPL_TRACE").is_ok() && pos < 48 {
                    let top = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    eprintln!(
                        "pos {pos:3} tgt {target:6} nll {tok_nll:7.3} | top1 {top:6} lg[t]={:.2} lg[top]={:.2}",
                        logits[target], logits[top]
                    );
                }
                nll += tok_nll;
                cnt += 1;
            }
            Ok((nll, cnt))
        })();
        self.nll_end();
        result
    }

    /// Score one post-layer hidden with the same final norm/head path used by
    /// decode. Keeping this in one helper is important for the production
    /// batch scorer: its rows stop before the final norm, just like the
    /// per-position O(1) path below.
    fn nll_from_hidden(&mut self, hidden: &[f32], target: u32, pos: usize) -> f64 {
        let normed = inference::rms_norm(
            hidden,
            &self.weights.final_norm,
            self.rms_eps,
            self.norm_style,
        );
        // lm_head_forward applies the final-logit softcap itself — capping
        // again here double-squashed gemma-class logits in earlier scorers.
        let mut logits = self.lm_head_forward(&normed);
        let target = target as usize;
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let lse: f64 = logits
            .iter()
            .map(|&v| ((v - max) as f64).exp())
            .sum::<f64>()
            .ln()
            + max as f64;
        let tok_nll = lse - logits[target] as f64;
        if std::env::var("CMF_PPL_TRACE").is_ok() && pos < 48 {
            let top = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!(
                "pos {pos:3} tgt {target:6} nll {tok_nll:7.3} | top1 {top:6} lg[t]={:.2} lg[top]={:.2}",
                logits[target], logits[top]
            );
        }
        attention::recycle_buf(&mut logits);
        tok_nll
    }

    /// Teacher-forced NLL of the CONVERTED model: the O(1) Nyström path
    /// is ACTIVE over the scored positions. Returns `Ok((nll sum, scored
    /// count))` over `prefill..len-1` and surfaces a post-mutation batch
    /// failure instead of returning a partial score.
    ///
    /// Runtime discipline, deliberately NOT the matrix probe's: the
    /// requested prefix plus any required deferred lead-in run the exact
    /// prompt pass — that pass is what freezes the landmarks and M — and
    /// every post-seal scored position goes through `NystromState::step()`,
    /// the same code decode runs.
    /// So the landmarks are PREFILL-frozen (what ships), not
    /// full-sequence oracles (what the published probe measured). When the
    /// requested prefix is shorter than the bounded transition, rows in the
    /// exact lead-in are still scored so the shifted target range is stable.
    ///
    /// Pair with `nll_ids_from(ids, prefill)` for the exact baseline
    /// over the identical token set — that ratio is the honest one.
    pub fn nll_ids_o1(&mut self, ids: &[u32], prefill: usize) -> Result<(f64, usize), String> {
        // This scorer consumes host hiddens, so never request the optional
        // token-graph lm_head side channel. `nll_begin` also consumes a
        // prior graph failure and clears only the cancel bit that failure
        // raised, leaving a caller-owned cancellation observable.
        self.nll_begin()?;
        let requested_prefix = (prefill > 0).then_some(prefill);
        self.o1_begin_with_prefix(requested_prefix);
        let n = ids.len().saturating_sub(1);
        let requested_start = prefill.min(n);
        // The exact prefix must reach the deferred boundary before a
        // collecting layer can convert. Rows between the requested start and
        // that boundary remain part of the public NLL range and are scored
        // from the same hidden pass below.
        let exact_end = if self.o1_active() {
            match requested_prefix {
                Some(requested) => self.o1_effective_boundary(requested),
                None => self
                    .o1_cfg
                    .as_ref()
                    .and_then(|c| crate::nystrom::o1_deferred_boundary(c.w, c.sink)),
            }
            .unwrap_or(requested_start)
            .min(n)
        } else {
            requested_start
        };
        let mut nll = 0f64;
        let mut cnt = 0usize;

        // Exact prompt pass over ids[..exact_end]: the seal consumes its
        // q/k/v. Rows at or after requested_start are scored here when the
        // bounded lead-in is longer than the caller's requested prefix.
        let mut pos = 0usize;
        if self.can_prefill_batched() {
            const CHUNK: usize = 128;
            while pos < exact_end {
                let end = (pos + CHUNK).min(exact_end);
                let hiddens = self.prefill_batch(&ids[pos..end], pos);
                if self
                    .graph_failed
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    self.cancel
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    self.nll_end();
                    return Err("GPU graph failed during O(1) NLL prefix".into());
                }
                for row in 0..end - pos {
                    let score_pos = pos + row;
                    if score_pos >= requested_start && score_pos < n {
                        nll += self.nll_from_hidden(
                            &hiddens[row * self.hidden_size..(row + 1) * self.hidden_size],
                            ids[score_pos + 1],
                            score_pos,
                        );
                        cnt += 1;
                    }
                }
                pos = end;
            }
        } else {
            while pos < exact_end {
                let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
                if self
                    .graph_failed
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    self.cancel
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    self.nll_end();
                    return Err("GPU graph failed during O(1) NLL prefix".into());
                }
                if pos >= requested_start && pos < n {
                    nll += self.nll_from_hidden(&hidden, ids[pos + 1], pos);
                    cnt += 1;
                }
                pos += 1;
            }
        }
        self.o1_seal_checked().map_err(|err| {
            self.nll_end();
            err
        })?;

        // Reuse the production whole-token batch graph for the post-seal
        // suffix when the caller explicitly enabled both routes. This is a
        // teacher-forced scorer, so every row is ids[pos] and its target is
        // ids[pos + 1]; no speculative tail or rollback state is involved.
        // A first Declined is safe to handle with the established serial O(1)
        // path. Once a chunk completes, however, the device recurrent state
        // owns the sequence and a later decline must be terminal rather than
        // falling back to stale CPU state.
        let batch_k = std::env::var("CMF_BATCH_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let batch_admitted = batch_k > 0
            && self.can_prefill_batched()
            && self.o1_active()
            && std::env::var("CMF_O1_GPU").as_deref() == Ok("1")
            && (0..self.num_layers).all(|li| {
                let cache = &self.kv_cache.layers[self.phys_layer(li)];
                cache.o1.is_none() || cache.o1_views().is_some()
            });
        if std::env::var("CMF_GRAPH_PROF").is_ok() {
            eprintln!(
                "nll-batch: phase=post-seal admission={} requested_k={} scored_rows={}",
                batch_admitted,
                batch_k,
                n.saturating_sub(exact_end),
            );
        }
        let mut batch_completed = false;
        if batch_admitted && exact_end < n {
            let hs = self.hidden_size;
            let mut batch_pos = exact_end;
            while batch_pos < n {
                let end = (batch_pos + batch_k).min(n);
                let bk = end - batch_pos;
                let mut hiddens = vec![0.0f32; bk * hs];
                for (row, &id) in ids[batch_pos..end].iter().enumerate() {
                    hiddens[row * hs..(row + 1) * hs].copy_from_slice(&self.embed_single(id));
                }
                let positions: Vec<usize> = (batch_pos..end).collect();
                let t_batch = std::time::Instant::now();
                let outcome = self.try_batch_graph_wgpu(&mut hiddens, &positions, bk, None);
                if std::env::var("CMF_GRAPH_PROF").is_ok() {
                    let ms = t_batch.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "nll-batch: phase=post-seal mode=o1 k={bk} pos={}..{} outcome={outcome:?} {ms:.1} ms ({:.1} tok/s)",
                        batch_pos,
                        end.saturating_sub(1),
                        bk as f64 / (ms / 1000.0),
                    );
                }
                if let Err(err) = self.nll_check_graph("batch graph", batch_pos) {
                    self.nll_end();
                    return Err(err);
                }
                match outcome {
                    crate::gpu::BatchGraphOutcome::Completed => {
                        batch_completed = true;
                        for row in 0..bk {
                            nll += self.nll_from_hidden(
                                &hiddens[row * hs..(row + 1) * hs],
                                ids[batch_pos + row + 1],
                                batch_pos + row,
                            );
                            cnt += 1;
                        }
                        batch_pos = end;
                    }
                    crate::gpu::BatchGraphOutcome::Declined => {
                        if batch_completed {
                            self.nll_end();
                            return Err(format!(
                                "O(1) NLL batch declined after completed chunk at position {batch_pos}"
                            ));
                        }
                        break;
                    }
                    crate::gpu::BatchGraphOutcome::Failed => {
                        self.nll_end();
                        return Err(format!(
                            "O(1) NLL batch graph failed after admission at position {batch_pos}"
                        ));
                    }
                }
            }
            if batch_completed && cnt == n.saturating_sub(requested_start) {
                self.nll_end();
                return Ok((nll, cnt));
            }
        }

        // Serial O(1) fallback/reference. It is intentionally retained when
        // batch admission declines before mutation; callers must label this
        // CMF_BATCH_K=0/per-position path separately from the production
        // whole-token batch route.
        for pos in exact_end..n {
            let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
            if self
                .graph_failed
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                self.cancel
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                self.nll_end();
                return Err(format!(
                    "GPU graph failed during O(1) NLL serial scoring at position {pos}"
                ));
            }
            nll += self.nll_from_hidden(&hidden, ids[pos + 1], pos);
            cnt += 1;
        }
        self.nll_end();
        Ok((nll, cnt))
    }

    /// Teacher-forced calibration data (B1): for each position, whether the
    /// argmax equals the actual next token, and the top-1 softmax prob
    /// (top-1 probability) under EACH temperature in `temps` — all from ONE forward
    /// pass (argmax/correctness are temperature-invariant; only p_max
    /// reshapes). Feeds `cortiq calibrate` (reliability/ECE + temperature
    /// fit): is the model's confidence a true property, or does it need a
    /// measured scaling?
    pub fn calib_ids(&mut self, ids: &[u32], temps: &[f32]) -> (Vec<bool>, Vec<Vec<f32>>) {
        self.clear_sequence_state();
        let n = ids.len().saturating_sub(1);
        let mut correct = Vec::with_capacity(n);
        let mut pmax = Vec::with_capacity(n);
        for pos in 0..n {
            let emb = self.embed_single(ids[pos]);
            let hidden = self.forward_layers(&emb, pos, None);
            let normed = inference::rms_norm(
                &hidden,
                &self.weights.final_norm,
                self.rms_eps,
                self.norm_style,
            );
            // lm_head_forward applies the final-logit softcap itself —
            // capping again here double-squashed gemma-class logits
            // (tanh∘tanh) and reported a flattered ppl.
            let logits = self.lm_head_forward(&normed);
            let target = ids[pos + 1] as usize;
            let (mut amax, mut mval) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits.iter().enumerate() {
                if v > mval {
                    mval = v;
                    amax = i;
                }
            }
            correct.push(amax == target);
            let row: Vec<f32> = temps
                .iter()
                .map(|&t| {
                    let tt = t.max(1e-3);
                    let s: f32 = logits.iter().map(|&v| ((v - mval) / tt).exp()).sum();
                    1.0 / s.max(1e-12) // numerator at the max is exp(0)=1
                })
                .collect();
            pmax.push(row);
        }
        self.clear_sequence_state();
        (correct, pmax)
    }

    /// Teacher-forced PPL with the dynamic router driving per-window
    /// skill switches (VMF experiment №2 measurement). Sequential (φ
    /// must update per token), returns (ppl, switch_count). The router
    /// must be enabled (`enable_dynamic_routing`); else this equals
    /// plain `ppl_ids`. The active skill when scoring token t shapes the
    /// logits for t+1 — on-policy over the held-out text itself.
    pub fn ppl_ids_dynamic(&mut self, ids: &[u32]) -> Result<(f64, usize), String> {
        if self.dyn_router.is_none() {
            return Ok((self.ppl_ids(ids)?, 0));
        }
        self.nll_begin()?;
        let saved_active = self.dyn_active;
        let mut router = self
            .dyn_router
            .take()
            .ok_or_else(|| "dynamic router disappeared before PPL scoring".to_string())?;
        router.reset();
        self.dyn_phi_seen = 0;
        let _ = self.set_active_skill(None);

        let result: Result<(f64, usize), String> = (|| {
            let mut nll = 0f64;
            let mut cnt = 0usize;
            for pos in 0..ids.len().saturating_sub(1) {
                let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
                self.nll_check_graph("dynamic serial forward", pos)?;
                let out_of_band = self.graph_logits.take();
                let mut logits = match out_of_band {
                    Some(lg) => lg,
                    None => {
                        let normed = inference::rms_norm(
                            &hidden,
                            &self.weights.final_norm,
                            self.rms_eps,
                            self.norm_style,
                        );
                        // lm_head_forward applies the final-logit softcap itself —
                        // capping again here double-squashed gemma-class logits
                        // and reported a flattered ppl.
                        self.lm_head_forward(&normed)
                    }
                };
                let target = ids[pos + 1] as usize;
                let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let lse: f64 = logits
                    .iter()
                    .map(|&v| ((v - max) as f64).exp())
                    .sum::<f64>()
                    .ln()
                    + max as f64;
                let tok_nll = lse - logits[target] as f64;
                if std::env::var("CMF_PPL_TRACE").is_ok() && pos < 48 {
                    let top = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    eprintln!(
                        "pos {pos:3} tgt {target:6} nll {tok_nll:7.3} | top1 {top:6} lg[t]={:.2} lg[top]={:.2}",
                        logits[target], logits[top]
                    );
                }
                nll += tok_nll;
                cnt += 1;
                attention::recycle_buf(&mut logits);
                // Route on the evolving phi (drives the NEXT token's skill).
                let phi = self.dyn_phi_ema.clone();
                if let Some(new_active) = router.step(&phi, pos) {
                    let _ = self.set_active_skill(new_active);
                }
            }
            Ok(((nll / cnt.max(1) as f64).exp(), router.switches.len()))
        })();

        // Restore the detached router and the active overlay on both success
        // and failure. The scoring state is cleared independently below.
        let _ = self.set_active_skill(saved_active);
        self.dyn_router = Some(router);
        self.nll_end();
        result
    }

    /// Routing probe φ (spec §9): mean-pooled hidden after `layer`.
    pub fn probe_phi(&mut self, ids: &[u32], layer: usize) -> Vec<f32> {
        self.clear_sequence_state();
        let mut acc = vec![0f32; self.hidden_size];
        for (pos, &id) in ids.iter().enumerate() {
            let h = self.forward_layers_upto(&self.embed_single(id), pos, None, Some(layer));
            for (a, v) in acc.iter_mut().zip(&h) {
                *a += v;
            }
        }
        let n = ids.len().max(1) as f32;
        for a in acc.iter_mut() {
            *a /= n;
        }
        self.clear_sequence_state();
        acc
    }

    /// Layer-major batched prefill (prefill-GEMM): full-attention —
    /// per-position with the existing operators (KV grows naturally,
    /// causality preserved), GDN projections / FFN / MoE — batched
    /// (a weight row is read from DRAM once per chunk, not per
    /// position). Returns the hidden of all positions [b × hidden].
    fn prefill_batch(&mut self, ids: &[u32], start_pos: usize) -> Vec<f32> {
        self.prefill_batch_masked(ids, start_pos, None)
    }

    /// `prefill_batch` with a task mask honored on the dense-FFN panels
    /// (the masked-inference fast path: full fused compute, mask lands on
    /// the activations). The whole-chunk GPU graph is skipped for masked
    /// layers by the callers' arms; the per-GEMM device paths stay in
    /// play because the zeroing happens on the host between them.
    fn prefill_batch_masked(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        task_mask: Option<&TaskMask>,
    ) -> Vec<f32> {
        self.prefill_batch_span(PrefillIn::Ids(ids), start_pos, task_mask, 0, usize::MAX)
    }

    /// The layer-major batched walk over a layer span [from..upto_excl):
    /// the whole prefill machinery (chunk graph, batched attends, GEMM
    /// panels) for a PARTIAL stack — the network split's prefill rides
    /// the same canon as the local one. Input is token ids (embeds
    /// itself, coordinator side) or ready boundary hiddens (worker side).
    fn prefill_batch_span(
        &mut self,
        input: PrefillIn<'_>,
        start_pos: usize,
        task_mask: Option<&TaskMask>,
        from: usize,
        upto_excl: usize,
    ) -> Vec<f32> {
        let hs = self.hidden_size;
        let b = match input {
            PrefillIn::Ids(ids) => ids.len(),
            PrefillIn::Hidden(hb) => hb.len() / hs,
        };
        let upto_excl = upto_excl.min(self.num_layers);
        // The CPU embed is deferred: when the chunk graph takes the run
        // from layer 0 it gathers the embeddings on the device instead.
        // A hidden input is ready by definition.
        let mut h: Vec<f32>;
        let mut h_ready;
        match input {
            PrefillIn::Ids(_) => {
                h = vec![0.0; b * hs];
                h_ready = false;
            }
            PrefillIn::Hidden(hb) => {
                h = hb.to_vec();
                h_ready = true;
            }
        }
        let fill_h = |h: &mut Vec<f32>, me: &Self| {
            if let PrefillIn::Ids(ids) = input {
                for (bi, &id) in ids.iter().enumerate() {
                    let e = me.embed_single(id);
                    h[bi * hs..(bi + 1) * hs].copy_from_slice(&e);
                }
                if let Ok(tp) = std::env::var("CMF_TRACE_POS") {
                    if let Ok(t) = tp.parse::<usize>() {
                        if t >= start_pos && t < start_pos + ids.len() {
                            let bi = t - start_pos;
                            let row = &h[bi * hs..(bi + 1) * hs];
                            let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                            eprintln!(
                                "BATCH pos {t} embed: id {} |h| = {n:.6} h0 {:.6} h1 {:.6} | b={} start={start_pos} ids[..8]={:?}",
                                ids[bi],
                                row[0],
                                row[1],
                                ids.len(),
                                &ids[..ids.len().min(8)]
                            );
                        }
                    }
                }
            }
        };
        let (_nkv, _hd, _rd, eps) = (
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rms_eps,
        );
        let pool = self.pool.clone();
        let norm_style = self.norm_style;
        let automatic_gpu_prefix = self.automatic_gpu_prefix();

        #[cfg(target_os = "macos")]
        let mut chunk_skip_until = 0usize;
        for li in from..upto_excl {
            let _capacity_tail = automatic_gpu_prefix
                .filter(|&prefix| li >= prefix)
                .map(|_| crate::gpu::enter_cpu_scope());
            crate::gpu::set_layer(li as i64); // layer-split GPU/CPU
            // GPU chunk graph (default-on under CMF_GPU=1): a run of
            // consecutive eligible layers for the whole chunk in ONE
            // Metal submission — norm, QKV, RoPE with fused mirror
            // append, causal attend, O, FFN, hidden device-resident
            // across the run. Any refusal falls through to the CPU path.
            #[cfg(target_os = "macos")]
            if task_mask.is_none() {
                if li < chunk_skip_until {
                    continue;
                }
                // Device-side embedding needs a q8_row embedding matrix;
                // with any other layout the CPU fills `h` first and the
                // graph starts from a ready hidden (refusing the whole
                // run over the embedding alone kept q4t models — the
                // whole Nanbeige/Bonsai class — on the CPU prefill).
                if !h_ready && li == 0 && self.weights.embed_tokens.q8_row_parts().is_none() {
                    fill_h(&mut h, self);
                    h_ready = true;
                }
                let ids_for_embed = match input {
                    PrefillIn::Ids(ids) => (!h_ready && li == 0).then_some(ids),
                    PrefillIn::Hidden(_) => None,
                };
                let end = self.chunk_run_gpu(li, &mut h, b, start_pos, ids_for_embed, upto_excl);
                if end > li {
                    h_ready = true;
                    chunk_skip_until = end;
                    // Looped Transformer: the graph stopped at a loop
                    // boundary — apply final norm before the next iteration.
                    if self.is_loop_end(end - 1) && end < self.num_layers {
                        for bi in 0..b {
                            let normed = inference::rms_norm(
                                &h[bi * hs..(bi + 1) * hs],
                                &self.weights.final_norm,
                                eps,
                                norm_style,
                            );
                            h[bi * hs..(bi + 1) * hs].copy_from_slice(&normed);
                        }
                    }
                    continue;
                }
            }
            if !h_ready {
                fill_h(&mut h, self);
                h_ready = true;
            }
            let lw = &self.weights.layers[self.phys_layer(li)];
            // ── attention ──
            match &lw.attn {
                AttnKind::Kda(w) => {
                    // Projections batched, recurrence sequential.
                    let cfg = self.kda_cfg.expect("kda layer without kda_cfg");
                    let mut normed = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        inference::rms_norm_into(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                            &mut normed[bi * hs..(bi + 1) * hs],
                        );
                    }
                    let attn = crate::linear_core::kda_forward_batch(
                        &normed,
                        b,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        pool.as_deref(),
                    );
                    for (dst, &a) in h.iter_mut().zip(&attn) {
                        *dst += a;
                    }
                }
                AttnKind::LinearGdn(w) => {
                    // Projections batched, recurrence sequential.
                    let cfg = self.gdn_cfg.expect("gdn layer without gdn_cfg");
                    let mut normed = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        let r = inference::rms_norm(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                        );
                        normed[bi * hs..(bi + 1) * hs].copy_from_slice(&r);
                    }
                    let attn = crate::linear_core::gdn_forward_batch(
                        &normed,
                        b,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        pool.as_deref(),
                    );
                    for (dst, &a) in h.iter_mut().zip(&attn) {
                        *dst += a;
                    }
                }
                AttnKind::ShortConv(w) => {
                    // Projections batched over the chunk; the conv walks the
                    // contiguous positions in order (same ring as decode).
                    let cfg = self
                        .short_conv_cfg
                        .expect("short-conv layer without short_conv_cfg");
                    let mut normed = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        inference::rms_norm_into(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                            &mut normed[bi * hs..(bi + 1) * hs],
                        );
                    }
                    let attn = short_conv_forward_batch(
                        &normed,
                        b,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        pool.as_deref(),
                    );
                    for (dst, &a) in h.iter_mut().zip(&attn) {
                        *dst += a;
                    }
                }
                AttnKind::Mla(w) => {
                    // Per-position prefill (correctness first; latent
                    // batching is a later optimization).
                    let inv_freq_l = self.layer_inv_freq(li);
                    let rs = self.layer_rope_scale(li);
                    let mut normed = vec![0.0f32; hs];
                    for bi in 0..b {
                        inference::rms_norm_into(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                            &mut normed,
                        );
                        let ao = mla_attention(
                            w,
                            &normed,
                            &mut self.kv_cache.layers[li],
                            start_pos + bi,
                            &inv_freq_l,
                            rs,
                            eps,
                            pool.as_deref(),
                        );
                        for (dst, &a) in h[bi * hs..(bi + 1) * hs].iter_mut().zip(&ao) {
                            *dst += a;
                        }
                    }
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate,
                    bias,
                } => {
                    // Chunk-GEMM QKV/O; per-position causal attention
                    // inside (roadmap §3 P0 — full-attention prefill no
                    // longer re-reads the projection weights b times).
                    let mut normed = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        inference::rms_norm_into(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                            &mut normed[bi * hs..(bi + 1) * hs],
                        );
                    }
                    let inv_freq_l = self.layer_inv_freq(li);
                    let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                    let cfg = QwenAttnCfg {
                        num_heads: self.layer_num_heads(li),
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position: start_pos,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        softcap: self.attn_softcap,
                        window: self.layer_window(li),
                        v_norm: self.attn_v_norm,
                        qk_norm_after_rope: self.qk_norm_after_rope,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        softplus_gate: softplus_gate
                            .as_ref()
                            .map(|(gate, per_head)| (gate, *per_head)),
                        rope_scale: self.layer_rope_scale(li),
                        bias: bias
                            .as_ref()
                            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        rms_eps: eps,
                        norm_style,
                        pool: pool.as_deref(),
                    };
                    let mut attn = attention::qwen_attention_batch(
                        &normed,
                        b,
                        wq,
                        wk,
                        wv,
                        wo,
                        &mut self.kv_cache.layers[li],
                        &cfg,
                    );
                    if let Some(w) = &lw.attn_out_norm {
                        for bi in 0..b {
                            inference::rms_norm_into(
                                &attn[bi * hs..(bi + 1) * hs],
                                w,
                                eps,
                                norm_style,
                                &mut normed[bi * hs..(bi + 1) * hs],
                            );
                        }
                        attn.copy_from_slice(&normed);
                    }
                    for (dst, &a) in h.iter_mut().zip(&attn) {
                        *dst += a;
                    }
                }
                AttnKind::Linear(w) => {
                    for bi in 0..b {
                        let normed = inference::rms_norm(
                            &h[bi * hs..(bi + 1) * hs],
                            &lw.input_norm,
                            eps,
                            norm_style,
                        );
                        vmf_phase_forward(
                            &normed,
                            w,
                            &self.vmf_cfg.expect("linear layer without vmf_cfg"),
                            &mut self.kv_cache.layers[li].linear_state,
                            pool.as_deref(),
                        )
                        .iter()
                        .enumerate()
                        .for_each(|(i, &a)| h[bi * hs + i] += a);
                    }
                }
            }

            // ── FFN batched ──
            let lw = &self.weights.layers[self.phys_layer(li)];
            let mut post = vec![0.0f32; b * hs];
            for bi in 0..b {
                let r =
                    inference::rms_norm(&h[bi * hs..(bi + 1) * hs], &lw.post_norm, eps, norm_style);
                post[bi * hs..(bi + 1) * hs].copy_from_slice(&r);
            }
            // A restrictive per-visit FFN row lands on the activations
            // inside the dense arm; an all-open row costs nothing.
            let mask_row = task_mask
                .filter(|m| m.ffn_active_count(li) < self.intermediate_size)
                .and_then(|m| m.ffn_masks.get(li))
                .map(|v| v.as_slice());
            let mut ffn = match &lw.ffn {
                FfnKind::Dense(d) if !d.segs.is_empty() => {
                    tube_ffn(d, &post, b, pool.as_deref(), mask_row)
                }
                FfnKind::Dense(d) => dense_ffn_batch(d, &post, b, pool.as_deref(), mask_row),
                FfnKind::Moe(m) => moe_ffn_batch(m, &post, b, hs, pool.as_deref(), None),
                // Dual-branch layers run per position (the expert branch
                // reads the raw residual — nothing to batch yet).
                FfnKind::DenseMoe(dm) => {
                    let mut out = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        let r = dense_moe_ffn(
                            dm,
                            &post[bi * hs..(bi + 1) * hs],
                            &h[bi * hs..(bi + 1) * hs],
                            eps,
                            norm_style,
                            pool.as_deref(),
                        );
                        out[bi * hs..(bi + 1) * hs].copy_from_slice(&r);
                    }
                    out
                }
            };
            if let Some(w) = &lw.ffn_out_norm {
                for bi in 0..b {
                    inference::rms_norm_into(
                        &ffn[bi * hs..(bi + 1) * hs],
                        w,
                        eps,
                        norm_style,
                        &mut post[bi * hs..(bi + 1) * hs],
                    );
                }
                ffn.copy_from_slice(&post);
            }
            for (dst, &f) in h.iter_mut().zip(&ffn) {
                *dst += f;
            }
            if let Some(sc) = lw.layer_scale {
                for v in h.iter_mut() {
                    *v *= sc;
                }
            }
            if let Ok(tp) = std::env::var("CMF_TRACE_POS") {
                if let Ok(t) = tp.parse::<usize>() {
                    if t >= start_pos && t < start_pos + b {
                        let bi = t - start_pos;
                        let row = &h[bi * hs..(bi + 1) * hs];
                        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        eprintln!(
                            "BATCH pos {t} after layer {li}: |h| = {n:.6} h0 {:.6} h1 {:.6}",
                            row[0], row[1]
                        );
                    }
                }
            }
            // CMF_DEBUG_LAYERS=1: per-layer hidden-state health of the
            // LAST prompt position — the knife for "which layer type
            // breaks first" on a new architecture.
            if std::env::var("CMF_DEBUG_LAYERS").is_ok() {
                let row = &h[(b - 1) * hs..b * hs];
                let rms =
                    (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hs as f64).sqrt();
                let mx = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                eprintln!(
                    "layer {li:>3} {:>10} ffn={:<5} rms={rms:>12.4} max={mx:>12.4}",
                    match &self.weights.layers[self.phys_layer(li)].attn {
                        AttnKind::LinearGdn(_) => "gdn",
                        AttnKind::Linear(_) => "vmf",
                        AttnKind::ShortConv(_) => "conv",
                        _ => "attn",
                    },
                    match &lw.ffn {
                        FfnKind::Moe(_) => "moe",
                        FfnKind::Dense(_) => "dense",
                        FfnKind::DenseMoe(_) => "dense+moe",
                    },
                );
            }
            // Looped Transformer: apply final norm at the end of each loop iteration.
            if self.is_loop_end(li) && li + 1 < self.num_layers {
                for bi in 0..b {
                    let normed = inference::rms_norm(
                        &h[bi * hs..(bi + 1) * hs],
                        &self.weights.final_norm,
                        eps,
                        norm_style,
                    );
                    h[bi * hs..(bi + 1) * hs].copy_from_slice(&normed);
                }
            }
            if std::env::var("CMF_TRACE_H").is_ok() {
                let n = h[..hs].iter().map(|v| v.abs()).sum::<f32>() / hs as f32;
                let mx = h[..hs].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                eprintln!(
                    "layer {li}: mean|h|={n:.4} max|h|={mx:.2} scale={:?}",
                    lw.layer_scale
                );
            }
        }
        crate::gpu::set_layer(-1); // lm_head/final ops outside layer-split
        // A batched span owns a complete set of positions. Publish any
        // collecting→sealed transition only after every layer has finished;
        // callers that cross into serial/device work must see the new epoch
        // before this function returns.
        self.o1_progress();
        h
    }

    /// Embed a single token.
    fn embed_single(&self, id: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; self.hidden_size];
        if (id as usize) < self.weights.embed_tokens.rows() {
            self.weights.embed_tokens.row_f32(id as usize, &mut out);
        }
        if self.embed_multiplier != 1.0 {
            for v in out.iter_mut() {
                *v *= self.embed_multiplier;
            }
        }
        // DeepSeek-V4's hash layers route by TOKEN ID, so the id has to
        // reach the forward. It rides in slot 0 (the forward re-reads the
        // real embedding itself from the table).
        if self.dsv4.is_some() || self.dsv41.is_some() || self.qwen4_exp.is_some() {
            let mut v = vec![0.0f32; self.hidden_size.max(1)];
            v[0] = id as f32;
            return v;
        }
        // Gemma-3n: the per-layer-embedding half needs the token ID, so
        // it rides appended to the embedding; the g3n forward splits it.
        if let Some(b) = &self.g3n {
            return b.0.extend_embedding(id, &out, self.pool.as_deref());
        }
        out
    }

    /// A run of consecutive prefill layers on the GPU for the whole
    /// chunk (default-on under CMF_GPU=1; CMF_GPU_CHUNK=0 disables).
    /// Eligibility per layer: q8_row weights, plain full attention
    /// (no output gate), F32 KV, no o1/masks/gemma extras. Returns the
    /// first layer index NOT processed (== `li0` when the run is empty).
    #[cfg(target_os = "macos")]
    fn chunk_run_gpu(
        &mut self,
        li0: usize,
        h: &mut [f32],
        b: usize,
        pos0: usize,
        embed_ids: Option<&[u32]>,
        cap: usize,
    ) -> usize {
        // (The old streaming attend needed a depth bound at ~1k; the
        // GEMM attention scales like the CPU path and lifted it.)
        // CMF_GPU_CHUNK=0 disables the graph.
        if !crate::gpu::enabled_here()
            || std::env::var("CMF_GPU_CHUNK")
                .map(|v| v == "0")
                .unwrap_or(false)
            || b < 32
            || self.swa.is_some()
            || self.global_attn.is_some()
            // Collection owns the exact Q trace and boundary conversion;
            // this chunk graph appends dense KV without feeding that trace.
            || self.o1_active()
            || self.attn_v_norm
            || (self.attn_scale - 1.0 / (self.head_dim as f32).sqrt()).abs() > 1e-9
        {
            return li0;
        }
        let Some(model) = self.model.clone() else {
            return li0;
        };
        let inv_freq = self.inv_freq.clone();
        let (nh, nkv, hd, hs) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
        );
        // Collect the longest run of consecutive eligible layers.
        // Looped Transformer: stop at the loop boundary so the CPU can
        // apply loop_final_norm between iterations.
        let loop_end = if self.loop_final_norm {
            ((li0 / self.physical_layers) + 1) * self.physical_layers
        } else {
            self.num_layers
        };
        let mut layers: Vec<crate::gpu_metal::ChunkLayer> = Vec::new();
        let mut stored_at: Vec<usize> = Vec::new();
        for li in li0..self.num_layers.min(loop_end).min(cap) {
            let lw = &self.weights.layers[self.phys_layer(li)];
            if lw.attn_out_norm.is_some() || lw.ffn_out_norm.is_some() || lw.layer_scale.is_some() {
                break;
            }
            let AttnKind::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                output_gate: false,
                softplus_gate: None,
                bias,
            } = &lw.attn
            else {
                break;
            };
            let FfnKind::Dense(d) = &lw.ffn else { break };
            if d.act != Act::Silu || !d.segs.is_empty() {
                break;
            }
            // q8_row (row_scale populated), or q4_tiled / q4tp (row_scale
            // empty — their scales are in the payload). Mixing across the
            // seven projections of one layer is fine; the encoder branches
            // per weight on the tensor's dtype. Anything else refuses.
            fn cw(t: &QTensor) -> Option<(usize, usize, usize, &[f32])> {
                t.q8_row_parts()
                    .or_else(|| t.q4t_parts().map(|(i, r, c)| (i, r, c, &[][..])))
                    .or_else(|| t.q4tp_parts().map(|(i, r, c)| (i, r, c, &[][..])))
            }
            let parts = (
                cw(wq),
                cw(wk),
                cw(wv),
                cw(wo),
                cw(&d.gate_proj),
                cw(&d.up_proj),
                cw(&d.down_proj),
            );
            let (Some(pq), Some(pk), Some(pv), Some(po), Some(pg), Some(pu), Some(pd)) = parts
            else {
                break;
            };
            let layer = &self.kv_cache.layers[li];
            if layer.mode != crate::kv_cache::KvMode::F32 || layer.o1.is_some() {
                break;
            }
            stored_at.push(layer.head_len(0));
            layers.push(crate::gpu_metal::ChunkLayer {
                model: &model,
                kv_id: self.graph_kv_id,
                layer: li,
                wq: pq,
                wk: pk,
                wv: pv,
                wo: po,
                gate: pg,
                up: pu,
                down: pd,
                input_norm: &lw.input_norm,
                post_norm: &lw.post_norm,
                bias: bias
                    .as_ref()
                    .map(|(a, bb, cc)| (a.as_slice(), bb.as_slice(), cc.as_slice())),
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                inv_freq: &inv_freq,
                rd: self.rotary_dim,
                nh,
                nkv,
                hd,
                hs,
                inter: d.gate_proj.rows(),
                gemma: matches!(self.norm_style, cortiq_core::NormStyle::Gemma),
                late_qk_norm: self.qk_norm_after_rope,
                eps: self.rms_eps as f32,
            });
        }
        if layers.is_empty() {
            return li0;
        }
        let row = nkv * hd;
        let mut store: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = stored_at
            .iter()
            .map(|&st| (vec![0f32; b * row], vec![0f32; b * row], vec![0f32; st + b]))
            .collect();
        let mut io: Vec<crate::gpu_metal::ChunkIo> = Vec::with_capacity(layers.len());
        for (i, (ok, ov, oi)) in store.iter_mut().enumerate() {
            let li = layers[i].layer;
            let layer = &self.kv_cache.layers[li];
            io.push(crate::gpu_metal::ChunkIo {
                cpu_stored: stored_at[i],
                cpu_k: (0..nkv).map(|g| layer.head_keys(g)).collect(),
                cpu_v: (0..nkv).map(|g| layer.head_values(g)).collect(),
                out_k: ok,
                out_v: ov,
                imp: oi,
            });
        }
        let n_run = layers.len();
        let last = layers.last().map(|l| l.layer + 1).unwrap_or(li0);
        // Device-side embedding when the run starts the model and the
        // embedding matrix is q8_row-mapped.
        let ep = embed_ids.and_then(|ids| {
            self.weights
                .embed_tokens
                .q8_row_parts()
                .map(|(idx, rows, _c, rs)| crate::gpu_metal::ChunkEmbed {
                    idx,
                    rows,
                    row_scale: rs,
                    ids,
                    mult: self.embed_multiplier,
                })
        });
        if embed_ids.is_some() && ep.is_none() {
            return li0;
        }
        if !crate::gpu_metal::chunk_run_gpu(&layers, &mut io, h, b, pos0, ep.as_ref()) {
            return li0;
        }
        drop(io);
        drop(layers);
        // CPU caches stay the owners of record: append the chunk rows
        // and bank the importance masses per layer.
        for (i, (ok, ov, oi)) in store.iter().enumerate().take(n_run) {
            let li = li0 + i;
            let layer = &mut self.kv_cache.layers[li];
            for bi in 0..b {
                layer.append(
                    &ok[bi * row..(bi + 1) * row],
                    &ov[bi * row..(bi + 1) * row],
                    &[],
                );
            }
            layer.accumulate_imp(oi);
        }
        last
    }

    /// Is layer `li` a sliding-window (local-RoPE) layer? Gemma-3:
    /// every `pattern`-th layer is global, the rest are local.
    fn layer_is_local(&self, li: usize) -> bool {
        if let Some(layers) = &self.sliding_layers {
            return layers.get(li).copied().unwrap_or(false);
        }
        match self.swa {
            Some((_, pattern)) => (li + 1) % pattern.max(1) != 0,
            None => false,
        }
    }

    /// The RoPE table for layer `li` (local layers may have their own;
    /// Gemma-4 global layers use the proportional padded table).
    fn layer_inv_freq(&self, li: usize) -> std::sync::Arc<Vec<f32>> {
        if self.layer_is_local(li) {
            if let Some(f) = &self.inv_freq_local {
                return f.clone();
            }
        } else if let Some(f) = &self.inv_freq_global {
            return f.clone();
        }
        self.inv_freq.clone()
    }

    /// The attend window for layer `li` (None = full context).
    fn layer_window(&self, li: usize) -> Option<usize> {
        self.swa
            .and_then(|(w, _)| self.layer_is_local(li).then_some(w))
    }

    fn layer_num_heads(&self, li: usize) -> usize {
        self.attention_heads_per_layer
            .as_ref()
            .and_then(|v| v.get(li).copied())
            .unwrap_or(self.num_heads)
    }

    fn layer_rope_scale(&self, li: usize) -> f32 {
        if self.layer_is_local(li) {
            self.rope_scale_local
        } else {
            self.rope_scale
        }
    }

    /// Attention geometry of layer `li`: (num_kv_heads, head_dim,
    /// rotary_dim). Gemma-4 global layers override all three.
    fn layer_geom(&self, li: usize) -> (usize, usize, usize) {
        if !self.layer_is_local(li) {
            if let Some((ghd, gkv)) = self.global_attn {
                return (gkv, ghd, ghd);
            }
        }
        (
            self.num_kv_heads,
            self.head_dim,
            if self.layer_is_local(li) {
                self.rotary_dim_local.unwrap_or(self.rotary_dim)
            } else {
                self.rotary_dim
            },
        )
    }

    /// Forward one position through all layers (hybrid dispatch).
    fn forward_layers(
        &mut self,
        hidden: &[f32],
        position: usize,
        task_mask: Option<&TaskMask>,
    ) -> Vec<f32> {
        let out = self.forward_layers_upto(hidden, position, task_mask, None);
        self.o1_progress();
        out
    }

    // ── Network pipeline-split building blocks (coordinator/worker) ──
    // A remote worker owns layers [from ..= upto] and their KV; the
    // coordinator owns the rest plus embed / final norm / head. Attention
    // causality is per-layer, so a whole prompt's boundary hiddens ship
    // as one batch and decode ships one vector per token.

    /// Embed one token id (embed multiplier applied).
    pub fn embed_id(&self, id: u32) -> Vec<f32> {
        self.embed_single(id)
    }

    /// Refuse the archs/modes whose forward cannot be cut at a layer
    /// boundary. Loud by design: a split that silently changed the math
    /// would be a chimera.
    pub fn split_supported(&self) -> Result<(), String> {
        if self.dsv4.is_some() {
            return Err(
                "network split: DeepSeek-V4 runs its own fused stack (not splittable yet)".into(),
            );
        }
        if self.dsv41.is_some() {
            return Err(
                "network split: DeepSeek-V4.1 owns the shared CED/CSA2 state (not splittable)"
                    .into(),
            );
        }
        if self.qwen4_exp.is_some() {
            return Err(
                "network split: Qwen3.8-Flash-Next hyper/QSA stack is not splittable yet".into(),
            );
        }
        if self.g3n.is_some() {
            return Err(
                "network split: Gemma-3n runs its own AltUp stack (not splittable yet)".into(),
            );
        }
        Ok(())
    }

    /// Forward `hidden` through layers [from ..= upto] at `position`,
    /// appending those layers' KV/state. Both split sides call this
    /// over their own range; a task mask applies to the span's own
    /// layers (each side masks what it runs).
    pub fn forward_span(
        &mut self,
        hidden: &[f32],
        position: usize,
        from: usize,
        upto: usize,
        task_mask: Option<&TaskMask>,
    ) -> Result<Vec<f32>, String> {
        self.split_supported()?;
        if from > upto || upto >= self.num_layers {
            return Err(format!(
                "forward_span: layer range {from}..={upto} outside 0..{}",
                self.num_layers
            ));
        }
        if hidden.len() != self.hidden_size {
            return Err(format!(
                "forward_span: hidden len {} ≠ hidden_size {}",
                hidden.len(),
                self.hidden_size
            ));
        }
        let out = self.forward_layers_span(hidden, position, task_mask, from, Some(upto));
        self.o1_progress();
        if self
            .graph_failed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.cancel
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.clear_sequence_state();
            return Err("forward_span: deferred O(1) transition failed".into());
        }
        Ok(out)
    }

    /// Final norm + lm_head over a boundary hidden (the final-logit
    /// softcap is applied by lm_head_forward itself).
    pub fn logits_from_hidden(&mut self, hidden: &[f32]) -> Vec<f32> {
        let normed = inference::rms_norm(
            hidden,
            &self.weights.final_norm,
            self.rms_eps,
            self.norm_style,
        );
        self.lm_head_forward(&normed)
    }

    /// Sample the next token with this pipeline's sampler state.
    pub fn sample_next(&mut self, logits: &[f32], past_tokens: &[u32]) -> u32 {
        sampler::sample_with_scratch(
            logits,
            &self.sampler_config,
            past_tokens,
            &mut self.rng,
            &mut self.sampler_scratch,
        )
    }

    /// Fresh sequence: clear KV, reuse history and device mirrors.
    pub fn reset_session(&mut self) {
        self.clear_sequence_state();
    }

    /// Batched span prefill from token ids (coordinator side): embed +
    /// layers [0 ..= upto]; returns the boundary hiddens of ALL positions
    /// (ids.len() × hidden). Rides the same layer-major machinery as the
    /// local prefill; falls back to the per-position walk under
    /// CMF_PREFILL=seq.
    pub fn prefill_span_ids(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        upto: usize,
        task_mask: Option<&TaskMask>,
    ) -> Result<Vec<f32>, String> {
        self.split_supported()?;
        if upto >= self.num_layers {
            return Err(format!(
                "prefill_span_ids: upto {upto} outside 0..{}",
                self.num_layers
            ));
        }
        // Same predicate as the whole-stack prefill: a span whose GDN
        // state lives on the device must walk positions through the
        // graph, not through the batched CPU span.
        if self.can_prefill_batched() && !self.graph_prefill_preferred() {
            let out =
                self.prefill_batch_span(PrefillIn::Ids(ids), start_pos, task_mask, 0, upto + 1);
            self.check_o1_progress_failure("prefill_span_ids")?;
            Ok(out)
        } else {
            let hs = self.hidden_size;
            let mut out = Vec::with_capacity(ids.len() * hs);
            for (i, &id) in ids.iter().enumerate() {
                let emb = self.embed_id(id);
                out.extend_from_slice(&self.forward_span(
                    &emb,
                    start_pos + i,
                    0,
                    upto,
                    task_mask,
                )?);
            }
            Ok(out)
        }
    }

    /// Batched span prefill from boundary hiddens (worker side): layers
    /// [from ..= upto] for every position in the batch; returns the batch.
    pub fn prefill_span_hidden(
        &mut self,
        hidden: &[f32],
        start_pos: usize,
        from: usize,
        upto: usize,
        task_mask: Option<&TaskMask>,
    ) -> Result<Vec<f32>, String> {
        self.split_supported()?;
        let hs = self.hidden_size;
        if hidden.is_empty() || hidden.len() % hs != 0 {
            return Err(format!(
                "prefill_span_hidden: {} floats is not a multiple of hidden {hs}",
                hidden.len()
            ));
        }
        if from > upto || upto >= self.num_layers {
            return Err(format!(
                "prefill_span_hidden: layer range {from}..={upto} outside 0..{}",
                self.num_layers
            ));
        }
        if self.can_prefill_batched() && !self.graph_prefill_preferred() {
            let out = self.prefill_batch_span(
                PrefillIn::Hidden(hidden),
                start_pos,
                task_mask,
                from,
                upto + 1,
            );
            self.check_o1_progress_failure("prefill_span_hidden")?;
            Ok(out)
        } else {
            let b = hidden.len() / hs;
            let mut out = Vec::with_capacity(hidden.len());
            for i in 0..b {
                let h = self.forward_span(
                    &hidden[i * hs..(i + 1) * hs],
                    start_pos + i,
                    from,
                    upto,
                    task_mask,
                )?;
                out.extend_from_slice(&h);
            }
            Ok(out)
        }
    }

    /// Build the whole-token wgpu graph for a pure-attention q1 model (every
    /// layer Full q1 + dense q1 FFN, no gate/bias). Returns the post-stack
    /// hidden (caller does final norm + lm_head), or None to fall back.
    fn try_token_graph_wgpu(
        &self,
        hidden: &[f32],
        position: usize,
        logits_out: &mut Vec<f32>,
        layers_run: &mut usize,
    ) -> Option<Result<Vec<f32>, ()>> {
        self.try_token_graph_wgpu_steps(
            hidden,
            position,
            logits_out,
            1,
            None,
            Some(layers_run),
            0,
            self.num_layers,
        )
    }

    /// The span twin (network split): the graph covers [from..upto_excl)
    /// — one submit per SEGMENT per token. lm_head folds in only when
    /// the span reaches the last layer.
    fn try_token_graph_wgpu_span(
        &self,
        hidden: &[f32],
        position: usize,
        logits_out: &mut Vec<f32>,
        from: usize,
        upto_excl: usize,
        layers_run: &mut usize,
    ) -> Option<Result<Vec<f32>, ()>> {
        self.try_token_graph_wgpu_steps(
            hidden,
            position,
            logits_out,
            1,
            None,
            Some(layers_run),
            from,
            upto_excl,
        )
    }

    /// Greedy burst: forward `t_next` and let the device pick + re-embed
    /// the next k−1 tokens — k frames, ONE submit, k ids back. The ZML
    /// trade, on wgpu. None ⇒ caller keeps the per-token path.
    fn try_multi_burst(&self, t_next: u32, position: usize, k: usize) -> Option<Vec<u32>> {
        if self.o1_active() || self.attn_softcap > 0.0 {
            return None;
        }
        let graph_on = crate::gpu::wgpu_graph_on(crate::gpu::GraphPhase::Decode);
        if !graph_on || crate::gpu::graph_unsupported() {
            // Same memo as the decode site: this path builds the very
            // same graph, so a model it cannot build for must not be
            // walked again here either. Missing this guard was worth
            // 2.5x on an Adreno — 0.361 tok/s against 0.905 — because
            // the burst retried per token what decode had already given
            // up on.
            return None;
        }
        let emb = self.embed_single(t_next);
        let mut lg = Vec::new();
        let mut ids = Vec::new();
        match self.try_token_graph_wgpu_steps(
            &emb,
            position,
            &mut lg,
            k,
            Some(&mut ids),
            None,
            0,
            self.num_layers,
        ) {
            Some(Ok(_)) => {}
            Some(Err(())) => {
                // Preserve the backend's post-admission failure through the
                // Option-based burst API.  The decode caller consumes this
                // flag and clears the sequence instead of falling through
                // to a stale CPU recurrent state.
                self.graph_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
            None => return None,
        }
        (ids.len() == k).then_some(ids)
    }

    /// Multi-step greedy: k whole frames in ONE submit, argmax and re-embed
    /// on the device. `ids_out` receives the k winner ids; the hidden/logits
    /// outputs are NOT produced in that mode.
    fn try_token_graph_wgpu_steps(
        &self,
        hidden: &[f32],
        position: usize,
        logits_out: &mut Vec<f32>,
        steps: usize,
        ids_out: Option<&mut Vec<u32>>,
        layers_run: Option<&mut usize>,
        from: usize,
        upto_excl: usize,
    ) -> Option<Result<Vec<f32>, ()>> {
        // O(1) Nyström decode runs off the sealed state, not the KV cache the
        // graph mirrors — never take the graph while o1 is active.
        let o1_gpu = std::env::var("CMF_O1_GPU").as_deref() == Ok("1");
        if (self.o1_active() && !o1_gpu) || self.attn_softcap > 0.0 {
            // Softcapped scores have no graph kernel yet — CPU owns them.
            // o1 rides the graph only behind CMF_O1_GPU=1 while the port
            // proves itself; without it the CPU path owns o1 as before.
            return None;
        }
        // Per-layer sealed o1 state for the graph. During prefill the
        // state is still Collecting -> views are None -> the graph
        // refuses below and the CPU prefill records the q trace and
        // seals, exactly as the o1 design requires.
        let o1_views: Vec<Option<Vec<crate::nystrom::O1DeviceView<'_>>>> = (from..upto_excl)
            .map(|li| {
                if !o1_gpu {
                    return None;
                }
                self.kv_cache.layers[self.phys_layer(li)].o1_views()
            })
            .collect();
        if self.o1_active() && o1_gpu {
            // Any o1 layer not sealed (or degenerate exact-only) keeps the
            // whole token on the CPU: half-graph forwards would desync.
            let want: usize = (from..upto_excl)
                .filter(|li| self.kv_cache.layers[self.phys_layer(*li)].o1.is_some())
                .count();
            let have = o1_views.iter().filter(|v| v.is_some()).count();
            if want == 0 || have != want {
                // The silent twin of the gpu-side o1 gates, found the
                // same way: a 15x decode drop with an empty log. Views
                // stay None until the layer's state SEALS, so `have`
                // lagging `want` early in a run is the o1 design working
                // — but it must say so, or the next reader spends a
                // night proving the kernels innocent.
                // On CHANGE, not once: the first decline is the legal
                // unsealed prefill, and a once-print buries the state
                // that matters — what the count reads AFTER the seal.
                use std::sync::atomic::{AtomicUsize, Ordering};
                static LAST: AtomicUsize = AtomicUsize::new(usize::MAX);
                let code = have * 1000 + want;
                if LAST.swap(code, Ordering::Relaxed) != code {
                    tracing::warn!(
                        "o1 graph: {have} of {want} layers sealed — per-op until all seal"
                    );
                }
                return None;
            }
        }
        let nh = self.num_heads;
        let (nkv, hd, rd) = self.layer_geom(0);
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        let mut layers = Vec::with_capacity(upto_excl - from);
        let mut model = None;
        let dbg = std::env::var("CMF_GRAPH_DEBUG").is_ok();
        fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
            if let Some((m, i, kind, rs)) = t
                .graph_weight()
                .or_else(|| t.graph_weight_descriptor())
            {
                let name = &m.tensors[i].name;
                let prism = if crate::prism::is_inverse_embedding(m, name) {
                    crate::gpu::GraphPrismOp::InverseEmbedding
                } else if crate::prism::is_forward_weight(m, name) {
                    crate::gpu::GraphPrismOp::Forward
                } else {
                    crate::gpu::GraphPrismOp::None
                };
                return Some(crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                    prism,
                    affine: crate::prism::is_affine_target(m, name),
                });
            }
            // Small unquantized projections (GDN in_proj_a/b) stay f32.
            match t.as_f32() {
                Some(d) => Some(crate::gpu::GraphW {
                    idx: 0,
                    kind: 4,
                    row_scale: &[],
                    data: d,
                    prism: crate::gpu::GraphPrismOp::None,
                    affine: false,
                }),
                None => {
                    if std::env::var_os("CMF_BATCH_DEBUG").is_some() {
                        eprintln!("batch graph: weight has no graph/f32 representation");
                    }
                    None
                }
            }
        }
        for li in from..upto_excl {
            let lw = &self.weights.layers[self.phys_layer(li)];
            if dbg {
                let ak = match &lw.attn {
                    AttnKind::Mla(_) => "Mla".into(),
                    AttnKind::Full {
                        output_gate, bias, ..
                    } => format!("Full gate={output_gate} bias={}", bias.is_some()),
                    AttnKind::LinearGdn(_) => "LinearGdn".into(),
                    AttnKind::Kda(_) => "Kda".into(),
                    AttnKind::Linear(_) => "Linear".into(),
                    AttnKind::ShortConv(_) => "ShortConv".into(),
                };
                let fk = match &lw.ffn {
                    FfnKind::Dense(_) => "Dense",
                    FfnKind::Moe(_) => "Moe",
                    FfnKind::DenseMoe(_) => "DenseMoe",
                };
                eprintln!("graph L{li}: attn={ak} ffn={fk}");
            }
            let gffn = match &lw.ffn {
                FfnKind::DenseMoe(_) => return None, // dual branch: CPU path
                // A tube layer is several matrices, not one — the
                // whole-layer graph has no shape for it yet.
                FfnKind::Dense(d) if !d.segs.is_empty() => return None,
                FfnKind::Dense(d) => crate::gpu::GraphFfn::Dense {
                    gate: gw(&d.gate_proj)?,
                    up: gw(&d.up_proj)?,
                    down: gw(&d.down_proj)?,
                },
                FfnKind::Moe(m) => {
                    // Adaptive τ and expert masks keep the CPU path, where
                    // they are implemented. Sigmoid routing with a selection
                    // bias (LFM2-MoE / DeepSeek noaux_tc), a routed scale ≠ 1
                    // and an UNGATED shared expert (HunYuan hy_v3: ×2.826 on
                    // the routed mix, the shared expert at weight 1) are all
                    // graphed — before, every such token fell to the per-op
                    // path whole (145 submits/token on Hy-MT2-30B-A3B).
                    if m.route_tau.is_some() || m.mask.is_some() {
                        return None;
                    }
                    let shared = m.shared.as_ref();
                    let has_shared = shared.is_some();
                    let shared_gated = matches!(shared, Some((_, Some(_))));
                    let sgate = match shared {
                        Some((_, Some(sg))) => gw(sg)?,
                        // No gate (hy_v3) or no shared expert at all: the
                        // router weight stands in so the plumbing stays
                        // total; the select kernels pin weight 1 or skip.
                        _ => gw(&m.router)?,
                    };
                    let router = gw(&m.router)?;
                    // The resident MoE kernels do not yet carry the
                    // descriptor-aware transform through router/shared-gate
                    // selection.  Refuse the complete layer instead of
                    // scoring with an untransformed Prism plane (the dense
                    // path has an explicit FWHT boundary below).
                    if router.prism != crate::gpu::GraphPrismOp::None
                        || sgate.prism != crate::gpu::GraphPrismOp::None
                        || router.affine
                        || sgate.affine
                    {
                        tracing::warn!(
                            "resident MoE declined: Prism/affine router or shared gate transform is not implemented"
                        );
                        return None;
                    }
                    let inter = m.experts.first()?.gate_proj.rows();
                    let mut experts = Vec::with_capacity(m.experts.len() + 1);
                    // q4t or q4tp, but not both in one layer — the kernels
                    // are picked per layer, not per expert.
                    let mut q4tp: Option<bool> = None;
                    // The mixed 2-bit profile: q2tp gate/up over a q4tp
                    // down. Uniform across the layer, like `q4tp` itself.
                    let mut gu_q2: Option<bool> = None;
                    for e in m.experts.iter().chain(shared.map(|(se, _)| se)) {
                        if !matches!(e.act, Act::Silu)
                            || e.gate_proj.rows() != inter
                            || e.up_proj.rows() != inter
                        {
                            return None;
                        }
                        // Expert tensors are packed into one resident buffer
                        // and the MoE kernels have no transform slot per
                        // expert.  Keep the CPU/per-op owner for Prism or
                        // affine experts rather than silently using raw bytes.
                        for expert_weight in [&e.gate_proj, &e.up_proj, &e.down_proj] {
                            let Some((em, ei, _, _)) = expert_weight
                                .graph_weight()
                                .or_else(|| expert_weight.graph_weight_descriptor())
                            else {
                                return None;
                            };
                            let name = &em.tensors[ei].name;
                            if crate::prism::is_forward_weight(em, name)
                                || crate::prism::is_inverse_embedding(em, name)
                                || crate::prism::is_affine_target(em, name)
                            {
                                tracing::warn!(
                                    "resident MoE declined: expert Prism/affine transform is not implemented"
                                );
                                return None;
                            }
                        }
                        let (mm, gi, ui, di, is_p, is_q2) = match e.gate_proj.mapped_q4t() {
                            Some((mm, gi)) => (
                                mm,
                                gi,
                                e.up_proj.mapped_q4t()?.1,
                                e.down_proj.mapped_q4t()?.1,
                                false,
                                false,
                            ),
                            None => match e.gate_proj.mapped_q2tp() {
                                Some((mm, gi)) => (
                                    mm,
                                    gi,
                                    e.up_proj.mapped_q2tp()?.1,
                                    e.down_proj.mapped_q4tp()?.1,
                                    true,
                                    true,
                                ),
                                None => {
                                    let (mm, gi) = e.gate_proj.mapped_q4tp()?;
                                    (
                                        mm,
                                        gi,
                                        e.up_proj.mapped_q4tp()?.1,
                                        e.down_proj.mapped_q4tp()?.1,
                                        true,
                                        false,
                                    )
                                }
                            },
                        };
                        if *q4tp.get_or_insert(is_p) != is_p || *gu_q2.get_or_insert(is_q2) != is_q2
                        {
                            // The shared expert rides in the same packed
                            // buffer as the routed ones, so a layer that
                            // mixes layouts cannot be indexed by one stride.
                            // Say so: the symptom is a whole model quietly
                            // running its MoE on the CPU.
                            tracing::warn!(
                                "MoE layer mixes expert layouts (q4tp={is_p}, q2tp gate/up={is_q2})                                  — every expert of a layer, INCLUDING the shared one, must share                                  a layout. The whole-token graph declines this layer."
                            );
                            return None;
                        }
                        model.get_or_insert_with(|| mm.clone());
                        experts.push((gi, ui, di));
                    }
                    crate::gpu::GraphFfn::Moe {
                        router,
                        shared_gate: sgate,
                        experts,
                        n_exp: m.experts.len(),
                        // CMF_TOPK_PROBE: timing probe only — output is WRONG.
                        // Fewer experts shrink the MoE arithmetic while the
                        // dispatch count stays identical, which is the only
                        // clean way to tell a launch-bound decode from a
                        // compute-bound one.
                        top_k: std::env::var("CMF_TOPK_PROBE")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .filter(|k| *k > 0 && *k <= m.top_k)
                            .unwrap_or(m.top_k),
                        inter,
                        norm_topk: m.norm_topk_prob,
                        q4tp: q4tp?,
                        gu_q2: gu_q2.unwrap_or(false),
                        sigmoid: m.router_sigmoid,
                        bias: m.expert_bias.as_deref(),
                        has_shared,
                        shared_gated,
                        route_scale: m.routed_scaling,
                    }
                }
            };
            let attn = match &lw.attn {
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate,
                    bias,
                } => {
                    if softplus_gate.is_some() || self.attention_heads_per_layer.is_some() {
                        return None;
                    }
                    let (m, _, _, _) = wq
                        .graph_weight()
                        .or_else(|| wq.graph_weight_descriptor())?;
                    model = Some(m.clone());
                    crate::gpu::GraphAttn::Full {
                        wq: gw(wq)?,
                        wk: gw(wk)?,
                        wv: gw(wv)?,
                        wo: gw(wo)?,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        late_qk_norm: self.qk_norm_after_rope,
                        bias: bias
                            .as_ref()
                            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        output_gate: *output_gate,
                        cpu_k: self.kv_cache.layers[li].k_heads(),
                        cpu_v: self.kv_cache.layers[li].v_heads(),
                    }
                }
                AttnKind::LinearGdn(w) => {
                    let cfg = self.gdn_cfg?;
                    let (m, _, _, _) = w
                        .in_proj_qkv
                        .graph_weight()
                        .or_else(|| w.in_proj_qkv.graph_weight_descriptor())?;
                    model = Some(m.clone());
                    crate::gpu::GraphAttn::Gdn {
                        qkv: gw(&w.in_proj_qkv)?,
                        z: gw(&w.in_proj_z)?,
                        a: gw(&w.in_proj_a)?,
                        b: gw(&w.in_proj_b)?,
                        out: gw(&w.out_proj)?,
                        conv1d: &w.conv1d,
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                        norm: &w.norm,
                        nv: cfg.num_v_heads,
                        nk: cfg.num_k_heads,
                        dk: cfg.key_head_dim,
                        dv: cfg.value_head_dim,
                        kk: cfg.conv_kernel,
                        cpu_state: &self.kv_cache.layers[self.phys_layer(li)].linear_state,
                    }
                }
                AttnKind::ShortConv(w) => {
                    let cfg = self.short_conv_cfg?;
                    let (m, _, _, _) = w
                        .in_proj
                        .graph_weight()
                        .or_else(|| w.in_proj.graph_weight_descriptor())?;
                    model = Some(m.clone());
                    crate::gpu::GraphAttn::ShortConv {
                        inp: gw(&w.in_proj)?,
                        out: gw(&w.out_proj)?,
                        taps: &w.conv,
                        kernel: cfg.kernel,
                        cpu_state: &self.kv_cache.layers[self.phys_layer(li)].linear_state,
                    }
                }
                _ => return None,
            };
            layers.push(crate::gpu::GraphLayer {
                input_norm: &lw.input_norm,
                attn,
                post_norm: &lw.post_norm,
                ffn: gffn,
            });
        }
        let model = model?;
        // Fold final-norm + lm_head into the graph when this call wants logits
        // and the lm_head is a graphable (quantized) weight — the graph then
        // reads back logits (into logits_out) instead of the hidden, dropping
        // the separate CPU/GPU lm_head op + its sync. Never the f32 fallback:
        // an unquantized lm_head is vocab·hidden and must not be uploaded.
        let lm_gw = if upto_excl == self.num_layers
            && self.graph_want_logits
            && std::env::var("CMF_GPU_LMHEAD")
                .map(|v| v != "0")
                .unwrap_or(true)
        {
            self.weights
                .lm_head
                .graph_weight()
                .or_else(|| self.weights.lm_head.graph_weight_descriptor())
                .map(|(m, i, kind, rs)| {
                let name = &m.tensors[i].name;
                let prism = if crate::prism::is_inverse_embedding(m, name) {
                    crate::gpu::GraphPrismOp::InverseEmbedding
                } else if crate::prism::is_forward_weight(m, name) {
                    crate::gpu::GraphPrismOp::Forward
                } else {
                    crate::gpu::GraphPrismOp::None
                };
                (
                    crate::gpu::GraphW {
                        idx: i,
                        kind,
                        row_scale: rs,
                        data: &[],
                        prism,
                        affine: crate::prism::is_affine_target(m, name),
                    },
                    self.weights.lm_head.rows(),
                )
            })
        } else {
            None
        };
        let lm = lm_gw.as_ref().map(|(gw, rows)| (gw, *rows));
        // Multi-step re-embeds the winner on the device.
        let emb_gw = if steps > 1 {
            self.weights
                .embed_tokens
                .graph_weight()
                .or_else(|| self.weights.embed_tokens.graph_weight_descriptor())
                .map(|(m, i, kind, rs)| {
                    let name = &m.tensors[i].name;
                    let prism = if crate::prism::is_inverse_embedding(m, name) {
                        crate::gpu::GraphPrismOp::InverseEmbedding
                    } else if crate::prism::is_forward_weight(m, name) {
                        crate::gpu::GraphPrismOp::Forward
                    } else {
                        crate::gpu::GraphPrismOp::None
                    };
                    (
                        crate::gpu::GraphW {
                            idx: i,
                            kind,
                            row_scale: rs,
                            data: &[],
                            prism,
                            affine: crate::prism::is_affine_target(m, name),
                        },
                        self.weights.embed_tokens.rows(),
                        self.embed_multiplier,
                    )
                })
        } else {
            None
        };

        // Loop boundaries: virtual layer indices after which final_norm is
        // applied (mid-stack only; the GLOBAL last layer's norm folds into
        // lm_head). Span-relative — the executor compares its enumerate
        // index. A span ending mid-stack keeps its boundary norm even when
        // it is the span's own last layer.
        let loop_norm_at: Vec<usize> = if self.loop_final_norm {
            (from..upto_excl.min(self.num_layers - 1))
                .filter(|&li| (li + 1) % self.physical_layers == 0)
                .map(|li| li - from)
                .collect()
        } else {
            Vec::new()
        };
        let mut h = hidden.to_vec();
        // The normal decode path only needs the fused lm-head logits.  A
        // CMF_LOGIT_DUMP diagnostic, however, promises a prompt-boundary
        // post-stack hidden alongside those logits; request the existing
        // second readback only for that explicit probe instead of dumping
        // the input copy left in `h` by a folded-head graph.
        let dump_hidden = std::env::var_os("CMF_LOGIT_DUMP").is_some();
        let outcome = crate::gpu::forward_token_graph(
            &model,
            self.graph_kv_id,
            &layers,
            &o1_views,
            self.o1_epoch,
            &self.inv_freq,
            &mut h,
            nh,
            nkv,
            hd,
            self.attn_scale,
            rd,
            self.hidden_size,
            self.intermediate_size,
            position,
            self.kv_cache.max_seq_len,
            gemma,
            self.rms_eps as f32,
            lm,
            &self.weights.final_norm,
            logits_out,
            &loop_norm_at,
            steps,
            emb_gw.as_ref().map(|(gw, rows, m)| (gw, *rows, *m)),
            ids_out,
            layers_run,
            from,
            dump_hidden,
        );
        match outcome {
            crate::gpu::TokenGraphOutcome::Completed => Some(Ok(h)),
            crate::gpu::TokenGraphOutcome::Failed => Some(Err(())),
            crate::gpu::TokenGraphOutcome::Declined => None,
        }
    }

    /// Batched prefill: k contiguous prompt positions through the whole wgpu
    /// graph in ONE submit (projections/FFN as GEMMs). `hiddens` is [k·hidden]
    /// in/out (embeddings in, layer output out); KV mirror / GDN state advance.
    /// false ⇒ unsupported → caller keeps the per-position graph.
    /// The b-row Metal graph plan for the whole model: every layer as a
    /// GDN run or a full-attention item, all-or-nothing (a layer outside the
    /// graph's contract → None, the caller runs plain). Shared by the
    /// speculative verify and the batched prefill.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn metal_rows_plan(
        &self,
    ) -> Option<(
        Vec<MetalRowsItem<'_>>,
        std::sync::Arc<cortiq_core::CmfModel>,
        Option<crate::gpu_metal::GdnGpuCfg>,
    )> {
        use crate::gpu_metal::{AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, MetalFfn};
        let graph_force = crate::gpu::q1_force() || crate::gpu::q2tp_gpu_opt_in();
        if !graph_force
            || !crate::gpu::enabled_here()
            || std::env::var("CMF_GPU_BLOCK")
                .map(|v| v == "0")
                .unwrap_or(false)
            || self.attn_softcap > 0.0
            || self.o1_active()
            || self.swa.is_some()
            || self.global_attn.is_some()
            || self.attention_heads_per_layer.is_some()
            || self.attn_v_norm
            || self.loop_final_norm
        {
            return None;
        }
        let attend_contract = self.head_dim % 4 == 0
            && self.head_dim <= 256
            && self.rotary_dim >= 2
            && self.rotary_dim <= self.head_dim
            && (self.rotary_dim / 2) % 32 == 0
            && self.num_kv_heads > 0
            && self.num_heads % self.num_kv_heads == 0;
        if !attend_contract {
            return None;
        }
        let mut plan: Vec<MetalRowsItem> = Vec::new();
        let mut model_ref: Option<std::sync::Arc<cortiq_core::CmfModel>> = None;
        for li in 0..self.num_layers {
            let lw = &self.weights.layers[self.phys_layer(li)];
            if lw.attn_out_norm.is_some() || lw.ffn_out_norm.is_some() || lw.layer_scale.is_some() {
                return None;
            }
            let ffn = match &lw.ffn {
                FfnKind::Dense(d) if d.act == Act::Silu && d.segs.is_empty() => {
                    let (Some(g), Some(u), Some(dn)) = (
                        d.gate_proj.metal_graph_parts(),
                        d.up_proj.metal_graph_parts(),
                        d.down_proj.metal_graph_parts(),
                    ) else {
                        return None;
                    };
                    MetalFfn::Dense {
                        gate: g,
                        up: u,
                        down: dn,
                    }
                }
                _ => return None,
            };
            match &lw.attn {
                AttnKind::LinearGdn(w) if self.gdn_cfg.is_some() => {
                    let (Some(qkv), Some(z), Some(a), Some(bb), Some(out)) = (
                        w.in_proj_qkv.metal_graph_parts(),
                        w.in_proj_z.metal_graph_parts(),
                        w.in_proj_a.f32_parts(),
                        w.in_proj_b.f32_parts(),
                        w.out_proj.metal_graph_parts(),
                    ) else {
                        return None;
                    };
                    if let QTensor::Mapped { model, .. } = &w.in_proj_qkv {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let gl = GdnGpuLayer {
                        attn_norm: &lw.input_norm,
                        post_norm: &lw.post_norm,
                        qkv,
                        z,
                        a,
                        b: bb,
                        out,
                        ffn,
                        conv1d: &w.conv1d,
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                        gnorm: &w.norm,
                    };
                    match plan.last_mut() {
                        Some(MetalRowsItem::Gdn { run, .. }) => run.push(gl),
                        _ => plan.push(MetalRowsItem::Gdn {
                            run: vec![gl],
                            first: li,
                        }),
                    }
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate: None,
                    bias: None,
                } => {
                    let (Some(pq), Some(pk), Some(pv), Some(po)) =
                        (
                            wq.metal_graph_parts(),
                            wk.metal_graph_parts(),
                            wv.metal_graph_parts(),
                            wo.metal_graph_parts(),
                        )
                    else {
                        return None;
                    };
                    if let QTensor::Mapped { model, .. } = wq {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let cache = &self.kv_cache.layers[li];
                    if cache.mode != crate::kv_cache::KvMode::F32 || cache.o1.is_some() {
                        return None;
                    }
                    plan.push(MetalRowsItem::Attn {
                        l: AttnGpuLayer {
                            attn_norm: &lw.input_norm,
                            post_norm: &lw.post_norm,
                            wq: pq,
                            wk: pk,
                            wv: pv,
                            wo: po,
                            ffn,
                        },
                        li,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                    });
                }
                _ => return None,
            }
        }
        let model = model_ref?;
        let gcfg = self.gdn_cfg.map(|cfg| GdnGpuCfg {
            nv: cfg.num_v_heads,
            nk: cfg.num_k_heads,
            dk: cfg.key_head_dim,
            dv: cfg.value_head_dim,
            kk: cfg.conv_kernel,
            hidden: self.hidden_size,
            inter: self.intermediate_size,
            c_dim: cfg.conv_dim(),
            eps: cfg.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        });
        Some((plan, model, gcfg))
    }

    /// `AttnDeviceParams` for a plan item over the CPU cache as it stands.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn metal_attn_params<'a>(
        li: usize,
        cache: &'a crate::kv_cache::LayerKvCache,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        output_gate: bool,
        inv_freq: &'a [f32],
        geom: (usize, usize, usize, usize),
        pos0: usize,
        kv_id: u64,
        scale: f32,
        eps: f32,
        gemma: bool,
        late_qk_norm: bool,
    ) -> (crate::gpu_metal::AttnDeviceParams<'a>, usize) {
        let (nh, nkv, hd, rd) = geom;
        let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
        let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
        let cpu_stored = cpu_k[0].len() / hd;
        (
            crate::gpu_metal::AttnDeviceParams {
                kv_id,
                layer: li,
                nh,
                nkv,
                hd,
                rd,
                position: pos0,
                scale,
                eps,
                gemma,
                late_qk_norm,
                output_gate,
                q_norm,
                k_norm,
                inv_freq,
                cpu_k,
                cpu_v,
                cpu_stored,
                o1: None,
            },
            cpu_stored,
        )
    }

    /// Run the rows plan over `hiddens` (b rows at `pos0..`): validate,
    /// encode every item, optionally the head, sync. Returns the graph
    /// (for the commit / state finish) plus the GDN layer indices and the
    /// attention layers with the row count they were encoded against.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn metal_rows_run(
        &mut self,
        hiddens: &mut [f32],
        pos0: usize,
        b: usize,
        prefill: bool,
        spec: Option<((usize, usize, usize), &[f32], &mut Vec<f32>)>,
        // Greedy verify: (row length scored, the b argmax ids out) — the
        // head's argmax runs on the device and the logits plane is NOT
        // read back (`spec.2` stays empty).
        mut argmax_out: Option<(usize, &mut Vec<u32>)>,
    ) -> MetalRowsRun {
        use crate::gpu_metal::{GraphDims, VerifyGraph};
        // The previous round's commit may still be replaying into the
        // trunk GDN owners on the second queue: this graph reads them
        // (zero-copy wraps) and may reallocate them below — collect the
        // replay first. Normally already complete (the draft chain ran
        // in between); a failed replay is terminal like a failed commit.
        if !crate::gpu_metal::wait_replay() {
            tracing::error!("Metal rows graph: the pending async replay failed");
            return MetalRowsRun::Failed;
        }
        spec_stamp("v.wait");
        let want = self.gdn_cfg.map(|c| c.state_len()).unwrap_or(0);
        for l in &mut self.kv_cache.layers {
            if l.linear_state.len() != want && want > 0 {
                l.linear_state = vec![0f32; want];
            }
        }
        let Some((plan, model, gcfg)) = self.metal_rows_plan() else {
            return MetalRowsRun::Declined;
        };
        spec_stamp("v.plan");
        let dims = GraphDims {
            hidden: self.hidden_size,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        let Some(mut graph) = (if prefill {
            VerifyGraph::new_prefill(&model, dims, hiddens, b)
        } else {
            VerifyGraph::new(&model, dims, hiddens, b)
        }) else {
            return MetalRowsRun::Declined;
        };
        let geom = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
        );
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        let eps = self.rms_eps as f32;
        let kv_id = self.graph_kv_id;
        let inv_freq = self.inv_freq.clone();
        for item in &plan {
            let ok = match item {
                MetalRowsItem::Gdn { run, .. } => gcfg
                    .as_ref()
                    .map(|gc| run.iter().all(|l| graph.gdn_ok(l, gc)))
                    .unwrap_or(false),
                MetalRowsItem::Attn {
                    l,
                    li,
                    q_norm,
                    k_norm,
                    output_gate,
                } => {
                    let (p, _) = Self::metal_attn_params(
                        *li,
                        &self.kv_cache.layers[*li],
                        *q_norm,
                        *k_norm,
                        *output_gate,
                        &inv_freq,
                        geom,
                        pos0,
                        kv_id,
                        self.attn_scale,
                        eps,
                        gemma,
                        self.qk_norm_after_rope,
                    );
                    graph.attn_ok(l, &p)
                }
            };
            if !ok {
                use std::sync::atomic::{AtomicBool, Ordering};
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!("metal rows graph: a layer failed preflight — declining");
                }
                return MetalRowsRun::Declined;
            }
        }
        let lm = match &spec {
            Some((lm, _, _)) => {
                if !graph.lm_head_ok(*lm) {
                    return MetalRowsRun::Declined;
                }
                Some(*lm)
            }
            None => None,
        };
        let mut gdn_layers = Vec::new();
        let mut attn_layers = Vec::new();
        for item in &plan {
            match item {
                MetalRowsItem::Gdn { run, first } => {
                    let ro: Vec<&[f32]> = self.kv_cache.layers[*first..*first + run.len()]
                        .iter()
                        .map(|l| l.linear_state.as_slice())
                        .collect();
                    if !graph.encode_gdn_run_b(run, &ro, gcfg.as_ref().unwrap()) {
                        return MetalRowsRun::Declined;
                    }
                    gdn_layers.extend(*first..*first + run.len());
                }
                MetalRowsItem::Attn {
                    l,
                    li,
                    q_norm,
                    k_norm,
                    output_gate,
                } => {
                    let (p, cpu_stored) = Self::metal_attn_params(
                        *li,
                        &self.kv_cache.layers[*li],
                        *q_norm,
                        *k_norm,
                        *output_gate,
                        &inv_freq,
                        geom,
                        pos0,
                        kv_id,
                        self.attn_scale,
                        eps,
                        gemma,
                        self.qk_norm_after_rope,
                    );
                    if !graph.encode_attn_b(l, &p) {
                        return MetalRowsRun::Declined;
                    }
                    attn_layers.push((*li, cpu_stored));
                }
            }
        }
        if let (Some(lm), Some((_, final_norm, _))) = (lm, spec.as_ref()) {
            if !graph.encode_lm_head_b(final_norm, lm) {
                return MetalRowsRun::Declined;
            }
            // The device argmax is an OPTIMISATION, never a reason to
            // decline the round: if it will not encode, drop it and read
            // the logits plane back the old way (the head is encoded
            // either way, so the rows are there).
            if let Some((n, _)) = argmax_out.as_ref() {
                if !graph.encode_argmax_b(*n) {
                    argmax_out = None;
                }
            }
        }
        spec_stamp("v.enc");
        if !graph.sync() {
            return MetalRowsRun::Failed;
        }
        spec_stamp("v.gpu");
        match (spec, argmax_out) {
            (Some(_), Some((_, ids))) => {
                ids.resize(b, 0);
                if !graph.read_argmax(ids) {
                    return MetalRowsRun::Failed;
                }
                spec_stamp("v.am");
            }
            (Some((lm, _, logits)), None) => {
                logits.resize(b * lm.1, 0.0);
                if !graph.read_logits(logits) {
                    return MetalRowsRun::Failed;
                }
                spec_stamp("v.lg");
            }
            (None, _) => {}
        }
        if !graph.read_hidden(hiddens) {
            return MetalRowsRun::Failed;
        }
        spec_stamp("v.hid");
        MetalRowsRun::Completed(MetalVerifyPending {
            graph,
            gdn_layers,
            attn_layers,
        })
    }

    /// Native-Metal twin of `try_batch_graph_wgpu`: the b rows through the
    /// whole model on the `VerifyGraph` (one submit), the head folded in
    /// when `spec` asks; `hiddens` come back as the last layer's output
    /// rows, `spec.2` as `[b][lm_rows]` logits. The graph is parked in
    /// `metal_verify` for `metal_verify_commit`.
    #[cfg(target_os = "macos")]
    fn try_batch_graph_metal(
        &mut self,
        hiddens: &mut [f32],
        positions: &[usize],
        b: usize,
        spec: Option<((usize, usize, usize), &[f32], &mut Vec<f32>)>,
        argmax_out: Option<(usize, &mut Vec<u32>)>,
    ) -> crate::gpu::BatchGraphOutcome {
        let _t0 = std::time::Instant::now();
        if positions.len() != b
            || positions.windows(2).any(|w| w[1] != w[0] + 1)
            || hiddens.len() != b * self.hidden_size
        {
            return crate::gpu::BatchGraphOutcome::Declined;
        }
        let pending = match self.metal_rows_run(hiddens, positions[0], b, false, spec, argmax_out) {
            MetalRowsRun::Declined => return crate::gpu::BatchGraphOutcome::Declined,
            MetalRowsRun::Failed => return crate::gpu::BatchGraphOutcome::Failed,
            MetalRowsRun::Completed(pending) => pending,
        };
        if std::env::var("CMF_GRAPH_SPEC_TIME").is_ok() {
            eprintln!(
                "metal-verify: {:.1} ms | b={b}",
                _t0.elapsed().as_secs_f64() * 1e3
            );
        }
        self.metal_verify = Some(pending);
        crate::gpu::BatchGraphOutcome::Completed
    }

    /// Batched prefill on the Metal rows graph: `ids` (≤ 512) at
    /// `start_pos..`, states written in place, K/V rows appended to the
    /// CPU caches; optional final norm/head logits are returned in `spec`.
    /// Declined means no command buffer was admitted; Failed is terminal.
    #[cfg(target_os = "macos")]
    fn prefill_rows_metal(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        spec: Option<((usize, usize, usize), &[f32], &mut Vec<f32>)>,
    ) -> MetalPrefillOutcome {
        let b = ids.len();
        if b == 0 || b > 512 {
            return MetalPrefillOutcome::Declined;
        }
        METAL_PREFILL_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let with_head = spec.is_some();
        let hs = self.hidden_size;
        let mut hiddens = vec![0f32; b * hs];
        for (j, &id) in ids.iter().enumerate() {
            let e = self.embed_single(id);
            hiddens[j * hs..(j + 1) * hs].copy_from_slice(&e);
        }
        let mut pending = match self.metal_rows_run(&mut hiddens, start_pos, b, true, spec, None) {
            MetalRowsRun::Declined => return MetalPrefillOutcome::Declined,
            MetalRowsRun::Failed => {
                METAL_PREFILL_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return MetalPrefillOutcome::Failed;
            }
            MetalRowsRun::Completed(pending) => pending,
        };
        // states are final: copy them to the owners
        let idxs = pending.gdn_layers.clone();
        let mut outs: Vec<&mut [f32]> = self
            .kv_cache
            .layers
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| idxs.binary_search(i).is_ok())
            .map(|(_, l)| l.linear_state.as_mut_slice())
            .collect();
        if !pending.graph.finish_states(&mut outs) {
            METAL_PREFILL_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return MetalPrefillOutcome::Failed;
        }
        let (nkv, hd) = (self.num_kv_heads, self.head_dim);
        // Read every layer before mutating any CPU cache.  A missing mirror
        // row is a terminal graph failure, not a reason to append a partial
        // prefix and replay the remainder serially.
        let mut rows = Vec::with_capacity(pending.attn_layers.len());
        for (li, cpu_stored) in &pending.attn_layers {
            let mut kbuf = vec![0f32; b * nkv * hd];
            let mut vbuf = vec![0f32; b * nkv * hd];
            if !crate::gpu_metal::kv_mirror_read_rows(
                self.graph_kv_id,
                *li,
                nkv,
                hd,
                *cpu_stored,
                b,
                &mut kbuf,
                &mut vbuf,
            ) {
                METAL_PREFILL_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return MetalPrefillOutcome::Failed;
            }
            rows.push((*li, *cpu_stored, kbuf, vbuf));
        }
        for (li, cpu_stored, kbuf, vbuf) in rows {
            let cache = &mut self.kv_cache.layers[li];
            for r in 0..b {
                cache.append(
                    &kbuf[r * nkv * hd..(r + 1) * nkv * hd],
                    &vbuf[r * nkv * hd..(r + 1) * nkv * hd],
                    &[],
                );
            }
            crate::gpu_metal::kv_mirror_set_stored(self.graph_kv_id, li, cpu_stored + b);
        }
        METAL_PREFILL_ROWS.fetch_add(b as u64, std::sync::atomic::Ordering::Relaxed);
        if with_head {
            METAL_PREFILL_HEAD_ROWS.fetch_add(b as u64, std::sync::atomic::Ordering::Relaxed);
        }
        MetalPrefillOutcome::Completed(hiddens)
    }

    #[cfg(target_os = "macos")]
    fn prefill_batch_metal(&mut self, ids: &[u32], start_pos: usize) -> MetalPrefillOutcome {
        self.prefill_rows_metal(ids, start_pos, None)
    }

    /// Exact teacher-forced NLL through the ordinary Metal rows graph.  This
    /// is intentionally separate from the serial TokenGraph scorer: every
    /// chunk owns a real b-row graph/head completion and the recurrent/KV
    /// handoff is committed before the next chunk begins.
    #[cfg(target_os = "macos")]
    fn nll_batch_metal(&mut self, ids: &[u32], start: usize) -> MetalBatchNllOutcome {
        if ids.len() < 2 || self.o1_active() || self.head_clusters.is_some() {
            return MetalBatchNllOutcome::Declined;
        }
        let Some(lm) = self.weights.lm_head.metal_graph_parts() else {
            return MetalBatchNllOutcome::Declined;
        };
        let chunk = std::env::var("CMF_METAL_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| (1..=512).contains(&v))
            .unwrap_or(32);
        let final_norm = self.weights.final_norm.clone();
        let mut nll = 0.0f64;
        let mut count = 0usize;
        let mut pos = 0usize;
        let mut completed = 0usize;
        while pos < ids.len() {
            let end = (pos + chunk).min(ids.len());
            let mut logits = Vec::new();
            let outcome = self.prefill_rows_metal(
                &ids[pos..end],
                pos,
                Some((lm, &final_norm, &mut logits)),
            );
            match outcome {
                MetalPrefillOutcome::Declined => {
                    return if completed == 0 {
                        MetalBatchNllOutcome::Declined
                    } else {
                        MetalBatchNllOutcome::Failed(format!(
                            "ordinary Metal NLL batch declined after {completed} chunks"
                        ))
                    };
                }
                MetalPrefillOutcome::Failed => {
                    return MetalBatchNllOutcome::Failed(
                        "ordinary Metal NLL batch failed after admission".to_string(),
                    );
                }
                MetalPrefillOutcome::Completed(_) => {}
            }
            completed += 1;
            let vocab = self.vocab_size.min(lm.1);
            if logits.len() != (end - pos) * lm.1 || vocab == 0 {
                return MetalBatchNllOutcome::Failed(
                    "ordinary Metal NLL head returned an invalid shape".to_string(),
                );
            }
            for row in 0..(end - pos) {
                let absolute = pos + row;
                if absolute < start || absolute + 1 >= ids.len() {
                    continue;
                }
                let lg = &mut logits[row * lm.1..row * lm.1 + vocab];
                if let Some(mu) = self.logit_multiplier {
                    for v in lg.iter_mut() {
                        *v *= mu;
                    }
                }
                if let Some(c) = self.final_softcap {
                    for v in lg.iter_mut() {
                        *v = c * (*v / c).tanh();
                    }
                }
                let target = ids[absolute + 1] as usize;
                if target >= vocab {
                    return MetalBatchNllOutcome::Failed(format!(
                        "target token {target} exceeds Metal head rows {vocab}"
                    ));
                }
                let max = lg.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let lse: f64 = lg
                    .iter()
                    .map(|&v| ((v - max) as f64).exp())
                    .sum::<f64>()
                    .ln()
                    + max as f64;
                nll += lse - lg[target] as f64;
                count += 1;
            }
            pos = end;
        }
        MetalBatchNllOutcome::Completed(nll, count)
    }

    /// Commit a Metal verify round: replay the GDN recurrences over the
    /// `a + 1` accepted positions into the CPU states, append the accepted
    /// K/V rows from the mirrors to the CPU caches, re-point the mirrors.
    #[cfg(target_os = "macos")]
    fn metal_verify_commit(&mut self, a: usize) -> bool {
        let Some(mut pending) = self.metal_verify.take() else {
            return false;
        };
        let n = a + 1;
        // encode order == ascending layer order (the plan walks 0..layers)
        let idxs = pending.gdn_layers.clone();
        let mut outs: Vec<&mut [f32]> = self
            .kv_cache
            .layers
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| idxs.binary_search(i).is_ok())
            .map(|(_, l)| l.linear_state.as_mut_slice())
            .collect();
        if !pending.graph.commit(n, &mut outs) {
            return false;
        }
        spec_stamp("c.replay");
        let (nkv, hd) = (self.num_kv_heads, self.head_dim);
        // Read every layer before mutating any CPU cache.  Missing rows are
        // terminal after the replay has executed; never append a partial KV
        // prefix and continue on a serial path.
        let mut rows = Vec::with_capacity(pending.attn_layers.len());
        for (li, cpu_stored) in &pending.attn_layers {
            let mut kbuf = vec![0f32; n * nkv * hd];
            let mut vbuf = vec![0f32; n * nkv * hd];
            if !crate::gpu_metal::kv_mirror_read_rows(
                self.graph_kv_id,
                *li,
                nkv,
                hd,
                *cpu_stored,
                n,
                &mut kbuf,
                &mut vbuf,
            ) {
                return false;
            }
            rows.push((*li, *cpu_stored, kbuf, vbuf));
        }
        for (li, cpu_stored, kbuf, vbuf) in rows {
            let cache = &mut self.kv_cache.layers[li];
            for r in 0..n {
                cache.append(
                    &kbuf[r * nkv * hd..(r + 1) * nkv * hd],
                    &vbuf[r * nkv * hd..(r + 1) * nkv * hd],
                    &[],
                );
            }
            crate::gpu_metal::kv_mirror_set_stored(self.graph_kv_id, li, cpu_stored + n);
        }
        spec_stamp("c.kv");
        true
    }

    /// The round's warm-ups as ONE b-row graph run over the MTP block on
    /// Metal: `pairs` = (trunk hidden, next token) at consecutive positions
    /// from `first_pos`; the block's input projection is folded in. This
    /// half encodes and SUBMITS (no wait); `mtp_warm_batch_finish` waits
    /// and pulls the appended K/V rows into the CPU MTP cache. None = the
    /// graph declined (nothing submitted, nothing appended).
    #[cfg(target_os = "macos")]
    fn mtp_warm_batch_submit(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> Option<MetalWarmPending> {
        use crate::gpu_metal::{AttnDeviceParams, AttnGpuLayer, GraphDims, MetalFfn, VerifyGraph};
        let b = pairs.len();
        if b == 0 || b > 512 || m.kv.mode != crate::kv_cache::KvMode::F32 || m.kv.o1.is_some() {
            return None;
        }
        let AttnKind::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            output_gate,
            softplus_gate: None,
            bias: None,
        } = &m.layer.attn
        else {
            return None;
        };
        let FfnKind::Dense(d) = &m.layer.ffn else {
            return None;
        };
        if !d.segs.is_empty() {
            return None;
        }
        let (Some(pq), Some(pk), Some(pv), Some(po)) =
            (wq.q1_parts(), wk.q1_parts(), wv.q1_parts(), wo.q1_parts())
        else {
            return None;
        };
        let (Some(g), Some(u), Some(dn)) = (
            d.gate_proj.q1_parts(),
            d.up_proj.q1_parts(),
            d.down_proj.q1_parts(),
        ) else {
            return None;
        };
        let Some(eh) = m.eh_proj.q1_parts() else {
            return None;
        };
        let QTensor::Mapped { model, .. } = wq else {
            return None;
        };
        let model = model.clone();
        let hs = self.hidden_size;
        // [enorm(embed(tok)); hnorm(hidden)] rows
        let mut cat = vec![0f32; b * 2 * hs];
        for (j, (h, tok)) in pairs.iter().enumerate() {
            let e = self.embed_single(*tok);
            let (ce, ch) = cat[j * 2 * hs..(j + 1) * 2 * hs].split_at_mut(hs);
            inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, ce);
            inference::rms_norm_into(h, &m.hnorm, self.rms_eps, self.norm_style, ch);
        }
        let dims = GraphDims {
            hidden: hs,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        spec_stamp("w.cat");
        let Some(mut graph) = VerifyGraph::new_via_proj(&model, dims, eh, &cat, b) else {
            return None;
        };
        spec_stamp("w.new");
        let l = AttnGpuLayer {
            attn_norm: &m.layer.input_norm,
            post_norm: &m.layer.post_norm,
            wq: pq,
            wk: pk,
            wv: pv,
            wo: po,
            ffn: MetalFfn::Dense {
                gate: g,
                up: u,
                down: dn,
            },
        };
        let (nh, nkv, hd, rd) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
        );
        let inv_freq = self.inv_freq.clone();
        let cpu_stored;
        {
            let cache = &m.kv;
            let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
            let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
            cpu_stored = cpu_k[0].len() / hd;
            // The cache may LAG the position (rows nobody warmed): the
            // pairs land at cpu_stored.. with their true RoPE positions
            // first_pos.., exactly what the one-by-one warm does. A cache
            // AHEAD of the position is a real inconsistency.
            if cpu_stored > first_pos {
                spec_stamp("w.decl");
                return None;
            }
            let p = AttnDeviceParams {
                kv_id: self.mtp_kv_id(),
                layer: Self::MTP_LAYER_BASE,
                nh,
                nkv,
                hd,
                rd,
                position: first_pos,
                scale: self.attn_scale,
                eps: self.rms_eps as f32,
                gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
                late_qk_norm: self.qk_norm_after_rope,
                output_gate: *output_gate,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                inv_freq: &inv_freq,
                cpu_k,
                cpu_v,
                cpu_stored,
                o1: None,
            };
            if !graph.attn_ok(&l, &p) || !graph.encode_attn_b(&l, &p) {
                return None;
            }
        }
        spec_stamp("w.enc");
        if !graph.submit() {
            return None;
        }
        spec_stamp("w.sub");
        Some(MetalWarmPending {
            graph,
            cpu_stored,
            b,
        })
    }

    /// Submit and finish in one call (the prefill's MTP warm-up, where
    /// nothing runs in between).
    #[cfg(target_os = "macos")]
    fn mtp_warm_batch_metal(
        &mut self,
        m: &mut MtpModule,
        pairs: &[(&[f32], u32)],
        first_pos: usize,
    ) -> bool {
        match self.mtp_warm_batch_submit(m, pairs, first_pos) {
            Some(p) => self.mtp_warm_batch_finish(m, p),
            None => false,
        }
    }

    /// Second half of the batched warm-up: wait for the submitted graph,
    /// pull its b appended K/V rows into the CPU MTP cache, re-point the
    /// mirror. False = the command buffer failed or the rows are missing
    /// (nothing appended; the caller falls back to the one-by-one warm).
    #[cfg(target_os = "macos")]
    fn mtp_warm_batch_finish(&mut self, m: &mut MtpModule, pending: MetalWarmPending) -> bool {
        let MetalWarmPending {
            mut graph,
            cpu_stored,
            b,
        } = pending;
        let (nkv, hd) = (self.num_kv_heads, self.head_dim);
        if !graph.sync() {
            return false;
        }
        spec_stamp("w.gpu");
        let mut kbuf = vec![0f32; b * nkv * hd];
        let mut vbuf = vec![0f32; b * nkv * hd];
        if !crate::gpu_metal::kv_mirror_read_rows(
            self.mtp_kv_id(),
            Self::MTP_LAYER_BASE,
            nkv,
            hd,
            cpu_stored,
            b,
            &mut kbuf,
            &mut vbuf,
        ) {
            return false;
        }
        for r in 0..b {
            m.kv.append(
                &kbuf[r * nkv * hd..(r + 1) * nkv * hd],
                &vbuf[r * nkv * hd..(r + 1) * nkv * hd],
                &[],
            );
        }
        crate::gpu_metal::kv_mirror_set_stored(
            self.mtp_kv_id(),
            Self::MTP_LAYER_BASE,
            cpu_stored + b,
        );
        spec_stamp("w.kv");
        true
    }

    /// A committed token id from the high table (Cyrillic, CJK and the
    /// like sit above 131072 in Qwen's vocabulary; Latin subwords past
    /// the 65536 cut are rare enough to lose as rejected drafts) switches
    /// the draft to the full head for the next 16 tokens; other ids count
    /// down. On an M4 the full 660 MB head costs 5.5 ms a draft step
    /// against 1.4 for the shortlist, so the streak is kept short.
    pub(crate) fn note_draft_id(&mut self, id: u32) {
        let cut = Self::draft_vocab_rows(usize::MAX).max(131_072);
        if (id as usize) >= cut {
            self.draft_full_streak = 16;
        } else {
            self.draft_full_streak = self.draft_full_streak.saturating_sub(1);
        }
    }

    /// The draft head's rows for the next step: the shortlist, or the full
    /// head while `draft_full_streak` runs.
    fn draft_head_rows(&self, head_rows: usize) -> usize {
        if self.draft_full_streak > 0 {
            head_rows
        } else {
            Self::draft_vocab_rows(head_rows)
        }
    }

    /// Draft-head shortlist size: `CMF_DRAFT_VOCAB` rows (default 65536,
    /// capped at the head; 0 = full head).
    fn draft_vocab_rows(head_rows: usize) -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let n = *N.get_or_init(|| {
            std::env::var("CMF_DRAFT_VOCAB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(65536)
        });
        if n == 0 { head_rows } else { n.min(head_rows) }
    }

    /// One MTP block step on the native Metal token graph: block input on
    /// the host, the attention layer + FFN device-resident over the MTP
    /// mirror, the head folded in when `want_logits`. The appended K/V row
    /// is pulled into the CPU MTP cache (owner of record) after the sync.
    #[cfg(target_os = "macos")]
    fn mtp_step_metal(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
        want_logits: bool,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        use crate::gpu_metal::{AttnDeviceParams, AttnGpuLayer, GraphDims, MetalFfn, TokenGraph};
        if std::env::var("CMF_MTP_GRAPH").as_deref() == Ok("0")
            || !crate::gpu::q1_force()
            || !crate::gpu::enabled_here()
            || self.attn_softcap > 0.0
            || self.attention_heads_per_layer.is_some()
            || m.kv.mode != crate::kv_cache::KvMode::F32
            || m.kv.o1.is_some()
        {
            return None;
        }
        let AttnKind::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            output_gate,
            softplus_gate: None,
            bias: None,
        } = &m.layer.attn
        else {
            return None;
        };
        let FfnKind::Dense(d) = &m.layer.ffn else {
            return None;
        };
        if d.act != Act::Silu || !d.segs.is_empty() {
            return None;
        }
        let (pq, pk, pv, po) = (
            wq.q1_parts()?,
            wk.q1_parts()?,
            wv.q1_parts()?,
            wo.q1_parts()?,
        );
        let (g, u, dn) = (
            d.gate_proj.q1_parts()?,
            d.up_proj.q1_parts()?,
            d.down_proj.q1_parts()?,
        );
        let QTensor::Mapped { model, .. } = wq else {
            return None;
        };
        let model = model.clone();
        let lm = if want_logits {
            Some(self.weights.lm_head.q1_parts()?)
        } else {
            None
        };
        let dims = GraphDims {
            hidden: self.hidden_size,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        // The block input `eh_proj · [enorm(e); hnorm(h)]` rides in the
        // graph (one submit a step); the host per-op matvec if it cannot.
        let hs = self.hidden_size;
        let mut x = vec![0f32; hs];
        let mut graph = TokenGraph::new(&model, dims, &x)?;
        let mut folded = false;
        if let Some(eh) = m.eh_proj.q1_parts() {
            let e = self.embed_single(next_token);
            let mut cat = vec![0.0f32; 2 * hs];
            let (cat_e, cat_h) = cat.split_at_mut(hs);
            inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, cat_e);
            inference::rms_norm_into(hidden, &m.hnorm, self.rms_eps, self.norm_style, cat_h);
            folded = graph.encode_input_proj(eh, &cat);
        }
        if !folded {
            x = self.mtp_block_input(m, hidden, next_token);
            graph = TokenGraph::new(&model, dims, &x)?;
        }
        spec_stamp("d.in");
        let l = AttnGpuLayer {
            attn_norm: &m.layer.input_norm,
            post_norm: &m.layer.post_norm,
            wq: pq,
            wk: pk,
            wv: pv,
            wo: po,
            ffn: MetalFfn::Dense {
                gate: g,
                up: u,
                down: dn,
            },
        };
        let (nh, nkv, hd, rd) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
        );
        let inv_freq = self.inv_freq.clone();
        {
            let cache = &m.kv;
            let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
            let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
            let cpu_stored = cpu_k[0].len() / hd;
            let p = AttnDeviceParams {
                kv_id: self.mtp_kv_id(),
                layer: Self::MTP_LAYER_BASE,
                nh,
                nkv,
                hd,
                rd,
                position,
                scale: self.attn_scale,
                eps: self.rms_eps as f32,
                gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
                late_qk_norm: self.qk_norm_after_rope,
                output_gate: *output_gate,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                inv_freq: &inv_freq,
                cpu_k,
                cpu_v,
                cpu_stored,
                o1: None,
            };
            if !graph.attn_device_ok(&l, &p) || !graph.encode_attn_device(&l, &p) {
                return None;
            }
        }
        // The draft's head over a vocabulary SHORTLIST (the first
        // CMF_DRAFT_VOCAB rows — BPE ids run roughly by merge rank, so the
        // low ids carry the mass): the verify keeps the full head, so a true
        // token past the cut is only a rejected draft, never a wrong token.
        // 662 MB a step on Qwen3.8 becomes 170 MB at 65536.
        let draft_rows = if let Some(lm) = lm {
            self.draft_head_rows(lm.1)
        } else {
            0
        };
        if let Some(lm) = lm {
            if !graph.lm_head_ok(lm) {
                return None;
            }
            if draft_rows < lm.1 {
                if !graph.encode_lm_head_part(&m.final_norm, lm, draft_rows) {
                    return None;
                }
            } else {
                graph.encode_lm_head(&m.final_norm, lm);
            }
        }
        spec_stamp("d.enc");
        if graph.sync_checked().is_err() {
            return None;
        }
        spec_stamp("d.gpu");
        let mut logits = Vec::new();
        if let Some(lm) = lm {
            let n_read = draft_rows.min(lm.1).min(self.vocab_size);
            logits = attention::take_buf(n_read);
            graph.read_logits(&mut logits);
            // ids past the shortlist: never drafted (−∞ in every chain)
            logits.resize(self.vocab_size, f32::NEG_INFINITY);
        }
        graph.finish(&mut x);
        let mut krow = attention::take_buf(nkv * hd);
        let mut vrow = attention::take_buf(nkv * hd);
        if crate::gpu_metal::kv_mirror_read_last(
            self.mtp_kv_id(),
            Self::MTP_LAYER_BASE,
            nkv,
            hd,
            &mut krow,
            &mut vrow,
        ) {
            m.kv.append(&krow, &vrow, &[]);
        }
        attention::recycle_buf(&mut krow);
        attention::recycle_buf(&mut vrow);
        spec_stamp("d.rd");
        Some((logits, x))
    }

    /// `CMF_MTP_CHAIN=0` keeps the per-step draft (one submit and one
    /// host round trip per MTP step); the default drafts the whole chain
    /// in one command buffer when the round is plain greedy.
    ///
    /// Measured on an M4 (24 GB), Qwen3.8-27B q4tp, P3 at 160 tokens,
    /// k=7, six runs per arm alternating inside one lock window — the
    /// round's draft phase (median over the 34 rounds of a run) is
    /// 34.5 ms per round old against 30.1 new, i.e. 4.93 → 4.31 ms per
    /// draft step. That is the whole prize: the 7 submits cost ~0.6 ms
    /// each in host and submit latency and nothing else changes —
    /// acceptance (3.41 of 7) and tokens per round (4.41) are identical,
    /// and the round is 289 → 285 ms, decode 13.8 → 14.0 tok/s.
    fn mtp_chain_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_MTP_CHAIN").as_deref() != Ok("0"))
    }

    /// The round's k greedy drafts as ONE command buffer on Metal: the MTP
    /// block k times back to back, each step's token embedding gathered
    /// on the device from the argmax the step before it wrote, the head
    /// over the round's shortlist (or the full head during a full-head
    /// streak — decided once, before the chain, exactly as the per-step
    /// path decides it per step, since `draft_full_streak` only moves on
    /// a commit). One wait, then the k ids and the k appended K/V rows
    /// come back; the CPU MTP cache ends where k `mtp_step_metal` calls
    /// would have left it. `Err(false)` = declined before anything was
    /// committed (the per-step path takes the round); `Err(true)` = the
    /// command buffer failed after commit.
    #[cfg(target_os = "macos")]
    fn mtp_draft_chain_metal(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        t_next: u32,
        position: usize,
        k: usize,
    ) -> Result<Vec<u32>, bool> {
        use crate::gpu_metal::{AttnDeviceParams, AttnGpuLayer, GraphDims, MetalFfn, TokenGraph};
        if k == 0
            || k > 64
            || !Self::mtp_chain_on()
            || std::env::var("CMF_MTP_GRAPH").as_deref() == Ok("0")
            || !crate::gpu::q1_force()
            || !crate::gpu::enabled_here()
            || self.attn_softcap > 0.0
            || self.attention_heads_per_layer.is_some()
            || m.kv.mode != crate::kv_cache::KvMode::F32
            || m.kv.o1.is_some()
            // the chain gathers embeddings itself: only the plain table
            || self.dsv4.is_some()
            || self.dsv41.is_some()
            || self.qwen4_exp.is_some()
            || self.g3n.is_some()
        {
            return Err(false);
        }
        let AttnKind::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            output_gate,
            softplus_gate: None,
            bias: None,
        } = &m.layer.attn
        else {
            return Err(false);
        };
        let FfnKind::Dense(d) = &m.layer.ffn else {
            return Err(false);
        };
        if d.act != Act::Silu || !d.segs.is_empty() {
            return Err(false);
        }
        let (Some(pq), Some(pk), Some(pv), Some(po)) =
            (wq.q1_parts(), wk.q1_parts(), wv.q1_parts(), wo.q1_parts())
        else {
            return Err(false);
        };
        let (Some(g), Some(u), Some(dn)) = (
            d.gate_proj.q1_parts(),
            d.up_proj.q1_parts(),
            d.down_proj.q1_parts(),
        ) else {
            return Err(false);
        };
        let (Some(eh), Some(lm)) = (m.eh_proj.q1_parts(), self.weights.lm_head.q1_parts()) else {
            return Err(false);
        };
        let QTensor::Mapped { model, .. } = wq else {
            return Err(false);
        };
        let model = model.clone();
        // the embedding table: a q4tp tensor of the SAME blob, no Prism
        // inverse-embedding post-pass
        let QTensor::Mapped {
            model: em,
            idx: eidx,
            dtype: cortiq_core::TensorDtype::Q4TiledP,
            ..
        } = &self.weights.embed_tokens
        else {
            return Err(false);
        };
        if !std::sync::Arc::ptr_eq(em, &model)
            || crate::prism::is_inverse_embedding(&model, &model.tensors[*eidx].name)
        {
            return Err(false);
        }
        let embed = (
            *eidx,
            self.weights.embed_tokens.rows(),
            self.weights.embed_tokens.cols(),
        );
        if embed.2 != self.hidden_size || hidden.len() != self.hidden_size {
            return Err(false);
        }
        let dims = GraphDims {
            hidden: self.hidden_size,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        let Some(mut graph) = TokenGraph::new(&model, dims, hidden) else {
            return Err(false);
        };
        if !graph.chain_embed_ok(embed) || !graph.lm_head_ok(lm) {
            return Err(false);
        }
        let l = AttnGpuLayer {
            attn_norm: &m.layer.input_norm,
            post_norm: &m.layer.post_norm,
            wq: pq,
            wk: pk,
            wv: pv,
            wo: po,
            ffn: MetalFfn::Dense {
                gate: g,
                up: u,
                down: dn,
            },
        };
        let (nh, nkv, hd, rd) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
        );
        let inv_freq = self.inv_freq.clone();
        let draft_rows = self.draft_head_rows(lm.1);
        let n_arg = draft_rows.min(lm.1).min(self.vocab_size);
        if n_arg == 0 {
            return Err(false);
        }
        // `CMF_MTP_CHAIN_SPLIT=1` commits each step as it is encoded, so
        // the GPU starts on step 0 while the host is still encoding step
        // 1 — a probe for whether the host encode is on the critical
        // path. It is not: three runs each, draft 30.0 ms per round split
        // against 30.1 whole, and the whole chain's host encode measures
        // 0.3 ms against a 29.7 ms wait. Kept as a probe, off by default.
        let split = std::env::var("CMF_MTP_CHAIN_SPLIT").as_deref() == Ok("1");
        let t_chain = std::time::Instant::now();
        graph.chain_ids_init(t_next, k);
        let cpu_stored;
        {
            let cache = &m.kv;
            let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
            let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
            cpu_stored = cpu_k[0].len() / hd;
            for j in 0..k {
                if !graph.encode_chain_input(
                    embed,
                    j as u32,
                    &m.enorm,
                    &m.hnorm,
                    self.embed_multiplier,
                    eh,
                ) {
                    return Err(false);
                }
                // step j's mirror row: the mirror is re-pointed at the CPU
                // rows before step 0 and advances by one per step; its
                // resync (never taken past step 0) reads the CPU rows
                let p = AttnDeviceParams {
                    kv_id: self.mtp_kv_id(),
                    layer: Self::MTP_LAYER_BASE,
                    nh,
                    nkv,
                    hd,
                    rd,
                    position: position + j,
                    scale: self.attn_scale,
                    eps: self.rms_eps as f32,
                    gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
                    late_qk_norm: self.qk_norm_after_rope,
                    output_gate: *output_gate,
                    q_norm: q_norm.as_deref(),
                    k_norm: k_norm.as_deref(),
                    inv_freq: &inv_freq,
                    cpu_k: cpu_k.clone(),
                    cpu_v: cpu_v.clone(),
                    cpu_stored: cpu_stored + j,
                    o1: None,
                };
                if !graph.attn_device_ok(&l, &p) || !graph.encode_attn_device(&l, &p) {
                    return Err(false);
                }
                if draft_rows < lm.1 {
                    if !graph.encode_lm_head_part(&m.final_norm, lm, draft_rows) {
                        return Err(false);
                    }
                } else {
                    graph.encode_lm_head(&m.final_norm, lm);
                }
                if !graph.encode_argmax(n_arg, j as u32 + 1) {
                    return Err(false);
                }
                if split {
                    // CMF_MTP_CHAIN_SPLIT=1: commit every step so the GPU
                    // starts on step 0 while the host encodes the rest
                    graph.commit();
                }
            }
        }
        let t_enc = t_chain.elapsed();
        if graph.sync_checked().is_err() {
            return Err(true);
        }
        if std::env::var_os("CMF_GRAPH_SPEC_TIME").is_some() {
            eprintln!(
                "mtp-chain: encode {:.1} ms | wait {:.1} ms (k={k}, head rows {draft_rows}{})",
                t_enc.as_secs_f64() * 1e3,
                (t_chain.elapsed() - t_enc).as_secs_f64() * 1e3,
                if split { ", split" } else { "" }
            );
        }
        let mut ids = vec![0u32; k];
        if !graph.chain_ids_read(&mut ids) {
            return Err(true);
        }
        let mut kbuf = vec![0f32; k * nkv * hd];
        let mut vbuf = vec![0f32; k * nkv * hd];
        if !crate::gpu_metal::kv_mirror_read_rows(
            self.mtp_kv_id(),
            Self::MTP_LAYER_BASE,
            nkv,
            hd,
            cpu_stored,
            k,
            &mut kbuf,
            &mut vbuf,
        ) {
            return Err(true);
        }
        for r in 0..k {
            m.kv.append(
                &kbuf[r * nkv * hd..(r + 1) * nkv * hd],
                &vbuf[r * nkv * hd..(r + 1) * nkv * hd],
                &[],
            );
        }
        Ok(ids)
    }

    fn try_batch_graph_wgpu(
        &self,
        hiddens: &mut [f32],
        positions: &[usize],
        k: usize,
        spec: Option<crate::gpu::SpecTail<'_>>,
    ) -> crate::gpu::BatchGraphOutcome {
        let _tb = std::time::Instant::now();
        let batch_debug = std::env::var_os("CMF_BATCH_DEBUG").is_some();
        if self.attn_softcap > 0.0 {
            return crate::gpu::BatchGraphOutcome::Declined; // capped scores: no graph kernel — CPU path
        }
        let nh = self.num_heads;
        let (nkv, hd, rd) = self.layer_geom(0);
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
            if let Some((m, i, kind, rs)) = t
                .graph_weight()
                .or_else(|| t.graph_weight_descriptor())
            {
                let name = &m.tensors[i].name;
                let prism = if crate::prism::is_inverse_embedding(m, name) {
                    crate::gpu::GraphPrismOp::InverseEmbedding
                } else if crate::prism::is_forward_weight(m, name) {
                    crate::gpu::GraphPrismOp::Forward
                } else {
                    crate::gpu::GraphPrismOp::None
                };
                return Some(crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                    prism,
                    affine: crate::prism::is_affine_target(m, name),
                });
            }
            if std::env::var_os("CMF_BATCH_DEBUG").is_some() {
                eprintln!(
                    "batch graph: tensor has no graph descriptor/f32 fallback rows={} cols={}",
                    t.rows(),
                    t.cols()
                );
            }
            t.as_f32().map(|d| crate::gpu::GraphW {
                idx: 0,
                kind: 4,
                row_scale: &[],
                data: d,
                prism: crate::gpu::GraphPrismOp::None,
                affine: false,
            })
        }
        let built: Option<(
            Vec<crate::gpu::GraphLayer<'_>>,
            std::sync::Arc<cortiq_core::CmfModel>,
        )> = (|| {
            let mut layers = Vec::with_capacity(self.num_layers);
            let mut model = None;
            for li in 0..self.num_layers {
                let lw = &self.weights.layers[self.phys_layer(li)];
                // MoE routes per token, so its experts are encoded token by
                // token inside the batched submit while attention and the
                // projections stay GEMMs. Refusing MoE here is what left
                // prefill running one position at a time: 33 tok/s against
                // 54 on decode, i.e. reading the prompt was slower than
                // writing the answer.
                let gffn = match &lw.ffn {
                    FfnKind::Dense(d) if !d.segs.is_empty() => {
                        if batch_debug {
                            eprintln!("batch graph: dense segmented FFN at layer {li}");
                        }
                        return None;
                    }
                    FfnKind::Dense(d) => crate::gpu::GraphFfn::Dense {
                        gate: gw(&d.gate_proj)?,
                        up: gw(&d.up_proj)?,
                        down: gw(&d.down_proj)?,
                    },
                    FfnKind::Moe(m) => {
                        // Adaptive τ and expert masks stay on the CPU path.
                        // Sigmoid scores, the selection bias, a routed scale
                        // ≠ 1 and an ungated shared expert (hy_v3) ride the
                        // same flags word as the token graph — before, this
                        // refusal sent every Hy-MT2-30B prompt to the chunked
                        // fallback (8 tok/s of ingest against 53 of decode).
                        if m.route_tau.is_some() || m.mask.is_some() {
                            return None;
                        }
                        // The batch MoE kernels need the shared slot (k+1
                        // rows); gated or not is a flag on the select kernel.
                        let (se, sg) = m.shared.as_ref()?;
                        let shared_gated = sg.is_some();
                        let sgate = match sg {
                            Some(sg) => gw(sg)?,
                            // Ungated: the router plane stands in so the
                            // plumbing stays total; the kernel pins weight 1.
                            None => gw(&m.router)?,
                        };
                        let router = gw(&m.router)?;
                        // The batch MoE kernels still consume raw per-token
                        // rows and do not carry the descriptor-aware Prism
                        // transform/affine bit for router or shared-gate
                        // planes.  Refuse rather than route an untransformed
                        // source activation.
                        if router.prism != crate::gpu::GraphPrismOp::None
                            || router.affine
                            || sgate.prism != crate::gpu::GraphPrismOp::None
                            || sgate.affine
                        {
                            return None;
                        }
                        let inter = m.experts.first()?.gate_proj.rows();
                        let mut experts = Vec::with_capacity(m.experts.len() + 1);
                        let mut q4tp: Option<bool> = None;
                        let mut gu_q2: Option<bool> = None;
                        for e in m.experts.iter().chain(std::iter::once(se)) {
                            if !matches!(e.act, Act::Silu)
                                || e.gate_proj.rows() != inter
                                || e.up_proj.rows() != inter
                            {
                                return None;
                            }
                            // Same ladder as the token graph: q4t → q2tp
                            // (mixed profile: 2-bit gate/up over a q4tp
                            // down) → q4tp. Uniform across the layer.
                            let (mm, gi, ui, di, is_p, is_q2) = match e.gate_proj.mapped_q4t() {
                                Some((mm, gi)) => (
                                    mm,
                                    gi,
                                    e.up_proj.mapped_q4t()?.1,
                                    e.down_proj.mapped_q4t()?.1,
                                    false,
                                    false,
                                ),
                                None => match e.gate_proj.mapped_q2tp() {
                                    Some((mm, gi)) => (
                                        mm,
                                        gi,
                                        e.up_proj.mapped_q2tp()?.1,
                                        e.down_proj.mapped_q4tp()?.1,
                                        true,
                                        true,
                                    ),
                                    None => {
                                        let (mm, gi) = e.gate_proj.mapped_q4tp()?;
                                        (
                                            mm,
                                            gi,
                                            e.up_proj.mapped_q4tp()?.1,
                                            e.down_proj.mapped_q4tp()?.1,
                                            true,
                                            false,
                                        )
                                    }
                                },
                            };
                            if *q4tp.get_or_insert(is_p) != is_p
                                || *gu_q2.get_or_insert(is_q2) != is_q2
                            {
                                return None;
                            }
                            if [gi, ui, di].into_iter().any(|idx| {
                                mm.tensors
                                    .get(idx)
                                    .is_some_and(|t| {
                                        crate::prism::is_forward_weight(mm, &t.name)
                                            || crate::prism::is_affine_target(mm, &t.name)
                                    })
                            }) {
                                return None;
                            }
                            model.get_or_insert_with(|| mm.clone());
                            experts.push((gi, ui, di));
                        }
                        crate::gpu::GraphFfn::Moe {
                            router,
                            shared_gate: sgate,
                            experts,
                            n_exp: m.experts.len(),
                            top_k: m.top_k,
                            inter,
                            norm_topk: m.norm_topk_prob,
                            q4tp: q4tp?,
                            gu_q2: gu_q2.unwrap_or(false),
                            sigmoid: m.router_sigmoid,
                            bias: m.expert_bias.as_deref(),
                            has_shared: true,
                            shared_gated,
                            route_scale: m.routed_scaling,
                        }
                    }
                    _ => return None,
                };
                let attn = match &lw.attn {
                    AttnKind::Full {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm,
                        k_norm,
                        output_gate,
                        softplus_gate,
                        bias,
                    } => {
                        if softplus_gate.is_some() || self.attention_heads_per_layer.is_some() {
                            if batch_debug {
                                eprintln!(
                                    "batch graph: unsupported Full attention gate at layer {li} softplus={} heads={}",
                                    softplus_gate.is_some(),
                                    self.attention_heads_per_layer.is_some()
                                );
                            }
                            return None;
                        }
                        let (m, _, _, _) = wq
                            .graph_weight()
                            .or_else(|| wq.graph_weight_descriptor())?;
                        model = Some(m.clone());
                        crate::gpu::GraphAttn::Full {
                            wq: gw(wq)?,
                            wk: gw(wk)?,
                            wv: gw(wv)?,
                            wo: gw(wo)?,
                            q_norm: q_norm.as_deref(),
                            k_norm: k_norm.as_deref(),
                            late_qk_norm: self.qk_norm_after_rope,
                            bias: bias
                                .as_ref()
                                .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                            output_gate: *output_gate,
                            cpu_k: self.kv_cache.layers[li].k_heads(),
                            cpu_v: self.kv_cache.layers[li].v_heads(),
                        }
                    }
                    AttnKind::LinearGdn(w) => {
                        let Some(cfg) = self.gdn_cfg else {
                            if batch_debug {
                                eprintln!("batch graph: no GDN config at layer {li}");
                            }
                            return None;
                        };
                        let (m, _, _, _) = w
                            .in_proj_qkv
                            .graph_weight()
                            .or_else(|| w.in_proj_qkv.graph_weight_descriptor())?;
                        model = Some(m.clone());
                        crate::gpu::GraphAttn::Gdn {
                            qkv: gw(&w.in_proj_qkv)?,
                            z: gw(&w.in_proj_z)?,
                            a: gw(&w.in_proj_a)?,
                            b: gw(&w.in_proj_b)?,
                            out: gw(&w.out_proj)?,
                            conv1d: &w.conv1d,
                            a_log: &w.a_log,
                            dt_bias: &w.dt_bias,
                            norm: &w.norm,
                            nv: cfg.num_v_heads,
                            nk: cfg.num_k_heads,
                            dk: cfg.key_head_dim,
                            dv: cfg.value_head_dim,
                            kk: cfg.conv_kernel,
                            cpu_state: &self.kv_cache.layers[self.phys_layer(li)].linear_state,
                        }
                    }
                    _ => return None,
                };
                layers.push(crate::gpu::GraphLayer {
                    input_norm: &lw.input_norm,
                    attn,
                    post_norm: &lw.post_norm,
                    ffn: gffn,
                });
            }
            Some((layers, model?))
        })();
        let Some((layers, model)) = built else {
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!("batch graph: BUILDER refused (layer weights/kinds)");
                }
            }
            return crate::gpu::BatchGraphOutcome::Declined;
        };
        if std::env::var("CMF_GRAPH_SPEC_TIME").is_ok() {
            eprintln!("batch-build: {:.1} ms", _tb.elapsed().as_secs_f64() * 1e3);
        }
        crate::gpu::forward_batch_graph(
            &model,
            self.graph_kv_id,
            &layers,
            &self.inv_freq,
            hiddens,
            nh,
            nkv,
            hd,
            rd,
            self.hidden_size,
            self.intermediate_size,
            positions,
            self.kv_cache.max_seq_len,
            gemma,
            self.rms_eps as f32,
            self.attn_scale,
            k,
            &(0..self.num_layers)
                .map(|li| self.kv_cache.layers[self.phys_layer(li)].o1_views())
                .collect::<Vec<_>>(),
            self.o1_epoch,
            spec,
        )
    }

    /// Same, stopping after layer `upto` inclusive (routing probe φ).
    /// `CMF_DSV4_DRAFT_PROBE=1` — grade the draft against what the trunk goes on
    /// to produce. Off by default; it runs a whole draft per decoded token.
    fn draft_probe() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_DSV4_DRAFT_PROBE").is_ok_and(|v| v != "0"))
    }

    /// `CMF_DSV4_DRAFT_PROBE=1`: measure how much of the draft the trunk
    /// would have agreed with, WITHOUT verifying or rolling anything back.
    ///
    /// The number this produces decides the whole speculation design — at
    /// acceptance a, a block of B positions yields 1 + a + a² + ... tokens
    /// per trunk pass — so it is worth measuring before any of the machinery
    /// that would exploit it exists. Each draft is parked with the position
    /// it was made at, and graded as the real tokens arrive.
    /// `CMF_DSV4_SPEC=1` — the DeepSeek-V4 speculative decode: draft five
    /// on the card, verify them in one batched trunk pass, commit the
    /// accepted prefix, roll the rest back.
    #[cfg(feature = "gpu")]
    fn dsv4_spec_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            // Test-only runtime gate: model loading still performs the same
            // reservation and trunk packing, which gives rollback parity a
            // topology-identical non-speculative control arm.
            if let Ok(v) = std::env::var("CMF_DSV4_SPEC_RUN") {
                return v != "0";
            }
            // An explicit value is a diagnostic force/escape hatch.  With no
            // knob, speculation is eligible only when model loading reserved
            // its bounded pack.  On small q4tp cards the geometric reserve
            // gate deliberately leaves this at zero: trying to build DSpark
            // after the exact trunk filled VRAM is both slower and a device
            // OOM (measured on A40).
            std::env::var("CMF_DSV4_SPEC")
                .map(|v| v != "0")
                .unwrap_or_else(|_| {
                    crate::gpu_wgpu::DRAFT_RESERVE.load(std::sync::atomic::Ordering::Relaxed) > 0
                })
        })
    }

    /// One speculative round at the decode tip. `t_next` is the token the
    /// sampler just committed for `next_pos`. Returns the EXTRA accepted
    /// tokens (possibly none) and the new position, with `graph_logits`
    /// left holding the last accepted position's logits — exactly what the
    /// loop top expects. `None` means "speculate not this round": nothing
    /// was committed, the caller forwards normally.
    #[cfg(feature = "gpu")]
    fn dsv4_spec_step(
        &mut self,
        tip_token: u32,
        t_next: u32,
        next_pos: usize,
        max_extra: usize,
        drafted: &mut usize,
        accepted_ctr: &mut usize,
    ) -> Option<(Vec<u32>, usize)> {
        let t_all = std::time::Instant::now();
        if std::env::var("CMF_DSV4_SPEC_TIME").is_ok() {
            thread_local! {
                static LAST: std::cell::Cell<Option<std::time::Instant>> =
                    const { std::cell::Cell::new(None) };
            }
            LAST.with(|l| {
                if let Some(prev) = l.get() {
                    eprintln!(
                        "между раундами {:.1} мс",
                        prev.elapsed().as_secs_f64() * 1e3
                    );
                }
                l.set(Some(std::time::Instant::now()));
            });
        }
        if std::env::var("CMF_DSV4_SPEC_DEBUG").is_ok() {
            eprintln!("spec_step: вход pos={next_pos}");
        }
        let n_layers = self.dsv4.as_ref().map(|b| b.1.len())?;
        let cfg = self.dsv4.as_ref().map(|b| b.2)?;
        // The draft state and its capture, armed exactly as the probe does.
        if self.dspark.is_none() {
            let t = crate::dsv4::dspark_targets(&self.dsv4_mtp, &cfg, n_layers);
            if t.is_empty() {
                return None;
            }
            crate::dsv4::dspark_arm(&t, cfg.dim);
            self.dspark = Some(crate::dsv4::DsparkState::new(
                self.dsv4_mtp.len(),
                &cfg,
                t.len(),
            ));
        }
        let targets = crate::dsv4::dspark_targets(&self.dsv4_mtp, &cfg, n_layers);
        let pack = crate::dsv4::dspark_pack_get(&self.dsv4_mtp, &cfg);
        if pack.is_none() && std::env::var("CMF_DSV4_SPEC_DEBUG").is_ok() {
            eprintln!("spec_step: пак не построился (targets {targets:?})");
        }
        let pack = pack?;
        let block = crate::dsv4::dspark_block();
        let b_box = self.dsv4.as_mut()?;
        let (g, layers, st) = (&b_box.0, &b_box.1, &mut b_box.3);
        let ds = self.dspark.as_mut()?;
        // The tip's captures: either this token ran on a normal path that
        // filled the thread-local, or the previous spec round left them.
        let dbg = std::env::var("CMF_DSV4_SPEC_DEBUG").is_ok();
        if !crate::dsv4::dspark_take(&mut ds.main_hidden) && !ds.have_hidden {
            if dbg {
                eprintln!("spec_step: нет захвата");
            }
            return None;
        }
        ds.have_hidden = true;
        let tip_pos = next_pos.checked_sub(1)?;
        let draft_started = std::time::Instant::now();
        let mut conf = Vec::new();
        let props = crate::dsv4::dspark_draft_gpu(
            g,
            &self.dsv4_mtp,
            &cfg,
            ds,
            pack,
            st.kv_id,
            tip_token,
            tip_pos,
            self.pool.as_deref(),
            &mut conf,
        );
        self.dspark_draft_ns += draft_started.elapsed().as_nanos();
        *drafted += block;
        if props.is_empty() || props[0] != t_next {
            if dbg {
                eprintln!(
                    "spec_step: черновик {} (props0={:?} t_next={t_next})",
                    if props.is_empty() {
                        "пуст"
                    } else {
                        "мимо"
                    },
                    props.first()
                );
            }
            return None;
        }
        // `fed[0]` is `t_next`, which the outer loop has already committed;
        // only `fed[1..]` become additional output tokens. Cap the verify
        // transaction itself to the caller's remaining output budget instead
        // of merely truncating the returned vector: otherwise the KV/state
        // would advance past `max_tokens` and a 64-token request could return
        // 66 tokens (and poison a reused session with two invisible steps).
        let mut k_verify = crate::dsv4::dspark_verify_k()
            .min(props.len())
            .min(max_extra.saturating_add(1));
        // Adaptive depth: positions the draft itself doubts are paid for on
        // every verify and delivered almost never (natural-text survival
        // [.67 .50 .29 .08 .04]). `CMF_DSPARK_CONF_MIN=p` trims the fed
        // prefix at the first proposal whose confidence drops below p; on
        // predictable text the confidences stay high and nothing changes.
        let conf_min = {
            static M: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            *M.get_or_init(|| {
                std::env::var("CMF_DSPARK_CONF_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0)
            })
        };
        if conf_min > 0.0 && conf.len() >= props.len() {
            let mut keep = 1usize;
            while keep < k_verify && conf.get(keep).copied().unwrap_or(0.0) >= conf_min {
                keep += 1;
            }
            k_verify = k_verify.min(keep.max(2));
        }
        if k_verify < 2 {
            return None;
        }
        let mut fed = Vec::with_capacity(k_verify);
        fed.push(t_next);
        fed.extend_from_slice(&props[1..k_verify]);
        let mut argmax = Vec::new();
        let mut logits_all = Vec::new();
        let mut walked = Vec::new();
        let txn = crate::dsv4::dsv4_verify_chunk(
            g,
            layers,
            &cfg,
            st,
            &fed,
            next_pos,
            &self.inv_freq,
            self.pool.as_deref(),
            &targets,
            &mut argmax,
            &mut logits_all,
            &mut walked,
        );
        if txn.is_none() && dbg {
            eprintln!("spec_step: verify отказал");
        }
        let txn = txn?;
        let spec_gpu_end = txn.gpu_end;
        let b = fed.len();
        let mut accepted = 1usize;
        while accepted < b && fed[accepted] == argmax[accepted - 1] {
            accepted += 1;
        }
        // `CMF_DSV4_SPEC_FORCE_REJECT=1` — accept nothing beyond the known
        // token, every round: the pure rollback exerciser. The output must
        // stay byte-identical to the plain walk; anything else is a
        // transaction bug, isolated from the acceptance logic.
        if std::env::var("CMF_DSV4_SPEC_FORCE_REJECT").is_ok_and(|v| v != "0") {
            accepted = 1;
        }
        if std::env::var("CMF_DSV4_SPEC_TRACE").is_ok() {
            eprintln!("spec@{next_pos}: fed={fed:?} argmax={argmax:?} accepted={accepted}");
        }
        let t_fin = std::time::Instant::now();
        if !crate::dsv4::dsv4_spec_finish(
            g,
            layers,
            &cfg,
            st,
            txn,
            accepted,
            &fed,
            &self.inv_freq,
            self.pool.as_deref(),
        ) {
            tracing::warn!("dsv4: спекулятивный откат не удался — состояние подозрительно");
            return None;
        }
        if std::env::var("CMF_DSV4_SPEC_TIME").is_ok() {
            eprintln!(
                "finish(k={accepted}): {:.1} мс",
                t_fin.elapsed().as_secs_f64() * 1e3
            );
        }
        *accepted_ctr += accepted - 1;
        // Captures per accepted token: device targets photographed by the
        // batch, host targets from the verify's own walk. The last one
        // becomes the new tip's draft input; every one owes the ring an
        // entry for its position.
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        // Complete-chain layers are photographed by the fused submission;
        // partial device layers overwrite that slot after exact host cold-
        // expert correction.  Thus every target in the contiguous device
        // prefix has a valid per-token capture.
        let dev_caps: Vec<usize> = targets
            .iter()
            .copied()
            .filter(|&t| t < spec_gpu_end)
            .collect();
        let mut caps_all = vec![0.0f32; dev_caps.len() * b * hc * dim];
        if !crate::gpu_wgpu::dsv4_spec_cap_read_all(b, dev_caps.len(), hc * dim, &mut caps_all) {
            return None;
        }
        for t in 0..accepted {
            let tip = t + 1 == accepted;
            for (slot, &tl) in targets.iter().enumerate() {
                if let Some(di) = dev_caps.iter().position(|&d| d == tl) {
                    let lo = (di * b + t) * hc * dim;
                    crate::dsv4::dspark_capture(
                        &caps_all[lo..lo + hc * dim],
                        &cfg,
                        slot,
                        &mut ds.main_hidden,
                    );
                } else if tip
                    && crate::dsv4::dspark_peek_slot(slot, dim, {
                        let lo = slot * dim;
                        &mut ds.main_hidden[lo..lo + dim]
                    })
                {
                    // The tip's host-layer captures are the walk's own
                    // per-layer notes — exact. (The walk that ran last ended
                    // on exactly this token, on both the accept-all and the
                    // rollback path.)
                } else {
                    // Intermediate tokens: the post-tail state stands in for
                    // the per-layer capture on host targets below the last
                    // layer. Ring-entry quality only; the tip is exact.
                    crate::dsv4::dspark_capture(
                        &walked[t * hc * dim..(t + 1) * hc * dim],
                        &cfg,
                        slot,
                        &mut ds.main_hidden,
                    );
                }
            }
            crate::dsv4::dspark_ring_append(
                g,
                &self.dsv4_mtp,
                &cfg,
                ds,
                next_pos + t,
                self.pool.as_deref(),
            );
        }
        let row = logits_all[(accepted - 1) * cfg.vocab..accepted * cfg.vocab].to_vec();
        self.graph_logits = Some(row);
        // The speculative loop never runs the probe, so the trunk tally has
        // no other place to cycle. Armed only when someone asked for the
        // dump; the host tail is the only tallying path here, which is
        // precisely the population a partial pack would serve.
        if std::env::var("CMF_DSV4_TRUNK_PICK_DUMP").is_ok() {
            crate::dsv4::trunk_freq_note(&crate::dsv4::pick_tally_take());
            crate::dsv4::pick_tally_arm();
        }
        if std::env::var("CMF_DSV4_SPEC_TIME").is_ok() {
            eprintln!(
                "spec_step total {:.1} мс (k={accepted})",
                t_all.elapsed().as_secs_f64() * 1e3
            );
        }
        Some((fed[1..accepted].to_vec(), next_pos + accepted))
    }

    fn dspark_probe(&mut self, position: usize, token_id: u32) {
        if self.dsv4_mtp.is_empty() || !Self::draft_probe() {
            return;
        }
        // What the trunk just routed to, for this token.
        let trunk_now = crate::dsv4::pick_tally_take();
        crate::dsv4::trunk_freq_note(&trunk_now);
        if !trunk_now.is_empty() {
            self.dspark_trunk_picks.push(trunk_now);
            let keep = crate::dsv4::dspark_block();
            if self.dspark_trunk_picks.len() > keep {
                self.dspark_trunk_picks.remove(0);
            }
        }
        // Grade whatever is waiting: the token just decoded sits at
        // `position`, so it answers the draft made at `position - 1 - i`.
        for p in std::mem::take(&mut self.dspark_pending) {
            let Some(i) = position.checked_sub(p.0 + 1) else {
                continue;
            };
            let mut p = p;
            if i < p.1.len() {
                if p.2 && p.1[i] == token_id {
                    p.3 = i + 1;
                } else {
                    p.2 = false;
                }
                if i + 1 < p.1.len() {
                    self.dspark_pending.push(p);
                    continue;
                }
            }
            self.dspark_hist.push(p.3);
            self.dspark_real.push(token_id);
        }
        let Some(b) = &mut self.dsv4 else { return };
        let (g, layers, cfg) = (&b.0, &b.1, b.2);
        let n_layers = layers.len();
        if self.dspark.is_none() {
            let t = crate::dsv4::dspark_targets(&self.dsv4_mtp, &cfg, n_layers);
            if t.is_empty() {
                return;
            }
            eprintln!(
                "DSpark: захват со слоёв {t:?}, блок {}",
                crate::dsv4::dspark_block()
            );
            crate::dsv4::dspark_arm(&t, cfg.dim);
            self.dspark = Some(crate::dsv4::DsparkState::new(
                self.dsv4_mtp.len(),
                &cfg,
                t.len(),
            ));
        }
        let ds = self.dspark.as_mut().unwrap();
        if !crate::dsv4::dspark_take(&mut ds.main_hidden) {
            return; // this token ran on a path that captures nothing
        }
        let mut conf = Vec::new();
        crate::dsv4::pick_tally_arm();
        // The trunk has already consumed the adaptive VRAM budget. Until the
        // draft owns an explicit bounded device pack, its tensors are an
        // out-of-core CPU/disk tier by contract: never let per-op probes try
        // to squeeze another multi-gigabyte MTP expert cache onto the card.
        let draft_started = std::time::Instant::now();
        #[cfg(feature = "gpu")]
        let gpu_draft = crate::dsv4::dspark_gpu_on();
        #[cfg(not(feature = "gpu"))]
        let gpu_draft = false;
        let props = if gpu_draft {
            #[cfg(feature = "gpu")]
            {
                let kv_id = b.3.kv_id;
                match crate::dsv4::dspark_pack_get(&self.dsv4_mtp, &cfg) {
                    Some(pk) => crate::dsv4::dspark_draft_gpu(
                        g,
                        &self.dsv4_mtp,
                        &cfg,
                        ds,
                        pk,
                        kv_id,
                        token_id,
                        position,
                        self.pool.as_deref(),
                        &mut conf,
                    ),
                    None => Vec::new(),
                }
            }
            #[cfg(not(feature = "gpu"))]
            Vec::new()
        } else {
            crate::gpu::cpu_scope(|| {
                crate::dsv4::dspark_draft(
                    g,
                    &self.dsv4_mtp,
                    &cfg,
                    ds,
                    token_id,
                    position,
                    self.pool.as_deref(),
                    &mut conf,
                )
            })
        };
        self.dspark_draft_ns += draft_started.elapsed().as_nanos();
        let draft_picks = crate::dsv4::pick_tally_take();
        crate::dsv4::dspark_freq_note(&draft_picks);
        // Re-arm for the NEXT trunk token; the probe runs after the forward,
        // so this is the only place that can.
        crate::dsv4::pick_tally_arm();
        if !props.is_empty() {
            // Two ratios, side by side: what a batched verify over the trunk
            // would read against what it asks for, and the same for the
            // draft's three stages. Near 1.0 means a batch amortises nothing.
            let (tu, tt) = {
                let flat: Vec<(usize, Vec<usize>)> = self
                    .dspark_trunk_picks
                    .iter()
                    .flat_map(|v| v.iter().cloned())
                    .collect();
                // Per layer, across the window of tokens.
                let mut per: std::collections::HashMap<usize, Vec<usize>> =
                    std::collections::HashMap::new();
                for (li, picks) in flat {
                    per.entry(li).or_default().extend(picks);
                }
                let n = per.len().max(1);
                let mut u = 0usize;
                let mut t = 0usize;
                for (_, v) in per {
                    t += v.len();
                    u += v.iter().collect::<std::collections::HashSet<_>>().len();
                }
                (u / n, t / n)
            };
            let (du, dt) = crate::dsv4::tally_unique(&draft_picks);
            self.dspark_exp.push((tu, tt, du, dt));
            self.dspark_pending.push((position, props, true, 0));
        }
        if self.dspark_hist.len() >= 8 && self.dspark_hist.len() % 8 == 0 {
            let n = self.dspark_hist.len() as f32;
            let mean: f32 = self.dspark_hist.iter().sum::<usize>() as f32 / n;
            let block = crate::dsv4::dspark_block();
            let mut at = vec![0usize; block + 1];
            for &k in &self.dspark_hist {
                at[k] += 1;
            }
            // Prefix survival: S_i = P(the first i positions all held).
            let mut surv = Vec::with_capacity(block);
            for i in 1..=block {
                let k = at[i..].iter().sum::<usize>() as f32 / n;
                surv.push(format!("{k:.2}"));
            }
            let distinct = self
                .dspark_real
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            let (tu, tt, du, dt) = self.dspark_exp.iter().fold((0, 0, 0, 0), |a, b| {
                (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
            });
            let m = self.dspark_exp.len().max(1);
            eprintln!(
                "DSpark: черновиков {}, принято в среднем {mean:.2} из {block} \
                 (токенов за проход {:.2}), распределение {at:?}, выживание [{}]",
                self.dspark_hist.len(),
                mean + 1.0,
                surv.join(" ")
            );
            eprintln!(
                "DSpark: разных токенов {distinct} из {} (вырожденность), \
                 эксперты ствол {}/{} на слой за {block} токенов, \
                 черновик {}/{} за блок, draft {:.2} мс/блок",
                self.dspark_real.len(),
                tu / m,
                tt / m,
                du / m,
                dt / m,
                self.dspark_draft_ns as f64 / self.dspark_exp.len().max(1) as f64 / 1e6
            );
        }
    }

    fn forward_layers_upto(
        &mut self,
        hidden: &[f32],
        position: usize,
        task_mask: Option<&TaskMask>,
        upto: Option<usize>,
    ) -> Vec<f32> {
        // In-process multi-GPU: each segment runs pinned to its card,
        // and the only thing crossing the boundary is one hidden vector
        // that never leaves this address space. Same layer split the
        // network mode does, minus the second process, the socket, the
        // serialization and the dir_hash handshake.
        if let Some(plan) = self.gpu_plan.clone() {
            if upto.is_none() && plan.len() > 1 {
                let mut h = hidden.to_vec();
                for &(dev, from, upto_incl) in plan.iter() {
                    h = crate::gpu::with_device(dev, || {
                        self.forward_layers_span(&h, position, task_mask, from, Some(upto_incl))
                    });
                }
                return h;
            }
        }
        self.forward_layers_span(hidden, position, task_mask, 0, upto)
    }

    /// Split this pipeline's layer stack across local GPUs: segment i
    /// runs on `devices[i]`. Contiguous and even by layer count — the
    /// VRAM-weighted planner is the next step, and an uneven card pair
    /// is why it will be needed. `None` clears the plan.
    pub fn set_gpu_plan(&mut self, devices: Option<&[usize]>) -> Result<(), String> {
        self.set_gpu_plan_at(devices, None)
    }

    /// The same, with an explicit first boundary (`--peer-split`): card
    /// 0 takes layers `[0..at)`, the rest split what remains. Uneven
    /// cards, or an attention-heavy head, are why this knob exists.
    pub fn set_gpu_plan_at(
        &mut self,
        devices: Option<&[usize]>,
        at: Option<usize>,
    ) -> Result<(), String> {
        let Some(devs) = devices.filter(|d| d.len() > 1) else {
            self.gpu_plan = None;
            return Ok(());
        };
        self.split_supported()?;
        let n = self.num_layers;
        if devs.len() > n {
            return Err(format!("{} devices for {n} layers", devs.len()));
        }
        if let Some(k) = at {
            if k == 0 || k >= n {
                return Err(format!("split at {k}: the model has {n} layers"));
            }
            if devs.len() == 2 {
                self.gpu_plan = Some(std::sync::Arc::new(vec![
                    (devs[0], 0, k - 1),
                    (devs[1], k, n - 1),
                ]));
                return Ok(());
            }
            return Err(format!(
                "an explicit split point takes exactly 2 devices, got {}",
                devs.len()
            ));
        }
        let per = n.div_ceil(devs.len());
        let mut plan = Vec::with_capacity(devs.len());
        let mut from = 0usize;
        for &d in devs {
            if from >= n {
                break;
            }
            let upto = (from + per - 1).min(n - 1);
            plan.push((d, from, upto));
            from = upto + 1;
        }
        self.gpu_plan = Some(std::sync::Arc::new(plan));
        Ok(())
    }

    /// The active in-process split, if any: (device, first layer, last).
    pub fn gpu_plan(&self) -> Option<Vec<(usize, usize, usize)>> {
        self.gpu_plan.as_ref().map(|p| p.as_ref().clone())
    }

    /// Layer span [from ..= upto] (upto None = last layer): the building
    /// block the network pipeline-split rides on. `from > 0` skips the
    /// arch escape hatches (the pub `forward_span` refuses those archs
    /// first) and the whole-token graph — the plain per-layer loop is
    /// the canonical executor for a partial stack.
    fn forward_layers_span(
        &mut self,
        hidden: &[f32],
        position: usize,
        task_mask: Option<&TaskMask>,
        from: usize,
        upto: Option<usize>,
    ) -> Vec<f32> {
        debug_assert!(
            from == 0
                || (self.dsv4.is_none()
                    && self.dsv41.is_none()
                    && self.qwen4_exp.is_none()
                    && self.g3n.is_none())
        );
        // Every plain forward — the whole-token Metal graph (`q1_graph_gpu`
        // wraps the GDN owners zero-copy and reallocates them on a size
        // change) and the CPU layer loop (reads/swaps `linear_state`) —
        // must see the previous speculative commit's asynchronous replay
        // complete. One mutex probe when nothing is pending.
        #[cfg(target_os = "macos")]
        if !crate::gpu_metal::wait_replay() {
            self.fail_metal_graph("the pending async replay failed before a plain forward");
            return vec![0.0; self.hidden_size];
        }
        if let Some(b) = &mut self.qwen4_exp {
            let _ = (task_mask, upto);
            let token_id = hidden.first().copied().unwrap_or(0.0) as u32;
            let mut logits = Vec::new();
            crate::qwen4_exp::forward_token(
                &b.0,
                &b.1,
                &b.2,
                &mut b.3,
                token_id,
                position,
                &self.inv_freq,
                self.pool.as_deref(),
                &mut logits,
                true,
            );
            self.graph_logits = Some(logits);
            return vec![0.0; self.hidden_size];
        }
        // DeepSeek-V4 runs its own stack: the state is hc_mult copies, and
        // the forward returns LOGITS, not a hidden — the head is inside it
        // (the final fold sits between the last layer and the norm). The
        // token id rides in `hidden[0]`, written by embed_single, because
        // the hash layers route by id rather than by content.
        if let Some(b) = &mut self.dsv4 {
            let _ = (task_mask, upto);
            let token_id = hidden.first().copied().unwrap_or(0.0) as u32;
            let (g, layers, cfg, st) = (&b.0, &b.1, b.2, &mut b.3);
            st.pos = position;
            let mut logits = Vec::new();
            crate::dsv4::forward_token(
                g,
                layers,
                &cfg,
                st,
                token_id,
                &self.inv_freq,
                self.pool.as_deref(),
                &mut logits,
            );
            self.graph_logits = Some(logits);
            self.dspark_probe(position, token_id);
            // The caller expects a hidden; the logits went out of band, as
            // with the fused lm_head path.
            return vec![0.0; self.hidden_size];
        }
        // DeepSeek-V4.1 owns its complete stack and emits logits out of band.
        if let Some(b) = &mut self.dsv41 {
            let _ = (task_mask, upto);
            let token_id = hidden.first().copied().unwrap_or(0.0) as u32;
            let mut logits = Vec::new();
            crate::dsv41::forward_token(
                &b.0,
                &b.1,
                &b.2,
                &mut b.3,
                token_id,
                position,
                self.pool.as_deref(),
                &mut logits,
            );
            self.graph_logits = Some(logits);
            return vec![0.0; self.hidden_size];
        }
        // Gemma-3n runs its own stack (4 AltUp replicas don't fit this
        // loop); `hidden` is the extended embedding from embed_single.
        if let Some(b) = &self.g3n {
            let _ = (task_mask, upto);
            return crate::g3n::g3n_forward(
                &b.0,
                &b.1,
                hidden,
                position,
                &mut self.kv_cache.layers,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.pool.as_deref(),
            );
        }
        let mut h = hidden.to_vec();
        // Split borrows: copy scalars / clone handles so the per-layer
        // cfg does not hold `&self` while the KV cache is `&mut`.
        let (nh, _nkv, _hd, hs, _rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let pool = self.pool.clone();
        // Opt-in wgpu token-graph attention (discrete Vulkan/DX12): the whole
        // attention sub-block runs resident in one submit. Off by default.
        // Whole-token wgpu graph: eligibility + arbitration.
        //  - explicit CMF_GPU_WGPU_GRAPH forces it on/off;
        //  - discrete adapters (4090: decode 76 -> 137 tok/s) and GDN
        //    hybrids (recurrent state device-resident, no CPU twin to
        //    race) TRUST it;
        //  - integrated/mobile adapters RACE it against the normal path
        //    at generation granularity (gpu::graph_race_*) — tiled
        //    mobile GPUs can turn the ~300-dispatch graph into seconds
        //    per token, while a fast phone GPU keeps its win.
        let graph_env = std::env::var("CMF_GPU_WGPU_GRAPH").ok();
        let graph_on = match graph_env.as_deref() {
            Some("0") => false,
            Some("prefill") => false, // decode keeps the per-op path
            Some(_) => true,
            // Unset: same discrete-only default as every other graph
            // site. "Is the GPU on" used to stand in here — which made
            // the 0.2 tok/s whole-token graph race-eligible on mobile
            // adapters and cost 12-14× on first tokens (cmfmobile
            // TUNING.md); integrated GPUs keep the per-op probe path.
            None => crate::gpu::wgpu_graph_default(),
        };
        let graph_trusted =
            graph_env.is_some() || crate::gpu::wgpu_graph_default() || self.gdn_cfg.is_some();
        let race_eligible = graph_on
            && upto.is_none()
            && task_mask.is_none()
            && from == 0
            && !crate::gpu::graph_unsupported();
        let mut tail_start = 0usize;
        if race_eligible && crate::gpu::graph_race_use_graph(graph_trusted) {
            let t_graph = std::time::Instant::now();
            let mut lg = Vec::new();
            let mut gl = 0usize;
            let built = self.try_token_graph_wgpu(hidden, position, &mut lg, &mut gl);
            let declined = built.is_none();
            let built = match built {
                Some(Ok(hh)) => Some(hh),
                Some(Err(())) => {
                    // O(1) state was admitted before the device failure; the
                    // CPU mirrors are stale by construction.  Clear the whole
                    // sequence and stop rather than walking that stale state.
                    self.clear_sequence_state();
                    self.graph_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::error!("token graph failed after admission; sequence state cleared");
                    return vec![0.0; self.hidden_size];
                }
                None => None,
            };
            // Past the transient guards (o1 still collecting, a softcap)
            // a refusal is about the weights and will never change —
            // remember it instead of walking every layer again next
            // token.
            if declined && !self.o1_active() && self.attn_softcap == 0.0 {
                crate::gpu::graph_mark_unsupported();
            }
            graph_note(built.is_some(), gl, self.num_layers);
            if let Some(hh) = built {
                let dur = t_graph.elapsed();
                if std::env::var("CMF_GRAPH_PROF").is_ok() {
                    eprintln!("graph-call: {:.2} ms total", dur.as_secs_f64() * 1000.0);
                }
                if gl > 0 && gl < self.num_layers {
                    // Device prefix: the graph ran layers 0..gl and handed
                    // back the boundary hidden — the loop below owns the
                    // tail. The prefix layers' KV/state advanced on the
                    // device; the tail's advances on the host below. One
                    // boundary crossing per token.
                    h = hh;
                    tail_start = gl;
                } else if graph_trusted || !crate::gpu::graph_race_first_token_hopeless(dur) {
                    if !graph_trusted {
                        crate::gpu::graph_race_record(true, dur);
                    }
                    if !lg.is_empty() {
                        // Graph produced logits (final-norm + lm_head folded in) —
                        // pad/cap to vocab and hand them to the sampler directly.
                        lg.resize(self.vocab_size, 0.0);
                        if let Some(c) = self.final_softcap {
                            for l in lg.iter_mut() {
                                *l = c * (*l / c).tanh();
                            }
                        }
                        self.graph_logits = Some(lg);
                    }
                    return hh;
                }
                // Hopeless first graph token: discard it and fall through
                // to the normal path. Safe exactly here — the prompt KV is
                // still CPU-owned (chunked prefill), so recomputing this
                // position is exact; the mirror's extra row is never read
                // (the race just settled on the normal path).
            }
        }
        // KIMI-LINEAR HAS NO SPLIT BUG. The 2.6× reported from the
        // model rotation (12.2 tok/s on one card against 4.6 on two)
        // was a single measurement of a model whose arm arbitration is
        // borderline, and it did not survive repetition. Three runs an
        // arm, same binary, back to back:
        //   probe on : 1 GPU 9.5 / 5.7 / 5.9   2 GPU 7.8 / 13.0 / 13.3
        //   pinned   : 1 GPU 5.6 / 5.3 / 5.2   2 GPU 3.5 / 4.2 / 3.4
        // With the arms pinned the split costs about 1.45×, which is
        // what a layer split costs. With the probe free, TWO CARDS RUN
        // FASTER — because for this model the CPU arm wins some op
        // classes and the probe finds that.
        //
        // Two things do stand, and both are measured. The token graph
        // builds NOTHING here (`covered 0 of 14 layers [0..14)`), so
        // every layer walks per-op on either arm — that is where the
        // headroom is, not in the split. And this model's benchmark is
        // unusable without `CMF_GPU_PROBE=0`: the arbitration alone
        // moves it by more than 2×.
        //
        // Span runs (network split): the graph covers exactly [from..=upto]
        // — one submit per SEGMENT per token. No race: its state is global
        // and calibrated on full stacks, so spans take the graph only where
        // it is trusted by default (discrete adapters / CMF_GPU_WGPU_GRAPH).
        let span = from > 0 || upto.is_some();
        if span && graph_on && task_mask.is_none() && graph_trusted {
            let upto_excl = upto.map_or(self.num_layers, |u| u + 1);
            let mut lg = Vec::new();
            let mut gl = 0usize;
            let span_res =
                self.try_token_graph_wgpu_span(hidden, position, &mut lg, from, upto_excl, &mut gl);
            let span_res = match span_res {
                Some(Ok(hh)) => Some(hh),
                Some(Err(())) => {
                    self.clear_sequence_state();
                    self.graph_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::error!(
                        "span token graph failed after admission; sequence state cleared"
                    );
                    return vec![0.0; self.hidden_size];
                }
                None => None,
            };
            graph_note(span_res.is_some(), gl, upto_excl - from);
            if std::env::var("CMF_GPU_DEBUG").is_ok() {
                // How much of the span the graph actually covered. A
                // prefix of nothing means every layer walks per-op and
                // the split's extra cost is elsewhere.
                static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    eprintln!(
                        "span graph: covered {gl} of {} layers [{from}..{upto_excl}) res={}",
                        upto_excl - from,
                        span_res.is_some()
                    );
                }
            }
            if let Some(hh) = span_res {
                if gl == upto_excl - from {
                    if !lg.is_empty() {
                        lg.resize(self.vocab_size, 0.0);
                        if let Some(c) = self.final_softcap {
                            for l in lg.iter_mut() {
                                *l = c * (*l / c).tanh();
                            }
                        }
                        self.graph_logits = Some(lg);
                    }
                    crate::gpu::set_layer(-1);
                    return hh;
                }
                // Partial device prefix of the span: CPU owns the tail.
                h = hh;
                tail_start = from + gl;
            }
        }
        let t_race_cpu = (race_eligible && !graph_trusted).then(std::time::Instant::now);

        // A partial graph is an explicit GPU-prefix / CPU-tail split. Keep
        // the tail PURE host-side: letting its QTensor hooks re-enter the
        // residency arena streams every omitted layer through Vulkan and the
        // driver's freed-allocation cache can grow to the full model size
        // (25.4 GiB observed with a 14 GiB budget on Granite 30B Q8_2F).
        let _host_tail = (tail_start > from).then(crate::gpu::enter_cpu_scope);
        let automatic_gpu_prefix = self.automatic_gpu_prefix();

        #[cfg(target_os = "macos")]
        let mut gpu_skip_until = 0usize;
        for li in tail_start.max(from)..self.num_layers {
            let _capacity_tail = automatic_gpu_prefix
                .filter(|&prefix| li >= prefix)
                .map(|_| crate::gpu::enter_cpu_scope());
            crate::gpu::set_layer(li as i64); // layer-split GPU/CPU (CMF_GPU_LAYERS)
            if let Some(u) = upto {
                if li > u {
                    break;
                }
            }
            if let Some(mask) = task_mask {
                if !mask.layer_alive(li) {
                    continue; // dead layer: residual pass-through
                }
            }
            // Whole-block q1 token graph: a run of consecutive q1
            // layers — GDN and full attention — executes with one sync
            // per CPU attend instead of per op (macOS/Metal).
            #[cfg(target_os = "macos")]
            {
                if li < gpu_skip_until {
                    continue;
                }
                if task_mask.is_none() {
                    let end = self.q1_graph_gpu(li, upto, position, &mut h);
                    if self
                        .graph_failed
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        // The graph may have mutated device state before a
                        // command-buffer error. Never continue with a CPU
                        // tail or read a stale host mirror after admission.
                        return vec![0.0; self.hidden_size];
                    }
                    if end > li {
                        gpu_skip_until = end;
                        // Looped Transformer: the graph stopped at a loop
                        // boundary — apply final norm before the next iteration.
                        if self.is_loop_end(end - 1) && end < self.num_layers {
                            h = inference::rms_norm(
                                &h,
                                &self.weights.final_norm,
                                self.rms_eps,
                                self.norm_style,
                            );
                        }
                        continue;
                    }
                }
            }

            let lw = &self.weights.layers[self.phys_layer(li)];
            if let Ok(tp) = std::env::var("CMF_TRACE_POS") {
                if tp.parse::<usize>().ok() == Some(position) {
                    let n: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "TRACE pos {position} layer {li}: |h| = {n:.6} h0 {:.6} h1 {:.6}",
                        h[0], h[1]
                    );
                }
            }
            // Norm into the pipeline scratch — the returning rms_norm
            // allocated twice per layer per token (roadmap §3 P0).
            inference::rms_norm_into(
                &h,
                &lw.input_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.n1,
            );

            let attn_out = match &lw.attn {
                AttnKind::Mla(w) => {
                    let inv_freq_l = self.layer_inv_freq(li);
                    let rs = self.layer_rope_scale(li);
                    let eps = self.rms_eps;
                    let pool = self.pool.clone();
                    mla_attention(
                        w,
                        &self.ws.n1,
                        &mut self.kv_cache.layers[li],
                        position,
                        &inv_freq_l,
                        rs,
                        eps,
                        pool.as_deref(),
                    )
                }
                AttnKind::Linear(w) => {
                    let cfg = self.vmf_cfg.expect("linear layer without vmf_cfg");
                    vmf_phase_forward(
                        &self.ws.n1,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::Kda(w) => {
                    let cfg = self.kda_cfg.expect("kda layer without kda_cfg");
                    crate::linear_core::kda_forward(
                        &self.ws.n1,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::LinearGdn(w) => {
                    let cfg = self.gdn_cfg.expect("gdn layer without gdn_cfg");
                    gdn_forward(
                        &self.ws.n1,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::ShortConv(w) => {
                    let cfg = self
                        .short_conv_cfg
                        .expect("short-conv layer without short_conv_cfg");
                    short_conv_forward(
                        &self.ws.n1,
                        w,
                        &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        self.pool.as_deref(),
                    )
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate,
                    bias,
                } if self.kv_cache.layers[li].o1_sealed() => {
                    // O(1) override: decode on the sealed Nyström state
                    // instead of the growing KV cache.
                    let inv_freq_l = self.layer_inv_freq(li);
                    let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                    let cfg = QwenAttnCfg {
                        num_heads: self.layer_num_heads(li),
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        softcap: self.attn_softcap,
                        window: None,
                        v_norm: self.attn_v_norm,
                        qk_norm_after_rope: self.qk_norm_after_rope,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        softplus_gate: softplus_gate
                            .as_ref()
                            .map(|(gate, per_head)| (gate, *per_head)),
                        rope_scale: self.layer_rope_scale(li),
                        bias: bias
                            .as_ref()
                            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        rms_eps: eps,
                        norm_style: self.norm_style,
                        pool: pool.as_deref(),
                    };
                    attention::qwen_attention_nystrom(
                        &self.ws.n1,
                        wq,
                        wk,
                        wv,
                        wo,
                        &mut self.kv_cache.layers[li],
                        &cfg,
                    )
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    softplus_gate,
                    bias,
                } => 'attn: {
                    // wgpu token-graph attention (opt-in): whole sub-block in
                    // one submit, device K/V mirror. q1 only, no gate/bias/mask.
                    if graph_on
                        && !*output_gate
                        && softplus_gate.is_none()
                        && self.attention_heads_per_layer.is_none()
                        && bias.is_none()
                        && task_mask.is_none()
                    {
                        let inv_freq_l = self.layer_inv_freq(li);
                        let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
                        if let (Some((gm, qi)), Some((_, ki)), Some((_, vi)), Some((_, oi))) = (
                            wq.mapped_q1(),
                            wk.mapped_q1(),
                            wv.mapped_q1(),
                            wo.mapped_q1(),
                        ) {
                            let gm = gm.clone();
                            let mut out = vec![0f32; hs];
                            let cache = &self.kv_cache.layers[li];
                            if crate::gpu::attn_dropin(
                                &gm,
                                self.graph_kv_id,
                                li,
                                &self.ws.n1,
                                qi,
                                ki,
                                vi,
                                oi,
                                q_norm.as_deref(),
                                k_norm.as_deref(),
                                self.qk_norm_after_rope,
                                &inv_freq_l,
                                nh,
                                nkv_l,
                                hd_l,
                                rd_l,
                                hs,
                                position,
                                self.kv_cache.max_seq_len,
                                gemma,
                                eps as f32,
                                cache.k_heads(),
                                cache.v_heads(),
                                &mut out,
                            ) {
                                break 'attn out;
                            }
                        }
                    }
                    let masked = task_mask
                        .map(|m| m.head_flags(li, self.num_heads).iter().any(|&a| !a))
                        .unwrap_or(false);
                    let f32_view = (wq.as_f32(), wk.as_f32(), wv.as_f32(), wo.as_f32());
                    match (masked, f32_view) {
                        // Historical masked path (f32 slices; the loader
                        // keeps masked models in f32).
                        (true, (Some(q), Some(k), Some(v), Some(o))) => {
                            let active_heads = task_mask.unwrap().head_flags(li, self.num_heads);
                            attention::multi_head_attention(
                                &self.ws.n1,
                                q,
                                k,
                                v,
                                o,
                                &mut self.kv_cache.layers[li],
                                self.num_heads,
                                self.num_kv_heads,
                                self.head_dim,
                                self.hidden_size,
                                position,
                                &active_heads,
                                &self.inv_freq,
                            )
                        }
                        (masked, _) => {
                            if masked {
                                tracing::warn!(
                                    "layer {li}: head mask on quantized weights not \
                                     supported yet — executing dense"
                                );
                            }
                            let inv_freq_l = self.layer_inv_freq(li);
                            let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                            let cfg = QwenAttnCfg {
                                num_heads: self.layer_num_heads(li),
                                num_kv_heads: nkv_l,
                                head_dim: hd_l,
                                hidden_size: hs,
                                position,
                                inv_freq: &inv_freq_l,
                                rotary_dim: rd_l,
                                scale: self.attn_scale,
                                softcap: self.attn_softcap,
                                window: self.layer_window(li),
                                v_norm: self.attn_v_norm,
                                qk_norm_after_rope: self.qk_norm_after_rope,
                                q_norm: q_norm.as_deref(),
                                k_norm: k_norm.as_deref(),
                                output_gate: *output_gate,
                                softplus_gate: softplus_gate
                                    .as_ref()
                                    .map(|(gate, per_head)| (gate, *per_head)),
                                rope_scale: self.layer_rope_scale(li),
                                bias: bias
                                    .as_ref()
                                    .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                                rms_eps: eps,
                                norm_style: self.norm_style,
                                pool: pool.as_deref(),
                            };
                            attention::qwen_attention(
                                &self.ws.n1,
                                wq,
                                wk,
                                wv,
                                wo,
                                &mut self.kv_cache.layers[li],
                                &cfg,
                            )
                        }
                    }
                }
            };
            // Gemma sandwich norm: normalize the attention branch before
            // it joins the residual stream.
            let attn_out = match &self.weights.layers[self.phys_layer(li)].attn_out_norm {
                Some(w) => inference::rms_norm(&attn_out, w, self.rms_eps, self.norm_style),
                None => attn_out,
            };
            let lw = &self.weights.layers[self.phys_layer(li)];
            inference::add_rmsnorm_fused_into(
                &mut h,
                &attn_out,
                &lw.post_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.p1,
            );
            let mut attn_out = attn_out;
            attention::recycle_buf(&mut attn_out);
            let post_normed = &self.ws.p1;

            let ffn_masked = task_mask
                .map(|m| m.ffn_active_count(li) < self.intermediate_size)
                .unwrap_or(false);
            // One masked dense CONTRACT, dispatched by cost. The
            // activation-zeroing arm (the batched sweep's, validated
            // against the replica to 0.8%) computes the FULL fused FFN
            // and zeroes the dead — right whenever most neurons live.
            // The sparse arm reads ONLY active rows and down columns —
            // per-row dots are slower per element than the fused kernel,
            // so it pays only once the mask is deep enough. The 0.5
            // crossover is first-principles (fused kernels run ~2x the
            // per-row dot throughput); a shallow specialist (95% alive)
            // stays fused, a --target-sparsity bake flips arms on its
            // own weight.
            let ffn_out = match (ffn_masked, &lw.ffn) {
                // A defragged tube layer answers its own mask: the core
                // always runs, each tube runs when its bit is on, and
                // the tubes that are off are never read from the mmap.
                (_, FfnKind::Dense(d)) if !d.segs.is_empty() => {
                    let row = task_mask
                        .and_then(|tm| tm.ffn_masks.get(li))
                        .map(|v| v.as_slice());
                    tube_ffn(d, post_normed, 1, self.pool.as_deref(), row)
                }
                (true, FfnKind::Dense(d)) => {
                    let tm = task_mask.unwrap();
                    let alive = tm.ffn_active_count(li);
                    let deep = alive * 2 <= self.intermediate_size;
                    if deep && d.down_proj.sparse_col_ok() && !d.gate_proj.has_prism_contract() {
                        let active = tm.ffn_active_indices(li);
                        sparse_ffn_quant(
                            d,
                            post_normed,
                            &active,
                            self.hidden_size,
                            self.pool.as_deref(),
                        )
                    } else if deep
                        && let (Some(g), Some(u), Some(dn)) = (
                            d.gate_proj.as_f32(),
                            d.up_proj.as_f32(),
                            d.down_proj.as_f32(),
                        )
                    {
                        let active = tm.ffn_active_indices(li);
                        inference::sparse_ffn_forward(
                            post_normed,
                            g,
                            u,
                            dn,
                            self.hidden_size,
                            self.intermediate_size,
                            &active,
                            self.pool.as_deref(),
                        )
                    } else {
                        let row = tm.ffn_masks.get(li).map(|v| v.as_slice());
                        dense_ffn_batch(d, post_normed, 1, self.pool.as_deref(), row)
                    }
                }
                (true, FfnKind::Moe(m)) => {
                    // MoE is sparse by expert selection; a task mask
                    // narrows the ROUTABLE set via its expert fields
                    // (spec §5) when it carries them.
                    let allowed = task_mask.and_then(|tm| tm.expert_flags(li, m.experts.len()));
                    ffn_forward(
                        &lw.ffn,
                        post_normed,
                        self.pool.as_deref(),
                        allowed.as_deref(),
                    )
                }
                (true, FfnKind::DenseMoe(dm)) => dense_moe_ffn(
                    dm,
                    post_normed,
                    &h,
                    self.rms_eps,
                    self.norm_style,
                    self.pool.as_deref(),
                ),
                (false, _) => match &lw.ffn {
                    FfnKind::DenseMoe(dm) => dense_moe_ffn(
                        dm,
                        post_normed,
                        &h,
                        self.rms_eps,
                        self.norm_style,
                        self.pool.as_deref(),
                    ),
                    _ => {
                        let allowed = match (&lw.ffn, task_mask) {
                            (FfnKind::Moe(m), Some(tm)) => tm.expert_flags(li, m.experts.len()),
                            _ => None,
                        };
                        ffn_forward(
                            &lw.ffn,
                            post_normed,
                            self.pool.as_deref(),
                            allowed.as_deref(),
                        )
                    }
                },
            };
            let ffn_out = match &self.weights.layers[self.phys_layer(li)].ffn_out_norm {
                Some(w) => inference::rms_norm(&ffn_out, w, self.rms_eps, self.norm_style),
                None => ffn_out,
            };
            for (i, &f) in ffn_out.iter().enumerate() {
                h[i] += f;
            }
            let mut ffn_out = ffn_out;
            attention::recycle_buf(&mut ffn_out);

            // Gemma-4: the layer output is scaled by a learned scalar.
            if let Some(sc) = self.weights.layers[self.phys_layer(li)].layer_scale {
                for v in h.iter_mut() {
                    *v *= sc;
                }
            }

            // Looped Transformer: apply final norm at the end of each loop iteration.
            // Nanbeige 4.2: after layer 21 (virtual), apply norm before looping back to layer 0.
            if self.is_loop_end(li) && li + 1 < self.num_layers {
                h = inference::rms_norm(
                    &h,
                    &self.weights.final_norm,
                    self.rms_eps,
                    self.norm_style,
                );
            }

            // Dynamic routing φ capture (on-policy): the
            // EMA of the post-residual hidden at the router's phi_layer,
            // updated as the context evolves during decode.
            if self.dyn_phi_layer == Some(li) {
                self.update_dyn_phi(&h);
            }
        }
        crate::gpu::set_layer(-1); // layers done — lm_head outside layer-split
        if let Some(t) = t_race_cpu {
            crate::gpu::graph_race_record(false, t.elapsed());
        }

        h
    }

    /// EMA of φ at the router layer (rolling, weight 0.2 = ~5-token
    /// horizon). First observation seeds it exactly.
    fn update_dyn_phi(&mut self, h: &[f32]) {
        const A: f32 = 0.2;
        if self.dyn_phi_ema.len() != h.len() {
            self.dyn_phi_ema = vec![0.0; h.len()];
            self.dyn_phi_seen = 0;
        }
        if self.dyn_phi_seen == 0 {
            self.dyn_phi_ema.copy_from_slice(h);
        } else {
            for (e, &v) in self.dyn_phi_ema.iter_mut().zip(h) {
                *e = (1.0 - A) * *e + A * v;
            }
        }
        self.dyn_phi_seen += 1;
    }

    /// Current router φ (EMA at phi_layer); empty until first capture.
    pub fn dyn_phi(&self) -> &[f32] {
        &self.dyn_phi_ema
    }

    /// Enable/disable φ capture at the router layer, reset the EMA.
    pub fn set_dyn_phi_layer(&mut self, layer: Option<usize>) {
        self.dyn_phi_layer = layer;
        self.dyn_phi_ema.clear();
        self.dyn_phi_seen = 0;
    }

    /// Skills eligible for dynamic switching: (index, id, phi_layer).
    pub fn dynamic_skills(&self) -> Vec<(usize, String, usize)> {
        let Some(model) = &self.model else {
            return Vec::new();
        };
        model
            .header
            .skills
            .iter()
            .enumerate()
            .filter_map(|(i, sk)| {
                let ok = matches!(self.dyn_skill_layers.get(i), Some(Some(_)));
                let sel = sk.selection.as_ref()?;
                (ok).then(|| (i, sk.id.clone(), sel.phi_layer))
            })
            .collect()
    }

    /// Index of the currently overlaid skill (None = backbone).
    pub fn active_skill(&self) -> Option<usize> {
        self.dyn_active
    }

    /// Enable dynamic per-token skill routing: build the hysteresis
    /// router from the container's routable skills, start φ capture at
    /// their (shared) phi_layer. Returns the number of routable skills
    /// (0 = nothing to route; router stays off). Idempotent.
    pub fn enable_dynamic_routing(&mut self) -> usize {
        use crate::swarm::{DynRouter, RoutableSkill};
        let Some(model) = self.model.clone() else {
            return 0;
        };
        // A blend materialized f32 working tensors into the layers; there
        // is no single skill index to revert from → refuse (honest).
        if self.dyn_blend_loaded {
            tracing::warn!("dynamic routing unavailable on a blend-loaded pipeline");
            return 0;
        }
        // A statically-overlaid skill that is NOT FFN-eligible can't be
        // cheaply reverted at generation start → refuse rather than
        // silently keep it overlaid.
        if let Some(a) = self.dyn_active {
            if !matches!(self.dyn_skill_layers.get(a), Some(Some(_))) {
                tracing::warn!("loaded skill is not FFN-eligible — dynamic routing unavailable");
                return 0;
            }
        }
        let hidden = self.hidden_size;
        let mut skills = Vec::new();
        for (idx, id, _phi) in self.dynamic_skills() {
            if let Some(sel) = model.header.skills[idx].selection.as_ref() {
                if let Some(rs) = RoutableSkill::from_descriptor(idx, id, sel, hidden) {
                    skills.push(rs);
                }
            }
        }
        if skills.is_empty() {
            return 0;
        }
        // Skills should share a phi_layer; warn (not fail) if they don't.
        let phi = skills[0].phi_layer;
        if skills.iter().any(|s| s.phi_layer != phi) {
            tracing::warn!("routable skills disagree on phi_layer; using {phi}");
        }
        let n = skills.len();
        self.set_dyn_phi_layer(Some(phi));
        self.dyn_router = Some(DynRouter::new(skills));
        n
    }

    /// Human-readable switch log from the last dynamic-routed generation.
    pub fn route_switches(&self) -> Vec<(usize, Option<String>, Option<String>)> {
        self.dyn_router
            .as_ref()
            .map(|r| r.switches.clone())
            .unwrap_or_default()
    }

    /// LM head: hidden → logits [vocab_size]. The dominant matvec of
    /// every decode step — row-parallel on the worker pool.
    fn lm_head_forward(&self, hidden: &[f32]) -> Vec<f32> {
        let rows = self.weights.lm_head.rows();
        let mut logits = attention::take_buf(rows.min(self.vocab_size));
        self.weights
            .lm_head
            .matvec(hidden, &mut logits, self.pool.as_deref());
        logits.resize(self.vocab_size, 0.0);
        if let Some(m) = self.logit_multiplier {
            for l in logits.iter_mut() {
                *l *= m;
            }
        }
        if let Some(c) = self.final_softcap {
            for l in logits.iter_mut() {
                *l = c * (*l / c).tanh();
            }
        }
        if let Some(cm) = self.head_clusters.as_ref() {
            self.hierarchical_head_logprobs(hidden, cm, &mut logits);
        }
        logits
    }

    /// Two-level head (Cortiq Embryo): in place, logits[v] ← log p(v) =
    /// (lc[c] − lse(lc)) + (logit[v] − lse over v's cluster block), c = v / S.
    fn hierarchical_head_logprobs(&self, hidden: &[f32], cm: &[f32], logits: &mut [f32]) {
        let h = hidden.len();
        let ncl = cm.len() / h.max(1);
        if ncl == 0 || logits.len() % ncl != 0 {
            return;
        }
        let cs = logits.len() / ncl;
        // cluster logits + log-softmax
        let mut lc = vec![0.0f32; ncl];
        for c in 0..ncl {
            let row = &cm[c * h..(c + 1) * h];
            let mut s = 0.0f32;
            for j in 0..h {
                s += row[j] * hidden[j];
            }
            lc[c] = s;
        }
        let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = mx + lc.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
        for c in 0..ncl {
            let blk = &mut logits[c * cs..(c + 1) * cs];
            let bm = blk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bl: f32 = bm + blk.iter().map(|v| (v - bm).exp()).sum::<f32>().ln();
            let add = lc[c] - lse - bl;
            for v in blk.iter_mut() {
                *v += add;
            }
        }
    }

    /// Prefill `ids` and return the next-token logits — what the model
    /// would predict next, WITHOUT committing to generation (introspection
    /// for `cortiq explain`). Clears and repopulates the KV cache; leaves
    /// the active overlay untouched.
    pub fn prefill_next_logits(&mut self, ids: &[u32], task_mask: Option<&TaskMask>) -> Vec<f32> {
        self.clear_sequence_state();
        // This helper is used by the pooled classification endpoint, where
        // every request is a fresh sequence. The shared reset also clears the
        // wgpu token graph's device-side recurrent state.
        crate::gpu::graph_race_begin_generation();
        if task_mask.is_none() {
            self.o1_begin();
        }
        let mut hidden = vec![0.0f32; self.hidden_size];
        for (pos, &id) in ids.iter().enumerate() {
            let emb = self.embed_single(id);
            hidden = self.forward_layers(&emb, pos, task_mask);
        }
        if let Err(err) = self.o1_seal_checked() {
            self.o1_fail(err);
        }
        inference::rms_norm_into(
            &hidden,
            &self.weights.final_norm,
            self.rms_eps,
            self.norm_style,
            &mut self.ws.n1,
        );
        self.lm_head_forward(&self.ws.n1)
    }
}

/// Convenience: deterministic tiny pipeline for tests.
pub fn create_test_pipeline(
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
) -> Pipeline {
    // Small pseudo-random weights: constant weights make attention
    // degenerate and hide indexing bugs.
    let synth = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 31 + salt * 17 + 7) % 97) as f32 / 97.0 - 0.5) * 0.2)
            .collect()
    };
    let qt = |rows: usize, cols: usize, salt: usize| -> QTensor {
        QTensor::from_f32(synth(rows * cols, salt), rows, cols)
    };
    let layer_weights: Vec<LayerWeights> = (0..num_layers)
        .map(|li| LayerWeights {
            input_norm: vec![1.0; hidden_size],
            post_norm: vec![1.0; hidden_size],
            attn_out_norm: None,
            ffn_out_norm: None,
            layer_scale: None,
            ffn: FfnKind::Dense(DenseFfn {
                gate_proj: qt(intermediate_size, hidden_size, li * 10 + 5),
                up_proj: qt(intermediate_size, hidden_size, li * 10 + 6),
                down_proj: qt(hidden_size, intermediate_size, li * 10 + 7),
                act: Act::Silu,
                down_t: None,
                segs: Vec::new(),
            }),
            attn: AttnKind::Full {
                bias: None,
                wq: qt(num_heads * head_dim, hidden_size, li * 10 + 1),
                wk: qt(num_kv_heads * head_dim, hidden_size, li * 10 + 2),
                wv: qt(num_kv_heads * head_dim, hidden_size, li * 10 + 3),
                wo: qt(hidden_size, num_heads * head_dim, li * 10 + 4),
                q_norm: None,
                k_norm: None,
                output_gate: false,
                softplus_gate: None,
            },
        })
        .collect();

    Pipeline::new(
        Tokenizer::byte_level(),
        PipelineWeights {
            embed_tokens: qt(vocab_size, hidden_size, 100),
            layers: layer_weights,
            lm_head: qt(vocab_size, hidden_size, 200),
            final_norm: vec![1.0; hidden_size],
        },
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        num_layers, // physical_layers = num_layers (non-looped)
        false,      // loop_final_norm
        vocab_size,
        1e-6,
        10_000.0,
        NormStyle::Qwen,
        4096,
        SamplerConfig {
            seed: Some(42),
            ..Default::default()
        },
    )
}

/// Batched dense-FFN: gate/up/down via matmat (element-wise the same
/// math as b × dense_ffn — the same dot kernels).
/// One mask bit, LSB-first per byte — `TaskMask::ffn_active_indices`'s
/// convention.
#[inline]
fn mask_bit(row: &[u8], j: usize) -> bool {
    (row.get(j >> 3).copied().unwrap_or(0) >> (j & 7)) & 1 != 0
}

/// Zero the CLOSED neurons' activations in a [rows × inter] panel — the
/// masked-inference fast path's whole trick: full fused quant compute,
/// then the mask lands on the ACTIVATIONS, which is arithmetically the
/// pruned network without touching a quantized weight byte. Whole open
/// bytes (0xFF = 8 open neurons) skip in one test.
/// `CMF_FFN_MASK_GAIN` — Patent 12 FIG. 4, variance-preserving
/// rescaling: truncation removes a share of the layer's output energy,
/// so the survivors are scaled up to put the variance back where the
/// downstream norm expects it. A scalar here; per layer it is
/// `sqrt(total energy / kept energy)`.
fn mask_gain() -> f32 {
    static G: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("CMF_FFN_MASK_GAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0)
    })
}

fn zero_masked_cols(g: &mut [f32], rows: usize, inter: usize, row: &[u8]) {
    // With CMF_FFN_MEANFILL a closed neuron contributes its average
    // instead of nothing — same bytes read, one constant restored.
    let fill = meanfill().and_then(|(i, v)| {
        let li = crate::gpu::cur_layer();
        (*i == inter && li >= 0).then(|| &v[li as usize * inter..(li as usize + 1) * inter])
    });
    for r in 0..rows {
        let base = r * inter;
        for (bi, &byte) in row.iter().enumerate() {
            if byte == 0xFF {
                continue;
            }
            let j0 = bi * 8;
            for bit in 0..8 {
                let j = j0 + bit;
                if j < inter && byte & (1 << bit) == 0 {
                    g[base + j] = fill.map_or(0.0, |f| f[j]);
                }
            }
        }
    }
    let gain = mask_gain();
    if gain != 1.0 {
        for v in g[..rows * inter].iter_mut() {
            *v *= gain;
        }
    }
}

/// True when neuron `i`'s bit is set (no mask = everything runs).
#[inline]
fn tube_bit(row: Option<&[u8]>, i: usize) -> bool {
    row.is_none_or(|r| mask_bit(r, i))
}

/// Every bit below `n` set — the common case for a tube file's CORE,
/// where only the tube bits vary per task.
fn all_bits_on(row: &[u8], n: usize) -> bool {
    (0..n).all(|i| mask_bit(row, i))
}

/// `CMF_TUBE_TOPK` — how many tubes a TOKEN may open (0 = the task mask
/// decides alone). This is the dense FFN read as a mixture: the tubes
/// are the experts a k-means over `gate_proj` rows found, and the token
/// picks among them. `CMF_TUBE_SCORE=gate` scores a tube by its own
/// gate (realizable: only `up`/`down` of the losers go unread),
/// `=oracle` scores by the true `silu(gate)·up` mass (the ceiling —
/// only `down` is saved, and the selection has read what it predicts).
fn tube_topk() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CMF_TUBE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn tube_score_oracle() -> bool {
    static O: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *O.get_or_init(|| std::env::var("CMF_TUBE_SCORE").is_ok_and(|v| v == "oracle"))
}

/// The routed arm of `tube_ffn`: a token opens only its best `k` tubes.
/// At `b == 1` (decode) the losers are genuinely never read — that is
/// the speed. At `b > 1` (the scoring sweep) every tube is computed and
/// the losers' activations are zeroed instead: same arithmetic, so the
/// perplexity is the routed model's, measured without a per-token
/// gather in the middle of a GEMM.
fn tube_ffn_routed(
    d: &DenseFfn,
    xs: &[f32],
    b: usize,
    pool: Option<&Pool>,
    mask_row: Option<&[u8]>,
    k: usize,
) -> Vec<f32> {
    let hidden = d.down_proj.rows();
    let core = d.gate_proj.rows();
    let core_full = mask_row.is_none_or(|r| all_bits_on(r, core));
    let mut out = match (b, core_full, mask_row) {
        (1, true, _) => dense_ffn(d, xs, pool),
        (1, false, Some(row)) => dense_ffn_masked(d, xs, pool, row),
        (_, true, _) => dense_ffn_batch(d, xs, b, pool, None),
        (_, false, row) => dense_ffn_batch(d, xs, b, pool, row),
    };
    let cand: Vec<usize> = (0..d.segs.len())
        .filter(|&i| tube_bit(mask_row, d.segs[i].start))
        .collect();
    if cand.is_empty() {
        return out;
    }
    // gate (and, where the score or the batch needs it, up) per tube.
    // The SCORE is taken at the point the serving path could take it:
    // off the gate alone, or off the finished activation for the oracle.
    let oracle = tube_score_oracle();
    let mut acts: Vec<Vec<f32>> = Vec::with_capacity(cand.len());
    let mut scores = vec![0f32; b * cand.len()];
    for (ci, &i) in cand.iter().enumerate() {
        let seg = &d.segs[i];
        let w = seg.width;
        let mut g = vec![0.0f32; b * w];
        if b == 1 {
            seg.gate.matvec(xs, &mut g, pool);
        } else {
            seg.gate.matmat(xs, b, &mut g, pool);
        }
        for v in g.iter_mut() {
            *v = Act::Silu.combine(*v, 1.0);
        }
        if !oracle {
            for t in 0..b {
                scores[t * cand.len() + ci] =
                    g[t * w..(t + 1) * w].iter().map(|v| v * v).sum::<f32>();
            }
        }
        if oracle || b > 1 {
            let mut u = vec![0.0f32; b * w];
            if b == 1 {
                seg.up.matvec(xs, &mut u, pool);
            } else {
                seg.up.matmat(xs, b, &mut u, pool);
            }
            for (a, &v) in g.iter_mut().zip(u.iter()) {
                *a *= v;
            }
            if oracle {
                for t in 0..b {
                    scores[t * cand.len() + ci] =
                        g[t * w..(t + 1) * w].iter().map(|v| v * v).sum::<f32>();
                }
            }
        }
        acts.push(g);
    }
    // per-token scores and the winners
    let keep = k.min(cand.len());
    let mut scratch: Vec<f32> = Vec::new();
    for t in 0..b {
        let mut sc: Vec<(f32, usize)> = (0..cand.len())
            .map(|ci| (scores[t * cand.len() + ci], ci))
            .collect();
        sc.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
        let mut alive = vec![false; cand.len()];
        for &(_, ci) in sc.iter().take(keep) {
            alive[ci] = true;
        }
        if b > 1 {
            for (ci, a) in acts.iter_mut().enumerate() {
                if !alive[ci] {
                    let w = d.segs[cand[ci]].width;
                    a[t * w..(t + 1) * w].fill(0.0);
                }
            }
        } else {
            // decode: finish only the winners — the losers' up/down
            // (and, with the gate score, everything but their gate)
            // are never touched.
            for (ci, &i) in cand.iter().enumerate() {
                if !alive[ci] {
                    continue;
                }
                let seg = &d.segs[i];
                let w = seg.width;
                let g = &mut acts[ci];
                if !tube_score_oracle() {
                    scratch.clear();
                    scratch.resize(w, 0.0);
                    seg.up.matvec(xs, &mut scratch, pool);
                    for (a, &v) in g.iter_mut().zip(scratch.iter()) {
                        *a *= v;
                    }
                }
                let mut acc = vec![0.0f32; hidden];
                seg.down.matvec(g, &mut acc, pool);
                for (o, a) in out.iter_mut().zip(&acc) {
                    *o += *a;
                }
            }
        }
    }
    if b > 1 {
        for (ci, &i) in cand.iter().enumerate() {
            let seg = &d.segs[i];
            let mut acc = vec![0.0f32; b * hidden];
            seg.down.matmat(&acts[ci], b, &mut acc, pool);
            for (o, a) in out.iter_mut().zip(&acc) {
                *o += *a;
            }
        }
    }
    out
}

/// FFN of a defragged tube layer: the always-on core plus the tubes the
/// task mask switches on. Each tube is a normal tensor triple, so the
/// same kernels run it and an inactive tube's bytes are never read —
/// that is the whole point of the defrag (a scattered mask cannot skip
/// bytes; a contiguous one is just a smaller matrix).
fn tube_ffn(
    d: &DenseFfn,
    xs: &[f32],
    b: usize,
    pool: Option<&Pool>,
    mask_row: Option<&[u8]>,
) -> Vec<f32> {
    if tube_topk() > 0 {
        return tube_ffn_routed(d, xs, b, pool, mask_row, tube_topk());
    }
    let hidden = d.down_proj.rows();
    let core = d.gate_proj.rows();
    let core_full = mask_row.is_none_or(|r| all_bits_on(r, core));
    let mut out = match (b, core_full, mask_row) {
        (1, true, _) => dense_ffn(d, xs, pool),
        (1, false, Some(row)) => dense_ffn_masked(d, xs, pool, row),
        (_, true, _) => dense_ffn_batch(d, xs, b, pool, None),
        (_, false, row) => dense_ffn_batch(d, xs, b, pool, row),
    };
    TUBE_SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let [g, u, acc] = &mut *sc;
        for seg in &d.segs {
            if !tube_bit(mask_row, seg.start) {
                continue;
            }
            let w = seg.width;
            g.resize(b * w, 0.0);
            if b == 1
                && d.act == Act::Silu
                && QTensor::matvec_silu_mul(&seg.gate, &seg.up, xs, g, pool)
            {
                // g holds silu(gate)·up.
            } else {
                u.resize(b * w, 0.0);
                if b == 1 {
                    QTensor::matvec_many([&seg.gate, &seg.up], xs, [g, u], pool);
                } else {
                    seg.gate.matmat(xs, b, g, pool);
                    seg.up.matmat(xs, b, u, pool);
                }
                for i in 0..b * w {
                    g[i] = d.act.combine(g[i], u[i]);
                }
            }
            acc.resize(b * hidden, 0.0);
            acc.fill(0.0);
            if b == 1 {
                seg.down.matvec(g, acc, pool);
            } else {
                seg.down.matmat(g, b, acc, pool);
            }
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o += *a;
            }
        }
        out
    })
}

thread_local! {
    /// gate / up / down-accumulator scratch for the tube loop — a tube
    /// runs once per layer per token, and a fresh Vec each time is a
    /// malloc per tube per layer per token.
    static TUBE_SCRATCH: std::cell::RefCell<[Vec<f32>; 3]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new()]) };
}

fn dense_ffn_batch(
    d: &DenseFfn,
    xs: &[f32],
    b: usize,
    pool: Option<&Pool>,
    mask_row: Option<&[u8]>,
) -> Vec<f32> {
    let inter = d.gate_proj.rows();
    let hidden = d.down_proj.rows();
    // Fused on-device SwiGLU when the device is in play: three separate
    // `matmat` calls are three round trips per layer, and the gate/up
    // panels (b × inter — 22 MB each at a 512-token chunk) cross the bus
    // twice for nothing. The kernel already existed for the image DiT;
    // the LLM prefill was simply never wired to it. A task mask needs the
    // activations on the host between the halves, so it keeps the CPU
    // arm below.
    if mask_row.is_none()
        && d.act == Act::Silu
        && b >= 32
        && crate::gpu::enabled_here()
        && !crate::gpu::mm_killed()
        // The refit pass needs this layer's activations on the host; the
        // fused chain keeps them on the device. Refusing it here costs
        // one round trip and keeps every GEMM on the card — the
        // alternative was running the whole calibration on the CPU.
        && refit_dir().is_none()
        // Same for the mass/hit probes. The accumulator at the bottom of
        // this function only sees `g` when `g` came back to the host, so
        // a fused batch would leave it summing nothing — a probe that
        // reports zeros rather than failing, which is worse.
        && !ffn_probe_active()
    {
        if let (Some((model, w1)), Some((_, w3)), Some((_, w2))) = (
            d.gate_proj.mapped_q4t(),
            d.up_proj.mapped_q4t(),
            d.down_proj.mapped_q4t(),
        ) {
            let mut out = vec![0.0f32; b * hidden];
            if crate::gpu::q4t_ffn(model, w1, w3, w2, xs, b, hidden, inter, &mut out) {
                return out;
            }
        }
        // The q4tp twin (same kernel family, scale from the row ladder) —
        // the DiT has run it in production since the pipeline containers;
        // the LLM prefill was simply never wired to it, so a q4tp model's
        // prefill panels stayed on the CPU.
        if let (Some((model, w1)), Some((_, w3)), Some((_, w2))) = (
            d.gate_proj.mapped_q4tp(),
            d.up_proj.mapped_q4tp(),
            d.down_proj.mapped_q4tp(),
        ) {
            let mut out = vec![0.0f32; b * hidden];
            if crate::gpu::q4tp_ffn(model, w1, w3, w2, xs, b, hidden, inter, &mut out) {
                return out;
            }
        }
    }
    let mut g = vec![0.0f32; b * inter];
    d.gate_proj.matmat(xs, b, &mut g, pool);
    let mut u = vec![0.0f32; b * inter];
    d.up_proj.matmat(xs, b, &mut u, pool);
    if gate_topk() > 0 && d.act == Act::Silu {
        for t in 0..b {
            let row = &mut g[t * inter..(t + 1) * inter];
            for v in row.iter_mut() {
                *v = Act::Silu.combine(*v, 1.0);
            }
            keep_top_k(row, gate_topk());
        }
        for i in 0..b * inter {
            g[i] *= u[i];
        }
    } else {
        for i in 0..b * inter {
            g[i] = d.act.combine(g[i], u[i]);
        }
    }
    if let Some(row) = mask_row {
        zero_masked_cols(&mut g, b, inter, row);
    }
    if oracle_topk() > 0 {
        for t in 0..b {
            keep_top_k(&mut g[t * inter..(t + 1) * inter], oracle_topk());
        }
    }
    let mut out = vec![0.0f32; b * hidden];
    d.down_proj.matmat(&g, b, &mut out, pool);
    if refit_dir().is_some() {
        let li = crate::gpu::cur_layer();
        if li >= 0 {
            refit_accumulate(li as usize, &g, b, inter, &out, hidden, pool);
        }
    }
    // The DTG-MA probe, on the batched path: one prefill sweep gives the
    // same per-neuron statistic the per-position probe does, and on a 27B
    // that is minutes instead of hours.
    FFN_PROBE.with(|pr| {
        if let Some(acc) = pr.borrow_mut().as_mut() {
            let li = crate::gpu::cur_layer();
            if li < 0 {
                return;
            }
            let Some(row) = acc.get_mut(li as usize) else {
                return;
            };
            let sq = probe_sq();
            for t in 0..b {
                for (a, &v) in row.iter_mut().zip(&g[t * inter..(t + 1) * inter]) {
                    *a += if sq {
                        (v as f64) * (v as f64)
                    } else {
                        (v as f64).abs()
                    };
                }
            }
        }
    });
    out
}

/// Batched MoE-FFN: router batched, positions are GROUPED by expert —
/// an expert's weights are read once for all its positions in the chunk
/// (the main prefill-GEMM win on MoE: 960MB/token of 35B experts).
/// Accumulate per-channel activation energy for `CMF_RMS_TRACE`.
fn accumulate_act(m: &MoeFfn, xs: &[f32], b: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("CMF_RMS_TRACE").is_ok());
    let dump = *DUMP.get_or_init(|| std::env::var("CMF_ACT_DUMP").is_ok());
    if (!on && !dump) || b == 0 {
        return;
    }
    let hidden = xs.len() / b;
    if on {
        let mut acc = m.act_sq.borrow_mut();
        if acc.len() < hidden {
            acc.resize(hidden, 0.0);
        }
        for t in 0..b {
            let row = &xs[t * hidden..(t + 1) * hidden];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += (v as f64) * (v as f64);
            }
        }
    }
    if dump {
        // Cap the capture: the covariance needs a few thousand rows, and a
        // whole prefill of every layer would be gigabytes for no extra rank.
        let cap: usize = std::env::var("CMF_ACT_DUMP_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let mut rows = m.act_rows.borrow_mut();
        if rows.len() < cap * hidden {
            let take = b.min((cap * hidden - rows.len()) / hidden.max(1));
            rows.extend_from_slice(&xs[..take * hidden]);
        }
    }
}

/// Send-able cursor over a Vec-of-Vecs: each pool worker writes only its
/// own slots (disjoint by construction in the caller).
#[derive(Clone, Copy)]
struct SendVecs(*mut Vec<f32>);
unsafe impl Send for SendVecs {}
unsafe impl Sync for SendVecs {}
impl SendVecs {
    #[inline]
    fn at(self, i: usize) -> *mut Vec<f32> {
        unsafe { self.0.add(i) }
    }
}

fn moe_ffn_batch(
    m: &MoeFfn,
    xs: &[f32],
    b: usize,
    hidden: usize,
    pool: Option<&Pool>,
    allowed: Option<&[bool]>,
) -> Vec<f32> {
    accumulate_act(m, xs, b);
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; b * ne];
    match &m.resonance {
        Some(r) => {
            let hdim = xs.len() / b.max(1);
            for bi in 0..b {
                r.scores(
                    &xs[bi * hdim..(bi + 1) * hdim],
                    &mut logits[bi * ne..(bi + 1) * ne],
                );
            }
        }
        None => m.router.matmat(xs, b, &mut logits, pool),
    }

    // Assignments: expert → [(position, weight)] — same routing as
    // moe_ffn, per position (see `moe_route`).
    let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
    {
        let mut st = m.stats.borrow_mut();
        if st.len() < ne {
            st.resize(ne, 0);
        }
        for bi in 0..b {
            let (idx, p, wsum) = moe_route(&logits[bi * ne..(bi + 1) * ne], m, allowed);
            for &e in &idx {
                st[e] += 1;
                assign[e].push((bi, p[e] / wsum));
            }
        }
    }

    let mut out = vec![0.0f32; b * hidden];
    let cols = m.experts[0].gate_proj.cols();
    let run_expert = |d: &DenseFfn, list: &[(usize, f32)], out: &mut [f32]| {
        let sb = list.len();
        let mut sub = vec![0.0f32; sb * cols];
        for (k, &(bi, _)) in list.iter().enumerate() {
            sub[k * cols..(k + 1) * cols].copy_from_slice(&xs[bi * cols..(bi + 1) * cols]);
        }
        let eo = dense_ffn_batch(d, &sub, sb, pool, None);
        for (k, &(bi, w)) in list.iter().enumerate() {
            for i in 0..hidden {
                out[bi * hidden + i] += w * eo[k * hidden + i];
            }
        }
    };
    // Routed experts: the panels are TINY (b·top_k spread over every
    // expert — a few positions each), so a pool dispatch per expert is
    // pure barrier cost. Invert the parallelism: workers take WHOLE
    // experts (serial math inside), then one deterministic scatter in
    // expert order — the exact accumulation order the serial loop had.
    let active: Vec<usize> = (0..ne).filter(|&e| !assign[e].is_empty()).collect();
    if pool.is_some() && active.len() >= 8 {
        let mut panels: Vec<Vec<f32>> = vec![Vec::new(); active.len()];
        {
            let panel_ptr = SendVecs(panels.as_mut_ptr());
            // Capture only the expert table: `m` itself carries RefCell
            // stats and must not cross the pool boundary.
            let experts = &m.experts;
            let (active_r, assign_r) = (&active, &assign);
            let run = |start: usize, end: usize| {
                for ai in start..end {
                    let e = active_r[ai];
                    let list = &assign_r[e];
                    let sb = list.len();
                    let mut sub = vec![0.0f32; sb * cols];
                    for (k, &(bi, _)) in list.iter().enumerate() {
                        sub[k * cols..(k + 1) * cols]
                            .copy_from_slice(&xs[bi * cols..(bi + 1) * cols]);
                    }
                    // SAFETY: each worker owns a disjoint panels[ai].
                    unsafe {
                        *panel_ptr.at(ai) = dense_ffn_batch(&experts[e], &sub, sb, None, None);
                    }
                }
            };
            match pool {
                Some(p) => p.run_rows(active.len(), &run),
                None => run(0, active.len()),
            }
        }
        for (ai, &e) in active.iter().enumerate() {
            for (k, &(bi, w)) in assign[e].iter().enumerate() {
                let eo = &panels[ai][k * hidden..(k + 1) * hidden];
                for i in 0..hidden {
                    out[bi * hidden + i] += w * eo[i];
                }
            }
        }
    } else {
        for &e in &active {
            run_expert(&m.experts[e], &assign[e], &mut out);
        }
    }
    if let Some((se, gate)) = &m.shared {
        let all: Vec<(usize, f32)> = if let Some(gate) = gate {
            let mut gl = vec![0.0f32; b];
            gate.matmat(xs, b, &mut gl, pool);
            (0..b)
                .map(|bi| (bi, 1.0 / (1.0 + (-gl[bi]).exp())))
                .collect()
        } else {
            (0..b).map(|bi| (bi, 1.0)).collect()
        };
        run_expert(se, &all, &mut out);
    }
    out
}

thread_local! {
    /// gate/up activation scratch for the dense FFN paths (single uses
    /// two slots, the fused pair all four) — these were fresh
    /// intermediate-size Vecs on every layer of every token.
    static FFN_SCRATCH: std::cell::RefCell<[Vec<f32>; 4]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new(), Vec::new()]) };
}

/// Dense SwiGLU FFN through QTensor matvecs (any storage).
fn dense_ffn(d: &DenseFfn, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    // Per-token sparsity, when the file was built for it: gate first,
    // then only the chosen neurons' up/down rows leave the mmap.
    if gate_topk() > 0
        && let Some(out) = dense_ffn_dynamic(d, x, pool, gate_topk())
    {
        return out;
    }
    // Whole-FFN GPU submit (этап 4.2 increment): gate → silu·up → down
    // chained in ONE command buffer with the intermediate activations
    // resident on the device — 3 per-op polls become 1 per layer. The
    // moe_block backend already implements exactly this chain; a dense
    // FFN is one expert with weight 1. Runtime probe: the chain still
    // pays one submit+poll per layer — alternate it against the pure-CPU
    // FFN and keep whichever is faster on this machine.
    // q1 FFNs offload at any practical size: the q1 CPU kernel is
    // compute-bound, so the UMA threshold logic does not apply — the
    // probe measures and decides either way.
    // The fused GPU block has no descriptor-aware Prism path: it would either
    // consume an unrotated activation or decline after inspecting the mixed
    // q2tp/q4tp tensors.  Do not let that structural refusal enter the FFN
    // probe's CPU_ONLY scope; the ordinary body below dispatches each matrix
    // through QTensor::matvec, which owns the signed FWHT + affine q2tp route.
    let prism_body = d.gate_proj.has_prism_contract()
        || d.up_proj.has_prism_contract()
        || d.down_proj.has_prism_contract();
    if !prism_body
        && crate::gpu::enabled_here()
        && (d.gate_proj.rows() >= crate::gpu::min_rows() || d.gate_proj.is_q1())
    {
        let arm = if d.gate_proj.is_q1() && crate::gpu::q1_force() {
            crate::gpu::ProbeArm::Gpu
        } else {
            crate::gpu::probe_arm(crate::gpu::OpClass::Ffn)
        };
        match arm {
            crate::gpu::ProbeArm::Gpu => {
                let t0 = std::time::Instant::now();
                if let Some(out) = dense_ffn_gpu(d, x, pool) {
                    crate::gpu::probe_record(crate::gpu::OpClass::Ffn, true, t0.elapsed());
                    return out;
                }
                // Declined: no timing exists, so say so. Silence here is
                // what left `ffn` undecided for 9000 calls and cost a
                // failed device attempt on half of them.
                crate::gpu::probe_note_decline(crate::gpu::OpClass::Ffn);
            }
            crate::gpu::ProbeArm::CpuTimed => {
                let t0 = std::time::Instant::now();
                let out = crate::gpu::cpu_scope(|| dense_ffn_cpu(d, x, pool));
                crate::gpu::probe_record(crate::gpu::OpClass::Ffn, false, t0.elapsed());
                return out;
            }
            crate::gpu::ProbeArm::Cpu => {
                return crate::gpu::cpu_scope(|| dense_ffn_cpu(d, x, pool));
            }
        }
    }
    dense_ffn_cpu(d, x, pool)
}

/// The pure-CPU dense-FFN body (also the fallback of every GPU refusal).
fn dense_ffn_cpu(d: &DenseFfn, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let inter = d.gate_proj.rows();
    FFN_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let [g, u, ..] = &mut *s;
        g.resize(inter, 0.0);
        // Fused gate+up+silu: one dispatch, no separate silu pass.
        // Falls back to matvec_many + silu loop for unsupported dtypes.
        if gate_topk() > 0 {
            // Gate first, select, and only then pay for `up`: the
            // measurement arm computes both and zeroes the losers, which
            // is the same arithmetic.
            u.resize(inter, 0.0);
            QTensor::matvec_many([&d.gate_proj, &d.up_proj], x, [g, u], pool);
            for i in 0..inter {
                g[i] = Act::Silu.combine(g[i], 1.0);
            }
            keep_top_k(g, gate_topk());
            for i in 0..inter {
                g[i] *= u[i];
            }
        } else if d.act == Act::Silu
            && QTensor::matvec_silu_mul(&d.gate_proj, &d.up_proj, x, g, pool)
        {
            // g now holds silu(gate)·up directly.
        } else {
            u.resize(inter, 0.0);
            // Multi-matrix job: gate+up under one pool dispatch.
            QTensor::matvec_many([&d.gate_proj, &d.up_proj], x, [g, u], pool);
            for i in 0..inter {
                g[i] = d.act.combine(g[i], u[i]);
            }
        }
        // DTG-MA bake probe (Patent 2): accumulate this layer's
        // per-neuron activation mass while a probe pass is active.
        // `CMF_FFN_PROBE_TOPK=k` switches the statistic from mass to a
        // HIT COUNT — how many tokens rank the neuron in their own top
        // k. Mass asks "how loud is this neuron overall", the count
        // asks "how often does this task actually need it", and the two
        // rank neurons differently whenever a few tokens are loud.
        FFN_PROBE.with(|pr| {
            if let Some(acc) = pr.borrow_mut().as_mut() {
                let li = crate::gpu::cur_layer();
                if li >= 0 {
                    if let Some(row) = acc.get_mut(li as usize) {
                        match probe_topk() {
                            0 if probe_sq() => {
                                for (a, &v) in row.iter_mut().zip(g.iter()) {
                                    *a += (v as f64) * (v as f64);
                                }
                            }
                            0 if probe_signed() => {
                                for (a, &v) in row.iter_mut().zip(g.iter()) {
                                    *a += v as f64;
                                }
                            }
                            0 => {
                                for (a, &v) in row.iter_mut().zip(g.iter()) {
                                    *a += (v as f64).abs();
                                }
                            }
                            k => {
                                let n = g.len();
                                let k = k.min(n);
                                let mut mag: Vec<f32> = g.iter().map(|v| v.abs()).collect();
                                let (_, kth, _) = mag.select_nth_unstable_by(k - 1, |a, b| {
                                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                let thr = *kth;
                                for (a, &v) in row.iter_mut().zip(g.iter()) {
                                    if v.abs() >= thr {
                                        *a += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        if oracle_topk() > 0 {
            keep_top_k(g, oracle_topk());
        }
        {
            let li = crate::gpu::cur_layer();
            if li >= 0 {
                adump_row(li as usize, g);
            }
        }
        let mut out = attention::take_buf(d.down_proj.rows());
        d.down_proj.matvec(g, &mut out, pool);
        out
    })
}

/// Online accumulators for the AWNP refit of a narrowed FFN.
///
/// The refit needs `Gss = A_SᵀA_S` and `YA = YᵀA_S` per layer, where `A_S`
/// are the calibration activations of the KEPT neurons and `Y` the full
/// FFN output. Both are small enough to hold; the thing that is not is
/// the activations they are built from — a 27B layer would dump a
/// gigabyte per thousand tokens. So they are accumulated as the
/// calibration runs and written once at the end.
///
/// `CMF_FFN_REFIT=<dir>` holds `support.<L>.u32` (a u32 count then the
/// kept indices) for every layer to accumulate; `CMF_FFN_REFIT_FROM/TO`
/// bound the layer span so the accumulators fit in RAM.
pub struct RefitAcc {
    pub support: Vec<u32>,
    pub gss: Vec<f32>,
    pub ya: Vec<f32>,
    pub hidden: usize,
    pub tokens: u64,
    /// Activations staged transposed ([ns, t] and [hidden, t]) until the
    /// batch is worth a GEMM. The product costs `ns²` to move and add
    /// REGARDLESS of how many tokens went into it, so folding 16 chunks
    /// into one call cuts that cost 16× — it was 15 TB of traffic per
    /// calibration pass at one call per 256 tokens.
    pub buf_g: Vec<f32>,
    pub buf_o: Vec<f32>,
    pub buf_t: usize,
}

/// The product buffer is SHARED across layers — one 473 MB allocation,
/// not one per layer (that was 30 GB of nothing on a 64-layer model).
/// It lives under the same lock as the accumulators.
type RefitState = (std::collections::HashMap<usize, RefitAcc>, Vec<f32>);

static REFIT: std::sync::OnceLock<Option<(String, std::sync::Mutex<RefitState>)>> =
    std::sync::OnceLock::new();

/// Is an FFN probe accumulator installed on this thread? The fused GPU
/// FFN must decline while one is, or the probe silently measures zero.
fn ffn_probe_active() -> bool {
    FFN_PROBE.with(|p| p.borrow().is_some())
}

fn refit_dir() -> Option<&'static (String, std::sync::Mutex<RefitState>)> {
    REFIT
        .get_or_init(|| {
            std::env::var("CMF_FFN_REFIT").ok().map(|d| {
                (
                    d,
                    std::sync::Mutex::new((std::collections::HashMap::new(), Vec::new())),
                )
            })
        })
        .as_ref()
}

/// Accumulate one prefill panel into the layer's refit statistics.
fn refit_accumulate(
    li: usize,
    g: &[f32],
    b: usize,
    inter: usize,
    out: &[f32],
    hidden: usize,
    pool: Option<&Pool>,
) {
    let Some((dir, map)) = refit_dir() else {
        return;
    };
    static SPAN: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    let (from, to) = *SPAN.get_or_init(|| {
        let g = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        (
            g("CMF_FFN_REFIT_FROM", 0),
            g("CMF_FFN_REFIT_TO", usize::MAX),
        )
    });
    if li < from || li > to {
        return;
    }
    let mut guard = map.lock().unwrap();
    let (map, shared) = &mut *guard;
    let acc = match map.entry(li) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let path = format!("{dir}/support.{li}.u32");
            let Ok(bytes) = std::fs::read(&path) else {
                eprintln!("refit: no {path} — layer {li} skipped");
                return;
            };
            let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
            let support: Vec<u32> = bytes[4..4 + n * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            eprintln!(
                "refit: layer {li} support {n} ({:.0} MB of accumulator)",
                (n * n + hidden * n) as f64 * 4.0 / 1e6
            );
            e.insert(RefitAcc {
                gss: vec![0.0; n * n],
                ya: vec![0.0; hidden * n],
                buf_g: Vec::new(),
                buf_o: Vec::new(),
                buf_t: 0,
                support,
                hidden,
                tokens: 0,
            })
        }
    };
    let ns = acc.support.len();
    // Stage this chunk transposed; the GEMM fires once the batch is full.
    let cap = refit_batch();
    if acc.buf_g.is_empty() {
        acc.buf_g = vec![0.0; ns * cap];
        acc.buf_o = vec![0.0; hidden * cap];
    }
    let take = b.min(cap - acc.buf_t);
    for t in 0..take {
        let col = acc.buf_t + t;
        for (j, &n) in acc.support.iter().enumerate() {
            acc.buf_g[j * cap + col] = g[t * inter + n as usize];
        }
        for h in 0..hidden {
            acc.buf_o[h * cap + col] = out[t * hidden + h];
        }
    }
    acc.buf_t += take;
    acc.tokens += take as u64;
    if acc.buf_t < cap {
        return;
    }
    let bt = acc.buf_t;
    acc.buf_t = 0;
    // The GEMM WRITES its C (it zeroes the accumulators it uses), so the
    // chunk product lands in scratch and is added on — the one thing that
    // silently turns a Gram over 13 000 tokens into a Gram over 256.
    // Both products are `C[n, m] += X[n, b] · Yᵀ[b, m]` with X and Y
    // stored row-major [·, b] — exactly `gemm_nt_f32`'s shape, so the
    // card does them when it is up (this is the whole calibration's
    // cost: O(|S|²) per token, 2.9 PFLOP for a 27B pass). The tiled CPU
    // loop stays as the fallback. Neither accumulates, so the product
    // lands in scratch and is added on.
    let RefitAcc {
        gss,
        ya,
        buf_g,
        buf_o,
        ..
    } = acc;
    let need = (ns * ns).max(hidden * ns);
    if shared.len() < need {
        shared.resize(need, 0.0);
    }
    let scratch = &mut shared[..];
    let _ = bt;
    if crate::gpu::gemm_nt_f32_transient(buf_g, buf_g, &mut scratch[..ns * ns], ns, cap, ns) {
        add_into(gss, &scratch[..ns * ns], pool);
        if crate::gpu::gemm_nt_f32_transient(
            buf_o,
            buf_g,
            &mut scratch[..hidden * ns],
            hidden,
            cap,
            ns,
        ) {
            add_into(ya, &scratch[..hidden * ns], pool);
        } else {
            accum_outer_t(ya, hidden, ns, cap, buf_o, buf_g, pool);
        }
    } else {
        accum_outer_t(gss, ns, ns, cap, buf_g, buf_g, pool);
        accum_outer_t(ya, hidden, ns, cap, buf_o, buf_g, pool);
    }
    // No zeroing: the batch is always filled exactly (cap is a multiple
    // of the prefill chunk), and a memset of 178 MB a layer would cost
    // more than the GEMM.
}

/// `CMF_FFN_REFIT_BATCH` — tokens staged before each GEMM (default 4096).
fn refit_batch() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("CMF_FFN_REFIT_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// `c[m, n] += Σ_t left[m, t]·right[n, t]` — both operands transposed,
/// the CPU fallback for the staged batch.
fn accum_outer_t(
    c: &mut [f32],
    m: usize,
    n: usize,
    b: usize,
    left: &[f32],
    right: &[f32],
    pool: Option<&Pool>,
) {
    let ptr = SendMut(c.as_mut_ptr());
    let body = |i: usize| {
        let ptr = &ptr;
        let row = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(i * n), n) };
        for t in 0..b {
            let a = left[i * b + t];
            if a == 0.0 {
                continue;
            }
            for (j, o) in row.iter_mut().enumerate() {
                *o += a * right[j * b + t];
            }
        }
    };
    match pool {
        Some(p) if m > 1 => p.run_rows(m, &|s, e| {
            for i in s..e {
                body(i);
            }
        }),
        _ => {
            for i in 0..m {
                body(i);
            }
        }
    }
}

/// `dst += src`, spread over the pool — at 118 M floats a layer this is
/// not a loop to leave on one core.
fn add_into(dst: &mut [f32], src: &[f32], pool: Option<&Pool>) {
    let n = dst.len().min(src.len());
    match pool {
        Some(p) if n >= 1 << 16 => {
            let ptr = SendMut(dst.as_mut_ptr());
            let f = |s: usize, e: usize| {
                let ptr = &ptr;
                for blk in s..e {
                    let (a, b) = (blk * 4096, ((blk + 1) * 4096).min(n));
                    for i in a..b {
                        unsafe { *ptr.0.add(i) += src[i] };
                    }
                }
            };
            p.run_rows(n.div_ceil(4096), &f);
        }
        _ => {
            for (d, v) in dst.iter_mut().zip(&src[..n]) {
                *d += *v;
            }
        }
    }
}

/// `c[m, n] += Σ_t left[t, m]·right[t, n]`, with `left` stored [m, t] and
/// `right` [t, n]. Tiled over the rows of `c` so a tile stays in cache
/// while each token's `right` row streams past it once, and parallel
/// over tiles.
fn accum_outer(
    c: &mut [f32],
    m: usize,
    n: usize,
    b: usize,
    left: &[f32],
    right: &[f32],
    pool: Option<&Pool>,
) {
    const TILE: usize = 32;
    let tiles = m.div_ceil(TILE);
    let cp = SendMut(c.as_mut_ptr());
    let body = |ti: usize| {
        let cp = &cp;
        let i0 = ti * TILE;
        let i1 = (i0 + TILE).min(m);
        for t in 0..b {
            let r = &right[t * n..t * n + n];
            for i in i0..i1 {
                let a = left[i * b + t];
                if a == 0.0 {
                    continue;
                }
                // SAFETY: tiles partition c's rows; workers never overlap.
                let row = unsafe { std::slice::from_raw_parts_mut(cp.0.add(i * n), n) };
                for (o, v) in row.iter_mut().zip(r) {
                    *o += a * *v;
                }
            }
        }
    };
    match pool {
        Some(p) if tiles > 1 => p.run_rows(tiles, &|s, e| {
            for ti in s..e {
                body(ti);
            }
        }),
        _ => {
            for ti in 0..tiles {
                body(ti);
            }
        }
    }
}

/// Write what the calibration accumulated: `gss.<L>.f32` and `ya.<L>.f32`.
pub fn refit_flush() -> usize {
    let Some((dir, map)) = refit_dir() else {
        return 0;
    };
    let guard = map.lock().unwrap();
    let mut n = 0;
    for (li, acc) in guard.0.iter() {
        // A silently truncated write here is a Gram that reshapes to
        // nothing an hour later — say it out loud instead.
        let w = |name: &str, v: &[f32]| {
            let path = format!("{dir}/{name}.{li}.f32");
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            match std::fs::write(&path, &bytes) {
                Ok(()) => {}
                Err(e) => eprintln!(
                    "refit: FAILED to write {path} ({} MB): {e}",
                    bytes.len() / 1_000_000
                ),
            }
        };
        w("gss", &acc.gss);
        w("ya", &acc.ya);
        println!(
            "refit L{li}: {} support, {} tokens, hidden {}",
            acc.support.len(),
            acc.tokens,
            acc.hidden
        );
        n += 1;
    }
    n
}

/// `CMF_FFN_ADUMP=<prefix>` — append every probed token's FFN activation
/// row to `<prefix>.<layer>.f16`. The co-activation record: which
/// neurons fire together, which is what a tube has to group if a token
/// is ever going to open one tube instead of sixteen.
fn adump_row(li: usize, g: &[f32]) {
    use std::io::Write as _;
    static FILES: std::sync::OnceLock<
        Option<(
            String,
            std::sync::Mutex<std::collections::HashMap<usize, std::fs::File>>,
        )>,
    > = std::sync::OnceLock::new();
    let Some((prefix, map)) = FILES
        .get_or_init(|| {
            std::env::var("CMF_FFN_ADUMP")
                .ok()
                .map(|p| (p, std::sync::Mutex::new(std::collections::HashMap::new())))
        })
        .as_ref()
    else {
        return;
    };
    // `CMF_FFN_ADUMP_FROM/_TO` narrow the dump to a layer span, so a big
    // calibration run fits on disk in a few passes instead of one.
    static SPAN: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    let (from, to) = *SPAN.get_or_init(|| {
        let g = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        (
            g("CMF_FFN_ADUMP_FROM", 0),
            g("CMF_FFN_ADUMP_TO", usize::MAX),
        )
    });
    if li < from || li > to {
        return;
    }
    let mut map = map.lock().unwrap();
    let f = map.entry(li).or_insert_with(|| {
        std::fs::File::create(format!("{prefix}.{li}.f16")).expect("adump file")
    });
    let mut bytes = Vec::with_capacity(g.len() * 2);
    for v in g {
        bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(*v).to_le_bytes());
    }
    let _ = f.write_all(&bytes);
}

/// `CMF_FFN_ORACLE_TOPK` — keep only the k largest |silu(g)·u| of each
/// token and zero the rest. Not a serving mode: it is the CEILING of
/// contextual sparsity — what a per-token router would be chasing —
/// measured by cheating, since the selection reads the very activations
/// it would have to predict.
fn oracle_topk() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CMF_FFN_ORACLE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// `CMF_FFN_GATE_TOPK` — the REALIZABLE cousin of the oracle: rank the
/// neurons by their gate alone (which the kernel has computed anyway
/// before it reads `up`), keep the k best, and drop the rest. Every
/// dropped neuron's `up` row and `down` column stay unread, so this is
/// the sparsity a serving path can actually take without a router.
fn gate_topk() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CMF_FFN_GATE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// `CMF_FFN_GATE_BLOCK` — select in blocks of B neurons instead of one
/// by one. A scattered per-neuron choice cannot be read efficiently (a
/// row at a time, no prefetch runway); a block of 32 is a contiguous
/// 32-row slab of `up` and of the transposed `down`, which the ordinary
/// kernels stream. The question the measurement answers is what the
/// block costs in quality.
fn gate_block() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("CMF_FFN_GATE_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// Zero all but the `k` largest BLOCKS (by summed square) of a row.
fn keep_top_blocks(g: &mut [f32], keep_n: usize, block: usize) {
    let n = g.len();
    let nb = n.div_ceil(block);
    let kb = (keep_n.div_ceil(block)).clamp(1, nb);
    if kb >= nb {
        return;
    }
    let mut score: Vec<f32> = (0..nb)
        .map(|b| {
            g[b * block..((b + 1) * block).min(n)]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
        })
        .collect();
    let mut ord = score.clone();
    let (_, kth, _) = ord.select_nth_unstable_by(kb - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let thr = *kth;
    for b in 0..nb {
        if score[b] < thr {
            g[b * block..((b + 1) * block).min(n)].fill(0.0);
        }
    }
    score.clear();
}

/// Zero all but the `k` largest magnitudes of one token's activation row.
fn keep_top_k(g: &mut [f32], k: usize) {
    if gate_block() > 1 {
        return keep_top_blocks(g, k, gate_block());
    }
    let n = g.len();
    if k == 0 || k >= n {
        return;
    }
    let mut mag: Vec<f32> = g.iter().map(|v| v.abs()).collect();
    let (_, kth, _) = mag.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let thr = *kth;
    for v in g.iter_mut() {
        if v.abs() < thr {
            *v = 0.0;
        }
    }
}

/// `CMF_FFN_PROBE_SQ` — accumulate Σa², so the dump divided by the token
/// count and square-rooted is the RMS activation trace Patent 12 weights
/// its matrices by.
fn probe_sq() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_FFN_PROBE_SQ").is_ok())
}

/// `CMF_FFN_PROBE_SIGNED` — accumulate the SIGNED activation sum
/// instead of its magnitude: what a dropped neuron contributes ON
/// AVERAGE, which is the bias a narrowed FFN can add back for free.
fn probe_signed() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_FFN_PROBE_SIGNED").is_ok())
}

/// `CMF_FFN_MEANFILL=<file>` — a masked-out neuron contributes its MEAN
/// activation instead of zero (`u32 layers, u32 inter, f32[…]`, the mass
/// dump layout, holding per-neuron means). Dropping a neuron outright
/// also drops its average contribution, which shifts the layer output by
/// a constant; filling the mean back is one add per layer and costs no
/// bytes off the bus. This is the measurement arm — in a tube file the
/// same correction ships as a per-task bias vector.
fn meanfill() -> Option<&'static (usize, Vec<f32>)> {
    static M: std::sync::OnceLock<Option<(usize, Vec<f32>)>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let p = std::env::var("CMF_FFN_MEANFILL").ok()?;
        let b = std::fs::read(&p).ok()?;
        let inter = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
        let vals: Vec<f32> = b[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("meanfill: {} value(s), inter {inter}", vals.len());
        Some((inter, vals))
    })
    .as_ref()
}

/// `CMF_FFN_PROBE_TOPK` — 0 (default) = accumulate mass, k>0 = count
/// how often a neuron lands in a token's top k.
fn probe_topk() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CMF_FFN_PROBE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

thread_local! {
    /// DTG-MA activation probe: per-layer per-neuron Σ|silu(g)·u|
    /// accumulator, alive only during `Pipeline::probe_ffn_mass`.
    static FFN_PROBE: std::cell::RefCell<Option<Vec<Vec<f64>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Per-token structured sparsity, paid for in bytes.
///
/// The gate is the cheapest third of an FFN and it already says which
/// neurons matter: `silu(gate)` near zero means the neuron contributes
/// nothing whatever `up` says. So compute every gate, keep the `k`
/// loudest, and read ONLY those neurons' `up` rows and `down` rows —
/// the latter needs `down_proj` stored transposed, otherwise a neuron's
/// down weights are a strided column and "reading only those" costs a
/// full cache line each.
///
/// Returns `None` when the file has no transposed `down` (the caller
/// then runs the ordinary dense path).
fn dense_ffn_dynamic(d: &DenseFfn, x: &[f32], pool: Option<&Pool>, k: usize) -> Option<Vec<f32>> {
    // The scatter path reads individual rows/columns and cannot express the
    // per-matrix signed FWHT boundary.  Let the descriptor-aware dense path
    // handle Prism files rather than silently running an unrotated sparse
    // approximation.
    if d.gate_proj.has_prism_contract()
        || d.up_proj.has_prism_contract()
        || d.down_proj.has_prism_contract()
    {
        return None;
    }
    let dt = d.down_t.as_ref()?;
    let inter = d.gate_proj.rows();
    let hidden = dt.cols();
    if k == 0 || k >= inter || d.act != Act::Silu {
        return None;
    }
    DYN_SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let DynScratch {
            g,
            mag,
            live,
            parts,
        } = &mut *sc;
        g.resize(inter, 0.0);
        d.gate_proj.matvec(x, g, pool);
        for v in g.iter_mut() {
            *v = inference::silu(*v);
        }
        // The k-th largest |silu(gate)| is the threshold; ties keep more,
        // which is the safe side.
        mag.clear();
        mag.extend(g.iter().map(|v| v.abs()));
        let (_, kth, _) = mag.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let thr = *kth;
        live.clear();
        live.extend((0..inter as u32).filter(|&n| g[n as usize].abs() >= thr));
        let mut out = vec![0.0f32; hidden];
        match pool {
            Some(p) if live.len() >= 64 => {
                let nw = p.n_workers() + 1;
                parts.clear();
                parts.resize(nw * hidden, 0.0);
                let ptr = SendMut(parts.as_mut_ptr());
                let n = live.len();
                let live_ref: &[u32] = live;
                let g_ref: &[f32] = g;
                p.run(&|w, workers| {
                    let chunk = n.div_ceil(workers);
                    let (s, e) = (w * chunk, ((w + 1) * chunk).min(n));
                    if s >= e {
                        return;
                    }
                    WORKER_SCRATCH.with(|ws| {
                        let mut ws = ws.borrow_mut();
                        let [scratch, acc] = &mut *ws;
                        scratch.resize(hidden.max(x.len()), 0.0);
                        acc.clear();
                        acc.resize(hidden, 0.0);
                        for (o, &nrm) in live_ref[s..e].iter().enumerate() {
                            // One neuron of runway: the next row's lines
                            // start moving while this one is multiplied.
                            if let Some(&nx) = live_ref[s..e].get(o + 1) {
                                d.up_proj.prefetch_row(nx as usize);
                                dt.prefetch_row(nx as usize);
                            }
                            let idx = nrm as usize;
                            let up = d.up_proj.row_dot(idx, x, scratch);
                            let a = g_ref[idx] * up;
                            if a != 0.0 {
                                dt.add_row_scaled(idx, a, acc, scratch);
                            }
                        }
                        for (j, v) in acc.iter().enumerate() {
                            unsafe { *ptr.at(w * hidden + j) = *v };
                        }
                    });
                });
                for w in 0..nw {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o += parts[w * hidden + j];
                    }
                }
            }
            _ => {
                WORKER_SCRATCH.with(|ws| {
                    let mut ws = ws.borrow_mut();
                    let [scratch, _acc] = &mut *ws;
                    scratch.resize(hidden.max(x.len()), 0.0);
                    for &nrm in live.iter() {
                        let idx = nrm as usize;
                        let up = d.up_proj.row_dot(idx, x, scratch);
                        let a = g[idx] * up;
                        if a != 0.0 {
                            dt.add_row_scaled(idx, a, &mut out, scratch);
                        }
                    }
                });
            }
        }
        Some(out)
    })
}

/// Caller-side scratch of the dynamic path — one allocation per thread,
/// not one per layer per token (that alone cost a third of the decode).
struct DynScratch {
    g: Vec<f32>,
    mag: Vec<f32>,
    live: Vec<u32>,
    parts: Vec<f32>,
}

thread_local! {
    static DYN_SCRATCH: std::cell::RefCell<DynScratch> = const {
        std::cell::RefCell::new(DynScratch {
            g: Vec::new(),
            mag: Vec::new(),
            live: Vec::new(),
            parts: Vec::new(),
        })
    };
    /// Pool-worker scratch: the row buffer and this worker's partial sum.
    static WORKER_SCRATCH: std::cell::RefCell<[Vec<f32>; 2]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new()]) };
}

/// `dense_ffn_cpu` with a per-visit mask landing on the activations —
/// the masked-inference fast path's decode arm. Full fused quant
/// compute, closed neurons zeroed before down: arithmetically the
/// pruned network, no dequant, no weight bytes touched.
fn dense_ffn_masked(d: &DenseFfn, x: &[f32], pool: Option<&Pool>, mask_row: &[u8]) -> Vec<f32> {
    let inter = d.gate_proj.rows();
    FFN_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let [g, u, ..] = &mut *s;
        g.resize(inter, 0.0);
        if d.act == Act::Silu && QTensor::matvec_silu_mul(&d.gate_proj, &d.up_proj, x, g, pool) {
            // g holds silu(gate)·up.
        } else {
            u.resize(inter, 0.0);
            QTensor::matvec_many([&d.gate_proj, &d.up_proj], x, [g, u], pool);
            for i in 0..inter {
                g[i] = d.act.combine(g[i], u[i]);
            }
        }
        zero_masked_cols(g, 1, inter, mask_row);
        let mut out = attention::take_buf(d.down_proj.rows());
        d.down_proj.matvec(g, &mut out, pool);
        out
    })
}

/// Dense FFN as one GPU submission via the MoE block path (single
/// expert, weight 1.0): gate → silu·up → down chained in one command
/// buffer, intermediate activations device-resident. None → weights
/// not q8-mapped in the primary shard / over the VRAM budget / backend
/// refusal → honest CPU path.
fn dense_ffn_gpu(d: &DenseFfn, x: &[f32], _pool: Option<&Pool>) -> Option<Vec<f32>> {
    if d.gate_proj.has_prism_contract()
        || d.up_proj.has_prism_contract()
        || d.down_proj.has_prism_contract()
    {
        return None;
    }
    // The GPU block hardcodes SiLU; GeLU FFNs (Gemma) stay on CPU.
    if d.act != Act::Silu {
        return None;
    }
    // Threshold: tiny FFNs are not worth a submission (q1 excepted —
    // see the caller's gate).
    if d.gate_proj.rows() < crate::gpu::min_rows() && !d.gate_proj.is_q1() {
        return None;
    }
    let mut jobs: Vec<crate::gpu::MoeJob> = Vec::with_capacity(1);
    let mut model_ref = None;
    moe_push_job(d, x, 1.0, &mut jobs, &mut model_ref)?;
    let model = model_ref?;
    let hidden = jobs[0].down.1;
    let mut out = attention::take_buf(hidden);
    if crate::gpu::moe_block(&model, &jobs, &mut out) {
        Some(out)
    } else {
        let mut out = out;
        attention::recycle_buf(&mut out);
        None
    }
}

/// q8-mapped primary-shard tensor parts for a GPU job: q8_2f carries
/// its column field, q8_row runs with empty col slices (the backend
/// skips the multiply). Shared by the MoE block and the dense-FFN
/// single-job path.
#[allow(clippy::type_complexity)]
#[allow(clippy::type_complexity)]
pub(crate) fn moe_parts(
    t: &QTensor,
) -> Option<(
    &std::sync::Arc<cortiq_core::CmfModel>,
    usize,
    usize,
    usize,
    &[f32],
    &[f32],
    bool,
    bool,
    bool,
)> {
    match t {
        QTensor::Mapped {
            model,
            idx,
            dtype: dt @ (cortiq_core::TensorDtype::Q8_2f | cortiq_core::TensorDtype::Q8Row),
            rows,
            cols,
            row_scale,
            col_field,
            ..
        } if (*dt == cortiq_core::TensorDtype::Q8Row) || !col_field.is_empty() => Some((
            model, *idx, *rows, *cols, row_scale, col_field, false, false, false,
        )),
        // q1: tile-embedded scales — empty rs/col slices, raw xs.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q1,
            rows,
            cols,
            ..
        } => Some((
            model,
            *idx,
            *rows,
            *cols,
            &[][..],
            &[][..],
            true,
            false,
            false,
        )),
        // q4_tiled: 18-byte tiles with embedded f16 scales — raw xs.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q4Tiled,
            rows,
            cols,
            ..
        } => Some((
            model,
            *idx,
            *rows,
            *cols,
            &[][..],
            &[][..],
            false,
            true,
            false,
        )),
        // q4tp: same raw-xs contract, different stride and scale plane.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q4TiledP,
            rows,
            cols,
            ..
        } => Some((
            model,
            *idx,
            *rows,
            *cols,
            &[][..],
            &[][..],
            false,
            true,
            false,
        )),
        // q2tp: the 2-bit expert plane of the mixed profile — q4 family
        // for stride bookkeeping, flagged q2 so the trio validation can
        // demand a q4tp down.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q2TiledP,
            rows,
            cols,
            ..
        } => Some((
            model,
            *idx,
            *rows,
            *cols,
            &[][..],
            &[][..],
            false,
            true,
            true,
        )),
        _ => None,
    }
}

/// Map a MoE onto the Metal token graph's contract: f32 router, a
/// shared expert (gated — Qwen — or ungated at weight 1 — DeepSeek-V3 /
/// HunYuan hy_v3), softmax or sigmoid scores with an optional selection
/// bias and routed scale, experts uniformly q4tp (or the mixed profile:
/// q2tp gate/up over a q4tp down). τ routers, masks, per-expert scales
/// and Gemma's router-input norm refuse here — those semantics stay on
/// the CPU path.
#[cfg(target_os = "macos")]
fn metal_moe_graph_parts(m: &MoeFfn, hidden: usize) -> Option<crate::gpu::GpuMoe<'_>> {
    if m.router_input_norm
        || m.route_tau.is_some()
        || m.mask.is_some()
        || m.per_expert_scale.is_some()
        || m.experts.is_empty()
        || m.top_k == 0
        || m.resonance.is_some()
    {
        return None;
    }
    // The select kernel always fills the shared slot: a model without a
    // shared expert (LFM2-MoE) stays on the CPU path here.
    let (sh, sg) = match &m.shared {
        Some((sh, sg)) => (sh, sg.as_ref()),
        None => return None,
    };
    let (rf, rr, rc) = m.router.f32_parts()?;
    if rr != m.experts.len() || rc != hidden {
        return None;
    }
    let shared_gated = sg.is_some();
    let sf = match sg {
        Some(sg) => {
            let (sf, sr, sc) = sg.f32_parts()?;
            if sr * sc != hidden {
                return None;
            }
            sf
        }
        // Ungated: the router's first row stands in for the gate matvec
        // (its logit is never read — the kernel pins weight 1).
        None => &rf[..hidden],
    };
    if let Some(b) = &m.expert_bias {
        if b.len() != m.experts.len() {
            return None;
        }
    }
    let inter = m.experts[0].gate_proj.rows();
    // The first expert's gate decides the profile; every trio (shared
    // included) must agree — the jobs ladder flips ONE kernel for all.
    let gu_q2 = m.experts[0].gate_proj.mapped_q2tp().is_some();
    let trio = |e: &DenseFfn| -> Option<(usize, usize, usize)> {
        if e.act != Act::Silu
            || e.gate_proj.rows() != inter
            || e.gate_proj.cols() != hidden
            || e.up_proj.rows() != inter
            || e.up_proj.cols() != hidden
            || e.down_proj.rows() != hidden
            || e.down_proj.cols() != inter
        {
            return None;
        }
        let pick = |t: &QTensor| -> Option<usize> {
            if gu_q2 {
                t.mapped_q2tp().map(|(_, i)| i)
            } else {
                t.mapped_q4tp().map(|(_, i)| i)
            }
        };
        Some((
            pick(&e.gate_proj)?,
            pick(&e.up_proj)?,
            e.down_proj.mapped_q4tp().map(|(_, i)| i)?,
        ))
    };
    let experts = m.experts.iter().map(trio).collect::<Option<Vec<_>>>()?;
    let shared = trio(sh)?;
    Some(crate::gpu::GpuMoe {
        router: rf,
        sgate: sf,
        experts,
        shared,
        n_exp: m.experts.len(),
        top_k: m.top_k,
        inter,
        norm_topk: m.norm_topk_prob,
        route_scale: m.routed_scaling,
        gu_q2,
        sigmoid: m.router_sigmoid,
        bias: m.expert_bias.as_deref(),
        shared_gated,
    })
}

/// Build one gate/up/down GPU job from three tensors. `moe_push_job` is the
/// DenseFfn-shaped caller; architectures that keep their experts in their own
/// structs (DeepSeek-V4) come here directly.
pub(crate) fn moe_push_job_parts<'a>(
    gate: &'a QTensor,
    up: &'a QTensor,
    down: &'a QTensor,
    x: &[f32],
    w: f32,
    swiglu_limit: f32,
    jobs: &mut Vec<crate::gpu::MoeJob<'a>>,
    model_ref: &mut Option<std::sync::Arc<cortiq_core::CmfModel>>,
) -> Option<()> {
    use crate::qtensor::prescale;
    let (gm, gi, gr, gc, grs, gcf, gq1, gq4, gq2) = moe_parts(gate)?;
    let (_, ui, ur, uc, urs, ucf, uq1, uq4, uq2) = moe_parts(up)?;
    let (_, di, dr, dc, drs, dcf, dq1, dq4, dq2) = moe_parts(down)?;
    if gq1 != uq1 || uq1 != dq1 || gq4 != uq4 || uq4 != dq4 || gq2 != uq2 {
        return None; // mixed-dtype trio — honest CPU path
    }
    // The 2-bit profile is gate/up q2tp over a PLAIN q4tp down; any other
    // 2-bit arrangement stays on the CPU.
    if gq2 && (dq2 || !dq4 || down.mapped_q4tp().is_none()) {
        return None;
    }
    if !gq2 && dq2 {
        return None;
    }
    model_ref.get_or_insert_with(|| gm.clone());
    let dt = |cf: &[f32]| {
        if cf.is_empty() {
            cortiq_core::TensorDtype::Q8Row
        } else {
            cortiq_core::TensorDtype::Q8_2f
        }
    };
    jobs.push(crate::gpu::MoeJob {
        gate: (gi, gr, gc, grs),
        up: (ui, ur, uc, urs),
        down: (di, dr, dc, drs),
        xs_gate: prescale(x, gcf, dt(gcf)).into_owned(),
        xs_up: prescale(x, ucf, dt(ucf)).into_owned(),
        down_col: dcf,
        w,
        q1: gq1,
        q4t: gq4 && !gq2 && gate.mapped_q4tp().is_none(),
        q4tp: gq4 && (gq2 || gate.mapped_q4tp().is_some()),
        gu_q2: gq2,
        swiglu_limit,
    });
    Some(())
}

/// Build one gate/up/down GPU job (see `moe_parts`).
fn moe_push_job<'a>(
    d: &'a DenseFfn,
    x: &[f32],
    w: f32,
    jobs: &mut Vec<crate::gpu::MoeJob<'a>>,
    model_ref: &mut Option<std::sync::Arc<cortiq_core::CmfModel>>,
) -> Option<()> {
    use crate::qtensor::prescale;
    if d.act != Act::Silu {
        return None; // GPU block hardcodes SiLU
    }
    let (gm, gi, gr, gc, grs, gcf, gq1, gq4, gq2) = moe_parts(&d.gate_proj)?;
    let (_, ui, ur, uc, urs, ucf, uq1, uq4, uq2) = moe_parts(&d.up_proj)?;
    let (_, di, dr, dc, drs, dcf, dq1, dq4, dq2) = moe_parts(&d.down_proj)?;
    if gq1 != uq1 || uq1 != dq1 || gq4 != uq4 || uq4 != dq4 || gq2 != uq2 {
        return None; // mixed-dtype trio — honest CPU path
    }
    if gq2 && (dq2 || !dq4 || d.down_proj.mapped_q4tp().is_none()) {
        return None;
    }
    if !gq2 && dq2 {
        return None;
    }
    model_ref.get_or_insert_with(|| gm.clone());
    let gdt = if gcf.is_empty() {
        cortiq_core::TensorDtype::Q8Row
    } else {
        cortiq_core::TensorDtype::Q8_2f
    };
    let udt = if ucf.is_empty() {
        cortiq_core::TensorDtype::Q8Row
    } else {
        cortiq_core::TensorDtype::Q8_2f
    };
    jobs.push(crate::gpu::MoeJob {
        gate: (gi, gr, gc, grs),
        up: (ui, ur, uc, urs),
        down: (di, dr, dc, drs),
        xs_gate: prescale(x, gcf, gdt).into_owned(),
        xs_up: prescale(x, ucf, udt).into_owned(),
        down_col: dcf,
        w,
        q1: gq1,
        q4t: gq4 && !gq2 && d.gate_proj.mapped_q4tp().is_none(),
        q4tp: gq4 && (gq2 || d.gate_proj.mapped_q4tp().is_some()),
        gu_q2: gq2,
        swiglu_limit: 0.0,
    });
    Some(())
}

/// Sparse dense-FFN directly on QUANTIZED weights (mask × mmap): reads
/// ONLY the active neurons' gate/up rows and down columns from the mmap
/// — no full-matrix dequant, no f32 model copy. This is what lets a
/// masked big model run at quantized RSS (the historical mask path
/// forced the whole model to f32). Semantics identical to the f32
/// sparse path within quant tolerance.
fn sparse_ffn_quant(
    d: &DenseFfn,
    x: &[f32],
    active: &[u16],
    hidden: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let n = active.len();
    let inter = d.gate_proj.rows();
    let mut act = vec![0.0f32; n];
    // Scratch is needed if EITHER projection is group-packed (q4/vbit);
    // gate/up normally share a dtype but sizing on both is robust.
    let need_scratch = !(d.gate_proj.sparse_col_ok() && d.up_proj.sparse_col_ok());
    let compute = |ai: usize| -> f32 {
        let idx = active[ai] as usize;
        if idx >= inter {
            return 0.0; // defensive parity with the f32 sparse path
        }
        let mut s = if need_scratch {
            vec![0.0f32; hidden]
        } else {
            Vec::new()
        };
        let gate = d.gate_proj.row_dot(idx, x, &mut s);
        let up = d.up_proj.row_dot(idx, x, &mut s);
        d.act.combine(gate, up)
    };
    match pool {
        Some(p) if n >= 256 => {
            let ptr = SendMut(act.as_mut_ptr());
            p.run(&|widx, nw| {
                let chunk = n.div_ceil(nw);
                let (s, e) = (widx * chunk, ((widx + 1) * chunk).min(n));
                for ai in s..e {
                    unsafe { *ptr.at(ai) = compute(ai) };
                }
            });
        }
        _ => {
            for (ai, a) in act.iter_mut().enumerate() {
                *a = compute(ai);
            }
        }
    }
    // Scatter through active down columns (reads only those columns).
    let mut out = vec![0.0f32; hidden];
    for (ai, &idx) in active.iter().enumerate() {
        let w = act[ai];
        if w.abs() >= 1e-12 && (idx as usize) < inter {
            d.down_proj.add_col_scaled(idx as usize, w, &mut out);
        }
    }
    out
}

/// Test-only re-export of the private sparse-quant FFN (mask × mmap gate).
#[doc(hidden)]
pub fn sparse_ffn_quant_for_test(
    d: &DenseFfn,
    x: &[f32],
    active: &[u16],
    hidden: usize,
) -> Vec<f32> {
    sparse_ffn_quant(d, x, active, hidden, None)
}

/// Dequantize a DenseFfn's three matrices to f32 (transient; only the
/// q4/vbit-masked fallback uses it — the memory-lean path is
/// sparse_ffn_quant). Reuses row_f32 row-by-row.
fn dequant_dense_f32(d: &DenseFfn) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let deq = |t: &QTensor| -> Vec<f32> {
        let (rows, cols) = (t.rows(), t.cols());
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            t.row_f32(r, &mut out[r * cols..(r + 1) * cols]);
        }
        out
    };
    (deq(&d.gate_proj), deq(&d.up_proj), deq(&d.down_proj))
}

/// Pointer wrapper for the worker-pool scatter (same pattern as qtensor).
struct SendMut(*mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}
impl SendMut {
    #[inline]
    // Deliberate unsynchronized scatter: pool workers write disjoint indices
    // in parallel, so returning `&mut` from `&self` is intentional here.
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize) -> &mut f32 {
        unsafe { &mut *self.0.add(i) }
    }
}

/// Router → (selected experts in torch.topk order, per-expert score
/// vector, normalizer). The final weight of expert `e` is `p[e] / wsum`.
///
/// Two regimes share this. Qwen: softmax over ALL experts, top-k of the
/// probabilities, optional renorm — `router_sigmoid=false`, no bias,
/// scale 1 → bit-identical to the historical path. LFM2-MoE /
/// DeepSeek-V3 `noaux_tc`: per-expert sigmoid scores, an optional
/// selection bias (top-k CHOICE only; weights stay unbiased), a 1e-6 renorm
/// floor and a routed scale.
pub(crate) fn moe_route(
    logits: &[f32],
    m: &MoeFfn,
    allowed: Option<&[bool]>,
) -> (Vec<usize>, Vec<f32>, f32) {
    let ne = logits.len();
    let p: Vec<f32> = if m.router_sigmoid {
        logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect()
    } else {
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut e: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        for v in &mut e {
            *v /= s;
        }
        e
    };
    // Expert restriction: the static env mask (CMF_MOE_MASK) AND the
    // active task mask's expert fields (spec §5) both narrow the
    // candidate set; selection happens over the admitted experts only.
    // With norm_topk the kept weights renormalize below; without it
    // the excluded mass is honestly dropped.
    let admit = |e: usize| {
        m.mask.as_ref().is_none_or(|mk| mk[e])
            && allowed.is_none_or(|a| a.get(e).copied().unwrap_or(false))
    };
    let mut idx: Vec<usize> = (0..ne).filter(|&e| admit(e)).collect();
    // Descending by selection score, lower index wins ties (torch.topk).
    match &m.expert_bias {
        Some(b) => idx.sort_unstable_by(|&x, &y| {
            (p[y] + b[y])
                .partial_cmp(&(p[x] + b[x]))
                .unwrap()
                .then(x.cmp(&y))
        }),
        None => idx.sort_unstable_by(|&x, &y| p[y].partial_cmp(&p[x]).unwrap().then(x.cmp(&y))),
    }
    idx.truncate(m.top_k);
    // Adaptive τ-routing: trim the tail experts once the kept mass is
    // enough. wsum below renormalizes over the KEPT set, so the output
    // stays a proper weighted average.
    if let Some(tau) = m.route_tau {
        let total: f32 = idx.iter().map(|&e| p[e]).sum();
        if total > 0.0 {
            let mut acc = 0.0f32;
            let mut keep = idx.len();
            for (i, &e) in idx.iter().enumerate() {
                acc += p[e];
                if acc >= tau * total {
                    keep = i + 1;
                    break;
                }
            }
            idx.truncate(keep);
        }
    }
    let wsum: f32 = if m.norm_topk_prob {
        let s: f32 = idx.iter().map(|&e| p[e]).sum();
        // LFM2 floors the denom (matches HF `+ 1e-6`); the softmax path's
        // probs already sum near 1, so it stays exactly as before.
        (if m.router_sigmoid { s + 1e-6 } else { s }) / m.routed_scaling
    } else {
        1.0 / m.routed_scaling
    };
    (idx, p, wsum)
}

/// See the call site: one `layer:e1,e2,…` line per routed token.
fn moe_trace(idx: &[usize]) {
    moe_trace_at(crate::gpu::cur_layer() as i32, idx)
}

/// The same, for callers that know their layer (DSV4 owns its layers and
/// never sets the pipeline's current-layer marker).
pub(crate) fn moe_trace_at(li: i32, idx: &[usize]) {
    use std::io::Write;
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> =
        std::sync::OnceLock::new();
    let Some(f) = F.get_or_init(|| {
        let p = std::env::var("CMF_MOE_TRACE").ok()?;
        Some(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()?,
        ))
    }) else {
        return;
    };
    let ids: Vec<String> = idx.iter().map(|e| e.to_string()).collect();
    let _ = writeln!(f.lock().unwrap(), "{li}:{}", ids.join(","));
}

/// MoE FFN: router → top-k experts (see `moe_route`). Only selected
/// experts' pages are touched in mmap.
pub(crate) fn moe_ffn(
    m: &MoeFfn,
    x: &[f32],
    pool: Option<&Pool>,
    allowed: Option<&[bool]>,
) -> Vec<f32> {
    accumulate_act(m, x, 1);
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; ne];
    match &m.resonance {
        Some(r) => r.scores(x, &mut logits),
        None => m.router.matvec(x, &mut logits, pool),
    }
    let (idx, p, wsum) = moe_route(&logits, m, allowed);
    {
        let mut st = m.stats.borrow_mut();
        if st.len() < ne {
            st.resize(ne, 0);
        }
        for &e in &idx {
            st[e] += 1;
        }
    }
    // `CMF_MOE_TRACE=<file>`: append one line per (layer, token) with the
    // selected expert ids. The cumulative `stats` above answer "which
    // experts are popular"; a residency design needs the question they
    // cannot answer — whether CONSECUTIVE tokens reuse experts (the
    // temporal locality an LRU cache lives on, FreeToken §4).
    moe_trace(&idx);
    // D5: the whole layer MoE block in one GPU command buffer (experts — the
    // same mmap via a no-copy buffer; intermediate activations on the GPU).
    // Same Ffn probe class as the dense chain: one submit per layer
    // either wins on this driver stack or it doesn't.
    if crate::gpu::enabled_here() {
        match crate::gpu::probe_arm(crate::gpu::OpClass::Ffn) {
            crate::gpu::ProbeArm::Gpu => {
                let t0 = std::time::Instant::now();
                if let Some(out) = moe_ffn_gpu(m, x, &idx, &p, wsum, pool) {
                    crate::gpu::probe_record(crate::gpu::OpClass::Ffn, true, t0.elapsed());
                    return out;
                }
            }
            crate::gpu::ProbeArm::CpuTimed => {
                let t0 = std::time::Instant::now();
                let out = crate::gpu::cpu_scope(|| moe_ffn_cpu(m, x, &idx, &p, wsum, pool));
                crate::gpu::probe_record(crate::gpu::OpClass::Ffn, false, t0.elapsed());
                return out;
            }
            crate::gpu::ProbeArm::Cpu => {
                return crate::gpu::cpu_scope(|| moe_ffn_cpu(m, x, &idx, &p, wsum, pool));
            }
        }
    }
    moe_ffn_cpu(m, x, &idx, &p, wsum, pool)
}

/// One-shot report of whether the whole-token wgpu graph actually formed.
/// A refusal silently reverts to the per-op path, which is how a model can
/// look "GPU-accelerated" while every layer walks the host.  A device prefix
/// is tracked separately because it still pays a host boundary for the tail.
fn graph_note(built: bool, layers_run: usize, total_layers: usize) {
    use std::sync::atomic::{AtomicBool, Ordering};
    if built {
        GRAPH_TOK_OK.fetch_add(1, Ordering::Relaxed);
        if total_layers > 0 && layers_run < total_layers {
            GRAPH_TOK_PREFIX.fetch_add(1, Ordering::Relaxed);
        } else {
            GRAPH_TOK_FULL.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        GRAPH_TOK_MISS.fetch_add(1, Ordering::Relaxed);
    }
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        if built {
            tracing::info!("wgpu whole-token graph: ACTIVE");
        } else {
            tracing::warn!("wgpu whole-token graph refused — per-op path");
        }
    }
}

/// Whole-token graph outcomes, process-wide: a benchmark that claims a
/// GPU number while MISS climbs is measuring the CPU — the honest-bench
/// contract makes that an error, not a footnote.
pub static GRAPH_TOK_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static GRAPH_TOK_MISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Graph calls that returned a hidden after running only a leading device
/// prefix.  These are valid hybrid executions but must not be reported as a
/// full GPU graph in benchmark evidence.
pub static GRAPH_TOK_PREFIX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Graph calls that covered the complete requested layer span.
pub static GRAPH_TOK_FULL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Native Metal TokenGraph completion counters. These are incremented only
/// after checked command-buffer completion and successful readback, so a
/// fused-head NLL report can prove the route rather than infer it from env.
pub static METAL_GRAPH_TOK_OK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_GRAPH_HEAD_OK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_GRAPH_HEAD_MISS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_GRAPH_LAYERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_GRAPH_ERRORS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Ordinary native-Metal rows-prefill admissions and completed rows.  These
/// counters are separate from TokenGraph token/head counts so a batch NLL
/// receipt cannot accidentally claim serial execution as batched.
pub static METAL_PREFILL_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_PREFILL_ROWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_PREFILL_HEAD_ROWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METAL_PREFILL_ERRORS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `CMF_MOE_BATCH=0` restores the per-expert serial loop — the A/B lever
/// for the batched kernel, and how its bit-identity is checked.
fn moe_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_MOE_BATCH").as_deref() != Ok("0"))
}

/// Two-dispatch CPU MoE: every routed expert (and the shared one) fused
/// into one gate/up/SiLU dispatch and one down dispatch, instead of two
/// pool barriers per expert. Bit-identical to the serial loop below —
/// see `moe_gate_up_many` / `moe_down_many`. `None` = the batched kernel
/// does not cover this layer, walk the serial path.
fn moe_ffn_cpu_batched(
    m: &MoeFfn,
    x: &[f32],
    idx: &[usize],
    p: &[f32],
    wsum: f32,
    pool: Option<&Pool>,
) -> Option<Vec<f32>> {
    if idx.is_empty() || !moe_batch_enabled() {
        return None;
    }
    // The bake probe reads per-neuron activation mass out of the
    // single-expert path; batching would skip it. Rare and offline —
    // hand those runs to the serial loop.
    if FFN_PROBE.with(|pr| pr.borrow().is_some()) {
        return None;
    }
    let n = idx.len() + usize::from(m.shared.is_some());
    let mut pairs = Vec::with_capacity(n);
    let mut downs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for &e in idx {
        let d = &m.experts[e];
        if d.act != Act::Silu {
            return None;
        }
        pairs.push((&d.gate_proj, &d.up_proj));
        downs.push(&d.down_proj);
        ws.push(p[e] / wsum * m.per_expert_scale.as_ref().map_or(1.0, |v| v[e]));
    }
    // The shared expert goes last, matching the serial loop's order —
    // the f32 accumulation order is part of the bit-identity claim.
    if let Some((se, gate)) = &m.shared {
        if se.act != Act::Silu {
            return None;
        }
        let g = gate.as_ref().map_or(1.0, |gate| {
            let mut gl = [0.0f32; 1];
            gate.matvec(x, &mut gl, pool);
            1.0 / (1.0 + (-gl[0]).exp())
        });
        pairs.push((&se.gate_proj, &se.up_proj));
        downs.push(&se.down_proj);
        ws.push(g);
    }
    let inter = pairs[0].0.rows();
    let mut gs: Vec<Vec<f32>> = (0..pairs.len()).map(|_| vec![0f32; inter]).collect();
    if !QTensor::moe_gate_up_many(&pairs, x, &mut gs, pool) {
        return None;
    }
    let mut out = attention::take_buf(x.len());
    if !QTensor::moe_down_many(&downs, &gs, &ws, &mut out, pool) {
        attention::recycle_buf(&mut out);
        return None;
    }
    Some(out)
}

/// Exact CPU completion for the routed experts a dynamic device cache did
/// not contain. The weights are already the router's final normalized mix.
/// Keeping this independent of `MoeFfn` makes the job `Sync`: its routing
/// statistics live in a `RefCell`, while the immutable expert tensors can be
/// evaluated safely in parallel with the GPU's resident subset.
pub(crate) fn moe_cold_experts_cpu(
    experts: &[(&DenseFfn, f32)],
    x: &[f32],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = attention::take_buf(x.len());
    if experts.is_empty() {
        return out;
    }
    let pairs: Vec<_> = experts
        .iter()
        .map(|(e, _)| (&e.gate_proj, &e.up_proj))
        .collect();
    let downs: Vec<_> = experts.iter().map(|(e, _)| &e.down_proj).collect();
    let weights: Vec<_> = experts.iter().map(|(_, w)| *w).collect();
    let inter = experts[0].0.gate_proj.rows();
    let mut activations: Vec<Vec<f32>> = (0..experts.len()).map(|_| vec![0.0; inter]).collect();
    if QTensor::moe_gate_up_many(&pairs, x, &mut activations, pool)
        && QTensor::moe_down_many(&downs, &activations, &weights, &mut out, pool)
    {
        return out;
    }
    out.fill(0.0);
    for &(expert, weight) in experts {
        let mut one = dense_ffn(expert, x, pool);
        for (o, v) in out.iter_mut().zip(&one) {
            *o += weight * v;
        }
        attention::recycle_buf(&mut one);
    }
    out
}

/// The pure-CPU MoE expert loop (also the fallback of every GPU refusal).
fn moe_ffn_cpu(
    m: &MoeFfn,
    x: &[f32],
    idx: &[usize],
    p: &[f32],
    wsum: f32,
    pool: Option<&Pool>,
) -> Vec<f32> {
    if let Some(out) = moe_ffn_cpu_batched(m, x, idx, p, wsum, pool) {
        return out;
    }
    let mut out = attention::take_buf(x.len());
    for &e in idx {
        let mut eo = dense_ffn(&m.experts[e], x, pool);
        let w = p[e] / wsum * m.per_expert_scale.as_ref().map_or(1.0, |v| v[e]);
        for i in 0..out.len() {
            out[i] += w * eo[i];
        }
        attention::recycle_buf(&mut eo);
    }
    if let Some((se, gate)) = &m.shared {
        let mut so = dense_ffn(se, x, pool);
        let g = gate.as_ref().map_or(1.0, |gate| {
            let mut gl = [0.0f32; 1];
            gate.matvec(x, &mut gl, pool);
            1.0 / (1.0 + (-gl[0]).exp())
        });
        for i in 0..out.len() {
            out[i] += g * so[i];
        }
        attention::recycle_buf(&mut so);
    }
    out
}

/// DeepSeek-V2 MLA forward, expand-to-MHA form (see `AttnKind::Mla`):
/// per token the latent expands to every head's K/V and the ordinary
/// cache + grouped attend do the rest. K head layout is [rope | nope]
/// (rotary_dim = qk_rope rotates the shared rope key and each q head's
/// prefix); V rows are zero-padded to the K head_dim inside the cache
/// and the pad is sliced off before O. Attention importance is not
/// accumulated for MLA yet (no eviction interplay).
#[allow(clippy::too_many_arguments)]
fn mla_attention(
    w: &MlaWeights,
    normed: &[f32],
    cache: &mut crate::kv_cache::LayerKvCache,
    position: usize,
    inv_freq: &[f32],
    rope_scale: f32,
    eps: f64,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let (nh, dr, dn, dv, lora) = (w.nh, w.qk_rope, w.qk_nope, w.v_dim, w.lora);
    let hd = dr + dn;
    let mut q = vec![0.0f32; nh * hd];
    match (&w.q_a, &w.q_a_norm) {
        (Some(qa), Some(qn)) => {
            let mut t = vec![0.0f32; qa.rows()];
            qa.matvec(normed, &mut t, pool);
            let tn = inference::rms_norm(&t, qn, eps, NormStyle::Qwen);
            w.q_proj.matvec(&tn, &mut q, pool);
        }
        _ => w.q_proj.matvec(normed, &mut q, pool),
    }
    let mut ca = vec![0.0f32; lora + dr];
    w.kv_a.matvec(normed, &mut ca, pool);
    let (c_lat, k_rope) = ca.split_at_mut(lora);
    let latn = inference::rms_norm(c_lat, &w.kv_a_norm, eps, NormStyle::Qwen);
    let mut kvb = vec![0.0f32; nh * (dn + dv)];
    w.kv_b.matvec(&latn, &mut kvb, pool);
    if !w.nope {
        attention::rope_rotate_scaled(k_rope, position, inv_freq, rope_scale);
    }
    for h in 0..nh {
        if !w.nope {
            attention::rope_rotate_scaled(
                &mut q[h * hd..h * hd + dr],
                position,
                inv_freq,
                rope_scale,
            );
        }
    }
    let mut k = vec![0.0f32; nh * hd];
    let mut v = vec![0.0f32; nh * hd];
    for h in 0..nh {
        k[h * hd..h * hd + dr].copy_from_slice(k_rope);
        k[h * hd + dr..(h + 1) * hd].copy_from_slice(&kvb[h * (dn + dv)..h * (dn + dv) + dn]);
        v[h * hd..h * hd + dv].copy_from_slice(&kvb[h * (dn + dv) + dn..(h + 1) * (dn + dv)]);
    }
    cache.append(&k, &v, &vec![true; nh]);
    let (ao, mut imp) = attention::attend_all_heads(&q, cache, nh, 1, hd, w.scale, None, 0.0);
    attention::recycle_buf(&mut imp);
    let mut ov = vec![0.0f32; nh * dv];
    for h in 0..nh {
        ov[h * dv..(h + 1) * dv].copy_from_slice(&ao[h * hd..h * hd + dv]);
    }
    let mut out = vec![0.0f32; w.o_proj.rows()];
    w.o_proj.matvec(&ov, &mut out, pool);
    out
}

/// Gemma-4 dual-branch FFN (spec: see `FfnKind::DenseMoe`). The dense
/// branch reads the pre-FFN-normed activation; the router and the
/// expert branch read the RAW residual — the router through a
/// scale-less rms norm (its constant gain is folded into the weights),
/// the experts through `pre_norm_2`. CPU path; GPU graphs refuse the
/// layer kind honestly.
fn dense_moe_ffn(
    dm: &DenseMoeFfn,
    x_normed: &[f32],
    h_raw: &[f32],
    eps: f64,
    norm_style: NormStyle,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut d = dense_ffn(&dm.dense, x_normed, pool);
    d = inference::rms_norm(&d, &dm.post_norm_1, eps, norm_style);
    let m = &dm.moe;
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; ne];
    if m.router_input_norm {
        let ss: f32 = h_raw.iter().map(|v| v * v).sum::<f32>() / h_raw.len() as f32;
        let inv = 1.0 / (ss + eps as f32).sqrt();
        let xr: Vec<f32> = h_raw.iter().map(|v| v * inv).collect();
        m.router.matvec(&xr, &mut logits, pool);
    } else {
        m.router.matvec(h_raw, &mut logits, pool);
    }
    let (idx, p, wsum) = moe_route(&logits, m, None);
    {
        let mut st = m.stats.borrow_mut();
        if st.len() < ne {
            st.resize(ne, 0);
        }
        for &e in &idx {
            st[e] += 1;
        }
    }
    let x2 = inference::rms_norm(h_raw, &dm.pre_norm_2, eps, norm_style);
    let mo = moe_ffn_cpu(m, &x2, &idx, &p, wsum, pool);
    let mo = inference::rms_norm(&mo, &dm.post_norm_2, eps, norm_style);
    for (di, mi) in d.iter_mut().zip(&mo) {
        *di += mi;
    }
    d
}

/// Building the MoE-layer GPU jobs: all selected experts (+shared) must
/// be q8_2f-Mapped from the primary mapping; otherwise None → CPU path.
/// One-shot report of why the MoE GPU block refused. A silent `?` here
/// sends every expert to the CPU with nothing in the logs to say so —
/// which is exactly how a q4tp MoE model looked "GPU-accelerated" while
/// running entirely on the host.
fn moe_gpu_refused(why: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!("MoE GPU block refused ({why}) — experts run on the CPU");
    }
}

fn moe_ffn_gpu(
    m: &MoeFfn,
    x: &[f32],
    idx: &[usize],
    p: &[f32],
    wsum: f32,
    pool: Option<&Pool>,
) -> Option<Vec<f32>> {
    use crate::gpu::MoeJob;

    let mut jobs: Vec<MoeJob> = Vec::with_capacity(idx.len() + 1);
    let mut model_ref = None;
    for &e in idx {
        if moe_push_job(&m.experts[e], x, p[e] / wsum, &mut jobs, &mut model_ref).is_none() {
            moe_gpu_refused("push_job(expert)");
            return None;
        }
    }
    if let Some((se, gate)) = &m.shared {
        let g = gate.as_ref().map_or(1.0, |gate| {
            let mut gl = [0.0f32; 1];
            gate.matvec(x, &mut gl, pool);
            1.0 / (1.0 + (-gl[0]).exp())
        });
        if moe_push_job(se, x, g, &mut jobs, &mut model_ref).is_none() {
            moe_gpu_refused("push_job(shared)");
            return None;
        }
    }
    let Some(model) = model_ref else {
        moe_gpu_refused("no model_ref");
        return None;
    };
    let hidden = jobs[0].down.1;
    let mut out = vec![0.0f32; hidden];
    if crate::gpu::moe_block(&model, &jobs, &mut out) {
        Some(out)
    } else {
        moe_gpu_refused("gpu::moe_block");
        None
    }
}

/// Single-position FFN dispatch.
fn ffn_forward(
    ffn: &FfnKind,
    x: &[f32],
    pool: Option<&Pool>,
    experts_allowed: Option<&[bool]>,
) -> Vec<f32> {
    match ffn {
        FfnKind::Dense(d) if !d.segs.is_empty() => tube_ffn(d, x, 1, pool, None),
        FfnKind::Dense(d) => dense_ffn(d, x, pool),
        FfnKind::Moe(m) => moe_ffn(m, x, pool, experts_allowed),
        // Dual-branch layers need the raw residual — their callers
        // dispatch dense_moe_ffn directly; the auxiliary paths that land
        // here (MTP draft, o1 replay) do not co-occur with gemma-4 MoE.
        FfnKind::DenseMoe(_) => unreachable!("DenseMoe dispatches via dense_moe_ffn"),
    }
}

/// Fused two-position FFN: gate/up/down streamed once (dense). MoE
/// falls back to two singles — expert sets differ per position, there
/// is nothing to fuse.
fn ffn_forward_pair(
    ffn: &FfnKind,
    x1: &[f32],
    x2: &[f32],
    pool: Option<&Pool>,
    experts_allowed: Option<&[bool]>,
) -> (Vec<f32>, Vec<f32>) {
    let d = match ffn {
        // A tube layer has nothing to fuse across the pair — the tubes
        // are separate matrices; two singles are the honest path.
        FfnKind::Dense(d) if !d.segs.is_empty() => {
            return (
                tube_ffn(d, x1, 1, pool, None),
                tube_ffn(d, x2, 1, pool, None),
            );
        }
        FfnKind::Dense(d) => d,
        FfnKind::Moe(m) => {
            return (
                moe_ffn(m, x1, pool, experts_allowed),
                moe_ffn(m, x2, pool, experts_allowed),
            );
        }
        FfnKind::DenseMoe(_) => unreachable!("DenseMoe dispatches via dense_moe_ffn"),
    };
    let inter = d.gate_proj.rows();
    FFN_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let [g1, g2, u1, u2] = &mut *s;
        g1.resize(inter, 0.0);
        g2.resize(inter, 0.0);
        u1.resize(inter, 0.0);
        u2.resize(inter, 0.0);
        // Multi-matrix pair job: gate+up under one pool dispatch
        // (o1s = lane-1 outputs across tensors, o2s = lane-2).
        QTensor::matvec2_many(
            [&d.gate_proj, &d.up_proj],
            x1,
            x2,
            [g1.as_mut_slice(), u1.as_mut_slice()],
            [g2.as_mut_slice(), u2.as_mut_slice()],
            pool,
        );
        for i in 0..inter {
            g1[i] = d.act.combine(g1[i], u1[i]);
            g2[i] = d.act.combine(g2[i], u2[i]);
        }
        let mut o1 = attention::take_buf(d.down_proj.rows());
        let mut o2 = attention::take_buf(d.down_proj.rows());
        d.down_proj.matvec2(g1, g2, &mut o1, &mut o2, pool);
        (o1, o2)
    })
}

#[cfg(test)]
mod tests {

    /// The 0.7.6 prefill-chunk rule: a plain dense stack wholly on a
    /// discrete card reads the prompt in wide chunks on x86; every other
    /// case keeps the width it had (the GDN-hybrid, MoE and DeepSeek paths
    /// were tuned on hardware not measured for this change).
    #[test]
    fn prefill_chunk_rule_widens_only_dense_on_discrete() {
        use super::{
            prefill_chunk_rule, ChunkHost, ChunkStackFacts, DISCRETE_DENSE_PREFILL_CHUNK,
        };
        let dense_card = ChunkStackFacts {
            plain_dense: true,
            discrete: true,
            gpu_on: true,
            ..Default::default()
        };
        assert!(dense_card.dense_on_discrete());
        // The bug: a dense Llama on a Vulkan RTX 3090 got 48.
        assert_eq!(
            prefill_chunk_rule(None, ChunkHost::Other, dense_card.dense_on_discrete()),
            DISCRETE_DENSE_PREFILL_CHUNK
        );
        assert!(DISCRETE_DENSE_PREFILL_CHUNK > 48);
        for (label, facts) in [
            ("GDN hybrid / MoE / DeepSeek stack", ChunkStackFacts { plain_dense: false, ..dense_card }),
            ("integrated GPU", ChunkStackFacts { discrete: false, ..dense_card }),
            ("CPU only", ChunkStackFacts { gpu_on: false, discrete: false, ..dense_card }),
            ("capacity split", ChunkStackFacts { capacity_split: true, ..dense_card }),
            ("multi-GPU plan", ChunkStackFacts { multi_gpu: true, ..dense_card }),
            ("O(1) layers", ChunkStackFacts { o1: true, ..dense_card }),
        ] {
            assert!(!facts.dense_on_discrete(), "{label}");
            assert_eq!(
                prefill_chunk_rule(None, ChunkHost::Other, facts.dense_on_discrete()),
                48,
                "{label} keeps the historical x86 chunk"
            );
        }
        // Other hosts are untouched whatever the model.
        for dense in [false, true] {
            assert_eq!(prefill_chunk_rule(None, ChunkHost::Macos, dense), 512);
            assert_eq!(prefill_chunk_rule(None, ChunkHost::Aarch64, dense), 256);
        }
        // CMF_PREFILL_CHUNK still wins everywhere (and is clamped to ≥ 1).
        for host in [ChunkHost::Macos, ChunkHost::Aarch64, ChunkHost::Other] {
            for dense in [false, true] {
                assert_eq!(prefill_chunk_rule(Some(48), host, dense), 48);
                assert_eq!(prefill_chunk_rule(Some(0), host, dense), 1);
            }
        }
    }

    #[test]
    fn nll_graph_policy_scopes_only_the_fused_head() {
        for (label, unmasked, prefer_graph, native_metal, want_graph, want_head) in [
            // A Vulkan/Wgpu hidden-only graph remains the quality route.
            ("vulkan graph", true, true, false, true, false),
            // Native Metal adds the strict fused graph-head contract.
            ("native Metal graph", true, true, true, true, true),
            // Masked NLL and the explicit non-graph fallback remain unchanged.
            ("masked", false, true, false, false, false),
            ("graph disabled", true, false, true, false, false),
        ] {
            let (graph_quality, graph_head_required) =
                super::nll_graph_policy(unmasked, prefer_graph, native_metal);
            assert_eq!(graph_quality, want_graph, "{label}: graph quality");
            assert_eq!(graph_head_required, want_head, "{label}: fused head");
        }
    }

    #[test]
    fn mtp_prefill_pair_boundaries_skip_only_final_prompt_row() {
        assert_eq!(mtp_prefill_pair_count(0, 128, 256), 128);
        assert_eq!(mtp_prefill_pair_count(128, 256, 256), 127);
        assert_eq!(mtp_prefill_pair_count(0, 256, 256), 255);
        assert_eq!(mtp_prefill_pair_count(256, 256, 256), 0);
        assert_eq!(mtp_prefill_pair_count(300, 320, 256), 0);
    }

    #[test]
    fn cancel_flag_stops_generation() {
        let mut p = create_test_pipeline(16, 32, 2, 2, 8, 2, 32);
        // Set before the call: the prefill loops honour it, the run
        // returns immediately with the cancelled reason and no tokens.
        p.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        let r = p.generate_from_ids(&[1, 2, 3], 8, None, None).unwrap();
        assert_eq!(r.finish_reason, "cancelled");
        assert!(
            r.token_ids.is_empty(),
            "no tokens after cancel: {:?}",
            r.token_ids
        );
        assert_eq!(p.kv_cache.seq_len(), 0);
        assert!(p.kv_history.is_empty());
        assert!(!p.graph_want_logits);
        assert!(p.graph_logits.is_none());
        // Flag auto-cleared: the next call generates normally.
        let r2 = p.generate_from_ids(&[1, 2, 3], 4, None, None).unwrap();
        assert_ne!(r2.finish_reason, "cancelled");
    }
    use super::*;

    /// sparse_ffn_quant must equal a dense FFN where inactive neurons are
    /// zeroed (mask × mmap correctness). On F32 tensors this is EXACT —
    /// it validates the row_dot / add_col_scaled / scatter indexing, the
    /// bug-prone part. The q8 branches reuse the golden-tested linear
    /// The per-token sparse path reads a transposed `down`; it must
    /// agree with the arm that computes everything and zeroes the
    /// losers, or the speed measurement is measuring a different model.
    #[test]
    fn dynamic_ffn_equals_the_zeroing_arm() {
        let (hidden, inter) = (8usize, 32usize);
        let synth = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 29 + salt * 13 + 7) % 89) as f32 / 89.0 - 0.5) * 0.6)
                .collect()
        };
        let down = synth(hidden * inter, 3);
        let mut down_t = vec![0.0f32; inter * hidden];
        for r in 0..hidden {
            for c in 0..inter {
                down_t[c * hidden + r] = down[r * inter + c];
            }
        }
        let d = DenseFfn {
            gate_proj: QTensor::from_f32(synth(inter * hidden, 1), inter, hidden),
            up_proj: QTensor::from_f32(synth(inter * hidden, 2), inter, hidden),
            down_proj: QTensor::from_f32(down.clone(), hidden, inter),
            act: Act::Silu,
            down_t: Some(QTensor::from_f32(down_t, inter, hidden)),
            segs: Vec::new(),
        };
        let x = synth(hidden, 11);
        let k = 12usize;
        let got = dense_ffn_dynamic(&d, &x, None, k).expect("down_t present");
        // Reference: full compute, keep the k loudest |silu(gate)|.
        let mut g = vec![0.0f32; inter];
        d.gate_proj.matvec(&x, &mut g, None);
        let mut u = vec![0.0f32; inter];
        d.up_proj.matvec(&x, &mut u, None);
        for v in g.iter_mut() {
            *v = inference::silu(*v);
        }
        keep_top_k(&mut g, k);
        for i in 0..inter {
            g[i] *= u[i];
        }
        let mut want = vec![0.0f32; hidden];
        d.down_proj.matvec(&g, &mut want, None);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-5, "dynamic {b} vs reference {a}");
        }
    }

    /// A tube layer is the same layer, re-cut. With every tube open the
    /// answer must equal the dense FFN over the concatenated neurons
    /// (the permutation is an identity on the layer's function); with a
    /// tube closed it must equal the dense FFN with those neurons
    /// zeroed — the mask semantics, now paid for in bytes not read.
    #[test]
    fn tube_ffn_open_equals_dense_and_closed_equals_masked() {
        let (hidden, core, tube) = (8usize, 12usize, 8usize);
        let inter = core + tube;
        let synth = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 41 + salt * 17 + 5) % 97) as f32 / 97.0 - 0.5) * 0.5)
                .collect()
        };
        let (g_all, u_all) = (synth(inter * hidden, 1), synth(inter * hidden, 2));
        let d_all = synth(hidden * inter, 3);
        // The dense layer, and the same weights cut into core + tube.
        let dense = DenseFfn {
            gate_proj: QTensor::from_f32(g_all.clone(), inter, hidden),
            up_proj: QTensor::from_f32(u_all.clone(), inter, hidden),
            down_proj: QTensor::from_f32(d_all.clone(), hidden, inter),
            act: Act::Silu,
            down_t: None,
            segs: Vec::new(),
        };
        let rows =
            |v: &[f32], a: usize, b: usize| -> Vec<f32> { v[a * hidden..b * hidden].to_vec() };
        let cols = |v: &[f32], a: usize, b: usize| -> Vec<f32> {
            let mut o = Vec::with_capacity(hidden * (b - a));
            for r in 0..hidden {
                o.extend_from_slice(&v[r * inter + a..r * inter + b]);
            }
            o
        };
        let tubed = DenseFfn {
            down_t: None,
            gate_proj: QTensor::from_f32(rows(&g_all, 0, core), core, hidden),
            up_proj: QTensor::from_f32(rows(&u_all, 0, core), core, hidden),
            down_proj: QTensor::from_f32(cols(&d_all, 0, core), hidden, core),
            act: Act::Silu,
            segs: vec![FfnSeg {
                gate: QTensor::from_f32(rows(&g_all, core, inter), tube, hidden),
                up: QTensor::from_f32(rows(&u_all, core, inter), tube, hidden),
                down: QTensor::from_f32(cols(&d_all, core, inter), hidden, tube),
                start: core,
                width: tube,
            }],
        };
        let x = synth(hidden, 7);
        let want = dense_ffn(&dense, &x, None);
        let got = tube_ffn(&tubed, &x, 1, None, None);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-5, "open tube: {a} vs {b}");
        }
        // Closed tube: bits on for the core, off for the tube.
        let mut bits = vec![0u8; inter.div_ceil(8)];
        for n in 0..core {
            bits[n / 8] |= 1 << (n % 8);
        }
        let closed = tube_ffn(&tubed, &x, 1, None, Some(&bits));
        let masked = dense_ffn_masked(&dense, &x, None, &bits);
        for (a, b) in masked.iter().zip(&closed) {
            assert!((a - b).abs() < 1e-5, "closed tube: {a} vs {b}");
        }
        // The batched arm must agree with the single-position one.
        let batch = tube_ffn(&tubed, &x, 1, None, Some(&bits));
        for (a, b) in closed.iter().zip(&batch) {
            assert_eq!(a, b, "batch arm disagrees with decode arm");
        }
    }

    /// scale, structurally identical to the matvec kernels.
    #[test]
    fn sparse_ffn_quant_equals_dense_with_inactive_zeroed() {
        let (hidden, inter) = (16usize, 40usize);
        let synth = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 37 + salt * 11 + 3) % 101) as f32 / 101.0 - 0.5) * 0.4)
                .collect()
        };
        let d = DenseFfn {
            gate_proj: QTensor::from_f32(synth(inter * hidden, 1), inter, hidden),
            up_proj: QTensor::from_f32(synth(inter * hidden, 2), inter, hidden),
            down_proj: QTensor::from_f32(synth(hidden * inter, 3), hidden, inter),
            act: Act::Silu,
            down_t: None,
            segs: Vec::new(),
        };
        let x = synth(hidden, 9);
        // Active = every 3rd neuron.
        let active: Vec<u16> = (0..inter as u16).filter(|i| i % 3 == 0).collect();

        let sparse = sparse_ffn_quant(&d, &x, &active, hidden, None);

        // Reference: full dense FFN but g[i]=0 for inactive neurons.
        let mut g = vec![0.0f32; inter];
        d.gate_proj.matvec(&x, &mut g, None);
        let mut u = vec![0.0f32; inter];
        d.up_proj.matvec(&x, &mut u, None);
        let act_set: std::collections::HashSet<u16> = active.iter().copied().collect();
        for i in 0..inter {
            g[i] = if act_set.contains(&(i as u16)) {
                inference::silu(g[i]) * u[i]
            } else {
                0.0
            };
        }
        let mut reference = vec![0.0f32; hidden];
        d.down_proj.matvec(&g, &mut reference, None);

        let max_d = sparse
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-5, "sparse != dense-zeroed: max|Δ| = {max_d}");
    }

    /// Attach a synthetic MTP head (same structure as a main layer).
    fn attach_test_mtp(p: &mut Pipeline) {
        let (h, inter, heads, kv, hd) = (
            p.hidden_size,
            p.intermediate_size,
            p.num_heads,
            p.num_kv_heads,
            p.head_dim,
        );
        let synth = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 29 + salt * 23 + 5) % 101) as f32 / 101.0 - 0.5) * 0.2)
                .collect()
        };
        let qt = |rows: usize, cols: usize, salt: usize| -> QTensor {
            QTensor::from_f32(synth(rows * cols, salt), rows, cols)
        };
        p.mtp = Some(MtpModule {
            enorm: vec![1.0; h],
            hnorm: vec![1.0; h],
            eh_proj: qt(h, 2 * h, 301),
            layer: LayerWeights {
                input_norm: vec![1.0; h],
                post_norm: vec![1.0; h],
                attn_out_norm: None,
                ffn_out_norm: None,
                layer_scale: None,
                ffn: FfnKind::Dense(DenseFfn {
                    gate_proj: qt(inter, h, 315),
                    up_proj: qt(inter, h, 316),
                    down_proj: qt(h, inter, 317),
                    act: Act::Silu,
                    down_t: None,
                    segs: Vec::new(),
                }),
                attn: AttnKind::Full {
                    bias: None,
                    wq: qt(heads * hd, h, 311),
                    wk: qt(kv * hd, h, 312),
                    wv: qt(kv * hd, h, 313),
                    wo: qt(h, heads * hd, 314),
                    q_norm: None,
                    k_norm: None,
                    output_gate: false,
                    softplus_gate: None,
                },
            },
            final_norm: vec![1.0; h],
            kv: crate::kv_cache::LayerKvCache::new(kv, hd),
        });
    }

    #[test]
    fn speculative_equals_vanilla_greedy() {
        // Speculative decode and the wgpu token graph are mutually
        // exclusive; a leaked CMF_GPU=wgpu from a parallel gpu test
        // would silently disable drafting. Pin the graph off.
        unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", "0") };
        let run = |spec: bool| {
            let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
            p.sampler_config.temperature = 0.0;
            attach_test_mtp(&mut p);
            p.speculative = spec;
            let r = p.generate("abcdef", 12, None, None).unwrap();
            (r.token_ids, r.mtp_drafted, r.mtp_accepted)
        };
        let (vanilla, d0, _) = run(false);
        let (spec, d1, a1) = run(true);
        assert_eq!(d0, 0, "vanilla path must not draft");
        assert!(d1 > 0, "speculative path must draft");
        assert_eq!(
            vanilla, spec,
            "speculative must reproduce the exact greedy sequence (accepted {a1}/{d1})"
        );
    }

    #[test]
    fn speculative_accepts_constant_oracle() {
        // See speculative_equals_vanilla_greedy: pin the wgpu graph off.
        unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", "0") };
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        // Constant lm_head → every logit equal → both the main model and
        // the draft head argmax to token 0: acceptance must be 100%.
        p.weights.lm_head = QTensor::from_f32(vec![0.01; 64 * 8], 64, 8);
        attach_test_mtp(&mut p);
        p.speculative = true;
        let r = p.generate("abcd", 10, None, None).unwrap();
        assert!(r.mtp_drafted > 0);
        assert_eq!(
            r.mtp_accepted, r.mtp_drafted,
            "constant logits → every draft accepted"
        );
        // Ties resolve to the same token in both the main and draft
        // heads — the sequence is one repeated token.
        assert!(r.token_ids.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn empty_prompt_is_an_error_not_a_panic() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 260);
        let r = p.generate("", 4, None, None);
        assert!(r.is_err(), "empty prompt must be a clean error");
    }

    #[test]
    fn every_token_enters_kv_exactly_once() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
        // Greedy so no RNG variance; byte tokenizer → 3 prompt tokens.
        p.sampler_config.temperature = 0.0;
        let r = p.generate("abc", 2, None, None).unwrap();
        assert_eq!(r.prompt_tokens, 3);
        // prompt(3) + first sampled token forwarded before second logits:
        // step0 samples from prefill hidden (no extra forward), then
        // forwards t1 → cache 4; step1 samples, loop ends (max_tokens).
        assert_eq!(
            p.kv_cache.seq_len(),
            3 + r.tokens_generated - 1,
            "each token must be cached exactly once (v1 cached the last prompt token twice)"
        );
    }

    #[test]
    fn generation_is_reproducible_with_seed() {
        let run = || {
            let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
            p.generate("hello", 8, None, None).unwrap().token_ids
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn resetting_sampler_restarts_the_seeded_stream() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
        let config = SamplerConfig {
            seed: Some(1234),
            ..SamplerConfig::default()
        };
        p.set_sampler_config(config.clone());
        let first = p.generate("hello", 8, None, None).unwrap().token_ids;
        p.set_sampler_config(config);
        let second = p.generate("hello", 8, None, None).unwrap().token_ids;
        assert_eq!(first, second);
    }

    #[test]
    fn eviction_bounds_the_cache() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 260);
        p.kv_cache.max_seq_len = 6;
        p.sampler_config.temperature = 0.0;
        let _ = p.generate("abcd", 12, None, None).unwrap();
        assert!(
            p.kv_cache.seq_len() <= 6 + 1,
            "cache must stay bounded by max_seq_len (got {})",
            p.kv_cache.seq_len()
        );
    }

    #[test]
    fn confidence_matches_tokens_and_is_a_probability() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        let r = p.generate("abcd", 10, None, None).unwrap();
        assert_eq!(
            r.token_confidence.len(),
            r.token_ids.len(),
            "one confidence per emitted token"
        );
        for &c in &r.token_confidence {
            assert!((0.0..=1.0).contains(&c), "confidence out of [0,1]: {c}");
        }
        // top1_prob is a valid softmax probability.
        let logits = [1.0f32, 3.0, 0.5, 3.0];
        let p0 = top1_prob_t(&logits, 1, 1.0);
        let p1 = top1_prob_t(&logits, 3, 1.0);
        assert!((p0 - p1).abs() < 1e-6, "equal logits → equal prob");
        assert!(p0 > 0.0 && p0 < 1.0);
        // Calibration temperature > 1 softens an over-confident peak.
        let sharp = top1_prob_t(&logits, 1, 1.0);
        let soft = top1_prob_t(&logits, 1, 2.0);
        assert!(soft < sharp, "higher temperature lowers peak confidence");
    }

    #[test]
    fn trace_is_opt_in_and_parallels_the_output() {
        // Off by default: the runtime is silent unless observation asked.
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        let r = p.generate("abcd", 10, None, None).unwrap();
        assert!(r.traces.is_empty(), "trace must be empty unless enabled");

        // On: exactly one row per emitted token, aligned with the output.
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        p.set_trace(true);
        let r = p.generate("abcd", 10, None, None).unwrap();
        assert_eq!(r.traces.len(), r.token_ids.len(), "one trace row per token");
        for (i, tr) in r.traces.iter().enumerate() {
            assert_eq!(tr.t, i, "trace index is sequential");
            assert_eq!(tr.token_id, r.token_ids[i], "trace token_id matches output");
            assert_eq!(
                tr.confidence, r.token_confidence[i],
                "trace confidence matches the confidence channel"
            );
            // No dynamic router in this pipeline → no skill, no coherence.
            assert!(tr.active_skill.is_none() && tr.recon.is_none() && !tr.switched);
        }
    }

    #[test]
    fn explain_prefill_logits_match_greedy_first_token() {
        // `cortiq explain` shows the next-token distribution from
        // prefill_next_logits; its argmax must equal what greedy generate
        // actually emits first — otherwise explain would lie.
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        let ids = p.tokenizer.encode("abcd");
        let logits = p.prefill_next_logits(&ids, None);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        let r = p.generate("abcd", 1, None, None).unwrap();
        assert_eq!(
            argmax, r.token_ids[0],
            "explain preview must match greedy emit"
        );
    }

    #[test]
    fn laguna_shared_expert_is_unconditionally_added() {
        let matrix = |values: Vec<f32>| QTensor::from_f32(values, 2, 2);
        let identity = || matrix(vec![1.0, 0.0, 0.0, 1.0]);
        let zero_dense = || DenseFfn {
            gate_proj: matrix(vec![0.0; 4]),
            up_proj: matrix(vec![0.0; 4]),
            down_proj: matrix(vec![0.0; 4]),
            act: Act::Silu,
            down_t: None,
            segs: Vec::new(),
        };
        let shared = DenseFfn {
            gate_proj: identity(),
            up_proj: identity(),
            down_proj: identity(),
            act: Act::Silu,
            down_t: None,
            segs: Vec::new(),
        };
        let x = [1.0, 2.0];
        let expected = dense_ffn(&shared, &x, None);
        let moe = MoeFfn {
            router: QTensor::from_f32(vec![0.0, 0.0], 1, 2),
            experts: vec![zero_dense()],
            top_k: 1,
            norm_topk_prob: true,
            router_sigmoid: true,
            expert_bias: None,
            routed_scaling: 1.0,
            route_tau: None,
            shared: Some((shared, None)),
            stats: std::cell::RefCell::new(Vec::new()),
            act_sq: std::cell::RefCell::new(Vec::new()),
            act_rows: std::cell::RefCell::new(Vec::new()),
            mask: None,
            per_expert_scale: None,
            router_input_norm: false,
            resonance: None,
        };
        let actual = moe_ffn_cpu(&moe, &x, &[0], &[0.0], 1.0, None);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn o1_batch_transition_publishes_one_epoch_before_serial_handoff() {
        const B: usize = 19;
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
        p.set_o1(Some(crate::nystrom::O1Cfg {
            layers: crate::nystrom::O1Layers::All,
            m: 4,
            w: 8,
            sink: 2,
            rect: crate::nystrom::O1Rect::Aggregate,
        }));
        p.o1_begin_with_prefix(Some(B));
        let ids: Vec<u32> = (0..B as u32).collect();
        let _ = p.prefill_batch_span(PrefillIn::Ids(&ids), 0, None, 0, p.num_layers);

        assert_eq!(p.o1_epoch, 1, "all layers publish one completed transition");
        assert!(p.kv_cache.layers.iter().all(|l| l.o1_sealed()));
        let next = p.embed_single(B as u32);
        let _ = p.forward_layers(&next, B, None);
        assert_eq!(p.o1_epoch, 1, "sealed handoff must not republish the epoch");
    }

    #[test]
    fn o1_pair_transition_commits_scratch_before_epoch_publication() {
        const B: usize = 19;
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
        // Keep a real recurrent layer ahead of the Full O(1) layer so the
        // pair test observes the GDN lane-2 scratch swap at the same
        // boundary, rather than only exercising an artificial scratch vec.
        let gdn_cfg = crate::linear_core::GdnCfg {
            num_v_heads: 2,
            num_k_heads: 1,
            key_head_dim: 2,
            value_head_dim: 4,
            conv_kernel: 3,
            hidden_size: 8,
            rms_eps: 1e-6,
            output_gate_sigmoid: false,
        };
        let synth = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt * 7) % 97) as f32 / 97.0 - 0.5) * 0.4)
                .collect()
        };
        let qt = |rows: usize, cols: usize, salt: usize| {
            crate::qtensor::QTensor::from_f32(synth(rows * cols, salt), rows, cols)
        };
        let c_dim = gdn_cfg.conv_dim();
        let vd = gdn_cfg.num_v_heads * gdn_cfg.value_head_dim;
        p.weights.layers[0].attn = AttnKind::LinearGdn(crate::linear_core::GdnWeights {
            in_proj_qkv: qt(c_dim, 8, 1),
            in_proj_z: qt(vd, 8, 2),
            in_proj_a: qt(gdn_cfg.num_v_heads, 8, 3),
            in_proj_b: qt(gdn_cfg.num_v_heads, 8, 4),
            conv1d: synth(c_dim * gdn_cfg.conv_kernel, 5),
            a_log: vec![0.2, 0.5],
            dt_bias: synth(gdn_cfg.num_v_heads, 6),
            norm: vec![1.0; gdn_cfg.value_head_dim],
            out_proj: qt(8, vd, 7),
        });
        p.gdn_cfg = Some(gdn_cfg);
        p.set_o1(Some(crate::nystrom::O1Cfg {
            layers: crate::nystrom::O1Layers::All,
            m: 4,
            w: 8,
            sink: 2,
            rect: crate::nystrom::O1Rect::Aggregate,
        }));
        p.o1_begin_with_prefix(Some(B));
        for pos in 0..B - 2 {
            let emb = p.embed_single(pos as u32);
            let _ = p.forward_layers(&emb, pos, None);
        }
        let lane1_state = p.kv_cache.layers[0].linear_state.clone();

        let e1 = p.embed_single((B - 2) as u32);
        let e2 = p.embed_single((B - 1) as u32);
        let _ = p.forward_pair(&e1, &e2, B - 2);

        assert_eq!(p.o1_epoch, 1, "pair crossing B publishes one epoch");
        assert!(
            p.kv_cache
                .layers
                .iter()
                .enumerate()
                .all(|(li, l)| !p.o1_flags[li] || l.o1_sealed())
        );
        assert!(!p.kv_cache.layers[0].linear_state.is_empty());
        assert_ne!(
            p.kv_cache.layers[0].linear_state, lane1_state,
            "real pair must commit GDN lane 2 before returning"
        );
        assert!(p.kv_cache.layers[0].linear_scratch.is_empty());
        let next = p.embed_single(B as u32);
        let _ = p.forward_layers(&next, B, None);
        assert_eq!(p.o1_epoch, 1, "serial continuation must reuse the epoch");
    }

    #[test]
    fn o1_error_observation_stays_terminal_until_reset() {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.set_o1(Some(crate::nystrom::O1Cfg {
            layers: crate::nystrom::O1Layers::All,
            m: 4,
            w: 8,
            sink: 2,
            rect: crate::nystrom::O1Rect::Aggregate,
        }));
        p.o1_begin();
        p.kv_cache.layers[0].o1_abort("synthetic transition failure".into());

        assert!(p.o1_seal_checked().is_err());
        assert!(
            p.o1_seal_checked().is_err(),
            "retry must see the sticky error"
        );
        let k = vec![0.2f32; 4];
        let v = vec![0.3f32; 4];
        p.kv_cache.layers[0].append(&k, &v, &[]);
        assert_eq!(p.kv_cache.layers[0].seq_len, 0);

        p.reset_session();
        p.o1_begin();
        p.kv_cache.layers[0].append(&k, &v, &[]);
        assert_eq!(p.kv_cache.layers[0].seq_len, 1);
    }

    #[test]
    fn nll_graph_failure_is_terminal_and_request_is_reusable() {
        let ids = vec![1u32, 2, 3, 4, 5, 6];
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.graph_logits = Some(vec![123.0]);
        p.graph_want_logits = true;
        p.graph_failed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        p.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        let err = p.nll_ids_from(&ids, 0).expect_err("prior graph failure");
        assert!(err.contains("before NLL"));
        assert!(p.graph_logits.is_none());
        assert!(!p.graph_want_logits);
        assert!(!p.graph_failed.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!p.cancel.load(std::sync::atomic::Ordering::Relaxed));

        let mut fresh = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        let expected = fresh.nll_ids_from(&ids, 0).expect("fresh NLL");
        let actual = p.nll_ids_from(&ids, 0).expect("reused NLL");
        assert_eq!(actual.1, expected.1);
        assert!((actual.0 - expected.0).abs() < 1e-9);
    }

    #[test]
    fn nll_forward_failure_discards_partial_score_and_clears_sidechannels() {
        let ids = vec![1u32, 2, 3, 4, 5, 6];
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.nll_test_fail_at = Some(1);
        let err = p
            .nll_ids_from(&ids, 0)
            .expect_err("one-shot forward failure");
        assert!(err.contains("forward") || err.contains("score row"));
        assert!(!p.graph_failed.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!p.graph_want_logits);
        assert!(p.graph_logits.is_none());
        assert!(p.kv_history.is_empty());

        let mut fresh = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        let expected = fresh.nll_ids_from(&ids, 0).expect("fresh NLL");
        let actual = p.nll_ids_from(&ids, 0).expect("reused NLL");
        assert_eq!(actual.1, expected.1);
        assert!((actual.0 - expected.0).abs() < 1e-9);
    }

    #[test]
    fn nll_serial_failure_before_first_row_is_reported() {
        let ids = vec![1u32, 2, 3, 4];
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.nll_test_force_serial = true;
        p.nll_test_fail_at = Some(0);
        let err = p.nll_ids_from(&ids, 0).expect_err("serial forward failure");
        assert!(err.contains("serial forward"));
        assert!(p.kv_history.is_empty());
        assert!(!p.graph_failed.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!p.cancel.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn ffn_probe_failure_discards_recorder_and_state() {
        let ids = vec![1u32, 2, 3, 4];
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.nll_test_fail_at = Some(0);
        let err = p
            .probe_ffn_mass_batch(&ids)
            .expect_err("probe forward failure");
        assert!(err.contains("NLL"));
        assert!(FFN_PROBE.with(|probe| probe.borrow().is_none()));
        assert!(p.kv_history.is_empty());
        assert!(!p.graph_failed.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn nll_test_controls_are_pipeline_scoped() {
        let ids = vec![1u32, 2, 3, 4];
        let mut failing = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        let mut unaffected = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        failing.nll_test_force_serial = true;
        failing.nll_test_fail_at = Some(0);

        assert!(!failing.can_prefill_batched());
        assert!(unaffected.can_prefill_batched());
        let expected = unaffected
            .nll_ids_from(&ids, 0)
            .expect("unaffected pipeline remains usable");
        let err = failing
            .nll_ids_from(&ids, 0)
            .expect_err("failure injection belongs to failing pipeline");
        assert!(err.contains("serial forward"));
        assert!(failing.nll_test_fail_at.is_none());
        assert!(unaffected.can_prefill_batched());
        let actual = unaffected
            .nll_ids_from(&ids, 0)
            .expect("unaffected pipeline remains reusable");
        assert_eq!(actual.1, expected.1);
        assert!((actual.0 - expected.0).abs() < 1e-9);
    }

    #[test]
    fn forward_ids_failure_channel_is_terminal_and_reusable() {
        let ids = vec![1u32, 2, 3, 4, 5, 6];
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 1, 64);
        p.graph_logits = Some(vec![123.0]);
        p.graph_want_logits = true;
        p.graph_failed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        p.cancel.store(true, std::sync::atomic::Ordering::Relaxed);

        let err = p
            .forward_ids(&ids, None)
            .expect_err("a failed forward must not become a valid head result");
        assert!(err.contains("forward_ids setup"));
        assert!(p.graph_logits.is_none());
        assert!(!p.graph_want_logits);
        assert!(!p.graph_failed.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!p.cancel.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(p.kv_cache.seq_len(), 0);

        let expected = create_test_pipeline(8, 16, 2, 1, 4, 1, 64)
            .forward_ids(&ids, None)
            .expect("fresh forward_ids");
        let actual = p
            .forward_ids(&ids, None)
            .expect("pipeline remains reusable after a failed forward");
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        assert_eq!(p.kv_cache.seq_len(), ids.len());
    }
}
