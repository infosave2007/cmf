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
    rng: SplitMix64,
    sampler_scratch: SamplerScratch,
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
    /// Bumped at every o1 seal — the GPU state mirror re-uploads when it
    /// sees a new epoch (each generate seals fresh CPU state).
    o1_epoch: u64,
    /// Per-layer o1 flags derived from `o1_cfg` (Full layers only).
    o1_flags: Vec<bool>,
    /// Emit a structured per-token trace (B4 telemetry channel). Off by
    /// default — the runtime is silent unless observation is requested.
    trace: bool,
    /// Confidence-calibration temperature (B1): reported Born mass is
    /// softmax(logits / calib_temp). 1.0 = raw. Set from header.calibration.
    calib_temp: f32,
    /// Process-unique id keying this pipeline's device KV mirrors.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    graph_kv_id: u64,
    /// Decode asks the token graph to also run final-norm + lm_head on
    /// the device (drops the separate per-op lm_head round trip).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    graph_want_logits: bool,
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
    /// Final-logit soft-capping C: logits = C·tanh(logits/C) (Gemma-4).
    pub final_softcap: Option<f32>,
    /// Gemma-2 attention-logit soft-capping (0.0 = off).
    pub attn_softcap: f32,
    /// Compute per-token Born confidence (a full-vocab softmax each
    /// token). On by default; `bench --core` turns it off to match
    /// llama-bench's core timing.
    confidence_on: bool,
}

#[cfg(target_os = "macos")]
impl Drop for Pipeline {
    fn drop(&mut self) {
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
    /// that was actually emitted (Born mass on the chosen state). High =
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
    /// Born mass on the emitted token (softmax prob) — how sure the model was.
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

/// Calibrated softmax probability of `id` under `logits` (the Born mass on
/// the emitted token) — the confidence signal, cheap from logits already
/// computed for sampling. `temp` is the calibration temperature (B1):
/// softmax(logits / temp); 1.0 = raw.
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
        prefill_batched() && !self.weights.layers.is_empty()
    }
}

/// Prefill chunk (positions per batched pass). On macOS the AMX GEMM
/// path wants tall panels — M=48 starves the matrix units (ggml uses
/// ubatch 512); elsewhere the historical 48 stays. CMF_PREFILL_CHUNK
/// overrides. Pub: the network split MUST chunk identically to the
/// local path — panel width reorders float accumulation, so a different
/// chunk is a different (equally valid) generation.
pub fn prefill_chunk() -> usize {
    if let Some(n) = std::env::var("CMF_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    if cfg!(target_os = "macos") {
        512
    } else if cfg!(target_arch = "aarch64") {
        // Mobile: big enough to feed the batched attend (gate b ≥ 32)
        // and the blocked SDOT GEMM without the memory of 512.
        256
    } else {
        48
    }
}

/// Callback for streaming tokens. Return `false` to cancel.
pub type TokenCallback = Box<dyn FnMut(&str) -> bool + Send>;

impl Pipeline {
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
        if !crate::gpu::enabled_here()
            || !crate::gpu::q1_force()
            || std::env::var("CMF_GPU_BLOCK")
                .map(|v| v == "0")
                .unwrap_or(false)
        {
            return false;
        }
        self.weights
            .layers
            .iter()
            .any(|lw| matches!(&lw.attn, AttnKind::LinearGdn(w) if w.in_proj_qkv.is_q1()))
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
        let graph_on = std::env::var("CMF_GPU_WGPU_GRAPH")
            .map(|v| v != "0")
            .unwrap_or_else(|_| {
                // Default ON for wgpu on DISCRETE adapters (4090:
                // decode 76 -> 137 tok/s); integrated/mobile GPUs keep
                // the per-op probe path — see gpu::wgpu_graph_default.
                crate::gpu::wgpu_graph_default()
            });
        if !graph_on || !crate::gpu::enabled_here() {
            return false;
        }
        self.weights
            .layers
            .iter()
            .any(|lw| matches!(&lw.attn, AttnKind::LinearGdn(_)))
    }

    #[cfg(target_os = "macos")]
    fn q1_graph_gpu(
        &mut self,
        start: usize,
        upto: Option<usize>,
        position: usize,
        h: &mut [f32],
    ) -> usize {
        use crate::gpu::{AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, GraphDims, MetalFfn, TokenGraph};
        if self.attn_softcap > 0.0 // capped scores: no graph kernel — CPU path
            || !crate::gpu::enabled_here()
            || !crate::gpu::q1_force()
            || std::env::var("CMF_GPU_BLOCK")
                .map(|v| v == "0")
                .unwrap_or(false)
        {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!(
                    "block-graph: front gate (softcap={} enabled_here={} q1_force={})",
                    self.attn_softcap > 0.0,
                    crate::gpu::enabled_here(),
                    crate::gpu::q1_force(),
                );
            }
            return start;
        }
        // The graph encodes SiLU FFN, 1/√hd attention scores and
        // full-context attend with no branch norms — Gemma-style archs
        // (sliding window, scale override, sandwich norms, GeLU) fall
        // back to the CPU path.
        if self.swa.is_some()
            || self.global_attn.is_some()
            || self.attention_heads_per_layer.is_some()
            || self.attn_v_norm
            || (self.attn_scale - 1.0 / (self.head_dim as f32).sqrt()).abs() > 1e-9
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
                FfnKind::Dense(d) => {
                    let (Some(g), Some(u), Some(dn)) = (
                        d.gate_proj.q1_parts(),
                        d.up_proj.q1_parts(),
                        d.down_proj.q1_parts(),
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
                        w.in_proj_qkv.q1_parts(),
                        w.in_proj_z.q1_parts(),
                        w.in_proj_a.f32_parts(),
                        w.in_proj_b.f32_parts(),
                        w.out_proj.q1_parts(),
                    );
                    let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = parts else {
                        if block_diag {
                            eprintln!(
                                "block-graph: L{scan} GDN parts refused (qkv={} z={} a_f32={} b_f32={} out={})",
                                w.in_proj_qkv.q1_parts().is_some(),
                                w.in_proj_z.q1_parts().is_some(),
                                w.in_proj_a.f32_parts().is_some(),
                                w.in_proj_b.f32_parts().is_some(),
                                w.out_proj.q1_parts().is_some(),
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
                } if !self.kv_cache.layers[scan].o1_sealed() => {
                    let parts = (wq.q1_parts(), wk.q1_parts(), wv.q1_parts(), wo.q1_parts());
                    let (Some(pq), Some(pk), Some(pv), Some(po)) = parts else {
                        break;
                    };
                    if let QTensor::Mapped { model, .. } = wq {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let cache = &self.kv_cache.layers[scan];
                    let full_gpu = attend_contract
                        && cache.mode == crate::kv_cache::KvMode::F32
                        && cache.o1.is_none()
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
            return start;
        };
        if plan.is_empty() {
            if std::env::var("CMF_GRAPH_DBG").is_ok() {
                eprintln!("q1-graph: empty plan at layer {start}");
            }
            return start;
        }
        let has_moe = plan.iter().any(|it| match it {
            Item::Gdn { run, .. } => run.iter().any(|l| matches!(l.ffn, MetalFfn::Moe(_))),
            Item::Attn { l, .. } => matches!(l.ffn, MetalFfn::Moe(_)),
        });
        let dev_attend = attend_contract
            && (self.head_dim <= 128
                || has_moe
                || attend_mode == "force"
                || attend_mode == "256");
        if !dev_attend {
            for it in &mut plan {
                if let Item::Attn { full_gpu, .. } = it {
                    *full_gpu = false;
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
                            Item::Gdn { run, first } =>
                                format!("GDN run L{first}+{}", run.len()),
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
                    if !graph.encode_gdn_run(run, &ro, gcfg.as_ref().unwrap()) {
                        // Unreachable: the plan was validated above.
                        tracing::error!("q1 graph: GDN run refused after validation");
                        return start;
                    }
                    // Early commit: the GPU starts the run while the
                    // CPU encodes the next layer (nothing to wait on).
                    graph.commit();
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
                    // ── Fully device-resident attention: no sync at all.
                    if *full_gpu {
                        let cache = &self.kv_cache.layers[*li];
                        let cpu_k: Vec<&[f32]> = (0..nkv).map(|g| cache.head_keys(g)).collect();
                        let cpu_v: Vec<&[f32]> = (0..nkv).map(|g| cache.head_values(g)).collect();
                        let cpu_stored = cpu_k[0].len() / hd;
                        let p = crate::gpu::AttnDeviceParams {
                            kv_id,
                            layer: *li,
                            nh,
                            nkv,
                            hd,
                            rd,
                            position,
                            eps: eps as f32,
                            gemma,
                            output_gate: *output_gate,
                            q_norm: *q_norm,
                            k_norm: *k_norm,
                            inv_freq: &inv_freq,
                            cpu_k,
                            cpu_v,
                            cpu_stored,
                        };
                        if graph.attn_device_ok(l, &p) && graph.encode_attn_device(l, &p) {
                            graph.commit();
                            dev_attn.push(*li);
                            continue;
                        }
                        // Mirror refused (nothing encoded) → sandwich.
                    }
                    graph.encode_attn_prefix(l);
                    graph.sync();
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
                    let mut ao = attention::qwen_attention_core(
                        q_raw,
                        k,
                        v,
                        &mut self.kv_cache.layers[*li],
                        &cfg,
                    );
                    graph.encode_attn_suffix(l, &ao);
                    // Early commit: the GPU starts O+FFN while the CPU
                    // encodes the following GDN run / attention prefix.
                    graph.commit();
                    attention::recycle_buf(&mut ao);
                }
            }
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
            if let Some(lm) = self.weights.lm_head.q1_parts() {
                if graph.lm_head_ok(lm) {
                    graph.encode_lm_head(&self.weights.final_norm, lm);
                    lm_rows = Some(lm.1);
                }
            }
        }
        graph.sync();
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
        if let Some(rows) = lm_rows {
            let mut lg = attention::take_buf(rows.min(self.vocab_size));
            graph.read_logits(&mut lg);
            lg.resize(self.vocab_size, 0.0);
            if let Some(c) = self.final_softcap {
                for l in lg.iter_mut() {
                    *l = c * (*l / c).tanh();
                }
            }
            self.graph_logits = Some(lg);
        }
        graph.finish(h);
        // Device-attended layers: replay the CPU bookkeeping — append
        // the mirror's new K/V row (rope'd on the GPU) into the owner
        // cache, then bank this token's Born-importance mass.
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
            kv_history: Vec::new(),
            short_conv_cfg: None,
            mtp: None,
            speculative: std::env::var("CMF_MTP").map(|v| v != "0").unwrap_or(true),
            rng,
            sampler_scratch: SamplerScratch::default(),
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
            final_softcap: None,
            attn_softcap: 0.0,
            graph_want_logits: false,
            graph_logits: None,
            graph_kv_id: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
        }
    }

    /// Enable/disable per-layer O(1) Nyström attention. Only Full
    /// layers are eligible (a linear layer keeps its own operator).
    /// Applies to generation (`generate*`/`forward_ids`): the prompt
    /// pass stays exact, the seal happens once after prefill, decode
    /// runs on the O(1) state. Teacher-forced scoring (`ppl_ids`)
    /// intentionally stays exact.
    pub fn set_o1(&mut self, cfg: Option<crate::nystrom::O1Cfg>) {
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

    /// Arm query collection on the o1 layers (fresh prompt pass).
    /// Reset the o1 layers to Collecting for a fresh sequence. Pub for the
    /// network split: each side runs the o1 lifecycle over ITS OWN layers
    /// (begin before prefill, seal at the prefill barrier).
    pub fn o1_begin(&mut self) {
        if let Some(c) = &self.o1_cfg {
            let (m, w, sink, rect) = (c.m, c.w, c.sink, c.rect);
            for (li, &f) in self.o1_flags.iter().enumerate() {
                if f {
                    self.kv_cache.layers[li].o1_begin(m, w, sink, rect);
                }
            }
        }
    }

    /// Freeze landmarks + skeleton state after the prompt pass and drop
    /// the o1 layers' full KV; decode then runs `step()` per token.
    /// Pub for the network split (see `o1_begin`).
    pub fn o1_seal(&mut self) {
        self.o1_epoch = self.o1_epoch.wrapping_add(1);
        if self.o1_cfg.is_none() {
            return;
        }
        for li in 0..self.num_layers {
            if self.o1_flags.get(li).copied().unwrap_or(false) {
                self.kv_cache.layers[li].o1_seal(self.num_heads);
            }
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

    /// Toggle the per-token Born-confidence reduction (a full-vocab
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

    /// The active calibration temperature (1.0 = raw Born mass).
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
            self.kv_cache.clear();
            self.kv_history.clear();
            crate::gpu::graph_kv_reset(self.graph_kv_id);
        } else if std::env::var("CMF_PREFILL_PROF").is_ok() {
            eprintln!(
                "kv-reuse: {} of {} prompt positions already cached",
                reuse_from,
                input_ids.len()
            );
        }
        crate::gpu::graph_race_begin_generation();
        self.o1_begin();

        // Speculative decode is off under o1: a rejected draft can't be
        // rolled back out of the far accumulators / ring window (the
        // Nyström insertion is irreversible by design).
        // The wgpu token graph owns a device K/V mirror that speculative
        // rollback would desync — the two are mutually exclusive.
        let graph_on = std::env::var("CMF_GPU_WGPU_GRAPH")
            .map(|v| v != "0")
            .unwrap_or_else(|_| {
                // Default ON for wgpu on DISCRETE adapters (4090:
                // decode 76 -> 137 tok/s); integrated/mobile GPUs keep
                // the per-op probe path — see gpu::wgpu_graph_default.
                crate::gpu::wgpu_graph_default()
            });
        // Graph speculative decode (`CMF_GRAPH_SPEC=1`, experimental): the
        // MTP head drafts, ONE batched graph submit verifies the whole
        // chain. The machinery is correct — 81% of drafts accepted on the
        // bench — but it stays OPT-IN until the batched graph itself is
        // fast: today it costs ~48 ms per verified position against the
        // token graph's 7.3, which turns a win into a 6x loss.
        let graph_spec = self.speculative
            && graph_on
            && self.mtp.is_some()
            && task_mask.is_none()
            && !self.o1_active()
            && self.sampler_config.temperature < 1e-6
            && self.sampler_config.repetition_penalty == 1.0
            && std::env::var("CMF_GRAPH_SPEC").is_ok_and(|v| v != "0");
        // GDN hybrids sit the fused-pair speculation out by default: the
        // recurrence is sequential, so the pair lane cannot parallelize
        // (the bench's own Pair line reads fused 1.28x TWO singles on the
        // 35B) and the draft's full-vocab head rides on top — measured 2x
        // SLOWER end to end (16.1 vs 32.4 tok/s on the 48-core stand).
        // CMF_MTP=1 forces it back for study.
        let pair_pays = self.gdn_cfg.is_none()
            || std::env::var("CMF_MTP").as_deref() == Ok("1");
        let spec_active = self.speculative
            && self.mtp.is_some()
            && task_mask.is_none()
            && !self.o1_active()
            && ((!graph_on && pair_pays) || graph_spec)
            && self.sampler_config.temperature < 1e-6;
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
        let batch_k = std::env::var("CMF_BATCH_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
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
        // With dynamic routing, prefill sequentially so the φ hook fires
        // over the PROMPT — the router enters decode with a warm φ (the
        // fused-pair path skips the per-layer φ capture). o1 layers
        // collect their query trace in both the single and pair paths.
        let dyn_prefill = router.is_some();
        // q1 hybrids on Metal: the per-position GPU token graph beats
        // the CPU chunk-GEMM (whose wall is the sequential scalar GDN
        // recurrence), so prefill goes position-by-position through the
        // same graph as decode. Pure-attention models keep the batched
        // path — there the chunk-GEMM amortization wins.
        let graph_prefill = self.graph_prefill_preferred();
        if task_mask.is_none()
            && !dyn_prefill
            && !graph_prefill
            && self.can_prefill_batched()
            && self.g3n.is_none()
            && input_ids.len() > 2
        {
            // Production prefill = the same chunked prefill-GEMM that
            // bench/PPL measure (roadmap §3 P0: generation used to warm
            // the prompt with the slower pair path — the published
            // prefill number didn't match real TTFT). MTP warm-up reads
            // each position's hidden straight from the chunk result.
            let chunk = prefill_chunk();
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
                                    let (dj, hj) =
                                        self.mtp_step_h(m, &hx, d_prev, p + 1 + j);
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
                            let (d1, mut hx) =
                                self.mtp_step_h(m, &h2, input_ids[pos + 2], pos + 1);
                            let mut ok = d1 == input_ids[pos + 3];
                            Self::chain_probe_note(0, ok);
                            let mut d_prev = d1;
                            let mut extra = 0usize;
                            for j in 1..probe {
                                if pos + 3 + j >= input_ids.len() {
                                    break;
                                }
                                let (dj, hj) =
                                    self.mtp_step_h(m, &hx, d_prev, pos + 2 + j);
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
        if batch_k > 0
            && graph_prefill
            && task_mask.is_none()
            && !self.o1_active()
            && mtp.is_none()
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
                let ok_b = self.try_batch_graph_wgpu(&mut hiddens, &positions, bk, None);
                if std::env::var("CMF_GRAPH_PROF").is_ok() {
                    let ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "batch-chunk: k={bk} ok={ok_b} {ms:.1} ms ({:.1} tok/s)",
                        bk as f64 / (ms / 1000.0)
                    );
                }
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static SAID: AtomicBool = AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        if ok_b {
                            tracing::info!("batched prefill: ACTIVE (k={bk})");
                        } else {
                            tracing::warn!("batched prefill declined — per-position graph");
                        }
                    }
                }
                if ok_b {
                    hidden.copy_from_slice(&hiddens[(bk - 1) * hs..]);
                    pos = end;
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
                            let (dj, hj) =
                                self.mtp_step_h(m, &hx, d_prev, pos + 1 + j);
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
        // Cancelled mid-prefill: the cache holds a partial prompt —
        // drop the reuse history and return an empty generation.
        if self
            .cancel
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.kv_history.clear();
            if let Some(m) = mtp {
                self.mtp = Some(m);
            }
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
        self.o1_seal();

        // Commit one token: push, check EOS, stream. Returns false = stop.
        macro_rules! commit {
            ($id:expr) => {{
                all_ids.push($id);
                generated += 1;
                if self.tokenizer.is_eos($id) {
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

        // ── Decode ──
        let mut next_pos = input_ids.len();
        'decode: while generated < max_tokens {
            if self
                .cancel
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                finish_reason = "cancelled".to_string();
                break 'decode;
            }
            let mut logits = match self.graph_logits.take() {
                Some(lg) => lg,
                None => {
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
            let t_next = sampler::sample_with_scratch(
                &logits,
                &self.sampler_config,
                &all_ids,
                &mut self.rng,
                &mut self.sampler_scratch,
            );
            if self.confidence_on {
                confidence.push(top1_prob_t(&logits, t_next, calib_temp));
            }
            attention::recycle_buf(&mut logits);
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

            if self.kv_cache.needs_eviction() {
                let keep = (self.kv_cache.max_seq_len / 2).max(1);
                self.kv_cache.evict(keep);
            }

            match &mut mtp {
                // ── Graph speculation: chain-draft, batch-verify on device ──
                #[cfg(feature = "gpu")]
                Some(m) if graph_spec && generated + 1 < max_tokens && next_pos > 0 => {
                    if let Some((extra, n_pos, new_h)) = self.graph_spec_step(
                        m,
                        &hidden,
                        t_next,
                        next_pos,
                        &mut drafted,
                        &mut accepted,
                    ) {
                        next_pos = n_pos;
                        hidden = new_h;
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
                    // Declined (batch graph refused): plain forward below.
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
                    let t_after = sampler::sample_with_scratch(
                        &logits1,
                        &self.sampler_config,
                        &all_ids,
                        &mut self.rng,
                        &mut self.sampler_scratch,
                    );
                    if self.confidence_on {
                        confidence.push(top1_prob_t(&logits1, t_after, calib_temp));
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
                    {
                        let tip_token = all_ids[all_ids.len() - 2];
                        if let Some((extra, n_pos)) = self.dsv4_spec_step(
                            tip_token,
                            t_next,
                            next_pos,
                            &mut drafted,
                            &mut accepted,
                        ) {
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

        self.graph_want_logits = false;
        self.graph_logits = None;
        // Restore backbone overlay and re-attach the router for reuse.
        if router.is_some() {
            let _ = self.set_active_skill(None);
        }
        self.dyn_router = router.or(self.dyn_router.take());
        self.mtp = mtp.or(self.mtp.take());

        let output_ids = &all_ids[input_ids.len()..];
        // Forwarded = prompt + all generated but the LAST sampled token
        // (emitted without being fed back). Exact only without MTP —
        // reuse is gated off when MTP is active.
        let forwarded = input_ids.len() + output_ids.len().saturating_sub(1);
        self.kv_history = all_ids[..forwarded.min(all_ids.len())].to_vec();
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
                .map(|(d, (n, k))| format!("d{}={:.0}%({n})", d + 1, 100.0 * *k as f64 / (*n).max(1) as f64))
                .collect();
            eprintln!("mtp-chain: {}", line.join(" "));
        }
    }

    /// `mtp_step` that also hands back the block's own output hidden — the
    /// state a CHAINED draft feeds the next step, the way a multi-token
    /// speculative round iterates the head on itself.
    fn mtp_step_h(
        &mut self,
        m: &mut MtpModule,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> (u32, Vec<f32>) {
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
        let mut lg = self.lm_head_forward(&self.ws.n1);
        let draft = sampler::argmax(&lg);
        attention::recycle_buf(&mut lg);
        (draft, x)
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
        inference::rms_norm_into(&x, &m.layer.input_norm, self.rms_eps, self.norm_style, &mut self.ws.n1);
        let attn = match &m.layer.attn {
            AttnKind::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, softplus_gate, bias } => {
                let mut cfg = self.attn_cfg(position);
                cfg.q_norm = q_norm.as_deref();
                cfg.k_norm = k_norm.as_deref();
                cfg.output_gate = *output_gate;
                cfg.softplus_gate = softplus_gate.as_ref().map(|(g, p)| (g, *p));
                cfg.bias = bias.as_ref().map(|(q, k, v)| (q.as_slice(), k.as_slice(), v.as_slice()));
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
    ) -> Option<(Vec<u32>, usize, Vec<f32>)> {
        let k_spec: usize = std::env::var("CMF_GRAPH_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| (1..=8).contains(&v))
            .unwrap_or(2);
        if next_pos == 0 {
            return None;
        }
        let t_round = std::time::Instant::now();
        // Draft the chain: first from the trunk's tip hidden, then the head
        // iterating on itself. Rows land in the MTP KV; the chain rows past
        // the first are speculation over speculative state and roll back
        // below, replaced by verified pairs.
        let mut drafts = Vec::with_capacity(k_spec);
        let (d1, mut hx) = self.mtp_step_h(m, hidden, t_next, next_pos - 1);
        drafts.push(d1);
        for j in 1..k_spec {
            let (dj, hj) = self.mtp_step_h(m, &hx, drafts[j - 1], next_pos - 1 + j);
            drafts.push(dj);
            hx = hj;
        }
        *drafted += k_spec;
        let t_draft = t_round.elapsed();
        // Verify batch: [t_next, d1 .. d_{k-1}] at next_pos.. — every row's
        // logits come back from the graph's own head.
        let b = k_spec + 1;
        let mut hiddens = vec![0.0f32; b * self.hidden_size];
        for (i, &t) in std::iter::once(&t_next).chain(drafts.iter()).enumerate() {
            let e = self.embed_single(t);
            hiddens[i * self.hidden_size..(i + 1) * self.hidden_size].copy_from_slice(&e);
        }
        let positions: Vec<usize> = (next_pos..next_pos + b).collect();
        let (lm_gw, lm_rows) = {
            let (_, i, kind, rs) = self.weights.lm_head.graph_weight()?;
            (
                crate::gpu::GraphW { idx: i, kind, row_scale: rs, data: &[] },
                self.weights.lm_head.rows(),
            )
        };
        let mut logits = Vec::new();
        let final_norm = self.weights.final_norm.clone();
        let ok = self.try_batch_graph_wgpu(
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
        if !ok {
            // Roll the draft rows back out of the MTP cache and decline —
            // the caller runs the plain path, nothing has changed.
            m.kv.truncate_last(k_spec);
            return None;
        }
        if std::env::var("CMF_GRAPH_SPEC_TIME").is_ok() {
            eprintln!(
                "spec-round: draft {:.1} ms | verify {:.1} ms",
                t_draft.as_secs_f64() * 1e3,
                (t_round.elapsed() - t_draft).as_secs_f64() * 1e3,
            );
        }
        // Acceptance: row i's argmax is the trunk's token after input i.
        let ids: Vec<u32> = (0..b)
            .map(|i| sampler::argmax(&logits[i * lm_rows..(i + 1) * lm_rows]))
            .collect();
        let mut a = 0usize;
        while a < k_spec && ids[a] == drafts[a] {
            a += 1;
        }
        // a fully-accepted round needs no restore: every input was real.
        if a + 1 < b {
            crate::gpu::gdn_spec_restore(self.graph_kv_id, a);
        }
        *accepted += a;
        // MTP cache: keep the first draft row (its inputs were real), drop
        // the chain's, then append the verified pairs the round produced.
        m.kv.truncate_last(k_spec.saturating_sub(1));
        for j in 0..a {
            let row = &hiddens[j * self.hidden_size..(j + 1) * self.hidden_size];
            let row = row.to_vec();
            self.mtp_warm(m, &row, ids[j], next_pos + j);
        }
        // The sampler's contract: logits of the LAST verified position.
        let mut row = logits[a * lm_rows..(a + 1) * lm_rows].to_vec();
        row.resize(self.vocab_size, 0.0);
        if let Some(c) = self.final_softcap {
            for l in row.iter_mut() {
                *l = c * (*l / c).tanh();
            }
        }
        self.graph_logits = Some(row);
        let new_hidden = hiddens[a * self.hidden_size..(a + 1) * self.hidden_size].to_vec();
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
        self.kv_cache.clear();
        self.kv_history.clear();
        self.o1_begin();
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut pos = 0usize;
        if self.can_prefill_batched() && ids.len() > 2 {
            // prefill-GEMM in chunks; only the last position's hidden is
            // needed. (o1-compatible: the batch path attends per position
            // through qwen_attention, which carries the collection hook.)
            let chunk = prefill_chunk();
            let hs = self.hidden_size;
            while pos < ids.len() {
                let end = (pos + chunk).min(ids.len());
                let hb = self.prefill_batch_masked(&ids[pos..end], pos, task_mask);
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
        }
        // Same guards as generation's prefill: CMF_PAIR=0 opts out, and a
        // model whose layers live outside `weights.layers` has no pair walk
        // to take (the tail loop below covers every position either way).
        if task_mask.is_none()
            && !std::env::var("CMF_PAIR").is_ok_and(|v| v == "0")
            && self.pair_supported()
        {
            while pos + 1 < ids.len() {
                let e1 = self.embed_single(ids[pos]);
                let e2 = self.embed_single(ids[pos + 1]);
                let (_, h2) = self.forward_pair(&e1, &e2, pos);
                self.commit_linear_scratch();
                hidden = h2;
                pos += 2;
            }
        }
        while pos < ids.len() {
            hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, task_mask);
            pos += 1;
        }
        // Harness contract: after forward_ids the cache is decode-ready —
        // under o1 that means sealed (bench measures the seal as part of
        // prefill, honestly).
        self.o1_seal();
        let normed = inference::rms_norm(
            &hidden,
            &self.weights.final_norm,
            self.rms_eps,
            self.norm_style,
        );
        Ok(self.lm_head_forward(&normed))
    }

    /// Teacher-forced perplexity over a token sequence (phase-C gate:
    /// honest quant comparisons instead of prompt vibes).
    ///
    /// Attention is EXACT even on a model whose layers are flagged for
    /// the O(1) kernel — scoring the backbone is the default on purpose
    /// (it is the yardstick). `nll_ids_o1` scores the CONVERTED model.
    pub fn ppl_ids(&mut self, ids: &[u32]) -> f64 {
        let (nll, cnt) = self.nll_ids_from(ids, 0);
        (nll / cnt.max(1) as f64).exp()
    }

    /// DTG-MA calibration pass (Patent 2): run `ids` through the model
    /// (CPU path, per position) and return each layer's per-neuron
    /// activation mass Σ|silu(gate)·up| — the statistic the task-guided
    /// FFN mask is derived from.
    pub fn probe_ffn_mass(&mut self, ids: &[u32]) -> Vec<Vec<f64>> {
        self.kv_cache.clear();
        self.kv_history.clear();
        FFN_PROBE.with(|p| {
            *p.borrow_mut() = Some(vec![vec![0f64; self.intermediate_size]; self.num_layers]);
        });
        crate::gpu::cpu_scope(|| {
            for (pos, &id) in ids.iter().enumerate() {
                let emb = self.embed_single(id);
                let _ = self.forward_layers(&emb, pos, None);
            }
        });
        self.kv_cache.clear();
        self.kv_history.clear();
        FFN_PROBE
            .with(|p| p.borrow_mut().take())
            .unwrap_or_default()
    }

    /// Teacher-forced PPL with a task mask active (sparse execution) —
    /// the quality gate for a DTG-MA-masked skill. Sequential per
    /// position: the batched prefill path is dense-only.
    pub fn ppl_ids_masked(&mut self, ids: &[u32], mask: &TaskMask) -> f64 {
        self.kv_cache.clear();
        self.kv_history.clear();
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
        }
        self.kv_cache.clear();
        self.kv_history.clear();
        (nll / cnt.max(1) as f64).exp()
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
    ) -> (f64, usize) {
        self.nll_ids_inner(ids, start, task_mask)
    }

    pub fn nll_ids_from(&mut self, ids: &[u32], start: usize) -> (f64, usize) {
        self.nll_ids_inner(ids, start, None)
    }

    fn nll_ids_inner(
        &mut self,
        ids: &[u32],
        start: usize,
        task_mask: Option<&TaskMask>,
    ) -> (f64, usize) {
        self.kv_cache.clear();
        self.kv_history.clear();
        let mut nll = 0f64;
        let mut cnt = 0usize;
        if self.can_prefill_batched() {
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
            self.kv_cache.clear();
            self.kv_history.clear();
            return (nll, cnt);
        }
        for pos in 0..ids.len().saturating_sub(1) {
            let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, task_mask);
            // Architectures whose head lives inside their own stack return
            // the logits out of band and a zero hidden — DeepSeek-V4 folds
            // its hyper-connection copies between the last layer and the
            // norm, so it cannot hand back a vector this loop could use.
            // Scoring the zeros gave a perplexity of exactly the vocabulary
            // size, which is a uniform distribution reported as a
            // measurement. `generate` already reads this channel.
            let out_of_band = self.graph_logits.take();
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
        self.kv_cache.clear();
        self.kv_history.clear();
        (nll, cnt)
    }

    /// Teacher-forced NLL of the CONVERTED model: the O(1) Nyström path
    /// is ACTIVE over the scored positions. Returns (nll sum, scored
    /// count) over `prefill..len-1`.
    ///
    /// Runtime discipline, deliberately NOT the matrix probe's: the
    /// first `prefill` tokens run the exact prompt pass — that pass is
    /// what freezes the landmarks and M — and every scored position then
    /// goes through `NystromState::step()`, the same code decode runs.
    /// So the landmarks are PREFILL-frozen (what ships), not
    /// full-sequence oracles (what the published probe measured), and
    /// every scored row carries a real far field rather than sitting
    /// inside the exact window.
    ///
    /// Pair with `nll_ids_from(ids, prefill)` for the exact baseline
    /// over the identical token set — that ratio is the honest one.
    pub fn nll_ids_o1(&mut self, ids: &[u32], prefill: usize) -> (f64, usize) {
        self.kv_cache.clear();
        self.kv_history.clear();
        self.o1_begin();
        let n = ids.len().saturating_sub(1);
        let p = prefill.min(n);
        // Exact prompt pass over ids[..p]: the seal consumes its q/k/v.
        let mut pos = 0usize;
        if self.can_prefill_batched() {
            const CHUNK: usize = 128;
            while pos < p {
                let end = (pos + CHUNK).min(p);
                let _ = self.prefill_batch(&ids[pos..end], pos);
                pos = end;
            }
        } else {
            while pos < p {
                let _ = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
                pos += 1;
            }
        }
        self.o1_seal();

        let mut nll = 0f64;
        let mut cnt = 0usize;
        for pos in p..n {
            let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
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
        self.kv_cache.clear();
        self.kv_history.clear();
        (nll, cnt)
    }

    /// Teacher-forced calibration data (B1): for each position, whether the
    /// argmax equals the actual next token, and the top-1 softmax prob
    /// (Born mass) under EACH temperature in `temps` — all from ONE forward
    /// pass (argmax/correctness are temperature-invariant; only p_max
    /// reshapes). Feeds `cortiq calibrate` (reliability/ECE + temperature
    /// fit): is the model's confidence a true property, or does it need a
    /// measured scaling?
    pub fn calib_ids(&mut self, ids: &[u32], temps: &[f32]) -> (Vec<bool>, Vec<Vec<f32>>) {
        self.kv_cache.clear();
        self.kv_history.clear();
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
        self.kv_cache.clear();
        self.kv_history.clear();
        (correct, pmax)
    }

    /// Teacher-forced PPL with the dynamic router driving per-window
    /// skill switches (VMF experiment №2 measurement). Sequential (φ
    /// must update per token), returns (ppl, switch_count). The router
    /// must be enabled (`enable_dynamic_routing`); else this equals
    /// plain `ppl_ids`. The active skill when scoring token t shapes the
    /// logits for t+1 — on-policy over the held-out text itself.
    pub fn ppl_ids_dynamic(&mut self, ids: &[u32]) -> (f64, usize) {
        let mut router = match self.dyn_router.take() {
            Some(r) => r,
            None => return (self.ppl_ids(ids), 0),
        };
        router.reset();
        self.dyn_phi_seen = 0;
        let _ = self.set_active_skill(None);

        self.kv_cache.clear();

        self.kv_history.clear();
        let mut nll = 0f64;
        let mut cnt = 0usize;
        for pos in 0..ids.len().saturating_sub(1) {
            let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
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
            // Route on the evolving φ (drives the NEXT token's skill).
            let phi = self.dyn_phi_ema.clone();
            if let Some(new_active) = router.step(&phi, pos) {
                let _ = self.set_active_skill(new_active);
            }
        }
        let switches = router.switches.len();
        let _ = self.set_active_skill(None);
        self.dyn_router = Some(router);
        self.kv_cache.clear();
        self.kv_history.clear();
        ((nll / cnt.max(1) as f64).exp(), switches)
    }

    /// Routing probe φ (spec §9): mean-pooled hidden after `layer`.
    pub fn probe_phi(&mut self, ids: &[u32], layer: usize) -> Vec<f32> {
        self.kv_cache.clear();
        self.kv_history.clear();
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
        self.kv_cache.clear();
        self.kv_history.clear();
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

        #[cfg(target_os = "macos")]
        let mut chunk_skip_until = 0usize;
        for li in from..upto_excl {
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
                if let Some(t) = tp.parse::<usize>().ok() {
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
        if self.dsv4.is_some() {
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
            if d.act != Act::Silu {
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
        self.forward_layers_upto(hidden, position, task_mask, None)
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
        Ok(self.forward_layers_span(hidden, position, task_mask, from, Some(upto)))
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
        self.kv_cache.clear();
        self.kv_history.clear();
        crate::gpu::graph_kv_reset(self.graph_kv_id);
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
        if self.can_prefill_batched() {
            Ok(self.prefill_batch_span(PrefillIn::Ids(ids), start_pos, task_mask, 0, upto + 1))
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
        if self.can_prefill_batched() {
            Ok(self.prefill_batch_span(
                PrefillIn::Hidden(hidden),
                start_pos,
                task_mask,
                from,
                upto + 1,
            ))
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
    ) -> Option<Vec<f32>> {
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
    ) -> Option<Vec<f32>> {
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
        let graph_on = match std::env::var("CMF_GPU_WGPU_GRAPH").ok().as_deref() {
            Some("0") => return None,
            Some(_) => true,
            None => crate::gpu::wgpu_graph_default(),
        };
        if !graph_on {
            return None;
        }
        let emb = self.embed_single(t_next);
        let mut lg = Vec::new();
        let mut ids = Vec::new();
        self.try_token_graph_wgpu_steps(
            &emb,
            position,
            &mut lg,
            k,
            Some(&mut ids),
            None,
            0,
            self.num_layers,
        )?;
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
    ) -> Option<Vec<f32>> {
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
                .filter(|li| !matches!(self.kv_cache.layers[self.phys_layer(*li)].o1, None))
                .count();
            let have = o1_views.iter().filter(|v| v.is_some()).count();
            if want == 0 || have != want {
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
            if let Some((_, i, kind, rs)) = t.graph_weight() {
                return Some(crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                });
            }
            // Small unquantized projections (GDN in_proj_a/b) stay f32.
            t.as_f32().map(|d| crate::gpu::GraphW {
                idx: 0,
                kind: 4,
                row_scale: &[],
                data: d,
            })
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
                FfnKind::Dense(d) => crate::gpu::GraphFfn::Dense {
                    gate: gw(&d.gate_proj)?,
                    up: gw(&d.up_proj)?,
                    down: gw(&d.down_proj)?,
                },
                FfnKind::Moe(m) => {
                    // v1 scope: softmax router + shared expert + uniform
                    // q4t expert trios (the MoE-hybrid coder class). The
                    // biased/sigmoid routers and adaptive τ keep the CPU
                    // path, where they are implemented.
                    if m.router_sigmoid
                        || m.expert_bias.is_some()
                        || m.route_tau.is_some()
                        || m.mask.is_some()
                    {
                        return None;
                    }
                    let (se, sg) = m.shared.as_ref()?;
                    let sgate = gw(sg.as_ref()?)?;
                    let router = gw(&m.router)?;
                    let inter = m.experts.first()?.gate_proj.rows();
                    let mut experts = Vec::with_capacity(m.experts.len() + 1);
                    // q4t or q4tp, but not both in one layer — the kernels
                    // are picked per layer, not per expert.
                    let mut q4tp: Option<bool> = None;
                    // The mixed 2-bit profile: q2tp gate/up over a q4tp
                    // down. Uniform across the layer, like `q4tp` itself.
                    let mut gu_q2: Option<bool> = None;
                    for e in m.experts.iter().chain(std::iter::once(se)) {
                        if !matches!(e.act, Act::Silu)
                            || e.gate_proj.rows() != inter
                            || e.up_proj.rows() != inter
                        {
                            return None;
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
                    let (m, _, _, _) = wq.graph_weight()?;
                    model = Some(m.clone());
                    crate::gpu::GraphAttn::Full {
                        wq: gw(wq)?,
                        wk: gw(wk)?,
                        wv: gw(wv)?,
                        wo: gw(wo)?,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
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
                    let (m, _, _, _) = w.in_proj_qkv.graph_weight()?;
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
            self.weights.lm_head.graph_weight().map(|(_, i, kind, rs)| {
                (
                    crate::gpu::GraphW {
                        idx: i,
                        kind,
                        row_scale: rs,
                        data: &[],
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
                .map(|(_, i, kind, rs)| {
                    (
                        crate::gpu::GraphW {
                            idx: i,
                            kind,
                            row_scale: rs,
                            data: &[],
                        },
                        self.weights.embed_tokens.rows(),
                        self.embed_multiplier as f32,
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
        crate::gpu::forward_token_graph(
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
        )
        .then_some(h)
    }

    /// Batched prefill: k contiguous prompt positions through the whole wgpu
    /// graph in ONE submit (projections/FFN as GEMMs). `hiddens` is [k·hidden]
    /// in/out (embeddings in, layer output out); KV mirror / GDN state advance.
    /// false ⇒ unsupported → caller keeps the per-position graph.
    fn try_batch_graph_wgpu(
        &self,
        hiddens: &mut [f32],
        positions: &[usize],
        k: usize,
        spec: Option<crate::gpu::SpecTail<'_>>,
    ) -> bool {
        let _tb = std::time::Instant::now();
        if self.attn_softcap > 0.0 {
            return false; // capped scores: no graph kernel — CPU path
        }
        if self.o1_active() {
            return false;
        }
        let nh = self.num_heads;
        let (nkv, hd, rd) = self.layer_geom(0);
        let gemma = self.norm_style == cortiq_core::NormStyle::Gemma;
        fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
            if let Some((_, i, kind, rs)) = t.graph_weight() {
                return Some(crate::gpu::GraphW {
                    idx: i,
                    kind,
                    row_scale: rs,
                    data: &[],
                });
            }
            t.as_f32().map(|d| crate::gpu::GraphW {
                idx: 0,
                kind: 4,
                row_scale: &[],
                data: d,
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
                    FfnKind::Dense(d) => crate::gpu::GraphFfn::Dense {
                        gate: gw(&d.gate_proj)?,
                        up: gw(&d.up_proj)?,
                        down: gw(&d.down_proj)?,
                    },
                    FfnKind::Moe(m) => {
                        if m.router_sigmoid
                            || m.expert_bias.is_some()
                            || m.route_tau.is_some()
                            || m.mask.is_some()
                        {
                            return None;
                        }
                        let (se, sg) = m.shared.as_ref()?;
                        let sgate = gw(sg.as_ref()?)?;
                        let router = gw(&m.router)?;
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
                            return None;
                        }
                        let (m, _, _, _) = wq.graph_weight()?;
                        model = Some(m.clone());
                        crate::gpu::GraphAttn::Full {
                            wq: gw(wq)?,
                            wk: gw(wk)?,
                            wv: gw(wv)?,
                            wo: gw(wo)?,
                            q_norm: q_norm.as_deref(),
                            k_norm: k_norm.as_deref(),
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
                        let (m, _, _, _) = w.in_proj_qkv.graph_weight()?;
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
            return false;
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
            k,
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
        *ON.get_or_init(|| std::env::var("CMF_DSV4_SPEC").map(|v| v != "0").unwrap_or(true))
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
                    eprintln!("между раундами {:.1} мс", prev.elapsed().as_secs_f64() * 1e3);
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
                    if props.is_empty() { "пуст" } else { "мимо" },
                    props.first()
                );
            }
            return None;
        }
        let mut k_verify = crate::dsv4::dspark_verify_k().min(props.len());
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
            eprintln!(
                "spec@{next_pos}: fed={fed:?} argmax={argmax:?} accepted={accepted}"
            );
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
            eprintln!("finish(k={accepted}): {:.1} мс", t_fin.elapsed().as_secs_f64() * 1e3);
        }
        *accepted_ctr += accepted - 1;
        // Captures per accepted token: device targets photographed by the
        // batch, host targets from the verify's own walk. The last one
        // becomes the new tip's draft input; every one owes the ring an
        // entry for its position.
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        // A PARTIAL capture layer never rides the chain, so the batch has
        // no photograph of it — its tip capture comes from the walk's own
        // note like any host layer's. Filtering on the device set alone
        // handed the draft a never-written photo slot for exactly the
        // most important input (the last layer feeds main_proj), and the
        // split configurations drafted at 27% no matter the residency.
        let dev_caps: Vec<usize> = targets
            .iter()
            .copied()
            .filter(|&t| {
                st.dev_set.get(t).copied().unwrap_or(false)
                    && !st.partial_set.get(t).copied().unwrap_or(false)
            })
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
            crate::dsv4::dspark_ring_append(g, &self.dsv4_mtp, &cfg, ds, next_pos + t, self.pool.as_deref());
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
            eprintln!("spec_step total {:.1} мс (k={accepted})", t_all.elapsed().as_secs_f64() * 1e3);
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
            eprintln!("DSpark: захват со слоёв {t:?}, блок {}", crate::dsv4::dspark_block());
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
        self.forward_layers_span(hidden, position, task_mask, 0, upto)
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
        debug_assert!(from == 0 || (self.dsv4.is_none() && self.g3n.is_none()));
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
        let race_eligible = graph_on && upto.is_none() && task_mask.is_none() && from == 0;
        let mut tail_start = 0usize;
        if race_eligible && crate::gpu::graph_race_use_graph(graph_trusted) {
            let t_graph = std::time::Instant::now();
            let mut lg = Vec::new();
            let mut gl = 0usize;
            let built = self.try_token_graph_wgpu(hidden, position, &mut lg, &mut gl);
            graph_note(built.is_some());
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
        // Span runs (network split): the graph covers exactly [from..=upto]
        // — one submit per SEGMENT per token. No race: its state is global
        // and calibrated on full stacks, so spans take the graph only where
        // it is trusted by default (discrete adapters / CMF_GPU_WGPU_GRAPH).
        let span = from > 0 || upto.is_some();
        if span && graph_on && task_mask.is_none() && graph_trusted {
            let upto_excl = upto.map_or(self.num_layers, |u| u + 1);
            let mut lg = Vec::new();
            let mut gl = 0usize;
            if let Some(hh) =
                self.try_token_graph_wgpu_span(hidden, position, &mut lg, from, upto_excl, &mut gl)
            {
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

        #[cfg(target_os = "macos")]
        let mut gpu_skip_until = 0usize;
        for li in tail_start.max(from)..self.num_layers {
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
                (true, FfnKind::Dense(d)) => {
                    let tm = task_mask.unwrap();
                    let alive = tm.ffn_active_count(li);
                    let deep = alive * 2 <= self.intermediate_size;
                    if deep && d.down_proj.sparse_col_ok() {
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

            // Dynamic routing φ capture (on-policy, fireball-style): the
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
        logits
    }

    /// Prefill `ids` and return the next-token logits — what the model
    /// would predict next, WITHOUT committing to generation (introspection
    /// for `cortiq explain`). Clears and repopulates the KV cache; leaves
    /// the active overlay untouched.
    pub fn prefill_next_logits(&mut self, ids: &[u32], task_mask: Option<&TaskMask>) -> Vec<f32> {
        self.kv_cache.clear();
        self.kv_history.clear();
        let mut hidden = vec![0.0f32; self.hidden_size];
        for (pos, &id) in ids.iter().enumerate() {
            let emb = self.embed_single(id);
            hidden = self.forward_layers(&emb, pos, task_mask);
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
fn zero_masked_cols(g: &mut [f32], rows: usize, inter: usize, row: &[u8]) {
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
                    g[base + j] = 0.0;
                }
            }
        }
    }
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
    for i in 0..b * inter {
        g[i] = d.act.combine(g[i], u[i]);
    }
    if let Some(row) = mask_row {
        zero_masked_cols(&mut g, b, inter, row);
    }
    let mut out = vec![0.0f32; b * hidden];
    d.down_proj.matmat(&g, b, &mut out, pool);
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
    m.router.matmat(xs, b, &mut logits, pool);

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
                        *panel_ptr.at(ai) =
                            dense_ffn_batch(&experts[e], &sub, sb, None, None);
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
    if crate::gpu::enabled_here()
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
        if d.act == Act::Silu && QTensor::matvec_silu_mul(&d.gate_proj, &d.up_proj, x, g, pool) {
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
        FFN_PROBE.with(|pr| {
            if let Some(acc) = pr.borrow_mut().as_mut() {
                let li = crate::gpu::cur_layer();
                if li >= 0 {
                    if let Some(row) = acc.get_mut(li as usize) {
                        for (a, &v) in row.iter_mut().zip(g.iter()) {
                            *a += (v as f64).abs();
                        }
                    }
                }
            }
        });
        let mut out = attention::take_buf(d.down_proj.rows());
        d.down_proj.matvec(g, &mut out, pool);
        out
    })
}

thread_local! {
    /// DTG-MA activation probe: per-layer per-neuron Σ|silu(g)·u|
    /// accumulator, alive only during `Pipeline::probe_ffn_mass`.
    static FFN_PROBE: std::cell::RefCell<Option<Vec<Vec<f64>>>> =
        const { std::cell::RefCell::new(None) };
}

/// `dense_ffn_cpu` with a per-visit mask landing on the activations —
/// the masked-inference fast path's decode arm. Full fused quant
/// compute, closed neurons zeroed before down: arithmetically the
/// pruned network, no dequant, no weight bytes touched.
fn dense_ffn_masked(
    d: &DenseFfn,
    x: &[f32],
    pool: Option<&Pool>,
    mask_row: &[u8],
) -> Vec<f32> {
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
        } => Some((model, *idx, *rows, *cols, &[][..], &[][..], true, false, false)),
        // q4_tiled: 18-byte tiles with embedded f16 scales — raw xs.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q4Tiled,
            rows,
            cols,
            ..
        } => Some((model, *idx, *rows, *cols, &[][..], &[][..], false, true, false)),
        // q4tp: same raw-xs contract, different stride and scale plane.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q4TiledP,
            rows,
            cols,
            ..
        } => Some((model, *idx, *rows, *cols, &[][..], &[][..], false, true, false)),
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
        } => Some((model, *idx, *rows, *cols, &[][..], &[][..], false, true, true)),
        _ => None,
    }
}

/// Map a softmax-router MoE onto the Metal token graph's contract:
/// f32 router, gated shared expert, experts uniformly q4tp (or the
/// mixed profile: q2tp gate/up over a q4tp down). Sigmoid/bias/τ
/// routers, masks, per-expert scales and Gemma's router-input norm
/// refuse here — those semantics stay on the CPU path.
#[cfg(target_os = "macos")]
fn metal_moe_graph_parts(m: &MoeFfn, hidden: usize) -> Option<crate::gpu::GpuMoe<'_>> {
    if m.router_sigmoid
        || m.router_input_norm
        || m.expert_bias.is_some()
        || m.route_tau.is_some()
        || m.mask.is_some()
        || m.per_expert_scale.is_some()
        || m.experts.is_empty()
        || m.top_k == 0
    {
        return None;
    }
    // The select kernel hard-codes the gated shared expert; an
    // ungated one would need its own weight-1 slot.
    let (sh, sg) = match &m.shared {
        Some((sh, Some(sg))) => (sh, sg),
        _ => return None,
    };
    let (rf, rr, rc) = m.router.f32_parts()?;
    if rr != m.experts.len() || rc != hidden {
        return None;
    }
    let (sf, sr, sc) = sg.f32_parts()?;
    if sr * sc != hidden {
        return None;
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
    let experts = m
        .experts
        .iter()
        .map(trio)
        .collect::<Option<Vec<_>>>()?;
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
fn moe_route(logits: &[f32], m: &MoeFfn, allowed: Option<&[bool]>) -> (Vec<usize>, Vec<f32>, f32) {
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

/// MoE FFN: router → top-k experts (see `moe_route`). Only selected
/// experts' pages are touched in mmap.
fn moe_ffn(m: &MoeFfn, x: &[f32], pool: Option<&Pool>, allowed: Option<&[bool]>) -> Vec<f32> {
    accumulate_act(m, x, 1);
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; ne];
    m.router.matvec(x, &mut logits, pool);
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
/// look "GPU-accelerated" while every layer walks the host.
fn graph_note(built: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        if built {
            tracing::info!("wgpu whole-token graph: ACTIVE");
        } else {
            tracing::warn!("wgpu whole-token graph refused — per-op path");
        }
    }
}

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
/// and the pad is sliced off before O. Born importance is not
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
        // Flag auto-cleared: the next call generates normally.
        let r2 = p.generate_from_ids(&[1, 2, 3], 4, None, None).unwrap();
        assert_ne!(r2.finish_reason, "cancelled");
    }
    use super::*;

    /// sparse_ffn_quant must equal a dense FFN where inactive neurons are
    /// zeroed (mask × mmap correctness). On F32 tensors this is EXACT —
    /// it validates the row_dot / add_col_scaled / scatter indexing, the
    /// bug-prone part. The q8 branches reuse the golden-tested linear
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
        };
        let shared = DenseFfn {
            gate_proj: identity(),
            up_proj: identity(),
            down_proj: identity(),
            act: Act::Silu,
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
        };
        let actual = moe_ffn_cpu(&moe, &x, &[0], &[0.0], 1.0, None);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
