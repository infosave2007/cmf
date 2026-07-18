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
    gdn_forward, gdn_pair, vmf_phase_forward, vmf_phase_pair, GdnCfg, GdnWeights, VmfPhaseCfg,
    VmfPhaseWeights,
};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::sampler::{self, SamplerConfig, SplitMix64};
use crate::tokenizer::Tokenizer;
use cortiq_core::mask::TaskMask;
use cortiq_core::types::NormStyle;

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
    pub num_layers: usize,
    pub vocab_size: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    pub norm_style: NormStyle,
    /// RoPE dims actually rotated (≤ head_dim; Qwen3.5 uses head_dim/4).
    pub rotary_dim: usize,
    /// Linear-core geometry (present when the model has linear layers).
    pub vmf_cfg: Option<VmfPhaseCfg>,
    /// GatedDeltaNet geometry (faithful vendor operator).
    pub gdn_cfg: Option<GdnCfg>,
    /// Multi-token-prediction head (None = absent).
    pub mtp: Option<MtpModule>,
    /// Speculative decode via MTP (greedy only; `CMF_MTP=0` disables).
    pub speculative: bool,
    rng: SplitMix64,
    /// Precomputed RoPE inverse frequencies [head_dim/2]. Arc: the
    /// forward path clones a handle to escape the &mut self borrow —
    /// cloning the table itself was a per-forward allocation.
    inv_freq: std::sync::Arc<Vec<f32>>,
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
    /// RoPE table of the sliding (local) layers, when they use their
    /// own base frequency (Gemma-3: 10k local vs 1M global).
    pub inv_freq_local: Option<std::sync::Arc<Vec<f32>>>,
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Act {
    #[default]
    Silu,
    GeluTanh,
}

impl Act {
    pub fn from_arch(name: &str) -> Self {
        if name == "gelu_tanh" {
            Self::GeluTanh
        } else {
            Self::Silu
        }
    }

    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Silu => inference::silu(x),
            Self::GeluTanh => inference::gelu_tanh(x),
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
}

pub struct MoeFfn {
    /// Router `mlp.gate.weight` [num_experts, hidden].
    pub router: QTensor,
    pub experts: Vec<DenseFfn>,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    /// Qwen2-MoE always-on shared expert: (FFN, sigmoid-gate [1, hidden]).
    pub shared: Option<(DenseFfn, QTensor)>,
    /// Expert-selection counters (truncated Fisher B-field of claim 12:
    /// routing frequency during calibration). Filled by every forward,
    /// read by the CLI via CMF_MOE_STATS. RefCell: decode is single-threaded.
    pub stats: std::cell::RefCell<Vec<u64>>,
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
        /// Qwen2-family projection biases (q, k, v).
        bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    },
    /// Canonical linear core (VMF phase attention).
    Linear(VmfPhaseWeights),
    /// Faithful vendor linear operator (Qwen3.5 GatedDeltaNet).
    LinearGdn(GdnWeights),
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
    std::env::var("CMF_PREFILL").map(|v| v != "seq").unwrap_or(true)
}

/// Prefill chunk (positions per batched pass). On macOS the AMX GEMM
/// path wants tall panels — M=48 starves the matrix units (ggml uses
/// ubatch 512); elsewhere the historical 48 stays. CMF_PREFILL_CHUNK
/// overrides.
fn prefill_chunk() -> usize {
    if let Some(n) =
        std::env::var("CMF_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok())
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
    #[cfg(target_os = "macos")]
    fn graph_prefill_preferred(&self) -> bool {
        if !crate::gpu::enabled_here()
            || !crate::gpu::q1_force()
            || std::env::var("CMF_GPU_BLOCK").map(|v| v == "0").unwrap_or(false)
        {
            return false;
        }
        self.weights.layers.iter().any(
            |lw| matches!(&lw.attn, AttnKind::LinearGdn(w) if w.in_proj_qkv.is_q1()),
        )
    }

    #[cfg(not(target_os = "macos"))]
    fn graph_prefill_preferred(&self) -> bool {
        false
    }

    #[cfg(target_os = "macos")]
    fn q1_graph_gpu(
        &mut self,
        start: usize,
        upto: Option<usize>,
        position: usize,
        h: &mut [f32],
    ) -> usize {
        use crate::gpu::{AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, GraphDims, TokenGraph};
        if !crate::gpu::enabled_here()
            || !crate::gpu::q1_force()
            || std::env::var("CMF_GPU_BLOCK").map(|v| v == "0").unwrap_or(false)
        {
            return start;
        }
        // The graph encodes SiLU FFN, 1/√hd attention scores and
        // full-context attend with no branch norms — Gemma-style archs
        // (sliding window, scale override, sandwich norms, GeLU) fall
        // back to the CPU path.
        if self.swa.is_some()
            || self.global_attn.is_some()
            || self.attn_v_norm
            || (self.attn_scale - 1.0 / (self.head_dim as f32).sqrt()).abs() > 1e-9
            || self.weights.layers.iter().any(|lw| {
                lw.attn_out_norm.is_some()
                    || lw.ffn_out_norm.is_some()
                    || lw.layer_scale.is_some()
                    || matches!(&lw.ffn, FfnKind::Dense(d) if d.act != Act::Silu)
            })
        {
            return start;
        }
        let limit = upto.map(|u| u + 1).unwrap_or(self.num_layers).min(self.num_layers);

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

        // Device-attend eligibility shared by every Full layer.
        let dev_attend = std::env::var("CMF_GPU_ATTEND").map(|v| v != "0").unwrap_or(true)
            && self.head_dim % 4 == 0
            && self.head_dim <= 128
            && self.rotary_dim >= 2
            && self.rotary_dim <= self.head_dim
            && (self.rotary_dim / 2) % 32 == 0
            && self.num_kv_heads > 0
            && self.num_heads % self.num_kv_heads == 0;

        let mut plan: Vec<Item> = Vec::new();
        let mut model_ref: Option<std::sync::Arc<cortiq_core::CmfModel>> = None;
        let mut scan = start;
        while scan < limit {
            let lw = &self.weights.layers[scan];
            let FfnKind::Dense(d) = &lw.ffn else { break };
            let (Some(g), Some(u), Some(dn)) =
                (d.gate_proj.q1_parts(), d.up_proj.q1_parts(), d.down_proj.q1_parts())
            else {
                break;
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
                    let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = parts else { break };
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
                        gate: g,
                        up: u,
                        down: dn,
                        conv1d: &w.conv1d,
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                        gnorm: &w.norm,
                    };
                    match plan.last_mut() {
                        Some(Item::Gdn { run, .. }) => run.push(gl),
                        _ => plan.push(Item::Gdn { run: vec![gl], first: scan }),
                    }
                }
                AttnKind::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, bias }
                    if !self.kv_cache.layers[scan].o1_sealed() =>
                {
                    let parts = (wq.q1_parts(), wk.q1_parts(), wv.q1_parts(), wo.q1_parts());
                    let (Some(pq), Some(pk), Some(pv), Some(po)) = parts else { break };
                    if let QTensor::Mapped { model, .. } = wq {
                        model_ref.get_or_insert_with(|| model.clone());
                    }
                    let cache = &self.kv_cache.layers[scan];
                    let full_gpu = dev_attend
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
                            gate: g,
                            up: u,
                            down: dn,
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
        let Some(model) = model_ref else { return start };
        if plan.is_empty() {
            return start;
        }
        let dims = GraphDims {
            hidden: self.hidden_size,
            eps: self.rms_eps as f32,
            gemma: self.norm_style == cortiq_core::NormStyle::Gemma,
        };
        let Some(mut graph) = TokenGraph::new(&model, dims, h) else { return start };
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
                Item::Attn { l, li, q_norm, k_norm, output_gate, bias, full_gpu } => {
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
                        window: None,
                        v_norm: false,
                        q_norm: *q_norm,
                        k_norm: *k_norm,
                        output_gate: *output_gate,
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
            && std::env::var("CMF_GPU_LMHEAD").map(|v| v != "0").unwrap_or(true)
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
            vocab_size,
            rms_eps,
            rope_base,
            norm_style,
            rotary_dim: head_dim,
            vmf_cfg: None,
            gdn_cfg: None,
            mtp: None,
            speculative: std::env::var("CMF_MTP").map(|v| v != "0").unwrap_or(true),
            rng,
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
            o1_flags: Vec::new(),
            trace: false,
            calib_temp: 1.0,
            confidence_on: true,
            embed_multiplier: 1.0,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            swa: None,
            inv_freq_local: None,
            global_attn: None,
            inv_freq_global: None,
            attn_v_norm: false,
            final_softcap: None,
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
                    if *f && !matches!(self.weights.layers[li].attn, AttnKind::Full { .. }) {
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
                self.num_layers, c.m, c.w, c.sink, c.rect
            );
        }
        self.o1_cfg = cfg;
    }

    /// True when at least one layer runs the O(1) kernel.
    pub fn o1_active(&self) -> bool {
        self.o1_cfg.is_some() && self.o1_flags.iter().any(|&f| f)
    }

    /// Arm query collection on the o1 layers (fresh prompt pass).
    fn o1_begin(&mut self) {
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
    fn o1_seal(&mut self) {
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
            window: None,
            v_norm: false,
            q_norm: None,
            k_norm: None,
            output_gate: false,
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

        // Fresh sequence — the cache holds absolute positions.
        self.kv_cache.clear();
        self.o1_begin();

        // Speculative decode is off under o1: a rejected draft can't be
        // rolled back out of the far accumulators / ring window (the
        // Nyström insertion is irreversible by design).
        let spec_active = self.speculative
            && self.mtp.is_some()
            && task_mask.is_none()
            && !self.o1_active()
            && self.sampler_config.temperature < 1e-6;
        // The MTP module is detached during generation so its mutable
        // state does not fight the borrow on `self`.
        let mut mtp = if spec_active { self.mtp.take() } else { None };
        if let Some(m) = &mut mtp {
            m.kv.clear();
        }
        // Dynamic router detached during decode (same borrow trick as MTP).
        // Speculative decode and dynamic routing are mutually exclusive
        // for now — the fused-pair path doesn't carry per-token φ.
        let mut router = if mtp.is_none() { self.dyn_router.take() } else { None };
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
        let mut pos = 0usize;
        // lm_head-in-graph is only sound when the very next logits
        // consumer is this loop's own (MTP and skill routing interleave
        // other forwards / can swap lm_head between forward and sample).
        let fuse_lm = mtp.is_none() && router.is_none();
        self.graph_logits = None;
        self.graph_want_logits = false;
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
            && prefill_batched()
            && input_ids.len() > 2
        {
            // Production prefill = the same chunked prefill-GEMM that
            // bench/PPL measure (roadmap §3 P0: generation used to warm
            // the prompt with the slower pair path — the published
            // prefill number didn't match real TTFT). MTP warm-up reads
            // each position's hidden straight from the chunk result.
            let chunk = prefill_chunk();
            let hs = self.hidden_size;
            while pos < input_ids.len() {
                let end = (pos + chunk).min(input_ids.len());
                let hb = self.prefill_batch(&input_ids[pos..end], pos);
                if let Some(m) = &mut mtp {
                    for p in pos..end {
                        if p + 1 < input_ids.len() {
                            let _ = self.mtp_step(
                                m,
                                &hb[(p - pos) * hs..(p - pos + 1) * hs],
                                input_ids[p + 1],
                                p,
                            );
                        }
                    }
                }
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
        }
        if task_mask.is_none() && !dyn_prefill && !graph_prefill {
            while pos + 1 < input_ids.len() {
                let e1 = self.embed_single(input_ids[pos]);
                let e2 = self.embed_single(input_ids[pos + 1]);
                let (h1, h2) = self.forward_pair(&e1, &e2, pos);
                // Both prefill tokens are real → commit lane-2 states.
                self.commit_linear_scratch();
                if let Some(m) = &mut mtp {
                    let _ = self.mtp_step(m, &h1, input_ids[pos + 1], pos);
                    if pos + 2 < input_ids.len() {
                        let _ = self.mtp_step(m, &h2, input_ids[pos + 2], pos + 1);
                    }
                }
                hidden = h2;
                pos += 2;
            }
        }
        while pos < input_ids.len() {
            self.graph_want_logits = fuse_lm && pos + 1 == input_ids.len();
            hidden = self.forward_layers(&self.embed_single(input_ids[pos]), pos, task_mask);
            if let Some(m) = &mut mtp {
                if pos + 1 < input_ids.len() {
                    let _ = self.mtp_step(m, &hidden, input_ids[pos + 1], pos);
                }
            }
            pos += 1;
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
            let t_next = sampler::sample(&logits, &self.sampler_config, &all_ids, &mut self.rng);
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
                // ── Speculative: draft t+2, verify in a fused pair ──
                Some(m) if generated + 1 < max_tokens => {
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
                    let t_after =
                        sampler::sample(&logits1, &self.sampler_config, &all_ids, &mut self.rng);
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
                            hidden = self
                                .forward_layers(&self.embed_single(t_after), next_pos + 1, None);
                        }
                        next_pos += 2;
                    }
                    if stop {
                        break 'decode;
                    }
                }
                // ── Vanilla: forward the sampled token ──
                _ => {
                    self.graph_want_logits = fuse_lm;
                    hidden = self.forward_layers(&self.embed_single(t_next), next_pos, task_mask);
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
    fn mtp_step(&mut self, m: &mut MtpModule, hidden: &[f32], next_token: u32, position: usize) -> u32 {
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
        inference::rms_norm_into(&x, &lw.input_norm, self.rms_eps, self.norm_style, &mut self.ws.n1);
        let attn = match &lw.attn {
            AttnKind::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                output_gate,
                bias,
            } => {
                let mut cfg = self.attn_cfg(position);
                cfg.q_norm = q_norm.as_deref();
                cfg.k_norm = k_norm.as_deref();
                cfg.output_gate = *output_gate;
                cfg.bias = bias
                    .as_ref()
                    .map(|(q, k, v)| (q.as_slice(), k.as_slice(), v.as_slice()));
                attention::qwen_attention(&self.ws.n1, wq, wk, wv, wo, &mut m.kv, &cfg)
            }
            AttnKind::Linear(_) | AttnKind::LinearGdn(_) => {
                unreachable!("MTP block is full attention")
            }
        };
        for (i, &a) in attn.iter().enumerate() {
            x[i] += a;
        }
        inference::rms_norm_into(&x, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p1);
        let ffn = ffn_forward(&lw.ffn, &self.ws.p1, self.pool.as_deref());
        for (i, &f) in ffn.iter().enumerate() {
            x[i] += f;
        }

        inference::rms_norm_into(&x, &m.final_norm, self.rms_eps, self.norm_style, &mut self.ws.n1);
        let mut lg = self.lm_head_forward(&self.ws.n1);
        let draft = sampler::argmax(&lg);
        attention::recycle_buf(&mut lg);
        draft
    }

    /// Micro-benchmark: two single-position forwards vs one fused pair
    /// from the current cache state (KV rewound after each probe).
    /// Returns (two_singles_ms, fused_pair_ms) per probe.
    pub fn measure_pair_fusion(&mut self, iters: usize) -> (f64, f64) {
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
    fn forward_pair(&mut self, emb1: &[f32], emb2: &[f32], position: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h1 = emb1.to_vec();
        let mut h2 = emb2.to_vec();
        let (nh, _nkv, _hd, hs, _rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let pool = self.pool.clone();

        for li in 0..self.num_layers {
            let lw = &self.weights.layers[li];
            // Norms into pipeline scratch (4 allocs/layer on the MTP
            // decode hot path before this).
            inference::rms_norm_into(&h1, &lw.input_norm, self.rms_eps, self.norm_style, &mut self.ws.n1);
            inference::rms_norm_into(&h2, &lw.input_norm, self.rms_eps, self.norm_style, &mut self.ws.n2);

            let (a1, a2) = match &lw.attn {
                AttnKind::Linear(w) => {
                    let cfg = self.vmf_cfg.expect("linear layer without vmf_cfg");
                    let layer = &mut self.kv_cache.layers[li];
                    let (state, scratch) = (&mut layer.linear_state, &mut layer.linear_scratch);
                    vmf_phase_pair(&self.ws.n1, &self.ws.n2, w, &cfg, state, scratch, self.pool.as_deref())
                }
                AttnKind::LinearGdn(w) => {
                    let cfg = self.gdn_cfg.expect("gdn layer without gdn_cfg");
                    let layer = &mut self.kv_cache.layers[li];
                    let (state, scratch) = (&mut layer.linear_state, &mut layer.linear_scratch);
                    gdn_pair(&self.ws.n1, &self.ws.n2, w, &cfg, state, scratch, self.pool.as_deref())
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    bias,
                } => {
                    let inv_freq_l = self.layer_inv_freq(li);
                    let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                    let cfg = QwenAttnCfg {
                        num_heads: nh,
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        window: self.layer_window(li),
                        v_norm: self.attn_v_norm,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
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
            let (a1, a2) = match &self.weights.layers[li].attn_out_norm {
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

            let lw = &self.weights.layers[li];
            inference::rms_norm_into(&h1, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p1);
            inference::rms_norm_into(&h2, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p2);
            let (f1, f2) =
                ffn_forward_pair(&lw.ffn, &self.ws.p1, &self.ws.p2, self.pool.as_deref());
            let (f1, f2) = match &self.weights.layers[li].ffn_out_norm {
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
            if let Some(sc) = self.weights.layers[li].layer_scale {
                for i in 0..self.hidden_size {
                    h1[i] *= sc;
                    h2[i] *= sc;
                }
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
        self.o1_begin();
        let mut hidden = vec![0.0f32; self.hidden_size];
        let mut pos = 0usize;
        if task_mask.is_none() && prefill_batched() && ids.len() > 2 {
            // prefill-GEMM in chunks; only the last position's hidden is
            // needed. (o1-compatible: the batch path attends per position
            // through qwen_attention, which carries the collection hook.)
            let chunk = prefill_chunk();
            let hs = self.hidden_size;
            while pos < ids.len() {
                let end = (pos + chunk).min(ids.len());
                let hb = self.prefill_batch(&ids[pos..end], pos);
                hidden.copy_from_slice(&hb[(end - pos - 1) * hs..]);
                pos = end;
            }
        }
        if task_mask.is_none() {
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
        FFN_PROBE.with(|p| {
            *p.borrow_mut() =
                Some(vec![vec![0f64; self.intermediate_size]; self.num_layers]);
        });
        crate::gpu::cpu_scope(|| {
            for (pos, &id) in ids.iter().enumerate() {
                let emb = self.embed_single(id);
                let _ = self.forward_layers(&emb, pos, None);
            }
        });
        self.kv_cache.clear();
        FFN_PROBE.with(|p| p.borrow_mut().take()).unwrap_or_default()
    }

    /// Teacher-forced PPL with a task mask active (sparse execution) —
    /// the quality gate for a DTG-MA-masked skill. Sequential per
    /// position: the batched prefill path is dense-only.
    pub fn ppl_ids_masked(&mut self, ids: &[u32], mask: &TaskMask) -> f64 {
        self.kv_cache.clear();
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
    pub fn nll_ids_from(&mut self, ids: &[u32], start: usize) -> (f64, usize) {
        self.kv_cache.clear();
        let mut nll = 0f64;
        let mut cnt = 0usize;
        if prefill_batched() {
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
                let hb = self.prefill_batch(&ids[pos..end], pos);
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
                    }
                    k0 = k1;
                }
                pos = end;
            }
            self.kv_cache.clear();
            return (nll, cnt);
        }
        for pos in 0..ids.len().saturating_sub(1) {
            let hidden = self.forward_layers(&self.embed_single(ids[pos]), pos, None);
            if pos < start {
                continue;
            }
            let normed = inference::rms_norm(
                &hidden,
                &self.weights.final_norm,
                self.rms_eps,
                self.norm_style,
            );
            let logits = self.lm_head_forward(&normed);
            let target = ids[pos + 1] as usize;
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln()
                + max as f64;
            nll += lse - logits[target] as f64;
            cnt += 1;
        }
        self.kv_cache.clear();
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
        self.o1_begin();
        let n = ids.len().saturating_sub(1);
        let p = prefill.min(n);
        // Exact prompt pass over ids[..p]: the seal consumes its q/k/v.
        let mut pos = 0usize;
        if prefill_batched() {
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
            let logits = self.lm_head_forward(&normed);
            let target = ids[pos + 1] as usize;
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln()
                + max as f64;
            nll += lse - logits[target] as f64;
            cnt += 1;
        }
        self.kv_cache.clear();
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
            let logits = self.lm_head_forward(&normed);
            let target = ids[pos + 1] as usize;
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln()
                + max as f64;
            nll += lse - logits[target] as f64;
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
        ((nll / cnt.max(1) as f64).exp(), switches)
    }

    /// Routing probe φ (spec §9): mean-pooled hidden after `layer`.
    pub fn probe_phi(&mut self, ids: &[u32], layer: usize) -> Vec<f32> {
        self.kv_cache.clear();
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
        acc
    }

    /// Layer-major batched prefill (prefill-GEMM): full-attention —
    /// per-position with the existing operators (KV grows naturally,
    /// causality preserved), GDN projections / FFN / MoE — batched
    /// (a weight row is read from DRAM once per chunk, not per
    /// position). Returns the hidden of all positions [b × hidden].
    fn prefill_batch(&mut self, ids: &[u32], start_pos: usize) -> Vec<f32> {
        let b = ids.len();
        let hs = self.hidden_size;
        // The CPU embed is deferred: when the chunk graph takes the run
        // from layer 0 it gathers the embeddings on the device instead.
        let mut h: Vec<f32> = vec![0.0; b * hs];
        let mut h_ready = false;
        let mut fill_h = |h: &mut Vec<f32>, me: &Self| {
            for (bi, &id) in ids.iter().enumerate() {
                let e = me.embed_single(id);
                h[bi * hs..(bi + 1) * hs].copy_from_slice(&e);
            }
        };
        let (nh, _nkv, _hd, _rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rms_eps,
        );
        let pool = self.pool.clone();
        let norm_style = self.norm_style;

        #[cfg(target_os = "macos")]
        let mut chunk_skip_until = 0usize;
        for li in 0..self.num_layers {
            crate::gpu::set_layer(li as i64); // layer-split GPU/CPU
            // GPU chunk graph (default-on under CMF_GPU=1): a run of
            // consecutive eligible layers for the whole chunk in ONE
            // Metal submission — norm, QKV, RoPE with fused mirror
            // append, causal attend, O, FFN, hidden device-resident
            // across the run. Any refusal falls through to the CPU path.
            #[cfg(target_os = "macos")]
            {
                if li < chunk_skip_until {
                    continue;
                }
                let ids_for_embed = (!h_ready && li == 0).then_some(ids);
                let end = self.chunk_run_gpu(li, &mut h, b, start_pos, ids_for_embed);
                if end > li {
                    h_ready = true;
                    chunk_skip_until = end;
                    continue;
                }
            }
            if !h_ready {
                fill_h(&mut h, self);
                h_ready = true;
            }
            let lw = &self.weights.layers[li];
            // ── attention ──
            match &lw.attn {
                AttnKind::LinearGdn(w) => {
                    // Projections batched, recurrence sequential.
                    let cfg = self.gdn_cfg.expect("gdn layer without gdn_cfg");
                    let mut normed = vec![0.0f32; b * hs];
                    for bi in 0..b {
                        let r = inference::rms_norm(
                            &h[bi * hs..(bi + 1) * hs], &lw.input_norm, eps, norm_style);
                        normed[bi * hs..(bi + 1) * hs].copy_from_slice(&r);
                    }
                    let attn = crate::linear_core::gdn_forward_batch(
                        &normed, b, w, &cfg,
                        &mut self.kv_cache.layers[li].linear_state,
                        pool.as_deref(),
                    );
                    for (dst, &a) in h.iter_mut().zip(&attn) {
                        *dst += a;
                    }
                }
                AttnKind::Full {
                    wq, wk, wv, wo, q_norm, k_norm, output_gate, bias,
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
                        num_heads: nh,
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position: start_pos,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        window: self.layer_window(li),
                        v_norm: self.attn_v_norm,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        bias: bias.as_ref().map(|(a, b, c)| {
                            (a.as_slice(), b.as_slice(), c.as_slice())
                        }),
                        rms_eps: eps,
                        norm_style,
                        pool: pool.as_deref(),
                    };
                    let mut attn = attention::qwen_attention_batch(
                        &normed, b, wq, wk, wv, wo,
                        &mut self.kv_cache.layers[li], &cfg);
                    if let Some(w) = &lw.attn_out_norm {
                        for bi in 0..b {
                            inference::rms_norm_into(
                                &attn[bi * hs..(bi + 1) * hs], w, eps, norm_style,
                                &mut normed[bi * hs..(bi + 1) * hs]);
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
                            &h[bi * hs..(bi + 1) * hs], &lw.input_norm, eps, norm_style);
                        vmf_phase_forward(
                            &normed, w,
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
            let lw = &self.weights.layers[li];
            let mut post = vec![0.0f32; b * hs];
            for bi in 0..b {
                let r = inference::rms_norm(
                    &h[bi * hs..(bi + 1) * hs], &lw.post_norm, eps, norm_style);
                post[bi * hs..(bi + 1) * hs].copy_from_slice(&r);
            }
            let mut ffn = match &lw.ffn {
                FfnKind::Dense(d) => dense_ffn_batch(d, &post, b, pool.as_deref()),
                FfnKind::Moe(m) => moe_ffn_batch(m, &post, b, hs, pool.as_deref()),
            };
            if let Some(w) = &lw.ffn_out_norm {
                for bi in 0..b {
                    inference::rms_norm_into(
                        &ffn[bi * hs..(bi + 1) * hs], w, eps, norm_style,
                        &mut post[bi * hs..(bi + 1) * hs]);
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
            if std::env::var("CMF_TRACE_H").is_ok() {
                let n = h[..hs].iter().map(|v| v.abs()).sum::<f32>() / hs as f32;
                let mx = h[..hs].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                eprintln!("layer {li}: mean|h|={n:.4} max|h|={mx:.2} scale={:?}", lw.layer_scale);
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
    ) -> usize {
        // (The old streaming attend needed a depth bound at ~1k; the
        // GEMM attention scales like the CPU path and lifted it.)
        // CMF_GPU_CHUNK=0 disables the graph.
        if !crate::gpu::enabled_here()
            || std::env::var("CMF_GPU_CHUNK").map(|v| v == "0").unwrap_or(false)
            || b < 32
            || self.swa.is_some()
            || self.global_attn.is_some()
            || self.attn_v_norm
            || (self.attn_scale - 1.0 / (self.head_dim as f32).sqrt()).abs() > 1e-9
        {
            return li0;
        }
        let Some(model) = self.model.clone() else { return li0 };
        let inv_freq = self.inv_freq.clone();
        let (nh, nkv, hd, hs) = (self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size);
        // Collect the longest run of consecutive eligible layers.
        let mut layers: Vec<crate::gpu_metal::ChunkLayer> = Vec::new();
        let mut stored_at: Vec<usize> = Vec::new();
        for li in li0..self.num_layers {
            let lw = &self.weights.layers[li];
            if lw.attn_out_norm.is_some() || lw.ffn_out_norm.is_some() || lw.layer_scale.is_some()
            {
                break;
            }
            let AttnKind::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate: false, bias } =
                &lw.attn
            else {
                break;
            };
            let FfnKind::Dense(d) = &lw.ffn else { break };
            if d.act != Act::Silu {
                break;
            }
            let parts = (
                wq.q8_row_parts(),
                wk.q8_row_parts(),
                wv.q8_row_parts(),
                wo.q8_row_parts(),
                d.gate_proj.q8_row_parts(),
                d.up_proj.q8_row_parts(),
                d.down_proj.q8_row_parts(),
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
            self.weights.embed_tokens.q8_row_parts().map(|(idx, rows, _c, rs)| {
                crate::gpu_metal::ChunkEmbed {
                    idx,
                    rows,
                    row_scale: rs,
                    ids,
                    mult: self.embed_multiplier,
                }
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
                layer.append(&ok[bi * row..(bi + 1) * row], &ov[bi * row..(bi + 1) * row], &[]);
            }
            layer.accumulate_imp(oi);
        }
        last
    }

    /// Is layer `li` a sliding-window (local-RoPE) layer? Gemma-3:
    /// every `pattern`-th layer is global, the rest are local.
    fn layer_is_local(&self, li: usize) -> bool {
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
        match self.swa {
            Some((w, _)) if self.layer_is_local(li) => Some(w),
            _ => None,
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
        (self.num_kv_heads, self.head_dim, self.rotary_dim)
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

    /// Same, stopping after layer `upto` inclusive (routing probe φ).
    fn forward_layers_upto(
        &mut self,
        hidden: &[f32],
        position: usize,
        task_mask: Option<&TaskMask>,
        upto: Option<usize>,
    ) -> Vec<f32> {
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

        #[cfg(target_os = "macos")]
        let mut gpu_skip_until = 0usize;
        for li in 0..self.num_layers {
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
                        continue;
                    }
                }
            }

            let lw = &self.weights.layers[li];
            // Norm into the pipeline scratch — the returning rms_norm
            // allocated twice per layer per token (roadmap §3 P0).
            inference::rms_norm_into(&h, &lw.input_norm, self.rms_eps, self.norm_style, &mut self.ws.n1);

            let attn_out = match &lw.attn {
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
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate,
                    bias,
                } if self.kv_cache.layers[li].o1_sealed() => {
                    // O(1) override: decode on the sealed Nyström state
                    // instead of the growing KV cache.
                    let inv_freq_l = self.layer_inv_freq(li);
                    let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
                    let cfg = QwenAttnCfg {
                        num_heads: nh,
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq_l,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        window: None,
                        v_norm: self.attn_v_norm,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
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
                    bias,
                } => {
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
                                num_heads: nh,
                                num_kv_heads: nkv_l,
                                head_dim: hd_l,
                                hidden_size: hs,
                                position,
                                inv_freq: &inv_freq_l,
                                rotary_dim: rd_l,
                                scale: self.attn_scale,
                                window: self.layer_window(li),
                                v_norm: self.attn_v_norm,
                                q_norm: q_norm.as_deref(),
                                k_norm: k_norm.as_deref(),
                                output_gate: *output_gate,
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
            let attn_out = match &self.weights.layers[li].attn_out_norm {
                Some(w) => inference::rms_norm(&attn_out, w, self.rms_eps, self.norm_style),
                None => attn_out,
            };
            for (i, &a) in attn_out.iter().enumerate() {
                h[i] += a;
            }
            let mut attn_out = attn_out;
            attention::recycle_buf(&mut attn_out);

            let lw = &self.weights.layers[li];
            inference::rms_norm_into(&h, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p1);
            let post_normed = &self.ws.p1;

            let ffn_masked = task_mask
                .map(|m| m.ffn_active_count(li) < self.intermediate_size)
                .unwrap_or(false);
            // Sparse mask path applies to dense f32 FFN only; MoE
            // layers route through the normal dispatch below.
            let f32_ffn = match &lw.ffn {
                FfnKind::Dense(d) => {
                    (d.gate_proj.as_f32(), d.up_proj.as_f32(), d.down_proj.as_f32())
                }
                FfnKind::Moe(_) => (None, None, None),
            };
            let ffn_out = match (ffn_masked, f32_ffn) {
                (true, (Some(g), Some(u), Some(d))) => {
                    let active = task_mask.unwrap().ffn_active_indices(li);
                    inference::sparse_ffn_forward(
                        &post_normed,
                        g,
                        u,
                        d,
                        self.hidden_size,
                        self.intermediate_size,
                        &active,
                        self.pool.as_deref(),
                    )
                }
                // Mask × quantized mmap: sparse FFN reads only active
                // neurons' rows/cols directly from the quant bytes — no
                // f32 model copy (a masked big model runs at quant RSS).
                (true, _) => match &lw.ffn {
                    FfnKind::Dense(d) if d.down_proj.sparse_col_ok() => {
                        let active = task_mask.unwrap().ffn_active_indices(li);
                        sparse_ffn_quant(
                            d,
                            &post_normed,
                            &active,
                            self.hidden_size,
                            self.pool.as_deref(),
                        )
                    }
                    // q4/vbit down_proj has no cheap column access → dequant
                    // the three matrices to f32 (transient) and run the f32
                    // sparse path. Correct (mask honored), just not
                    // memory-lean for those dtypes — a rare masked case.
                    FfnKind::Dense(d) => {
                        let active = task_mask.unwrap().ffn_active_indices(li);
                        let (gf, uf, df) = dequant_dense_f32(d);
                        inference::sparse_ffn_forward(
                            &post_normed,
                            &gf,
                            &uf,
                            &df,
                            self.hidden_size,
                            self.intermediate_size,
                            &active,
                            self.pool.as_deref(),
                        )
                    }
                    FfnKind::Moe(_) => {
                        // MoE is already sparse by expert selection; masks
                        // don't apply to routed experts.
                        ffn_forward(&lw.ffn, &post_normed, self.pool.as_deref())
                    }
                },
                (false, _) => ffn_forward(&lw.ffn, &post_normed, self.pool.as_deref()),
            };
            let ffn_out = match &self.weights.layers[li].ffn_out_norm {
                Some(w) => inference::rms_norm(&ffn_out, w, self.rms_eps, self.norm_style),
                None => ffn_out,
            };
            for (i, &f) in ffn_out.iter().enumerate() {
                h[i] += f;
            }
            let mut ffn_out = ffn_out;
            attention::recycle_buf(&mut ffn_out);

            // Gemma-4: the layer output is scaled by a learned scalar.
            if let Some(sc) = self.weights.layers[li].layer_scale {
                for v in h.iter_mut() {
                    *v *= sc;
                }
            }

            // Dynamic routing φ capture (on-policy, fireball-style): the
            // EMA of the post-residual hidden at the router's phi_layer,
            // updated as the context evolves during decode.
            if self.dyn_phi_layer == Some(li) {
                self.update_dyn_phi(&h);
            }
        }
        crate::gpu::set_layer(-1); // layers done — lm_head outside layer-split

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
        let Some(model) = &self.model else { return Vec::new() };
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
        let Some(model) = self.model.clone() else { return 0 };
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
                tracing::warn!(
                    "loaded skill is not FFN-eligible — dynamic routing unavailable"
                );
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
fn dense_ffn_batch(d: &DenseFfn, xs: &[f32], b: usize, pool: Option<&Pool>) -> Vec<f32> {
    let inter = d.gate_proj.rows();
    let hidden = d.down_proj.rows();
    let mut g = vec![0.0f32; b * inter];
    d.gate_proj.matmat(xs, b, &mut g, pool);
    let mut u = vec![0.0f32; b * inter];
    d.up_proj.matmat(xs, b, &mut u, pool);
    for i in 0..b * inter {
        g[i] = d.act.apply(g[i]) * u[i];
    }
    let mut out = vec![0.0f32; b * hidden];
    d.down_proj.matmat(&g, b, &mut out, pool);
    out
}

/// Batched MoE-FFN: router batched, positions are GROUPED by expert —
/// an expert's weights are read once for all its positions in the chunk
/// (the main prefill-GEMM win on MoE: 960MB/token of 35B experts).
fn moe_ffn_batch(m: &MoeFfn, xs: &[f32], b: usize, hidden: usize, pool: Option<&Pool>) -> Vec<f32> {
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; b * ne];
    m.router.matmat(xs, b, &mut logits, pool);

    // Assignments: expert → [(position, weight)] — the same top-k semantics
    // as moe_ffn (softmax over all, torch.topk order, optional renorm).
    let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
    {
        let mut st = m.stats.borrow_mut();
        if st.len() < ne {
            st.resize(ne, 0);
        }
        for bi in 0..b {
            let lg = &logits[bi * ne..(bi + 1) * ne];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = lg.iter().map(|&l| (l - mx).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_unstable_by(|&x, &y| p[y].partial_cmp(&p[x]).unwrap().then(x.cmp(&y)));
            order.truncate(m.top_k);
            let wsum: f32 = if m.norm_topk_prob {
                order.iter().map(|&e| p[e]).sum()
            } else {
                1.0
            };
            for &e in &order {
                st[e] += 1;
                assign[e].push((bi, p[e] / wsum));
            }
        }
    }

    let mut out = vec![0.0f32; b * hidden];
    let cols = m.experts[0].gate_proj.cols();
    let mut run_expert = |d: &DenseFfn, list: &[(usize, f32)]| {
        let sb = list.len();
        let mut sub = vec![0.0f32; sb * cols];
        for (k, &(bi, _)) in list.iter().enumerate() {
            sub[k * cols..(k + 1) * cols].copy_from_slice(&xs[bi * cols..(bi + 1) * cols]);
        }
        let eo = dense_ffn_batch(d, &sub, sb, pool);
        for (k, &(bi, w)) in list.iter().enumerate() {
            for i in 0..hidden {
                out[bi * hidden + i] += w * eo[k * hidden + i];
            }
        }
    };
    for e in 0..ne {
        if !assign[e].is_empty() {
            run_expert(&m.experts[e], &assign[e]);
        }
    }
    if let Some((se, gate)) = &m.shared {
        let mut gl = vec![0.0f32; b];
        gate.matmat(xs, b, &mut gl, pool);
        let all: Vec<(usize, f32)> = (0..b)
            .map(|bi| (bi, 1.0 / (1.0 + (-gl[bi]).exp())))
            .collect();
        run_expert(se, &all);
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
        u.resize(inter, 0.0);
        // Multi-matrix job: gate+up under one pool dispatch.
        QTensor::matvec_many([&d.gate_proj, &d.up_proj], x, [g, u], pool);
        for i in 0..inter {
            g[i] = d.act.apply(g[i]) * u[i];
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
fn moe_parts(
    t: &QTensor,
) -> Option<(&std::sync::Arc<cortiq_core::CmfModel>, usize, usize, usize, &[f32], &[f32], bool)> {
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
        } if (*dt == cortiq_core::TensorDtype::Q8Row) || !col_field.is_empty() => {
            Some((model, *idx, *rows, *cols, row_scale, col_field, false))
        }
        // q1: tile-embedded scales — empty rs/col slices, raw xs.
        QTensor::Mapped {
            model,
            idx,
            dtype: cortiq_core::TensorDtype::Q1,
            rows,
            cols,
            ..
        } => Some((model, *idx, *rows, *cols, &[][..], &[][..], true)),
        _ => None,
    }
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
    let (gm, gi, gr, gc, grs, gcf, gq1) = moe_parts(&d.gate_proj)?;
    let (_, ui, ur, uc, urs, ucf, uq1) = moe_parts(&d.up_proj)?;
    let (_, di, dr, dc, drs, dcf, dq1) = moe_parts(&d.down_proj)?;
    if gq1 != uq1 || uq1 != dq1 {
        return None; // mixed-dtype trio — honest CPU path
    }
    model_ref.get_or_insert_with(|| gm.clone());
    let gdt = if gcf.is_empty() { cortiq_core::TensorDtype::Q8Row } else { cortiq_core::TensorDtype::Q8_2f };
    let udt = if ucf.is_empty() { cortiq_core::TensorDtype::Q8Row } else { cortiq_core::TensorDtype::Q8_2f };
    jobs.push(crate::gpu::MoeJob {
        gate: (gi, gr, gc, grs),
        up: (ui, ur, uc, urs),
        down: (di, dr, dc, drs),
        xs_gate: prescale(x, gcf, gdt).into_owned(),
        xs_up: prescale(x, ucf, udt).into_owned(),
        down_col: dcf,
        w,
        q1: gq1,
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
        let mut s = if need_scratch { vec![0.0f32; hidden] } else { Vec::new() };
        let gate = d.gate_proj.row_dot(idx, x, &mut s);
        let up = d.up_proj.row_dot(idx, x, &mut s);
        d.act.apply(gate) * up
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

/// MoE FFN: router softmax over ALL experts → top-k (HF Qwen2/3-MoE
/// semantics: probabilities BEFORE selection; optional renorm of the
/// selected k). Only selected experts' pages are touched in mmap.
fn moe_ffn(m: &MoeFfn, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; ne];
    m.router.matvec(x, &mut logits, pool);
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = p.iter().sum();
    for v in &mut p {
        *v /= s;
    }
    let mut idx: Vec<usize> = (0..ne).collect();
    // Descending by prob, lower index wins ties — torch.topk order.
    idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(m.top_k);
    let wsum: f32 = if m.norm_topk_prob {
        idx.iter().map(|&e| p[e]).sum()
    } else {
        1.0
    };
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

/// The pure-CPU MoE expert loop (also the fallback of every GPU refusal).
fn moe_ffn_cpu(
    m: &MoeFfn,
    x: &[f32],
    idx: &[usize],
    p: &[f32],
    wsum: f32,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = attention::take_buf(x.len());
    for &e in idx {
        let mut eo = dense_ffn(&m.experts[e], x, pool);
        let w = p[e] / wsum;
        for i in 0..out.len() {
            out[i] += w * eo[i];
        }
        attention::recycle_buf(&mut eo);
    }
    if let Some((se, gate)) = &m.shared {
        let mut so = dense_ffn(se, x, pool);
        let mut gl = vec![0.0f32; 1];
        gate.matvec(x, &mut gl, pool);
        let g = 1.0 / (1.0 + (-gl[0]).exp());
        for i in 0..out.len() {
            out[i] += g * so[i];
        }
        attention::recycle_buf(&mut so);
    }
    out
}

/// Building the MoE-layer GPU jobs: all selected experts (+shared) must
/// be q8_2f-Mapped from the primary mapping; otherwise None → CPU path.
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
        moe_push_job(&m.experts[e], x, p[e] / wsum, &mut jobs, &mut model_ref)?;
    }
    if let Some((se, gate)) = &m.shared {
        let mut gl = vec![0.0f32; 1];
        gate.matvec(x, &mut gl, pool);
        let g = 1.0 / (1.0 + (-gl[0]).exp());
        moe_push_job(se, x, g, &mut jobs, &mut model_ref)?;
    }
    let model = model_ref?;
    let hidden = jobs[0].down.1;
    let mut out = vec![0.0f32; hidden];
    crate::gpu::moe_block(&model, &jobs, &mut out).then_some(out)
}

/// Single-position FFN dispatch.
fn ffn_forward(ffn: &FfnKind, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    match ffn {
        FfnKind::Dense(d) => dense_ffn(d, x, pool),
        FfnKind::Moe(m) => moe_ffn(m, x, pool),
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
) -> (Vec<f32>, Vec<f32>) {
    let d = match ffn {
        FfnKind::Dense(d) => d,
        FfnKind::Moe(m) => return (moe_ffn(m, x1, pool), moe_ffn(m, x2, pool)),
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
            g1[i] = d.act.apply(g1[i]) * u1[i];
            g2[i] = d.act.apply(g2[i]) * u2[i];
        }
        let mut o1 = attention::take_buf(d.down_proj.rows());
        let mut o2 = attention::take_buf(d.down_proj.rows());
        d.down_proj.matvec2(g1, g2, &mut o1, &mut o2, pool);
        (o1, o2)
    })
}

#[cfg(test)]
mod tests {
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
                },
            },
            final_norm: vec![1.0; h],
            kv: crate::kv_cache::LayerKvCache::new(kv, hd),
        });
    }

    #[test]
    fn speculative_equals_vanilla_greedy() {
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
        assert_eq!(argmax, r.token_ids[0], "explain preview must match greedy emit");
    }
}
