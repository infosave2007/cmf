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
    pub tokenizer: Tokenizer,
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
    pub post_norm: Vec<f32>,
    pub ffn: FfnKind,
    pub attn: AttnKind,
}

/// Dense SwiGLU triple — the FFN of a dense layer or of one expert.
pub struct DenseFfn {
    pub gate_proj: QTensor,
    pub up_proj: QTensor,
    pub down_proj: QTensor,
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

/// Callback for streaming tokens. Return `false` to cancel.
pub type TokenCallback = Box<dyn FnMut(&str) -> bool + Send>;

impl Pipeline {
    /// Build a pipeline from parts (used by the loader and tests).
    #[allow(clippy::too_many_arguments)]
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
            tokenizer,
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
        let input_ids = self.tokenizer.encode(prompt);
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
        // With dynamic routing, prefill sequentially so the φ hook fires
        // over the PROMPT — the router enters decode with a warm φ (the
        // fused-pair path skips the per-layer φ capture). o1 layers
        // collect their query trace in both the single and pair paths.
        let dyn_prefill = router.is_some();
        if task_mask.is_none() && !dyn_prefill {
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
            inference::rms_norm_into(
                &hidden,
                &self.weights.final_norm,
                self.rms_eps,
                self.norm_style,
                &mut self.ws.n1,
            );
            let logits = self.lm_head_forward(&self.ws.n1);
            let t_next = sampler::sample(&logits, &self.sampler_config, &all_ids, &mut self.rng);
            confidence.push(top1_prob_t(&logits, t_next, calib_temp));
            if trace_on {
                // active_skill = the overlay in force while this token was
                // generated; recon/switched are filled after the post-emit
                // routing eval below (freshest coherence for this token).
                let skill = router.as_ref().and_then(|r| r.active_id());
                traces.push(TokenTrace {
                    t: generated,
                    token_id: t_next,
                    confidence: *confidence.last().unwrap(),
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
                    let logits1 = self.lm_head_forward(&self.ws.n1);
                    let t_after =
                        sampler::sample(&logits1, &self.sampler_config, &all_ids, &mut self.rng);
                    confidence.push(top1_prob_t(&logits1, t_after, calib_temp));
                    if trace_on {
                        // Speculative decode is mutually exclusive with
                        // dynamic routing (router is None here) — no skill.
                        traces.push(TokenTrace {
                            t: generated,
                            token_id: t_after,
                            confidence: *confidence.last().unwrap(),
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
        sampler::argmax(&self.lm_head_forward(&self.ws.n1))
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
        let (nh, nkv, hd, hs, rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let inv_freq = self.inv_freq.clone();
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
                    let cfg = QwenAttnCfg {
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq,
                        rotary_dim: rd,
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
            for i in 0..self.hidden_size {
                h1[i] += a1[i];
                h2[i] += a2[i];
            }

            let lw = &self.weights.layers[li];
            inference::rms_norm_into(&h1, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p1);
            inference::rms_norm_into(&h2, &lw.post_norm, self.rms_eps, self.norm_style, &mut self.ws.p2);
            let (f1, f2) =
                ffn_forward_pair(&lw.ffn, &self.ws.p1, &self.ws.p2, self.pool.as_deref());
            for i in 0..self.hidden_size {
                h1[i] += f1[i];
                h2[i] += f2[i];
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
            const CHUNK: usize = 48;
            let hs = self.hidden_size;
            while pos < ids.len() {
                let end = (pos + CHUNK).min(ids.len());
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
        let mut h: Vec<f32> = Vec::with_capacity(b * hs);
        for &id in ids {
            h.extend_from_slice(&self.embed_single(id));
        }
        let (nh, nkv, hd, rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rms_eps,
        );
        let inv_freq = self.inv_freq.clone();
        let pool = self.pool.clone();
        let norm_style = self.norm_style;

        for li in 0..self.num_layers {
            crate::gpu::set_layer(li as i64); // layer-split GPU/CPU
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
                _ => {
                    for bi in 0..b {
                        let normed = inference::rms_norm(
                            &h[bi * hs..(bi + 1) * hs], &lw.input_norm, eps, norm_style);
                        let position = start_pos + bi;
                        let attn = match &lw.attn {
                            AttnKind::Linear(w) => {
                                let cfg = self.vmf_cfg.expect("linear layer without vmf_cfg");
                                vmf_phase_forward(
                                    &normed, w, &cfg,
                                    &mut self.kv_cache.layers[li].linear_state,
                                    pool.as_deref(),
                                )
                            }
                            AttnKind::LinearGdn(_) => unreachable!(),
                            AttnKind::Full {
                                wq, wk, wv, wo, q_norm, k_norm, output_gate, bias,
                            } => {
                                let cfg = QwenAttnCfg {
                                    num_heads: nh,
                                    num_kv_heads: nkv,
                                    head_dim: hd,
                                    hidden_size: hs,
                                    position,
                                    inv_freq: &inv_freq,
                                    rotary_dim: rd,
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
                                attention::qwen_attention(
                                    &normed, wq, wk, wv, wo,
                                    &mut self.kv_cache.layers[li], &cfg)
                            }
                        };
                        for (i, &a) in attn.iter().enumerate() {
                            h[bi * hs + i] += a;
                        }
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
            let ffn = match &lw.ffn {
                FfnKind::Dense(d) => dense_ffn_batch(d, &post, b, pool.as_deref()),
                FfnKind::Moe(m) => moe_ffn_batch(m, &post, b, hs, pool.as_deref()),
            };
            for (dst, &f) in h.iter_mut().zip(&ffn) {
                *dst += f;
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
        out
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
        let (nh, nkv, hd, hs, rd, eps) = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.hidden_size,
            self.rotary_dim,
            self.rms_eps,
        );
        let inv_freq = self.inv_freq.clone();
        let pool = self.pool.clone();

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
                    let cfg = QwenAttnCfg {
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        hidden_size: hs,
                        position,
                        inv_freq: &inv_freq,
                        rotary_dim: rd,
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
                            let cfg = QwenAttnCfg {
                                num_heads: nh,
                                num_kv_heads: nkv,
                                head_dim: hd,
                                hidden_size: hs,
                                position,
                                inv_freq: &inv_freq,
                                rotary_dim: rd,
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
            for (i, &a) in attn_out.iter().enumerate() {
                h[i] += a;
            }

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
            for (i, &f) in ffn_out.iter().enumerate() {
                h[i] += f;
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
        let mut logits = vec![0.0f32; rows.min(self.vocab_size)];
        self.weights
            .lm_head
            .matvec(hidden, &mut logits, self.pool.as_deref());
        logits.resize(self.vocab_size, 0.0);
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
            ffn: FfnKind::Dense(DenseFfn {
                gate_proj: qt(intermediate_size, hidden_size, li * 10 + 5),
                up_proj: qt(intermediate_size, hidden_size, li * 10 + 6),
                down_proj: qt(hidden_size, intermediate_size, li * 10 + 7),
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
        g[i] = inference::silu(g[i]) * u[i];
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
    let inter = d.gate_proj.rows();
    FFN_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let [g, u, ..] = &mut *s;
        g.resize(inter, 0.0);
        u.resize(inter, 0.0);
        d.gate_proj.matvec(x, g, pool);
        d.up_proj.matvec(x, u, pool);
        for i in 0..inter {
            g[i] = inference::silu(g[i]) * u[i];
        }
        let mut out = vec![0.0f32; d.down_proj.rows()];
        d.down_proj.matvec(g, &mut out, pool);
        out
    })
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
        inference::silu(gate) * up
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
    if crate::gpu::enabled_here() {
        if let Some(out) = moe_ffn_gpu(m, x, &idx, &p, wsum, pool) {
            return out;
        }
    }

    let mut out = vec![0.0f32; x.len()];
    for &e in &idx {
        let eo = dense_ffn(&m.experts[e], x, pool);
        let w = p[e] / wsum;
        for i in 0..out.len() {
            out[i] += w * eo[i];
        }
    }
    if let Some((se, gate)) = &m.shared {
        let so = dense_ffn(se, x, pool);
        let mut gl = vec![0.0f32; 1];
        gate.matvec(x, &mut gl, pool);
        let g = 1.0 / (1.0 + (-gl[0]).exp());
        for i in 0..out.len() {
            out[i] += g * so[i];
        }
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
    use crate::qtensor::prescale;
    use cortiq_core::TensorDtype;

    fn parts(
        t: &QTensor,
    ) -> Option<(&std::sync::Arc<cortiq_core::CmfModel>, usize, usize, usize, &[f32], &[f32])>
    {
        match t {
            QTensor::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q8_2f,
                rows,
                cols,
                row_scale,
                col_field,
                ..
            } if !col_field.is_empty() => {
                Some((model, *idx, *rows, *cols, row_scale, col_field))
            }
            _ => None,
        }
    }

    fn push<'a>(
        d: &'a DenseFfn,
        x: &[f32],
        w: f32,
        jobs: &mut Vec<MoeJob<'a>>,
        model_ref: &mut Option<std::sync::Arc<cortiq_core::CmfModel>>,
    ) -> Option<()> {
        let (gm, gi, gr, gc, grs, gcf) = parts(&d.gate_proj)?;
        let (_, ui, ur, uc, urs, ucf) = parts(&d.up_proj)?;
        let (_, di, dr, dc, drs, dcf) = parts(&d.down_proj)?;
        model_ref.get_or_insert_with(|| gm.clone());
        jobs.push(MoeJob {
            gate: (gi, gr, gc, grs),
            up: (ui, ur, uc, urs),
            down: (di, dr, dc, drs),
            xs_gate: prescale(x, gcf, TensorDtype::Q8_2f).into_owned(),
            xs_up: prescale(x, ucf, TensorDtype::Q8_2f).into_owned(),
            down_col: dcf,
            w,
        });
        Some(())
    }

    let mut jobs: Vec<MoeJob> = Vec::with_capacity(idx.len() + 1);
    let mut model_ref = None;
    for &e in idx {
        push(&m.experts[e], x, p[e] / wsum, &mut jobs, &mut model_ref)?;
    }
    if let Some((se, gate)) = &m.shared {
        let mut gl = vec![0.0f32; 1];
        gate.matvec(x, &mut gl, pool);
        let g = 1.0 / (1.0 + (-gl[0]).exp());
        push(se, x, g, &mut jobs, &mut model_ref)?;
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
        d.gate_proj.matvec2(x1, x2, g1, g2, pool);
        d.up_proj.matvec2(x1, x2, u1, u2, pool);
        for i in 0..inter {
            g1[i] = inference::silu(g1[i]) * u1[i];
            g2[i] = inference::silu(g2[i]) * u2[i];
        }
        let mut o1 = vec![0.0f32; d.down_proj.rows()];
        let mut o2 = vec![0.0f32; d.down_proj.rows()];
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
                ffn: FfnKind::Dense(DenseFfn {
                    gate_proj: qt(inter, h, 315),
                    up_proj: qt(inter, h, 316),
                    down_proj: qt(h, inter, 317),
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
