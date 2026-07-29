//! Core types for CMF format.

use serde::{Deserialize, Serialize};

/// Per-tensor storage dtype. IDs are shared with the `.vmfc` directory
/// encoding and must never be reused for a different meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TensorDtype {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
    Q8Row = 3,
    Q4Block = 4,
    Mix84 = 5,
    U8 = 6,
    Q4Col = 7,
    Vbit = 8,
    Q8_2f = 9,
    /// Vbit with an explicit row-offset table (roadmap §4.2): same bit
    /// packing and grouped f16 scales as `Vbit`, plus
    /// `row_offsets[rows+1]` (u32, relative to the packed area) between
    /// the scales and the packed rows — O(1) row access straight from
    /// the file, no load-time prefix scan. New id on purpose: the byte
    /// semantics of `Vbit = 8` must never change.
    VbitRo = 10,
    /// q4 with interleaved per-group tiles (roadmap §4.3):
    /// `repeat per 32-group { f16 scale; 16B packed nibbles }` — one
    /// sequential memory stream instead of two distant ones (nibbles…,
    /// scales…). Measured kernel-level: ×1.66 on Apple Silicon, ×1.13
    /// on AVX2. 2-D tensors with cols % 32 == 0 only. New id — the
    /// byte semantics of `Q4Block = 4` never change.
    Q4Tiled = 11,
    /// 1-bit binary weights (roadmap: 1-bit-TRAINED models — Bonsai /
    /// BitNet class; as post-training quantization of a normal model
    /// this destroys quality, so converters expose it only as an
    /// explicit opt-in). Tiled like `q4_tiled`: `repeat per 32-group
    /// { f16 scale; 4B bits }` — 6 bytes per 32 weights (1.5 bits/w),
    /// one sequential stream. Bit k of byte j (LSB-first) is weight
    /// j·8+k of the group; value = scale · (2·bit − 1) ∈ {−s, +s}.
    /// 2-D tensors with cols % 32 == 0 only.
    Q1 = 12,
    /// 1-bit PTQ with a sparse high-precision outlier overlay (holographic
    /// transfer / SpQR-style): a `Q1` base (per 32-group `[f16 scale][4B
    /// bits]`, outliers excluded from the scale) followed by
    /// `[u32 count]` then `count × [u32 flat-index][f16 value]` — the
    /// salient weights the two-field mask kept at full precision, restored
    /// verbatim at dequant. Variable length (the count self-describes), so
    /// `expected_nbytes` returns None and the reader trusts the stored
    /// span. Lets a NORMAL checkpoint survive 1-bit where plain `q1` cannot.
    Q1S = 13,
    /// Ternary (BitNet b1.58-style) `{−s, 0, +s}` with a sparse outlier
    /// overlay. Per 32-group `[f16 scale][7B : base-3 codes, 5 ternary
    /// values/byte since 3^5 = 243 ≤ 256]` (code 0 → 0, 1 → +s, 2 → −s;
    /// ~2.25 bpw) then a per-row overlay `[u32 row_ptr[rows+1]][(u16 col,
    /// f16 value)]` grouped by row (row `r`'s outliers are
    /// `[row_ptr[r], row_ptr[r+1])`; `col` is a within-row index, so `cols`
    /// must fit `u16`) — 4 B/outlier, no binary search. Capturing the many
    /// near-zero weights exactly is the decisive PTQ win over binary. Variable
    /// length.
    Q1T = 14,
    /// q4 tiled with PREDICTED scales: the same nibbles as `Q4Tiled`, but
    /// a tile's scale is a 5-bit code into a per-row geometric ladder
    /// instead of a standalone f16 — 4.17 bits/weight against 4.50, i.e.
    /// 7.3% off every q4t tensor.
    ///
    /// `[nibbles: rows·gpr·16][row params: rows × (f16 lo, f16 step)]
    ///  [codes: rows × ceil(gpr·5/8), 5-bit LSB-first, row-aligned]`
    ///
    /// Scale of tile `g` in row `r` is `2^(lo[r] + code·step[r])`, so a
    /// kernel expands one row's 32-entry ladder once (a single exp2 and
    /// 31 multiplies) and then reads scales by table lookup. `lo`/`step`
    /// come from the row's exact min/max log-scale, so no code is ever out
    /// of range and the format needs no escape hatch.
    ///
    /// Quantizing the SAME fp32 weights both ways, q4tp's error against the
    /// source is 9.71% where q4t's is 9.71% — a 0.1% relative increase at
    /// the median within-row scale spread measured on KAT-Coder-V2.5 (1.27
    /// in log2), 0.3% at its 90th percentile, 0.8% past the tail. A coarser
    /// scale barely matters because the nibbles simply re-round against it;
    /// the 4-bit grid dominates the error either way. (Comparing q4tp to
    /// q4t's OUTPUT instead reads ~1.1% RMS, but that measures the distance
    /// between two representations, not a loss of accuracy.)
    ///
    /// A per-tensor Lloyd codebook was measured and rejected (0.0576% vs
    /// 0.0591% NMSE at 4 bits): within a row the log-scales are close to
    /// uniform, so the ladder gains nothing from being non-uniform.
    /// Nibbles are 16 B and therefore BETTER aligned than q4t's 18 B
    /// stride. 2-D tensors with cols % 32 == 0 only.
    Q4TiledP = 15,
}

impl TensorDtype {
    pub fn from_id(id: u8) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Bf16,
            3 => Self::Q8Row,
            4 => Self::Q4Block,
            5 => Self::Mix84,
            6 => Self::U8,
            7 => Self::Q4Col,
            8 => Self::Vbit,
            9 => Self::Q8_2f,
            10 => Self::VbitRo,
            11 => Self::Q4Tiled,
            12 => Self::Q1,
            13 => Self::Q1S,
            14 => Self::Q1T,
            15 => Self::Q4TiledP,
            _ => return None,
        })
    }

    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8Row => "q8_row",
            Self::Q4Block => "q4_block",
            Self::Mix84 => "mix8_4",
            Self::U8 => "u8",
            Self::Q4Col => "q4_col",
            Self::Vbit => "vbit",
            Self::Q8_2f => "q8_2f",
            Self::VbitRo => "vbit_ro",
            Self::Q4Tiled => "q4_tiled",
            Self::Q1 => "q1",
            Self::Q1S => "q1s",
            Self::Q1T => "q1t",
            Self::Q4TiledP => "q4tp",
        }
    }

    /// Dtypes the current runtime can decode into f32.
    /// (Vbit was missing here long after the fused kernels and
    /// `dequant_vbit` shipped — roadmap §4.9.)
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::Bf16
                | Self::Q8Row
                | Self::Q4Block
                | Self::Q8_2f
                | Self::Vbit
                | Self::VbitRo
                | Self::Q4Tiled
                | Self::Q1
                | Self::Q1S
                | Self::Q1T
                | Self::Q4TiledP
        )
    }
}

/// File-level default quantization (informational; per-tensor truth
/// lives in the tensor directory).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantType {
    #[serde(rename = "Q8_ROW")]
    Q8Row,
    #[serde(rename = "Q4_BLOCK")]
    Q4Block,
    #[serde(rename = "Q8_2F")]
    Q8_2f,
    #[serde(rename = "VBIT")]
    Vbit,
    BF16,
    F16,
    F32,
}

/// RMS-norm weight semantics. Gemma applies `x̂·(1+w)`, Qwen/Llama `x̂·w`.
/// Getting this wrong corrupts every normalization in the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NormStyle {
    #[default]
    Qwen,
    Gemma,
}

/// Layer type in hybrid architecture (Qwen3.5-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Standard multi-head attention (Q/K/V/O projections)
    FullAttention,
    /// Standard GQA restricted to a causal sliding window. It remains a
    /// full-attention operator; the distinct tag preserves an explicit,
    /// potentially irregular per-layer schedule (Laguna).
    SlidingAttention,
    /// Linear attention (executed by the canonical linear core;
    /// original operator, e.g. GatedDeltaNet, is folded at convert time)
    LinearAttention,
    /// Gated short convolution mixer (LFM2 / LFM2-MoE): in_proj → (B,C,x)
    /// gates + a causal depthwise conv1d over recent tokens, no KV cache.
    /// `linear_conv_kernel_dim` carries the kernel width.
    ShortConv,
    /// Kimi Delta Attention (Kimi Linear / Kimi-K3): delta rule with a
    /// PER-CHANNEL log decay (diagonal, vs GDN's per-head scalar),
    /// separate q/k/v projections each behind its own causal short
    /// convolution, and a sigmoid-gated output RMSNorm. Geometry rides
    /// the `linear_*` fields; state = 3 conv rings + S[h·dk·dv].
    Kda,
}

/// Multi-token-prediction head carried by the file (DeepSeek/Qwen-MTP
/// style). Tensors live under `model.mtp.*` (see spec §2.1); the head
/// shares the embedding and lm_head with the main model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpConfig {
    /// Number of MTP transformer blocks (`model.mtp.layers.{i}.*`).
    pub num_layers: usize,
    #[serde(default = "default_true")]
    pub share_lm_head: bool,
    #[serde(default = "default_true")]
    pub share_embed: bool,
}

fn default_true() -> bool {
    true
}

/// Mixture-of-Experts FFN descriptor (Qwen2-MoE / Qwen3-MoE family).
/// Which layers are MoE is decided by tensor presence: a layer is MoE
/// iff its router `model.layers.{i}.mlp.gate.weight` is in the
/// directory (dense fallback layers keep `mlp.{gate,up,down}_proj`).
/// Experts are ordinary directory entries
/// (`…mlp.experts.{e}.{gate,up,down}_proj.weight`) — each may carry its
/// own dtype, which is what per-expert bit allocation (P15 claim 12)
/// rides on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: usize,
    /// Experts activated per token (`num_experts_per_tok`).
    pub top_k: usize,
    /// Intermediate size of each routed expert.
    pub moe_intermediate_size: usize,
    /// Renormalize the top-k probabilities to sum to 1 (Qwen3-MoE).
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Qwen2-MoE always-on shared expert (None = absent, Qwen3-MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_intermediate_size: Option<usize>,
    /// Router scores each expert with a sigmoid instead of a softmax over
    /// all experts (LFM2-MoE / DeepSeek-V3 `noaux_tc`). Selection may add a
    /// per-expert bias (`mlp.expert_bias`) for the top-k choice while the
    /// gathered weights come from the *unbiased* sigmoid scores. False =
    /// classic Qwen softmax routing (bit-identical to the historical path).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub router_sigmoid: bool,
    /// Top-k weights are multiplied by this after the optional renorm
    /// (LFM2-MoE `routed_scaling_factor`). None = 1.0 (no scaling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_scaling_factor: Option<f32>,
}

/// Optional YaRN parameters for the model's global RoPE profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YarnConfig {
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_yarn_beta_fast")]
    pub beta_fast: f32,
    #[serde(default = "default_yarn_beta_slow")]
    pub beta_slow: f32,
    #[serde(default = "default_one")]
    pub attention_factor: f32,
    /// DeepSeek-V2 YaRN: softmax-scale correction exponent base
    /// (`mscale_all_dim`); the loader folds
    /// (0.1·m·ln(factor)+1)² into the attention scale. 0 = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mscale_all_dim: Option<f32>,
}

/// DeepSeek-V2 Multi-head Latent Attention geometry (spec: additive
/// header block; tensors ride under their HF names).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlaConfig {
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    /// Present on the big V2/V3 (compressed q); Lite projects q directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_lora_rank: Option<usize>,
    /// Kimi Linear NoPE: full-attention layers apply NO rotary at all
    /// (the KDA layers carry position). The [rope|nope] head layout is
    /// kept; only the rotation is skipped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub nope: bool,
}

fn default_yarn_beta_fast() -> f32 {
    32.0
}

fn default_yarn_beta_slow() -> f32 {
    1.0
}

/// Gemma-3n (E-series) stack geometry: AltUp replicas, LAuReL rank,
/// per-layer embeddings, KV sharing and the activation-sparsity
/// schedule. Presence of this block switches the runtime to the
/// dedicated gemma-3n forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3nConfig {
    pub altup_num_inputs: usize,
    pub laurel_rank: usize,
    /// hidden_size_per_layer_input (PLE width per layer).
    pub ple_dim: usize,
    /// vocab_size_per_layer_input — ids past it contribute zero PLE.
    pub ple_vocab: usize,
    /// The LAST n layers reuse the KV of the last non-shared layer of
    /// their own attention type.
    pub num_kv_shared_layers: usize,
    /// Per-layer gaussian-top-k sparsity (0 = dense).
    pub activation_sparsity: Vec<f32>,
}

/// Model architecture descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArch {
    /// Architecture name (e.g., "qwen3.5", "llama", "mistral")
    pub arch_name: String,
    /// Hidden dimension
    pub hidden_size: usize,
    /// FFN intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads (for full_attention layers)
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Per-layer type schedule
    pub layer_types: Vec<LayerType>,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RMS-norm weight semantics
    #[serde(default)]
    pub norm_style: NormStyle,
    /// RoPE base frequency
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// lm_head shares the embedding matrix
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Fraction of head_dim rotated by RoPE (Qwen3.5: 0.25)
    #[serde(default = "default_prf")]
    pub partial_rotary_factor: f32,
    /// Optional YaRN frequency interpolation for the global RoPE profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yarn: Option<YarnConfig>,
    /// Per-layer Q-head counts for architectures whose attention width varies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_heads_per_layer: Option<Vec<usize>>,
    /// FFN activation: "silu" (default) or "gelu_tanh" (Gemma's GeGLU).
    #[serde(default = "default_hidden_act", skip_serializing_if = "is_default_act")]
    pub hidden_act: String,
    /// Token embeddings are multiplied by this at input (Gemma: √hidden).
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub embed_multiplier: f32,
    /// Attention scale = 1/√this (None → 1/√head_dim). Gemma family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Sliding-window attention (Gemma-3): window width…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<usize>,
    /// …the every-Nth-layer-is-global pattern…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// …and the local layers' own RoPE base (global layers use rope_theta).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base_freq: Option<f64>,
    /// Local/SWA layers may rotate a different fraction of each head.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_partial_rotary_factor: Option<f32>,
    /// Gemma-4: global (full-attention) layers use their own head dim…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// …their own KV head count (1 = MQA)…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// …and a proportional partial rotary: the first `factor·head_dim`
    /// dims rotate, the rest ride at angle 0 (identity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_partial_rotary_factor: Option<f32>,
    /// Final-logit soft-capping: logits = C·tanh(logits/C) (Gemma-4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,
    /// DeepSeek-V2 MLA block; None for every other family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla: Option<MlaConfig>,
    /// Kimi-K3 situ activation betas (hidden_act == "situ").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_situ_beta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_situ_linear_beta: Option<f64>,
    /// Gemma-2: attention scores pass tanh(s/c)·c before the causal
    /// softmax (attn_logit_softcapping). None = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_logit_softcapping: Option<f64>,
    /// Scale-less RMS normalization of V heads before caching (Gemma-4).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub attn_v_norm: bool,
    /// Multi-token-prediction head (None = absent)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpConfig>,
    /// Mixture-of-Experts FFN (None = all-dense model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeConfig>,
    /// Canonical linear core carried by the file (None = no linear layers
    /// or not folded yet)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_core: Option<LinearCoreConfig>,
    /// Max position embeddings
    pub max_position_embeddings: usize,
    // Linear attention specific
    /// Conv kernel dim for linear attention layers
    pub linear_conv_kernel_dim: Option<usize>,
    /// Number of key heads in linear attention
    pub linear_num_key_heads: Option<usize>,
    /// Number of value heads in linear attention
    pub linear_num_value_heads: Option<usize>,
    /// Key head dim in linear attention
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_key_head_dim: Option<usize>,
    /// Value head dim in linear attention
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_value_head_dim: Option<usize>,
    /// Per-frequency rope divisors (MiniCPM3 longrope short_factor):
    /// inv_freq[i] /= factors[i], served at the native window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_freq_factors: Option<Vec<f64>>,
    /// Logit scale folded at load (MiniCPM: dim_model_base/hidden —
    /// the lm_head is tied to the embedding, so the divisor cannot be
    /// baked into the shared tensor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_multiplier: Option<f32>,
    /// Gemma-3n stack (AltUp/LAuReL/PLE/KV-sharing/sparsity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g3n: Option<G3nConfig>,
    /// KDA lower-bound decay gate (Kimi-K3): log-decay =
    /// `lb·σ(exp(A_log)·(f+dt_bias))` instead of the standard
    /// `−exp(A_log)·softplus(f+dt_bias)`. None = standard formula.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_gate_lower_bound: Option<f64>,
    /// Looped Transformer: number of times the layer stack is re-applied
    /// (Nanbeige 4.2: 22 physical layers × 2 loops = 44 virtual layers).
    /// Default 1 = standard non-looped architecture.
    #[serde(default = "default_one_usize", skip_serializing_if = "is_one_usize")]
    pub num_loops: usize,
    /// Apply final normalization after each loop iteration (Nanbeige 4.2).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub loop_final_norm: bool,
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_hidden_act() -> String {
    "silu".into()
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_act(s: &String) -> bool {
    s == "silu"
}

fn default_one() -> f32 {
    1.0
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_one(v: &f32) -> bool {
    *v == 1.0
}

fn default_prf() -> f32 {
    1.0
}

fn default_one_usize() -> usize {
    1
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_one_usize(v: &usize) -> bool {
    *v == 1
}

/// Linear-core selector: the runtime picks the linear-attention
/// operator by `kind` (descriptor-driven ops). "gated_delta_net" =
/// faithful vendor operator carried 1:1 (default for GDN models);
/// "vmf_phase" = canonical core folded at convert time (+offline heal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearCoreConfig {
    /// "gated_delta_net" | "vmf_phase"
    pub kind: String,
    pub num_heads: usize,
    /// Phases per head (vmf_phase only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nphase: Option<usize>,
    pub value_head_dim: usize,
}

impl ModelArch {
    /// Bytes per FFN bitfield row (one layer).
    pub fn ffn_mask_bytes(&self) -> usize {
        self.intermediate_size.div_ceil(8)
    }

    /// Bytes per head bitfield row (one layer).
    pub fn head_mask_bytes(&self) -> usize {
        self.num_attention_heads.div_ceil(8)
    }

    /// Bytes for the layer-gates bitfield.
    pub fn gates_mask_bytes(&self) -> usize {
        self.num_layers.div_ceil(8)
    }

    /// Size of one mask blob in the binary masks section.
    pub fn mask_blob_len(&self) -> usize {
        self.num_layers * (self.ffn_mask_bytes() + self.head_mask_bytes()) + self.gates_mask_bytes()
    }

    /// Bytes per MoE-expert bitfield row (one layer); 0 for dense models.
    pub fn expert_mask_bytes(&self) -> usize {
        self.moe
            .as_ref()
            .map(|m| m.num_experts.div_ceil(8))
            .unwrap_or(0)
    }

    /// Blob size when the optional expert area (spec §5) is present.
    pub fn mask_blob_len_with_experts(&self) -> usize {
        self.mask_blob_len() + self.num_layers * self.expert_mask_bytes()
    }
}

/// Execution mode determined at runtime. Only implemented modes exist
/// (anti-principle №6: no declaration-only enum variants); GPU modes
/// return WITH the Metal path, not before it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// All computation on CPU with SIMD
    CpuOnly { simd_type: SimdType, threads: usize },
    /// Apple Silicon unified memory (CPU compute today; label for metrics)
    AppleUnified { metal_layers: Vec<usize> },
}

/// SIMD instruction set available on CPU.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimdType {
    Avx2,
    Avx512,
    Neon,
    Amx,
    None,
}

/// Per-layer runtime statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerStats {
    pub layer_idx: usize,
    pub active_neurons: usize,
    pub total_neurons: usize,
    pub active_heads: usize,
    pub total_heads: usize,
    pub is_alive: bool,
    pub placement: String, // "gpu" | "cpu"
    pub avg_forward_ms: f64,
}

/// Global runtime performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_generated: u64,
    pub avg_tokens_per_sec: f64,
    pub avg_time_to_first_token_ms: f64,
    pub last_switch_latency_ms: f64,
    pub total_switches: u64,
    pub uptime_seconds: u64,
    pub vram_used_mb: f64,
    pub ram_used_mb: f64,
}
