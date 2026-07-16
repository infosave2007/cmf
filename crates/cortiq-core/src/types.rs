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
    /// Linear attention (executed by the canonical linear core;
    /// original operator, e.g. GatedDeltaNet, is folded at convert time)
    LinearAttention,
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
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_prf() -> f32 {
    1.0
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
        (self.intermediate_size + 7) / 8
    }

    /// Bytes per head bitfield row (one layer).
    pub fn head_mask_bytes(&self) -> usize {
        (self.num_attention_heads + 7) / 8
    }

    /// Bytes for the layer-gates bitfield.
    pub fn gates_mask_bytes(&self) -> usize {
        (self.num_layers + 7) / 8
    }

    /// Size of one mask blob in the binary masks section.
    pub fn mask_blob_len(&self) -> usize {
        self.num_layers * (self.ffn_mask_bytes() + self.head_mask_bytes()) + self.gates_mask_bytes()
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
