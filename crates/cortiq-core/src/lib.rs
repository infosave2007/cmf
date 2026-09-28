//! cortiq-core — CMF v2 container: types, tensor directory, masks, quant.
//!
//! Format specification: `docs/CMF_V2_SPEC.md`.

pub mod format;
pub mod hadamard;
pub mod hash;
pub mod knowledge;
pub mod mask;
pub mod quant;
pub mod types;

pub use format::{
    AppendReport, CMF_MAGIC, CMF_VERSION, CmfError, CmfHeader, CmfModel, PendingAppend,
    RoutingCalibration, SelectionDescriptor, SkillRecord, SparseIndexEntry, TensorEntry,
    TensorSpec, TensorSpecRef, build_sparse_index, mtp_sidecar_path,
};
pub use hash::hash64;
pub use knowledge::{
    ExecHashes, ExpertAppend, ExpertTensorSpec, GenomeInfo, GenomeParent, LineageEvent, LookupInfo,
    PhiSpec, RouterPolicy, Segment, SkillBound, SkillOverride, StateEffect, expert_append_base,
    expert_append_layout, expert_append_state_effect, ffn_replace_state_effect, genome_moe_experts,
    key_hash, lookup_state_effect, lookup_tensor_name, lookup_tensors, moe_layers, normalize_key,
    normalized_key_hash, trunk_expert_rank, trunk_hash, trunk_hash_exec,
    validate_expert_append_values, validate_lookup_values,
};
pub use mask::{MaskCatalog, MaskDiff, MaskPriority, Quality, TaskMask};
pub use types::{
    AnchorCoreConfig, ExecutionMode, FarFieldConfig, G3nConfig, LayerStats,
    LayerType, LinearCoreConfig, MlaConfig, ModelArch, MoeConfig, MtpConfig, NormStyle,
    PerformanceMetrics, PrismHadamardConfig, QuantType, Qwen4ExpConfig, SimdType, TensorDtype,
};
