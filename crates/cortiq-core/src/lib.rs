//! cortiq-core — CMF v2 container: types, tensor directory, masks, quant.
//!
//! Format specification: `docs/CMF_V2_SPEC.md`.

pub mod format;
pub mod hadamard;
pub mod hash;
pub mod mask;
pub mod quant;
pub mod types;

pub use format::{
    build_sparse_index, CmfError, CmfHeader, CmfModel, SelectionDescriptor, SkillRecord,
    SparseIndexEntry, TensorEntry, TensorSpec, TensorSpecRef, CMF_MAGIC, CMF_VERSION,
};
pub use hash::hash64;
pub use mask::{MaskCatalog, MaskDiff, MaskPriority, Quality, TaskMask};
pub use types::{
    ExecutionMode, G3nConfig, Glm5NextConfig, LayerStats, LayerType, LinearCoreConfig, MlaConfig,
    ModelArch, MoeConfig, MtpConfig, NormStyle, PerformanceMetrics, PrismHadamardConfig, QuantType,
    Qwen4ExpConfig, SimdType, TensorDtype,
};
