//! cortiq-core — CMF v2 container: types, tensor directory, masks, quant.
//!
//! Format specification: `docs/CMF_V2_SPEC.md`.

pub mod format;
pub mod hash;
pub mod mask;
pub mod quant;
pub mod types;

pub use format::{
    CMF_MAGIC, CMF_VERSION, CmfError, CmfHeader, CmfModel, SelectionDescriptor, SkillRecord,
    SparseIndexEntry, TensorEntry, TensorSpec, build_sparse_index,
};
pub use hash::hash64;
pub use mask::{MaskCatalog, MaskDiff, MaskPriority, Quality, TaskMask};
pub use types::{
    ExecutionMode, LayerStats, LayerType, LinearCoreConfig, ModelArch, MoeConfig, MtpConfig,
    NormStyle, PerformanceMetrics, QuantType, SimdType, TensorDtype,
};
