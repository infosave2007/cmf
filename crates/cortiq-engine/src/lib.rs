//! Cortiq inference engine — sparse forward pass, attention, tokenization, sampling.

pub mod attention;
pub mod gpu;
#[cfg(target_os = "macos")]
pub mod gpu_metal;
#[cfg(feature = "gpu")]
pub mod gpu_wgpu;
pub mod inference;
pub mod kv_cache;
pub mod linear_core;
pub mod loader;
pub mod pipeline;
pub mod pool;
pub mod qtensor;
pub mod router;
pub mod runtime;
pub mod sampler;
pub mod swarm;
pub mod tokenizer;

pub use pipeline::{GenerateResult, Pipeline, TokenCallback, TokenTrace};
pub use runtime::CortiqRuntime;
pub use sampler::SamplerConfig;
