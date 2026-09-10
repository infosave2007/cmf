//! Cortiq Embryo — a self-developing core born and grown on the CMF
//! format (docs/NATIVE_MODEL_TECH.ru.md). This crate is the native
//! training stack: no ML framework, no Python — Rust on the CPU for the
//! reference math and gradchecks, our own Metal kernels on Apple Silicon
//! for the real work.
//!
//! Layout:
//! - `metal`  — device context + kernels (GEMM, AdamW, norms, CE, …)
//! - `bench`  — the TFLOPS measurement the plan (§4.1) asks for first
//! - `ops`    — CPU reference ops with hand-rolled backwards
//! - `model`  — the Embryo-0 genome: config, parameter layout, forward/backward
//! - `train`  — data shards, the step loop, checkpoints
//! - `export` — the genome as a `.cmf` container the runtime loads

#[cfg(target_os = "macos")]
pub mod bench;
#[cfg(target_os = "macos")]
pub mod cli;
pub mod corpus;
pub mod data;
pub mod export;
#[cfg(target_os = "macos")]
pub mod growth;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod model;
#[cfg(target_os = "macos")]
pub mod mtp;
pub mod ops;
#[cfg(target_os = "macos")]
pub mod skill;
#[cfg(target_os = "macos")]
pub mod sleep;
pub mod tokenizer;
pub mod train;
