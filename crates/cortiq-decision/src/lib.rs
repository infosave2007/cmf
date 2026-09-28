//! Cortiq Decision: typed decisions (choice, score, noul) from one CMF file.
//!
//! A decision file (`arch_name` `cortiq-decision-ph-v1`, required feature
//! bit `DECISION` = 0x800) carries a native BERT/WordPiece encoder, the
//! `cortiq-hashfeat-v1` hashing contract and resonance skills: one
//! isolated affine topology per label, decided by the minimal f32
//! reconstruction error and gated by a certified temperature, novelty
//! threshold and confidence threshold. The generative pipeline refuses
//! these files; they are served by `cortiq decide` and `cortiq serve`.
//!
//! CPU is the unchanged, bit-exact default. Opt in with
//! `CORTIQ_DECISION_DEVICE=metal` on Apple Silicon or `vulkan` (feature `vulkan`):
//! resident FP32 encoder, pooling and sequential reconstruction run on the GPU.
//! Tokenization, hashing and gates stay on the CPU;
//! the same encoder golden tolerance applies, and GPU errors never silently
//! fall back to CPU.
//!
//! Modules, in dependency order:
//! - numerics: [`hashfeat`], [`resonance`], [`packed`], [`eigen`], [`fit`],
//!   [`specfn`], [`certify`];
//! - the CMF decision profile: [`canonical`], [`manifest`], [`rows`],
//!   [`container`];
//! - the encoder and signal: [`unicode_tables`], [`wordpiece`], [`bert`],
//!   [`signal`];
//! - training: [`data`], [`build`], [`eval`];
//! - protocol and service: [`protocol`], [`matching`], [`answer`],
//!   [`metering`], [`keys`], [`ledger`], [`config`], [`statedir`],
//!   [`service`], [`shadow`] (the router API's shadow mode);
//! - the oracle cascade: [`pii`], [`oracle`], [`oracle_setup`] (the
//!   two-step `--oracle MODEL`), [`cache`], [`buffer`], [`learn`],
//!   [`generation`], [`cascade`].

// Numerics core.
pub mod certify;
pub mod eigen;
pub mod fit;
// A verbatim copy (spec §1.5; tests/hashfeat.rs checks it against the embryo
// file): `cargo fmt` must not reformat it.
#[rustfmt::skip]
pub mod hashfeat;
pub mod packed;
pub mod resonance;
pub mod specfn;

// CMF decision profile.
pub mod canonical;
pub mod container;
pub mod manifest;
pub mod rows;

// Native encoder and signal composition.
pub mod bert;
pub mod signal;
pub mod unicode_tables;
pub mod wordpiece;

// Training pipeline.
pub mod build;
pub mod data;
pub mod eval;

// Protocol and service.
pub mod answer;
pub mod config;
pub mod keys;
pub mod ledger;
pub mod matching;
pub mod metering;
pub mod protocol;
pub mod service;
pub mod shadow;
pub mod statedir;

// Oracle cascade.
pub mod buffer;
pub mod cache;
pub mod cascade;
pub mod generation;
pub mod learn;
pub mod oracle;
pub mod oracle_setup;
pub mod pii;

#[cfg(feature = "vulkan")]
mod gpu_vulkan;
