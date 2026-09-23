//! Z-Image-Turbo on native Metal (plan WP3). OWNER: WP3 — this file only.
//!
//! Implements the `gpu::zimage_*` / `gpu::vae_decode_chain` contract
//! (`gpu.rs`, "Z-Image-Turbo device contract"). Rules (plan §2.2):
//! - reach the parent's private kernels, MSL sources, `ctx()` and buffer
//!   pools through `super::`; new MSL goes into this module's own library
//!   string; never edit parent functions;
//! - build pipelines into a module-local, lazily created cache (`OnceLock`);
//!   never add fields to the parent's context struct;
//! - `release` touches module-local state only and must never bring a
//!   device up (it is called unconditionally by `gpu::zimage_release`);
//! - every entry returns `false` before changing any output when it cannot
//!   honour the full contract — the caller then runs the CPU path.
//!
//! The module is `pub` (doc-hidden) so WP3's examples/tests can reach
//! `pub` helpers added here without touching `gpu_metal.rs`.
//!
//! WP0 state: stubs, every entry declines.

use crate::gpu::{ZGeom, ZPrepareArgs, ZStepArgs, ZBlockRef};
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Build/refresh the per-(prompt, resolution) state for `a.key`.
pub(crate) fn prepare(_a: &ZPrepareArgs) -> bool {
    false
}

/// One DiT forward for a prepared `a.key`; writes `a.out`.
pub(crate) fn step(_a: &mut ZStepArgs) -> bool {
    false
}

/// Drop all module-local device state (prepared states, VAE chain).
pub(crate) fn release() {}

/// Unmodulated context refiner on the device (s = 0, gate = 1).
pub(crate) fn refine_caption(
    _model: &Arc<CmfModel>,
    _geom: &ZGeom,
    _blocks: &[ZBlockRef],
    _rope_cap: (&[f32], &[f32]),
    _cap: &mut [f32],
) -> bool {
    false
}

/// Resident Flux-VAE decoder; `z` is already de-normalised.
pub(crate) fn vae_decode_chain(
    _a: &crate::vae::VaeChainArgs,
    _z: &[f32],
    _h: usize,
    _w: usize,
    _out: &mut [f32],
) -> bool {
    false
}
