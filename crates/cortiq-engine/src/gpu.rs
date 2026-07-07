//! Facade for GPU backends: a single call entry point for qtensor/pipeline/
//! linear_core. Job types and the threshold are canonical HERE; behind the
//! facade dispatch goes to a platform backend:
//!   - `gpu_metal` (Apple Silicon, unified memory + no-copy buffers);
//!   - `gpu_wgpu` (C1: Vulkan/DX12/Metal — NVIDIA/Radeon/Intel/Apple,
//!     weights resident in VRAM), available under `--features gpu`.
//!
//! Runtime selection via `CMF_GPU`: `1` — native Metal (macOS) or wgpu
//! (other OSes); `wgpu` — force wgpu (including for the local
//! Metal-via-wgpu parity test). Any backend refusal — `false` and the honest
//! CPU path, no partial results.

use cortiq_core::CmfModel;
use std::cell::Cell;
use std::sync::{Arc, OnceLock};

thread_local! {
    /// Index of the current forward layer (−1 = outside a numbered layer:
    /// lm_head/embed — always allowed). The pipeline sets it before
    /// each layer so that the GPU/CPU layer-split works.
    static CUR_LAYER: Cell<i64> = const { Cell::new(-1) };
}

/// Pipeline: mark the current layer (or −1 outside layers) for layer-split.
pub fn set_layer(l: i64) {
    CUR_LAYER.with(|c| c.set(l));
}

/// Parse `CMF_GPU_LAYERS` («0-19», «0,2,4», «0-9,30-39») once.
/// None = no restriction (all layers on GPU). Garbage → also no restriction.
fn layer_ranges() -> &'static Option<Vec<(i64, i64)>> {
    static R: OnceLock<Option<Vec<(i64, i64)>>> = OnceLock::new();
    R.get_or_init(|| {
        let s = std::env::var("CMF_GPU_LAYERS").ok()?;
        let mut v = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            match part.split_once('-') {
                Some((a, b)) => v.push((a.trim().parse().ok()?, b.trim().parse().ok()?)),
                None => {
                    let x: i64 = part.parse().ok()?;
                    v.push((x, x));
                }
            }
        }
        Some(v)
    })
}

fn layer_allowed() -> bool {
    match layer_ranges() {
        None => true,
        Some(ranges) => {
            let cur = CUR_LAYER.with(|c| c.get());
            cur < 0 || ranges.iter().any(|(a, b)| cur >= *a && cur <= *b)
        }
    }
}

/// GPU allowed FOR THE CURRENT LAYER: backend is initialized AND the layer
/// falls within `CMF_GPU_LAYERS` (GPU/CPU layer-split). Op gates call this.
pub fn enabled_here() -> bool {
    enabled() && layer_allowed()
}

/// Default row threshold: the GPU takes only larger matrices (lm_head
/// class). Below it, the dispatch/readback cost does not pay off on unified memory.
pub const GPU_MIN_ROWS: usize = 65_536;

/// Effective threshold: `CMF_GPU_MIN_ROWS` overrides the default — on a
/// discrete card it is worth lowering it (VRAM bandwidth pays off even for FFN),
/// on unified memory — raising it. A «squeeze out the maximum» tuning on the server.
pub fn min_rows() -> usize {
    std::env::var("CMF_GPU_MIN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GPU_MIN_ROWS)
}

/// A single MoE-FFN job (an expert with its own weight), executed in one
/// submission: (rows, cols, idx, row_scale) for gate/up/down + prescaled
/// inputs + the down θ-field + the blending weight.
pub struct MoeJob<'a> {
    pub gate: (usize, usize, usize, &'a [f32]),
    pub up: (usize, usize, usize, &'a [f32]),
    pub down: (usize, usize, usize, &'a [f32]),
    pub xs_gate: Vec<f32>,
    pub xs_up: Vec<f32>,
    pub down_col: &'a [f32],
    pub w: f32,
}

/// A single independent batch matvec (GDN projections of one input).
pub struct BatchJob<'a> {
    pub idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_scale: &'a [f32],
    pub xs: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    None,
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(feature = "gpu")]
    Wgpu,
}

fn backend() -> Backend {
    #[cfg(feature = "gpu")]
    if crate::gpu_wgpu::selected() {
        return if crate::gpu_wgpu::enabled() { Backend::Wgpu } else { Backend::None };
    }
    #[cfg(target_os = "macos")]
    if crate::gpu_metal::enabled() {
        return Backend::Metal;
    }
    Backend::None
}

/// GPU enabled and initialized on the selected backend?
pub fn enabled() -> bool {
    backend() != Backend::None
}

/// q8_row/q8_2f matvec, rows [row0, row0+rows). `xs` — prescaled by the θ-field.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn q8_matvec_range(
    model: &Arc<CmfModel>,
    idx: usize,
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            crate::gpu_metal::q8_matvec_range(model, idx, row0, row_scale, xs, rows, cols, out)
        }
        #[cfg(feature = "gpu")]
        Backend::Wgpu => {
            crate::gpu_wgpu::q8_matvec_range(model, idx, row0, row_scale, xs, rows, cols, out)
        }
        Backend::None => false,
    }
}

/// GEMM of a prefill batch: `pre` — prescaled inputs row-major [b, cols],
/// out — row-major [b, rows].
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn q8_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            crate::gpu_metal::q8_matmat(model, idx, row_scale, pre, b, rows, cols, out)
        }
        #[cfg(feature = "gpu")]
        Backend::Wgpu => {
            crate::gpu_wgpu::q8_matmat(model, idx, row_scale, pre, b, rows, cols, out)
        }
        Backend::None => false,
    }
}

/// A layer's MoE-FFN in one submission (amortizing the dispatch cost).
#[allow(unused_variables)]
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::moe_block(model, jobs, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::moe_block(model, jobs, out),
        Backend::None => false,
    }
}

/// Independent matvecs of one input in a single submission (GDN projections).
#[allow(unused_variables)]
pub fn matvec_batch(model: &Arc<CmfModel>, jobs: &[BatchJob], out: &mut [&mut [f32]]) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::matvec_batch(model, jobs, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::matvec_batch(model, jobs, out),
        Backend::None => false,
    }
}
