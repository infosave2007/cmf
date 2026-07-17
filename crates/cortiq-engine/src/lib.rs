//! Cortiq inference engine — sparse forward pass, attention, tokenization, sampling.

pub mod attention;
pub mod fcd;
pub mod fcd_ops;
pub mod gpu;
#[cfg(target_os = "macos")]
pub mod gpu_metal;
#[cfg(feature = "gpu")]
pub mod gpu_wgpu;
pub mod inference;
pub mod kv_cache;
pub mod linear_core;
pub mod loader;
pub mod nystrom;
pub mod pipeline;
pub mod pool;
pub mod qtensor;
pub mod router;
pub mod runtime;
pub mod sampler;
pub mod skillbake;
pub mod swarm;
pub mod tokenizer;

pub use nystrom::NystromState;
pub use pipeline::{GenerateResult, Pipeline, TokenCallback, TokenTrace};
pub use runtime::CortiqRuntime;

/// Test-only: N empty Metal command-buffer round trips, total seconds.
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_empty_submit_for_test(n: usize) -> f64 {
    gpu_metal::empty_submit_bench(n)
}

/// Test-only: N pipelined empty submits, one final wait.
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_pipelined_submit_for_test(n: usize) -> f64 {
    gpu_metal::pipelined_submit_bench(n)
}

/// Test-only: build a q1 MoeJob trio (weight 1.0).
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_moe_job_for_test(
    gi: usize,
    ui: usize,
    di: usize,
    inter: usize,
    hidden: usize,
    x: Vec<f32>,
) -> gpu::MoeJob<'static> {
    gpu::MoeJob {
        gate: (gi, inter, hidden, &[]),
        up: (ui, inter, hidden, &[]),
        down: (di, hidden, inter, &[]),
        xs_gate: x.clone(),
        xs_up: x,
        down_col: &[],
        w: 1.0,
        q1: true,
    }
}

/// Test-only: run the metal moe_block on one job.
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_moe_block_for_test(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    job: gpu::MoeJob<'_>,
    out: &mut [f32],
) -> bool {
    gpu_metal::moe_block(model, &[job], out)
}

/// Test-only: q1 matvec_batch — jobs (idx, rows, cols); first two share
/// x, the third takes xi.
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_batch_q1_for_test(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    shapes: &[(usize, usize, usize)],
    x: &[f32],
    xi: &[f32],
    outs: &mut [&mut [f32]],
) -> bool {
    let jobs: Vec<gpu::BatchJob> = shapes
        .iter()
        .enumerate()
        .map(|(k, &(idx, rows, cols))| gpu::BatchJob {
            idx,
            rows,
            cols,
            row_scale: &[],
            xs: if k < 2 { x.to_vec() } else { xi.to_vec() },
            q1: true,
        })
        .collect();
    gpu_metal::matvec_batch(model, &jobs, outs)
}

/// Test-only direct handle to the Metal q1 matvec (micro-benchmarks).
#[doc(hidden)]
#[cfg(target_os = "macos")]
pub fn gpu_q1_matvec_for_test(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    gpu_metal::q1_matvec(model, idx, xs, rows, cols, out)
}
pub use sampler::SamplerConfig;
