//! The device side of the V4.1 fixture binaries (`dsv41_fixture_gpu*.rs`):
//! backend selection, one decode with the device paths, and the path
//! counters that prove those paths ran.

#![allow(dead_code)]

use crate::dsv41_fixture::*;
use cortiq_engine::dsv41::{self, Dsv41Cfg, Dsv41State};
use cortiq_engine::qtensor::float_activations_scope;

/// Bounds on a device decode against the f64 reference, with the host parts
/// exact: the median per-position relative logit error and the worst one.
/// The median is what separates a correct device path from a broken one
/// (one BF16 flip carried by the FP4/FP8 caches moves single positions by up
/// to ~7.5e-2 either way); the worst bound is a cap. Measured in
/// `dsv41_fixture_gpu.rs`.
pub const GPU_MEDIAN: f64 = 2.5e-2;
pub const GPU_WORST: f64 = 0.12;

/// Device attention tail calls of one fixture decode: L0..L3 for each of
/// the 12 tokens. L4's F32 attention linears are not a q4tp frame, so its
/// 12 calls fall back to the host tail.
pub const GPU_ATTN_CALLS: u64 = 48;
pub const GPU_ATTN_FALLBACKS: u64 = 12;

/// Turn the V4.1 path counters on and size the expert bank before anything
/// initializes the backend, then follow the repo's GPU-test convention
/// (`gpu_wgpu::selected_and_up` / `skip_or_fail`, as the DSV4 parity tests):
/// skip when nobody asked for wgpu, FAIL when `CMF_GPU=wgpu` was set and no
/// context came up (a broken driver must not pass as a skip). Returns false
/// (after saying why) when the binary should skip.
///
/// Nobody asked means `CMF_GPU` unset or naming another backend: the CI
/// runners (`cargo test --workspace`), i.e. ubuntu without an adapter and
/// the macOS VM, whose paravirtual Metal adapter is not a reference device
/// (binding arrays, limits and Metal's own rounding differ). The M4 and the
/// pod run these binaries with `CMF_GPU=wgpu` exported.
pub fn select_wgpu() -> bool {
    // SAFETY: the first call of the single test in each GPU binary, before
    // any backend or V4.1 code runs.
    unsafe {
        std::env::set_var("CMF_DSV41_PROF", "1");
        // An adapter that reports no VRAM budget (Apple unified memory via
        // wgpu) sizes the V4.1 expert bank from u64::MAX: ~96 GiB of slots
        // on paper, ~18 s of device waits per token, and every MoE call
        // falling back to the host. The bank allocator keeps a workspace
        // reserve below the budget: 2 GiB leaves no bank at all, 4 GiB gives
        // a 1 GiB bank (measured on the M4), plenty for the fixture.
        if std::env::var_os("CMF_GPU_VRAM_MB").is_none() {
            std::env::set_var("CMF_GPU_VRAM_MB", "4096");
        }
    }
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            false
        }
        Some(false) => panic!(
            "{}: CMF_GPU={} selects wgpu but no wgpu context came up",
            module_path!(),
            std::env::var("CMF_GPU").unwrap_or_default()
        ),
        Some(true) => true,
    }
}

pub struct GpuRun {
    pub logits: Vec<Vec<f32>>,
    /// `dsv41::profile_counters()` over this run:
    /// `(gpu_attn, gpu_attn_fallbacks, gpu_moe, cpu_moe)`.
    pub counters: (u64, u64, u64, u64),
    /// Encodes of the frame's act_quant(o) kernel over this run
    /// (`gpu_wgpu::fp8_act::frame_encodes()`): one per device attention
    /// tail under `CMF_DSV41_O_FP8=1`, none otherwise.
    pub fp8_encodes: u64,
}

/// Decode `TOKENS` through the loader and `forward_token` with the device
/// paths live and the host parts exact (`float_activations_scope`: no A8W8
/// in the host matvecs), so the comparison measures the device kernels. The
/// model's device caches (cached weights, expert bank) are released at the
/// end, so a second variant in the same process starts from a clean card.
pub fn run_gpu(fx: &Fixture, tag: &str) -> GpuRun {
    let model = fx.open(tag);
    let source = model.arch().deepseek_v41.clone().expect("source config");
    let (cfg, n_layers) = Dsv41Cfg::from_source(&source, model.arch()).expect("parse cfg");
    let before = dsv41::profile_counters();
    let fp8_before = cortiq_engine::gpu_wgpu::fp8_act::frame_encodes();
    let logits = float_activations_scope(|| {
        let (globals, layers, hash) =
            dsv41::load(&model, &cfg, n_layers, fx.token_map()).expect("load layers");
        let mut st = Dsv41State::new(&cfg, hash);
        TOKENS
            .iter()
            .enumerate()
            .map(|(p, &t)| {
                let mut logits = Vec::new();
                dsv41::forward_token(&globals, &layers, &cfg, &mut st, t, p, None, &mut logits);
                logits
            })
            .collect::<Vec<_>>()
    });
    let after = dsv41::profile_counters();
    let fp8_encodes = cortiq_engine::gpu_wgpu::fp8_act::frame_encodes() - fp8_before;
    cortiq_engine::gpu_wgpu::release_model_buffers(&model);
    GpuRun {
        logits,
        counters: (
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            after.3 - before.3,
        ),
        fp8_encodes,
    }
}

/// The device-path assertions shared by the GPU binaries: every attention
/// call of a q4tp layer ran the device frame, and, where the adapter can
/// hold the segmented expert bank at all, the MoE ran on the device too.
pub fn assert_device_paths(name: &str, counters: (u64, u64, u64, u64)) {
    let (attn, fallbacks, moe, cpu_moe) = counters;
    println!(
        "dsv41 fixture gpu [{name}]: gpu_attn {attn} fallbacks {fallbacks} gpu_moe {moe} cpu_moe {cpu_moe}"
    );
    assert_eq!(
        (attn, fallbacks),
        (GPU_ATTN_CALLS, GPU_ATTN_FALLBACKS),
        "[{name}] device attention tail calls / host fallbacks"
    );
    if cortiq_engine::gpu_wgpu::dsv4_global_moe_supported() {
        assert!(moe > 0, "[{name}] no MoE call ran on the device");
    } else {
        eprintln!(
            "[{name}] this adapter cannot hold the segmented expert bank \
             (no binding arrays): MoE device check not applicable"
        );
    }
}
