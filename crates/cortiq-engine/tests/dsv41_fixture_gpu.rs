//! DeepSeek-V4.1 regression fixture on the WGPU path.
//!
//!     cargo test -p cortiq-engine --features gpu --test dsv41_fixture_gpu -- --nocapture
//!
//! Same five-layer synthetic checkpoint as `dsv41_fixture.rs`, decoded with
//! the device paths V4.1 uses: the DSV4 attention tail frame and the shared
//! dynamic MoE bank (q4tp gate/up layers and q2tp gate/up layers). It skips
//! without an adapter. A separate test binary because the GPU backend is
//! process-global: `CMF_GPU` has to be set before it initializes, and a CPU
//! comparison cannot be produced in the same process afterwards; the f64
//! reference stands in for it.
//!
//! The run prints the `CMF_DSV41_PROF` counters: a device path that refuses
//! the fixture geometry falls back to the host silently and still passes the
//! numeric check, so read `gpu_attn_calls` / `gpu_moe_calls` there. Expected
//! on an M4 (wgpu/Metal): attention 48 device calls + 12 fallbacks (L4 has
//! F32 attention linears; the frame takes q4tp), MoE 36 device calls + 24
//! host calls (the bank is created with the first layer's q4tp gate/up
//! layout; the q2tp layers L1/L3 stay on the host).

#![cfg(feature = "gpu")]
// The fixture's `json!` config nests past the default macro depth.
#![recursion_limit = "256"]

#[path = "common/dsv41_fixture.rs"]
mod dsv41_fixture;

use cortiq_core::CmfModel;
use cortiq_engine::dsv41::{self, Dsv41Cfg, Dsv41State};
use dsv41_fixture::*;
use std::sync::Arc;

#[test]
fn gpu_forward_tracks_f64_reference() {
    // SAFETY: set before any backend initialization in this process; this
    // binary holds one test.
    unsafe {
        std::env::set_var("CMF_GPU", "wgpu");
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
    if !cortiq_engine::gpu_wgpu::enabled() {
        eprintln!("skipped: no WGPU adapter");
        return;
    }
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let path = fx.write("gpu");
    let model = Arc::new(CmfModel::open(&path).expect("open fixture"));
    let source = model.arch().deepseek_v41.clone().expect("source config");
    let (cfg, n_layers) = Dsv41Cfg::from_source(&source, model.arch()).expect("parse cfg");
    let (globals, layers, hash) =
        dsv41::load(&model, &cfg, n_layers, fx.token_map()).expect("load layers");
    let mut st = Dsv41State::new(&cfg, hash);
    let mut got = Vec::with_capacity(TOKENS.len());
    for (p, &t) in TOKENS.iter().enumerate() {
        let mut logits = Vec::new();
        dsv41::forward_token(&globals, &layers, &cfg, &mut st, t, p, None, &mut logits);
        got.push(logits);
    }
    dsv41::profile_report();
    let want = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let (worst, at) = worst_rel(&got, &want);
    let agree = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| argmax32(g) == argmax64(w))
        .count();
    println!(
        "dsv41 fixture gpu: worst relative logit error {worst:.3e} at {at}, argmax {agree}/{}",
        TOKENS.len()
    );
    assert!(got.iter().flatten().all(|v| v.is_finite()), "non-finite logits");
    // Same bound as the default CPU path: the device kernels are not
    // bit-exact with the f64 reference and the FP4 caches amplify that.
    assert!(worst < 0.25, "GPU V4.1 forward: {worst:.3e} at position {at}");
    assert!(
        agree + 2 >= TOKENS.len(),
        "argmax agrees at {agree}/{}",
        TOKENS.len()
    );
}
