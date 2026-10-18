//! DeepSeek-V4.1 regression fixture on the WGPU path.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test dsv41_fixture_gpu -- --nocapture
//!
//! Same five-layer synthetic checkpoint as `dsv41_fixture.rs`, decoded with
//! the device paths V4.1 uses: the DSV4 attention tail frame and the shared
//! dynamic MoE bank. A separate test binary because the GPU backend is
//! process-global: `CMF_GPU` has to be set before it initializes, and a CPU
//! comparison cannot be produced in the same process afterwards; the f64
//! reference stands in for it. It runs only with `CMF_GPU=wgpu` exported
//! (skips otherwise, as the CI runners do) and fails when that is set and no
//! wgpu context comes up (see `common/dsv41_gpu.rs`).
//!
//! A device path that refuses the fixture geometry falls back to the host
//! silently and still passes a numeric check, so the test also asserts the
//! `CMF_DSV41_PROF` counters: 48 device attention tails (L0..L3 x 12 tokens;
//! L4's F32 attention linears are not a q4tp frame, 12 host fallbacks) and
//! device MoE calls. The bank takes the first MoE layer's gate/up layout,
//! so two variants run: the fixture as built (q4tp bank, the q2tp layers
//! L1/L3 on the host; M4: 36 device / 24 host) and `build_q2_even` (q2tp
//! bank on L0/L2/L4, q4tp L1/L3 on the host).
//!
//! The host parts are exact (`float_activations_scope`), so the error is
//! the device kernels' f32 arithmetic, amplified where a BF16 boundary flips
//! and the FP4/FP8 caches carry it. Per-position relative logit error
//! against the f64 reference, M4 (wgpu over Metal), with two broken device
//! paths for scale (`CMF_DSV41_TEST_MUTATE=sink|rope`; both must FAIL this
//! test and `dsv41_fixture_gpu_o_fp8`):
//!
//! | run | median | worst |
//! |---|---|---|
//! | q4tp bank, clean | 1.04e-2 | 7.5e-2 |
//! | q2tp bank, clean | 6.1e-3 | 8.2e-3 |
//! | zero attention sinks in the frame, q4tp / q2tp | 4.9e-2 / 3.8e-2 | 9.6e-2 / 8.2e-2 |
//! | inverse RoPE one position off, q4tp / q2tp | 5.2e-2 / 4.2e-2 | 1.45e-1 / 1.42e-1 |
//!
//! The worst position does not separate them (one BF16 flip carried by the
//! caches moves a single position by up to ~7.5e-2, clean or broken); the
//! median does, by 3.6x or more. Bounds (`common/dsv41_gpu.rs`): median
//! < 2.5e-2, worst < 0.12 (the old worst < 0.25 passed both mutations).

#![cfg(feature = "gpu")]
// The fixture's `json!` config nests past the default macro depth.
#![recursion_limit = "256"]

#[path = "common/dsv41_fixture.rs"]
mod dsv41_fixture;
#[path = "common/dsv41_gpu.rs"]
mod dsv41_gpu;

use dsv41_fixture::*;
use dsv41_gpu::*;

#[test]
fn gpu_forward_tracks_f64_reference() {
    if !select_wgpu() {
        return;
    }
    let variants = [
        ("q4tp bank", "gpu", Fixture::build(RELEASE_COMPRESSED_VOCAB)),
        (
            "q2tp bank",
            "gpu-q2",
            Fixture::build_q2_even(RELEASE_COMPRESSED_VOCAB),
        ),
    ];
    for (name, tag, fx) in &variants {
        let run = run_gpu(fx, tag);
        let want = reference_logits(fx, &TOKENS, RefOpts::cortiq());
        let errs = per_position(&run.logits, &want);
        let med = median(&errs);
        let (worst, at) = worst_rel(&run.logits, &want);
        let agree = argmax_agree(&run.logits, &want);
        println!(
            "dsv41 fixture gpu [{name}]: per-position relative logit error [{}]; \
             median {med:.3e}, worst {worst:.3e} at {at}, argmax {agree}/{}",
            fmt_errs(&errs),
            TOKENS.len()
        );
        assert!(
            run.logits.iter().flatten().all(|v| v.is_finite()),
            "[{name}] non-finite logits"
        );
        assert_device_paths(name, run.counters);
        assert!(
            med < GPU_MEDIAN,
            "[{name}] GPU V4.1 forward: median per-position error {med:.3e} [{}]",
            fmt_errs(&errs)
        );
        assert!(
            worst < GPU_WORST,
            "[{name}] GPU V4.1 forward: {worst:.3e} at position {at}"
        );
        assert!(
            agree + 1 >= TOKENS.len(),
            "[{name}] argmax agrees at {agree}/{}",
            TOKENS.len()
        );
    }
}
