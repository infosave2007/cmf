//! `CMF_DSV41_O_FP8=1` on the V4.1 fixture with the device paths: the DSV4
//! attention frame quantizes the attended output with its `fp8_act_quant32`
//! kernel (bit-identical to the host quantizer, see the `dsv41` unit test
//! `fp8_act_quant32_device_matches_host_bits`) instead of falling back to the
//! CPU tail, and the logits follow the reference with `act_quant(o)`.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test dsv41_fixture_gpu_o_fp8 -- --nocapture
//!
//! Its own test binary: the switch and the backend are read once per process.

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
fn gpu_o_fp8_follows_reference_act_quant() {
    // SAFETY: the only test in this binary, before any V4.1 code runs.
    unsafe { std::env::set_var("CMF_DSV41_O_FP8", "1") };
    if !select_wgpu() {
        return;
    }
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let run = run_gpu(&fx, "gpu-ofp8");
    let with = reference_logits(
        &fx,
        &TOKENS,
        RefOpts {
            o_fp8: true,
            ..RefOpts::cortiq()
        },
    );
    let without = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let errs = per_position(&run.logits, &with);
    let med = median(&errs);
    let (worst, at) = worst_rel(&run.logits, &with);
    let apart = per_position(&run.logits, &without);
    let med_apart = median(&apart);
    let agree = argmax_agree(&run.logits, &with);
    println!(
        "dsv41 fixture gpu O_FP8: per-position relative logit error against the \
         act_quant(o) reference [{}]; median {med:.3e}, worst {worst:.3e} at {at}, \
         argmax {agree}/{}; against the reference without it [{}], median {med_apart:.3e}",
        fmt_errs(&errs),
        TOKENS.len(),
        fmt_errs(&apart)
    );
    assert!(
        run.logits.iter().flatten().all(|v| v.is_finite()),
        "non-finite logits"
    );
    // The O_FP8 path must not lose the device frame (it used to force the
    // CPU tail).
    assert_device_paths("O_FP8", run.counters);
    assert!(med < GPU_MEDIAN, "GPU O_FP8 forward: median {med:.3e}");
    assert!(worst < GPU_WORST, "GPU O_FP8 forward: {worst:.3e} at {at}");
    assert!(
        agree + 1 >= TOKENS.len(),
        "argmax agrees at {agree}/{}",
        TOKENS.len()
    );
}
