//! `CMF_DSV41_O_FP8=1` on the V4.1 fixture with the device paths: the DSV4
//! attention frame quantizes the attended output with its `fp8_act_quant32`
//! kernel (bit-identical to the host quantizer, see the `dsv41` unit test
//! `fp8_act_quant32_device_matches_host_bits`) instead of falling back to the
//! CPU tail, and the logits follow the reference with `act_quant(o)`.
//!
//! The logit bounds alone cannot show that the frame quantized: the same
//! device decode WITHOUT act_quant(o) also meets them against this
//! reference (median ~1.4e-2 on the M4, bound 2.5e-2). So the test also
//! asserts that every device attention tail encoded the kernel
//! (`fp8_act::frame_encodes()`), and checks the frame's wiring directly
//! (`frame_wiring`): its `attn` tap with the flag is the reference
//! quantizer of the tap without it, and `o_lora_a` reads the quantized
//! output.
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

use cortiq_engine::gpu_wgpu::{self, Dsv4AttnGeom, Dsv4AttnW};
use cortiq_engine::qtensor::{QTensor, float_activations_scope};
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
    // CPU tail), and every device tail must quantize.
    assert_device_paths("O_FP8", run.counters);
    assert_bank_family("O_FP8", run.bank);
    assert_eq!(
        run.fp8_encodes, GPU_ATTN_CALLS,
        "act_quant(o) encodes over the device attention tails"
    );
    assert!(med < GPU_MEDIAN, "GPU O_FP8 forward: median {med:.3e}");
    assert!(worst < GPU_WORST, "GPU O_FP8 forward: {worst:.3e} at {at}");
    assert!(
        agree + 1 >= TOKENS.len(),
        "argmax agrees at {agree}/{}",
        TOKENS.len()
    );
    frame_wiring(&fx);
}

/// Relative L2 distance of `got` from `want`.
fn rel(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).powi(2))
        .sum();
    let den: f64 = want.iter().map(|&w| (w as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The frame's act_quant(o) wiring, deterministically. One device attention
/// tail (L0's q4tp weights, a fixed BF16 query and cache, V4.1 geometry)
/// runs with `o_fp8` off and on and is read back at the frame's taps:
/// - `attn`: on equals the reference quantizer (`act_quant_fp8`) applied to
///   off, value for value, and most values moved: the kernel ran over the
///   whole BF16 output after the inverse RoPE;
/// - `mid` (the BF16 grouped `wo_a` output): each arm is much closer to the
///   host's `wo_a` of its OWN `attn` than of the other arm's, so `o_lora_a`
///   read the quantized output: the kernel ran before it.
fn frame_wiring(fx: &Fixture) {
    let model = fx.open("gpu-ofp8-frame");
    let idx = |n: &str| {
        let name = format!("model.layers.0.self_attn.{n}.weight");
        model
            .tensors
            .iter()
            .position(|e| e.name == name)
            .unwrap_or_else(|| panic!("no {name}"))
    };
    let q_norm = vec![1.0f32; Q_LORA];
    let sink = [0.4f32, -0.6];
    let inv_freq: Vec<f32> = (0..ROPE_DIM / 2)
        .map(|i| 1.0 / 10_000f32.powf(2.0 * i as f32 / ROPE_DIM as f32))
        .collect();
    let npos = 8usize;
    let cache: Vec<f32> = (0..npos * HEAD_DIM)
        .map(|i| bf32(((i * 5) as f32 * 0.011).cos()))
        .collect();
    let idxs: Vec<u32> = (0..npos as u32).filter(|i| i % 3 != 1).collect();
    let q: Vec<f32> = (0..N_HEADS * HEAD_DIM)
        .map(|i| bf32(((i * 11) as f32 * 0.019).sin()))
        .collect();
    // Not a Dsv41State id (those carry 0xD541 in the top bits).
    const KV: u64 = 0xF8A7_0000_0000_0001;
    assert!(
        gpu_wgpu::dsv4_cache_write(KV, 0, 0, &cache, npos * HEAD_DIM),
        "cache upload"
    );
    let w = Dsv4AttnW {
        wq_a: idx("wq_a"),
        wq_b: idx("wq_b"),
        wo_a: idx("wo_a"),
        wo_b: idx("wo_b"),
        q_norm: &q_norm,
        sink: &sink,
    };
    let n = N_HEADS * HEAD_DIM;
    let mid_n = O_GROUPS * O_LORA;
    let frame = |o_fp8: bool, tap: &str, len: usize| -> Vec<f32> {
        // SAFETY: the only test in this binary; nothing else reads the
        // variable between the set and the call it steers.
        unsafe { std::env::set_var("CMF_DSV4_FRAME_TAP", tap) };
        let g = Dsv4AttnGeom {
            dim: DIM,
            nh: N_HEADS,
            hd: HEAD_DIM,
            rd: ROPE_DIM,
            q_lora: Q_LORA,
            o_lora: O_LORA,
            o_groups: O_GROUPS,
            eps: EPS as f32,
            scale: (HEAD_DIM as f32).powf(-0.5),
            bf16: true,
            q_rms: false,
            o_fp8,
        };
        let mut out = vec![0.0f32; n.max(DIM)];
        assert!(
            gpu_wgpu::dsv4_attn_frame(
                &model,
                &w,
                g,
                &[],
                None,
                Some(&q),
                KV,
                0,
                &idxs,
                &inv_freq,
                5,
                None,
                &mut out,
            ),
            "the frame declined (o_fp8 {o_fp8}, tap {tap})"
        );
        out.truncate(len);
        out
    };
    let encodes = gpu_wgpu::fp8_act::frame_encodes();
    let attn_off = frame(false, "attn", n);
    let attn_on = frame(true, "attn", n);
    let mid_off = frame(false, "mid", mid_n);
    let mid_on = frame(true, "mid", mid_n);
    let encodes = gpu_wgpu::fp8_act::frame_encodes() - encodes;
    unsafe { std::env::remove_var("CMF_DSV4_FRAME_TAP") };
    gpu_wgpu::dsv4_cache_clear(KV);

    let mut want: Vec<f64> = attn_off.iter().map(|&v| v as f64).collect();
    act_quant_fp8(&mut want);
    let differ = attn_on
        .iter()
        .zip(&want)
        .filter(|&(&g, &w)| g as f64 != w)
        .count();
    let moved = attn_on
        .iter()
        .zip(&attn_off)
        .filter(|&(a, b)| a != b)
        .count();

    let wo_a = QTensor::from_model(&model, &model.tensors[w.wo_a].name).expect("wo_a");
    let per_group = n / O_GROUPS;
    let host_mid = |attn: &[f32]| -> Vec<f32> {
        float_activations_scope(|| {
            let mut scratch = vec![0.0f32; per_group];
            (0..mid_n)
                .map(|i| {
                    let g = i / O_LORA;
                    bf32(wo_a.row_dot(i, &attn[g * per_group..(g + 1) * per_group], &mut scratch))
                })
                .collect()
        })
    };
    let (host_off, host_on) = (host_mid(&attn_off), host_mid(&attn_on));
    let (on_own, on_other) = (rel(&mid_on, &host_on), rel(&mid_on, &host_off));
    let (off_own, off_other) = (rel(&mid_off, &host_off), rel(&mid_off, &host_on));
    gpu_wgpu::release_model_buffers(&model);
    println!(
        "dsv41 fixture gpu O_FP8 frame wiring: attn tap {differ}/{n} values differ from \
         act_quant of the unquantized tap ({moved}/{n} moved by it); mid against the host \
         wo_a of its own / the other arm's attn: on {on_own:.2e} / {on_other:.2e}, \
         off {off_own:.2e} / {off_other:.2e}; {encodes} kernel encodes"
    );
    assert_eq!(
        encodes, 2,
        "the frame encodes the kernel exactly when o_fp8 is set"
    );
    assert_eq!(
        differ, 0,
        "the frame's quantized attn is not act_quant of its unquantized attn"
    );
    assert!(moved > n / 2, "act_quant(o) moved only {moved}/{n} values");
    assert!(
        on_own * 4.0 < on_other && off_own * 4.0 < off_other,
        "o_lora_a did not read the arm's own attn: on {on_own:.2e} / {on_other:.2e}, \
         off {off_own:.2e} / {off_other:.2e}"
    );
}
