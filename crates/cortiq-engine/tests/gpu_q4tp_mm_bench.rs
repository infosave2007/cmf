//! The batched prefill's q4tp GEMM kernels in isolation, TFLOP/s on the
//! Spark-X2.6-4B shapes (hidden 2560, FFN 10240, 512-row chunks): the
//! in-kernel-dequant cooperative kernel, its pipelined twin, the `zi_mm`
//! route (plane dequant + f16 cast + 128x128x32 tile + store scale) and the
//! scalar tile. `CMF_MM_BENCH_VARIANTS=zi,coop,v2,scalar` picks,
//! `CMF_MM_BENCH_K=512`. The second test holds every variant to the scalar
//! GEMM's answer.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --release \
//!         --test gpu_q4tp_mm_bench -- --nocapture

#[cfg(feature = "gpu")]
fn device_up() -> bool {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            false
        }
        Some(false) => panic!("wgpu selected but the context did not come up"),
        Some(true) => true,
    }
}

#[cfg(feature = "gpu")]
#[test]
fn q4tp_mm_kernels_tflops_needs_cmf_gpu() {
    if !device_up() {
        return;
    }
    let k: usize = std::env::var("CMF_MM_BENCH_K").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let variants = std::env::var("CMF_MM_BENCH_VARIANTS").unwrap_or_else(|_| "zi,coop,v2,scalar".into());
    // (name, rows, cols): gate-up, down, qkv, o
    let shapes: &[(&str, usize, usize)] = &[
        ("gate-up", 20480, 2560),
        ("down", 2560, 10240),
        ("qkv", 6144, 2560),
        ("o", 2560, 4096),
    ];
    for v in variants.split(',') {
        let mut total_ms = 0.0;
        let mut total_fl = 0.0;
        for &(name, rows, cols) in shapes {
            match cortiq_engine::gpu_wgpu::q4tp_mm_bench_for_test(rows, cols, k, v, 10) {
                Some(ms) => {
                    let fl = 2.0 * rows as f64 * cols as f64 * k as f64;
                    total_ms += ms;
                    total_fl += fl;
                    println!("{v:>7} {name:>8} {rows:>6}x{cols:<6} k={k}: {ms:8.2} ms  {:6.2} TFLOP/s", fl / ms / 1e9);
                }
                None => println!("{v:>7} {name:>8}: missing"),
            }
        }
        if total_ms > 0.0 {
            println!("{v:>7} per-layer sum {total_ms:8.2} ms  {:6.2} TFLOP/s  (x36 layers = {:7.1} ms a chunk)", total_fl / total_ms / 1e9, total_ms * 36.0);
        }
    }
}

/// Every tensor-core variant against the scalar tile GEMM: f16 operands
/// with the device activation scale sit within 1e-2 relative rms of the
/// f32 answer on these operands (unit weight scales, |x| < 1).
#[cfg(feature = "gpu")]
#[test]
fn q4tp_mm_variants_match_scalar_needs_cmf_gpu() {
    if !device_up() {
        return;
    }
    let mut failed = Vec::new();
    for &(rows, cols, k) in &[(2560usize, 2560usize, 512usize), (512, 4096, 128)] {
        for v in ["zi", "coop", "v2"] {
            match cortiq_engine::gpu_wgpu::q4tp_mm_rel_for_test(rows, cols, k, v) {
                Some(r) => {
                    println!("{v:>5} {rows}x{cols} k={k}: rel rms {r:.3e}");
                    if !(r < 1e-2) {
                        failed.push(format!("{v} {rows}x{cols} k={k} rel {r:.3e}"));
                    }
                }
                None => println!("{v:>5} {rows}x{cols} k={k}: missing"),
            }
        }
    }
    assert!(failed.is_empty(), "GEMM variants drifted from the scalar arm: {failed:?}");
}
