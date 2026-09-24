//! Metal Z-Image kernels against f64 host references (plan WP3). Skips
//! (passes) where there is no Metal device. The whole-model gates need the
//! published files and oracles: `examples/zimage_metal_check.rs`.
#![cfg(target_os = "macos")]

use cortiq_engine::gpu_metal::zimage::{bench_flash, bench_gemm};

#[test]
fn q8_gemm_matches_f64() {
    // two token counts: a full tile and a partial one (n % 64 != 0)
    for n in [128usize, 96] {
        let Some((_, _, rel)) = bench_gemm(256, 512, n, 1, "base") else {
            eprintln!("no Metal device: skipped");
            return;
        };
        assert!(rel < 1e-5, "zi_q8mm n {n}: rel {rel:e}");
    }
}

#[test]
fn flash_matches_f64() {
    for v in ["q64pf", "q32"] {
        let Some((_, _, rel)) = bench_flash(96, 1, v) else {
            eprintln!("no Metal device: skipped");
            return;
        };
        // half q/k/v/P and a half output: ~2.6e-4 is the f16 floor
        assert!(rel < 1e-3, "zi_flash {v}: rel {rel:e}");
    }
}
