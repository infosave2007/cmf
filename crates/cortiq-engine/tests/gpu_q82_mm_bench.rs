//! The q8_2f frame GEMM in isolation at the shapes of a Fast Code frame
//! (GDN `out` 2560 x 6144, GDN value 6144 x 2560, GDN key 2048 x 2560, a
//! QSA projection 2560 x 2560): `q82_mm_coop_s` against its pipelined twin
//! `q82_mm_coop_v2_s`, at frame-sized batches. A frame launch is as long as
//! one workgroup's walk over K, so the figure to read is ms a launch, not
//! TFLOP/s. Needs a device with cooperative matrices:
//!   CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --release \
//!     --test gpu_q82_mm_bench -- --nocapture
//! `CMF_MM_BENCH_K` picks the batch sizes (default 32,64,128,256).
#![cfg(feature = "gpu")]

use cortiq_engine::gpu_wgpu::{q82_mm_bench_for_test, q82_mm_v2_diff_for_test};

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

const SHAPES: [(&str, usize, usize); 4] = [
    ("gdn-out", 2560, 6144),
    ("gdn-val", 6144, 2560),
    ("gdn-key", 2048, 2560),
    ("qsa-proj", 2560, 2560),
];

fn batches() -> Vec<usize> {
    std::env::var("CMF_MM_BENCH_K")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![32, 64, 128, 256])
}

#[test]
fn q82_mm_kernels_ms_needs_cmf_gpu() {
    if !device_up() {
        return;
    }
    for k in batches() {
        let mut sums = [0f64; 2];
        for (name, rows, cols) in SHAPES {
            let mut line = format!("q82 {name:>8} {rows:>5}x{cols:<5} k={k:<3}:");
            for (i, variant) in ["q82", "q82v2"].into_iter().enumerate() {
                match q82_mm_bench_for_test(rows, cols, k, variant, 200) {
                    Some(ms) => {
                        sums[i] += ms;
                        let steps = (cols / 32) as f64;
                        line += &format!("  {variant} {ms:.3} ms ({:.2} us a K step)", ms * 1e3 / steps);
                    }
                    None => line += &format!("  {variant} missing"),
                }
            }
            println!("{line}");
        }
        println!("q82 sum k={k}: q82 {:.3} ms, q82v2 {:.3} ms", sums[0], sums[1]);
    }
}

#[test]
fn q82_mm_v2_matches_coop_needs_cmf_gpu() {
    if !device_up() {
        return;
    }
    for (name, rows, cols) in SHAPES {
        for k in [31usize, 255] {
            let Some((bits, cells, rel)) = q82_mm_v2_diff_for_test(rows, cols, k) else {
                eprintln!("q82 diff {name}: kernels missing — skipping");
                return;
            };
            println!("q82 diff {name:>8} {rows}x{cols} k={k}: {bits} of {cells} cells differ, relative rms {rel:.3e}");
            assert!(rel < 1e-5, "the pipelined q8_2f GEMM drifted from the coop one: {rel:.3e}");
        }
    }
}
