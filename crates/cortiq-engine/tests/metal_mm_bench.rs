//! Metal q4tp GEMM cost by batch (ignored; needs a real q4tp file):
//! `CMF_BENCH_MODEL=/path/model.cmf cargo test --release -p cortiq-engine --test metal_mm_bench -- --ignored --nocapture`
#![cfg(target_os = "macos")]
use std::sync::Arc;
use cortiq_core::CmfModel;

#[test]
#[ignore]
fn metal_q4tp_matmat_cost_by_batch() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    // the dense FFN gate of layer 0 and the down projection: the two shapes a verify streams most
    let picks = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"];
    for name in picks {
        let idx = model.tensor_index(name).expect("tensor");
        let e = &model.tensors[idx];
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let mb = e.nbytes as f64 / 1e6;
        for (which, label) in [(0u32, "mv1"), (1, "bk"), (2, "gemm")] {
            for &b in &[1usize, 2, 5, 8] {
                if which == 0 && b != 1 { continue; }
                if let Some(ms) = cortiq_engine::gpu_metal::q4tp_kernel_bench(&model, idx, b, rows, cols, which, 20) {
                    println!("KERNEL {label:4} {name} b={b}: {ms:6.3} ms/dispatch  ({:.0} GB/s of weight)", mb / ms);
                }
            }
        }
    }
    for name in picks {
        let idx = model.tensor_index(name).expect("tensor");
        let e = &model.tensors[idx];
        let (rows, cols) = (e.shape[0], e.shape[1]);
        // the batched matvec (weights once): b ≤ 8
        for &b in &[1usize, 2, 5, 8] {
            let xs: Vec<f32> = (0..b * cols).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
            let mut out = vec![0f32; b * rows];
            assert!(cortiq_engine::gpu_metal::q4tp_matvec_batch(&model, idx, &xs, b, rows, cols, &mut out));
            let reps = 5;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                cortiq_engine::gpu_metal::q4tp_matvec_batch(&model, idx, &xs, b, rows, cols, &mut out);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let bytes = e.nbytes as f64;
            println!("BK  {name} {rows}x{cols} b={b:2}: {ms:7.2} ms  ({:.0} GB/s of weight, {:.2} ms/row)", bytes / ms / 1e6, ms / b as f64);
        }
        for &b in &[1usize, 2, 5, 8, 16, 32] {
            let xs: Vec<f32> = (0..b * cols).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
            let mut out = vec![0f32; b * rows];
            // warm
            assert!(cortiq_engine::gpu_metal::q4tp_matmat(&model, idx, &xs, b, rows, cols, &mut out));
            let reps = 5;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                cortiq_engine::gpu_metal::q4tp_matmat(&model, idx, &xs, b, rows, cols, &mut out);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let bytes = e.nbytes as f64;
            println!("GEMM {name} {rows}x{cols} b={b:2}: {ms:7.2} ms  ({:.0} GB/s of weight, {:.1} ms/row)", bytes / ms / 1e6, ms / b as f64);
        }
    }
}
