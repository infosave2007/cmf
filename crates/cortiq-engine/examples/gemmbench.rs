//! Which side should a small f32 GEMM run on: `gemm_nt`'s own arbitration,
//! or pinned to the host?
//!
//! Written to answer that for the LTX LoRA branch, whose shapes are narrow
//! (rank 128) against a wide activation. The generic `GemmNt` probe will send
//! them to the device, where they queue behind the q4tp projection they are
//! standing beside — the arithmetic is a fraction of the base GEMM's, the
//! latency is not.
//!
//!   cargo run --release -p cortiq-engine --example gemmbench

fn main() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let cases = [(384usize, 4096usize, 128usize), (384, 128, 4096), (384, 16384, 128), (384, 128, 16384)];
    for (n, k, m) in cases {
        let x = vec![0.01f32; n * k];
        let w = vec![0.02f32; m * k];
        let mut y = vec![0f32; n * m];
        // warm
        cortiq_engine::fcd_ops::gemm_nt(&x, &w, &mut y, n, k, m, None);
        let t = std::time::Instant::now();
        for _ in 0..10 { cortiq_engine::fcd_ops::gemm_nt(&x, &w, &mut y, n, k, m, None); }
        let d = t.elapsed().as_secs_f64() / 10.0;
        let g = 2.0 * n as f64 * k as f64 * m as f64 / d / 1e9;
        let t2 = std::time::Instant::now();
        for _ in 0..10 {
            cortiq_engine::gpu::cpu_scope(|| cortiq_engine::fcd_ops::gemm_nt(&x, &w, &mut y, n, k, m, None));
        }
        let d2 = t2.elapsed().as_secs_f64() / 10.0;
        let g2 = 2.0 * n as f64 * k as f64 * m as f64 / d2 / 1e9;
        println!("{n}x{k}x{m}:  default {:.2} ms ({g:.0} GF/s)   cpu_scope {:.2} ms ({g2:.0} GF/s)", d * 1e3, d2 * 1e3);
    }
}
