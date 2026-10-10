//! GEMM micro-benchmark for the EmbeddingGemma 2 Metal kernels (macOS).
//! `cargo run --release -p cortiq-engine --example egemma2_gemm_bench -- [kernel]`
//! Shapes are the text encoder's at 8192 tokens; `CMF_EGEMMA2_MSL=<file>`
//! benchmarks an edited kernel source without a rebuild.
fn main() {
    #[cfg(target_os = "macos")]
    {
        let kernel = std::env::args().nth(1);
        let shapes = [
            (8192usize, 512usize, 2048usize),
            (8192, 512, 1024),
            (8192, 2048, 512),
            (8192, 512, 512),
            (8192, 1024, 512),
        ];
        let mut tot_fl = 0f64;
        let mut tot_ms = 0f64;
        for (n, k, m) in shapes {
            match cortiq_engine::gpu_metal::egemma2::bench_gemm(n, k, m, 5, kernel.as_deref()) {
                Ok((tf, err)) => {
                    let fl = 2.0 * (n * k * m) as f64;
                    tot_fl += fl;
                    tot_ms += fl / tf / 1e9;
                    println!("{n}x{k} · {m}x{k}ᵀ: {tf:.2} TF/s, rel err {err:.2e}");
                }
                Err(e) => println!("{n}x{k}x{m}: {e}"),
            }
        }
        println!("all shapes: {:.2} TF/s", tot_fl / tot_ms / 1e9);
    }
}
