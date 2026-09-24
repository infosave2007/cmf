//! Metal Z-Image kernel microbench (plan WP3, M5).
//!
//! zimage_metal_bench gemm [n=1056,4224] [reps=5] [variants=base,wt]
//!
//! Times `zi_q8mm` (int8 weights staged as half, f32 accumulation) on the
//! four DiT GEMM shapes, alternating the variants inside one process, and
//! prints the GPU time (min / median of `reps`), the effective TFLOP/s and
//! the relative error against an f64 host reference on sampled outputs.
#[cfg(target_os = "macos")]
fn main() {
    use cortiq_engine::gpu_metal::zimage::bench_gemm;
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("gemm");
    let ns: Vec<usize> = a
        .get(2)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or(vec![1056, 4224]);
    let reps: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(5);
    let variants: Vec<String> = a
        .get(4)
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or(vec!["base".into(), "wt".into()]);
    if mode == "peak" {
        for round in 0..3 {
            for ty in ["h", "f"] {
                let tf = cortiq_engine::gpu_metal::zimage::bench_mma_peak(ty, 5).unwrap();
                println!("mma peak {ty} r{round}: {tf:.2} TF/s");
            }
        }
    }
    if mode == "flash" {
        let vs: Vec<String> = a
            .get(4)
            .map(|s| s.split(',').map(str::to_string).collect())
            .unwrap_or(vec!["q64pf".into(), "q32".into(), "q32s".into(), "q64pfs".into()]);
        for &n in &ns {
            for round in 0..2 {
                for v in &vs {
                    let (mn, med, err) = cortiq_engine::gpu_metal::zimage::bench_flash(n, reps, v).unwrap();
                    let fl = 4.0 * (n as f64) * (n as f64) * 3840.0;
                    println!(
                        "flash n {n:5} {v:4} r{round}: min {mn:7.3} ms ({:.2} TF)  med {med:7.3} ms ({:.2} TF)  rel {err:.2e}",
                        fl / mn / 1e9,
                        fl / med / 1e9
                    );
                }
            }
        }
    }
    if mode == "gemm" {
        let shapes = [(3840usize, 3840usize, "qkv/o"), (10240, 3840, "w1/w3"), (3840, 10240, "w2")];
        for &n in &ns {
            for &(rows, k, name) in &shapes {
                for round in 0..2 {
                    for v in &variants {
                        let Some((mn, med, err)) = bench_gemm(rows, k, n, reps, v) else {
                            eprintln!("no Metal device");
                            return;
                        };
                        let fl = 2.0 * rows as f64 * k as f64 * n as f64;
                        println!(
                            "n {n:5} {name:6} {rows:5}x{k:5} {v:5} r{round}: min {mn:7.3} ms ({:.2} TF)  med {med:7.3} ms ({:.2} TF)  rel {err:.2e}",
                            fl / mn / 1e9,
                            fl / med / 1e9
                        );
                    }
                }
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}
