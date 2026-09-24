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
    if mode == "corun" {
        // CPU (Accelerate sgemm) alone, GPU (zi_q8mm) alone, then both at once
        let (n, k, m) = (1056usize, 3840usize, 2560usize);
        let x: Vec<f32> = (0..n * k).map(|i| ((i * 7) % 13) as f32 * 0.01).collect();
        let w: Vec<f32> = (0..m * k).map(|i| ((i * 5) % 11) as f32 * 0.01).collect();
        let mut y = vec![0f32; n * m];
        let fl_cpu = 2.0 * (n * k * m) as f64;
        let cpu_run = |y: &mut Vec<f32>, secs: f64| -> f64 {
            let t = std::time::Instant::now();
            let mut it = 0usize;
            while t.elapsed().as_secs_f64() < secs {
                cortiq_engine::fcd_ops::gemm_nt(&x, &w, y, n, k, m, None);
                it += 1;
            }
            fl_cpu * it as f64 / t.elapsed().as_secs_f64() / 1e12
        };
        let gpu_run = |reps: usize| -> f64 {
            let (_, med, _) = bench_gemm(10240, 3840, 1056, reps, "base").unwrap();
            2.0 * 10240.0 * 3840.0 * 1056.0 / med / 1e9
        };
        for round in 0..2 {
            let c = cpu_run(&mut y, 4.0);
            let g = gpu_run(40);
            let (c2, g2) = std::thread::scope(|s| {
                let h = s.spawn(|| {
                    let mut yy = vec![0f32; n * m];
                    cpu_run(&mut yy, 4.0)
                });
                std::thread::sleep(std::time::Duration::from_millis(200));
                let g2 = gpu_run(80);
                (h.join().unwrap(), g2)
            });
            println!("corun r{round}: cpu alone {c:.2} TF · gpu alone {g:.2} TF · together cpu {c2:.2} + gpu {g2:.2} = {:.2} TF", c2 + g2);
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
