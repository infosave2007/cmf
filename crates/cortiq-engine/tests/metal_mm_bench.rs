//! Metal q4tp GEMM cost by batch (ignored; needs a real q4tp file):
//! `CMF_BENCH_MODEL=/path/model.cmf cargo test --release -p cortiq-engine --test metal_mm_bench -- --ignored --nocapture`
#![cfg(target_os = "macos")]
use cortiq_core::CmfModel;
use std::sync::Arc;

#[test]
#[ignore]
fn metal_q4tp_matmat_cost_by_batch() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    // the dense FFN gate of layer 0 and the down projection: the two shapes a verify streams most
    let picks = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
    ];
    for name in picks {
        let idx = model.tensor_index(name).expect("tensor");
        let e = &model.tensors[idx];
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let mb = e.nbytes as f64 / 1e6;
        for (which, label) in [(0u32, "mv1"), (1, "bk"), (2, "gemm"), (3, "n8")] {
            for &b in &[1usize, 2, 3, 4, 5, 8, 128, 256, 512] {
                if which == 0 && b != 1 {
                    continue;
                }
                if which != 2 && b > 8 {
                    continue;
                }
                if let Some(ms) = cortiq_engine::gpu_metal::q4tp_kernel_bench(
                    &model, idx, b, rows, cols, which, 20,
                ) {
                    println!(
                        "KERNEL {label:4} {name} b={b}: {ms:6.3} ms/dispatch  ({:.0} GB/s of weight)",
                        mb / ms
                    );
                }
            }
        }
    }
    // `CMF_BENCH_KERNELS_ONLY=1`: the kernel table alone
    if std::env::var("CMF_BENCH_KERNELS_ONLY").is_ok() {
        return;
    }
    for name in picks {
        let idx = model.tensor_index(name).expect("tensor");
        let e = &model.tensors[idx];
        let (rows, cols) = (e.shape[0], e.shape[1]);
        for &b in &[1usize, 5, 8] {
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
                .collect();
            let mut o1 = vec![0f32; b * rows];
            let mut o2 = vec![0f32; b * rows];
            assert!(cortiq_engine::gpu_metal::q4tp_matvec_batch(
                &model, idx, &xs, b, rows, cols, &mut o1
            ));
            assert!(cortiq_engine::gpu_metal::q4tp_mm_n8_batch(
                &model, idx, &xs, b, rows, cols, &mut o2
            ));
            let mut md = 0f32;
            let mut rms1 = 0f64;
            let mut rmsd = 0f64;
            for (a, c) in o1.iter().zip(&o2) {
                md = md.max((a - c).abs());
                rms1 += (*a as f64).powi(2);
                rmsd += ((a - c) as f64).powi(2);
            }
            println!(
                "PARITY n8 vs bk {name} b={b}: max|d|={md:.3e} rel-rms={:.3e}",
                (rmsd / rms1.max(1e-30)).sqrt()
            );
        }
        for &b in &[32usize, 64, 512, 700] {
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
                .collect();
            let mut o1 = vec![0f32; b * rows];
            let mut o2 = vec![0f32; b * rows];
            // reference: bk in slices of 8
            for c0 in (0..b).step_by(8) {
                let bb = (b - c0).min(8);
                assert!(cortiq_engine::gpu_metal::q4tp_matvec_batch(
                    &model,
                    idx,
                    &xs[c0 * cols..],
                    bb,
                    rows,
                    cols,
                    &mut o1[c0 * rows..]
                ));
            }
            assert!(cortiq_engine::gpu_metal::q4tp_matmat(
                &model, idx, &xs, b, rows, cols, &mut o2
            ));
            let mut md = 0f32;
            let mut rms1 = 0f64;
            let mut rmsd = 0f64;
            for (a, c) in o1.iter().zip(&o2) {
                md = md.max((a - c).abs());
                rms1 += (*a as f64).powi(2);
                rmsd += ((a - c) as f64).powi(2);
            }
            println!(
                "PARITY gemm vs bk {name} b={b}: max|d|={md:.3e} rel-rms={:.3e}",
                (rmsd / rms1.max(1e-30)).sqrt()
            );
        }
        // the batched matvec (weights once): b ≤ 8
        for &b in &[1usize, 2, 5, 8] {
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
                .collect();
            let mut out = vec![0f32; b * rows];
            assert!(cortiq_engine::gpu_metal::q4tp_matvec_batch(
                &model, idx, &xs, b, rows, cols, &mut out
            ));
            let reps = 5;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                cortiq_engine::gpu_metal::q4tp_matvec_batch(
                    &model, idx, &xs, b, rows, cols, &mut out,
                );
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let bytes = e.nbytes as f64;
            println!(
                "BK  {name} {rows}x{cols} b={b:2}: {ms:7.2} ms  ({:.0} GB/s of weight, {:.2} ms/row)",
                bytes / ms / 1e6,
                ms / b as f64
            );
        }
        for &b in &[1usize, 2, 5, 8, 16, 32] {
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
                .collect();
            let mut out = vec![0f32; b * rows];
            // warm
            assert!(cortiq_engine::gpu_metal::q4tp_matmat(
                &model, idx, &xs, b, rows, cols, &mut out
            ));
            let reps = 5;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                cortiq_engine::gpu_metal::q4tp_matmat(&model, idx, &xs, b, rows, cols, &mut out);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let bytes = e.nbytes as f64;
            println!(
                "GEMM {name} {rows}x{cols} b={b:2}: {ms:7.2} ms  ({:.0} GB/s of weight, {:.1} ms/row)",
                bytes / ms / 1e6,
                ms / b as f64
            );
        }
    }
}

/// Batch-kernel lab: MSL from a file against the built-in batched matvec and narrow GEMM, over the
/// q4tp projections of the first layers, arms alternating.
/// `CMF_BENCH_MODEL=model.cmf CMF_LAB_MSL=lab.metal CMF_LAB_ARMS=name:tg_rows:tg_threads,...
///  [CMF_LAB_B=2,3,4] [CMF_LAB_LAYERS=4] [CMF_LAB_ROUNDS=3] cargo test --release -p cortiq-engine
///  --test metal_mm_bench batch_kernel_lab -- --ignored --nocapture`
#[test]
#[ignore]
fn batch_kernel_lab() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    let src = std::env::var("CMF_LAB_MSL")
        .ok()
        .map(|p| std::fs::read_to_string(p).expect("CMF_LAB_MSL"));
    let num = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let layers = num("CMF_LAB_LAYERS", 4);
    let rounds = num("CMF_LAB_ROUNDS", 3);
    let bs: Vec<usize> = std::env::var("CMF_LAB_B")
        .unwrap_or_else(|_| "2,3,4".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let idxs: Vec<usize> = (0..model.tensors.len())
        .filter(|&i| {
            let e = &model.tensors[i];
            let layer = e
                .name
                .strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok());
            e.dtype == cortiq_core::TensorDtype::Q4TiledP
                && e.shape.len() == 2
                && e.shape[1] % 64 == 0
                && layer.is_some_and(|l| l < layers)
        })
        .collect();
    let bytes: f64 = idxs.iter().map(|&i| model.tensors[i].nbytes as f64).sum();
    println!("LAB {} tensors, {:.0} MB a pass", idxs.len(), bytes / 1e6);
    let mut arms: Vec<(String, bool, usize, usize)> =
        vec![("bk".into(), false, 32, 256), ("n8".into(), false, 64, 128), ("bs".into(), false, 32, 256)];
    for a in std::env::var("CMF_LAB_ARMS").unwrap_or_default().split(',') {
        let f: Vec<&str> = a.split(':').collect();
        if f.len() == 3 {
            arms.push((f[0].into(), true, f[1].parse().unwrap(), f[2].parse().unwrap()));
        }
    }
    for &b in &bs {
        let mut best = vec![f64::MAX; arms.len()];
        let mut err = vec![0f32; arms.len()];
        let mut reference: Option<Vec<f32>> = None;
        for round in 0..rounds + 1 {
            for (ai, (name, lab, tg_rows, tg_threads)) in arms.iter().enumerate() {
                let s = if *lab { src.as_deref() } else { None };
                let Some((ms, y)) = cortiq_engine::gpu_metal::q4tp_batch_kernel_lab(
                    &model, &idxs, b, name, s, *tg_rows, *tg_threads, 3,
                ) else {
                    continue;
                };
                // round 0 warms the pipelines and the file
                if round > 0 {
                    best[ai] = best[ai].min(ms);
                }
                match &reference {
                    None => reference = Some(y),
                    Some(r) => {
                        let rms = (r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
                        let d = (r.iter().zip(&y).map(|(a, c)| ((a - c) as f64).powi(2)).sum::<f64>()
                            / r.len() as f64)
                            .sqrt();
                        err[ai] = (d / rms.max(1e-30)) as f32;
                    }
                }
            }
        }
        for (ai, (name, ..)) in arms.iter().enumerate() {
            if best[ai] < f64::MAX {
                println!(
                    "LAB b={b} {name:20} {:8.2} ms/pass {:6.1} GB/s  rel rms vs bk {:.2e}",
                    best[ai],
                    bytes / best[ai] / 1e6,
                    err[ai]
                );
            } else {
                println!("LAB b={b} {name:20} unavailable");
            }
        }
    }
}
