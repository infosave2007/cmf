//! Metal q8_2f decode matvec cost per projection kind (ignored; needs a real
//! q8_2f file):
//! `CMF_BENCH_MODEL=/path/model.cmf cargo test --release -p cortiq-engine --test metal_q8f_bench -- --ignored --nocapture`
//! `CMF_Q8F_MSL=/path/k.metal` adds the candidate entry points listed in
//! `CMF_Q8F_KERNELS` (`name:rows_per_sg:sgs`, comma-separated), each checked
//! against `q8f_matvec` on the same input.
#![cfg(target_os = "macos")]
use cortiq_core::CmfModel;
use std::sync::Arc;

#[test]
#[ignore]
fn metal_q8f_matvec_cost_by_kind() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    let src = std::env::var("CMF_Q8F_MSL")
        .ok()
        .map(|p| std::fs::read_to_string(p).expect("read MSL"));
    let mut arms: Vec<(String, Option<&str>, usize, usize)> = vec![
        ("q8f_matvec".into(), None, 1, 8),
        ("q8f_matvec_r4".into(), None, 4, 8),
    ];
    if let (Some(src), Ok(list)) = (src.as_deref(), std::env::var("CMF_Q8F_KERNELS")) {
        for spec in list.split(',') {
            let p: Vec<&str> = spec.split(':').collect();
            arms.push((p[0].into(), Some(src), p[1].parse().unwrap(), p[2].parse().unwrap()));
        }
    }
    let reps: usize = std::env::var("CMF_Q8F_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let layers = (0..)
        .take_while(|l| {
            model
                .tensor_index(&format!("model.layers.{l}.mlp.gate_proj.weight"))
                .is_some()
        })
        .count();
    let kinds = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ];
    let mut sets: Vec<(String, Vec<usize>)> = kinds
        .iter()
        .map(|k| {
            let v = (0..layers)
                .map(|l| {
                    model
                        .tensor_index(&format!("model.layers.{l}.{k}.weight"))
                        .expect("tensor")
                })
                .collect();
            (k.to_string(), v)
        })
        .collect();
    if let Some(h) = model
        .tensor_index("lm_head.weight")
        .or_else(|| model.tensor_index("model.embed_tokens.weight"))
    {
        sets.push(("head".into(), vec![h]));
    }
    for (kind, idxs) in &sets {
        let misaligned = idxs
            .iter()
            .filter(|&&i| model.entry_abs_offset(&model.tensors[i]).is_none_or(|a| a % 16 != 0))
            .count();
        if misaligned > 0 {
            println!("{kind}: {misaligned} tensors not 16-byte aligned");
        }
        let bytes: f64 = idxs.iter().map(|&i| model.tensors[i].nbytes as f64).sum();
        let mut reference: Option<Vec<f32>> = None;
        for (name, src, rps, sgs) in &arms {
            // One warm-up pass, then the timed reps.
            let _ = cortiq_engine::gpu_metal::q8f_kernel_bench(&model, idxs, name, *src, *rps, *sgs, 1);
            let Some((ms, y)) =
                cortiq_engine::gpu_metal::q8f_kernel_bench(&model, idxs, name, *src, *rps, *sgs, reps)
            else {
                println!("KERNEL {name} {kind}: unavailable");
                continue;
            };
            let err = match &reference {
                None => {
                    reference = Some(y.clone());
                    0.0
                }
                Some(r) => {
                    let scale = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                    r.iter().zip(&y).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
                }
            };
            println!(
                "KERNEL {name:22} {kind:17} {:7.3} ms/pass {:6.1} GB/s  rel err {err:.2e}",
                ms,
                bytes / ms / 1e6
            );
        }
    }
}

/// The prefill GEMM over the same projection kinds, b rows per pass
/// (`CMF_Q8MM_B`, default 512). `CMF_Q8F_MSL` + `CMF_Q8MM_KERNELS`
/// (`name:tile_rows:tile_b:threads`, comma-separated) add candidates,
/// checked against `q8_mul_mm`.
#[test]
#[ignore]
fn metal_q8_mul_mm_cost_by_kind() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    let src = std::env::var("CMF_Q8F_MSL")
        .ok()
        .map(|p| std::fs::read_to_string(p).expect("read MSL"));
    let b: usize = std::env::var("CMF_Q8MM_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let mut arms: Vec<(String, Option<&str>, usize, usize, usize)> =
        vec![("q8_mul_mm".into(), None, 64, 32, 128)];
    if let (Some(src), Ok(list)) = (src.as_deref(), std::env::var("CMF_Q8MM_KERNELS")) {
        for spec in list.split(',') {
            let p: Vec<&str> = spec.split(':').collect();
            arms.push((
                p[0].into(),
                Some(src),
                p[1].parse().unwrap(),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
            ));
        }
    }
    let layers: usize = std::env::var("CMF_Q8MM_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    for kind in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ] {
        let idxs: Vec<usize> = (0..layers)
            .map(|l| {
                model
                    .tensor_index(&format!("model.layers.{l}.{kind}.weight"))
                    .expect("tensor")
            })
            .collect();
        let flop: f64 = idxs
            .iter()
            .map(|&i| 2.0 * (model.tensors[i].shape[0] * model.tensors[i].shape[1] * b) as f64)
            .sum();
        let mut reference: Option<Vec<f32>> = None;
        for (name, src, tr, tb, th) in &arms {
            let run = |reps| {
                cortiq_engine::gpu_metal::q8_mm_kernel_bench(
                    &model, &idxs, name, *src, b, *tr, *tb, *th, reps,
                )
            };
            let _ = run(1);
            let Some((ms, y)) = run(3) else {
                println!("MM {name} {kind}: unavailable");
                continue;
            };
            let err = match &reference {
                None => {
                    reference = Some(y);
                    0.0
                }
                Some(r) => {
                    let scale = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                    r.iter().zip(&y).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
                }
            };
            println!(
                "MM {name:20} {kind:17} b={b} {:8.3} ms/pass {:6.2} TFLOP/s  rel err {err:.2e}",
                ms,
                flop / ms / 1e9
            );
        }
    }
}
