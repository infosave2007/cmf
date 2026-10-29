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

/// The q4tp decode matvec per projection kind (`q4tp_matvec_m`) and the
/// fused gate|up|GELU pair, over every layer. `CMF_Q8F_MSL` +
/// `CMF_Q4MV_KERNELS` (`name:rows_per_sg:sgs`, comma-separated; a name
/// containing `_gu` takes the gate|up pair) add candidates, checked against
/// the built-in of their kind.
#[test]
#[ignore]
fn metal_q4tp_matvec_cost_by_kind() {
    let Ok(path) = std::env::var("CMF_BENCH_MODEL") else {
        eprintln!("CMF_BENCH_MODEL not set — skipping");
        return;
    };
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));
    let src = std::env::var("CMF_Q8F_MSL")
        .ok()
        .map(|p| std::fs::read_to_string(p).expect("read MSL"));
    let mut extra: Vec<(String, usize, usize)> = Vec::new();
    if let (Some(_), Ok(list)) = (src.as_deref(), std::env::var("CMF_Q4MV_KERNELS")) {
        for spec in list.split(',') {
            let p: Vec<&str> = spec.split(':').collect();
            extra.push((p[0].into(), p[1].parse().unwrap(), p[2].parse().unwrap()));
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
    let set = |k: &str| -> Vec<usize> {
        (0..layers)
            .map(|l| {
                model
                    .tensor_index(&format!("model.layers.{l}.{k}.weight"))
                    .expect("tensor")
            })
            .collect()
    };
    let mut kinds: Vec<(String, Vec<usize>, Option<Vec<usize>>)> = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    ]
    .iter()
    .map(|k| (k.to_string(), set(k), None))
    .collect();
    kinds.push(("gate|up".into(), set("mlp.gate_proj"), Some(set("mlp.up_proj"))));
    if let Some(h) = model.tensor_index("model.embed_tokens.weight") {
        kinds.push(("head".into(), vec![h], None));
    }
    for (kind, idxs, up) in &kinds {
        let mut bytes: f64 = idxs.iter().map(|&i| model.tensors[i].nbytes as f64).sum();
        if let Some(u) = up {
            bytes += u.iter().map(|&i| model.tensors[i].nbytes as f64).sum::<f64>();
        }
        let builtin = if up.is_some() {
            "q4tp_matvec_m_gu_gelu"
        } else {
            "q4tp_matvec_m"
        };
        let mut arms: Vec<(String, Option<&str>, usize, usize)> = vec![(builtin.into(), None, 4, 4)];
        for (n, r, s) in &extra {
            if n.contains("_gu") == up.is_some() {
                arms.push((n.clone(), src.as_deref(), *r, *s));
            }
        }
        let mut reference: Option<Vec<f32>> = None;
        for (name, src, rps, sgs) in &arms {
            let run = |reps| {
                cortiq_engine::gpu_metal::q4tp_mv_kernel_bench(
                    &model,
                    idxs,
                    up.as_deref(),
                    name,
                    *src,
                    *rps,
                    *sgs,
                    reps,
                )
            };
            let _ = run(1);
            let Some((ms, y)) = run(reps) else {
                println!("KERNEL {name} {kind}: unavailable");
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
                "KERNEL {name:24} {kind:17} {ms:7.3} ms/pass {:6.1} GB/s  rel err {err:.2e}",
                bytes / ms / 1e6
            );
        }
    }
}

/// The decode attend kernels alone on synthetic K/V (Spark-X2.6 geometry:
/// 16 Q heads, 4 KV heads, width 256), GPU µs per dispatch by depth.
#[test]
#[ignore]
fn metal_decode_attend_cost_by_depth() {
    let ns: Vec<usize> = std::env::var("CMF_ATT_NS")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![64, 150, 512, 1000, 3000]);
    for n in ns {
        for (which, label) in [(0u32, "per-head"), (1, "split-K")] {
            let _ = cortiq_engine::gpu_metal::attend_kernel_bench(16, 4, 256, n, which, 4);
            if let Some(us) = cortiq_engine::gpu_metal::attend_kernel_bench(16, 4, 256, n, which, 64) {
                println!("ATTEND {label:9} n={n:5} {us:8.1} us/dispatch");
            }
        }
    }
}

/// Prefill GEMM rate of q8 (`CMF_BENCH_MODEL`, q8_2f file) against q4tp
/// (`CMF_BENCH_MODEL2`, same model in q4tp) on the layer-0 gate, b rows,
/// alternating rounds in one process (a drifting GPU loads both alike).
#[test]
#[ignore]
fn metal_gemm_q8_vs_q4tp() {
    let (Ok(p8), Ok(p4)) = (std::env::var("CMF_BENCH_MODEL"), std::env::var("CMF_BENCH_MODEL2")) else {
        eprintln!("CMF_BENCH_MODEL / CMF_BENCH_MODEL2 not set — skipping");
        return;
    };
    let m8 = Arc::new(CmfModel::open_sharded(&p8).expect("open q8"));
    let m4 = Arc::new(CmfModel::open_sharded(&p4).expect("open q4tp"));
    let b: usize = std::env::var("CMF_Q8MM_B").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    for kind in ["mlp.gate_proj", "mlp.down_proj"] {
        let name = format!("model.layers.0.{kind}.weight");
        let i8 = m8.tensor_index(&name).unwrap();
        let i4 = m4.tensor_index(&name).unwrap();
        let (rows, cols) = (m4.tensors[i4].shape[0], m4.tensors[i4].shape[1]);
        let flop = 2.0 * (rows * cols * b) as f64;
        for round in 0..6 {
            let (ms8, _) = cortiq_engine::gpu_metal::q8_mm_kernel_bench(
                &m8, &[i8], "q8_mul_mm", None, b, 64, 32, 128, 8,
            )
            .unwrap();
            let ms4 =
                cortiq_engine::gpu_metal::q4tp_kernel_bench(&m4, i4, b, rows, cols, 2, 8).unwrap();
            println!(
                "GEMM {kind:14} b={b} r{round}: q8 {:.2} TFLOP/s | q4tp {:.2} TFLOP/s",
                flop / ms8 / 1e9,
                flop / ms4 / 1e9
            );
        }
    }
}
