//! The B-axis q4tp kernel against the single-vector one, bit for bit, on a
//! real toy tensor. The batched frame leans on `mv4_b` for q, wo_b and
//! next-q; if its add order differs from the walk's `mvw` family, every
//! "exact" comparison downstream is chasing this.
#![cfg(feature = "gpu")]

use std::sync::Arc;

#[test]
fn batched_q4tp_matvec_matches_single_bitwise() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    let dir = std::env::var("CMF_TOY_DIR").unwrap_or_else(|_| {
        "/private/tmp/claude-501/-Users-oleg-Documents-cortiq-bot-cmfpublic/674db62a-643b-4641-b330-5feed6d40b67/scratchpad".into()
    });
    let path = std::path::Path::new(&dir).join("q4.cmf");
    if !path.exists() {
        eprintln!("стендов нет — пропуск");
        return;
    }
    let model = Arc::new(cortiq_core::CmfModel::open_sharded(path.to_str().unwrap()).unwrap());
    for name in [
        "model.layers.0.self_attn.wq_b.weight",
        "model.layers.0.self_attn.wo_b.weight",
        "model.layers.0.self_attn.wq_a.weight",
    ] {
        let e = model
            .tensors
            .iter()
            .position(|t| t.name == name)
            .unwrap_or_else(|| panic!("нет {name}"));
        let (rows, cols) = {
            let t = &model.tensors[e];
            (t.shape[0], t.shape[1])
        };
        let b = 5usize;
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i as f32 * 0.61).sin() * 0.5) + ((i % 13) as f32 * 0.01))
            .collect();
        let mut single = vec![0.0f32; b * rows];
        for t in 0..b {
            let mut y = vec![0.0f32; rows];
            assert!(cortiq_engine::gpu_wgpu::q4tp_matvec(
                &model,
                e,
                &xs[t * cols..(t + 1) * cols],
                rows,
                cols,
                &mut y,
            ));
            single[t * rows..(t + 1) * rows].copy_from_slice(&y);
        }
        let mut batched = vec![0.0f32; b * rows];
        assert!(cortiq_engine::gpu_wgpu::q4tp_matvec_batch_for_test(
            &model,
            e,
            &xs,
            b,
            rows,
            cols,
            &mut batched,
        ));
        let n_diff = single
            .iter()
            .zip(&batched)
            .filter(|(a, c)| a.to_bits() != c.to_bits())
            .count();
        let mx = single
            .iter()
            .zip(&batched)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            n_diff == 0,
            "{name}: {n_diff}/{} бит-различий, max {mx:e}",
            single.len()
        );
        println!("{name}: bit-exact на B={b}");
    }
}
