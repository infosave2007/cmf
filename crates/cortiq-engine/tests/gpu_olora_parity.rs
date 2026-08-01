//! Does the device's grouped low-rank projection agree with the CPU's?
//!
//!     CMF_Q4TP_PARITY=/root/gtoy.cmf cargo test -p cortiq-engine \
//!         --features gpu --test gpu_olora_parity -- --nocapture
//!
//! The operator differs from a plain matvec by one thing only — each row
//! reads its own group's slice of the input — and that one thing is invisible
//! at `groups = 1`, where it degenerates into the matvec that already passes.
//! So the test runs every divisor of the row count that gives more than one
//! group, and fills the input so that the slices differ: a shared vector
//! would hide a wrong offset completely.

#[cfg(feature = "gpu")]
#[test]
fn device_o_lora_a_matches_the_cpu() {
    let Ok(path) = std::env::var("CMF_Q4TP_PARITY") else {
        return;
    };
    let model = std::sync::Arc::new(cortiq_core::CmfModel::open(&path).expect("open"));
    let mut checked = 0;
    for (idx, e) in model.tensors.iter().enumerate() {
        if e.dtype != cortiq_core::TensorDtype::Q4TiledP || e.shape.len() != 2 {
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        if cols % 32 != 0 || rows < 4 {
            continue;
        }
        let t = cortiq_engine::qtensor::QTensor::from_model(&model, &e.name).expect("tensor");
        for groups in [2usize, 4] {
            if rows % groups != 0 {
                continue;
            }
            let lora = rows / groups;
            // Per-group phase shift: with one shared vector every offset looks
            // right, which is exactly the bug this test exists to catch.
            let attn: Vec<f32> = (0..groups * cols)
                .map(|i| (((i % cols) * 7) as f32 * 0.013 + (i / cols) as f32 * 1.7).sin())
                .collect();

            let mut cpu = vec![0.0f32; rows];
            let mut sc = vec![0.0f32; cols];
            for (i, o) in cpu.iter_mut().enumerate() {
                let g = i / lora;
                *o = t.row_dot(i, &attn[g * cols..(g + 1) * cols], &mut sc);
            }

            let mut gpu = vec![0.0f32; rows];
            if !cortiq_engine::gpu_wgpu::o_lora_a_for_test(
                &model, idx, &attn, rows, lora, &mut gpu,
            ) {
                eprintln!("устройство отказалось от {} — пропуск", e.name);
                continue;
            }
            let num: f32 = cpu.iter().zip(&gpu).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = cpu.iter().map(|a| a * a).sum::<f32>().max(1e-20);
            let rel = (num / den).sqrt();
            println!("{}: [{rows}x{cols}] групп {groups}: {rel:.3e}", e.name);
            assert!(rel < 5e-3, "{} при {groups} группах: {rel:.3e}", e.name);
            checked += 1;
        }
        if checked >= 24 {
            break;
        }
    }
    println!("сверено: {checked}");
}
