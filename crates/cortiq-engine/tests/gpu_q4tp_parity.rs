//! Does the device's q4tp matvec compute what the CPU's does?
//!
//! Runs only when pointed at a real q4tp model and a working GPU:
//!
//!     CMF_Q4TP_PARITY=/root/gtoy.cmf cargo test -p cortiq-engine --test gpu_q4tp_parity -- --nocapture
//!
//! It exists to bisect a divergence seen end to end: a whole MoE block on
//! the device answered 220% away from the CPU, and the two candidates —
//! the kernel itself and the block that wires it — cannot be told apart
//! from the output of a generation.

#[cfg(feature = "gpu")]
#[test]
fn device_q4tp_matvec_matches_the_cpu() {
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
        if cols % 32 != 0 || rows == 0 {
            continue;
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7) as f32 * 0.013).sin()).collect();

        let t = cortiq_engine::qtensor::QTensor::from_model(&model, &e.name).expect("tensor");
        let mut cpu = vec![0.0f32; rows];
        t.matvec(&xs, &mut cpu, None);

        let mut gpu = vec![0.0f32; rows];
        if !cortiq_engine::gpu_wgpu::q4tp_matvec_for_test(&model, idx, &xs, rows, cols, &mut gpu) {
            eprintln!("устройство отказалось от {} — пропуск", e.name);
            continue;
        }
        let num: f32 = cpu.iter().zip(&gpu).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = cpu.iter().map(|a| a * a).sum::<f32>().max(1e-20);
        let rel = (num / den).sqrt();
        println!("{}: [{rows}x{cols}] расхождение {rel:.3e}", e.name);
        assert!(
            rel < 1e-3,
            "{}: устройство разошлось с CPU на {rel:.3e}",
            e.name
        );
        checked += 1;
        if checked >= 6 {
            break;
        }
    }
    println!("сверено тензоров: {checked}");
}
