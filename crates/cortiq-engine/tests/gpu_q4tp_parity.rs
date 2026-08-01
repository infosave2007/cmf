//! Does the device's q4tp matvec compute what the CPU's does?
//!
//! Runs only when pointed at a real q4tp model and a working GPU:
//!
//!     CMF_Q4TP_PARITY=/root/gtoy.cmf cargo test -p cortiq-engine --test gpu_q4tp_parity -- --nocapture
//!
//!
//! Run with `CMF_SDOT=0`. Without it the CPU arm quantizes ACTIVATIONS to
//! int8 (the A8W8 path, on by default wherever AVX2 or ARM dotprod exists),
//! and the comparison measures that approximation — ~9e-4 relative, uniform
//! across tensors — instead of the device. It cost an evening once.
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

        // A third opinion, from the format's own dequantizer: plain f32 dots
        // over fully expanded weights. When the device and the matvec differ,
        // this says which of the two moved.
        let mut plain = vec![0.0f32; rows];
        {
            let bytes = model.tensor_bytes(&e.name).expect("bytes");
            let mut w = vec![0.0f32; rows * cols];
            cortiq_core::quant::dequant_q4tp(bytes, rows, cols, &mut w);
            for (r, o) in plain.iter_mut().enumerate() {
                *o = w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&xs)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        let dnum: f32 = plain.iter().zip(&cpu).map(|(a, b)| (a - b) * (a - b)).sum();
        let dden: f32 = plain.iter().map(|a| a * a).sum::<f32>().max(1e-20);
        let dq = (dnum / dden).sqrt();

        let mut gpu = vec![0.0f32; rows];
        if !cortiq_engine::gpu_wgpu::q4tp_matvec_for_test(&model, idx, &xs, rows, cols, &mut gpu) {
            eprintln!("устройство отказалось от {} — пропуск", e.name);
            continue;
        }
        let num: f32 = cpu.iter().zip(&gpu).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = cpu.iter().map(|a| a * a).sum::<f32>().max(1e-20);
        let rel = (num / den).sqrt();
        let pnum: f32 = plain.iter().zip(&gpu).map(|(a, b)| (a - b) * (a - b)).sum();
        let pgpu = (pnum / dden).sqrt();
        println!(
            "{}: [{rows}x{cols}] gpu↔matvec {rel:.3e}  деквант↔matvec {dq:.3e}  деквант↔gpu {pgpu:.3e}",
            e.name
        );
        if std::env::var("CMF_Q4TP_SHOW").is_ok() {
            // A constant ratio across rows means a scale read at a different
            // precision; scatter means a layout or ordering fault. The two
            // want opposite fixes, and the norm alone tells them apart badly.
            for r in 0..4.min(rows) {
                println!(
                    "    строка {r}: cpu {:+.6e}  gpu {:+.6e}  отношение {:.6}",
                    cpu[r],
                    gpu[r],
                    gpu[r] / cpu[r]
                );
            }
        }
        // 5e-3, not 1e-3: a [4x256] tensor reduces over four rows and the
        // summation order alone moves it that far. A real layout error is
        // orders of magnitude bigger, so this still catches what matters.
        assert!(
            rel < 5e-3,
            "{}: устройство разошлось с CPU на {rel:.3e}",
            e.name
        );
        checked += 1;
        if checked >= 24 {
            break;
        }
    }
    println!("сверено тензоров: {checked}");
}
