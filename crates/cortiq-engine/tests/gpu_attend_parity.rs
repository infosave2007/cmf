//! Sparse attention with a learned sink, device against CPU.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test gpu_attend_parity -- --nocapture

#[cfg(feature = "gpu")]
#[test]
fn device_sparse_attend_matches_the_cpu() {
    let (nh, hd, npos) = (4usize, 64usize, 37usize);
    let q: Vec<f32> = (0..nh * hd).map(|i| ((i * 11) as f32 * 0.019).sin()).collect();
    let kv: Vec<f32> = (0..npos * hd).map(|i| ((i * 5) as f32 * 0.013).cos()).collect();
    // A sink large enough to dominate on one head and negligible on another:
    // both regimes have to come out right, and the failure modes differ.
    let sink: Vec<f32> = vec![-8.0, 0.1, 6.0, 0.0];
    let idxs: Vec<u32> = (0..npos as u32).filter(|i| i % 3 != 1).collect();
    let scale = (hd as f32).powf(-0.5);

    let mut want = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let idx_us: Vec<usize> = idxs.iter().map(|&i| i as usize).collect();
        let mut oh = vec![0.0f32; hd];
        cortiq_engine::dsv4::sparse_attend(
            &q[h * hd..(h + 1) * hd], &kv, &idx_us, sink[h], scale, hd, &mut oh,
        );
        want[h * hd..(h + 1) * hd].copy_from_slice(&oh);
    }

    let mut got = vec![0.0f32; nh * hd];
    if !cortiq_engine::gpu_wgpu::sparse_attend_for_test(
        &q, &kv, &idxs, &sink, scale, nh, hd, &mut got,
    ) {
        eprintln!("устройство недоступно — пропуск");
        return;
    }
    let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = want.iter().map(|a| a * a).sum::<f32>().max(1e-20);
    let rel = (num / den).sqrt();
    println!("разрежённое внимание: {rel:.3e}");
    assert!(rel < 1e-5, "устройство разошлось с CPU: {rel:.3e}");
}
