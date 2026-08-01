//! Per-head RMS and the rope tail, device against CPU, in all three uses.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test gpu_rope_parity -- --nocapture

#[cfg(feature = "gpu")]
#[test]
fn device_rope_heads_matches_the_cpu() {
    use cortiq_engine::dsv4;
    let (nh, hd, rd, eps) = (3usize, 32usize, 8usize, 1e-6f32);
    let freq: Vec<f32> = (0..rd / 2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / rd as f32))
        .collect();

    // Position 0 is where BOTH pairing conventions agree, so it proves
    // nothing on its own — every case here runs at a position where they
    // differ.
    for &(rms, inverse) in &[(true, false), (false, false), (false, true)] {
        for &pos in &[1usize, 7, 129] {
            let src: Vec<f32> = (0..nh * hd)
                .map(|i| ((i * 13 + 1) as f32 * 0.021).sin() * 2.0)
                .collect();
            let mut want = src.clone();
            for h in 0..nh {
                let head = &mut want[h * hd..(h + 1) * hd];
                if rms {
                    dsv4::rms_inplace(head, eps);
                }
                dsv4::rope_tail(head, &freq, pos, rd, inverse);
            }
            let mut got = src.clone();
            if !cortiq_engine::gpu_wgpu::rope_heads_for_test(
                &mut got, &freq, nh, hd, rd, pos, eps, rms, inverse,
            ) {
                panic!("устройство не поднялось — тест не проверил ничего");
            }
            let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|a| a * a).sum::<f32>().max(1e-20);
            let rel = (num / den).sqrt();
            println!("rms={rms} inverse={inverse} поз={pos}: {rel:.3e}");
            assert!(rel < 1e-5, "rms={rms} inverse={inverse} поз={pos}: {rel:.3e}");
        }
    }
}
