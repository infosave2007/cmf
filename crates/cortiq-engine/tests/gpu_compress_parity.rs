//! Compressor pooling, indexer scores and top-k, against the CPU.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu \
//!         --test gpu_compress_parity -- --nocapture
//!
//! Each of the three has a way of being subtly wrong that generated text
//! would never reveal, so each is checked at the place it can go wrong: the
//! pooling with an absent previous window (whose slots must vote -inf, not
//! zero), the indexer with a causal limit and negative dots (relu before the
//! head weight, not after), and the top-k with ties and masked entries.

#![cfg(feature = "gpu")]

fn noise(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.7 + seed) * 1.9).sin() * 1.4)
        .collect()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(1e-20);
    (num / den).sqrt()
}

#[test]
fn device_kv_pool_matches_the_cpu() {
    let (ratio, d) = (4usize, 128usize);
    let cur_kv = noise(ratio * 2 * d, 0.3);
    let cur_sc = noise(ratio * 2 * d, 1.1);
    let prev_kv = noise(ratio * 2 * d, 2.7);
    let prev_sc = noise(ratio * 2 * d, 3.5);

    for have_prev in [true, false] {
        let (pk, ps): (&[f32], &[f32]) = if have_prev {
            (&prev_kv, &prev_sc)
        } else {
            (&[], &[])
        };
        let mut cpu = vec![0.0f32; d];
        cortiq_engine::dsv4::compress_window_overlap(pk, ps, &cur_kv, &cur_sc, ratio, d, &mut cpu);
        let mut gpu = vec![0.0f32; d];
        if !cortiq_engine::gpu_wgpu::kv_pool_for_test(
            pk, ps, &cur_kv, &cur_sc, None, ratio, d, true, &mut gpu,
        ) {
            panic!("устройство не поднялось — тест не проверил ничего");
        }
        let r = rel(&cpu, &gpu);
        println!("перекрытие, prev={have_prev}: {r:.3e}");
        assert!(r < 1e-5, "перекрытие prev={have_prev}: {r:.3e}");
    }

    // The plain compressor: a wide window and the APE bias folded in.
    let (r2, w) = (16usize, 96usize);
    let kv = noise(r2 * w, 4.2);
    let sc = noise(r2 * w, 5.9);
    let ape = noise(r2 * w, 6.4);
    let mut cpu = vec![0.0f32; w];
    cortiq_engine::dsv4::compress_window(&kv, &sc, &ape, r2, w, &mut cpu);
    let mut gpu = vec![0.0f32; w];
    assert!(cortiq_engine::gpu_wgpu::kv_pool_for_test(
        &[],
        &[],
        &kv,
        &sc,
        Some(&ape),
        r2,
        w,
        false,
        &mut gpu,
    ));
    let r = rel(&cpu, &gpu);
    println!("плоский компрессор с APE: {r:.3e}");
    assert!(r < 1e-5, "плоский компрессор: {r:.3e}");
}

#[test]
fn device_index_scores_match_the_cpu() {
    let (nh, hd, n_pos) = (16usize, 64usize, 200usize);
    let q = noise(nh * hd, 0.9);
    let kv = noise(n_pos * hd, 2.2);
    // Mixed signs in the head weights, so relu-after-weighting would show up.
    let hw: Vec<f32> = (0..nh).map(|h| if h % 3 == 0 { -0.4 } else { 0.6 }).collect();
    let limit = 137usize;

    let mut cpu = Vec::new();
    cortiq_engine::dsv4::index_scores(&q, &kv, &hw, nh, hd, n_pos, limit, &mut cpu);
    let mut gpu = vec![0.0f32; n_pos];
    if !cortiq_engine::gpu_wgpu::index_scores_for_test(
        &q, &kv, &hw, nh, hd, n_pos, limit, &mut gpu,
    ) {
        panic!("устройство не поднялось — тест не проверил ничего");
    }
    for t in limit..n_pos {
        assert!(gpu[t] < -1e30, "позиция {t} за пределом не замаскирована");
    }
    let r = rel(&cpu[..limit], &gpu[..limit]);
    println!("индексатор ({limit} из {n_pos} позиций): {r:.3e}");
    assert!(r < 1e-5, "индексатор: {r:.3e}");
}

#[test]
fn device_top_k_matches_the_cpu() {
    for (n, k) in [(200usize, 64usize), (700, 512), (40, 512)] {
        let mut sc: Vec<f32> = (0..n).map(|i| ((i * 37 % 61) as f32) * 0.5).collect();
        // Ties are the whole point: the CPU breaks them by the lower index.
        for i in (0..n).step_by(7) {
            sc[i] = 12.0;
        }
        for i in (0..n).step_by(11) {
            sc[i] = f32::NEG_INFINITY;
        }
        let mut cpu = Vec::new();
        cortiq_engine::dsv4::top_k_positions(&sc, k, &mut cpu);
        let mut gpu = Vec::new();
        if !cortiq_engine::gpu_wgpu::top_k_for_test(&sc, k, &mut gpu) {
            panic!("устройство не поднялось — тест не проверил ничего");
        }
        let cpu32: Vec<u32> = cpu.iter().map(|&x| x as u32).collect();
        println!("top-k n={n} k={k}: выбрано {} против {}", gpu.len(), cpu.len());
        assert_eq!(cpu32, gpu, "top-k разошлись при n={n} k={k}");
    }
}
