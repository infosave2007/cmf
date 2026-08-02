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
    // Nobody asked for wgpu here — a legitimate skip, and what CI runners
    // look like. Asked-for-and-absent is a different thing and fails below:
    // a reserved word in one shader once took the context down and every GPU
    // test reported success by skipping.
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
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
    // Nobody asked for wgpu here — a legitimate skip, and what CI runners
    // look like. Asked-for-and-absent is a different thing and fails below:
    // a reserved word in one shader once took the context down and every GPU
    // test reported success by skipping.
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    let (nh, hd, n_pos) = (16usize, 64usize, 200usize);
    let q = noise(nh * hd, 0.9);
    let kv = noise(n_pos * hd, 2.2);
    // Mixed signs in the head weights, so relu-after-weighting would show up.
    let hw: Vec<f32> = (0..nh).map(|h| if h % 3 == 0 { -0.4 } else { 0.6 }).collect();
    let limit = 137usize;

    let mut cpu = Vec::new();
    cortiq_engine::dsv4::index_scores(&q, &kv, &hw, nh, hd, n_pos, limit, None, &mut cpu);
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
    // Nobody asked for wgpu here — a legitimate skip, and what CI runners
    // look like. Asked-for-and-absent is a different thing and fails below:
    // a reserved word in one shader once took the context down and every GPU
    // test reported success by skipping.
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
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

/// The compressor's STATE on the device, over a run of tokens: the slot a
/// token lands in, the stride at which the window closes, the shuffle of the
/// closed window into `prev`, and the rope position of the folded entry —
/// which is the window's FIRST token, not the one that closed it.
///
/// The projections are handed in rather than computed, because the q4tp
/// matvec has its own parity test and this one is about the bookkeeping that
/// used to live on the host between every pair of GPU submissions.
#[test]
fn device_compressor_state_matches_the_cpu() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    // Both shapes the release uses: the overlapping ratio-4 compressor and
    // the flat one, whose bias is added inside the pooling kernel instead.
    for (overlap, ratio) in [(true, 4usize), (false, 2usize)] {
        let width = if overlap { 64 } else { 32 };
        let ew = if overlap { width / 2 } else { width };
        let rd = 8;
        let eps = 1e-6;
        let norm = noise(ew, 3.0);
        let ape = noise(ratio * width, 5.0);
        let inv_freq: Vec<f32> = (0..rd / 2).map(|i| 1.0 / (10000f32.powf(i as f32 / 4.0))).collect();
        let kv_id = if overlap { 7001 } else { 7002 };

        // The CPU reference, kept by hand so the state is visible.
        let (mut pend_kv, mut pend_sc) = (Vec::new(), Vec::new());
        let (mut prev_kv, mut prev_sc) = (Vec::new(), Vec::new());

        for pos in 0..3 * ratio {
            let ckv = noise(width, pos as f32 * 1.3 + 11.0);
            let mut csc = noise(width, pos as f32 * 0.9 + 23.0);
            if overlap {
                let slot = pos % ratio;
                for (c, a) in csc.iter_mut().zip(&ape[slot * width..(slot + 1) * width]) {
                    *c += a;
                }
            }
            pend_kv.extend_from_slice(&ckv);
            pend_sc.extend_from_slice(&csc);
            let cpu = if pend_kv.len() / width < ratio {
                None
            } else {
                let mut folded = vec![0.0f32; ew];
                if overlap {
                    cortiq_engine::dsv4::compress_window_overlap(
                        &prev_kv, &prev_sc, &pend_kv, &pend_sc, ratio, ew, &mut folded,
                    );
                    prev_kv = std::mem::take(&mut pend_kv);
                    prev_sc = std::mem::take(&mut pend_sc);
                } else {
                    cortiq_engine::dsv4::compress_window(
                        &pend_kv, &pend_sc, &ape, ratio, width, &mut folded,
                    );
                }
                cortiq_engine::dsv4::rms_weighted(&mut folded, &norm, eps);
                cortiq_engine::dsv4::rope_tail(&mut folded, &inv_freq, pos + 1 - ratio, rd, false);
                pend_kv.clear();
                pend_sc.clear();
                Some(folded)
            };

            // The device, driven the same way. `csc` already carries the
            // overlap bias, exactly as the frame's axpy leaves it.
            let mut got = Vec::new();
            let folded = cortiq_engine::gpu_wgpu::dsv4_comp_state_for_test(
                &ckv, &csc, &norm, &ape, &inv_freq, width, ratio, overlap, rd, eps, pos,
                kv_id, &mut got,
            )
            .expect("кадр компрессора отказал");

            assert_eq!(
                folded,
                cpu.is_some(),
                "перекрытие={overlap} поз={pos}: окно закрылось не там"
            );
            if let Some(want) = cpu {
                let r = rel(&want, &got);
                assert!(r < 1e-5, "перекрытие={overlap} поз={pos}: {r:.3e}");
            }
        }
    }
}

/// The attended-position list, assembled on the device the way the host used
/// to assemble it: the window in cache order, then the indexer's picks
/// shifted by the window's CAPACITY. The two shifts differ whenever the
/// window is not yet full, which is every sequence's first hundred tokens —
/// and reading the wrong keys there would still produce fluent text.
#[test]
fn device_index_list_matches_the_host_mapping() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    let window = 128usize;
    for (win_len, pick) in [
        (128usize, vec![0u32, 3, 9]),  // full window
        (5usize, vec![1u32, 2]),       // early: fill != capacity
        (17usize, vec![]),             // nothing compressed picked
    ] {
        // What the host builds today, from `prep.idxs` mapped through the
        // frame's own `win_len`/`window` rule.
        let want: Vec<u32> = (0..win_len as u32)
            .chain(pick.iter().map(|&p| window as u32 + p))
            .collect();
        let mut got = Vec::new();
        assert!(
            cortiq_engine::gpu_wgpu::dsv4_idx_build_for_test(&pick, win_len, window, &mut got),
            "сборщик списка отказал"
        );
        assert_eq!(want, got, "win_len={win_len} picks={pick:?}");
    }
}
