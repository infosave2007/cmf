//! The token-axis router against the single-token device router, several
//! tokens at once — scored with bias (the q4 stand's shape), and forced.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu \
//!         --test gpu_bt_route_parity -- --nocapture

#![cfg(feature = "gpu")]

#[test]
fn batched_routing_matches_the_single_router() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            eprintln!("wgpu не запрошен (CMF_GPU=wgpu) — пропуск");
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    let n = 16usize;
    let b = 3usize;
    let top_k = 2usize;
    let scale = 2.5f32;
    let scores: Vec<f32> = (0..b * n)
        .map(|i| ((i as f32 * 0.37).sin() * 3.0) + ((i % 7) as f32 * 0.13))
        .collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.7).cos()) * 0.8).collect();

    for use_bias in [false, true] {
        let mut bi = Vec::new();
        let mut bw = Vec::new();
        assert!(
            cortiq_engine::gpu_wgpu::bt_moe_route_for_test(
                &scores,
                b,
                use_bias.then_some(&bias[..]),
                None,
                top_k,
                scale,
                &mut bi,
                &mut bw,
            ),
            "пакетный роутер не запустился"
        );
        for t in 0..b {
            let mut si = Vec::new();
            let mut sw = Vec::new();
            assert!(cortiq_engine::gpu_wgpu::moe_route_for_test(
                &scores[t * n..(t + 1) * n],
                use_bias.then_some(&bias[..]),
                None,
                None,
                top_k,
                scale,
                true,
                &mut si,
                &mut sw,
            ));
            let slots = top_k + 1;
            assert_eq!(
                &bi[t * slots..(t + 1) * slots],
                &si[..],
                "токен {t} (bias={use_bias}): другие эксперты"
            );
            for (j, (a, e)) in bw[t * slots..(t + 1) * slots]
                .iter()
                .zip(&sw)
                .enumerate()
            {
                assert!(
                    (a - e).abs() <= 1e-6 * e.abs().max(1.0),
                    "токен {t} слот {j} (bias={use_bias}): вес {a} против {e}"
                );
            }
        }
    }

    // Forced rows differ per token — each must land in its own slot row.
    let rows: Vec<Vec<usize>> = vec![vec![3, 7], vec![1, 15], vec![9, 0]];
    let mut bi = Vec::new();
    let mut bw = Vec::new();
    assert!(cortiq_engine::gpu_wgpu::bt_moe_route_for_test(
        &scores,
        b,
        None,
        Some(&rows),
        top_k,
        scale,
        &mut bi,
        &mut bw,
    ));
    for t in 0..b {
        let mut si = Vec::new();
        let mut sw = Vec::new();
        assert!(cortiq_engine::gpu_wgpu::moe_route_for_test(
            &scores[t * n..(t + 1) * n],
            None,
            None,
            Some(&rows[t]),
            top_k,
            scale,
            true,
            &mut si,
            &mut sw,
        ));
        let slots = top_k + 1;
        assert_eq!(&bi[t * slots..(t + 1) * slots], &si[..], "forced токен {t}");
    }
    println!("пакетный роутер совпал с одиночным на {b} токенах");
}
