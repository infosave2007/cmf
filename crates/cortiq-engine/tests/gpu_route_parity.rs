//! MoE routing on the device against `dsv4::route`.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu \
//!         --test gpu_route_parity -- --nocapture
//!
//! The check that matters is not "same experts" but "same experts in the same
//! order with the same weights". The bias shifts the choice and not the
//! weight, so a kernel that normalises the biased score instead picks an
//! identical set and weighs it wrong.

#![cfg(feature = "gpu")]

#[test]
fn device_routing_matches_the_cpu() {
    let n = 256usize;
    let top_k = 8usize;
    let scale = 2.5f32;
    let scores: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.37).sin() * 3.0) + ((i % 17) as f32 * 0.1))
        .collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.7).cos()) * 0.8).collect();
    let mut mask = vec![true; n];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = i % 5 != 0; // a task mask keeping four experts in five
    }
    let forced: Vec<usize> = vec![3, 200, 17, 17, 99, 1, 255, 40];

    let cases: Vec<(&str, Option<&[f32]>, Option<&[bool]>, Option<&[usize]>)> = vec![
        ("голые оценки", None, None, None),
        ("со сдвигом", Some(&bias), None, None),
        ("сдвиг и маска", Some(&bias), Some(&mask), None),
        ("хеш-слой", None, None, Some(&forced)),
    ];

    for (name, b, m, f) in cases {
        let mut ci = Vec::new();
        let mut cw = Vec::new();
        cortiq_engine::dsv4::route(&scores, b, top_k, scale, f, m, &mut ci, &mut cw);

        let mut gi = Vec::new();
        let mut gw = Vec::new();
        if !cortiq_engine::gpu_wgpu::moe_route_for_test(
            &scores, b, m, f, top_k, scale, &mut gi, &mut gw,
        ) {
            eprintln!("нет устройства — пропуск");
            return;
        }
        assert_eq!(ci, gi, "{name}: разные эксперты или порядок");
        let d = cw
            .iter()
            .zip(&gw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{name}: {ci:?}, max|Δw| = {d:.3e}");
        assert!(d < 1e-6, "{name}: веса разошлись на {d:.3e}");
    }

    // Every expert masked out: the CPU returns nothing rather than picking
    // the first one, and so must the device.
    let none = vec![false; n];
    let mut ci = Vec::new();
    let mut cw = Vec::new();
    cortiq_engine::dsv4::route(
        &scores,
        Some(&bias),
        top_k,
        scale,
        None,
        Some(&none),
        &mut ci,
        &mut cw,
    );
    let mut gi = Vec::new();
    let mut gw = Vec::new();
    assert!(cortiq_engine::gpu_wgpu::moe_route_for_test(
        &scores,
        Some(&bias),
        Some(&none),
        None,
        top_k,
        scale,
        &mut gi,
        &mut gw,
    ));
    assert_eq!(ci, gi, "полностью закрытая маска");
    println!("полностью закрытая маска: выбрано {}", gi.len());
}
