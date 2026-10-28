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
fn device_routing_matches_the_cpu_needs_cmf_gpu() {
    // Nobody asked for wgpu here — a legitimate skip, and what CI runners
    // look like. Asked-for-and-absent is a different thing and fails below:
    // a reserved word in one shader once took the context down and every GPU
    // test reported success by skipping.
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu was requested but the context did not come up"),
        Some(true) => {}
    }
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
        ("bare scores", None, None, None),
        ("with a bias", Some(&bias), None, None),
        ("bias and mask", Some(&bias), Some(&mask), None),
        ("hash layer", None, None, Some(&forced)),
    ];

    for (name, b, m, f) in cases {
        let mut ci = Vec::new();
        let mut cw = Vec::new();
        cortiq_engine::dsv4::route(&scores, b, top_k, scale, f, m, &mut ci, &mut cw);

        let mut gi = Vec::new();
        let mut gw = Vec::new();
        if !cortiq_engine::gpu_wgpu::moe_route_for_test(
            &scores, b, m, f, top_k, scale, false, &mut gi, &mut gw,
        ) {
            panic!("the device did not come up — the test checked nothing");
        }
        assert_eq!(ci, gi, "{name}: different experts or order");
        let d = cw
            .iter()
            .zip(&gw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{name}: {ci:?}, max|Δw| = {d:.3e}");
        assert!(d < 1e-6, "{name}: the weights diverged by {d:.3e}");
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
        false,
        &mut gi,
        &mut gw,
    ));
    assert_eq!(ci, gi, "fully closed mask");
    println!("fully closed mask: selected {}", gi.len());

    // The msel/mwt shape the batched expert kernels take: every slot written,
    // the shared expert pinned last at weight 1, and a slot the router could
    // not fill carrying weight ZERO rather than a stale index.
    let mut gi = Vec::new();
    let mut gw = Vec::new();
    assert!(cortiq_engine::gpu_wgpu::moe_route_for_test(
        &scores,
        Some(&bias),
        Some(&none),
        None,
        top_k,
        scale,
        true,
        &mut gi,
        &mut gw,
    ));
    assert_eq!(gi.len(), top_k + 1, "there must be top_k+1 slots");
    assert_eq!(gi[top_k], n, "the shared expert comes last");
    assert_eq!(gw[top_k], 1.0, "the shared expert weighs one");
    assert!(
        gw[..top_k].iter().all(|&x| x == 0.0),
        "unfilled slots must weigh zero: {gw:?}"
    );
    // And with the router free to choose, the routed slots come back live.
    let mut gi = Vec::new();
    let mut gw = Vec::new();
    assert!(cortiq_engine::gpu_wgpu::moe_route_for_test(
        &scores,
        Some(&bias),
        None,
        None,
        top_k,
        scale,
        true,
        &mut gi,
        &mut gw,
    ));
    let mut ci = Vec::new();
    let mut cw = Vec::new();
    cortiq_engine::dsv4::route(
        &scores,
        Some(&bias),
        top_k,
        scale,
        None,
        None,
        &mut ci,
        &mut cw,
    );
    assert_eq!(&gi[..top_k], &ci[..], "the routing slots diverged");
    assert_eq!(gi[top_k], n);
    println!("msel/mwt format: {gi:?}");
}

#[test]
fn qwen_selected_softmax_route_matches_cpu_needs_cmf_gpu() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu was requested but the context did not come up"),
        Some(true) => {}
    }

    // Deliberately non-monotonic, with a close pair below the cut. This
    // catches both ranking transformed scores and normalising all experts
    // instead of only the selected Qwen top-k.
    let logits = [2.0f32, -1.0, 0.5, 3.0, 0.2, 1.999];
    let top_k = 2;
    let mut gi = Vec::new();
    let mut gw = Vec::new();
    assert!(cortiq_engine::gpu_wgpu::qwen_moe_route_for_test(
        &logits, top_k, &mut gi, &mut gw,
    ));

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(top_k);
    let max = order
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = order.iter().map(|&i| (logits[i] - max).exp()).sum();
    let expected: Vec<f32> = order.iter().map(|&i| (logits[i] - max).exp() / z).collect();

    assert_eq!(gi, order, "Qwen picked different experts");
    let d = expected
        .iter()
        .zip(&gw)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Qwen selected-softmax: {gi:?}, {gw:?}, max|Δw|={d:.3e}");
    assert!(d < 1e-6, "the Qwen weights diverged by {d:.3e}");
}
