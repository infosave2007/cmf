use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};
use cortiq_embryo::ops::{blend_top2_residual, causal_k4_top2_argmin, top2_argmin};

#[test]
fn top2_margin_ties_and_disabled_identity_are_deterministic() {
    // Row-major [rows=3, experts=3].  Row 0 has a tie for the best score;
    // lowest index wins and expert 1 is the runner-up.  Row 1 is wide-margin
    // and therefore keeps the top-1-only sentinel.  Row 2 has a close pair.
    let scores = vec![
        1.0, 1.0, 4.0, // tie
        0.0, 5.0, 9.0, // wide
        2.0, 2.2, 7.0, // close
    ];
    let out = top2_argmin(&scores, &[0.0, 0.0, 0.0], 1, 3, 3, 0.25);
    assert_eq!(out.assign, vec![0, 0, 0]);
    assert_eq!(out.runner_up, vec![1, usize::MAX, 1]);
    assert_eq!(out.runner_weight, vec![0.5, 0.0, 0.5]);
    assert_eq!(out.fallback_count, 2);
    assert!((out.fallback_rate() - 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(out.margin[0], 0.0);
    assert_eq!(out.margin[1], 5.0);
    assert!((out.margin[2] - 0.2).abs() < 1e-6);

    // Non-positive thresholds are an exact top-1 identity (including all
    // diagnostics); no runner-up is marked.
    let disabled = top2_argmin(&scores, &[0.0, 0.0, 0.0], 1, 3, 3, 0.0);
    assert_eq!(disabled.assign, out.assign);
    assert_eq!(disabled.runner_up, vec![usize::MAX; 3]);
    assert_eq!(disabled.runner_weight, vec![0.0; 3]);
    assert_eq!(disabled.fallback_count, 0);
}

#[test]
fn top2_causal_window_never_crosses_sequence_boundary() {
    // Two sequences of length two.  The second sequence begins with a close
    // pair despite the previous sequence ending in a very different row;
    // causal-k4 must not leak that preceding sequence's scores.
    let scores = vec![
        0.0, 10.0, // b0,t0
        10.0, 0.0, // b0,t1
        2.0, 2.1, // b1,t0: close pair
        10.0, 0.0, // b1,t1
    ];
    let out = causal_k4_top2_argmin(&scores, &[0.0, 0.0], 2, 2, 2, 0.2);
    assert_eq!(out.assign, vec![0, 0, 0, 1]);
    assert_eq!(out.runner_up, vec![usize::MAX, 1, 1, usize::MAX]);
    assert_eq!(out.fallback_count, 2);
}

#[test]
fn top2_config_is_serialized_default_off_and_parameter_neutral() {
    let base = EmbryoCfg::tiny();
    assert_eq!(base.router_top2_margin, None);
    assert!(!base.router_top2_enabled());
    let mut enabled = base.clone();
    enabled.router_top2_margin = Some(0.25);
    assert!(enabled.router_top2_enabled());
    let base_layout = Layout::new(&base);
    let enabled_layout = Layout::new(&enabled);
    assert_eq!(base.params(), enabled.params());
    assert_eq!(base_layout.total, enabled_layout.total);
    assert_eq!(base_layout.names, enabled_layout.names);
    assert_eq!(
        init_params(&base, &base_layout, 17),
        init_params(&enabled, &enabled_layout, 17)
    );

    let encoded = serde_json::to_string(&enabled).unwrap();
    assert!(encoded.contains("router_top2_margin"));
    let decoded: EmbryoCfg = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.router_top2_margin, Some(0.25));
    let mut legacy_json = serde_json::to_value(&base).unwrap();
    legacy_json
        .as_object_mut()
        .unwrap()
        .remove("router_top2_margin");
    let legacy: EmbryoCfg = serde_json::from_value(legacy_json).unwrap();
    assert_eq!(legacy.router_top2_margin, None);
    assert!(!legacy.router_top2_enabled());
}

#[test]
fn top2_blend_changes_only_routed_residual_rows() {
    let primary = [1.0f32, 3.0, 5.0, 7.0];
    let runner = [9.0f32, 11.0, 13.0, 15.0];
    let blended = blend_top2_residual(&primary, &runner, &[0.0, 0.5], 2);
    assert_eq!(&blended[0..2], &[1.0, 3.0]);
    assert_eq!(&blended[2..4], &[9.0, 11.0]);
}

#[cfg(target_os = "macos")]
#[test]
fn route_top2_kernel_matches_cpu_and_reports_fallbacks() {
    use cortiq_embryo::metal::{Cmd, GBuf, RouteDims, ctx};

    let Some(c) = ctx() else {
        eprintln!("Metal unavailable; skipping top2 route kernel witness");
        return;
    };
    let (batch, seq, h, experts) = (2usize, 3usize, 4usize, 3usize);
    let rows = batch * seq;
    let x_data: Vec<f32> = (0..rows * h)
        .map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.125)
        .collect();
    let mu_data: Vec<f32> = (0..experts * h)
        .map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.11)
        .collect();
    let bias_data = [0.0f32, 0.03, -0.02];
    let raw = |row: usize, e: usize| {
        (0..h)
            .map(|j| {
                let d = x_data[row * h + j] - mu_data[e * h + j];
                d * d
            })
            .sum::<f32>()
    };
    let mut raw_scores = vec![0.0f32; rows * experts];
    for row in 0..rows {
        for e in 0..experts {
            raw_scores[row * experts + e] = raw(row, e);
        }
    }
    let threshold = 0.75f32;
    let want = causal_k4_top2_argmin(&raw_scores, &bias_data, batch, seq, experts, threshold);
    let x = GBuf::from_slice(c, &x_data);
    let mu = GBuf::from_slice(c, &mu_data);
    let u = GBuf::zeros(c, 0);
    let bias = GBuf::from_slice(c, &bias_data);
    let assign = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let runner = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let margin = GBuf::zeros(c, rows);
    let weight = GBuf::zeros(c, rows);
    let res = GBuf::zeros(c, rows);
    let count = GBuf::zeros(c, 1);
    let dims = RouteDims {
        rows,
        h,
        e: experts,
        k: 0,
        cap: 64,
    };
    let cmd = Cmd::new(c);
    cmd.route_top2(
        &dims, seq, &x, &mu, 0, &u, 0, &bias, 0, threshold, &assign, &runner, &margin, &weight,
        &res, &count,
    );
    cmd.commit();
    let got_assign: Vec<usize> = assign.as_u32_slice().iter().map(|&v| v as usize).collect();
    let got_runner: Vec<usize> = runner
        .as_u32_slice()
        .iter()
        .map(|&v| {
            if v == u32::MAX {
                usize::MAX
            } else {
                v as usize
            }
        })
        .collect();
    assert_eq!(got_assign, want.assign);
    assert_eq!(got_runner, want.runner_up);
    assert_eq!(weight.as_slice(), want.runner_weight.as_slice());
    assert_eq!(count.as_u32_slice()[0] as usize, want.fallback_count);
    for row in 0..rows {
        assert!((margin.as_slice()[row] - want.margin[row]).abs() < 1e-4);
        let e = want.assign[row];
        assert!((res.as_slice()[row] - raw_scores[row * experts + e]).abs() < 1e-4);
    }

    // seq=1 is an identity for the score window, matching raw top-2.
    let want_raw = top2_argmin(&raw_scores, &bias_data, batch, seq, experts, threshold);
    let assign_raw = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let runner_raw = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let margin_raw = GBuf::zeros(c, rows);
    let weight_raw = GBuf::zeros(c, rows);
    let res_raw = GBuf::zeros(c, rows);
    let count_raw = GBuf::zeros(c, 1);
    let cmd = Cmd::new(c);
    cmd.route_top2(
        &dims,
        1,
        &x,
        &mu,
        0,
        &u,
        0,
        &bias,
        0,
        threshold,
        &assign_raw,
        &runner_raw,
        &margin_raw,
        &weight_raw,
        &res_raw,
        &count_raw,
    );
    cmd.commit();
    let got_assign_raw: Vec<usize> = assign_raw
        .as_u32_slice()
        .iter()
        .map(|&v| v as usize)
        .collect();
    let got_runner_raw: Vec<usize> = runner_raw
        .as_u32_slice()
        .iter()
        .map(|&v| {
            if v == u32::MAX {
                usize::MAX
            } else {
                v as usize
            }
        })
        .collect();
    assert_eq!(got_assign_raw, want_raw.assign);
    assert_eq!(got_runner_raw, want_raw.runner_up);
}

#[cfg(target_os = "macos")]
#[test]
fn top2_production_like_route_and_capacity_are_bounded() {
    use cortiq_embryo::metal::{Cmd, GBuf, RouteDims, ctx};

    let Some(c) = ctx() else { return };
    let (batch, seq, h, experts) = (8usize, 1024usize, 2048usize, 4usize);
    let rows = batch * seq;
    let x = GBuf::zeros(c, rows * h);
    let mu = GBuf::zeros(c, experts * h);
    let bias = GBuf::zeros(c, experts);
    let u = GBuf::zeros(c, 0);
    let assign = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let runner = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let slot = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let slot2 = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let margin = GBuf::zeros(c, rows);
    let weight = GBuf::zeros(c, rows);
    let res = GBuf::zeros(c, rows);
    let fallback = GBuf::zeros(c, 1);
    let count = GBuf::from_u32(c, &vec![0; experts]);
    let count2 = GBuf::from_u32(c, &vec![0; experts]);
    let dims = RouteDims {
        rows,
        h,
        e: experts,
        k: 0,
        cap: (2 * rows / experts).div_ceil(64) * 64,
    };
    let cmd = Cmd::new(c);
    cmd.route_top2(
        &dims, seq, &x, &mu, 0, &u, 0, &bias, 0, 0.1, &assign, &runner, &margin, &weight, &res,
        &fallback,
    );
    cmd.route_group(&dims, &assign, &slot, &count, 0);
    cmd.route_group(&dims, &runner, &slot2, &count2, 0);
    cmd.commit();
    assert_eq!(fallback.as_u32_slice()[0] as usize, rows);
    assert_eq!(
        count
            .as_u32_slice()
            .iter()
            .map(|&n| n as usize)
            .sum::<usize>(),
        rows
    );
    assert_eq!(
        count2
            .as_u32_slice()
            .iter()
            .map(|&n| n as usize)
            .sum::<usize>(),
        rows
    );
    // The deterministic serial grouping keeps all assignments in index order;
    // rows beyond `cap` are explicitly dropped by gather/scatter rather than
    // silently reassigning them.
    assert!(count.as_u32_slice().iter().any(|&n| n > dims.cap as u32));
    assert!(count2.as_u32_slice().iter().any(|&n| n > dims.cap as u32));
    assert!(weight.as_slice().iter().all(|&w| w == 0.5));
    assert!(
        runner
            .as_u32_slice()
            .iter()
            .all(|&e| (e as usize) < experts)
    );
}
