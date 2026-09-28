use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};
use cortiq_embryo::ops::{causal_k4_argmin, causal_k4_score_smooth};

#[test]
fn causal_k4_boundaries_ties_and_determinism() {
    // Two independent sequences: row 3 must not see row 2 from the prior
    // sequence, and a one-position sequence is an exact identity.
    let scores = vec![
        1.0, 9.0, // b0,t0
        3.0, 7.0, // b0,t1
        5.0, 5.0, // b0,t2
        9.0, 1.0, // b1,t0 (sequence boundary)
        7.0, 3.0, // b1,t1
        5.0, 5.0, // b1,t2
    ];
    let smoothed = causal_k4_score_smooth(&scores, 2, 3, 2);
    assert_eq!(&smoothed[0..2], &[1.0, 9.0]);
    assert_eq!(&smoothed[2..4], &[2.0, 8.0]);
    assert_eq!(&smoothed[4..6], &[3.0, 7.0]);
    assert_eq!(&smoothed[6..8], &[9.0, 1.0]);
    assert_eq!(&smoothed[8..10], &[8.0, 2.0]);
    assert_eq!(&smoothed[10..12], &[7.0, 3.0]);

    let one = vec![2.5, -4.0, 8.0];
    assert_eq!(causal_k4_score_smooth(&one, 3, 1, 1), one);

    // Equal adjusted scores retain the lowest expert index, exactly as the
    // strict `<` comparison in the Metal kernels does.
    let ties = vec![1.0; 12];
    assert_eq!(
        causal_k4_argmin(&ties, &[0.0, 0.0], 2, 3, 2),
        vec![0, 0, 0, 0, 0, 0]
    );

    // Repeated evaluation is deterministic (including at sequence edges).
    assert_eq!(
        causal_k4_score_smooth(&scores, 2, 3, 2),
        causal_k4_score_smooth(&scores, 2, 3, 2)
    );
}

#[test]
fn router_smoothing_is_default_off_parameter_neutral_and_checkpointed() {
    let base = EmbryoCfg::tiny();
    assert!(!base.router_smooth_k4);
    let mut smooth = base.clone();
    smooth.router_smooth_k4 = true;
    let base_layout = Layout::new(&base);
    let smooth_layout = Layout::new(&smooth);
    assert_eq!(base.params(), smooth.params());
    assert_eq!(base_layout.total, smooth_layout.total);
    assert_eq!(base_layout.names, smooth_layout.names);
    assert_eq!(
        init_params(&base, &base_layout, 7),
        init_params(&smooth, &smooth_layout, 7)
    );

    let encoded = serde_json::to_string(&smooth).unwrap();
    assert!(encoded.contains("router_smooth_k4"));
    let decoded: EmbryoCfg = serde_json::from_str(&encoded).unwrap();
    assert!(decoded.router_smooth_k4);

    // Configs written before this discriminator existed still decode to the
    // exact disabled legacy behavior.
    let mut legacy_json = serde_json::to_value(&base).unwrap();
    legacy_json
        .as_object_mut()
        .unwrap()
        .remove("router_smooth_k4");
    let legacy: EmbryoCfg = serde_json::from_value(legacy_json).unwrap();
    assert!(!legacy.router_smooth_k4);

    let path = std::env::temp_dir().join(format!("cmf-router-smooth-{}.ckpt", std::process::id()));
    let params = init_params(&smooth, &smooth_layout, 11);
    cortiq_embryo::train::save_checkpoint(
        &path,
        &smooth,
        19,
        &params,
        Some(&params),
        Some(&params),
        &[],
    )
    .unwrap();
    let loaded = cortiq_embryo::train::load_checkpoint(&path).unwrap();
    assert!(loaded.cfg.router_smooth_k4);
    assert_eq!(loaded.params, params);
    assert_eq!(loaded.m.as_deref(), Some(params.as_slice()));
    assert_eq!(loaded.v.as_deref(), Some(params.as_slice()));
    std::fs::remove_file(path).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn route_smooth_k4_kernel_matches_cpu_and_preserves_raw_resonance() {
    use cortiq_embryo::metal::{Cmd, GBuf, RouteDims, ctx};

    let Some(c) = ctx() else {
        eprintln!("Metal unavailable; skipping route smoothing kernel witness");
        return;
    };
    let (batch, seq, h, experts) = (2usize, 5usize, 4usize, 3usize);
    let rows = batch * seq;
    let x_data: Vec<f32> = (0..rows * h)
        .map(|i| ((i * 17 % 23) as f32 - 11.0) * 0.125)
        .collect();
    let mu_data: Vec<f32> = (0..experts * h)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2)
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
    let want_assign = causal_k4_argmin(&raw_scores, &bias_data, batch, seq, experts);
    let x = GBuf::from_slice(c, &x_data);
    let mu = GBuf::from_slice(c, &mu_data);
    let u = GBuf::zeros(c, 0);
    let bias = GBuf::from_slice(c, &bias_data);
    let assign = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let res = GBuf::zeros(c, rows);
    let dims = RouteDims {
        rows,
        h,
        e: experts,
        k: 0,
        cap: 64,
    };
    let cmd = Cmd::new(c);
    cmd.route_smooth_k4(&dims, seq, &x, &mu, 0, &u, 0, &bias, 0, &assign, &res);
    cmd.commit();
    let got_assign: Vec<usize> = assign.as_u32_slice().iter().map(|&v| v as usize).collect();
    assert_eq!(got_assign, want_assign);
    for row in 0..rows {
        let e = want_assign[row];
        assert!((res.as_slice()[row] - raw_scores[row * experts + e]).abs() < 1e-5);
    }

    // A second identical dispatch is byte-identical, including at the
    // sequence boundary and at any equal-score tie.
    let assign_repeat = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let res_repeat = GBuf::zeros(c, rows);
    let cmd = Cmd::new(c);
    cmd.route_smooth_k4(
        &dims,
        seq,
        &x,
        &mu,
        0,
        &u,
        0,
        &bias,
        0,
        &assign_repeat,
        &res_repeat,
    );
    cmd.commit();
    assert_eq!(assign.as_u32_slice(), assign_repeat.as_u32_slice());
    assert_eq!(res.as_slice(), res_repeat.as_slice());

    // seq=1 must be an identity transform: smoothed routing and legacy
    // routing produce identical assignment and raw resonance diagnostics.
    let assign_legacy = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let res_legacy = GBuf::zeros(c, rows);
    let assign_identity = GBuf::from_u32(c, &vec![u32::MAX; rows]);
    let res_identity = GBuf::zeros(c, rows);
    let cmd = Cmd::new(c);
    cmd.route(
        &dims,
        &x,
        &mu,
        0,
        &u,
        0,
        &bias,
        0,
        &assign_legacy,
        &res_legacy,
    );
    cmd.route_smooth_k4(
        &dims,
        1,
        &x,
        &mu,
        0,
        &u,
        0,
        &bias,
        0,
        &assign_identity,
        &res_identity,
    );
    cmd.commit();
    assert_eq!(assign_legacy.as_u32_slice(), assign_identity.as_u32_slice());
    assert_eq!(res_legacy.as_slice(), res_identity.as_slice());
}
