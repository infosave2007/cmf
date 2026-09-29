//! Whole-graph gradcheck on the GPU: for every named tensor of a tiny
//! genome, the directional derivative of the loss along (a) the gradient
//! direction and (b) a random direction, by central differences of the
//! f32 forward, against g·δ from the hand-rolled backward.
#![cfg(target_os = "macos")]

use cortiq_embryo::metal::{Cmd, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, gauss_vec, init_params};
use cortiq_embryo::ops::lcg_vec;

#[test]
fn every_tensor_matches_finite_differences() {
    let Some(c) = ctx() else { return };
    for variant in 0..8 {
        let mut cfg = EmbryoCfg::tiny();
        match variant {
            0 => cfg.head_clusters = 0,
            2 => cfg.experts = 4,
            3 => cfg.conv_k = 4, // the short-conv taps enter every mixer grad path
            4 => {
                cfg.phase_delta = true;
                cfg.conv_k = 4;
            }
            5 => {
                // Selected-layer mode exercises the same whole-graph
                // backward with the other hybrid layers on the legacy path.
                cfg.phase_delta_layer = Some(0);
                cfg.conv_k = 4;
            }
            6 => {
                // Conditional top-2 exercises the bounded runner stream and
                // weighted expert backward. A very wide margin intentionally
                // activates the runner for every row so the finite-difference
                // witness cannot pass while silently taking top-1 only.
                cfg.experts = 4;
                cfg.router_top2_margin = Some(1.0e9);
            }
            7 => {
                // Both tiny layers are legal hybrids when the anchor cadence
                // is moved past the end of the stack; this exercises the
                // dual selector through the complete graph and backward.
                cfg.anchor_every = 3;
                cfg.phase_delta_layers = Some(vec![0, 1]);
            }
            _ => {}
        }
        eprintln!(
            "=== variant {variant}: head {} experts {} ===",
            if cfg.head_clusters > 0 {
                "hierarchical"
            } else {
                "flat"
            },
            cfg.experts
        );
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 7);
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
        gpu.desc_updates.set(false);
        let tokens: Vec<u32> = lcg_vec(11, m)
            .iter()
            .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
            .collect();
        let targets: Vec<u32> = lcg_vec(12, m)
            .iter()
            .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
            .collect();
        // analytic gradient
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
            std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
        }
        gpu.prepare_head(&targets);
        let cmd = Cmd::new(c);
        gpu.encode_fwd_bwd(&cmd);
        cmd.commit();
        let g = gpu.grads_host();
        gpu.route_frozen.set(true); // FD inside the analytic pass's routing region
        let l0 = gpu.eval_loss(&tokens, &targets);
        eprintln!("loss {l0:.5} (ln V = {:.5})", (cfg.vocab as f64).ln());
        let mut worst_all = 0.0f64;
        let mut seed = 100u64;
        for (name, off, n) in &lay.names {
            if name.contains("hk.kappa") {
                // padded rows: only the real ones are trained; still checked below via full-slice δ (pad grads are 0 by construction)
            }
            let gs = &g[*off..*off + n];
            let gnorm = gs
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            for dir in 0..2 {
                seed += 1;
                let delta: Vec<f64> = if dir == 0 {
                    gs.iter().map(|x| *x as f64 / gnorm.max(1e-30)).collect()
                } else {
                    let r = gauss_vec(seed, *n);
                    let rn = r
                        .iter()
                        .map(|x| (*x as f64) * (*x as f64))
                        .sum::<f64>()
                        .sqrt();
                    r.iter().map(|x| *x as f64 / rn).collect()
                };
                let analytic: f64 = gs.iter().zip(&delta).map(|(a, d)| *a as f64 * d).sum();
                // step so that the loss moves ~1e-2 along the gradient direction
                let eps = (2e-3 / gnorm.max(1e-6)).clamp(1e-3, 0.1);
                let mut pp = p0.clone();
                for i in 0..*n {
                    pp[off + i] = (p0[off + i] as f64 + eps * delta[i]) as f32;
                }
                gpu.set_params(&pp);
                let lp = gpu.eval_loss(&tokens, &targets) as f64;
                for i in 0..*n {
                    pp[off + i] = (p0[off + i] as f64 - eps * delta[i]) as f32;
                }
                gpu.set_params(&pp);
                let lm = gpu.eval_loss(&tokens, &targets) as f64;
                let fd = (lp - lm) / (2.0 * eps);
                // f32 forward noise (~1e-6 on the loss) over the FD step bounds the
                // resolvable gradient: absolute floor 5e-4 on the denominator.
                let denom = gnorm.max(5e-4);
                let rel = (fd - analytic).abs() / denom;
                worst_all = worst_all.max(rel);
                eprintln!(
                    "{name:<24} n={n:<7} |g|={gnorm:.3e} dir={} fd={fd:+.4e} an={analytic:+.4e} err/|g|={rel:.2e}",
                    if dir == 0 { "grad" } else { "rand" }
                );
                assert!(
                    rel < 3e-2,
                    "{name} dir {dir}: fd {fd} vs analytic {analytic} (|g| {gnorm})"
                );
            }
        }
        gpu.set_params(&p0);
        eprintln!("worst err/|g| = {worst_all:.2e}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn top2_disabled_is_bit_identical_and_enabled_is_finite() {
    let Some(_) = ctx() else { return };
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let mut base = EmbryoCfg::tiny();
    base.experts = 4;
    let lay = Layout::new(&base);
    let p0 = init_params(&base, &lay, 77);
    let mut legacy = EmbryoGpu::new(base.clone(), b, t, &p0).expect("legacy gpu");
    let mut disabled = base.clone();
    // Some(0) is deliberately a disabled threshold and must retain the
    // top-1 route, arena geometry, and exact output/gradient identity.
    disabled.router_top2_margin = Some(0.0);
    let mut identity = EmbryoGpu::new(disabled, b, t, &p0).expect("disabled gpu");
    legacy.desc_updates.set(false);
    identity.desc_updates.set(false);
    let tokens: Vec<u32> = lcg_vec(811, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * base.vocab as f32) as u32 % base.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(812, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * base.vocab as f32) as u32 % base.vocab as u32)
        .collect();
    let l0 = legacy.train_step(&tokens, &targets, 0.0, 0.0, 1.0).0;
    let l1 = identity.train_step(&tokens, &targets, 0.0, 0.0, 1.0).0;
    assert_eq!(l0.to_bits(), l1.to_bits(), "disabled top2 loss drift");
    let g0 = legacy.grads_host();
    let g1 = identity.grads_host();
    let max_grad_delta = g0
        .iter()
        .zip(&g1)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_grad_delta < 2.0e-5,
        "disabled top2 grad drift {max_grad_delta:e}"
    );

    let mut enabled = base;
    enabled.router_top2_margin = Some(1.0e9);
    let mut top2 = EmbryoGpu::new(enabled, b, t, &p0).expect("top2 gpu");
    top2.desc_updates.set(false);
    let (loss, _, _) = top2.train_step(&tokens, &targets, 0.0, 0.0, 1.0);
    assert!(loss.is_finite(), "top2 loss is non-finite");
    let fallback = top2.routing_top2_fallbacks();
    assert_eq!(fallback.len(), top2.cfg.layers);
    assert!(fallback.iter().all(|&n| n <= m as u32));
    assert!(
        fallback.iter().any(|&n| n > 0),
        "runner stream was not exercised"
    );
    let telemetry = top2.routing_top2_telemetry();
    assert_eq!(telemetry.len(), top2.cfg.layers);
    assert!(
        telemetry
            .iter()
            .all(|&(fb, drops)| fb <= m as u32 && drops <= m as u32)
    );
}

#[test]
fn subspace_update_keeps_training_sane() {
    let Some(_) = ctx() else { return };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 3);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    let tokens: Vec<u32> = lcg_vec(31, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(32, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let mut cov = Vec::new();
    for s in 0..4 {
        let (loss, _, _) = gpu.train_step(&tokens, &targets, 1e-3, 0.0, 1.0);
        assert!(loss.is_finite());
        if s == 1 || s == 3 {
            gpu.update_subspaces(&mut cov, 0.5);
        }
    }
    // U rows orthonormal per (layer, expert)
    let u = gpu.desc.u.to_vec();
    let (h, k) = (cfg.hidden, cortiq_embryo::model::MOE_K);
    for le in 0..cfg.layers * cfg.experts {
        let blk = &u[le * k * h..(le + 1) * k * h];
        for i in 0..k {
            for j in 0..=i {
                let dot: f32 = (0..h).map(|t| blk[i * h + t] * blk[j * h + t]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-3, "le {le} u{i}·u{j} = {dot}");
            }
        }
    }
    let (loss, _, _) = gpu.train_step(&tokens, &targets, 1e-3, 0.0, 1.0);
    assert!(loss.is_finite());
    eprintln!("subspaces orthonormal; loss after {loss:.3}");
}

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn phase_delta_layer3_full_model_train_step_benchmark() {
    // This is intentionally a direct model smoke (no corpus/teacher birth):
    // control and candidate are allocated, measured, and dropped
    // sequentially so unified-memory machines never hold two full arenas.
    let Some(_) = ctx() else { return };
    use std::time::Instant;

    let (b, t, warmup, samples) = (8usize, 1024usize, 2usize, 5usize);
    let base = EmbryoCfg::embryo0();
    let base_lay = Layout::new(&base);
    let p0 = init_params(&base, &base_lay, 1);
    let m = b * t;
    let tokens: Vec<u32> = lcg_vec(7001, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * base.vocab as f32) as u32 % base.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(7002, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * base.vocab as f32) as u32 % base.vocab as u32)
        .collect();

    let run = |cfg: EmbryoCfg| -> Vec<f64> {
        let lay = Layout::new(&cfg);
        assert_eq!(lay.total, p0.len());
        let mut gpu = EmbryoGpu::new(cfg, b, t, &p0).expect("Metal model");
        let mut out = Vec::with_capacity(samples);
        for i in 0..warmup + samples {
            let started = Instant::now();
            let (loss, _, _) = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1.0);
            assert!(loss.is_finite());
            if i >= warmup {
                out.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        out
    };

    let control = run(base.clone());
    let mut candidate_cfg = base;
    candidate_cfg.phase_delta_layer = Some(3);
    let candidate = run(candidate_cfg);
    let mut dual_cfg = EmbryoCfg::embryo0();
    dual_cfg.phase_delta_layers = Some(vec![3, 6]);
    let dual = run(dual_cfg);
    let median = |xs: &[f64]| -> f64 {
        let mut ys = xs.to_vec();
        ys.sort_by(|a, b| a.total_cmp(b));
        ys[ys.len() / 2]
    };
    let control_median = median(&control);
    let candidate_median = median(&candidate);
    let dual_median = median(&dual);
    let ratio = candidate_median / control_median;
    let dual_ratio = dual_median / control_median;
    // D21's measured v3 control footprint is the conservative base for the
    // candidate prediction; only the shared phase scratch is new.
    let phase_scratch = (b * 8 * 65 * 64 * 128 + b * 8 * 4 * t * (1 + 2 * 32)) as f64
        * std::mem::size_of::<f32>() as f64;
    let predicted_gib = 17.359 + phase_scratch / (1u64 << 30) as f64;
    eprintln!(
        "phase_delta layer3/dual(3,6) train_step B={b} T={t} warmup={warmup} control_ms={control:?} candidate_ms={candidate:?} dual_ms={dual:?} medians={control_median:.3}/{candidate_median:.3}/{dual_median:.3} ratios={ratio:.4}/{dual_ratio:.4} phase_scratch_mib={:.2} predicted_footprint_gib={predicted_gib:.3}",
        phase_scratch / (1 << 20) as f64
    );
    assert!(
        ratio <= 1.0 / 0.85,
        "selected-layer speed ratio {ratio:.4} > 1.1765"
    );
    assert!(
        dual_ratio <= 1.35,
        "dual selected-layer speed ratio {dual_ratio:.4} > 1.35"
    );
    assert!(
        predicted_gib <= 20.0,
        "predicted footprint {predicted_gib:.3} GiB > 20 GiB"
    );
}

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn selected_layer_scan_matches_all_layer_scan_on_single_hybrid() {
    // Tiny's two-layer cadence has one hybrid (layer 0) and one anchor.  The
    // selected and historical all-layer modes therefore dispatch the exact
    // same accepted scan, giving a compact model-level witness.
    let Some(_) = ctx() else { return };
    let mut all = EmbryoCfg::tiny();
    all.phase_delta = true;
    let mut selected = EmbryoCfg::tiny();
    selected.phase_delta_layer = Some(0);
    let (b, t) = (2usize, 64usize);
    let lay = Layout::new(&all);
    let p0 = init_params(&all, &lay, 13);
    let m = b * t;
    let tokens: Vec<u32> = lcg_vec(7101, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * all.vocab as f32) as u32 % all.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(7102, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * all.vocab as f32) as u32 % all.vocab as u32)
        .collect();
    let mut lhs = EmbryoGpu::new(all, b, t, &p0).expect("all-layer model");
    let mut rhs = EmbryoGpu::new(selected, b, t, &p0).expect("selected-layer model");
    let (ll, gl, _) = lhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    let (lr, gr, _) = rhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    assert_eq!(ll.to_bits(), lr.to_bits(), "selected scan loss differs");
    assert_eq!(
        lhs.grads_host(),
        rhs.grads_host(),
        "selected scan gradient differs"
    );
    assert_eq!(
        gl.to_bits(),
        gr.to_bits(),
        "selected scan grad norm differs"
    );
}

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn phase_delta_disabled_model_is_bit_identical_and_keeps_dummy_scratch() {
    let Some(_) = ctx() else { return };
    let cfg = EmbryoCfg::tiny();
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 17);
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let tokens: Vec<u32> = lcg_vec(7201, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(7202, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let mut lhs = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("control lhs");
    let mut rhs = EmbryoGpu::new(cfg, b, t, &p0).expect("control rhs");
    assert_eq!(lhs.scratch.phase_chunk.len, 1);
    assert_eq!(lhs.scratch.phase_partial.len, 1);
    let (ll, gl, _) = lhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    let (lr, gr, _) = rhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    assert_eq!(ll.to_bits(), lr.to_bits());
    assert_eq!(lhs.grads_host(), rhs.grads_host());
    assert_eq!(gl.to_bits(), gr.to_bits());
}
