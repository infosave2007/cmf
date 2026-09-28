//! Native Vulkan operator/reference gates.
//!
//! These tests deliberately compare the resident WGSL scans with the existing
//! f64 reference laws over more than one 64-token chunk.  They are not a CPU
//! fallback: a missing/non-NVIDIA Vulkan adapter fails the test process rather
//! than being reported as a pass.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, HkDims, HkGrads, HkWork, ctx, hk_pow_table};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::{
    hk_decay_grid, hk_ref_bwd, hk_ref_fwd, lcg_vec, phase_delta_ref_bwd, phase_delta_ref_fwd,
};

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0f64, |acc, x| acc.max(x.abs()))
        .max(1e-12);
    got.iter()
        .zip(want)
        .map(|(x, y)| (*x as f64 - *y).abs())
        .fold(0.0, f64::max)
        / scale
}

fn assert_close(name: &str, got: &[f32], want: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), want.len(), "{name} length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w).abs();
        let lim = atol + rtol * w.abs();
        assert!(
            err <= lim,
            "{name}[{i}] got {g:?}, want {w:?}, err {err:e} > {lim:e}"
        );
    }
}

fn as_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| *v as f64).collect()
}

#[test]
fn legacy_hybrid_k_cross_chunk_matches_reference_forward_and_backward() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    // T=129 crosses both the 64-token boundary and the non-multiple tail;
    // B=2 and NH=2 also catch state-boundary indexing mistakes.
    let d = HkDims {
        b: 2,
        t: 128,
        nh: 2,
        nph: 5,
        // Two value blocks exercise the production Vulkan legacy reverse scan
        // rather than the serial seam fallback.
        dv: 48,
    };
    let rows = d.b * d.t;
    let thq: Vec<f32> = lcg_vec(101, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 2.0)
        .collect();
    let thk: Vec<f32> = lcg_vec(102, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 2.0)
        .collect();
    let v = lcg_vec(103, rows * d.nh * d.dv);
    let kappa: Vec<f32> = lcg_vec(104, rows * d.nh)
        .iter()
        .map(|x| 0.25 + 0.5 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(105, rows * d.nh * d.dv);
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let td = as_f64(&thq);
    let kd = as_f64(&thk);
    let vd = as_f64(&v);
    let kapd = as_f64(&kappa);
    let dod = as_f64(&dout);
    let want_o = hk_ref_fwd(&d, &td, &kd, &vd, &kapd, &as_f64(&decay));
    let (want_q, want_k, want_v, want_kap) =
        hk_ref_bwd(&d, &td, &kd, &vd, &kapd, &as_f64(&decay), &dod);

    let z = |n: usize| GBuf::zeros(c, n);
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kappa),
    );
    let p2 = d.p2();
    let (gphq, gphk, gkv, gout) = (
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * d.dv),
    );
    let nst = d.b * d.nh * (d.t.div_ceil(64) + 1) * p2 * d.dv;
    let states = z(nst);
    let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + p2));
    let w = HkWork {
        thq: &gthq,
        thk: &gthk,
        v: &gv,
        kappa: &gkap,
        pow: &GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, d.nph)),
        pow_off: 0,
        phq: &gphq,
        phk: &gphk,
        kv: &gkv,
        states: &states,
        out: &gout,
        phase_chunk: None,
        phase_partial: Some(&gpartial),
    };
    let (gdst, gdkv, gdphq, gdphk) = (
        z(nst),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
    );
    let (gdthq, gdthk, gdv, gdkap) = (
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.dv),
        z(rows * d.nh),
    );
    let gdout = GBuf::from_slice(c, &dout);
    let gr = HkGrads {
        dout: &gdout,
        dstates: &gdst,
        dkv: &gdkv,
        dphq: &gdphq,
        dphk: &gdphk,
        dthq: &gdthq,
        dthk: &gdthk,
        dv: &gdv,
        dkappa: &gdkap,
    };
    let cmd = Cmd::new(c);
    cmd.hk_forward(&d, &w);
    cmd.hk_backward(&d, &w, &gr, 0.0);
    cmd.commit();

    for (name, err, limit) in [
        ("forward", rel_err(&gout.to_vec(), &want_o), 5e-4),
        ("dtheta-q", rel_err(&gdthq.to_vec(), &want_q), 5e-4),
        ("dtheta-k", rel_err(&gdthk.to_vec(), &want_k), 5e-4),
        ("dv", rel_err(&gdv.to_vec(), &want_v), 5e-4),
        ("dkappa", rel_err(&gdkap.to_vec(), &want_kap), 5e-4),
    ] {
        assert!(err < limit, "legacy hybrid-k {name} relative error {err:e}");
    }
}

#[test]
fn phase_delta_cross_chunk_matches_reference_forward_and_backward() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let d = HkDims {
        b: 2,
        t: 128,
        nh: 2,
        nph: 3,
        dv: 8,
    };
    let rows = d.b * d.t;
    let thq: Vec<f32> = lcg_vec(201, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 1.5)
        .collect();
    let thk: Vec<f32> = lcg_vec(202, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 1.5)
        .collect();
    let v = lcg_vec(203, rows * d.nh * d.dv);
    let kappa: Vec<f32> = lcg_vec(204, rows * d.nh)
        .iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(205, rows * d.nh * d.dv);
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let td = as_f64(&thq);
    let kd = as_f64(&thk);
    let vd = as_f64(&v);
    let kapd = as_f64(&kappa);
    let dod = as_f64(&dout);
    let decayd = as_f64(&decay);
    let want_o = phase_delta_ref_fwd(&d, &td, &kd, &vd, &kapd, &decayd);
    let (want_q, want_k, want_v, want_kap) =
        phase_delta_ref_bwd(&d, &td, &kd, &vd, &kapd, &decayd, &dod);

    let z = |n: usize| GBuf::zeros(c, n);
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kappa),
    );
    let p2 = d.p2();
    let nst = d.b * d.nh * (d.t.div_ceil(64) + 1) * p2 * d.dv;
    let states = z(nst);
    let gchunk = z(d.b * d.nh * 65 * p2 * d.dv);
    let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + p2));
    let gpow = GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, d.nph));
    let (gphq, gphk, gkv, gout) = (
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * d.dv),
    );
    let w = HkWork {
        thq: &gthq,
        thk: &gthk,
        v: &gv,
        kappa: &gkap,
        pow: &gpow,
        pow_off: 0,
        phq: &gphq,
        phk: &gphk,
        kv: &gkv,
        states: &states,
        out: &gout,
        phase_chunk: Some(&gchunk),
        phase_partial: Some(&gpartial),
    };
    let (gdst, gdkv, gdphq, gdphk) = (
        z(nst),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
    );
    let (gdthq, gdthk, gdv, gdkap) = (
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.dv),
        z(rows * d.nh),
    );
    let gdout = GBuf::from_slice(c, &dout);
    let gr = HkGrads {
        dout: &gdout,
        dstates: &gdst,
        dkv: &gdkv,
        dphq: &gdphq,
        dphk: &gdphk,
        dthq: &gdthq,
        dthk: &gdthk,
        dv: &gdv,
        dkappa: &gdkap,
    };
    let cmd = Cmd::new(c);
    cmd.phase_delta_forward_reset(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();

    for (name, err, limit) in [
        ("forward", rel_err(&gout.to_vec(), &want_o), 2e-4),
        ("dtheta-q", rel_err(&gdthq.to_vec(), &want_q), 3e-3),
        ("dtheta-k", rel_err(&gdthk.to_vec(), &want_k), 3e-3),
        ("dv", rel_err(&gdv.to_vec(), &want_v), 3e-4),
        ("dkappa", rel_err(&gdkap.to_vec(), &want_kap), 3e-3),
    ] {
        assert!(err < limit, "Phase-Delta {name} relative error {err:e}");
    }
}

fn run_tiny_variant(mut cfg: EmbryoCfg, label: &str, phase: bool) {
    let b = 1usize;
    let t = 128usize;
    let m = b * t;
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 0x5eed);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &params)
        .unwrap_or_else(|| panic!("native Vulkan adapter unavailable for {label}"));
    gpu.desc_updates.set(false);
    let tokens: Vec<u32> = (0..m).map(|i| (i * 37 % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (0..m).map(|i| ((i * 37 + 1) % cfg.vocab) as u32).collect();
    let eval = gpu.eval_loss(&tokens, &targets);
    assert!(eval.is_finite(), "{label} eval loss={eval:?}");
    let (loss, grad, elapsed) = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1.0);
    assert!(loss.is_finite() && grad.is_finite() && elapsed.is_finite());
    assert!(
        gpu.params_host().iter().all(|x| x.is_finite()),
        "{label} params"
    );
    assert!(
        gpu.grads_host().iter().all(|x| x.is_finite()),
        "{label} grads"
    );
    assert!(
        gpu.m.to_vec().iter().all(|x| x.is_finite()),
        "{label} Adam m"
    );
    assert!(
        gpu.v.to_vec().iter().all(|x| x.is_finite()),
        "{label} Adam v"
    );
    if phase {
        let telemetry = gpu.phase_delta_telemetry();
        assert!(
            !telemetry.is_empty(),
            "{label} missing Phase-Delta telemetry"
        );
        assert!(
            telemetry.iter().all(|x| x.finite),
            "{label} Phase-Delta telemetry"
        );
    }
}

#[test]
fn tiny_whole_graph_variants_are_finite_over_cross_chunk_batch() {
    // Hierarchical head + anchor + legacy hybrid-k.
    run_tiny_variant(EmbryoCfg::tiny(), "legacy-anchor-hierarchical", false);

    // Selected Phase-Delta layer plus causal conv4; layer 1 is the hybrid in
    // tiny's default anchor cadence.
    let mut phase_cfg = EmbryoCfg::tiny();
    phase_cfg.phase_delta_layer = Some(0);
    phase_cfg.conv_k = 4;
    run_tiny_variant(phase_cfg, "selected-phase-conv4", true);

    // Frozen custom resonance descriptors and the same hierarchical
    // vocabulary head. This keeps the routing law explicit; no softmax
    // router is substituted.
    let mut route_cfg = EmbryoCfg::tiny();
    route_cfg.experts = 4;
    run_tiny_variant(route_cfg, "custom-resonance-top1", false);
}

#[test]
fn skill_mask_backward_reduces_per_neuron_and_respects_offset() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let (rows, n, off) = (3usize, 4usize, 5usize);
    let logits = GBuf::from_slice(
        c,
        &[99.0; 5]
            .into_iter()
            .chain([0.0, 1.0, -1.0, 2.0])
            .collect::<Vec<_>>(),
    );
    let hh_pre = vec![
        0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -2.0, 0.75, 1.0, -0.5, 0.25,
    ];
    let dh0 = vec![
        1.0, -0.5, 0.25, 2.0, -1.0, 0.75, -1.5, 0.5, 0.25, 1.25, -0.25, 0.5,
    ];
    let hh = GBuf::from_slice(c, &hh_pre);
    let dh = GBuf::from_slice(c, &dh0);
    let dm0 = vec![7.0, -3.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0];
    let dm = GBuf::from_slice(c, &dm0);
    let tau = 0.5f32;
    let l1 = 0.125f32;
    let cmd = Cmd::new(c);
    cmd.mask_bwd(&dh, &hh, &logits, off, &dm, rows, n, false, tau, l1);
    cmd.commit();

    let mut want_dh = dh0.clone();
    let mut want_dm = dm0.clone();
    let logits_host = unsafe { logits.as_slice() };
    for col in 0..n {
        let s = 1.0 / (1.0 + (-logits_host[off + col]).exp());
        let ds = s * (1.0 - s);
        let mut acc = 0.0;
        for row in 0..rows {
            let idx = row * n + col;
            acc += dh0[idx] * hh_pre[idx];
            want_dh[idx] *= s;
        }
        want_dm[off + col] += (acc + l1) * ds;
    }
    assert_close(
        "skill mask dh",
        &dh.to_vec(),
        &want_dh.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        3e-6,
        3e-6,
    );
    assert_close(
        "skill mask dm",
        &dm.to_vec(),
        &want_dm.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        3e-6,
        3e-6,
    );

    let dh_hard = GBuf::from_slice(c, &dh0);
    let dm_hard = GBuf::from_slice(c, &dm0);
    let cmd = Cmd::new(c);
    cmd.mask_bwd(
        &dh_hard, &hh, &logits, off, &dm_hard, rows, n, true, tau, l1,
    );
    cmd.commit();
    let want_hard: Vec<f32> = dh0
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let col = idx % n;
            let s = 1.0 / (1.0 + (-logits_host[off + col]).exp());
            x * if s > tau { 1.0 } else { 0.0 }
        })
        .collect();
    assert_close(
        "hard skill mask dh",
        &dh_hard.to_vec(),
        &want_hard.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        3e-6,
        3e-6,
    );
    assert_eq!(dm_hard.to_vec(), dm0);
}

#[test]
fn scatter_add_is_float_deterministic_for_repeated_rows() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let dst = GBuf::from_slice(c, &[10.0, -3.0, 1.5, 2.5, -7.0, 4.0]);
    let idx = GBuf::from_u32(c, &[1, 1, 2, u32::MAX]);
    let src = GBuf::from_slice(c, &[0.25, 0.5, 1.0, -2.0, 3.0, 4.0, 8.0, 9.0]);
    let cmd = Cmd::new(c);
    cmd.scatter_add_rows(&dst, &idx, &src, 4, 2);
    cmd.commit();
    assert_eq!(
        dst.to_vec(),
        vec![10.0, -3.0, 2.75, 1.0, -4.0, 8.0],
        "repeated/invalid scatter rows must use f32 addition"
    );
}
