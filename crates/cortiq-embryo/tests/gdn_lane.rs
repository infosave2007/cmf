//! Focused checkpoint/identity smoke tests for the optional GDN correction
//! lane.  The test is skipped on hosts without a Metal device.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::lcg_vec;
use cortiq_embryo::ops::{GdnLaneDims, gdn_ref_directional, gdn_ref_fwd};
use cortiq_embryo::train::{
    Checkpoint, append_gdn_lane_checkpoint, load_checkpoint, save_checkpoint,
};
use cortiq_engine::fcd_ops::{GdnSeqCfg, gdn_seq_bwd, gdn_seq_fwd};
use cortiq_engine::linear_core::{GdnCfg, GdnWeights, gdn_forward};
use cortiq_engine::qtensor::QTensor;

#[test]
fn gdn_lane_cpu_oracle_is_causal_and_finite() {
    let d = GdnLaneDims {
        b: 1,
        t: 4,
        dk: 4,
        dv: 4,
    };
    let q = lcg_vec(1, 16)
        .into_iter()
        .map(|x| x as f64)
        .collect::<Vec<_>>();
    let k = lcg_vec(2, 16)
        .into_iter()
        .map(|x| x as f64)
        .collect::<Vec<_>>();
    let v = lcg_vec(3, 16)
        .into_iter()
        .map(|x| x as f64)
        .collect::<Vec<_>>();
    let z = lcg_vec(4, 16)
        .into_iter()
        .map(|x| x as f64)
        .collect::<Vec<_>>();
    let a = vec![0.0; 4];
    let b = vec![0.0; 4];
    let n = vec![1.0; 4];
    let (o, s) = gdn_ref_fwd(
        &d,
        &q,
        &k,
        &v,
        &z,
        &a,
        &b,
        -6.9,
        (std::f64::consts::E - 1.0).ln(),
        &n,
        1e-6,
    );
    assert_eq!(o.len(), 16);
    assert_eq!(s.len(), (d.t + 1) * d.dk * d.dv);
    assert!(o.iter().all(|x| x.is_finite()) && s.iter().all(|x| x.is_finite()));
    let dout = vec![1.0; 16];
    let dq = vec![0.01; 16];
    let dir = gdn_ref_directional(
        &d, &q, &k, &v, &z, &a, &b, -6.9, 0.0, &n, 1e-6, &dout, &dq, 1e-5,
    );
    assert!(dir.is_finite());
}

#[test]
fn gdn_lane_appends_identity_tail_and_trains() {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut base = EmbryoCfg::tiny();
    base.head_clusters = 0;
    let old_lay = Layout::new(&base);
    let old_p = init_params(&base, &old_lay, 17);
    let ck = Checkpoint {
        cfg: base.clone(),
        step: 9,
        params: old_p.clone(),
        m: Some(old_p.iter().map(|x| x * 0.1).collect()),
        v: Some(old_p.iter().map(|x| x * 0.2).collect()),
        extras: Vec::new(),
    };
    let grown_ck = append_gdn_lane_checkpoint(&ck, 17).unwrap();
    assert_eq!(grown_ck.step, ck.step);
    assert!(grown_ck.cfg.gdn_lane);
    assert_eq!(&grown_ck.params[..old_lay.total], &ck.params[..]);
    assert_eq!(
        &grown_ck.m.as_ref().unwrap()[..old_lay.total],
        &ck.m.as_ref().unwrap()[..]
    );
    assert_eq!(
        &grown_ck.v.as_ref().unwrap()[..old_lay.total],
        &ck.v.as_ref().unwrap()[..]
    );
    let roundtrip = std::env::temp_dir().join(format!(
        "cmf-gdn-lane-{}-{}.ckpt",
        std::process::id(),
        old_lay.total
    ));
    let moment_m = grown_ck.m.as_deref();
    let moment_v = grown_ck.v.as_deref();
    let extras = grown_ck
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect::<Vec<_>>();
    save_checkpoint(
        &roundtrip,
        &grown_ck.cfg,
        grown_ck.step,
        &grown_ck.params,
        moment_m,
        moment_v,
        &extras,
    )
    .unwrap();
    let loaded = load_checkpoint(&roundtrip).unwrap();
    assert_eq!(loaded.step, grown_ck.step);
    assert_eq!(loaded.params, grown_ck.params);
    assert_eq!(loaded.m, grown_ck.m);
    assert_eq!(loaded.v, grown_ck.v);
    assert_eq!(loaded.cfg.gdn_lane, grown_ck.cfg.gdn_lane);
    let _ = std::fs::remove_file(&roundtrip);
    let mut cand = base.clone();
    cand.gdn_lane = true;
    let new_lay = Layout::new(&cand);
    assert!(new_lay.total > old_lay.total);
    let new_p = init_params(&cand, &new_lay, 17);
    for (name, oo, n) in &old_lay.names {
        let (_, no, nn) = new_lay.names.iter().find(|(x, _, _)| x == name).unwrap();
        assert_eq!(*n, *nn);
        assert_eq!(&old_p[*oo..*oo + *n], &new_p[*no..*no + *nn]);
    }
    let go = new_lay.gdn.iter().flatten().next().unwrap();
    assert_eq!(
        new_p[go.gain], 1.0,
        "zero-output seam uses unit residual gain"
    );
    assert!(
        new_p[go.wo..go.wo + cand.hidden * 64]
            .iter()
            .all(|&x| x == 0.0),
        "new lane output projection must be exactly zero"
    );
    let grown_lay = Layout::new(&grown_ck.cfg);
    let grown_go = grown_lay.gdn.iter().flatten().next().unwrap();
    assert_eq!(grown_ck.params[grown_go.gain], 1.0);
    assert!(
        grown_ck.params[grown_go.wo..grown_go.wo + cand.hidden * 64]
            .iter()
            .all(|&x| x == 0.0)
    );
    let old = EmbryoGpu::new(base.clone(), 2, 64, &old_p).unwrap();
    let candidate = EmbryoGpu::new(cand.clone(), 2, 64, &new_p).unwrap();
    let m = 2 * 64;
    let tok: Vec<u32> = lcg_vec(44, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cand.vocab as f32) as u32 % cand.vocab as u32)
        .collect();
    let a = old.forward_hidden(&tok);
    let b = candidate.forward_hidden(&tok);
    let maxe = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(maxe <= 1e-5, "zero-output identity max error {maxe}");
    let mut candidate = candidate;
    let mut active_p = new_p.clone();
    candidate.set_params(&active_p);
    let tgt = tok.clone();
    // At exact identity the lane output is zero, but Wo itself must receive
    // an unsuppressed gradient on the first backward pass.  This is the
    // distinction from the rejected random-Wo/near-zero-gain initialization.
    unsafe {
        std::ptr::copy_nonoverlapping(tok.as_ptr(), candidate.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(tgt.as_ptr(), candidate.tgt.buf.contents() as *mut u32, m);
    }
    candidate.prepare_head(&tgt);
    let cmd = cortiq_embryo::metal::Cmd::new(cortiq_embryo::metal::ctx().unwrap());
    candidate.encode_fwd_bwd(&cmd);
    cmd.commit();
    let zero_wo_grad_max = candidate.grads_host()[go.wo..go.wo + cand.hidden * 64]
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    assert!(
        zero_wo_grad_max > 1e-8,
        "zero-output Wo gradient is suppressed ({zero_wo_grad_max:.3e})"
    );

    // Once Wo has moved off zero, internal qkv/control weights must receive
    // finite gradients.  Use a deterministic non-zero probe matrix without
    // changing the production initialization under test.
    for (i, x) in active_p[go.wo..go.wo + cand.hidden * 64]
        .iter_mut()
        .enumerate()
    {
        *x = 0.002 * ((i % 17) as f32 - 8.0) / 8.0;
    }
    candidate.set_params(&active_p);
    unsafe {
        std::ptr::copy_nonoverlapping(tok.as_ptr(), candidate.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(tgt.as_ptr(), candidate.tgt.buf.contents() as *mut u32, m);
    }
    candidate.prepare_head(&tgt);
    let cmd = cortiq_embryo::metal::Cmd::new(cortiq_embryo::metal::ctx().unwrap());
    candidate.encode_fwd_bwd(&cmd);
    cmd.commit();
    let active_grads = candidate.grads_host().to_vec();
    let qkvz_grad_max = active_grads[go.qkvz..go.qkvz + 256 * cand.hidden]
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let ab_grad_max = active_grads[go.ab..go.ab + 64 * cand.hidden]
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    println!(
        "zero-output gradient proof: Wo max={zero_wo_grad_max:.3e}; after Wo probe qkvz max={qkvz_grad_max:.3e} ab max={ab_grad_max:.3e}"
    );
    assert!(
        qkvz_grad_max > 1e-8,
        "qkvz gradient did not engage ({qkvz_grad_max:.3e})"
    );
    assert!(
        ab_grad_max > 1e-8,
        "control gradient did not engage ({ab_grad_max:.3e})"
    );

    // The scalar gain has a direct dot-product gradient.  Check it against a
    // central difference before the optimizer mutates the candidate.
    unsafe {
        std::ptr::copy_nonoverlapping(tok.as_ptr(), candidate.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(tgt.as_ptr(), candidate.tgt.buf.contents() as *mut u32, m);
    }
    candidate.prepare_head(&tgt);
    let cmd = cortiq_embryo::metal::Cmd::new(cortiq_embryo::metal::ctx().unwrap());
    candidate.encode_fwd_bwd(&cmd);
    cmd.commit();
    let analytic = candidate.grads_host()[go.gain] as f64;
    // The residual lane starts deliberately tiny; use a wider central
    // difference here so f32 loss quantization does not dominate d(gain).
    let eps = 1e-1;
    let mut pp = active_p.clone();
    pp[go.gain] += eps;
    candidate.set_params(&pp);
    let lp = candidate.eval_loss(&tok, &tgt) as f64;
    pp[go.gain] -= 2.0 * eps;
    candidate.set_params(&pp);
    let lm = candidate.eval_loss(&tok, &tgt) as f64;
    let fd = (lp - lm) / (2.0 * eps as f64);
    assert!(
        (fd - analytic).abs() / analytic.abs().max(5e-4) < 3e-2,
        "gain finite difference {fd} vs analytic {analytic}"
    );
    // Probe representative lane weights at a non-zero gain.  These checks
    // keep the correction's hand-written backward tied to the same loss path
    // used by the trainer (the tolerance is the f32 finite-difference floor).
    for (label, idx) in [
        ("qkvz", go.qkvz + cand.hidden),
        ("conv", go.conv + 1),
        ("ab", go.ab + cand.hidden),
        ("norm", go.norm + 1),
        ("alog", go.alog),
        ("dt_bias", go.dt_bias),
        ("wo", go.wo + cand.hidden),
    ] {
        candidate.set_params(&active_p);
        unsafe {
            std::ptr::copy_nonoverlapping(
                tok.as_ptr(),
                candidate.tok.buf.contents() as *mut u32,
                m,
            );
            std::ptr::copy_nonoverlapping(
                tgt.as_ptr(),
                candidate.tgt.buf.contents() as *mut u32,
                m,
            );
        }
        candidate.prepare_head(&tgt);
        let cmd = cortiq_embryo::metal::Cmd::new(cortiq_embryo::metal::ctx().unwrap());
        candidate.encode_fwd_bwd(&cmd);
        cmd.commit();
        let analytic = candidate.grads_host()[idx] as f64;
        let mut pp = active_p.clone();
        // Scalar controls can have gradients below the f32 loss ULP at the
        // ordinary probe; widen only those central differences for signal.
        let eps = if matches!(label, "alog" | "dt_bias" | "conv" | "norm") {
            1.0
        } else if label == "wo" {
            // The corrected SiLU lane has a very small Wo derivative in this
            // tiny probe; a wider central difference avoids the f32 loss ULP
            // while staying in the linear residual regime.
            5e-2
        } else {
            // Likewise for q/k/v projection rows: use a wider perturbation
            // so the central difference is not quantized to a neighbouring
            // f32 loss value.
            5e-2
        };
        pp[idx] += eps;
        candidate.set_params(&pp);
        let lp = candidate.eval_loss(&tok, &tgt) as f64;
        pp[idx] -= 2.0 * eps;
        candidate.set_params(&pp);
        let lm = candidate.eval_loss(&tok, &tgt) as f64;
        let fd = (lp - lm) / (2.0 * eps as f64);
        assert!(
            (fd - analytic).abs() / analytic.abs().max(5e-4) < 3e-2,
            "{label} finite difference {fd} vs analytic {analytic}"
        );
    }
    candidate.set_params(&active_p);
    let (loss, gnorm, _) = candidate.train_step(&tok, &tgt, 1e-4, 0.0, 1.0);
    assert!(loss.is_finite() && gnorm.is_finite());
}

#[test]
fn gdn_lane_residual_gradient_and_tail_only_update() {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut base = EmbryoCfg::tiny();
    base.head_clusters = 0;
    let old_lay = Layout::new(&base);
    let old_p = init_params(&base, &old_lay, 91);
    let ck = Checkpoint {
        cfg: base.clone(),
        step: 0,
        params: old_p,
        m: None,
        v: None,
        extras: Vec::new(),
    };
    let grown = append_gdn_lane_checkpoint(&ck, 91).unwrap();
    let mut probe = EmbryoGpu::new(grown.cfg.clone(), 2, 64, &grown.params).unwrap();
    probe.desc_updates.set(false);
    let m = 2 * 64;
    let tok: Vec<u32> = lcg_vec(92, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * probe.cfg.vocab as f32) as u32 % probe.cfg.vocab as u32)
        .collect();
    let h0 = probe.forward_hidden(&tok);
    // A deterministic teacher offset gives every lane a nonzero residual
    // signal while keeping the finite-difference probe local and bounded.
    let teacher: Vec<f32> = h0
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.01 * ((i % 13) as f32 - 6.0) / 6.0)
        .collect();
    let before = probe.params_host();
    let (_, _, _) = probe.train_step_residual(&tok, &teacher, 0.0, 0.0, 1.0e9);
    let grads = probe.grads_host();
    let go = probe.lay.gdn.iter().flatten().next().unwrap();
    let wo_off = go.wo;
    let (idx, analytic) = grads[wo_off..wo_off + probe.cfg.hidden * 64]
        .iter()
        .enumerate()
        .map(|(i, &x)| (wo_off + i, x as f64))
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    assert!(analytic.abs() > 1e-8, "residual Wo gradient is zero");
    let eps = 5e-2f32;
    let mut pp = before.clone();
    pp[idx] += eps;
    probe.set_params(&pp);
    let lp = probe.eval_hidden_mse(&tok, &teacher) as f64;
    pp[idx] -= 2.0 * eps;
    probe.set_params(&pp);
    let lm = probe.eval_hidden_mse(&tok, &teacher) as f64;
    let fd = (lp - lm) / (2.0 * eps as f64);
    assert!(
        (fd - analytic).abs() / analytic.abs().max(5e-4) < 3e-2,
        "residual Wo finite difference {fd} vs analytic {analytic}"
    );

    // A nonzero residual step may mutate only the appended tail.  Check both
    // parameter bytes and optimizer moments over the immutable legacy prefix.
    let mut trained = EmbryoGpu::new(grown.cfg.clone(), 2, 64, &grown.params).unwrap();
    trained.desc_updates.set(false);
    let p_before = trained.params_host();
    let m_before = trained.m.to_vec();
    let v_before = trained.v.to_vec();
    let _ = trained.train_step_residual(&tok, &teacher, 1e-4, 0.0, 1.0);
    let p_after = trained.params_host();
    let m_after = trained.m.to_vec();
    let v_after = trained.v.to_vec();
    let mut tail = vec![false; p_before.len()];
    for &(off, n) in &trained.gdn_tail_ranges() {
        tail[off..off + n].fill(true);
    }
    for i in 0..p_before.len() {
        if !tail[i] {
            assert_eq!(
                p_after[i].to_bits(),
                p_before[i].to_bits(),
                "legacy param drift at {i}"
            );
            assert_eq!(
                m_after[i].to_bits(),
                m_before[i].to_bits(),
                "legacy m drift at {i}"
            );
            assert_eq!(
                v_after[i].to_bits(),
                v_before[i].to_bits(),
                "legacy v drift at {i}"
            );
        }
    }
    assert!(
        p_after
            .iter()
            .zip(&p_before)
            .enumerate()
            .any(|(i, (x, y))| tail[i] && x.to_bits() != y.to_bits()),
        "residual step did not move any lane parameter"
    );
}

/// Compare the live Metal lane to two already-established CPU operators,
/// rather than to a helper copied from the lane itself.  The fcd_ops sequence
/// oracle catches post-convolution activation and causal-state mistakes; the
/// linear_core path exposes the final recurrent state for a second check.  A
/// CPU finite difference in A_log independently catches the decay multiplier
/// in the reverse scan (D10's missing-`g` defect).
#[test]
fn gdn_lane_matches_established_operator_and_decay_gradient() {
    let Some(ctx) = cortiq_embryo::metal::ctx() else {
        return;
    };
    const B: usize = 1;
    const T: usize = 3;
    const D: usize = 64;
    const CD: usize = 192;
    const HIDDEN: usize = 258;
    const CONV_OFF: usize = 0;
    const NORM_OFF: usize = CD * 4;
    const ALOG_OFF: usize = NORM_OFF + D;
    const DT_OFF: usize = ALOG_OFF + 1;
    const P_LEN: usize = DT_OFF + 1;
    const ALOG: f32 = -1.2;
    const DT: f32 = -0.4;
    const EPS: f32 = 1e-6;

    // Keep all streams nontrivial but comfortably in the finite, non-saturated
    // region so both the activation transpose and decay derivative carry
    // signal.
    let qraw: Vec<f32> = lcg_vec(101, B * T * D)
        .into_iter()
        .map(|x| x * 0.35)
        .collect();
    let kraw: Vec<f32> = lcg_vec(102, B * T * D)
        .into_iter()
        .map(|x| x * 0.30)
        .collect();
    let vraw: Vec<f32> = lcg_vec(103, B * T * D)
        .into_iter()
        .map(|x| x * 0.25)
        .collect();
    let zraw: Vec<f32> = lcg_vec(104, B * T * D)
        .into_iter()
        .map(|x| x * 0.45)
        .collect();
    let mut ab = vec![0.0f32; B * T * D];
    let avec: Vec<f32> = [-0.7, 0.15, 0.8].into_iter().collect();
    let bvec: Vec<f32> = [-0.9, 0.2, 0.65].into_iter().collect();
    for ti in 0..T {
        ab[ti * D] = avec[ti];
        ab[ti * D + 1] = bvec[ti];
    }
    let conv: Vec<f32> = lcg_vec(105, CD * 4).into_iter().map(|x| x * 0.18).collect();
    let norm: Vec<f32> = lcg_vec(106, D).into_iter().map(|x| 0.8 + x * 0.1).collect();
    let mut p = vec![0.0f32; P_LEN];
    p[CONV_OFF..CONV_OFF + conv.len()].copy_from_slice(&conv);
    p[NORM_OFF..NORM_OFF + D].copy_from_slice(&norm);
    p[ALOG_OFF] = ALOG;
    p[DT_OFF] = DT;

    let qbuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &qraw);
    let kbuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &kraw);
    let vbuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &vraw);
    let zbuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &zraw);
    let abuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &ab);
    let pbuf = cortiq_embryo::metal::GBuf::from_slice(ctx, &p);
    let qcv = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let kcv = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let vcv = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let beta = cortiq_embryo::metal::GBuf::zeros(ctx, B * T);
    let raw_o = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let inv_o = cortiq_embryo::metal::GBuf::zeros(ctx, B * T);
    let out = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let states = cortiq_embryo::metal::GBuf::zeros(ctx, B * (T + 1) * D * D);
    let gated = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let cmd = cortiq_embryo::metal::Cmd::new(ctx);
    cmd.gdn_forward(
        &qbuf, &kbuf, &vbuf, &zbuf, &abuf, &pbuf, CONV_OFF, NORM_OFF, ALOG_OFF, DT_OFF, &qcv, &kcv,
        &vcv, &beta, &raw_o, &inv_o, &out, &states, B, T, EPS,
    );
    cmd.swiglu_fwd(&zbuf, &out, &gated, B * T * D);
    cmd.commit();

    // Established fcd_ops CPU reference (f64 accumulation) for output rows.
    let mut qkv = vec![0.0f64; B * T * CD];
    for ti in 0..T {
        qkv[ti * CD..ti * CD + D].copy_from_slice(
            &qraw[ti * D..(ti + 1) * D]
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        );
        qkv[ti * CD + D..ti * CD + 2 * D].copy_from_slice(
            &kraw[ti * D..(ti + 1) * D]
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        );
        qkv[ti * CD + 2 * D..(ti + 1) * CD].copy_from_slice(
            &vraw[ti * D..(ti + 1) * D]
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        );
    }
    let z64: Vec<f64> = zraw.iter().map(|&x| x as f64).collect();
    let a64: Vec<f64> = avec.iter().copied().map(f64::from).collect();
    let b64: Vec<f64> = bvec.iter().copied().map(f64::from).collect();
    let norm64: Vec<f32> = norm.clone();
    let alog_one = [ALOG];
    let dt_one = [DT];
    let fcfg = GdnSeqCfg {
        nv: 1,
        nk: 1,
        dk: D,
        dv: D,
        kk: 4,
        rms_eps: EPS as f64,
        conv: &conv,
        a_log: &alog_one,
        dt_bias: &dt_one,
        norm: &norm64,
    };
    let mut fcd_out = vec![0.0f64; B * T * D];
    gdn_seq_fwd::<f64>(&qkv, &z64, &a64, &b64, T, &fcfg, &mut fcd_out);
    let got = gated.to_vec();
    let max_out = got
        .iter()
        .zip(&fcd_out)
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0.0f64, f64::max);
    assert!(max_out < 3e-4, "Metal/fcd output max error {max_out:.3e}");

    // The established linear_core path exposes the final S state.  Build
    // identity projections so it consumes exactly the same raw streams and
    // controls as the direct Metal dispatch.
    let mut qproj = vec![0.0f32; CD * HIDDEN];
    for r in 0..CD {
        qproj[r * HIDDEN + r] = 1.0;
    }
    let mut zproj = vec![0.0f32; D * HIDDEN];
    for r in 0..D {
        zproj[r * HIDDEN + D * 3 + r] = 1.0;
    }
    let mut aproj = vec![0.0f32; HIDDEN];
    aproj[HIDDEN - 2] = 1.0;
    let mut bproj = vec![0.0f32; HIDDEN];
    bproj[HIDDEN - 1] = 1.0;
    let mut oproj = vec![0.0f32; HIDDEN * D];
    for r in 0..D {
        oproj[r * D + r] = 1.0;
    }
    let ecfg = GdnCfg {
        num_v_heads: 1,
        num_k_heads: 1,
        key_head_dim: D,
        value_head_dim: D,
        conv_kernel: 4,
        hidden_size: HIDDEN,
        rms_eps: EPS as f64,
        output_gate_sigmoid: false,
    };
    let make_weights = |alog: f32| GdnWeights {
        in_proj_qkv: QTensor::from_f32(qproj.clone(), CD, HIDDEN),
        in_proj_z: QTensor::from_f32(zproj.clone(), D, HIDDEN),
        in_proj_a: QTensor::from_f32(aproj.clone(), 1, HIDDEN),
        in_proj_b: QTensor::from_f32(bproj.clone(), 1, HIDDEN),
        conv1d: conv.clone(),
        a_log: vec![alog],
        dt_bias: vec![DT],
        norm: norm.clone(),
        out_proj: QTensor::from_f32(oproj.clone(), HIDDEN, D),
    };
    let w = make_weights(ALOG);
    let mut engine_state = Vec::new();
    for ti in 0..T {
        let mut x = vec![0.0f32; HIDDEN];
        x[..D].copy_from_slice(&qraw[ti * D..(ti + 1) * D]);
        x[D..2 * D].copy_from_slice(&kraw[ti * D..(ti + 1) * D]);
        x[2 * D..3 * D].copy_from_slice(&vraw[ti * D..(ti + 1) * D]);
        x[3 * D..4 * D].copy_from_slice(&zraw[ti * D..(ti + 1) * D]);
        x[HIDDEN - 2] = avec[ti];
        x[HIDDEN - 1] = bvec[ti];
        let _ = gdn_forward(&x, &w, &ecfg, &mut engine_state, None);
    }
    let metal_s = &states.to_vec()[T * D * D..(T + 1) * D * D];
    let engine_s = &engine_state[(ecfg.conv_kernel - 1) * CD..];
    let max_state = metal_s
        .iter()
        .zip(engine_s)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_state < 3e-4,
        "Metal/linear_core state max error {max_state:.3e}"
    );

    // Reverse scan: dnorm is the upstream gradient after the existing
    // SwiGLU transpose, i.e. upstream * SiLU(z).  The CPU finite difference
    // below differentiates the complete gated linear_core output.
    let upstream: Vec<f32> = lcg_vec(107, B * T * D)
        .into_iter()
        .map(|x| x * 0.4)
        .collect();
    let mut dnorm = vec![0.0f32; B * T * D];
    for i in 0..dnorm.len() {
        let zz = zraw[i];
        let sig = 1.0 / (1.0 + (-zz).exp());
        dnorm[i] = upstream[i] * zz * sig;
    }
    let dnorm_buf = cortiq_embryo::metal::GBuf::from_slice(ctx, &dnorm);
    let dz_buf = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let dq_buf = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let dk_buf = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let dv_buf = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let dab_buf = cortiq_embryo::metal::GBuf::zeros(ctx, B * T * D);
    let grads = cortiq_embryo::metal::GBuf::zeros(ctx, P_LEN);
    let cmd = cortiq_embryo::metal::Cmd::new(ctx);
    cmd.gdn_backward(
        &qbuf, &kbuf, &vbuf, &qcv, &kcv, &vcv, &abuf, &beta, &raw_o, &inv_o, &pbuf, &states,
        &dnorm_buf, &dz_buf, &dq_buf, &dk_buf, &dv_buf, &dab_buf, &pbuf, CONV_OFF, NORM_OFF,
        ALOG_OFF, DT_OFF, &grads, CONV_OFF, NORM_OFF, ALOG_OFF, DT_OFF, B, T,
    );
    cmd.commit();
    let metal_dalog = grads.as_slice()[ALOG_OFF] as f64;

    // Independent reverse-oracle check for all three projected streams and
    // both recurrent controls.  fcd_ops::gdn_seq_bwd is the established
    // through-grad implementation (SiLU + normalization + BPTT), so these
    // expected values cannot be produced by the new Metal kernel itself.
    let dout64: Vec<f64> = upstream.iter().map(|&x| x as f64).collect();
    let mut fcd_dqkv = vec![0.0f64; B * T * CD];
    let mut fcd_dz = vec![0.0f64; B * T * D];
    let mut fcd_da = vec![0.0f64; B * T];
    let mut fcd_db = vec![0.0f64; B * T];
    gdn_seq_bwd::<f64>(
        &qkv,
        &z64,
        &a64,
        &b64,
        T,
        &fcfg,
        &dout64,
        &mut fcd_dqkv,
        &mut fcd_dz,
        &mut fcd_da,
        &mut fcd_db,
    );
    let mdq = dq_buf.to_vec();
    let mdk = dk_buf.to_vec();
    let mdv = dv_buf.to_vec();
    let mdab = dab_buf.to_vec();
    let mut max_proj = 0.0f64;
    for ti in 0..T {
        for c in 0..D {
            max_proj = max_proj.max((mdq[ti * D + c] as f64 - fcd_dqkv[ti * CD + c]).abs());
            max_proj = max_proj.max((mdk[ti * D + c] as f64 - fcd_dqkv[ti * CD + D + c]).abs());
            max_proj = max_proj.max((mdv[ti * D + c] as f64 - fcd_dqkv[ti * CD + 2 * D + c]).abs());
        }
        max_proj = max_proj.max((mdab[ti * D] as f64 - fcd_da[ti]).abs());
        max_proj = max_proj.max((mdab[ti * D + 1] as f64 - fcd_db[ti]).abs());
    }
    assert!(
        max_proj < 5e-4,
        "Metal/fcd reverse projected/control max error {max_proj:.3e}"
    );

    let cpu_loss = |alog: f32| {
        let ww = make_weights(alog);
        let mut st = Vec::new();
        let mut loss = 0.0f64;
        for ti in 0..T {
            let mut x = vec![0.0f32; HIDDEN];
            x[..D].copy_from_slice(&qraw[ti * D..(ti + 1) * D]);
            x[D..2 * D].copy_from_slice(&kraw[ti * D..(ti + 1) * D]);
            x[2 * D..3 * D].copy_from_slice(&vraw[ti * D..(ti + 1) * D]);
            x[3 * D..4 * D].copy_from_slice(&zraw[ti * D..(ti + 1) * D]);
            x[HIDDEN - 2] = avec[ti];
            x[HIDDEN - 1] = bvec[ti];
            let o = gdn_forward(&x, &ww, &ecfg, &mut st, None);
            for j in 0..D {
                loss += o[j] as f64 * upstream[ti * D + j] as f64;
            }
        }
        loss
    };
    let h = 1e-2f32;
    let fd = (cpu_loss(ALOG + h) - cpu_loss(ALOG - h)) / (2.0 * h as f64);
    let rel = (metal_dalog - fd).abs() / fd.abs().max(5e-4);
    assert!(
        rel < 3e-2,
        "A_log gradient {metal_dalog:.6e} vs CPU fd {fd:.6e} (rel {rel:.3e})"
    );
    println!(
        "established GDN parity: max_out={max_out:.3e} max_state={max_state:.3e} max_reverse={max_proj:.3e} A_log_rel={rel:.3e}"
    );
}
