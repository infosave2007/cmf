//! hybrid_k CPU oracle: the closed-form analytic backward against central
//! finite differences of the literal recurrence (f64, tight tolerance).
use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd, hk_ref_fwd, lcg_vec};

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

#[test]
fn hk_closed_form_backward_matches_finite_differences() {
    let d = HkDims {
        b: 2,
        t: 9,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let n_k = d.b * d.t * d.nh;
    let thq: Vec<f64> = to64(&lcg_vec(1, n_th)).iter().map(|x| x * 3.0).collect();
    let thk: Vec<f64> = to64(&lcg_vec(2, n_th)).iter().map(|x| x * 3.0).collect();
    let v = to64(&lcg_vec(3, n_v));
    let kappa: Vec<f64> = to64(&lcg_vec(4, n_k))
        .iter()
        .map(|x| 0.5 + 0.4 * x)
        .collect();
    let decay = to64(&hk_decay_grid(d.nh, d.nph, 2.0, 16.0));
    let dout = to64(&lcg_vec(5, n_v));

    let loss = |thq: &[f64], thk: &[f64], v: &[f64], kappa: &[f64]| -> f64 {
        let o = hk_ref_fwd(&d, thq, thk, v, kappa, &decay);
        o.iter().zip(&dout).map(|(a, b)| a * b).sum()
    };
    let (dthq, dthk, dv, dkap) = hk_ref_bwd(&d, &thq, &thk, &v, &kappa, &decay, &dout);

    let eps = 1e-5;
    let check = |name: &str, x: &[f64], g: &[f64], f: &dyn Fn(&[f64]) -> f64| {
        let mut worst = 0.0f64;
        for i in 0..x.len() {
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
            let err = (fd - g[i]).abs() / (1.0 + fd.abs());
            worst = worst.max(err);
        }
        assert!(worst < 1e-6, "{name}: worst rel err {worst:e}");
        eprintln!("{name}: worst rel err {worst:e}");
    };
    check("dthq", &thq, &dthq, &|x| loss(x, &thk, &v, &kappa));
    check("dthk", &thk, &dthk, &|x| loss(&thq, x, &v, &kappa));
    check("dv", &v, &dv, &|x| loss(&thq, &thk, x, &kappa));
    check("dkappa", &kappa, &dkap, &|x| loss(&thq, &thk, &v, x));
}

#[test]
fn hk_decay_gradient_matches_finite_differences() {
    use cortiq_embryo::ops::hk_ref_dgamma;
    let d = HkDims {
        b: 2,
        t: 9,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let n_k = d.b * d.t * d.nh;
    let thq: Vec<f64> = to64(&lcg_vec(1, n_th)).iter().map(|x| x * 3.0).collect();
    let thk: Vec<f64> = to64(&lcg_vec(2, n_th)).iter().map(|x| x * 3.0).collect();
    let v = to64(&lcg_vec(3, n_v));
    let kappa: Vec<f64> = to64(&lcg_vec(4, n_k))
        .iter()
        .map(|x| 0.5 + 0.4 * x)
        .collect();
    let decay = to64(&hk_decay_grid(d.nh, d.nph, 2.0, 16.0));
    let dout = to64(&lcg_vec(5, n_v));
    let loss = |dec: &[f64]| -> f64 {
        let o = hk_ref_fwd(&d, &thq, &thk, &v, &kappa, dec);
        o.iter().zip(&dout).map(|(a, b)| a * b).sum()
    };
    let dg = hk_ref_dgamma(&d, &thq, &thk, &v, &kappa, &decay, &dout);
    let eps = 1e-6;
    let mut worst = 0.0f64;
    for i in 0..decay.len() {
        let mut dp = decay.clone();
        dp[i] += eps;
        let mut dm = decay.clone();
        dm[i] -= eps;
        let fd = (loss(&dp) - loss(&dm)) / (2.0 * eps);
        let err = (fd - dg[i]).abs() / (1.0 + fd.abs());
        worst = worst.max(err);
    }
    eprintln!("dgamma: worst rel err {worst:e}");
    assert!(worst < 1e-6);
}

#[test]
fn phase_delta_backward_matches_finite_differences() {
    use cortiq_embryo::ops::{phase_delta_ref_bwd, phase_delta_ref_fwd};
    let d = HkDims {
        b: 1,
        t: 5,
        nh: 1,
        nph: 3,
        dv: 3,
    };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let n_k = d.b * d.t * d.nh;
    let thq: Vec<f64> = to64(&lcg_vec(11, n_th)).iter().map(|x| x * 2.0).collect();
    let thk: Vec<f64> = to64(&lcg_vec(12, n_th)).iter().map(|x| x * 2.0).collect();
    let v = to64(&lcg_vec(13, n_v));
    let kappa: Vec<f64> = to64(&lcg_vec(14, n_k))
        .iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect();
    let decay = to64(&hk_decay_grid(d.nh, d.nph, 2.0, 16.0));
    let dout = to64(&lcg_vec(15, n_v));
    let loss = |q: &[f64], k: &[f64], vv: &[f64], kap: &[f64]| -> f64 {
        phase_delta_ref_fwd(&d, q, k, vv, kap, &decay)
            .iter()
            .zip(&dout)
            .map(|(a, b)| a * b)
            .sum()
    };
    let (gq, gk, gv, gkap) = phase_delta_ref_bwd(&d, &thq, &thk, &v, &kappa, &decay, &dout);
    let eps = 1e-6;
    let check = |name: &str, x: &[f64], g: &[f64], f: &dyn Fn(&[f64]) -> f64| {
        let mut worst: f64 = 0.0;
        for i in 0..x.len() {
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
            worst = worst.max((fd - g[i]).abs() / (1.0 + fd.abs()));
        }
        assert!(worst < 3e-6, "{name}: worst relative error {worst:e}");
    };
    check("phase_delta dtheta_q", &thq, &gq, &|x| {
        loss(x, &thk, &v, &kappa)
    });
    check("phase_delta dtheta_k", &thk, &gk, &|x| {
        loss(&thq, x, &v, &kappa)
    });
    check("phase_delta dv", &v, &gv, &|x| loss(&thq, &thk, x, &kappa));
    check("phase_delta dkappa", &kappa, &gkap, &|x| {
        loss(&thq, &thk, &v, x)
    });
}

#[test]
fn phase_delta_phase_pairs_are_unit_norm() {
    let nph = 8usize;
    let th: Vec<f64> = to64(&lcg_vec(99, nph));
    let scale = 1.0 / (nph as f64).sqrt();
    let norm2: f64 = th
        .iter()
        .map(|x| (scale * x.cos()).powi(2) + (scale * x.sin()).powi(2))
        .sum();
    assert!((norm2 - 1.0).abs() < 1e-12, "norm²={norm2}");
}

#[test]
fn phase_delta_is_parameter_neutral_and_round_trips() {
    use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};
    let base = EmbryoCfg::tiny();
    let mut delta = base.clone();
    delta.phase_delta = true;
    let l0 = Layout::new(&base);
    let l1 = Layout::new(&delta);
    assert_eq!(base.params(), delta.params());
    assert_eq!(l0.total, l1.total);
    assert_eq!(l0.names, l1.names);
    assert_eq!(init_params(&base, &l0, 1), init_params(&delta, &l1, 1));
    let encoded = serde_json::to_string(&delta).unwrap();
    let decoded: EmbryoCfg = serde_json::from_str(&encoded).unwrap();
    assert!(decoded.phase_delta);
    let legacy: EmbryoCfg = serde_json::from_str(&serde_json::to_string(&base).unwrap()).unwrap();
    assert!(!legacy.phase_delta);
    let path = std::env::temp_dir().join(format!("cmf-phase-delta-{}.ckpt", std::process::id()));
    let params = init_params(&delta, &l1, 3);
    cortiq_embryo::train::save_checkpoint(
        &path,
        &delta,
        17,
        &params,
        Some(&params),
        Some(&params),
        &[],
    )
    .unwrap();
    let ck = cortiq_embryo::train::load_checkpoint(&path).unwrap();
    assert_eq!(ck.step, 17);
    assert!(ck.cfg.phase_delta);
    assert_eq!(ck.params, params);
    assert_eq!(ck.m.as_deref(), Some(params.as_slice()));
    assert_eq!(ck.v.as_deref(), Some(params.as_slice()));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn phase_delta_continuation_matches_single_pass() {
    use cortiq_embryo::ops::{phase_delta_ref_fwd, phase_delta_ref_fwd_state};
    let d = HkDims {
        b: 1,
        t: 8,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let thq = to64(&lcg_vec(201, n_th));
    let thk = to64(&lcg_vec(202, n_th));
    let v = to64(&lcg_vec(203, n_v));
    let kap = to64(&lcg_vec(204, d.b * d.t * d.nh))
        .iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect::<Vec<_>>();
    let dec = to64(&hk_decay_grid(d.nh, d.nph, 2.0, 16.0));
    let full = phase_delta_ref_fwd(&d, &thq, &thk, &v, &kap, &dec);
    let mut state = vec![0.0; d.b * d.nh * d.p2() * d.dv];
    let continued = phase_delta_ref_fwd_state(&d, &thq, &thk, &v, &kap, &dec, &mut state);
    assert!(
        full.iter()
            .zip(continued)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
}

#[test]
fn phase_delta_continuation_matches_at_chunk_boundary_splits() {
    use cortiq_embryo::ops::{phase_delta_ref_fwd, phase_delta_step};
    let d = HkDims {
        b: 1,
        t: 129,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let thq = to64(&lcg_vec(211, n_th));
    let thk = to64(&lcg_vec(212, n_th));
    let v = to64(&lcg_vec(213, n_v));
    let kap = to64(&lcg_vec(214, d.b * d.t * d.nh))
        .into_iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect::<Vec<_>>();
    let dec = to64(&hk_decay_grid(d.nh, d.nph, 2.0, 16.0));
    let full = phase_delta_ref_fwd(&d, &thq, &thk, &v, &kap, &dec);
    // These are the historically fragile split points around the 64-token
    // replay boundary; each segment carries the exact recurrent state.
    for split in [1usize, 63, 64, 65, d.t - 1] {
        let mut state = vec![0.0f64; d.nh * d.p2() * d.dv];
        let mut got = vec![0.0f64; full.len()];
        for (lo, hi) in [(0, split), (split, d.t)] {
            for t in lo..hi {
                let row = t;
                for h in 0..d.nh {
                    let q0 = row * d.nh * d.nph + h * d.nph;
                    let v0 = row * d.nh * d.dv + h * d.dv;
                    let s0 = h * d.p2() * d.dv;
                    let out = phase_delta_step(
                        &mut state[s0..s0 + d.p2() * d.dv],
                        &thq[q0..q0 + d.nph],
                        &thk[q0..q0 + d.nph],
                        &v[v0..v0 + d.dv],
                        kap[row * d.nh + h],
                        &dec[h * d.p2()..(h + 1) * d.p2()],
                    );
                    got[row * d.nh * d.dv + h * d.dv..row * d.nh * d.dv + (h + 1) * d.dv]
                        .copy_from_slice(&out);
                }
            }
        }
        assert!(
            got.iter().zip(&full).all(|(a, b)| (a - b).abs() < 1e-12),
            "split {split} changed continuation outputs"
        );
    }
}

#[test]
fn phase_delta_beta_one_self_key_is_exact_overwrite() {
    use cortiq_embryo::ops::phase_delta_step;
    let nph = 2usize;
    let dv = 3usize;
    let th = [0.37f64, -1.1];
    let mut state = vec![0.0; 2 * nph * dv];
    let old = [0.2, -0.4, 0.7];
    let new = [-0.6, 0.8, 0.1];
    let decay = vec![1.0; 2 * nph];
    let _ = phase_delta_step(&mut state, &th, &th, &old, 1.0, &decay);
    let out = phase_delta_step(&mut state, &th, &th, &new, 1.0, &decay);
    for (got, want) in out.iter().zip(new) {
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");
    }
}
