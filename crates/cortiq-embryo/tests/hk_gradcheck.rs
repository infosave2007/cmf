//! hybrid_k CPU oracle: the closed-form analytic backward against central
//! finite differences of the literal recurrence (f64, tight tolerance).
use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd, hk_ref_fwd, lcg_vec};

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

#[test]
fn hk_closed_form_backward_matches_finite_differences() {
    let d = HkDims { b: 2, t: 9, nh: 2, nph: 3, dv: 4 };
    let n_th = d.b * d.t * d.nh * d.nph;
    let n_v = d.b * d.t * d.nh * d.dv;
    let n_k = d.b * d.t * d.nh;
    let thq: Vec<f64> = to64(&lcg_vec(1, n_th)).iter().map(|x| x * 3.0).collect();
    let thk: Vec<f64> = to64(&lcg_vec(2, n_th)).iter().map(|x| x * 3.0).collect();
    let v = to64(&lcg_vec(3, n_v));
    let kappa: Vec<f64> = to64(&lcg_vec(4, n_k)).iter().map(|x| 0.5 + 0.4 * x).collect();
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
