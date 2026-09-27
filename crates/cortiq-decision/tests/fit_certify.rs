//! The fitter `gram-eigh-f64-v1` against a dense one-sided Jacobi SVD, and the
//! certification pieces against scipy 1.13.1 / numpy 1.26.4 (spec §3.4, §3.6,
//! §6.1): `fminbound` on three known functions and on the temperature objective,
//! `BetaInv` on the 3×14 Clopper–Pearson bounds of the shipped v3 gate grids,
//! the f32 θ rule, and the certification bookkeeping.
use cortiq_decision::certify::{
    self, Calibration, GridRow, THRESHOLDS, alpha, certify, choose, fit_temperature, halves,
    novelty_theta,
};
use cortiq_decision::fit::{
    self, DEFAULT_K, EIG_DROP, ERR_STD_FLOOR, TaskFit64, fit_task, fit_task_f64,
};
use cortiq_decision::resonance::{ErrStats, decide, reference_error};
use cortiq_decision::specfn::{beta_inc, beta_inv, clopper_pearson_lower, fminbound};
use sha2::{Digest, Sha256};

// ------------------------------------------------------------------ data

/// xorshift64*: deterministic test data.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Orthonormalise the `k` vectors of length `d` in place (classical GS, twice).
fn orthonormalise(v: &mut [f64], k: usize, d: usize) {
    for _ in 0..2 {
        for j in 0..k {
            for l in 0..j {
                let c = dot(&v[j * d..(j + 1) * d], &v[l * d..(l + 1) * d]);
                for i in 0..d {
                    v[j * d + i] -= c * v[l * d + i];
                }
            }
            let n = dot(&v[j * d..(j + 1) * d], &v[j * d..(j + 1) * d]).sqrt();
            for x in &mut v[j * d..(j + 1) * d] {
                *x /= n;
            }
        }
    }
}

/// `n` rows of dimension `d` with the centred part `W·diag(σ)·Uᵀ` (W orthogonal
/// to the ones vector, so the rows' own mean is `μ`), rounded to f32.
fn synthetic(rng: &mut Rng, n: usize, d: usize, sigma: &[f64]) -> Vec<f32> {
    let r = sigma.len();
    // U: r orthonormal directions in R^d.
    let mut u: Vec<f64> = (0..r * d).map(|_| rng.f()).collect();
    orthonormalise(&mut u, r, d);
    // W: r orthonormal columns in R^n orthogonal to the ones vector (stored as r rows of length n).
    let mut w: Vec<f64> = (0..r * n).map(|_| rng.f()).collect();
    for j in 0..r {
        let m = w[j * n..(j + 1) * n].iter().sum::<f64>() / n as f64;
        for x in &mut w[j * n..(j + 1) * n] {
            *x -= m;
        }
    }
    orthonormalise(&mut w, r, n);
    let mu: Vec<f64> = (0..d).map(|_| rng.f() * 0.3).collect();
    let mut x = vec![0.0f32; n * d];
    for i in 0..n {
        for c in 0..d {
            let mut v = mu[c];
            for j in 0..r {
                v += w[j * n + i] * sigma[j] * u[j * d + c];
            }
            x[i * d + c] = v as f32;
        }
    }
    x
}

/// Right singular vectors of the centred rows by one-sided (Hestenes) Jacobi on
/// the rows themselves: rotate pairs of rows until all are orthogonal; the rows
/// are then `σ_j u_j`. Returns (σ² descending, unit vectors) with the fitter's
/// sign rule applied.
fn jacobi_svd(rows: &[f32], n: usize, d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut mean = vec![0.0f64; d];
    for r in rows.chunks_exact(d) {
        for (m, &v) in mean.iter_mut().zip(r) {
            *m += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut w: Vec<Vec<f64>> = rows
        .chunks_exact(d)
        .map(|r| r.iter().zip(&mean).map(|(&v, m)| v as f64 - m).collect())
        .collect();
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let a = dot(&w[p], &w[p]);
                let b = dot(&w[q], &w[q]);
                let g = dot(&w[p], &w[q]);
                if g == 0.0 || g.abs() <= 1e-15 * (a * b).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (b - a) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                let (lo, hi) = w.split_at_mut(q);
                for (xp, xq) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                    let (x, y) = (*xp, *xq);
                    *xp = c * x - s * y;
                    *xq = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut pairs: Vec<(f64, Vec<f64>)> = w
        .into_iter()
        .map(|v| {
            let s2 = dot(&v, &v);
            let nrm = s2.sqrt();
            let mut u: Vec<f64> = v
                .iter()
                .map(|x| if nrm > 0.0 { x / nrm } else { 0.0 })
                .collect();
            fit::apply_sign_rule(&mut u);
            (s2, u)
        })
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    pairs.into_iter().unzip()
}

/// Largest principal angle (upper bound via the Frobenius norm of the residual)
/// between span(a) and span(b), both orthonormal row sets.
fn max_angle(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut fro = 0.0f64;
    for v in a {
        let mut r = v.clone();
        for q in b {
            let c = dot(&r, q);
            for (x, y) in r.iter_mut().zip(q) {
                *x -= c * y;
            }
        }
        fro += dot(&r, &r);
    }
    fro.sqrt().min(1.0).asin()
}

fn rows_of(f: &TaskFit64) -> Vec<Vec<f64>> {
    (0..f.k).map(|j| f.row(j).to_vec()).collect()
}

// ------------------------------------------------------------------ fitter

#[test]
fn fitter_matches_dense_jacobi_svd() {
    let mut rng = Rng(0x5eed);
    // (n, d, K, singular values of the centred part)
    let geometric = |r: usize| {
        (0..r)
            .map(|j| 2.0 * 0.8f64.powi(j as i32))
            .collect::<Vec<_>>()
    };
    let cases: Vec<(usize, usize, usize, Vec<f64>)> = vec![
        (40, 60, 16, geometric(39)),
        (10, 50, 16, geometric(9)),
        (30, 20, 16, geometric(20)),
        (25, 40, 16, geometric(5)),
        (64, 96, 8, geometric(40)),
        (3, 7, 16, geometric(2)),
    ];
    for (n, d, k_max, sigma) in cases {
        let rows = synthetic(&mut rng, n, d, &sigma);
        let f = fit_task_f64(&rows, d, k_max).unwrap();
        let (s2, u) = jacobi_svd(&rows, n, d);
        let kept = s2.iter().filter(|&&l| l / n as f64 > EIG_DROP).count();
        let want_k = k_max.min(n - 1).min(kept);
        assert_eq!(f.k, want_k, "n={n} d={d}");
        assert_eq!(f.k, k_max.min(sigma.len()), "the true rank bounds k");
        let angle = max_angle(&rows_of(&f), &u[..f.k]);
        assert!(angle <= 1e-9, "n={n} d={d}: principal angle {angle:e}");
        // With distinct singular values every vector matches, sign rule included.
        for (j, uj) in u.iter().enumerate().take(f.k) {
            let diff = f
                .row(j)
                .iter()
                .zip(uj)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(diff <= 1e-9, "n={n} d={d} vector {j}: {diff:e}");
            let (imax, _) = f.row(j).iter().enumerate().fold((0, -1.0f64), |b, (i, v)| {
                if v.abs() > b.1 { (i, v.abs()) } else { b }
            });
            assert!(f.row(j)[imax] > 0.0, "sign rule");
        }
        // Eigenvalues λ/n against σ²/n.
        for (j, &sj) in s2.iter().enumerate().take(f.k) {
            let rel = (f.eigenvalues[j] - sj / n as f64).abs() / (sj / n as f64);
            assert!(rel <= 1e-9, "eigenvalue {j}: {rel:e}");
        }
        // Orthonormal f64 basis.
        for a in 0..f.k {
            for b in 0..f.k {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot(f.row(a), f.row(b)) - want).abs() < 1e-13);
            }
        }
        // Error statistics: population statistics of the rows' own f64 errors.
        let errs: Vec<f64> = rows
            .chunks_exact(d)
            .map(|r| fit::error_f64(r, &f.mean, &f.basis))
            .collect();
        let m = errs.iter().sum::<f64>() / n as f64;
        let s = (errs.iter().map(|e| (e - m) * (e - m)).sum::<f64>() / n as f64).sqrt();
        assert!((f.err_mean - m).abs() <= 1e-15 + 1e-12 * m.abs());
        assert!((f.err_std - s.max(ERR_STD_FLOOR)).abs() <= 1e-15 + 1e-12 * s);
        // The f32 topology is the rounded f64 fit.
        let t = fit_task(&rows, d, k_max).unwrap();
        assert_eq!(t.k, f.k);
        assert_eq!(t.n_train, n);
        assert!(
            t.topology
                .mean
                .iter()
                .zip(&f.mean)
                .all(|(&a, &b)| a == b as f32)
        );
        assert!(
            t.topology
                .basis
                .iter()
                .zip(&f.basis)
                .all(|(&a, &b)| a == b as f32)
        );
        assert_eq!(t.stats(), ErrStats::from_f64(f.err_mean, f.err_std));
    }
}

#[test]
fn fitter_rank_rules() {
    let mut rng = Rng(11);
    // k = min(K, n−1): two rows give one direction; K = 0 gives none.
    let rows = synthetic(&mut rng, 2, 9, &[1.0]);
    assert_eq!(fit_task_f64(&rows, 9, 16).unwrap().k, 1);
    assert_eq!(fit_task_f64(&rows, 9, 0).unwrap().k, 0);
    // Duplicate rows: every eigenvalue is dropped, E = ‖x − μ‖² = 0.
    let one: Vec<f32> = (0..9).map(|_| rng.f() as f32).collect();
    let dup: Vec<f32> = one.iter().chain(&one).chain(&one).copied().collect();
    let f = fit_task_f64(&dup, 9, 16).unwrap();
    assert_eq!((f.k, f.err_mean, f.err_std), (0, 0.0, ERR_STD_FLOOR));
    assert_eq!(f.eigenvalues.len(), 3);
    // Deterministic: two fits give the same bits.
    let rows = synthetic(
        &mut rng,
        33,
        45,
        &(0..20).map(|j| 1.0 / (1.0 + j as f64)).collect::<Vec<_>>(),
    );
    let a = fit_task(&rows, 45, DEFAULT_K).unwrap();
    let b = fit_task(&rows, 45, DEFAULT_K).unwrap();
    assert_eq!(a, b);
    // Invalid inputs.
    assert!(fit_task(&[], 3, 16).is_err());
    assert!(fit_task(&[1.0, 2.0], 3, 16).is_err());
    assert!(fit_task(&[f32::NAN, 0.0, 0.0], 3, 16).is_err());
    assert!(fit_task(&[1.0], 0, 16).is_err());
}

#[test]
fn fitted_topology_scores_its_own_rows_like_the_f64_statistics() {
    // The runtime f32 error of a training row stays close to the f64 error the
    // statistics were computed from.
    let mut rng = Rng(21);
    let d = 80;
    let rows = synthetic(
        &mut rng,
        50,
        d,
        &(0..30).map(|j| 0.9f64.powi(j)).collect::<Vec<_>>(),
    );
    let f64fit = fit_task_f64(&rows, d, DEFAULT_K).unwrap();
    let t = f64fit.to_f32();
    for r in rows.chunks_exact(d) {
        let e32 = reference_error(r, &t.topology.mean, &t.topology.basis) as f64;
        let e64 = fit::error_f64(r, &f64fit.mean, &f64fit.basis);
        assert!((e32 - e64).abs() <= 1e-5 * (1.0 + e64), "{e32} vs {e64}");
    }
}

// ------------------------------------------------------------------ fminbound

#[test]
fn fminbound_matches_scipy_on_three_functions() {
    // scipy 1.13.1: minimize_scalar(f, bounds=b, method='bounded') -> (x, fun, nfev).
    type Case = (&'static str, fn(f64) -> f64, (f64, f64), f64, f64, usize);
    let cases: [Case; 3] = [
        (
            "parabola",
            |x| (x - 0.3) * (x - 0.3) + 1.0,
            (-1.0, 2.0),
            0.29999999999999993,
            1.0,
            6,
        ),
        (
            "quartic",
            |x| x.powi(4) - 3.0 * x.powi(3) + 2.0,
            (0.0, 5.0),
            2.2499994695531518,
            -6.542968749997151,
            12,
        ),
        (
            "exp_lin",
            |x| x.exp() - 2.0 * x,
            (-3.0, 3.0),
            0.6931478936143736,
            0.6137056388806177,
            10,
        ),
    ];
    for (name, f, (a, b), x, fun, nfev) in cases {
        let r = fminbound(f, a, b, 1e-5, 500).unwrap();
        assert!(r.success(), "{name}");
        assert!((r.x - x).abs() <= 1e-9, "{name}: x {} vs scipy {x}", r.x);
        assert!(
            (r.fun - fun).abs() <= 1e-9,
            "{name}: fun {} vs scipy {fun}",
            r.fun
        );
        assert_eq!(r.nfev, nfev, "{name}: evaluations");
    }
    // maxiter reached is reported.
    let r = fminbound(|x| (x - 0.3) * (x - 0.3), -1.0, 2.0, 1e-12, 3).unwrap();
    assert_eq!((r.status, r.nfev), (1, 3));
}

#[test]
fn temperature_matches_scipy_on_a_fixed_matrix() {
    // E[i][j] = f32(((7i + 3j) mod 11)·0.05 + (0 if j == i mod 4 else 0.3)),
    // truth = i mod 4; scipy: x = -2.1996557369987824, nll 0.7531379564995294,
    // 11 evaluations, T = f32(exp(x)) bits 1038287045 (0.1108413115143776).
    let (n, k) = (12usize, 4usize);
    let e: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..k).map(move |j| {
                (((i * 7 + j * 3) % 11) as f64 * 0.05 + if j == i % 4 { 0.0 } else { 0.3 }) as f32
            })
        })
        .collect();
    let rows: Vec<usize> = (0..n).collect();
    let truth: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let (t, fit) = fit_temperature(&e, k, &rows, &truth).unwrap();
    assert!(
        (fit.x - -2.1996557369987824).abs() <= 1e-9,
        "log T {}",
        fit.x
    );
    assert!(
        (fit.fun - 0.7531379564995294).abs() <= 1e-9,
        "nll {}",
        fit.fun
    );
    assert_eq!(fit.nfev, 11);
    assert_eq!(t.to_bits(), 1038287045, "T {t}");
    assert!(fit_temperature(&e, k, &[], &[]).is_err());
}

// ------------------------------------------------------------------ BetaInv

/// (correct, accepted, scipy beta.ppf(0.05/14, k, n−k+1)) of the shipped v3 gate
/// grids (`{banking77,clinc150,massive}-product/training.json`), θ on (42 rows)
/// and θ off (42 rows).
const V3_GRID_LB: [(u64, u64, f64); 84] = [
    // banking77, theta on
    (663, 698, 0.9233984601904901),
    (663, 698, 0.9233984601904901),
    (658, 692, 0.9244777161767814),
    (653, 678, 0.9391250177361361),
    (646, 665, 0.949299437841493),
    (635, 649, 0.9580843339464022),
    (613, 622, 0.9672340635203133),
    (592, 598, 0.9731808738462493),
    (574, 579, 0.9749191117143761),
    (552, 557, 0.9739370691083346),
    (489, 494, 0.9706457231695658),
    (424, 425, 0.981768285240075),
    (371, 372, 0.9791944798647167),
    (257, 258, 0.97012216793685),
    // clinc150, theta on
    (1406, 1414, 0.9865221333976131),
    (1406, 1414, 0.9865221333976131),
    (1406, 1414, 0.9865221333976131),
    (1403, 1411, 0.9864935906111021),
    (1401, 1408, 0.9874983380708804),
    (1394, 1401, 0.9874361124750217),
    (1381, 1388, 0.987318891920503),
    (1352, 1358, 0.988133665682933),
    (1332, 1337, 0.989087046033599),
    (1297, 1301, 0.9899933082998513),
    (1225, 1229, 0.9894092362933161),
    (1115, 1118, 0.9898222809041067),
    (1021, 1022, 0.9923827889784759),
    (778, 778, 0.9927835046989394),
    // massive, theta on
    (983, 1103, 0.863673141015187),
    (964, 1064, 0.8794679516472023),
    (930, 1007, 0.898273402495037),
    (898, 955, 0.9166866847481182),
    (867, 917, 0.9221318399228111),
    (829, 866, 0.9353186479827605),
    (779, 809, 0.9412657272722162),
    (710, 727, 0.9571234069561823),
    (666, 682, 0.9562320394640683),
    (607, 622, 0.954162777573569),
    (526, 537, 0.956951807231019),
    (412, 418, 0.9617725919586332),
    (334, 338, 0.9618637509282513),
    (204, 206, 0.953836102778628),
    // banking77, theta off
    (684, 749, 0.8819497998750533),
    (675, 721, 0.9077588811957802),
    (665, 704, 0.9172678391641471),
    (655, 681, 0.9375725684844107),
    (646, 665, 0.949299437841493),
    (635, 649, 0.9580843339464022),
    (613, 622, 0.9672340635203133),
    (592, 598, 0.9731808738462493),
    (574, 579, 0.9749191117143761),
    (552, 557, 0.9739370691083346),
    (489, 494, 0.9706457231695658),
    (424, 425, 0.981768285240075),
    (371, 372, 0.9791944798647167),
    (257, 258, 0.97012216793685),
    // clinc150, theta off
    (1463, 1500, 0.9624752606631356),
    (1457, 1483, 0.9711362270513583),
    (1449, 1469, 0.9760280367354162),
    (1431, 1443, 0.982912065565027),
    (1415, 1424, 0.9856125359473743),
    (1401, 1408, 0.9874983380708804),
    (1382, 1389, 0.9873279864863413),
    (1352, 1358, 0.988133665682933),
    (1332, 1337, 0.989087046033599),
    (1297, 1301, 0.9899933082998513),
    (1225, 1229, 0.9894092362933161),
    (1115, 1118, 0.9898222809041067),
    (1021, 1022, 0.9923827889784759),
    (778, 778, 0.9927835046989394),
    // massive, theta off
    (997, 1144, 0.842762433239698),
    (967, 1069, 0.877949982743358),
    (930, 1007, 0.898273402495037),
    (898, 955, 0.9166866847481182),
    (867, 917, 0.9221318399228111),
    (829, 866, 0.9353186479827605),
    (779, 809, 0.9412657272722162),
    (710, 727, 0.9571234069561823),
    (666, 682, 0.9562320394640683),
    (607, 622, 0.954162777573569),
    (526, 537, 0.956951807231019),
    (412, 418, 0.9617725919586332),
    (334, 338, 0.9618637509282513),
    (204, 206, 0.953836102778628),
];

/// Extra scipy values: small n, k = n, k = 1, large n.
const EXTRA_LB: [(u64, u64, f64); 10] = [
    (1, 1, 0.0035714285714285718),
    (5, 5, 0.3240174454577801),
    (100, 100, 0.9452102437906554),
    (100, 120, 0.7239001685985186),
    (1, 50, 7.155386685762436e-05),
    (50, 51, 0.8566756450460267),
    (778, 778, 0.9927835046989394),
    (1406, 1414, 0.9865221333976131),
    (3, 7, 0.049117997549016565),
    (2000, 2100, 0.9384884681367531),
];

#[test]
fn beta_inv_matches_scipy_on_the_shipped_gate_grids() {
    assert_eq!(alpha(), 0.05 / 14.0);
    let mut worst = 0.0f64;
    for &(k, n, lb) in V3_GRID_LB.iter().chain(&EXTRA_LB) {
        let got = clopper_pearson_lower(k, n, alpha());
        let d = (got - lb).abs();
        worst = worst.max(d);
        assert!(d <= 1e-9, "k={k} n={n}: {got} vs scipy {lb} (Δ {d:e})");
        // The quantile inverts the distribution function.
        assert!((beta_inc(k, n - k + 1, got) - alpha()).abs() <= 1e-12);
    }
    eprintln!(
        "max |BetaInv − scipy| over {} values: {worst:e}",
        V3_GRID_LB.len() + EXTRA_LB.len()
    );
    assert_eq!(clopper_pearson_lower(0, 10, alpha()), 0.0);
    assert_eq!(beta_inv(0.0, 3, 4), 0.0);
    assert_eq!(beta_inv(1.0, 3, 4), 1.0);
}

// ------------------------------------------------------------------ θ rule

fn theta_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 + 13) % 10007) as f32 / 10007.0f32)
        .collect()
}

#[test]
fn theta_rule_matches_numpy_f32() {
    // numpy evaluate_v3/common.py::novelty_theta on v_i = f32((7919 i + 13) mod 10007) / f32(10007).
    let cases: [(usize, u32); 9] = [
        (1, 985096625),
        (2, 1061876057),
        (3, 1061876057),
        (11, 1064650744),
        (21, 1063924799),
        (41, 1064521650),
        (749, 1064521650),
        (1144, 1064519973),
        (1500, 1064504884),
    ];
    for (n, bits) in cases {
        let th = novelty_theta(&theta_values(n)).unwrap();
        assert_eq!(th.to_bits(), bits, "n={n}: θ {th}");
    }
    // Cap at 0.999f32 and the empty set.
    assert_eq!(novelty_theta(&[0.9995; 5]).unwrap(), 0.999f32);
    assert!(novelty_theta(&[]).is_err());
    assert!(novelty_theta(&[f32::NAN]).is_err());
}

// ------------------------------------------------------------------ certification

fn sha(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

/// A synthetic calibration set: `tasks` topologies, rows whose truth has a
/// small error with probability `p_clean`.
fn synthetic_calibration(
    rng: &mut Rng,
    rows: usize,
    tasks: usize,
    p_clean: f64,
) -> (Vec<f32>, Vec<Option<usize>>, Vec<String>) {
    let mut e = Vec::with_capacity(rows * tasks);
    let mut truth = Vec::with_capacity(rows);
    let mut shas = Vec::with_capacity(rows);
    for i in 0..rows {
        let y = (rng.next() % tasks as u64) as usize;
        let clean = (rng.f() + 1.0) / 2.0 < p_clean;
        for j in 0..tasks {
            let base = 0.3 + 0.2 * (rng.f() + 1.0);
            let v = if j == y && clean {
                0.05 + 0.1 * (rng.f() + 1.0)
            } else {
                base
            };
            e.push(v as f32);
        }
        truth.push(if i % 97 == 5 { None } else { Some(y) });
        shas.push(sha(&format!("row {i}")));
    }
    (e, truth, shas)
}

#[test]
fn certify_bookkeeping_is_consistent() {
    let mut rng = Rng(314);
    let (rows, tasks) = (1500usize, 12usize);
    let (e, truth, shas) = synthetic_calibration(&mut rng, rows, tasks, 0.97);
    let stats: Vec<ErrStats> = (0..tasks)
        .map(|j| ErrStats {
            err_mean: 0.1 + 0.001 * j as f32,
            err_std: 0.03,
        })
        .collect();
    let (even, odd) = halves(&shas);
    assert_eq!((even.len(), odd.len()), (750, 750));
    let cal = Calibration {
        errors: &e,
        tasks,
        stats: &stats,
        truth: &truth,
        even: &even,
        odd: &odd,
    };
    let c = certify(&cal).unwrap();

    // T: the even rows with an active truth only.
    let (t_rows, t_truth): (Vec<usize>, Vec<usize>) = even
        .iter()
        .filter_map(|&i| truth[i].map(|y| (i, y)))
        .unzip();
    assert!(
        t_rows.len() < even.len(),
        "the fixture has rows without an active task"
    );
    let (t, fit) = fit_temperature(&e, tasks, &t_rows, &t_truth).unwrap();
    assert_eq!(
        (c.temperature.to_bits(), c.temperature_fit),
        (t.to_bits(), fit)
    );
    assert_eq!(c.even_t_n, t_rows.len());
    assert!(c.temperature >= 1e-3 && c.temperature <= 1.0);

    // θ from the even half's runtime novelties at T.
    let gates = certify::row_gates(&cal, c.temperature).unwrap();
    let nov: Vec<f32> = even.iter().map(|&i| gates[i].novelty).collect();
    assert_eq!(
        c.novelty_theta.to_bits(),
        novelty_theta(&nov).unwrap().to_bits()
    );

    // Grid rows recounted by hand; rows without an active truth are wrong.
    for (row, off) in c.grid.iter().zip(&c.grid_theta_off) {
        let tf = row.threshold as f32;
        let (mut acc, mut ok, mut rej, mut acc_off, mut ok_off) = (0, 0, 0, 0, 0);
        for &i in &odd {
            let d = decide(&e[i * tasks..(i + 1) * tasks], &stats, c.temperature).unwrap();
            let good = truth[i].is_some() && d.winner == truth[i];
            if d.p_top >= tf {
                acc_off += 1;
                ok_off += usize::from(good);
                if d.novelty <= c.novelty_theta {
                    acc += 1;
                    ok += usize::from(good);
                } else {
                    rej += 1;
                }
            }
        }
        assert_eq!(
            (row.accepted, row.correct, row.novelty_rejected),
            (acc, ok, rej)
        );
        assert_eq!(
            (off.accepted, off.correct, off.novelty_rejected),
            (acc_off, ok_off, 0)
        );
        assert_eq!(
            row.lower_bound,
            clopper_pearson_lower(ok as u64, acc as u64, alpha())
        );
    }
    assert_eq!(c.grid.len(), THRESHOLDS.len());
    assert_eq!(c.chosen, choose(&c.grid));
    assert!(c.certified, "a clean fixture certifies: {:?}", c.grid);
    let chosen = c.chosen.unwrap();
    assert_eq!(c.tau, chosen.threshold as f32);
    assert_eq!(
        (c.odd_accepted, c.odd_correct),
        (chosen.accepted, chosen.correct)
    );
    assert!(chosen.accepted >= certify::MIN_ACCEPTED && chosen.lower_bound >= certify::TARGET);

    // Errors unrelated to the truth: no qualifying threshold, τ = 0, not certified.
    let (e2, truth2, shas2) = synthetic_calibration(&mut rng, 600, tasks, 0.0);
    let (even2, odd2) = halves(&shas2);
    let cal2 = Calibration {
        errors: &e2,
        tasks,
        stats: &stats,
        truth: &truth2,
        even: &even2,
        odd: &odd2,
    };
    let c2 = certify(&cal2).unwrap();
    assert!(!c2.certified && c2.tau == 0.0 && c2.chosen.is_none());
    assert_eq!((c2.odd_accepted, c2.odd_correct), (0, 0));

    // Refusals: overlapping halves, no even row, wrong shapes.
    let bad = Calibration { odd: &even, ..cal };
    assert!(certify(&bad).is_err());
    let bad = Calibration { even: &[], ..cal };
    assert!(certify(&bad).is_err());
    let bad = Calibration { tasks: 11, ..cal };
    assert!(certify(&bad).is_err());
}

#[test]
fn choose_rule_and_grid_shape() {
    let row = |t: f64, a: usize, lb: f64| GridRow {
        threshold: t,
        accepted: a,
        correct: a,
        lower_bound: lb,
        novelty_rejected: 0,
    };
    // BANKING77-like: 0.75 misses the target by 5e-4, 0.8 is chosen.
    let grid = [
        row(0.7, 678, 0.939),
        row(0.75, 665, 0.9493),
        row(0.8, 649, 0.958),
        row(0.85, 622, 0.967),
    ];
    assert_eq!(choose(&grid).unwrap().threshold, 0.8);
    // Equal accepted counts: the smaller threshold.
    let grid = [row(0.0, 1414, 0.9865), row(0.5, 1414, 0.9865)];
    assert_eq!(choose(&grid).unwrap().threshold, 0.0);
    assert_eq!(THRESHOLDS[0], 0.0);
    assert_eq!(THRESHOLDS[13], 0.999);
}
