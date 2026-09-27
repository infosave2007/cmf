//! Local gates F1–F3 (spec §6.2): the v4 fitter and certifier on the stored
//! Python features of the shipped v3 PH skills. `#[ignore]`: it needs
//! `CORTIQ_DECISION_V3_DIR` (= `artifacts/decision-v3-20260926`) and the split
//! files the v3 builds recorded, and it should run optimised:
//!
//! ```text
//! CMF_GPU=0 CORTIQ_DECISION_V3_DIR=$CMFPUBLIC/artifacts/decision-v3-20260926 \
//!   cargo test --release -p cortiq-decision --test parity_v3 -- --ignored --nocapture
//! ```
//!
//! The signal is rebuilt exactly as the v3 fit saw it (`[l2f32(φ_P) ; 0.5·φ_H]`
//! from `features-product` and `hash-features`). Only train, dev and
//! calibration are read; no test split is opened.
//!
//! * control: the v3 topologies (`{ds}-product/cortiq.cmf`) through the v4
//!   runtime and certifier reproduce the shipped dev count, T, θ, τ and odd half;
//! * F1: dev exactly 1396/1498, 2888/2998, 1755/2025 with 0 winner flips
//!   against the v3 topologies;
//! * F2: per task the largest principal angle against the v3 (numpy SVD)
//!   basis ≤ 1e-6 rad and the relative E difference on dev ≤ 1e-6;
//! * F3: T within 1 f32 ulp of v3, θ within 2 ulp, τ equal, odd half
//!   649/635, 1414/1406, 727/710.
//!
//! It also reports (without asserting a time — the Mac is shared) the
//! single-thread resonance time per dev row of the v4 packed scorer against the
//! v3 runtime's scorer in the same process, and asserts that both decide
//! identically.
//!
//! A one-line JSON summary is printed after `PARITY_V3_JSON `; with
//! `CORTIQ_DECISION_PARITY_OUT=<file>` it is also written there.
mod common;

use common::{V3Skill, V3Split, read_v3_skill, read_v3_split, v3_dir};
use cortiq_decision::certify::{Calibration, Certification, certify, halves};
use cortiq_decision::fit::{DEFAULT_K, TaskFit64, fit_task_f64};
use cortiq_decision::packed::Packed;
use cortiq_decision::resonance::{ErrStats, TaskView, decide};
use serde_json::json;
use std::time::Instant;

/// (dataset, dev correct, dev n, odd accepted, odd correct) — spec §6.2 F1/F3.
const EXPECT: [(&str, usize, usize, usize, usize); 3] = [
    ("banking77", 1396, 1498, 649, 635),
    ("clinc150", 2888, 2998, 1414, 1406),
    ("massive", 1755, 2025, 727, 710),
];
const MAX_ANGLE: f64 = 1e-6;
const MAX_REL_E: f64 = 1e-6;

/// The v3 runtime's scorer (tools/cortiq-decision/src/packed.rs: four lanes,
/// three sweeps per basis row), verbatim apart from taking task views: the
/// same-process timing baseline for the v4 packed scorer (spec §6.6).
mod v3packed {
    #![allow(clippy::needless_range_loop)]
    use cortiq_decision::resonance::TaskView;
    struct Block {
        mean: Vec<[f32; 4]>,
        basis: Vec<[f32; 4]>,
        rank: usize,
    }
    pub struct Packed {
        blocks: Vec<Block>,
        dim: usize,
        tasks: usize,
        residual: Vec<[f32; 4]>,
    }
    impl Packed {
        pub fn new(tasks: &[TaskView]) -> Self {
            let d = tasks[0].dim();
            let mut blocks = vec![];
            for ts in tasks.chunks(4) {
                let rank = ts.iter().map(|t| t.rank()).max().unwrap();
                let mut mean = vec![[0.; 4]; d];
                let mut basis = vec![[0.; 4]; d * rank];
                for (lane, t) in ts.iter().enumerate() {
                    for i in 0..d {
                        mean[i][lane] = t.mean[i];
                    }
                    for k in 0..t.rank() {
                        let b = t.row(k);
                        for i in 0..d {
                            basis[k * d + i][lane] = b[i];
                        }
                    }
                }
                blocks.push(Block { mean, basis, rank });
            }
            Self {
                blocks,
                dim: d,
                tasks: tasks.len(),
                residual: vec![[0.; 4]; d],
            }
        }
        pub fn errors_into(&mut self, x: &[f32], out: &mut [f32]) {
            assert!(
                x.len() == self.dim && x.iter().all(|v| v.is_finite()) && out.len() == self.tasks
            );
            for (bi, b) in self.blocks.iter().enumerate() {
                let r = &mut self.residual;
                for i in 0..self.dim {
                    for l in 0..4 {
                        r[i][l] = x[i] - b.mean[i][l];
                    }
                }
                for k in 0..b.rank {
                    let w = &b.basis[k * self.dim..(k + 1) * self.dim];
                    let mut c = [0.; 4];
                    for i in 0..self.dim {
                        for l in 0..4 {
                            c[l] += r[i][l] * w[i][l];
                        }
                    }
                    for i in 0..self.dim {
                        for l in 0..4 {
                            r[i][l] -= c[l] * w[i][l];
                        }
                    }
                }
                let mut e = [0.; 4];
                for row in r.iter() {
                    for l in 0..4 {
                        e[l] += row[l] * row[l];
                    }
                }
                for l in 0..4 {
                    if bi * 4 + l < self.tasks {
                        out[bi * 4 + l] = e[l];
                    }
                }
            }
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Single-thread resonance time per row (errors of every task + the decision)
/// of the v4 packed scorer and of the v3 runtime's scorer on the same rows,
/// interleaved row by row; both must give the same bits.
fn resonance_timing(
    views: &[TaskView],
    stats: &[ErrStats],
    x: &[f32],
    dim: usize,
) -> serde_json::Value {
    let v4 = Packed::new(views).unwrap();
    let mut v3 = v3packed::Packed::new(views);
    let t = views.len();
    let (mut o4, mut o3) = (vec![0.0f32; t], vec![0.0f32; t]);
    let mut scratch = v4.scratch();
    let (mut us4, mut us3) = (Vec::new(), Vec::new());
    for (i, xr) in x.chunks_exact(dim).enumerate() {
        let a = Instant::now();
        v4.errors_with(xr, &mut o4, &mut scratch).unwrap();
        let d4 = decide(&o4, stats, 0.03).unwrap();
        let t4 = a.elapsed().as_secs_f64() * 1e6;
        let b = Instant::now();
        v3.errors_into(xr, &mut o3);
        let d3 = decide(&o3, stats, 0.03).unwrap();
        let t3 = b.elapsed().as_secs_f64() * 1e6;
        assert_eq!(d4, d3, "row {i}: v4 and v3 scorers disagree");
        if i >= 50 {
            us4.push(t4);
            us3.push(t3);
        }
    }
    let (p50_4, p50_3) = (percentile(&mut us4, 0.5), percentile(&mut us3, 0.5));
    json!({"rows_timed": us4.len(), "warmup_rows": 50, "threads": 1,
           "v4_packed_us": {"p50": p50_4, "p95": percentile(&mut us4, 0.95), "p99": percentile(&mut us4, 0.99)},
           "v3_packed_us": {"p50": p50_3, "p95": percentile(&mut us3, 0.95), "p99": percentile(&mut us3, 0.99)},
           "p50_ratio_v4_over_v3": p50_4 / p50_3, "bit_equal": true})
}

fn threads() -> usize {
    std::env::var("CORTIQ_TEST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(1)
}

fn ulps(a: f32, b: f32) -> u32 {
    assert!(a >= 0.0 && b >= 0.0);
    a.to_bits().abs_diff(b.to_bits())
}

/// `rows × tasks` errors of the packed topologies, rows split across threads.
fn error_matrix(p: &Packed, x: &[f32], dim: usize) -> Vec<f32> {
    let n = x.len() / dim;
    let t = p.tasks();
    let mut out = vec![0.0f32; n * t];
    let per = n.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (xs, os) in x.chunks(per * dim).zip(out.chunks_mut(per * t)) {
            s.spawn(move || {
                let mut scratch = p.scratch();
                for (xr, or) in xs.chunks_exact(dim).zip(os.chunks_exact_mut(t)) {
                    p.errors_with(xr, or, &mut scratch).expect("errors");
                }
            });
        }
    });
    out
}

/// Winner (task index) of every row at the given statistics (the argmin by the
/// runtime's stable score order; temperature does not change the winner).
fn winners(e: &[f32], tasks: usize, stats: &[ErrStats]) -> Vec<usize> {
    e.chunks_exact(tasks)
        .map(|r| decide(r, stats, 1.0).unwrap().winner.unwrap())
        .collect()
}

/// Orthonormalise row vectors in f64 (modified Gram–Schmidt, twice).
fn orthonormal_f64(rows: &[f32], dim: usize) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = rows
        .chunks_exact(dim)
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect();
    for _ in 0..2 {
        for j in 0..v.len() {
            let (done, rest) = v.split_at_mut(j);
            let b = &mut rest[0];
            for q in done.iter() {
                let c: f64 = b.iter().zip(q).map(|(x, y)| x * y).sum();
                for (x, y) in b.iter_mut().zip(q) {
                    *x -= c * y;
                }
            }
            let n = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in b.iter_mut() {
                *x /= n;
            }
        }
    }
    v
}

/// Upper bound of the largest principal angle between span(a) (orthonormal f64
/// rows) and span(b) (orthonormal): asin of the Frobenius norm of the residual.
fn max_angle(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut fro = 0.0f64;
    for v in a {
        let mut r = v.clone();
        for q in b {
            let c: f64 = r.iter().zip(q).map(|(x, y)| x * y).sum();
            for (x, y) in r.iter_mut().zip(q) {
                *x -= c * y;
            }
        }
        fro += r.iter().map(|x| x * x).sum::<f64>();
    }
    fro.sqrt().min(1.0).asin()
}

fn fit_all(train: &V3Split, labels: &[String]) -> Vec<TaskFit64> {
    let dim = train.dim;
    let mut by_label: Vec<Vec<f32>> = vec![Vec::new(); labels.len()];
    for (i, r) in train.rows.iter().enumerate() {
        let j = labels
            .binary_search(&r.label)
            .expect("train label in the taxonomy");
        by_label[j].extend_from_slice(&train.x[i * dim..(i + 1) * dim]);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut fits: Vec<Option<TaskFit64>> = vec![None; labels.len()];
    let done: Vec<Vec<(usize, TaskFit64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads())
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= labels.len() {
                            break mine;
                        }
                        mine.push((j, fit_task_f64(&by_label[j], dim, DEFAULT_K).expect("fit")));
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (j, f) in done.into_iter().flatten() {
        fits[j] = Some(f);
    }
    fits.into_iter().map(|f| f.unwrap()).collect()
}

fn certify_on(
    cal: &V3Split,
    labels: &[String],
    packed: &Packed,
    stats: &[ErrStats],
) -> (Certification, Vec<f32>) {
    let e = error_matrix(packed, &cal.x, cal.dim);
    let truth: Vec<Option<usize>> = cal
        .rows
        .iter()
        .map(|r| labels.binary_search(&r.label).ok())
        .collect();
    let shas: Vec<String> = cal.rows.iter().map(|r| r.text_sha256()).collect();
    let (even, odd) = halves(&shas);
    let c = certify(&Calibration {
        errors: &e,
        tasks: labels.len(),
        stats,
        truth: &truth,
        even: &even,
        odd: &odd,
    })
    .expect("certify");
    (c, e)
}

fn gate_json(c: &Certification, v3: &V3Skill) -> serde_json::Value {
    json!({
        "temperature": c.temperature as f64, "temperature_ulps_vs_v3": ulps(c.temperature, v3.temperature),
        "novelty_theta": c.novelty_theta as f64, "theta_ulps_vs_v3": ulps(c.novelty_theta, v3.novelty_theta),
        "tau": c.tau as f64, "tau_v3": v3.tau.map(|t| t as f64), "certified": c.certified,
        "odd_half": {"n": c.odd_n, "accepted": c.odd_accepted, "correct": c.odd_correct},
        "calibration": {"correct": c.calibration_correct, "n": c.calibration_n},
        "nll_even": c.nll_even, "fminbound_evaluations": c.temperature_fit.nfev,
        "grid": c.grid.iter().map(|r| json!({"t": r.threshold, "accepted": r.accepted, "correct": r.correct,
            "lb": r.lower_bound, "novelty_rejected": r.novelty_rejected})).collect::<Vec<_>>(),
    })
}

#[test]
#[ignore = "local gate: needs CORTIQ_DECISION_V3_DIR and the v3 split files; run with --release"]
fn parity_v3_gates_f1_f2_f3() {
    let v3 = v3_dir().expect("set CORTIQ_DECISION_V3_DIR to artifacts/decision-v3-20260926");
    let mut report = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for (ds, dev_correct, dev_n, odd_acc, odd_ok) in EXPECT {
        let t0 = Instant::now();
        let skill = read_v3_skill(&v3, ds);
        let train = read_v3_split(&v3, ds, "train");
        let dev = read_v3_split(&v3, ds, "dev");
        let cal = read_v3_split(&v3, ds, "calibration");
        let dim = train.dim;
        assert_eq!(dim, skill.dim);
        let v3_tasks = skill.active();
        assert_eq!(
            v3_tasks.len(),
            skill.tasks.len(),
            "every v3 PH task is active"
        );
        let labels: Vec<String> = v3_tasks.iter().map(|t| t.label.clone()).collect();
        let mut sorted = labels.clone();
        sorted.sort();
        assert_eq!(labels, sorted, "v3 tasks are in bytewise label order");
        let read_s = t0.elapsed().as_secs_f64();

        // Control: the v3 topologies through the v4 runtime and certifier.
        let t1 = Instant::now();
        let v3_views: Vec<TaskView> = v3_tasks
            .iter()
            .map(|t| TaskView::new(&t.mean, &t.basis).unwrap())
            .collect();
        let v3_stats: Vec<ErrStats> = v3_tasks
            .iter()
            .map(|t| ErrStats {
                err_mean: t.err_mean,
                err_std: t.err_std,
            })
            .collect();
        let v3_packed = Packed::new(&v3_views).unwrap();
        let e_dev_v3 = error_matrix(&v3_packed, &dev.x, dim);
        let w_dev_v3 = winners(&e_dev_v3, labels.len(), &v3_stats);
        let v3_dev_correct = w_dev_v3
            .iter()
            .zip(&dev.rows)
            .filter(|(w, r)| labels[**w] == r.label)
            .count();
        let (c_v3, _) = certify_on(&cal, &labels, &v3_packed, &v3_stats);
        let control_s = t1.elapsed().as_secs_f64();

        // The v4 fit on the same train rows.
        let t2 = Instant::now();
        let fits = fit_all(&train, &labels);
        let fit_s = t2.elapsed().as_secs_f64();
        let f32fits: Vec<_> = fits.iter().map(|f| f.to_f32()).collect();
        let views: Vec<TaskView> = f32fits.iter().map(|f| f.topology.view()).collect();
        let stats: Vec<ErrStats> = f32fits.iter().map(|f| f.stats()).collect();

        // F2: principal angles and ranks.
        let (mut worst_angle, mut worst_label, mut rank_mismatch) = (0.0f64, String::new(), 0usize);
        let (mut mean_bits_equal, mut basis_values_equal, mut basis_values) =
            (0usize, 0usize, 0usize);
        let (mut stats_equal, mut angles) = (0usize, Vec::new());
        for ((f, g), t) in fits.iter().zip(&f32fits).zip(&v3_tasks) {
            if f.k != t.rank() {
                rank_mismatch += 1;
            }
            let mine: Vec<Vec<f64>> = (0..f.k).map(|j| f.row(j).to_vec()).collect();
            let theirs = orthonormal_f64(&t.basis, dim);
            let a = max_angle(&mine, &theirs).max(max_angle(&theirs, &mine));
            angles.push(a);
            if a > worst_angle {
                worst_angle = a;
                worst_label = t.label.clone();
            }
            mean_bits_equal += usize::from(
                g.topology
                    .mean
                    .iter()
                    .zip(&t.mean)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            );
            if g.topology.basis.len() == t.basis.len() {
                basis_values += t.basis.len();
                basis_values_equal += g
                    .topology
                    .basis
                    .iter()
                    .zip(&t.basis)
                    .filter(|(a, b)| a.to_bits() == b.to_bits())
                    .count();
            }
            let s = g.stats();
            stats_equal += usize::from(
                s.err_mean.to_bits() == t.err_mean.to_bits()
                    && s.err_std.to_bits() == t.err_std.to_bits(),
            );
        }

        // F1 and the E comparison on dev.
        let t3 = Instant::now();
        let packed = Packed::new(&views).unwrap();
        let e_dev = error_matrix(&packed, &dev.x, dim);
        let w_dev = winners(&e_dev, labels.len(), &stats);
        let correct = w_dev
            .iter()
            .zip(&dev.rows)
            .filter(|(w, r)| labels[**w] == r.label)
            .count();
        let flips = w_dev.iter().zip(&w_dev_v3).filter(|(a, b)| a != b).count();
        let mut worst_rel_e = 0.0f64;
        for (a, b) in e_dev.iter().zip(&e_dev_v3) {
            let rel = (*a as f64 - *b as f64).abs() / (*b as f64).max(f64::MIN_POSITIVE);
            worst_rel_e = worst_rel_e.max(rel);
        }
        // F3: certification of the v4 fit.
        let (c, _) = certify_on(&cal, &labels, &packed, &stats);
        let eval_s = t3.elapsed().as_secs_f64();
        let timing = resonance_timing(&views, &stats, &dev.x, dim);

        let mut fail = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("{ds}: {what}"));
            }
        };
        // Control (the v3 topologies reproduce the shipped numbers).
        fail(
            v3_dev_correct == dev_correct,
            format!("control dev {v3_dev_correct} != {dev_correct}"),
        );
        fail(
            ulps(c_v3.temperature, skill.temperature) <= 1,
            format!("control T {} vs {}", c_v3.temperature, skill.temperature),
        );
        fail(
            ulps(c_v3.novelty_theta, skill.novelty_theta) <= 2,
            format!(
                "control θ {} vs {}",
                c_v3.novelty_theta, skill.novelty_theta
            ),
        );
        fail(
            Some(c_v3.tau) == skill.tau && c_v3.certified == skill.certified,
            format!("control τ {} vs {:?}", c_v3.tau, skill.tau),
        );
        fail(
            (c_v3.odd_accepted, c_v3.odd_correct) == (odd_acc, odd_ok),
            format!("control odd {}/{}", c_v3.odd_accepted, c_v3.odd_correct),
        );
        // F1.
        fail(
            dev.rows.len() == dev_n && correct == dev_correct,
            format!(
                "F1 dev {correct}/{} != {dev_correct}/{dev_n}",
                dev.rows.len()
            ),
        );
        fail(flips == 0, format!("F1 {flips} winner flips"));
        // F2.
        fail(
            rank_mismatch == 0,
            format!("F2 {rank_mismatch} rank mismatches"),
        );
        fail(
            worst_angle <= MAX_ANGLE,
            format!("F2 angle {worst_angle:e} ({worst_label})"),
        );
        fail(
            worst_rel_e <= MAX_REL_E,
            format!("F2 relative E {worst_rel_e:e}"),
        );
        // F3.
        fail(
            ulps(c.temperature, skill.temperature) <= 1,
            format!("F3 T {} vs v3 {}", c.temperature, skill.temperature),
        );
        fail(
            ulps(c.novelty_theta, skill.novelty_theta) <= 2,
            format!("F3 θ {} vs v3 {}", c.novelty_theta, skill.novelty_theta),
        );
        fail(
            Some(c.tau) == skill.tau && c.certified == skill.certified,
            format!("F3 τ {} vs {:?}", c.tau, skill.tau),
        );
        fail(
            (c.odd_accepted, c.odd_correct) == (odd_acc, odd_ok),
            format!(
                "F3 odd {}/{} != {odd_acc}/{odd_ok}",
                c.odd_accepted, c.odd_correct
            ),
        );

        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let entry = json!({
            "dataset": ds, "tasks": labels.len(), "dim": dim,
            "rows": {"train": train.rows.len(), "dev": dev.rows.len(), "calibration": cal.rows.len()},
            "control_v3_topologies": {"dev_correct": v3_dev_correct, "gate": gate_json(&c_v3, &skill)},
            "f1": {"dev_correct": correct, "dev_n": dev.rows.len(), "expected": dev_correct, "flips_vs_v3": flips},
            "f2": {"max_principal_angle_rad": worst_angle, "worst_label": worst_label,
                   "median_angle_rad": angles[angles.len() / 2], "rank_mismatches": rank_mismatch,
                   "max_relative_e_dev": worst_rel_e, "means_bit_equal": mean_bits_equal,
                   "basis_values_bit_equal": [basis_values_equal, basis_values],
                   "err_stats_f32_equal": stats_equal},
            "f3": gate_json(&c, &skill),
            "v3": {"temperature": skill.temperature as f64, "novelty_theta": skill.novelty_theta as f64,
                   "tau": skill.tau.map(|t| t as f64), "certified": skill.certified, "odd_half": [odd_acc, odd_ok]},
            "seconds": {"read": read_s, "control": control_s, "fit": fit_s, "eval_certify": eval_s},
            "resonance_dev_single_thread": timing,
        });
        eprintln!("{ds}: {entry}");
        report.push(entry);
    }
    let summary = json!({
        "gates": "F1-F3 (spec §6.2)", "v3_dir": v3.display().to_string(),
        "threads": threads(), "pass": failures.is_empty(), "failures": failures, "datasets": report,
    });
    println!("PARITY_V3_JSON {summary}");
    if let Some(out) = std::env::var_os("CORTIQ_DECISION_PARITY_OUT") {
        std::fs::write(out, serde_json::to_vec_pretty(&summary).unwrap())
            .expect("write parity json");
    }
    assert!(failures.is_empty(), "parity gates failed: {failures:#?}");
}
