//! Gate certification (spec §3.6): temperature `T`, novelty threshold `θ` and
//! confidence threshold `τ` of a skill, from the runtime f32 errors of its
//! calibration rows — the procedure that certified the shipped PH skills
//! (`ship_v3/build_ph.py`, `evaluate_v3/common.py:322-396`).
//!
//! * halves: calibration rows sorted by the hex sha256 of their UTF-8 text
//!   (stable); even positions fit `T` and `θ`, odd positions test the gate;
//! * `T`: [`fminbound`] of the mean NLL of `softmax(−E/T)` over the even rows
//!   whose label has an active task, in `log T ∈ [ln 1e-3, ln 1]`,
//!   `xatol = 1e-5`, `maxiter = 500`; logsumexp in f64 with the max shift, sums
//!   pairwise as numpy; the result is rounded to f32;
//! * `θ`: the f32 novelties of the even half sorted ascending,
//!   `pos = f32(n−1)·(1 − 0.05f32)`, `idx = min(floor(f64(pos) + 0.5), n−1)`,
//!   `θ = min(v[idx] + 1e-4f32, 0.999f32)`;
//! * `τ`: per threshold `t` of [`THRESHOLDS`] on the odd half, accepted =
//!   `p_top ≥ f32(t)` and `novelty ≤ θ`; `lb = BetaInv(α; k, n−k+1)` with
//!   `α = 0.05/14` (0 when `k = 0`); a threshold qualifies with
//!   `accepted ≥ 100` and `lb ≥ 0.95`; the qualifying one with the most accepted
//!   rows wins, ties to the smaller threshold. None qualifies: `certified = false`,
//!   `τ = 0` (the gate is then θ alone). The grid with θ off is kept as evidence.

use crate::fit::pairwise_sum;
use crate::resonance::{Decision, ErrStats, decide};
use crate::specfn::{Fmin, clopper_pearson_lower, fminbound};
use anyhow::{Result, ensure};

/// The fixed threshold grid (Bonferroni over its 14 entries).
pub const THRESHOLDS: [f64; 14] = [
    0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999,
];
/// Family-wise level of the Clopper–Pearson bounds.
pub const ALPHA_FAMILY: f64 = 0.05;
/// The certification target of the selective accuracy.
pub const TARGET: f64 = 0.95;
/// Rows a threshold must accept on the odd half.
pub const MIN_ACCEPTED: usize = 100;
/// Bounds of the temperature (the runtime clamps `T` at `1e-3`).
pub const T_BOUNDS: (f64, f64) = (1e-3, 1.0);
/// `minimize_scalar(method='bounded')` defaults.
pub const T_XATOL: f64 = 1e-5;
pub const T_MAXITER: usize = 500;
/// router.rs `calibrate_novelty`: target false-positive rate, nudge and cap.
pub const THETA_TARGET_FPR: f32 = 0.05;
pub const THETA_NUDGE: f32 = 1e-4;
pub const THETA_CAP: f32 = 0.999;
/// The halves rule, as recorded in a skill manifest.
pub const HALVES_RULE: &str =
    "calibration rows sorted by sha256(utf8 text) hex; even -> T,theta; odd -> gate";
/// The halves rule of an auto-skill (no build rows, 0.8.6): its calibration
/// subset C is a property of each learned row, never stored — a row is in C
/// iff the first 8 bytes of `sha256(phi_P as f32 little-endian)` read as a
/// little-endian u64 give 4 modulo 5 (≈ 20 %), so a row never moves between
/// the fit and C as more rows arrive (a positional carve-out would, and the
/// champion would be scored on rows it fitted). Inside C the halves follow
/// [`halves`] over the same hex keys. [`auto_row_key`] computes both.
pub const HALVES_RULE_AUTO: &str = "learned rows: in C iff u64le(sha256(phi_P f32le)[..8]) % 5 == 4; C sorted by that sha256 hex; even -> T,theta; odd -> gate";
/// The modulus of [`HALVES_RULE_AUTO`] (one row in five is calibration).
pub const AUTO_CAL_EVERY: u64 = 5;

/// The key of a learned row under [`HALVES_RULE_AUTO`]: the hex sha256 of its
/// φ_P as little-endian f32 bytes, and whether the row is in the calibration
/// subset C. Text-free (the buffer holds vectors only) and deterministic.
pub fn auto_row_key(phi_p: &[f32]) -> (String, bool) {
    let (hex, head) = phi_p_digest(phi_p);
    (hex, head % AUTO_CAL_EVERY == AUTO_CAL_EVERY - 1)
}

/// Whether a text of an auto-skill that still has quarantined labels is
/// *explored* (DESIGN A16): its locally accepted answer is escalated to the
/// oracle anyway when `u64le(sha256(φ_P f32le)[..8]) % every == 0`, so a
/// rare label collects examples at `1/every` of its traffic even while the
/// gate confidently misnames it. `every == 0` turns exploration off. The same
/// digest as [`auto_row_key`]: with the default 4 the explored rows (0 mod 4)
/// and the calibration rows (4 mod 5) are different residue classes, and a
/// text is explored or not regardless of order and population.
pub fn auto_explores(phi_p: &[f32], every: u64) -> bool {
    every != 0 && phi_p_digest(phi_p).1.is_multiple_of(every)
}

/// The hex sha256 of φ_P as little-endian f32 bytes and its first 8 bytes as
/// a little-endian u64 (the key of [`auto_row_key`] and [`auto_explores`]).
fn phi_p_digest(phi_p: &[f32]) -> (String, u64) {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for v in phi_p {
        h.update(v.to_le_bytes());
    }
    let digest = h.finalize();
    let head = u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"));
    (format!("{digest:x}"), head)
}

/// `0.05 / 14`.
pub fn alpha() -> f64 {
    ALPHA_FAMILY / THRESHOLDS.len() as f64
}

/// Even and odd positions of the rows sorted by their hex sha256 (stable: equal
/// texts keep their input order, as Python's `sorted`).
pub fn halves<S: AsRef<str>>(sha256_hex: &[S]) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..sha256_hex.len()).collect();
    order.sort_by(|&a, &b| sha256_hex[a].as_ref().cmp(sha256_hex[b].as_ref()));
    let even = order.iter().step_by(2).copied().collect();
    let odd = order.iter().skip(1).step_by(2).copied().collect();
    (even, odd)
}

/// The error matrix of a calibration set over the active tasks of a skill.
#[derive(Clone, Copy, Debug)]
pub struct Calibration<'a> {
    /// `rows × tasks` f32 runtime errors, row-major, tasks in skill order.
    pub errors: &'a [f32],
    pub tasks: usize,
    /// Statistics of each active task.
    pub stats: &'a [ErrStats],
    /// Index of the row's label among the active tasks; `None` when the label
    /// has no active task (such rows are skipped by `T` and always wrong).
    pub truth: &'a [Option<usize>],
    /// Row indices of the even half (`T`, `θ`) and of the odd half (gate).
    pub even: &'a [usize],
    pub odd: &'a [usize],
}

impl Calibration<'_> {
    fn rows(&self) -> usize {
        self.truth.len()
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.errors[i * self.tasks..(i + 1) * self.tasks]
    }

    fn check(&self) -> Result<()> {
        let n = self.rows();
        ensure!(
            self.tasks > 0,
            "certification needs at least one active task"
        );
        ensure!(
            self.stats.len() == self.tasks,
            "{} statistics for {} tasks",
            self.stats.len(),
            self.tasks
        );
        ensure!(
            self.errors.len() == n * self.tasks,
            "error matrix is not rows × tasks"
        );
        ensure!(
            self.truth.iter().flatten().all(|&t| t < self.tasks),
            "truth index out of range"
        );
        let mut seen = vec![false; n];
        for &i in self.even.iter().chain(self.odd) {
            ensure!(i < n && !seen[i], "halves must be disjoint row indices");
            seen[i] = true;
        }
        ensure!(!self.even.is_empty(), "the even half is empty");
        Ok(())
    }
}

/// One row of a gate grid (the manifest's `evidence.odd.grid`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridRow {
    pub threshold: f64,
    pub accepted: usize,
    pub correct: usize,
    /// Clopper–Pearson lower bound of `correct/accepted` at level [`alpha`].
    pub lower_bound: f64,
    /// Rows with `p_top ≥ t` removed by the novelty threshold.
    pub novelty_rejected: usize,
}

/// The certified gate of a skill and its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Certification {
    pub temperature: f32,
    /// Mean NLL of the even rows with an active label at the chosen `log T`.
    pub nll_even: f64,
    /// The optimiser outcome (`x` = `log T` before rounding).
    pub temperature_fit: Fmin,
    pub novelty_theta: f32,
    /// The chosen threshold (0 when not certified).
    pub tau: f32,
    pub certified: bool,
    /// The chosen grid row when certified.
    pub chosen: Option<GridRow>,
    pub grid: Vec<GridRow>,
    pub grid_theta_off: Vec<GridRow>,
    pub even_n: usize,
    /// Even rows that entered `T` (label with an active task).
    pub even_t_n: usize,
    pub odd_n: usize,
    /// Odd rows accepted by the final gate (θ on, `p_top ≥ τ`), and how many are correct.
    pub odd_accepted: usize,
    pub odd_correct: usize,
    /// Rows of the whole calibration set whose winner is the truth.
    pub calibration_correct: usize,
    pub calibration_n: usize,
}

/// The per-row runtime quantities at a temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowGate {
    pub winner: Option<usize>,
    pub p_top: f32,
    pub novelty: f32,
}

impl RowGate {
    fn of(d: &Decision) -> Self {
        Self {
            winner: d.winner,
            p_top: d.p_top,
            novelty: d.novelty,
        }
    }
}

/// The f64 mean NLL of `softmax(−E/exp(log_t))` over `rows` (numpy/scipy order:
/// `z = −E / exp(log_t)`, `logsumexp` with the row max, sums pairwise).
pub fn mean_nll(errors: &[f32], tasks: usize, rows: &[usize], truth: &[usize], log_t: f64) -> f64 {
    let t = log_t.exp();
    let mut per_row = Vec::with_capacity(rows.len());
    let mut z = vec![0.0f64; tasks];
    let mut tmp = vec![0.0f64; tasks];
    for (&i, &y) in rows.iter().zip(truth) {
        let e = &errors[i * tasks..(i + 1) * tasks];
        for (zj, &ej) in z.iter_mut().zip(e) {
            *zj = -(ej as f64) / t;
        }
        let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shift = if mx.is_finite() { mx } else { 0.0 };
        for (tj, zj) in tmp.iter_mut().zip(&z) {
            *tj = (zj - shift).exp();
        }
        let lse = pairwise_sum(&tmp).ln() + shift;
        per_row.push(lse - z[y]);
    }
    pairwise_sum(&per_row) / rows.len() as f64
}

/// The fitted temperature (f32) and the optimiser result.
pub fn fit_temperature(
    errors: &[f32],
    tasks: usize,
    rows: &[usize],
    truth: &[usize],
) -> Result<(f32, Fmin)> {
    ensure!(
        !rows.is_empty(),
        "no calibration rows to fit the temperature"
    );
    ensure!(rows.len() == truth.len(), "rows and truth differ in length");
    let (lo, hi) = (T_BOUNDS.0.ln(), T_BOUNDS.1.ln());
    let fit = fminbound(
        |lt| mean_nll(errors, tasks, rows, truth, lt),
        lo,
        hi,
        T_XATOL,
        T_MAXITER,
    )?;
    ensure!(
        fit.success(),
        "temperature fit failed (status {})",
        fit.status
    );
    Ok((fit.x.exp() as f32, fit))
}

/// router.rs `calibrate_novelty` on f32 novelties (`common.py:347-354`).
pub fn novelty_theta(values: &[f32]) -> Result<f32> {
    ensure!(!values.is_empty(), "no rows for the novelty threshold");
    ensure!(
        values.iter().all(|v| !v.is_nan()),
        "novelty must not be NaN"
    );
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("not NaN"));
    let n = v.len();
    let position = (n - 1) as f32 * (1.0f32 - THETA_TARGET_FPR);
    let idx = (((position as f64) + 0.5).floor() as usize).min(n - 1);
    Ok((v[idx] + THETA_NUDGE).min(THETA_CAP))
}

/// The Clopper–Pearson grid over [`THRESHOLDS`] (`theta = None`: θ off).
pub fn gate_grid(
    gates: &[RowGate],
    correct: &[bool],
    rows: &[usize],
    theta: Option<f32>,
) -> Vec<GridRow> {
    let a = alpha();
    THRESHOLDS
        .iter()
        .map(|&t| {
            let tf = t as f32;
            let (mut accepted, mut ok, mut rejected) = (0usize, 0usize, 0usize);
            for &i in rows {
                let g = &gates[i];
                let conf = g.winner.is_some() && g.p_top >= tf;
                let in_scope = theta.is_none_or(|th| g.novelty <= th);
                if conf && in_scope {
                    accepted += 1;
                    ok += usize::from(correct[i]);
                } else if conf {
                    rejected += 1;
                }
            }
            GridRow {
                threshold: t,
                accepted,
                correct: ok,
                lower_bound: clopper_pearson_lower(ok as u64, accepted as u64, a),
                novelty_rejected: rejected,
            }
        })
        .collect()
}

/// The qualifying grid row with the most accepted rows (ties: the smaller threshold).
pub fn choose(grid: &[GridRow]) -> Option<GridRow> {
    let mut best: Option<GridRow> = None;
    for r in grid {
        if r.accepted >= MIN_ACCEPTED
            && r.lower_bound >= TARGET
            && best.is_none_or(|b| r.accepted > b.accepted)
        {
            best = Some(*r);
        }
    }
    best
}

/// The runtime gate quantities of every calibration row at `temperature`.
pub fn row_gates(cal: &Calibration<'_>, temperature: f32) -> Result<Vec<RowGate>> {
    (0..cal.rows())
        .map(|i| decide(cal.row(i), cal.stats, temperature).map(|d| RowGate::of(&d)))
        .collect()
}

/// Certify a skill from its calibration errors (spec §3.6).
pub fn certify(cal: &Calibration<'_>) -> Result<Certification> {
    cal.check()?;
    // T on the even rows whose label has an active task.
    let (t_rows, t_truth): (Vec<usize>, Vec<usize>) = cal
        .even
        .iter()
        .filter_map(|&i| cal.truth[i].map(|y| (i, y)))
        .unzip();
    let (temperature, fit) = fit_temperature(cal.errors, cal.tasks, &t_rows, &t_truth)?;
    let gates = row_gates(cal, temperature)?;
    let correct: Vec<bool> = gates
        .iter()
        .zip(cal.truth)
        .map(|(g, t)| g.winner.is_some() && g.winner == *t)
        .collect();
    let even_nov: Vec<f32> = cal.even.iter().map(|&i| gates[i].novelty).collect();
    let theta = novelty_theta(&even_nov)?;
    let grid = gate_grid(&gates, &correct, cal.odd, Some(theta));
    let grid_theta_off = gate_grid(&gates, &correct, cal.odd, None);
    let chosen = choose(&grid);
    let tau = chosen.map_or(0.0, |r| r.threshold as f32);
    let (mut odd_accepted, mut odd_correct) = (0usize, 0usize);
    for &i in cal.odd {
        let g = &gates[i];
        let acc = chosen.is_some() && g.winner.is_some() && g.p_top >= tau && g.novelty <= theta;
        if acc {
            odd_accepted += 1;
            odd_correct += usize::from(correct[i]);
        }
    }
    Ok(Certification {
        temperature,
        nll_even: fit.fun,
        temperature_fit: fit,
        novelty_theta: theta,
        tau,
        certified: chosen.is_some(),
        chosen,
        grid,
        grid_theta_off,
        even_n: cal.even.len(),
        even_t_n: t_rows.len(),
        odd_n: cal.odd.len(),
        odd_accepted,
        odd_correct,
        calibration_correct: correct.iter().filter(|&&c| c).count(),
        calibration_n: cal.rows(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_are_stable_on_equal_keys() {
        let keys = ["b", "a", "b", "a", "c"];
        let (even, odd) = halves(&keys);
        assert_eq!(even, vec![1, 0, 4]);
        assert_eq!(odd, vec![3, 2]);
    }

    /// The auto calibration membership is a function of the row alone (the
    /// same φ_P gives the same key and answer in any order and population)
    /// and lands close to one row in five over a large sample.
    #[test]
    fn auto_row_key_is_deterministic_and_about_a_fifth() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut vec = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect()
        };
        let rows: Vec<Vec<f32>> = (0..20_000).map(|_| vec(16)).collect();
        let first: Vec<(String, bool)> = rows.iter().map(|r| auto_row_key(r)).collect();
        let again: Vec<(String, bool)> = rows.iter().rev().map(|r| auto_row_key(r)).collect();
        assert!(first.iter().eq(again.iter().rev()));
        assert!(first.iter().all(|(k, _)| k.len() == 64));
        let in_c = first.iter().filter(|(_, c)| *c).count();
        let share = in_c as f64 / rows.len() as f64;
        assert!((0.18..=0.22).contains(&share), "share in C {share}");
        // Keys are the hex sha256 of the little-endian bytes.
        let (k, _) = auto_row_key(&[1.0, -2.5]);
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(k, crate::manifest::sha256_hex(&bytes));
    }

    /// Exploration (DESIGN A16) is the residue of the same digest's head:
    /// `every == 0` is off, `every == 1` explores everything, and with the
    /// default 4 about a quarter of the rows are explored, never a row of
    /// the calibration subset (the residue classes 0 mod 4 and 4 mod 5 meet
    /// only on 1/20 of the rows — both rules hold for those, and the fit
    /// never sees them).
    #[test]
    fn auto_explores_follows_the_hash_rule_and_zero_turns_it_off() {
        let phi = [1.0f32, -2.5];
        let mut bytes = Vec::new();
        for v in phi {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        let head = u64::from_le_bytes(digest[..8].try_into().unwrap());
        for every in 1..=7u64 {
            assert_eq!(
                auto_explores(&phi, every),
                head % every == 0,
                "every {every}"
            );
        }
        assert!(auto_explores(&phi, 1));
        assert!(!auto_explores(&phi, 0));
        assert!(!auto_explores(&[], 0));
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let rows: Vec<Vec<f32>> = (0..20_000)
            .map(|_| {
                (0..16)
                    .map(|_| {
                        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect()
            })
            .collect();
        let explored = rows.iter().filter(|r| auto_explores(r, 4)).count();
        let share = explored as f64 / rows.len() as f64;
        assert!((0.23..=0.27).contains(&share), "explored share {share}");
        assert!(rows.iter().all(|r| !auto_explores(r, 0)));
        // The same row answers the same in any order (a property of the row).
        let again = rows.iter().rev().filter(|r| auto_explores(r, 4)).count();
        assert_eq!(again, explored);
    }

    #[test]
    fn choose_prefers_more_accepted_then_smaller_threshold() {
        let row = |t, a, lb| GridRow {
            threshold: t,
            accepted: a,
            correct: a,
            lower_bound: lb,
            novelty_rejected: 0,
        };
        let grid = [
            row(0.0, 150, 0.94),
            row(0.5, 140, 0.96),
            row(0.6, 140, 0.97),
            row(0.7, 90, 0.99),
        ];
        assert_eq!(choose(&grid).unwrap().threshold, 0.5);
        assert!(choose(&[row(0.0, 99, 0.99)]).is_none());
    }
}
