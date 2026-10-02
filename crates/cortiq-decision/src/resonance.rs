//! Reference f32 resonance: the sequential reconstruction error of one affine
//! topology and the decision from a vector of errors (spec §3.5).
//!
//! Ported from `tools/cortiq-decision/src/lib.rs` (`reference_error`,
//! `Seed::from_errors`, lines 207-298), the Rust runtime whose f32 outputs
//! certified the shipped v3 skills. Every operation keeps the reference order:
//!
//! * error: `r = x − μ`; for each basis row `b` in order, `c = Σ r_i b_i`
//!   (f32, index order, from `+0.0`), then `r_i −= c·b_i`; `E = Σ r_i²` (f32,
//!   index order). The basis is NOT assumed orthonormal: the shortcut
//!   `‖r‖² − Σc²` (release `router::recon_error`) is never used;
//! * decision: stable sort by the f32 score `1/(1+E)` (descending; equal scores
//!   keep the candidate order), `margin = s0 − s1`,
//!   `p = softmax(−E/max(T, 1e-3))` in sorted order, `z = (E − err_mean)/err_std`
//!   of the winner, `novelty = 0.5·σ(z) + 0.25/(1+8·margin) + 0.25·(1−p_top)`;
//! * gate: accepted when `p_top ≥ τ` and `novelty ≤ θ` (`is_novel = novelty > θ`).
//!
//! No FMA, no reassociation and no SIMD across coordinates: rustc never contracts
//! or reassociates f32 arithmetic, so the loops below are the numerical contract.
//! [`crate::packed`] computes the same errors several tasks at a time, bit-exact.

use anyhow::{Result, ensure};

/// Error statistics of a task's own training rows, as the runtime uses them
/// (stored as f64 in the skill manifest, used as f32; spec §2.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrStats {
    pub err_mean: f32,
    pub err_std: f32,
}

impl ErrStats {
    /// The f32 statistics of the f64 values a fitter or a manifest carries
    /// (round to nearest, as serde reads a JSON number into an f32).
    pub fn from_f64(err_mean: f64, err_std: f64) -> Self {
        Self {
            err_mean: err_mean as f32,
            err_std: err_std as f32,
        }
    }

    /// Finite, `err_mean ≥ 0`, `err_std > 0`.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.err_mean.is_finite()
                && self.err_mean >= 0.0
                && self.err_std.is_finite()
                && self.err_std > 0.0,
            "invalid training error statistics (err_mean {}, err_std {})",
            self.err_mean,
            self.err_std
        );
        Ok(())
    }
}

/// A borrowed affine topology: `mean` [dim] and `basis` [rank × dim] row-major.
/// `rank` may be 0 (the error is then `‖x − μ‖²`).
#[derive(Clone, Copy, Debug)]
pub struct TaskView<'a> {
    pub mean: &'a [f32],
    pub basis: &'a [f32],
}

impl<'a> TaskView<'a> {
    /// A checked view (see [`TaskView::check`]).
    pub fn new(mean: &'a [f32], basis: &'a [f32]) -> Result<Self> {
        let t = Self { mean, basis };
        t.check()?;
        Ok(t)
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    pub fn rank(&self) -> usize {
        if self.mean.is_empty() {
            0
        } else {
            self.basis.len() / self.mean.len()
        }
    }

    /// Basis row `j`.
    pub fn row(&self, j: usize) -> &'a [f32] {
        let d = self.mean.len();
        &self.basis[j * d..(j + 1) * d]
    }

    /// Shape and finiteness of the parameters.
    pub fn check(&self) -> Result<()> {
        let d = self.mean.len();
        ensure!(d > 0, "topology dimension must be positive");
        ensure!(
            self.basis.len().is_multiple_of(d),
            "basis length {} is not a multiple of the dimension {d}",
            self.basis.len()
        );
        ensure!(
            self.mean.iter().chain(self.basis).all(|v| v.is_finite()),
            "topology parameters must be finite"
        );
        Ok(())
    }
}

/// An owned affine topology (the fitter's f32 output, or a loaded task).
#[derive(Clone, Debug, PartialEq)]
pub struct Topology {
    /// `[dim]`.
    pub mean: Vec<f32>,
    /// `[rank × dim]`, row-major.
    pub basis: Vec<f32>,
}

impl Topology {
    pub fn view(&self) -> TaskView<'_> {
        TaskView {
            mean: &self.mean,
            basis: &self.basis,
        }
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    pub fn rank(&self) -> usize {
        self.view().rank()
    }
}

/// Validate an input signal against a dimension (finite values only).
pub fn check_input(x: &[f32], dim: usize) -> Result<()> {
    ensure!(
        x.len() == dim,
        "input has {} values, the topologies expect {dim}",
        x.len()
    );
    ensure!(x.iter().all(|v| v.is_finite()), "input must be finite");
    Ok(())
}

/// The reference sequential reconstruction error (spec §3.5; reference
/// `lib.rs:282-298`). `x`, `mean` and every basis row have the same length;
/// the caller has validated shapes (see [`errors_reference`] for a checked form).
pub fn reference_error(x: &[f32], mean: &[f32], basis: &[f32]) -> f32 {
    let d = mean.len();
    debug_assert_eq!(x.len(), d);
    debug_assert!(d == 0 || basis.len().is_multiple_of(d));
    let mut r: Vec<f32> = x.iter().zip(mean).map(|(a, b)| a - b).collect();
    if d > 0 {
        for b in basis.chunks_exact(d) {
            let mut c = 0.0f32;
            for (ri, bi) in r.iter().zip(b) {
                c += ri * bi;
            }
            for (ri, bi) in r.iter_mut().zip(b) {
                *ri -= c * bi;
            }
        }
    }
    let mut e = 0.0f32;
    for v in r {
        e += v * v;
    }
    e
}

/// Errors of `x` against every topology, in order, with shape and finiteness checks.
pub fn errors_reference(x: &[f32], tasks: &[TaskView<'_>]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(tasks.len());
    for t in tasks {
        t.check()?;
        check_input(x, t.dim())?;
        out.push(reference_error(x, t.mean, t.basis));
    }
    ensure!(out.iter().all(|v| v.is_finite()), "numeric overflow");
    Ok(out)
}

/// The f32 score of an error, `1/(1+E)`.
#[inline]
pub fn score(e: f32) -> f32 {
    1.0 / (1.0 + e)
}

/// The clamped temperature the softmax divides by.
#[inline]
pub fn effective_temperature(temperature: f32) -> f32 {
    temperature.max(1e-3)
}

/// One candidate in the ranked order of a [`Decision`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranked {
    /// Position of the candidate in the `errors` slice given to [`decide`].
    pub index: usize,
    pub error: f32,
    /// `1/(1+E)`.
    pub score: f32,
    /// `softmax(−E/max(T, 1e-3))` over the candidates.
    pub probability: f32,
}

/// The runtime decision over a set of candidate topologies (spec §3.5).
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Position of the winning candidate; `None` when there was no candidate
    /// (an empty active set always abstains).
    pub winner: Option<usize>,
    /// Softmax probability of the winner (`p_top`, the reference's `confidence`).
    pub p_top: f32,
    /// Score `1/(1+E)` of the winner (the reference's `raw_confidence`).
    pub raw_confidence: f32,
    /// `s0 − s1`; the winner's own score when there is one candidate.
    pub margin: f32,
    /// `(E − err_mean)/err_std` of the winner.
    pub z: f32,
    /// `0.5·σ(z) + 0.25/(1+8·margin) + 0.25·(1−p_top)`; 1 for an empty set.
    pub novelty: f32,
    /// Candidates sorted by score (descending, stable).
    pub ranked: Vec<Ranked>,
}

impl Decision {
    /// The abstention of an empty candidate set (reference `__uninitialized__`).
    pub fn abstain() -> Self {
        Self {
            winner: None,
            p_top: 0.0,
            raw_confidence: 0.0,
            margin: 0.0,
            z: 0.0,
            novelty: 1.0,
            ranked: Vec::new(),
        }
    }

    /// `novelty > θ`; always novel without a winner.
    pub fn is_novel(&self, novelty_theta: f32) -> bool {
        self.winner.is_none() || self.novelty > novelty_theta
    }

    /// The gate: `p_top ≥ τ` and `novelty ≤ θ` (and a winner exists).
    pub fn accepted(&self, tau: f32, novelty_theta: f32) -> bool {
        self.winner.is_some() && self.p_top >= tau && !self.is_novel(novelty_theta)
    }

    /// Probability of the candidate at position `index` of the `errors` slice.
    pub fn probability_of(&self, index: usize) -> Option<f32> {
        self.ranked
            .iter()
            .find(|r| r.index == index)
            .map(|r| r.probability)
    }

    /// The winner's error.
    pub fn top_error(&self) -> Option<f32> {
        self.ranked.first().map(|r| r.error)
    }
}

/// The decision from the errors of the candidate topologies (reference
/// `Seed::from_errors`, `lib.rs:207-278`). `stats[i]` belongs to `errors[i]`.
/// Errors must be finite and non-negative; an empty slice abstains.
pub fn decide(errors: &[f32], stats: &[ErrStats], temperature: f32) -> Result<Decision> {
    ensure!(
        errors.len() == stats.len(),
        "{} errors for {} task statistics",
        errors.len(),
        stats.len()
    );
    ensure!(
        errors.iter().all(|x| x.is_finite() && *x >= 0.0),
        "invalid errors"
    );
    for s in stats {
        s.check()?;
    }
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "invalid temperature {temperature}"
    );
    if errors.is_empty() {
        return Ok(Decision::abstain());
    }
    let mut scored: Vec<(Ranked, f32)> = errors
        .iter()
        .zip(stats)
        .enumerate()
        .map(|(index, (&e, s))| {
            (
                Ranked {
                    index,
                    error: e,
                    score: score(e),
                    probability: 0.0,
                },
                (e - s.err_mean) / s.err_std,
            )
        })
        .collect();
    // Stable: equal scores (including ties created by f32 rounding) keep the
    // candidate order, as the reference's `sort_by`.
    scored.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).expect("finite scores"));
    let raw_confidence = scored[0].0.score;
    let margin = if scored.len() > 1 {
        raw_confidence - scored[1].0.score
    } else {
        raw_confidence
    };
    let t = effective_temperature(temperature);
    let mut logits: Vec<f32> = scored.iter().map(|(s, _)| -s.error / t).collect();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in &mut logits {
        *p = (*p - mx).exp();
        sum += *p;
    }
    let inv = 1.0 / sum;
    for (i, (s, _)) in scored.iter_mut().enumerate() {
        logits[i] *= inv;
        s.probability = logits[i];
    }
    let confidence = logits[0];
    let z = scored[0].1;
    let novelty = 0.5 * (1.0 / (1.0 + (-z).exp()))
        + 0.25 * (1.0 / (1.0 + margin * 8.0))
        + 0.25 * (1.0 - confidence);
    Ok(Decision {
        winner: Some(scored[0].0.index),
        p_top: confidence,
        raw_confidence,
        margin,
        z,
        novelty,
        ranked: scored.into_iter().map(|(r, _)| r).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_candidate_margin_is_its_score() {
        let d = decide(&[0.25], &[ErrStats::from_f64(0.2, 0.05)], 0.02).unwrap();
        assert_eq!(d.winner, Some(0));
        assert_eq!(d.margin, d.raw_confidence);
        assert_eq!(d.p_top, 1.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let s = ErrStats::from_f64(0.2, 0.05);
        assert!(decide(&[f32::NAN], &[s], 0.1).is_err());
        assert!(decide(&[-1.0], &[s], 0.1).is_err());
        assert!(decide(&[1.0], &[], 0.1).is_err());
        assert!(decide(&[1.0], &[ErrStats::from_f64(0.2, 0.0)], 0.1).is_err());
        assert!(decide(&[1.0], &[s], 0.0).is_err());
        assert!(TaskView::new(&[1.0, 2.0], &[1.0]).is_err());
        assert!(TaskView::new(&[], &[]).is_err());
    }
}
