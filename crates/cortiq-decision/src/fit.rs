//! Task fitter `gram-eigh-f64-v1` (spec §3.4): the affine topology of one label
//! from that label's own rows — the mean and the top-k principal directions of
//! the centred rows — reproducing the subspace of the numpy SVD fit of the v3
//! research code (`research_v3/resonance.py:116-127`).
//!
//! 1. rows in f64; `μ = Σx/n` (sequential over rows); `Xc = X − μ`;
//! 2. Gram `G = Xc·Xcᵀ` (n×n, f64; each entry a sequential sum over the
//!    coordinates, entries `i ≤ j` computed and mirrored);
//! 3. `tred2` + `tql2` ([`crate::eigen`]) in f64, no LAPACK;
//! 4. eigenvalues by decreasing value, ties by eigen index; `λ/n > 1e-8` kept;
//!    `keep = min(min(K, max(n−1, 1)), kept)`;
//! 5. `b_j = Xcᵀ v_j / sqrt(λ_j)`, then modified Gram–Schmidt in f64, twice
//!    (two full passes);
//! 6. sign: the component of largest magnitude is positive (ties: the lower
//!    index). A sign flip changes no bit of the f32 error (`c` and `b` both flip);
//! 7. `err_mean`/`err_std`: errors of the task's own rows with the f64 mean and
//!    basis (sequential projection, f64); population std with floor `1e-4`
//!    (`resonance.py:109-113`); mean and variance summed pairwise as numpy;
//! 8. μ and the basis are rounded to f32. One row gives an empty basis.
//!
//! The router's power iteration (`linalg.rs:111-155`) is not used: it does not
//! converge on nearly degenerate eigenvalues (E up to 2.64 %, angles up to 18.9°
//! against numpy in `pca_compare.json`).

use crate::eigen::sym_eigen;
use crate::resonance::{ErrStats, Topology};
use anyhow::{Result, ensure};

/// Name of this fit in the skill recipe (spec §2.4).
pub const FIT_NAME: &str = "gram-eigh-f64-v1";
/// Default number of principal directions per topology.
pub const DEFAULT_K: usize = 16;
/// Components whose eigenvalue `λ/n` is not above this are dropped.
pub const EIG_DROP: f64 = 1e-8;
/// Floor of the population standard deviation of the training errors.
pub const ERR_STD_FLOOR: f64 = 1e-4;
/// Rows a task needs to be scored (fewer: registered, inactive).
pub const MIN_ROWS_ACTIVE: usize = 2;
/// The sign rule recorded in the recipe.
pub const SIGN_RULE: &str = "max-abs-positive";

/// The f64 fit of one task (before rounding to f32).
#[derive(Clone, Debug)]
pub struct TaskFit64 {
    pub n: usize,
    pub dim: usize,
    /// `[dim]`.
    pub mean: Vec<f64>,
    /// `[k × dim]`, orthonormal rows, sign rule applied.
    pub basis: Vec<f64>,
    pub k: usize,
    /// All `λ/n` in decreasing order (`n` values): the squared singular values
    /// of the centred rows divided by `n`.
    pub eigenvalues: Vec<f64>,
    pub err_mean: f64,
    pub err_std: f64,
}

/// The fitted topology as it is stored and served.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskFit {
    pub n_train: usize,
    pub k: usize,
    pub topology: Topology,
    /// f64 statistics (the manifest value; the runtime uses them as f32).
    pub err_mean: f64,
    pub err_std: f64,
}

impl TaskFit {
    /// The runtime f32 statistics.
    pub fn stats(&self) -> ErrStats {
        ErrStats::from_f64(self.err_mean, self.err_std)
    }
}

impl TaskFit64 {
    /// Round the mean and the basis to f32 (step 8).
    pub fn to_f32(&self) -> TaskFit {
        TaskFit {
            n_train: self.n,
            k: self.k,
            topology: Topology {
                mean: self.mean.iter().map(|&v| v as f32).collect(),
                basis: self.basis.iter().map(|&v| v as f32).collect(),
            },
            err_mean: self.err_mean,
            err_std: self.err_std,
        }
    }

    /// Basis row `j` in f64.
    pub fn row(&self, j: usize) -> &[f64] {
        &self.basis[j * self.dim..(j + 1) * self.dim]
    }
}

/// Fit one task from its rows (`rows`: `n × dim` row-major f32) with at most
/// `k_max` directions (`K`); the stored f32 topology.
pub fn fit_task(rows: &[f32], dim: usize, k_max: usize) -> Result<TaskFit> {
    Ok(fit_task_f64(rows, dim, k_max)?.to_f32())
}

/// [`fit_task`] before rounding to f32.
pub fn fit_task_f64(rows: &[f32], dim: usize, k_max: usize) -> Result<TaskFit64> {
    ensure!(dim > 0, "dimension must be positive");
    ensure!(
        !rows.is_empty() && rows.len() % dim == 0,
        "rows must be a non-empty n × {dim} matrix (got {} values)",
        rows.len()
    );
    ensure!(rows.iter().all(|v| v.is_finite()), "rows must be finite");
    let n = rows.len() / dim;

    // 1. mean (sequential over rows, from +0.0, then / n) and centred rows.
    let mut mean = vec![0.0f64; dim];
    for row in rows.chunks_exact(dim) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    let nf = n as f64;
    for m in &mut mean {
        *m /= nf;
    }
    let xc: Vec<f64> = rows
        .chunks_exact(dim)
        .flat_map(|row| row.iter().zip(&mean).map(|(&v, m)| v as f64 - m))
        .collect();

    // 2. Gram matrix, i ≤ j, sequential dot products.
    let g = gram(&xc, n, dim);

    // 3. eigenpairs (ascending), 4. order by decreasing value, ties by index.
    let eig = sym_eigen(&g, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eig.values[b]
            .partial_cmp(&eig.values[a])
            .expect("finite eigenvalues")
            .then(a.cmp(&b))
    });
    let eigenvalues: Vec<f64> = order.iter().map(|&j| eig.values[j] / nf).collect();
    let kept = eigenvalues.iter().filter(|&&l| l > EIG_DROP).count();
    let k = k_max.min(n.saturating_sub(1).max(1)).min(kept);

    // 5. b_j = Xcᵀ v_j / sqrt(λ_j), then MGS twice.
    let mut basis = vec![0.0f64; k * dim];
    for (j, b) in basis.chunks_exact_mut(dim).enumerate() {
        let col = order[j];
        let v = eig.vector(col);
        for (i, xrow) in xc.chunks_exact(dim).enumerate() {
            let w = v[i];
            for (bd, &x) in b.iter_mut().zip(xrow) {
                *bd += x * w;
            }
        }
        let s = eig.values[col].sqrt();
        for bd in b.iter_mut() {
            *bd /= s;
        }
    }
    for _ in 0..2 {
        modified_gram_schmidt(&mut basis, k, dim)?;
    }
    // 6. sign rule.
    for b in basis.chunks_exact_mut(dim) {
        apply_sign_rule(b);
    }

    // 7. error statistics of the task's own rows with the f64 parameters.
    let errs: Vec<f64> = rows
        .chunks_exact(dim)
        .map(|row| error_f64(row, &mean, &basis))
        .collect();
    let (err_mean, err_std) = population_stats(&errs);

    Ok(TaskFit64 {
        n,
        dim,
        mean,
        basis,
        k,
        eigenvalues,
        err_mean,
        err_std,
    })
}

/// `G[i][j] = Σ_d xc[i][d]·xc[j][d]` (row-major n×n), each entry a sequential sum
/// from `+0.0` over `d`; four `j` at a time only to hide the add latency.
fn gram(xc: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        let a = &xc[i * dim..(i + 1) * dim];
        let mut j = i;
        while j + 4 <= n {
            let b0 = &xc[j * dim..(j + 1) * dim];
            let b1 = &xc[(j + 1) * dim..(j + 2) * dim];
            let b2 = &xc[(j + 2) * dim..(j + 3) * dim];
            let b3 = &xc[(j + 3) * dim..(j + 4) * dim];
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for d in 0..dim {
                let x = a[d];
                s0 += x * b0[d];
                s1 += x * b1[d];
                s2 += x * b2[d];
                s3 += x * b3[d];
            }
            for (o, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                g[i * n + j + o] = s;
                g[(j + o) * n + i] = s;
            }
            j += 4;
        }
        while j < n {
            let b = &xc[j * dim..(j + 1) * dim];
            let mut s = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                s += x * y;
            }
            g[i * n + j] = s;
            g[j * n + i] = s;
            j += 1;
        }
    }
    g
}

/// One pass of modified Gram–Schmidt over the `k` rows of `basis` (in place).
fn modified_gram_schmidt(basis: &mut [f64], k: usize, dim: usize) -> Result<()> {
    for j in 0..k {
        let (done, rest) = basis.split_at_mut(j * dim);
        let b = &mut rest[..dim];
        for l in 0..j {
            let q = &done[l * dim..(l + 1) * dim];
            let mut c = 0.0f64;
            for (x, y) in b.iter().zip(q) {
                c += x * y;
            }
            for (x, y) in b.iter_mut().zip(q) {
                *x -= c * y;
            }
        }
        let mut s = 0.0f64;
        for x in b.iter() {
            s += x * x;
        }
        let norm = s.sqrt();
        ensure!(
            norm.is_finite() && norm > 0.0,
            "degenerate basis direction {j} in the fit"
        );
        for x in b.iter_mut() {
            *x /= norm;
        }
    }
    Ok(())
}

/// Make the component of largest magnitude positive (ties: the lower index).
pub fn apply_sign_rule(b: &mut [f64]) {
    let mut best = 0usize;
    let mut best_abs = -1.0f64;
    for (i, v) in b.iter().enumerate() {
        if v.abs() > best_abs {
            best_abs = v.abs();
            best = i;
        }
    }
    if !b.is_empty() && b[best] < 0.0 {
        for v in b.iter_mut() {
            *v = -*v;
        }
    }
}

/// The sequential projection error in f64 (the f64 twin of the runtime error).
pub fn error_f64(x: &[f32], mean: &[f64], basis: &[f64]) -> f64 {
    let d = mean.len();
    let mut r: Vec<f64> = x.iter().zip(mean).map(|(&a, b)| a as f64 - b).collect();
    for b in basis.chunks_exact(d) {
        let mut c = 0.0f64;
        for (ri, bi) in r.iter().zip(b) {
            c += ri * bi;
        }
        for (ri, bi) in r.iter_mut().zip(b) {
            *ri -= c * bi;
        }
    }
    let mut e = 0.0f64;
    for v in r {
        e += v * v;
    }
    e
}

/// numpy's pairwise summation of a contiguous f64 array (`np.add.reduce`):
/// 8 partial sums over blocks of at most 128 values, halves split at multiples of 8.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Mean and population standard deviation (floor [`ERR_STD_FLOOR`]) as numpy
/// computes `err.mean()` and `sqrt(mean((err − m)²))`.
pub fn population_stats(errs: &[f64]) -> (f64, f64) {
    let n = errs.len() as f64;
    let m = pairwise_sum(errs) / n;
    let sq: Vec<f64> = errs.iter().map(|e| (e - m) * (e - m)).collect();
    let s = (pairwise_sum(&sq) / n).sqrt();
    (m, s.max(ERR_STD_FLOOR))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_row_has_an_empty_basis() {
        let f = fit_task_f64(&[1.0, 2.0, 3.0], 3, 16).unwrap();
        assert_eq!((f.n, f.k), (1, 0));
        assert!(f.basis.is_empty());
        assert_eq!(f.mean, vec![1.0, 2.0, 3.0]);
        assert_eq!((f.err_mean, f.err_std), (0.0, ERR_STD_FLOOR));
    }

    #[test]
    fn duplicate_rows_keep_no_direction() {
        let rows = [0.5f32, -1.0, 0.25, 0.5, -1.0, 0.25, 0.5, -1.0, 0.25];
        let f = fit_task_f64(&rows, 3, 16).unwrap();
        assert_eq!(f.k, 0);
    }

    #[test]
    fn pairwise_matches_sequential_on_small_inputs() {
        let a: Vec<f64> = (0..7).map(|i| i as f64 * 0.1).collect();
        let mut s = 0.0;
        for v in &a {
            s += v;
        }
        assert_eq!(pairwise_sum(&a), s);
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn sign_rule_ties_take_the_lower_index() {
        let mut b = [0.5, -0.5, 0.1];
        apply_sign_rule(&mut b);
        assert_eq!(b, [0.5, -0.5, 0.1]);
        let mut b = [-0.5, 0.5, 0.1];
        apply_sign_rule(&mut b);
        assert_eq!(b, [0.5, -0.5, -0.1]);
    }
}
