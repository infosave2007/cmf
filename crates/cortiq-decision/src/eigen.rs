//! Symmetric eigendecomposition in f64: Householder tridiagonalisation `tred2`
//! and the implicit QL iteration `tql2` (spec §3.4).
//!
//! A line-by-line port of JAMA's `EigenvalueDecomposition` (public domain), itself
//! derived from the Algol procedures of Bowdler, Martin, Reinsch and Wilkinson
//! (Handbook for Automatic Computation, Vol. II) and the EISPACK Fortran. No
//! LAPACK, no libm: `hypot` is JAMA's `Maths.hypot` (IEEE `sqrt`, `mul`, `div`),
//! so the result depends only on IEEE-754 double arithmetic. The eigenvector
//! matrix is stored column-major (`v[c·n + r]` is JAMA's `V[r][c]`); this only
//! changes memory access, not a single operation.

use anyhow::{Result, ensure};

/// Eigenpairs of a symmetric matrix.
#[derive(Clone, Debug)]
pub struct SymEigen {
    /// Order of the matrix.
    pub n: usize,
    /// Eigenvalues in ascending order (tql2's final selection sort).
    pub values: Vec<f64>,
    /// Eigenvectors, column-major: vector `j` is `vectors[j·n .. (j+1)·n]`.
    pub vectors: Vec<f64>,
}

impl SymEigen {
    /// Eigenvector `j` (the one of `values[j]`).
    pub fn vector(&self, j: usize) -> &[f64] {
        &self.vectors[j * self.n..(j + 1) * self.n]
    }
}

/// `sqrt(a² + b²)` without under/overflow (JAMA `Maths.hypot`).
pub fn hypot(a: f64, b: f64) -> f64 {
    if a.abs() > b.abs() {
        let r = b / a;
        a.abs() * (1.0 + r * r).sqrt()
    } else if b != 0.0 {
        let r = a / b;
        b.abs() * (1.0 + r * r).sqrt()
    } else {
        0.0
    }
}

/// Eigendecomposition of the symmetric `n × n` matrix `a` (row-major; only its
/// symmetry is assumed, both triangles are read as JAMA does).
pub fn sym_eigen(a: &[f64], n: usize) -> Result<SymEigen> {
    ensure!(
        a.len() == n * n,
        "matrix has {} values, expected {n}²",
        a.len()
    );
    ensure!(a.iter().all(|v| v.is_finite()), "matrix must be finite");
    if n == 0 {
        return Ok(SymEigen {
            n,
            values: Vec::new(),
            vectors: Vec::new(),
        });
    }
    // Column-major V = A (JAMA copies A into V; for a symmetric A the
    // transpose is the same matrix, but copy explicitly for exactness).
    let mut v = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            v[c * n + r] = a[r * n + c];
        }
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    tred2(n, &mut v, &mut d, &mut e);
    tql2(n, &mut v, &mut d, &mut e)?;
    Ok(SymEigen {
        n,
        values: d,
        vectors: v,
    })
}

// V[r][c] of JAMA.
macro_rules! at {
    ($v:expr, $n:expr, $r:expr, $c:expr) => {
        $v[($c) * $n + ($r)]
    };
}

/// Householder reduction to tridiagonal form (JAMA `tred2`).
fn tred2(n: usize, v: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    for j in 0..n {
        d[j] = at!(v, n, n - 1, j);
    }
    for i in (1..n).rev() {
        // Scale to avoid under/overflow.
        let mut scale = 0.0f64;
        let mut h = 0.0f64;
        for dk in d.iter().take(i) {
            scale += dk.abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = at!(v, n, i - 1, j);
                at!(v, n, i, j) = 0.0;
                at!(v, n, j, i) = 0.0;
            }
        } else {
            // Generate Householder vector.
            for dk in d.iter_mut().take(i) {
                *dk /= scale;
                h += *dk * *dk;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for ej in e.iter_mut().take(i) {
                *ej = 0.0;
            }
            // Apply similarity transformation to remaining columns.
            for j in 0..i {
                f = d[j];
                at!(v, n, j, i) = f;
                g = e[j] + at!(v, n, j, j) * f;
                for k in j + 1..i {
                    g += at!(v, n, k, j) * d[k];
                    e[k] += at!(v, n, k, j) * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    at!(v, n, k, j) -= f * e[k] + g * d[k];
                }
                d[j] = at!(v, n, i - 1, j);
                at!(v, n, i, j) = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate transformations.
    for i in 0..n - 1 {
        at!(v, n, n - 1, i) = at!(v, n, i, i);
        at!(v, n, i, i) = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = at!(v, n, k, i + 1) / h;
            }
            for j in 0..=i {
                let mut g = 0.0f64;
                for k in 0..=i {
                    g += at!(v, n, k, i + 1) * at!(v, n, k, j);
                }
                for k in 0..=i {
                    at!(v, n, k, j) -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            at!(v, n, k, i + 1) = 0.0;
        }
    }
    for j in 0..n {
        d[j] = at!(v, n, n - 1, j);
        at!(v, n, n - 1, j) = 0.0;
    }
    at!(v, n, n - 1, n - 1) = 1.0;
    e[0] = 0.0;
}

/// Symmetric tridiagonal QL algorithm (JAMA `tql2`), then the ascending sort.
fn tql2(n: usize, v: &mut [f64], d: &mut [f64], e: &mut [f64]) -> Result<()> {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    let eps = 2.0f64.powi(-52);
    // JAMA does not bound the iterations; a bound only turns a hang on a
    // pathological input into an error (the QL iteration converges cubically).
    const MAX_ITER: usize = 1000;
    for l in 0..n {
        // Find small subdiagonal element.
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        // If m == l, d[l] is an eigenvalue, otherwise iterate. (e[n-1] = 0,
        // so m < n whenever the loop breaks; m == n cannot happen.)
        if m > l {
            let mut iter = 0usize;
            loop {
                iter += 1;
                ensure!(
                    iter <= MAX_ITER,
                    "tql2 did not converge in {MAX_ITER} iterations"
                );
                // Compute implicit shift.
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = hypot(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;
                // Implicit QL transformation.
                p = d[m];
                let mut c = 1.0f64;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0f64;
                let mut s2 = 0.0f64;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    // Accumulate transformation (columns i and i+1).
                    let (lo, hi) = v.split_at_mut((i + 1) * n);
                    let vi = &mut lo[i * n..];
                    let vi1 = &mut hi[..n];
                    for k in 0..n {
                        h = vi1[k];
                        vi1[k] = s * vi[k] + c * h;
                        vi[k] = c * vi[k] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                // Check for convergence.
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    // Sort eigenvalues and corresponding vectors (ascending, selection sort).
    for i in 0..n - 1 {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().skip(i + 1) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in 0..n {
                v.swap(i * n + j, k * n + j);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(e: &SymEigen) -> Vec<f64> {
        let n = e.n;
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            let v = e.vector(j);
            for r in 0..n {
                for c in 0..n {
                    a[r * n + c] += e.values[j] * v[r] * v[c];
                }
            }
        }
        a
    }

    #[test]
    fn small_matrices_decompose() {
        for n in [1usize, 2, 3, 7, 16] {
            let mut a = vec![0.0; n * n];
            for r in 0..n {
                for c in 0..=r {
                    let x =
                        ((r * 7 + c * 3) % 11) as f64 * 0.1 - 0.4 + if r == c { 2.0 } else { 0.0 };
                    a[r * n + c] = x;
                    a[c * n + r] = x;
                }
            }
            let e = sym_eigen(&a, n).unwrap();
            assert!(e.values.windows(2).all(|w| w[0] <= w[1]));
            let b = reconstruct(&e);
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-12, "n={n}: {x} vs {y}");
            }
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = e
                        .vector(i)
                        .iter()
                        .zip(e.vector(j))
                        .map(|(a, b)| a * b)
                        .sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn zero_and_diagonal_matrices() {
        let e = sym_eigen(&[0.0; 9], 3).unwrap();
        assert_eq!(e.values, vec![0.0; 3]);
        let e = sym_eigen(&[3.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(e.values, vec![1.0, 3.0]);
        assert!(sym_eigen(&[1.0, 2.0], 2).is_err());
        assert!(sym_eigen(&[f64::NAN], 1).is_err());
    }
}
