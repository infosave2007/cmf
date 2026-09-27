//! Special functions of the gate certification (spec §3.6): scipy's bounded
//! scalar minimiser `fminbound`, the regularised incomplete beta function and
//! its inverse (the Clopper–Pearson lower bound).
//!
//! * [`fminbound`] is a statement-by-statement port of
//!   `scipy.optimize._optimize._minimize_scalar_bounded` (scipy 1.13.1; the
//!   Brent golden-section/parabolic search that `minimize_scalar(method=
//!   'bounded')` runs, `xatol = 1e-5`, `maxiter = 500` by default).
//! * [`beta_inc`] evaluates `I_x(a, b)` for integer `a, b ≥ 1` with the modified
//!   Lentz continued fraction in f64 (the symmetric form when
//!   `x > (a+1)/(a+b+2)`); `ln B(a, b)` is a sum of logarithms (exact factorials).
//! * [`beta_inv`] bisects `I_x(a, b) = p` on `[0, 1]` down to an interval of
//!   `1e-15`. The Clopper–Pearson lower bound of `k` successes out of `n` at
//!   level `α` is `beta_inv(α, k, n − k + 1)` (0 when `k = 0`), scipy's
//!   `beta.ppf(α, k, n − k + 1)`.

use anyhow::{Result, ensure};

/// Result of [`fminbound`] (scipy `OptimizeResult` fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fmin {
    /// The minimiser `xf`.
    pub x: f64,
    /// `f(xf)`.
    pub fun: f64,
    /// Function evaluations.
    pub nfev: usize,
    /// 0 converged, 1 `maxiter` reached, 2 a NaN was met.
    pub status: u8,
}

impl Fmin {
    pub fn success(&self) -> bool {
        self.status == 0
    }
}

/// `np.sign(v) + (v == 0)`: −1 or +1 (NaN stays NaN).
fn sign_nonzero(v: f64) -> f64 {
    let s = if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        f64::NAN
    };
    s + if v == 0.0 { 1.0 } else { 0.0 }
}

/// Bounded scalar minimisation of `f` on `[x1, x2]` (scipy
/// `_minimize_scalar_bounded`, ported operation by operation).
pub fn fminbound<F: FnMut(f64) -> f64>(
    mut f: F,
    x1: f64,
    x2: f64,
    xatol: f64,
    maxiter: usize,
) -> Result<Fmin> {
    ensure!(
        x1.is_finite() && x2.is_finite(),
        "optimization bounds must be finite scalars"
    );
    ensure!(x1 <= x2, "the lower bound exceeds the upper bound");
    let maxfun = maxiter;
    let mut flag = 0u8;
    let sqrt_eps = (2.2e-16f64).sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = 0.0f64;
    let mut e = 0.0f64;
    let mut x;
    let mut fx = f(xf);
    let mut num = 1usize;
    let mut fu = f64::INFINITY;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        // Check for parabolic fit.
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            // Check for acceptability of parabola.
            if (p.abs() < (0.5 * q * r).abs()) && (p > q * (a - xf)) && (p < q * (b - xf)) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if ((x - a) < tol2) || ((b - x) < tol2) {
                    let si = sign_nonzero(xm - xf);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            // Golden-section step.
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign_nonzero(rat);
        x = xf + si * rat.abs().max(tol1);
        fu = f(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if (fu <= fnfc) || (nfc == xf) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if (fu <= ffulc) || (fulc == xf) || (fulc == nfc) {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            flag = 1;
            break;
        }
    }
    if xf.is_nan() || fx.is_nan() || fu.is_nan() {
        flag = 2;
    }
    Ok(Fmin {
        x: xf,
        fun: fx,
        nfev: num,
        status: flag,
    })
}

/// `ln B(a, b)` for integers `a, b ≥ 1`, as a sum of logarithms:
/// `B(a, b) = (m−1)! / (M·(M+1)···(M+m−1))` with `m = min(a, b)`, `M = max(a, b)`.
pub fn ln_beta_int(a: u64, b: u64) -> f64 {
    assert!(a >= 1 && b >= 1, "ln_beta_int needs a, b >= 1");
    let (m, big) = if a <= b { (a, b) } else { (b, a) };
    let mut s = 0.0f64;
    for i in 1..m {
        s += (i as f64).ln();
    }
    for i in big..big + m {
        s -= (i as f64).ln();
    }
    s
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 10_000;
    const EPS: f64 = 1e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    h
}

/// The regularised incomplete beta function `I_x(a, b)` for integers `a, b ≥ 1`.
pub fn beta_inc(a: u64, b: u64, x: f64) -> f64 {
    assert!(a >= 1 && b >= 1, "beta_inc needs a, b >= 1");
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let (af, bf) = (a as f64, b as f64);
    let front = (af * x.ln() + bf * (1.0 - x).ln() - ln_beta_int(a, b)).exp();
    if x < (af + 1.0) / (af + bf + 2.0) {
        front * beta_cf(af, bf, x) / af
    } else {
        1.0 - front * beta_cf(bf, af, 1.0 - x) / bf
    }
}

/// The `p`-quantile of the Beta(a, b) distribution for integers `a, b ≥ 1`:
/// bisection of `I_x(a, b) = p` until the bracket is at most `1e-15` wide.
pub fn beta_inv(p: f64, a: u64, b: u64) -> f64 {
    assert!(a >= 1 && b >= 1, "beta_inv needs a, b >= 1");
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    while hi - lo > 1e-15 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if beta_inc(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The one-sided Clopper–Pearson lower bound of `correct` out of `accepted` at
/// level `alpha` (`beta.ppf(alpha, k, n − k + 1)`; 0 when `k = 0`).
pub fn clopper_pearson_lower(correct: u64, accepted: u64, alpha: f64) -> f64 {
    assert!(correct <= accepted, "more correct than accepted");
    if correct == 0 {
        0.0
    } else {
        beta_inv(alpha, correct, accepted - correct + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_inc_closed_forms() {
        // I_x(a, 1) = x^a and I_x(1, b) = 1 − (1 − x)^b.
        for &x in &[0.1f64, 0.37, 0.5, 0.93] {
            assert!((beta_inc(5, 1, x) - x.powi(5)).abs() < 1e-14);
            assert!((beta_inc(1, 7, x) - (1.0 - (1.0 - x).powi(7))).abs() < 1e-14);
        }
        assert!((ln_beta_int(3, 4) - (1.0f64 / 60.0).ln()).abs() < 1e-14);
        assert!((beta_inv(0.5, 1, 1) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn fminbound_parabola() {
        let r = fminbound(|x| (x - 0.3) * (x - 0.3), -1.0, 2.0, 1e-5, 500).unwrap();
        assert!(r.success());
        assert!((r.x - 0.3).abs() < 1e-5);
        assert!(fminbound(|x| x, 1.0, 0.0, 1e-5, 500).is_err());
        assert!(fminbound(|x| x, f64::NEG_INFINITY, 0.0, 1e-5, 500).is_err());
    }
}
