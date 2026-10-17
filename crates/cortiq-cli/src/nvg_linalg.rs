//! Dense f64 linear algebra for the NVG encoder (no LAPACK): parallel
//! GEMM over `matrixmultiply`, blocked Cholesky, triangular inverse, a
//! symmetric eigensolver (Householder tridiagonalisation + implicit QL, the
//! EISPACK `tred2`/`tql2` scheme with the O(n³) loops spread over a pool),
//! and the polar factor. Row-major throughout.

use cortiq_engine::matrixmultiply;
use cortiq_engine::pool::Pool;

#[derive(Clone, Debug)]
pub struct Mat {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f64>,
}

impl Mat {
    pub fn zeros(r: usize, c: usize) -> Self {
        Mat {
            r,
            c,
            d: vec![0.0; r * c],
        }
    }
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.d[i * n + i] = 1.0;
        }
        m
    }
    pub fn from_vec(r: usize, c: usize, d: Vec<f64>) -> Self {
        assert_eq!(d.len(), r * c);
        Mat { r, c, d }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let c = self.c;
        &mut self.d[i * c..(i + 1) * c]
    }
    pub fn t(&self) -> Mat {
        let mut o = Mat::zeros(self.c, self.r);
        for i in 0..self.r {
            for j in 0..self.c {
                o.d[j * self.r + i] = self.d[i * self.c + j];
            }
        }
        o
    }
    pub fn sub(&self, o: &Mat) -> Mat {
        assert!(self.r == o.r && self.c == o.c);
        Mat {
            r: self.r,
            c: self.c,
            d: self.d.iter().zip(&o.d).map(|(a, b)| a - b).collect(),
        }
    }
    pub fn diag(&self) -> Vec<f64> {
        (0..self.r.min(self.c)).map(|i| self.at(i, i)).collect()
    }
    pub fn frob2(&self) -> f64 {
        self.d.iter().map(|v| v * v).sum()
    }
}

#[derive(Clone, Copy)]
struct SP(*mut f64);
unsafe impl Send for SP {}
unsafe impl Sync for SP {}

// Passing the wrapper by value keeps parallel closures from precisely capturing
// its non-Sync raw-pointer field.
#[inline]
fn sp_ptr(sp: SP) -> *mut f64 {
    sp.0
}

fn workers(p: Option<&Pool>) -> usize {
    p.map(|p| p.n_workers() + 1).unwrap_or(1)
}

/// Run `f(lo, hi)` over `0..n` in chunks of at least `min_chunk`.
pub fn par_chunks(p: Option<&Pool>, n: usize, min_chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    if n == 0 {
        return;
    }
    let w = workers(p);
    let cs = n.div_ceil(w * 2).max(min_chunk).max(1);
    let nch = n.div_ceil(cs);
    match p {
        Some(pool) if nch > 1 => pool.run_rows(nch, &|s, e| {
            for ch in s..e {
                let lo = ch * cs;
                let hi = ((ch + 1) * cs).min(n);
                if lo < hi {
                    f(lo, hi);
                }
            }
        }),
        _ => f(0, n),
    }
}

/// `C = alpha · op(A) · op(B) + beta · C` (row-major; `op` = transpose when the
/// flag is set). Parallel over the larger output dimension.
#[allow(clippy::too_many_arguments)]
pub fn gemm_into(
    p: Option<&Pool>,
    alpha: f64,
    a: &Mat,
    ta: bool,
    b: &Mat,
    tb: bool,
    beta: f64,
    c: &mut Mat,
) {
    let (m, k) = if ta { (a.c, a.r) } else { (a.r, a.c) };
    let (k2, n) = if tb { (b.c, b.r) } else { (b.r, b.c) };
    assert_eq!(k, k2, "gemm inner dims");
    assert!(c.r == m && c.c == n, "gemm output dims");
    if m == 0 || n == 0 {
        return;
    }
    // strides of op(A) [m×k] and op(B) [k×n]
    let (rsa, csa) = if ta {
        (1isize, a.c as isize)
    } else {
        (a.c as isize, 1isize)
    };
    let (rsb, csb) = if tb {
        (1isize, b.c as isize)
    } else {
        (b.c as isize, 1isize)
    };
    let cp = SP(c.d.as_mut_ptr());
    let ldc = n as isize;
    let split_m = m >= n;
    let flops = m * n * k.max(1);
    let run = |lo: usize, hi: usize| {
        let cp = cp;
        unsafe {
            if split_m {
                matrixmultiply::dgemm(
                    hi - lo,
                    k,
                    n,
                    alpha,
                    a.d.as_ptr().offset(lo as isize * rsa),
                    rsa,
                    csa,
                    b.d.as_ptr(),
                    rsb,
                    csb,
                    beta,
                    cp.0.offset(lo as isize * ldc),
                    ldc,
                    1,
                );
            } else {
                matrixmultiply::dgemm(
                    m,
                    k,
                    hi - lo,
                    alpha,
                    a.d.as_ptr(),
                    rsa,
                    csa,
                    b.d.as_ptr().offset(lo as isize * csb),
                    rsb,
                    csb,
                    beta,
                    cp.0.offset(lo as isize),
                    ldc,
                    1,
                );
            }
        }
    };
    if flops < 1 << 18 {
        run(0, if split_m { m } else { n });
    } else {
        par_chunks(p, if split_m { m } else { n }, 32, &run);
    }
}

pub fn gemm(p: Option<&Pool>, a: &Mat, ta: bool, b: &Mat, tb: bool) -> Mat {
    let m = if ta { a.c } else { a.r };
    let n = if tb { b.r } else { b.c };
    let mut c = Mat::zeros(m, n);
    gemm_into(p, 1.0, a, ta, b, tb, 0.0, &mut c);
    c
}

/// Lower Cholesky factor of an SPD matrix (blocked, right-looking). `None`
/// when a pivot is not positive.
pub fn cholesky(p: Option<&Pool>, a: &Mat) -> Option<Mat> {
    let n = a.r;
    assert_eq!(a.c, n);
    let mut l = a.clone();
    let nb = 96usize;
    let lp = SP(l.d.as_mut_ptr());
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + nb).min(n);
        // diagonal block (unblocked)
        for j in k0..k1 {
            let mut s = l.d[j * n + j];
            for t in k0..j {
                s -= l.d[j * n + t] * l.d[j * n + t];
            }
            if !(s > 0.0) || !s.is_finite() {
                return None;
            }
            let dj = s.sqrt();
            l.d[j * n + j] = dj;
            for i in j + 1..k1 {
                let mut v = l.d[i * n + j];
                for t in k0..j {
                    v -= l.d[i * n + t] * l.d[j * n + t];
                }
                l.d[i * n + j] = v / dj;
            }
        }
        if k1 < n {
            // panel rows below: L21 = A21 L11^{-T}
            par_chunks(p, n - k1, 16, &|lo, hi| {
                let lp = lp;
                let ld = unsafe { std::slice::from_raw_parts_mut(lp.0, n * n) };
                for i in k1 + lo..k1 + hi {
                    for j in k0..k1 {
                        let mut v = ld[i * n + j];
                        for t in k0..j {
                            v -= ld[i * n + t] * ld[j * n + t];
                        }
                        ld[i * n + j] = v / ld[j * n + j];
                    }
                }
            });
            // trailing update, lower block-rows: A[i0..i1, k1..i1] -= L21[i0..i1] L21[k1..i1]ᵀ
            let rb = 64usize;
            let nrb = (n - k1).div_ceil(rb);
            let job = |s: usize, e: usize| {
                let lp = lp;
                for blk in s..e {
                    let i0 = k1 + blk * rb;
                    let i1 = (i0 + rb).min(n);
                    unsafe {
                        matrixmultiply::dgemm(
                            i1 - i0,
                            k1 - k0,
                            i1 - k1,
                            -1.0,
                            lp.0.add(i0 * n + k0),
                            n as isize,
                            1,
                            lp.0.add(k1 * n + k0),
                            1,
                            n as isize,
                            1.0,
                            lp.0.add(i0 * n + k1),
                            n as isize,
                            1,
                        );
                    }
                }
            };
            match p {
                Some(pool) if nrb > 1 => pool.run_rows(nrb, &job),
                _ => job(0, nrb),
            }
        }
        k0 = k1;
    }
    for i in 0..n {
        for j in i + 1..n {
            l.d[i * n + j] = 0.0;
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b`.
pub fn chol_solve(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.r;
    let mut y = b.to_vec();
    for i in 0..n {
        let row = l.row(i);
        let mut v = y[i];
        for t in 0..i {
            v -= row[t] * y[t];
        }
        y[i] = v / row[i];
    }
    for i in (0..n).rev() {
        let mut v = y[i];
        for t in i + 1..n {
            v -= l.d[t * n + i] * y[t];
        }
        y[i] = v / l.d[i * n + i];
    }
    y
}

/// `(A + rel·tr(A)/n·I) x = b` by Cholesky; the ridge grows ×100 on a failed
/// factorisation (the prototype's lstsq fallback never fired on real layers).
pub fn solve_spd(p: Option<&Pool>, a: &Mat, b: &[f64], rel: f64) -> Vec<f64> {
    let n = a.r;
    let tr = (0..n).map(|i| a.at(i, i)).sum::<f64>() / n.max(1) as f64;
    let mut ridge = rel * tr.max(1e-300);
    loop {
        let mut m = a.clone();
        for i in 0..n {
            m.d[i * n + i] += ridge;
        }
        if let Some(l) = cholesky(p, &m) {
            return chol_solve(&l, b);
        }
        ridge = (ridge * 100.0).max(1e-300);
        if ridge > tr.abs() * 10.0 + 1e-200 {
            return vec![0.0; n];
        }
    }
}

/// Inverse of a lower-triangular matrix (lower).
pub fn tri_lower_inverse(p: Option<&Pool>, l: &Mat) -> Mat {
    let n = l.r;
    // rows of Y = L^{-T}... computed as columns of X = L^{-1}, stored as rows of Y = Xᵀ
    let mut y = Mat::zeros(n, n);
    let yp = SP(y.d.as_mut_ptr());
    par_chunks(p, n, 8, &|lo, hi| {
        let yp = yp;
        for j in lo..hi {
            let row = unsafe { std::slice::from_raw_parts_mut(yp.0.add(j * n), n) };
            row[j] = 1.0 / l.d[j * n + j];
            for i in j + 1..n {
                let lr = &l.d[i * n..i * n + i];
                let mut v = 0.0;
                for t in j..i {
                    v += lr[t] * row[t];
                }
                row[i] = -v / l.d[i * n + i];
            }
        }
    });
    y.t()
}

/// Symmetric eigendecomposition: eigenvalues in DESCENDING order and the
/// eigenvectors as the columns of the returned matrix.
pub fn sym_eig(p: Option<&Pool>, a_in: &Mat) -> (Vec<f64>, Mat) {
    let n = a_in.r;
    assert_eq!(a_in.c, n);
    if n == 1 {
        return (vec![a_in.d[0]], Mat::eye(1));
    }
    let mut a = a_in.clone();
    // symmetrise (the inputs are Gram matrices; guard against drift)
    for i in 0..n {
        for j in i + 1..n {
            let v = 0.5 * (a.d[i * n + j] + a.d[j * n + i]);
            a.d[i * n + j] = v;
            a.d[j * n + i] = v;
        }
    }
    let mut d = vec![0f64; n];
    let mut e = vec![0f64; n];
    let mut refl: Vec<(Vec<f64>, f64)> = vec![(Vec::new(), 0.0); n];
    let ap = SP(a.d.as_mut_ptr());
    for i in (1..n).rev() {
        let scale: f64 = a.row(i)[..i].iter().map(|v| v.abs()).sum();
        if scale == 0.0 {
            e[i] = a.d[i * n + i - 1];
            d[i] = a.d[i * n + i];
            continue;
        }
        let mut u: Vec<f64> = a.row(i)[..i].iter().map(|v| v / scale).collect();
        let mut h: f64 = u.iter().map(|v| v * v).sum();
        let f = u[i - 1];
        let g = if f > 0.0 { -h.sqrt() } else { h.sqrt() };
        e[i] = scale * g;
        h -= f * g;
        u[i - 1] = f - g;
        // p = A u / h over the leading i×i block
        let mut pv = vec![0f64; i];
        {
            let pp = SP(pv.as_mut_ptr());
            let u = &u;
            let run = |lo: usize, hi: usize| {
                let pp = pp;
                let ad = unsafe { std::slice::from_raw_parts(sp_ptr(ap) as *const f64, n * n) };
                for k in lo..hi {
                    let row = &ad[k * n..k * n + i];
                    let mut s = 0.0;
                    for t in 0..i {
                        s += row[t] * u[t];
                    }
                    unsafe { *sp_ptr(pp).add(k) = s / h };
                }
            };
            if i >= 128 {
                par_chunks(p, i, 32, &run);
            } else {
                run(0, i);
            }
        }
        let kk: f64 = u.iter().zip(&pv).map(|(a, b)| a * b).sum::<f64>() / (2.0 * h);
        let q: Vec<f64> = pv.iter().zip(&u).map(|(pv, u)| pv - kk * u).collect();
        {
            let (u, q) = (&u, &q);
            let run = |lo: usize, hi: usize| {
                let ap = ap;
                let ad = unsafe { std::slice::from_raw_parts_mut(ap.0, n * n) };
                for k in lo..hi {
                    let (uk, qk) = (u[k], q[k]);
                    let row = &mut ad[k * n..k * n + i];
                    for t in 0..i {
                        row[t] -= uk * q[t] + qk * u[t];
                    }
                }
            };
            if i >= 128 {
                par_chunks(p, i, 32, &run);
            } else {
                run(0, i);
            }
        }
        d[i] = a.d[i * n + i];
        refl[i] = (u, h);
    }
    d[0] = a.d[0];
    e[0] = 0.0;
    drop(a);
    // Q = P_{n-1} ⋯ P_1, built by applying P_1, P_2, … to I from the left.
    let mut v = Mat::eye(n);
    let vp = SP(v.d.as_mut_ptr());
    for i in 1..n {
        let (u, h) = (&refl[i].0, refl[i].1);
        if u.is_empty() || h == 0.0 {
            continue;
        }
        // w[t] = Σ_{k<i} u[k] Q[k][t], t < i
        let mut w = vec![0f64; i];
        for k in 0..i {
            let uk = u[k];
            if uk == 0.0 {
                continue;
            }
            let row = &v.d[k * n..k * n + i];
            for t in 0..i {
                w[t] += uk * row[t];
            }
        }
        let w = &w;
        let run = |lo: usize, hi: usize| {
            let vp = vp;
            let vd = unsafe { std::slice::from_raw_parts_mut(vp.0, n * n) };
            for k in lo..hi {
                let c = u[k] / h;
                let row = &mut vd[k * n..k * n + i];
                for t in 0..i {
                    row[t] -= c * w[t];
                }
            }
        };
        if i >= 128 {
            par_chunks(p, i, 32, &run);
        } else {
            run(0, i);
        }
    }
    // tql2 (implicit QL) with the rotations applied row-parallel.
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    let eps = 2f64.powi(-52);
    let mut rots: Vec<(usize, f64, f64)> = Vec::with_capacity(n);
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m == n {
            m = n - 1;
        }
        if m > l {
            let mut iter = 0;
            loop {
                iter += 1;
                let g = d[l];
                let mut pp = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = pp.hypot(1.0);
                if pp < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (pp + r);
                d[l + 1] = e[l] * (pp + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;
                pp = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                rots.clear();
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * pp;
                    r = pp.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = pp / r;
                    pp = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    rots.push((i, c, s));
                }
                {
                    let rots = &rots;
                    let run = |lo: usize, hi: usize| {
                        let vp = vp;
                        let vd = unsafe { std::slice::from_raw_parts_mut(vp.0, n * n) };
                        for k in lo..hi {
                            let row = &mut vd[k * n..(k + 1) * n];
                            for &(i, c, s) in rots {
                                let h = row[i + 1];
                                row[i + 1] = s * row[i] + c * h;
                                row[i] = c * row[i] - s * h;
                            }
                        }
                    };
                    if rots.len() * n >= 1 << 15 {
                        par_chunks(p, n, 32, &run);
                    } else {
                        run(0, n);
                    }
                }
                pp = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * pp;
                d[l] = c * pp;
                if e[l].abs() <= eps * tst1 || iter > 60 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    // descending order
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));
    let vals: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let mut out = Mat::zeros(n, n);
    for k in 0..n {
        let src = &v.d[k * n..(k + 1) * n];
        let dst = &mut out.d[k * n..(k + 1) * n];
        for (j, &i) in idx.iter().enumerate() {
            dst[j] = src[i];
        }
    }
    (vals, out)
}

/// Orthogonal polar factor `R = U Vᵀ` of `M = U Σ Vᵀ` (square), through the
/// eigendecomposition of `MᵀM`.
pub fn polar(p: Option<&Pool>, m: &Mat) -> Mat {
    let n = m.r;
    assert_eq!(m.c, n);
    let g = gemm(p, m, true, m, false);
    let (lam, e) = sym_eig(p, &g);
    let lmax = lam.first().copied().unwrap_or(0.0).max(1e-300);
    let mut me = gemm(p, m, false, &e, false);
    for j in 0..n {
        let s = lam[j].max(lmax * 1e-24).sqrt();
        for i in 0..n {
            me.d[i * n + j] /= s;
        }
    }
    gemm(p, &me, false, &e, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn eig_reconstructs_and_is_orthogonal() {
        let pool = Pool::new(3);
        let mut s = 5u64;
        for &n in &[1usize, 2, 7, 64, 200] {
            let x = Mat::from_vec(n, n + 3, (0..n * (n + 3)).map(|_| rnd(&mut s)).collect());
            let a = gemm(Some(&pool), &x, false, &x, true);
            let (lam, v) = sym_eig(Some(&pool), &a);
            for w in lam.windows(2) {
                assert!(w[0] >= w[1]);
            }
            // A v_j = λ_j v_j and VᵀV = I
            let av = gemm(None, &a, false, &v, false);
            let vtv = gemm(None, &v, true, &v, false);
            let scale = lam[0].abs().max(1e-30);
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (av.at(i, j) - lam[j] * v.at(i, j)).abs() < 1e-9 * scale,
                        "n{n} {i},{j}"
                    );
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((vtv.at(i, j) - want).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn cholesky_solve_inverse_polar() {
        let pool = Pool::new(3);
        let mut s = 9u64;
        let n = 300;
        let x = Mat::from_vec(n, n + 10, (0..n * (n + 10)).map(|_| rnd(&mut s)).collect());
        let mut a = gemm(Some(&pool), &x, false, &x, true);
        for i in 0..n {
            a.d[i * n + i] += 0.1;
        }
        let l = cholesky(Some(&pool), &a).unwrap();
        let llt = gemm(None, &l, false, &l, true);
        for i in 0..n * n {
            assert!((llt.d[i] - a.d[i]).abs() < 1e-9 * (1.0 + a.d[i].abs()));
        }
        let b: Vec<f64> = (0..n).map(|_| rnd(&mut s)).collect();
        let xs = chol_solve(&l, &b);
        for i in 0..n {
            let v: f64 = (0..n).map(|j| a.at(i, j) * xs[j]).sum();
            assert!((v - b[i]).abs() < 1e-8);
        }
        let li = tri_lower_inverse(Some(&pool), &l);
        let id = gemm(None, &li, false, &l, false);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id.at(i, j) - want).abs() < 1e-8);
            }
        }
        let m = Mat::from_vec(40, 40, (0..1600).map(|_| rnd(&mut s)).collect());
        let r = polar(Some(&pool), &m);
        let rtr = gemm(None, &r, true, &r, false);
        for i in 0..40 {
            for j in 0..40 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr.at(i, j) - want).abs() < 1e-8);
            }
        }
        // RᵀM is symmetric positive semidefinite
        let s_ = gemm(None, &r, true, &m, false);
        for i in 0..40 {
            for j in 0..40 {
                assert!((s_.at(i, j) - s_.at(j, i)).abs() < 1e-8);
            }
        }
    }
}
