//! S1 curves and the S3 fold on the device (f32, the NVG encoder's kernels:
//! exact-FMA GEMM over sub-matrix views, blocked Cholesky). The same algebra
//! as [`super::illum`] — which stays the host reference — at the sizes where
//! the host is the run: Qwen3.8-27B folds 64 layers at n = 17408, plus the
//! n×n factor of every curve.
//!
//! The Cholesky factors are of the Jacobi-equilibrated matrix (unit
//! diagonal, as the device GPTQ fold): f32 then sees correlations, not raw
//! activation energies (outlier neurons are ~10⁴× the median). The refit is
//! solved for the read-out update `Aᵀ` (h right-hand sides) instead of `Π`
//! (p of them), and every error term is a trace against `Σ_SP·W_Pᵀ` or
//! `Σ_SS·Aᵀ`: with h < p on wide FFNs those are the small products.

use super::illum::{FoldInfo, Folded};
use anyhow::{Result, ensure};
use cortiq_engine::gpu_wgpu::nvg_dev::{self as dv, DMat, FrobBatch, INPLACE, LOWER, Pick, V};

/// The device fold kernels compile on this process's wgpu adapter.
/// `CMF_FOLD_ON=host` keeps S1/S3 on the host.
pub fn available() -> bool {
    std::env::var("CMF_FOLD_ON").as_deref() != Ok("host") && dv::available()
}

/// A row-major Gram as f32 on the device.
pub fn upload_gram(g: &[f64], n: usize) -> DMat {
    DMat::from_f64(n, n, g)
}

pub fn upload_gram_f32(g: &[f32], n: usize) -> DMat {
    DMat::from_f32(n, n, g)
}

fn vec_dev(v: &[f64]) -> DMat {
    DMat::from_f64(1, v.len(), v)
}

fn idx_dev(v: &[usize]) -> cortiq_engine::wgpu::Buffer {
    let u: Vec<u32> = v.iter().map(|&x| x as u32).collect();
    dv::upload_u32(&u)
}

/// [`super::illum::refit_curve`] on the device: the jitter steps up from
/// 1e-6 of the mean diagonal until the f32 factor of the equilibrated,
/// importance-ordered Gram exists (f32 needs more than the host's f64 on
/// the weakest directions; the curve only conditions the plan).
pub fn refit_curve(sig: &DMat, diag: &[f64], n: usize, wdown: &DMat, h: usize, ord: &[usize]) -> Result<Vec<f64>> {
    let mean_diag = diag.iter().sum::<f64>() / n as f64;
    let id = idx_dev(ord);
    let pick = Pick { idx: &id, ro: Some(0), co: Some(0) };
    let f = DMat::zeros(n, n);
    let mut d = Vec::new();
    let mut used = None;
    for rel in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let jitter = rel * mean_diag.max(1e-30);
        d = ord.iter().map(|&o| (diag[o] + jitter).max(1e-300).sqrt()).collect();
        let rd = vec_dev(&d.iter().map(|&x| 1.0 / x).collect::<Vec<_>>());
        dv::gather(n, n, (sig, n), Some(&pick), jitter as f32, Some(&rd), Some(&rd), (&f, 0, n));
        if dv::chol_in_place(&f) == Some(true) {
            used = Some(rel);
            break;
        }
    }
    let Some(rel) = used else { anyhow::bail!("Gram is not positive definite even with 1e-2 jitter (device f32 factor)") };
    if rel > 1e-6 {
        eprintln!("  refit curve: jitter {rel:e} of the mean diagonal (device f32 factor)");
    }
    // the factor's strict upper triangle is undefined: zero it
    dv::scale_rc(n, n, None, (&f, 0, n), None, None, INPLACE | LOWER, 1.0, 0.0);
    // W in importance order, columns scaled by d (Σ = D·F·D, so W·L = (W·D)·L_F)
    let wp = DMat::zeros(h, n);
    let dd = vec_dev(&d);
    let cols = Pick { idx: &id, ro: None, co: Some(0) };
    dv::gather(h, n, (wdown, n), Some(&cols), 0.0, None, Some(&dd), (&wp, 0, n));
    let m = DMat::zeros(h, n);
    dv::gemm_v(h, n, n, 1.0, V::n(&wp), V::n(&f), 0.0, &m, 0, n, None);
    let colsq = dv::colsum_prod(h, n, &m, 0, n, &m, 0, n);
    let mut c = vec![0f64; n + 1];
    for k in (0..n).rev() {
        c[k] = c[k + 1] + colsq[k];
    }
    let tot = c[0].max(1e-300);
    let mut out: Vec<f64> = c.iter().map(|v| v / tot).collect();
    out.push(tot);
    Ok(out)
}

/// What the error terms read from one Gram, in the transposed layout the
/// refit is solved in (`X = Aᵀ`, k×h): `Σ_SS`, `C = Σ_SP·W_Pᵀ`, and the
/// ridge-independent traces.
struct Side {
    ss: DMat,
    c: DMat,
    /// `‖W_P‖²_Σ_PP`: the plain cut's error
    e_pp: f64,
    /// `‖W‖²_Σ`: the layer's output energy
    full: f64,
}

#[allow(clippy::too_many_arguments)]
fn side(g: &DMat, n: usize, id: &cortiq_engine::wgpu::Buffer, k: usize, p: usize, h: usize, wst: &DMat, wpt: &DMat) -> Side {
    let ss = DMat::zeros(k, k);
    dv::gather(k, k, (g, n), Some(&Pick { idx: id, ro: Some(0), co: Some(0) }), 0.0, None, None, (&ss, 0, k));
    let xpp = DMat::zeros(p, h);
    {
        let pp = DMat::zeros(p, p);
        dv::gather(p, p, (g, n), Some(&Pick { idx: id, ro: Some(k), co: Some(k) }), 0.0, None, None, (&pp, 0, p));
        dv::gemm_v(p, h, p, 1.0, V::n(&pp), V::n(wpt), 0.0, &xpp, 0, h, None);
    }
    let c = DMat::zeros(k, h);
    {
        let sp = DMat::zeros(k, p);
        dv::gather(k, p, (g, n), Some(&Pick { idx: id, ro: Some(0), co: Some(k) }), 0.0, None, None, (&sp, 0, p));
        dv::gemm_v(k, h, p, 1.0, V::n(&sp), V::n(wpt), 0.0, &c, 0, h, None);
    }
    let xss = DMat::zeros(k, h);
    dv::gemm_v(k, h, k, 1.0, V::n(&ss), V::n(wst), 0.0, &xss, 0, h, None);
    let mut fb = FrobBatch::new();
    fb.add(&xpp, wpt, p * h);
    fb.add(&c, wst, k * h);
    fb.add(&xss, wst, k * h);
    let e = fb.read();
    Side { ss, c, e_pp: e[0], full: (e[2] + 2.0 * e[1] + e[0]).max(1e-300) }
}

/// `‖W_P − A‖²_Σ` of the refit `X = Aᵀ`: `e_pp − 2⟨C, X⟩ + ⟨Σ_SS·X, X⟩`.
fn refit_err(s: &Side, x: &DMat, k: usize, h: usize) -> f64 {
    let y = DMat::zeros(k, h);
    dv::gemm_v(k, h, k, 1.0, V::n(&s.ss), V::n(x), 0.0, &y, 0, h, None);
    let mut fb = FrobBatch::new();
    fb.add(&s.c, x, k * h);
    fb.add(&y, x, k * h);
    let e = fb.read();
    s.e_pp - 2.0 * e[0] + e[1]
}

/// `L·Lᵀ·X = Z` for `Z [k×h]`, the factor and its 128-block inverses as
/// [`dv::chol_blocked_inv`] leaves them: forward and backward block
/// substitution, every step one GEMM. The solution ends in `z` (`y` is
/// scratch of the same shape).
fn llt_solve(l: &DMat, inv: &DMat, z: &DMat, y: &DMat, k: usize, h: usize) {
    let nb = k.div_ceil(128);
    let blk = |b: usize| (b * 128, 128.min(k - b * 128), b * 128 * 128);
    // L·Y = Z: each block's rows solved, then taken out of the rows below
    for b in 0..nb {
        let (r0, bs, io) = blk(b);
        let r1 = r0 + bs;
        dv::gemm_v(bs, h, bs, 1.0, V { m: inv, off: io, ld: 128, t: false }, V { m: z, off: r0 * h, ld: h, t: false }, 0.0, y, r0 * h, h, None);
        if r1 < k {
            dv::gemm_v(k - r1, h, bs, -1.0, V { m: l, off: r1 * k + r0, ld: k, t: false }, V { m: y, off: r0 * h, ld: h, t: false }, 1.0, z, r1 * h, h, None);
        }
    }
    // Lᵀ·X = Y: from the last block up
    for b in (0..nb).rev() {
        let (r0, bs, io) = blk(b);
        dv::gemm_v(bs, h, bs, 1.0, V { m: inv, off: io, ld: 128, t: true }, V { m: y, off: r0 * h, ld: h, t: false }, 0.0, z, r0 * h, h, None);
        if r0 > 0 {
            dv::gemm_v(r0, h, bs, -1.0, V { m: l, off: r0 * k, ld: k, t: true }, V { m: z, off: r0 * h, ld: h, t: false }, 1.0, y, 0, h, None);
        }
    }
}

/// [`super::illum::fold_layer`] on the device: `train`/`val` are the
/// layer's Grams already there (`diag` the train Gram's diagonal). The
/// refit `Π = (Σ_SS + εI)⁻¹Σ_SP` is never formed: its read-out update
/// `Aᵀ = (Σ_SS + εI)⁻¹·(Σ_SP·W_Pᵀ)` is solved for h right-hand sides.
#[allow(clippy::too_many_arguments)]
pub fn fold_layer(
    layer: usize,
    train: &DMat,
    diag: &[f64],
    val: Option<&DMat>,
    n: usize,
    wdown: &[f32],
    h: usize,
    ord: &[usize],
    k: usize,
    ridges: &[f64],
    allow_cut: bool,
) -> Result<Folded> {
    let mut keep: Vec<usize> = ord[..k].to_vec();
    keep.sort_unstable();
    let mut drop_: Vec<usize> = ord[k..].to_vec();
    drop_.sort_unstable();
    let p = drop_.len();
    if p == 0 {
        let info = FoldInfo { layer, width: n, keep: k, ridge: None, val_err_cut: 0.0, val_err: 0.0, train_err: 0.0 };
        let down = (0..h).flat_map(|r| keep.iter().map(move |&c| wdown[r * n + c])).collect();
        return Ok(Folded { keep, down, info });
    }
    let t0 = std::time::Instant::now();
    let prof = std::env::var("CMF_FOLD_PROF").as_deref() == Ok("1");
    let lap = |name: &str, t: &mut std::time::Instant| {
        if prof {
            dv::sync();
            eprint!(" {name} {:.2}", t.elapsed().as_secs_f64());
            *t = std::time::Instant::now();
        }
    };
    let mut t = std::time::Instant::now();
    // one index buffer: keep at 0, drop at k
    let all: Vec<usize> = keep.iter().chain(&drop_).copied().collect();
    let id = idx_dev(&all);
    // Wᵀ's kept and dropped rows (k×h, p×h)
    let wst = DMat::zeros(k, h);
    let wpt = DMat::zeros(p, h);
    {
        let wt = dv::transpose(&DMat::from_f32(h, n, wdown));
        dv::gather(k, h, (&wt, h), Some(&Pick { idx: &id, ro: Some(0), co: None }), 0.0, None, None, (&wst, 0, h));
        dv::gather(p, h, (&wt, h), Some(&Pick { idx: &id, ro: Some(k), co: None }), 0.0, None, None, (&wpt, 0, h));
    }
    let tr = side(train, n, &id, k, p, h, &wst, &wpt);
    let va = val.map(|v| side(v, n, &id, k, p, h, &wst, &wpt));
    let sc = va.as_ref().unwrap_or(&tr);
    lap("energy", &mut t);
    let cut_err = sc.e_pp / sc.full;
    let mean_diag = keep.iter().map(|&i| diag[i]).sum::<f64>() / k as f64;
    let mut best: (f64, Option<f64>, Option<DMat>) = (if allow_cut { cut_err } else { f64::INFINITY }, None, None);
    let f = DMat::zeros(k, k);
    let inv = DMat::zeros(1, dv::chol_inv_len(k));
    let z = DMat::zeros(k, h);
    let y = DMat::zeros(k, h);
    for &rho in ridges {
        let ridge = rho * mean_diag.max(1e-30);
        let rd: Vec<f64> = keep.iter().map(|&i| 1.0 / (diag[i] + ridge).max(1e-300).sqrt()).collect();
        let rd = vec_dev(&rd);
        // F = D⁻¹(Σ_SS + εI)D⁻¹ = L·Lᵀ
        dv::gather(k, k, (train, n), Some(&Pick { idx: &id, ro: Some(0), co: Some(0) }), ridge as f32, Some(&rd), Some(&rd), (&f, 0, k));
        let ok = dv::chol_blocked_inv(&f, Some(&inv)) == Some(true);
        lap("chol", &mut t);
        if !ok {
            continue;
        }
        // X = D⁻¹·F⁻¹·D⁻¹·C
        dv::scale_rc(k, h, Some((&tr.c, 0, h)), (&z, 0, h), Some(&rd), None, 0, 1.0, 0.0);
        llt_solve(&f, &inv, &z, &y, k, h);
        let x = DMat::zeros(k, h);
        dv::scale_rc(k, h, Some((&z, 0, h)), (&x, 0, h), Some(&rd), None, 0, 1.0, 0.0);
        lap("solve", &mut t);
        let e = refit_err(sc, &x, k, h) / sc.full;
        lap("err", &mut t);
        if e < best.0 {
            best = (e, Some(rho), Some(x));
        }
    }
    let (val_err, ridge, x) = best;
    ensure!(val_err.is_finite(), "layer {layer}: no refit was positive definite and the plain cut is disallowed");
    let train_err = match &x {
        Some(x) => {
            let te = refit_err(&tr, x, k, h) / tr.full;
            dv::axpby(1.0, x, 1.0, &wst);
            te
        }
        None => tr.e_pp / tr.full,
    };
    // W_S' = (Wᵀ_S + X)ᵀ, back to h×k on the host
    let dt = wst.to_f32();
    let mut down = vec![0f32; h * k];
    for (i, row) in dt.chunks_exact(h).enumerate() {
        for (r, &v) in row.iter().enumerate() {
            down[r * k + i] = v;
        }
    }
    lap("apply", &mut t);
    if prof {
        eprintln!();
        eprintln!("  fold L{layer} {n}→{k} (device): {:.2}s", t0.elapsed().as_secs_f64());
    }
    let info = FoldInfo { layer, width: n, keep: k, ridge, val_err_cut: cut_err, val_err, train_err };
    Ok(Folded { keep, down, info })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    /// Correlated activations with a few outlier neurons (Qwen's FFN has
    /// channels ~10⁴× the median energy) so the equilibration matters.
    fn fixture(n: usize, h: usize, t: usize) -> (Vec<f64>, Vec<f64>, Vec<f32>) {
        let mut s = 11u64;
        let base: Vec<f32> = (0..n * 8).map(|_| lcg(&mut s)).collect();
        let gram = |s: &mut u64| {
            let mut sig = vec![0f64; n * n];
            for _ in 0..t {
                let z: Vec<f32> = (0..8).map(|_| lcg(s)).collect();
                let a: Vec<f64> = (0..n)
                    .map(|j| {
                        let v = (0..8).map(|q| base[j * 8 + q] * z[q]).sum::<f32>() as f64 + 0.05 * lcg(s) as f64;
                        if j % 37 == 5 { v * 60.0 } else { v }
                    })
                    .collect();
                for i in 0..n {
                    for j in 0..n {
                        sig[i * n + j] += a[i] * a[j];
                    }
                }
            }
            sig
        };
        let tr = gram(&mut s);
        let va = gram(&mut s);
        let w: Vec<f32> = (0..h * n).map(|_| lcg(&mut s)).collect();
        (tr, va, w)
    }

    /// The device fold reproduces the host fold (f64 Cholesky, the dropped
    /// Gram's error form) to f32 accuracy: same ridge choice, the errors and
    /// the folded read-out within 1e-3 relative.
    #[test]
    fn device_fold_matches_host() {
        if !available() {
            eprintln!("no wgpu device: skipped");
            return;
        }
        let (n, h) = (300, 96);
        let (tr, va, w) = fixture(n, h, 1200);
        let imp: Vec<f64> = (0..n).map(|j| tr[j * n + j]).collect();
        let ord = super::super::illum::order(&imp);
        let diag: Vec<f64> = (0..n).map(|j| tr[j * n + j]).collect();
        let (dtr, dva) = (upload_gram(&tr, n), upload_gram(&va, n));
        for k in [96, 200, 256] {
            let host = super::super::illum::fold_layer(0, &tr, Some(&va), n, &w, h, &ord, k, &[0.03, 0.1, 0.3], true, 2).unwrap();
            let dev = fold_layer(0, &dtr, &diag, Some(&dva), n, &w, h, &ord, k, &[0.03, 0.1, 0.3], true).unwrap();
            assert_eq!(host.keep, dev.keep);
            assert_eq!(host.info.ridge, dev.info.ridge, "k={k}: ridge choice");
            for (a, b, what) in [
                (host.info.val_err, dev.info.val_err, "val"),
                (host.info.val_err_cut, dev.info.val_err_cut, "cut"),
                (host.info.train_err, dev.info.train_err, "train"),
            ] {
                assert!((a - b).abs() <= 1e-3 * a.abs().max(1e-3), "k={k} {what}: host {a} vs device {b}");
            }
            let (num, den) = host.down.iter().zip(&dev.down).fold((0f64, 0f64), |(n2, d2), (&x, &y)| (n2 + ((x - y) as f64).powi(2), d2 + (x as f64).powi(2)));
            assert!((num / den).sqrt() < 1e-3, "k={k}: folded read-out differs by {:.2e}", (num / den).sqrt());
        }
    }

    /// One layer at a chosen geometry (default Qwen3.8-27B: n 17408, h 5120)
    /// on a synthetic Gram: `CMF_FOLD_BENCH="n,h,k1,k2,…"`, `CMF_FOLD_BENCH_HOST=1`
    /// also times the host fold (f64) at the first k.
    #[test]
    #[ignore]
    fn bench_fold_layer() {
        if !available() {
            eprintln!("no wgpu device: skipped");
            return;
        }
        let spec: Vec<usize> = std::env::var("CMF_FOLD_BENCH")
            .unwrap_or_else(|_| "17408,5120,4352,10496".into())
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        let (n, h, ks) = (spec[0], spec[1], &spec[2..]);
        let t = n + n / 4;
        let mut s = 3u64;
        // Σ = XᵀX over t random rows (some outlier columns), built on the device
        let xs: Vec<f32> = (0..t * n).map(|i| lcg(&mut s) * if i % n % 97 == 3 { 40.0 } else { 1.0 }).collect();
        let x = DMat::from_f32(t, n, &xs);
        drop(xs);
        let g = DMat::zeros(n, n);
        let t0 = std::time::Instant::now();
        dv::gemm_v(n, n, t, 1.0, V::t(&x), V::n(&x), 0.0, &g, 0, n, None);
        dv::sync();
        eprintln!("Gram {n}×{n} over {t} rows: {:.2}s", t0.elapsed().as_secs_f64());
        drop(x);
        let diag = dv::diag(&g, n, 0, n);
        let w: Vec<f32> = (0..h * n).map(|_| lcg(&mut s)).collect();
        let imp: Vec<f64> = diag.clone();
        let ord = super::super::illum::order(&imp);
        let t0 = std::time::Instant::now();
        let c = refit_curve(&g, &diag, n, &DMat::from_f32(h, n, &w), h, &ord).unwrap();
        dv::sync();
        eprintln!("curve: {:.2}s (c[n/2] {:.4})", t0.elapsed().as_secs_f64(), c[n / 2]);
        for &k in ks {
            let t0 = std::time::Instant::now();
            let f = fold_layer(0, &g, &diag, Some(&g), n, &w, h, &ord, k, &[0.03, 0.1, 0.3], true).unwrap();
            eprintln!("fold {n}→{k} (device): {:.2}s · ridge {:?} · val {:.4} cut {:.4}", t0.elapsed().as_secs_f64(), f.info.ridge, f.info.val_err, f.info.val_err_cut);
        }
        if std::env::var("CMF_FOLD_BENCH_HOST").as_deref() == Ok("1") {
            let gh = g.to_f64();
            drop(g);
            let t0 = std::time::Instant::now();
            let f = super::super::illum::fold_layer(0, &gh, Some(&gh), n, &w, h, &ord, ks[0], &[0.03, 0.1, 0.3], true, crate::compress::stages::cpu_budget()).unwrap();
            eprintln!("fold {n}→{} (host): {:.2}s · ridge {:?} · val {:.4}", ks[0], t0.elapsed().as_secs_f64(), f.info.ridge, f.info.val_err);
            let t0 = std::time::Instant::now();
            let c = super::super::illum::refit_curve(&gh, n, &w, h, &ord, crate::compress::stages::cpu_budget()).unwrap();
            eprintln!("curve (host): {:.2}s (c[n/2] {:.4})", t0.elapsed().as_secs_f64(), c[n / 2]);
        }
    }

    #[test]
    fn device_curve_matches_host() {
        if !available() {
            eprintln!("no wgpu device: skipped");
            return;
        }
        let (n, h) = (256, 64);
        let (tr, _, w) = fixture(n, h, 1000);
        let imp: Vec<f64> = (0..n).map(|j| tr[j * n + j]).collect();
        let ord = super::super::illum::order(&imp);
        let diag: Vec<f64> = (0..n).map(|j| tr[j * n + j]).collect();
        let host = super::super::illum::refit_curve(&tr, n, &w, h, &ord, 2).unwrap();
        let dev = refit_curve(&upload_gram(&tr, n), &diag, n, &DMat::from_f32(h, n, &w), h, &ord).unwrap();
        assert!((host[n + 1] - dev[n + 1]).abs() <= 1e-3 * host[n + 1], "energy {} vs {}", host[n + 1], dev[n + 1]);
        for kk in [32, 128, 200] {
            assert!((host[kk] - dev[kk]).abs() <= 2e-3 * host[kk].max(1e-3), "k={kk}: host {} vs device {}", host[kk], dev[kk]);
        }
    }
}
