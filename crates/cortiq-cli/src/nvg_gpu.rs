//! The NVG encoder on the device (wgpu, f32) — `--blr-device gpu`.
//!
//! Stage for stage the CPU encoder of `nvg.rs` (the f64 reference, kept as
//! `--blr-device cpu`), with the work moved to `cortiq_engine::gpu_wgpu::
//! nvg_dev`:
//!
//! * per input: `H` (damped), the held-out Gram, the act order and the upper
//!   Cholesky factor `Uc` of the permuted `H⁻¹` (equilibrated f32 Cholesky +
//!   blocked inverse — the GPTQ fold's kernels), and the lower factor `L`
//!   when a tensor reading the input has more rows than columns;
//! * latents: the min(m, n)-sized Gram of the whitened weight on the device,
//!   its r largest eigenpairs by device Householder tridiagonalisation, host
//!   f64 bisection + inverse iteration on the tridiagonal and a device
//!   back-transform (eigenvector signs may differ from the host solver's;
//!   the encoder is invariant to them);
//! * ITQ gauge: device GEMMs and a Newton–Schulz polar factor of the r×r
//!   cross term (the host takes it through an r×r eigendecomposition);
//! * metric Ť: the ℓ/h/g block least squares from device GEMMs; the ℓ and g
//!   systems are solved on the device (equilibrated f32 Cholesky), the
//!   per-row 2×2 h systems on the host. `H·gs_p` and `WH·gs_p` are cached
//!   per path and recomputed only when that path's g or V-signs change;
//! * output-side knots: the exact single-bit flips, one device thread per
//!   row (the same greedy passes);
//! * input-side knots: the column-sequential sign search with the
//!   Schur-complement feedback, carried in the r-dimensional latent space —
//!   `Mᵀw_j` of every column is one GEMM up front, and the feedback of a
//!   column's error `e_j` reaches the others as `Mᵀe_j = (Mᵀw_j − g_j K s_j)
//!   / Uc[j,j]`, so no m-length vector is touched in the sequential part
//!   (128-column blocks in one workgroup, the trailing update a GEMM);
//! * errors `tr(E X Eᵀ)` (X = H, the held-out Gram, the raw fit Gram) from
//!   the factor-sized expansion `tr(WXWᵀ) − 2Σ⟨hsl_p, WX·gs_p⟩ +
//!   Σ⟨hsl_pᵀhsl_q, gs_pᵀX gs_q⟩`.

use crate::nvg::{BlrModel, BlrPath, Encoded, PathsChoice, Prof, pack};
use crate::nvg_linalg::{Mat, solve_spd, sym_eig};
use cortiq_core::blr::{blr_bits, blr_rank};
use cortiq_engine::gpu_wgpu::nvg_dev::{self as dv, ABS, DMat, INPLACE, LOWER, SIGN, V};
use cortiq_engine::pool::Pool;
use std::cell::RefCell;
use std::time::Instant;

/// Time a device stage (synchronised when profiling, so the seconds land on
/// the stage that queued the work).
fn ps<T>(p: Option<&Prof>, k: &'static str, f: impl FnOnce() -> T) -> T {
    match p {
        Some(p) => {
            let t = Instant::now();
            let v = f();
            dv::sync();
            p.add(k, t.elapsed().as_secs_f64(), 0);
            v
        }
        None => f(),
    }
}

fn vec_dev(v: &[f64]) -> DMat {
    DMat::from_f64(1, v.len(), v)
}

/// f32 flip tolerance, relative (the f64 host path uses 1e-12;
/// `CMF_NVG_FLIP_TOL` overrides). 1e-7 sits at the f32 resolution: 1e-6
/// stopped the discrete search early enough to cost ~2 % ppl end to end on
/// Qwen3.5-0.8B at 1.0 bpw (686.5 vs 672.5; host reference 656.4).
static FLIP_TOL_V: std::sync::LazyLock<f32> =
    std::sync::LazyLock::new(|| std::env::var("CMF_NVG_FLIP_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-7));

// ───────────────────────────── per-input context ─────────────────────────────

/// One input's calibration context on the device.
pub struct HCtxD {
    pub n: usize,
    /// `hfit + c·I`, c = ρ·tr/n
    pub h: DMat,
    pub hsel: DMat,
    pub c_fit: f64,
    pub sqd: Vec<f64>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub perm: Vec<usize>,
    perm_buf: cortiq_engine::wgpu::Buffer,
    pub uc: DMat,
    d: Vec<f64>,
    l: std::sync::Mutex<Option<std::sync::Arc<DMat>>>,
}

impl HCtxD {
    pub fn new(hfit: &Mat, hsel: &Mat, rho: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(dv::available(), "no wgpu device for the NVG encoder");
        let n = hfit.r;
        let tr = (0..n).map(|i| hfit.at(i, i)).sum::<f64>() / n as f64;
        let c_fit = rho * tr;
        let mut hf: Vec<f32> = hfit.d.iter().map(|&v| v as f32).collect();
        for i in 0..n {
            hf[i * n + i] = (hfit.at(i, i) + c_fit) as f32;
        }
        let h = DMat::from_f32(n, n, &hf);
        drop(hf);
        let hsel_d = DMat::from_f64(n, n, &hsel.d);
        let dg: Vec<f64> = (0..n).map(|i| hfit.at(i, i) + c_fit).collect();
        let sqd: Vec<f64> = dg.iter().map(|v| v.sqrt()).collect();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&a, &b| dg[b].partial_cmp(&dg[a]).unwrap_or(std::cmp::Ordering::Equal));
        let perm_buf = dv::upload_u32(&perm.iter().map(|&v| v as u32).collect::<Vec<_>>());
        // Uc = J inv(chol(J Hp J)) J, equilibrated: the factor of D⁻¹HpD⁻¹,
        // columns scaled back by the permuted D.
        let f = DMat::zeros(n, n);
        let dvv = vec_dev(&sqd);
        let mut ridge = 0f32;
        loop {
            dv::equil(&h, &f, &dvv, ridge, Some(&perm_buf));
            match dv::chol_in_place(&f) {
                Some(true) => break,
                Some(false) => {
                    ridge = if ridge == 0.0 { 1e-6 } else { ridge * 10.0 };
                    anyhow::ensure!(ridge < 1e-1, "permuted Gram not SPD on the device");
                    eprintln!("  nvg-gpu: permuted Gram needed a {ridge:e} ridge (f32 Cholesky)");
                }
                None => anyhow::bail!("device Cholesky unavailable"),
            }
        }
        let x = DMat::zeros(n, n);
        dv::tri_inv(&f, &x).ok_or_else(|| anyhow::anyhow!("device triangular inverse unavailable"))?;
        drop(f);
        let dp: Vec<f64> = perm.iter().map(|&p| sqd[p]).collect();
        let uc = DMat::zeros(n, n);
        dv::make_uc(&x, &vec_dev(&dp), &uc);
        Ok(HCtxD { n, h, hsel: hsel_d, c_fit, sqd: sqd.clone(), perm, perm_buf, uc, d: sqd, l: std::sync::Mutex::new(None) })
    }

    /// Lower Cholesky factor of `H` (built on first use; shared by threads).
    fn l(&self) -> anyhow::Result<std::sync::Arc<DMat>> {
        let mut g = self.l.lock().unwrap();
        if g.is_none() {
            let n = self.n;
            let f = DMat::zeros(n, n);
            let dvv = vec_dev(&self.d);
            dv::equil(&self.h, &f, &dvv, 0.0, None);
            anyhow::ensure!(dv::chol_in_place(&f) == Some(true), "calibration Gram not SPD on the device");
            // L = D·L̃ (lower)
            dv::scale_rc(n, n, None, (&f, 0, n), Some(&dvv), None, INPLACE | LOWER, 1.0, 0.0);
            *g = Some(std::sync::Arc::new(f));
        }
        Ok(g.as_ref().unwrap().clone())
    }
}

// ───────────────────────────── model on the device ─────────────────────────────

struct PathD {
    /// scales on the device (1 × m, 1 × r, 1 × n)
    h: DMat,
    l: DMat,
    g: DMat,
    /// ±1, m × r
    su: DMat,
    /// ±1, n × r
    sv: DMat,
    /// H·gs (n × r), WH·gs (m × r): valid while g and sv are unchanged
    hg: RefCell<Option<DMat>>,
    whg: RefCell<Option<DMat>>,
}

impl PathD {
    fn dup(&self) -> PathD {
        PathD {
            h: self.h.dup(),
            l: self.l.dup(),
            g: self.g.dup(),
            su: self.su.dup(),
            sv: self.sv.dup(),
            hg: RefCell::new(self.hg.borrow().as_ref().map(|m| m.dup())),
            whg: RefCell::new(self.whg.borrow().as_ref().map(|m| m.dup())),
        }
    }
    fn invalidate(&self) {
        *self.hg.borrow_mut() = None;
        *self.whg.borrow_mut() = None;
    }
}

struct ModelD {
    m: usize,
    n: usize,
    r: usize,
    paths: Vec<PathD>,
}

impl ModelD {
    fn dup(&self) -> ModelD {
        ModelD { m: self.m, n: self.n, r: self.r, paths: self.paths.iter().map(|p| p.dup()).collect() }
    }
    /// `diag(g) S_V` (n × r)
    fn gs(&self, p: usize) -> DMat {
        let q = &self.paths[p];
        let o = DMat::zeros(self.n, self.r);
        dv::scale_into(&q.sv, &o, Some(&q.g), None, 0, 1.0);
        o
    }
    /// `diag(h) S_U diag(ℓ)` (m × r)
    fn hsl(&self, p: usize) -> DMat {
        let q = &self.paths[p];
        let o = DMat::zeros(self.m, self.r);
        dv::scale_into(&q.su, &o, Some(&q.h), Some(&q.l), 0, 1.0);
        o
    }
    /// `diag(h) S_U` (m × r)
    fn hs(&self, p: usize) -> DMat {
        let q = &self.paths[p];
        let o = DMat::zeros(self.m, self.r);
        dv::scale_into(&q.su, &o, Some(&q.h), None, 0, 1.0);
        o
    }
    /// `S_U diag(ℓ)` (m × r)
    fn sl(&self, p: usize) -> DMat {
        let q = &self.paths[p];
        let o = DMat::zeros(self.m, self.r);
        dv::scale_into(&q.su, &o, None, Some(&q.l), 0, 1.0);
        o
    }
    fn ensure_caches(&self, p: usize, c: &TCtxD) {
        let q = &self.paths[p];
        if q.hg.borrow().is_some() && q.whg.borrow().is_some() {
            return;
        }
        let gs = self.gs(p);
        if q.hg.borrow().is_none() {
            *q.hg.borrow_mut() = Some(dv::mm(&c.hc.h, false, &gs, false));
        }
        if q.whg.borrow().is_none() {
            *q.whg.borrow_mut() = Some(dv::mm(&c.wh, false, &gs, false));
        }
    }
    fn hg(&self, p: usize, c: &TCtxD) -> std::cell::Ref<'_, DMat> {
        self.ensure_caches(p, c);
        std::cell::Ref::map(self.paths[p].hg.borrow(), |o| o.as_ref().unwrap())
    }
    fn whg(&self, p: usize, c: &TCtxD) -> std::cell::Ref<'_, DMat> {
        self.ensure_caches(p, c);
        std::cell::Ref::map(self.paths[p].whg.borrow(), |o| o.as_ref().unwrap())
    }
    /// Host copy (signs and scales) for packing.
    fn to_host(&self) -> BlrModel {
        let paths = self
            .paths
            .iter()
            .map(|q| BlrPath {
                h: q.h.to_f64(),
                su: Mat::from_vec(self.m, self.r, q.su.to_f64()),
                l: q.l.to_f64(),
                sv: Mat::from_vec(self.n, self.r, q.sv.to_f64()),
                g: q.g.to_f64(),
            })
            .collect();
        BlrModel { paths, m: self.m, n: self.n, r: self.r }
    }
}

/// One tensor's device context.
struct TCtxD<'a> {
    w: DMat,
    wh: DMat,
    whsel: DMat,
    /// tr(W H Wᵀ), tr(W Hsel Wᵀ), ‖W‖²
    a_h: f64,
    a_sel: f64,
    w2: f64,
    hc: &'a HCtxD,
    pool: Option<&'a Pool>,
    prof: Option<&'a Prof>,
}

#[derive(Clone, Copy, PartialEq)]
enum Metric {
    /// the damped fitting Gram
    H,
    /// the raw fitting Gram (H − c·I)
    Fit,
    /// the held-out Gram
    Sel,
}

/// `tr((W−Ŵ) X (W−Ŵ)ᵀ)` by the factor-sized expansion (all the inner
/// products read back together).
fn herr_d(md: &ModelD, c: &TCtxD, x: Metric) -> f64 {
    let np = md.paths.len();
    let (m, r) = (md.m, md.r);
    let gss: Vec<DMat> = (0..np).map(|p| md.gs(p)).collect();
    let hsls: Vec<DMat> = (0..np).map(|p| md.hsl(p)).collect();
    let mut fb = dv::FrobBatch::new();
    let mut keep: Vec<DMat> = Vec::new();
    let mut bslots = Vec::new();
    let mut fslots = Vec::new();
    for p in 0..np {
        match x {
            Metric::H | Metric::Fit => bslots.push(fb.add(&hsls[p], &md.whg(p, c), m * r)),
            Metric::Sel => {
                let t = dv::mm(&c.whsel, false, &gss[p], false);
                bslots.push(fb.add(&hsls[p], &t, m * r));
                keep.push(t);
            }
        }
        if x == Metric::Fit {
            let t = dv::mm(&c.w, false, &gss[p], false);
            fslots.push(fb.add(&hsls[p], &t, m * r));
            keep.push(t);
        }
    }
    let xg: Vec<Option<DMat>> = (0..np).map(|q| (x == Metric::Sel).then(|| dv::mm(&c.hc.hsel, false, &gss[q], false))).collect();
    let mut cslots = Vec::new();
    let mut c2slots = Vec::new();
    for p in 0..np {
        for q in 0..np {
            let t = dv::mm(&hsls[p], true, &hsls[q], false);
            let u = match x {
                Metric::Sel => dv::mm(&gss[p], true, xg[q].as_ref().unwrap(), false),
                _ => dv::mm(&gss[p], true, &md.hg(q, c), false),
            };
            cslots.push(fb.add(&t, &u, r * r));
            if x == Metric::Fit {
                let u2 = dv::mm(&gss[p], true, &gss[q], false);
                c2slots.push(fb.add(&t, &u2, r * r));
                keep.push(u2);
            }
            keep.push(t);
            keep.push(u);
        }
    }
    let v = fb.read();
    drop(keep);
    let b: f64 = bslots.iter().map(|&k| v[k]).sum::<f64>() - c.hc.c_fit * fslots.iter().map(|&k| v[k]).sum::<f64>();
    let cc: f64 = cslots.iter().map(|&k| v[k]).sum::<f64>() - c.hc.c_fit * c2slots.iter().map(|&k| v[k]).sum::<f64>();
    let a = match x {
        Metric::H => c.a_h,
        Metric::Fit => c.a_h - c.hc.c_fit * c.w2,
        Metric::Sel => c.a_sel,
    };
    (a - 2.0 * b + cc).max(0.0)
}

fn rel_d(md: &ModelD, c: &TCtxD, x: Metric) -> f64 {
    let a = match x {
        Metric::H => c.a_h,
        Metric::Fit => c.a_h - c.hc.c_fit * c.w2,
        Metric::Sel => c.a_sel,
    };
    (herr_d(md, c, x) / a.max(1e-300)).sqrt()
}

// ───────────────────────────── initialisers ─────────────────────────────

/// The `r` largest eigenpairs of the symmetric device matrix `g`
/// (destroyed): device tridiagonalisation + host f64 tridiagonal solve
/// (`CMF_NVG_HOST_EIG=1`: the host `sym_eig` of the whole matrix instead).
fn eig_top(c: &TCtxD, g: &DMat, r: usize) -> (Vec<f64>, DMat) {
    let k = g.r;
    if std::env::var("CMF_NVG_HOST_EIG").as_deref() == Ok("1") {
        let (lam, e) = sym_eig(c.pool, &Mat::from_vec(k, k, g.to_f64()));
        let mut v = vec![0f64; k * r];
        for i in 0..k {
            v[i * r..(i + 1) * r].copy_from_slice(&e.row(i)[..r]);
        }
        return (lam[..r].to_vec(), DMat::from_f64(k, r, &v));
    }
    dv::sym_eig_top(g, r, c.pool)
}

/// Rank-`r` latents of `t` in the H metric (see `nvg::latents`): the Gram
/// `t H tᵀ` (m ≤ n, from `th = t·H`) or `(tL)ᵀ(tL)` on the device, its
/// eigendecomposition on the host in f64.
fn latents_d(c: &TCtxD, t: &DMat, th: Option<&DMat>, r: usize) -> anyhow::Result<(DMat, DMat)> {
    let (m, n) = (t.r, t.c);
    let prof = c.prof;
    let (sig, ur) = if m <= n {
        let th = th.ok_or_else(|| anyhow::anyhow!("latents: t·H missing"))?;
        let g = ps(prof, "lat.gemm", || dv::mm(th, false, t, true));
        let (lam, ur) = ps(prof, "lat.eig", || eig_top(c, &g, r));
        let sig: Vec<f64> = lam[..r].iter().map(|v| v.max(0.0).sqrt()).collect();
        (sig, ur)
    } else {
        let l = c.hc.l()?;
        let tl = ps(prof, "lat.gemm", || dv::mm(t, false, &l, false));
        let g = ps(prof, "lat.gemm", || dv::mm(&tl, true, &tl, false));
        let (lam, vr) = ps(prof, "lat.eig", || eig_top(c, &g, r));
        let sig: Vec<f64> = lam[..r].iter().map(|v| v.max(0.0).sqrt()).collect();
        let ur = ps(prof, "lat.gemm", || dv::mm(&tl, false, &vr, false));
        let floor = sig[0].max(1e-300) * 1e-15;
        let inv: Vec<f64> = sig.iter().map(|s| 1.0 / s.max(floor)).collect();
        dv::scale_rc(m, r, None, (&ur, 0, r), None, Some(&vec_dev(&inv)), INPLACE, 1.0, 0.0);
        (sig, ur)
    };
    let floor = sig[0].max(1e-300) * 1e-15;
    let up = DMat::zeros(m, r);
    let sq: Vec<f64> = sig.iter().map(|s| s.sqrt()).collect();
    dv::scale_into(&ur, &up, None, Some(&vec_dev(&sq)), 0, 1.0);
    let vp = ps(prof, "lat.gemm", || dv::mm(t, true, &ur, false));
    let isq: Vec<f64> = sig.iter().map(|s| 1.0 / s.max(floor).sqrt()).collect();
    dv::scale_rc(n, r, None, (&vp, 0, r), None, Some(&vec_dev(&isq)), INPLACE, 1.0, 0.0);
    Ok((up, vp))
}

/// Orthogonal polar factor of the square `mm` by Newton–Schulz
/// (`X ← X(3I − XᵀX)/2` from `M/‖M‖_F`), until `‖XᵀX − I‖_F ≤ 1e-4`.
fn polar_d(mm: &DMat, prof: Option<&Prof>) -> DMat {
    let r = mm.r;
    let mut x = DMat::zeros(r, r);
    dv::normalize_frob(mm, &x);
    let mut y = DMat::zeros(r, r);
    let a = DMat::zeros(r, r);
    // Checked after NS_FIRST iterations and then every NS_STEP: each check
    // is a readback, so they are few.
    const NS_FIRST: usize = 12;
    const NS_STEP: usize = 4;
    let mut it = 0;
    loop {
        dv::gemm(1.0, &x, true, &x, false, 0.0, &a);
        if it >= NS_FIRST && (it - NS_FIRST) % NS_STEP == 0 {
            let mut fb = dv::FrobBatch::new();
            fb.add(&a, &a, r * r);
            let fa = fb.read()[0];
            let tr: f64 = dv::diag(&a, r, 0, r).iter().sum();
            if (fa - 2.0 * tr + r as f64).max(0.0).sqrt() <= 1e-4 || it >= 60 {
                break;
            }
        }
        dv::gemm(1.0, &x, false, &a, false, 0.0, &y);
        dv::axpby(1.5, &x, -0.5, &y);
        std::mem::swap(&mut x, &mut y);
        it += 1;
    }
    if let Some(p) = prof {
        p.add("itq.ns_iters", 0.0, it as u64);
    }
    x
}

/// H-weighted ITQ gauge (see `nvg::itq_weighted`).
fn itq_d(c: &TCtxD, up: &DMat, vp: &DMat, wu: &[f64], wv: &[f64], iters: usize) -> (DMat, DMat) {
    let (m, n, r) = (up.r, vp.r, up.c);
    let nn = m + n;
    let prof = c.prof;
    let x = DMat::zeros(nn, r);
    dv::scale_rc(m, r, Some((up, 0, r)), (&x, 0, r), None, None, 0, 1.0, 0.0);
    dv::scale_rc(n, r, Some((vp, 0, r)), (&x, m * r, r), None, None, 0, 1.0, 0.0);
    dv::rownorm(nn, r, &x, 0, r);
    let mut w2: Vec<f64> = wu.iter().chain(wv).map(|v| v * v).collect();
    let mean = w2.iter().sum::<f64>() / nn as f64;
    w2.iter_mut().for_each(|v| *v /= mean.max(1e-300));
    let w2d = vec_dev(&w2);
    let mut eye = vec![0f64; r * r];
    for i in 0..r {
        eye[i * r + i] = 1.0;
    }
    let mut rot = DMat::from_f64(r, r, &eye);
    let xr = DMat::zeros(nn, r);
    let mm = DMat::zeros(r, r);
    for _ in 0..iters {
        ps(prof, "itq.gemm", || {
            dv::gemm(1.0, &x, false, &rot, false, 0.0, &xr);
            dv::scale_rc(nn, r, None, (&xr, 0, r), Some(&w2d), None, INPLACE | SIGN, 1.0, 0.0);
            dv::gemm(1.0, &x, true, &xr, false, 0.0, &mm);
        });
        rot = ps(prof, "itq.polar", || polar_d(&mm, prof));
    }
    ps(prof, "itq.gemm", || (dv::mm(up, false, &rot, false), dv::mm(vp, false, &rot, false)))
}

/// Non-negative rank-1 fit of `mabs` (rows × c): the host's power
/// iteration (`nvg::rank1_pos`) on the c×c Gram, which is the same map.
fn rank1_d(c: &TCtxD, mabs: &DMat) -> (Vec<f64>, Vec<f64>) {
    let (rows, cc) = (mabs.r, mabs.c);
    let g = Mat::from_vec(cc, cc, dv::mm(mabs, true, mabs, false).to_f64());
    let mut v = vec![1.0 / (cc as f64).sqrt(); cc];
    let gv = |v: &[f64]| -> Vec<f64> {
        crate::nvg_linalg::gemm(c.pool, &g, false, &Mat::from_vec(cc, 1, v.to_vec()), false).d
    };
    for _ in 0..60 {
        let mut v2 = gv(&v);
        let nv = v2.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v2.iter_mut().for_each(|x| *x /= nv);
        let diff = v2.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        v = v2;
        if diff < 1e-9 {
            break;
        }
    }
    let vd = DMat::from_f64(cc, 1, &v);
    let mut u = dv::mm(mabs, false, &vd, false).to_f64();
    debug_assert_eq!(u.len(), rows);
    let s = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    u.iter_mut().for_each(|x| *x /= s.max(1e-300));
    let rs = s.sqrt();
    let mut left: Vec<f64> = u.iter().map(|x| rs * x).collect();
    let mut right: Vec<f64> = v.iter().map(|x| rs * x).collect();
    let lmax = left.iter().cloned().fold(0.0, f64::max);
    let rmax = right.iter().cloned().fold(0.0, f64::max);
    left.iter_mut().for_each(|x| *x = x.max(1e-12 * lmax));
    right.iter_mut().for_each(|x| *x = x.max(1e-12 * rmax));
    (left, right)
}

/// Signs of the latents plus rank-1 magnitude fits.
fn svid_d(c: &TCtxD, up: &DMat, vp: &DMat) -> PathD {
    let (m, n, r) = (up.r, vp.r, up.c);
    let su = DMat::zeros(m, r);
    let sv = DMat::zeros(n, r);
    dv::scale_into(up, &su, None, None, SIGN, 1.0);
    dv::scale_into(vp, &sv, None, None, SIGN, 1.0);
    let ua = DMat::zeros(m, r);
    dv::scale_into(up, &ua, None, None, ABS, 1.0);
    let (h, lu) = rank1_d(c, &ua);
    drop(ua);
    let va = DMat::zeros(n, r);
    dv::scale_into(vp, &va, None, None, ABS, 1.0);
    let (g, lv) = rank1_d(c, &va);
    let l: Vec<f64> = lu.iter().zip(&lv).map(|(a, b)| a * b).collect();
    PathD { h: vec_dev(&h), l: vec_dev(&l), g: vec_dev(&g), su, sv, hg: RefCell::new(None), whg: RefCell::new(None) }
}

// ───────────────────────────── closed-form metric (Ť) ─────────────────────────────

fn host_solve() -> bool {
    std::env::var("CMF_NVG_HOST_SOLVE").as_deref() == Ok("1")
}

/// Exact block least squares for ℓ (joint), h (per row), g (per path), the
/// scales staying on the device (one readback per factorisation: its pivot
/// flag).
fn als_d(md: &mut ModelD, c: &TCtxD, sweeps: usize) {
    let (m, n, r) = (md.m, md.n, md.r);
    let np = md.paths.len();
    let pr = np * r;
    let prof = c.prof;
    for _ in 0..sweeps {
        // ℓ
        let bhb = ps(prof, "als.l", || {
            let gss: Vec<DMat> = (0..np).map(|p| md.gs(p)).collect();
            let hss: Vec<DMat> = (0..np).map(|p| md.hs(p)).collect();
            let bhb = DMat::zeros(pr, pr);
            for p in 0..np {
                for q in 0..np {
                    let hg = md.hg(q, c);
                    dv::gemm_v(r, r, n, 1.0, V::t(&gss[p]), V::n(&hg), 0.0, &bhb, p * r * pr + q * r, pr, None);
                }
            }
            let sys = DMat::zeros(pr, pr);
            for p in 0..np {
                for q in 0..np {
                    dv::gemm_v(r, r, m, 1.0, V::t(&hss[p]), V::n(&hss[q]), 0.0, &sys, p * r * pr + q * r, pr, Some(&bhb));
                }
            }
            let rhs = DMat::zeros(1, pr);
            for p in 0..np {
                dv::colsum_dev(m, r, &hss[p], 0, r, &md.whg(p, c), 0, r, &rhs, p * r);
            }
            let lam = if host_solve() {
                let x = solve_spd(c.pool, &Mat::from_vec(pr, pr, sys.to_f64()), &rhs.to_f64(), 1e-10);
                vec_dev(&x)
            } else {
                dv::spd_solve_d(&sys, &rhs, 1e-10)
            };
            for p in 0..np {
                dv::copy_range(&lam, p * r, &md.paths[p].l, 0, r);
            }
            bhb
        });
        // h (B unchanged since the ℓ step: reuse its blocks)
        ps(prof, "als.h", || {
            let sls: Vec<DMat> = (0..np).map(|p| md.sl(p)).collect();
            let rh: Vec<DMat> = (0..2).map(|_| DMat::zeros(1, m)).collect();
            let gm: Vec<DMat> = (0..3).map(|_| DMat::zeros(1, m)).collect();
            for a in 0..np {
                dv::rowdot(m, r, &md.whg(a, c), 0, r, &sls[a], 0, r, &rh[a]);
                for cq in a..np {
                    let mk = DMat::zeros(m, r);
                    let kac = V { m: &bhb, off: a * r * pr + cq * r, ld: pr, t: false };
                    dv::gemm_v(m, r, r, 1.0, V::n(&sls[a]), kac, 0.0, &mk, 0, r, None);
                    dv::rowdot(m, r, &mk, 0, r, &sls[cq], 0, r, &gm[a + cq]);
                }
            }
            if np == 1 {
                let unused = DMat::zeros(1, 4);
                dv::hsolve(m, 1, &rh[0], &rh[1], &gm[0], &gm[1], &gm[2], &md.paths[0].h, &unused);
            } else {
                let (p0, p1) = md.paths.split_at(1);
                dv::hsolve(m, 2, &rh[0], &rh[1], &gm[0], &gm[1], &gm[2], &p0[0].h, &p1[0].h);
            }
        });
        drop(bhb);
        // g
        ps(prof, "als.g", || {
            for p in 0..np {
                let hsl = md.hsl(p);
                let y = dv::mm(&c.wh, true, &hsl, false);
                for q in 0..np {
                    if q == p {
                        continue;
                    }
                    let t = dv::mm(&md.hsl(q), true, &hsl, false);
                    dv::gemm(-1.0, &md.hg(q, c), false, &t, false, 1.0, &y);
                }
                let rhs = DMat::zeros(1, n);
                dv::rowdot(n, r, &md.paths[p].sv, 0, r, &y, 0, r, &rhs);
                drop(y);
                let k = dv::mm(&hsl, true, &hsl, false);
                let svk = dv::mm(&md.paths[p].sv, false, &k, false);
                let ctc = DMat::zeros(n, n);
                dv::gemm_v(n, n, r, 1.0, V::n(&svk), V::t(&md.paths[p].sv), 0.0, &ctc, 0, n, Some(&c.hc.h));
                drop(svk);
                md.paths[p].g = dv::spd_solve_d(&ctc, &rhs, 1e-10);
                md.paths[p].invalidate();
            }
        });
    }
}

// ───────────────────────────── discrete knots ─────────────────────────────

/// Output-side knots (see `nvg::flip_u`).
fn flip_u_d(md: &mut ModelD, c: &TCtxD, pi: usize) -> u64 {
    let (m, r) = (md.m, md.r);
    let np = md.paths.len();
    let gs = md.gs(pi);
    let k = dv::mm(&gs, true, &md.hg(pi, c), false);
    if std::env::var("CMF_NVG_FLIPBLK").as_deref() != Ok("0") {
        // row-major: whb = WH·gs − Σ_q hsl_q (H gs_q)ᵀ gs, qs = S_U Q
        let whb = md.whg(pi, c).dup();
        for q in 0..np {
            if q == pi {
                continue;
            }
            let kqp = dv::mm(&md.hg(q, c), true, &gs, false);
            dv::gemm(-1.0, &md.hsl(q), false, &kqp, false, 1.0, &whb);
        }
        let qm = DMat::zeros(r, r);
        let ld = &md.paths[pi].l;
        dv::scale_into(&k, &qm, Some(ld), Some(ld), 0, 1.0);
        let qs = dv::mm(&md.paths[pi].su, false, &qm, false);
        if let Some(n) = dv::flipu_blocked(&md.paths[pi].su, &qs, &whb, &qm, &md.paths[pi].h, ld, m, r, *FLIP_TOL_V) {
            return n;
        }
    }
    let whbt = dv::mm(&gs, true, &c.wh, true);
    for q in 0..np {
        if q == pi {
            continue;
        }
        let kpq = dv::mm(&gs, true, &md.hg(q, c), false);
        dv::gemm(-1.0, &kpq, false, &md.hsl(q), true, 1.0, &whbt);
    }
    let q = DMat::zeros(r, r);
    let ld = &md.paths[pi].l;
    dv::scale_into(&k, &q, Some(ld), Some(ld), 0, 1.0);
    let qst = dv::mm(&q, false, &md.paths[pi].su, true);
    let sut = dv::transpose(&md.paths[pi].su);
    let flips = dv::flipu(&sut, &qst, &whbt, &q, &md.paths[pi].h, ld, m, r, *FLIP_TOL_V);
    md.paths[pi].su = dv::transpose(&sut);
    flips
}

/// Input-side knots (see `nvg::gptq_v`), in the latent space.
fn gptq_v_d(md: &mut ModelD, c: &TCtxD, pi: usize) -> u64 {
    let (n, r) = (md.n, md.r);
    let np = md.paths.len();
    let mm = md.hsl(pi);
    let k = dv::mm(&mm, true, &mm, false);
    let bvn = dv::mm(&c.w, true, &mm, false);
    for q in 0..np {
        if q == pi {
            continue;
        }
        let t = dv::mm(&md.hsl(q), true, &mm, false);
        dv::gemm(-1.0, &md.gs(q), false, &t, false, 1.0, &bvn);
    }
    let bv = DMat::zeros(n, r);
    dv::permute_rows(&bvn, &bv, &c.hc.perm_buf, false);
    drop(bvn);
    let s = DMat::zeros(n, r);
    dv::permute_rows(&md.paths[pi].sv, &s, &c.hc.perm_buf, false);
    let ks0 = dv::mm(&s, false, &k, false);
    let eg = DMat::zeros(1, 128 * r + n);
    let cnt = DMat::zeros(1, 4);
    let mut b0 = 0;
    while b0 < n {
        let b1 = (b0 + 128).min(n);
        dv::gv_block(&bv, &s, &ks0, &k, &c.hc.uc, &eg, &cnt, b0, b1, 4 * r, *FLIP_TOL_V);
        b0 = b1;
    }
    dv::permute_rows(&s, &md.paths[pi].sv, &c.hc.perm_buf, true);
    dv::scatter_vec(&eg, 128 * r, &md.paths[pi].g, &c.hc.perm_buf, n);
    md.paths[pi].invalidate();
    cnt.to_f32()[0].to_bits() as u64
}

/// Discrete-knot refinement alternating with the exact metric; a step is
/// kept only when it lowers the fit error.
fn refine_d(mut md: ModelD, c: &TCtxD, sweeps: usize) -> ModelD {
    let prof = c.prof;
    let mut cur = ps(prof, "err", || herr_d(&md, c, Metric::H));
    for _ in 0..sweeps {
        for pi in 0..md.paths.len() {
            for kind in 0..2 {
                let mut trial = md.dup();
                if kind == 0 {
                    let t = Instant::now();
                    let f = flip_u_d(&mut trial, c, pi);
                    if let Some(p) = prof {
                        dv::sync();
                        p.add("flip_u", t.elapsed().as_secs_f64(), 1);
                        p.add("flip_u.flips", 0.0, f);
                    }
                } else {
                    let t = Instant::now();
                    let f = gptq_v_d(&mut trial, c, pi);
                    if let Some(p) = prof {
                        p.add("gptq_v", t.elapsed().as_secs_f64(), 1);
                        p.add("gptq_v.flips", 0.0, f);
                    }
                }
                als_d(&mut trial, c, 3);
                let e = ps(prof, "err", || herr_d(&trial, c, Metric::H));
                if e < cur {
                    md = trial;
                    cur = e;
                }
            }
        }
    }
    md
}

/// Whitened topology, H-weighted ITQ gauge, closed-form metric; the residual
/// path on the H-residual of the refitted primary.
fn build_init_d(c: &TCtxD, r: usize, paths: usize) -> anyhow::Result<ModelD> {
    let (m, n) = (c.w.r, c.w.c);
    let prof = c.prof;
    let lat = |t: &DMat, th: Option<&DMat>| -> anyhow::Result<PathD> {
        let (up, vp) = latents_d(c, t, th, r)?;
        let rn = DMat::zeros(1, m.max(n));
        dv::rowdot(m, r, &up, 0, r, &up, 0, r, &rn);
        let wu: Vec<f64> = rn.to_f64()[..m].iter().map(|v| v.max(0.0).sqrt()).collect();
        dv::rowdot(n, r, &vp, 0, r, &vp, 0, r, &rn);
        let wv: Vec<f64> = rn.to_f64()[..n].iter().zip(&c.hc.sqd).map(|(v, s)| v.max(0.0).sqrt() * s).collect();
        let (up, vp) = itq_d(c, &up, &vp, &wu, &wv, 50);
        Ok(ps(prof, "svid", || svid_d(c, &up, &vp)))
    };
    let mut md = ModelD { m, n, r, paths: vec![lat(&c.w, Some(&c.wh))?] };
    als_d(&mut md, c, 3);
    if paths == 2 {
        let (res, resh) = ps(prof, "recon", || {
            let hsl = md.hsl(0);
            let res = c.w.dup();
            dv::gemm(-1.0, &hsl, false, &md.gs(0), true, 1.0, &res);
            let resh = (m <= n).then(|| {
                let resh = c.wh.dup();
                dv::gemm(-1.0, &hsl, false, &md.hg(0, c), true, 1.0, &resh);
                resh
            });
            (res, resh)
        });
        let p2 = lat(&res, resh.as_ref())?;
        drop((res, resh));
        md.paths.push(p2);
        als_d(&mut md, c, 6);
    }
    Ok(md)
}

/// NVG for one tensor on the device (`nvg::encode_nvg`). `hc` holds the
/// tensor's input Grams.
#[allow(clippy::too_many_arguments)]
pub fn encode_nvg_dev(
    pool: Option<&Pool>,
    w: &Mat,
    hc: &HCtxD,
    b: f64,
    mult: f64,
    choice: PathsChoice,
    prof: Option<&Prof>,
) -> anyhow::Result<Option<Encoded>> {
    let (m, n) = (w.r, w.c);
    anyhow::ensure!(n == hc.n, "{m}x{n} weight vs a {}-wide Gram", hc.n);
    let c = ps(prof, "wh", || {
        let wd = DMat::from_f64(m, n, &w.d);
        let wh = dv::mm(&wd, false, &hc.h, false);
        let whsel = dv::mm(&wd, false, &hc.hsel, false);
        let (a_h, a_sel, w2) = (dv::frob(&wd, &wh, m * n), dv::frob(&wd, &whsel, m * n), dv::frob(&wd, &wd, m * n));
        TCtxD { w: wd, wh, whsel, a_h, a_sel, w2, hc, pool, prof }
    });
    let mut cands: Vec<(usize, ModelD)> = Vec::new();
    let mut notes = Vec::new();
    if choice != PathsChoice::One {
        let r2 = blr_rank(m, n, b, 2, mult);
        anyhow::ensure!(r2 <= dv::GV_MAX_RANK, "rank {r2} above the device search limit");
        let n3 = build_init_d(&c, r2, 2)?;
        let e3 = ps(prof, "select", || rel_d(&n3, &c, Metric::Sel));
        let n4 = refine_d(n3, &c, 2);
        notes.push(format!("init {e3:.4}"));
        cands.push((2, n4));
    }
    if choice != PathsChoice::Two {
        let r1 = blr_rank(m, n, b, 1, mult);
        anyhow::ensure!(r1 <= dv::GV_MAX_RANK, "rank {r1} above the device search limit");
        let p1 = refine_d(build_init_d(&c, r1, 1)?, &c, 2);
        cands.push((1, p1));
    }
    let mut best: Option<(f64, usize)> = None;
    for (ci, (p, md)) in cands.iter().enumerate() {
        let e = ps(prof, "select", || rel_d(md, &c, Metric::Sel));
        notes.push(format!("P{p} {e:.4}"));
        if best.is_none_or(|(be, _)| e < be) {
            best = Some((e, ci));
        }
    }
    let Some((_, bi)) = best else { return Ok(None) };
    let (p, mut md) = cands.swap_remove(bi);
    drop(cands);
    let Some((bytes, st)) = ps(prof, "select", || pack(&md.to_host())) else { return Ok(None) };
    // the stored (rebalanced, f16-rounded) scales, same signs
    for (q, s) in md.paths.iter_mut().zip(&st.paths) {
        q.h = vec_dev(&s.h);
        q.l = vec_dev(&s.l);
        q.g = vec_dev(&s.g);
        q.invalidate();
    }
    let (ef, es) = ps(prof, "select", || (rel_d(&md, &c, Metric::Fit), rel_d(&md, &c, Metric::Sel)));
    Ok(Some(Encoded {
        bytes,
        paths: p,
        rank: md.r,
        bits: blr_bits(m, n, md.r, p),
        line: format!("pick P{p} r{} | sel: {} | stored fit {ef:.4} sel {es:.4}", md.r, notes.join(" ")),
    }))
}

/// `nvg-bench --device gpu`: one tensor from host Grams.
#[allow(clippy::too_many_arguments)]
pub fn encode_tensor(
    w: &Mat,
    hfit: &Mat,
    hsel: &Mat,
    rho: f64,
    b: f64,
    mult: f64,
    choice: PathsChoice,
    prof: Option<&Prof>,
    threads: usize,
) -> anyhow::Result<Option<Encoded>> {
    let pool = Pool::new(threads.saturating_sub(1).max(1));
    let t = Instant::now();
    let hc = HCtxD::new(hfit, hsel, rho)?;
    dv::sync();
    if let Some(p) = prof {
        p.add("hctx", t.elapsed().as_secs_f64(), 0);
    }
    encode_nvg_dev(Some(&pool), w, &hc, b, mult, choice, prof)
}

/// A host model on the device (same signs and scales).
#[cfg_attr(not(test), allow(dead_code))]
fn model_to_dev(md: &BlrModel) -> ModelD {
    let paths = md
        .paths
        .iter()
        .map(|q| PathD {
            h: vec_dev(&q.h),
            l: vec_dev(&q.l),
            g: vec_dev(&q.g),
            su: DMat::from_f64(md.m, md.r, &q.su.d),
            sv: DMat::from_f64(md.n, md.r, &q.sv.d),
            hg: RefCell::new(None),
            whg: RefCell::new(None),
        })
        .collect();
    ModelD { m: md.m, n: md.n, r: md.r, paths }
}

#[cfg(test)]
mod tests {
    //! Device stages against the f64 host encoder, stage by stage, on a
    //! small synthetic layer (skips without a wgpu device).
    use super::*;
    use crate::nvg::{self, HCtx, TCtx};
    use crate::nvg_linalg::gemm;

    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    /// m × n layer with a decaying spectrum and three heavy input channels;
    /// the fit Gram and a second (held-out) Gram from other samples.
    fn synth(m: usize, n: usize) -> (Mat, Mat, Mat) {
        let t = 2048usize;
        let mut s = 11u64;
        let mut w = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                w.d[i * n + j] = rnd(&mut s) * (1.0 + j as f64).powf(-0.4) * (1.0 + i as f64).powf(-0.2);
            }
        }
        let mut sc: Vec<f64> = (0..n).map(|_| (rnd(&mut s) * 1.6).exp()).collect();
        for k in 0..3 {
            sc[k * 7] *= 20.0;
        }
        let gram = |s: &mut u64| {
            let mut x = Mat::zeros(n, t);
            for i in 0..n {
                for k in 0..t {
                    x.d[i * t + k] = sc[i] * rnd(s);
                }
            }
            let mut h = gemm(None, &x, false, &x, true);
            h.d.iter_mut().for_each(|v| *v /= t as f64);
            h
        };
        let hfit = gram(&mut s);
        let hsel = gram(&mut s);
        (w, hfit, hsel)
    }

    fn reld(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
        (num / den).sqrt()
    }

    struct Fx {
        w: Mat,
        hfit: Mat,
        hsel: Mat,
        hc: HCtx,
        hcd: HCtxD,
        pool: Pool,
    }

    fn fixture(m: usize, n: usize) -> Option<Fx> {
        if !dv::available() {
            eprintln!("no device — skipped");
            return None;
        }
        let (w, hfit, hsel) = synth(m, n);
        let pool = Pool::new(2);
        let hc = HCtx::new(Some(&pool), &hfit, 0.03).unwrap();
        let hcd = HCtxD::new(&hfit, &hsel, 0.03).unwrap();
        Some(Fx { w, hfit, hsel, hc, hcd, pool })
    }

    fn tctx_d<'a>(f: &'a Fx) -> TCtxD<'a> {
        let (m, n) = (f.w.r, f.w.c);
        let wd = DMat::from_f64(m, n, &f.w.d);
        let wh = dv::mm(&wd, false, &f.hcd.h, false);
        let whsel = dv::mm(&wd, false, &f.hcd.hsel, false);
        let (a_h, a_sel, w2) = (dv::frob(&wd, &wh, m * n), dv::frob(&wd, &whsel, m * n), dv::frob(&wd, &wd, m * n));
        TCtxD { w: wd, wh, whsel, a_h, a_sel, w2, hc: &f.hcd, pool: Some(&f.pool), prof: None }
    }

    fn tctx<'a>(f: &'a Fx) -> TCtx<'a> {
        TCtx { w: &f.w, wh: gemm(Some(&f.pool), &f.w, false, &f.hc.hf, false), hc: &f.hc, pool: Some(&f.pool), prof: None }
    }

    fn err_cpu(f: &Fx, md: &BlrModel, x: &Mat) -> f64 {
        nvg::herr(Some(&f.pool), &f.w.sub(&md.recon(Some(&f.pool), None)), x)
    }

    #[test]
    fn errors_and_contexts_match_host() {
        let Some(f) = fixture(160, 128) else { return };
        // Uc and L against the host factors
        let uc = f.hcd.uc.to_f64();
        assert!(reld(&uc, &f.hc.uc.d) < 1e-3, "Uc {}", reld(&uc, &f.hc.uc.d));
        assert_eq!(f.hcd.perm, f.hc.perm);
        let l = f.hcd.l().unwrap().to_f64();
        assert!(reld(&l, &f.hc.l.d) < 1e-3, "L {}", reld(&l, &f.hc.l.d));
        let c = tctx_d(&f);
        let md = nvg::build_dual_svid(Some(&f.pool), &f.w, 24, 2);
        let mdd = model_to_dev(&md);
        let hfit_damped = &f.hc.hf;
        for (x, hx) in [(Metric::H, hfit_damped), (Metric::Fit, &f.hfit), (Metric::Sel, &f.hsel)] {
            let (got, want) = (herr_d(&mdd, &c, x), err_cpu(&f, &md, hx));
            assert!((got - want).abs() <= 1e-4 * want, "herr {got} vs {want}");
        }
    }

    #[test]
    fn latents_itq_polar_match_host() {
        for &(m, n) in &[(96usize, 128usize), (160, 64)] {
            let Some(f) = fixture(m, n) else { return };
            let c = tctx_d(&f);
            let r = 16;
            let (up, vp) = nvg::latents(Some(&f.pool), &f.w, r, Some(&f.hc.l), None);
            let (upd, vpd) = latents_d(&c, &c.w, Some(&c.wh), r).unwrap();
            let want = gemm(None, &up, false, &vp, true);
            let got = gemm(None, &Mat::from_vec(m, r, upd.to_f64()), false, &Mat::from_vec(n, r, vpd.to_f64()), true);
            assert!(reld(&got.d, &want.d) < 1e-4, "{m}x{n} latents {}", reld(&got.d, &want.d));
            // ITQ from the same latents
            let wu: Vec<f64> = (0..m).map(|i| up.row(i).iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
            let wv: Vec<f64> = (0..n).map(|j| vp.row(j).iter().map(|v| v * v).sum::<f64>().sqrt() * f.hc.sqd[j]).collect();
            let (iu, _) = nvg::itq_weighted(Some(&f.pool), &up, &vp, &wu, &wv, 50, None);
            let (iud, _) = itq_d(&c, &DMat::from_f64(m, r, &up.d), &DMat::from_f64(n, r, &vp.d), &wu, &wv, 50);
            let agree = iu.d.iter().zip(iud.to_f64()).filter(|(a, b)| (**a >= 0.0) == (*b >= 0.0)).count();
            assert!(agree as f64 >= 0.98 * (m * r) as f64, "ITQ sign agreement {agree}/{}", m * r);
        }
        let mut s = 3u64;
        let mm = Mat::from_vec(40, 40, (0..1600).map(|_| rnd(&mut s)).collect());
        let want = crate::nvg_linalg::polar(None, &mm);
        let got = polar_d(&DMat::from_f64(40, 40, &mm.d), None).to_f64();
        assert!(reld(&got, &want.d) < 1e-4, "polar {}", reld(&got, &want.d));
    }

    #[test]
    fn als_and_knots_match_host() {
        let Some(f) = fixture(160, 128) else { return };
        let c = tctx_d(&f);
        let tc = tctx(&f);
        let r = 24;
        let base = nvg::build_nvg_init(&tc, r, 2);
        // one ALS sweep
        let mut a = base.clone();
        nvg::als(&mut a, &tc, 1);
        let mut ad = model_to_dev(&base);
        als_d(&mut ad, &c, 1);
        for p in 0..2 {
            let (l, h, g) = (ad.paths[p].l.to_f64(), ad.paths[p].h.to_f64(), ad.paths[p].g.to_f64());
            assert!(reld(&l, &a.paths[p].l) < 2e-3, "l{p} {}", reld(&l, &a.paths[p].l));
            assert!(reld(&h, &a.paths[p].h) < 2e-3, "h{p} {}", reld(&h, &a.paths[p].h));
            assert!(reld(&g, &a.paths[p].g) < 2e-3, "g{p} {}", reld(&g, &a.paths[p].g));
        }
        let (e, ed) = (err_cpu(&f, &a, &f.hc.hf), herr_d(&ad, &c, Metric::H));
        assert!((e - ed).abs() <= 1e-3 * e, "als err {ed} vs {e}");
        // output-side flips from the same model
        let mut b = base.clone();
        let nf = nvg::flip_u(&mut b, &tc, 0);
        let mut bd = model_to_dev(&base);
        let nfd = flip_u_d(&mut bd, &c, 0);
        let su = bd.paths[0].su.to_f64();
        let diff = su.iter().zip(&b.paths[0].su.d).filter(|(x, y)| x != y).count();
        assert!(diff * 200 <= su.len(), "flip_u: {diff} of {} signs differ ({nf} vs {nfd} flips)", su.len());
        // input-side search from the same model
        let mut g = base.clone();
        nvg::gptq_v(&mut g, &tc, 1);
        let mut gd = model_to_dev(&base);
        gptq_v_d(&mut gd, &c, 1);
        let sv = gd.paths[1].sv.to_f64();
        let diff = sv.iter().zip(&g.paths[1].sv.d).filter(|(x, y)| x != y).count();
        assert!(diff * 100 <= sv.len(), "gptq_v: {diff} of {} signs differ", sv.len());
        let gg = gd.paths[1].g.to_f64();
        assert!(reld(&gg, &g.paths[1].g) < 2e-2, "gptq_v g {}", reld(&gg, &g.paths[1].g));
        let (e, ed) = (err_cpu(&f, &g, &f.hc.hf), herr_d(&gd, &c, Metric::H));
        assert!((e - ed).abs() <= 1e-2 * e, "gptq_v err {ed} vs {e}");
    }

    /// Whole-tensor driver: a valid payload at the rank rule, the held-out
    /// error within 2 % of the host encoder's.
    #[test]
    fn encode_matches_host() {
        let Some(f) = fixture(160, 128) else { return };
        let b = 2.0;
        let cpu = nvg::encode_nvg(Some(&f.pool), &f.w, &f.hc, &f.hfit, &f.hsel, b, 1.0, PathsChoice::Auto, None).unwrap();
        let gpu = encode_nvg_dev(Some(&f.pool), &f.w, &f.hcd, b, 1.0, PathsChoice::Auto, None).unwrap().unwrap();
        assert!(cortiq_core::blr::validate_blr(&[160, 128], &gpu.bytes).is_ok());
        assert_eq!(gpu.bits, blr_bits(160, 128, gpu.rank, gpu.paths));
        let stored = |bytes: &[u8]| {
            let mut v = vec![0f32; 160 * 128];
            cortiq_core::blr::dequant_blr(bytes, 160, 128, &mut v).unwrap();
            Mat::from_vec(160, 128, v.into_iter().map(|x| x as f64).collect())
        };
        let (ec, eg) = (nvg::rel(None, &f.w, &stored(&cpu.bytes), &f.hsel), nvg::rel(None, &f.w, &stored(&gpu.bytes), &f.hsel));
        eprintln!("cpu: {}\ngpu: {}", cpu.line, gpu.line);
        assert!(eg <= ec * 1.02, "held-out error gpu {eg} vs cpu {ec}");
    }
}

/// One tensor of a concurrent batch ([`encode_many`]).
pub struct Job<'a> {
    pub w: Mat,
    pub hc: &'a HCtxD,
    pub mult: f64,
}

/// What [`encode_many`] reports per job: the encoding, its wall seconds and
/// (when profiling) the stage line.
pub struct Done {
    pub enc: anyhow::Result<Option<Encoded>>,
    pub secs: f64,
    pub prof: Option<String>,
}

/// Encode `jobs` on the device with up to `njobs` tensors in flight (one host
/// thread each, with its own pool of `threads / njobs` workers): the column
/// search of a tensor occupies one multiprocessor, so tensors overlap on the
/// card. Largest tensors start first. `on_done(i, &done)` runs as each
/// finishes; results come back in job order.
pub fn encode_many(
    jobs: &[Job],
    b: f64,
    choice: PathsChoice,
    threads: usize,
    njobs: usize,
    profile: bool,
    on_done: &(dyn Fn(usize, &Done) + Sync),
) -> Vec<Done> {
    let njobs = njobs.clamp(1, jobs.len().max(1));
    let per = (threads / njobs).max(2);
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(jobs[i].w.r * jobs[i].w.c));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Option<Done>>> = std::sync::Mutex::new((0..jobs.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..njobs {
            s.spawn(|| {
                let pool = Pool::new(per - 1);
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= order.len() {
                        break;
                    }
                    let i = order[k];
                    let j = &jobs[i];
                    let prof = Prof::default();
                    let t = Instant::now();
                    let enc = encode_nvg_dev(Some(&pool), &j.w, j.hc, b, j.mult, choice, profile.then_some(&prof));
                    let done = Done { enc, secs: t.elapsed().as_secs_f64(), prof: profile.then(|| prof.line()) };
                    on_done(i, &done);
                    out.lock().unwrap()[i] = Some(done);
                }
            });
        }
    });
    out.into_inner().unwrap().into_iter().map(|d| d.expect("every job ran")).collect()
}
