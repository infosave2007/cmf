//! Binary low-rank encoders — `cortiq convert --quant blr` (docs/BLR_NVG.md §4).
//!
//! Two methods write the same `blr` container (dtype 19,
//! `cortiq_core::blr`):
//!
//! * `nvg` — the NVG method, training-free and data-aware: the topology Ĉ
//!   comes from a whitened SVD (the calibration Gram's Cholesky factor) with
//!   an H-weighted ITQ gauge; the metric Ť is closed form (exact block least
//!   squares for ℓ, h, g in the `tr(E H Eᵀ)` norm); discrete knots by exact
//!   argmin (row flips; column-sequential GPTQ-style sign search on V); and a
//!   held-out choice between P = 2 at rank r and P = 1 at the same bits.
//!   Calibration is SEQUENTIAL: layer `l` is fitted on activations the
//!   engine produces with layers `< l` already in `blr`.
//! * `dual-svid` — the data-free LittleBit initialiser (Lee et al., NeurIPS
//!   2025), kept as the comparison baseline: truncated SVD of `W`, signs of
//!   the latents, rank-1 magnitude fits, residual path on `W − Ŵ_pri`.
//!
//! Numerics: f64 throughout, fixed iteration counts, explicit damping ρ
//! (`--blr-rho`, default 0.03).

use crate::nvg_linalg::{
    Mat, cholesky, gemm, gemm_into, par_chunks, polar, solve_spd, sym_eig, tri_lower_inverse,
};
use cortiq_core::blr::{BlrPathIn, blr_bits, blr_rank, encode_blr};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_core::quant::{dequant_tensor, f16_to_f32, f32_to_f16};
use cortiq_core::types::{QuantType, TensorDtype};
use cortiq_engine::pool::Pool;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[inline]
fn sgn(v: f64) -> f64 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

// ───────────────────────────── stage profile ─────────────────────────────

/// Per-stage wall time and counters of one tensor's encode (`nvg-bench`,
/// and the converter's `CMF_NVG_PROF=1` lines). Timing only: it never
/// changes a result.
#[derive(Default)]
pub struct Prof {
    m: std::sync::Mutex<Vec<(&'static str, f64, u64)>>,
}

impl Prof {
    pub fn add(&self, k: &'static str, secs: f64, n: u64) {
        let mut g = self.m.lock().unwrap();
        match g.iter_mut().find(|e| e.0 == k) {
            Some(e) => {
                e.1 += secs;
                e.2 += n;
            }
            None => g.push((k, secs, n)),
        }
    }
    pub fn entries(&self) -> Vec<(&'static str, f64, u64)> {
        self.m.lock().unwrap().clone()
    }
    /// `stage secs [count]` pairs in first-seen order.
    pub fn line(&self) -> String {
        self.entries()
            .iter()
            .map(|(k, s, n)| if *n > 0 { format!("{k} {s:.2}s/{n}") } else { format!("{k} {s:.2}s") })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// Time `f` under stage `k` (no-op without a profile).
#[inline]
pub fn pt<T>(p: Option<&Prof>, k: &'static str, f: impl FnOnce() -> T) -> T {
    match p {
        Some(p) => {
            let t = Instant::now();
            let v = f();
            p.add(k, t.elapsed().as_secs_f64(), 0);
            v
        }
        None => f(),
    }
}

#[inline]
fn pcount(p: Option<&Prof>, k: &'static str, n: u64) {
    if let Some(p) = p {
        p.add(k, 0.0, n);
    }
}

// ───────────────────────────── model of one tensor ─────────────────────────────

#[derive(Clone)]
pub struct BlrPath {
    pub h: Vec<f64>,
    /// ±1, m × r
    pub su: Mat,
    pub l: Vec<f64>,
    /// ±1, n × r
    pub sv: Mat,
    pub g: Vec<f64>,
}

#[derive(Clone)]
pub struct BlrModel {
    pub paths: Vec<BlrPath>,
    pub m: usize,
    pub n: usize,
    pub r: usize,
}

impl BlrModel {
    /// `diag(h) S_U diag(ℓ)` (m × r).
    fn hsl(&self, p: usize) -> Mat {
        let q = &self.paths[p];
        let mut o = q.su.clone();
        for i in 0..self.m {
            let row = o.row_mut(i);
            for k in 0..self.r {
                row[k] *= q.h[i] * q.l[k];
            }
        }
        o
    }
    /// `diag(h) S_U` (m × r).
    fn hs(&self, p: usize) -> Mat {
        let q = &self.paths[p];
        let mut o = q.su.clone();
        for i in 0..self.m {
            let hi = q.h[i];
            o.row_mut(i).iter_mut().for_each(|v| *v *= hi);
        }
        o
    }
    /// `diag(g) S_V` (n × r).
    fn gs(&self, p: usize) -> Mat {
        let q = &self.paths[p];
        let mut o = q.sv.clone();
        for j in 0..self.n {
            let gj = q.g[j];
            o.row_mut(j).iter_mut().for_each(|v| *v *= gj);
        }
        o
    }
    pub fn recon(&self, pool: Option<&Pool>, skip: Option<usize>) -> Mat {
        let mut out = Mat::zeros(self.m, self.n);
        for p in 0..self.paths.len() {
            if Some(p) == skip {
                continue;
            }
            gemm_into(
                pool,
                1.0,
                &self.hsl(p),
                false,
                &self.gs(p),
                true,
                1.0,
                &mut out,
            );
        }
        out
    }
    /// `Σ_{q≠p} Ŵ_q H` without forming `Ŵ_q H` the long way.
    fn recon_h(&self, pool: Option<&Pool>, hf: &Mat, skip: usize) -> Mat {
        let mut out = Mat::zeros(self.m, self.n);
        for q in 0..self.paths.len() {
            if q == skip {
                continue;
            }
            let hb = gemm(pool, hf, false, &self.gs(q), false);
            gemm_into(pool, 1.0, &self.hsl(q), false, &hb, true, 1.0, &mut out);
        }
        out
    }
}

/// `tr(E H Eᵀ)`.
pub fn herr(pool: Option<&Pool>, e: &Mat, h: &Mat) -> f64 {
    let eh = gemm(pool, e, false, h, false);
    eh.d.iter().zip(&e.d).map(|(a, b)| a * b).sum()
}

/// `sqrt(tr((W−Ŵ)H(W−Ŵ)ᵀ) / tr(W H Wᵀ))`.
pub fn rel(pool: Option<&Pool>, w: &Mat, what: &Mat, h: &Mat) -> f64 {
    (herr(pool, &w.sub(what), h) / herr(pool, w, h).max(1e-300)).sqrt()
}

// ───────────────────────────── per-input context ─────────────────────────────

/// Everything derived from one input's calibration Gram, shared by the
/// tensors that read that input.
pub struct HCtx {
    /// `XXᵀ/T + ρ·tr/n·I`
    pub hf: Mat,
    /// lower Cholesky factor of `hf`
    pub l: Mat,
    /// `sqrt(diag(hf))`
    pub sqd: Vec<f64>,
    /// act order (diag descending) and the upper Cholesky factor of the
    /// permuted inverse, for the column-sequential V search
    pub perm: Vec<usize>,
    pub uc: Mat,
}

impl HCtx {
    pub fn new(pool: Option<&Pool>, h_mean: &Mat, rho: f64) -> anyhow::Result<Self> {
        let n = h_mean.r;
        let mut hf = h_mean.clone();
        let tr = (0..n).map(|i| hf.at(i, i)).sum::<f64>() / n as f64;
        for i in 0..n {
            hf.d[i * n + i] += rho * tr;
        }
        let l = cholesky(pool, &hf).ok_or_else(|| anyhow::anyhow!("calibration Gram not SPD"))?;
        let sqd: Vec<f64> = (0..n).map(|i| hf.at(i, i).sqrt()).collect();
        let mut perm: Vec<usize> = (0..n).collect();
        let dg = hf.diag();
        perm.sort_by(|&a, &b| {
            dg[b]
                .partial_cmp(&dg[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Uc = J · inv(chol(J Hp J)) · J  is the upper Cholesky factor of Hp⁻¹.
        let mut jhj = Mat::zeros(n, n);
        for i in 0..n {
            let pi = perm[n - 1 - i];
            for j in 0..n {
                jhj.d[i * n + j] = hf.at(pi, perm[n - 1 - j]);
            }
        }
        let lp = cholesky(pool, &jhj).ok_or_else(|| anyhow::anyhow!("permuted Gram not SPD"))?;
        let li = tri_lower_inverse(pool, &lp);
        let mut uc = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                uc.d[i * n + j] = li.at(n - 1 - i, n - 1 - j);
            }
        }
        Ok(HCtx {
            hf,
            l,
            sqd,
            perm,
            uc,
        })
    }
}

/// One tensor's fitting context.
pub(crate) struct TCtx<'a> {
    pub(crate) w: &'a Mat,
    pub(crate) wh: Mat,
    pub(crate) hc: &'a HCtx,
    pub(crate) pool: Option<&'a Pool>,
    pub(crate) prof: Option<&'a Prof>,
}

// ───────────────────────────── initialisers ─────────────────────────────

/// Rank-`r` latents `U′ (m×r)`, `V′ (n×r)` with `U′V′ᵀ` the best rank-`r`
/// approximation of `t` in the metric `L Lᵀ` (identity when `l` is None):
/// `U′ = U_r Σ^{1/2}`, `V′ = tᵀ U_r Σ^{-1/2}`.
pub(crate) fn latents(
    pool: Option<&Pool>,
    t: &Mat,
    r: usize,
    l: Option<&Mat>,
    prof: Option<&Prof>,
) -> (Mat, Mat) {
    let (m, n) = (t.r, t.c);
    let tl = pt(prof, "lat.gemm", || match l {
        Some(l) => gemm(pool, t, false, l, false),
        None => t.clone(),
    });
    let (sig, ur) = if m <= n {
        let g = pt(prof, "lat.gemm", || gemm(pool, &tl, false, &tl, true));
        let (lam, e) = pt(prof, "lat.eig", || sym_eig(pool, &g));
        let sig: Vec<f64> = lam[..r].iter().map(|v| v.max(0.0).sqrt()).collect();
        let mut ur = Mat::zeros(m, r);
        for i in 0..m {
            ur.row_mut(i).copy_from_slice(&e.row(i)[..r]);
        }
        (sig, ur)
    } else {
        let g = pt(prof, "lat.gemm", || gemm(pool, &tl, true, &tl, false));
        let (lam, e) = pt(prof, "lat.eig", || sym_eig(pool, &g));
        let sig: Vec<f64> = lam[..r].iter().map(|v| v.max(0.0).sqrt()).collect();
        let mut vr = Mat::zeros(n, r);
        for j in 0..n {
            vr.row_mut(j).copy_from_slice(&e.row(j)[..r]);
        }
        let mut ur = pt(prof, "lat.gemm", || gemm(pool, &tl, false, &vr, false));
        let floor = sig[0].max(1e-300) * 1e-15;
        for i in 0..m {
            let row = ur.row_mut(i);
            for k in 0..r {
                row[k] /= sig[k].max(floor);
            }
        }
        (sig, ur)
    };
    let floor = sig[0].max(1e-300) * 1e-15;
    let mut up = ur.clone();
    for i in 0..m {
        let row = up.row_mut(i);
        for k in 0..r {
            row[k] *= sig[k].sqrt();
        }
    }
    let mut vp = pt(prof, "lat.gemm", || gemm(pool, t, true, &ur, false));
    for j in 0..n {
        let row = vp.row_mut(j);
        for k in 0..r {
            row[k] /= sig[k].max(floor).sqrt();
        }
    }
    (up, vp)
}

fn row_norms(a: &Mat) -> Vec<f64> {
    (0..a.r)
        .map(|i| a.row(i).iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

/// H-weighted ITQ gauge: orthogonal `R` making the row-normalised
/// `[U′; V′] R` closest to the hypercube, rows weighted by `w²`.
pub(crate) fn itq_weighted(
    pool: Option<&Pool>,
    up: &Mat,
    vp: &Mat,
    wu: &[f64],
    wv: &[f64],
    iters: usize,
    prof: Option<&Prof>,
) -> (Mat, Mat) {
    let (m, n, r) = (up.r, vp.r, up.c);
    let nn = m + n;
    let mut x = Mat::zeros(nn, r);
    for i in 0..m {
        x.row_mut(i).copy_from_slice(up.row(i));
    }
    for j in 0..n {
        x.row_mut(m + j).copy_from_slice(vp.row(j));
    }
    for i in 0..nn {
        let row = x.row_mut(i);
        let nr = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        row.iter_mut().for_each(|v| *v /= nr);
    }
    let mut w2: Vec<f64> = wu.iter().chain(wv).map(|v| v * v).collect();
    let mean = w2.iter().sum::<f64>() / nn as f64;
    w2.iter_mut().for_each(|v| *v /= mean.max(1e-300));
    let mut rot = Mat::eye(r);
    let mut bw = Mat::zeros(nn, r);
    for _ in 0..iters {
        let xr = pt(prof, "itq.gemm", || gemm(pool, &x, false, &rot, false));
        for i in 0..nn {
            let (src, dst) = (xr.row(i), &mut bw.d[i * r..(i + 1) * r]);
            for k in 0..r {
                dst[k] = sgn(src[k]) * w2[i];
            }
        }
        let mm = pt(prof, "itq.gemm", || gemm(pool, &x, true, &bw, false));
        rot = pt(prof, "itq.polar", || polar(pool, &mm));
    }
    pt(prof, "itq.gemm", || {
        (
            gemm(pool, up, false, &rot, false),
            gemm(pool, vp, false, &rot, false),
        )
    })
}

/// Non-negative rank-1 fit `M ≈ left · rightᵀ` by power iteration from 1/√c.
fn rank1_pos(mabs: &Mat) -> (Vec<f64>, Vec<f64>) {
    let (rows, c) = (mabs.r, mabs.c);
    let mut v = vec![1.0 / (c as f64).sqrt(); c];
    let mv = |v: &[f64]| -> Vec<f64> {
        (0..rows)
            .map(|i| mabs.row(i).iter().zip(v).map(|(a, b)| a * b).sum())
            .collect()
    };
    for _ in 0..60 {
        let u = mv(&v);
        let mut v2 = vec![0f64; c];
        for i in 0..rows {
            let ui = u[i];
            for (o, a) in v2.iter_mut().zip(mabs.row(i)) {
                *o += a * ui;
            }
        }
        let nv = v2.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v2.iter_mut().for_each(|x| *x /= nv);
        let diff = v2
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        v = v2;
        if diff < 1e-9 {
            break;
        }
    }
    let mut u = mv(&v);
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

/// Signs of the latents plus rank-1 magnitude fits (the dual-SVID step).
pub(crate) fn svid_path(up: &Mat, vp: &Mat) -> BlrPath {
    let su = Mat::from_vec(up.r, up.c, up.d.iter().map(|&v| sgn(v)).collect());
    let sv = Mat::from_vec(vp.r, vp.c, vp.d.iter().map(|&v| sgn(v)).collect());
    let ua = Mat::from_vec(up.r, up.c, up.d.iter().map(|v| v.abs()).collect());
    let va = Mat::from_vec(vp.r, vp.c, vp.d.iter().map(|v| v.abs()).collect());
    let (h, lu) = rank1_pos(&ua);
    let (g, lv) = rank1_pos(&va);
    let l = lu.iter().zip(&lv).map(|(a, b)| a * b).collect();
    BlrPath { h, su, l, sv, g }
}

/// The data-free dual-SVID initialiser (the LittleBit baseline).
pub fn build_dual_svid(pool: Option<&Pool>, w: &Mat, r: usize, paths: usize) -> BlrModel {
    let (up, vp) = latents(pool, w, r, None, None);
    let mut md = BlrModel {
        paths: vec![svid_path(&up, &vp)],
        m: w.r,
        n: w.c,
        r,
    };
    if paths == 2 {
        let res = w.sub(&md.recon(pool, None));
        let (up, vp) = latents(pool, &res, r, None, None);
        md.paths.push(svid_path(&up, &vp));
    }
    md
}

// ───────────────────────────── closed-form metric (Ť) ─────────────────────────────

fn hstack(ms: &[Mat]) -> Mat {
    let rows = ms[0].r;
    let cols: usize = ms.iter().map(|m| m.c).sum();
    let mut o = Mat::zeros(rows, cols);
    for i in 0..rows {
        let mut off = 0;
        for m in ms {
            o.d[i * cols + off..i * cols + off + m.c].copy_from_slice(m.row(i));
            off += m.c;
        }
    }
    o
}

/// Exact block least squares in the H metric: ℓ (joint over paths), h (per
/// row, joint over paths), g (per path). Each block solve is a minimum, so
/// the error never increases.
pub(crate) fn als(md: &mut BlrModel, c: &TCtx, sweeps: usize) {
    let (m, n, r) = (md.m, md.n, md.r);
    let np = md.paths.len();
    let pr = np * r;
    let pool = c.pool;
    let hf = &c.hc.hf;
    let prof = c.prof;
    for _ in 0..sweeps {
        // ℓ
        let t_l = Instant::now();
        let bs: Vec<Mat> = (0..np).map(|p| md.gs(p)).collect();
        let b = hstack(&bs);
        let hb = gemm(pool, hf, false, &b, false);
        let bhb = gemm(pool, &b, true, &hb, false);
        let whb = gemm(pool, &c.wh, false, &b, false);
        let a = hstack(&(0..np).map(|p| md.hs(p)).collect::<Vec<_>>());
        let ata = gemm(pool, &a, true, &a, false);
        let mut sys = Mat::zeros(pr, pr);
        for i in 0..pr * pr {
            sys.d[i] = ata.d[i] * bhb.d[i];
        }
        let mut rhs = vec![0f64; pr];
        for i in 0..m {
            let (ar, wr) = (a.row(i), whb.row(i));
            for k in 0..pr {
                rhs[k] += ar[k] * wr[k];
            }
        }
        let lam = solve_spd(pool, &sys, &rhs, 1e-10);
        for p in 0..np {
            md.paths[p].l.copy_from_slice(&lam[p * r..(p + 1) * r]);
        }
        if let Some(pf) = prof {
            pf.add("als.l", t_l.elapsed().as_secs_f64(), 1);
        }
        let t_h = Instant::now();
        // h (B unchanged since the ℓ step: reuse its blocks)
        let ms: Vec<Mat> = (0..np)
            .map(|p| {
                let q = &md.paths[p];
                let mut o = q.su.clone();
                for i in 0..m {
                    let row = o.row_mut(i);
                    for k in 0..r {
                        row[k] *= q.l[k];
                    }
                }
                o
            })
            .collect();
        let mut gm = vec![[[0f64; 2]; 2]; m];
        let mut rh = vec![[0f64; 2]; m];
        for a_ in 0..np {
            for i in 0..m {
                let (wr, mr) = (&whb.row(i)[a_ * r..(a_ + 1) * r], ms[a_].row(i));
                rh[i][a_] = wr.iter().zip(mr).map(|(x, y)| x * y).sum();
            }
            for c_ in a_..np {
                let mut kac = Mat::zeros(r, r);
                for x in 0..r {
                    kac.row_mut(x)
                        .copy_from_slice(&bhb.row(a_ * r + x)[c_ * r..(c_ + 1) * r]);
                }
                let mk = gemm(pool, &ms[a_], false, &kac, false);
                for i in 0..m {
                    let v: f64 = mk
                        .row(i)
                        .iter()
                        .zip(ms[c_].row(i))
                        .map(|(x, y)| x * y)
                        .sum();
                    gm[i][a_][c_] = v;
                    gm[i][c_][a_] = v;
                }
            }
        }
        for i in 0..m {
            let tr: f64 = (0..np).map(|a_| gm[i][a_][a_]).sum::<f64>() / np as f64;
            let ridge = 1e-10 * tr.max(1e-300);
            if np == 1 {
                md.paths[0].h[i] = rh[i][0] / (gm[i][0][0] + ridge);
            } else {
                let (a11, a12, a22) = (gm[i][0][0] + ridge, gm[i][0][1], gm[i][1][1] + ridge);
                let det = a11 * a22 - a12 * a12;
                let (h0, h1) = if det.abs() > 0.0 {
                    (
                        (a22 * rh[i][0] - a12 * rh[i][1]) / det,
                        (a11 * rh[i][1] - a12 * rh[i][0]) / det,
                    )
                } else {
                    (md.paths[0].h[i], md.paths[1].h[i])
                };
                md.paths[0].h[i] = h0;
                md.paths[1].h[i] = h1;
            }
        }
        if let Some(pf) = prof {
            pf.add("als.h", t_h.elapsed().as_secs_f64(), 1);
        }
        let t_g = Instant::now();
        // g
        for p in 0..np {
            let wt_h = if np > 1 {
                c.wh.sub(&md.recon_h(pool, hf, p))
            } else {
                c.wh.clone()
            };
            let mm = md.hsl(p);
            let cmat = gemm(pool, &mm, false, &md.paths[p].sv, true);
            let k = gemm(pool, &mm, true, &mm, false);
            let svk = gemm(pool, &md.paths[p].sv, false, &k, false);
            let mut ctc = gemm(pool, &svk, false, &md.paths[p].sv, true);
            for i in 0..n * n {
                ctc.d[i] *= hf.d[i];
            }
            let mut rhs = vec![0f64; n];
            for i in 0..m {
                let (cr, wr) = (cmat.row(i), wt_h.row(i));
                for j in 0..n {
                    rhs[j] += cr[j] * wr[j];
                }
            }
            md.paths[p].g = solve_spd(pool, &ctc, &rhs, 1e-10);
        }
        if let Some(pf) = prof {
            pf.add("als.g", t_g.elapsed().as_secs_f64(), 1);
        }
    }
}

// ───────────────────────────── discrete knots ─────────────────────────────

#[derive(Clone, Copy)]
struct SP(*mut f64);
unsafe impl Send for SP {}
unsafe impl Sync for SP {}

/// Output-side knots: per row, exact single-bit flip deltas of the binary QP,
/// greedy passes until no flip (≤ 4), monotone.
pub(crate) fn flip_u(md: &mut BlrModel, c: &TCtx, pi: usize) -> usize {
    let t0 = Instant::now();
    let n = flip_u_inner(md, c, pi);
    if let Some(pf) = c.prof {
        pf.add("flip_u", t0.elapsed().as_secs_f64(), 1);
        pf.add("flip_u.flips", 0.0, n as u64);
    }
    n
}

fn flip_u_inner(md: &mut BlrModel, c: &TCtx, pi: usize) -> usize {
    let (m, r) = (md.m, md.r);
    let pool = c.pool;
    let hf = &c.hc.hf;
    let wt_h = if md.paths.len() > 1 {
        c.wh.sub(&md.recon_h(pool, hf, pi))
    } else {
        c.wh.clone()
    };
    let b = md.gs(pi);
    let hb = gemm(pool, hf, false, &b, false);
    let k = gemm(pool, &b, true, &hb, false);
    let whb = gemm(pool, &wt_h, false, &b, false);
    let l = md.paths[pi].l.clone();
    let mut q = Mat::zeros(r, r);
    for x in 0..r {
        for y in 0..r {
            q.d[x * r + y] = l[x] * k.at(x, y) * l[y];
        }
    }
    let qd = q.diag();
    let mut qs = gemm(pool, &md.paths[pi].su, false, &q, false);
    let h = md.paths[pi].h.clone();
    let su = SP(md.paths[pi].su.d.as_mut_ptr());
    let qsp = SP(qs.d.as_mut_ptr());
    let flips = std::sync::atomic::AtomicUsize::new(0);
    par_chunks(pool, m, 16, &|lo, hi| {
        let (su, qsp) = (su, qsp);
        let mut nf = 0usize;
        for i in lo..hi {
            let s = unsafe { std::slice::from_raw_parts_mut(su.0.add(i * r), r) };
            let qsr = unsafe { std::slice::from_raw_parts_mut(qsp.0.add(i * r), r) };
            let hi_ = h[i];
            let wr = whb.row(i);
            for _ in 0..4 {
                let mut changed = 0;
                for kk in 0..r {
                    let bm = wr[kk] * l[kk];
                    let d = -4.0 * hi_ * hi_ * s[kk] * (qsr[kk] - qd[kk] * s[kk])
                        + 4.0 * hi_ * s[kk] * bm;
                    if d < -1e-12 * (hi_ * hi_ * qd[kk] + 1e-300) {
                        let sk = s[kk];
                        let qrow = q.row(kk);
                        for t in 0..r {
                            qsr[t] -= 2.0 * sk * qrow[t];
                        }
                        s[kk] = -sk;
                        changed += 1;
                    }
                }
                nf += changed;
                if changed == 0 {
                    break;
                }
            }
        }
        flips.fetch_add(nf, std::sync::atomic::Ordering::Relaxed);
    });
    flips.into_inner()
}

/// Input-side knots: column-sequential argmin with Schur-complement (H⁻¹)
/// error feedback (OBS/GPTQ style), act order, lazy 128-column blocks.
pub(crate) fn gptq_v(md: &mut BlrModel, c: &TCtx, pi: usize) {
    let t0 = Instant::now();
    let flips = gptq_v_inner(md, c, pi);
    if let Some(pf) = c.prof {
        pf.add("gptq_v", t0.elapsed().as_secs_f64(), 1);
        pf.add("gptq_v.flips", 0.0, flips);
    }
}

fn gptq_v_inner(md: &mut BlrModel, c: &TCtx, pi: usize) -> u64 {
    let (m, n, r) = (md.m, md.n, md.r);
    let pool = c.pool;
    let wt = if md.paths.len() > 1 {
        c.w.sub(&md.recon(pool, Some(pi)))
    } else {
        c.w.clone()
    };
    let mm = md.hsl(pi);
    let k = gemm(pool, &mm, true, &mm, false);
    let kd = k.diag();
    let perm = &c.hc.perm;
    let uc = &c.hc.uc;
    // WpT row j = column perm[j] of Wt
    let mut wpt = Mat::zeros(n, m);
    for i in 0..m {
        let row = wt.row(i);
        for j in 0..n {
            wpt.d[j * m + i] = row[perm[j]];
        }
    }
    let mut s_all = Mat::zeros(n, r);
    let mut g = vec![0f64; n];
    for j in 0..n {
        s_all
            .row_mut(j)
            .copy_from_slice(md.paths[pi].sv.row(perm[j]));
        g[j] = md.paths[pi].g[perm[j]];
    }
    let max_flips = 4 * r;
    let mut total_flips = 0u64;
    let block = 128usize;
    let mut b0 = 0;
    let mut bv = vec![0f64; r];
    let mut ks = vec![0f64; r];
    let mut dlt = vec![0f64; r];
    let mut qv = vec![0f64; m];
    while b0 < n {
        let b1 = (b0 + block).min(n);
        let mut errt = Mat::zeros(b1 - b0, m);
        for j in b0..b1 {
            // bv = Mᵀ w
            bv.iter_mut().for_each(|v| *v = 0.0);
            {
                let w = &wpt.d[j * m..(j + 1) * m];
                for i in 0..m {
                    let wi = w[i];
                    if wi == 0.0 {
                        continue;
                    }
                    for (o, x) in bv.iter_mut().zip(mm.row(i)) {
                        *o += x * wi;
                    }
                }
            }
            let s = s_all.row_mut(j);
            for x in 0..r {
                ks[x] = k.row(x).iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            }
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let mut gj = dot(&bv, s) / dot(s, &ks).max(1e-300);
            if gj < 0.0 {
                s.iter_mut().for_each(|v| *v = -*v);
                ks.iter_mut().for_each(|v| *v = -*v);
                gj = -gj;
            }
            for _ in 0..3 {
                let mut nf = 0;
                while nf < max_flips {
                    let mut best = 0usize;
                    for x in 0..r {
                        dlt[x] = -4.0 * gj * gj * s[x] * (ks[x] - kd[x] * s[x])
                            + 4.0 * gj * s[x] * bv[x];
                        if dlt[x] < dlt[best] {
                            best = x;
                        }
                    }
                    if dlt[best] >= -1e-12 * (gj * gj * kd[best] + 1e-300) {
                        break;
                    }
                    let sb = s[best];
                    for (o, kv) in ks.iter_mut().zip(k.row(best)) {
                        *o -= 2.0 * sb * kv;
                    }
                    s[best] = -sb;
                    nf += 1;
                }
                total_flips += nf as u64;
                let mut g2 = dot(&bv, s) / dot(s, &ks).max(1e-300);
                if g2 < 0.0 {
                    s.iter_mut().for_each(|v| *v = -*v);
                    ks.iter_mut().for_each(|v| *v = -*v);
                    g2 = -g2;
                }
                if nf == 0 || (g2 - gj).abs() <= 1e-9 * gj.abs() {
                    gj = g2;
                    break;
                }
                gj = g2;
            }
            // q = gj · M s ; e = (w − q)/Uc[j,j]
            for i in 0..m {
                qv[i] = gj * dot(mm.row(i), s);
            }
            let ujj = uc.at(j, j);
            let e: Vec<f64> = (0..m).map(|i| (wpt.d[j * m + i] - qv[i]) / ujj).collect();
            for jj in j + 1..b1 {
                let cf = uc.at(j, jj);
                if cf != 0.0 {
                    let row = &mut wpt.d[jj * m..(jj + 1) * m];
                    for i in 0..m {
                        row[i] -= cf * e[i];
                    }
                }
            }
            errt.row_mut(j - b0).copy_from_slice(&e);
            g[j] = gj;
        }
        if b1 < n {
            // WpT[b1:, :] -= Uc[b0:b1, b1:]ᵀ · ErrT
            let mut ucb = Mat::zeros(b1 - b0, n - b1);
            for x in b0..b1 {
                ucb.row_mut(x - b0).copy_from_slice(&uc.row(x)[b1..]);
            }
            let upd = gemm(pool, &ucb, true, &errt, false);
            for (o, u) in wpt.d[b1 * m..].iter_mut().zip(&upd.d) {
                *o -= u;
            }
        }
        b0 = b1;
    }
    let path = &mut md.paths[pi];
    for j in 0..n {
        path.sv.row_mut(perm[j]).copy_from_slice(s_all.row(j));
        path.g[perm[j]] = g[j];
    }
    total_flips
}

/// Discrete-knot refinement alternating with the exact metric; a step is kept
/// only when it lowers the fit error.
fn refine(mut md: BlrModel, c: &TCtx, sweeps: usize) -> BlrModel {
    let pool = c.pool;
    let prof = c.prof;
    let mut cur = pt(prof, "err", || herr(pool, &c.w.sub(&md.recon(pool, None)), &c.hc.hf));
    for _ in 0..sweeps {
        for pi in 0..md.paths.len() {
            for kind in 0..2 {
                let mut trial = md.clone();
                if kind == 0 {
                    flip_u(&mut trial, c, pi);
                } else {
                    gptq_v(&mut trial, c, pi);
                }
                als(&mut trial, c, 3);
                let e = pt(prof, "err", || herr(pool, &c.w.sub(&trial.recon(pool, None)), &c.hc.hf));
                if e < cur {
                    md = trial;
                    cur = e;
                }
            }
        }
    }
    md
}

/// Whitened topology, H-weighted ITQ gauge, closed-form metric; the
/// residual path on the H-residual of the refitted primary.
pub(crate) fn build_nvg_init(c: &TCtx, r: usize, paths: usize) -> BlrModel {
    let pool = c.pool;
    let prof = c.prof;
    let lat = |t: &Mat| {
        let (up, vp) = latents(pool, t, r, Some(&c.hc.l), prof);
        let wu = row_norms(&up);
        let wv: Vec<f64> = row_norms(&vp)
            .iter()
            .zip(&c.hc.sqd)
            .map(|(a, b)| a * b)
            .collect();
        let (up, vp) = itq_weighted(pool, &up, &vp, &wu, &wv, 50, prof);
        pt(prof, "svid", || svid_path(&up, &vp))
    };
    let mut md = BlrModel {
        paths: vec![lat(c.w)],
        m: c.w.r,
        n: c.w.c,
        r,
    };
    als(&mut md, c, 3);
    if paths == 2 {
        let res = pt(prof, "recon", || c.w.sub(&md.recon(pool, None)));
        md.paths.push(lat(&res));
        als(&mut md, c, 6);
    }
    md
}

// ───────────────────────────── packing ─────────────────────────────

/// Rebalance (h/a, g/c, ℓ·a·c with a = rms h, c = rms g), round scales to
/// f16, pack. Returns the bytes and the model with the stored (rounded)
/// scales. None when a scale is not finite or a nonzero ℓ underflows f16.
pub fn pack(md: &BlrModel) -> Option<(Vec<u8>, BlrModel)> {
    let mut st = md.clone();
    for p in st.paths.iter_mut() {
        let a = (p.h.iter().map(|v| v * v).sum::<f64>() / p.h.len() as f64).sqrt();
        let c = (p.g.iter().map(|v| v * v).sum::<f64>() / p.g.len() as f64).sqrt();
        if !(a > 0.0 && c > 0.0 && a.is_finite() && c.is_finite()) {
            return None;
        }
        p.h.iter_mut().for_each(|v| *v /= a);
        p.g.iter_mut().for_each(|v| *v /= c);
        p.l.iter_mut().for_each(|v| *v *= a * c);
        let q = |v: f64| f16_to_f32(f32_to_f16(v as f32)) as f64;
        for v in p.l.iter() {
            if !v.is_finite() || (*v != 0.0 && (q(*v) == 0.0 || v.abs() < 6.1e-5)) {
                return None;
            }
        }
        if p.h
            .iter()
            .chain(&p.g)
            .any(|v| !v.is_finite() || v.abs() > 65000.0)
        {
            return None;
        }
        p.h.iter_mut().for_each(|v| *v = q(*v));
        p.g.iter_mut().for_each(|v| *v = q(*v));
        p.l.iter_mut().for_each(|v| *v = q(*v));
    }
    let (m, n, r) = (md.m, md.n, md.r);
    let owned: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>, Vec<bool>)> = st
        .paths
        .iter()
        .map(|p| {
            (
                p.h.iter().map(|&v| v as f32).collect(),
                p.g.iter().map(|&v| v as f32).collect(),
                p.l.iter().map(|&v| v as f32).collect(),
                p.su.d.iter().map(|&v| v > 0.0).collect(),
                p.sv.d.iter().map(|&v| v > 0.0).collect(),
            )
        })
        .collect();
    let ins: Vec<BlrPathIn> = owned
        .iter()
        .map(|(h, g, l, su, sv)| BlrPathIn { h, g, l, su, sv })
        .collect();
    Some((encode_blr(m, n, r, &ins), st))
}

// ───────────────────────────── per-tensor drivers ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    Nvg,
    DualSvid,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Nvg => "nvg",
            Method::DualSvid => "dual-svid",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PathsChoice {
    /// nvg: argmin over {P=2 at r, P=1 at the same bits} on the held-out
    /// slice; dual-svid: P=2
    Auto,
    One,
    Two,
}

impl PathsChoice {
    pub fn name(self) -> &'static str {
        match self {
            PathsChoice::Auto => "auto",
            PathsChoice::One => "1",
            PathsChoice::Two => "2",
        }
    }
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub paths: usize,
    pub rank: usize,
    /// Strict bit count: scales/signs only, excluding the fixed 16-byte
    /// payload header.
    pub bits: u64,
    pub line: String,
}

/// Dual-SVID for one tensor (data-free; the LittleBit baseline).
pub fn encode_dual_svid(
    pool: Option<&Pool>,
    w: &Mat,
    b: f64,
    mult: f64,
    paths: usize,
) -> Option<Encoded> {
    let (m, n) = (w.r, w.c);
    let r = blr_rank(m, n, b, paths, mult);
    let md = build_dual_svid(pool, w, r, paths);
    let (bytes, st) = pack(&md)?;
    let wf = (w.sub(&st.recon(pool, None)).frob2() / w.frob2().max(1e-300)).sqrt();
    Some(Encoded {
        bytes,
        paths,
        rank: r,
        bits: blr_bits(m, n, r, paths),
        line: format!("P{paths} r{r} frob {wf:.4}"),
    })
}

/// NVG for one tensor. `hsel` / `hfit` are the raw (undamped) mean Grams
/// used for selection and reporting.
#[allow(clippy::too_many_arguments)]
pub fn encode_nvg(
    pool: Option<&Pool>,
    w: &Mat,
    hc: &HCtx,
    hfit: &Mat,
    hsel: &Mat,
    b: f64,
    mult: f64,
    choice: PathsChoice,
    prof: Option<&Prof>,
) -> Option<Encoded> {
    let (m, n) = (w.r, w.c);
    let c = TCtx {
        w,
        wh: pt(prof, "wh", || gemm(pool, w, false, &hc.hf, false)),
        hc,
        pool,
        prof,
    };
    let mut cands: Vec<(usize, BlrModel)> = Vec::new();
    let mut notes = Vec::new();
    if choice != PathsChoice::One {
        let r2 = blr_rank(m, n, b, 2, mult);
        let n3 = build_nvg_init(&c, r2, 2);
        let e3 = pt(prof, "select", || rel(pool, w, &n3.recon(pool, None), hsel));
        let n4 = refine(n3, &c, 2);
        notes.push(format!("init {e3:.4}"));
        cands.push((2, n4));
    }
    if choice != PathsChoice::Two {
        let r1 = blr_rank(m, n, b, 1, mult);
        let p1 = refine(build_nvg_init(&c, r1, 1), &c, 2);
        cands.push((1, p1));
    }
    let mut best: Option<(f64, usize)> = None;
    for (ci, (p, md)) in cands.iter().enumerate() {
        let e = pt(prof, "select", || rel(pool, w, &md.recon(pool, None), hsel));
        notes.push(format!("P{p} {e:.4}"));
        if best.is_none_or(|(be, _)| e < be) {
            best = Some((e, ci));
        }
    }
    let (_, bi) = best?;
    let (p, md) = cands.swap_remove(bi);
    let (bytes, st) = pack(&md)?;
    let (ef, es) = pt(prof, "select", || {
        let rhat = st.recon(pool, None);
        (rel(pool, w, &rhat, hfit), rel(pool, w, &rhat, hsel))
    });
    Some(Encoded {
        bytes,
        paths: p,
        rank: md.r,
        bits: blr_bits(m, n, md.r, p),
        line: format!(
            "pick P{p} r{} | sel: {} | stored fit {ef:.4} sel {es:.4}",
            md.r,
            notes.join(" ")
        ),
    })
}

// ───────────────────────────── conversion driver ─────────────────────────────

/// Where the NVG encoder and its calibration run.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BlrDevice {
    /// f64 on the host (the reference), calibration on the CPU prefill
    Cpu,
    /// f32 on the wgpu device, calibration on per-op device GEMMs with
    /// device Grams (`nvg_gpu.rs`)
    Gpu,
}

impl BlrDevice {
    pub fn name(self) -> &'static str {
        match self {
            BlrDevice::Cpu => "cpu",
            BlrDevice::Gpu => "gpu",
        }
    }
    /// `auto`: the device when a wgpu adapter is up (not on macOS, whose
    /// Metal chunk graph runs layer spans out of the capture hook's sight).
    pub fn resolve(s: &str) -> anyhow::Result<Self> {
        match s {
            "cpu" => Ok(BlrDevice::Cpu),
            "gpu" => {
                anyhow::ensure!(cfg!(feature = "gpu"), "--blr-device gpu needs a build with the gpu feature");
                Ok(BlrDevice::Gpu)
            }
            "auto" => Ok(
                if cfg!(feature = "gpu")
                    && !cfg!(target_os = "macos")
                    && std::env::var("CMF_GPU").as_deref() != Ok("0")
                    && cortiq_engine::gpu::backend_available()
                {
                    BlrDevice::Gpu
                } else {
                    BlrDevice::Cpu
                },
            ),
            other => anyhow::bail!("--blr-device {other}: expected auto, cpu or gpu"),
        }
    }
}

pub struct BlrOpts {
    pub bpw: f64,
    pub method: Method,
    pub calib: Option<String>,
    pub kv_factor: f64,
    pub paths: PathsChoice,
    pub fit_tokens: usize,
    pub sel_tokens: usize,
    pub window: usize,
    pub rho: f64,
    pub threads: usize,
    pub keep_src: bool,
    pub device: BlrDevice,
}

/// The calibration work file of layer `l`: only layers `0..=l` (later
/// layers are never run), no MTP head. None when the architecture carries
/// per-layer structure this cut does not know how to shorten.
fn truncated_header(header: &cortiq_core::format::CmfHeader, l: usize) -> Option<cortiq_core::format::CmfHeader> {
    let a = &header.arch;
    if a.moe.is_some()
        || a.qwen4_exp.is_some()
        || a.mla.is_some()
        || a.linear_core.is_some()
        || a.anchor_core.is_some()
        || a.g3n.is_some()
        || a.prism_hadamard.is_some()
        || a.deepseek_v41.is_some()
        || a.num_loops > 1
        || l + 1 >= a.num_layers
    {
        return None;
    }
    let mut h = header.clone();
    let nl = l + 1;
    h.arch.num_layers = nl;
    if h.arch.layer_types.len() > nl {
        h.arch.layer_types.truncate(nl);
    }
    for v in [&mut h.arch.attention_heads_per_layer, &mut h.arch.kv_heads_per_layer].into_iter().flatten() {
        v.truncate(nl);
    }
    h.arch.mtp = None;
    Some(h)
}

fn keep_in_truncated(name: &str, l: usize) -> bool {
    if name.starts_with("model.mtp.") || name.starts_with("mtp.") {
        return false;
    }
    match name.strip_prefix("model.layers.").and_then(|r| r.split_once('.')).and_then(|(k, _)| k.parse::<usize>().ok()) {
        Some(k) => k <= l,
        None => true,
    }
}

/// Body linears the encoder replaces: `model.layers.{l}.{tail}` → (l, short).
/// Attention q/k/v/o, dense FFN gate/up/down, and the GDN block's
/// in_proj_qkv / in_proj_z / out_proj (its tiny in_proj_a/b stay f16 — the
/// converter's force-f16 rule).
pub(crate) fn body_linear(name: &str) -> Option<(usize, &'static str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, tail) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    let short = match tail {
        "self_attn.q_proj.weight" => "q",
        "self_attn.k_proj.weight" => "k",
        "self_attn.v_proj.weight" => "v",
        "self_attn.o_proj.weight" => "o",
        "mlp.gate_proj.weight" => "gate",
        "mlp.up_proj.weight" => "up",
        "mlp.down_proj.weight" => "down",
        "linear_attn.in_proj_qkv.weight" => "qkv",
        "linear_attn.in_proj_z.weight" => "z",
        "linear_attn.out_proj.weight" => "out",
        _ => return None,
    };
    Some((l, short))
}

/// The tensor whose input Gram a body linear shares (q/k/v → q, gate/up →
/// gate, GDN qkv/z → qkv).
pub(crate) fn input_lead(short: &str) -> &'static str {
    match short {
        "q" | "k" | "v" => "q",
        "o" => "o",
        "gate" | "up" => "gate",
        "qkv" | "z" => "qkv",
        "out" => "out",
        _ => "down",
    }
}

pub(crate) fn lead_name(l: usize, lead: &str) -> String {
    match lead {
        "q" => format!("model.layers.{l}.self_attn.q_proj.weight"),
        "o" => format!("model.layers.{l}.self_attn.o_proj.weight"),
        "gate" => format!("model.layers.{l}.mlp.gate_proj.weight"),
        "qkv" => format!("model.layers.{l}.linear_attn.in_proj_qkv.weight"),
        "out" => format!("model.layers.{l}.linear_attn.out_proj.weight"),
        _ => format!("model.layers.{l}.mlp.down_proj.weight"),
    }
}

/// Every projection matrix of the layer stack, whatever codec it ends in —
/// the denominator of the all-linears bit count.
fn is_layer_projection(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name.starts_with("model.layers.")
        && name
            .strip_suffix(".weight")
            .and_then(|stem| stem.rsplit('.').next())
            .is_some_and(|last| last.contains("proj"))
}

pub(crate) fn load_f64(model: &CmfModel, idx: usize) -> anyhow::Result<Mat> {
    let e = &model.tensors[idx];
    let (r, c) = (e.shape[0], e.shape[1]);
    let mut v = vec![0f32; r * c];
    dequant_tensor(e, model.entry_bytes(e), &mut v).map_err(|m| anyhow::anyhow!("{m}"))?;
    Ok(Mat::from_vec(
        r,
        c,
        v.into_iter().map(|x| x as f64).collect(),
    ))
}

pub(crate) fn mean_gram(acc: &cortiq_engine::gptq_capture::HessianAcc) -> Mat {
    let n = acc.cols;
    let t = acc.count.max(1) as f64;
    Mat::from_vec(n, n, acc.h.iter().map(|v| v / t).collect())
}

/// `cortiq convert --quant blr`.
pub fn run_convert_blr(
    model: &str,
    output: &str,
    hf_token: Option<&str>,
    o: &BlrOpts,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        o.rho.is_finite() && o.rho >= 0.0,
        "--blr-rho {} must be finite and non-negative",
        o.rho
    );
    let gpu = o.method == Method::Nvg && o.device == BlrDevice::Gpu;
    if o.method == Method::Nvg {
        anyhow::ensure!(o.calib.is_some(), "--blr-method nvg needs --calib FILE");
        if gpu {
            // As quantize-gptq: per-op device GEMMs keep every projection's
            // input in the capture hook's sight (the resident token graph
            // would hide it), the already-encoded blr layers run on the
            // device, and the Grams accumulate there.
            // SAFETY: single-threaded here — before any pipeline/pool spawn.
            if std::env::var("CMF_GPU_WGPU_GRAPH").is_err() {
                unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", "0") };
            }
            cortiq_engine::gptq_capture::set_device(true);
        } else {
            // Calibration runs the CPU batched prefill (the capture hook lives in
            // matmat). The GPU gate is process-wide and read at backend init, so
            // it is set before anything (the f16 source conversion included).
            // SAFETY: single-threaded here — before any pipeline/pool spawn.
            unsafe { std::env::set_var("CMF_GPU", "0") };
        }
    }
    let t0 = Instant::now();
    let src = format!("{output}.src-f16.cmf");
    if !std::path::Path::new(&src).exists() {
        eprintln!("blr: float source → {src}");
        crate::convert::run_convert(model, "f16", &src, hf_token, None, None, false, |_| {})?;
    }
    let srcm = Arc::new(CmfModel::open_sharded(&src)?);
    let arch = srcm.arch().clone();
    let nl = arch.num_layers;
    let gqa = arch.num_kv_heads < arch.num_attention_heads;
    // Plan: every tensor starts at its final non-blr codec; blr candidates
    // start at q8_2f (what not-yet-encoded layers run as during calibration).
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(srcm.tensors.len());
    let mut cand: Vec<(usize, usize, &'static str)> = Vec::new(); // (slot, layer, short)
    for (slot, e) in srcm.tensors.iter().enumerate() {
        let two_d = e.shape.len() == 2;
        let float = matches!(
            e.dtype,
            TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16
        );
        let (dtype, data) = if two_d && float && !crate::convert::keeps_float(&e.name) {
            let mut v = vec![0f32; e.shape[0] * e.shape[1]];
            dequant_tensor(e, srcm.entry_bytes(e), &mut v).map_err(|m| anyhow::anyhow!("{m}"))?;
            if let Some((l, short)) = body_linear(&e.name)
                && e.shape[1] % 32 == 0
                && e.shape[0] % 16 == 0
            {
                cand.push((slot, l, short));
            }
            crate::convert::quantize_2d(crate::convert::Quant::Q8_2f, &v, e.shape[0], e.shape[1])
        } else {
            (e.dtype, srcm.entry_bytes(e).to_vec())
        };
        specs.push(TensorSpec {
            name: e.name.clone(),
            dtype,
            shape: e.shape.clone(),
            data,
        });
    }
    eprintln!(
        "blr: {} body linears over {nl} layers, method {} ({}), target {} BPW, kv×{} ({}), paths {}",
        cand.len(),
        o.method.name(),
        if o.method == Method::Nvg { o.device.name() } else { "cpu" },
        o.bpw,
        o.kv_factor,
        if gqa { "GQA" } else { "MHA" },
        o.paths.name()
    );
    let mult_of = |short: &str| {
        if gqa && (short == "k" || short == "v") {
            o.kv_factor
        } else {
            1.0
        }
    };
    let threads = o.threads.max(1);
    let mut total_bits = 0u64;
    let mut total_w = 0u64;
    let mut report: Vec<String> = Vec::new();
    let mut header = srcm.header.clone();
    header.quant_type = QuantType::Q8_2f;
    header.section_hashes = None;

    match o.method {
        Method::DualSvid => {
            let paths = if o.paths == PathsChoice::One { 1 } else { 2 };
            let per = 16usize.min(threads).max(1);
            let outer = (threads / per).max(1);
            let next = std::sync::atomic::AtomicUsize::new(0);
            let results = std::sync::Mutex::new(Vec::<(usize, Option<Encoded>)>::new());
            std::thread::scope(|s| {
                for _ in 0..outer {
                    s.spawn(|| {
                        let pool = Pool::new(per.saturating_sub(1).max(1));
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= cand.len() {
                                break;
                            }
                            let (slot, _, short) = cand[i];
                            let Ok(w) = load_f64(&srcm, slot) else {
                                results.lock().unwrap().push((slot, None));
                                continue;
                            };
                            let enc =
                                encode_dual_svid(Some(&pool), &w, o.bpw, mult_of(short), paths);
                            if let Some(e) = &enc {
                                eprintln!(
                                    "  {} {}x{} {}",
                                    srcm.tensors[slot].name, w.r, w.c, e.line
                                );
                            }
                            results.lock().unwrap().push((slot, enc));
                        }
                    });
                }
            });
            for (slot, enc) in results.into_inner().unwrap() {
                let e = &srcm.tensors[slot];
                match enc {
                    Some(enc) => {
                        total_bits += enc.bits;
                        total_w += (e.shape[0] * e.shape[1]) as u64;
                        report.push(format!("{} P{} r{} {}", e.name, enc.paths, enc.rank, enc.line));
                        specs[slot].dtype = TensorDtype::BinaryLowRank;
                        specs[slot].data = enc.bytes;
                    }
                    None => eprintln!("  {}: blr refused, kept q8_2f", e.name),
                }
            }
        }
        Method::Nvg => {
            let calib = o.calib.as_deref().expect("checked above");
            let text = std::fs::read_to_string(calib)?;
            // Two alternating work files: a fresh path per layer never truncates
            // a file some lingering mapping might still read.
            let works = [format!("{output}.work0.cmf"), format!("{output}.work1.cmf")];
            let window = o.window.max(16);
            let mut fit_w: Vec<Vec<u32>> = Vec::new();
            let mut sel_w: Vec<Vec<u32>> = Vec::new();
            for l in 0..nl {
                let mine: Vec<(usize, usize, &'static str)> =
                    cand.iter().filter(|c| c.1 == l).cloned().collect();
                if mine.is_empty() {
                    continue;
                }
                // The inputs this layer's candidates read, in a fixed order.
                let mut leads: Vec<&'static str> = Vec::new();
                for &(_, _, short) in &mine {
                    let lead = input_lead(short);
                    if !leads.contains(&lead) {
                        leads.push(lead);
                    }
                }
                let tl = Instant::now();
                let work = &works[l % 2];
                let _ = std::fs::remove_file(work);
                // The device calibration runs only layers 0..=l: a work file
                // cut there (the blr layers below are small) instead of the
                // whole model per layer.
                match truncated_header(&header, l).filter(|_| gpu) {
                    Some(th) => {
                        let cut: Vec<TensorSpec> = specs
                            .iter()
                            .filter(|t| keep_in_truncated(&t.name, l))
                            .map(|t| TensorSpec { name: t.name.clone(), dtype: t.dtype, shape: t.shape.clone(), data: t.data.clone() })
                            .collect();
                        CmfModel::write(work, &th, &cut, None, srcm.vocab.as_deref())
                    }
                    None => CmfModel::write(work, &header, &specs, None, srcm.vocab.as_deref()),
                }
                .map_err(|e| anyhow::anyhow!("write {work}: {e}"))?;
                let wm = Arc::new(CmfModel::open(work)?);
                let mut pipe = cortiq_engine::Pipeline::from_model(
                    &wm,
                    cortiq_engine::SamplerConfig::default(),
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                pipe.set_confidence(false);
                if fit_w.is_empty() {
                    let ids = pipe.tokenizer.encode(&text);
                    let need = o.fit_tokens + o.sel_tokens;
                    anyhow::ensure!(
                        ids.len() >= need,
                        "calibration text has {} tokens, need {need}",
                        ids.len()
                    );
                    for ch in ids[..o.fit_tokens].chunks(window) {
                        if ch.len() >= 2 {
                            fit_w.push(ch.to_vec());
                        }
                    }
                    for ch in ids[o.fit_tokens..need].chunks(window) {
                        if ch.len() >= 2 {
                            sel_w.push(ch.to_vec());
                        }
                    }
                    eprintln!(
                        "blr: calibration FIT {} windows, SEL {} windows of ≤{window} tokens",
                        fit_w.len(),
                        sel_w.len()
                    );
                }
                let names: HashSet<String> = leads.iter().map(|s| lead_name(l, s)).collect();
                let mut grams: HashMap<&str, (Mat, Mat)> = HashMap::new();
                let mut fit_ppl = 0f64;
                let keep = names.clone();
                cortiq_engine::gptq_capture::begin_filtered(true, move |n| keep.contains(n));
                for wv in &fit_w {
                    if gpu {
                        pipe.calib_forward(wv, l + 1, window).map_err(|e| anyhow::anyhow!(e))?;
                    } else {
                        fit_ppl += pipe.ppl_ids(wv).map_err(|e| anyhow::anyhow!(e))?.ln();
                    }
                }
                let hf = cortiq_engine::gptq_capture::end();
                let keep = names.clone();
                cortiq_engine::gptq_capture::begin_filtered(true, move |n| keep.contains(n));
                for wv in &sel_w {
                    if gpu {
                        pipe.calib_forward(wv, l + 1, window).map_err(|e| anyhow::anyhow!(e))?;
                    } else {
                        pipe.ppl_ids(wv).map_err(|e| anyhow::anyhow!(e))?;
                    }
                }
                let hs = cortiq_engine::gptq_capture::end();
                drop(pipe);
                drop(wm);
                for &lead in &leads {
                    let nm = lead_name(l, lead);
                    let (Some(a), Some(b)) = (hf.get(&nm), hs.get(&nm)) else {
                        anyhow::bail!(
                            "layer {l}: no calibration capture for {nm} (captured: {:?})",
                            hf.keys().collect::<Vec<_>>()
                        );
                    };
                    grams.insert(lead, (mean_gram(a), mean_gram(b)));
                }
                let tc = tl.elapsed().as_secs_f64();
                #[cfg(feature = "gpu")]
                if gpu {
                    // One tensor at a time on the device, grouped by input so
                    // each input's context is built once and freed after.
                    let pool = Pool::new(threads.saturating_sub(1).max(1));
                    let mut results: Vec<(usize, Option<Encoded>)> = Vec::new();
                    for &lead in &leads {
                        let (hfit, hsel) = &grams[lead];
                        let th = Instant::now();
                        let hc = crate::nvg_gpu::HCtxD::new(hfit, hsel, o.rho)?;
                        let thc = th.elapsed().as_secs_f64();
                        for &(slot, _, short) in mine.iter().filter(|m| input_lead(m.2) == lead) {
                            let w = load_f64(&srcm, slot)?;
                            let te = Instant::now();
                            let prof = Prof::default();
                            let want_prof = std::env::var("CMF_NVG_PROF").as_deref() == Ok("1");
                            let enc = crate::nvg_gpu::encode_nvg_dev(
                                Some(&pool),
                                &w,
                                &hc,
                                o.bpw,
                                mult_of(short),
                                o.paths,
                                want_prof.then_some(&prof),
                            )?;
                            if let Some(e) = &enc {
                                eprintln!(
                                    "  L{l} {short:4} {}x{} {} ({:.1}s)",
                                    w.r,
                                    w.c,
                                    e.line,
                                    te.elapsed().as_secs_f64()
                                );
                            }
                            if want_prof {
                                eprintln!("    profile: hctx {thc:.2}s | {}", prof.line());
                            }
                            results.push((slot, enc));
                        }
                    }
                    for (slot, enc) in results {
                        let e = &srcm.tensors[slot];
                        match enc {
                            Some(enc) => {
                                total_bits += enc.bits;
                                total_w += (e.shape[0] * e.shape[1]) as u64;
                                report.push(format!("{} P{} r{} {}", e.name, enc.paths, enc.rank, enc.line));
                                specs[slot].dtype = TensorDtype::BinaryLowRank;
                                specs[slot].data = enc.bytes;
                            }
                            None => eprintln!("  {}: blr refused, kept q8_2f", e.name),
                        }
                    }
                    eprintln!(
                        "blr: layer {l}/{nl} done — calib {tc:.1}s, layer {:.1}s, total {:.0}s",
                        tl.elapsed().as_secs_f64(),
                        t0.elapsed().as_secs_f64()
                    );
                    continue;
                }
                // per-input contexts (parallel over the inputs)
                let nlead = leads.len().max(1);
                let hctx: HashMap<&str, HCtx> = std::thread::scope(|s| {
                    let hs: Vec<_> = leads
                        .iter()
                        .map(|&lead| {
                            let g = &grams[lead].0;
                            let per = (threads / nlead).max(1);
                            s.spawn(move || {
                                let pool = Pool::new(per.saturating_sub(1).max(1));
                                (lead, HCtx::new(Some(&pool), g, o.rho))
                            })
                        })
                        .collect();
                    hs.into_iter()
                        .map(|h| {
                            let (lead, c) = h.join().unwrap();
                            (lead, c)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .map(|(lead, c)| c.map(|c| (lead, c)))
                .collect::<anyhow::Result<_>>()?;
                let per = (threads / mine.len().max(1)).max(2);
                let results: Vec<(usize, Option<Encoded>)> = std::thread::scope(|s| {
                    let hs: Vec<_> = mine
                        .iter()
                        .map(|&(slot, _, short)| {
                            let (srcm, grams, hctx) = (&srcm, &grams, &hctx);
                            s.spawn(move || {
                                let pool = Pool::new(per - 1);
                                let w = load_f64(srcm, slot).expect("weight");
                                let lead = input_lead(short);
                                let (hfit, hsel) = &grams[lead];
                                let te = Instant::now();
                                let enc = encode_nvg(
                                    Some(&pool),
                                    &w,
                                    &hctx[lead],
                                    hfit,
                                    hsel,
                                    o.bpw,
                                    mult_of(short),
                                    o.paths,
                                    None,
                                );
                                if let Some(e) = &enc {
                                    eprintln!(
                                        "  L{l} {short:4} {}x{} {} ({:.0}s)",
                                        w.r,
                                        w.c,
                                        e.line,
                                        te.elapsed().as_secs_f64()
                                    );
                                }
                                (slot, enc)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for (slot, enc) in results {
                    let e = &srcm.tensors[slot];
                    match enc {
                        Some(enc) => {
                            total_bits += enc.bits;
                            total_w += (e.shape[0] * e.shape[1]) as u64;
                            report.push(format!(
                                "{} P{} r{} {}",
                                e.name, enc.paths, enc.rank, enc.line
                            ));
                            specs[slot].dtype = TensorDtype::BinaryLowRank;
                            specs[slot].data = enc.bytes;
                        }
                        None => eprintln!("  {}: blr refused, kept q8_2f", e.name),
                    }
                }
                eprintln!(
                    "blr: layer {l}/{nl} done — calib {tc:.0}s (fit ppl {:.2}), layer {:.0}s, total {:.0}s",
                    (fit_ppl / fit_w.len().max(1) as f64).exp(),
                    tl.elapsed().as_secs_f64(),
                    t0.elapsed().as_secs_f64()
                );
            }
            for w in &works {
                let _ = std::fs::remove_file(w);
            }
        }
    }
    // Strict accounting: the blr tensors alone (the container formula,
    // header excluded), and every projection of the layer stack whatever its
    // codec (blr bits + 8 · payload bytes of the rest) — the honest body
    // number when some projections stayed f16/q8_2f.
    let body_bpw = total_bits as f64 / total_w.max(1) as f64;
    let (mut all_bits, mut all_w) = (0u64, 0u64);
    for s in &specs {
        if !is_layer_projection(&s.name, &s.shape) {
            continue;
        }
        let n = (s.shape[0] * s.shape[1]) as u64;
        all_w += n;
        all_bits += if s.dtype == TensorDtype::BinaryLowRank {
            let h = cortiq_core::blr::blr_parse_header(&s.data)
                .map_err(|e| anyhow::anyhow!("{}: {e}", s.name))?;
            blr_bits(h.d_out, h.d_in, h.rank, h.paths)
        } else {
            8 * s.data.len() as u64
        };
    }
    let all_bpw = all_bits as f64 / all_w.max(1) as f64;
    let mut prov = header
        .provenance
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));
    prov["weight_quant"] = serde_json::json!("blr");
    prov["blr"] = serde_json::json!({
        "method": o.method.name(),
        "target_bpw": o.bpw,
        "blr_body_bpw": body_bpw,
        "blr_body_weights": total_w,
        "layer_projection_bpw": all_bpw,
        "layer_projection_weights": all_w,
        "kv_factor": o.kv_factor,
        "paths": o.paths.name(),
        "calibration": o.calib.as_ref().filter(|_| o.method == Method::Nvg).map(|c| serde_json::json!({
            "file": std::path::Path::new(c).file_name().map(|s| s.to_string_lossy().into_owned()),
            "fit_tokens": o.fit_tokens, "sel_tokens": o.sel_tokens, "window": o.window, "rho": o.rho,
            "sequential": true,
        })),
        "encoder_device": if o.method == Method::Nvg { o.device.name() } else { "cpu" },
        "embeddings": "q8_2f",
    });
    header.provenance = Some(prov);
    CmfModel::write(output, &header, &specs, None, srcm.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    let rep = format!("{output}.blr.txt");
    let _ = std::fs::write(&rep, report.join("\n") + "\n");
    eprintln!(
        "blr: wrote {output} — blr body {:.4} BPW over {} weights ({:.1} MB of blr bits); \
         all layer projections {:.4} BPW over {} weights; {:.0}s",
        body_bpw,
        total_w,
        total_bits as f64 / 8e6,
        all_bpw,
        all_w,
        t0.elapsed().as_secs_f64()
    );
    if !o.keep_src {
        let _ = std::fs::remove_file(&src);
    }
    Ok(())
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

    /// Synthetic layer: decaying spectrum, three heavy input channels.
    fn synth() -> (Mat, Mat) {
        let (m, n, t) = (96usize, 64usize, 1024usize);
        let mut s = 1u64;
        let mut w = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                w.d[i * n + j] = rnd(&mut s) * (1.0 + j as f64).powf(-0.4);
            }
        }
        let mut sc: Vec<f64> = (0..n).map(|_| (rnd(&mut s) * 1.6).exp()).collect();
        sc[0] *= 20.0;
        sc[1] *= 20.0;
        sc[2] *= 20.0;
        let mut x = Mat::zeros(n, t);
        for i in 0..n {
            for k in 0..t {
                x.d[i * t + k] = sc[i] * rnd(&mut s);
            }
        }
        let mut h = gemm(None, &x, false, &x, true);
        h.d.iter_mut().for_each(|v| *v /= t as f64);
        (w, h)
    }

    #[test]
    fn als_monotone_and_refine_monotone() {
        let pool = Pool::new(2);
        let p = Some(&pool);
        let (w, h) = synth();
        let hc = HCtx::new(p, &h, 1e-3).unwrap();
        let c = TCtx {
            w: &w,
            wh: gemm(p, &w, false, &hc.hf, false),
            hc: &hc,
            pool: p,
            prof: None,
        };
        let r = 16;
        let e = |md: &BlrModel| herr(p, &w.sub(&md.recon(p, None)), &hc.hf);
        let mut md = build_dual_svid(p, &w, r, 2);
        let mut prev = e(&md);
        for _ in 0..5 {
            als(&mut md, &c, 1);
            let now = e(&md);
            assert!(now <= prev * (1.0 + 1e-9), "{prev} -> {now}");
            prev = now;
        }
        let init = build_nvg_init(&c, r, 2);
        let e_init = e(&init);
        let mut q = init.clone();
        flip_u(&mut q, &c, 0);
        assert!(e(&q) <= e_init * (1.0 + 1e-9));
        let refined = refine(init, &c, 2);
        assert!(e(&refined) <= e_init * (1.0 + 1e-12));
        // data-aware beats data-free on the H metric
        assert!(e(&refined) < e(&build_dual_svid(p, &w, r, 2)));
        // two paths beat one at the same rank (at init)
        let one = build_dual_svid(p, &w, r, 1);
        let two = build_dual_svid(p, &w, r, 2);
        assert!(w.sub(&two.recon(p, None)).frob2() < w.sub(&one.recon(p, None)).frob2());
        // packing round-trips through the core dequant within f16 rounding
        let (bytes, st) = pack(&refined).unwrap();
        let mut deq = vec![0f32; w.r * w.c];
        cortiq_core::blr::dequant_blr(&bytes, w.r, w.c, &mut deq).unwrap();
        let rec = st.recon(p, None);
        for i in 0..deq.len() {
            assert!((deq[i] as f64 - rec.d[i]).abs() < 1e-4 * (1.0 + rec.d[i].abs()));
        }
    }

    #[test]
    fn gptq_v_uc_is_upper_cholesky_of_permuted_inverse() {
        let (_, h) = synth();
        let hc = HCtx::new(None, &h, 1e-2).unwrap();
        let n = h.r;
        // Ucᵀ Uc · Hp = I
        let mut hp = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                hp.d[i * n + j] = hc.hf.at(hc.perm[i], hc.perm[j]);
            }
        }
        let utu = gemm(None, &hc.uc, true, &hc.uc, false);
        let id = gemm(None, &utu, false, &hp, false);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id.at(i, j) - want).abs() < 1e-6, "{i},{j} {}", id.at(i, j));
            }
            for j in 0..i {
                assert_eq!(hc.uc.at(i, j), 0.0);
            }
        }
    }

    /// Per-tensor drivers at a real budget: both methods emit a payload that
    /// validates against the shape, at the rank the rule gives, with the
    /// strict bit count; NVG beats dual-SVID in the calibration metric.
    #[test]
    fn drivers_emit_valid_payloads_at_the_rank_rule() {
        let pool = Pool::new(2);
        let p = Some(&pool);
        let (w, h) = synth();
        let b = 3.0;
        let d = encode_dual_svid(p, &w, b, 1.0, 2).unwrap();
        assert_eq!(d.rank, blr_rank(96, 64, b, 2, 1.0));
        assert_eq!(d.bits, blr_bits(96, 64, d.rank, 2));
        assert!(cortiq_core::blr::validate_blr(&[96, 64], &d.bytes).is_ok());
        let hc = HCtx::new(p, &h, 0.03).unwrap();
        let n = encode_nvg(p, &w, &hc, &h, &h, b, 1.0, PathsChoice::Auto, None).unwrap();
        assert_eq!(n.rank, blr_rank(96, 64, b, n.paths, 1.0));
        assert_eq!(n.bits, blr_bits(96, 64, n.rank, n.paths));
        assert!(cortiq_core::blr::validate_blr(&[96, 64], &n.bytes).is_ok());
        let stored = |bytes: &[u8]| {
            let mut v = vec![0f32; 96 * 64];
            cortiq_core::blr::dequant_blr(bytes, 96, 64, &mut v).unwrap();
            Mat::from_vec(96, 64, v.into_iter().map(|x| x as f64).collect())
        };
        let (ed, en) = (
            rel(p, &w, &stored(&d.bytes), &h),
            rel(p, &w, &stored(&n.bytes), &h),
        );
        assert!(en < ed, "nvg {en} vs dual-svid {ed}");
    }

    #[test]
    fn body_linears_cover_attention_ffn_and_gdn() {
        assert_eq!(body_linear("model.layers.3.self_attn.k_proj.weight"), Some((3, "k")));
        assert_eq!(body_linear("model.layers.0.mlp.down_proj.weight"), Some((0, "down")));
        assert_eq!(
            body_linear("model.layers.2.linear_attn.in_proj_qkv.weight"),
            Some((2, "qkv"))
        );
        assert_eq!(body_linear("model.layers.2.linear_attn.in_proj_z.weight"), Some((2, "z")));
        assert_eq!(body_linear("model.layers.2.linear_attn.out_proj.weight"), Some((2, "out")));
        assert_eq!(body_linear("model.layers.2.linear_attn.in_proj_a.weight"), None);
        assert_eq!(body_linear("model.embed_tokens.weight"), None);
        assert_eq!(body_linear("model.layers.1.mlp.experts.3.up_proj.weight"), None);
        assert_eq!(input_lead("z"), "qkv");
        assert_eq!(input_lead("v"), "q");
        assert_eq!(input_lead("up"), "gate");
        for short in ["q", "k", "v", "o", "gate", "up", "down", "qkv", "z", "out"] {
            let lead = lead_name(5, input_lead(short));
            assert_eq!(body_linear(&lead).map(|(l, s)| (l, input_lead(s))), Some((5, input_lead(short))));
        }
        assert!(is_layer_projection("model.layers.0.linear_attn.in_proj_a.weight", &[16, 1024]));
        assert!(is_layer_projection(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            &[6144, 1024]
        ));
        assert!(is_layer_projection("model.layers.3.self_attn.q_proj.weight", &[4096, 1024]));
        assert!(!is_layer_projection("model.layers.3.self_attn.q_norm.weight", &[256, 1]));
        // every encoder candidate is counted in the all-projections total
        for short in ["q", "k", "v", "o", "gate", "up", "down", "qkv", "z", "out"] {
            let name = lead_name(2, input_lead(short));
            assert!(is_layer_projection(&name, &[64, 64]), "{name}");
        }
        assert!(!is_layer_projection("model.layers.0.linear_attn.conv1d.weight", &[6144, 4]));
        assert!(!is_layer_projection("lm_head.weight", &[10, 4]));
    }
}
