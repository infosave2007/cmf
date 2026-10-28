//! S1 (illumination) and S3 (fold) numerics.
//!
//! The illumination is the GPTQ capture hook pointed at one tensor per
//! layer: the input of `down_proj`, `a = act(gate)·up`. Its Gram
//! `Σ = Σ_t a_t a_tᵀ` gives both the energy of every neuron (the diagonal)
//! and everything the closed-form fold needs.
//!
//! For a keep-set S and the dropped rest P, the dropped activations are
//! regressed on the kept ones, `Π = (Σ_SS + εI)⁻¹ Σ_SP`, and folded into the
//! read-out: `W_down' = W_down[:, S] + W_down[:, P]·Πᵀ`. With the neurons in
//! importance order and `Σ = L·Lᵀ`, the output error left after that refit
//! for the prefix of size k is `‖(W·L)[:, k..]‖²` — one Cholesky and one GEMM
//! per layer give the cost of every width at once, which is what the planner
//! allocates by.

use super::data::Rec;
use anyhow::{Result, anyhow, ensure};
use cortiq_engine::Pipeline;
use cortiq_engine::gptq_capture::{self, HessianAcc};
use std::collections::BTreeMap;

pub fn down_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.down_proj.weight")
}

/// Calibration windows of whole records: a record longer than `window` is
/// cut into windows with a half-window stride, plus a window flush with its
/// end whenever the stride left a tail uncovered (the notebook dropped tails
/// under 128 tokens — here that would lose answer tokens). Records never
/// share a window. `budget` caps the tokens taken.
pub fn windows<'a>(recs: impl Iterator<Item = &'a Rec>, window: usize, budget: Option<usize>) -> (Vec<Vec<u32>>, usize) {
    let window = window.max(16);
    let stride = (window / 2).max(1);
    let (mut out, mut n) = (Vec::new(), 0usize);
    'outer: for r in recs {
        let ids = &r.ids;
        let mut parts: Vec<&[u32]> = Vec::new();
        if ids.len() <= window {
            parts.push(ids);
        } else {
            let mut st = 0;
            while st + window <= ids.len() {
                parts.push(&ids[st..st + window]);
                st += stride;
            }
            let covered = st - stride + window;
            if covered < ids.len() {
                parts.push(&ids[ids.len() - window..]);
            }
        }
        for p in parts {
            if p.len() < 2 {
                continue;
            }
            if budget.is_some_and(|b| n + p.len() > b) {
                break 'outer;
            }
            n += p.len();
            out.push(p.to_vec());
        }
    }
    (out, n)
}

/// One forward over `windows` with the down_proj inputs of `layers`
/// captured; returns the full symmetric Grams by layer.
pub fn capture(pipe: &mut Pipeline, windows: &[Vec<u32>], layers: &[usize], label: &str) -> Result<BTreeMap<usize, HessianAcc>> {
    ensure!(!layers.is_empty(), "capture: no layers");
    let names: std::collections::HashSet<String> = layers.iter().map(|&l| down_name(l)).collect();
    let upto = layers.iter().max().unwrap() + 1;
    // Grams on the device where a wgpu adapter is up (as GPTQ calibrates):
    // each batch is one XᵀX dispatch instead of a host fold. macOS keeps the
    // host (its calibration forward runs on the CPU); CMF_ILLUM_DEVICE_GRAM=0
    // forces the host fold.
    let device_gram = !cfg!(target_os = "macos")
        && std::env::var("CMF_GPU").as_deref() != Ok("0")
        && std::env::var("CMF_ILLUM_DEVICE_GRAM").as_deref() != Ok("0")
        && cortiq_engine::gpu::backend_available();
    gptq_capture::set_device(device_gram);
    gptq_capture::begin_filtered(true, move |n| names.contains(n));
    let t0 = std::time::Instant::now();
    let total: usize = windows.iter().map(Vec::len).sum();
    let mut done = 0usize;
    for (k, w) in windows.iter().enumerate() {
        // one prefill step per window: on a device each step is a round
        // trip per projection, so fewer, longer steps keep it busy
        let step: usize = std::env::var("CMF_ILLUM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(w.len().max(1));
        if let Err(e) = pipe.calib_forward(w, upto, step) {
            let _ = gptq_capture::end();
            anyhow::bail!("{label}: window {k}: {e}");
        }
        done += w.len();
        if k % 16 == 15 || k + 1 == windows.len() {
            let el = t0.elapsed().as_secs_f64();
            eprint!(
                "\r  {label}: {done}/{total} tokens · {:.0} tok/s · ~{} left   ",
                done as f64 / el.max(1e-9),
                crate::fmt_eta((total - done) as f64 * el / done.max(1) as f64)
            );
        }
    }
    eprintln!();
    cortiq_engine::cpuprof::report(label, total);
    let mut hess = gptq_capture::end();
    let mut out = BTreeMap::new();
    for &l in layers {
        let acc = hess
            .remove(&down_name(l))
            .ok_or_else(|| anyhow!("{label}: layer {l}: down_proj input was never captured (the forward bypassed the hook)"))?;
        ensure!(
            acc.count > 0 && acc.sumsq.iter().any(|&v| v > 0.0),
            "{label}: layer {l}: captured energy is all zero — a fused device path hid the FFN from the hook"
        );
        out.insert(l, acc);
    }
    Ok(out)
}

/// Neuron importance `‖W_down[:, j]‖² · ⟨a_j²⟩` (energy times read-out
/// weight: what the neuron contributes to the layer output).
pub fn importance(wdown: &[f32], h: usize, n: usize, sumsq: &[f64], count: usize) -> Vec<f64> {
    let mut col = vec![0f64; n];
    for r in 0..h {
        let row = &wdown[r * n..(r + 1) * n];
        for (c, &v) in col.iter_mut().zip(row) {
            *c += (v as f64) * (v as f64);
        }
    }
    let inv = 1.0 / count.max(1) as f64;
    col.iter().zip(sumsq).map(|(&c, &s)| c * s * inv).collect()
}

/// Indices by descending importance (ties: lower index first).
pub fn order(imp: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..imp.len()).collect();
    o.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    o
}

/// `C[m×n] = A[m×k]·B[k×n]` (row-major), rows split over `threads`.
pub fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let per = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(per * n).enumerate() {
            let rows = chunk.len() / n;
            let src = &a[ci * per * k..(ci * per + rows) * k];
            s.spawn(move || cortiq_engine::qtensor::sgemm_public(rows, n, k, 1.0, src, k, b, n, false, chunk, n));
        }
    });
    c
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// `Σ[rows, cols]` sub-block as f32 (row-major).
fn block(sig: &[f64], n: usize, rows: &[usize], cols: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        let row = &sig[r * n..(r + 1) * n];
        out.extend(cols.iter().map(|&c| row[c] as f32));
    }
    out
}

/// Columns `cols` of the row-major `w[h×n]`.
fn columns(w: &[f32], h: usize, n: usize, cols: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(h * cols.len());
    for r in 0..h {
        let row = &w[r * n..(r + 1) * n];
        out.extend(cols.iter().map(|&c| row[c]));
    }
    out
}

/// Relative output error left after the refit for every keep size:
/// `c[k] = ‖(W_perm·L)[:, k..]‖² / ‖W_perm·L‖²`, `c[0] = 1`, `c[n] = 0`;
/// one extra trailing entry holds the layer's output energy `‖W_perm·L‖²`.
pub fn refit_curve(sig: &[f64], n: usize, wdown: &[f32], h: usize, ord: &[usize], threads: usize) -> Result<Vec<f64>> {
    let mean_diag = (0..n).map(|i| sig[i * n + i]).sum::<f64>() / n as f64;
    // A Gram accumulated in f32 (on a device) is indefinite at the f32
    // rounding level in its weakest directions: the jitter steps up until
    // the factor exists. It only conditions the plan's curve.
    let mut p = Vec::new();
    let mut ok = false;
    for rel in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let jitter = rel * mean_diag.max(1e-30);
        p = vec![0f64; n * n];
        for (i, &oi) in ord.iter().enumerate() {
            for (j, &oj) in ord.iter().enumerate().take(i + 1) {
                p[i * n + j] = sig[oi * n + oj];
                p[j * n + i] = sig[oi * n + oj];
            }
            p[i * n + i] += jitter;
        }
        if crate::gptq::cholesky_lower_par(&mut p, n, threads) {
            if rel > 1e-6 {
                eprintln!("  refit curve: jitter {rel:e} of the mean diagonal (f32-accumulated Gram)");
            }
            ok = true;
            break;
        }
    }
    ensure!(ok, "Gram is not positive definite even with 1e-2 jitter");
    let l32: Vec<f32> = p.iter().map(|&v| v as f32).collect();
    drop(p);
    let wp = columns(wdown, h, n, ord);
    let m = gemm(&wp, &l32, h, n, n, threads);
    let mut colsq = vec![0f64; n];
    for r in 0..h {
        for (c, &v) in colsq.iter_mut().zip(&m[r * n..(r + 1) * n]) {
            *c += v as f64 * v as f64;
        }
    }
    let mut c = vec![0f64; n + 1];
    for k in (0..n).rev() {
        c[k] = c[k + 1] + colsq[k];
    }
    let tot = c[0].max(1e-300);
    let mut out: Vec<f64> = c.iter().map(|v| v / tot).collect();
    // the layer's own output energy (Σ over tokens), kept past the curve
    out.push(tot);
    Ok(out)
}

/// Lower Cholesky of a k×k f64 matrix (in place; false = not positive).
fn chol(a: &mut [f64], k: usize, threads: usize) -> bool {
    crate::gptq::cholesky_lower_par(a, k, threads)
}

/// Solve `(L·Lᵀ) X = B` for `B[k×p]` (row-major), columns split over threads.
fn chol_solve(l: &[f64], k: usize, b: &mut [f64], p: usize, threads: usize) {
    let per = p.div_ceil(threads.max(1)).max(1);
    let cols: Vec<(usize, usize)> = (0..p).step_by(per).map(|c0| (c0, (c0 + per).min(p))).collect();
    let bp = SyncPtr(b.as_mut_ptr());
    std::thread::scope(|s| {
        for &(c0, c1) in &cols {
            let bp = &bp;
            s.spawn(move || {
                // Each worker owns columns c0..c1 of every row; rows are read
                // and written as contiguous slices of that column block.
                let row = |i: usize| unsafe { std::slice::from_raw_parts_mut(bp.0.add(i * p + c0), c1 - c0) };
                // forward: L y = b
                for i in 0..k {
                    let bi = row(i);
                    for j in 0..i {
                        let lij = l[i * k + j];
                        if lij != 0.0 {
                            for (x, &y) in bi.iter_mut().zip(row(j).iter()) {
                                *x -= lij * y;
                            }
                        }
                    }
                    let inv = 1.0 / l[i * k + i];
                    bi.iter_mut().for_each(|x| *x *= inv);
                }
                // backward: Lᵀ x = y
                for i in (0..k).rev() {
                    let bi = row(i);
                    for j in i + 1..k {
                        let lji = l[j * k + i];
                        if lji != 0.0 {
                            for (x, &y) in bi.iter_mut().zip(row(j).iter()) {
                                *x -= lji * y;
                            }
                        }
                    }
                    let inv = 1.0 / l[i * k + i];
                    bi.iter_mut().for_each(|x| *x *= inv);
                }
            });
        }
    });
}

struct SyncPtr(*mut f64);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

/// What the fold decided for one layer.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FoldInfo {
    pub layer: usize,
    pub width: usize,
    pub keep: usize,
    /// Relative ridge of the chosen refit (None = plain cut).
    pub ridge: Option<f64>,
    /// Relative val output error without / with the chosen refit.
    pub val_err_cut: f64,
    pub val_err: f64,
    /// Relative train output error of the chosen read-out.
    pub train_err: f64,
}

/// The narrowed tensors of one layer.
pub struct Folded {
    pub keep: Vec<usize>,
    pub down: Vec<f32>,
    pub info: FoldInfo,
}

/// Output error `tr(W_P (Σ_PP − Πᵀ Σ_SP − Σ_PS Π + Πᵀ Σ_SS Π) W_Pᵀ)` of the
/// read-out `A = W_P·Πᵀ` on the Gram `sig`, with D = W_P.
fn out_err(sig: &[f64], n: usize, s: &[usize], p: &[usize], d: &[f32], a: Option<&[f32]>, h: usize, threads: usize) -> f64 {
    let spp = block(sig, n, p, p);
    let x1 = gemm(d, &spp, h, p.len(), p.len(), threads);
    let e_pp = dot(&x1, d);
    let Some(a) = a else { return e_pp };
    let ssp = block(sig, n, s, p);
    let x2 = gemm(a, &ssp, h, s.len(), p.len(), threads);
    let e_sp = dot(&x2, d);
    let sss = block(sig, n, s, s);
    let x3 = gemm(a, &sss, h, s.len(), s.len(), threads);
    let e_ss = dot(&x3, a);
    e_pp - 2.0 * e_sp + e_ss
}

/// Fold one layer to the first `k` neurons of `ord`.
/// `ridges` are the relative ridges tried; `allow_cut` lets a plain cut win
/// when the val Gram says the refit does not generalize (`--refit auto`).
#[allow(clippy::too_many_arguments)]
pub fn fold_layer(
    layer: usize,
    train: &[f64],
    val: Option<&[f64]>,
    n: usize,
    wdown: &[f32],
    h: usize,
    ord: &[usize],
    k: usize,
    ridges: &[f64],
    allow_cut: bool,
    threads: usize,
) -> Result<Folded> {
    let mut keep: Vec<usize> = ord[..k].to_vec();
    keep.sort_unstable();
    let mut drop_: Vec<usize> = ord[k..].to_vec();
    drop_.sort_unstable();
    let ws = columns(wdown, h, n, &keep);
    if drop_.is_empty() {
        let info = FoldInfo { layer, width: n, keep: k, ridge: None, val_err_cut: 0.0, val_err: 0.0, train_err: 0.0 };
        return Ok(Folded { keep, down: ws, info });
    }
    let d = columns(wdown, h, n, &drop_);
    let (kk, pp) = (keep.len(), drop_.len());
    let full_energy = |sig: &[f64]| -> f64 {
        let s = block(sig, n, &(0..n).collect::<Vec<_>>(), &(0..n).collect::<Vec<_>>());
        let x = gemm(wdown, &s, h, n, n, threads);
        dot(&x, wdown).max(1e-300)
    };
    let e_tr_tot = full_energy(train);
    let e_va_tot = val.map(full_energy);
    let scoring = val.unwrap_or(train);
    let scoring_tot = e_va_tot.unwrap_or(e_tr_tot);

    // Σ_SS and Σ_SP from the train Gram, f64.
    let mut sss = vec![0f64; kk * kk];
    for (i, &a) in keep.iter().enumerate() {
        for (j, &b) in keep.iter().enumerate() {
            sss[i * kk + j] = train[a * n + b];
        }
    }
    let mut ssp = vec![0f64; kk * pp];
    for (i, &a) in keep.iter().enumerate() {
        for (j, &b) in drop_.iter().enumerate() {
            ssp[i * pp + j] = train[a * n + b];
        }
    }
    let mean_diag = (0..kk).map(|i| sss[i * kk + i]).sum::<f64>() / kk as f64;

    let cut_err = out_err(scoring, n, &keep, &drop_, &d, None, h, threads) / scoring_tot;
    let mut best: (f64, Option<f64>, Option<Vec<f32>>) = (if allow_cut { cut_err } else { f64::INFINITY }, None, None);
    for &rho in ridges {
        let mut l = sss.clone();
        for i in 0..kk {
            l[i * kk + i] += rho * mean_diag.max(1e-30);
        }
        if !chol(&mut l, kk, threads) {
            continue;
        }
        let mut pi = ssp.clone(); // becomes Π [k×p]
        chol_solve(&l, kk, &mut pi, pp, threads);
        let pit: Vec<f32> = {
            // Πᵀ [p×k]
            let mut t = vec![0f32; pp * kk];
            for i in 0..kk {
                for j in 0..pp {
                    t[j * kk + i] = pi[i * pp + j] as f32;
                }
            }
            t
        };
        let a = gemm(&d, &pit, h, pp, kk, threads); // W_P·Πᵀ [h×k]
        let e = out_err(scoring, n, &keep, &drop_, &d, Some(&a), h, threads) / scoring_tot;
        if e < best.0 {
            best = (e, Some(rho), Some(a));
        }
    }
    let (val_err, ridge, a) = best;
    ensure!(val_err.is_finite(), "layer {layer}: no refit was positive definite and the plain cut is disallowed");
    let down: Vec<f32> = match &a {
        Some(a) => ws.iter().zip(a).map(|(&x, &y)| x + y).collect(),
        None => ws,
    };
    let train_err = out_err(train, n, &keep, &drop_, &d, a.as_deref(), h, threads) / e_tr_tot;
    let info = FoldInfo {
        layer,
        width: n,
        keep: k,
        ridge,
        val_err_cut: cut_err,
        val_err,
        train_err,
    };
    Ok(Folded { keep, down, info })
}

/// Rows `keep` of the row-major `w[rows×cols]`.
pub fn rows(w: &[f32], cols: usize, keep: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(keep.len() * cols);
    for &r in keep {
        out.extend_from_slice(&w[r * cols..(r + 1) * cols]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    /// Correlated activations so the refit has something to recover.
    fn fixture(n: usize, h: usize, t: usize) -> (Vec<f64>, Vec<f32>) {
        let mut s = 7u64;
        let base: Vec<f32> = (0..n * 4).map(|_| lcg(&mut s)).collect();
        let mut sig = vec![0f64; n * n];
        for _ in 0..t {
            let z: Vec<f32> = (0..4).map(|_| lcg(&mut s)).collect();
            let a: Vec<f64> = (0..n)
                .map(|j| (0..4).map(|q| base[j * 4 + q] * z[q]).sum::<f32>() as f64 + 0.05 * lcg(&mut s) as f64)
                .collect();
            for i in 0..n {
                for j in 0..n {
                    sig[i * n + j] += a[i] * a[j];
                }
            }
        }
        let w: Vec<f32> = (0..h * n).map(|_| lcg(&mut s)).collect();
        (sig, w)
    }

    #[test]
    fn curve_matches_explicit_refit_error() {
        let (n, h) = (24, 6);
        let (sig, w) = fixture(n, h, 400);
        let imp: Vec<f64> = (0..n).map(|j| sig[j * n + j]).collect();
        let ord = order(&imp);
        let c = refit_curve(&sig, n, &w, h, &ord, 2).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-9 && c[n].abs() < 1e-12 && c[n + 1] > 0.0);
        for k in [6, 12, 18] {
            // the same Gram for train and scoring, no ridge → the curve value
            let f = fold_layer(0, &sig, Some(&sig), n, &w, h, &ord, k, &[1e-9], false, 2).unwrap();
            assert!(
                (f.info.val_err - c[k]).abs() < 2e-3 * c[k].max(1e-3),
                "k={k}: fold {} vs curve {}",
                f.info.val_err,
                c[k]
            );
            assert!(f.info.val_err < f.info.val_err_cut, "refit must beat the plain cut on its own Gram");
        }
    }

    #[test]
    fn full_keep_is_identity() {
        let (n, h) = (16, 4);
        let (sig, w) = fixture(n, h, 100);
        let ord: Vec<usize> = (0..n).rev().collect();
        let f = fold_layer(3, &sig, None, n, &w, h, &ord, n, &[1e-3], true, 1).unwrap();
        assert_eq!(f.keep, (0..n).collect::<Vec<_>>());
        assert_eq!(f.down, w);
    }

    #[test]
    fn windows_never_cross_records_and_cover_the_tail() {
        let mk = |len: usize| Rec {
            rid: format!("r{len}"),
            split: crate::tdata::TdataSplit::Train,
            category: String::new(),
            ids: (0..len as u32).collect(),
            prompt: 1,
            think: 0,
            answer: len - 1,
        };
        let recs = [mk(10), mk(1000)];
        let (w, n) = windows(recs.iter(), 256, None);
        assert_eq!(w[0].len(), 10);
        assert!(w.iter().skip(1).all(|x| x.len() == 256));
        assert_eq!(*w.last().unwrap().last().unwrap(), 999);
        assert_eq!(n, w.iter().map(Vec::len).sum::<usize>());
    }
}
