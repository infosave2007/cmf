//! `cortiq hy4-alias`: resonance substitutes for the routed experts a
//! Hy4-flash cut drops. The router keeps choosing among all 512 experts with
//! its own weights; a pick of a dropped expert `d` is served by the kept
//! expert `k` that reconstructs `d`'s output best on `d`'s own inputs
//! (argmin of the reconstruction error, the resonance rule). Inputs come
//! from full-model captures (`hy4-sfq --capture-only` without
//! `CMF_HY4_CAP_NOX`: `L<li>.x` latent rows, `.picks`, `.w`).
//!
//! Per dropped expert: its `rows` heaviest captured inputs (by mixing
//! weight), its output on them, the `cands` kept experts the full model
//! co-picks with it most (its neighbours in the router's space; the router
//! reads the hidden state, the capture holds the latent rows), and among
//! them the one with the least weight-summed squared error. Written as
//! `{"<layer>": [512 ids]}` (`CMF_HY4_ALIAS`; a kept expert maps to itself).
//! The printed errors are relative to the dropped output's energy: 1.0 is
//! what dropping costs; `co-pick` is the most co-picked kept expert.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use cortiq_core::format::CmfModel;
use cortiq_core::quant::{dequant_tensor, f16_to_f32};

use crate::moegptq::expert_out;

pub(crate) struct Hy4AliasOpts<'a> {
    pub model: &'a str,
    pub captures: &'a [String],
    pub keep: &'a str,
    pub output: &'a str,
    pub rows: usize,
    pub cands: usize,
    pub threads: usize,
    /// Ridge of the correction, relative to the Gram's mean diagonal
    pub ridge: f64,
}

fn read_u16(path: &Path) -> anyhow::Result<Vec<u16>> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn read_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn tensor_f32(model: &CmfModel, name: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let e = model.tensor(name).ok_or_else(|| anyhow::anyhow!("{name}: not in the file"))?;
    let mut v = vec![0f32; e.n_elems()];
    dequant_tensor(e, model.entry_bytes(e), &mut v).map_err(anyhow::Error::msg)?;
    Ok((v, e.shape.iter().map(|&s| s as usize).collect()))
}

pub(crate) fn run(o: &Hy4AliasOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let model = CmfModel::open(o.model)?;
    let keep: HashMap<String, Vec<usize>> = serde_json::from_str(&std::fs::read_to_string(o.keep)?)?;
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("hy4_flash")).and_then(|v| v.get("text_config")).cloned().unwrap_or_default();
    let limit = prov["swiglu_limit"].as_f64().unwrap_or(0.0) as f32;
    let metas: Vec<(std::path::PathBuf, serde_json::Value)> = o
        .captures
        .iter()
        .map(|d| {
            let p = std::path::PathBuf::from(d);
            let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p.join("meta.json"))?)?;
            Ok((p, m))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut layers: Vec<usize> = keep.keys().filter_map(|k| k.parse().ok()).collect();
    layers.sort_unstable();
    let mut out = serde_json::Map::new();
    let (mut s_best, mut s_router, mut s_n) = (0f64, 0f64, 0usize);
    for li in layers {
        let tl = Instant::now();
        let m = format!("model.layers.{li}.mlp.");
        let (_, bs) = tensor_f32(&model, &format!("{m}gate.e_score_correction_bias"))?;
        let ne = bs[0];
        let (_, ls) = tensor_f32(&model, &format!("{m}latent_down.weight"))?;
        let dim = ls[0];
        let kept: Vec<usize> = keep[&li.to_string()].clone();
        let mut is_kept = vec![false; ne];
        for &e in &kept {
            is_kept[e] = true;
        }
        // every capture's rows of this layer, per expert: (weight, capture, token)
        let mut xs: Vec<Vec<u16>> = Vec::new();
        let mut pks: Vec<(Vec<u16>, usize)> = Vec::new();
        let mut by_e: Vec<Vec<(f32, usize, usize)>> = vec![Vec::new(); ne];
        for (ci, (p, meta)) in metas.iter().enumerate() {
            let n = meta["tokens"].as_u64().unwrap_or(0) as usize;
            let k = meta["top_k"].as_u64().unwrap_or(16) as usize;
            let x = read_u16(&p.join(format!("L{li}.x")))?;
            anyhow::ensure!(x.len() == n * dim, "{}: L{li}.x has {} values, want {n}×{dim} (capture without CMF_HY4_CAP_NOX)", p.display(), x.len());
            let picks = read_u16(&p.join(format!("L{li}.picks")))?;
            let w = read_f32(&p.join(format!("L{li}.w")))?;
            for t in 0..n {
                for j in 0..k {
                    by_e[picks[t * k + j] as usize].push((w[t * k + j], ci, t));
                }
            }
            xs.push(x);
            pks.push((picks, k));
        }
        // every expert of the layer decoded once (f32)
        let names = ["gate_proj", "up_proj", "down_proj"];
        let (_, gs) = tensor_f32(&model, &format!("{m}experts.0.gate_proj.weight"))?;
        let inter = gs[0];
        let dec: Vec<anyhow::Result<[Vec<f32>; 3]>> = {
            let nt = o.threads.max(1);
            let mut res: Vec<Option<anyhow::Result<[Vec<f32>; 3]>>> = (0..ne).map(|_| None).collect();
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..nt)
                    .map(|w| {
                        let (model, m) = (&model, &m);
                        sc.spawn(move || {
                            (w..ne)
                                .step_by(nt)
                                .map(|e| {
                                    let r = (|| -> anyhow::Result<[Vec<f32>; 3]> {
                                        let g = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[0]))?.0;
                                        let u = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[1]))?.0;
                                        let d = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[2]))?.0;
                                        Ok([g, u, d])
                                    })();
                                    (e, r)
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                for h in hs {
                    for (e, r) in h.join().unwrap_or_default() {
                        res[e] = Some(r);
                    }
                }
            });
            res.into_iter().map(|r| r.unwrap_or_else(|| Err(anyhow::anyhow!("an expert decode worker panicked")))).collect()
        };
        let experts: Vec<[Vec<f32>; 3]> = dec.into_iter().collect::<anyhow::Result<_>>()?;
        let dropped: Vec<usize> = (0..ne).filter(|&e| !is_kept[e]).collect();
        // per dropped expert: (alias, best error, router-choice error, rows)
        let nt = o.threads.max(1);
        let mut found: Vec<(usize, usize, f64, f64, usize)> = Vec::with_capacity(dropped.len());
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|wk| {
                    let (dropped, by_e, xs, pks, experts, is_kept) = (&dropped, &by_e, &xs, &pks, &experts, &is_kept);
                    sc.spawn(move || {
                        let mut r = Vec::new();
                        for &d in dropped.iter().skip(wk).step_by(nt) {
                            let mut rows = by_e[d].clone();
                            if rows.is_empty() {
                                // never picked in the captures: the router's own top kept choice overall
                                r.push((d, usize::MAX, f64::NAN, f64::NAN, 0));
                                continue;
                            }
                            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                            rows.truncate(o.rows);
                            let mr = rows.len();
                            let mut x = vec![0f32; mr * dim];
                            for (i, &(_, ci, t)) in rows.iter().enumerate() {
                                for (v, &h) in x[i * dim..(i + 1) * dim].iter_mut().zip(&xs[ci][t * dim..(t + 1) * dim]) {
                                    *v = f16_to_f32(h);
                                }
                            }
                            let wts: Vec<f64> = rows.iter().map(|r| r.0 as f64).collect();
                            let [g, u, dn] = &experts[d];
                            let yd = expert_out(&x, mr, dim, g, u, dn, inter, limit, 1);
                            let energy: f64 = (0..mr).map(|i| wts[i] * yd[i * dim..(i + 1) * dim].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sum::<f64>().max(1e-30);
                            // the kept experts the full model co-picks with d, weighted by d's weight
                            let mut co: HashMap<usize, f64> = HashMap::new();
                            for &(wd, ci, t) in &by_e[d] {
                                let (p, k) = (&pks[ci].0, pks[ci].1);
                                for &e in &p[t * k..(t + 1) * k] {
                                    if is_kept[e as usize] {
                                        *co.entry(e as usize).or_default() += wd as f64;
                                    }
                                }
                            }
                            let mut order: Vec<usize> = co.keys().copied().collect();
                            order.sort_by(|a, b| co[b].partial_cmp(&co[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
                            order.truncate(o.cands.max(1));
                            if order.is_empty() {
                                r.push((d, usize::MAX, f64::NAN, f64::NAN, 0));
                                continue;
                            }
                            let mut errs: HashMap<usize, f64> = HashMap::new();
                            for &j in &order {
                                let [g, u, dn] = &experts[j];
                                let yk = expert_out(&x, mr, dim, g, u, dn, inter, limit, 1);
                                let e: f64 = (0..mr)
                                    .map(|i| wts[i] * yd[i * dim..(i + 1) * dim].iter().zip(&yk[i * dim..(i + 1) * dim]).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>())
                                    .sum();
                                errs.insert(j, e);
                            }
                            let (bj, be) = order.iter().map(|&j| (j, errs[&j])).fold((order[0], f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
                            r.push((d, bj, be / energy, errs[&order[0]] / energy, mr));
                        }
                        r
                    })
                })
                .collect();
            for h in hs {
                found.extend(h.join().unwrap_or_default());
            }
        });
        let mut alias: Vec<usize> = (0..ne).collect();
        let (mut lb, mut lr, mut ln) = (0f64, 0f64, 0usize);
        // a dropped expert never seen in the captures takes the layer's most used kept expert
        let fallback = kept.iter().copied().max_by_key(|&e| by_e[e].len()).unwrap_or(0);
        for &(d, k, be, re, mr) in &found {
            alias[d] = if k == usize::MAX { fallback } else { k };
            if mr > 0 && be.is_finite() && re.is_finite() {
                lb += be;
                lr += re;
                ln += 1;
            }
        }
        s_best += lb;
        s_router += lr;
        s_n += ln;
        let uniq = {
            let mut v: Vec<usize> = dropped.iter().map(|&d| alias[d]).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        eprintln!(
            "hy4-alias L{li}: {} dropped → {uniq} kept experts; mean relative error resonance {:.3}, co-pick {:.3} (dropping 1.000); {:.0} s",
            dropped.len(),
            lb / ln.max(1) as f64,
            lr / ln.max(1) as f64,
            tl.elapsed().as_secs_f64()
        );
        out.insert(li.to_string(), serde_json::json!(alias));
    }
    std::fs::write(o.output, serde_json::to_string(&serde_json::Value::Object(out))?)?;
    eprintln!(
        "hy4-alias: {} → {}; mean relative error resonance {:.3}, co-pick {:.3}; {:.0} s",
        o.keep,
        o.output,
        s_best / s_n.max(1) as f64,
        s_router / s_n.max(1) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `cortiq hy4-alias --probe N`: the routed FFN's latent output error of the
/// ways to halve a layer's expert bytes, on N held-out captured tokens (every
/// 8th; the rest calibrate), each relative to the full output's energy:
/// `drop` — the router's own 16 of all experts, the dropped ones' picks
/// zero; `drop+renorm` — the same with the kept picks' weights rescaled to
/// the full sum; `half` — every expert kept with its `frac` most important
/// neurons (DTG-MA-style: Σ weight·|activation|·‖down column‖ over the
/// expert's calibration rows), the router untouched; `half+fcd` — the same
/// neurons with the down projection refit by least squares to the full
/// expert's output (D′ = D·H[:,K]·(H[K,K] + λI)⁻¹, H the weighted Gram of
/// the activations), the closed-form correction step.
pub(crate) fn probe(o: &Hy4AliasOpts, n_eval: usize, frac: f64) -> anyhow::Result<()> {
    let model = CmfModel::open(o.model)?;
    let keep: HashMap<String, Vec<usize>> = serde_json::from_str(&std::fs::read_to_string(o.keep)?)?;
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("hy4_flash")).and_then(|v| v.get("text_config")).cloned().unwrap_or_default();
    let limit = prov["swiglu_limit"].as_f64().unwrap_or(0.0) as f32;
    let mut layers: Vec<usize> = keep.keys().filter_map(|k| k.parse().ok()).collect();
    layers.sort_unstable();
    let nt = o.threads.max(1);
    let (mut sd, mut sr, mut sh, mut sf) = (0f64, 0f64, 0f64, 0f64);
    for &li in &layers {
        let tl = Instant::now();
        let m = format!("model.layers.{li}.mlp.");
        let ne = tensor_f32(&model, &format!("{m}gate.e_score_correction_bias"))?.1[0];
        let dim = tensor_f32(&model, &format!("{m}latent_down.weight"))?.1[0];
        let inter = tensor_f32(&model, &format!("{m}experts.0.gate_proj.weight"))?.1[0];
        let mut is_kept = vec![false; ne];
        for &e in &keep[&li.to_string()] {
            is_kept[e] = true;
        }
        // tokens: (x row, picks, weights); every 8th held out for the evaluation
        let (mut cal, mut ev): (Vec<(Vec<f32>, Vec<(usize, f32)>)>, Vec<(Vec<f32>, Vec<(usize, f32)>)>) = (Vec::new(), Vec::new());
        for d in o.captures {
            let p = Path::new(d);
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p.join("meta.json"))?)?;
            let n = meta["tokens"].as_u64().unwrap_or(0) as usize;
            let k = meta["top_k"].as_u64().unwrap_or(16) as usize;
            let x = read_u16(&p.join(format!("L{li}.x")))?;
            let picks = read_u16(&p.join(format!("L{li}.picks")))?;
            let w = read_f32(&p.join(format!("L{li}.w")))?;
            for t in 0..n {
                let row: Vec<f32> = x[t * dim..(t + 1) * dim].iter().map(|&h| f16_to_f32(h)).collect();
                let pw: Vec<(usize, f32)> = (0..k).map(|j| (picks[t * k + j] as usize, w[t * k + j])).collect();
                if t % 8 == 0 { ev.push((row, pw)) } else { cal.push((row, pw)) }
            }
        }
        let step = (ev.len() / n_eval.max(1)).max(1);
        let ev: Vec<_> = ev.into_iter().step_by(step).take(n_eval).collect();
        // per expert: its calibration rows (heaviest `rows`), its neuron importance, its kept neurons
        let mut rows_of: Vec<Vec<(f32, usize)>> = vec![Vec::new(); ne];
        for (i, (_, pw)) in cal.iter().enumerate() {
            for &(e, wt) in pw {
                rows_of[e].push((wt, i));
            }
        }
        let names = ["gate_proj", "up_proj", "down_proj"];
        let mut res: Vec<Option<anyhow::Result<([Vec<f32>; 3], Vec<usize>, Vec<f32>)>>> = (0..ne).map(|_| None).collect();
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|wk| {
                    let (model, m, rows_of, cal) = (&model, &m, &rows_of, &cal);
                    sc.spawn(move || {
                        (wk..ne)
                            .step_by(nt)
                            .map(|e| {
                                let r = (|| -> anyhow::Result<([Vec<f32>; 3], Vec<usize>, Vec<f32>)> {
                                    let g = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[0]))?.0;
                                    let u = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[1]))?.0;
                                    let d = tensor_f32(model, &format!("{m}experts.{e}.{}.weight", names[2]))?.0;
                                    let mut rs = rows_of[e].clone();
                                    rs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                                    rs.truncate(o.rows.max(1));
                                    let mut imp = vec![0f64; inter];
                                    let mut hrows: Vec<f32> = Vec::new();
                                    if !rs.is_empty() {
                                        let mut x = Vec::with_capacity(rs.len() * dim);
                                        for &(_, i) in &rs {
                                            x.extend_from_slice(&cal[i].0);
                                        }
                                        let h = crate::moegptq::swiglu_rows_pub(&x, rs.len(), dim, &g, &u, inter, limit, 1);
                                        let dn: Vec<f64> = (0..inter).map(|i| (0..dim).map(|r| (d[r * inter + i] as f64).powi(2)).sum::<f64>().sqrt()).collect();
                                        for (ri, &(wt, _)) in rs.iter().enumerate() {
                                            for i in 0..inter {
                                                imp[i] += wt as f64 * (h[ri * inter + i] as f64).abs() * dn[i];
                                            }
                                        }
                                        hrows = h;
                                    }
                                    let mut ord: Vec<usize> = (0..inter).collect();
                                    ord.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
                                    ord.truncate(((inter as f64 * frac).round() as usize).max(1));
                                    ord.sort_unstable();
                                    // the correction: D′ (dim × |K|) from the weighted Gram of the activations
                                    let kn = ord.len();
                                    let mut dfix = vec![0f32; dim * kn];
                                    let nr = rs.len();
                                    if nr > 0 {
                                        let mut hka = vec![0f64; kn * inter];
                                        for (ri, &(wt, _)) in rs.iter().enumerate() {
                                            let hr = &hrows[ri * inter..(ri + 1) * inter];
                                            for (a, &ka) in ord.iter().enumerate() {
                                                let v = wt as f64 * hr[ka] as f64;
                                                if v != 0.0 {
                                                    for (o, &hb) in hka[a * inter..(a + 1) * inter].iter_mut().zip(hr) {
                                                        *o += v * hb as f64;
                                                    }
                                                }
                                            }
                                        }
                                        let mut hkk: Vec<f64> = (0..kn * kn).map(|ab| hka[(ab / kn) * inter + ord[ab % kn]]).collect();
                                        let tr = (0..kn).map(|a| hkk[a * kn + a]).sum::<f64>() / kn as f64;
                                        let lam = o.ridge * tr.max(1e-30);
                                        for a in 0..kn {
                                            hkk[a * kn + a] += lam;
                                        }
                                        if let Some(l) = cholesky(&mut hkk, kn) {
                                            // per output row r: solve (H_KK) z = H_KA · D[r,:]ᵀ, then D′[r,:] = z
                                            for r in 0..dim {
                                                let drow = &d[r * inter..(r + 1) * inter];
                                                let mut b: Vec<f64> = (0..kn).map(|a| hka[a * inter..(a + 1) * inter].iter().zip(drow).map(|(&x, &y)| x * y as f64).sum()).collect();
                                                chol_solve(&l, kn, &mut b);
                                                for a in 0..kn {
                                                    dfix[r * kn + a] = b[a] as f32;
                                                }
                                            }
                                        } else {
                                            for r in 0..dim {
                                                for (a, &ka) in ord.iter().enumerate() {
                                                    dfix[r * kn + a] = d[r * inter + ka];
                                                }
                                            }
                                        }
                                    } else {
                                        for r in 0..dim {
                                            for (a, &ka) in ord.iter().enumerate() {
                                                dfix[r * kn + a] = d[r * inter + ka];
                                            }
                                        }
                                    }
                                    Ok(([g, u, d], ord, dfix))
                                })();
                                (e, r)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in hs {
                for (e, r) in h.join().unwrap_or_default() {
                    res[e] = Some(r);
                }
            }
        });
        let ex: Vec<([Vec<f32>; 3], Vec<usize>, Vec<f32>)> = res.into_iter().map(|r| r.unwrap_or_else(|| Err(anyhow::anyhow!("a worker panicked")))).collect::<anyhow::Result<_>>()?;
        // the held-out tokens: full, drop, drop+renorm, half
        let parts: Vec<(f64, f64, f64, f64, f64)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|wk| {
                    let (ev, ex, is_kept) = (&ev, &ex, &is_kept);
                    sc.spawn(move || {
                        let mut acc = (0f64, 0f64, 0f64, 0f64, 0f64);
                        for (x, pw) in ev.iter().skip(wk).step_by(nt) {
                            let (mut yf, mut yk, mut yh, mut yc) = (vec![0f64; dim], vec![0f64; dim], vec![0f64; dim], vec![0f64; dim]);
                            let (mut wall, mut wkept) = (0f64, 0f64);
                            for &(e, wt) in pw {
                                let ([g, u, d], keepn, dfix) = &ex[e];
                                let kn = keepn.len();
                                let h = crate::moegptq::swiglu_rows_pub(x, 1, dim, g, u, inter, limit, 1);
                                wall += wt as f64;
                                for r in 0..dim {
                                    let drow = &d[r * inter..(r + 1) * inter];
                                    let full: f64 = drow.iter().zip(&h).map(|(&a, &b)| a as f64 * b as f64).sum();
                                    let half: f64 = keepn.iter().map(|&i| drow[i] as f64 * h[i] as f64).sum();
                                    yf[r] += wt as f64 * full;
                                    yh[r] += wt as f64 * half;
                                    let fixd: f64 = keepn.iter().enumerate().map(|(a, &i)| dfix[r * kn + a] as f64 * h[i] as f64).sum();
                                    yc[r] += wt as f64 * fixd;
                                    if is_kept[e] {
                                        yk[r] += wt as f64 * full;
                                    }
                                }
                                if is_kept[e] {
                                    wkept += wt as f64;
                                }
                            }
                            let sc = if wkept > 0.0 { wall / wkept } else { 0.0 };
                            for r in 0..dim {
                                acc.0 += yf[r] * yf[r];
                                acc.1 += (yf[r] - yk[r]).powi(2);
                                acc.2 += (yf[r] - sc * yk[r]).powi(2);
                                acc.3 += (yf[r] - yh[r]).powi(2);
                                acc.4 += (yf[r] - yc[r]).powi(2);
                            }
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        let (e, a, b, c, f) = parts.iter().fold((0f64, 0f64, 0f64, 0f64, 0f64), |s, p| (s.0 + p.0, s.1 + p.1, s.2 + p.2, s.3 + p.3, s.4 + p.4));
        let (a, b, c, f) = (a / e.max(1e-30), b / e.max(1e-30), c / e.max(1e-30), f / e.max(1e-30));
        sd += a;
        sr += b;
        sh += c;
        sf += f;
        eprintln!("hy4-probe L{li}: {} held-out tokens, inter {inter}: error drop {a:.4}, drop+renorm {b:.4}, half neurons {c:.4}, half+fcd {f:.4}; {:.0} s", ev.len(), tl.elapsed().as_secs_f64());
    }
    let n = layers.len().max(1) as f64;
    eprintln!("hy4-probe mean: drop {:.4}, drop+renorm {:.4}, half neurons ({frac}) {:.4}, half+fcd {:.4}", sd / n, sr / n, sh / n, sf / n);
    Ok(())
}

/// In-place Cholesky of a symmetric positive matrix (lower factor); None if not positive.
fn cholesky(a: &mut [f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solves L·Lᵀ·x = b in place.
fn chol_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// `cortiq hy4-alias --ghost N`: whether a cheap linear stand-in recovers
/// the dropped experts' share of a routed FFN (the router's own 16 of all
/// experts, the dropped picks absent, no renormalization). Per layer, on N
/// held-out tokens: `drop` alone; `+shared` — one least-squares map per layer
/// from `s·x` (s = the dropped picks' summed weight) to their summed output;
/// `+ghost` — a least-squares linear map `B_d` per dropped expert from its
/// own calibration rows (`rows` of them, every k-th), each pick of `d`
/// adding `w·B_d·x`. Errors relative to the full output's energy.
pub(crate) fn ghost(o: &Hy4AliasOpts, n_eval: usize) -> anyhow::Result<()> {
    let model = CmfModel::open(o.model)?;
    let keep: HashMap<String, Vec<usize>> = serde_json::from_str(&std::fs::read_to_string(o.keep)?)?;
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("hy4_flash")).and_then(|v| v.get("text_config")).cloned().unwrap_or_default();
    let limit = prov["swiglu_limit"].as_f64().unwrap_or(0.0) as f32;
    let mut layers: Vec<usize> = keep.keys().filter_map(|k| k.parse().ok()).collect();
    layers.sort_unstable();
    let nt = o.threads.max(1);
    let (mut s0, mut s1, mut s2) = (0f64, 0f64, 0f64);
    for &li in &layers {
        let tl = Instant::now();
        let m = format!("model.layers.{li}.mlp.");
        let ne = tensor_f32(&model, &format!("{m}gate.e_score_correction_bias"))?.1[0];
        let dim = tensor_f32(&model, &format!("{m}latent_down.weight"))?.1[0];
        let inter = tensor_f32(&model, &format!("{m}experts.0.gate_proj.weight"))?.1[0];
        let mut is_kept = vec![false; ne];
        for &e in &keep[&li.to_string()] {
            is_kept[e] = true;
        }
        let (mut cal, mut ev): (Vec<(Vec<f32>, Vec<(usize, f32)>)>, Vec<(Vec<f32>, Vec<(usize, f32)>)>) = (Vec::new(), Vec::new());
        for d in o.captures {
            let p = Path::new(d);
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p.join("meta.json"))?)?;
            let n = meta["tokens"].as_u64().unwrap_or(0) as usize;
            let k = meta["top_k"].as_u64().unwrap_or(16) as usize;
            let x = read_u16(&p.join(format!("L{li}.x")))?;
            let picks = read_u16(&p.join(format!("L{li}.picks")))?;
            let w = read_f32(&p.join(format!("L{li}.w")))?;
            for t in 0..n {
                let row: Vec<f32> = x[t * dim..(t + 1) * dim].iter().map(|&h| f16_to_f32(h)).collect();
                let pw: Vec<(usize, f32)> = (0..k).map(|j| (picks[t * k + j] as usize, w[t * k + j])).collect();
                if t % 8 == 0 { ev.push((row, pw)) } else { cal.push((row, pw)) }
            }
        }
        let step = (ev.len() / n_eval.max(1)).max(1);
        let ev: Vec<_> = ev.into_iter().step_by(step).take(n_eval).collect();
        let names = ["gate_proj", "up_proj", "down_proj"];
        let load = |e: usize| -> anyhow::Result<[Vec<f32>; 3]> {
            Ok([
                tensor_f32(&model, &format!("{m}experts.{e}.{}.weight", names[0]))?.0,
                tensor_f32(&model, &format!("{m}experts.{e}.{}.weight", names[1]))?.0,
                tensor_f32(&model, &format!("{m}experts.{e}.{}.weight", names[2]))?.0,
            ])
        };
        // calibration rows of every dropped expert (every k-th, up to `rows`)
        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (i, (_, pw)) in cal.iter().enumerate() {
            for &(e, _) in pw {
                if !is_kept[e] {
                    rows_of[e].push(i);
                }
            }
        }
        let dropped: Vec<usize> = (0..ne).filter(|&e| !is_kept[e]).collect();
        // per dropped expert: its weights and ghost B_d (dim × dim, y = B x), and its outputs on its calibration rows
        let ghosts: Vec<anyhow::Result<(usize, [Vec<f32>; 3], Vec<f32>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|wk| {
                    let (dropped, rows_of, cal, load) = (&dropped, &rows_of, &cal, &load);
                    sc.spawn(move || {
                        let mut out = Vec::new();
                        for &d in dropped.iter().skip(wk).step_by(nt) {
                            let r = (|| -> anyhow::Result<(usize, [Vec<f32>; 3], Vec<f32>)> {
                                let ew = load(d)?;
                                let all = &rows_of[d];
                                let st = (all.len() / o.rows.max(1)).max(1);
                                let rs: Vec<usize> = all.iter().copied().step_by(st).take(o.rows.max(1)).collect();
                                let mr = rs.len();
                                let mut b = vec![0f32; dim * dim];
                                if mr >= 8 {
                                    let mut x = Vec::with_capacity(mr * dim);
                                    for &i in &rs {
                                        x.extend_from_slice(&cal[i].0);
                                    }
                                    let y = expert_out(&x, mr, dim, &ew[0], &ew[1], &ew[2], inter, limit, 1);
                                    // G = XᵀX + λI, R = XᵀY; B = Rᵀ G⁻¹ (row r of B solves G b = R[:, r])
                                    let mut g = vec![0f64; dim * dim];
                                    let mut rr = vec![0f64; dim * dim];
                                    for i in 0..mr {
                                        let xi = &x[i * dim..(i + 1) * dim];
                                        let yi = &y[i * dim..(i + 1) * dim];
                                        for a in 0..dim {
                                            let xa = xi[a] as f64;
                                            if xa == 0.0 {
                                                continue;
                                            }
                                            let ga = &mut g[a * dim..(a + 1) * dim];
                                            for (gv, &xb) in ga.iter_mut().zip(xi) {
                                                *gv += xa * xb as f64;
                                            }
                                            let ra = &mut rr[a * dim..(a + 1) * dim];
                                            for (rv, &yb) in ra.iter_mut().zip(yi) {
                                                *rv += xa * yb as f64;
                                            }
                                        }
                                    }
                                    let tr = (0..dim).map(|a| g[a * dim + a]).sum::<f64>() / dim as f64;
                                    for a in 0..dim {
                                        g[a * dim + a] += o.ridge * tr.max(1e-30);
                                    }
                                    if let Some(l) = cholesky(&mut g, dim) {
                                        for r in 0..dim {
                                            let mut col: Vec<f64> = (0..dim).map(|a| rr[a * dim + r]).collect();
                                            chol_solve(&l, dim, &mut col);
                                            for a in 0..dim {
                                                b[r * dim + a] = col[a] as f32;
                                            }
                                        }
                                    }
                                }
                                Ok((d, ew, b))
                            })();
                            out.push(r);
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });
        let mut gmap: HashMap<usize, ([Vec<f32>; 3], Vec<f32>)> = HashMap::new();
        for g in ghosts {
            let (d, ew, b) = g?;
            gmap.insert(d, (ew, b));
        }
        // the kept experts decoded once
        let kept_ids: Vec<usize> = (0..ne).filter(|&e| is_kept[e]).collect();
        let kw: Vec<anyhow::Result<(usize, [Vec<f32>; 3])>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|wk| {
                    let (kept_ids, load) = (&kept_ids, &load);
                    sc.spawn(move || kept_ids.iter().skip(wk).step_by(nt).map(|&e| load(e).map(|w| (e, w))).collect::<Vec<_>>())
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });
        let mut kmap: HashMap<usize, [Vec<f32>; 3]> = HashMap::new();
        for r in kw {
            let (e, w) = r?;
            kmap.insert(e, w);
        }
        // the shared map: z = s·x → the dropped picks' summed output, over the calibration tokens (every 4th)
        let mut gs = vec![0f64; dim * dim];
        let mut rs = vec![0f64; dim * dim];
        for (x, pw) in cal.iter().step_by(8) {
            let s: f64 = pw.iter().filter(|(e, _)| !is_kept[*e]).map(|(_, w)| *w as f64).sum();
            if s == 0.0 {
                continue;
            }
            let mut mo = vec![0f64; dim];
            for &(e, wt) in pw.iter().filter(|(e, _)| !is_kept[*e]) {
                let (ew, _) = &gmap[&e];
                let y = expert_out(x, 1, dim, &ew[0], &ew[1], &ew[2], inter, limit, 1);
                for (a, &v) in mo.iter_mut().zip(&y) {
                    *a += wt as f64 * v as f64;
                }
            }
            for a in 0..dim {
                let za = s * x[a] as f64;
                for bb in 0..dim {
                    gs[a * dim + bb] += za * s * x[bb] as f64;
                    rs[a * dim + bb] += za * mo[bb];
                }
            }
        }
        let tr = (0..dim).map(|a| gs[a * dim + a]).sum::<f64>() / dim as f64;
        for a in 0..dim {
            gs[a * dim + a] += o.ridge * tr.max(1e-30);
        }
        let mut bsh = vec![0f32; dim * dim];
        if let Some(l) = cholesky(&mut gs, dim) {
            for r in 0..dim {
                let mut col: Vec<f64> = (0..dim).map(|a| rs[a * dim + r]).collect();
                chol_solve(&l, dim, &mut col);
                for a in 0..dim {
                    bsh[r * dim + a] = col[a] as f32;
                }
            }
        }
        // the held-out tokens
        let (mut ef, mut e0, mut e1, mut e2) = (0f64, 0f64, 0f64, 0f64);
        for (x, pw) in &ev {
            let (mut yf, mut yk, mut yg) = (vec![0f64; dim], vec![0f64; dim], vec![0f64; dim]);
            let mut s = 0f64;
            for &(e, wt) in pw {
                let full = if is_kept[e] {
                    let ew = &kmap[&e];
                    expert_out(x, 1, dim, &ew[0], &ew[1], &ew[2], inter, limit, 1)
                } else {
                    let (ew, b) = &gmap[&e];
                    s += wt as f64;
                    let gx: Vec<f64> = (0..dim).map(|r| b[r * dim..(r + 1) * dim].iter().zip(x).map(|(&p, &q)| p as f64 * q as f64).sum()).collect();
                    for (a, v) in yg.iter_mut().zip(&gx) {
                        *a += wt as f64 * v;
                    }
                    expert_out(x, 1, dim, &ew[0], &ew[1], &ew[2], inter, limit, 1)
                };
                for r in 0..dim {
                    yf[r] += wt as f64 * full[r] as f64;
                    if is_kept[e] {
                        yk[r] += wt as f64 * full[r] as f64;
                    }
                }
            }
            for r in 0..dim {
                let sh: f64 = bsh[r * dim..(r + 1) * dim].iter().zip(x).map(|(&p, &q)| p as f64 * s * q as f64).sum();
                ef += yf[r] * yf[r];
                e0 += (yf[r] - yk[r]).powi(2);
                e1 += (yf[r] - yk[r] - sh).powi(2);
                e2 += (yf[r] - yk[r] - yg[r]).powi(2);
            }
        }
        let (a, b, c) = (e0 / ef.max(1e-30), e1 / ef.max(1e-30), e2 / ef.max(1e-30));
        s0 += a;
        s1 += b;
        s2 += c;
        eprintln!("hy4-ghost L{li}: {} held-out tokens: error drop {a:.4}, +shared {b:.4}, +ghost {c:.4}; {:.0} s", ev.len(), tl.elapsed().as_secs_f64());
    }
    let n = layers.len().max(1) as f64;
    eprintln!("hy4-ghost mean: drop {:.4}, +shared {:.4}, +ghost {:.4}", s0 / n, s1 / n, s2 / n);
    Ok(())
}

/// `cortiq hy4-alias --dump DIR`: every routed expert of the `--keep` layers
/// decoded to f16 for the offline studies: `DIR/L<li>.experts.f16`, expert
/// after expert, gate `[inter, latent]`, up `[inter, latent]`, down
/// `[latent, inter]`; `DIR/L<li>.json` the shapes.
pub(crate) fn dump(o: &Hy4AliasOpts, dir: &str) -> anyhow::Result<()> {
    use std::io::Write;
    let model = CmfModel::open(o.model)?;
    let keep: HashMap<String, Vec<usize>> = serde_json::from_str(&std::fs::read_to_string(o.keep)?)?;
    std::fs::create_dir_all(dir)?;
    let mut layers: Vec<usize> = keep.keys().filter_map(|k| k.parse().ok()).collect();
    layers.sort_unstable();
    let nt = o.threads.max(1);
    for li in layers {
        let tl = Instant::now();
        let m = format!("model.layers.{li}.mlp.");
        let ne = tensor_f32(&model, &format!("{m}gate.e_score_correction_bias"))?.1[0];
        let dim = tensor_f32(&model, &format!("{m}latent_down.weight"))?.1[0];
        let inter = tensor_f32(&model, &format!("{m}experts.0.gate_proj.weight"))?.1[0];
        let mut f = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(Path::new(dir).join(format!("L{li}.experts.f16")))?);
        for chunk in (0..ne).collect::<Vec<_>>().chunks(nt) {
            let parts: Vec<anyhow::Result<Vec<u8>>> = std::thread::scope(|sc| {
                let hs: Vec<_> = chunk
                    .iter()
                    .map(|&e| {
                        let (model, m) = (&model, &m);
                        sc.spawn(move || -> anyhow::Result<Vec<u8>> {
                            let mut b = Vec::with_capacity(3 * inter * dim * 2);
                            for nm in ["gate_proj", "up_proj", "down_proj"] {
                                for v in tensor_f32(model, &format!("{m}experts.{e}.{nm}.weight"))?.0 {
                                    b.extend_from_slice(&cortiq_core::quant::f32_to_f16(v).to_le_bytes());
                                }
                            }
                            Ok(b)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a dump worker panicked")))).collect()
            });
            for p in parts {
                f.write_all(&p?)?;
            }
        }
        f.flush()?;
        std::fs::write(Path::new(dir).join(format!("L{li}.json")), serde_json::json!({"experts": ne, "latent": dim, "inter": inter}).to_string())?;
        eprintln!("hy4-dump L{li}: {ne} experts, latent {dim}, inter {inter}, {:.0} s", tl.elapsed().as_secs_f64());
    }
    Ok(())
}
