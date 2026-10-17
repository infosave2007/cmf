//! `cortiq quantize-moe`: GPTQ for the experts of a Flash-Next-class MoE
//! (`qwen4_exp`): routed and shared experts get gate/up in q2tp and down in
//! q4tp (the layout the q2tp device path expects for every expert slot);
//! every other tensor is copied from the calibration file.
//!
//! * Calibration: the calibration model (a q4tp file of the same model) runs
//!   its ordinary device frames over the corpus; each layer frame in the
//!   captured range copies its MoE input rows and router logits out right
//!   after routing (`Pipeline::qwen4_capture`). A token's routed experts are
//!   the top-k of its logits; the shared expert sees every token.
//! * Per expert, on the GPU: the input Hessian of gate/up from its rows, the
//!   down input `silu(X·W_gᵀ)⊙(X·W_uᵀ)` from the full-precision weights (f16
//!   planes, matrix units) and its Hessian, both accumulated in row chunks;
//!   then the device folds, the q2tp sweeps (placed rule, debiased) and the
//!   q4tp sweep.
//! * Full-precision weights come from the original safetensors shards,
//!   fetched one layer at a time (the next layer's in the background) and
//!   deleted when done; the output is streamed, so neither the bf16
//!   checkpoint nor the result is ever held whole.
//! * `--patch-shared FILE` re-encodes only the shared experts of an existing
//!   output in place (their q2tp payloads fit the old slots).

use crate::convert::{Quant, fetch, open_safetensors, quantize_2d, to_f32};
use crate::gptq::{CalibSource, calib_source_tokens, gemm_rows_square, gptq_quantize_q2tp_dev, gptq_quantize_q4tp_dev, q2tp_debias_rows, split_calib_budget};
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::quant::f32_to_f16;
use cortiq_core::types::TensorDtype;
use cortiq_engine::gpu_wgpu::{Ctx, train_mm as tm, train_ops as to};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub struct MoeOpts<'a> {
    pub calib_model: &'a str,
    pub hf: &'a str,
    pub calib: &'a [String],
    pub tokens: usize,
    pub window: usize,
    pub layers_per_pass: usize,
    pub lambda: f64,
    pub min_rows: usize,
    pub scratch: &'a str,
    pub threads: usize,
    pub output: &'a str,
    pub patch_shared: Option<&'a str>,
}

/// Input rows per device pass of one expert.
const CHUNK: usize = 16384;

pub fn run(o: &MoeOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let c = to::dev().ok_or_else(|| anyhow::anyhow!("quantize-moe needs a wgpu GPU"))?;
    let cmodel = Arc::new(CmfModel::open_sharded(o.calib_model)?);
    let arch = cmodel.arch().clone();
    let nl = arch.num_layers;
    let hidden = arch.hidden_size;
    let mut pipe = Pipeline::from_model(&cmodel, SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    let (dirs0, _) = pipe.qwen4_expert_dirs(0).ok_or_else(|| anyhow::anyhow!("not a qwen4_exp (Flash-Next-class) model"))?;
    let ne = dirs0.len();
    let top_k = arch.moe.as_ref().map(|m| m.top_k).unwrap_or(10);
    let inter = cmodel.tensors[dirs0[0].0].shape[0];
    let patch = o.patch_shared;
    eprintln!(
        "quantize-moe: {nl} layers, {ne} experts, top-{top_k}, hidden {hidden}, expert width {inter}{}",
        if patch.is_some() { " — shared experts only, patched in place" } else { "" }
    );

    // calibration windows
    let srcs: Vec<CalibSource> = o.calib.iter().map(|s| CalibSource::parse(s)).collect::<anyhow::Result<_>>()?;
    let budgets = split_calib_budget(&srcs, o.tokens);
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for (src, &n) in srcs.iter().zip(&budgets) {
        let ids = calib_source_tokens(&src.path, n, &|t| pipe.tokenizer.encode(t))?;
        windows.extend(ids.chunks(o.window).filter(|w| w.len() >= 16).map(<[u32]>::to_vec));
    }
    let ntok: usize = windows.iter().map(Vec::len).sum();
    eprintln!("calibrating on {ntok} tokens ({} windows)", windows.len());
    // CMF_MOE_DIAG="0,2560,…": per-expert output error on held-out windows
    // (every CMF_MOE_HOLDOUT-th, default 8th), round-to-nearest against GPTQ
    // with each listed prior (pseudo-rows, 0 = none); no file is written.
    // CMF_MOE_PRIOR = n0: the prior the production encode uses (n0/4 for
    // down), from CMF_MOE_PRIOR_S (8192) evenly spaced calibration tokens.
    // CMF_MOE_RANGE = lo..hi restricts the layers.
    let env_f = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let diag: Option<Vec<f64>> = std::env::var("CMF_MOE_DIAG").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let holdout = if diag.is_some() { env_f("CMF_MOE_HOLDOUT", 8.0) as usize } else { 0 };
    let prior_n0 = env_f("CMF_MOE_PRIOR", 0.0);
    let prior_s = env_f("CMF_MOE_PRIOR_S", 8192.0) as usize;
    let mut held: Vec<bool> = Vec::with_capacity(ntok);
    for (wi, w) in windows.iter().enumerate() {
        held.extend(std::iter::repeat_n(holdout > 0 && wi % holdout == holdout - 1, w.len()));
    }
    let (range_lo, range_hi) = std::env::var("CMF_MOE_RANGE")
        .ok()
        .and_then(|v| v.split_once("..").and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))))
        .unwrap_or((0, nl));
    let (range_lo, range_hi) = (range_lo.min(nl), range_hi.min(nl));
    if let Some(v) = &diag {
        eprintln!("DIAG: held-out every {holdout}th window, priors {v:?} (pseudo-rows), layers {range_lo}..{range_hi}; no file is written");
    }
    if prior_n0 > 0.0 {
        eprintln!("prior: {prior_n0} pseudo-rows (down {}) from {prior_s} calibration tokens", prior_n0 / 4.0);
    }
    // diag accumulators per row-count bin: (experts, Σ‖out‖², Σ err RTN, Σ err per prior)
    const BINS: [usize; 4] = [640, 2560, 10240, usize::MAX];
    let nvar = diag.as_ref().map_or(0, Vec::len);
    let diag_tot = std::sync::Mutex::new(vec![(0usize, 0f64, vec![0f64; nvar + 1]); BINS.len()]);

    // full-precision source: the safetensors index of the original checkpoint
    let scratch = PathBuf::from(o.scratch);
    std::fs::create_dir_all(&scratch)?;
    let token = std::env::var("HF_TOKEN").ok();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(20))
        .timeout_read(std::time::Duration::from_secs(300))
        .build();
    let base = format!("https://huggingface.co/{}/resolve/main", o.hf);
    let idx_path = scratch.join("model.safetensors.index.json");
    fetch(&agent, &format!("{base}/model.safetensors.index.json"), &idx_path, token.as_deref(), true, 8)?;
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path)?)?;
    let wmap = idx["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("bad index"))?.clone();
    // `part` below `….layers.{li}.mlp.` → (source name, shard)
    let src_name = |li: usize, part: &str| -> anyhow::Result<(String, String)> {
        for prefix in ["model.language_model.layers", "model.layers"] {
            let n = format!("{prefix}.{li}.mlp.{part}");
            if let Some(s) = wmap.get(&n).and_then(|v| v.as_str()) {
                return Ok((n, s.to_string()));
            }
        }
        anyhow::bail!("layer {li}: no mlp.{part} in the source index")
    };
    const ROUTED: [&str; 2] = ["experts.gate_up_proj", "experts.down_proj"];
    const SHARED: [&str; 3] = ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight", "shared_expert.down_proj.weight"];

    let mut expert_slots: HashSet<usize> = HashSet::new();
    let mut layer_dirs = Vec::with_capacity(nl);
    let mut shared_dirs = Vec::with_capacity(nl);
    for li in 0..nl {
        let (d, s) = pipe.qwen4_expert_dirs(li).ok_or_else(|| anyhow::anyhow!("layer {li}: no experts"))?;
        for &(g, u, dn) in d.iter().chain(s.iter()) {
            expert_slots.extend([g, u, dn]);
        }
        layer_dirs.push(d);
        shared_dirs.push(s);
    }
    let inter_s = shared_dirs.iter().flatten().map(|&(g, _, _)| cmodel.tensors[g].shape[0]).max().unwrap_or(0);
    let layer_shards = |li: usize| -> anyhow::Result<Vec<String>> {
        let mut v = Vec::new();
        if patch.is_none() {
            for p in ROUTED {
                v.push(src_name(li, p)?.1);
            }
        }
        if shared_dirs[li].is_some() {
            for p in SHARED {
                v.push(src_name(li, p)?.1);
            }
        }
        v.sort();
        v.dedup();
        Ok(v)
    };

    // output: the experts layer by layer (gate, up, down of an expert
    // adjacent, so the expert store reads each in one request), then every
    // other tensor straight from the calibration file. Patch mode collects
    // the shared experts' payloads for the existing file instead.
    let target = patch.map(CmfModel::open_sharded).transpose()?;
    let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::new();
    let mut out = match patch {
        None if diag.is_none() => {
            let avg_name = cmodel.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cmodel.tensors.len().max(1);
            Some(CmfStreamWriter::new(o.output, CmfStreamWriter::head_reserve_for(cmodel.tensors.len(), avg_name))?)
        }
        _ => None,
    };
    let mut emit = |name: &str, dt: TensorDtype, shape: &[usize], bytes: Vec<u8>| -> anyhow::Result<()> {
        match (&mut out, &target) {
            (Some(w), _) => w.push(name, dt, shape, &bytes).map_err(Into::into),
            (None, Some(t)) => {
                let i = t.tensor_index(name).ok_or_else(|| anyhow::anyhow!("{name} not in the patched file"))?;
                anyhow::ensure!(t.tensors[i].shape == shape, "{name}: shape differs in the patched file");
                anyhow::ensure!(bytes.len() as u64 <= t.tensors[i].nbytes, "{name}: new payload larger than its slot");
                patches.push((i, dt, bytes));
                Ok(())
            }
            (None, None) => Ok(()), // diagnostics: nothing is written
        }
    };

    let width = hidden + ne;
    let inter_max = inter.max(inter_s);
    let main_bufs = wgpu_bufs::Bufs::new(c, CHUNK, hidden, inter_max);
    let mut stats = (0usize, 0usize);
    let mut prefetch: Option<std::thread::JoinHandle<anyhow::Result<()>>> = None;
    let pass = o.layers_per_pass.max(1);
    for lo in (range_lo..range_hi).step_by(pass) {
        let hi = (lo + pass).min(range_hi);
        // ── capture ──
        let tc = Instant::now();
        let mut rows: Vec<Vec<u16>> = (lo..hi).map(|_| Vec::with_capacity(ntok * width)).collect();
        for (wi, w) in windows.iter().enumerate() {
            let cap = pipe.qwen4_capture(w, lo, hi).map_err(|e| anyhow::anyhow!("capture: {e}"))?;
            for (k, r) in cap.into_iter().enumerate() {
                rows[k].extend(r.iter().map(|&v| f32_to_f16(v)));
            }
            eprint!("\r  layers {lo}..{hi}: captured {}/{} windows  {:.0} tok/s   ", wi + 1, windows.len(), rows[0].len() as f64 / width as f64 / tc.elapsed().as_secs_f64());
        }
        eprintln!();
        for li in lo..hi {
            let tl = Instant::now();
            let lr = &rows[li - lo];
            let n = lr.len() / width;
            let tf = Instant::now();
            // full-precision weights of this layer (the next layer's shards
            // download in the background meanwhile)
            if let Some(h) = prefetch.take() {
                h.join().map_err(|_| anyhow::anyhow!("prefetch thread panicked"))??;
            }
            let mut shards = HashMap::new();
            for sh in layer_shards(li)? {
                let p = scratch.join(&sh);
                fetch(&agent, &format!("{base}/{sh}"), &p, token.as_deref(), true, 16)?;
                shards.insert(sh, open_safetensors(&p)?);
            }
            if li + 1 < nl {
                let next = layer_shards(li + 1)?;
                let (agent, base, token, scratch) = (agent.clone(), base.clone(), token.clone(), scratch.clone());
                prefetch = Some(std::thread::spawn(move || -> anyhow::Result<()> {
                    for sh in next {
                        fetch(&agent, &format!("{base}/{sh}"), &scratch.join(&sh), token.as_deref(), true, 16)?;
                    }
                    Ok(())
                }));
            }
            let find = |part: &str| {
                let (name, sh) = src_name(li, part)?;
                find_in(&shards, &name, &sh)
            };
            let t_fetch = tf.elapsed().as_secs_f64();
            // stage seconds summed over workers: inputs (gather, grams, down input), gate/up fold, q2tp sweeps + debias, down fold, q4tp sweep
            let mut tim = [0f64; 5];
            let te = &cmodel.tensors;

            if patch.is_none() {
                // routing: top-k of the logits per token
                let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
                let mut lists_ho: Vec<Vec<u32>> = vec![Vec::new(); ne];
                let mut lg = vec![0f32; ne];
                for t in 0..n {
                    for (e, v) in lg.iter_mut().enumerate() {
                        *v = cortiq_core::quant::f16_to_f32(lr[t * width + hidden + e]);
                    }
                    let mut ord: Vec<usize> = (0..ne).collect();
                    ord.select_nth_unstable_by(top_k - 1, |&a, &b| lg[b].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal));
                    for &e in &ord[..top_k] {
                        if held[t] { lists_ho[e].push(t as u32) } else { lists[e].push(t as u32) }
                    }
                }
                {
                    let mut cnt: Vec<usize> = lists.iter().map(Vec::len).collect();
                    cnt.sort_unstable();
                    eprintln!(
                        "  layer {li}: routed rows per expert min {} median {} max {}; {} experts under {} rows",
                        cnt[0],
                        cnt[ne / 2],
                        cnt[ne - 1],
                        cnt.iter().filter(|&&v| v < o.min_rows).count(),
                        o.min_rows
                    );
                }
                // the prior's sample: evenly spaced calibration tokens
                let cal: Vec<u32> = (0..n).filter(|&t| !held[t]).map(|t| t as u32).collect();
                let sample: Vec<u32> = cal.iter().step_by((cal.len() / prior_s.max(1)).max(1)).take(prior_s).copied().collect();
                let prod_prior = (prior_n0 > 0.0).then(|| Prior { rows: &sample, n0: prior_n0, n0_down: prior_n0 / 4.0 });
                let diag_layer = std::sync::Mutex::new(vec![(0usize, 0f64, vec![0f64; nvar + 1]); BINS.len()]);
                let (gst, gm) = find(ROUTED[0])?;
                let (dst, dm) = find(ROUTED[1])?;
                anyhow::ensure!(gm.shape == [ne, 2 * inter, hidden] && dm.shape == [ne, hidden, inter], "layer {li}: unexpected expert layout {:?} / {:?}", gm.shape, dm.shape);
                let gbytes = gst.bytes(gm);
                let dbytes = dst.bytes(dm);
                let esz = |shape: &[usize]| shape[1] * shape[2] * 2; // bf16
                let (gsz, dsz) = (esz(&gm.shape), esz(&dm.shape));
                // experts in parallel: each worker owns its device scratch and a
                // share of the host threads; results are pushed in expert order
                let workers = std::env::var("CMF_MOE_WORKERS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 16);
                let wthreads = (o.threads / workers).max(1);
                let next = std::sync::atomic::AtomicUsize::new(0);
                let results: Vec<std::sync::Mutex<Option<[Vec<u8>; 3]>>> = (0..ne).map(|_| std::sync::Mutex::new(None)).collect();
                let timers = std::sync::Mutex::new([0f64; 5]);
                let (n_gptq, n_rtn) = (std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0));
                let worker = || -> anyhow::Result<()> {
                    let mut wt = [0f64; 5];
                    let mut bufs: Option<wgpu_bufs::Bufs> = None;
                    loop {
                        let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if e >= ne {
                            break;
                        }
                        let gu = to_f32(&gm.dtype, &gbytes[e * gsz..(e + 1) * gsz])?;
                        let wd = to_f32(&dm.dtype, &dbytes[e * dsz..(e + 1) * dsz])?;
                        let (wg, wu) = gu.split_at(inter * hidden);
                        if let Some(vars) = &diag {
                            let (ho, ce) = (&lists_ho[e], lists[e].len());
                            if ho.is_empty() || ce < o.min_rows {
                                continue;
                            }
                            let m = ho.len();
                            let mut xh = vec![0f32; m * hidden];
                            par_rows(&mut xh, hidden, wthreads, |r, row| {
                                let t = ho[r] as usize;
                                for (d, &s) in row.iter_mut().zip(&lr[t * width..t * width + hidden]) {
                                    *d = cortiq_core::quant::f16_to_f32(s);
                                }
                            });
                            let o_fp = expert_out(&xh, m, hidden, wg, wu, &wd, inter, wthreads);
                            let base: f64 = o_fp.iter().map(|&v| (v as f64) * (v as f64)).sum();
                            let err_of = |enc: &[Vec<u8>; 3]| -> f64 {
                                let [qg, qu, qd] = dequant_expert(enc, inter, hidden);
                                let oq = expert_out(&xh, m, hidden, &qg, &qu, &qd, inter, wthreads);
                                oq.iter().zip(&o_fp).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum()
                            };
                            let rtn = [quantize_2d(Quant::Q2TiledP, wg, inter, hidden).1, quantize_2d(Quant::Q2TiledP, wu, inter, hidden).1, quantize_2d(Quant::Q4TiledP, &wd, hidden, inter).1];
                            let mut errs = vec![err_of(&rtn)];
                            let b = bufs.get_or_insert_with(|| wgpu_bufs::Bufs::new(c, CHUNK, hidden, inter));
                            for &n0 in vars {
                                let pr = (n0 > 0.0).then(|| Prior { rows: &sample, n0, n0_down: n0 / 4.0 });
                                let enc = encode_expert(c, b, lr, width, hidden, &lists[e], wg, wu, &wd, inter, o.lambda, wthreads, pr, &mut wt)?;
                                errs.push(err_of(&enc));
                            }
                            let bin = BINS.iter().position(|&lim| ce < lim).unwrap_or(BINS.len() - 1);
                            let mut d = diag_layer.lock().unwrap();
                            d[bin].0 += 1;
                            d[bin].1 += base;
                            for (a, v) in d[bin].2.iter_mut().zip(errs) {
                                *a += v;
                            }
                            n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                        let enc = if lists[e].len() < o.min_rows {
                            n_rtn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            [quantize_2d(Quant::Q2TiledP, wg, inter, hidden).1, quantize_2d(Quant::Q2TiledP, wu, inter, hidden).1, quantize_2d(Quant::Q4TiledP, &wd, hidden, inter).1]
                        } else {
                            n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let b = bufs.get_or_insert_with(|| wgpu_bufs::Bufs::new(c, CHUNK, hidden, inter));
                            encode_expert(c, b, lr, width, hidden, &lists[e], wg, wu, &wd, inter, o.lambda, wthreads, prod_prior, &mut wt)?
                        };
                        *results[e].lock().unwrap() = Some(enc);
                    }
                    let mut t = timers.lock().unwrap();
                    for (a, b) in t.iter_mut().zip(wt) {
                        *a += b;
                    }
                    Ok(())
                };
                std::thread::scope(|sc| -> anyhow::Result<()> {
                    let hs: Vec<_> = (0..workers).map(|_| sc.spawn(&worker)).collect();
                    for h in hs {
                        h.join().map_err(|_| anyhow::anyhow!("expert worker panicked"))??;
                    }
                    Ok(())
                })?;
                tim = *timers.lock().unwrap();
                stats.0 += n_gptq.into_inner();
                stats.1 += n_rtn.into_inner();
                if let Some(vars) = &diag {
                    let d = diag_layer.into_inner().unwrap();
                    let mut tot = diag_tot.lock().unwrap();
                    for (bi, (cnt, base, errs)) in d.iter().enumerate() {
                        if *cnt == 0 {
                            continue;
                        }
                        let rel: Vec<String> = errs.iter().map(|e| format!("{:.4}", e / base)).collect();
                        eprintln!("  DIAG layer {li} rows<{}: {cnt} experts, rel. output error RTN {} | priors {vars:?}: {}", if BINS[bi] == usize::MAX { "∞".into() } else { BINS[bi].to_string() }, rel[0], rel[1..].join(" "));
                        tot[bi].0 += cnt;
                        tot[bi].1 += base;
                        for (a, v) in tot[bi].2.iter_mut().zip(errs) {
                            *a += v;
                        }
                    }
                }
                for (e, slot) in results.into_iter().enumerate() {
                    if diag.is_some() {
                        break;
                    }
                    let [gate_b, up_b, down_b] = slot.into_inner().unwrap().ok_or_else(|| anyhow::anyhow!("expert {e} missing"))?;
                    let (gi, ui, di) = layer_dirs[li][e];
                    emit(&te[gi].name, TensorDtype::Q2TiledP, &te[gi].shape, gate_b)?;
                    emit(&te[ui].name, TensorDtype::Q2TiledP, &te[ui].shape, up_b)?;
                    emit(&te[di].name, TensorDtype::Q4TiledP, &te[di].shape, down_b)?;
                }
            }

            // the shared expert: every token is its input
            let mut t_shared = 0.0;
            if let (Some((gi, ui, di)), None) = (shared_dirs[li], &diag) {
                let ts = Instant::now();
                let is = te[gi].shape[0];
                let get = |part: &str, shape: [usize; 2]| -> anyhow::Result<Vec<f32>> {
                    let (st, m) = find(part)?;
                    anyhow::ensure!(m.shape == shape, "layer {li}: {part} has shape {:?}, want {shape:?}", m.shape);
                    to_f32(&m.dtype, st.bytes(m))
                };
                let wg = get(SHARED[0], [is, hidden])?;
                let wu = get(SHARED[1], [is, hidden])?;
                let wd = get(SHARED[2], [hidden, is])?;
                let all: Vec<u32> = (0..n).filter(|&t| !held[t]).map(|t| t as u32).collect();
                let mut st = [0f64; 5];
                let [gate_b, up_b, down_b] = encode_expert(c, &main_bufs, lr, width, hidden, &all, &wg, &wu, &wd, is, o.lambda, o.threads, None, &mut st)?;
                emit(&te[gi].name, TensorDtype::Q2TiledP, &te[gi].shape, gate_b)?;
                emit(&te[ui].name, TensorDtype::Q2TiledP, &te[ui].shape, up_b)?;
                emit(&te[di].name, TensorDtype::Q4TiledP, &te[di].shape, down_b)?;
                t_shared = ts.elapsed().as_secs_f64();
            }

            // shards this layer was the last user of
            let keep: Vec<String> = if li + 1 < nl { layer_shards(li + 1)? } else { Vec::new() };
            for sh in shards.keys() {
                if !keep.contains(sh) {
                    let _ = std::fs::remove_file(scratch.join(sh));
                }
            }
            eprintln!(
                "  layer {li}: {n} tokens, experts GPTQ {} / RTN {} so far, {:.0}s ({:.0}s total); fetch wait {t_fetch:.1}s, inputs {:.1} fold {:.1} q2tp {:.1} down fold {:.1} q4tp {:.1}, shared {t_shared:.1}s",
                stats.0,
                stats.1,
                tl.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64(),
                tim[0],
                tim[1],
                tim[2],
                tim[3],
                tim[4]
            );
            if std::env::var("CMF_MOE_STOP_AFTER").ok().and_then(|v| v.parse::<usize>().ok()) == Some(li + 1) {
                anyhow::bail!("CMF_MOE_STOP_AFTER={}: stopped before writing the file", li + 1);
            }
        }
    }
    drop(emit);
    if let Some(vars) = &diag {
        eprintln!("DIAG total, layers {range_lo}..{range_hi}, relative expert output error on held-out rows:");
        for (bi, (cnt, base, errs)) in diag_tot.into_inner().unwrap().iter().enumerate() {
            if *cnt == 0 {
                continue;
            }
            let rel: Vec<String> = errs.iter().map(|e| format!("{:.4}", e / base)).collect();
            eprintln!("  rows<{}: {cnt} experts | RTN {} | GPTQ priors {vars:?}: {}", if BINS[bi] == usize::MAX { "∞".into() } else { BINS[bi].to_string() }, rel[0], rel[1..].join(" "));
        }
        return Ok(());
    }
    if let Some(path) = patch {
        drop(target);
        let n = patches.len();
        CmfModel::recode_entries_in_place(path, &patches).map_err(|e| anyhow::anyhow!("patch {path}: {e}"))?;
        eprintln!("patched {n} shared-expert tensors in place in {path} in {:.0}s", t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let mut out = out.expect("stream writer in full mode");
    for (slot, e) in cmodel.tensors.iter().enumerate() {
        if !expert_slots.contains(&slot) {
            out.push_bounded(&e.name, e.dtype, &e.shape, cmodel.entry_bytes(e), 1 << 28)?;
        }
    }
    let mut header = cmodel.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    prov["weight_quant"] = serde_json::json!("q2tp");
    prov["gptq_moe"] = serde_json::json!({
        "experts": "routed and shared: gate/up q2tp (placed, debiased), down q4tp",
        "calibration_tokens": ntok,
        "window": o.window,
        "lambda": o.lambda,
        "source": o.hf,
        "calibration_model": std::path::Path::new(o.calib_model).file_name().map(|f| f.to_string_lossy().into_owned()),
        "experts_gptq": stats.0,
        "experts_rtn": stats.1,
        "min_rows": o.min_rows,
    });
    header.provenance = Some(prov);
    out.finish(&header, None, cmodel.vocab.as_deref())?;
    eprintln!("wrote {} in {:.0}s", o.output, t0.elapsed().as_secs_f64());
    Ok(())
}

/// GPTQ of one expert from its input rows (`rows` index the captured
/// `width`-wide f16 rows `lr`, input first): both input Hessians accumulate
/// over `CHUNK`-row device passes, then gate/up fold + q2tp sweeps + debias,
/// down fold + q4tp sweep. Returns the gate, up and down payloads.
/// Layer-wide prior of the expert Hessians: `rows` (a sample of the layer's
/// tokens) enter both input Hessians as `n0` / `n0_down` pseudo-rows, so
/// directions an expert's own rows never visit carry the layer's statistics
/// instead of nothing (an expert routed fewer tokens than its input width
/// has a rank-deficient Hessian of its own).
#[derive(Clone, Copy)]
struct Prior<'a> {
    rows: &'a [u32],
    n0: f64,
    n0_down: f64,
}

#[allow(clippy::too_many_arguments)]
fn encode_expert(
    c: &Ctx,
    bufs: &wgpu_bufs::Bufs,
    lr: &[u16],
    width: usize,
    hidden: usize,
    rows: &[u32],
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    inter: usize,
    lambda: f64,
    threads: usize,
    prior: Option<Prior>,
    tim: &mut [f64; 5],
) -> anyhow::Result<[Vec<u8>; 3]> {
    use cortiq_engine::gpu;
    let ts = Instant::now();
    let hid = gpu::gram_alloc(hidden).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
    let did = gpu::gram_alloc(inter).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
    let (pg, pu) = (tm::f16_from_f32(c, wg), tm::f16_from_f32(c, wu));
    let gather = |xs: &mut [f32], part: &[u32], scale: f32| {
        par_rows(xs, hidden, threads, |r, row| {
            let t = part[r] as usize;
            for (d, &s) in row.iter_mut().zip(&lr[t * width..t * width + hidden]) {
                *d = cortiq_core::quant::f16_to_f32(s) * scale;
            }
        });
    };
    let mut xe = vec![0f32; CHUNK.min(rows.len().max(prior.map_or(0, |p| p.rows.len()))) * hidden];
    for part in rows.chunks(CHUNK) {
        let m = part.len();
        let xs = &mut xe[..m * hidden];
        gather(xs, part, 1.0);
        to::write(c, &bufs.x, xs);
        anyhow::ensure!(gpu::gram_add_dev(hid, &bufs.x, m), "gram");
        tm::plane_linear(c, &pg, &bufs.x, &bufs.act16, &bufs.yg, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
        tm::plane_linear(c, &pu, &bufs.x, &bufs.act16, &bufs.yu, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
        to::run(c, &[to::elt(c, "swiglu_fwd", m * inter, &bufs.yg, &bufs.yu, Some(&bufs.yd), None, None, None).ok_or_else(|| anyhow::anyhow!("swiglu"))?]);
        anyhow::ensure!(gpu::gram_add_dev(did, &bufs.yd, m), "gram");
    }
    if let Some(pr) = prior.filter(|p| !p.rows.is_empty() && p.n0 > 0.0) {
        // S sample rows as n0 pseudo-rows: gram of a·x with a = √(n0/S); the
        // down input of the sample through √(n0_down/S)·W_u (silu(g)⊙(b·u) = b·y)
        let s = pr.rows.len() as f64;
        let (a, b) = ((pr.n0 / s).sqrt() as f32, (pr.n0_down / s).sqrt() as f32);
        let wub: Vec<f32> = wu.iter().map(|&v| v * b).collect();
        let pub_ = tm::f16_from_f32(c, &wub);
        drop(wub);
        for part in pr.rows.chunks(CHUNK) {
            let m = part.len();
            let xs = &mut xe[..m * hidden];
            gather(xs, part, a);
            to::write(c, &bufs.x, xs);
            anyhow::ensure!(gpu::gram_add_dev(hid, &bufs.x, m), "gram");
            gather(xs, part, 1.0);
            to::write(c, &bufs.x, xs);
            tm::plane_linear(c, &pg, &bufs.x, &bufs.act16, &bufs.yg, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
            tm::plane_linear(c, &pub_, &bufs.x, &bufs.act16, &bufs.yu, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
            to::run(c, &[to::elt(c, "swiglu_fwd", m * inter, &bufs.yg, &bufs.yu, Some(&bufs.yd), None, None, None).ok_or_else(|| anyhow::anyhow!("swiglu"))?]);
            anyhow::ensure!(gpu::gram_add_dev(did, &bufs.yd, m), "gram");
        }
    }
    drop(xe);
    let hgu = gpu::gram_read(hid).ok_or_else(|| anyhow::anyhow!("gram read"))?;
    tim[0] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let fid = gpu::gptq_fold_dev(hid, hidden, lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
    let (dead, hdiag, _) = gpu::gptq_fold_info(fid).unwrap_or_default();
    tim[1] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let mut gb = gptq_quantize_q2tp_dev(wg, inter, hidden, fid, &dead, &hdiag, threads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
    let mut ub = gptq_quantize_q2tp_dev(wu, inter, hidden, fid, &dead, &hdiag, threads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
    gpu::gptq_fold_free(fid);
    // debias: H·w for every gate/up row from two GEMMs W·H
    let h32: Vec<f32> = hgu.iter().map(|&v| v as f32).collect();
    drop(hgu);
    let hwg = gemm_rows_square(wg, inter, hidden, &h32, threads);
    q2tp_debias_rows(&mut gb, wg, inter, hidden, &hwg, threads);
    drop(hwg);
    let hwu = gemm_rows_square(wu, inter, hidden, &h32, threads);
    q2tp_debias_rows(&mut ub, wu, inter, hidden, &hwu, threads);
    tim[2] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let fd = gpu::gptq_fold_dev(did, inter, lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
    let (dd, _, _) = gpu::gptq_fold_info(fd).unwrap_or_default();
    tim[3] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let db = gptq_quantize_q4tp_dev(wd, hidden, inter, fd, &dd).ok_or_else(|| anyhow::anyhow!("q4tp sweep"))?;
    gpu::gptq_fold_free(fd);
    tim[4] += ts.elapsed().as_secs_f64();
    Ok([gb, ub, db])
}

fn find_in<'a>(
    shards: &'a HashMap<String, crate::convert::SafeTensors>,
    name: &str,
    sh: &str,
) -> anyhow::Result<(&'a crate::convert::SafeTensors, &'a crate::convert::TensorMeta)> {
    let st = shards.get(sh).ok_or_else(|| anyhow::anyhow!("shard {sh} not open"))?;
    st.tensors.iter().find(|m| m.name == name).map(|m| (st, m)).ok_or_else(|| anyhow::anyhow!("{name} not in {sh}"))
}

/// `A·Wᵀ` for row-major `a` (`m × k`) and `w` (`n × k`), rows of `a` split
/// across `threads`.
fn gemm_nt(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    if m == 0 {
        return out;
    }
    let per = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * n).enumerate() {
            let rows = chunk.len() / n;
            let src = &a[ci * per * k..(ci * per + rows) * k];
            s.spawn(move || cortiq_engine::qtensor::sgemm_public(rows, n, k, 1.0, src, k, w, k, true, chunk, n));
        }
    });
    out
}

/// Expert output `swiglu(X·W_gᵀ, X·W_uᵀ)·W_dᵀ` of the rows `x` (`m × hidden`).
fn expert_out(x: &[f32], m: usize, hidden: usize, wg: &[f32], wu: &[f32], wd: &[f32], inter: usize, threads: usize) -> Vec<f32> {
    let g = gemm_nt(x, m, hidden, wg, inter, threads);
    let u = gemm_nt(x, m, hidden, wu, inter, threads);
    let y: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
    gemm_nt(&y, m, inter, wd, hidden, threads)
}

/// Dequantized gate, up, down of an encoded expert.
fn dequant_expert(enc: &[Vec<u8>; 3], inter: usize, hidden: usize) -> [Vec<f32>; 3] {
    let (mut g, mut u, mut d) = (vec![0f32; inter * hidden], vec![0f32; inter * hidden], vec![0f32; hidden * inter]);
    cortiq_core::quant::dequant_q2tp(&enc[0], inter, hidden, &mut g);
    cortiq_core::quant::dequant_q2tp(&enc[1], inter, hidden, &mut u);
    cortiq_core::quant::dequant_q4tp(&enc[2], hidden, inter, &mut d);
    [g, u, d]
}

/// `f(row, row_slice)` over the `width`-wide rows of `v`, split across `threads`.
fn par_rows(v: &mut [f32], width: usize, threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = v.len() / width;
    let per = rows.div_ceil(threads.max(1)).max(1);
    let f = &f;
    std::thread::scope(|s| {
        for (ci, chunk) in v.chunks_mut(per * width).enumerate() {
            s.spawn(move || {
                for (k, row) in chunk.chunks_exact_mut(width).enumerate() {
                    f(ci * per + k, row);
                }
            });
        }
    });
}

/// Device scratch of the expert pass (`CHUNK` rows).
mod wgpu_bufs {
    use cortiq_engine::gpu_wgpu::{train_mm as tm, train_ops as to};
    pub struct Bufs {
        pub x: cortiq_engine::wgpu::Buffer,
        pub act16: cortiq_engine::wgpu::Buffer,
        pub yg: cortiq_engine::wgpu::Buffer,
        pub yu: cortiq_engine::wgpu::Buffer,
        pub yd: cortiq_engine::wgpu::Buffer,
    }
    impl Bufs {
        pub fn new(c: &cortiq_engine::gpu_wgpu::Ctx, rows: usize, hidden: usize, inter: usize) -> Self {
            let rows = tm::pad_rows(rows);
            Bufs {
                x: to::zeros(c, rows * hidden),
                act16: tm::f16_buf(c, rows * hidden),
                yg: to::zeros(c, rows * inter),
                yu: to::zeros(c, rows * inter),
                yd: to::zeros(c, rows * inter),
            }
        }
    }
}
