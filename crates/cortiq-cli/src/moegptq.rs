//! `cortiq quantize-moe`: GPTQ for the routed experts of a Flash-Next-class
//! MoE (`qwen4_exp`): gate/up into q2tp, down into q4tp, every other tensor
//! copied from the calibration file.
//!
//! * Calibration: the calibration model (a q4tp file of the same model) runs
//!   its ordinary device frames over the corpus; each layer frame in the
//!   captured range copies its MoE input rows and router logits out right
//!   after routing (`Pipeline::qwen4_capture`). A token's experts are the
//!   top-k of its logits.
//! * Per expert, on the GPU: the input Hessian of gate/up from the rows
//!   routed to it, the device fold, the q2tp sweep (placed rule, debiased);
//!   the down input `silu(X·W_gᵀ)⊙(X·W_uᵀ)` from the full-precision weights
//!   (f16 planes, matrix units), its Hessian, the fold and the q4tp sweep.
//! * Full-precision weights come from the original safetensors shards,
//!   fetched one layer at a time and deleted when done; the output is
//!   streamed, so neither the bf16 checkpoint nor the result is ever held
//!   whole.

use crate::convert::{Quant, fetch, open_safetensors, quantize_2d, to_f32};
use crate::gptq::{CalibSource, calib_source_tokens, gemm_rows_square, gptq_quantize_q2tp_dev, gptq_quantize_q4tp_dev, q2tp_debias_rows, split_calib_budget};
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::quant::f32_to_f16;
use cortiq_core::types::TensorDtype;
use cortiq_engine::gpu_wgpu::{train_mm as tm, train_ops as to};
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
}

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
    eprintln!("quantize-moe: {nl} layers, {ne} experts, top-{top_k}, hidden {hidden}, expert width {inter}");

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
    let src_name = |li: usize, part: &str| -> anyhow::Result<(String, String)> {
        for prefix in ["model.language_model.layers", "model.layers"] {
            let n = format!("{prefix}.{li}.mlp.experts.{part}");
            if let Some(s) = wmap.get(&n).and_then(|v| v.as_str()) {
                return Ok((n, s.to_string()));
            }
        }
        anyhow::bail!("layer {li}: no {part} in the source index")
    };

    // output: the experts layer by layer (gate, up, down of an expert
    // adjacent, so the expert store reads each in one request), then every
    // other tensor straight from the calibration file
    let mut expert_slots: HashSet<usize> = HashSet::new();
    let mut layer_dirs = Vec::with_capacity(nl);
    for li in 0..nl {
        let (d, _) = pipe.qwen4_expert_dirs(li).ok_or_else(|| anyhow::anyhow!("layer {li}: no experts"))?;
        for &(g, u, dn) in &d {
            expert_slots.extend([g, u, dn]);
        }
        layer_dirs.push(d);
    }
    let avg_name = cmodel.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cmodel.tensors.len().max(1);
    let mut out = CmfStreamWriter::new(o.output, CmfStreamWriter::head_reserve_for(cmodel.tensors.len(), avg_name))?;

    let width = hidden + ne;
    let mut stats = (0usize, 0usize);
    let mut prefetch: Option<std::thread::JoinHandle<anyhow::Result<()>>> = None;
    let pass = o.layers_per_pass.max(1);
    for lo in (0..nl).step_by(pass) {
        let hi = (lo + pass).min(nl);
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
            // routing: top-k of the logits per token
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
            let mut lg = vec![0f32; ne];
            for t in 0..n {
                for (e, v) in lg.iter_mut().enumerate() {
                    *v = cortiq_core::quant::f16_to_f32(lr[t * width + hidden + e]);
                }
                let mut ord: Vec<usize> = (0..ne).collect();
                ord.select_nth_unstable_by(top_k - 1, |&a, &b| lg[b].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal));
                for &e in &ord[..top_k] {
                    lists[e].push(t as u32);
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
            let tf = Instant::now();
            // full-precision weights of this layer (the next layer's shards
            // download in the background meanwhile)
            if let Some(h) = prefetch.take() {
                h.join().map_err(|_| anyhow::anyhow!("prefetch thread panicked"))??;
            }
            let (gu_name, gu_shard) = src_name(li, "gate_up_proj")?;
            let (dn_name, dn_shard) = src_name(li, "down_proj")?;
            let mut shards = HashMap::new();
            for sh in [&gu_shard, &dn_shard] {
                if !shards.contains_key(sh) {
                    let p = scratch.join(sh);
                    fetch(&agent, &format!("{base}/{sh}"), &p, token.as_deref(), true, 16)?;
                    shards.insert(sh.clone(), open_safetensors(&p)?);
                }
            }
            if li + 1 < nl {
                let next = [src_name(li + 1, "gate_up_proj")?.1, src_name(li + 1, "down_proj")?.1];
                let (agent, base, token, scratch) = (agent.clone(), base.clone(), token.clone(), scratch.clone());
                prefetch = Some(std::thread::spawn(move || -> anyhow::Result<()> {
                    for sh in next {
                        fetch(&agent, &format!("{base}/{sh}"), &scratch.join(&sh), token.as_deref(), true, 16)?;
                    }
                    Ok(())
                }));
            }
            let find = |sh: &str, name: &str| {
                let st = &shards[sh];
                st.tensors.iter().find(|m| m.name == name).map(|m| (st, m)).ok_or_else(|| anyhow::anyhow!("{name} not in {sh}"))
            };
            let t_fetch = tf.elapsed().as_secs_f64();
            // stage seconds summed over workers: gram+read, fold, q2tp sweeps+debias, down input, down fold+sweep
            let (gst, gm) = find(&gu_shard, &gu_name)?;
            let (dst, dm) = find(&dn_shard, &dn_name)?;
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
                let mut tim = [0f64; 5];
                let mut scratch_dev: Option<(usize, wgpu_bufs::Bufs)> = None;
                loop {
                    let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if e >= ne {
                        break;
                    }
                    let gu = to_f32(&gm.dtype, &gbytes[e * gsz..(e + 1) * gsz])?;
                    let wd = to_f32(&dm.dtype, &dbytes[e * dsz..(e + 1) * dsz])?;
                    let (wg, wu) = gu.split_at(inter * hidden);
                    let ce = lists[e].len();
                    let enc = if ce < o.min_rows {
                        n_rtn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        [quantize_2d(Quant::Q2TiledP, wg, inter, hidden).1, quantize_2d(Quant::Q2TiledP, wu, inter, hidden).1, quantize_2d(Quant::Q4TiledP, &wd, hidden, inter).1]
                    } else {
                        n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let pr = tm::pad_rows(ce);
                        if scratch_dev.as_ref().is_none_or(|(cap, _)| *cap < pr) {
                            // grow to the layer's busiest expert at once
                            let want = tm::pad_rows(cmax_of(&lists).max(ce));
                            scratch_dev = Some((want, wgpu_bufs::Bufs::new(c, want, hidden, inter)));
                        }
                        let bufs = &scratch_dev.as_ref().unwrap().1;
                        let ts = Instant::now();
                        let mut xe = vec![0f32; ce * hidden];
                        xe.par_chunks_rows(hidden, wthreads, |r, row| {
                            let t = lists[e][r] as usize;
                            for (d, &s) in row.iter_mut().zip(&lr[t * width..t * width + hidden]) {
                                *d = cortiq_core::quant::f16_to_f32(s);
                            }
                        });
                        to::write(c, &bufs.x, &xe);
                        drop(xe);
                        // gate/up: Hessian of the routed rows, fold, q2tp sweeps
                        let hid = cortiq_engine::gpu::gram_alloc(hidden).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
                        anyhow::ensure!(cortiq_engine::gpu::gram_add_dev(hid, &bufs.x, ce), "gram");
                        let hgu = cortiq_engine::gpu::gram_read(hid).ok_or_else(|| anyhow::anyhow!("gram read"))?;
                        tim[0] += ts.elapsed().as_secs_f64();
                        let ts = Instant::now();
                        let fid = cortiq_engine::gpu::gptq_fold_dev(hid, hidden, o.lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
                        let (dead, hdiag, _) = cortiq_engine::gpu::gptq_fold_info(fid).unwrap_or_default();
                        tim[1] += ts.elapsed().as_secs_f64();
                        let ts = Instant::now();
                        let mut gb = gptq_quantize_q2tp_dev(wg, inter, hidden, fid, &dead, &hdiag, wthreads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
                        let mut ub = gptq_quantize_q2tp_dev(wu, inter, hidden, fid, &dead, &hdiag, wthreads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
                        cortiq_engine::gpu::gptq_fold_free(fid);
                        // debias: H·w for every gate/up row in one GEMM [W_g; W_u]·H
                        let h32: Vec<f32> = hgu.iter().map(|&v| v as f32).collect();
                        drop(hgu);
                        let hw = gemm_rows_square(&gu, 2 * inter, hidden, &h32, wthreads);
                        q2tp_debias_rows(&mut gb, wg, inter, hidden, &hw[..inter * hidden], wthreads);
                        q2tp_debias_rows(&mut ub, wu, inter, hidden, &hw[inter * hidden..], wthreads);
                        tim[2] += ts.elapsed().as_secs_f64();
                        let ts = Instant::now();
                        // down: input silu(X·W_gᵀ)⊙(X·W_uᵀ) from the full-precision planes
                        let (pg, pu) = (tm::f16_from_f32(c, wg), tm::f16_from_f32(c, wu));
                        tm::plane_linear(c, &pg, &bufs.x, &bufs.act16, &bufs.yg, ce, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
                        tm::plane_linear(c, &pu, &bufs.x, &bufs.act16, &bufs.yu, ce, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
                        to::run(c, &[to::elt(c, "swiglu_fwd", ce * inter, &bufs.yg, &bufs.yu, Some(&bufs.yd), None, None, None).ok_or_else(|| anyhow::anyhow!("swiglu"))?]);
                        tim[3] += ts.elapsed().as_secs_f64();
                        let ts = Instant::now();
                        let did = cortiq_engine::gpu::gram_alloc(inter).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
                        anyhow::ensure!(cortiq_engine::gpu::gram_add_dev(did, &bufs.yd, ce), "gram");
                        let fd = cortiq_engine::gpu::gptq_fold_dev(did, inter, o.lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
                        let (dd, _, _) = cortiq_engine::gpu::gptq_fold_info(fd).unwrap_or_default();
                        let db = gptq_quantize_q4tp_dev(&wd, hidden, inter, fd, &dd).ok_or_else(|| anyhow::anyhow!("q4tp sweep"))?;
                        cortiq_engine::gpu::gptq_fold_free(fd);
                        tim[4] += ts.elapsed().as_secs_f64();
                        [gb, ub, db]
                    };
                    *results[e].lock().unwrap() = Some(enc);
                }
                let mut t = timers.lock().unwrap();
                for (a, b) in t.iter_mut().zip(tim) {
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
            let tim = *timers.lock().unwrap();
            stats.0 += n_gptq.into_inner();
            stats.1 += n_rtn.into_inner();
            let te = &cmodel.tensors;
            for (e, slot) in results.into_iter().enumerate() {
                let [gate_b, up_b, down_b] = slot.into_inner().unwrap().ok_or_else(|| anyhow::anyhow!("expert {e} missing"))?;
                let (gi, ui, di) = layer_dirs[li][e];
                out.push(&te[gi].name, TensorDtype::Q2TiledP, &te[gi].shape, &gate_b)?;
                out.push(&te[ui].name, TensorDtype::Q2TiledP, &te[ui].shape, &up_b)?;
                out.push(&te[di].name, TensorDtype::Q4TiledP, &te[di].shape, &down_b)?;
            }
            // shards this layer was the last user of
            for sh in shards.keys() {
                let next_needs = li + 1 < nl && [src_name(li + 1, "gate_up_proj")?.1, src_name(li + 1, "down_proj")?.1].contains(sh);
                if !next_needs {
                    let _ = std::fs::remove_file(scratch.join(sh));
                }
            }
            eprintln!(
                "  layer {li}: {n} tokens, experts GPTQ {} / RTN {} so far, {:.0}s ({:.0}s total); fetch wait {t_fetch:.1}s, gram {:.1} fold {:.1} q2tp {:.1} down-in {:.1} down {:.1}",
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
        "experts": "gate/up q2tp (placed, debiased), down q4tp",
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

fn cmax_of(lists: &[Vec<u32>]) -> usize {
    lists.iter().map(Vec::len).max().unwrap_or(0).max(1)
}

trait ParRows {
    fn par_chunks_rows(&mut self, width: usize, threads: usize, f: impl Fn(usize, &mut [f32]) + Sync);
}

impl ParRows for Vec<f32> {
    /// `f(row, row_slice)` over the `width`-wide rows, split across `threads`.
    fn par_chunks_rows(&mut self, width: usize, threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
        let rows = self.len() / width;
        let per = rows.div_ceil(threads.max(1)).max(1);
        let f = &f;
        std::thread::scope(|s| {
            for (ci, chunk) in self.chunks_mut(per * width).enumerate() {
                s.spawn(move || {
                    for (k, row) in chunk.chunks_exact_mut(width).enumerate() {
                        f(ci * per + k, row);
                    }
                });
            }
        });
    }
}

/// Per-worker device scratch of the expert pass.
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
