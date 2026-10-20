//! `cortiq nvg-bench` — one tensor through the NVG encoder with a per-stage
//! profile, on the CPU reference path and/or the device path, from the
//! same calibration Grams (captured once, cached as f32 files).
//!
//! The Grams of the tensor's input are captured the way the converter's
//! first layer sees them: the model's own forward (layers `0..=l`) over the
//! FIT and SEL token windows of the calibration text. Every later run on the
//! same `--gram-cache` reads them back, so CPU and device encodes compare on
//! identical inputs.

use crate::nvg::{self, HCtx, PathsChoice, Prof};
use crate::nvg_linalg::Mat;
use cortiq_core::format::CmfModel;
use cortiq_engine::pool::Pool;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Instant;

pub struct BenchOpts<'a> {
    pub model: &'a str,
    pub tensor: Option<&'a str>,
    pub calib: Option<&'a str>,
    pub fit_tokens: usize,
    pub sel_tokens: usize,
    pub bpw: f64,
    pub kv_factor: f64,
    pub paths: PathsChoice,
    pub device: &'a str,
    pub threads: usize,
    pub rho: f64,
    pub gram_cache: Option<&'a str>,
    /// Encode every body linear of this layer (device: `jobs` in flight)
    pub layer: Option<usize>,
    pub jobs: usize,
}

const MAGIC: u32 = 0x4E56_4747; // "NVGG"

fn save_gram(path: &str, h: &Mat, count: usize) -> anyhow::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&(h.r as u32).to_le_bytes())?;
    f.write_all(&(count as u64).to_le_bytes())?;
    let v: Vec<f32> = h.d.iter().map(|&x| x as f32).collect();
    f.write_all(bytemuck_f32(&v))?;
    Ok(())
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is reinterpreted as its bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn load_gram(path: &str) -> anyhow::Result<(Mat, usize)> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    anyhow::ensure!(u32::from_le_bytes(b4) == MAGIC, "{path}: not an nvg-bench Gram");
    f.read_exact(&mut b4)?;
    let n = u32::from_le_bytes(b4) as usize;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8)?;
    let count = u64::from_le_bytes(b8) as usize;
    let mut raw = vec![0u8; n * n * 4];
    f.read_exact(&mut raw)?;
    let d: Vec<f64> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    Ok((Mat::from_vec(n, n, d), count))
}

/// FIT and SEL mean Grams of `lead`'s input (layer `l`).
fn capture(model: &Arc<CmfModel>, o: &BenchOpts, l: usize, lead: &str) -> anyhow::Result<(Mat, Mat)> {
    let calib = o.calib.ok_or_else(|| anyhow::anyhow!("no --gram-cache file yet: --calib FILE is needed"))?;
    let text = std::fs::read_to_string(calib)?;
    // NVG_BENCH_TRUNC=<path>: capture from a copy cut after layer l, the
    // converter's calibration work file (a check of the cut).
    let cut;
    let model = match std::env::var("NVG_BENCH_TRUNC") {
        Ok(path) => {
            let th = nvg::truncated_header(&model.header, l).ok_or_else(|| anyhow::anyhow!("architecture not cut"))?;
            let specs: Vec<cortiq_core::format::TensorSpec> = model
                .tensors
                .iter()
                .filter(|e| nvg::keep_in_truncated(&e.name, l))
                .map(|e| cortiq_core::format::TensorSpec { name: e.name.clone(), dtype: e.dtype, shape: e.shape.clone(), data: model.entry_bytes(e).to_vec() })
                .collect();
            CmfModel::write(&path, &th, &specs, None, model.vocab.as_deref()).map_err(|e| anyhow::anyhow!("{e}"))?;
            cut = Arc::new(CmfModel::open(&path)?);
            eprintln!("nvg-bench: capturing from {path} ({} layers)", th.arch.num_layers);
            &cut
        }
        Err(_) => model,
    };
    let mut pipe = cortiq_engine::Pipeline::from_model(model, cortiq_engine::SamplerConfig::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    let ids = pipe.tokenizer.encode(&text);
    let need = o.fit_tokens + o.sel_tokens;
    anyhow::ensure!(ids.len() >= need, "calibration text has {} tokens, need {need}", ids.len());
    let win = |a: usize, b: usize| -> Vec<Vec<u32>> {
        ids[a..b].chunks(512).filter(|c| c.len() >= 2).map(<[u32]>::to_vec).collect()
    };
    let mut out = Vec::new();
    for ws in [win(0, o.fit_tokens), win(o.fit_tokens, need)] {
        let name = lead.to_string();
        cortiq_engine::gptq_capture::begin_filtered(true, move |n| n == name);
        let t = Instant::now();
        for w in &ws {
            // The converter's host path calibrates with scored forwards
            // (ppl_ids); the batched calib_forward hides the capture hook.
            pipe.ppl_ids(w).map_err(|e| anyhow::anyhow!(e))?;
        }
        let mut hs = cortiq_engine::gptq_capture::end();
        let acc = hs
            .remove(lead)
            .ok_or_else(|| anyhow::anyhow!("no capture for {lead} (captured {:?})", hs.keys().collect::<Vec<_>>()))?;
        eprintln!(
            "nvg-bench: captured {lead} ({} cols) over {} tokens in {:.1}s",
            acc.cols,
            acc.count,
            t.elapsed().as_secs_f64()
        );
        out.push((nvg::mean_gram(&acc), acc.count));
    }
    if let Some(p) = o.gram_cache {
        save_gram(&format!("{p}.fit.gram"), &out[0].0, out[0].1)?;
        save_gram(&format!("{p}.sel.gram"), &out[1].0, out[1].1)?;
    }
    let sel = out.pop().unwrap().0;
    let fit = out.pop().unwrap().0;
    Ok((fit, sel))
}

pub fn run(o: &BenchOpts) -> anyhow::Result<()> {
    // As quantize-gptq: the per-op device GEMMs keep every projection input
    // visible to the capture hook (the resident token graph would hide it),
    // and the Grams accumulate on the card when one is up.
    let gpu_calib = !cfg!(target_os = "macos")
        && std::env::var("CMF_GPU").as_deref() != Ok("0")
        && cortiq_engine::gpu::backend_available();
    if gpu_calib && std::env::var("CMF_GPU_WGPU_GRAPH").is_err() {
        // SAFETY: single-threaded, before any pipeline exists.
        unsafe { std::env::set_var("CMF_GPU_WGPU_GRAPH", "0") };
    }
    cortiq_engine::gptq_capture::set_device(gpu_calib);
    let model = Arc::new(CmfModel::open_sharded(o.model)?);
    if let (None, Some(l)) = (o.tensor, o.layer) {
        return run_layer(&model, o, l);
    }
    let Some(tname) = o.tensor else {
        for e in &model.tensors {
            if nvg::body_linear(&e.name).is_some() {
                println!("{} {:?} {:?}", e.name, e.shape, e.dtype);
            }
        }
        return Ok(());
    };
    let slot = model
        .tensors
        .iter()
        .position(|e| e.name == tname)
        .ok_or_else(|| anyhow::anyhow!("no tensor {tname}"))?;
    let (l, short) = nvg::body_linear(tname).ok_or_else(|| anyhow::anyhow!("{tname} is not an encoder body linear"))?;
    let lead = nvg::lead_name(l, nvg::input_lead(short));
    let cached = o
        .gram_cache
        .filter(|p| std::path::Path::new(&format!("{p}.fit.gram")).exists());
    let (hfit, hsel) = match cached {
        Some(p) => {
            let (a, ca) = load_gram(&format!("{p}.fit.gram"))?;
            let (b, cb) = load_gram(&format!("{p}.sel.gram"))?;
            eprintln!("nvg-bench: Grams from {p}.{{fit,sel}}.gram ({ca} + {cb} tokens)");
            (a, b)
        }
        None => capture(&model, o, l, &lead)?,
    };
    let mut w = if let Ok(p) = std::env::var("NVG_BENCH_W_FROM_RAW") {
        // Transport target M (row-major f32, d_out x d_in) from a raw file:
        // the cross-covariance corrected target W*C*G^-1.
        let (r, c) = (model.tensors[slot].shape[0], model.tensors[slot].shape[1]);
        let raw = std::fs::read(&p)?;
        anyhow::ensure!(raw.len() == r * c * 4, "{p}: {} bytes for {r}x{c} f32", raw.len());
        let d: Vec<f64> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        eprintln!("nvg-bench: transport target M from {p}");
        Mat::from_vec(r, c, d)
    } else if let Ok(p) = std::env::var("NVG_BENCH_W_FROM") {
        // The waves come from `--model` (e.g. a masked/student file), the
        // encoded target from another file (the true weights).
        let wsrc = Arc::new(CmfModel::open_sharded(&p)?);
        let wslot = wsrc.tensor_index(tname).ok_or_else(|| anyhow::anyhow!("{p}: no tensor {tname}"))?;
        eprintln!("nvg-bench: target W from {p}");
        nvg::load_f64(&wsrc, wslot)?
    } else {
        nvg::load_f64(&model, slot)?
    };
    anyhow::ensure!(w.c == hfit.r, "{tname}: {} input columns vs a {}-wide Gram", w.c, hfit.r);
    let gqa = model.arch().num_kv_heads < model.arch().num_attention_heads;
    let mult = if gqa && (short == "k" || short == "v") { o.kv_factor } else { 1.0 };
    eprintln!(
        "nvg-bench: {tname} {}x{} at {} bpw (P2 r{}, P1 r{}), rho {}",
        w.r,
        w.c,
        o.bpw,
        cortiq_core::blr::blr_rank(w.r, w.c, o.bpw, 2, mult),
        cortiq_core::blr::blr_rank(w.r, w.c, o.bpw, 1, mult),
        o.rho
    );
    // Research gauge (A10): scale each head block of the target matrix by
    // sqrt(w_j), w_j = median(head output energy)/head output energy, so the
    // fit minimizes the per-head *relative* error in front of the per-head
    // normalisers. The gauge is undone exactly after packing by dividing the
    // stored per-row h scales by the same factor — no extra bits, no runtime
    // change. NVG_BENCH_ROWWEIGHT_SHUFFLE=<seed> permutes the weights across
    // the heads (same multiset) as the placebo control.
    let mut hscale: Option<Vec<f64>> = None;
    if let Ok(mode) = std::env::var("NVG_BENCH_ROWWEIGHT") {
        anyhow::ensure!(mode == "invE", "NVG_BENCH_ROWWEIGHT={mode}: expected invE");
        let blocks = row_head_blocks(model.arch(), short, w.r).ok_or_else(|| {
            anyhow::anyhow!("NVG_BENCH_ROWWEIGHT: {tname} ({short}) has no per-head row structure")
        })?;
        let tpool = Pool::new(o.threads.saturating_sub(1).max(1));
        let g = crate::nvg_linalg::gemm(Some(&tpool), &w, false, &hfit, false);
        let energy: Vec<f64> = blocks
            .iter()
            .map(|&(s, t)| (s..t).map(|i| (0..w.c).map(|c| g.at(i, c) * w.at(i, c)).sum::<f64>()).sum())
            .collect();
        let mut sorted = energy.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let eref = sorted[sorted.len() / 2];
        let mut wj: Vec<f64> = energy.iter().map(|&ej| (eref / ej.max(1e-300)).clamp(1.0 / 32.0, 32.0)).collect();
        let shuffled = if let Ok(seed) = std::env::var("NVG_BENCH_ROWWEIGHT_SHUFFLE") {
            let mut s: u64 = seed.parse().unwrap_or(1) | 1;
            for i in (1..wj.len()).rev() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = ((s >> 33) as usize) % (i + 1);
                wj.swap(i, j);
            }
            " (shuffled)"
        } else {
            ""
        };
        eprintln!(
            "nvg-bench: rowweight{shuffled} over {} heads: head energy {:.4e}..{:.4e} (median {:.4e}), scale {:.3}..{:.3}",
            blocks.len(),
            sorted[0],
            sorted[sorted.len() - 1],
            eref,
            wj.iter().cloned().fold(f64::INFINITY, f64::min).sqrt(),
            wj.iter().cloned().fold(0.0f64, f64::max).sqrt(),
        );
        let mut d = vec![1.0f64; w.r];
        for (j, &(s, t)) in blocks.iter().enumerate() {
            let dj = wj[j].sqrt();
            for i in s..t {
                d[i] = dj;
            }
        }
        for i in 0..w.r {
            let di = d[i];
            w.row_mut(i).iter_mut().for_each(|v| *v *= di);
        }
        hscale = Some(d);
    }
    let devices: Vec<&str> = match o.device {
        "both" => vec!["cpu", "gpu"],
        d => vec![d],
    };
    for dev in devices {
        let prof = Prof::default();
        let t0 = Instant::now();
        let enc = match dev {
            "cpu" => {
                let pool = Pool::new(o.threads.saturating_sub(1).max(1));
                let th = Instant::now();
                let hc = HCtx::new(Some(&pool), &hfit, o.rho)?;
                prof.add("hctx", th.elapsed().as_secs_f64(), 0);
                nvg::encode_nvg(Some(&pool), &w, &hc, &hfit, &hsel, o.bpw, mult, o.paths, Some(&prof))
            }
            #[cfg(feature = "gpu")]
            "gpu" => crate::nvg_gpu::encode_tensor(&w, &hfit, &hsel, o.rho, o.bpw, mult, o.paths, Some(&prof), o.threads)?,
            other => anyhow::bail!("--device {other}: expected cpu, gpu or both"),
        };
        let total = t0.elapsed().as_secs_f64();
        match enc {
            Some(mut e) => {
                if let Some(d) = &hscale {
                    undo_row_gauge(&mut e.bytes, d)?;
                }
                if let Ok(path) = std::env::var("NVG_BENCH_DUMP") {
                    std::fs::write(&path, &e.bytes)?;
                    eprintln!("nvg-bench: dumped payload ({} bytes) to {path}", e.bytes.len());
                }
                println!("[{dev}] {tname} {}x{} {} ({total:.1}s)", w.r, w.c, e.line);
            }
            None => println!("[{dev}] {tname}: blr refused"),
        }
        println!("[{dev}] profile: {}", prof.line());
    }
    Ok(())
}

/// Capture the input Grams of one layer's leads: the model's own forward over
/// the calibration text, the FIT slice first and the held-out SEL slice after.
/// Returns lead → (fit Gram, sel Gram, fit tokens, sel tokens).
pub fn capture_layer_grams(
    model: &Arc<CmfModel>,
    layer: usize,
    leads: &[&'static str],
    calib: &str,
    fit_tokens: usize,
    sel_tokens: usize,
) -> anyhow::Result<std::collections::HashMap<&'static str, (Mat, Mat, usize, usize)>> {
    let text = std::fs::read_to_string(calib)?;
    let mut pipe = cortiq_engine::Pipeline::from_model(model, cortiq_engine::SamplerConfig::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    let ids = pipe.tokenizer.encode(&text);
    let need = fit_tokens + sel_tokens;
    anyhow::ensure!(ids.len() >= need, "calibration text has {} tokens, need {need}", ids.len());
    let names: std::collections::HashSet<String> = leads.iter().map(|lead| nvg::lead_name(layer, lead)).collect();
    let mut caps = Vec::new();
    for (a, b) in [(0, fit_tokens), (fit_tokens, need)] {
        let keep = names.clone();
        cortiq_engine::gptq_capture::begin_filtered(true, move |n| keep.contains(n));
        let t = Instant::now();
        for w in ids[a..b].chunks(512).filter(|c| c.len() >= 2) {
            pipe.calib_forward(w, layer + 1, 512).map_err(|e| anyhow::anyhow!(e))?;
        }
        caps.push(cortiq_engine::gptq_capture::end());
        eprintln!(
            "nvg-bench: layer {layer} capture of {} inputs over {} tokens in {:.1}s",
            names.len(),
            b - a,
            t.elapsed().as_secs_f64()
        );
    }
    let mut out: std::collections::HashMap<&'static str, (Mat, Mat, usize, usize)> =
        std::collections::HashMap::new();
    for &lead in leads {
        let nm = nvg::lead_name(layer, lead);
        let fit = caps[0].get(&nm).ok_or_else(|| anyhow::anyhow!("no capture for {nm}"))?;
        let sel = caps[1].get(&nm).ok_or_else(|| anyhow::anyhow!("no capture for {nm}"))?;
        let c = (nvg::mean_gram(fit), nvg::mean_gram(sel), fit.count, sel.count);
        out.insert(lead, c);
    }
    Ok(out)
}

/// Re-encode one layer's body linears at the budget the plan gives them, with
/// the encode target taken from `w_from` (the teacher's weights) when it is
/// given. The layer's input Grams come from `grams` (the student's own
/// forward). Returns (tensor name, payload bytes, the delivered bits per
/// weight) for every body linear of the layer.
///
/// This is the allocation step of the sign-training pipeline: the plan moves
/// the bits to the layers that need them, and the training then optimises the
/// metric (and the signs) of the re-planned payloads.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu")]
pub fn recode_layer(
    model: &Arc<CmfModel>,
    w_from: Option<&Arc<CmfModel>>,
    layer: usize,
    plan: &crate::nvg::Plan,
    default_bpw: f64,
    grams: &std::collections::HashMap<&'static str, (Mat, Mat)>,
    rowweight: bool,
    rho: f64,
    jobs: usize,
    threads: usize,
) -> anyhow::Result<Vec<(String, Vec<u8>, f64)>> {
    let mine: Vec<(usize, &'static str)> = model
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(slot, e)| {
            nvg::body_linear(&e.name)
                .filter(|(ll, _)| *ll == layer)
                .map(|(_, short)| (slot, short))
        })
        .collect();
    anyhow::ensure!(!mine.is_empty(), "layer {layer}: no encoder body linears");
    let gqa = model.arch().num_kv_heads < model.arch().num_attention_heads;
    let mut hctx = std::collections::HashMap::new();
    for &(_, short) in &mine {
        let lead = nvg::input_lead(short);
        let (a, b) = grams
            .get(lead)
            .ok_or_else(|| anyhow::anyhow!("layer {layer}: no Gram for the {lead} input"))?;
        hctx.insert(lead, crate::nvg_gpu::HCtxD::new(a, b, rho)?);
    }
    cortiq_engine::gpu_wgpu::nvg_dev::sync();
    let tpool = Pool::new(threads.saturating_sub(1).max(1));
    let mut js = Vec::new();
    let mut gauges: Vec<Option<Vec<f64>>> = Vec::new();
    let mut wanted: Vec<f64> = Vec::new();
    for &(slot, short) in &mine {
        let tname = &model.tensors[slot].name;
        let mut w = match w_from {
            Some(src) => {
                let s = src
                    .tensor_index(tname)
                    .ok_or_else(|| anyhow::anyhow!("the teacher has no tensor {tname}"))?;
                nvg::load_f64(src, s)?
            }
            None => nvg::load_f64(model, slot)?,
        };
        let mult = if gqa && (short == "k" || short == "v") { 1.0 } else { 1.0 };
        let bpw = plan.bpw_for(tname, layer, default_bpw);
        let mut gauge: Option<Vec<f64>> = None;
        if rowweight {
            if let Some(blocks) = row_head_blocks(model.arch(), short, w.r) {
                let hfit = &grams[nvg::input_lead(short)].0;
                let g = crate::nvg_linalg::gemm(Some(&tpool), &w, false, hfit, false);
                let energy: Vec<f64> = blocks
                    .iter()
                    .map(|&(s, t)| (s..t).map(|i| (0..w.c).map(|c| g.at(i, c) * w.at(i, c)).sum::<f64>()).sum())
                    .collect();
                let mut sorted = energy.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let eref = sorted[sorted.len() / 2];
                let wj: Vec<f64> = energy.iter().map(|&ej| (eref / ej.max(1e-300)).clamp(1.0 / 32.0, 32.0)).collect();
                let mut d = vec![1.0f64; w.r];
                for (j, &(s, t)) in blocks.iter().enumerate() {
                    let dj = wj[j].sqrt();
                    for i in s..t {
                        d[i] = dj;
                    }
                }
                for i in 0..w.r {
                    let di = d[i];
                    w.row_mut(i).iter_mut().for_each(|v| *v *= di);
                }
                gauge = Some(d);
            }
        }
        js.push(crate::nvg_gpu::Job { w, hc: &hctx[nvg::input_lead(short)], mult, bpw });
        gauges.push(gauge);
        wanted.push(bpw);
    }
    let done = crate::nvg_gpu::encode_many(&js, crate::nvg::PathsChoice::Auto, threads, jobs, false, &|_, _| {});
    let mut out = Vec::new();
    for (i, d) in done.iter().enumerate() {
        let Ok(Some(e)) = d.enc.as_ref() else {
            continue;
        };
        let mut bytes = e.bytes.clone();
        if let Some(g) = gauges[i].as_ref() {
            undo_row_gauge(&mut bytes, g)?;
        }
        let h = cortiq_core::blr::blr_parse_header(&bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        let achieved =
            cortiq_core::blr::blr_bits(h.d_out, h.d_in, h.rank, h.paths) as f64 / (h.d_out * h.d_in) as f64;
        out.push((model.tensors[mine[i].0].name.clone(), bytes, achieved));
    }
    Ok(out)
}

/// `--layer L`: every body linear of layer L, the input contexts once per
/// input, the tensors with `--jobs` in flight on the device.
fn run_layer(model: &Arc<CmfModel>, o: &BenchOpts, l: usize) -> anyhow::Result<()> {
    let mine: Vec<(usize, &'static str)> = model
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(slot, e)| nvg::body_linear(&e.name).filter(|(ll, _)| *ll == l).map(|(_, short)| (slot, short)))
        .collect();
    anyhow::ensure!(!mine.is_empty(), "layer {l}: no encoder body linears");
    let mut leads: Vec<&'static str> = Vec::new();
    for &(_, short) in &mine {
        let lead = nvg::input_lead(short);
        if !leads.contains(&lead) {
            leads.push(lead);
        }
    }
    let path = |lead: &str, kind: &str| o.gram_cache.map(|p| format!("{p}.{lead}.{kind}.gram"));
    let mut grams: std::collections::HashMap<&str, (Mat, Mat)> = std::collections::HashMap::new();
    let missing: Vec<&str> = leads
        .iter()
        .copied()
        .filter(|lead| path(lead, "fit").is_none_or(|p| !std::path::Path::new(&p).exists()))
        .collect();
    for &lead in &leads {
        if !missing.contains(&lead) {
            let (a, _) = load_gram(&path(lead, "fit").unwrap())?;
            let (b, _) = load_gram(&path(lead, "sel").unwrap())?;
            grams.insert(lead, (a, b));
        }
    }
    if !missing.is_empty() {
        let calib = o.calib.ok_or_else(|| anyhow::anyhow!("--calib FILE is needed to capture the Grams"))?;
        let captured = capture_layer_grams(model, l, &missing, calib, o.fit_tokens, o.sel_tokens)?;
        for lead in missing {
            let (a, b, ca, cb) = captured[lead].clone();
            if let (Some(pf), Some(ps)) = (path(lead, "fit"), path(lead, "sel")) {
                save_gram(&pf, &a, ca)?;
                save_gram(&ps, &b, cb)?;
            }
            grams.insert(lead, (a, b));
        }
    }
    let gqa = model.arch().num_kv_heads < model.arch().num_attention_heads;
    let mult_of = |short: &str| if gqa && (short == "k" || short == "v") { o.kv_factor } else { 1.0 };
    anyhow::ensure!(o.device == "gpu" || o.device == "both", "--layer runs the device encoder (--device gpu|both)");
    // NVG_BENCH_CAPTURE_ONLY=1: capture the layer's input Grams and stop.
    // The capture is a CPU forward over the calibration text, so doing it
    // up front keeps the device sweep free of CPU contention.
    if std::env::var("NVG_BENCH_CAPTURE_ONLY").as_deref() == Ok("1") {
        println!(
            "[gpu] layer {l}: captured {} input contexts, capture only",
            grams.len()
        );
        return Ok(());
    }
    let sel_of = |line: &str| -> Option<f64> { line.rsplit("sel ").next()?.trim().parse().ok() };
    let mut gpu_sel: Vec<Option<f64>> = vec![None; mine.len()];
    // The same knobs as the single-tensor path, so a layer batch and a single
    // tensor produce the same payload: the encode target may come from
    // another file (NVG_BENCH_W_FROM), the per-head row gauge may be applied
    // (NVG_BENCH_ROWWEIGHT=invE) and the payloads may be dumped
    // (NVG_BENCH_DUMP_DIR) for the per-tensor ppl sweep.
    let wsrc: Option<Arc<CmfModel>> = match std::env::var("NVG_BENCH_W_FROM") {
        Ok(p) => Some(Arc::new(CmfModel::open_sharded(&p)?)),
        Err(_) => None,
    };
    let rowweight = std::env::var("NVG_BENCH_ROWWEIGHT").ok();
    if let Some(r) = &rowweight {
        anyhow::ensure!(r == "invE", "NVG_BENCH_ROWWEIGHT={r}: expected invE");
    }
    let dump_dir = std::env::var("NVG_BENCH_DUMP_DIR").ok().map(std::path::PathBuf::from);
    #[cfg(feature = "gpu")]
    {
        let t0 = Instant::now();
        let th = Instant::now();
        let mut hctx = std::collections::HashMap::new();
        for &lead in &leads {
            let (a, b) = &grams[lead];
            hctx.insert(lead, crate::nvg_gpu::HCtxD::new(a, b, o.rho)?);
        }
        cortiq_engine::gpu_wgpu::nvg_dev::sync();
        let thc = th.elapsed().as_secs_f64();
        // Per tensor: the target matrix (teacher or student) and the optional
        // row gauge that is undone on the packed bytes after the encode.
        let tpool = Pool::new(o.threads.saturating_sub(1).max(1));
        let mut jobs = Vec::new();
        let mut gauges: Vec<Option<Vec<f64>>> = Vec::new();
        let mut dumps: Vec<Option<std::path::PathBuf>> = Vec::new();
        for &(slot, short) in &mine {
            let tname = &model.tensors[slot].name;
            let mut w = match &wsrc {
                Some(src) => {
                    let s = src
                        .tensor_index(tname)
                        .ok_or_else(|| anyhow::anyhow!("W_FROM: no tensor {tname}"))?;
                    nvg::load_f64(src, s)?
                }
                None => nvg::load_f64(model, slot)?,
            };
            let mut gauge: Option<Vec<f64>> = None;
            if rowweight.is_some() {
                // A batch may mix heads (qkv/z/k/v) and plain matrices
                // (gate/up/down/out): the gauge only touches the former.
                let Some(blocks) = row_head_blocks(model.arch(), short, w.r) else {
                    eprintln!("nvg-bench: {tname} ({short}) has no per-head rows: no row gauge");
                    gauges.push(None);
                    jobs.push(crate::nvg_gpu::Job { w, hc: &hctx[nvg::input_lead(short)], mult: mult_of(short), bpw: o.bpw });
                    dumps.push(dump_dir.as_ref().map(|dir| {
                        let stem = tname.strip_prefix("model.layers.").unwrap_or(tname).replace('.', "_");
                        dir.join(format!("{stem}.blr"))
                    }));
                    continue;
                };
                let hfit = &grams[nvg::input_lead(short)].0;
                let g = crate::nvg_linalg::gemm(Some(&tpool), &w, false, hfit, false);
                let energy: Vec<f64> = blocks
                    .iter()
                    .map(|&(s, t)| (s..t).map(|i| (0..w.c).map(|c| g.at(i, c) * w.at(i, c)).sum::<f64>()).sum())
                    .collect();
                let mut sorted = energy.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let eref = sorted[sorted.len() / 2];
                let wj: Vec<f64> = energy.iter().map(|&ej| (eref / ej.max(1e-300)).clamp(1.0 / 32.0, 32.0)).collect();
                let mut d = vec![1.0f64; w.r];
                for (j, &(s, t)) in blocks.iter().enumerate() {
                    let dj = wj[j].sqrt();
                    for i in s..t {
                        d[i] = dj;
                    }
                }
                for i in 0..w.r {
                    let di = d[i];
                    w.row_mut(i).iter_mut().for_each(|v| *v *= di);
                }
                eprintln!(
                    "nvg-bench: rowweight over {} heads of {tname}: scale {:.3}..{:.3}",
                    blocks.len(),
                    wj.iter().cloned().fold(f64::INFINITY, f64::min).sqrt(),
                    wj.iter().cloned().fold(0.0f64, f64::max).sqrt(),
                );
                gauge = Some(d);
            }
            jobs.push(crate::nvg_gpu::Job { w, hc: &hctx[nvg::input_lead(short)], mult: mult_of(short), bpw: o.bpw });
            gauges.push(gauge);
            dumps.push(dump_dir.as_ref().map(|dir| {
                let stem = tname.strip_prefix("model.layers.").unwrap_or(tname).replace('.', "_");
                dir.join(format!("{stem}.blr"))
            }));
        }
        let te = Instant::now();
        // stage syncs would serialise concurrent tensors: profile only with one job
        let done = crate::nvg_gpu::encode_many(&jobs, o.paths, o.threads, o.jobs, o.jobs == 1, &|i, d| {
            let name = &model.tensors[mine[i].0].name;
            match &d.enc {
                Ok(Some(e)) => {
                    println!("[gpu] {name} {}x{} {} ({:.1}s)", jobs[i].w.r, jobs[i].w.c, e.line, d.secs);
                    // Undo the row gauge on the packed bytes and write the
                    // payload out: the sweep patches it into a model and
                    // measures the perplexity of the result.
                    let mut bytes = e.bytes.clone();
                    if let Some(g) = gauges.get(i).and_then(|x| x.as_ref()) {
                        if let Err(err) = undo_row_gauge(&mut bytes, g) {
                            eprintln!("nvg-bench: row gauge undo failed for {name}: {err}");
                        }
                    }
                    if let Some(path) = dumps.get(i).and_then(|x| x.as_ref()) {
                        if let Some(dir) = path.parent() {
                            let _ = std::fs::create_dir_all(dir);
                        }
                        match std::fs::write(path, &bytes) {
                            Ok(()) => println!("[dump] {name} {} {}", bytes.len(), path.display()),
                            Err(err) => eprintln!("nvg-bench: dump failed for {name}: {err}"),
                        }
                    }
                }
                Ok(None) => println!("[gpu] {name}: blr refused"),
                Err(e) => println!("[gpu] {name}: error {e}"),
            }
            if let Some(p) = &d.prof {
                println!("[gpu] {name} profile: {p}");
            }
        });
        let tenc = te.elapsed().as_secs_f64();
        for (k, d) in done.iter().enumerate() {
            if let Ok(Some(e)) = &d.enc {
                gpu_sel[k] = sel_of(&e.line);
            }
        }
        let sum: f64 = done.iter().map(|d| d.secs).sum();
        println!(
            "[gpu] layer {l}: {} tensors, input contexts {thc:.1}s, encode {tenc:.1}s wall ({sum:.1}s summed over tensors, jobs {}), total {:.1}s",
            jobs.len(),
            o.jobs,
            t0.elapsed().as_secs_f64()
        );
    }
    if o.device == "both" {
        // the f64 host encoder on the same Grams, tensor by tensor
        let pool = Pool::new(o.threads.saturating_sub(1).max(1));
        let mut hc = std::collections::HashMap::new();
        for &lead in &leads {
            hc.insert(lead, HCtx::new(Some(&pool), &grams[lead].0, o.rho)?);
        }
        for (k, &(slot, short)) in mine.iter().enumerate() {
            let w = nvg::load_f64(model, slot)?;
            let (hfit, hsel) = &grams[nvg::input_lead(short)];
            let t = Instant::now();
            let enc = nvg::encode_nvg(Some(&pool), &w, &hc[nvg::input_lead(short)], hfit, hsel, o.bpw, mult_of(short), o.paths, None);
            let name = &model.tensors[slot].name;
            if let Some(e) = &enc {
                println!("[cpu] {name} {}x{} {} ({:.1}s)", w.r, w.c, e.line, t.elapsed().as_secs_f64());
                if let (Some(c), Some(g)) = (sel_of(&e.line), gpu_sel[k]) {
                    println!("[cmp] {name} sel cpu {c:.4} gpu {g:.4} rel {:+.3}%", (g - c) / c * 100.0);
                }
            }
        }
    }
    Ok(())
}

/// Row blocks of `short`'s output dimension that a per-head normaliser
/// consumes (GDN q/k/v/z, attention k/v). `None` when the tensor has no such
/// structure or `rows` does not match the architecture.
fn row_head_blocks(arch: &cortiq_core::types::ModelArch, short: &str, rows: usize) -> Option<Vec<(usize, usize)>> {
    let blocks_of = |count: usize, dim: usize| -> Option<Vec<(usize, usize)>> {
        (count > 0 && dim > 0).then(|| (0..count).map(|j| (j * dim, (j + 1) * dim)).collect())
    };
    let b: Vec<(usize, usize)> = match short {
        "qkv" => {
            let (nk, dk) = (arch.linear_num_key_heads?, arch.linear_key_head_dim?);
            let (nv, dv) = (arch.linear_num_value_heads?, arch.linear_value_head_dim?);
            let mut v = Vec::new();
            let mut off = 0usize;
            for (count, dim) in [(nk, dk), (nk, dk), (nv, dv)] {
                for (s, t) in blocks_of(count, dim)? {
                    v.push((s + off, t + off));
                }
                off += count * dim;
            }
            v
        }
        "z" => blocks_of(arch.linear_num_value_heads?, arch.linear_value_head_dim?)?,
        "q" => blocks_of(arch.num_attention_heads, arch.head_dim)?,
        "k" => blocks_of(arch.num_kv_heads, arch.head_dim)?,
        "v" => {
            let hd = if short == "v" { arch.v_head_dim.unwrap_or(arch.head_dim) } else { arch.head_dim };
            blocks_of(arch.num_kv_heads, hd)?
        }
        _ => return None,
    };
    (b.last().map(|&(_, t)| t) == Some(rows)).then_some(b)
}

/// Decode-side undo of the row-weight gauge: divide the stored per-row left
/// scales `h` of every path by `d`.
fn undo_row_gauge(bytes: &mut [u8], d: &[f64]) -> anyhow::Result<()> {
    let h = cortiq_core::blr::blr_parse_header(bytes).map_err(|e| anyhow::anyhow!(e))?;
    anyhow::ensure!(h.d_out == d.len(), "row gauge: {} rows vs {} scales", h.d_out, d.len());
    for p in 0..h.paths {
        let o = cortiq_core::blr::blr_path_offsets(&h, p);
        for i in 0..h.d_out {
            let off = o.h + 2 * i;
            let cur = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            let v = cortiq_core::quant::f32_to_f16((cur as f64 / d[i]) as f32);
            bytes[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}
