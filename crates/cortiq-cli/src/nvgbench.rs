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
            pipe.calib_forward(w, l + 1, 512).map_err(|e| anyhow::anyhow!(e))?;
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
    let w = nvg::load_f64(&model, slot)?;
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
            Some(e) => println!("[{dev}] {tname} {}x{} {} ({total:.1}s)", w.r, w.c, e.line),
            None => println!("[{dev}] {tname}: blr refused"),
        }
        println!("[{dev}] profile: {}", prof.line());
    }
    Ok(())
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
        let text = std::fs::read_to_string(calib)?;
        let mut pipe = cortiq_engine::Pipeline::from_model(model, cortiq_engine::SamplerConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        pipe.set_confidence(false);
        let ids = pipe.tokenizer.encode(&text);
        let need = o.fit_tokens + o.sel_tokens;
        anyhow::ensure!(ids.len() >= need, "calibration text has {} tokens, need {need}", ids.len());
        let names: std::collections::HashSet<String> = missing.iter().map(|lead| nvg::lead_name(l, lead)).collect();
        let mut caps = Vec::new();
        for (a, b) in [(0, o.fit_tokens), (o.fit_tokens, need)] {
            let keep = names.clone();
            cortiq_engine::gptq_capture::begin_filtered(true, move |n| keep.contains(n));
            let t = Instant::now();
            for w in ids[a..b].chunks(512).filter(|c| c.len() >= 2) {
                pipe.calib_forward(w, l + 1, 512).map_err(|e| anyhow::anyhow!(e))?;
            }
            caps.push(cortiq_engine::gptq_capture::end());
            eprintln!("nvg-bench: layer {l} capture of {} inputs over {} tokens in {:.1}s", names.len(), b - a, t.elapsed().as_secs_f64());
        }
        for &lead in &missing {
            let nm = nvg::lead_name(l, lead);
            let fit = caps[0].get(&nm).ok_or_else(|| anyhow::anyhow!("no capture for {nm}"))?;
            let sel = caps[1].get(&nm).ok_or_else(|| anyhow::anyhow!("no capture for {nm}"))?;
            let (a, b) = (nvg::mean_gram(fit), nvg::mean_gram(sel));
            if let (Some(pf), Some(ps)) = (path(lead, "fit"), path(lead, "sel")) {
                save_gram(&pf, &a, fit.count)?;
                save_gram(&ps, &b, sel.count)?;
            }
            grams.insert(lead, (a, b));
        }
    }
    let gqa = model.arch().num_kv_heads < model.arch().num_attention_heads;
    let mult_of = |short: &str| if gqa && (short == "k" || short == "v") { o.kv_factor } else { 1.0 };
    anyhow::ensure!(o.device == "gpu" || o.device == "both", "--layer runs the device encoder (--device gpu|both)");
    let sel_of = |line: &str| -> Option<f64> { line.rsplit("sel ").next()?.trim().parse().ok() };
    let mut gpu_sel: Vec<Option<f64>> = vec![None; mine.len()];
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
        let mut jobs = Vec::new();
        for &(slot, short) in &mine {
            jobs.push(crate::nvg_gpu::Job { w: nvg::load_f64(model, slot)?, hc: &hctx[nvg::input_lead(short)], mult: mult_of(short) });
        }
        let te = Instant::now();
        // stage syncs would serialise concurrent tensors: profile only with one job
        let done = crate::nvg_gpu::encode_many(&jobs, o.bpw, o.paths, o.threads, o.jobs, o.jobs == 1, &|i, d| {
            let name = &model.tensors[mine[i].0].name;
            match &d.enc {
                Ok(Some(e)) => println!("[gpu] {name} {}x{} {} ({:.1}s)", jobs[i].w.r, jobs[i].w.c, e.line, d.secs),
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
