//! `cortiq nvg-train`: NVG Ť — light distillation training of a compressed
//! model's metric (and optionally its signs) against its full-precision
//! teacher, entirely on the GPU (`cortiq_engine::nvg_train`).
//!
//! The teacher pass runs first (its top-k next-token distribution per token
//! is kept on the host, optionally cached on disk, the teacher is then
//! freed); the student trains its binary low-rank metric (h, ℓ, g of every
//! path) and every RMSNorm gain with Adam, a cosine schedule after a short
//! warm-up, global gradient-norm clipping, and the f16 gradient scale halved
//! on a non-finite step. With `--train-signs` every sign also gets a
//! straight-through latent, updated inside the reverse sweep with the
//! factored optimizer of `gpu_wgpu::train_sign` on the same schedule shape.
//! The held-out windows pick the best checkpoint (metric and bits), which is
//! written into a copy of the student file (same layout and sizes: only
//! payload values change).

use crate::gptq::{CalibSource, calib_source_tokens, split_calib_budget};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_engine::gpu_wgpu::train_sign as ts;
use cortiq_engine::nvg_train as nt;
use std::io::{Read, Write};
use std::time::Instant;

/// Sign-training settings (`--train-signs`).
pub struct SignOpts {
    pub opt: ts::SignOpt,
    pub tau: f64,
    /// Peak sign learning rate; None = auto (min(0.15·τ/√steps, 4e-3·τ)).
    pub lr: Option<f64>,
    /// Sign steps at η = 0 while the second-moment statistics settle.
    pub warm: usize,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub clamp: f64,
    pub hyst: f64,
    /// Halve η when the per-step flip rate exceeds this 3 steps running.
    pub max_step_flip: f64,
    /// Halve η when the distinct flipped fraction exceeds this (checked per eval).
    pub max_total_flip: f64,
    pub seed: u64,
}

pub struct NvgTrainOpts<'a> {
    pub student: &'a str,
    pub teacher: &'a str,
    pub calib: &'a [String],
    pub tokens: usize,
    pub window: usize,
    pub batch: usize,
    pub epochs: usize,
    pub lr: f64,
    pub topk: usize,
    pub eval_windows: usize,
    pub eval_every: usize,
    pub clip: f64,
    pub seed: u64,
    pub output: &'a str,
    pub signs: Option<SignOpts>,
    pub topk_cache: Option<&'a str>,
    /// Directory for the resumable state (written every `state_every` steps).
    pub state_dir: Option<&'a str>,
    pub state_every: usize,
    /// Delete the previous state file before writing the next.
    pub state_replace: bool,
    /// Resume from the latest state in this directory.
    pub resume: Option<&'a str>,
    /// A bit-budget plan (`layer <n> <bpw>` / `tensor <name> <bpw>`): the
    /// student's BLR tensors are re-encoded at the planned budget before the
    /// training starts, with the teacher's weights as the encode target.
    pub budget_plan: Option<&'a str>,
    /// Directory with the cached input Grams (`L{layer}.{lead}.{kind}.gram`).
    pub budget_gram_cache: Option<&'a str>,
    /// The per-head row gauge on the re-encoded tensors.
    pub budget_rowweight: bool,
    /// The capture's FIT / SEL tokens for the re-encode.
    pub budget_fit_tokens: usize,
    pub budget_sel_tokens: usize,
    /// The device jobs of the re-encode.
    pub budget_jobs: usize,
}

const TOPK_MAGIC: &[u8; 8] = b"NVGTOPK1";

/// Key of a top-k cache: teacher file (size, mtime) and every argument the
/// windows and the teacher batches depend on.
fn topk_key(o: &NvgTrainOpts, n_batches: usize) -> u64 {
    let meta = std::fs::metadata(o.teacher).ok();
    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    feed(&size.to_le_bytes());
    feed(&mtime.to_le_bytes());
    for c in o.calib {
        feed(c.as_bytes());
        feed(&[0]);
    }
    for v in [o.tokens, o.window, o.batch, o.seed as usize, o.topk, o.eval_windows, n_batches] {
        feed(&(v as u64).to_le_bytes());
    }
    h
}

#[allow(clippy::type_complexity)]
fn topk_load(path: &str, key: u64, nb: usize, per: usize) -> Option<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
    let mut f = std::io::BufReader::with_capacity(8 << 20, std::fs::File::open(path).ok()?);
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).ok()?;
    let word = |i: usize| u64::from_le_bytes(hdr[i..i + 8].try_into().unwrap());
    if &hdr[..8] != TOPK_MAGIC || word(8) != key || word(16) != nb as u64 || word(24) != per as u64 {
        return None;
    }
    let (mut ids, mut ps) = (Vec::with_capacity(nb), Vec::with_capacity(nb));
    let mut buf = vec![0u8; per * 4];
    for _ in 0..nb {
        f.read_exact(&mut buf).ok()?;
        ids.push(buf.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect());
        f.read_exact(&mut buf).ok()?;
        ps.push(buf.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
    }
    Some((ids, ps))
}

fn topk_save(path: &str, key: u64, ids: &[Vec<u32>], ps: &[Vec<f32>]) -> anyhow::Result<()> {
    let tmp = format!("{path}.tmp{}", std::process::id());
    {
        let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
        f.write_all(TOPK_MAGIC)?;
        f.write_all(&key.to_le_bytes())?;
        f.write_all(&(ids.len() as u64).to_le_bytes())?;
        f.write_all(&(ids.first().map(|v| v.len()).unwrap_or(0) as u64).to_le_bytes())?;
        for (i, p) in ids.iter().zip(ps) {
            let bi: Vec<u8> = i.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bi)?;
            let bp: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bp)?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The allocation step of the training pipeline: re-encode the student's BLR
/// tensors whose budget the plan changes, with the teacher's weights as the
/// encode target and the student's own input Grams as the calibration. The
/// layers the plan does not name keep their payloads untouched.
///
/// Returns the re-planned student (a new model: the payloads of the re-encoded
/// tensors differ, so the file is rewritten).
fn apply_budget_plan(
    student: &CmfModel,
    teacher: &CmfModel,
    plan: &crate::nvg::Plan,
    o: &NvgTrainOpts,
) -> anyhow::Result<CmfModel> {
    let student = std::sync::Arc::new(CmfModel::open_sharded(&student.path)?);
    let teacher = std::sync::Arc::new(CmfModel::open_sharded(&teacher.path)?);
    use cortiq_core::blr::{blr_bits, blr_parse_header};
    use cortiq_core::format::TensorSpec;
    use cortiq_core::types::TensorDtype;
    // the budget every tensor has now
    let mut current: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for e in student.tensors.iter() {
        if e.dtype != TensorDtype::BinaryLowRank {
            continue;
        }
        let bytes = student.entry_bytes(e);
        let Ok(h) = blr_parse_header(bytes) else { continue };
        let n = (e.shape[0] * e.shape[1]) as f64;
        current.insert(
            e.name.clone(),
            blr_bits(h.d_out, h.d_in, h.rank, h.paths) as f64 / n,
        );
    }
    // the wanted budget per tensor: the plan's rule, else the tensor's own
    let mut want: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for (name, cur) in &current {
        let layer = crate::nvg::body_linear(name).map(|(l, _)| l);
        let w = plan.bpw_for(name, layer.unwrap_or(0), *cur);
        want.insert(name.clone(), w);
        if (w - cur).abs() > 1e-9 {
            if let Some(l) = layer {
                layers.insert(l);
            }
        }
    }
    if layers.is_empty() {
        eprintln!("nvg-train: the plan asks for the budget every tensor already has — nothing to re-encode");
        return Ok(CmfModel::open_sharded(o.student)?);
    }
    eprintln!(
        "nvg-train: budget plan — re-encoding {} layer(s) at the planned budget ({} layer rules, {} tensor rules)",
        layers.len(),
        plan.layers().len(),
        plan.tensors().len()
    );
    let mut payloads: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
    for &layer in &layers {
        let names: Vec<String> = current
            .keys()
            .filter(|n| {
                crate::nvg::body_linear(n.as_str())
                    .is_some_and(|(l, _)| l == layer && (want[n.as_str()] - current[n.as_str()]).abs() > 1e-9)
            })
            .cloned()
            .collect();
        if names.is_empty() {
            continue;
        }
        // the layer's input Grams: the cached ones, else the student's own forward
        let mut leads: Vec<&'static str> = Vec::new();
        for n in &names {
            if let Some((_, short)) = crate::nvg::body_linear(n) {
                let lead = crate::nvg::input_lead(short);
                if !leads.contains(&lead) {
                    leads.push(lead);
                }
            }
        }
        let mut grams: std::collections::HashMap<&'static str, (crate::nvg_linalg::Mat, crate::nvg_linalg::Mat)> =
            std::collections::HashMap::new();
        let mut missing: Vec<&'static str> = Vec::new();
        for &lead in &leads {
            let (a, b) = match o.budget_gram_cache {
                Some(dir) => {
                    let p = format!("{dir}/L{layer}.{lead}.fit.gram");
                    let q = format!("{dir}/L{layer}.{lead}.sel.gram");
                    match (std::fs::read(&p), std::fs::read(&q)) {
                        (Ok(a), Ok(b)) => (load_gram_bytes(&a)?, load_gram_bytes(&b)?),
                        _ => {
                            missing.push(lead);
                            continue;
                        }
                    }
                }
                None => {
                    missing.push(lead);
                    continue;
                }
            };
            grams.insert(lead, (a, b));
        }
        if !missing.is_empty() {
            let calib = o
                .calib
                .first()
                .ok_or_else(|| anyhow::anyhow!("--budget-plan needs --calib for the Gram capture"))?;
            let path = calib.split('=').next().unwrap_or(calib);
            let captured = crate::nvgbench::capture_layer_grams(
                &student,
                layer,
                &missing,
                path,
                o.budget_fit_tokens,
                o.budget_sel_tokens,
            )?;
            for lead in missing {
                let (a, b, _, _) = captured[lead].clone();
                grams.insert(lead, (a, b));
            }
        }
        let t0 = Instant::now();
        let done = crate::nvgbench::recode_layer(
            &student,
            Some(&teacher),
            layer,
            plan,
            0.0,
            &grams,
            o.budget_rowweight,
            0.03,
            o.budget_jobs.max(1),
            8,
        )?;
        for (name, bytes, achieved) in done {
            eprintln!("  {name}: {} bytes, {:.4} bpw", bytes.len(), achieved);
            payloads.insert(name, bytes);
        }
        eprintln!("  layer {layer} re-encoded in {:.0}s", t0.elapsed().as_secs_f64());
    }
    // the re-planned student
    let specs: Vec<TensorSpec> = student
        .tensors
        .iter()
        .map(|e| {
            let data = payloads
                .get(&e.name)
                .cloned()
                .unwrap_or_else(|| student.entry_bytes(e).to_vec());
            TensorSpec {
                name: e.name.clone(),
                dtype: if payloads.contains_key(&e.name) {
                    TensorDtype::BinaryLowRank
                } else {
                    e.dtype
                },
                shape: e.shape.clone(),
                data,
            }
        })
        .collect();
    let mut header = student.header.clone();
    header.section_hashes = None;
    let path = format!("{}.planned.cmf", o.output);
    CmfModel::write(&path, &header, &specs, None, student.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!("nvg-train: re-planned student → {path}");
    Ok(CmfModel::open_sharded(&path)?)
}

/// Read a cached Gram file (the `nvg-bench` format: magic, n, count, f32 data).
fn load_gram_bytes(raw: &[u8]) -> anyhow::Result<crate::nvg_linalg::Mat> {
    anyhow::ensure!(raw.len() > 16, "gram file too short");
    let n = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    anyhow::ensure!(raw.len() >= 16 + n * n * 4, "gram file truncated");
    let d: Vec<f64> = raw[16..16 + n * n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    Ok(crate::nvg_linalg::Mat::from_vec(n, n, d))
}

pub fn run(o: &NvgTrainOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    // The plan is parsed first: a malformed rule is a cheap error, and it
    // must not be hidden behind a missing GPU or a missing model.
    let plan = match o.budget_plan {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("--budget-plan {path}: {e}"))?;
            Some(crate::nvg::Plan::parse(&text).map_err(|e| anyhow::anyhow!("--budget-plan {path}: {e}"))?)
        }
        None => None,
    };
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().ok_or_else(|| anyhow::anyhow!("nvg-train needs a wgpu GPU"))?;
    let tmodel = CmfModel::open_sharded(o.teacher)?;
    // The allocation step: the plan moves the bits to the layers that need
    // them, and the training then optimises the metric (and the signs) of the
    // re-planned payloads. The layers the plan does not name keep their own
    // budget — their payloads are not touched.
    let smodel = match &plan {
        Some(plan) => {
            let base = CmfModel::open_sharded(o.student)?;
            apply_budget_plan(&base, &tmodel, plan, o)?
        }
        None => CmfModel::open_sharded(o.student)?,
    };
    let vocab = tmodel.vocab.as_deref().ok_or_else(|| anyhow::anyhow!("teacher has no embedded tokenizer"))?;
    let tok = cortiq_engine::tokenizer::Tokenizer::from_bytes(vocab).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let srcs: Vec<CalibSource> = o.calib.iter().map(|s| CalibSource::parse(s)).collect::<anyhow::Result<_>>()?;
    let want = o.tokens + o.eval_windows * o.window;
    let budgets = split_calib_budget(&srcs, want);
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for (src, &n) in srcs.iter().zip(&budgets) {
        // added tokens recognized, so chat-format text trains the real <|im_start|>… ids
        let ids = calib_source_tokens(&src.path, n, &|t| tok.encode(t))?;
        windows.extend(ids.chunks(o.window).filter(|w| w.len() == o.window).map(<[u32]>::to_vec));
    }
    anyhow::ensure!(windows.len() > o.eval_windows + o.batch, "too few {}-token windows ({})", o.window, windows.len());
    // interleave the sources deterministically, then hold out the tail
    let mut rng = o.seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for i in (1..windows.len()).rev() {
        windows.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    let eval: Vec<Vec<u32>> = windows.split_off(windows.len() - o.eval_windows);
    let eval_n = eval.len() / o.batch * o.batch;
    let train = windows;
    eprintln!(
        "nvg-train: {} training windows + {} held out of {} tokens, batch {}, top-{} teacher",
        train.len(),
        eval_n,
        o.window,
        o.batch,
        o.topk
    );

    // ── teacher pass ──
    // Whole training batches, then the held-out batches: teacher batch b is
    // exactly `batch_ids(b)` (a training count that is not a multiple of the
    // batch would otherwise shift every held-out batch onto the wrong text).
    let k = o.topk;
    let whole = train.len() / o.batch * o.batch;
    let n_all_b = (whole + eval_n) / o.batch;
    let per = o.batch * o.window * k;
    let key = topk_key(o, n_all_b);
    let cached = o.topk_cache.and_then(|p| topk_load(p, key, n_all_b, per));
    let (t_ids, t_p) = if let Some(v) = cached {
        eprintln!("  teacher top-k from {} ({n_all_b} batches)", o.topk_cache.unwrap_or(""));
        v
    } else {
        let mut t_ids: Vec<Vec<u32>> = Vec::new();
        let mut t_p: Vec<Vec<f32>> = Vec::new();
        let tnet = nt::Net::load(c, &tmodel, true, false, None).map_err(|e| anyhow::anyhow!(e))?;
        let tw = nt::Work::new(c, &tnet, o.batch, o.window);
        let all: Vec<&Vec<u32>> = train[..whole].iter().chain(eval.iter().take(eval_n)).collect();
        let tt = Instant::now();
        for (bi, chunk) in all.chunks(o.batch).enumerate() {
            anyhow::ensure!(chunk.len() == o.batch, "teacher batch {bi}: {} windows", chunk.len());
            let ids: Vec<u32> = chunk.iter().flat_map(|w| w.iter().copied()).collect();
            let (ti, tp) = nt::teacher_topk(c, &tnet, &tw, &ids, k).ok_or_else(|| anyhow::anyhow!("teacher pass failed"))?;
            t_ids.push(ti);
            t_p.push(tp);
            eprint!("\r  teacher {}/{}  {:.0} tok/s   ", bi + 1, all.len() / o.batch, ((bi + 1) * o.batch * o.window) as f64 / tt.elapsed().as_secs_f64());
        }
        eprintln!();
        if let Some(p) = o.topk_cache {
            match topk_save(p, key, &t_ids, &t_p) {
                Ok(()) => eprintln!("  teacher top-k cached in {p}"),
                Err(e) => eprintln!("  warning: top-k cache {p} not written: {e}"),
            }
        }
        (t_ids, t_p)
    };
    let n_train_b = train.len() / o.batch;
    let batch_ids = |b: usize| -> Vec<u32> {
        let src = if b < n_train_b { &train[b * o.batch..(b + 1) * o.batch] } else { &eval[(b - n_train_b) * o.batch..(b - n_train_b + 1) * o.batch] };
        src.iter().flat_map(|w| w.iter().copied()).collect()
    };

    // ── student ──
    let total_steps = n_train_b * o.epochs;
    let sign_init = o.signs.as_ref().map(|s| ts::SignInit { opt: s.opt, tau: s.tau as f32, seed: s.seed as u32, keep_bits0: true, defer: false });
    let snet = nt::Net::load(c, &smodel, true, true, sign_init.as_ref()).map_err(|e| anyhow::anyhow!(e))?;
    let sw = nt::Work::new(c, &snet, o.batch, o.window);
    let train_signs = o.signs.is_some();
    let live = snet.sign_live();
    // auto: the random-walk spread of the reference QAT run (0.15·τ/√steps),
    // capped at its per-step move (lr/τ = 4e-3): on the 0.8B the uncapped
    // 2.3e-3 for 256 steps flipped 0.2 % of the signs in one step and diverged
    let eta0 = o.signs.as_ref().map(|s| s.lr.unwrap_or((0.15 * s.tau / (total_steps.max(1) as f64).sqrt()).min(4e-3 * s.tau))).unwrap_or(0.0);
    if let Some(s) = &o.signs {
        eprintln!(
            "sign training: {} signs in {} matrices, optimizer {} ({} B/sign state ≈ {:.2} GB, scratch {:.2} GB), τ {}, η0 {:.3e}{}, warm {} sign steps",
            live,
            snet.n_sign_mats,
            s.opt.name(),
            s.opt.bytes_per_sign(),
            (live as f64 * s.opt.bytes_per_sign() as f64 + live as f64 / 8.0) / 1e9,
            sw.sg.as_ref().map(|x| x.bytes() as f64 / 1e9).unwrap_or(0.0),
            s.tau,
            eta0,
            if s.lr.is_none() { " (auto min(0.15·τ/√steps, 4e-3·τ))" } else { "" },
            s.warm
        );
    }
    let n_eval_b = eval_n / o.batch;
    let eval_loss = |net: &nt::Net| -> anyhow::Result<f64> {
        let mut s = 0.0;
        for e in 0..n_eval_b {
            let b = n_train_b + e;
            s += nt::loss_only(c, net, &sw, &batch_ids(b), &t_ids[b], &t_p[b], k).ok_or_else(|| anyhow::anyhow!("eval failed"))?;
        }
        Ok(s / n_eval_b.max(1) as f64)
    };
    // resumed run: the state's metric, Adam moments, bits and sign state, the
    // best snapshot, and the loop counters (meta.json)
    let meta: Option<serde_json::Value> = match o.resume {
        Some(dir) => {
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/meta.json"))?)?;
            nt::load_state(c, &snet, std::path::Path::new(&format!("{dir}/{}", m["state"].as_str().unwrap_or("")))).map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("resumed {dir} at step {} (epoch {}, batch {} of the epoch)", m["done"], m["epoch"], m["consumed"]);
            Some(m)
        }
        None => None,
    };
    let mu = |k: &str| meta.as_ref().and_then(|m| m[k].as_u64()).unwrap_or(0);
    let mf = |k: &str, d: f64| meta.as_ref().and_then(|m| m[k].as_f64()).unwrap_or(d);
    let e0 = match &meta {
        Some(m) => m["e0"].as_f64().unwrap_or(f64::NAN),
        None => eval_loss(&snet)?,
    };
    eprintln!("held-out KL before training: {e0:.5}");
    let warm = (total_steps / 50).max(1);
    let shape = |d: usize| -> f64 {
        if d <= warm {
            d as f64 / warm as f64
        } else {
            let pr = (d - warm) as f64 / (total_steps - warm).max(1) as f64;
            0.5 * (1.0 + (std::f64::consts::PI * pr.min(1.0)).cos())
        }
    };
    let mut best = (e0, nt::snapshot(c, &snet, train_signs).ok_or_else(|| anyhow::anyhow!("snapshot"))?, 0usize);
    if let (Some(m), Some(dir)) = (&meta, o.resume) {
        let snap = nt::load_snap(&best.1, std::path::Path::new(&format!("{dir}/{}", m["best"].as_str().unwrap_or("")))).map_err(|e| anyhow::anyhow!(e))?;
        best = (m["best_kl"].as_f64().unwrap_or(e0), snap, m["best_step"].as_u64().unwrap_or(0) as usize);
    }
    let mut gscale = mf("gscale", 4096.0) as f32;
    let mut done = mu("done") as usize;
    let mut skipped = mu("skipped") as usize;
    let mut sign_steps = mu("sign_steps") as u32;
    let mut eta_mul = mf("eta_mul", 1.0);
    let (mut hi_streak, mut bad_evals) = (mu("hi_streak") as usize, mu("bad_evals") as usize);
    let (mut step_flips_total, mut sat_total, mut clamp_total) = (mu("step_flips_total"), mu("sat_total"), mu("clamp_total"));
    let mut last_distinct = 0u64;
    let mut step_secs = 0f64;
    let done0 = done;
    let mut eval_log: Vec<serde_json::Value> = meta.as_ref().and_then(|m| m["evals"].as_array().cloned()).unwrap_or_default();
    let (resume_ep, resume_pos) = (mu("epoch") as usize, mu("consumed") as usize);
    // probe runs: stop after this many steps (the schedule still spans the full run)
    let stop_after: usize = std::env::var("CMF_NVG_STOP_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let ts0 = Instant::now();
    let mut order: Vec<usize> = (0..n_train_b).collect();
    for ep in 0..o.epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        if meta.is_some() && ep < resume_ep {
            continue;
        }
        let first = if meta.is_some() && ep == resume_ep { resume_pos } else { 0 };
        for (pos, &b) in order.iter().enumerate().skip(first) {
            if done >= stop_after {
                break;
            }
            let tstep = Instant::now();
            let cfg = o.signs.as_ref().map(|s| {
                let sn = sign_steps + 1;
                let eta = if (sn as usize) <= s.warm { 0.0 } else { eta0 * eta_mul * shape(done + 1) };
                ts::SignStepCfg {
                    opt: s.opt,
                    eta: eta as f32,
                    step: sn,
                    b1: s.b1 as f32,
                    b2: s.b2 as f32,
                    eps: s.eps as f32,
                    clamp: s.clamp as f32,
                    hyst: s.hyst as f32,
                    k_ss: (2.0 / s.tau) as f32,
                    seed: s.seed as u32,
                    defer: false,
                }
            });
            let st = nt::step(c, &snet, &sw, &batch_ids(b), &t_ids[b], &t_p[b], k, gscale, cfg.as_ref()).ok_or_else(|| anyhow::anyhow!("step failed"))?;
            if st.backward && train_signs {
                sign_steps += 1;
                step_flips_total += st.flips;
                sat_total += st.saturated;
                clamp_total += st.clamped;
            }
            let gn = if st.backward { nt::grad_norm(c, &snet).unwrap_or(f64::NAN) } else { f64::NAN };
            if !st.finite || !gn.is_finite() || st.g_nonfinite > 0 {
                nt::zero_grads(c, &snet);
                gscale = (gscale / 2.0).max(1.0);
                skipped += 1;
                eprintln!("\n  non-finite step (loss {:.4}, |g| {gn:.3e}, {} non-finite sign gradients): gscale -> {gscale}", st.loss, st.g_nonfinite);
                continue;
            }
            done += 1;
            let lr = o.lr * shape(done);
            let gmul = if gn > o.clip { (o.clip / gn) as f32 } else { 1.0 };
            nt::adam(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul).ok_or_else(|| anyhow::anyhow!("adam failed"))?;
            step_secs += tstep.elapsed().as_secs_f64();
            // CMF_NVG_PROF: resolve this step's timestamps (outside the timer)
            nt::prof::step_end();
            let mut sign_note = String::new();
            if let (Some(s), Some(cf)) = (&o.signs, &cfg) {
                let rate = st.flips as f64 / live.max(1) as f64;
                sign_note = format!("  η {:.2e} flips {} ({:.1e})", cf.eta, st.flips, rate);
                if rate > s.max_step_flip {
                    hi_streak += 1;
                    if hi_streak >= 3 {
                        eta_mul *= 0.5;
                        hi_streak = 0;
                        eprintln!("\n  guard: flip rate {rate:.2e} > {:.1e} for 3 steps — sign η halved (×{eta_mul})", s.max_step_flip);
                    }
                } else {
                    hi_streak = 0;
                }
            }
            eprint!(
                "\r  epoch {} step {done}/{total_steps}  loss {:.4}  |g| {:.3e}  lr {:.2e}{sign_note}  {:.2} s/step   ",
                ep + 1,
                st.loss,
                gn,
                lr,
                step_secs / (done - done0).max(1) as f64
            );
            if done % o.eval_every == 0 || done == total_steps {
                let e = eval_loss(&snet)?;
                eprintln!("\n  step {done}: held-out KL {e:.5} (best {:.5} at {})", best.0.min(e), if e < best.0 { done } else { best.2 });
                let mut rec = serde_json::json!({ "step": done, "heldout_kl": e });
                if let Some(s) = &o.signs {
                    let stats = nt::sign_stats(c, &snet).ok_or_else(|| anyhow::anyhow!("sign stats"))?;
                    let distinct: u64 = stats.iter().map(|x| x.1).sum();
                    last_distinct = distinct;
                    let frac = distinct as f64 / live.max(1) as f64;
                    let roles: Vec<String> = stats.iter().map(|(n, f, l)| format!("{n} {:.2}%", 100.0 * *f as f64 / (*l).max(1) as f64)).collect();
                    eprintln!(
                        "    signs: {:.3}% flipped ({distinct}), churn {:.2}, saturated {:.2e}, clamped {:.2e} of sign updates; {}",
                        100.0 * frac,
                        step_flips_total as f64 / distinct.max(1) as f64,
                        sat_total as f64 / (live as f64 * sign_steps.max(1) as f64),
                        clamp_total as f64 / (live as f64 * sign_steps.max(1) as f64),
                        roles.join(", ")
                    );
                    rec["flipped_fraction"] = serde_json::json!(frac);
                    rec["churn"] = serde_json::json!(step_flips_total as f64 / distinct.max(1) as f64);
                    if frac > s.max_total_flip {
                        eta_mul *= 0.5;
                        eprintln!("    guard: {:.2}% of signs flipped > {:.1}% — sign η halved (×{eta_mul})", 100.0 * frac, 100.0 * s.max_total_flip);
                    }
                    if e > 1.05 * best.0 {
                        bad_evals += 1;
                        if bad_evals >= 2 {
                            eta_mul *= 0.5;
                            bad_evals = 0;
                            eprintln!("    guard: held-out KL above 1.05× best twice — sign η halved (×{eta_mul})");
                        }
                    } else {
                        bad_evals = 0;
                    }
                }
                eval_log.push(rec);
                if e < best.0 {
                    best = (e, nt::snapshot(c, &snet, train_signs).ok_or_else(|| anyhow::anyhow!("snapshot"))?, done);
                }
            }
            if let Some(dir) = o.state_dir
                && o.state_every > 0
                && done % o.state_every == 0
                && done < total_steps
            {
                let ws = Instant::now();
                std::fs::create_dir_all(dir)?;
                let (sf, bf) = (format!("state-{done}.bin"), format!("best-{done}.bin"));
                if o.state_replace {
                    let old: Option<serde_json::Value> = std::fs::read(format!("{dir}/meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
                    if let Some(f) = old.as_ref().and_then(|m| m["state"].as_str()) {
                        let _ = std::fs::remove_file(format!("{dir}/{f}"));
                    }
                }
                let nb = nt::save_state(c, &snet, std::path::Path::new(&format!("{dir}/{sf}"))).map_err(|e| anyhow::anyhow!(e))?;
                nt::save_snap(&best.1, std::path::Path::new(&format!("{dir}/{bf}"))).map_err(|e| anyhow::anyhow!(e))?;
                let m = serde_json::json!({
                    "state": sf, "best": bf, "done": done, "epoch": ep, "consumed": pos + 1, "skipped": skipped,
                    "sign_steps": sign_steps, "gscale": gscale, "eta_mul": eta_mul, "hi_streak": hi_streak, "bad_evals": bad_evals,
                    "step_flips_total": step_flips_total, "sat_total": sat_total, "clamp_total": clamp_total,
                    "best_kl": best.0, "best_step": best.2, "e0": e0, "evals": eval_log.clone(),
                });
                let tmp = format!("{dir}/meta.json.tmp");
                std::fs::write(&tmp, serde_json::to_vec_pretty(&m)?)?;
                let old: Option<serde_json::Value> = std::fs::read(format!("{dir}/meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
                std::fs::rename(&tmp, format!("{dir}/meta.json"))?;
                if let Some(old) = old {
                    for key in ["state", "best"] {
                        if let Some(f) = old[key].as_str()
                            && f != m[key].as_str().unwrap_or("")
                        {
                            let _ = std::fs::remove_file(format!("{dir}/{f}"));
                        }
                    }
                }
                eprintln!("\n  state saved: {dir}/{sf} ({:.2} GB, {:.0} s)", nb as f64 / 1e9, ws.elapsed().as_secs_f64());
            }
        }
    }
    let train_secs = ts0.elapsed().as_secs_f64();
    eprintln!(
        "\nbest held-out KL {:.5} at step {} (start {e0:.5}); {skipped} non-finite steps skipped; {:.2} s/step over {done} steps ({:.0} s training incl. evals)",
        best.0,
        best.2,
        step_secs / (done - done0).max(1) as f64,
        train_secs
    );
    nt::prof::report();
    nt::restore(c, &snet, &best.1);
    let mut sign_prov = serde_json::Value::Null;
    if let Some(s) = &o.signs {
        let stats = nt::sign_stats(c, &snet).ok_or_else(|| anyhow::anyhow!("sign stats"))?;
        let distinct: u64 = stats.iter().map(|x| x.1).sum();
        let per_role: serde_json::Map<String, serde_json::Value> = stats.iter().map(|(n, f, l)| (n.clone(), serde_json::json!(*f as f64 / (*l).max(1) as f64))).collect();
        eprintln!("exported bits: {:.3}% of {live} signs flipped (last eval {last_distinct})", 100.0 * distinct as f64 / live.max(1) as f64);
        sign_prov = serde_json::json!({
            "opt": s.opt.name(),
            "tau": s.tau,
            "eta0": eta0,
            "eta_final_multiplier": eta_mul,
            "steps": sign_steps,
            "warm": s.warm,
            "flipped_fraction": distinct as f64 / live.max(1) as f64,
            "per_role": per_role,
        });
    }
    let patched = nt::export(c, &snet, &smodel).map_err(|e| anyhow::anyhow!(e))?;
    let map: std::collections::HashMap<String, Vec<u8>> = patched.into_iter().collect();
    let specs: Vec<TensorSpec> = smodel
        .tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: map.get(&e.name).cloned().unwrap_or_else(|| smodel.entry_bytes(e).to_vec()),
        })
        .collect();
    let mut header = smodel.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    let mut nv = serde_json::json!({
        "teacher": std::path::Path::new(o.teacher).file_name().map(|f| f.to_string_lossy().into_owned()),
        "tokens": train.len() * o.window * o.epochs,
        "window": o.window,
        "batch": o.batch,
        "lr": o.lr,
        "topk": k,
        "steps": done,
        "best_step": best.2,
        "heldout_kl_start": e0,
        "heldout_kl_best": best.0,
    });
    if !sign_prov.is_null() {
        nv["signs"] = sign_prov;
    }
    nv["evals"] = serde_json::Value::Array(eval_log);
    prov["nvg_train"] = nv;
    header.provenance = Some(prov);
    CmfModel::write(o.output, &header, &specs, None, smodel.vocab.as_deref()).map_err(|e| anyhow::anyhow!("write {}: {e}", o.output))?;
    eprintln!("wrote {} in {:.0}s", o.output, t0.elapsed().as_secs_f64());
    Ok(())
}
