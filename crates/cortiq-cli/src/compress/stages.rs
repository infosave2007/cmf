//! The stage bodies of `cortiq compress` (each runs in its own process).

use super::cmfio::{self, Encoding, Replacement};
use super::data::{Data, LossOn, Stream};
#[cfg(feature = "gpu")]
use super::heal_gpu;
use super::illum;
use super::plan::{self, PlanInput, Target};
use super::{Run, STAGES, stage_index};
use crate::convert::Quant;
use anyhow::{Context, Result, bail, ensure};
use cortiq_core::CmfModel;
use cortiq_core::types::TensorDtype;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn run_stage(run: &Run, stage: &str) -> Result<()> {
    let moe = run.done(0).and_then(|v| v.get("moe").cloned()).is_some_and(|v| !v.is_null());
    match stage_index(stage)? {
        0 => s0_prepare(run),
        1 if moe => super::moe::s1_routing(run),
        1 => s1_illuminate(run),
        2 if moe => super::moe::s2_plan(run),
        2 => s2_plan(run),
        3 if moe => super::moe::s3_drop(run),
        3 => s3_fold(run),
        4 if moe => {
            run.log("S4: no heal for a MoE model yet — the expert drop goes to the output as is");
            let s3 = run.require(3)?;
            run.mark_done(4, json!({ "f16": s3["f16"], "q8": s3["q8"], "healed": false }))
        }
        4 => s4_heal(run),
        5 => s5_quant(run),
        6 => super::eval::s6_eval(run),
        _ => unreachable!(),
    }
}

pub(super) fn path_of(v: &Value, key: &str) -> Option<PathBuf> {
    v.get(key).and_then(Value::as_str).map(PathBuf::from)
}

pub(super) fn open(p: &Path) -> Result<Arc<CmfModel>> {
    Ok(Arc::new(CmfModel::open_sharded(p).with_context(|| format!("opening {}", p.display()))?))
}

/// Memory this process may use: physical RAM, capped by a cgroup limit
/// (in a container `free` reports the host, not the budget).
pub fn mem_budget() -> u64 {
    let phys = cortiq_engine::fcd::physical_total_bytes().unwrap_or(16 << 30);
    let cg = ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .min();
    cg.map_or(phys, |c| c.min(phys))
}

/// CPU threads this process may use: the cores, capped by a cgroup CPU
/// quota (a container sees the host's cores — 128 on a pod with a 13.6-core
/// quota, where 32 threads only add contention).
pub fn cpu_budget() -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let quota = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok().and_then(|s| {
        let mut it = s.split_whitespace();
        let q: f64 = it.next()?.parse().ok()?;
        let p: f64 = it.next()?.parse().ok()?;
        Some((q / p).floor().max(1.0) as usize)
    });
    quota.map_or(cores, |q| q.min(cores))
}

/// Trunk layers with a dense FFN: (hidden, width per layer).
fn ffn_layout(m: &CmfModel) -> Result<(usize, Vec<usize>)> {
    let arch = m.arch();
    let mut widths = Vec::with_capacity(arch.num_layers);
    for l in 0..arch.num_layers {
        let e = m.tensor(&illum::down_name(l)).with_context(|| {
            format!(
                "layer {l} has no dense FFN (model.layers.{l}.mlp.down_proj.weight) — structural compression needs dense FFNs; \
                 for MoE use --recipe quant-only"
            )
        })?;
        ensure!(e.shape.len() == 2 && e.shape[0] == arch.hidden_size, "layer {l}: down_proj shape {:?}", e.shape);
        widths.push(e.shape[1]);
    }
    Ok((arch.hidden_size, widths))
}

fn params_of(m: &CmfModel) -> (u64, u64) {
    let (mut all, mut mtp) = (0u64, 0u64);
    for e in &m.tensors {
        let n = e.n_elems() as u64;
        all += n;
        if e.name.starts_with("model.mtp.") {
            mtp += n;
        }
    }
    (all - mtp, mtp)
}

/// Dominant dtype of the 2-D weights (by elements).
fn dominant_dtype(m: &CmfModel) -> TensorDtype {
    let mut by: Vec<(TensorDtype, u64)> = Vec::new();
    for e in m.tensors.iter().filter(|e| e.shape.len() == 2) {
        match by.iter_mut().find(|(d, _)| *d == e.dtype) {
            Some((_, n)) => *n += e.n_elems() as u64,
            None => by.push((e.dtype, e.n_elems() as u64)),
        }
    }
    by.into_iter().max_by_key(|&(_, n)| n).map(|(d, _)| d).unwrap_or(TensorDtype::F32)
}

fn set_quant_type(h: &mut cortiq_core::format::CmfHeader, q: Quant) {
    use cortiq_core::types::QuantType;
    h.quant_type = match q {
        Quant::F16 => QuantType::F16,
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::Vbit => QuantType::Vbit,
        _ => QuantType::Q8_2f,
    };
}

// ───────────────────────────── S0 prepare ─────────────────────────────

/// The task's teacher data: the source model answers every prompt (its
/// 8-bit copy runs them — the answers are text, so its rounding hardly
/// matters), split 80/10/10 by category. An interrupted generation is
/// continued by the next run; a finished one is reused.
fn generate_task_data(run: &Run, prompts: &str, teacher: &Path) -> Result<()> {
    let a = &run.a;
    if Data::load(&a.data).is_ok() {
        return Ok(());
    }
    // `--teacher`: another local model, or an API model whose answers the
    // source's tokenizer turns into ids; default: the source itself
    let api = a.teacher.as_deref().is_some_and(crate::teacher_api::is_api_spec);
    let teacher_spec = a.teacher.clone().unwrap_or_else(|| teacher.display().to_string());
    let sampling = if api {
        // a reasoning model loops greedily: the Qwen recommendation
        match a.gen_sampling.as_str() {
            "auto" | "file" => "temperature=0.6,top_p=0.95,top_k=20".to_string(),
            s => s.to_string(),
        }
    } else {
        let m = open(Path::new(&teacher_spec))?;
        let s = match a.gen_sampling.as_str() {
            "auto" if m.header.provenance.as_ref().and_then(|p| p.get("sampling")).is_some() => "file".to_string(),
            "auto" => "greedy".to_string(),
            s => s.to_string(),
        };
        s
    };
    let who = match &a.teacher {
        Some(t) => format!("teacher {t}"),
        None => "the source".to_string(),
    };
    run.log(&format!(
        "S0: {who} answers the prompts of {prompts} (≤{} tokens each, {} per prompt, {sampling} decoding{}) → {}",
        a.gen_max_tokens(),
        a.gen_samples,
        if a.no_think { ", no reasoning" } else { "" },
        a.data
    ));
    crate::cmd_teacher_data_generate(
        &teacher_spec,
        prompts,
        &a.data,
        crate::TeacherDataGenerateOptions {
            max_tokens: a.gen_max_tokens(),
            samples: a.gen_samples,
            seed: 42,
            split_seed: 42,
            split: "80/10/10".into(),
            topk: 0,
            raw: false,
            no_think: a.no_think,
            keep_truncated: true,
            disable_mtp: false,
            limit: None,
            shard: None,
            sampling,
            tokenizer: Some(teacher.display().to_string()),
            api: a.api.clone(),
        },
    )
}

fn s0_prepare(run: &Run) -> Result<()> {
    let a = &run.a;
    let dir = run.stage_dir(0);
    std::fs::create_dir_all(&dir)?;
    let src = Path::new(&a.source);
    let (f16, q8, kind): (Option<PathBuf>, PathBuf, &str) = if src.is_file() {
        ensure!(a.source.ends_with(".cmf"), "SOURCE {} is a file but not a .cmf", a.source);
        let m = open(src)?;
        match dominant_dtype(&m) {
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 => {
                let q8 = dir.join("src-q8_2f.cmf");
                if !q8.exists() {
                    println!("S0: q8_2f calibration copy of {} …", a.source);
                    cmfio::rewrite(&m, &q8, Encoding::Profile(Quant::Q8_2f), &|_| Ok(None), &|h| set_quant_type(h, Quant::Q8_2f), run.threads)?;
                }
                (Some(src.to_path_buf()), q8, "cmf-float")
            }
            TensorDtype::Q8_2f | TensorDtype::Q8Row => {
                eprintln!(
                    "S0: warning: the source is 8-bit — the fold and the heal start from its rounded weights \
                     (an f16/bf16 source or an HF checkpoint is exact)"
                );
                (None, src.to_path_buf(), "cmf-q8")
            }
            d => {
                ensure!(
                    a.allow_lossy_source || !run.structural,
                    "the source is {d:?}: folding on 4-bit (or lower) weights adds their rounding to the fold's error; \
                     give an f16/bf16/q8 source or an HF checkpoint, or --allow-lossy-source"
                );
                eprintln!("S0: warning: lossy {d:?} source");
                (None, src.to_path_buf(), "cmf-lossy")
            }
        }
    } else {
        // HF directory or hub id: one streamed conversion, two outputs.
        let f16 = dir.join("src-f16.cmf");
        let q8 = dir.join("src-q8_2f.cmf");
        if !(f16.exists() && q8.exists()) {
            println!("S0: converting {} → f16 + q8_2f …", a.source);
            let st = std::process::Command::new(std::env::current_exe()?)
                .args(["convert", "--model", &a.source, "--quant", "f16", "--output"])
                .arg(&f16)
                .arg("--also-output")
                .arg(format!("q8_2f={}", q8.display()))
                .status()?;
            ensure!(st.success(), "S0: cortiq convert failed ({st})");
        }
        (Some(f16), q8, "hf")
    };
    if let Some(prompts) = &run.prompts {
        generate_task_data(run, prompts, &q8)?;
    }
    let m = open(&q8)?;
    let data = Data::load(&a.data)?;
    data.check_tokenizer(&m, "source")?;
    let (params, mtp_params) = params_of(&m);
    let mut info = json!({
        "kind": kind,
        "f16": f16.as_ref().map(|p| p.display().to_string()),
        "q8": q8.display().to_string(),
        "arch": m.arch().arch_name,
        "params": params,
        "mtp_params": mtp_params,
        "data_records": data.recs.len(),
        "data_rejected": data.rejected,
        "data_teacher": data.teacher_id(),
    });
    if run.structural && m.arch().moe.is_some() {
        info["moe"] = super::moe::layout(&m)?;
    } else if run.structural {
        let (hidden, widths) = ffn_layout(&m)?;
        let ffn: u64 = widths.iter().map(|&w| 3 * hidden as u64 * w as u64).sum();
        info["hidden"] = json!(hidden);
        info["widths"] = json!(widths);
        info["other_params"] = json!(params - ffn);
    }
    let (tr, va, te) = (
        data.count(crate::tdata::TdataSplit::Train),
        data.count(crate::tdata::TdataSplit::Val),
        data.count(crate::tdata::TdataSplit::Test),
    );
    info["splits"] = json!({"train": tr, "val": va, "test": te});
    run.log(&format!(
        "S0: {} ({}, {:.2}B params{}) · data {} usable records ({} rejected): train {}/{} tok, val {}/{}, test {}/{}",
        a.source,
        m.arch().arch_name,
        params as f64 / 1e9,
        if mtp_params > 0 { format!(" + MTP {:.2}B", mtp_params as f64 / 1e9) } else { String::new() },
        data.recs.len(),
        data.rejected,
        tr.0, tr.1, va.0, va.1, te.0, te.1
    ));
    ensure!(te.0 > 0, "the data has no test records — the result could not be evaluated honestly (resplit the data)");
    if run.structural {
        ensure!(tr.0 > 0 && va.0 > 0, "structural compression needs train and val records");
    }
    run.mark_done(0, info)
}

// ─────────────────────────── S1 illuminate ────────────────────────────

/// Gram storage of S1 (f32 full matrices) for S3 to reuse.
fn gram_path(run: &Run, layer: usize, split: &str) -> PathBuf {
    run.stage_dir(1).join("grams").join(format!("L{layer}.{split}.f32"))
}

fn save_gram(p: &Path, g: &[f64]) -> Result<()> {
    std::fs::create_dir_all(p.parent().unwrap())?;
    let bytes: Vec<u8> = g.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
    std::fs::write(p, bytes)?;
    Ok(())
}

fn load_gram(p: &Path, n: usize) -> Result<Vec<f64>> {
    let b = std::fs::read(p)?;
    ensure!(b.len() == n * n * 4, "{}: {} bytes, expected {}", p.display(), b.len(), n * n * 4);
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect())
}

fn write_u32s(p: &Path, rows: &[Vec<usize>]) -> Result<()> {
    let mut b = Vec::new();
    for r in rows {
        b.extend((r.len() as u32).to_le_bytes());
        for &v in r {
            b.extend((v as u32).to_le_bytes());
        }
    }
    std::fs::write(p, b)?;
    Ok(())
}

fn read_u32s(p: &Path) -> Result<Vec<Vec<usize>>> {
    let b = std::fs::read(p)?;
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let n = u32::from_le_bytes(b[i..i + 4].try_into()?) as usize;
        i += 4;
        out.push((0..n).map(|k| u32::from_le_bytes(b[i + 4 * k..i + 4 * k + 4].try_into().unwrap()) as usize).collect());
        i += 4 * n;
    }
    Ok(out)
}

fn write_f64s(p: &Path, rows: &[Vec<f64>]) -> Result<()> {
    let mut b = Vec::new();
    for r in rows {
        b.extend((r.len() as u32).to_le_bytes());
        for &v in r {
            b.extend(v.to_le_bytes());
        }
    }
    std::fs::write(p, b)?;
    Ok(())
}

fn read_f64s(p: &Path) -> Result<Vec<Vec<f64>>> {
    let b = std::fs::read(p)?;
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let n = u32::from_le_bytes(b[i..i + 4].try_into()?) as usize;
        i += 4;
        out.push((0..n).map(|k| f64::from_le_bytes(b[i + 8 * k..i + 8 * k + 8].try_into().unwrap())).collect());
        i += 8 * n;
    }
    Ok(out)
}

/// Layers per capture pass: the host Grams (`8·n²` bytes per layer) of one
/// split within a third of the memory budget.
fn chunk_layers(widths: &[usize]) -> usize {
    let per = widths.iter().map(|&w| 8 * w * w).max().unwrap_or(1) as u64;
    ((mem_budget() / 3) / per.max(1)).clamp(1, widths.len() as u64) as usize
}

/// What runs the illumination forward: the runtime with its capture hook
/// (on the q8_2f sibling), or the trainer's kernels on the f16 source.
enum Capturer {
    Runtime(cortiq_engine::Pipeline),
    #[cfg(feature = "gpu")]
    Device(heal_gpu::DevCapture),
}

struct Calib {
    cap: Capturer,
    train: Vec<Vec<u32>>,
    val: Vec<Vec<u32>>,
    tokens: (usize, usize),
}

impl Calib {
    fn capture(&mut self, val: bool, layers: &[usize], label: &str) -> Result<BTreeMap<usize, cortiq_engine::gptq_capture::HessianAcc>> {
        let w = if val { &self.val } else { &self.train };
        match &mut self.cap {
            Capturer::Runtime(p) => illum::capture(p, w, layers, label),
            #[cfg(feature = "gpu")]
            Capturer::Device(d) => d.capture(w, layers, label),
        }
    }
}

/// `CMF_ILLUM_ON=runtime` keeps the illumination on the runtime's hook.
fn capturer(run: &Run, q8: &Path, f16: Option<&Path>) -> Result<Capturer> {
    #[cfg(feature = "gpu")]
    if let Some(f) = f16
        && std::env::var("CMF_ILLUM_ON").as_deref() != Ok("runtime")
    {
        let m = open(f)?;
        if heal_gpu::usable(&m).is_ok() {
            drop(m);
            run.log("  illumination forward on the device (trainer kernels, f16 source)");
            return Ok(Capturer::Device(heal_gpu::DevCapture::new(f, run.a.window)?));
        }
    }
    let _ = (run, f16);
    let m = open(q8)?;
    let mut pipe = cortiq_engine::Pipeline::from_model(&m, cortiq_engine::SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    Ok(Capturer::Runtime(pipe))
}

fn calib(run: &Run, q8: &Path, f16: Option<&Path>, data: &Data, max_width: usize) -> Result<Calib> {
    use crate::tdata::TdataSplit;
    let cap = capturer(run, q8, f16)?;
    let budget = match run.a.illum_tokens {
        Some(0) => None,
        Some(n) => Some(n),
        None => Some(20 * max_width),
    };
    // A budget takes records in a fixed shuffled order (by record id): the
    // file groups prompts by category, and its first tokens would light
    // only the first categories.
    let shuffled = |sp: TdataSplit| {
        let mut v: Vec<&super::data::Rec> = data.split(sp).collect();
        v.sort_by_cached_key(|r| crate::knowledge::sha256_hex(r.rid.as_bytes()));
        v
    };
    let (train, nt) = illum::windows(shuffled(TdataSplit::Train).into_iter(), run.a.window, budget);
    let (val, nv) = illum::windows(shuffled(TdataSplit::Val).into_iter(), run.a.window, budget.map(|b| (b / 4).max(run.a.window)));
    let all_t: usize = data.split(TdataSplit::Train).map(|r| r.ids.len()).sum();
    let all_v: usize = data.split(TdataSplit::Val).map(|r| r.ids.len()).sum();
    run.log(&format!(
        "  illumination: train {} windows / {nt} tokens (records hold {all_t}), val {} / {nv} (records {all_v}); window {}, long records overlap by half",
        train.len(),
        val.len(),
        run.a.window
    ));
    Ok(Calib { cap, train, val, tokens: (nt, nv) })
}

fn s1_illuminate(run: &Run) -> Result<()> {
    let s0 = run.require(0)?;
    let data = Data::load(&run.a.data)?;
    let q8 = path_of(&s0, "q8").unwrap();
    let wsrc = open(&path_of(&s0, "f16").unwrap_or(q8.clone()))?;
    let widths: Vec<usize> = serde_json::from_value(s0["widths"].clone())?;
    let hidden = s0["hidden"].as_u64().unwrap() as usize;
    let f16 = path_of(&s0, "f16");
    let mut cal = calib(run, &q8, f16.as_deref(), &data, widths.iter().copied().max().unwrap_or(0))?;
    let nl = widths.len();
    let chunk = chunk_layers(&widths);
    let gram_bytes: u64 = widths.iter().map(|&w| 4 * w as u64 * w as u64 * 2).sum();
    let keep_grams = gram_bytes <= 16 << 30 && gram_bytes * 3 < disk_free(&run.work);
    run.log(&format!(
        "S1: {nl} layers, {chunk} per capture pass; Grams {:.1} GB {}",
        gram_bytes as f64 / 1e9,
        if keep_grams { "kept for the fold" } else { "recomputed by the fold" }
    ));
    let (mut curves, mut orders, mut energy) = (vec![Vec::new(); nl], vec![Vec::new(); nl], vec![0f64; nl]);
    let layers: Vec<usize> = (0..nl).collect();
    for part in layers.chunks(chunk) {
        let tr = cal.capture(false, part, &format!("train L{}-{}", part[0], part[part.len() - 1]))?;
        for (&l, acc) in &tr {
            let n = widths[l];
            let w = cmfio::tensor_f32(&wsrc, wsrc.tensor(&illum::down_name(l)).unwrap())?;
            let imp = illum::importance(&w, hidden, n, &acc.sumsq, acc.count);
            let ord = illum::order(&imp);
            curves[l] = illum::refit_curve(&acc.h, n, &w, hidden, &ord, run.threads)?;
            energy[l] = acc.sumsq.iter().map(|s| (s / acc.count.max(1) as f64).sqrt()).sum();
            orders[l] = ord;
            if keep_grams {
                save_gram(&gram_path(run, l, "train"), &acc.h)?;
            }
        }
        drop(tr);
        if keep_grams {
            let va = cal.capture(true, part, &format!("val L{}-{}", part[0], part[part.len() - 1]))?;
            for (&l, acc) in &va {
                save_gram(&gram_path(run, l, "val"), &acc.h)?;
            }
        }
        eprintln!("  layers {}..={} illuminated", part[0], part[part.len() - 1]);
    }
    let dir = run.stage_dir(1);
    std::fs::create_dir_all(&dir)?;
    write_f64s(&dir.join("curves.bin"), &curves)?;
    write_u32s(&dir.join("order.bin"), &orders)?;
    let mut by: Vec<(usize, f64)> = energy.iter().copied().enumerate().collect();
    by.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let fmt = |v: &[(usize, f64)]| v.iter().map(|(l, e)| format!("L{l} {e:.1}")).collect::<Vec<_>>().join(", ");
    run.log(&format!("S1: layer energy top-5: {} · bottom-5: {}", fmt(&by[..5.min(nl)]), fmt(&by[nl.saturating_sub(5)..])));
    // the refit cost of a uniform 75% / 50% cut, as a quick read of the curves
    let at = |f: f64| -> f64 { (0..nl).map(|l| curves[l][(widths[l] as f64 * f) as usize]).sum::<f64>() / nl as f64 };
    run.log(&format!(
        "S1: mean relative read-out error after refit — 75% width {:.4}, 50% width {:.4} (train Grams)",
        at(0.75),
        at(0.5)
    ));
    run.mark_done(
        1,
        json!({
            "curves": dir.join("curves.bin"), "orders": dir.join("order.bin"),
            "energy": energy, "grams": keep_grams,
            "train_tokens": cal.tokens.0, "val_tokens": cal.tokens.1,
        }),
    )
}

fn disk_free(p: &Path) -> u64 {
    let out = std::process::Command::new("df").arg("-k").arg(p).output();
    out.ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .nth(1)
                .and_then(|l| l.split_whitespace().nth(3).and_then(|v| v.parse::<u64>().ok()))
        })
        .map_or(u64::MAX, |kb| kb * 1024)
}

// ────────────────────────────── S2 plan ───────────────────────────────

fn s2_plan(run: &Run) -> Result<()> {
    let a = &run.a;
    let s0 = run.require(0)?;
    let s1 = run.require(1)?;
    let widths: Vec<usize> = serde_json::from_value(s0["widths"].clone())?;
    let hidden = s0["hidden"].as_u64().unwrap() as usize;
    let mut curves = read_f64s(&path_of(&s1, "curves").unwrap())?;
    let nl = widths.len();
    // curves carry the layer's output energy as one trailing entry (older S1
    // results do not): `abs` weighs each layer's relative error by it, so a
    // layer whose FFN writes more into the residual stream is cut less
    let energy: Option<Vec<f64>> = curves
        .iter()
        .zip(&widths)
        .map(|(c, &w)| (c.len() == w + 2).then(|| c[w + 1]))
        .collect();
    for (c, &w) in curves.iter_mut().zip(&widths) {
        c.truncate(w + 1);
    }
    let protect: Vec<bool> = (0..nl).map(|l| l < a.protect_first || l + a.protect_last >= nl).collect();
    let target = target_of(a)?;
    ensure!(matches!(a.allocate.as_str(), "curve" | "abs" | "uniform"), "--allocate: curve, abs or uniform");
    if a.allocate == "abs" {
        let e = energy.as_ref().context("--allocate abs needs an S1 run that records layer output energies (rerun S1)")?;
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        for (c, &el) in curves.iter_mut().zip(e) {
            c.iter_mut().for_each(|v| *v *= el / mean);
        }
    }
    let p = plan::allocate(
        &PlanInput {
            hidden,
            widths: &widths,
            curves: Some(&curves),
            protect: &protect,
            align: a.align,
            min_width: a.min_width,
            uniform: a.allocate == "uniform",
            other_params: s0["other_params"].as_u64().unwrap(),
            train_tokens: s1["train_tokens"].as_u64().unwrap_or(0) as usize,
        },
        &target,
    )?;
    let mut warnings = p.warnings.clone();
    let thin: Vec<usize> = p.layers.iter().filter(|l| l.keep < l.width && l.tokens_per_unit < 10.0).map(|l| l.layer).collect();
    if !thin.is_empty() {
        warnings.push(format!(
            "layers {thin:?}: fewer than 10 train tokens per kept neuron — the refit there is chosen on the val split (measured: too few tokens make it overfit)"
        ));
    }
    let mut p = p;
    let mut orders_out: Option<PathBuf> = None;
    match a.select.as_str() {
        "importance" => {}
        "dtgma" => {
            // the same total, distributed and ordered by the trained mask
            let total: usize = p.layers.iter().map(|l| l.keep).sum();
            let keep_frac = total as f64 / widths.iter().sum::<usize>() as f64;
            let data = Data::load(&a.data)?;
            let model = open(&path_of(&s0, "f16").or_else(|| path_of(&s0, "q8")).unwrap())?;
            let logits = super::dtgma::learn(run, &data, &model, keep_frac)?;
            ensure!(logits.len() == nl && logits.iter().zip(&widths).all(|(l, &w)| l.len() == w), "DTG-MA logits do not match the FFN widths");
            let floor: Vec<usize> = widths.iter().zip(&protect).map(|(&w, &pr)| if pr { w } else { ((w as f64 * a.min_width) as usize).div_ceil(a.align.max(1)) * a.align.max(1) }).collect();
            let keeps = super::dtgma::keeps_from(&logits, &widths, total, a.align, &floor);
            for (l, k) in p.layers.iter_mut().zip(&keeps) {
                l.keep = *k;
            }
            let ffn_after: u64 = keeps.iter().map(|&k| 3 * hidden as u64 * k as u64).sum();
            p.params_after = p.params_after - p.ffn_params_after + ffn_after;
            p.ffn_params_after = ffn_after;
            let op = run.stage_dir(2).join("order.bin");
            std::fs::create_dir_all(op.parent().unwrap())?;
            write_u32s(&op, &super::dtgma::orders_from(&logits))?;
            orders_out = Some(op);
        }
        o => bail!("--select {o:?}: importance or dtgma"),
    }
    let path = run.stage_dir(2).join("plan.json");
    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut pv = serde_json::to_value(&p)?;
    pv["warnings"] = json!(warnings);
    std::fs::write(&path, serde_json::to_vec_pretty(&pv)?)?;
    run.mark_done(2, json!({ "plan": path, "params_after": p.params_after, "params_before": p.params_before, "orders": orders_out }))
}

/// `--plan-only`: the plan arithmetic before any data pass — what a uniform
/// cut to the target gives (the data-driven widths come from S1/S2:
/// `--until plan`).
pub fn print_arithmetic_plan(run: &Run) -> Result<()> {
    let a = &run.a;
    let s0 = run.require(0)?;
    if !run.structural {
        run.log(&format!("plan: quant-only — {:.3}B parameters re-encoded as {}", s0["params"].as_f64().unwrap_or(0.0) / 1e9, crate::convert::quant_name(run.quant)));
        return Ok(());
    }
    if !s0["moe"].is_null() {
        run.log("plan: a MoE model — the experts kept depend on the routing counts of S1 (run without --plan-only)");
        return Ok(());
    }
    let widths: Vec<usize> = serde_json::from_value(s0["widths"].clone())?;
    let hidden = s0["hidden"].as_u64().unwrap() as usize;
    let nl = widths.len();
    let protect: Vec<bool> = (0..nl).map(|l| l < a.protect_first || l + a.protect_last >= nl).collect();
    let target = target_of(a)?;
    let p = plan::allocate(
        &PlanInput {
            hidden,
            widths: &widths,
            curves: None,
            protect: &protect,
            align: a.align,
            min_width: a.min_width,
            uniform: true,
            other_params: s0["other_params"].as_u64().unwrap(),
            train_tokens: s0["splits"]["train"][1].as_u64().unwrap_or(0) as usize,
        },
        &target,
    )?;
    let q8 = std::fs::metadata(path_of(&s0, "q8").unwrap()).map(|m| m.len()).unwrap_or(0) as f64;
    let ratio = p.params_after as f64 / p.params_before as f64;
    run.log(&format!(
        "plan (arithmetic, uniform cut): params {:.3}B → {:.3}B; FFN width {} → {} on {} of {nl} layers (protected: first {}, last {});\n  \
         q8_2f size ≈ {:.2} GB → {:.2} GB; the measured per-layer widths come from `--until plan`",
        p.params_before as f64 / 1e9,
        p.params_after as f64 / 1e9,
        widths.iter().max().unwrap_or(&0),
        p.layers.iter().filter(|l| !l.protected).map(|l| l.keep).max().unwrap_or(0),
        p.layers.iter().filter(|l| !l.protected).count(),
        a.protect_first,
        a.protect_last,
        q8 / 1e9,
        q8 * ratio / 1e9
    ));
    for w in &p.warnings {
        run.log(&format!("  warning: {w}"));
    }
    Ok(())
}

fn target_of(a: &super::CompressArgs) -> Result<Target> {
    Ok(if let Some(f) = &a.ffn {
        let v: f64 = f.parse().map_err(|_| anyhow::anyhow!("--ffn {f:?}: a fraction (0.75) or a width (12288)"))?;
        if v <= 1.0 { Target::Ffn(v) } else { Target::FfnWidth(v as usize) }
    } else if let Some(s) = &a.size {
        Target::Params(plan::parse_size(s)?)
    } else if let Some(r) = a.ratio {
        Target::Ratio(r)
    } else {
        bail!("no target (--ffn, --size or --ratio)");
    })
}

pub fn print_plan(run: &Run, s2: &Value) -> Result<()> {
    let p: Value = serde_json::from_slice(&std::fs::read(path_of(s2, "plan").unwrap())?)?;
    let layers = p["layers"].as_array().cloned().unwrap_or_default();
    let mut line = String::new();
    for l in &layers {
        line += &format!(
            "{}{}{} ",
            l["keep"].as_u64().unwrap_or(0),
            if l["protected"].as_bool() == Some(true) { "*" } else { "" },
            ""
        );
    }
    run.log(&format!(
        "plan: params {:.3}B → {:.3}B (FFN {:.3}B → {:.3}B); widths per layer (* = protected):\n  {}",
        p["params_before"].as_f64().unwrap_or(0.0) / 1e9,
        p["params_after"].as_f64().unwrap_or(0.0) / 1e9,
        p["ffn_params_before"].as_f64().unwrap_or(0.0) / 1e9,
        p["ffn_params_after"].as_f64().unwrap_or(0.0) / 1e9,
        line.trim_end()
    ));
    for w in p["warnings"].as_array().into_iter().flatten() {
        run.log(&format!("  warning: {}", w.as_str().unwrap_or("")));
    }
    Ok(())
}

// ────────────────────────────── S3 fold ───────────────────────────────

fn s3_fold(run: &Run) -> Result<()> {
    let a = &run.a;
    let s0 = run.require(0)?;
    let s1 = run.require(1)?;
    let s2 = run.require(2)?;
    let plan: Value = serde_json::from_slice(&std::fs::read(path_of(&s2, "plan").unwrap())?)?;
    let keeps: Vec<usize> = plan["layers"].as_array().unwrap().iter().map(|l| l["keep"].as_u64().unwrap() as usize).collect();
    let widths: Vec<usize> = serde_json::from_value(s0["widths"].clone())?;
    let hidden = s0["hidden"].as_u64().unwrap() as usize;
    // the selection's neuron order: the trained DTG-MA mask's when S2 made
    // one, else the illumination's
    let orders = read_u32s(&path_of(&s2, "orders").or_else(|| path_of(&s1, "orders")).unwrap())?;
    let q8 = path_of(&s0, "q8").unwrap();
    let wpath = path_of(&s0, "f16").unwrap_or(q8.clone());
    let wsrc = open(&wpath)?;
    let ridges: Vec<f64> = a
        .ridge
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|_| anyhow::anyhow!("--ridge {:?}", a.ridge)))
        .collect::<Result<_>>()?;
    let (ridges, allow_cut) = match a.refit.as_str() {
        "auto" => (ridges, true),
        "on" => (ridges, false),
        "off" => (Vec::new(), true),
        other => bail!("--refit {other:?}: auto, on or off"),
    };
    let nl = widths.len();
    let todo: Vec<usize> = (0..nl).filter(|&l| keeps[l] < widths[l]).collect();
    let grams_kept = s1["grams"].as_bool().unwrap_or(false);
    let fold_t0 = std::time::Instant::now();
    let max_w = widths.iter().copied().max().unwrap_or(0);
    let f16 = path_of(&s0, "f16");
    let mut cal = if grams_kept { None } else { Some(calib(run, &q8, f16.as_deref(), &Data::load(&a.data)?, max_w)?) };
    let mut repl: HashMap<String, Replacement> = HashMap::new();
    let mut infos: Vec<illum::FoldInfo> = Vec::new();
    let chunk = chunk_layers(&widths).max(1);
    for part in todo.chunks(chunk) {
        let (mut tr, mut va): (BTreeMap<usize, Vec<f64>>, BTreeMap<usize, Vec<f64>>) = (BTreeMap::new(), BTreeMap::new());
        if let Some(c) = cal.as_mut() {
            let label = format!("L{}-{}", part[0], part[part.len() - 1]);
            for (l, acc) in c.capture(false, part, &format!("train {label}"))? {
                tr.insert(l, acc.h);
            }
            for (l, acc) in c.capture(true, part, &format!("val {label}"))? {
                va.insert(l, acc.h);
            }
        } else {
            for &l in part {
                tr.insert(l, load_gram(&gram_path(run, l, "train"), widths[l])?);
                va.insert(l, load_gram(&gram_path(run, l, "val"), widths[l])?);
            }
        }
        for &l in part {
            let n = widths[l];
            let w = cmfio::tensor_f32(&wsrc, wsrc.tensor(&illum::down_name(l)).unwrap())?;
            let f = illum::fold_layer(l, &tr[&l], Some(&va[&l]), n, &w, hidden, &orders[l], keeps[l], &ridges, allow_cut, run.threads)?;
            let p = format!("model.layers.{l}.mlp.");
            for name in ["gate_proj", "up_proj"] {
                let e = wsrc.tensor(&format!("{p}{name}.weight")).with_context(|| format!("layer {l}: {p}{name}.weight"))?;
                let v = cmfio::tensor_f32(&wsrc, e)?;
                repl.insert(format!("{p}{name}.weight"), (vec![f.keep.len(), hidden], illum::rows(&v, hidden, &f.keep)));
                if let Some(b) = wsrc.tensor(&format!("{p}{name}.bias")) {
                    let bv = cmfio::tensor_f32(&wsrc, b)?;
                    repl.insert(format!("{p}{name}.bias"), (vec![f.keep.len()], f.keep.iter().map(|&i| bv[i]).collect()));
                }
            }
            repl.insert(format!("{p}down_proj.weight"), (vec![hidden, f.keep.len()], f.down));
            let k = infos.len() + 1;
            eprintln!(
                "  [{k}/{}] L{l}: {} → {} · val error: cut {:.4}, kept {:.4} ({}) · train {:.4} · ~{} left",
                todo.len(),
                n,
                f.keep.len(),
                f.info.val_err_cut,
                f.info.val_err,
                f.info.ridge.map_or("no refit".to_string(), |r| format!("refit ε={r}")),
                f.info.train_err,
                crate::fmt_eta(fold_t0.elapsed().as_secs_f64() / k as f64 * (todo.len() - k) as f64)
            );
            infos.push(f.info);
        }
    }
    let refit_n = infos.iter().filter(|i| i.ridge.is_some()).count();
    let mean = |f: &dyn Fn(&illum::FoldInfo) -> f64| infos.iter().map(f).sum::<f64>() / infos.len().max(1) as f64;
    run.log(&format!(
        "S3: {} layers narrowed, refit kept on {refit_n}; mean val read-out error {:.4} (plain cut would be {:.4})",
        infos.len(),
        mean(&|i| i.val_err),
        mean(&|i| i.val_err_cut)
    ));
    let max_keep = keeps.iter().copied().max().unwrap_or(0);
    // The MTP draft keeps its own FFN: record its width explicitly once the
    // trunk's intermediate_size no longer describes it.
    let mtp_widths: Option<Vec<usize>> = wsrc.arch().mtp.as_ref().map(|m| {
        (0..m.num_layers)
            .filter_map(|i| wsrc.tensor(&format!("model.mtp.layers.{i}.mlp.gate_proj.weight")).map(|e| e.shape[0]))
            .collect()
    });
    let fold_prov = json!({
        "source": a.source, "data": a.data, "widths": keeps, "refit": a.refit,
        "layers": infos,
    });
    let edit = |h: &mut cortiq_core::format::CmfHeader| {
        h.arch.intermediate_size = max_keep;
        if let (Some(m), Some(w)) = (h.arch.mtp.as_mut(), mtp_widths.as_ref())
            && w.len() == m.num_layers
            && !w.is_empty()
        {
            m.ffn_intermediate_sizes = Some(w.clone());
        }
        cmfio::provenance_set(h, "compress_fold", fold_prov.clone());
    };
    let dir = run.stage_dir(3);
    let f16_out = dir.join("fold-f16.cmf");
    let q8_out = dir.join("fold-q8_2f.cmf");
    let lookup = |name: &str| -> Result<Option<Replacement>> { Ok(repl.get(name).cloned()) };
    let b1 = cmfio::rewrite(&wsrc, &f16_out, Encoding::KeepF16, &lookup, &edit, run.threads)?;
    let b2 = cmfio::rewrite(&wsrc, &q8_out, Encoding::Profile(Quant::Q8_2f), &lookup, &|h| {
        edit(h);
        set_quant_type(h, Quant::Q8_2f);
    }, run.threads)?;
    run.log(&format!("S3: wrote {} ({:.2} GB) and its q8_2f copy ({:.2} GB)", f16_out.display(), b1 as f64 / 1e9, b2 as f64 / 1e9));
    run.mark_done(3, json!({ "f16": f16_out, "q8": q8_out, "layers": infos }))
}

// ────────────────────────────── S4 heal ───────────────────────────────

fn s4_heal(run: &Run) -> Result<()> {
    use cortiq_engine::fcd::{CacheModels, FcdHyper, FcdStream, GenGateCfg, run_polish_stream};
    use cortiq_engine::nystrom::O1Cfg;
    let a = &run.a;
    let s0 = run.require(0)?;
    let s3 = run.require(3)?;
    if !run.heal {
        run.log("S4: healing off — the fold goes to quantization as is");
        return run.mark_done(4, json!({ "f16": s3["f16"], "q8": s3["q8"], "healed": false }));
    }
    let data = Data::load(&a.data)?;
    let student = open(&path_of(&s3, "f16").unwrap())?;
    let teacher_path = path_of(&s0, "f16").unwrap_or(path_of(&s0, "q8").unwrap());
    #[cfg(feature = "gpu")]
    if a.heal_device != "cpu" {
        match heal_gpu::usable(&student) {
            Ok(()) => return s4_heal_device(run, &data, &student, &path_of(&s3, "f16").unwrap(), &teacher_path),
            Err(e) if a.heal_device == "gpu" => anyhow::bail!("--heal-device gpu: {e}"),
            Err(e) => run.log(&format!("S4: heal on the CPU ({e})")),
        }
    }
    anyhow::ensure!(a.heal_device != "gpu", "--heal-device gpu: this build has no GPU trainer");
    let teacher = open(&teacher_path)?;
    // the window cache runs the q8_2f copies: the device executes those
    let teacher_rt = open(&path_of(&s0, "q8").unwrap())?;
    let student_rt = open(&path_of(&s3, "q8").unwrap())?;
    data.check_tokenizer(&student, "fold")?;
    let on = LossOn::parse(&a.loss_on)?;
    let eos = student
        .header
        .tokenizer_config
        .as_ref()
        .and_then(|t| t.eos_token_ids.first().copied())
        .unwrap_or(0);
    use crate::tdata::TdataSplit;
    let tr = Stream::build(data.split(TdataSplit::Train), on, eos, a.seq + 1);
    let va = Stream::build(data.split(TdataSplit::Val), on, eos, a.seq + 1);
    let nl = student.arch().num_layers;
    let params = params_of(&student).0 as f64;
    // f32 replicas of student and teacher + Adam state of the trainables
    let need_all = params * 4.0 * 2.0 + (params * 0.8) * 16.0;
    let spec = match a.heal_layers.as_str() {
        "auto" if need_all < mem_budget() as f64 * 0.6 => "all".to_string(),
        "auto" => format!("deep{}", ((5 * nl) as f64 / 16.0).round() as usize),
        other => other.to_string(),
    };
    let o1 = O1Cfg::from_spec(&spec, None, None, None, None).with_context(|| format!("--heal-layers {spec:?}"))?;
    let n_train = data.split(TdataSplit::Train).count();
    let steps = ((run.heal_epochs * n_train as f64) / a.bs as f64).ceil().max(1.0) as usize;
    let hp = FcdHyper {
        steps,
        lr: a.heal_lr.unwrap_or(1e-5),
        kl_w: a.kd,
        eval_every: a.eval_every,
        bs: a.bs,
        seq: a.seq,
        seed: 0,
        polish_only: true,
    };
    // Generation gate: val prompts continued greedily. A checkpoint passes
    // when no prompt loops more than the fold did before training (+0.10);
    // the absolute threshold is off — a small model's greedy code already
    // repeats 4-grams on some prompts, and an absolute bar would reject
    // every checkpoint and silently turn the heal into the identity.
    let prompts: Vec<Vec<u32>> = data
        .split(TdataSplit::Val)
        .take(6)
        .map(|r| {
            let p = &r.ids[..r.prompt];
            p[p.len().saturating_sub(400)..].to_vec()
        })
        .collect();
    let gate = GenGateCfg { prompts, gen_tokens: 96, threshold: 1.0, baseline_slack: 0.10 };
    run.log(&format!(
        "S4: KD heal of layers {spec} ({steps} steps × {}×{} tokens, lr {}, KL weight {}, loss on {:?}); \
         train {} scored tokens, val {}; teacher {}",
        a.bs,
        a.seq,
        hp.lr,
        a.kd,
        on,
        tr.scored,
        va.scored,
        teacher_path.display()
    ));
    let out = run.stage_dir(4).join("healed.cmf");
    std::fs::create_dir_all(out.parent().unwrap())?;
    let rep = run_polish_stream(
        &student,
        Some(&teacher),
        &o1,
        &hp,
        FcdStream { ids: &tr.ids, weight: Some(&tr.weight), starts: Some(&tr.starts) },
        FcdStream { ids: &va.ids, weight: Some(&va.weight), starts: Some(&va.starts) },
        &out,
        Some(&gate),
        Some(CacheModels { teacher: &teacher_rt, student: &student_rt }),
    )
    .map_err(|e| anyhow::anyhow!("heal: {e}"))?;
    run.log(&format!(
        "S4: val ppl (scored tokens) teacher {:.3} · fold {:.3} → healed {:.3} (best step {}, {:.1}s/step)",
        rep.teacher_ppl, rep.ppl_start, rep.ppl_final, rep.best_step, rep.sec_per_step
    ));
    let healed = open(&out)?;
    let q8 = run.stage_dir(4).join("healed-q8_2f.cmf");
    cmfio::rewrite(&healed, &q8, Encoding::Profile(Quant::Q8_2f), &|_| Ok(None), &|h| set_quant_type(h, Quant::Q8_2f), run.threads)?;
    run.mark_done(
        4,
        json!({
            "f16": out, "q8": q8, "healed": true, "layers": spec, "steps": steps,
            "teacher_ppl": rep.teacher_ppl, "ppl_start": rep.ppl_start, "ppl_final": rep.ppl_final,
            "best_step": rep.best_step, "sec_per_step": rep.sec_per_step,
            "gate_chosen": rep.gate.as_ref().map(|g| g.chosen),
        }),
    )
}

/// The layers `--heal-layers` names: all | deepN | auto (`auto`: all when
/// the f32 weights with their Adam state fit `room` bytes, else the deepest
/// layers that do, at least one).
fn heal_train_from(spec: &str, nl: usize, per_layer: f64, room: f64) -> Result<usize> {
    Ok(match spec {
        "all" => 0,
        "auto" => {
            let fit = ((room / per_layer.max(1.0)).floor() as usize).clamp(1, nl);
            nl - fit
        }
        s => {
            let n: usize = s.strip_prefix("deep").and_then(|v| v.parse().ok()).with_context(|| format!("--heal-layers {s:?}: all | deepN | auto"))?;
            nl - n.clamp(1, nl)
        }
    })
}

#[cfg(feature = "gpu")]
fn s4_heal_device(run: &Run, data: &Data, student: &CmfModel, student_path: &Path, teacher_path: &Path) -> Result<()> {
    let a = &run.a;
    let on = LossOn::parse(&a.loss_on)?;
    let nl = student.arch().num_layers;
    let ffn_only = match a.heal_params.as_str() {
        "all" => false,
        "ffn" => true,
        o => anyhow::bail!("--heal-params {o:?}: all | ffn"),
    };
    // Device memory while training: the head's planes both ways, ~6 GB of
    // workspace and logits, and per trained layer its f16 planes both ways
    // plus f32 weights, gradient and two moments (20 B/weight of what it
    // trains, 4 B/weight of its frozen rest). The frozen prefix runs once
    // beforehand (its activations cached on the host), so it takes no room.
    let (total, _) = params_of(student);
    let arch = student.arch();
    let head = 2.0 * 2.0 * (arch.vocab_size * arch.hidden_size) as f64;
    let layer_w = (total as f64 - (arch.vocab_size * arch.hidden_size) as f64) / nl as f64;
    let ffn_w = student
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("model.layers.") && e.name.contains(".mlp.") && e.name.ends_with("_proj.weight"))
        .map(|e| e.shape.iter().product::<usize>() as f64)
        .sum::<f64>()
        / nl as f64;
    let trained = if ffn_only { ffn_w.min(layer_w) } else { layer_w };
    let per_layer = 20.0 * trained + 4.0 * (layer_w - trained);
    let budget = cortiq_engine::gpu_wgpu::dsv4_vram_budget().unwrap_or(8 << 30) as f64;
    let room = budget - head - 6e9;
    let train_from = heal_train_from(&a.heal_layers, nl, per_layer, room.max(per_layer))?;
    let need = per_layer * (nl - train_from) as f64;
    anyhow::ensure!(
        need <= room.max(per_layer) * 1.05,
        "--heal-layers {}: training {} layers needs ~{:.1} GB on the device, the budget leaves {:.1} GB — \
         use --heal-layers auto (the deepest layers that fit) or --heal-params ffn",
        a.heal_layers,
        nl - train_from,
        need / 1e9,
        room.max(0.0) / 1e9
    );
    if a.heal_layers == "auto" {
        run.log(&format!(
            "S4: --heal-layers auto → the last {} of {nl} layers ({:.1} GB of {:.1} GB device budget for training state)",
            nl - train_from,
            room.max(0.0) / 1e9,
            budget / 1e9
        ));
    }
    // the more layers train, the smaller the step each may take (0.8B, all
    // 24 layers, 6 epochs: 1e-5 0.331, 5e-6 0.310, 3e-6 0.307; 8 layers at
    // 1e-5 0.328)
    let frac = (nl - train_from) as f64 / nl as f64;
    let lr = a.heal_lr.unwrap_or(1e-5 * (0.3 / frac).min(1.0));
    let epochs = a.heal_epochs.unwrap_or(if a.recipe == "full" { 12.0 } else { 6.0 });
    let cfg = heal_gpu::Cfg {
        train_from,
        ffn_only,
        epochs,
        lr,
        window: a.heal_window,
        batch: a.bs,
        topk: a.heal_topk,
        eval_every: a.eval_every,
        clip: 1.0,
        warm: std::env::var("CMF_HEAL_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1),
        genesis: match a.heal_schedule.as_str() {
            "flat" => false,
            "genesis" => true,
            o => anyhow::bail!("--heal-schedule {o:?}: flat | genesis"),
        },
        ce: match a.heal_loss.as_str() {
            "kd" => 0.0,
            "ce" => 1.0,
            m => m
                .strip_prefix("mix=")
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| (0.0..=1.0).contains(v))
                .with_context(|| format!("--heal-loss {m:?}: kd | ce | mix=A with 0 ≤ A ≤ 1"))?,
        },
    };
    let out = run.stage_dir(4).join("healed.cmf");
    std::fs::create_dir_all(out.parent().unwrap())?;
    let rep = heal_gpu::heal(data, on, student_path, teacher_path, &out, &cfg, &|s| run.log(s))?;
    let healed = open(&out)?;
    let q8 = run.stage_dir(4).join("healed-q8_2f.cmf");
    cmfio::rewrite(&healed, &q8, Encoding::Profile(Quant::Q8_2f), &|_| Ok(None), &|h| set_quant_type(h, Quant::Q8_2f), run.threads)?;
    run.mark_done(
        4,
        json!({
            "f16": out, "q8": q8, "healed": true, "device": "wgpu", "train_from": train_from, "ffn_only": ffn_only,
            "steps": rep.steps, "kl_start": rep.kl_start, "kl_best": rep.kl_best, "best_step": rep.best_step,
            "sec_per_step": rep.sec_per_step, "teacher_secs": rep.teacher_secs, "trained_weights": rep.trained,
        }),
    )
}

// ────────────────────────────── S5 quant ──────────────────────────────

/// `--escort`: tensors of a low-bit output kept at q8_2f.
fn escort_patterns(spec: &str) -> Vec<String> {
    match spec {
        "none" | "" => Vec::new(),
        "kv" => vec![
            "model.layers.*.self_attn.k_proj.weight=q8_2f".into(),
            "model.layers.*.self_attn.v_proj.weight=q8_2f".into(),
        ],
        "kvo" => vec![
            "model.layers.*.self_attn.k_proj.weight=q8_2f".into(),
            "model.layers.*.self_attn.v_proj.weight=q8_2f".into(),
            "model.layers.*.self_attn.o_proj.weight=q8_2f".into(),
        ],
        list => list.split(',').map(|g| format!("{}=q8_2f", g.trim())).collect(),
    }
}

fn s5_quant(run: &Run) -> Result<()> {
    let a = &run.a;
    let s0 = run.require(0)?;
    let (input, cal) = if run.structural {
        let s4 = run.require(4)?;
        (path_of(&s4, "f16").unwrap(), path_of(&s4, "q8").unwrap())
    } else {
        let q8 = path_of(&s0, "q8").unwrap();
        (path_of(&s0, "f16").unwrap_or(q8.clone()), q8)
    };
    let out = run.stage_dir(5).join("out.cmf");
    std::fs::create_dir_all(out.parent().unwrap())?;
    let _ = std::fs::remove_file(&out);
    let low_bit = matches!(run.quant, Quant::Q4TiledP | Quant::Q2TiledP | Quant::Q4Tiled | Quant::Q4Block | Quant::Q1 | Quant::Q1p | Quant::Q1s | Quant::Q1t | Quant::Mxfp4 | Quant::Fp4);
    let mut overrides = if low_bit { escort_patterns(&a.escort) } else { Vec::new() };
    overrides.extend(a.tensor_quant.iter().cloned());
    crate::convert::set_tensor_quant_overrides(&overrides)?;
    let data = Data::load(&a.data)?;
    let train_tokens: usize = data.split(crate::tdata::TdataSplit::Train).map(|r| r.ids.len()).sum();
    let mut scored_as: Option<PathBuf> = None;
    match run.quant {
        Quant::Q4TiledP | Quant::Q2TiledP => {
            let tokens = run.calib_tokens.unwrap_or(65536).min(train_tokens);
            run.log(&format!(
                "S5: GPTQ {} on {tokens} train tokens (windows ≤{} inside records), escort {:?}",
                crate::convert::quant_name(run.quant),
                a.calib_window,
                overrides
            ));
            if tokens < 32768 {
                run.log(&format!(
                    "  note: {tokens} calibration tokens — measured: GPTQ fits 2048 tokens and gains up to ~32k; more data helps"
                ));
            }
            let root = a.data.rfind(".tdata").map(|i| &a.data[..i + 6]).unwrap_or(&a.data);
            let spec = format!("{root}:train");
            let opts = crate::gptq::GptqTiledOpts {
                input: input.to_str().unwrap(),
                calib_model: Some(cal.to_str().unwrap()),
                calib: vec![crate::gptq::CalibSource { path: spec, share: 1.0 }],
                output: out.to_str().unwrap(),
                tokens,
                window: a.calib_window,
                lambda: 0.01,
                threads: run.threads,
                act_order: false,
                hessians: None,
                profile: run.quant,
                q2tp_rule: crate::gptq::Q2tpRule::parse("placed")?,
                q2tp_rule_overrides: Vec::new(),
                q2tp_debias: true,
                q4tp_debias: false,
                chunk_layers: 0,
            };
            crate::gptq::run_quantize_gptq_tiled(&opts)?;
        }
        Quant::F16 => {
            // the source's own precision: the healed f16 file as it is — a
            // hard link where the filesystem allows (no second copy on disk);
            // S6 scores it through its q8_2f copy (the f16 runtime path is
            // ×20 slower and the weights are the same)
            scored_as = Some(cal.clone());
            run.log("S5: output in f16, the source's precision (quantize it separately if a smaller file is wanted)");
            if std::fs::hard_link(&input, &out).is_err() {
                let m = open(&input)?;
                cmfio::rewrite(&m, &out, Encoding::KeepF16, &|_| Ok(None), &|h| set_quant_type(h, Quant::F16), run.threads)?;
            }
        }
        q => {
            run.log(&format!("S5: encoding as {} (no calibration needed)", crate::convert::quant_name(q)));
            let m = open(&input)?;
            cmfio::rewrite(&m, &out, Encoding::Profile(q), &|_| Ok(None), &|h| set_quant_type(h, q), run.threads)?;
        }
    }
    let bytes = std::fs::metadata(&out)?.len();
    run.log(&format!("S5: {} ({:.2} GB)", out.display(), bytes as f64 / 1e9));
    run.mark_done(5, json!({ "out": out, "bytes": bytes, "escort": overrides, "scored_as": scored_as }))
}

/// Write the final file: the S5 container with the compress provenance.
pub fn publish(run: &Run, target: &Path) -> Result<()> {
    let s5 = run.require(5)?;
    let s6 = run.require(6)?;
    let m = open(&path_of(&s5, "out").unwrap())?;
    let mut prov = json!({
        "tool": format!("cortiq {}", env!("CARGO_PKG_VERSION")),
        "source": run.a.source,
        "data": run.a.data,
        "quant": crate::convert::quant_name(run.quant),
        "recipe": run.a.recipe,
        "stages": STAGES.iter().enumerate().filter_map(|(i, s)| run.done(i).map(|v| (s.to_string(), v["key"].clone()))).collect::<serde_json::Map<_, _>>(),
        "eval": s6.get("summary").cloned().unwrap_or(Value::Null),
    });
    if let Some(s0) = run.done(0) {
        prov["data_teacher"] = s0["data_teacher"].clone();
    }
    if run.structural
        && let Some(s2) = run.done(2)
    {
        prov["params"] = json!({ "before": s2["params_before"], "after": s2["params_after"] });
    }
    cmfio::restamp(&m, target, &|h| {
        // the compressed model runs exact attention
        if let Some(serde_json::Value::Object(p)) = h.provenance.as_mut() {
            p.remove("o1_attn");
        }
        cmfio::provenance_set(h, "compress", prov.clone())
    })?;
    Ok(())
}
