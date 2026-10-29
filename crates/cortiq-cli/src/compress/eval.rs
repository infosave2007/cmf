//! S6: what the compression cost, measured on records no stage saw.
//!
//! Every candidate (the source, the fold, the healed model, the final
//! file) scores the teacher's test answers (NLL per generated token) and
//! continues the first test prompts greedily: loops (distinct 5-grams of
//! the tail), length against the teacher, how often it stops by itself.
//! The gates compare the final file with the source on the same prompts.

use super::data::{Data, LossOn};
use super::{Run, STAGES};
use crate::tdata::TdataSplit;
use anyhow::{Context, Result};
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

fn greedy() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(1),
        suppress_tokens: Vec::new(),
        penalty_window: 0,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Scores of one candidate.
fn evaluate(run: &Run, label: &str, path: &PathBuf, data: &Data, on: LossOn) -> Result<Value> {
    let m = Arc::new(CmfModel::open_sharded(path).with_context(|| format!("opening {}", path.display()))?);
    let test: Vec<_> = data.split(TdataSplit::Test).collect();
    // teacher-forced NLL of the teacher's test answers: on the device's
    // trainer kernels when they run here (the runtime's prefill scores
    // ~200 tok/s there), else through the runtime
    let t0 = std::time::Instant::now();
    let (mut nll, mut cnt, mut how) = (0f64, 0usize, "runtime");
    // records past what the card's attention scratch holds go through the
    // runtime (`device_nll_max`: the trainer keeps a window² block per head)
    let mut rest: Vec<_> = test.clone();
    #[cfg(feature = "gpu")]
    if nll_on_device(&m) {
        let max = super::heal_gpu::device_nll_max(&m);
        let (mut short, long): (Vec<_>, Vec<_>) = test.iter().copied().partition(|r| r.ids.len() <= max);
        // a record past the device window is scored on its first `max`
        // tokens there (the same cut for every candidate): through the
        // runtime a 27B scores ~200 tok/s, and its 8k-token answers were 63 %
        // of the test tokens. CMF_COMPRESS_EVAL_LONG=runtime scores them whole.
        let whole = std::env::var("CMF_COMPRESS_EVAL_LONG").as_deref() == Ok("runtime");
        let cut: Vec<super::data::Rec> = if whole {
            Vec::new()
        } else {
            long.iter().filter(|r| on.first_target(r) + 1 < max).map(|r| super::data::Rec { ids: r.ids[..max].to_vec(), ..(*r).clone() }).collect()
        };
        if !cut.is_empty() {
            eprintln!("  {label}: {} records past {max} tokens are scored on their first {max} (CMF_COMPRESS_EVAL_LONG=runtime: whole, on the runtime)", cut.len());
        }
        short.extend(cut.iter());
        if !short.is_empty() {
            (nll, cnt) = super::heal_gpu::test_nll(path, &short, on)?;
            how = if whole && !long.is_empty() { "device + runtime" } else if cut.is_empty() { "device" } else { "device, long records cut" };
        }
        rest = if whole { long } else { Vec::new() };
    }
    let mut pipe = Pipeline::from_model(&m, greedy()).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    pipe.set_confidence(false);
    {
        for r in &rest {
            let first = on.first_target(r).max(1);
            let (s, c) = pipe.nll_ids_from(&r.ids, first - 1).map_err(|e| anyhow::anyhow!("{label}: {}: {e}", r.rid))?;
            nll += s;
            cnt += c;
        }
    }
    let mean_nll = nll / cnt.max(1) as f64;
    eprintln!("  {label}: test NLL {mean_nll:.4} over {cnt} tokens ({:.0}s, {how})", t0.elapsed().as_secs_f64());
    // greedy continuation of the first test prompts
    let cap = run.a.eval_gen_tokens;
    let mut gens = Vec::new();
    let (mut gen_tok, mut gen_s) = (0usize, 0f64);
    let (mut mtp_drafted, mut mtp_accepted) = (0usize, 0usize);
    // one generation per distinct prompt: the teacher answered each prompt
    // twice, and two greedy runs of one prompt are one sample counted twice
    // (16 generations were 8 prompts)
    let mut seen = std::collections::HashSet::new();
    let gen_recs: Vec<_> = test.iter().filter(|r| seen.insert(&r.ids[..r.prompt])).take(run.a.eval_prompts).collect();
    let n_gen = gen_recs.len();
    for (gi, r) in gen_recs.into_iter().enumerate() {
        let prompt = &r.ids[..r.prompt];
        let t = std::time::Instant::now();
        if gi > 0 {
            eprint!(
                "\r  {label}: generation {gi}/{n_gen} · {:.0} tok/s · ~{} left   ",
                gen_tok as f64 / gen_s.max(1e-9),
                crate::fmt_eta(gen_s / gi as f64 * (n_gen - gi) as f64)
            );
        }
        let g = pipe.generate_from_ids(prompt, cap, None, None).map_err(|e| anyhow::anyhow!("{label}: {}: {e}", r.rid))?;
        gen_s += t.elapsed().as_secs_f64();
        gen_tok += g.tokens_generated;
        mtp_drafted += g.mtp_drafted;
        mtp_accepted += g.mtp_accepted;
        let ids = &g.token_ids[g.token_ids.len().saturating_sub(g.tokens_generated)..];
        let d5 = crate::teacher_data::tail_distinct_5gram(ids);
        let teacher_len = r.generated().min(cap).max(1);
        gens.push(json!({
            "rid": r.rid, "tokens": g.tokens_generated, "finish": g.finish_reason,
            "teacher_tokens": r.generated(), "len_ratio": g.tokens_generated as f64 / teacher_len as f64,
            "distinct5": d5.ratio, "loop": g.tokens_generated >= 64 && d5.ratio < 0.30,
            "text": g.text.chars().take(600).collect::<String>(),
        }));
    }
    let loops = gens.iter().filter(|g| g["loop"].as_bool() == Some(true)).count();
    let stops = gens.iter().filter(|g| g["finish"].as_str().is_some_and(|f| f == "stop" || f == "eos")).count();
    let ratios: Vec<f64> = gens.iter().filter_map(|g| g["len_ratio"].as_f64()).collect();
    // optional general text: forgetting outside the task
    let general = match &run.a.general_eval {
        Some(p) => {
            let text = std::fs::read_to_string(p)?;
            let ids = pipe.tokenizer.encode(&text);
            let (mut s, mut c) = (0f64, 0usize);
            for w in ids.chunks(1024).take(16).filter(|w| w.len() > 32) {
                let (a, b) = pipe.nll_ids_from(w, 0).map_err(|e| anyhow::anyhow!("{label}: general: {e}"))?;
                s += a;
                c += b;
            }
            Some((s / c.max(1) as f64).exp())
        }
        None => None,
    };
    // the runtime keeps a model's uploaded weights in the device cache after
    // its pipeline is gone: the 27B source and its fold held 41 GB there and
    // the next candidate's trainer pass ran out of memory
    drop(pipe);
    #[cfg(feature = "gpu")]
    cortiq_engine::gpu_wgpu::release_model_buffers(&m);
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(json!({
        "label": label, "path": path, "bytes": bytes, "params": m.total_param_count(),
        "nll": mean_nll, "ppl": mean_nll.exp(), "scored_tokens": cnt,
        "gen": {
            "prompts": gens.len(), "loops": loops, "stops": stops,
            "median_len_ratio": median(ratios), "tok_s": gen_tok as f64 / gen_s.max(1e-9),
            "mtp_drafted": mtp_drafted, "mtp_accepted": mtp_accepted,
        },
        "general_ppl": general,
        "samples": gens,
    }))
}

/// `--eval-nll`: auto = on the device's trainer kernels when usable.
#[cfg(feature = "gpu")]
fn nll_on_device(m: &CmfModel) -> bool {
    match std::env::var("CMF_COMPRESS_EVAL_NLL").as_deref() {
        Ok("runtime") => false,
        _ => super::heal_gpu::usable(m).is_ok(),
    }
}

pub fn s6_eval(run: &Run) -> Result<()> {
    let a = &run.a;

    let data = Data::load(&a.data)?;
    let on = LossOn::parse(&a.loss_on)?;
    let path = |i: usize, k: &str| -> Option<PathBuf> {
        run.done(i).and_then(|v| v.get(k).and_then(Value::as_str).map(PathBuf::from))
    };
    let mut cands: Vec<(String, PathBuf)> = vec![("source".into(), path(0, "q8").context("S0 output")?)];
    if run.structural {
        cands.push(("fold".into(), path(3, "q8").context("S3 output")?));
        if run.done(4).and_then(|v| v["healed"].as_bool()) == Some(true) {
            cands.push(("healed".into(), path(4, "q8").context("S4 output")?));
        }
    }
    // an f16 output is the healed (or folded) f16 file itself: scored
    // through the q8_2f copy of the same weights, the file only opened
    let out = path(5, "out").context("S5 output")?;
    let scored_as = path(5, "scored_as");
    if let Some(q8) = &scored_as {
        CmfModel::open_sharded(&out).with_context(|| format!("the output {} does not open", out.display()))?;
        run.log(&format!("S6: the output keeps the f16 weights of {} — scored as that q8_2f copy", q8.display()));
    }
    cands.push(("final".into(), scored_as.clone().unwrap_or_else(|| out.clone())));
    let n_own = cands.len();
    for c in &a.compare {
        let p = PathBuf::from(c);
        anyhow::ensure!(p.is_file(), "--compare {c}: no such file");
        cands.push((format!("vs {}", c.trim_end_matches(".cmf")), p));
    }
    run.log(&format!(
        "S6: {} test records, up to {} generation prompts (distinct ones); intermediate stages are scored on their q8_2f copies",
        data.count(TdataSplit::Test).0,
        a.eval_prompts.min(data.count(TdataSplit::Test).0)
    ));
    let mut res = Vec::new();
    std::fs::create_dir_all(run.stage_dir(6))?;
    for (ci, (label, p)) in cands.iter().enumerate() {
        // A file's scores depend only on its bytes, the data and the eval
        // settings: they are kept beside the file (`<file>.eval.json`), so
        // the source of an A/B series (shared S0) is scored once.
        let key = eval_key(run, p);
        // a --compare model's scores stay in this run (its file may be
        // another run's or read-only)
        let cache = if ci < n_own {
            PathBuf::from(format!("{}.eval.json", p.display()))
        } else {
            run.stage_dir(6).join(format!("vs-{}.eval.json", &crate::knowledge::sha256_hex(p.display().to_string().as_bytes())[..12]))
        };
        let cached: Option<Value> = std::fs::read(&cache)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .filter(|v| v["key"].as_str() == Some(key.as_str()));
        let mut r = match cached {
            Some(v) => {
                eprintln!("  {label}: scores reused from {}", cache.display());
                v["result"].clone()
            }
            None => {
                let r = evaluate(run, label, p, &data, on)?;
                let _ = std::fs::write(&cache, serde_json::to_vec(&json!({"key": key, "result": r}))?);
                r
            }
        };
        r["label"] = json!(label);
        if label == "final" && scored_as.is_some() {
            r["bytes"] = json!(std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0));
            r["path"] = json!(out);
        }
        res.push(r);
    }
    let src = &res[0];
    let fin = &res[n_own - 1];
    let g = |v: &Value, k: &str| v["gen"][k].as_f64().unwrap_or(f64::NAN);
    let src_ratio = g(src, "median_len_ratio");
    let ratio_floor = (0.7f64).min(0.9 * src_ratio);
    let gates = json!({
        "G0_load": fin["nll"].as_f64().is_some_and(f64::is_finite),
        "G1_loops": g(fin, "loops") <= g(src, "loops") + 1.0,
        "G1_length": g(fin, "median_len_ratio") >= ratio_floor,
        "G1_stops": g(fin, "stops") + 0.1 * g(fin, "prompts") >= g(src, "stops"),
    });
    let pass = gates.as_object().unwrap().values().all(|v| v.as_bool() == Some(true));
    let summary: Vec<Value> = res
        .iter()
        .map(|r| {
            json!({
                "label": r["label"], "bytes": r["bytes"], "params": r["params"], "nll": r["nll"], "ppl": r["ppl"],
                "loops": r["gen"]["loops"], "stops": r["gen"]["stops"], "prompts": r["gen"]["prompts"],
                "median_len_ratio": r["gen"]["median_len_ratio"], "tok_s": r["gen"]["tok_s"],
                "mtp_drafted": r["gen"]["mtp_drafted"], "mtp_accepted": r["gen"]["mtp_accepted"],
                "general_ppl": r["general_ppl"],
            })
        })
        .collect();
    let dir = run.stage_dir(6);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({ "candidates": res, "gates": gates, "pass": pass }))?)?;
    let mut samples = String::new();
    for r in &res {
        for s in r["samples"].as_array().into_iter().flatten() {
            samples += &format!("{}\n", json!({"model": r["label"], "sample": s}));
        }
    }
    std::fs::write(dir.join("samples.jsonl"), samples)?;
    let md = report_md(run, &summary, &gates, pass);
    std::fs::write(dir.join("report.md"), &md)?;
    run.mark_done(6, json!({ "summary": summary, "gates": gates, "pass": pass, "report": dir.join("report.md") }))
}

/// Identity of one file's scores: its size and mtime, the data manifest,
/// and every setting the scoring uses.
fn eval_key(run: &Run, p: &std::path::Path) -> String {
    let m = std::fs::metadata(p).ok();
    let mt = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let root = run.a.data.rfind(".tdata").map(|i| &run.a.data[..i + 6]).unwrap_or(&run.a.data);
    let manifest = std::fs::read(std::path::Path::new(root).join("manifest.json")).unwrap_or_default();
    let k = json!({
        "size": m.map(|m| m.len()), "mtime": mt.map(|d| d.as_nanos() as u64),
        "data": crate::knowledge::sha256_hex(&manifest), "spec": run.a.data,
        "loss_on": run.a.loss_on, "prompts": run.a.eval_prompts, "gen": run.a.eval_gen_tokens,
        "general": run.a.general_eval, "v": 4, "o1": "off",
    });
    crate::knowledge::sha256_hex(k.to_string().as_bytes())[..16].to_string()
}

fn report_md(run: &Run, summary: &[Value], gates: &Value, pass: bool) -> String {
    let mut s = format!(
        "# compress report\n\n- source: `{}`\n- data: `{}`\n- output: `{}` ({})\n- recipe: {}\n\n",
        run.a.source,
        run.a.data,
        run.a.output,
        crate::convert::quant_name(run.quant),
        run.a.recipe
    );
    s += "| model | size GB | params | test NLL | ppl | loops | stops | len/teacher | tok/s | MTP accepted | general ppl |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    for r in summary {
        let (md, ma) = (r["mtp_drafted"].as_u64().unwrap_or(0), r["mtp_accepted"].as_u64().unwrap_or(0));
        let mtp = if md > 0 { format!("{:.0} % of {md}", ma as f64 * 100.0 / md as f64) } else { "—".to_string() };
        s += &format!(
            "| {} | {:.3} | {:.3}B | {:.4} | {:.3} | {}/{} | {}/{} | {:.2} | {:.1} | {mtp} | {} |\n",
            r["label"].as_str().unwrap_or(""),
            r["bytes"].as_f64().unwrap_or(0.0) / 1e9,
            r["params"].as_f64().unwrap_or(0.0) / 1e9,
            r["nll"].as_f64().unwrap_or(f64::NAN),
            r["ppl"].as_f64().unwrap_or(f64::NAN),
            r["loops"], r["prompts"], r["stops"], r["prompts"],
            r["median_len_ratio"].as_f64().unwrap_or(f64::NAN),
            r["tok_s"].as_f64().unwrap_or(f64::NAN),
            r["general_ppl"].as_f64().map_or("—".to_string(), |v| format!("{v:.3}")),
        );
    }
    s += &format!("\ngates: {} — {}\n", if pass { "PASS" } else { "FAIL" }, gates);
    // the size against the source in the output's own codec (the table's
    // source and fold rows are their q8_2f scoring copies)
    let src_same = run.done(0).and_then(|v| {
        let key = if run.quant == crate::convert::Quant::F16 { "f16" } else { "q8" };
        v.get(key).and_then(Value::as_str).map(String::from)
    });
    if let (Some(src), Some(fin)) = (src_same, summary.iter().find(|r| r["label"] == "final")) {
        let sb = std::fs::metadata(&src).map(|m| m.len() as f64).unwrap_or(0.0);
        let fb = fin["bytes"].as_f64().unwrap_or(0.0);
        if sb > 0.0 && fb > 0.0 {
            s += &format!(
                "size: {:.3} GB → {:.3} GB = {:.1} % of the source in the same codec ({})\n",
                sb / 1e9,
                fb / 1e9,
                100.0 * fb / sb,
                crate::convert::quant_name(run.quant)
            );
        }
    }
    for (i, st) in STAGES.iter().enumerate() {
        if let Some(v) = run.done(i) {
            s += &format!("\n- {st}: key {} at {}", v["key"].as_str().unwrap_or(""), v["finished"].as_str().unwrap_or(""));
        }
    }
    s += "\n";
    s
}

pub fn print_report(run: &Run, s6: &Value) -> Result<()> {
    let md = std::fs::read_to_string(s6["report"].as_str().unwrap_or_default()).unwrap_or_default();
    for line in md.lines().filter(|l| l.starts_with('|') || l.starts_with("gates") || l.starts_with("size:")) {
        run.log(line);
    }
    Ok(())
}
