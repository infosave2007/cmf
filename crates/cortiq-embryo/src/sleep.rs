//! Sleep = idle time (docs/NATIVE_MODEL_TECH.ru.md §2, §5): a daemon that
//! watches the OOD buffer `cortiq serve` fills (CMF_OOD_DIR) and, when the
//! organism is idle, runs the night — bake a skill from the buffer,
//! gate it, commit or roll back, archive the buffer, optionally requant
//! under a ppl gate — every step atomic and append-only, preempted the
//! moment a request arrives (the idle marker moves), resumed by the next
//! idle window. Everything is journaled (`journal.jsonl`).

use crate::skill::{BakeArgs, append_to_cmf, bake};
use crate::tokenizer::{Bpe, EOT};
use crate::train::{Checkpoint, Shard, load_checkpoint};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct SleepArgs {
    pub ckpt: PathBuf,
    pub tokenizer: PathBuf,
    pub cmf: PathBuf,
    pub ood_dir: PathBuf,
    /// minutes without requests before a night may start
    pub idle_min: f64,
    /// buffered tokens needed for a skill
    pub min_tokens: usize,
    /// required held-out improvement (fraction of loss) to commit a skill
    pub gate: f32,
    /// requant to q4tp when the ppl loss ratio stays below this (0 = off)
    pub requant_gate: f32,
    /// held-out text for the requant gate
    pub held_out: Option<PathBuf>,
    /// path of the `cortiq` binary (requant + ppl)
    pub cortiq_bin: String,
    pub layers: Vec<usize>,
    pub steps_a: usize,
    pub steps_b: usize,
    pub batch: usize,
    pub seq: usize,
    /// run one cycle (if conditions hold, or --force) and exit
    pub once: bool,
    /// ignore idle / min-tokens conditions (demo)
    pub force: bool,
    pub poll_secs: u64,
    /// try growth (new experts) after this many consecutive rejected nights (0 = off)
    pub grow_after: usize,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn journal(dir: &Path, event: &str, details: serde_json::Value) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("journal.jsonl")) {
        let _ = writeln!(f, "{}", serde_json::json!({"ts": now(), "event": event, "details": details}));
    }
    eprintln!("[sleep] {event}: {details}");
}

/// mtime of the idle marker (None when absent; None < Some).
fn marker_mtime(dir: &Path) -> Option<SystemTime> {
    std::fs::metadata(dir.join("last_request")).and_then(|m| m.modified()).ok()
}

/// Seconds since the last request (∞ when the marker is absent).
fn idle_secs(dir: &Path) -> f64 {
    match std::fs::metadata(dir.join("last_request")).and_then(|m| m.modified()) {
        Ok(t) => SystemTime::now().duration_since(t).map(|d| d.as_secs_f64()).unwrap_or(0.0),
        Err(_) => f64::INFINITY,
    }
}

/// Buffer contents: (texts, total tokens).
fn read_buffer(dir: &Path) -> (Vec<String>, usize) {
    let Ok(s) = std::fs::read_to_string(dir.join("buffer.jsonl")) else { return (Vec::new(), 0) };
    let mut texts = Vec::new();
    let mut toks = 0usize;
    for line in s.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = v["text"].as_str() {
                texts.push(t.to_string());
                toks += v["tokens"].as_u64().unwrap_or(0) as usize;
            }
        }
    }
    (texts, toks)
}

/// The tensor directory of a .cmf as (name → hash) — the byte-identity proof.
fn tensor_hashes(path: &Path) -> anyhow::Result<std::collections::HashMap<String, u64>> {
    let m = cortiq_core::format::CmfModel::open(path)?;
    Ok(m.tensors.iter().map(|t| (t.name.clone(), t.hash)).collect())
}

pub fn run(a: SleepArgs) -> anyhow::Result<()> {
    let ck: Checkpoint = load_checkpoint(&a.ckpt)?;
    let bpe = Bpe::load(&a.tokenizer)?;
    let eot = bpe.special_id(EOT).unwrap_or(0) as u16;
    journal(&a.ood_dir, "daemon_start", serde_json::json!({"cmf": a.cmf, "idle_min": a.idle_min, "min_tokens": a.min_tokens, "gate": a.gate}));
    loop {
        let idle = idle_secs(&a.ood_dir);
        let (texts, toks) = read_buffer(&a.ood_dir);
        let ready = a.force || (idle >= a.idle_min * 60.0 && toks >= a.min_tokens);
        if !ready {
            if a.once {
                journal(&a.ood_dir, "nothing_to_do", serde_json::json!({"idle_s": idle, "buffer_tokens": toks}));
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(a.poll_secs));
            continue;
        }
        if texts.is_empty() {
            journal(&a.ood_dir, "nothing_to_do", serde_json::json!({"reason": "empty buffer"}));
            if a.once {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(a.poll_secs));
            continue;
        }
        // ---- the night begins ----
        let t0 = Instant::now();
        let id = format!("night-{}", now());
        journal(&a.ood_dir, "night_start", serde_json::json!({"skill": id, "buffer_tokens": toks, "docs": texts.len(), "idle_s": idle}));
        // corpus from the buffer
        let mut tokens: Vec<u16> = Vec::new();
        let mut cache = std::collections::HashMap::new();
        for t in &texts {
            let mut ids = Vec::new();
            bpe.encode(t, &mut cache, &mut ids);
            tokens.extend(ids.iter().map(|&i| i as u16));
            tokens.push(eot);
        }
        // a small corpus is repeated so the trainer's windows exist; the
        // held-out split inside `bake` still separates the tail
        while tokens.len() < 40 * (a.seq + 2) {
            let copy = tokens.clone();
            tokens.extend(copy);
        }
        let corpus = Shard { tokens };
        // preemption: the idle marker moving = a request arrived
        let marker_at_start = marker_mtime(&a.ood_dir);
        let should_stop = || !a.force && marker_mtime(&a.ood_dir) > marker_at_start;
        let nl = ck.cfg.layers;
        let bargs = BakeArgs {
            id: id.clone(),
            layers: a.layers.clone(),
            steps_a: a.steps_a,
            steps_b: a.steps_b,
            lr_a: 3e-2,
            lr_b: 5e-5,
            l1: 1e-3,
            tau: 0.5,
            eval_every: 30,
            batch: a.batch,
            seq: a.seq,
            phi_layer: nl * 2 / 3,
            phi_len: 48,
            rank: 8,
            seed: now(),
        };
        let baked = bake(&ck, &corpus, &bargs, &should_stop);
        let (tensors, sel, kept, (l0, la, lb)) = match baked {
            Ok(v) => v,
            Err(e) if e.to_string().contains("preempted") => {
                journal(&a.ood_dir, "preempted", serde_json::json!({"skill": id, "after_s": t0.elapsed().as_secs_f64()}));
                if a.once {
                    return Ok(());
                }
                continue; // wait for the next idle window; the buffer is untouched
            }
            Err(e) => return Err(e),
        };
        // ---- gate ----
        let improvement = (l0 - lb) / l0.max(1e-6);
        let quality = serde_json::json!({
            "held_out_loss": {"base": l0, "mask": la, "mask+fcd": lb},
            "held_out_ppl": {"base": l0.exp(), "mask": la.exp(), "mask+fcd": lb.exp()},
            "kept_fraction": kept, "improvement": improvement,
        });
        if improvement < a.gate {
            journal(&a.ood_dir, "skill_rejected", serde_json::json!({"skill": id, "quality": quality, "gate": a.gate}));
        } else {
            let next = a.cmf.with_extension("cmf.next");
            let before = tensor_hashes(&a.cmf)?;
            let unchanged = append_to_cmf(&a.cmf, &next, &id, &a.layers, &tensors, sel, quality.clone())?;
            // byte-identity proof: every pre-existing tensor keeps its hash
            let after = tensor_hashes(&next)?;
            let mut broken = Vec::new();
            for (name, h) in &before {
                if after.get(name) != Some(h) {
                    broken.push(name.clone());
                }
            }
            if !broken.is_empty() {
                let _ = std::fs::remove_file(&next);
                journal(&a.ood_dir, "rollback", serde_json::json!({"skill": id, "reason": "base tensors changed", "tensors": broken}));
            } else {
                // atomic commit; a running `serve` keeps its old mapping until it reloads
                std::fs::rename(&next, &a.cmf)?;
                journal(&a.ood_dir, "skill_committed", serde_json::json!({"skill": id, "unchanged_tensors": unchanged, "quality": quality, "seconds": t0.elapsed().as_secs_f64(), "note": "restart/reload serve to activate"}));
                // the router recalibrates itself over every skill's held-out φ
                match crate::skill::calibrate_file(&a.cmf, 0.05) {
                    Ok(Some(c)) => journal(&a.ood_dir, "router_calibrated", serde_json::json!({"temperature": c.temperature, "novelty_theta": c.novelty_theta, "samples": c.samples})),
                    Ok(None) => journal(&a.ood_dir, "router_calibration_skipped", serde_json::json!({"reason": "no held-out φ"})),
                    Err(e) => journal(&a.ood_dir, "router_calibration_error", serde_json::json!({"error": e.to_string()})),
                }
            }
        }
        // archive the consumed buffer (append-only history)
        let arch = a.ood_dir.join("archive");
        let _ = std::fs::create_dir_all(&arch);
        let _ = std::fs::rename(a.ood_dir.join("buffer.jsonl"), arch.join(format!("buffer-{}.jsonl", now())));
        // ---- optional requant under a ppl gate ----
        if a.requant_gate > 0.0 {
            if let Some(held) = &a.held_out {
                match requant_gate(&a.cortiq_bin, &a.cmf, held, a.requant_gate) {
                    Ok(Some((p0, p1, out))) => journal(&a.ood_dir, "requant_committed", serde_json::json!({"ppl_f32": p0, "ppl_q4tp": p1, "out": out})),
                    Ok(None) => journal(&a.ood_dir, "requant_rejected", serde_json::json!({"gate": a.requant_gate})),
                    Err(e) => journal(&a.ood_dir, "requant_error", serde_json::json!({"error": e.to_string()})),
                }
            }
        }
        // ---- growth trigger (§4.3 "маска исчерпана"): K nights in a row
        // whose skill failed the gate while the buffer kept coming → the
        // masks stopped helping → try new experts on the archived buffers,
        // gated on held-out; the grown genome replaces the checkpoint and
        // the served file (skills re-appended: their base bytes are intact).
        if a.grow_after > 0 {
            let rejected = consecutive_rejections(&a.ood_dir);
            if rejected >= a.grow_after {
                journal(&a.ood_dir, "growth_start", serde_json::json!({"rejected_in_a_row": rejected}));
                match try_growth(&a, &ck, &bpe, eot, &should_stop) {
                    Ok(Some((l0, l1))) => {
                        journal(&a.ood_dir, "growth_committed", serde_json::json!({"held_out_before": l0, "held_out_after": l1}));
                        // the daemon's genome changed: reload it
                        return run(a);
                    }
                    Ok(None) => journal(&a.ood_dir, "growth_rejected", serde_json::json!({})),
                    Err(e) if e.to_string().contains("preempted") => journal(&a.ood_dir, "growth_preempted", serde_json::json!({})),
                    Err(e) => journal(&a.ood_dir, "growth_error", serde_json::json!({"error": e.to_string()})),
                }
            }
        }
        journal(&a.ood_dir, "night_end", serde_json::json!({"skill": id, "seconds": t0.elapsed().as_secs_f64()}));
        if a.once {
            return Ok(());
        }
    }
}

/// Consecutive `skill_rejected` nights at the tail of the journal (a
/// commit or a growth event resets the count).
fn consecutive_rejections(dir: &Path) -> usize {
    let Ok(s) = std::fs::read_to_string(dir.join("journal.jsonl")) else { return 0 };
    let mut n = 0usize;
    for line in s.lines().rev() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        match v["event"].as_str() {
            Some("skill_rejected") => n += 1,
            Some("skill_committed") | Some("growth_committed") | Some("growth_rejected") => break,
            _ => {}
        }
    }
    n
}

/// Growth attempt: corpus = every archived buffer; E → E+1 experts; train
/// the new ones; gate; on success write `<ckpt>` (backup kept as
/// `<ckpt>.gen<N>`), export the grown genome and re-append the served
/// file's skills onto it. Returns Some((before, after)) when committed.
fn try_growth(a: &SleepArgs, ck: &Checkpoint, bpe: &Bpe, eot: u16, should_stop: &dyn Fn() -> bool) -> anyhow::Result<Option<(f32, f32)>> {
    use crate::growth::{GrowArgs, grow_experts, train_new_experts};
    let arch = a.ood_dir.join("archive");
    let mut texts = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&arch) {
        for e in rd.flatten() {
            if let Ok(s) = std::fs::read_to_string(e.path()) {
                for line in s.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        if let Some(t) = v["text"].as_str() {
                            texts.push(t.to_string());
                        }
                    }
                }
            }
        }
    }
    anyhow::ensure!(!texts.is_empty(), "no archived buffers to grow on");
    let mut tokens: Vec<u16> = Vec::new();
    let mut cache = std::collections::HashMap::new();
    for t in &texts {
        let mut ids = Vec::new();
        bpe.encode(t, &mut cache, &mut ids);
        tokens.extend(ids.iter().map(|&i| i as u16));
        tokens.push(eot);
    }
    while tokens.len() < 40 * (a.seq + 2) {
        let c = tokens.clone();
        tokens.extend(c);
    }
    let corpus = Shard { tokens };
    let (grown, sources) = grow_experts(ck, 1e-3, 0.1, now());
    let ga = GrowArgs { steps: a.steps_a + a.steps_b, lr: 3e-4, batch: a.batch, seq: a.seq, eval_every: 30, seed: now() };
    let (trained, l0, l1) = train_new_experts(&grown, &corpus, &ga, should_stop)?;
    let imp = (l0 - l1) / l0.max(1e-6);
    journal(&a.ood_dir, "growth_trained", serde_json::json!({"experts": grown.cfg.experts, "sources": sources, "held_out_before": l0, "held_out_after": l1, "improvement": imp}));
    if imp < a.gate {
        return Ok(None);
    }
    // commit: checkpoint (old kept as a generation backup) + served file
    let generation = std::fs::read_dir(a.ckpt.parent().unwrap_or(Path::new(".")))?.flatten().filter(|e| e.file_name().to_string_lossy().contains(".gen")).count();
    let backup = a.ckpt.with_extension(format!("ckpt.gen{generation}"));
    std::fs::copy(&a.ckpt, &backup)?;
    let d: Vec<(&str, &[f32])> = trained.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
    crate::train::save_checkpoint(&a.ckpt, &trained.cfg, trained.step, &trained.params, None, None, &d)?;
    let tj = std::fs::read(&a.tokenizer)?;
    let next = a.cmf.with_extension("cmf.grown");
    crate::export::export(&trained, &tj, &next)?;
    // re-append the served file's skills (their base tensors are unchanged)
    let old = cortiq_core::format::CmfModel::open(&a.cmf)?;
    let mut cur = next.clone();
    for sk in &old.header.skills {
        let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = old
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&format!("skill.{}.", sk.id)))
            .map(|t| {
                let bytes = old.tensor_bytes(&t.name).unwrap_or(&[]);
                let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                (t.name.clone(), t.shape.clone(), data)
            })
            .collect();
        let Some(sel) = sk.selection.clone() else { continue };
        let tmp = a.cmf.with_extension("cmf.grown2");
        append_to_cmf(&cur, &tmp, &sk.id, &sk.layers, &tensors, sel, sk.quality.clone().unwrap_or(serde_json::json!({})))?;
        std::fs::rename(&tmp, &next)?;
        cur = next.clone();
    }
    drop(old);
    std::fs::rename(&next, &a.cmf)?;
    let _ = crate::skill::calibrate_file(&a.cmf, 0.05);
    Ok(Some((l0, l1)))
}

/// q4tp requant of `cmf` next to it (`<stem>-q4tp.cmf`), kept only if
/// ppl(q4tp)/ppl(f32) − 1 ≤ gate on the held-out text. Uses our CLI.
fn requant_gate(bin: &str, cmf: &Path, held: &Path, gate: f32) -> anyhow::Result<Option<(f64, f64, String)>> {
    let out = cmf.with_file_name(format!("{}-q4tp.cmf", cmf.file_stem().and_then(|s| s.to_str()).unwrap_or("model")));
    let tmp = out.with_extension("cmf.tmp");
    let st = std::process::Command::new(bin).args(["requant", "--quant", "q4tp-quantize", "--output"]).arg(&tmp).arg(cmf).output()?;
    anyhow::ensure!(st.status.success(), "requant failed: {}", String::from_utf8_lossy(&st.stderr));
    let ppl = |m: &Path| -> anyhow::Result<f64> {
        let o = std::process::Command::new(bin).env("CMF_GPU", "0").args(["ppl", "--windows", "8", "--window-len", "512", "--file"]).arg(held).arg(m).output()?;
        let s = String::from_utf8_lossy(&o.stdout);
        let line = s.lines().find(|l| l.starts_with("PPL =")).ok_or_else(|| anyhow::anyhow!("no PPL line: {s}"))?;
        Ok(line.split_whitespace().nth(2).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN))
    };
    let p0 = ppl(cmf)?;
    let p1 = ppl(&tmp)?;
    if p1 / p0 - 1.0 <= gate as f64 {
        std::fs::rename(&tmp, &out)?;
        Ok(Some((p0, p1, out.display().to_string())))
    } else {
        let _ = std::fs::remove_file(&tmp);
        Ok(None)
    }
}
