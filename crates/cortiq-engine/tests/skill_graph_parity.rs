//! The resident wgpu Embryo graph with a STATIC skill pipeline
//! (`Pipeline::from_model_with_skill`) executes the skill's FFN tensors:
//! per-op CPU logits vs resident-graph logits for the backbone AND for the
//! skill pipeline of a tiny GDN + bounded-anchor genome carrying one
//! `ffn_replace` record (written by the test with the core writers), over
//! a prompt walk and a few decode steps.
//!
//! The wgpu context is process-global and `CMF_GPU` is read at backend
//! init, so the test re-runs its own binary once per leg (`cpu`, `graph`)
//! and compares the dumps. Where the graph cannot run (no wgpu adapter,
//! or the device refuses the resident path) the graph leg reports it and
//! the test SKIPS — unless `CMF_SKILL_PARITY_REQUIRE=1`.
//! Tolerance: `CMF_EMBRYO_PARITY_TOL` (default 2e-3, the resident parity
//! gate's default).

#![cfg(feature = "gpu")]

#[path = "common/embryo_synth.rs"]
mod embryo_synth;
#[path = "common/knowledge_synth.rs"]
mod knowledge_synth;

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const CHILD_ENV: &str = "CMF_SKILL_PARITY_CHILD";
const DIR_ENV: &str = "CMF_SKILL_PARITY_DIR";
const TEST_NAME: &str = "static_skill_pipeline_resident_graph_uses_skill_ffn";
const PROMPT: usize = 40;
const DECODE: usize = 6;

fn greedy() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(0),
        ..Default::default()
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// Logits of the last prompt position and of DECODE greedy steps, fed
/// with the CPU leg's tokens (`forced`) so both legs walk the same ids.
fn walk(p: &mut Pipeline, ids: &[u32], forced: Option<&[u32]>) -> (Vec<f32>, Vec<u32>, bool) {
    let mut all = p.prefill_next_logits(ids, None);
    let mut resident = p.device_state_bytes().is_some();
    let mut toks = Vec::new();
    let mut last = all.clone();
    for s in 0..DECODE {
        let t = forced.map(|f| f[s]).unwrap_or_else(|| argmax(&last));
        toks.push(t);
        last = p.decode_step_logits(t, ids.len() + s);
        all.extend_from_slice(&last);
        resident &= p.device_state_bytes().is_some();
    }
    (all, toks, resident)
}

fn write_f32(path: &Path, v: &[f32]) {
    std::fs::write(
        path,
        v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn child(leg: &str, dir: &Path) {
    let model = Arc::new(CmfModel::open(dir.join("f1.cmf")).expect("open F1"));
    let ids = embryo_synth::synth_ids(PROMPT, 11, model.arch().vocab_size);
    for (tag, skill) in [
        ("backbone", None),
        ("skill", Some(knowledge_synth::SKILL_ID)),
    ] {
        let mut p = Pipeline::from_model_with_skill(&model, greedy(), skill).expect("pipeline");
        assert_eq!(p.active_skill().is_some(), skill.is_some());
        let forced: Option<Vec<u32>> = (leg == "graph").then(|| {
            std::fs::read_to_string(dir.join(format!("cpu-{tag}.tok")))
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        });
        let (logits, toks, resident) = walk(&mut p, &ids, forced.as_deref());
        write_f32(&dir.join(format!("{leg}-{tag}.f32")), &logits);
        let toks: Vec<String> = toks.iter().map(u32::to_string).collect();
        std::fs::write(dir.join(format!("{leg}-{tag}.tok")), toks.join(" ")).unwrap();
        std::fs::write(
            dir.join(format!("{leg}-{tag}.resident")),
            resident.to_string(),
        )
        .unwrap();
        eprintln!("{leg}/{tag}: {} logits, resident={resident}", logits.len());
    }
}

fn run_leg(dir: &Path, leg: &str) {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, leg)
        .env(DIR_ENV, dir)
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_GPU_WGPU_GRAPH");
    if leg == "cpu" {
        cmd.env("CMF_GPU", "0");
    } else {
        cmd.env("CMF_GPU", "wgpu")
            .env("CMF_GPU_PROBE", "0")
            .env("CMF_GPU_WGPU_GRAPH", "1")
            .env("CMF_EMBRYO_RESIDENT", "parallel");
    }
    let out = cmd.output().expect("spawn leg");
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{leg} leg failed");
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0f32,
            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
        )
}

#[test]
fn static_skill_pipeline_resident_graph_uses_skill_ffn() {
    if let Ok(leg) = std::env::var(CHILD_ENV) {
        child(&leg, &PathBuf::from(std::env::var(DIR_ENV).unwrap()));
        return;
    }
    let dir = std::env::temp_dir().join(format!("cmf-skill-graph-{}", std::process::id()));
    // The writer computes φ on the backbone: keep it on the CPU here.
    // SAFETY: set before any pipeline of this (parent) process exists.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    knowledge_synth::write_knowledge_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    run_leg(&dir, "cpu");
    run_leg(&dir, "graph");
    let tol: f32 = std::env::var("CMF_EMBRYO_PARITY_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2e-3);
    let read = |leg: &str, tag: &str| read_f32(&dir.join(format!("{leg}-{tag}.f32")));
    let resident = |tag: &str| {
        std::fs::read_to_string(dir.join(format!("graph-{tag}.resident"))).unwrap() == "true"
    };
    // The skill must matter, or the parity below proves nothing.
    let skill_effect = max_diff(&read("cpu", "skill"), &read("cpu", "backbone"));
    assert!(
        skill_effect > 50.0 * tol,
        "the ffn_replace record barely changes the logits ({skill_effect:.3e})"
    );
    if !(resident("backbone") && resident("skill")) {
        let msg = format!(
            "resident graph did not run (backbone={}, skill={}) — no wgpu adapter or a \
             refused resident path on this host",
            resident("backbone"),
            resident("skill")
        );
        assert!(
            std::env::var("CMF_SKILL_PARITY_REQUIRE").as_deref() != Ok("1"),
            "{msg}"
        );
        eprintln!("SKIP: {msg}");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    for tag in ["backbone", "skill"] {
        let d = max_diff(&read("graph", tag), &read("cpu", tag));
        eprintln!("{tag}: resident graph vs per-op CPU max|Δ| = {d:.3e} (tol {tol:.1e})");
        assert!(
            d <= tol,
            "{tag}: graph logits diverge from CPU: {d:.3e} > {tol:.1e}"
        );
    }
    // A graph that ran the TRUNK FFN under the skill pipeline would sit on
    // the backbone's logits, `skill_effect` away from the CPU skill's.
    let wrong = max_diff(&read("graph", "skill"), &read("cpu", "backbone"));
    assert!(
        wrong > 10.0 * tol,
        "graph skill logits equal the backbone's"
    );
    eprintln!("skill effect {skill_effect:.3e}; graph-skill vs cpu-backbone {wrong:.3e}");
    let _ = std::fs::remove_dir_all(&dir);
}
