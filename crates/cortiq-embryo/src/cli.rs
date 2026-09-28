//! Trainer commands (Metal on macOS, Vulkan with `--features vulkan` on
//! Linux): step timing and the birth loop. Optimizer moments are read back
//! through owned copies (`GBuf::to_vec`) so the same code drives both
//! backends; on Metal that is the same bytes as the unified-memory slice.

pub use crate::growth::{BiasMode, DescMode, ShellMode, SourceMode};
use crate::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use crate::train::{
    AnchorGraft, Checkpoint, Mix, Sampler, Shard, append_gdn_lane_checkpoint, append_gqa_lane_checkpoint,
    graft_layer4_anchor_checkpoint, load_checkpoint, lr_at, save_checkpoint,
};
use cortiq_core::types::TensorDtype;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Fail-closed message when no training context can be built: no device,
/// or (Vulkan) a genome outside the resident graph's contract.
const NO_DEVICE: &str = "no training device: Metal (macOS) or Vulkan (`--features vulkan`, Linux); \
on Vulkan the constructor also refuses gdn_lane / learn_decay / router_smooth_k4 / router_top2_margin";
const NO_DEVICE_TEACHER: &str = "no training device for the --distill-from teacher (same contract as the student)";

/// Arguments for the host-only layer-4 donor-anchor graft and its immediate
/// fixed validation gate.
pub struct AnchorGraftArgs {
    pub student: PathBuf,
    pub donor: PathBuf,
    pub out: PathBuf,
    pub shard: Vec<String>,
    pub val: Option<PathBuf>,
    pub batch: usize,
    pub seq: usize,
}

/// Construct and immediately falsify the second-anchor candidate.  The
/// checkpoint transform itself is host-only.  Evaluation loads the student
/// and candidate sequentially so two full Metal arenas are never resident.
pub fn anchor_graft(a: AnchorGraftArgs) {
    let student = load_checkpoint(&a.student).expect("load --student checkpoint");
    let donor = load_checkpoint(&a.donor).expect("load --donor checkpoint");
    let graft = graft_layer4_anchor_checkpoint(&student, &donor)
        .expect("deterministic layer-4 anchor graft");
    println!(
        "anchor-graft source: student={} donor={} target=anchor_every=4",
        a.student.display(),
        a.donor.display()
    );
    println!(
        "anchor-graft: student {:.0} params / candidate {:.0} params; copied {} student tensors",
        graft.student_params, graft.candidate_params, graft.student_tensors
    );
    for t in &graft.donor_tensors {
        println!("  donor {} shape {:?} source {}", t.name, t.shape, t.source);
    }
    let AnchorGraft {
        checkpoint: candidate,
        ..
    } = graft;
    let candidate_params = candidate.params.clone();
    let candidate_m = candidate.m.clone();
    let candidate_v = candidate.v.clone();
    let candidate_extras = candidate.extras.clone();
    let candidate_cfg = candidate.cfg.clone();
    let candidate_step = candidate.step;
    let ex: Vec<(&str, &[f32])> = candidate_extras
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    save_checkpoint(
        &a.out,
        &candidate.cfg,
        candidate.step,
        &candidate.params,
        candidate.m.as_deref(),
        candidate.v.as_deref(),
        &ex,
    )
    .expect("save graft candidate");
    drop(candidate);
    drop(student);
    drop(donor);

    // Byte-level checkpoint framing and payload round trip before allocating
    // a Metal model.  This also proves extras/step/config were retained.
    let loaded = load_checkpoint(&a.out).expect("round-trip load graft candidate");
    assert_eq!(
        loaded.step, candidate_step,
        "candidate step changed on round trip"
    );
    assert_eq!(
        loaded.cfg.anchor_every, 4,
        "candidate cadence changed on round trip"
    );
    assert_eq!(
        serde_json::to_value(&loaded.cfg).unwrap(),
        serde_json::to_value(&candidate_cfg).unwrap()
    );
    assert_eq!(
        loaded.params, candidate_params,
        "candidate params changed on round trip"
    );
    assert_eq!(loaded.m, candidate_m, "candidate m changed on round trip");
    assert_eq!(loaded.v, candidate_v, "candidate v changed on round trip");
    assert_eq!(
        loaded.extras, candidate_extras,
        "student extras changed on round trip"
    );
    assert!(
        loaded.params.iter().all(|x| x.is_finite()),
        "candidate params are non-finite"
    );
    if let Some(m) = &loaded.m {
        assert!(m.iter().all(|x| x.is_finite()), "candidate m is non-finite");
    }
    if let Some(v) = &loaded.v {
        assert!(v.iter().all(|x| x.is_finite()), "candidate v is non-finite");
    }
    drop(loaded);

    let val = match &a.val {
        Some(path) => Shard::load(path).expect("load --val shard"),
        None if !a.shard.is_empty() => {
            let (_train, tail) = Mix::load(&a.shard, 0.005, a.seq).expect("load validation shards");
            tail
        }
        None => panic!("anchor-graft requires --val or at least one --shard"),
    };
    let val_batches = (val.tokens.len() / (a.batch * a.seq + 1)).clamp(1, 8);
    let eval_one = |path: &std::path::Path| -> f32 {
        let ck = load_checkpoint(path).expect("load evaluation checkpoint");
        let gpu = EmbryoGpu::new(ck.cfg.clone(), a.batch, a.seq, &ck.params)
            .expect("Metal evaluation model");
        gpu.set_desc(&ck.extras);
        gpu.desc_updates.set(false);
        let mut tokens = Vec::new();
        let mut targets = Vec::new();
        let mut sum = 0.0f32;
        for i in 0..val_batches {
            Sampler::fixed_batch(&val, a.batch, a.seq, i, &mut tokens, &mut targets);
            sum += gpu.eval_loss(&tokens, &targets);
        }
        drop(gpu);
        drop(ck);
        sum / val_batches as f32
    };
    let student_nll = eval_one(&a.student);
    let candidate_nll = eval_one(&a.out);
    let improvement = student_nll - candidate_nll;
    println!(
        "anchor-graft fixed-val: batches={} student_nll={student_nll:.6} candidate_nll={candidate_nll:.6} improvement={improvement:.6} gate={}",
        val_batches,
        if improvement >= 0.01 { "PASS" } else { "FAIL" }
    );
}

/// Arguments for the Phase-2 zero-output layer-4 additive GQA lane.
pub struct GqaLaneArgs {
    pub resume: PathBuf,
    pub donor: PathBuf,
    pub shard: Vec<String>,
    pub val: Option<PathBuf>,
    pub out: PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub steps: usize,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub seed: u64,
}

/// Run the bounded +50/+100 lane-only CE capacity gate.  The donor and base
/// checkpoints are inspected host-side; Metal models are created/dropped
/// sequentially and only the candidate is resident during optimization.
pub fn gqa_lane(a: GqaLaneArgs) {
    let (train, val) = {
        let holdout = if a.val.is_some() { 0.0 } else { 0.005 };
        let (mix, tail) = Mix::load(&a.shard, holdout, a.seq).expect("load shards");
        let val = match &a.val {
            Some(v) => Shard::load(v).expect("load val shard"),
            None => tail,
        };
        (mix, val)
    };
    let base = load_checkpoint(&a.resume).expect("load base checkpoint");
    let donor = load_checkpoint(&a.donor).expect("load donor checkpoint");
    let old_cfg = base.cfg.clone();
    let old_params = base.params.clone();
    let old_m = base.m.clone();
    let old_v = base.v.clone();
    let old_extras = base.extras.clone();
    let graft = append_gqa_lane_checkpoint(&base, &donor).expect("append gqa lane");
    for t in &graft.donor_tensors {
        println!(
            "gqa-lane donor {} shape {:?} source {}",
            t.name, t.shape, t.source
        );
    }
    let grown = graft.checkpoint;
    let cfg = grown.cfg.clone();
    let step0 = grown.step;
    assert!(
        a.steps >= step0 as usize + 100,
        "steps must include +100 gate"
    );
    // Fresh base NLL before candidate allocation.  This is the reference for
    // exact identity and both staged quality thresholds.
    let m = a.batch * a.seq;
    let val_batches = (val.tokens.len() / (m + 1)).clamp(1, 8);
    let eval_nll = |model: &EmbryoGpu, tok: &mut Vec<u32>, tgt: &mut Vec<u32>| -> f32 {
        (0..val_batches)
            .map(|i| {
                Sampler::fixed_batch(&val, a.batch, a.seq, i, tok, tgt);
                model.eval_loss(tok, tgt)
            })
            .sum::<f32>()
            / val_batches as f32
    };
    let mut tokens = Vec::new();
    let mut targets = Vec::new();
    let nll0 = {
        let base_ref =
            EmbryoGpu::new(old_cfg.clone(), a.batch, a.seq, &old_params).expect("Metal base");
        base_ref.set_desc(&old_extras);
        base_ref.desc_updates.set(false);
        let n = eval_nll(&base_ref, &mut tokens, &mut targets);
        drop(base_ref);
        n
    };
    let mut gpu =
        EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &grown.params).expect("Metal candidate");
    if let (Some(m0), Some(v0)) = (grown.m.as_ref(), grown.v.as_ref()) {
        gpu.m.write_from(m0);
        gpu.v.write_from(v0);
    }
    gpu.set_desc(&grown.extras);
    gpu.desc_updates.set(false);
    gpu.step = step0;
    let cand0 = eval_nll(&gpu, &mut tokens, &mut targets);
    println!(
        "gqa-lane identity: base_nll={nll0:.6} candidate_nll={cand0:.6} delta={:.3e}",
        cand0 - nll0
    );
    assert!(
        (cand0 - nll0).abs() <= 1e-5,
        "gqa lane initial identity failed"
    );
    let tail_before = gpu.params_host();
    let ranges = gpu.gqa_tail_ranges();
    assert!(!ranges.is_empty());
    let t_start = Instant::now();
    let mut sampler = crate::train::Sampler::new(a.batch, a.seq, a.seed.wrapping_add(step0 as u64));
    for step in step0 as usize..a.steps {
        sampler.batch_mix(&train, &mut tokens, &mut targets);
        let (loss, gnorm, _ms) = gpu.train_step_gqa_lane(
            &tokens,
            &targets,
            lr_at(step, a.steps, 0, a.lr, a.lr * 0.1),
            a.wd,
            a.clip,
        );
        if !loss.is_finite() || !gnorm.is_finite() || !gpu.gqa_tail_finite() {
            panic!("gqa lane non-finite at step {step}");
        }
        let done = gpu.step - step0;
        if done == 50 || done == 100 {
            let nll = eval_nll(&gpu, &mut tokens, &mut targets);
            let elapsed = t_start.elapsed().as_secs_f64().max(1e-9);
            let throughput = done as f64 * m as f64 / elapsed;
            let p = gpu.params_host();
            let changed = ranges
                .iter()
                .map(|&(o, n)| {
                    p[o..o + n]
                        .iter()
                        .zip(&tail_before[o..o + n])
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count()
                })
                .sum::<usize>();
            let path = out_staged_path(&a.out, gpu.step);
            let d = gpu.desc_host();
            let ex: Vec<(&str, &[f32])> = d.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            save_checkpoint(
                &path,
                &cfg,
                gpu.step,
                &p,
                Some(&gpu.m.to_vec()),
                Some(&gpu.v.to_vec()),
                &ex,
            )
            .expect("save staged gqa checkpoint");
            println!(
                "gqa-lane +{done}: NLL {nll:.6} Δ={:.6} throughput {throughput:.1} tok/s tail_changed={changed}",
                nll0 - nll
            );
            if done == 50 && nll > nll0 - 0.002 {
                println!("gqa-lane +50 FAIL — stopping before +100");
                return;
            }
            if done == 100 {
                let legacy = gpu.params_host()[..old_params.len()] == old_params[..]
                    && old_m
                        .as_ref()
                        .map_or(true, |x| gpu.m.to_vec()[..x.len()] == x[..])
                    && old_v
                        .as_ref()
                        .map_or(true, |x| gpu.v.to_vec()[..x.len()] == x[..]);
                let gate = nll <= nll0 - 0.005
                    && gpu.gqa_tail_finite()
                    && legacy
                    && changed > 0
                    && throughput > 0.0;
                println!(
                    "gqa-lane +100 {} legacy_byte_identity={legacy} finite={} tail_changed={changed} throughput={throughput:.1}",
                    if gate { "PASS" } else { "FAIL" },
                    gpu.gqa_tail_finite()
                );
                let p = gpu.params_host();
                save_checkpoint(
                    &a.out,
                    &cfg,
                    gpu.step,
                    &p,
                    Some(&gpu.m.to_vec()),
                    Some(&gpu.v.to_vec()),
                    &ex,
                )
                .expect("save final gqa checkpoint");
                return;
            }
        }
    }
}

fn out_staged_path(out: &PathBuf, step: u32) -> PathBuf {
    let stem = out
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("candidate");
    out.with_file_name(format!("{stem}-step-{step}.ckpt"))
}

pub fn step_bench(batch: usize, seq: usize, steps: usize, tiny: bool, cfg_json: Option<&str>) {
    let mut cfg = if tiny {
        EmbryoCfg::tiny()
    } else {
        EmbryoCfg::embryo0()
    };
    if let Some(js) = cfg_json {
        let mut base = serde_json::to_value(&cfg).expect("cfg to json");
        let over: serde_json::Value =
            serde_json::from_str(js).expect("--cfg-json must be a JSON object");
        if let (Some(b), Some(o)) = (base.as_object_mut(), over.as_object()) {
            for (k, v) in o {
                b.insert(k.clone(), v.clone());
            }
        }
        cfg = serde_json::from_value(base).expect("cfg overrides");
        println!("cfg overrides applied: {js}");
    }
    let lay = Layout::new(&cfg);
    let (total, active) = cfg.params();
    println!(
        "genome: {} layers, hidden {}, vocab {}; arena {:.2} M params (§3 count {:.1} M / {:.1} M active)",
        cfg.layers,
        cfg.hidden,
        cfg.vocab,
        lay.total as f64 / 1e6,
        total as f64 / 1e6,
        active as f64 / 1e6
    );
    let t0 = Instant::now();
    let p = init_params(&cfg, &lay, 1);
    let mut gpu = EmbryoGpu::new(cfg.clone(), batch, seq, &p).expect(NO_DEVICE);
    println!("alloc + init: {:.1} s", t0.elapsed().as_secs_f64());
    let m = batch * seq;
    let tokens: Vec<u32> = crate::ops::lcg_vec(3, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let targets: Vec<u32> = crate::ops::lcg_vec(4, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let mut best = f64::MAX;
    for s in 0..steps {
        let t = Instant::now();
        let (loss, gnorm, gpu_ms) = gpu.train_step(&tokens, &targets, 1e-4, 0.1, 1.0);
        let wall = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(wall);
        println!(
            "step {s}: loss {loss:.4} |g| {gnorm:.3} gpu {gpu_ms:.0} ms wall {wall:.0} ms → {:.0} tok/s",
            m as f64 / (wall * 1e-3)
        );
        if s + 1 == steps {
            for (l, cnt) in gpu.routing_counts().iter().enumerate() {
                println!("  layer {l} experts: {cnt:?} (of {m}, cap {})", gpu.moe_cap);
            }
        }
    }
    if std::env::var("EMBRYO_PROFILE").is_ok() {
        println!("--- per-phase GPU ms ---");
        for (name, ms) in gpu.profile_step() {
            println!("{name:<28} {ms:>8.1} ms");
        }
        for (name, ms) in gpu.profile_hk_layer(0) {
            println!("{name:<44} {ms:>8.1} ms");
        }
    }
    let flop_per_tok = 6.0 * lay.total as f64;
    println!(
        "best step {best:.0} ms = {:.0} tok/s; 6·N·tok = {:.2} TFLOPS effective; 500 M tokens ≈ {:.1} h",
        m as f64 / (best * 1e-3),
        flop_per_tok * m as f64 / (best * 1e-3) / 1e12,
        500e6 / (m as f64 / (best * 1e-3)) / 3600.0
    );
}

pub struct BirthArgs {
    pub shard: Vec<String>,
    pub val: Option<PathBuf>,
    pub out: PathBuf,
    pub resume: Option<PathBuf>,
    pub batch: usize,
    pub seq: usize,
    pub steps: usize,
    pub warmup: usize,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub eval_every: usize,
    pub save_every: usize,
    pub pca_every: usize,
    pub tiny: bool,
    pub vocab: Option<usize>,
    pub anchor_every: Option<usize>,
    /// Short-conv taps before the mixer projections (0 = off; 4 = the
    /// Qwen/LFM-class local mixing the per-token diagnosis called for)
    pub conv_k: Option<usize>,
    /// append the checkpoint-compatible one-head GDN correction lane
    pub gdn_lane: bool,
    /// use the parameter-neutral in-place Phase-Delta hybrid mixer
    pub phase_delta: bool,
    /// use Phase-Delta at exactly one zero-based hybrid layer (selection
    /// implies Phase-Delta and takes precedence over the all-layer switch)
    pub phase_delta_layer: Option<usize>,
    /// use Phase-Delta at exactly two zero-based hybrid layers (for example
    /// `--phase-delta-layers 3,6`); rejects anchors, duplicates, and out of range
    /// indices during deterministic config validation
    pub phase_delta_layers: Option<Vec<usize>>,
    /// smooth per-expert resonance scores causally over the current and
    /// preceding three positions before top-1 selection
    pub router_smooth_k4: bool,
    /// fixed reconstruction-error margin for conditional top-2 routing;
    /// score/Metal parity scaffold only (default off)
    pub router_top2_margin: Option<f32>,
    /// bounded anchor `swa_sink_v1`: served exact window in keys (0 = the
    /// legacy full-causal anchor). On `--resume` of a legacy checkpoint this
    /// switches the continuation to the bounded operator (S4 probe): the
    /// mask is semantic config, every parameter stays as loaded.
    pub anchor_window: Option<usize>,
    /// trained NoPE sink vectors per KV head (needs `--anchor-window`). On
    /// `--resume` of a checkpoint without sinks the arena is extended by the
    /// new `layers.{l}.attn.sink_k/sink_v` tensors (name-based append like
    /// growth: legacy parameters and AdamW moments bit-identical, sinks at
    /// their deterministic init with zero moments).
    pub anchor_sink: Option<usize>,
    /// SWAX stochastic training windows sampled per step, e.g. `64,128`
    /// (each ≤ the served window); the last 10% of the steps use the served
    /// window
    pub anchor_train_windows: Option<Vec<usize>>,
    /// explicit zero-based anchor schedule, e.g. `3,7` (default: every
    /// `--anchor-every`-th layer); on `--resume` it must name the
    /// checkpoint's anchors exactly (the arena layout follows it)
    pub anchor_layers: Option<Vec<usize>>,
    /// `hybrid_k` | `gdn` — the mixer of the non-anchor layers (fresh births)
    pub mixer: Option<String>,
    pub gdn_heads: Option<usize>,
    pub gdn_dk: Option<usize>,
    pub gdn_dv: Option<usize>,
    pub cfg_json: Option<String>,
    /// Warm-start donor: copy name+size-matching tensors (twin: everything
    /// but the mixers).
    pub init_from: Option<PathBuf>,
    /// Feature-distillation teacher (MSE on the final-normed hidden).
    pub distill_from: Option<PathBuf>,
    pub distill_w: f32,
    /// Teacher micro-batch (0 = the training batch).  The teacher is a full
    /// trainer allocation, so a student that already fills the card (carry +
    /// dropless at B8/T1024 = 17.4 GB of 24) keeps the teacher at 1–2 rows and
    /// runs its forward `batch / distill_batch` times per step.
    pub distill_batch: usize,
    /// Keep descriptor extras byte-identical during a bounded continuation.
    pub freeze_desc: bool,
    /// Hold the donated tensors still for the first N steps (progressive
    /// replacement: the fresh mixers learn to fit the frozen network first).
    pub freeze_donor: usize,
    pub seed: u64,
    /// State carry-over across consecutive windows of one document stream
    /// (plan S6b): stream sampler + `carry_begin`/`carry_commit` per step.
    pub carry: bool,
    /// restart every row's stream after N windows (0 = only shard end / EOT)
    pub carry_reset_every: usize,
    /// end-of-text id: a window containing it ends the row's stream
    pub carry_eot: Option<u32>,
    /// on `--resume`: run the warmup+cosine schedule over the steps of THIS
    /// run (step − resume step over steps − resume step) instead of the
    /// absolute step (a recovery/continuation with its own schedule)
    pub lr_restart: bool,
    /// dropless expert routing (capacity = every row) instead of capacity 2
    pub dropless: bool,
    /// extra held-out shards reported at every eval: `name=path`
    pub val_extra: Vec<String>,
}

pub fn birth(a: BirthArgs) {
    // corpus: shards mixed by weight (`path[:weight]`); validation = the
    // held-out tail (0.5%) of every shard, or an explicit --val shard.
    let (train, val) = {
        let holdout = if a.val.is_some() { 0.0 } else { 0.005 };
        let (mix, tail) = Mix::load(&a.shard, holdout, a.seq).expect("load shards");
        let val = match &a.val {
            Some(v) => Shard::load(v).expect("load val shard"),
            None => tail,
        };
        (mix, val)
    };
    let (mut cfg, params, step0, m0, v0, mut extras) = match &a.resume {
        Some(p) => {
            let ck = load_checkpoint(p).expect("load checkpoint");
            println!("resumed {} at step {}", p.display(), ck.step);
            (ck.cfg, ck.params, ck.step, ck.m, ck.v, ck.extras)
        }
        None => {
            let mut cfg = if a.tiny {
                EmbryoCfg::tiny()
            } else {
                EmbryoCfg::embryo0()
            };
            if let Some(v) = a.vocab {
                assert!(v % 64 == 0, "vocab must be a multiple of 64");
                cfg.vocab = v;
            }
            if let Some(ae) = a.anchor_every {
                cfg.anchor_every = ae.max(1);
            }
            if let Some(ck) = a.conv_k {
                cfg.conv_k = ck;
            }
            if a.gdn_lane {
                cfg.gdn_lane = true;
            }
            if a.phase_delta {
                cfg.phase_delta = true;
            }
            if let Some(layer) = a.phase_delta_layer {
                cfg.phase_delta_layer = Some(layer);
            }
            if let Some(layers) = &a.phase_delta_layers {
                cfg.phase_delta_layers = Some(layers.clone());
            }
            if a.router_smooth_k4 {
                cfg.router_smooth_k4 = true;
            }
            if let Some(margin) = a.router_top2_margin {
                cfg.router_top2_margin = Some(margin);
            }
            if let Some(layers) = &a.anchor_layers {
                cfg.anchor_layers = Some(layers.clone());
            }
            if let Some(w) = a.anchor_window {
                cfg.anchor_window = w;
            }
            if let Some(s) = a.anchor_sink {
                cfg.anchor_sink = s;
            }
            if let Some(ws) = &a.anchor_train_windows {
                cfg.anchor_train_windows = ws.clone();
            }
            if let Some(mx) = &a.mixer {
                cfg.mixer = match mx.as_str() {
                    "gdn" => crate::model::Mixer::Gdn,
                    "hybrid_k" => crate::model::Mixer::HybridK,
                    other => panic!("--mixer {other}: expected hybrid_k or gdn"),
                };
            }
            if let Some(v) = a.gdn_heads {
                cfg.gdn_heads = v;
            }
            if let Some(v) = a.gdn_dk {
                cfg.gdn_dk = v;
            }
            if let Some(v) = a.gdn_dv {
                cfg.gdn_dv = v;
            }
            if let Some(js) = &a.cfg_json {
                let mut base = serde_json::to_value(&cfg).expect("cfg to json");
                let over: serde_json::Value =
                    serde_json::from_str(js).expect("--cfg-json must be a JSON object");
                if let (Some(b), Some(o)) = (base.as_object_mut(), over.as_object()) {
                    for (k, v) in o {
                        b.insert(k.clone(), v.clone());
                    }
                }
                cfg = serde_json::from_value(base).expect("cfg overrides");
                println!("cfg overrides applied: {js}");
            }
            let lay = Layout::new(&cfg);
            (
                cfg.clone(),
                init_params(&cfg, &lay, a.seed),
                0,
                None,
                None,
                Vec::new(),
            )
        }
    };
    // Explicit Phase-Delta CLI switches also apply when resuming a checkpoint;
    // a selector is semantic config, not an arena migration.  Fresh births
    // already applied these above (before any --cfg-json merge).
    if a.resume.is_some() {
        if a.phase_delta {
            cfg.phase_delta = true;
        }
        if let Some(layer) = a.phase_delta_layer {
            cfg.phase_delta_layer = Some(layer);
        }
        if let Some(layers) = &a.phase_delta_layers {
            cfg.phase_delta_layers = Some(layers.clone());
        }
        if a.router_smooth_k4 {
            cfg.router_smooth_k4 = true;
        }
        if let Some(margin) = a.router_top2_margin {
            cfg.router_top2_margin = Some(margin);
        }
    }
    if let Some(v) = a.vocab {
        cfg.vocab = v;
    }
    // Explicitly growing a baseline checkpoint is a name-based append-only
    // migration: every legacy parameter and optimizer moment is copied
    // bit-for-bit, while the new lane tail keeps its deterministic init/zero
    // moments.  The gdn lane and the bounded-anchor sinks are the only
    // resume paths that change the arena shape.
    let mut m0 = m0;
    let mut v0 = v0;
    let mut params = params;
    if a.resume.is_some()
        && (a.anchor_window.is_some()
            || a.anchor_sink.is_some()
            || a.anchor_train_windows.is_some()
            || a.anchor_layers.is_some())
    {
        if let Some(layers) = &a.anchor_layers {
            // The layout follows the schedule: only a schedule naming the
            // checkpoint's anchors exactly keeps every tensor in place.
            let want: Vec<usize> = (0..cfg.layers).filter(|&l| cfg.is_anchor(l)).collect();
            let mut got = layers.clone();
            got.sort_unstable();
            assert_eq!(
                got, want,
                "--anchor-layers {got:?} does not match the checkpoint's anchors {want:?}"
            );
            cfg.anchor_layers = Some(layers.clone());
        }
        let window = a.anchor_window.unwrap_or(cfg.anchor_window);
        let sink = a.anchor_sink.unwrap_or(cfg.anchor_sink);
        let train_windows = a
            .anchor_train_windows
            .clone()
            .unwrap_or_else(|| cfg.anchor_train_windows.clone());
        assert!(
            window > 0,
            "--anchor-sink/--anchor-train-windows on resume need a window (--anchor-window)"
        );
        let grown = crate::train::append_anchor_sinks_checkpoint(
            &crate::train::Checkpoint {
                cfg: cfg.clone(),
                step: step0,
                params: params.clone(),
                m: m0.take(),
                v: v0.take(),
                extras: extras.clone(),
            },
            window,
            sink,
            &train_windows,
            a.seed,
        )
        .expect("bounded-anchor continuation");
        let added = grown.params.len() - params.len();
        cfg = grown.cfg;
        params = grown.params;
        m0 = grown.m;
        v0 = grown.v;
        extras = grown.extras;
        println!(
            "bounded anchor: window {window} sink {sink} train_windows {train_windows:?}; appended {added} parameters; legacy prefix/moments copied"
        );
    }
    if a.gdn_lane && !cfg.gdn_lane {
        let grown = append_gdn_lane_checkpoint(
            &crate::train::Checkpoint {
                cfg: cfg.clone(),
                step: step0,
                params: params.clone(),
                m: m0.take(),
                v: v0.take(),
                extras: extras.clone(),
            },
            a.seed,
        )
        .expect("append gdn lane");
        let added = grown.params.len() - params.len();
        cfg = grown.cfg;
        params = grown.params;
        m0 = grown.m;
        v0 = grown.v;
        extras = grown.extras;
        println!("gdn-lane: appended {added} parameters; legacy prefix/moments copied");
    }
    let lay = Layout::new(&cfg);
    let mut donated: Vec<(usize, usize)> = Vec::new();
    if let Some(ip) = &a.init_from {
        let don = load_checkpoint(ip).expect("load --init-from donor");
        let dlay = Layout::new(&don.cfg);
        let dmap: std::collections::HashMap<&str, (usize, usize)> = dlay
            .names
            .iter()
            .map(|(n, o, l)| (n.as_str(), (*o, *l)))
            .collect();
        let (mut hit, mut miss) = (0usize, 0usize);
        for (n, o, l) in &lay.names {
            match dmap.get(n.as_str()) {
                Some((doff, dlen)) if dlen == l => {
                    params[*o..*o + *l].copy_from_slice(&don.params[*doff..*doff + *l]);
                    donated.push((*o, *l));
                    hit += 1;
                }
                _ => miss += 1,
            }
        }
        println!(
            "init-from {}: {hit} tensors copied, {miss} left at init (the fresh mixers)",
            ip.display()
        );
        // The donor's expert descriptors travel with its expert weights —
        // fresh descriptors against copied experts skew the routing.
        if extras.is_empty() {
            extras = don.extras;
        }
    }
    let params = params;
    // Feature-distillation teacher: its own arena, forward-only use.
    let teacher = a.distill_from.as_ref().map(|tp| {
        let ck = load_checkpoint(tp).expect("load --distill-from teacher");
        assert_eq!(ck.cfg.hidden, cfg.hidden, "teacher hidden must match");
        assert_eq!(ck.cfg.vocab, cfg.vocab, "teacher vocab must match");
        let tb = if a.distill_batch == 0 { a.batch } else { a.distill_batch };
        assert!(
            tb <= a.batch && a.batch % tb == 0,
            "--distill-batch must divide --batch"
        );
        let mut t = EmbryoGpu::new(ck.cfg.clone(), tb, a.seq, &ck.params).expect(NO_DEVICE_TEACHER);
        t.set_desc(&ck.extras);
        println!(
            "distill-from {} (step {}), w = {}, teacher batch {}",
            tp.display(),
            ck.step,
            a.distill_w,
            tb
        );
        (t, tb)
    });
    println!(
        "genome: {} layers, hidden {}, vocab {}, arena {:.2} M; train {} tok, val {} tok; B={} T={} steps={}",
        cfg.layers,
        cfg.hidden,
        cfg.vocab,
        lay.total as f64 / 1e6,
        train.total_tokens(),
        val.tokens.len(),
        a.batch,
        a.seq,
        a.steps
    );
    let mut gpu = if a.carry {
        EmbryoGpu::new_carry(cfg.clone(), a.batch, a.seq, &params, a.dropless).expect(NO_DEVICE)
    } else if a.dropless {
        EmbryoGpu::new_eval_dropless(cfg.clone(), a.batch, a.seq, &params).expect(NO_DEVICE)
    } else {
        EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &params).expect(NO_DEVICE)
    };
    if a.dropless {
        println!("dropless routing: capacity = every row");
    }
    if a.carry {
        println!(
            "carry: state carried across windows (reset every {} windows, eot {:?}); anchor keys carried: {}",
            a.carry_reset_every,
            a.carry_eot,
            if gpu.carry_pad() > 0 { format!("{} columns", gpu.carry_pad()) } else { "no (legacy full-causal anchor)".into() }
        );
    }
    if let (Some(m), Some(v)) = (m0, v0) {
        gpu.m.write_from(&m);
        gpu.v.write_from(&v);
    }
    gpu.set_desc(&extras);
    if a.freeze_desc {
        gpu.desc_updates.set(false);
        println!("descriptor updates disabled");
    }
    gpu.step = step0;
    if a.freeze_donor > 0 && !donated.is_empty() {
        gpu.freeze = donated.clone();
        println!(
            "freeze-donor: {} tensors held for the first {} steps",
            donated.len(),
            a.freeze_donor
        );
    }
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed.wrapping_add(step0 as u64));
    let mut stream = a.carry.then(|| {
        crate::train::StreamSampler::new(
            a.batch,
            a.seq,
            a.seed.wrapping_add(step0 as u64),
            a.carry_reset_every,
            a.carry_eot,
        )
    });
    let mut carry_acc = a.carry.then(|| CarryAcc::new(cfg.layers));
    let mut last_reset: Vec<bool> = vec![true; a.batch];
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    let m = a.batch * a.seq;
    // Keep the historical ten-step log cadence unless explicitly requested.
    // Phase-Delta terminal diagnostics are a separate opt-in readback and
    // therefore cannot perturb the timed train step or the legacy control.
    let log_every = std::env::var("EMBRYO_LOG_EVERY")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(10);
    let phase_telemetry =
        cfg.phase_delta_active() && std::env::var_os("EMBRYO_PHASE_TELEMETRY").is_some();
    let val_batches = (val.tokens.len() / (m + 1)).clamp(1, 8);
    let eval = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| -> f32 {
        let mut s = 0.0f32;
        for i in 0..val_batches {
            Sampler::fixed_batch(&val, a.batch, a.seq, i, tokens, targets);
            s += gpu.eval_loss(tokens, targets);
        }
        s / val_batches as f32
    };
    // extra held-out shards (`name=path`), reported next to the main one
    let val_extra: Vec<(String, Shard)> = a
        .val_extra
        .iter()
        .map(|spec| {
            let (name, path) = spec.split_once('=').expect("--val-extra name=path");
            (name.to_string(), Shard::load(Path::new(path)).expect("load --val-extra shard"))
        })
        .collect();
    let eval_extra = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| -> String {
        let mut out = String::new();
        for (name, sh) in &val_extra {
            let nb = (sh.tokens.len() / (m + 1)).clamp(1, 8);
            let mut s = 0.0f32;
            for i in 0..nb {
                Sampler::fixed_batch(sh, a.batch, a.seq, i, tokens, targets);
                s += gpu.eval_loss(tokens, targets);
            }
            out.push_str(&format!("  {name} {:.4} (ppl {:.2}, {nb} batches)", s / nb as f32, (s / nb as f32).exp()));
        }
        out
    };
    let t_start = Instant::now();
    let mut ema = 0.0f32;
    let mut cov_ema: Vec<f32> = Vec::new();
    // the last 10% of the run trains on the served anchor window (relative
    // to the resume point with --lr-restart, like the schedule)
    let served_from = if a.lr_restart {
        a.steps - (a.steps - step0 as usize) / 10
    } else {
        a.steps - a.steps / 10
    };
    for step in step0 as usize..a.steps {
        if let Some(st) = stream.as_mut() {
            let reset = st.batch_mix(&train, &mut tokens, &mut targets);
            gpu.carry_begin(&reset);
            last_reset = reset;
        } else {
            sampler.batch_mix(&train, &mut tokens, &mut targets);
        }
        if !cfg.anchor_train_windows.is_empty()
            && step >= served_from
            && !gpu.anchor_fixed_window.get()
        {
            gpu.anchor_fixed_window.set(true);
            println!(
                "bounded anchor: step {step}: training on the served window {} from here",
                cfg.anchor_window
            );
        }
        if step == a.freeze_donor && !gpu.freeze.is_empty() {
            gpu.freeze.clear();
            println!("freeze-donor: thawed at step {step}");
        }
        let lr = if a.lr_restart {
            lr_at(step - step0 as usize, a.steps - step0 as usize, a.warmup, a.lr, a.lr * 0.1)
        } else {
            lr_at(step, a.steps, a.warmup, a.lr, a.lr * 0.1)
        };
        // AdamW's decoupled decay shrinks even zero-grad params — no weight
        // decay while the donor is held, or 1000 frozen steps cost it ~6%.
        let wd = if gpu.freeze.is_empty() { a.wd } else { 0.0 };
        let t = Instant::now();
        let (loss, dloss, gnorm, _gpu_ms) = match &teacher {
            Some((tch, tb)) => {
                let xft = if *tb == a.batch {
                    tch.forward_hidden(&tokens)
                } else {
                    // teacher micro-batches: rows are independent, so the
                    // concatenation is the full-batch teacher hidden exactly
                    let mut xft = Vec::with_capacity(a.batch * a.seq * cfg.hidden);
                    for chunk in tokens.chunks_exact(tb * a.seq) {
                        xft.extend_from_slice(&tch.forward_hidden(chunk));
                    }
                    xft
                };
                gpu.train_step_distill(&tokens, &targets, lr, wd, a.clip, &xft, a.distill_w)
            }
            None => {
                let (l, g, ms) = gpu.train_step(&tokens, &targets, lr, wd, a.clip);
                (l, 0.0, g, ms)
            }
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if let (Some(st), Some(acc)) = (stream.as_ref(), carry_acc.as_mut()) {
            let per = gpu.per_position_loss();
            gpu.carry_commit();
            let stats = gpu.carry_state_stats();
            let line = acc.push(gpu.step, &per, a.seq, &last_reset, &st.depths(), &stats);
            if step % log_every == 0 {
                println!("{line}");
            }
        }
        ema = if step == step0 as usize {
            loss
        } else {
            0.98 * ema + 0.02 * loss
        };
        if a.pca_every > 0 && (step + 1) % a.pca_every == 0 {
            gpu.update_subspaces(&mut cov_ema, 0.9);
        }
        if step % log_every == 0 || step + 1 == a.steps {
            let dtag = if teacher.is_some() {
                format!(" dist {dloss:.4}")
            } else {
                String::new()
            };
            println!(
                "step {step:>6} loss {loss:.4} (ema {ema:.4}){dtag} |g| {gnorm:.3} lr {lr:.2e} {ms:.0} ms {:.0} tok/s  [{:.1} min]",
                m as f64 / (ms * 1e-3),
                t_start.elapsed().as_secs_f64() / 60.0
            );
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
        }
        // The recurrence witness is intentionally sampled only after the
        // final timed step.  It reads host-visible activation/state buffers
        // and performs a bounded host reconstruction; no command buffer from
        // this diagnostic is included in the reported step duration.
        if phase_telemetry && step + 1 == a.steps {
            for x in gpu.phase_delta_telemetry() {
                println!(
                    "phase_delta telemetry layer={} beta_p01={:.6} beta_p50={:.6} beta_p99={:.6} r_rms={:.6e} e_rms={:.6e} correction_rms={:.6e} v_rms={:.6e} state_rms={:.6e} state_max={:.6e} Wk_grad={:.6e} Wv_grad={:.6e} Wkappa_grad={:.6e} finite={}",
                    x.layer,
                    x.beta_p01,
                    x.beta_p50,
                    x.beta_p99,
                    x.r_rms,
                    x.e_rms,
                    x.correction_rms,
                    x.v_rms,
                    x.state_rms,
                    x.state_max,
                    x.wk_grad_l2,
                    x.wv_grad_l2,
                    x.wkappa_grad_l2,
                    x.finite,
                );
            }
        }
        if !loss.is_finite() {
            eprintln!("loss is not finite at step {step} — stopping");
            break;
        }
        // with --lr-restart the eval/save cadence counts from the resume point,
        // so "every 500" means 500/1000/… steps into THIS run
        let tick = if a.lr_restart { step + 1 - step0 as usize } else { step + 1 };
        if tick % a.eval_every == 0 || step + 1 == a.steps {
            let vl = eval(&gpu, &mut tokens, &mut targets);
            println!("  val loss {vl:.4}  ppl {:.2}{}", vl.exp(), eval_extra(&gpu, &mut tokens, &mut targets));
            let rc = gpu.routing_counts();
            if !rc.is_empty() {
                let cap = gpu.moe_cap;
                let summary: Vec<String> = rc
                    .iter()
                    .map(|c| {
                        let dropped: u32 = c.iter().map(|&n| n.saturating_sub(cap as u32)).sum();
                        format!(
                            "{:?}{}",
                            c,
                            if dropped > 0 {
                                format!("(-{dropped})")
                            } else {
                                String::new()
                            }
                        )
                    })
                    .collect();
                println!("  experts/layer: {}", summary.join(" "));
            }
        }
        if tick % a.save_every == 0 || step + 1 == a.steps {
            let p = gpu.params_host();
            let d = gpu.desc_host();
            let ex: Vec<(&str, &[f32])> = d.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            save_checkpoint(
                &a.out,
                &cfg,
                gpu.step,
                &p,
                Some(&gpu.m.to_vec()),
                Some(&gpu.v.to_vec()),
                &ex,
            )
            .expect("save");
            println!("  saved {} (step {})", a.out.display(), gpu.step);
        }
    }
    if let Some(acc) = carry_acc.as_ref() {
        print!("{}", acc.report());
    }
}

/// Depth profile of a checkpoint (no-grad): one carried stream of
/// `windows` consecutive T-windows per batch row — the mean NLL of every
/// window index and the per-layer recurrent-state RMS/max after it — and the
/// same windows fresh (state reset every window) as the reference. Tells
/// whether the model breaks with depth in the TRAINER forward (state drift
/// / overflow) or only in the runtime long-prefix path.
pub struct CarryProfileArgs {
    pub ckpt: PathBuf,
    pub shard: PathBuf,
    pub windows: usize,
    pub batch: usize,
    pub seq: usize,
    pub seed: u64,
    /// also run the first `one_pass` windows of the first two rows as ONE
    /// long forward (T = one_pass·seq, no carry) and print its per-window
    /// NLL next to the carried numbers (carry ≡ one pass witness)
    pub one_pass: usize,
}

pub fn carry_profile(a: CarryProfileArgs) {
    let ck = load_checkpoint(&a.ckpt).expect("load checkpoint");
    let shard = Shard::load(&a.shard).expect("load shard");
    let cfg = ck.cfg.clone();
    let mut gpu = EmbryoGpu::new_carry(cfg.clone(), a.batch, a.seq, &ck.params, true).expect(NO_DEVICE);
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false);
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    println!(
        "carry-profile: {} (step {}), {} layers, mixer {:?}, anchor window {} (carried columns {}), B={} T={} windows={} shard {}",
        a.ckpt.display(),
        ck.step,
        cfg.layers,
        cfg.mixer,
        cfg.anchor_window,
        gpu.carry_pad(),
        a.batch,
        a.seq,
        a.windows,
        a.shard.display()
    );
    let mut st = crate::train::StreamSampler::new(a.batch, a.seq, a.seed, a.windows, None);
    let mut wins: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(a.windows);
    for _ in 0..a.windows {
        let _ = st.batch(&shard, &mut tokens, &mut targets);
        wins.push((tokens.clone(), targets.clone()));
    }
    let row_stats = |per: &[f32]| -> (f64, f32, f32) {
        let rows: Vec<f32> = (0..a.batch)
            .map(|bi| per[bi * a.seq..(bi + 1) * a.seq].iter().sum::<f32>() / a.seq as f32)
            .collect();
        (
            rows.iter().map(|x| *x as f64).sum::<f64>() / a.batch as f64,
            rows.iter().cloned().fold(f32::INFINITY, f32::min),
            rows.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    };
    // reference: every window fresh
    let mut fresh = Vec::with_capacity(a.windows);
    for (tk, tg) in &wins {
        gpu.carry_begin(&vec![true; a.batch]);
        let _ = gpu.eval_loss(tk, tg);
        fresh.push(row_stats(&gpu.per_position_loss()));
        gpu.carry_commit();
    }
    // one long forward of the first two rows (no carry): the carried stream
    // must reproduce it window by window
    let one_pass: Vec<f64> = if a.one_pass > 0 {
        let n = a.one_pass.min(a.windows);
        let tt = n * a.seq;
        // CMF_WITNESS_ROWS=k (default 2): the one-pass model gets k rows, rows
        // 2.. being other windows' rows — a batch-dependence probe (only rows
        // 0-1 are compared)
        let rows: usize = std::env::var("CMF_WITNESS_ROWS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(2)
            .max(2);
        let one = EmbryoGpu::new_eval_dropless(cfg.clone(), rows, tt, &ck.params).expect(NO_DEVICE);
        one.set_desc(&ck.extras);
        one.desc_updates.set(false);
        let (mut tk, mut tg) = (Vec::new(), Vec::new());
        for row in 0..rows {
            let src_row = row % 2;
            let w0 = if row < 2 { 0 } else { (row / 2) % a.windows };
            for w in 0..n {
                let wi = (w0 + w) % a.windows;
                tk.extend_from_slice(&wins[wi].0[src_row * a.seq..(src_row + 1) * a.seq]);
                tg.extend_from_slice(&wins[wi].1[src_row * a.seq..(src_row + 1) * a.seq]);
            }
        }
        if rows > 2 {
            println!("one-pass witness with {rows} rows (rows 2.. = other windows)");
        }
        let _ = one.eval_loss(&tk, &tg);
        let per = one.per_position_loss();
        // where inside the window the two disagree: the same tokens through
        // the T-row model (fresh, rows 0-1) vs the one-pass model, in 64-bins
        {
            gpu.carry_begin(&vec![true; a.batch]);
            let _ = gpu.eval_loss(&wins[0].0, &wins[0].1);
            let fresh0 = gpu.per_position_loss();
            gpu.carry_commit();
            // CMF_WITNESS_ROUTE=1: per layer, how many of rows 0-1's tokens
            // were routed to a different expert by the two models, and the
            // largest resonance difference on tokens routed the same way
            if std::env::var("CMF_WITNESS_ROUTE").is_ok() {
                for l in 0..gpu.moe.len().min(one.moe.len()) {
                    let a1: Vec<u32> = gpu.moe[l].assign.to_vec().iter().map(|x| x.to_bits()).collect();
                    let a2: Vec<u32> = one.moe[l].assign.to_vec().iter().map(|x| x.to_bits()).collect();
                    let r1 = gpu.moe[l].res.to_vec();
                    let r2 = one.moe[l].res.to_vec();
                    let (mut flips, mut dres, mut n) = (0usize, 0.0f32, 0usize);
                    let mut where_ = Vec::new();
                    let mut first_diff: Option<(usize, usize, f32)> = None;
                    for row in 0..2 {
                        for j in 0..a.seq {
                            let (i1, i2) = (row * a.seq + j, row * tt + j);
                            n += 1;
                            let d = (r1[i1] - r2[i2]).abs();
                            if d > 0.0 && first_diff.is_none() {
                                first_diff = Some((row, j, d));
                            }
                            if a1[i1] != a2[i2] {
                                flips += 1;
                                if where_.len() < 12 {
                                    where_.push((row, j));
                                }
                            } else {
                                dres = dres.max(d);
                            }
                        }
                    }
                    println!(
                        "route witness L{l}: {flips}/{n} tokens routed differently, max|Δres| same-routed {dres:.3e}; first Δres≠0 at {first_diff:?}; flips at {where_:?}"
                    );
                }
            }
            {
                // the 8 largest per-token NLL differences (row, pos, Δ)
                let mut top: Vec<(f32, usize, usize)> = Vec::new();
                for row in 0..2 {
                    for j in 0..a.seq {
                        top.push((fresh0[row * a.seq + j] - per[row * tt + j], row, j));
                    }
                }
                top.sort_by(|x, y| y.0.abs().partial_cmp(&x.0.abs()).unwrap());
                let n_big = top.iter().filter(|x| x.0.abs() > 0.5).count();
                println!(
                    "window 1 per-token |Δ|>0.5: {n_big}/{}; largest: {:?}",
                    2 * a.seq,
                    top.iter().take(8).map(|(d, r, j)| (*r, *j, format!("{d:+.2}"))).collect::<Vec<_>>()
                );
            }
            let bins = a.seq / 64;
            let mut line = String::from("window 1, fresh(T rows) − one-pass, per 64-position bin:");
            for bi in 0..bins {
                let mut d = 0.0f64;
                for row in 0..2 {
                    for j in bi * 64..(bi + 1) * 64 {
                        d += fresh0[row * a.seq + j] as f64 - per[row * tt + j] as f64;
                    }
                }
                line.push_str(&format!(" {:+.1e}", d / 128.0));
            }
            println!("{line}");
        }
        (0..n)
            .map(|w| {
                (0..2)
                    .map(|row| per[row * tt + w * a.seq..row * tt + (w + 1) * a.seq].iter().sum::<f32>() as f64 / a.seq as f64)
                    .sum::<f64>()
                    / 2.0
            })
            .collect()
    } else {
        Vec::new()
    };
    // carried stream
    println!("window  carried NLL (min..max rows)   fresh NLL   Δ      [rows 0-1: carried vs one-pass] | per-layer state rms / max after the window");
    for (w, (tk, tg)) in wins.iter().enumerate() {
        gpu.carry_begin(&vec![w == 0; a.batch]);
        let _ = gpu.eval_loss(tk, tg);
        let per = gpu.per_position_loss();
        let (m, lo, hi) = row_stats(&per);
        let two: f64 = (0..2.min(a.batch))
            .map(|row| per[row * a.seq..(row + 1) * a.seq].iter().sum::<f32>() as f64 / a.seq as f64)
            .sum::<f64>()
            / 2.0f64.min(a.batch as f64);
        gpu.carry_commit();
        let stats = gpu.carry_state_stats();
        let cmp = if w < one_pass.len() {
            format!("[{two:.4} vs {:.4}, Δ {:+.2e}]", one_pass[w], two - one_pass[w])
        } else {
            String::new()
        };
        let mut line = format!(
            "{:>4}    {m:.4} ({lo:.3}..{hi:.3})   {:.4}   {:+.4} {cmp} |",
            w + 1,
            fresh[w].0,
            m - fresh[w].0
        );
        for (l, sst) in stats.iter().enumerate() {
            if let Some((rms, mx, _)) = sst {
                line.push_str(&format!(" L{l} {rms:.2e}/{mx:.2e}"));
            }
        }
        println!("{line}");
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
    }
}

/// Carry-over diagnostics (plan S6b): the per-window training loss split
/// into fresh windows (stream start) and carried windows, by stream depth,
/// with the recurrent-state RMS/max of every layer at the window boundary.
pub struct CarryAcc {
    layers: usize,
    /// per step: (fresh_sum, fresh_n, carried_sum, carried_n)
    ring: std::collections::VecDeque<(f64, usize, f64, usize)>,
    /// depth (windows since reset, 1-based, capped at 32) → (loss sum, n)
    depth_loss: Vec<(f64, usize)>,
    /// [layer][depth] → (rms sum, n)
    depth_rms: Vec<Vec<(f64, usize)>>,
    pub steps: usize,
}

impl CarryAcc {
    pub fn new(layers: usize) -> CarryAcc {
        CarryAcc {
            layers,
            ring: std::collections::VecDeque::new(),
            depth_loss: vec![(0.0, 0); 33],
            depth_rms: vec![vec![(0.0, 0); 33]; layers],
            steps: 0,
        }
    }
    /// Record one step; returns the log line.
    pub fn push(
        &mut self,
        step: u32,
        per_pos: &[f32],
        t: usize,
        reset: &[bool],
        depths: &[usize],
        stats: &[Option<(f32, f32, Vec<f32>)>],
    ) -> String {
        let b = reset.len();
        let (mut fs, mut fnn, mut cs, mut cn) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut dmax = 0usize;
        for bi in 0..b {
            let row: f64 = per_pos[bi * t..(bi + 1) * t].iter().map(|x| *x as f64).sum::<f64>() / t as f64;
            let d = depths[bi].clamp(1, 32);
            dmax = dmax.max(depths[bi]);
            self.depth_loss[d].0 += row;
            self.depth_loss[d].1 += 1;
            if reset[bi] {
                fs += row;
                fnn += 1;
            } else {
                cs += row;
                cn += 1;
            }
            for (l, st) in stats.iter().enumerate() {
                if let Some((_, _, rows)) = st {
                    self.depth_rms[l][d].0 += rows[bi] as f64;
                    self.depth_rms[l][d].1 += 1;
                }
            }
        }
        self.ring.push_back((fs, fnn, cs, cn));
        if self.ring.len() > 100 {
            self.ring.pop_front();
        }
        self.steps += 1;
        let mut line = format!(
            "carry step={step} fresh={} (n={fnn}) carried={} (n={cn}) depth_max={dmax}",
            if fnn > 0 { format!("{:.4}", fs / fnn as f64) } else { "-".into() },
            if cn > 0 { format!("{:.4}", cs / cn as f64) } else { "-".into() },
        );
        for (l, st) in stats.iter().enumerate() {
            if let Some((rms, mx, _)) = st {
                line.push_str(&format!(" | L{l} rms={rms:.3e} max={mx:.3e}"));
            }
        }
        line
    }
    /// Last-100-step means (fresh, carried, gap) — the stage gate.
    pub fn last100(&self) -> (f64, f64, f64) {
        let (mut fs, mut fnn, mut cs, mut cn) = (0.0, 0usize, 0.0, 0usize);
        for (a, b, c, d) in &self.ring {
            fs += a;
            fnn += b;
            cs += c;
            cn += d;
        }
        let f = if fnn > 0 { fs / fnn as f64 } else { f64::NAN };
        let c = if cn > 0 { cs / cn as f64 } else { f64::NAN };
        (f, c, c - f)
    }
    pub fn report(&self) -> String {
        let (f, c, g) = self.last100();
        let mut out = format!(
            "carry summary: last-100-step fresh={f:.4} carried={c:.4} gap={g:+.4} nats ({} steps)\n",
            self.steps
        );
        out.push_str("carry depth table (windows since reset → mean window loss, per-layer state rms):\n");
        for d in 1..=32 {
            let (ls, ln) = self.depth_loss[d];
            if ln == 0 {
                continue;
            }
            out.push_str(&format!("  depth {d:>2}: loss {:.4} (n={ln})", ls / ln as f64));
            for l in 0..self.layers {
                let (rs, rn) = self.depth_rms[l][d];
                if rn > 0 {
                    out.push_str(&format!("  L{l} {:.3e}", rs / rn as f64));
                }
            }
            out.push('\n');
        }
        out
    }
}

/// Synthetic recall probe (plan S7 metrics: MQAR / NIAH-lite) through the
/// TRAINER forward — no runtime, so the number is the operator the trainer
/// optimizes, not an export.
pub struct ProbeRecallArgs {
    pub ckpt: PathBuf,
    /// filler text source (u16 shard)
    pub shard: PathBuf,
    /// K key→value pairs per sequence
    pub pairs: usize,
    /// query distances D (tokens between a pair and its query)
    pub dists: Vec<usize>,
    /// sequences per distance
    pub trials: usize,
    pub seed: u64,
    /// skip distances whose window T would exceed this (attention scores
    /// are materialised [T, T] per anchor head in the trainer)
    pub max_seq: usize,
}

/// MQAR/NIAH-lite: `K` random (key, value) token pairs are written at
/// position 16 of a real-text window, the sequence continues with real text,
/// and at distance `D` after the pairs one key is repeated; the model must
/// predict its value. Reports accuracy(D) (argmax over the full vocabulary),
/// mean NLL and rank of the value at the query, the per-position NLL of the
/// whole window by position bucket, and the NLL on repeated vs fresh filler
/// tokens (the `CMF_PPL_TRACE` diagnostic of the plan).
pub fn probe_recall(a: ProbeRecallArgs) {
    let ck = load_checkpoint(&a.ckpt).expect("load checkpoint");
    let shard = Shard::load(&a.shard).expect("load filler shard");
    let cfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    assert_eq!(ck.params.len(), lay.total, "checkpoint/layout mismatch");
    let (h, v, ncl) = (cfg.hidden, cfg.vocab, cfg.head_clusters);
    let e = &ck.params[lay.embed..lay.embed + v * h];
    let cm = if ncl > 0 {
        &ck.params[lay.head_clusters..lay.head_clusters + ncl * h]
    } else {
        &[][..]
    };
    // full-vocabulary log-probs of one final-normed hidden row (the
    // hierarchical head as the runtime evaluates it)
    let logprobs = |x: &[f32]| -> Vec<f32> {
        if ncl == 0 {
            let mut lg: Vec<f32> = (0..v)
                .map(|i| (0..h).map(|j| e[i * h + j] * x[j]).sum())
                .collect();
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + lg.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
            for z in &mut lg {
                *z -= lse;
            }
            return lg;
        }
        let cs = v / ncl;
        let mut lc: Vec<f32> = (0..ncl)
            .map(|c| (0..h).map(|j| cm[c * h + j] * x[j]).sum())
            .collect();
        let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + lc.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
        for z in &mut lc {
            *z -= lse;
        }
        let mut out = vec![0.0f32; v];
        for c in 0..ncl {
            let lg: Vec<f32> = (0..cs)
                .map(|s| (0..h).map(|j| e[(c * cs + s) * h + j] * x[j]).sum())
                .collect();
            let bm = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bl = bm + lg.iter().map(|z| (z - bm).exp()).sum::<f32>().ln();
            for s in 0..cs {
                out[c * cs + s] = lc[c] + lg[s] - bl;
            }
        }
        out
    };
    let mut rng = a.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5245_4341_4C4C;
    let mut next = move || -> u64 {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let k = a.pairs.max(1);
    let p0 = 16usize;
    // key/value ids: the upper 3/4 of the vocabulary (away from bytes / specials)
    let id_lo = (v / 4) as u64;
    let id_span = (v as u64 - id_lo).max(1);
    println!(
        "probe-recall: {} (step {}), {} layers, mixer {:?}, anchor window {}, K={k}, trials {}, filler {}",
        a.ckpt.display(),
        ck.step,
        cfg.layers,
        cfg.mixer,
        cfg.anchor_window,
        a.trials,
        a.shard.display()
    );
    let buckets = [0usize, 64, 256, 1024, 4096, usize::MAX];
    for &d in &a.dists {
        let need = p0 + 2 * k + d + 2 + 8;
        let t = need.div_ceil(64) * 64;
        if t > a.max_seq {
            println!("D={d}: window T={t} exceeds --max-seq {} — skipped", a.max_seq);
            continue;
        }
        let mut gpu = EmbryoGpu::new(cfg.clone(), 1, t, &ck.params).expect(NO_DEVICE);
        gpu.set_desc(&ck.extras);
        gpu.desc_updates.set(false);
        let mut hits = 0usize;
        let mut nll_q = 0.0f64;
        let mut rank_q = 0.0f64;
        let mut bsum = vec![0.0f64; buckets.len() - 1];
        let mut bcnt = vec![0usize; buckets.len() - 1];
        let (mut rep_sum, mut rep_n, mut fresh_sum, mut fresh_n) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut tokens = vec![0u32; t];
        let mut targets = vec![0u32; t];
        let t0 = Instant::now();
        for _ in 0..a.trials {
            let n = shard.tokens.len();
            let start = (next() % (n - t - 2) as u64) as usize;
            for i in 0..t + 1 {
                tokens.push(0);
                tokens[i.min(t - 1)] = shard.tokens[start + i.min(t - 1)] as u32;
            }
            tokens.truncate(t);
            let next_tok = shard.tokens[start + t] as u32;
            // distinct keys, random values
            let mut keys: Vec<u32> = Vec::with_capacity(k);
            while keys.len() < k {
                let c = (id_lo + next() % id_span) as u32;
                if !keys.contains(&c) {
                    keys.push(c);
                }
            }
            let vals: Vec<u32> = (0..k).map(|_| (id_lo + next() % id_span) as u32).collect();
            for i in 0..k {
                tokens[p0 + 2 * i] = keys[i];
                tokens[p0 + 2 * i + 1] = vals[i];
            }
            let j = (next() % k as u64) as usize;
            let pq = p0 + 2 * k + d; // the query key sits here; its value is the target
            tokens[pq] = keys[j];
            tokens[pq + 1] = vals[j];
            for i in 0..t - 1 {
                targets[i] = tokens[i + 1];
            }
            targets[t - 1] = next_tok;
            // query: full-vocabulary log-probs from the final hidden
            let xf = gpu.forward_hidden(&tokens);
            let lp = logprobs(&xf[pq * h..(pq + 1) * h]);
            let want = vals[j] as usize;
            let (argmax, _) = lp
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap();
            hits += (argmax == want) as usize;
            nll_q += -(lp[want] as f64);
            rank_q += lp.iter().filter(|&&z| z > lp[want]).count() as f64;
            // per-position NLL of the whole window (trainer head)
            let _ = gpu.eval_loss(&tokens, &targets);
            let per = gpu.per_position_loss();
            let mut seen = std::collections::HashSet::new();
            for i in 0..t {
                let in_pairs = (p0..p0 + 2 * k).contains(&i) || i == pq || i + 1 == pq || i == pq + 1;
                let b = (0..buckets.len() - 1)
                    .find(|&bi| i >= buckets[bi] && i < buckets[bi + 1])
                    .unwrap();
                bsum[b] += per[i] as f64;
                bcnt[b] += 1;
                if !in_pairs && i + 1 < t {
                    if seen.contains(&targets[i]) {
                        rep_sum += per[i] as f64;
                        rep_n += 1;
                    } else {
                        fresh_sum += per[i] as f64;
                        fresh_n += 1;
                    }
                }
                seen.insert(tokens[i]);
            }
        }
        let n = a.trials as f64;
        println!(
            "D={d:>6} T={t:>6}: acc {:.3} ({hits}/{})  NLL@query {:.3}  mean rank {:.1}  [{:.1} s]",
            hits as f64 / n,
            a.trials,
            nll_q / n,
            rank_q / n,
            t0.elapsed().as_secs_f64()
        );
        let mut parts = Vec::new();
        for bi in 0..buckets.len() - 1 {
            if bcnt[bi] > 0 {
                let hi = if buckets[bi + 1] == usize::MAX {
                    "∞".to_string()
                } else {
                    buckets[bi + 1].to_string()
                };
                parts.push(format!("[{},{hi}) {:.3}", buckets[bi], bsum[bi] / bcnt[bi] as f64));
            }
        }
        println!("         per-position NLL: {}", parts.join("  "));
        println!(
            "         filler NLL: repeated {:.3} (n={rep_n})  fresh {:.3} (n={fresh_n})  gap {:+.3}",
            rep_sum / rep_n.max(1) as f64,
            fresh_sum / fresh_n.max(1) as f64,
            rep_sum / rep_n.max(1) as f64 - fresh_sum / fresh_n.max(1) as f64
        );
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
    }
}

/// Arguments for the bounded lane-only final-hidden teacher-residual run.
/// This is deliberately a separate command from `birth`: no CE loss, legacy
/// updates, descriptor/PCA mutation, or optimizer-clock reset is permitted.
pub struct ResidualPretrainArgs {
    pub shard: Vec<String>,
    pub val: Option<PathBuf>,
    pub resume: PathBuf,
    pub teacher: PathBuf,
    pub out: PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub steps: usize,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub seed: u64,
}

/// Run the D22/D23 staged, lane-only teacher-residual experiment.
pub fn residual_pretrain(a: ResidualPretrainArgs) {
    let (train, val) = {
        let holdout = if a.val.is_some() { 0.0 } else { 0.005 };
        let (mix, tail) = Mix::load(&a.shard, holdout, a.seq).expect("load shards");
        let val = match &a.val {
            Some(v) => Shard::load(v).expect("load val shard"),
            None => tail,
        };
        (mix, val)
    };
    let base = load_checkpoint(&a.resume).expect("load residual base checkpoint");
    let teacher_ck = load_checkpoint(&a.teacher).expect("load residual teacher checkpoint");
    assert!(
        teacher_ck.cfg.hidden == base.cfg.hidden && teacher_ck.cfg.vocab == base.cfg.vocab,
        "teacher hidden/vocab must match the base checkpoint"
    );
    assert!(
        a.steps >= base.step as usize + 100,
        "steps must include +100 gate"
    );
    assert!(
        !base.cfg.gdn_lane,
        "resume must be the immutable pre-lane checkpoint"
    );

    let old_cfg = base.cfg.clone();
    let old_params = base.params.clone();
    let old_m = base.m.clone();
    let old_v = base.v.clone();
    let old_extras = base.extras.clone();
    let grown = append_gdn_lane_checkpoint(&base, a.seed).expect("append gdn lane");
    let cfg = grown.cfg.clone();
    let grown_step = grown.step;
    // Build the candidate before any other full Metal model.  The interrupted
    // attempt constructed teacher + candidate + base reference concurrently;
    // on unified memory that transient triple can exceed the process limit even
    // though no trainer duplicate exists.  The candidate copies its params,
    // moments, and descriptors, so the host-side grown checkpoint can be
    // released before constructing the next model.
    let mut gpu =
        EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &grown.params).expect("Metal candidate");
    if let (Some(m), Some(v)) = (grown.m.as_ref(), grown.v.as_ref()) {
        gpu.m.write_from(m);
        gpu.v.write_from(v);
    }
    gpu.set_desc(&grown.extras);
    gpu.desc_updates.set(false);
    gpu.step = grown_step;
    drop(grown);
    let initial_params = gpu.params_host();
    let initial_m = gpu.m.to_vec();
    let initial_v = gpu.v.to_vec();
    let initial_desc = gpu.desc_host();
    assert_eq!(initial_params[..old_params.len()], old_params[..]);
    if let (Some(m), Some(v)) = (old_m.as_ref(), old_v.as_ref()) {
        assert_eq!(
            &initial_m[..m.len()],
            &m[..],
            "legacy m changed during append"
        );
        assert_eq!(
            &initial_v[..v.len()],
            &v[..],
            "legacy v changed during append"
        );
    }

    let m = a.batch * a.seq;
    let val_batches = (val.tokens.len() / (m + 1)).clamp(1, 8);
    let eval_nll = |model: &EmbryoGpu, tok: &mut Vec<u32>, tgt: &mut Vec<u32>| -> f32 {
        let mut s = 0.0f32;
        for i in 0..val_batches {
            Sampler::fixed_batch(&val, a.batch, a.seq, i, tok, tgt);
            s += model.eval_loss(tok, tgt);
        }
        s / val_batches as f32
    };
    let eval_residual = |model: &EmbryoGpu, tch: &mut EmbryoGpu, tok: &mut Vec<u32>| -> f32 {
        let mut s = 0.0f32;
        for i in 0..val_batches {
            let mut tgt = Vec::new();
            Sampler::fixed_batch(&val, a.batch, a.seq, i, tok, &mut tgt);
            let ht = tch.forward_hidden(tok);
            s += model.eval_hidden_mse(tok, &ht);
        }
        s / val_batches as f32
    };
    let mut tokens = Vec::new();
    let mut targets = Vec::new();
    Sampler::fixed_batch(&val, a.batch, a.seq, 0, &mut tokens, &mut targets);
    // Instantiate the immutable pre-lane model briefly for the required
    // candidate/base hidden identity and fresh NLL0.  Drop it before the
    // staged run so the large recurrent scratch arena is not retained.
    let base_ref = EmbryoGpu::new(old_cfg.clone(), a.batch, a.seq, &base.params)
        .expect("Metal base reference");
    base_ref.set_desc(&base.extras);
    base_ref.desc_updates.set(false);
    // The immutable checkpoint is no longer needed once the base reference has
    // copied its parameters/descriptors into Metal; keep only the byte-level
    // witnesses captured above for the terminal legacy-identity check.
    drop(base);
    let base0 = base_ref.forward_hidden(&tokens);
    let candidate0 = gpu.forward_hidden(&tokens);
    let initial_hidden_max = candidate0
        .iter()
        .zip(&base0)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("residual identity: candidate/base hidden max {initial_hidden_max:.3e}");
    assert!(
        initial_hidden_max <= 1e-5,
        "candidate/base identity failed: {initial_hidden_max:.3e}"
    );
    let nll0 = eval_nll(&base_ref, &mut tokens, &mut targets);
    drop(base_ref);
    // Construct the teacher only after the base reference has been dropped.
    // This keeps the steady run to candidate + teacher Metal arenas (and one
    // host checkpoint), avoiding the attempt-1 triple-model OOM window.
    let mut teacher = EmbryoGpu::new(teacher_ck.cfg.clone(), a.batch, a.seq, &teacher_ck.params)
        .expect("Metal teacher");
    teacher.set_desc(&teacher_ck.extras);
    teacher.desc_updates.set(false);
    drop(teacher_ck);
    let er0 = eval_residual(&gpu, &mut teacher, &mut tokens);
    println!(
        "residual gates: NLL0 {nll0:.6} (PPL {:.3}) E_R0 {er0:.6} over {val_batches} batches",
        nll0.exp()
    );
    let tail_before = initial_params.clone();
    let t_start = Instant::now();
    let mut teacher_ms = 0.0f64;
    let mut candidate_ms = 0.0f64;
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed.wrapping_add(grown_step as u64));
    let step0 = grown_step as usize;
    for step in step0..a.steps {
        sampler.batch_mix(&train, &mut tokens, &mut targets);
        let tt = Instant::now();
        let ht = teacher.forward_hidden(&tokens);
        teacher_ms += tt.elapsed().as_secs_f64() * 1e3;
        let ct = Instant::now();
        let (er_train, gnorm, _gpu_ms) = gpu.train_step_residual(
            &tokens,
            &ht,
            lr_at(step, a.steps, 0, a.lr, a.lr * 0.1),
            a.wd,
            a.clip,
        );
        candidate_ms += ct.elapsed().as_secs_f64() * 1e3;
        if !er_train.is_finite() || !gnorm.is_finite() {
            panic!("residual pretrain non-finite at step {step}: E_R {er_train} |g| {gnorm}");
        }
        let completed = gpu.step - grown_step;
        if completed == 50 || completed == 100 {
            let er = eval_residual(&gpu, &mut teacher, &mut tokens);
            let nll = eval_nll(&gpu, &mut tokens, &mut targets);
            let elapsed = t_start.elapsed().as_secs_f64();
            let throughput = (completed as f64 * m as f64) / elapsed.max(1e-9);
            println!(
                "stage +{completed}: E_R {er:.6} ({:.3}×E_R0) NLL {nll:.6} PPL {:.3} throughput {:.1} tok/s",
                er / er0.max(1e-12),
                nll.exp(),
                throughput
            );
            let path = staged_checkpoint_path(&a.out, gpu.step);
            let p = gpu.params_host();
            let d = gpu.desc_host();
            let ex: Vec<(&str, &[f32])> = d.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            save_checkpoint(
                &path,
                &cfg,
                gpu.step,
                &p,
                Some(&gpu.m.to_vec()),
                Some(&gpu.v.to_vec()),
                &ex,
            )
            .expect("save staged residual checkpoint");
            if completed == 50
                && (er > 0.98 * er0
                    || !er.is_finite()
                    || !gpu.gdn_tail_finite()
                    || !gpu.gdn_state_finite())
            {
                println!("stage +50 FAIL — stopping before +100");
                return;
            }
            if completed == 100 {
                let finite = gpu.gdn_tail_finite() && gpu.gdn_state_finite();
                let legacy = legacy_identity(
                    &gpu,
                    &old_params,
                    old_m.as_deref(),
                    old_v.as_deref(),
                    &old_extras,
                    &initial_desc,
                );
                let tail_after = gpu.params_host();
                let engaged = gpu
                    .lay
                    .gdn
                    .iter()
                    .flatten()
                    .map(|go| {
                        ranges_for_go(go, gpu.cfg.hidden)
                            .iter()
                            .map(|&(off, n)| {
                                tail_after[off..off + n]
                                    .iter()
                                    .zip(&tail_before[off..off + n])
                                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                                    .count()
                            })
                            .sum::<usize>()
                    })
                    .collect::<Vec<_>>();
                println!(
                    "stage +100 integrity: finite={finite} legacy_byte_identity={legacy} tail_changed_per_layer={engaged:?}"
                );
                let gate = er <= 0.90 * er0
                    && nll <= (nll0 - 0.005).min(4.0277)
                    && throughput >= 0.85 * 2258.6
                    && finite
                    && legacy
                    && engaged.iter().all(|&n| n > 0);
                println!(
                    "stage +100 {} (teacher {:.0} ms, candidate {:.0} ms)",
                    if gate { "PASS" } else { "FAIL" },
                    teacher_ms,
                    candidate_ms
                );
                let p = gpu.params_host();
                let d = gpu.desc_host();
                let ex: Vec<(&str, &[f32])> = d.iter().map(|(n, x)| (*n, x.as_slice())).collect();
                save_checkpoint(
                    &a.out,
                    &cfg,
                    gpu.step,
                    &p,
                    Some(&gpu.m.to_vec()),
                    Some(&gpu.v.to_vec()),
                    &ex,
                )
                .expect("save final residual checkpoint");
                if !gate {
                    println!("residual pretrain gate failed; no matched end-task probe authorized");
                }
            }
        }
        if gpu.step - grown_step >= 100 {
            break;
        }
    }
}

fn staged_checkpoint_path(out: &PathBuf, step: u32) -> PathBuf {
    let stem = out
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("candidate");
    out.with_file_name(format!("{stem}-step-{step}.ckpt"))
}

fn ranges_for_go(go: &crate::model::GdnOffs, h: usize) -> Vec<(usize, usize)> {
    let mut r = vec![
        (go.qkvz, 256 * h),
        (go.conv, 192 * 4),
        (go.ab, 64 * h),
        (go.norm, 64),
        (go.wo, h * 64),
    ];
    if go.alog != usize::MAX {
        r.push((go.alog, 1));
    }
    if go.dt_bias != usize::MAX {
        r.push((go.dt_bias, 1));
    }
    if go.gain != usize::MAX {
        r.push((go.gain, 1));
    }
    r
}

fn legacy_identity(
    gpu: &EmbryoGpu,
    params: &[f32],
    m: Option<&[f32]>,
    v: Option<&[f32]>,
    extras: &[(String, Vec<f32>)],
    initial_desc: &[(&'static str, Vec<f32>)],
) -> bool {
    let p = gpu.params_host();
    if p[..params.len()] != params[..] {
        return false;
    }
    if let Some(m0) = m {
        if gpu.m.to_vec()[..m0.len()] != m0[..] {
            return false;
        }
    }
    if let Some(v0) = v {
        if gpu.v.to_vec()[..v0.len()] != v0[..] {
            return false;
        }
    }
    let now = gpu.desc_host();
    now.len() == initial_desc.len()
        && now
            .iter()
            .zip(initial_desc)
            .all(|((na, xa), (nb, xb))| na == nb && xa == xb)
        && (extras.is_empty()
            || now
                .iter()
                .all(|(n, x)| extras.iter().any(|(en, ex)| n == en && x == ex)))
}

// ───────────────────────── grow (growth records) ─────────────────────────

/// Prefix of the gate refusal of [`grow`] (exit status 2 in `main`).
pub const GROW_REJECTED: &str = "growth REJECTED (gate)";

/// `cortiq-embryo grow …` (SPEC_GROWTH_RECORDS §3 + addendum).
pub struct GrowCli {
    pub ckpt: PathBuf,
    pub tokenizer: PathBuf,
    pub corpus: Vec<PathBuf>,
    pub held: Vec<PathBuf>,
    pub general: Option<PathBuf>,
    /// token budget of the trace passes (0 = whole corpus): evenly spaced windows
    pub trace_tokens: usize,
    /// document budget of the held-out coverage trace (0 = all)
    pub trace_docs: usize,
    pub experts: usize,
    pub layers: Option<Vec<usize>>,
    /// What the shells are calibrated on (`--shell-mode`).
    pub shell_mode: ShellMode,
    pub shell_quantile: f32,
    /// general-target: the fraction of a layer's general tokens a grown
    /// expert's shell may admit (`--shell-target-shift`).
    pub shell_target_shift: f32,
    /// The grown bias, frozen for the whole training (`--bias-mode`).
    pub bias_mode: BiasMode,
    /// Where the copies and their descriptors come from (`--source-mode`):
    /// `novel` needs `--general`.
    pub source_mode: SourceMode,
    /// `τ_l` = this quantile of the general shard's min trunk error
    /// (`--novel-quantile`); the novelty pass runs whenever `--general` is
    /// given (reported for both source modes).
    pub novel_quantile: f32,
    /// Whether the grown descriptors move while training (`--desc-mode`;
    /// None = the source mode's default: adapt for hottest, frozen for
    /// novel).
    pub desc_mode: Option<DescMode>,
    pub record_out: Option<PathBuf>,
    pub base: Option<PathBuf>,
    pub id: Option<String>,
    pub out_ckpt: Option<PathBuf>,
    pub export: Option<PathBuf>,
    pub steps: usize,
    pub lr: f32,
    pub batch: usize,
    pub seq: usize,
    pub gate: f32,
    /// Held-out budget in `[batch, seq]` batches (0 = the default 16).
    pub held_batches: usize,
    pub noise: f32,
    pub shift: f32,
    pub seed: u64,
}

/// Tokenize document files (jsonl(.gz) "text" / .txt) into one flat shard
/// (an EOT after every document), the documents themselves (for the
/// document-wise witness) and their description for the record's origin
/// (`{role, path, docs, tokens, sha256}`).
fn tokenize_docs(
    paths: &[PathBuf],
    role: &str,
    bpe: &crate::tokenizer::Bpe,
    eot: u16,
) -> anyhow::Result<(Shard, Vec<Vec<u16>>, Vec<serde_json::Value>)> {
    let mut toks: Vec<u16> = Vec::new();
    let mut docs: Vec<Vec<u16>> = Vec::new();
    let mut roles = Vec::new();
    let mut cache = std::collections::HashMap::new();
    for p in paths {
        let t0 = toks.len();
        let n = crate::data::for_each_doc(p, |text| {
            let mut ids = Vec::new();
            bpe.encode(text, &mut cache, &mut ids);
            let d: Vec<u16> = ids.iter().map(|&i| i as u16).collect();
            toks.extend_from_slice(&d);
            toks.push(eot);
            docs.push(d);
        })
        .map_err(|e| anyhow::anyhow!("--{role} {}: {e}", p.display()))?;
        roles.push(serde_json::json!({
            "role": role,
            "path": p.display().to_string(),
            "docs": n,
            "tokens": toks.len() - t0,
            "sha256": crate::skill::sha256_file(p)?,
        }));
    }
    Ok((Shard { tokens: toks }, docs, roles))
}

/// The canonical file of `p` (its directory canonicalized + file name),
/// for comparing output paths that may not exist yet.
fn canonical_target(flag: &str, p: &Path) -> anyhow::Result<PathBuf> {
    let name = p
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("{flag} {} names no file", p.display()))?;
    let parent = match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let parent_c = std::fs::canonicalize(&parent)
        .map_err(|e| anyhow::anyhow!("{flag} directory {}: {e}", parent.display()))?;
    Ok(parent_c.join(name))
}

/// `--export` writes a FULL grown genome with the core writer, which
/// truncates its target: refuse an existing file (a genome file is never
/// rewritten — [`crate::export::refuse_genome_overwrite`] — and a full
/// export never replaces anything), and refuse the paths of `--base` and
/// `--record-out`.
fn check_export_path(export: &Path, base: Option<&Path>, record_out: Option<&Path>) -> anyhow::Result<()> {
    crate::export::refuse_genome_overwrite(export)?;
    anyhow::ensure!(
        !export.exists(),
        "refusing to overwrite existing --export {} (a full grown genome is a new file)",
        export.display()
    );
    let ex_c = canonical_target("--export", export)?;
    if let Some(b) = base {
        let b_c = std::fs::canonicalize(b).map_err(|e| anyhow::anyhow!("--base {}: {e}", b.display()))?;
        anyhow::ensure!(
            ex_c != b_c,
            "refusing: --export {} is --base {} (the genome file is never rewritten)",
            export.display(),
            b.display()
        );
    }
    if let Some(r) = record_out {
        anyhow::ensure!(
            ex_c != canonical_target("--record-out", r)?,
            "refusing: --export and --record-out name the same file {}",
            export.display()
        );
    }
    Ok(())
}

/// The checkpoint the runtime EXECUTES for `base`'s genome: for an f16
/// genome every tensor the export stores in half precision is rounded
/// f32 → f16 → f32 in the arena (`export::served_params`), so the grown
/// copies, their training and every witness (shells, coverage, routing
/// shift) run on the trunk the runtime computes x2 with; an f32 genome
/// returns the checkpoint unchanged. Returns (checkpoint, encoding,
/// values that changed).
pub fn served_checkpoint(ck: &Checkpoint, base: &cortiq_core::format::CmfModel) -> anyhow::Result<(Checkpoint, String, usize)> {
    let g = base
        .header
        .genome
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--base carries no GENOME"))?;
    let dtype = match g.encoding.as_str() {
        "f32" => TensorDtype::F32,
        "f16" => TensorDtype::F16,
        e => anyhow::bail!("genome '{}' is encoded as {e}; growth binds to an f32/f16 genome", g.id),
    };
    let vocab = base
        .vocab
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--base has no VOCAB section (tokenizer)"))?;
    let params = crate::export::served_params(ck, vocab, dtype)?;
    let changed = params.iter().zip(&ck.params).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    Ok((
        Checkpoint {
            cfg: ck.cfg.clone(),
            step: ck.step,
            params,
            m: None,
            v: None,
            extras: ck.extras.clone(),
        },
        g.encoding.clone(),
        changed,
    ))
}

/// The tokenized inputs of a growth ([`grow`]) or a reshell
/// ([`reshell`]): the growth corpus (+ `--held`, else the 10 % tail), the
/// trace budgets and the general shard.
pub struct GrowInputs {
    pub bpe: crate::tokenizer::Bpe,
    pub eot: u16,
    pub train: Shard,
    pub held: Shard,
    pub held_docs: Vec<Vec<u16>>,
    pub held_source: String,
    /// `{role, path, docs, tokens, sha256}` per input file (the record's origin).
    pub roles: Vec<serde_json::Value>,
    /// The train corpus within `--trace-tokens` (evenly spaced windows).
    pub train_trace: Shard,
    /// The held-out documents within `--trace-docs` (evenly spaced).
    pub held_docs_trace: Vec<Vec<u16>>,
    pub general: Option<Shard>,
}

/// Tokenize the corpus (+ `--held`), split the held-out, apply the trace
/// budgets (the wins / inits / shells passes see evenly spaced windows,
/// the coverage trace evenly spaced held-out documents; training uses
/// everything) and load `--general`.
#[allow(clippy::too_many_arguments)]
fn load_growth_inputs(
    tokenizer: &Path,
    corpus: &[PathBuf],
    held: &[PathBuf],
    general: Option<&Path>,
    seq: usize,
    trace_tokens: usize,
    trace_docs: usize,
    what: &str,
) -> anyhow::Result<GrowInputs> {
    use crate::growth::{split_docs_at_eot, split_tail};
    let bpe = crate::tokenizer::Bpe::load(tokenizer)?;
    let eot = bpe.special_id(crate::tokenizer::EOT).unwrap_or(0) as u16;
    let (corpus_sh, corpus_docs, mut roles) = tokenize_docs(corpus, "corpus", &bpe, eot)?;
    let (train, held_sh, held_docs, held_source) = if held.is_empty() {
        let (t, h) = split_tail(&corpus_sh, seq)?;
        let docs = split_docs_at_eot(&h.tokens, eot);
        (t, h, docs, "10% tail of --corpus".to_string())
    } else {
        let (h, d, r) = tokenize_docs(held, "held", &bpe, eot)?;
        roles.extend(r);
        (corpus_sh, h, d, "--held files (group held-out)".to_string())
    };
    drop(corpus_docs);
    let train_trace = subsample_windows(&train, seq, trace_tokens);
    let held_docs_trace = subsample_docs(&held_docs, trace_docs);
    let general = match general {
        Some(p) => {
            let s = Shard::load(p).map_err(|e| anyhow::anyhow!("--general {}: {e}", p.display()))?;
            roles.push(serde_json::json!({
                "role": "general", "path": p.display().to_string(), "tokens": s.tokens.len(),
                "sha256": crate::skill::sha256_file(p)?,
            }));
            Some(s)
        }
        None => None,
    };
    eprintln!(
        "{what}: train {} tokens, held-out {} tokens in {} documents ({held_source}){}",
        train.tokens.len(),
        held_sh.tokens.len(),
        held_docs.len(),
        general
            .as_ref()
            .map(|g| format!(", general {} tokens", g.tokens.len()))
            .unwrap_or_default()
    );
    Ok(GrowInputs {
        bpe,
        eot,
        train,
        held: held_sh,
        held_docs,
        held_source,
        roles,
        train_trace,
        held_docs_trace,
        general,
    })
}

/// `--shell-mode general-target` calibrates on the general shard: refuse
/// without `--general`.
fn check_shell_args(mode: ShellMode, q: f32, target: f32, general: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        q.is_finite() && (0.0..=1.0).contains(&q),
        "--shell-quantile {q} must lie in [0, 1]"
    );
    anyhow::ensure!(
        target.is_finite() && (0.0..=1.0).contains(&target),
        "--shell-target-shift {target} must lie in [0, 1]"
    );
    anyhow::ensure!(
        mode != ShellMode::GeneralTarget || general,
        "--shell-mode general-target calibrates the shells on the general shard: pass --general \
         <shard.u16> (or --shell-mode won-quantile)"
    );
    Ok(())
}

/// `--source-mode novel` takes its novelty threshold from the general
/// shard: refuse without `--general`; the quantile must be a fraction.
fn check_source_args(mode: SourceMode, novel_quantile: f32, general: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        novel_quantile.is_finite() && (0.0..=1.0).contains(&novel_quantile),
        "--novel-quantile {novel_quantile} must lie in [0, 1]"
    );
    anyhow::ensure!(
        mode != SourceMode::Novel || general,
        "--source-mode novel takes the novelty threshold τ from the general shard: pass --general \
         <shard.u16> (or --source-mode hottest)"
    );
    Ok(())
}

/// The novelty pass on the PRE-growth instance (whenever `--general` is
/// given): `τ_l` = the `q` quantile of the general shard's min trunk
/// error per grown layer, then the novel set of the corpus trace.
fn novelty_pass(
    gpu0: &EmbryoGpu,
    inputs: &GrowInputs,
    layers: &[usize],
    desc0: &crate::growth::GrowthDesc,
    q: f32,
    seed: u64,
) -> anyhow::Result<Option<(Vec<crate::growth::NovelSet>, usize)>> {
    use crate::growth::{novel_sets, novel_taus, trunk_min_errors};
    let Some(g) = &inputs.general else {
        return Ok(None);
    };
    let errs = trunk_min_errors(gpu0, g, layers, desc0)?;
    let n_gen = errs.first().map(|v| v.len()).unwrap_or(0);
    let taus = novel_taus(&errs, q)?;
    drop(errs);
    let sets = novel_sets(gpu0, &inputs.train_trace, layers, desc0, &taus, seed)?;
    eprintln!(
        "novelty (q {q}): τ per grown layer {:?}; novel share of the corpus trace {:?} ({:?} of {} tokens; \
         {} general tokens)",
        taus,
        sets.iter().map(|s| s.share()).collect::<Vec<_>>(),
        sets.iter().map(|s| s.novel).collect::<Vec<_>>(),
        sets.first().map(|s| s.tokens).unwrap_or(0),
        n_gen
    );
    Ok(Some((sets, n_gen)))
}

/// The grown descriptors of `trained` (`[grown layer index][k]` μ and U)
/// equal `inits` bit for bit — the witness of [`DescMode::Frozen`].
fn grown_desc_equals_inits(trained: &Checkpoint, e0: usize, layers: &[usize], inits: &[Vec<crate::growth::ClusterInit>]) -> bool {
    let (e1, h) = (trained.cfg.experts, trained.cfg.hidden);
    let k = crate::model::MOE_K;
    let ex = |name: &str| trained.extras.iter().find(|(n, _)| n == name).map(|(_, x)| x.as_slice());
    let (Some(mu), Some(u)) = (ex("desc.mu"), ex("desc.u")) else {
        return false;
    };
    layers.iter().zip(inits).all(|(&l, row)| {
        row.iter().enumerate().all(|(kk, c)| {
            let et = e0 + kk;
            c.rows > 0
                && c.mu.len() == h
                && c.u.len() == k * h
                && mu[(l * e1 + et) * h..(l * e1 + et + 1) * h].iter().zip(&c.mu).all(|(a, b)| a.to_bits() == b.to_bits())
                && u[(l * e1 + et) * k * h..(l * e1 + et + 1) * k * h].iter().zip(&c.u).all(|(a, b)| a.to_bits() == b.to_bits())
        })
    })
}

/// What [`grow`] and [`reshell`] share after training: on a dropless
/// instance of the trained checkpoint the witness with EXACTLY the runtime
/// formula — shells by `--shell-mode` (the won-quantile of the TRAIN
/// trace, or calibrated on `--general` to admit at most
/// `--shell-target-shift` of a layer's general tokens), `coverage` on the
/// held-out documents rendered as prompts (cmf-im-v1 frame, one per row),
/// `routing_shift` on `--general` (no shell / shell) — then the grown
/// checkpoint (`--out-ckpt`), the record (`--record-out`: the grown bias
/// as `--bias-mode` fixes it, verified against the trained checkpoint),
/// the full export (`--export`) and the one JSON summary line. The
/// caller's `origin` / `quality` / `summary` maps carry what only it
/// knows (the training; `"reshell": true`); the shared fields are added
/// here.
pub struct GrowFinish<'a> {
    /// The trained grown checkpoint (`E0 + K` experts in every layer).
    pub trained: &'a Checkpoint,
    /// The PRE-growth checkpoint the record binds with (`RecordArgs::ck0`).
    pub ck0: &'a Checkpoint,
    pub e0: usize,
    /// The grown layers (ascending).
    pub layers: &'a [usize],
    /// `[grown layer index][k]` trunk sources of the copies.
    pub sources: &'a [Vec<usize>],
    pub inputs: &'a GrowInputs,
    pub shell_mode: ShellMode,
    pub shell_quantile: f32,
    pub shell_target_shift: f32,
    pub bias_mode: BiasMode,
    pub source_mode: SourceMode,
    pub desc_mode: DescMode,
    /// `--novel-quantile`.
    pub novel_quantile: f32,
    /// The novelty pass (with `--general`): τ, novel shares, the masks
    /// for `novel_coverage`.
    pub novel: Option<&'a crate::growth::NovelWitness>,
    pub batch: usize,
    pub seq: usize,
    /// `(out, base, id)` of `--record-out`.
    pub record: Option<(&'a Path, &'a Path, &'a str)>,
    pub genome: Option<&'a cortiq_core::knowledge::GenomeInfo>,
    pub out_ckpt: Option<&'a Path>,
    pub export: Option<&'a Path>,
    pub base: Option<&'a Path>,
    pub tokenizer: &'a Path,
    /// `{encoding, rounded_values}` of the served trunk.
    pub served: serde_json::Value,
    pub origin: serde_json::Map<String, serde_json::Value>,
    pub quality: serde_json::Map<String, serde_json::Value>,
    pub summary: serde_json::Map<String, serde_json::Value>,
}

pub fn finish_growth(f: GrowFinish) -> anyhow::Result<serde_json::Value> {
    use crate::growth::{
        GrowthDesc, RecordArgs, coverage, expected_grown_bias, frame_docs, grown_bias_of, novel_coverage,
        routing_shift, shells_general_target, shells_won_quantile, trace_routes, trace_routes_rows,
        write_growth_record,
    };
    let (trained, e0, layers, inp) = (f.trained, f.e0, f.layers, f.inputs);
    check_shell_args(f.shell_mode, f.shell_quantile, f.shell_target_shift, inp.general.is_some())?;
    check_source_args(f.source_mode, f.novel_quantile, inp.general.is_some())?;
    anyhow::ensure!(
        f.source_mode == SourceMode::Hottest || f.novel.is_some(),
        "--source-mode novel without a novelty witness"
    );
    // ---- the grown bias: what --bias-mode fixes, verified against the
    // trained checkpoint (frozen for the whole training) ----
    let grown_bias = expected_grown_bias(f.ck0, layers, f.sources, f.bias_mode)?;
    let have = grown_bias_of(trained, e0, layers)?;
    for (li, &l) in layers.iter().enumerate() {
        for (kk, (h, w)) in have[li].iter().zip(&grown_bias[li]).enumerate() {
            anyhow::ensure!(
                h.to_bits() == w.to_bits(),
                "the trained checkpoint carries bias {h} for grown expert {} of layer {l}; --bias-mode {} \
                 wants {w} (source expert {})",
                e0 + kk,
                f.bias_mode.as_str(),
                f.sources[li][kk]
            );
        }
    }
    // ---- the grown checkpoint first: the expensive part is done ----
    if let Some(out_ckpt) = f.out_ckpt {
        let d: Vec<(&str, &[f32])> = trained
            .extras
            .iter()
            .map(|(n, x)| (n.as_str(), x.as_slice()))
            .collect();
        save_checkpoint(out_ckpt, &trained.cfg, trained.step, &trained.params, None, None, &d)?;
        eprintln!("grown checkpoint saved → {}", out_ckpt.display());
    }
    // ---- the runtime formula on the host: shells, coverage, routing shift ----
    let desc = GrowthDesc::from_checkpoint(trained, e0)?;
    let frame = crate::skill::phi_spec(&inp.bpe, 0).ok();
    let (prefix, suffix): (Vec<u32>, Vec<u32>) = match &frame {
        Some(fr) => (fr.prefix_ids.clone(), fr.suffix_ids.clone()),
        None => (Vec::new(), Vec::new()),
    };
    let (witness, cov, held_rows, shift, ncov) = {
        let gpu = EmbryoGpu::new_eval_dropless(trained.cfg.clone(), f.batch, f.seq, &trained.params)
            .ok_or_else(|| anyhow::anyhow!(NO_DEVICE))?;
        gpu.set_desc(&trained.extras);
        gpu.desc_updates.set(false);
        let tr_train = trace_routes(&gpu, &inp.train_trace, layers, &desc)?;
        let tr_general = match &inp.general {
            Some(g) => Some(trace_routes(&gpu, g, layers, &desc)?),
            None => None,
        };
        let witness = match f.shell_mode {
            ShellMode::WonQuantile => shells_won_quantile(&tr_train, f.shell_quantile),
            ShellMode::GeneralTarget => shells_general_target(
                &tr_train,
                tr_general.as_deref().expect("checked above"),
                f.shell_quantile,
                f.shell_target_shift,
            )?,
        };
        // the held-out as the runtime sees it: one document per row,
        // framed, fresh state at position 0
        let (rows, spans) = frame_docs(&inp.held_docs_trace, &prefix, &suffix, f.seq, inp.eot as u32)?;
        let tr_held = trace_routes_rows(&gpu, &rows, &spans, layers, &desc)?;
        let cov = coverage(&tr_held, &witness.shells);
        let shift = tr_general.as_ref().map(|g| routing_shift(g, &witness.shells));
        // the novel corpus tokens (trunk novelty, pre-growth instance) the
        // trained expert wins — the same trace windows in the same order
        let ncov = match f.novel {
            Some(nv) => Some(novel_coverage(&tr_train, &nv.masks, &witness.shells)?),
            None => None,
        };
        (witness, cov, rows.len(), shift, ncov)
    };
    let shells = &witness.shells;
    let wins = &witness.wins;
    let held_witness = serde_json::json!({
        "frame": if frame.is_some() { "cmf-im-v1" } else { "none" },
        "docs": inp.held_docs.len(), "rows": held_rows, "tokens": cov.tokens,
    });
    eprintln!(
        "shells ({}, q {}{}) per grown layer: {:?}; wins on train {:?}; coverage on held-out {:.4} \
         (per layer {:?}; {} documents as {} framed rows, frame {})",
        f.shell_mode.as_str(),
        f.shell_quantile,
        match f.shell_mode {
            ShellMode::GeneralTarget => format!(", target shift {}", f.shell_target_shift),
            ShellMode::WonQuantile => String::new(),
        },
        shells,
        wins,
        cov.overall,
        cov.per_layer,
        inp.held_docs.len(),
        held_rows,
        held_witness["frame"]
    );
    if let Some(sh) = witness.general_share.as_ref() {
        eprintln!(
            "general share per grown expert (no shell) {:?}; rule {:?}; applied quantile {:?}",
            sh, witness.rule, witness.applied_quantile
        );
    }
    if let Some(s) = &shift {
        eprintln!(
            "routing shift on --general: no shell {:.5} / shell {:.5} (per layer {:?} / {:?})",
            s.overall_noshell, s.overall_shell, s.per_layer_noshell, s.per_layer_shell
        );
    }
    eprintln!("grown bias ({}) per grown layer: {:?}", f.bias_mode.as_str(), grown_bias);
    if let Some(nv) = f.novel {
        let nc = ncov.as_ref().expect("novel coverage follows the witness");
        eprintln!(
            "novelty (q {}): τ {:?}; novel share of the corpus {:?}; novel coverage (grown wins among the novel \
             corpus tokens) shell {:.4} / no shell {:.4} (per layer {:?} / {:?}); source mode {}, desc mode {}",
            nv.quantile,
            nv.tau,
            nv.share_corpus,
            nc.overall_shell,
            nc.overall_noshell,
            nc.per_layer_shell,
            nc.per_layer_noshell,
            f.source_mode.as_str(),
            f.desc_mode.as_str()
        );
    }
    let source_rule = match f.source_mode {
        SourceMode::Hottest => "hottest on the growth corpus (trunk wins, runtime formula)",
        SourceMode::Novel => "hottest on each K-means cluster of the corpus tokens novel for the trunk (min trunk \
                               error > τ, the novel quantile of the general shard's)",
    };
    let novel_fields = |m: &mut serde_json::Map<String, serde_json::Value>| {
        for (k, v) in [
            ("source_mode", serde_json::json!(f.source_mode.as_str())),
            ("source_rule", serde_json::json!(source_rule)),
            ("desc_mode", serde_json::json!(f.desc_mode.as_str())),
            ("novel_quantile", serde_json::json!(f.novel_quantile)),
            ("novel_tau", serde_json::json!(f.novel.map(|n| &n.tau))),
            ("novel_share_corpus", serde_json::json!(f.novel.map(|n| &n.share_corpus))),
            ("novel_tokens", serde_json::json!(f.novel.map(|n| &n.novel_tokens))),
            ("novel_witness", serde_json::json!(f.novel)),
            ("novel_coverage", serde_json::json!(ncov.as_ref().map(|c| c.overall_shell))),
            ("novel_coverage_noshell", serde_json::json!(ncov.as_ref().map(|c| c.overall_noshell))),
            ("novel_coverage_detail", serde_json::json!(&ncov)),
        ] {
            m.insert(k.to_string(), v);
        }
    };
    // ---- outputs ----
    let mut record_summary = serde_json::Value::Null;
    if let Some((out, base, id)) = f.record {
        let g = f.genome.expect("a record needs the bound genome");
        let mut origin = f.origin.clone();
        for (k, v) in [
            ("trigger", serde_json::json!("user_corpus")),
            ("dataset_sha256", serde_json::json!(inp.roles.iter().map(|r| r["sha256"].clone()).collect::<Vec<_>>())),
            ("inputs", serde_json::json!(&inp.roles)),
            ("recipe", serde_json::json!(crate::growth::RECIPE_GROWTH)),
            ("served", f.served.clone()),
            ("layers", serde_json::json!(layers)),
            ("sources", serde_json::json!(f.sources)),
            ("shell_mode", serde_json::json!(f.shell_mode.as_str())),
            ("shell_quantile", serde_json::json!(f.shell_quantile)),
            ("shell_target_shift", serde_json::json!(f.shell_target_shift)),
            ("shells", serde_json::json!(shells)),
            ("wins", serde_json::json!(wins)),
            ("shell_witness", serde_json::json!(&witness)),
            ("bias_mode", serde_json::json!(f.bias_mode.as_str())),
            ("grown_bias", serde_json::json!(&grown_bias)),
            ("routing_shift", serde_json::json!(&shift)),
            ("coverage", serde_json::json!(&cov)),
            ("held_witness", held_witness.clone()),
            ("ckpt_step", serde_json::json!(f.ck0.step)),
            ("genome", serde_json::json!({"id": &g.id, "generation": g.generation, "trunk_hash": &g.trunk_hash})),
        ] {
            origin.insert(k.to_string(), v);
        }
        novel_fields(&mut origin);
        let mut quality = f.quality.clone();
        for (k, v) in [
            ("coverage", serde_json::json!(cov.overall)),
            ("routing_shift_noshell", serde_json::json!(shift.as_ref().map(|s| s.overall_noshell))),
            ("routing_shift_shell", serde_json::json!(shift.as_ref().map(|s| s.overall_shell))),
            ("shell_mode", serde_json::json!(f.shell_mode.as_str())),
            ("shell_target_shift", serde_json::json!(f.shell_target_shift)),
            ("bias_mode", serde_json::json!(f.bias_mode.as_str())),
            ("source_mode", serde_json::json!(f.source_mode.as_str())),
            ("desc_mode", serde_json::json!(f.desc_mode.as_str())),
            ("novel_quantile", serde_json::json!(f.novel_quantile)),
            ("novel_coverage", serde_json::json!(ncov.as_ref().map(|c| c.overall_shell))),
        ] {
            quality.insert(k.to_string(), v);
        }
        record_summary = write_growth_record(&RecordArgs {
            base,
            out,
            id,
            ck0: f.ck0,
            trained,
            e0,
            layers,
            shell_quantile: f.shell_quantile,
            shells,
            bias_mode: f.bias_mode,
            grown_bias: &grown_bias,
            origin: serde_json::Value::Object(origin),
            quality: serde_json::Value::Object(quality),
        })?;
        eprintln!("record '{id}' appended → {}", out.display());
    }
    if let Some(e) = f.export {
        // checked before training; the base / record may have appeared since
        check_export_path(e, f.base, None)?;
        let tj = std::fs::read(f.tokenizer)?;
        crate::export::export(trained, &tj, e)?;
        eprintln!(
            "exported the FULL grown genome → {} (not a genome record: arch.moe.num_experts = {})",
            e.display(),
            trained.cfg.experts
        );
    }
    let mut summary = f.summary;
    for (k, v) in [
        ("layers", serde_json::json!(layers)),
        ("sources", serde_json::json!(f.sources)),
        ("served", f.served.clone()),
        ("shell_mode", serde_json::json!(f.shell_mode.as_str())),
        ("shell_quantile", serde_json::json!(f.shell_quantile)),
        ("shell_target_shift", serde_json::json!(f.shell_target_shift)),
        ("shells", serde_json::json!(shells)),
        ("wins", serde_json::json!(wins)),
        ("shell_witness", serde_json::json!(&witness)),
        ("bias_mode", serde_json::json!(f.bias_mode.as_str())),
        ("grown_bias", serde_json::json!(&grown_bias)),
        ("routing_shift_noshell", serde_json::json!(shift.as_ref().map(|s| s.overall_noshell))),
        ("routing_shift_shell", serde_json::json!(shift.as_ref().map(|s| s.overall_shell))),
        ("routing_shift", serde_json::json!(&shift)),
        ("coverage", serde_json::json!(cov.overall)),
        ("coverage_detail", serde_json::json!(&cov)),
        ("held_witness", held_witness),
        ("record", record_summary),
        ("out_ckpt", serde_json::json!(f.out_ckpt.map(|p| p.display().to_string()))),
        ("export", serde_json::json!(f.export.map(|p| p.display().to_string()))),
    ] {
        summary.insert(k.to_string(), v);
    }
    novel_fields(&mut summary);
    Ok(serde_json::Value::Object(summary))
}

/// The growth, end to end: refuse early (K, `E0 + K ≤ 8`, quantile,
/// shell mode (general-target needs `--general`), layers, `--record-out`
/// ≠ `--base`, `--export` new and ≠ `--base`, genome / E0 / checkpoint
/// binding of `--base`, no earlier growth record in the grown layers),
/// tokenize the corpus (+ `--held`), take the trunk the runtime executes
/// (f16 genome: rounded arena), choose the K sources and descriptors per
/// grown layer (`--source-mode hottest`: the trunk experts hottest on the
/// growth corpus and the tokens they win; `novel`: the K-means clusters
/// of the corpus tokens novel for the trunk, τ from `--general`, and the
/// expert hottest on each), grow, train only the new experts (dropless,
/// the bias pinned at 0 or at the source's — `--bias-mode`; descriptors
/// adapting or frozen — `--desc-mode`), gate on the held-out loss against
/// the GENOME, then [`finish_growth`]: shells / coverage / routing shift /
/// novel coverage with EXACTLY the runtime formula, save / record /
/// export, and the one JSON summary line.
pub fn grow(a: &GrowCli) -> anyhow::Result<serde_json::Value> {
    use crate::growth::{
        GrowSpec, GrowTrain, GrowthDesc, MAX_RUNTIME_EXPERTS, NovelWitness, check_base_growth_records,
        check_layers, cluster_inits, grow_experts_from, novel_inits, sources_by_corpus_wins,
        train_grown_experts, trunk_corpus_wins,
    };
    use crate::growth::GrowArgs;
    anyhow::ensure!(a.experts >= 1, "--experts must be ≥ 1");
    check_shell_args(a.shell_mode, a.shell_quantile, a.shell_target_shift, a.general.is_some())?;
    check_source_args(a.source_mode, a.novel_quantile, a.general.is_some())?;
    let desc_mode = a.desc_mode.unwrap_or_else(|| DescMode::default_for(a.source_mode));
    anyhow::ensure!(a.batch >= 1 && a.seq >= 2, "--batch ≥ 1 and --seq ≥ 2");
    let record = match (&a.record_out, &a.base, &a.id) {
        (None, _, _) => None,
        (Some(out), Some(base), Some(id)) => Some((out.clone(), base.clone(), id.clone())),
        (Some(_), None, _) => anyhow::bail!("--record-out needs --base (the genome file F0)"),
        (Some(_), _, None) => anyhow::bail!("--record-out needs --id"),
    };
    if let Some((out, base, _)) = &record {
        crate::skill::check_out_path(base, out)?;
    }
    if let Some(e) = &a.export {
        check_export_path(e, a.base.as_deref(), a.record_out.as_deref())?;
    }
    let ck = load_checkpoint(&a.ckpt)?;
    let e0 = ck.cfg.experts;
    anyhow::ensure!(e0 >= 1, "the checkpoint has no routed experts");
    anyhow::ensure!(
        e0 + a.experts <= MAX_RUNTIME_EXPERTS,
        "E0 {e0} + K {} > {MAX_RUNTIME_EXPERTS}: the runtime graph handles at most \
         {MAX_RUNTIME_EXPERTS} experts per layer (above it the file runs per-op, many times slower)",
        a.experts
    );
    let layers = match &a.layers {
        Some(l) => check_layers(ck.cfg.layers, l)?,
        None => (0..ck.cfg.layers).collect(),
    };
    // bind to the genome file BEFORE training: E0, resonance MoE, trunk
    // bytes, no earlier record in the grown layers; and take the trunk the
    // runtime executes (an f16 genome rounds the arena)
    let (ck_served, genome_info, served) = if let Some((_, base, _)) = &record {
        let base_m = cortiq_core::format::CmfModel::open(base)?;
        let g = base_m.header.genome.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "--base {} carries no GENOME: export it with `export --genome-id … --genome-status …`",
                base.display()
            )
        })?;
        let e0_file = cortiq_core::knowledge::genome_moe_experts(&base_m.header)?;
        anyhow::ensure!(
            e0_file == e0,
            "--base genome '{}' has E0 = {e0_file}, the checkpoint {e0} experts",
            g.id
        );
        let moe = cortiq_core::knowledge::moe_layers(&base_m.tensors);
        for &l in &layers {
            anyhow::ensure!(moe.contains(&l), "layer {l} is not an MoE layer of --base (MoE layers {moe:?})");
        }
        check_base_growth_records(&base_m.header, &layers)?;
        let b = crate::skill::bind_ckpt_to_base(&ck, &base_m)?;
        let (ck_s, enc, changed) = served_checkpoint(&ck, &base_m)?;
        eprintln!(
            "grow: --ckpt is the checkpoint of genome '{}' ({} trunk tensors byte-identical); \
             served encoding {enc}: {changed} arena values rounded",
            g.id, b.trunk_tensors
        );
        (ck_s, Some(g), serde_json::json!({"encoding": enc, "rounded_values": changed}))
    } else {
        (
            Checkpoint {
                cfg: ck.cfg.clone(),
                step: ck.step,
                params: ck.params.clone(),
                m: None,
                v: None,
                extras: ck.extras.clone(),
            },
            None,
            serde_json::json!({"encoding": "f32", "rounded_values": 0}),
        )
    };
    let inputs = load_growth_inputs(
        &a.tokenizer,
        &a.corpus,
        &a.held,
        a.general.as_deref(),
        a.seq,
        a.trace_tokens,
        a.trace_docs,
        "grow",
    )?;
    let spec = GrowSpec {
        experts: a.experts,
        layers: layers.clone(),
        noise: a.noise,
        shift: a.shift,
        seed: a.seed,
        // the copy's bias: 0, or the source's (inherited by the surgery,
        // frozen by the training)
        zero_bias: a.bias_mode == BiasMode::Zero,
    };
    // ---- sources and descriptors of the new experts on the pre-growth
    // instance: `hottest` = the trunk experts hottest ON THE GROWTH CORPUS
    // and the tokens they win; `novel` = the K-means clusters of the corpus
    // tokens novel for the trunk (τ from --general) and the expert hottest
    // on each; the novelty pass runs (and is reported) whenever --general
    // is given ----
    let (sources_grown, trunk_wins, inits, novel) = {
        let gpu0 = EmbryoGpu::new_eval_dropless(ck_served.cfg.clone(), a.batch, a.seq, &ck_served.params)
            .ok_or_else(|| anyhow::anyhow!(NO_DEVICE))?;
        gpu0.set_desc(&ck_served.extras);
        gpu0.desc_updates.set(false);
        let desc0 = GrowthDesc::from_checkpoint(&ck_served, e0)?;
        let wins = trunk_corpus_wins(&gpu0, &inputs.train_trace, &layers, &desc0)?;
        let nv = novelty_pass(&gpu0, &inputs, &layers, &desc0, a.novel_quantile, a.seed)?;
        match a.source_mode {
            SourceMode::Hottest => {
                let sources = sources_by_corpus_wins(&wins, &desc0.bias, e0, &layers, a.experts)?;
                let inits = cluster_inits(&gpu0, &inputs.train_trace, &layers, &desc0, &sources, a.seed)?;
                let novel = nv.map(|(sets, n_gen)| NovelWitness::from_sets(a.novel_quantile, n_gen, &sets, None));
                (sources, wins, inits, novel)
            }
            SourceMode::Novel => {
                let (sets, n_gen) = nv.expect("checked: --source-mode novel needs --general");
                let (sources, inits, cluster_rows) = novel_inits(&sets, a.experts, e0, a.seed)?;
                let novel = NovelWitness::from_sets(a.novel_quantile, n_gen, &sets, Some(cluster_rows));
                (sources, wins, inits, Some(novel))
            }
        }
    };
    let init_rows: Vec<Vec<usize>> = inits.iter().map(|v| v.iter().map(|c| c.rows).collect()).collect();
    let (grown, sources) = grow_experts_from(&ck_served, &spec, &sources_grown, Some(&inits))?;
    debug_assert_eq!(
        layers.iter().map(|&l| sources[l].clone()).collect::<Vec<_>>(),
        sources_grown
    );
    eprintln!(
        "grown: experts {} → {} in layers {:?} (of {}); trunk wins on the corpus per grown layer {:?}; \
         sources ({}) {:?}; descriptor init tokens {:?}; bias mode {}; desc mode {}",
        e0,
        grown.cfg.experts,
        layers,
        grown.cfg.layers,
        trunk_wins,
        match a.source_mode {
            SourceMode::Hottest => "hottest on the corpus",
            SourceMode::Novel => "hottest on each novel cluster",
        },
        sources_grown,
        init_rows,
        a.bias_mode.as_str(),
        desc_mode.as_str()
    );
    let ga = GrowArgs {
        steps: a.steps,
        lr: a.lr,
        batch: a.batch,
        seq: a.seq,
        eval_every: 30,
        seed: a.seed,
        held_batches: a.held_batches,
    };
    let gt = GrowTrain {
        e0,
        layers: &layers,
        train: &inputs.train,
        held: &inputs.held,
        freeze_bias: true,
        freeze_desc: desc_mode == DescMode::Frozen,
        genome: Some(&ck_served),
    };
    let (trained, ho) = train_grown_experts(&grown, &gt, &ga, &|| false)?;
    let l_genome = ho.genome.expect("genome given");
    let imp = ho.improvement();
    eprintln!(
        "held-out ({} windows): genome {l_genome:.4} (ppl {:.1}), grown untrained {:.4}, after {:.4} \
         (ppl {:.1}); improvement over the genome {imp:.4} vs gate {}",
        ho.windows,
        l_genome.exp(),
        ho.untrained,
        ho.after,
        ho.after.exp(),
        a.gate
    );
    anyhow::ensure!(
        imp >= a.gate,
        "{GROW_REJECTED}: held-out improvement over the genome {imp:.4} < gate {} — nothing written",
        a.gate
    );
    let held_out = serde_json::json!({
        "genome": l_genome, "grown_untrained": ho.untrained, "after": ho.after,
        "before": l_genome,
        "improvement": imp, "gate": a.gate, "source": &inputs.held_source,
        "tokens": inputs.held.tokens.len(), "windows": ho.windows, "batches": ho.batches,
        "instance": "dropless",
    });
    let origin = serde_json::json!({
        "steps": a.steps, "lr": a.lr, "batch": a.batch, "seq": a.seq, "seed": a.seed,
        "noise": a.noise, "shift": a.shift,
        "experts": a.experts,
        "trunk_corpus_wins": &trunk_wins, "trace_tokens": a.trace_tokens, "trace_docs": a.trace_docs,
        "descriptor_init_tokens": &init_rows,
        "held_out": &held_out,
    });
    let quality = serde_json::json!({
        "set": "growth held-out (raw LM loss, dropless, vs the genome)",
        "held_out": &held_out,
    });
    let summary = serde_json::json!({
        "id": a.id,
        "K": a.experts,
        "trunk_corpus_wins": &trunk_wins, "trace_tokens": a.trace_tokens, "trace_docs": a.trace_docs,
        "descriptor_init_tokens": &init_rows,
        "held_genome": l_genome,
        "held_untrained": ho.untrained,
        "held_before": l_genome,
        "held_after": ho.after,
        "held_windows": ho.windows,
        "improvement": imp,
    });
    let as_map = |v: serde_json::Value| match v {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("object literal"),
    };
    finish_growth(GrowFinish {
        trained: &trained,
        ck0: &ck,
        e0,
        layers: &layers,
        sources: &sources_grown,
        inputs: &inputs,
        shell_mode: a.shell_mode,
        shell_quantile: a.shell_quantile,
        shell_target_shift: a.shell_target_shift,
        bias_mode: a.bias_mode,
        source_mode: a.source_mode,
        desc_mode,
        novel_quantile: a.novel_quantile,
        novel: novel.as_ref(),
        batch: a.batch,
        seq: a.seq,
        record: record.as_ref().map(|(o, b, i)| (o.as_path(), b.as_path(), i.as_str())),
        genome: genome_info.as_ref(),
        out_ckpt: a.out_ckpt.as_deref(),
        export: a.export.as_deref(),
        base: a.base.as_deref(),
        tokenizer: &a.tokenizer,
        served,
        origin: as_map(origin),
        quality: as_map(quality),
        summary: as_map(summary),
    })
}

/// `cortiq-embryo reshell …`: the post-training tail of [`grow`] redone
/// from a saved TRAINED grown checkpoint (`grow --out-ckpt`).
pub struct ReshellCli {
    /// The trained grown checkpoint (`E0 + K` experts in every layer).
    pub ckpt: PathBuf,
    /// The PRE-growth checkpoint of `--base` (F0's): required for an f16
    /// genome, whose master the grown checkpoint (served, rounded trunk)
    /// cannot reproduce; an f32 genome recovers it from `--ckpt`.
    pub genome_ckpt: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub corpus: Vec<PathBuf>,
    pub held: Vec<PathBuf>,
    pub general: Option<PathBuf>,
    pub trace_tokens: usize,
    pub trace_docs: usize,
    /// The grown layers; default = the layers whose grown slots are live
    /// in `--ckpt` (must equal them when given).
    pub layers: Option<Vec<usize>>,
    pub shell_mode: ShellMode,
    pub shell_quantile: f32,
    pub shell_target_shift: f32,
    /// None = inferred from `--ckpt` (all grown biases 0 → zero, else
    /// source — verified against the source biases either way).
    pub bias_mode: Option<BiasMode>,
    /// The growth's `--source-mode` (the sources are recomputed by it).
    pub source_mode: SourceMode,
    pub novel_quantile: f32,
    /// None = inferred from `--ckpt`: the grown μ / U equal to the
    /// recomputed initialisation bit for bit → frozen, else adapt
    /// (`frozen` given but differing → refused).
    pub desc_mode: Option<DescMode>,
    /// The growth's `--seed` (the novel clustering and the descriptor
    /// inits are seeded by it).
    pub seed: u64,
    pub record_out: Option<PathBuf>,
    /// The genome file F0 (E0, trunk binding, the record's base).
    pub base: PathBuf,
    pub id: Option<String>,
    pub batch: usize,
    pub seq: usize,
}

/// ONLY the post-training part of [`grow`] from a trained grown
/// checkpoint: bind `--base` (genome, E0 from the genome block, the grown
/// layers from the live slots of the checkpoint, the pre-growth checkpoint
/// = `--genome-ckpt` or the E0 shrink of `--ckpt` on an f32 genome; the
/// trunk of `--ckpt` must be the served trunk of that checkpoint), the
/// same tokenized inputs and trace budgets, the trunk sources recomputed
/// on the pre-growth instance (the same corpus wins), the bias mode
/// inferred / verified, then [`finish_growth`]: shells by `--shell-mode`,
/// coverage, routing shift, the record and the one JSON summary line
/// (`"reshell": true`).
pub fn reshell(a: &ReshellCli) -> anyhow::Result<serde_json::Value> {
    use crate::growth::{
        GrowthDesc, MAX_RUNTIME_EXPERTS, NovelWitness, check_base_growth_records, check_layers, cluster_inits,
        expected_grown_bias, grown_bias_of, grown_layers_of, novel_inits, shrink_experts, sources_by_corpus_wins,
        trunk_corpus_wins,
    };
    check_shell_args(a.shell_mode, a.shell_quantile, a.shell_target_shift, a.general.is_some())?;
    check_source_args(a.source_mode, a.novel_quantile, a.general.is_some())?;
    anyhow::ensure!(a.batch >= 1 && a.seq >= 2, "--batch ≥ 1 and --seq ≥ 2");
    let record = match (&a.record_out, &a.id) {
        (None, _) => None,
        (Some(out), Some(id)) => Some((out.clone(), a.base.clone(), id.clone())),
        (Some(_), None) => anyhow::bail!("--record-out needs --id"),
    };
    if let Some((out, base, _)) = &record {
        crate::skill::check_out_path(base, out)?;
    }
    let ck1 = load_checkpoint(&a.ckpt)?;
    let base_m = cortiq_core::format::CmfModel::open(&a.base)?;
    let g = base_m.header.genome.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "--base {} carries no GENOME: export it with `export --genome-id … --genome-status …`",
            a.base.display()
        )
    })?;
    let e0 = cortiq_core::knowledge::genome_moe_experts(&base_m.header)?;
    let e1 = ck1.cfg.experts;
    anyhow::ensure!(
        e1 > e0,
        "--ckpt has {e1} experts, --base genome '{}' E0 = {e0}: not a grown checkpoint of this genome",
        g.id
    );
    anyhow::ensure!(
        e1 <= MAX_RUNTIME_EXPERTS,
        "E0 {e0} + K {} > {MAX_RUNTIME_EXPERTS}: the runtime graph handles at most \
         {MAX_RUNTIME_EXPERTS} experts per layer",
        e1 - e0
    );
    let kn = e1 - e0;
    let layers_ck = grown_layers_of(&ck1, e0)?;
    let layers = match &a.layers {
        Some(l) => {
            let l = check_layers(ck1.cfg.layers, l)?;
            anyhow::ensure!(
                l == layers_ck,
                "--layers {l:?} but the checkpoint's live grown slots are in layers {layers_ck:?}"
            );
            l
        }
        None => layers_ck,
    };
    let moe = cortiq_core::knowledge::moe_layers(&base_m.tensors);
    for &l in &layers {
        anyhow::ensure!(moe.contains(&l), "layer {l} is not an MoE layer of --base (MoE layers {moe:?})");
    }
    check_base_growth_records(&base_m.header, &layers)?;
    // ---- the pre-growth checkpoint the record binds with ----
    let ck0 = match &a.genome_ckpt {
        Some(p) => {
            let c = load_checkpoint(p)?;
            anyhow::ensure!(
                c.cfg.experts == e0,
                "--genome-ckpt has {} experts, --base genome '{}' E0 = {e0}",
                c.cfg.experts,
                g.id
            );
            c
        }
        None => {
            anyhow::ensure!(
                g.encoding == "f32",
                "--base genome '{}' is encoded as {}: its master trunk cannot be recovered from the \
                 grown checkpoint (served, rounded trunk) — pass --genome-ckpt <the checkpoint of F0>",
                g.id,
                g.encoding
            );
            shrink_experts(&ck1, e0)?
        }
    };
    let b = crate::skill::bind_ckpt_to_base(&ck0, &base_m)?;
    let (ck0_served, enc, changed) = served_checkpoint(&ck0, &base_m)?;
    // the grown checkpoint must carry exactly that served trunk
    let shrunk = shrink_experts(&ck1, e0)?;
    anyhow::ensure!(
        serde_json::to_value(&shrunk.cfg).ok() == serde_json::to_value(&ck0_served.cfg).ok()
            && shrunk.params.len() == ck0_served.params.len()
            && shrunk.params.iter().zip(&ck0_served.params).all(|(x, y)| x.to_bits() == y.to_bits()),
        "--ckpt is not a grown checkpoint of --base genome '{}': its trunk differs from the served \
         trunk of the genome checkpoint",
        g.id
    );
    for (n, x) in &shrunk.extras {
        if let Some((_, y)) = ck0_served.extras.iter().find(|(m, _)| m == n) {
            anyhow::ensure!(
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "--ckpt: the trunk descriptors ({n}) differ from the genome checkpoint's"
            );
        }
    }
    drop(shrunk);
    eprintln!(
        "reshell: --ckpt is a grown checkpoint ({e0} → {e1} experts, live in layers {layers:?}) of genome '{}' \
         ({} trunk tensors byte-identical); served encoding {enc}: {changed} arena values rounded",
        g.id, b.trunk_tensors
    );
    let served = serde_json::json!({"encoding": enc, "rounded_values": changed});
    let inputs = load_growth_inputs(
        &a.tokenizer,
        &a.corpus,
        &a.held,
        a.general.as_deref(),
        a.seq,
        a.trace_tokens,
        a.trace_docs,
        "reshell",
    )?;
    // ---- the trunk sources and descriptor inits: the same passes as the
    // growth's on the pre-growth instance (the same corpus wins / novelty,
    // the same seed) ----
    let (sources, trunk_wins, inits, novel) = {
        let gpu0 = EmbryoGpu::new_eval_dropless(ck0_served.cfg.clone(), a.batch, a.seq, &ck0_served.params)
            .ok_or_else(|| anyhow::anyhow!(NO_DEVICE))?;
        gpu0.set_desc(&ck0_served.extras);
        gpu0.desc_updates.set(false);
        let desc0 = GrowthDesc::from_checkpoint(&ck0_served, e0)?;
        let wins = trunk_corpus_wins(&gpu0, &inputs.train_trace, &layers, &desc0)?;
        let nv = novelty_pass(&gpu0, &inputs, &layers, &desc0, a.novel_quantile, a.seed)?;
        match a.source_mode {
            SourceMode::Hottest => {
                let sources = sources_by_corpus_wins(&wins, &desc0.bias, e0, &layers, kn)?;
                let inits = cluster_inits(&gpu0, &inputs.train_trace, &layers, &desc0, &sources, a.seed)?;
                let novel = nv.map(|(sets, n_gen)| NovelWitness::from_sets(a.novel_quantile, n_gen, &sets, None));
                (sources, wins, inits, novel)
            }
            SourceMode::Novel => {
                let (sets, n_gen) = nv.expect("checked: --source-mode novel needs --general");
                let (sources, inits, cluster_rows) = novel_inits(&sets, kn, e0, a.seed)?;
                let novel = NovelWitness::from_sets(a.novel_quantile, n_gen, &sets, Some(cluster_rows));
                (sources, wins, inits, Some(novel))
            }
        }
    };
    // ---- the desc mode: given (frozen verified) or inferred from the
    // grown descriptors against the recomputed initialisation ----
    let desc_equal = grown_desc_equals_inits(&ck1, e0, &layers, &inits);
    drop(inits);
    let desc_mode = match a.desc_mode {
        Some(DescMode::Frozen) => {
            anyhow::ensure!(
                desc_equal,
                "--desc-mode frozen, but the grown μ / U of --ckpt differ from the descriptor initialisation \
                 recomputed with --source-mode {} --seed {} (a frozen growth keeps them bit for bit)",
                a.source_mode.as_str(),
                a.seed
            );
            DescMode::Frozen
        }
        Some(DescMode::Adapt) => DescMode::Adapt,
        None => {
            if desc_equal {
                DescMode::Frozen
            } else {
                DescMode::Adapt
            }
        }
    };
    // ---- the bias mode: given (verified in finish_growth) or inferred ----
    let bias_mode = match a.bias_mode {
        Some(m) => m,
        None => {
            let have = grown_bias_of(&ck1, e0, &layers)?;
            if have.iter().all(|v| v.iter().all(|b| *b == 0.0)) {
                BiasMode::Zero
            } else {
                let want = expected_grown_bias(&ck0, &layers, &sources, BiasMode::Source)?;
                anyhow::ensure!(
                    have == want,
                    "the grown biases of --ckpt {have:?} are neither 0 (--bias-mode zero) nor their sources' \
                     {want:?} (--bias-mode source; sources {sources:?})"
                );
                BiasMode::Source
            }
        }
    };
    eprintln!(
        "reshell: trunk wins on the corpus per grown layer {trunk_wins:?}; sources ({}) {sources:?}; bias mode {} \
         ({}); desc mode {} ({})",
        a.source_mode.as_str(),
        bias_mode.as_str(),
        if a.bias_mode.is_some() { "given" } else { "inferred from the checkpoint" },
        desc_mode.as_str(),
        if a.desc_mode.is_some() { "given" } else { "inferred from the checkpoint" }
    );
    let origin = serde_json::json!({
        "reshell": true,
        "ckpt": a.ckpt.display().to_string(),
        "genome_ckpt": a.genome_ckpt.as_ref().map(|p| p.display().to_string()),
        "batch": a.batch, "seq": a.seq, "seed": a.seed,
        "experts": kn,
        "trunk_corpus_wins": &trunk_wins, "trace_tokens": a.trace_tokens, "trace_docs": a.trace_docs,
        "desc_mode_inferred": a.desc_mode.is_none(),
    });
    let quality = serde_json::json!({
        "set": "reshell: shells / coverage / routing shift of a trained grown checkpoint (no held-out loss)",
    });
    let summary = serde_json::json!({
        "reshell": true,
        "id": a.id,
        "K": kn,
        "ckpt": a.ckpt.display().to_string(),
        "trunk_corpus_wins": &trunk_wins, "trace_tokens": a.trace_tokens, "trace_docs": a.trace_docs,
    });
    let as_map = |v: serde_json::Value| match v {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("object literal"),
    };
    finish_growth(GrowFinish {
        trained: &ck1,
        ck0: &ck0,
        e0,
        layers: &layers,
        sources: &sources,
        inputs: &inputs,
        shell_mode: a.shell_mode,
        shell_quantile: a.shell_quantile,
        shell_target_shift: a.shell_target_shift,
        bias_mode,
        source_mode: a.source_mode,
        desc_mode,
        novel_quantile: a.novel_quantile,
        novel: novel.as_ref(),
        batch: a.batch,
        seq: a.seq,
        record: record.as_ref().map(|(o, b, i)| (o.as_path(), b.as_path(), i.as_str())),
        genome: Some(&g),
        out_ckpt: None,
        export: None,
        base: Some(&a.base),
        tokenizer: &a.tokenizer,
        served,
        origin: as_map(origin),
        quality: as_map(quality),
        summary: as_map(summary),
    })
}

/// Evenly spaced `[t]` windows of a shard within a token budget (0 = the whole shard).
fn subsample_windows(s: &crate::train::Shard, t: usize, max_tokens: usize) -> crate::train::Shard {
    let n_w = s.tokens.len() / t.max(1);
    let keep = max_tokens / t.max(1);
    if max_tokens == 0 || keep == 0 || n_w <= keep {
        return crate::train::Shard { tokens: s.tokens.clone() };
    }
    let mut tokens = Vec::with_capacity(keep * t);
    for i in 0..keep {
        let w = i * n_w / keep;
        tokens.extend_from_slice(&s.tokens[w * t..(w + 1) * t]);
    }
    crate::train::Shard { tokens }
}

/// Evenly spaced documents within a budget (0 = all).
fn subsample_docs(docs: &[Vec<u16>], max_docs: usize) -> Vec<Vec<u16>> {
    if max_docs == 0 || docs.len() <= max_docs {
        return docs.to_vec();
    }
    (0..max_docs).map(|i| docs[i * docs.len() / max_docs].clone()).collect()
}
