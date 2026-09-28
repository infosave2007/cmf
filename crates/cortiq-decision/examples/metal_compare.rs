//! Same-process CPU/GPU parity and wall-time comparison; no oracle/network.
//! MODEL DATA.jsonl SKILL [LIMIT] [metal|vulkan] [paired|separate]
use anyhow::{Result, ensure};
use cortiq_decision::{
    bert::{Encoder, EncoderDevice, GOLDEN_MAX_ABS},
    container::{DecisionModel, Verify},
    eval::SkillScorer,
    resonance::Decision,
    signal::{Features, SignalEncoder},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::BufRead, path::Path, time::Instant};

#[derive(Deserialize)]
struct Row {
    text: String,
    #[serde(default)]
    label: Option<String>,
}
struct Measured {
    features: Features,
    errors: Vec<f32>,
    decision: Decision,
    total: f64,
    encode: f64,
    resonance: f64,
    gpu: f64,
    tokens: usize,
}
fn run(enc: &SignalEncoder, scorer: &SkillScorer, text: &str) -> Result<Measured> {
    let start = Instant::now();
    let mut scored = enc.score_timed(text, &[scorer.packed()])?;
    let t = scored.timings;
    let tr = Instant::now();
    let errors = scored.errors.remove(0);
    let decision = scorer.decide_errors(&errors)?;
    let end = Instant::now();
    Ok(Measured {
        features: scored.features,
        errors,
        decision,
        total: (end - start).as_secs_f64() * 1000.0,
        encode: t.encode.as_secs_f64() * 1000.0,
        resonance: (scored.resonance + (end - tr)).as_secs_f64() * 1000.0,
        gpu: t.gpu.as_secs_f64() * 1000.0,
        tokens: t.tokens,
    })
}
fn delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn percentiles(v: &mut [f64]) -> Value {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    json!({"p50":v[n/2],"p95":v[((n as f64*0.95).ceil() as usize-1).min(n-1)],"p99":v[((n as f64*0.99).ceil() as usize-1).min(n-1)]})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 4,
        "MODEL DATA.jsonl SKILL [LIMIT] [metal|vulkan] [paired|separate]"
    );
    let limit = args
        .get(4)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(usize::MAX);
    let backend = args.get(5).map(String::as_str).unwrap_or("metal");
    let device = match backend {
        "metal" => EncoderDevice::Metal,
        "vulkan" => EncoderDevice::Vulkan,
        _ => anyhow::bail!("benchmark device must be metal or vulkan"),
    };
    let mode = args.get(6).map(String::as_str).unwrap_or("paired");
    ensure!(
        matches!(mode, "paired" | "separate"),
        "mode must be paired or separate"
    );
    let dataset = std::fs::read(&args[2])?;
    let rows: Vec<Row> = dataset
        .lines()
        .take(limit)
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()?;
    ensure!(!rows.is_empty(), "empty dataset");
    let model = DecisionModel::open(Path::new(&args[1]), Verify::Full)?;
    let cpu = Encoder::from_model(&model)?.with_device(EncoderDevice::Cpu)?;
    let gpu = cpu.clone().with_device(device)?;
    let golden_cpu = cpu.verify_golden(&model)?;
    let golden_gpu = gpu.verify_golden(&model)?;
    let cpu = SignalEncoder::new(cpu);
    let gpu = SignalEncoder::new(gpu);
    let scorer = SkillScorer::from_model(&model, &args[3])?.with_device(EncoderDevice::Cpu)?;
    let gpu_scorer = SkillScorer::from_model(&model, &args[3])?.with_device(device)?;
    for row in rows.iter().cycle().take(50) {
        run(&cpu, &scorer, &row.text)?;
        run(&gpu, &gpu_scorer, &row.text)?;
    }
    // Separate phases avoid inserting a slow CPU request between GPU requests.
    // Keep the same per-row parity checks; the power/load conditions differ
    // from the default alternating test, so report the mode explicitly.
    let mut reference = std::collections::VecDeque::new();
    if mode == "separate" {
        for row in &rows {
            reference.push_back(run(&cpu, &scorer, &row.text)?);
        }
        for row in rows.iter().cycle().take(50) {
            run(&gpu, &gpu_scorer, &row.text)?;
        }
    }
    let gpu_before = gpu.encoder().gpu_submissions();
    let resonance_before = gpu_scorer.packed().gpu_submissions();
    let mut times: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::new());
    let (mut max_phi, mut max_error, mut max_error_scaled) = (0.0f32, 0.0f32, 0.0f32);
    let (mut winner_changes, mut gate_changes, mut hash_changes) = (0, 0, 0);
    let (mut correct_cpu, mut correct_gpu, mut accepted_cpu, mut accepted_gpu) = (0, 0, 0, 0);
    let mut differences = Vec::new();
    let mut buckets: [(Vec<f64>, Vec<f64>); 5] = std::array::from_fn(|_| (Vec::new(), Vec::new()));
    for (i, row) in rows.iter().enumerate() {
        // Alternate order to avoid always timing one backend after the other.
        let (c, g) = if let Some(c) = reference.pop_front() {
            (c, run(&gpu, &gpu_scorer, &row.text)?)
        } else if i % 2 == 0 {
            (
                run(&cpu, &scorer, &row.text)?,
                run(&gpu, &gpu_scorer, &row.text)?,
            )
        } else {
            let g = run(&gpu, &gpu_scorer, &row.text)?;
            (run(&cpu, &scorer, &row.text)?, g)
        };
        max_phi = max_phi.max(delta(&c.features.phi_p, &g.features.phi_p));
        ensure!(c.tokens == g.tokens, "token count changed");
        let bucket = match c.tokens {
            0..=8 => 0,
            9..=16 => 1,
            17..=32 => 2,
            33..=64 => 3,
            _ => 4,
        };
        buckets[bucket].0.push(c.total);
        buckets[bucket].1.push(g.total);
        max_error = max_error.max(delta(&c.errors, &g.errors));
        for (c, g) in c.errors.iter().zip(&g.errors) {
            max_error_scaled = max_error_scaled.max((c - g).abs() / (1.0 + c.abs()));
        }
        hash_changes += usize::from(c.features.phi_h != g.features.phi_h);
        let ca = scorer.accepted(&c.decision);
        let ga = scorer.accepted(&g.decision);
        winner_changes += usize::from(c.decision.winner != g.decision.winner);
        gate_changes += usize::from(ca != ga);
        accepted_cpu += usize::from(ca);
        accepted_gpu += usize::from(ga);
        if c.decision.winner != g.decision.winner || ca != ga {
            differences.push(i);
        }
        for (slot, value) in times.iter_mut().zip([
            c.total,
            g.total,
            c.encode,
            g.encode,
            c.resonance,
            g.resonance,
            g.gpu,
        ]) {
            slot.push(value);
        }
        if let Some(label) = &row.label {
            correct_cpu += usize::from(
                c.decision
                    .winner
                    .is_some_and(|w| &scorer.labels()[w] == label),
            );
            correct_gpu += usize::from(
                g.decision
                    .winner
                    .is_some_and(|w| &scorer.labels()[w] == label),
            );
        }
    }
    let gpu_submissions = gpu.encoder().gpu_submissions() - gpu_before;
    let measures: Vec<_> = times.iter_mut().map(|v| percentiles(v)).collect();
    let by_tokens: Vec<_> = buckets.iter_mut().zip(["1-8", "9-16", "17-32", "33-64", "65+"])
        .filter(|((c, _), _)| !c.is_empty())
        .map(|((c, g), range)| json!({"tokens":range,"n":c.len(),"cpu_ms":percentiles(c),"gpu_ms":percentiles(g)})).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cortiq-decision-metal-comparison/2","model_sha":model.model_sha(),"skill":args[3],"n":rows.len(),
            "dataset_sha256":format!("{:x}",Sha256::digest(&dataset)),"device":gpu.encoder().device_name(),
            "warmup":50,"mode":mode,
            "order": if mode == "paired" { "paired serial; CPU-first/GPU-first alternates per row; same process, model, scorer and texts; no oracle" }
                else { "separate serial phases: all CPU references, 50 additional GPU warmups, all GPU rows. Same process/model/texts and parity checks; no inserted CPU inference between GPU requests; no oracle." },
            "golden_max_abs":{"cpu":golden_cpu.max_abs,"metal":golden_gpu.max_abs},
            "max_embedding_abs":max_phi,"max_reconstruction_error_abs":max_error,"max_reconstruction_error_scaled":max_error_scaled,
            "winner_changes":winner_changes,"gate_changes":gate_changes,"hash_changes":hash_changes,"different_rows":differences,
            "correct":{"cpu":correct_cpu,"metal":correct_gpu},"accepted":{"cpu":accepted_cpu,"metal":accepted_gpu},
            "metal_command_buffers":gpu_submissions,"cpu_command_buffers":cpu.encoder().gpu_submissions(),
        "metal_resonance_dispatches":gpu_scorer.packed().gpu_submissions()-resonance_before,
            "ms":{"cpu_total":measures[0],"metal_total":measures[1],"cpu_encoder":measures[2],"metal_split_encoder":measures[3],
                "cpu_resonance":measures[4],"metal_ranking_cpu":measures[5],"metal_joint_pipeline":measures[6]},
            "by_tokens":by_tokens,
            "gpu_scope":"One command buffer: embedding lookup, transformer layers, row-order pooling and FP32 L2, reconstruction errors; tokenization, hashing and ranking/gates on CPU. Joint GPU time is not split into fictitious encoder/resonance wall times."
        }))?.replace("metal", backend).replace("Metal", if backend == "vulkan" { "Vulkan" } else { "Metal" })
    );
    ensure!(
        gpu_submissions == rows.len() as u64,
        "not all rows executed on the selected GPU"
    );
    ensure!(
        gpu_scorer.packed().gpu_submissions() - resonance_before == rows.len() as u64,
        "not all reconstruction rows executed on the selected GPU"
    );
    ensure!(
        winner_changes == 0
            && gate_changes == 0
            && hash_changes == 0
            && max_phi <= GOLDEN_MAX_ABS
            && max_error_scaled <= 2e-5,
        "CPU/GPU parity failed"
    );
    Ok(())
}
