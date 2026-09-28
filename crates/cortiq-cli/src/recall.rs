//! `cortiq probe-recall` — synthetic associative recall (MQAR / NIAH-lite)
//! through the RUNTIME, not a training-side probe: the prompt is fed to
//! `Pipeline::forward_ids`, so it runs on whatever the environment selects
//! (`CMF_GPU=0` → CPU; `CMF_GPU=wgpu CMF_GPU_PROBE=0 CMF_GPU_WGPU_GRAPH=1
//! CMF_EMBRYO_RESIDENT=parallel` → the resident Embryo graph, chunked
//! prefill included).
//!
//! One trial: `[BOS] k₁ v₁ k₂ v₂ … k_K v_K  <D filler tokens>  k_q` and the
//! model must put `v_q` first.  Keys, values and filler are random token
//! ids from the middle of the vocabulary (no special ids), keys and values
//! disjoint, filler never equal to a key or a value.  Reported per
//! distance: `acc@1` (argmax == v_q), `acc@5` (v_q in the top 5),
//! `any_value` (argmax is one of the K values — copying without
//! addressing), and the mean log-probability of `v_q`.  This is the
//! retrieval metric the O(1) plan requires (S0/S7): perplexity is blind to
//! it, and a window-W anchor with no recurrent carry is expected to fall to
//! chance for D ≫ W.

use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

pub struct RecallArgs<'a> {
    pub models: &'a [String],
    pub pairs: usize,
    pub distances: Vec<usize>,
    pub trials: usize,
    pub seed: u64,
    /// `random` (fresh ids per position) or `repeat` (one id repeated).
    pub filler: String,
    pub json: bool,
}

/// splitmix64 — deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

struct DistanceResult {
    distance: usize,
    trials: usize,
    acc1: f64,
    acc5: f64,
    any_value: f64,
    mean_logp: f64,
    ms_per_trial: f64,
}

struct ModelResult {
    model: String,
    pairs: usize,
    filler: String,
    seed: u64,
    backend: String,
    results: Vec<DistanceResult>,
}

fn log_softmax_at(logits: &[f32], id: usize) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    logits[id] as f64 - max - sum.ln()
}

/// Rank of `id` among the logits (0 = argmax).
fn rank_of(logits: &[f32], id: usize) -> usize {
    let v = logits[id];
    logits.iter().filter(|&&x| x > v).count()
}

pub fn cmd_probe_recall(a: RecallArgs<'_>) -> anyhow::Result<()> {
    if a.pairs == 0 || a.trials == 0 || a.distances.is_empty() {
        anyhow::bail!("probe-recall: pairs, trials and distances must be non-empty");
    }
    let longest = a.distances.iter().copied().max().unwrap_or(0) + 2 * a.pairs + 8;
    // A legacy (growing-KV) file must not evict mid-prompt; a bounded
    // file ignores the cap (its operator has none).
    if std::env::var("CMF_MAX_SEQ").is_err() {
        // SAFETY: single-threaded here — before any pipeline/pool spawn.
        unsafe { std::env::set_var("CMF_MAX_SEQ", longest.to_string()) };
    }
    let backend = format!(
        "CMF_GPU={} resident={}",
        std::env::var("CMF_GPU").unwrap_or_else(|_| "unset".into()),
        std::env::var("CMF_EMBRYO_RESIDENT").unwrap_or_else(|_| "unset".into())
    );
    let mut all = Vec::new();
    for path in a.models {
        let model = Arc::new(CmfModel::open_sharded(path).with_context(|| path.clone())?);
        let mut pipeline = Pipeline::from_model(
            &model,
            SamplerConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                seed: Some(0),
                ..Default::default()
            },
        )?;
        let vocab = model.arch().vocab_size;
        // Token pool: skip the byte/special region at both ends.
        let lo = 256usize.min(vocab / 8);
        let hi = vocab.saturating_sub(64).max(lo + 16);
        let span = (hi - lo) as u64;
        let bos = pipeline.tokenizer.with_bos(Vec::new());
        if !a.json {
            println!(
                "probe-recall: {path} | pairs={} trials={} filler={} pool=[{lo},{hi}) bos={} | {backend}",
                a.pairs,
                a.trials,
                a.filler,
                bos.len()
            );
            println!("{:>9} {:>7} {:>7} {:>9} {:>10} {:>9}", "D", "acc@1", "acc@5", "any_val", "mean_logp", "ms/trial");
        }
        let mut results = Vec::new();
        for &d in &a.distances {
            let mut rng = Rng(a.seed ^ (d as u64).wrapping_mul(0x5851_F42D_4C95_7F2D));
            let (mut hit1, mut hit5, mut anyv) = (0usize, 0usize, 0usize);
            let mut logp_sum = 0f64;
            let t0 = std::time::Instant::now();
            for _ in 0..a.trials {
                // Distinct keys, distinct values, keys ∩ values = ∅.
                let mut used: Vec<u32> = Vec::with_capacity(2 * a.pairs);
                let draw = |rng: &mut Rng, used: &mut Vec<u32>| -> u32 {
                    loop {
                        let id = (lo as u64 + rng.below(span)) as u32;
                        if !used.contains(&id) {
                            used.push(id);
                            return id;
                        }
                    }
                };
                let keys: Vec<u32> = (0..a.pairs).map(|_| draw(&mut rng, &mut used)).collect();
                let vals: Vec<u32> = (0..a.pairs).map(|_| draw(&mut rng, &mut used)).collect();
                let q = rng.below(a.pairs as u64) as usize;
                let mut ids = bos.clone();
                for i in 0..a.pairs {
                    ids.push(keys[i]);
                    ids.push(vals[i]);
                }
                let repeat_id = draw(&mut rng, &mut used);
                for _ in 0..d {
                    let f = if a.filler == "repeat" {
                        repeat_id
                    } else {
                        loop {
                            let id = (lo as u64 + rng.below(span)) as u32;
                            if !used.contains(&id) {
                                break id;
                            }
                        }
                    };
                    ids.push(f);
                }
                ids.push(keys[q]);
                let logits = pipeline
                    .forward_ids(&ids, None)
                    .map_err(|e| anyhow::anyhow!("{path}: forward_ids: {e}"))?;
                let target = vals[q] as usize;
                let r = rank_of(&logits, target);
                if r == 0 {
                    hit1 += 1;
                }
                if r < 5 {
                    hit5 += 1;
                }
                let argmax = logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
                    .0 as u32;
                if vals.contains(&argmax) {
                    anyv += 1;
                }
                logp_sum += log_softmax_at(&logits, target);
            }
            let n = a.trials as f64;
            let res = DistanceResult {
                distance: d,
                trials: a.trials,
                acc1: hit1 as f64 / n,
                acc5: hit5 as f64 / n,
                any_value: anyv as f64 / n,
                mean_logp: logp_sum / n,
                ms_per_trial: t0.elapsed().as_secs_f64() * 1e3 / n,
            };
            if !a.json {
                println!(
                    "{:>9} {:>7.3} {:>7.3} {:>9.3} {:>10.3} {:>9.1}",
                    res.distance, res.acc1, res.acc5, res.any_value, res.mean_logp, res.ms_per_trial
                );
            }
            results.push(res);
        }
        all.push(ModelResult {
            model: path.clone(),
            pairs: a.pairs,
            filler: a.filler.clone(),
            seed: a.seed,
            backend: backend.clone(),
            results,
        });
    }
    if a.json {
        let arr: Vec<serde_json::Value> = all
            .iter()
            .map(|m| {
                serde_json::json!({
                    "model": m.model,
                    "pairs": m.pairs,
                    "filler": m.filler,
                    "seed": m.seed,
                    "backend": m.backend,
                    "results": m.results.iter().map(|r| serde_json::json!({
                        "distance": r.distance,
                        "trials": r.trials,
                        "acc1": r.acc1,
                        "acc5": r.acc5,
                        "any_value": r.any_value,
                        "mean_logp": r.mean_logp,
                        "ms_per_trial": r.ms_per_trial,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    }
    Ok(())
}
