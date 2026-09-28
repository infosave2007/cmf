//! Scoring a skill and batch evaluation of decision files (spec §3.5, §4.1).
//!
//! [`SkillScorer`] is the runtime of one skill: the active tasks of the skill in
//! task order, interleaved for [`Packed`] scoring (bit-exact with the reference
//! error), their f32 statistics and the f32 gate. The build pipeline certifies
//! with it, the post-write self-check re-runs it on the file, and `cortiq decide
//! --input` ([`Evaluator`]) answers with it.
//!
//! Batch output (one JSON object per input row, never the text):
//! `{i, text_sha256, skill, choice, p_top, confidence, novelty, margin, accepted,
//! certified, input_tokens, errors_top5, timings_us, correct?}`:
//! * `choice`: the winning label (`null` when the skill has no active task);
//! * `confidence`: `(N·p_top − 1)/(N − 1)` in f32 with `N` the number of active
//!   labels (Jev's formula, spec §4.7; `p_top` when `N = 1`);
//! * `accepted`: `p_top ≥ τ` and `novelty ≤ θ` (spec §3.5);
//! * `certified`: the skill's gate is certified and the winner's task comes from
//!   the data (`origin = data`) — the exact-match rule of spec §4.5 for a text
//!   state; it does not depend on `accepted`;
//! * `input_tokens`: WordPiece tokens of the text without `[CLS]`/`[SEP]` and
//!   without truncation (the state part of the metering of spec §4.9);
//! * `errors_top5`: the 5 smallest errors, label → E, in rank order;
//! * `timings_us`: `tokenize`, `encode` (BERT, pooling, both L2), `hash` (φ_H),
//!   `resonance` (errors and decision), `total` (the row, parse excluded);
//! * `correct`: present when the row has a label: `choice == label`.
//!
//! f32 numbers are written in their shortest f32 form ([`f32_json`]).

use crate::container::{DecisionModel, TopologyRef};
use crate::data::{self, check_label, check_text};
use crate::manifest::{GateParams, SkillManifest, TaskOrigin, TaskRecord};
use crate::packed::{LANES, Packed};
use crate::resonance::{Decision, ErrStats, TaskView, Topology, decide};
use crate::rows::Row;
use crate::signal::SignalEncoder;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Errors listed per row.
pub const TOP_ERRORS: usize = 5;
/// Warm-up texts of `--bench` (spec §6.6).
pub const BENCH_WARMUP: usize = 50;

/// An f32 as a JSON number in its shortest f32 form (`0.97`, not
/// `0.9700000286102295`); non-finite values are `null`.
pub fn f32_json(x: f32) -> Value {
    if !x.is_finite() {
        return Value::Null;
    }
    let v: f64 = x.to_string().parse().expect("f32 Display parses as f64");
    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
}

/// Jev's confidence `(N·p_max − 1)/(N − 1)` in f32; `p_max` when `N ≤ 1`.
pub fn jev_confidence(p_max: f32, n: usize) -> f32 {
    if n <= 1 {
        return p_max;
    }
    let nf = n as f32;
    (nf * p_max - 1.0) / (nf - 1.0)
}

/// `threads` for the build and evaluation: 0 means the available parallelism.
pub fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    }
}

/// `f(i)` for `i in 0..n` on up to `threads` scoped threads, in index order
/// (the result does not depend on `threads`).
pub fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = resolve_threads(threads).clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        out.push((i, f(i)));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for part in parts {
        for (i, v) in part {
            slots[i] = Some(v);
        }
    }
    slots
        .into_iter()
        .map(|v| v.expect("every index computed"))
        .collect()
}

// ------------------------------------------------------------------ skill scorer

/// The runtime of one skill: its active tasks (the candidates, in task order),
/// packed for scoring, with their statistics and the gate.
pub struct SkillScorer {
    id: String,
    /// Task index (in the skill) of each candidate.
    tasks: Vec<usize>,
    /// Label of each candidate.
    labels: Vec<String>,
    origins: Vec<TaskOrigin>,
    stats: Vec<ErrStats>,
    /// Candidate of each task of the skill (`None`: not active).
    candidate_of: Vec<Option<usize>>,
    packed: Packed,
    dim: usize,
    gate: GateParams,
}

impl std::fmt::Debug for SkillScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillScorer")
            .field("id", &self.id)
            .field("candidates", &self.labels.len())
            .field("dim", &self.dim)
            .field("gate", &self.gate)
            .finish()
    }
}

impl SkillScorer {
    /// A scorer from task records and their topologies (`views[i]` belongs to
    /// `tasks[i]`); only the active tasks are scored. `dim` is the signal
    /// dimension.
    pub fn new(
        id: &str,
        tasks: &[TaskRecord],
        views: &[Option<TaskView<'_>>],
        gate: GateParams,
        dim: usize,
    ) -> Result<Self> {
        ensure!(
            tasks.len() == views.len(),
            "skill '{id}': {} topologies for {} tasks",
            views.len(),
            tasks.len()
        );
        let mut cand_views = Vec::new();
        let mut s = Self {
            id: id.to_string(),
            tasks: Vec::new(),
            labels: Vec::new(),
            origins: Vec::new(),
            stats: Vec::new(),
            candidate_of: vec![None; tasks.len()],
            packed: Packed::new(&[])?,
            dim,
            gate,
        };
        for (i, (t, v)) in tasks.iter().zip(views).enumerate() {
            if !t.is_active() {
                continue;
            }
            let Some(v) = v else {
                bail!(
                    "skill '{id}': active task {i} ('{}') has no topology",
                    t.label
                );
            };
            v.check()?;
            ensure!(
                v.dim() == dim,
                "skill '{id}': task {i} has dimension {}, the signal {dim}",
                v.dim()
            );
            ensure!(
                v.rank() as u64 == t.k,
                "skill '{id}': task {i} has rank {}, its record k = {}",
                v.rank(),
                t.k
            );
            let stats = t.stats();
            stats
                .check()
                .with_context(|| format!("skill '{id}': task {i}"))?;
            s.candidate_of[i] = Some(s.tasks.len());
            s.tasks.push(i);
            s.labels.push(t.label.clone());
            s.origins.push(t.origin);
            s.stats.push(stats);
            cand_views.push(*v);
        }
        s.packed = Packed::new(&cand_views)?;
        Ok(s)
    }

    /// A scorer from a skill manifest and owned topologies (one per task).
    pub fn from_manifest(
        manifest: &SkillManifest,
        topologies: &[Option<Topology>],
        dim: usize,
    ) -> Result<Self> {
        let views: Vec<Option<TaskView<'_>>> = topologies
            .iter()
            .map(|t| t.as_ref().map(Topology::view))
            .collect();
        Self::new(
            &manifest.id,
            &manifest.tasks,
            &views,
            manifest.gate.params(),
            dim,
        )
    }

    /// The scorer of a skill of a loaded file (base + overlay).
    pub fn from_model(model: &DecisionModel, id: &str) -> Result<Self> {
        let skill = model.skill(id).ok_or_else(|| {
            anyhow::anyhow!(
                "no skill {} in this file",
                crate::config::quote_unless_key(id)
            )
        })?;
        let m = &skill.manifest;
        let mut held: Vec<Option<TopologyRef<'_>>> = Vec::new();
        for (i, t) in m.tasks.iter().enumerate() {
            held.push(if t.is_active() {
                model.task_view(id, i)?
            } else {
                None
            });
        }
        let views: Vec<Option<TaskView<'_>>> = held
            .iter()
            .map(|h| {
                h.as_ref().map(|(mean, basis)| TaskView {
                    mean: mean.as_ref(),
                    basis: basis.as_ref(),
                })
            })
            .collect();
        Self::new(id, &m.tasks, &views, m.gate.params(), model.signal_dim())?
            .with_device(crate::bert::EncoderDevice::from_env()?)
    }

    /// Select the reconstruction backend without changing topology parameters.
    pub fn with_device(mut self, device: crate::bert::EncoderDevice) -> Result<Self> {
        self.packed = self.packed.with_device(device)?;
        Ok(self)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Number of candidates (active tasks).
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Signal dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Labels of the candidates, in task order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Task index (in the skill) of each candidate.
    pub fn tasks(&self) -> &[usize] {
        &self.tasks
    }

    /// Origin of each candidate's task.
    pub fn origins(&self) -> &[TaskOrigin] {
        &self.origins
    }

    /// The f32 statistics of the candidates.
    pub fn stats(&self) -> &[ErrStats] {
        &self.stats
    }

    /// The candidate of a skill task (`None` when the task is not active).
    pub fn candidate_of(&self, task: usize) -> Option<usize> {
        self.candidate_of.get(task).copied().flatten()
    }

    pub fn gate(&self) -> GateParams {
        self.gate
    }

    /// Replace the gate (the build certifies after the topologies exist).
    pub fn set_gate(&mut self, gate: GateParams) {
        self.gate = gate;
    }

    pub fn packed(&self) -> &Packed {
        &self.packed
    }

    /// Errors of a signal against every candidate.
    pub fn errors(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.len()];
        let mut scratch = self.packed.scratch();
        self.errors_with(x, &mut out, &mut scratch)?;
        Ok(out)
    }

    /// [`SkillScorer::errors`] into `out` with a caller-owned scratch.
    pub fn errors_with(
        &self,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut Vec<[f32; LANES]>,
    ) -> Result<()> {
        ensure!(
            x.len() == self.dim,
            "signal has {} values, the skill {}",
            x.len(),
            self.dim
        );
        self.packed.errors_with(x, out, scratch)
    }

    /// The decision over all candidates from their errors, at the skill's `T`.
    pub fn decide_errors(&self, errors: &[f32]) -> Result<Decision> {
        decide(errors, &self.stats, self.gate.temperature)
    }

    /// Errors and decision of a signal.
    pub fn decide(&self, x: &[f32]) -> Result<(Vec<f32>, Decision)> {
        let e = self.errors(x)?;
        let d = self.decide_errors(&e)?;
        Ok((e, d))
    }

    /// The gate: `p_top ≥ τ` and `novelty ≤ θ`.
    pub fn accepted(&self, d: &Decision) -> bool {
        d.accepted(self.gate.tau, self.gate.novelty_theta)
    }

    /// Certified gate and a winner whose task comes from the data.
    pub fn certified(&self, d: &Decision) -> bool {
        self.gate.certified
            && d.winner
                .is_some_and(|w| self.origins[w] == TaskOrigin::Data)
    }

    /// The row-major error matrix `rows × candidates` of stored rows (their
    /// signals `[φ_P ; 0.5·φ_H]`), on up to `threads` threads.
    pub fn error_matrix(&self, rows: &[&Row], dim_h: usize, threads: usize) -> Result<Vec<f32>> {
        let n = self.len();
        let chunk = 64usize;
        let blocks = rows.len().div_ceil(chunk);
        let parts: Vec<Result<Vec<f32>>> = par_map(blocks, threads, |b| {
            let lo = b * chunk;
            let hi = (lo + chunk).min(rows.len());
            let mut out = vec![0.0f32; (hi - lo) * n];
            let mut x = vec![0.0f32; self.dim];
            let mut scratch = self.packed.scratch();
            for (r, o) in rows[lo..hi].iter().zip(out.chunks_exact_mut(n.max(1))) {
                ensure!(
                    r.phi_p.len() + dim_h == self.dim,
                    "row signal dimension {} differs from the skill's {}",
                    r.phi_p.len() + dim_h,
                    self.dim
                );
                r.signal_into(dim_h, &mut x);
                if n > 0 {
                    self.errors_with(&x, o, &mut scratch)?;
                }
            }
            Ok(out)
        });
        let mut all = Vec::with_capacity(rows.len() * n);
        for p in parts {
            all.extend(p?);
        }
        Ok(all)
    }
}

// ------------------------------------------------------------------ batch input

/// One row of `cortiq decide --input`: `{"text","label"?}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalInput {
    pub text: String,
    pub label: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInput {
    text: String,
    #[serde(default)]
    label: Option<String>,
}

/// Parse batch input JSONL (the text rule of spec §3.2; any other key is an error).
pub fn parse_input(name: &str, bytes: &[u8]) -> Result<Vec<EvalInput>> {
    let (lines, _) = data::jsonl_lines(bytes);
    let mut out = Vec::with_capacity(lines.len());
    for (line, l) in lines {
        let r: RawInput = serde_json::from_slice(l)
            .map_err(|e| anyhow::anyhow!("{name}:{line}: expected {{\"text\",\"label\"?}}: {e}"))?;
        check_text(&r.text).map_err(|e| anyhow::anyhow!("{name}:{line}: {e}"))?;
        if let Some(lab) = &r.label {
            check_label(lab).map_err(|e| anyhow::anyhow!("{name}:{line}: {e}"))?;
        }
        out.push(EvalInput {
            text: r.text,
            label: r.label,
        });
    }
    Ok(out)
}

/// Read batch input JSONL.
pub fn read_input(path: impl AsRef<Path>) -> Result<Vec<EvalInput>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_input(&path.display().to_string(), &bytes)
}

/// The skill of a batch run: `--skill`, else the only skill of the file, else
/// an error listing the skills (spec §4.1).
pub fn select_skill(model: &DecisionModel, skill: Option<&str>) -> Result<String> {
    let ids: Vec<&str> = model.skills().iter().map(|s| s.id()).collect();
    match skill {
        Some(id) => {
            ensure!(
                ids.contains(&id),
                "no skill {} in this file (skills: {})",
                crate::config::quote_unless_key(id),
                ids.join(", ")
            );
            Ok(id.to_string())
        }
        None => match ids.as_slice() {
            [one] => Ok(one.to_string()),
            [] => bail!("this file has no skill"),
            _ => bail!(
                "this file has {} skills; choose one with --skill ({})",
                ids.len(),
                ids.join(", ")
            ),
        },
    }
}

// ------------------------------------------------------------------ evaluation

/// Wall time of the stages of one row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    pub tokenize: Duration,
    pub encode: Duration,
    /// Joint GPU encoder + reconstruction, excluding CPU ranking.
    pub gpu: Duration,
    pub hash: Duration,
    pub resonance: Duration,
    pub total: Duration,
}

impl StageTimings {
    pub const STAGES: [&'static str; 6] =
        ["tokenize", "encode", "hash", "gpu", "resonance", "total"];

    pub fn get(&self, stage: &str) -> Duration {
        match stage {
            "tokenize" => self.tokenize,
            "encode" => self.encode,
            "hash" => self.hash,
            "gpu" => self.gpu,
            "resonance" => self.resonance,
            _ => self.total,
        }
    }

    /// Whole microseconds per stage.
    pub fn to_json(&self) -> Value {
        let us = |d: Duration| d.as_micros() as u64;
        let mut value = json!({
            "tokenize": us(self.tokenize), "encode": us(self.encode), "hash": us(self.hash),
            "resonance": us(self.resonance), "total": us(self.total),
        });
        if !self.gpu.is_zero() {
            value["gpu"] = json!(us(self.gpu));
        }
        value
    }
}

/// The decision of one text.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDecision {
    pub errors: Vec<f32>,
    pub decision: Decision,
    /// WordPiece tokens without `[CLS]`/`[SEP]`, untruncated.
    pub input_tokens: usize,
    /// Tokens through the encoder (`[CLS]`/`[SEP]` included, truncated).
    pub processed_tokens: usize,
    pub timings: StageTimings,
}

/// One output row of a batch run.
#[derive(Clone, Debug, PartialEq)]
pub struct RowResult {
    pub i: usize,
    pub text_sha256: String,
    pub skill: String,
    pub choice: Option<String>,
    pub p_top: f32,
    pub confidence: f32,
    pub novelty: f32,
    pub margin: f32,
    pub accepted: bool,
    pub certified: bool,
    pub input_tokens: usize,
    /// The smallest errors, label and E, in rank order.
    pub errors_top5: Vec<(String, f32)>,
    pub timings: StageTimings,
    pub correct: Option<bool>,
}

impl RowResult {
    /// The output object (fields in the order of spec §4.1).
    pub fn to_json(&self) -> Value {
        let mut errors = Map::new();
        for (l, e) in &self.errors_top5 {
            errors.insert(l.clone(), f32_json(*e));
        }
        let mut v = json!({
            "i": self.i,
            "text_sha256": self.text_sha256,
            "skill": self.skill,
            "choice": self.choice,
            "p_top": f32_json(self.p_top),
            "confidence": f32_json(self.confidence),
            "novelty": f32_json(self.novelty),
            "margin": f32_json(self.margin),
            "accepted": self.accepted,
            "certified": self.certified,
            "input_tokens": self.input_tokens,
            "errors_top5": errors,
            "timings_us": self.timings.to_json(),
        });
        if let Some(c) = self.correct {
            v["correct"] = json!(c);
        }
        v
    }

    /// The row without its timings (for bit-exact comparisons of two runs).
    pub fn without_timings(&self) -> Self {
        Self {
            timings: StageTimings::default(),
            ..self.clone()
        }
    }
}

/// p50/p95/p99 and mean of one stage, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub mean: f64,
}

/// numpy's default (linear) percentile of sorted values.
pub fn percentile_sorted(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let rank = q / 100.0 * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (rank - lo as f64)
}

impl Percentiles {
    pub fn of(values_us: &[f64]) -> Self {
        let mut v = values_us.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
        Self {
            p50: percentile_sorted(&v, 50.0),
            p95: percentile_sorted(&v, 95.0),
            p99: percentile_sorted(&v, 99.0),
            mean: if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            },
        }
    }
}

/// Options of a batch run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalOptions {
    /// Stage percentiles in the summary.
    pub bench: bool,
    /// Texts decided (and discarded) before the timed run (spec §6.6: 50).
    pub warmup: usize,
}

/// Totals of a batch run.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalSummary {
    pub skill: String,
    pub model_sha: String,
    pub n: usize,
    /// Rows with a label.
    pub labelled: usize,
    pub correct: usize,
    pub accepted: usize,
    pub accepted_labelled: usize,
    pub accepted_correct: usize,
    pub certified_accepted: usize,
    pub gate: GateParams,
    pub candidates: usize,
    /// Warm-up texts and per-stage percentiles (µs) with `bench`.
    pub bench: Option<(usize, Vec<(String, Percentiles)>)>,
}

impl EvalSummary {
    pub fn to_json(&self) -> Value {
        let ratio = |a: usize, b: usize| {
            if b == 0 {
                Value::Null
            } else {
                json!(a as f64 / b as f64)
            }
        };
        let mut v = json!({
            "skill": self.skill,
            "model_sha": self.model_sha,
            "n": self.n,
            "labelled": self.labelled,
            "correct": self.correct,
            "accuracy": ratio(self.correct, self.labelled),
            "accepted": self.accepted,
            "accepted_labelled": self.accepted_labelled,
            "accepted_correct": self.accepted_correct,
            "selective_accuracy": ratio(self.accepted_correct, self.accepted_labelled),
            "certified_accepted": self.certified_accepted,
            "candidates": self.candidates,
            "gate": {
                "temperature": f32_json(self.gate.temperature),
                "novelty_theta": f32_json(self.gate.novelty_theta),
                "tau": f32_json(self.gate.tau),
                "certified": self.gate.certified,
            },
        });
        if let Some((warmup, stages)) = &self.bench {
            let mut b = Map::new();
            b.insert("warmup".into(), json!(warmup));
            for (name, p) in stages {
                b.insert(
                    name.clone(),
                    json!({"p50_us": p.p50, "p95_us": p.p95, "p99_us": p.p99, "mean_us": p.mean}),
                );
            }
            v["bench"] = Value::Object(b);
        }
        v
    }

    /// Human-readable lines (stderr of `cortiq decide --input`).
    pub fn render(&self) -> String {
        let pct = |a: usize, b: usize| {
            if b == 0 {
                "-".to_string()
            } else {
                format!("{:.2}%", 100.0 * a as f64 / b as f64)
            }
        };
        let mut s = format!(
            "skill {} ({} labels, model {}): {} rows",
            self.skill,
            self.candidates,
            &self.model_sha[..12.min(self.model_sha.len())],
            self.n
        );
        if self.labelled > 0 {
            s.push_str(&format!(
                "; all rows {}/{} = {}",
                self.correct,
                self.labelled,
                pct(self.correct, self.labelled)
            ));
        }
        s.push_str(&format!(
            "\ngate (T {}, theta {}, tau {}, certified {}): accepted {}/{}",
            self.gate.temperature,
            self.gate.novelty_theta,
            self.gate.tau,
            self.gate.certified,
            self.accepted,
            self.n
        ));
        if self.accepted_labelled > 0 {
            s.push_str(&format!(
                ", correct {}/{} = {}",
                self.accepted_correct,
                self.accepted_labelled,
                pct(self.accepted_correct, self.accepted_labelled)
            ));
        }
        if let Some((warmup, stages)) = &self.bench {
            s.push_str(&format!("\nbench (warm-up {warmup}; µs p50/p95/p99):"));
            for (name, p) in stages {
                s.push_str(&format!(
                    "\n  {name:<10} {:>10.1} {:>10.1} {:>10.1}",
                    p.p50, p.p95, p.p99
                ));
            }
        }
        s
    }
}

/// Text → decision of one skill of a decision file (`cortiq decide`).
pub struct Evaluator {
    encoder: SignalEncoder,
    scorer: SkillScorer,
    model_sha: String,
}

impl std::fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator")
            .field("scorer", &self.scorer)
            .field("model_sha", &self.model_sha)
            .finish()
    }
}

impl Evaluator {
    /// The encoder of the file (golden-checked) and the scorer of `skill`.
    pub fn new(model: &DecisionModel, skill: &str) -> Result<Self> {
        let (encoder, _) = SignalEncoder::from_model(model)?;
        Ok(Self {
            scorer: SkillScorer::from_model(model, skill)?,
            encoder,
            model_sha: model.model_sha().to_string(),
        })
    }

    /// From an encoder and a scorer (`model_sha` is only reported).
    pub fn from_parts(encoder: SignalEncoder, scorer: SkillScorer, model_sha: String) -> Self {
        Self {
            encoder,
            scorer,
            model_sha,
        }
    }

    pub fn encoder(&self) -> &SignalEncoder {
        &self.encoder
    }

    pub fn scorer(&self) -> &SkillScorer {
        &self.scorer
    }

    /// Encode and decide one text, with the stage timings.
    pub fn decide_text(&self, text: &str) -> Result<TextDecision> {
        let t0 = Instant::now();
        let mut scored = self.encoder.score_timed(text, &[self.scorer.packed()])?;
        let t = scored.timings;
        let tokenizer = self.encoder.encoder().tokenizer();
        let input_tokens = if t.tokens < tokenizer.max_length() {
            t.tokens.saturating_sub(2)
        } else {
            tokenizer.encode_pieces(text).len()
        };
        let tr = Instant::now();
        let errors = scored.errors.remove(0);
        let decision = self.scorer.decide_errors(&errors)?;
        let end = Instant::now();
        Ok(TextDecision {
            errors,
            decision,
            input_tokens,
            processed_tokens: t.tokens,
            timings: StageTimings {
                tokenize: t.tokenize,
                encode: t.encode,
                hash: t.hash,
                gpu: t.gpu,
                resonance: scored.resonance + (end - tr),
                total: end - t0,
            },
        })
    }

    /// The output row of input `i`.
    pub fn row(&self, i: usize, input: &EvalInput) -> Result<RowResult> {
        let td = self.decide_text(&input.text)?;
        Ok(self.row_of(i, input, &td))
    }

    /// The output row of a decided text.
    pub fn row_of(&self, i: usize, input: &EvalInput, td: &TextDecision) -> RowResult {
        let d = &td.decision;
        let s = &self.scorer;
        let choice = d.winner.map(|w| s.labels()[w].clone());
        let errors_top5 = d
            .ranked
            .iter()
            .take(TOP_ERRORS)
            .map(|r| (s.labels()[r.index].clone(), r.error))
            .collect();
        RowResult {
            i,
            text_sha256: data::text_sha256(&input.text),
            skill: s.id().to_string(),
            correct: input
                .label
                .as_ref()
                .map(|l| choice.as_deref() == Some(l.as_str())),
            choice,
            p_top: d.p_top,
            confidence: if d.winner.is_some() {
                jev_confidence(d.p_top, s.len())
            } else {
                0.0
            },
            novelty: d.novelty,
            margin: d.margin,
            accepted: s.accepted(d),
            certified: s.certified(d),
            input_tokens: td.input_tokens,
            errors_top5,
            timings: td.timings,
        }
    }

    /// Decide every input row in order (one text at a time); `emit` receives
    /// each row as it is decided.
    pub fn run(
        &self,
        inputs: &[EvalInput],
        opts: EvalOptions,
        mut emit: impl FnMut(&RowResult) -> Result<()>,
    ) -> Result<EvalSummary> {
        let warmup = opts.warmup.min(inputs.len());
        for input in &inputs[..warmup] {
            self.decide_text(&input.text)?;
        }
        let s = &self.scorer;
        let mut sum = EvalSummary {
            skill: s.id().to_string(),
            model_sha: self.model_sha.clone(),
            n: 0,
            labelled: 0,
            correct: 0,
            accepted: 0,
            accepted_labelled: 0,
            accepted_correct: 0,
            certified_accepted: 0,
            gate: s.gate(),
            candidates: s.len(),
            bench: None,
        };
        let mut stage_us: Vec<Vec<f64>> = vec![Vec::new(); StageTimings::STAGES.len()];
        for (i, input) in inputs.iter().enumerate() {
            let r = self.row(i, input)?;
            sum.n += 1;
            if let Some(c) = r.correct {
                sum.labelled += 1;
                sum.correct += usize::from(c);
                if r.accepted {
                    sum.accepted_labelled += 1;
                    sum.accepted_correct += usize::from(c);
                }
            }
            if r.accepted {
                sum.accepted += 1;
                sum.certified_accepted += usize::from(r.certified);
            }
            if opts.bench {
                for (k, name) in StageTimings::STAGES.iter().enumerate() {
                    stage_us[k].push(r.timings.get(name).as_nanos() as f64 / 1000.0);
                }
            }
            emit(&r)?;
        }
        if opts.bench {
            sum.bench = Some((
                warmup,
                StageTimings::STAGES
                    .iter()
                    .zip(&stage_us)
                    .filter(|(name, values)| **name != "gpu" || values.iter().any(|&v| v > 0.0))
                    .map(|(n, v)| (n.to_string(), Percentiles::of(v)))
                    .collect(),
            ));
        }
        Ok(sum)
    }
}
