//! The training pipeline (spec §3): data rules → encode → fit → certify →
//! write → self-check.
//!
//! * [`train`] (`cortiq decision train --encoder FROM.cmf`): the encoder of FROM
//!   copied byte for byte plus one new skill;
//! * [`add_skill`] (`cortiq decision add-skill IN.cmf`): every tensor and
//!   manifest of IN copied byte for byte plus one new skill; an existing skill id
//!   is refused (spec §3.7: only `decision.manifest` is new);
//! * [`init_encoder`] (`cortiq decision init`): an encoder-only file from an
//!   export directory.
//!
//! One skill is built as follows:
//! 1. **Data** ([`crate::data::prepare`], spec §3.2): one or more training files
//!    (concatenated), the calibration file or the carve-out, halves and holdout
//!    flags, the optional dev file and question.
//! 2. **Encode** (spec §3.3): `x = [φ_P ; 0.5·φ_H]` of every text, in parallel over
//!    texts (`threads`; the result does not depend on it), kept as
//!    `cortiq-decision-rows-v1` rows (φ_H sparse).
//! 3. **Fit** (spec §3.4, [`crate::fit`]): each task from its own training rows
//!    in the stored order (input order), `k = min(K, n−1)` after the eigenvalue
//!    drop; a one-row label is an inactive task with `k = 0`.
//! 4. **Certify** (spec §3.6, [`crate::certify`]): the f32 runtime errors of every
//!    calibration row against the active tasks ([`SkillScorer`], packed and
//!    bit-exact with the reference), `T` and `θ` on the even half, `τ` on the odd
//!    half.
//! 5. **Dev** (optional): the rows the built skill decides correctly, recorded
//!    in the manifest.
//! 6. **Write**: [`FileBuilder`] writes a staging file next to the output (temp
//!    file, fsync, strict re-open), then
//! 7. **Self-check** on the staging file: the file is opened again
//!    ([`Verify::Full`] and the encoder golden), the calibration errors,
//!    probabilities and novelties are recomputed from the stored rows and
//!    topologies and must equal the values the gate was certified on bit for
//!    bit, the gate must equal the certification, and the first 64 calibration
//!    texts re-encoded with the stored encoder must equal their stored rows bit
//!    for bit.
//! 8. **Publish**: the staging file is hard-linked to the output, which never
//!    replaces an existing file (a `create_new` copy where links are
//!    unsupported); a file that fails its self-check is never published.
//!
//! With the same inputs and `created_unix` (`SOURCE_DATE_EPOCH`, else 0) the
//! output bytes are identical.

use crate::bert::EncoderExport;
use crate::certify::{self, Calibration, Certification, RowGate};
use crate::container::{DecisionModel, FileBuilder, NewSkill, OutputExists, Verify, WriteReport};
use crate::data::{self, CalibrationSource, DataInputs, DataReport, SkillData};
use crate::eval::{SkillScorer, par_map, resolve_threads};
use crate::fit::{self, TaskFit};
use crate::manifest::{
    self, CalibrationRecord, DEFAULT_ENCODER_GOLDEN_TEXTS, DEFAULT_MODEL_ID, DEFAULT_NAME,
    DataRecord, DevRecord, Gate, GateParams, InputPart, Recipe, RowsRecord, SkillManifest,
    TaskOrigin, TaskRecord, TaskState, TrainRecord,
};
use crate::resonance::Topology;
use crate::rows::{Row, Rows, Source, Split};
use crate::signal::{Features, SignalEncoder};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Calibration texts re-encoded by the self-check.
pub const SELF_CHECK_TEXTS: usize = 64;
/// Texts encoded per batch (bounds the memory of dense φ_H).
pub const ENCODE_CHUNK: usize = 512;

/// The inputs and settings of one skill (`train` / `add-skill`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainOptions {
    /// Skill id (`[a-z0-9][a-z0-9_-]{0,63}`).
    pub skill: String,
    /// One or more training files, concatenated in order (train ∪ dev: spec §3.9).
    pub train: Vec<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub dev: Option<PathBuf>,
    pub question: Option<PathBuf>,
    /// `K`, the most directions per task (spec §3.9: per skill).
    pub k: usize,
    /// Where `K` came from when it is not the default (recorded in the recipe).
    pub k_source: Option<String>,
    /// Encoder, fit and scoring threads; 0 = the available parallelism.
    pub threads: usize,
    /// `created_unix` of the new `decision.manifest` (`None`: `SOURCE_DATE_EPOCH`
    /// or 0).
    pub created_unix: Option<u64>,
}

impl TrainOptions {
    /// Options with the default `K` (16) and threads.
    pub fn new(skill: impl Into<String>, train: Vec<PathBuf>) -> Self {
        Self {
            skill: skill.into(),
            train,
            calibration: None,
            dev: None,
            question: None,
            k: fit::DEFAULT_K,
            k_source: None,
            threads: 0,
            created_unix: None,
        }
    }

    fn check(&self) -> Result<()> {
        ensure!(
            manifest::valid_skill_id(&self.skill),
            "invalid skill id '{}' (expected [a-z0-9][a-z0-9_-]{{0,63}})",
            self.skill
        );
        ensure!(
            !self.train.is_empty(),
            "at least one --train file is needed"
        );
        ensure!(self.k >= 1, "--k must be at least 1");
        if let Some(s) = &self.k_source {
            ensure!(
                !s.is_empty() && s.len() <= 1024,
                "k_source must have 1..=1024 bytes"
            );
        }
        Ok(())
    }
}

/// Dev rows decided by the built skill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevEval {
    pub n: usize,
    pub correct: usize,
    /// Rows accepted by the certified gate, and how many of them are correct.
    pub accepted: usize,
    pub accepted_correct: usize,
}

/// What the self-check compared (every comparison is bit for bit).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelfCheck {
    /// Calibration rows whose errors, winner, `p_top` and novelty were recomputed
    /// from the file.
    pub calibration_rows: usize,
    /// Error values compared.
    pub errors_compared: usize,
    /// Calibration texts re-encoded with the file's encoder.
    pub texts_reencoded: usize,
}

/// Wall time of the build stages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BuildTimings {
    pub load: Duration,
    pub encode: Duration,
    pub fit: Duration,
    pub certify: Duration,
    pub dev: Duration,
    pub write: Duration,
    pub self_check: Duration,
    pub total: Duration,
}

/// The outcome of [`train`] / [`add_skill`].
#[derive(Clone, Debug)]
pub struct BuildReport {
    /// The published file.
    pub out: WriteReport,
    pub skill: String,
    /// sha256 of the new skill's manifest bytes.
    pub skill_manifest_sha256: String,
    /// The sealed manifest of the new skill.
    pub manifest: SkillManifest,
    /// Skills of the output file, in order.
    pub skills: Vec<String>,
    /// Rows of the whole calibration set whose winner is the label.
    pub calibration_correct: usize,
    pub dev: Option<DevEval>,
    pub data: DataReport,
    pub self_check: SelfCheck,
    pub threads: usize,
    pub texts_encoded: usize,
    pub timings: BuildTimings,
    /// Data and build warnings.
    pub warnings: Vec<String>,
}

impl BuildReport {
    /// The report as JSON (printed by `cortiq decision train/add-skill --json`).
    pub fn to_json(&self) -> Value {
        let m = &self.manifest;
        let g = &m.gate;
        let chosen = g
            .evidence
            .odd
            .grid
            .iter()
            .find(|r| g.certified && r.t as f32 as f64 == g.tau);
        let secs = |d: Duration| d.as_secs_f64();
        let t = &self.timings;
        let active = m.tasks.iter().filter(|t| t.is_active()).count();
        let mut ranks: Vec<u64> = m.tasks.iter().map(|t| t.k).collect();
        ranks.sort_unstable();
        json!({
            "out": {
                "path": self.out.path.display().to_string(),
                "sha256": self.out.sha256,
                "bytes": self.out.bytes,
                "model_sha": self.out.model_sha,
                "tensors": self.out.tensors,
            },
            "skills": self.skills,
            "skill": {
                "id": self.skill,
                "manifest_sha256": self.skill_manifest_sha256,
                "labels": m.labels.len(),
                "active": active,
                "inactive": m.tasks.len() - active,
                "K": m.recipe.k_max,
                "k_min": ranks.first(),
                "k_max": ranks.last(),
                "rows": {"train": m.rows.n_train, "calibration": m.rows.n_calibration},
                "calibration_source": m.data.calibration.source,
            },
            "gate": {
                "temperature": g.temperature,
                "novelty_theta": g.novelty_theta,
                "tau": g.tau,
                "certified": g.certified,
                "even": {"n": g.evidence.even.n, "n_t": g.evidence.even.n_t, "nll": g.evidence.even.nll},
                "odd": {
                    "n": g.evidence.odd.n,
                    "accepted": g.evidence.odd.accepted,
                    "correct": g.evidence.odd.correct,
                    "lb": chosen.map(|r| r.lb),
                },
            },
            "calibration": {"n": m.data.calibration.n, "correct": self.calibration_correct},
            "dev": self.dev.map(|d| json!({
                "n": d.n, "correct": d.correct, "accepted": d.accepted, "accepted_correct": d.accepted_correct,
            })),
            "data": self.data.to_json(),
            "self_check": {
                "bit_exact": true,
                "calibration_rows": self.self_check.calibration_rows,
                "errors_compared": self.self_check.errors_compared,
                "texts_reencoded": self.self_check.texts_reencoded,
            },
            "threads": self.threads,
            "texts_encoded": self.texts_encoded,
            "timings_s": {
                "load": secs(t.load), "encode": secs(t.encode), "fit": secs(t.fit),
                "certify": secs(t.certify), "dev": secs(t.dev), "write": secs(t.write),
                "self_check": secs(t.self_check), "total": secs(t.total),
            },
            "warnings": self.warnings,
        })
    }
}

// ------------------------------------------------------------------ building blocks

/// Encode texts into rows (`task`, `split` and `flags` per text), in order, in
/// batches of [`ENCODE_CHUNK`] texts on `threads` threads.
pub fn encode_rows(
    encoder: &SignalEncoder,
    texts: &[&str],
    meta: &[(u32, Split, u8)],
    threads: usize,
) -> Vec<Row> {
    assert_eq!(texts.len(), meta.len(), "one (task, split, flags) per text");
    let mut out = Vec::with_capacity(texts.len());
    for (chunk, m) in texts.chunks(ENCODE_CHUNK).zip(meta.chunks(ENCODE_CHUNK)) {
        let feats = encoder.features_batch(chunk, resolve_threads(threads));
        for (f, &(task, split, flags)) in feats.iter().zip(m) {
            out.push(f.to_row(
                task,
                split,
                flags,
                Source::Data,
                Source::Data.default_weight(),
            ));
        }
    }
    out
}

/// Fit one task from rows (their signals `[φ_P ; 0.5·φ_H]`, in the given order)
/// with at most `k` directions.
pub fn fit_task_rows(rows: &[&Row], dim_h: usize, k: usize) -> Result<TaskFit> {
    ensure!(!rows.is_empty(), "a task needs at least one row");
    let dim = rows[0].phi_p.len() + dim_h;
    let mut x = vec![0.0f32; rows.len() * dim];
    for (r, o) in rows.iter().zip(x.chunks_exact_mut(dim)) {
        ensure!(
            r.phi_p.len() + dim_h == dim,
            "rows of one task differ in dimension"
        );
        r.signal_into(dim_h, o);
    }
    fit::fit_task(&x, dim, k)
}

/// Fit every task `0..tasks` from the rows of the given splits (in blob order),
/// on `threads` threads; `None` for a task without rows.
pub fn fit_tasks(
    rows: &Rows,
    tasks: usize,
    splits: &[Split],
    k: usize,
    threads: usize,
) -> Result<Vec<Option<TaskFit>>> {
    let mut per_task: Vec<Vec<&Row>> = vec![Vec::new(); tasks];
    for r in &rows.rows {
        if splits.contains(&r.split) {
            let t = r.task as usize;
            ensure!(t < tasks, "row task {t} out of range ({tasks} tasks)");
            per_task[t].push(r);
        }
    }
    let fits = par_map(tasks, threads, |t| {
        if per_task[t].is_empty() {
            Ok(None)
        } else {
            fit_task_rows(&per_task[t], rows.dim_h, k)
                .map(Some)
                .with_context(|| format!("fit of task {t}"))
        }
    });
    fits.into_iter().collect()
}

/// The manifest record of a fitted data task (hashes are filled by the writer).
pub fn task_record(i: usize, label: &str, fit: &TaskFit) -> TaskRecord {
    TaskRecord {
        i: i as u64,
        label: label.to_string(),
        state: if fit.n_train >= fit::MIN_ROWS_ACTIVE {
            TaskState::Active
        } else {
            TaskState::Inactive
        },
        origin: TaskOrigin::Data,
        k: fit.k as u64,
        n_train: fit.n_train as u64,
        err_mean: fit.err_mean,
        err_std: fit.err_std,
        mean_sha256: None,
        basis_sha256: None,
    }
}

/// The calibration error matrix of a skill (spec §3.6): the calibration rows of
/// a rows blob in blob order, their truth among the scorer's candidates and the
/// halves from the rows' flags.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationErrors {
    /// `rows × candidates`, row-major.
    pub errors: Vec<f32>,
    pub candidates: usize,
    /// Candidate of the row's label; `None` when its task is not active.
    pub truth: Vec<Option<usize>>,
    /// Row indices (into the calibration rows) of the even and odd halves.
    pub even: Vec<usize>,
    pub odd: Vec<usize>,
}

impl CalibrationErrors {
    /// The certification input with the scorer's statistics.
    pub fn calibration<'a>(&'a self, scorer: &'a SkillScorer) -> Calibration<'a> {
        Calibration {
            errors: &self.errors,
            tasks: self.candidates,
            stats: scorer.stats(),
            truth: &self.truth,
            even: &self.even,
            odd: &self.odd,
        }
    }
}

/// The calibration errors of the calibration rows of `rows` against `scorer`.
pub fn calibration_errors(
    scorer: &SkillScorer,
    rows: &Rows,
    threads: usize,
) -> Result<CalibrationErrors> {
    let cal: Vec<&Row> = rows
        .rows
        .iter()
        .filter(|r| r.split == Split::Calibration)
        .collect();
    let errors = scorer.error_matrix(&cal, rows.dim_h, threads)?;
    let mut even = Vec::new();
    let mut odd = Vec::new();
    for (j, r) in cal.iter().enumerate() {
        if r.odd_half() {
            odd.push(j);
        } else {
            even.push(j);
        }
    }
    Ok(CalibrationErrors {
        errors,
        candidates: scorer.len(),
        truth: cal
            .iter()
            .map(|r| scorer.candidate_of(r.task as usize))
            .collect(),
        even,
        odd,
    })
}

/// Certify a skill on the calibration rows of its rows blob (spec §3.6; also the
/// recertification of spec §5.9).
pub fn certify_rows(
    scorer: &SkillScorer,
    rows: &Rows,
    threads: usize,
) -> Result<(Certification, CalibrationErrors)> {
    ensure!(
        !scorer.is_empty(),
        "skill '{}' has no active task to certify",
        scorer.id()
    );
    let ce = calibration_errors(scorer, rows, threads)?;
    ensure!(
        !ce.truth.is_empty(),
        "skill '{}' has no calibration rows",
        scorer.id()
    );
    ensure!(
        ce.even.iter().any(|&j| ce.truth[j].is_some()),
        "skill '{}': no calibration row of the even half has an active label, so the temperature cannot be fitted",
        scorer.id()
    );
    let cert = certify::certify(&ce.calibration(scorer))?;
    Ok((cert, ce))
}

/// Decide dev rows (`truth`: the task of each row's label, if any).
pub fn evaluate_rows(
    scorer: &SkillScorer,
    rows: &[Row],
    truth: &[Option<usize>],
    dim_h: usize,
    threads: usize,
) -> Result<DevEval> {
    let refs: Vec<&Row> = rows.iter().collect();
    let errors = scorer.error_matrix(&refs, dim_h, threads)?;
    let n = scorer.len();
    let mut ev = DevEval {
        n: rows.len(),
        ..DevEval::default()
    };
    for (j, t) in truth.iter().enumerate() {
        let d = scorer.decide_errors(&errors[j * n..(j + 1) * n])?;
        let ok = match (d.winner, t) {
            (Some(w), Some(t)) => scorer.tasks()[w] == *t,
            _ => false,
        };
        ev.correct += usize::from(ok);
        if scorer.accepted(&d) {
            ev.accepted += 1;
            ev.accepted_correct += usize::from(ok);
        }
    }
    Ok(ev)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn gates_bit_eq(a: &[RowGate], b: &[RowGate]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| {
        x.winner != y.winner
            || x.p_top.to_bits() != y.p_top.to_bits()
            || x.novelty.to_bits() != y.novelty.to_bits()
    })
}

/// The post-write self-check (spec §3.6) of skill `id` in the file at `path`.
fn self_check(
    path: &Path,
    id: &str,
    expected: &CalibrationErrors,
    expected_gates: &[RowGate],
    gate: GateParams,
    texts: &[&str],
    threads: usize,
) -> Result<SelfCheck> {
    let model = DecisionModel::open(path, Verify::Full)
        .map_err(|e| anyhow::anyhow!("re-open for the self-check: {e}"))?;
    let (encoder, _) = SignalEncoder::from_model(&model)?;
    let scorer = SkillScorer::from_model(&model, id)?;
    let fg = scorer.gate();
    ensure!(
        fg.temperature.to_bits() == gate.temperature.to_bits()
            && fg.novelty_theta.to_bits() == gate.novelty_theta.to_bits()
            && fg.tau.to_bits() == gate.tau.to_bits()
            && fg.certified == gate.certified,
        "self-check: the stored gate {fg:?} differs from the certification {gate:?}"
    );
    let rows = model.rows(id)?;
    let got = calibration_errors(&scorer, &rows, threads)?;
    ensure!(
        got.truth == expected.truth && got.even == expected.even && got.odd == expected.odd,
        "self-check: the stored calibration rows differ in labels or halves"
    );
    ensure!(
        bits_eq(&got.errors, &expected.errors),
        "self-check: calibration errors recomputed from the file differ from the certified ones"
    );
    let gates = certify::row_gates(&got.calibration(&scorer), fg.temperature)?;
    if let Some(j) = gates_bit_eq(&gates, expected_gates) {
        bail!(
            "self-check: calibration row {j}: winner, p_top or novelty recomputed from the file differs"
        );
    }
    let cal: Vec<&Row> = rows
        .rows
        .iter()
        .filter(|r| r.split == Split::Calibration)
        .collect();
    let m = texts.len().min(SELF_CHECK_TEXTS).min(cal.len());
    let feats = encoder.features_batch(&texts[..m], resolve_threads(threads));
    for (j, f) in feats.iter().enumerate() {
        ensure!(
            f.bit_eq(&Features::from_row(cal[j], rows.dim_h)),
            "self-check: calibration text {j} re-encoded with the stored encoder differs from its stored row"
        );
    }
    Ok(SelfCheck {
        calibration_rows: cal.len(),
        errors_compared: got.errors.len(),
        texts_reencoded: m,
    })
}

// ------------------------------------------------------------------ publishing

/// Removes a path on drop unless disarmed.
struct RemoveOnDrop(Option<PathBuf>);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn out_dir(out: &Path) -> PathBuf {
    match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Refuse early: an existing output, a missing directory.
fn check_output(out: &Path) -> Result<()> {
    if std::fs::symlink_metadata(out).is_ok() {
        return Err(OutputExists(out.to_path_buf()).into());
    }
    let dir = out_dir(out);
    ensure!(
        dir.is_dir(),
        "output directory {} does not exist",
        dir.display()
    );
    ensure!(
        out.file_name().is_some(),
        "output {} has no file name",
        out.display()
    );
    Ok(())
}

static STAGING_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A staging path next to `out` that does not exist yet (the writer refuses
/// to replace it should it appear in between).
fn staging_path(out: &Path) -> PathBuf {
    let name = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    loop {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let k = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = out_dir(out).join(format!(
            ".{name}.{}.{nanos}.{k}.staging.cmf",
            std::process::id()
        ));
        if std::fs::symlink_metadata(&p).is_err() {
            return p;
        }
    }
}

fn file_sha256(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((format!("{:x}", h.finalize()), n))
}

/// Publish `staging` as `out` without ever replacing an existing file.
fn publish(staging: &Path, out: &Path) -> Result<()> {
    match std::fs::hard_link(staging, out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(OutputExists(out.to_path_buf()).into());
        }
        Err(_) => {
            // No hard links on this file system: copy into a new file.
            let mut dst = match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out)
            {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(OutputExists(out.to_path_buf()).into());
                }
                Err(e) => return Err(e.into()),
            };
            let mut guard = RemoveOnDrop(Some(out.to_path_buf()));
            let mut src = std::fs::File::open(staging)?;
            std::io::copy(&mut src, &mut dst)?;
            dst.flush()?;
            dst.sync_all()?;
            guard.0 = None;
        }
    }
    if let Ok(d) = std::fs::File::open(out_dir(out)) {
        let _ = d.sync_all();
    }
    Ok(())
}

// ------------------------------------------------------------------ the pipeline

/// Where the new skill goes.
#[derive(Clone, Copy)]
enum Target<'a> {
    /// `train`: the encoder of this file.
    EncoderOf(&'a DecisionModel),
    /// `add-skill`: every skill of this file.
    AddTo(&'a DecisionModel),
}

fn build(
    target: Target<'_>,
    data: SkillData,
    opts: &TrainOptions,
    out: &Path,
    t_start: Instant,
    t_load: Duration,
) -> Result<BuildReport> {
    let threads = resolve_threads(opts.threads);
    let model = match target {
        Target::EncoderOf(m) | Target::AddTo(m) => m,
    };
    let mut builder = match target {
        Target::EncoderOf(m) => FileBuilder::encoder_of(m),
        Target::AddTo(m) => FileBuilder::from_model(m)?,
    };
    ensure!(
        !builder.skill_ids().contains(&opts.skill.as_str()),
        "skill '{}' already exists in the input file (skills: {})",
        opts.skill,
        builder.skill_ids().join(", ")
    );
    if let Some(t) = opts.created_unix {
        builder.set_created_unix(t);
    }
    let (encoder, _) = SignalEncoder::from_model(model)?;
    let dim_p = encoder.dim_p();
    let dim_h = encoder.dim_h();
    let dim = encoder.dim();
    ensure!(
        dim == builder.signal_dim(),
        "encoder signal dimension {dim} differs from the file's {}",
        builder.signal_dim()
    );
    ensure!(
        opts.k <= dim,
        "--k {} exceeds the signal dimension {dim}",
        opts.k
    );

    // 2. Encode: train and calibration rows (stored order), then dev.
    let t0 = Instant::now();
    let mut texts: Vec<&str> = Vec::new();
    let mut meta: Vec<(u32, Split, u8)> = Vec::new();
    for r in &data.train {
        texts.push(&r.text);
        meta.push((r.task as u32, Split::Train, 0));
    }
    for r in &data.calibration {
        texts.push(&r.text);
        meta.push((r.task as u32, Split::Calibration, r.flags));
    }
    let n_stored = texts.len();
    if let Some(dev) = &data.dev {
        for r in dev {
            texts.push(&r.text);
            // The dev rows are never stored: task/split are placeholders.
            meta.push((0, Split::Train, 0));
        }
    }
    tracing::info!(
        skill = %opts.skill,
        texts = texts.len(),
        threads,
        "encoding"
    );
    let mut encoded = encode_rows(&encoder, &texts, &meta, threads);
    let dev_rows: Vec<Row> = encoded.split_off(n_stored);
    let rows = Rows {
        dim_p,
        dim_h,
        rows: encoded,
    };
    let rows_blob = rows.encode()?;
    let t_encode = t0.elapsed();

    // 3. Fit.
    let t0 = Instant::now();
    let fits = fit_tasks(&rows, data.labels.len(), &[Split::Train], opts.k, threads)?;
    let mut tasks = Vec::with_capacity(fits.len());
    let mut topologies: Vec<Option<Topology>> = Vec::with_capacity(fits.len());
    for (i, f) in fits.into_iter().enumerate() {
        let f = f.ok_or_else(|| anyhow::anyhow!("task {i} has no training rows"))?;
        tasks.push(task_record(i, &data.labels[i], &f));
        topologies.push(Some(f.topology));
    }
    let t_fit = t0.elapsed();

    // 4. Certify on the calibration rows.
    let t0 = Instant::now();
    let views: Vec<_> = topologies
        .iter()
        .map(|t| t.as_ref().map(Topology::view))
        .collect();
    let placeholder = GateParams {
        temperature: 1.0,
        novelty_theta: 1.0,
        tau: 0.0,
        certified: false,
    };
    let mut scorer = SkillScorer::new(&opts.skill, &tasks, &views, placeholder, dim)?;
    let (cert, cal_errors) = certify_rows(&scorer, &rows, threads)?;
    let gate = Gate::from_certification(&cert);
    scorer.set_gate(gate.params());
    let expected_gates = certify::row_gates(&cal_errors.calibration(&scorer), cert.temperature)?;
    let t_certify = t0.elapsed();
    tracing::info!(
        skill = %opts.skill,
        temperature = cert.temperature,
        theta = cert.novelty_theta,
        tau = cert.tau,
        certified = cert.certified,
        "certified"
    );

    // 5. Dev.
    let t0 = Instant::now();
    let dev = match &data.dev {
        Some(d) => {
            let truth: Vec<Option<usize>> = d.iter().map(|r| r.task).collect();
            Some(evaluate_rows(&scorer, &dev_rows, &truth, dim_h, threads)?)
        }
        None => None,
    };
    let t_dev = t0.elapsed();

    // The manifest of the new skill.
    let mut recipe = Recipe::standard(opts.k as u64);
    recipe.k_source.clone_from(&opts.k_source);
    let manifest = SkillManifest {
        schema: manifest::SKILL_SCHEMA.into(),
        id: opts.skill.clone(),
        taxonomy_version: 1,
        representation_id: builder.representation_id().to_string(),
        recipe,
        labels: data.labels.clone(),
        tasks,
        gate,
        rubric: data.rubric.clone(),
        data: DataRecord {
            train: TrainRecord {
                n: data.train.len() as u64,
                sha256: data.train_sha256.clone(),
                parts: data
                    .train_parts
                    .iter()
                    .map(|p| InputPart {
                        name: p.name.clone(),
                        n: p.train as u64,
                        sha256: p.sha256.clone(),
                    })
                    .collect(),
            },
            calibration: CalibrationRecord {
                n: data.calibration.len() as u64,
                sha256: data.calibration_sha256().to_string(),
                source: match data.calibration_source {
                    CalibrationSource::File { .. } => manifest::CALIBRATION_FROM_FILE.into(),
                    CalibrationSource::CarveOut => manifest::CALIBRATION_CARVE_OUT.into(),
                },
            },
            dev: match (&dev, &data.dev_file) {
                (Some(d), Some((_, sha))) => Some(DevRecord {
                    n: d.n as u64,
                    correct: d.correct as u64,
                    sha256: sha.clone(),
                }),
                _ => None,
            },
            halves_rule: certify::HALVES_RULE.into(),
            holdout_rule: manifest::HOLDOUT_RULE.into(),
        },
        rows: RowsRecord::default(),
        rows_learned: None,
        learned: None,
    };

    // 6. Write the staging file.
    let t0 = Instant::now();
    let sealed = builder
        .add_skill(NewSkill {
            manifest,
            topologies,
            rows: rows_blob,
            rows_learned: None,
        })?
        .clone();
    let skill_manifest_sha256 = manifest::sha256_hex(&crate::canonical::vec_of(&sealed)?);
    let skills: Vec<String> = builder.skill_ids().iter().map(|s| s.to_string()).collect();
    check_output(out)?;
    let staging = staging_path(out);
    // The staging name is ours from here on: removed whatever happens.
    let _staging_guard = RemoveOnDrop(Some(staging.clone()));
    let staged = builder.write(&staging)?;
    let t_write = t0.elapsed();

    // 7. Self-check on the staging file.
    let t0 = Instant::now();
    let cal_texts: Vec<&str> = data.calibration.iter().map(|r| r.text.as_str()).collect();
    let check = self_check(
        &staging,
        &opts.skill,
        &cal_errors,
        &expected_gates,
        scorer.gate(),
        &cal_texts,
        threads,
    )?;
    let t_self = t0.elapsed();

    // 8. Publish.
    publish(&staging, out)?;
    let (sha, bytes) = file_sha256(out)?;
    ensure!(
        sha == staged.sha256 && bytes == staged.bytes,
        "{} changed while it was published",
        out.display()
    );
    let reopened = DecisionModel::open(out, Verify::Light)
        .map_err(|e| anyhow::anyhow!("re-open {}: {e}", out.display()))?;
    ensure!(
        reopened.model_sha() == staged.model_sha,
        "{} re-opens with another model_sha",
        out.display()
    );
    drop(reopened);
    let mut warnings = data.report.warnings.clone();
    if !cert.certified {
        warnings.push(format!(
            "skill '{}': no threshold qualifies (≥ {} accepted odd-half rows with a Clopper–Pearson bound ≥ {}): the gate is not certified (tau = 0, theta only)",
            opts.skill,
            certify::MIN_ACCEPTED,
            certify::TARGET
        ));
    }
    Ok(BuildReport {
        out: WriteReport {
            path: out.to_path_buf(),
            ..staged
        },
        skill: opts.skill.clone(),
        skill_manifest_sha256,
        manifest: sealed,
        skills,
        calibration_correct: cert.calibration_correct,
        dev,
        data: data.report,
        self_check: check,
        threads,
        texts_encoded: texts.len(),
        timings: BuildTimings {
            load: t_load,
            encode: t_encode,
            fit: t_fit,
            certify: t_certify,
            dev: t_dev,
            write: t_write,
            self_check: t_self,
            total: t_start.elapsed(),
        },
        warnings,
    })
}

fn load_data(opts: &TrainOptions) -> Result<SkillData> {
    opts.check()?;
    let inputs = DataInputs::read(
        &opts.train,
        opts.calibration.as_deref(),
        opts.dev.as_deref(),
        opts.question.as_deref(),
    )?;
    data::prepare(inputs)
}

/// `cortiq decision train --encoder FROM.cmf …`: the encoder of `encoder_from`
/// (byte for byte) and one new skill, written to `out` (never replaced).
pub fn train(encoder_from: &Path, opts: &TrainOptions, out: &Path) -> Result<BuildReport> {
    let t_start = Instant::now();
    check_output(out)?;
    let data = load_data(opts)?;
    let model =
        DecisionModel::open(encoder_from, Verify::Full).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t_load = t_start.elapsed();
    build(Target::EncoderOf(&model), data, opts, out, t_start, t_load)
}

/// `cortiq decision add-skill IN.cmf …`: every skill of `input` (byte for byte)
/// and one new skill, written to `out` (never replaced).
pub fn add_skill(input: &Path, opts: &TrainOptions, out: &Path) -> Result<BuildReport> {
    let t_start = Instant::now();
    check_output(out)?;
    let data = load_data(opts)?;
    let model = DecisionModel::open(input, Verify::Full).map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        model.skill(&opts.skill).is_none(),
        "skill '{}' already exists in {} (skills: {})",
        opts.skill,
        input.display(),
        model
            .skills()
            .iter()
            .map(|s| s.id())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let t_load = t_start.elapsed();
    build(Target::AddTo(&model), data, opts, out, t_start, t_load)
}

/// `cortiq decision init --encoder-dir DIR -o OUT`: an encoder-only decision
/// file from an export directory (`tools/decision_export_encoder.py`), with the
/// φ_P of the default golden texts. Like `train` and `add-skill`, the file is
/// written to a staging path next to `out`, opened there with full
/// verification and golden-checked, and only then published without clobber:
/// a file that fails its golden check is never published.
pub fn init_encoder(
    export_dir: &Path,
    out: &Path,
    created_unix: Option<u64>,
) -> Result<WriteReport> {
    check_output(out)?;
    let ex = EncoderExport::read(export_dir)?;
    let init = ex.init_tensors(&DEFAULT_ENCODER_GOLDEN_TEXTS)?;
    let mut b = FileBuilder::new(DEFAULT_MODEL_ID, DEFAULT_NAME, init.record, init.tensors)?;
    if let Some(t) = created_unix {
        b.set_created_unix(t);
    }
    check_output(out)?;
    let staging = staging_path(out);
    // The staging name is ours from here on: removed whatever happens.
    let _staging_guard = RemoveOnDrop(Some(staging.clone()));
    let staged = b.write(&staging)?;
    {
        let model =
            DecisionModel::open(&staging, Verify::Full).map_err(|e| anyhow::anyhow!("{e}"))?;
        SignalEncoder::from_model(&model)
            .with_context(|| format!("golden check of {}", out.display()))?;
    }
    publish(&staging, out)?;
    let (sha, bytes) = file_sha256(out)?;
    ensure!(
        sha == staged.sha256 && bytes == staged.bytes,
        "{} changed while it was published",
        out.display()
    );
    let reopened = DecisionModel::open(out, Verify::Light)
        .map_err(|e| anyhow::anyhow!("re-open {}: {e}", out.display()))?;
    ensure!(
        reopened.model_sha() == staged.model_sha,
        "{} re-opens with another model_sha",
        out.display()
    );
    Ok(WriteReport {
        path: out.to_path_buf(),
        ..staged
    })
}
