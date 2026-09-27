//! Self-learning with recertification and cold start (spec §5.7–§5.9), and the
//! offline pre-training `cortiq decision learn` (spec §5.14).
//!
//! **Rows of a task.** The topology of task i is always the fit (spec §3.4) of,
//! in this order: its train rows (base blob order), its learned rows in the base
//! blob (split 2), its rows in `rows.learned` (the served generation), then the
//! pending examples of its label (buffer order). An example is *pending* when no
//! served learned row of the task has bit-identical φ_P; after a rollback the
//! examples of the discarded generations are pending again (the buffer is kept).
//!
//! **An attempt** ([`attempt`], one label of one skill, when its counter reaches
//! `learning.refit_min_new` = 25):
//! 1. *challenger*: task i refitted on its rows plus the pending examples
//!    (calibration never enters a fit); a label without an active task is an
//!    *activation* (an inactive data task) or a *cold start* (a new task
//!    `origin=cold_start` appended after the others), both `taxonomy_version+1`;
//! 2. *holdout* (calibration rows flagged holdout, spec §3.2): the winner by the
//!    argmin over every active task; the accuracy of the task and the macro
//!    accuracy over the labels with holdout rows must satisfy `chall + 1e-4 ≥
//!    champ` (a label without holdout rows is not checked). A refit or an
//!    activation (the label is already in the skill, its rows may be in the
//!    holdout) is refused on regression; a cold start records the numbers only
//!    (spec §5.8 names no holdout gate for a new label, as cortiq-router's
//!    `maybe_promote_cold_task`; its own label has no holdout row);
//! 3. *recertification* (spec §5.9): `T`, `θ`, `τ` by the build's procedure
//!    (spec §3.6) on the skill's calibration rows; the errors of the unchanged
//!    tasks are reused (their topologies did not change) and only the new
//!    task's column is computed. A certified champion whose recertified gate has
//!    no qualifying `τ` keeps its place (the challenger is refused; a cold-start
//!    label stays in quarantine until the next +25);
//! 4. *promotion*: the generation is written and fsynced, opened with the overlay
//!    loader, checked for *isolation* (the `mean`/`basis` sha256 and `k` of every
//!    other task of every skill are unchanged, else the promotion is cancelled),
//!    made `CURRENT`, and swapped into the model handle. The skill's book then
//!    takes the manifest the new generation serves (with the writer's tensor
//!    sha256), so the next attempt of any label of the skill builds on it.
//!
//! Every attempt (promoted, refused or skipped) resets the label's counter
//! (cortiq-router `selflearn.rs:75-86`) and is logged in `learn.log`.
//!
//! **Offline** ([`learn_offline`], spec §5.14): every text of a traffic file is
//! decided by the input model; accepted texts are left alone (the oracle is never
//! asked about them); for an abstention the answer is taken from the answer
//! ledgers by the sha256 of the request body, else from a live call (live calls
//! enabled without a key in the environment are refused up front; `calls`
//! counts the calls sent, refusals are reported apart). Answers
//! become examples (dedup 0.995). After the pass every label with a new example
//! (label order) gets a challenger with the holdout gate against the current
//! champion; the promotions accumulate. The gate is then recertified once; a
//! certified gate that loses its qualifying `τ` undoes every promotion of the
//! skill. The output is a self-contained file: the encoder and the other skills
//! byte for byte, the learned skill with its new topologies, the learned rows in
//! its rows blob (split 2, source oracle) and a `learned` record in its manifest.

use crate::buffer::{AddOutcome, AttemptRecord, Example, LearnLog, LearningBuffer, LogRecord};
use crate::certify::{self, Calibration, Certification};
use crate::config::{LearningConfig, OracleConfig};
use crate::container::{
    DecisionModel, FileBuilder, LearnedRows, NewSkill, OverlayBuilder, Verify, WriteReport,
};
use crate::eval::{self, SkillScorer, resolve_threads};
use crate::fit::{self, TaskFit};
use crate::generation;
use crate::manifest::{
    Gate, GateParams, LearnedRecord, Rubric, SkillManifest, TaskOrigin, TaskRecord, TaskState,
};
use crate::oracle::{self, CallOutcome, Caller, KeyLookup, OracleClient};
use crate::protocol::{Question, QuestionKind};
use crate::resonance::{ErrStats, Topology, decide};
use crate::rows::{Row, Rows, Source, Split};
use crate::service::ModelHandle;
use crate::signal::SignalEncoder;
use crate::statedir::StateDir;
use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Tolerance of the holdout comparison (cortiq-router `selflearn.rs:13`).
pub const REGRESSION_EPS: f64 = 1e-4;

// ------------------------------------------------------------------ kinds

/// What a challenger changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChangeKind {
    /// An active task refitted.
    Refit,
    /// A task of the skill that was not scored (inactive or quarantined) fitted
    /// and activated.
    Activate,
    /// A new task for a label the skill did not have (spec §5.8).
    ColdStart,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Refit => "promote",
            ChangeKind::Activate => "activate",
            ChangeKind::ColdStart => "cold_start",
        }
    }
}

// ------------------------------------------------------------------ calibration state

/// The champion's calibration errors over its candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibState {
    /// Candidate → task index (the active tasks, in task order).
    pub cands: Vec<usize>,
    pub stats: Vec<ErrStats>,
    /// `calibration rows × candidates`, row-major (base blob order).
    pub matrix: Vec<f32>,
}

/// The learning view of one skill of a model.
#[derive(Clone, Debug)]
pub struct SkillBook {
    pub id: String,
    /// sha256 of the served skill manifest the book belongs to.
    pub manifest_sha: String,
    pub manifest: SkillManifest,
    pub dim_p: usize,
    pub dim_h: usize,
    /// The base rows blob (train, calibration, learned split 2); never changes.
    pub base: Arc<Rows>,
    /// Rows of `rows.learned` served now.
    pub gen_learned: Vec<Row>,
    /// Base row indices of the calibration rows, blob order.
    pub cal: Vec<usize>,
    /// Temperature used to rank candidates (the argmin does not depend on it).
    pub temperature: f32,
    calib: Option<CalibState>,
}

/// One challenger (see the module notes).
#[derive(Clone, Debug)]
pub struct Challenger {
    pub label: String,
    pub task: usize,
    pub kind: ChangeKind,
    /// The skill manifest with the task changed (tensor hashes are filled by the
    /// writer; the gate is the champion's until recertified).
    pub manifest: SkillManifest,
    pub topology: Topology,
    /// `rows.learned` of the new generation (every task, task order).
    pub gen_learned: Vec<Row>,
    /// The pending examples of the label, as learned rows of the task.
    pub new_rows: Vec<Row>,
    /// Rows the task was fitted on.
    pub n_fit: usize,
    pub calib: CalibState,
}

/// The holdout comparison of a challenger.
#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutReport {
    pub rows: usize,
    pub labels: usize,
    pub champ_task: Option<f64>,
    pub chall_task: Option<f64>,
    pub champ_macro: Option<f64>,
    pub chall_macro: Option<f64>,
    /// No regression (`chall + 1e-4 ≥ champ` for both measured metrics).
    pub passed: bool,
    /// Whether this attempt is refused on a regression (refits and
    /// activations; a cold start is not gated).
    pub gated: bool,
}

impl HoldoutReport {
    pub fn to_json(&self) -> Value {
        json!({
            "rows": self.rows, "labels": self.labels,
            "champion": {"task": self.champ_task, "macro": self.champ_macro},
            "challenger": {"task": self.chall_task, "macro": self.chall_macro},
            "passed": self.passed, "gated": self.gated, "eps": REGRESSION_EPS,
        })
    }
}

fn gate_json(g: &Gate) -> Value {
    json!({
        "temperature": g.temperature, "novelty_theta": g.novelty_theta,
        "tau": g.tau, "certified": g.certified,
        "odd": {"accepted": g.evidence.odd.accepted, "correct": g.evidence.odd.correct},
    })
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn placeholder_gate() -> GateParams {
    GateParams {
        temperature: 1.0,
        novelty_theta: 1.0,
        tau: 0.0,
        certified: false,
    }
}

impl SkillBook {
    /// The book of `skill` of `model`; `base` reuses decoded base rows.
    pub fn from_model(model: &DecisionModel, skill: &str, base: Option<Arc<Rows>>) -> Result<Self> {
        let s = model
            .skill(skill)
            .ok_or_else(|| anyhow::anyhow!("no skill '{skill}' in this model"))?;
        let base = match base {
            Some(b) => b,
            None => Arc::new(model.rows(skill)?),
        };
        let gen_learned = model
            .rows_learned(skill)?
            .map(|r| r.rows)
            .unwrap_or_default();
        let cal = base
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.split == Split::Calibration)
            .map(|(i, _)| i)
            .collect();
        Ok(Self {
            id: skill.to_string(),
            manifest_sha: s.sha256.clone(),
            manifest: s.manifest.clone(),
            dim_p: base.dim_p,
            dim_h: base.dim_h,
            temperature: s.manifest.gate.params().temperature,
            base,
            gen_learned,
            cal,
            calib: None,
        })
    }

    /// The task of a label.
    pub fn task_of(&self, label: &str) -> Option<usize> {
        self.manifest.task_of(label)
    }

    /// φ_P of every stored row of a task (base rows of every split and
    /// `rows.learned`): the dedup set of its label.
    pub fn stored_phi(&self, task: usize) -> Vec<&[f32]> {
        self.base
            .rows
            .iter()
            .chain(&self.gen_learned)
            .filter(|r| r.task as usize == task)
            .map(|r| r.phi_p.as_slice())
            .collect()
    }

    fn fit_rows(&self, task: usize) -> Vec<&Row> {
        let t = task as u32;
        let train = self
            .base
            .rows
            .iter()
            .filter(|r| r.task == t && r.split == Split::Train);
        let base_learned = self
            .base
            .rows
            .iter()
            .filter(|r| r.task == t && r.split == Split::Learned);
        let generation = self.gen_learned.iter().filter(|r| r.task == t);
        train.chain(base_learned).chain(generation).collect()
    }

    /// The examples not yet among the served learned rows of `task`.
    pub fn pending<'e>(&self, task: Option<usize>, examples: &'e [Example]) -> Vec<&'e Example> {
        let served: HashSet<Vec<u32>> = match task {
            Some(t) => self
                .base
                .rows
                .iter()
                .filter(|r| r.split == Split::Learned)
                .chain(&self.gen_learned)
                .filter(|r| r.task as usize == t)
                .map(|r| bits(&r.phi_p))
                .collect(),
            None => HashSet::new(),
        };
        examples
            .iter()
            .filter(|e| !served.contains(&bits(&e.phi_p)))
            .collect()
    }

    fn cal_rows(&self) -> Vec<&Row> {
        self.cal.iter().map(|&i| &self.base.rows[i]).collect()
    }

    /// The champion's calibration state, computed on first use.
    pub fn calib(&mut self, model: &DecisionModel, threads: usize) -> Result<&CalibState> {
        if self.calib.is_none() {
            let scorer = SkillScorer::from_model(model, &self.id)?;
            let matrix = scorer.error_matrix(&self.cal_rows(), self.dim_h, threads)?;
            self.calib = Some(CalibState {
                cands: scorer.tasks().to_vec(),
                stats: scorer.stats().to_vec(),
                matrix,
            });
        }
        Ok(self.calib.as_ref().expect("just computed"))
    }

    /// The calibration errors of one task (a single-task scorer: the same packed
    /// arithmetic as the service, bit for bit).
    pub fn column(
        &self,
        record: &TaskRecord,
        topology: &Topology,
        threads: usize,
    ) -> Result<Vec<f32>> {
        let mut rec = record.clone();
        rec.state = TaskState::Active;
        let dim = self.dim_p + self.dim_h;
        let scorer = SkillScorer::new(
            &self.id,
            std::slice::from_ref(&rec),
            &[Some(topology.view())],
            placeholder_gate(),
            dim,
        )?;
        scorer.error_matrix(&self.cal_rows(), self.dim_h, threads)
    }

    /// Fit the challenger of `label` from its pending examples.
    pub fn challenger(
        &mut self,
        model: &DecisionModel,
        label: &str,
        pending: &[&Example],
        threads: usize,
    ) -> Result<Challenger> {
        ensure!(!pending.is_empty(), "no pending example of '{label}'");
        let task = self.task_of(label);
        let (task, kind) = match task {
            Some(t) if self.manifest.tasks[t].is_active() => (t, ChangeKind::Refit),
            Some(t) => (t, ChangeKind::Activate),
            None => (self.manifest.tasks.len(), ChangeKind::ColdStart),
        };
        let t32 = task as u32;
        let new_rows: Vec<Row> = pending.iter().map(|e| e.to_row(t32)).collect();
        let mut rows: Vec<&Row> = if kind == ChangeKind::ColdStart {
            Vec::new()
        } else {
            self.fit_rows(task)
        };
        rows.extend(new_rows.iter());
        let n_fit = rows.len();
        ensure!(
            n_fit >= fit::MIN_ROWS_ACTIVE,
            "'{label}' has {n_fit} rows, an active task needs {}",
            fit::MIN_ROWS_ACTIVE
        );
        let k_max = self.manifest.recipe.k_max as usize;
        let f: TaskFit = crate::build::fit_task_rows(&rows, self.dim_h, k_max)
            .with_context(|| format!("fit of '{label}'"))?;
        let mut record = crate::build::task_record(task, label, &f);
        record.origin = match kind {
            ChangeKind::ColdStart => TaskOrigin::ColdStart,
            _ => self.manifest.tasks[task].origin,
        };
        record.state = TaskState::Active;
        let column = self.column(&record, &f.topology, threads)?;

        let mut manifest = self.manifest.clone();
        match kind {
            ChangeKind::ColdStart => {
                manifest.labels.push(label.to_string());
                manifest.tasks.push(record.clone());
                manifest.taxonomy_version += 1;
            }
            ChangeKind::Activate => {
                manifest.tasks[task] = record.clone();
                manifest.taxonomy_version += 1;
            }
            ChangeKind::Refit => manifest.tasks[task] = record.clone(),
        }

        // Challenger calibration state: the champion's columns, the new one.
        let champ = self.calib(model, threads)?.clone();
        let n = self.cal.len();
        let cands: Vec<usize> = manifest
            .tasks
            .iter()
            .filter(|t| t.is_active())
            .map(|t| t.i as usize)
            .collect();
        let stats: Vec<ErrStats> = cands.iter().map(|&t| manifest.tasks[t].stats()).collect();
        let old_pos: HashMap<usize, usize> = champ
            .cands
            .iter()
            .enumerate()
            .map(|(p, &t)| (t, p))
            .collect();
        let m = cands.len();
        let mc = champ.cands.len();
        let mut matrix = vec![0.0f32; n * m];
        for (c, &t) in cands.iter().enumerate() {
            if t == task {
                for j in 0..n {
                    matrix[j * m + c] = column[j];
                }
            } else {
                let p = *old_pos
                    .get(&t)
                    .ok_or_else(|| anyhow::anyhow!("task {t} has no champion column"))?;
                for j in 0..n {
                    matrix[j * m + c] = champ.matrix[j * mc + p];
                }
            }
        }

        let mut gen_learned: Vec<Row> = self.gen_learned.clone();
        gen_learned.extend(new_rows.iter().cloned());
        gen_learned.sort_by_key(|r| r.task);

        Ok(Challenger {
            label: label.to_string(),
            task,
            kind,
            manifest,
            topology: f.topology,
            gen_learned,
            new_rows,
            n_fit,
            calib: CalibState {
                cands,
                stats,
                matrix,
            },
        })
    }

    /// Per task: (correct, total) holdout rows under `state`.
    fn holdout_accuracy(&self, st: &CalibState) -> Result<BTreeMap<usize, (usize, usize)>> {
        let m = st.cands.len();
        let mut acc: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        for (j, &bi) in self.cal.iter().enumerate() {
            let r = &self.base.rows[bi];
            if !r.holdout() {
                continue;
            }
            let d = decide(&st.matrix[j * m..(j + 1) * m], &st.stats, self.temperature)?;
            let win = d.winner.map(|w| st.cands[w]);
            let e = acc.entry(r.task as usize).or_insert((0, 0));
            e.1 += 1;
            e.0 += usize::from(win == Some(r.task as usize));
        }
        Ok(acc)
    }

    /// The holdout comparison of a challenger with the champion.
    pub fn holdout(
        &mut self,
        model: &DecisionModel,
        ch: &Challenger,
        threads: usize,
    ) -> Result<HoldoutReport> {
        let champ_state = self.calib(model, threads)?.clone();
        let champ = self.holdout_accuracy(&champ_state)?;
        let chall = self.holdout_accuracy(&ch.calib)?;
        let frac = |c: usize, n: usize| c as f64 / n as f64;
        let task_acc =
            |a: &BTreeMap<usize, (usize, usize)>| a.get(&ch.task).map(|&(c, n)| frac(c, n));
        let macro_acc = |a: &BTreeMap<usize, (usize, usize)>| {
            (!a.is_empty())
                .then(|| a.values().map(|&(c, n)| frac(c, n)).sum::<f64>() / a.len() as f64)
        };
        let (champ_task, chall_task) = (task_acc(&champ), task_acc(&chall));
        let (champ_macro, chall_macro) = (macro_acc(&champ), macro_acc(&chall));
        let ok = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(c), Some(h)) => h + REGRESSION_EPS >= c,
            _ => true,
        };
        Ok(HoldoutReport {
            rows: champ.values().map(|v| v.1).sum(),
            labels: champ.len(),
            passed: ok(champ_task, chall_task) && ok(champ_macro, chall_macro),
            gated: ch.kind != ChangeKind::ColdStart,
            champ_task,
            chall_task,
            champ_macro,
            chall_macro,
        })
    }

    /// Certify a calibration state (spec §3.6; the recertification of §5.9).
    pub fn certify(&self, st: &CalibState) -> Result<Certification> {
        let mut even = Vec::new();
        let mut odd = Vec::new();
        let mut truth = Vec::with_capacity(self.cal.len());
        let pos: HashMap<usize, usize> =
            st.cands.iter().enumerate().map(|(p, &t)| (t, p)).collect();
        for (j, &bi) in self.cal.iter().enumerate() {
            let r = &self.base.rows[bi];
            if r.odd_half() {
                odd.push(j);
            } else {
                even.push(j);
            }
            truth.push(pos.get(&(r.task as usize)).copied());
        }
        ensure!(
            !st.cands.is_empty(),
            "skill '{}' has no active task",
            self.id
        );
        ensure!(
            even.iter().any(|&j| truth[j].is_some()),
            "skill '{}': no even calibration row has an active label",
            self.id
        );
        certify::certify(&Calibration {
            errors: &st.matrix,
            tasks: st.cands.len(),
            stats: &st.stats,
            truth: &truth,
            even: &even,
            odd: &odd,
        })
    }

    /// Adopt a promoted challenger as the champion of an offline pass (nothing
    /// is served): the challenger's manifest with `gate`. Its changed task keeps
    /// `mean_sha256`/`basis_sha256` unset (the writer fills them), so this book
    /// must never feed an [`OverlayBuilder`]; online promotions use
    /// [`SkillBook::adopt_served`].
    pub fn adopt(&mut self, ch: &Challenger, gate: Gate) {
        self.manifest = ch.manifest.clone();
        self.manifest.gate = gate;
        self.temperature = self.manifest.gate.params().temperature;
        self.gen_learned = ch.gen_learned.clone();
        self.calib = Some(ch.calib.clone());
    }

    /// Adopt a promoted challenger as the champion after its generation was
    /// published: the manifest is the one the new generation serves (the
    /// writer's tensor sha256, the `rows.learned` record, the recertified
    /// gate), so the next challenger of any label of the skill passes the
    /// overlay builder's "unchanged task keeps its record" check; the rows and
    /// the calibration columns are the challenger's (the same bits the served
    /// model gives).
    pub fn adopt_served(&mut self, ch: &Challenger, served: &crate::container::LoadedSkill) {
        debug_assert_eq!(served.manifest.id, self.id);
        self.manifest = served.manifest.clone();
        self.manifest_sha = served.sha256.clone();
        self.temperature = self.manifest.gate.params().temperature;
        self.gen_learned = ch.gen_learned.clone();
        self.calib = Some(ch.calib.clone());
    }
}

/// Differences in the task tensors of every skill between two models, except
/// task `changed` of skill `skill` (a cold start adds exactly that task).
pub fn isolation_violations(
    before: &DecisionModel,
    after: &DecisionModel,
    skill: &str,
    changed: usize,
) -> Vec<String> {
    let mut v = Vec::new();
    for s in before.skills() {
        let Some(a) = after.skill(s.id()) else {
            v.push(format!("skill '{}' disappeared", s.id()));
            continue;
        };
        for t in &s.manifest.tasks {
            if s.id() == skill && t.i as usize == changed {
                continue;
            }
            match a.manifest.tasks.get(t.i as usize) {
                Some(n)
                    if n.k == t.k
                        && n.mean_sha256 == t.mean_sha256
                        && n.basis_sha256 == t.basis_sha256
                        && n.state == t.state => {}
                _ => v.push(format!("skill '{}' task {} changed", s.id(), t.i)),
            }
        }
        let extra = a
            .manifest
            .tasks
            .len()
            .saturating_sub(s.manifest.tasks.len());
        let allowed = usize::from(s.id() == skill && changed >= s.manifest.tasks.len());
        if extra > allowed {
            v.push(format!("skill '{}' gained {extra} tasks", s.id()));
        }
    }
    v
}

// ------------------------------------------------------------------ online attempts

/// The books of the served skills, keyed by skill; a book is valid for the
/// skill manifest sha256 it was built for.
#[derive(Debug, Default)]
pub struct Books {
    books: HashMap<String, SkillBook>,
    bases: HashMap<String, Arc<Rows>>,
}

impl Books {
    /// The book of `skill` for the served `model` (rebuilt when the skill's
    /// manifest changed, e.g. after a rollback).
    pub fn get(
        &mut self,
        model: &DecisionModel,
        skill: &str,
        base: Option<Arc<Rows>>,
    ) -> Result<&mut SkillBook> {
        let sha = model
            .skill(skill)
            .ok_or_else(|| anyhow::anyhow!("no skill '{skill}'"))?
            .sha256
            .clone();
        let valid = self.books.get(skill).is_some_and(|b| b.manifest_sha == sha);
        if !valid {
            let base = base.or_else(|| self.bases.get(skill).cloned());
            let b = SkillBook::from_model(model, skill, base)?;
            self.bases.insert(skill.to_string(), Arc::clone(&b.base));
            self.books.insert(skill.to_string(), b);
        }
        Ok(self.books.get_mut(skill).expect("just inserted"))
    }

    /// Forget every calibration matrix (after a rollback).
    pub fn invalidate(&mut self) {
        self.books.clear();
    }
}

/// What an attempt did.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Promoted { generation: u64, sha256: String },
    Rejected(String),
    Skipped(String),
}

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Promoted { .. } => "promoted",
            Outcome::Rejected(_) => "rejected",
            Outcome::Skipped(_) => "skipped",
        }
    }
}

/// The report of one learning attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct AttemptReport {
    pub skill: String,
    pub label: String,
    pub kind: Option<ChangeKind>,
    pub task: Option<usize>,
    pub outcome: Outcome,
    pub pending: usize,
    pub n_fit: usize,
    pub holdout: Option<HoldoutReport>,
    pub gate_before: Option<Gate>,
    pub gate_after: Option<Gate>,
    pub isolation_violations: usize,
}

impl AttemptReport {
    fn skipped(skill: &str, label: &str, why: impl Into<String>) -> Self {
        Self {
            skill: skill.into(),
            label: label.into(),
            kind: None,
            task: None,
            outcome: Outcome::Skipped(why.into()),
            pending: 0,
            n_fit: 0,
            holdout: None,
            gate_before: None,
            gate_after: None,
            isolation_violations: 0,
        }
    }

    pub fn to_json(&self) -> Value {
        let (generation, sha256, reason) = match &self.outcome {
            Outcome::Promoted { generation, sha256 } => {
                (Some(*generation), Some(sha256.clone()), None)
            }
            Outcome::Rejected(r) | Outcome::Skipped(r) => (None, None, Some(r.clone())),
        };
        json!({
            "skill": self.skill, "label": self.label,
            "kind": self.kind.map(ChangeKind::as_str), "task": self.task,
            "outcome": self.outcome.as_str(), "reason": reason,
            "generation": generation, "sha256": sha256,
            "pending": self.pending, "n_fit": self.n_fit,
            "holdout": self.holdout.as_ref().map(HoldoutReport::to_json),
            "gate_before": self.gate_before.as_ref().map(gate_json),
            "gate_after": self.gate_after.as_ref().map(gate_json),
            "isolation_violations": self.isolation_violations,
        })
    }
}

/// Everything an online attempt touches.
pub struct LearnContext<'a> {
    pub handle: &'a ModelHandle,
    pub state: &'a StateDir,
    pub buffer: &'a Mutex<LearningBuffer>,
    pub log: &'a LearnLog,
    pub books: &'a Mutex<Books>,
    pub cfg: &'a LearningConfig,
    /// The decoded base rows of a skill (shared with the dedup path).
    pub base_rows: &'a (dyn Fn(&DecisionModel, &str) -> Result<Arc<Rows>> + Sync),
    pub threads: usize,
    /// `created_unix` of new generations (`None`: `SOURCE_DATE_EPOCH` or 0).
    pub created_unix: Option<u64>,
}

/// Run one attempt for (skill, label) now (see the module notes). The caller
/// serialises attempts.
pub fn attempt(ctx: &LearnContext<'_>, skill: &str, label: &str) -> Result<AttemptReport> {
    let report = attempt_inner(ctx, skill, label);
    let (outcome, generation) = match &report {
        Ok(r) => (
            r.outcome.as_str().to_string(),
            match &r.outcome {
                Outcome::Promoted { generation, .. } => *generation,
                _ => ctx.handle.current().generation(),
            },
        ),
        Err(_) => ("error".to_string(), ctx.handle.current().generation()),
    };
    ctx.buffer.lock().reset(skill, label);
    ctx.log.append(&LogRecord::Attempt(AttemptRecord {
        skill: skill.into(),
        label: label.into(),
        outcome,
        generation,
    }))?;
    report
}

fn attempt_inner(ctx: &LearnContext<'_>, skill: &str, label: &str) -> Result<AttemptReport> {
    let threads = resolve_threads(ctx.threads);
    let loaded = ctx.handle.current();
    let model = loaded.model();
    let Some(sk) = model.skill(skill) else {
        return Ok(AttemptReport::skipped(
            skill,
            label,
            "the skill is not served",
        ));
    };
    let is_active = sk
        .manifest
        .task_of(label)
        .is_some_and(|t| sk.manifest.tasks[t].is_active());
    if !is_active && !ctx.cfg.cold_start {
        return Ok(AttemptReport::skipped(
            skill,
            label,
            "cold start is disabled",
        ));
    }
    let examples: Vec<Example> = ctx
        .buffer
        .lock()
        .examples(skill, label)
        .into_iter()
        .cloned()
        .collect();
    let base = (ctx.base_rows)(model, skill)?;
    let mut books = ctx.books.lock();
    let book = books.get(model, skill, Some(base))?;
    let task = book.task_of(label);
    let pending = book.pending(task, &examples);
    if pending.is_empty() {
        return Ok(AttemptReport::skipped(skill, label, "no pending example"));
    }
    let n_pending = pending.len();
    let gate_before = book.manifest.gate.clone();
    let ch = match book.challenger(model, label, &pending, threads) {
        Ok(c) => c,
        Err(e) => {
            let mut r = AttemptReport::skipped(skill, label, format!("{e:#}"));
            r.pending = n_pending;
            return Ok(r);
        }
    };
    let mut report = AttemptReport {
        skill: skill.into(),
        label: label.into(),
        kind: Some(ch.kind),
        task: Some(ch.task),
        outcome: Outcome::Skipped(String::new()),
        pending: n_pending,
        n_fit: ch.n_fit,
        holdout: None,
        gate_before: Some(gate_before.clone()),
        gate_after: None,
        isolation_violations: 0,
    };
    let holdout = book.holdout(model, &ch, threads)?;
    report.holdout = Some(holdout.clone());
    if holdout.gated && !holdout.passed {
        report.outcome = Outcome::Rejected("holdout_regression".into());
        return Ok(report);
    }
    let cert = book.certify(&ch.calib)?;
    let gate = Gate::from_certification(&cert);
    report.gate_after = Some(gate.clone());
    if gate_before.certified && !cert.certified {
        report.outcome = Outcome::Rejected("gate_lost".into());
        return Ok(report);
    }

    // Write, check isolation, make CURRENT, swap.
    let g = generation::next_generation(ctx.state, model.generation())?;
    let mut b = OverlayBuilder::new(model, g)?;
    if let Some(t) = ctx.created_unix {
        b.set_created_unix(t);
    }
    b.push_event(
        skill,
        label,
        ch.kind.as_str(),
        holdout.to_json(),
        json!({"before": gate_json(&gate_before), "after": gate_json(&gate)}),
    );
    let mut manifest = ch.manifest.clone();
    manifest.gate = gate.clone();
    let learned = Rows {
        dim_p: book.dim_p,
        dim_h: book.dim_h,
        rows: ch.gen_learned.clone(),
    }
    .encode()?;
    b.set_skill(
        manifest,
        BTreeMap::from([(ch.task, Some(ch.topology.clone()))]),
        LearnedRows::Replace(learned),
    )?;
    let mut violations = Vec::new();
    let published = generation::publish(ctx.state, model.base_path(), &b, g, |m| {
        violations = isolation_violations(model, m, skill, ch.task);
        if violations.is_empty() {
            Ok(())
        } else {
            bail!("isolation violated: {}", violations.join("; "))
        }
    });
    let published = match published {
        Ok(p) => p,
        Err(e) if !violations.is_empty() => {
            tracing::error!(error = %e, "promotion cancelled");
            report.isolation_violations = violations.len();
            report.outcome = Outcome::Rejected("isolation".into());
            return Ok(report);
        }
        Err(e) => return Err(e),
    };
    let served_skill = published
        .model
        .skill(skill)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("the generation lost skill '{skill}'"))?;
    let sha256 = published.report.sha256.clone();
    let next = loaded.derive(published.model)?;
    book.adopt_served(&ch, &served_skill);
    drop(books);
    ctx.handle.promote(next);
    // A label may come from client feedback (a cold start): the line names
    // its hash and task index only (spec §4.3).
    tracing::info!(
        skill,
        task = ch.task,
        label_sha = %label_tag(label),
        generation = g,
        kind = ch.kind.as_str(),
        "promoted"
    );
    report.outcome = Outcome::Promoted {
        generation: g,
        sha256,
    };
    Ok(report)
}

/// What a log line shows of a label: the first 12 hex characters of its
/// SHA-256 (a label may be free text of a client's feedback, spec §4.3).
pub fn label_tag(label: &str) -> String {
    sha256_hex(label.as_bytes())[..12].to_string()
}

/// `text` with every `'{label}'` — the form in which this crate's messages
/// name a label — replaced by `label#{tag}` ([`label_tag`]), for log lines.
pub fn redact_label(text: &str, label: &str) -> String {
    if label.is_empty() {
        return text.to_string();
    }
    text.replace(
        &format!("'{label}'"),
        &format!("label#{}", label_tag(label)),
    )
}

// ------------------------------------------------------------------ offline

/// Options of `cortiq decision learn` (spec §5.14).
#[derive(Clone, Debug)]
pub struct OfflineOptions {
    /// The skill (default: the only skill of the file).
    pub skill: Option<String>,
    /// JSONL `{"text"}` (a `label` is allowed and ignored).
    pub traffic: PathBuf,
    /// Answer ledgers of the v4 driver, reused by the sha256 of the body.
    pub answers: Vec<PathBuf>,
    /// The oracle section of `--oracle-config` (body, budget, stop rules).
    pub oracle: OracleConfig,
    /// Reservation ledger of live calls (required when live calls are enabled).
    pub ledger: Option<PathBuf>,
    /// cos φ_P of a duplicate example.
    pub dedup: f32,
    pub threads: usize,
    /// `created_unix` of the output (`None`: `SOURCE_DATE_EPOCH` or 0).
    pub created_unix: Option<u64>,
}

impl OfflineOptions {
    pub fn new(traffic: impl Into<PathBuf>, oracle: OracleConfig) -> Self {
        Self {
            skill: None,
            traffic: traffic.into(),
            answers: Vec::new(),
            oracle,
            ledger: None,
            dedup: LearningConfig::default().dedup,
            threads: 0,
            created_unix: None,
        }
    }
}

/// One label of an offline pass.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineLabel {
    pub label: String,
    pub examples: usize,
    pub outcome: String,
    pub reason: Option<String>,
    pub holdout: Option<HoldoutReport>,
}

/// What `cortiq decision learn` did.
#[derive(Clone, Debug)]
pub struct OfflineReport {
    pub skill: String,
    pub traffic_sha256: String,
    pub texts: usize,
    /// Rows of the traffic that carried a label (ignored).
    pub labelled: usize,
    pub accepted: usize,
    pub abstained: usize,
    pub answers_reused: usize,
    /// Calls sent to the oracle (answered or failed); `learned.calls`.
    pub live_calls: usize,
    /// Sent calls that failed (HTTP error, transport, invalid answer, …).
    pub failed_calls: usize,
    /// Calls not sent, by reason (`budget`, `stopped`, `oracle_disabled`,
    /// `ledger_write`); their texts are unanswered.
    pub refused_calls: BTreeMap<String, usize>,
    pub unanswered: usize,
    pub pii_redacted: usize,
    pub examples: usize,
    pub duplicates: usize,
    pub labels: Vec<OfflineLabel>,
    pub promoted_labels: Vec<String>,
    pub rejected_labels: Vec<String>,
    /// Every promotion was undone because the certified gate was lost.
    pub rolled_back: bool,
    pub gate_before: Gate,
    pub gate_after: Gate,
    pub oracle_spent_usd: f64,
    pub out: WriteReport,
}

impl OfflineReport {
    pub fn to_json(&self) -> Value {
        json!({
            "skill": self.skill, "traffic_sha256": self.traffic_sha256, "texts": self.texts,
            "labelled": self.labelled, "accepted": self.accepted, "abstained": self.abstained,
            "answers_reused": self.answers_reused, "live_calls": self.live_calls,
            "failed_calls": self.failed_calls, "refused_calls": self.refused_calls,
            "unanswered": self.unanswered, "pii_redacted": self.pii_redacted,
            "examples": self.examples, "duplicates": self.duplicates,
            "labels": self.labels.iter().map(|l| json!({
                "label": l.label, "examples": l.examples, "outcome": l.outcome,
                "reason": l.reason, "holdout": l.holdout.as_ref().map(HoldoutReport::to_json),
            })).collect::<Vec<_>>(),
            "promoted_labels": self.promoted_labels, "rejected_labels": self.rejected_labels,
            "rolled_back": self.rolled_back,
            "gate_before": gate_json(&self.gate_before), "gate_after": gate_json(&self.gate_after),
            "oracle_spent_usd": self.oracle_spent_usd,
            "out": {"path": self.out.path, "sha256": self.out.sha256, "bytes": self.out.bytes,
                    "model_sha": self.out.model_sha},
        })
    }
}

/// The oracle's question of a skill: `task`, choice, the rubric's instructions
/// and criteria in the question file's order (the v4 driver's request).
pub fn rubric_question(m: &SkillManifest) -> Result<Question> {
    let r = m.rubric.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "skill '{}' has no rubric (train it with --question): the oracle needs instructions and criteria",
            m.id
        )
    })?;
    Ok(question_of_rubric(r))
}

/// The oracle's `task` question of a rubric (criteria in the question file's
/// order, which is the order of the schema enum).
pub fn question_of_rubric(r: &Rubric) -> Question {
    Question {
        id: "task".into(),
        kind: QuestionKind::Choice,
        instructions: Value::String(r.instructions.clone()),
        criteria: Some(Value::Object(r.ordered_criteria())),
    }
}

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// `cortiq decision learn IN.cmf --traffic T.jsonl …` (see the module notes).
pub fn learn_offline(
    input: &Path,
    opts: &OfflineOptions,
    key: KeyLookup,
    out: &Path,
) -> Result<OfflineReport> {
    let threads = resolve_threads(opts.threads);
    ensure!(
        !out.exists(),
        "refusing to overwrite existing output {}",
        out.display()
    );
    let model = DecisionModel::open(input, Verify::Full)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", input.display()))?;
    let skill = eval::select_skill(&model, opts.skill.as_deref())?;
    let manifest_in = model
        .skill(&skill)
        .expect("selected skills exist")
        .manifest
        .clone();
    let question = rubric_question(&manifest_in)?;

    let traffic_bytes =
        std::fs::read(&opts.traffic).with_context(|| format!("read {}", opts.traffic.display()))?;
    let traffic_sha256 = sha256_hex(&traffic_bytes);
    let inputs = eval::parse_input(&opts.traffic.display().to_string(), &traffic_bytes)?;
    let labelled = inputs.iter().filter(|r| r.label.is_some()).count();
    let answers = oracle::read_answer_ledgers(&opts.answers)?;
    let client = match (&opts.ledger, opts.oracle.enabled) {
        (Some(l), true) => {
            // Refused up front (before the ledger is created): a run that was
            // configured for live calls must not quietly learn from the
            // ledgers alone. Only the presence of the key is checked.
            ensure!(
                key(&opts.oracle.api_key_env).is_some(),
                "oracle.enabled is true but the environment variable {} holds no key: set it, \
                 or set oracle.enabled=false to learn from the --answers ledgers only",
                opts.oracle.api_key_env
            );
            Some(OracleClient::open(&opts.oracle, l, None, key)?)
        }
        (None, true) => bail!("live oracle calls need a reservation ledger path"),
        (_, false) => None,
    };

    let (encoder, _) = SignalEncoder::from_model(&model)?;
    let scorer = SkillScorer::from_model(&model, &skill)?;
    let mut book = SkillBook::from_model(&model, &skill, None)?;
    let mut buffer = LearningBuffer::new(opts.dedup);
    let caller_id = format!("learn-{}", &traffic_sha256[..12]);
    let options = question.options();
    let mut asked: HashMap<String, Option<String>> = HashMap::new();
    let (mut accepted, mut abstained, mut reused, mut live, mut unanswered, mut redacted) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut failed = 0usize;
    // Calls that were not sent, by reason (`budget`, `stopped`, …).
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let (mut examples, mut duplicates) = (0usize, 0usize);
    for chunk in inputs.chunks(crate::build::ENCODE_CHUNK) {
        let texts: Vec<&str> = chunk.iter().map(|r| r.text.as_str()).collect();
        let feats = encoder.features_batch(&texts, threads);
        for (text, f) in texts.iter().zip(&feats) {
            let (_, d) = scorer.decide(&f.signal())?;
            if scorer.accepted(&d) {
                accepted += 1;
                continue;
            }
            abstained += 1;
            let raw = Value::String((*text).to_string());
            let state = if opts.oracle.redact_pii {
                let (v, c) = crate::pii::redact_value(&raw);
                redacted += usize::from(c);
                v
            } else {
                raw
            };
            let body = oracle::request_body(&opts.oracle, &[&question], &state);
            let sha = sha256_hex(&body);
            let answer = if let Some(a) = asked.get(&sha) {
                a.clone()
            } else {
                let a = if let Some(c) = answers.get(&sha) {
                    reused += 1;
                    Some(c.clone())
                } else if let Some(cl) = &client {
                    let caller = Caller {
                        request_id: &caller_id,
                        account: "decision-learn",
                        key12: None,
                        key_budget_usd: None,
                        credit_left_usd: None,
                    };
                    // A live call is one that was sent (answered or failed);
                    // refusals (budget, stop rule, …) are counted apart.
                    match cl.call_body(&caller, &[&question], &body) {
                        CallOutcome::Answered(a) => {
                            live += 1;
                            match a.verdicts.first() {
                                Some(crate::answer::OracleAnswer::Choice(c)) => Some(c.clone()),
                                _ => None,
                            }
                        }
                        CallOutcome::Failed(f) if f.call_id.is_some() => {
                            live += 1;
                            failed += 1;
                            None
                        }
                        CallOutcome::Failed(f) => {
                            *refusals.entry(f.error).or_default() += 1;
                            None
                        }
                        CallOutcome::Refused(r) => {
                            *refusals.entry(r.flag().to_string()).or_default() += 1;
                            None
                        }
                    }
                } else {
                    None
                };
                asked.insert(sha, a.clone());
                a
            };
            let Some(label) = answer.filter(|l| options.contains(&l.as_str())) else {
                unanswered += 1;
                continue;
            };
            let ex = Example::from_features(&skill, &label, Source::Oracle, f, 0);
            let stored = match book.task_of(&label) {
                Some(t) => book.stored_phi(t),
                None => Vec::new(),
            };
            match buffer.add(ex, &stored) {
                AddOutcome::Stored => examples += 1,
                AddOutcome::Duplicate | AddOutcome::Full => duplicates += 1,
            }
        }
    }

    // Challengers in label order, promotions accumulate.
    let mut order: Vec<String> = manifest_in.labels.clone();
    let mut extra: Vec<String> = buffer
        .labels()
        .into_iter()
        .map(|l| l.label)
        .filter(|l| !order.contains(l))
        .collect();
    extra.sort();
    order.extend(extra);
    let mut labels_out = Vec::new();
    let mut promoted: Vec<Challenger> = Vec::new();
    for label in order {
        let exs: Vec<Example> = buffer
            .examples(&skill, &label)
            .into_iter()
            .cloned()
            .collect();
        if exs.is_empty() {
            continue;
        }
        let pending = book.pending(book.task_of(&label), &exs);
        if pending.is_empty() {
            continue;
        }
        let mut entry = OfflineLabel {
            label: label.clone(),
            examples: pending.len(),
            outcome: String::new(),
            reason: None,
            holdout: None,
        };
        let ch = match book.challenger(&model, &label, &pending, threads) {
            Ok(c) => c,
            Err(e) => {
                entry.outcome = "skipped".into();
                entry.reason = Some(format!("{e:#}"));
                labels_out.push(entry);
                continue;
            }
        };
        let h = book.holdout(&model, &ch, threads)?;
        entry.holdout = Some(h.clone());
        if h.passed {
            entry.outcome = "promoted".into();
            let gate = book.manifest.gate.clone();
            book.adopt(&ch, gate);
            promoted.push(ch);
        } else {
            entry.outcome = "rejected".into();
            entry.reason = Some("holdout_regression".into());
        }
        labels_out.push(entry);
    }

    let gate_before = manifest_in.gate.clone();
    let mut rolled_back = false;
    let mut gate_after = gate_before.clone();
    if !promoted.is_empty() {
        let st = book.calib(&model, threads)?.clone();
        let cert = book.certify(&st)?;
        if gate_before.certified && !cert.certified {
            rolled_back = true;
            for l in labels_out.iter_mut().filter(|l| l.outcome == "promoted") {
                l.outcome = "rejected".into();
                l.reason = Some("gate_lost".into());
            }
            promoted.clear();
        } else {
            gate_after = Gate::from_certification(&cert);
        }
    }
    let promoted_labels: Vec<String> = promoted.iter().map(|c| c.label.clone()).collect();
    let rejected_labels: Vec<String> = labels_out
        .iter()
        .filter(|l| l.outcome == "rejected")
        .map(|l| l.label.clone())
        .collect();

    // The output file.
    let mut manifest = if rolled_back || promoted.is_empty() {
        manifest_in.clone()
    } else {
        let mut m = book.manifest.clone();
        m.gate = gate_after.clone();
        m
    };
    let oracle_spent = client.as_ref().map_or(0.0, |c| c.totals().spent);
    manifest.learned = Some(LearnedRecord {
        traffic_sha256: traffic_sha256.clone(),
        oracle_model: opts.oracle.model.clone(),
        calls: live as u64,
        answers_reused: reused as u64,
        promoted_labels: promoted_labels.clone(),
        rejected_labels: rejected_labels.clone(),
        gate_before: gate_before.clone(),
        gate_after: gate_after.clone(),
    });
    let changed: BTreeMap<usize, &Challenger> = promoted.iter().map(|c| (c.task, c)).collect();
    let mut builder = FileBuilder::encoder_of(&model);
    if let Some(t) = opts.created_unix {
        builder.set_created_unix(t);
    }
    for s in model.skills() {
        let id = s.id();
        if id == skill {
            let mut topologies = Vec::with_capacity(manifest.tasks.len());
            for t in &manifest.tasks {
                let i = t.i as usize;
                topologies.push(match changed.get(&i) {
                    Some(c) => Some(c.topology.clone()),
                    None if t.mean_sha256.is_some() => model.topology(id, i)?,
                    None => None,
                });
            }
            // Rows: the base blob, the input's rows.learned (merged, so the fit
            // order of every task stays train ++ learned), then the new rows.
            let mut rows = (*book.base).clone();
            let merged = model.rows_learned(id)?.map(|r| r.rows).unwrap_or_default();
            rows.rows.extend(merged);
            for c in changed.values() {
                rows.rows.extend(c.new_rows.iter().cloned());
            }
            manifest.rows_learned = None;
            builder.add_skill(NewSkill {
                manifest: manifest.clone(),
                topologies,
                rows: rows.encode()?,
                rows_learned: None,
            })?;
        } else {
            let m = s.manifest.clone();
            let mut topologies = Vec::with_capacity(m.tasks.len());
            for t in &m.tasks {
                topologies.push(if t.mean_sha256.is_some() {
                    model.topology(id, t.i as usize)?
                } else {
                    None
                });
            }
            let rows = model.base().tensor_bytes(&m.rows.tensor)?.to_vec();
            let rows_learned = match &m.rows_learned {
                Some(r) => Some(model.tensor_bytes(&r.tensor)?.to_vec()),
                None => None,
            };
            builder.add_skill(NewSkill {
                manifest: m,
                topologies,
                rows,
                rows_learned,
            })?;
        }
    }
    let report = builder.write(out)?;

    // Isolation: the other skills byte for byte, the other tasks unchanged.
    let written = DecisionModel::open(out, Verify::Light)
        .map_err(|e| anyhow::anyhow!("re-open {}: {e}", out.display()))?;
    for s in model.skills().iter().filter(|s| s.id() != skill) {
        let w = written
            .skill(s.id())
            .ok_or_else(|| anyhow::anyhow!("skill '{}' missing in the output", s.id()))?;
        ensure!(
            w.sha256 == s.sha256,
            "isolation: skill '{}' changed its manifest",
            s.id()
        );
    }
    for t in &manifest_in.tasks {
        let i = t.i as usize;
        if changed.contains_key(&i) {
            continue;
        }
        let n = &written.skill(&skill).expect("written").manifest.tasks[i];
        ensure!(
            n.mean_sha256 == t.mean_sha256 && n.basis_sha256 == t.basis_sha256 && n.k == t.k,
            "isolation: task {i} of '{skill}' changed"
        );
    }

    Ok(OfflineReport {
        skill,
        traffic_sha256,
        texts: inputs.len(),
        labelled,
        accepted,
        abstained,
        answers_reused: reused,
        live_calls: live,
        failed_calls: failed,
        refused_calls: refusals,
        unanswered,
        pii_redacted: redacted,
        examples,
        duplicates,
        labels: labels_out,
        promoted_labels,
        rejected_labels,
        rolled_back,
        gate_before,
        gate_after,
        oracle_spent_usd: oracle_spent,
        out: report,
    })
}

/// A model and its learning view in one place for tests and tools: the
/// calibration state of a served skill recomputed from scratch.
pub fn calib_from_model(model: &DecisionModel, skill: &str, threads: usize) -> Result<CalibState> {
    let mut b = SkillBook::from_model(model, skill, None)?;
    Ok(b.calib(model, threads)?.clone())
}
