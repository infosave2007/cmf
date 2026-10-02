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
//! **Auto-skills** ([`attempt_auto`], 0.8.6, DESIGN D5/D6/A2/A10): an attempt of
//! an `auto-` id (an untrained choice contract learned from oracle answers,
//! [`crate::cascade`]) fits the WHOLE skill at once, with no build rows:
//! 1. R = the served `rows.learned` ∪ the pending examples of every label of
//!    the contract; each row is in the calibration subset C iff
//!    [`certify::auto_row_key`] says so (a property of the row: it never moves
//!    between the fit and C), the rest fit; eligible labels have at least
//!    `learning.auto_min_rows` fit rows (an already active one at least 2);
//!    the attempt needs ≥ 2 eligible labels including the triggering one, and
//!    — for the first activation — the eligible labels must hold
//!    `learning.auto_min_coverage` of the contract's rows (so that the labels
//!    left quarantined are genuinely rare); else `skipped`;
//! 2. *challenger*: every eligible label refitted (`K = min(auto_k, n − 1)`,
//!    origin `cold_start`, state `active`), the others quarantined without a
//!    topology; `taxonomy_version + 1` when the active label set changes;
//! 3. *gate*: C of the eligible labels only (`|C| ≥ 2`, else skipped), halves by
//!    [`certify::halves`] over the same keys, [`certify::certify`] as a build
//!    (τ certifies only with 100 accepted odd rows, so the gate is θ-only for
//!    a long time; serving adds the `learning.auto_tau` floor); the first
//!    activation (`auto_start`) needs a macro agreement with the oracle's
//!    labels on C of `learning.auto_min_agreement` and ≥ 0.5 per label, else
//!    `rejected: auto_agreement`; later (`auto_refit`) the champion (served
//!    topologies) and the challenger are scored on the same C and the holdout
//!    rule applies (`chall + 1e-4 ≥ champ`, task and macro accuracy);
//!    `gate_lost` never applies (never certified while young);
//! 4. *promotion*: the generation carries the auto-skill fully (born with
//!    [`OverlayBuilder::add_skill`] when the model does not serve it, changed
//!    with [`OverlayBuilder::set_skill`] otherwise, `rows.learned` = R);
//!    isolation exempts the whole auto-skill and nothing else.
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

use crate::buffer::{
    AddOutcome, AttemptRecord, ContractRegistry, Example, LearnLog, LearningBuffer, LogRecord,
};
use crate::certify::{self, Calibration, Certification};
use crate::config::{LearningConfig, OracleConfig};
use crate::container::{
    DecisionModel, FileBuilder, LearnedRows, NewSkill, OverlayBuilder, Verify, WriteReport,
};
use crate::eval::{self, SkillScorer, resolve_threads};
use crate::fit::{self, TaskFit};
use crate::generation;
use crate::manifest::{
    self, Gate, GateParams, LearnedRecord, Rubric, SkillManifest, TaskOrigin, TaskRecord, TaskState,
};
use crate::oracle::{self, CallOutcome, Caller, KeyLookup, OracleClient};
use crate::protocol::{Question, QuestionKind};
use crate::resonance::{ErrStats, TaskView, Topology, decide};
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
    /// The first activation of an auto-skill (every eligible label fitted).
    AutoStart,
    /// A later whole-skill refit of an auto-skill.
    AutoRefit,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Refit => "promote",
            ChangeKind::Activate => "activate",
            ChangeKind::ColdStart => "cold_start",
            ChangeKind::AutoStart => "auto_start",
            ChangeKind::AutoRefit => "auto_refit",
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
        let s = model.skill(skill).ok_or_else(|| {
            anyhow::anyhow!(
                "no skill {} in this model",
                crate::config::quote_unless_key(skill)
            )
        })?;
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
            // A per-label challenger is never built for an auto-skill
            // (`attempt_auto` refits the whole skill).
            ChangeKind::AutoStart | ChangeKind::AutoRefit => {
                bail!(
                    "skill '{}' is an auto-skill: no per-label challenger",
                    self.id
                )
            }
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
/// task `changed` of skill `skill` (a cold start adds exactly that task). When
/// `skill` is an auto-skill the exemption is the whole skill (an attempt refits
/// every eligible label at once, DESIGN D6), and it is the only skill allowed
/// to appear in `after` without being in `before` (born in this generation).
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
        let whole = s.id() == skill && a.manifest.is_auto();
        for t in &s.manifest.tasks {
            if s.id() == skill && (whole || t.i as usize == changed) {
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
        let allowed = if whole {
            usize::MAX
        } else {
            usize::from(s.id() == skill && changed >= s.manifest.tasks.len())
        };
        if extra > allowed {
            v.push(format!("skill '{}' gained {extra} tasks", s.id()));
        }
    }
    for a in after.skills() {
        if before.skill(a.id()).is_none() && !(a.id() == skill && a.manifest.is_auto()) {
            v.push(format!("skill '{}' appeared", a.id()));
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
            .ok_or_else(|| anyhow::anyhow!("no skill {}", crate::config::quote_unless_key(skill)))?
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
    /// The whole-skill numbers of an auto-skill attempt.
    pub auto: Option<AutoReport>,
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
            auto: None,
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
            "auto": self.auto.as_ref().map(AutoReport::to_json),
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
    /// The contracts of the auto-skills (DESIGN D1).
    pub contracts: &'a Mutex<ContractRegistry>,
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
    // The auto-skill branch forks before the served-skill lookup (DESIGN
    // A10): the first attempt of a contract, and every attempt after a
    // rollback that dropped the auto-skill, finds no served skill. It is
    // governed by `auto_skills` alone (`cold_start` is the rule of a data
    // skill's new labels).
    if manifest::is_auto_skill_id(skill) {
        if !ctx.cfg.auto_skills {
            return Ok(AttemptReport::skipped(
                skill,
                label,
                "auto-skills are disabled",
            ));
        }
        return attempt_auto(ctx, skill, label);
    }
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
        auto: None,
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

// ------------------------------------------------------------------ auto-skills

/// The whole-skill numbers of an auto-skill attempt (the attempt JSON's
/// `auto`; labels appear there as they do in every learning listing, never
/// in a log line).
#[derive(Clone, Debug, PartialEq)]
pub struct AutoReport {
    /// Labels of the contract.
    pub labels: usize,
    /// Labels fitted and active in the challenger (sorted).
    pub eligible: Vec<String>,
    /// Labels left quarantined (sorted).
    pub quarantined: Vec<String>,
    /// Rows of the contract: all, in the fit, in the calibration subset C
    /// (of the eligible labels), its even and odd halves.
    pub rows: usize,
    pub fit_rows: usize,
    pub cal_rows: usize,
    pub even: usize,
    pub odd: usize,
    /// Share of the rows the eligible labels hold.
    pub coverage: f64,
    /// Agreement of the challenger's argmin with the oracle's labels on C:
    /// macro over the labels with a C row, the smallest per-label value, and
    /// each label's.
    pub agreement_macro: Option<f64>,
    pub agreement_min: Option<f64>,
    pub agreement: BTreeMap<String, f64>,
}

impl AutoReport {
    pub fn to_json(&self) -> Value {
        json!({
            "labels": self.labels, "eligible": self.eligible, "quarantined": self.quarantined,
            "rows": {"total": self.rows, "fit": self.fit_rows, "calibration": self.cal_rows,
                     "even": self.even, "odd": self.odd},
            "coverage": self.coverage,
            "agreement": {"macro": self.agreement_macro, "min": self.agreement_min,
                          "labels": self.agreement},
        })
    }
}

/// One row of an auto-skill's R with its task and calibration key.
struct AutoRow {
    task: usize,
    row: Row,
    key: String,
    cal: bool,
}

/// Per task: (correct, total) over the C rows under a scorer's argmin, with
/// `truth[j]` the row's candidate in that scorer (`None`: always wrong).
fn auto_accuracy(
    matrix: &[f32],
    cands: usize,
    stats: &[ErrStats],
    temperature: f32,
    cal: &[&AutoRow],
    truth: &[Option<usize>],
) -> Result<BTreeMap<usize, (usize, usize)>> {
    let mut acc: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (j, r) in cal.iter().enumerate() {
        let win = if cands == 0 {
            None
        } else {
            decide(&matrix[j * cands..(j + 1) * cands], stats, temperature)?.winner
        };
        let e = acc.entry(r.task).or_insert((0, 0));
        e.1 += 1;
        e.0 += usize::from(win.is_some() && win == truth[j]);
    }
    Ok(acc)
}

fn macro_of(a: &BTreeMap<usize, (usize, usize)>) -> Option<f64> {
    let frac = |&(c, n): &(usize, usize)| c as f64 / n as f64;
    (!a.is_empty()).then(|| a.values().map(frac).sum::<f64>() / a.len() as f64)
}

/// One attempt of an auto-skill (see the module notes).
fn attempt_auto(ctx: &LearnContext<'_>, skill: &str, label: &str) -> Result<AttemptReport> {
    let threads = resolve_threads(ctx.threads);
    let cfg = ctx.cfg;
    let loaded = ctx.handle.current();
    let model = loaded.model();
    let Some(contract) = ctx.contracts.lock().get(skill).cloned() else {
        return Ok(AttemptReport::skipped(
            skill,
            label,
            "no contract is registered for the auto-skill",
        ));
    };
    if !contract.has_label(label) {
        return Ok(AttemptReport::skipped(
            skill,
            label,
            "the label is not an option of the contract",
        ));
    }
    let served = model.skill(skill);
    let ids: Vec<&str> = contract.ids.iter().map(String::as_str).collect();
    let manifest = match served {
        Some(s) => {
            ensure!(s.manifest.is_auto(), "skill '{skill}' is not an auto-skill");
            s.manifest.clone()
        }
        None => SkillManifest::auto_skeleton(&ids, Some(contract.rubric()), cfg.auto_k),
    };
    // The manifest's labels are the contract's ids in sorted order; a contract
    // whose ids differ from a served skill of the same id cannot happen (the id
    // is their hash), but the examples are attributed by label, never by index.
    let (dim_p, dim_h) = (model.encoder_dim(), model.hashing_dim());
    let dim = model.signal_dim();
    let k_max = cfg.auto_k.min(manifest.recipe.k_max) as usize;
    let n_tasks = manifest.tasks.len();

    // R: the served learned rows, then the pending examples of every label.
    let gen_learned: Vec<Row> = match served {
        Some(_) => model
            .rows_learned(skill)?
            .map(|r| r.rows)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let served_bits: HashSet<(u32, Vec<u32>)> = gen_learned
        .iter()
        .map(|r| (r.task, bits(&r.phi_p)))
        .collect();
    let mut rows: Vec<AutoRow> = Vec::new();
    let of = |task: usize, row: Row| {
        let (key, cal) = certify::auto_row_key(&row.phi_p);
        AutoRow {
            task,
            row,
            key,
            cal,
        }
    };
    for r in gen_learned {
        ensure!(
            (r.task as usize) < n_tasks,
            "skill '{skill}': a learned row names task {} of {n_tasks}",
            r.task
        );
        rows.push(of(r.task as usize, r));
    }
    let mut pending_trigger = 0usize;
    {
        let b = ctx.buffer.lock();
        for (t, l) in manifest.labels.iter().enumerate() {
            for ex in b.examples(skill, l) {
                if served_bits.contains(&(t as u32, bits(&ex.phi_p))) {
                    continue;
                }
                if l == label {
                    pending_trigger += 1;
                }
                rows.push(of(t, ex.to_row(t as u32)));
            }
        }
    }
    if pending_trigger == 0 {
        return Ok(AttemptReport::skipped(skill, label, "no pending example"));
    }
    let trigger = manifest.task_of(label).expect("the label is a contract id");

    // Eligibility per label.
    let mut fit_of: Vec<Vec<usize>> = vec![Vec::new(); n_tasks];
    let mut cal_of: Vec<Vec<usize>> = vec![Vec::new(); n_tasks];
    for (i, r) in rows.iter().enumerate() {
        if r.cal {
            cal_of[r.task].push(i);
        } else {
            fit_of[r.task].push(i);
        }
    }
    let was_active = |t: usize| manifest.tasks[t].is_active();
    let eligible: Vec<usize> = (0..n_tasks)
        .filter(|&t| {
            fit_of[t].len() >= cfg.auto_min_rows
                || (was_active(t) && fit_of[t].len() >= fit::MIN_ROWS_ACTIVE)
        })
        .collect();
    let kind = if (0..n_tasks).any(was_active) {
        ChangeKind::AutoRefit
    } else {
        ChangeKind::AutoStart
    };
    let n_fit_trigger = fit_of[trigger].len();
    let eligible_rows: usize = eligible
        .iter()
        .map(|&t| fit_of[t].len() + cal_of[t].len())
        .sum();
    let coverage = if rows.is_empty() {
        0.0
    } else {
        eligible_rows as f64 / rows.len() as f64
    };
    let mut report = AttemptReport {
        skill: skill.into(),
        label: label.into(),
        kind: Some(kind),
        task: Some(trigger),
        outcome: Outcome::Skipped(String::new()),
        pending: pending_trigger,
        n_fit: n_fit_trigger,
        holdout: None,
        gate_before: served.map(|s| s.manifest.gate.clone()),
        gate_after: None,
        isolation_violations: 0,
        auto: Some(AutoReport {
            labels: n_tasks,
            eligible: eligible
                .iter()
                .map(|&t| manifest.labels[t].clone())
                .collect(),
            quarantined: (0..n_tasks)
                .filter(|t| !eligible.contains(t))
                .map(|t| manifest.labels[t].clone())
                .collect(),
            rows: rows.len(),
            fit_rows: eligible.iter().map(|&t| fit_of[t].len()).sum(),
            cal_rows: 0,
            even: 0,
            odd: 0,
            coverage,
            agreement_macro: None,
            agreement_min: None,
            agreement: BTreeMap::new(),
        }),
    };
    let skip = |mut r: AttemptReport, why: String| {
        r.outcome = Outcome::Skipped(why);
        Ok(r)
    };
    if eligible.len() < 2 || !eligible.contains(&trigger) {
        return skip(
            report,
            format!(
                "auto-skill needs 2 labels with >= {} fit rows including the triggering one (has {} eligible, the trigger {n_fit_trigger} fit rows)",
                cfg.auto_min_rows,
                eligible.len()
            ),
        );
    }
    if kind == ChangeKind::AutoStart && coverage < f64::from(cfg.auto_min_coverage) {
        return skip(
            report,
            format!(
                "auto-skill coverage {coverage:.3} of the eligible labels is below {}",
                cfg.auto_min_coverage
            ),
        );
    }

    // Challenger: every eligible label refitted, the others quarantined.
    let mut next = manifest.clone();
    let mut topologies: BTreeMap<usize, Option<Topology>> = BTreeMap::new();
    for &t in &eligible {
        let fit_rows: Vec<&Row> = fit_of[t].iter().map(|&i| &rows[i].row).collect();
        // The error may reach a log line: the label by its tag only (§4.3).
        let f: TaskFit =
            crate::build::fit_task_rows(&fit_rows, dim_h, k_max).with_context(|| {
                format!("fit of task {t} (label#{})", label_tag(&manifest.labels[t]))
            })?;
        let mut rec = crate::build::task_record(t, &manifest.labels[t], &f);
        rec.origin = TaskOrigin::ColdStart;
        rec.state = TaskState::Active;
        next.tasks[t] = rec;
        topologies.insert(t, Some(f.topology));
    }
    let active_before: Vec<usize> = (0..n_tasks).filter(|&t| was_active(t)).collect();
    if active_before != eligible {
        next.taxonomy_version += 1;
    }
    let views: Vec<Option<TaskView<'_>>> = (0..n_tasks)
        .map(|t| {
            topologies
                .get(&t)
                .and_then(|o| o.as_ref())
                .map(Topology::view)
        })
        .collect();
    let scorer = SkillScorer::new(skill, &next.tasks, &views, placeholder_gate(), dim)?;

    // C of the eligible labels, in (task, key) order; halves over the keys.
    let mut cal: Vec<&AutoRow> = eligible
        .iter()
        .flat_map(|&t| cal_of[t].iter().map(|&i| &rows[i]))
        .collect();
    cal.sort_by(|a, b| (a.task, &a.key).cmp(&(b.task, &b.key)));
    let keys: Vec<&str> = cal.iter().map(|r| r.key.as_str()).collect();
    let (even, odd) = certify::halves(&keys);
    if let Some(a) = report.auto.as_mut() {
        a.cal_rows = cal.len();
        a.even = even.len();
        a.odd = odd.len();
    }
    if cal.len() < 2 {
        return skip(
            report,
            format!(
                "the calibration subset has {} rows of the eligible labels; needs at least 2",
                cal.len()
            ),
        );
    }
    let cal_rows: Vec<&Row> = cal.iter().map(|r| &r.row).collect();
    let matrix = scorer.error_matrix(&cal_rows, dim_h, threads)?;
    let truth: Vec<Option<usize>> = cal.iter().map(|r| scorer.candidate_of(r.task)).collect();
    let cert = book_certify(&matrix, &scorer, &truth, &even, &odd)?;
    let gate = Gate::from_certification(&cert);
    report.gate_after = Some(gate.clone());

    // Agreement of the challenger with the oracle on C.
    let chall = auto_accuracy(
        &matrix,
        scorer.len(),
        scorer.stats(),
        cert.temperature,
        &cal,
        &truth,
    )?;
    let per_label: BTreeMap<String, f64> = chall
        .iter()
        .map(|(&t, &(c, n))| (manifest.labels[t].clone(), c as f64 / n as f64))
        .collect();
    let agreement_macro = macro_of(&chall);
    let agreement_min = per_label.values().copied().reduce(f64::min);
    if let Some(a) = report.auto.as_mut() {
        a.agreement = per_label;
        a.agreement_macro = agreement_macro;
        a.agreement_min = agreement_min;
    }
    match kind {
        ChangeKind::AutoStart => {
            let ok = agreement_macro.is_some_and(|m| m >= f64::from(cfg.auto_min_agreement))
                && agreement_min.is_some_and(|m| m >= 0.5);
            if !ok {
                report.outcome = Outcome::Rejected("auto_agreement".into());
                return Ok(report);
            }
        }
        _ => {
            // The champion's topologies (served) on the same C.
            let champ = SkillScorer::from_model(model, skill)?;
            let cm = champ.error_matrix(&cal_rows, dim_h, threads)?;
            let ct: Vec<Option<usize>> = cal.iter().map(|r| champ.candidate_of(r.task)).collect();
            let champ_acc = auto_accuracy(
                &cm,
                champ.len(),
                champ.stats(),
                champ.gate().temperature,
                &cal,
                &ct,
            )?;
            let frac = |a: &BTreeMap<usize, (usize, usize)>| {
                a.get(&trigger).map(|&(c, n)| c as f64 / n as f64)
            };
            let (champ_task, chall_task) = (frac(&champ_acc), frac(&chall));
            let (champ_macro, chall_macro) = (macro_of(&champ_acc), agreement_macro);
            let ok = |a: Option<f64>, b: Option<f64>| match (a, b) {
                (Some(c), Some(h)) => h + REGRESSION_EPS >= c,
                _ => true,
            };
            let holdout = HoldoutReport {
                rows: cal.len(),
                labels: chall.len(),
                champ_task,
                chall_task,
                champ_macro,
                chall_macro,
                passed: ok(champ_task, chall_task) && ok(champ_macro, chall_macro),
                gated: true,
            };
            let passed = holdout.passed;
            report.holdout = Some(holdout);
            if !passed {
                report.outcome = Outcome::Rejected("holdout_regression".into());
                return Ok(report);
            }
        }
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
        kind.as_str(),
        report
            .holdout
            .as_ref()
            .map_or_else(|| json!(null), HoldoutReport::to_json),
        json!({
            "before": report.gate_before.as_ref().map(gate_json),
            "after": gate_json(&gate),
            "auto": report.auto.as_ref().map(AutoReport::to_json),
        }),
    );
    next.gate = gate;
    let mut learned: Vec<Row> = rows.into_iter().map(|r| r.row).collect();
    learned.sort_by_key(|r| r.task);
    let learned = Rows {
        dim_p,
        dim_h,
        rows: learned,
    }
    .encode()?;
    if served.is_none() {
        let topologies: Vec<Option<Topology>> = (0..n_tasks)
            .map(|t| topologies.remove(&t).flatten())
            .collect();
        b.add_skill(NewSkill {
            manifest: next,
            topologies,
            rows: Rows::empty_blob(dim_p, dim_h)?,
            rows_learned: Some(learned),
        })?;
    } else {
        b.set_skill(next, topologies, LearnedRows::Replace(learned))?;
    }
    let mut violations = Vec::new();
    let published = generation::publish(ctx.state, model.base_path(), &b, g, |m| {
        violations = isolation_violations(model, m, skill, trigger);
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
    let sha256 = published.report.sha256.clone();
    let next = loaded.derive(published.model)?;
    ctx.handle.promote(next);
    tracing::info!(
        skill,
        task = trigger,
        label_sha = %label_tag(label),
        generation = g,
        kind = kind.as_str(),
        eligible = eligible.len(),
        labels = n_tasks,
        "promoted"
    );
    report.outcome = Outcome::Promoted {
        generation: g,
        sha256,
    };
    Ok(report)
}

/// [`certify::certify`] over a scorer's error matrix of calibration rows.
fn book_certify(
    matrix: &[f32],
    scorer: &SkillScorer,
    truth: &[Option<usize>],
    even: &[usize],
    odd: &[usize],
) -> Result<Certification> {
    certify::certify(&Calibration {
        errors: matrix,
        tasks: scorer.len(),
        stats: scorer.stats(),
        truth,
        even,
        odd,
    })
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
    /// `no_key`, `ledger_write`); their texts are unanswered.
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
    // An auto-skill has no calibration rows for the holdout and gate of this
    // pass, and its rows must stay in `rows.learned` (DESIGN D4); it learns
    // online only.
    ensure!(
        !manifest_in.is_auto(),
        "skill {} is an auto-skill (learned online by the server); offline learning does not apply",
        crate::config::quote_unless_key(&skill)
    );
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
            // ledgers alone. The key is checked, never shown.
            let var = &opts.oracle.api_key_env;
            match oracle::read_key(&key, var).0 {
                oracle::KeyState::Missing => bail!(
                    "oracle.enabled is true but the environment variable {var} holds no key: set \
                     it, or set oracle.enabled=false to learn from the --answers ledgers only"
                ),
                oracle::KeyState::Bad(p) => bail!(
                    "oracle.enabled is true but {}: fix the variable, or set oracle.enabled=false \
                     to learn from the --answers ledgers only",
                    oracle::bad_key_text(var, &p)
                ),
                oracle::KeyState::Usable { trimmed } if trimmed > 0 => {
                    tracing::warn!("{}", oracle::trimmed_warning(var, trimmed));
                }
                oracle::KeyState::Usable { .. } => {}
            }
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
            // Overlay-first: an auto-skill's (empty) blob is not in the base.
            let rows = model.tensor_bytes(&m.rows.tensor)?.to_vec();
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
