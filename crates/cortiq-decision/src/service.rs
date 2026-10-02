//! `DecisionService` and the `Escalator` trait (spec §4.3–§4.11, §4.7b, §10).
//!
//! The transport-free core of `cortiq serve` on a decision file. The HTTP layer
//! (cortiq-server) maps routes to these calls, runs them on blocking threads and
//! writes the returned JSON; the oracle cascade (`cascade::Cascade`) plugs in as
//! an [`Escalator`].
//!
//! **One request** ([`DecisionService::decide`]):
//! 1. the model snapshot of the [`ModelHandle`] is taken once (a promotion
//!    during the request does not change its answers); a pinned
//!    `cortiq/decision@<12 hex>` must name it (else 404);
//! 2. every question is matched to a skill ([`crate::matching`]);
//! 3. untrained questions (no match, superset, score, noul) need the oracle: when
//!    the call is not allowed the request fails with 422 before any work;
//! 4. the encoder and the hash run once on the state text; each exact or subset
//!    question is decided by its skill (the errors of a skill are computed once
//!    per request), and the gate of the profile decides `local` or not;
//! 5. undetermined questions (gate rejected, untrained) go to the escalator in
//!    one call when the oracle is allowed (spec §5.1: `oracle.enabled`, the
//!    key's `oracle_allowed` and `cmf.oracle` / `default_per_request`; the
//!    escalator adds the key, budget and stop checks): `oracle` or `cache`
//!    answers replace them; otherwise a trained question stays `abstain` with a
//!    flag and an untrained one fails the request (422, 502 or 503). A
//!    trained question refused because the oracle is not ready (flags
//!    `oracle_disabled`, `no_key`, `budget`, `stopped`) also gives the answer
//!    one line `cmf.hint` saying what to do ([`oracle_hint`]: an admin limit
//!    of `oracle.state` that binds is named with the admin request that
//!    lifts it, [`AdminBinding::server_text`]; logged at most
//!    once a minute per hint, once per process at INFO on a server without
//!    an oracle, [`DecisionService::log_oracle_hint`]);
//! 6. the response, the metering (spec §4.9) and one usage-ledger record are
//!    produced; the escalator observes the decided questions (feedback ring).
//!
//! **Certified** (spec §4.5): an exact match, a string state, the `balanced` or
//! `quality-first` profile, a certified skill gate and a winner whose task comes
//! from the data. It does not depend on the gate accepting (as in `cortiq decide`:
//! the certified guarantee covers the answers with `action: local`); oracle and
//! cache answers are never certified.
//!
//! **Profiles** (spec §4.7b): `balanced` accepts `p_top ≥ τ ∧ novelty ≤ θ` (the
//! certified gate, the rule of `cortiq decide`); `quality-first` also needs
//! `margin ≥ 0.08` and `novelty ≤ min(θ, 0.50)`; `cost-saver` accepts
//! `novelty ≤ θ` without τ and is never certified.
//!
//! **Router extras per question** (spec §4.7b, formulas of cortiq-router
//! `api.rs:1082-1165`): `confident`, `complexity` {score, tier, factors {base,
//! ambiguity, novelty, margin, length}} where ambiguity = 1 − p_top (the
//! router's calibrated confidence), margin = 1 − clamp(8·margin, 0, 1), length =
//! clamp(words/40, 0, 1) over the whitespace words of the state text; oracle and
//! cache answers use p_top = 1 and the local novelty and margin when there is a
//! local decision (else novelty 1, margin 0); `routing` {target, reason} when
//! `routing_tiers` maps the tier; `decision_path` ∈ {`router:certified`,
//! `router:uncertified`, `router:uncertified_subset`, `escalate→cache`,
//! `escalate→oracle`, `escalate→oracle_unavailable`, `escalate→disabled`};
//! `explanation` {top1_vs_top2, decision_path} with `cmf.explain`.
//!
//! Deviation from §4.7b (declared): the spec lists six `decision_path` values
//! and none of them names an exact local answer that is not certified — a JSON
//! state, the `cost-saver` profile (θ only), a skill whose gate did not certify,
//! or a cold-start winner. `router:certified` would claim a certification that
//! does not exist and `router:uncertified_subset` is for subset matches, so that
//! answer is `router:uncertified` (the closest correct name, the same prefix).

use crate::answer::{self, OracleAnswer, Rounding};
use crate::config::Config;
use crate::container::DecisionModel;
use crate::eval::{SkillScorer, TOP_ERRORS, f32_json, jev_confidence};
use crate::keys::{AuthFailure, KeyRecord, KeyStore, NewKey, RateLimiter, now_unix};
use crate::ledger::{Actions, Totals, UsageLedger, UsageRecord};
use crate::manifest::{GateParams, SkillManifest, TaskOrigin};
use crate::matching::{MatchKind, SkillLabels, SkillMatch, match_question};
use crate::metering::{self, Cost, Rates, TokenCache, Usd};
use crate::protocol::{
    ApiError, DecisionRequest, FeedbackRequest, MODEL_ID, ModelRef, PROVIDER, Profile, Question,
    QuestionKind, Reason, RequestLimits, model_name, new_request_id, parse_feedback, parse_request,
};
use crate::resonance::{Decision, decide as decide_errors};
use crate::signal::{Features, SignalEncoder};
use anyhow::{Context, Result, ensure};
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Router quality-first margin threshold (router `policy.rs:54-59`).
pub const QUALITY_FIRST_MARGIN: f32 = 0.08;
/// Router quality-first novelty threshold.
pub const QUALITY_FIRST_NOVELTY: f32 = 0.50;
/// Account of the open mode (no keys, auth not required).
pub const OPEN_ACCOUNT: &str = "anonymous";
/// Default base difficulty of a label (router `api.rs:1095`).
pub const DEFAULT_TASK_COMPLEXITY: f32 = 0.4;
/// Size of the question-contract token memo.
pub const TOKEN_CACHE_CAP: usize = 4096;
/// Instructions of a skill's own question when it has no rubric (the router
/// question of such a skill).
pub const DEFAULT_ROUTE_INSTRUCTIONS: &str =
    "Classify the input into exactly one of the task labels.";

// ------------------------------------------------------------------ loaded model

/// One skill ready to decide.
#[derive(Debug)]
pub struct SkillRuntime {
    scorer: SkillScorer,
    manifest: SkillManifest,
}

impl SkillRuntime {
    pub fn id(&self) -> &str {
        &self.manifest.id
    }

    pub fn scorer(&self) -> &SkillScorer {
        &self.scorer
    }

    pub fn manifest(&self) -> &SkillManifest {
        &self.manifest
    }

    pub fn gate(&self) -> GateParams {
        self.scorer.gate()
    }

    /// The skill's own question over its active labels (the question of a
    /// router input, id `id`): the rubric's instructions, and for each active
    /// label its rubric criterion in the question file's order (the label
    /// itself when the rubric has none); [`DEFAULT_ROUTE_INSTRUCTIONS`]
    /// without a rubric.
    pub fn rubric_question(&self, id: &str) -> Question {
        let active = self.scorer.labels();
        let mut criteria = Map::new();
        let instructions = match &self.manifest.rubric {
            Some(r) => {
                for (k, v) in r.ordered_criteria() {
                    if active.contains(&k) {
                        criteria.insert(k, v);
                    }
                }
                r.instructions.clone()
            }
            None => DEFAULT_ROUTE_INSTRUCTIONS.to_string(),
        };
        for l in active {
            if !criteria.contains_key(l) {
                criteria.insert(l.clone(), Value::String(l.clone()));
            }
        }
        Question {
            id: id.to_string(),
            kind: QuestionKind::Choice,
            instructions: Value::String(instructions),
            criteria: Some(Value::Object(criteria)),
        }
    }

    /// Whether `q` asks the oracle exactly what the skill's own question asks
    /// ([`SkillRuntime::rubric_question`]): a choice with the same
    /// instructions and the same criterion for the same options (in any
    /// order). Only then does the oracle's answer describe the skill's labels
    /// as the skill defines them, whoever asked.
    pub fn follows_rubric(&self, q: &Question) -> bool {
        let own = self.rubric_question(&q.id);
        q.kind == QuestionKind::Choice
            && q.instructions == own.instructions
            && q.criteria == own.criteria
    }
}

/// A decision file (base + generation overlay) with its encoder and skill
/// scorers.
pub struct LoadedModel {
    model: DecisionModel,
    encoder: Arc<SignalEncoder>,
    skills: Vec<SkillRuntime>,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("model", &self.model)
            .finish()
    }
}

impl LoadedModel {
    /// Build the encoder (golden-checked) and a scorer per skill.
    pub fn new(model: DecisionModel) -> Result<Self> {
        let (encoder, _golden) = SignalEncoder::from_model(&model)?;
        Self::with_encoder(model, Arc::new(encoder))
    }

    fn with_encoder(model: DecisionModel, encoder: Arc<SignalEncoder>) -> Result<Self> {
        ensure!(
            encoder.dim() == model.signal_dim(),
            "encoder signal dimension {} differs from the file's {}",
            encoder.dim(),
            model.signal_dim()
        );
        let mut skills = Vec::with_capacity(model.skills().len());
        for s in model.skills() {
            // A skill with no active task (an auto-skill whose labels are all
            // quarantined) gets an empty scorer: listed by `/v1/skills`, no
            // active labels for the matcher (`relate` skips it).
            skills.push(SkillRuntime {
                scorer: SkillScorer::from_model(&model, s.id())
                    .with_context(|| format!("skill '{}'", s.id()))?,
                manifest: s.manifest.clone(),
            });
        }
        Ok(Self {
            model,
            encoder,
            skills,
        })
    }

    /// A new generation of the same file (same representation): the encoder
    /// is shared, the skills are rebuilt.
    pub fn derive(&self, model: DecisionModel) -> Result<Self> {
        ensure!(
            model.representation_id() == self.model.representation_id(),
            "a generation must keep the representation ({} != {})",
            model.representation_id(),
            self.model.representation_id()
        );
        Self::with_encoder(model, Arc::clone(&self.encoder))
    }

    pub fn model(&self) -> &DecisionModel {
        &self.model
    }

    pub fn encoder(&self) -> &Arc<SignalEncoder> {
        &self.encoder
    }

    pub fn skills(&self) -> &[SkillRuntime] {
        &self.skills
    }

    pub fn skill(&self, id: &str) -> Option<&SkillRuntime> {
        self.skills.iter().find(|s| s.id() == id)
    }

    fn skill_index(&self, id: &str) -> Option<usize> {
        self.skills.iter().position(|s| s.id() == id)
    }

    /// sha256 of `decision.manifest` (with an overlay: of the overlay manifest).
    pub fn model_sha(&self) -> &str {
        self.model.model_sha()
    }

    /// `model_sha[..12]`.
    pub fn sha12(&self) -> &str {
        &self.model.model_sha()[..12]
    }

    /// `cortiq/decision@<sha12>`.
    pub fn name(&self) -> String {
        model_name(self.model.model_sha())
    }

    pub fn generation(&self) -> u64 {
        self.model.generation()
    }

    pub fn representation_id(&self) -> &str {
        self.model.representation_id()
    }

    /// The labels of every skill as the matcher sees them: the active ones
    /// (candidates) and, for an auto-skill, its whole contract as known.
    pub fn skill_labels(&self) -> Vec<SkillLabels<'_>> {
        self.skills
            .iter()
            .map(|s| {
                let active = s.scorer.labels();
                if s.manifest.is_auto() {
                    SkillLabels {
                        id: s.id(),
                        active,
                        known: &s.manifest.labels,
                        auto: true,
                    }
                } else {
                    SkillLabels::data(s.id(), active)
                }
            })
            .collect()
    }

    /// Served auto-skills.
    pub fn auto_skills(&self) -> usize {
        self.skills.iter().filter(|s| s.manifest.is_auto()).count()
    }
}

/// The served model: `RwLock<Arc<LoadedModel>>` (spec §10). A request takes one
/// snapshot; [`ModelHandle::promote`] swaps the model atomically.
#[derive(Debug)]
pub struct ModelHandle {
    current: RwLock<Arc<LoadedModel>>,
}

impl ModelHandle {
    pub fn new(model: LoadedModel) -> Self {
        Self {
            current: RwLock::new(Arc::new(model)),
        }
    }

    /// The model now.
    pub fn current(&self) -> Arc<LoadedModel> {
        Arc::clone(&self.current.read())
    }

    /// Serve `model` from now on; returns the previous one.
    pub fn promote(&self, model: LoadedModel) -> Arc<LoadedModel> {
        self.promote_arc(Arc::new(model))
    }

    pub fn promote_arc(&self, model: Arc<LoadedModel>) -> Arc<LoadedModel> {
        std::mem::replace(&mut *self.current.write(), model)
    }
}

// ------------------------------------------------------------------ principal

/// Who is calling: an API key's account, or the open mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    pub account: String,
    /// `hash[..12]` of the key (`None` in open mode).
    pub key12: Option<String>,
    pub plan: String,
    pub rate_per_min: u32,
    pub decision_quota: u64,
    pub token_quota: u64,
    pub credit_usd: Option<Usd>,
    pub oracle_budget_usd: Option<Usd>,
    pub oracle_allowed: bool,
    /// May teach the shared model: its feedback is learned (a new label
    /// starts a cold start), and the oracle's answers to its questions
    /// become examples whatever their instructions (see
    /// [`crate::cascade`]).
    pub learning_allowed: bool,
    /// The implicit open mode ([`Principal::open_implicit`]): nobody is
    /// identified, so nothing this caller brings teaches the shared model —
    /// not even an oracle answer to a skill's own question, which teaches
    /// for any other caller. Such answers are still cached.
    pub implicit_open: bool,
}

impl Principal {
    /// The open mode's caller when `auth.require` is explicitly false: no
    /// limits, the oracle and learning allowed (subject to the server's
    /// configuration and budget).
    pub fn open() -> Self {
        Self {
            account: OPEN_ACCOUNT.into(),
            key12: None,
            plan: "open".into(),
            rate_per_min: 0,
            decision_quota: 0,
            token_quota: 0,
            credit_usd: None,
            oracle_budget_usd: None,
            oracle_allowed: true,
            learning_allowed: true,
            implicit_open: false,
        }
    }

    /// The open mode's caller when it holds only because `auth.require` is
    /// `null` and the server listens on loopback: the listening address says
    /// nothing about the client (a reverse proxy on the same host forwards
    /// anyone), so this caller may neither reach the oracle nor teach the
    /// model. `serve --oracle` lets it reach the oracle
    /// ([`DecisionService::with_open_oracle`]); it never teaches.
    pub fn open_implicit() -> Self {
        Self {
            oracle_allowed: false,
            learning_allowed: false,
            implicit_open: true,
            ..Self::open()
        }
    }

    pub fn from_key(k: &KeyRecord) -> Result<Self> {
        Ok(Self {
            account: k.account.clone(),
            key12: Some(k.hash12().to_string()),
            plan: k.plan.clone(),
            rate_per_min: k.rate_per_min,
            decision_quota: k.decision_quota,
            token_quota: k.token_quota,
            credit_usd: k.credit()?,
            oracle_budget_usd: k.oracle_budget()?,
            oracle_allowed: k.oracle_allowed,
            learning_allowed: k.learning_allowed,
            implicit_open: false,
        })
    }

    pub fn is_open(&self) -> bool {
        self.key12.is_none()
    }
}

/// The key of an `Authorization` header (`Bearer <key>`, or a bare key as the
/// router accepts) or of `x-api-key`.
pub fn presented_key<'a>(
    authorization: Option<&'a str>,
    x_api_key: Option<&'a str>,
) -> Option<&'a str> {
    let from_auth = authorization.map(|h| {
        let h = h.trim();
        match h.split_once(' ') {
            Some((scheme, rest)) if scheme.eq_ignore_ascii_case("bearer") => rest.trim(),
            _ => h,
        }
    });
    from_auth
        .filter(|k| !k.is_empty())
        .or_else(|| x_api_key.map(str::trim).filter(|k| !k.is_empty()))
}

// ------------------------------------------------------------------ escalation

/// Why the oracle was not called.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalReason {
    /// `oracle.enabled` is false, no cascade, or the admin switch is off.
    OracleDisabled,
    /// The environment variable `oracle.api_key_env` is unset or empty.
    NoKey,
    /// The variable holds a value that is not usable as a key (see
    /// [`crate::oracle::check_key`]); it is never sent.
    BadKey,
    /// The key has `oracle_allowed: false`, or the request withheld consent.
    ConsentOff,
    /// The reservation does not fit the budget (global, per key) or `max_calls`.
    Budget,
    /// A stop rule switched the oracle off.
    Stopped,
}

impl RefusalReason {
    /// The reason's own name (`no_key` for [`RefusalReason::NoKey`]).
    pub fn flag(self) -> &'static str {
        match self {
            RefusalReason::OracleDisabled => "oracle_disabled",
            RefusalReason::NoKey => "no_key",
            RefusalReason::BadKey => "bad_key",
            RefusalReason::ConsentOff => "consent_off",
            RefusalReason::Budget => "budget",
            RefusalReason::Stopped => "stopped",
        }
    }

    /// The flags of a question abstained for this reason: the reason's name,
    /// and for a missing or unusable key also `oracle_disabled` (the oracle
    /// is off for every request until the variable is fixed).
    pub fn flags(self) -> &'static [&'static str] {
        match self {
            RefusalReason::NoKey => &["oracle_disabled", "no_key"],
            RefusalReason::BadKey => &["oracle_disabled", "bad_key"],
            RefusalReason::OracleDisabled => &["oracle_disabled"],
            RefusalReason::ConsentOff => &["consent_off"],
            RefusalReason::Budget => &["budget"],
            RefusalReason::Stopped => &["stopped"],
        }
    }
}

/// Whether the oracle can be called now, as `GET /v1/admin/oracle` names it in
/// `status` ([`OracleStatus::label`]).
#[derive(Clone, Debug, PartialEq)]
pub enum OracleStatus {
    /// Enabled, a key in the environment, not stopped, a budget left.
    Ready,
    /// The variable `oracle.api_key_env` is unset or empty.
    NoKey,
    /// The variable holds a value that is not usable as a key: what is wrong
    /// (lengths and positions, never the key).
    BadKey(String),
    /// Not configured (`oracle.enabled` false, no cascade), or switched off
    /// by the admin API (`by_admin`).
    Disabled { by_admin: bool },
    /// Something was spent or reserved, and what is left of the budget
    /// (global or the admin's) cannot hold even the smallest possible call,
    /// or a call it refused, or `max_calls` calls were made. `admin`: an
    /// admin limit of `oracle.state` is among what refuses the next call.
    BudgetExhausted { admin: Option<AdminBinding> },
    /// Nothing was spent, and the budget cannot hold even one call, or
    /// `max_calls` is 0. `min_usd` (`Some` only when the budget is short):
    /// the least budget a call needs — the smallest possible call's
    /// reservation, or, once the budget refused a real (longer) call, the
    /// smallest such refusal's. `calls_zero`: `max_calls` is 0 (no call is
    /// allowed whatever the budget). `admin`: an admin limit of
    /// `oracle.state` is among what refuses the call.
    BudgetTooSmall {
        min_usd: Option<f64>,
        calls_zero: bool,
        admin: Option<AdminBinding>,
    },
    /// A stop rule switched it off (the reason of `oracle.state`, a code of
    /// the closed set: [`crate::oracle::shown_code`]).
    Stopped(String),
}

/// An admin limit of `oracle.state` (`budget_usd` or `max_calls`, set by
/// `POST /v1/admin/oracle`) that refuses the oracle's next call. The file
/// keeps it across restarts, so a larger `--oracle-budget` or
/// `--oracle-max-calls` alone cannot lift it; the admin API takes values up
/// to the configured limits ([`AdminBinding::server_text`] words both).
#[derive(Clone, Debug, PartialEq)]
pub struct AdminBinding {
    /// `max_calls` of `oracle.state`, when the calls made reach it.
    pub max_calls: Option<u64>,
    /// `budget_usd` of `oracle.state` (USD), when it cannot hold the next
    /// call.
    pub budget_usd: Option<f64>,
    /// The configured call limit (`--oracle-max-calls`, `oracle.max_calls`).
    pub configured_max_calls: u64,
    /// The configured budget (`--oracle-budget`, `oracle.budget_usd`), USD.
    pub configured_budget_usd: f64,
    /// The least call limit that admits the next call: the calls made + 1.
    pub need_calls: u64,
    /// The least budget that admits the next call (USD): what was spent and
    /// the call's reservation.
    pub need_usd: f64,
    /// `need_usd` counts the smallest possible call (no real one was
    /// refused yet): a longer question needs more.
    pub need_is_smallest: bool,
}

/// `x` rounded up to the micro-dollar as a JSON number for the admin API
/// (`0.00179`), [`crate::oracle_setup::usd_ceil`] without its `$`.
fn usd_ceil_number(x: f64) -> String {
    let s = crate::oracle_setup::usd_ceil(x);
    s.strip_prefix('$').unwrap_or(&s).to_string()
}

impl AdminBinding {
    /// The configured call limit is reached too.
    pub fn configured_calls_out(&self) -> bool {
        self.need_calls > self.configured_max_calls
    }

    /// The configured budget cannot hold the next call either.
    pub fn configured_budget_short(&self) -> bool {
        self.need_usd > self.configured_budget_usd
    }

    /// What refuses the next call and how a server's operator lifts it, for
    /// the decisions-API hint and the startup line (`file`: where
    /// `oracle.state` is, in words or as a path): the admin limit named with
    /// its value, a figure the admin API accepts (a budget rounded up to the
    /// micro-dollar, at most the configured one) or `null`, and the restart
    /// flag only where a configured limit binds as well.
    pub fn server_text(&self, file: &str) -> String {
        use crate::oracle_setup::{usd_ceil, usd_fine};
        let mut parts = Vec::new();
        let (need_calls, conf_calls) = (self.need_calls, self.configured_max_calls);
        match self.max_calls {
            Some(c) if self.configured_calls_out() => parts.push(format!(
                "the admin limit max_calls {c} in {file} binds, and so does the configured call \
                 limit {conf_calls} (--oracle-max-calls): lift the admin limit with POST \
                 /v1/admin/oracle {{\"max_calls\": null}} and restart the server with \
                 --oracle-max-calls of at least {need_calls}"
            )),
            Some(c) => parts.push(format!(
                "the admin limit max_calls {c} in {file} binds (the file keeps it across \
                 restarts, so --oracle-max-calls cannot lift it): raise it with POST \
                 /v1/admin/oracle {{\"max_calls\": {need_calls}}} (at most the configured \
                 {conf_calls}) or lift it with {{\"max_calls\": null}}"
            )),
            None if self.configured_calls_out() => parts.push(format!(
                "the call limit {conf_calls} (--oracle-max-calls) is reached: restart the server \
                 with --oracle-max-calls of at least {need_calls}"
            )),
            None => {}
        }
        let needs = format!(
            "the next call needs a budget of {}{}",
            usd_fine(self.need_usd),
            if self.need_is_smallest {
                " or more (a longer question more)"
            } else {
                ""
            }
        );
        let conf = usd_fine(self.configured_budget_usd);
        match self.budget_usd {
            Some(b) if self.configured_budget_short() => parts.push(format!(
                "the admin limit budget_usd {} in {file} binds, and so does the configured budget \
                 {conf} (--oracle-budget); {needs}: lift the admin limit with POST \
                 /v1/admin/oracle {{\"budget_usd\": null}} and restart the server with \
                 --oracle-budget of at least {}",
                usd_fine(b),
                usd_ceil(self.need_usd)
            )),
            Some(b) => {
                // Never a figure the admin API refuses (above the
                // configured budget).
                let x = usd_ceil_number(self.need_usd);
                let raise = if x
                    .parse::<f64>()
                    .is_ok_and(|v| v <= self.configured_budget_usd)
                {
                    format!(
                        "raise it with POST /v1/admin/oracle {{\"budget_usd\": {x}}} (at most \
                         the configured {conf}) or lift it with {{\"budget_usd\": null}}"
                    )
                } else {
                    format!(
                        "lift it with POST /v1/admin/oracle {{\"budget_usd\": null}} (the \
                         configured {conf} then applies)"
                    )
                };
                parts.push(format!(
                    "the admin limit budget_usd {} in {file} binds (the file keeps it across \
                     restarts, so --oracle-budget cannot lift it; {needs}): {raise}",
                    usd_fine(b)
                ));
            }
            None if self.configured_budget_short() => parts.push(format!(
                "the budget {conf} (--oracle-budget) cannot hold the next call ({needs}): restart \
                 the server with --oracle-budget of at least {}",
                usd_ceil(self.need_usd)
            )),
            None => {}
        }
        parts.join("; ")
    }
}

impl OracleStatus {
    /// `ready`, `no_key`, `bad_key`, `disabled`, `budget_exhausted`,
    /// `budget_too_small` or `stopped: <reason>`.
    pub fn label(&self) -> String {
        match self {
            OracleStatus::Ready => "ready".into(),
            OracleStatus::NoKey => "no_key".into(),
            OracleStatus::BadKey(_) => "bad_key".into(),
            OracleStatus::Disabled { .. } => "disabled".into(),
            OracleStatus::BudgetExhausted { .. } => "budget_exhausted".into(),
            OracleStatus::BudgetTooSmall { .. } => "budget_too_small".into(),
            OracleStatus::Stopped(r) => format!("stopped: {}", crate::oracle::shown_code(r)),
        }
    }

    pub fn is_ready(&self) -> bool {
        *self == OracleStatus::Ready
    }

    /// The admin limit among what refuses the next call (`None`: none).
    pub fn admin_binding(&self) -> Option<&AdminBinding> {
        match self {
            OracleStatus::BudgetExhausted { admin }
            | OracleStatus::BudgetTooSmall { admin, .. } => admin.as_ref(),
            _ => None,
        }
    }
}

/// How the hint of a decisions-API answer names `oracle.state` (a client
/// is not shown the server's paths; the startup line gives the path).
pub const ADMIN_STATE_FILE_WORDS: &str = "the server's oracle.state";

/// The one-line hint of a trained question that abstained because the oracle
/// was refused for `reason` (`cmf.hint` of a decisions-API answer, and the
/// rate-limited log line of both surfaces). `None` for a refusal the caller
/// chose (`consent_off`). `key_env` is the variable's name, never its value.
pub fn oracle_hint(reason: RefusalReason, status: &OracleStatus, key_env: &str) -> Option<String> {
    Some(match (reason, status) {
        (RefusalReason::ConsentOff, _) => return None,
        (RefusalReason::NoKey, _) | (RefusalReason::OracleDisabled, OracleStatus::NoKey) => {
            format!(
                "the oracle key is not set: set {key_env} in the server's environment and restart it"
            )
        }
        (RefusalReason::BadKey | RefusalReason::OracleDisabled, OracleStatus::BadKey(p)) => {
            format!(
                "the oracle key is not usable ({}): fix {key_env} in the server's environment and restart it",
                p
            )
        }
        (RefusalReason::BadKey, _) => format!(
            "the oracle key is not usable: fix {key_env} in the server's environment and restart it"
        ),
        (RefusalReason::OracleDisabled, OracleStatus::Disabled { by_admin: true }) => {
            "the oracle is switched off by the admin API: POST /v1/admin/oracle {\"enabled\":true} turns it on"
                .into()
        }
        (RefusalReason::OracleDisabled, _) => format!(
            "no oracle answers the questions the local model cannot decide: start the server with --oracle MODEL and set {key_env}"
        ),
        // An admin limit of `oracle.state` binds: a restart alone cannot lift
        // it, so the hint names it and the admin request that does.
        (RefusalReason::Budget, OracleStatus::BudgetExhausted { admin: Some(a) }) => format!(
            "the oracle budget is used up: {}",
            a.server_text(ADMIN_STATE_FILE_WORDS)
        ),
        (RefusalReason::Budget, OracleStatus::BudgetTooSmall { admin: Some(a), .. }) => format!(
            "no oracle call fits: {}",
            a.server_text(ADMIN_STATE_FILE_WORDS)
        ),
        (RefusalReason::Budget, OracleStatus::BudgetExhausted { admin: None }) => {
            "the oracle budget is used up: restart the server with a larger --oracle-budget (or --oracle-max-calls)"
                .into()
        }
        (
            RefusalReason::Budget,
            OracleStatus::BudgetTooSmall {
                min_usd,
                calls_zero,
                admin: None,
            },
        ) => {
            use crate::oracle_setup::{usd_ceil, usd_fine};
            match (min_usd, calls_zero) {
                (Some(m), false) => format!(
                    "the oracle budget cannot hold one call (it needs {} reserved before it is sent): restart the server with --oracle-budget of at least {}",
                    usd_fine(*m),
                    usd_ceil(*m)
                ),
                (Some(m), true) => format!(
                    "max_calls 0 allows no oracle call, and the budget cannot hold one call (it needs {} reserved before it is sent): restart the server with --oracle-max-calls of at least 1 and --oracle-budget of at least {}",
                    usd_fine(*m),
                    usd_ceil(*m)
                ),
                (None, _) => {
                    "max_calls 0 allows no oracle call: restart the server with --oracle-max-calls of at least 1"
                        .into()
                }
            }
        }
        (RefusalReason::Budget, _) => {
            "the oracle budget left (the server's, or this API key's oracle budget or credit) cannot hold this call's reservation: see GET /v1/admin/oracle"
                .into()
        }
        (RefusalReason::Stopped, OracleStatus::Stopped(r)) => format!(
            "the oracle was stopped by a stop rule ({}): see GET /v1/admin/oracle; after the fix POST /v1/admin/oracle {{\"enabled\":true}} resumes it",
            crate::oracle::shown_code(r)
        ),
        (RefusalReason::Stopped, _) => {
            "the oracle was stopped by a stop rule: see GET /v1/admin/oracle; after the fix POST /v1/admin/oracle {\"enabled\":true} resumes it"
                .into()
        }
    })
}

/// At most one log line of the same oracle hint per this interval.
pub const HINT_LOG_EVERY: Duration = Duration::from_secs(60);
/// Hint texts remembered before stale ones are dropped.
const HINT_KINDS_KEPT: usize = 32;

/// Flag of a question whose oracle call failed.
pub const FLAG_ORACLE_UNAVAILABLE: &str = "oracle_unavailable";

/// What happened to one undetermined question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// A fresh oracle verdict.
    Oracle(OracleAnswer),
    /// A cached oracle verdict (no call).
    Cache(OracleAnswer),
    /// Not sent.
    Refused(RefusalReason),
    /// Sent and failed (transport, status, parse, schema); the text is for
    /// logs and never carries request content.
    Failed(String),
}

/// One resolution and extra flags (e.g. `pii_redacted`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub resolution: Resolution,
    pub flags: Vec<String>,
}

impl Resolved {
    pub fn new(resolution: Resolution) -> Self {
        Self {
            resolution,
            flags: Vec::new(),
        }
    }
}

/// The oracle's usage of one request (successful calls only are billed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OracleUsage {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Σ `usage.cost` of the successful calls.
    pub cost: Usd,
}

/// The escalator's answer: one [`Resolved`] per pending question, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EscalationResult {
    pub resolved: Vec<Resolved>,
    pub usage: OracleUsage,
}

/// One undetermined question.
#[derive(Clone, Copy, Debug)]
pub struct Pending<'a> {
    /// Position in `request.questions`.
    pub index: usize,
    pub question: &'a Question,
    pub matched: &'a SkillMatch,
    /// The rejected local decision of a trained question.
    pub local: Option<&'a LocalDecision>,
}

/// The undetermined questions of one request, with everything the cascade
/// needs (the state for the oracle body, φ_P for the cache, the features for
/// the learning buffer).
#[derive(Debug)]
pub struct Escalation<'a> {
    pub request_id: &'a str,
    pub principal: &'a Principal,
    pub model: &'a Arc<LoadedModel>,
    pub request: &'a DecisionRequest,
    pub features: &'a Features,
    pub pending: Vec<Pending<'a>>,
    /// The most the oracle may cost this request (the provider's own USD)
    /// before the caller's `credit_usd` is used up (see
    /// [`DecisionService::oracle_credit_left`]); `None`: no such limit.
    pub oracle_credit_usd: Option<f64>,
}

/// Admin operations served by the escalator (spec §5b).
#[derive(Clone, Debug, PartialEq)]
pub enum AdminCommand {
    /// `GET /v1/admin/oracle`.
    OracleStatus,
    /// `POST /v1/admin/oracle` with its JSON body.
    OracleUpdate(Value),
    /// `GET /v1/admin/learning`.
    Learning,
    /// `GET /v1/admin/generations`.
    Generations,
    /// `POST /v1/admin/rollback {"generation": N}`.
    Rollback { generation: u64 },
}

/// A decided request as the escalator observes it (for the feedback ring).
#[derive(Debug)]
pub struct Observation<'a> {
    pub request_id: &'a str,
    pub principal: &'a Principal,
    pub model: &'a Arc<LoadedModel>,
    pub request: &'a DecisionRequest,
    pub features: &'a Features,
    pub questions: &'a [QuestionOutcome],
}

/// The oracle cascade seen from the service (spec §10). Implemented by
/// `cascade::Cascade`.
pub trait Escalator: Send + Sync {
    /// Resolve the undetermined questions of one request (spec §5.2): the
    /// cache, single flight, one oracle call. Called only after the service's
    /// consent checks passed; must return one [`Resolved`] per pending question.
    fn escalate(&self, escalation: &Escalation<'_>) -> EscalationResult;

    /// `POST /v1/feedback` (spec §5.11): only decisions of the caller's account.
    fn feedback(
        &self,
        feedback: &FeedbackRequest,
        principal: &Principal,
    ) -> Result<Value, ApiError>;

    /// `/v1/admin/{oracle,learning,generations,rollback}`.
    fn admin(&self, command: &AdminCommand) -> Result<Value, ApiError>;

    /// Every successful response (after it is billed). Default: nothing.
    fn observe(&self, _observation: &Observation<'_>) {}

    /// The oracle's readiness (`None`: this escalator does not know; the
    /// service then reports [`OracleStatus::Ready`] when the oracle is
    /// configured).
    fn oracle_status(&self) -> Option<OracleStatus> {
        None
    }
}

// ------------------------------------------------------------------ outcomes

/// The action of a question (spec §4.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Local,
    Abstain,
    Cache,
    Oracle,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Local => "local",
            Action::Abstain => "abstain",
            Action::Cache => "cache",
            Action::Oracle => "oracle",
        }
    }

    /// `source` of the answer.
    pub fn source(self) -> &'static str {
        match self {
            Action::Local | Action::Abstain => "local",
            Action::Cache => "cache",
            Action::Oracle => "oracle",
        }
    }
}

/// The local decision of an exact or subset question.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalDecision {
    pub skill: String,
    /// The candidates decided over, in candidate order (their labels).
    pub labels: Vec<String>,
    /// Task index in the skill of each candidate.
    pub tasks: Vec<usize>,
    /// Error of each candidate.
    pub errors: Vec<f32>,
    pub decision: Decision,
    pub choice: Option<String>,
    /// `(N·p_top − 1)/(N − 1)` in f32 (unclamped, as `cortiq decide` reports it).
    pub confidence: f32,
    /// `p_top ≥ τ ∧ novelty ≤ θ` (the certified gate).
    pub gate_accepted: bool,
    /// Accepted under the request's profile.
    pub accepted: bool,
    pub certified: bool,
    pub gate: GateParams,
    pub profile: Profile,
}

impl LocalDecision {
    pub fn is_novel(&self) -> bool {
        self.decision.is_novel(self.gate.novelty_theta)
    }

    /// Probability of a candidate label.
    pub fn probability(&self, label: &str) -> Option<f32> {
        let i = self.labels.iter().position(|l| l == label)?;
        self.decision.probability_of(i)
    }

    /// (label, error) in rank order, `limit` of them.
    pub fn ranked_errors(&self, limit: usize) -> Vec<(&str, f32)> {
        self.decision
            .ranked
            .iter()
            .take(limit)
            .map(|r| (self.labels[r.index].as_str(), r.error))
            .collect()
    }
}

/// One question of a decided request.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestionOutcome {
    pub id: String,
    pub kind: QuestionKind,
    pub matched: SkillMatch,
    pub action: Action,
    pub local: Option<LocalDecision>,
    pub oracle: Option<OracleAnswer>,
    pub answer: Value,
    pub certified: bool,
    pub confident: bool,
    pub flags: Vec<String>,
    pub decision_path: &'static str,
}

/// Wall time of the stages of a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestTimings {
    pub tokenize: Duration,
    pub encode: Duration,
    /// Combined Metal encoder/reconstruction wall time; zero for split execution.
    pub gpu: Duration,
    pub hash: Duration,
    pub resonance: Duration,
    pub oracle: Duration,
    pub total: Duration,
}

impl RequestTimings {
    pub fn to_json(&self) -> Value {
        let us = |d: Duration| d.as_micros() as u64;
        let mut value = json!({
            "tokenize": us(self.tokenize), "encode": us(self.encode), "hash": us(self.hash),
            "resonance": us(self.resonance), "oracle": us(self.oracle), "total": us(self.total),
        });
        if !self.gpu.is_zero() {
            value["gpu"] = json!(us(self.gpu));
        }
        value
    }
}

/// Metered tokens and money of a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metered {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub processed_tokens: u64,
    pub cost: Cost,
    pub oracle: OracleUsage,
}

/// A decided request.
#[derive(Clone, Debug)]
pub struct Decided {
    /// `cmf-dec-…` (also `x-request-id`).
    pub id: String,
    pub created: u64,
    /// The response body.
    pub response: Value,
    /// The usage-ledger record written for it.
    pub record: UsageRecord,
    pub questions: Vec<QuestionOutcome>,
    pub metered: Metered,
    pub timings: RequestTimings,
}

/// A request decided locally only ([`DecisionService::decide_local`]).
#[derive(Clone, Debug)]
pub struct LocalOnly {
    /// `cortiq/decision@<sha12>` of the snapshot that decided it.
    pub model: String,
    pub generation: u64,
    /// Per question, in request order: its match and, for an exact or subset
    /// match, its local decision (`accepted` under the request's profile).
    pub questions: Vec<(SkillMatch, Option<LocalDecision>)>,
    /// `oracle` is always zero.
    pub timings: RequestTimings,
}

// ------------------------------------------------------------------ service

/// Decrements the in-flight count when dropped.
#[derive(Debug)]
pub struct InflightGuard {
    count: Arc<AtomicUsize>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The decision service.
pub struct DecisionService {
    handle: Arc<ModelHandle>,
    cfg: Arc<Config>,
    rates: Rates,
    limits: RequestLimits,
    escalator: Option<Arc<dyn Escalator>>,
    keys: Option<Arc<KeyStore>>,
    ledger: Option<Arc<UsageLedger>>,
    /// Usage totals when there is no ledger.
    memory_totals: Mutex<BTreeMap<String, Totals>>,
    limiter: RateLimiter,
    auth_required: bool,
    /// The implicit open mode (loopback, `auth.require: null`) may use the
    /// oracle: the operator enabled it on the command line (`serve --oracle`).
    open_oracle: bool,
    admin_token: Option<String>,
    inflight: Arc<AtomicUsize>,
    tokens: TokenCache,
    /// When each oracle hint was last logged ([`HINT_LOG_EVERY`], per hint).
    hint_logged: Mutex<HashMap<String, Instant>>,
    /// The hint of a server without an oracle was logged (once per process).
    off_hint_logged: AtomicBool,
    /// Hints are logged ([`DecisionService::without_hint_log`]: not, for a
    /// command line that words its own).
    log_hints: bool,
}

impl std::fmt::Debug for DecisionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionService")
            .field("model", &self.handle.current().name())
            .field("escalator", &self.escalator.is_some())
            .field("keys", &self.keys.as_ref().map(|k| k.len()))
            .field("ledger", &self.ledger.is_some())
            .field("auth_required", &self.auth_required)
            .field("open_oracle", &self.open_oracle)
            .field("admin", &self.admin_token.is_some())
            .finish()
    }
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    tracing::error!(error = %e, "decision service internal error");
    ApiError::internal("internal error")
}

fn auth_error(f: AuthFailure) -> ApiError {
    ApiError::new(Reason::Unauthorized, f.to_string())
}

impl DecisionService {
    /// A service over `model` (spec §10). Authentication is required unless
    /// `auth.require` is false (see [`DecisionService::with_loopback`]); the
    /// admin token comes from the configured environment variable.
    pub fn open(
        model: Arc<ModelHandle>,
        cfg: Config,
        escalator: Option<Arc<dyn Escalator>>,
    ) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            rates: cfg.rates()?,
            limits: (&cfg.limits).into(),
            auth_required: cfg.auth.required(false),
            open_oracle: false,
            admin_token: cfg.auth.admin_token(),
            handle: model,
            escalator,
            keys: None,
            ledger: None,
            memory_totals: Mutex::new(BTreeMap::new()),
            limiter: RateLimiter::new(),
            inflight: Arc::new(AtomicUsize::new(0)),
            tokens: TokenCache::new(TOKEN_CACHE_CAP),
            hint_logged: Mutex::new(HashMap::new()),
            off_hint_logged: AtomicBool::new(false),
            log_hints: true,
            cfg: Arc::new(cfg),
        })
    }

    /// Never log the oracle hints (`cmf.hint` is kept): `cortiq decide`
    /// prints its own, worded for the command line, and a server's words
    /// ("set … in the server's environment and restart it") would appear
    /// above them whatever RUST_LOG says.
    pub fn without_hint_log(mut self) -> Self {
        self.log_hints = false;
        self
    }

    /// Resolve `auth.require: null` for the listening address.
    pub fn with_loopback(mut self, loopback: bool) -> Self {
        self.auth_required = self.cfg.auth.required(loopback);
        self
    }

    /// Let the open mode that holds only because of a loopback address
    /// (`auth.require: null`) use the oracle: the operator enabled the oracle
    /// explicitly (`cortiq serve --oracle MODEL`). It still may not teach the
    /// model: its feedback is not learned and the oracle's answers to its
    /// questions — a skill's own question, as `/v1/route` asks it, included —
    /// are cached but never become examples ([`Principal::implicit_open`]);
    /// an explicit `auth.require: false` gives both, as before.
    pub fn with_open_oracle(mut self, allowed: bool) -> Self {
        self.open_oracle = allowed;
        self
    }

    /// The key store of the state directory.
    pub fn with_keys(mut self, keys: Arc<KeyStore>) -> Self {
        self.keys = Some(keys);
        self
    }

    /// The usage ledger of the state directory.
    pub fn with_ledger(mut self, ledger: Arc<UsageLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// Override the admin token (tests; the server uses the environment).
    pub fn with_admin_token(mut self, token: Option<String>) -> Self {
        self.admin_token = token.filter(|t| !t.is_empty());
        self
    }

    pub fn handle(&self) -> &Arc<ModelHandle> {
        &self.handle
    }

    pub fn config(&self) -> &Arc<Config> {
        &self.cfg
    }

    pub fn limits(&self) -> RequestLimits {
        self.limits
    }

    pub fn keys(&self) -> Option<&Arc<KeyStore>> {
        self.keys.as_ref()
    }

    pub fn ledger(&self) -> Option<&Arc<UsageLedger>> {
        self.ledger.as_ref()
    }

    pub fn escalator(&self) -> Option<&Arc<dyn Escalator>> {
        self.escalator.as_ref()
    }

    /// Whether the oracle can be called now: disabled without a cascade or
    /// with `oracle.enabled` false, else what the cascade reports.
    pub fn oracle_status(&self) -> OracleStatus {
        match &self.escalator {
            Some(e) if self.cfg.oracle.enabled => e.oracle_status().unwrap_or(OracleStatus::Ready),
            _ => OracleStatus::Disabled { by_admin: false },
        }
    }

    /// The hint of a trained question abstained for `reason` (see
    /// [`oracle_hint`]); `None` without a cascade (`cortiq decide`, which
    /// never calls the oracle) and for `consent_off`.
    pub fn oracle_hint(&self, reason: RefusalReason) -> Option<String> {
        self.escalator.as_ref()?;
        oracle_hint(reason, &self.oracle_status(), &self.cfg.oracle.api_key_env)
    }

    /// Log `hint` (the router surface carries no hint in its answers; the
    /// log is where an operator sees it). A server run without an oracle
    /// (`oracle.enabled` false: the operator's choice) logs it once per
    /// process at INFO; otherwise each distinct hint is a WARN at most once
    /// per [`HINT_LOG_EVERY`], so one kind never hides another.
    pub fn log_oracle_hint(&self, hint: &str) {
        if !self.log_hints {
            return;
        }
        if self.escalator.is_none() || !self.cfg.oracle.enabled {
            if !self.off_hint_logged.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "a question the local model could not decide abstained: {hint} (logged once)"
                );
            }
            return;
        }
        let now = Instant::now();
        let mut last = self.hint_logged.lock();
        if last
            .get(hint)
            .is_some_and(|t| now.duration_since(*t) < HINT_LOG_EVERY)
        {
            return;
        }
        // A handful of hint texts exist (a stop reason names one of a few
        // rules); stale entries go before the map could grow, and it never
        // holds more than HINT_KINDS_KEPT.
        if last.len() >= HINT_KINDS_KEPT {
            last.retain(|_, t| now.duration_since(*t) < HINT_LOG_EVERY);
            if last.len() >= HINT_KINDS_KEPT {
                last.clear();
            }
        }
        last.insert(hint.to_string(), now);
        drop(last);
        tracing::warn!("a question the local model could not decide abstained: {hint}");
    }

    /// Whether a key is required now (false only in open mode: no keys and
    /// auth not required).
    pub fn auth_enabled(&self) -> bool {
        self.auth_required || self.keys.as_ref().is_some_and(|k| !k.is_empty())
    }

    // ---------------------------------------------------------------- auth

    /// The caller of a request (spec §4.10): open mode, or the account of a
    /// valid, active, unexpired key (else 401).
    pub fn authenticate(
        &self,
        authorization: Option<&str>,
        x_api_key: Option<&str>,
    ) -> Result<Principal, ApiError> {
        if let Some(k) = &self.keys
            && let Err(e) = k.maybe_reload()
        {
            tracing::error!(error = %e, "keys.json reload failed; keeping the loaded keys");
        }
        if !self.auth_enabled() {
            // Open mode: explicitly configured, or only because of a loopback
            // address (`auth.require: null`).
            return Ok(if self.cfg.auth.require == Some(false) {
                Principal::open()
            } else {
                Principal {
                    oracle_allowed: self.open_oracle,
                    ..Principal::open_implicit()
                }
            });
        }
        let raw = presented_key(authorization, x_api_key)
            .ok_or_else(|| auth_error(AuthFailure::Missing))?;
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| auth_error(AuthFailure::Invalid))?;
        let rec = keys.authenticate(raw, now_unix()).map_err(auth_error)?;
        Principal::from_key(&rec).map_err(internal)
    }

    /// Rate window (429) and quotas (402), checked before any work (spec §4.10).
    pub fn admit(&self, p: &Principal) -> Result<(), ApiError> {
        self.admit_at(p, now_unix())
    }

    /// [`DecisionService::admit`] at a given unix time.
    pub fn admit_at(&self, p: &Principal, now: u64) -> Result<(), ApiError> {
        if let Err(retry) = self.limiter.check(&p.account, p.rate_per_min, now) {
            return Err(ApiError::new(
                Reason::RateLimited,
                format!(
                    "rate limit of {} requests per minute exceeded",
                    p.rate_per_min
                ),
            )
            .with_retry_after(retry)
            .with_detail("rate_per_min", json!(p.rate_per_min)));
        }
        self.check_quotas(p)
    }

    /// The quotas alone (402), with the usage recorded so far: checked by
    /// [`DecisionService::admit`] before a request and again before every
    /// input of a router batch after the first, so that one batch cannot
    /// run past a quota or a credit limit.
    pub fn check_quotas(&self, p: &Principal) -> Result<(), ApiError> {
        let t = self.totals(&p.account);
        let quota = |what: &str, used: Value, limit: Value| {
            ApiError::new(Reason::QuotaExceeded, format!("{what} quota exhausted"))
                .with_detail("quota", json!(what))
                .with_detail("used", used)
                .with_detail("limit", limit)
        };
        if p.decision_quota > 0 && t.decisions >= p.decision_quota {
            return Err(quota(
                "decision",
                json!(t.decisions),
                json!(p.decision_quota),
            ));
        }
        if p.token_quota > 0 && t.tokens() >= p.token_quota {
            return Err(quota("token", json!(t.tokens()), json!(p.token_quota)));
        }
        if let Some(c) = p.credit_usd
            && t.cost_usd >= c
        {
            return Err(quota(
                "credit",
                json!(t.cost_usd.to_string()),
                json!(c.to_string()),
            ));
        }
        Ok(())
    }

    /// Room for `questions` more decisions under the decision quota (402):
    /// a request with more questions than the quota has left is refused
    /// before any work, so one `/v1/decisions` request cannot run past it.
    /// With the quota used up the error is [`DecisionService::check_quotas`]'s.
    pub fn check_decision_room(&self, p: &Principal, questions: usize) -> Result<(), ApiError> {
        if p.decision_quota == 0 {
            return Ok(());
        }
        let used = self.totals(&p.account).decisions;
        let asked = questions as u64;
        if used >= p.decision_quota {
            return self.check_quotas(p);
        }
        if used.saturating_add(asked) > p.decision_quota {
            let left = p.decision_quota - used;
            return Err(ApiError::new(
                Reason::QuotaExceeded,
                format!(
                    "decision quota exceeded: the request has {asked} questions and {left} decisions are left"
                ),
            )
            .with_detail("quota", json!("decision"))
            .with_detail("used", json!(used))
            .with_detail("limit", json!(p.decision_quota))
            .with_detail("requested", json!(asked)));
        }
        Ok(())
    }

    /// The most the oracle may cost a request of `p` (the provider's own USD)
    /// before `p`'s `credit_usd` is used up: `(credit − cost so far) / markup`
    /// with passthrough. `None` without a credit limit, or when the oracle is
    /// not billed (no passthrough, or a zero markup). The cascade refuses a
    /// call whose reservation does not fit (`budget`); the request's own local
    /// prices can still take it past the credit by one request.
    pub fn oracle_credit_left(&self, p: &Principal) -> Option<f64> {
        let credit = p.credit_usd?;
        if !self.rates.oracle_passthrough {
            return None;
        }
        let markup = self.rates.oracle_markup.to_f64();
        if markup <= 0.0 {
            return None;
        }
        let used = self.totals(&p.account).cost_usd.to_f64();
        Some((credit.to_f64() - used).max(0.0) / markup)
    }

    /// Take an in-flight slot (`max_inflight`, else 429 `OVERLOADED`).
    pub fn enter(&self) -> Result<InflightGuard, ApiError> {
        let max = self.cfg.limits.max_inflight;
        let n = self.inflight.fetch_add(1, Ordering::AcqRel) + 1;
        let guard = InflightGuard {
            count: Arc::clone(&self.inflight),
        };
        if n > max {
            drop(guard);
            return Err(ApiError::new(
                Reason::Overloaded,
                format!("more than {max} requests in flight"),
            )
            .with_retry_after(1));
        }
        Ok(guard)
    }

    /// Requests in flight now.
    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    /// An account's usage totals (ledger, or in memory without one).
    pub fn totals(&self, account: &str) -> Totals {
        match &self.ledger {
            Some(l) => l.totals(account),
            None => self
                .memory_totals
                .lock()
                .get(account)
                .cloned()
                .unwrap_or_default(),
        }
    }

    fn record_usage(&self, r: &UsageRecord) -> Result<(), ApiError> {
        match &self.ledger {
            Some(l) => l.append(r).map_err(internal),
            None => self
                .memory_totals
                .lock()
                .entry(r.account.clone())
                .or_default()
                .add(r)
                .map_err(internal),
        }
    }

    // ---------------------------------------------------------------- decide

    /// Parse and decide a request body.
    pub fn decide_body(&self, body: &[u8], p: &Principal) -> Result<Decided, ApiError> {
        let req = parse_request(body, &self.limits)?;
        self.decide(&req, p)
    }

    /// Parse and decide a TypeSafe/Jev System One request.  The adapter has a
    /// slightly more permissive wire contract than the native/OpenRouter
    /// endpoint (for example the SDK may omit `model`), but malformed System
    /// One bodies are schema errors (422) rather than a change to the native
    /// endpoint's established 400 contract.
    pub fn decide_systemone_body(&self, body: &[u8], p: &Principal) -> Result<Decided, ApiError> {
        let req =
            crate::protocol::parse_systemone_request(body, &self.limits).map_err(|mut e| {
                if e.reason == crate::protocol::Reason::InvalidRequest {
                    e.status = 422;
                }
                e
            })?;
        self.decide(&req, p)
    }

    /// Decide a request (see the module notes).
    pub fn decide(&self, req: &DecisionRequest, p: &Principal) -> Result<Decided, ApiError> {
        let t0 = Instant::now();
        let created = now_unix();
        let id = new_request_id(created);
        let model = self.handle.current();
        check_pinned(&model, req)?;
        self.check_decision_room(p, req.questions.len())?;

        // Matching.
        let matches = match_questions(&model, req)?;
        let consent = self.consent(req, p);
        if let Err(reason) = consent {
            let untrained: Vec<usize> = (0..matches.len())
                .filter(|&i| !matches[i].kind.is_local())
                .collect();
            if !untrained.is_empty() {
                let reasons: Vec<(usize, Resolution)> = untrained
                    .iter()
                    .map(|&i| (i, Resolution::Refused(reason)))
                    .collect();
                return Err(untrained_error(req, &matches, &reasons));
            }
        }

        // Encoder and hash once, local decisions.
        let LocalStage {
            features,
            signal: st,
            mut locals,
            resonance,
        } = local_stage(&model, req, &matches, self.cfg.learning.auto_tau)?;

        // Undetermined questions.
        let pending_idx: Vec<usize> = (0..matches.len())
            .filter(|&i| locals[i].as_ref().is_none_or(|l| !l.accepted))
            .collect();
        let mut resolved: Vec<Option<Resolved>> = vec![None; matches.len()];
        let mut oracle_usage = OracleUsage::default();
        let to = Instant::now();
        if !pending_idx.is_empty() {
            match (consent, &self.escalator) {
                (Ok(()), Some(esc)) => {
                    let pending: Vec<Pending<'_>> = pending_idx
                        .iter()
                        .map(|&i| Pending {
                            index: i,
                            question: &req.questions[i],
                            matched: &matches[i],
                            local: locals[i].as_ref(),
                        })
                        .collect();
                    let e = Escalation {
                        request_id: &id,
                        principal: p,
                        model: &model,
                        request: req,
                        features: &features,
                        pending,
                        oracle_credit_usd: self.oracle_credit_left(p),
                    };
                    let result = esc.escalate(&e);
                    oracle_usage = result.usage;
                    if result.resolved.len() == pending_idx.len() {
                        for (&i, r) in pending_idx.iter().zip(result.resolved) {
                            resolved[i] = Some(r);
                        }
                    } else {
                        let why = format!(
                            "the escalator returned {} resolutions for {} questions",
                            result.resolved.len(),
                            pending_idx.len()
                        );
                        tracing::error!("{why}");
                        oracle_usage = OracleUsage::default();
                        for &i in &pending_idx {
                            resolved[i] = Some(Resolved::new(Resolution::Failed(why.clone())));
                        }
                    }
                }
                (Err(reason), _) => {
                    for &i in &pending_idx {
                        resolved[i] = Some(Resolved::new(Resolution::Refused(reason)));
                    }
                }
                (Ok(()), None) => {
                    for &i in &pending_idx {
                        resolved[i] = Some(Resolved::new(Resolution::Refused(
                            RefusalReason::OracleDisabled,
                        )));
                    }
                }
            }
            // A verdict outside its question's schema fails. One oracle call
            // answers every pending question, so such a verdict fails the whole
            // call (spec §5.4), and a failed call is not billed.
            let mut failed_call = false;
            for &i in &pending_idx {
                if let Some(r) = &mut resolved[i]
                    && let Resolution::Oracle(a) | Resolution::Cache(a) = &r.resolution
                    && let Err(why) = a.check(&req.questions[i])
                {
                    tracing::warn!(
                        request = %id,
                        "an oracle verdict did not fit its question's schema"
                    );
                    failed_call |= matches!(r.resolution, Resolution::Oracle(_));
                    r.resolution = Resolution::Failed(why);
                }
            }
            if failed_call {
                for &i in &pending_idx {
                    if let Some(r) = &mut resolved[i]
                        && matches!(r.resolution, Resolution::Oracle(_))
                    {
                        r.resolution = Resolution::Failed(
                            "the call returned a verdict outside a question's schema".into(),
                        );
                    }
                }
                oracle_usage = OracleUsage::default();
            }
        }
        let oracle_time = to.elapsed();

        // Untrained questions without a verdict fail the request.
        let unresolved: Vec<(usize, Resolution)> = (0..matches.len())
            .filter(|&i| locals[i].is_none())
            .filter_map(|i| {
                let r = resolved[i].as_ref().map(|r| r.resolution.clone())?;
                match r {
                    Resolution::Oracle(_) | Resolution::Cache(_) => None,
                    other => Some((i, other)),
                }
            })
            .collect();
        if !unresolved.is_empty() {
            return Err(untrained_error(req, &matches, &unresolved));
        }

        // Answers.
        let rounding = req
            .cmf
            .round
            .unwrap_or_else(|| Rounding::from_config(self.cfg.response.round));
        // A trained question refused the oracle: one hint (the first such
        // question's reason), in `cmf.hint` and the rate-limited log.
        let hint = (0..matches.len())
            .filter(|&i| locals[i].is_some())
            .find_map(|i| match resolved[i].as_ref().map(|r| &r.resolution) {
                Some(Resolution::Refused(r)) => self.oracle_hint(*r),
                _ => None,
            });
        if let Some(h) = &hint {
            self.log_oracle_hint(h);
        }
        let mut outcomes = Vec::with_capacity(matches.len());
        for (i, (q, m)) in req.questions.iter().zip(matches).enumerate() {
            outcomes.push(self.outcome(q, m, locals[i].take(), resolved[i].take(), rounding));
        }

        // Metering.
        let tokenizer = model.encoder().encoder().tokenizer();
        let mut input_tokens = metering::state_tokens(tokenizer, &req.state_text, st.tokens);
        for q in &req.questions {
            input_tokens += self.tokens.contract_tokens(
                tokenizer,
                &q.contract(),
                &q.instructions,
                q.criteria.as_ref().unwrap_or(&Value::Null),
            );
        }
        let output_tokens: u64 = outcomes
            .iter()
            .map(|o| metering::answer_output_tokens(&o.answer))
            .sum();
        let cost = self
            .rates
            .cost(input_tokens, output_tokens, oracle_usage.cost)
            .map_err(internal)?;
        let metered = Metered {
            input_tokens,
            output_tokens,
            processed_tokens: st.tokens as u64,
            cost,
            oracle: oracle_usage,
        };
        let timings = RequestTimings {
            tokenize: st.tokenize,
            encode: st.encode,
            gpu: st.gpu,
            hash: st.hash,
            resonance,
            oracle: oracle_time,
            total: t0.elapsed(),
        };
        let mut actions = Actions::default();
        for o in &outcomes {
            match o.action {
                Action::Local => actions.local += 1,
                Action::Abstain => actions.abstain += 1,
                Action::Cache => actions.cache += 1,
                Action::Oracle => actions.oracle += 1,
            }
        }
        let record = UsageRecord {
            ts: created,
            id: id.clone(),
            account: p.account.clone(),
            key12: p.key12.clone(),
            model: model.name(),
            generation: model.generation(),
            input_tokens,
            output_tokens,
            cost_usd: cost.total.to_f64(),
            cost_local_usd: cost.local.to_f64(),
            cost_oracle_usd: cost.oracle_billed.to_f64(),
            oracle_calls: oracle_usage.calls,
            cache_hits: actions.cache,
            questions: outcomes.len() as u64,
            actions,
        };
        let mut response =
            self.response_json(&id, created, &model, req, &outcomes, &metered, &timings);
        if let Some(h) = hint {
            response["cmf"]["hint"] = Value::String(h);
        }
        self.record_usage(&record)?;
        if let Some(esc) = &self.escalator {
            esc.observe(&Observation {
                request_id: &id,
                principal: p,
                model: &model,
                request: req,
                features: &features,
                questions: &outcomes,
            });
        }
        Ok(Decided {
            id,
            created,
            response,
            record,
            questions: outcomes,
            metered,
            timings,
        })
    }

    /// Only the local part of [`DecisionService::decide`] (the shadow mode of
    /// `cortiq serve --shadow-of`, spec §4.15): the same matching, encoder,
    /// resonance and gate of the request's profile, so a local question gets
    /// exactly the decision `decide` would start from. Nothing else happens:
    /// no escalator (no oracle, no cache, no single flight), no metering, no
    /// usage record and no quota, no observation (no feedback ring, no learning
    /// buffer). An untrained question has no local decision (`None`) instead of
    /// failing the request.
    pub fn decide_local(&self, req: &DecisionRequest) -> Result<LocalOnly, ApiError> {
        let t0 = Instant::now();
        let model = self.handle.current();
        check_pinned(&model, req)?;
        let matches = match_questions(&model, req)?;
        let stage = local_stage(&model, req, &matches, self.cfg.learning.auto_tau)?;
        let timings = RequestTimings {
            tokenize: stage.signal.tokenize,
            encode: stage.signal.encode,
            gpu: stage.signal.gpu,
            hash: stage.signal.hash,
            resonance: stage.resonance,
            oracle: Duration::ZERO,
            total: t0.elapsed(),
        };
        Ok(LocalOnly {
            model: model.name(),
            generation: model.generation(),
            questions: matches.into_iter().zip(stage.locals).collect(),
            timings,
        })
    }

    /// The service-level oracle consent (spec §5.1).
    fn consent(&self, req: &DecisionRequest, p: &Principal) -> Result<(), RefusalReason> {
        if self.escalator.is_none() || !self.cfg.oracle.enabled {
            return Err(RefusalReason::OracleDisabled);
        }
        let asked = req
            .cmf
            .oracle
            .unwrap_or(self.cfg.oracle.default_per_request);
        if !p.oracle_allowed || !asked {
            return Err(RefusalReason::ConsentOff);
        }
        Ok(())
    }

    fn outcome(
        &self,
        q: &Question,
        m: SkillMatch,
        local: Option<LocalDecision>,
        resolved: Option<Resolved>,
        rounding: Rounding,
    ) -> QuestionOutcome {
        let mut flags = Vec::new();
        let (action, oracle, answer, decision_path) = match (&local, resolved) {
            (Some(l), None) => (
                Action::Local,
                None,
                local_answer(q, l, rounding),
                if l.certified {
                    "router:certified"
                } else if m.kind == MatchKind::Subset {
                    "router:uncertified_subset"
                } else {
                    "router:uncertified"
                },
            ),
            (_, Some(r)) => {
                flags.extend(r.flags);
                match r.resolution {
                    Resolution::Oracle(a) => (
                        Action::Oracle,
                        Some(a.clone()),
                        a.to_answer(q),
                        "escalate→oracle",
                    ),
                    Resolution::Cache(a) => (
                        Action::Cache,
                        Some(a.clone()),
                        a.to_answer(q),
                        "escalate→cache",
                    ),
                    Resolution::Refused(reason) => {
                        flags.extend(reason.flags().iter().map(|f| f.to_string()));
                        (
                            Action::Abstain,
                            None,
                            local_answer_or_null(q, local.as_ref(), rounding),
                            "escalate→disabled",
                        )
                    }
                    Resolution::Failed(_) => {
                        flags.push(FLAG_ORACLE_UNAVAILABLE.to_string());
                        (
                            Action::Abstain,
                            None,
                            local_answer_or_null(q, local.as_ref(), rounding),
                            "escalate→oracle_unavailable",
                        )
                    }
                }
            }
            (None, None) => (Action::Abstain, None, Value::Null, "escalate→disabled"),
        };
        // Local and abstained answers carry the gate's certified flag (the rule
        // of `cortiq decide`, independent of acceptance); oracle and cache
        // answers are never certified.
        let certified = matches!(action, Action::Local | Action::Abstain)
            && local.as_ref().is_some_and(|l| l.certified);
        let confident = match action {
            Action::Local => true,
            Action::Oracle | Action::Cache => true,
            Action::Abstain => false,
        };
        QuestionOutcome {
            id: q.id.clone(),
            kind: q.kind,
            matched: m,
            action,
            local,
            oracle,
            answer,
            certified,
            confident,
            flags,
            decision_path,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn response_json(
        &self,
        id: &str,
        created: u64,
        model: &LoadedModel,
        req: &DecisionRequest,
        outcomes: &[QuestionOutcome],
        metered: &Metered,
        timings: &RequestTimings,
    ) -> Value {
        let words = req.state_text.split_whitespace().count();
        let mut answers = Map::new();
        let mut questions = Map::new();
        for o in outcomes {
            answers.insert(o.id.clone(), o.answer.clone());
            questions.insert(o.id.clone(), self.question_json(o, words, req.cmf.explain));
        }
        let c = &metered.cost;
        json!({
            "id": id,
            "created": created,
            "model": model.name(),
            "provider": PROVIDER,
            "answers": Value::Object(answers),
            "usage": {
                "input_tokens": metered.input_tokens,
                "output_tokens": metered.output_tokens,
                "cost": c.total.to_f64(),
            },
            "cmf": {
                "generation": model.generation(),
                "model_sha": model.model_sha(),
                "representation_id": model.representation_id(),
                "timings_us": timings.to_json(),
                "questions": Value::Object(questions),
                "usage": {
                    "local": {
                        "input_tokens": metered.input_tokens,
                        "output_tokens": metered.output_tokens,
                        "processed_tokens": metered.processed_tokens,
                        "cost": c.local.to_f64(),
                    },
                    "oracle": {
                        "calls": metered.oracle.calls,
                        "input_tokens": metered.oracle.input_tokens,
                        "output_tokens": metered.oracle.output_tokens,
                        "cost": metered.oracle.cost.to_f64(),
                        "billed": c.oracle_billed.to_f64(),
                        "passthrough": self.rates.oracle_passthrough,
                    },
                },
            },
        })
    }

    fn question_json(&self, o: &QuestionOutcome, words: usize, explain: bool) -> Value {
        let mut m = Map::new();
        m.insert("action".into(), json!(o.action.as_str()));
        m.insert("source".into(), json!(o.action.source()));
        m.insert("skill".into(), json!(o.matched.skill));
        m.insert("match".into(), json!(o.matched.kind.as_str()));
        m.insert("certified".into(), json!(o.certified));
        let (gate, errors) = match &o.local {
            Some(l) => {
                let d = &l.decision;
                let gate = json!({
                    "accepted": l.accepted,
                    "p_top": f32_json(d.p_top),
                    "tau": f32_json(l.gate.tau),
                    "novelty": f32_json(d.novelty),
                    "theta": f32_json(l.gate.novelty_theta),
                    "is_novel": l.is_novel(),
                    "margin": f32_json(d.margin),
                    "profile": l.profile.as_str(),
                });
                let limit = if explain { usize::MAX } else { TOP_ERRORS };
                let errors: Map<String, Value> = l
                    .ranked_errors(limit)
                    .into_iter()
                    .map(|(lab, e)| (lab.to_string(), f32_json(e)))
                    .collect();
                (gate, Value::Object(errors))
            }
            None => (Value::Null, Value::Object(Map::new())),
        };
        m.insert("gate".into(), gate);
        m.insert("errors".into(), errors);
        m.insert("flags".into(), json!(o.flags));
        if !o.matched.unknown.is_empty() {
            m.insert("unknown_options".into(), json!(o.matched.unknown));
        }
        if let Some(r) = &o.matched.reason {
            m.insert("reason".into(), json!(r));
        }
        m.insert("confident".into(), json!(o.confident));
        let cx = self.complexity(o, words);
        m.insert("complexity".into(), cx.to_json());
        if let Some(r) = self.routing(o, &cx) {
            m.insert("routing".into(), r);
        }
        m.insert("decision_path".into(), json!(o.decision_path));
        if explain {
            m.insert(
                "explanation".into(),
                json!({
                    "top1_vs_top2": top1_vs_top2(o.local.as_ref()),
                    "decision_path": o.decision_path,
                }),
            );
        }
        Value::Object(m)
    }

    /// The router's complexity estimate of a question (spec §4.7b).
    pub fn complexity(&self, o: &QuestionOutcome, words: usize) -> Complexity {
        let label = match (&o.oracle, &o.local) {
            (Some(a), _) => a.label().map(str::to_string),
            (None, Some(l)) => l.choice.clone(),
            (None, None) => None,
        };
        let (conf, novelty, margin) = match (&o.oracle, &o.local) {
            (Some(_), Some(l)) => (1.0, l.decision.novelty, l.decision.margin),
            (Some(_), None) => (1.0, 1.0, 0.0),
            (None, Some(l)) => (l.decision.p_top, l.decision.novelty, l.decision.margin),
            (None, None) => (0.0, 1.0, 0.0),
        };
        estimate_complexity(&self.cfg, label.as_deref(), conf, novelty, margin, words)
    }

    fn routing(&self, o: &QuestionOutcome, c: &Complexity) -> Option<Value> {
        let target = self.cfg.routing_tiers.get(&c.tier)?;
        let label = match (&o.oracle, &o.local) {
            (Some(a), _) => a.label().map(str::to_string),
            (None, Some(l)) => l.choice.clone(),
            _ => None,
        }
        .unwrap_or_else(|| o.kind.as_str().to_string());
        Some(json!({
            "target": target,
            "reason": format!("{label} @ complexity {:.2} ({})", c.score, c.tier),
        }))
    }

    // ---------------------------------------------------------------- listings

    /// `GET /v1/models` in OpenRouter's legacy provider listing (spec §4.12).
    pub fn models_json(&self) -> Value {
        let model = self.handle.current();
        let tokenizer = model.encoder().encoder().tokenizer();
        let skills: Vec<Value> = model
            .skills()
            .iter()
            .map(|s| {
                let g = s.gate();
                let gate = &s.manifest.gate;
                let lb = gate
                    .evidence
                    .odd
                    .grid
                    .iter()
                    .find(|e| gate.certified && (e.t as f32) as f64 == gate.tau)
                    .map(|e| e.lb);
                json!({
                    "id": s.id(),
                    "labels": s.scorer.len(),
                    "certified": g.certified,
                    "tau": f32_json(g.tau),
                    "theta": f32_json(g.novelty_theta),
                    "temperature": f32_json(g.temperature),
                    "odd_half": {
                        "accepted": gate.evidence.odd.accepted,
                        "correct": gate.evidence.odd.correct,
                        "lb": lb,
                    },
                })
            })
            .collect();
        json!({
            "data": [{
                "id": MODEL_ID,
                "name": model.model().manifest().name,
                "created": model.model().manifest().created_unix,
                "description": "Typed decisions from one CMF file: a native BERT encoder, a hashing contract and certified resonance skills. It abstains when unsure (certified selective accuracy) and can escalate only the abstentions to an opt-in oracle.",
                "input_modalities": ["text"],
                "output_modalities": ["decisions"],
                "context_length": tokenizer.max_length(),
                "max_output_length": crate::protocol::MAX_CHOICE_OPTIONS,
                "quantization": "fp32",
                "hugging_face_id": "infosave/cortiq-decision",
                "pricing": {
                    "prompt": self.rates.input_per_1m.per_token_string(),
                    "completion": self.rates.output_per_1m.per_token_string(),
                    "request": self.rates.request.to_string(),
                    "image": "0",
                },
                "supported_sampling_parameters": [],
                "supported_features": [],
                "cmf": {
                    "model_sha": model.model_sha(),
                    "generation": model.generation(),
                    "skills": skills,
                },
            }]
        })
    }

    /// `GET /v1/skills`.
    pub fn skills_json(&self) -> Value {
        let model = self.handle.current();
        let skills: Vec<Value> = model.skills().iter().map(skill_summary).collect();
        json!({"model": model.name(), "generation": model.generation(), "skills": skills})
    }

    /// `GET /v1/skills/{id}` (404 for an unknown skill).
    pub fn skill_json(&self, id: &str) -> Result<Value, ApiError> {
        let model = self.handle.current();
        let s = model.skill(id).ok_or_else(|| {
            ApiError::not_found(format!("no skill {}", crate::config::quote_unless_key(id)))
        })?;
        let mut v = skill_summary(s);
        let tasks: Vec<Value> = s
            .manifest
            .tasks
            .iter()
            .map(|t| {
                json!({"i": t.i, "label": t.label, "state": t.state, "origin": t.origin, "k": t.k, "n_train": t.n_train})
            })
            .collect();
        v["tasks"] = Value::Array(tasks);
        v["rubric"] = match &s.manifest.rubric {
            Some(r) => {
                json!({"instructions": r.instructions, "criteria": Value::Object(r.ordered_criteria())})
            }
            None => Value::Null,
        };
        v["gate"] = serde_json::to_value(&s.manifest.gate).unwrap_or(Value::Null);
        Ok(v)
    }

    /// `GET /healthz`.
    pub fn healthz_json(&self) -> Value {
        let model = self.handle.current();
        json!({
            "status": "ok",
            "model": model.name(),
            "model_sha": model.model_sha(),
            "generation": model.generation(),
            "skills": model.skills().len(),
            "auto_skills": model.auto_skills(),
            "inflight": self.inflight(),
            "encoder_device": model.encoder().encoder().device_name(),
            "metal_encoder_submissions": model.encoder().encoder().metal_submissions(),
            "metal_resonance_submissions": model.skills().iter().map(|s| s.scorer().packed().metal_submissions()).sum::<u64>(),
            "vulkan_encoder_submissions": model.encoder().encoder().vulkan_submissions(),
            "vulkan_resonance_submissions": model.skills().iter().map(|s| s.scorer().packed().vulkan_submissions()).sum::<u64>(),
            "oracle": self.escalator.is_some() && self.cfg.oracle.enabled,
            "oracle_status": self.oracle_status().label(),
        })
    }

    /// `GET /v1/usage`: the caller's account only.
    pub fn usage_json(&self, p: &Principal) -> Value {
        let t = self.totals(&p.account);
        json!({
            "account": p.account,
            "plan": p.plan,
            "usage": t.to_json(),
            "limits": {
                "rate_per_min": p.rate_per_min,
                "decision_quota": p.decision_quota,
                "token_quota": p.token_quota,
                "credit_usd": p.credit_usd.map(|c| c.to_string()),
                "oracle_budget_usd": p.oracle_budget_usd.map(|c| c.to_string()),
                "oracle_allowed": p.oracle_allowed,
            },
        })
    }

    /// `POST /v1/feedback` (spec §5.11), delegated to the escalator.
    pub fn feedback(&self, body: &[u8], p: &Principal) -> Result<Value, ApiError> {
        let fb = parse_feedback(body, &self.limits)?;
        self.feedback_request(&fb, p)
    }

    /// A parsed feedback (the router API builds its own), delegated to the
    /// escalator.
    pub fn feedback_request(&self, fb: &FeedbackRequest, p: &Principal) -> Result<Value, ApiError> {
        match &self.escalator {
            Some(e) => e.feedback(fb, p),
            None => Err(ApiError::not_found(format!(
                "no pending decision {} / {} (learning is not enabled on this server)",
                fb.id, fb.question
            ))),
        }
    }

    // ---------------------------------------------------------------- admin

    /// The admin guard (spec §5b): 404 `ADMIN_DISABLED` without a token in the
    /// environment, 401 for a wrong `x-admin-token` (constant-time compare).
    pub fn admin_authorize(&self, token: Option<&str>) -> Result<(), ApiError> {
        use subtle::ConstantTimeEq;
        let Some(expected) = &self.admin_token else {
            return Err(ApiError::new(
                Reason::AdminDisabled,
                format!(
                    "the admin API is disabled (set {})",
                    self.cfg.auth.admin_token_env
                ),
            ));
        };
        let given = token.unwrap_or("").as_bytes();
        let ok = given.len() == expected.len() && bool::from(given.ct_eq(expected.as_bytes()));
        if ok {
            Ok(())
        } else {
            Err(ApiError::new(
                Reason::Unauthorized,
                "invalid or missing x-admin-token",
            ))
        }
    }

    fn key_store(&self) -> Result<&Arc<KeyStore>, ApiError> {
        self.keys
            .as_ref()
            .ok_or_else(|| ApiError::not_found("key management needs a state directory"))
    }

    /// `POST /v1/admin/keys`: create a key; the raw key is in this response only.
    pub fn admin_create_key(&self, body: &[u8]) -> Result<Value, ApiError> {
        let v = crate::protocol::parse_json(body)?;
        let new: NewKey = serde_json::from_value(v)
            .map_err(|e| ApiError::invalid(format!("key request: {e}")))?;
        let now = now_unix();
        let created = self
            .key_store()?
            .create(&new, &self.cfg.auth.plans, now)
            .map_err(|e| ApiError::invalid(format!("{e:#}")))?;
        Ok(created.to_json(now))
    }

    /// `GET /v1/admin/keys`: hash12, limits and usage of every key, as it is
    /// now (an imported router key may answer as its static configuration key,
    /// [`KeyRecord::effective`]).
    pub fn admin_list_keys(&self) -> Result<Value, ApiError> {
        let now = now_unix();
        let keys: Vec<Value> = self
            .key_store()?
            .records()
            .iter()
            .map(|k| {
                let k = k.effective(now);
                let mut v = k.listing(now);
                v["usage"] = self.totals(&k.account).to_json();
                v
            })
            .collect();
        Ok(json!({"count": keys.len(), "keys": keys}))
    }

    /// `DELETE /v1/admin/keys/{account}`.
    pub fn admin_revoke_account(&self, account: &str) -> Result<Value, ApiError> {
        let n = self
            .key_store()?
            .revoke_account(account)
            .map_err(internal)?;
        Ok(json!({"account": account, "revoked": n}))
    }

    /// `DELETE /v1/admin/keys/hash/{hash12}`.
    pub fn admin_revoke_hash(&self, hash_prefix: &str) -> Result<Value, ApiError> {
        let n = self
            .key_store()?
            .revoke_hash_prefix(hash_prefix)
            .map_err(|e| ApiError::invalid(format!("{e:#}")))?;
        Ok(json!({"hash": hash_prefix, "revoked": n}))
    }

    /// `GET /v1/admin/usage`: every account.
    pub fn admin_usage(&self) -> Value {
        let all = match &self.ledger {
            Some(l) => l.all_totals(),
            None => self.memory_totals.lock().clone(),
        };
        let accounts: Map<String, Value> =
            all.iter().map(|(a, t)| (a.clone(), t.to_json())).collect();
        json!({"accounts": Value::Object(accounts)})
    }

    /// The escalator's admin operations (404 without a cascade).
    pub fn admin(&self, command: &AdminCommand) -> Result<Value, ApiError> {
        match &self.escalator {
            Some(e) => e.admin(command),
            None => Err(ApiError::not_found(
                "the oracle cascade is not configured on this server",
            )),
        }
    }
}

/// A pinned `cortiq/decision@<12 hex>` must name the served model (404).
fn check_pinned(model: &LoadedModel, req: &DecisionRequest) -> Result<(), ApiError> {
    if let ModelRef::Pinned(sha) = &req.model
        && sha != model.sha12()
    {
        return Err(ApiError::new(
            Reason::ModelNotFound,
            format!(
                "model {MODEL_ID}@{sha} is not served (current: {})",
                model.name()
            ),
        )
        .with_detail("current", json!(model.name())));
    }
    Ok(())
}

/// Every question matched to a skill of `model` (spec §4.4).
fn match_questions(
    model: &LoadedModel,
    req: &DecisionRequest,
) -> Result<Vec<SkillMatch>, ApiError> {
    let labels = model.skill_labels();
    req.questions
        .iter()
        .map(|q| match_question(&labels, q, req.cmf.skill.as_deref()))
        .collect()
}

/// The encoder and hash of a request (once) and the local decision of each
/// exact or subset question (the errors of a skill once per request).
struct LocalStage {
    features: Features,
    signal: crate::signal::Timings,
    locals: Vec<Option<LocalDecision>>,
    resonance: Duration,
}

fn local_stage(
    model: &LoadedModel,
    req: &DecisionRequest,
    matches: &[SkillMatch],
    auto_tau: f32,
) -> Result<LocalStage, ApiError> {
    let mut indices = Vec::new();
    for m in matches.iter().filter(|m| m.kind.is_local()) {
        let sid = m.skill.as_deref().unwrap_or_default();
        let si = model
            .skill_index(sid)
            .ok_or_else(|| internal(format!("matched skill '{sid}' is missing")))?;
        if !indices.contains(&si) {
            indices.push(si);
        }
    }
    let packed: Vec<_> = indices
        .iter()
        .map(|&i| model.skills[i].scorer.packed())
        .collect();
    let scored = model
        .encoder()
        .score_timed(&req.state_text, &packed)
        .map_err(internal)?;
    let features = scored.features;
    let signal = scored.timings;
    let tr = Instant::now();
    let skill_errors: HashMap<usize, Vec<f32>> = indices.into_iter().zip(scored.errors).collect();
    let mut locals: Vec<Option<LocalDecision>> = Vec::with_capacity(matches.len());
    for m in matches {
        if !m.kind.is_local() {
            locals.push(None);
            continue;
        }
        let sid = m.skill.as_deref().unwrap_or_default();
        let si = model
            .skill_index(sid)
            .ok_or_else(|| internal(format!("matched skill '{sid}' is missing")))?;
        locals.push(Some(
            local_decision(
                &model.skills[si],
                m,
                &skill_errors[&si],
                req.cmf.profile,
                req.state.is_text(),
                auto_tau,
            )
            .map_err(internal)?,
        ));
    }
    Ok(LocalStage {
        features,
        signal,
        locals,
        resonance: scored.resonance + tr.elapsed(),
    })
}

/// The local decision of an exact or subset question over its candidates.
/// `auto_tau` is the confidence floor of an auto-skill (`learning.auto_tau`,
/// DESIGN D5/A8): its gate's τ stays 0 while uncertified, so `balanced` and
/// `quality-first` additionally require `p_top ≥ auto_tau` (`cost-saver` is
/// θ-only by spec §4.7b); an abstention escalates and teaches as any other.
fn local_decision(
    s: &SkillRuntime,
    m: &SkillMatch,
    all_errors: &[f32],
    profile: Profile,
    state_is_text: bool,
    auto_tau: f32,
) -> Result<LocalDecision> {
    let scorer = &s.scorer;
    let gate = scorer.gate();
    let exact = m.kind == MatchKind::Exact;
    let (errors, stats, labels, tasks): (Vec<f32>, Vec<_>, Vec<String>, Vec<usize>) = if exact {
        (
            all_errors.to_vec(),
            scorer.stats().to_vec(),
            scorer.labels().to_vec(),
            scorer.tasks().to_vec(),
        )
    } else {
        (
            m.candidates.iter().map(|&c| all_errors[c]).collect(),
            m.candidates.iter().map(|&c| scorer.stats()[c]).collect(),
            m.candidates
                .iter()
                .map(|&c| scorer.labels()[c].clone())
                .collect(),
            m.candidates.iter().map(|&c| scorer.tasks()[c]).collect(),
        )
    };
    let decision = decide_errors(&errors, &stats, gate.temperature)?;
    let gate_accepted = decision.accepted(gate.tau, gate.novelty_theta);
    let floor = !s.manifest.is_auto() || decision.p_top >= auto_tau;
    let accepted = match profile {
        Profile::Balanced => gate_accepted && floor,
        Profile::QualityFirst => {
            gate_accepted
                && floor
                && decision.margin >= QUALITY_FIRST_MARGIN
                && decision.novelty <= gate.novelty_theta.min(QUALITY_FIRST_NOVELTY)
        }
        Profile::CostSaver => decision.winner.is_some() && !decision.is_novel(gate.novelty_theta),
    };
    let winner_from_data = decision.winner.is_some_and(|w| {
        let c = if exact { w } else { m.candidates[w] };
        scorer.origins()[c] == TaskOrigin::Data
    });
    let certified = exact
        && state_is_text
        && profile != Profile::CostSaver
        && gate.certified
        && winner_from_data;
    let choice = decision.winner.map(|w| labels[w].clone());
    let confidence = if decision.winner.is_some() {
        jev_confidence(decision.p_top, labels.len())
    } else {
        0.0
    };
    Ok(LocalDecision {
        skill: s.id().to_string(),
        labels,
        tasks,
        errors,
        decision,
        choice,
        confidence,
        gate_accepted,
        accepted,
        certified,
        gate,
        profile,
    })
}

/// The choice answer of a local decision: every option in request order.
fn local_answer(q: &Question, l: &LocalDecision, rounding: Rounding) -> Value {
    let options = q.options();
    let probs: Vec<(&str, f32)> = options
        .iter()
        .map(|o| (*o, l.probability(o).unwrap_or(0.0)))
        .collect();
    answer::choice_answer(
        &probs,
        l.choice.as_deref().unwrap_or_default(),
        answer::answer_confidence(l.decision.p_top, options.len()),
        rounding,
    )
}

fn local_answer_or_null(q: &Question, l: Option<&LocalDecision>, rounding: Rounding) -> Value {
    l.map_or(Value::Null, |l| local_answer(q, l, rounding))
}

/// `"<a> leads <b> by <Δscore> score"` (router `api.rs:1141-1165`).
fn top1_vs_top2(l: Option<&LocalDecision>) -> String {
    let Some(l) = l else {
        return "no candidates".into();
    };
    let r = &l.decision.ranked;
    match r.as_slice() {
        [a, b, ..] => format!(
            "{} leads {} by {:.3} score",
            l.labels[a.index],
            l.labels[b.index],
            a.score - b.score
        ),
        [a] => format!("{} only candidate", l.labels[a.index]),
        [] => "no candidates".into(),
    }
}

/// The 422/502/503 of untrained questions without a verdict.
fn untrained_error(
    req: &DecisionRequest,
    matches: &[SkillMatch],
    unresolved: &[(usize, Resolution)],
) -> ApiError {
    let mut details = Map::new();
    let mut worst = Reason::UnsupportedQuestion;
    let rank = |r: Reason| match r {
        Reason::OracleUnavailable => 3,
        Reason::OracleBudgetExhausted => 2,
        Reason::OracleDisabled => 1,
        _ => 0,
    };
    for (i, res) in unresolved {
        let m = &matches[*i];
        let (reason, oracle) = match res {
            Resolution::Failed(_) => (Reason::OracleUnavailable, "the oracle call failed"),
            Resolution::Refused(RefusalReason::Budget) => (
                Reason::OracleBudgetExhausted,
                "the oracle budget left cannot hold the call",
            ),
            Resolution::Refused(RefusalReason::Stopped) => {
                (Reason::OracleDisabled, "the oracle is stopped")
            }
            Resolution::Refused(RefusalReason::ConsentOff) => (
                Reason::UnsupportedQuestion,
                "the oracle is not allowed for this request or key",
            ),
            Resolution::Refused(RefusalReason::NoKey) => (
                Reason::UnsupportedQuestion,
                "the oracle key is not set in the server's environment",
            ),
            Resolution::Refused(RefusalReason::BadKey) => (
                Reason::UnsupportedQuestion,
                "the oracle key in the server's environment is not usable",
            ),
            Resolution::Refused(RefusalReason::OracleDisabled)
            | Resolution::Oracle(_)
            | Resolution::Cache(_) => (Reason::UnsupportedQuestion, "the oracle is disabled"),
        };
        if rank(reason) > rank(worst) {
            worst = reason;
        }
        let why = match (&m.reason, m.kind) {
            (Some(r), _) => r.clone(),
            (None, MatchKind::Superset) => format!(
                "options {} are not trained in skill '{}'",
                m.unknown.join(", "),
                m.skill.as_deref().unwrap_or_default()
            ),
            (None, k) => format!("{} match", k.as_str()),
        };
        details.insert(
            req.questions[*i].id.clone(),
            json!({"match": m.kind.as_str(), "skill": m.skill, "reason": why, "oracle": oracle}),
        );
    }
    let message = match worst {
        Reason::OracleUnavailable => {
            "the oracle failed for questions the local model cannot answer"
        }
        Reason::OracleBudgetExhausted => {
            "the oracle budget left cannot hold the call and the local model cannot answer these questions"
        }
        Reason::OracleDisabled => {
            "the oracle is stopped and the local model cannot answer these questions"
        }
        _ => "the local model cannot answer these questions and the oracle is not allowed",
    };
    ApiError::new(worst, message).with_detail("questions", Value::Object(details))
}

/// `GET /v1/skills` entry (keys append-only: clients index the listing by
/// position). `auto`, `active_labels`, `quarantined_labels` and `examples`
/// (learned rows) are 0.8.6 additions, present for every skill.
fn skill_summary(s: &SkillRuntime) -> Value {
    let g = s.gate();
    let m = &s.manifest;
    let quarantined: Vec<&str> = m
        .tasks
        .iter()
        .filter(|t| !t.is_active())
        .map(|t| t.label.as_str())
        .collect();
    json!({
        "id": s.id(),
        "taxonomy_version": m.taxonomy_version,
        "labels": s.scorer.labels(),
        "certified": g.certified,
        "tau": f32_json(g.tau),
        "theta": f32_json(g.novelty_theta),
        "temperature": f32_json(g.temperature),
        "has_rubric": m.rubric.is_some(),
        "auto": m.is_auto(),
        "active_labels": s.scorer.labels().len(),
        "quarantined_labels": quarantined,
        "examples": m.rows_learned.as_ref().map_or(0, |r| r.n),
    })
}

/// The complexity estimate (router `estimate_complexity`, `api.rs:1082-1134`).
#[derive(Clone, Debug, PartialEq)]
pub struct Complexity {
    pub score: f32,
    pub tier: String,
    pub base: f32,
    pub ambiguity: f32,
    pub novelty: f32,
    pub margin: f32,
    pub length: f32,
}

impl Complexity {
    pub fn to_json(&self) -> Value {
        json!({
            "score": f32_json(self.score),
            "tier": self.tier,
            "factors": {
                "base": f32_json(self.base),
                "ambiguity": f32_json(self.ambiguity),
                "novelty": f32_json(self.novelty),
                "margin": f32_json(self.margin),
                "length": f32_json(self.length),
            },
        })
    }
}

/// Router formula: base = `task_complexity[label]` (0.4), ambiguity = 1 −
/// confidence, margin = 1 − clamp(8·margin, 0, 1), length = clamp(words/40,
/// 0, 1); score = clamp(Σ w·factor, 0, 1); the first tier whose max ≥ score.
pub fn estimate_complexity(
    cfg: &Config,
    label: Option<&str>,
    confidence: f32,
    novelty: f32,
    margin: f32,
    words: usize,
) -> Complexity {
    let base = label
        .and_then(|l| cfg.task_complexity.get(l).copied())
        .unwrap_or(DEFAULT_TASK_COMPLEXITY);
    let amb = 1.0 - confidence;
    let mrg = 1.0 - (margin * 8.0).clamp(0.0, 1.0);
    let len = (words as f32 / 40.0).clamp(0.0, 1.0);
    let w = &cfg.complexity_weights;
    let score =
        (w.base * base + w.ambiguity * amb + w.novelty * novelty + w.margin * mrg + w.length * len)
            .clamp(0.0, 1.0);
    let tier = cfg
        .complexity_tiers
        .iter()
        .find(|b| score <= b.max)
        .or(cfg.complexity_tiers.last())
        .map_or_else(|| "high".to_string(), |b| b.tier.clone());
    Complexity {
        score,
        tier,
        base,
        ambiguity: amb,
        novelty,
        margin: mrg,
        length: len,
    }
}
