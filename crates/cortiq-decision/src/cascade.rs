//! Oracle cascade, `cascade::Cascade` implementing `Escalator` (spec §5).
//!
//! The service calls [`Cascade`] only for undetermined questions (the gate
//! rejected them, or they are untrained) and only after its own consent checks
//! (`oracle.enabled`, the key's `oracle_allowed`, `cmf.oracle` /
//! `default_per_request`); without that consent it asks the cache alone
//! (0.8.10, step 1). A question the gate accepted never reaches it (spec
//! §5.1). For one request (spec §5.2):
//! 1. **permission**: the admin switch, a key in the environment, no stop
//!    reason, a budget left (globally, calls, the key's `oracle_budget_usd`
//!    and what its `credit_usd` has left); otherwise (0.8.10) every question
//!    the cache answers is answered from it — no call, nothing sent — and
//!    the others are refused (`oracle_disabled`, `no_key`, `stopped`,
//!    `budget`). A request the service's own consent checks refused
//!    (`oracle.enabled`, `oracle_allowed`, `cmf.oracle`) gets the same
//!    cache-only answer ([`Escalator::resolve_without_oracle`]): a cache hit
//!    has no cost and no egress;
//! 2. **cache** ([`crate::cache`]): a hit answers the question at no cost —
//!    by default only the same question (scope and input digest) hits;
//!    near reuse at cos φ_P ≥ `cache.threshold` is opt-in (0.8.10);
//! 3. **single flight**: a question of the same scope as one in flight in
//!    another request, with the same input (or, near reuse on, cos φ_P ≥
//!    `cache.threshold`: [`crate::cache::reuses`]), waits for that call and
//!    reuses its answer (as a cache answer); the others lead. A follower of
//!    the same input puts the answer in the cache and `learn.log` as an
//!    oracle answer is (0.8.10; it dedups with its leader's put, whichever
//!    comes first). A follower of another input (near reuse) stores nothing:
//!    the oracle never read its input, and an entry under its digest would
//!    answer it exactly — after near reuse is turned off as well;
//! 4. **one call** for every leading question ([`crate::oracle`]); the state,
//!    the questions' instructions and their criteria's descriptions (0.8.8,
//!    DESIGN B4; never the option ids) are PII-redacted when
//!    `oracle.redact_pii` is on and the request did not set
//!    `cmf.allow_pii_egress` (flag `pii_redacted`); the cache scope, the
//!    contract key and the learned example keep the question as asked;
//! 5. **success**: the answers are cached (and logged; the scope holds the
//!    question's contract, so an answer is reused only for the same
//!    instructions and criteria), and a choice answer of a question matched
//!    to a skill becomes an example of the learning buffer (source oracle)
//!    when it may teach the shared skill: the caller's key has
//!    `learning_allowed`, or the question is an exact match that asks exactly
//!    the skill's own question ([`crate::service::SkillRuntime::follows_rubric`]
//!    — the router's `task` question is one) so that the answer is the
//!    skill's rubric applied to the text, not the caller's instructions
//!    (never for the implicit open mode of a loopback address, `serve
//!    --oracle` without keys: [`Principal::implicit_open`] — its answers are
//!    only cached); an example that brings its label to `learning.refit_min_new` new examples
//!    starts a learning attempt ([`crate::learn::attempt`]) — inline with
//!    `learning.synchronous`, else on the one background worker;
//! 6. **failure**: the question is failed (the service answers a trained one
//!    locally with `abstain` + `oracle_unavailable`, an untrained one with 502).
//!
//! **Feedback** (spec §5.11): every decided question matched to a skill is kept
//! in a ring of `feedback.pending_cap` entries (vectors only). `POST
//! /v1/feedback` finds the caller's own entry (another account's is not found),
//! takes a label among the question's options and adds an example of weight 3
//! (a label the skill does not have starts a cold start) — only for a caller
//! whose key has `learning_allowed` (the open mode with `auth.require: false`
//! has it); anyone else's feedback is consumed and answered, but teaches
//! nothing (`learned: false`).
//!
//! **Buffer limits**: at most [`MAX_EXAMPLES_PER_LABEL`] examples of one
//! (skill, label) and [`MAX_PENDING_NEW_LABELS`] labels per skill that are
//! not tasks of it yet (pending cold starts); an example past either limit is
//! refused (`full`).
//!
//! **Auto-skills** (0.8.6, DESIGN D1–D3, A4, A5, A7): a choice question no
//! skill fits (`match: untrained`, not ambiguous, no `cmf.skill`) is a
//! *contract*; when the caller's key has `learning_allowed` (only that key
//! teaches an auto-skill, before and after its activation — the rubric rule
//! of `teaches` does not apply to `auto-` ids, since the stored rubric is the
//! contract itself) and `learning.auto_skills` is on, the oracle's answer
//! becomes an example of the auto-skill `auto-<sha12 of the contract>` — the
//! contract being the question's instructions and criteria with their
//! descriptions, not its id set (DESIGN A18, [`crate::manifest::auto_skill_id`]),
//! so the same ids under other instructions, or with a changed description,
//! are another contract. The contract is registered once,
//! under the buffer lock, as a [`LogRecord::Contract`] written before its
//! first example; the served model is the registry's second source: an
//! auto-skill it carries without a record (a materialised file on a fresh
//! state directory, a lost `learn.log`) is registered from its manifest at
//! open and after a rollback, so it keeps learning. A contract with fewer
//! than 2 or more than `learning.auto_max_labels` ids, or past
//! `learning.auto_max_skills` stateful (`auto_max_stateless_skills`
//! state-less) contracts, is answered by the oracle and not recorded
//! (`auto_skipped`).
//! The labels of an auto-skill are closed: an example whose label is not one
//! of the contract's ids is refused (`full`), so a superset request or a
//! router feedback can never grow it; the pending-labels cap does not apply
//! to it, `learning.auto_max_examples_per_label` replaces the per-label cap.
//! The trigger is the same counter; the attempt is [`crate::learn`]'s
//! whole-skill one. While a label stays quarantined the service *explores*
//! (DESIGN A16, [`crate::service::LocalDecision::explore`]): one accepted
//! text in `learning.auto_explore_every` of the contract arrives here as a
//! pending question like an abstention, is answered by the oracle and
//! learned under `teaches` — the one gate-accepted question that reaches
//! the oracle, so a label the gate confidently misnames still collects its
//! examples.
//!
//! **State-less contracts** (0.8.8, DESIGN A19/A20): a question of a request
//! with an empty `state` reads its own instructions
//! ([`crate::protocol::DecisionRequest::reads_instructions`]): its φ_P is its
//! instructions' (the cache, single flight and its example use it), its cache
//! scope and its auto-skill hash the criteria alone
//! ([`crate::cache::scope_of_as`], [`Contract::stateless`]), and its
//! instructions leave for the oracle PII-redacted like a state. Such a
//! contract is registered only from its `learning.auto_min_sightings`-th
//! sighting (an escalation of a learnable question of it; counted in a
//! bounded in-memory LRU of `learning.auto_sightings_cap` contracts, lost on
//! restart), so one-off contracts — a multiple-choice benchmark whose options
//! change with every item — never reach `learn.log` nor use up
//! `learning.auto_max_stateless_skills`; the answers before that are not
//! learned. Stateful contracts register at their
//! `learning.auto_min_sightings_stateful`-th sighting (1 by default: the
//! first, as in 0.8.6; DESIGN B2; counted in the same LRU, the contract id
//! tells the kinds apart), against their own cap `learning.auto_max_skills`,
//! so stateful one-offs never crowd a repeated state-less contract out.
//!
//! **Admin** (spec §5b): oracle status and switches, learning status (buffer,
//! cache, quarantine, attempts, task hashes), generations and rollback (the
//! buffer is kept, the counters restart).
//!
//! **State** (`--state DIR`): `oracle.jsonl`, `oracle.state`, `learn.log`,
//! `generations/`, `CURRENT`. At start the cache and the buffer are rebuilt from
//! `learn.log` and the budget from `oracle.jsonl`.

use crate::answer::OracleAnswer;
use crate::buffer::{
    AddOutcome, Contract, ContractRegistry, Example, LearnLog, LearningBuffer, LogRecord,
};
use crate::cache::{
    CacheEntry, InputDigest, SemanticCache, input_digest, reuses, scope_of_as, state_digest,
};
use crate::config::Config;
use crate::container::{DecisionModel, Verify};
use crate::generation;
use crate::learn::{self, AttemptReport, Books, LearnContext, Outcome};
use crate::manifest;
use crate::matching::{DescriptionMap, MatchKind};
use crate::metering::Usd;
use crate::oracle::{CallOutcome, Caller, KeyLookup, OracleClient, process_env};
use crate::pii::{FLAG_PII_REDACTED, redact_value};
use crate::protocol::{ApiError, FeedbackRequest, MAX_LABEL_BYTES, Question, QuestionKind};
use crate::rows::{Rows, Source};
use crate::service::{
    AdminCommand, Escalation, EscalationResult, Escalator, ModelHandle, Observation, OracleStatus,
    OracleUsage, Pending, Principal, RefusalReason, Resolution, Resolved,
};
use crate::statedir::StateDir;
use anyhow::{Context, Result, ensure};
use parking_lot::{Condvar, Mutex};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Attempts kept for `GET /v1/admin/learning`.
pub const RECENT_ATTEMPTS: usize = 32;
/// Extra wait of a single-flight follower beyond the oracle deadline.
pub const FOLLOWER_GRACE: Duration = Duration::from_secs(5);
/// `refused` of a feedback answer to a caller without `learning_allowed`.
pub const LEARNING_NOT_ALLOWED: &str = "learning_not_allowed";
/// Most examples of one (skill, label) in the learning buffer.
pub const MAX_EXAMPLES_PER_LABEL: usize = 5_000;
/// Most labels per skill in the buffer that are not tasks of the skill yet.
pub const MAX_PENDING_NEW_LABELS: usize = 32;
/// How often a WARN names a contract that was not learned (`auto_skipped`).
pub const AUTO_SKIPPED_LOG_EVERY: Duration = Duration::from_secs(3600);

/// Options of [`Cascade::open_with`].
#[derive(Clone)]
pub struct CascadeOptions {
    /// Where the oracle key is read from (the process environment by default).
    pub key: KeyLookup,
    /// Threads of a learning attempt (0: all cores).
    pub threads: usize,
    /// `created_unix` of new generations (`None`: `SOURCE_DATE_EPOCH` or 0).
    pub created_unix: Option<u64>,
}

impl Default for CascadeOptions {
    fn default() -> Self {
        Self {
            key: process_env(),
            threads: 0,
            created_unix: None,
        }
    }
}

impl std::fmt::Debug for CascadeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CascadeOptions")
            .field("threads", &self.threads)
            .field("created_unix", &self.created_unix)
            .finish()
    }
}

// ------------------------------------------------------------------ single flight

#[derive(Debug, Default)]
struct Slot {
    result: Mutex<Option<Resolution>>,
    ready: Condvar,
}

impl Slot {
    fn set(&self, r: Resolution) {
        let mut g = self.result.lock();
        if g.is_none() {
            *g = Some(r);
        }
        self.ready.notify_all();
    }

    fn wait(&self, timeout: Duration) -> Option<Resolution> {
        let deadline = Instant::now() + timeout;
        let mut g = self.result.lock();
        while g.is_none() {
            if self.ready.wait_until(&mut g, deadline).timed_out() {
                break;
            }
        }
        g.clone()
    }
}

#[derive(Debug)]
struct Flight {
    id: u64,
    scope: String,
    input: InputDigest,
    phi_p: Vec<f32>,
    slot: Arc<Slot>,
}

/// What the cache knows a pending question by, in pending order: φ_P (the
/// state's, or in a state-less request its instructions', DESIGN A19.4), the
/// scope and the input digest ([`crate::cache`]).
struct QuestionKeys<'a> {
    phis: Vec<&'a [f32]>,
    scopes: Vec<String>,
    inputs: Vec<InputDigest>,
}

impl<'a> QuestionKeys<'a> {
    fn of(e: &'a Escalation<'_>) -> Self {
        let mut state: Option<InputDigest> = None;
        let mut keys = Self {
            phis: Vec::with_capacity(e.pending.len()),
            scopes: Vec::with_capacity(e.pending.len()),
            inputs: Vec::with_capacity(e.pending.len()),
        };
        for p in &e.pending {
            let reads = e.request.reads_instructions(p.question);
            keys.phis.push(&e.features_of(p.index).phi_p);
            keys.scopes.push(scope_of_as(p.question, p.matched, reads));
            keys.inputs.push(if reads {
                input_digest(e.request, p.question)
            } else {
                *state.get_or_insert_with(|| state_digest(e.request))
            });
        }
        keys
    }
}

/// The questions answered from `cache` alone (`None`: the cache is off) —
/// no call, nothing sent, no single flight, nothing learned — the others
/// refused with `reason`; and the positions of those misses.
fn answer_from(
    cache: Option<&Mutex<SemanticCache>>,
    keys: &QuestionKeys<'_>,
    reason: RefusalReason,
) -> (Vec<Resolved>, Vec<usize>) {
    let mut cache = cache.map(|c| c.lock());
    let mut misses = Vec::new();
    let resolved = (0..keys.scopes.len())
        .map(|i| {
            let hit = cache
                .as_mut()
                .and_then(|c| c.get(&keys.scopes[i], &keys.inputs[i], keys.phis[i]));
            match hit {
                Some((a, _)) => Resolved::new(Resolution::Cache(a)),
                None => {
                    misses.push(i);
                    Resolved::new(Resolution::Refused(reason))
                }
            }
        })
        .collect();
    (resolved, misses)
}

/// A state directory's cache of oracle answers, read only (0.8.10): what
/// `cortiq decide --oracle` without a usable key answers from — the oracle's
/// answers to earlier runs (or a server) on that directory, by the rules of
/// the cascade's cache — with no call, no lock taken and nothing written.
pub struct CacheView {
    /// `None`: `cache.enabled` is off.
    cache: Option<Mutex<SemanticCache>>,
}

impl CacheView {
    /// The cache puts of `learn_log` replayed under `cfg` (a missing file is
    /// an empty cache; a cut or corrupt tail — a writer's last record — is
    /// skipped, the file left as it is).
    pub fn load(cfg: &crate::config::CacheConfig, learn_log: &Path) -> Result<Self> {
        if !cfg.enabled {
            return Ok(Self { cache: None });
        }
        let bytes = match std::fs::read(learn_log) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                return Err(e).with_context(|| format!("read {}", learn_log.display()));
            }
        };
        let mut cache = SemanticCache::from_config(cfg);
        for r in crate::buffer::read_records(&bytes).0 {
            if let LogRecord::CachePut(e) = r {
                cache.put(e);
            }
        }
        Ok(Self {
            cache: Some(Mutex::new(cache)),
        })
    }

    /// Entries of the cache.
    pub fn len(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.lock().len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The escalation's questions the cache holds are answered from it
    /// (`cache`), the others refused with `reason`
    /// ([`Escalator::resolve_without_oracle`] of the cascade).
    pub fn resolve(&self, e: &Escalation<'_>, reason: RefusalReason) -> EscalationResult {
        let (resolved, _) = answer_from(self.cache.as_ref(), &QuestionKeys::of(e), reason);
        EscalationResult {
            resolved,
            usage: OracleUsage::default(),
        }
    }
}

/// Removes a request's flights from the registry and resolves them (failed,
/// unless the leader resolved them first) whatever happens.
struct FlightGuard<'a> {
    flights: &'a Mutex<Vec<Flight>>,
    mine: Vec<(u64, Arc<Slot>)>,
}

impl Drop for FlightGuard<'_> {
    fn drop(&mut self) {
        let mut fl = self.flights.lock();
        fl.retain(|f| !self.mine.iter().any(|(id, _)| *id == f.id));
        drop(fl);
        for (_, s) in &self.mine {
            s.set(Resolution::Failed("single-flight leader ended".into()));
        }
    }
}

// ------------------------------------------------------------------ feedback ring

/// Sparse features of a decided state (shared by its questions).
#[derive(Debug)]
struct SparseState {
    phi_p: Vec<f32>,
    h_idx: Vec<u16>,
    h_val: Vec<f32>,
}

#[derive(Debug)]
struct PendingEntry {
    request_id: String,
    question: String,
    account: String,
    skill: String,
    options: Vec<String>,
    /// A description match's option labels (DESIGN C2): feedback names an
    /// option id, the example its label.
    by_descriptions: Option<DescriptionMap>,
    state: Arc<SparseState>,
}

// ------------------------------------------------------------------ stats

#[derive(Debug, Default)]
struct Stats {
    attempts: u64,
    promotions: u64,
    rejections: u64,
    cold_starts: u64,
    skipped: u64,
    errors: u64,
    isolation_violations: u64,
    rollbacks: u64,
    examples: u64,
    feedback: u64,
    /// Untrained contracts not learned (too many or too few ids, the
    /// registry full).
    auto_skipped: u64,
    /// Contracts this process registered (DESIGN A20).
    auto_registered: u64,
    auto_skipped_logged: Option<Instant>,
    recent: VecDeque<Value>,
}

/// How often each contract under a sightings gate was seen before it was
/// registered (DESIGN A20, B2): a bounded LRU of contract ids, oldest touched
/// evicted first. A state-less and a stateful contract never share an id.
#[derive(Debug, Default)]
pub struct Sightings {
    cap: usize,
    tick: u64,
    counts: HashMap<String, (u32, u64)>,
    order: std::collections::BTreeMap<u64, String>,
}

impl Sightings {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            ..Self::default()
        }
    }

    /// Count one sighting of `key` (it becomes the most recent); the count
    /// after it. Past `cap` keys the least recently seen is forgotten.
    pub fn see(&mut self, key: &str) -> u32 {
        self.tick += 1;
        let tick = self.tick;
        let n = match self.counts.get_mut(key) {
            Some((n, t)) => {
                self.order.remove(t);
                *n = n.saturating_add(1);
                *t = tick;
                *n
            }
            None => {
                self.counts.insert(key.to_string(), (1, tick));
                1
            }
        };
        self.order.insert(tick, key.to_string());
        while self.counts.len() > self.cap {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.counts.remove(&oldest);
        }
        n
    }

    /// The sightings of `key` (0 when unseen or forgotten).
    pub fn count(&self, key: &str) -> u32 {
        self.counts.get(key).map_or(0, |c| c.0)
    }

    /// Forget `key` (registered: its sightings no longer matter).
    pub fn forget(&mut self, key: &str) {
        if let Some((_, t)) = self.counts.remove(key) {
            self.order.remove(&t);
        }
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

struct Inner {
    cfg: Config,
    handle: Arc<ModelHandle>,
    state: StateDir,
    oracle: OracleClient,
    cache: Mutex<SemanticCache>,
    buffer: Mutex<LearningBuffer>,
    log: LearnLog,
    books: Mutex<Books>,
    /// The contracts of the auto-skills (restored from `learn.log` first).
    contracts: Mutex<ContractRegistry>,
    /// Sightings of gated contracts not registered yet (DESIGN A20, B2).
    sightings: Mutex<Sightings>,
    bases: Mutex<HashMap<String, Arc<Rows>>>,
    flights: Mutex<Vec<Flight>>,
    next_flight: AtomicU64,
    /// Learning jobs sent to the worker and not finished.
    queued: AtomicUsize,
    pending: Mutex<VecDeque<PendingEntry>>,
    learn_lock: Mutex<()>,
    stats: Mutex<Stats>,
    threads: usize,
    created_unix: Option<u64>,
}

/// The oracle cascade (see the module notes).
pub struct Cascade {
    inner: Arc<Inner>,
    jobs: Mutex<Option<mpsc::Sender<(String, String)>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for Cascade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cascade")
            .field("state", &self.inner.state.root())
            .field("oracle", &self.inner.oracle)
            .finish()
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Cascade {
    /// The cascade over the served `handle` with its state directory (the key
    /// from the process environment).
    pub fn open(handle: Arc<ModelHandle>, cfg: &Config, state: StateDir) -> Result<Arc<Self>> {
        Self::open_with(handle, cfg, state, CascadeOptions::default())
    }

    /// [`Cascade::open`] with explicit options.
    pub fn open_with(
        handle: Arc<ModelHandle>,
        cfg: &Config,
        state: StateDir,
        opts: CascadeOptions,
    ) -> Result<Arc<Self>> {
        cfg.validate()?;
        let oracle = OracleClient::open(
            &cfg.oracle,
            &state.oracle_ledger_path(),
            Some(&state.oracle_state_path()),
            opts.key,
        )?;
        let (log, replayed) = LearnLog::open(&state.learn_log_path())?;
        let mut cache = SemanticCache::from_config(&cfg.cache);
        let mut buffer = LearningBuffer::new(cfg.learning.dedup);
        let mut contracts = ContractRegistry::default();
        let mut orphans = 0u64;
        for r in &replayed.records {
            match r {
                LogRecord::CachePut(e) => {
                    cache.put(e.clone());
                }
                LogRecord::Contract(c) => {
                    contracts.insert(c.clone());
                }
                // An example of an auto-skill whose contract is not in the log
                // (cannot happen in a consistent log: the contract record
                // precedes the first example) has no rubric to learn under.
                LogRecord::Example(x)
                    if manifest::is_auto_skill_id(&x.skill) && !contracts.contains(&x.skill) =>
                {
                    orphans += 1;
                }
                other => buffer.apply(other),
            }
        }
        if orphans > 0 {
            tracing::warn!(
                examples = orphans,
                "learn.log: examples of auto-skills without a contract record were dropped"
            );
        }
        let seeded = seed_contracts(&handle.current(), &mut contracts, &log)?;
        tracing::info!(
            cache = cache.len(),
            examples = buffer.len(),
            contracts = contracts.len(),
            seeded,
            dropped_bytes = replayed.dropped_bytes,
            "cascade state restored from learn.log"
        );
        let inner = Arc::new(Inner {
            cfg: cfg.clone(),
            handle,
            state,
            oracle,
            cache: Mutex::new(cache),
            buffer: Mutex::new(buffer),
            log,
            books: Mutex::new(Books::default()),
            contracts: Mutex::new(contracts),
            sightings: Mutex::new(Sightings::new(cfg.learning.auto_sightings_cap)),
            bases: Mutex::new(HashMap::new()),
            flights: Mutex::new(Vec::new()),
            next_flight: AtomicU64::new(1),
            queued: AtomicUsize::new(0),
            pending: Mutex::new(VecDeque::new()),
            learn_lock: Mutex::new(()),
            stats: Mutex::new(Stats::default()),
            threads: opts.threads,
            created_unix: opts.created_unix,
        });
        let (jobs, worker) = if cfg.learning.enabled && !cfg.learning.synchronous {
            let (tx, rx) = mpsc::channel::<(String, String)>();
            let w = Arc::clone(&inner);
            let h = std::thread::Builder::new()
                .name("cortiq-decision-learn".into())
                .spawn(move || {
                    while let Ok((skill, label)) = rx.recv() {
                        w.maybe_learn(&skill, &label);
                        w.queued.fetch_sub(1, Ordering::AcqRel);
                    }
                })?;
            (Some(tx), Some(h))
        } else {
            (None, None)
        };
        Ok(Arc::new(Self {
            inner,
            jobs: Mutex::new(jobs),
            worker: Mutex::new(worker),
        }))
    }

    /// The oracle client (status, ledger totals).
    pub fn oracle(&self) -> &OracleClient {
        &self.inner.oracle
    }

    /// Entries of the semantic cache.
    pub fn cache_len(&self) -> usize {
        self.inner.cache.lock().len()
    }

    /// Examples of the learning buffer.
    pub fn buffer_len(&self) -> usize {
        self.inner.buffer.lock().len()
    }

    /// The contracts learned so far (auto-skills, served or not).
    pub fn contracts(&self) -> Vec<Contract> {
        self.inner.contracts.lock().iter().cloned().collect()
    }

    /// Examples of (skill, label) kept and added since its last attempt.
    pub fn label_counts(&self, skill: &str, label: &str) -> (usize, usize) {
        let b = self.inner.buffer.lock();
        (b.examples(skill, label).len(), b.new_count(skill, label))
    }

    /// Run an attempt for (skill, label) now, whatever its counter.
    pub fn learn_now(&self, skill: &str, label: &str) -> Result<AttemptReport> {
        self.inner.run_attempt(skill, label)
    }

    /// Wait until the background worker has no queued job (tests, shutdown).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while self.inner.queued.load(Ordering::Acquire) > 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// `GET /v1/admin/learning`.
    pub fn learning_json(&self) -> Value {
        self.inner.learning_json()
    }

    fn schedule(&self, jobs: Vec<(String, String)>) {
        if jobs.is_empty() {
            return;
        }
        if self.inner.cfg.learning.synchronous {
            for (s, l) in jobs {
                self.inner.maybe_learn(&s, &l);
            }
        } else if let Some(tx) = self.jobs.lock().as_ref() {
            for j in jobs {
                self.inner.queued.fetch_add(1, Ordering::AcqRel);
                if tx.send(j).is_err() {
                    self.inner.queued.fetch_sub(1, Ordering::AcqRel);
                }
            }
        }
    }
}

impl Drop for Cascade {
    fn drop(&mut self) {
        self.jobs.lock().take();
        if let Some(h) = self.worker.lock().take() {
            let _ = h.join();
        }
    }
}

/// A label a feedback can teach: 1..[`MAX_LABEL_BYTES`] bytes, as a task
/// label of a skill manifest.
fn learnable_label(label: &str) -> bool {
    !label.is_empty() && label.len() <= MAX_LABEL_BYTES
}

/// Whether the oracle's answer to a pending question may become an example of
/// the shared `skill` (see the module notes): the caller may teach, or the
/// question is an exact match asking exactly the skill's own question — except
/// for the implicit open mode of a loopback address, which never teaches. An
/// auto-skill's rubric is the contract its callers send, so the rubric rule
/// would let every key teach it: only `learning_allowed` does (DESIGN A7).
fn teaches(e: &Escalation<'_>, p: &Pending<'_>, skill: &str) -> bool {
    e.principal.learning_allowed
        || (!e.principal.implicit_open
            && !manifest::is_auto_skill_id(skill)
            && p.matched.kind == MatchKind::Exact
            && e.model
                .skill(skill)
                .is_some_and(|s| s.follows_rubric(p.question)))
}

/// The contract of a pending untrained choice question that may be learned
/// into an auto-skill (DESIGN D1/D2): no skill fits (not ambiguous), no
/// `cmf.skill`, the caller may teach, 2..=`auto_max_labels` option ids. The
/// registry cap is checked when the contract is registered.
fn auto_contract(cfg: &Config, e: &Escalation<'_>, p: &Pending<'_>) -> Option<Contract> {
    if !cfg.learning.auto_skills
        || !e.principal.learning_allowed
        || e.request.cmf.skill.is_some()
        || p.question.kind != QuestionKind::Choice
        || !p.matched.is_foreign()
    {
        return None;
    }
    let criteria = p.question.criteria.as_ref()?.as_object()?;
    if criteria.len() < 2 || criteria.len() > cfg.learning.auto_max_labels {
        return None;
    }
    // A state-less question's instructions are its text: its contract is
    // the criteria alone (DESIGN A19.1).
    Some(if e.request.reads_instructions(p.question) {
        Contract::stateless(criteria, now_unix())
    } else {
        Contract::new(&p.question.instructions, criteria, now_unix())
    })
}

/// Register the contracts of the served auto-skills the registry lacks, read
/// back from their manifests ([`Contract::of_manifest`]), each appended to
/// `learn.log` as a [`LogRecord::Contract`]; how many were seeded. The served
/// model is the registry's second source: a materialised file served on a
/// fresh state directory (ORACLE.md recommends it with many contracts), or
/// generations kept after `learn.log` was lost, carry auto-skills no record
/// describes — without a contract the skill is matched and served but every
/// example of it is refused and every attempt skipped. Runs at open, after
/// the replay (A4 holds: no example of such a skill is in the log, orphans
/// were dropped), and after a rollback (which may bring a skill forward).
fn seed_contracts(
    model: &crate::service::LoadedModel,
    reg: &mut ContractRegistry,
    log: &LearnLog,
) -> Result<usize> {
    let created = model.model().manifest().created_unix;
    let mut seeded = 0;
    for s in model.skills() {
        let m = s.manifest();
        if !m.is_auto() || reg.contains(s.id()) {
            continue;
        }
        let Some(c) = Contract::of_manifest(m, created) else {
            tracing::warn!(
                skill = s.id(),
                "a served auto-skill's id is not the hash of its rubric: not learned"
            );
            continue;
        };
        log.append(&LogRecord::Contract(c.clone()))?;
        reg.insert(c);
        seeded += 1;
    }
    if seeded > 0 {
        tracing::info!(
            seeded,
            "contracts of served auto-skills without a learn.log record registered"
        );
    }
    Ok(seeded)
}

impl Inner {
    /// Store an answer in the cache and log it when the cache took it.
    fn cache_put(&self, entry: CacheEntry) {
        let stored = self.cache.lock().put(entry.clone());
        if stored && let Err(err) = self.log.append(&LogRecord::CachePut(entry)) {
            tracing::error!(error = %err, "learn.log: cache put not recorded");
        }
    }

    /// The questions answered from the cache alone ([`answer_from`]).
    fn answer_from_cache(
        &self,
        keys: &QuestionKeys<'_>,
        reason: RefusalReason,
    ) -> (Vec<Resolved>, Vec<usize>) {
        answer_from(self.cfg.cache.enabled.then_some(&self.cache), keys, reason)
    }

    fn base_rows(&self, model: &DecisionModel, skill: &str) -> Result<Arc<Rows>> {
        let mut b = self.bases.lock();
        if let Some(r) = b.get(skill) {
            return Ok(Arc::clone(r));
        }
        let r = Arc::new(model.rows(skill)?);
        b.insert(skill.to_string(), Arc::clone(&r));
        Ok(r)
    }

    /// φ_P of the stored rows of a label (base rows and `rows.learned`).
    fn stored_phi(&self, model: &DecisionModel, skill: &str, label: &str) -> Result<Vec<Vec<f32>>> {
        let Some(s) = model.skill(skill) else {
            return Ok(Vec::new());
        };
        let Some(t) = s.manifest.task_of(label) else {
            return Ok(Vec::new());
        };
        let t = t as u32;
        let base = self.base_rows(model, skill)?;
        let mut out: Vec<Vec<f32>> = base
            .rows
            .iter()
            .filter(|r| r.task == t)
            .map(|r| r.phi_p.clone())
            .collect();
        if let Some(l) = model.rows_learned(skill)? {
            out.extend(l.rows.into_iter().filter(|r| r.task == t).map(|r| r.phi_p));
        }
        Ok(out)
    }

    /// Add an example (dedup), log it; the (skill, label) to consider for
    /// learning when it reached the threshold. `contract` is the contract of an
    /// untrained question's auto-skill (registered here, before the example,
    /// when it is new; `Err` when the registry is full — the caller counts it).
    fn add_example(
        &self,
        ex: Example,
        contract: Option<&Contract>,
    ) -> Result<(AddOutcome, Option<(String, String)>)> {
        let model = self.handle.current();
        let stored = self.stored_phi(model.model(), &ex.skill, &ex.label)?;
        let refs: Vec<&[f32]> = stored.iter().map(Vec::as_slice).collect();
        let key = (ex.skill.clone(), ex.label.clone());
        let is_task = |label: &str| {
            model
                .model()
                .skill(&ex.skill)
                .is_some_and(|s| s.manifest.task_of(label).is_some())
        };
        let auto = manifest::is_auto_skill_id(&ex.skill);
        let mut b = self.buffer.lock();
        if auto {
            // The registry is written under the buffer lock, so the contract
            // record precedes the first example of its auto-skill in
            // `learn.log` and two first sightings cannot race (DESIGN A4).
            let mut reg = self.contracts.lock();
            if let Some(c) = contract
                && !reg.contains(&c.skill)
            {
                // Each kind has its own cap: stateful contracts register at
                // their first sighting, so their one-offs must not take the
                // slots of the repeated state-less ones (DESIGN A20).
                let l = &self.cfg.learning;
                let (cap, key) = if c.stateless {
                    (l.auto_max_stateless_skills, "auto_max_stateless_skills")
                } else {
                    (l.auto_max_skills, "auto_max_skills")
                };
                let held = reg.count(c.stateless);
                ensure!(
                    held < cap,
                    "the auto-skill registry holds {held} such contracts (learning.{key})"
                );
                self.log.append(&LogRecord::Contract(c.clone()))?;
                reg.insert(c.clone());
                self.stats.lock().auto_registered += 1;
                self.sightings.lock().forget(&c.skill);
            }
            // Closed label set: a label outside the contract (a superset
            // request, a router feedback) is refused (DESIGN A5). A served
            // auto-skill the registry lacks (seeded at open and rollback;
            // this is the fallback) has the same closed set: its task table.
            let allowed = match reg.get(&ex.skill) {
                Some(c) => c.has_label(&ex.label),
                None => model
                    .model()
                    .skill(&ex.skill)
                    .is_some_and(|s| s.manifest.is_auto() && is_task(&ex.label)),
            };
            if !allowed {
                return Ok((AddOutcome::Full, None));
            }
        }
        let held = b.examples(&ex.skill, &ex.label).len();
        let cap = if auto {
            self.cfg
                .learning
                .auto_max_examples_per_label
                .min(MAX_EXAMPLES_PER_LABEL)
        } else {
            MAX_EXAMPLES_PER_LABEL
        };
        if held >= cap {
            return Ok((AddOutcome::Full, None));
        }
        if held == 0 && !auto && !is_task(&ex.label) {
            let pending_new = b
                .labels()
                .iter()
                .filter(|c| c.skill == ex.skill && !is_task(&c.label))
                .count();
            if pending_new >= MAX_PENDING_NEW_LABELS {
                return Ok((AddOutcome::Full, None));
            }
        }
        if b.is_duplicate(&ex.skill, &ex.label, &ex.phi_p, &refs) {
            b.count_duplicate();
            return Ok((AddOutcome::Duplicate, None));
        }
        self.log.append(&LogRecord::Example(ex.clone()))?;
        b.insert(ex);
        self.stats.lock().examples += 1;
        let due = b.new_count(&key.0, &key.1) >= self.cfg.learning.refit_min_new;
        Ok((AddOutcome::Stored, due.then_some(key)))
    }

    /// An untrained contract the oracle answered that is not learned (ids
    /// outside 2..=`auto_max_labels`, or the registry full): counted, and
    /// named by its size in a WARN at most once an hour (never its ids).
    fn note_auto_skipped(&self, ids: usize) {
        let mut st = self.stats.lock();
        st.auto_skipped += 1;
        let due = st
            .auto_skipped_logged
            .is_none_or(|t| t.elapsed() >= AUTO_SKIPPED_LOG_EVERY);
        if due {
            st.auto_skipped_logged = Some(Instant::now());
            tracing::warn!(
                ids,
                skipped = st.auto_skipped,
                max_labels = self.cfg.learning.auto_max_labels,
                max_skills = self.cfg.learning.auto_max_skills,
                max_stateless_skills = self.cfg.learning.auto_max_stateless_skills,
                "an untrained contract is answered by the oracle but not learned (auto-skill limits)"
            );
        }
    }

    fn maybe_learn(&self, skill: &str, label: &str) {
        let due = self.buffer.lock().new_count(skill, label) >= self.cfg.learning.refit_min_new;
        if !due {
            return;
        }
        if let Err(e) = self.run_attempt(skill, label) {
            // Labels may come from client feedback: the line names the
            // label's hash only (spec §4.3), also inside the error text.
            tracing::error!(
                error = %learn::redact_label(&format!("{e:#}"), label),
                skill,
                label_sha = %learn::label_tag(label),
                "learning attempt failed"
            );
        }
    }

    fn run_attempt(&self, skill: &str, label: &str) -> Result<AttemptReport> {
        let _g = self.learn_lock.lock();
        let base_rows = |m: &DecisionModel, s: &str| self.base_rows(m, s);
        let ctx = LearnContext {
            handle: &self.handle,
            state: &self.state,
            buffer: &self.buffer,
            log: &self.log,
            books: &self.books,
            contracts: &self.contracts,
            cfg: &self.cfg.learning,
            base_rows: &base_rows,
            threads: self.threads,
            created_unix: self.created_unix,
        };
        let r = learn::attempt(&ctx, skill, label);
        let mut st = self.stats.lock();
        st.attempts += 1;
        match &r {
            Ok(rep) => {
                match &rep.outcome {
                    Outcome::Promoted { .. } => {
                        st.promotions += 1;
                        if rep.kind == Some(learn::ChangeKind::ColdStart) {
                            st.cold_starts += 1;
                        }
                    }
                    Outcome::Rejected(_) => st.rejections += 1,
                    Outcome::Skipped(_) => st.skipped += 1,
                }
                st.isolation_violations += rep.isolation_violations as u64;
                st.recent.push_back(rep.to_json());
            }
            Err(e) => {
                st.errors += 1;
                st.recent.push_back(json!({"skill": skill, "label": label, "outcome": "error", "reason": format!("{e:#}")}));
            }
        }
        while st.recent.len() > RECENT_ATTEMPTS {
            st.recent.pop_front();
        }
        r
    }

    fn learning_json(&self) -> Value {
        let model = self.handle.current();
        let b = self.buffer.lock();
        let labels = b.labels();
        let mut quarantine = Vec::new();
        let mut buffer = Vec::new();
        for l in &labels {
            let active = model
                .skill(&l.skill)
                .and_then(|s| {
                    s.manifest()
                        .task_of(&l.label)
                        .map(|t| s.manifest().tasks[t].is_active())
                })
                .unwrap_or(false);
            let v = json!({"skill": l.skill, "label": l.label, "examples": l.total, "new": l.new});
            if !active {
                quarantine.push(v.clone());
            }
            buffer.push(v);
        }
        let total = b.len();
        let dups = b.duplicates();
        // Every contract learned so far: its labels, the rows per label an
        // attempt would see (the served learned rows of the label plus the
        // pending examples not among them — the buffer keeps every example
        // after a promotion, so buffer + served would count them twice), what
        // the served model has of it.
        let auto_skills: Vec<Value> = self
            .contracts
            .lock()
            .iter()
            .map(|c| {
                let served = model.skill(&c.skill);
                let per_label = learn::auto_rows_per_label(model.model(), &b, c).unwrap_or_else(|e| {
                    tracing::error!(error = %format!("{e:#}"), skill = c.skill, "auto-skill rows");
                    c.ids
                        .iter()
                        .map(|l| (l.clone(), b.examples(&c.skill, l).len()))
                        .collect()
                });
                let examples: serde_json::Map<String, Value> = per_label
                    .into_iter()
                    .map(|(l, n)| (l, json!(n)))
                    .collect();
                // Active labels in task order (the listing's order).
                let active: Vec<&str> = served.map_or_else(Vec::new, |s| {
                    s.manifest()
                        .tasks
                        .iter()
                        .filter(|t| t.is_active())
                        .map(|t| t.label.as_str())
                        .collect()
                });
                json!({
                    "id": c.skill,
                    "labels": c.ids,
                    "stateless": c.stateless,
                    "examples": examples,
                    "created_unix": c.created_unix,
                    "served": served.map(|s| json!({
                        "generation": model.generation(),
                        "taxonomy_version": s.manifest().taxonomy_version,
                        "active": active,
                    })),
                })
            })
            .collect();
        let auto_contracts = self.contracts.lock().len();
        let auto_sightings = self.sightings.lock().len();
        drop(b);
        let c = self.cache.lock();
        let cache = json!({"entries": c.len(), "hits": c.hits(), "lookups": c.lookups(),
                           "threshold": c.threshold(), "legacy_cos": c.legacy_cos(),
                           "enabled": self.cfg.cache.enabled});
        drop(c);
        let mut tasks = serde_json::Map::new();
        for s in model.skills() {
            let m: serde_json::Map<String, Value> = s
                .manifest()
                .tasks
                .iter()
                .map(|t| {
                    (
                        t.label.clone(),
                        json!({"i": t.i, "state": t.state, "origin": t.origin, "k": t.k,
                               "n_train": t.n_train, "mean_sha256": t.mean_sha256,
                               "basis_sha256": t.basis_sha256}),
                    )
                })
                .collect();
            tasks.insert(
                s.id().to_string(),
                json!({"taxonomy_version": s.manifest().taxonomy_version, "tasks": m}),
            );
        }
        let st = self.stats.lock();
        json!({
            "enabled": self.cfg.learning.enabled,
            "synchronous": self.cfg.learning.synchronous,
            "cold_start": self.cfg.learning.cold_start,
            "refit_min_new": self.cfg.learning.refit_min_new,
            "dedup": self.cfg.learning.dedup,
            "generation": model.generation(),
            "model_sha": model.model_sha(),
            "buffer": {"examples": total, "duplicates_refused": dups, "labels": buffer},
            "quarantine": quarantine,
            "cache": cache,
            "pending_feedback": self.pending.lock().len(),
            "attempts": st.attempts, "promotions": st.promotions, "rejections": st.rejections,
            "cold_starts": st.cold_starts, "skipped": st.skipped, "errors": st.errors,
            "isolation_violations": st.isolation_violations, "rollbacks": st.rollbacks,
            "examples_added": st.examples, "feedback": st.feedback,
            "auto_contracts": auto_contracts,
            "auto_skipped": st.auto_skipped,
            "auto_sightings": auto_sightings,
            "auto_registered": st.auto_registered,
            "auto_skills": auto_skills,
            "recent": st.recent.iter().cloned().collect::<Vec<_>>(),
            "skills": Value::Object(tasks),
        })
    }

    fn rollback(&self, to: u64) -> Result<Value, ApiError> {
        let _g = self.learn_lock.lock();
        let current = self.handle.current();
        let base = current.model().base_path().to_path_buf();
        let model = generation::rollback(&self.state, &base, to, Verify::Light).map_err(|e| {
            let msg = format!("{e:#}");
            if msg.contains("does not exist") {
                ApiError::not_found(msg)
            } else {
                ApiError::invalid(msg)
            }
        })?;
        let next = current.derive(model).map_err(|e| {
            tracing::error!(error = %e, "rollback: the model does not load");
            ApiError::internal("the generation does not load")
        })?;
        let name = next.name();
        let sha = next.model_sha().to_string();
        self.handle.promote(next);
        self.books.lock().invalidate();
        // The base rows of a skill born in a generation (an auto-skill) vanish
        // with it; a later generation may bring the id back with other rows.
        self.bases.lock().clear();
        {
            // A generation rolled forward may carry an auto-skill whose
            // contract record is not in learn.log (lost with it): register it
            // from the manifest, under the buffer lock as `add_example` does
            // (the record precedes any example of it, DESIGN A4).
            let mut b = self.buffer.lock();
            let mut reg = self.contracts.lock();
            if let Err(e) = seed_contracts(&self.handle.current(), &mut reg, &self.log) {
                tracing::error!(error = %format!("{e:#}"), "rollback: contracts of served auto-skills not recorded");
            }
            b.reset_all();
        }
        if let Err(e) = self.log.append(&LogRecord::Rollback { generation: to }) {
            tracing::error!(error = %e, "learn.log: could not record the rollback");
        }
        self.stats.lock().rollbacks += 1;
        Ok(
            json!({"generation": to, "model": name, "model_sha": sha, "buffer_kept": true, "counters_reset": true}),
        )
    }
}

impl Escalator for Cascade {
    fn escalate(&self, e: &Escalation<'_>) -> EscalationResult {
        let inner = &self.inner;
        let cfg = &inner.cfg;
        let n = e.pending.len();
        let caller = Caller {
            request_id: e.request_id,
            account: &e.principal.account,
            key12: e.principal.key12.as_deref(),
            key_budget_usd: e.principal.oracle_budget_usd.map(Usd::to_f64),
            credit_left_usd: e.oracle_credit_usd,
        };
        // The state and the questions as they would leave for the oracle (PII
        // redacted unless the request allows its egress), and whether
        // anything was redacted. Every question's instructions (a state-less
        // question's input, DESIGN A19.3) and its criteria's descriptions are
        // redacted like a state (DESIGN B4) — the option ids, `true`/`false`
        // and a level's position are object keys and indices, never touched.
        // Only the egress copy changes: the scopes, the contract keys and the
        // examples use the question as asked.
        let redact = cfg.oracle.redact_pii && !e.request.cmf.allow_pii_egress;
        let egress = |idx: &[usize]| -> (Value, Vec<Question>, bool) {
            let raw = e.request.state.to_value();
            let (state, mut redacted) = if redact {
                redact_value(&raw)
            } else {
                (raw, false)
            };
            let qs = idx
                .iter()
                .map(|&i| {
                    let q = e.pending[i].question;
                    if redact {
                        let (instructions, changed) = redact_value(&q.instructions);
                        redacted |= changed;
                        let criteria = q.criteria.as_ref().map(|c| {
                            let (c, changed) = redact_value(c);
                            redacted |= changed;
                            c
                        });
                        Question {
                            instructions,
                            criteria,
                            ..q.clone()
                        }
                    } else {
                        q.clone()
                    }
                })
                .collect();
            (state, qs, redacted)
        };
        // φ_P, scope and input of each question: the state's and its
        // contract, or in a state-less request its instructions' and the
        // criteria's (DESIGN A19.4).
        let keys = QuestionKeys::of(e);
        let (phis, scopes, inputs) = (&keys.phis, &keys.scopes, &keys.inputs);
        if let Err(r) = inner.oracle.permission(&caller) {
            // What the cache holds is answered all the same (0.8.10): no
            // call, nothing sent; only the misses are refused.
            let (resolved, misses) = inner.answer_from_cache(&keys, r);
            if r == RefusalReason::Budget && !misses.is_empty() {
                // Refused before a body was built: the status and the hints
                // name what this request's call would reserve.
                let (state, qs, _) = egress(&misses);
                let qs: Vec<&Question> = qs.iter().collect();
                inner.oracle.note_budget_refusal(&qs, &state);
            }
            return EscalationResult {
                resolved,
                usage: OracleUsage::default(),
            };
        }
        let mut out: Vec<Option<Resolved>> = vec![None; n];

        // Sightings of the gated contracts not registered yet (DESIGN A20,
        // B2: every state-less one, a stateful one when
        // `auto_min_sightings_stateful` > 1), one per contract and request,
        // whatever answers them next.
        let mut seen: HashMap<String, u32> = HashMap::new();
        if cfg.learning.enabled {
            for p in &e.pending {
                let Some(c) = auto_contract(cfg, e, p) else {
                    continue;
                };
                // State-less contracts are always counted (the admin
                // `auto_sightings` of 0.8.7), stateful ones only when gated.
                let gated = c.stateless || cfg.learning.auto_min_sightings_stateful > 1;
                if !gated || seen.contains_key(&c.skill) {
                    continue;
                }
                let registered = inner.contracts.lock().contains(&c.skill);
                if !registered {
                    let k = inner.sightings.lock().see(&c.skill);
                    seen.insert(c.skill, k);
                }
            }
        }

        // 2. Cache.
        if cfg.cache.enabled {
            let mut c = inner.cache.lock();
            for i in 0..n {
                if let Some((a, _)) = c.get(&scopes[i], &inputs[i], phis[i]) {
                    out[i] = Some(Resolved::new(Resolution::Cache(a)));
                }
            }
        }

        // 3. Single flight.
        let mut leaders = Vec::new();
        // A follower, its leader's slot, and whether it asks the leader's
        // input (else a near duplicate, near reuse on).
        let mut followers: Vec<(usize, Arc<Slot>, bool)> = Vec::new();
        let mut guard = FlightGuard {
            flights: &inner.flights,
            mine: Vec::new(),
        };
        {
            let mut fl = inner.flights.lock();
            for i in 0..n {
                if out[i].is_some() {
                    continue;
                }
                let phi = phis[i];
                let found = fl.iter().find(|f| {
                    f.scope == scopes[i]
                        && reuses(cfg.cache.threshold, &f.input, &f.phi_p, &inputs[i], phi)
                });
                if let Some(f) = found {
                    followers.push((i, Arc::clone(&f.slot), f.input == inputs[i]));
                    continue;
                }
                // A leader that finished between the first cache lookup and
                // this registration left its answer in the cache.
                if cfg.cache.enabled
                    && let Some((a, _)) = inner.cache.lock().recheck(&scopes[i], &inputs[i], phi)
                {
                    out[i] = Some(Resolved::new(Resolution::Cache(a)));
                } else {
                    let id = inner.next_flight.fetch_add(1, Ordering::Relaxed);
                    let slot = Arc::new(Slot::default());
                    fl.push(Flight {
                        id,
                        scope: scopes[i].clone(),
                        input: inputs[i],
                        phi_p: phi.to_vec(),
                        slot: Arc::clone(&slot),
                    });
                    guard.mine.push((id, slot));
                    leaders.push(i);
                }
            }
        }

        // 4. One call for the leaders.
        let mut usage = OracleUsage::default();
        let mut learn_jobs: Vec<(String, String)> = Vec::new();
        if !leaders.is_empty() {
            let (state, qs, redacted) = egress(&leaders);
            let qs: Vec<&Question> = qs.iter().collect();
            let outcome = inner.oracle.call(&caller, &qs, &state);
            let flags: Vec<String> = if redacted {
                vec![FLAG_PII_REDACTED.to_string()]
            } else {
                Vec::new()
            };
            match outcome {
                CallOutcome::Answered(a) => {
                    usage = OracleUsage {
                        calls: 1,
                        input_tokens: a.usage.input_tokens,
                        output_tokens: a.usage.output_tokens,
                        reasoning_tokens: a.usage.reasoning_tokens.unwrap_or(0),
                        cost: Usd::from_f64(a.usage.cost).unwrap_or_else(|_| {
                            tracing::error!("oracle cost not representable");
                            Usd::from_units(0).expect("zero")
                        }),
                    };
                    let ts = now_unix();
                    for (k, &i) in leaders.iter().enumerate() {
                        let v = a.verdicts[k].clone();
                        out[i] = Some(Resolved {
                            resolution: Resolution::Oracle(v.clone()),
                            flags: flags.clone(),
                        });
                        guard.mine[k].1.set(Resolution::Oracle(v.clone()));
                        if cfg.cache.enabled {
                            inner.cache_put(CacheEntry {
                                scope: scopes[i].clone(),
                                input: Some(inputs[i]),
                                phi_p: phis[i].to_vec(),
                                answer: v.clone(),
                                ts,
                            });
                        }
                        let p = &e.pending[i];
                        if !cfg.learning.enabled {
                            continue;
                        }
                        let OracleAnswer::Choice(id) = &v.answer else {
                            continue;
                        };
                        // A description match names the skill's label by
                        // the option's description; its none option names
                        // no label and teaches nothing (DESIGN C2).
                        let Some(label) = p.matched.label_of(id) else {
                            continue;
                        };
                        // A matched skill learns under `teaches`; an untrained
                        // contract learns into its auto-skill (see the module
                        // notes), its contract registered with the example.
                        let (skill, contract) = match &p.matched.skill {
                            Some(skill) if teaches(e, p, skill) => (skill.clone(), None),
                            Some(_) => continue,
                            None => match auto_contract(cfg, e, p) {
                                // A contract seen fewer than its kind's
                                // `auto_min_sightings*` times is not
                                // registered yet: its answer is not learned
                                // (DESIGN A20, B2).
                                Some(c)
                                    if seen.get(&c.skill).is_some_and(|&k| {
                                        k < cfg.learning.min_sightings(c.stateless)
                                    }) =>
                                {
                                    continue;
                                }
                                Some(c) => (c.skill.clone(), Some(c)),
                                None => {
                                    if cfg.learning.auto_skills
                                        && e.principal.learning_allowed
                                        && p.matched.is_foreign()
                                        && e.request.cmf.skill.is_none()
                                    {
                                        inner.note_auto_skipped(p.question.options().len());
                                    }
                                    continue;
                                }
                            },
                        };
                        let ex = Example::from_features(
                            &skill,
                            label,
                            Source::Oracle,
                            e.features_of(p.index),
                            ts,
                        );
                        match inner.add_example(ex, contract.as_ref()) {
                            Ok((_, Some(job))) => learn_jobs.push(job),
                            Ok(_) => {}
                            Err(err) if contract.is_some() => {
                                // The registry is full: the oracle answered, the
                                // contract is not learned (counted, rate-limited).
                                tracing::debug!(error = %format!("{err:#}"), "auto-skill");
                                inner.note_auto_skipped(p.question.options().len());
                            }
                            Err(err) => {
                                tracing::error!(error = %format!("{err:#}"), "learning buffer")
                            }
                        }
                    }
                }
                CallOutcome::Failed(f) => {
                    tracing::warn!(request = e.request_id, error = %f.error, "oracle call failed");
                    for (k, &i) in leaders.iter().enumerate() {
                        let r = Resolution::Failed(f.error.clone());
                        guard.mine[k].1.set(r.clone());
                        out[i] = Some(Resolved {
                            resolution: r,
                            flags: flags.clone(),
                        });
                    }
                }
                CallOutcome::Refused(r) => {
                    for (k, &i) in leaders.iter().enumerate() {
                        guard.mine[k].1.set(Resolution::Refused(r));
                        out[i] = Some(Resolved::new(Resolution::Refused(r)));
                    }
                }
            }
        }
        drop(guard);

        // Followers wait for their leader.
        let wait = Duration::from_secs_f64(cfg.oracle.escalation_deadline_s()) + FOLLOWER_GRACE;
        for (i, slot, same_input) in followers {
            let r = match slot.wait(wait) {
                Some(Resolution::Oracle(a)) => {
                    // The answer to this very question (the leader's input):
                    // cached and logged as an oracle answer is (0.8.10; the
                    // leader's put and this one dedup, whichever comes
                    // first). A near duplicate's is not: the oracle never
                    // read its input, and an entry under its digest would
                    // answer it exactly, also once near reuse is off —
                    // its repeat hits the leader's entry while near reuse
                    // is on. Learned it is not: the leader's call taught.
                    if cfg.cache.enabled && same_input {
                        inner.cache_put(CacheEntry {
                            scope: scopes[i].clone(),
                            input: Some(inputs[i]),
                            phi_p: phis[i].to_vec(),
                            answer: a.clone(),
                            ts: now_unix(),
                        });
                    }
                    Resolution::Cache(a)
                }
                Some(other) => other,
                None => Resolution::Failed("single-flight wait timed out".into()),
            };
            out[i] = Some(Resolved::new(r));
        }

        self.schedule(learn_jobs);
        EscalationResult {
            resolved: out
                .into_iter()
                .map(|r| {
                    r.unwrap_or_else(|| Resolved::new(Resolution::Failed("unresolved".into())))
                })
                .collect(),
            usage,
        }
    }

    /// The cache alone (0.8.10): a request the service's consent checks
    /// refused is answered with what the cache holds — no call, nothing
    /// sent, no single flight, nothing learned, no sighting — and its misses
    /// are refused with `refused`.
    fn resolve_without_oracle(
        &self,
        e: &Escalation<'_>,
        refused: RefusalReason,
    ) -> EscalationResult {
        let (resolved, _) = self.inner.answer_from_cache(&QuestionKeys::of(e), refused);
        EscalationResult {
            resolved,
            usage: OracleUsage::default(),
        }
    }

    /// The cache is on and holds something: an empty cache answers
    /// nothing, so a request without consent then fails before any work
    /// as before 0.8.10.
    fn answers_without_oracle(&self) -> bool {
        self.inner.cfg.cache.enabled && !self.inner.cache.lock().is_empty()
    }

    fn feedback(&self, fb: &FeedbackRequest, principal: &Principal) -> Result<Value, ApiError> {
        let inner = &self.inner;
        if !inner.cfg.learning.enabled {
            return Err(ApiError::not_found(
                "learning is disabled on this server: feedback is not kept",
            ));
        }
        let not_found = || {
            ApiError::not_found(format!(
                "no pending decision {} / question {} for this account",
                fb.id, fb.question
            ))
        };
        let entry = {
            let mut ring = inner.pending.lock();
            let pos = ring
                .iter()
                .position(|p| {
                    p.request_id == fb.id
                        && p.question == fb.question
                        && p.account == principal.account
                })
                .ok_or_else(not_found)?;
            // The router API's feedback may name any label (a new one starts a
            // cold start, router `api.rs:1283-1292`); the decisions API's only
            // an option of the question.
            if fb.any_label && !learnable_label(&fb.label) {
                // The router accepts an empty or over-long label too (200,
                // the request consumed); it names no task that could be
                // learned, so nothing is stored here.
                let entry = ring.remove(pos).ok_or_else(not_found)?;
                inner.stats.lock().feedback += 1;
                return Ok(json!({
                    "id": fb.id, "question": fb.question, "skill": entry.skill,
                    "accepted": false, "learned": false,
                    "reason": format!(
                        "the label must be 1..{MAX_LABEL_BYTES} bytes to be learned"
                    ),
                }));
            }
            if !fb.any_label && !ring[pos].options.contains(&fb.label) {
                return Err(ApiError::invalid_field(
                    "label",
                    format!(
                        "'{}' is not an option of question {}",
                        fb.label, fb.question
                    ),
                ));
            }
            ring.remove(pos).ok_or_else(not_found)?
        };
        // Under a description match the option names the skill's label; the
        // none option names none (DESIGN C2): consumed, nothing to learn.
        let label = match &entry.by_descriptions {
            Some(d) if !fb.any_label => match d.label_of(&fb.label) {
                Some(l) => l.to_string(),
                None => {
                    inner.stats.lock().feedback += 1;
                    return Ok(json!({
                        "id": fb.id, "question": fb.question, "skill": entry.skill,
                        "label": fb.label, "accepted": false, "learned": false,
                        "reason": "the none option names no label of the skill",
                    }));
                }
            },
            _ => fb.label.clone(),
        };
        let model = inner.handle.current();
        let known = model
            .skill(&entry.skill)
            .and_then(|s| s.manifest().task_of(&label))
            .is_some();
        if !principal.learning_allowed {
            // The entry is consumed, as a learned feedback's is; nothing is
            // stored, so no key without the permission writes to the model
            // every account is served.
            inner.stats.lock().feedback += 1;
            return Ok(json!({
                "id": fb.id, "question": fb.question, "skill": entry.skill, "label": fb.label,
                "accepted": false, "learned": false, "known_label": known,
                "refused": LEARNING_NOT_ALLOWED,
                "reason": "this key may not teach the model (learning_allowed is false)",
            }));
        }
        let ex = Example {
            skill: entry.skill.clone(),
            label: label.clone(),
            source: Source::ClientFeedback,
            weight: Source::ClientFeedback.default_weight(),
            ts: now_unix(),
            phi_p: entry.state.phi_p.clone(),
            h_idx: entry.state.h_idx.clone(),
            h_val: entry.state.h_val.clone(),
        };
        let (added, job) = inner.add_example(ex, None).map_err(|e| {
            tracing::error!(
                error = %learn::redact_label(&format!("{e:#}"), &label),
                "feedback example"
            );
            ApiError::internal("the feedback could not be stored")
        })?;
        inner.stats.lock().feedback += 1;
        let (total, new) = {
            let b = inner.buffer.lock();
            (
                b.examples(&entry.skill, &label).len(),
                b.new_count(&entry.skill, &label),
            )
        };
        let mut learning = Value::Null;
        if let Some((s, l)) = job {
            if inner.cfg.learning.synchronous {
                learning = match inner.run_attempt(&s, &l) {
                    Ok(r) => r.to_json(),
                    Err(e) => json!({"outcome": "error", "reason": format!("{e:#}")}),
                };
            } else {
                self.schedule(vec![(s, l)]);
                learning = json!({"outcome": "scheduled"});
            }
        }
        Ok(json!({
            "id": fb.id, "question": fb.question, "skill": entry.skill, "label": fb.label,
            "accepted": added == AddOutcome::Stored, "learned": added == AddOutcome::Stored,
            "duplicate": added == AddOutcome::Duplicate, "full": added == AddOutcome::Full,
            "known_label": known, "cold_start": !known, "weight": Source::ClientFeedback.default_weight(),
            "examples": total, "new_examples": new, "refit_min_new": inner.cfg.learning.refit_min_new,
            "learning": learning,
        }))
    }

    fn admin(&self, command: &AdminCommand) -> Result<Value, ApiError> {
        let inner = &self.inner;
        match command {
            AdminCommand::OracleStatus => {
                let mut v = inner.oracle.status_json();
                v["configured"] = json!(inner.cfg.oracle.enabled);
                Ok(v)
            }
            AdminCommand::OracleUpdate(body) => {
                let mut v = inner.oracle.update(body)?;
                v["configured"] = json!(inner.cfg.oracle.enabled);
                Ok(v)
            }
            AdminCommand::Learning => Ok(inner.learning_json()),
            AdminCommand::Generations => {
                let list = generation::list(&inner.state).map_err(|e| {
                    tracing::error!(error = %format!("{e:#}"), "generations");
                    ApiError::internal("the generations could not be listed")
                })?;
                let current = inner.handle.current();
                Ok(json!({
                    "current": current.generation(),
                    "model": current.name(),
                    "base_model_sha": current.model().base_model_sha(),
                    "generations": list.iter().map(generation::GenerationInfo::to_json).collect::<Vec<_>>(),
                }))
            }
            AdminCommand::Rollback { generation } => inner.rollback(*generation),
        }
    }

    fn oracle_status(&self) -> Option<OracleStatus> {
        Some(self.inner.oracle.status())
    }

    fn observe(&self, o: &Observation<'_>) {
        let inner = &self.inner;
        if !inner.cfg.learning.enabled {
            return;
        }
        // One sparse copy per input text (the state's, or each state-less
        // question's instructions', DESIGN A19).
        let mut shared: Vec<(*const crate::signal::Features, Arc<SparseState>)> = Vec::new();
        let mut new = Vec::new();
        for (index, q) in o.questions.iter().enumerate() {
            let Some(skill) = &q.matched.skill else {
                continue;
            };
            if q.kind != QuestionKind::Choice {
                continue;
            }
            let Some(question) = o.request.question(&q.id) else {
                continue;
            };
            let f = o.features_of(index);
            let st = match shared.iter().find(|(k, _)| std::ptr::eq(*k, f)) {
                Some((_, st)) => Arc::clone(st),
                None => {
                    let (h_idx, h_val) = crate::rows::sparse_from_dense(&f.phi_h);
                    let st = Arc::new(SparseState {
                        phi_p: f.phi_p.clone(),
                        h_idx,
                        h_val,
                    });
                    shared.push((f, Arc::clone(&st)));
                    st
                }
            };
            new.push(PendingEntry {
                request_id: o.request_id.to_string(),
                question: q.id.clone(),
                account: o.principal.account.clone(),
                skill: skill.clone(),
                options: question.options().into_iter().map(str::to_string).collect(),
                by_descriptions: q.matched.by_descriptions.clone(),
                state: st,
            });
        }
        if new.is_empty() {
            return;
        }
        let cap = inner.cfg.feedback.pending_cap;
        let mut ring = inner.pending.lock();
        for p in new {
            if ring.len() >= cap {
                ring.pop_front();
            }
            ring.push_back(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sightings LRU (DESIGN A20) counts per contract, evicts the least
    /// recently seen past its cap and forgets a registered contract.
    #[test]
    fn sightings_are_a_bounded_lru() {
        let mut s = Sightings::new(2);
        assert_eq!(s.see("a"), 1);
        assert_eq!(s.see("b"), 1);
        assert_eq!(s.see("a"), 2);
        // "b" is now the least recently seen: "c" evicts it.
        assert_eq!(s.see("c"), 1);
        assert_eq!((s.len(), s.count("a"), s.count("b")), (2, 2, 0));
        assert_eq!(s.see("b"), 1, "an evicted contract starts again");
        assert_eq!(s.count("a"), 0, "a was the oldest then");
        s.forget("b");
        assert_eq!((s.len(), s.count("b")), (1, 0));
        assert!(!s.is_empty());
        let mut one = Sightings::new(0);
        one.see("x");
        one.see("y");
        assert_eq!(one.len(), 1, "the cap is at least 1");
    }
}
