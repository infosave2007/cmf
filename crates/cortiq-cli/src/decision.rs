//! `cortiq decide`, `cortiq decision …` and the decision branch of `cortiq
//! serve` (spec decision-v4 §3.1, §3.8, §4.1, §4.2, §4.10, §4.15, §5.10, §5.14).
//!
//! Everything here is a thin command-line shell over `cortiq_decision`
//! (training, evaluation, the decisions service, generations, keys, offline
//! learning) and `cortiq_server::decisions` (the HTTP server). No GPU backend is
//! initialised on behalf of these commands: the encoder runs on the host GEMM,
//! and a decision file never reaches `Pipeline`, whose guard refuses it (`run`,
//! `bench`, `route` …).
//!
//! * `cortiq decide FILE -p TEXT [--skill ID | --labels a,b,c] [--state DIR]
//!   [--json] [--round 2]`: one text, decided by the same service as `POST
//!   /v1/decisions` (skill matching of spec §4.5, the certified gate, metering),
//!   without an oracle — a question the gate rejects stays `abstain`;
//! * `cortiq decide FILE --input rows.jsonl [--skill ID] [--out out.jsonl]
//!   [--bench]`: one JSON object per row (never the text), totals on stderr —
//!   the tool of the numerical gates;
//! * `cortiq decision init | train | add-skill | learn | info | verify |
//!   materialize | rollback | keys`;
//! * `cortiq serve FILE` on a decision file: the decisions server on
//!   127.0.0.1 unless `--host` says otherwise; the language-model flags are
//!   refused.

use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Args, Subcommand};
use cortiq_core::CmfModel;
use cortiq_core::format::features;
use cortiq_decision::build::{self, BuildReport, TrainOptions};
use cortiq_decision::config::Config;
use cortiq_decision::container::{self, DecisionModel, Verify, WriteReport};
use cortiq_decision::eval::{self, EvalOptions, Evaluator, SkillScorer};
use cortiq_decision::generation;
use cortiq_decision::keys::{
    self as keys_mod, ImportFormat, ImportReport, KeyStore, NewKey, UsageImportReport, now_unix,
};
use cortiq_decision::learn::{self, OfflineOptions, OfflineReport};
use cortiq_decision::ledger::UsageLedger;
use cortiq_decision::manifest::{Gate, SkillManifest, TaskState};
use cortiq_decision::oracle;
use cortiq_decision::protocol::{ApiError, MODEL_ID, model_name};
use cortiq_decision::service::{
    Decided, DecisionService, LoadedModel, ModelHandle, Principal, QuestionOutcome,
};
use cortiq_decision::shadow::{SHADOW_LOG_FILE, upstream_base};
use cortiq_decision::signal::SignalEncoder;
use cortiq_decision::statedir::{StateDir, generation_name};
use cortiq_server::decisions::{self as server, ServeOptions};
use serde_json::{Map, Value, json};
use std::io::{BufWriter, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ------------------------------------------------------------------ serve

/// `--host` of `cortiq serve` for a language model when the flag is absent
/// (unchanged from 0.7.7).
pub const LLM_DEFAULT_HOST: &str = "0.0.0.0";

/// Does an opened CMF file carry the DECISION feature bit (spec §2.1)?
pub fn is_decision_model(model: &CmfModel) -> bool {
    model.required_features & features::DECISION != 0
}

/// The address `cortiq serve` binds: `--host` when given, else 0.0.0.0 for a
/// language model (as before) and 127.0.0.1 for a decision file (spec §4.2).
pub fn resolve_serve_host(host: Option<&str>, decision: bool) -> &str {
    match host {
        Some(h) => h,
        None if decision => server::DEFAULT_HOST,
        None => LLM_DEFAULT_HOST,
    }
}

/// The flags of `cortiq serve` that apply only to a decision file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServeFlags {
    /// `--decision-config PATH` (spec §4.13).
    pub decision_config: Option<PathBuf>,
    /// `--state DIR` (spec §4.14; default `<FILE>.state` or `state_dir`).
    pub state: Option<PathBuf>,
    /// `--break-lock`: remove a state `LOCK` left by a dead process (spec §4.11).
    pub break_lock: bool,
    /// `--shadow-of URL`: shadow mode of the router API (spec §4.15).
    pub shadow_of: Option<String>,
    /// `--shadow-timeout-s N`: deadline of one request forwarded to the old
    /// router (default [`cortiq_decision::shadow::UPSTREAM_TIMEOUT`], 60 s).
    pub shadow_timeout_s: Option<u64>,
}

impl ServeFlags {
    /// The decision-only flags given, as spelled on the command line.
    pub fn given(&self) -> Vec<&'static str> {
        [
            ("--decision-config", self.decision_config.is_some()),
            ("--state", self.state.is_some()),
            ("--break-lock", self.break_lock),
            ("--shadow-of", self.shadow_of.is_some()),
            ("--shadow-timeout-s", self.shadow_timeout_s.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect()
    }
}

/// Refuse the `cortiq serve` flags that do not apply to the file: the
/// language-model flags (`llm_given`: `--task`, `--o1*`, `--peer*`, `--gpus`,
/// `--compat-port`, …) on a decision file, the decision flags on a language
/// model (spec §4.2).
pub fn check_serve_flags(
    model: &str,
    decision: bool,
    llm_given: &[&'static str],
    flags: &ServeFlags,
) -> Result<()> {
    if decision {
        ensure!(
            llm_given.is_empty(),
            "{} is a decision file: {} apply only to language models (a decision server takes \
             --host, --port, --decision-config, --state, --break-lock and --shadow-of)",
            model,
            llm_given.join(", ")
        );
    } else {
        let given = flags.given();
        ensure!(
            given.is_empty(),
            "{} apply only to decision files (DECISION feature bit); {} is a language model",
            given.join(", "),
            model
        );
    }
    Ok(())
}

fn socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve --host {host}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("--host {host} resolves to no address"))
}

/// `cortiq serve FILE` on a decision file: the decisions server of
/// `cortiq_server::decisions` (no `Pipeline`, no GPU) until Ctrl-C/SIGTERM.
pub async fn serve(model: &str, host: &str, port: u16, flags: &ServeFlags) -> Result<()> {
    let config = match &flags.decision_config {
        Some(p) => Config::load(p)?,
        None => Config::default(),
    };
    let mut opts = ServeOptions::new(model, config);
    opts.addr = socket_addr(host, port)?;
    opts.state_dir = flags.state.clone();
    opts.break_lock = flags.break_lock;
    opts.shadow_of = flags.shadow_of.clone();
    if let Some(t) = flags.shadow_timeout_s {
        ensure!(
            flags.shadow_of.is_some(),
            "--shadow-timeout-s applies only with --shadow-of"
        );
        ensure!(t > 0, "--shadow-timeout-s must be at least 1 second");
        opts.shadow_timeout = std::time::Duration::from_secs(t);
    }
    println!(
        "  Decision file: decisions API on http://{} (state {})",
        opts.addr,
        opts.state_root().display()
    );
    if let Some(url) = &opts.shadow_of {
        // Checked before it is printed (credentials in it are refused).
        let base = upstream_base(url)?;
        println!(
            "  Shadow mode: the router API is answered by {base} (deadline {} s per request); \
             /v1/route and /v1/route:batch are also decided locally (no oracle, learning or \
             billing) and compared in {}",
            opts.shadow_timeout.as_secs(),
            opts.state_root().join(SHADOW_LOG_FILE).display()
        );
    }
    println!();
    server::serve(opts).await
}

// ------------------------------------------------------------------ decide

/// `cortiq decide`.
#[derive(Args, Debug, Clone, PartialEq)]
pub struct DecideArgs {
    /// Decision file (.cmf with the DECISION feature bit)
    pub model: PathBuf,
    /// One text to decide
    #[arg(
        short = 'p',
        long = "prompt",
        required_unless_present = "input",
        conflicts_with = "input"
    )]
    pub prompt: Option<String>,
    /// Batch mode: JSONL rows {"text","label"?}; one JSON result per row
    /// (never the text) on stdout or --out, totals on stderr
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// Skill to decide with (default: from --labels, else the file's only skill)
    #[arg(long)]
    pub skill: Option<String>,
    /// Candidate labels, comma-separated, matched to a skill like a decisions
    /// API question (exact or subset of one skill's labels)
    #[arg(long, value_delimiter = ',', conflicts_with = "input")]
    pub labels: Vec<String>,
    /// State directory of a decision server: decide with the generation its
    /// CURRENT names (base + overlay)
    #[arg(long)]
    pub state: Option<PathBuf>,
    /// Print the whole response (the decisions API shape) as one JSON line
    #[arg(long, conflicts_with = "input")]
    pub json: bool,
    /// Round probabilities to hundredths, as Jev (only 2 is accepted)
    #[arg(long, conflicts_with = "input")]
    pub round: Option<u8>,
    /// Batch mode: write the result rows to this file (never overwritten)
    #[arg(long, conflicts_with = "prompt")]
    pub out: Option<PathBuf>,
    /// Batch mode: 50 warm-up texts, then p50/p95/p99 of every stage
    #[arg(long, conflicts_with = "prompt")]
    pub bench: bool,
}

/// Open a decision file, with the generation `CURRENT` of `state` names when
/// a state directory is given (it must exist: nothing is created for a read).
fn open_model(path: &Path, state: Option<&Path>, verify: Verify) -> Result<DecisionModel> {
    match state {
        None => DecisionModel::open(path, verify)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("open {}", path.display())),
        Some(dir) => {
            let st = existing_state(dir)?;
            generation::open_served(path, &st, verify)
                .with_context(|| format!("open {} with state {}", path.display(), dir.display()))
        }
    }
}

fn existing_state(dir: &Path) -> Result<StateDir> {
    ensure!(
        dir.is_dir(),
        "state directory {} does not exist",
        dir.display()
    );
    StateDir::open(dir)
}

fn api_error(e: ApiError) -> anyhow::Error {
    match &e.details {
        Some(d) if !d.is_empty() => anyhow::anyhow!("{e} {}", Value::Object((**d).clone())),
        _ => anyhow::anyhow!("{e}"),
    }
}

/// `cortiq decide`.
pub fn run_decide(a: &DecideArgs) -> Result<()> {
    if let Some(out) = &a.out {
        ensure!(
            !out.exists(),
            "refusing to overwrite existing output {}",
            out.display()
        );
    }
    let model = open_model(&a.model, a.state.as_deref(), Verify::Light)?;
    match &a.input {
        Some(input) => decide_batch(&model, a, input),
        None => decide_one(model, a),
    }
}

fn decide_batch(model: &DecisionModel, a: &DecideArgs, input: &Path) -> Result<()> {
    let skill = eval::select_skill(model, a.skill.as_deref())?;
    let ev = Evaluator::new(model, &skill)?;
    let inputs = eval::read_input(input)?;
    let opts = EvalOptions {
        bench: a.bench,
        warmup: if a.bench { eval::BENCH_WARMUP } else { 0 },
    };
    let summary = match &a.out {
        Some(path) => {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("create {}", path.display()))?;
            let mut w = BufWriter::new(file);
            let run = ev
                .run(&inputs, opts, |r| write_row(&mut w, &r.to_json()))
                .and_then(|s| {
                    w.flush()?;
                    w.get_ref().sync_all()?;
                    Ok(s)
                });
            if run.is_err() {
                // No partial output is left behind.
                let _ = std::fs::remove_file(path);
            }
            run?
        }
        None => {
            let mut w = BufWriter::new(std::io::stdout().lock());
            let s = ev.run(&inputs, opts, |r| write_row(&mut w, &r.to_json()))?;
            w.flush()?;
            s
        }
    };
    eprintln!("{}", summary.render());
    eprintln!("{}", json!({ "summary": summary.to_json() }));
    Ok(())
}

fn write_row(w: &mut impl Write, v: &Value) -> Result<()> {
    serde_json::to_writer(&mut *w, v)?;
    w.write_all(b"\n")?;
    Ok(())
}

/// The request of `cortiq decide -p`: the text as a string state and one
/// `task` choice question (spec §4.1: `--skill`, else `--labels` by the
/// matching rules of §4.5, else the only skill of the file).
fn single_request(model: &LoadedModel, a: &DecideArgs, text: &str) -> Result<Value> {
    let rubric_of = |id: &str| model.skill(id).and_then(|s| s.manifest().rubric.clone());
    let (question, forced) = if a.labels.is_empty() {
        let id = eval::select_skill(model.model(), a.skill.as_deref())?;
        let s = model.skill(&id).expect("selected skills exist");
        let q = server::route_question(s);
        let mut v = json!({"type": "choice", "instructions": q.instructions});
        v["criteria"] = q.criteria.unwrap_or(Value::Null);
        (v, Some(id))
    } else {
        if let Some(id) = &a.skill {
            eval::select_skill(model.model(), Some(id))?;
        }
        let rubric = a.skill.as_deref().and_then(rubric_of);
        let described = rubric
            .as_ref()
            .map(|r| r.ordered_criteria())
            .unwrap_or_default();
        let mut criteria = Map::new();
        for l in &a.labels {
            ensure!(!l.is_empty(), "--labels has an empty label");
            ensure!(!criteria.contains_key(l), "--labels names '{l}' twice");
            criteria.insert(l.clone(), described.get(l).cloned().unwrap_or(Value::Null));
        }
        let instructions = rubric.map_or_else(
            || server::DEFAULT_ROUTE_INSTRUCTIONS.to_string(),
            |r| r.instructions,
        );
        (
            json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            a.skill.clone(),
        )
    };
    let mut body = json!({
        "model": MODEL_ID,
        "state": text,
        "questions": {server::ROUTE_QUESTION_ID: question},
    });
    let mut cmf = Map::new();
    if let Some(id) = forced {
        cmf.insert("skill".into(), Value::String(id));
    }
    if let Some(r) = a.round {
        cmf.insert("round".into(), json!(r));
    }
    if !cmf.is_empty() {
        body["cmf"] = Value::Object(cmf);
    }
    Ok(body)
}

fn decide_one(model: DecisionModel, a: &DecideArgs) -> Result<()> {
    let text = a.prompt.as_deref().expect("clap: -p or --input");
    let handle = Arc::new(ModelHandle::new(LoadedModel::new(model)?));
    let body = single_request(&handle.current(), a, text)?;
    // No escalator and the oracle disabled: `cortiq decide` never calls it.
    let svc = DecisionService::open(Arc::clone(&handle), Config::default(), None)?;
    let decided = svc
        .decide_body(&serde_json::to_vec(&body)?, &Principal::open())
        .map_err(api_error)?;
    if a.json {
        println!("{}", decided.response);
    } else {
        print!("{}", render_decided(&decided));
    }
    Ok(())
}

fn num(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        other => other.to_string(),
    }
}

/// Human-readable lines of a decided `cortiq decide -p`.
fn render_decided(d: &Decided) -> String {
    let r = &d.response;
    let mut s = String::new();
    for o in &d.questions {
        s.push_str(&render_question(o));
    }
    s.push_str(&format!(
        "model:      {} (generation {}), {} input tokens, {} µs\n",
        r["model"].as_str().unwrap_or("?"),
        num(&r["cmf"]["generation"]),
        num(&r["usage"]["input_tokens"]),
        num(&r["cmf"]["timings_us"]["total"]),
    ));
    s
}

fn render_question(o: &QuestionOutcome) -> String {
    let mut s = String::new();
    let Some(l) = &o.local else {
        // Only exact and subset questions reach the output without an oracle.
        return format!("{}: {} ({})\n", o.id, o.action.as_str(), o.answer);
    };
    s.push_str(&format!(
        "choice:     {}\n",
        l.choice.as_deref().unwrap_or("-")
    ));
    let why = if o.action.as_str() == "local" {
        "accepted by the gate".to_string()
    } else {
        format!(
            "the gate rejected it; `cortiq decide` never calls the oracle{}",
            if o.flags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", o.flags.join(", "))
            }
        )
    };
    s.push_str(&format!(
        "action:     {} ({why}), certified {}\n",
        o.action.as_str(),
        o.certified
    ));
    s.push_str(&format!(
        "skill:      {} ({} match, {} candidates)\n",
        l.skill,
        o.matched.kind.as_str(),
        l.labels.len()
    ));
    let d = &l.decision;
    s.push_str(&format!(
        "gate:       p_top {} (tau {}), novelty {} (theta {}), margin {}, confidence {}\n",
        d.p_top, l.gate.tau, d.novelty, l.gate.novelty_theta, d.margin, l.confidence
    ));
    let errors: Vec<String> = l
        .ranked_errors(eval::TOP_ERRORS)
        .into_iter()
        .map(|(label, e)| format!("{label} {e}"))
        .collect();
    s.push_str(&format!("errors:     {}\n", errors.join(", ")));
    s
}

// ------------------------------------------------------------------ decision …

/// The inputs of one skill (`train` and `add-skill`).
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct SkillArgs {
    /// Skill id ([a-z0-9][a-z0-9_-]{0,63})
    #[arg(long)]
    pub skill: String,
    /// Training JSONL {"text","label"}; repeat to train on the union of
    /// several files, read in order (e.g. train and dev)
    #[arg(long = "train", required = true)]
    pub train: Vec<PathBuf>,
    /// Calibration JSONL (default: carved out of the training rows, every
    /// fifth row of a label in sha256 order)
    #[arg(long)]
    pub calibration: Option<PathBuf>,
    /// Dev JSONL: decided after the build, the correct count is recorded
    #[arg(long)]
    pub dev: Option<PathBuf>,
    /// Question JSON {"instructions","criteria"} (criteria keys = the labels),
    /// stored as the skill's rubric (the oracle needs it)
    #[arg(long)]
    pub question: Option<PathBuf>,
    /// K, the most directions per task (k = min(K, n-1))
    #[arg(long, default_value_t = cortiq_decision::fit::DEFAULT_K)]
    pub k: usize,
    /// Where K came from (recorded in the skill's recipe), e.g. a CV report
    #[arg(long)]
    pub k_source: Option<String>,
    /// Encoder, fit and scoring threads (0 = all cores; the output does not
    /// depend on it)
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// Print the build report as one JSON line
    #[arg(long)]
    pub json: bool,
}

impl SkillArgs {
    fn options(&self) -> TrainOptions {
        let mut o = TrainOptions::new(self.skill.clone(), self.train.clone());
        o.calibration = self.calibration.clone();
        o.dev = self.dev.clone();
        o.question = self.question.clone();
        o.k = self.k;
        o.k_source = self.k_source.clone();
        o.threads = self.threads;
        o
    }
}

/// `--state DIR` (and the configuration that names plans and the key prefix)
/// of `cortiq decision keys …`.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct KeyState {
    /// State directory of the decision server (created when missing)
    #[arg(long)]
    pub state: PathBuf,
    /// Decision server configuration (plans, key prefix)
    #[arg(long)]
    pub decision_config: Option<PathBuf>,
}

/// `cortiq decision keys import --help` (spec §4.15).
const KEYS_IMPORT_HELP: &str = "\
Import the API keys of cortiq-router: a JSON export of its MySQL api_keys
table (--format mysql-json) or the [[api_keys]] of its TOML configuration
(--format router-toml, raw keys hashed as they are read). Only sha256
hashes are stored, the same digest the router stores, so the keys keep
working without reissue. With --usage the router's usage_counters continue
in the usage ledger, so quotas continue. Every input is checked before
anything is written; the summary names no key and no hash.

Exports from the router's database (JSON lines, one row per line):

  mysql -N -B -r -e \"SELECT JSON_OBJECT('key_hash',key_hash,
    'account',account,'plan',plan,'email',email,'label',label,
    'active',active,'rate_per_min',rate_per_min,
    'decision_quota',decision_quota,'expires_at',expires_at,
    'created_at',created_at) FROM api_keys\" DB > api_keys.jsonl

  mysql -N -B -r -e \"SELECT JSON_OBJECT('account',account,
    'decisions',decisions,'oracle_calls',oracle_calls)
    FROM usage_counters\" DB > usage_counters.jsonl

MySQL Shell (--json, --result-format=json/array or ndjson), MySQL Workbench
and phpMyAdmin JSON exports are read too.

Idempotent: a second import of the same export writes nothing. A key already
stored is never overwritten, an active key the export marks inactive is
revoked, nothing is re-activated or deleted. A key in both the database and
the configuration follows the router in either import order: the database
row while it is active and unexpired, else the static key of the
configuration. Accounts, plans and labels are taken as the router's columns
hold them (any text up to 128, 64 and 255 characters). Limits of 0 stay
unlimited; the email column is not stored.

Oracle: imported keys may escalate to the oracle (oracle_allowed true), as
every key could in cortiq-router; the server's oracle switch, budgets and
stop rules still apply. --oracle-allowed=false imports them without it.
Given explicitly (true or false), the value also reaches the keys an earlier
import brought from the router; without it those keep theirs.

Learning: imported keys do not teach the shared model (learning_allowed
false): their /v1/feedback is answered and consumed but not learned, and the
oracle's answers teach it only for the skill's own question (the router's
route question is one). --learning-allowed gives them the router's
behaviour; given explicitly, the value also reaches keys imported before.";

/// `cortiq decision keys …` (spec §4.10, §4.15, §5b).
#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum KeysCmd {
    /// Create an API key; the key is printed once, only its sha256 is stored
    Create {
        #[command(flatten)]
        at: KeyState,
        /// Plan: starter | developer | pro | scale (or the configuration's)
        #[arg(long)]
        plan: Option<String>,
        /// Account (default acct_ + 12 hex)
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        label: Option<String>,
        /// Lifetime in days (0 = never expires; default: the plan's)
        #[arg(long)]
        days: Option<u32>,
        /// Requests per minute (0 = unlimited; default: the plan's)
        #[arg(long)]
        rate_per_min: Option<u32>,
        /// Answered questions allowed (0 = unlimited; default: the plan's)
        #[arg(long)]
        decision_quota: Option<u64>,
        /// Input tokens allowed (0 = unlimited)
        #[arg(long)]
        token_quota: Option<u64>,
        /// Spending limit in USD (decimal string)
        #[arg(long)]
        credit_usd: Option<String>,
        /// Oracle spending limit of the key in USD (decimal string)
        #[arg(long)]
        oracle_budget_usd: Option<String>,
        /// Allow this key's undetermined questions to reach the oracle
        #[arg(long)]
        oracle_allowed: bool,
        /// Allow this key to teach the shared model: its feedback, cold
        /// starts of new labels, and the oracle's answers to its own
        /// questions (default: no key teaches it)
        #[arg(long)]
        learning_allowed: bool,
        /// Print the created key as one JSON line
        #[arg(long)]
        json: bool,
    },
    /// List the keys (hash prefix, limits; never a key)
    List {
        #[command(flatten)]
        at: KeyState,
        #[arg(long)]
        json: bool,
    },
    /// Revoke the keys of an account, or one key by its hash prefix
    #[command(group(ArgGroup::new("which").required(true).args(["account", "hash"])))]
    Revoke {
        #[command(flatten)]
        at: KeyState,
        #[arg(long)]
        account: Option<String>,
        /// At least 12 hex characters of the key's sha256
        #[arg(long)]
        hash: Option<String>,
    },
    /// Import the API keys of cortiq-router: a JSON export of its MySQL
    /// api_keys table or the [[api_keys]] of its TOML configuration; the keys
    /// keep working without reissue (only sha256 hashes are stored). With
    /// --usage its usage_counters continue in the usage ledger. Idempotent;
    /// the summary names no key and no hash
    #[command(long_about = KEYS_IMPORT_HELP)]
    #[command(group(ArgGroup::new("what").required(true).multiple(true).args(["from", "usage"])))]
    Import {
        #[command(flatten)]
        at: KeyState,
        /// Keys: rows of the router's api_keys table as JSON (an array,
        /// JSON lines, MySQL Shell --json, phpMyAdmin), or its configuration
        #[arg(long)]
        from: Option<PathBuf>,
        /// Format of --from (default: .toml = router-toml, .json/.jsonl =
        /// mysql-json, else by the content)
        #[arg(long, requires = "from", value_parser = ImportFormat::NAMES)]
        format: Option<String>,
        /// Usage: rows of the router's usage_counters table as JSON; the
        /// ledger is written, so no server may hold the state directory
        #[arg(long)]
        usage: Option<PathBuf>,
        /// Whether the imported keys may escalate to the oracle (default:
        /// true for new keys, as in cortiq-router; given explicitly, also for
        /// keys imported before)
        #[arg(
            long,
            requires = "from",
            value_name = "BOOL",
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            value_parser = clap::value_parser!(bool)
        )]
        oracle_allowed: Option<bool>,
        /// Whether the imported keys may teach the shared model (feedback,
        /// new labels, the oracle's answers to their own questions; default:
        /// false for new keys; given explicitly, also for keys imported
        /// before)
        #[arg(
            long,
            requires = "from",
            value_name = "BOOL",
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            value_parser = clap::value_parser!(bool)
        )]
        learning_allowed: Option<bool>,
        /// Print the summary as one JSON object
        #[arg(long)]
        json: bool,
    },
}

/// `cortiq decision …`.
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum DecisionCmd {
    /// Pack an exported encoder (tools/decision_export_encoder.py) into an
    /// encoder-only decision file
    Init {
        /// Export directory (encoder.json, *.npy, vocab.txt, tokenizer.json)
        #[arg(long)]
        encoder_dir: PathBuf,
        /// Output .cmf (never overwritten)
        #[arg(short = 'o', long = "output", visible_alias = "out")]
        out: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Train one skill on the encoder of a decision file: the encoder byte for
    /// byte plus the new skill
    Train {
        /// Decision file whose encoder is used
        #[arg(long)]
        encoder: PathBuf,
        #[command(flatten)]
        skill: SkillArgs,
        /// Output .cmf (never overwritten)
        #[arg(short = 'o', long = "output", visible_alias = "out")]
        out: PathBuf,
    },
    /// Add a skill to a decision file: every existing skill byte for byte plus
    /// the new one
    AddSkill {
        /// Input decision file
        input: PathBuf,
        #[command(flatten)]
        skill: SkillArgs,
        /// Output .cmf (never overwritten)
        #[arg(short = 'o', long = "output", visible_alias = "out")]
        out: PathBuf,
    },
    /// Pre-train a skill through the oracle on unlabelled traffic: only the
    /// texts the gate rejects are asked (answers reused from ledgers first)
    Learn {
        /// Input decision file
        input: PathBuf,
        /// Traffic JSONL {"text"} (a label is ignored)
        #[arg(long)]
        traffic: PathBuf,
        /// Skill (default: the file's only skill)
        #[arg(long)]
        skill: Option<String>,
        /// Decision configuration JSON with the "oracle" section (or the
        /// oracle section alone); the key is read from its api_key_env
        #[arg(long)]
        oracle_config: PathBuf,
        /// Answer ledgers of the oracle driver, reused by request sha256
        #[arg(long, num_args = 1..)]
        answers: Vec<PathBuf>,
        /// Reservation ledger of live calls (default <OUTPUT>.oracle.jsonl
        /// when oracle.enabled)
        #[arg(long)]
        oracle_ledger: Option<PathBuf>,
        /// Encoder and fit threads (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Output .cmf (self-contained, never overwritten)
        #[arg(short = 'o', long = "output", visible_alias = "out")]
        out: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Describe a decision file: encoder, skills, labels, gates
    Info {
        model: PathBuf,
        /// The decision manifest and every skill manifest as one JSON line
        #[arg(long)]
        json: bool,
    },
    /// Check every sha256, the hashing contract and the encoder golden
    Verify {
        model: PathBuf,
        /// Verify the generation CURRENT of this state directory names too
        #[arg(long)]
        state: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Write the served model (base + the generation CURRENT names) as one
    /// self-contained file
    Materialize {
        model: PathBuf,
        #[arg(long)]
        state: PathBuf,
        /// Output .cmf (never overwritten)
        #[arg(short = 'o', long = "output", visible_alias = "out")]
        out: PathBuf,
    },
    /// Serve generation N from now on (0 = the base file), without a server
    Rollback {
        #[arg(long)]
        state: PathBuf,
        #[arg(long)]
        to: u64,
        /// The base file (checks the generation against it; needed for --to 0
        /// when there is no generation yet)
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// API keys of a state directory
    Keys {
        #[command(subcommand)]
        cmd: KeysCmd,
    },
}

/// `cortiq decision …`.
pub fn run_decision(cmd: &DecisionCmd) -> Result<()> {
    match cmd {
        DecisionCmd::Init {
            encoder_dir,
            out,
            json,
        } => {
            let r = build::init_encoder(encoder_dir, out, None)?;
            if *json {
                println!("{}", write_report_json(&r));
            } else {
                println!("wrote {}", render_write_report(&r));
            }
            Ok(())
        }
        DecisionCmd::Train {
            encoder,
            skill,
            out,
        } => {
            let r = build::train(encoder, &skill.options(), out)?;
            print_build(&r, skill.json);
            Ok(())
        }
        DecisionCmd::AddSkill { input, skill, out } => {
            let r = build::add_skill(input, &skill.options(), out)?;
            print_build(&r, skill.json);
            Ok(())
        }
        DecisionCmd::Learn {
            input,
            traffic,
            skill,
            oracle_config,
            answers,
            oracle_ledger,
            threads,
            out,
            json,
        } => {
            let cfg = load_oracle_config(oracle_config)?;
            let mut o = OfflineOptions::new(traffic, cfg.oracle.clone());
            o.skill = skill.clone();
            o.answers = answers.clone();
            o.dedup = cfg.learning.dedup;
            o.threads = *threads;
            o.ledger = match oracle_ledger {
                Some(p) => Some(p.clone()),
                None if cfg.oracle.enabled => Some(suffixed(out, ".oracle.jsonl")),
                None => None,
            };
            let r = learn::learn_offline(input, &o, oracle::process_env(), out)?;
            if *json {
                println!("{}", r.to_json());
            } else {
                print!("{}", render_learn(&r, o.ledger.as_deref()));
            }
            Ok(())
        }
        DecisionCmd::Info { model, json } => info(model, *json),
        DecisionCmd::Verify { model, state, json } => verify(model, state.as_deref(), *json),
        DecisionCmd::Materialize { model, state, out } => {
            ensure!(
                !out.exists(),
                "refusing to overwrite existing output {}",
                out.display()
            );
            let st = existing_state(state)?;
            let m = generation::open_served(model, &st, Verify::Full).with_context(|| {
                format!("open {} with state {}", model.display(), state.display())
            })?;
            let generation = m.generation();
            let r = container::materialize(&m, out)?;
            println!(
                "wrote {} (generation {generation} of {} materialised)",
                render_write_report(&r),
                model.display()
            );
            Ok(())
        }
        DecisionCmd::Rollback { state, to, model } => {
            let st = existing_state(state)?;
            let cur = generation::rollback_state(&st, *to, model.as_deref())?;
            println!(
                "CURRENT = {} {} (generation {}{})",
                generation_name(cur.generation),
                cur.sha256,
                cur.generation,
                if cur.generation == 0 {
                    ", the base file"
                } else {
                    ""
                }
            );
            for g in generation::list(&st)? {
                println!(
                    "  {} {} parent {} events {}{}",
                    generation_name(g.generation),
                    &g.sha256[..12],
                    g.parent,
                    g.events.len(),
                    if g.current { "  <- CURRENT" } else { "" }
                );
            }
            Ok(())
        }
        DecisionCmd::Keys { cmd } => keys(cmd),
    }
}

fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// `--oracle-config`: a decision configuration (its `oracle` and
/// `learning.dedup`), or the oracle section alone.
fn load_oracle_config(path: &Path) -> Result<Config> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    match Config::from_json(&bytes) {
        Ok(c) => Ok(c),
        Err(full) => {
            let v: Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("{} is not JSON", path.display()))?;
            if v.as_object().is_some_and(|m| !m.contains_key("oracle")) {
                let wrapped = serde_json::to_vec(&json!({ "oracle": v }))?;
                if let Ok(c) = Config::from_json(&wrapped) {
                    return Ok(c);
                }
            }
            Err(full).with_context(|| format!("{}", path.display()))
        }
    }
}

fn write_report_json(r: &WriteReport) -> Value {
    json!({
        "path": r.path.display().to_string(),
        "sha256": r.sha256,
        "bytes": r.bytes,
        "model_sha": r.model_sha,
        "model": model_name(&r.model_sha),
        "tensors": r.tensors,
    })
}

fn render_write_report(r: &WriteReport) -> String {
    format!(
        "{} ({} bytes, sha256 {}, model {}, {} tensors)",
        r.path.display(),
        r.bytes,
        r.sha256,
        model_name(&r.model_sha),
        r.tensors
    )
}

fn gate_line(g: &Gate) -> String {
    format!(
        "T {} theta {} tau {} certified {}",
        g.temperature, g.novelty_theta, g.tau, g.certified
    )
}

fn print_build(r: &BuildReport, json: bool) {
    if json {
        println!("{}", r.to_json());
        return;
    }
    let m = &r.manifest;
    let active = m.tasks.iter().filter(|t| t.is_active()).count();
    println!("wrote {}", render_write_report(&r.out));
    println!("  skills:      {}", r.skills.join(", "));
    println!(
        "  skill {}: {} labels ({} active), K {}, rows train {} / calibration {} ({})",
        r.skill,
        m.labels.len(),
        active,
        m.recipe.k_max,
        m.rows.n_train,
        m.rows.n_calibration,
        m.data.calibration.source
    );
    let odd = &m.gate.evidence.odd;
    println!(
        "  gate:        {}; odd half accepted {} / correct {} of {}",
        gate_line(&m.gate),
        odd.accepted,
        odd.correct,
        odd.n
    );
    println!(
        "  calibration: {}/{} correct",
        r.calibration_correct, m.data.calibration.n
    );
    if let Some(d) = r.dev {
        println!(
            "  dev:         {}/{} correct; gate accepted {} ({} correct)",
            d.correct, d.n, d.accepted, d.accepted_correct
        );
    }
    println!(
        "  self-check:  bit-exact ({} calibration rows, {} errors, {} texts re-encoded)",
        r.self_check.calibration_rows, r.self_check.errors_compared, r.self_check.texts_reencoded
    );
    println!(
        "  time:        {:.1} s ({} texts encoded on {} threads)",
        r.timings.total.as_secs_f64(),
        r.texts_encoded,
        r.threads
    );
    for w in &r.warnings {
        println!("  warning:     {w}");
    }
}

fn render_learn(r: &OfflineReport, ledger: Option<&Path>) -> String {
    let mut s = format!("wrote {}\n", render_write_report(&r.out));
    s.push_str(&format!(
        "  skill {}: {} texts ({} with a label, ignored), {} accepted locally, {} abstained\n",
        r.skill, r.texts, r.labelled, r.accepted, r.abstained
    ));
    let refused: Vec<String> = r
        .refused_calls
        .iter()
        .map(|(k, v)| format!("{k} {v}"))
        .collect();
    s.push_str(&format!(
        "  oracle:      {} answers reused, {} calls sent ({} failed), refused: {}, {} unanswered, ${}\n",
        r.answers_reused,
        r.live_calls,
        r.failed_calls,
        if refused.is_empty() {
            "none".to_string()
        } else {
            refused.join(", ")
        },
        r.unanswered,
        r.oracle_spent_usd
    ));
    if let Some(l) = ledger {
        s.push_str(&format!("  ledger:      {}\n", l.display()));
    }
    s.push_str(&format!(
        "  examples:    {} ({} duplicates); promoted: {}; rejected: {}{}\n",
        r.examples,
        r.duplicates,
        list_or_dash(&r.promoted_labels),
        list_or_dash(&r.rejected_labels),
        if r.rolled_back {
            " (every promotion undone: the certified gate was lost)"
        } else {
            ""
        }
    ));
    s.push_str(&format!("  gate before: {}\n", gate_line(&r.gate_before)));
    s.push_str(&format!("  gate after:  {}\n", gate_line(&r.gate_after)));
    s
}

fn list_or_dash(v: &[String]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

// ------------------------------------------------------------------ info / verify

fn skill_summary(m: &SkillManifest) -> Value {
    let count = |s: TaskState| m.tasks.iter().filter(|t| t.state == s).count();
    let odd = &m.gate.evidence.odd;
    let lb = odd
        .grid
        .iter()
        .find(|e| m.gate.certified && e.t as f32 as f64 == m.gate.tau)
        .map(|e| e.lb);
    json!({
        "id": m.id,
        "taxonomy_version": m.taxonomy_version,
        "labels": m.labels.len(),
        "tasks": {
            "active": count(TaskState::Active),
            "inactive": count(TaskState::Inactive),
            "quarantined": count(TaskState::Quarantined),
        },
        "K": m.recipe.k_max,
        "k_source": m.recipe.k_source,
        "gate": {
            "temperature": m.gate.temperature,
            "novelty_theta": m.gate.novelty_theta,
            "tau": m.gate.tau,
            "certified": m.gate.certified,
            "odd_half": {"n": odd.n, "accepted": odd.accepted, "correct": odd.correct, "lb": lb},
        },
        "rows": {
            "train": m.rows.n_train,
            "calibration": m.rows.n_calibration,
            "learned": m.rows.n_learned + m.rows_learned.as_ref().map_or(0, |r| r.n),
        },
        "data": {
            "train": {"n": m.data.train.n, "sha256": m.data.train.sha256, "parts": m.data.train.parts.len().max(1)},
            "calibration": {"n": m.data.calibration.n, "source": m.data.calibration.source},
            "dev": m.data.dev.as_ref().map(|d| json!({"n": d.n, "correct": d.correct})),
        },
        "rubric": m.rubric.is_some(),
        "learned": m.learned.as_ref().map(|l| json!({
            "oracle_model": l.oracle_model, "calls": l.calls, "answers_reused": l.answers_reused,
            "promoted_labels": l.promoted_labels, "rejected_labels": l.rejected_labels,
        })),
    })
}

fn info(path: &Path, as_json: bool) -> Result<()> {
    let model = open_model(path, None, Verify::Light)?;
    let bytes = std::fs::metadata(path)?.len();
    let manifest = model.manifest();
    let rep = model.representation();
    if as_json {
        let skills: Vec<Value> = model
            .skills()
            .iter()
            .map(|s| serde_json::to_value(&s.manifest))
            .collect::<Result<_, _>>()?;
        let v = json!({
            "path": path.display().to_string(),
            "bytes": bytes,
            "model_sha": model.model_sha(),
            "model": model_name(model.model_sha()),
            "manifest": serde_json::to_value(manifest)?,
            "skills": skills,
            "summary": model.skills().iter().map(|s| skill_summary(&s.manifest)).collect::<Vec<_>>(),
            "warnings": model.warnings(),
        });
        println!("{v}");
        return Ok(());
    }
    let e = &rep.encoder;
    println!("Decision file: {} ({} bytes)", path.display(), bytes);
    println!(
        "  model:          {} ({}, generation {})",
        model_name(model.model_sha()),
        manifest.name,
        manifest.generation
    );
    println!("  model_sha:      {}", model.model_sha());
    println!("  representation: {}", model.representation_id());
    println!(
        "  encoder:        {} {} layers, hidden {}, {} heads, vocab {}, max {} tokens ({})",
        e.kind,
        e.config.layers,
        e.config.hidden,
        e.config.heads,
        e.config.vocab,
        e.tokenizer.truncation.max_length,
        e.source.name
    );
    println!(
        "  signal:         {} = phi_P {} + 0.5 phi_H {}",
        model.signal_dim(),
        model.encoder_dim(),
        model.hashing_dim()
    );
    println!("  skills:         {}", model.skills().len());
    for s in model.skills() {
        let v = skill_summary(&s.manifest);
        let m = &s.manifest;
        println!(
            "    {}: {} labels ({} active, {} inactive, {} quarantined), K {}, rows train {} / calibration {} / learned {}",
            m.id,
            v["labels"],
            v["tasks"]["active"],
            v["tasks"]["inactive"],
            v["tasks"]["quarantined"],
            m.recipe.k_max,
            v["rows"]["train"],
            v["rows"]["calibration"],
            v["rows"]["learned"],
        );
        println!(
            "      gate {}; odd half {}/{} (lb {})",
            gate_line(&m.gate),
            m.gate.evidence.odd.correct,
            m.gate.evidence.odd.accepted,
            num(&v["gate"]["odd_half"]["lb"])
        );
        if let Some(d) = &m.data.dev {
            println!("      dev {}/{} correct", d.correct, d.n);
        }
        if let Some(l) = &m.learned {
            println!(
                "      learned through {}: {} calls, {} answers reused, promoted {}",
                l.oracle_model,
                l.calls,
                l.answers_reused,
                list_or_dash(&l.promoted_labels)
            );
        }
        let labels: Vec<&str> = m.labels.iter().map(String::as_str).collect();
        println!("      labels: {}", labels.join(", "));
    }
    for w in model.warnings() {
        println!("  warning: {w}");
    }
    Ok(())
}

/// `cortiq decision verify`: the file opened with [`Verify::Full`] — the base's
/// bytes against its own manifests (hash64 integrity, every sha256, the rows
/// blobs, finite topologies), then, with `--state`, the generation `CURRENT`
/// names (its integrity, every replaced tensor's sha256, the learned rows) —
/// plus the encoder golden and a scorer per skill (what `serve` builds).
///
/// The open with [`Verify::Full`] already runs every check of
/// `DecisionModel::verify_full` (the base file against its own skill
/// manifests, a generation's replaced skills against the overlay's), so it is
/// not run a second time.
fn verify(path: &Path, state: Option<&Path>, as_json: bool) -> Result<()> {
    let model = open_model(path, state, Verify::Full)?;
    let (_, golden) = SignalEncoder::from_model(&model)?;
    for s in model.skills() {
        SkillScorer::from_model(&model, s.id())?;
    }
    let mut tensors = model.base().tensors.len();
    let mut bytes: u64 = model.base().tensors.iter().map(|e| e.nbytes).sum();
    if let Some(o) = model.overlay() {
        tensors += o.tensors.len();
        bytes += o.tensors.iter().map(|e| e.nbytes).sum::<u64>();
    }
    if as_json {
        println!(
            "{}",
            json!({
                "ok": true,
                "model_sha": model.model_sha(),
                "base_model_sha": model.base_model_sha(),
                "generation": model.generation(),
                "skills": model.skills().len(),
                "tensors": tensors,
                "tensor_bytes": bytes,
                "golden": {"rows": golden.rows, "bit_exact_rows": golden.bit_exact_rows, "max_abs": golden.max_abs},
                "warnings": model.warnings(),
            })
        );
        return Ok(());
    }
    println!("Verifying {} ...", path.display());
    println!("  ✓ envelope, DECISION bit, manifests, hashing contract");
    println!("  ✓ every sha256 and the rows blobs ({tensors} tensors, {bytes} bytes)");
    println!(
        "  ✓ encoder golden: {}/{} texts bit-exact (max |Δ| {})",
        golden.bit_exact_rows, golden.rows, golden.max_abs
    );
    println!("  ✓ {} skills load", model.skills().len());
    println!(
        "  model {} (generation {})",
        model_name(model.model_sha()),
        model.generation()
    );
    for w in model.warnings() {
        println!("  warning: {w}");
    }
    println!("OK");
    Ok(())
}

// ------------------------------------------------------------------ keys

fn key_store(at: &KeyState) -> Result<(KeyStore, Config)> {
    let cfg = match &at.decision_config {
        Some(p) => Config::load(p)?,
        None => Config::default(),
    };
    let dir = StateDir::open(&at.state)
        .with_context(|| format!("state directory {}", at.state.display()))?;
    Ok((KeyStore::open(dir.keys_path(), &cfg.auth.key_prefix)?, cfg))
}

fn keys(cmd: &KeysCmd) -> Result<()> {
    let now = now_unix();
    match cmd {
        KeysCmd::Create {
            at,
            plan,
            account,
            label,
            days,
            rate_per_min,
            decision_quota,
            token_quota,
            credit_usd,
            oracle_budget_usd,
            oracle_allowed,
            learning_allowed,
            json,
        } => {
            let (store, cfg) = key_store(at)?;
            let new = NewKey {
                plan: plan.clone(),
                account: account.clone(),
                label: label.clone(),
                days: *days,
                rate_per_min: *rate_per_min,
                decision_quota: *decision_quota,
                token_quota: *token_quota,
                credit_usd: credit_usd.clone(),
                oracle_budget_usd: oracle_budget_usd.clone(),
                oracle_allowed: oracle_allowed.then_some(true),
                learning_allowed: learning_allowed.then_some(true),
            };
            let created = store.create(&new, &cfg.auth.plans, now)?;
            if *json {
                println!("{}", created.to_json(now));
            } else {
                let r = &created.record;
                println!("{}", created.raw);
                eprintln!(
                    "created key {} for account {} (plan {}, {} requests/min, decision quota {}, expires {}); \
                     the key is shown only now, {} keeps its sha256",
                    r.hash12(),
                    keys_mod::shown(&r.account),
                    keys_mod::shown(&r.plan),
                    r.rate_per_min,
                    r.decision_quota,
                    r.expires.map_or("never".to_string(), |e| e.to_string()),
                    store.path().display()
                );
            }
            Ok(())
        }
        KeysCmd::List { at, json } => {
            let (store, _) = key_store(at)?;
            // What each key is now (an imported router key may answer as its
            // static configuration key, as in the router).
            let records: Vec<keys_mod::KeyRecord> = store
                .records()
                .iter()
                .map(|r| r.effective(now).into_owned())
                .collect();
            if *json {
                let list: Vec<Value> = records.iter().map(|r| r.listing(now)).collect();
                println!("{}", Value::Array(list));
                return Ok(());
            }
            println!(
                "{:<12}  {:<24} {:<10} {:<7} {:>8} {:>10} {:>12}  {:<8} learning",
                "hash12", "account", "plan", "state", "rate/min", "quota", "expires", "oracle"
            );
            for r in &records {
                let state = if !r.active {
                    "revoked"
                } else if r.is_expired(now) {
                    "expired"
                } else {
                    "active"
                };
                println!(
                    "{:<12}  {:<24} {:<10} {:<7} {:>8} {:>10} {:>12}  {:<8} {}",
                    r.hash12(),
                    keys_mod::shown(&r.account),
                    keys_mod::shown(&r.plan),
                    state,
                    r.rate_per_min,
                    r.decision_quota,
                    r.expires.map_or("never".to_string(), |e| e.to_string()),
                    if r.oracle_allowed { "allowed" } else { "no" },
                    if r.learning_allowed { "allowed" } else { "no" }
                );
            }
            println!("{} key(s) in {}", records.len(), store.path().display());
            Ok(())
        }
        KeysCmd::Revoke { at, account, hash } => {
            let (store, _) = key_store(at)?;
            let (n, what) = match (account, hash) {
                (Some(a), _) => (
                    store.revoke_account(a)?,
                    format!("account {}", keys_mod::shown(a)),
                ),
                (None, Some(h)) => (store.revoke_hash_prefix(h)?, format!("hash {h}")),
                (None, None) => bail!("--account or --hash is required"),
            };
            ensure!(n > 0, "no active key matches {what}");
            println!("revoked {n} key(s) of {what}");
            Ok(())
        }
        KeysCmd::Import {
            at,
            from,
            format,
            usage,
            oracle_allowed,
            learning_allowed,
            json,
        } => keys_import(
            at,
            &ImportArgs {
                from: from.as_deref(),
                format: format.as_deref(),
                usage: usage.as_deref(),
                oracle_allowed: *oracle_allowed,
                learning_allowed: *learning_allowed,
            },
            *json,
            now,
        ),
    }
}

/// The inputs of `cortiq decision keys import`.
struct ImportArgs<'a> {
    from: Option<&'a Path>,
    format: Option<&'a str>,
    usage: Option<&'a Path>,
    /// `--oracle-allowed[=BOOL]`.
    oracle_allowed: Option<bool>,
    /// `--learning-allowed[=BOOL]`.
    learning_allowed: Option<bool>,
}

/// `cortiq decision keys import` (spec §4.15). Every input is read and
/// checked before anything is written; the usage ledger is written only
/// under the state directory's LOCK.
fn keys_import(at: &KeyState, args: &ImportArgs<'_>, json: bool, now: u64) -> Result<()> {
    let keys_in = args
        .from
        .map(|p| -> Result<_> {
            let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let fmt = match args.format {
                Some(f) => ImportFormat::parse(f)?,
                None => ImportFormat::detect(p, &bytes),
            };
            let mut keys = keys_mod::read_router_keys(&bytes, fmt, now).with_context(|| {
                format!("{} ({}): nothing was imported", p.display(), fmt.name())
            })?;
            if let Some(v) = args.oracle_allowed {
                keys = keys.with_oracle_allowed(v);
            }
            if let Some(v) = args.learning_allowed {
                keys = keys.with_learning_allowed(v);
            }
            Ok((p, keys))
        })
        .transpose()?;
    let usage_in = args
        .usage
        .map(|p| -> Result<_> {
            let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let rows = keys_mod::read_router_usage(&bytes).with_context(|| {
                format!("{} (usage_counters): nothing was imported", p.display())
            })?;
            Ok((p, rows))
        })
        .transpose()?;
    let (store, _) = key_store(at)?;
    let dir = StateDir::open(&at.state)
        .with_context(|| format!("state directory {}", at.state.display()))?;
    let _lock = match &usage_in {
        Some(_) => Some(dir.lock(false).context(
            "importing usage writes the usage ledger: stop the server of this state directory first",
        )?),
        None => None,
    };
    let key_report = keys_in
        .map(|(p, k)| store.import_router_keys(&k, now).map(|r| (p, r)))
        .transpose()?;
    let usage_report = usage_in
        .map(|(p, rows)| -> Result<_> {
            let ledger = UsageLedger::open(dir.usage_dir())?;
            Ok((p, keys_mod::import_router_usage(&ledger, &rows, now)?))
        })
        .transpose()?;
    if json {
        println!(
            "{}",
            json!({
                "keys": key_report.as_ref().map(|(_, r)| r.to_json()),
                "usage": usage_report.as_ref().map(|(_, r)| r.to_json()),
            })
        );
        return Ok(());
    }
    if let Some((p, r)) = &key_report {
        print_key_import(p, r, store.path());
    }
    if let Some((p, r)) = &usage_report {
        print_usage_import(p, r);
    }
    Ok(())
}

/// The human summary of a key import: counts and accounts, never a key or a hash.
fn print_key_import(from: &Path, r: &ImportReport, keys_json: &Path) {
    println!(
        "keys from {} ({}): {} read; {} imported ({} active, {} inactive, {} expired), \
         {} unchanged, {} revoked as in the export, {} kept as they are (differ from the export)",
        from.display(),
        r.format.map_or("-", ImportFormat::name),
        r.read,
        r.imported,
        r.imported_active(),
        r.imported_inactive,
        r.imported_expired,
        r.unchanged,
        r.revoked,
        r.kept
    );
    if !r.accounts.is_empty() {
        let accounts: Vec<String> = r.accounts.iter().map(|a| keys_mod::shown(a)).collect();
        println!("  accounts of the new keys: {}", accounts.join(", "));
    }
    if r.layered > 0 {
        println!(
            "  {} keys in both the router's database and its configuration: the database row \
             is laid over the static key, as in the router",
            r.layered
        );
    }
    if r.static_fallback > 0 {
        println!(
            "  {} keys answer as their static configuration key (database row inactive or \
             expired), as in the router",
            r.static_fallback
        );
    }
    if r.imported > 0 {
        println!(
            "  new keys: oracle escalation {}",
            if r.oracle_allowed {
                "allowed, as in cortiq-router (--oracle-allowed=false to import without it)"
            } else {
                "not allowed (--oracle-allowed=false)"
            }
        );
    }
    if r.oracle_updated > 0 {
        println!(
            "  {} keys imported before: oracle escalation {} (--oracle-allowed={})",
            r.oracle_updated,
            if r.oracle_allowed {
                "allowed"
            } else {
                "withdrawn"
            },
            r.oracle_allowed
        );
    }
    if r.imported > 0 {
        println!(
            "  new keys: teaching the model {}",
            if r.learning_allowed {
                "allowed (--learning-allowed)"
            } else {
                "not allowed: their feedback is answered but not learned (--learning-allowed to allow it)"
            }
        );
    }
    if r.learning_updated > 0 {
        println!(
            "  {} keys imported before: teaching the model {} (--learning-allowed={})",
            r.learning_updated,
            if r.learning_allowed {
                "allowed"
            } else {
                "withdrawn"
            },
            r.learning_allowed
        );
    }
    if r.ignored_empty > 0 {
        println!(
            "  {} entries with an empty key ignored (the router ignores them too)",
            r.ignored_empty
        );
    }
    if r.duplicates > 0 {
        println!(
            "  {} repeated keys: the last entry wins, as in the router",
            r.duplicates
        );
    }
    if r.emails_not_stored > 0 {
        println!(
            "  {} emails not stored (keys.json keeps no email)",
            r.emails_not_stored
        );
    }
    println!(
        "  {} {}",
        keys_json.display(),
        if r.written { "written" } else { "unchanged" }
    );
}

fn print_usage_import(from: &Path, r: &UsageImportReport) {
    println!(
        "usage from {} (usage_counters): {} read; {} carried over (+{} decisions, +{} oracle calls), \
         {} unchanged, {} behind an earlier import (left alone); usage ledger {}",
        from.display(),
        r.read,
        r.carried,
        r.decisions,
        r.oracle_calls,
        r.unchanged,
        r.behind,
        if r.written { "written" } else { "unchanged" }
    );
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Commands};
    use clap::Parser;

    /// Parse on a main-thread-sized stack (the whole command tree of an
    /// unoptimised build exceeds libtest's 2 MiB worker stack).
    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || Cli::try_parse_from(args))
            .unwrap()
            .join()
            .unwrap()
    }

    fn serve_host(args: &[&str]) -> Option<String> {
        match parse(args).unwrap().command {
            Commands::Serve { host, .. } => host,
            _ => panic!("not serve"),
        }
    }

    #[test]
    fn serve_host_defaults_by_file_kind() {
        // A language model keeps the 0.7.7 default (every interface).
        let host = serve_host(&["cortiq", "serve", "llm.cmf"]);
        assert_eq!(host, None);
        assert_eq!(resolve_serve_host(host.as_deref(), false), "0.0.0.0");
        // A decision file listens on loopback only.
        assert_eq!(resolve_serve_host(host.as_deref(), true), "127.0.0.1");
        // An explicit --host wins for both.
        let host = serve_host(&["cortiq", "serve", "m.cmf", "--host", "10.0.0.7"]);
        assert_eq!(resolve_serve_host(host.as_deref(), false), "10.0.0.7");
        assert_eq!(resolve_serve_host(host.as_deref(), true), "10.0.0.7");
        let host = serve_host(&["cortiq", "serve", "m.cmf", "--host", "127.0.0.1"]);
        assert_eq!(resolve_serve_host(host.as_deref(), false), "127.0.0.1");
    }

    #[test]
    fn serve_flags_are_checked_against_the_file_kind() {
        let none = ServeFlags::default();
        let d = ServeFlags {
            decision_config: Some("cfg.json".into()),
            state: Some("st".into()),
            break_lock: true,
            shadow_of: Some("https://router.example.com".into()),
            shadow_timeout_s: Some(300),
        };
        assert_eq!(
            d.given(),
            [
                "--decision-config",
                "--state",
                "--break-lock",
                "--shadow-of",
                "--shadow-timeout-s"
            ]
        );
        // Decision file: no language-model flag.
        assert!(check_serve_flags("d.cmf", true, &[], &d).is_ok());
        let e = check_serve_flags("d.cmf", true, &["--task", "--gpus"], &none).unwrap_err();
        assert!(e.to_string().contains("--task, --gpus"), "{e}");
        // Language model: no decision flag.
        assert!(check_serve_flags("m.cmf", false, &["--task"], &none).is_ok());
        let e = check_serve_flags("m.cmf", false, &[], &d).unwrap_err();
        assert!(e.to_string().contains("--decision-config"), "{e}");
    }

    #[test]
    fn serve_parses_the_decision_flags_and_optional_task() {
        match parse(&[
            "cortiq",
            "serve",
            "d.cmf",
            "--decision-config",
            "c.json",
            "--state",
            "s",
            "--break-lock",
            "--shadow-of",
            "https://router.example.com",
            "--shadow-timeout-s",
            "300",
        ])
        .unwrap()
        .command
        {
            Commands::Serve {
                decision_config,
                state,
                break_lock,
                shadow_of,
                shadow_timeout_s,
                task,
                ..
            } => {
                assert_eq!(decision_config.as_deref(), Some("c.json"));
                assert_eq!(state.as_deref(), Some("s"));
                assert!(break_lock);
                assert_eq!(shadow_of.as_deref(), Some("https://router.example.com"));
                assert_eq!(shadow_timeout_s, Some(300));
                // `--task` is optional so that its presence can be refused on
                // a decision file (a language model still defaults to general).
                assert_eq!(task, None);
            }
            _ => panic!("not serve"),
        }
    }

    #[test]
    fn decide_arguments_parse_and_conflict() {
        match parse(&[
            "cortiq", "decide", "d.cmf", "-p", "hello", "--labels", "a,b,c", "--json", "--round",
            "2",
        ])
        .unwrap()
        .command
        {
            Commands::Decide(a) => {
                assert_eq!(a.prompt.as_deref(), Some("hello"));
                assert_eq!(a.labels, ["a", "b", "c"]);
                assert!(a.json);
                assert_eq!(a.round, Some(2));
            }
            _ => panic!("not decide"),
        }
        match parse(&[
            "cortiq", "decide", "d.cmf", "--input", "r.jsonl", "--skill", "s", "--out", "o.jsonl",
            "--bench",
        ])
        .unwrap()
        .command
        {
            Commands::Decide(a) => {
                assert_eq!(a.input.as_deref(), Some(Path::new("r.jsonl")));
                assert!(a.bench);
            }
            _ => panic!("not decide"),
        }
        // One of -p and --input; batch-only and single-only flags.
        assert!(parse(&["cortiq", "decide", "d.cmf"]).is_err());
        assert!(parse(&["cortiq", "decide", "d.cmf", "-p", "x", "--input", "r"]).is_err());
        assert!(parse(&["cortiq", "decide", "d.cmf", "-p", "x", "--bench"]).is_err());
        assert!(
            parse(&[
                "cortiq", "decide", "d.cmf", "--input", "r", "--labels", "a,b"
            ])
            .is_err()
        );
    }

    #[test]
    fn decision_subcommands_parse() {
        match parse(&[
            "cortiq",
            "decision",
            "train",
            "--encoder",
            "e.cmf",
            "--skill",
            "s",
            "--train",
            "a",
            "--train",
            "b",
            "--k",
            "8",
            "--threads",
            "2",
            "-o",
            "out.cmf",
        ])
        .unwrap()
        .command
        {
            Commands::Decision {
                cmd: DecisionCmd::Train { skill, out, .. },
            } => {
                assert_eq!(skill.train, [PathBuf::from("a"), PathBuf::from("b")]);
                assert_eq!((skill.k, skill.threads), (8, 2));
                assert_eq!(out, PathBuf::from("out.cmf"));
            }
            _ => panic!("not train"),
        }
        match parse(&[
            "cortiq",
            "decision",
            "learn",
            "in.cmf",
            "--traffic",
            "t",
            "--oracle-config",
            "c",
            "--answers",
            "a1",
            "a2",
            "-o",
            "o.cmf",
        ])
        .unwrap()
        .command
        {
            Commands::Decision {
                cmd: DecisionCmd::Learn { answers, .. },
            } => assert_eq!(answers, [PathBuf::from("a1"), PathBuf::from("a2")]),
            _ => panic!("not learn"),
        }
        assert!(parse(&["cortiq", "decision", "keys", "revoke", "--state", "s"]).is_err());
        // keys import: --from and/or --usage; --format needs --from and names a format.
        let import = |extra: &[&str]| {
            let mut a = vec!["cortiq", "decision", "keys", "import", "--state", "s"];
            a.extend_from_slice(extra);
            parse(&a)
        };
        assert!(import(&[]).is_err());
        assert!(import(&["--format", "mysql-json", "--usage", "u.json"]).is_err());
        assert!(import(&["--from", "k.json", "--format", "csv"]).is_err());
        match import(&[
            "--from",
            "router.toml",
            "--format",
            "router-toml",
            "--usage",
            "u.json",
        ])
        .unwrap()
        .command
        {
            Commands::Decision {
                cmd:
                    DecisionCmd::Keys {
                        cmd:
                            KeysCmd::Import {
                                from,
                                format,
                                usage,
                                json,
                                ..
                            },
                    },
            } => assert_eq!(
                (from, format.as_deref(), usage, json),
                (
                    Some(PathBuf::from("router.toml")),
                    Some("router-toml"),
                    Some(PathBuf::from("u.json")),
                    false
                )
            ),
            _ => panic!("not keys import"),
        }
        assert!(import(&["--usage", "u.json", "--json"]).is_ok());
        assert!(
            parse(&[
                "cortiq", "decision", "rollback", "--state", "s", "--to", "0"
            ])
            .is_ok()
        );
        assert!(
            parse(&[
                "cortiq",
                "decision",
                "add-skill",
                "in.cmf",
                "--skill",
                "s",
                "-o",
                "o"
            ])
            .is_err()
        );
    }

    #[test]
    fn oracle_config_accepts_a_full_config_or_the_oracle_section() {
        // Removed on drop, a failed assertion included.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let full = dir.join("full.json");
        std::fs::write(
            &full,
            br#"{"oracle":{"enabled":true,"model":"m/x"},"learning":{"dedup":0.9}}"#,
        )
        .unwrap();
        let c = load_oracle_config(&full).unwrap();
        assert!(c.oracle.enabled);
        assert_eq!(c.oracle.model, "m/x");
        assert_eq!(c.learning.dedup, 0.9);
        let section = dir.join("section.json");
        std::fs::write(&section, br#"{"enabled":false,"model":"m/y"}"#).unwrap();
        let c = load_oracle_config(&section).unwrap();
        assert_eq!(c.oracle.model, "m/y");
        let bad = dir.join("bad.json");
        std::fs::write(&bad, br#"{"oracle":{"nope":1}}"#).unwrap();
        assert!(load_oracle_config(&bad).is_err());
    }
}
