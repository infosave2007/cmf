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
//! * `--oracle MODEL` on either (decision-v4 U2, the oracle in two steps: the
//!   key in `OPENROUTER_API_KEY`, then this flag): the local decision first,
//!   and only a question it cannot decide (the gate rejects it, or no skill
//!   has the labels) goes to the OpenRouter model, through the server's own
//!   cascade and client — the reservation before every call, the stop rules,
//!   PII redaction by default, the key read from the environment at the call
//!   and printed nowhere. A text the gate accepts touches no network and no
//!   state. The reservation ledger and the answer cache are the state
//!   directory's (`--state DIR`, else `<FILE>.state`, under its `LOCK`);
//!   `--oracle-budget` caps what the run spends (default $1.00). The stop
//!   rules hold as on a server: a stop (a refused key, no credit, another
//!   model, a cost above the reservation, `max_errors` failures in a row —
//!   counted across runs) is written to `oracle.state` and keeps the oracle
//!   off for every later run on that directory until `--oracle-resume`; an
//!   interrupted run releases the `LOCK` (SIGINT, SIGTERM, SIGHUP; a signal
//!   inherited as ignored, as SIGHUP under `nohup`, stays ignored), and one
//!   left by a process that is gone (killed, crashed) is taken over by the
//!   next run (`--break-lock` only where the filesystem has no advisory
//!   locks);
//! * `cortiq decision oracle check [--model M] [--key-env VAR] [--base-url
//!   URL] [--max-price IN,OUT] [--test-call] [--json]`: is the oracle ready —
//!   the key, the account (`GET /auth/key`), the model's endpoints, and with
//!   `--test-call` one tiny structured call; exit code 0 only when ready;
//! * the key, on every oracle surface: read from its variable less
//!   surrounding whitespace (a warning says so), `bad_key` when what is left
//!   is not a key (never sent); a missing one is worded for the command line
//!   (`OPENROUTER_API_KEY is not set (decide --oracle reads the key from the
//!   environment)`), also when labels no skill has leave only the oracle.
//!   Without a usable key `decide --oracle` still answers what the state
//!   directory's cache holds (0.8.11: read only, no call, no `LOCK`, nothing
//!   written); the rest is refused `no_key` / `bad_key`;
//! * `cortiq decision init | train | add-skill | learn | info | verify |
//!   materialize | rollback | keys | oracle check`;
//! * `cortiq serve FILE` on a decision file: the decisions server on
//!   127.0.0.1 unless `--host` says otherwise; the language-model flags are
//!   refused.

use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Args, Subcommand};
use cortiq_core::CmfModel;
use cortiq_core::format::features;
use cortiq_decision::build::{self, BuildReport, TrainOptions};
use cortiq_decision::cascade::{CacheView, Cascade};
use cortiq_decision::config::{
    Config, DEFAULT_ORACLE_BASE_URL, DEFAULT_ORACLE_KEY_ENV, DEFAULT_ORACLE_MODEL,
};
use cortiq_decision::container::{self, DecisionModel, Verify, WriteReport};
use cortiq_decision::eval::{
    self, EvalInput, EvalOptions, EvalSummary, Evaluator, RowResult, SkillScorer,
};
use cortiq_decision::generation;
use cortiq_decision::keys::{
    self as keys_mod, ImportFormat, ImportReport, KeyStore, NewKey, UsageImportReport, now_unix,
};
use cortiq_decision::learn::{self, OfflineOptions, OfflineReport};
use cortiq_decision::ledger::UsageLedger;
use cortiq_decision::manifest::{Gate, SkillManifest, TaskState};
use cortiq_decision::oracle::{self, LedgerTotals, OracleState};
use cortiq_decision::oracle_setup::{
    self, CheckOptions, OracleFlags, OracleSetup, host_of, usd, usd_ceil, usd_fine,
};
use cortiq_decision::protocol::{
    self, ApiError, FeedbackRequest, MODEL_ID, SYSTEMONE_MODEL_ID, model_name,
};
use cortiq_decision::service::{
    Action, AdminCommand, Decided, DecisionService, Escalation, EscalationResult, Escalator,
    LoadedModel, LocalDecision, ModelHandle, OracleStatus, Principal, QuestionOutcome,
    RefusalReason, Resolution, Resolved,
};
use cortiq_decision::shadow::{SHADOW_LOG_FILE, upstream_base};
use cortiq_decision::signal::SignalEncoder;
use cortiq_decision::statedir::{Locked, StateDir, StateLock, generation_name};
use cortiq_server::decisions::{self as server, ServeOptions};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
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

/// `--oracle MODEL` and its companions (`cortiq serve` on a decision file):
/// the oracle in two steps — the key in `OPENROUTER_API_KEY`, then this flag.
/// See `cortiq_decision::oracle_setup`.
#[derive(Args, Clone, Debug, Default, PartialEq)]
pub struct OracleArgs {
    /// Decision file only: let this OpenRouter model answer the questions the
    /// local model cannot decide (e.g. deepseek/deepseek-v4.1-flash). The key
    /// is read from OPENROUTER_API_KEY (--oracle-key-env), never from a file.
    /// At start one public GET of the model's endpoint listing (no key sent)
    /// checks that it supports structured outputs and sets the max price to
    /// twice its cheapest endpoint. Overrides --decision-config
    #[arg(long, value_name = "MODEL")]
    pub oracle: Option<String>,
    /// With --oracle: the most this server spends on the oracle, USD
    /// [default: 1.0, or oracle.budget_usd of --decision-config]
    #[arg(long, value_name = "USD", requires = "oracle", value_parser = Quiet(parse_budget))]
    pub oracle_budget: Option<f64>,
    /// With --oracle: the most oracle calls this server makes
    #[arg(long, value_name = "N", requires = "oracle", value_parser = Quiet(parse_calls))]
    pub oracle_max_calls: Option<u64>,
    /// With --oracle: the environment variable holding the OpenRouter key
    /// [default: OPENROUTER_API_KEY]
    #[arg(long, value_name = "VAR", requires = "oracle")]
    pub oracle_key_env: Option<String>,
    /// With --oracle: the OpenRouter API base (https; plain http only to a
    /// loopback address) [default: https://openrouter.ai/api/v1]
    #[arg(long, value_name = "URL", requires = "oracle")]
    pub oracle_base_url: Option<String>,
    /// With --oracle: max price in USD per 1M prompt and completion tokens,
    /// e.g. 0.1,0.5 [default: twice the model's cheapest structured-output
    /// endpoint]
    #[arg(long, value_name = "IN,OUT", requires = "oracle", value_parser = Quiet(parse_max_price))]
    pub oracle_max_price: Option<(f64, f64)>,
    /// With --oracle: do not learn (the oracle's answers and feedback leave
    /// the served model as it is)
    #[arg(long, requires = "oracle")]
    pub no_oracle_learning: bool,
}

/// A value parser that never prints the value it refuses: clap's own
/// "invalid value '…'" would echo a key pasted into a numeric flag. The
/// error names the flag, the value's length and why.
#[derive(Clone)]
pub struct Quiet<T>(fn(&str) -> std::result::Result<T, String>);

impl<T: Clone + Send + Sync + 'static> clap::builder::TypedValueParser for Quiet<T> {
    type Value = T;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> std::result::Result<T, clap::Error> {
        let why = match value.to_str() {
            Some(s) => match (self.0)(s) {
                Ok(v) => return Ok(v),
                Err(e) => e,
            },
            None => "it is not UTF-8".to_string(),
        };
        let flag = arg.map_or_else(|| "a flag".to_string(), |a| format!("'{a}'"));
        Err(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            format!(
                "invalid value for {flag} ({} bytes, not shown): {why}\n",
                value.len()
            ),
        )
        .with_cmd(cmd))
    }
}

fn parse_max_price(s: &str) -> std::result::Result<(f64, f64), String> {
    oracle_setup::parse_max_price(s).map_err(|e| e.to_string())
}

/// The arguments of `args` (the program's name aside) that look like a key
/// ([`cortiq_decision::config::looks_like_key`]): a whole argument, or the
/// value of `--flag=VALUE`. Longest first.
pub fn key_like_args(args: &[std::ffi::OsString]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for a in args.iter().skip(1) {
        let a = a.to_string_lossy();
        let mut candidates = vec![a.to_string()];
        if a.starts_with('-')
            && let Some((_, v)) = a.split_once('=')
        {
            candidates.push(v.to_string());
        }
        for c in candidates {
            if !c.is_empty()
                && cortiq_decision::config::looks_like_key(&c).is_some()
                && !out.contains(&c)
            {
                out.push(c);
            }
        }
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.len()));
    out
}

/// `text` with every argument of `secrets` replaced by its length.
pub fn redact_args(text: &str, secrets: &[String]) -> String {
    let mut t = text.to_string();
    for s in secrets {
        t = t.replace(s.as_str(), &format!("({} bytes, not shown)", s.len()));
    }
    t
}

/// Report a command-line error of clap and exit. clap's own messages quote
/// what they refuse (`unexpected argument '…' found`, `unexpected value '…'
/// for '--flag' found`): a key pasted as a stray argument would land in the
/// terminal and in every captured log. When an argument looks like a key,
/// the message shows it only by its length (uncoloured); otherwise clap
/// reports as usual.
pub fn exit_on_clap_error(e: clap::Error) -> ! {
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let secrets = key_like_args(&args);
    if secrets.is_empty() || !e.use_stderr() {
        e.exit();
    }
    eprint!("{}", redact_args(&e.to_string(), &secrets));
    std::process::exit(e.exit_code());
}

/// `--oracle-budget USD`: a finite non-negative number (never echoed).
fn parse_budget(s: &str) -> std::result::Result<f64, String> {
    oracle_setup::refuse_key_in_number(s).map_err(|e| e.to_string())?;
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
        _ => Err("expected a non-negative number of USD, e.g. 0.5".into()),
    }
}

/// `--oracle-max-calls N`: a non-negative integer (never echoed).
fn parse_calls(s: &str) -> std::result::Result<u64, String> {
    oracle_setup::refuse_key_in_number(s).map_err(|e| e.to_string())?;
    s.trim()
        .parse::<u64>()
        .map_err(|_| "expected a non-negative integer, e.g. 100".into())
}

impl OracleArgs {
    /// The library's flags (`None` without `--oracle`).
    pub fn flags(&self) -> Option<OracleFlags> {
        Some(OracleFlags {
            model: self.oracle.clone()?,
            budget_usd: self.oracle_budget,
            max_calls: self.oracle_max_calls,
            key_env: self.oracle_key_env.clone(),
            base_url: self.oracle_base_url.clone(),
            max_price: self.oracle_max_price,
            no_learning: self.no_oracle_learning,
        })
    }

    /// The flags given, as spelled on the command line.
    pub fn given(&self) -> Vec<&'static str> {
        [
            ("--oracle", self.oracle.is_some()),
            ("--oracle-budget", self.oracle_budget.is_some()),
            ("--oracle-max-calls", self.oracle_max_calls.is_some()),
            ("--oracle-key-env", self.oracle_key_env.is_some()),
            ("--oracle-base-url", self.oracle_base_url.is_some()),
            ("--oracle-max-price", self.oracle_max_price.is_some()),
            ("--no-oracle-learning", self.no_oracle_learning),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect()
    }
}

/// The flags of `cortiq serve` that apply only to a decision file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServeFlags {
    /// `--decision-config PATH` (spec §4.13).
    pub decision_config: Option<PathBuf>,
    /// `--state DIR` (spec §4.14; default `<FILE>.state` or `state_dir`).
    pub state: Option<PathBuf>,
    /// `--break-lock`: remove a state `LOCK` left by a dead process where the
    /// filesystem has no advisory locks (elsewhere such a `LOCK` is taken over
    /// without it; spec §4.11).
    pub break_lock: bool,
    /// `--jev-compatible`: expose TypeSafe/Jev System One request handling at
    /// `POST /v1/systemone` (the response identifies the local CMF model).
    pub jev_compatible: bool,
    /// `--shadow-of URL`: shadow mode of the router API (spec §4.15).
    pub shadow_of: Option<String>,
    /// `--shadow-timeout-s N`: deadline of one request forwarded to the old
    /// router (default [`cortiq_decision::shadow::UPSTREAM_TIMEOUT`], 60 s).
    pub shadow_timeout_s: Option<u64>,
    /// `--oracle MODEL` and its companions.
    pub oracle: OracleArgs,
}

impl ServeFlags {
    /// The decision-only flags given, as spelled on the command line.
    pub fn given(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = [
            ("--decision-config", self.decision_config.is_some()),
            ("--state", self.state.is_some()),
            ("--break-lock", self.break_lock),
            ("--jev-compatible", self.jev_compatible),
            ("--shadow-of", self.shadow_of.is_some()),
            ("--shadow-timeout-s", self.shadow_timeout_s.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect();
        v.extend(self.oracle.given());
        v
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
             --host, --port, --decision-config, --state, --break-lock, --jev-compatible, --shadow-of and --oracle*)",
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
    let (mut config, config_sets_provider) = match &flags.decision_config {
        Some(p) => {
            let cfg = Config::load(p)?;
            // Whether the file sets `oracle.provider` (and so its max price).
            let raw: Value = serde_json::from_slice(
                &std::fs::read(p).with_context(|| format!("read {}", p.display()))?,
            )
            .with_context(|| format!("{}", p.display()))?;
            (cfg, raw.pointer("/oracle/provider").is_some())
        }
        None => (Config::default(), false),
    };
    // `--oracle MODEL`: applied before anything is opened; a model the oracle
    // cannot use stops here (one public GET of its endpoint listing, no key).
    let setup = match flags.oracle.flags() {
        Some(f) => {
            let (cfg, setup) = tokio::task::spawn_blocking(move || {
                let mut cfg = config;
                oracle_setup::apply(&mut cfg, &f, config_sets_provider, &oracle::process_env())
                    .map(|s| (cfg, s))
            })
            .await
            .context("oracle setup")??;
            config = cfg;
            for w in &setup.warnings {
                tracing::warn!("{w}");
            }
            Some(setup)
        }
        None => None,
    };
    let mut opts = ServeOptions::new(model, config);
    opts.oracle_from_flag = setup.is_some();
    opts.oracle_note = setup.as_ref().map(OracleSetup::price_note);
    opts.addr = socket_addr(host, port)?;
    opts.state_dir = flags.state.clone();
    opts.break_lock = flags.break_lock;
    opts.jev_compatible = flags.jev_compatible;
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
    if opts.jev_compatible {
        println!(
            "  Jev-compatible API: POST http://{}/v1/systemone (responds as {})",
            opts.addr, SYSTEMONE_MODEL_ID
        );
    }
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
    /// CURRENT names (base + overlay). With --oracle also where the oracle's
    /// ledger and answer cache are kept (created when missing; default
    /// <FILE>.state)
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
    #[command(flatten)]
    pub oracle: DecideOracleArgs,
}

/// `--oracle MODEL` and its companions of `cortiq decide`: the oracle in two
/// steps — the key in `OPENROUTER_API_KEY`, then this flag.
#[derive(Args, Clone, Debug, Default, PartialEq)]
pub struct DecideOracleArgs {
    /// Ask this OpenRouter model what the local model cannot decide (the gate
    /// rejects it, or no skill has the labels), e.g.
    /// deepseek/deepseek-v4.1-flash. A text the gate accepts never reaches it.
    /// The key is read from OPENROUTER_API_KEY (--oracle-key-env), never
    /// printed; the text sent is PII-redacted. The oracle's ledger and answer
    /// cache are kept in the state directory (--state DIR, else <FILE>.state)
    #[arg(long, value_name = "MODEL")]
    pub oracle: Option<String>,
    /// With --oracle: the most this run spends on the oracle, USD; every call
    /// is reserved against it before it is sent [default: 1.0]
    #[arg(long, value_name = "USD", requires = "oracle", value_parser = Quiet(parse_budget))]
    pub oracle_budget: Option<f64>,
    /// With --oracle: the most oracle calls this run makes [default: 10000]
    #[arg(long, value_name = "N", requires = "oracle", value_parser = Quiet(parse_calls))]
    pub oracle_max_calls: Option<u64>,
    /// With --oracle: the environment variable holding the OpenRouter key
    /// [default: OPENROUTER_API_KEY]
    #[arg(long, value_name = "VAR", requires = "oracle")]
    pub oracle_key_env: Option<String>,
    /// With --oracle: the OpenRouter API base (https; plain http only to a
    /// loopback address) [default: https://openrouter.ai/api/v1]
    #[arg(long, value_name = "URL", requires = "oracle")]
    pub oracle_base_url: Option<String>,
    /// With --oracle: max price in USD per 1M prompt and completion tokens,
    /// e.g. 0.1,0.5 [default: twice the model's cheapest structured-output
    /// endpoint]
    #[arg(long, value_name = "IN,OUT", requires = "oracle", value_parser = Quiet(parse_max_price))]
    pub oracle_max_price: Option<(f64, f64)>,
    /// With --oracle: switch the oracle of the state directory on again after
    /// the fix of what stopped it — a stop rule (a refused key, no credit,
    /// another model, a cost above the reservation, failures in a row) keeps
    /// it off for every later run until then. As POST /v1/admin/oracle
    /// {"enabled":true} on a server of that directory
    #[arg(long, requires = "oracle")]
    pub oracle_resume: bool,
    /// With --oracle, where the state directory's filesystem has no advisory
    /// locks: remove its LOCK left by a process that is no longer running (an
    /// interrupted run). Elsewhere such a LOCK is taken over without it; the
    /// LOCK of a running process is never removed
    #[arg(long, requires = "oracle")]
    pub break_lock: bool,
}

impl DecideOracleArgs {
    /// The library's flags (`None` without `--oracle`); `decide` never
    /// teaches the model.
    pub fn flags(&self) -> Option<OracleFlags> {
        Some(OracleFlags {
            model: self.oracle.clone()?,
            budget_usd: self.oracle_budget,
            max_calls: self.oracle_max_calls,
            key_env: self.oracle_key_env.clone(),
            base_url: self.oracle_base_url.clone(),
            max_price: self.oracle_max_price,
            no_learning: true,
        })
    }
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
    let oracle = a.oracle.flags();
    if let Some(f) = &oracle {
        // Every oracle flag is checked before any work; the network is used
        // only when a question needs the oracle.
        oracle_setup::check_flags(f)?;
    }
    // With --oracle, --state also names where the oracle's ledger is kept
    // and may not exist yet: until it does, the base file decides.
    let read_state = match a.state.as_deref() {
        Some(d) if oracle.is_some() && !d.exists() => None,
        other => other,
    };
    let model = open_model(&a.model, read_state, Verify::Light)?;
    if let Some(f) = oracle.as_ref().filter(|_| a.oracle.oracle_resume) {
        let root = a
            .state
            .clone()
            .unwrap_or_else(|| StateDir::default_for(&a.model));
        resume_oracle(&root, f, a.oracle.break_lock)?;
    }
    match (&a.input, oracle) {
        (Some(input), None) => decide_batch(&model, a, input),
        (Some(input), Some(f)) => decide_batch_oracle(model, a, input, &f),
        (None, oracle) => decide_one(model, a, oracle.as_ref()),
    }
}

/// `--oracle-resume`: the oracle of the state directory switched on again
/// (its stop cleared) under the directory's `LOCK`, before the run.
fn resume_oracle(root: &Path, flags: &OracleFlags, break_lock: bool) -> Result<()> {
    let file = root.join(cortiq_decision::statedir::ORACLE_STATE_FILE);
    if !file.exists() {
        eprintln!(
            "oracle: nothing to resume — {} has no oracle.state (the oracle was never stopped there)",
            root.display()
        );
        return Ok(());
    }
    let dir = StateDir::open(root)?;
    let _lock = take_lock(&dir, break_lock)?;
    let key_env = flags.key_env.as_deref().unwrap_or(DEFAULT_ORACLE_KEY_ENV);
    let path = dir.oracle_state_path();
    let limits = || {
        oracle::read_state_file(&path)
            .ok()
            .and_then(|st| admin_limits_text(&st))
            .map(|t| format!("; {t}"))
            .unwrap_or_default()
    };
    match oracle::resume_state_file(&path)? {
        Some(before) if before.is_off() => eprintln!(
            "oracle: resumed — the oracle of state directory {} was {}; it may be called again{}",
            root.display(),
            off_text(&before, key_env, &flags.model),
            limits()
        ),
        Some(before) => eprintln!(
            "oracle: the oracle of state directory {} was not stopped; cleared its {} failed call{} in a row{} (they count toward oracle.max_errors){}",
            root.display(),
            before.consecutive_errors,
            if before.consecutive_errors == 1 {
                ""
            } else {
                "s"
            },
            before
                .last_error
                .as_deref()
                .map(|e| format!(
                    " ({})",
                    oracle_setup::explain_last_error(e, key_env, &flags.model)
                ))
                .unwrap_or_default(),
            limits()
        ),
        None => eprintln!(
            "oracle: nothing to resume — the oracle of state directory {} is not stopped{}",
            root.display(),
            limits()
        ),
    }
    Ok(())
}

/// The admin limits of `oracle.state` in words (`None` without any): they
/// count the whole ledger, earlier runs included, and only a server's admin
/// API changes them.
fn admin_limits_text(st: &OracleState) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(b) = st.budget_usd {
        parts.push(format!("budget {}", usd(b)));
    }
    if let Some(c) = st.max_calls {
        parts.push(format!("{c} call{}", if c == 1 { "" } else { "s" }));
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!(
        "admin limits kept in its oracle.state: {} over the whole ledger (set by a server's POST /v1/admin/oracle; {} there lifts them)",
        parts.join(", "),
        ADMIN_LIFT
    ))
}

/// The admin request that lifts both admin limits of `oracle.state`.
const ADMIN_LIFT: &str = r#"{"budget_usd":null,"max_calls":null}"#;

/// Why `oracle.state` keeps the oracle off, in words.
fn off_text(st: &OracleState, key_env: &str, model: &str) -> String {
    match &st.stop_reason {
        Some(r) => format!(
            "stopped by the stop rule {r}: {}",
            oracle_setup::explain_stop(r, st.last_error.as_deref(), key_env, model)
        ),
        None => "switched off by a server's admin API".to_string(),
    }
}

/// The state directory's `LOCK` for `cortiq decide --oracle`: one process per
/// directory keeps the oracle's budget exact. A `LOCK` whose process is gone
/// (an interrupted or killed run, a crashed server) is taken over with a
/// warning; the lock of a running process is never broken. Where the
/// filesystem has no advisory locks the file alone says the directory is
/// taken, and `--break-lock` removes it — only the lock it found, so two runs
/// breaking the same stale lock never both proceed.
fn take_lock(dir: &StateDir, break_lock: bool) -> Result<StateLock> {
    dir.lock(break_lock)
        .map_err(|e| match e.downcast_ref::<Locked>() {
            Some(l) if l.held => anyhow::anyhow!(
                "state directory {} is held by pid {} (a running `cortiq serve`, or another \
                 `cortiq decide --oracle`): one process per state directory keeps the oracle's \
                 budget exact. Ask that server (POST /v1/decisions), wait for that run, or give \
                 this run a directory of its own with --state DIR",
                l.dir,
                l.pid
            ),
            Some(l) => anyhow::anyhow!(
                "state directory {} has a LOCK of pid {} ({}), and its filesystem has no advisory \
                 locks, so a LOCK left by an interrupted run stays there: if that process is \
                 gone, pass --break-lock to remove it; else wait for it, or give this run a \
                 directory of its own with --state DIR",
                l.dir,
                l.pid,
                l.lock
            ),
            None => e,
        })
}

/// Whether `sig` is ignored (`SIG_IGN`) in this process: inherited from
/// `nohup` (SIGHUP) or from a non-interactive shell's background job
/// (SIGINT). Such a signal must stay ignored.
#[cfg(unix)]
fn signal_ignored(sig: libc::c_int) -> bool {
    // SAFETY: with a null new action, sigaction(2) only reads the current
    // disposition into `old`, a zeroed plain-data struct.
    unsafe {
        let mut old: libc::sigaction = std::mem::zeroed();
        libc::sigaction(sig, std::ptr::null(), &mut old) == 0 && old.sa_sigaction == libc::SIG_IGN
    }
}

/// On SIGINT, SIGTERM or SIGHUP while `cortiq decide --oracle` holds the
/// state directory's `LOCK`: remove it and exit (130, 143, 129), so the next
/// run finds the directory free. A signal the process inherited as ignored
/// (`nohup`, a background job of a script) stays ignored and is not listened
/// to: the run goes on, as without this handler. The ledger needs nothing: a
/// call in flight has its reservation line, charged in full when the ledger
/// is next opened.
fn release_lock_on_signal(lock: &StateLock) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{Signal, SignalKind, signal};
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let _in_runtime = rt.enter();
        // Checked before any registration: tokio's handler replaces the
        // disposition for the rest of the process.
        let listen = |kind: SignalKind, sig: libc::c_int| {
            if signal_ignored(sig) {
                None
            } else {
                signal(kind).ok()
            }
        };
        let mut int = listen(SignalKind::interrupt(), libc::SIGINT);
        let mut term = listen(SignalKind::terminate(), libc::SIGTERM);
        let mut hup = listen(SignalKind::hangup(), libc::SIGHUP);
        if int.is_none() && term.is_none() && hup.is_none() {
            return;
        }
        /// Resolves when `s` delivers; never for a signal not listened to.
        async fn arrived(s: &mut Option<Signal>) {
            if let Some(s) = s
                && s.recv().await.is_some()
            {
                return;
            }
            std::future::pending::<()>().await
        }
        let release = lock.release_handle();
        rt.spawn(async move {
            let (name, code) = tokio::select! {
                () = arrived(&mut int) => ("SIGINT", 130),
                () = arrived(&mut term) => ("SIGTERM", 143),
                () = arrived(&mut hup) => ("SIGHUP", 129),
            };
            if release.release() {
                eprintln!(
                    "\ninterrupted ({name}): the state directory's LOCK is released; a call in flight is charged its full reservation"
                );
            }
            std::process::exit(code);
        });
    }
    #[cfg(not(unix))]
    let _ = lock;
}

fn decide_batch(model: &DecisionModel, a: &DecideArgs, input: &Path) -> Result<()> {
    let skill = eval::select_skill(model, a.skill.as_deref())?;
    let ev = Evaluator::new(model, &skill)?;
    let inputs = eval::read_input(input)?;
    let summary = run_batch(&ev, a, &inputs, |r| Ok(r.to_json()))?;
    eprintln!("{}", summary.render());
    eprintln!("{}", json!({ "summary": summary.to_json() }));
    Ok(())
}

/// The rows of a batch run, on stdout or in `--out` (removed when the run
/// fails: no partial output is left behind); `row` gives each row's JSON.
fn run_batch(
    ev: &Evaluator,
    a: &DecideArgs,
    inputs: &[EvalInput],
    mut row: impl FnMut(&RowResult) -> Result<Value>,
) -> Result<EvalSummary> {
    let opts = EvalOptions {
        bench: a.bench,
        warmup: if a.bench { eval::BENCH_WARMUP } else { 0 },
    };
    match &a.out {
        Some(path) => {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("create {}", path.display()))?;
            let mut w = BufWriter::new(file);
            let run = ev
                .run(inputs, opts, |r| write_row(&mut w, &row(r)?))
                .and_then(|s| {
                    w.flush()?;
                    w.get_ref().sync_all()?;
                    Ok(s)
                });
            if run.is_err() {
                // No partial output is left behind.
                let _ = std::fs::remove_file(path);
            }
            run
        }
        None => {
            let mut w = BufWriter::new(std::io::stdout().lock());
            let s = ev.run(inputs, opts, |r| write_row(&mut w, &row(r)?))?;
            w.flush()?;
            Ok(s)
        }
    }
}

/// `cortiq decide --input … --oracle MODEL`: every row is decided locally
/// first (the row of a run without `--oracle`, byte for byte, plus `action`,
/// `source`, `answer`, `oracle_cost_usd`, `flags` and, for a labelled row,
/// `answer_correct`); a row the gate rejects is asked again through the
/// oracle's service. The oracle is set up before the first row: a model it
/// cannot use, or a state directory another process holds, stops the run
/// before any output.
fn decide_batch_oracle(
    model: DecisionModel,
    a: &DecideArgs,
    input: &Path,
    flags: &OracleFlags,
) -> Result<()> {
    let handle = Arc::new(ModelHandle::new(LoadedModel::new(model)?));
    let current = handle.current();
    let skill = eval::select_skill(current.model(), a.skill.as_deref())?;
    let ev = Evaluator::new(current.model(), &skill)?;
    let inputs = eval::read_input(input)?;
    let run = OracleRun::open(
        &handle,
        &a.model,
        a.state.as_deref(),
        flags,
        a.oracle.break_lock,
    )?;
    eprintln!("{}", run.start_line());
    let principal = OracleRun::principal();
    let mut tally = BatchTally::default();
    let summary = run_batch(&ev, a, &inputs, |r| {
        let input = &inputs[r.i];
        let row = if r.accepted {
            RowOracle::local(r)
        } else {
            let body = single_request(&current, a, &input.text)?;
            let d = run
                .svc
                .decide_body(&serde_json::to_vec(&body)?, &principal)
                .map_err(api_error)?;
            RowOracle::of(&d)
        };
        tally.add(&row, input.label.as_deref());
        let mut v = r.to_json();
        row.write(&mut v, input.label.as_deref());
        Ok(v)
    })?;
    let hint = run.hint(&tally.reasons());
    eprintln!("{}", summary.render());
    eprintln!("{}", tally.render(&run, hint.as_deref()));
    let mut sj = summary.to_json();
    sj["oracle"] = tally.to_json(&run, hint.as_deref());
    eprintln!("{}", json!({ "summary": sj }));
    Ok(())
}

/// The cascade's part of one batch row.
struct RowOracle {
    action: Action,
    /// The final answer: the oracle's (or its cache's), else the local choice.
    answer: Option<String>,
    /// `usage.cost` of this row's oracle call (0 without one).
    cost: f64,
    flags: Vec<String>,
}

impl RowOracle {
    fn local(r: &RowResult) -> Self {
        Self {
            action: Action::Local,
            answer: r.choice.clone(),
            cost: 0.0,
            flags: Vec::new(),
        }
    }

    fn of(d: &Decided) -> Self {
        let o = &d.questions[0];
        let answer = match o.action {
            Action::Oracle | Action::Cache => o.oracle.as_ref().and_then(|a| a.label()),
            Action::Local | Action::Abstain => o.local.as_ref().and_then(|l| l.choice.as_deref()),
        };
        Self {
            action: o.action,
            answer: answer.map(str::to_string),
            cost: d.metered.oracle.cost.to_f64(),
            flags: o.flags.clone(),
        }
    }

    fn write(&self, v: &mut Value, label: Option<&str>) {
        v["action"] = json!(self.action.as_str());
        v["source"] = json!(self.action.source());
        v["answer"] = json!(self.answer);
        v["oracle_cost_usd"] = json!(self.cost);
        v["flags"] = json!(self.flags);
        if let Some(l) = label {
            v["answer_correct"] = json!(self.answer.as_deref() == Some(l));
        }
    }

    /// Why an abstained row was not answered (its most specific flag).
    fn reason(&self) -> &str {
        for key in ["no_key", "bad_key"] {
            if self.flags.iter().any(|f| f == key) {
                return key;
            }
        }
        self.flags
            .iter()
            .map(String::as_str)
            .find(|f| *f != cortiq_decision::pii::FLAG_PII_REDACTED)
            .unwrap_or("abstain")
    }
}

/// The oracle's totals of a batch run.
#[derive(Default)]
struct BatchTally {
    rows: usize,
    /// Rows the gate rejected (asked again through the oracle).
    rejected: usize,
    oracle: usize,
    cache: usize,
    abstained: usize,
    /// Abstained rows by reason (`budget`, `no_key`, `bad_key`, `stopped`,
    /// `oracle_unavailable`, …).
    reasons: BTreeMap<String, usize>,
    /// Σ `usage.cost` of the answered calls.
    cost: f64,
    labelled: usize,
    correct: usize,
}

impl BatchTally {
    fn add(&mut self, row: &RowOracle, label: Option<&str>) {
        self.rows += 1;
        match row.action {
            Action::Local => {}
            Action::Oracle => self.oracle += 1,
            Action::Cache => self.cache += 1,
            Action::Abstain => {
                self.abstained += 1;
                *self.reasons.entry(row.reason().to_string()).or_default() += 1;
            }
        }
        if row.action != Action::Local {
            self.rejected += 1;
        }
        self.cost += row.cost;
        if let Some(l) = label {
            self.labelled += 1;
            self.correct += usize::from(row.answer.as_deref() == Some(l));
        }
    }

    fn reasons(&self) -> Vec<String> {
        self.reasons.keys().cloned().collect()
    }

    fn to_json(&self, run: &OracleRun, hint: Option<&str>) -> Value {
        let mut v = run.json();
        v["rows_rejected"] = json!(self.rejected);
        v["answered_by_oracle"] = json!(self.oracle);
        v["answered_from_cache"] = json!(self.cache);
        v["abstained"] = json!(self.abstained);
        v["abstained_by"] = json!(self.reasons);
        v["cost_usd"] = json!(self.cost);
        v["answers_labelled"] = json!(self.labelled);
        v["answers_correct"] = json!(self.correct);
        v["answer_accuracy"] = if self.labelled == 0 {
            Value::Null
        } else {
            json!(self.correct as f64 / self.labelled as f64)
        };
        v["hint"] = json!(hint);
        v
    }

    fn render(&self, run: &OracleRun, hint: Option<&str>) -> String {
        let reasons: Vec<String> = self
            .reasons
            .iter()
            .map(|(r, n)| format!("{r} {n}"))
            .collect();
        let mut s = format!(
            "oracle {} via {}: the gate rejected {} of {} rows: {} answered by the oracle, {} from its cache, {} abstained{}; {}",
            run.model,
            host_of(&run.base_url),
            self.rejected,
            self.rows,
            self.oracle,
            self.cache,
            self.abstained,
            if reasons.is_empty() {
                String::new()
            } else {
                format!(" ({})", reasons.join(", "))
            },
            run.spend_text()
        );
        if self.labelled > 0 {
            s.push_str(&format!(
                "\nanswers (local + oracle): correct {}/{} = {:.2}%",
                self.correct,
                self.labelled,
                100.0 * self.correct as f64 / self.labelled as f64
            ));
        }
        if let Some(h) = hint {
            s.push_str(&format!("\nhint: {h}"));
        }
        s
    }
}

/// The oracle of `decide --oracle` without a usable key: an undetermined
/// question the state directory's cache holds is answered from it (0.8.11,
/// as a server answers it from its cache when its key is missing), every
/// other one is refused with `no_key` (the variable is unset) or `bad_key`
/// (it holds something that is not a key), as a server refuses it — no
/// network; the state directory is only read, never created or locked.
struct KeylessOracle {
    /// `NoKey` or `BadKey`.
    reason: RefusalReason,
    status: OracleStatus,
    /// The state directory's cache when it holds anything.
    cache: Option<CacheView>,
}

impl Escalator for KeylessOracle {
    fn escalate(&self, e: &Escalation<'_>) -> EscalationResult {
        if let Some(c) = &self.cache {
            return c.resolve(e, self.reason);
        }
        EscalationResult {
            resolved: e
                .pending
                .iter()
                .map(|_| Resolved::new(Resolution::Refused(self.reason)))
                .collect(),
            usage: Default::default(),
        }
    }

    fn feedback(&self, _: &FeedbackRequest, _: &Principal) -> Result<Value, ApiError> {
        Err(ApiError::not_found("`cortiq decide` keeps no feedback"))
    }

    fn admin(&self, _: &AdminCommand) -> Result<Value, ApiError> {
        Err(ApiError::not_found("`cortiq decide` has no admin API"))
    }

    fn oracle_status(&self) -> Option<OracleStatus> {
        Some(self.status.clone())
    }
}

/// The limit that refuses an oracle call for the budget.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Limit {
    /// `--oracle-max-calls` of this run.
    RunCalls,
    /// `--oracle-budget` of this run.
    RunBudget,
    /// `max_calls` of `oracle.state` (a server's admin API), over the ledger.
    AdminCalls(u64),
    /// `budget_usd` of `oracle.state`, over the ledger.
    AdminBudget(f64),
}

/// "1 call", "N calls".
fn calls_text(n: u64) -> String {
    if n == 1 {
        "1 call".to_string()
    } else {
        format!("{n} calls")
    }
}

/// The account `cortiq decide --oracle` records its calls under in the
/// reservation ledger.
pub const DECIDE_ACCOUNT: &str = "cortiq-decide";

/// Money and calls of one run and of its ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RunTotals {
    /// Charged in this run (USD) …
    spent: f64,
    /// … of which the full reservations of failed calls that reported no
    /// cost (OpenRouter may not have billed them, as for a refused key) …
    unknown_cost: f64,
    /// … of which those of calls OpenRouter refused (401, 402, 403, 429):
    /// likely not billed.
    refused_cost: f64,
    calls: u64,
    /// Everything the ledger holds (earlier runs included).
    ledger_spent: f64,
    ledger_calls: u64,
}

/// The oracle of one `cortiq decide --oracle` run: the server's cascade and
/// client over the state directory (`--state DIR`, else `<FILE>.state`,
/// under its `LOCK` for the run), with the budget and call limit of this run
/// on top of what its ledger already holds; learning is off. The stop rules
/// hold as on a server: a stop is written to `oracle.state` and keeps the
/// oracle off for later runs until `--oracle-resume`. Without a usable key,
/// [`KeylessOracle`] (no network; the state directory's cache read only).
struct OracleRun {
    svc: DecisionService,
    cascade: Option<Arc<Cascade>>,
    model: String,
    key_env: String,
    base_url: String,
    /// What this run may spend, USD.
    budget: f64,
    /// Calls this run may make.
    max_calls: u64,
    /// The ledger when the run started.
    before: LedgerTotals,
    ledger: Option<PathBuf>,
    /// Where the max price came from.
    price_note: Option<String>,
    /// The state directory (`None` without the key).
    state_root: Option<PathBuf>,
    /// `oracle.state` at the start when it kept the oracle off (a stop of an
    /// earlier run or a server, or a server's admin switch): it holds for the
    /// run.
    inherited: Option<OracleState>,
    /// Released last, after the cascade has written its last line.
    _lock: Option<StateLock>,
}

impl OracleRun {
    fn open(
        handle: &Arc<ModelHandle>,
        model_path: &Path,
        state: Option<&Path>,
        flags: &OracleFlags,
        break_lock: bool,
    ) -> Result<Self> {
        let mut cfg = Config::default();
        // `decide` never teaches the model: the oracle's answers are cached,
        // not learned.
        cfg.learning.enabled = false;
        let key_env = flags
            .key_env
            .clone()
            .unwrap_or_else(|| DEFAULT_ORACLE_KEY_ENV.to_string());
        let keyless = match oracle::read_key(&oracle::process_env(), &key_env).0 {
            oracle::KeyState::Missing => Some((RefusalReason::NoKey, OracleStatus::NoKey)),
            oracle::KeyState::Bad(p) => Some((RefusalReason::BadKey, OracleStatus::BadKey(p))),
            oracle::KeyState::Usable { trimmed } => {
                if trimmed > 0 {
                    eprintln!("warning: {}", oracle::trimmed_warning(&key_env, trimmed));
                }
                None
            }
        };
        if let Some((reason, status)) = keyless {
            oracle_setup::prepare(&mut cfg, flags, false)?;
            // What the oracle answered earlier runs on this state directory
            // is answered all the same (no call; read only, so no LOCK and
            // nothing written; a directory that is not there stays so).
            let learn_log = state
                .map(Path::to_path_buf)
                .unwrap_or_else(|| StateDir::default_for(model_path))
                .join(cortiq_decision::statedir::LEARN_LOG_FILE);
            let cache = if learn_log.is_file() {
                match CacheView::load(&cfg.cache, &learn_log) {
                    Ok(c) => (!c.is_empty()).then_some(c),
                    Err(e) => {
                        eprintln!("warning: the oracle's cached answers are not used: {e:#}");
                        None
                    }
                }
            } else {
                None
            };
            let esc: Arc<dyn Escalator> = Arc::new(KeylessOracle {
                reason,
                status,
                cache,
            });
            return Ok(Self {
                svc: DecisionService::open(Arc::clone(handle), cfg.clone(), Some(esc))?
                    .without_hint_log(),
                cascade: None,
                model: flags.model.clone(),
                key_env,
                base_url: cfg.oracle.base_url.clone(),
                budget: cfg.oracle.budget_usd,
                max_calls: cfg.oracle.max_calls,
                before: LedgerTotals::default(),
                ledger: None,
                price_note: None,
                state_root: None,
                inherited: None,
                _lock: None,
            });
        }
        // One public GET of the model's endpoint listing (no key): the max
        // price, or the refusal of a model the oracle cannot use.
        let setup = oracle_setup::apply(&mut cfg, flags, false, &oracle::process_env())?;
        for w in &setup.warnings {
            eprintln!("warning: {w}");
        }
        let root = state
            .map(Path::to_path_buf)
            .unwrap_or_else(|| StateDir::default_for(model_path));
        let dir = StateDir::open(&root).with_context(|| {
            format!(
                "state directory {} (where `cortiq decide --oracle` keeps the oracle's ledger; \
                 --state DIR puts it elsewhere)",
                root.display()
            )
        })?;
        let lock = take_lock(&dir, break_lock)?;
        release_lock_on_signal(&lock);
        let ledger = dir.oracle_ledger_path();
        let before = oracle::ledger_totals(&ledger)?;
        let (budget, max_calls) = (cfg.oracle.budget_usd, cfg.oracle.max_calls);
        // The run's budget and calls come on top of what the ledger holds.
        cfg.oracle.budget_usd = before.spent + budget;
        cfg.oracle.max_calls = before.calls.saturating_add(max_calls);
        // The server's cascade and client, `oracle.state` included: a stop of
        // this run is written there and holds for later runs, as on a server.
        let cascade = Cascade::open(Arc::clone(handle), &cfg, dir)?;
        let st = cascade.oracle().state();
        let inherited = st.is_off().then_some(st);
        let esc: Arc<dyn Escalator> = cascade.clone();
        Ok(Self {
            svc: DecisionService::open(Arc::clone(handle), cfg.clone(), Some(esc))?
                .without_hint_log(),
            cascade: Some(cascade),
            model: flags.model.clone(),
            key_env,
            base_url: cfg.oracle.base_url.clone(),
            budget,
            max_calls,
            before,
            ledger: Some(ledger),
            price_note: Some(setup.price_note()),
            state_root: Some(root),
            inherited,
            _lock: Some(lock),
        })
    }

    fn explain(&self, code: &str) -> String {
        oracle_setup::explain_oracle_error(code, &self.key_env, &self.model)
    }

    /// "fix …, then " before `run again with --oracle-resume`.
    fn fix_first(&self, reason: Option<&str>) -> String {
        match reason {
            Some("http_401" | "http_403") => format!(
                "fix the key (`{}` tests it), then ",
                self.check_command(false)
            ),
            Some("http_402") => "add credits, then ".to_string(),
            Some(_) => format!(
                "after the fix (`{}` shows what OpenRouter answers) ",
                self.check_command(true)
            ),
            None => String::new(),
        }
    }

    /// How the oracle of this run's state directory is switched on again.
    fn resume_text(&self, reason: Option<&str>) -> String {
        format!(
            "{}run again with --oracle-resume (a server of that directory: POST /v1/admin/oracle {{\"enabled\":true}})",
            self.fix_first(reason)
        )
    }

    /// What to do about a stop or switch-off found at the start.
    fn inherited_hint(&self) -> Option<String> {
        let st = self.inherited.as_ref()?;
        let root = self.state_root.as_ref()?;
        Some(format!(
            "the oracle of state directory {} is {} (recorded in its oracle.state by an earlier run or a server). It stays off until resumed: {}",
            root.display(),
            off_text(st, &self.key_env, &self.model),
            self.resume_text(st.stop_reason.as_deref())
        ))
    }

    /// The stop rule that fired in this run (it is in `oracle.state` now),
    /// in words.
    fn new_stop(&self) -> Option<(String, String)> {
        if self.inherited.is_some() {
            return None;
        }
        let st = self.cascade.as_ref()?.oracle().state();
        let r = st.stop_reason?;
        let words =
            oracle_setup::explain_stop(&r, st.last_error.as_deref(), &self.key_env, &self.model);
        Some((r, words))
    }

    /// The admin limits of this run's `oracle.state`, in words.
    fn admin_limits(&self) -> Option<String> {
        admin_limits_text(&self.cascade.as_ref()?.oracle().state())
    }

    /// Which limit refuses a call that reserves `next` USD (`None`: it fits).
    /// An admin limit is named first whenever it refuses the call, whatever
    /// the run's own limit says: `oracle.state` keeps it for every run, so no
    /// run flag lifts it ([`Self::budget_text`] adds the run's limit when it
    /// refuses the call as well).
    fn binding_limit(&self, next: f64) -> Option<Limit> {
        let o = self.cascade.as_ref()?.oracle();
        let (st, t, cfg) = (o.state(), o.totals(), o.config());
        // Admin limits count the whole ledger; the run's own sit on top of
        // what the ledger held at the start (`cfg` holds them so).
        if let Some(c) = st.max_calls.filter(|c| t.calls >= *c) {
            return Some(Limit::AdminCalls(c));
        }
        if t.calls >= cfg.max_calls {
            return Some(Limit::RunCalls);
        }
        let used = t.spent + t.inflight;
        if let Some(b) = st.budget_usd.filter(|b| used + next > *b) {
            return Some(Limit::AdminBudget(b));
        }
        (used + next > cfg.budget_usd).then_some(Limit::RunBudget)
    }

    /// Whether the run's call limit (on top of what the ledger held at the
    /// start) is reached.
    fn run_calls_out(&self) -> bool {
        self.cascade.as_ref().is_some_and(|c| {
            let o = c.oracle();
            o.totals().calls >= o.config().max_calls
        })
    }

    /// Whether the run's budget (on top of what the ledger held at the
    /// start) cannot hold a call reserving `next`.
    fn run_budget_short(&self, next: f64) -> bool {
        self.cascade.as_ref().is_some_and(|c| {
            let o = c.oracle();
            let t = o.totals();
            t.spent + t.inflight + next > o.config().budget_usd
        })
    }

    /// What refuses the oracle's calls for the budget, and what to do about
    /// it; `next` is the reservation that did not fit (or the smallest one).
    fn budget_text(&self, next: Option<f64>) -> String {
        let root = self
            .state_root
            .as_ref()
            .map_or("-".into(), |p| p.display().to_string());
        let t = self.totals();
        // Without a refused call the figure is the smallest possible call's:
        // a lower bound.
        let more = if next.is_none() {
            ", more for longer questions"
        } else {
            ""
        };
        let (next, reserves) = match next {
            Some(r) => (r, format!("the next call reserves {}", usd_fine(r))),
            None => {
                let least = self.least_reservation();
                (
                    least,
                    format!("every call reserves at least {}", usd_fine(least)),
                )
            }
        };
        let file = self.state_root.as_ref().map_or_else(
            || "oracle.state".to_string(),
            |p| {
                p.join(cortiq_decision::statedir::ORACLE_STATE_FILE)
                    .display()
                    .to_string()
            },
        );
        match self.binding_limit(next) {
            Some(Limit::AdminCalls(c)) => {
                let also = if self.run_calls_out() {
                    format!(
                        "; the {} this run may make {} made as well: pass a larger \
                         --oracle-max-calls too",
                        calls_text(self.max_calls),
                        if self.max_calls == 1 { "is" } else { "are" }
                    )
                } else {
                    String::new()
                };
                format!(
                    "the admin limit max_calls {c} in {file} binds: it counts every call of the \
                     ledger, which holds {}, and --oracle-max-calls cannot raise it. Lift it on a \
                     server of state directory {root}: POST /v1/admin/oracle {{\"max_calls\": \
                     null}}, or {{\"max_calls\": {}}} (at most that server's configured \
                     limit){also}",
                    t.ledger_calls,
                    t.ledger_calls.saturating_add(1)
                )
            }
            Some(Limit::AdminBudget(b)) => {
                let (need, run_need) = self.budget_needs(next);
                let also = if self.run_budget_short(next) {
                    format!(
                        "; the oracle budget of this run ({}) cannot hold it either: pass \
                         --oracle-budget of at least {} too",
                        usd(self.budget),
                        usd_ceil(run_need)
                    )
                } else {
                    String::new()
                };
                format!(
                    "the admin limit budget_usd {} in {file} binds: it counts all the ledger's \
                     spending, {} so far, and {reserves}, so it needs at least {}; \
                     --oracle-budget cannot raise it. Lift it on a server of state directory \
                     {root}: POST /v1/admin/oracle {{\"budget_usd\": null}}, or \
                     {{\"budget_usd\": {}}} (at most that server's configured budget){also}",
                    usd_fine(b),
                    usd(t.ledger_spent),
                    usd_fine(need),
                    &usd_ceil(need)[1..]
                )
            }
            Some(Limit::RunCalls) if self.max_calls == 0 && self.run_budget_short(next) => {
                format!(
                    "--oracle-max-calls 0 allows no oracle call, and the oracle budget of this run \
                     ({}) cannot hold one ({reserves} before it is sent): pass --oracle-max-calls \
                     of at least 1 and --oracle-budget of at least {}{more}",
                    usd(self.budget),
                    usd_ceil(next),
                )
            }
            Some(Limit::RunCalls) if self.max_calls == 0 => {
                "--oracle-max-calls 0 allows no oracle call: pass a larger --oracle-max-calls"
                    .to_string()
            }
            Some(Limit::RunCalls) => format!(
                "the {} this run may make {} made (--oracle-max-calls): pass a larger \
                 --oracle-max-calls",
                calls_text(self.max_calls),
                if self.max_calls == 1 { "is" } else { "are" }
            ),
            Some(Limit::RunBudget) | None if t.calls == 0 => format!(
                "the oracle budget of this run ({}) is too small to hold one call ({reserves} \
                 before it is sent): pass --oracle-budget of at least {}{more}",
                usd(self.budget),
                usd_ceil(next),
            ),
            Some(Limit::RunBudget) | None => format!(
                "the oracle budget of this run is used up ({} of {} spent, {} of {} calls; \
                 {reserves}, which does not fit): pass a larger --oracle-budget",
                usd(t.spent),
                usd(self.budget),
                t.calls,
                self.max_calls,
            ),
        }
    }

    /// The least budgets that admit a call reserving `next` USD: the admin
    /// limit's (over the whole ledger: its spending, the reservations in
    /// flight and `next`) and this run's (its own share of those).
    fn budget_needs(&self, next: f64) -> (f64, f64) {
        match &self.cascade {
            Some(c) => {
                let t = c.oracle().totals();
                let need = t.spent + t.inflight + next;
                (need, (need - self.before.spent).max(next))
            }
            None => (next, next),
        }
    }

    /// The smallest reservation of one call (USD).
    fn least_reservation(&self) -> f64 {
        self.cascade
            .as_ref()
            .map_or(0.0, |c| c.oracle().min_reservation_usd())
    }

    /// Who asks: the operator of this command line (the oracle allowed, no
    /// teaching).
    fn principal() -> Principal {
        Principal {
            account: DECIDE_ACCOUNT.into(),
            plan: "cli".into(),
            learning_allowed: false,
            ..Principal::open()
        }
    }

    /// The oracle's status for this run: the client's, except that a budget
    /// that refused the run's first call (nothing spent or reserved in this
    /// run, or in the ledger for an admin limit) is `budget_too_small`, with
    /// the reservation that did not fit — `budget_exhausted` is kept for a
    /// budget something was spent from.
    fn status(&self) -> OracleStatus {
        let s = self.svc.oracle_status();
        let Some(c) = &self.cascade else {
            return s;
        };
        let t = self.totals();
        if t.calls > 0 {
            return s;
        }
        let refused = c.oracle().last_budget_refusal();
        let next = match (&s, refused) {
            (OracleStatus::BudgetExhausted { .. } | OracleStatus::BudgetTooSmall { .. }, r) => {
                r.unwrap_or_else(|| self.least_reservation())
            }
            (OracleStatus::Ready, Some(r)) => r,
            _ => return s,
        };
        let spent_before = t.ledger_calls > 0 || t.ledger_spent > 0.0;
        // The call limit binds first; the budget may be short as well. An
        // admin limit the client found binding is kept with the status.
        let admin = s.admin_binding().cloned();
        let too_small = |calls_zero: bool| OracleStatus::BudgetTooSmall {
            min_usd: (!calls_zero || self.run_budget_short(next)).then_some(next),
            calls_zero,
            admin: admin.clone(),
        };
        match self.binding_limit(next) {
            Some(Limit::RunBudget) => too_small(false),
            Some(Limit::RunCalls) => too_small(true),
            Some(Limit::AdminBudget(_)) if !spent_before => too_small(false),
            Some(Limit::AdminCalls(_)) if !spent_before => too_small(true),
            Some(_) => OracleStatus::BudgetExhausted {
                admin: admin.clone(),
            },
            None => s,
        }
    }

    /// Why a missing or unusable key keeps the oracle off, in the command
    /// line's words, and what to do (`None`: the key is usable). Never a
    /// byte of the key.
    fn key_problem(&self) -> Option<String> {
        let k = &self.key_env;
        match self.svc.oracle_status() {
            OracleStatus::NoKey => Some(format!(
                "{k} is not set (decide --oracle reads the key from the environment): export \
                 {k}=<your OpenRouter key> (create one at {}) and run again; `{}` tests it",
                oracle_setup::KEYS_PAGE,
                self.check_command(false),
            )),
            OracleStatus::BadKey(p) => Some(format!(
                "{} (decide --oracle reads the key from the environment; nothing was sent with \
                 it): fix the variable and run again; `{}` tests it",
                oracle::bad_key_text(k, &p),
                self.check_command(false),
            )),
            _ => None,
        }
    }

    /// The error of `decide -p` whose labels no skill has when the oracle
    /// could not answer them, in the command line's words (the service's
    /// are a server's).
    fn untrained_error(&self, e: ApiError, labels: &[String]) -> anyhow::Error {
        let untrained = e
            .details
            .as_ref()
            .is_some_and(|d| d.contains_key("questions"));
        if !untrained {
            return api_error(e);
        }
        let why = self.key_problem().or_else(|| {
            let flag = match e.reason {
                protocol::Reason::OracleBudgetExhausted => "budget",
                protocol::Reason::OracleDisabled => "stopped",
                protocol::Reason::OracleUnavailable => {
                    cortiq_decision::service::FLAG_ORACLE_UNAVAILABLE
                }
                _ => return None,
            };
            self.hint(&[flag.to_string()])
        });
        let what = if labels.is_empty() {
            "no skill decides this question".to_string()
        } else {
            format!("no skill has the labels {}", labels.join(", "))
        };
        match why {
            Some(w) => anyhow::anyhow!("{what}, so only the oracle can answer, and {w}"),
            None => api_error(e),
        }
    }

    /// Money and calls of this run (the ledger's own counting) and of the
    /// whole ledger.
    fn totals(&self) -> RunTotals {
        match &self.cascade {
            Some(c) => {
                let t = c.oracle().totals();
                RunTotals {
                    spent: (t.spent - self.before.spent).max(0.0),
                    unknown_cost: (t.unknown_cost - self.before.unknown_cost).max(0.0),
                    refused_cost: (t.refused_cost - self.before.refused_cost).max(0.0),
                    calls: t.calls.saturating_sub(self.before.calls),
                    ledger_spent: t.spent,
                    ledger_calls: t.calls,
                }
            }
            None => RunTotals::default(),
        }
    }

    /// "$X spent in this run (N calls), budget $B; ledger PATH: $T over M
    /// calls in all" (a reservation charged for a failed call with no
    /// reported cost is named as such).
    fn spend_text(&self) -> String {
        let t = self.totals();
        let mut unknown = String::new();
        if t.refused_cost > 0.0 {
            unknown.push_str(&format!(
                "; {} of it is the reservation of calls OpenRouter refused (HTTP 401, 402, 403 or 429) without a cost, likely not billed",
                usd(t.refused_cost)
            ));
        }
        let other = t.unknown_cost - t.refused_cost;
        if other > 1e-12 {
            unknown.push_str(&format!(
                "; {} of it is the reservation of failed calls that reported no cost (a timeout, a lost connection, an interrupted run, a bad answer), which OpenRouter may have billed",
                usd(other)
            ));
        }
        let mut s = format!(
            "{} spent in this run ({} call{}{unknown}), budget {}",
            usd(t.spent),
            t.calls,
            if t.calls == 1 { "" } else { "s" },
            usd(self.budget)
        );
        if let Some(l) = &self.ledger {
            s.push_str(&format!(
                "; ledger {}: {} over {} call{} in all",
                l.display(),
                usd(t.ledger_spent),
                t.ledger_calls,
                if t.ledger_calls == 1 { "" } else { "s" }
            ));
        }
        s
    }

    fn max_price(&self) -> Option<(f64, f64)> {
        self.cascade.as_ref().map(|c| c.oracle().max_price())
    }

    /// `cortiq decision oracle check` with this run's settings.
    fn check_command(&self, test_call: bool) -> String {
        let mut s = "cortiq decision oracle check".to_string();
        if self.model != DEFAULT_ORACLE_MODEL {
            s.push_str(&format!(" --model {}", self.model));
        }
        if self.key_env != DEFAULT_ORACLE_KEY_ENV {
            s.push_str(&format!(" --key-env {}", self.key_env));
        }
        if self.base_url != DEFAULT_ORACLE_BASE_URL {
            s.push_str(&format!(" --base-url {}", self.base_url));
        }
        if test_call {
            s.push_str(" --test-call");
        }
        s
    }

    /// The first line of a batch run (stderr), never the key.
    fn start_line(&self) -> String {
        let what = format!("{} via {}", self.model, host_of(&self.base_url));
        match self.status() {
            OracleStatus::Ready => {
                let (p, c) = self.max_price().unwrap_or_default();
                format!(
                    "oracle: ready — {what}, budget {} for this run, max price in/out {}/{} per 1M ({}); ledger {}{}",
                    usd(self.budget),
                    usd(p),
                    usd(c),
                    self.price_note.as_deref().unwrap_or("-"),
                    self.ledger
                        .as_ref()
                        .map_or("-".into(), |l| l.display().to_string()),
                    self.admin_limits()
                        .map(|a| format!("; {a}"))
                        .unwrap_or_default()
                )
            }
            OracleStatus::NoKey | OracleStatus::BadKey(_) => format!(
                "oracle: NOT ready — {}. The rows the gate rejects abstain",
                self.key_problem().unwrap_or_default()
            ),
            OracleStatus::BudgetExhausted { .. } | OracleStatus::BudgetTooSmall { .. } => {
                format!("oracle: NOT ready — {} ({what})", self.budget_text(None))
            }
            other => match self.inherited_hint() {
                Some(h) => format!("oracle: NOT ready — {h}"),
                None => format!("oracle: NOT ready — {} ({what})", other.label()),
            },
        }
    }

    /// What to do about the refusals behind `flags` and a stop rule that
    /// fired in this run (never the key).
    fn hint(&self, flags: &[String]) -> Option<String> {
        let has = |f: &str| flags.iter().any(|x| x == f);
        if has("no_key") || has("bad_key") {
            return self.key_problem();
        }
        if has("stopped") || has("oracle_disabled") {
            if let Some(h) = self.inherited_hint() {
                return Some(h);
            }
        }
        // A stop of this run: also after an answered call (a cost above its
        // reservation keeps the answer).
        if let Some((r, words)) = self.new_stop() {
            return Some(format!(
                "a stop rule stopped the oracle in this run: {words} (stop rule {r}). It stays off for state directory {} until resumed: {}",
                self.state_root
                    .as_ref()
                    .map_or("-".into(), |p| p.display().to_string()),
                self.resume_text(Some(&r))
            ));
        }
        if has("budget") {
            let next = self
                .cascade
                .as_ref()
                .and_then(|c| c.oracle().last_budget_refusal());
            return Some(self.budget_text(next));
        }
        if has(cortiq_decision::service::FLAG_ORACLE_UNAVAILABLE) {
            let why = self
                .cascade
                .as_ref()
                .and_then(|c| c.oracle().last_error())
                .map_or_else(|| "no error code".to_string(), |c| self.explain(&c));
            return Some(format!(
                "the oracle call failed: {why}; `{}` tests the setup",
                self.check_command(true)
            ));
        }
        None
    }

    /// `cmf.oracle` of `--json` (never the key).
    fn json(&self) -> Value {
        let t = self.totals();
        let st = self.cascade.as_ref().map(|c| c.oracle().state());
        let status = self.status();
        json!({
            "model": self.model,
            "asked": true,
            "status": status.label(),
            "key_problem": match &status {
                OracleStatus::BadKey(p) => Some(p.as_str()),
                _ => None,
            },
            "min_call_usd": match &status {
                OracleStatus::BudgetTooSmall { min_usd, .. } => *min_usd,
                _ => None,
            },
            // Codes of the closed set only (oracle.state may be an older
            // version's, or edited by hand).
            "stop_reason": st.as_ref().and_then(|s| s.stop_reason.as_deref().map(oracle::shown_code)),
            "last_error": st.as_ref().and_then(|s| s.last_error.clone())
                .or_else(|| self.cascade.as_ref().and_then(|c| c.oracle().last_error()))
                .map(|e| oracle::shown_code(&e).to_string()),
            "key_env": self.key_env,
            "base_url": self.base_url,
            "budget_usd": self.budget,
            "spent_usd": t.spent,
            "unknown_cost_usd": t.unknown_cost,
            "refused_cost_usd": t.refused_cost,
            "admin_budget_usd": self.cascade.as_ref().and_then(|c| c.oracle().state().budget_usd),
            "admin_max_calls": self.cascade.as_ref().and_then(|c| c.oracle().state().max_calls),
            "calls": t.calls,
            "max_calls": self.max_calls,
            "max_price": self.max_price().map(|(p, c)| json!({"prompt": p, "completion": c})),
            "ledger": self.ledger.as_ref().map(|l| l.display().to_string()),
            "ledger_spent_usd": self.ledger.as_ref().map(|_| t.ledger_spent),
            "ledger_calls": self.ledger.as_ref().map(|_| t.ledger_calls),
        })
    }

    /// The `oracle:` line of the human output (`None` without the key).
    fn line(&self) -> Option<String> {
        self.ledger.as_ref()?;
        Some(format!(
            "{} via {}: {}",
            self.model,
            host_of(&self.base_url),
            self.spend_text()
        ))
    }
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
            || Value::String(server::DEFAULT_ROUTE_INSTRUCTIONS.to_string()),
            |r| r.instructions,
        );
        // `--labels` alone: the data skill they name (exact or subset, spec
        // §4.5), named with `cmf.skill` — the labels are that skill's own
        // question, which a subset must show (DESIGN C2.1). Labels no skill
        // has, ambiguous or superset ones go as they are: the service says
        // why, or the oracle answers.
        let forced = a.skill.clone().or_else(|| {
            let labels: Vec<&str> = a.labels.iter().map(String::as_str).collect();
            cortiq_decision::matching::skill_for_labels(&model.skill_labels(), &labels).ok()
        });
        (
            json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            forced,
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

fn decide_one(model: DecisionModel, a: &DecideArgs, oracle: Option<&OracleFlags>) -> Result<()> {
    let text = a.prompt.as_deref().expect("clap: -p or --input");
    let handle = Arc::new(ModelHandle::new(LoadedModel::new(model)?));
    let body = serde_json::to_vec(&single_request(&handle.current(), a, text)?)?;
    // No escalator: this service never calls the oracle.
    let local = DecisionService::open(Arc::clone(&handle), Config::default(), None)?;
    let Some(flags) = oracle else {
        let decided = local
            .decide_body(&body, &Principal::open())
            .map_err(api_error)?;
        return print_decided(&decided, a.json, None);
    };
    let req = protocol::parse_request(&body, &local.limits()).map_err(api_error)?;
    let undetermined = local
        .decide_local(&req)
        .map_err(api_error)?
        .questions
        .iter()
        .any(|(_, l)| l.as_ref().is_none_or(LocalDecision::undetermined));
    if !undetermined {
        // The gate accepted every question: the oracle is not asked (no
        // network, no state directory).
        let mut decided = local.decide(&req, &Principal::open()).map_err(api_error)?;
        decided.response["cmf"]["oracle"] = json!({"model": flags.model, "asked": false});
        let view = OracleView {
            model: flags.model.clone(),
            line: None,
            hint: None,
        };
        return print_decided(&decided, a.json, Some(&view));
    }
    let run = OracleRun::open(
        &handle,
        &a.model,
        a.state.as_deref(),
        flags,
        a.oracle.break_lock,
    )?;
    let mut decided = match run.svc.decide(&req, &OracleRun::principal()) {
        Ok(d) => d,
        Err(e) => return Err(run.untrained_error(e, &a.labels)),
    };
    let flags_seen: Vec<String> = decided
        .questions
        .iter()
        .flat_map(|q| q.flags.iter().cloned())
        .collect();
    // The command line's own hint (the service words its hint for a server).
    let hint = run.hint(&flags_seen);
    let cmf = &mut decided.response["cmf"];
    match &hint {
        Some(h) => cmf["hint"] = json!(h),
        None => {
            if let Some(m) = cmf.as_object_mut() {
                m.remove("hint");
            }
        }
    }
    cmf["oracle"] = run.json();
    let view = OracleView {
        model: flags.model.clone(),
        line: run.line(),
        hint,
    };
    print_decided(&decided, a.json, Some(&view))
}

fn print_decided(d: &Decided, as_json: bool, oracle: Option<&OracleView>) -> Result<()> {
    if as_json {
        println!("{}", d.response);
    } else {
        print!("{}", render_decided(d, oracle));
    }
    Ok(())
}

/// What the human output of `decide --oracle` says about the oracle.
struct OracleView {
    model: String,
    /// The run's spend, budget and ledger (`None`: not asked, or no key).
    line: Option<String>,
    hint: Option<String>,
}

fn num(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        other => other.to_string(),
    }
}

/// Human-readable lines of a decided `cortiq decide -p`.
fn render_decided(d: &Decided, oracle: Option<&OracleView>) -> String {
    let r = &d.response;
    let cost = d.metered.oracle.cost.to_f64();
    let mut s = String::new();
    for o in &d.questions {
        s.push_str(&render_question(o, cost, oracle));
    }
    if let Some(v) = oracle {
        if let Some(l) = &v.line {
            s.push_str(&format!("oracle:     {l}\n"));
        }
        if let Some(h) = &v.hint {
            s.push_str(&format!("hint:       {h}\n"));
        }
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

fn render_question(o: &QuestionOutcome, cost: f64, oracle: Option<&OracleView>) -> String {
    let flags = if o.flags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", o.flags.join(", "))
    };
    let answered = match (o.action, o.oracle.as_ref().and_then(|a| a.label()), oracle) {
        (Action::Oracle, Some(c), Some(v)) => {
            Some(format!("{c} (from oracle {}, {})", v.model, usd(cost)))
        }
        (Action::Cache, Some(c), Some(v)) => Some(format!(
            "{c} (from the cache of oracle {}: its answer to the same text, $0.00)",
            v.model
        )),
        _ => None,
    };
    let Some(l) = &o.local else {
        // No skill decides these labels: only the oracle can answer.
        return match answered {
            Some(c) => format!(
                "choice:     {c}\naction:     {} (no skill has these labels){flags}\n",
                o.action.as_str()
            ),
            None => format!("{}: {} ({})\n", o.id, o.action.as_str(), o.answer),
        };
    };
    let mut s = String::new();
    s.push_str(&format!(
        "choice:     {}\n",
        answered.unwrap_or_else(|| l.choice.as_deref().unwrap_or("-").to_string())
    ));
    let why = match (o.action, oracle) {
        (Action::Local, None) => "accepted by the gate".to_string(),
        (Action::Local, Some(_)) => "accepted by the gate; the oracle is not asked".to_string(),
        (Action::Oracle | Action::Cache, _) => format!(
            "the gate rejected the local choice {}{flags}",
            l.choice.as_deref().unwrap_or("-")
        ),
        (Action::Abstain, None) => format!(
            "the gate rejected it; without --oracle MODEL `cortiq decide` does not ask the oracle{flags}"
        ),
        (Action::Abstain, Some(_)) => {
            format!("the gate rejected it and the oracle did not answer{flags}")
        }
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
        /// Whether this key's undetermined questions may reach the oracle
        /// (default: true, as imported router keys; --oracle-allowed=false
        /// creates a key that never calls the oracle — its undetermined
        /// questions are still answered from the server's shared cache when
        /// it holds the same question, 0.8.11). The server's oracle switch,
        /// budgets and stop rules still apply
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
            value_parser = clap::value_parser!(bool)
        )]
        oracle_allowed: Option<bool>,
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
    /// The OpenRouter oracle: `oracle check` tells whether it is ready
    Oracle {
        #[command(subcommand)]
        cmd: OracleCmd,
    },
}

/// `cortiq decision oracle …`.
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum OracleCmd {
    /// Is the oracle ready? (a) the key is in its variable; (b) GET
    /// /auth/key accepts it (free; the key's credit limit and usage); (c) the
    /// model is listed with structured outputs (free, no key; its cheapest
    /// price); (d) with --test-call, one tiny structured call and its cost.
    /// Exit code 0 only when ready. The key is never printed
    Check {
        /// OpenRouter model id
        #[arg(long, value_name = "MODEL", default_value = DEFAULT_ORACLE_MODEL)]
        model: String,
        /// The environment variable holding the OpenRouter key
        #[arg(long, value_name = "VAR", default_value = DEFAULT_ORACLE_KEY_ENV)]
        key_env: String,
        /// OpenRouter API base (https; plain http only to a loopback address)
        #[arg(long, value_name = "URL", default_value = DEFAULT_ORACLE_BASE_URL)]
        base_url: String,
        /// Also make one tiny structured call (a two-option choice,
        /// max_tokens 16: a small fraction of a cent) and report its cost;
        /// made only when the other checks pass
        #[arg(long)]
        test_call: bool,
        /// Max price in USD per 1M prompt and completion tokens, e.g. 1,4:
        /// what --oracle-max-price would give --oracle. Needed for a model
        /// listed only with variable pricing (e.g. openrouter/auto); for
        /// another, some structured-output endpoint must fit it [default:
        /// twice the model's cheapest structured-output endpoint]
        #[arg(long, value_name = "IN,OUT", value_parser = Quiet(parse_max_price))]
        max_price: Option<(f64, f64)>,
        /// Print the report as one JSON line
        #[arg(long)]
        json: bool,
    },
}

/// `cortiq decision oracle check`: the report on stdout, exit code 0 only
/// when ready.
fn oracle_check(opts: &CheckOptions, as_json: bool) -> Result<()> {
    let report = oracle_setup::check(opts, &oracle::process_env())?;
    let trimmed = report.key.trimmed();
    if trimmed > 0 {
        eprintln!(
            "warning: {}",
            oracle::trimmed_warning(&opts.key_env, trimmed)
        );
    }
    if as_json {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.render());
    }
    let n = report.problems().len();
    ensure!(
        n == 0,
        "the oracle is not ready ({n} problem{})",
        if n == 1 { "" } else { "s" }
    );
    Ok(())
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
        DecisionCmd::Oracle {
            cmd:
                OracleCmd::Check {
                    model,
                    key_env,
                    base_url,
                    test_call,
                    max_price,
                    json,
                },
        } => oracle_check(
            &CheckOptions {
                model: model.clone(),
                key_env: key_env.clone(),
                base_url: base_url.clone(),
                test_call: *test_call,
                max_price: *max_price,
            },
            *json,
        ),
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
        "auto": m.is_auto(),
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
            "    {}{}: {} labels ({} active, {} inactive, {} quarantined), K {}, rows train {} / calibration {} / learned {}",
            m.id,
            if m.is_auto() { " (auto)" } else { "" },
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
                oracle_allowed: Some(oracle_allowed.unwrap_or(true)),
                learning_allowed: learning_allowed.then_some(true),
            };
            let created = store.create(&new, &cfg.auth.plans, now)?;
            if *json {
                println!("{}", created.to_json(now));
            } else {
                let r = &created.record;
                println!("{}", created.raw);
                eprintln!(
                    "created key {} for account {} (plan {}, {} requests/min, decision quota {}, expires {}, oracle {}); \
                     the key is shown only now, {} keeps its sha256",
                    r.hash12(),
                    keys_mod::shown(&r.account),
                    keys_mod::shown(&r.plan),
                    r.rate_per_min,
                    r.decision_quota,
                    r.expires.map_or("never".to_string(), |e| e.to_string()),
                    if r.oracle_allowed {
                        "allowed"
                    } else {
                        "not allowed"
                    },
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
            jev_compatible: true,
            shadow_of: Some("https://router.example.com".into()),
            shadow_timeout_s: Some(300),
            oracle: OracleArgs {
                oracle: Some("m/x".into()),
                oracle_budget: Some(2.0),
                no_oracle_learning: true,
                ..OracleArgs::default()
            },
        };
        assert_eq!(
            d.given(),
            [
                "--decision-config",
                "--state",
                "--break-lock",
                "--jev-compatible",
                "--shadow-of",
                "--shadow-timeout-s",
                "--oracle",
                "--oracle-budget",
                "--no-oracle-learning"
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
        let adapter = ServeFlags {
            jev_compatible: true,
            ..ServeFlags::default()
        };
        let e = check_serve_flags("m.cmf", false, &[], &adapter).unwrap_err();
        assert!(e.to_string().contains("--jev-compatible"), "{e}");
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
            "--jev-compatible",
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
                jev_compatible,
                shadow_of,
                shadow_timeout_s,
                task,
                ..
            } => {
                assert_eq!(decision_config.as_deref(), Some("c.json"));
                assert_eq!(state.as_deref(), Some("s"));
                assert!(break_lock);
                assert!(jev_compatible);
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
    fn serve_parses_the_oracle_flags_which_need_oracle() {
        match parse(&[
            "cortiq",
            "serve",
            "d.cmf",
            "--oracle",
            "deepseek/deepseek-v4.1-flash",
            "--oracle-budget",
            "5",
            "--oracle-max-calls",
            "100",
            "--oracle-key-env",
            "MY_KEY",
            "--oracle-base-url",
            "http://127.0.0.1:9/api/v1",
            "--oracle-max-price",
            "0.2,0.8",
            "--no-oracle-learning",
        ])
        .unwrap()
        .command
        {
            Commands::Serve { oracle, .. } => {
                let f = oracle.flags().unwrap();
                assert_eq!(f.model, "deepseek/deepseek-v4.1-flash");
                assert_eq!(f.budget_usd, Some(5.0));
                assert_eq!(f.max_calls, Some(100));
                assert_eq!(f.key_env.as_deref(), Some("MY_KEY"));
                assert_eq!(f.base_url.as_deref(), Some("http://127.0.0.1:9/api/v1"));
                assert_eq!(f.max_price, Some((0.2, 0.8)));
                assert!(f.no_learning);
            }
            _ => panic!("not serve"),
        }
        match parse(&["cortiq", "serve", "d.cmf"]).unwrap().command {
            Commands::Serve { oracle, .. } => assert_eq!(oracle.flags(), None),
            _ => panic!("not serve"),
        }
        // The companions need --oracle; a bad price pair is refused.
        for a in [
            &["--oracle-budget", "5"][..],
            &["--oracle-max-calls", "1"],
            &["--oracle-key-env", "K"],
            &["--oracle-base-url", "http://127.0.0.1:9"],
            &["--oracle-max-price", "1,2"],
            &["--no-oracle-learning"],
        ] {
            let mut v = vec!["cortiq", "serve", "d.cmf"];
            v.extend_from_slice(a);
            assert!(parse(&v).is_err(), "{a:?}");
        }
        assert!(
            parse(&[
                "cortiq",
                "serve",
                "d.cmf",
                "--oracle",
                "m/x",
                "--oracle-max-price",
                "1"
            ])
            .is_err()
        );
    }

    #[test]
    fn keys_create_allows_the_oracle_unless_opted_out() {
        let allowed = |extra: &[&str]| {
            let mut v = vec!["cortiq", "decision", "keys", "create", "--state", "s"];
            v.extend_from_slice(extra);
            match parse(&v).unwrap().command {
                Commands::Decision {
                    cmd:
                        DecisionCmd::Keys {
                            cmd: KeysCmd::Create { oracle_allowed, .. },
                        },
                } => oracle_allowed,
                _ => panic!("not keys create"),
            }
        };
        assert_eq!(allowed(&[]), None);
        assert_eq!(allowed(&["--oracle-allowed"]), Some(true));
        assert_eq!(allowed(&["--oracle-allowed=false"]), Some(false));
        assert!(
            parse(&[
                "cortiq",
                "decision",
                "keys",
                "create",
                "--state",
                "s",
                "--oracle-allowed=x"
            ])
            .is_err()
        );
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
    fn decide_parses_the_oracle_flags_which_need_oracle() {
        match parse(&[
            "cortiq",
            "decide",
            "d.cmf",
            "--input",
            "r.jsonl",
            "--oracle",
            "deepseek/deepseek-v4.1-flash",
            "--oracle-budget",
            "0.5",
            "--oracle-max-calls",
            "20",
            "--oracle-key-env",
            "MY_KEY",
            "--oracle-base-url",
            "http://127.0.0.1:9/api/v1",
            "--oracle-max-price",
            "0.2,0.8",
            "--oracle-resume",
            "--break-lock",
            "--state",
            "st",
        ])
        .unwrap()
        .command
        {
            Commands::Decide(a) => {
                assert!(a.oracle.oracle_resume && a.oracle.break_lock);
                let f = a.oracle.flags().unwrap();
                assert_eq!(f.model, "deepseek/deepseek-v4.1-flash");
                assert_eq!((f.budget_usd, f.max_calls), (Some(0.5), Some(20)));
                assert_eq!(f.key_env.as_deref(), Some("MY_KEY"));
                assert_eq!(f.base_url.as_deref(), Some("http://127.0.0.1:9/api/v1"));
                assert_eq!(f.max_price, Some((0.2, 0.8)));
                assert!(f.no_learning, "decide never teaches");
                assert_eq!(a.state.as_deref(), Some(Path::new("st")));
            }
            _ => panic!("not decide"),
        }
        match parse(&["cortiq", "decide", "d.cmf", "-p", "x"])
            .unwrap()
            .command
        {
            Commands::Decide(a) => assert_eq!(a.oracle.flags(), None),
            _ => panic!("not decide"),
        }
        for extra in [
            &["--oracle-budget", "1"][..],
            &["--oracle-max-calls", "1"],
            &["--oracle-key-env", "K"],
            &["--oracle-base-url", "http://127.0.0.1:9"],
            &["--oracle-max-price", "1,2"],
            &["--oracle-resume"],
            &["--break-lock"],
        ] {
            let mut v = vec!["cortiq", "decide", "d.cmf", "-p", "x"];
            v.extend_from_slice(extra);
            assert!(parse(&v).is_err(), "{extra:?}");
        }
        // decision oracle check: defaults, and every flag.
        match parse(&["cortiq", "decision", "oracle", "check"])
            .unwrap()
            .command
        {
            Commands::Decision {
                cmd:
                    DecisionCmd::Oracle {
                        cmd:
                            OracleCmd::Check {
                                model,
                                key_env,
                                base_url,
                                test_call,
                                max_price,
                                json,
                            },
                    },
            } => {
                assert_eq!(model, DEFAULT_ORACLE_MODEL);
                assert_eq!(key_env, "OPENROUTER_API_KEY");
                assert_eq!(base_url, "https://openrouter.ai/api/v1");
                assert!(!test_call && !json);
                assert_eq!(max_price, None);
            }
            _ => panic!("not oracle check"),
        }
        match parse(&[
            "cortiq",
            "decision",
            "oracle",
            "check",
            "--model",
            "a/b",
            "--key-env",
            "K",
            "--base-url",
            "http://127.0.0.1:9",
            "--test-call",
            "--max-price",
            "1,4",
            "--json",
        ])
        .unwrap()
        .command
        {
            Commands::Decision {
                cmd:
                    DecisionCmd::Oracle {
                        cmd: OracleCmd::Check { max_price, .. },
                    },
            } => assert_eq!(max_price, Some((1.0, 4.0))),
            _ => panic!("not oracle check"),
        }
        assert!(parse(&["cortiq", "decision", "oracle", "check", "--max-price", "1"]).is_err());
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
