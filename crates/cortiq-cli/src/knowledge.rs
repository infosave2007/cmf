//! Format-v2 knowledge tools (spec §9.2–§9.6): the router-v2 request
//! decision as the CLI tools use it (`Lanes`), and the gate commands —
//!
//! * `route-eval`    — decision per prompt, accept/recall rate with its
//!   exact Clopper–Pearson 95 % upper bound, per-`src` / per-`lang`
//!   breakdown, input sha256 (gate G3);
//! * `dump-logits` / `logits-compare` — the cmf-im-v1 prompt + N greedy
//!   tokens, every position's logits, and the bitwise comparison of two
//!   dumps (gate G2: F0 without a router vs F1 `--route auto`, every
//!   F1-backbone record bit-identical to F0's, and no F0-backbone prompt
//!   taken by a `key_first` lookup record);
//! * `skill-gate`    — status + measured gate of a v2 skill, written by a
//!   header-only tail append with a lineage event;
//! * `genome-verify` — trunk hash, trunk entries and the byte prefix of an
//!   append-only successor (gate G1).

use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_core::knowledge::{LineageEvent, hex64, is_trunk_tensor};
use cortiq_engine::lookup::{self, LookupMode, LookupOutcome, LookupTable, LookupTables};
use cortiq_engine::router::{self, RouteDecision, RouteOptions, RouteTarget};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

// ───────────────────────── routing for the tools ─────────────────────────

/// What a tool runs each prompt on (`--route`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteMode {
    /// No routing at all — the plain pipeline (legacy files by default:
    /// exactly the behaviour before router v2).
    None,
    /// The per-request decision (`router::route_request`).
    Auto,
    Backbone,
    Skill(String),
}

impl RouteMode {
    /// `--route auto|backbone|<id>`; without the flag a ROUTER_V2 file
    /// routes (`auto`) and a legacy file does not (`None`).
    pub fn resolve(flag: Option<&str>, model: &CmfModel) -> anyhow::Result<Self> {
        Ok(match flag {
            None if router::is_router_v2(model) => Self::Auto,
            None => Self::None,
            Some("auto") => Self::Auto,
            Some("backbone") | Some("none") => Self::Backbone,
            Some(id) => {
                anyhow::ensure!(
                    model.header.skills.iter().any(|s| s.id == id),
                    "--route {id}: no such skill (header.skills: {:?})",
                    model
                        .header
                        .skills
                        .iter()
                        .map(|s| &s.id)
                        .collect::<Vec<_>>()
                );
                Self::Skill(id.to_string())
            }
        })
    }

    pub fn label(&self) -> String {
        match self {
            Self::None => "none".into(),
            Self::Auto => "auto".into(),
            Self::Backbone => "backbone".into(),
            Self::Skill(s) => s.clone(),
        }
    }
}

/// One pipeline per route target, created on first use (the backbone
/// lane is a plain pipeline — no overlay; a skill lane is loaded WITH the
/// skill, never switched), plus a separate backbone pipeline that only
/// computes φ, so a decision never resets a lane's sequence state.
pub struct Lanes {
    model: Arc<CmfModel>,
    sampler: SamplerConfig,
    opts: RouteOptions,
    probe: Option<Pipeline>,
    lanes: BTreeMap<Option<String>, Pipeline>,
    /// Every lane runs per-op (`Pipeline::mark_graph_refused` at
    /// creation): the host route, the path `growth-eval` counts on.
    per_op: bool,
    /// The file's lookup records (spec §9.5.2), opened on first use. A
    /// lookup target has no lane: the table answers, or the backbone lane
    /// runs (with the card prepended in `context` mode).
    lookups: LookupTables,
    lookup_mode: LookupMode,
}

impl Lanes {
    pub fn new(model: Arc<CmfModel>, sampler: SamplerConfig, opts: RouteOptions) -> Self {
        Self {
            lookups: LookupTables::new(model.clone()),
            model,
            sampler,
            opts,
            probe: None,
            lanes: BTreeMap::new(),
            per_op: false,
            lookup_mode: LookupMode::default(),
        }
    }

    /// Pin every lane to the per-op forward (see [`ForwardPath`]).
    pub fn per_op(mut self, on: bool) -> Self {
        self.per_op = on;
        self
    }

    /// What a request routed to a lookup record does (`--lookup-mode`,
    /// `CMF_LOOKUP_MODE`).
    pub fn lookup_mode(mut self, mode: LookupMode) -> Self {
        self.lookup_mode = mode;
        self
    }

    /// [`Self::decide`], then the lookup step: a decision for a lookup
    /// record becomes its outcome (`lookup::resolve_lookup_gated`) — no
    /// key in the message sends the request to the backbone unchanged.
    /// Under `--route auto` the decision is the router's, so a `key_first`
    /// record may take a backbone decision on a strong key of the message
    /// (the tools frame every prompt as one cmf-im-v1 turn); a pinned
    /// route (`backbone`, `<id>`, no routing) is taken as given.
    pub fn decide_lookup(
        &mut self,
        mode: &RouteMode,
        user_text: &str,
    ) -> anyhow::Result<(RouteDecision, LookupOutcome)> {
        self.decide_lookup_turns(mode, &[user_text], router::PromptFrame::CmfImV1)
    }

    /// [`Self::decide_lookup`] with conversation memory, the way `serve`'s
    /// lookup pre-pass decides (`SkillRouter::decide_lookup_turns`):
    /// `turns` are the user messages, the LAST one first — the decision
    /// (φ, the prompt contract against `frame`) is made on it alone — then
    /// the earlier ones back in time (at most `lookup::MEMORY_TURNS` in
    /// all); the key comes from the most recent turn that holds one, the
    /// field and the language from the last. Under `--route auto` a
    /// `key_first` record may take a backbone decision on a strong key of
    /// the LAST message only.
    pub fn decide_lookup_turns(
        &mut self,
        mode: &RouteMode,
        turns: &[&str],
        frame: router::PromptFrame,
    ) -> anyhow::Result<(RouteDecision, LookupOutcome)> {
        let last = turns.first().copied().unwrap_or("");
        let d = self.decide_framed(mode, last, frame)?;
        let gate = (*mode == RouteMode::Auto).then(|| lookup::KeyFirstGate::new(self.opts, frame));
        lookup::resolve_lookup_gated(&self.lookups, d, gate, turns, self.lookup_mode)
            .map_err(anyhow::Error::msg)
    }

    /// The table of lookup record `id` (opened on first use); `None` when
    /// `id` is not a lookup record.
    pub fn lookup_table(&self, id: &str) -> anyhow::Result<Option<Arc<LookupTable>>> {
        self.lookups.get(id).map_err(anyhow::Error::msg)
    }

    /// Ids of the file's lookup records, in header order.
    pub fn lookup_ids(&self) -> Vec<String> {
        self.lookups.ids()
    }

    /// The decision for one user message under `mode` (the tools' frame:
    /// one cmf-im-v1 turn).
    pub fn decide(&mut self, mode: &RouteMode, user_text: &str) -> anyhow::Result<RouteDecision> {
        self.decide_framed(mode, user_text, router::PromptFrame::CmfImV1)
    }

    /// [`Self::decide`] for a request that generates under `frame`: the
    /// router's pick must meet its prompt contract in that frame (as
    /// `serve` checks the tokenizer's chat frame; no re-rendering).
    pub fn decide_framed(
        &mut self,
        mode: &RouteMode,
        user_text: &str,
        frame: router::PromptFrame,
    ) -> anyhow::Result<RouteDecision> {
        Ok(match mode {
            RouteMode::None => RouteDecision::forced(
                RouteTarget::Backbone,
                "no routing (the file declares no router policy)",
            ),
            RouteMode::Backbone => {
                RouteDecision::forced(RouteTarget::Backbone, "forced: --route backbone")
            }
            RouteMode::Skill(id) => RouteDecision::forced(
                RouteTarget::Skill(id.clone()),
                format!("forced: --route {id}"),
            ),
            RouteMode::Auto => {
                if self.probe.is_none() {
                    self.probe = Some(Pipeline::from_model(&self.model, SamplerConfig::default())?);
                }
                let probe = self.probe.as_mut().expect("probe pipeline");
                let d = router::route_request_with(&self.model, probe, user_text, self.opts);
                // The tools render every prompt as one cmf-im-v1 turn
                // (utility::chat_prefix) and pass that frame; probe-dialog
                // passes the tokenizer's chat frame, as serve does. The
                // skill's contract is checked against it.
                router::enforce_prompt_contract(&self.model.header, d, frame, false).0
            }
        })
    }

    /// The entry of lookup record `id` that `src` — a prompt's source
    /// name, e.g. the plant a recall prompt was made from — resolves to
    /// through the runtime's key extraction; `None` when `id` is not a
    /// lookup record or `src` holds no key. `probe-utility` compares it
    /// with the entry a hit answered from (`lookup_src_match`).
    pub fn lookup_entry_of(&self, id: &str, src: &str) -> anyhow::Result<Option<u32>> {
        let table = self.lookups.get(id).map_err(anyhow::Error::msg)?;
        Ok(table.and_then(|t| t.find_key(src).map(|h| h.entry)))
    }

    /// The pipeline that runs `target`. A lookup record has no lane of its
    /// own: the BACKBONE pipeline — the same object F0 runs — serves it
    /// (the table never touches the network).
    pub fn lane(&mut self, target: &RouteTarget) -> anyhow::Result<&mut Pipeline> {
        let key = match target {
            RouteTarget::Skill(id) if LookupTable::is_lookup(&self.model, id) => None,
            _ => target_key(target),
        };
        if !self.lanes.contains_key(&key) {
            let p =
                Pipeline::from_model_with_skill(&self.model, self.sampler.clone(), key.as_deref())
                    .with_context(|| {
                        format!("loading the {} lane", key.as_deref().unwrap_or("backbone"))
                    })?;
            if self.per_op {
                p.mark_graph_refused();
            }
            self.lanes.insert(key.clone(), p);
        }
        Ok(self.lanes.get_mut(&key).expect("lane just inserted"))
    }
}

fn target_key(target: &RouteTarget) -> Option<String> {
    match target {
        RouteTarget::Backbone => None,
        RouteTarget::Skill(s) => Some(s.clone()),
    }
}

/// `dump-logits` route byte: 0 = backbone, `1 + i` = `header.skills[i]`.
pub fn route_code(model: &CmfModel, target: &RouteTarget) -> anyhow::Result<u8> {
    match target {
        RouteTarget::Backbone => Ok(0),
        RouteTarget::Skill(id) => {
            let i = model
                .header
                .skills
                .iter()
                .position(|s| &s.id == id)
                .ok_or_else(|| anyhow::anyhow!("skill '{id}' not in header.skills"))?;
            u8::try_from(i + 1)
                .map_err(|_| anyhow::anyhow!("skill index {i} does not fit a u8 code"))
        }
    }
}

/// Human line for stderr.
pub fn describe(d: &RouteDecision) -> String {
    d.describe()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The backend φ is computed on in THIS process: `gpu` when a GPU backend
/// is up (after the first pipeline was created — call it after that),
/// else `cpu`. `route-fit` records it in `router.measured.phi_backend`.
pub fn phi_backend_label() -> &'static str {
    if cortiq_engine::gpu::enabled() {
        "gpu"
    } else {
        "cpu"
    }
}

/// Warn (stderr) when the file's router was fitted on another backend
/// than this process runs φ on: a coop-GEMM GPU computes φ to tf32 class
/// (≈ 1e-4 relative) against the CPU's 1e-6, so `E_base` / `E_skill`
/// shift by the same order and prompts near θ or the margin change their
/// verdict — the measured G3 false-accept does not transfer. Returns the
/// fitted backend when it differs.
pub fn warn_phi_backend(model: &CmfModel, tool: &str) -> Option<String> {
    let fitted = model
        .header
        .router
        .as_ref()
        .and_then(|r| r.measured.as_ref())
        .and_then(|m| m.get("phi_backend"))
        .and_then(|b| b.as_str())?;
    let here = phi_backend_label();
    if fitted == here {
        return None;
    }
    eprintln!(
        "warning: {tool}: the router of this file was fitted with φ on the {fitted} backend, \
         this process computes φ on the {here} backend — E_base / E_skill differ by the \
         backends' numerical class (≈ 1e-4 on a coop-GEMM GPU), the measured gate does not \
         transfer; refit (route-fit) or re-gate (route-eval) on this backend"
    );
    Some(fitted.to_string())
}

// ───────────────────────── prompts ─────────────────────────

pub struct EvalRow {
    pub prompt: String,
    pub lang: String,
    pub src: String,
}

/// JSONL `{"prompt", "lang"?, "src"?, …}` per line (the probe-utility
/// shape; other keys ignored).
pub fn parse_eval_jsonl(text: &str) -> anyhow::Result<Vec<EvalRow>> {
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("prompts jsonl line {}", ln + 1))?;
        let prompt = v
            .get("prompt")
            .and_then(|p| p.as_str())
            .ok_or_else(|| anyhow::anyhow!("prompts jsonl line {}: no \"prompt\"", ln + 1))?;
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("?").to_string();
        rows.push(EvalRow {
            prompt: prompt.to_string(),
            lang: s("lang"),
            src: s("src"),
        });
    }
    anyhow::ensure!(!rows.is_empty(), "prompt set is empty");
    Ok(rows)
}

// ───────────────────────── route-eval ─────────────────────────

pub struct RouteEvalArgs<'a> {
    pub model: &'a str,
    pub prompts_jsonl: &'a str,
    pub expect: &'a str,
    pub include_quarantine: bool,
    /// The φ router alone, without the `key_first` step (a diagnostic;
    /// not a gate for a key_first record).
    pub router_only: bool,
    pub json: bool,
}

/// The lookup records of `model` whose EFFECTIVE policy is `key_first`
/// (header order).
pub fn key_first_records(model: &CmfModel) -> Vec<String> {
    model
        .header
        .skills
        .iter()
        .filter(|s| {
            s.kind.as_deref() == Some(cortiq_core::knowledge::skill_kind::LOOKUP)
                && s.lookup
                    .as_ref()
                    .is_some_and(|l| lookup::LookupPolicy::of(l) == lookup::LookupPolicy::KeyFirst)
        })
        .map(|s| s.id.clone())
        .collect()
}

/// Does `gate` (a route-eval summary) measure the `key_first` step for
/// record `id` — the decision serve makes on a key_first file? A measured
/// run with `key_first_step: true` whose `key_first_records` lists `id`
/// (review KF-2). `skill-gate --status active` demands it for a key_first
/// record; `lookup-policy` keeps an active record active on a switch to
/// key_first only with such a gate.
pub fn gate_covers_key_first(gate: &serde_json::Value, id: &str) -> bool {
    gate.get("status").and_then(|s| s.as_str()) == Some("measured")
        && gate.get("key_first_step") == Some(&serde_json::Value::Bool(true))
        && gate
            .get("key_first_records")
            .and_then(|r| r.as_array())
            .is_some_and(|r| r.iter().any(|x| x.as_str() == Some(id)))
}

#[derive(Default, Clone, Copy)]
struct Tally {
    n: usize,
    ok: usize,
}

impl Tally {
    fn add(&mut self, ok: bool) {
        self.n += 1;
        self.ok += usize::from(ok);
    }
    fn json(&self) -> serde_json::Value {
        let err = self.n - self.ok;
        serde_json::json!({
            "n": self.n,
            "accepted": self.ok,
            "errors": err,
            "rate": if self.n > 0 { self.ok as f64 / self.n as f64 } else { f64::NAN },
            "error_upper95": router::clopper_pearson_upper(err, self.n, 0.95),
        })
    }
}

/// G3 thresholds (spec §6): false accept upper95 ≤ 0.02 on a general set
/// of at least [`G3_MIN_GENERAL_N`] prompts, in-scope recall ≥ 0.90.
pub const G3_FALSE_ACCEPT_UPPER95: f64 = 0.02;
pub const G3_IN_SCOPE_RECALL: f64 = 0.90;
/// G3 needs n ≥ 500 general prompts: with zero errors the CP bound is
/// already ≤ 0.02 at n = 183, so the bound alone does not enforce it.
pub const G3_MIN_GENERAL_N: usize = 500;

/// Is this decision "the backbone ran"? By the TARGET, never by the label:
/// a skill id is never spelled like the backbone (core reserves
/// `backbone`/`none`/`auto` for v2 ids, R8), and even a legacy record so
/// named is a skill here.
fn decision_ok(expect: &str, d: &RouteDecision) -> bool {
    if expect == "backbone" {
        d.skill().is_none()
    } else {
        d.skill() == Some(expect)
    }
}

/// Summary of a route-eval run. `vacuous`: why no skill could win any
/// prompt under the evaluated options (see [`eval_candidates`]) — the
/// gate is then `pass: false, vacuous: true`, whatever the counts say.
/// The summary is a valid gate file for `skill-gate`: `status` is
/// `measured` for a real measurement and `vacuous` otherwise, so a
/// vacuous run can never activate a record (`is_auto_routable` needs
/// `gate.status == "measured"`).
#[cfg(test)]
pub fn route_eval_summary(
    expect: &str,
    decisions: &[(EvalRow, RouteDecision)],
    vacuous: Option<&str>,
) -> serde_json::Value {
    route_eval_summary_by(expect, decisions, &vec![None; decisions.len()], vacuous)
}

/// The summary of a route-eval run (as described for `route_eval_summary`,
/// its test-only shorthand without `decided_by`) with who decided each
/// row (`decided_by[i]`:
/// `Some` for a decision naming a lookup record — the router's pick or a
/// `key_first` take — `None` otherwise): `decided_by_counts` counts them,
/// and `key_first_accepts` the rows `key_first` sent to a record (each an
/// accept of that record like any other — a false accept on a general set).
pub fn route_eval_summary_by(
    expect: &str,
    decisions: &[(EvalRow, RouteDecision)],
    decided_by: &[Option<lookup::DecidedBy>],
    vacuous: Option<&str>,
) -> serde_json::Value {
    let mut all = Tally::default();
    let mut per_src: BTreeMap<String, Tally> = BTreeMap::new();
    let mut per_lang: BTreeMap<String, Tally> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (i, (row, d)) in decisions.iter().enumerate() {
        let ok = decision_ok(expect, d);
        all.add(ok);
        per_src.entry(row.src.clone()).or_default().add(ok);
        per_lang.entry(row.lang.clone()).or_default().add(ok);
        *counts.entry(d.target_label().to_string()).or_default() += 1;
        if let Some(by) = decided_by.get(i).copied().flatten() {
            *by_counts.entry(by.label()).or_default() += 1;
        }
    }
    let err = all.n - all.ok;
    let rate = if all.n > 0 {
        all.ok as f64 / all.n as f64
    } else {
        f64::NAN
    };
    let upper = router::clopper_pearson_upper(err, all.n, 0.95);
    let backbone = expect == "backbone";
    let gate = if backbone {
        let enough = all.n >= G3_MIN_GENERAL_N;
        serde_json::json!({
            "metric": "false_accept_upper95",
            "threshold": G3_FALSE_ACCEPT_UPPER95,
            "min_n": G3_MIN_GENERAL_N,
            "n_ok": enough,
            "vacuous": vacuous.is_some(),
            "reason": vacuous,
            "pass": vacuous.is_none() && enough && upper <= G3_FALSE_ACCEPT_UPPER95,
        })
    } else {
        serde_json::json!({
            "metric": "in_scope_recall",
            "threshold": G3_IN_SCOPE_RECALL,
            "vacuous": vacuous.is_some(),
            "reason": vacuous,
            "pass": vacuous.is_none() && rate >= G3_IN_SCOPE_RECALL,
        })
    };
    let per = |m: BTreeMap<String, Tally>| -> serde_json::Value {
        m.into_iter().map(|(k, t)| (k, t.json())).collect()
    };
    serde_json::json!({
        "status": if vacuous.is_some() { "vacuous" } else { "measured" },
        "expect": expect,
        "n": all.n,
        "accepted": all.ok,
        "errors": err,
        "accept_rate": rate,
        "error_rate": if all.n > 0 { err as f64 / all.n as f64 } else { f64::NAN },
        "error_upper95": upper,
        // named the way the gate reads them
        "metric": if backbone { "backbone_accept" } else { "in_scope_recall" },
        "false_accept": if backbone { serde_json::json!(err as f64 / all.n.max(1) as f64) } else { serde_json::Value::Null },
        "false_accept_upper95": if backbone { serde_json::json!(upper) } else { serde_json::Value::Null },
        "in_scope_recall": if backbone { serde_json::Value::Null } else { serde_json::json!(rate) },
        "gate": gate,
        "route_counts": counts,
        "decided_by_counts": by_counts,
        "key_first_accepts": by_counts.get("key_first").copied().unwrap_or(0),
        "per_src": per(per_src),
        "per_lang": per(per_lang),
    })
}

/// The skills a route-eval run can see win under `opts` (`Ok`), or why
/// none can (`Err` — the measurement is vacuous: every prompt runs the
/// backbone by construction, not by the router's judgement). Router-v2
/// file: policy + calibration + matching `skills_hash` + ≥ 1 class
/// routable under `opts` ([`router::no_candidate_reason`]); legacy file:
/// ≥ 1 v1 record with a selection descriptor.
pub fn eval_candidates(model: &CmfModel, opts: RouteOptions) -> Result<Vec<String>, String> {
    if router::is_router_v2(model) {
        if let Some(why) = router::no_candidate_reason(&model.header, opts) {
            return Err(why);
        }
        return Ok(router::routable_skills(&model.header, opts)
            .into_iter()
            .map(|s| s.id.clone())
            .collect());
    }
    let v1: Vec<String> = model
        .header
        .skills
        .iter()
        .filter(|s| !s.is_v2() && s.selection.is_some())
        .map(|s| s.id.clone())
        .collect();
    if v1.is_empty() {
        return Err("no router policy and no routable legacy skill in this file".into());
    }
    Ok(v1)
}

/// Refuse a gate set that IS a calibration set (`router.measured.
/// general_sha256` / `in_sha256`): G3 needs a disjoint set.
fn refuse_calibration_set(model: &CmfModel, sha: &str) -> anyhow::Result<()> {
    let measured = model.header.router.as_ref().and_then(|r| r.measured.as_ref());
    for key in ["general_sha256", "in_sha256"] {
        if measured
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.eq_ignore_ascii_case(sha))
        {
            anyhow::bail!(
                "route-eval: the prompt set is the router's calibration set (router.measured.{key} \
                 = {sha}) — G3 is measured on a DISJOINT set"
            );
        }
    }
    Ok(())
}

pub fn cmd_route_eval(a: RouteEvalArgs<'_>) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    let bytes = std::fs::read(a.prompts_jsonl).with_context(|| a.prompts_jsonl.to_string())?;
    let rows = parse_eval_jsonl(std::str::from_utf8(&bytes).context("prompts are not UTF-8")?)?;
    let prompts_sha = sha256_hex(&bytes);
    refuse_calibration_set(&model, &prompts_sha)?;
    let opts = RouteOptions {
        include_quarantine: a.include_quarantine,
    };
    // Before a single decision: can any skill win at all? A skill in
    // quarantine without --include-quarantine, a stale skills_hash or no
    // calibration all send every prompt to the backbone — a backbone
    // "accept rate" measured there says nothing about false accepts.
    let candidates = eval_candidates(&model, opts);
    let mut vacuous = candidates.as_ref().err().cloned();
    if a.expect != "backbone" {
        let s = model
            .header
            .skills
            .iter()
            .find(|s| s.id == a.expect)
            .ok_or_else(|| anyhow::anyhow!("--expect {}: no such skill", a.expect))?;
        if vacuous.is_none() && !candidates.as_ref().is_ok_and(|c| c.contains(&s.id)) {
            vacuous = Some(format!(
                "skill '{}' cannot win under these options (status {:?}, gate {:?}; \
                 --include-quarantine scores any non-retired class)",
                s.id,
                s.status,
                s.gate.as_ref().and_then(|g| g.get("status"))
            ));
        }
    }
    if let Some(why) = &vacuous {
        eprintln!("route-eval: VACUOUS measurement — {why}");
    }
    let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
    let fitted_backend = warn_phi_backend(&model, "route-eval");
    // The decision serve makes (review KF-2): on a file with a key_first
    // lookup record, the φ router's decision THEN the key_first step (a
    // strong key of the prompt takes a backbone decision); a take is an
    // accept of the record. Everything else is the router's decision as
    // it was — a router pick of a lookup record counts even when the
    // table would then miss (conservative: the old G3 numbers stand).
    let kf_records = key_first_records(&model);
    let key_first_step = !a.router_only && !kf_records.is_empty();
    if a.router_only && !kf_records.is_empty() {
        eprintln!(
            "route-eval: --router-only on a file with key_first record(s) {kf_records:?} — the \
             φ router alone is measured; serve also applies the key_first step, and this summary \
             is no gate for those records"
        );
    }
    let tables = LookupTables::new(model.clone());
    let gate = lookup::KeyFirstGate::new(opts, router::PromptFrame::CmfImV1);
    let mut decisions = Vec::with_capacity(rows.len());
    let mut decided_by = Vec::with_capacity(rows.len());
    for row in rows {
        let d = router::route_request_with(&model, &mut probe, &row.prompt, opts);
        let (d, by) = if key_first_step {
            let (dk, outcome) = lookup::resolve_lookup_gated(
                &tables,
                d.clone(),
                Some(gate),
                &[row.prompt.as_str()],
                LookupMode::Answer,
            )
            .map_err(anyhow::Error::msg)?;
            match outcome.decided_by() {
                Some(lookup::DecidedBy::KeyFirst) => (dk, Some(lookup::DecidedBy::KeyFirst)),
                Some(by) => (d, Some(by)),
                None => (d, None),
            }
        } else {
            let by = d
                .skill()
                .filter(|id| LookupTable::is_lookup(&model, id))
                .map(|_| lookup::DecidedBy::Router);
            (d, by)
        };
        decisions.push((row, d));
        decided_by.push(by);
    }
    let mut summary =
        route_eval_summary_by(a.expect, &decisions, &decided_by, vacuous.as_deref());
    summary["key_first_step"] = serde_json::json!(key_first_step);
    summary["key_first_records"] = serde_json::json!(if key_first_step { kf_records.clone() } else { Vec::new() });
    summary["router_only"] = serde_json::json!(a.router_only);
    summary["model"] = serde_json::json!(a.model);
    summary["router_v2"] = serde_json::json!(router::is_router_v2(&model));
    summary["phi_backend"] = serde_json::json!(phi_backend_label());
    summary["phi_backend_fitted"] = serde_json::json!(fitted_backend);
    summary["include_quarantine"] = serde_json::json!(a.include_quarantine);
    summary["candidates"] = match &candidates {
        Ok(c) => serde_json::json!(c),
        Err(_) => serde_json::json!([]),
    };
    summary["vacuous"] = serde_json::json!(vacuous.is_some());
    summary["prompts"] = serde_json::json!(a.prompts_jsonl);
    summary["prompts_sha256"] = serde_json::json!(prompts_sha);
    if a.json {
        summary["rows"] = decisions
            .iter()
            .enumerate()
            .map(|(i, (row, d))| {
                let mut j = d.summary_json();
                j["index"] = serde_json::json!(i);
                j["lang"] = serde_json::json!(row.lang);
                j["src"] = serde_json::json!(row.src);
                j["ok"] = serde_json::json!(decision_ok(a.expect, d));
                j["decided_by"] = serde_json::json!(decided_by[i].map(|b| b.label()));
                j
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        for (i, (row, d)) in decisions.iter().enumerate() {
            let ok = decision_ok(a.expect, d);
            println!(
                "{i:4} {} {}:{} | {}",
                if ok { "ok  " } else { "MISS" },
                row.lang,
                row.src,
                describe(d)
            );
        }
        println!(
            "== {} vs expect '{}': {}/{} accepted (rate {:.4}), errors {} → CP95 upper {:.4} | gate {} | sha256 {}",
            a.model,
            a.expect,
            summary["accepted"],
            summary["n"],
            summary["accept_rate"].as_f64().unwrap_or(f64::NAN),
            summary["errors"],
            summary["error_upper95"].as_f64().unwrap_or(f64::NAN),
            summary["gate"],
            summary["prompts_sha256"].as_str().unwrap_or("")
        );
        println!("   per_lang {}", summary["per_lang"]);
        println!(
            "   route_counts {} | decided_by {} | key_first_step {}",
            summary["route_counts"], summary["decided_by_counts"], summary["key_first_step"]
        );
    }
    // A vacuous gate is a failed measurement, not a pass: a script reading
    // the exit status must not feed it to `skill-gate --status active`.
    if let Some(why) = vacuous {
        anyhow::bail!("route-eval: vacuous gate (gate.pass = false) — {why}");
    }
    Ok(())
}

// ───────────────────────── dump-logits ─────────────────────────

/// Record header: index u32, route u8, n_positions u32, vocab u32 (LE),
/// then `n_positions × vocab` f32 LE — the last prompt position first,
/// then every generated position.
pub const RECORD_HEADER: usize = 4 + 1 + 4 + 4;

pub fn write_record<W: Write>(
    w: &mut W,
    index: u32,
    route: u8,
    rows: &[Vec<f32>],
) -> std::io::Result<()> {
    let vocab = rows.first().map_or(0, |r| r.len());
    w.write_all(&index.to_le_bytes())?;
    w.write_all(&[route])?;
    w.write_all(&(rows.len() as u32).to_le_bytes())?;
    w.write_all(&(vocab as u32).to_le_bytes())?;
    for r in rows {
        assert_eq!(r.len(), vocab, "every position has the same vocab");
        for x in r {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

/// The forward path a measurement runs on (`--path`). G2 compares logits
/// bit for bit and `growth-eval` counts the wins the HOST route makes
/// (`moe_route`): the resident graph selects its expert on the device
/// (`embryo_core_route_pick`, a tree reduction against the host's
/// sequential sum — parity ≤ 5e-6, not equality) and exports no counter,
/// so a token near an expert's shell or near the best trunk score can win
/// on one path and lose on the other. Both sides of a comparison, and the
/// index set that restricts it, must therefore come from ONE path; the
/// dumps record theirs in `<out>.meta.json` and `logits-compare` refuses
/// a mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardPath {
    /// The host route on every lane (`Pipeline::mark_graph_refused`): what
    /// `growth-eval` measures. The default.
    PerOp,
    /// The engine's own choice — the resident / whole-token graph where the
    /// backend has one (what `serve` runs), per-op otherwise.
    Auto,
}

impl ForwardPath {
    pub fn parse(s: Option<&str>) -> anyhow::Result<Self> {
        match s {
            None | Some("per-op") => Ok(Self::PerOp),
            Some("auto") => Ok(Self::Auto),
            Some(other) => anyhow::bail!("--path {other}: expected per-op | auto"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PerOp => "per-op",
            Self::Auto => "auto",
        }
    }
}

/// `<dump>.meta.json`: how a `dump-logits` file was produced.
pub fn dump_meta_path(dump: &str) -> String {
    format!("{dump}.meta.json")
}

/// The sidecar's fields `logits-compare` checks (absent = an older dump).
#[derive(Debug, Clone, Default)]
pub struct DumpMeta {
    pub path: Option<String>,
    pub shell: Option<String>,
    pub growth: Option<String>,
    /// Record indices a `key_first` lookup record took over a backbone
    /// decision of the router (`None` = a dump from before the field).
    pub key_first: Option<Vec<u32>>,
    /// The dump's `lookup_mode` (present on dumps that know lookup).
    pub lookup_mode: Option<String>,
}

pub fn read_dump_meta(dump: &str) -> anyhow::Result<Option<DumpMeta>> {
    let p = dump_meta_path(dump);
    let text = match std::fs::read_to_string(&p) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| p),
    };
    let v: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("{p}: not JSON"))?;
    let field = |k: &str| v.get(k).and_then(|x| x.as_str()).map(String::from);
    let key_first = v.get("key_first_indices").and_then(|x| x.as_array()).map(|a| {
        a.iter()
            .filter_map(|i| i.as_u64().and_then(|i| u32::try_from(i).ok()))
            .collect()
    });
    Ok(Some(DumpMeta {
        path: field("path"),
        shell: field("shell"),
        growth: field("growth"),
        key_first,
        lookup_mode: field("lookup_mode"),
    }))
}

pub struct DumpArgs<'a> {
    pub model: &'a str,
    pub prompts_jsonl: &'a str,
    pub tokens: usize,
    pub route: Option<&'a str>,
    pub include_quarantine: bool,
    pub out: &'a str,
    /// `per-op` | `auto`; None = per-op (see [`ForwardPath`]).
    pub path: Option<&'a str>,
    /// `answer` | `context` | `off`; None = `CMF_LOOKUP_MODE` / `answer`.
    /// A lookup hit in `answer` mode has no logits of its own: the record
    /// carries the BACKBONE's logits on the plain prompt (bit-identical to
    /// F0 by construction) under the skill's route code; `context` runs
    /// the backbone on the card-prepended prompt.
    pub lookup_mode: Option<&'a str>,
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// A dump being written: records go to `<out>.partial`, renamed to `<out>`
/// only when every prompt was dumped. An error (or a panic) removes the
/// partial file — a truncated dump never looks like a complete one (R3).
struct PartialOut {
    tmp: std::path::PathBuf,
    out: std::path::PathBuf,
    done: bool,
}

impl PartialOut {
    fn new(out: &str) -> Self {
        let out = std::path::PathBuf::from(out);
        let mut tmp = out.clone().into_os_string();
        tmp.push(".partial");
        Self {
            tmp: tmp.into(),
            out,
            done: false,
        }
    }

    fn commit(mut self) -> std::io::Result<()> {
        std::fs::rename(&self.tmp, &self.out)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for PartialOut {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

pub fn cmd_dump_logits(a: DumpArgs<'_>) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    // `run`/`serve` apply the catalog's fallback mask to every request —
    // the backbone's included — while this dump runs the unmasked forward:
    // on a file with masks it would not measure what those paths execute.
    if !model.masks.masks.is_empty() {
        anyhow::bail!(
            "dump-logits: {} carries a mask catalog ({} masks, fallback '{}') that run/serve \
             apply to every request; this tool measures the unmasked forward and refuses such a \
             file (G2 is defined on mask-free genomes)",
            a.model,
            model.masks.masks.len(),
            model
                .masks
                .fallback()
                .map(|m| m.name.as_str())
                .unwrap_or("—")
        );
    }
    let bytes = std::fs::read(a.prompts_jsonl).with_context(|| a.prompts_jsonl.to_string())?;
    let rows = parse_eval_jsonl(std::str::from_utf8(&bytes).context("prompts are not UTF-8")?)?;
    let mode = RouteMode::resolve(a.route, &model)?;
    let path = ForwardPath::parse(a.path)?;
    let greedy = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(0),
        ..Default::default()
    };
    let opts = RouteOptions {
        include_quarantine: a.include_quarantine,
    };
    let lookup_mode = LookupMode::resolve(a.lookup_mode).map_err(anyhow::Error::msg)?;
    let mut lanes = Lanes::new(model.clone(), greedy, opts)
        .per_op(path == ForwardPath::PerOp)
        .lookup_mode(lookup_mode);
    let (mut lookup_hits, mut lookup_key_first) = (0usize, 0usize);
    // Records a key_first take answered from a table over the router's
    // backbone decision: `logits-compare` fails G2 on each one whose F0
    // reference ran the backbone (review KF-2).
    let mut key_first_indices: Vec<u32> = Vec::new();
    let partial = PartialOut::new(a.out);
    // The sidecar goes with the dump: a stale one must not describe the
    // new file (removed here, written after the dump is published).
    let meta_path = dump_meta_path(a.out);
    match std::fs::remove_file(&meta_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| meta_path.clone()),
    }
    // The target is replaced (it was truncated before this fix): a failed
    // run must not leave an older complete dump posing as this one.
    match std::fs::remove_file(a.out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| a.out.to_string()),
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&partial.tmp)
            .with_context(|| partial.tmp.display().to_string())?,
    );
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        let (d, outcome) = lanes.decide_lookup(&mode, &row.prompt)?;
        let code = route_code(&model, &d.target)?;
        eprintln!(
            "[{i}] {}{}",
            describe(&d),
            outcome
                .describe()
                .map(|s| format!(" | {s}"))
                .unwrap_or_default()
        );
        *counts.entry(d.target_label().to_string()).or_default() += 1;
        lookup_hits += outcome.is_hit() as usize;
        if outcome.decided_by() == Some(lookup::DecidedBy::KeyFirst) {
            lookup_key_first += 1;
            key_first_indices.push(i as u32);
        }
        // A lookup target runs the backbone lane: on the plain prompt
        // (`answer` — the table's text is the answer, the logits are the
        // network's untouched view), or on the card-prepended prompt
        // (`context`).
        let user_text = outcome.generation_text(&row.prompt);
        let p = lanes.lane(&d.target)?;
        let ids = p
            .tokenizer
            .encode(&crate::utility::chat_prefix(&user_text));
        // A fresh sequence per prompt: no prefix reuse across records.
        p.reset_session();
        let first = p
            .forward_ids(&ids, None)
            .map_err(|e| anyhow::anyhow!("prompt {i}: forward_ids: {e}"))?;
        let mut positions = Vec::with_capacity(a.tokens + 1);
        positions.push(first);
        for s in 0..a.tokens {
            let t = argmax(positions.last().expect("≥ 1 row"));
            let next = p.decode_step_logits(t, ids.len() + s);
            anyhow::ensure!(
                next.len() == positions[0].len(),
                "prompt {i}: vocab changed between positions"
            );
            positions.push(next);
        }
        write_record(&mut out, i as u32, code, &positions)?;
    }
    out.flush()?;
    out.get_ref().sync_all()?;
    drop(out);
    partial
        .commit()
        .with_context(|| format!("publishing the dump as {}", a.out))?;
    let codes: BTreeMap<String, String> =
        std::iter::once(("0".to_string(), "backbone".to_string()))
            .chain(
                model
                    .header
                    .skills
                    .iter()
                    .enumerate()
                    .map(|(i, s)| ((i + 1).to_string(), s.id.clone())),
            )
            .collect();
    let prompts_sha = sha256_hex(&bytes);
    // The sidecar: the path (what `logits-compare` checks against the
    // other dump and the index set), and the growth switches for the
    // record (`growth-eval --indices-out` writes the same names).
    let meta = serde_json::json!({
        "dump": a.out,
        "model": a.model,
        "prompts": a.prompts_jsonl,
        "prompts_sha256": prompts_sha,
        "records": rows.len(),
        "tokens": a.tokens,
        "positions_per_record": a.tokens + 1,
        "route_mode": mode.label(),
        "path": path.label(),
        "growth": cortiq_engine::loader::growth_mode().label(),
        "shell": if cortiq_engine::pipeline::growth_shell_enabled() { "on" } else { "off" },
        "lookup_mode": lookup_mode.label(),
        "key_first_records": key_first_records(&model),
        "key_first_indices": key_first_indices,
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
        .with_context(|| meta_path.clone())?;
    let mut summary = meta;
    summary["include_quarantine"] = serde_json::Value::Bool(a.include_quarantine);
    summary["lookup_hits"] = serde_json::json!(lookup_hits);
    // hits a `key_first` record took over a backbone decision of the router
    summary["lookup_key_first"] = serde_json::json!(lookup_key_first);
    summary["route_counts"] = serde_json::to_value(&counts)?;
    summary["route_codes"] = serde_json::to_value(&codes)?;
    summary["out"] = serde_json::Value::String(a.out.to_string());
    summary["meta"] = serde_json::Value::String(meta_path);
    summary["bytes"] = serde_json::json!(std::fs::metadata(a.out).map(|m| m.len()).unwrap_or(0));
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

// ───────────────────────── logits-compare ─────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct RecordRef {
    pub index: u32,
    pub route: u8,
    pub n_positions: u32,
    pub vocab: u32,
    /// Payload offset in the file.
    pub offset: u64,
}

impl RecordRef {
    fn payload_bytes(&self) -> u64 {
        self.n_positions as u64 * self.vocab as u64 * 4
    }
}

/// Scan a dump's record headers (payloads are skipped, not read).
pub fn index_records(path: &str) -> anyhow::Result<Vec<RecordRef>> {
    let mut f = std::fs::File::open(path).with_context(|| path.to_string())?;
    let len = f.metadata()?.len();
    let mut out = Vec::new();
    let mut pos = 0u64;
    let mut h = [0u8; RECORD_HEADER];
    while pos < len {
        anyhow::ensure!(
            pos + RECORD_HEADER as u64 <= len,
            "{path}: truncated record header at byte {pos}"
        );
        f.seek(SeekFrom::Start(pos))?;
        f.read_exact(&mut h)?;
        let r = RecordRef {
            index: u32::from_le_bytes(h[0..4].try_into().unwrap()),
            route: h[4],
            n_positions: u32::from_le_bytes(h[5..9].try_into().unwrap()),
            vocab: u32::from_le_bytes(h[9..13].try_into().unwrap()),
            offset: pos + RECORD_HEADER as u64,
        };
        anyhow::ensure!(
            r.offset + r.payload_bytes() <= len,
            "{path}: record {} payload runs past the end of the file",
            r.index
        );
        pos = r.offset + r.payload_bytes();
        out.push(r);
    }
    Ok(out)
}

/// (bit-identical, max|Δ|) of two equally shaped payloads, streamed.
fn compare_payloads(
    fa: &mut std::fs::File,
    ra: &RecordRef,
    fb: &mut std::fs::File,
    rb: &RecordRef,
) -> anyhow::Result<(bool, f32)> {
    const CHUNK: usize = 1 << 20;
    fa.seek(SeekFrom::Start(ra.offset))?;
    fb.seek(SeekFrom::Start(rb.offset))?;
    let (mut ba, mut bb) = (vec![0u8; CHUNK], vec![0u8; CHUNK]);
    let mut left = ra.payload_bytes() as usize;
    let (mut same, mut max) = (true, 0.0f32);
    while left > 0 {
        let n = left.min(CHUNK);
        fa.read_exact(&mut ba[..n])?;
        fb.read_exact(&mut bb[..n])?;
        if ba[..n] != bb[..n] {
            same = false;
            for (x, y) in ba[..n].chunks_exact(4).zip(bb[..n].chunks_exact(4)) {
                let d = (f32::from_le_bytes(x.try_into().unwrap())
                    - f32::from_le_bytes(y.try_into().unwrap()))
                .abs();
                max = if d.is_nan() {
                    f32::INFINITY
                } else {
                    max.max(d)
                };
            }
        }
        left -= n;
    }
    Ok((same, max))
}

/// The record indices `logits-compare --only-indices` restricts a
/// comparison to, with the forward path they were measured on when the
/// file says so (the object `growth-eval --indices-out` writes; a plain
/// array carries no path).
#[derive(Debug, Clone, Default)]
pub struct IndexSet {
    pub indices: std::collections::BTreeSet<u32>,
    pub path: Option<String>,
    pub shell: Option<String>,
    pub growth: Option<String>,
}

/// A JSON array of integers, or an object with an `indices` array (what
/// `growth-eval --indices-out` writes, with its `path`). Empty = error.
pub fn read_indices_file(path: &str) -> anyhow::Result<IndexSet> {
    let text = std::fs::read_to_string(path).with_context(|| path.to_string())?;
    let v: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("{path}: not JSON"))?;
    let mut set = IndexSet::default();
    let arr = match &v {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(o) => {
            let field = |k: &str| o.get(k).and_then(|x| x.as_str()).map(String::from);
            set.path = field("path");
            set.shell = field("shell");
            set.growth = field("growth");
            o.get("indices")
                .and_then(|i| i.as_array())
                .ok_or_else(|| anyhow::anyhow!("{path}: object without an \"indices\" array"))?
        }
        _ => anyhow::bail!("{path}: expected a JSON array of record indices"),
    };
    for x in arr {
        let i = x
            .as_u64()
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| anyhow::anyhow!("{path}: {x} is not a record index"))?;
        set.indices.insert(i);
    }
    anyhow::ensure!(!set.indices.is_empty(), "{path}: the index set is empty");
    Ok(set)
}

/// The path rule of a comparison (see [`ForwardPath`]): two dumps that
/// both record a path must record the same one, and an index set measured
/// on a path (`growth-eval --indices-out`) accepts only dumps that record
/// that path — a dump without a sidecar is of unknown path and is refused
/// under such a set (re-dump it with `--path`). A refusal is an error, not
/// a `g2_pass: false`: the comparison is undefined, not failed.
fn check_paths(
    a: &str,
    b: &str,
    ma: &Option<DumpMeta>,
    mb: &Option<DumpMeta>,
    only: Option<&IndexSet>,
) -> anyhow::Result<()> {
    let (pa, pb) = (
        ma.as_ref().and_then(|m| m.path.as_deref()),
        mb.as_ref().and_then(|m| m.path.as_deref()),
    );
    if let (Some(x), Some(y)) = (pa, pb) {
        anyhow::ensure!(
            x == y,
            "logits-compare: {a} ran the {x} forward path, {b} the {y} one — bit-identity is \
             defined on one path (dump both with the same --path)"
        );
    }
    let Some(pi) = only.and_then(|o| o.path.as_deref()) else {
        return Ok(());
    };
    for (name, p) in [(a, pa), (b, pb)] {
        match p {
            Some(x) if x == pi => {}
            Some(x) => anyhow::bail!(
                "logits-compare: the index set was measured on the {pi} forward path, {name} \
                 ran {x} — the no-hit set of one path says nothing about the other (dump with \
                 --path {pi})"
            ),
            None => anyhow::bail!(
                "logits-compare: the index set was measured on the {pi} forward path, {name} \
                 records no path ({} missing — an older dump, or the sidecar was removed): \
                 re-dump it with --path {pi}",
                dump_meta_path(name)
            ),
        }
    }
    Ok(())
}

/// The comparison of two dumps as JSON. `b` is the candidate (F1): its
/// backbone records (route 0) must all be bit-identical to `a`'s (G2),
/// and they are compared only with `a` records that ran the backbone too
/// (route 0 — `a` may itself be a routed file after a re-bake). G2 also
/// requires the two dumps to cover the SAME prompt set completely: equal
/// record counts, no index in only one of them, no duplicate index, no
/// shape mismatch — a dump cut short at prompt 120 of 500 compares 120
/// identical records and must not pass (R3). And no prompt `a` ran on
/// the backbone may have been taken by a `key_first` lookup record in `b`
/// (`b`'s sidecar lists them, `key_first_indices`): an `answer`-mode hit
/// carries the backbone's own logits, so only this count sees that the
/// user got a table answer instead (`a_backbone_b_key_first`, review
/// KF-2 — G2 is the general set's gate).
///
/// `only`: compare just these record indices (`--only-indices`, the
/// no-hit set of `growth-eval`): every listed index must exist in BOTH
/// dumps (`only_missing_in_a/b` = 0) and the rules above apply to the
/// restricted set — a growth file (F1 = F0 + expert_append, no router)
/// passes G2 when every no-hit record is bit-identical to F0's.
pub fn compare_dumps(
    a: &str,
    b: &str,
    only_set: Option<&IndexSet>,
) -> anyhow::Result<serde_json::Value> {
    let (ma, mb) = (read_dump_meta(a)?, read_dump_meta(b)?);
    check_paths(a, b, &ma, &mb, only_set)?;
    let only = only_set.map(|o| &o.indices);
    let (ia_all, ib_all) = (index_records(a)?, index_records(b)?);
    let keep = |r: &RecordRef| only.is_none_or(|s| s.contains(&r.index));
    let ia: Vec<RecordRef> = ia_all.iter().copied().filter(keep).collect();
    let ib: Vec<RecordRef> = ib_all.iter().copied().filter(keep).collect();
    let by_a: BTreeMap<u32, RecordRef> = ia.iter().map(|r| (r.index, *r)).collect();
    let by_b: BTreeMap<u32, RecordRef> = ib.iter().map(|r| (r.index, *r)).collect();
    let only_missing_a = only.map_or(0, |s| s.iter().filter(|i| !by_a.contains_key(i)).count());
    let only_missing_b = only.map_or(0, |s| s.iter().filter(|i| !by_b.contains_key(i)).count());
    let (dup_a, dup_b) = (ia.len() - by_a.len(), ib.len() - by_b.len());
    let (mut fa, mut fb) = (std::fs::File::open(a)?, std::fs::File::open(b)?);
    let (mut matched, mut identical, mut shape_mismatch) = (0usize, 0usize, 0usize);
    let mut max_abs = 0.0f32;
    let mut per_route: BTreeMap<u8, (usize, usize, f32)> = BTreeMap::new();
    let mut differing: Vec<u32> = Vec::new();
    let mut backbone_missing = 0usize;
    let mut backbone_ref_not_backbone = 0usize;
    for (idx, rb) in &by_b {
        let Some(ra) = by_a.get(idx) else {
            backbone_missing += usize::from(rb.route == 0);
            continue;
        };
        if ra.n_positions != rb.n_positions || ra.vocab != rb.vocab {
            shape_mismatch += 1;
            differing.push(*idx);
            per_route.entry(rb.route).or_default().0 += 1;
            continue;
        }
        if rb.route == 0 && ra.route != 0 {
            // The reference ran a skill on this prompt: no backbone
            // reference exists for it.
            backbone_ref_not_backbone += 1;
        }
        matched += 1;
        let (same, d) = compare_payloads(&mut fa, ra, &mut fb, rb)?;
        identical += usize::from(same);
        max_abs = max_abs.max(d);
        let e = per_route.entry(rb.route).or_default();
        e.0 += 1;
        e.1 += usize::from(same);
        e.2 = e.2.max(d);
        if !same {
            differing.push(*idx);
        }
    }
    let only_a = by_a.keys().filter(|k| !by_b.contains_key(k)).count();
    let only_b = by_b.keys().filter(|k| !by_a.contains_key(k)).count();
    // F0 = backbone, F1 = a lookup table through key_first (review KF-2):
    // the router of F1 sent the prompt to the backbone and a strong key
    // took it — the user gets the table, not the backbone F0 ran. G2
    // (zero forgetting on the general set) fails on every such record; a
    // hit record in `answer` mode carries the backbone's logits, so the
    // bitwise comparison alone could never see it.
    let kf_set: Option<std::collections::BTreeSet<u32>> =
        mb.as_ref().and_then(|m| m.key_first.as_ref()).map(|v| v.iter().copied().collect());
    let b_key_first_known = kf_set.is_some();
    let (mut b_key_first, mut a_backbone_b_key_first) = (0usize, 0usize);
    let mut key_first_listed: Vec<u32> = Vec::new();
    if let Some(set) = &kf_set {
        for (idx, rb) in &by_b {
            if !set.contains(idx) {
                continue;
            }
            b_key_first += 1;
            if rb.route != 0 && by_a.get(idx).is_some_and(|ra| ra.route == 0) {
                a_backbone_b_key_first += 1;
                if key_first_listed.len() < 20 {
                    key_first_listed.push(*idx);
                }
            }
        }
    } else if mb.as_ref().is_some_and(|m| m.lookup_mode.is_some()) {
        eprintln!(
            "warning: logits-compare: {b} records a lookup mode but no key_first_indices (a dump \
             from an older build) — key_first takes over backbone prompts cannot be counted; \
             re-dump with this build"
        );
    }
    let (bb_n, bb_same, bb_max) = per_route.get(&0).copied().unwrap_or_default();
    let routes: serde_json::Map<String, serde_json::Value> = per_route
        .iter()
        .map(|(r, (n, s, m))| {
            (
                r.to_string(),
                serde_json::json!({"n": n, "bit_identical": s, "max_abs_diff": m}),
            )
        })
        .collect();
    Ok(serde_json::json!({
        "a": a,
        "b": b,
        "records_a": ia.len(),
        "records_b": ib.len(),
        "records_a_total": ia_all.len(),
        "records_b_total": ib_all.len(),
        "only_indices": only.map(|s| s.len()),
        "path_a": ma.as_ref().and_then(|m| m.path.clone()),
        "path_b": mb.as_ref().and_then(|m| m.path.clone()),
        "path_indices": only_set.and_then(|o| o.path.clone()),
        "growth_b": mb.as_ref().and_then(|m| m.growth.clone()),
        "shell_b": mb.as_ref().and_then(|m| m.shell.clone()),
        "only_missing_in_a": only_missing_a,
        "only_missing_in_b": only_missing_b,
        "matched": matched,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "shape_mismatch": shape_mismatch,
        "bit_identical": identical,
        "max_abs_diff": max_abs,
        "differing_indices": differing,
        "per_route_b": routes,
        "b_backbone_records": bb_n,
        "b_backbone_bit_identical": bb_same,
        "b_backbone_max_abs_diff": bb_max,
        "b_backbone_missing_in_a": backbone_missing,
        "b_backbone_reference_not_backbone": backbone_ref_not_backbone,
        "b_key_first_known": b_key_first_known,
        "b_key_first_records": b_key_first,
        "a_backbone_b_key_first": a_backbone_b_key_first,
        "a_backbone_b_key_first_indices": key_first_listed,
        "duplicate_indices_a": dup_a,
        "duplicate_indices_b": dup_b,
        // G2: both dumps cover the same prompt set completely, every
        // backbone-routed record of b is bit-identical to a's backbone one,
        // and no prompt F0 answered with the backbone was taken from the
        // backbone by key_first.
        "g2_pass": bb_n > 0
            && bb_same == bb_n
            && a_backbone_b_key_first == 0
            && backbone_missing == 0
            && backbone_ref_not_backbone == 0
            && only_a == 0
            && only_b == 0
            && only_missing_a == 0
            && only_missing_b == 0
            && shape_mismatch == 0
            && dup_a == 0
            && dup_b == 0
            && ia.len() == ib.len(),
    }))
}

pub fn cmd_logits_compare(a: &str, b: &str, only_indices: Option<&str>) -> anyhow::Result<()> {
    let only = only_indices.map(read_indices_file).transpose()?;
    let j = compare_dumps(a, b, only.as_ref())?;
    println!("{}", serde_json::to_string_pretty(&j)?);
    Ok(())
}

// ───────────────────────── skill-gate ─────────────────────────

/// `--status active` needs a gate with `"status": "measured"` (the
/// route-eval summary of a non-vacuous run): an active record whose gate
/// is not measured is not auto-routable — every `--route auto` would run
/// the backbone while the header says "active". `allow_unmeasured`
/// commits such a state anyway (debugging only).
pub fn cmd_skill_gate(
    path: &str,
    id: &str,
    gate_path: &str,
    status: &str,
    allow_unmeasured: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(status, "active" | "quarantine" | "retired"),
        "--status {status}: expected active | quarantine | retired"
    );
    let gate_bytes = std::fs::read(gate_path).with_context(|| gate_path.to_string())?;
    let gate: serde_json::Value =
        serde_json::from_slice(&gate_bytes).with_context(|| format!("{gate_path}: not JSON"))?;
    anyhow::ensure!(
        gate.is_object(),
        "{gate_path}: the gate must be a JSON object"
    );
    let model = CmfModel::open(path).with_context(|| path.to_string())?;
    let rec = model
        .header
        .skills
        .iter()
        .find(|s| s.id == id)
        .ok_or_else(|| anyhow::anyhow!("skill '{id}' not in {path}"))?;
    anyhow::ensure!(
        rec.is_v2(),
        "skill '{id}' is a v1 record (no kind): status/gate exist only on v2 records"
    );
    let from = rec.status.clone();
    let gate_status = gate.get("status").and_then(|s| s.as_str()).map(str::to_string);
    let measured = gate_status.as_deref() == Some("measured");
    if status == "active" && !measured {
        anyhow::ensure!(
            allow_unmeasured,
            "skill-gate: gate.status is {} — an active record with an unmeasured gate would \
             not auto-route (every --route auto runs the backbone while the header says \
             \"active\"); pass the route-eval summary of a non-vacuous run (it carries \
             \"status\": \"measured\"), or --allow-unmeasured to commit anyway",
            gate_status
                .as_deref()
                .map_or("absent".to_string(), |s| format!("{s:?}"))
        );
        eprintln!(
            "warning: gate.status is not \"measured\" — an active skill still will not \
             auto-route until its gate is measured (--allow-unmeasured)"
        );
    }
    // A key_first lookup record takes requests the router sent to the
    // backbone: its gate must be measured WITH that step (review KF-2) —
    // a route-eval of the φ router alone misses exactly those accepts.
    let key_first = rec.kind.as_deref() == Some(cortiq_core::knowledge::skill_kind::LOOKUP)
        && rec
            .lookup
            .as_ref()
            .is_some_and(|l| lookup::LookupPolicy::of(l) == lookup::LookupPolicy::KeyFirst);
    let covers_key_first = gate_covers_key_first(&gate, id);
    if status == "active" && key_first && !covers_key_first {
        anyhow::ensure!(
            allow_unmeasured,
            "skill-gate: '{id}' routes by key_first, and this gate was not measured with the \
             key_first step (route-eval of this build applies it to a key_first file and writes \
             \"key_first_step\": true with \"key_first_records\" [\"{id}\"]; --router-only and \
             older summaries do not) — its false-accept bound does not cover the requests \
             key_first takes; re-run route-eval, or --allow-unmeasured to commit anyway"
        );
        eprintln!(
            "warning: '{id}' is key_first and the gate does not cover the key_first step \
             (--allow-unmeasured)"
        );
    }
    if status == "active" && gate.get("gate").and_then(|g| g.get("pass")) == Some(&serde_json::json!(false))
    {
        eprintln!(
            "warning: the gate file says gate.pass = false ({}) — activating a record that \
             failed its gate",
            gate["gate"]["reason"]
        );
    }
    let hash_before = router::skills_hash(&model.header);
    let seq = model.header.lineage.last().map_or(0, |e| e.seq + 1);
    drop(model);
    let event = LineageEvent::now(
        seq,
        if status == "retired" {
            "skill_retired"
        } else {
            "skill_gated"
        },
        serde_json::json!({
            "id": id,
            "status_from": from,
            "status_to": status,
            "gate_status": gate.get("status"),
            "gate_sha256": sha256_hex(&gate_bytes),
        }),
    );
    let (id_s, status_s) = (id.to_string(), status.to_string());
    let report = CmfModel::update_header_append(path, move |h| {
        if let Some(s) = h.skills.iter_mut().find(|s| s.id == id_s) {
            s.gate = Some(gate);
            s.status = Some(status_s);
        }
        h.lineage.push(event);
    })
    .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    let after = CmfModel::open(path)?;
    let stale = router::skills_hash(&after.header) != hash_before;
    let j = serde_json::json!({
        "file": path,
        "skill": id,
        "status_from": from,
        "status_to": status,
        "gate_measured": measured,
        "key_first": key_first,
        "gate_covers_key_first": covers_key_first,
        "auto_routable": after.header.skills.iter().find(|s| s.id == id).is_some_and(|s| s.is_auto_routable()),
        "lineage_seq": seq,
        "old_len": report.old_len,
        "new_len": report.new_len,
        "calibration_stale": stale,
    });
    println!("{}", serde_json::to_string_pretty(&j)?);
    if stale {
        eprintln!(
            "note: the calibrated class set changed (skills_hash) — the router runs the \
             backbone until it is recalibrated"
        );
    }
    Ok(())
}

// ───────────────────────── genome files are append-only ─────────────────

/// Refuse a FULL rewrite of a genome file (`header.genome`, bit GENOME) by
/// a CLI tool: a rewrite lays the prefix out afresh (G1's byte identity
/// with F0 is gone, the segment table is dropped) and may slip in what
/// the trunk executes with (a mask catalog — `skill add --sparse`). Skills
/// enter a genome only by the tail append (`CmfModel::append_skill`, the
/// trainer's `skill-bake`); a changed trunk is a NEW genome (NF-1).
pub fn refuse_genome_rewrite(model: &CmfModel, what: &str) -> anyhow::Result<()> {
    if let Some(g) = &model.header.genome {
        anyhow::bail!(
            "{what}: {} is a frozen genome ('{}' gen {}) — this command rewrites the whole file, \
             which a genome never takes (the prefix and trunk must stay byte-identical; a skill \
             enters only by the append-only path `cortiq-embryo skill-bake` / \
             CmfModel::append_skill, a changed trunk is a new genome)",
            model.path.display(),
            g.id,
            g.generation
        );
    }
    Ok(())
}

// ───────────────────────── genome-verify ─────────────────────────

/// First differing byte offset of `f1[128..len(f0))` against `f0[128..)`
/// (None = equal; `Some(len(f0))` when f1 is shorter).
fn prefix_first_diff(f0: &str, f1: &str) -> anyhow::Result<Option<u64>> {
    const CHUNK: usize = 1 << 20;
    let (mut a, mut b) = (std::fs::File::open(f0)?, std::fs::File::open(f1)?);
    let (la, lb) = (a.metadata()?.len(), b.metadata()?.len());
    if lb < la {
        return Ok(Some(lb.min(la)));
    }
    let start = 128u64.min(la);
    a.seek(SeekFrom::Start(start))?;
    b.seek(SeekFrom::Start(start))?;
    let (mut ba, mut bb) = (vec![0u8; CHUNK], vec![0u8; CHUNK]);
    let mut pos = start;
    while pos < la {
        let n = ((la - pos) as usize).min(CHUNK);
        a.read_exact(&mut ba[..n])?;
        b.read_exact(&mut bb[..n])?;
        if ba[..n] != bb[..n] {
            let i = ba[..n]
                .iter()
                .zip(&bb[..n])
                .position(|(x, y)| x != y)
                .unwrap();
            return Ok(Some(pos + i as u64));
        }
        pos += n as u64;
    }
    Ok(None)
}

pub fn genome_verify(f0: &str, f1: &str) -> anyhow::Result<serde_json::Value> {
    let (m0, m1) = (
        CmfModel::open(f0).with_context(|| f0.to_string())?,
        CmfModel::open(f1).with_context(|| f1.to_string())?,
    );
    let (h0, h1) = (m0.trunk_hash(), m1.trunk_hash());
    let same_entry = |a: &cortiq_core::format::TensorEntry| {
        m1.tensor(&a.name).is_some_and(|b| {
            b.dtype == a.dtype
                && b.shape == a.shape
                && b.off == a.off
                && b.nbytes == a.nbytes
                && b.hash == a.hash
        })
    };
    let trunk: Vec<_> = m0
        .tensors
        .iter()
        .filter(|t| is_trunk_tensor(&t.name))
        .collect();
    let trunk_equal = trunk.iter().filter(|t| same_entry(t)).count();
    let all_equal = m0.tensors.iter().all(same_entry);
    let trunk_f1 = m1
        .tensors
        .iter()
        .filter(|t| is_trunk_tensor(&t.name))
        .count();
    let diff = prefix_first_diff(f0, f1)?;
    let equal = h0 == h1;
    let pass = equal
        && trunk_equal == trunk.len()
        && trunk_f1 == trunk.len()
        && all_equal
        && diff.is_none();
    Ok(serde_json::json!({
        "f0": f0,
        "f1": f1,
        "trunk_hash_f0": hex64(h0),
        "trunk_hash_f1": hex64(h1),
        "equal": equal,
        "genome_f0": m0.header.genome.as_ref().map(|g| &g.trunk_hash),
        "genome_f1": m1.header.genome.as_ref().map(|g| &g.trunk_hash),
        "trunk_entries": trunk.len(),
        "trunk_entries_f1": trunk_f1,
        "trunk_entries_equal": trunk_equal,
        "entries_f0": m0.tensors.len(),
        "entries_f1": m1.tensors.len(),
        "all_entries_equal": all_equal,
        "len_f0": std::fs::metadata(f0)?.len(),
        "len_f1": std::fs::metadata(f1)?.len(),
        "prefix_bytes_equal": diff.is_none(),
        "prefix_first_diff": diff,
        "pass": pass,
    }))
}

pub fn cmd_genome_verify(f0: &str, f1: &str) -> anyhow::Result<()> {
    let j = genome_verify(f0, f1)?;
    println!("{}", serde_json::to_string_pretty(&j)?);
    anyhow::ensure!(
        j["pass"] == true,
        "genome-verify: F1 is not an append-only successor of F0 (see the JSON)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clopper_pearson_upper_bounds_match_the_reference_values() {
        for (k, n, want) in [
            (0usize, 500usize, 0.0074f64),
            (1, 500, 0.0111),
            (0, 100, 0.0362),
        ] {
            let got = router::clopper_pearson_upper(k, n, 0.95);
            assert!((got - want).abs() < 5e-5, "{k}/{n}: {got} vs {want}");
        }
        // exact: 0/500 closed form 1 − 0.025^(1/500)
        let exact = 1.0 - 0.025f64.powf(1.0 / 500.0);
        assert!((router::clopper_pearson_upper(0, 500, 0.95) - exact).abs() < 1e-12);
        // monotone in k
        let u: Vec<f64> = (0..6)
            .map(|k| router::clopper_pearson_upper(k, 500, 0.95))
            .collect();
        assert!(u.windows(2).all(|w| w[0] < w[1]), "{u:?}");
    }

    fn row(lang: &str, src: &str) -> EvalRow {
        EvalRow {
            prompt: "x".into(),
            lang: lang.into(),
            src: src.into(),
        }
    }

    #[test]
    fn route_eval_summary_counts_breakdowns_and_bounds() {
        let skill = |id: &str| RouteDecision::forced(RouteTarget::Skill(id.into()), "t");
        let bb = || RouteDecision::forced(RouteTarget::Backbone, "t");
        let mut d = Vec::new();
        for i in 0..498 {
            d.push((row(if i % 2 == 0 { "en" } else { "ru" }, "general"), bb()));
        }
        d.push((row("en", "audit"), skill("herbs")));
        d.push((row("ru", "audit"), bb()));
        let s = route_eval_summary("backbone", &d, None);
        assert_eq!(s["n"], 500);
        assert_eq!(s["accepted"], 499);
        assert_eq!(s["errors"], 1);
        let up = s["false_accept_upper95"].as_f64().unwrap();
        assert!((up - 0.0111).abs() < 5e-5, "{up}");
        assert_eq!(s["gate"]["pass"], true);
        assert_eq!(s["route_counts"]["herbs"], 1);
        assert_eq!(s["per_src"]["audit"]["n"], 2);
        assert_eq!(s["per_src"]["audit"]["errors"], 1);
        assert_eq!(s["per_lang"]["en"]["errors"], 1);
        assert_eq!(s["per_lang"]["ru"]["errors"], 0);
        // recall mode
        let d: Vec<_> = (0..10)
            .map(|i| (row("ru", "dev"), if i < 9 { skill("herbs") } else { bb() }))
            .collect();
        let s = route_eval_summary("herbs", &d, None);
        assert_eq!(s["in_scope_recall"], 0.9);
        assert_eq!(s["gate"]["pass"], true);
        assert!(s["false_accept"].is_null());
    }

    fn dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("cmf-knowledge-cli-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Review KF-2: a key_first take is an accept of the record in G3, a
    /// gate covers key_first only when route-eval applied the step, and G2
    /// fails on a prompt F0 answered with the backbone that key_first
    /// took — even though the `answer`-mode record is bit-identical.
    #[test]
    fn key_first_takes_count_in_g3_and_fail_g2() {
        let skill = |id: &str| RouteDecision::forced(RouteTarget::Skill(id.into()), "t");
        let bb = || RouteDecision::forced(RouteTarget::Backbone, "t");
        let mut d = Vec::new();
        let mut by = Vec::new();
        for _ in 0..497 {
            d.push((row("en", "general"), bb()));
            by.push(None);
        }
        d.push((row("en", "general"), skill("herbs")));
        by.push(Some(lookup::DecidedBy::KeyFirst));
        d.push((row("en", "general"), skill("herbs")));
        by.push(Some(lookup::DecidedBy::KeyFirst));
        d.push((row("en", "general"), skill("herbs")));
        by.push(Some(lookup::DecidedBy::Router));
        let s = route_eval_summary_by("backbone", &d, &by, None);
        assert_eq!((s["n"].as_u64(), s["errors"].as_u64()), (Some(500), Some(3)), "{s}");
        assert_eq!(s["decided_by_counts"], serde_json::json!({"key_first": 2, "router": 1}));
        assert_eq!(s["key_first_accepts"], 2);
        assert_eq!(route_eval_summary("backbone", &d, None)["key_first_accepts"], 0);
        // The gate file.
        let mut g = s.clone();
        g["key_first_step"] = serde_json::json!(true);
        g["key_first_records"] = serde_json::json!(["herbs"]);
        assert!(gate_covers_key_first(&g, "herbs"));
        assert!(!gate_covers_key_first(&g, "other"));
        g["key_first_step"] = serde_json::json!(false);
        assert!(!gate_covers_key_first(&g, "herbs"), "--router-only");
        g["key_first_step"] = serde_json::json!(true);
        g["status"] = serde_json::json!("vacuous");
        assert!(!gate_covers_key_first(&g, "herbs"));
        assert!(!gate_covers_key_first(&serde_json::json!({"status": "measured"}), "herbs"));

        // G2: F0 ran the backbone on records 0..3; F1's record 1 is a
        // key_first take in answer mode (skill code, the backbone's own
        // logits — bit-identical), record 2 a router pick.
        let dd = dir("kf-g2");
        let rows = |seed: f32| -> Vec<Vec<f32>> { (0..2).map(|p| (0..5).map(|v| seed + (p * 5 + v) as f32).collect()).collect() };
        let (a, b) = (dd.join("a.bin"), dd.join("b.bin"));
        let mut fa = std::fs::File::create(&a).unwrap();
        let mut fb = std::fs::File::create(&b).unwrap();
        for i in 0..3u32 {
            write_record(&mut fa, i, 0, &rows(i as f32)).unwrap();
            write_record(&mut fb, i, if i == 0 { 0 } else { 1 }, &rows(i as f32)).unwrap();
        }
        drop((fa, fb));
        let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());
        let meta = |kf: serde_json::Value| {
            std::fs::write(
                dump_meta_path(b),
                serde_json::json!({"path": "per-op", "lookup_mode": "answer", "key_first_indices": kf}).to_string(),
            )
            .unwrap();
            std::fs::write(dump_meta_path(a), serde_json::json!({"path": "per-op"}).to_string()).unwrap();
        };
        meta(serde_json::json!([1]));
        let j = compare_dumps(a, b, None).unwrap();
        assert_eq!(j["bit_identical"], 3, "{j}");
        assert_eq!(j["b_backbone_bit_identical"], 1);
        assert_eq!((j["b_key_first_known"].as_bool(), j["b_key_first_records"].as_u64()), (Some(true), Some(1)));
        assert_eq!(j["a_backbone_b_key_first"], 1);
        assert_eq!(j["a_backbone_b_key_first_indices"], serde_json::json!([1]));
        assert_eq!(j["g2_pass"], false, "F0's backbone prompt answered from a table: {j}");
        meta(serde_json::json!([]));
        let j = compare_dumps(a, b, None).unwrap();
        assert_eq!((j["a_backbone_b_key_first"].as_u64(), j["g2_pass"].as_bool()), (Some(0), Some(true)), "{j}");
        let _ = std::fs::remove_dir_all(&dd);
    }

    #[test]
    fn dump_records_round_trip_and_compare() {
        let d = dir("dump");
        let rows = |seed: f32, n: usize| -> Vec<Vec<f32>> {
            (0..n)
                .map(|p| (0..7).map(|v| seed + p as f32 * 0.5 + v as f32).collect())
                .collect()
        };
        let (a, b) = (d.join("a.bin"), d.join("b.bin"));
        let mut fa = std::fs::File::create(&a).unwrap();
        write_record(&mut fa, 0, 0, &rows(1.0, 3)).unwrap();
        write_record(&mut fa, 1, 0, &rows(2.0, 3)).unwrap();
        write_record(&mut fa, 2, 0, &rows(3.0, 3)).unwrap();
        drop(fa);
        let mut fb = std::fs::File::create(&b).unwrap();
        write_record(&mut fb, 0, 0, &rows(1.0, 3)).unwrap(); // identical backbone
        let mut changed = rows(2.0, 3);
        changed[2][4] += 0.25;
        write_record(&mut fb, 1, 1, &changed).unwrap(); // skill-routed, differs
        write_record(&mut fb, 2, 0, &rows(3.0, 3)).unwrap(); // identical backbone
        write_record(&mut fb, 7, 0, &rows(9.0, 2)).unwrap(); // only in b
        drop(fb);

        let ia = index_records(a.to_str().unwrap()).unwrap();
        assert_eq!(ia.len(), 3);
        assert_eq!(
            (ia[1].index, ia[1].route, ia[1].n_positions, ia[1].vocab),
            (1, 0, 3, 7)
        );
        assert_eq!(
            ia[1].offset,
            (RECORD_HEADER + 3 * 7 * 4 + RECORD_HEADER) as u64
        );

        let j = compare_dumps(a.to_str().unwrap(), b.to_str().unwrap(), None).unwrap();
        assert_eq!(j["matched"], 3);
        assert_eq!(j["bit_identical"], 2);
        assert!(
            (j["max_abs_diff"].as_f64().unwrap() - 0.25).abs() < 1e-6,
            "{j}"
        );
        assert_eq!(j["only_in_b"], 1);
        assert_eq!(j["b_backbone_records"], 2);
        assert_eq!(j["b_backbone_bit_identical"], 2);
        assert_eq!(j["b_backbone_missing_in_a"], 1, "{j}");
        assert_eq!(
            j["g2_pass"], false,
            "a backbone record of b has no reference"
        );
        assert_eq!(j["differing_indices"], serde_json::json!([1]));

        // A truncated file is refused, not silently shortened.
        let t = d.join("t.bin");
        let bytes = std::fs::read(&a).unwrap();
        std::fs::write(&t, &bytes[..bytes.len() - 3]).unwrap();
        assert!(index_records(t.to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// R2: a measurement in which no skill could win is vacuous — the gate
    /// fails whatever the counts; the backbone gate also needs n ≥ 500
    /// (0 errors in 183 prompts already bound CP ≤ 0.02).
    #[test]
    fn route_eval_gate_refuses_vacuous_and_small_measurements() {
        let bb = || RouteDecision::forced(RouteTarget::Backbone, "t");
        let rows = |n: usize| -> Vec<(EvalRow, RouteDecision)> {
            (0..n).map(|_| (row("en", "general"), bb())).collect()
        };
        // 500/500 in the backbone, but the skill was in quarantine.
        let s = route_eval_summary("backbone", &rows(500), Some("no routable skill class"));
        assert_eq!(s["false_accept"], 0.0);
        assert!(s["false_accept_upper95"].as_f64().unwrap() < 0.0075);
        assert_eq!(s["gate"]["pass"], false, "{s}");
        assert_eq!(s["gate"]["vacuous"], true);
        assert!(s["gate"]["reason"].as_str().unwrap().contains("no routable"));
        // The same counts with a candidate skill: pass.
        let s = route_eval_summary("backbone", &rows(500), None);
        assert_eq!(s["gate"]["pass"], true, "{s}");
        // 0/183 bounds CP ≤ 0.02, and still fails: n < 500.
        let s = route_eval_summary("backbone", &rows(183), None);
        assert!(s["false_accept_upper95"].as_f64().unwrap() <= 0.02);
        assert_eq!(s["gate"]["n_ok"], false);
        assert_eq!(s["gate"]["pass"], false, "{s}");
        // Recall: vacuous fails too.
        let herbs: Vec<_> = (0..10)
            .map(|_| {
                (
                    row("ru", "dev"),
                    RouteDecision::forced(RouteTarget::Skill("herbs".into()), "t"),
                )
            })
            .collect();
        assert_eq!(route_eval_summary("herbs", &herbs, None)["gate"]["pass"], true);
        assert_eq!(
            route_eval_summary("herbs", &herbs, Some("stale"))["gate"]["pass"],
            false
        );
    }

    /// R8: the backbone is the TARGET, not a label — a (legacy) skill named
    /// "backbone" winning a general prompt is a false accept.
    #[test]
    fn a_skill_named_backbone_is_not_the_backbone() {
        let mut d = Vec::new();
        for _ in 0..499 {
            d.push((row("en", "general"), RouteDecision::forced(RouteTarget::Backbone, "t")));
        }
        d.push((
            row("en", "general"),
            RouteDecision::forced(RouteTarget::Skill("backbone".into()), "t"),
        ));
        let s = route_eval_summary("backbone", &d, None);
        assert_eq!(s["errors"], 1, "{s}");
        assert_eq!(s["accepted"], 499);
    }

    /// R3: G2 needs both dumps to cover the same prompt set completely,
    /// and a backbone record of b is compared only with a backbone record
    /// of a.
    #[test]
    fn g2_refuses_partial_dumps_and_non_backbone_references() {
        let d = dir("g2strict");
        let rows = |seed: f32| -> Vec<Vec<f32>> {
            (0..3)
                .map(|p| (0..5).map(|v| seed + p as f32 + v as f32 * 0.5).collect())
                .collect()
        };
        let write = |name: &str, recs: &[(u32, u8)]| -> String {
            let p = d.join(name);
            let mut f = std::fs::File::create(&p).unwrap();
            for &(i, r) in recs {
                write_record(&mut f, i, r, &rows(i as f32)).unwrap();
            }
            p.to_str().unwrap().to_string()
        };
        let full: Vec<(u32, u8)> = (0..6).map(|i| (i, 0)).collect();
        let a = write("a.bin", &full);
        // The complete twin passes.
        let b = write("b.bin", &full);
        assert_eq!(compare_dumps(&a, &b, None).unwrap()["g2_pass"], true);
        // b stopped after 2 of 6 prompts: its 2 records are bit-identical,
        // and G2 still fails.
        let short = write("short.bin", &full[..2]);
        let j = compare_dumps(&a, &short, None).unwrap();
        assert_eq!(j["b_backbone_bit_identical"], 2);
        assert_eq!(j["only_in_a"], 4);
        assert_eq!(j["g2_pass"], false, "{j}");
        // a ran a skill on prompt 3, b the backbone: no backbone reference.
        let mut a_routed = full.clone();
        a_routed[3].1 = 1;
        let ar = write("a_routed.bin", &a_routed);
        let j = compare_dumps(&ar, &b, None).unwrap();
        assert_eq!(j["b_backbone_reference_not_backbone"], 1, "{j}");
        assert_eq!(j["g2_pass"], false);
        // A duplicated index is not a complete dump either.
        let mut dup = full.clone();
        dup.push((2, 0));
        let dp = write("dup.bin", &dup);
        let j = compare_dumps(&a, &dp, None).unwrap();
        assert_eq!(j["duplicate_indices_b"], 1, "{j}");
        assert_eq!(j["g2_pass"], false);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// R3: a failed dump leaves no file behind (no partial dump posing as
    /// a complete one), a committed one appears whole.
    #[test]
    fn a_partial_dump_is_removed_on_failure() {
        let d = dir("partial");
        let out = d.join("x.bin");
        std::fs::write(&out, b"stale complete dump").unwrap();
        {
            let _ = std::fs::remove_file(&out);
            let p = PartialOut::new(out.to_str().unwrap());
            std::fs::write(&p.tmp, b"half").unwrap();
            // error path: dropped without commit
        }
        assert!(!out.exists());
        assert!(!d.join("x.bin.partial").exists());
        let p = PartialOut::new(out.to_str().unwrap());
        std::fs::write(&p.tmp, b"whole").unwrap();
        p.commit().unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"whole");
        assert!(!d.join("x.bin.partial").exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
