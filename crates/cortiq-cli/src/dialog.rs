//! `cortiq probe-dialog` — the conversation memory of a `lookup` record
//! (CMF_V2_SPEC §9.5.2) measured on scripted dialogs of USER turns.
//!
//! Every turn is decided exactly as `serve`'s lookup pre-pass decides a
//! chat request (`SkillRouter::decide_lookup_turns`): the φ router on the
//! turn's own text, the chosen record's prompt contract against the
//! tokenizer's chat frame (`router::chat_frame`), then
//! `lookup::resolve_lookup_gated` over the user turns of the window the
//! chat endpoint passes — the turn itself first, then up to
//! `lookup::MEMORY_TURNS - 1` earlier ones, most recent first
//! (`cortiq_server::route::recent_user_texts`); the key comes from the
//! most recent turn that holds one (`lookup_turn` = how many turns back),
//! the field and the language from the turn itself; a `key_first` record
//! takes a backbone decision only on a strong key of the turn itself.
//!
//! Per turn: the lane that ran, the decided target and who decided it,
//! the hit, `lookup_turn`, the entry matched and its Latin binomial, the
//! expected entry (the `expect_src` binomial as an EXACT key of the table;
//! a text without one falls back to the runtime's key extraction) and
//! whether they agree, the field, the answer. `answer` mode generates
//! nothing; `context` mode generates at most [`CONTEXT_MAX_TOKENS`]
//! greedy tokens on the chat transcript serve would render — the earlier
//! user turns with the answers this tool produced, the card prepended to
//! the last user message; `off` decides only.
//!
//! Summary slices: `per_lang` (= `per_turn_lang`) by the script of EACH
//! turn — the language its answer is chosen in — and `per_dialog_lang` by
//! the dialog's language; `--route auto` on a file without ROUTER_V2 is
//! reported in `warnings` (serve never routes such a file).
//!
//! Input: JSONL, one dialog per line —
//! `{"src": "<latin>", "turns": ["…", "…"], "expect_src": "<latin>"}`;
//! `expect_src` is the plant every turn should resolve to, or an array
//! with one entry per turn (`null` = not checked) for a dialog that
//! switches plant; missing = `src`. Optional `lang` (default: the script
//! of the first turn) and `id`.

use crate::knowledge::{Lanes, RouteMode};
use anyhow::Context;
use cortiq_core::{CmfModel, key_hash};
use cortiq_engine::SamplerConfig;
use cortiq_engine::lookup::{self, LookupMode, LookupOutcome, LookupTable};
use cortiq_engine::router::{self, PromptFrame, RouteOptions, RouteTarget};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Generation budget of `context` mode (a larger `--max-tokens` is capped).
pub const CONTEXT_MAX_TOKENS: usize = 32;

/// Misses listed in the summary (all are counted).
pub const MISSES_LISTED: usize = 20;

pub struct DialogArgs<'a> {
    pub model: &'a str,
    pub dialogs_jsonl: &'a str,
    /// `--route auto|backbone|<id>` (unset: `auto` on a ROUTER_V2 file).
    pub route: Option<String>,
    /// `--lookup-mode answer|context|off` (unset: `CMF_LOOKUP_MODE`, else
    /// `answer`).
    pub lookup_mode: Option<String>,
    pub include_quarantine: bool,
    /// Tokens generated per turn in `context` mode (≤ [`CONTEXT_MAX_TOKENS`]).
    pub max_tokens: usize,
    pub json: bool,
}

/// One scripted conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dialog {
    /// `id`, else `src`, else `line N`.
    pub id: String,
    pub src: Option<String>,
    /// `lang`, else `ru` when the first turn has Cyrillic, else `en`.
    pub lang: String,
    pub turns: Vec<String>,
    /// The plant each turn should resolve to (`None` = not checked).
    pub expect: Vec<Option<String>>,
}

fn opt_string(v: Option<&Value>, what: &str, at: &str) -> anyhow::Result<Option<String>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => anyhow::bail!("{at}: \"{what}\" must be a string, got {other}"),
    }
}

/// The dialog file: one JSON object per line (blank lines and `#`
/// comments skipped).
pub fn parse_dialogs_jsonl(text: &str) -> anyhow::Result<Vec<Dialog>> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = format!("dialogs jsonl line {}", ln + 1);
        let v: Value = serde_json::from_str(line).with_context(|| at.clone())?;
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{at}: not a JSON object"))?;
        let turns: Vec<String> = match obj.get("turns") {
            Some(Value::Array(a)) if !a.is_empty() => a
                .iter()
                .map(|t| {
                    t.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| anyhow::anyhow!("{at}: every turn must be a string"))
                })
                .collect::<anyhow::Result<_>>()?,
            _ => anyhow::bail!("{at}: \"turns\" must be a non-empty array of strings"),
        };
        let src = opt_string(obj.get("src"), "src", &at)?;
        let expect = match obj.get("expect_src") {
            None | Some(Value::Null) => vec![src.clone(); turns.len()],
            Some(Value::Array(a)) => {
                anyhow::ensure!(
                    a.len() == turns.len(),
                    "{at}: \"expect_src\" has {} entries for {} turns",
                    a.len(),
                    turns.len()
                );
                a.iter()
                    .map(|e| opt_string(Some(e), "expect_src[]", &at))
                    .collect::<anyhow::Result<_>>()?
            }
            e => vec![opt_string(e, "expect_src", &at)?; turns.len()],
        };
        let script = if lookup::has_cyrillic(&turns[0]) { "ru" } else { "en" };
        let lang = opt_string(obj.get("lang"), "lang", &at)?.unwrap_or_else(|| script.to_string());
        let id = opt_string(obj.get("id"), "id", &at)?
            .or_else(|| src.clone())
            .unwrap_or_else(|| format!("line {}", ln + 1));
        out.push(Dialog {
            id,
            src,
            lang,
            turns,
            expect,
        });
    }
    anyhow::ensure!(!out.is_empty(), "dialog set is empty");
    Ok(out)
}

/// The user turns `serve` hands the lookup pre-pass at turn `t`: the
/// turn itself first, then the earlier ones back in time, at most
/// `lookup::MEMORY_TURNS` in all.
pub fn turn_window(turns: &[String], t: usize) -> Vec<String> {
    cortiq_server::route::recent_user_texts(
        turns[..=t].iter().map(|u| ("user", u.clone())),
        lookup::MEMORY_TURNS,
    )
}

/// The entry `expect` names in `table`: the whole text as an EXACT key
/// (a Latin binomial is one), else the runtime's key extraction over it.
pub fn expect_entry(table: &LookupTable, expect: &str) -> Option<(u32, &'static str)> {
    if let Some(e) = table.find_hash(key_hash(expect)) {
        return Some((e, "exact"));
    }
    table.find_key(expect).map(|k| (k.entry, "extract"))
}

/// The Latin binomial of `entry` as its cards spell it: the first
/// capitalised `Genus species` pair of any slot (card, then fields) that
/// is a key of that very entry.
pub fn entry_binomial(table: &LookupTable, entry: u32) -> Option<String> {
    for lang in 0..table.langs().len() {
        let Ok(card) = table.card(entry, lang) else {
            continue;
        };
        for text in std::iter::once(&card.card).chain(card.fields.values()) {
            for pair in lookup::capitalised_binomials(text) {
                if table.find_hash(key_hash(&pair)) == Some(entry) {
                    return Some(pair);
                }
            }
        }
    }
    None
}

/// The script of one turn — the language its answer is chosen in
/// (`lookup::pick_lang`): `ru` for Cyrillic, else `en`.
pub fn turn_script(text: &str) -> &'static str {
    if lookup::has_cyrillic(text) { "ru" } else { "en" }
}

fn frame_label(f: PromptFrame) -> &'static str {
    match f {
        PromptFrame::Raw => "raw",
        PromptFrame::CmfImV1 => "cmf-im-v1",
        PromptFrame::Other => "other",
    }
}

fn rate(n: usize, d: usize) -> Value {
    if d == 0 { Value::Null } else { json!(n as f64 / d as f64) }
}

fn mean(sum: usize, n: usize) -> Value {
    rate(sum, n)
}

/// Counters of one slice (all dialogs, or one language).
#[derive(Default)]
struct Tally {
    dialogs: usize,
    turns: usize,
    hits: usize,
    hit_turn_sum: usize,
    first: usize,
    first_hits: usize,
    first_checked: usize,
    first_correct: usize,
    followups: usize,
    followup_hits: usize,
    followup_memory_hits: usize,
    followup_hit_turn_sum: usize,
    /// Follow-up turns whose expectation resolved to an entry.
    followup_expected: usize,
    /// Follow-up hits whose expectation resolved to an entry.
    followup_checked: usize,
    followup_correct: usize,
}

impl Tally {
    fn add(&mut self, t: usize, hit: Option<usize>, expected: bool, correct: Option<bool>) {
        self.turns += 1;
        if let Some(back) = hit {
            self.hits += 1;
            self.hit_turn_sum += back;
        }
        if t == 0 {
            self.first += 1;
            self.first_hits += hit.is_some() as usize;
            self.first_checked += correct.is_some() as usize;
            self.first_correct += (correct == Some(true)) as usize;
            return;
        }
        self.followups += 1;
        self.followup_expected += expected as usize;
        if let Some(back) = hit {
            self.followup_hits += 1;
            self.followup_memory_hits += (back > 0) as usize;
            self.followup_hit_turn_sum += back;
        }
        self.followup_checked += correct.is_some() as usize;
        self.followup_correct += (correct == Some(true)) as usize;
    }

    fn json(&self) -> Value {
        json!({
            "dialogs": self.dialogs,
            "turns": self.turns,
            "hits": self.hits,
            "first_turns": self.first,
            "first_turn_hits": self.first_hits,
            "first_turn_hit_rate": rate(self.first_hits, self.first),
            "first_turn_checked": self.first_checked,
            "first_turn_correct": self.first_correct,
            "first_turn_correct_rate": rate(self.first_correct, self.first_checked),
            "followup_turns": self.followups,
            "followup_hits": self.followup_hits,
            "followup_hit_rate": rate(self.followup_hits, self.followups),
            // hits whose key came from an EARLIER turn (lookup_turn ≥ 1)
            "followup_memory_hits": self.followup_memory_hits,
            // hits answered from the expected entry / hits with a resolved expectation
            "followup_checked": self.followup_checked,
            "followup_correct": self.followup_correct,
            "followup_correct_rate": rate(self.followup_correct, self.followup_checked),
            // end to end: correct hits / follow-up turns with a resolved expectation
            "followup_expected": self.followup_expected,
            "followup_correct_of_expected": rate(self.followup_correct, self.followup_expected),
            "mean_lookup_turn": mean(self.hit_turn_sum, self.hits),
            "mean_lookup_turn_followup": mean(self.followup_hit_turn_sum, self.followup_hits),
        })
    }
}

fn clip(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

pub fn cmd_probe_dialog(a: DialogArgs<'_>) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(a.dialogs_jsonl)
        .with_context(|| a.dialogs_jsonl.to_string())?;
    let dialogs = parse_dialogs_jsonl(&text)?;
    let lookup_mode = LookupMode::resolve(a.lookup_mode.as_deref()).map_err(anyhow::Error::msg)?;
    let max_tokens = a.max_tokens.clamp(1, CONTEXT_MAX_TOKENS);
    let backend = format!(
        "CMF_GPU={} resident={}",
        std::env::var("CMF_GPU").unwrap_or_else(|_| "unset".into()),
        std::env::var("CMF_EMBRYO_RESIDENT").unwrap_or_else(|_| "unset".into())
    );
    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    let mode = RouteMode::resolve(a.route.as_deref(), &model)?;
    // serve routes (and runs its lookup pre-pass) only on a ROUTER_V2
    // file: `--route auto` on any other file measures a decision serve
    // never makes (review KF-7).
    let mut warnings: Vec<String> = Vec::new();
    if mode == RouteMode::Auto && !router::is_router_v2(&model) {
        warnings.push(format!(
            "--route auto on {} which declares no router policy (ROUTER_V2): serve does not route \
             this file and never runs its lookup pre-pass — these decisions are not serve's",
            a.model
        ));
    }
    for w in &warnings {
        eprintln!("warning: probe-dialog: {w}");
    }
    let mut lanes = Lanes::new(
        model.clone(),
        SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            min_p: 0.0,
            seed: Some(0),
            ..Default::default()
        },
        RouteOptions {
            include_quarantine: a.include_quarantine,
        },
    )
    .lookup_mode(lookup_mode);
    let lookup_ids = lanes.lookup_ids();
    let mut policies = BTreeMap::new();
    for id in &lookup_ids {
        if let Some(t) = lanes.lookup_table(id)? {
            policies.insert(id.clone(), t.policy().label());
        }
    }
    // serve routes a chat under its tokenizer's chat frame.
    let frame = router::chat_frame(&lanes.lane(&RouteTarget::Backbone)?.tokenizer);
    if !a.json {
        println!(
            "probe-dialog: {} | {} dialogs, {} turns | route={} | lookup={} | frame={} | memory {} turns | lookup records {:?} | {backend}",
            a.model,
            dialogs.len(),
            dialogs.iter().map(|d| d.turns.len()).sum::<usize>(),
            mode.label(),
            lookup_mode.label(),
            frame_label(frame),
            lookup::MEMORY_TURNS,
            policies
        );
        if lookup_ids.is_empty() {
            println!("   (the file carries no lookup record: every turn runs the backbone)");
        }
    }

    let mut all = Tally::default();
    // By the dialog's language (`lang`, else the script of its first turn)
    // and by the script of EACH turn (review KF-7: a Russian dialog with
    // English follow-ups — the answer language is chosen per turn — is
    // one `ru` dialog but two slices of turns).
    let mut per_dialog_lang: BTreeMap<String, Tally> = BTreeMap::new();
    let mut per_turn_lang: BTreeMap<String, Tally> = BTreeMap::new();
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut route_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut decided_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut decided_by_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut field_counts: BTreeMap<String, usize> = BTreeMap::new();
    let (mut expect_unresolved, mut misses_total) = (0usize, 0usize);
    let mut misses = Vec::new();
    let mut per_dialog = Vec::new();

    for (di, d) in dialogs.iter().enumerate() {
        all.dialogs += 1;
        per_dialog_lang.entry(d.lang.clone()).or_default().dialogs += 1;
        let mut turn_langs_seen: Vec<&'static str> = Vec::new();
        // (user text, the answer this tool produced) of the earlier turns:
        // the transcript `context` mode renders, as a client resends it.
        let mut history: Vec<(String, String)> = Vec::new();
        let mut rows = Vec::new();
        for (t, text) in d.turns.iter().enumerate() {
            let window = turn_window(&d.turns, t);
            let wrefs: Vec<&str> = window.iter().map(String::as_str).collect();
            let (decision, outcome) = lanes.decide_lookup_turns(&mode, &wrefs, frame)?;
            let target = decision.target_label().to_string();
            let decided = outcome.lookup_id().unwrap_or(target.as_str()).to_string();
            *route_counts.entry(target.clone()).or_default() += 1;
            *decided_counts.entry(decided.clone()).or_default() += 1;
            if let Some(by) = outcome.decided_by() {
                *decided_by_counts.entry(by.label().to_string()).or_default() += 1;
            }
            let hit = outcome.hit();
            // The table the turn is checked against: the one that answered,
            // else the one decided on, else the file's first lookup record.
            let table_id = hit
                .map(|h| h.id.clone())
                .or_else(|| outcome.lookup_id().map(str::to_string))
                .or_else(|| lookup_ids.first().cloned());
            let table = match &table_id {
                Some(id) => lanes.lookup_table(id)?,
                None => None,
            };
            let expect = d.expect[t].as_deref();
            let (exp_entry, exp_via) = match (&table, expect) {
                (Some(tb), Some(e)) => match expect_entry(tb, e) {
                    Some((x, via)) => (Some(x), Some(via)),
                    None => (None, None),
                },
                _ => (None, None),
            };
            // No table to resolve against (a file without a lookup record):
            // nothing is counted.
            expect_unresolved += (expect.is_some() && table.is_some() && exp_entry.is_none()) as usize;
            let entry = hit.map(|h| h.key.entry);
            let binomial = match (&table, entry) {
                (Some(tb), Some(e)) => entry_binomial(tb, e),
                _ => None,
            };
            let correct = match (entry, exp_entry) {
                (Some(e), Some(x)) => Some(e == x),
                _ => None,
            };
            let back = hit.map(|h| h.key.turn);
            if let Some(b) = back {
                *hist.entry(b).or_default() += 1;
            }
            if let Some(f) = hit.and_then(|h| h.field.as_deref()) {
                *field_counts.entry(f.to_string()).or_default() += 1;
            }
            all.add(t, back, exp_entry.is_some(), correct);
            per_dialog_lang
                .get_mut(&d.lang)
                .expect("lang tally")
                .add(t, back, exp_entry.is_some(), correct);
            let turn_lang = turn_script(text);
            let tl = per_turn_lang.entry(turn_lang.to_string()).or_default();
            if !turn_langs_seen.contains(&turn_lang) {
                turn_langs_seen.push(turn_lang);
                tl.dialogs += 1;
            }
            tl.add(t, back, exp_entry.is_some(), correct);

            // The answer: the table's text (`answer`), a short greedy
            // generation on serve's transcript (`context`), nothing (`off`
            // or an `answer`-mode turn the table did not take).
            let (answer, generated, finish): (Option<String>, usize, String) =
                match (&outcome, lookup_mode) {
                    (LookupOutcome::Answer(h), _) => (Some(h.text.clone()), 0, "lookup".into()),
                    (_, LookupMode::Context) => {
                        let pipeline = lanes.lane(&decision.target)?;
                        let mut msgs: Vec<Value> = Vec::new();
                        for (u, asst) in &history {
                            msgs.push(json!({"role": "user", "content": u}));
                            msgs.push(json!({"role": "assistant", "content": asst}));
                        }
                        msgs.push(json!({"role": "user", "content": outcome.generation_text(text)}));
                        let ids = pipeline.tokenizer.apply_chat_template_json(&msgs, None, None);
                        let r = pipeline
                            .generate_from_ids(&ids, max_tokens, None, None)
                            .map_err(|e| anyhow::anyhow!("{}: generate: {e}", a.model))?;
                        let full = pipeline.tokenizer.decode(&r.token_ids);
                        let ans = full
                            .split("<|im_end|>")
                            .next()
                            .unwrap_or("")
                            .split("<|endoftext|>")
                            .next()
                            .unwrap_or("")
                            .to_string();
                        (Some(ans), r.token_ids.len(), r.finish_reason)
                    }
                    _ => (None, 0, "none".into()),
                };
            history.push((text.clone(), answer.clone().unwrap_or_default()));

            let mut route = decision.summary_json();
            outcome.annotate(&mut route, lookup_mode);
            let miss_kind = match (hit.is_some(), correct) {
                (false, _) => Some("no_hit"),
                (true, Some(false)) => Some("wrong_entry"),
                _ => None,
            };
            if let Some(kind) = miss_kind {
                misses_total += 1;
                if misses.len() < MISSES_LISTED {
                    misses.push(json!({
                        "dialog": di,
                        "id": d.id,
                        "t": t,
                        "kind": kind,
                        "text": text,
                        "target": target,
                        "decided_target": decided,
                        "entry": entry,
                        "expect_src": expect,
                        "expect_entry": exp_entry,
                        "reason": decision.reason,
                    }));
                }
            }
            if !a.json {
                let hit_s = match hit {
                    Some(h) => format!(
                        "hit {}back {} | entry {} {:?} | expect {} | field {}",
                        if h.key.via == lookup::MatchVia::Stem { "(stem) " } else { "" },
                        h.key.turn,
                        h.key.entry,
                        binomial.as_deref().unwrap_or("?"),
                        match correct {
                            Some(true) => "ok".to_string(),
                            Some(false) => format!("WRONG (expected entry {})", exp_entry.unwrap_or(0)),
                            None => "unchecked".to_string(),
                        },
                        h.field.as_deref().unwrap_or("-")
                    ),
                    None => "miss".to_string(),
                };
                println!(
                    "{} t{t}: route {target} (decided {decided}{}) | {hit_s} | {:?}",
                    d.id,
                    outcome
                        .decided_by()
                        .map(|b| format!(" by {}", b.label()))
                        .unwrap_or_default(),
                    clip(text, 60)
                );
                if let Some(ans) = &answer {
                    println!("   answer: {:?}", clip(ans, 160));
                }
            }
            rows.push(json!({
                "t": t,
                "text": text,
                "window": window.len(),
                "target": target,
                "decided_target": decided,
                "decided_by": outcome.decided_by().map(|b| b.label()),
                "lookup_hit": hit.is_some(),
                "lookup_turn": back,
                "lookup_key": hit.map(|h| h.key.key.clone()),
                "lookup_match": hit.map(|h| h.key.via.label()),
                "lookup_key_words": hit.map(|h| h.key.words),
                "entry": entry,
                "entry_binomial": binomial,
                "expect_src": expect,
                "expect_entry": exp_entry,
                "expect_via": exp_via,
                "correct": correct,
                "field": hit.and_then(|h| h.field.clone()),
                "lang": hit.map(|h| h.lang.clone()),
                "turn_lang": turn_lang,
                "answer": answer,
                "generated": generated,
                "finish_reason": finish,
                "route": route,
            }));
        }
        per_dialog.push(json!({
            "dialog": di,
            "id": d.id,
            "src": d.src,
            "lang": d.lang,
            "turns": rows,
        }));
    }

    let mut summary = json!({
        "model": a.model,
        "backend": backend,
        "route_mode": mode.label(),
        "lookup_mode": lookup_mode.label(),
        "frame": frame_label(frame),
        "memory_turns": lookup::MEMORY_TURNS,
        "include_quarantine": a.include_quarantine,
        "max_tokens": if lookup_mode == LookupMode::Context { json!(max_tokens) } else { Value::Null },
        "lookup_records": policies,
        "router_v2": router::is_router_v2(&model),
        "warnings": warnings,
    });
    if let (Value::Object(m), Value::Object(t)) = (&mut summary, all.json()) {
        m.extend(t);
    }
    let slices = |m: &BTreeMap<String, Tally>| {
        json!(m.iter().map(|(k, v)| (k.clone(), v.json())).collect::<BTreeMap<_, _>>())
    };
    // `per_lang` = by the script of each TURN (the language its answer is
    // chosen in); `per_dialog_lang` = by the dialog's language.
    summary["per_lang"] = slices(&per_turn_lang);
    summary["per_turn_lang"] = slices(&per_turn_lang);
    summary["per_dialog_lang"] = slices(&per_dialog_lang);
    summary["lookup_turn_hist"] = json!(hist
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<BTreeMap<_, _>>());
    summary["route_counts"] = json!(route_counts);
    summary["decided_counts"] = json!(decided_counts);
    summary["decided_by_counts"] = json!(decided_by_counts);
    summary["field_counts"] = json!(field_counts);
    summary["expect_unresolved"] = json!(expect_unresolved);
    summary["misses_total"] = json!(misses_total);
    summary["misses"] = json!(misses);
    summary["per_dialog"] = json!(per_dialog);

    if a.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let f = |v: &Value| match v.as_f64() {
        Some(x) => format!("{x:.3}"),
        None => "-".into(),
    };
    let line = |label: &str, s: &Value| {
        println!(
            "== {label}: dialogs {}, turns {} | first-turn hits {}/{} ({}) | follow-up hits {}/{} ({}), from memory {} | follow-up correct entry {}/{} ({}), of expected {}/{} | mean lookup_turn {} (follow-ups {})",
            s["dialogs"],
            s["turns"],
            s["first_turn_hits"],
            s["first_turns"],
            f(&s["first_turn_hit_rate"]),
            s["followup_hits"],
            s["followup_turns"],
            f(&s["followup_hit_rate"]),
            s["followup_memory_hits"],
            s["followup_correct"],
            s["followup_checked"],
            f(&s["followup_correct_rate"]),
            s["followup_correct"],
            s["followup_expected"],
            f(&s["mean_lookup_turn"]),
            f(&s["mean_lookup_turn_followup"]),
        );
    };
    line(a.model, &summary);
    for (lang, t) in &per_turn_lang {
        line(&format!("turns in {lang}"), &t.json());
    }
    for (lang, t) in &per_dialog_lang {
        line(&format!("dialogs in {lang}"), &t.json());
    }
    for w in &warnings {
        println!("   warning: {w}");
    }
    println!(
        "   routes {route_counts:?} (decided {decided_counts:?}, by {decided_by_counts:?}) | lookup_turn {:?} | fields {field_counts:?} | expectations unresolved {expect_unresolved} | misses {misses_total}",
        hist
    );
    for m in &misses {
        println!(
            "   miss {} t{} [{}]: {:?} → {} (decided {})",
            m["id"].as_str().unwrap_or("?"),
            m["t"],
            m["kind"].as_str().unwrap_or("?"),
            clip(m["text"].as_str().unwrap_or(""), 60),
            m["target"].as_str().unwrap_or("?"),
            m["decided_target"].as_str().unwrap_or("?"),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialogs_parse_expectations_per_turn_and_default_to_src() {
        let text = "\
# comment
{\"src\": \"Matricaria chamomilla\", \"turns\": [\"Что такое ромашка аптечная?\", \"А семейство?\"], \"expect_src\": \"Matricaria chamomilla\"}

{\"src\": \"Salvia officinalis\", \"turns\": [\"a\", \"b\", \"c\"], \"expect_src\": [\"Salvia officinalis\", null, \"Conium maculatum\"], \"id\": \"switch\"}
{\"src\": \"Calendula officinalis\", \"turns\": [\"Tell me about pot marigold.\"], \"lang\": \"en\"}
{\"turns\": [\"x\"]}
";
        let d = parse_dialogs_jsonl(text).unwrap();
        assert_eq!(d.len(), 4);
        assert_eq!(d[0].id, "Matricaria chamomilla");
        assert_eq!(d[0].lang, "ru");
        assert_eq!(d[0].expect, vec![Some("Matricaria chamomilla".to_string()); 2]);
        assert_eq!(d[1].id, "switch");
        assert_eq!(d[1].lang, "en", "the script of the first turn");
        assert_eq!(
            d[1].expect,
            vec![Some("Salvia officinalis".into()), None, Some("Conium maculatum".into())]
        );
        assert_eq!(d[2].expect, vec![Some("Calendula officinalis".to_string())]);
        assert_eq!(d[3].id, "line 6");
        assert_eq!(d[3].expect, vec![None]);
        // Refusals.
        assert!(parse_dialogs_jsonl("{\"turns\": []}").is_err());
        assert!(parse_dialogs_jsonl("{\"turns\": [1]}").is_err());
        assert!(parse_dialogs_jsonl("{\"turns\": [\"a\"], \"expect_src\": [\"x\", \"y\"]}").is_err());
        assert!(parse_dialogs_jsonl("{\"turns\": [\"a\"], \"expect_src\": 3}").is_err());
        assert!(parse_dialogs_jsonl("[1]").is_err());
        assert!(parse_dialogs_jsonl("\n# only a comment\n").is_err());
    }

    #[test]
    fn the_window_is_the_turn_then_earlier_turns_capped_at_memory_turns() {
        let turns: Vec<String> = (0..9).map(|i| format!("u{i}")).collect();
        assert_eq!(turn_window(&turns, 0), vec!["u0"]);
        assert_eq!(turn_window(&turns, 2), vec!["u2", "u1", "u0"]);
        let w = turn_window(&turns, 8);
        assert_eq!(w.len(), lookup::MEMORY_TURNS);
        assert_eq!(w[0], "u8");
        assert_eq!(w[lookup::MEMORY_TURNS - 1], format!("u{}", 9 - lookup::MEMORY_TURNS));
    }
}
