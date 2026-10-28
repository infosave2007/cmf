//! A teacher behind any OpenAI-compatible chat API: `teacher-data generate
//! openrouter:qwen/qwen3.8-27b prompts.jsonl out.tdata --tokenizer
//! model.cmf`, `openai:MODEL`, or `api:MODEL --api-base URL` for vLLM,
//! SGLang, llama.cpp, Ollama, LM Studio, DeepSeek, Together, Groq, … .
//! Renting a server for a large teacher costs more than its tokens through
//! an API; the answers come back as text and become teacher data through
//! the tokenizer of the model they will compress (the `teacher-data import`
//! path).
//!
//! Requests run `--api-concurrency` at a time; every finished answer is
//! appended to a JSONL cache next to the output, so an interrupted run is
//! continued by the same command. Servers differ in which optional fields
//! they accept (`top_k`, `seed`, reasoning switches, `max_tokens` versus
//! `max_completion_tokens`): a field a server refuses is dropped once, said
//! so, and the run goes on. The key is read from the environment and never
//! written anywhere.

use crate::tdata::TdataMessage;
use crate::teacher_data::TeacherPrompt;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The API options shared by `teacher-data generate` and `compress`.
#[derive(clap::Args, Debug, Clone)]
pub struct ApiArgs {
    /// API teacher: the server's OpenAI-compatible base URL (…/v1);
    /// default per prefix — `openrouter:` → https://openrouter.ai/api/v1,
    /// `openai:` → https://api.openai.com/v1, `api:` → $CORTIQ_API_BASE
    #[arg(long)]
    pub api_base: Option<String>,
    /// API teacher: extra JSON fields for every request, e.g.
    /// '{"provider":{"order":["deepinfra"]}}' or '{"reasoning_effort":"low"}'
    /// (a field set to null is not sent)
    #[arg(long)]
    pub api_extra: Option<String>,
    /// API teacher: requests in flight at once
    #[arg(long, default_value_t = 8)]
    pub api_concurrency: usize,
}

impl Default for ApiArgs {
    fn default() -> Self {
        Self { api_base: None, api_extra: None, api_concurrency: 8 }
    }
}

/// Which request fields a server family understands.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dialect {
    /// `reasoning: {enabled}`, `usage: {include}` (cost), `top_k`
    OpenRouter,
    /// `max_completion_tokens`, no sampler extensions
    OpenAi,
    /// vLLM / SGLang / llama.cpp style: `top_k`,
    /// `chat_template_kwargs: {enable_thinking}`
    Generic,
}

/// The weight formats an OpenRouter provider may serve the teacher in.
const OPENROUTER_QUANTS: [&str; 4] = ["bf16", "fp16", "fp32", "fp8"];

/// A model behind an OpenAI-compatible `/chat/completions`.
pub struct ApiTeacher {
    pub base: String,
    pub model: String,
    key: Option<String>,
    dialect: Dialect,
    extra: serde_json::Map<String, Value>,
    agent: ureq::Agent,
    /// Optional fields the server refused, and `max_tokens` renamed.
    refused: Mutex<HashSet<String>>,
}

/// `PREFIX:MODEL` → (dialect, default base, key variables, key required).
fn provider(spec: &str) -> Option<(Dialect, &str, Option<&'static str>, &'static [&'static str], bool)> {
    if let Some(m) = spec.strip_prefix("openrouter:") {
        return Some((Dialect::OpenRouter, m, Some("https://openrouter.ai/api/v1"), &["OPENROUTER_API_KEY", "CORTIQ_API_KEY"], true));
    }
    if let Some(m) = spec.strip_prefix("openai:") {
        return Some((Dialect::OpenAi, m, Some("https://api.openai.com/v1"), &["OPENAI_API_KEY", "CORTIQ_API_KEY"], true));
    }
    if let Some(m) = spec.strip_prefix("api:") {
        // a local server usually needs no key
        return Some((Dialect::Generic, m, None, &["CORTIQ_API_KEY", "OPENAI_API_KEY"], false));
    }
    None
}

/// True for a teacher spec that names an API model, not a local file.
pub fn is_api_spec(spec: &str) -> bool {
    provider(spec).is_some()
}

impl ApiTeacher {
    /// The teacher a spec names, or None for a local .cmf.
    pub fn parse(spec: &str, args: &ApiArgs) -> Result<Option<Self>> {
        let Some((mut dialect, model, default_base, key_vars, key_required)) = provider(spec) else {
            return Ok(None);
        };
        anyhow::ensure!(!model.is_empty(), "teacher {spec:?}: no model name after the prefix");
        let base = match args.api_base.clone().or_else(|| std::env::var("CORTIQ_API_BASE").ok()) {
            Some(b) => b,
            None => default_base
                .with_context(|| format!("teacher {spec:?}: give the server's OpenAI-compatible base URL with --api-base (e.g. http://localhost:8000/v1) or CORTIQ_API_BASE"))?
                .to_string(),
        };
        let base = base.trim_end_matches('/').trim_end_matches("/chat/completions").to_string();
        anyhow::ensure!(base.starts_with("http://") || base.starts_with("https://"), "--api-base {base:?}: an http(s) URL");
        // a generic spec pointed at a known service speaks its dialect
        if dialect == Dialect::Generic && base.contains("openrouter.ai") {
            dialect = Dialect::OpenRouter;
        }
        if dialect == Dialect::Generic && base.contains("api.openai.com") {
            dialect = Dialect::OpenAi;
        }
        let key = key_vars.iter().find_map(|v| std::env::var(v).ok().filter(|k| !k.trim().is_empty()));
        anyhow::ensure!(!key_required || key.is_some(), "teacher {spec:?}: put the API key in {}", key_vars.join(" or "));
        let extra = match &args.api_extra {
            Some(e) => match serde_json::from_str::<Value>(e).with_context(|| format!("--api-extra {e:?}: not JSON"))? {
                Value::Object(m) => m,
                _ => anyhow::bail!("--api-extra: a JSON object"),
            },
            None => Default::default(),
        };
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout(Duration::from_secs(1800)).build();
        Ok(Some(Self { base, model: model.to_string(), key, dialect, extra, agent, refused: Mutex::new(HashSet::new()) }))
    }

    pub fn label(&self) -> String {
        let route = if self.dialect == Dialect::OpenRouter && !self.extra.contains_key("provider") {
            " (cheapest provider with 8/16-bit weights)"
        } else {
            ""
        };
        format!("{} @ {}{route}", self.model, self.base)
    }

    /// The request body, without the fields the server refused.
    fn body(&self, messages: &[TdataMessage], max_tokens: usize, sampling: &ApiSampling, think: bool, seed: u64) -> Value {
        let refused = self.refused.lock().unwrap().clone();
        let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
        let mut b = json!({ "model": self.model, "messages": msgs });
        // OpenAI's reasoning models take only max_completion_tokens
        if self.dialect == Dialect::OpenAi || refused.contains("max_tokens") {
            b["max_completion_tokens"] = json!(max_tokens);
        } else {
            b["max_tokens"] = json!(max_tokens);
        }
        let mut opt = |k: &str, v: Value| {
            if !refused.contains(k) {
                b[k] = v;
            }
        };
        opt("temperature", json!(sampling.temperature));
        opt("seed", json!(seed));
        if let Some(p) = sampling.top_p {
            opt("top_p", json!(p));
        }
        match self.dialect {
            Dialect::OpenRouter => {
                if let Some(k) = sampling.top_k {
                    opt("top_k", json!(k));
                }
                // providers of one model differ ×2.5 in price and some serve
                // it in fp4: the cheapest one with 8- or 16-bit weights
                opt("provider", json!({"sort": "price", "quantizations": OPENROUTER_QUANTS}));
                opt("usage", json!({"include": true}));
                opt("reasoning", if think { json!({"enabled": true, "exclude": false}) } else { json!({"enabled": false}) });
            }
            Dialect::Generic => {
                if let Some(k) = sampling.top_k {
                    opt("top_k", json!(k));
                }
                opt("chat_template_kwargs", json!({"enable_thinking": think}));
            }
            Dialect::OpenAi => {}
        }
        for (k, v) in &self.extra {
            if v.is_null() {
                b.as_object_mut().unwrap().remove(k);
            } else {
                b[k] = v.clone();
            }
        }
        b
    }

    /// The optional fields a 400 answer says the server does not take; true
    /// when at least one is newly dropped (then the request is resent).
    fn refuse_from(&self, sent: &Value, error: &str) -> bool {
        let low = error.to_lowercase();
        // no provider matches the default preference: let OpenRouter choose
        if self.dialect == Dialect::OpenRouter
            && !self.extra.contains_key("provider")
            && sent.get("provider").is_some()
            && (low.contains("endpoint") || low.contains("provider"))
            && self.refused.lock().unwrap().insert("provider".into())
        {
            eprintln!("teacher-data: no 8/16-bit provider serves this model — letting OpenRouter choose (any precision)");
            return true;
        }
        let says_unsupported = [
            "unsupported", "unrecognized", "unknown", "not supported", "does not support", "not permitted",
            "not allowed", "extra inputs", "extra fields", "extra_forbidden", "unexpected", "only the default", "invalid parameter",
        ]
        .iter()
        .any(|w| low.contains(w));
        if !says_unsupported {
            return false;
        }
        let user_set: HashSet<&String> = self.extra.keys().collect();
        let mut dropped = Vec::new();
        let mut refused = self.refused.lock().unwrap();
        for k in ["max_tokens", "temperature", "seed", "top_p", "top_k", "usage", "reasoning", "chat_template_kwargs"] {
            if sent.get(k).is_some() && !user_set.contains(&k.to_string()) && low.contains(k) && refused.insert(k.to_string()) {
                dropped.push(k);
            }
        }
        drop(refused);
        for k in &dropped {
            if *k == "max_tokens" {
                eprintln!("teacher-data: the API refused max_tokens — sending max_completion_tokens instead");
            } else {
                eprintln!("teacher-data: the API refused `{k}` — continuing without it");
            }
        }
        !dropped.is_empty()
    }
}

/// Decoding of the API requests.
pub struct ApiSampling {
    pub temperature: f64,
    pub top_p: Option<f64>,
    /// An integer: strict servers refuse `20.0`.
    pub top_k: Option<u64>,
}

impl ApiSampling {
    /// From `--sampling`: greedy | auto | file | K=V,… (an API teacher has
    /// no file to carry recommended settings: `auto`/`file` use the
    /// reasoning-model recommendation 0.6 / 0.95 / 20).
    pub fn parse(spec: &str) -> Result<Self> {
        let mut s = Self { temperature: 0.0, top_p: None, top_k: None };
        match spec {
            "greedy" => {}
            "file" | "auto" => {
                // a reasoning model loops greedily: the Qwen recommendation
                s = Self { temperature: 0.6, top_p: Some(0.95), top_k: Some(20) };
            }
            list => {
                for kv in list.split(',') {
                    let (k, v) = kv.split_once('=').with_context(|| format!("--sampling {list:?}: KEY=VALUE pairs"))?;
                    let v: f64 = v.trim().parse()?;
                    match k.trim() {
                        "temperature" => s.temperature = v,
                        "top_p" => s.top_p = Some(v),
                        "top_k" => s.top_k = Some(v.round().max(0.0) as u64),
                        other => anyhow::bail!("--sampling: {other} is not sent to an API teacher (temperature, top_p, top_k; anything else goes in --api-extra)"),
                    }
                }
            }
        }
        Ok(s)
    }
}

/// What one answer cost and produced.
struct Done {
    tokens: u64,
    cost: f64,
    /// The text stops inside the reasoning (no `</think>`).
    open_think: bool,
    /// Who served it (OpenRouter names the provider).
    provider: Option<String>,
}

/// A request the server will never accept as sent (bad key, unknown model,
/// …): the whole run stops instead of failing every prompt alike.
struct Fatal(String);

/// One chat completion; retried on rate limits, server and network errors.
fn ask(t: &ApiTeacher, messages: &[TdataMessage], max_tokens: usize, sampling: &ApiSampling, think: bool, seed: u64) -> std::result::Result<(String, String, Done), (String, bool)> {
    let mut wait = 2u64;
    let mut last = String::new();
    let mut attempt = 0;
    let url = format!("{}/chat/completions", t.base);
    while attempt < 6 {
        let body = t.body(messages, max_tokens, sampling, think, seed);
        let mut req = t.agent.post(&url).set("Content-Type", "application/json");
        if let Some(k) = &t.key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        if t.dialect == Dialect::OpenRouter {
            req = req.set("X-Title", "cortiq teacher-data");
        }
        match req.send_json(body.clone()) {
            Ok(resp) => {
                let v: Value = match resp.into_json() {
                    Ok(v) => v,
                    Err(e) => {
                        last = format!("the answer is not JSON: {e}");
                        attempt += 1;
                        continue;
                    }
                };
                if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
                    // a provider failure reported inside a 200
                    last = format!("API error: {}", e.to_string().chars().take(300).collect::<String>());
                } else {
                    let ch = &v["choices"][0];
                    let msg = &ch["message"];
                    let content = msg["content"].as_str().unwrap_or("");
                    let reasoning = msg["reasoning"].as_str().or_else(|| msg["reasoning_content"].as_str()).unwrap_or("");
                    // servers name the token cap differently; the import knows "length"
                    let finish = match ch["finish_reason"].as_str().unwrap_or("") {
                        "length" | "max_tokens" | "max_output_tokens" => "length".to_string(),
                        f => f.to_string(),
                    };
                    if content.is_empty() && reasoning.is_empty() {
                        last = format!("empty answer (finish {finish:?})");
                    } else {
                        let (text, open_think) = compose(reasoning, content, think);
                        let provider = v["provider"].as_str().map(str::to_string);
                        let usage = &v["usage"];
                        let done = Done {
                            tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
                            cost: usage["cost"].as_f64().unwrap_or(0.0),
                            open_think,
                            provider,
                        };
                        return Ok((text, finish, done));
                    }
                }
            }
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                if matches!(code, 400 | 404 | 422) && t.refuse_from(&body, &text) {
                    // resend without the refused field; not an attempt
                    continue;
                }
                let short: String = text.chars().take(400).collect();
                if code == 429 || code == 408 || code >= 500 {
                    last = format!("HTTP {code}: {short}");
                } else {
                    return Err((format!("HTTP {code} from {url}: {short}"), true));
                }
            }
            Err(e) => last = e.to_string(),
        }
        attempt += 1;
        if attempt < 6 {
            std::thread::sleep(Duration::from_secs(wait));
            wait = (wait * 2).min(60);
        }
    }
    Err((last, false))
}

/// The answer as the student's template continues it, and whether it stops
/// inside the reasoning. With reasoning on, the prompt ends with
/// `<think>\n`: the text is the reasoning, the `</think>` boundary, the
/// reply. A reply that carries its own `<think>…</think>` keeps it, minus
/// the opening tag; a reply with no reasoning at all (a server that hides
/// it) is the plain reply, which the import renders after an empty
/// reasoning block. With reasoning off, any reasoning is left out.
fn compose(reasoning: &str, content: &str, think: bool) -> (String, bool) {
    let c = content.trim_start();
    if !think {
        let reply = match c.strip_prefix("<think>").and_then(|r| r.split_once("</think>")) {
            Some((_, after)) => after.trim_start().to_string(),
            None => content.to_string(),
        };
        return (reply, false);
    }
    if let Some(rest) = c.strip_prefix("<think>") {
        let rest = rest.trim_start_matches('\n').to_string();
        let open = !rest.contains("</think>");
        return (rest, open);
    }
    if reasoning.trim().is_empty() {
        return (content.to_string(), false);
    }
    if c.is_empty() {
        // cut off while still reasoning
        return (reasoning.to_string(), true);
    }
    (format!("{}\n</think>\n\n{}", reasoning.trim_end(), c), false)
}

/// Answer every prompt (`samples` each) into the JSONL cache `cache`
/// (one `{id, category, messages, response, meta}` row per answer); rows
/// already there are kept and skipped.
#[allow(clippy::too_many_arguments)]
pub fn answer_all(
    t: &ApiTeacher,
    prompts: &[TeacherPrompt],
    samples: u32,
    max_tokens: usize,
    sampling: &ApiSampling,
    think: bool,
    seed: u64,
    concurrency: usize,
    cache: &std::path::Path,
) -> Result<()> {
    let mut have: HashMap<String, u32> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(cache) {
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Ok(v) = serde_json::from_str::<Value>(line)
                && let Some(id) = v["id"].as_str()
            {
                *have.entry(id.to_string()).or_default() += 1;
            }
        }
    }
    let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
    for (i, p) in prompts.iter().enumerate() {
        let want = p.samples.unwrap_or(samples);
        for s in have.get(&p.id).copied().unwrap_or(0)..want {
            queue.push_back((i, s));
        }
    }
    let total = queue.len();
    let kept = have.values().sum::<u32>();
    eprintln!(
        "teacher-data: {} — {total} answers to request{}, {concurrency} at a time",
        t.label(),
        if kept > 0 { format!(" ({kept} already in {})", cache.display()) } else { String::new() }
    );
    if total == 0 {
        return Ok(());
    }
    let queue = Arc::new(Mutex::new(queue));
    let out = Arc::new(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(cache).with_context(|| format!("opening {}", cache.display()))?));
    struct Stats {
        done: usize,
        tokens: u64,
        cost: f64,
        failed: Vec<String>,
        fatal: Option<String>,
        /// Answers cut by the token cap, and of them inside the reasoning.
        cut: usize,
        cut_thinking: usize,
    }
    let stats = Arc::new(Mutex::new(Stats { done: 0, tokens: 0, cost: 0.0, failed: Vec::new(), fatal: None, cut: 0, cut_thinking: 0 }));
    let stop = AtomicBool::new(false);
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..concurrency.max(1).min(total) {
            let (queue, out, stats, stop) = (queue.clone(), out.clone(), stats.clone(), &stop);
            sc.spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Some((i, s)) = queue.lock().unwrap().pop_front() else { break };
                    let p = &prompts[i];
                    let max = p.max_tokens.unwrap_or(max_tokens);
                    let record_seed = crate::teacher_data::record_seed(seed, &p.id, s);
                    match ask(t, &p.messages, max, sampling, think, record_seed) {
                        Ok((response, finish, d)) => {
                            let msgs: Vec<Value> = p.messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
                            let mut meta = json!({"finish": finish, "sample": s, "completion_tokens": d.tokens});
                            if d.cost > 0.0 {
                                meta["cost"] = json!(d.cost);
                            }
                            if d.open_think {
                                meta["open_think"] = json!(true);
                            }
                            if let Some(p) = &d.provider {
                                meta["provider"] = json!(p);
                            }
                            let row = json!({"id": p.id, "category": p.category, "messages": msgs, "response": response, "meta": meta});
                            {
                                let mut f = out.lock().unwrap();
                                let _ = writeln!(f, "{row}");
                                let _ = f.flush();
                            }
                            let mut st = stats.lock().unwrap();
                            st.done += 1;
                            st.tokens += d.tokens;
                            st.cost += d.cost;
                            st.cut += usize::from(finish == "length");
                            st.cut_thinking += usize::from(d.open_think);
                            let el = t0.elapsed().as_secs_f64();
                            let left = total - st.done - st.failed.len();
                            eprintln!(
                                "teacher-data: [{}/{total}] {}#{s} finish={finish} {} tok · {:.0} tok/s overall{} · ~{} left",
                                st.done,
                                p.id,
                                d.tokens,
                                st.tokens as f64 / el.max(1e-9),
                                if st.cost > 0.0 { format!(" · ${:.4} so far", st.cost) } else { String::new() },
                                crate::fmt_eta(el / st.done as f64 * left as f64)
                            );
                        }
                        Err((e, fatal)) => {
                            let mut st = stats.lock().unwrap();
                            if fatal {
                                if st.fatal.is_none() {
                                    st.fatal = Some(e);
                                }
                                stop.store(true, Ordering::Relaxed);
                            } else {
                                eprintln!("teacher-data: {}#{s}: {e}", p.id);
                                st.failed.push(p.id.clone());
                            }
                        }
                    }
                }
            });
        }
    });
    let st = stats.lock().unwrap();
    eprintln!(
        "teacher-data: {} answers, {} generated tokens{} in {}",
        st.done,
        st.tokens,
        if st.cost > 0.0 { format!(", ${:.4}", st.cost) } else { String::new() },
        crate::fmt_eta(t0.elapsed().as_secs_f64())
    );
    if st.cut * 4 >= st.done.max(1) {
        eprintln!(
            "teacher-data: note — {} of {} answers hit the {max_tokens}-token cap ({} still inside the reasoning, so without a reply); \
             they are kept as truncated records. For complete replies raise the cap (--max-tokens; in compress --gen-max-tokens) or turn the reasoning off (--no-think)",
            st.cut, st.done, st.cut_thinking
        );
    }
    if let Some(e) = &st.fatal {
        anyhow::bail!("the API refused the request — {e}");
    }
    anyhow::ensure!(
        st.failed.is_empty(),
        "{} answers failed ({}…) — rerun the same command to retry them; the {} finished ones are kept",
        st.failed.len(),
        st.failed.iter().take(3).cloned().collect::<Vec<_>>().join(", "),
        st.done
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher(spec: &str, base: Option<&str>, extra: Option<&str>) -> ApiTeacher {
        let args = ApiArgs { api_base: base.map(Into::into), api_extra: extra.map(Into::into), api_concurrency: 1 };
        ApiTeacher::parse(spec, &args).unwrap().unwrap()
    }

    fn msgs() -> Vec<TdataMessage> {
        vec![TdataMessage { role: "user".into(), content: json!("hi") }]
    }

    #[test]
    fn answers_continue_the_reasoning_prompt() {
        assert_eq!(compose("plan", "code", true), ("plan\n</think>\n\ncode".into(), false));
        assert_eq!(compose("", "code", true), ("code".into(), false));
        assert_eq!(compose("", "<think>\nplan\n</think>\n\ncode", true), ("plan\n</think>\n\ncode".into(), false));
        // cut off inside the reasoning
        assert_eq!(compose("plan…", "", true), ("plan…".into(), true));
        assert_eq!(compose("", "<think>\nplan…", true), ("plan…".into(), true));
        // reasoning off: only the reply
        assert_eq!(compose("plan", "code", false), ("code".into(), false));
        assert_eq!(compose("", "<think>\nplan\n</think>\n\ncode", false), ("code".into(), false));
    }

    #[test]
    fn specs_pick_a_dialect_and_a_base() {
        assert!(ApiTeacher::parse("model.cmf", &ApiArgs::default()).unwrap().is_none());
        // a local server: no key needed, base required
        let t = teacher("api:Qwen/Qwen3-32B", Some("http://localhost:8000/v1/"), None);
        assert_eq!((t.base.as_str(), t.dialect), ("http://localhost:8000/v1", Dialect::Generic));
        let b = t.body(&msgs(), 64, &ApiSampling::parse("auto").unwrap(), true, 7);
        assert_eq!(b["max_tokens"], 64);
        assert_eq!(b["top_k"], 20);
        assert!(b["top_k"].is_u64());
        assert_eq!(b["chat_template_kwargs"]["enable_thinking"], true);
        assert!(b.get("reasoning").is_none());
        // the full endpoint is accepted too
        let t = teacher("api:m", Some("https://openrouter.ai/api/v1/chat/completions"), None);
        assert_eq!((t.base.as_str(), t.dialect), ("https://openrouter.ai/api/v1", Dialect::OpenRouter));
    }

    #[test]
    fn refused_fields_are_dropped_and_extra_wins() {
        let t = teacher("api:m", Some("http://h/v1"), Some(r#"{"top_k":null,"provider":{"order":["x"]}}"#));
        let s = ApiSampling::parse("auto").unwrap();
        let b = t.body(&msgs(), 64, &s, false, 1);
        assert!(b.get("top_k").is_none());
        assert_eq!(b["provider"]["order"][0], "x");
        // OpenAI-style refusals
        assert!(t.refuse_from(&b, "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."));
        assert!(t.refuse_from(&b, r#"{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}"#));
        let b = t.body(&msgs(), 64, &s, false, 1);
        assert_eq!(b["max_completion_tokens"], 64);
        assert!(b.get("max_tokens").is_none() && b.get("chat_template_kwargs").is_none());
        // a context-length error is not a refusal: nothing is dropped
        assert!(!t.refuse_from(&b, "max_completion_tokens is too large: this model's maximum context length is 8192"));
        assert!(!t.refuse_from(&b, "Unsupported parameter: 'max_completion_tokens'"));
    }
}
