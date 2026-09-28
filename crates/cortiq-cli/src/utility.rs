//! `cortiq probe-utility` — the S10 utility gate through the RUNTIME under
//! the corrected cmf-im-v1 chat contract
//! (`Lunacy/.../quality-final-contract-audit.md`, trainer template
//! `crates/cortiq-embryo/src/sft.rs`): the prompt is
//! `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n` encoded
//! with `Tokenizer::encode` (added tokens `<|im_start|>` = 32761,
//! `<|im_end|>` = 32762 on the Embryo BPE; the terminal assistant newline,
//! id 10, is part of the prefix — the historical 0/11 came from a shell
//! `$(printf …)` that stripped it), no BOS unless asked.  Greedy decode of
//! `--max-tokens` (≥ 128) tokens; per prompt: exact / keyword hit against
//! the expected substrings, the loop metrics (distinct-5-gram ratio over
//! the last 128 generated tokens — healthy > 0.30 — and the longest run
//! with a period ≤ 16; ≥ 64 tokens = loop), the first assistant token's
//! top-5 with probabilities, TTFT and steady tok/s.  Prompt file: TSV
//! `LABEL<TAB>EXPECTED[|ALT…]<TAB>PROMPT`, `#` comments (the frozen
//! `prompts-v1.txt` format); default = the audit's 11 utility prompts.
//! Runs on whatever the environment selects (CPU or the resident graph).

use crate::knowledge::{Lanes, RouteMode};
use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_engine::SamplerConfig;
use cortiq_engine::lookup::{LookupMode, LookupOutcome};
use cortiq_engine::router::{RouteOptions, RouteTarget};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct UtilityArgs<'a> {
    pub models: &'a [String],
    pub prompts: Option<String>,
    /// JSONL prompt file: one `{"lang","prompt","expect":[…],"src"}` per
    /// line; keyword hit = any `expect` string (case-insensitive) in the
    /// answer.
    pub prompts_jsonl: Option<String>,
    pub max_tokens: usize,
    pub rep_penalty: f32,
    pub bos: bool,
    pub json: bool,
    /// `--route auto|backbone|<id>`: the pipeline each prompt runs on.
    /// Unset: a ROUTER_V2 file routes per prompt (`auto`), a legacy file
    /// runs the plain pipeline (the behaviour before router v2).
    pub route: Option<String>,
    /// Score quarantined v2 skills too (gate measurement).
    pub include_quarantine: bool,
    /// `--lookup-mode answer|context|off` (None = `CMF_LOOKUP_MODE`, else
    /// `answer`): what a prompt routed to a lookup record does — the table
    /// answers (no generation), the backbone generates with the card
    /// prepended, or the table is ignored.
    pub lookup_mode: Option<String>,
}

pub struct PromptRow {
    pub label: String,
    pub expected: Vec<String>,
    pub prompt: String,
    /// The JSONL row's `src` — the source the prompt was made from (a
    /// plant name in the recall sets). A lookup hit is checked against
    /// it: does the card that answered belong to the entry `src` names?
    pub src: Option<String>,
}

/// The 11 frozen utility prompts of the working-sft-2 audit
/// (`Lunacy/runs/embryo-working-model/phases/native/artifacts/working-sft-2/prompts-v1.txt`).
pub const DEFAULT_PROMPTS: &str = "\
EN_FACT\tParis\tName the capital city of France. Answer with only the city name.
EN_EXPLAIN\taxis\tIn two concise sentences, explain why Earth has seasons.
EN_HASH\tconstant-time\tFor a beginner, explain why a hash table usually offers constant-time lookup.
RU_FACT\tПариж\tНазови столицу Франции. Ответь только названием города.
RU_EXPLAIN\tось\tВ двух коротких предложениях объясни, почему на Земле меняются времена года.
RU_HASH\tхеш-таблиц\tКратко объясни новичку, зачем нужна хеш-таблица.
CODE_RUST\tfn max_value\tWrite a Rust function `fn max_value(xs: &[i32]) -> Option<i32>` that returns the maximum element.
CODE_PYTHON\tdef max_value\tWrite a Python function `def max_value(xs):` that returns the maximum integer in a non-empty list.
MATH\t346\tCompute exactly: 17 * 19 + 23.
MATH_ALGEBRA\tx = 5\tSolve exactly for x: 3x + 7 = 22.
MIXED\tH2O\tIn one sentence in English, say what water is made of, then give the same sentence in Russian.
";

pub fn parse_prompts(text: &str) -> anyhow::Result<Vec<PromptRow>> {
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let (label, expected, prompt) = (parts.next(), parts.next(), parts.next());
        let (Some(label), Some(expected), Some(prompt)) = (label, expected, prompt) else {
            anyhow::bail!("prompts line {}: expected LABEL<TAB>EXPECTED<TAB>PROMPT", ln + 1);
        };
        rows.push(PromptRow {
            label: label.trim().to_string(),
            expected: expected
                .split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            prompt: prompt.to_string(),
            src: None,
        });
    }
    if rows.is_empty() {
        anyhow::bail!("prompt set is empty");
    }
    Ok(rows)
}

/// JSONL prompt set (`{"lang","prompt","expect":[…],"src"}` per line):
/// the label is `lang:src` (or `lang:N`), the expected strings are the
/// `expect` array (a string is accepted too).
pub fn parse_prompts_jsonl(text: &str) -> anyhow::Result<Vec<PromptRow>> {
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("prompts jsonl line {}", ln + 1))?;
        let prompt = v
            .get("prompt")
            .and_then(|p| p.as_str())
            .ok_or_else(|| anyhow::anyhow!("prompts jsonl line {}: no \"prompt\"", ln + 1))?
            .to_string();
        let expected: Vec<String> = match v.get("expect") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|e| e.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect(),
            Some(serde_json::Value::String(e)) => vec![e.trim().to_string()],
            _ => Vec::new(),
        };
        let lang = v.get("lang").and_then(|l| l.as_str()).unwrap_or("?");
        let src = v
            .get("src")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string());
        rows.push(PromptRow {
            label: format!("{lang}:{}", src.clone().unwrap_or_else(|| format!("{}", ln + 1))),
            expected,
            prompt,
            src,
        });
    }
    if rows.is_empty() {
        anyhow::bail!("prompt set is empty");
    }
    Ok(rows)
}

/// The corrected cmf-im-v1 prefix: the assistant newline is part of it.
/// One definition for every tool and `run` (the engine's renderer).
pub fn chat_prefix(prompt: &str) -> String {
    cortiq_engine::router::render_cmf_im_v1(prompt)
}

fn normalize(s: &str) -> String {
    s.trim()
        .trim_end_matches(['.', '!', '?', ';', ':'])
        .to_lowercase()
}

/// Distinct 5-gram ratio over `ids` (1.0 when fewer than 5 tokens).
pub fn distinct_5gram_ratio(ids: &[u32]) -> f64 {
    if ids.len() < 5 {
        return 1.0;
    }
    let mut seen = std::collections::HashSet::new();
    let total = ids.len() - 4;
    for w in ids.windows(5) {
        seen.insert(w.to_vec());
    }
    seen.len() as f64 / total as f64
}

/// Longest run of tokens satisfying `ids[i] == ids[i − p]` for one period
/// `p ≤ max_period` (run length counts the repeated span plus its seed
/// period). Returns `(period, run_len)` of the longest such run.
pub fn longest_periodic_run(ids: &[u32], max_period: usize) -> (usize, usize) {
    let mut best = (0usize, 0usize);
    for p in 1..=max_period.min(ids.len().saturating_sub(1)) {
        let mut run = 0usize;
        for i in p..ids.len() {
            if ids[i] == ids[i - p] {
                run += 1;
                let len = run + p;
                if len > best.1 {
                    best = (p, len);
                }
            } else {
                run = 0;
            }
        }
    }
    best
}

fn top5(logits: &[f32]) -> Vec<(u32, f64)> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.iter()
        .take(5)
        .map(|&i| (i as u32, (logits[i] as f64 - max).exp() / sum))
        .collect()
}

pub fn cmd_probe_utility(a: UtilityArgs<'_>) -> anyhow::Result<()> {
    let rows = match (&a.prompts, &a.prompts_jsonl) {
        (Some(path), _) => parse_prompts(&std::fs::read_to_string(path).with_context(|| path.clone())?)?,
        (None, Some(path)) => {
            parse_prompts_jsonl(&std::fs::read_to_string(path).with_context(|| path.clone())?)?
        }
        (None, None) => parse_prompts(DEFAULT_PROMPTS)?,
    };
    let backend = format!(
        "CMF_GPU={} resident={}",
        std::env::var("CMF_GPU").unwrap_or_else(|_| "unset".into()),
        std::env::var("CMF_EMBRYO_RESIDENT").unwrap_or_else(|_| "unset".into())
    );
    let lookup_mode = LookupMode::resolve(a.lookup_mode.as_deref()).map_err(anyhow::Error::msg)?;
    let mut all = Vec::new();
    for path in a.models {
        let model = Arc::new(CmfModel::open_sharded(path).with_context(|| path.clone())?);
        let mode = RouteMode::resolve(a.route.as_deref(), &model)?;
        let mut lanes = Lanes::new(
            model.clone(),
            SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: a.rep_penalty,
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
        let im_start = lanes.lane(&RouteTarget::Backbone)?.tokenizer.im_start_id;
        if !a.json {
            println!(
                "probe-utility: {path} | {} prompts, max_tokens={}, rep_penalty={}, bos={} | im_start={:?} | route={} | lookup={} | {backend}",
                rows.len(),
                a.max_tokens,
                a.rep_penalty,
                a.bos,
                im_start,
                mode.label(),
                lookup_mode.label()
            );
        }
        let mut per_prompt = Vec::new();
        let (mut hits, mut exacts, mut loops, mut lookup_hits) = (0usize, 0usize, 0usize, 0usize);
        // Lookup bookkeeping (spec §5, G5 key-miss rate): how often the
        // ROUTER chose a lookup record (before a miss rewrote the target
        // to the backbone), how many of those found no key, how many hits
        // answered from the entry the row's `src` names.
        let (mut lookup_targets, mut lookup_misses, mut lookup_off) = (0usize, 0usize, 0usize);
        let (mut lookup_src_known, mut lookup_src_matches) = (0usize, 0usize);
        let mut route_counts: std::collections::BTreeMap<String, usize> = Default::default();
        let mut decided_counts: std::collections::BTreeMap<String, usize> = Default::default();
        // Who sent the lookup-target rows: the router, or the `key_first`
        // policy over a backbone decision (sums to `lookup_targets`).
        let mut decided_by_counts: std::collections::BTreeMap<String, usize> = Default::default();
        for row in &rows {
            let (decision, outcome) = lanes.decide_lookup(&mode, &row.prompt)?;
            *route_counts
                .entry(decision.target_label().to_string())
                .or_default() += 1;
            *decided_counts
                .entry(
                    outcome
                        .lookup_id()
                        .unwrap_or(decision.target_label())
                        .to_string(),
                )
                .or_default() += 1;
            if mode != RouteMode::None && !a.json {
                eprintln!("{}: {}", row.label, crate::knowledge::describe(&decision));
                if let Some(line) = outcome.describe() {
                    eprintln!("{}: {line}", row.label);
                }
            }
            if let Some(by) = outcome.decided_by() {
                *decided_by_counts.entry(by.label().to_string()).or_default() += 1;
            }
            lookup_hits += outcome.is_hit() as usize;
            lookup_targets += outcome.lookup_id().is_some() as usize;
            lookup_misses += matches!(outcome, LookupOutcome::Miss { .. }) as usize;
            lookup_off += matches!(outcome, LookupOutcome::Off { .. }) as usize;
            let (src_entry, src_match) = match (outcome.hit(), row.src.as_deref()) {
                (Some(ans), Some(src)) => {
                    let e = lanes.lookup_entry_of(&ans.id, src)?;
                    (e, e.map(|e| e == ans.key.entry))
                }
                _ => (None, None),
            };
            lookup_src_known += src_entry.is_some() as usize;
            lookup_src_matches += (src_match == Some(true)) as usize;
            let mut route_json = decision.summary_json();
            outcome.annotate(&mut route_json, lookup_mode);
            if outcome.is_hit() {
                route_json["lookup_src_entry"] = serde_json::json!(src_entry);
                route_json["lookup_src_match"] = serde_json::json!(src_match);
            }
            // A lookup hit in `answer` mode: the table's text IS the
            // answer — nothing is generated, no first-token distribution.
            // Everything else runs a lane: the plain prompt, or the
            // card-prepended one in `context` mode.
            struct Gen {
                answer: String,
                out_ids: Vec<u32>,
                t5: Vec<(u32, String, f64)>,
                prompt_tokens: usize,
                terminal_newline: Option<u32>,
                ttft_ms: f64,
                tok_s: f64,
                finish_reason: String,
            }
            let g = if let LookupOutcome::Answer(ans) = &outcome {
                Gen {
                    answer: ans.text.clone(),
                    out_ids: Vec::new(),
                    t5: Vec::new(),
                    prompt_tokens: 0,
                    terminal_newline: None,
                    ttft_ms: 0.0,
                    tok_s: 0.0,
                    finish_reason: "lookup".into(),
                }
            } else {
                let pipeline = lanes.lane(&decision.target)?;
                let text = chat_prefix(&outcome.generation_text(&row.prompt));
                let mut ids = pipeline.tokenizer.encode(&text);
                if a.bos {
                    ids = pipeline.tokenizer.with_bos(ids);
                }
                let terminal_newline = ids.last().copied();
                // First assistant token: the distribution the contract asks for.
                let logits = pipeline
                    .forward_ids(&ids, None)
                    .map_err(|e| anyhow::anyhow!("{path}: forward_ids: {e}"))?;
                let t5: Vec<(u32, String, f64)> = top5(&logits)
                    .into_iter()
                    .map(|(id, p)| (id, pipeline.tokenizer.decode_token(id), p))
                    .collect();
                // Greedy generation with per-token stamps.
                let stamps: Arc<Mutex<Vec<Instant>>> = Arc::default();
                let st = stamps.clone();
                let cb: cortiq_engine::TokenCallback = Box::new(move |_t| {
                    st.lock().unwrap().push(Instant::now());
                    true
                });
                let t0 = Instant::now();
                let r = pipeline
                    .generate_from_ids(&ids, a.max_tokens, None, Some(cb))
                    .map_err(|e| anyhow::anyhow!("{path}: generate: {e}"))?;
                let stamps = stamps.lock().unwrap().clone();
                let ttft_ms = stamps
                    .first()
                    .map(|s| s.duration_since(t0).as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                let tok_s = if stamps.len() >= 2 {
                    (stamps.len() - 1) as f64
                        / stamps[stamps.len() - 1]
                            .duration_since(stamps[0])
                            .as_secs_f64()
                            .max(1e-9)
                } else {
                    0.0
                };
                let full = pipeline.tokenizer.decode(&r.token_ids);
                let answer = full
                    .split("<|im_end|>")
                    .next()
                    .unwrap_or("")
                    .split("<|endoftext|>")
                    .next()
                    .unwrap_or("")
                    .to_string();
                Gen {
                    answer,
                    out_ids: r.token_ids,
                    t5,
                    prompt_tokens: ids.len(),
                    terminal_newline,
                    ttft_ms,
                    tok_s,
                    finish_reason: r.finish_reason,
                }
            };
            let Gen {
                answer,
                out_ids,
                t5,
                prompt_tokens,
                terminal_newline,
                ttft_ms,
                tok_s,
                finish_reason,
            } = g;
            let out_ids = &out_ids;
            let first_line = answer.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
            let exact = row
                .expected
                .iter()
                .any(|e| normalize(first_line) == normalize(e));
            let keyword = row
                .expected
                .iter()
                .any(|e| answer.to_lowercase().contains(&e.to_lowercase()));
            let tail = if out_ids.len() > 128 { &out_ids[out_ids.len() - 128..] } else { &out_ids[..] };
            let distinct5 = distinct_5gram_ratio(tail);
            let (period, run) = longest_periodic_run(tail, 16);
            let looped = run >= 64;
            hits += (exact || keyword) as usize;
            exacts += exact as usize;
            loops += looped as usize;
            if !a.json {
                let t5s: Vec<String> = t5
                    .iter()
                    .map(|(id, s, p)| format!("{id}:{:?}={p:.3}", s))
                    .collect();
                println!(
                    "-- {} | exact={} keyword={} | distinct5={distinct5:.3} period={period} run={run} loop={} | ttft {ttft_ms:.1} ms, {tok_s:.1} tok/s, {} tokens, finish={} | top5 {}",
                    row.label,
                    exact as u8,
                    keyword as u8,
                    looped as u8,
                    out_ids.len(),
                    finish_reason,
                    t5s.join(" ")
                );
                let preview: String = answer.chars().take(240).collect();
                println!("   answer: {:?}", preview);
            }
            per_prompt.push(serde_json::json!({
                "label": row.label,
                "expected": row.expected,
                "prompt_tokens": prompt_tokens,
                "terminal_token": terminal_newline,
                "exact": exact,
                "keyword": keyword,
                "distinct5_last128": distinct5,
                "loop_period": period,
                "loop_run": run,
                "loop": looped,
                "top5": t5.iter().map(|(id, s, p)| serde_json::json!({"id": id, "token": s, "p": p})).collect::<Vec<_>>(),
                "ttft_ms": ttft_ms,
                "tok_s": tok_s,
                "generated": out_ids.len(),
                "finish_reason": finish_reason,
                "answer": answer,
                "route": route_json,
                "lookup_hit": outcome.is_hit(),
            }));
        }
        if !a.json {
            println!(
                "== {path}: exact {exacts}/{}, exact_or_keyword {hits}/{}, loops {loops}/{} | routes {:?} (decided {:?}, by {:?}) | lookup targets {lookup_targets}, hits {lookup_hits}, key misses {lookup_misses}, src matches {lookup_src_matches}/{lookup_src_known}",
                rows.len(),
                rows.len(),
                rows.len(),
                route_counts,
                decided_counts,
                decided_by_counts
            );
        }
        all.push(serde_json::json!({
            "model": path,
            "backend": backend,
            "max_tokens": a.max_tokens,
            "rep_penalty": a.rep_penalty,
            "bos": a.bos,
            "prompts": rows.len(),
            "exact": exacts,
            "exact_or_keyword": hits,
            "loops": loops,
            "route_mode": mode.label(),
            "include_quarantine": a.include_quarantine,
            // the lane that ran each prompt (a lookup miss ran the backbone)
            "route_counts": route_counts,
            // the decision's choice (the router's, or a `key_first` record's
            // over a backbone decision), before the lookup step rewrote a miss
            "decided_counts": decided_counts,
            // lookup-target rows by who sent them: "router" | "key_first"
            "decided_by_counts": decided_by_counts,
            "lookup_mode": lookup_mode.label(),
            "lookup_targets": lookup_targets,
            "lookup_hits": lookup_hits,
            "lookup_misses": lookup_misses,
            "lookup_off": lookup_off,
            "lookup_src_known": lookup_src_known,
            "lookup_src_matches": lookup_src_matches,
            "per_prompt": per_prompt,
        }));
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&all)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prompts_parse_to_eleven_rows_with_the_audit_labels() {
        let rows = parse_prompts(DEFAULT_PROMPTS).unwrap();
        assert_eq!(rows.len(), 11);
        assert_eq!(rows[0].label, "EN_FACT");
        assert_eq!(rows[0].expected, vec!["Paris".to_string()]);
        assert_eq!(rows[10].label, "MIXED");
        assert!(chat_prefix(&rows[0].prompt).ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn jsonl_prompts_take_expect_arrays_as_keywords() {
        let rows = parse_prompts_jsonl(
            "{\"lang\":\"ru\",\"prompt\":\"К какому семейству относится Кливия?\",\"expect\":[\"Амариллисовые\",\"Amaryllidaceae\"],\"src\":\"Кливия\"}\n\n{\"lang\":\"en\",\"prompt\":\"x\",\"expect\":\"y\"}\n",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "ru:Кливия");
        assert_eq!(rows[0].expected, vec!["Амариллисовые", "Amaryllidaceae"]);
        assert_eq!(rows[1].label, "en:3");
        assert_eq!(rows[1].expected, vec!["y"]);
    }

    #[test]
    fn loop_metrics_flag_periodic_tails_and_pass_varied_ones() {
        let looped: Vec<u32> = (0..128).map(|i| (i % 3) as u32 + 7).collect();
        let (p, run) = longest_periodic_run(&looped, 16);
        assert_eq!(p, 3);
        assert!(run >= 64);
        assert!(distinct_5gram_ratio(&looped) < 0.05);
        let varied: Vec<u32> = (0..128).map(|i| (i * 7919 % 1009) as u32).collect();
        let (_, run) = longest_periodic_run(&varied, 16);
        assert!(run < 64);
        assert!(distinct_5gram_ratio(&varied) > 0.9);
    }
}
