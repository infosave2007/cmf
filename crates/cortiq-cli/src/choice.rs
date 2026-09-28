//! `cortiq probe-choice` — a FORCED-CHOICE fact probe for small models
//! whose free generation is too weak for the keyword matching of
//! `probe-utility`.
//!
//! Every prompt is rendered as the corrected cmf-im-v1 chat prefix
//! (`utility::chat_prefix`: `<|im_start|>user\n{prompt}<|im_end|>\n
//! <|im_start|>assistant\n`, no BOS unless `--bos`), forwarded ONCE, and
//! every candidate string `c` is scored as `log P(tokens(c) | prefix)` —
//! the sum of its token log-probs (`--norm mean`: the per-token mean).
//! The candidates share the prefix state: they are laid out as a token
//! trie, the host KV state (attention KV, GDN recurrent state, bounded
//! anchor, O(1) skeleton — everything `LayerKvCache` owns) is snapshotted
//! at every branching node and rolled back before each sibling, so one
//! decode step is spent per internal trie node and none per leaf.
//! `--scorer full` re-forwards `prefix + candidate` per candidate through
//! the engine's own `nll_ids_from` — the slow cross-check of the shared
//! walk (the test suite asserts both agree).
//!
//! Prediction = argmax over the candidates; a row is correct iff the
//! prediction equals `expect[0]` (case-insensitive, trimmed). The report
//! carries top-1 / top-5 accuracy against the majority-class baseline
//! (the most frequent `expect[0]` of the file) and chance (1 / #candidates),
//! the mean log-prob of the correct candidate against the best wrong one,
//! a per-`lang` breakdown and the per-row detail.
//!
//! Runs per-op on the host route (the KV snapshot lives on the host);
//! `CMF_GPU`, `CMF_GROWTH` and `CMF_GROWTH_SHELL` come from the
//! environment exactly as for `probe-utility` — growth records and their
//! shells apply.

use crate::knowledge::{Lanes, RouteMode};
use crate::utility::chat_prefix;
use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_engine::Pipeline;
use cortiq_engine::SamplerConfig;
use cortiq_engine::router::RouteOptions;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct ChoiceArgs<'a> {
    pub model: &'a str,
    pub prompts_jsonl: &'a str,
    /// One `{"text": …}` per line; None = the distinct `expect[0]` of the
    /// prompt set.
    pub candidates_jsonl: Option<&'a str>,
    /// `sum` | `mean`.
    pub norm: &'a str,
    /// `shared` | `full`.
    pub scorer: &'a str,
    pub bos: bool,
    /// Cloze mode: the prompt is a raw text prefix (no chat frame) and
    /// every candidate is scored with a leading space (" Pinaceae").
    pub raw: bool,
    pub route: Option<&'a str>,
    pub include_quarantine: bool,
    pub json: bool,
}

// ───────────────────────── inputs ─────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceRow {
    pub lang: String,
    pub src: String,
    pub prompt: String,
    /// `expect[0]` is the answer; the rest are aliases (kept for the report).
    pub expect: Vec<String>,
}

/// JSONL rows `{"lang","prompt","expect":[answer, …],"src"}`; the answer
/// (`expect[0]`, or a string `expect`) is required — a forced choice has
/// nothing to score without it.
pub fn parse_rows(text: &str) -> anyhow::Result<Vec<ChoiceRow>> {
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
        let expect: Vec<String> = match v.get("expect") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|e| e.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect(),
            Some(serde_json::Value::String(e)) if !e.trim().is_empty() => {
                vec![e.trim().to_string()]
            }
            _ => Vec::new(),
        };
        anyhow::ensure!(
            !expect.is_empty(),
            "prompts jsonl line {}: no \"expect\" answer (a forced choice needs expect[0])",
            ln + 1
        );
        let lang = v
            .get("lang")
            .and_then(|l| l.as_str())
            .unwrap_or("?")
            .to_string();
        let src = v
            .get("src")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}", ln + 1));
        rows.push(ChoiceRow {
            lang,
            src,
            prompt,
            expect,
        });
    }
    anyhow::ensure!(!rows.is_empty(), "prompt set is empty");
    Ok(rows)
}

/// The comparison key of a candidate / answer: trimmed, lower-cased.
pub fn norm_key(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Keep the first spelling of every distinct key, in first-seen order.
fn dedupe(texts: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for t in texts {
        let t = t.trim().to_string();
        if t.is_empty() {
            continue;
        }
        if seen.insert(norm_key(&t)) {
            out.push(t);
        }
    }
    out
}

/// `--candidates-jsonl`: one `{"text": …}` per line (a bare JSON string
/// is accepted too), de-duplicated by key, first-seen order.
pub fn parse_candidates_jsonl(text: &str) -> anyhow::Result<Vec<String>> {
    let mut texts = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("candidates jsonl line {}", ln + 1))?;
        let t = match &v {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(o) => o
                .get("text")
                .and_then(|t| t.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("candidates jsonl line {}: no \"text\"", ln + 1)
                })?
                .to_string(),
            _ => anyhow::bail!(
                "candidates jsonl line {}: expected {{\"text\": …}}",
                ln + 1
            ),
        };
        texts.push(t);
    }
    let out = dedupe(texts);
    anyhow::ensure!(!out.is_empty(), "candidate set is empty");
    Ok(out)
}

/// The default candidate set: the distinct `expect[0]` of the rows.
pub fn default_candidates(rows: &[ChoiceRow]) -> Vec<String> {
    dedupe(rows.iter().map(|r| r.expect[0].clone()))
}

/// The majority-class baseline: `(spelling, count)` of the most frequent
/// answer key (ties → the first seen).
pub fn majority_class(rows: &[ChoiceRow]) -> (String, usize) {
    let mut counts: Vec<(String, String, usize)> = Vec::new();
    for r in rows {
        let key = norm_key(&r.expect[0]);
        match counts.iter_mut().find(|(k, _, _)| *k == key) {
            Some((_, _, n)) => *n += 1,
            None => counts.push((key, r.expect[0].trim().to_string(), 1)),
        }
    }
    counts
        .into_iter()
        .fold(None::<(String, usize)>, |best, (_, s, n)| match best {
            Some((_, bn)) if bn >= n => best,
            _ => Some((s, n)),
        })
        .unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    Sum,
    Mean,
}

impl Norm {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "mean" | "avg" => Ok(Self::Mean),
            other => anyhow::bail!("--norm {other}: expected sum | mean"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scorer {
    Shared,
    Full,
}

impl Scorer {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "shared" => Ok(Self::Shared),
            "full" => Ok(Self::Full),
            other => anyhow::bail!("--scorer {other}: expected shared | full"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Full => "full",
        }
    }
}

// ───────────────────────── scoring ─────────────────────────

/// The token trie of the candidate set: node 0 is the root (the prefix),
/// an edge is one candidate token, `members[node]` lists the candidates
/// whose path runs through the node.
#[derive(Debug, Clone)]
pub struct Trie {
    pub children: Vec<Vec<(u32, usize)>>,
    pub members: Vec<Vec<usize>>,
}

impl Trie {
    pub fn build(cands: &[Vec<u32>]) -> Self {
        let mut t = Trie {
            children: vec![Vec::new()],
            members: vec![Vec::new()],
        };
        for (ci, toks) in cands.iter().enumerate() {
            let mut node = 0usize;
            t.members[0].push(ci);
            for &tok in toks {
                let next = match t.children[node].iter().find(|(x, _)| *x == tok) {
                    Some(&(_, c)) => c,
                    None => {
                        let c = t.children.len();
                        t.children.push(Vec::new());
                        t.members.push(Vec::new());
                        t.children[node].push((tok, c));
                        c
                    }
                };
                t.members[next].push(ci);
                node = next;
            }
        }
        t
    }

    /// Decode steps the shared walk spends per prompt: one per internal
    /// node below the root (a leaf's last token is never forwarded).
    pub fn decode_steps(&self) -> usize {
        (1..self.children.len())
            .filter(|&n| !self.children[n].is_empty())
            .count()
    }
}

/// `log Σ exp(logits)` in f64.
pub fn log_sum_exp(logits: &[f32]) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    logits
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
        + max
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandScore {
    /// `log P(tokens | prefix)`, natural log.
    pub lp_sum: f64,
    pub n_tokens: usize,
}

impl CandScore {
    pub fn value(&self, norm: Norm) -> f64 {
        match norm {
            Norm::Sum => self.lp_sum,
            Norm::Mean => self.lp_sum / self.n_tokens.max(1) as f64,
        }
    }
}

/// Depth-first walk of the trie from `node`, whose next-token `logits`
/// the pipeline has just produced at sequence position `pos` (the cache
/// holds `[0, pos)`). Accumulates every edge's log-prob into the
/// candidates below it; the host KV state is snapshotted at the node
/// before the first expansion and rolled back before every further one.
fn walk(p: &mut Pipeline, trie: &Trie, node: usize, logits: &[f32], pos: usize, acc: &mut [f64]) {
    let kids = &trie.children[node];
    if kids.is_empty() {
        return;
    }
    let lse = log_sum_exp(logits);
    let mut snap = None;
    let mut dirty = false;
    for &(tok, child) in kids {
        let lp = logits[tok as usize] as f64 - lse;
        for &m in &trie.members[child] {
            acc[m] += lp;
        }
        if trie.children[child].is_empty() {
            continue;
        }
        if snap.is_none() {
            snap = Some(p.kv_cache.layers.clone());
        }
        if dirty {
            p.kv_cache
                .layers
                .clone_from(snap.as_ref().expect("snapshot taken"));
        }
        let next = p.decode_step_logits(tok, pos);
        dirty = true;
        walk(p, trie, child, &next, pos + 1, acc);
    }
}

/// One prefix forward, then the trie walk with KV rollback.
pub fn score_shared(
    p: &mut Pipeline,
    prefix: &[u32],
    cands: &[Vec<u32>],
) -> anyhow::Result<Vec<CandScore>> {
    let trie = Trie::build(cands);
    p.reset_session();
    let logits = p
        .forward_ids(prefix, None)
        .map_err(|e| anyhow::anyhow!("forward_ids: {e}"))?;
    let mut acc = vec![0f64; cands.len()];
    walk(p, &trie, 0, &logits, prefix.len(), &mut acc);
    Ok(cands
        .iter()
        .zip(acc)
        .map(|(c, lp_sum)| CandScore {
            lp_sum,
            n_tokens: c.len(),
        })
        .collect())
}

/// A fresh forward of `prefix + candidate` per candidate through the
/// engine's `nll_ids_from` (positions `≥ prefix.len() − 1` predict the
/// candidate tokens).
pub fn score_full(
    p: &mut Pipeline,
    prefix: &[u32],
    cands: &[Vec<u32>],
) -> anyhow::Result<Vec<CandScore>> {
    let mut out = Vec::with_capacity(cands.len());
    for toks in cands {
        let mut all = Vec::with_capacity(prefix.len() + toks.len());
        all.extend_from_slice(prefix);
        all.extend_from_slice(toks);
        let (nll, cnt) = p
            .nll_ids_from(&all, prefix.len() - 1)
            .map_err(|e| anyhow::anyhow!(e))?;
        anyhow::ensure!(
            cnt == toks.len(),
            "nll_ids_from scored {cnt} positions for a {}-token candidate",
            toks.len()
        );
        out.push(CandScore {
            lp_sum: -nll,
            n_tokens: toks.len(),
        });
    }
    Ok(out)
}

// ───────────────────────── ranking ─────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Ranked {
    /// Candidate indices, best first (ties → the lower index).
    pub order: Vec<usize>,
    pub predicted: usize,
    /// 1-based rank of the correct candidate (None: not in the set).
    pub rank_of_correct: Option<usize>,
    pub lp_correct: Option<f64>,
    pub lp_best: f64,
    /// The best score among the candidates other than the correct one.
    pub lp_best_wrong: Option<f64>,
}

/// Rank `scores` (higher is better); `correct` is the index of the
/// expected candidate when it is in the set.
pub fn rank(scores: &[f64], correct: Option<usize>) -> Ranked {
    assert!(!scores.is_empty(), "rank: no candidates");
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let predicted = order[0];
    let rank_of_correct = correct.and_then(|c| order.iter().position(|&i| i == c).map(|r| r + 1));
    let lp_best_wrong = order
        .iter()
        .find(|&&i| Some(i) != correct)
        .map(|&i| scores[i]);
    Ranked {
        predicted,
        rank_of_correct,
        lp_correct: correct.map(|c| scores[c]),
        lp_best: scores[predicted],
        lp_best_wrong,
        order,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowResult {
    pub index: usize,
    pub lang: String,
    pub src: String,
    pub expect: Vec<String>,
    pub predicted: String,
    pub correct: bool,
    pub rank_of_correct: Option<usize>,
    pub lp_correct: Option<f64>,
    pub lp_best: f64,
    pub lp_best_wrong: Option<f64>,
    pub tokens_correct: Option<usize>,
    pub tokens_predicted: usize,
    pub route: serde_json::Value,
}

fn mean(xs: impl Iterator<Item = f64>) -> Option<f64> {
    let (s, n) = xs.fold((0f64, 0usize), |(s, n), x| (s + x, n + 1));
    (n > 0).then(|| s / n as f64)
}

fn block(rows: &[&RowResult]) -> serde_json::Value {
    let n = rows.len();
    let top1 = rows.iter().filter(|r| r.correct).count();
    let top5 = rows
        .iter()
        .filter(|r| r.rank_of_correct.is_some_and(|k| k <= 5))
        .count();
    let scored: Vec<&&RowResult> = rows.iter().filter(|r| r.lp_correct.is_some()).collect();
    serde_json::json!({
        "n": n,
        "top1": top1,
        "top5": top5,
        "top1_acc": if n > 0 { top1 as f64 / n as f64 } else { 0.0 },
        "top5_acc": if n > 0 { top5 as f64 / n as f64 } else { 0.0 },
        "mean_lp_correct": mean(scored.iter().filter_map(|r| r.lp_correct)),
        "mean_lp_best_wrong": mean(scored.iter().filter_map(|r| r.lp_best_wrong)),
        "mean_margin": mean(scored.iter().filter_map(|r| Some(r.lp_correct? - r.lp_best_wrong?))),
        "mean_rank_of_correct": mean(scored.iter().filter_map(|r| r.rank_of_correct.map(|k| k as f64))),
    })
}

/// The report: totals, baselines, per-lang, per-row.
pub fn summarize(
    rows: &[RowResult],
    candidates: &[String],
    majority: &(String, usize),
) -> serde_json::Value {
    let n = rows.len();
    let all: Vec<&RowResult> = rows.iter().collect();
    let mut v = block(&all);
    let mut per_lang: BTreeMap<String, Vec<&RowResult>> = BTreeMap::new();
    for r in rows {
        per_lang.entry(r.lang.clone()).or_default().push(r);
    }
    let per_lang: BTreeMap<String, serde_json::Value> = per_lang
        .into_iter()
        .map(|(l, rs)| (l, block(&rs)))
        .collect();
    v["candidates"] = serde_json::json!(candidates.len());
    v["chance"] = serde_json::json!(if candidates.is_empty() {
        0.0
    } else {
        1.0 / candidates.len() as f64
    });
    v["majority"] = serde_json::json!({
        "label": majority.0,
        "n": majority.1,
        "acc": if n > 0 { majority.1 as f64 / n as f64 } else { 0.0 },
    });
    v["expect_outside_candidates"] =
        serde_json::json!(rows.iter().filter(|r| r.rank_of_correct.is_none()).count());
    v["per_lang"] = serde_json::to_value(per_lang).expect("per_lang serializes");
    v["per_row"] = serde_json::Value::Array(
        rows.iter()
            .map(|r| {
                serde_json::json!({
                    "index": r.index,
                    "lang": r.lang,
                    "src": r.src,
                    "expect": r.expect,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "rank_of_correct": r.rank_of_correct,
                    "lp_correct": r.lp_correct,
                    "lp_best": r.lp_best,
                    "lp_best_wrong": r.lp_best_wrong,
                    "tokens_correct": r.tokens_correct,
                    "tokens_predicted": r.tokens_predicted,
                    "route": r.route,
                })
            })
            .collect(),
    );
    v
}

// ───────────────────────── the command ─────────────────────────

pub fn cmd_probe_choice(a: ChoiceArgs<'_>) -> anyhow::Result<()> {
    let norm = Norm::parse(a.norm)?;
    let scorer = Scorer::parse(a.scorer)?;
    let rows = parse_rows(
        &std::fs::read_to_string(a.prompts_jsonl).with_context(|| a.prompts_jsonl.to_string())?,
    )?;
    let candidates = match a.candidates_jsonl {
        Some(path) => {
            parse_candidates_jsonl(&std::fs::read_to_string(path).with_context(|| path.to_string())?)?
        }
        None => default_candidates(&rows),
    };
    let cand_index: std::collections::HashMap<String, usize> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (norm_key(c), i))
        .collect();
    let majority = majority_class(&rows);
    let backend = format!(
        "CMF_GPU={} resident={}",
        std::env::var("CMF_GPU").unwrap_or_else(|_| "unset".into()),
        std::env::var("CMF_EMBRYO_RESIDENT").unwrap_or_else(|_| "unset".into())
    );
    let growth = cortiq_engine::loader::growth_mode().label();
    let shell = if cortiq_engine::pipeline::growth_shell_enabled() { "on" } else { "off" };

    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    let mode = RouteMode::resolve(a.route, &model)?;
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
    .per_op(true);

    // Candidate token ids per lane (a skill lane shares the tokenizer, but
    // the map is keyed by lane label to stay correct if one ever differs).
    let mut cand_ids_by_lane: BTreeMap<String, (Vec<Vec<u32>>, usize)> = BTreeMap::new();

    if !a.json {
        println!(
            "probe-choice: {} | {} prompts, {} candidates, norm={}, scorer={}, bos={} | route={} | growth={growth} shell={shell} | {backend}",
            a.model,
            rows.len(),
            candidates.len(),
            norm.label(),
            scorer.label(),
            a.bos,
            mode.label()
        );
        println!(
            "baselines: majority {:?} {}/{} = {:.4}, chance 1/{} = {:.4}",
            majority.0,
            majority.1,
            rows.len(),
            majority.1 as f64 / rows.len() as f64,
            candidates.len(),
            1.0 / candidates.len() as f64
        );
    }
    let t0 = std::time::Instant::now();
    let mut results: Vec<RowResult> = Vec::with_capacity(rows.len());
    let mut route_counts: BTreeMap<String, usize> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        let decision = lanes.decide(&mode, &row.prompt)?;
        let lane_label = decision.target_label().to_string();
        *route_counts.entry(lane_label.clone()).or_default() += 1;
        let p = lanes.lane(&decision.target)?;
        if !cand_ids_by_lane.contains_key(&lane_label) {
            let mut ids = Vec::with_capacity(candidates.len());
            for c in &candidates {
                let t = if a.raw { p.tokenizer.encode(&format!(" {c}")) } else { p.tokenizer.encode(c) };
                anyhow::ensure!(!t.is_empty(), "candidate {c:?} tokenizes to nothing");
                if let Some(&bad) = t.iter().find(|&&x| x as usize >= p.vocab_size) {
                    anyhow::bail!(
                        "candidate {c:?}: token {bad} outside the vocab ({})",
                        p.vocab_size
                    );
                }
                ids.push(t);
            }
            let steps = Trie::build(&ids).decode_steps();
            if !a.json {
                println!(
                    "lane {lane_label}: {} candidate tokens over {} candidates, {steps} shared decode steps per prompt",
                    ids.iter().map(|t| t.len()).sum::<usize>(),
                    ids.len()
                );
            }
            cand_ids_by_lane.insert(lane_label.clone(), (ids, steps));
        }
        let (cand_ids, _) = cand_ids_by_lane.get(&lane_label).expect("lane candidates");
        let mut prefix = if a.raw {
            p.tokenizer.encode(&row.prompt)
        } else {
            p.tokenizer.encode(&chat_prefix(&row.prompt))
        };
        if a.bos {
            prefix = p.tokenizer.with_bos(prefix);
        }
        anyhow::ensure!(!prefix.is_empty(), "row {i}: empty prefix");
        let scores = match scorer {
            Scorer::Shared => score_shared(p, &prefix, cand_ids)?,
            Scorer::Full => score_full(p, &prefix, cand_ids)?,
        };
        let values: Vec<f64> = scores.iter().map(|s| s.value(norm)).collect();
        let correct = cand_index.get(&norm_key(&row.expect[0])).copied();
        let r = rank(&values, correct);
        let res = RowResult {
            index: i,
            lang: row.lang.clone(),
            src: row.src.clone(),
            expect: row.expect.clone(),
            predicted: candidates[r.predicted].clone(),
            correct: correct == Some(r.predicted),
            rank_of_correct: r.rank_of_correct,
            lp_correct: r.lp_correct,
            lp_best: r.lp_best,
            lp_best_wrong: r.lp_best_wrong,
            tokens_correct: correct.map(|c| scores[c].n_tokens),
            tokens_predicted: scores[r.predicted].n_tokens,
            route: decision.summary_json(),
        };
        if !a.json {
            println!(
                "-- [{i}] {}:{} | expect {:?} | predicted {:?} | rank {} | lp_correct {} lp_best {:.3}",
                res.lang,
                res.src,
                row.expect[0],
                res.predicted,
                res.rank_of_correct
                    .map(|k| k.to_string())
                    .unwrap_or_else(|| "-".into()),
                res.lp_correct
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".into()),
                res.lp_best
            );
        } else if (i + 1) % 20 == 0 || i + 1 == rows.len() {
            eprintln!(
                "probe-choice: {}/{} rows, {:.1} s",
                i + 1,
                rows.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        results.push(res);
    }
    let mut summary = summarize(&results, &candidates, &majority);
    summary["model"] = serde_json::json!(a.model);
    summary["prompts"] = serde_json::json!(a.prompts_jsonl);
    summary["candidates_source"] = serde_json::json!(a
        .candidates_jsonl
        .map(|s| s.to_string())
        .unwrap_or_else(|| "expect[0] of the prompt set".into()));
    summary["candidate_list"] = serde_json::json!(candidates);
    summary["norm"] = serde_json::json!(norm.label());
    summary["scorer"] = serde_json::json!(scorer.label());
    summary["bos"] = serde_json::json!(a.bos);
    summary["raw"] = serde_json::json!(a.raw);
    summary["route_mode"] = serde_json::json!(mode.label());
    summary["include_quarantine"] = serde_json::json!(a.include_quarantine);
    summary["route_counts"] = serde_json::to_value(&route_counts)?;
    summary["backend"] = serde_json::json!(backend);
    summary["growth"] = serde_json::json!(growth);
    summary["shell"] = serde_json::json!(shell);
    summary["seconds"] = serde_json::json!(t0.elapsed().as_secs_f64());
    if a.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "== {}: top1 {}/{} = {:.4}, top5 {}/{} = {:.4} | majority {:.4} ({:?}) | chance {:.4} | mean lp correct {} vs best wrong {} | routes {:?} | {:.1} s",
            a.model,
            summary["top1"],
            rows.len(),
            summary["top1_acc"].as_f64().unwrap_or(0.0),
            summary["top5"],
            rows.len(),
            summary["top5_acc"].as_f64().unwrap_or(0.0),
            summary["majority"]["acc"].as_f64().unwrap_or(0.0),
            majority.0,
            summary["chance"].as_f64().unwrap_or(0.0),
            summary["mean_lp_correct"],
            summary["mean_lp_best_wrong"],
            route_counts,
            t0.elapsed().as_secs_f64()
        );
        for (lang, b) in summary["per_lang"].as_object().into_iter().flatten() {
            println!(
                "   {lang}: n {} top1 {:.4} top5 {:.4} mean lp correct {} vs best wrong {}",
                b["n"],
                b["top1_acc"].as_f64().unwrap_or(0.0),
                b["top5_acc"].as_f64().unwrap_or(0.0),
                b["mean_lp_correct"],
                b["mean_lp_best_wrong"]
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(lang: &str, src: &str, expect: &[&str]) -> ChoiceRow {
        ChoiceRow {
            lang: lang.into(),
            src: src.into(),
            prompt: format!("What family is {src}?"),
            expect: expect.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn rows_need_an_answer_and_keep_lang_src() {
        let rows = parse_rows(
            "{\"lang\":\"ru\",\"prompt\":\"К какому семейству относится Пихта?\",\"expect\":[\"Pinaceae\",\"Сосновые\"],\"src\":\"Abies balsamea\"}\n\n{\"prompt\":\"x\",\"expect\":\" Fabaceae \"}\n",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].lang, "ru");
        assert_eq!(rows[0].src, "Abies balsamea");
        assert_eq!(rows[0].expect, vec!["Pinaceae", "Сосновые"]);
        assert_eq!(rows[1].lang, "?");
        assert_eq!(rows[1].src, "3");
        assert_eq!(rows[1].expect, vec!["Fabaceae"]);
        let err = parse_rows("{\"prompt\":\"x\",\"expect\":[]}\n").unwrap_err();
        assert!(err.to_string().contains("expect[0]"), "{err}");
    }

    #[test]
    fn candidates_default_to_distinct_answers_and_parse_from_jsonl() {
        let rows = vec![
            row("ru", "a", &["Pinaceae", "Сосновые"]),
            row("en", "b", &["Fabaceae", "Бобовые"]),
            row("ru", "c", &["fabaceae"]),
            row("en", "d", &[" Asteraceae "]),
            row("en", "e", &["Fabaceae"]),
        ];
        assert_eq!(default_candidates(&rows), vec!["Pinaceae", "Fabaceae", "Asteraceae"]);
        assert_eq!(majority_class(&rows), ("Fabaceae".to_string(), 3));
        let c = parse_candidates_jsonl(
            "{\"text\":\"Rosaceae\"}\n\"Lamiaceae\"\n{\"text\":\" rosaceae\"}\n\n{\"text\":\"Apiaceae\",\"n\":1}\n",
        )
        .unwrap();
        assert_eq!(c, vec!["Rosaceae", "Lamiaceae", "Apiaceae"]);
        assert!(parse_candidates_jsonl("{\"name\":\"x\"}\n").is_err());
        assert!(parse_candidates_jsonl("\n").is_err());
    }

    #[test]
    fn trie_shares_prefixes_and_counts_internal_nodes() {
        // A=[1,2,3] B=[1,2,4] C=[1,5] D=[6]; nodes in creation order:
        // 1=[1] 2=[1,2] 3=[1,2,3] 4=[1,2,4] 5=[1,5] 6=[6].
        let t = Trie::build(&[vec![1, 2, 3], vec![1, 2, 4], vec![1, 5], vec![6]]);
        assert_eq!(t.children[0], vec![(1, 1), (6, 6)]);
        assert_eq!(t.members[0], vec![0, 1, 2, 3]);
        assert_eq!(t.members[1], vec![0, 1, 2]);
        assert_eq!(t.children[1], vec![(2, 2), (5, 5)]);
        assert_eq!(t.members[2], vec![0, 1]);
        assert_eq!(t.members[5], vec![2]);
        assert_eq!(t.members[6], vec![3]);
        assert!(t.children[3].is_empty() && t.children[6].is_empty());
        // Internal nodes below the root: [1] and [1,2] — two decode steps
        // for four candidates of nine tokens.
        assert_eq!(t.decode_steps(), 2);
        assert_eq!(t.children.len(), 7);
    }

    #[test]
    fn log_sum_exp_matches_a_direct_softmax() {
        let l = [1.0f32, 2.0, 3.0, -4.0];
        let lse = log_sum_exp(&l);
        let direct: f64 = l.iter().map(|&v| (v as f64).exp()).sum::<f64>().ln();
        assert!((lse - direct).abs() < 1e-12);
        let p: f64 = l.iter().map(|&v| (v as f64 - lse).exp()).sum();
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ranking_orders_by_score_breaks_ties_by_index_and_finds_the_correct_one() {
        let r = rank(&[-3.0, -1.0, -1.0, -7.0], Some(3));
        assert_eq!(r.order, vec![1, 2, 0, 3]);
        assert_eq!(r.predicted, 1);
        assert_eq!(r.rank_of_correct, Some(4));
        assert_eq!(r.lp_correct, Some(-7.0));
        assert_eq!(r.lp_best, -1.0);
        assert_eq!(r.lp_best_wrong, Some(-1.0));
        let r = rank(&[-3.0, -1.0], Some(1));
        assert_eq!(r.rank_of_correct, Some(1));
        assert_eq!(r.lp_best_wrong, Some(-3.0));
        let r = rank(&[-3.0, -1.0], None);
        assert_eq!(r.rank_of_correct, None);
        assert_eq!(r.lp_correct, None);
        assert_eq!(r.lp_best_wrong, Some(-1.0));
        assert_eq!(
            CandScore { lp_sum: -6.0, n_tokens: 3 }.value(Norm::Mean),
            -2.0
        );
        assert_eq!(
            CandScore { lp_sum: -6.0, n_tokens: 3 }.value(Norm::Sum),
            -6.0
        );
    }

    #[test]
    fn summary_on_synthetic_log_probs() {
        let cands: Vec<String> = ["A", "B", "C", "D", "E", "F"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = vec![
            row("ru", "r1", &["A"]),
            row("ru", "r2", &["A"]),
            row("en", "e1", &["B"]),
            row("en", "e2", &["Z"]),
        ];
        // Scores per row over the six candidates.
        let scores: [[f64; 6]; 4] = [
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], // A first: correct
            [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0], // A last (rank 6): wrong, outside top5
            [-2.0, -1.5, -3.0, -4.0, -5.0, -6.0], // B first: correct
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], // Z not a candidate
        ];
        let idx = |s: &str| cands.iter().position(|c| norm_key(c) == norm_key(s));
        let results: Vec<RowResult> = rows
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (row, sc))| {
                let correct = idx(&row.expect[0]);
                let r = rank(sc, correct);
                RowResult {
                    index: i,
                    lang: row.lang.clone(),
                    src: row.src.clone(),
                    expect: row.expect.clone(),
                    predicted: cands[r.predicted].clone(),
                    correct: correct == Some(r.predicted),
                    rank_of_correct: r.rank_of_correct,
                    lp_correct: r.lp_correct,
                    lp_best: r.lp_best,
                    lp_best_wrong: r.lp_best_wrong,
                    tokens_correct: correct.map(|_| 1),
                    tokens_predicted: 1,
                    route: serde_json::Value::Null,
                }
            })
            .collect();
        let s = summarize(&results, &cands, &majority_class(&rows));
        assert_eq!(s["n"], 4);
        assert_eq!(s["top1"], 2);
        assert_eq!(s["top5"], 2);
        assert_eq!(s["top1_acc"], 0.5);
        assert_eq!(s["candidates"], 6);
        assert!((s["chance"].as_f64().unwrap() - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(s["majority"]["label"], "A");
        assert_eq!(s["majority"]["n"], 2);
        assert_eq!(s["majority"]["acc"], 0.5);
        assert_eq!(s["expect_outside_candidates"], 1);
        // Means over the three scorable rows: lp_correct −1, −6, −1.5;
        // best wrong −2, −1, −2.
        let mc = s["mean_lp_correct"].as_f64().unwrap();
        let mw = s["mean_lp_best_wrong"].as_f64().unwrap();
        assert!((mc - (-1.0 - 6.0 - 1.5) / 3.0).abs() < 1e-12, "{mc}");
        assert!((mw - (-2.0 - 1.0 - 2.0) / 3.0).abs() < 1e-12, "{mw}");
        assert!((s["mean_margin"].as_f64().unwrap() - (mc - mw)).abs() < 1e-12);
        assert_eq!(s["per_lang"]["ru"]["n"], 2);
        assert_eq!(s["per_lang"]["ru"]["top1"], 1);
        assert_eq!(s["per_lang"]["ru"]["top5"], 1);
        assert_eq!(s["per_lang"]["en"]["n"], 2);
        assert_eq!(s["per_lang"]["en"]["top1"], 1);
        let pr = s["per_row"].as_array().unwrap();
        assert_eq!(pr.len(), 4);
        assert_eq!(pr[1]["rank_of_correct"], 6);
        assert_eq!(pr[1]["predicted"], "F");
        assert_eq!(pr[3]["rank_of_correct"], serde_json::Value::Null);
        assert_eq!(pr[3]["correct"], false);
        assert_eq!(pr[2]["lp_best"], -1.5);
        assert_eq!(pr[2]["lp_best_wrong"], -2.0);
    }
}
