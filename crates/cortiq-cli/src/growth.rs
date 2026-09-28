//! `cortiq growth-eval` — gate data for `expert_append` growth records
//! (spec §2, the herbs experiment's G2 input): every prompt of a JSONL set
//! rendered as cmf-im-v1 (as probe-utility / dump-logits), the prefill and
//! `--max-tokens` greedy steps forwarded EXACTLY as `dump-logits --tokens`
//! forwards them (so a record here is a record there), and the MoE
//! routing counters snapshotted before / after → grown-expert wins per
//! layer. Per prompt `hit` = a grown expert won at least once anywhere in
//! the record; summary: hit rate, n, the exact Clopper–Pearson 95 % upper
//! bound, per-layer / per-record rates, per-`src` / per-`lang` breakdown;
//! `--indices-out` writes the no-hit indices for
//! `logits-compare --only-indices` (G2: those records are bit-identical
//! to F0's by construction — the gate measures that it holds).
//!
//! `--shell on|off|both` decides whether the growth shell applies
//! (`Resonance::scores`' −∞ rule); the default follows
//! `CMF_GROWTH_SHELL`. `--growth active|all|off` selects the records the
//! loader mounts (`CMF_GROWTH`); a quarantined record needs `all`.
//!
//! The routing counters live on the per-op path (the host `moe_route`;
//! the resident graph selects on the device and exports no counter), so
//! the whole-token / resident graphs are refused for this measurement and
//! `--path` accepts only `per-op`. The no-hit set is a statement about
//! THAT path: the graph's tree-reduced error is within parity of the
//! host's sequential sum, not equal to it, and a token near a shell or
//! near the best trunk score can win on one path and lose on the other.
//! The indices file records `path`, `dump-logits --path` records the
//! dump's in `<out>.meta.json`, and `logits-compare --only-indices`
//! refuses to compare across paths — G2 and G3 hold for the measured path
//! (spec §9.5.1).

use crate::knowledge::{EvalRow, ForwardPath, parse_eval_jsonl, sha256_hex};
use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_engine::loader::{growth_mode, mounted_growth_records};
use cortiq_engine::pipeline::{FfnKind, GrownExpert, growth_shell_enabled, set_growth_shell};
use cortiq_engine::router::clopper_pearson_upper;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct GrowthEvalArgs<'a> {
    pub model: &'a str,
    pub prompts_jsonl: &'a str,
    pub max_tokens: usize,
    pub json: bool,
    pub indices_out: Option<&'a str>,
    /// `on` | `off` | `both`; None = the environment (`CMF_GROWTH_SHELL`).
    pub shell: Option<&'a str>,
    /// `active` | `all` | `off`; None = the environment (`CMF_GROWTH`).
    pub growth: Option<&'a str>,
    /// `per-op` (the only measurable path; None = per-op) — the same
    /// flag as `dump-logits --path`, so a gate script passes one value
    /// to both tools.
    pub path: Option<&'a str>,
}

/// The MoE layers of a pipeline: layer → (trunk expert count, grown tail).
fn growth_layout(p: &Pipeline) -> BTreeMap<usize, (usize, Vec<GrownExpert>)> {
    p.weights
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, lw)| match &lw.ffn {
            FfnKind::Moe(m) => Some((li, (m.experts.len() - m.grown.len(), m.grown.clone()))),
            _ => None,
        })
        .collect()
}

/// Routing counters per MoE layer, padded to the expert count.
fn moe_stats(p: &Pipeline) -> BTreeMap<usize, Vec<u64>> {
    p.weights
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, lw)| match &lw.ffn {
            FfnKind::Moe(m) => {
                let mut st = m.stats.borrow().clone();
                st.resize(m.experts.len(), 0);
                Some((li, st))
            }
            _ => None,
        })
        .collect()
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

#[derive(Default, Clone)]
struct Tally {
    n: usize,
    hits: usize,
}

impl Tally {
    fn add(&mut self, hit: bool) {
        self.n += 1;
        self.hits += usize::from(hit);
    }
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "n": self.n,
            "hits": self.hits,
            "rate": if self.n > 0 { self.hits as f64 / self.n as f64 } else { f64::NAN },
            "hit_upper95": clopper_pearson_upper(self.hits, self.n, 0.95),
        })
    }
}

/// One prompt's routing under one shell mode.
struct PromptResult {
    index: usize,
    prompt_tokens: usize,
    /// layer → (grown wins, routed tokens)
    layers: BTreeMap<usize, (u64, u64)>,
    /// record id → wins
    by_record: BTreeMap<String, u64>,
    answer: String,
}

impl PromptResult {
    fn hit(&self) -> bool {
        self.layers.values().any(|(g, _)| *g > 0)
    }
}

/// Every prompt through one pipeline (one shell mode).
fn run_mode(
    model: &Arc<CmfModel>,
    rows: &[EvalRow],
    max_tokens: usize,
    shell_on: bool,
    quiet: bool,
) -> anyhow::Result<(BTreeMap<usize, (usize, Vec<GrownExpert>)>, Vec<PromptResult>)> {
    set_growth_shell(Some(shell_on));
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
    let mut p = Pipeline::from_model(model, greedy)?;
    // The counters are incremented where the host routes (`moe_route`);
    // a whole-token or resident graph selects on the device and counts
    // nothing — this measurement runs per-op.
    p.mark_graph_refused();
    let layout = growth_layout(&p);
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let ids = p
            .tokenizer
            .encode(&crate::utility::chat_prefix(&row.prompt));
        p.reset_session();
        let before = moe_stats(&p);
        let mut last = p
            .forward_ids(&ids, None)
            .map_err(|e| anyhow::anyhow!("prompt {i}: forward_ids: {e}"))?;
        let mut generated = Vec::with_capacity(max_tokens);
        for s in 0..max_tokens {
            let t = argmax(&last);
            generated.push(t);
            last = p.decode_step_logits(t, ids.len() + s);
        }
        let after = moe_stats(&p);
        let mut layers = BTreeMap::new();
        let mut by_record: BTreeMap<String, u64> = BTreeMap::new();
        for (li, (e0, grown)) in &layout {
            let (b, a) = (&before[li], &after[li]);
            let wins: Vec<u64> = a.iter().zip(b).map(|(x, y)| x.saturating_sub(*y)).collect();
            let routed: u64 = wins.iter().sum();
            let grown_wins: u64 = wins[*e0..].iter().sum();
            for (g, w) in grown.iter().zip(&wins[*e0..]) {
                *by_record.entry(g.record.clone()).or_default() += w;
            }
            layers.insert(*li, (grown_wins, routed));
        }
        let answer: String = p
            .tokenizer
            .decode(&generated)
            .split("<|im_end|>")
            .next()
            .unwrap_or("")
            .chars()
            .take(160)
            .collect();
        let r = PromptResult {
            index: i,
            prompt_tokens: ids.len(),
            layers,
            by_record,
            answer,
        };
        if !quiet {
            let per_layer: Vec<String> = r
                .layers
                .iter()
                .map(|(l, (g, t))| format!("L{l}:{g}/{t}"))
                .collect();
            println!(
                "{i:4} {} shell={} {}:{} | grown wins {} | {:?}",
                if r.hit() { "HIT " } else { "none" },
                if shell_on { "on" } else { "off" },
                row.lang,
                row.src,
                per_layer.join(" "),
                r.answer
            );
        }
        out.push(r);
    }
    Ok((layout, out))
}

fn mode_summary(
    shell_on: bool,
    rows: &[EvalRow],
    layout: &BTreeMap<usize, (usize, Vec<GrownExpert>)>,
    results: &[PromptResult],
) -> serde_json::Value {
    let mut all = Tally::default();
    let mut per_src: BTreeMap<String, Tally> = BTreeMap::new();
    let mut per_lang: BTreeMap<String, Tally> = BTreeMap::new();
    let mut per_layer: BTreeMap<usize, (usize, u64, u64)> = layout
        .keys()
        .map(|l| (*l, (0usize, 0u64, 0u64)))
        .collect();
    let mut per_record: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut no_hit = Vec::new();
    for (row, r) in rows.iter().zip(results) {
        let hit = r.hit();
        all.add(hit);
        per_src.entry(row.src.clone()).or_default().add(hit);
        per_lang.entry(row.lang.clone()).or_default().add(hit);
        for (l, (g, t)) in &r.layers {
            let e = per_layer.entry(*l).or_default();
            e.0 += usize::from(*g > 0);
            e.1 += g;
            e.2 += t;
        }
        for (id, w) in &r.by_record {
            let e = per_record.entry(id.clone()).or_default();
            e.0 += usize::from(*w > 0);
            e.1 += w;
        }
        if !hit {
            no_hit.push(r.index);
        }
    }
    let n = all.n;
    let layers: serde_json::Map<String, serde_json::Value> = per_layer
        .iter()
        .map(|(l, (ph, gw, rt))| {
            (
                l.to_string(),
                serde_json::json!({
                    "trunk_experts": layout[l].0,
                    "grown_experts": layout[l].1.len(),
                    "prompts_hit": ph,
                    "prompt_rate": if n > 0 { *ph as f64 / n as f64 } else { f64::NAN },
                    "grown_wins": gw,
                    "routed_tokens": rt,
                    "token_rate": if *rt > 0 { *gw as f64 / *rt as f64 } else { f64::NAN },
                }),
            )
        })
        .collect();
    let records: serde_json::Map<String, serde_json::Value> = per_record
        .iter()
        .map(|(id, (ph, w))| {
            (
                id.clone(),
                serde_json::json!({"prompts_hit": ph, "prompt_rate": if n > 0 { *ph as f64 / n as f64 } else { f64::NAN }, "wins": w}),
            )
        })
        .collect();
    let per = |m: BTreeMap<String, Tally>| -> serde_json::Value {
        m.into_iter().map(|(k, t)| (k, t.json())).collect()
    };
    serde_json::json!({
        "shell": if shell_on { "on" } else { "off" },
        "n": n,
        "hits": all.hits,
        "no_hit": n - all.hits,
        "hit_rate": if n > 0 { all.hits as f64 / n as f64 } else { f64::NAN },
        "hit_upper95": clopper_pearson_upper(all.hits, n, 0.95),
        "per_layer": layers,
        "per_record": records,
        "per_src": per(per_src),
        "per_lang": per(per_lang),
        "no_hit_indices": no_hit,
    })
}

pub fn cmd_growth_eval(a: GrowthEvalArgs<'_>) -> anyhow::Result<()> {
    let path = ForwardPath::parse(a.path)?;
    anyhow::ensure!(
        path == ForwardPath::PerOp,
        "growth-eval --path {}: the wins this tool counts are the host route's (`moe_route` \
         increments `MoeFfn.stats`); the resident graph selects its expert on the device and \
         exports no counter, so only per-op is measurable — dump the logits for G2 with \
         `dump-logits --path per-op` (the default) to stay on the measured path",
        path.label()
    );
    if let Some(g) = a.growth {
        anyhow::ensure!(
            matches!(g, "active" | "all" | "off"),
            "--growth {g}: expected active | all | off"
        );
        // SAFETY: before any pipeline exists; the loader reads it once per
        // layer on the main thread.
        unsafe { std::env::set_var("CMF_GROWTH", g) };
    }
    let modes: Vec<bool> = match a.shell {
        None => vec![growth_shell_enabled()],
        Some("on") => vec![true],
        Some("off") => vec![false],
        Some("both") => vec![true, false],
        Some(other) => anyhow::bail!("--shell {other}: expected on | off | both"),
    };
    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    let bytes = std::fs::read(a.prompts_jsonl).with_context(|| a.prompts_jsonl.to_string())?;
    let rows = parse_eval_jsonl(std::str::from_utf8(&bytes).context("prompts are not UTF-8")?)?;
    let mode = growth_mode();
    let mounted: Vec<serde_json::Value> = mounted_growth_records(&model.header)
        .iter()
        .map(|(at, s)| {
            serde_json::json!({
                "index": at,
                "id": s.id,
                "status": s.status,
                "layers": s.layers,
                "count": s.experts.as_ref().map(|e| e.count),
                "rank": s.experts.as_ref().map(|e| e.rank),
                "shell_quantile": s.experts.as_ref().map(|e| e.shell_quantile),
            })
        })
        .collect();
    let all_records: Vec<serde_json::Value> = model
        .header
        .skills
        .iter()
        .filter(|s| s.kind.as_deref() == Some(cortiq_core::knowledge::skill_kind::EXPERT_APPEND))
        .map(|s| serde_json::json!({"id": s.id, "status": s.status, "layers": s.layers}))
        .collect();
    let vacuous = if all_records.is_empty() {
        Some("the file carries no expert_append record".to_string())
    } else if mounted.is_empty() {
        Some(format!(
            "no expert_append record is mounted under CMF_GROWTH={} (records: {}); a \
             quarantined record needs --growth all",
            mode.label(),
            all_records
                .iter()
                .map(|r| format!("{}:{}", r["id"], r["status"]))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    } else {
        None
    };
    if let Some(why) = &vacuous {
        eprintln!("growth-eval: VACUOUS measurement — {why}");
    }
    if !a.json {
        println!(
            "growth-eval: {} | {} prompts, max_tokens={} (positions per record {}), growth={}, path={}, shell modes {:?}, mounted {}",
            a.model,
            rows.len(),
            a.max_tokens,
            a.max_tokens + 1,
            mode.label(),
            path.label(),
            modes
                .iter()
                .map(|m| if *m { "on" } else { "off" })
                .collect::<Vec<_>>(),
            mounted
                .iter()
                .map(|m| format!("{}({})", m["id"], m["status"]))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let mut summaries = Vec::new();
    let mut row_json: Vec<serde_json::Value> = Vec::new();
    let mut indices_written: Option<(bool, Vec<usize>)> = None;
    for &shell_on in &modes {
        let (layout, results) = run_mode(&model, &rows, a.max_tokens, shell_on, a.json)?;
        let summary = mode_summary(shell_on, &rows, &layout, &results);
        if !a.json {
            println!(
                "== shell {}: {}/{} prompts hit a grown expert (rate {:.4}, CP95 upper {:.4}) | per_layer {} | per_record {}",
                summary["shell"],
                summary["hits"],
                summary["n"],
                summary["hit_rate"].as_f64().unwrap_or(f64::NAN),
                summary["hit_upper95"].as_f64().unwrap_or(f64::NAN),
                summary["per_layer"],
                summary["per_record"]
            );
            println!("   per_src {}", summary["per_src"]);
            println!("   per_lang {}", summary["per_lang"]);
        }
        // `--indices-out`: the no-hit set of the shell-ON run when both
        // modes run (the production forward), else the only run's.
        let take = match &indices_written {
            None => true,
            Some((was_on, _)) => shell_on && !was_on,
        };
        if take {
            let idx: Vec<usize> = results
                .iter()
                .filter(|r| !r.hit())
                .map(|r| r.index)
                .collect();
            indices_written = Some((shell_on, idx));
        }
        if a.json {
            row_json.extend(rows.iter().zip(&results).map(|(row, r)| {
                serde_json::json!({
                    "shell": if shell_on { "on" } else { "off" },
                    "index": r.index,
                    "lang": row.lang,
                    "src": row.src,
                    "prompt_tokens": r.prompt_tokens,
                    "hit": r.hit(),
                    "layers": r.layers.iter().map(|(l, (g, t))| (l.to_string(), serde_json::json!({"grown_wins": g, "routed": t}))).collect::<serde_json::Map<_, _>>(),
                    "by_record": r.by_record,
                    "answer": r.answer,
                })
            }));
        }
        summaries.push(summary);
    }
    let prompts_sha = sha256_hex(&bytes);
    if let (Some(out), Some((shell_on, idx))) = (a.indices_out, &indices_written) {
        let j = serde_json::json!({
            "model": a.model,
            "prompts": a.prompts_jsonl,
            "prompts_sha256": prompts_sha,
            "shell": if *shell_on { "on" } else { "off" },
            "growth": mode.label(),
            "path": path.label(),
            "max_tokens": a.max_tokens,
            "n": rows.len(),
            "indices": idx,
        });
        std::fs::write(out, serde_json::to_string_pretty(&j)?).with_context(|| out.to_string())?;
        if !a.json {
            println!("no-hit indices ({}) → {out}", idx.len());
        }
    }
    let mut out = serde_json::json!({
        "model": a.model,
        "prompts": a.prompts_jsonl,
        "prompts_sha256": prompts_sha,
        "n": rows.len(),
        "max_tokens": a.max_tokens,
        "positions_per_record": a.max_tokens + 1,
        "growth_mode": mode.label(),
        "path": path.label(),
        "expert_append_records": all_records,
        "mounted": mounted,
        "vacuous": vacuous.is_some(),
        "reason": vacuous,
        "modes": summaries,
    });
    if a.json {
        out["rows"] = serde_json::Value::Array(row_json);
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    if let Some(why) = vacuous {
        anyhow::bail!("growth-eval: vacuous measurement — {why}");
    }
    Ok(())
}
