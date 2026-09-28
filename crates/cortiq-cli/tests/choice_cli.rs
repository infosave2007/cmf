//! `cortiq probe-choice` end to end through the binary on the synthetic
//! GDN + bounded genome (F0 of the knowledge fixture; byte-level vocab,
//! so a family name is one token per byte and the candidate trie
//! branches deep): the default candidate set (distinct `expect[0]`) vs
//! `--candidates-jsonl`, the ranking and JSON summary (baselines,
//! per-lang, per-row), `--norm mean`, and the shared prefix walk (host
//! KV snapshot + rollback between trie siblings) against the full
//! re-forward scorer (`--scorer full`, the engine's `nll_ids_from`).
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_GROWTH")
        .env_remove("CMF_GROWTH_SHELL")
        .output()
        .expect("spawn cortiq")
}

fn json(args: &[&str]) -> serde_json::Value {
    let out = cortiq(args);
    let (so, se) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(
        out.status.success(),
        "cortiq {args:?} failed\nstdout:\n{so}\nstderr:\n{se}"
    );
    serde_json::from_str(&so).unwrap_or_else(|e| panic!("cortiq {args:?}: not JSON ({e}):\n{so}"))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

fn strings(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// (lang, src, prompt, expect) — four distinct answers over six rows,
/// `Asteraceae` the majority (3/6, one spelled in lower case); the Latin
/// names share leading bytes (`A…`, `R…`, `…aceae`) so the trie has real
/// branches at several depths.
const ROWS: &[(&str, &str, &str, &[&str])] = &[
    ("ru", "Achillea", "К какому семейству относится тысячелистник?", &["Asteraceae", "Астровые"]),
    ("en", "Salvia", "Which plant family does sage belong to?", &["Lamiaceae", "Яснотковые"]),
    ("ru", "Rosa", "К какому семейству относится шиповник?", &["Rosaceae", "Розовые"]),
    ("en", "Arnica", "Which plant family does arnica belong to?", &["asteraceae"]),
    ("ru", "Aconitum", "К какому семейству относится борец?", &["Ranunculaceae", "Лютиковые"]),
    ("en", "Tanacetum", "Which plant family does tansy belong to?", &["Asteraceae", "Астровые"]),
];

const DEFAULT_CANDIDATES: [&str; 4] = ["Asteraceae", "Lamiaceae", "Rosaceae", "Ranunculaceae"];

struct Fixture {
    dir: PathBuf,
    f0: PathBuf,
    prompts: PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    // SAFETY: set before any pipeline of this process exists (the writer
    // computes φ); every test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-choice-cli-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_knowledge_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let prompts = dir.join("prompts.jsonl");
    let text: String = ROWS
        .iter()
        .map(|(lang, src, prompt, expect)| {
            serde_json::json!({"lang": lang, "src": src, "prompt": prompt, "expect": expect})
                .to_string()
                + "\n"
        })
        .collect();
    std::fs::write(&prompts, text).unwrap();
    Fixture {
        dir,
        f0: files.f0,
        prompts,
    }
}

/// The per-row invariants of a report against its candidate list.
fn check_rows(v: &serde_json::Value, candidates: &[String]) {
    let rows = v["per_row"].as_array().unwrap();
    assert_eq!(rows.len(), ROWS.len(), "{v}");
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r["index"], i);
        assert_eq!(r["lang"], ROWS[i].0);
        assert_eq!(r["src"], ROWS[i].1);
        assert_eq!(strings(&r["expect"]), ROWS[i].3);
        let predicted = r["predicted"].as_str().unwrap();
        assert!(candidates.iter().any(|c| c == predicted), "{r}");
        let lp_best = f(&r["lp_best"]);
        assert!(lp_best.is_finite() && lp_best < 0.0, "{r}");
        let in_set = candidates
            .iter()
            .any(|c| c.to_lowercase() == ROWS[i].3[0].to_lowercase());
        if in_set {
            let rank = r["rank_of_correct"].as_u64().unwrap() as usize;
            assert!((1..=candidates.len()).contains(&rank), "{r}");
            let lp_correct = f(&r["lp_correct"]);
            assert!(lp_correct <= lp_best + 1e-12, "{r}");
            assert_eq!(r["correct"], rank == 1, "{r}");
            if rank == 1 {
                assert_eq!(predicted.to_lowercase(), ROWS[i].3[0].to_lowercase());
                assert_eq!(lp_correct, lp_best);
                assert!(f(&r["lp_best_wrong"]) <= lp_best, "{r}");
            } else {
                assert_eq!(f(&r["lp_best_wrong"]), lp_best, "{r}");
            }
            assert_eq!(r["tokens_correct"], ROWS[i].3[0].len() as u64, "{r}");
            top1 += (rank == 1) as usize;
            top5 += (rank <= 5) as usize;
        } else {
            assert_eq!(r["rank_of_correct"], serde_json::Value::Null, "{r}");
            assert_eq!(r["lp_correct"], serde_json::Value::Null);
            assert_eq!(r["tokens_correct"], serde_json::Value::Null);
            assert_eq!(r["correct"], false);
            assert_eq!(f(&r["lp_best_wrong"]), lp_best);
        }
        assert_eq!(r["tokens_predicted"], predicted.len() as u64, "{r}");
        assert_eq!(r["route"]["target"], "backbone", "{r}");
    }
    assert_eq!(v["top1"], top1 as u64, "{v}");
    assert_eq!(v["top5"], top5 as u64, "{v}");
    assert_eq!(f(&v["top1_acc"]), top1 as f64 / ROWS.len() as f64);
    assert_eq!(f(&v["top5_acc"]), top5 as f64 / ROWS.len() as f64);
    // Per-lang blocks partition the rows.
    let per_lang = v["per_lang"].as_object().unwrap();
    assert_eq!(per_lang.len(), 2);
    assert_eq!(per_lang["ru"]["n"], 3);
    assert_eq!(per_lang["en"]["n"], 3);
    let lang_top1: u64 = per_lang.values().map(|b| b["top1"].as_u64().unwrap()).sum();
    assert_eq!(lang_top1, top1 as u64);
}

#[test]
fn default_candidates_ranking_and_summary() {
    let fx = fixture("default");
    let v = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--json",
    ]);
    assert_eq!(v["n"], 6, "{v}");
    // The distinct expect[0] in first-seen order; `asteraceae` (row 4)
    // folds into the first spelling.
    let list = strings(&v["candidate_list"]);
    assert_eq!(list, DEFAULT_CANDIDATES);
    assert_eq!(v["candidates"], 4);
    assert!((f(&v["chance"]) - 0.25).abs() < 1e-12);
    assert_eq!(v["majority"]["label"], "Asteraceae");
    assert_eq!(v["majority"]["n"], 3);
    assert_eq!(f(&v["majority"]["acc"]), 0.5);
    assert_eq!(v["expect_outside_candidates"], 0);
    // Four candidates: every scorable row is within the top 5.
    assert_eq!(v["top5"], 6);
    assert_eq!(v["candidates_source"], "expect[0] of the prompt set");
    assert_eq!(v["norm"], "sum");
    assert_eq!(v["scorer"], "shared");
    assert_eq!(v["bos"], false);
    assert_eq!(v["route_mode"], "none");
    assert_eq!(v["route_counts"]["backbone"], 6);
    assert_eq!(v["growth"], "active");
    assert_eq!(v["shell"], "on");
    let mc = f(&v["mean_lp_correct"]);
    let mw = f(&v["mean_lp_best_wrong"]);
    assert!(mc.is_finite() && mw.is_finite());
    assert!((f(&v["mean_margin"]) - (mc - mw)).abs() < 1e-9, "{v}");
    check_rows(&v, &list);
    // Deterministic: a second run is identical row for row.
    let again = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--json",
    ]);
    assert_eq!(again["per_row"], v["per_row"]);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn explicit_candidates_and_mean_norm() {
    let fx = fixture("cands");
    // Three candidates, one duplicate spelling, one answer (Ranunculaceae)
    // and the lower-case Asteraceae row's answer both need case folding;
    // Lamiaceae and Ranunculaceae are absent from the set.
    let cands = fx.dir.join("cands.jsonl");
    std::fs::write(
        &cands,
        "{\"text\":\"Asteraceae\"}\n{\"text\":\"Rosaceae\"}\n{\"text\":\" asteraceae\"}\n{\"text\":\"Apiaceae\"}\n",
    )
    .unwrap();
    let sum = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--candidates-jsonl",
        s(&cands),
        "--json",
    ]);
    let list = strings(&sum["candidate_list"]);
    assert_eq!(list, ["Asteraceae", "Rosaceae", "Apiaceae"]);
    assert_eq!(sum["candidates"], 3);
    assert!((f(&sum["chance"]) - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(sum["candidates_source"], s(&cands));
    // Salvia (Lamiaceae) and Aconitum (Ranunculaceae) cannot be scored.
    assert_eq!(sum["expect_outside_candidates"], 2, "{sum}");
    let rows = sum["per_row"].as_array().unwrap();
    assert_eq!(rows[1]["rank_of_correct"], serde_json::Value::Null);
    assert_eq!(rows[4]["rank_of_correct"], serde_json::Value::Null);
    assert!(rows[0]["rank_of_correct"].is_u64());
    assert!(rows[3]["rank_of_correct"].is_u64(), "lower-case answer folds: {}", rows[3]);
    // The majority baseline is a property of the prompt set, not the candidates.
    assert_eq!(sum["majority"]["label"], "Asteraceae");
    assert_eq!(sum["majority"]["n"], 3);
    check_rows(&sum, &list);

    let mean = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--candidates-jsonl",
        s(&cands),
        "--norm",
        "mean",
        "--json",
    ]);
    assert_eq!(mean["norm"], "mean");
    check_rows(&mean, &list);
    // mean = sum / tokens for the correct candidate, row by row.
    for (a, b) in sum["per_row"]
        .as_array()
        .unwrap()
        .iter()
        .zip(mean["per_row"].as_array().unwrap())
    {
        if let Some(tok) = a["tokens_correct"].as_f64() {
            let want = f(&a["lp_correct"]) / tok;
            assert!((f(&b["lp_correct"]) - want).abs() < 1e-9, "{a}\n{b}");
        }
    }
    let bad = cortiq(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--norm",
        "median",
        "--json",
    ]);
    assert!(!bad.status.success());
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// The shared walk (one prefix forward, KV rollback between siblings —
/// attention KV, GDN recurrent state and the bounded anchor restored
/// from the snapshot) must reproduce the full re-forward of every
/// `prefix + candidate` (batched prefill through `nll_ids_from`).
#[test]
fn shared_walk_matches_full_reforward() {
    let fx = fixture("scorer");
    let shared = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--json",
    ]);
    let full = json(&[
        "probe-choice",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--scorer",
        "full",
        "--json",
    ]);
    assert_eq!(full["scorer"], "full");
    let (a, b) = (
        shared["per_row"].as_array().unwrap(),
        full["per_row"].as_array().unwrap(),
    );
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        for key in ["lp_correct", "lp_best", "lp_best_wrong"] {
            let (p, q) = (f(&x[key]), f(&y[key]));
            assert!(
                (p - q).abs() < 1e-3,
                "{key}: shared {p} vs full {q}\n{x}\n{y}"
            );
        }
        assert_eq!(x["predicted"], y["predicted"], "{x}\n{y}");
        assert_eq!(x["rank_of_correct"], y["rank_of_correct"]);
    }
    assert_eq!(shared["top1"], full["top1"]);
    let _ = std::fs::remove_dir_all(&fx.dir);
}
