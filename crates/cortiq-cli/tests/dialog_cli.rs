//! `cortiq probe-dialog` on the synthetic GDN + bounded genome F0 and its
//! successor F1 = F0 + one `lookup` record (`herbs`) + a calibrated router,
//! with a 3-dialog fixture: the first turn names the plant, the follow-ups
//! do not; the third dialog switches plant mid-way.
//!
//! * `--route herbs` (pinned): the memory mechanics exactly — every
//!   follow-up answered from the entry named earlier (`lookup_turn` = turns
//!   back), the switch moves the entry, the field comes from the turn
//!   itself; the window is capped at `lookup::MEMORY_TURNS`; a wrong
//!   `expect_src` and a keyless first turn are listed as misses;
//! * `--route auto`: every turn's φ decision equals probe-utility's on
//!   the same text; a turn the router sends to the table hits from memory,
//!   a backbone turn never does under `router_and_key`; under `key_first`
//!   a strong key of the turn itself takes a backbone decision, an earlier
//!   turn's key never;
//! * `--lookup-mode context` generates (≤ 32 tokens), `off` only decides;
//!   F0 (no router, no table) runs the backbone throughout.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use knowledge_synth::{SKILL_ID, SKILL_TEXTS, lookup_entries};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_LOOKUP_MODE")
        .output()
        .expect("spawn cortiq")
}

fn ok(args: &[&str]) -> String {
    let out = cortiq(args);
    let (so, se) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(out.status.success(), "cortiq {args:?} failed\nstdout:\n{so}\nstderr:\n{se}");
    so
}

fn json_of(args: &[&str]) -> Value {
    let so = ok(args);
    serde_json::from_str(&so).unwrap_or_else(|e| panic!("cortiq {args:?}: not JSON ({e}):\n{so}"))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn write_lines(path: &Path, rows: &[Value]) {
    let text: String = rows.iter().map(|r| r.to_string() + "\n").collect();
    std::fs::write(path, text).unwrap();
}

/// The fixture: (dialog, the entry each turn names or remembers, the turn
/// the key comes from).
fn dialogs() -> Vec<Value> {
    vec![
        json!({
            "src": "Matricaria chamomilla",
            "turns": [SKILL_TEXTS[0], "А к какому семейству она относится?", "Есть ли у неё противопоказания?"],
            "expect_src": "Matricaria chamomilla",
        }),
        json!({
            "src": "Calendula officinalis",
            "turns": ["Tell me about pot marigold.", "Which family does it belong to?", "What are its side effects?"],
            "expect_src": "Calendula officinalis",
        }),
        json!({
            "id": "switch",
            "src": "Salvia officinalis",
            "turns": [SKILL_TEXTS[2], "Какое у него семейство?", SKILL_TEXTS[5], "К какому семейству он относится?"],
            "expect_src": ["Salvia officinalis", "Salvia officinalis", "Conium maculatum", "Conium maculatum"],
        }),
    ]
}

/// Per dialog, per turn: (entry, lookup_turn) when every turn is sent to
/// the table (pinned route).
const PINNED: &[&[(u64, u64)]] = &[
    &[(0, 0), (0, 1), (0, 2)],
    &[(4, 0), (4, 1), (4, 2)],
    &[(2, 0), (2, 1), (5, 0), (5, 1)],
];

struct Fx {
    dir: PathBuf,
    f0: PathBuf,
    f1: PathBuf,
    set: PathBuf,
}

fn fixture(tag: &str) -> Fx {
    // SAFETY: set before any pipeline of this process exists (the writer
    // computes φ); every test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-dialog-cli-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_lookup_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let set = dir.join("dialogs.jsonl");
    write_lines(&set, &dialogs());
    Fx {
        dir,
        f0: files.f0,
        f1: files.f1,
        set,
    }
}

fn turns_of(v: &Value) -> Vec<&Value> {
    v["per_dialog"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|d| d["turns"].as_array().unwrap())
        .collect()
}

#[test]
fn pinned_route_answers_follow_ups_from_the_entry_named_earlier() {
    let fx = fixture("pinned");
    let entries = lookup_entries();
    let v = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--route", SKILL_ID, "--json"]);
    assert_eq!(v["route_mode"], SKILL_ID, "{v}");
    assert_eq!(v["lookup_mode"], "answer");
    assert_eq!(v["memory_turns"], 6);
    assert_eq!(v["lookup_records"], json!({SKILL_ID: "router_and_key"}));
    assert!(v["max_tokens"].is_null());
    assert_eq!((v["dialogs"].as_u64(), v["turns"].as_u64()), (Some(3), Some(10)));
    assert_eq!((v["first_turns"].as_u64(), v["first_turn_hits"].as_u64()), (Some(3), Some(3)));
    assert_eq!(v["first_turn_hit_rate"], 1.0);
    assert_eq!(v["first_turn_correct"], 3);
    assert_eq!((v["followup_turns"].as_u64(), v["followup_hits"].as_u64()), (Some(7), Some(7)));
    assert_eq!(v["followup_hit_rate"], 1.0);
    // The switch dialog's third turn names its plant itself.
    assert_eq!(v["followup_memory_hits"], 6);
    assert_eq!((v["followup_checked"].as_u64(), v["followup_correct"].as_u64()), (Some(7), Some(7)));
    assert_eq!(v["followup_correct_rate"], 1.0);
    assert_eq!(v["followup_correct_of_expected"], 1.0);
    assert!((v["mean_lookup_turn"].as_f64().unwrap() - 0.8).abs() < 1e-12, "{}", v["mean_lookup_turn"]);
    assert!((v["mean_lookup_turn_followup"].as_f64().unwrap() - 8.0 / 7.0).abs() < 1e-12);
    assert_eq!(v["lookup_turn_hist"], json!({"0": 4, "1": 4, "2": 2}));
    assert_eq!(v["route_counts"], json!({SKILL_ID: 10}));
    assert_eq!(v["decided_by_counts"], json!({"router": 10}));
    assert_eq!(v["field_counts"], json!({"family": 4, "safety": 2}));
    assert_eq!((v["misses_total"].as_u64(), v["expect_unresolved"].as_u64()), (Some(0), Some(0)));
    assert_eq!(v["misses"], json!([]));
    assert_eq!(v["per_lang"]["ru"]["dialogs"], 2, "{}", v["per_lang"]);
    assert_eq!(v["per_lang"]["ru"]["turns"], 7);
    assert_eq!(v["per_lang"]["ru"]["followup_hits"], 5);
    assert_eq!(v["per_lang"]["en"]["dialogs"], 1);
    assert_eq!(v["per_lang"]["en"]["followup_correct"], 2);
    let dl = v["per_dialog"].as_array().unwrap();
    assert_eq!(dl[2]["id"], "switch");
    for (d, want) in PINNED.iter().enumerate() {
        let turns = dl[d]["turns"].as_array().unwrap();
        assert_eq!(turns.len(), want.len());
        for (t, (row, &(entry, back))) in turns.iter().zip(want.iter()).enumerate() {
            assert_eq!(row["t"], t);
            assert_eq!(row["lookup_hit"], true, "{row}");
            assert_eq!((row["entry"].as_u64(), row["lookup_turn"].as_u64()), (Some(entry), Some(back)), "{row}");
            assert_eq!(row["route"]["lookup_turn"], back);
            assert_eq!(row["correct"], true, "{row}");
            assert_eq!(row["expect_entry"], entry);
            assert_eq!(row["expect_via"], "exact");
            assert_eq!(row["window"], t + 1);
            assert_eq!(row["target"], SKILL_ID);
            assert_eq!(row["decided_by"], "router");
            assert_eq!(row["finish_reason"], "lookup");
            assert_eq!(row["generated"], 0);
        }
    }
    let binomial = |d: usize, t: usize| dl[d]["turns"][t]["entry_binomial"].as_str().unwrap().to_string();
    assert_eq!(binomial(0, 2), "Matricaria chamomilla");
    assert_eq!(binomial(1, 1), "Calendula officinalis");
    assert_eq!(binomial(2, 1), "Salvia officinalis");
    assert_eq!(binomial(2, 3), "Conium maculatum");
    // The field and the language come from the turn itself.
    let row = |d: usize, t: usize| &dl[d]["turns"][t];
    assert_eq!((row(0, 1)["field"].as_str(), row(0, 1)["lang"].as_str()), (Some("family"), Some("ru")));
    assert_eq!(row(0, 1)["answer"], entries[0].1["fields"]["family"]);
    assert_eq!(row(0, 2)["field"], "safety");
    assert_eq!(row(0, 2)["answer"], entries[0].1["fields"]["safety"]);
    assert!(row(0, 0)["field"].is_null());
    assert_eq!(row(0, 0)["answer"], entries[0].1["card"]);
    assert_eq!((row(1, 1)["field"].as_str(), row(1, 1)["lang"].as_str()), (Some("family"), Some("en")));
    assert_eq!(row(1, 1)["answer"], entries[4].2["fields"]["family"]);
    assert_eq!(row(1, 2)["answer"], entries[4].2["fields"]["safety"]);
    assert_eq!(row(2, 1)["answer"], entries[2].1["fields"]["family"]);
    assert_eq!(row(2, 3)["answer"], entries[5].1["fields"]["family"]);

    // Misses: a wrong expectation, a keyless first turn, the memory cap.
    let far = std::iter::once(SKILL_TEXTS[4])
        .chain(std::iter::repeat_n("Расскажи подробнее.", 6))
        .collect::<Vec<_>>();
    let set2 = fx.dir.join("misses.jsonl");
    write_lines(
        &set2,
        &[
            json!({"src": "Salvia officinalis", "turns": [SKILL_TEXTS[1]]}),
            json!({"src": "Calendula officinalis", "turns": ["Какая сегодня погода?", "А семейство?"]}),
            json!({"src": "Calendula officinalis", "turns": far}),
            json!({"src": "Nonexistia plantae", "turns": [SKILL_TEXTS[3]]}),
        ],
    );
    let v = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&set2), "--route", SKILL_ID, "--json"]);
    let m = v["misses"].as_array().unwrap();
    assert_eq!(v["misses_total"], 4, "{v}");
    assert_eq!((m[0]["kind"].as_str(), m[0]["t"].as_u64()), (Some("wrong_entry"), Some(0)), "{}", m[0]);
    assert_eq!((m[0]["entry"].as_u64(), m[0]["expect_entry"].as_u64()), (Some(1), Some(2)));
    assert_eq!((m[1]["kind"].as_str(), m[1]["t"].as_u64()), (Some("no_hit"), Some(0)));
    assert_eq!((m[2]["kind"].as_str(), m[2]["t"].as_u64()), (Some("no_hit"), Some(1)));
    // The seventh turn: the naming turn is 6 back — outside the window.
    assert_eq!((m[3]["kind"].as_str(), m[3]["id"].as_str(), m[3]["t"].as_u64()), (Some("no_hit"), Some("Calendula officinalis"), Some(6)));
    let far_turns = v["per_dialog"][2]["turns"].as_array().unwrap();
    for (t, row) in far_turns.iter().enumerate().take(6) {
        assert_eq!((row["lookup_hit"].as_bool(), row["lookup_turn"].as_u64()), (Some(true), Some(t as u64)), "{row}");
    }
    assert_eq!(far_turns[6]["window"], 6);
    assert_eq!(v["expect_unresolved"], 1, "the unknown binomial");
    assert!(v["per_dialog"][3]["turns"][0]["correct"].is_null());
    assert_eq!(v["first_turn_correct_rate"], 0.5, "{v}");

    // Per-turn languages (review KF-7): a Russian dialog with an English
    // follow-up is one `ru` dialog, but its turns fall in `ru` and `en`
    // (the answer language is chosen per turn).
    let set3 = fx.dir.join("mixed.jsonl");
    write_lines(
        &set3,
        &[json!({"src": "Matricaria chamomilla", "turns": [SKILL_TEXTS[0], "Which family does it belong to?", "А противопоказания?"]})],
    );
    let v = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&set3), "--route", SKILL_ID, "--json"]);
    assert_eq!(v["per_dialog_lang"].as_object().unwrap().len(), 1, "{}", v["per_dialog_lang"]);
    assert_eq!(v["per_dialog_lang"]["ru"]["turns"], 3);
    assert_eq!(v["per_lang"], v["per_turn_lang"]);
    assert_eq!((v["per_lang"]["ru"]["turns"].as_u64(), v["per_lang"]["en"]["turns"].as_u64()), (Some(2), Some(1)), "{}", v["per_lang"]);
    assert_eq!((v["per_lang"]["en"]["dialogs"].as_u64(), v["per_lang"]["en"]["followup_hits"].as_u64()), (Some(1), Some(1)));
    assert_eq!(v["per_lang"]["ru"]["dialogs"], 1);
    let t1 = &v["per_dialog"][0]["turns"][1];
    assert_eq!((t1["turn_lang"].as_str(), t1["lang"].as_str(), t1["field"].as_str()), (Some("en"), Some("en"), Some("family")), "{t1}");
    assert_eq!(t1["entry"], 0);

    // The human report.
    let so = ok(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--route", SKILL_ID]);
    assert!(so.contains("follow-up hits 7/7"), "{so}");
    assert!(so.contains("== turns in ru:") && so.contains("== dialogs in en:"), "{so}");
    assert!(so.contains("switch t3: route herbs"), "{so}");
    assert!(so.contains("\"Conium maculatum\""), "{so}");
    // Refusals: a bad mode, a malformed file.
    assert!(!cortiq(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--lookup-mode", "maybe"]).status.success());
    let bad = fx.dir.join("bad.jsonl");
    std::fs::write(&bad, "{\"turns\": []}\n").unwrap();
    assert!(!cortiq(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&bad)]).status.success());
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn auto_route_decides_each_turn_as_serve_does() {
    let fx = fixture("auto");
    // ── router_and_key (F1) ──
    let v = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--route", "auto", "--json"]);
    assert_eq!(v["route_mode"], "auto", "{v}");
    let turns = turns_of(&v);
    assert_eq!(turns.len(), 10);
    // The φ decision of every turn is probe-utility's on the same text.
    let texts: Vec<&str> = turns.iter().map(|r| r["text"].as_str().unwrap()).collect();
    let pu_set = fx.dir.join("turns.jsonl");
    write_lines(
        &pu_set,
        &texts.iter().map(|t| json!({"prompt": t, "lang": "x", "expect": []})).collect::<Vec<_>>(),
    );
    let pu = json_of(&["probe-utility", s(&fx.f1), "--prompts-jsonl", s(&pu_set), "--max-tokens", "1", "--route", "auto", "--json"]);
    for (row, pr) in turns.iter().zip(pu[0]["per_prompt"].as_array().unwrap()) {
        let router_pick = pr["route"]
            .get("decided_target")
            .and_then(Value::as_str)
            .unwrap_or_else(|| pr["route"]["target"].as_str().unwrap());
        assert_eq!(row["decided_target"].as_str(), Some(router_pick), "{row}\n{pr}");
    }
    let mut to_table = 0;
    for row in &turns {
        if row["decided_target"] == SKILL_ID {
            // Every window of the fixture holds the plant: a turn the
            // router sends to the table is answered, from the right entry.
            to_table += 1;
            assert_eq!(row["lookup_hit"], true, "{row}");
            assert_eq!(row["target"], SKILL_ID);
            assert_eq!(row["correct"], true, "{row}");
            assert_eq!(row["decided_by"], "router");
        } else {
            assert_eq!(row["lookup_hit"], false, "{row}");
            assert_eq!(row["target"], "backbone");
            assert!(row["answer"].is_null(), "answer mode generates nothing: {row}");
        }
    }
    // The in-scope first turns (the router's own skill class) are hits.
    let dl = v["per_dialog"].as_array().unwrap();
    for (d, t, entry) in [(0usize, 0usize, 0u64), (2, 0, 2), (2, 2, 5)] {
        let row = &dl[d]["turns"][t];
        assert_eq!((row["lookup_hit"].as_bool(), row["entry"].as_u64()), (Some(true), Some(entry)), "{row}");
        assert_eq!(row["lookup_turn"], 0);
    }
    assert_eq!(v["route_counts"][SKILL_ID].as_u64().unwrap_or(0), to_table);
    assert_eq!(
        v["first_turn_hits"].as_u64().unwrap() + v["followup_hits"].as_u64().unwrap(),
        to_table
    );
    assert_eq!(v["misses_total"].as_u64().unwrap(), 10 - to_table);
    eprintln!(
        "router_and_key: first-turn hits {}/3, follow-up hits {}/7 (from memory {}), routes {}",
        v["first_turn_hits"], v["followup_hits"], v["followup_memory_hits"], v["route_counts"]
    );

    // ── key_first (a copy switched by lookup-policy) ──
    let fk = fx.dir.join("f1-key-first.cmf");
    std::fs::copy(&fx.f1, &fk).unwrap();
    // --keep-gate: the active record stays active (the fixture's gate is
    // synthetic); without it the switch would demand a re-gate.
    json_of(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "key_first", "--keep-gate"]);
    let k = json_of(&["probe-dialog", s(&fk), "--dialogs-jsonl", s(&fx.set), "--route", "auto", "--json"]);
    assert_eq!(k["lookup_records"], json!({SKILL_ID: "key_first"}), "{k}");
    let kturns = turns_of(&k);
    for (row, base) in kturns.iter().zip(&turns) {
        // The router's decision is the same file's; key_first acts only on
        // a backbone decision and a strong key of the turn itself.
        if row["decided_by"] == "key_first" {
            assert_eq!(base["decided_target"], "backbone", "{row}");
            assert_eq!(row["lookup_turn"], 0, "{row}");
            assert!(row["lookup_key_words"].as_u64().unwrap() >= 2, "{row}");
            assert_eq!(row["correct"], true, "{row}");
        } else {
            assert_eq!(row["lookup_hit"], base["lookup_hit"], "{row}\n{base}");
            assert_eq!(row["decided_target"], base["decided_target"]);
        }
    }
    // "Tell me about pot marigold." — a two-word key: the table answers
    // whoever decided; its keyless follow-ups hit only where the router
    // itself sends them to the table.
    let d2 = k["per_dialog"][1]["turns"].as_array().unwrap();
    assert_eq!((d2[0]["lookup_hit"].as_bool(), d2[0]["entry"].as_u64()), (Some(true), Some(4)), "{}", d2[0]);
    for row in &d2[1..] {
        assert_eq!(row["lookup_hit"].as_bool(), Some(row["decided_target"] == SKILL_ID && row["decided_by"] == "router"), "{row}");
    }
    eprintln!(
        "key_first: first-turn hits {}/3, follow-up hits {}/7, by {}",
        k["first_turn_hits"], k["followup_hits"], k["decided_by_counts"]
    );
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn context_generates_off_only_decides_and_f0_runs_the_backbone() {
    let fx = fixture("modes");
    let v = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--route", SKILL_ID, "--lookup-mode", "context", "--max-tokens", "2", "--json"]);
    assert_eq!((v["lookup_mode"].as_str(), v["max_tokens"].as_u64()), (Some("context"), Some(2)), "{v}");
    assert_eq!(v["followup_hits"], 7);
    for row in turns_of(&v) {
        assert_eq!(row["route"]["lookup_mode"], "context");
        assert_eq!(row["lookup_hit"], true, "{row}");
        assert_ne!(row["finish_reason"], "lookup", "{row}");
        assert!(row["generated"].as_u64().unwrap() <= 2, "{row}");
        assert!(row["answer"].is_string(), "{row}");
    }
    // The budget is capped at 32 tokens.
    let one = fx.dir.join("one.jsonl");
    write_lines(&one, &[json!({"src": "Matricaria chamomilla", "turns": [SKILL_TEXTS[0]]})]);
    let c = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&one), "--route", SKILL_ID, "--lookup-mode", "context", "--max-tokens", "100", "--json"]);
    assert_eq!(c["max_tokens"], 32, "{c}");
    assert!(c["per_dialog"][0]["turns"][0]["generated"].as_u64().unwrap() <= 32);
    // off: decided, nothing looked up, nothing generated.
    let o = json_of(&["probe-dialog", s(&fx.f1), "--dialogs-jsonl", s(&fx.set), "--route", SKILL_ID, "--lookup-mode", "off", "--json"]);
    assert_eq!((o["first_turn_hits"].as_u64(), o["followup_hits"].as_u64()), (Some(0), Some(0)), "{o}");
    assert_eq!(o["route_counts"], json!({"backbone": 10}));
    assert_eq!(o["decided_counts"], json!({SKILL_ID: 10}));
    assert_eq!(o["misses_total"], 10);
    for row in turns_of(&o) {
        assert!(row["answer"].is_null(), "{row}");
        assert_eq!(row["finish_reason"], "none");
    }
    // F0: no router, no table.
    let z = json_of(&["probe-dialog", s(&fx.f0), "--dialogs-jsonl", s(&fx.set), "--json"]);
    assert_eq!(z["route_mode"], "none", "{z}");
    assert_eq!(z["lookup_records"], json!({}));
    assert_eq!(z["route_counts"], json!({"backbone": 10}));
    assert_eq!((z["first_turn_hits"].as_u64(), z["followup_hits"].as_u64()), (Some(0), Some(0)));
    assert_eq!(z["expect_unresolved"], 0, "no table to resolve against: nothing counted");
    assert_eq!((z["router_v2"].as_bool(), z["warnings"].as_array().map(Vec::len)), (Some(false), Some(0)));
    // `--route auto` on a file without a router policy: serve never routes
    // it — said, not silently measured (review KF-7).
    let w = cortiq(&["probe-dialog", s(&fx.f0), "--dialogs-jsonl", s(&one), "--route", "auto", "--json"]);
    assert!(w.status.success());
    let wj: Value = serde_json::from_slice(&w.stdout).unwrap();
    assert!(wj["warnings"][0].as_str().unwrap().contains("ROUTER_V2"), "{wj}");
    assert!(String::from_utf8_lossy(&w.stderr).contains("serve does not route"), "stderr");
    let _ = std::fs::remove_dir_all(&fx.dir);
}
