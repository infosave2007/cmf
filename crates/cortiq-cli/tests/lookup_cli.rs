//! The `lookup` record through the `cortiq` binary (lookup spec §3), on a
//! synthetic GDN + bounded genome F0 and its successor F1 = F0 + one
//! lookup record + a calibrated router:
//!
//! * `run` — a key question is answered from the table (nothing
//!   generated), a general one runs the backbone, `--lookup-mode off` /
//!   `CMF_LOOKUP_MODE=context` switch the behaviour;
//! * `probe-utility --route auto` — per-row `route` with `lookup_hit`,
//!   `lookup_key`, `field`; the answer is the table's text;
//! * `dump-logits` F0 vs F1 `--route auto` + `logits-compare` — every
//!   backbone record (general prompts and an in-domain prompt without a
//!   key) bit-identical to F0; the hit records carry the backbone's own
//!   logits in `answer` mode and differ in `context` mode;
//! * `lookup-policy --policy key_first` (a header-only switch) — plant
//!   prompts the φ router sends to the backbone are answered from the
//!   table under `key_first` and NOT under `router_and_key`; one-word
//!   keys and general prompts keep the backbone, bit-identical to F0.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS, lookup_entries};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const NO_KEY_TEXT: &str = "Какие лечебные свойства у мяты перечной?";

fn cortiq_env(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cortiq"));
    c.args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_LOOKUP_MODE");
    for (k, v) in envs {
        c.env(k, v);
    }
    c.output().expect("spawn cortiq")
}

fn cortiq(args: &[&str]) -> Output {
    cortiq_env(args, &[])
}

fn ok_env(args: &[&str], envs: &[(&str, &str)]) -> (String, String) {
    let out = cortiq_env(args, envs);
    let (so, se) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(
        out.status.success(),
        "cortiq {args:?} failed\nstdout:\n{so}\nstderr:\n{se}"
    );
    (so, se)
}

fn ok(args: &[&str]) -> (String, String) {
    ok_env(args, &[])
}

fn json(args: &[&str]) -> serde_json::Value {
    let (so, _) = ok(args);
    serde_json::from_str(&so).unwrap_or_else(|e| panic!("cortiq {args:?}: not JSON ({e}):\n{so}"))
}

fn write_jsonl(path: &Path, rows: &[(&str, &str, &str)]) {
    let text: String = rows
        .iter()
        .map(|(p, lang, src)| {
            serde_json::json!({"prompt": p, "lang": lang, "src": src, "expect": []}).to_string()
                + "\n"
        })
        .collect();
    std::fs::write(path, text).unwrap();
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

struct Fixture {
    dir: PathBuf,
    f0: PathBuf,
    f1: PathBuf,
    /// general[i], skill[i] interleaved, then the in-domain no-key prompt.
    mixed: PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    // SAFETY: set before any pipeline of this process exists (the writer
    // computes φ); every test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-lookup-cli-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_lookup_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let mixed = dir.join("mixed.jsonl");
    let mut rows: Vec<(&str, &str, &str)> = GENERAL_TEXTS
        .iter()
        .zip(SKILL_TEXTS)
        .flat_map(|(g, k)| [(*g, "en", "general"), (*k, "ru", "herbs")])
        .collect();
    rows.push((NO_KEY_TEXT, "ru", "nokey"));
    write_jsonl(&mixed, &rows);
    Fixture {
        dir,
        f0: files.f0,
        f1: files.f1,
        mixed,
    }
}

fn rows_total(m: &serde_json::Value) -> u64 {
    m["per_prompt"].as_array().unwrap().len() as u64
}

fn family_ru() -> String {
    lookup_entries()[4].1["fields"]["family"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn run_answers_from_the_table_and_keeps_the_backbone_for_the_rest() {
    let fx = fixture("run");
    let family = family_ru();
    // A key question: the table answers, nothing is generated.
    let (so, se) = ok(&["run", s(&fx.f1), "-p", SKILL_TEXTS[4], "-n", "2", "--greedy"]);
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    assert!(se.contains(&format!("lookup: {SKILL_ID}")), "{se}");
    assert!(se.contains("field family") && se.contains("mode answer"), "{se}");
    assert!(so.contains(&family), "{so}");
    assert!(so.contains("answered from the table"), "{so}");
    assert!(!so.contains("tokens"), "something was generated:\n{so}");
    // A general question: the backbone, no lookup line.
    let (_, se) = ok(&["run", s(&fx.f1), "-p", GENERAL_TEXTS[0], "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone"), "{se}");
    assert!(!se.contains("lookup:"), "{se}");
    // An in-domain question without a key: the backbone, unchanged.
    let (so, se) = ok(&["run", s(&fx.f1), "-p", NO_KEY_TEXT, "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone"), "{se}");
    assert!(!so.contains(&family), "{so}");
    // --lookup-mode off: the backbone runs the key question.
    let (so, se) = ok(&[
        "run",
        s(&fx.f1),
        "-p",
        SKILL_TEXTS[4],
        "-n",
        "2",
        "--greedy",
        "--lookup-mode",
        "off",
    ]);
    assert!(se.contains("route: backbone") && se.contains("mode off"), "{se}");
    assert!(!so.contains(&family), "{so}");
    // CMF_LOOKUP_MODE=context: the backbone generates from the prompt
    // with the card in front of it.
    let (_, se) = ok_env(
        &["run", s(&fx.f1), "-p", SKILL_TEXTS[4], "-n", "1", "--greedy"],
        &[("CMF_LOOKUP_MODE", "context"), ("CMF_PROMPT_DUMP", "1")],
    );
    assert!(se.contains("mode context"), "{se}");
    // The rendered prompt is decoded from byte tokens: check its ASCII.
    assert!(se.contains("Reference card:"), "{se}");
    assert!(
        se.contains("(Calendula officinalis)"),
        "the card is not in the rendered prompt:\n{se}"
    );
    // The flag wins over the environment; a bad value is refused.
    let (so, _) = ok_env(
        &["run", s(&fx.f1), "-p", SKILL_TEXTS[4], "-n", "1", "--greedy", "--lookup-mode", "answer"],
        &[("CMF_LOOKUP_MODE", "off")],
    );
    assert!(so.contains(&family), "{so}");
    assert!(!cortiq(&["run", s(&fx.f1), "-p", "x", "--lookup-mode", "maybe"]).status.success());
    // A pinned lookup record with a prompt takes the same path.
    let (so, se) = ok(&["run", s(&fx.f1), "-p", SKILL_TEXTS[4], "--skill", SKILL_ID]);
    assert!(se.contains("pinned") && se.contains("lookup key"), "{se}");
    assert!(so.contains(&family), "{so}");
    // F0 knows no table: the plain pipeline.
    let (so, se) = ok(&["run", s(&fx.f0), "-p", SKILL_TEXTS[4], "-n", "2", "--greedy"]);
    assert!(!se.contains("lookup:"), "{se}");
    assert!(!so.contains(&family), "{so}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn probe_utility_reports_route_and_lookup_hit_per_row() {
    let fx = fixture("probe");
    let entries = lookup_entries();
    let v = json(&[
        "probe-utility",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--route",
        "auto",
        "--json",
    ]);
    let m = &v[0];
    assert_eq!(m["route_mode"], "auto", "{m}");
    assert_eq!(m["lookup_mode"], "answer");
    assert_eq!(m["lookup_hits"], SKILL_TEXTS.len());
    assert_eq!(m["route_counts"][SKILL_ID], SKILL_TEXTS.len());
    assert_eq!(m["route_counts"]["backbone"], GENERAL_TEXTS.len() + 1);
    // The key-miss rate (spec §5): decisions FOR the record vs. hits vs.
    // misses; `decided_counts` keeps the router's choice, `route_counts`
    // the lane that ran (a miss ran the backbone).
    let targets = m["lookup_targets"].as_u64().unwrap();
    let misses = m["lookup_misses"].as_u64().unwrap();
    assert_eq!(targets, SKILL_TEXTS.len() as u64 + misses, "{m}");
    assert_eq!(m["lookup_off"], 0);
    assert_eq!(m["decided_counts"][SKILL_ID], targets);
    assert_eq!(
        m["decided_counts"]["backbone"].as_u64().unwrap() + targets,
        rows_total(m)
    );
    // The fixture's `src` values are set names, not plant names: no hit
    // can be checked against its source.
    assert_eq!(m["lookup_src_known"], 0);
    assert_eq!(m["lookup_src_matches"], 0);
    let rows = m["per_prompt"].as_array().unwrap();
    assert_eq!(rows.len(), 2 * SKILL_TEXTS.len() + 1);
    for (i, row) in rows.iter().enumerate() {
        let is_skill = i % 2 == 1 && i < 2 * SKILL_TEXTS.len();
        assert_eq!(row["lookup_hit"], is_skill, "{row}");
        assert_eq!(row["route"]["lookup_hit"], is_skill, "{row}");
        if is_skill {
            let e = i / 2;
            assert_eq!(row["route"]["target"], SKILL_ID);
            assert_eq!(row["route"]["decided_target"], SKILL_ID);
            assert_eq!(row["route"]["lookup_entry"], e);
            assert_eq!(row["route"]["lookup_lang"], "ru");
            assert_eq!(row["route"]["lookup_mode"], "answer");
            assert!(row["route"]["lookup_src_entry"].is_null(), "{row}");
            assert!(row["route"]["lookup_src_match"].is_null());
            assert_eq!(row["finish_reason"], "lookup");
            assert_eq!(row["generated"], 0);
            let card = &entries[e].1;
            let want = if e == 4 {
                assert_eq!(row["route"]["field"], "family");
                card["fields"]["family"].as_str().unwrap()
            } else {
                assert!(row["route"]["field"].is_null(), "{row}");
                card["card"].as_str().unwrap()
            };
            assert_eq!(row["answer"], want);
        } else {
            assert_eq!(row["route"]["target"], "backbone", "{row}");
            assert_ne!(row["finish_reason"], "lookup");
        }
    }
    // The no-key in-domain row: backbone, not a hit, whatever the router
    // said first — and what it said first is kept as `decided_target`.
    let last = &rows[rows.len() - 1];
    assert_eq!(last["label"], "ru:nokey");
    assert_eq!(last["route"]["target"], "backbone", "{last}");
    assert_eq!(last["lookup_hit"], false);
    if misses == 1 {
        assert_eq!(last["route"]["decided_target"], SKILL_ID, "{last}");
        assert_eq!(last["route"]["lookup_mode"], "answer");
    } else {
        assert!(last["route"].get("decided_target").is_none(), "{last}");
    }
    // Rows whose `src` names the plant: the hit is checked against the
    // entry the source resolves to (G5 reports the share of hits that
    // answered from the right card).
    let srcd = fx.dir.join("srcd.jsonl");
    write_jsonl(
        &srcd,
        &[
            (SKILL_TEXTS[4], "ru", "Calendula officinalis"),
            (SKILL_TEXTS[2], "ru", "Шалфей лекарственный"),
            (SKILL_TEXTS[0], "ru", "Salvia officinalis"),
            (GENERAL_TEXTS[0], "en", "Calendula officinalis"),
        ],
    );
    let v = json(&["probe-utility", s(&fx.f1), "--prompts-jsonl", s(&srcd), "--max-tokens", "2", "--route", "auto", "--json"]);
    let m = &v[0];
    assert_eq!(m["lookup_hits"], 3, "{m}");
    assert_eq!(m["lookup_src_known"], 3);
    assert_eq!(m["lookup_src_matches"], 2, "the chamomile question's src names sage");
    let srows = m["per_prompt"].as_array().unwrap();
    assert_eq!((srows[0]["route"]["lookup_src_entry"].as_u64(), srows[0]["route"]["lookup_src_match"].as_bool()), (Some(4), Some(true)), "{}", srows[0]);
    assert_eq!((srows[1]["route"]["lookup_src_entry"].as_u64(), srows[1]["route"]["lookup_src_match"].as_bool()), (Some(2), Some(true)));
    assert_eq!((srows[2]["route"]["lookup_src_entry"].as_u64(), srows[2]["route"]["lookup_src_match"].as_bool()), (Some(2), Some(false)), "{}", srows[2]);
    assert!(srows[3]["route"].get("lookup_src_entry").is_none(), "no hit: nothing to check");
    // Mode off: everything runs the backbone, nothing is looked up.
    let v = json(&[
        "probe-utility",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--lookup-mode",
        "off",
        "--json",
    ]);
    assert_eq!(v[0]["lookup_mode"], "off");
    assert_eq!(v[0]["lookup_hits"], 0);
    assert_eq!(v[0]["route_counts"]["backbone"], rows.len());
    // Mode context: the hits generate (on the backbone, with the card).
    let v = json(&[
        "probe-utility",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--lookup-mode",
        "context",
        "--json",
    ]);
    assert_eq!(v[0]["lookup_hits"], SKILL_TEXTS.len());
    let row = &v[0]["per_prompt"][1];
    assert_eq!(row["route"]["target"], SKILL_ID, "{row}");
    assert_eq!(row["route"]["lookup_mode"], "context");
    assert_eq!(row["route"]["lookup_hit"], true);
    assert_ne!(row["finish_reason"], "lookup", "{row}");
    assert!(row["prompt_tokens"].as_u64().unwrap() > 0, "{row}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn dump_logits_backbone_records_stay_bit_identical_to_f0() {
    let fx = fixture("dump");
    let n_bb = GENERAL_TEXTS.len() + 1;
    let (d0, d1, d2) = (
        fx.dir.join("f0.bin"),
        fx.dir.join("f1.bin"),
        fx.dir.join("f1-context.bin"),
    );
    let j0 = json(&[
        "dump-logits",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--tokens",
        "3",
        "--out",
        s(&d0),
    ]);
    assert_eq!(j0["route_mode"], "none");
    assert_eq!(j0["lookup_hits"], 0);
    let j1 = json(&[
        "dump-logits",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--tokens",
        "3",
        "--route",
        "auto",
        "--out",
        s(&d1),
    ]);
    assert_eq!(j1["lookup_mode"], "answer", "{j1}");
    assert_eq!(j1["lookup_hits"], SKILL_TEXTS.len());
    assert_eq!(j1["route_counts"]["backbone"], n_bb);
    assert_eq!(j1["route_counts"][SKILL_ID], SKILL_TEXTS.len());
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.dir.join("f1.bin.meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["lookup_mode"], "answer");
    let c = json(&["logits-compare", s(&d0), s(&d1)]);
    assert_eq!(c["matched"], n_bb + SKILL_TEXTS.len(), "{c}");
    assert_eq!(c["b_backbone_records"], n_bb);
    assert_eq!(c["b_backbone_bit_identical"], n_bb);
    assert_eq!(c["b_backbone_max_abs_diff"], 0.0);
    assert_eq!(c["g2_pass"], true);
    // `answer` mode: the hit records carry the backbone's untouched view
    // of the plain prompt — bit-identical to F0 as well.
    assert_eq!(c["per_route_b"]["1"]["n"], SKILL_TEXTS.len());
    assert_eq!(c["per_route_b"]["1"]["bit_identical"], SKILL_TEXTS.len());
    assert_eq!(c["max_abs_diff"], 0.0);
    // `context` mode: the hits run the backbone on the card-prepended
    // prompt — different logits, still G2 (the backbone records are
    // untouched).
    let j2 = json(&[
        "dump-logits",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--tokens",
        "3",
        "--route",
        "auto",
        "--lookup-mode",
        "context",
        "--out",
        s(&d2),
    ]);
    assert_eq!(j2["lookup_hits"], SKILL_TEXTS.len());
    let c = json(&["logits-compare", s(&d0), s(&d2)]);
    assert_eq!(c["b_backbone_bit_identical"], n_bb, "{c}");
    assert_eq!(c["g2_pass"], true);
    assert_eq!(c["per_route_b"]["1"]["bit_identical"], 0);
    assert!(c["max_abs_diff"].as_f64().unwrap() > 0.0);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// Plant-naming prompts in ENGLISH framing (the synthetic router's
/// backbone class) with a STRONG key — a capitalised binomial or a
/// two-word English name — and the entry each names.
const STRONG_KEY_GENERAL: &[(&str, usize)] = &[
    ("Write a Rust function that returns the family of Matricaria chamomilla.", 0),
    ("What is the capital of France, and where does pot marigold grow?", 4),
    ("Explain in two sentences why Hypericum perforatum has yellow flowers.", 1),
    ("Say what poison hemlock is made of.", 5),
];
/// A one-word key (`calendula`) in English framing.
const ONE_WORD_KEY_GENERAL: &str = "Say what calendula is made of.";

#[test]
fn key_first_answers_backbone_routed_plant_prompts_from_the_table() {
    let fx = fixture("keyfirst");
    let entries = lookup_entries();
    let fk = fx.dir.join("f1-key-first.cmf");
    std::fs::copy(&fx.f1, &fk).unwrap();
    let len1 = std::fs::metadata(&fk).unwrap().len();

    // ── lookup-policy: the header-only switch ──
    // The default already reads as router_and_key: nothing to append
    // (review KF-6).
    let p = json(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "router_and_key"]);
    assert_eq!(p["changed"], false, "{p}");
    assert_eq!(std::fs::metadata(&fk).unwrap().len(), len1);
    let p = json(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "key_first"]);
    assert_eq!(p["policy_from"], "router_and_key", "{p}");
    assert_eq!(p["policy_to"], "key_first");
    assert_eq!(p["changed"], true);
    assert_eq!(p["calibration_stale"], false, "the policy is not in skills_hash");
    // The record was active with a gate measured WITHOUT the key_first
    // step: it goes to stale_regate (review KF-2).
    assert_eq!((p["status_from"].as_str(), p["status"].as_str()), (Some("active"), Some("stale_regate")), "{p}");
    assert_eq!(p["regate_required"], true);
    assert_eq!((p["auto_routable"].as_bool(), p["auto_routable_before"].as_bool()), (Some(false), Some(true)));
    assert_eq!(p["old_len"], len1);
    let len2 = std::fs::metadata(&fk).unwrap().len();
    assert!(len2 > len1);
    let g = json(&["genome-verify", s(&fx.f1), s(&fk)]);
    assert_eq!(g["pass"], true, "{g}");
    let m = cortiq_core::CmfModel::open(&fk).unwrap();
    let rec = &m.header.skills[0];
    assert_eq!(rec.lookup.as_ref().unwrap().policy.as_deref(), Some("key_first"));
    let ev = m.header.lineage.last().unwrap();
    assert_eq!((ev.event.as_str(), ev.detail["policy_to"].as_str()), ("lookup_policy", Some("key_first")));
    assert_eq!(ev.detail["status_to"], "stale_regate");
    assert_eq!(rec.status.as_deref(), Some("stale_regate"));
    drop(m);
    // Idempotent: the same policy again appends nothing.
    let p = json(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "key_first"]);
    assert_eq!(p["changed"], false, "{p}");
    assert_eq!(std::fs::metadata(&fk).unwrap().len(), len2);
    // Refusals leave the file alone.
    let bad = cortiq(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "always"]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("router_and_key | key_first"));
    assert!(!cortiq(&["lookup-policy", s(&fk), "--id", "ghost", "--policy", "key_first"]).status.success());
    assert_eq!(std::fs::metadata(&fk).unwrap().len(), len2);
    // --keep-gate: the operator vouches — active stays active.
    let fkeep = fx.dir.join("f1-key-first-keep.cmf");
    std::fs::copy(&fx.f1, &fkeep).unwrap();
    let (so, se) = ok(&["lookup-policy", s(&fkeep), "--id", SKILL_ID, "--policy", "key_first", "--keep-gate"]);
    let pk: serde_json::Value = serde_json::from_str(&so).unwrap();
    assert_eq!((pk["status"].as_str(), pk["auto_routable"].as_bool(), pk["regate_required"].as_bool()), (Some("active"), Some(true), Some(false)), "{pk}");
    assert!(se.contains("--keep-gate"), "{se}");

    // ── the re-gate: route-eval applies the key_first step (a take is an
    // accept of the record), skill-gate accepts only such a gate ──
    let set = fx.dir.join("keyfirst.jsonl");
    let mut rows: Vec<(&str, &str, &str)> =
        STRONG_KEY_GENERAL.iter().map(|(t, _)| (*t, "en", "strong")).collect();
    rows.push((ONE_WORD_KEY_GENERAL, "en", "oneword"));
    rows.extend(GENERAL_TEXTS.iter().map(|t| (*t, "en", "general")));
    write_jsonl(&set, &rows);
    let n = rows.len();
    let ns = STRONG_KEY_GENERAL.len();
    let re = |f: &Path, extra: &[&str]| {
        let mut args = vec!["route-eval", s(f), "--prompts-jsonl", s(&set), "--expect", "backbone", "--include-quarantine", "--json"];
        args.extend_from_slice(extra);
        json(&args)
    };
    let g = re(&fk, &[]);
    assert_eq!((g["status"].as_str(), g["key_first_step"].as_bool()), (Some("measured"), Some(true)), "{g}");
    assert_eq!(g["key_first_records"], serde_json::json!([SKILL_ID]));
    assert_eq!((g["n"].as_u64(), g["errors"].as_u64()), (Some(n as u64), Some(ns as u64)), "every take is a false accept: {g}");
    assert_eq!(g["key_first_accepts"], ns);
    assert_eq!(g["decided_by_counts"], serde_json::json!({"key_first": ns}));
    for (i, r) in g["rows"].as_array().unwrap().iter().enumerate() {
        assert_eq!(r["decided_by"].as_str(), (i < ns).then_some("key_first"), "{r}");
    }
    // The φ router alone (the old G3) sees none of it — no gate for key_first.
    let g_router = re(&fk, &["--router-only"]);
    assert_eq!((g_router["errors"].as_u64(), g_router["key_first_step"].as_bool()), (Some(0), Some(false)), "{g_router}");
    let gate_router = fx.dir.join("gate-router-only.json");
    std::fs::write(&gate_router, g_router.to_string()).unwrap();
    let out = cortiq(&["skill-gate", s(&fk), "--id", SKILL_ID, "--gate", s(&gate_router), "--status", "active"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not measured with the key_first step"), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::metadata(&fk).unwrap().len(), len2, "a refused skill-gate touched the file");
    // The same file under router_and_key: the step does not apply.
    let g1 = re(&fx.f1, &[]);
    assert_eq!((g1["errors"].as_u64(), g1["key_first_step"].as_bool()), (Some(0), Some(false)), "{g1}");
    let gate = fx.dir.join("gate-key-first.json");
    std::fs::write(&gate, g.to_string()).unwrap();
    let sg = json(&["skill-gate", s(&fk), "--id", SKILL_ID, "--gate", s(&gate), "--status", "active"]);
    assert_eq!((sg["key_first"].as_bool(), sg["gate_covers_key_first"].as_bool(), sg["auto_routable"].as_bool()), (Some(true), Some(true), Some(true)), "{sg}");
    // With the key_first gate on record, a round trip through
    // router_and_key keeps the record active.
    let p = json(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "router_and_key"]);
    assert_eq!((p["changed"].as_bool(), p["status"].as_str()), (Some(true), Some("active")), "{p}");
    let p = json(&["lookup-policy", s(&fk), "--id", SKILL_ID, "--policy", "key_first"]);
    assert_eq!((p["gate_covers_key_first"].as_bool(), p["regate_required"].as_bool(), p["status"].as_str()), (Some(true), Some(false), Some("active")), "{p}");

    // ── probe-utility: router_and_key (F1) vs key_first (Fk) ──
    let probe = |f: &Path| {
        json(&["probe-utility", s(f), "--prompts-jsonl", s(&set), "--max-tokens", "2", "--route", "auto", "--json"])
    };
    // router_and_key: the φ router sends every row to the backbone — the
    // plant prompts included (the recall miss key_first is for).
    let v = probe(&fx.f1);
    let m1 = &v[0];
    assert_eq!(m1["route_counts"]["backbone"], n, "precondition: {m1}");
    assert_eq!(m1["lookup_hits"], 0);
    assert_eq!(m1["lookup_targets"], 0);
    assert_eq!(m1["decided_by_counts"], serde_json::json!({}));
    for row in m1["per_prompt"].as_array().unwrap() {
        assert_eq!(row["route"]["target"], "backbone", "{row}");
        assert_eq!(row["lookup_hit"], false);
        assert!(row["route"].get("decided_by").is_none(), "{row}");
        assert_ne!(row["finish_reason"], "lookup");
    }
    // key_first: the plant prompts are answered from the table.
    let v = probe(&fk);
    let mk = &v[0];
    assert_eq!(mk["lookup_hits"], ns, "{mk}");
    assert_eq!(mk["lookup_targets"], ns);
    assert_eq!(mk["lookup_misses"], 0);
    assert_eq!(mk["route_counts"][SKILL_ID], ns);
    assert_eq!(mk["route_counts"]["backbone"], n - ns);
    assert_eq!(mk["decided_counts"][SKILL_ID], ns);
    assert_eq!(mk["decided_counts"]["backbone"], n - ns);
    assert_eq!(mk["decided_by_counts"], serde_json::json!({"key_first": ns}));
    let prows = mk["per_prompt"].as_array().unwrap();
    for (i, row) in prows.iter().enumerate() {
        let r = &row["route"];
        if i < ns {
            let e = STRONG_KEY_GENERAL[i].1;
            assert_eq!(row["lookup_hit"], true, "{row}");
            assert_eq!(r["target"], SKILL_ID);
            assert_eq!(r["decided_target"], SKILL_ID);
            assert_eq!(r["decided_by"], "key_first", "{row}");
            assert_eq!(r["lookup_entry"], e);
            assert_eq!(r["lookup_lang"], "en");
            assert!(r["lookup_key_words"].as_u64().unwrap() >= 2, "{row}");
            assert!(r["reason"].as_str().unwrap().starts_with("key_first: lookup 'herbs'"), "{row}");
            assert_eq!(row["finish_reason"], "lookup");
            assert_eq!(row["generated"], 0);
            let card = &entries[e].2;
            let want = match r["field"].as_str() {
                Some(f) => card["fields"][f].as_str().unwrap(),
                None => card["card"].as_str().unwrap(),
            };
            assert_eq!(row["answer"], want, "{row}");
        } else {
            assert_eq!(row["lookup_hit"], false, "{row}");
            assert_eq!(r["target"], "backbone", "{row}");
            assert!(r.get("decided_by").is_none(), "{row}");
            // The same backbone answer as under router_and_key.
            assert_eq!(row["answer"], m1["per_prompt"][i]["answer"], "{row}");
        }
    }
    assert_eq!(prows[0]["route"]["field"], "family");
    assert_eq!(prows[0]["answer"], entries[0].2["fields"]["family"]);
    // The in-scope set on key_first: the router's own decisions stay
    // `router` (one-word key `календулы` included), counts consistent.
    let v = json(&["probe-utility", s(&fk), "--prompts-jsonl", s(&fx.mixed), "--max-tokens", "2", "--route", "auto", "--json"]);
    let mm = &v[0];
    assert_eq!(mm["lookup_hits"], SKILL_TEXTS.len(), "{mm}");
    let by = &mm["decided_by_counts"];
    assert_eq!(by["router"].as_u64().unwrap(), mm["lookup_targets"].as_u64().unwrap(), "{mm}");
    assert!(by.get("key_first").is_none(), "{mm}");
    // Pinned routes are taken as given: --route backbone never looks up.
    let v = json(&["probe-utility", s(&fk), "--prompts-jsonl", s(&set), "--max-tokens", "2", "--route", "backbone", "--json"]);
    assert_eq!(v[0]["lookup_hits"], 0, "{}", v[0]);
    assert_eq!(v[0]["route_counts"]["backbone"], n);

    // ── run ──
    let family_en = entries[0].2["fields"]["family"].as_str().unwrap();
    let (so, se) = ok(&["run", s(&fk), "-p", STRONG_KEY_GENERAL[0].0, "-n", "2", "--greedy"]);
    assert!(se.contains("route: herbs") && se.contains("key_first: lookup 'herbs'"), "{se}");
    assert!(se.contains("decided by key_first"), "{se}");
    assert!(so.contains(family_en) && so.contains("answered from the table"), "{so}");
    let (so, se) = ok(&["run", s(&fx.f1), "-p", STRONG_KEY_GENERAL[0].0, "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone") && !se.contains("lookup:"), "{se}");
    assert!(!so.contains("answered from the table"), "{so}");
    let (so, se) = ok(&["run", s(&fk), "-p", ONE_WORD_KEY_GENERAL, "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone") && !se.contains("key_first"), "{se}");
    assert!(!so.contains("answered from the table"), "{so}");
    // --skill none pins the backbone; --lookup-mode off ignores the table.
    let (so, _) = ok(&["run", s(&fk), "-p", STRONG_KEY_GENERAL[0].0, "-n", "2", "--greedy", "--skill", "none"]);
    assert!(!so.contains("answered from the table"), "{so}");
    let (so, se) = ok(&["run", s(&fk), "-p", STRONG_KEY_GENERAL[0].0, "-n", "2", "--greedy", "--lookup-mode", "off"]);
    assert!(!so.contains("answered from the table") && !se.contains("key_first"), "{se}");

    // ── dump-logits: every backbone record bit-identical to F0 ──
    let (d0, dk) = (fx.dir.join("kf0.bin"), fx.dir.join("kfk.bin"));
    json(&["dump-logits", s(&fx.f0), "--prompts-jsonl", s(&set), "--tokens", "3", "--out", s(&d0)]);
    let jk = json(&["dump-logits", s(&fk), "--prompts-jsonl", s(&set), "--tokens", "3", "--route", "auto", "--out", s(&dk)]);
    assert_eq!(jk["lookup_hits"], ns, "{jk}");
    assert_eq!(jk["lookup_key_first"], ns);
    assert_eq!(jk["route_counts"]["backbone"], n - ns);
    assert_eq!(jk["key_first_indices"], serde_json::json!((0..ns).collect::<Vec<_>>()));
    let c = json(&["logits-compare", s(&d0), s(&dk)]);
    assert_eq!(c["b_backbone_records"], n - ns, "{c}");
    assert_eq!(c["b_backbone_bit_identical"], n - ns);
    // `answer` mode: the key_first hits carry the backbone's own view of
    // the plain prompt — bit-identical too — yet F0 answered these
    // prompts with the backbone and F1 answers them from the table: G2
    // fails on them (review KF-2).
    assert_eq!(c["per_route_b"]["1"]["bit_identical"], ns, "{c}");
    assert_eq!(c["max_abs_diff"], 0.0);
    assert_eq!((c["b_key_first_known"].as_bool(), c["b_key_first_records"].as_u64()), (Some(true), Some(ns as u64)));
    assert_eq!(c["a_backbone_b_key_first"], ns);
    assert_eq!(c["g2_pass"], false, "{c}");
    // Without the plant prompts (a general set no strong key hits) G2
    // passes on the key_first file.
    let only_general = fx.dir.join("only-general.json");
    std::fs::write(&only_general, serde_json::json!((ns..n).collect::<Vec<_>>()).to_string()).unwrap();
    let c = json(&["logits-compare", s(&d0), s(&dk), "--only-indices", s(&only_general)]);
    assert_eq!((c["a_backbone_b_key_first"].as_u64(), c["g2_pass"].as_bool()), (Some(0), Some(true)), "{c}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}
