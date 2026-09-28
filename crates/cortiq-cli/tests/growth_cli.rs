//! The growth-record gate tools end to end through the `cortiq` binary
//! (spec §2), on a synthetic resonance genome (F0) and its append-only
//! successor with one quarantined `expert_append` record (F1):
//!
//! * `genome-verify F0 F1` — G1 holds for a grown file;
//! * `growth-eval` — vacuous (non-zero exit) under the default growth
//!   mode on a quarantined record; `--growth all --shell both` reports
//!   both modes (shell on: no grown win anywhere, shell off: the magnet
//!   wins), breakdowns, the CP-95 bound, `--indices-out` = the no-hit set
//!   of the shell-on run; `CMF_GROWTH_SHELL=off` is honoured without a
//!   `--shell` flag;
//! * `dump-logits` F0 vs F1 (`CMF_GROWTH=all`) + `logits-compare
//!   --only-indices` — G2: every no-hit record bit-identical; the same
//!   dump with `CMF_GROWTH_SHELL=off` differs (the magnet ran), and a
//!   plain JSON array restricts the comparison too;
//! * one forward path for hits and dumps — `growth-eval` measures per-op
//!   only and records it, `dump-logits` records its `--path` in
//!   `<out>.meta.json`, `logits-compare` refuses dumps / index sets of
//!   different (or unknown) paths.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/growth_synth.rs"]
mod growth_synth;

use growth_synth::{COUNT, E0, GROWN_LAYERS, RANK, RECORD_ID, tempdir, write_growth_pair};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PROMPTS: &[(&str, &str, &str)] = &[
    ("What is the capital of France?", "en", "general"),
    ("Write a Rust function that returns the maximum element.", "en", "code"),
    ("Explain why Earth has seasons in two sentences.", "en", "general"),
    ("Compute exactly: 17 * 19 + 23.", "en2", "math"),
    ("Say what water is made of.", "en2", "general"),
    ("Why does a hash table offer constant-time lookup?", "en2", "code"),
];

fn cortiq(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cortiq"));
    c.args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_GROWTH")
        .env_remove("CMF_GROWTH_SHELL");
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().expect("spawn cortiq")
}

fn ok(args: &[&str], env: &[(&str, &str)]) -> (String, String) {
    let out = cortiq(args, env);
    let (so, se) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(
        out.status.success(),
        "cortiq {args:?} {env:?} failed\nstdout:\n{so}\nstderr:\n{se}"
    );
    (so, se)
}

fn json(args: &[&str], env: &[(&str, &str)]) -> serde_json::Value {
    let (so, _) = ok(args, env);
    serde_json::from_str(&so).unwrap_or_else(|e| panic!("cortiq {args:?}: not JSON ({e}):\n{so}"))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

struct Fixture {
    dir: PathBuf,
    f0: PathBuf,
    f1: PathBuf,
    prompts: PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    // SAFETY: before any pipeline of this process exists (the fixture
    // writer opens no pipeline, but the binary under test inherits it).
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = tempdir(&format!("cli-{tag}"));
    let (f0, f1) = write_growth_pair(&dir, "quarantine");
    let prompts = dir.join("prompts.jsonl");
    let text: String = PROMPTS
        .iter()
        .map(|(p, lang, src)| {
            serde_json::json!({"prompt": p, "lang": lang, "src": src}).to_string() + "\n"
        })
        .collect();
    std::fs::write(&prompts, text).unwrap();
    Fixture {
        dir,
        f0,
        f1,
        prompts,
    }
}

#[test]
fn genome_verify_growth_eval_and_logit_identity_on_a_grown_file() {
    let fx = fixture("gates");
    let n = PROMPTS.len();

    // G1: the grown file is an append-only successor of the genome.
    let g1 = json(&["genome-verify", s(&fx.f0), s(&fx.f1)], &[]);
    assert_eq!(g1["pass"], true, "{g1}");
    assert_eq!(g1["equal"], true);
    assert_eq!(g1["prefix_bytes_equal"], true);
    assert_eq!(g1["trunk_hash_f0"], g1["trunk_hash_f1"]);
    assert!(g1["entries_f1"].as_u64() > g1["entries_f0"].as_u64());
    let bad = cortiq(&["genome-verify", s(&fx.f1), s(&fx.f0)], &[]);
    assert!(!bad.status.success(), "F0 is not a successor of F1");

    // A quarantined record under the default growth mode: vacuous.
    fn eval_args<'a>(fx: &'a Fixture, extra: &[&'a str]) -> Vec<&'a str> {
        let mut a = vec![
            "growth-eval",
            s(&fx.f1),
            "--prompts-jsonl",
            s(&fx.prompts),
            "--max-tokens",
            "4",
            "--json",
        ];
        a.extend_from_slice(extra);
        a
    }
    let eval = |extra: &[&str]| -> Vec<String> {
        eval_args(&fx, extra).into_iter().map(String::from).collect()
    };
    fn refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    let v = cortiq(&refs(&eval(&[])), &[]);
    assert!(!v.status.success(), "a vacuous measurement passed");
    let se = String::from_utf8_lossy(&v.stderr);
    assert!(se.contains("VACUOUS"), "{se}");
    assert!(se.contains("--growth all"), "{se}");
    let vj: serde_json::Value = serde_json::from_slice(&v.stdout).expect("JSON even when vacuous");
    assert_eq!(vj["vacuous"], true);
    assert_eq!(vj["mounted"].as_array().unwrap().len(), 0);
    assert_eq!(vj["expert_append_records"][0]["id"], RECORD_ID);
    assert_eq!(vj["expert_append_records"][0]["status"], "quarantine");

    // Both shell modes on the mounted record.
    let idx = fx.dir.join("nohit.json");
    let ev = json(
        &refs(&eval(&["--growth", "all", "--shell", "both", "--indices-out", s(&idx)])),
        &[],
    );
    assert_eq!(ev["vacuous"], false, "{ev}");
    assert_eq!(ev["growth_mode"], "all");
    assert_eq!(ev["n"], n);
    assert_eq!(ev["max_tokens"], 4);
    assert_eq!(ev["positions_per_record"], 5);
    let mounted = ev["mounted"].as_array().unwrap();
    assert_eq!(mounted.len(), 1);
    assert_eq!(mounted[0]["id"], RECORD_ID);
    assert_eq!(mounted[0]["index"], 0);
    assert_eq!(mounted[0]["status"], "quarantine");
    assert_eq!(mounted[0]["layers"], serde_json::json!(GROWN_LAYERS));
    assert_eq!(mounted[0]["count"], COUNT);
    assert_eq!(mounted[0]["rank"], RANK);
    let modes = ev["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 2);
    let (on, off) = (&modes[0], &modes[1]);
    assert_eq!(on["shell"], "on");
    assert_eq!(off["shell"], "off");
    // Shell on: the far expert is outside its shell, the magnet too —
    // no grown win anywhere, every prompt is a no-hit.
    assert_eq!(on["n"], n);
    assert_eq!(on["hits"], 0, "{on}");
    assert_eq!(on["no_hit"], n);
    assert_eq!(on["hit_rate"], 0.0);
    let up = on["hit_upper95"].as_f64().unwrap();
    let want = 1.0 - 0.025f64.powf(1.0 / n as f64);
    assert!((up - want).abs() < 1e-9, "{up} vs {want}");
    assert_eq!(
        on["no_hit_indices"],
        serde_json::json!((0..n).collect::<Vec<_>>())
    );
    for l in GROWN_LAYERS {
        let pl = &on["per_layer"][l.to_string()];
        assert_eq!(pl["trunk_experts"], E0, "{pl}");
        assert_eq!(pl["grown_experts"], COUNT);
        assert_eq!(pl["grown_wins"], 0);
        assert_eq!(pl["prompts_hit"], 0);
        assert!(pl["routed_tokens"].as_u64().unwrap() >= 5 * n as u64);
    }
    // Layer 0 is an MoE layer the record does not grow: listed, empty tail.
    assert_eq!(on["per_layer"]["0"]["trunk_experts"], E0, "{on}");
    assert_eq!(on["per_layer"]["0"]["grown_experts"], 0);
    assert_eq!(on["per_layer"]["0"]["grown_wins"], 0);
    assert_eq!(on["per_record"][RECORD_ID]["wins"], 0);
    assert_eq!(on["per_src"]["general"]["n"], 3);
    assert_eq!(on["per_src"]["general"]["hits"], 0);
    assert_eq!(on["per_lang"]["en2"]["n"], 3);
    // Shell off: the magnet wins tokens; at least one prompt hits.
    assert_eq!(off["n"], n);
    let hits_off = off["hits"].as_u64().unwrap();
    assert!(hits_off > 0, "{off}");
    assert!(hits_off <= n as u64);
    assert!(off["per_record"][RECORD_ID]["wins"].as_u64().unwrap() > 0);
    assert!(
        GROWN_LAYERS
            .iter()
            .any(|l| off["per_layer"][l.to_string()]["grown_wins"].as_u64().unwrap() > 0)
    );
    let rows = ev["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2 * n);
    assert!(rows[..n].iter().all(|r| r["shell"] == "on" && r["hit"] == false));
    assert!(rows[n..].iter().all(|r| r["shell"] == "off"));
    assert_eq!(rows[n..].iter().filter(|r| r["hit"] == true).count() as u64, hits_off);
    // The indices file is the shell-on no-hit set.
    let ij: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx).unwrap()).unwrap();
    assert_eq!(ij["shell"], "on", "{ij}");
    assert_eq!(ij["growth"], "all");
    assert_eq!(ij["n"], n);
    assert_eq!(ij["indices"], serde_json::json!((0..n).collect::<Vec<_>>()));
    assert_eq!(ij["prompts_sha256"], ev["prompts_sha256"]);

    // `CMF_GROWTH_SHELL=off` without a --shell flag: one mode, off.
    let e2 = json(&refs(&eval(&["--growth", "all"])), &[("CMF_GROWTH_SHELL", "off")]);
    let m2 = e2["modes"].as_array().unwrap();
    assert_eq!(m2.len(), 1);
    assert_eq!(m2[0]["shell"], "off");
    assert_eq!(m2[0]["hits"], hits_off);
    let e3 = json(&refs(&eval(&["--growth", "all"])), &[]);
    assert_eq!(e3["modes"][0]["shell"], "on");
    assert_eq!(e3["modes"][0]["hits"], 0);

    // G2: F0 vs F1 (growth mounted, shell on) on the no-hit set.
    let (d0, d1, d1off) = (
        fx.dir.join("f0.bin"),
        fx.dir.join("f1.bin"),
        fx.dir.join("f1off.bin"),
    );
    let dump = |model: &Path, out: &Path, env: &[(&str, &str)]| {
        json(
            &[
                "dump-logits",
                s(model),
                "--prompts-jsonl",
                s(&fx.prompts),
                "--tokens",
                "4",
                "--out",
                s(out),
            ],
            env,
        )
    };
    let j0 = dump(&fx.f0, &d0, &[]);
    assert_eq!(j0["route_mode"], "none");
    assert_eq!(j0["positions_per_record"], 5);
    let j1 = dump(&fx.f1, &d1, &[("CMF_GROWTH", "all")]);
    assert_eq!(j1["route_mode"], "none");
    assert_eq!(j1["route_counts"]["backbone"], n);
    let c = json(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&idx)],
        &[],
    );
    assert_eq!(c["only_indices"], n, "{c}");
    assert_eq!(c["matched"], n);
    assert_eq!(c["bit_identical"], n);
    assert_eq!(c["max_abs_diff"], 0.0);
    assert_eq!(c["only_missing_in_a"], 0);
    assert_eq!(c["only_missing_in_b"], 0);
    assert_eq!(c["g2_pass"], true);
    let c_all = json(&["logits-compare", s(&d0), s(&d1)], &[]);
    assert_eq!(c_all["bit_identical"], n, "{c_all}");
    assert_eq!(c_all["g2_pass"], true);
    assert!(c_all["only_indices"].is_null());

    // The same dump with the shell off ran the magnet: not F0's.
    let j1off = dump(
        &fx.f1,
        &d1off,
        &[("CMF_GROWTH", "all"), ("CMF_GROWTH_SHELL", "off")],
    );
    assert_eq!(j1off["records"], n);
    let c_off = json(
        &["logits-compare", s(&d0), s(&d1off), "--only-indices", s(&idx)],
        &[],
    );
    assert_eq!(c_off["matched"], n, "{c_off}");
    assert_eq!(
        c_off["bit_identical"].as_u64().unwrap(),
        n as u64 - hits_off,
        "exactly the no-hit prompts of the shell-off run stay identical: {c_off}"
    );
    assert!(c_off["max_abs_diff"].as_f64().unwrap() > 0.0);
    assert_eq!(c_off["g2_pass"], false);
    // A plain JSON array restricts the comparison too.
    let arr = fx.dir.join("two.json");
    std::fs::write(&arr, "[0, 2]").unwrap();
    let c_two = json(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&arr)],
        &[],
    );
    assert_eq!(c_two["only_indices"], 2, "{c_two}");
    assert_eq!(c_two["matched"], 2);
    assert_eq!(c_two["bit_identical"], 2);
    assert_eq!(c_two["g2_pass"], true);
    // An index outside the dumps is a refusal of G2, not a silent skip.
    std::fs::write(&arr, "[0, 99]").unwrap();
    let c_miss = json(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&arr)],
        &[],
    );
    assert_eq!(c_miss["only_missing_in_a"], 1, "{c_miss}");
    assert_eq!(c_miss["g2_pass"], false);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// G2 is defined on ONE forward path: the no-hit set is measured per-op
/// (the routing counters live on the host route; the graph exports none),
/// so `growth-eval` accepts `--path per-op` only and records it,
/// `dump-logits` records its path in `<out>.meta.json` (default per-op),
/// and `logits-compare` refuses to compare across paths — a comparison of
/// a graph dump against a per-op no-hit set is undefined, not failed.
#[test]
fn g2_refuses_dumps_and_index_sets_of_different_forward_paths() {
    let fx = fixture("paths");
    let n = PROMPTS.len();
    let idx = fx.dir.join("nohit.json");
    let base: Vec<&str> = vec![
        "growth-eval",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.prompts),
        "--max-tokens",
        "2",
        "--json",
        "--growth",
        "all",
        "--indices-out",
        s(&idx),
    ];
    let with = |extra: &[&str]| -> Vec<String> {
        base.iter().copied().chain(extra.iter().copied()).map(String::from).collect()
    };
    fn refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    // growth-eval: per-op is the only measurable path, and it is recorded
    // in the summary and in the index set.
    let ev = json(&refs(&with(&[])), &[]);
    assert_eq!(ev["path"], "per-op", "{ev}");
    assert_eq!(ev["modes"][0]["hits"], 0);
    let ij: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx).unwrap()).unwrap();
    assert_eq!(ij["path"], "per-op", "{ij}");
    assert_eq!(ij["indices"].as_array().unwrap().len(), n);
    let ev2 = json(&refs(&with(&["--path", "per-op"])), &[]);
    assert_eq!(ev2["path"], "per-op");
    let auto = cortiq(&refs(&with(&["--path", "auto"])), &[]);
    assert!(!auto.status.success(), "growth-eval cannot measure the graph");
    let se = String::from_utf8_lossy(&auto.stderr);
    assert!(se.contains("only per-op is measurable"), "{se}");
    let bad = cortiq(&refs(&with(&["--path", "graph"])), &[]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("expected per-op | auto"));

    // dump-logits: the default path is per-op; the sidecar records it with
    // the growth / shell modes of the run.
    let dump = |model: &Path, out: &Path, extra: &[&str], env: &[(&str, &str)]| {
        let mut a = vec![
            "dump-logits",
            s(model),
            "--prompts-jsonl",
            s(&fx.prompts),
            "--tokens",
            "2",
            "--out",
            s(out),
        ];
        a.extend_from_slice(extra);
        json(&a, env)
    };
    let meta_of = |out: &Path| -> serde_json::Value {
        let p = format!("{}.meta.json", s(out));
        serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))).unwrap()
    };
    let d0 = fx.dir.join("f0.bin");
    let j0 = dump(&fx.f0, &d0, &[], &[]);
    assert_eq!(j0["path"], "per-op", "{j0}");
    assert_eq!(j0["meta"], format!("{}.meta.json", s(&d0)));
    let m0 = meta_of(&d0);
    assert_eq!(m0["path"], "per-op", "{m0}");
    assert_eq!(m0["growth"], "active");
    assert_eq!(m0["shell"], "on");
    assert_eq!(m0["records"], n);
    assert_eq!(m0["prompts_sha256"], ij["prompts_sha256"]);
    let d1 = fx.dir.join("f1.bin");
    let j1 = dump(&fx.f1, &d1, &["--path", "per-op"], &[("CMF_GROWTH", "all")]);
    assert_eq!(j1["path"], "per-op");
    assert_eq!(meta_of(&d1)["growth"], "all");
    let c = json(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&idx)],
        &[],
    );
    assert_eq!(c["g2_pass"], true, "{c}");
    assert_eq!(c["bit_identical"], n);
    assert_eq!(c["path_a"], "per-op");
    assert_eq!(c["path_b"], "per-op");
    assert_eq!(c["path_indices"], "per-op");
    assert_eq!(c["growth_b"], "all");

    // The candidate dumped on `auto`: refused under the per-op index set
    // (an error, not a `g2_pass: false`) and against the per-op reference
    // even without one.
    let d1auto = fx.dir.join("f1auto.bin");
    let ja = dump(&fx.f1, &d1auto, &["--path", "auto"], &[("CMF_GROWTH", "all")]);
    assert_eq!(ja["path"], "auto", "{ja}");
    assert_eq!(meta_of(&d1auto)["path"], "auto");
    let r = cortiq(
        &["logits-compare", s(&d0), s(&d1auto), "--only-indices", s(&idx)],
        &[],
    );
    assert!(!r.status.success(), "a per-op reference against an auto candidate");
    let se = String::from_utf8_lossy(&r.stderr);
    assert!(se.contains("per-op") && se.contains("auto"), "{se}");
    let r = cortiq(&["logits-compare", s(&d0), s(&d1auto)], &[]);
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("defined on one path"));
    // Two `auto` dumps compare with each other (the graph on both sides —
    // per-op here, CMF_GPU=0) and with a plain array, never with the
    // per-op index set.
    let d0auto = fx.dir.join("f0auto.bin");
    dump(&fx.f0, &d0auto, &["--path", "auto"], &[]);
    let ca = json(&["logits-compare", s(&d0auto), s(&d1auto)], &[]);
    assert_eq!(ca["g2_pass"], true, "{ca}");
    assert_eq!(ca["path_a"], "auto");
    let arr = fx.dir.join("two.json");
    std::fs::write(&arr, "[0, 1]").unwrap();
    let ca2 = json(
        &["logits-compare", s(&d0auto), s(&d1auto), "--only-indices", s(&arr)],
        &[],
    );
    assert_eq!(ca2["only_indices"], 2, "{ca2}");
    assert!(ca2["path_indices"].is_null());
    let r = cortiq(
        &["logits-compare", s(&d0auto), s(&d1auto), "--only-indices", s(&idx)],
        &[],
    );
    assert!(!r.status.success(), "a per-op index set on auto dumps");

    // A dump without its sidecar is of unknown path: refused under the
    // per-op index set, accepted with a plain array or without a set.
    std::fs::remove_file(format!("{}.meta.json", s(&d1))).unwrap();
    let r = cortiq(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&idx)],
        &[],
    );
    assert!(!r.status.success());
    let se = String::from_utf8_lossy(&r.stderr);
    assert!(se.contains("records no path"), "{se}");
    let c_arr = json(
        &["logits-compare", s(&d0), s(&d1), "--only-indices", s(&arr)],
        &[],
    );
    assert_eq!(c_arr["g2_pass"], true, "{c_arr}");
    assert!(c_arr["path_b"].is_null());
    let c_no = json(&["logits-compare", s(&d0), s(&d1)], &[]);
    assert_eq!(c_no["g2_pass"], true, "{c_no}");
    // An index set that claims another path than the dumps: refused.
    let mut foreign = ij.clone();
    foreign["path"] = serde_json::Value::String("graph".into());
    let idx2 = fx.dir.join("foreign.json");
    std::fs::write(&idx2, foreign.to_string()).unwrap();
    let r = cortiq(
        &["logits-compare", s(&d0), s(&d0), "--only-indices", s(&idx2)],
        &[],
    );
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("measured on the graph forward path"));
    // Re-dumping refreshes the sidecar (a stale one never describes a
    // new file).
    dump(&fx.f1, &d1, &[], &[("CMF_GROWTH", "all")]);
    assert_eq!(meta_of(&d1)["path"], "per-op");
    let _ = std::fs::remove_dir_all(&fx.dir);
}
