//! The format-v2 knowledge commands end to end through the `cortiq`
//! binary, on a synthetic GDN + bounded genome (F0) and its append-only
//! successor with one `ffn_replace` skill and a calibrated router (F1):
//!
//! * `genome-verify F0 F1` — G1 (trunk hash, entries, byte prefix);
//! * `route-eval` — accept/recall, CP upper bound, breakdowns, sha256;
//! * `dump-logits` F0 (no router) vs F1 `--route auto`, `logits-compare` —
//!   G2: every F1 record routed to the backbone is bit-identical to F0's;
//! * `skill-gate` — quarantine by a header-only append (lineage event,
//!   still a G1 successor), then routing ignores the skill unless
//!   `--include-quarantine`;
//! * `probe-utility --route`, `run` and `explain` print the decision.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .output()
        .expect("spawn cortiq")
}

fn ok(args: &[&str]) -> (String, String) {
    let out = cortiq(args);
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
    general: PathBuf,
    skill: PathBuf,
    mixed: PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    // SAFETY: set before any pipeline of this process exists (the writer
    // computes φ); every test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-knowledge-cli-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_knowledge_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let general = dir.join("general.jsonl");
    let skill = dir.join("skill.jsonl");
    let mixed = dir.join("mixed.jsonl");
    let g: Vec<(&str, &str, &str)> = GENERAL_TEXTS
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (
                *t,
                if i % 2 == 0 { "en" } else { "en2" },
                if i < 2 { "audit" } else { "general" },
            )
        })
        .collect();
    let k: Vec<(&str, &str, &str)> = SKILL_TEXTS
        .iter()
        .map(|t| (*t, "ru", "herbs-dev"))
        .collect();
    write_jsonl(&general, &g);
    write_jsonl(&skill, &k);
    let m: Vec<(&str, &str, &str)> = g.iter().zip(&k).flat_map(|(a, b)| [*a, *b]).collect();
    write_jsonl(&mixed, &m);
    Fixture {
        dir,
        f0: files.f0,
        f1: files.f1,
        general,
        skill,
        mixed,
    }
}

#[test]
fn genome_verify_route_eval_and_logit_identity() {
    let fx = fixture("gates");

    // G1: F1 is an append-only successor of F0.
    let g1 = json(&["genome-verify", s(&fx.f0), s(&fx.f1)]);
    assert_eq!(g1["equal"], true, "{g1}");
    assert_eq!(g1["all_entries_equal"], true);
    assert_eq!(g1["prefix_bytes_equal"], true);
    assert_eq!(g1["trunk_entries"], g1["trunk_entries_equal"]);
    assert_eq!(g1["trunk_hash_f0"], g1["trunk_hash_f1"]);
    assert!(g1["entries_f1"].as_u64() > g1["entries_f0"].as_u64());
    assert_eq!(g1["pass"], true);
    // Swapped: F0 is not a successor of F1 (shorter, fewer entries).
    let bad = cortiq(&["genome-verify", s(&fx.f1), s(&fx.f0)]);
    assert!(
        !bad.status.success(),
        "a non-successor passed genome-verify"
    );

    // G3 data: backbone accept on the general set, recall on the skill set.
    let bytes = std::fs::read(&fx.general).unwrap();
    let r = json(&[
        "route-eval",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.general),
        "--expect",
        "backbone",
        "--json",
    ]);
    let n = GENERAL_TEXTS.len();
    assert_eq!(r["n"], n);
    assert_eq!(r["accepted"], n, "{r}");
    assert_eq!(r["false_accept"], 0.0);
    let up = r["false_accept_upper95"].as_f64().unwrap();
    let want = 1.0 - 0.025f64.powf(1.0 / n as f64);
    assert!((up - want).abs() < 1e-9, "{up} vs {want}");
    assert_eq!(r["per_src"]["audit"]["n"], 2);
    assert_eq!(r["per_lang"]["en"]["n"], n.div_ceil(2));
    assert_eq!(r["route_counts"]["backbone"], n);
    assert_eq!(r["rows"].as_array().unwrap().len(), n);
    use sha2::{Digest, Sha256};
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(r["prompts_sha256"], sha);
    let r = json(&[
        "route-eval",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.skill),
        "--expect",
        SKILL_ID,
        "--json",
    ]);
    assert_eq!(r["in_scope_recall"], 1.0, "{r}");
    assert_eq!(r["gate"]["pass"], true);

    // G2: F0 without a router vs F1 --route auto, 4 greedy tokens.
    let (d0, d1) = (fx.dir.join("f0.bin"), fx.dir.join("f1.bin"));
    let j0 = json(&[
        "dump-logits",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--tokens",
        "4",
        "--out",
        s(&d0),
    ]);
    assert_eq!(j0["route_mode"], "none");
    assert_eq!(j0["positions_per_record"], 5);
    let j1 = json(&[
        "dump-logits",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--tokens",
        "4",
        "--route",
        "auto",
        "--out",
        s(&d1),
    ]);
    assert_eq!(j1["route_counts"]["backbone"], n, "{j1}");
    assert_eq!(j1["route_counts"][SKILL_ID], SKILL_TEXTS.len());
    assert_eq!(j1["route_codes"]["1"], SKILL_ID);
    let c = json(&["logits-compare", s(&d0), s(&d1)]);
    assert_eq!(c["matched"], n + SKILL_TEXTS.len(), "{c}");
    assert_eq!(c["b_backbone_records"], n);
    assert_eq!(c["b_backbone_bit_identical"], n);
    assert_eq!(c["b_backbone_max_abs_diff"], 0.0);
    assert_eq!(c["g2_pass"], true);
    // The skill-routed records ran the skill: they differ from F0.
    assert_eq!(c["per_route_b"]["1"]["n"], SKILL_TEXTS.len());
    assert_eq!(c["per_route_b"]["1"]["bit_identical"], 0);
    assert!(c["max_abs_diff"].as_f64().unwrap() > 0.01);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn skill_gate_quarantine_and_the_debug_flag() {
    let fx = fixture("gate");
    let f2 = fx.dir.join("f2.cmf");
    std::fs::copy(&fx.f1, &f2).unwrap();
    let gate = fx.dir.join("gate.json");
    std::fs::write(&gate, r#"{"status": "pending", "note": "re-measure"}"#).unwrap();
    let g = json(&[
        "skill-gate",
        s(&f2),
        "--id",
        SKILL_ID,
        "--gate",
        s(&gate),
        "--status",
        "quarantine",
    ]);
    assert_eq!(g["status_to"], "quarantine", "{g}");
    assert_eq!(g["auto_routable"], false);
    assert_eq!(
        g["calibration_stale"], false,
        "a status change keeps the calibration"
    );
    let m = cortiq_core::CmfModel::open(&f2).unwrap();
    let ev = m.header.lineage.last().unwrap();
    assert_eq!(ev.event, "skill_gated");
    assert_eq!(ev.detail["status_to"], "quarantine");
    assert_eq!(
        m.header.skills[0].gate.as_ref().unwrap()["note"],
        "re-measure"
    );
    // Header-only append: still an append-only successor of F0 and F1.
    assert_eq!(json(&["genome-verify", s(&fx.f1), s(&f2)])["pass"], true);
    assert_eq!(json(&["genome-verify", s(&fx.f0), s(&f2)])["pass"], true);

    let eval = |extra: &[&str]| {
        let mut a = vec![
            "route-eval",
            s(&f2),
            "--prompts-jsonl",
            s(&fx.skill),
            "--expect",
            SKILL_ID,
            "--json",
        ];
        a.extend_from_slice(extra);
        json(&a)
    };
    // Without the debug flag no skill can win: the measurement is vacuous
    // — gate.pass false, a non-zero exit, and still no auto-routing.
    let out = cortiq(&[
        "route-eval",
        s(&f2),
        "--prompts-jsonl",
        s(&fx.skill),
        "--expect",
        SKILL_ID,
        "--json",
    ]);
    assert!(!out.status.success(), "a vacuous route-eval exited 0");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["in_scope_recall"], 0.0, "a quarantined skill auto-routed");
    assert_eq!(v["gate"]["vacuous"], true, "{v}");
    assert_eq!(v["gate"]["pass"], false);
    assert_eq!(eval(&["--include-quarantine"])["in_scope_recall"], 1.0);
    // Bad inputs are refused, the file is untouched.
    let len = std::fs::metadata(&f2).unwrap().len();
    assert!(
        !cortiq(&[
            "skill-gate",
            s(&f2),
            "--id",
            "nope",
            "--gate",
            s(&gate),
            "--status",
            "active"
        ])
        .status
        .success()
    );
    assert!(
        !cortiq(&[
            "skill-gate",
            s(&f2),
            "--id",
            SKILL_ID,
            "--gate",
            s(&gate),
            "--status",
            "bogus"
        ])
        .status
        .success()
    );
    assert_eq!(std::fs::metadata(&f2).unwrap().len(), len);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn run_explain_and_probe_utility_route_per_request() {
    let fx = fixture("run");
    let (_, se) = ok(&[
        "run",
        s(&fx.f1),
        "-p",
        SKILL_TEXTS[0],
        "-n",
        "2",
        "--greedy",
    ]);
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    let (_, se) = ok(&[
        "run",
        s(&fx.f1),
        "-p",
        GENERAL_TEXTS[0],
        "-n",
        "2",
        "--greedy",
    ]);
    assert!(se.contains("route: backbone"), "{se}");
    let (_, se) = ok(&[
        "run",
        s(&fx.f1),
        "-p",
        SKILL_TEXTS[0],
        "-n",
        "2",
        "--greedy",
        "--skill",
        "none",
    ]);
    assert!(se.contains("route: backbone | pinned"), "{se}");
    let (_, se) = ok(&[
        "run",
        s(&fx.f1),
        "-p",
        SKILL_TEXTS[0],
        "-n",
        "2",
        "--greedy",
        "--route-dynamic",
    ]);
    assert!(se.contains("per-token switching is off"), "{se}");
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    // F0 (no router) prints no decision.
    let (_, se) = ok(&[
        "run",
        s(&fx.f0),
        "-p",
        SKILL_TEXTS[0],
        "-n",
        "2",
        "--greedy",
    ]);
    assert!(!se.contains("route:"), "{se}");

    let (so, se) = ok(&["explain", s(&fx.f1), "-p", SKILL_TEXTS[1]]);
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    assert!(so.contains("@backbone") && so.contains("← chosen"), "{so}");

    let v = json(&[
        "probe-utility",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--json",
    ]);
    let m = &v[0];
    assert_eq!(m["route_mode"], "auto", "{m}");
    assert_eq!(m["route_counts"]["backbone"], GENERAL_TEXTS.len());
    assert_eq!(m["route_counts"][SKILL_ID], SKILL_TEXTS.len());
    let rows = m["per_prompt"].as_array().unwrap();
    assert_eq!(rows[0]["route"]["target"], "backbone");
    assert_eq!(rows[1]["route"]["target"], SKILL_ID);
    let v = json(&[
        "probe-utility",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--route",
        "backbone",
        "--json",
    ]);
    assert_eq!(
        v[0]["route_counts"]["backbone"],
        GENERAL_TEXTS.len() + SKILL_TEXTS.len()
    );
    // Legacy default: no routing.
    let v = json(&[
        "probe-utility",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.mixed),
        "--max-tokens",
        "2",
        "--json",
    ]);
    assert_eq!(v[0]["route_mode"], "none");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// Write the synthetic F0/F1 pair to `$CMF_KNOWLEDGE_SYNTH_OUT` (a
/// directory) for manual runs of `serve` / `run` / the gate tools.
#[test]
#[ignore]
fn write_knowledge_synth() {
    let out = std::env::var("CMF_KNOWLEDGE_SYNTH_OUT").expect("set CMF_KNOWLEDGE_SYNTH_OUT=<dir>");
    // SAFETY: set before any pipeline of this process exists.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let files = knowledge_synth::write_knowledge_pair(
        Path::new(&out),
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    eprintln!("wrote {} and {}", files.f0.display(), files.f1.display());
}

fn cortiq_env(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cortiq"));
    c.args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT");
    for (k, v) in envs {
        c.env(k, v);
    }
    c.output().expect("spawn cortiq")
}

fn sha_file(p: &Path) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(std::fs::read(p).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// R2: a route-eval in which no skill can win is VACUOUS — a quarantined
/// skill right after a bake without `--include-quarantine` sends every
/// general prompt to the backbone, and that must not read as "0 false
/// accepts": gate.pass false, exit status non-zero. The debug flag covers
/// `stale_regate` too, and a calibration set is refused as a gate set.
#[test]
fn route_eval_refuses_vacuous_measurements_and_calibration_sets() {
    let fx = fixture("vacuous");
    let f2 = fx.dir.join("f2.cmf");
    std::fs::copy(&fx.f1, &f2).unwrap();
    cortiq_core::CmfModel::update_header_append(&f2, |h| {
        h.skills[0].status = Some("quarantine".into());
    })
    .unwrap();
    let args = |extra: &[&str]| {
        let mut a = vec![
            "route-eval",
            s(&f2),
            "--prompts-jsonl",
            s(&fx.general),
            "--expect",
            "backbone",
            "--json",
        ];
        a.extend_from_slice(extra);
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>()
    };
    let run = |extra: &[&str]| {
        let a = args(extra);
        let a: Vec<&str> = a.iter().map(String::as_str).collect();
        cortiq(&a)
    };
    let out = run(&[]);
    assert!(!out.status.success(), "a vacuous backbone gate exited 0");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["false_accept"], 0.0, "{v}");
    assert_eq!(v["gate"]["vacuous"], true);
    assert_eq!(v["gate"]["pass"], false);
    assert!(
        v["gate"]["reason"]
            .as_str()
            .unwrap()
            .contains("no routable skill class"),
        "{v}"
    );
    // With the debug flag the quarantined class is a candidate: a real
    // measurement (still no pass: 6 prompts < 500).
    let out = run(&["--include-quarantine"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["vacuous"], false, "{v}");
    assert_eq!(v["candidates"], serde_json::json!([SKILL_ID]));
    assert_eq!(v["gate"]["n_ok"], false);
    assert_eq!(v["gate"]["pass"], false);
    // stale_regate is re-gated with the same flag.
    cortiq_core::CmfModel::update_header_append(&f2, |h| {
        h.skills[0].status = Some("stale_regate".into());
    })
    .unwrap();
    assert!(!run(&[]).status.success());
    let r = json(&[
        "route-eval",
        s(&f2),
        "--prompts-jsonl",
        s(&fx.skill),
        "--expect",
        SKILL_ID,
        "--include-quarantine",
        "--json",
    ]);
    assert_eq!(r["in_scope_recall"], 1.0, "{r}");
    // The router's own calibration set is not a gate set.
    let f3 = fx.dir.join("f3.cmf");
    std::fs::copy(&fx.f1, &f3).unwrap();
    let sha = sha_file(&fx.general);
    cortiq_core::CmfModel::update_header_append(&f3, move |h| {
        let r = h.router.as_mut().unwrap();
        let mut m = r.measured.clone().unwrap_or_else(|| serde_json::json!({}));
        m["general_sha256"] = serde_json::json!(sha);
        r.measured = Some(m);
    })
    .unwrap();
    let out = cortiq(&[
        "route-eval",
        s(&f3),
        "--prompts-jsonl",
        s(&fx.general),
        "--expect",
        "backbone",
        "--json",
    ]);
    assert!(!out.status.success());
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(se.contains("calibration set"), "{se}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// R3: F1 dumped over a PART of the prompt set compares bit-identical on
/// what it has — and G2 still fails.
#[test]
fn logits_compare_refuses_a_partial_dump() {
    let fx = fixture("partial");
    let short = fx.dir.join("short.jsonl");
    write_jsonl(&short, &[(GENERAL_TEXTS[0], "en", "general")]);
    let (d0, d1) = (fx.dir.join("f0.bin"), fx.dir.join("f1.bin"));
    json(&[
        "dump-logits",
        s(&fx.f0),
        "--prompts-jsonl",
        s(&fx.general),
        "--tokens",
        "2",
        "--out",
        s(&d0),
    ]);
    json(&[
        "dump-logits",
        s(&fx.f1),
        "--prompts-jsonl",
        s(&short),
        "--tokens",
        "2",
        "--route",
        "auto",
        "--out",
        s(&d1),
    ]);
    assert!(!fx.dir.join("f1.bin.partial").exists());
    let c = json(&["logits-compare", s(&d0), s(&d1)]);
    assert_eq!(c["b_backbone_bit_identical"], 1, "{c}");
    assert_eq!(c["only_in_a"], GENERAL_TEXTS.len() - 1);
    assert_eq!(c["g2_pass"], false, "{c}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// NF-1: a genome never takes a full rewrite from a CLI tool (`skill add`
/// used to rewrite F1 in place, `--sparse` adding a mask catalog the
/// backbone then ran under), and `dump-logits` refuses a file whose mask
/// catalog run/serve would apply.
#[test]
fn genome_refuses_full_rewrites_and_dumps_refuse_masked_files() {
    let fx = fixture("rewrite");
    let before = sha_file(&fx.f1);
    let out = cortiq(&[
        "skill",
        "add",
        s(&fx.f1),
        "--from",
        "/nonexistent-donor",
        "--id",
        "x",
    ]);
    assert!(!out.status.success());
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(se.contains("frozen genome"), "{se}");
    assert_eq!(sha_file(&fx.f1), before, "the genome file was touched");

    // A (non-genome) file carrying a mask catalog.
    let plain = fx.dir.join("plain.cmf");
    embryo_synth::write_synth_genome(&plain, &embryo_synth::SynthGeom::tiny_gdn_bounded());
    let m = cortiq_core::CmfModel::open(&plain).unwrap();
    let a = m.header.arch.clone();
    let l = a.num_layers;
    let catalog = cortiq_core::MaskCatalog {
        masks: vec![cortiq_core::TaskMask {
            task_id: 0,
            name: "general".into(),
            description: None,
            sparsity: 0.3,
            quality: None,
            ffn_masks: vec![vec![0x55u8; a.ffn_mask_bytes()]; l],
            head_masks: vec![vec![0xffu8; a.head_mask_bytes()]; l],
            layer_gates: vec![true; l],
            expert_masks: Vec::new(),
            parent: None,
            has_hot_pack: false,
            priority: cortiq_core::MaskPriority::Fallback,
        }],
        default_task: "general".into(),
    };
    let specs: Vec<cortiq_core::TensorSpec> = m
        .tensors
        .iter()
        .map(|e| cortiq_core::TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: m.entry_bytes(e).to_vec(),
        })
        .collect();
    let masked = fx.dir.join("masked.cmf");
    cortiq_core::CmfModel::write(&masked, &m.header, &specs, Some(&catalog), m.vocab.as_deref())
        .unwrap();
    let out = cortiq(&[
        "dump-logits",
        s(&masked),
        "--prompts-jsonl",
        s(&fx.general),
        "--tokens",
        "1",
        "--out",
        s(&fx.dir.join("m.bin")),
    ]);
    assert!(!out.status.success());
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(se.contains("mask catalog"), "{se}");
    assert!(!fx.dir.join("m.bin").exists());
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// PHI-1: `run` generates a cmf-im-v1 skill's answer from the rendered
/// user turn (the frame it was trained and gated under — the one
/// dump-logits / probe-utility measure); the backbone keeps the raw
/// prompt (G2 identity with F0). The first id of the rendered prompt is
/// the synthetic `<|im_start|>` (256).
#[test]
fn run_renders_the_skill_contract_and_keeps_the_backbone_raw() {
    let fx = fixture("contract");
    let dump = [("CMF_PROMPT_DUMP", "1")];
    let out = cortiq_env(
        &["run", s(&fx.f1), "-p", SKILL_TEXTS[0], "-n", "1", "--greedy"],
        &dump,
    );
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{se}");
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    assert!(se.contains("prompt rendered as cmf-im-v1"), "{se}");
    assert!(se.contains("head 256:"), "{se}");
    let out = cortiq_env(
        &["run", s(&fx.f1), "-p", GENERAL_TEXTS[0], "-n", "1", "--greedy"],
        &dump,
    );
    let se = String::from_utf8_lossy(&out.stderr);
    assert!(se.contains("route: backbone"), "{se}");
    assert!(!se.contains("prompt rendered"), "{se}");
    assert!(!se.contains("head 256:"), "{se}");
    // explain shows the skill's first token on the rendered turn too.
    let (_, se) = ok(&["explain", s(&fx.f1), "-p", SKILL_TEXTS[1]]);
    assert!(se.contains("prompt rendered as cmf-im-v1"), "{se}");
    let _ = std::fs::remove_dir_all(&fx.dir);
}
