//! Local build gates on BANKING77 with the release encoder (spec §3.8, §6.2).
//! Every test is `#[ignore]`: it needs the exported encoder and the benchmark
//! splits, and should run optimised, one heavy process at a time:
//!
//! ```text
//! CMF_GPU=0 CORTIQ_DECISION_ENCODER_DIR=$ENC \
//!   [CORTIQ_DECISION_ENCODER_CMF=enc.cmf] [CORTIQ_DECISION_BUILD_OUT=<dir>] \
//!   [CORTIQ_DECISION_BANKING77_DIR=…/decision-v2-20260926/splits/banking77] \
//!   [CORTIQ_DECISION_MAX_RECIPE=…/max-recipe/cv.json] [CORTIQ_DECISION_THREADS=4] \
//!   [CMFPUBLIC=<checkout with artifacts/ and reports/>] \
//!   cargo test --release -p cortiq-decision --test build_real -- --ignored --nocapture --test-threads=1
//! ```
//!
//! * `banking77_build_gates` — `train` on train.jsonl with calibration.jsonl,
//!   dev.jsonl and question.json, `K = 16` (the reproduction build of spec §3.8).
//!   Gates (work package WP4, from spec §6.2): dev `1396 ± 2 / 1498`, `τ = 0.8`,
//!   `T` within 1 % of `0.0245571`, `θ` within 0.005 of `0.788527`, the
//!   Clopper–Pearson bound of the chosen odd-half threshold `≥ 0.95`; the
//!   post-write self-check passes (bit-exact). Reported besides: calibration
//!   (v3: 1388/1498), odd half accepted/correct (v3: 649/635), timings.
//! * `banking77_train_dev_union` — the published recipe of spec §3.9: topologies
//!   on train ∪ dev with `K` from `cv.json` (`chosen_K`), calibration for the gate
//!   only. No gate is asserted beyond a passing build and self-check; the numbers
//!   are reported.
//!
//! No test split is opened. Each test prints `BUILD_GATE_JSON <name> <json>` and
//! writes `<name>.json` into `CORTIQ_DECISION_BUILD_OUT` when set; the built
//! files stay there too (otherwise they go to a temp dir that is removed).

use cortiq_decision::build::{self, BuildReport, TrainOptions};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const ENCODER_CMF_ENV: &str = "CORTIQ_DECISION_ENCODER_CMF";
const ENCODER_DIR_ENV: &str = "CORTIQ_DECISION_ENCODER_DIR";
const OUT_ENV: &str = "CORTIQ_DECISION_BUILD_OUT";
const BANKING77_ENV: &str = "CORTIQ_DECISION_BANKING77_DIR";
const MAX_RECIPE_ENV: &str = "CORTIQ_DECISION_MAX_RECIPE";
const THREADS_ENV: &str = "CORTIQ_DECISION_THREADS";

/// The checkout that holds the local artifacts and reports; the defaults of
/// the two paths below are relative to it.
const CMFPUBLIC_ENV: &str = "CMFPUBLIC";
const BANKING77_DEFAULT: &str = "artifacts/decision-v2-20260926/splits/banking77";
const MAX_RECIPE_DEFAULT: &str = "reports/decision-v4-20260926/max-recipe/cv.json";

// v3 reference (reports/decision-v3-20260926/evaluate/RESULT_RU.md, spec §6.2).
const V3_DEV: i64 = 1396;
const V3_DEV_N: u64 = 1498;
const V3_T: f64 = 0.0245571;
const V3_THETA: f64 = 0.788527;
const V3_TAU: f64 = 0.8;
const V3_CAL: u64 = 1388;
const V3_ODD: (u64, u64) = (649, 635);

/// The path in `name`, else `default` under `$CMFPUBLIC` when that is set.
fn env_path(name: &str, default: Option<&str>) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).or_else(|| {
        let root = std::env::var_os(CMFPUBLIC_ENV)?;
        default.map(|d| PathBuf::from(root).join(d))
    })
}

fn threads() -> usize {
    std::env::var(THREADS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

fn sha256_file(p: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(p).unwrap()))
}

/// The output directory (kept) or a temp dir (removed on drop).
struct Out {
    dir: PathBuf,
    _tmp: Option<tempfile::TempDir>,
}

fn out_dir(name: &str) -> Out {
    match env_path(OUT_ENV, None) {
        Some(d) => {
            let dir = d.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            Out { dir, _tmp: None }
        }
        None => {
            let t = tempfile::tempdir().unwrap();
            Out {
                dir: t.path().to_path_buf(),
                _tmp: Some(t),
            }
        }
    }
}

/// The release encoder as a decision file: `CORTIQ_DECISION_ENCODER_CMF`, or
/// `init` from `CORTIQ_DECISION_ENCODER_DIR` into `dir`.
fn encoder_cmf(dir: &Path) -> PathBuf {
    if let Some(p) = env_path(ENCODER_CMF_ENV, None) {
        return p;
    }
    let export = env_path(ENCODER_DIR_ENV, None).unwrap_or_else(|| {
        panic!("set {ENCODER_CMF_ENV} or {ENCODER_DIR_ENV} (see the module docs)")
    });
    let out = dir.join("encoder.cmf");
    if !out.exists() {
        let t = std::time::Instant::now();
        let r = build::init_encoder(&export, &out, None).expect("init the release encoder");
        eprintln!(
            "init: {} ({} bytes, sha256 {}) in {:.1} s",
            out.display(),
            r.bytes,
            r.sha256,
            t.elapsed().as_secs_f64()
        );
    }
    out
}

fn banking77() -> PathBuf {
    let d = env_path(BANKING77_ENV, Some(BANKING77_DEFAULT))
        .unwrap_or_else(|| panic!("set {BANKING77_ENV} or {CMFPUBLIC_ENV}"));
    // The splits the v3 skill was certified on.
    let splits: Value =
        serde_json::from_slice(&std::fs::read(d.join("splits.json")).unwrap()).unwrap();
    for s in ["train", "dev", "calibration"] {
        assert_eq!(
            sha256_file(&d.join(format!("{s}.jsonl"))),
            splits["splits"][s]["sha256"].as_str().unwrap(),
            "{s}.jsonl changed"
        );
    }
    d
}

fn report(name: &str, dir: &Path, v: &Value) {
    println!("BUILD_GATE_JSON {name} {v}");
    if env_path(OUT_ENV, None).is_some() {
        std::fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_string_pretty(v).unwrap(),
        )
        .unwrap();
    }
}

fn chosen_lb(r: &BuildReport) -> Option<f64> {
    let g = &r.manifest.gate;
    g.evidence
        .odd
        .grid
        .iter()
        .find(|x| g.certified && x.t as f32 as f64 == g.tau)
        .map(|x| x.lb)
}

#[test]
#[ignore = "needs the release encoder and the BANKING77 splits (see the module docs)"]
fn banking77_build_gates() {
    let out = out_dir("banking77-train");
    let enc = encoder_cmf(&out.dir);
    let b = banking77();
    let mut o = TrainOptions::new("banking77", vec![b.join("train.jsonl")]);
    o.calibration = Some(b.join("calibration.jsonl"));
    o.dev = Some(b.join("dev.jsonl"));
    o.question = Some(b.join("question.json"));
    o.threads = threads();
    let file = out.dir.join("banking77.cmf");
    if file.exists() {
        panic!(
            "{} exists; remove it or choose another CORTIQ_DECISION_BUILD_OUT",
            file.display()
        );
    }
    let r = build::train(&enc, &o, &file).expect("build BANKING77");
    let g = &r.manifest.gate;
    let dev = r.dev.expect("dev evaluated");
    let lb = chosen_lb(&r);
    let t_rel = (g.temperature - V3_T).abs() / V3_T;
    let theta_abs = (g.novelty_theta - V3_THETA).abs();
    let checks = json!({
        "dev_within_2": (dev.correct as i64 - V3_DEV).abs() <= 2 && dev.n as u64 == V3_DEV_N,
        "tau_equal": g.tau == V3_TAU as f32 as f64,
        "t_within_1pct": t_rel <= 0.01,
        "theta_within_0005": theta_abs <= 0.005,
        "odd_lb_ge_095": lb.is_some_and(|x| x >= 0.95),
        "self_check": r.self_check.calibration_rows == 1498 && r.self_check.texts_reencoded == 64,
    });
    let pass = checks
        .as_object()
        .unwrap()
        .values()
        .all(|v| v == &json!(true));
    let v = json!({
        "pass": pass,
        "checks": checks,
        "dev": {"n": dev.n, "correct": dev.correct, "v3": V3_DEV, "accepted": dev.accepted, "accepted_correct": dev.accepted_correct},
        "calibration": {"n": g.evidence.calibration.n, "correct": r.calibration_correct, "v3": V3_CAL},
        "gate": {
            "temperature": g.temperature, "t_rel_diff": t_rel, "v3_temperature": V3_T,
            "novelty_theta": g.novelty_theta, "theta_abs_diff": theta_abs, "v3_theta": V3_THETA,
            "tau": g.tau, "certified": g.certified, "lb": lb,
            "odd": {"n": g.evidence.odd.n, "accepted": g.evidence.odd.accepted, "correct": g.evidence.odd.correct, "v3": [V3_ODD.0, V3_ODD.1]},
            "nll_even": g.evidence.even.nll,
        },
        "k": r.manifest.tasks.iter().map(|t| t.k).collect::<Vec<_>>(),
        "build": r.to_json(),
    });
    report("banking77_build_gates", &out.dir, &v);
    assert!(pass, "BANKING77 build gates failed: {}", v["checks"]);
}

#[test]
#[ignore = "needs the release encoder, the BANKING77 splits and max-recipe/cv.json (see the module docs)"]
fn banking77_train_dev_union() {
    let out = out_dir("banking77-train-dev");
    let enc = encoder_cmf(&out.dir);
    let b = banking77();
    let cv_path = env_path(MAX_RECIPE_ENV, Some(MAX_RECIPE_DEFAULT))
        .unwrap_or_else(|| panic!("set {MAX_RECIPE_ENV} or {CMFPUBLIC_ENV}"));
    let (k, k_source) = match std::fs::read(&cv_path) {
        Ok(bytes) => {
            let cv: Value = serde_json::from_slice(&bytes).unwrap();
            let k = cv["chosen_K"]["banking77"]
                .as_u64()
                .expect("chosen_K.banking77") as usize;
            let sha = Sha256::digest(&bytes);
            (
                k,
                Some(format!("max-recipe/cv.json chosen_K (sha256 {sha:x})")),
            )
        }
        Err(_) => (16, None),
    };
    let mut o = TrainOptions::new(
        "banking77",
        vec![b.join("train.jsonl"), b.join("dev.jsonl")],
    );
    o.calibration = Some(b.join("calibration.jsonl"));
    o.question = Some(b.join("question.json"));
    o.k = k;
    o.k_source = k_source;
    o.threads = threads();
    let file = out.dir.join("banking77-train-dev.cmf");
    if file.exists() {
        panic!(
            "{} exists; remove it or choose another CORTIQ_DECISION_BUILD_OUT",
            file.display()
        );
    }
    let r = build::train(&enc, &o, &file).expect("build BANKING77 on train ∪ dev");
    let g = &r.manifest.gate;
    assert_eq!(r.manifest.data.train.n, 6996 + 1498);
    assert_eq!(r.manifest.data.train.parts.len(), 2);
    assert_eq!(r.manifest.recipe.k_max as usize, k);
    assert_eq!(r.self_check.calibration_rows, 1498);
    let v = json!({
        "K": k,
        "calibration": {"n": g.evidence.calibration.n, "correct": r.calibration_correct},
        "gate": {
            "temperature": g.temperature, "novelty_theta": g.novelty_theta, "tau": g.tau,
            "certified": g.certified, "lb": chosen_lb(&r),
            "odd": {"n": g.evidence.odd.n, "accepted": g.evidence.odd.accepted, "correct": g.evidence.odd.correct},
        },
        "build": r.to_json(),
    });
    report("banking77_train_dev_union", &out.dir, &v);
}
