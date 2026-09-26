//! Local encoder gates E1–E5 (spec §1.6) on the release encoder. Every test is
//! `#[ignore]`: it needs the exported encoder, the stored v3 features and the
//! benchmark splits, and it should run optimised, one heavy process at a time:
//!
//! ```text
//! python3 tools/decision_export_encoder.py --onnx …/encoder.onnx \
//!     --tokenizer-dir …/encoder_tokenizer --out $ENC
//! python3 tools/decision_encoder_parity.py hf-ids --tokenizer-dir …/encoder_tokenizer --out $IDS
//! python3 tools/decision_encoder_parity.py unicode-probe --out $PROBE
//! CMF_GPU=0 CORTIQ_DECISION_ENCODER_DIR=$ENC CORTIQ_DECISION_HF_IDS_DIR=$IDS \
//!   CORTIQ_DECISION_UNICODE_PROBE=$PROBE \
//!   CORTIQ_DECISION_V3_DIR=/Users/oleg/dev/cmfpublic/artifacts/decision-v3-20260926 \
//!   [CORTIQ_DECISION_ENCODER_OUT=<dir>] [CORTIQ_DECISION_THREADS=4] \
//!   cargo test --release -p cortiq-decision --test encoder_real -- --ignored --nocapture --test-threads=1
//! ```
//!
//! * **E1** token ids equal HF `tokenizers` 0.22.2 on 100 % of the texts of
//!   train/dev/calibration/test of the three datasets, CLINC150 oos/latency and
//!   MASSIVE latency; stress-set differences are listed, not blocking;
//! * **E2** φ_P against `features-product/{ds}/{dev,calibration,test}.npy`
//!   (ONNX Runtime 1.19.2, batch 64): max |Δ| ≤ 1e-6, min cos ≥ 0.999999;
//! * **E3** φ_H against `hash-features/{ds}/{train,dev,calibration}` (the Python
//!   port): the non-zero index sets are equal on every row but the listed ones,
//!   max |Δ| ≤ 1e-7;
//! * **E4** winners on the v3 topologies (`{ds}-product/cortiq.cmf`) of
//!   `x_v4 = [φ_P ; 0.5·φ_H]` against the v3 signal
//!   `[l2f32(stored φ_P) ; 0.5·φ_H]`: flips ≤ ceil(0.2 %·n) on dev and test;
//! * **E5** φ_P bit-identical between two runs, between thread counts and
//!   between two processes on 256 texts; the differences under
//!   `VECLIB_MAXIMUM_THREADS=1` and under `CMF_ACCEL=0` (the non-Accelerate
//!   GEMM) are reported;
//! * encoder p50/p95/p99 at one thread (`CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1`,
//!   a child process), per stage, on BANKING77 dev;
//! * every Unicode scalar through the normalizer and the pre-tokenizer against
//!   HF (`unicode-probe`);
//! * the real encoder on a long text never touches a GPU backend.
//!
//! Every test-split, oos and latency file is logged to `test-access.log` before
//! it is opened. Each test prints `ENCODER_GATE_JSON <name> <json>` and writes
//! `<name>.json` into `CORTIQ_DECISION_ENCODER_OUT` when set.
mod common;

use common::{
    DATASETS, Row, l2f32_numpy, log_test_access, read_hash_npz_dense, read_jsonl_rows, read_npy,
    read_v3_skill, sha256_file, v3_dir,
};
use cortiq_decision::bert::{Encoder, EncoderExport};
use cortiq_decision::packed::Packed;
use cortiq_decision::resonance::{ErrStats, TaskView, decide};
use cortiq_decision::signal::{self, SignalEncoder};
use cortiq_decision::wordpiece::{self, WordPiece};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;

const ENCODER_DIR_ENV: &str = "CORTIQ_DECISION_ENCODER_DIR";
const HF_IDS_ENV: &str = "CORTIQ_DECISION_HF_IDS_DIR";
const PROBE_ENV: &str = "CORTIQ_DECISION_UNICODE_PROBE";
const OUT_ENV: &str = "CORTIQ_DECISION_ENCODER_OUT";
const THREADS_ENV: &str = "CORTIQ_DECISION_THREADS";
const CHILD_OUT_ENV: &str = "CORTIQ_DECISION_ENCODER_CHILD_OUT";

const E2_MAX_ABS: f64 = 1e-6;
const E2_MIN_COS: f64 = 0.999999;
const E3_MAX_ABS: f64 = 1e-7;
const E4_FLIP_RATE: f64 = 0.002;

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} (see the module docs)"))
}

fn threads() -> usize {
    std::env::var(THREADS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

fn export() -> EncoderExport {
    EncoderExport::read(env_path(ENCODER_DIR_ENV)).expect("encoder export")
}

fn encoder() -> Encoder {
    export().encoder().expect("encoder")
}

fn report(name: &str, v: &Value) {
    println!("ENCODER_GATE_JSON {name} {v}");
    if let Some(dir) = std::env::var_os(OUT_ENV) {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_string_pretty(v).unwrap(),
        )
        .unwrap();
    }
}

fn sealed(split: &str) -> bool {
    matches!(split, "test" | "oos" | "latency")
}

/// The split file the v3 features were made from (`representation.json`),
/// sha256 checked; sealed splits are logged before they are read.
fn split_rows(v3: &Path, ds: &str, split: &str, purpose: &str) -> (PathBuf, Vec<Row>) {
    let rep: Value = serde_json::from_slice(
        &std::fs::read(v3.join(format!("features-product/{ds}/representation.json"))).unwrap(),
    )
    .unwrap();
    let path = PathBuf::from(rep["source_dir"].as_str().unwrap()).join(format!("{split}.jsonl"));
    if sealed(split) {
        log_test_access(&path, purpose);
    }
    assert_eq!(
        sha256_file(&path),
        rep["source_files_sha256"][split].as_str().unwrap(),
        "{} changed since the features were made",
        path.display()
    );
    let rows = read_jsonl_rows(&path);
    assert_eq!(
        rows.len() as u64,
        rep["splits"][split]["n"].as_u64().unwrap(),
        "{}",
        path.display()
    );
    (path, rows)
}

/// Run `f` over `0..n` on `threads` scoped threads, results in index order.
fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        out.push((i, f(i)));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for p in parts {
        for (i, v) in p {
            slots[i] = Some(v);
        }
    }
    slots.into_iter().map(Option::unwrap).collect()
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

// ------------------------------------------------------------------ E1

#[test]
#[ignore]
fn e1_token_ids_equal_hf() {
    let wp: WordPiece = export().wordpiece().unwrap();
    let dir = env_path(HF_IDS_ENV);
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    assert_eq!(index["tokenizers"], "0.22.2");
    let mut per_file = Vec::new();
    let (mut total, mut equal) = (0usize, 0usize);
    let mut failures = Vec::new();
    let mut stress_diffs = Vec::new();
    for f in index["files"].as_array().unwrap() {
        let name = f["file"].as_str().unwrap();
        let g: Value = serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap();
        let hf = g["ids"].as_array().unwrap();
        if f["dataset"] == "stress" {
            for (t, want) in g["texts"].as_array().unwrap().iter().zip(hf) {
                let t = t.as_str().unwrap();
                let (got, want) = (wp.encode(t), ids_of(want));
                if got != want {
                    stress_diffs.push(
                        json!({"text": t, "native": wp.tokens(&got), "hf": wp.tokens(&want)}),
                    );
                }
            }
            per_file.push(json!({"file": name, "n": hf.len(), "differ": stress_diffs.len()}));
            continue;
        }
        let src = PathBuf::from(g["source"].as_str().unwrap());
        if f["sealed"].as_bool().unwrap() {
            log_test_access(
                &src,
                "decision-v4 WP3 gate E1: native WordPiece ids vs HF (tests/encoder_real.rs)",
            );
        }
        assert_eq!(
            sha256_file(&src),
            g["sha256"].as_str().unwrap(),
            "{}",
            src.display()
        );
        let rows = read_jsonl_rows(&src);
        assert_eq!(rows.len(), hf.len(), "{name}");
        let mut eq = 0usize;
        for (r, want) in rows.iter().zip(hf) {
            let want = ids_of(want);
            let got = wp.encode(&r.text);
            if got == want {
                eq += 1;
            } else if failures.len() < 20 {
                failures.push(format!(
                    "{name}: {:?}: native {:?} vs HF {:?}",
                    r.text,
                    wp.tokens(&got),
                    wp.tokens(&want)
                ));
            }
        }
        total += rows.len();
        equal += eq;
        per_file.push(json!({"file": name, "n": rows.len(), "equal": eq}));
    }
    let out = json!({
        "gate": "E1", "texts": total, "equal": equal, "pass": equal == total,
        "files": per_file, "stress_differences": stress_diffs,
    });
    report("e1", &out);
    assert!(failures.is_empty(), "E1 failed:\n{}", failures.join("\n"));
    assert_eq!(equal, total);
}

// ------------------------------------------------------------------ E2, E3, E4

/// φ_H from the stored Python port against the native contract.
fn e3_split(
    v3: &Path,
    ds: &str,
    split: &str,
    rows: &[Row],
    native: &[Vec<f32>],
) -> (Value, Vec<f32>) {
    let hdir = v3.join("hash-features").join(ds);
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(hdir.join(format!("manifest-{split}.json"))).unwrap(),
    )
    .unwrap();
    let npz = hdir.join(format!("{split}.npz"));
    assert_eq!(sha256_file(&npz), manifest["npz_sha256"].as_str().unwrap());
    let (n, stored) = read_hash_npz_dense(&npz);
    assert_eq!(n, rows.len());
    let dim = stored.len() / n;
    assert_eq!(dim, signal::PHI_H_DIM);
    let (mut max_abs, mut index_diff_rows) = (0.0f64, Vec::new());
    for i in 0..n {
        let s = &stored[i * dim..(i + 1) * dim];
        let v = &native[i];
        let mut sets_equal = true;
        for (a, b) in v.iter().zip(s) {
            if (*a != 0.0) != (*b != 0.0) {
                sets_equal = false;
            }
            max_abs = max_abs.max((*a as f64 - *b as f64).abs());
        }
        if !sets_equal {
            index_diff_rows.push(i);
        }
    }
    let pass = index_diff_rows.is_empty() && max_abs <= E3_MAX_ABS;
    (
        json!({"split": split, "n": n, "index_set_differs": index_diff_rows.len(),
               "rows": index_diff_rows.iter().take(50).collect::<Vec<_>>(), "max_abs": max_abs, "pass": pass}),
        stored,
    )
}

/// φ_P against the stored ORT features.
fn e2_split(v3: &Path, ds: &str, split: &str, native: &[Vec<f32>]) -> (Value, Vec<f32>) {
    let npy = v3
        .join("features-product")
        .join(ds)
        .join(format!("{split}.npy"));
    if sealed(split) {
        log_test_access(
            &npy,
            "decision-v4 WP3 gate E2: stored ORT φ_P vs native encoder",
        );
    }
    let a = read_npy(&npy);
    assert_eq!(a.shape, vec![native.len(), 384], "{}", npy.display());
    let stored = a.f32();
    let (mut max_abs, mut min_cos, mut bit_equal) = (0.0f64, f64::INFINITY, 0usize);
    let (mut worst_row, mut sum_abs) = (0usize, 0.0f64);
    for (i, v) in native.iter().enumerate() {
        let s = &stored[i * 384..(i + 1) * 384];
        let (mut dot, mut nv, mut ns, mut row_max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, y) in v.iter().zip(s) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            nv += x * x;
            ns += y * y;
            row_max = row_max.max((x - y).abs());
            sum_abs += (x - y).abs();
        }
        if v.iter().zip(s).all(|(x, y)| x.to_bits() == y.to_bits()) {
            bit_equal += 1;
        }
        if row_max > max_abs {
            max_abs = row_max;
            worst_row = i;
        }
        min_cos = min_cos.min(dot / (nv.sqrt() * ns.sqrt()));
    }
    let pass = max_abs <= E2_MAX_ABS && min_cos >= E2_MIN_COS;
    (
        json!({"split": split, "n": native.len(), "max_abs": max_abs, "worst_row": worst_row,
               "mean_abs": sum_abs / (native.len() * 384) as f64, "min_cos": min_cos,
               "bit_equal_rows": bit_equal, "pass": pass}),
        stored,
    )
}

#[test]
#[ignore]
fn e2_e3_e4_features_and_decisions() {
    let v3 = v3_dir().expect("set CORTIQ_DECISION_V3_DIR");
    let se = SignalEncoder::new(encoder());
    let th = threads();
    let mut out = serde_json::Map::new();
    let mut all_pass = true;
    for ds in DATASETS {
        let skill = read_v3_skill(&v3, ds);
        let active = skill.active();
        let views: Vec<TaskView<'_>> = active
            .iter()
            .map(|t| TaskView::new(&t.mean, &t.basis).unwrap())
            .collect();
        let stats: Vec<ErrStats> = active
            .iter()
            .map(|t| ErrStats {
                err_mean: t.err_mean,
                err_std: t.err_std,
            })
            .collect();
        let packed = Packed::new(&views).unwrap();
        let mut dsj = serde_json::Map::new();
        // E3 on train (φ_H only).
        let (_, train) = split_rows(&v3, ds, "train", "");
        let t0 = Instant::now();
        let train_h = par_map(train.len(), th, |i| signal::phi_h(&train[i].text));
        let (e3_train, _) = e3_split(&v3, ds, "train", &train, &train_h);
        all_pass &= e3_train["pass"].as_bool().unwrap();
        dsj.insert("e3_train".into(), e3_train);
        let hash_secs = t0.elapsed().as_secs_f64();
        for split in ["dev", "calibration", "test"] {
            let (_, rows) = split_rows(
                &v3,
                ds,
                split,
                "decision-v4 WP3 gates E2/E4: native encoder on the test split",
            );
            let t0 = Instant::now();
            let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
            let feats = se.features_batch(&texts, th);
            let enc_secs = t0.elapsed().as_secs_f64();
            let phi_p: Vec<Vec<f32>> = feats.iter().map(|f| f.phi_p.clone()).collect();
            let phi_h: Vec<Vec<f32>> = feats.iter().map(|f| f.phi_h.clone()).collect();
            let (e2, stored_p) = e2_split(&v3, ds, split, &phi_p);
            all_pass &= e2["pass"].as_bool().unwrap();
            let mut sj = serde_json::Map::new();
            sj.insert("encode_seconds".into(), json!(enc_secs));
            sj.insert("e2".into(), e2);
            // The v3 φ_H: the stored Python port where it exists (E3), else the
            // native contract (equal to it wherever E3 compares them).
            let ref_h: Vec<Vec<f32>> = if split == "test" {
                phi_h.clone()
            } else {
                let (e3, stored_h) = e3_split(&v3, ds, split, &rows, &phi_h);
                all_pass &= e3["pass"].as_bool().unwrap();
                sj.insert("e3".into(), e3);
                stored_h
                    .chunks_exact(signal::PHI_H_DIM)
                    .map(<[f32]>::to_vec)
                    .collect()
            };
            if split != "calibration" {
                // E4: winners on the v3 topologies.
                let decided = par_map(rows.len(), th, |i| {
                    let mut x3 = l2f32_numpy(&stored_p[i * 384..(i + 1) * 384]);
                    x3.extend(ref_h[i].iter().map(|h| 0.5 * h));
                    let x4 = feats[i].signal();
                    let w = |x: &[f32]| {
                        let e = packed.errors(x).unwrap();
                        decide(&e, &stats, skill.temperature)
                            .unwrap()
                            .winner
                            .unwrap()
                    };
                    (w(&x3), w(&x4))
                });
                let n = rows.len();
                let flips = decided.iter().filter(|(a, b)| a != b).count();
                let correct = |k: usize| {
                    decided
                        .iter()
                        .zip(&rows)
                        .filter(|(d, r)| active[if k == 0 { d.0 } else { d.1 }].label == r.label)
                        .count()
                };
                let limit = (E4_FLIP_RATE * n as f64).ceil() as usize;
                let pass = flips <= limit;
                all_pass &= pass;
                sj.insert(
                    "e4".into(),
                    json!({"n": n, "flips": flips, "limit": limit, "correct_v3_signal": correct(0),
                           "correct_v4_signal": correct(1), "pass": pass}),
                );
            }
            dsj.insert(split.into(), Value::Object(sj));
        }
        dsj.insert("train_hash_seconds".into(), json!(hash_secs));
        eprintln!("{ds}: {}", Value::Object(dsj.clone()));
        out.insert(ds.into(), Value::Object(dsj));
    }
    out.insert("pass".into(), json!(all_pass));
    out.insert("threads".into(), json!(th));
    let out = Value::Object(out);
    report("e2_e3_e4", &out);
    assert!(all_pass, "E2/E3/E4 failed: {out}");
}

// ------------------------------------------------------------------ E5

/// 256 non-test texts: the first dev rows of each dataset.
fn e5_texts() -> Vec<String> {
    let v3 = v3_dir().expect("set CORTIQ_DECISION_V3_DIR");
    let mut texts = Vec::new();
    for (ds, k) in [("banking77", 86), ("clinc150", 85), ("massive", 85)] {
        let (_, rows) = split_rows(&v3, ds, "dev", "");
        texts.extend(rows.into_iter().take(k).map(|r| r.text));
    }
    assert_eq!(texts.len(), 256);
    texts
}

fn phi_bits(enc: &Encoder, texts: &[String]) -> Vec<u32> {
    texts
        .iter()
        .flat_map(|t| enc.encode(t))
        .map(f32::to_bits)
        .collect()
}

/// Child process of `e5_determinism` and `encoder_speed_one_thread`: does its
/// work only when `CORTIQ_DECISION_ENCODER_CHILD_OUT` is set.
#[test]
#[ignore]
fn encoder_child() {
    let Some(out) = std::env::var_os(CHILD_OUT_ENV) else {
        return;
    };
    let out = PathBuf::from(out);
    let enc = encoder();
    if std::env::var("CORTIQ_DECISION_CHILD_MODE").as_deref() == Ok("speed") {
        let v3 = v3_dir().expect("set CORTIQ_DECISION_V3_DIR");
        let (_, rows) = split_rows(&v3, "banking77", "dev", "");
        let se = SignalEncoder::new(enc);
        for r in rows.iter().take(50) {
            let _ = se.features_timed(&r.text);
        }
        let mut t = Vec::with_capacity(rows.len());
        for r in &rows {
            let t0 = Instant::now();
            let (_, tm) = se.features_timed(&r.text);
            let total = t0.elapsed();
            t.push([
                tm.tokenize.as_secs_f64(),
                tm.encode.as_secs_f64(),
                tm.hash.as_secs_f64(),
                total.as_secs_f64(),
                tm.tokens as f64,
            ]);
        }
        let pct = |k: usize, p: f64| {
            let mut v: Vec<f64> = t.iter().map(|r| r[k]).collect();
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * p).round() as usize] * 1e3
        };
        let mut j = serde_json::Map::new();
        for (k, name) in ["tokenize_ms", "encode_ms", "hash_ms", "total_ms", "tokens"]
            .iter()
            .enumerate()
        {
            let scale = if k == 4 { 1e-3 } else { 1.0 };
            j.insert(
                (*name).into(),
                json!({"p50": pct(k, 0.5) * scale, "p95": pct(k, 0.95) * scale, "p99": pct(k, 0.99) * scale}),
            );
        }
        j.insert("n".into(), json!(rows.len()));
        std::fs::write(&out, Value::Object(j).to_string()).unwrap();
        return;
    }
    let bits = phi_bits(&enc, &e5_texts());
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    std::fs::write(&out, bytes).unwrap();
}

fn run_child(tag: &str, envs: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cortiq-decision-encoder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(format!("{tag}.out"));
    let _ = std::fs::remove_file(&out);
    let exe = std::env::current_exe().unwrap();
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "--ignored",
        "--exact",
        "encoder_child",
        "--nocapture",
        "--test-threads=1",
    ])
    .env(CHILD_OUT_ENV, &out);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let st = cmd.status().expect("spawn child");
    assert!(st.success(), "child {tag} failed");
    assert!(out.exists(), "child {tag} wrote nothing");
    out
}

fn compare_bits(a: &[u32], b: &[u32], dim: usize) -> Value {
    let rows = a.len() / dim;
    let mut same = 0usize;
    let mut max_abs = 0.0f64;
    for i in 0..rows {
        let (x, y) = (&a[i * dim..(i + 1) * dim], &b[i * dim..(i + 1) * dim]);
        if x == y {
            same += 1;
        }
        for (p, q) in x.iter().zip(y) {
            max_abs = max_abs.max((f32::from_bits(*p) as f64 - f32::from_bits(*q) as f64).abs());
        }
    }
    json!({"rows": rows, "bit_identical_rows": same, "max_abs": max_abs})
}

#[test]
#[ignore]
fn e5_determinism() {
    let enc = encoder();
    let texts = e5_texts();
    let dim = enc.dim();
    let run1 = phi_bits(&enc, &texts);
    let run2 = phi_bits(&enc, &texts);
    let se = SignalEncoder::new(enc);
    let batch: Vec<u32> = se
        .features_batch(&texts, 4)
        .iter()
        .flat_map(|f| f.phi_p.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        .collect();
    let read = |p: PathBuf| -> Vec<u32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let child = read(run_child("default", &[]));
    let child_veclib1 = read(run_child("veclib1", &[("VECLIB_MAXIMUM_THREADS", "1")]));
    let child_noaccel = read(run_child("noaccel", &[("CMF_ACCEL", "0")]));
    let runs = compare_bits(&run1, &run2, dim);
    let threads = compare_bits(&run1, &batch, dim);
    let procs = compare_bits(&run1, &child, dim);
    let pass = run1 == run2 && run1 == batch && run1 == child;
    let out = json!({
        "gate": "E5", "texts": texts.len(), "pass": pass,
        "two_runs": runs, "threads_1_vs_4": threads, "two_processes": procs,
        "veclib_max_threads_1_vs_default": compare_bits(&run1, &child_veclib1, dim),
        "cmf_accel_0_vs_default": compare_bits(&run1, &child_noaccel, dim),
    });
    report("e5", &out);
    assert!(pass, "E5 failed: {out}");
}

#[test]
#[ignore]
fn encoder_speed_one_thread() {
    let out = run_child(
        "speed",
        &[
            ("CORTIQ_DECISION_CHILD_MODE", "speed"),
            ("CMF_THREADS", "1"),
            ("VECLIB_MAXIMUM_THREADS", "1"),
        ],
    );
    let mut v: Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    v["texts"] = json!("banking77 dev (50 warm-up, then all rows)");
    v["env"] = json!("CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1, release build, child process");
    report("speed", &v);
}

// ------------------------------------------------------------------ init → file → golden

#[test]
#[ignore]
fn release_encoder_file_round_trip() {
    use cortiq_decision::container::{DecisionModel, FileBuilder, Verify};
    use cortiq_decision::manifest::{DEFAULT_ENCODER_GOLDEN_TEXTS, DEFAULT_MODEL_ID, DEFAULT_NAME};
    let ex = export();
    let t0 = Instant::now();
    let init = ex.init_tensors(&DEFAULT_ENCODER_GOLDEN_TEXTS).unwrap();
    let init_secs = t0.elapsed().as_secs_f64();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("encoder.cmf");
    let w = FileBuilder::new(
        DEFAULT_MODEL_ID,
        DEFAULT_NAME,
        init.record.clone(),
        init.tensors,
    )
    .unwrap()
    .write(&out)
    .unwrap();
    let model = DecisionModel::open(&out, Verify::Full).unwrap();
    assert_eq!(model.signal_dim(), 4480);
    let t0 = Instant::now();
    let (se, golden) = SignalEncoder::from_model(&model).unwrap();
    let load_secs = t0.elapsed().as_secs_f64();
    assert_eq!(golden.bit_exact_rows, golden.rows);
    let v = se.signal("How do I top up my card?");
    assert_eq!(v.len(), 4480);
    let golden_sha = common::sha256_hex(
        &init
            .golden
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    report(
        "init_file",
        &json!({"bytes": w.bytes, "file_sha256": w.sha256, "model_sha": w.model_sha,
                "tensors_sha256": init.record.tensors_sha256, "golden_f32le_sha256": golden_sha,
                "golden_rows_bit_exact": golden.bit_exact_rows, "init_seconds": init_secs,
                "load_and_golden_seconds": load_secs, "tables": init.record.tokenizer.tables}),
    );
}

// ------------------------------------------------------------------ Unicode, exhaustive

#[test]
#[ignore]
fn unicode_exhaustive_matches_hf() {
    let probe = std::fs::read_to_string(env_path(PROBE_ENV)).unwrap();
    let mut expect: HashMap<u32, (char, String)> = HashMap::new();
    for line in probe.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split('\t');
        let cp = u32::from_str_radix(it.next().unwrap(), 16).unwrap();
        let cls = it.next().unwrap().chars().next().unwrap();
        let norm: String = it
            .next()
            .unwrap_or("")
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap())
            .collect();
        expect.insert(cp, (cls, norm));
    }
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        checked += 1;
        let (cls, norm) = expect
            .get(&cp)
            .cloned()
            .unwrap_or_else(|| ('o', c.to_string()));
        let mine_norm = wordpiece::normalize(&c.to_string());
        let s = format!("a{c}a");
        let pieces = wordpiece::pre_tokenize(&s);
        let c_str = c.to_string();
        let mine_cls = if pieces == ["a", "a"] {
            'w'
        } else if pieces == ["a", c_str.as_str(), "a"] {
            'p'
        } else if pieces == [s.as_str()] {
            'o'
        } else {
            '?'
        };
        if mine_norm != norm || mine_cls != cls {
            mismatches.push(json!({
                "cp": format!("U+{cp:04X}"), "assigned_unicode_9": cortiq_decision::unicode_tables::is_assigned(c),
                "hf": {"class": cls.to_string(), "norm": norm.chars().map(|x| format!("U+{:04X}", x as u32)).collect::<Vec<_>>()},
                "native": {"class": mine_cls.to_string(), "norm": mine_norm.chars().map(|x| format!("U+{:04X}", x as u32)).collect::<Vec<_>>()},
            }));
        }
    }
    let out = json!({
        "scalars": checked, "mismatches": mismatches.len(),
        "rust_std_unicode": format!("{:?}", char::UNICODE_VERSION),
        "list": mismatches.iter().take(200).collect::<Vec<_>>(),
    });
    report("unicode", &out);
    assert!(
        mismatches.is_empty(),
        "{} scalars differ from HF",
        mismatches.len()
    );
}

// ------------------------------------------------------------------ GPU

static GPU_EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct GpuWatch;

impl tracing::Subscriber for GpuWatch {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let target = event.metadata().target();
        if target.starts_with("cortiq_engine::gpu") {
            GPU_EVENTS.lock().unwrap().push(target.to_string());
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
#[ignore]
fn real_encoder_never_touches_the_gpu() {
    let _ = tracing::subscriber::set_global_default(GpuWatch);
    let gemm = cortiq_engine::fcd::prof::GEMM_CALLS.load(Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    let submits = cortiq_engine::gpu_metal::METAL_SUBMITS.load(Ordering::Relaxed);
    let enc = encoder();
    // 29 tokens or more: the size gemm_nt would send to wgpu (n·k·m ≥ 2^22).
    let long = "I ordered a new card two weeks ago and it still has not arrived, so could you please tell me where it is and when I can expect it to be delivered to my home address?";
    let ids = enc.tokenize(long);
    assert!(ids.len() >= 29, "{} tokens", ids.len());
    let v = enc.encode(long);
    assert!(v.iter().all(|x| x.is_finite()));
    assert_eq!(
        cortiq_engine::fcd::prof::GEMM_CALLS.load(Ordering::Relaxed),
        gemm
    );
    #[cfg(target_os = "macos")]
    assert_eq!(
        cortiq_engine::gpu_metal::METAL_SUBMITS.load(Ordering::Relaxed),
        submits
    );
    let events = GPU_EVENTS.lock().unwrap().clone();
    report(
        "gpu",
        &json!({"tokens": ids.len(), "gpu_events": events, "gemm_nt_calls_delta": 0,
                "cmf_gpu": std::env::var("CMF_GPU").ok()}),
    );
    assert!(events.is_empty(), "GPU events: {events:?}");
}
