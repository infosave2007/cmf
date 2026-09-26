//! The training pipeline end to end on the toy encoder (spec §3, §6.1):
//! `tests/fixtures/toy/encoder` (2 layers, hidden 32) → `init` → `train` /
//! `add-skill` on synthetic data whose labels differ by vocabulary.
//!
//! * carve-out, halves and holdout flags against orders computed by hand with
//!   Python `hashlib` (the expected indices below);
//! * data rules: extra keys, empty and oversized texts, label lengths, a
//!   calibration label without training rows, question checks, unions;
//! * the build: labels, tasks, the one-row warning, gate evidence grids, the rows
//!   blob order and flags, the post-write self-check, the encoder bytes copied
//!   from the base, dev counts equal to a batch evaluation;
//! * determinism: two builds (and different thread counts) give one sha256;
//! * zero forgetting: `add-skill` keeps every prior tensor and manifest byte for
//!   byte and every prior decision bit for bit;
//! * the output is never overwritten and no staging file is left behind.

use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::data::{
    self, CalibrationSource, DataInputs, InputFile, calibration_flags, carve_out, parse_question,
};
use cortiq_decision::eval::{self, EvalInput, EvalOptions, Evaluator};
use cortiq_decision::manifest::{self, TaskState};
use cortiq_decision::rows::{FLAG_HOLDOUT, FLAG_ODD_HALF, Split};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const EPOCH: u64 = 1_790_000_000;

fn toy_export() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder")
}

fn sha_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn file_sha(p: &Path) -> String {
    sha_hex(&std::fs::read(p).unwrap())
}

/// A toy encoder file in `dir`.
fn encoder_file(dir: &Path) -> PathBuf {
    let out = dir.join("enc.cmf");
    build::init_encoder(&toy_export(), &out, Some(EPOCH)).expect("init toy encoder");
    out
}

fn jsonl(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(t, l)| json!({"text": t, "label": l}).to_string() + "\n")
        .collect()
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// Deterministic word generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const POOLS: [(&str, [&str; 8]); 4] = [
    (
        "Weather",
        [
            "rain",
            "snow",
            "forecast",
            "sunny",
            "wind",
            "storm",
            "cloudy",
            "temperature",
        ],
    ),
    (
        "billing",
        [
            "invoice",
            "charge",
            "refund",
            "payment",
            "bill",
            "fee",
            "receipt",
            "statement",
        ],
    ),
    (
        "cards",
        [
            "card",
            "pin",
            "atm",
            "contactless",
            "debit",
            "credit",
            "freeze",
            "replace",
        ],
    ),
    (
        "travel",
        [
            "flight", "hotel", "airport", "booking", "passport", "luggage", "visa", "train",
        ],
    ),
];
const FILLER: [&str; 8] = [
    "please", "help", "my", "the", "today", "need", "about", "with",
];

/// `per_label` rows per label of `POOLS[..labels]`, interleaved by label.
fn synth(labels: usize, per_label: usize, seed: u64, tag: &str) -> Vec<(String, String)> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for i in 0..per_label {
        for (label, pool) in POOLS.iter().take(labels) {
            let mut words = Vec::new();
            for _ in 0..2 + rng.below(3) {
                words.push(pool[rng.below(pool.len())]);
            }
            for _ in 0..1 + rng.below(3) {
                words.push(FILLER[rng.below(FILLER.len())]);
            }
            // Shuffle a little and make the text unique.
            let r = rng.below(words.len());
            words.swap(0, r);
            let text = format!("{} {tag}{i}", words.join(" "));
            out.push((text, label.to_string()));
        }
    }
    out
}

fn question_for(labels: &[&str]) -> String {
    let mut c = serde_json::Map::new();
    for l in labels.iter().rev() {
        c.insert(l.to_string(), json!(format!("The message is about {l}.")));
    }
    json!({"instructions": "Which topic is the message about?", "criteria": c}).to_string()
}

struct ToyData {
    train: PathBuf,
    calibration: PathBuf,
    dev: PathBuf,
    question: PathBuf,
    dev_rows: Vec<(String, String)>,
    cal_rows: Vec<(String, String)>,
}

/// Four labels (30 training rows each, 80 calibration, 20 dev) plus a one-row
/// label `zz_single`.
fn toy_data(dir: &Path) -> ToyData {
    let mut train = synth(4, 30, 1, "t");
    train.insert(17, ("a lonely single row".into(), "zz_single".into()));
    let cal = synth(4, 80, 2, "c");
    let dev = synth(4, 20, 3, "d");
    ToyData {
        train: write(dir, "train.jsonl", &jsonl(&train)),
        calibration: write(dir, "calibration.jsonl", &jsonl(&cal)),
        dev: write(dir, "dev.jsonl", &jsonl(&dev)),
        question: write(
            dir,
            "question.json",
            &question_for(&["Weather", "billing", "cards", "travel", "zz_single"]),
        ),
        dev_rows: dev,
        cal_rows: cal,
    }
}

fn options(skill: &str, d: &ToyData) -> TrainOptions {
    let mut o = TrainOptions::new(skill, vec![d.train.clone()]);
    o.calibration = Some(d.calibration.clone());
    o.dev = Some(d.dev.clone());
    o.question = Some(d.question.clone());
    o.threads = 2;
    o.created_unix = Some(EPOCH);
    o
}

fn no_leftovers(dir: &Path) {
    for e in std::fs::read_dir(dir).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with('.'),
            "temporary file {name} left in {}",
            dir.display()
        );
    }
}

/// Every tensor of a file: name → (dtype, shape, sha256 of the bytes).
fn tensor_digests(m: &DecisionModel) -> BTreeMap<String, (String, Vec<usize>, String)> {
    m.base()
        .tensors
        .iter()
        .map(|e| {
            (
                e.name.clone(),
                (
                    format!("{:?}", e.dtype),
                    e.shape.clone(),
                    sha_hex(m.base().entry_bytes(e)),
                ),
            )
        })
        .collect()
}

fn eval_inputs(rows: &[(String, String)]) -> Vec<EvalInput> {
    rows.iter()
        .map(|(t, l)| EvalInput {
            text: t.clone(),
            label: Some(l.clone()),
        })
        .collect()
}

fn run_eval(
    model: &DecisionModel,
    skill: &str,
    inputs: &[EvalInput],
) -> (Vec<eval::RowResult>, eval::EvalSummary) {
    let ev = Evaluator::new(model, skill).unwrap();
    let mut rows = Vec::new();
    let s = ev
        .run(inputs, EvalOptions::default(), |r| {
            rows.push(r.clone());
            Ok(())
        })
        .unwrap();
    (rows, s)
}

// ------------------------------------------------------------------ data rules

/// The rows of `order_expect.py` (training input, in file order).
fn carve_input() -> Vec<(String, String)> {
    let v: Value = serde_json::from_str(r#"[["alpha one", "alpha"], ["bravo one", "bravo"], ["alpha two", "alpha"], ["alpha three", "alpha"], ["bravo two", "bravo"], ["Zulu solo", "Zulu"], ["alpha four", "alpha"], ["alpha five", "alpha"], ["bravo three", "bravo"], ["bravo four", "bravo"], ["alpha six", "alpha"], ["bravo five", "bravo"], ["alpha seven", "alpha"], ["alpha eight", "alpha"], ["bravo six", "bravo"], ["alpha nine", "alpha"], ["alpha ten", "alpha"], ["alpha eleven", "alpha"]]"#).unwrap();
    pairs(&v)
}

/// Calibration rows (one text twice, with two labels).
fn cal_input() -> Vec<(String, String)> {
    let v: Value = serde_json::from_str(r#"[["alpha cal 0", "alpha"], ["bravo cal 0", "bravo"], ["Zulu cal 0", "Zulu"], ["alpha cal 1", "alpha"], ["bravo cal 1", "bravo"], ["alpha cal 2", "alpha"], ["bravo cal 2", "bravo"], ["alpha cal 3", "alpha"], ["bravo cal 3", "bravo"], ["alpha cal 3", "Zulu"], ["alpha cal 4", "alpha"], ["bravo cal 4", "bravo"], ["alpha cal 5", "alpha"]]"#).unwrap();
    pairs(&v)
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                p[0].as_str().unwrap().to_string(),
                p[1].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn input(name: &str, rows: &[(String, String)]) -> InputFile {
    InputFile::parse(name, jsonl(rows).into_bytes()).unwrap()
}

#[test]
fn carve_out_halves_and_holdout_match_hand_computed_orders() {
    // Expected values from Python hashlib (sorted by sha256(text).hexdigest(),
    // stable): see the module docs of data.rs for the rules.
    let train = carve_input();
    let d = data::prepare(DataInputs {
        train: vec![input("train.jsonl", &train)],
        calibration: None,
        dev: None,
        question: None,
    })
    .unwrap();
    assert_eq!(
        d.labels,
        vec!["Zulu", "alpha", "bravo"],
        "bytewise label order"
    );
    let train_order: Vec<usize> = d.train.iter().map(|r| r.input).collect();
    assert_eq!(
        train_order,
        vec![5, 0, 2, 3, 7, 12, 13, 15, 16, 17, 1, 4, 8, 11, 14]
    );
    assert_eq!(d.task_counts, vec![1, 9, 5]);
    let cal_order: Vec<usize> = d.calibration.iter().map(|r| r.input).collect();
    assert_eq!(cal_order, vec![6, 9, 10]);
    let flags: Vec<u8> = d.calibration.iter().map(|r| r.flags).collect();
    assert_eq!(flags, vec![0, FLAG_ODD_HALF, 0]);
    assert_eq!(d.calibration_source, CalibrationSource::CarveOut);
    assert_eq!(d.calibration_sha256(), d.train_sha256);
    assert_eq!(d.train_sha256, sha_hex(jsonl(&train).as_bytes()));
    assert_eq!(d.report.one_row_labels, vec!["Zulu"]);
    assert!(
        d.report
            .warnings
            .iter()
            .any(|w| w.contains("'Zulu' has one training row")),
        "{:?}",
        d.report.warnings
    );
    // The same carve-out through the public helper.
    let ex: Vec<data::Example> = train
        .iter()
        .map(|(t, l)| data::Example {
            text: t.clone(),
            label: l.clone(),
        })
        .collect();
    assert_eq!(carve_out(&ex).1, vec![6, 9, 10]);

    // A calibration file: halves over all rows, holdout per label.
    let cal = cal_input();
    let d = data::prepare(DataInputs {
        train: vec![input("train.jsonl", &train)],
        calibration: Some(input("calibration.jsonl", &cal)),
        dev: None,
        question: None,
    })
    .unwrap();
    assert_eq!(d.train.len(), 18, "no carve-out with --calibration");
    let order: Vec<usize> = d.calibration.iter().map(|r| r.input).collect();
    assert_eq!(order, vec![6, 7, 9, 11, 3, 12, 5, 2, 10, 8, 0, 1, 4]);
    let flags: Vec<u8> = d.calibration.iter().map(|r| r.flags).collect();
    assert_eq!(flags, vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 2]);
    assert_eq!(flags[8], FLAG_HOLDOUT);
    // Equal texts keep the input order (7 before 9), and the conflict is reported.
    assert_eq!(d.calibration[1].text, d.calibration[2].text);
    assert_eq!(d.report.conflicts_total, 1);
    assert_eq!(d.report.conflicts[0].labels, vec!["Zulu", "alpha"]);
    assert_eq!(d.report.conflicts[0].splits, vec!["calibration"]);
    assert_eq!(d.report.calibration.conflicting_texts, 1);
    assert_eq!(
        calibration_flags(&["a", "b", "a", "a", "a", "a"]),
        vec![0, 1, 0, 1, 0, 1 | 2]
    );
    match &d.calibration_source {
        CalibrationSource::File { name, sha256 } => {
            assert_eq!(name, "calibration.jsonl");
            assert_eq!(sha256, &sha_hex(jsonl(&cal).as_bytes()));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn data_rules_refuse_bad_rows_and_questions() {
    let bad = |body: &str| {
        InputFile::parse("x.jsonl", body.as_bytes().to_vec())
            .unwrap_err()
            .to_string()
    };
    assert!(bad("{\"text\":\"a\",\"label\":\"b\",\"id\":1}\n").contains("x.jsonl:1"));
    assert!(bad("{\"text\":\"a\"}\n").contains("label"));
    assert!(bad("{\"text\":\"a\",\"label\":3}\n").contains("x.jsonl:1"));
    assert!(bad("{\"text\":\"\",\"label\":\"b\"}\n").contains("text is empty"));
    assert!(bad("{\"text\":\"a\",\"label\":\"\"}\n").contains("label has 0 bytes"));
    assert!(
        bad(&format!(
            "{{\"text\":\"a\",\"label\":\"{}\"}}\n",
            "x".repeat(257)
        ))
        .contains("257 bytes")
    );
    assert!(
        bad(&format!(
            "{{\"text\":\"{}\",\"label\":\"b\"}}\n",
            "y".repeat(32 * 1024 + 1)
        ))
        .contains("at most 32768")
    );
    assert!(bad("{\"text\":\"a\",\"text\":\"c\",\"label\":\"b\"}\n").contains("duplicate"));
    assert!(bad("{\"text\":\"a\",\"label\":\"b\"}\nnot json\n").contains("x.jsonl:2"));
    // A text of exactly 32 KiB, blank lines and CRLF are fine.
    let ok = format!(
        "{{\"text\":\"{}\",\"label\":\"b\"}}\r\n\n  \n{{\"text\":\"z\",\"label\":\"b\"}}",
        "y".repeat(32 * 1024)
    );
    let f = InputFile::parse("ok.jsonl", ok.into_bytes()).unwrap();
    assert_eq!(f.examples.len(), 2);
    assert_eq!(f.blank_lines, 2);

    // Duplicates are kept and counted.
    let dup = vec![
        ("same".to_string(), "a".to_string()),
        ("same".to_string(), "a".to_string()),
        ("other".to_string(), "a".to_string()),
        ("third".to_string(), "b".to_string()),
        ("fourth".to_string(), "b".to_string()),
    ];
    let d = data::prepare(DataInputs {
        train: vec![input("t.jsonl", &dup)],
        calibration: Some(input("c.jsonl", &dup[..2])),
        dev: None,
        question: None,
    })
    .unwrap();
    assert_eq!(d.train.len(), 5);
    assert_eq!(d.report.train.duplicates, 1);
    assert_eq!(d.report.calibration.duplicates, 1);
    assert_eq!(d.report.overlap_train_calibration, 1);

    // A calibration label without training rows is an error.
    let cal = vec![
        ("x".to_string(), "a".to_string()),
        ("y".to_string(), "ghost".to_string()),
    ];
    let err = data::prepare(DataInputs {
        train: vec![input("t.jsonl", &dup)],
        calibration: Some(input("c.jsonl", &cal)),
        dev: None,
        question: None,
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("'ghost' (1 rows)"), "{err}");
    assert!(err.contains("no training row"), "{err}");

    // Questions.
    let q = |s: &str| parse_question(s.as_bytes());
    assert!(q(r#"{"instructions":"i","criteria":{"a":"x"},"extra":1}"#).is_err());
    assert!(
        q(r#"{"instructions":"i","criteria":{"a":"x","a":"y"}}"#)
            .unwrap_err()
            .to_string()
            .contains("duplicate criteria key")
    );
    assert!(q(r#"{"instructions":"","criteria":{"a":"x"}}"#).is_err());
    assert!(q(r#"{"instructions":"i","criteria":{"a":1}}"#).is_err());
    assert!(
        q(&format!(
            r#"{{"instructions":"i","criteria":{{"a":"{}"}}}}"#,
            "z".repeat(24_000)
        ))
        .is_err()
    );
    let r = q(r#"{"instructions":"i","criteria":{"b":"x","a":null}}"#).unwrap();
    assert_eq!(r.order(), vec!["b", "a"], "file order kept");
    let err = data::prepare(DataInputs {
        train: vec![input("t.jsonl", &dup)],
        calibration: Some(input("c.jsonl", &dup[..2])),
        dev: None,
        question: Some(q(r#"{"instructions":"i","criteria":{"a":"x","c":"y"}}"#).unwrap()),
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("criteria keys must be exactly"), "{err}");
    let ok = data::prepare(DataInputs {
        train: vec![input("t.jsonl", &dup)],
        calibration: Some(input("c.jsonl", &dup[..2])),
        dev: None,
        question: Some(r),
    });
    assert!(ok.is_ok());

    // A tiny carve-out leaves no calibration row.
    let err = data::prepare(DataInputs {
        train: vec![input("t.jsonl", &dup)],
        calibration: None,
        dev: None,
        question: None,
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("calibration set is empty"), "{err}");
}

#[test]
fn a_union_of_training_files_is_recorded_part_by_part() {
    let a = carve_input();
    let b = vec![
        ("bravo seven".to_string(), "bravo".to_string()),
        ("alpha twelve".to_string(), "alpha".to_string()),
        ("charlie one".to_string(), "charlie".to_string()),
        ("charlie two".to_string(), "charlie".to_string()),
    ];
    let (fa, fb) = (input("train.jsonl", &a), input("dev.jsonl", &b));
    let cat = [jsonl(&a), jsonl(&b)].concat();
    let d = data::prepare(DataInputs {
        train: vec![fa, fb],
        calibration: None,
        dev: None,
        question: None,
    })
    .unwrap();
    assert_eq!(
        d.train_sha256,
        sha_hex(cat.as_bytes()),
        "sha256 of the bytes concatenated"
    );
    assert_eq!(d.labels, vec!["Zulu", "alpha", "bravo", "charlie"]);
    assert_eq!(d.train_parts.len(), 2);
    assert_eq!(d.train_parts[0].name, "train.jsonl");
    assert_eq!(d.train_parts[1].name, "dev.jsonl");
    assert_eq!(d.train_parts[0].read, 18);
    assert_eq!(d.train_parts[1].read, 4);
    assert_eq!(
        d.train_parts.iter().map(|p| p.train).sum::<usize>(),
        d.train.len(),
        "after the carve-out the parts add up to the training rows"
    );
    assert_eq!(d.train.len() + d.calibration.len(), 22);
    // Hand-computed (Python hashlib): the carve-out of the union.
    let mut cal: Vec<usize> = d.calibration.iter().map(|r| r.input).collect();
    cal.sort_unstable();
    assert_eq!(cal, vec![2, 3, 14]);
    assert_eq!((d.train_parts[0].train, d.train_parts[1].train), (15, 4));
    // Rows of the second file follow the first file within their task.
    let alpha: Vec<usize> = d
        .train
        .iter()
        .filter(|r| r.task == 1)
        .map(|r| r.input)
        .collect();
    assert_eq!(*alpha.last().unwrap(), 19, "alpha twelve is input row 19");
}

// ------------------------------------------------------------------ builds

#[test]
fn toy_train_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    let out = dir.path().join("support.cmf");
    let rep = build::train(&enc, &options("support", &d), &out).unwrap();
    let m = &rep.manifest;

    // Labels, tasks, the one-row label.
    assert_eq!(
        m.labels,
        vec!["Weather", "billing", "cards", "travel", "zz_single"]
    );
    for t in &m.tasks[..4] {
        assert_eq!(t.state, TaskState::Active);
        assert_eq!(t.n_train, 30);
        assert_eq!(t.k, 16, "k = min(K, n − 1) with K = 16");
        assert!(t.basis_sha256.is_some());
    }
    let single = &m.tasks[4];
    assert_eq!(single.state, TaskState::Inactive);
    assert_eq!((single.k, single.n_train), (0, 1));
    assert!(single.mean_sha256.is_some() && single.basis_sha256.is_none());
    assert!(
        rep.warnings
            .iter()
            .any(|w| w.contains("'zz_single' has one training row")),
        "{:?}",
        rep.warnings
    );
    assert_eq!(m.recipe.k_max, 16);
    assert_eq!(m.taxonomy_version, 1);
    assert!(m.rubric.is_some());

    // Gate and its evidence.
    let g = &m.gate;
    assert_eq!(g.evidence.odd.grid.len(), 14);
    assert_eq!(g.evidence.odd.grid_theta_off.len(), 14);
    assert_eq!(g.evidence.even.n + g.evidence.odd.n, 320);
    assert_eq!(g.evidence.even.n, 160);
    assert_eq!(g.evidence.calibration.n, 320);
    assert!(g.temperature > 0.0 && g.temperature <= 1.0);
    assert!(g.novelty_theta > 0.0 && g.novelty_theta <= 0.999);
    assert!(g.certified, "the separable toy skill certifies: {:?}", g);
    let chosen = g
        .evidence
        .odd
        .grid
        .iter()
        .find(|r| r.t as f32 as f64 == g.tau)
        .unwrap();
    assert!(chosen.accepted >= 100 && chosen.lb >= 0.95);
    assert_eq!(chosen.accepted, g.evidence.odd.accepted);
    for (on, off) in g
        .evidence
        .odd
        .grid
        .iter()
        .zip(&g.evidence.odd.grid_theta_off)
    {
        assert!(on.accepted <= off.accepted, "theta can only remove rows");
    }

    // Data records.
    assert_eq!(m.data.train.n, 121);
    assert_eq!(m.data.train.sha256, file_sha(&d.train));
    assert_eq!(m.data.calibration.sha256, file_sha(&d.calibration));
    assert_eq!(m.data.calibration.source, "file");
    let dev = m.data.dev.as_ref().unwrap();
    assert_eq!(dev.n, 80);
    assert_eq!(dev.sha256, file_sha(&d.dev));

    // The stored rows: train by task in input order, calibration by sha256 with
    // flags computed here independently.
    let model = DecisionModel::open(&out, Verify::Full).unwrap();
    let rows = model.rows("support").unwrap();
    assert_eq!(rows.count(Split::Train), 121);
    assert_eq!(rows.count(Split::Calibration), 320);
    let train_tasks: Vec<u32> = rows
        .rows
        .iter()
        .filter(|r| r.split == Split::Train)
        .map(|r| r.task)
        .collect();
    assert!(
        train_tasks.windows(2).all(|w| w[0] <= w[1]),
        "train rows grouped by task"
    );
    let mut order: Vec<usize> = (0..d.cal_rows.len()).collect();
    order.sort_by_key(|&i| sha_hex(d.cal_rows[i].0.as_bytes()));
    let labels_in_order: Vec<&str> = order.iter().map(|&i| d.cal_rows[i].1.as_str()).collect();
    let want_flags = calibration_flags(&labels_in_order);
    let got: Vec<(u32, u8)> = rows
        .rows
        .iter()
        .filter(|r| r.split == Split::Calibration)
        .map(|r| (r.task, r.flags))
        .collect();
    let want: Vec<(u32, u8)> = order
        .iter()
        .zip(&want_flags)
        .map(|(&i, &f)| (m.task_of(&d.cal_rows[i].1).unwrap() as u32, f))
        .collect();
    assert_eq!(got, want);
    assert_eq!(
        got.iter().filter(|(_, f)| f & FLAG_HOLDOUT != 0).count(),
        4 * 16,
        "holdout: every 5th of 80 rows per label"
    );

    // Self-check.
    assert_eq!(rep.self_check.calibration_rows, 320);
    assert_eq!(rep.self_check.texts_reencoded, 64);
    assert_eq!(rep.self_check.errors_compared, 320 * 4);

    // The encoder of the base, byte for byte; only the new skill besides it.
    let base = DecisionModel::open(&enc, Verify::Full).unwrap();
    let before = tensor_digests(&base);
    let after = tensor_digests(&model);
    for (name, digest) in &before {
        if name.starts_with(manifest::ENCODER_PREFIX) {
            assert_eq!(after.get(name), Some(digest), "{name} copied byte for byte");
        }
    }
    assert!(after.keys().all(|n| n.starts_with(manifest::ENCODER_PREFIX)
        || n.starts_with("decision.skill.support.")
        || n == manifest::MANIFEST_TENSOR));
    assert_eq!(model.representation_id(), base.representation_id());

    // Dev counts equal a batch evaluation of the written file.
    let (res, sum) = run_eval(&model, "support", &eval_inputs(&d.dev_rows));
    assert_eq!(sum.correct as u64, dev.correct);
    assert_eq!(rep.dev.unwrap().correct as u64, dev.correct);
    assert_eq!(rep.dev.unwrap().accepted, sum.accepted);
    assert_eq!(rep.dev.unwrap().accepted_correct, sum.accepted_correct);
    assert!(
        dev.correct >= 70,
        "the toy skill separates its labels: {}",
        dev.correct
    );
    assert_eq!(res.len(), 80);

    // The report serialises.
    let j = rep.to_json();
    assert_eq!(j["skill"]["id"], "support");
    assert_eq!(j["self_check"]["bit_exact"], true);
    assert_eq!(j["out"]["sha256"], file_sha(&out));
    no_leftovers(dir.path());
}

#[test]
fn two_builds_are_byte_identical_whatever_the_thread_count() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    let mut o = options("support", &d);
    o.threads = 1;
    let a = build::train(&enc, &o, &dir.path().join("a.cmf")).unwrap();
    o.threads = 3;
    let b = build::train(&enc, &o, &dir.path().join("b.cmf")).unwrap();
    assert_eq!(a.out.sha256, b.out.sha256);
    assert_eq!(a.out.bytes, b.out.bytes);
    assert_eq!(file_sha(&dir.path().join("a.cmf")), a.out.sha256);
    assert_eq!(file_sha(&dir.path().join("b.cmf")), a.out.sha256);
    no_leftovers(dir.path());
}

#[test]
fn add_skill_forgets_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    let s1 = dir.path().join("s1.cmf");
    build::train(&enc, &options("support", &d), &s1).unwrap();

    // A second skill: three labels, carve-out calibration, K = 4 from a "CV".
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let train2 = write(&sub, "train.jsonl", &jsonl(&synth(3, 90, 7, "s")));
    let mut o2 = TrainOptions::new("topics-3", vec![train2]);
    o2.k = 4;
    o2.k_source = Some("max-recipe/cv.json chosen_K".into());
    o2.created_unix = Some(EPOCH);
    let s2 = dir.path().join("s2.cmf");
    let rep = build::add_skill(&s1, &o2, &s2).unwrap();
    assert_eq!(rep.skills, vec!["support", "topics-3"]);
    assert_eq!(rep.manifest.recipe.k_max, 4);
    assert_eq!(rep.manifest.data.calibration.source, "carve-out");
    assert_eq!(
        rep.manifest.data.calibration.n, 54,
        "every 5th of 90 rows per label"
    );
    assert!(rep.manifest.tasks.iter().all(|t| t.k == 4));
    assert!(rep.manifest.data.dev.is_none() && rep.manifest.rubric.is_none());

    let m1 = DecisionModel::open(&s1, Verify::Full).unwrap();
    let m2 = DecisionModel::open(&s2, Verify::Full).unwrap();
    let (t1, t2) = (tensor_digests(&m1), tensor_digests(&m2));
    for (name, digest) in &t1 {
        if name == manifest::MANIFEST_TENSOR {
            assert_ne!(
                t2.get(name),
                Some(digest),
                "decision.manifest is the only new tensor"
            );
        } else {
            assert_eq!(t2.get(name), Some(digest), "{name} kept byte for byte");
        }
    }
    let k1 = m1.skill("support").unwrap();
    let k2 = m2.skill("support").unwrap();
    assert_eq!(k1.bytes, k2.bytes);
    assert_eq!(k1.sha256, k2.sha256);

    // Decisions of the prior skill, bit for bit (timings aside).
    let inputs = eval_inputs(&d.dev_rows);
    let (r1, _) = run_eval(&m1, "support", &inputs);
    let (r2, _) = run_eval(&m2, "support", &inputs);
    let strip = |v: &[eval::RowResult]| v.iter().map(|r| r.without_timings()).collect::<Vec<_>>();
    assert_eq!(strip(&r1), strip(&r2));
    for (a, b) in r1.iter().zip(&r2) {
        assert_eq!(a.p_top.to_bits(), b.p_top.to_bits());
        assert_eq!(a.novelty.to_bits(), b.novelty.to_bits());
    }

    // An existing id is refused, before any output is written.
    let err = build::add_skill(&s2, &options("support", &d), &dir.path().join("s3.cmf"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("already exists"), "{err}");
    assert!(!dir.path().join("s3.cmf").exists());

    // `train` from a file with skills keeps only its encoder.
    let s4 = dir.path().join("s4.cmf");
    let rep4 = build::train(&s2, &options("support", &d), &s4).unwrap();
    assert_eq!(rep4.skills, vec!["support"]);
    let m4 = DecisionModel::open(&s4, Verify::Full).unwrap();
    let t4 = tensor_digests(&m4);
    for (name, digest) in &t2 {
        if name.starts_with(manifest::ENCODER_PREFIX) {
            assert_eq!(t4.get(name), Some(digest));
        }
    }
    assert!(!t4.keys().any(|n| n.starts_with("decision.skill.topics-3.")));
    // Same encoder, same data, same epoch: the skill equals the first build.
    assert_eq!(
        m4.skill("support").unwrap().bytes,
        k1.bytes,
        "a skill does not depend on the file it is built into"
    );
    no_leftovers(dir.path());
}

#[test]
fn the_output_is_never_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    let out = dir.path().join("taken.cmf");
    std::fs::write(&out, b"precious").unwrap();
    let err = build::train(&enc, &options("support", &d), &out)
        .unwrap_err()
        .to_string();
    assert!(err.contains("refusing to overwrite"), "{err}");
    assert_eq!(std::fs::read(&out).unwrap(), b"precious");
    let err = build::init_encoder(&toy_export(), &out, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("refusing to overwrite"), "{err}");
    assert_eq!(std::fs::read(&out).unwrap(), b"precious");
    // A missing output directory is refused before any work.
    assert!(
        build::train(
            &enc,
            &options("support", &d),
            &dir.path().join("no/such.cmf")
        )
        .is_err()
    );
    no_leftovers(dir.path());
}

#[test]
fn build_refusals() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    // A calibration label without training rows.
    let mut cal = d.cal_rows.clone();
    cal.push(("a label nobody trained".into(), "ghost".into()));
    let mut o = options("support", &d);
    o.calibration = Some(write(dir.path(), "cal-ghost.jsonl", &jsonl(&cal)));
    let err = build::train(&enc, &o, &dir.path().join("x.cmf"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("'ghost'") && err.contains("no training row"),
        "{err}"
    );
    // Bad skill ids and K.
    let mut o = options("Support", &d);
    assert!(
        build::train(&enc, &o, &dir.path().join("x.cmf"))
            .unwrap_err()
            .to_string()
            .contains("invalid skill id")
    );
    o.skill = "support".into();
    o.k = 0;
    assert!(build::train(&enc, &o, &dir.path().join("x.cmf")).is_err());
    // A language-model-like input (not a decision file) is refused.
    let junk = write(dir.path(), "junk.cmf", "not a cmf file");
    assert!(build::train(&junk, &options("support", &d), &dir.path().join("x.cmf")).is_err());
    assert!(!dir.path().join("x.cmf").exists());
    no_leftovers(dir.path());
}

#[test]
fn an_uncertifiable_calibration_gives_an_uncertified_gate() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    // 2 labels × 20 rows, carve-out: 8 calibration rows, far below 100 accepted.
    let train = write(dir.path(), "train.jsonl", &jsonl(&synth(2, 20, 9, "u")));
    let mut o = TrainOptions::new("small", vec![train]);
    o.created_unix = Some(EPOCH);
    o.threads = 1;
    let rep = build::train(&enc, &o, &dir.path().join("small.cmf")).unwrap();
    let g = &rep.manifest.gate;
    assert!(!g.certified);
    assert_eq!(g.tau, 0.0);
    assert_eq!(g.evidence.calibration.n, 8);
    assert_eq!(g.evidence.odd.grid.len(), 14);
    assert!(
        rep.warnings.iter().any(|w| w.contains("not certified")),
        "{:?}",
        rep.warnings
    );
    // Rows are decided with theta alone; certified is false on every row.
    let m = DecisionModel::open(dir.path().join("small.cmf"), Verify::Full).unwrap();
    let (rows, sum) = run_eval(&m, "small", &eval_inputs(&synth(2, 5, 10, "v")));
    assert!(rows.iter().all(|r| !r.certified));
    assert_eq!(sum.candidates, 2);
    no_leftovers(dir.path());
}

#[test]
fn batch_evaluation_rows() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder_file(dir.path());
    let d = toy_data(dir.path());
    let out = dir.path().join("support.cmf");
    build::train(&enc, &options("support", &d), &out).unwrap();
    let model = DecisionModel::open(&out, Verify::Light).unwrap();
    assert_eq!(eval::select_skill(&model, None).unwrap(), "support");
    assert!(eval::select_skill(&model, Some("nope")).is_err());

    let body = format!(
        "{}\n{}\n",
        json!({"text": d.dev_rows[0].0, "label": d.dev_rows[0].1}),
        json!({"text": "rain and snow forecast for the airport"})
    );
    let inputs = eval::parse_input("in.jsonl", body.as_bytes()).unwrap();
    assert!(eval::parse_input("in.jsonl", b"{\"text\":\"a\",\"x\":1}\n").is_err());
    let ev = Evaluator::new(&model, "support").unwrap();
    let mut out_rows = Vec::new();
    let sum = ev
        .run(
            &inputs,
            EvalOptions {
                bench: true,
                warmup: 1,
            },
            |r| {
                out_rows.push(r.to_json());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(sum.n, 2);
    assert_eq!(sum.labelled, 1);
    let keys: Vec<&str> = out_rows[0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec![
            "i",
            "text_sha256",
            "skill",
            "choice",
            "p_top",
            "confidence",
            "novelty",
            "margin",
            "accepted",
            "certified",
            "input_tokens",
            "errors_top5",
            "timings_us",
            "correct"
        ]
    );
    assert!(out_rows[1].get("correct").is_none());
    let r0 = &out_rows[0];
    assert_eq!(r0["text_sha256"], sha_hex(d.dev_rows[0].0.as_bytes()));
    assert_eq!(
        r0["errors_top5"].as_object().unwrap().len(),
        4,
        "4 active labels"
    );
    assert!(r0.get("text").is_none(), "no text in the output");
    // Jev's confidence and the shortest f32 numbers.
    let p = r0["p_top"].as_f64().unwrap() as f32;
    let c = r0["confidence"].as_f64().unwrap() as f32;
    assert_eq!(c.to_bits(), eval::jev_confidence(p, 4).to_bits());
    assert_eq!(eval::f32_json(0.97), json!(0.97));
    let bench = sum.to_json()["bench"].clone();
    assert_eq!(bench["warmup"], 1);
    assert!(bench["total"]["p50_us"].as_f64().unwrap() > 0.0);
    assert!(sum.render().contains("gate"));
    // The certified flag follows the gate and the winner's origin.
    assert_eq!(r0["certified"], json!(ev.scorer().gate().certified));
}
