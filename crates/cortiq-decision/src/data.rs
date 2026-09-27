//! Training data rules (spec §3.2): JSONL input, the question file, the
//! calibration carve-out, the calibration halves and the self-learning holdout.
//!
//! * **Input.** JSONL, one `{"text","label"}` object per line: the text is
//!   non-empty and at most 32 KiB, the label has 1..=256 bytes; any other key, a
//!   duplicate key or another JSON type is an error (with the file name and the
//!   line number). Lines that are empty or ASCII whitespace are skipped and
//!   counted. Duplicates are kept; conflicts (one text, several labels) are kept
//!   and reported. Several training files (train ∪ dev, spec §3.9) are read in
//!   order and concatenated; the recorded sha256 of such a union is that of the
//!   files' bytes concatenated in that order (`cat a b | sha256sum`), for one
//!   file the file's own sha256.
//! * **Labels.** The labels of the training rows sorted bytewise (Python's code
//!   point order) are the tasks `0..n`. A label with one training row becomes an
//!   inactive task (`k = 0`, never predicted) with a warning; a calibration label
//!   without a training row is an error.
//! * **Carve-out** (no `--calibration`; router `bootstrap.rs:114-136`): within
//!   each label the training rows are ordered by the hex sha256 of their UTF-8
//!   text (stable: equal texts keep the input order) and the positions
//!   `i % 5 == 4` become calibration rows; the others stay training rows.
//! * **Stored order** (spec §2.5): training rows by task, within a task in input
//!   order; calibration rows by the hex sha256 of their text (stable).
//! * **Halves** (`build_ph.py:211-212`): even positions of the calibration order
//!   fit `T` and `θ`, odd positions (flag bit 0) test the gate.
//! * **Holdout** of self-learning (flag bit 1): within a label, the calibration
//!   rows in the calibration order, positions `i % 5 == 4`.
//! * **Question** (`--question`): `{"instructions","criteria"}` and nothing else;
//!   the criteria keys must be exactly the labels (no duplicates); each value is
//!   a string, an object, an array or null of at most 24,000 bytes of canonical
//!   JSON (the Jev option limit, spec §4.4); the file's key order is kept
//!   ([`Rubric::new`]).

use crate::canonical;
use crate::manifest::{self, MAX_TEXT_BYTES, Rubric};
use crate::rows::{FLAG_HOLDOUT, FLAG_ODD_HALF};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde::de::{self, MapAccess, Visitor};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// Positions `i % CARVE_EVERY == CARVE_EVERY − 1` of a label go to calibration.
pub const CARVE_EVERY: usize = 5;
/// Positions `i % HOLDOUT_EVERY == HOLDOUT_EVERY − 1` of a label are holdout.
pub const HOLDOUT_EVERY: usize = 5;
/// Largest canonical JSON of one criteria value (the Jev option limit, spec §4.4).
pub const MAX_CRITERION_BYTES: usize = 24_000;
/// Largest rubric instructions.
pub const MAX_INSTRUCTIONS_BYTES: usize = 1 << 20;
/// Most tasks a skill can hold (the manifest limit).
pub const MAX_LABELS: usize = 1 << 16;
/// Conflicts listed in a [`DataReport`] (all are counted).
pub const MAX_LISTED_CONFLICTS: usize = 100;

/// Lowercase hex sha256 of the UTF-8 bytes of a text.
pub fn text_sha256(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// Check the text rule of every input row: non-empty, at most 32 KiB.
pub fn check_text(text: &str) -> Result<()> {
    ensure!(!text.is_empty(), "text is empty");
    ensure!(
        text.len() <= MAX_TEXT_BYTES,
        "text has {} bytes (at most {MAX_TEXT_BYTES})",
        text.len()
    );
    Ok(())
}

/// Check the label rule: 1..=256 bytes.
pub fn check_label(label: &str) -> Result<()> {
    ensure!(
        manifest::valid_label(label),
        "label has {} bytes (1..=256)",
        label.len()
    );
    Ok(())
}

/// The non-blank lines of a JSONL file as `(line number from 1, line)` (a
/// trailing `\r` removed) and the number of empty or whitespace-only lines.
pub fn jsonl_lines(bytes: &[u8]) -> (Vec<(usize, &[u8])>, usize) {
    let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    // A final newline does not open another line.
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let mut blank = 0usize;
    let mut out = Vec::with_capacity(lines.len());
    for (i, l) in lines.into_iter().enumerate() {
        let l = l.strip_suffix(b"\r").unwrap_or(l);
        if l.iter().all(u8::is_ascii_whitespace) {
            blank += 1;
        } else {
            out.push((i + 1, l));
        }
    }
    (out, blank)
}

/// One labelled row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub text: String,
    pub label: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExample {
    text: String,
    label: String,
}

/// A JSONL input file as read.
#[derive(Clone, Debug)]
pub struct InputFile {
    /// Display name (the file name of the path; 1..=256 bytes).
    pub name: String,
    /// sha256 of the file bytes.
    pub sha256: String,
    /// The file bytes (kept for the sha256 of a union).
    pub raw: Vec<u8>,
    pub examples: Vec<Example>,
    /// Skipped empty or whitespace-only lines.
    pub blank_lines: usize,
}

impl InputFile {
    /// Read and check a JSONL file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        Self::parse(&name, bytes)
    }

    /// Parse JSONL bytes (`name` is used in errors and recorded).
    pub fn parse(name: &str, bytes: Vec<u8>) -> Result<Self> {
        let (lines, blank) = jsonl_lines(&bytes);
        let mut examples = Vec::with_capacity(lines.len());
        for (line, l) in lines {
            let raw: RawExample = serde_json::from_slice(l).map_err(|e| {
                anyhow::anyhow!("{name}:{line}: expected {{\"text\",\"label\"}}: {e}")
            })?;
            check_text(&raw.text).map_err(|e| anyhow::anyhow!("{name}:{line}: {e}"))?;
            check_label(&raw.label).map_err(|e| anyhow::anyhow!("{name}:{line}: {e}"))?;
            examples.push(Example {
                text: raw.text,
                label: raw.label,
            });
        }
        Ok(Self {
            name: record_name(name),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            raw: bytes,
            examples,
            blank_lines: blank,
        })
    }
}

/// A name the manifest can record (1..=256 bytes, cut at a char boundary).
fn record_name(name: &str) -> String {
    if name.is_empty() {
        return "input".into();
    }
    let mut end = name.len().min(256);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    name[..end].to_string()
}

/// sha256 of the files' bytes concatenated in order.
pub fn union_sha256(files: &[InputFile]) -> String {
    let mut h = Sha256::new();
    for f in files {
        h.update(&f.raw);
    }
    format!("{:x}", h.finalize())
}

// ------------------------------------------------------------------ question

/// Map entries in file order, refusing duplicate keys.
struct OrderedEntries(Vec<(String, Value)>);

impl<'de> Deserialize<'de> for OrderedEntries {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedEntries;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object of criteria")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut m: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out: Vec<(String, Value)> = Vec::new();
                let mut seen = BTreeSet::new();
                while let Some((k, v)) = m.next_entry::<String, Value>()? {
                    if !seen.insert(k.clone()) {
                        return Err(de::Error::custom(format!("duplicate criteria key '{k}'")));
                    }
                    out.push((k, v));
                }
                Ok(OrderedEntries(out))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQuestion {
    instructions: String,
    criteria: OrderedEntries,
}

/// Parse a question file (`{"instructions","criteria"}`) into a rubric. The
/// criteria keys are checked against the labels by [`prepare`].
pub fn parse_question(bytes: &[u8]) -> Result<Rubric> {
    let q: RawQuestion = serde_json::from_slice(bytes).map_err(|e| {
        anyhow::anyhow!("question: expected {{\"instructions\",\"criteria\"}}: {e}")
    })?;
    ensure!(
        !q.instructions.is_empty() && q.instructions.len() <= MAX_INSTRUCTIONS_BYTES,
        "question: instructions must have 1..={MAX_INSTRUCTIONS_BYTES} bytes"
    );
    ensure!(!q.criteria.0.is_empty(), "question: criteria is empty");
    let mut criteria = Map::new();
    for (k, v) in q.criteria.0 {
        check_label(&k).map_err(|e| anyhow::anyhow!("question: criteria key '{k}': {e}"))?;
        ensure!(
            matches!(
                v,
                Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null
            ),
            "question: criteria '{k}' must be a string, an object, an array or null"
        );
        let n = canonical::to_string(&v).len();
        ensure!(
            n <= MAX_CRITERION_BYTES,
            "question: criteria '{k}' has {n} bytes (at most {MAX_CRITERION_BYTES})"
        );
        criteria.insert(k, v);
    }
    Ok(Rubric::new(q.instructions, criteria))
}

/// Read a question file.
pub fn read_question(path: impl AsRef<Path>) -> Result<Rubric> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_question(&bytes).with_context(|| path.display().to_string())
}

// ------------------------------------------------------------------ orders

/// Indices of `keys` sorted by key (stable: equal keys keep their order, as
/// Python's `sorted`).
pub fn stable_order<S: AsRef<str>>(keys: &[S]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| keys[a].as_ref().cmp(keys[b].as_ref()));
    order
}

/// The carve-out: `(train, calibration)` indices of `examples`, both in input
/// order. Within each label the rows are ordered by the hex sha256 of their text
/// (stable) and the positions `i % 5 == 4` go to calibration.
pub fn carve_out(examples: &[Example]) -> (Vec<usize>, Vec<usize>) {
    let shas: Vec<String> = examples.iter().map(|e| text_sha256(&e.text)).collect();
    let mut by_label: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, e) in examples.iter().enumerate() {
        by_label.entry(e.label.as_str()).or_default().push(i);
    }
    let mut to_cal = vec![false; examples.len()];
    for idx in by_label.values() {
        let keys: Vec<&str> = idx.iter().map(|&i| shas[i].as_str()).collect();
        for (pos, &o) in stable_order(&keys).iter().enumerate() {
            if pos % CARVE_EVERY == CARVE_EVERY - 1 {
                to_cal[idx[o]] = true;
            }
        }
    }
    let train = (0..examples.len()).filter(|&i| !to_cal[i]).collect();
    let cal = (0..examples.len()).filter(|&i| to_cal[i]).collect();
    (train, cal)
}

/// Flags of calibration rows given in the calibration order (by their labels):
/// [`FLAG_ODD_HALF`] at odd positions, [`FLAG_HOLDOUT`] at positions
/// `i % 5 == 4` within a label.
pub fn calibration_flags<S: AsRef<str>>(labels_in_order: &[S]) -> Vec<u8> {
    let mut per_label: HashMap<&str, usize> = HashMap::new();
    labels_in_order
        .iter()
        .enumerate()
        .map(|(pos, l)| {
            let k = per_label.entry(l.as_ref()).or_insert(0);
            let within = *k;
            *k += 1;
            let mut f = 0u8;
            if pos % 2 == 1 {
                f |= FLAG_ODD_HALF;
            }
            if within % HOLDOUT_EVERY == HOLDOUT_EVERY - 1 {
                f |= FLAG_HOLDOUT;
            }
            f
        })
        .collect()
}

// ------------------------------------------------------------------ prepared data

/// A training row in the stored order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainRow {
    pub text: String,
    /// Task (index of the label in [`SkillData::labels`]).
    pub task: usize,
    /// Index of the row in the concatenated training input.
    pub input: usize,
}

/// A calibration row in the stored (sha256) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationRow {
    pub text: String,
    pub task: usize,
    /// [`FLAG_ODD_HALF`] | [`FLAG_HOLDOUT`].
    pub flags: u8,
    /// Index of the row in the calibration input: the calibration file, or the
    /// concatenated training input for a carve-out.
    pub input: usize,
    /// Hex sha256 of the text.
    pub sha256: String,
}

/// A dev row (evaluated after the build, never fitted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevRow {
    pub text: String,
    /// Task of the label; `None` when the label has no training row (always wrong).
    pub task: Option<usize>,
}

/// Where the calibration rows come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationSource {
    /// `--calibration`.
    File { name: String, sha256: String },
    /// The router holdout rule over the training input.
    CarveOut,
}

/// One training file of a union.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainPart {
    pub name: String,
    pub sha256: String,
    /// Rows read from the file.
    pub read: usize,
    /// Rows of the file that stayed training rows (after a carve-out).
    pub train: usize,
}

/// Row counts of one split.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitStats {
    pub n: usize,
    /// Rows whose `(text, label)` appeared earlier in the same split.
    pub duplicates: usize,
    /// Distinct texts of the split that carry several labels within it.
    pub conflicting_texts: usize,
    pub blank_lines: usize,
}

/// One text with several labels (over all given splits).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conflict {
    pub text_sha256: String,
    pub labels: Vec<String>,
    /// Splits the text occurs in (`train`, `calibration`, `dev`).
    pub splits: Vec<String>,
}

/// What [`prepare`] found in the data (spec §3.2: duplicates and conflicts are
/// kept and reported).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DataReport {
    pub train: SplitStats,
    pub calibration: SplitStats,
    pub dev: Option<SplitStats>,
    /// Distinct texts shared between two splits.
    pub overlap_train_calibration: usize,
    pub overlap_train_dev: usize,
    pub overlap_calibration_dev: usize,
    /// Texts with several labels over all splits; the first
    /// [`MAX_LISTED_CONFLICTS`] (by text sha256) are listed.
    pub conflicts_total: usize,
    pub conflicts: Vec<Conflict>,
    /// Labels with one training row (inactive tasks).
    pub one_row_labels: Vec<String>,
    /// Labels with training rows but no calibration row.
    pub labels_without_calibration: Vec<String>,
    /// Dev labels without a training row, with their row counts.
    pub dev_unknown_labels: Vec<(String, usize)>,
    pub warnings: Vec<String>,
}

impl DataReport {
    pub fn to_json(&self) -> Value {
        let split = |s: &SplitStats| json!({"n": s.n, "duplicates": s.duplicates, "conflicting_texts": s.conflicting_texts, "blank_lines": s.blank_lines});
        json!({
            "train": split(&self.train),
            "calibration": split(&self.calibration),
            "dev": self.dev.as_ref().map(split),
            "overlap": {
                "train_calibration": self.overlap_train_calibration,
                "train_dev": self.overlap_train_dev,
                "calibration_dev": self.overlap_calibration_dev,
            },
            "conflicts_total": self.conflicts_total,
            "conflicts": self.conflicts.iter().map(|c| json!({
                "text_sha256": c.text_sha256, "labels": c.labels, "splits": c.splits,
            })).collect::<Vec<_>>(),
            "one_row_labels": self.one_row_labels,
            "labels_without_calibration": self.labels_without_calibration,
            "dev_unknown_labels": self.dev_unknown_labels.iter()
                .map(|(l, n)| json!({"label": l, "rows": n})).collect::<Vec<_>>(),
            "warnings": self.warnings,
        })
    }
}

/// The inputs of one skill.
#[derive(Clone, Debug)]
pub struct DataInputs {
    /// One or more training files, concatenated in order.
    pub train: Vec<InputFile>,
    pub calibration: Option<InputFile>,
    pub dev: Option<InputFile>,
    pub question: Option<Rubric>,
}

impl DataInputs {
    /// Read the files of one skill.
    pub fn read<P: AsRef<Path>>(
        train: &[P],
        calibration: Option<&Path>,
        dev: Option<&Path>,
        question: Option<&Path>,
    ) -> Result<Self> {
        Ok(Self {
            train: train
                .iter()
                .map(InputFile::read)
                .collect::<Result<Vec<_>>>()?,
            calibration: calibration.map(InputFile::read).transpose()?,
            dev: dev.map(InputFile::read).transpose()?,
            question: question.map(read_question).transpose()?,
        })
    }
}

/// The rows of one skill, in the stored order, with their records.
#[derive(Clone, Debug)]
pub struct SkillData {
    /// Task labels, sorted bytewise.
    pub labels: Vec<String>,
    /// Training rows: by task, within a task in input order.
    pub train: Vec<TrainRow>,
    /// Training rows per task.
    pub task_counts: Vec<usize>,
    /// Calibration rows in the sha256 order, with their flags.
    pub calibration: Vec<CalibrationRow>,
    pub calibration_source: CalibrationSource,
    pub dev: Option<Vec<DevRow>>,
    /// Name and sha256 of the dev file.
    pub dev_file: Option<(String, String)>,
    /// sha256 of the training input ([`union_sha256`] for several files).
    pub train_sha256: String,
    /// The files of a union (empty for one file).
    pub train_parts: Vec<TrainPart>,
    pub rubric: Option<Rubric>,
    pub report: DataReport,
}

impl SkillData {
    /// sha256 recorded for the calibration rows: the file's, or the training
    /// input's for a carve-out.
    pub fn calibration_sha256(&self) -> &str {
        match &self.calibration_source {
            CalibrationSource::File { sha256, .. } => sha256,
            CalibrationSource::CarveOut => &self.train_sha256,
        }
    }

    /// Tasks with at least [`crate::fit::MIN_ROWS_ACTIVE`] training rows.
    pub fn active(&self, task: usize) -> bool {
        self.task_counts[task] >= crate::fit::MIN_ROWS_ACTIVE
    }
}

fn split_stats(examples: &[Example], blank: usize) -> SplitStats {
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut labels_of: HashMap<&str, BTreeSet<&str>> = HashMap::new();
    let mut duplicates = 0usize;
    for e in examples {
        if !seen.insert((e.text.as_str(), e.label.as_str())) {
            duplicates += 1;
        }
        labels_of
            .entry(e.text.as_str())
            .or_default()
            .insert(&e.label);
    }
    SplitStats {
        n: examples.len(),
        duplicates,
        conflicting_texts: labels_of.values().filter(|s| s.len() > 1).count(),
        blank_lines: blank,
    }
}

/// Apply the data rules of spec §3.2 to the inputs of one skill.
pub fn prepare(inputs: DataInputs) -> Result<SkillData> {
    let DataInputs {
        train: train_files,
        calibration: cal_file,
        dev: dev_file,
        question,
    } = inputs;
    ensure!(!train_files.is_empty(), "no training file");

    // The training input: the files concatenated in order.
    let mut all_train: Vec<Example> = Vec::new();
    let mut part_of: Vec<usize> = Vec::new();
    let mut train_blank = 0usize;
    for (p, f) in train_files.iter().enumerate() {
        all_train.extend(f.examples.iter().cloned());
        part_of.extend(std::iter::repeat_n(p, f.examples.len()));
        train_blank += f.blank_lines;
    }
    ensure!(!all_train.is_empty(), "the training input has no rows");
    let train_sha256 = if train_files.len() == 1 {
        train_files[0].sha256.clone()
    } else {
        union_sha256(&train_files)
    };

    let labels: Vec<String> = all_train
        .iter()
        .map(|e| e.label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    ensure!(
        labels.len() <= MAX_LABELS,
        "{} labels (at most {MAX_LABELS})",
        labels.len()
    );
    let task_of: HashMap<&str, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();

    // Calibration rows (with their index in the calibration input) and the rows
    // that stay training rows.
    let (train_idx, cal_rows, calibration_source, cal_blank) = match &cal_file {
        Some(f) => (
            (0..all_train.len()).collect::<Vec<_>>(),
            f.examples.iter().cloned().enumerate().collect::<Vec<_>>(),
            CalibrationSource::File {
                name: f.name.clone(),
                sha256: f.sha256.clone(),
            },
            f.blank_lines,
        ),
        None => {
            let (t, c) = carve_out(&all_train);
            let cal = c.iter().map(|&i| (i, all_train[i].clone())).collect();
            (t, cal, CalibrationSource::CarveOut, 0)
        }
    };
    let cal_examples: Vec<Example> = cal_rows.iter().map(|(_, e)| e.clone()).collect();

    // Every calibration label must have a training row.
    let mut missing: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &cal_examples {
        if !task_of.contains_key(e.label.as_str()) {
            *missing.entry(e.label.as_str()).or_insert(0) += 1;
        }
    }
    if !missing.is_empty() {
        let list: Vec<String> = missing
            .iter()
            .take(20)
            .map(|(l, n)| format!("'{l}' ({n} rows)"))
            .collect();
        bail!(
            "{} calibration label(s) have no training row: {}{}",
            missing.len(),
            list.join(", "),
            if missing.len() > 20 { ", …" } else { "" }
        );
    }
    ensure!(
        !cal_examples.is_empty(),
        "the calibration set is empty (the carve-out takes every 5th row of a label in sha256 order): give --calibration or more rows per label"
    );

    // The question.
    if let Some(r) = &question {
        let keys: BTreeSet<&str> = r.criteria.keys().map(String::as_str).collect();
        let want: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
        if keys != want {
            let extra: Vec<&&str> = keys.difference(&want).take(20).collect();
            let absent: Vec<&&str> = want.difference(&keys).take(20).collect();
            bail!(
                "question: the criteria keys must be exactly the {} training labels (keys that are not labels: {extra:?}; labels without a criterion: {absent:?})",
                want.len()
            );
        }
    }

    // Training rows in the stored order.
    let mut per_task: Vec<Vec<usize>> = vec![Vec::new(); labels.len()];
    for &i in &train_idx {
        per_task[task_of[all_train[i].label.as_str()]].push(i);
    }
    let task_counts: Vec<usize> = per_task.iter().map(Vec::len).collect();
    ensure!(
        task_counts
            .iter()
            .any(|&n| n >= crate::fit::MIN_ROWS_ACTIVE),
        "no label has {} training rows: nothing could be predicted",
        crate::fit::MIN_ROWS_ACTIVE
    );
    let mut train = Vec::with_capacity(train_idx.len());
    for (t, idx) in per_task.iter().enumerate() {
        for &i in idx {
            train.push(TrainRow {
                text: all_train[i].text.clone(),
                task: t,
                input: i,
            });
        }
    }
    let mut train_parts = Vec::new();
    if train_files.len() > 1 {
        let mut kept = vec![0usize; train_files.len()];
        for &i in &train_idx {
            kept[part_of[i]] += 1;
        }
        for (p, f) in train_files.iter().enumerate() {
            train_parts.push(TrainPart {
                name: f.name.clone(),
                sha256: f.sha256.clone(),
                read: f.examples.len(),
                train: kept[p],
            });
        }
    }

    // Calibration rows in the sha256 order, with their flags.
    let cal_shas: Vec<String> = cal_examples.iter().map(|e| text_sha256(&e.text)).collect();
    let order = stable_order(&cal_shas);
    let ordered_labels: Vec<&str> = order
        .iter()
        .map(|&j| cal_examples[j].label.as_str())
        .collect();
    let flags = calibration_flags(&ordered_labels);
    let calibration: Vec<CalibrationRow> = order
        .iter()
        .zip(&flags)
        .map(|(&j, &f)| CalibrationRow {
            text: cal_examples[j].text.clone(),
            task: task_of[cal_examples[j].label.as_str()],
            flags: f,
            input: cal_rows[j].0,
            sha256: cal_shas[j].clone(),
        })
        .collect();

    // Dev rows.
    let dev: Option<Vec<DevRow>> = dev_file.as_ref().map(|f| {
        f.examples
            .iter()
            .map(|e| DevRow {
                text: e.text.clone(),
                task: task_of.get(e.label.as_str()).copied(),
            })
            .collect()
    });

    // The report.
    let train_examples: Vec<Example> = train_idx.iter().map(|&i| all_train[i].clone()).collect();
    let dev_examples: &[Example] = dev_file.as_ref().map_or(&[], |f| &f.examples);
    let mut report = DataReport {
        train: split_stats(&train_examples, train_blank),
        calibration: split_stats(&cal_examples, cal_blank),
        dev: dev_file
            .as_ref()
            .map(|f| split_stats(&f.examples, f.blank_lines)),
        ..DataReport::default()
    };
    let texts =
        |ex: &[Example]| -> BTreeSet<String> { ex.iter().map(|e| e.text.clone()).collect() };
    let (t_train, t_cal, t_dev) = (
        texts(&train_examples),
        texts(&cal_examples),
        texts(dev_examples),
    );
    report.overlap_train_calibration = t_train.intersection(&t_cal).count();
    report.overlap_train_dev = t_train.intersection(&t_dev).count();
    report.overlap_calibration_dev = t_cal.intersection(&t_dev).count();
    let mut by_text: BTreeMap<&str, (BTreeSet<&str>, BTreeSet<&str>)> = BTreeMap::new();
    for (split, ex) in [
        ("train", &train_examples[..]),
        ("calibration", &cal_examples[..]),
        ("dev", dev_examples),
    ] {
        for e in ex {
            let ent = by_text.entry(e.text.as_str()).or_default();
            ent.0.insert(e.label.as_str());
            ent.1.insert(split);
        }
    }
    let mut conflicts: Vec<Conflict> = by_text
        .iter()
        .filter(|(_, (l, _))| l.len() > 1)
        .map(|(t, (l, s))| Conflict {
            text_sha256: text_sha256(t),
            labels: l.iter().map(|x| x.to_string()).collect(),
            splits: ["train", "calibration", "dev"]
                .iter()
                .filter(|x| s.contains(*x))
                .map(|x| x.to_string())
                .collect(),
        })
        .collect();
    conflicts.sort_by(|a, b| a.text_sha256.cmp(&b.text_sha256));
    report.conflicts_total = conflicts.len();
    conflicts.truncate(MAX_LISTED_CONFLICTS);
    report.conflicts = conflicts;
    let mut cal_count = vec![0usize; labels.len()];
    for r in &calibration {
        cal_count[r.task] += 1;
    }
    for (t, l) in labels.iter().enumerate() {
        if task_counts[t] == 1 {
            report.one_row_labels.push(l.clone());
            report.warnings.push(format!(
                "label '{l}' has one training row: its task is inactive (k = 0) and never predicted"
            ));
        }
        if cal_count[t] == 0 {
            report.labels_without_calibration.push(l.clone());
        }
    }
    if !report.labels_without_calibration.is_empty() {
        report.warnings.push(format!(
            "{} label(s) have no calibration row: {:?}",
            report.labels_without_calibration.len(),
            report
                .labels_without_calibration
                .iter()
                .take(20)
                .collect::<Vec<_>>()
        ));
    }
    let mut unknown: BTreeMap<&str, usize> = BTreeMap::new();
    for e in dev_examples {
        if !task_of.contains_key(e.label.as_str()) {
            *unknown.entry(e.label.as_str()).or_insert(0) += 1;
        }
    }
    report.dev_unknown_labels = unknown.iter().map(|(l, n)| (l.to_string(), *n)).collect();
    if !unknown.is_empty() {
        report.warnings.push(format!(
            "{} dev label(s) have no training row and count as wrong: {:?}",
            unknown.len(),
            unknown.keys().take(20).collect::<Vec<_>>()
        ));
    }
    if report.conflicts_total > 0 {
        report.warnings.push(format!(
            "{} text(s) carry several labels (kept; listed in the data report)",
            report.conflicts_total
        ));
    }
    let duplicates = report.train.duplicates + report.calibration.duplicates;
    if duplicates > 0 {
        report.warnings.push(format!(
            "{duplicates} duplicate row(s) in train/calibration (kept)"
        ));
    }
    if report.overlap_train_calibration > 0 {
        report.warnings.push(format!(
            "{} text(s) occur in both train and calibration",
            report.overlap_train_calibration
        ));
    }

    Ok(SkillData {
        labels,
        train,
        task_counts,
        calibration,
        calibration_source,
        dev,
        dev_file: dev_file.map(|f| (f.name, f.sha256)),
        train_sha256,
        train_parts,
        rubric: question,
        report,
    })
}
