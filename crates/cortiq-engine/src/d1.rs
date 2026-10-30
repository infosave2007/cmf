//! d1 — Liquid AI's System One decision models (d1-2B on LFM3-VL): named
//! questions over a state (text, JSON, or pictures) answered from ONE
//! non-generating forward pass, each as a calibrated distribution.
//!
//! This is a port of the checkpoint's own `prompt.py`, `runner.py`,
//! `pointer.py`, `box.py`, `grid.py`, `calibration.py`, `parsing.py` and
//! `api.py`; names follow them. In short:
//!
//! * **Prompt.** `[BOS]<|im_start|>user\n[media][state]\n\n\nQUESTION:\n`
//!   then, per question, its block, `<|im_end|>\n<|im_start|>assistant\n`,
//!   the lead `<think></think>` and, for a typed answer, named markers
//!   (`start:`, `present:`, `top:`, ...) whose colon tokens are the slots.
//! * **`set` attention.** Up to `d1_set_size` questions follow one state,
//!   each after its own `QUESTION:\n`; the full-attention layers see both
//!   ways (the state ↔ everything, a question ↔ the state and itself), the
//!   sliding and convolution layers stay causal (`pipeline::masked`).
//! * **Readout.** A code slot is a softmax over its options' token groups
//!   (each group max-pooled over its spellings) of the log-probabilities, the
//!   untyped questions tempered by the calibration's temperature; a read's
//!   start/end slots point at state positions by the dot product of final
//!   hidden states (position 0, the BOS, is "nothing").
//! * **Answers.** Through the calibration's monotone tables: `found:*`,
//!   `multiple`, `estimate` quantile levels, box confidences (Halton draws).

use crate::Pipeline;
use crate::pipeline::masked::SetMask;
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use serde_json::{Map, Value, json};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const LEAD: &str = "<think></think>";
const ZWSP: char = '\u{200b}';
const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
const INSTANCES: usize = 16;
const MAX_INSTANCES: usize = 64;
const DRAWS: usize = 512;

// ─────────────────────────────── spec ───────────────────────────────

/// The calibration a release ships (`calibration.json`).
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    default: f64,
    temperatures: Map<String, Value>,
    tables: Map<String, Value>,
}

/// `knots` (increasing `[raw, calibrated]` pairs), linear between, flat
/// past the ends.
fn through_knots(value: f64, knots: &[(f64, f64)]) -> f64 {
    if knots.is_empty() {
        return value;
    }
    let index = knots.partition_point(|&(raw, _)| raw < value);
    if index == 0 {
        return knots[0].1;
    }
    if index == knots.len() {
        return knots[knots.len() - 1].1;
    }
    let ((a, pa), (b, pb)) = (knots[index - 1], knots[index]);
    pa + (value - a) * (pb - pa) / (b - a)
}

impl Calibration {
    fn from_json(v: &Value) -> Self {
        Self {
            default: v.get("default").and_then(Value::as_f64).unwrap_or(1.0),
            temperatures: v.get("temperatures").and_then(Value::as_object).cloned().unwrap_or_default(),
            tables: v.get("tables").and_then(Value::as_object).cloned().unwrap_or_default(),
        }
    }

    fn knots(&self, name: &str) -> Vec<(f64, f64)> {
        self.tables
            .get(name)
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| Some((r.get(0)?.as_f64()?, r.get(1)?.as_f64()?)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn temperature(&self, bucket: &str) -> f64 {
        self.temperatures.get(bucket).and_then(Value::as_f64).unwrap_or(self.default)
    }

    fn through(&self, name: &str, value: f64) -> f64 {
        through_knots(value, &self.knots(name))
    }

    /// The level whose observed share is `share`: the table read backwards.
    fn level(&self, name: &str, share: f64) -> f64 {
        let flipped: Vec<(f64, f64)> = self.knots(name).into_iter().map(|(x, y)| (y, x)).collect();
        through_knots(share, &flipped)
    }
}

/// Everything a d1 file records beside its weights (`provenance.d1`).
#[derive(Clone, Debug)]
pub struct D1Spec {
    codes: u8,
    set_size: usize,
    numbers: bool,
    vision_max_pixels: usize,
    edges_interp: bool,
    prompt: u8,
    calibration: Calibration,
}

/// Whether `model` is a d1 decision model.
pub fn is_d1(model: &CmfModel) -> bool {
    model
        .header
        .provenance
        .as_ref()
        .is_some_and(|p| p.get("d1").is_some())
}

impl D1Spec {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("d1"))
            .ok_or("not a d1 file (no provenance.d1)")?;
        let c = &p["config"];
        let attention = c.get("d1_attention").and_then(Value::as_str).unwrap_or("causal");
        if attention != "set" {
            return Err(format!("d1: attention '{attention}' is not supported (set only)"));
        }
        Ok(Self {
            codes: c.get("d1_codes").and_then(Value::as_u64).unwrap_or(1) as u8,
            set_size: c.get("d1_set_size").and_then(Value::as_u64).unwrap_or(16) as usize,
            numbers: c.get("d1_numbers").and_then(Value::as_bool).unwrap_or(false),
            vision_max_pixels: c
                .get("d1_vision_max_pixels")
                .and_then(Value::as_u64)
                .unwrap_or(1_638_400) as usize,
            edges_interp: c.get("d1_box_edges").and_then(Value::as_str) == Some("interp"),
            prompt: c.get("d1_prompt").and_then(Value::as_u64).unwrap_or(1) as u8,
            calibration: Calibration::from_json(&p["calibration"]),
        })
    }
}

// ───────────────────────────── questions ─────────────────────────────

#[derive(Clone, Debug, PartialEq)]
enum ReadKind {
    Text,
    Number,
    Date,
}

impl ReadKind {
    fn name(&self) -> &'static str {
        match self {
            ReadKind::Text => "text",
            ReadKind::Number => "number",
            ReadKind::Date => "date",
        }
    }
    /// A read's longest span, in tokens (`pointer.SPAN_CAPS`).
    fn cap(&self) -> usize {
        match self {
            ReadKind::Text => 512,
            _ => 32,
        }
    }
}

#[derive(Clone, Debug)]
enum Question {
    Choice { said: String, criteria: Vec<(String, Option<String>)>, multiple: bool },
    Noul { said: String, yes_no: Option<(String, String)> },
    Score { said: String, levels: Vec<String> },
    Read { said: String, kind: ReadKind, unit: Option<String> },
    Estimate { said: String, grid: Grid, unit: Option<String> },
    Box { said: String },
    Boxes { said: String, max: Option<usize> },
}

impl Question {
    fn said(&self) -> &str {
        match self {
            Question::Choice { said, .. }
            | Question::Noul { said, .. }
            | Question::Score { said, .. }
            | Question::Read { said, .. }
            | Question::Estimate { said, .. }
            | Question::Box { said }
            | Question::Boxes { said, .. } => said,
        }
    }

    /// Typed answers are read at their own slots and never tempered.
    fn typed(&self) -> bool {
        matches!(
            self,
            Question::Read { .. }
                | Question::Estimate { .. }
                | Question::Box { .. }
                | Question::Boxes { .. }
                | Question::Choice { multiple: true, .. }
        )
    }
}

fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

/// A caller's question in the API's JSON, checked as `prompt.check_question`
/// and built as `prompt.as_question`.
fn parse_question(q: &Value, prompt: u8) -> Result<Question, String> {
    let obj = q.as_object().ok_or(
        "a question is an object such as {\"type\": \"noul\", \"instructions\": \"Is it urgent?\"}",
    )?;
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
    const TYPES: [&str; 8] = ["noul", "choice", "score", "text", "number", "date", "box", "boxes"];
    if !TYPES.contains(&kind) {
        return Err(format!("\"type\" is one of {}, not {kind:?}", TYPES.join(", ")));
    }
    let said = match obj.get("instructions") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
        Some(Value::Object(_)) => return Err("spoken instructions need an audio model (d1-Omni)".into()),
        _ => return Err("\"instructions\" is the question as text".into()),
    };
    let criteria = obj.get("criteria");
    if obj.get("multiple").and_then(Value::as_bool).unwrap_or(false) && kind != "choice" {
        return Err("\"multiple\" belongs to a choice".into());
    }
    if obj.contains_key("max") {
        let most = if prompt == 1 { INSTANCES } else { MAX_INSTANCES };
        let ok = kind == "boxes"
            && obj["max"].as_u64().is_some_and(|m| m >= 1 && m as usize <= most);
        if !ok {
            return Err(format!("\"max\" belongs to boxes, a whole number from 1 to {most}"));
        }
    }
    if obj.contains_key("step") && !obj.contains_key("range") {
        return Err("\"step\" needs a \"range\"".into());
    }
    let unit = obj.get("unit").and_then(Value::as_str).map(str::to_string);
    if let Some(range) = obj.get("range") {
        let bounds: Vec<f64> = range
            .as_array()
            .filter(|a| a.len() == 2)
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        if kind != "number" || bounds.len() != 2 {
            return Err("an estimate is {\"type\": \"number\", \"range\": [low, high]}, with an optional \"step\"".into());
        }
        let (low, high) = (bounds[0], bounds[1]);
        let step = obj.get("step").and_then(Value::as_f64);
        let even = |s: f64| ((high - low) / s - ((high - low) / s).round()).abs() <= 1e-9 * ((high - low) / s).abs().max(1.0);
        if low >= high || step.is_some_and(|s| !(s > 0.0 && even(s))) {
            return Err("an estimate's range runs low to high and its step, above 0, divides it".into());
        }
        let size = match step {
            None => 128,
            Some(s) => 128.min(((high - low) / s).round() as usize + 1),
        };
        return Ok(Question::Estimate { said, grid: Grid { low, high, size }, unit });
    }
    Ok(match kind {
        "text" => Question::Read { said, kind: ReadKind::Text, unit },
        "number" => Question::Read { said, kind: ReadKind::Number, unit },
        "date" => Question::Read { said, kind: ReadKind::Date, unit },
        "box" => Question::Box { said },
        "boxes" => Question::Boxes { said, max: obj.get("max").and_then(Value::as_u64).map(|m| m as usize) },
        "noul" => {
            let yes_no = match criteria {
                None | Some(Value::Null) => None,
                Some(Value::Object(m)) => Some((
                    py_str(m.get("true").unwrap_or(&Value::Null)),
                    py_str(m.get("false").unwrap_or(&Value::Null)),
                )),
                _ => return Err("a noul's \"criteria\", when given, is {\"true\": ..., \"false\": ...}".into()),
            };
            Question::Noul { said, yes_no }
        }
        "score" => {
            let levels: Vec<String> = criteria
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default();
            let n = criteria.and_then(Value::as_array).map_or(0, |a| a.len());
            if !(2..=10).contains(&n) || levels.len() != n {
                return Err("a score's \"criteria\" lists 2 to 10 levels, lowest first".into());
            }
            Question::Score { said, levels }
        }
        _ => {
            let m = criteria.and_then(Value::as_object).filter(|m| !m.is_empty()).ok_or(
                "a choice's \"criteria\" names the options: {\"billing\": \"Charges and refunds\"}",
            )?;
            Question::Choice {
                said,
                criteria: m
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().map(str::to_string)))
                    .collect(),
                multiple: obj.get("multiple").and_then(Value::as_bool).unwrap_or(false),
            }
        }
    })
}

// ─────────────────────────────── grids ───────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
struct Grid {
    low: f64,
    high: f64,
    size: usize,
}

/// An edge's values as numbers: 0-999 thousandths.
const THOUSANDTHS: Grid = Grid { low: 0.0, high: 0.999, size: 1000 };
/// An edge's values as codes, 128 of them.
const EDGE_CODES: Grid = Grid { low: 0.0, high: 1.0, size: 128 };

impl Grid {
    fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }

    fn values(&self) -> Vec<f64> {
        let step = self.step();
        let mut v: Vec<f64> = (0..self.size - 1).map(|i| self.low + i as f64 * step).collect();
        v.push(self.high);
        v
    }

    /// Mean, median and the p10/p90 quantiles; `level` maps a quantile to
    /// the level it is read at.
    fn read(&self, probs: &[f64], level: Option<&dyn Fn(f64) -> f64>) -> Map<String, Value> {
        let total: f64 = probs.iter().sum();
        let values = self.values();
        let mut out = Map::new();
        out.insert(
            "mean".into(),
            json!(values.iter().zip(probs).map(|(v, p)| v * (p / total)).sum::<f64>()),
        );
        let at: Vec<(&str, f64)> = [("p10", 0.1), ("median", 0.5), ("p90", 0.9)]
            .iter()
            .map(|&(k, q)| (k, level.map_or(q, |f| f(q))))
            .collect();
        let mut mass = 0.0;
        for (value, p) in values.iter().zip(probs) {
            mass += p / total;
            for &(key, quantile) in &at {
                if !out.contains_key(key) && mass >= quantile {
                    out.insert(key.into(), json!(*value));
                }
            }
        }
        for &(key, _) in &at {
            out.entry(key.to_string()).or_insert(json!(values[values.len() - 1]));
        }
        out
    }
}

/// Python's `format(x, "g")`: 6 significant digits, trailing zeros dropped,
/// exponent form below 1e-4 or from 1e6.
fn py_g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sci = format!("{:.5e}", x);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        return format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{:.*}", decimals, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

// ───────────────────────────── tokenizer ─────────────────────────────

const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
];

fn option_codes(labels: &[String], scheme: u8) -> Vec<String> {
    let labs: Vec<String> = labels.iter().map(|x| x.trim().to_string()).collect();
    if !labs.is_empty() && labs.iter().all(|k| k.chars().count() == 1 && k.chars().all(char::is_alphabetic)) {
        return labs;
    }
    if labs.len() <= 26 {
        return LETTERS[..labs.len()].iter().map(|s| s.to_string()).collect();
    }
    if scheme == 2 {
        let mut all: Vec<String> = LETTERS.iter().map(|s| s.to_string()).collect();
        for a in LETTERS {
            for b in LETTERS {
                all.push(format!("{a}{b}"));
            }
        }
        all.truncate(labs.len());
        return all;
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

struct Tok<'a> {
    tok: &'a Tokenizer,
    specials: Vec<String>,
}

impl<'a> Tok<'a> {
    fn new(tok: &'a Tokenizer) -> Self {
        let mut specials: Vec<String> = tok
            .special_texts()
            .into_iter()
            .filter(|s| s.chars().count() > 1 && !s.contains(ZWSP))
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Self { tok, specials }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    fn single(&self, text: &str) -> Option<u32> {
        let ids = self.encode(text);
        (ids.len() == 1).then(|| ids[0])
    }

    /// The single-token encodings of `texts`, each once (`prompt._ids`).
    fn ids(&self, texts: &[String]) -> Vec<u32> {
        let mut out = Vec::new();
        for t in texts {
            if let Some(id) = self.single(t) {
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        out
    }

    /// Each label's distinct single-token code (`prompt.aliases`).
    fn aliases(&self, labels: &[String], scheme: u8) -> Result<Vec<(String, u32)>, String> {
        let codes = option_codes(labels, scheme);
        let mut pool: Vec<String> = ('A'..='Z').map(|c| c.to_string()).collect();
        pool.extend((0..100).map(|i| format!("{i:02}")));
        pool.extend(('a'..='z').map(|c| c.to_string()));
        pool.extend((0..200).map(|i| format!("#{i}")));
        for a in 'A'..='Z' {
            for b in 'A'..='Z' {
                pool.push(format!("{a}{b}"));
            }
        }
        let mut used = Vec::new();
        let mut out = Vec::new();
        for code in codes {
            let mut take = |raw: &str| -> bool {
                match self.single(raw) {
                    Some(id) if !used.contains(&id) => {
                        out.push((raw.to_string(), id));
                        used.push(id);
                        true
                    }
                    _ => false,
                }
            };
            if take(&code) {
                continue;
            }
            if !pool.iter().any(|raw| take(raw)) {
                return Err(format!("no single-token alias left for {} options", labels.len()));
            }
        }
        Ok(out)
    }

    /// `text` with every special token's string broken by a zero-width space
    /// after its first character (`prompt.plain`), and when `offsets` the map
    /// from each boundary of the result to the original text.
    fn plain_offsets(&self, text: &str) -> (String, Vec<usize>) {
        let mut text: Vec<char> = text.chars().collect();
        let mut boundaries: Vec<usize> = (0..=text.len()).collect();
        if self.specials.is_empty() {
            return (text.into_iter().collect(), boundaries);
        }
        let pats: Vec<Vec<char>> = self.specials.iter().map(|s| s.chars().collect()).collect();
        loop {
            // Leftmost-first, longest alternative first (the regex alternation).
            let mut hits = Vec::new();
            let mut i = 0;
            while i < text.len() {
                let found = pats.iter().find(|p| text[i..].starts_with(p));
                match found {
                    Some(p) => {
                        hits.push(i);
                        i += p.len();
                    }
                    None => i += 1,
                }
            }
            if hits.is_empty() {
                break;
            }
            for &at in hits.iter().rev() {
                boundaries.insert(at + 1, boundaries[at + 1]);
                text.insert(at + 1, ZWSP);
            }
        }
        (text.into_iter().collect(), boundaries)
    }

    fn plain(&self, text: &str) -> String {
        self.plain_offsets(text).0
    }

    /// Token ids and their character offsets in `text` (byte-level: each
    /// token's bytes mapped back to the characters they fall in).
    fn encode_offsets(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        let ids = self.encode(text);
        let char_of_byte: Vec<usize> = {
            let mut v = vec![0usize; text.len() + 1];
            let mut ci = 0;
            for (bi, ch) in text.char_indices() {
                for k in 0..ch.len_utf8() {
                    v[bi + k] = ci;
                }
                ci += 1;
            }
            v[text.len()] = ci;
            v
        };
        let mut offsets = Vec::with_capacity(ids.len());
        let mut at = 0usize;
        for &id in &ids {
            let n = self.tok.token_bytes(id).len();
            let (a, b) = (at.min(text.len()), (at + n).min(text.len()));
            let start = char_of_byte[a];
            let end = if b > a { char_of_byte[b - 1] + 1 } else { start };
            offsets.push((start, end));
            at += n;
        }
        (ids, offsets)
    }
}

// ───────────────────────────── readouts ─────────────────────────────

/// One answer position's readout.
#[derive(Clone, Debug)]
enum Slot {
    /// A read's start or end: a pointer over state positions.
    Position,
    /// Option token groups; `bucket` names the calibration temperature of
    /// an untyped question.
    Codes { groups: Vec<Vec<u32>>, bucket: Option<String> },
}

fn bucket(kind: &str, k: usize) -> String {
    match kind {
        "noul" => "noul".into(),
        "score" => format!("score:{k}"),
        _ if k <= 3 => "choice:small".into(),
        _ if k <= 8 => "choice:mid".into(),
        _ if k <= 26 => "choice:large".into(),
        _ => "choice:inventory".into(),
    }
}

struct Proto<'a> {
    t: Tok<'a>,
    spec: &'a D1Spec,
}

impl Proto<'_> {
    fn yes_no(&self) -> Result<Vec<Vec<u32>>, String> {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let (yes, no) = (self.t.ids(&s(&["yes", "Yes", "YES"])), self.t.ids(&s(&["no", "No", "NO"])));
        if yes.is_empty() || no.is_empty() {
            return Err("tokenizer has no single-token yes/no".into());
        }
        Ok(vec![yes, no])
    }

    fn score_groups(&self, n: usize) -> Result<Vec<Vec<u32>>, String> {
        let groups: Vec<Vec<u32>> = (0..n).map(|i| self.t.ids(&[i.to_string()])).collect();
        if groups.iter().any(Vec::is_empty) {
            return Err(format!("score with {n} levels needs single-token digits"));
        }
        Ok(groups)
    }

    fn choice_groups(&self, labels: &[String]) -> Result<Vec<Vec<u32>>, String> {
        let mut groups = Vec::new();
        for (code, tid) in self.t.aliases(labels, self.spec.codes)? {
            let mut g = vec![tid];
            g.extend(self.t.ids(&[format!(" {code}")]).into_iter().filter(|&i| i != tid));
            groups.push(g);
        }
        Ok(groups)
    }

    /// Each grid value times `scale` as one token, plain and after a space
    /// (`prompt.number_ids`); None when the tokenizer cannot write one so.
    fn number_ids(&self, grid: &Grid, scale: usize) -> Option<Vec<Vec<u32>>> {
        let mut groups = Vec::new();
        for value in grid.values() {
            let x = value * scale as f64;
            let n = x.round_ties_even();
            if (n - x).abs() > 1e-6 {
                return None;
            }
            let n = n as i64;
            let group = self.t.ids(&[n.to_string(), format!(" {n}")]);
            if group.is_empty() || self.t.encode(&n.to_string()).len() != 1 {
                return None;
            }
            groups.push(group);
        }
        Some(groups)
    }

    /// The labels of a grid's bins on the code readout (`grid_question`).
    fn grid_labels(grid: &Grid) -> Vec<String> {
        grid.values().iter().map(|&v| py_g(v)).collect()
    }

    /// A grid's slot: the tokenizer's numbers on a checkpoint that reads
    /// them so, else digits (≤ 10 values) or option codes.
    fn grid_slot(&self, grid: &Grid, scale: usize) -> Result<Slot, String> {
        if self.spec.numbers {
            if let Some(groups) = self.number_ids(grid, scale) {
                return Ok(Slot::Codes { groups, bucket: None });
            }
        }
        let labels = Self::grid_labels(grid);
        let groups = if grid.size <= 10 {
            self.score_groups(grid.size)?
        } else {
            let keys: Vec<String> = (0..labels.len()).map(|i| i.to_string()).collect();
            self.choice_groups(&keys)?
        };
        Ok(Slot::Codes { groups, bucket: None })
    }

    fn uses_numbers(&self, grid: &Grid, scale: usize) -> bool {
        self.spec.numbers && self.number_ids(grid, scale).is_some()
    }

    /// An edge's grid and scale (`box.edge_grid`).
    fn edge_grid(&self) -> (Grid, usize) {
        if self.uses_numbers(&THOUSANDTHS, 1000) {
            (THOUSANDTHS, 1000)
        } else {
            (EDGE_CODES, 1)
        }
    }

    fn box_slots(&self) -> Result<Vec<Slot>, String> {
        let (grid, scale) = self.edge_grid();
        let mut out = vec![Slot::Codes { groups: self.yes_no()?, bucket: None }];
        for _ in EDGES {
            out.push(self.grid_slot(&grid, scale)?);
        }
        Ok(out)
    }

    fn slots(&self, q: &Question) -> Result<Vec<Slot>, String> {
        Ok(match q {
            Question::Read { .. } => vec![Slot::Position, Slot::Position],
            Question::Estimate { grid, .. } => vec![self.grid_slot(grid, 1)?],
            Question::Box { .. } => self.box_slots()?,
            Question::Boxes { max, .. } => {
                let one = self.box_slots()?;
                let blocks = max.unwrap_or(INSTANCES);
                (0..blocks).flat_map(|_| one.clone()).collect()
            }
            Question::Choice { criteria, multiple: true, .. } => {
                let yn = self.yes_no()?;
                criteria.iter().map(|_| Slot::Codes { groups: yn.clone(), bucket: None }).collect()
            }
            Question::Noul { .. } => vec![Slot::Codes { groups: self.yes_no()?, bucket: Some(bucket("noul", 2)) }],
            Question::Score { levels, .. } => vec![Slot::Codes {
                groups: self.score_groups(levels.len())?,
                bucket: Some(bucket("score", levels.len())),
            }],
            Question::Choice { criteria, .. } => {
                let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
                vec![Slot::Codes {
                    groups: self.choice_groups(&labels)?,
                    bucket: Some(bucket("choice", labels.len())),
                }]
            }
        })
    }

    // ─────────────────────────── prompt text ───────────────────────────

    fn edge_legend(&self) -> Result<String, String> {
        let (grid, scale) = self.edge_grid();
        if scale > 1 {
            return Ok("Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its \
                       top edge, left and right from its left edge."
                .into());
        }
        let keys: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
        let codes = self.t.aliases(&keys, self.spec.codes)?;
        Ok(format!(
            "Each edge is a code on a {}-value scale: {} is 0 (top or left), {} is 1 (bottom or right). Values are \
             equally spaced.",
            grid.size,
            codes[0].0,
            codes[codes.len() - 1].0
        ))
    }

    fn question_block(&self, q: &Question) -> Result<String, String> {
        let said = q.said();
        Ok(match q {
            Question::Box { .. } => {
                let one = if self.edge_grid().1 > 1 { "one number" } else { "one code" };
                format!(
                    "{said}\n\nPresence is yes or no. {} Reply with yes or no at present, then {one} at top, bottom, \
                     left and right.",
                    self.edge_legend()?
                )
            }
            Question::Boxes { .. } => {
                let most = if self.spec.prompt == 1 { format!("{INSTANCES} at most, ") } else { String::new() };
                format!(
                    "{said}\n\nEvery one of them, {most}in reading order (left edge, then top). Present k is yes \
                     when there are at least k. {} Reply with yes or no at each present, then its top, bottom, left \
                     and right.",
                    self.edge_legend()?
                )
            }
            Question::Estimate { grid, unit, .. } => {
                let unit = unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
                if self.uses_numbers(grid, 1) {
                    format!(
                        "{said}\n\nEstimate from {} to {}{unit}, in steps of {}. Reply with the number only.",
                        py_g(grid.low),
                        py_g(grid.high),
                        py_g(grid.step())
                    )
                } else {
                    let codes: Vec<String> = if grid.size <= 10 {
                        (0..grid.size).map(|i| i.to_string()).collect()
                    } else {
                        let keys: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
                        self.t.aliases(&keys, self.spec.codes)?.into_iter().map(|(c, _)| c).collect()
                    };
                    format!(
                        "{said}\n\nEstimate from {} to {}{unit} on {} equally spaced values, step {}: {} is {}, {} \
                         is {}. Reply with the code only.",
                        py_g(grid.low),
                        py_g(grid.high),
                        grid.size,
                        py_g(grid.step()),
                        codes[0],
                        py_g(grid.low),
                        codes[codes.len() - 1],
                        py_g(grid.high)
                    )
                }
            }
            Question::Read { kind, unit, .. } => {
                let unit = match (kind, unit) {
                    (ReadKind::Number, Some(u)) => format!(" (unit: {u})"),
                    _ => String::new(),
                };
                format!("{said}\n\nRead the {}{unit} from the state, or nothing if it is absent.", kind.name())
            }
            Question::Choice { criteria, multiple: true, .. } => {
                let lines: Vec<String> = criteria
                    .iter()
                    .map(|(name, desc)| {
                        format!("{}: {}", serde_json::to_string(name).unwrap(), desc.as_deref().unwrap_or(name))
                    })
                    .collect();
                format!("{said}\n\nOptions:\n{}\n\nReply with yes or no for each option.", lines.join("\n"))
            }
            Question::Choice { criteria, .. } => {
                let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
                let codes = self.t.aliases(&labels, self.spec.codes)?;
                let lines: Vec<String> = codes
                    .iter()
                    .zip(criteria)
                    .map(|((code, _), (label, desc))| {
                        let text = desc.clone().unwrap_or_else(|| label.replace('_', " "));
                        format!("{code} {text}")
                    })
                    .collect();
                format!("{said}\n\nOptions:\n{}\n\nReply with the option code only.", lines.join("\n"))
            }
            Question::Noul { yes_no, .. } => {
                let extra = yes_no
                    .as_ref()
                    .map(|(y, n)| format!("\nYes: {y}\nNo: {n}"))
                    .unwrap_or_default();
                format!("{said}{extra}\n\nReply with yes or no only.")
            }
            Question::Score { levels, .. } => {
                let legend: Vec<String> = levels.iter().enumerate().map(|(i, l)| format!("{i} {l}")).collect();
                format!(
                    "{said}\n\n{}\n\nReply with a single digit 0-{} only.",
                    legend.join("\n"),
                    levels.len() - 1
                )
            }
        })
    }

    /// The question and the assistant header up to its answer slot(s), and
    /// each slot's token offset from the suffix's last token
    /// (`prompt.suffix_layout`).
    fn suffix_layout(&self, q: &Question) -> Result<(String, Vec<isize>), String> {
        let body = self.question_block(q)?;
        let names: Vec<String> = match q {
            Question::Read { .. } => vec!["start".into(), "end".into()],
            Question::Box { .. } => ["present"].iter().chain(EDGES.iter()).map(|s| s.to_string()).collect(),
            Question::Boxes { max, .. } => {
                let blocks = max.unwrap_or(INSTANCES);
                (1..=blocks)
                    .flat_map(|k| ["present"].iter().chain(EDGES.iter()).map(move |n| format!("{n} {k}")))
                    .collect()
            }
            Question::Choice { criteria, multiple: true, .. } => {
                criteria.iter().map(|(n, _)| serde_json::to_string(n).unwrap()).collect()
            }
            _ => {
                return Ok((
                    format!("{}{IM_END}\n{IM_START}assistant\n{LEAD}", self.t.plain(&body)),
                    vec![0],
                ));
            }
        };
        let after = match q {
            Question::Boxes { max, .. } if max.unwrap_or(INSTANCES) < INSTANCES || self.spec.prompt > 1 => "\n",
            _ => "",
        };
        self.marked_suffix(&body, &names, after)
    }

    fn marked_suffix(&self, body: &str, names: &[String], after: &str) -> Result<(String, Vec<isize>), String> {
        let mut text = format!("{}{IM_END}\n{IM_START}assistant\n{LEAD}", self.t.plain(body));
        let mut ends = Vec::new();
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(&self.t.plain(name));
            text.push(':');
            ends.push(text.chars().count());
        }
        if !after.is_empty() {
            let ids = self.t.encode(&format!("{text}{after}"));
            let last = self.t.tok.decode_token(*ids.last().unwrap());
            if last.trim_end().ends_with(':') {
                text.push_str(after);
            }
        }
        let (ids, offsets) = self.t.encode_offsets(&text);
        let chars: Vec<char> = text.chars().collect();
        let mut slots = Vec::new();
        for &end in &ends {
            let matching: Vec<(usize, usize)> = offsets
                .iter()
                .enumerate()
                .filter(|(_, se)| se.0 < end && end - 1 < se.1)
                .map(|(i, se)| (i, se.1))
                .collect();
            if matching.len() != 1 {
                return Err("a slot marker must have an exact colon token position".into());
            }
            let (tok_i, stop) = matching[0];
            let tail: String = chars[end..stop.min(chars.len())].iter().collect();
            if !tail.trim().is_empty() || !self.t.tok.decode_token(ids[tok_i]).trim_end().ends_with(':') {
                return Err("a slot marker token must end with its colon".into());
            }
            slots.push(tok_i as isize - ids.len() as isize + 1);
        }
        Ok((text, slots))
    }
}

// ──────────────────────────── state text ────────────────────────────

/// The state as the block before `QUESTION:` (`prompt.state_block`, style
/// `json_only`): a string as written, anything else as indented JSON.
fn state_block(state: &Value) -> String {
    match state {
        Value::String(s) => format!("{s}\n\n"),
        other => format!("{}\n\n", serde_json::to_string_pretty(other).unwrap_or_default()),
    }
}

/// RFC 6901 paths of the string values in rendered JSON and their character
/// spans, quotes excluded (`json_spans.string_paths`).
fn string_paths(text: &str) -> Vec<(usize, usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    fn skip(c: &[char], mut at: usize) -> usize {
        while at < c.len() && c[at].is_whitespace() {
            at += 1;
        }
        at
    }
    fn string_end(c: &[char], at: usize) -> Option<usize> {
        // `at` is the opening quote; returns the index after the closing one.
        let mut i = at + 1;
        while i < c.len() {
            match c[i] {
                '\\' => i += 2,
                '"' => return Some(i + 1),
                _ => i += 1,
            }
        }
        None
    }
    fn value(c: &[char], at: usize, path: &str, out: &mut Vec<(usize, usize, String)>) -> Option<usize> {
        let at = skip(c, at);
        match c.get(at)? {
            '{' => {
                let mut at = skip(c, at + 1);
                while c.get(at)? != &'}' {
                    let end = string_end(c, at)?;
                    let key: String = serde_json::from_str(&c[at..end].iter().collect::<String>()).ok()?;
                    let colon = skip(c, end);
                    if c.get(colon)? != &':' {
                        return None;
                    }
                    let name = key.replace('~', "~0").replace('/', "~1");
                    at = skip(c, value(c, colon + 1, &format!("{path}/{name}"), out)?);
                    if c.get(at)? == &',' {
                        at = skip(c, at + 1);
                    }
                }
                Some(at + 1)
            }
            '[' => {
                let mut at = skip(c, at + 1);
                let mut index = 0;
                while c.get(at)? != &']' {
                    at = skip(c, value(c, at, &format!("{path}/{index}"), out)?);
                    index += 1;
                    if c.get(at)? == &',' {
                        at = skip(c, at + 1);
                    }
                }
                Some(at + 1)
            }
            '"' => {
                let end = string_end(c, at)?;
                out.push((at + 1, end - 1, path.to_string()));
                Some(end)
            }
            _ => {
                let mut i = at;
                while i < c.len() && !matches!(c[i], ',' | '}' | ']') && !c[i].is_whitespace() {
                    i += 1;
                }
                Some(i)
            }
        }
    }
    if value(&chars, 0, "", &mut out).is_none() {
        out.clear();
    }
    out
}

/// Token positions mapped to the rendered state (`pointer.StatePositions`).
struct StatePositions {
    text: Vec<char>,
    offsets: Vec<Option<(usize, usize)>>,
    paths: Vec<(usize, usize, String)>,
}

impl StatePositions {
    fn pointer_mask(&self) -> Vec<bool> {
        let valid: Vec<bool> = self.offsets.iter().enumerate().map(|(i, s)| i == 0 || s.is_some()).collect();
        let last = valid.iter().rposition(|&v| v).unwrap_or(0);
        valid[..=last].to_vec()
    }

    fn read(&self, first: usize, last: usize) -> Result<Map<String, Value>, String> {
        let mut out = Map::new();
        if first == 0 && last == 0 {
            out.insert("text".into(), Value::Null);
            out.insert("span".into(), Value::Null);
            return Ok(out);
        }
        let (a, b) = match (self.offsets.get(first).copied().flatten(), self.offsets.get(last).copied().flatten()) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err("pointer leaves the state text".into()),
        };
        let (mut s0, mut s1) = (a.0, b.1);
        while s0 < s1 && self.text[s0].is_whitespace() {
            s0 += 1;
        }
        while s1 > s0 && self.text[s1 - 1].is_whitespace() {
            s1 -= 1;
        }
        let held: Vec<&(usize, usize, String)> =
            self.paths.iter().filter(|(start, end, _)| *start < s1 && s0 < *end).collect();
        if held.len() == 1 {
            let (start, end, path) = held[0];
            let (s0, s1) = (s0.max(*start), s1.min(*end));
            let slice: String = self.text[s0..s1].iter().collect();
            let text = serde_json::from_str::<String>(&format!("\"{slice}\"")).unwrap_or(slice);
            out.insert("text".into(), json!(text));
            out.insert("span".into(), json!([s0, s1]));
            out.insert("path".into(), json!(path));
            return Ok(out);
        }
        out.insert("text".into(), json!(self.text[s0..s1].iter().collect::<String>()));
        out.insert("span".into(), json!([s0, s1]));
        Ok(out)
    }
}

/// `pointer.state_positions`: align the prompt `full` (prefix + first
/// suffix) to the rendered state.
fn state_positions(
    t: &Tok,
    state: &Value,
    prefix: &str,
    full: &str,
    offsets: &[(usize, usize)],
) -> Result<StatePositions, String> {
    let full_c: Vec<char> = full.chars().collect();
    let prefix_len = prefix.chars().count();
    let (text, boundaries, start) = if state.is_null() {
        (String::new(), vec![0usize], prefix_len)
    } else {
        let body = state_block(state);
        let (escaped, boundaries) = t.plain_offsets(&body);
        let tail = format!("{escaped}\nQUESTION:\n");
        if !prefix.ends_with(&tail) {
            return Err("state rendering differs from the prompt prefix".into());
        }
        let start = prefix_len - tail.chars().count();
        let text: String = body.chars().take(body.chars().count() - 2).collect();
        (text, boundaries, start)
    };
    let text_len = text.chars().count();
    let text_c: Vec<char> = text.chars().collect();
    let stop = start + boundaries.partition_point(|&b| b <= text_len) - 1;
    let blank = |s: &[char]| s.iter().all(|c| c.is_whitespace());
    let mut out = Vec::with_capacity(offsets.len());
    for &(a, b) in offsets {
        let (lo, hi) = (a.max(start), b.min(stop));
        let mut span = None;
        if lo < hi && blank(&full_c[a..lo]) && blank(&full_c[hi..b.max(hi)]) {
            let (left, right) = (boundaries[lo - start], boundaries[hi - start]);
            if left < right && right <= text_len && !blank(&text_c[left..right]) {
                span = Some((left, right));
            }
        }
        out.push(span);
    }
    if out.first().is_none_or(|s| s.is_some()) {
        return Err("position 0 must be a template token for the nothing readout".into());
    }
    let paths = if !state.is_null() && !state.is_string() { string_paths(&text) } else { Vec::new() };
    Ok(StatePositions { text: text_c, offsets: out, paths })
}

// ─────────────────────────────── boxes ───────────────────────────────

fn halton() -> Vec<[f64; 4]> {
    let radical = |mut i: usize, base: usize| {
        let (mut value, mut scale) = (0.0, 1.0);
        while i > 0 {
            let digit = i % base;
            i /= base;
            scale /= base as f64;
            value += digit as f64 * scale;
        }
        value
    };
    (1..=DRAWS).map(|i| [radical(i, 2), radical(i, 3), radical(i, 5), radical(i, 7)]).collect()
}

fn grid_of(row: &[f64]) -> Grid {
    if row.len() == THOUSANDTHS.size { THOUSANDTHS } else { EDGE_CODES }
}

fn ordered(a: f64, b: f64, grid: &Grid) -> (f64, f64) {
    if a < b {
        return (a, b);
    }
    let half = grid.step() / 2.0;
    let middle = (1.0 - half).min(half.max((a + b) / 2.0));
    (middle - half, middle + half)
}

fn rectangle(e: [f64; 4], grid: &Grid) -> [f64; 4] {
    let (top, bottom) = ordered(e[0], e[1], grid);
    let (left, right) = ordered(e[2], e[3], grid);
    [top, bottom, left, right]
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let area = |x: &[f64; 4]| (x[1] - x[0]) * (x[3] - x[2]);
    let overlap = (a[1].min(b[1]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[2].max(b[2])).max(0.0);
    overlap / (area(a) + area(b) - overlap)
}

fn crossing(probs: &[f64], grid: &Grid) -> f64 {
    let total: f64 = probs.iter().sum();
    let mut mass = 0.0;
    for (value, p) in grid.values().iter().zip(probs) {
        let share = p / total;
        if share > 0.0 && mass + share >= 0.5 {
            return grid.high.min(grid.low.max(value - grid.step() / 2.0 + (0.5 - mass) / share * grid.step()));
        }
        mass += share;
    }
    grid.high
}

fn medians(edges: &[Vec<f64>], interp: bool) -> [f64; 4] {
    let grid = grid_of(&edges[0]);
    let at = |row: &Vec<f64>| {
        if interp {
            crossing(row, &grid)
        } else {
            grid.read(row, None)["median"].as_f64().unwrap()
        }
    };
    rectangle([at(&edges[0]), at(&edges[1]), at(&edges[2]), at(&edges[3])], &grid)
}

fn box_confidence(edges: &[Vec<f64>], b: &[f64; 4]) -> f64 {
    let grid = grid_of(&edges[0]);
    let values = grid.values();
    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .map(|row| {
            let total: f64 = row.iter().sum();
            let mut acc = 0.0;
            row.iter()
                .map(|p| {
                    acc += p / total;
                    acc
                })
                .collect()
        })
        .collect();
    let hits = halton()
        .iter()
        .filter(|point| {
            let mut e = [0f64; 4];
            for (k, (cdf, &u)) in cdfs.iter().zip(point.iter()).enumerate() {
                let i = cdf.partition_point(|&c| c < u).min(grid.size - 1);
                e[k] = values[i];
            }
            iou(&rectangle(e, &grid), b) >= 0.5
        })
        .count();
    hits as f64 / DRAWS as f64
}

fn box_json(b: &[f64; 4]) -> Value {
    json!({"top": b[0], "bottom": b[1], "left": b[2], "right": b[3]})
}

// ─────────────────────────────── API ───────────────────────────────

/// One request: a state (string, any JSON, or null with media alone), named
/// questions, and image sources (path, URL or `data:`).
pub struct Request {
    pub state: Value,
    pub questions: Map<String, Value>,
    pub images: Vec<Value>,
}

/// A d1 engine over a loaded pipeline.
pub struct D1 {
    spec: D1Spec,
}

/// A probability readout of one question.
enum Probs {
    Codes(Vec<Vec<f64>>),
    Read(Map<String, Value>),
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

fn log_softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return x.to_vec();
    }
    let lse = m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    x.iter().map(|v| v - lse).collect()
}

impl D1 {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        Ok(Self { spec: D1Spec::from_model(model)? })
    }

    /// `system_one`: `{"answers": {name: answer}, "usage": {...}}`.
    pub fn system_one(&self, pipeline: &Pipeline, req: &Request) -> Result<Value, String> {
        let p = Proto { t: Tok::new(&pipeline.tokenizer), spec: &self.spec };
        let mut questions = Vec::new();
        for (name, q) in &req.questions {
            let q = parse_question(q, self.spec.prompt).map_err(|e| format!("question {name:?}: {e}"))?;
            questions.push((name.clone(), q));
        }
        if questions.is_empty() {
            return Err("no questions".into());
        }
        if req.state.is_null() && req.images.is_empty() {
            return Err("a request needs a state, media, or both".into());
        }
        let boxes = questions.iter().any(|(_, q)| matches!(q, Question::Box { .. } | Question::Boxes { .. }));
        if boxes && req.images.len() != 1 {
            return Err(format!(
                "a box or boxes question reads exactly one picture; this request has {}",
                req.images.len()
            ));
        }
        // Media: each picture capped to the checkpoint's pixel budget, its
        // processor expansion at the head of the user turn.
        let mut markup = String::new();
        let mut media_rows: Vec<f32> = Vec::new();
        if !req.images.is_empty() {
            let model = pipeline.model.as_ref().ok_or("d1 media need a file-backed pipeline")?;
            let tower = crate::lfm3_vl::Lfm3VlTower::from_model(model, crate::pool::Pool::from_env())?;
            for src in &req.images {
                let frame = crate::media::decode_rgb(&crate::media::load_image_bytes(src)?)?;
                let frame = cap_pixels(&frame, self.spec.vision_max_pixels);
                let input = crate::lfm3_vl::prepare_image(&frame, &tower.cfg, None)?;
                media_rows.extend(tower.forward(&input)?);
                markup.push_str("<|vision_start|>");
                markup.push_str(&"<|image|>".repeat(input.tokens()));
                markup.push_str("<|vision_end|>");
            }
        }
        let bos = pipeline
            .tokenizer
            .bos_token_id
            .map(|id| pipeline.tokenizer.decode_for_protocol(&[id]))
            .unwrap_or_default();
        let header = if req.state.is_null() { "" } else { "QUESTION:\n" };
        let prefix = if req.state.is_null() {
            format!("{bos}{IM_START}user\n{markup}")
        } else {
            format!("{bos}{IM_START}user\n{markup}{}\n{header}", p.t.plain(&state_block(&req.state)))
        };
        let layouts: Vec<(String, Vec<isize>)> =
            questions.iter().map(|(_, q)| p.suffix_layout(q)).collect::<Result<_, _>>()?;
        let slot_kinds: Vec<Vec<Slot>> = questions.iter().map(|(_, q)| p.slots(q)).collect::<Result<_, _>>()?;
        let mut probs: Vec<Option<Probs>> = (0..questions.len()).map(|_| None).collect();
        let mut read_tokens = 0usize;
        let step = self.spec.set_size.max(1);
        for k0 in (0..questions.len()).step_by(step) {
            let k1 = (k0 + step).min(questions.len());
            let first_text = format!("{prefix}{}", layouts[k0].0);
            if std::env::var_os("CMF_D1_DEBUG").is_some() {
                eprintln!("d1: prompt {first_text:?}");
            }
            let (mut ids, offsets) = p.t.encode_offsets(&first_text);
            let state_len = ids.len() - p.t.encode(&layouts[k0].0).len();
            let mut group = vec![-1i32; state_len];
            group.extend(std::iter::repeat_n(1, ids.len() - state_len));
            let mut ends: Vec<Vec<usize>> =
                vec![layouts[k0].1.iter().map(|o| (ids.len() as isize - 1 + o) as usize).collect()];
            for (j, k) in (k0 + 1..k1).enumerate() {
                let own = p.t.encode(&format!("{header}{}", layouts[k].0));
                ids.extend(&own);
                group.extend(std::iter::repeat_n(j as i32 + 2, own.len()));
                ends.push(layouts[k].1.iter().map(|o| (ids.len() as isize - 1 + o) as usize).collect());
            }
            read_tokens += ids.len();
            let needs_positions = (k0..k1).any(|k| matches!(questions[k].1, Question::Read { .. }));
            let positions = if needs_positions {
                Some(state_positions(&p.t, &req.state, &prefix, &first_text, &offsets)?)
            } else {
                None
            };
            // Prompt rows: the embedding, the tower's rows at the pictures.
            let h = pipeline.hidden_size;
            let image_id = p.t.single("<|image|>").ok_or("tokenizer lacks <|image|>")?;
            let mut rows = Vec::with_capacity(ids.len() * h);
            let mut at = 0usize;
            for &id in &ids {
                if id == image_id && !media_rows.is_empty() {
                    rows.extend_from_slice(media_rows.get(at..at + h).ok_or("picture rows run short")?);
                    at += h;
                } else {
                    rows.extend(pipeline.embed_id(id));
                }
            }
            if at != media_rows.len() {
                return Err("picture placeholders and rows differ".into());
            }
            // Every code slot of the group through the head in one GEMM,
            // on the device with the forward where it runs.
            let mut code_at: Vec<usize> = Vec::new();
            for (j, k) in (k0..k1).enumerate() {
                if !matches!(questions[k].1, Question::Read { .. }) {
                    code_at.extend(&ends[j]);
                }
            }
            code_at.sort_unstable();
            code_at.dedup();
            let t_fwd = std::time::Instant::now();
            let (hidden, logz_all) =
                pipeline.forward_masked_head(&rows, &SetMask { state: state_len, group: &group }, &code_at)?;
            if std::env::var_os("CMF_D1_DEBUG").is_some() {
                eprintln!(
                    "d1: forward {} tokens + head over {} slots in {:.0} ms",
                    ids.len(),
                    code_at.len(),
                    t_fwd.elapsed().as_secs_f64() * 1e3
                );
            }
            let vocab = pipeline.vocab_size;
            let logz_of = |pos: usize| {
                let r = code_at.binary_search(&pos).expect("code slot");
                &logz_all[r * vocab..(r + 1) * vocab]
            };
            for (j, k) in (k0..k1).enumerate() {
                let q = &questions[k].1;
                let slots = &slot_kinds[k];
                if slots.len() != ends[j].len() {
                    return Err("slot layout and readouts disagree".into());
                }
                if let Question::Read { kind, .. } = q {
                    let pos = positions.as_ref().unwrap();
                    let mask = pos.pointer_mask();
                    let width = mask.len();
                    let logits: Vec<Vec<f64>> = ends[j]
                        .iter()
                        .map(|&slot| {
                            let qv = &hidden[slot * h..(slot + 1) * h];
                            (0..width)
                                .map(|i| {
                                    if !mask[i] {
                                        return f64::NEG_INFINITY;
                                    }
                                    let kv = &hidden[i * h..(i + 1) * h];
                                    let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
                                    (dot / (h as f32).sqrt()) as f64
                                })
                                .collect()
                        })
                        .collect();
                    let (first, last, found) = span(&logits[0], &logits[1], kind.cap());
                    let mut read = pos.read(first, last)?;
                    read.insert("found".into(), json!(found));
                    probs[k] = Some(Probs::Read(read));
                    continue;
                }
                let mut rows_p = Vec::new();
                for (slot, &end) in slots.iter().zip(&ends[j]) {
                    let Slot::Codes { groups, bucket } = slot else {
                        return Err("a pointer slot outside a read".into());
                    };
                    let logz = logz_of(end);
                    let mut scores: Vec<f64> = groups
                        .iter()
                        .map(|g| g.iter().map(|&i| logz[i as usize] as f64).fold(f64::NEG_INFINITY, f64::max))
                        .collect();
                    if let (Some(b), false) = (bucket, q.typed()) {
                        let temp = self.spec.calibration.temperature(b);
                        for s in &mut scores {
                            *s /= temp;
                        }
                    }
                    rows_p.push(softmax(&scores));
                }
                probs[k] = Some(Probs::Codes(rows_p));
            }
        }
        let context = |q: &Question| match q {
            Question::Read { .. } if !req.state.is_null() => {
                let b = state_block(&req.state);
                b[..b.len() - 2].to_string()
            }
            _ => String::new(),
        };
        let debug = std::env::var_os("CMF_D1_DEBUG").is_some();
        let mut answers = Map::new();
        for ((name, q), pr) in questions.iter().zip(probs) {
            let pr = pr.ok_or("unanswered question")?;
            if debug {
                match &pr {
                    Probs::Codes(rows) => eprintln!("d1: {name}: {rows:?}"),
                    Probs::Read(r) => eprintln!("d1: {name}: {r:?}"),
                }
            }
            answers.insert(name.clone(), self.answer(q, pr, &context(q))?);
        }
        Ok(json!({"answers": answers, "usage": {"input_tokens": read_tokens, "output_tokens": 0}}))
    }

    /// `api.answer`: a question's typed answer from its readouts.
    fn answer(&self, q: &Question, probs: Probs, context: &str) -> Result<Value, String> {
        let cal = &self.spec.calibration;
        Ok(match (q, probs) {
            (Question::Read { kind, unit, .. }, Probs::Read(r)) => {
                let raw = r["found"].as_f64().unwrap_or(0.0);
                let found = cal.through(&format!("found:{}", kind.name()), raw);
                let present = found >= 0.5;
                let text = if present { r.get("text").cloned().unwrap_or(Value::Null) } else { Value::Null };
                let span = if present { r.get("span").cloned().unwrap_or(Value::Null) } else { Value::Null };
                let mut out = Map::new();
                out.insert("type".into(), json!(kind.name()));
                let (mut value, mut unit_out) = (text.clone(), unit.clone().map(Value::String).unwrap_or(Value::Null));
                if let Some(s) = text.as_str() {
                    match kind {
                        ReadKind::Number => {
                            if unit.is_none() {
                                unit_out = crate::d1_parse::written_unit(s).map(Value::String).unwrap_or(Value::Null);
                            }
                            value = crate::d1_parse::parse_number(s, context, unit.as_deref());
                        }
                        ReadKind::Date => value = crate::d1_parse::parse_date(s, context).0.map(Value::String).unwrap_or(Value::Null),
                        ReadKind::Text => {}
                    }
                }
                out.insert(kind.name().into(), value);
                out.insert("span".into(), span);
                out.insert("found".into(), json!(found));
                if *kind != ReadKind::Text {
                    out.insert("text".into(), text);
                }
                if present {
                    if let Some(path) = r.get("path") {
                        out.insert("path".into(), path.clone());
                    }
                }
                if *kind == ReadKind::Number {
                    out.insert("unit".into(), unit_out);
                }
                Value::Object(out)
            }
            (Question::Box { .. }, Probs::Codes(rows)) => {
                let found = cal.through("found:box", rows[0][0]);
                if found < 0.5 {
                    json!({"type": "box", "box": null, "found": found, "confidence": 0.0})
                } else {
                    let b = medians(&rows[1..5], self.spec.edges_interp);
                    let conf = cal.through("box", box_confidence(&rows[1..5], &b));
                    json!({"type": "box", "box": box_json(&b), "found": found, "confidence": conf})
                }
            }
            (Question::Boxes { .. }, Probs::Codes(rows)) => {
                let mut presence = Vec::new();
                let mut below = 1.0f64;
                for k in (0..rows.len()).step_by(5) {
                    below = below.min(cal.through("found:boxes", rows[k][0]));
                    presence.push(below);
                }
                let mut out: Vec<([f64; 4], f64, f64)> = Vec::new();
                for (k, &found) in presence.iter().enumerate() {
                    if found < 0.5 {
                        break;
                    }
                    let block = &rows[5 * k + 1..5 * k + 5];
                    let b = medians(block, self.spec.edges_interp);
                    out.push((b, found, cal.through("boxes", box_confidence(block, &b))));
                }
                out.sort_by(|a, b| (a.0[2], a.0[0]).partial_cmp(&(b.0[2], b.0[0])).unwrap());
                let count = out.len();
                json!({
                    "type": "boxes",
                    "boxes": out.iter().map(|(b, f, c)| json!({"box": box_json(b), "found": f, "confidence": c})).collect::<Vec<_>>(),
                    "count": count,
                    "at_least": count > 0 && count == presence.len(),
                    "presence": presence,
                })
            }
            (Question::Estimate { grid, unit, .. }, Probs::Codes(rows)) => {
                let level = |share: f64| cal.level("estimate", share);
                let has = !cal.knots("estimate").is_empty();
                let v = grid.read(&rows[0], if has { Some(&level) } else { None });
                json!({"type": "number", "number": v["median"], "interval": [v["p10"], v["p90"]], "unit": unit})
            }
            (Question::Noul { .. }, Probs::Codes(rows)) => json!({"type": "noul", "noul": rows[0][0]}),
            (Question::Choice { criteria, multiple: true, .. }, Probs::Codes(rows)) => {
                let ps: Vec<f64> = rows.iter().map(|r| cal.through("multiple", r[0])).collect();
                let names: Vec<&String> = criteria.iter().map(|(n, _)| n).collect();
                let chosen: Vec<&String> = names.iter().zip(&ps).filter(|(_, p)| **p >= 0.5).map(|(n, _)| *n).collect();
                let probs: Map<String, Value> = names.iter().zip(&ps).map(|(n, p)| ((*n).clone(), json!(p))).collect();
                json!({"type": "choice", "choices": chosen, "probabilities": probs})
            }
            (Question::Choice { criteria, .. }, Probs::Codes(rows)) => {
                let ps = &rows[0];
                let best = argmax(ps);
                let probs: Map<String, Value> = criteria.iter().zip(ps).map(|((n, _), p)| (n.clone(), json!(p))).collect();
                json!({"type": "choice", "choice": criteria[best].0, "confidence": ps[best], "probabilities": probs})
            }
            (Question::Score { levels, .. }, Probs::Codes(rows)) => {
                let ps = &rows[0];
                let best = argmax(ps);
                let total: f64 = ps.iter().sum();
                let mut mass = 0.0;
                let mut level = levels.len() - 1;
                for (i, p) in ps.iter().enumerate() {
                    mass += p / total;
                    if mass >= 0.5 {
                        level = i;
                        break;
                    }
                }
                let probs: Map<String, Value> = ps.iter().enumerate().map(|(i, p)| (i.to_string(), json!(p))).collect();
                let legend: Map<String, Value> =
                    levels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
                json!({"type": "score", "score": level as f64, "confidence": ps[best], "probabilities": probs, "legend": legend})
            }
            _ => return Err("readout and question disagree".into()),
        })
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `pointer.spans` for one read: the likeliest span within `cap` tokens of
/// its start/end scores over state positions, `(0, 0)` when not found.
fn span(start_logits: &[f64], end_logits: &[f64], cap: usize) -> (usize, usize, f64) {
    let start = log_softmax(start_logits);
    let end = log_softmax(end_logits);
    let found = -start[0].exp_m1();
    let t = start.len();
    if t == 1 {
        return (0, 0, found);
    }
    let cap = cap.min(t - 1);
    let mut best = (f64::NEG_INFINITY, 1usize, 1usize);
    for i in 1..t {
        // max over end[i..i+cap] (first index of the max on ties)
        let mut mj = i;
        for j in i..(i + cap).min(t) {
            if end[j] > end[mj] {
                mj = j;
            }
        }
        let score = start[i] + end[mj];
        if score > best.0 {
            best = (score, i, mj);
        }
    }
    if found < 0.5 {
        return (0, 0, found);
    }
    (best.1, best.2, found)
}

/// `media.cap_pixels`: a picture larger than `max_pixels`, scaled down
/// (PIL bicubic, `int` sizes).
fn cap_pixels(frame: &crate::media::RgbFrame, max_pixels: usize) -> crate::media::RgbFrame {
    let (w, h) = (frame.width, frame.height);
    if w * h <= max_pixels {
        return frame.clone();
    }
    let scale = (max_pixels as f64 / (w * h) as f64).sqrt();
    let (nw, nh) = (((w as f64 * scale) as usize).max(1), ((h as f64 * scale) as usize).max(1));
    crate::egemma2_vision::resize_bicubic_aa(frame, nh, nw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_g_matches_python() {
        for (x, want) in [
            (0.0, "0"),
            (10.0, "10"),
            (0.5, "0.5"),
            (0.999, "0.999"),
            (1234567.0, "1.23457e+06"),
            (0.00001, "1e-05"),
            (0.1111111, "0.111111"),
            (100000.0, "100000"),
            (-2.5, "-2.5"),
        ] {
            assert_eq!(py_g(x), want, "{x}");
        }
    }

    #[test]
    fn option_codes_follow_the_schemes() {
        let labels = |n: usize| (0..n).map(|i| format!("o{i}")).collect::<Vec<_>>();
        assert_eq!(option_codes(&labels(3), 2), vec!["A", "B", "C"]);
        assert_eq!(option_codes(&labels(28), 2)[26..], ["AA".to_string(), "AB".to_string()]);
        assert_eq!(option_codes(&labels(28), 1)[27], "27");
        assert_eq!(option_codes(&["x".into(), "y".into()], 2), vec!["x", "y"]);
    }

    #[test]
    fn grid_reads_quantiles() {
        let g = Grid { low: 0.0, high: 10.0, size: 11 };
        let mut p = vec![0.0; 11];
        p[2] = 0.6;
        p[3] = 0.4;
        let r = g.read(&p, None);
        assert_eq!(r["median"], json!(2.0));
        assert_eq!(r["p90"], json!(3.0));
        assert_eq!(r["p10"], json!(2.0));
    }

    #[test]
    fn string_paths_of_rendered_json() {
        let text = "{\n  \"customer\": {\n    \"name\": \"Maria\"\n  },\n  \"m\": [\n    \"a\",\n    1\n  ]\n}";
        let p = string_paths(text);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].2, "/customer/name");
        assert_eq!(p[1].2, "/m/0");
    }

    #[test]
    fn span_picks_the_best_pair_or_nothing() {
        let ninf = f64::NEG_INFINITY;
        let (a, b, f) = span(&[-5.0, 3.0, 0.0, ninf], &[-5.0, 0.0, 3.0, ninf], 32);
        assert_eq!((a, b), (1, 2));
        assert!(f > 0.5);
        let (a, b, _) = span(&[5.0, -3.0, -3.0], &[5.0, -3.0, -3.0], 32);
        assert_eq!((a, b), (0, 0));
    }
}
