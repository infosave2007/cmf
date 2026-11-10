//! `cortiq decide-pointer`: typed decisions — `choice`, `noul` (yes/no) and
//! `score` — from a pointer-head decision model (MNN-Pocket-0.3B-Dec): a
//! request `{"state", "questions"}` in, every question's candidate
//! probabilities out, without generating a token.
//!
//! The request is serialized exactly as the authors' `inference.py` does:
//! the state behind a state marker, then one branch per question (its
//! instructions, each candidate between an option and a close marker, a
//! decide marker last). The reference runs all branches in one sequence with
//! a mask that lets a branch see the state and itself only, at positions that
//! continue the state's — the same attention as an ordinary causal pass over
//! `state + branch`, which is what runs here, one pass per question. A
//! candidate's logit is the scaled dot product of two float32 projections
//! (`decision.pointer.{q,k}`) of the final-norm hidden rows at the decide
//! marker and at the candidate's close marker; probabilities are the softmax
//! at the calibrated temperature the file carries
//! (`decision.pointer.temperature`).

use anyhow::{Context, Result, anyhow, bail, ensure};
use cortiq_core::CmfModel;
use cortiq_core::types::TensorDtype;
use cortiq_engine::{Pipeline, SamplerConfig};
use serde_json::{Map, Value, json};
use std::sync::Arc;

/// The control markers: state, question, option, close, decide.
const MARKERS: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];
/// The longest request the model was trained for; longer input is refused.
const MAX_LEN: usize = 32_768;

/// The float32 head and temperature a decision file carries.
struct Head {
    width: usize,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    temperature: f64,
}

fn f32_tensor(model: &CmfModel, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let e = model
        .tensor(name)
        .ok_or_else(|| anyhow!("{name}: not in the file (not a pointer decision model)"))?;
    ensure!(
        e.dtype == TensorDtype::F32 && e.shape == shape,
        "{name}: {:?} {:?}, expected F32 {shape:?}",
        e.dtype,
        e.shape
    );
    let b = model.tensor_bytes(name)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl Head {
    fn of(model: &CmfModel, hidden: usize) -> Result<Self> {
        let width = model
            .tensor("decision.pointer.q.weight")
            .map(|e| e.shape[0])
            .ok_or_else(|| {
                anyhow!("decision.pointer.q.weight: not in the file (not a pointer decision model)")
            })?;
        let t = f32_tensor(model, "decision.pointer.temperature", &[1])?[0] as f64;
        ensure!(
            t.is_finite() && t > 0.0,
            "decision.pointer.temperature: {t}"
        );
        Ok(Self {
            width,
            q_w: f32_tensor(model, "decision.pointer.q.weight", &[width, hidden])?,
            q_b: f32_tensor(model, "decision.pointer.q.bias", &[width])?,
            k_w: f32_tensor(model, "decision.pointer.k.weight", &[width, hidden])?,
            k_b: f32_tensor(model, "decision.pointer.k.bias", &[width])?,
            temperature: t,
        })
    }

    fn project(w: &[f32], b: &[f32], h: &[f32]) -> Vec<f64> {
        b.iter()
            .enumerate()
            .map(|(r, &bias)| {
                let row = &w[r * h.len()..(r + 1) * h.len()];
                bias as f64
                    + row
                        .iter()
                        .zip(h)
                        .map(|(&a, &x)| a as f64 * x as f64)
                        .sum::<f64>()
            })
            .collect()
    }

    /// Logits of the candidates (rows `options`) for the decide row `decide`.
    fn logits(&self, decide: &[f32], options: &[&[f32]]) -> Vec<f64> {
        let q = Self::project(&self.q_w, &self.q_b, decide);
        let scale = 1.0 / (self.width as f64).sqrt();
        options
            .iter()
            .map(|o| {
                Self::project(&self.k_w, &self.k_b, o)
                    .iter()
                    .zip(&q)
                    .map(|(k, q)| k * q)
                    .sum::<f64>()
                    * scale
            })
            .collect()
    }
}

/// Python's `repr` of a float (what `str()` prints): shortest round-trip
/// digits, fixed notation for decimal exponents -4..16, else `1e+16` style.
fn py_float(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp < 0 {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        } else if exp + 1 >= n {
            format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..(exp + 1) as usize],
                &digits[(exp + 1) as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    }
}

/// `str()` of a JSON scalar as Python prints the parsed value.
fn scalar(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => py_float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => s.clone(),
        _ => unreachable!(),
    }
}

/// The training serializer (`render` of `inference.py`): insertion order,
/// two-space indents, `key: value`, `- item`.
fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Array(a) => a
            .iter()
            .map(|item| format!("{pad}- {}", render(item, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(o) => o
            .iter()
            .map(|(k, item)| match item {
                Value::Object(_) | Value::Array(_) => {
                    format!("{pad}{k}:\n{}", render(item, indent + 1))
                }
                _ => format!("{pad}{k}: {}", render(item, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => scalar(v),
    }
}

/// `<|name|>` in user text becomes `<¦name¦>`, so no request can spell a marker.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("<|") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 2..];
        let name_len = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        if name_len > 0 && tail[name_len..].starts_with("|>") {
            out.push_str("<¦");
            out.push_str(&tail[..name_len]);
            out.push_str("¦>");
            rest = &tail[name_len + 2..];
        } else {
            out.push_str("<|");
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

/// One question: its type, candidate keys, candidate texts and instructions.
struct Question {
    id: String,
    kind: String,
    keys: Vec<String>,
    options: Vec<String>,
    instructions: String,
}

fn question(id: &str, v: &Value) -> Result<Question> {
    let o = v
        .as_object()
        .ok_or_else(|| anyhow!("question {id:?} must be an object"))?;
    let kind = o.get("type").and_then(|t| t.as_str()).unwrap_or_default();
    let criteria = o.get("criteria").unwrap_or(&Value::Null);
    let option = |key: &str, d: Option<&Value>| match d {
        None | Some(Value::Null) => key.to_string(),
        Some(Value::String(s)) if s.is_empty() => key.to_string(),
        Some(d) => format!("{key}: {}", render(d, 0)),
    };
    let (keys, options) = match kind {
        "choice" => {
            let c = criteria.as_object().filter(|c| !c.is_empty());
            let c = c.ok_or_else(|| {
                anyhow!("question {id:?}: choice criteria must be a nonempty object")
            })?;
            (
                c.keys().cloned().collect::<Vec<_>>(),
                c.iter().map(|(k, d)| option(k, Some(d))).collect(),
            )
        }
        "noul" => {
            let empty = Map::new();
            let c = match criteria {
                Value::Null => &empty,
                Value::Object(c) if c.keys().all(|k| k == "false" || k == "true") => c,
                _ => bail!("question {id:?}: noul criteria may contain only false and true"),
            };
            (
                vec!["false".into(), "true".into()],
                vec![option("no", c.get("false")), option("yes", c.get("true"))],
            )
        }
        "score" => {
            let c = criteria.as_array().filter(|c| !c.is_empty());
            let c = c.ok_or_else(|| {
                anyhow!("question {id:?}: score criteria must be a nonempty ordered list")
            })?;
            (
                (0..c.len()).map(|i| i.to_string()).collect(),
                c.iter().map(|d| render(d, 0)).collect(),
            )
        }
        _ => bail!("question {id:?}: type must be choice, noul or score"),
    };
    ensure!(
        (1..=255).contains(&keys.len()),
        "question {id:?}: expected 1..255 candidates"
    );
    Ok(Question {
        id: id.to_string(),
        kind: kind.to_string(),
        keys,
        options,
        instructions: render(o.get("instructions").unwrap_or(&Value::Null), 0),
    })
}

/// Answer one request; `temperature` overrides the file's calibrated one.
fn decide(
    p: &mut Pipeline,
    head: &Head,
    marks: &[u32; 5],
    request: &Value,
    temperature: Option<f64>,
) -> Result<Value> {
    let state = request
        .get("state")
        .ok_or_else(|| anyhow!("request must contain state and questions"))?;
    let qs = request
        .get("questions")
        .and_then(|q| q.as_object())
        .ok_or_else(|| anyhow!("questions must be a nonempty object"))?;
    ensure!(!qs.is_empty(), "questions must be a nonempty object");
    ensure!(
        qs.keys().all(|k| !k.is_empty()),
        "question IDs must be nonempty strings"
    );
    let t = temperature.unwrap_or(head.temperature);
    ensure!(
        t.is_finite() && t > 0.0,
        "temperature must be a finite positive number"
    );
    let tok = |s: &str| p.tokenizer.encode(&escape(s));
    let [state_id, question_id, option_id, close_id, decide_id] = *marks;
    let mut state_ids = vec![state_id];
    state_ids.extend(tok(&render(state, 0)));
    // Every branch first, so an oversized request is refused before any pass.
    let mut branches = Vec::new();
    let mut total = state_ids.len();
    for (id, v) in qs {
        let q = question(id, v)?;
        let mut b = vec![question_id];
        b.extend(tok(&q.instructions));
        let mut ends = Vec::new();
        for o in &q.options {
            b.push(option_id);
            b.extend(tok(o));
            b.push(close_id);
            ends.push(state_ids.len() + b.len() - 1);
        }
        b.push(decide_id);
        total += b.len();
        branches.push((q, b, ends));
    }
    ensure!(
        total <= MAX_LEN,
        "input has {total} tokens; maximum is {MAX_LEN}; no truncation"
    );
    let hs = p.hidden_size;
    let mut answers = Map::new();
    for (q, b, ends) in branches {
        let mut ids = state_ids.clone();
        ids.extend(&b);
        let h = p
            .encode_hidden_states(&ids)
            .map_err(|e| anyhow!("question {:?}: {e}", q.id))?;
        let row = |i: usize| &h[i * hs..(i + 1) * hs];
        let opts: Vec<&[f32]> = ends.iter().map(|&i| row(i)).collect();
        let scores = head.logits(row(ids.len() - 1), &opts);
        ensure!(
            scores.iter().all(|s| s.is_finite()),
            "question {:?}: non-finite decision logits",
            q.id
        );
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| ((s - max) / t).exp()).collect();
        let sum: f64 = w.iter().sum();
        let probs: Vec<f64> = w.iter().map(|x| x / sum).collect();
        let winner = scores.iter().position(|&s| s == max).unwrap_or(0);
        let mut a = Map::new();
        a.insert("type".into(), json!(q.kind));
        a.insert(
            "probabilities".into(),
            Value::Object(
                q.keys
                    .iter()
                    .cloned()
                    .zip(probs.iter().map(|&x| json!(x)))
                    .collect(),
            ),
        );
        match q.kind.as_str() {
            "choice" => {
                a.insert("choice".into(), json!(q.keys[winner]));
            }
            "noul" => {
                a.insert("noul".into(), json!(probs[1]));
                a.insert("decision".into(), json!(winner == 1));
            }
            _ => {
                a.insert(
                    "score".into(),
                    json!(
                        probs
                            .iter()
                            .enumerate()
                            .map(|(i, x)| i as f64 * x)
                            .sum::<f64>()
                    ),
                );
                a.insert(
                    "legend".into(),
                    Value::Object(
                        q.keys
                            .iter()
                            .cloned()
                            .zip(q.options.iter().map(|o| json!(o)))
                            .collect(),
                    ),
                );
            }
        }
        answers.insert(q.id, Value::Object(a));
    }
    Ok(json!({"answers": answers, "input_tokens": total, "temperature": t}))
}

/// `cortiq decide-pointer MODEL --input FILE|-`: one request object, or a
/// list of them; the answers as JSON on stdout (a list for a list).
pub fn run(model_path: &str, input: &str, temperature: Option<f64>) -> Result<()> {
    let text = if input == "-" {
        std::io::read_to_string(std::io::stdin())?
    } else {
        std::fs::read_to_string(input).with_context(|| format!("read {input}"))?
    };
    let request: Value = serde_json::from_str(&text).context("the request is not JSON")?;
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut p = Pipeline::from_model(&model, SamplerConfig::default())?;
    let head = Head::of(&model, p.hidden_size)?;
    let mut marks = [0u32; 5];
    for (m, name) in marks.iter_mut().zip(MARKERS) {
        *m = p
            .tokenizer
            .token_to_id(name)
            .ok_or_else(|| anyhow!("the tokenizer has no {name}"))?;
    }
    let out = match &request {
        Value::Array(list) => Value::Array(
            list.iter()
                .map(|r| decide(&mut p, &head, &marks, r, temperature))
                .collect::<Result<Vec<_>>>()?,
        ),
        r => decide(&mut p, &head, &marks, r, temperature)?,
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        for (x, want) in [
            (1.5, "1.5"),
            (3.0, "3.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.0, "123456789012345.0"),
            (-2.25e-7, "-2.25e-07"),
            (1.0e22, "1e+22"),
        ] {
            assert_eq!(py_float(x), want, "{x}");
        }
    }

    #[test]
    fn render_matches_the_training_serializer() {
        let v: Value = serde_json::from_str(r#"{"ticket": {"subject": "Crash", "plan": "pro", "n": 3, "ok": true, "tags": ["a", {"b": 1}]}, "x": null}"#).unwrap();
        assert_eq!(
            render(&v, 0),
            "ticket:\n  subject: Crash\n  plan: pro\n  n: 3\n  ok: True\n  tags:\n    - a\n    - b: 1\nx: "
        );
    }

    #[test]
    fn markers_in_user_text_are_escaped() {
        assert_eq!(
            escape("a <|im_start|> b <|x y|> <|"),
            "a <¦im_start¦> b <|x y|> <|"
        );
    }
}
