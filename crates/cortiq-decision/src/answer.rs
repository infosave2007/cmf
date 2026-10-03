//! Answers per question type, confidence and rounding (spec §4.7).
//!
//! * **Local and abstained choice answers** `{type:"choice", choice,
//!   probabilities, confidence}`: `probabilities` lists every option of the
//!   request in request order; `confidence = (N·p_max − 1)/(N − 1)` in f32 with
//!   N the number of options — Jev's formula ([`crate::eval::jev_confidence`],
//!   the value `cortiq decide` reports). The answer carries it clamped at 0
//!   ([`answer_confidence`]): f32 rounding can make `N·p_max` fall a few ulp
//!   below 1 when all options tie, and Jev's schema bounds confidence to
//!   [0, 1]; the value is otherwise unchanged.
//! * **Numbers**: the shortest decimal of the f32 (`0.97`, not
//!   `0.9700000286102295`); with `round = 2` ([`Rounding::Hundredths`]) every
//!   probability and the confidence are quantised to hundredths (round half
//!   away from zero of the exact f32 value × 100) and 0 and 1 are written as
//!   JSON integers — the form of every stored Jev answer.
//! * **Oracle and cache answers** (the OpenRouter schema makes probabilities and
//!   confidence optional): choice `{type, choice}`; score `{type, score: level,
//!   legend: {"0": level 0, …}}`; noul `{type, noul: 1|0, value_semantics:
//!   "boolean_verdict_not_probability"}`. On `/v1/systemone` a choice verdict
//!   is given the one-hot distribution ([`complete_choice`]): Jev's schema,
//!   and the Decision Index validator, require a probability for every option.

use crate::eval::{f32_json, jev_confidence};
use crate::protocol::{Question, QuestionKind};
use serde_json::{Map, Value, json};

/// How answer numbers are written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Shortest f32 decimal.
    #[default]
    Exact,
    /// Hundredths, 0 and 1 as integers (`round = 2`).
    Hundredths,
}

impl Rounding {
    /// From `response.round` / `cmf.round` (`Some(2)` → hundredths).
    pub fn from_config(round: Option<u8>) -> Self {
        match round {
            Some(crate::config::ROUND_HUNDREDTHS) => Rounding::Hundredths,
            _ => Rounding::Exact,
        }
    }
}

/// `value_semantics` of a noul answer from the oracle.
pub const NOUL_SEMANTICS: &str = "boolean_verdict_not_probability";

/// Hundredths of an f32 in [0, 1]: `round(p·100)/100` as a JSON number, 0 and 1
/// as integers.
pub fn round2(p: f32) -> Value {
    let k = (f64::from(p) * 100.0).round();
    if k <= 0.0 {
        return json!(0);
    }
    if k >= 100.0 {
        return json!(1);
    }
    let q = k / 100.0;
    serde_json::Number::from_f64(q).map_or(Value::Null, Value::Number)
}

/// A probability or confidence as a JSON number under `rounding`.
pub fn number(p: f32, rounding: Rounding) -> Value {
    match rounding {
        Rounding::Exact => f32_json(p),
        Rounding::Hundredths => round2(p),
    }
}

/// Jev's confidence clamped at 0 (see the module notes).
pub fn answer_confidence(p_max: f32, n: usize) -> f32 {
    jev_confidence(p_max, n).max(0.0)
}

/// A choice answer: `options` = (option id, probability) in request order.
pub fn choice_answer(
    options: &[(&str, f32)],
    choice: &str,
    confidence: f32,
    rounding: Rounding,
) -> Value {
    let mut p = Map::new();
    for (id, prob) in options {
        p.insert((*id).to_string(), number(*prob, rounding));
    }
    json!({
        "type": "choice",
        "choice": choice,
        "probabilities": Value::Object(p),
        "confidence": number(confidence, rounding),
    })
}

/// Give a choice verdict without a distribution (the oracle's or the cache's)
/// the one-hot one, for the System One surface (0.8.7): every option of `q` in
/// request order, the chosen one 1 and the others 0, confidence 1 — the
/// verdict as Jev writes a certain answer. Clients validating Jev's schema
/// (the Decision Index kit) otherwise reject every oracle answer. A local
/// answer (it has `probabilities`) and any other answer are left as they are.
pub fn complete_choice(answer: &mut Value, q: &Question) {
    if answer.get("type").and_then(Value::as_str) != Some("choice")
        || answer.get("probabilities").is_some()
    {
        return;
    }
    let Some(choice) = answer.get("choice").and_then(Value::as_str) else {
        return;
    };
    let options = q.options();
    if !options.contains(&choice) {
        return;
    }
    let one_hot: Vec<(&str, f32)> = options
        .iter()
        .map(|&o| (o, if o == choice { 1.0 } else { 0.0 }))
        .collect();
    let choice = choice.to_string();
    *answer = choice_answer(&one_hot, &choice, 1.0, Rounding::Hundredths);
}

/// A verdict of the oracle (or of the cache of its answers).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OracleAnswer {
    Choice(String),
    /// Level index, from 0.
    Score(u32),
    Noul(bool),
}

impl OracleAnswer {
    pub fn kind(&self) -> QuestionKind {
        match self {
            OracleAnswer::Choice(_) => QuestionKind::Choice,
            OracleAnswer::Score(_) => QuestionKind::Score,
            OracleAnswer::Noul(_) => QuestionKind::Noul,
        }
    }

    /// The verdict fits the question: same type, an option of the request, a
    /// level below the number of levels.
    pub fn check(&self, q: &Question) -> Result<(), String> {
        if self.kind() != q.kind {
            return Err(format!(
                "a {} verdict for a {} question",
                self.kind().as_str(),
                q.kind.as_str()
            ));
        }
        match self {
            OracleAnswer::Choice(c) => {
                if q.options().contains(&c.as_str()) {
                    Ok(())
                } else {
                    Err(format!("'{c}' is not an option of question '{}'", q.id))
                }
            }
            OracleAnswer::Score(s) => {
                if (*s as usize) < q.levels().len() {
                    Ok(())
                } else {
                    Err(format!(
                        "level {s} is outside 0..{} of question '{}'",
                        q.levels().len(),
                        q.id
                    ))
                }
            }
            OracleAnswer::Noul(_) => Ok(()),
        }
    }

    /// The answer object (see the module notes).
    pub fn to_answer(&self, q: &Question) -> Value {
        match self {
            OracleAnswer::Choice(c) => json!({"type": "choice", "choice": c}),
            OracleAnswer::Score(s) => {
                let legend: Map<String, Value> = q
                    .levels()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect();
                json!({"type": "score", "score": s, "legend": Value::Object(legend)})
            }
            OracleAnswer::Noul(b) => json!({
                "type": "noul",
                "noul": u8::from(*b),
                "value_semantics": NOUL_SEMANTICS,
            }),
        }
    }

    /// The label the verdict names (choice only).
    pub fn label(&self) -> Option<&str> {
        match self {
            OracleAnswer::Choice(c) => Some(c),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round2_matches_jev_forms() {
        assert_eq!(round2(0.0).to_string(), "0");
        assert_eq!(round2(0.004).to_string(), "0");
        assert_eq!(round2(0.97).to_string(), "0.97");
        assert_eq!(round2(0.995).to_string(), "1");
        assert_eq!(round2(1.0).to_string(), "1");
        assert_eq!(round2(0.015).to_string(), "0.01"); // f32 0.015 is 0.01499999…
        assert_eq!(round2(0.125).to_string(), "0.13"); // exact half: away from zero
        assert_eq!(round2(0.5).to_string(), "0.5");
        assert_eq!(number(0.97, Rounding::Exact).to_string(), "0.97");
    }

    #[test]
    fn a_choice_verdict_gets_the_one_hot_distribution() {
        let q = Question {
            id: "q".into(),
            kind: QuestionKind::Choice,
            instructions: json!("pick"),
            criteria: Some(json!({"b": "B", "a": "A", "c": "C"})),
        };
        let mut a = OracleAnswer::Choice("a".into()).to_answer(&q);
        complete_choice(&mut a, &q);
        assert_eq!(
            a.to_string(),
            r#"{"type":"choice","choice":"a","probabilities":{"b":0,"a":1,"c":0},"confidence":1}"#
        );
        // A local answer keeps its distribution; a noul verdict is untouched.
        let p = [("b", 0.25), ("a", 0.5), ("c", 0.25)];
        let local = choice_answer(&p, "a", 0.25, Rounding::Exact);
        let mut l = local.clone();
        complete_choice(&mut l, &q);
        assert_eq!(l, local);
        let mut n = json!({"type": "noul", "noul": 1});
        complete_choice(&mut n, &q);
        assert_eq!(n, json!({"type": "noul", "noul": 1}));
    }

    #[test]
    fn confidence_is_jev_formula_clamped() {
        assert_eq!(answer_confidence(1.0, 77), 1.0);
        assert_eq!(answer_confidence(0.5, 2), 0.0);
        let p = 0.9f32;
        assert_eq!(answer_confidence(p, 3), (3.0 * p - 1.0) / 2.0);
        assert_eq!(answer_confidence(0.3, 3), 0.0);
        assert_eq!(answer_confidence(0.7, 1), 0.7);
    }
}
