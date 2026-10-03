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
//! * **Oracle and cache answers** ([`Verdict`], 0.8.8, DESIGN C3) carry the
//!   oracle's distribution: choice `{type, choice, probabilities, confidence}`
//!   (every option in request order, confidence = p(choice)); score `{type,
//!   score: level, legend: {"0": level 0, …}, probabilities: {"0": p, …},
//!   confidence}`; noul `{type, noul: 1|0, value_semantics:
//!   "boolean_verdict_not_probability", probability: p(true)}` on
//!   `/v1/decisions` (the documented verdict, the probability beside it) and
//!   Jev's `{type, noul: p(true)}` on `/v1/systemone` ([`systemone_answer`]).
//!   A verdict without a distribution (the oracle gave none, a cache entry of
//!   0.8.7) is one-hot: the verdict 1, the rest 0, confidence 1, written as
//!   JSON integers. Before 0.8.8 a native oracle answer was `{type, choice}` /
//!   `{type, score, legend}` (the distribution is an additive change); on
//!   `/v1/systemone` a choice verdict was given the one-hot distribution
//!   ([`complete_choice`]): Jev's schema, and the Decision Index validator,
//!   require a probability for every option.
//! * **Normalization** of the oracle's listed probabilities
//!   ([`Verdict::normalize`]): the listed options keep their mass
//!   (renormalized to 1 when it is above 1), the rest is spread uniformly over
//!   the unlisted ones (all listed and below 1: renormalized); the verdict is
//!   the argmax, a tie going to the oracle's stated verdict; nothing listed is
//!   the one-hot verdict. A noul's probability of true decides it (> 0.5 true,
//!   < 0.5 false, 0.5 the stated verdict).

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
/// the one-hot one, for the System One surface (0.8.8): every option of `q` in
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

/// Jev's System One form of an answer (DESIGN C3): a choice verdict without
/// a distribution gets the one-hot one ([`complete_choice`]), a noul answer
/// becomes `{type, noul: p(true)}` (its `probability`); anything else, a
/// local answer included, is left as it is.
pub fn systemone_answer(answer: &mut Value, q: &Question) {
    complete_choice(answer, q);
    if answer.get("type").and_then(Value::as_str) == Some("noul")
        && let Some(p) = answer.get("probability").cloned()
    {
        *answer = json!({"type": "noul", "noul": p});
    }
}

/// The key of a noul distribution's one entry, p(true).
pub const NOUL_TRUE: &str = "true";

/// An oracle verdict with its distribution (0.8.8, DESIGN C3).
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub answer: OracleAnswer,
    /// The normalized distribution: choice — (option id, p) of every option;
    /// score — (level index, p) of every level; noul — (`"true"`, p(true)).
    /// Empty: the verdict alone, read as one-hot.
    pub probabilities: Vec<(String, f32)>,
}

impl From<OracleAnswer> for Verdict {
    fn from(answer: OracleAnswer) -> Self {
        Self::one_hot(answer)
    }
}

impl Verdict {
    /// The verdict alone (no distribution).
    pub fn one_hot(answer: OracleAnswer) -> Self {
        Self {
            answer,
            probabilities: Vec::new(),
        }
    }

    pub fn is_one_hot(&self) -> bool {
        self.probabilities.is_empty()
    }

    pub fn kind(&self) -> QuestionKind {
        self.answer.kind()
    }

    /// The label the verdict names (choice only).
    pub fn label(&self) -> Option<&str> {
        self.answer.label()
    }

    /// The probability of `key` (an option id, a level index, `"true"`);
    /// one-hot without a distribution.
    pub fn probability(&self, key: &str) -> f32 {
        if self.probabilities.is_empty() {
            let hit = match &self.answer {
                OracleAnswer::Choice(c) => c == key,
                OracleAnswer::Score(s) => s.to_string() == key,
                OracleAnswer::Noul(b) => *b && key == NOUL_TRUE,
            };
            return if hit { 1.0 } else { 0.0 };
        }
        self.probabilities
            .iter()
            .find(|(k, _)| k == key)
            .map_or(0.0, |(_, p)| *p)
    }

    /// The verdict fits the question ([`OracleAnswer::check`]) and its
    /// distribution names only the question's options or levels.
    pub fn check(&self, q: &Question) -> Result<(), String> {
        self.answer.check(q)?;
        let keys = distribution_keys(q);
        if self.probabilities.iter().any(|(k, _)| !keys.contains(k)) {
            return Err(format!(
                "a probability outside the options of question '{}'",
                q.id
            ));
        }
        Ok(())
    }

    /// The verdict from the oracle's stated verdict and the probabilities it
    /// listed (`listed`: (option id | level index | `"true"`, p), each key of
    /// the question, finite, in [0, 1], at most once — the parser checks
    /// them); see the module notes.
    pub fn normalize(q: &Question, stated: OracleAnswer, listed: &[(String, f64)]) -> Self {
        if listed.is_empty() {
            return Self::one_hot(stated);
        }
        if let OracleAnswer::Noul(b) = stated {
            let p = listed[0].1.clamp(0.0, 1.0);
            let verdict = if p > 0.5 {
                true
            } else if p < 0.5 {
                false
            } else {
                b
            };
            return Self {
                answer: OracleAnswer::Noul(verdict),
                probabilities: vec![(NOUL_TRUE.to_string(), p as f32)],
            };
        }
        let keys = distribution_keys(q);
        let stated_key = match &stated {
            OracleAnswer::Choice(c) => c.clone(),
            OracleAnswer::Score(s) => s.to_string(),
            OracleAnswer::Noul(_) => unreachable!("handled above"),
        };
        let mass: f64 = listed.iter().map(|(_, p)| p).sum();
        let unlisted = keys
            .iter()
            .filter(|k| !listed.iter().any(|(l, _)| l == *k))
            .count();
        let (scale, rest) = if mass > 1.0 || (unlisted == 0 && mass > 0.0) {
            (1.0 / mass, 0.0)
        } else if unlisted > 0 {
            (1.0, (1.0 - mass) / unlisted as f64)
        } else {
            // Every option listed at 0: nothing to go by.
            return Self::one_hot(stated);
        };
        let dist: Vec<(String, f64)> = keys
            .iter()
            .map(|k| {
                let p = listed
                    .iter()
                    .find(|(l, _)| l == k)
                    .map_or(rest, |(_, p)| p * scale);
                (k.clone(), p)
            })
            .collect();
        let best = dist.iter().map(|(_, p)| *p).fold(f64::MIN, f64::max);
        let winner = if dist.iter().any(|(k, p)| *k == stated_key && *p == best) {
            stated_key
        } else {
            dist.iter()
                .find(|(_, p)| *p == best)
                .map(|(k, _)| k.clone())
                .unwrap_or(stated_key)
        };
        let answer = match stated {
            OracleAnswer::Choice(_) => OracleAnswer::Choice(winner),
            OracleAnswer::Score(s) => OracleAnswer::Score(winner.parse().unwrap_or(s)),
            OracleAnswer::Noul(_) => unreachable!("handled above"),
        };
        Self {
            answer,
            probabilities: dist.into_iter().map(|(k, p)| (k, p as f32)).collect(),
        }
    }

    /// The answer object on `/v1/decisions` (see the module notes); a
    /// one-hot verdict is written with integers whatever `rounding`.
    pub fn to_answer(&self, q: &Question, rounding: Rounding) -> Value {
        let rounding = if self.is_one_hot() {
            Rounding::Hundredths
        } else {
            rounding
        };
        match &self.answer {
            OracleAnswer::Choice(c) => {
                let probs: Vec<(&str, f32)> = q
                    .options()
                    .into_iter()
                    .map(|o| (o, self.probability(o)))
                    .collect();
                choice_answer(&probs, c, self.probability(c), rounding)
            }
            OracleAnswer::Score(s) => {
                let mut v = self.answer.to_answer(q);
                let probs: Map<String, Value> = (0..q.levels().len())
                    .map(|i| {
                        let k = i.to_string();
                        let p = number(self.probability(&k), rounding);
                        (k, p)
                    })
                    .collect();
                v["probabilities"] = Value::Object(probs);
                v["confidence"] = number(self.probability(&s.to_string()), rounding);
                v
            }
            OracleAnswer::Noul(_) => {
                let mut v = self.answer.to_answer(q);
                v["probability"] = number(self.probability(NOUL_TRUE), rounding);
                v
            }
        }
    }
}

/// The keys of a question's distribution: its option ids, its level indices
/// or `"true"`.
fn distribution_keys(q: &Question) -> Vec<String> {
    match q.kind {
        QuestionKind::Choice => q.options().into_iter().map(str::to_string).collect(),
        QuestionKind::Score => (0..q.levels().len()).map(|i| i.to_string()).collect(),
        QuestionKind::Noul => vec![NOUL_TRUE.to_string()],
    }
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

    fn choice_q(ids: &[&str]) -> Question {
        let c: Map<String, Value> = ids.iter().map(|i| (i.to_string(), json!(i))).collect();
        Question {
            id: "q".into(),
            kind: QuestionKind::Choice,
            instructions: json!("pick"),
            criteria: Some(Value::Object(c)),
        }
    }

    fn listed(v: &[(&str, f64)]) -> Vec<(String, f64)> {
        v.iter().map(|(k, p)| (k.to_string(), *p)).collect()
    }

    fn dist(v: &Verdict) -> Vec<(String, f32)> {
        v.probabilities.clone()
    }

    /// DESIGN C3 normalization: listed mass kept (renormalized above 1), the
    /// rest spread over the unlisted, all listed below 1 renormalized, the
    /// argmax with ties to the stated verdict, nothing listed one-hot.
    #[test]
    fn oracle_distributions_are_normalized() {
        let q = choice_q(&["a", "b", "c", "d", "e"]);
        let c = |s: &str| OracleAnswer::Choice(s.into());
        // Listed 0.7 + 0.1, the rest 0.2 over three.
        let v = Verdict::normalize(&q, c("a"), &listed(&[("a", 0.7), ("b", 0.1)]));
        assert_eq!(v.answer, c("a"));
        let d = dist(&v);
        assert_eq!(
            d.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c", "d", "e"]
        );
        assert!((d[2].1 - 0.2 / 3.0).abs() < 1e-6, "{d:?}");
        assert!((d.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-6);
        // Above 1: renormalized, the unlisted 0.
        let v = Verdict::normalize(&q, c("b"), &listed(&[("a", 0.9), ("b", 0.6)]));
        assert_eq!(v.answer, c("a"), "the argmax, not the stated verdict");
        assert!((v.probability("a") - 0.6).abs() < 1e-6 && v.probability("c") == 0.0);
        // A tie goes to the stated verdict; a tie without it to the first.
        let v = Verdict::normalize(&q, c("b"), &listed(&[("a", 0.5), ("b", 0.5)]));
        assert_eq!(v.answer, c("b"));
        let v = Verdict::normalize(&q, c("e"), &listed(&[("c", 0.4), ("b", 0.4)]));
        assert_eq!(v.answer, c("b"));
        // All listed below 1: renormalized; all at 0: one-hot.
        let two = choice_q(&["x", "y"]);
        let v = Verdict::normalize(&two, c("x"), &listed(&[("x", 0.3), ("y", 0.1)]));
        assert!((v.probability("x") - 0.75).abs() < 1e-6);
        let v = Verdict::normalize(&two, c("y"), &listed(&[("x", 0.0), ("y", 0.0)]));
        assert!(v.is_one_hot() && v.answer == c("y"));
        // Nothing listed: one-hot.
        let v = Verdict::normalize(&q, c("d"), &[]);
        assert!(v.is_one_hot() && v.probability("d") == 1.0 && v.probability("a") == 0.0);
        // Score: levels by index; noul: p(true) decides.
        let score = Question {
            id: "s".into(),
            kind: QuestionKind::Score,
            instructions: json!("how"),
            criteria: Some(json!(["lo", "mid", "hi"])),
        };
        let v = Verdict::normalize(
            &score,
            OracleAnswer::Score(0),
            &listed(&[("0", 0.2), ("1", 0.7)]),
        );
        assert_eq!(v.answer, OracleAnswer::Score(1));
        assert!((v.probability("2") - 0.1).abs() < 1e-6);
        let noul = Question {
            id: "n".into(),
            kind: QuestionKind::Noul,
            instructions: json!("is it?"),
            criteria: None,
        };
        let n = |b: bool, p: f64| {
            Verdict::normalize(&noul, OracleAnswer::Noul(b), &listed(&[(NOUL_TRUE, p)]))
        };
        assert_eq!(n(true, 0.2).answer, OracleAnswer::Noul(false));
        assert_eq!(n(false, 0.9).answer, OracleAnswer::Noul(true));
        assert_eq!(n(false, 0.5).answer, OracleAnswer::Noul(false));
        // The answers: native and System One.
        let v = Verdict::normalize(&two, c("x"), &listed(&[("x", 0.75), ("y", 0.25)]));
        assert_eq!(
            v.to_answer(&two, Rounding::Exact).to_string(),
            r#"{"type":"choice","choice":"x","probabilities":{"x":0.75,"y":0.25},"confidence":0.75}"#
        );
        let one = Verdict::one_hot(c("y")).to_answer(&two, Rounding::Exact);
        assert_eq!(
            one.to_string(),
            r#"{"type":"choice","choice":"y","probabilities":{"x":0,"y":1},"confidence":1}"#
        );
        let mut a = n(true, 0.875).to_answer(&noul, Rounding::Exact);
        assert_eq!(
            a.to_string(),
            r#"{"type":"noul","noul":1,"value_semantics":"boolean_verdict_not_probability","probability":0.875}"#
        );
        systemone_answer(&mut a, &noul);
        assert_eq!(a.to_string(), r#"{"type":"noul","noul":0.875}"#);
        let a = v_score(&score);
        assert_eq!(
            a.to_string(),
            r#"{"type":"score","score":1,"legend":{"0":"lo","1":"mid","2":"hi"},"probabilities":{"0":0.2,"1":0.7,"2":0.1},"confidence":0.7}"#
        );
        // A distribution must name the question's options.
        let mut bad = v.clone();
        bad.probabilities.push(("zz".into(), 0.0));
        assert!(bad.check(&two).is_err() && v.check(&two).is_ok());
    }

    fn v_score(score: &Question) -> Value {
        Verdict::normalize(
            score,
            OracleAnswer::Score(1),
            &[("0".into(), 0.2), ("1".into(), 0.7), ("2".into(), 0.1)],
        )
        .to_answer(score, Rounding::Hundredths)
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
