//! Skill matching of a question: exact, subset, superset, untrained (spec §4.5).
//!
//! L is the set of option ids of a choice question; A(s) the active labels of
//! skill s (its scored tasks). In order:
//!
//! 1. `cmf.skill` names the skill (an unknown id → 400); the rules below are
//!    then applied to that skill alone;
//! 2. **exact**: L = A(s) for exactly one skill → decided by the skill's gate;
//! 3. **subset**: L ⊊ A(s) for exactly one skill, |L| ≥ 2 → decided over L
//!    only (argmin, softmax, margin and novelty over L; the same T, θ, τ),
//!    never certified;
//! 4. **superset**: A(s) ⊊ L for exactly one skill (the rest of L unknown) →
//!    not decided locally; an oracle answer teaches that skill;
//! 5. **untrained**: everything else — no skill fits, several skills fit
//!    (ambiguous: name one with `cmf.skill`), score and noul questions, a
//!    skill without active tasks.
//!
//! Only option ids are compared; instructions and criteria descriptions matter
//! to the oracle alone.

use crate::protocol::{ApiError, Question, QuestionKind};
use serde_json::json;
use std::collections::HashSet;

/// The labels of one skill, as the matcher sees them.
#[derive(Clone, Copy, Debug)]
pub struct SkillLabels<'a> {
    pub id: &'a str,
    /// Active labels in candidate (task) order.
    pub active: &'a [String],
}

/// How a question relates to the skills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatchKind {
    Exact,
    Subset,
    Superset,
    Untrained,
}

impl MatchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchKind::Exact => "exact",
            MatchKind::Subset => "subset",
            MatchKind::Superset => "superset",
            MatchKind::Untrained => "untrained",
        }
    }

    /// Decided by the local model (exact or subset).
    pub fn is_local(self) -> bool {
        matches!(self, MatchKind::Exact | MatchKind::Subset)
    }
}

/// The match of one question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillMatch {
    pub kind: MatchKind,
    /// The skill (exact, subset, superset).
    pub skill: Option<String>,
    /// Exact and subset: the candidates (indices into the skill's active labels)
    /// the question is decided over, in candidate order.
    pub candidates: Vec<usize>,
    /// Superset: the options that are not active labels of the skill, in
    /// request order.
    pub unknown: Vec<String>,
    /// Untrained: why (for the 422 details).
    pub reason: Option<String>,
}

impl SkillMatch {
    fn untrained(reason: impl Into<String>) -> Self {
        Self {
            kind: MatchKind::Untrained,
            skill: None,
            candidates: Vec::new(),
            unknown: Vec::new(),
            reason: Some(reason.into()),
        }
    }
}

/// Relation of L to one skill.
fn relate(options: &[&str], set: &HashSet<&str>, s: &SkillLabels<'_>) -> Option<SkillMatch> {
    if s.active.is_empty() {
        return None;
    }
    let inside: Vec<usize> = s
        .active
        .iter()
        .enumerate()
        .filter(|(_, l)| set.contains(l.as_str()))
        .map(|(i, _)| i)
        .collect();
    let all_known = inside.len() == options.len();
    let covers = inside.len() == s.active.len();
    let kind = match (all_known, covers) {
        (true, true) => MatchKind::Exact,
        (true, false) if options.len() >= 2 => MatchKind::Subset,
        (false, true) => MatchKind::Superset,
        _ => return None,
    };
    let unknown = if kind == MatchKind::Superset {
        let active: HashSet<&str> = s.active.iter().map(String::as_str).collect();
        options
            .iter()
            .filter(|o| !active.contains(*o))
            .map(|o| o.to_string())
            .collect()
    } else {
        Vec::new()
    };
    Some(SkillMatch {
        kind,
        skill: Some(s.id.to_string()),
        candidates: if kind.is_local() { inside } else { Vec::new() },
        unknown,
        reason: None,
    })
}

/// Match option ids against the skills (rules 2–5, see the module notes).
pub fn match_labels(skills: &[SkillLabels<'_>], options: &[&str]) -> SkillMatch {
    let set: HashSet<&str> = options.iter().copied().collect();
    let found: Vec<SkillMatch> = skills
        .iter()
        .filter_map(|s| relate(options, &set, s))
        .collect();
    for kind in [MatchKind::Exact, MatchKind::Subset, MatchKind::Superset] {
        let hits: Vec<&SkillMatch> = found.iter().filter(|m| m.kind == kind).collect();
        match hits.as_slice() {
            [] => continue,
            [one] => return (*one).clone(),
            many => {
                let ids: Vec<&str> = many.iter().filter_map(|m| m.skill.as_deref()).collect();
                return SkillMatch::untrained(format!(
                    "ambiguous: the options are an {} match of skills {}; name one with cmf.skill",
                    kind.as_str(),
                    ids.join(", ")
                ));
            }
        }
    }
    SkillMatch::untrained("no skill is trained on these options")
}

/// Match a question (rule 1 with `forced`, then [`match_labels`]).
pub fn match_question(
    skills: &[SkillLabels<'_>],
    q: &Question,
    forced: Option<&str>,
) -> Result<SkillMatch, ApiError> {
    let pool: Vec<SkillLabels<'_>> = match forced {
        Some(id) => {
            let Some(s) = skills.iter().find(|s| s.id == id) else {
                let ids: Vec<&str> = skills.iter().map(|s| s.id).collect();
                return Err(ApiError::invalid_field(
                    "cmf.skill",
                    format!("unknown skill '{id}' (skills: {})", ids.join(", ")),
                )
                .with_detail("skills", json!(ids)));
            };
            vec![*s]
        }
        None => skills.to_vec(),
    };
    if q.kind != QuestionKind::Choice {
        return Ok(SkillMatch::untrained(format!(
            "{} questions are not trained locally",
            q.kind.as_str()
        )));
    }
    let options = q.options();
    let m = match_labels(&pool, &options);
    if m.kind == MatchKind::Untrained
        && let Some(id) = forced
    {
        return Ok(SkillMatch::untrained(format!(
            "the options do not match skill '{id}' (neither its labels, a subset nor a superset of them)"
        )));
    }
    Ok(m)
}

/// The skill of `cortiq decide --labels a,b,c`: the skill the labels match as
/// exact or subset (spec §4.1).
pub fn skill_for_labels(skills: &[SkillLabels<'_>], labels: &[&str]) -> anyhow::Result<String> {
    let m = match_labels(skills, labels);
    match (m.kind, m.skill) {
        (MatchKind::Exact | MatchKind::Subset, Some(s)) => Ok(s),
        (MatchKind::Superset, Some(s)) => anyhow::bail!(
            "labels {} are not trained in skill '{s}'",
            m.unknown.join(", ")
        ),
        _ => anyhow::bail!(
            "{}",
            m.reason
                .unwrap_or_else(|| "no skill matches these labels".into())
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn exact_subset_superset_untrained() {
        let a = labels(&["a", "b", "c"]);
        let d = labels(&["d", "e"]);
        let skills = [
            SkillLabels {
                id: "s1",
                active: &a,
            },
            SkillLabels {
                id: "s2",
                active: &d,
            },
        ];
        let m = match_labels(&skills, &["c", "a", "b"]);
        assert_eq!(m.kind, MatchKind::Exact);
        assert_eq!(m.skill.as_deref(), Some("s1"));
        assert_eq!(m.candidates, vec![0, 1, 2]);
        let m = match_labels(&skills, &["c", "a"]);
        assert_eq!(m.kind, MatchKind::Subset);
        assert_eq!(m.candidates, vec![0, 2]);
        let m = match_labels(&skills, &["e", "x", "d"]);
        assert_eq!(m.kind, MatchKind::Superset);
        assert_eq!(m.skill.as_deref(), Some("s2"));
        assert_eq!(m.unknown, vec!["x"]);
        assert_eq!(
            match_labels(&skills, &["a", "d"]).kind,
            MatchKind::Untrained
        );
        assert_eq!(
            match_labels(&skills, &["x", "y"]).kind,
            MatchKind::Untrained
        );
    }

    #[test]
    fn ambiguity_is_untrained() {
        let a = labels(&["a", "b", "c"]);
        let b = labels(&["a", "b", "d"]);
        let skills = [
            SkillLabels {
                id: "s1",
                active: &a,
            },
            SkillLabels {
                id: "s2",
                active: &b,
            },
        ];
        let m = match_labels(&skills, &["a", "b"]);
        assert_eq!(m.kind, MatchKind::Untrained);
        assert!(m.reason.unwrap().contains("ambiguous"));
        // An exact match wins over a subset of another skill.
        let m = match_labels(&skills, &["a", "b", "c"]);
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("s1")));
    }
}
