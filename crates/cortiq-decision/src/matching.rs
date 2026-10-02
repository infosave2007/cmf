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
//!
//! **Auto-skills** (0.8.6, DESIGN D8/A6). The labels of an auto-skill that are
//! not active yet (quarantined, too few examples) still count as *known*: a
//! question whose ids are all known to it is exact or subset over its ACTIVE
//! labels (a quarantined option gets probability 0; a text of such a label is
//! expected to abstain on novelty and keeps teaching it). For a data skill
//! known = active, so nothing changes for it. When exactly one data skill and
//! any number of auto-skills hit the best kind, the data skill wins (no
//! ambiguity); several data skills, or auto-skills alone, stay ambiguous.

use crate::protocol::{ApiError, Question, QuestionKind};
use serde_json::json;
use std::collections::HashSet;

/// The labels of one skill, as the matcher sees them.
#[derive(Clone, Copy, Debug)]
pub struct SkillLabels<'a> {
    pub id: &'a str,
    /// Active labels in candidate (task) order.
    pub active: &'a [String],
    /// Every label the skill has, active or not (an auto-skill's whole
    /// contract); the active ones for a data skill.
    pub known: &'a [String],
    /// An auto-skill (loses a tie against a data skill).
    pub auto: bool,
}

impl<'a> SkillLabels<'a> {
    /// A data skill: known = active.
    pub fn data(id: &'a str, active: &'a [String]) -> Self {
        Self {
            id,
            active,
            known: active,
            auto: false,
        }
    }
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
    /// Untrained because several skills fit (the structural form of the
    /// reason: such a contract is not learned into an auto-skill, DESIGN A9).
    pub ambiguous: bool,
}

impl SkillMatch {
    fn untrained(reason: impl Into<String>) -> Self {
        Self {
            kind: MatchKind::Untrained,
            skill: None,
            candidates: Vec::new(),
            unknown: Vec::new(),
            reason: Some(reason.into()),
            ambiguous: false,
        }
    }

    /// Learnable into an auto-skill: untrained because no skill fits (not
    /// ambiguous, not a forced mismatch — the caller checks `cmf.skill`).
    pub fn is_foreign(&self) -> bool {
        self.kind == MatchKind::Untrained && self.skill.is_none() && !self.ambiguous
    }
}

/// Relation of L to one skill: the candidates are the active labels in L,
/// "known" covers the quarantined labels of an auto-skill too.
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
    let known: HashSet<&str> = s.known.iter().map(String::as_str).collect();
    let all_known = options.iter().all(|o| known.contains(o));
    let covers = inside.len() == s.active.len();
    let kind = match (all_known, covers) {
        (true, true) => MatchKind::Exact,
        // Known but none active (every option quarantined): not decidable.
        (true, false) if options.len() >= 2 && !inside.is_empty() => MatchKind::Subset,
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
        ambiguous: false,
    })
}

/// Match option ids against the skills (rules 2–5, see the module notes).
pub fn match_labels(skills: &[SkillLabels<'_>], options: &[&str]) -> SkillMatch {
    let set: HashSet<&str> = options.iter().copied().collect();
    let found: Vec<(SkillMatch, bool)> = skills
        .iter()
        .filter_map(|s| relate(options, &set, s).map(|m| (m, s.auto)))
        .collect();
    for kind in [MatchKind::Exact, MatchKind::Subset, MatchKind::Superset] {
        let hits: Vec<&(SkillMatch, bool)> = found.iter().filter(|m| m.0.kind == kind).collect();
        match hits.as_slice() {
            [] => continue,
            [one] => return one.0.clone(),
            many => {
                // One data skill among auto-skills wins the tie (DESIGN D8).
                let data: Vec<&SkillMatch> = many.iter().filter(|m| !m.1).map(|m| &m.0).collect();
                if let [one] = data.as_slice() {
                    return (*one).clone();
                }
                let ids: Vec<&str> = many.iter().filter_map(|m| m.0.skill.as_deref()).collect();
                let mut m = SkillMatch::untrained(format!(
                    "ambiguous: the options are an {} match of skills {}; name one with cmf.skill",
                    kind.as_str(),
                    ids.join(", ")
                ));
                m.ambiguous = true;
                return m;
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
        let skills = [SkillLabels::data("s1", &a), SkillLabels::data("s2", &d)];
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
        let skills = [SkillLabels::data("s1", &a), SkillLabels::data("s2", &b)];
        let m = match_labels(&skills, &["a", "b"]);
        assert_eq!(m.kind, MatchKind::Untrained);
        assert!(m.ambiguous && !m.is_foreign());
        assert!(m.reason.unwrap().contains("ambiguous"));
        assert!(match_labels(&skills, &["x", "y"]).is_foreign());
        // An exact match wins over a subset of another skill.
        let m = match_labels(&skills, &["a", "b", "c"]);
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("s1")));
    }

    #[test]
    fn an_auto_skill_relates_over_its_known_labels_and_decides_over_the_active_ones() {
        let active = labels(&["food", "travel"]);
        let known = labels(&["cruise", "food", "travel"]);
        let auto = SkillLabels {
            id: "auto-1",
            active: &active,
            known: &known,
            auto: true,
        };
        // The whole contract: exact over the two active candidates.
        let m = match_labels(&[auto], &["food", "travel", "cruise"]);
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Exact, Some("auto-1"))
        );
        assert_eq!(m.candidates, vec![0, 1]);
        // Two known ids, one active: a subset over that candidate.
        let m = match_labels(&[auto], &["cruise", "travel"]);
        assert_eq!((m.kind, m.candidates.clone()), (MatchKind::Subset, vec![1]));
        // Only quarantined ids: not decidable, foreign (a new contract).
        let other = labels(&["cruise", "ship"]);
        let ship = labels(&["ship"]);
        let a2 = SkillLabels {
            id: "auto-2",
            active: &ship,
            known: &other,
            auto: true,
        };
        assert!(match_labels(&[auto], &["cruise", "food"]).kind == MatchKind::Subset);
        let known3 = labels(&["cruise", "food", "travel", "zoo"]);
        let a3 = SkillLabels {
            id: "auto-3",
            active: &active,
            known: &known3,
            auto: true,
        };
        assert!(match_labels(&[a3], &["cruise", "zoo"]).is_foreign());
        // An unknown id with every active one: superset, the quarantined ids unknown too.
        let m = match_labels(&[auto], &["food", "travel", "cruise", "x"]);
        assert_eq!(m.kind, MatchKind::Superset);
        assert_eq!(m.unknown, vec!["cruise", "x"]);
        // A fully quarantined auto-skill is invisible.
        let none = SkillLabels {
            id: "auto-0",
            active: &[],
            known: &known,
            auto: true,
        };
        assert!(match_labels(&[none, a2], &["food", "travel", "cruise"]).is_foreign());
    }

    #[test]
    fn a_data_skill_wins_a_tie_against_auto_skills_only() {
        let a = labels(&["a", "b", "c"]);
        let data = SkillLabels::data("s1", &a);
        let auto1 = SkillLabels {
            id: "auto-1",
            active: &a,
            known: &a,
            auto: true,
        };
        let auto2 = SkillLabels {
            id: "auto-2",
            active: &a,
            known: &a,
            auto: true,
        };
        let m = match_labels(&[auto1, data, auto2], &["a", "b", "c"]);
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("s1")));
        let m = match_labels(&[auto1, data], &["a", "b"]);
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Subset, Some("s1"))
        );
        // Auto-skills alone: ambiguous, and not learnable.
        let m = match_labels(&[auto1, auto2], &["a", "b", "c"]);
        assert!(m.kind == MatchKind::Untrained && m.ambiguous && !m.is_foreign());
        // Two data skills: ambiguous as before, whatever the auto-skills.
        let data2 = SkillLabels::data("s2", &a);
        let m = match_labels(&[data, auto1, data2], &["a", "b", "c"]);
        assert!(m.ambiguous);
        assert!(m.reason.unwrap().contains("s1, auto-1, s2"));
    }
}
