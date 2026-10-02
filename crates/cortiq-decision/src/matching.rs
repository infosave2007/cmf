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
//! For a data skill only option ids are compared; instructions and criteria
//! descriptions matter to the oracle alone.
//!
//! **Auto-skills** (0.8.6, DESIGN A18/A6). An auto-skill is one exact
//! contract — the sha256 of `{type, instructions, criteria}` with the
//! descriptions ([`crate::manifest::contract_sha256`]) — and is matched by
//! that sha alone: a choice question whose contract sha equals the skill's is
//! **exact** over the skill's ACTIVE labels; subset and superset relations are
//! never computed against an auto-skill (a client that drops or adds a label,
//! changes a description or the instructions has another contract, learned
//! separately), so two clients using the same ids under different questions
//! never share a skill. The labels of an auto-skill that are not active yet
//! (quarantined, too few examples) count as *known* within the exact match: a
//! quarantined option gets probability 0, and a text of such a label is
//! expected to abstain on novelty and keep teaching it. When exactly one data
//! skill and any number of auto-skills hit the best kind, the data skill wins
//! (no ambiguity); several data skills, or auto-skills alone, stay ambiguous.

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
    /// An auto-skill: its contract sha (the question's must equal it; it
    /// loses a tie against a data skill). `None` for a data skill.
    pub contract: Option<&'a str>,
}

impl<'a> SkillLabels<'a> {
    /// A data skill: known = active, matched by its ids.
    pub fn data(id: &'a str, active: &'a [String]) -> Self {
        Self {
            id,
            active,
            known: active,
            contract: None,
        }
    }

    /// An auto-skill (DESIGN A18): matched only by `contract`, decided over
    /// `active`, every label of `known` a known one.
    pub fn auto(id: &'a str, active: &'a [String], known: &'a [String], contract: &'a str) -> Self {
        Self {
            id,
            active,
            known,
            contract: Some(contract),
        }
    }

    pub fn is_auto(&self) -> bool {
        self.contract.is_some()
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

/// Relation of L to one skill: the candidates are the active labels in L.
/// An auto-skill relates only to its own contract (`contract` is the
/// question's sha), as exact over its active labels — the quarantined ones
/// are known within it (DESIGN A18/A6) — or not at all.
fn relate(
    options: &[&str],
    set: &HashSet<&str>,
    contract: Option<&str>,
    s: &SkillLabels<'_>,
) -> Option<SkillMatch> {
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
    let kind = if let Some(own) = s.contract {
        // Equal shas mean equal criteria keys: L is the whole contract, the
        // active labels are all inside it.
        if contract != Some(own) {
            return None;
        }
        debug_assert!(inside.len() == s.active.len());
        MatchKind::Exact
    } else {
        let known: HashSet<&str> = s.known.iter().map(String::as_str).collect();
        let all_known = options.iter().all(|o| known.contains(o));
        let covers = inside.len() == s.active.len();
        match (all_known, covers) {
            (true, true) => MatchKind::Exact,
            (true, false) if options.len() >= 2 && !inside.is_empty() => MatchKind::Subset,
            (false, true) => MatchKind::Superset,
            _ => return None,
        }
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

/// Match option ids against the skills (rules 2–5, see the module notes);
/// `contract` is the question's contract sha, the only thing an auto-skill
/// is matched by (without it, `cortiq decide --labels`, auto-skills are
/// invisible).
pub fn match_labels(
    skills: &[SkillLabels<'_>],
    options: &[&str],
    contract: Option<&str>,
) -> SkillMatch {
    let set: HashSet<&str> = options.iter().copied().collect();
    let found: Vec<(SkillMatch, bool)> = skills
        .iter()
        .filter_map(|s| relate(options, &set, contract, s).map(|m| (m, s.is_auto())))
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
    let contract = q.contract_sha256();
    let m = match_labels(&pool, &options, contract.as_deref());
    if m.kind == MatchKind::Untrained
        && let Some(id) = forced
    {
        return Ok(SkillMatch::untrained(if pool[0].is_auto() {
            format!(
                "the question is not the contract of auto-skill '{id}' (its instructions and criteria)"
            )
        } else {
            format!(
                "the options do not match skill '{id}' (neither its labels, a subset nor a superset of them)"
            )
        }));
    }
    Ok(m)
}

/// The skill of `cortiq decide --labels a,b,c`: the data skill the labels
/// match as exact or subset (spec §4.1). An auto-skill is one exact contract
/// (instructions and criteria), which labels alone do not name: `--skill`
/// names it and its rubric supplies the contract.
pub fn skill_for_labels(skills: &[SkillLabels<'_>], labels: &[&str]) -> anyhow::Result<String> {
    let m = match_labels(skills, labels, None);
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
    use serde_json::Value;

    fn labels(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn exact_subset_superset_untrained() {
        let a = labels(&["a", "b", "c"]);
        let d = labels(&["d", "e"]);
        let skills = [SkillLabels::data("s1", &a), SkillLabels::data("s2", &d)];
        let m = match_labels(&skills, &["c", "a", "b"], None);
        assert_eq!(m.kind, MatchKind::Exact);
        assert_eq!(m.skill.as_deref(), Some("s1"));
        assert_eq!(m.candidates, vec![0, 1, 2]);
        let m = match_labels(&skills, &["c", "a"], None);
        assert_eq!(m.kind, MatchKind::Subset);
        assert_eq!(m.candidates, vec![0, 2]);
        let m = match_labels(&skills, &["e", "x", "d"], None);
        assert_eq!(m.kind, MatchKind::Superset);
        assert_eq!(m.skill.as_deref(), Some("s2"));
        assert_eq!(m.unknown, vec!["x"]);
        assert_eq!(
            match_labels(&skills, &["a", "d"], None).kind,
            MatchKind::Untrained
        );
        assert_eq!(
            match_labels(&skills, &["x", "y"], None).kind,
            MatchKind::Untrained
        );
    }

    #[test]
    fn ambiguity_is_untrained() {
        let a = labels(&["a", "b", "c"]);
        let b = labels(&["a", "b", "d"]);
        let skills = [SkillLabels::data("s1", &a), SkillLabels::data("s2", &b)];
        let m = match_labels(&skills, &["a", "b"], None);
        assert_eq!(m.kind, MatchKind::Untrained);
        assert!(m.ambiguous && !m.is_foreign());
        assert!(m.reason.unwrap().contains("ambiguous"));
        assert!(match_labels(&skills, &["x", "y"], None).is_foreign());
        // An exact match wins over a subset of another skill.
        let m = match_labels(&skills, &["a", "b", "c"], None);
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("s1")));
    }

    /// A choice question under `instructions` with a description per label.
    fn question(instructions: Value, labels: &[&str]) -> Question {
        let mut c = serde_json::Map::new();
        for l in labels {
            c.insert(l.to_string(), json!(format!("about {l}")));
        }
        Question {
            id: "task".into(),
            kind: QuestionKind::Choice,
            instructions,
            criteria: Some(Value::Object(c)),
        }
    }

    /// An auto-skill is matched by its contract sha alone (DESIGN A18): the
    /// same contract is exact over the active labels, the quarantined ones
    /// known within it; a subset, a superset, other instructions or another
    /// description is no relation at all — a different contract, foreign.
    #[test]
    fn an_auto_skill_is_one_exact_contract() {
        let active = labels(&["food", "travel"]);
        let known = labels(&["cruise", "food", "travel"]);
        let which = json!("Which topic?");
        let q = question(which.clone(), &["food", "travel", "cruise"]);
        let sha = q.contract_sha256().unwrap();
        let auto = SkillLabels::auto("auto-1", &active, &known, &sha);
        // The whole contract: exact over the two active candidates.
        let m = match_question(&[auto], &q, None).unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Exact, Some("auto-1"))
        );
        assert_eq!(m.candidates, vec![0, 1]);
        // The criteria in another order: the same contract.
        let q2 = question(which.clone(), &["cruise", "travel", "food"]);
        assert_eq!(q2.contract_sha256().unwrap(), sha);
        assert_eq!(
            match_question(&[auto], &q2, None).unwrap().kind,
            MatchKind::Exact
        );
        // A subset of the ids, a superset, other instructions, a changed
        // description: other contracts, never subset/superset, foreign.
        for other in [
            question(which.clone(), &["food", "travel"]),
            question(which.clone(), &["cruise", "travel"]),
            question(which.clone(), &["food", "travel", "cruise", "x"]),
            question(json!("Is it urgent?"), &["food", "travel", "cruise"]),
            question(Value::Null, &["food", "travel", "cruise"]),
            {
                let mut q = q.clone();
                q.criteria.as_mut().unwrap()["food"] = json!("meals");
                q
            },
        ] {
            assert_ne!(other.contract_sha256().unwrap(), sha);
            let m = match_question(&[auto], &other, None).unwrap();
            assert!(m.is_foreign(), "{m:?}");
        }
        // Forced with `cmf.skill`: untrained, the reason names the contract
        // (a forced question is never learned, the caller checks `cmf.skill`).
        let m = match_question(
            &[auto],
            &question(which.clone(), &["food", "travel"]),
            Some("auto-1"),
        )
        .unwrap();
        assert_eq!(m.kind, MatchKind::Untrained);
        assert!(
            m.reason
                .unwrap()
                .contains("contract of auto-skill 'auto-1'")
        );
        // Labels alone (`decide --labels`) never name an auto-skill.
        assert!(skill_for_labels(&[auto], &["food", "travel", "cruise"]).is_err());
        // A fully quarantined auto-skill is invisible: its contract is foreign
        // (and keeps teaching it).
        let none = SkillLabels::auto("auto-0", &[], &known, &sha);
        assert!(match_question(&[none], &q, None).unwrap().is_foreign());
        // Two auto-skills of the same ids under different instructions: each
        // answers its own question only.
        let urgent = question(json!("Is it urgent?"), &["food", "travel", "cruise"]);
        let sha_u = urgent.contract_sha256().unwrap();
        let auto_u = SkillLabels::auto("auto-2", &active, &known, &sha_u);
        let m = match_question(&[auto, auto_u], &q, None).unwrap();
        assert_eq!(m.skill.as_deref(), Some("auto-1"));
        let m = match_question(&[auto, auto_u], &urgent, None).unwrap();
        assert_eq!(m.skill.as_deref(), Some("auto-2"));
    }

    #[test]
    fn a_data_skill_wins_a_tie_against_auto_skills_only() {
        let a = labels(&["a", "b", "c"]);
        let data = SkillLabels::data("s1", &a);
        let q = question(json!("Which?"), &["a", "b", "c"]);
        let sha = q.contract_sha256().unwrap();
        let auto1 = SkillLabels::auto("auto-1", &a, &a, &sha);
        let auto2 = SkillLabels::auto("auto-2", &a, &a, &sha);
        let m = match_question(&[auto1, data, auto2], &q, None).unwrap();
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("s1")));
        // A part of the ids: the data skill's subset; no auto-skill relates.
        let m = match_question(
            &[auto1, data],
            &question(json!("Which?"), &["a", "b"]),
            None,
        )
        .unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Subset, Some("s1"))
        );
        // Auto-skills alone: ambiguous, and not learnable.
        let m = match_question(&[auto1, auto2], &q, None).unwrap();
        assert!(m.kind == MatchKind::Untrained && m.ambiguous && !m.is_foreign());
        // Two data skills: ambiguous as before, whatever the auto-skills.
        let data2 = SkillLabels::data("s2", &a);
        let m = match_question(&[data, auto1, data2], &q, None).unwrap();
        assert!(m.ambiguous);
        assert!(m.reason.unwrap().contains("s1, auto-1, s2"));
    }
}
