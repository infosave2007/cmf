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
//!
//! **Matching by descriptions** (0.8.8, DESIGN C2). A choice question whose
//! option ids relate to no skill (untrained, not ambiguous — or, in a
//! state-less request, an exact auto-skill contract: a data skill wins over
//! an auto-skill as in the tie rule above) is related to a DATA skill
//! through its option descriptions ([`match_descriptions`]): every
//! description a string that normalizes ([`normalize_description`]: trimmed,
//! lowercase, `_`, `-` and runs of whitespace → one space) to a distinct
//! active label of the skill, the labels normalized alike — an id→label map
//! ([`DescriptionMap`]). The relation is exact or subset exactly as with
//! ids, and the answer is given in the question's option ids. Besides the
//! mapped options at most one **none option** is allowed (a description
//! that maps to no label and normalizes to a text starting with
//! "out of scope" or "none of"): the question is then decided over every
//! active label of the skill, and a text the gate rejects, or whose winner is
//! not a listed label, is answered locally with the none option. The
//! relation must hold for exactly one data skill (two → no description
//! match); auto-skills are never matched this way. The match reports `"by":
//! "descriptions"` next to its kind.

use crate::protocol::{ApiError, Question, QuestionKind};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

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

/// The option ids of a question mapped to a data skill's labels through
/// their descriptions (DESIGN C2).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DescriptionMap {
    /// (option id, skill label) of the mapped options, in request order.
    pub pairs: Vec<(String, String)>,
    /// The none option ("out of scope…", "none of…"), when the question has
    /// one: the answer to a text the gate rejects.
    pub none: Option<String>,
}

impl DescriptionMap {
    /// The skill label of option `id` (`None` for the none option).
    pub fn label_of(&self, id: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(i, _)| i == id)
            .map(|(_, l)| l.as_str())
    }

    /// The option id of skill label `label`, when it is listed.
    pub fn id_of(&self, label: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(_, l)| l == label)
            .map(|(i, _)| i.as_str())
    }
}

/// A description or label as description matching compares them (DESIGN
/// C2): trimmed, lowercase, `_` and `-` as spaces, every run of whitespace
/// one space.
pub fn normalize_description(s: &str) -> String {
    let lower = s.to_lowercase();
    let spaced: String = lower
        .chars()
        .map(|c| if c == '_' || c == '-' { ' ' } else { c })
        .collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A normalized description of the none option.
fn is_none_description(norm: &str) -> bool {
    norm.starts_with("out of scope") || norm.starts_with("none of")
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
    /// Matched through the option descriptions (DESIGN C2): the option ids'
    /// labels and the none option. `None` for a match by ids or contract.
    pub by_descriptions: Option<DescriptionMap>,
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
            by_descriptions: None,
        }
    }

    /// An exact match of `skill` by ids over all its `n` active labels (the
    /// router's question).
    pub fn exact(skill: &str, n: usize) -> Self {
        Self {
            kind: MatchKind::Exact,
            skill: Some(skill.to_string()),
            candidates: (0..n).collect(),
            unknown: Vec::new(),
            reason: None,
            ambiguous: false,
            by_descriptions: None,
        }
    }

    /// The skill label an option id of the question names: the id itself,
    /// or its mapped label under a description match (`None` for the none
    /// option).
    pub fn label_of<'a>(&'a self, id: &'a str) -> Option<&'a str> {
        match &self.by_descriptions {
            Some(d) => d.label_of(id),
            None => Some(id),
        }
    }

    /// The option id that names skill label `label` (the label itself
    /// without a description match).
    pub fn id_of<'a>(&'a self, label: &'a str) -> Option<&'a str> {
        match &self.by_descriptions {
            Some(d) => d.id_of(label),
            None => Some(label),
        }
    }

    /// The none option of a description match (DESIGN C2).
    pub fn none_option(&self) -> Option<&str> {
        self.by_descriptions.as_ref()?.none.as_deref()
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
        by_descriptions: None,
    })
}

/// The description match of a choice question (DESIGN C2, see the module
/// notes) against the data skills of `skills`; `None` when the descriptions
/// map onto no data skill or onto several.
pub fn match_descriptions(skills: &[SkillLabels<'_>], q: &Question) -> Option<SkillMatch> {
    if q.kind != QuestionKind::Choice {
        return None;
    }
    let criteria = q.criteria.as_ref()?.as_object()?;
    let mut options: Vec<(&str, String)> = Vec::with_capacity(criteria.len());
    for (id, d) in criteria {
        let Value::String(d) = d else {
            return None;
        };
        options.push((id.as_str(), normalize_description(d)));
    }
    let mut found: Option<SkillMatch> = None;
    for s in skills
        .iter()
        .filter(|s| !s.is_auto() && !s.active.is_empty())
    {
        let Some(m) = relate_descriptions(&options, s) else {
            continue;
        };
        if found.is_some() {
            return None;
        }
        found = Some(m);
    }
    found
}

/// [`match_descriptions`] against one data skill.
fn relate_descriptions(options: &[(&str, String)], s: &SkillLabels<'_>) -> Option<SkillMatch> {
    // Normalized label → active index; a form two labels share maps to none.
    let mut by_form: HashMap<String, Option<usize>> = HashMap::new();
    for (i, l) in s.active.iter().enumerate() {
        by_form
            .entry(normalize_description(l))
            .and_modify(|e| *e = None)
            .or_insert(Some(i));
    }
    let mut pairs = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    let mut none: Option<String> = None;
    for (id, form) in options {
        match by_form.get(form) {
            Some(Some(i)) if used.insert(*i) => {
                pairs.push((id.to_string(), s.active[*i].clone()));
            }
            None if is_none_description(form) && none.is_none() => {
                none = Some(id.to_string());
            }
            _ => return None,
        }
    }
    let covers = used.len() == s.active.len();
    let kind = match (covers, none.is_some()) {
        (true, _) => MatchKind::Exact,
        (false, true) if !used.is_empty() => MatchKind::Subset,
        (false, false) if used.len() >= 2 => MatchKind::Subset,
        _ => return None,
    };
    // With a none option the text is decided over every label of the skill
    // (a winner outside the listed ones is answered with the none option).
    let candidates: Vec<usize> = if covers || none.is_some() {
        (0..s.active.len()).collect()
    } else {
        let mut c: Vec<usize> = used.into_iter().collect();
        c.sort_unstable();
        c
    };
    Some(SkillMatch {
        kind,
        skill: Some(s.id.to_string()),
        candidates,
        unknown: Vec::new(),
        reason: None,
        ambiguous: false,
        by_descriptions: Some(DescriptionMap { pairs, none }),
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
    match_question_as(skills, q, forced, false)
}

/// [`match_question`] for a question whose instructions may be its input
/// (`reads_instructions`, a state-less request, DESIGN A19.2): data skills
/// are matched by ids as always; an auto-skill only by the question's
/// state-less contract sha then (criteria alone), and by its stateful one
/// otherwise — the two never equal, so a state-less question never reaches
/// a stateful auto-skill nor the reverse.
pub fn match_question_as(
    skills: &[SkillLabels<'_>],
    q: &Question,
    forced: Option<&str>,
    reads_instructions: bool,
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
    let contract = q.contract_sha256_as(reads_instructions);
    let mut m = match_labels(&pool, &options, contract.as_deref());
    // No skill by ids (or, state-less, only an auto-skill's contract, which
    // a data skill outranks): the descriptions may name a data skill's
    // labels (DESIGN C2).
    let auto_only = reads_instructions
        && m.kind == MatchKind::Exact
        && pool
            .iter()
            .any(|s| s.is_auto() && Some(s.id) == m.skill.as_deref());
    if (m.is_foreign() || auto_only)
        && let Some(d) = match_descriptions(&pool, q)
    {
        m = d;
    }
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

    /// A state-less question (DESIGN A19.2) reaches an auto-skill only
    /// through its state-less contract (criteria alone, whatever the
    /// instructions — they are its text); the same question read with a
    /// state reaches only the stateful contract; data skills match by ids
    /// either way.
    #[test]
    fn a_stateless_question_matches_only_the_stateless_contract() {
        let all = labels(&["cruise", "food", "travel"]);
        let a = question(
            json!("Classify: I want pasta"),
            &["food", "travel", "cruise"],
        );
        let b = question(
            json!("Classify: a cheap ferry"),
            &["food", "travel", "cruise"],
        );
        let sl = a.contract_sha256_as(true).unwrap();
        assert_eq!(sl, b.contract_sha256_as(true).unwrap());
        assert_ne!(sl, a.contract_sha256().unwrap());
        assert_eq!(a.contract_sha256_as(false), a.contract_sha256());
        let stateless = SkillLabels::auto("auto-s", &all, &all, &sl);
        let sf = a.contract_sha256().unwrap();
        let stateful = SkillLabels::auto("auto-f", &all, &all, &sf);
        let m = match_question_as(&[stateless, stateful], &b, None, true).unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Exact, Some("auto-s"))
        );
        let m = match_question_as(&[stateless, stateful], &a, None, false).unwrap();
        assert_eq!(m.skill.as_deref(), Some("auto-f"));
        assert!(
            match_question_as(&[stateless], &a, None, false)
                .unwrap()
                .is_foreign()
        );
        assert!(
            match_question_as(&[stateful], &b, None, true)
                .unwrap()
                .is_foreign()
        );
        let data = SkillLabels::data("d", &all);
        let m = match_question_as(&[data], &b, None, true).unwrap();
        assert_eq!((m.kind, m.skill.as_deref()), (MatchKind::Exact, Some("d")));
    }

    /// A question whose criteria map `ids` to `descriptions`.
    fn described(instructions: &str, pairs: &[(&str, &str)]) -> Question {
        let mut c = serde_json::Map::new();
        for (id, d) in pairs {
            c.insert(id.to_string(), json!(d));
        }
        Question {
            id: "q1".into(),
            kind: QuestionKind::Choice,
            instructions: json!(instructions),
            criteria: Some(Value::Object(c)),
        }
    }

    /// Matching by descriptions (DESIGN C2): positional ids whose
    /// descriptions are a data skill's labels (normalized) relate to it as
    /// exact or subset; one none option makes it decided over every label;
    /// two skills, a description outside the labels, two none options, a
    /// repeated label or a non-string description are no description match;
    /// auto-skills never match this way, and an id match keeps precedence.
    #[test]
    fn descriptions_name_a_data_skill() {
        assert_eq!(
            normalize_description("  Card_Arrival -  soon\t"),
            "card arrival soon"
        );
        let bank = labels(&["card_arrival", "exchange_rate", "Refund_not_showing_up"]);
        let shop = labels(&["exchange_rate", "food"]);
        let skills = [
            SkillLabels::data("bank", &bank),
            SkillLabels::data("shop", &shop),
        ];
        // Exact, BANKING77-style (descriptions = label names).
        let q = described(
            "Classify:\nmy card",
            &[
                ("option_0", "Refund_not_showing_up"),
                ("option_1", "card_arrival"),
                ("option_2", "exchange_rate"),
            ],
        );
        let m = match_question_as(&skills, &q, None, true).unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Exact, Some("bank"))
        );
        assert_eq!(m.candidates, vec![0, 1, 2]);
        let d = m.by_descriptions.as_ref().unwrap();
        assert_eq!(d.label_of("option_0"), Some("Refund_not_showing_up"));
        assert_eq!(m.id_of("card_arrival"), Some("option_1"));
        assert_eq!(m.none_option(), None);
        // CLINC-style: labels with spaces plus the none option.
        let q = described(
            "Classify:\nx",
            &[
                ("option_0", "card arrival"),
                ("option_1", "exchange rate"),
                ("option_2", "Out of scope: none of the listed intents"),
                ("option_3", "refund not showing up"),
            ],
        );
        let m = match_question_as(&skills, &q, None, true).unwrap();
        assert_eq!(m.kind, MatchKind::Exact);
        assert_eq!(m.none_option(), Some("option_2"));
        assert_eq!(m.label_of("option_2"), None);
        // A part of the labels and a none option: subset, decided over all.
        let q = described("x", &[("a", "card arrival"), ("b", "none of these")]);
        let m = match_question_as(&skills, &q, None, false).unwrap();
        assert_eq!(
            (m.kind, m.candidates.clone()),
            (MatchKind::Subset, vec![0, 1, 2])
        );
        // A part without a none option: subset over the listed labels.
        let q = described(
            "x",
            &[("a", "refund not showing up"), ("b", "card arrival")],
        );
        let m = match_question_as(&skills, &q, None, false).unwrap();
        assert_eq!(
            (m.kind, m.candidates.clone()),
            (MatchKind::Subset, vec![0, 2])
        );
        // Not a description match.
        for pairs in [
            // both skills have the label
            vec![("a", "exchange rate"), ("b", "none of them")],
            // a description outside the labels
            vec![
                ("a", "card arrival"),
                ("b", "exchange rate"),
                ("c", "pizza"),
            ],
            // two none options
            vec![
                ("a", "card arrival"),
                ("b", "none of it"),
                ("c", "out of scope"),
            ],
            // a label twice
            vec![("a", "card arrival"), ("b", "Card_Arrival")],
            // one label of a subset is not enough
            vec![("a", "card arrival"), ("b", "pizza")],
        ] {
            let m = match_question_as(&skills, &described("x", &pairs), None, false).unwrap();
            assert!(m.is_foreign(), "{pairs:?}: {m:?}");
        }
        let mut q = described("x", &[("a", "card arrival"), ("b", "exchange rate")]);
        q.criteria.as_mut().unwrap()["b"] = Value::Null;
        assert!(match_question(&skills, &q, None).unwrap().is_foreign());
        // Ids that match keep their match (no description map).
        let q = described(
            "x",
            &[("card_arrival", "exchange rate"), ("exchange_rate", "x")],
        );
        let m = match_question(&skills, &q, None).unwrap();
        assert!(m.kind == MatchKind::Subset && m.by_descriptions.is_none());
        // Auto-skills are never matched by descriptions; in a state-less
        // request a data skill's descriptions outrank an auto-skill's
        // contract, under a state the contract keeps it.
        let q = described(
            "Classify:\nx",
            &[
                ("A", "card arrival"),
                ("B", "exchange rate"),
                ("C", "refund not showing up"),
            ],
        );
        let ids = labels(&["A", "B", "C"]);
        let sl = q.contract_sha256_as(true).unwrap();
        let sf = q.contract_sha256().unwrap();
        let auto_sl = SkillLabels::auto("auto-sl", &ids, &ids, &sl);
        let auto_sf = SkillLabels::auto("auto-sf", &ids, &ids, &sf);
        let m = match_question_as(&[auto_sl], &q, None, true).unwrap();
        assert_eq!(m.skill.as_deref(), Some("auto-sl"));
        let m = match_question_as(&[auto_sl, skills[0]], &q, None, true).unwrap();
        assert_eq!(m.skill.as_deref(), Some("bank"));
        let m = match_question_as(&[auto_sf, skills[0]], &q, None, false).unwrap();
        assert_eq!(m.skill.as_deref(), Some("auto-sf"));
        // `cmf.skill` names the skill the descriptions must match.
        let m = match_question_as(&skills, &q, Some("bank"), true).unwrap();
        assert_eq!(m.skill.as_deref(), Some("bank"));
        let m = match_question_as(&skills, &q, Some("shop"), true).unwrap();
        assert_eq!(m.kind, MatchKind::Untrained);
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
