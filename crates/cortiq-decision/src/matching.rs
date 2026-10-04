//! Skill matching of a question: exact, subset, superset, untrained (spec §4.5).
//!
//! L is the set of option ids of a choice question; A(s) the active labels of
//! skill s (its scored tasks). In order:
//!
//! 1. `cmf.skill` names the skill (an unknown id → 400); the rules below are
//!    then applied to that skill alone;
//! 2. **exact**: L = A(s) for exactly one skill → decided by the skill's gate;
//! 3. **subset**: L ⊊ A(s) for exactly one skill, |L| ≥ 2, with evidence
//!    that the question is the skill's task (0.8.10, see "Subset evidence")
//!    → decided over L only (argmin, softmax, margin and novelty over L; the
//!    same T, θ, τ), never certified;
//! 4. **superset**: A(s) ⊊ L for exactly one skill (the rest of L unknown) →
//!    not decided locally; an oracle answer teaches that skill;
//! 5. **untrained**: everything else — no skill fits, several skills fit
//!    (ambiguous: name one with `cmf.skill`), score and noul questions, a
//!    skill without active tasks.
//!
//! For a data skill the option ids are compared; instructions and criteria
//! descriptions matter to the matcher only through a description match
//! (C2) and as the evidence of a subset (C2.1), and to the oracle.
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
//!
//! **Subset evidence** (0.8.10, DESIGN C2.1). A relation claims that the
//! question *is* the skill's task. An exact relation shows it by itself: the
//! whole label set of a trained skill is its own question. A subset names
//! only a part of that set, and a part is evidence of the task only when it
//! is specific to the skill — a few common words recur in unrelated
//! questions. So a subset relation of a data skill, by ids or by
//! descriptions, holds only with one of these:
//!
//! * **the skill's own question**: `cmf.skill` names the skill (rule 1);
//!   the instructions are the skill's own ([`SkillLabels::asks`]: its
//!   rubric's, or [`DEFAULT_ROUTE_INSTRUCTIONS`] for a skill without a
//!   rubric, whose route question they are — a generic sentence is no one
//!   else's evidence); every listed option is described by the skill's
//!   rubric criterion of its label, verbatim; or the listed labels are the
//!   skill's trained label set ([`SkillLabels::trained`]: its active labels
//!   from the data), the exact question the skill was built for, which a
//!   label the skill learns later (a cold start, spec §5.8) turns into a
//!   subset — growth does not turn the skill's own question away;
//! * **specific labels**: polar answers ([`POLAR_ANSWERS`]: yes, no, maybe,
//!   true, false, normalized as in C2) count for nothing. They answer every
//!   yes/no or true/false question; a skill that has them as classes (an
//!   assistant's intents "the user affirms", "the user denies") classifies
//!   what an utterance does, while a question listing them asks whether
//!   something holds of the state — an answer form, not the skill's
//!   classes, whatever its descriptions (`"yes": "Yes"` restates the word).
//!   Of the other listed labels, three or more are evidence by their ids
//!   (as before 0.8.10). Fewer must each be **specific**: a compound name
//!   (two words or more, normalized as in C2, such as `card_arrival`),
//!   which an unrelated question does not use by accident, or a single word
//!   that the option's description gives as the skill's own name for it —
//!   the label itself in the C2 form (a description match is so by
//!   construction) or the skill's rubric criterion ([`describes`]); a
//!   description that merely contains the word ("about travel", "a value of
//!   type date") is no evidence, an unrelated question describes its
//!   options in the same words. One label alone, beside polar answers or a
//!   none option, must be compound.
//!
//! Measured before 0.8.10 on a public decision benchmark: binary questions
//! of unrelated tasks (aspect presence, tool relevance, sarcasm, answer
//! selection, causal queries, forecasts) were all taken as a subset of an
//! intent skill with yes/no intents and answered locally at 1–45 %
//! accuracy, never reaching the oracle. What the rule still lets through
//! is a collision of an unrelated question with three of a skill's
//! one-word labels, or with compound ones; `cmf.skill` takes any subset,
//! and anything else goes to the oracle.
//!
//! A subset without that evidence is no relation to the skill: another skill
//! may still relate (two that do are ambiguous, as before), and with none
//! the question is untrained and foreign (the oracle answers it, and it may
//! be learned as an auto-skill of its own contract); the untrained reason
//! names the refused subset. `match_labels` and `skill_for_labels` see
//! option ids alone and apply no evidence rule: `cortiq decide --labels`
//! resolves its skill with them and names it with `cmf.skill`.

use crate::manifest::Rubric;
use crate::protocol::{ApiError, Question, QuestionKind};
use crate::service::DEFAULT_ROUTE_INSTRUCTIONS;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

/// Polar answers, normalized ([`normalize_description`]): the answer words
/// of any yes/no or true/false question. As labels of a subset relation they
/// are no evidence of a skill's task (DESIGN C2.1).
pub const POLAR_ANSWERS: [&str; 5] = ["yes", "no", "maybe", "true", "false"];

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
    /// A data skill's rubric (its own instructions and label criteria), the
    /// evidence of a subset relation (DESIGN C2.1); `None` without one.
    pub rubric: Option<&'a Rubric>,
    /// A data skill's trained label set: its active labels from the data
    /// (not a cold start), the exact question it was built for — a subset
    /// that lists them all is its own question (DESIGN C2.1). `active` when
    /// not given.
    pub trained: &'a [String],
}

impl<'a> SkillLabels<'a> {
    /// A data skill: known = active, matched by its ids.
    pub fn data(id: &'a str, active: &'a [String]) -> Self {
        Self {
            id,
            active,
            known: active,
            contract: None,
            rubric: None,
            trained: active,
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
            rubric: None,
            trained: active,
        }
    }

    /// The skill with its rubric (a data skill's own question, DESIGN C2.1).
    pub fn with_rubric(mut self, rubric: Option<&'a Rubric>) -> Self {
        self.rubric = rubric;
        self
    }

    /// The skill with its trained label set ([`Self::trained`], DESIGN C2.1).
    pub fn with_trained(mut self, trained: &'a [String]) -> Self {
        self.trained = trained;
        self
    }

    pub fn is_auto(&self) -> bool {
        self.contract.is_some()
    }

    /// Whether `instructions` are the skill's own (DESIGN C2.1): its rubric's
    /// (a string), or — for a skill without a rubric only —
    /// [`DEFAULT_ROUTE_INSTRUCTIONS`], its route question. A skill with a
    /// rubric never asks that generic sentence: any client may send it.
    pub fn asks(&self, instructions: &Value) -> bool {
        match self.rubric {
            Some(r) => r.instructions.is_string() && r.instructions == *instructions,
            None => matches!(instructions, Value::String(s) if s == DEFAULT_ROUTE_INSTRUCTIONS),
        }
    }

    /// Whether `labels` (distinct) are the skill's whole trained label set.
    fn is_trained_set(&self, labels: &[&str]) -> bool {
        labels.len() == self.trained.len()
            && labels.iter().all(|l| self.trained.iter().any(|t| t == l))
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

/// A label that is a polar answer ([`POLAR_ANSWERS`]).
fn is_polar(label: &str) -> bool {
    POLAR_ANSWERS.contains(&normalize_description(label).as_str())
}

/// The words of `s` normalized as in C2, split at every character that is
/// not alphanumeric (so punctuation is no part of a word).
fn words(s: &str) -> Vec<String> {
    normalize_description(s)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// A compound label: two words or more ([`words`]: `card_arrival`,
/// `Refund_not_showing_up`), a name an unrelated question does not use by
/// accident (DESIGN C2.1).
fn is_compound(label: &str) -> bool {
    words(label).len() >= 2
}

/// Whether `d` is `rubric`'s criterion of `label`, verbatim (a non-empty
/// string: `null` describes nothing).
fn is_criterion(rubric: Option<&Rubric>, label: &str, d: &Value) -> bool {
    rubric.is_some_and(|r| {
        matches!(r.criteria.get(label), Some(c @ Value::String(t)) if !t.trim().is_empty() && c == d)
    })
}

/// Whether description `d` of an option gives skill label `label` by the
/// skill's own name for it (DESIGN C2.1): the label itself, normalized as
/// in C2 (`"Alarm"`, `"card arrival"`), or the skill's rubric criterion of
/// the label, verbatim. A text that merely contains the label's words
/// ("about alarm", "a value of type date"), `null` or another structure
/// does not.
pub fn describes(d: &Value, label: &str, rubric: Option<&Rubric>) -> bool {
    is_criterion(rubric, label, d)
        || matches!(d, Value::String(t) if normalize_description(t) == normalize_description(label))
}

/// Whether question `q` shows that its subset relation to data skill `s`
/// is the skill's task (DESIGN C2.1, see the module notes). `listed` holds
/// (option id, skill label) of the listed options (the none option of a
/// description match is not among them); `by_descriptions` for a
/// description match, whose descriptions give the labels in the C2 form
/// already. The error says why not, for the untrained reason.
fn subset_evidence(
    s: &SkillLabels<'_>,
    q: &Question,
    listed: &[(&str, &str)],
    by_descriptions: bool,
) -> Result<(), String> {
    let criteria = q.criteria.as_ref().and_then(Value::as_object);
    let description = |id: &str| criteria.and_then(|c| c.get(id));
    let labels: Vec<&str> = listed.iter().map(|(_, l)| *l).collect();
    // The skill's own question: its instructions, its criteria verbatim, or
    // its trained label set.
    if s.asks(&q.instructions)
        || listed
            .iter()
            .all(|(id, l)| description(id).is_some_and(|d| is_criterion(s.rubric, l, d)))
        || s.is_trained_set(&labels)
    {
        return Ok(());
    }
    // Specific labels: polar answers count for nothing; three others are
    // evidence by their ids, fewer must each be specific.
    let other: Vec<&(&str, &str)> = listed.iter().filter(|(_, l)| !is_polar(l)).collect();
    let named = |id: &str, l: &str| {
        is_compound(l)
            || by_descriptions
            || description(id).is_some_and(|d| describes(d, l, s.rubric))
    };
    match other.as_slice() {
        [] => Err(format!(
            "options {} are polar answers, a subset of skill '{}' that does not name its task",
            labels.join(", "),
            s.id
        )),
        [(_, l)] if !is_compound(l) => Err(format!(
            "options {} name one label of skill '{}' beside polar answers or a none option, the single word '{l}'",
            labels.join(", "),
            s.id
        )),
        [_] => Ok(()),
        [_, _] => match other.iter().find(|(id, l)| !named(id, l)) {
            Some((_, l)) => Err(format!(
                "options {} are a subset of skill '{}' with two labels besides polar answers, and the single word '{l}' is described neither by its name nor by the skill's criterion",
                labels.join(", "),
                s.id
            )),
            None => Ok(()),
        },
        _ => Ok(()),
    }
}

/// The untrained reason of a question no skill fits.
const NO_SKILL: &str = "no skill is trained on these options";

/// The untrained reason of a question no skill fits, after the `refused`
/// subset when there was one (DESIGN C2.1).
fn untrained_reason(refused: Option<String>) -> String {
    match refused {
        Some(why) => format!("{NO_SKILL} ({why}; name the skill with cmf.skill to use it)"),
        None => NO_SKILL.into(),
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
/// map onto no data skill or onto several. A subset needs the evidence of
/// DESIGN C2.1 (not only polar answers, or the skill's own instructions).
pub fn match_descriptions(skills: &[SkillLabels<'_>], q: &Question) -> Option<SkillMatch> {
    describe(skills, q, true).0
}

/// [`match_descriptions`], the subset evidence checked when `evidence`
/// (not under `cmf.skill`); also the reason of the first subset refused.
fn describe(
    skills: &[SkillLabels<'_>],
    q: &Question,
    evidence: bool,
) -> (Option<SkillMatch>, Option<String>) {
    if q.kind != QuestionKind::Choice {
        return (None, None);
    }
    let Some(criteria) = q.criteria.as_ref().and_then(Value::as_object) else {
        return (None, None);
    };
    let mut options: Vec<(&str, String)> = Vec::with_capacity(criteria.len());
    for (id, d) in criteria {
        let Value::String(d) = d else {
            return (None, None);
        };
        options.push((id.as_str(), normalize_description(d)));
    }
    let mut found: Option<SkillMatch> = None;
    let mut refused: Option<String> = None;
    for s in skills
        .iter()
        .filter(|s| !s.is_auto() && !s.active.is_empty())
    {
        let Some(m) = relate_descriptions(&options, s) else {
            continue;
        };
        if evidence && m.kind == MatchKind::Subset {
            let listed: Vec<(&str, &str)> = m
                .by_descriptions
                .as_ref()
                .map(|d| {
                    d.pairs
                        .iter()
                        .map(|(i, l)| (i.as_str(), l.as_str()))
                        .collect()
                })
                .unwrap_or_default();
            if let Err(why) = subset_evidence(s, q, &listed, true) {
                refused.get_or_insert(why);
                continue;
            }
        }
        if found.is_some() {
            return (None, refused);
        }
        found = Some(m);
    }
    (found, refused)
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
/// invisible). The ids alone, as the skill's own question names them: no
/// subset evidence is asked (DESIGN C2.1 applies to a request's question,
/// [`match_question_as`]).
pub fn match_labels(
    skills: &[SkillLabels<'_>],
    options: &[&str],
    contract: Option<&str>,
) -> SkillMatch {
    match_ids(skills, options, contract, None)
}

/// [`match_labels`]; with `evidence` (the question, not under `cmf.skill`)
/// a subset relation holds only with the evidence of DESIGN C2.1 — one
/// without it is no relation, and the untrained reason names it.
fn match_ids(
    skills: &[SkillLabels<'_>],
    options: &[&str],
    contract: Option<&str>,
    evidence: Option<&Question>,
) -> SkillMatch {
    let set: HashSet<&str> = options.iter().copied().collect();
    let mut refused: Option<String> = None;
    let found: Vec<(SkillMatch, bool)> = skills
        .iter()
        .filter_map(|s| {
            let m = relate(options, &set, contract, s)?;
            if m.kind == MatchKind::Subset
                && let Some(q) = evidence
            {
                // A subset by ids: every option is one of the skill's labels.
                let listed: Vec<(&str, &str)> = options.iter().map(|o| (*o, *o)).collect();
                if let Err(why) = subset_evidence(s, q, &listed, false) {
                    refused.get_or_insert(why);
                    return None;
                }
            }
            Some((m, s.is_auto()))
        })
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
    SkillMatch::untrained(untrained_reason(refused))
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
    // `cmf.skill` names the task: a subset of it needs no other evidence
    // (DESIGN C2.1).
    let evidence = forced.is_none();
    let mut m = match_ids(&pool, &options, contract.as_deref(), evidence.then_some(q));
    // No skill by ids (or, state-less, only an auto-skill's contract, which
    // a data skill outranks): the descriptions may name a data skill's
    // labels (DESIGN C2).
    let auto_only = reads_instructions
        && m.kind == MatchKind::Exact
        && pool
            .iter()
            .any(|s| s.is_auto() && Some(s.id) == m.skill.as_deref());
    if m.is_foreign() || auto_only {
        match describe(&pool, q, evidence) {
            (Some(d), _) => m = d,
            // The ids refused no subset, the descriptions did: say so.
            (None, Some(why)) if m.is_foreign() && m.reason.as_deref() == Some(NO_SKILL) => {
                m.reason = Some(untrained_reason(Some(why)));
            }
            _ => {}
        }
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
        // Ids that match keep their match (no description map): compound
        // ids are evidence by themselves, whatever the descriptions (DESIGN
        // C2.1).
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
        // A part of the ids (described by their names, the evidence two
        // one-word labels need, DESIGN C2.1): the data skill's subset; no
        // auto-skill relates.
        let m = match_question(
            &[auto1, data],
            &described("Which?", &[("a", "a"), ("b", "b")]),
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

    /// An assistant-intent skill whose classes include the polar answers
    /// (the user affirms, denies, hesitates), one-word intents and a
    /// compound one, with its rubric; a banking skill of compound labels
    /// without a rubric (DESIGN C2.1).
    struct Intents {
        assistant: Vec<String>,
        bank: Vec<String>,
        rubric: Rubric,
    }

    impl Intents {
        fn new() -> Self {
            let assistant = labels(&[
                "alarm",
                "calculator",
                "cancel",
                "date",
                "maybe",
                "no",
                "share_location",
                "time",
                "timer",
                "weather",
                "yes",
            ]);
            let mut crit = serde_json::Map::new();
            for l in &assistant {
                crit.insert(
                    l.clone(),
                    json!(format!("{l}. Labeled training examples: …")),
                );
            }
            // A criterion that does not repeat its label's words.
            crit.insert("timer".into(), json!("Count down some minutes"));
            Self {
                assistant,
                bank: labels(&[
                    "Refund_not_showing_up",
                    "card_arrival",
                    "card_delivery_estimate",
                    "card_payment_not_recognised",
                    "compromised_card",
                    "exchange_rate",
                    "lost_or_stolen_card",
                ]),
                rubric: Rubric::new("Classify the assistant intent.", crit),
            }
        }

        fn skills(&self) -> [SkillLabels<'_>; 2] {
            [
                SkillLabels::data("assistant", &self.assistant).with_rubric(Some(&self.rubric)),
                SkillLabels::data("bank", &self.bank),
            ]
        }

        fn criterion(&self, label: &str) -> Value {
            self.rubric.criteria[label].clone()
        }
    }

    /// A choice question under `instructions` with `criteria` as given.
    fn asked(instructions: &str, criteria: Value) -> Question {
        Question {
            id: "q".into(),
            kind: QuestionKind::Choice,
            instructions: json!(instructions),
            criteria: Some(criteria),
        }
    }

    /// The match of `q` (no `cmf.skill`) as (kind, skill).
    fn kind_of(skills: &[SkillLabels<'_>], q: &Question) -> (MatchKind, Option<String>) {
        let m = match_question(skills, q, None).unwrap();
        (m.kind, m.skill)
    }

    /// `q` is foreign (no skill), its reason containing `why`, read with a
    /// state and state-less alike.
    fn refused(skills: &[SkillLabels<'_>], q: &Question, why: &str) {
        for state_less in [false, true] {
            let m = match_question_as(skills, q, None, state_less).unwrap();
            assert!(m.is_foreign(), "{:?}: {m:?}", q.criteria);
            let r = m.reason.unwrap();
            assert!(r.contains(why), "{:?}: {r}", q.criteria);
        }
    }

    #[test]
    fn describes_is_the_skills_own_name_for_a_label() {
        let r = Intents::new().rubric;
        for (d, l) in [
            (json!("Alarm"), "alarm"),
            (json!(" card  arrival "), "card_arrival"),
            (json!("Card-Arrival"), "card_arrival"),
            (json!("Yes"), "yes"),
        ] {
            assert!(describes(&d, l, None), "{d} {l}");
        }
        // Text that merely contains the label's words is no name of it.
        for (d, l) in [
            (json!("about travel"), "travel"),
            (json!("Set an alarm"), "alarm"),
            (json!("a value of type date"), "date"),
            (json!("The card arrival is late."), "card_arrival"),
            (json!("timers"), "timer"),
            (Value::Null, "alarm"),
            (json!({"text": "alarm"}), "alarm"),
        ] {
            assert!(!describes(&d, l, Some(&r)), "{d} {l}");
        }
        // The skill's own criterion names its label, whatever its words.
        let count = json!("Count down some minutes");
        assert!(describes(&count, "timer", Some(&r)));
        assert!(!describes(&count, "alarm", Some(&r)));
        assert!(!describes(&count, "timer", None));
        // Compound labels: two words or more, normalized as in C2.
        for l in [
            "card_arrival",
            "Refund_not_showing_up",
            "reverted_card_payment?",
        ] {
            assert!(is_compound(l), "{l}");
        }
        for l in ["alarm", "w2", "Weather", "yes"] {
            assert!(!is_compound(l), "{l}");
        }
    }

    /// Binary questions whose answers are named yes/no — by ids or by
    /// descriptions, described as propositions about the state, restating
    /// the word, or `null` — are no subset of a skill that has yes/no as
    /// intents: untrained and foreign (the oracle; learnable as their own
    /// contract), the reason naming the refused subset. Polar labels count
    /// for nothing beside others either: one more one-word label is no
    /// evidence. `cmf.skill`, the skill's own instructions and its own
    /// criteria verbatim still take them.
    #[test]
    fn polar_answers_are_no_subset_evidence() {
        let x = Intents::new();
        let skills = x.skills();
        let polar = [
            // A proposition about the state per answer.
            asked(
                "Does the review express this aspect?",
                json!({"yes": "The exact category/sentiment pair is present.",
                       "no": "The exact category/sentiment pair is absent."}),
            ),
            // The descriptions restate the answer words.
            asked(
                "Is this text intended to be sarcastic?",
                json!({"no": "No", "yes": "Yes"}),
            ),
            asked("Is it relevant?", json!({"yes": null, "no": null})),
            asked(
                "Does the abstract support it?",
                json!({"yes": "yes", "no": "no", "maybe": "maybe"}),
            ),
            // Intents described in prose: not the skill's own criteria.
            asked(
                "Did the user agree?",
                json!({"yes": "The user affirms", "no": "The user denies"}),
            ),
            // Positional ids, the answer words as descriptions (C2 path).
            asked("Would it be more likely?", json!({"A": "yes", "B": "no"})),
            asked(
                "Which option is correct?",
                json!({"Yes": "Yes", "No": "No"}),
            ),
            asked("Which?", json!({"A": "yes", "B": "out of scope"})),
        ];
        for q in &polar {
            refused(&skills, q, "polar answers, a subset of skill 'assistant'");
        }
        // One one-word label beside polar answers (or a none option) is no
        // evidence, by ids or by descriptions, however described.
        for q in [
            asked(
                "Should the agent go ahead with the refund?",
                json!({"yes": "Proceed", "no": "Do not proceed", "cancel": "Abort the whole operation"}),
            ),
            asked(
                "Which?",
                json!({"yes": "about yes", "no": "about no", "alarm": "Alarm"}),
            ),
            asked("Proceed?", json!({"A": "Yes", "B": "No", "C": "Cancel"})),
            asked("Proceed?", json!({"A": "Yes", "B": "Cancel"})),
            asked(
                "Is it an alarm?",
                json!({"A": "alarm", "B": "out of scope"}),
            ),
        ] {
            refused(&skills, &q, "beside polar answers or a none option");
        }
        // Two one-word labels beside them need their names; three are
        // evidence by their ids; a compound one is evidence by itself.
        refused(
            &skills,
            &asked(
                "Which?",
                json!({"yes": null, "no": null, "alarm": null, "timer": null}),
            ),
            "the single word 'alarm' is described neither",
        );
        let assistant = |k| (k, Some("assistant".to_string()));
        for c in [
            json!({"yes": null, "no": null, "alarm": "Alarm", "timer": "timer"}),
            json!({"yes": null, "no": null, "alarm": null, "timer": null, "date": null}),
            json!({"yes": null, "no": null, "share_location": null}),
        ] {
            assert_eq!(
                kind_of(&skills, &asked("Which?", c.clone())),
                assistant(MatchKind::Subset),
                "{c}"
            );
        }
        // The skill's own question: `cmf.skill`, the rubric's instructions,
        // its criteria verbatim under other instructions.
        let yes_no = json!({"yes": "Yes", "no": "No"});
        let m = match_question(
            &skills,
            &asked("Is it sarcastic?", yes_no.clone()),
            Some("assistant"),
        )
        .unwrap();
        assert_eq!(m.kind, MatchKind::Subset);
        assert_eq!(
            kind_of(&skills, &asked("Classify the assistant intent.", yes_no)),
            assistant(MatchKind::Subset)
        );
        let own = json!({"yes": x.criterion("yes"), "no": x.criterion("no")});
        assert_eq!(
            kind_of(&skills, &asked("Did the user agree to the booking?", own)),
            assistant(MatchKind::Subset)
        );
        // Not when one of them is missing or another text.
        let half = json!({"yes": x.criterion("yes"), "no": "No"});
        refused(
            &skills,
            &asked("Did the user agree to the booking?", half),
            "polar answers",
        );
        // Labels alone are the skill's own question (`skill_for_labels`).
        assert_eq!(
            skill_for_labels(&skills, &["yes", "no"]).unwrap(),
            "assistant"
        );
    }

    /// A subset of fewer than three labels needs each one specific: a
    /// compound id is (the published examples: two banking ids, `null` or
    /// paraphrased descriptions); a one-word id needs its own name or the
    /// skill's criterion as the description — a description that only
    /// contains the word is no evidence. From three ids on the ids are the
    /// evidence. Real skill questions keep their matches.
    #[test]
    fn a_subset_needs_specific_labels() {
        let x = Intents::new();
        let skills = x.skills();
        let assistant = |k| (k, Some("assistant".to_string()));
        let bank = |k| (k, Some("bank".to_string()));
        // The skill's own question: exact, by ids, under foreign instructions.
        let own = Value::Object(x.rubric.ordered_criteria());
        assert_eq!(
            kind_of(&skills, &asked("Which intent?", own)),
            assistant(MatchKind::Exact)
        );
        // Two compound ids: `null`, paraphrased or no instructions at all.
        for q in [
            asked(
                "Classify the message.",
                json!({"card_arrival": null, "card_delivery_estimate": null}),
            ),
            asked(
                "Classify the banking customer message.",
                json!({"card_arrival": "The customer is waiting for a card to arrive",
                       "lost_or_stolen_card": "The card was lost or stolen"}),
            ),
            Question {
                instructions: Value::Null,
                ..asked(
                    "",
                    json!({"card_arrival": null, "card_delivery_estimate": null}),
                )
            },
        ] {
            assert_eq!(kind_of(&skills, &q), bank(MatchKind::Subset), "{q:?}");
        }
        // Two one-word ids described by their names or the rubric's criteria;
        // a compound id beside a named one-word id.
        for c in [
            json!({"alarm": "Alarm", "weather": "weather"}),
            json!({"timer": "Count down some minutes", "alarm": x.criterion("alarm")}),
            json!({"share_location": null, "weather": "Weather"}),
        ] {
            assert_eq!(
                kind_of(&skills, &asked("Which intent?", c.clone())),
                assistant(MatchKind::Subset),
                "{c}"
            );
        }
        // Two one-word ids without their names: untrained, unless the
        // instructions are the skill's own.
        for c in [
            json!({"time": null, "date": null}),
            json!({"time": "a value of type time", "date": "a value of type date"}),
            json!({"calculator": "A calculator for arithmetic", "weather": "Look up the weather"}),
            json!({"alarm": "Set an alarm", "timer": "Start a countdown"}),
            json!({"share_location": null, "weather": "Look up the weather"}),
        ] {
            refused(
                &skills,
                &asked("Which tool is relevant?", c.clone()),
                "two labels besides polar answers, and the single word",
            );
            assert_eq!(
                kind_of(&skills, &asked("Classify the assistant intent.", c)),
                assistant(MatchKind::Subset)
            );
        }
        // Three ids: `null` or paraphrased descriptions keep the match.
        assert_eq!(
            kind_of(
                &skills,
                &asked("Which?", json!({"time": null, "date": null, "alarm": null}))
            ),
            assistant(MatchKind::Subset)
        );
        // Positional ids described by the label names (C2): two labels are
        // a description match, named by construction.
        let q = asked(
            "Classify:\nwake me at 7",
            json!({"option_0": "alarm", "option_1": "timer"}),
        );
        let m = match_question_as(&skills, &q, None, true).unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Subset, Some("assistant"))
        );
        assert!(m.by_descriptions.is_some());
    }

    /// A skill's trained label set is its own question: a label it learns
    /// later (a cold start) makes that question a subset, which stays the
    /// skill's — polar or one-word labels included — while other parts of
    /// the grown set need evidence as any subset (DESIGN C2.1).
    #[test]
    fn label_growth_keeps_the_trained_question() {
        let grown = labels(&["no", "yes", "maybe"]);
        let trained = labels(&["no", "yes"]);
        let polar = SkillLabels::data("claims", &grown).with_trained(&trained);
        let q = asked(
            "Is the claim supported?",
            json!({"yes": "It is", "no": "It is not"}),
        );
        // Before the growth: exact.
        let before = SkillLabels::data("claims", &trained);
        assert_eq!(
            match_question(&[before], &q, None).unwrap().kind,
            MatchKind::Exact
        );
        // After: the trained set is still the skill's question.
        let m = match_question(&[polar], &q, None).unwrap();
        assert_eq!(
            (m.kind, m.skill.as_deref()),
            (MatchKind::Subset, Some("claims"))
        );
        refused(
            &[polar],
            &asked("Is it?", json!({"yes": null, "maybe": null})),
            "polar answers",
        );
        // One-word labels alike.
        let grown = labels(&["alarm", "snooze", "timer"]);
        let trained = labels(&["alarm", "timer"]);
        let clock = SkillLabels::data("clock", &grown).with_trained(&trained);
        let m = match_question(
            &[clock],
            &asked("Which?", json!({"timer": null, "alarm": null})),
            None,
        )
        .unwrap();
        assert_eq!(m.kind, MatchKind::Subset);
        refused(
            &[clock],
            &asked("Which?", json!({"alarm": null, "snooze": null})),
            "the single word",
        );
    }

    /// The route instructions are the own question of a skill without a
    /// rubric only: a skill with one never asks that generic sentence, so a
    /// client sending it gains nothing (DESIGN C2.1).
    #[test]
    fn route_instructions_are_a_rubricless_skills_own() {
        let x = Intents::new();
        let skills = x.skills();
        refused(
            &skills,
            &asked(
                DEFAULT_ROUTE_INSTRUCTIONS,
                json!({"yes": "The claim is supported", "no": "The claim is not supported"}),
            ),
            "polar answers",
        );
        refused(
            &skills,
            &asked(
                DEFAULT_ROUTE_INSTRUCTIONS,
                json!({"alarm": null, "timer": null}),
            ),
            "the single word",
        );
        let plain = labels(&["alarm", "timer", "weather"]);
        let rubricless = [SkillLabels::data("plain", &plain)];
        assert_eq!(
            kind_of(
                &rubricless,
                &asked(
                    DEFAULT_ROUTE_INSTRUCTIONS,
                    json!({"alarm": null, "timer": null})
                )
            ),
            (MatchKind::Subset, Some("plain".to_string()))
        );
        refused(
            &rubricless,
            &asked("Which?", json!({"alarm": null, "timer": null})),
            "the single word",
        );
    }

    /// Every request of the published API reference (`docs/decision/hf/
    /// API.md`, its `-d '…'` bodies) whose choice question lists banking
    /// labels is answered locally there: it must match the banking skill
    /// next to an intent skill with yes/no intents, as the shipped model
    /// has them — the docs and the matcher stay in step.
    #[test]
    fn the_documented_banking_examples_match_locally() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/decision/hf/API.md");
        let doc = std::fs::read_to_string(&path).unwrap();
        let x = Intents::new();
        let skills = x.skills();
        let mut seen = 0;
        for (i, _) in doc.match_indices("-d '") {
            let rest = &doc[i + 4..];
            let Ok(body) = serde_json::from_str::<Value>(&rest[..rest.find('\'').unwrap()]) else {
                continue;
            };
            let Some(questions) = body.get("questions").and_then(Value::as_object) else {
                continue;
            };
            for q in questions.values() {
                let Some(criteria) = q.get("criteria").and_then(Value::as_object) else {
                    continue;
                };
                if q["type"] != "choice" || !criteria.keys().all(|k| x.bank.contains(k)) {
                    continue;
                }
                let q = Question {
                    id: "intent".into(),
                    kind: QuestionKind::Choice,
                    instructions: q.get("instructions").cloned().unwrap_or(Value::Null),
                    criteria: Some(Value::Object(criteria.clone())),
                };
                let m = match_question(&skills, &q, None).unwrap();
                assert_eq!(
                    (m.kind, m.skill.as_deref()),
                    (MatchKind::Subset, Some("bank")),
                    "{q:?}"
                );
                seen += 1;
            }
        }
        // 3.1, the explanation example, /v1/feedback, the System One adapter.
        assert_eq!(seen, 4);
    }
}
