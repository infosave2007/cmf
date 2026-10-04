//! Cache of oracle answers (spec §5.6; exact-first since 0.8.10).
//!
//! An entry is `{scope, input, φ_P, answer, ts}`:
//! * **scope** of a question matched to a skill (exact, subset, superset):
//!   `skill:<id>:<sha256 of canonical JSON of the sorted option ids>:<sha256 of
//!   its contract>`; of any other question: `contract:<sha256(canonical({type,
//!   instructions, criteria}))>` ([`scope_of`]). The contract — what the
//!   oracle was told — is part of every scope: the cache is shared by all
//!   accounts, and an answer given under one caller's instructions is never
//!   served to a question with other instructions or criteria (spec §5.6
//!   named only the skill and the options; that let one account's
//!   instructions decide another's answers). Single flight uses the same
//!   scopes. A `cache` answer does tell its caller that some account asked
//!   the same text (a near-identical one with near reuse on) under the same
//!   contract;
//! * **input** (0.8.10, [`input_digest`]) is the sha256 of what the oracle
//!   read besides the scope, as asked (before PII redaction): the canonical
//!   JSON `{"state": <state>}`, or for a question that reads its
//!   instructions (a state-less request, DESIGN A19) `{"instructions":
//!   <instructions>}` — the whole value, the part past the encoder's 512
//!   tokens and a state-less question's lead-in line included. The scope and
//!   the input together are the question exactly: the same contract over the
//!   same state;
//! * **φ_P** is the encoder's unit vector of the state (the router's embedding,
//!   cortiq-router `cache.rs:66-78`), so cos is the dot product;
//! * a question of a **state-less** request (0.8.8, DESIGN A19.4) reads its
//!   instructions: its φ_P is theirs and its scope hashes the state-less
//!   contract `{type, input: "instructions", criteria}` instead
//!   ([`scope_of_as`]) — the text is in φ_P and the input, not in the scope.
//!
//! **Lookup** ([`SemanticCache::get`]) scans the entries of the question's
//! scope, oldest first:
//! 1. **exact** — the earliest entry with the question's input digest hits,
//!    whatever its φ_P;
//! 2. **legacy** — an entry replayed from a `learn.log` record written before
//!    0.8.10 has no digest: the best of those by cos φ_P (the earliest on a
//!    tie) hits when cos ≥ `cache.legacy_cos` (default [`EXACT_COS`], 0.9999:
//!    a repeat of the text as far as the encoder reads it). That is not
//!    exact — two states that differ past the encoder's 512 tokens, or two
//!    state-less questions whose texts differ only in the lead-in line, have
//!    cos 1 — so `cache.legacy_cos` 1 ([`LEGACY_OFF`]) turns such entries
//!    off: they are not loaded and answer nothing, and an oracle pass gives
//!    each question an entry with its digest;
//! 3. **near** (opt-in) — only when `cache.threshold` is below 1: the best
//!    entry by cos φ_P, digest or not (the earliest on a tie), hits when cos ≥
//!    `cache.threshold`.
//!
//! The default `cache.threshold` is 1 (0.97 before 0.8.10): near reuse is
//! off and only the same question hits. Decision states that look alike
//! differ in the detail that decides — a note in a JSON score, a number in a
//! causal question, one address in an e-mail — and at 0.97 a Decision Index
//! run through the gateway answered ~60k of 282k questions with another row's
//! oracle answer (index 58 → 51.9; 57.78 with 0.9999). A threshold below 1
//! turns near reuse back on for traffic whose paraphrases share an answer.
//!
//! **Put** ([`SemanticCache::put`]): an entry of the same scope, input digest
//! and verdict makes the put a no-op (a digest-less entry — only a replayed
//! one — is a no-op next to a digest-less entry of the same scope and verdict
//! with cos ≥ [`PUT_DEDUP`], the rule before 0.8.10, and is not kept at all
//! with legacy reuse off); at capacity (50,000) the oldest entry leaves (a
//! ring). The cache is consulted only for a question
//! that was escalated — also when the oracle may not be called for it (0.8.10:
//! consent off, the oracle disabled, stopped or out of budget; a hit costs
//! nothing and sends nothing) — and its puts are kept in `learn.log` and
//! replayed at start.
//!
//! **`learn.log`** (0.8.10): the digest travels in the scope field of the
//! unchanged `CachePut` / `CachePutP` record, as the prefix
//! `exact:<64 hex>|` (`CacheEntry::encode`). A scope starts with `skill:`
//! or `contract:`, so the prefix is unambiguous: a record without it (every
//! record of 0.8.9 and older) replays as a digest-less entry, unchanged, and
//! a 0.8.9 binary replays a prefixed record as an entry of a scope no lookup
//! names — it never hits — instead of stopping its replay (and truncating
//! the log) at a record kind it does not know.
//!
//! **Distributions** (0.8.8, DESIGN C3): an entry keeps the oracle's verdict
//! with its normalized distribution ([`Verdict`]), so a hit answers with it.
//! An entry without one (a one-hot verdict) is logged as the `CachePut`
//! record of 0.8.7; an entry with one as a `CachePutP` record (the same
//! payload, then the distribution) — a 0.8.7 binary stops its replay at the
//! first such record. Two answers are "the same" for the put's dedup when
//! their verdicts are, whatever their distributions.
//!
//! **Scope index** (0.8.8, DESIGN B3): next to the global ring, each scope
//! keeps the sequence numbers of its entries in insertion order, so a lookup
//! and a put's dedup scan only that scope's entries — the same entries in the
//! same order as a scan of the whole ring filtered by scope, hence the same
//! answers, ties and evictions. The index is derived from the ring (a replay
//! of `learn.log` rebuilds it through [`SemanticCache::put`]).

use crate::answer::{OracleAnswer, Verdict};
use crate::buffer::{Dec, Enc, dot};
use crate::canonical;
use crate::config::CacheConfig;
use crate::matching::SkillMatch;
use crate::protocol::{DecisionRequest, Question};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

/// cos φ_P above which a put of a digest-less entry of the same scope and
/// answer is skipped.
pub const PUT_DEDUP: f32 = 0.999;
/// cos φ_P from which a digest-less entry (logged before 0.8.10) answers a
/// question of its scope by default (`cache.legacy_cos`): the repeat of a
/// text as far as the encoder reads it.
pub const EXACT_COS: f32 = 0.9999;
/// The default `cache.threshold`: near reuse off, only the same question hits.
pub const NEAR_OFF: f32 = 1.0;
/// The `cache.legacy_cos` that turns digest-less entries off: they are not
/// kept and answer nothing.
pub const LEGACY_OFF: f32 = 1.0;
/// Prefix of a logged scope that carries the entry's input digest.
const INPUT_TAG: &str = "exact:";

/// sha256 of a question's input ([`input_digest`]).
pub type InputDigest = [u8; 32];

/// The input digest of question `q` of `req` (see the module notes): the
/// sha256 of the canonical JSON `{"instructions": …}` when the question reads
/// its instructions, else of `{"state": …}`.
pub fn input_digest(req: &DecisionRequest, q: &Question) -> InputDigest {
    if req.reads_instructions(q) {
        digest_of(&json!({"instructions": q.instructions}))
    } else {
        state_digest(req)
    }
}

/// The input digest of a question that reads the state (the same for every
/// such question of `req`).
pub fn state_digest(req: &DecisionRequest) -> InputDigest {
    digest_of(&json!({"state": req.state.to_value()}))
}

fn digest_of(v: &Value) -> InputDigest {
    Sha256::digest(canonical::to_vec(v)).into()
}

fn hex(d: &InputDigest) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<InputDigest> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (nibble(b[2 * i])? << 4) | nibble(b[2 * i + 1])?;
    }
    Some(out)
}

/// A logged scope split into the input digest it carries and the scope; a
/// scope without the prefix (or with a malformed one, which is never written)
/// is returned whole, without a digest.
fn split_logged_scope(s: String) -> (Option<InputDigest>, String) {
    let parsed = s
        .strip_prefix(INPUT_TAG)
        .and_then(|rest| rest.split_at_checked(64))
        .and_then(|(h, rest)| Some((unhex(h)?, rest.strip_prefix('|')?)));
    match parsed {
        Some((d, scope)) => (Some(d), scope.to_string()),
        None => (None, s),
    }
}

/// Whether the answer to one question may answer another of the same scope
/// — the rule single flight shares with the lookup: the same input, or, with
/// near reuse on (`threshold` < 1), cos φ_P ≥ `threshold`.
pub fn reuses(
    threshold: f32,
    input: &InputDigest,
    phi_p: &[f32],
    other_input: &InputDigest,
    other_phi_p: &[f32],
) -> bool {
    input == other_input
        || (threshold < NEAR_OFF
            && phi_p.len() == other_phi_p.len()
            && dot(phi_p, other_phi_p) >= threshold)
}

/// One cached oracle answer.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheEntry {
    pub scope: String,
    /// The question's input digest ([`input_digest`]); `None` for an entry
    /// replayed from a record written before 0.8.10.
    pub input: Option<InputDigest>,
    pub phi_p: Vec<f32>,
    pub answer: Verdict,
    pub ts: u64,
}

impl CacheEntry {
    /// The scope as logged: `exact:<64 hex>|<scope>` with a digest, else the
    /// scope (see the module notes).
    fn logged_scope(&self) -> std::borrow::Cow<'_, str> {
        match &self.input {
            Some(d) => format!("{INPUT_TAG}{}|{}", hex(d), self.scope).into(),
            None => self.scope.as_str().into(),
        }
    }

    /// The `CachePut` payload (the verdict, not its distribution).
    pub(crate) fn encode(&self, e: &mut Enc) {
        e.str(&self.logged_scope());
        match &self.answer.answer {
            OracleAnswer::Choice(c) => {
                e.u8(0).str(c);
            }
            OracleAnswer::Score(s) => {
                e.u8(1).u32(*s);
            }
            OracleAnswer::Noul(b) => {
                e.u8(2).u8(u8::from(*b));
            }
        }
        e.u64(self.ts).f32s(&self.phi_p);
    }

    /// The `CachePutP` payload: the `CachePut` one, then the distribution
    /// (`u32` count, then `str` key and `f32` probability each).
    pub(crate) fn encode_with_probabilities(&self, e: &mut Enc) {
        self.encode(e);
        e.u32(self.answer.probabilities.len() as u32);
        for (k, p) in &self.answer.probabilities {
            e.str(k).f32(*p);
        }
    }

    /// Decode a `CachePutP` payload.
    pub(crate) fn decode_with_probabilities(d: &mut Dec<'_>) -> Result<Self> {
        let mut entry = Self::decode(d)?;
        let n = d.u32()? as usize;
        let mut probabilities = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            probabilities.push((d.str()?, d.f32()?));
        }
        entry.answer.probabilities = probabilities;
        Ok(entry)
    }

    /// Decode a `CachePut` payload (a one-hot verdict).
    pub(crate) fn decode(d: &mut Dec<'_>) -> Result<Self> {
        let (input, scope) = split_logged_scope(d.str()?);
        let answer = match d.u8()? {
            0 => OracleAnswer::Choice(d.str()?),
            1 => OracleAnswer::Score(d.u32()?),
            2 => OracleAnswer::Noul(match d.u8()? {
                0 => false,
                1 => true,
                v => bail!("bad boolean {v}"),
            }),
            t => bail!("unknown cached answer type {t}"),
        };
        Ok(Self {
            scope,
            input,
            answer: Verdict::one_hot(answer),
            ts: d.u64()?,
            phi_p: d.f32s()?,
        })
    }
}

/// `skill:<id>:<sha256(canonical(sorted option ids))>:<sha256(canonical(contract))>`.
pub fn skill_scope(skill: &str, q: &Question) -> String {
    let mut ids: Vec<&str> = q.options();
    ids.sort_unstable();
    let v = Value::Array(ids.into_iter().map(|s| Value::String(s.into())).collect());
    format!(
        "skill:{skill}:{}:{}",
        canonical::sha256_hex(&v),
        canonical::sha256_hex(&q.contract())
    )
}

/// `contract:<sha256(canonical({type, instructions, criteria}))>`.
pub fn contract_scope(q: &Question) -> String {
    format!("contract:{}", canonical::sha256_hex(&q.contract()))
}

/// The scope of a question (see the module notes).
pub fn scope_of(q: &Question, m: &SkillMatch) -> String {
    scope_of_as(q, m, false)
}

/// [`scope_of`] for a question whose instructions may be its input
/// (DESIGN A19.4): then the contract hashed is the state-less one
/// ([`Question::stateless_contract`], the instructions left out — they are
/// the text, whose φ_P and input digest the entry carries), so the same text
/// under the same criteria hits; otherwise exactly [`scope_of`].
pub fn scope_of_as(q: &Question, m: &SkillMatch, reads_instructions: bool) -> String {
    if !reads_instructions {
        return match &m.skill {
            Some(s) => skill_scope(s, q),
            None => contract_scope(q),
        };
    }
    let contract = canonical::sha256_hex(&q.stateless_contract());
    match &m.skill {
        Some(s) => {
            let mut ids: Vec<&str> = q.options();
            ids.sort_unstable();
            let v = Value::Array(ids.into_iter().map(|s| Value::String(s.into())).collect());
            format!("skill:{s}:{}:{contract}", canonical::sha256_hex(&v))
        }
        None => format!("contract:{contract}"),
    }
}

/// The cache (see the module notes).
#[derive(Debug)]
pub struct SemanticCache {
    /// The ring, oldest first; `entries[i]` has the sequence number `base + i`.
    entries: VecDeque<CacheEntry>,
    /// The sequence number of `entries[0]`.
    base: u64,
    /// The sequence numbers of each scope's entries, oldest first.
    index: HashMap<String, VecDeque<u64>>,
    threshold: f32,
    legacy_cos: f32,
    cap: usize,
    hits: u64,
    lookups: u64,
}

impl SemanticCache {
    /// A cache of `cap` entries; near reuse at cos ≥ `threshold` when it is
    /// below 1 ([`NEAR_OFF`]); digest-less entries at cos ≥ [`EXACT_COS`]
    /// ([`SemanticCache::with_legacy_cos`] changes it).
    pub fn new(threshold: f32, cap: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            base: 0,
            index: HashMap::new(),
            threshold,
            legacy_cos: EXACT_COS,
            cap: cap.max(1),
            hits: 0,
            lookups: 0,
        }
    }

    /// The cache of a configuration (`cache.threshold`, `cache.legacy_cos`,
    /// `cache.cap`).
    pub fn from_config(c: &CacheConfig) -> Self {
        Self::new(c.threshold, c.cap).with_legacy_cos(c.legacy_cos)
    }

    /// Digest-less entries answer at cos ≥ `cos`; 1 ([`LEGACY_OFF`]) turns
    /// them off (set it before the replay: they are then not kept).
    pub fn with_legacy_cos(mut self, cos: f32) -> Self {
        self.legacy_cos = cos;
        self
    }

    /// Whether near reuse is on (`cache.threshold` < 1).
    pub fn near_reuse(&self) -> bool {
        self.threshold < NEAR_OFF
    }

    /// Whether digest-less entries are kept and answer (`cache.legacy_cos`
    /// < 1).
    pub fn legacy_reuse(&self) -> bool {
        self.legacy_cos < LEGACY_OFF
    }

    /// The entries of `scope`, oldest first.
    fn scope_entries<'a>(&'a self, scope: &str) -> impl Iterator<Item = &'a CacheEntry> + 'a {
        let base = self.base;
        self.index
            .get(scope)
            .into_iter()
            .flatten()
            .map(move |&seq| &self.entries[(seq - base) as usize])
    }

    /// The best entry of `scope` by cos φ_P and its cos, hit or not, whatever
    /// the digests (a diagnostic; [`SemanticCache::find`] is the lookup).
    pub fn nearest(&self, scope: &str, phi_p: &[f32]) -> Option<(&CacheEntry, f32)> {
        let mut best: Option<(&CacheEntry, f32)> = None;
        for e in self.scope_entries(scope) {
            if e.phi_p.len() != phi_p.len() {
                continue;
            }
            let c = dot(&e.phi_p, phi_p);
            if best.is_none_or(|(_, b)| c > b) {
                best = Some((e, c));
            }
        }
        best
    }

    /// The entry that answers a question of `scope` with `input` and `phi_p`
    /// (the lookup of the module notes: exact, legacy, near) and its cos — 1
    /// for an exact match; nothing is counted.
    pub fn find(
        &self,
        scope: &str,
        input: &InputDigest,
        phi_p: &[f32],
    ) -> Option<(&CacheEntry, f32)> {
        let (near, old) = (self.near_reuse(), self.legacy_reuse());
        let mut legacy: Option<(&CacheEntry, f32)> = None;
        let mut nearest: Option<(&CacheEntry, f32)> = None;
        for e in self.scope_entries(scope) {
            if e.input.as_ref() == Some(input) {
                return Some((e, 1.0));
            }
            let usable = match e.input {
                Some(_) => near,
                None => old,
            };
            if !usable || e.phi_p.len() != phi_p.len() {
                continue;
            }
            let c = dot(&e.phi_p, phi_p);
            if e.input.is_none() && legacy.is_none_or(|(_, b)| c > b) {
                legacy = Some((e, c));
            }
            if near && nearest.is_none_or(|(_, b)| c > b) {
                nearest = Some((e, c));
            }
        }
        legacy
            .filter(|(_, c)| *c >= self.legacy_cos)
            .or_else(|| nearest.filter(|(_, c)| *c >= self.threshold))
    }

    /// A hit: the answer and its cos (counted in the statistics).
    pub fn get(
        &mut self,
        scope: &str,
        input: &InputDigest,
        phi_p: &[f32],
    ) -> Option<(Verdict, f32)> {
        self.lookups += 1;
        self.recheck(scope, input, phi_p)
    }

    /// A second lookup of the same question (e.g. under the single-flight lock):
    /// a hit is counted, the lookup is not (it was counted by [`SemanticCache::get`]).
    pub fn recheck(
        &mut self,
        scope: &str,
        input: &InputDigest,
        phi_p: &[f32],
    ) -> Option<(Verdict, f32)> {
        let found = self
            .find(scope, input, phi_p)
            .map(|(e, c)| (e.answer.clone(), c));
        if found.is_some() {
            self.hits += 1;
        }
        found
    }

    /// Store an answer; `false` when an entry already holds it (the same
    /// scope, input and verdict; for a digest-less entry the same scope and
    /// verdict, digest-less, at cos ≥ [`PUT_DEDUP`]) or when it is a
    /// digest-less entry and legacy reuse is off (it would answer nothing).
    pub fn put(&mut self, entry: CacheEntry) -> bool {
        if entry.input.is_none() && !self.legacy_reuse() {
            return false;
        }
        let dup = self.scope_entries(&entry.scope).any(|e| {
            e.answer.answer == entry.answer.answer
                && match (&e.input, &entry.input) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => {
                        e.phi_p.len() == entry.phi_p.len()
                            && dot(&e.phi_p, &entry.phi_p) >= PUT_DEDUP
                    }
                    _ => false,
                }
        });
        if dup {
            return false;
        }
        if self.entries.len() >= self.cap
            && let Some(old) = self.entries.pop_front()
        {
            // The globally oldest entry is the oldest of its scope.
            if let Some(seqs) = self.index.get_mut(&old.scope) {
                debug_assert_eq!(seqs.front(), Some(&self.base));
                seqs.pop_front();
                if seqs.is_empty() {
                    self.index.remove(&old.scope);
                }
            }
            self.base += 1;
        }
        let seq = self.base + self.entries.len() as u64;
        self.index
            .entry(entry.scope.clone())
            .or_default()
            .push_back(seq);
        self.entries.push_back(entry);
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn lookups(&self) -> u64 {
        self.lookups
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn legacy_cos(&self) -> f32 {
        self.legacy_cos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{CmfOptions, ModelRef, QuestionKind, State};

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// A test input digest.
    fn d(n: u8) -> InputDigest {
        [n; 32]
    }

    fn put_as(
        c: &mut SemanticCache,
        scope: &str,
        input: Option<InputDigest>,
        v: &[f32],
        label: &str,
    ) -> bool {
        c.put(CacheEntry {
            scope: scope.into(),
            input,
            phi_p: unit(v),
            answer: OracleAnswer::Choice(label.into()).into(),
            ts: 0,
        })
    }

    fn put(c: &mut SemanticCache, scope: &str, input: u8, v: &[f32], label: &str) -> bool {
        put_as(c, scope, Some(d(input)), v, label)
    }

    /// The label a lookup answers with.
    fn label(c: &mut SemanticCache, scope: &str, input: u8, v: &[f32]) -> Option<String> {
        c.get(scope, &d(input), &unit(v))
            .and_then(|(a, _)| a.label().map(str::to_string))
    }

    #[test]
    fn exact_repeats_hit_and_near_duplicates_do_not_by_default() {
        let mut c = SemanticCache::new(NEAR_OFF, 100);
        assert!(!c.near_reuse());
        assert!(put(&mut c, "a", 1, &[1.0, 0.0, 0.0], "x"));
        // The same question hits, whatever its φ_P.
        assert_eq!(
            label(&mut c, "a", 1, &[1.0, 0.0, 0.0]).as_deref(),
            Some("x")
        );
        assert_eq!(
            label(&mut c, "a", 1, &[0.0, 1.0, 0.0]).as_deref(),
            Some("x")
        );
        // Another state of the scope never does: not at cos 0.99, not with
        // the very same φ_P (two states that differ past the encoder's
        // window, or in a detail the encoder hardly sees).
        assert_eq!(label(&mut c, "a", 2, &[0.99, 0.05, 0.0]), None);
        assert_eq!(label(&mut c, "a", 2, &[1.0, 0.0, 0.0]), None);
        // The same input under another contract or option set: no hit.
        assert_eq!(label(&mut c, "b", 1, &[1.0, 0.0, 0.0]), None);
        // Several answers to one question: the earliest answers.
        assert!(put(&mut c, "a", 1, &[1.0, 0.0, 0.0], "y"));
        assert_eq!(
            label(&mut c, "a", 1, &[1.0, 0.0, 0.0]).as_deref(),
            Some("x")
        );
        assert_eq!((c.hits(), c.lookups()), (3, 6));
        // The cos of an exact hit is 1 by definition.
        assert_eq!(c.find("a", &d(1), &unit(&[0.0, 1.0, 0.0])).unwrap().1, 1.0);
    }

    #[test]
    fn near_reuse_is_opt_in_and_still_works() {
        let mut c = SemanticCache::new(0.97, 100);
        assert!(c.near_reuse());
        assert!(put(&mut c, "a", 1, &[1.0, 0.0, 0.0], "x"));
        // Another input at cos ≥ 0.97: a near hit; far or another scope: no.
        assert_eq!(
            label(&mut c, "a", 2, &[0.99, 0.05, 0.0]).as_deref(),
            Some("x")
        );
        assert_eq!(label(&mut c, "a", 3, &[0.0, 1.0, 0.0]), None);
        assert_eq!(label(&mut c, "b", 1, &[1.0, 0.0, 0.0]), None);
        assert_eq!((c.hits(), c.lookups()), (1, 3));
        // The same input and answer: skipped; another answer: kept; another
        // input with the same answer and φ_P: kept (the dedup is by input).
        assert!(!put(&mut c, "a", 1, &[1.0, 0.0001, 0.0], "x"));
        assert!(put(&mut c, "a", 1, &[1.0, 0.0001, 0.0], "y"));
        assert!(put(&mut c, "a", 2, &[1.0, 0.0, 0.0], "x"));
        assert_eq!(c.len(), 3);
        // An exact match wins over a nearer entry of another input.
        assert!(put(&mut c, "a", 4, &[0.0, 1.0, 0.0], "z"));
        assert_eq!(
            label(&mut c, "a", 4, &[1.0, 0.0, 0.0]).as_deref(),
            Some("z")
        );
    }

    /// Entries logged before 0.8.10 carry no digest: by default they answer
    /// at cos ≥ EXACT_COS, with near reuse on at the threshold, and their
    /// puts dedup as before.
    #[test]
    fn digest_less_entries_hit_at_the_exact_cos() {
        let mut c = SemanticCache::new(NEAR_OFF, 100);
        assert!(put_as(&mut c, "a", None, &[1.0, 0.0, 0.0], "x"));
        let close = [1.0, 0.01, 0.0]; // cos 0.99995
        let near = [1.0, 0.05, 0.0]; // cos 0.99875
        assert!(dot(&unit(&close), &unit(&[1.0, 0.0, 0.0])) >= EXACT_COS);
        assert_eq!(
            label(&mut c, "a", 5, &[1.0, 0.0, 0.0]).as_deref(),
            Some("x")
        );
        assert_eq!(label(&mut c, "a", 5, &close).as_deref(), Some("x"));
        assert_eq!(label(&mut c, "a", 5, &near), None);
        // The rule of 0.8.9 between digest-less entries.
        assert!(!put_as(&mut c, "a", None, &[1.0, 0.0001, 0.0], "x"));
        assert!(put_as(&mut c, "a", None, &[1.0, 0.0001, 0.0], "y"));
        // An exact entry of the same φ_P answers its own input only.
        assert!(put(&mut c, "a", 1, &[1.0, 0.0, 0.0], "z"));
        assert_eq!(
            label(&mut c, "a", 1, &[1.0, 0.0, 0.0]).as_deref(),
            Some("z")
        );
        assert_eq!(
            label(&mut c, "a", 2, &[1.0, 0.0, 0.0]).as_deref(),
            Some("x")
        );
        let mut n = SemanticCache::new(0.97, 100);
        put_as(&mut n, "a", None, &[1.0, 0.0, 0.0], "x");
        assert_eq!(label(&mut n, "a", 5, &near).as_deref(), Some("x"));
    }

    /// `cache.legacy_cos`: another cosine for digest-less entries, or 1 to
    /// turn them off — not kept, so they answer nothing, near reuse on or
    /// not, and an oracle answer to the same text gets an entry of its own.
    #[test]
    fn legacy_reuse_has_its_own_cosine_and_turns_off() {
        let near = [1.0, 0.05, 0.0]; // cos 0.99875
        let mut c = SemanticCache::new(NEAR_OFF, 100).with_legacy_cos(0.998);
        assert!(c.legacy_reuse());
        assert!(put_as(&mut c, "a", None, &[1.0, 0.0, 0.0], "x"));
        assert_eq!(label(&mut c, "a", 5, &near).as_deref(), Some("x"));
        assert_eq!(label(&mut c, "a", 5, &[1.0, 0.1, 0.0]), None);
        for threshold in [NEAR_OFF, 0.9] {
            let mut c = SemanticCache::new(threshold, 100).with_legacy_cos(LEGACY_OFF);
            assert!(!c.legacy_reuse());
            assert!(!put_as(&mut c, "a", None, &[1.0, 0.0, 0.0], "x"));
            assert!(c.is_empty());
            assert_eq!(label(&mut c, "a", 5, &[1.0, 0.0, 0.0]), None);
            // The same text asked again is a miss, so its oracle answer is
            // stored with its digest and answers it from then on.
            assert!(put(&mut c, "a", 5, &[1.0, 0.0, 0.0], "y"));
            assert_eq!(
                label(&mut c, "a", 5, &[1.0, 0.0, 0.0]).as_deref(),
                Some("y")
            );
        }
        let cfg = CacheConfig {
            legacy_cos: LEGACY_OFF,
            ..CacheConfig::default()
        };
        let c = SemanticCache::from_config(&cfg);
        assert_eq!((c.threshold(), c.legacy_cos()), (NEAR_OFF, LEGACY_OFF));
        let c = SemanticCache::from_config(&CacheConfig::default());
        assert_eq!((c.threshold(), c.legacy_cos()), (NEAR_OFF, EXACT_COS));
    }

    #[test]
    fn single_flight_reuses_by_the_same_rule() {
        let (a, b) = (unit(&[1.0, 0.0]), unit(&[0.99, 0.05]));
        assert!(reuses(NEAR_OFF, &d(1), &a, &d(1), &b));
        assert!(!reuses(NEAR_OFF, &d(1), &a, &d(2), &a));
        assert!(reuses(0.97, &d(1), &a, &d(2), &b));
        assert!(!reuses(0.97, &d(1), &a, &d(2), &unit(&[0.0, 1.0])));
        assert!(!reuses(0.97, &d(1), &a, &d(2), &unit(&[1.0, 0.0, 0.0])));
    }

    #[test]
    fn logged_scopes_carry_the_digest_and_old_records_replay_unchanged() {
        let e = CacheEntry {
            scope: "contract:c".into(),
            input: Some(d(0xab)),
            phi_p: vec![0.6, 0.8],
            answer: OracleAnswer::Choice("x".into()).into(),
            ts: 7,
        };
        let mut enc = Enc::default();
        e.encode(&mut enc);
        let mut want = Enc::default();
        want.str(&format!("exact:{}|contract:c", "ab".repeat(32)))
            .u8(0)
            .str("x")
            .u64(7)
            .f32s(&[0.6, 0.8]);
        assert_eq!(enc.0, want.0);
        let mut dec = Dec::new(&enc.0);
        assert_eq!(CacheEntry::decode(&mut dec).unwrap(), e);
        dec.finish().unwrap();
        // A 0.8.9 record: the scope whole, no digest.
        let old = CacheEntry {
            input: None,
            ..e.clone()
        };
        let mut enc = Enc::default();
        old.encode(&mut enc);
        let mut want = Enc::default();
        want.str("contract:c")
            .u8(0)
            .str("x")
            .u64(7)
            .f32s(&[0.6, 0.8]);
        assert_eq!(enc.0, want.0);
        assert_eq!(CacheEntry::decode(&mut Dec::new(&enc.0)).unwrap(), old);
        // A malformed prefix (never written) is a scope like any other.
        for s in [
            "exact:zz|contract:c".to_string(),
            format!("exact:{}contract:c", "ab".repeat(32)),
            format!("exact:{}|x", "AB".repeat(32)),
        ] {
            assert_eq!(split_logged_scope(s.clone()), (None, s));
        }
    }

    fn request(state: State, questions: Vec<Question>) -> DecisionRequest {
        DecisionRequest {
            model: ModelRef::Latest,
            state_text: state.text(),
            state,
            questions,
            cmf: CmfOptions::default(),
            user: None,
            session_id: None,
        }
    }

    fn question(id: &str, instructions: Value) -> Question {
        Question {
            id: id.into(),
            kind: QuestionKind::Choice,
            instructions,
            criteria: Some(json!({"a": null, "b": null})),
        }
    }

    #[test]
    fn input_digests_name_the_whole_input() {
        let qs = vec![
            question("p", json!("Which?")),
            question("q", json!("Other?")),
        ];
        let r = request(State::Json(json!({"x": 1, "y": [1, 2]})), qs.clone());
        // Every question of a state reads the same input, whatever its
        // instructions (they are in the scope).
        let s = state_digest(&r);
        assert_eq!(input_digest(&r, &qs[0]), s);
        assert_eq!(input_digest(&r, &qs[1]), s);
        assert_eq!(s, digest_of(&json!({"state": {"x": 1, "y": [1, 2]}})));
        // Key order does not matter (canonical JSON); a value does.
        let r2 = request(State::Json(json!({"y": [1, 2], "x": 1})), qs.clone());
        assert_eq!(state_digest(&r2), s);
        let r3 = request(State::Json(json!({"x": 1, "y": [1, 3]})), qs.clone());
        assert_ne!(state_digest(&r3), s);
        // A long state is hashed whole: a change at its end is another input.
        let long = "word ".repeat(2000);
        let a = request(State::Text(format!("{long}A")), qs.clone());
        let b = request(State::Text(format!("{long}B")), qs.clone());
        assert_ne!(state_digest(&a), state_digest(&b));
        // A state-less question reads its instructions, its lead-in line
        // included (the scope has only the criteria).
        let sl = vec![
            question("p", json!("Is it spam?:\nhello there")),
            question("q", json!("Is it English?:\nhello there")),
        ];
        let r = request(State::Json(json!({})), sl.clone());
        assert!(r.reads_instructions(&sl[0]));
        let (p, q) = (input_digest(&r, &sl[0]), input_digest(&r, &sl[1]));
        assert_ne!(p, q);
        assert_eq!(
            p,
            digest_of(&json!({"instructions": "Is it spam?:\nhello there"}))
        );
        assert_ne!(p, state_digest(&r));
    }

    #[test]
    fn ring_evicts_the_oldest() {
        let mut c = SemanticCache::new(0.99, 2);
        put(&mut c, "s", 1, &[1.0, 0.0], "a");
        put(&mut c, "s", 2, &[0.0, 1.0], "b");
        put(&mut c, "s", 3, &[-1.0, 0.0], "c");
        assert_eq!(c.len(), 2);
        assert!(c.get("s", &d(1), &unit(&[1.0, 0.0])).is_none());
        assert!(c.get("s", &d(2), &unit(&[0.0, 1.0])).is_some());
    }

    /// The cache as one scan of the whole ring per lookup and put (the
    /// pre-0.8.8 structure, with the 0.8.10 rules).
    struct Linear {
        entries: VecDeque<CacheEntry>,
        threshold: f32,
        legacy_cos: f32,
        cap: usize,
    }

    impl Linear {
        fn best(
            &self,
            scope: &str,
            phi_p: &[f32],
            digest_less: bool,
        ) -> Option<(&CacheEntry, f32)> {
            let mut best: Option<(&CacheEntry, f32)> = None;
            for e in &self.entries {
                if e.scope != scope
                    || e.phi_p.len() != phi_p.len()
                    || (digest_less && e.input.is_some())
                {
                    continue;
                }
                let c = dot(&e.phi_p, phi_p);
                if best.is_none_or(|(_, b)| c > b) {
                    best = Some((e, c));
                }
            }
            best
        }

        fn find(
            &self,
            scope: &str,
            input: &InputDigest,
            phi_p: &[f32],
        ) -> Option<(&CacheEntry, f32)> {
            if let Some(e) = self
                .entries
                .iter()
                .find(|e| e.scope == scope && e.input.as_ref() == Some(input))
            {
                return Some((e, 1.0));
            }
            if let Some(hit) = self
                .best(scope, phi_p, true)
                .filter(|(_, c)| *c >= self.legacy_cos)
            {
                return Some(hit);
            }
            if self.threshold < NEAR_OFF {
                return self
                    .best(scope, phi_p, false)
                    .filter(|(_, c)| *c >= self.threshold);
            }
            None
        }

        fn put(&mut self, entry: CacheEntry) -> bool {
            if entry.input.is_none() && self.legacy_cos >= LEGACY_OFF {
                return false;
            }
            let dup = self.entries.iter().any(|e| {
                e.scope == entry.scope
                    && e.answer.answer == entry.answer.answer
                    && match (&e.input, &entry.input) {
                        (Some(a), Some(b)) => a == b,
                        (None, None) => {
                            e.phi_p.len() == entry.phi_p.len()
                                && dot(&e.phi_p, &entry.phi_p) >= PUT_DEDUP
                        }
                        _ => false,
                    }
            });
            if dup {
                return false;
            }
            if self.entries.len() >= self.cap {
                self.entries.pop_front();
            }
            self.entries.push_back(entry);
            true
        }
    }

    /// The indexed cache answers, dedups and evicts exactly like the linear
    /// one (DESIGN B3) on random operations: few scopes, inputs and answers
    /// (exact hits and dedups), digest-less entries (the legacy rule), coarse
    /// vectors (exact duplicates make ties), two dimensions (a φ of another
    /// length is never compared), small rings (evictions), near reuse on in
    /// every other round, legacy reuse off in every third.
    #[test]
    fn indexed_cache_equals_the_linear_one() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for round in 0..40 {
            let cap = 1 + next(24) as usize;
            let threshold = if round % 2 == 0 { 0.97 } else { NEAR_OFF };
            let legacy_cos = if round % 3 == 0 {
                LEGACY_OFF
            } else {
                EXACT_COS
            };
            let mut fast = SemanticCache::new(threshold, cap).with_legacy_cos(legacy_cos);
            let mut slow = Linear {
                entries: VecDeque::new(),
                threshold,
                legacy_cos,
                cap,
            };
            let scopes = 1 + next(5);
            for step in 0..600 {
                let scope = format!("s{}", next(scopes));
                let dim = if next(10) == 0 { 2 } else { 3 };
                let v: Vec<f32> = (0..dim).map(|_| next(4) as f32 - 1.0).collect();
                if v.iter().all(|x| *x == 0.0) {
                    continue;
                }
                let v = unit(&v);
                let n = next(5) as u8;
                if next(2) == 0 {
                    let e = CacheEntry {
                        scope,
                        input: (n > 0).then(|| d(n)),
                        phi_p: v,
                        answer: OracleAnswer::Choice(format!("l{}", next(3))).into(),
                        ts: step,
                    };
                    assert_eq!(
                        fast.put(e.clone()),
                        slow.put(e),
                        "round {round} step {step}"
                    );
                } else {
                    let input = d(n);
                    let want = slow.find(&scope, &input, &v).map(|(e, c)| (e.ts, c));
                    assert_eq!(
                        fast.find(&scope, &input, &v).map(|(e, c)| (e.ts, c)),
                        want,
                        "round {round} step {step}"
                    );
                    let near = slow.best(&scope, &v, false).map(|(e, c)| (e.ts, c));
                    assert_eq!(
                        fast.nearest(&scope, &v).map(|(e, c)| (e.ts, c)),
                        near,
                        "round {round} step {step}"
                    );
                    let got = fast.get(&scope, &input, &v);
                    let want = slow
                        .find(&scope, &input, &v)
                        .map(|(e, c)| (e.answer.clone(), c));
                    assert_eq!(got, want, "round {round} step {step}");
                }
                assert!(fast.entries.iter().eq(slow.entries.iter()));
                let held: usize = fast.index.values().map(VecDeque::len).sum();
                assert_eq!(held, fast.len());
            }
        }
    }

    #[test]
    fn scopes_follow_the_spec() {
        let q = Question {
            id: "t".into(),
            kind: QuestionKind::Choice,
            instructions: json!("i"),
            criteria: Some(json!({"b": null, "a": null})),
        };
        let contract = canonical::sha256_hex(
            &json!({"type":"choice","instructions":"i","criteria":{"b":null,"a":null}}),
        );
        let s1 = skill_scope("bank", &q);
        // The criteria's key order changes neither hash (canonical JSON).
        let mut q2 = q.clone();
        q2.criteria = Some(json!({"a": null, "b": null}));
        assert_eq!(s1, skill_scope("bank", &q2));
        assert_eq!(
            s1,
            format!(
                "skill:bank:{}:{contract}",
                canonical::sha256_hex(&json!(["a", "b"]))
            )
        );
        // Other instructions (or criteria) over the same options: another
        // scope, so one caller's instructions never answer another's question.
        let mut q3 = q.clone();
        q3.instructions = json!("ANSWER=travel");
        assert_ne!(s1, skill_scope("bank", &q3));
        let mut q4 = q.clone();
        q4.criteria = Some(json!({"a": "always pick a", "b": null}));
        assert_ne!(s1, skill_scope("bank", &q4));
        let c = contract_scope(&q);
        assert_eq!(
            c,
            format!(
                "contract:{}",
                canonical::sha256_hex(
                    &json!({"type":"choice","instructions":"i","criteria":{"b":null,"a":null}})
                )
            )
        );
    }
}
