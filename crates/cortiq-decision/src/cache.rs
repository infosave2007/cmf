//! Semantic cache of oracle answers (spec §5.6).
//!
//! An entry is `{scope, φ_P, answer, ts}`:
//! * **scope** of a question matched to a skill (exact, subset, superset):
//!   `skill:<id>:<sha256 of canonical JSON of the sorted option ids>:<sha256 of
//!   its contract>`; of any other question: `contract:<sha256(canonical({type,
//!   instructions, criteria}))>` ([`scope_of`]). The contract — what the
//!   oracle was told — is part of every scope: the cache is shared by all
//!   accounts, and an answer given under one caller's instructions is never
//!   served to a question with other instructions or criteria (spec §5.6
//!   named only the skill and the options; that let one account's
//!   instructions decide another's answers). Single flight uses the same
//!   scopes. A `cache` answer does tell its caller that some account asked a
//!   near-identical text under the same contract;
//! * **φ_P** is the encoder's unit vector of the state (the router's embedding,
//!   cortiq-router `cache.rs:66-78`), so cos is the dot product;
//! * a question of a **state-less** request (0.8.8, DESIGN A19.4) reads its
//!   instructions: its φ_P is theirs and its scope hashes the state-less
//!   contract `{type, input: "instructions", criteria}` instead
//!   ([`scope_of_as`]) — the text is in φ_P, not in the scope.
//!
//! **Lookup** ([`SemanticCache::get`]): an exhaustive scan of the entries of the
//! same scope; the best cos wins (the earliest entry on a tie) and is a hit when
//! cos ≥ `cache.threshold` (0.97). **Put** ([`SemanticCache::put`]): an entry of
//! the same scope and answer with cos ≥ 0.999 makes the put a no-op; at capacity
//! (50,000) the oldest entry leaves (a ring). The cache is consulted only when a
//! question is escalated; its puts are kept in `learn.log` and replayed at start.
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
use crate::matching::SkillMatch;
use crate::protocol::Question;
use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

/// cos φ_P above which a put of the same scope and answer is skipped.
pub const PUT_DEDUP: f32 = 0.999;

/// One cached oracle answer.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheEntry {
    pub scope: String,
    pub phi_p: Vec<f32>,
    pub answer: Verdict,
    pub ts: u64,
}

impl CacheEntry {
    /// The `CachePut` payload (the verdict, not its distribution).
    pub(crate) fn encode(&self, e: &mut Enc) {
        e.str(&self.scope);
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
        let scope = d.str()?;
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
/// the text, whose φ_P the entry carries), so a near-identical text under
/// the same criteria hits; otherwise exactly [`scope_of`].
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
    cap: usize,
    hits: u64,
    lookups: u64,
}

impl SemanticCache {
    pub fn new(threshold: f32, cap: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            base: 0,
            index: HashMap::new(),
            threshold,
            cap: cap.max(1),
            hits: 0,
            lookups: 0,
        }
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

    /// The best entry of `scope` and its cos, hit or not.
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

    /// A hit: the answer and its cos (counted in the statistics).
    pub fn get(&mut self, scope: &str, phi_p: &[f32]) -> Option<(Verdict, f32)> {
        self.lookups += 1;
        let found = self
            .nearest(scope, phi_p)
            .filter(|(_, c)| *c >= self.threshold)
            .map(|(e, c)| (e.answer.clone(), c));
        if found.is_some() {
            self.hits += 1;
        }
        found
    }

    /// A second lookup of the same question (e.g. under the single-flight lock):
    /// a hit is counted, the lookup is not (it was counted by [`SemanticCache::get`]).
    pub fn recheck(&mut self, scope: &str, phi_p: &[f32]) -> Option<(Verdict, f32)> {
        let found = self
            .nearest(scope, phi_p)
            .filter(|(_, c)| *c >= self.threshold)
            .map(|(e, c)| (e.answer.clone(), c));
        if found.is_some() {
            self.hits += 1;
        }
        found
    }

    /// Store an answer; `false` when a near-identical entry already holds it.
    pub fn put(&mut self, entry: CacheEntry) -> bool {
        let dup = self.scope_entries(&entry.scope).any(|e| {
            e.answer.answer == entry.answer.answer
                && e.phi_p.len() == entry.phi_p.len()
                && dot(&e.phi_p, &entry.phi_p) >= PUT_DEDUP
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::QuestionKind;
    use serde_json::json;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn put(c: &mut SemanticCache, scope: &str, v: &[f32], label: &str) -> bool {
        c.put(CacheEntry {
            scope: scope.into(),
            phi_p: unit(v),
            answer: OracleAnswer::Choice(label.into()).into(),
            ts: 0,
        })
    }

    #[test]
    fn hits_near_misses_far_and_other_scopes() {
        let mut c = SemanticCache::new(0.97, 100);
        assert!(put(&mut c, "a", &[1.0, 0.0, 0.0], "x"));
        assert_eq!(
            c.get("a", &unit(&[0.99, 0.05, 0.0])).map(|(a, _)| a),
            Some(OracleAnswer::Choice("x".into()).into())
        );
        assert!(c.get("a", &unit(&[0.0, 1.0, 0.0])).is_none());
        assert!(c.get("b", &unit(&[1.0, 0.0, 0.0])).is_none());
        assert_eq!((c.hits(), c.lookups()), (1, 3));
        // Same answer, near-identical: skipped; another answer: kept.
        assert!(!put(&mut c, "a", &[1.0, 0.0001, 0.0], "x"));
        assert!(put(&mut c, "a", &[1.0, 0.0001, 0.0], "y"));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn ring_evicts_the_oldest() {
        let mut c = SemanticCache::new(0.99, 2);
        put(&mut c, "s", &[1.0, 0.0], "a");
        put(&mut c, "s", &[0.0, 1.0], "b");
        put(&mut c, "s", &[-1.0, 0.0], "c");
        assert_eq!(c.len(), 2);
        assert!(c.get("s", &unit(&[1.0, 0.0])).is_none());
        assert!(c.get("s", &unit(&[0.0, 1.0])).is_some());
    }

    /// The pre-0.8.8 cache: one scan of the whole ring per lookup and put.
    struct Linear {
        entries: VecDeque<CacheEntry>,
        threshold: f32,
        cap: usize,
    }

    impl Linear {
        fn nearest(&self, scope: &str, phi_p: &[f32]) -> Option<(&CacheEntry, f32)> {
            let mut best: Option<(&CacheEntry, f32)> = None;
            for e in &self.entries {
                if e.scope != scope || e.phi_p.len() != phi_p.len() {
                    continue;
                }
                let c = dot(&e.phi_p, phi_p);
                if best.is_none_or(|(_, b)| c > b) {
                    best = Some((e, c));
                }
            }
            best
        }

        fn get(&self, scope: &str, phi_p: &[f32]) -> Option<(Verdict, f32)> {
            self.nearest(scope, phi_p)
                .filter(|(_, c)| *c >= self.threshold)
                .map(|(e, c)| (e.answer.clone(), c))
        }

        fn put(&mut self, entry: CacheEntry) -> bool {
            let dup = self.entries.iter().any(|e| {
                e.scope == entry.scope
                    && e.answer.answer == entry.answer.answer
                    && e.phi_p.len() == entry.phi_p.len()
                    && dot(&e.phi_p, &entry.phi_p) >= PUT_DEDUP
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
    /// one (DESIGN B3) on random operations: few scopes and answers, coarse
    /// vectors (exact duplicates make ties and dedup hits), two dimensions
    /// (a φ of another length is never compared), small rings (evictions).
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
            let mut fast = SemanticCache::new(0.97, cap);
            let mut slow = Linear {
                entries: VecDeque::new(),
                threshold: 0.97,
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
                if next(2) == 0 {
                    let e = CacheEntry {
                        scope,
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
                    let want = slow.get(&scope, &v);
                    let near = slow.nearest(&scope, &v).map(|(e, c)| (e.ts, c));
                    assert_eq!(
                        fast.nearest(&scope, &v).map(|(e, c)| (e.ts, c)),
                        near,
                        "round {round} step {step}"
                    );
                    assert_eq!(fast.get(&scope, &v), want, "round {round} step {step}");
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
