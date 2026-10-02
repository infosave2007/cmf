//! Learning buffer of oracle and feedback examples, and `learn.log` (spec §5.7).
//!
//! **Example** ([`Example`]): the vectors of the state the model saw — φ_P and
//! the sparse φ_H — with its skill, label, source and weight; never text or a
//! hash of it. Weights are recorded (client_feedback 3.0, oracle 1.0, agreement
//! 0.5) and do not enter the fit (the router's fit is unweighted).
//!
//! **Dedup** ([`LearningBuffer::add`]): an example is refused when cos φ_P ≥
//! `learning.dedup` (0.995) with any example of the same skill and label or with
//! any stored row of that label (the caller passes the base and learned rows).
//! φ_P is unit length, so cos is the dot product (f32, in index order).
//!
//! **Counters**: examples added per (skill, label) since the last learning
//! attempt of that label; an attempt resets its label, a rollback every label.
//!
//! **Contracts** ([`Contract`], [`ContractRegistry`], 0.8.6): an untrained
//! choice question whose oracle answers are learned into an auto-skill
//! (`crate::manifest::auto_skill_id`) is registered once, before its first
//! example, as the caller first sent it — option ids in request order,
//! instructions and criteria (the auto-skill's rubric). The registry is
//! rebuilt from the log before the examples, so an example of an auto-skill
//! always finds its contract (one without is dropped with a warning: it cannot
//! happen in a consistent log).
//!
//! **`learn.log`** ([`LearnLog`]): one record per cache put, example, attempt,
//! rollback and contract, `magic "CDLG" | u32 len | u8 kind | payload[len] |
//! u32 crc32` (little-endian; the crc covers `len`, `kind` and the payload),
//! fsynced one by one. On open, a tail that is cut or fails its crc is dropped
//! and the file is truncated to the last whole record; replaying the records
//! restores the cache, the buffer, the counters and the contract registry.
//! A record of a kind this binary does not know ends the replay the same way
//! (a 0.8.5 binary on a 0.8.6 state directory truncates the log at its first
//! contract record — never run an older binary on it).

use crate::cache::CacheEntry;
use crate::canonical;
use crate::manifest::{self, Rubric};
use crate::rows::{Row, Source, Split};
use crate::signal::Features;
use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Magic of a `learn.log` record.
pub const LOG_MAGIC: [u8; 4] = *b"CDLG";
/// Largest payload of one record (64 MiB).
pub const MAX_RECORD: usize = 64 << 20;

const KIND_CACHE_PUT: u8 = 1;
const KIND_EXAMPLE: u8 = 2;
const KIND_ATTEMPT: u8 = 3;
const KIND_ROLLBACK: u8 = 4;
/// 0.8.6: the contract of an auto-skill (a 0.8.5 binary stops replaying here).
const KIND_CONTRACT: u8 = 5;

// ------------------------------------------------------------------ codec

/// Little-endian payload writer.
#[derive(Debug, Default)]
pub struct Enc(pub Vec<u8>);

impl Enc {
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
        self
    }
    pub fn f32s(&mut self, v: &[f32]) -> &mut Self {
        self.u32(v.len() as u32);
        for x in v {
            self.f32(*x);
        }
        self
    }
    pub fn u16s(&mut self, v: &[u16]) -> &mut Self {
        self.u32(v.len() as u32);
        for x in v {
            self.u16(*x);
        }
        self
    }
}

/// Little-endian payload reader.
#[derive(Debug)]
pub struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "record payload is short");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(std::str::from_utf8(self.take(n)?)?.to_string())
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.u32()? as usize;
        ensure!(n <= MAX_RECORD / 4, "vector too long");
        (0..n).map(|_| self.f32()).collect()
    }
    pub fn u16s(&mut self) -> Result<Vec<u16>> {
        let n = self.u32()? as usize;
        ensure!(n <= MAX_RECORD / 2, "vector too long");
        (0..n).map(|_| self.u16()).collect()
    }
    /// Every byte was read.
    pub fn finish(&self) -> Result<()> {
        ensure!(self.p == self.b.len(), "record payload has trailing bytes");
        Ok(())
    }
}

// ------------------------------------------------------------------ examples

/// The dot product in index order (the cosine of unit vectors).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// One learned example (vectors only).
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub skill: String,
    pub label: String,
    pub source: Source,
    pub weight: f32,
    pub ts: u64,
    pub phi_p: Vec<f32>,
    /// Strictly ascending indices of the non-zero φ_H entries.
    pub h_idx: Vec<u16>,
    pub h_val: Vec<f32>,
}

impl Example {
    /// An example from the features of a state, with the source's weight.
    pub fn from_features(
        skill: &str,
        label: &str,
        source: Source,
        features: &Features,
        ts: u64,
    ) -> Self {
        let (h_idx, h_val) = crate::rows::sparse_from_dense(&features.phi_h);
        Self {
            skill: skill.into(),
            label: label.into(),
            source,
            weight: source.default_weight(),
            ts,
            phi_p: features.phi_p.clone(),
            h_idx,
            h_val,
        }
    }

    /// The learned row of this example for task `task`.
    pub fn to_row(&self, task: u32) -> Row {
        Row {
            task,
            split: Split::Learned,
            flags: 0,
            source: self.source,
            weight: self.weight,
            phi_p: self.phi_p.clone(),
            h_idx: self.h_idx.clone(),
            h_val: self.h_val.clone(),
        }
    }

    fn encode(&self, e: &mut Enc) {
        e.str(&self.skill)
            .str(&self.label)
            .u8(self.source as u8)
            .f32(self.weight)
            .u64(self.ts)
            .f32s(&self.phi_p)
            .u16s(&self.h_idx)
            .f32s(&self.h_val);
    }

    fn decode(d: &mut Dec<'_>) -> Result<Self> {
        let ex = Self {
            skill: d.str()?,
            label: d.str()?,
            source: Source::from_u8(d.u8()?)?,
            weight: d.f32()?,
            ts: d.u64()?,
            phi_p: d.f32s()?,
            h_idx: d.u16s()?,
            h_val: d.f32s()?,
        };
        ensure!(
            ex.h_idx.len() == ex.h_val.len(),
            "example φ_H indices and values differ in length"
        );
        ensure!(
            ex.h_idx.windows(2).all(|w| w[0] < w[1]),
            "example φ_H indices are not ascending"
        );
        Ok(ex)
    }
}

// ------------------------------------------------------------------ contracts

/// A choice contract learned into an auto-skill, as first seen (DESIGN D1).
#[derive(Clone, Debug, PartialEq)]
pub struct Contract {
    /// The auto-skill id, [`manifest::auto_skill_id`] of `ids`.
    pub skill: String,
    /// The option ids in the first request's order.
    pub ids: Vec<String>,
    /// The first request's `instructions` (string, object, array or null).
    pub instructions: Value,
    /// The first request's `criteria` object (descriptions kept, `null` too).
    pub criteria: Value,
    pub created_unix: u64,
}

impl Contract {
    /// The contract of a choice question with `ids` as its options.
    pub fn new(ids: &[&str], instructions: &Value, criteria: &Value, created_unix: u64) -> Self {
        Self {
            skill: manifest::auto_skill_id(ids),
            ids: ids.iter().map(|s| s.to_string()).collect(),
            instructions: instructions.clone(),
            criteria: criteria.clone(),
            created_unix,
        }
    }

    /// Whether `label` is one of the contract's option ids (the closed label
    /// set of its auto-skill, DESIGN A5).
    pub fn has_label(&self, label: &str) -> bool {
        self.ids.iter().any(|i| i == label)
    }

    /// The auto-skill's rubric: the instructions as sent when they are a
    /// string, else their canonical JSON text (`Rubric.instructions` is a
    /// string); the criteria as sent (DESIGN D4).
    pub fn rubric(&self) -> Rubric {
        let instructions = match &self.instructions {
            Value::String(s) => s.clone(),
            other => canonical::to_string(other),
        };
        // Criteria in the ids' (request) order whatever the map's order after
        // a replay (the record stores the canonical, key-sorted text).
        let criteria: Map<String, Value> = self
            .ids
            .iter()
            .map(|i| {
                (
                    i.clone(),
                    self.criteria.get(i).cloned().unwrap_or(Value::Null),
                )
            })
            .collect();
        Rubric::new(instructions, criteria)
    }

    fn encode(&self, e: &mut Enc) {
        e.str(&self.skill).u64(self.created_unix);
        e.u32(self.ids.len() as u32);
        for i in &self.ids {
            e.str(i);
        }
        e.str(&canonical::to_string(&self.instructions));
        e.str(&canonical::to_string(&self.criteria));
    }

    fn decode(d: &mut Dec<'_>) -> Result<Self> {
        let skill = d.str()?;
        let created_unix = d.u64()?;
        let n = d.u32()? as usize;
        ensure!(
            n <= crate::protocol::MAX_CHOICE_OPTIONS,
            "too many contract ids"
        );
        let ids = (0..n).map(|_| d.str()).collect::<Result<Vec<_>>>()?;
        let instructions = canonical::parse(d.str()?.as_bytes())?;
        let criteria = canonical::parse(d.str()?.as_bytes())?;
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        ensure!(
            skill == manifest::auto_skill_id(&refs),
            "contract record: the skill id does not match its option ids"
        );
        Ok(Self {
            skill,
            ids,
            instructions,
            criteria,
            created_unix,
        })
    }
}

/// The contracts learned so far, by auto-skill id (first seen wins).
#[derive(Clone, Debug, Default)]
pub struct ContractRegistry {
    contracts: BTreeMap<String, Contract>,
}

impl ContractRegistry {
    pub fn get(&self, skill: &str) -> Option<&Contract> {
        self.contracts.get(skill)
    }

    pub fn contains(&self, skill: &str) -> bool {
        self.contracts.contains_key(skill)
    }

    pub fn len(&self) -> usize {
        self.contracts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }

    /// Register a contract unless its id is known; whether it was new.
    pub fn insert(&mut self, c: Contract) -> bool {
        if self.contracts.contains_key(&c.skill) {
            return false;
        }
        self.contracts.insert(c.skill.clone(), c);
        true
    }

    /// Every contract, by id.
    pub fn iter(&self) -> impl Iterator<Item = &Contract> {
        self.contracts.values()
    }
}

/// A learning attempt (resets the counter of its label).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    pub skill: String,
    pub label: String,
    /// `promoted`, `rejected`, `skipped`, …
    pub outcome: String,
    /// The generation written by a promotion (else the served one).
    pub generation: u64,
}

/// One record of `learn.log`.
#[derive(Clone, Debug, PartialEq)]
pub enum LogRecord {
    CachePut(CacheEntry),
    Example(Example),
    Attempt(AttemptRecord),
    /// A rollback to `generation` (resets every counter).
    Rollback {
        generation: u64,
    },
    /// The contract of an auto-skill, written before its first example (0.8.6).
    Contract(Contract),
}

impl LogRecord {
    fn encode(&self) -> (u8, Vec<u8>) {
        let mut e = Enc::default();
        let kind = match self {
            LogRecord::CachePut(c) => {
                c.encode(&mut e);
                KIND_CACHE_PUT
            }
            LogRecord::Example(x) => {
                x.encode(&mut e);
                KIND_EXAMPLE
            }
            LogRecord::Attempt(a) => {
                e.str(&a.skill)
                    .str(&a.label)
                    .str(&a.outcome)
                    .u64(a.generation);
                KIND_ATTEMPT
            }
            LogRecord::Rollback { generation } => {
                e.u64(*generation);
                KIND_ROLLBACK
            }
            LogRecord::Contract(c) => {
                c.encode(&mut e);
                KIND_CONTRACT
            }
        };
        (kind, e.0)
    }

    fn decode(kind: u8, payload: &[u8]) -> Result<Self> {
        let mut d = Dec::new(payload);
        let r = match kind {
            KIND_CACHE_PUT => LogRecord::CachePut(CacheEntry::decode(&mut d)?),
            KIND_EXAMPLE => LogRecord::Example(Example::decode(&mut d)?),
            KIND_ATTEMPT => LogRecord::Attempt(AttemptRecord {
                skill: d.str()?,
                label: d.str()?,
                outcome: d.str()?,
                generation: d.u64()?,
            }),
            KIND_ROLLBACK => LogRecord::Rollback {
                generation: d.u64()?,
            },
            KIND_CONTRACT => LogRecord::Contract(Contract::decode(&mut d)?),
            k => bail!("unknown learn.log record kind {k}"),
        };
        d.finish()?;
        Ok(r)
    }

    /// The framed bytes of the record.
    pub fn frame(&self) -> Vec<u8> {
        let (kind, payload) = self.encode();
        let len = (payload.len() as u32).to_le_bytes();
        let mut h = crc32fast::Hasher::new();
        h.update(&len);
        h.update(&[kind]);
        h.update(&payload);
        let crc = h.finalize();
        let mut out = Vec::with_capacity(payload.len() + 13);
        out.extend_from_slice(&LOG_MAGIC);
        out.extend_from_slice(&len);
        out.push(kind);
        out.extend_from_slice(&payload);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }
}

/// Decode framed records; returns the records and the length of the valid prefix.
pub fn read_records(bytes: &[u8]) -> (Vec<LogRecord>, usize) {
    let mut out = Vec::new();
    let mut p = 0usize;
    loop {
        let rest = &bytes[p..];
        if rest.len() < 13 || rest[..4] != LOG_MAGIC {
            break;
        }
        let len = u32::from_le_bytes(rest[4..8].try_into().expect("4 bytes")) as usize;
        if len > MAX_RECORD || rest.len() < 13 + len {
            break;
        }
        let kind = rest[8];
        let payload = &rest[9..9 + len];
        let crc = u32::from_le_bytes(rest[9 + len..13 + len].try_into().expect("4 bytes"));
        let mut h = crc32fast::Hasher::new();
        h.update(&rest[4..8]);
        h.update(&[kind]);
        h.update(payload);
        if h.finalize() != crc {
            break;
        }
        match LogRecord::decode(kind, payload) {
            Ok(r) => out.push(r),
            Err(_) => break,
        }
        p += 13 + len;
    }
    (out, p)
}

/// The append-only `learn.log`.
#[derive(Debug)]
pub struct LearnLog {
    path: PathBuf,
    file: Mutex<File>,
}

/// What [`LearnLog::open`] found.
#[derive(Debug)]
pub struct Replayed {
    pub records: Vec<LogRecord>,
    /// Bytes of a cut or corrupt tail that were dropped.
    pub dropped_bytes: u64,
}

impl LearnLog {
    /// Open (create) the log, drop a bad tail and return the records.
    pub fn open(path: &Path) -> Result<(Self, Replayed)> {
        let mut bytes = Vec::new();
        match File::open(path) {
            Ok(mut f) => {
                f.read_to_end(&mut bytes)
                    .with_context(|| format!("read {}", path.display()))?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("open {}", path.display())),
        }
        let (records, valid) = read_records(&bytes);
        let dropped = (bytes.len() - valid) as u64;
        let mut o = OpenOptions::new();
        o.create(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let file = o
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        if dropped > 0 {
            tracing::warn!(
                bytes = dropped,
                records = records.len(),
                "learn.log: dropping a cut or corrupt tail"
            );
            file.set_len(valid as u64)
                .with_context(|| format!("truncate {}", path.display()))?;
            file.sync_all()?;
        }
        let mut file = file;
        use std::io::Seek;
        file.seek(std::io::SeekFrom::End(0))?;
        Ok((
            Self {
                path: path.to_path_buf(),
                file: Mutex::new(file),
            },
            Replayed {
                records,
                dropped_bytes: dropped,
            },
        ))
    }

    /// Append one record and fsync it.
    pub fn append(&self, r: &LogRecord) -> Result<()> {
        let frame = r.frame();
        let mut f = self.file.lock();
        f.write_all(&frame)
            .and_then(|()| f.sync_data())
            .with_context(|| format!("append {}", self.path.display()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ------------------------------------------------------------------ buffer

type Key = (String, String);

/// Per-label summary of the buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelCount {
    pub skill: String,
    pub label: String,
    /// Examples kept.
    pub total: usize,
    /// Examples added since the last attempt.
    pub new: usize,
}

/// Why [`LearningBuffer::add`] refused an example.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddOutcome {
    Stored,
    Duplicate,
    /// Refused by a buffer limit of the cascade
    /// ([`crate::cascade::MAX_EXAMPLES_PER_LABEL`],
    /// [`crate::cascade::MAX_PENDING_NEW_LABELS`]).
    Full,
}

/// The examples in memory (restored from `learn.log`).
#[derive(Debug)]
pub struct LearningBuffer {
    dedup: f32,
    examples: Vec<Example>,
    by_label: HashMap<Key, Vec<usize>>,
    new_since: HashMap<Key, usize>,
    duplicates: u64,
}

impl LearningBuffer {
    pub fn new(dedup: f32) -> Self {
        Self {
            dedup,
            examples: Vec::new(),
            by_label: HashMap::new(),
            new_since: HashMap::new(),
            duplicates: 0,
        }
    }

    fn key(skill: &str, label: &str) -> Key {
        (skill.to_string(), label.to_string())
    }

    /// cos φ_P ≥ dedup with an example of (skill, label) or one of `stored`.
    pub fn is_duplicate(&self, skill: &str, label: &str, phi_p: &[f32], stored: &[&[f32]]) -> bool {
        let near = |v: &[f32]| v.len() == phi_p.len() && dot(v, phi_p) >= self.dedup;
        if stored.iter().any(|v| near(v)) {
            return true;
        }
        self.by_label
            .get(&Self::key(skill, label))
            .is_some_and(|ix| ix.iter().any(|&i| near(&self.examples[i].phi_p)))
    }

    /// Keep an example (no dedup) and count it as new.
    pub fn insert(&mut self, ex: Example) {
        let k = Self::key(&ex.skill, &ex.label);
        self.by_label
            .entry(k.clone())
            .or_default()
            .push(self.examples.len());
        *self.new_since.entry(k).or_insert(0) += 1;
        self.examples.push(ex);
    }

    /// Count a refused duplicate (the caller checked [`LearningBuffer::is_duplicate`]).
    pub fn count_duplicate(&mut self) {
        self.duplicates += 1;
    }

    /// Dedup, then keep (see the module notes).
    pub fn add(&mut self, ex: Example, stored: &[&[f32]]) -> AddOutcome {
        if self.is_duplicate(&ex.skill, &ex.label, &ex.phi_p, stored) {
            self.count_duplicate();
            return AddOutcome::Duplicate;
        }
        self.insert(ex);
        AddOutcome::Stored
    }

    /// Examples of (skill, label) in insertion order.
    pub fn examples(&self, skill: &str, label: &str) -> Vec<&Example> {
        self.by_label
            .get(&Self::key(skill, label))
            .map(|ix| ix.iter().map(|&i| &self.examples[i]).collect())
            .unwrap_or_default()
    }

    /// Examples added since the last attempt of (skill, label).
    pub fn new_count(&self, skill: &str, label: &str) -> usize {
        self.new_since
            .get(&Self::key(skill, label))
            .copied()
            .unwrap_or(0)
    }

    /// An attempt of (skill, label) happened.
    pub fn reset(&mut self, skill: &str, label: &str) {
        self.new_since.remove(&Self::key(skill, label));
    }

    /// A rollback happened.
    pub fn reset_all(&mut self) {
        self.new_since.clear();
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Refused duplicates since start (not persisted).
    pub fn duplicates(&self) -> u64 {
        self.duplicates
    }

    /// Every (skill, label) with examples, sorted.
    pub fn labels(&self) -> Vec<LabelCount> {
        let mut v: Vec<LabelCount> = self
            .by_label
            .iter()
            .map(|((s, l), ix)| LabelCount {
                skill: s.clone(),
                label: l.clone(),
                total: ix.len(),
                new: self.new_count(s, l),
            })
            .collect();
        v.sort_by(|a, b| (&a.skill, &a.label).cmp(&(&b.skill, &b.label)));
        v
    }

    /// Replay one `learn.log` record (cache records are ignored here).
    pub fn apply(&mut self, r: &LogRecord) {
        match r {
            LogRecord::Example(x) => self.insert(x.clone()),
            LogRecord::Attempt(a) => self.reset(&a.skill, &a.label),
            LogRecord::Rollback { .. } => self.reset_all(),
            LogRecord::CachePut(_) | LogRecord::Contract(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::OracleAnswer;

    fn ex(label: &str, v: [f32; 3]) -> Example {
        Example {
            skill: "s".into(),
            label: label.into(),
            source: Source::Oracle,
            weight: 1.0,
            ts: 7,
            phi_p: v.to_vec(),
            h_idx: vec![1, 5],
            h_val: vec![0.5, -0.25],
        }
    }

    #[test]
    fn dedup_is_per_label_and_counts_new_examples() {
        let mut b = LearningBuffer::new(0.995);
        assert_eq!(b.add(ex("a", [1.0, 0.0, 0.0]), &[]), AddOutcome::Stored);
        assert_eq!(b.add(ex("a", [1.0, 0.0, 0.0]), &[]), AddOutcome::Duplicate);
        assert_eq!(b.add(ex("b", [1.0, 0.0, 0.0]), &[]), AddOutcome::Stored);
        assert_eq!(b.add(ex("a", [0.0, 1.0, 0.0]), &[]), AddOutcome::Stored);
        let stored = [0.0f32, 0.0, 1.0];
        assert_eq!(
            b.add(ex("a", [0.0, 0.0, 1.0]), &[&stored]),
            AddOutcome::Duplicate
        );
        assert_eq!(b.new_count("s", "a"), 2);
        b.reset("s", "a");
        assert_eq!(b.new_count("s", "a"), 0);
        assert_eq!(b.examples("s", "a").len(), 2);
        assert_eq!(b.duplicates(), 2);
    }

    #[test]
    fn log_round_trips_and_drops_a_bad_tail() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("learn.log");
        let recs = vec![
            LogRecord::Example(ex("a", [0.6, 0.8, 0.0])),
            LogRecord::CachePut(CacheEntry {
                scope: "skill:s:x".into(),
                phi_p: vec![1.0, 0.0],
                answer: OracleAnswer::Score(3),
                ts: 9,
            }),
            LogRecord::Attempt(AttemptRecord {
                skill: "s".into(),
                label: "a".into(),
                outcome: "rejected".into(),
                generation: 0,
            }),
            LogRecord::Rollback { generation: 2 },
        ];
        {
            let (log, rep) = LearnLog::open(&p).unwrap();
            assert!(rep.records.is_empty());
            for r in &recs {
                log.append(r).unwrap();
            }
        }
        // A cut record at the end.
        let mut bytes = std::fs::read(&p).unwrap();
        let good = bytes.len();
        let extra = LogRecord::Rollback { generation: 5 }.frame();
        bytes.extend_from_slice(&extra[..extra.len() - 2]);
        std::fs::write(&p, &bytes).unwrap();
        let (log, rep) = LearnLog::open(&p).unwrap();
        assert_eq!(rep.records, recs);
        assert_eq!(rep.dropped_bytes, (extra.len() - 2) as u64);
        assert_eq!(std::fs::metadata(&p).unwrap().len(), good as u64);
        log.append(&LogRecord::Rollback { generation: 1 }).unwrap();
        drop(log);
        // A flipped payload byte fails the crc: that record and the rest are dropped.
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[good - 3] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        let (_, rep) = LearnLog::open(&p).unwrap();
        assert_eq!(rep.records.len(), 3);
    }

    #[test]
    fn a_contract_registry_is_rebuilt_from_the_log_first_seen_wins() {
        let a = Contract::new(
            &["x", "y"],
            &serde_json::json!("Which?"),
            &serde_json::json!({"x": "ex", "y": "why"}),
            1,
        );
        let a2 = Contract::new(&["y", "x"], &serde_json::json!("Other"), &a.criteria, 2);
        assert_eq!(a.skill, a2.skill);
        assert!(a.has_label("y") && !a.has_label("z"));
        let r = a.rubric();
        assert_eq!(r.instructions, "Which?");
        assert_eq!(r.order(), ["x", "y"]);
        // Non-string instructions become canonical JSON text; null criteria stay.
        let o = Contract::new(
            &["q", "p"],
            &serde_json::json!({"b": 1, "a": [true]}),
            &serde_json::json!({"q": null, "p": {"d": "x"}}),
            3,
        );
        let r = o.rubric();
        assert_eq!(r.instructions, r#"{"a":[true],"b":1}"#);
        assert_eq!(r.criteria["q"], Value::Null);
        assert_eq!(r.order(), ["q", "p"]);
        // Ids may hold any byte (the key hashes a JSON array, not a join).
        let odd = Contract::new(
            &["b", "a\nc"],
            &serde_json::json!({"k": [1, null]}),
            &serde_json::json!({"b": "B", "a\nc": null}),
            11,
        );
        let frame = LogRecord::Contract(odd.clone()).frame();
        let (recs, used) = read_records(&frame);
        assert_eq!((recs, used), (vec![LogRecord::Contract(odd)], frame.len()));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("learn.log");
        {
            let (log, _) = LearnLog::open(&p).unwrap();
            log.append(&LogRecord::Contract(a.clone())).unwrap();
            log.append(&LogRecord::Contract(a2.clone())).unwrap();
            log.append(&LogRecord::Contract(o.clone())).unwrap();
        }
        let (_, rep) = LearnLog::open(&p).unwrap();
        let mut reg = ContractRegistry::default();
        for rec in &rep.records {
            if let LogRecord::Contract(c) = rec {
                reg.insert(c.clone());
            }
        }
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get(&a.skill), Some(&a));
        assert_eq!(reg.get(&o.skill), Some(&o));
        let ids: Vec<&str> = reg.iter().map(|c| c.skill.as_str()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }
}
