//! Usage ledger: append, replay and snapshot (spec §4.11).
//!
//! * **Files** `usage/YYYY-MM.jsonl` (UTC month), one line per successful
//!   request, in this field order: `{ts, id, account, key12, model, generation,
//!   input_tokens, output_tokens, cost_usd, cost_local_usd, cost_oracle_usd,
//!   oracle_calls, cache_hits, questions, actions}` — `actions` counts the
//!   questions per action `{local, abstain, cache, oracle}`. Neither text nor a
//!   hash of text is written. Errors are not billed and never written.
//! * **Durability**: [`UsageLedger::append`] updates the totals in memory and
//!   queues the line; [`UsageLedger::flush`] writes the queue and `fsync`s the
//!   touched files. A background [`Flusher`] flushes every second and once more
//!   when it stops, so a crash loses at most about one second of records.
//! * **Replay**: at open, `usage/totals.json` (totals plus the byte offset of
//!   every month file they cover) is loaded, then the tail of every file after
//!   its offset is replayed. Without a (valid) snapshot every file is replayed
//!   from the start. A trailing partial line (a crash in the middle of a write)
//!   is cut off. A line that is not a valid record stops the open: billing
//!   data is never skipped silently.
//! * **Exactness**: money is summed as exact decimals ([`Usd`]) of each
//!   record's JSON number, so the totals do not depend on the order or the
//!   split between snapshot and tail: live totals, a replay from scratch and a
//!   snapshot plus tail are equal.
//!
//! The month of a line is the month of `max(ts, the last month written)`, so
//! lines are appended in file order even when clocks step back.

use crate::metering::Usd;
use crate::statedir::{atomic_write, sync_dir};
use anyhow::{Context, Result, bail, ensure};
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Name of the snapshot in the usage directory.
pub const SNAPSHOT_FILE: &str = "totals.json";
/// Snapshot schema.
pub const SNAPSHOT_VERSION: u32 = 1;
/// Flush period of the background flusher (spec §4.11).
pub const FLUSH_EVERY: Duration = Duration::from_secs(1);
/// A snapshot is refreshed at most this often by the flusher.
pub const SNAPSHOT_EVERY: Duration = Duration::from_secs(300);

/// Questions per action of one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actions {
    pub local: u64,
    pub abstain: u64,
    pub cache: u64,
    pub oracle: u64,
}

impl Actions {
    fn add(&mut self, o: &Actions) {
        self.local += o.local;
        self.abstain += o.abstain;
        self.cache += o.cache;
        self.oracle += o.oracle;
    }

    pub fn total(&self) -> u64 {
        self.local + self.abstain + self.cache + self.oracle
    }
}

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageRecord {
    pub ts: u64,
    pub id: String,
    pub account: String,
    /// First 12 hex characters of the key's hash (`null` in open mode).
    pub key12: Option<String>,
    pub model: String,
    pub generation: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub cost_local_usd: f64,
    pub cost_oracle_usd: f64,
    pub oracle_calls: u64,
    pub cache_hits: u64,
    /// Decisions: answered questions.
    pub questions: u64,
    pub actions: Actions,
}

impl UsageRecord {
    fn validate(&self) -> Result<()> {
        for (k, v) in [
            ("cost_usd", self.cost_usd),
            ("cost_local_usd", self.cost_local_usd),
            ("cost_oracle_usd", self.cost_oracle_usd),
        ] {
            ensure!(
                v.is_finite() && v >= 0.0,
                "{k} must be finite and non-negative"
            );
        }
        ensure!(
            self.actions.total() == self.questions,
            "actions must count every question"
        );
        ensure!(!self.account.is_empty(), "a usage record needs an account");
        Ok(())
    }

    /// The ledger line (without the newline).
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("a usage record serialises")
    }

    /// Parse a ledger line (floats correctly rounded).
    pub fn from_line(line: &[u8]) -> Result<Self> {
        let v = crate::canonical::parse(line)?;
        let r: Self = serde_json::from_value(v)?;
        r.validate()?;
        Ok(r)
    }
}

/// The totals of one account.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub requests: u64,
    pub decisions: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: Usd,
    pub cost_local_usd: Usd,
    pub cost_oracle_usd: Usd,
    pub oracle_calls: u64,
    pub cache_hits: u64,
    pub actions: Actions,
}

/// Snapshot form of [`Totals`] (money as exact decimal strings).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TotalsRecord {
    requests: u64,
    decisions: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: String,
    cost_local_usd: String,
    cost_oracle_usd: String,
    oracle_calls: u64,
    cache_hits: u64,
    actions: Actions,
}

impl Totals {
    /// Add one record (money as the exact decimal of its JSON number).
    pub fn add(&mut self, r: &UsageRecord) -> Result<()> {
        self.requests += 1;
        self.decisions += r.questions;
        self.input_tokens += r.input_tokens;
        self.output_tokens += r.output_tokens;
        self.cost_usd = self.cost_usd.checked_add(Usd::from_f64(r.cost_usd)?)?;
        self.cost_local_usd = self
            .cost_local_usd
            .checked_add(Usd::from_f64(r.cost_local_usd)?)?;
        self.cost_oracle_usd = self
            .cost_oracle_usd
            .checked_add(Usd::from_f64(r.cost_oracle_usd)?)?;
        self.oracle_calls += r.oracle_calls;
        self.cache_hits += r.cache_hits;
        self.actions.add(&r.actions);
        Ok(())
    }

    /// Billed tokens (input + output), the unit of `token_quota`.
    pub fn tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// `/v1/usage` form.
    pub fn to_json(&self) -> Value {
        json!({
            "requests": self.requests,
            "decisions": self.decisions,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd.to_f64(),
            "cost_local_usd": self.cost_local_usd.to_f64(),
            "cost_oracle_usd": self.cost_oracle_usd.to_f64(),
            "oracle_calls": self.oracle_calls,
            "cache_hits": self.cache_hits,
            "actions": self.actions,
        })
    }

    fn record(&self) -> TotalsRecord {
        TotalsRecord {
            requests: self.requests,
            decisions: self.decisions,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cost_usd: self.cost_usd.to_string(),
            cost_local_usd: self.cost_local_usd.to_string(),
            cost_oracle_usd: self.cost_oracle_usd.to_string(),
            oracle_calls: self.oracle_calls,
            cache_hits: self.cache_hits,
            actions: self.actions,
        }
    }

    fn from_record(r: &TotalsRecord) -> Result<Self> {
        Ok(Self {
            requests: r.requests,
            decisions: r.decisions,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            cost_usd: Usd::parse(&r.cost_usd)?,
            cost_local_usd: Usd::parse(&r.cost_local_usd)?,
            cost_oracle_usd: Usd::parse(&r.cost_oracle_usd)?,
            oracle_calls: r.oracle_calls,
            cache_hits: r.cache_hits,
            actions: r.actions,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    /// Month file name → bytes covered.
    files: BTreeMap<String, u64>,
    accounts: BTreeMap<String, TotalsRecord>,
}

/// The UTC month `YYYY-MM` of a unix time (civil-from-days, proleptic Gregorian).
pub fn month_of(ts: u64) -> String {
    let days = (ts / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}")
}

fn is_month_file(name: &str) -> bool {
    let b = name.as_bytes();
    name.len() == 13
        && name.ends_with(".jsonl")
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
}

#[derive(Debug, Default)]
struct Inner {
    totals: BTreeMap<String, Totals>,
    /// Bytes of every month file that are on disk (flushed).
    offsets: BTreeMap<String, u64>,
    /// Lines queued for their month file.
    pending: Vec<(String, String)>,
    last_month: Option<String>,
    last_snapshot: Option<Instant>,
}

/// The usage ledger of one state directory.
#[derive(Debug)]
pub struct UsageLedger {
    dir: PathBuf,
    inner: Mutex<Inner>,
    /// Lines replayed at open (after the snapshot).
    replayed: u64,
    warnings: Vec<String>,
}

/// Read a month file from `offset`: the complete lines and the end of the last
/// complete line (a trailing partial line is not included).
fn read_tail(path: &Path, offset: u64) -> Result<(Vec<Vec<u8>>, u64, u64)> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len();
    ensure!(
        offset <= len,
        "{} is shorter ({len} B) than its snapshot offset {offset}",
        path.display()
    );
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::with_capacity((len - offset) as usize);
    f.read_to_end(&mut buf)?;
    let complete = buf.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
    let lines = buf[..complete]
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(<[u8]>::to_vec)
        .collect();
    Ok((lines, offset + complete as u64, len))
}

impl UsageLedger {
    /// Open the ledger in `dir` (`state/usage`), replaying snapshot and tails.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let mut warnings = Vec::new();
        let mut months: Vec<String> = Vec::new();
        for e in std::fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let name = e?.file_name().to_string_lossy().into_owned();
            if is_month_file(&name) {
                months.push(name);
            }
        }
        months.sort();
        let snap = Self::load_snapshot(&dir, &months, &mut warnings);
        let (mut totals, start) = match snap {
            Some((t, files)) => (t, files),
            None => (BTreeMap::new(), BTreeMap::new()),
        };
        let mut offsets = BTreeMap::new();
        let mut replayed = 0u64;
        for m in &months {
            let path = dir.join(m);
            let from = start.get(m).copied().unwrap_or(0);
            let (lines, end, len) = read_tail(&path, from)?;
            for (i, line) in lines.iter().enumerate() {
                let r = UsageRecord::from_line(line).with_context(|| {
                    format!("{}: record {} after byte {from}", path.display(), i + 1)
                })?;
                totals.entry(r.account.clone()).or_default().add(&r)?;
                replayed += 1;
            }
            if end < len {
                warnings.push(format!(
                    "{}: cut a partial last line ({} B)",
                    path.display(),
                    len - end
                ));
                let f = OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .with_context(|| format!("open {}", path.display()))?;
                f.set_len(end)?;
                f.sync_all()?;
            }
            offsets.insert(m.clone(), end);
        }
        let last_month = months.last().map(|m| m[..7].to_string());
        Ok(Self {
            dir,
            inner: Mutex::new(Inner {
                totals,
                offsets,
                pending: Vec::new(),
                last_month,
                last_snapshot: None,
            }),
            replayed,
            warnings,
        })
    }

    /// The snapshot, if present and consistent with the files on disk.
    #[allow(clippy::type_complexity)]
    fn load_snapshot(
        dir: &Path,
        months: &[String],
        warnings: &mut Vec<String>,
    ) -> Option<(BTreeMap<String, Totals>, BTreeMap<String, u64>)> {
        let path = dir.join(SNAPSHOT_FILE);
        let bytes = std::fs::read(&path).ok()?;
        let parsed = (|| -> Result<(BTreeMap<String, Totals>, BTreeMap<String, u64>)> {
            let v = crate::canonical::parse(&bytes)?;
            let s: Snapshot = serde_json::from_value(v)?;
            ensure!(s.version == SNAPSHOT_VERSION, "version {}", s.version);
            for (m, &off) in &s.files {
                ensure!(months.contains(m), "{m} is missing");
                let len = std::fs::metadata(dir.join(m))?.len();
                ensure!(off <= len, "{m} is shorter than {off} B");
            }
            let mut totals = BTreeMap::new();
            for (a, t) in &s.accounts {
                totals.insert(a.clone(), Totals::from_record(t)?);
            }
            Ok((totals, s.files))
        })();
        match parsed {
            Ok(x) => Some(x),
            Err(e) => {
                warnings.push(format!(
                    "{}: ignored ({e:#}); replaying every month file",
                    path.display()
                ));
                None
            }
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Records replayed at open after the snapshot.
    pub fn replayed(&self) -> u64 {
        self.replayed
    }

    /// Messages of the open (snapshot ignored, partial line cut).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Queue a record (it reaches the disk at the next flush) and add it to
    /// the totals.
    pub fn append(&self, r: &UsageRecord) -> Result<()> {
        r.validate()?;
        let line = r.to_line();
        let mut g = self.inner.lock();
        let mut month = month_of(r.ts);
        if let Some(last) = &g.last_month
            && *last > month
        {
            month.clone_from(last);
        }
        let mut t = g.totals.get(&r.account).cloned().unwrap_or_default();
        t.add(r)?;
        g.totals.insert(r.account.clone(), t);
        g.last_month = Some(month.clone());
        g.pending.push((month, line));
        Ok(())
    }

    /// Write the queued lines and `fsync` the files they went to.
    pub fn flush(&self) -> Result<()> {
        let mut g = self.inner.lock();
        self.flush_locked(&mut g)
    }

    fn flush_locked(&self, g: &mut Inner) -> Result<()> {
        if g.pending.is_empty() {
            return Ok(());
        }
        let mut by_month: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (m, line) in &g.pending {
            let b = by_month.entry(m.clone()).or_default();
            b.extend_from_slice(line.as_bytes());
            b.push(b'\n');
        }
        let mut new_file = false;
        for (m, bytes) in &by_month {
            let name = format!("{m}.jsonl");
            let path = self.dir.join(&name);
            new_file |= !path.exists();
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("open {}", path.display()))?;
            f.write_all(bytes)
                .and_then(|()| f.sync_all())
                .with_context(|| format!("write {}", path.display()))?;
            *g.offsets.entry(name).or_insert(0) += bytes.len() as u64;
        }
        if new_file {
            sync_dir(&self.dir)?;
        }
        g.pending.clear();
        Ok(())
    }

    /// Flush, then replace `totals.json` with the totals and the offsets they
    /// cover.
    pub fn snapshot(&self) -> Result<()> {
        let mut g = self.inner.lock();
        self.flush_locked(&mut g)?;
        let s = Snapshot {
            version: SNAPSHOT_VERSION,
            files: g.offsets.clone(),
            accounts: g
                .totals
                .iter()
                .map(|(a, t)| (a.clone(), t.record()))
                .collect(),
        };
        let mut bytes = serde_json::to_vec(&s)?;
        bytes.push(b'\n');
        atomic_write(&self.dir.join(SNAPSHOT_FILE), &bytes)?;
        g.last_snapshot = Some(Instant::now());
        Ok(())
    }

    /// Flush and snapshot (at shutdown).
    pub fn close(&self) -> Result<()> {
        self.snapshot()
    }

    /// Flush; snapshot when the last one is older than [`SNAPSHOT_EVERY`].
    pub fn tick(&self) -> Result<()> {
        let due = {
            let g = self.inner.lock();
            g.last_snapshot
                .is_none_or(|t| t.elapsed() >= SNAPSHOT_EVERY)
        };
        if due { self.snapshot() } else { self.flush() }
    }

    /// Lines queued but not yet on disk.
    pub fn pending(&self) -> usize {
        self.inner.lock().pending.len()
    }

    /// The totals of an account (zero when it has no record).
    pub fn totals(&self, account: &str) -> Totals {
        self.inner
            .lock()
            .totals
            .get(account)
            .cloned()
            .unwrap_or_default()
    }

    /// Every account's totals.
    pub fn all_totals(&self) -> BTreeMap<String, Totals> {
        self.inner.lock().totals.clone()
    }

    /// Start a thread that calls [`UsageLedger::tick`] every `period` and
    /// [`UsageLedger::close`] when the returned handle is stopped or dropped.
    pub fn start_flusher(self: &Arc<Self>, period: Duration) -> Flusher {
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let ledger = Arc::clone(self);
        let st = Arc::clone(&state);
        let handle = std::thread::Builder::new()
            .name("cmf-usage-flush".into())
            .spawn(move || {
                let (lock, cv) = &*st;
                let mut stop = lock.lock();
                while !*stop {
                    cv.wait_for(&mut stop, period);
                    if *stop {
                        break;
                    }
                    drop(stop);
                    if let Err(e) = ledger.tick() {
                        tracing::error!(error = %e, "usage ledger flush failed");
                    }
                    stop = lock.lock();
                }
                drop(stop);
                if let Err(e) = ledger.close() {
                    tracing::error!(error = %e, "usage ledger close failed");
                }
            })
            .expect("spawn the usage flusher");
        Flusher {
            state,
            handle: Some(handle),
        }
    }
}

/// The background flusher of a [`UsageLedger`].
#[derive(Debug)]
pub struct Flusher {
    state: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Flusher {
    /// Stop the thread after a final flush and snapshot.
    pub fn stop(mut self) -> Result<()> {
        self.stop_inner()
    }

    fn stop_inner(&mut self) -> Result<()> {
        {
            let (lock, cv) = &*self.state;
            *lock.lock() = true;
            cv.notify_all();
        }
        if let Some(h) = self.handle.take()
            && h.join().is_err()
        {
            bail!("the usage flusher panicked");
        }
        Ok(())
    }
}

impl Drop for Flusher {
    fn drop(&mut self) {
        let _ = self.stop_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn months_are_utc_civil_months() {
        assert_eq!(month_of(0), "1970-01");
        assert_eq!(month_of(1_790_500_000), "2026-09");
        assert_eq!(month_of(951_782_400), "2000-02"); // 2000-02-29
        assert_eq!(month_of(951_868_799), "2000-02");
        assert_eq!(month_of(951_868_800), "2000-03");
        assert_eq!(month_of(1_704_067_199), "2023-12");
        assert_eq!(month_of(1_704_067_200), "2024-01");
        assert!(is_month_file("2026-09.jsonl"));
        assert!(!is_month_file("totals.json"));
    }
}
