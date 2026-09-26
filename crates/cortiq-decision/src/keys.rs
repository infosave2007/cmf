//! API keys, plans, rate windows and quotas (spec §4.10, §4.15; port of the
//! router's `auth.rs`, `config.rs:198-241`, `api.rs:374-543`, `store.rs:54-57`).
//!
//! * **Key** = `key_prefix` (default `cortiq_`) + 40 lowercase hex from the OS
//!   RNG. It is shown once, when it is created; `keys.json` holds only
//!   `sha256(raw)` in lowercase hex — the same digest the router stores, so
//!   imported router keys keep working without being reissued.
//! * **Lookup**: the sha256 of the presented key is compared with every stored
//!   digest in constant time (`subtle`), without an early exit.
//! * **Record** `{hash, account, plan, label, created, expires, active,
//!   rate_per_min, decision_quota, token_quota, credit_usd, oracle_budget_usd,
//!   oracle_allowed}`: `expires` unix seconds or `null`; `rate_per_min`,
//!   `decision_quota` and `token_quota` 0 = unlimited; `credit_usd` and
//!   `oracle_budget_usd` decimal strings or `null` (no limit). A revoked key
//!   keeps its record with `active: false`. `oracle_allowed` defaults to false:
//!   the oracle is opt-in per key.
//! * **`keys.json`** `{"version":1,"keys":[…]}` is replaced atomically on every
//!   change; a server re-reads it when its mtime or size changes, checked at
//!   most every 15 s ([`KeyStore::maybe_reload`]).
//! * **Rate window**: a fixed minute (`unix / 60`) per account, counting
//!   requests; over the limit → 429 with `Retry-After` = seconds to the next
//!   minute ([`RateLimiter`]).
//!
//! Quotas and credit are checked by the service against the usage ledger
//! before a request is processed (spec §4.10).

use crate::config::PlanConfig;
use crate::metering::Usd;
use crate::statedir::atomic_write;
use anyhow::{Context, Result, bail, ensure};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use subtle::ConstantTimeEq;

/// Random hex characters of a minted key.
pub const KEY_HEX_CHARS: usize = 40;
/// Schema version of `keys.json`.
pub const KEYS_FILE_VERSION: u32 = 1;
/// How often a server looks at the mtime of `keys.json`.
pub const RELOAD_EVERY: Duration = Duration::from_secs(15);
/// The plan of a key created without one.
pub const DEFAULT_PLAN: &str = "starter";
/// Characters of the hash prefix shown in listings.
pub const HASH_PREFIX_CHARS: usize = 12;

/// Lowercase hex sha256 of a raw key (router `store.rs:54-57`).
pub fn hash_key(raw: &str) -> String {
    format!("{:x}", Sha256::digest(raw.as_bytes()))
}

/// `n` random bytes of the OS RNG as lowercase hex.
pub fn random_hex(n_bytes: usize) -> Result<String> {
    use rand_core::{OsRng, RngCore};
    let mut b = vec![0u8; n_bytes];
    OsRng
        .try_fill_bytes(&mut b)
        .map_err(|e| anyhow::anyhow!("OS random number generator: {e}"))?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

/// A new raw key: `prefix` + 40 hex.
pub fn generate_key(prefix: &str) -> Result<String> {
    Ok(format!("{prefix}{}", random_hex(KEY_HEX_CHARS / 2)?))
}

/// Seconds since the Unix epoch.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn check_money(what: &str, v: &Option<String>) -> Result<Option<Usd>> {
    v.as_deref()
        .map(|s| Usd::parse(s).with_context(|| format!("{what} '{s}'")))
        .transpose()
}

/// One stored key (never the key itself).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRecord {
    /// sha256 of the raw key, lowercase hex.
    pub hash: String,
    pub account: String,
    pub plan: String,
    pub label: String,
    pub created: u64,
    /// Unix seconds; `null` = never.
    pub expires: Option<u64>,
    pub active: bool,
    pub rate_per_min: u32,
    pub decision_quota: u64,
    pub token_quota: u64,
    pub credit_usd: Option<String>,
    pub oracle_budget_usd: Option<String>,
    pub oracle_allowed: bool,
}

impl KeyRecord {
    /// The first 12 hex characters of the hash (listings, ledger `key12`).
    pub fn hash12(&self) -> &str {
        &self.hash[..HASH_PREFIX_CHARS]
    }

    pub fn is_expired(&self, now: u64) -> bool {
        self.expires.is_some_and(|e| e <= now)
    }

    pub fn credit(&self) -> Result<Option<Usd>> {
        check_money("credit_usd", &self.credit_usd)
    }

    pub fn oracle_budget(&self) -> Result<Option<Usd>> {
        check_money("oracle_budget_usd", &self.oracle_budget_usd)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            is_hash(&self.hash),
            "key hash must be 64 lowercase hex characters"
        );
        check_account(&self.account)?;
        ensure!(
            self.plan.len() <= 64 && self.label.len() <= 256,
            "key plan ≤ 64 bytes, label ≤ 256 bytes"
        );
        self.credit()?;
        self.oracle_budget()?;
        Ok(())
    }

    /// The listing entry (spec §5b: hash12, limits, usage added by the caller).
    pub fn listing(&self, now: u64) -> Value {
        json!({
            "hash12": self.hash12(),
            "account": self.account,
            "plan": self.plan,
            "label": self.label,
            "created": self.created,
            "expires": self.expires,
            "expired": self.is_expired(now),
            "active": self.active,
            "rate_per_min": self.rate_per_min,
            "decision_quota": self.decision_quota,
            "token_quota": self.token_quota,
            "credit_usd": self.credit_usd,
            "oracle_budget_usd": self.oracle_budget_usd,
            "oracle_allowed": self.oracle_allowed,
        })
    }
}

/// An account id: 1..128 bytes of `[A-Za-z0-9_.@-]`.
pub fn check_account(a: &str) -> Result<()> {
    ensure!(
        !a.is_empty()
            && a.len() <= 128
            && a.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'@')),
        "account '{a}' must be 1..128 characters of [A-Za-z0-9_.@-]"
    );
    Ok(())
}

/// `keys.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeysFile {
    pub version: u32,
    pub keys: Vec<KeyRecord>,
}

/// Options of a new key; `None` takes the plan's value (or the default).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewKey {
    #[serde(default)]
    pub plan: Option<String>,
    /// Else `acct_` + 12 hex.
    #[serde(default)]
    pub account: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    /// Key lifetime in days (0 = never expires); else the plan's.
    #[serde(default)]
    pub days: Option<u32>,
    #[serde(default)]
    pub rate_per_min: Option<u32>,
    #[serde(default)]
    pub decision_quota: Option<u64>,
    #[serde(default)]
    pub token_quota: Option<u64>,
    #[serde(default)]
    pub credit_usd: Option<String>,
    #[serde(default)]
    pub oracle_budget_usd: Option<String>,
    #[serde(default)]
    pub oracle_allowed: Option<bool>,
}

/// A created key: the raw key (shown once) and its record.
#[derive(Clone, PartialEq, Eq)]
pub struct CreatedKey {
    pub raw: String,
    pub record: KeyRecord,
}

impl std::fmt::Debug for CreatedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreatedKey")
            .field("raw", &"<redacted>")
            .field("record", &self.record)
            .finish()
    }
}

impl CreatedKey {
    /// The one response that carries the raw key.
    pub fn to_json(&self, now: u64) -> Value {
        let mut v = self.record.listing(now);
        v["key"] = Value::String(self.raw.clone());
        v
    }
}

/// Why a presented key is not accepted (all 401).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthFailure {
    #[error("missing API key (Authorization: Bearer … or x-api-key)")]
    Missing,
    #[error("invalid API key")]
    Invalid,
    #[error("API key expired")]
    Expired,
    #[error("API key revoked")]
    Revoked,
}

#[derive(Debug, Default)]
struct Loaded {
    records: Vec<KeyRecord>,
    digests: Vec<[u8; 32]>,
}

impl Loaded {
    fn from_file(f: KeysFile) -> Result<Self> {
        let mut digests = Vec::with_capacity(f.keys.len());
        let mut seen = std::collections::HashSet::new();
        for k in &f.keys {
            k.validate()
                .with_context(|| format!("key {}", &k.hash[..k.hash.len().min(12)]))?;
            ensure!(
                seen.insert(k.hash.clone()),
                "duplicate key hash {}",
                k.hash12()
            );
            let mut d = [0u8; 32];
            for (i, byte) in d.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&k.hash[2 * i..2 * i + 2], 16).expect("validated hex");
            }
            digests.push(d);
        }
        Ok(Self {
            records: f.keys,
            digests,
        })
    }
}

#[derive(Debug)]
struct ReloadState {
    last_check: Option<Instant>,
    stamp: Option<(SystemTime, u64)>,
}

/// The keys of one state directory.
#[derive(Debug)]
pub struct KeyStore {
    path: PathBuf,
    prefix: String,
    loaded: RwLock<Loaded>,
    reload: Mutex<ReloadState>,
    /// Serialises read-modify-write of the file within this process.
    write: Mutex<()>,
}

fn file_stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
}

fn read_file(path: &Path) -> Result<KeysFile> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let v = crate::canonical::parse(&bytes)
                .with_context(|| format!("{} is not valid JSON", path.display()))?;
            let f: KeysFile =
                serde_json::from_value(v).with_context(|| format!("{}", path.display()))?;
            ensure!(
                f.version == KEYS_FILE_VERSION,
                "{}: unsupported version {}",
                path.display(),
                f.version
            );
            Ok(f)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(KeysFile {
            version: KEYS_FILE_VERSION,
            keys: Vec::new(),
        }),
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

impl KeyStore {
    /// The keys at `path` (`keys.json`; missing = no keys).
    pub fn open(path: impl AsRef<Path>, key_prefix: &str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let stamp = file_stamp(&path);
        let loaded = Loaded::from_file(read_file(&path)?)?;
        Ok(Self {
            path,
            prefix: key_prefix.to_string(),
            loaded: RwLock::new(loaded),
            reload: Mutex::new(ReloadState {
                last_check: Some(Instant::now()),
                stamp,
            }),
            write: Mutex::new(()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// No key at all (open mode is possible, spec §4.10).
    pub fn is_empty(&self) -> bool {
        self.loaded.read().records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.loaded.read().records.len()
    }

    /// Every record, in file order.
    pub fn records(&self) -> Vec<KeyRecord> {
        self.loaded.read().records.clone()
    }

    /// Re-read the file now if its mtime or size changed. Returns whether it
    /// was reloaded. A file that fails to parse keeps the keys in memory.
    pub fn reload(&self) -> Result<bool> {
        let stamp = file_stamp(&self.path);
        let mut r = self.reload.lock();
        r.last_check = Some(Instant::now());
        if stamp == r.stamp {
            return Ok(false);
        }
        let loaded = Loaded::from_file(read_file(&self.path)?)?;
        *self.loaded.write() = loaded;
        r.stamp = stamp;
        Ok(true)
    }

    /// [`KeyStore::reload`] at most every [`RELOAD_EVERY`].
    pub fn maybe_reload(&self) -> Result<bool> {
        {
            let r = self.reload.lock();
            if r.last_check.is_some_and(|t| t.elapsed() < RELOAD_EVERY) {
                return Ok(false);
            }
        }
        self.reload()
    }

    /// The record of a raw key: constant-time comparison of its sha256 with
    /// every stored digest; then active and expiry.
    pub fn authenticate(&self, raw: &str, now: u64) -> Result<KeyRecord, AuthFailure> {
        if raw.is_empty() {
            return Err(AuthFailure::Missing);
        }
        let d: [u8; 32] = Sha256::digest(raw.as_bytes()).into();
        let l = self.loaded.read();
        let mut found: Option<usize> = None;
        for (i, s) in l.digests.iter().enumerate() {
            let eq: bool = s.ct_eq(&d).into();
            if eq && found.is_none() {
                found = Some(i);
            }
        }
        let rec = found
            .map(|i| l.records[i].clone())
            .ok_or(AuthFailure::Invalid)?;
        if !rec.active {
            return Err(AuthFailure::Revoked);
        }
        if rec.is_expired(now) {
            return Err(AuthFailure::Expired);
        }
        Ok(rec)
    }

    /// Apply `f` to the file's records (re-read under the write lock), write the
    /// result atomically and take it into memory.
    fn modify<T>(&self, f: impl FnOnce(&mut Vec<KeyRecord>) -> Result<T>) -> Result<T> {
        let _w = self.write.lock();
        let mut file = read_file(&self.path)?;
        let out = f(&mut file.keys)?;
        let loaded = Loaded::from_file(file.clone())?;
        let mut bytes = serde_json::to_vec_pretty(&file)?;
        bytes.push(b'\n');
        atomic_write(&self.path, &bytes)?;
        *self.loaded.write() = loaded;
        let mut r = self.reload.lock();
        r.stamp = file_stamp(&self.path);
        r.last_check = Some(Instant::now());
        Ok(out)
    }

    /// Create a key from a plan and overrides (spec §5b `POST /v1/admin/keys`).
    pub fn create(
        &self,
        new: &NewKey,
        plans: &BTreeMap<String, PlanConfig>,
        now: u64,
    ) -> Result<CreatedKey> {
        let plan_name = new.plan.clone().unwrap_or_else(|| DEFAULT_PLAN.into());
        let plan = plans
            .get(&plan_name)
            .ok_or_else(|| anyhow::anyhow!("unknown plan '{plan_name}'"))?;
        let account = match &new.account {
            Some(a) if !a.trim().is_empty() => a.trim().to_string(),
            _ => format!("acct_{}", random_hex(6)?),
        };
        check_account(&account)?;
        let days = new.days.or(plan.days).filter(|&d| d > 0);
        let raw = generate_key(&self.prefix)?;
        let record = KeyRecord {
            hash: hash_key(&raw),
            account,
            plan: plan_name,
            label: new.label.clone().unwrap_or_default(),
            created: now,
            expires: days.map(|d| now + u64::from(d) * 86_400),
            active: true,
            rate_per_min: new.rate_per_min.unwrap_or(plan.rate_per_min),
            decision_quota: new.decision_quota.unwrap_or(plan.decision_quota),
            token_quota: new.token_quota.unwrap_or(0),
            credit_usd: new.credit_usd.clone(),
            oracle_budget_usd: new.oracle_budget_usd.clone(),
            oracle_allowed: new.oracle_allowed.unwrap_or(false),
        };
        record.validate()?;
        let rec = record.clone();
        self.modify(move |keys| {
            ensure!(
                keys.iter().all(|k| k.hash != rec.hash),
                "key collision (retry)"
            );
            keys.push(rec);
            Ok(())
        })?;
        Ok(CreatedKey { raw, record })
    }

    /// Revoke every active key of an account; returns how many were revoked.
    pub fn revoke_account(&self, account: &str) -> Result<usize> {
        self.modify(|keys| {
            let mut n = 0;
            for k in keys.iter_mut().filter(|k| k.account == account && k.active) {
                k.active = false;
                n += 1;
            }
            Ok(n)
        })
    }

    /// Revoke the active key whose hash starts with `prefix` (at least 12 hex
    /// characters; an ambiguous prefix is an error).
    pub fn revoke_hash_prefix(&self, prefix: &str) -> Result<usize> {
        ensure!(
            prefix.len() >= HASH_PREFIX_CHARS
                && prefix.len() <= 64
                && prefix
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "a key is named by at least {HASH_PREFIX_CHARS} lowercase hex characters of its hash"
        );
        self.modify(|keys| {
            let hits: Vec<usize> = keys
                .iter()
                .enumerate()
                .filter(|(_, k)| k.hash.starts_with(prefix))
                .map(|(i, _)| i)
                .collect();
            if hits.len() > 1 {
                bail!("hash prefix {prefix} names {} keys", hits.len());
            }
            let mut n = 0;
            for i in hits {
                if keys[i].active {
                    keys[i].active = false;
                    n += 1;
                }
            }
            Ok(n)
        })
    }

    /// Import router keys (spec §4.15): a JSON export of the router's MySQL
    /// `api_keys` table (rows `{key_hash, account, plan, email?, label?,
    /// active?, rate_per_min?, decision_quota?, expires_at?, created_at?}`;
    /// numbers may be numeric strings), and/or its config's `api_keys`
    /// entries `{key, account, rate_per_min?, decision_quota?}` (raw keys are
    /// hashed on import). Accepted shapes: an array of rows, `{"api_keys":
    /// [...]}`, `{"rows"|"data": [...]}`, or JSON lines. Hashes already
    /// present are skipped. Imported keys have no token quota, no credit
    /// limit and `oracle_allowed: false`.
    pub fn import_router(&self, bytes: &[u8], now: u64) -> Result<ImportReport> {
        let rows = import_rows(bytes)?;
        let mut parsed = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            parsed.push(import_record(row, now).with_context(|| format!("row {}", i + 1))?);
        }
        self.modify(|keys| {
            let mut report = ImportReport::default();
            for rec in parsed {
                if keys.iter().any(|k| k.hash == rec.hash) {
                    report.skipped += 1;
                } else {
                    report.imported += 1;
                    report.accounts.insert(rec.account.clone());
                    keys.push(rec);
                }
            }
            Ok(report)
        })
    }
}

/// What an import did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportReport {
    pub imported: usize,
    pub skipped: usize,
    pub accounts: std::collections::BTreeSet<String>,
}

fn import_rows(bytes: &[u8]) -> Result<Vec<Map<String, Value>>> {
    let objects = |a: &[Value]| -> Result<Vec<Map<String, Value>>> {
        a.iter()
            .map(|v| {
                v.as_object()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("every key row must be a JSON object"))
            })
            .collect()
    };
    if let Ok(v) = crate::canonical::parse(bytes) {
        return match &v {
            Value::Array(a) => objects(a),
            Value::Object(m) => {
                for k in ["api_keys", "rows", "data"] {
                    if let Some(Value::Array(a)) = m.get(k) {
                        return objects(a);
                    }
                }
                if m.contains_key("key_hash") || m.contains_key("key") {
                    Ok(vec![m.clone()])
                } else {
                    bail!("expected an array of key rows or {{\"api_keys\": [...]}}")
                }
            }
            _ => bail!("expected an array of key rows or {{\"api_keys\": [...]}}"),
        };
    }
    let mut out = Vec::new();
    for (i, line) in bytes.split(|&b| b == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let v =
            crate::canonical::parse(line).with_context(|| format!("line {} is not JSON", i + 1))?;
        out.push(
            v.as_object()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("line {} is not a JSON object", i + 1))?,
        );
    }
    Ok(out)
}

fn num_field(row: &Map<String, Value>, k: &str) -> Result<Option<u64>> {
    match row.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("{k} must be a non-negative integer")),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => s
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("{k} must be a non-negative integer")),
        Some(_) => bail!("{k} must be a non-negative integer"),
    }
}

fn str_field(row: &Map<String, Value>, k: &str) -> Option<String> {
    row.get(k).and_then(Value::as_str).map(str::to_string)
}

fn import_record(row: &Map<String, Value>, now: u64) -> Result<KeyRecord> {
    let hash = match (str_field(row, "key_hash"), str_field(row, "key")) {
        (Some(h), _) => h.trim().to_ascii_lowercase(),
        (None, Some(raw)) => hash_key(&raw),
        (None, None) => bail!("a row needs key_hash (MySQL api_keys) or key (router config)"),
    };
    let account = str_field(row, "account").unwrap_or_else(|| "default".into());
    let active = match row.get("active") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => num_field(row, "active")?.is_none_or(|v| v != 0),
    };
    let rate = num_field(row, "rate_per_min")?.unwrap_or(0);
    let rec = KeyRecord {
        hash,
        account,
        plan: str_field(row, "plan").unwrap_or_default(),
        label: str_field(row, "label").unwrap_or_default(),
        created: num_field(row, "created_at")?.unwrap_or(now),
        expires: num_field(row, "expires_at")?,
        active,
        rate_per_min: u32::try_from(rate).context("rate_per_min out of range")?,
        decision_quota: num_field(row, "decision_quota")?.unwrap_or(0),
        token_quota: 0,
        credit_usd: None,
        oracle_budget_usd: None,
        oracle_allowed: false,
    };
    rec.validate()?;
    Ok(rec)
}

// ------------------------------------------------------------------ rate window

/// Fixed one-minute windows per account (router `auth.rs:151-163`).
#[derive(Debug, Default)]
pub struct RateLimiter {
    windows: Mutex<HashMap<String, (u64, u32)>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one request of `account` at `now` (unix seconds) against
    /// `rate_per_min` (0 = unlimited). `Err(retry_after_s)` when over the limit.
    pub fn check(&self, account: &str, rate_per_min: u32, now: u64) -> Result<(), u64> {
        if rate_per_min == 0 {
            return Ok(());
        }
        let minute = now / 60;
        let mut w = self.windows.lock();
        if w.len() > 100_000 {
            w.retain(|_, (m, _)| *m == minute);
        }
        let e = w.entry(account.to_string()).or_insert((minute, 0));
        if e.0 != minute {
            *e = (minute, 0);
        }
        e.1 = e.1.saturating_add(1);
        if e.1 <= rate_per_min {
            Ok(())
        } else {
            Err(60 - now % 60)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_the_router() {
        // sha256("secret-key"), as the router's store.rs computes it.
        assert_eq!(
            hash_key("secret-key"),
            format!("{:x}", Sha256::digest(b"secret-key"))
        );
        let k = generate_key("cortiq_").unwrap();
        assert!(k.starts_with("cortiq_"));
        assert_eq!(k.len(), 7 + KEY_HEX_CHARS);
        assert!(k[7..].bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn rate_window_is_a_fixed_minute() {
        let r = RateLimiter::new();
        assert!(r.check("a", 2, 120).is_ok());
        assert!(r.check("a", 2, 130).is_ok());
        assert_eq!(r.check("a", 2, 150), Err(30));
        assert!(r.check("b", 2, 150).is_ok());
        assert!(r.check("a", 2, 180).is_ok());
        assert!(r.check("a", 0, 180).is_ok());
    }
}
