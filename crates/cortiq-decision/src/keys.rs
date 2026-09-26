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
//!   keeps its record with `active: false`. `oracle_allowed` defaults to false
//!   for a key created here (the oracle is opt-in per key) and to true for a
//!   key imported from cortiq-router, which escalated for every key.
//! * **Account, plan, label**: any text the router's `api_keys` columns hold
//!   — 1..128, ≤ 64 and ≤ 255 Unicode characters (`VARCHAR(128)`,
//!   `VARCHAR(64)`, `VARCHAR(255)`, router `store.rs:119-130`), e.g.
//!   `client+tag@example.com` or Cyrillic. On disk they are JSON strings
//!   (escaped by serde); log lines and CLI tables show them through
//!   [`shown`] (control characters escaped).
//! * **`keys.json`** `{"version":1,"keys":[…]}` is replaced atomically on every
//!   change; a server re-reads it when its mtime or size changes, checked at
//!   most every 15 s ([`KeyStore::maybe_reload`]).
//! * **Rate window**: a fixed minute (`unix / 60`) per account, counting
//!   requests; over the limit → 429 with `Retry-After` = seconds to the next
//!   minute ([`RateLimiter`]).
//! * **Router import** (spec §4.15, `cortiq decision keys import`): the keys of
//!   a running cortiq-router move over without reissue — rows of its MySQL
//!   `api_keys` table ([`ImportFormat::MysqlJson`]) or the `[[api_keys]]` of its
//!   configuration ([`ImportFormat::RouterToml`], raw keys hashed as they are
//!   read) — through [`read_router_keys`] (checks everything first) and
//!   [`KeyStore::import_router_keys`] (idempotent); its `usage_counters`
//!   continue in the usage ledger through [`read_router_usage`] and
//!   [`import_router_usage`]. An imported record remembers its origin
//!   ([`RouterOrigin`]): a key both in the database and in the configuration
//!   follows the router's precedence — the database row while it is active
//!   and unexpired, else the static key — whatever the import order.
//!
//! Quotas and credit are checked by the service against the usage ledger
//! before a request is processed (spec §4.10).

use crate::config::PlanConfig;
use crate::ledger::{Actions, UsageLedger, UsageRecord};
use crate::metering::Usd;
use crate::statedir::atomic_write;
use anyhow::{Context, Result, bail, ensure};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use subtle::ConstantTimeEq;

mod toml_lite;

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
/// Most characters of an account: the router's `account VARCHAR(128)`.
pub const MAX_ACCOUNT_CHARS: usize = 128;
/// Most characters of a plan: the router's `plan VARCHAR(64)`.
pub const MAX_PLAN_CHARS: usize = 64;
/// Most characters of a label: the router's `label VARCHAR(255)`.
pub const MAX_LABEL_CHARS: usize = 255;
/// `oracle_allowed` of an imported router key without `--oracle-allowed`:
/// the router escalated for every key (`allow_oracle` defaults to true,
/// router `api.rs:67`), subject to its global oracle switch — here the
/// server's `oracle.enabled`, budgets and stop rules still apply.
pub const ROUTER_ORACLE_ALLOWED: bool = true;

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
    /// Origin of a key imported from cortiq-router; absent for a key of this
    /// server (created here, or revoked here after its import).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterOrigin>,
}

/// Where an imported key comes from in cortiq-router (spec §4.15).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterOrigin {
    /// The record's attributes are a row of the router's `api_keys` table.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub db: bool,
    /// The key is also (or only) a static key of the router's configuration
    /// (`[[api_keys]]`, router `main.rs:327-341`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<StaticKey>,
}

impl RouterOrigin {
    /// `mysql`, `config` or `mysql+config` (listings).
    pub fn name(&self) -> &'static str {
        match (self.db, self.config.is_some()) {
            (true, true) => "mysql+config",
            (true, false) => "mysql",
            _ => "config",
        }
    }
}

/// A static key of the router's configuration: what it authenticates as.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticKey {
    pub account: String,
    pub rate_per_min: u32,
    pub decision_quota: u64,
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
            self.plan.chars().count() <= MAX_PLAN_CHARS,
            "key plan must be at most {MAX_PLAN_CHARS} characters"
        );
        ensure!(
            self.label.chars().count() <= MAX_LABEL_CHARS,
            "key label must be at most {MAX_LABEL_CHARS} characters"
        );
        if let Some(o) = &self.router {
            ensure!(
                o.db || o.config.is_some(),
                "a router origin names the database, the configuration or both"
            );
            if let Some(c) = &o.config {
                check_account(&c.account)?;
            }
        }
        self.credit()?;
        self.oracle_budget()?;
        Ok(())
    }

    /// The router's static key this record authenticates as at `now`: a key
    /// that is also a static key of the router's configuration and whose
    /// database row is inactive or expired. The router lays its database over
    /// its configuration and loads only active, unexpired rows (router
    /// `auth.rs:83-89`, `main.rs:43-60`, `store.rs:322`), so the static key
    /// answers then.
    pub fn static_fallback(&self, now: u64) -> Option<KeyRecord> {
        let o = self.router.as_ref()?;
        let s = o.config.as_ref()?;
        if !o.db || (self.active && !self.is_expired(now)) {
            return None;
        }
        Some(KeyRecord {
            hash: self.hash.clone(),
            account: s.account.clone(),
            plan: ROUTER_STATIC_PLAN.to_string(),
            label: String::new(),
            created: self.created,
            expires: None,
            active: true,
            rate_per_min: s.rate_per_min,
            decision_quota: s.decision_quota,
            token_quota: 0,
            credit_usd: None,
            oracle_budget_usd: None,
            oracle_allowed: self.oracle_allowed,
            router: self.router.clone(),
        })
    }

    /// What the key is at `now`: its [`KeyRecord::static_fallback`], else
    /// the record itself.
    pub fn effective(&self, now: u64) -> std::borrow::Cow<'_, KeyRecord> {
        match self.static_fallback(now) {
            Some(s) => std::borrow::Cow::Owned(s),
            None => std::borrow::Cow::Borrowed(self),
        }
    }

    /// Whether a revocation here changes anything: active, or answering as
    /// its static key.
    fn revocable(&self) -> bool {
        self.active
            || self
                .router
                .as_ref()
                .is_some_and(|o| o.db && o.config.is_some())
    }

    /// Revoke here: inactive, and no longer a router key (no static
    /// fallback; a later import keeps it as this server's record and never
    /// re-activates it).
    fn revoke(&mut self) {
        self.active = false;
        self.router = None;
    }

    /// Whether `account` names this key (its own or its static key's).
    fn has_account(&self, account: &str) -> bool {
        self.account == account
            || self
                .router
                .as_ref()
                .and_then(|o| o.config.as_ref())
                .is_some_and(|c| c.account == account)
    }

    /// The listing entry (spec §5b: hash12, limits, usage added by the caller).
    pub fn listing(&self, now: u64) -> Value {
        let mut v = json!({
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
        });
        if let Some(o) = &self.router {
            v["router_origin"] = json!(o.name());
        }
        v
    }
}

/// An account id: 1..128 Unicode characters, any text the router's
/// `account VARCHAR(128)` column holds (router `store.rs:122`; its admin API
/// takes any non-blank string, `api.rs:454-457`). Control characters are
/// kept as they are in `keys.json` (JSON-escaped) and escaped by [`shown`]
/// wherever an account is printed or logged.
pub fn check_account(a: &str) -> Result<()> {
    let n = a.chars().count();
    ensure!(
        (1..=MAX_ACCOUNT_CHARS).contains(&n),
        "account '{}' must be 1..{MAX_ACCOUNT_CHARS} characters (it has {n})",
        shown(&a.chars().take(MAX_ACCOUNT_CHARS).collect::<String>())
    );
    Ok(())
}

/// A text as a log line or a CLI table shows it: control and other
/// non-printable characters escaped (`\n`, `\u{7}`), printable Unicode kept.
pub fn shown(text: &str) -> String {
    text.escape_debug().to_string()
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
        if let Some(s) = rec.static_fallback(now) {
            return Ok(s);
        }
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
        self.modify_if(|keys| Ok((f(keys)?, true)))
    }

    /// [`KeyStore::modify`] that leaves the file untouched (same bytes, same
    /// mtime) when `f` reports no change.
    fn modify_if<T>(&self, f: impl FnOnce(&mut Vec<KeyRecord>) -> Result<(T, bool)>) -> Result<T> {
        let _w = self.write.lock();
        let mut file = read_file(&self.path)?;
        let (out, changed) = f(&mut file.keys)?;
        if !changed {
            return Ok(out);
        }
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
            router: None,
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

    /// Revoke every active key of an account (its own account or, for an
    /// imported router key, its static key's); returns how many were revoked.
    /// A revoked router key loses its origin: no static fallback, and a later
    /// import never re-activates it.
    pub fn revoke_account(&self, account: &str) -> Result<usize> {
        self.modify(|keys| {
            let mut n = 0;
            for k in keys
                .iter_mut()
                .filter(|k| k.has_account(account) && k.revocable())
            {
                k.revoke();
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
                if keys[i].revocable() {
                    keys[i].revoke();
                    n += 1;
                }
            }
            Ok(n)
        })
    }

    /// Import keys read from a cortiq-router export ([`read_router_keys`],
    /// spec §4.15). Idempotent, never deletes and never re-activates:
    ///
    /// * a hash not stored yet is added with the export's account, plan,
    ///   label, limits, expiry and `active` (inactive and expired keys are
    ///   kept and refused at authentication, as the router refuses them) and
    ///   its origin ([`RouterOrigin`]);
    /// * a key in both the router's database and its configuration follows
    ///   the router's precedence whatever the import order: the database row
    ///   is laid over the configuration key (the record takes the row, the
    ///   static key is kept as its [`KeyRecord::static_fallback`] for when the
    ///   row is inactive or expired) — `layered`;
    /// * a stored active key the export's database marks inactive is revoked
    ///   here too (a revocation in the router reaches this server on the next
    ///   import; a static key of the configuration still answers, as in the
    ///   router);
    /// * a stored key with the same router attributes is unchanged; one with
    ///   other attributes is kept as it is (this server's record wins), and so
    ///   is a key created here or revoked here;
    /// * `oracle_allowed`: new keys take [`RouterKeys::oracle_allowed`]
    ///   ([`ROUTER_ORACLE_ALLOWED`] unless set); when it was set explicitly
    ///   ([`RouterKeys::with_oracle_allowed`]) the stored keys of this export
    ///   that came from the router take it too (`oracle_updated`).
    ///
    /// `keys.json` is written only when something changed: importing the same
    /// export again leaves the file byte for byte (and its mtime) as it was.
    pub fn import_router_keys(&self, keys: &RouterKeys, now: u64) -> Result<ImportReport> {
        self.modify_if(|stored| {
            let mut r = ImportReport {
                format: Some(keys.format),
                read: keys.entries,
                ignored_empty: keys.ignored_empty,
                duplicates: keys.duplicates,
                emails_not_stored: keys.emails,
                oracle_allowed: keys.oracle_allowed(),
                ..ImportReport::default()
            };
            let from_db = keys.format == ImportFormat::MysqlJson;
            let mut at: HashMap<String, usize> = stored
                .iter()
                .enumerate()
                .map(|(i, k)| (k.hash.clone(), i))
                .collect();
            for rec in &keys.records {
                let Some(i) = at.get(&rec.hash).copied() else {
                    r.imported += 1;
                    if !rec.active {
                        r.imported_inactive += 1;
                    } else if rec.is_expired(now) {
                        r.imported_expired += 1;
                    }
                    r.accounts.insert(rec.account.clone());
                    at.insert(rec.hash.clone(), stored.len());
                    stored.push(rec.clone());
                    r.written = true;
                    continue;
                };
                let s = &mut stored[i];
                let Some(origin) = s.router.clone() else {
                    // This server's record (created or revoked here).
                    if from_db && s.active && !rec.active {
                        s.active = false;
                        r.revoked += 1;
                        r.written = true;
                    } else if same_router_fields(s, rec) {
                        r.unchanged += 1;
                    } else {
                        r.kept += 1;
                    }
                    continue;
                };
                let incoming_static = rec
                    .router
                    .as_ref()
                    .and_then(|o| o.config.clone())
                    .unwrap_or_else(|| StaticKey {
                        account: rec.account.clone(),
                        rate_per_min: rec.rate_per_min,
                        decision_quota: rec.decision_quota,
                    });
                if from_db && !origin.db {
                    // A configuration key: the database row goes over it.
                    let mut row = rec.clone();
                    row.oracle_allowed = s.oracle_allowed;
                    row.router = Some(RouterOrigin {
                        db: true,
                        config: origin.config,
                    });
                    *s = row;
                    r.layered += 1;
                    r.written = true;
                } else if !from_db && origin.config.is_none() {
                    // A database row that is also a configuration key.
                    s.router = Some(RouterOrigin {
                        db: true,
                        config: Some(incoming_static),
                    });
                    r.layered += 1;
                    r.written = true;
                } else if from_db && s.active && !rec.active {
                    s.active = false;
                    r.revoked += 1;
                    r.written = true;
                } else if (from_db && same_router_fields(s, rec))
                    || (!from_db && origin.config.as_ref() == Some(&incoming_static))
                {
                    r.unchanged += 1;
                } else {
                    r.kept += 1;
                }
                if let Some(v) = keys.oracle_explicit
                    && s.oracle_allowed != v
                {
                    s.oracle_allowed = v;
                    r.oracle_updated += 1;
                    r.written = true;
                }
            }
            r.static_fallback = keys
                .records
                .iter()
                .filter_map(|k| at.get(&k.hash))
                .filter(|&&i| stored[i].static_fallback(now).is_some())
                .count();
            let changed = r.written;
            Ok((r, changed))
        })
    }
}

/// The attributes a router export carries (the creation time is not one of
/// them: a configuration key has none).
fn same_router_fields(a: &KeyRecord, b: &KeyRecord) -> bool {
    a.account == b.account
        && a.plan == b.plan
        && a.label == b.label
        && a.expires == b.expires
        && a.active == b.active
        && a.rate_per_min == b.rate_per_min
        && a.decision_quota == b.decision_quota
}

// ------------------------------------------------------------------ router import

/// Format of a cortiq-router key export (`keys import --format`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportFormat {
    /// Rows of the router's MySQL `api_keys` table as JSON.
    MysqlJson,
    /// The router's TOML configuration (its `[[api_keys]]`).
    RouterToml,
}

impl ImportFormat {
    pub const NAMES: [&'static str; 2] = ["mysql-json", "router-toml"];

    pub fn name(self) -> &'static str {
        match self {
            Self::MysqlJson => "mysql-json",
            Self::RouterToml => "router-toml",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "mysql-json" => Ok(Self::MysqlJson),
            "router-toml" => Ok(Self::RouterToml),
            _ => bail!("unknown key export format '{s}' (mysql-json | router-toml)"),
        }
    }

    /// The format of a file given without `--format`: `.toml` is the router
    /// configuration; `.json`, `.jsonl`, `.ndjson` a MySQL export; otherwise
    /// JSON content is a MySQL export and anything else the configuration.
    pub fn detect(path: &Path, bytes: &[u8]) -> Self {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("toml") => Self::RouterToml,
            Some("json" | "jsonl" | "ndjson") => Self::MysqlJson,
            _ if json_documents(bytes).is_ok() => Self::MysqlJson,
            _ => Self::RouterToml,
        }
    }
}

/// The columns of the router's MySQL `api_keys` table
/// (cortiq-router `store.rs:119-130`, `ensure_api_key_columns`).
pub const ROUTER_KEY_COLUMNS: [&str; 10] = [
    "key_hash",
    "account",
    "plan",
    "email",
    "label",
    "active",
    "rate_per_min",
    "decision_quota",
    "expires_at",
    "created_at",
];

/// The columns of the router's MySQL `usage_counters` table
/// (cortiq-router `store.rs:132-136`).
pub const ROUTER_USAGE_COLUMNS: [&str; 3] = ["account", "decisions", "oracle_calls"];

/// The plan the router gives the keys of its configuration
/// (cortiq-router `main.rs:336`).
pub const ROUTER_STATIC_PLAN: &str = "static";

/// The id of the ledger lines that carry router usage over
/// ([`import_router_usage`]).
pub const ROUTER_USAGE_RECORD_ID: &str = "import:cortiq-router:usage_counters";

/// The model name of those lines.
pub const ROUTER_USAGE_MODEL: &str = "cortiq-router";

/// Keys read from a router export: hashed, checked, nothing written yet.
/// Holds no raw key (a configuration's keys are hashed while it is read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterKeys {
    pub format: ImportFormat,
    /// Rows (MySQL) or `[[api_keys]]` entries (configuration) read.
    pub entries: usize,
    /// One record per distinct key, in export order.
    pub records: Vec<KeyRecord>,
    /// Configuration entries with an empty `key` (the router skips them,
    /// `main.rs:330`).
    pub ignored_empty: usize,
    /// Configuration entries that repeat an earlier key: the last one wins,
    /// as in the router's key map.
    pub duplicates: usize,
    /// MySQL rows with an email: not stored (keys.json holds no email).
    pub emails: usize,
    /// `oracle_allowed` given explicitly for this import (`keys import
    /// --oracle-allowed`); `None` = [`ROUTER_ORACLE_ALLOWED`] for new keys,
    /// stored keys unchanged.
    pub oracle_explicit: Option<bool>,
}

impl RouterKeys {
    /// Set `oracle_allowed` of every key of the export explicitly: new keys
    /// take it, and so do the stored keys that came from the router.
    pub fn with_oracle_allowed(mut self, allowed: bool) -> Self {
        self.oracle_explicit = Some(allowed);
        for r in &mut self.records {
            r.oracle_allowed = allowed;
        }
        self
    }

    /// `oracle_allowed` of the new keys of this import.
    pub fn oracle_allowed(&self) -> bool {
        self.oracle_explicit.unwrap_or(ROUTER_ORACLE_ALLOWED)
    }
}

/// What an import of keys did (no key and no hash in it).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportReport {
    pub format: Option<ImportFormat>,
    pub read: usize,
    /// New keys (including the inactive and expired ones below).
    pub imported: usize,
    pub imported_inactive: usize,
    pub imported_expired: usize,
    /// Already stored with the same router attributes.
    pub unchanged: usize,
    /// Stored active, inactive in the export: revoked here.
    pub revoked: usize,
    /// Stored with other attributes: kept as they are.
    pub kept: usize,
    pub ignored_empty: usize,
    pub duplicates: usize,
    pub emails_not_stored: usize,
    /// Keys in both the router's database and its configuration, joined by
    /// this import (the row laid over the static key, as in the router).
    pub layered: usize,
    /// Keys of the export that answer as their static key now (database row
    /// inactive or expired, also a key of the configuration).
    pub static_fallback: usize,
    /// `oracle_allowed` of the new keys.
    pub oracle_allowed: bool,
    /// Stored router keys whose `oracle_allowed` an explicit
    /// `--oracle-allowed` changed.
    pub oracle_updated: usize,
    /// Accounts of the new keys.
    pub accounts: BTreeSet<String>,
    /// Whether keys.json was written.
    pub written: bool,
}

impl ImportReport {
    /// New keys that authenticate now.
    pub fn imported_active(&self) -> usize {
        self.imported - self.imported_inactive - self.imported_expired
    }

    pub fn to_json(&self) -> Value {
        json!({
            "format": self.format.map(ImportFormat::name),
            "read": self.read,
            "imported": self.imported,
            "imported_active": self.imported_active(),
            "imported_inactive": self.imported_inactive,
            "imported_expired": self.imported_expired,
            "unchanged": self.unchanged,
            "revoked": self.revoked,
            "kept": self.kept,
            "ignored_empty": self.ignored_empty,
            "duplicates": self.duplicates,
            "emails_not_stored": self.emails_not_stored,
            "layered": self.layered,
            "static_fallback": self.static_fallback,
            "oracle_allowed": self.oracle_allowed,
            "oracle_updated": self.oracle_updated,
            "accounts": self.accounts,
            "written": self.written,
        })
    }
}

/// Read the keys of a cortiq-router export (spec §4.15). Every entry is
/// checked before anything is written: one bad entry refuses the whole file.
/// Error messages name the row or line, never a key or a hash.
///
/// * `mysql-json`: rows of the router's `api_keys` table (columns
///   [`ROUTER_KEY_COLUMNS`]; a missing column takes the table's default, an
///   unknown one is refused). `key_hash` must be 64 lowercase hex characters,
///   the router's `sha256(raw)` (`store.rs:54-57`), so the raw keys keep
///   working without reissue. Numbers may be JSON numbers or numeric
///   strings; `active` is true only for 1 (the router loads `WHERE
///   active=1`, `store.rs:322`); `rate_per_min`/`decision_quota` 0 stay 0 =
///   unlimited (router `auth.rs:152,167`); the email is not stored. Accepted
///   containers: an array of rows (`JSON_ARRAYAGG(JSON_OBJECT(…))`, MySQL
///   Workbench, `mysqlsh --result-format=json/array`), JSON lines
///   (`mysql -N -B -e "SELECT JSON_OBJECT(…) FROM api_keys"`, `mysqlsh
///   --result-format=ndjson`), MySQL Shell `--json` documents (`{"rows":
///   […]}`, its info and warning documents skipped), `{"api_keys"|"data":
///   […]}`, and a phpMyAdmin JSON export (its `api_keys` table).
/// * `router-toml`: the router's configuration; its `[[api_keys]]` entries
///   `{key, account = "default", rate_per_min = 0, decision_quota = 0}`
///   (router `config.rs:136-156`) become plan `static` keys without expiry
///   created now (`main.rs:327-341`). Raw keys are hashed as they are read
///   and never kept; an empty key is skipped and a repeated key keeps its
///   last entry, as in the router.
pub fn read_router_keys(bytes: &[u8], format: ImportFormat, now: u64) -> Result<RouterKeys> {
    match format {
        ImportFormat::MysqlJson => mysql_keys(bytes),
        ImportFormat::RouterToml => toml_keys(bytes, now),
    }
}

fn mysql_keys(bytes: &[u8]) -> Result<RouterKeys> {
    let rows = export_rows(bytes, "api_keys", &ROUTER_KEY_COLUMNS)?;
    let mut out = RouterKeys {
        format: ImportFormat::MysqlJson,
        entries: rows.len(),
        records: Vec::with_capacity(rows.len()),
        ignored_empty: 0,
        duplicates: 0,
        emails: 0,
        oracle_explicit: None,
    };
    let mut first: HashMap<String, usize> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        let (rec, email) = mysql_key_record(row).with_context(|| format!("row {}", i + 1))?;
        if let Some(j) = first.insert(rec.hash.clone(), i + 1) {
            bail!(
                "rows {j} and {} have the same key_hash, the primary key of the router's table: the export is broken",
                i + 1
            );
        }
        out.emails += usize::from(email);
        out.records.push(rec);
    }
    Ok(out)
}

fn mysql_key_record(row: &Map<String, Value>) -> Result<(KeyRecord, bool)> {
    let hash = match row.get("key_hash") {
        Some(Value::String(s)) => s.trim().to_string(),
        None | Some(Value::Null) => bail!("key_hash is missing"),
        Some(_) => bail!("key_hash must be a string"),
    };
    ensure!(
        is_hash(&hash),
        "key_hash is not a sha256 in lowercase hex (64 characters 0-9a-f, as the router's store.rs:54-57 writes it)"
    );
    let account =
        text_field(row, "account")?.ok_or_else(|| anyhow::anyhow!("account is missing"))?;
    let email = text_field(row, "email")?.is_some_and(|e| !e.trim().is_empty());
    let active = match row.get("active") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => int_field(row, "active")? == Some(1),
    };
    let rate = uint_field(row, "rate_per_min")?.unwrap_or(0);
    let rec = KeyRecord {
        hash,
        account,
        plan: text_field(row, "plan")?.unwrap_or_default(),
        label: text_field(row, "label")?.unwrap_or_default(),
        created: uint_field(row, "created_at")?.unwrap_or(0),
        expires: uint_field(row, "expires_at")?,
        active,
        rate_per_min: u32::try_from(rate)
            .map_err(|_| anyhow::anyhow!("rate_per_min is larger than the router's INT"))?,
        decision_quota: uint_field(row, "decision_quota")?.unwrap_or(0),
        token_quota: 0,
        credit_usd: None,
        oracle_budget_usd: None,
        oracle_allowed: ROUTER_ORACLE_ALLOWED,
        router: Some(RouterOrigin {
            db: true,
            config: None,
        }),
    };
    rec.validate()?;
    Ok((rec, email))
}

fn toml_keys(bytes: &[u8], now: u64) -> Result<RouterKeys> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("the router configuration is not UTF-8 text"))?;
    let root = toml_lite::parse(text)?;
    let entries: Vec<&Map<String, Value>> = match root.get("api_keys") {
        None => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_object()
                    .ok_or_else(|| anyhow::anyhow!("api_keys entry {} is not a table", i + 1))
            })
            .collect::<Result<_>>()?,
        Some(_) => bail!("api_keys must be an array of tables ([[api_keys]])"),
    };
    let mut out = RouterKeys {
        format: ImportFormat::RouterToml,
        entries: entries.len(),
        records: Vec::with_capacity(entries.len()),
        ignored_empty: 0,
        duplicates: 0,
        emails: 0,
        oracle_explicit: None,
    };
    let mut at: HashMap<String, usize> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        let rec = (|| -> Result<Option<KeyRecord>> {
            let hash = match e.get("key") {
                None => return Ok(None),
                Some(Value::String(k)) if k.is_empty() => return Ok(None),
                Some(Value::String(k)) => hash_key(k),
                Some(_) => bail!("key must be a string"),
            };
            let account = match e.get("account") {
                None => "default".to_string(),
                Some(Value::String(a)) => a.clone(),
                Some(_) => bail!("account must be a string"),
            };
            let rate = toml_uint(e, "rate_per_min")?;
            let static_key = StaticKey {
                account,
                rate_per_min: u32::try_from(rate)
                    .map_err(|_| anyhow::anyhow!("rate_per_min is larger than the router's u32"))?,
                decision_quota: toml_uint(e, "decision_quota")?,
            };
            let rec = KeyRecord {
                hash,
                account: static_key.account.clone(),
                plan: ROUTER_STATIC_PLAN.to_string(),
                label: String::new(),
                created: now,
                expires: None,
                active: true,
                rate_per_min: static_key.rate_per_min,
                decision_quota: static_key.decision_quota,
                token_quota: 0,
                credit_usd: None,
                oracle_budget_usd: None,
                oracle_allowed: ROUTER_ORACLE_ALLOWED,
                router: Some(RouterOrigin {
                    db: false,
                    config: Some(static_key),
                }),
            };
            rec.validate()?;
            Ok(Some(rec))
        })()
        .with_context(|| format!("api_keys entry {}", i + 1))?;
        let Some(rec) = rec else {
            out.ignored_empty += 1;
            continue;
        };
        match at.get(&rec.hash) {
            Some(&j) => {
                out.records[j] = rec;
                out.duplicates += 1;
            }
            None => {
                at.insert(rec.hash.clone(), out.records.len());
                out.records.push(rec);
            }
        }
    }
    Ok(out)
}

fn toml_uint(e: &Map<String, Value>, k: &str) -> Result<u64> {
    match e.get(k) {
        None => Ok(0),
        Some(v) => v
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("{k} must be a non-negative integer")),
    }
}

/// Every JSON document of `bytes` (one, several concatenated, or JSON lines).
fn json_documents(bytes: &[u8]) -> Result<Vec<Value>> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    serde_json::Deserializer::from_slice(bytes)
        .into_iter::<Value>()
        .map(|d| d.map_err(|e| anyhow::anyhow!("not JSON: {e}")))
        .collect()
}

/// MySQL Shell documents that carry no rows.
const MYSQLSH_META: [&str; 11] = [
    "info",
    "note",
    "warning",
    "warnings",
    "warningCount",
    "warningsCount",
    "hasData",
    "executionTime",
    "affectedRowCount",
    "affectedItemsCount",
    "autoIncrementValue",
];

/// A column name as it may appear in a message: short identifiers only (a
/// misplaced key must not be echoed).
fn shown_column(k: &str) -> String {
    let ident = !k.is_empty()
        && k.len() <= 32
        && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !k.as_bytes()[0].is_ascii_digit();
    if ident {
        format!("'{k}'")
    } else {
        "(a name that is not an identifier)".to_string()
    }
}

/// The rows of `table` in a JSON export (see [`read_router_keys`] for the
/// containers), every column checked against `columns`.
fn export_rows(bytes: &[u8], table: &str, columns: &[&str]) -> Result<Vec<Map<String, Value>>> {
    fn objects(items: &[Value], table: &str) -> Result<Vec<Map<String, Value>>> {
        items
            .iter()
            .map(|v| {
                v.as_object()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("every row of {table} must be a JSON object"))
            })
            .collect()
    }
    let mut rows = Vec::new();
    for (d, doc) in json_documents(bytes)?.into_iter().enumerate() {
        let d = d + 1;
        match doc {
            Value::Array(items) => {
                let phpmyadmin = items.iter().any(|v| {
                    v.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|t| matches!(t, "header" | "database" | "table"))
                });
                if !phpmyadmin {
                    rows.extend(objects(&items, table)?);
                    continue;
                }
                let mut found = false;
                for t in &items {
                    if t.get("type").and_then(Value::as_str) == Some("table")
                        && t.get("name").and_then(Value::as_str) == Some(table)
                    {
                        found = true;
                        match t.get("data") {
                            Some(Value::Array(a)) => rows.extend(objects(a, table)?),
                            _ => bail!(
                                "document {d}: the phpMyAdmin table {table} has no data array"
                            ),
                        }
                    }
                }
                ensure!(
                    found,
                    "document {d}: the phpMyAdmin export has no table {table}"
                );
            }
            Value::Object(m) => {
                if let Some(v) = ["rows", "data", table].iter().find_map(|k| m.get(*k)) {
                    match v {
                        Value::Array(a) => rows.extend(objects(a, table)?),
                        _ => bail!("document {d}: its rows are not an array"),
                    }
                } else if m.keys().any(|k| columns.contains(&k.as_str())) {
                    rows.push(m);
                } else if m.contains_key("error") {
                    bail!("document {d} is a MySQL error, not rows of {table}");
                } else if !m.keys().all(|k| MYSQLSH_META.contains(&k.as_str())) {
                    bail!(
                        "document {d} is not a row of {table} (columns {})",
                        columns.join(", ")
                    );
                }
            }
            _ => bail!(
                "document {d} is not a row of {table}: expected objects with the columns {}",
                columns.join(", ")
            ),
        }
    }
    for (i, row) in rows.iter().enumerate() {
        for k in row.keys() {
            ensure!(
                columns.contains(&k.as_str()),
                "row {}: column {} is not a column of the router's {table} table ({})",
                i + 1,
                shown_column(k),
                columns.join(", ")
            );
        }
    }
    Ok(rows)
}

/// A MySQL integer (JSON number or numeric string); `null`/absent = `None`.
fn int_field(row: &Map<String, Value>, k: &str) -> Result<Option<i64>> {
    match row.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_i64()
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("{k} must be an integer")),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => s
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("{k} must be an integer")),
        Some(_) => bail!("{k} must be an integer"),
    }
}

fn uint_field(row: &Map<String, Value>, k: &str) -> Result<Option<u64>> {
    int_field(row, k)?
        .map(|v| u64::try_from(v).map_err(|_| anyhow::anyhow!("{k} must not be negative")))
        .transpose()
}

fn text_field(row: &Map<String, Value>, k: &str) -> Result<Option<String>> {
    match row.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => bail!("{k} must be a string"),
    }
}

// ------------------------------------------------------------------ router usage

/// One row of the router's `usage_counters` (lifetime counters per account).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterUsage {
    pub account: String,
    pub decisions: u64,
    pub oracle_calls: u64,
}

/// Read a JSON export of the router's MySQL `usage_counters` table (columns
/// [`ROUTER_USAGE_COLUMNS`], the containers of [`read_router_keys`]). An
/// account may appear once (the table's primary key).
pub fn read_router_usage(bytes: &[u8]) -> Result<Vec<RouterUsage>> {
    let rows = export_rows(bytes, "usage_counters", &ROUTER_USAGE_COLUMNS)?;
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let u = (|| -> Result<RouterUsage> {
            let account =
                text_field(row, "account")?.ok_or_else(|| anyhow::anyhow!("account is missing"))?;
            check_account(&account)?;
            Ok(RouterUsage {
                account,
                decisions: uint_field(row, "decisions")?.unwrap_or(0),
                oracle_calls: uint_field(row, "oracle_calls")?.unwrap_or(0),
            })
        })()
        .with_context(|| format!("row {}", i + 1))?;
        if let Some(j) = seen.insert(u.account.clone(), i + 1) {
            bail!(
                "rows {j} and {} are both account {}, the primary key of the router's table",
                i + 1,
                u.account
            );
        }
        out.push(u);
    }
    Ok(out)
}

/// What an import of usage did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UsageImportReport {
    pub read: usize,
    /// Accounts whose counters grew since the last import: one ledger line each.
    pub carried: usize,
    /// Accounts already carried over up to these counters.
    pub unchanged: usize,
    /// Accounts whose export is below what was carried over before (an older
    /// export): left alone.
    pub behind: usize,
    pub decisions: u64,
    pub oracle_calls: u64,
    pub accounts: BTreeSet<String>,
    /// Whether the ledger was written.
    pub written: bool,
}

impl UsageImportReport {
    pub fn to_json(&self) -> Value {
        json!({
            "read": self.read,
            "carried": self.carried,
            "unchanged": self.unchanged,
            "behind": self.behind,
            "decisions": self.decisions,
            "oracle_calls": self.oracle_calls,
            "accounts": self.accounts,
            "written": self.written,
        })
    }
}

/// Carry the router's usage counters into the usage ledger so that quotas
/// continue (spec §4.15; the router compares an account's lifetime
/// `decisions` with `decision_quota`, `auth.rs:166-173`, as the service
/// compares the ledger's `decisions`). Per account one line
/// `{id: "import:cortiq-router:usage_counters", model: "cortiq-router",
/// questions: Δdecisions, oracle_calls: Δoracle_calls, actions.local:
/// Δdecisions}` with no tokens and no cost, where Δ is the export minus what
/// earlier imports carried over (read back from the ledger): importing the
/// same export again writes nothing, a newer export adds only the growth,
/// an older one is left alone. The router does not record which decisions
/// its oracle answered, so the carried decisions are booked as `local`.
///
/// The caller holds the state directory's `LOCK` (no server is running).
pub fn import_router_usage(
    ledger: &UsageLedger,
    rows: &[RouterUsage],
    now: u64,
) -> Result<UsageImportReport> {
    let mut done: HashMap<String, (u64, u64)> = HashMap::new();
    ledger.for_each_record(|r| {
        if r.id == ROUTER_USAGE_RECORD_ID {
            let e = done.entry(r.account.clone()).or_default();
            e.0 += r.questions;
            e.1 += r.oracle_calls;
        }
    })?;
    let mut rep = UsageImportReport {
        read: rows.len(),
        ..UsageImportReport::default()
    };
    for u in rows {
        let (d0, o0) = done.get(&u.account).copied().unwrap_or_default();
        if u.decisions < d0 || u.oracle_calls < o0 {
            rep.behind += 1;
            continue;
        }
        let (dd, dor) = (u.decisions - d0, u.oracle_calls - o0);
        if dd == 0 && dor == 0 {
            rep.unchanged += 1;
            continue;
        }
        ledger.append(&UsageRecord {
            ts: now,
            id: ROUTER_USAGE_RECORD_ID.to_string(),
            account: u.account.clone(),
            key12: None,
            model: ROUTER_USAGE_MODEL.to_string(),
            generation: 0,
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 0.0,
            cost_local_usd: 0.0,
            cost_oracle_usd: 0.0,
            oracle_calls: dor,
            cache_hits: 0,
            questions: dd,
            actions: Actions {
                local: dd,
                ..Actions::default()
            },
        })?;
        rep.carried += 1;
        rep.decisions += dd;
        rep.oracle_calls += dor;
        rep.accounts.insert(u.account.clone());
    }
    if rep.carried > 0 {
        ledger.snapshot()?;
        rep.written = true;
    }
    Ok(rep)
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

    // -------------------------------------------------------------- router import

    const RAW_A: &str = "cortiq_0123456789abcdef0123456789abcdef01234567";
    const RAW_B: &str = "cortiq_fedcba9876543210fedcba9876543210fedcba98";

    fn row(raw: &str, extra: Value) -> Value {
        let mut m = json!({"key_hash": hash_key(raw), "account": "acct_a1b2c3d4e5f6"});
        for (k, v) in extra.as_object().unwrap() {
            m[k] = v.clone();
        }
        m
    }

    fn mysql(v: &str) -> RouterKeys {
        read_router_keys(v.as_bytes(), ImportFormat::MysqlJson, 99).unwrap()
    }

    fn refused(bytes: &str, format: ImportFormat) -> String {
        let e = read_router_keys(bytes.as_bytes(), format, 99).unwrap_err();
        format!("{e:#}")
    }

    #[test]
    fn mysql_export_containers_all_read_the_same_rows() {
        let a = row(RAW_A, json!({"plan": "pro", "rate_per_min": 600}));
        let b = row(RAW_B, json!({"account": "beta", "active": 0}));
        let want = mysql(&json!([a, b]).to_string());
        assert_eq!(want.records.len(), 2);
        let lines = format!("{a}\n\n{b}\n");
        let mysqlsh = format!(
            "{}\n{}",
            json!({"warning": "Using a password on the command line interface can be insecure."}),
            serde_json::to_string_pretty(&json!({"hasData": true, "rows": [a, b],
                "executionTime": "0.0008 sec", "affectedRowCount": 0, "warningCount": 0,
                "warnings": [], "info": "", "autoIncrementValue": 0}))
            .unwrap()
        );
        let phpmyadmin = json!([
            {"type": "header", "version": "5.2.1", "comment": "Export to JSON plugin for PHPMyAdmin"},
            {"type": "database", "name": "cortiq"},
            {"type": "table", "name": "usage_counters", "database": "cortiq", "data": [{"account": "x"}]},
            {"type": "table", "name": "api_keys", "database": "cortiq", "data": [a, b]}
        ]);
        let wrapped = json!({"api_keys": [a, b]});
        for text in [lines, mysqlsh, phpmyadmin.to_string(), wrapped.to_string()] {
            assert_eq!(mysql(&text), want, "{text}");
        }
        assert_eq!(mysql("").records.len(), 0);
        assert_eq!(mysql("[]").entries, 0);
    }

    #[test]
    fn mysql_rows_map_like_the_router() {
        let k = mysql(
            &json!([
                // phpMyAdmin style: every value a string.
                row(
                    RAW_A,
                    json!({"plan": "developer", "email": "ops@example.com", "label": "ci",
                    "active": "1", "rate_per_min": "120", "decision_quota": "100000",
                    "expires_at": "2000000000", "created_at": "1700000000"})
                ),
                // Only the NOT NULL columns: the table's defaults (active 1, limits 0).
                row(RAW_B, json!({"email": "", "expires_at": null})),
                row("inactive", json!({"active": 0})),
                row("tinyint-2", json!({"active": 2})),
                row("bool", json!({"active": false})),
                row("expired", json!({"expires_at": 1000})),
            ])
            .to_string(),
        );
        let r = &k.records;
        assert_eq!(
            (r[0].plan.as_str(), r[0].label.as_str(), r[0].active),
            ("developer", "ci", true)
        );
        assert_eq!(
            (
                r[0].rate_per_min,
                r[0].decision_quota,
                r[0].expires,
                r[0].created
            ),
            (120, 100_000, Some(2_000_000_000), 1_700_000_000)
        );
        assert_eq!(
            (
                r[1].plan.as_str(),
                r[1].active,
                r[1].rate_per_min,
                r[1].decision_quota
            ),
            ("", true, 0, 0)
        );
        assert_eq!((r[1].expires, r[1].created), (None, 0));
        assert_eq!(
            r[2..5].iter().map(|k| k.active).collect::<Vec<_>>(),
            [false, false, false]
        );
        assert!(r[5].is_expired(1000) && !r[5].is_expired(999));
        // The router escalated for every key: imported keys may use the oracle.
        assert!(r.iter().all(|k| k.oracle_allowed && k.token_quota == 0));
        assert!(r.iter().all(|k| k.router
            == Some(RouterOrigin {
                db: true,
                config: None
            })));
        assert_eq!(k.emails, 1, "one non-empty email, never stored");
        assert!(
            !serde_json::to_string(&k.records)
                .unwrap()
                .contains("example.com")
        );
    }

    #[test]
    fn malformed_exports_are_refused_without_echoing_a_key() {
        let h = hash_key(RAW_A);
        let upper = h.to_ascii_uppercase();
        let cases: Vec<(String, &str)> = vec![
            (
                json!([{"key_hash": upper, "account": "a"}]).to_string(),
                "lowercase hex",
            ),
            (
                json!([{"key_hash": &h[..63], "account": "a"}]).to_string(),
                "lowercase hex",
            ),
            (
                json!([{"key_hash": format!("{}g", &h[..63]), "account": "a"}]).to_string(),
                "lowercase hex",
            ),
            (json!([{"account": "a"}]).to_string(), "key_hash is missing"),
            (
                json!([{"key_hash": 12, "account": "a"}]).to_string(),
                "must be a string",
            ),
            (json!([{"key_hash": h}]).to_string(), "account is missing"),
            (
                json!([{"key_hash": h, "account": ""}]).to_string(),
                "must be 1..128 characters",
            ),
            (
                json!([{"key_hash": h, "account": "я".repeat(129)}]).to_string(),
                "must be 1..128 characters (it has 129)",
            ),
            (
                json!([{"key_hash": h, "account": "a", "plan": "p".repeat(65)}]).to_string(),
                "plan must be at most 64 characters",
            ),
            (
                json!([{"key_hash": h, "account": "a", "label": "ж".repeat(256)}]).to_string(),
                "label must be at most 255 characters",
            ),
            (
                json!([{"key_hash": h, "account": "a", "key": RAW_A}]).to_string(),
                "column 'key'",
            ),
            (
                json!([{"key_hash": h, "account": "a", (RAW_A): 1}]).to_string(),
                "not an identifier",
            ),
            (
                json!([{"key_hash": h, "account": "a", "rate_per_min": -1}]).to_string(),
                "negative",
            ),
            (
                json!([{"key_hash": h, "account": "a", "rate_per_min": 1u64 << 32}]).to_string(),
                "INT",
            ),
            (
                json!([{"key_hash": h, "account": "a", "decision_quota": "lots"}]).to_string(),
                "integer",
            ),
            (
                json!([{"key_hash": h, "account": "a"}, {"key_hash": h, "account": "b"}])
                    .to_string(),
                "rows 1 and 2 have the same key_hash",
            ),
            (
                json!([{"account": "a", "decisions": 3}]).to_string(),
                "column 'decisions'",
            ),
            (json!({"error": "Access denied"}).to_string(), "MySQL error"),
            (
                json!([{"type": "header"}, {"type": "table", "name": "other", "data": []}])
                    .to_string(),
                "no table api_keys",
            ),
            (format!("[{{\"key_hash\": \"{h}\""), "not JSON"),
        ];
        for (text, want) in cases {
            let e = refused(&text, ImportFormat::MysqlJson);
            assert!(e.contains(want), "{text}: {e}");
            for secret in [RAW_A, h.as_str(), upper.as_str(), &h[..12]] {
                assert!(!e.contains(secret), "{e}");
            }
        }
    }

    #[test]
    fn router_toml_hashes_raw_keys_like_the_router() {
        let text = format!(
            r#"
bind = "0.0.0.0:8080"
[[api_keys]]
key = "{RAW_A}"
account = "acme-corp"
rate_per_min = 600
decision_quota = 1_000_000
plan = "ignored, as the router ignores unknown fields"

[[api_keys]]
key = ""                # the router skips an empty key
account = "nobody"

[[api_keys]]
key = '{RAW_B}'         # defaults: account "default", unlimited

[[api_keys]]
key = "{RAW_A}"
account = "acme-corp"
rate_per_min = 60       # repeated: the last entry wins

[auth]
require = true
"#
        );
        let k = read_router_keys(text.as_bytes(), ImportFormat::RouterToml, 1234).unwrap();
        assert_eq!(
            (k.entries, k.ignored_empty, k.duplicates, k.records.len()),
            (4, 1, 1, 2)
        );
        let a = &k.records[0];
        assert_eq!(a.hash, hash_key(RAW_A));
        assert_eq!(
            (
                a.account.as_str(),
                a.plan.as_str(),
                a.rate_per_min,
                a.decision_quota
            ),
            ("acme-corp", ROUTER_STATIC_PLAN, 60, 0)
        );
        assert_eq!((a.created, a.expires, a.active), (1234, None, true));
        let b = &k.records[1];
        assert_eq!(
            (
                b.hash.as_str(),
                b.account.as_str(),
                b.rate_per_min,
                b.decision_quota
            ),
            (hash_key(RAW_B).as_str(), "default", 0, 0)
        );
        let shown = format!("{k:?}");
        assert!(!shown.contains(RAW_A) && !shown.contains(RAW_B));
        // An inline array reads too; no api_keys is no key.
        let inline = format!("api_keys = [ {{ key = \"{RAW_A}\", account = \"x\" }} ]\n");
        let k2 = read_router_keys(inline.as_bytes(), ImportFormat::RouterToml, 1).unwrap();
        assert_eq!(k2.records[0].hash, hash_key(RAW_A));
        let none = read_router_keys(b"bind = \"x\"\n", ImportFormat::RouterToml, 1).unwrap();
        assert_eq!((none.entries, none.records.len()), (0, 0));
        for (text, want) in [
            (
                format!("[[api_keys]]\nkey = \"{RAW_A}\"\nrate_per_min = -1\n"),
                "api_keys entry 1: rate_per_min",
            ),
            (
                format!("[[api_keys]]\nkey = \"{RAW_A}\"\nrate_per_min = 4294967296\n"),
                "u32",
            ),
            (
                format!("[[api_keys]]\nkey = \"{RAW_A}\"\ndecision_quota = \"5\"\n"),
                "decision_quota",
            ),
            (
                format!(
                    "[[api_keys]]\nkey = \"{RAW_A}\"\naccount = \"{}\"\n",
                    "x".repeat(129)
                ),
                "account",
            ),
            (format!("[[api_keys]]\nkey = {RAW_A}\n"), "TOML line 2"),
            (format!("api_keys = \"{RAW_A}\"\n"), "array of tables"),
        ] {
            let e = refused(&text, ImportFormat::RouterToml);
            assert!(e.contains(want), "{text}: {e}");
            assert!(!e.contains(RAW_A) && !e.contains(&hash_key(RAW_A)), "{e}");
        }
    }

    #[test]
    fn format_detection() {
        let p = |n: &str| PathBuf::from(n);
        assert_eq!(
            ImportFormat::detect(&p("router.toml"), b"[]"),
            ImportFormat::RouterToml
        );
        assert_eq!(
            ImportFormat::detect(&p("k.json"), b"x"),
            ImportFormat::MysqlJson
        );
        assert_eq!(
            ImportFormat::detect(&p("export"), b"[{\"a\":1}]"),
            ImportFormat::MysqlJson
        );
        assert_eq!(
            ImportFormat::detect(&p("export"), b"[[api_keys]]\n"),
            ImportFormat::RouterToml
        );
        assert_eq!(
            ImportFormat::parse("router-toml").unwrap(),
            ImportFormat::RouterToml
        );
        assert!(ImportFormat::parse("csv").is_err());
    }

    #[test]
    fn import_is_idempotent_revokes_and_never_reactivates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let store = KeyStore::open(&path, "cortiq_").unwrap();
        let export = json!([
            row(
                RAW_A,
                json!({"plan": "pro", "rate_per_min": 0, "decision_quota": 0})
            ),
            row(RAW_B, json!({"account": "beta", "active": 0})),
            row("expired", json!({"account": "old", "expires_at": 10})),
        ])
        .to_string();
        let r = store.import_router_keys(&mysql(&export), 50).unwrap();
        assert_eq!(
            (
                r.imported,
                r.imported_active(),
                r.imported_inactive,
                r.imported_expired,
                r.written
            ),
            (3, 1, 1, 1, true)
        );
        assert_eq!(
            r.accounts.iter().map(String::as_str).collect::<Vec<_>>(),
            ["acct_a1b2c3d4e5f6", "beta", "old"]
        );
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(!text.contains(RAW_A) && text.contains(&hash_key(RAW_A)));
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(store.authenticate(RAW_A, 50).unwrap().plan, "pro");
        assert_eq!(store.authenticate(RAW_B, 50), Err(AuthFailure::Revoked));
        assert_eq!(store.authenticate("expired", 50), Err(AuthFailure::Expired));

        // Again: nothing changes, not even the mtime.
        std::thread::sleep(Duration::from_millis(20));
        let r = store.import_router_keys(&mysql(&export), 60).unwrap();
        assert_eq!((r.imported, r.unchanged, r.written), (0, 3, false));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);

        // The router revoked A and re-enabled B; B's limits changed.
        let later = json!([
            row(RAW_A, json!({"plan": "pro", "active": 0})),
            row(
                RAW_B,
                json!({"account": "beta", "active": 1, "rate_per_min": 5})
            ),
        ])
        .to_string();
        let r = store.import_router_keys(&mysql(&later), 70).unwrap();
        assert_eq!((r.revoked, r.kept, r.imported, r.written), (1, 1, 0, true));
        assert_eq!(store.authenticate(RAW_A, 70), Err(AuthFailure::Revoked));
        assert_eq!(store.authenticate(RAW_B, 70), Err(AuthFailure::Revoked));
    }

    #[test]
    fn usage_import_carries_only_the_growth() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = UsageLedger::open(dir.path()).unwrap();
        let usage = |d: u64, o: u64| {
            read_router_usage(
                json!([{"account": "acme", "decisions": d.to_string(), "oracle_calls": o},
                       {"account": "beta", "decisions": 7}])
                .to_string()
                .as_bytes(),
            )
            .unwrap()
        };
        let r = import_router_usage(&ledger, &usage(100, 4), 1_790_000_000).unwrap();
        assert_eq!(
            (r.carried, r.decisions, r.oracle_calls, r.written),
            (2, 107, 4, true)
        );
        let t = ledger.totals("acme");
        assert_eq!(
            (t.decisions, t.oracle_calls, t.actions.local),
            (100, 4, 100)
        );
        assert_eq!(t.cost_usd, Usd::ZERO);
        let snapshot = std::fs::read(dir.path().join(crate::ledger::SNAPSHOT_FILE)).unwrap();

        let again = import_router_usage(&ledger, &usage(100, 4), 1_790_000_100).unwrap();
        assert_eq!(
            (again.carried, again.unchanged, again.written),
            (0, 2, false)
        );
        assert_eq!(
            std::fs::read(dir.path().join(crate::ledger::SNAPSHOT_FILE)).unwrap(),
            snapshot
        );
        // A newer export adds its growth only; an older one is left alone.
        let newer = import_router_usage(&ledger, &usage(130, 4), 1_790_000_200).unwrap();
        assert_eq!(
            (newer.carried, newer.decisions, newer.unchanged),
            (1, 30, 1)
        );
        let older = import_router_usage(&ledger, &usage(90, 4), 1_790_000_300).unwrap();
        assert_eq!((older.behind, older.carried), (1, 0));
        drop(ledger);
        let reopened = UsageLedger::open(dir.path()).unwrap();
        assert_eq!(reopened.totals("acme").decisions, 130);
        assert_eq!(reopened.totals("beta").decisions, 7);
        for (text, want) in [
            (
                json!([{"account": "a", "decisions": 1}, {"account": "a"}]).to_string(),
                "both account a",
            ),
            (
                json!([{"account": "a", "key_hash": "x"}]).to_string(),
                "column 'key_hash'",
            ),
            (
                json!([{"account": "a", "decisions": -5}]).to_string(),
                "negative",
            ),
            (json!([{"decisions": 5}]).to_string(), "account is missing"),
        ] {
            let e = format!("{:#}", read_router_usage(text.as_bytes()).unwrap_err());
            assert!(e.contains(want), "{text}: {e}");
        }
    }

    // ------------------------------------------------- F1: migration-critical fixes

    /// Accounts, plans and labels as the router's columns hold them (any
    /// Unicode text up to VARCHAR(128)/(64)/(255) characters) import, survive
    /// keys.json and authenticate; log/CLI output escapes control characters.
    #[test]
    fn router_accounts_plans_and_labels_import_as_the_router_holds_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let store = KeyStore::open(&path, "cortiq_").unwrap();
        let long_account = "Ш".repeat(MAX_ACCOUNT_CHARS);
        let export = json!([
            row(
                RAW_A,
                json!({"account": "client+tag@example.com", "plan": "enterprise"})
            ),
            row(
                RAW_B,
                json!({"account": "Иван Петров", "plan": "п".repeat(64),
                              "label": "метка ".repeat(42) + "abc"})
            ),
            row("tabbed", json!({"account": "tab\there\nline"})),
            row("long", json!({"account": long_account})),
        ])
        .to_string();
        let k = mysql(&export);
        assert_eq!(k.records[1].label.chars().count(), 255);
        assert!(
            k.records[1].label.len() > 256,
            "more bytes than the old limit"
        );
        let r = store.import_router_keys(&k, 50).unwrap();
        assert_eq!((r.imported, r.imported_active()), (4, 4));
        // Re-read from disk (canonical JSON parser) and authenticate.
        let again = KeyStore::open(&path, "cortiq_").unwrap();
        assert_eq!(
            again.authenticate(RAW_A, 60).unwrap().account,
            "client+tag@example.com"
        );
        assert_eq!(
            again.authenticate(RAW_B, 60).unwrap().account,
            "Иван Петров"
        );
        assert_eq!(
            again.authenticate("tabbed", 60).unwrap().account,
            "tab\there\nline"
        );
        assert_eq!(
            again.authenticate("long", 60).unwrap().account,
            long_account
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("Иван Петров") && text.contains("tab\\there\\nline"));
        assert_eq!(shown("tab\there\nline\u{7}"), "tab\\there\\nline\\u{7}");
        assert_eq!(shown("Иван client+tag@x"), "Иван client+tag@x");
        // Revocation by such an account works too.
        assert_eq!(again.revoke_account("Иван Петров").unwrap(), 1);
        assert_eq!(again.authenticate(RAW_B, 60), Err(AuthFailure::Revoked));
        // The router's usage counters of such accounts import too.
        let u = read_router_usage(
            json!([{"account": "client+tag@example.com", "decisions": 3}])
                .to_string()
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(u[0].account, "client+tag@example.com");
    }

    fn toml_of(entries: &[(&str, &str, u32)]) -> RouterKeys {
        let mut t = String::from("bind = \"0.0.0.0:8080\"\n");
        for (key, account, rate) in entries {
            t += &format!(
                "[[api_keys]]\nkey = \"{key}\"\naccount = \"{account}\"\nrate_per_min = {rate}\n"
            );
        }
        read_router_keys(t.as_bytes(), ImportFormat::RouterToml, 40).unwrap()
    }

    /// A key both in the router's database and in its configuration follows
    /// the router's precedence (database over static configuration) in either
    /// import order: the same keys.json, the row while it is active and
    /// unexpired, the static key otherwise.
    #[test]
    fn a_key_in_the_database_and_the_configuration_follows_the_router_in_any_order() {
        let db = |active: u8, expires: Option<u64>| {
            mysql(
                &json!([
                    row(
                        RAW_A,
                        json!({"account": "db-acct", "plan": "pro", "active": active,
                                      "rate_per_min": 5, "expires_at": expires,
                                      "created_at": 1_700_000_000u64})
                    ),
                    row(RAW_B, json!({"account": "db-only", "plan": "pro"})),
                ])
                .to_string(),
            )
        };
        let cfg = || toml_of(&[(RAW_A, "cfg-acct", 7), ("static-only", "cfg-only", 0)]);
        let records = |store: &KeyStore| {
            let mut v = store.records();
            v.sort_by(|a, b| a.hash.cmp(&b.hash));
            v
        };
        for (active, expires) in [(1u8, None), (0, None), (1, Some(30u64))] {
            let d1 = tempfile::tempdir().unwrap();
            let s1 = KeyStore::open(d1.path().join("keys.json"), "cortiq_").unwrap();
            let a = s1.import_router_keys(&db(active, expires), 50).unwrap();
            let b = s1.import_router_keys(&cfg(), 50).unwrap();
            assert_eq!((a.imported, b.imported, b.layered), (2, 1, 1));
            let d2 = tempfile::tempdir().unwrap();
            let s2 = KeyStore::open(d2.path().join("keys.json"), "cortiq_").unwrap();
            let c = s2.import_router_keys(&cfg(), 50).unwrap();
            let d = s2.import_router_keys(&db(active, expires), 50).unwrap();
            assert_eq!((c.imported, d.imported, d.layered), (2, 1, 1));
            // The same records whatever the order.
            assert_eq!(
                records(&s1),
                records(&s2),
                "active {active} expires {expires:?}"
            );
            let fallback = active == 0 || expires.is_some();
            assert_eq!(d.static_fallback, usize::from(fallback));
            for s in [&s1, &s2] {
                let k = s.authenticate(RAW_A, 60).unwrap();
                if fallback {
                    // The router drops an inactive or expired row: its
                    // configuration key answers.
                    assert_eq!(
                        (
                            k.account.as_str(),
                            k.plan.as_str(),
                            k.rate_per_min,
                            k.expires
                        ),
                        ("cfg-acct", ROUTER_STATIC_PLAN, 7, None)
                    );
                } else {
                    assert_eq!(
                        (k.account.as_str(), k.plan.as_str(), k.rate_per_min),
                        ("db-acct", "pro", 5)
                    );
                }
                assert_eq!(s.authenticate(RAW_B, 60).unwrap().account, "db-only");
                assert_eq!(
                    s.authenticate("static-only", 60).unwrap().account,
                    "cfg-only"
                );
                // Importing both again changes nothing.
                let x = s.import_router_keys(&db(active, expires), 70).unwrap();
                let y = s.import_router_keys(&cfg(), 70).unwrap();
                assert_eq!(
                    (x.written, y.written, x.unchanged, y.unchanged),
                    (false, false, 2, 2)
                );
            }
            // The router later deactivates the row: the static key answers.
            if !fallback {
                let r = s1.import_router_keys(&db(0, None), 80).unwrap();
                assert_eq!((r.revoked, r.static_fallback), (1, 1));
                assert_eq!(s1.authenticate(RAW_A, 80).unwrap().account, "cfg-acct");
            }
            // Revoked here (by either account): gone for good, and a later
            // import keeps it revoked.
            assert_eq!(s2.revoke_account("cfg-acct").unwrap(), 1);
            assert_eq!(s2.authenticate(RAW_A, 90), Err(AuthFailure::Revoked));
            s2.import_router_keys(&cfg(), 90).unwrap();
            s2.import_router_keys(&db(1, None), 90).unwrap();
            assert_eq!(s2.authenticate(RAW_A, 90), Err(AuthFailure::Revoked));
        }
    }

    /// Imported keys escalate to the oracle as in the router unless the
    /// import opts out; an explicit value also reaches keys imported before,
    /// an import without it leaves them as they are.
    #[test]
    fn imported_keys_may_use_the_oracle_unless_the_import_opts_out() {
        let dir = tempfile::tempdir().unwrap();
        let store = KeyStore::open(dir.path().join("keys.json"), "cortiq_").unwrap();
        let created = store
            .create(
                &NewKey {
                    account: Some("local".into()),
                    ..NewKey::default()
                },
                &crate::config::default_plans(),
                10,
            )
            .unwrap();
        assert!(
            !created.record.oracle_allowed,
            "a key created here is opt-in"
        );
        let export = || mysql(&json!([row(RAW_A, json!({})), row(RAW_B, json!({}))]).to_string());
        let r = store
            .import_router_keys(&export().with_oracle_allowed(false), 20)
            .unwrap();
        assert!(!r.oracle_allowed);
        assert!(!store.authenticate(RAW_A, 20).unwrap().oracle_allowed);
        // No flag: stored keys unchanged (a periodic re-import does not flip them).
        let r = store.import_router_keys(&export(), 30).unwrap();
        assert_eq!((r.oracle_updated, r.written), (0, false));
        assert!(!store.authenticate(RAW_B, 30).unwrap().oracle_allowed);
        // Explicitly allowed: every router key of the export, not the local one.
        let r = store
            .import_router_keys(&export().with_oracle_allowed(true), 40)
            .unwrap();
        assert_eq!((r.oracle_updated, r.written, r.unchanged), (2, true, 2));
        assert!(store.authenticate(RAW_A, 40).unwrap().oracle_allowed);
        assert!(!store.authenticate(&created.raw, 40).unwrap().oracle_allowed);
        // A new key of a plain import may use the oracle (router default).
        let d2 = tempfile::tempdir().unwrap();
        let s2 = KeyStore::open(d2.path().join("keys.json"), "cortiq_").unwrap();
        let r = s2
            .import_router_keys(&toml_of(&[(RAW_A, "a", 0)]), 1)
            .unwrap();
        assert!(r.oracle_allowed && s2.authenticate(RAW_A, 1).unwrap().oracle_allowed);
    }
}
