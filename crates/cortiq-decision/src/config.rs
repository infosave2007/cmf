//! Decision server configuration, `--decision-config` (spec §4.13, §4.7b, §4.15).
//!
//! A JSON object; every field is optional and an unknown key is an error (a typo
//! must not silently fall back to a default). Defaults:
//!
//! ```json
//! {"state_dir":"<FILE>.state",
//!  "auth":{"require":null,"admin_token_env":"CORTIQ_DECISION_ADMIN_TOKEN","key_prefix":"cortiq_"},
//!  "limits":{"body_bytes":1048576,"state_bytes":32768,"questions":32,"max_inflight":64},
//!  "pricing":{"input_usd_per_1m":"0","output_usd_per_1m":"0","request_usd":"0","oracle_passthrough":true,"oracle_markup":1.0},
//!  "response":{"round":null},
//!  "oracle":{"enabled":false,"default_per_request":true,"base_url":"https://openrouter.ai/api/v1","api_key_env":"OPENROUTER_API_KEY",
//!   "model":"deepseek/deepseek-v4.1-flash","provider":{"sort":"price","require_parameters":true,"allow_fallbacks":true,"max_price":{"prompt":0.1,"completion":0.5}},
//!   "max_tokens_per_question":64,"deadline_s":30,"budget_usd":1.0,"max_calls":10000,"max_errors":30,"redact_pii":true,"title":"cortiq-decision","data_collection":null},
//!  "cache":{"enabled":true,"threshold":0.97,"cap":50000},
//!  "learning":{"enabled":true,"refit_min_new":25,"dedup":0.995,"cold_start":true,"synchronous":false},
//!  "feedback":{"pending_cap":50000}}
//! ```
//!
//! plus the router-compatible fields of spec §4.7b/§4.15: `complexity_weights`
//! (base 0.40, ambiguity 0.25, novelty 0.15, margin 0.10, length 0.10),
//! `complexity_tiers` (low ≤ 0.33, medium ≤ 0.66, high ≤ 1.0), `task_complexity`
//! {label → 0..1} (default base 0.4), `routing_tiers` {tier → the client's model}
//! and `default_skill`. `auth.plans` may override the plan table of spec §4.10.
//!
//! **Secrets are never part of the configuration.** The admin token is read from
//! the environment variable named by `auth.admin_token_env` and the oracle key
//! from the one named by `oracle.api_key_env` (spec §4.13); the file only holds
//! the variable names.
//!
//! Prices are decimal strings (spec §4.9) and parsed exactly ([`crate::metering::Usd`]).
//! `auth.require: null` means "required unless the server listens on loopback"
//! ([`AuthConfig::required`]).

use crate::canonical;
use crate::manifest::valid_skill_id;
use crate::metering::{Rates, Usd};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Suffix of the default state directory next to the model file (spec §4.13).
pub const STATE_DIR_SUFFIX: &str = ".state";
/// Default environment variable of the admin token.
pub const DEFAULT_ADMIN_TOKEN_ENV: &str = "CORTIQ_DECISION_ADMIN_TOKEN";
/// Default prefix of minted API keys.
pub const DEFAULT_KEY_PREFIX: &str = "cortiq_";
/// Default OpenRouter base URL.
pub const DEFAULT_ORACLE_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Default environment variable of the oracle key.
pub const DEFAULT_ORACLE_KEY_ENV: &str = "OPENROUTER_API_KEY";
/// Default oracle model.
pub const DEFAULT_ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";
/// The largest `max_tokens` of one oracle call (spec §5.3: `64·q (≤4096)`).
pub const MAX_ORACLE_TOKENS: u32 = 4096;
/// The only rounding `response.round` / `cmf.round` accept (hundredths, spec §4.7).
pub const ROUND_HUNDREDTHS: u8 = 2;

/// The whole configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// `null`: `<FILE>.state` next to the model ([`Config::state_dir_for`]).
    pub state_dir: Option<String>,
    /// The skill of router-API requests without `taxonomy_id` (spec §4.15).
    pub default_skill: Option<String>,
    pub auth: AuthConfig,
    pub limits: Limits,
    pub pricing: Pricing,
    pub response: ResponseConfig,
    pub oracle: OracleConfig,
    pub cache: CacheConfig,
    pub learning: LearningConfig,
    pub feedback: FeedbackConfig,
    pub complexity_weights: ComplexityWeights,
    pub complexity_tiers: Vec<ComplexityTier>,
    /// Base difficulty per label, 0..1 (default 0.4).
    pub task_complexity: BTreeMap<String, f32>,
    /// Complexity tier → the client's own model id (no built-in model names).
    pub routing_tiers: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            state_dir: None,
            default_skill: None,
            auth: AuthConfig::default(),
            limits: Limits::default(),
            pricing: Pricing::default(),
            response: ResponseConfig::default(),
            oracle: OracleConfig::default(),
            cache: CacheConfig::default(),
            learning: LearningConfig::default(),
            feedback: FeedbackConfig::default(),
            complexity_weights: ComplexityWeights::default(),
            complexity_tiers: default_tiers(),
            task_complexity: BTreeMap::new(),
            routing_tiers: BTreeMap::new(),
        }
    }
}

/// Authentication (spec §4.10).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    /// `null`: required unless the server listens on loopback.
    pub require: Option<bool>,
    /// Name of the environment variable holding the admin token.
    pub admin_token_env: String,
    /// Prefix of minted keys (`cortiq_` + 40 hex).
    pub key_prefix: String,
    /// Plan table (spec §4.10); the file's entries replace the defaults by name.
    pub plans: BTreeMap<String, PlanConfig>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            require: None,
            admin_token_env: DEFAULT_ADMIN_TOKEN_ENV.into(),
            key_prefix: DEFAULT_KEY_PREFIX.into(),
            plans: default_plans(),
        }
    }
}

impl AuthConfig {
    /// Whether a key is required: the explicit setting, else everywhere except
    /// on loopback (spec §4.10).
    pub fn required(&self, loopback: bool) -> bool {
        self.require.unwrap_or(!loopback)
    }

    /// The admin token from the environment (`None` when unset or empty: the
    /// admin API answers 404 `ADMIN_DISABLED`).
    pub fn admin_token(&self) -> Option<String> {
        std::env::var(&self.admin_token_env)
            .ok()
            .filter(|t| !t.is_empty())
    }
}

/// A billing plan: requests per minute (0 = unlimited), lifetime decision quota
/// (0 = unlimited) and key lifetime in days (`null` = no expiry).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanConfig {
    pub rate_per_min: u32,
    pub decision_quota: u64,
    #[serde(default)]
    pub days: Option<u32>,
}

/// The plan table of cortiq-router (`config.rs:198-236`, spec §4.10): starter
/// 60/min, no quota; developer 120/min, 100k; pro 600/min, 1M; scale
/// 3000/min, 10M — keys of every plan expire after 30 days, as the router's
/// `duration_days = 30` (a portal that mints keys through `POST
/// /v1/admin/keys` gets the same expiry after the switch).
pub fn default_plans() -> BTreeMap<String, PlanConfig> {
    let plan = |rate_per_min, decision_quota| PlanConfig {
        rate_per_min,
        decision_quota,
        days: Some(DEFAULT_PLAN_DAYS),
    };
    BTreeMap::from([
        ("starter".to_string(), plan(60, 0)),
        ("developer".to_string(), plan(120, 100_000)),
        ("pro".to_string(), plan(600, 1_000_000)),
        ("scale".to_string(), plan(3000, 10_000_000)),
    ])
}

/// Key lifetime of every default plan (router `duration_days = 30`).
pub const DEFAULT_PLAN_DAYS: u32 = 30;

/// Request limits (spec §4.4, §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub body_bytes: usize,
    pub state_bytes: usize,
    pub questions: usize,
    pub max_inflight: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            body_bytes: 1 << 20,
            state_bytes: 32 * 1024,
            questions: 32,
            max_inflight: 64,
        }
    }
}

/// Prices (spec §4.9): decimal strings, all `"0"` by default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pricing {
    pub input_usd_per_1m: String,
    pub output_usd_per_1m: String,
    pub request_usd: String,
    /// Bill the oracle's own cost of successful calls to the client.
    pub oracle_passthrough: bool,
    pub oracle_markup: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            input_usd_per_1m: "0".into(),
            output_usd_per_1m: "0".into(),
            request_usd: "0".into(),
            oracle_passthrough: true,
            oracle_markup: 1.0,
        }
    }
}

impl Pricing {
    /// The parsed rates.
    pub fn rates(&self) -> Result<Rates> {
        Rates::new(
            Usd::parse(&self.input_usd_per_1m).context("pricing.input_usd_per_1m")?,
            Usd::parse(&self.output_usd_per_1m).context("pricing.output_usd_per_1m")?,
            Usd::parse(&self.request_usd).context("pricing.request_usd")?,
            self.oracle_passthrough,
            self.oracle_markup,
        )
        .context("pricing")
    }
}

/// Response options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResponseConfig {
    /// `2`: answers quantised to hundredths like Jev; `null`: shortest f32.
    pub round: Option<u8>,
}

/// The OpenRouter oracle (spec §5). Used by the cascade; the service reads
/// `enabled` and `default_per_request` for the consent check (spec §5.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OracleConfig {
    pub enabled: bool,
    /// The consent of a request without `cmf.oracle`.
    pub default_per_request: bool,
    pub base_url: String,
    /// Name of the environment variable holding the OpenRouter key.
    pub api_key_env: String,
    pub model: String,
    /// OpenRouter provider preferences, sent as the body's `provider` object
    /// (`data_collection` is added only when set). `max_price` {prompt,
    /// completion} (USD per 1M tokens) is required: the reservation uses it.
    pub provider: Map<String, Value>,
    pub max_tokens_per_question: u32,
    pub deadline_s: f64,
    pub budget_usd: f64,
    pub max_calls: u64,
    pub max_errors: u32,
    pub redact_pii: bool,
    pub title: String,
    pub data_collection: Option<String>,
}

impl Default for OracleConfig {
    fn default() -> Self {
        let provider = json!({
            "sort": "price",
            "require_parameters": true,
            "allow_fallbacks": true,
            "max_price": {"prompt": 0.1, "completion": 0.5},
        });
        Self {
            enabled: false,
            default_per_request: true,
            base_url: DEFAULT_ORACLE_BASE_URL.into(),
            api_key_env: DEFAULT_ORACLE_KEY_ENV.into(),
            model: DEFAULT_ORACLE_MODEL.into(),
            provider: match provider {
                Value::Object(m) => m,
                _ => unreachable!("a JSON object literal"),
            },
            max_tokens_per_question: 64,
            deadline_s: 30.0,
            budget_usd: 1.0,
            max_calls: 10_000,
            max_errors: 30,
            redact_pii: true,
            title: "cortiq-decision".into(),
            data_collection: None,
        }
    }
}

impl OracleConfig {
    /// `max_price` {prompt, completion} of the provider preferences (USD per
    /// 1M tokens).
    pub fn max_price(&self) -> Result<(f64, f64)> {
        let mp = self
            .provider
            .get("max_price")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("oracle.provider.max_price must be an object"))?;
        let get = |k: &str| -> Result<f64> {
            let v = mp
                .get(k)
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("oracle.provider.max_price.{k} must be a number"))?;
            ensure!(
                v.is_finite() && v >= 0.0,
                "oracle.provider.max_price.{k} must be finite and non-negative"
            );
            Ok(v)
        };
        Ok((get("prompt")?, get("completion")?))
    }

    /// The `provider` object of a request body: the preferences plus
    /// `data_collection` when it is set.
    pub fn provider_value(&self) -> Value {
        let mut p = self.provider.clone();
        if let Some(dc) = &self.data_collection {
            p.insert("data_collection".into(), Value::String(dc.clone()));
        }
        Value::Object(p)
    }

    /// Whether the environment holds a non-empty oracle key (only its presence
    /// is checked here).
    pub fn key_present(&self) -> bool {
        std::env::var_os(&self.api_key_env).is_some_and(|v| !v.is_empty())
    }
}

/// The semantic cache of oracle answers (spec §5.6).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    pub enabled: bool,
    /// cos φ_P of a hit.
    pub threshold: f32,
    pub cap: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.97,
            cap: 50_000,
        }
    }
}

/// Self-learning (spec §5.7).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearningConfig {
    pub enabled: bool,
    pub refit_min_new: usize,
    /// cos φ_P above which a new example duplicates a stored one.
    pub dedup: f32,
    pub cold_start: bool,
    pub synchronous: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            refit_min_new: 25,
            dedup: 0.995,
            cold_start: true,
            synchronous: false,
        }
    }
}

/// Feedback (spec §5.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeedbackConfig {
    pub pending_cap: usize,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            pending_cap: 50_000,
        }
    }
}

/// Coefficients of the complexity score (router `config.rs:84-110`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComplexityWeights {
    pub base: f32,
    pub ambiguity: f32,
    pub novelty: f32,
    pub margin: f32,
    pub length: f32,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            base: 0.40,
            ambiguity: 0.25,
            novelty: 0.15,
            margin: 0.10,
            length: 0.10,
        }
    }
}

/// One complexity band: the score falls into the first band whose `max` it
/// does not exceed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexityTier {
    pub tier: String,
    pub max: f32,
}

/// low ≤ 0.33, medium ≤ 0.66, high ≤ 1.0 (router `config.rs:118-133`).
pub fn default_tiers() -> Vec<ComplexityTier> {
    [("low", 0.33), ("medium", 0.66), ("high", 1.0)]
        .into_iter()
        .map(|(t, m)| ComplexityTier {
            tier: t.into(),
            max: m,
        })
        .collect()
}

/// `oracle.base_url`: https, or plain http to a loopback host only (a local
/// proxy or mock) — the OpenRouter key (`Authorization`) and the questions
/// travel in every request, as the client keys of `--shadow-of` do. No
/// credentials, query or fragment.
fn check_oracle_base_url(url: &str) -> Result<()> {
    let Some((scheme, rest)) = url.split_once("://") else {
        bail!("oracle.base_url must be an https URL");
    };
    ensure!(
        !rest.contains(['?', '#']),
        "oracle.base_url takes no query or fragment"
    );
    let authority = rest.split('/').next().unwrap_or_default();
    ensure!(
        !authority.is_empty() && !authority.contains('@'),
        "oracle.base_url must name a host, without credentials"
    );
    match scheme {
        "https" => Ok(()),
        "http" => {
            let host = match authority.strip_prefix('[') {
                Some(v6) => v6.split(']').next().unwrap_or_default(),
                None => authority.rsplit_once(':').map_or(authority, |(h, _)| h),
            };
            ensure!(
                crate::shadow::is_loopback_host(host),
                "oracle.base_url: plain http only to a loopback address (the OpenRouter key and the questions travel in every request); use https"
            );
            Ok(())
        }
        _ => bail!("oracle.base_url must be an https URL"),
    }
}

/// A valid environment variable name: 1..128 bytes of `[A-Za-z0-9_]`.
pub fn is_env_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// The value is never echoed: a secret pasted where its variable's name
/// belongs would otherwise land in the error and every captured log.
fn check_env_name(what: &str, name: &str) -> Result<()> {
    ensure!(
        is_env_name(name),
        "{what} must be the NAME of an environment variable ([A-Za-z0-9_], 1..128 bytes), not \
         the secret it holds; the given value ({} bytes) is not shown",
        name.len()
    );
    Ok(())
}

fn check_unit(what: &str, x: f32, lo_exclusive: bool) -> Result<()> {
    let lo_ok = if lo_exclusive { x > 0.0 } else { x >= 0.0 };
    ensure!(
        x.is_finite() && lo_ok && x <= 1.0,
        "{what} must be in {}0, 1], got {x}",
        if lo_exclusive { "(" } else { "[" }
    );
    Ok(())
}

impl Config {
    /// Parse and validate a configuration file's bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let v = canonical::parse(bytes).context("decision config is not valid JSON")?;
        ensure!(v.is_object(), "decision config must be a JSON object");
        let cfg: Self = serde_json::from_value(v).context("decision config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read, parse and validate a configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&bytes).with_context(|| format!("{}", path.display()))
    }

    /// The configuration as JSON (every field, defaults included).
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("the configuration serialises")
    }

    /// The state directory of a model file: `state_dir`, else `<FILE>.state`.
    pub fn state_dir_for(&self, model_path: &Path) -> PathBuf {
        match &self.state_dir {
            Some(d) => PathBuf::from(d),
            None => {
                let mut s = model_path.as_os_str().to_os_string();
                s.push(STATE_DIR_SUFFIX);
                PathBuf::from(s)
            }
        }
    }

    /// The price rates.
    pub fn rates(&self) -> Result<Rates> {
        self.pricing.rates()
    }

    /// Every rule the fields must satisfy.
    pub fn validate(&self) -> Result<()> {
        if let Some(d) = &self.state_dir {
            ensure!(!d.is_empty(), "state_dir must not be empty");
        }
        if let Some(s) = &self.default_skill {
            ensure!(
                valid_skill_id(s),
                "default_skill '{s}' is not a skill id ([a-z0-9][a-z0-9_-]{{0,63}})"
            );
        }
        // auth
        check_env_name("auth.admin_token_env", &self.auth.admin_token_env)?;
        let p = &self.auth.key_prefix;
        ensure!(
            !p.is_empty()
                && p.len() <= 32
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "auth.key_prefix must be 1..32 characters of [A-Za-z0-9_-], got '{p}'"
        );
        for (name, plan) in &self.auth.plans {
            ensure!(
                !name.is_empty() && name.len() <= 64,
                "auth.plans: plan names are 1..64 bytes"
            );
            if let Some(d) = plan.days {
                ensure!(d > 0, "auth.plans.{name}.days must be positive or null");
            }
        }
        // limits
        let l = &self.limits;
        ensure!(
            l.body_bytes >= 1024,
            "limits.body_bytes must be at least 1024"
        );
        ensure!(
            l.state_bytes >= 1 && l.state_bytes <= l.body_bytes,
            "limits.state_bytes must be in 1..=body_bytes"
        );
        ensure!(
            (1..=1024).contains(&l.questions),
            "limits.questions must be in 1..=1024"
        );
        ensure!(l.max_inflight >= 1, "limits.max_inflight must be positive");
        // pricing
        self.pricing.rates()?;
        // response
        if let Some(r) = self.response.round {
            ensure!(
                r == ROUND_HUNDREDTHS,
                "response.round must be 2 or null, got {r}"
            );
        }
        // oracle
        let o = &self.oracle;
        check_oracle_base_url(&o.base_url)?;
        check_env_name("oracle.api_key_env", &o.api_key_env)?;
        ensure!(
            !o.model.is_empty() && o.model.len() <= 256,
            "oracle.model must be 1..256 bytes"
        );
        o.max_price()?;
        if o.provider.contains_key("data_collection") {
            bail!("set data_collection at oracle.data_collection, not inside oracle.provider");
        }
        ensure!(
            (1..=MAX_ORACLE_TOKENS).contains(&o.max_tokens_per_question),
            "oracle.max_tokens_per_question must be in 1..={MAX_ORACLE_TOKENS}"
        );
        ensure!(
            o.deadline_s.is_finite() && o.deadline_s > 0.0,
            "oracle.deadline_s must be positive"
        );
        ensure!(
            o.budget_usd.is_finite() && o.budget_usd >= 0.0,
            "oracle.budget_usd must be finite and non-negative"
        );
        ensure!(o.max_errors >= 1, "oracle.max_errors must be positive");
        ensure!(
            !o.title.is_empty() && o.title.len() <= 256,
            "oracle.title must be 1..256 bytes"
        );
        if let Some(dc) = &o.data_collection {
            ensure!(
                dc == "allow" || dc == "deny",
                "oracle.data_collection must be \"allow\", \"deny\" or null"
            );
        }
        // cache, learning, feedback
        check_unit("cache.threshold", self.cache.threshold, true)?;
        ensure!(self.cache.cap >= 1, "cache.cap must be positive");
        ensure!(
            self.learning.refit_min_new >= 1,
            "learning.refit_min_new must be positive"
        );
        check_unit("learning.dedup", self.learning.dedup, true)?;
        ensure!(
            self.feedback.pending_cap >= 1,
            "feedback.pending_cap must be positive"
        );
        // complexity
        let w = &self.complexity_weights;
        for (k, v) in [
            ("base", w.base),
            ("ambiguity", w.ambiguity),
            ("novelty", w.novelty),
            ("margin", w.margin),
            ("length", w.length),
        ] {
            ensure!(
                v.is_finite() && v >= 0.0,
                "complexity_weights.{k} must be finite and non-negative"
            );
        }
        ensure!(
            !self.complexity_tiers.is_empty(),
            "complexity_tiers must not be empty"
        );
        let mut prev = f32::NEG_INFINITY;
        for t in &self.complexity_tiers {
            ensure!(
                !t.tier.is_empty() && t.tier.len() <= 64,
                "complexity tier names are 1..64 bytes"
            );
            ensure!(
                t.max.is_finite() && t.max > prev,
                "complexity_tiers must have finite, strictly ascending max values"
            );
            prev = t.max;
        }
        for (label, v) in &self.task_complexity {
            check_unit(&format!("task_complexity.{label}"), *v, false)?;
        }
        for (tier, target) in &self.routing_tiers {
            ensure!(
                !target.is_empty() && target.len() <= 256,
                "routing_tiers.{tier} must be 1..256 bytes"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_match_the_spec() {
        let c = Config::default();
        c.validate().unwrap();
        assert_eq!(c.limits.body_bytes, 1_048_576);
        assert_eq!(c.limits.state_bytes, 32_768);
        assert_eq!(c.limits.questions, 32);
        assert_eq!(c.limits.max_inflight, 64);
        assert_eq!(c.oracle.max_price().unwrap(), (0.1, 0.5));
        assert!(!c.oracle.enabled);
        assert_eq!(c.cache.threshold, 0.97);
        assert_eq!(c.learning.refit_min_new, 25);
        // The router's plans: every key expires after 30 days.
        for p in ["starter", "developer", "pro", "scale"] {
            assert_eq!(c.auth.plans[p].days, Some(30), "{p}");
        }
        assert_eq!(c.auth.plans["scale"].decision_quota, 10_000_000);
        assert!(c.auth.required(false));
        assert!(!c.auth.required(true));
        assert!(c.rates().unwrap().is_free());
    }

    #[test]
    fn parse_rejects_unknown_keys_and_bad_values() {
        assert!(Config::from_json(br#"{"limits":{"body_bytes":65536}}"#).is_ok());
        // state_bytes (default 32768) may not exceed body_bytes.
        assert!(Config::from_json(br#"{"limits":{"body_bytes":2048}}"#).is_err());
        assert!(Config::from_json(br#"{"limit":{}}"#).is_err());
        assert!(Config::from_json(br#"{"response":{"round":3}}"#).is_err());
        assert!(Config::from_json(br#"{"pricing":{"input_usd_per_1m":0.1}}"#).is_err());
        assert!(Config::from_json(br#"{"pricing":{"input_usd_per_1m":"-1"}}"#).is_err());
        assert!(Config::from_json(br#"{"cache":{"threshold":1.5}}"#).is_err());
        assert!(Config::from_json(br#"{"oracle":{"provider":{}}}"#).is_err());
        assert!(Config::from_json(br#"{"auth":{"require":true,"key_prefix":"sk-"}}"#).is_ok());
        assert!(Config::from_json(br#"{"auth":{"admin_token_env":"A B"}}"#).is_err());
        assert!(Config::from_json(br#"[]"#).is_err());
    }

    #[test]
    fn a_secret_given_as_a_variable_name_is_never_echoed() {
        let secret = "sk-or-v1-0123456789abcdef0123456789abcdef";
        let mut c = Config::default();
        c.oracle.api_key_env = secret.into();
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(
            !e.contains(secret) && !e.contains("0123456789abcdef"),
            "{e}"
        );
        assert!(e.contains("oracle.api_key_env must be the NAME"), "{e}");
        assert!(
            e.contains(&format!("({} bytes) is not shown", secret.len())),
            "{e}"
        );
        let mut c = Config::default();
        c.auth.admin_token_env = "hunter2 hunter2".into();
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(!e.contains("hunter2"), "{e}");
    }

    #[test]
    fn oracle_base_url_is_https_or_loopback_http() {
        let with = |u: &str| {
            let mut c = Config::default();
            c.oracle.base_url = u.to_string();
            c.validate()
        };
        for ok in [
            "https://openrouter.ai/api/v1",
            "https://proxy.internal:8443/v1",
            "http://127.0.0.1:9199/api/v1",
            "http://localhost:8080",
            "http://[::1]:9000/v1",
        ] {
            assert!(with(ok).is_ok(), "{ok}");
        }
        for bad in [
            "http://openrouter.ai/api/v1",
            "http://10.0.0.5:8080/v1",
            "http://[2001:db8::1]:80/v1",
            "https://user:pw@openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1?x=1",
            "ftp://openrouter.ai",
            "openrouter.ai/api/v1",
        ] {
            assert!(with(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn state_dir_defaults_next_to_the_file() {
        let c = Config::default();
        assert_eq!(
            c.state_dir_for(Path::new("/m/cortiq-decision.cmf")),
            PathBuf::from("/m/cortiq-decision.cmf.state")
        );
    }

    #[test]
    fn provider_value_adds_data_collection_only_when_set() {
        let mut o = OracleConfig::default();
        assert!(o.provider_value().get("data_collection").is_none());
        o.data_collection = Some("deny".into());
        assert_eq!(o.provider_value()["data_collection"], "deny");
    }
}
