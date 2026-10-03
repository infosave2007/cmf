//! The oracle in two steps (`cortiq serve FILE --oracle MODEL`, decision-v4 U1).
//!
//! Step one is the key in the environment (`OPENROUTER_API_KEY`, or the
//! variable `--oracle-key-env` names); step two is `--oracle MODEL`. Everything
//! else has a default, and every safety property of [`crate::oracle`] stays:
//! the oracle is asked only about undetermined questions, every call is
//! reserved against the budget before it is sent, the stop rules hold, the
//! state is PII-redacted by default and the key is read only from the
//! environment at the moment of a call, never stored, logged or printed.
//!
//! [`apply`] turns [`OracleFlags`] into the `oracle` section of a
//! [`Config`] (the flags override the `--decision-config` values; without
//! `--oracle` nothing is changed):
//!
//! * `enabled: true`, `model`, and, when given, `budget_usd`
//!   (`--oracle-budget`, default 1.0), `max_calls` (`--oracle-max-calls`),
//!   `api_key_env` (`--oracle-key-env`), `base_url` (`--oracle-base-url`;
//!   https, plain http only to a loopback address) and `learning.enabled:
//!   false` (`--no-oracle-learning`);
//! * `provider` keeps its defaults {sort: price, require_parameters: true,
//!   allow_fallbacks: true};
//! * `provider.max_price`: `--oracle-max-price IN,OUT` when given, else the
//!   value of a `--decision-config` that sets `oracle.provider`, else twice
//!   the prices of the cheapest endpoint of the model that supports
//!   structured outputs, found by one public `GET {base_url}/models/{model}/endpoints`
//!   (no key is sent; "cheapest" is the smallest reservation of a 1 KiB
//!   call). When the listing cannot be fetched the model is not checked and
//!   the price falls back to {prompt 0.1, completion 0.5} USD per 1M with a
//!   warning.
//!
//! A model the listing does not know (HTTP 400/404, or no endpoint) or whose
//! endpoints all lack structured outputs is refused before anything is
//! opened, with up to [`SUGGESTIONS`] of the cheapest models of the public
//! `GET {base_url}/models` listing that support structured outputs. So is a
//! model listed only with variable pricing (`-1`, e.g. `openrouter/auto`)
//! unless the max price is given (`--oracle-max-price` or the configuration),
//! and a given max price below every structured-output endpoint (OpenRouter
//! would refuse every call).
//!
//! What is typed where a secret must not be never comes back in a message,
//! only its length (and why), and is never put in a listing URL. The
//! key-like test ([`crate::config::looks_like_key`]) is, in this order: the
//! value holds `sk-or-` anywhere or starts with `sk-` (ASCII letters in any
//! case); it starts with `Bearer ` (any case, a space after the word); it
//! has leading or trailing ASCII whitespace; it is 41 bytes or longer
//! without a `/`; it holds 32 or more hexadecimal digits in a row. It
//! applies as it is to an `--oracle` (and `check --model`) model id, which
//! must also be 1..256 bytes with no whitespace or control character and be
//! `author/slug` (a `/`, no empty, `.` or `..` segment); to the value of a
//! numeric flag (`--oracle-budget`, `--oracle-max-calls`,
//! `--oracle-max-price`, `check --max-price`) with its surrounding
//! whitespace trimmed; and to every command-line argument, and the value of
//! a `--flag=VALUE`, that a usage error would quote. A variable's name
//! (`--oracle-key-env`, `check --key-env`;
//! [`crate::config::name_looks_like_key`]) that is all `[A-Z0-9_]` and starts
//! with an upper-case letter or `_` is a name at any length unless it holds
//! 32 hexadecimal digits in a row; another is judged by the key-like test
//! and, besides, taken for a random token when it is 16 bytes or longer with
//! letters and digits and no `_`; what passes must still be 1..128 bytes of
//! `[A-Za-z0-9_]` starting with a letter or `_`. A base URL holding `sk-or-`
//! in any case is refused unshown. A request that fails is a fixed code
//! ([`crate::oracle::transport_code`]), never the HTTP library's text, which
//! may hold the `Authorization` header; a name a listing answers (a
//! provider, a suggested model id) is kept only through
//! [`crate::oracle::clean_upstream`] with the configured key (the value of
//! its variable, [`crate::oracle::key_for_cleaning`]; a listing is fetched
//! without it), and a suggested model id must be one `--oracle` takes too.
//!
//! Out of scope: an upstream at the configured base URL (OpenRouter, or the
//! operator's proxy) that already received the key and deliberately echoes
//! it, or pieces of it, in its own answers holds the key already — our code
//! still copies no upstream free text into another client's answer, and
//! every code it shows comes from a closed set. So is a key the operator
//! types into an argument that is not about the key or the oracle (a file
//! path, `--host`, a skill id).
//!
//! `cortiq decide --oracle MODEL` takes the same flags ([`check_flags`] before
//! any work, [`apply`] only when a question needs the oracle).
//!
//! `cortiq decision oracle check` ([`check`], [`CheckReport`]): (a) the key
//! is in its variable and usable ([`crate::oracle::read_key`]: surrounding
//! whitespace trimmed and said, `bad_key` otherwise, never sent); (b) `GET
//! {base_url}/auth/key` with it (free) — valid, and the key's credit limit
//! and usage when reported (its `label`, a masked form of the key, is never
//! read); (c) the model's public endpoint listing (free, no key) — listed,
//! structured outputs, the cheapest price, or `--max-price IN,OUT` (needed
//! for a model listed only with variable pricing; for another, some
//! structured-output endpoint must fit it); (d) with `--test-call` only, one
//! tiny structured call ([`test_call`]: a two-option choice, `max_tokens`
//! 16, through the oracle client and its reservation) and its cost. Ready
//! only when every check passed — a test call that answered but tripped a
//! stop rule (a cost above its reservation, another model) is a failure,
//! since a server would stop its oracle at that call.
//! [`explain_oracle_error`] words a stop reason or a failed call's error code
//! for both commands; [`explain_stop`] adds the last failure to a
//! `max_errors` stop.

use crate::config::Config;
use crate::oracle::{KeyLookup, max_tokens, reservation_usd};
use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

/// Deadline of one listing request.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// `max_price` = this × the cheapest endpoint's prices.
pub const PRICE_HEADROOM: f64 = 2.0;
/// `max_price` {prompt, completion} (USD per 1M) when the listing cannot be
/// fetched: the configuration's default.
pub const FALLBACK_MAX_PRICE: (f64, f64) = (0.1, 0.5);
/// Models suggested when `--oracle MODEL` cannot be used.
pub const SUGGESTIONS: usize = 3;
/// Largest endpoint listing read.
const MAX_ENDPOINTS_BYTES: u64 = 4 << 20;
/// Largest model listing read.
const MAX_MODELS_BYTES: u64 = 64 << 20;
/// Body size of the nominal call "cheapest" is measured on.
const NOMINAL_BODY_BYTES: usize = 1024;

/// The `--oracle*` flags of `cortiq serve`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OracleFlags {
    /// `--oracle MODEL`: the OpenRouter model id.
    pub model: String,
    /// `--oracle-budget USD`.
    pub budget_usd: Option<f64>,
    /// `--oracle-max-calls N`.
    pub max_calls: Option<u64>,
    /// `--oracle-key-env VAR` (default `OPENROUTER_API_KEY`).
    pub key_env: Option<String>,
    /// `--oracle-base-url URL` (default `https://openrouter.ai/api/v1`).
    pub base_url: Option<String>,
    /// `--oracle-max-price IN,OUT` (USD per 1M tokens).
    pub max_price: Option<(f64, f64)>,
    /// `--no-oracle-learning`: `learning.enabled` false.
    pub no_learning: bool,
}

/// Refuse a value given to a numeric flag that looks like a key
/// ([`crate::config::looks_like_key`], its surrounding whitespace aside),
/// without showing it: only its length.
pub fn refuse_key_in_number(s: &str) -> Result<()> {
    if let Some(why) = crate::config::looks_like_key(s.trim()) {
        bail!(
            "a number is expected here, not the key: put the key in its environment variable \
             (OPENROUTER_API_KEY by default). The given value looks like a key ({why}); it ({} \
             bytes) is not shown",
            s.len()
        );
    }
    Ok(())
}

/// `IN,OUT`: two finite non-negative numbers, USD per 1M tokens. An error
/// never holds the value (a key pasted here would land in the terminal and
/// every captured log); the flag's parser states its length.
pub fn parse_max_price(s: &str) -> Result<(f64, f64)> {
    refuse_key_in_number(s)?;
    let Some((a, b)) = s.split_once(',') else {
        bail!(
            "expected IN,OUT (USD per 1M prompt and completion tokens), e.g. 0.1,0.5; the given \
             value has no comma"
        );
    };
    let num = |x: &str, what: &str| -> Result<f64> {
        let v: f64 = x.trim().parse().map_err(|_| {
            anyhow::anyhow!("{what} price is not a number; expected IN,OUT, e.g. 0.1,0.5")
        })?;
        ensure!(
            v.is_finite() && v >= 0.0,
            "{what} price must be finite and non-negative"
        );
        Ok(v)
    };
    Ok((num(a, "the prompt")?, num(b, "the completion")?))
}

/// A model id usable in a listing URL: 1..256 bytes, no whitespace or
/// control character, `author/slug` (a `/`, no empty, `.` or `..` path
/// segment), and not a key (whatever its prefix,
/// [`crate::config::looks_like_key`]). A value refused is never shown (only
/// its length) and never put in a URL.
fn check_model_id(model: &str) -> Result<()> {
    check_model_id_as(model, "--oracle", "--oracle-key-env")
}

/// [`check_model_id`] with the flags' names as the command spells them.
fn check_model_id_as(model: &str, flag: &str, key_env_flag: &str) -> Result<()> {
    if let Some(why) = crate::config::looks_like_key(model) {
        bail!(
            "{flag} takes an OpenRouter model id such as deepseek/deepseek-v4.1-flash, not the \
             key: put the key in OPENROUTER_API_KEY (or the variable {key_env_flag} names). The \
             given value looks like a key ({why}); it ({} bytes) is not shown",
            model.len()
        );
    }
    // Never echoed either: a secret without a key's usual shape would
    // otherwise land in the error (and, accepted, in a listing URL).
    let why = if model.is_empty() || model.len() > 256 {
        Some("it is not 1..256 bytes")
    } else if model.chars().any(|c| c.is_whitespace() || c.is_control()) {
        Some("it holds whitespace or a control character")
    } else if !model.contains('/') {
        Some("it has no '/' (an OpenRouter model id is author/slug)")
    } else if model.split('/').any(|seg| matches!(seg, "" | "." | "..")) {
        Some("it has an empty, '.' or '..' segment")
    } else {
        None
    };
    if let Some(why) = why {
        bail!(
            "{flag} expects an OpenRouter model id such as deepseek/deepseek-v4.1-flash: {why}; \
             the given value ({} bytes) is not shown",
            model.len()
        );
    }
    Ok(())
}

/// A variable's name given with `flag` (`--oracle-key-env`, `--key-env`):
/// `[A-Za-z0-9_]`, and not a key ([`crate::config::looks_like_key`]); the
/// value is never shown, only its length.
fn check_key_env_name(name: &str, flag: &str) -> Result<()> {
    let why = crate::config::name_looks_like_key(name)
        .map(|w| format!(": the given value looks like a key ({w})"))
        .unwrap_or_default();
    ensure!(
        why.is_empty() && crate::config::is_env_name(name),
        "{flag} takes the NAME of the variable that holds the key (e.g. OPENROUTER_API_KEY), not \
         the key itself{why}; the given value ({} bytes) is not shown",
        name.len()
    );
    Ok(())
}

/// `model` with every byte outside `[A-Za-z0-9-._~/:]` percent-encoded (the
/// `/` of `author/slug` stays a path separator, as OpenRouter's route expects).
fn encode_path(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for b in model.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/:".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `{base_url}/models/{model}/endpoints`.
pub fn endpoints_url(base_url: &str, model: &str) -> String {
    format!(
        "{}/models/{}/endpoints",
        base_url.trim_end_matches('/'),
        encode_path(model)
    )
}

/// `{base_url}/models`.
pub fn models_url(base_url: &str) -> String {
    format!("{}/models", base_url.trim_end_matches('/'))
}

/// The host (and port) of a base URL, for messages (`openrouter.ai`).
pub fn host_of(base_url: &str) -> &str {
    let rest = base_url
        .split_once("://")
        .map_or(base_url, |(_, rest)| rest);
    rest.split('/').next().unwrap_or(rest)
}

/// A dollar amount for messages: at least two decimals, up to six
/// significant ones (`$0.07`, `$0.055`, `$5.00`, `$0.000013`).
pub fn usd(x: f64) -> String {
    if !x.is_finite() {
        return format!("${x}");
    }
    let mut s = format!("{x:.6}");
    while s.ends_with('0') && s.split_once('.').is_some_and(|(_, d)| d.len() > 2) {
        s.pop();
    }
    if s == "0.00" && x > 0.0 {
        s = format!("{x:.2e}");
    }
    format!("${s}")
}

/// A minimum given as advice for a flag ("pass --oracle-budget of at least
/// …"): `x` rounded UP to the micro-dollar, so that passing the printed value
/// admits what needs `x` ([`usd`] rounds to nearest, which may print less).
pub fn usd_ceil(x: f64) -> String {
    if !x.is_finite() || x <= 0.0 {
        return usd(x);
    }
    let mut m = (x * 1e6).ceil();
    // `m / 1e6` is the value the printed decimal parses back to.
    if m / 1e6 < x {
        m += 1.0;
    }
    usd(m / 1e6)
}

/// An amount stated (not advised) with up to eight decimals, so that a
/// reservation such as $0.00178924 is not shown rounded to the budget
/// that could not hold it.
pub fn usd_fine(x: f64) -> String {
    if !x.is_finite() {
        return format!("${x}");
    }
    let mut s = format!("{x:.8}");
    while s.ends_with('0') && s.split_once('.').is_some_and(|(_, d)| d.len() > 2) {
        s.pop();
    }
    if s == "0.00" && x > 0.0 {
        s = format!("{x:.2e}");
    }
    format!("${s}")
}

/// A listing price (USD per token, a decimal string or a number) in USD per
/// 1M tokens, rounded to 1e-9 so that `0.00000007` is `0.07`. `None` when it
/// is missing, not a number, negative (OpenRouter's `-1` of variable
/// pricing) or not finite.
fn per_million(v: Option<&Value>) -> Option<f64> {
    let x = match v? {
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        Value::Number(n) => n.as_f64()?,
        _ => return None,
    };
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    Some((x * 1e6 * 1e9).round() / 1e9)
}

/// Whether `supported_parameters` holds structured outputs
/// (`structured_outputs` or `response_format`).
fn supports_structured(v: Option<&Value>) -> bool {
    v.and_then(Value::as_array).is_some_and(|a| {
        a.iter().any(|p| {
            matches!(
                p.as_str(),
                Some("structured_outputs") | Some("response_format")
            )
        })
    })
}

/// One endpoint (provider) of a model in OpenRouter's listing.
#[derive(Clone, Debug, PartialEq)]
pub struct Endpoint {
    /// `provider_name` (else `name`, else `tag`).
    pub provider: String,
    /// USD per 1M prompt tokens.
    pub prompt: f64,
    /// USD per 1M completion tokens.
    pub completion: f64,
    /// `supported_parameters` holds `structured_outputs` or `response_format`.
    pub structured: bool,
}

/// What `GET {base_url}/models/{model}/endpoints` said.
#[derive(Clone, Debug, PartialEq)]
pub enum EndpointsProbe {
    /// The model's endpoints with a usable price.
    Found(Vec<Endpoint>),
    /// The listing does not know the model (why, for the message).
    NotFound(String),
    /// The model is listed, but no endpoint has a fixed price (OpenRouter's
    /// `-1` of variable pricing, e.g. `openrouter/auto`); `structured`: one of
    /// them supports structured outputs.
    VariablePrice { structured: bool },
    /// No usable answer (transport, status, shape): the model is not checked.
    Unreachable(String),
}

/// A model of `GET {base_url}/models` that supports structured outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct ListedModel {
    pub id: String,
    /// USD per 1M prompt tokens.
    pub prompt: f64,
    /// USD per 1M completion tokens.
    pub completion: f64,
}

/// One GET: `(status, body)` or why not, in fixed words: a transport or
/// read error is its code ([`crate::oracle::transport_code`]), never the
/// library's text, which may hold the request's `Authorization` header (the
/// key) or its URL. `key` is sent as `Authorization: Bearer` only when given
/// (checked by [`crate::oracle::read_key`] first; ureq's debug log redacts
/// it); the listings are fetched without one.
fn get(url: &str, cap: u64, key: Option<&str>) -> std::result::Result<(u16, Vec<u8>), String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(PROBE_TIMEOUT)
        .redirects(0)
        .build();
    let mut req = agent.get(url).set("Accept", "application/json");
    if let Some(k) = key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(_, r)) => r,
        Err(ureq::Error::Transport(t)) => {
            return Err(format!(
                "{}: the request did not complete",
                crate::oracle::transport_code(&t)
            ));
        }
    };
    let status = resp.status();
    let mut buf = Vec::new();
    resp.into_reader()
        .take(cap + 1)
        .read_to_end(&mut buf)
        .map_err(|e| {
            format!(
                "{}: the answer could not be read",
                crate::oracle::read_code(&e)
            )
        })?;
    if buf.len() as u64 > cap {
        return Err(format!("the answer is larger than {cap} bytes"));
    }
    Ok((status, buf))
}

/// An endpoint-listing body (`data.endpoints`): the endpoints with a fixed
/// price, and for each one without it whether it supports structured
/// outputs. `None` when the body does not have that shape. Provider names
/// are kept only through [`crate::oracle::clean_upstream`] with `key`, the
/// configured key ([`crate::oracle::key_for_cleaning`]).
fn parse_listing(body: &[u8], key: Option<&str>) -> Option<(Vec<Endpoint>, Vec<bool>)> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let eps = v.get("data")?.get("endpoints")?.as_array()?;
    let mut priced = Vec::new();
    let mut unpriced = Vec::new();
    for e in eps {
        let structured = supports_structured(e.get("supported_parameters"));
        let pricing = e.get("pricing");
        let price = |k: &str| per_million(pricing.and_then(|p| p.get(k)));
        let (Some(prompt), Some(completion)) = (price("prompt"), price("completion")) else {
            unpriced.push(structured);
            continue;
        };
        let name = ["provider_name", "name", "tag"]
            .iter()
            .find_map(|k| e.get(*k).and_then(Value::as_str))
            .unwrap_or("unnamed");
        priced.push(Endpoint {
            provider: crate::oracle::clean_upstream(name, key),
            prompt,
            completion,
            structured,
        });
    }
    Some((priced, unpriced))
}

/// The endpoints with a fixed price of an endpoint-listing body
/// (`data.endpoints`), `None` when the body does not have that shape;
/// provider names cleaned with the configured `key`.
pub fn parse_endpoints(body: &[u8], key: Option<&str>) -> Option<Vec<Endpoint>> {
    parse_listing(body, key).map(|(priced, _)| priced)
}

/// `GET {base_url}/models/{model}/endpoints`, without a key; `key` (the
/// configured one, [`crate::oracle::key_for_cleaning`]) only cleans the
/// provider names it answers.
pub fn probe_endpoints(base_url: &str, model: &str, key: Option<&str>) -> EndpointsProbe {
    let url = endpoints_url(base_url, model);
    match get(&url, MAX_ENDPOINTS_BYTES, None) {
        Err(why) => EndpointsProbe::Unreachable(why),
        Ok((status @ (400 | 404), _)) => {
            EndpointsProbe::NotFound(format!("{url} answered HTTP {status}"))
        }
        Ok((200, body)) => match parse_listing(&body, key) {
            Some((eps, unpriced)) if eps.is_empty() && unpriced.is_empty() => {
                EndpointsProbe::NotFound(format!("{url} lists no endpoint"))
            }
            Some((eps, unpriced)) if eps.is_empty() => EndpointsProbe::VariablePrice {
                structured: unpriced.contains(&true),
            },
            Some((eps, _)) => EndpointsProbe::Found(eps),
            None => EndpointsProbe::Unreachable(format!("{url}: not an endpoint listing")),
        },
        Ok((status, _)) => EndpointsProbe::Unreachable(format!("{url} answered HTTP {status}")),
    }
}

/// The models of a model-listing body (`data[]`) that support structured
/// outputs, have a positive price and are not a `:free` variant. A model id
/// is suggested only as it is: one `--oracle` would refuse, or that
/// [`crate::oracle::upstream_name_ok`] (with the configured `key`) would not
/// keep, is dropped.
pub fn parse_models(body: &[u8], key: Option<&str>) -> Option<Vec<ListedModel>> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let data = v.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?;
                if id.ends_with(":free")
                    || !crate::oracle::upstream_name_ok(id, key)
                    || check_model_id(id).is_err()
                    || !supports_structured(m.get("supported_parameters"))
                {
                    return None;
                }
                let pricing = m.get("pricing")?;
                let prompt = per_million(pricing.get("prompt"))?;
                let completion = per_million(pricing.get("completion"))?;
                (prompt + completion > 0.0).then(|| ListedModel {
                    id: id.to_string(),
                    prompt,
                    completion,
                })
            })
            .collect(),
    )
}

/// `GET {base_url}/models`, without a key: the models with structured
/// outputs (ids checked with the configured `key`, [`parse_models`]).
pub fn list_models(
    base_url: &str,
    key: Option<&str>,
) -> std::result::Result<Vec<ListedModel>, String> {
    let url = models_url(base_url);
    match get(&url, MAX_MODELS_BYTES, None)? {
        (200, body) => {
            parse_models(&body, key).ok_or_else(|| format!("{url}: not a model listing"))
        }
        (status, _) => Err(format!("{url} answered HTTP {status}")),
    }
}

/// The reservation of a nominal 1 KiB call at these prices: what "cheapest"
/// ranks by (the budget counts reservations).
pub fn nominal_cost(prompt: f64, completion: f64, max_tokens_per_question: u32) -> f64 {
    reservation_usd(
        NOMINAL_BODY_BYTES,
        max_tokens(max_tokens_per_question, 1),
        (prompt, completion),
    )
}

/// The cheapest endpoint with structured outputs.
pub fn cheapest_structured(endpoints: &[Endpoint], mtq: u32) -> Option<&Endpoint> {
    endpoints.iter().filter(|e| e.structured).min_by(|a, b| {
        nominal_cost(a.prompt, a.completion, mtq).total_cmp(&nominal_cost(
            b.prompt,
            b.completion,
            mtq,
        ))
    })
}

/// Up to `n` of the cheapest `models`, other than `except`.
pub fn cheapest_models(
    models: &[ListedModel],
    except: &str,
    mtq: u32,
    n: usize,
) -> Vec<ListedModel> {
    let mut v: Vec<&ListedModel> = models.iter().filter(|m| m.id != except).collect();
    v.sort_by(|a, b| {
        nominal_cost(a.prompt, a.completion, mtq)
            .total_cmp(&nominal_cost(b.prompt, b.completion, mtq))
            .then_with(|| a.id.cmp(&b.id))
    });
    v.into_iter().take(n).cloned().collect()
}

/// Where `provider.max_price` came from.
#[derive(Clone, Debug, PartialEq)]
pub enum PriceSource {
    /// `--oracle-max-price`.
    Flag,
    /// `oracle.provider` of `--decision-config`.
    Config,
    /// [`PRICE_HEADROOM`] × the cheapest structured-output endpoint.
    Listing {
        provider: String,
        prompt: f64,
        completion: f64,
    },
    /// [`FALLBACK_MAX_PRICE`]: the listing could not be fetched.
    Fallback,
}

/// What [`apply`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct OracleSetup {
    pub model: String,
    /// `provider.max_price` {prompt, completion}, USD per 1M tokens.
    pub max_price: (f64, f64),
    pub source: PriceSource,
    /// The endpoint listing's answer.
    pub probe: EndpointsProbe,
    /// Lines to log as warnings (never a key).
    pub warnings: Vec<String>,
}

impl OracleSetup {
    /// Where the max price came from, for the startup line.
    pub fn price_note(&self) -> String {
        match &self.source {
            PriceSource::Flag => "--oracle-max-price".into(),
            PriceSource::Config => "oracle.provider.max_price of --decision-config".into(),
            PriceSource::Listing {
                provider,
                prompt,
                completion,
            } => format!(
                "{PRICE_HEADROOM}× the cheapest structured-output endpoint, {provider} at {}/{}",
                usd(*prompt),
                usd(*completion)
            ),
            PriceSource::Fallback => {
                "the default: the endpoint listing could not be fetched".into()
            }
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "max_price": {"prompt": self.max_price.0, "completion": self.max_price.1},
            "max_price_source": self.price_note(),
            "warnings": self.warnings,
        })
    }
}

/// Where a given max price came from, for messages.
fn price_origin(source: &PriceSource) -> &'static str {
    match source {
        PriceSource::Flag => "--oracle-max-price",
        PriceSource::Config => "oracle.provider.max_price of --decision-config",
        PriceSource::Listing { .. } => "the endpoint listing",
        PriceSource::Fallback => "the default",
    }
}

/// A price for a suggested `--oracle-max-price`: rounded up to 4 decimals.
fn round_up(x: f64) -> String {
    // The epsilon keeps 0.07 (0.0700000…01 in binary) from becoming 0.0701.
    let v = (x * 1e4 - 1e-6).ceil() / 1e4;
    let mut s = format!("{v:.4}");
    while s.ends_with('0') && !s.ends_with(".0") {
        s.pop();
    }
    s
}

fn set_max_price(cfg: &mut Config, (prompt, completion): (f64, f64)) {
    let mut mp = Map::new();
    mp.insert("prompt".into(), json!(prompt));
    mp.insert("completion".into(), json!(completion));
    cfg.oracle
        .provider
        .insert("max_price".into(), Value::Object(mp));
}

/// Up to [`SUGGESTIONS`] of the cheapest structured-output models of the
/// public model listing of `base_url` other than `model` (one GET, no key;
/// `key`, the configured one, checks the ids).
pub fn suggest_models(
    base_url: &str,
    model: &str,
    mtq: u32,
    key: Option<&str>,
) -> std::result::Result<Vec<ListedModel>, String> {
    list_models(base_url, key).map(|models| cheapest_models(&models, model, mtq, SUGGESTIONS))
}

/// The sentence that follows a refused model: the suggestions, or why there
/// are none.
fn suggestions_tail(suggested: &std::result::Result<Vec<ListedModel>, String>) -> String {
    match suggested {
        Ok(best) if best.is_empty() => {
            " No model of the listing supports structured outputs.".to_string()
        }
        Ok(best) => {
            let names: Vec<String> = best
                .iter()
                .map(|m| {
                    format!(
                        "{} ({}/{} per 1M in/out)",
                        m.id,
                        usd(m.prompt),
                        usd(m.completion)
                    )
                })
                .collect();
            format!(
                " Cheap models with structured outputs: {}. Start with --oracle {}",
                names.join(", "),
                best[0].id
            )
        }
        Err(why) => format!(" (the model listing could not be fetched for suggestions: {why})"),
    }
}

/// The refusal of a model the oracle cannot use, with suggestions.
fn refuse(cfg: &Config, model: &str, problem: &str, key: Option<&str>) -> anyhow::Error {
    let suggested = suggest_models(
        &cfg.oracle.base_url,
        model,
        cfg.oracle.max_tokens_per_question,
        key,
    );
    anyhow::anyhow!(
        "--oracle {model}: {problem}.{}",
        suggestions_tail(&suggested)
    )
}

/// Check `flags` without touching the network: the model id, the budget, the
/// key variable's name (never echoed), the base URL (https, or plain http to
/// a loopback address) and the max price. `cortiq decide --oracle` runs it
/// before any work, the listing is fetched only when a question needs the
/// oracle.
pub fn check_flags(flags: &OracleFlags) -> Result<()> {
    prepare(&mut Config::default(), flags, false).map(|_| ())
}

/// Everything [`apply`] does before the network: the flags into `cfg`, then
/// the validation. `Some` names where a fixed max price came from.
pub fn prepare(
    cfg: &mut Config,
    flags: &OracleFlags,
    config_sets_max_price: bool,
) -> Result<Option<PriceSource>> {
    check_model_id(&flags.model)?;
    let o = &mut cfg.oracle;
    o.enabled = true;
    o.model = flags.model.clone();
    if let Some(b) = flags.budget_usd {
        ensure!(
            b.is_finite() && b >= 0.0,
            "--oracle-budget must be a finite non-negative number of USD"
        );
        o.budget_usd = b;
    }
    if let Some(n) = flags.max_calls {
        o.max_calls = n;
    }
    if let Some(k) = &flags.key_env {
        // Never echoed: a key pasted here by mistake must not reach a log.
        check_key_env_name(k, "--oracle-key-env")?;
        o.api_key_env = k.clone();
    }
    if let Some(u) = &flags.base_url {
        // Named by its flag, not by the configuration field it sets.
        crate::config::check_oracle_base_url("--oracle-base-url", u)?;
        o.base_url = u.clone();
    }
    if flags.no_learning {
        cfg.learning.enabled = false;
    }
    // `None`: the max price comes from the listing.
    let fixed = match flags.max_price {
        Some(mp) => {
            set_max_price(cfg, mp);
            Some(PriceSource::Flag)
        }
        None => config_sets_max_price.then_some(PriceSource::Config),
    };
    // The URL (https, or http to loopback) and the variable name are checked
    // before the listing is fetched.
    cfg.validate()?;
    Ok(fixed)
}

/// Apply `flags` to `cfg` (see the module notes) and check the model against
/// the endpoint listing of `cfg.oracle.base_url` (one GET, no key; one more
/// for suggestions when the model is refused). `config_sets_max_price`: the
/// `--decision-config` file sets `oracle.provider` itself. `key` reads the
/// configured key's variable only to clean what the listings answer
/// ([`crate::oracle::key_for_cleaning`]); nothing is sent with it.
pub fn apply(
    cfg: &mut Config,
    flags: &OracleFlags,
    config_sets_max_price: bool,
    key: &KeyLookup,
) -> Result<OracleSetup> {
    let fixed = prepare(cfg, flags, config_sets_max_price)?;
    let model = flags.model.clone();
    let mtq = cfg.oracle.max_tokens_per_question;
    let clean_key = crate::oracle::key_for_cleaning(key, &cfg.oracle.api_key_env);
    let clean_key = clean_key.as_deref();
    let probe = probe_endpoints(&cfg.oracle.base_url, &model, clean_key);
    let mut warnings = Vec::new();
    let source = match (&probe, fixed) {
        (EndpointsProbe::NotFound(why), _) => {
            return Err(refuse(
                cfg,
                &model,
                &format!(
                    "{} does not list this model ({why})",
                    host_of(&cfg.oracle.base_url)
                ),
                clean_key,
            ));
        }
        (EndpointsProbe::Found(eps), fixed) => {
            let Some(best) = cheapest_structured(eps, mtq) else {
                return Err(refuse(
                    cfg,
                    &model,
                    &format!(
                        "none of its {} endpoints supports structured outputs (response_format with a JSON schema), which the oracle needs for typed verdicts",
                        eps.len()
                    ),
                    clean_key,
                ));
            };
            match fixed {
                None => {
                    set_max_price(
                        cfg,
                        (
                            best.prompt * PRICE_HEADROOM,
                            best.completion * PRICE_HEADROOM,
                        ),
                    );
                    PriceSource::Listing {
                        provider: best.provider.clone(),
                        prompt: best.prompt,
                        completion: best.completion,
                    }
                }
                Some(fixed) => {
                    let (p, c) = cfg.oracle.max_price()?;
                    let fits = eps
                        .iter()
                        .any(|e| e.structured && e.prompt <= p && e.completion <= c);
                    ensure!(
                        fits,
                        "--oracle {model}: no structured-output endpoint is within the max price \
                         in/out {}/{} per 1M of {} (the cheapest, {}, costs {}/{}), so OpenRouter \
                         would refuse every call. Raise it (e.g. --oracle-max-price {},{}) or leave \
                         it out to use {PRICE_HEADROOM}× the cheapest endpoint",
                        usd(p),
                        usd(c),
                        price_origin(&fixed),
                        best.provider,
                        usd(best.prompt),
                        usd(best.completion),
                        round_up(best.prompt * PRICE_HEADROOM),
                        round_up(best.completion * PRICE_HEADROOM),
                    );
                    fixed
                }
            }
        }
        (EndpointsProbe::VariablePrice { structured: false }, _) => {
            return Err(refuse(
                cfg,
                &model,
                "none of its endpoints supports structured outputs (response_format with a JSON schema), which the oracle needs for typed verdicts",
                clean_key,
            ));
        }
        (EndpointsProbe::VariablePrice { structured: true }, None) => {
            return Err(refuse(
                cfg,
                &model,
                &format!(
                    "{} lists it with no endpoint of a fixed price (variable pricing, so the max price cannot be taken from the listing; give it with --oracle-max-price IN,OUT, or use a concrete model)",
                    host_of(&cfg.oracle.base_url)
                ),
                clean_key,
            ));
        }
        (EndpointsProbe::VariablePrice { structured: true }, Some(fixed)) => {
            let (p, c) = cfg.oracle.max_price()?;
            warnings.push(format!(
                "oracle: {model} has only variable-price endpoints: the max price in/out {}/{} per 1M of {} is not checked against the listing; it caps every call's reservation, and a call that costs more than its reservation stops the oracle",
                usd(p),
                usd(c),
                price_origin(&fixed)
            ));
            fixed
        }
        (EndpointsProbe::Unreachable(why), None) => {
            set_max_price(cfg, FALLBACK_MAX_PRICE);
            warnings.push(format!(
                "oracle: the endpoint listing of {model} could not be fetched ({why}): the model is not checked and the max price in/out falls back to {}/{} per 1M (--oracle-max-price IN,OUT sets it)",
                usd(FALLBACK_MAX_PRICE.0),
                usd(FALLBACK_MAX_PRICE.1)
            ));
            PriceSource::Fallback
        }
        (EndpointsProbe::Unreachable(why), Some(fixed)) => {
            warnings.push(format!(
                "oracle: the endpoint listing of {model} could not be fetched ({why}): the model is not checked"
            ));
            fixed
        }
    };
    cfg.validate()?;
    let max_price = cfg.oracle.max_price()?;
    Ok(OracleSetup {
        model,
        max_price,
        source,
        probe,
        warnings,
    })
}

// ------------------------------------------------------------------ oracle check

/// Largest `GET /auth/key` answer read.
const MAX_KEY_BYTES: u64 = 64 << 10;
/// `max_tokens` of the test call of `cortiq decision oracle check --test-call`.
pub const TEST_CALL_MAX_TOKENS: u32 = 16;
/// The most the test call may reserve, USD (a few thousand times its cost).
pub const TEST_CALL_BUDGET_USD: f64 = 0.01;
/// The state of the test call: a two-option choice.
pub const TEST_CALL_STATE: &str = "Hello there, how are you today?";
/// Where OpenRouter keys are created (for messages).
pub const KEYS_PAGE: &str = "https://openrouter.ai/keys";
/// Where OpenRouter credits are added (for messages).
pub const CREDITS_PAGE: &str = "https://openrouter.ai/settings/credits";

/// A stop reason of [`crate::oracle`] (`http_401`, `unexpected_model`,
/// `cost_above_reservation`, `max_errors`, …) or the error code of a failed
/// call, in words (without the code itself); `key_env` names the key's
/// variable (never the key), `model` the oracle model. A code outside the
/// closed set ([`crate::oracle::shown_code`]) is worded as `unknown_code`.
fn error_words(code: &str, key_env: &str, model: &str) -> String {
    let code = crate::oracle::shown_code(code);
    let status = code.strip_prefix("http_").unwrap_or("");
    match code {
        "http_401" | "http_403" => {
            format!("OpenRouter refused the key in {key_env} (HTTP {status})")
        }
        "http_402" => format!(
            "OpenRouter answered HTTP 402: the key in {key_env} has no credits left (add credits at \
             {CREDITS_PAGE}, or raise the key's own limit)"
        ),
        "unexpected_model" => format!("OpenRouter answered with a model other than {model}"),
        "cost_above_reservation" => "the provider billed a call more than was reserved for it \
             (the reservation assumes the max price in/out; --oracle-max-price IN,OUT sets it)"
            .to_string(),
        "max_errors" => "too many oracle calls failed in a row (oracle.max_errors)".to_string(),
        "http_429" => "OpenRouter rate-limited the call (HTTP 429)".to_string(),
        crate::oracle::UNKNOWN_CODE => "a code this version does not know was recorded (it is \
             not shown)"
            .to_string(),
        c if c.starts_with("transport_") || c.starts_with("read_") => "the request did not \
             complete (the network, a firewall, or the deadline oracle.deadline_s)"
            .to_string(),
        _ if !status.is_empty() => format!("OpenRouter answered HTTP {status}"),
        _ => "the answer was not usable".to_string(),
    }
}

/// A stop reason of [`crate::oracle`] (`http_401`, `unexpected_model`,
/// `cost_above_reservation`, `max_errors`, …) or the error code of a failed
/// call, in words; `key_env` names the key's variable (never the key),
/// `model` the oracle model. A code the words do not name (a transport
/// error, an HTTP status, a bad answer) is given in parentheses, and only
/// from the closed code set ([`crate::oracle::shown_code`]; another is
/// `unknown_code`).
pub fn explain_oracle_error(code: &str, key_env: &str, model: &str) -> String {
    let code = crate::oracle::shown_code(code);
    let status = code.strip_prefix("http_").unwrap_or("");
    let words = error_words(code, key_env, model);
    match code {
        "http_401"
        | "http_402"
        | "http_403"
        | "http_429"
        | "unexpected_model"
        | "cost_above_reservation"
        | "max_errors"
        | crate::oracle::UNKNOWN_CODE => words,
        c if c.starts_with("transport_") || c.starts_with("read_") => format!(
            "the request did not complete ({c}: the network, a firewall, or the deadline \
             oracle.deadline_s)"
        ),
        c if !status.is_empty() => format!("{words} ({c})"),
        c => format!("the answer was not usable ({c})"),
    }
}

/// "the last error: CODE — WORDS": the failed call behind a `max_errors`
/// stop (or failures in a row), its code named (from the closed set).
pub fn explain_last_error(code: &str, key_env: &str, model: &str) -> String {
    let code = crate::oracle::shown_code(code);
    format!(
        "the last error: {code} — {}",
        error_words(code, key_env, model)
    )
}

/// A stop reason in words, as [`explain_oracle_error`], with the last failed
/// call's error code for `max_errors` (`last_error` of `oracle.state`): "too
/// many oracle calls failed in a row (oracle.max_errors); the last error:
/// http_500 — OpenRouter answered HTTP 500".
pub fn explain_stop(code: &str, last_error: Option<&str>, key_env: &str, model: &str) -> String {
    let words = explain_oracle_error(code, key_env, model);
    match last_error {
        Some(last) if code == "max_errors" => {
            format!("{words}; {}", explain_last_error(last, key_env, model))
        }
        _ => words,
    }
}

/// `{base_url}/auth/key`.
pub fn auth_key_url(base_url: &str) -> String {
    format!("{}/auth/key", base_url.trim_end_matches('/'))
}

/// What `GET {base_url}/auth/key` reports about a key (amounts in USD).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyInfo {
    /// `limit`: the key's credit limit (`None`: no limit, or not reported).
    pub limit: Option<f64>,
    /// `usage`: spent through the key.
    pub usage: Option<f64>,
    /// `limit_remaining` (`None` without a limit).
    pub limit_remaining: Option<f64>,
    /// `is_free_tier`.
    pub free_tier: Option<bool>,
}

impl KeyInfo {
    /// The key's limit is used up: OpenRouter refuses its paid calls.
    pub fn out_of_credit(&self) -> bool {
        self.limit_remaining.is_some_and(|r| r <= 0.0)
            || matches!((self.limit, self.usage), (Some(l), Some(u)) if u >= l)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "limit_usd": self.limit,
            "usage_usd": self.usage,
            "limit_remaining_usd": self.limit_remaining,
            "free_tier": self.free_tier,
        })
    }
}

/// A key-description body (`data`: `limit`, `usage`, `limit_remaining`,
/// `is_free_tier`); `None` when the body does not have that shape. Its
/// `label`, a masked form of the key, is never read.
pub fn parse_key_info(body: &[u8]) -> Option<KeyInfo> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let d = v.get("data")?.as_object()?;
    let num = |k: &str| d.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
    Some(KeyInfo {
        limit: num("limit"),
        usage: num("usage"),
        limit_remaining: num("limit_remaining"),
        free_tier: d.get("is_free_tier").and_then(Value::as_bool),
    })
}

/// What `GET {base_url}/auth/key` (free) said about a key.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyProbe {
    /// HTTP 200: a valid key.
    Valid(KeyInfo),
    /// HTTP 401 or 403: not a valid key.
    Refused(u16),
    /// No usable answer (transport, another status, shape).
    Unreachable(String),
}

/// `GET {base_url}/auth/key` with `key` (never part of a message: an answer
/// is reduced to its status and the amounts above).
pub fn probe_key(base_url: &str, key: &str) -> KeyProbe {
    let url = auth_key_url(base_url);
    match get(&url, MAX_KEY_BYTES, Some(key)) {
        Err(why) => KeyProbe::Unreachable(why),
        Ok((200, body)) => parse_key_info(&body).map_or_else(
            || KeyProbe::Unreachable(format!("{url}: not a key description")),
            KeyProbe::Valid,
        ),
        Ok((status @ (401 | 403), _)) => KeyProbe::Refused(status),
        Ok((status, _)) => KeyProbe::Unreachable(format!("{url} answered HTTP {status}")),
    }
}

/// Whether a model can be the oracle, from its public endpoint listing.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCheck {
    pub probe: EndpointsProbe,
    /// The cheapest endpoint with structured outputs.
    pub cheapest: Option<Endpoint>,
    /// What is wrong: a code (`unknown_model`, `no_structured_outputs`,
    /// `variable_price`, `max_price_too_low`, `listing_unreachable`) and a
    /// message.
    pub problem: Option<(&'static str, String)>,
    /// Cheap models with structured outputs (only when the model cannot be
    /// used), or why the model listing could not be fetched.
    pub suggestions: Option<std::result::Result<Vec<ListedModel>, String>>,
}

/// Check `model` against the endpoint listing of `base_url` (one GET, no
/// key; one more for suggestions when it cannot be used). `max_price`: the
/// max price given (`--max-price IN,OUT`), which a model listed only with
/// variable pricing needs and which some structured-output endpoint of a
/// priced model must fit.
pub fn check_model(
    base_url: &str,
    model: &str,
    mtq: u32,
    max_price: Option<(f64, f64)>,
    key: Option<&str>,
) -> ModelCheck {
    let probe = probe_endpoints(base_url, model, key);
    let no_json = |n: usize| {
        (
            "no_structured_outputs",
            format!(
                "none of its {n} endpoints supports structured outputs (response_format with a \
                 JSON schema), which the oracle needs for typed verdicts"
            ),
        )
    };
    let (cheapest, problem) = match &probe {
        EndpointsProbe::Found(eps) => match (cheapest_structured(eps, mtq), max_price) {
            (None, _) => (None, Some(no_json(eps.len()))),
            (Some(best), Some((p, c)))
                if !eps
                    .iter()
                    .any(|e| e.structured && e.prompt <= p && e.completion <= c) =>
            {
                (
                    Some(best.clone()),
                    Some((
                        "max_price_too_low",
                        format!(
                            "no structured-output endpoint is within the max price in/out {}/{} \
                             per 1M of --max-price (the cheapest, {}, costs {}/{}), so OpenRouter \
                             would refuse every call: raise it (e.g. --max-price {},{}) or leave it \
                             out to use {PRICE_HEADROOM}× the cheapest endpoint",
                            usd(p),
                            usd(c),
                            best.provider,
                            usd(best.prompt),
                            usd(best.completion),
                            round_up(best.prompt * PRICE_HEADROOM),
                            round_up(best.completion * PRICE_HEADROOM),
                        ),
                    )),
                )
            }
            (Some(best), _) => (Some(best.clone()), None),
        },
        EndpointsProbe::NotFound(why) => (
            None,
            Some((
                "unknown_model",
                format!("{} does not list this model ({why})", host_of(base_url)),
            )),
        ),
        EndpointsProbe::VariablePrice { structured: false } => (
            None,
            Some((
                "no_structured_outputs",
                "none of its endpoints supports structured outputs (response_format with a JSON \
                 schema), which the oracle needs for typed verdicts"
                    .to_string(),
            )),
        ),
        EndpointsProbe::VariablePrice { structured: true } if max_price.is_some() => (None, None),
        EndpointsProbe::VariablePrice { structured: true } => (
            None,
            Some((
                "variable_price",
                format!(
                    "{} lists it only with variable pricing, so the max price cannot be taken \
                     from the listing: give it here with --max-price IN,OUT (USD per 1M tokens, \
                     e.g. --max-price 1,4) and to --oracle with --oracle-max-price IN,OUT, or use a \
                     concrete model",
                    host_of(base_url)
                ),
            )),
        ),
        EndpointsProbe::Unreachable(why) => (
            None,
            Some((
                "listing_unreachable",
                format!("the endpoint listing could not be fetched ({why})"),
            )),
        ),
    };
    let suggestions = match &problem {
        Some((code, _)) if !matches!(*code, "listing_unreachable" | "max_price_too_low") => {
            Some(suggest_models(base_url, model, mtq, key))
        }
        _ => None,
    };
    ModelCheck {
        probe,
        cheapest,
        problem,
        suggestions,
    }
}

/// The test call of `oracle check --test-call`.
#[derive(Clone, Debug, PartialEq)]
pub struct TestCall {
    /// The oracle answered with one of the two options.
    pub answered: bool,
    pub choice: Option<String>,
    /// `usage.cost` (also of a billed failure).
    pub cost_usd: Option<f64>,
    pub reserved_usd: f64,
    pub provider: Option<String>,
    pub latency_ms: Option<f64>,
    /// A short error code (`http_402`, `transport_io`, `invalid_json`, a
    /// refusal such as `budget`), never content.
    pub error: Option<String>,
    /// The stop rule the call tripped (`cost_above_reservation`,
    /// `unexpected_model`, `http_401`, …): a server would stop its oracle at
    /// such a call, so the check fails even when the call answered.
    pub stop: Option<String>,
}

impl TestCall {
    /// Answered and tripped no stop rule.
    pub fn ok(&self) -> bool {
        self.answered && self.stop.is_none()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "ok": self.ok(),
            "answered": self.answered,
            "choice": self.choice,
            "cost_usd": self.cost_usd,
            "reserved_usd": self.reserved_usd,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "stop_reason": self.stop,
        })
    }

    /// What went wrong, in words (`None`: ok).
    fn problem(&self, key_env: &str, model: &str) -> Option<String> {
        if self.ok() {
            return None;
        }
        let why = |code: &str| explain_oracle_error(code, key_env, model);
        let billed = match (self.answered, self.cost_usd) {
            (false, Some(c)) => format!(", billed {}", usd(c)),
            _ => String::new(),
        };
        Some(match (&self.stop, &self.error) {
            (None, Some(code)) if code == "budget" && !self.answered => format!(
                "not sent: its reservation, {} at the max price, is above the {} a test call may \
                 reserve (a lower --max-price IN,OUT fits it)",
                usd(self.reserved_usd),
                usd(TEST_CALL_BUDGET_USD)
            ),
            (Some(stop), _) if stop == "cost_above_reservation" && self.answered => format!(
                "answered '{}', but OpenRouter billed {} for it, more than the {} reserved at \
                 the max price: a server stops its oracle at such a call (stop rule \
                 cost_above_reservation). Give a higher --oracle-max-price IN,OUT, or pick \
                 another model",
                self.choice.as_deref().unwrap_or("?"),
                self.cost_usd.map_or("?".to_string(), usd),
                usd(self.reserved_usd)
            ),
            (Some(stop), _) => format!(
                "failed with the stop rule {stop}{billed}: {} (a server stops its oracle at \
                 such a call)",
                why(stop)
            ),
            // The code is named once, in the parentheses.
            (None, Some(code)) => format!(
                "failed ({code}{billed}): {}",
                error_words(code, key_env, model)
            ),
            (None, None) => "no answer".to_string(),
        })
    }
}

/// A temporary reservation ledger for the test call (removed after it).
fn temp_ledger() -> Result<std::path::PathBuf> {
    use rand_core::{OsRng, RngCore};
    let mut b = [0u8; 8];
    OsRng
        .try_fill_bytes(&mut b)
        .map_err(|e| anyhow::anyhow!("OS random number generator: {e}"))?;
    let tag: String = b.iter().map(|x| format!("{x:02x}")).collect();
    Ok(std::env::temp_dir().join(format!(
        "cortiq-oracle-check-{}-{tag}.jsonl",
        std::process::id()
    )))
}

/// One tiny structured call through [`crate::oracle::OracleClient`] (the
/// reservation, the response checks): a two-option choice about
/// [`TEST_CALL_STATE`] with `max_tokens` [`TEST_CALL_MAX_TOKENS`], admitted
/// only when its reservation fits [`TEST_CALL_BUDGET_USD`]. `oracle` gives the
/// model, the base URL, the key variable and the max price. The ledger is a
/// temporary file, removed after the call.
pub fn test_call(oracle: &crate::config::OracleConfig, key: KeyLookup) -> Result<TestCall> {
    use crate::oracle::{CallOutcome, Caller, OracleClient};
    use crate::protocol::{Question, QuestionKind};
    let mut cfg = oracle.clone();
    cfg.enabled = true;
    cfg.max_tokens_per_question = TEST_CALL_MAX_TOKENS;
    // The bare verdict: the tiniest call that proves the structured output.
    cfg.probabilities = false;
    cfg.budget_usd = TEST_CALL_BUDGET_USD;
    cfg.max_calls = 1;
    cfg.max_errors = 1;
    let q = Question {
        id: "greeting".into(),
        kind: QuestionKind::Choice,
        instructions: json!("Is the text a greeting?"),
        criteria: Some(json!({
            "yes": "The text greets someone.",
            "no": "The text does not greet anyone.",
        })),
    };
    let ledger = temp_ledger()?;
    let outcome = (|| -> Result<(CallOutcome, Option<String>)> {
        // The report says what a fired stop rule means: no WARN line.
        let client = OracleClient::open(&cfg, &ledger, None, key)?.quiet_stops();
        let caller = Caller {
            request_id: "oracle-check",
            account: "oracle-check",
            key12: None,
            key_budget_usd: None,
            credit_left_usd: None,
        };
        let outcome = client.call(&caller, &[&q], &json!(TEST_CALL_STATE));
        // `max_errors` is 1 here, so any failure stops the client: only the
        // rules a server would apply to this call count.
        let stop = client
            .state()
            .stop_reason
            .filter(|r| r != "max_errors")
            .map(|r| crate::oracle::shown_code(&r).to_string());
        Ok((outcome, stop))
    })();
    let _ = std::fs::remove_file(&ledger);
    let body = crate::oracle::request_body(&cfg, &[&q], &json!(TEST_CALL_STATE));
    let reserved_usd = reservation_usd(
        body.len(),
        crate::oracle::call_max_tokens(&cfg, 1),
        cfg.max_price()?,
    );
    let (outcome, stop) = outcome?;
    Ok(match outcome {
        CallOutcome::Answered(a) => TestCall {
            answered: true,
            choice: a
                .verdicts
                .first()
                .and_then(|v| v.label())
                .map(str::to_string),
            cost_usd: Some(a.usage.cost),
            reserved_usd: a.reserved_usd,
            provider: a.provider,
            latency_ms: Some(a.latency.as_secs_f64() * 1e3),
            error: None,
            stop,
        },
        CallOutcome::Failed(f) => TestCall {
            answered: false,
            choice: None,
            cost_usd: f.billed.map(|u| u.cost),
            reserved_usd,
            provider: None,
            latency_ms: None,
            error: Some(crate::oracle::shown_code(&f.error).to_string()),
            stop,
        },
        CallOutcome::Refused(r) => TestCall {
            answered: false,
            choice: None,
            cost_usd: None,
            reserved_usd,
            provider: None,
            latency_ms: None,
            error: Some(r.flag().to_string()),
            stop,
        },
    })
}

/// What `cortiq decision oracle check` checks.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckOptions {
    pub model: String,
    /// The NAME of the variable holding the key.
    pub key_env: String,
    pub base_url: String,
    /// Make one tiny structured call ([`test_call`]).
    pub test_call: bool,
    /// `--max-price IN,OUT` (USD per 1M tokens): the max price `--oracle`
    /// would be given (`--oracle-max-price`); a model listed only with
    /// variable pricing needs it. `None`: twice the cheapest endpoint's.
    pub max_price: Option<(f64, f64)>,
}

/// The answer of `cortiq decision oracle check` (never a key).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckReport {
    pub options: CheckOptions,
    /// (a) what the variable holds ([`crate::oracle::read_key`]).
    pub key: crate::oracle::KeyState,
    /// (b) `GET /auth/key` (`None`: not checked, no usable key).
    pub account: Option<KeyProbe>,
    /// (c) `GET /models/{model}/endpoints`.
    pub model: ModelCheck,
    /// The max price `--oracle` would set: `--max-price`, else twice the
    /// cheapest endpoint's.
    pub max_price: Option<(f64, f64)>,
    /// (d) with `--test-call`: the call, or why it was not made.
    pub test_call: Option<std::result::Result<TestCall, String>>,
}

/// One failed check: (check, code, message).
pub type Problem = (&'static str, &'static str, String);

impl CheckReport {
    /// What is not ready, in check order (empty: ready).
    pub fn problems(&self) -> Vec<Problem> {
        let o = &self.options;
        let mut v: Vec<Problem> = Vec::new();
        match &self.key {
            crate::oracle::KeyState::Missing => v.push((
                "key",
                "no_key",
                format!(
                    "{} is not set: export {}=<your OpenRouter key> (create one at {KEYS_PAGE})",
                    o.key_env, o.key_env
                ),
            )),
            crate::oracle::KeyState::Bad(p) => v.push((
                "key",
                "bad_key",
                format!(
                    "{}; fix the variable (nothing was sent with it)",
                    crate::oracle::bad_key_text(&o.key_env, p)
                ),
            )),
            crate::oracle::KeyState::Usable { .. } => {}
        }
        match &self.account {
            Some(KeyProbe::Refused(status)) => v.push((
                "account",
                "key_refused",
                format!(
                    "OpenRouter refused the key (HTTP {status}): {} does not hold a valid key \
                     (create one at {KEYS_PAGE})",
                    o.key_env
                ),
            )),
            Some(KeyProbe::Unreachable(why)) => v.push((
                "account",
                "account_unreachable",
                format!("the key could not be checked: {why}"),
            )),
            Some(KeyProbe::Valid(info)) if info.out_of_credit() => v.push((
                "account",
                "no_credit",
                format!(
                    "the key has no credit left ({}): raise its limit or add credits on openrouter.ai",
                    credit_text(info)
                ),
            )),
            _ => {}
        }
        if let Some((code, msg)) = &self.model.problem {
            v.push(("model", code, msg.clone()));
        }
        if let Some(Ok(t)) = &self.test_call
            && let Some(why) = t.problem(&o.key_env, &o.model)
        {
            v.push(("test_call", "test_call_failed", why));
        }
        v
    }

    /// Every check passed (and the test call answered, when asked).
    pub fn ready(&self) -> bool {
        self.problems().is_empty()
    }

    /// The flags that give `--oracle` this check's settings.
    pub fn oracle_flags(&self) -> String {
        let o = &self.options;
        let mut s = format!("--oracle {}", o.model);
        if o.key_env != crate::config::DEFAULT_ORACLE_KEY_ENV {
            s.push_str(&format!(" --oracle-key-env {}", o.key_env));
        }
        if o.base_url != crate::config::DEFAULT_ORACLE_BASE_URL {
            s.push_str(&format!(" --oracle-base-url {}", o.base_url));
        }
        if let Some((p, c)) = o.max_price {
            s.push_str(&format!(" --oracle-max-price {p},{c}"));
        }
        s
    }

    pub fn to_json(&self) -> Value {
        let o = &self.options;
        let account = match &self.account {
            None => json!({"checked": false, "ok": false}),
            Some(KeyProbe::Valid(info)) => {
                let mut v = info.to_json();
                v["checked"] = json!(true);
                v["ok"] = json!(!info.out_of_credit());
                v["status"] = json!(200);
                v
            }
            Some(KeyProbe::Refused(s)) => json!({"checked": true, "ok": false, "status": s}),
            Some(KeyProbe::Unreachable(why)) => {
                json!({"checked": true, "ok": false, "error": why})
            }
        };
        let m = &self.model;
        let (endpoints, structured) = match &m.probe {
            EndpointsProbe::Found(eps) => (
                json!(eps.len()),
                json!(eps.iter().filter(|e| e.structured).count()),
            ),
            _ => (Value::Null, Value::Null),
        };
        let found = match &m.probe {
            EndpointsProbe::Found(_) | EndpointsProbe::VariablePrice { .. } => json!(true),
            EndpointsProbe::NotFound(_) => json!(false),
            EndpointsProbe::Unreachable(_) => Value::Null,
        };
        let suggestions = match &m.suggestions {
            Some(Ok(list)) => json!(
                list.iter()
                    .map(|s| json!({"id": s.id, "prompt": s.prompt, "completion": s.completion}))
                    .collect::<Vec<_>>()
            ),
            _ => json!([]),
        };
        let test_call = match &self.test_call {
            None => Value::Null,
            Some(Ok(t)) => t.to_json(),
            Some(Err(why)) => json!({"ok": false, "made": false, "reason": why}),
        };
        let problems: Vec<Value> = self
            .problems()
            .into_iter()
            .map(|(check, code, message)| json!({"check": check, "code": code, "message": message}))
            .collect();
        json!({
            "ready": problems.is_empty(),
            "model": o.model,
            "base_url": o.base_url,
            "key_env": o.key_env,
            "key": {
                "present": self.key != crate::oracle::KeyState::Missing,
                "ok": matches!(self.key, crate::oracle::KeyState::Usable { .. }),
                "state": self.key.label(),
                "problem": self.key.problem(),
                "trimmed": self.key.trimmed() > 0,
            },
            "account": account,
            "model_check": {
                "ok": m.problem.is_none(),
                "found": found,
                "endpoints": endpoints,
                "structured_endpoints": structured,
                "cheapest": m.cheapest.as_ref().map(|e| json!({
                    "provider": e.provider, "prompt": e.prompt, "completion": e.completion,
                })),
                "max_price": self.max_price.map(|(p, c)| json!({"prompt": p, "completion": c})),
                "max_price_source": self.max_price.map(|_| {
                    if o.max_price.is_some() { "--max-price" } else { "listing" }
                }),
                "variable_price": matches!(m.probe, EndpointsProbe::VariablePrice { .. }),
                "suggestions": suggestions,
            },
            "test_call": test_call,
            "problems": problems,
        })
    }

    /// The human-readable report (never a key).
    pub fn render(&self) -> String {
        let o = &self.options;
        let problems = self.problems();
        let failed = |check: &str| problems.iter().find(|p| p.0 == check);
        let mut s = format!("Oracle check: {} via {}\n", o.model, host_of(&o.base_url));
        let line = |s: &mut String, ok: Option<bool>, what: &str, text: &str| {
            let mark = match ok {
                Some(true) => "✓",
                Some(false) => "✗",
                None => "–",
            };
            s.push_str(&format!("  {mark} {what:<10} {text}\n"));
        };
        match failed("key") {
            Some(p) => line(&mut s, Some(false), "key", &p.2),
            None if self.key.trimmed() > 0 => line(
                &mut s,
                Some(true),
                "key",
                &format!(
                    "{} is set (the key had surrounding whitespace, trimmed: {} byte{})",
                    o.key_env,
                    self.key.trimmed(),
                    if self.key.trimmed() == 1 { "" } else { "s" }
                ),
            ),
            None => line(&mut s, Some(true), "key", &format!("{} is set", o.key_env)),
        }
        match (&self.account, failed("account")) {
            (None, _) if self.key == crate::oracle::KeyState::Missing => {
                line(&mut s, None, "account", "not checked (no key)")
            }
            (None, _) => line(
                &mut s,
                None,
                "account",
                "not checked (the key is not usable)",
            ),
            (_, Some(p)) => line(&mut s, Some(false), "account", &p.2),
            (Some(KeyProbe::Valid(info)), None) => line(
                &mut s,
                Some(true),
                "account",
                &format!("the key is valid ({})", credit_text(info)),
            ),
            (Some(_), None) => {}
        }
        match failed("model") {
            Some(p) => {
                let tail = self
                    .model
                    .suggestions
                    .as_ref()
                    .map(suggestions_tail)
                    .unwrap_or_default();
                line(&mut s, Some(false), "model", &format!("{}.{tail}", p.2));
            }
            None => {
                let (n, js) = match &self.model.probe {
                    EndpointsProbe::Found(eps) => {
                        (eps.len(), eps.iter().filter(|e| e.structured).count())
                    }
                    _ => (0, 0),
                };
                let best = self.model.cheapest.as_ref();
                let variable = matches!(self.model.probe, EndpointsProbe::VariablePrice { .. });
                let text = match (best, self.max_price) {
                    (_, Some((p, c))) if variable => format!(
                        "listed only with variable pricing; the max price in/out {}/{} per 1M of \
                         --max-price caps every call's reservation (a call billed above it stops \
                         the oracle)",
                        usd(p),
                        usd(c)
                    ),
                    (Some(e), Some((p, c))) if o.max_price.is_some() => format!(
                        "{n} endpoints, {js} with structured outputs; the cheapest is {} at \
                         {}/{} per 1M in/out, within the max price {}/{} of --max-price",
                        e.provider,
                        usd(e.prompt),
                        usd(e.completion),
                        usd(p),
                        usd(c)
                    ),
                    (Some(e), Some((p, c))) => format!(
                        "{n} endpoints, {js} with structured outputs; the cheapest is {} at \
                         {}/{} per 1M in/out, so --oracle sets the max price to {}/{}",
                        e.provider,
                        usd(e.prompt),
                        usd(e.completion),
                        usd(p),
                        usd(c)
                    ),
                    _ => format!("{n} endpoints, {js} with structured outputs"),
                };
                line(&mut s, Some(true), "model", &text);
            }
        }
        match &self.test_call {
            None => line(
                &mut s,
                None,
                "test call",
                "not made (--test-call makes one tiny structured call, a small fraction of a cent)",
            ),
            Some(Err(why)) => line(&mut s, None, "test call", &format!("not made: {why}")),
            Some(Ok(t)) if t.ok() => line(
                &mut s,
                Some(true),
                "test call",
                &format!(
                    "answered '{}' for {} (reserved {}{}{})",
                    t.choice.as_deref().unwrap_or("?"),
                    t.cost_usd.map_or("?".to_string(), usd),
                    usd(t.reserved_usd),
                    t.provider
                        .as_deref()
                        .map(|p| format!(", provider {p}"))
                        .unwrap_or_default(),
                    t.latency_ms
                        .map(|ms| format!(", {ms:.0} ms"))
                        .unwrap_or_default()
                ),
            ),
            Some(Ok(_)) => {
                let why = failed("test_call").map_or("failed", |p| p.2.as_str());
                line(&mut s, Some(false), "test call", why);
            }
        }
        if problems.is_empty() {
            let flags = self.oracle_flags();
            s.push_str(&format!(
                "ready: cortiq serve FILE {flags}   (or: cortiq decide FILE -p TEXT {flags})\n"
            ));
        } else {
            s.push_str(&format!(
                "NOT ready: {} problem{} (✗ above)\n",
                problems.len(),
                if problems.len() == 1 { "" } else { "s" }
            ));
        }
        s
    }
}

/// `credit limit $10.00, $1.25 used, $8.75 left` (or what the key reports).
fn credit_text(info: &KeyInfo) -> String {
    let used = info.usage.map(|u| format!("{} used", usd(u)));
    let mut parts = Vec::new();
    match info.limit {
        Some(l) => parts.push(format!("credit limit {}", usd(l))),
        None => parts.push("no credit limit on the key".to_string()),
    }
    parts.extend(used);
    if let Some(r) = info.limit_remaining {
        parts.push(format!("{} left", usd(r.max(0.0))));
    }
    if info.free_tier == Some(true) {
        parts.push("free tier".to_string());
    }
    parts.join(", ")
}

/// `cortiq decision oracle check` (see [`CheckReport`]): (a) the key in the
/// variable ([`crate::oracle::read_key`]: trimmed of surrounding whitespace,
/// `bad_key` when it is not usable), (b) `GET {base}/auth/key` with it
/// (free; only with a usable key), (c) the model's public endpoint listing
/// (free, no key), (d) with `test_call` one tiny structured call — made
/// only when (a)–(c) passed. The flags are checked before the network (a
/// key typed as the model or the variable's name is refused without being
/// shown); the key is read from `key` and never kept in the report.
pub fn check(opts: &CheckOptions, key: &KeyLookup) -> Result<CheckReport> {
    check_model_id_as(&opts.model, "--model", "--key-env")?;
    check_key_env_name(&opts.key_env, "--key-env")?;
    crate::config::check_oracle_base_url("--base-url", &opts.base_url)?;
    let mut cfg = Config::default();
    cfg.oracle.model = opts.model.clone();
    cfg.oracle.api_key_env = opts.key_env.clone();
    cfg.oracle.base_url = opts.base_url.clone();
    if let Some(mp) = opts.max_price {
        set_max_price(&mut cfg, mp);
    }
    cfg.validate()?;
    let base = &opts.base_url;
    let mtq = cfg.oracle.max_tokens_per_question;
    let (key_state, usable) = crate::oracle::read_key(key, &opts.key_env);
    let account = usable.map(|k| probe_key(base, &k));
    let clean_key = crate::oracle::key_for_cleaning(key, &opts.key_env);
    let model = check_model(base, &opts.model, mtq, opts.max_price, clean_key.as_deref());
    let max_price = opts.max_price.or_else(|| {
        model
            .cheapest
            .as_ref()
            .map(|e| (e.prompt * PRICE_HEADROOM, e.completion * PRICE_HEADROOM))
    });
    let mut report = CheckReport {
        options: opts.clone(),
        key: key_state,
        account,
        model,
        max_price,
        test_call: None,
    };
    if opts.test_call {
        let blocked = report.problems();
        report.test_call = Some(match (blocked.first(), report.max_price) {
            (Some((check, _, _)), _) => Err(format!("the {check} check failed")),
            (None, None) => Err("no max price".into()),
            (None, Some(mp)) => {
                set_max_price(&mut cfg, mp);
                Ok(test_call(&cfg.oracle, Arc::clone(key))?)
            }
        });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A loopback HTTP server answering `GET path` from a table (404 else);
    /// records the request heads.
    struct Mock {
        base: String,
        heads: Arc<Mutex<Vec<String>>>,
    }

    fn mock(routes: Vec<(&'static str, u16, String)>) -> Mock {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/api/v1", l.local_addr().unwrap());
        let heads = Arc::new(Mutex::new(Vec::new()));
        let h = heads.clone();
        std::thread::spawn(move || {
            for c in l.incoming() {
                let Ok(mut s) = c else { continue };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                let head = String::from_utf8_lossy(&buf).to_string();
                let path = head.split(' ').nth(1).unwrap_or("").to_string();
                h.lock().unwrap().push(head);
                let (status, body) = routes.iter().find(|(p, _, _)| *p == path).map_or(
                    (404, "{\"error\":{\"code\":404}}".to_string()),
                    |(_, s, b)| (*s, b.clone()),
                );
                let _ = write!(
                    s,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Mock { base, heads }
    }

    fn endpoints(eps: Value) -> String {
        json!({"data": {"id": "a/b", "endpoints": eps}}).to_string()
    }

    fn ep(name: &str, prompt: &str, completion: &str, structured: bool) -> Value {
        let params = if structured {
            json!([
                "max_tokens",
                "temperature",
                "response_format",
                "structured_outputs"
            ])
        } else {
            json!(["max_tokens", "temperature"])
        };
        json!({"provider_name": name, "pricing": {"prompt": prompt, "completion": completion, "request": "0"},
               "supported_parameters": params})
    }

    fn models() -> String {
        json!({"data": [
            {"id": "x/pricey", "pricing": {"prompt": "0.000003", "completion": "0.000015"}, "supported_parameters": ["structured_outputs"]},
            {"id": "x/cheap", "pricing": {"prompt": "0.00000002", "completion": "0.0000001"}, "supported_parameters": ["response_format"]},
            {"id": "x/mid", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}, "supported_parameters": ["structured_outputs"]},
            {"id": "x/cheap:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["structured_outputs"]},
            {"id": "x/nojson", "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}, "supported_parameters": ["tools"]},
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}, "supported_parameters": ["structured_outputs"]},
            {"id": "x/low", "pricing": {"prompt": "0.00000005", "completion": "0.0000002"}, "supported_parameters": ["structured_outputs"]}
        ]})
        .to_string()
    }

    /// No key in any variable (the listings' names are cleaned without one).
    fn no_key() -> KeyLookup {
        Arc::new(|_| None)
    }

    fn flags(model: &str, base: &str) -> OracleFlags {
        OracleFlags {
            model: model.into(),
            base_url: Some(base.into()),
            ..OracleFlags::default()
        }
    }

    #[test]
    fn listing_sets_twice_the_cheapest_structured_price_without_a_key() {
        let m = mock(vec![(
            "/api/v1/models/a/b/endpoints",
            200,
            endpoints(json!([
                ep("Plain", "0.00000001", "0.00000001", false),
                ep("Dear", "0.0000002", "0.0000008", true),
                ep("Cheap", "0.000000035", "0.00000029", true),
            ])),
        )]);
        let mut cfg = Config::default();
        let mut f = flags("a/b", &m.base);
        f.budget_usd = Some(5.0);
        f.max_calls = Some(7);
        f.no_learning = true;
        let s = apply(&mut cfg, &f, false, &no_key()).unwrap();
        assert_eq!(s.max_price, (0.07, 0.58));
        assert_eq!(cfg.oracle.max_price().unwrap(), (0.07, 0.58));
        assert!(matches!(&s.source, PriceSource::Listing { provider, .. } if provider == "Cheap"));
        assert!(s.warnings.is_empty());
        assert!(cfg.oracle.enabled);
        assert_eq!(cfg.oracle.model, "a/b");
        assert_eq!((cfg.oracle.budget_usd, cfg.oracle.max_calls), (5.0, 7));
        assert!(!cfg.learning.enabled);
        assert_eq!(cfg.oracle.provider["sort"], "price");
        assert_eq!(cfg.oracle.provider["require_parameters"], true);
        assert_eq!(cfg.oracle.provider["allow_fallbacks"], true);
        assert!(
            s.price_note().contains("Cheap at $0.035/$0.29"),
            "{}",
            s.price_note()
        );
        // One public GET: no Authorization header.
        let heads = m.heads.lock().unwrap().clone();
        assert_eq!(heads.len(), 1);
        assert!(!heads[0].to_ascii_lowercase().contains("authorization"));
    }

    #[test]
    fn unknown_or_unstructured_models_are_refused_with_suggestions() {
        let m = mock(vec![
            (
                "/api/v1/models/a/plain/endpoints",
                200,
                endpoints(json!([ep("P", "0.0000001", "0.0000001", false)])),
            ),
            ("/api/v1/models", 200, models()),
        ]);
        let e = apply(
            &mut Config::default(),
            &flags("a/nope", &m.base),
            false,
            &no_key(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("--oracle a/nope") && e.contains("HTTP 404"),
            "{e}"
        );
        assert!(
            e.contains(
                "x/cheap ($0.02/$0.10 per 1M in/out), x/low ($0.05/$0.20 per 1M in/out), x/mid"
            ),
            "{e}"
        );
        assert!(
            !e.contains("pricey") && !e.contains(":free") && !e.contains("auto"),
            "{e}"
        );
        let e = apply(
            &mut Config::default(),
            &flags("a/plain", &m.base),
            false,
            &no_key(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("none of its 1 endpoints supports structured outputs"),
            "{e}"
        );
        assert!(e.contains("Start with --oracle x/cheap"), "{e}");
        // No model listing: the refusal still names the problem.
        let bare = mock(vec![]);
        let e = apply(
            &mut Config::default(),
            &flags("a/nope", &bare.base),
            false,
            &no_key(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("could not be fetched for suggestions"), "{e}");
    }

    #[test]
    fn unreachable_listing_falls_back_and_flags_or_config_win() {
        // A closed loopback port.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let base = format!("http://127.0.0.1:{port}/api/v1");
        let mut cfg = Config::default();
        let s = apply(&mut cfg, &flags("a/b", &base), false, &no_key()).unwrap();
        assert_eq!(s.max_price, FALLBACK_MAX_PRICE);
        assert_eq!(s.source, PriceSource::Fallback);
        assert!(
            s.warnings[0].contains("falls back to $0.10/$0.50"),
            "{:?}",
            s.warnings
        );
        // --oracle-max-price wins over the listing; a price below every
        // structured-output endpoint is refused (every call would be).
        let m = mock(vec![(
            "/api/v1/models/a/b/endpoints",
            200,
            endpoints(json!([ep("E", "0.0000001", "0.0000004", true)])),
        )]);
        let mut cfg = Config::default();
        let mut f = flags("a/b", &m.base);
        f.max_price = Some((0.3, 0.9));
        let s = apply(&mut cfg, &f, false, &no_key()).unwrap();
        assert_eq!((s.max_price, &s.source), ((0.3, 0.9), &PriceSource::Flag));
        assert!(s.warnings.is_empty());
        f.max_price = Some((0.01, 0.01));
        let e = apply(&mut Config::default(), &f, false, &no_key())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(
                "no structured-output endpoint is within the max price in/out $0.01/$0.01 per 1M of --oracle-max-price (the cheapest, E, costs $0.10/$0.40)"
            ) && e.contains("--oracle-max-price 0.2,0.8"),
            "{e}"
        );
        // The same from the configuration file names it.
        let mut low = Config::from_json(
            br#"{"oracle":{"provider":{"max_price":{"prompt":0.01,"completion":0.01}}}}"#,
        )
        .unwrap();
        let e = apply(&mut low, &flags("a/b", &m.base), true, &no_key())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("of oracle.provider.max_price of --decision-config"),
            "{e}"
        );
        // A configuration that sets the provider keeps its max price.
        let mut cfg = Config::from_json(
            br#"{"oracle":{"provider":{"sort":"price","max_price":{"prompt":0.2,"completion":0.8}}}}"#,
        )
        .unwrap();
        let s = apply(&mut cfg, &flags("a/b", &m.base), true, &no_key()).unwrap();
        assert_eq!((s.max_price, &s.source), ((0.2, 0.8), &PriceSource::Config));
    }

    #[test]
    fn variable_pricing_needs_a_given_max_price() {
        let auto = |structured: bool| endpoints(json!([ep("Auto", "-1", "-1", structured)]));
        let m = mock(vec![
            ("/api/v1/models/openrouter/auto/endpoints", 200, auto(true)),
            ("/api/v1/models/v/plain/endpoints", 200, auto(false)),
            ("/api/v1/models/v/none/endpoints", 200, endpoints(json!([]))),
            ("/api/v1/models", 200, models()),
        ]);
        // Listed, but no fixed price: refused, and the message says so.
        let e = apply(
            &mut Config::default(),
            &flags("openrouter/auto", &m.base),
            false,
            &no_key(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("lists it with no endpoint of a fixed price (variable pricing")
                && e.contains("--oracle-max-price IN,OUT")
                && !e.contains("does not list"),
            "{e}"
        );
        assert!(e.contains("Start with --oracle x/cheap"), "{e}");
        // With a given max price it starts, with a warning.
        let mut f = flags("openrouter/auto", &m.base);
        f.max_price = Some((0.2, 0.8));
        let mut cfg = Config::default();
        let s = apply(&mut cfg, &f, false, &no_key()).unwrap();
        assert_eq!((s.max_price, &s.source), ((0.2, 0.8), &PriceSource::Flag));
        assert!(
            s.warnings[0].contains("only variable-price endpoints")
                && s.warnings[0].contains("$0.20/$0.80"),
            "{:?}",
            s.warnings
        );
        // Variable pricing without structured outputs is refused either way.
        let mut f = flags("v/plain", &m.base);
        f.max_price = Some((0.2, 0.8));
        let e = apply(&mut Config::default(), &f, false, &no_key())
            .unwrap_err()
            .to_string();
        assert!(e.contains("supports structured outputs"), "{e}");
        // An empty listing: not listed.
        let e = apply(
            &mut Config::default(),
            &flags("v/none", &m.base),
            false,
            &no_key(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("does not list this model") && e.contains("lists no endpoint)"),
            "{e}"
        );
        assert_eq!(round_up(0.035 * 2.0), "0.07");
        assert_eq!(round_up(0.29 * 2.0), "0.58");
        assert_eq!(round_up(0.00001), "0.0001");
    }

    #[test]
    fn flags_are_checked_before_the_network() {
        let bad = |f: OracleFlags| {
            apply(&mut Config::default(), &f, false, &no_key())
                .unwrap_err()
                .to_string()
        };
        let e = bad(flags("a/b", "http://10.0.0.5:8080/v1"));
        assert!(e.contains("loopback"), "{e}");
        let e = bad(flags("a b", "http://127.0.0.1:9/v1"));
        assert!(e.contains("OpenRouter model id"), "{e}");
        let e = bad(flags("a/../b", "http://127.0.0.1:9/v1"));
        assert!(e.contains("OpenRouter model id"), "{e}");
        // A value without '/' (a short secret of no known shape) or with 32
        // hexadecimal digits in a row is not a model id, and is never shown.
        for (v, why) in [
            (
                "97b727bdfe44e04718e4047763da731a",
                "32 or more hexadecimal digits",
            ),
            ("SECRETVALUE1234", "has no '/'"),
        ] {
            let e = bad(flags(v, "http://127.0.0.1:9/v1"));
            assert!(e.contains(why) && !e.contains(&v[..8]), "{e}");
        }
        let e = bad(flags("SK-OR-V1-ABCDEF0123", "http://127.0.0.1:9/v1"));
        assert!(
            e.contains("looks like a key") && !e.contains("ABCDEF"),
            "{e}"
        );
        // A key typed where the variable's name belongs is refused and never
        // shown; so is a key given as the model (and it is not fetched).
        let key = "sk-or-v1-00112233445566778899aabbccddeeff";
        for v in ["NOT A NAME", key] {
            let mut f = flags("a/b", "http://127.0.0.1:9/v1");
            f.key_env = Some(v.into());
            let e = bad(f);
            assert!(
                e.contains("--oracle-key-env takes the NAME of the variable that holds the key"),
                "{e}"
            );
            assert!(!e.contains(v) && !e.contains("00112233"), "{e}");
        }
        let m = mock(vec![]);
        let e = bad(flags(key, &m.base));
        assert!(
            e.contains("not the key: put the key in OPENROUTER_API_KEY"),
            "{e}"
        );
        assert!(!e.contains("00112233"), "{e}");
        assert!(m.heads.lock().unwrap().is_empty(), "no request was made");
        let mut f = flags("a/b", "http://127.0.0.1:9/v1");
        f.budget_usd = Some(f64::NAN);
        assert!(bad(f).contains("--oracle-budget"));
        assert_eq!(parse_max_price("0.1, 0.5").unwrap(), (0.1, 0.5));
        assert!(parse_max_price("0.1").is_err());
        assert!(parse_max_price("-1,2").is_err());
        assert!(parse_max_price("x,2").is_err());
        // Never echoed: a key pasted here is refused by its length only.
        for v in [
            "sk-or-v1-0123456789abcdef",
            "0.1,sk-or-v1-0123456789abcdef",
            "notanumber-secret,0.5",
            "0.1 secretvalue",
        ] {
            let e = format!("{:#}", parse_max_price(v).unwrap_err());
            assert!(!e.contains("0123456789") && !e.contains("secret"), "{e}");
            // A key-like value names its length; the others leave it to the
            // flag's parser (stated once).
            assert_eq!(e.contains("bytes"), v.contains("sk-or-"), "{e}");
        }
    }

    fn key_description() -> String {
        json!({"data": {"label": "sk-or-v1-abc...xyz", "limit": 10, "usage": 1.25,
                        "limit_remaining": 8.75, "is_free_tier": false}})
        .to_string()
    }

    fn check_opts(model: &str, base: &str) -> CheckOptions {
        CheckOptions {
            model: model.into(),
            key_env: "UNIT_OR_KEY".into(),
            base_url: base.into(),
            test_call: false,
            max_price: None,
        }
    }

    #[test]
    fn check_reports_the_key_the_account_and_the_model() {
        let m = mock(vec![
            ("/api/v1/auth/key", 200, key_description()),
            (
                "/api/v1/models/a/b/endpoints",
                200,
                endpoints(json!([
                    ep("Plain", "0.00000001", "0.00000001", false),
                    ep("Cheap", "0.000000035", "0.00000029", true),
                ])),
            ),
            (
                "/api/v1/models/a/plain/endpoints",
                200,
                endpoints(json!([ep("P", "0.0000001", "0.0000001", false)])),
            ),
            ("/api/v1/models", 200, models()),
        ]);
        let secret = "sk-or-v1-unit-check-0011223344556677";
        let key: KeyLookup =
            crate::oracle::key_lookup(move |n| (n == "UNIT_OR_KEY").then(|| secret.to_string()));
        let r = check(&check_opts("a/b", &m.base), &key).unwrap();
        assert!(r.ready(), "{:?}", r.problems());
        assert_eq!(r.max_price, Some((0.07, 0.58)));
        assert_eq!(
            r.account,
            Some(KeyProbe::Valid(KeyInfo {
                limit: Some(10.0),
                usage: Some(1.25),
                limit_remaining: Some(8.75),
                free_tier: Some(false),
            }))
        );
        let text = r.render();
        assert!(
            text.contains(
                "✓ account    the key is valid (credit limit $10.00, $1.25 used, $8.75 left)"
            ),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "ready: cortiq serve FILE --oracle a/b --oracle-key-env UNIT_OR_KEY --oracle-base-url {}",
                m.base
            )),
            "{text}"
        );
        let j = r.to_json().to_string();
        for t in [&text, &j] {
            assert!(!t.contains(secret) && !t.contains("sk-or-v1-abc"), "{t}");
        }
        // The key went to /auth/key only, as a bearer; the listing had none.
        let heads = m.heads.lock().unwrap().clone();
        for h in &heads {
            let bearer = h.contains(&format!("Bearer {secret}"));
            assert_eq!(bearer, h.starts_with("GET /api/v1/auth/key "), "{h}");
        }

        // No key: the account is not checked, the model still is.
        let none: KeyLookup = Arc::new(|_| None);
        let r = check(&check_opts("a/b", &m.base), &none).unwrap();
        let p = r.problems();
        assert_eq!((p.len(), p[0].0, p[0].1), (1, "key", "no_key"));
        assert!(r.account.is_none() && r.model.problem.is_none());
        assert!(r.render().contains("– account    not checked (no key)"));
        // --test-call is not made when a check failed.
        let mut o = check_opts("a/b", &m.base);
        o.test_call = true;
        let r = check(&o, &none).unwrap();
        assert_eq!(r.test_call, Some(Err("the key check failed".into())));
        // An unknown model and one without structured outputs, with suggestions.
        let r = check(&check_opts("a/nope", &m.base), &key).unwrap();
        let (code, msg) = r.model.problem.clone().unwrap();
        assert_eq!(code, "unknown_model");
        assert!(msg.contains("HTTP 404"), "{msg}");
        assert!(
            r.render()
                .contains("Cheap models with structured outputs: x/cheap"),
            "{}",
            r.render()
        );
        let r = check(&check_opts("a/plain", &m.base), &key).unwrap();
        assert_eq!(r.model.problem.as_ref().unwrap().0, "no_structured_outputs");
        assert_eq!(r.to_json()["model_check"]["structured_endpoints"], 0);
    }

    #[test]
    fn check_refuses_keys_in_flags_and_reads_refused_or_empty_accounts() {
        let refused = mock(vec![(
            "/api/v1/auth/key",
            401,
            "{\"error\":{\"code\":401}}".into(),
        )]);
        assert_eq!(probe_key(&refused.base, "k"), KeyProbe::Refused(401));
        let broke = mock(vec![(
            "/api/v1/auth/key",
            200,
            json!({"data": {"limit": 5, "usage": 5.0, "limit_remaining": 0}}).to_string(),
        )]);
        let KeyProbe::Valid(info) = probe_key(&broke.base, "k") else {
            panic!("valid")
        };
        assert!(info.out_of_credit());
        let unlimited = parse_key_info(br#"{"data":{"limit":null,"usage":0.5}}"#).unwrap();
        assert!(!unlimited.out_of_credit());
        assert_eq!(
            credit_text(&unlimited),
            "no credit limit on the key, $0.50 used"
        );
        assert!(matches!(
            probe_key(&mock(vec![]).base, "k"),
            KeyProbe::Unreachable(_)
        ));
        // A key typed as the model or the variable's name: refused, not shown,
        // nothing sent.
        let m = mock(vec![]);
        let key = "sk-or-v1-00112233445566778899aabbccddeeff";
        let none: KeyLookup = Arc::new(|_| None);
        let e = check(&check_opts(key, &m.base), &none)
            .unwrap_err()
            .to_string();
        assert!(e.contains("--model takes an OpenRouter model id"), "{e}");
        let mut o = check_opts("a/b", &m.base);
        o.key_env = key.into();
        let e2 = check(&o, &none).unwrap_err().to_string();
        assert!(e2.contains("--key-env takes the NAME"), "{e2}");
        for e in [&e, &e2] {
            assert!(!e.contains("00112233"), "{e}");
        }
        assert!(m.heads.lock().unwrap().is_empty(), "no request was made");
        // Plain http to a non-loopback address is refused before the network.
        let e = check(&check_opts("a/b", "http://10.0.0.5:8080/v1"), &none)
            .unwrap_err()
            .to_string();
        assert!(e.contains("loopback"), "{e}");
        // check_flags: the same checks as apply, without the network.
        let mut f = flags(key, &m.base);
        assert!(check_flags(&f).is_err());
        f.model = "a/b".into();
        assert!(check_flags(&f).is_ok());
        assert!(m.heads.lock().unwrap().is_empty(), "no request was made");
    }

    #[test]
    fn check_trims_a_key_refuses_a_bad_one_unsent_and_takes_a_max_price() {
        let m = mock(vec![
            ("/api/v1/auth/key", 200, key_description()),
            (
                "/api/v1/models/a/b/endpoints",
                200,
                endpoints(json!([ep("Cheap", "0.000000035", "0.00000029", true)])),
            ),
            (
                "/api/v1/models/openrouter/auto/endpoints",
                200,
                endpoints(json!([ep("Auto", "-1", "-1", true)])),
            ),
            ("/api/v1/models", 200, models()),
        ]);
        let secret = "sk-or-v1-unit-TRIM-0011223344556677";
        let lookup = |raw: String| -> KeyLookup {
            crate::oracle::key_lookup(move |n| (n == "UNIT_OR_KEY").then(|| raw.clone()))
        };
        // Surrounding whitespace (a .env file's CR LF): trimmed, the account
        // checked with the key alone, and the report says so.
        let r = check(
            &check_opts("a/b", &m.base),
            &lookup(format!("  {secret}\r\n")),
        )
        .unwrap();
        assert!(r.ready(), "{:?}", r.problems());
        assert_eq!(r.key, crate::oracle::KeyState::Usable { trimmed: 4 });
        let text = r.render();
        assert!(
            text.contains("✓ key        UNIT_OR_KEY is set (the key had surrounding whitespace, trimmed: 4 bytes)"),
            "{text}"
        );
        assert_eq!(r.to_json()["key"]["trimmed"], true);
        let heads = m.heads.lock().unwrap().clone();
        let auth: Vec<&String> = heads.iter().filter(|h| h.contains("/auth/key")).collect();
        assert_eq!(auth.len(), 1);
        assert!(
            auth[0].contains(&format!("Bearer {secret}\r\n")),
            "{}",
            auth[0]
        );
        // Not a key after the trim: bad_key, nothing sent with it.
        let before = m.heads.lock().unwrap().len();
        for raw in [
            format!("Bearer {secret}"),
            format!("{} {}", &secret[..12], &secret[12..]),
            format!("{secret}\0"),
            format!("{secret}é"),
            format!("\"{secret}\""),
        ] {
            let r = check(&check_opts("a/b", &m.base), &lookup(raw.clone())).unwrap();
            let p = r.problems();
            assert_eq!((p[0].0, p[0].1), ("key", "bad_key"), "{raw:?}: {p:?}");
            assert!(r.account.is_none());
            let text = r.render();
            assert!(
                text.contains("– account    not checked (the key is not usable)"),
                "{text}"
            );
            let j = r.to_json();
            assert_eq!(j["key"]["state"], "bad", "{j}");
            for t in [&text, &j.to_string()] {
                assert!(
                    !t.contains("TRIM-0011") && !t.contains("0011223344556677"),
                    "{t}"
                );
            }
        }
        let heads = m.heads.lock().unwrap().clone();
        assert!(
            heads[before..].iter().all(|h| !h.contains("/auth/key")),
            "a bad key reached /auth/key"
        );
        // A model listed only with variable pricing needs --max-price.
        let key = lookup(secret.to_string());
        let r = check(&check_opts("openrouter/auto", &m.base), &key).unwrap();
        let (code, msg) = r.model.problem.clone().unwrap();
        assert_eq!(code, "variable_price");
        assert!(msg.contains("--max-price IN,OUT"), "{msg}");
        let mut o = check_opts("openrouter/auto", &m.base);
        o.max_price = Some((1.0, 4.0));
        let r = check(&o, &key).unwrap();
        assert!(r.ready(), "{:?}", r.problems());
        assert_eq!(r.max_price, Some((1.0, 4.0)));
        let text = r.render();
        assert!(text.contains("listed only with variable pricing"), "{text}");
        assert!(text.contains("--oracle openrouter/auto"), "{text}");
        assert!(text.contains("--oracle-max-price 1,4"), "{text}");
        assert_eq!(
            r.to_json()["model_check"]["max_price_source"],
            "--max-price"
        );
        // A given max price below every structured-output endpoint.
        let mut o = check_opts("a/b", &m.base);
        o.max_price = Some((0.01, 0.01));
        let r = check(&o, &key).unwrap();
        let (code, msg) = r.model.problem.clone().unwrap();
        assert_eq!(code, "max_price_too_low");
        assert!(msg.contains("--max-price 0.07,0.58"), "{msg}");
    }

    #[test]
    fn key_like_values_of_any_prefix_are_refused_unshown_and_unsent() {
        let m = mock(vec![]);
        let none: KeyLookup = Arc::new(|_| None);
        let hex64 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        for v in [
            hex64,
            "xx-sk-or-v1-0123456789abcdef",
            "Bearer 0123456789abcdef",
            " 0123456789abcdef",
            "0123456789abcdef\t",
        ] {
            let e = check(&check_opts(v, &m.base), &none)
                .unwrap_err()
                .to_string();
            assert!(e.contains("--model takes an OpenRouter model id"), "{e}");
            assert!(
                e.contains(&format!("({} bytes) is not shown", v.len())),
                "{e}"
            );
            let mut o = check_opts("a/b", &m.base);
            o.key_env = v.into();
            let e2 = check(&o, &none).unwrap_err().to_string();
            assert!(e2.contains("--key-env takes the NAME"), "{e2}");
            let mut f = flags(v, &m.base);
            let e3 = check_flags(&f).unwrap_err().to_string();
            f.model = "a/b".into();
            f.key_env = Some(v.into());
            let e4 = check_flags(&f).unwrap_err().to_string();
            assert!(e4.contains("--oracle-key-env takes the NAME"), "{e4}");
            for e in [&e, &e2, &e3, &e4] {
                assert!(!e.contains("0123456789abcdef"), "{e}");
            }
        }
        assert!(m.heads.lock().unwrap().is_empty(), "no request was made");
    }

    #[test]
    fn errors_are_codes_and_a_max_errors_stop_names_the_last_one() {
        // A closed port: the reason is a fixed code, not the library's text.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let base = format!("http://127.0.0.1:{port}/api/v1");
        match probe_key(&base, "sk-or-v1-0123456789abcdef") {
            KeyProbe::Unreachable(why) => {
                assert_eq!(why, "transport_connect: the request did not complete")
            }
            other => panic!("{other:?}"),
        }
        let e = explain_stop("max_errors", Some("http_503"), "K", "m/x");
        assert_eq!(
            e,
            "too many oracle calls failed in a row (oracle.max_errors); the last error: http_503 — OpenRouter answered HTTP 503"
        );
        let e = explain_stop("max_errors", Some("http_429"), "K", "m/x");
        assert!(
            e.ends_with("the last error: http_429 — OpenRouter rate-limited the call (HTTP 429)"),
            "{e}"
        );
        let e = explain_stop("max_errors", Some("transport_timeout"), "K", "m/x");
        assert!(
            e.contains("the last error: transport_timeout — the request did not complete"),
            "{e}"
        );
        assert_eq!(
            explain_oracle_error("http_500", "K", "m/x"),
            "OpenRouter answered HTTP 500 (http_500)"
        );
    }

    #[test]
    fn listing_names_are_kept_only_as_allowed_and_checked_against_the_configured_key() {
        let secret = "sk-or-v1-9f8e7d6c5b4a39281706aabbccddeeff00112233";
        let key: KeyLookup = crate::oracle::key_lookup(move |n| {
            (n == "UNIT_OR_KEY" || n == "OPENROUTER_API_KEY").then(|| format!(" {secret}\n"))
        });
        let names = [
            // A piece of the configured key (a proxy that saw it with a call).
            ("P 9f8e7d6c", "0.00000001", true),
            // Interleaved with dots: among the letters and digits.
            ("q/5.b.4.a.3.9.2.8", "0.00000002", true),
            // Outside the allowed characters, or longer than 64 bytes.
            ("Nov!ta", "0.00000003", true),
            ("Ok (EU)", "0.00000004", true),
            ("Long", "0.00000005", true),
        ];
        let long = format!("L/{}", "z".repeat(63));
        let eps: Vec<Value> = names
            .iter()
            .map(|(n, p, st)| {
                let n = if *n == "Long" { long.as_str() } else { n };
                ep(n, p, "0.0000001", *st)
            })
            .collect();
        let listing = endpoints(Value::Array(eps));
        let got: Vec<String> = parse_endpoints(listing.as_bytes(), Some(secret))
            .unwrap()
            .into_iter()
            .map(|e| e.provider)
            .collect();
        assert_eq!(
            got,
            [
                "[redacted]",
                "[redacted]",
                "[redacted]",
                "Ok (EU)",
                "[redacted]"
            ]
        );
        // Without the key only the shapes are refused.
        let got: Vec<String> = parse_endpoints(listing.as_bytes(), None)
            .unwrap()
            .into_iter()
            .map(|e| e.provider)
            .collect();
        assert_eq!(
            got,
            [
                "P 9f8e7d6c",
                "q/5.b.4.a.3.9.2.8",
                "[redacted]",
                "Ok (EU)",
                "[redacted]"
            ]
        );
        // apply and check read the configured variable for it: the cheapest
        // provider (an echo of the key) is [redacted] in every message.
        let m = mock(vec![
            (
                "/api/v1/models/a/b/endpoints",
                200,
                endpoints(json!([ep("P 9f8e7d6c", "0.000000035", "0.00000029", true)])),
            ),
            ("/api/v1/auth/key", 200, key_description()),
            ("/api/v1/models/a/nope/endpoints", 404, "{}".into()),
            (
                "/api/v1/models",
                200,
                json!({"data": [
                    {"id": "x/9f8e7d6c-mini", "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}, "supported_parameters": ["structured_outputs"]},
                    {"id": "x/a b", "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}, "supported_parameters": ["structured_outputs"]},
                    {"id": "noslash", "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}, "supported_parameters": ["structured_outputs"]},
                    {"id": format!("x/{}", "y".repeat(70)), "pricing": {"prompt": "0.00000001", "completion": "0.00000001"}, "supported_parameters": ["structured_outputs"]},
                    {"id": "x/fine", "pricing": {"prompt": "0.00000002", "completion": "0.00000002"}, "supported_parameters": ["structured_outputs"]}
                ]})
                .to_string(),
            ),
        ]);
        let mut cfg = Config::default();
        let s = apply(&mut cfg, &flags("a/b", &m.base), false, &key).unwrap();
        assert!(
            matches!(&s.source, PriceSource::Listing { provider, .. } if provider == "[redacted]"),
            "{:?}",
            s.source
        );
        assert!(s.price_note().contains("[redacted] at $0.035/$0.29"));
        let e = apply(
            &mut Config::default(),
            &flags("a/nope", &m.base),
            false,
            &key,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("Cheap models with structured outputs: x/fine (") && !e.contains("9f8e7d6c"),
            "{e}"
        );
        let mut o = check_opts("a/b", &m.base);
        o.key_env = "UNIT_OR_KEY".into();
        let r = check(&o, &key).unwrap();
        assert_eq!(r.model.cheapest.as_ref().unwrap().provider, "[redacted]");
        for t in [r.render(), r.to_json().to_string()] {
            assert!(!t.contains("9f8e7d6c"), "{t}");
        }
        let r = check(&check_opts("a/nope", &m.base), &key).unwrap();
        let ids: Vec<String> = r
            .model
            .suggestions
            .clone()
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, ["x/fine"]);
        // The listings were fetched without it.
        for h in m.heads.lock().unwrap().iter() {
            assert!(
                !h.contains("9f8e7d6c") || h.starts_with("GET /api/v1/auth/key "),
                "{h}"
            );
        }
    }

    #[test]
    fn codes_outside_the_closed_set_are_worded_as_unknown_code() {
        let echo = "finish_Bearer_sk_or_v1_0123456789abcdef";
        for text in [
            explain_oracle_error(echo, "K", "m/x"),
            explain_last_error(echo, "K", "m/x"),
            explain_stop("max_errors", Some(echo), "K", "m/x"),
            explain_stop(echo, None, "K", "m/x"),
            explain_oracle_error("http_4011", "K", "m/x"),
        ] {
            assert!(
                !text.contains("0123456789") && !text.contains("4011"),
                "{text}"
            );
        }
        assert_eq!(
            explain_last_error(echo, "K", "m/x"),
            "the last error: unknown_code — a code this version does not know was recorded (it is not shown)"
        );
        assert_eq!(
            explain_oracle_error("finish_length", "K", "m/x"),
            "the answer was not usable (finish_length)"
        );
    }

    #[test]
    fn helpers_format_and_encode() {
        // Advice rounds up (the printed figure holds the amount), a
        // statement shows more digits.
        assert_eq!(usd(0.00178924), "$0.001789");
        assert_eq!(usd_ceil(0.00178924), "$0.00179");
        assert_eq!(usd_fine(0.00178924), "$0.00178924");
        assert_eq!(usd_ceil(0.001789), "$0.001789");
        assert_eq!(usd_ceil(1.0), "$1.00");
        assert_eq!(usd_ceil(1e-9), "$0.000001");
        for x in [0.00178924, 0.00034288, 0.000283, 1.3e-5, 0.1 + 0.2, 7.0e-6] {
            let printed: f64 = usd_ceil(x)[1..].parse().unwrap();
            assert!(printed >= x, "{x} -> {printed}");
            assert!(printed - x < 1.000_001e-6, "{x} -> {printed}");
        }
        assert_eq!(usd(0.07), "$0.07");
        assert_eq!(usd(0.055), "$0.055");
        assert_eq!(usd(5.0), "$5.00");
        assert_eq!(usd(0.000013), "$0.000013");
        assert_eq!(usd(0.0), "$0.00");
        assert_eq!(host_of("https://openrouter.ai/api/v1"), "openrouter.ai");
        assert_eq!(host_of("http://127.0.0.1:9/v1"), "127.0.0.1:9");
        assert_eq!(
            endpoints_url("https://openrouter.ai/api/v1/", "a/b:nitro"),
            "https://openrouter.ai/api/v1/models/a/b:nitro/endpoints"
        );
        assert_eq!(encode_path("a/b?c#d"), "a/b%3Fc%23d");
        assert_eq!(per_million(Some(&json!("0.00000007"))), Some(0.07));
        assert_eq!(per_million(Some(&json!("-1"))), None);
        assert_eq!(per_million(Some(&json!(2e-7))), Some(0.2));
    }
}
