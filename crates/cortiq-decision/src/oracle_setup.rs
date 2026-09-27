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
//! What is typed where a secret must not be never comes back in a message:
//! `--oracle-key-env` takes the variable's name, and a value that is not a
//! name (an OpenRouter key pasted by mistake) is refused without being shown;
//! so is an `--oracle` model id that looks like a key (`sk-…`), which is
//! also never sent in a listing URL.

use crate::config::Config;
use crate::oracle::{max_tokens, reservation_usd};
use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};
use std::io::Read;
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

/// `IN,OUT`: two finite non-negative numbers, USD per 1M tokens.
pub fn parse_max_price(s: &str) -> Result<(f64, f64)> {
    let Some((a, b)) = s.split_once(',') else {
        bail!("expected IN,OUT (USD per 1M prompt and completion tokens), e.g. 0.1,0.5");
    };
    let num = |x: &str, what: &str| -> Result<f64> {
        let v: f64 = x
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("{what} price '{}' is not a number", x.trim()))?;
        ensure!(
            v.is_finite() && v >= 0.0,
            "{what} price must be finite and non-negative"
        );
        Ok(v)
    };
    Ok((num(a, "the prompt")?, num(b, "the completion")?))
}

/// A model id usable in a listing URL: 1..256 bytes, no whitespace or
/// control character, no empty, `.` or `..` path segment, and not a key
/// (`sk-…`; refused without being shown, and never put in a URL).
fn check_model_id(model: &str) -> Result<()> {
    ensure!(
        !model.starts_with("sk-"),
        "--oracle takes an OpenRouter model id such as deepseek/deepseek-v4.1-flash, not the \
         key: put the key in OPENROUTER_API_KEY (or the variable --oracle-key-env names); the \
         given value ({} bytes) is not shown",
        model.len()
    );
    ensure!(
        !model.is_empty()
            && model.len() <= 256
            && !model.chars().any(|c| c.is_whitespace() || c.is_control())
            && model.split('/').all(|seg| !matches!(seg, "" | "." | "..")),
        "--oracle expects an OpenRouter model id such as deepseek/deepseek-v4.1-flash, got '{}'",
        model.escape_debug()
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

/// One GET without a key: `(status, body)` or a transport error.
fn get(url: &str, cap: u64) -> std::result::Result<(u16, Vec<u8>), String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(PROBE_TIMEOUT)
        .redirects(0)
        .build();
    let resp = match agent.get(url).set("Accept", "application/json").call() {
        Ok(r) => r,
        Err(ureq::Error::Status(_, r)) => r,
        Err(ureq::Error::Transport(t)) => return Err(t.to_string()),
    };
    let status = resp.status();
    let mut buf = Vec::new();
    resp.into_reader()
        .take(cap + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("reading the answer: {e}"))?;
    if buf.len() as u64 > cap {
        return Err(format!("the answer is larger than {cap} bytes"));
    }
    Ok((status, buf))
}

/// An endpoint-listing body (`data.endpoints`): the endpoints with a fixed
/// price, and for each one without it whether it supports structured
/// outputs. `None` when the body does not have that shape.
fn parse_listing(body: &[u8]) -> Option<(Vec<Endpoint>, Vec<bool>)> {
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
            provider: name.chars().filter(|c| !c.is_control()).take(80).collect(),
            prompt,
            completion,
            structured,
        });
    }
    Some((priced, unpriced))
}

/// The endpoints with a fixed price of an endpoint-listing body
/// (`data.endpoints`), `None` when the body does not have that shape.
pub fn parse_endpoints(body: &[u8]) -> Option<Vec<Endpoint>> {
    parse_listing(body).map(|(priced, _)| priced)
}

/// `GET {base_url}/models/{model}/endpoints`, without a key.
pub fn probe_endpoints(base_url: &str, model: &str) -> EndpointsProbe {
    let url = endpoints_url(base_url, model);
    match get(&url, MAX_ENDPOINTS_BYTES) {
        Err(why) => EndpointsProbe::Unreachable(why),
        Ok((status @ (400 | 404), _)) => {
            EndpointsProbe::NotFound(format!("{url} answered HTTP {status}"))
        }
        Ok((200, body)) => match parse_listing(&body) {
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
/// outputs, have a positive price and are not a `:free` variant.
pub fn parse_models(body: &[u8]) -> Option<Vec<ListedModel>> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let data = v.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?;
                if id.ends_with(":free")
                    || id.len() > 256
                    || id.chars().any(|c| c.is_whitespace() || c.is_control())
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

/// `GET {base_url}/models`, without a key: the models with structured outputs.
pub fn list_models(base_url: &str) -> std::result::Result<Vec<ListedModel>, String> {
    let url = models_url(base_url);
    match get(&url, MAX_MODELS_BYTES)? {
        (200, body) => parse_models(&body).ok_or_else(|| format!("{url}: not a model listing")),
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

/// The refusal of a model the oracle cannot use, with suggestions.
fn refuse(cfg: &Config, model: &str, problem: &str) -> anyhow::Error {
    let base = &cfg.oracle.base_url;
    let mtq = cfg.oracle.max_tokens_per_question;
    let tail = match list_models(base) {
        Ok(models) => {
            let best = cheapest_models(&models, model, mtq, SUGGESTIONS);
            if best.is_empty() {
                " No model of the listing supports structured outputs.".to_string()
            } else {
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
        }
        Err(why) => format!(" (the model listing could not be fetched for suggestions: {why})"),
    };
    anyhow::anyhow!("--oracle {model}: {problem}.{tail}")
}

/// Apply `flags` to `cfg` (see the module notes) and check the model against
/// the endpoint listing of `cfg.oracle.base_url` (one GET, no key; one more
/// for suggestions when the model is refused). `config_sets_max_price`: the
/// `--decision-config` file sets `oracle.provider` itself.
pub fn apply(
    cfg: &mut Config,
    flags: &OracleFlags,
    config_sets_max_price: bool,
) -> Result<OracleSetup> {
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
        ensure!(
            crate::config::is_env_name(k),
            "--oracle-key-env takes the NAME of the variable that holds the key (e.g. \
             OPENROUTER_API_KEY), not the key itself; the given value ({} bytes) is not shown",
            k.len()
        );
        o.api_key_env = k.clone();
    }
    if let Some(u) = &flags.base_url {
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
    let model = flags.model.clone();
    let mtq = cfg.oracle.max_tokens_per_question;
    let probe = probe_endpoints(&cfg.oracle.base_url, &model);
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
        let s = apply(&mut cfg, &f, false).unwrap();
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
        let e = apply(&mut Config::default(), &flags("a/nope", &m.base), false)
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
        let e = apply(&mut Config::default(), &flags("a/plain", &m.base), false)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("none of its 1 endpoints supports structured outputs"),
            "{e}"
        );
        assert!(e.contains("Start with --oracle x/cheap"), "{e}");
        // No model listing: the refusal still names the problem.
        let bare = mock(vec![]);
        let e = apply(&mut Config::default(), &flags("a/nope", &bare.base), false)
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
        let s = apply(&mut cfg, &flags("a/b", &base), false).unwrap();
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
        let s = apply(&mut cfg, &f, false).unwrap();
        assert_eq!((s.max_price, &s.source), ((0.3, 0.9), &PriceSource::Flag));
        assert!(s.warnings.is_empty());
        f.max_price = Some((0.01, 0.01));
        let e = apply(&mut Config::default(), &f, false)
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
        let e = apply(&mut low, &flags("a/b", &m.base), true)
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
        let s = apply(&mut cfg, &flags("a/b", &m.base), true).unwrap();
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
        let s = apply(&mut cfg, &f, false).unwrap();
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
        let e = apply(&mut Config::default(), &f, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("supports structured outputs"), "{e}");
        // An empty listing: not listed.
        let e = apply(&mut Config::default(), &flags("v/none", &m.base), false)
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
            apply(&mut Config::default(), &f, false)
                .unwrap_err()
                .to_string()
        };
        let e = bad(flags("a/b", "http://10.0.0.5:8080/v1"));
        assert!(e.contains("loopback"), "{e}");
        let e = bad(flags("a b", "http://127.0.0.1:9/v1"));
        assert!(e.contains("OpenRouter model id"), "{e}");
        let e = bad(flags("a/../b", "http://127.0.0.1:9/v1"));
        assert!(e.contains("OpenRouter model id"), "{e}");
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
    }

    #[test]
    fn helpers_format_and_encode() {
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
