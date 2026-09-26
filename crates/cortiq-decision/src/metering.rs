//! Token metering and price (spec §4.9).
//!
//! **Tokens** (WordPiece of the file's tokenizer, `[CLS]`/`[SEP]` excluded, no
//! truncation), counted once per request like Jev counts its input:
//!
//! * `input_tokens` = tokens of the `state` the model received (a string as is,
//!   an object or array as canonical JSON) + for every question the tokens of
//!   its `instructions` (a string as is, an object or array as canonical JSON:
//!   [`value_text`]) and of its `criteria`: every key as text (choice option ids,
//!   the `true`/`false` of a noul question) and every value as its canonical
//!   JSON ([`criterion_text`]: a string with its quotes, `null` as `null`) —
//!   choice descriptions, score levels, noul descriptions;
//! * `output_tokens` = per answer, the number of values in `probabilities`
//!   (choice and score answers that carry them), else 1 (oracle and cache
//!   answers, noul);
//! * `processed_tokens` (`cmf.usage.local`) = tokens the encoder actually ran:
//!   the state with `[CLS]`/`[SEP]`, truncated to 512.
//!
//! Oracle tokens are not part of `usage.input_tokens`; they are reported under
//! `cmf.usage.oracle`.
//!
//! **Cost** = `in·input_usd_per_1m/1e6 + out·output_usd_per_1m/1e6 +
//! request_usd`, plus, with `oracle_passthrough`, `Σ usage.cost` of the
//! successful oracle calls × `oracle_markup`. All prices default to `"0"`: a
//! local answer costs 0 and an oracle answer its own cost. Money is exact
//! decimal arithmetic ([`Usd`], units of 1e-24 USD), so `4372 × "0.042"/1e6` is
//! exactly `0.000183624`; the JSON number is the f64 nearest to the exact value.

use crate::canonical;
use crate::wordpiece::WordPiece;
use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;

/// Decimal places of [`Usd`].
pub const USD_SCALE: u32 = 24;

/// A non-negative amount of US dollars in units of 1e-24 USD (exact for every
/// decimal price with at most 24 decimals).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Usd(i128);

impl std::fmt::Display for Usd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format_scaled(self.0, USD_SCALE))
    }
}

/// `m · 10^-scale` as a decimal without exponent and trailing zeros.
fn format_scaled(m: i128, scale: u32) -> String {
    let neg = m < 0;
    let digits = m.unsigned_abs().to_string();
    let scale = scale as usize;
    let (int, frac) = if digits.len() > scale {
        let (a, b) = digits.split_at(digits.len() - scale);
        (a.to_string(), b.to_string())
    } else {
        ("0".to_string(), format!("{digits:0>scale$}"))
    };
    let frac = frac.trim_end_matches('0');
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    s.push_str(&int);
    if !frac.is_empty() {
        s.push('.');
        s.push_str(frac);
    }
    s
}

/// Parse a plain decimal (`digits[.digits]`) into a mantissa at `scale`
/// decimals; extra decimals are rounded half to even when `round`, else an
/// error.
fn parse_scaled(s: &str, scale: u32, round: bool) -> Result<i128> {
    let (int, frac) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    ensure!(
        !int.is_empty() && int.bytes().all(|b| b.is_ascii_digit()),
        "'{s}' is not a non-negative decimal (digits[.digits])"
    );
    ensure!(
        frac.bytes().all(|b| b.is_ascii_digit()) && !(s.contains('.') && frac.is_empty()),
        "'{s}' is not a non-negative decimal (digits[.digits])"
    );
    let scale = scale as usize;
    let (keep, rest) = if frac.len() > scale {
        if !round && frac[scale..].bytes().any(|b| b != b'0') {
            bail!("'{s}' has more than {scale} decimals");
        }
        (&frac[..scale], &frac[scale..])
    } else {
        (frac, "")
    };
    let int = int.trim_start_matches('0');
    let text = format!("{int}{keep:0<scale$}");
    let mut m: i128 = if text.is_empty() {
        0
    } else {
        text.parse()
            .map_err(|_| anyhow::anyhow!("'{s}' is too large"))?
    };
    if !rest.is_empty() {
        let first = rest.as_bytes()[0];
        let tail_nonzero = rest[1..].bytes().any(|b| b != b'0');
        let up = first > b'5' || (first == b'5' && (tail_nonzero || m % 2 == 1));
        if up {
            m = m
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("'{s}' is too large"))?;
        }
    }
    Ok(m)
}

impl Usd {
    pub const ZERO: Usd = Usd(0);

    /// A decimal price string (`"0"`, `"0.042"`, `"1.5"`): no sign, no
    /// exponent, at most 24 decimals.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(Self(parse_scaled(s.trim(), USD_SCALE, false)?))
    }

    /// An f64 amount (an oracle's `usage.cost`): finite and non-negative, taken
    /// at its shortest round-trip decimal and rounded half to even at 1e-24.
    pub fn from_f64(x: f64) -> Result<Self> {
        ensure!(
            x.is_finite() && x >= 0.0,
            "a cost must be finite and non-negative, got {x}"
        );
        Ok(Self(parse_scaled(&format!("{x}"), USD_SCALE, true)?))
    }

    /// The raw units of 1e-24 USD.
    pub fn units(self) -> i128 {
        self.0
    }

    /// From raw units of 1e-24 USD (non-negative).
    pub fn from_units(units: i128) -> Result<Self> {
        ensure!(units >= 0, "a USD amount is non-negative");
        Ok(Self(units))
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The f64 nearest to the exact value.
    pub fn to_f64(self) -> f64 {
        self.to_string()
            .parse()
            .expect("a plain decimal parses as f64")
    }

    pub fn checked_add(self, o: Usd) -> Result<Usd> {
        self.0
            .checked_add(o.0)
            .map(Usd)
            .ok_or_else(|| anyhow::anyhow!("USD amount overflow"))
    }

    /// `self + o`, saturating (totals never wrap).
    pub fn saturating_add(self, o: Usd) -> Usd {
        Usd(self.0.saturating_add(o.0))
    }

    /// The price of `tokens` at `self` USD per 1M tokens (rounded half to even
    /// at 1e-24 when the price has more than 18 decimals).
    pub fn per_million(self, tokens: u64) -> Result<Usd> {
        let p = self
            .0
            .checked_mul(tokens as i128)
            .ok_or_else(|| anyhow::anyhow!("price overflow"))?;
        Ok(Usd(div_round_half_even(p, 1_000_000)))
    }

    /// `self × r`, rounded half to even at 1e-24.
    pub fn times(self, r: Ratio) -> Result<Usd> {
        let p = self
            .0
            .checked_mul(r.mantissa)
            .ok_or_else(|| anyhow::anyhow!("markup overflow"))?;
        Ok(Usd(div_round_half_even(p, 10i128.pow(r.scale))))
    }

    /// The price per token of a price per 1M tokens, as a decimal string
    /// without exponent (OpenRouter pricing strings, spec §4.12).
    pub fn per_token_string(self) -> String {
        format_scaled(self.0, USD_SCALE + 6)
    }
}

fn div_round_half_even(n: i128, d: i128) -> i128 {
    let q = n / d;
    let r = n % d;
    let twice = r * 2;
    if twice > d || (twice == d && q % 2 == 1) {
        q + 1
    } else {
        q
    }
}

/// A non-negative decimal factor (the oracle markup): `mantissa · 10^-scale`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    mantissa: i128,
    scale: u32,
}

impl Ratio {
    pub const ONE: Ratio = Ratio {
        mantissa: 1,
        scale: 0,
    };

    /// An f64 factor at its shortest round-trip decimal (at most 18 decimals).
    pub fn from_f64(x: f64) -> Result<Self> {
        ensure!(
            x.is_finite() && x >= 0.0,
            "a markup must be finite and non-negative, got {x}"
        );
        let s = format!("{x}");
        let decimals = s.split_once('.').map_or(0, |(_, f)| f.len()) as u32;
        ensure!(decimals <= 18, "markup {x} has more than 18 decimals");
        Ok(Self {
            mantissa: parse_scaled(&s, decimals, false)?,
            scale: decimals,
        })
    }

    pub fn to_f64(self) -> f64 {
        format_scaled(self.mantissa, self.scale)
            .parse()
            .expect("a plain decimal parses as f64")
    }
}

/// The parsed price configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    pub input_per_1m: Usd,
    pub output_per_1m: Usd,
    pub request: Usd,
    pub oracle_passthrough: bool,
    pub oracle_markup: Ratio,
}

impl Default for Rates {
    fn default() -> Self {
        Self {
            input_per_1m: Usd::ZERO,
            output_per_1m: Usd::ZERO,
            request: Usd::ZERO,
            oracle_passthrough: true,
            oracle_markup: Ratio::ONE,
        }
    }
}

/// The money of one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// Input, output and request prices.
    pub local: Usd,
    /// The oracle's own cost of the request's successful calls.
    pub oracle_raw: Usd,
    /// What the client pays for the oracle (`oracle_raw × markup` with
    /// passthrough, else 0).
    pub oracle_billed: Usd,
    /// `local + oracle_billed` (`usage.cost`).
    pub total: Usd,
}

impl Rates {
    pub fn new(
        input_per_1m: Usd,
        output_per_1m: Usd,
        request: Usd,
        oracle_passthrough: bool,
        oracle_markup: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_per_1m,
            output_per_1m,
            request,
            oracle_passthrough,
            oracle_markup: Ratio::from_f64(oracle_markup)?,
        })
    }

    /// Every price is zero.
    pub fn is_free(&self) -> bool {
        self.input_per_1m.is_zero() && self.output_per_1m.is_zero() && self.request.is_zero()
    }

    /// The cost of a request (spec §4.9).
    pub fn cost(&self, input_tokens: u64, output_tokens: u64, oracle_raw: Usd) -> Result<Cost> {
        let local = self
            .input_per_1m
            .per_million(input_tokens)?
            .checked_add(self.output_per_1m.per_million(output_tokens)?)?
            .checked_add(self.request)?;
        let oracle_billed = if self.oracle_passthrough {
            oracle_raw.times(self.oracle_markup)?
        } else {
            Usd::ZERO
        };
        Ok(Cost {
            local,
            oracle_raw,
            oracle_billed,
            total: local.checked_add(oracle_billed)?,
        })
    }
}

// ------------------------------------------------------------------ tokens

/// The text of a JSON value that is metered: a string as is, `null` as
/// nothing, anything else as its canonical JSON.
pub fn value_text(v: &Value) -> Cow<'_, str> {
    match v {
        Value::String(s) => Cow::Borrowed(s.as_str()),
        Value::Null => Cow::Borrowed(""),
        other => Cow::Owned(canonical::to_string(other)),
    }
}

/// WordPiece tokens of a text (`[CLS]`/`[SEP]` excluded, untruncated).
pub fn text_tokens(tokenizer: &WordPiece, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    tokenizer.encode_pieces(text).len() as u64
}

/// Tokens of a JSON value ([`value_text`]).
pub fn value_tokens(tokenizer: &WordPiece, v: &Value) -> u64 {
    text_tokens(tokenizer, &value_text(v))
}

/// The metered text of a criteria value: its canonical JSON (spec §4.9), so a
/// string counts with its quotes and `null` as `null`.
pub fn criterion_text(v: &Value) -> String {
    canonical::to_string(v)
}

/// Tokens of the state from the encoder's own count: `processed` includes
/// `[CLS]`/`[SEP]` and is truncated at `max_length`; a truncated state is
/// re-tokenized without truncation.
pub fn state_tokens(tokenizer: &WordPiece, state_text: &str, processed: usize) -> u64 {
    if processed < tokenizer.max_length() {
        processed.saturating_sub(2) as u64
    } else {
        text_tokens(tokenizer, state_text)
    }
}

/// Tokens of one question's contract: `instructions` ([`value_text`]) plus
/// every criteria key as text and every criteria value as canonical JSON
/// ([`criterion_text`]); a noul question without criteria adds nothing.
pub fn contract_tokens(tokenizer: &WordPiece, instructions: &Value, criteria: &Value) -> u64 {
    let mut n = value_tokens(tokenizer, instructions);
    match criteria {
        Value::Object(m) => {
            for (k, v) in m {
                n += text_tokens(tokenizer, k) + text_tokens(tokenizer, &criterion_text(v));
            }
        }
        Value::Array(a) => {
            for v in a {
                n += text_tokens(tokenizer, &criterion_text(v));
            }
        }
        Value::Null => {}
        other => n += text_tokens(tokenizer, &criterion_text(other)),
    }
    n
}

/// Output tokens of one answer object: the size of its `probabilities`, else 1.
pub fn answer_output_tokens(answer: &Value) -> u64 {
    match answer.get("probabilities") {
        Some(Value::Object(p)) => p.len() as u64,
        _ => 1,
    }
}

/// A bounded memo of question-contract token counts, keyed by the sha256 of
/// the contract's canonical JSON (the rubric of a Jev request is the same for
/// every row; tokenizing 150 criteria costs more than the decision).
#[derive(Debug)]
pub struct TokenCache {
    cap: usize,
    map: Mutex<HashMap<[u8; 32], u64>>,
}

impl TokenCache {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            map: Mutex::new(HashMap::new()),
        }
    }

    /// The tokens of a contract, from the memo or counted now.
    pub fn contract_tokens(
        &self,
        tokenizer: &WordPiece,
        contract: &Value,
        instructions: &Value,
        criteria: &Value,
    ) -> u64 {
        let key: [u8; 32] = Sha256::digest(canonical::to_string(contract).as_bytes()).into();
        if let Some(&n) = self.map.lock().get(&key) {
            return n;
        }
        let n = contract_tokens(tokenizer, instructions, criteria);
        let mut m = self.map.lock();
        if m.len() >= self.cap {
            m.clear();
        }
        m.insert(key, n);
        n
    }

    pub fn len(&self) -> usize {
        self.map.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usd_parse_format_and_exact_products() {
        assert_eq!(Usd::parse("0").unwrap(), Usd::ZERO);
        assert_eq!(Usd::parse("0.042").unwrap().to_string(), "0.042");
        assert_eq!(Usd::parse("00.50").unwrap().to_string(), "0.5");
        assert_eq!(Usd::parse("12").unwrap().to_string(), "12");
        for bad in ["", "-1", "1e-3", ".5", "5.", "1.2.3", " ", "0x1", "+1"] {
            assert!(Usd::parse(bad).is_err(), "{bad:?}");
        }
        assert!(Usd::parse("0.0000000000000000000000001").is_err());
        let p = Usd::parse("0.042").unwrap();
        let c = p.per_million(4372).unwrap();
        assert_eq!(c.to_string(), "0.000183624");
        assert_eq!(c.to_f64(), 0.000183624);
        assert_eq!(p.per_token_string(), "0.000000042");
        assert_eq!(Usd::ZERO.per_token_string(), "0");
        assert_eq!(Usd::from_f64(1.7472e-5).unwrap().to_string(), "0.000017472");
        assert!(Usd::from_f64(-1.0).is_err());
        assert!(Usd::from_f64(f64::NAN).is_err());
    }

    #[test]
    fn rounding_is_half_even() {
        assert_eq!(div_round_half_even(5, 2), 2);
        assert_eq!(div_round_half_even(7, 2), 4);
        assert_eq!(div_round_half_even(6, 4), 2);
        assert_eq!(div_round_half_even(10, 4), 2);
        assert_eq!(div_round_half_even(11, 4), 3);
        // 25 decimals: the 25th rounds half to even.
        let u = Usd::from_f64(1.5e-24).unwrap();
        assert_eq!(u.units(), 2);
        let u = Usd::from_f64(2.5e-24).unwrap();
        assert_eq!(u.units(), 2);
    }

    #[test]
    fn cost_follows_the_formula() {
        let r = Rates::new(
            Usd::parse("0.042").unwrap(),
            Usd::parse("0.5").unwrap(),
            Usd::parse("0.0001").unwrap(),
            true,
            1.5,
        )
        .unwrap();
        let oracle = Usd::from_f64(0.00002).unwrap();
        let c = r.cost(1000, 10, oracle).unwrap();
        // 1000·0.042e-6 + 10·0.5e-6 + 0.0001 = 0.000042 + 0.000005 + 0.0001
        assert_eq!(c.local.to_string(), "0.000147");
        assert_eq!(c.oracle_billed.to_string(), "0.00003");
        assert_eq!(c.total.to_string(), "0.000177");
        let free = Rates::default();
        let c = free.cost(5000, 77, oracle).unwrap();
        assert!(c.local.is_zero());
        assert_eq!(c.total, oracle);
        let no_pass = Rates {
            oracle_passthrough: false,
            ..free
        };
        assert!(no_pass.cost(1, 1, oracle).unwrap().total.is_zero());
    }

    #[test]
    fn value_text_rules() {
        assert_eq!(value_text(&Value::from("a b")), "a b");
        assert_eq!(value_text(&Value::Null), "");
        assert_eq!(
            value_text(&serde_json::json!({"b":1,"a":"x"})),
            r#"{"a":"x","b":1}"#
        );
        assert_eq!(value_text(&serde_json::json!(["x", 2])), r#"["x",2]"#);
        assert_eq!(criterion_text(&Value::from("a b")), r#""a b""#);
        assert_eq!(criterion_text(&Value::Null), "null");
    }
}
