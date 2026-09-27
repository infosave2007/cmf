//! Canonical JSON: sorted keys, `,:` separators, UTF-8 without escapes (spec §2.3).
//!
//! The bytes equal Python's
//! `json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`
//! for every value a [`serde_json::Value`] can hold:
//!
//! * object keys are sorted by code point (Python `sorted` on `str`), which is the
//!   byte order of their UTF-8 (Rust `str` ordering); arrays keep their order;
//! * strings escape exactly what Python escapes with `ensure_ascii=False`: `"`,
//!   `\`, the short forms `\b \f \n \r \t` and every other code point below
//!   U+0020 as `\u00xx` (lowercase hex); everything else, including U+007F,
//!   U+2028/U+2029 and astral characters, is written as raw UTF-8;
//! * integers (`u64`/`i64`) are written in decimal; a JSON `1.0` stays a float;
//! * floats are written as Python's `float.__repr__`: the shortest digits that
//!   round-trip, correctly rounded with ties to even, in positional notation when
//!   the decimal exponent `decpt` (value `= 0.d₁d₂… × 10^decpt`) satisfies
//!   `-4 < decpt ≤ 16`, else as `d[.ddd]e±XX` with at least two exponent digits;
//!   an integral positional value gets `.0`.
//!
//! Plain shortest formatting (Rust's `{}`/`{:e}`, ryu) resolves an exact tie in the
//! last digit upwards where Python's dtoa rounds to even (`1997107851181081.25`
//! prints as `…081.3` instead of `…081.2`); [`format_f64`] therefore re-rounds the
//! shortest digit count with correct rounding. Checked against CPython 3.9 `repr`
//! on 5,000,607 doubles (random bit patterns, decimal magnitudes 1e-330..1e308,
//! f32 values and exact binary fractions): 0 differences.
//!
//! **Parsing.** `serde_json` without its `float_roundtrip` feature (which this
//! crate does not turn on for the whole workspace) reads some 17-digit decimals one
//! ulp off: `3.4028234663852886e+38` parses to a neighbour of `f32::MAX as f64`, so
//! canonical text would not survive a `serde_json` round trip. [`parse`] is a
//! strict RFC 8259 parser whose numbers go through Rust's correctly rounded
//! `str::parse::<f64>`; every manifest is read with it. Like Python it reads `-0`
//! as the integer 0 and keeps the last of duplicate keys.
//!
//! Limits (not expressible in a `Value` without `arbitrary_precision`): NaN and
//! infinities (Python writes `NaN`/`Infinity`; the drivers use `allow_nan=False`;
//! [`parse`] refuses them) and integers outside `i64`/`u64` (read as floats,
//! where Python keeps an exact integer).

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Canonical text of a JSON value.
pub fn to_string(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

/// Canonical UTF-8 bytes of a JSON value.
pub fn to_vec(v: &Value) -> Vec<u8> {
    to_string(v).into_bytes()
}

/// Canonical text of any serialisable value (through [`serde_json::to_value`]).
pub fn of<T: Serialize + ?Sized>(t: &T) -> anyhow::Result<String> {
    Ok(to_string(&serde_json::to_value(t)?))
}

/// Canonical bytes of any serialisable value.
pub fn vec_of<T: Serialize + ?Sized>(t: &T) -> anyhow::Result<Vec<u8>> {
    Ok(of(t)?.into_bytes())
}

/// Lowercase hex sha256 of the canonical bytes of `v`.
pub fn sha256_hex(v: &Value) -> String {
    sha256_bytes_hex(to_string(v).as_bytes())
}

/// Lowercase hex sha256 of raw bytes.
pub fn sha256_bytes_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Parse `bytes` as JSON and require that they are already canonical (the parse
/// re-serialised equals the input byte for byte). A duplicate object key, a
/// space, an escaped non-ASCII character, an unsorted key or a non-shortest
/// number all fail this check.
pub fn parse_canonical(bytes: &[u8]) -> anyhow::Result<Value> {
    let v = parse(bytes)?;
    let again = to_string(&v);
    if again.as_bytes() != bytes {
        let at = again
            .as_bytes()
            .iter()
            .zip(bytes)
            .position(|(a, b)| a != b)
            .unwrap_or(again.len().min(bytes.len()));
        anyhow::bail!("JSON is not canonical (first difference at byte {at})");
    }
    Ok(v)
}

/// Maximum nesting depth [`parse`] accepts.
pub const MAX_DEPTH: usize = 128;

/// Parse JSON text (RFC 8259, UTF-8) into a [`Value`] with correctly rounded
/// floats (see the module notes). Integers that fit `u64`/`i64` stay integers,
/// `-0` is the integer 0, a duplicate key keeps its last value, trailing
/// non-whitespace is an error. An error reads `not valid JSON: <what> at byte
/// N` (or `not UTF-8 JSON: …`), for a caller to name the input before it.
pub fn parse(bytes: &[u8]) -> anyhow::Result<Value> {
    let text = std::str::from_utf8(bytes).map_err(|e| anyhow::anyhow!("not UTF-8 JSON: {e}"))?;
    let mut p = Parser {
        s: text.as_bytes(),
        text,
        i: 0,
    };
    p.ws();
    let v = p
        .value(0)
        .map_err(|e| anyhow::anyhow!("not valid JSON: {e} at byte {}", p.i))?;
    p.ws();
    if p.i != p.s.len() {
        anyhow::bail!("not valid JSON: trailing characters at byte {}", p.i);
    }
    Ok(v)
}

/// [`parse`] of a `str`.
pub fn parse_str(text: &str) -> anyhow::Result<Value> {
    parse(text.as_bytes())
}

struct Parser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(&c) = self.s.get(self.i) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, lit: &str) -> Result<(), String> {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!("expected '{lit}'"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("nesting deeper than 128".into());
        }
        match self.s.get(self.i) {
            None => Err("unexpected end".into()),
            Some(b'n') => self.eat("null").map(|()| Value::Null),
            Some(b't') => self.eat("true").map(|()| Value::Bool(true)),
            Some(b'f') => self.eat("false").map(|()| Value::Bool(false)),
            Some(b'"') => self.string().map(Value::String),
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Array(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Array(a));
                        }
                        _ => return Err("expected ',' or ']'".into()),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut m = serde_json::Map::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Object(m));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return Err("expected a string key".into());
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return Err("expected ':'".into());
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    m.insert(k, v);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Object(m));
                        }
                        _ => return Err("expected ',' or '}'".into()),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err("unexpected character".into()),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| "truncated \\u escape".to_string())?;
        let h = std::str::from_utf8(h).map_err(|_| "bad \\u escape".to_string())?;
        let v = u32::from_str_radix(h, 16).map_err(|_| "bad \\u escape".to_string())?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("bad \\u escape".into());
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&c) = self.s.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            // `start..i` holds no quote, backslash or control byte, so it ends on
            // a character boundary of the (valid UTF-8) text.
            out.push_str(&self.text[start..self.i]);
            match self.s.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let e = *self.s.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xd800..0xdc00).contains(&hi) {
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err("lone high surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err("bad low surrogate".into());
                                }
                                0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                            } else if (0xdc00..0xe000).contains(&hi) {
                                return Err("lone low surrogate".into());
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or("bad code point")?);
                        }
                        _ => return Err("bad escape".into()),
                    }
                }
                Some(_) => return Err("control character in string".into()),
            }
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        let neg = self.s[self.i] == b'-';
        if neg {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let d0 = p.i;
            while p.s.get(p.i).is_some_and(u8::is_ascii_digit) {
                p.i += 1;
            }
            p.i - d0
        };
        let int_start = self.i;
        let n_int = digits(self);
        if n_int == 0 {
            return Err("expected digits".into());
        }
        if n_int > 1 && self.s[int_start] == b'0' {
            return Err("leading zero".into());
        }
        let mut float = false;
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return Err("expected fraction digits".into());
            }
            float = true;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err("expected exponent digits".into());
            }
            float = true;
        }
        let t = &self.text[start..self.i];
        if !float {
            if !neg {
                if let Ok(u) = t.parse::<u64>() {
                    return Ok(Value::from(u));
                }
            } else if let Ok(i) = t.parse::<i64>() {
                return Ok(Value::from(i)); // `-0` is the integer 0, as in Python
            }
        }
        let x: f64 = t.parse().map_err(|_| "bad number".to_string())?;
        if !x.is_finite() {
            return Err("number out of range".into());
        }
        Ok(f64_value(x))
    }
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else {
                // Without `arbitrary_precision` every other Number is a finite f64.
                out.push_str(&format_f64(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_string(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (k, x)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, x);
            }
            out.push('}');
        }
    }
}

/// Append a JSON string literal as Python writes it with `ensure_ascii=False`.
pub fn write_string(out: &mut String, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let b = c as u32 as usize;
                out.push_str("\\u00");
                out.push(HEX[b >> 4] as char);
                out.push(HEX[b & 15] as char);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)` of `x` (see the module notes). NaN and the infinities are
/// written as Python writes them (`NaN`, `Infinity`, `-Infinity`), although a
/// canonical document never contains them.
pub fn format_f64(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let a = x.abs();
    // Shortest round-trip digit count, then the correctly rounded (ties to even)
    // decimal of that many significant digits: Python's dtoa mode 0 result.
    let shortest = format!("{a:e}");
    let n = shortest
        .split_once('e')
        .map(|(m, _)| m.bytes().filter(u8::is_ascii_digit).count())
        .unwrap_or(1)
        .max(1);
    let exact = format!("{:.*e}", n - 1, a);
    let s = if exact.parse::<f64>().ok() == Some(a) {
        exact
    } else {
        shortest
    };
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let nd = digits.len() as i32;
    let decpt = exp + 1;
    let mut out = String::with_capacity(nd as usize + 8);
    if x < 0.0 {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if nd > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.unsigned_abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= nd {
        out.push_str(&digits);
        for _ in 0..decpt - nd {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// The JSON number of an f64, as a `Value` (non-finite values become `null`).
pub fn f64_value(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    #[allow(clippy::excessive_precision)] // exact binary ties, written out on purpose
    fn python_repr_of_floats() {
        for (x, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e100, "1e+100"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (123456789012345680000.0, "1.2345678901234568e+20"),
            (1997107851181081.25, "1997107851181081.2"),
            (0.125, "0.125"),
            (0.8f32 as f64, "0.800000011920929"),
            (0.024557100608944893, "0.024557100608944893"),
            (-111275153569243.125, "-111275153569243.12"),
        ] {
            assert_eq!(format_f64(x), want, "{x:e}");
        }
    }

    #[test]
    fn sorted_keys_and_escapes() {
        let v = json!({"b": [2, 1], "a": {"z": null, "y": "q\"\\\n\u{1}\u{7f}é😀"}, "\u{e000}": 1, "😀": 2});
        assert_eq!(
            to_string(&v),
            "{\"a\":{\"y\":\"q\\\"\\\\\\n\\u0001\u{7f}é😀\",\"z\":null},\"b\":[2,1],\"\u{e000}\":1,\"😀\":2}"
        );
        assert!(parse_canonical(to_string(&v).as_bytes()).is_ok());
        assert!(parse_canonical(b"{\"b\":1,\"a\":2}").is_err());
        assert!(parse_canonical(b"{\"a\":1,\"a\":1}").is_err());
        assert!(parse_canonical(b"{\"a\": 1}").is_err());
        assert!(parse_canonical(b"{\"a\":\"\\u00e9\"}").is_err());
    }

    #[test]
    fn parser_is_exact_and_strict() {
        // serde_json (no float_roundtrip) reads this one ulp off; `parse` does not.
        let v = parse(b"[3.4028234663852886e+38, -0, 1e400]");
        assert!(v.is_err(), "1e400 is out of range");
        let v = parse(b"[3.4028234663852886e+38, -0, -1, 18446744073709551615, 1.0]").unwrap();
        assert_eq!(v[0].as_f64().unwrap(), f32::MAX as f64);
        assert_eq!(
            to_string(&v),
            "[3.4028234663852886e+38,0,-1,18446744073709551615,1.0]"
        );
        let v = parse(br#"{"a":"\u00e9\ud83d\ude00\n","a":2}"#).unwrap();
        assert_eq!(to_string(&v), "{\"a\":2}");
        let v = parse(br#""\ud83d\ude00\u00e9\/""#).unwrap();
        assert_eq!(v, Value::String("😀é/".into()));
        for bad in [
            &b"01"[..],
            b"1.",
            b"-",
            b".5",
            b"[1,]",
            b"{\"a\" 1}",
            b"\"\\ud83d\"",
            b"\"\\udc00\"",
            b"\"a\nb\"",
            b"nul",
            b"[1] x",
            b"NaN",
            b"\"\\x\"",
        ] {
            assert!(parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        let deep = "[".repeat(200) + &"]".repeat(200);
        assert!(parse(deep.as_bytes()).is_err());
    }
}
