//! A TOML 1.0 reader for the `[[api_keys]]` of a cortiq-router configuration
//! (spec decision-v4 §4.15, `cortiq decision keys import --format
//! router-toml`).
//!
//! No TOML crate is in the workspace and the import must not add one, so this
//! reader parses the TOML grammar into JSON values: tables, arrays of tables,
//! dotted keys, inline tables, arrays, the four string forms with their
//! escapes, integers (decimal, `0x`, `0o`, `0b`, `_` separators), floats and
//! booleans. Datetimes and `inf`/`nan` become strings: the import never reads
//! them, they only have to be skipped. A whole router configuration therefore
//! reads, not just its key section.
//!
//! Errors name the line only, never its text: a configuration line may hold a
//! raw API key.

use anyhow::{Result, bail};
use serde_json::{Map, Value};

/// Deepest nesting of arrays and inline tables accepted.
const MAX_DEPTH: usize = 64;

/// Parse a TOML document into its root table.
pub fn parse(text: &str) -> Result<Map<String, Value>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    Parser {
        s: text.as_bytes(),
        i: 0,
    }
    .document()
}

type Table = Map<String, Value>;

/// The table at `path` below `t`, created when missing; an array on the path
/// stands for its last table (an array of tables).
fn table_at<'m>(mut t: &'m mut Table, path: &[String]) -> Option<&'m mut Table> {
    for k in path {
        t = match t
            .entry(k.clone())
            .or_insert_with(|| Value::Object(Map::new()))
        {
            Value::Object(m) => m,
            Value::Array(a) => match a.last_mut() {
                Some(Value::Object(m)) => m,
                _ => return None,
            },
            _ => return None,
        };
    }
    Some(t)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn line(&self) -> usize {
        1 + self.s[..self.i.min(self.s.len())]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
    }

    fn err<T>(&self, msg: &str) -> Result<T> {
        bail!("TOML line {}: {msg}", self.line())
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn peek_at(&self, k: usize) -> Option<u8> {
        self.s.get(self.i + k).copied()
    }

    fn starts(&self, lit: &[u8]) -> bool {
        self.s[self.i..].starts_with(lit)
    }

    /// Spaces and tabs.
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t')) {
            self.i += 1;
        }
    }

    fn comment(&mut self) {
        if self.peek() == Some(b'#') {
            while let Some(c) = self.peek() {
                if c == b'\n' {
                    break;
                }
                self.i += 1;
            }
        }
    }

    fn newline(&mut self) -> bool {
        match self.peek() {
            Some(b'\n') => {
                self.i += 1;
                true
            }
            Some(b'\r') if self.peek_at(1) == Some(b'\n') => {
                self.i += 2;
                true
            }
            _ => false,
        }
    }

    /// Whitespace, comments and newlines.
    fn ws_nl(&mut self) {
        loop {
            self.ws();
            self.comment();
            if !self.newline() {
                break;
            }
        }
    }

    fn end_of_line(&mut self) -> Result<()> {
        self.ws();
        self.comment();
        if self.peek().is_none() || self.newline() {
            Ok(())
        } else {
            self.err("expected the end of the line")
        }
    }

    fn document(&mut self) -> Result<Table> {
        let mut root = Table::new();
        let mut current: Vec<String> = Vec::new();
        loop {
            self.ws_nl();
            match self.peek() {
                None => return Ok(root),
                Some(b'[') => {
                    let array = self.peek_at(1) == Some(b'[');
                    self.i += if array { 2 } else { 1 };
                    self.ws();
                    let path = self.key()?;
                    self.ws();
                    let close: &[u8] = if array { b"]]" } else { b"]" };
                    if !self.starts(close) {
                        return self.err("unterminated table header");
                    }
                    self.i += close.len();
                    if array {
                        let (last, parent) = path.split_last().expect("a key has a part");
                        let Some(t) = table_at(&mut root, parent) else {
                            return self.err("a table header goes through a value");
                        };
                        match t
                            .entry(last.clone())
                            .or_insert_with(|| Value::Array(Vec::new()))
                        {
                            Value::Array(a) => a.push(Value::Object(Map::new())),
                            _ => return self.err("an array of tables redefines a key"),
                        }
                    } else if table_at(&mut root, &path).is_none() {
                        return self.err("a table header goes through a value");
                    }
                    self.end_of_line()?;
                    current = path;
                }
                Some(_) => {
                    let path = self.key()?;
                    self.ws();
                    if self.peek() != Some(b'=') {
                        return self.err("expected '=' after a key");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(0)?;
                    let mut full = current.clone();
                    full.extend(path);
                    self.insert(&mut root, &full, v)?;
                    self.end_of_line()?;
                }
            }
        }
    }

    fn insert(&self, t: &mut Table, path: &[String], v: Value) -> Result<()> {
        let (last, parent) = path.split_last().expect("a key has a part");
        let Some(t) = table_at(t, parent) else {
            return self.err("a dotted key goes through a value");
        };
        if t.contains_key(last) {
            return self.err("duplicate key");
        }
        t.insert(last.clone(), v);
        Ok(())
    }

    /// A dotted key.
    fn key(&mut self) -> Result<Vec<String>> {
        let mut parts = vec![self.simple_key()?];
        loop {
            self.ws();
            if self.peek() != Some(b'.') {
                return Ok(parts);
            }
            self.i += 1;
            self.ws();
            parts.push(self.simple_key()?);
        }
    }

    fn simple_key(&mut self) -> Result<String> {
        match self.peek() {
            Some(b'"') if self.starts(b"\"\"\"") => self.err("a key cannot be a multi-line string"),
            Some(b'"') => self.basic_string(),
            Some(b'\'') if self.starts(b"'''") => self.err("a key cannot be a multi-line string"),
            Some(b'\'') => self.literal_string(),
            _ => {
                let start = self.i;
                while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
                {
                    self.i += 1;
                }
                if start == self.i {
                    return self.err("expected a key");
                }
                Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
            }
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return self.err("values nested too deeply");
        }
        match self.peek() {
            Some(b'"') if self.starts(b"\"\"\"") => self.ml_basic_string().map(Value::String),
            Some(b'"') => self.basic_string().map(Value::String),
            Some(b'\'') if self.starts(b"'''") => self.ml_literal_string().map(Value::String),
            Some(b'\'') => self.literal_string().map(Value::String),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.inline_table(depth),
            Some(_) if self.starts(b"true") && !self.word_char_at(4) => {
                self.i += 4;
                Ok(Value::Bool(true))
            }
            Some(_) if self.starts(b"false") && !self.word_char_at(5) => {
                self.i += 5;
                Ok(Value::Bool(false))
            }
            Some(_) => self.scalar(),
            None => self.err("expected a value"),
        }
    }

    fn word_char_at(&self, k: usize) -> bool {
        self.peek_at(k)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            self.ws_nl();
            if self.peek() == Some(b']') {
                self.i += 1;
                return Ok(Value::Array(out));
            }
            out.push(self.value(depth + 1)?);
            self.ws_nl();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(out));
                }
                _ => return self.err("expected ',' or ']' in an array"),
            }
        }
    }

    fn inline_table(&mut self, depth: usize) -> Result<Value> {
        self.i += 1;
        let mut t = Table::new();
        loop {
            self.ws_nl();
            if self.peek() == Some(b'}') {
                self.i += 1;
                return Ok(Value::Object(t));
            }
            let path = self.key()?;
            self.ws();
            if self.peek() != Some(b'=') {
                return self.err("expected '=' after a key");
            }
            self.i += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            self.insert(&mut t, &path, v)?;
            self.ws_nl();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(t));
                }
                _ => return self.err("expected ',' or '}' in an inline table"),
            }
        }
    }

    /// Integer, float, datetime (as a string) or `inf`/`nan` (as a string).
    fn scalar(&mut self) -> Result<Value> {
        let start = self.i;
        let token_char =
            |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'+' | b'-' | b'.' | b':');
        while self.peek().is_some_and(token_char) {
            self.i += 1;
        }
        // "1979-05-27 07:32:00": a date, a space, a time.
        let is_date = |t: &[u8]| {
            t.len() == 10
                && t[..4].iter().all(u8::is_ascii_digit)
                && t[4] == b'-'
                && t[5..7].iter().all(u8::is_ascii_digit)
                && t[7] == b'-'
                && t[8..].iter().all(u8::is_ascii_digit)
        };
        if is_date(&self.s[start..self.i])
            && self.peek() == Some(b' ')
            && self.peek_at(1).is_some_and(|c| c.is_ascii_digit())
        {
            self.i += 1;
            while self.peek().is_some_and(token_char) {
                self.i += 1;
            }
        }
        let tok = std::str::from_utf8(&self.s[start..self.i]).unwrap_or_default();
        if tok.is_empty() {
            return self.err("expected a value");
        }
        let b = tok.as_bytes();
        let date_like = (b.len() >= 5 && b[..4].iter().all(u8::is_ascii_digit) && b[4] == b'-')
            || tok.contains(':');
        if date_like {
            return Ok(Value::String(tok.to_string()));
        }
        let unsigned = tok.trim_start_matches(['+', '-']);
        if matches!(unsigned, "inf" | "nan") {
            return Ok(Value::String(tok.to_string()));
        }
        if !underscores_ok(unsigned) {
            return self.err("invalid number");
        }
        let digits: String = tok.chars().filter(|&c| c != '_').collect();
        for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
            if let Some(rest) = digits.strip_prefix(prefix) {
                return match i64::from_str_radix(rest, radix) {
                    Ok(n) if !rest.is_empty() && !rest.starts_with(['+', '-']) => {
                        Ok(Value::from(n))
                    }
                    _ => self.err("invalid integer"),
                };
            }
        }
        if digits.contains(['.', 'e', 'E']) {
            return match digits.parse::<f64>() {
                Ok(f) if f.is_finite() && digits.bytes().any(|c| c.is_ascii_digit()) => {
                    Ok(serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number))
                }
                _ => self.err("invalid float"),
            };
        }
        let body = digits.trim_start_matches(['+', '-']);
        if body.is_empty()
            || !body.bytes().all(|c| c.is_ascii_digit())
            || (body.len() > 1 && body.starts_with('0'))
            || digits.len() - body.len() > 1
        {
            return self.err("invalid value");
        }
        match digits.parse::<i64>() {
            Ok(n) => Ok(Value::from(n)),
            Err(_) => self.err("integer out of range"),
        }
    }

    fn basic_string(&mut self) -> Result<String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None | Some(b'\n') | Some(b'\r') => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return self.utf8(out);
                }
                Some(b'\\') => self.escape(&mut out)?,
                Some(c) if (c < 0x20 && c != b'\t') || c == 0x7f => {
                    return self.err("control character in a string");
                }
                Some(c) => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }

    fn ml_basic_string(&mut self) -> Result<String> {
        self.i += 3;
        self.newline();
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return self.err("unterminated multi-line string"),
                Some(b'"') => {
                    let q = self.s[self.i..].iter().take_while(|&&c| c == b'"').count();
                    if q >= 3 {
                        if q > 5 {
                            return self.err("too many quotes at the end of a string");
                        }
                        out.extend(std::iter::repeat_n(b'"', q - 3));
                        self.i += q;
                        return self.utf8(out);
                    }
                    out.extend(std::iter::repeat_n(b'"', q));
                    self.i += q;
                }
                Some(b'\\') => {
                    // A line-ending backslash trims the whitespace and newlines after it.
                    let mut j = self.i + 1;
                    while matches!(self.s.get(j), Some(b' ' | b'\t')) {
                        j += 1;
                    }
                    if matches!(self.s.get(j), Some(b'\n' | b'\r')) {
                        self.i = j;
                        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                            self.i += 1;
                        }
                    } else {
                        self.escape(&mut out)?;
                    }
                }
                Some(b'\r') if self.peek_at(1) == Some(b'\n') => {
                    out.extend_from_slice(b"\r\n");
                    self.i += 2;
                }
                Some(c) if (c < 0x20 && !matches!(c, b'\t' | b'\n')) || c == 0x7f => {
                    return self.err("control character in a string");
                }
                Some(c) => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }

    fn literal_string(&mut self) -> Result<String> {
        self.i += 1;
        let start = self.i;
        loop {
            match self.peek() {
                None | Some(b'\n') | Some(b'\r') => return self.err("unterminated string"),
                Some(b'\'') => {
                    let v = self.s[start..self.i].to_vec();
                    self.i += 1;
                    return self.utf8(v);
                }
                Some(c) if (c < 0x20 && c != b'\t') || c == 0x7f => {
                    return self.err("control character in a string");
                }
                Some(_) => self.i += 1,
            }
        }
    }

    fn ml_literal_string(&mut self) -> Result<String> {
        self.i += 3;
        self.newline();
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return self.err("unterminated multi-line string"),
                Some(b'\'') => {
                    let q = self.s[self.i..].iter().take_while(|&&c| c == b'\'').count();
                    if q >= 3 {
                        if q > 5 {
                            return self.err("too many quotes at the end of a string");
                        }
                        out.extend(std::iter::repeat_n(b'\'', q - 3));
                        self.i += q;
                        return self.utf8(out);
                    }
                    out.extend(std::iter::repeat_n(b'\'', q));
                    self.i += q;
                }
                Some(c) if (c < 0x20 && !matches!(c, b'\t' | b'\n' | b'\r')) || c == 0x7f => {
                    return self.err("control character in a string");
                }
                Some(c) => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }

    /// One escape sequence at `\`.
    fn escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let c = self.peek_at(1);
        self.i += 2;
        let simple = match c {
            Some(b'b') => Some(0x08),
            Some(b't') => Some(b'\t'),
            Some(b'n') => Some(b'\n'),
            Some(b'f') => Some(0x0c),
            Some(b'r') => Some(b'\r'),
            Some(b'e') => Some(0x1b),
            Some(b'"') => Some(b'"'),
            Some(b'\\') => Some(b'\\'),
            _ => None,
        };
        if let Some(b) = simple {
            out.push(b);
            return Ok(());
        }
        let len = match c {
            Some(b'x') => 2,
            Some(b'u') => 4,
            Some(b'U') => 8,
            _ => return self.err("invalid escape in a string"),
        };
        let hex = self
            .s
            .get(self.i..self.i + len)
            .and_then(|h| std::str::from_utf8(h).ok())
            .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()));
        let ch = hex
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .and_then(char::from_u32);
        let Some(ch) = ch else {
            return self.err("invalid unicode escape in a string");
        };
        self.i += len;
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    fn utf8(&self, v: Vec<u8>) -> Result<String> {
        match String::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(_) => self.err("a string is not UTF-8"),
        }
    }
}

/// `_` only between two digits.
fn underscores_ok(t: &str) -> bool {
    let b = t.as_bytes();
    b.iter().enumerate().all(|(i, &c)| {
        c != b'_'
            || (i > 0
                && i + 1 < b.len()
                && b[i - 1].is_ascii_alphanumeric()
                && b[i + 1].is_ascii_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_router_configuration_reads_whole() {
        let text = r#"
# cortiq-router configuration
bind = "0.0.0.0:8080"   # listen
taxonomy_id = 'data-assistant'
database_url = ""
complexity_tiers = [
  { tier = "low", max = 0.33 },   # comment inside an array
  { tier = "medium", max = 0.66 },
  { tier = "high", max = 1.0 },
]
started = 1979-05-27 07:32:00Z
big = 1_000_000
hex = 0xDEAD_beef
neg = -17
flt = 6.02e23
nan = nan
notes = """
line one \
    continued "quoted" ""two""\u00e9"""
raw = '''C:\path\'''

[[api_keys]]
key            = "k-one"
account        = "acme-corp"
rate_per_min   = 600
decision_quota = 1_000_000

[[api_keys]]
key = 'k-two'   # literal
"account" = "beta"

[auth]
require = true
[auth.plans.starter]
rate_per_min = 60
[auth.plans."my plan"]
duration_days = 0

[task_complexity]
"data-analysis" = 0.8
code.base = 0.9
"#;
        let root = parse(text).unwrap();
        assert_eq!(
            root["api_keys"],
            json!([
                {"key": "k-one", "account": "acme-corp", "rate_per_min": 600, "decision_quota": 1_000_000},
                {"key": "k-two", "account": "beta"}
            ])
        );
        assert_eq!(
            root["complexity_tiers"][2],
            json!({"tier": "high", "max": 1.0})
        );
        assert_eq!(root["started"], "1979-05-27 07:32:00Z");
        assert_eq!(root["big"], 1_000_000);
        assert_eq!(root["hex"], 0xdead_beef_i64);
        assert_eq!(root["neg"], -17);
        assert_eq!(root["nan"], "nan");
        assert_eq!(
            root["notes"],
            "line one continued \"quoted\" \"\"two\"\"\u{e9}"
        );
        assert_eq!(root["raw"], "C:\\path\\");
        assert_eq!(root["auth"]["require"], true);
        assert_eq!(root["auth"]["plans"]["my plan"]["duration_days"], 0);
        assert_eq!(root["task_complexity"]["code"]["base"], 0.9);
    }

    #[test]
    fn inline_api_keys_and_crlf() {
        let root =
            parse("api_keys = [\r\n  {key = \"a\", account = \"x\"},\r\n  {key = \"b\"}\r\n]\r\n")
                .unwrap();
        assert_eq!(
            root["api_keys"],
            json!([{"key": "a", "account": "x"}, {"key": "b"}])
        );
    }

    #[test]
    fn errors_name_the_line_never_the_text() {
        for (text, line) in [
            ("a = 1\nkey = \"secret-raw-key", 2),
            ("a = 1\n\nkey = secret-raw-key", 3),
            ("[[api_keys]]\nkey = \"secret-raw-key\" x", 2),
            ("key = \"secret-raw-key\"\nkey = \"secret-raw-key\"", 2),
            ("[api_keys\nkey = \"secret-raw-key\"", 1),
            ("n = 1__0", 1),
            ("n = 012", 1),
        ] {
            let e = format!("{:#}", parse(text).unwrap_err());
            assert!(
                e.starts_with(&format!("TOML line {line}:")),
                "{text:?}: {e}"
            );
            assert!(!e.contains("secret"), "{e}");
        }
    }
}
