//! The Jinja environment chat templates render in — transformers'
//! semantics, not minijinja's defaults.
//!
//! `transformers` renders `chat_template` in a jinja2 sandbox with
//! `trim_blocks` + `lstrip_blocks`, loop controls, and a handful of
//! helpers it injects. The one that matters most is `tojson`: it is NOT
//! jinja2's built-in (which HTML-escapes), but
//!
//! ```python
//! def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
//!     return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
//!                       separators=separators, sort_keys=sort_keys)
//! ```
//!
//! minijinja's own `tojson` differs on every axis a tool prompt touches:
//! compact separators (`{"a":1}` against `{"a": 1}`), `<`/`>`/`&`/`'`
//! escaped as `\u003c`…, no `ensure_ascii`/`separators`/`sort_keys`
//! keywords (MiniCPM5's `tojson(ensure_ascii=False)` was a hard render
//! error), and — without minijinja's `preserve_order` — map keys sorted.
//! Every tool declaration therefore reached the model in a shape it was
//! never trained on, or not at all. [`py_json_dumps`] reproduces
//! `json.dumps` byte for byte, including float `repr` and the
//! `ensure_ascii` surrogate-pair escapes.

use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};

/// Options of Python's `json.dumps` that transformers' `tojson` exposes.
#[derive(Debug, Clone, Default)]
pub struct DumpsOptions {
    pub ensure_ascii: bool,
    /// `None` = single line. `Some(s)` = newline + `s` per level (Python
    /// turns an int `n` into `n` spaces; a string is used verbatim).
    pub indent: Option<String>,
    /// `(item_separator, key_separator)`; `None` = Python's default,
    /// which depends on `indent`: `(", ", ": ")` single-line,
    /// `(",", ": ")` indented.
    pub separators: Option<(String, String)>,
    pub sort_keys: bool,
}

/// Build the environment every chat template renders in.
pub(crate) fn environment<'a>() -> Environment<'a> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // HF templates use python string/dict methods (.startswith, .get …).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson_filter);
    // `{{ x }}` and `x | string` print the way Python's `str()` does:
    // `True`/`False`/`None`, float `repr`, and containers as Python
    // literals (`['a', 1]`, `{'k': True}`). minijinja's own spelling
    // (`true`, `none`, `["a", 1]`) reached tool-call HISTORY: MiniCPM5
    // renders a non-string argument with `{{ param_value }}`, Qwen3.5 /
    // Qwen3-coder with `args_value | string`, and a boolean argument came
    // back as a token sequence the model never saw in training. Strings,
    // integers and every other kind keep minijinja's default path.
    env.set_formatter(|out, state, value| {
        if needs_py_str(value) {
            write!(out, "{}", py_str(value))
                .map_err(|_| Error::new(ErrorKind::WriteFailure, "formatter write failed"))
        } else {
            minijinja::escape_formatter(out, state, value)
        }
    });
    env.add_filter("string", |value: Value| -> Value {
        if value.kind() == ValueKind::String {
            value
        } else if needs_py_str(&value) {
            Value::from(py_str(&value))
        } else {
            Value::from(value.to_string())
        }
    });
    // `raise_exception` is another transformers helper: templates call it
    // to reject a malformed conversation. Unregistered, the render still
    // fails, but with "unknown function" instead of the template's reason.
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    // `visible_text` is a helper transformers injects into its template
    // env (it flattens multimodal content to its text). Nanbeige's
    // template calls it unconditionally in the tools branch; without it
    // the render errors and the fallback quietly serves a TOOLLESS prompt.
    env.add_function("visible_text", |v: Value| -> String {
        if let Some(s) = v.as_str() {
            return s.to_string();
        }
        if let Ok(iter) = v.try_iter() {
            let mut out = Vec::new();
            for item in iter {
                if let Some(s) = item.as_str() {
                    out.push(s.to_string());
                } else if let Ok(t) = item.get_attr("text") {
                    if let Some(s) = t.as_str() {
                        out.push(s.to_string());
                    }
                }
            }
            return out.join("\n");
        }
        String::new()
    });
    env
}

/// `x | tojson(ensure_ascii=False, indent=None, separators=None, sort_keys=False)`.
///
/// Positional arguments bind in that order, exactly as in the Python
/// signature — so `tojson(2)` sets `ensure_ascii`, not an indent, the
/// same as it does under transformers.
fn tojson_filter(value: &Value, args: Rest<Value>) -> Result<Value, Error> {
    // minijinja passes keyword arguments as a trailing kwargs value.
    let mut positional: Vec<Value> = args.0;
    let kwargs = match positional.last() {
        Some(last) if last.is_kwargs() => Kwargs::try_from(positional.pop().unwrap())?,
        _ => Kwargs::from_iter(std::iter::empty::<(String, Value)>()),
    };
    if positional.len() > 4 {
        return Err(Error::new(
            ErrorKind::TooManyArguments,
            "tojson() takes at most 4 positional arguments",
        ));
    }
    let mut positional = positional.into_iter();
    let ensure_ascii = positional.next();
    let indent = positional.next();
    let separators = positional.next();
    let sort_keys = positional.next();
    let pick = |pos: Option<Value>, name: &str| -> Result<Option<Value>, Error> {
        let kw: Option<Value> = kwargs.get(name)?;
        match (pos, kw) {
            (Some(_), Some(_)) => Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson() got multiple values for argument '{name}'"),
            )),
            (p, k) => Ok(p.or(k).filter(|v| !v.is_none() && !v.is_undefined())),
        }
    };
    let ensure_ascii = pick(ensure_ascii, "ensure_ascii")?;
    let indent = pick(indent, "indent")?;
    let separators = pick(separators, "separators")?;
    let sort_keys = pick(sort_keys, "sort_keys")?;
    kwargs.assert_all_used()?;

    let indent = match indent {
        None => None,
        Some(v) if v.kind() == ValueKind::String => Some(v.as_str().unwrap_or("").to_string()),
        Some(v) if v.kind() == ValueKind::Bool => Some(" ".repeat(v.is_true() as usize)),
        Some(v) => {
            let n = v.as_i64().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson(): indent must be an int or a string, got {v}"),
                )
            })?;
            Some(" ".repeat(n.max(0) as usize))
        }
    };
    let separators = match separators {
        None => None,
        Some(v) => {
            let items: Vec<Value> = v.try_iter().map(|it| it.collect()).map_err(|_| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson(): separators must be a (item, key) pair",
                )
            })?;
            match items.as_slice() {
                [a, b] if a.as_str().is_some() && b.as_str().is_some() => Some((
                    a.as_str().unwrap().to_string(),
                    b.as_str().unwrap().to_string(),
                )),
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson(): separators must be a pair of strings",
                    ));
                }
            }
        }
    };
    let opts = DumpsOptions {
        ensure_ascii: ensure_ascii.is_some_and(|v| v.is_true()),
        indent,
        separators,
        sort_keys: sort_keys.is_some_and(|v| v.is_true()),
    };
    // Not HTML-escaped (transformers returns a plain str) and marked safe
    // so an autoescaping template would not escape it again either.
    py_json_dumps(value, &opts).map(Value::from_safe_string)
}

/// Kinds whose Python `str()` differs from minijinja's `Display`.
fn needs_py_str(v: &Value) -> bool {
    match v.kind() {
        ValueKind::None | ValueKind::Bool | ValueKind::Seq | ValueKind::Map => true,
        ValueKind::Number => !v.is_integer(),
        _ => false,
    }
}

/// Python `str(x)` for template values (strings verbatim, the rest `repr`).
pub fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::String => v.as_str().unwrap_or("").to_string(),
        ValueKind::Undefined => String::new(),
        _ => py_repr(v),
    }
}

/// Python `repr(x)` for template values.
pub fn py_repr(v: &Value) -> String {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => "None".into(),
        ValueKind::Bool => (if v.is_true() { "True" } else { "False" }).into(),
        ValueKind::Number if v.is_integer() => v.to_string(),
        ValueKind::Number => {
            let f = f64::try_from(v.clone()).unwrap_or(f64::NAN);
            if f.is_nan() {
                "nan".into()
            } else if f.is_infinite() {
                (if f > 0.0 { "inf" } else { "-inf" }).into()
            } else {
                py_float_repr(f)
            }
        }
        ValueKind::String => py_str_repr(v.as_str().unwrap_or("")),
        ValueKind::Seq => {
            let items: Vec<String> = v
                .try_iter()
                .map(|it| it.map(|x| py_repr(&x)).collect())
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let mut items = Vec::new();
            if let Ok(keys) = v.try_iter() {
                for k in keys {
                    let item = v.get_item(&k).unwrap_or(Value::UNDEFINED);
                    items.push(format!("{}: {}", py_repr(&k), py_repr(&item)));
                }
            }
            format!("{{{}}}", items.join(", "))
        }
        _ => v.to_string(),
    }
}

/// Python `repr(str)`: single quotes unless the text holds a `'` and no
/// `"`; backslash escapes for that quote, the backslash, newline, CR,
/// tab and other control characters.
fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `json.dumps(value, **opts)` over a template value.
pub fn py_json_dumps(value: &Value, opts: &DumpsOptions) -> Result<String, Error> {
    let (item_sep, key_sep) = match &opts.separators {
        Some((i, k)) => (i.as_str(), k.as_str()),
        None if opts.indent.is_some() => (",", ": "),
        None => (", ", ": "),
    };
    let mut out = String::new();
    let mut d = Dumper {
        opts,
        item_sep,
        key_sep,
        out: &mut out,
    };
    d.value(value, 0)?;
    Ok(out)
}

struct Dumper<'o> {
    opts: &'o DumpsOptions,
    item_sep: &'o str,
    key_sep: &'o str,
    out: &'o mut String,
}

impl Dumper<'_> {
    fn newline(&mut self, level: usize) {
        if let Some(ind) = &self.opts.indent {
            self.out.push('\n');
            for _ in 0..level {
                self.out.push_str(ind);
            }
        }
    }

    fn value(&mut self, v: &Value, level: usize) -> Result<(), Error> {
        match v.kind() {
            // json.dumps(None) → null. An undefined value cannot reach
            // Python's json.dumps at all (jinja2 raises); null is the
            // forgiving spelling rather than a failed render.
            ValueKind::Undefined | ValueKind::None => self.out.push_str("null"),
            ValueKind::Bool => self
                .out
                .push_str(if v.is_true() { "true" } else { "false" }),
            ValueKind::Number => {
                if v.is_integer() {
                    self.out.push_str(&v.to_string());
                } else {
                    let f = f64::try_from(v.clone()).map_err(|_| {
                        Error::new(ErrorKind::InvalidOperation, "tojson(): bad number")
                    })?;
                    self.out.push_str(&py_float_repr(f));
                }
            }
            ValueKind::String => self.string(v.as_str().unwrap_or("")),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = v.try_iter()?.collect();
                if items.is_empty() {
                    self.out.push_str("[]");
                    return Ok(());
                }
                self.out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(self.item_sep);
                    }
                    self.newline(level + 1);
                    self.value(item, level + 1)?;
                }
                self.newline(level);
                self.out.push(']');
            }
            ValueKind::Map | ValueKind::Plain => {
                let mut entries: Vec<(String, Value)> = Vec::new();
                for key in v.try_iter()? {
                    let item = v.get_item(&key)?;
                    entries.push((self.key_text(&key)?, item));
                }
                if self.opts.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                if entries.is_empty() {
                    self.out.push_str("{}");
                    return Ok(());
                }
                self.out.push('{');
                for (i, (k, item)) in entries.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(self.item_sep);
                    }
                    self.newline(level + 1);
                    self.string(k);
                    self.out.push_str(self.key_sep);
                    self.value(item, level + 1)?;
                }
                self.newline(level);
                self.out.push('}');
            }
            other => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson(): object of type {other} is not JSON serializable"),
                ));
            }
        }
        Ok(())
    }

    /// Python coerces non-string keys: int → "1", float → repr, bool →
    /// "true", None → "null".
    fn key_text(&self, k: &Value) -> Result<String, Error> {
        Ok(match k.kind() {
            ValueKind::String => k.as_str().unwrap_or("").to_string(),
            ValueKind::None | ValueKind::Undefined => "null".into(),
            ValueKind::Bool => (if k.is_true() { "true" } else { "false" }).into(),
            ValueKind::Number if k.is_integer() => k.to_string(),
            ValueKind::Number => py_float_repr(f64::try_from(k.clone()).unwrap_or(f64::NAN)),
            other => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson(): keys must be str, int, float, bool or None, not {other}"),
                ));
            }
        })
    }

    fn string(&mut self, s: &str) {
        self.out.push('"');
        for c in s.chars() {
            match c {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                '\u{08}' => self.out.push_str("\\b"),
                '\u{0c}' => self.out.push_str("\\f"),
                c if (c as u32) < 0x20 => {
                    self.out.push_str(&format!("\\u{:04x}", c as u32));
                }
                // ensure_ascii escapes everything outside printable ASCII
                // (space..~), DEL included; astral chars as a UTF-16
                // surrogate pair, lower-case hex — Python's spelling.
                c if self.opts.ensure_ascii && !(' '..='~').contains(&c) => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        self.out.push_str(&format!("\\u{:04x}", unit));
                    }
                }
                c => self.out.push(c),
            }
        }
        self.out.push('"');
    }
}

/// Python's `float.__repr__`: the shortest round-trip digits, fixed
/// notation for decimal exponents in (-4, 16], otherwise `d.ddde±XX`
/// with at least two exponent digits; `json.dumps` spells the
/// non-finite values `NaN` / `Infinity` / `-Infinity`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` is the shortest round-trip representation too.
    let e = format!("{:e}", f.abs());
    let (mant, exp) = e.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent");
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1; // value = 0.DIGITS × 10^decpt
    let mut s = String::new();
    if f < 0.0 {
        s.push('-');
    }
    if -4 < decpt && decpt <= 16 {
        let n = digits.len() as i32;
        if decpt <= 0 {
            s.push_str("0.");
            for _ in 0..(-decpt) {
                s.push('0');
            }
            s.push_str(&digits);
        } else if decpt >= n {
            s.push_str(&digits);
            for _ in 0..(decpt - n) {
                s.push('0');
            }
            s.push_str(".0");
        } else {
            s.push_str(&digits[..decpt as usize]);
            s.push('.');
            s.push_str(&digits[decpt as usize..]);
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if exp < 0 { '-' } else { '+' });
        s.push_str(&format!("{:02}", exp.abs()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(tpl: &str, ctx: serde_json::Value) -> Result<String, Error> {
        let mut env = environment();
        env.add_template("t", tpl)?;
        env.get_template("t")?.render(Value::from_serialize(&ctx))
    }

    fn tool() -> serde_json::Value {
        // Deliberately NOT alphabetical: insertion order must survive.
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Погода <city> & \"quotes\" 🌧",
                "parameters": {
                    "type": "object",
                    "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        })
    }

    /// Default call: Python's `", "` / `": "` separators, insertion
    /// order, non-ASCII verbatim, and NO HTML escaping of `<`, `&`, `'`.
    /// Expected strings are `json.dumps(tool)` from CPython 3.12.
    #[test]
    fn default_matches_python_json_dumps() {
        let got = render("{{ t | tojson }}", serde_json::json!({"t": tool()})).unwrap();
        assert_eq!(
            got,
            r#"{"type": "function", "function": {"name": "get_weather", "description": "Погода <city> & \"quotes\" 🌧", "parameters": {"type": "object", "properties": {"unit": {"type": "string"}, "city": {"type": "string"}}, "required": ["city"]}}}"#
        );
        // MiniCPM5 spells the default out — it was a hard error before.
        let explicit = render(
            "{{ t | tojson(ensure_ascii=False) }}",
            serde_json::json!({"t": tool()}),
        )
        .unwrap();
        assert_eq!(explicit, got);
    }

    #[test]
    fn ensure_ascii_escapes_like_python() {
        let got = render(
            "{{ s | tojson(ensure_ascii=True) }}",
            serde_json::json!({"s": "é\u{7f}🌧\u{1}\t'"}),
        )
        .unwrap();
        // json.dumps("é\x7f🌧\x01\t'", ensure_ascii=True)
        assert_eq!(got, r#""\u00e9\u007f\ud83c\udf27\u0001\t'""#);
    }

    #[test]
    fn sort_keys_indent_and_separators() {
        let v = serde_json::json!({"b": [1, 2], "a": {}, "c": []});
        let ctx = serde_json::json!({"v": v});
        assert_eq!(
            render("{{ v | tojson(sort_keys=True) }}", ctx.clone()).unwrap(),
            r#"{"a": {}, "b": [1, 2], "c": []}"#
        );
        // json.dumps(v, indent=2): item separator loses its space.
        assert_eq!(
            render("{{ v | tojson(indent=2) }}", ctx.clone()).unwrap(),
            "{\n  \"b\": [\n    1,\n    2\n  ],\n  \"a\": {},\n  \"c\": []\n}"
        );
        assert_eq!(
            render(
                "{{ v | tojson(indent='\\t', sort_keys=true) }}",
                ctx.clone()
            )
            .unwrap(),
            "{\n\t\"a\": {},\n\t\"b\": [\n\t\t1,\n\t\t2\n\t],\n\t\"c\": []\n}"
        );
        assert_eq!(
            render("{{ v | tojson(separators=(',', ':')) }}", ctx.clone()).unwrap(),
            r#"{"b":[1,2],"a":{},"c":[]}"#
        );
        // indent=0: newlines, no indentation.
        assert_eq!(
            render("{{ [1] | tojson(indent=0) }}", ctx.clone()).unwrap(),
            "[\n1\n]"
        );
        // Positional arguments follow the Python signature: the first
        // one is ensure_ascii, not an indent.
        assert_eq!(
            render("{{ 'é' | tojson(true) }}", ctx.clone()).unwrap(),
            r#""\u00e9""#
        );
    }

    #[test]
    fn unknown_keyword_is_an_error() {
        assert!(render("{{ 1 | tojson(bogus=1) }}", serde_json::json!({})).is_err());
    }

    #[test]
    fn scalars_follow_python() {
        let got = render(
            "{{ v | tojson }}",
            serde_json::json!({"v": [1, -7, 1.5, 18.0, 1e16, 1.0e-5, 0.0001, 123456789012345678u64, true, null]}),
        )
        .unwrap();
        // json.dumps([1, -7, 1.5, 18.0, 1e16, 1e-05, 0.0001, 123456789012345678, True, None])
        assert_eq!(
            got,
            "[1, -7, 1.5, 18.0, 1e+16, 1e-05, 0.0001, 123456789012345678, true, null]"
        );
    }

    #[test]
    fn float_repr_table() {
        for (f, want) in [
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (-2.5e-7, "-2.5e-07"),
            (1e100, "1e+100"),
            (-0.0, "-0.0"),
            (f64::INFINITY, "Infinity"),
        ] {
            assert_eq!(py_float_repr(f), want, "{f:e}");
        }
    }

    /// Template-level iteration keeps request order too (Python dicts
    /// are insertion-ordered; minijinja sorted them before
    /// `preserve_order`).
    #[test]
    fn dict_items_keep_insertion_order() {
        let got = render(
            "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}",
            serde_json::json!({"d": {"zeta": 1, "alpha": 2, "mid": 3}}),
        )
        .unwrap();
        assert_eq!(got, "zeta=1;alpha=2;mid=3;");
    }

    /// `{{ x }}` / `x | string` print Python's `str()`; the expected
    /// strings are CPython's output for the same values.
    #[test]
    fn printing_follows_python_str() {
        let ctx = serde_json::json!({
            "b": true, "n": null, "f": 1.5e-7, "i": 42, "s": "plain",
            "l": ["a", "it's", 2, false, null], "d": {"k": true, "z": [1.0]}
        });
        let got = render(
            "{{ b }}|{{ n }}|{{ f }}|{{ i }}|{{ s }}|{{ l }}|{{ d }}|{{ b | string }}|{{ 'x' ~ i }}",
            ctx,
        )
        .unwrap();
        assert_eq!(
            got,
            r#"True|None|1.5e-07|42|plain|['a', "it's", 2, False, None]|{'k': True, 'z': [1.0]}|True|x42"#
        );
        // repr("a\nb'\"\\") == 'a\nb\'"\\'
        assert_eq!(py_str_repr("a\nb'\"\\"), r#"'a\nb\'"\\'"#);
    }

    #[test]
    fn raise_exception_carries_the_template_message() {
        let err = render(
            "{{ raise_exception('roles must alternate') }}",
            serde_json::json!({}),
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("roles must alternate"),
            "{err:#}"
        );
    }
}
