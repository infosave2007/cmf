//! Tool calls out of generated text — every grammar the embedded
//! templates teach, normalised to the OpenAI `tool_calls` shape.
//!
//! Two families:
//!
//! * **Hermes / Qwen**: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`,
//!   and Nanbeige's XML body inside the same wrapper
//!   (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`).
//! * **MiniCPM5**: `<function name="NAME"><param name="K">V</param>…</function>`,
//!   several calls in a row, values optionally wrapped in `<![CDATA[…]]>`.
//!   Parsed like SGLang's `minicpm5` tool-call parser
//!   (`python/sglang/srt/function_call/minicpm5_detector.py`): values are
//!   stripped; a value whose declared JSON-schema type is not `string` is
//!   parsed as JSON, then as a Python literal (`True`, `None`,
//!   `['a', 'b']` — the spelling the template itself renders history in),
//!   and kept as a string when neither parses; a block that names a tool
//!   the request did not declare, uses an undeclared or duplicated
//!   parameter, or misses a required one stays TEXT instead of becoming a
//!   call a client would execute. One deliberate difference: a type list
//!   that includes `"string"` (`["string", "null"]`) keeps the raw text —
//!   SGLang would turn `"75001"` into a number there.
//!
//! Calls are only looked for OUTSIDE the reasoning block: a model that
//! drafts `<function …>` while thinking has not called anything yet (the
//! same split vLLM/SGLang make when a reasoning parser is active).
//!
//! `arguments` is always a STRING of JSON, per the OpenAI contract.

use serde_json::{Map, Value};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Extract tool calls. Returns the text outside the calls (trimmed) and
/// the calls in OpenAI shape. `tools` is the request's declared tool list
/// (needed for the MiniCPM5 grammar's schema-typed values and name
/// validation; the Hermes grammar carries JSON and needs none).
pub fn extract_tool_calls(text: &str, tools: Option<&[Value]>) -> (String, Vec<Value>) {
    let (start, end) = content_region(text);
    let (plain_mid, mut calls) = extract_hermes(&text[start..end]);
    let (plain_mid, minicpm) = extract_minicpm5(&plain_mid, tools.unwrap_or(&[]));
    calls.extend(minicpm);
    let mut plain = String::with_capacity(text.len());
    plain.push_str(&text[..start]);
    plain.push_str(&plain_mid);
    plain.push_str(&text[end..]);
    (plain.trim().to_string(), calls)
}

/// Byte range of `text` that is answer rather than reasoning: after the
/// first `</think>` when there is one (the opening tag may have been in
/// the prompt), before an unterminated `<think>` otherwise.
fn content_region(text: &str) -> (usize, usize) {
    if let Some(p) = text.find(THINK_CLOSE) {
        return (p + THINK_CLOSE.len(), text.len());
    }
    if let Some(p) = text.find(THINK_OPEN) {
        return (0, p);
    }
    (0, text.len())
}

fn openai_call(name: &str, args: &Value) -> Value {
    serde_json::json!({
        "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
        "type": "function",
        "function": {
            "name": name,
            "arguments": serde_json::to_string(args).unwrap_or_else(|_| "{}".into()),
        }
    })
}

// ─── Hermes / Qwen ───────────────────────────────────────────────────

/// `<tool_call>{...}</tool_call>` blocks. A block whose body does not
/// parse stays in the text verbatim — a client can read prose, but it
/// cannot execute garbage; an unterminated block (truncated output) too.
fn extract_hermes(text: &str) -> (String, Vec<Value>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut rest = text;
    let mut plain = String::new();
    let mut calls = Vec::new();
    while let Some(i) = rest.find(OPEN) {
        let Some(j) = rest[i + OPEN.len()..].find(CLOSE) else {
            break;
        };
        let body = rest[i + OPEN.len()..i + OPEN.len() + j].trim();
        let block_end = i + OPEN.len() + j + CLOSE.len();
        // Two trained grammars share the wrapper: the JSON object, and
        // Nanbeige's XML `<function=name><parameter=k>v...`.
        let parsed = serde_json::from_str::<Value>(body)
            .ok()
            .or_else(|| parse_nanbeige_function(body));
        match parsed {
            Some(v) if v.get("name").map(|n| n.is_string()) == Some(true) => {
                plain.push_str(&rest[..i]);
                let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                calls.push(openai_call(v["name"].as_str().unwrap(), &args));
            }
            _ => plain.push_str(&rest[..block_end]),
        }
        rest = &rest[block_end..];
    }
    plain.push_str(rest);
    (plain, calls)
}

/// Nanbeige's XML tool grammar, normalised to the JSON shape:
/// `<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>`.
/// Parameter values keep inner newlines; the single newline the grammar
/// puts around a value is trimmed.
fn parse_nanbeige_function(body: &str) -> Option<Value> {
    let t = body.trim();
    let name_start = t.find("<function=")? + "<function=".len();
    let name_end = t[name_start..].find(['>', '\n'])? + name_start;
    let name = t[name_start..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = Map::new();
    let mut rest = &t[name_end..];
    while let Some(ps) = rest.find("<parameter=") {
        let key_start = ps + "<parameter=".len();
        let key_end = rest[key_start..].find('>')? + key_start;
        let key = rest[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let val_end = rest[val_start..].find("</parameter>")? + val_start;
        let raw = &rest[val_start..val_end];
        let val = raw.strip_prefix('\n').unwrap_or(raw);
        let val = val.strip_suffix('\n').unwrap_or(val);
        args.insert(key, Value::String(val.to_string()));
        rest = &rest[val_end + "</parameter>".len()..];
    }
    Some(serde_json::json!({"name": name, "arguments": args}))
}

// ─── MiniCPM5 ────────────────────────────────────────────────────────

const FN_OPEN: &str = "<function";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<param";
const PARAM_CLOSE: &str = "</param>";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

/// Position of the next `<function` that opens a tag (followed by
/// whitespace) — `<functions>` or `<function=` (Nanbeige) are not ours.
fn find_fn_open(s: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(p) = s[from..].find(FN_OPEN) {
        let at = from + p;
        let next = s[at + FN_OPEN.len()..].chars().next();
        if next.is_some_and(|c| c.is_whitespace()) {
            return Some(at);
        }
        from = at + FN_OPEN.len();
    }
    None
}

fn extract_minicpm5(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    let mut plain = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(i) = find_fn_open(rest) {
        let Some(len) = minicpm5_block_len(&rest[i..]) else {
            break; // unterminated: the rest stays text
        };
        let block = &rest[i..i + len];
        match parse_minicpm5_block(block, tools) {
            Some((name, args)) => {
                plain.push_str(&rest[..i]);
                calls.push(openai_call(&name, &Value::Object(args)));
            }
            None => plain.push_str(&rest[..i + len]),
        }
        rest = &rest[i + len..];
    }
    plain.push_str(rest);
    (plain, calls)
}

/// Length of the `<function …>…</function>` block at the start of `s`.
/// CDATA sections are skipped as opaque, so a value that contains
/// `</function>` (code, markup) does not end the call early.
fn minicpm5_block_len(s: &str) -> Option<usize> {
    let mut pos = FN_OPEN.len();
    loop {
        let close = s[pos..].find(FN_CLOSE).map(|p| p + pos);
        let cdata = s[pos..].find(CDATA_OPEN).map(|p| p + pos);
        match (close, cdata) {
            (Some(c), Some(d)) if d < c => {
                let end = s[d + CDATA_OPEN.len()..].find(CDATA_CLOSE)?;
                pos = d + CDATA_OPEN.len() + end + CDATA_CLOSE.len();
            }
            (Some(c), _) => return Some(c + FN_CLOSE.len()),
            (None, _) => return None,
        }
    }
}

/// `name="…"` (or single quotes) inside a tag's attribute text.
fn attr_name(attrs: &str) -> Option<String> {
    let mut from = 0;
    while let Some(p) = attrs[from..].find("name") {
        let at = from + p;
        let boundary_ok = at == 0 || attrs[..at].ends_with(char::is_whitespace);
        let after = attrs[at + 4..].trim_start();
        if boundary_ok {
            if let Some(after_eq) = after.strip_prefix('=') {
                let after_eq = after_eq.trim_start();
                let quote = after_eq.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let body = &after_eq[1..];
                    let end = body.find(quote)?;
                    return Some(body[..end].trim().to_string());
                }
                return None;
            }
        }
        from = at + 4;
    }
    None
}

/// One `<function name="…">…</function>` block → (name, arguments), or
/// None when it is not a call the request allows (see module docs).
fn parse_minicpm5_block(block: &str, tools: &[Value]) -> Option<(String, Map<String, Value>)> {
    let open_end = block.find('>')?;
    let name = attr_name(&block[FN_OPEN.len()..open_end])?;
    let tool = tools
        .iter()
        .find(|t| t["function"]["name"].as_str() == Some(name.as_str()))?;
    let schema = &tool["function"]["parameters"];
    let props = schema_properties(schema);

    let body = &block[open_end + 1..block.len() - FN_CLOSE.len()];
    let mut args = Map::new();
    let mut rest = body;
    while let Some(p) = rest.find(PARAM_OPEN) {
        let after = &rest[p + PARAM_OPEN.len()..];
        // `<param>` or `<param …>` only — `<parameters>` is not a param.
        if !after.starts_with(|c: char| c == '>' || c.is_whitespace()) {
            rest = after;
            continue;
        }
        let tag_end = after.find('>')?;
        let key = attr_name(&after[..tag_end])?; // a param without a name: invalid
        let value_src = &after[tag_end + 1..];
        let (raw, consumed) = param_value(value_src)?;
        if !props.is_empty() && !props.contains_key(&key) {
            return None; // not a parameter of this tool
        }
        if args.contains_key(&key) {
            return None; // duplicated
        }
        let declared = props.get(&key).and_then(|s| s.get("type"));
        args.insert(key, typed_value(raw, declared));
        rest = &value_src[consumed..];
    }
    let required = schema.get("required").and_then(|r| r.as_array());
    if let Some(req) = required {
        if req
            .iter()
            .filter_map(|r| r.as_str())
            .any(|r| !args.contains_key(r))
        {
            return None;
        }
    }
    Some((name, args))
}

/// Parameter value at the start of `s` (just after `<param …>`), and how
/// many bytes through the closing `</param>` it took.
fn param_value(s: &str) -> Option<(String, usize)> {
    let lead = s.len() - s.trim_start().len();
    let t = &s[lead..];
    if let Some(inner) = t.strip_prefix(CDATA_OPEN) {
        let end = inner.find(CDATA_CLOSE)?;
        let after = &inner[end + CDATA_CLOSE.len()..];
        let gap = after.len() - after.trim_start().len();
        if after[gap..].starts_with(PARAM_CLOSE) {
            let consumed =
                lead + CDATA_OPEN.len() + end + CDATA_CLOSE.len() + gap + PARAM_CLOSE.len();
            return Some((inner[..end].trim().to_string(), consumed));
        }
        // Text after the CDATA section: fall through to the plain form.
    }
    let end = s.find(PARAM_CLOSE)?;
    Some((xml_unescape(s[..end].trim()), end + PARAM_CLOSE.len()))
}

/// XML character references, when the value is well-formed XML text (as
/// SGLang's XML parse would see it); otherwise the raw text (its regex
/// fallback). A raw `<` or a bare `&` means "not XML" — kept verbatim.
fn xml_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    if s.contains('<') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        let tail = &rest[p..];
        let Some(semi) = tail.find(';') else {
            return s.to_string();
        };
        let ent = &tail[1..semi];
        let ch = match ent {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                u32::from_str_radix(&ent[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
            }
            _ if ent.starts_with('#') => ent[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => out.push(c),
            None => return s.to_string(),
        }
        rest = &tail[semi + 1..];
    }
    out.push_str(rest);
    out
}

/// Top-level `properties` of a parameters schema, descending into
/// `anyOf`/`oneOf`/`allOf` when the top level declares none (SGLang's
/// `get_schema_properties`).
fn schema_properties(schema: &Value) -> Map<String, Value> {
    if let Some(p) = schema.get("properties").and_then(|p| p.as_object()) {
        return p.clone();
    }
    let mut merged = Map::new();
    for kw in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = schema.get(kw).and_then(|b| b.as_array()) {
            for b in branches {
                for (k, v) in schema_properties(b) {
                    merged.entry(k).or_insert(v);
                }
            }
        }
    }
    merged
}

/// A value as the declared type wants it (see module docs).
fn typed_value(raw: String, declared: Option<&Value>) -> Value {
    let is_string = match declared {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t.as_str() == Some("string")),
        _ => false,
    };
    if is_string {
        return Value::String(raw);
    }
    if let Ok(v) = serde_json::from_str::<Value>(&raw) {
        return v;
    }
    py_literal(&raw).unwrap_or(Value::String(raw))
}

/// Python literal (`ast.literal_eval` subset): None/True/False, numbers,
/// single- or double-quoted strings, lists/tuples and dicts of those.
fn py_literal(s: &str) -> Option<Value> {
    let mut p = PyLit {
        s: s.as_bytes(),
        src: s,
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    (p.i == p.s.len()).then_some(v)
}

struct PyLit<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl PyLit<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.ws();
        match *self.s.get(self.i)? {
            b'[' | b'(' => {
                let close = if self.s[self.i] == b'[' { b']' } else { b')' };
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    if self.eat(close) {
                        return Some(Value::Array(items));
                    }
                    items.push(self.value()?);
                    if !self.eat(b',') {
                        return self.eat(close).then_some(Value::Array(items));
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut map = Map::new();
                loop {
                    if self.eat(b'}') {
                        return Some(Value::Object(map));
                    }
                    let key = match self.value()? {
                        Value::String(k) => k,
                        Value::Number(n) => n.to_string(),
                        _ => return None,
                    };
                    if !self.eat(b':') {
                        return None;
                    }
                    let v = self.value()?;
                    map.insert(key, v);
                    if !self.eat(b',') {
                        return self.eat(b'}').then_some(Value::Object(map));
                    }
                }
            }
            q @ (b'\'' | b'"') => {
                self.i += 1;
                let mut out = String::new();
                let start = self.i;
                let mut seg = start;
                while self.i < self.s.len() {
                    let c = self.s[self.i];
                    if c == q {
                        out.push_str(&self.src[seg..self.i]);
                        self.i += 1;
                        return Some(Value::String(out));
                    }
                    if c == b'\\' && self.i + 1 < self.s.len() {
                        out.push_str(&self.src[seg..self.i]);
                        let e = self.s[self.i + 1];
                        out.push(match e {
                            b'n' => '\n',
                            b't' => '\t',
                            b'r' => '\r',
                            b'\\' => '\\',
                            b'\'' => '\'',
                            b'"' => '"',
                            _ => return None,
                        });
                        self.i += 2;
                        seg = self.i;
                        continue;
                    }
                    self.i += 1;
                }
                None
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len()
                    && !matches!(self.s[self.i], b',' | b']' | b')' | b'}' | b':')
                    && !self.s[self.i].is_ascii_whitespace()
                {
                    self.i += 1;
                }
                match &self.src[start..self.i] {
                    "None" => Some(Value::Null),
                    "True" => Some(Value::Bool(true)),
                    "False" => Some(Value::Bool(false)),
                    num => serde_json::from_str::<serde_json::Number>(num)
                        .ok()
                        .map(Value::Number),
                }
            }
        }
    }
}

// ─── Streaming holdback ──────────────────────────────────────────────

/// Streams content while keeping tool-call markup out of it.
///
/// Once a call marker appears outside a reasoning block, everything from
/// it on is HELD: it is parsed whole at the end ([`Self::finish`]) and
/// shipped as a `tool_calls` delta, with whatever was not a call flushed
/// as content — nothing is dropped, even when the block turns out not to
/// be a valid call. Until a marker is certain, the last few bytes stay
/// buffered so one split across tokens cannot leak.
#[derive(Default)]
pub struct ToolHoldback {
    tail: String,
    held: Option<String>,
    in_think: bool,
}

const MARKERS: &[&str] = &["<tool_call>", "<function ", "<function\n", "<function\t"];

impl ToolHoldback {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one decoded token; returns the text that may go out now.
    pub fn push(&mut self, token: &str) -> String {
        if let Some(h) = &mut self.held {
            h.push_str(token);
            return String::new();
        }
        self.tail.push_str(token);
        let mut out = String::new();
        loop {
            let think = if self.in_think {
                self.tail.find(THINK_CLOSE).map(|p| (p, THINK_CLOSE.len()))
            } else {
                self.tail.find(THINK_OPEN).map(|p| (p, THINK_OPEN.len()))
            };
            let marker = if self.in_think {
                None
            } else {
                MARKERS.iter().filter_map(|m| self.tail.find(m)).min()
            };
            match (think, marker) {
                (Some((tp, tl)), m) if m.is_none_or(|mp| tp < mp) => {
                    out.extend(self.tail.drain(..tp + tl));
                    self.in_think = !self.in_think;
                }
                (_, Some(mp)) => {
                    out.extend(self.tail.drain(..mp));
                    self.held = Some(std::mem::take(&mut self.tail));
                    return out;
                }
                _ => break,
            }
        }
        // Keep a marker's worth of tail (minus one byte) unsent.
        let keep = MARKERS
            .iter()
            .chain([THINK_OPEN, THINK_CLOSE].iter())
            .map(|m| m.len())
            .max()
            .unwrap()
            - 1;
        if self.tail.len() > keep {
            let mut cut = self.tail.len() - keep;
            while !self.tail.is_char_boundary(cut) {
                cut -= 1;
            }
            out.extend(self.tail.drain(..cut));
        }
        out
    }

    /// End of generation: (content still to send, tool calls).
    pub fn finish(self, tools: Option<&[Value]>) -> (String, Vec<Value>) {
        match self.held {
            Some(held) => {
                let (plain, calls) = extract_tool_calls(&held, tools);
                if calls.is_empty() {
                    (held, calls) // not a call after all: all of it is content
                } else {
                    (plain, calls)
                }
            }
            None => (self.tail, Vec::new()),
        }
    }

    /// True once a call marker has been seen (content is being held).
    pub fn holding(&self) -> bool {
        self.held.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(call: &Value) -> Value {
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
    }

    fn weather_tools() -> Vec<Value> {
        vec![
            serde_json::json!({"type": "function", "function": {"name": "get_weather",
                "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                   "metric": {"type": "boolean"}, "zip": {"type": ["string", "null"]},
                                   "hours": {"type": "array"}, "opts": {"type": "object"}},
                    "required": ["city"]}}}),
            serde_json::json!({"type": "function", "function": {"name": "run_code",
                "parameters": {"type": "object",
                    "properties": {"code": {"type": "string"}}, "required": ["code"]}}}),
        ]
    }

    // ── Hermes (unchanged behaviour) ──

    #[test]
    fn hermes_single() {
        let (text, calls) = extract_tool_calls(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            None,
        );
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(args(&calls[0])["city"], "Paris");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    }

    #[test]
    fn hermes_text_and_multiple() {
        let (text, calls) = extract_tool_calls(
            "Let me check both.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>",
            None,
        );
        assert_eq!(text, "Let me check both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn hermes_malformed_and_unterminated_stay_text() {
        let (text, calls) =
            extract_tool_calls("<tool_call>\nnot json at all\n</tool_call> done", None);
        assert!(calls.is_empty());
        assert!(text.contains("not json at all"));
        let (text, calls) = extract_tool_calls("<tool_call>\n{\"name\": \"a\"", None);
        assert!(calls.is_empty());
        assert!(text.contains("<tool_call>"));
    }

    #[test]
    fn nanbeige_xml_inside_tool_call() {
        let (_, calls) = extract_tool_calls(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n</parameter>\n</function>\n</tool_call>",
            None,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(args(&calls[0])["city"], "New\nYork");
    }

    // ── MiniCPM5 ──

    #[test]
    fn minicpm5_single_call() {
        let tools = weather_tools();
        let (text, calls) = extract_tool_calls(
            "<function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
            Some(&tools),
        );
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            calls[0]["function"]["arguments"].as_str().unwrap(),
            r#"{"city":"Paris"}"#
        );
    }

    #[test]
    fn minicpm5_several_calls_text_and_reasoning() {
        let tools = weather_tools();
        let out = "<think>\nI could call <function name=\"get_weather\"><param name=\"city\">Rome</param></function> here.\n</think>\n\nChecking both.\n<function name=\"get_weather\"><param name=\"city\">Paris</param></function>\n<function name='get_weather'><param name='city'>Oslo</param><param name=\"days\">3</param></function>";
        let (text, calls) = extract_tool_calls(out, Some(&tools));
        assert_eq!(
            calls.len(),
            2,
            "the call drafted inside <think> is not a call"
        );
        assert_eq!(args(&calls[0]), serde_json::json!({"city": "Paris"}));
        assert_eq!(
            args(&calls[1]),
            serde_json::json!({"city": "Oslo", "days": 3})
        );
        assert!(text.starts_with("<think>") && text.contains("Rome"));
        assert!(text.ends_with("Checking both."), "{text}");
    }

    #[test]
    fn minicpm5_cdata_and_multiline_values() {
        let tools = weather_tools();
        let code = "fn main() {\n    if a < b && c { println!(\"</function>\"); }\n}";
        let out = format!(
            "<function name=\"run_code\"><param name=\"code\"><![CDATA[\n{code}\n]]></param></function>"
        );
        let (text, calls) = extract_tool_calls(&out, Some(&tools));
        assert_eq!(text, "");
        assert_eq!(
            calls.len(),
            1,
            "a </function> inside CDATA must not end the call"
        );
        assert_eq!(args(&calls[0])["code"], code);
        // Multi-line without CDATA, and XML entities in a well-formed value.
        let (_, calls) = extract_tool_calls(
            "<function name=\"get_weather\"><param name=\"city\">\nSt. John&apos;s &amp; Co\n</param></function>",
            Some(&tools),
        );
        assert_eq!(args(&calls[0])["city"], "St. John's & Co");
        // A raw & is not XML: kept verbatim.
        let (_, calls) = extract_tool_calls(
            "<function name=\"get_weather\"><param name=\"city\">A & B</param></function>",
            Some(&tools),
        );
        assert_eq!(args(&calls[0])["city"], "A & B");
    }

    #[test]
    fn minicpm5_values_follow_the_schema_type() {
        let tools = weather_tools();
        let (_, calls) = extract_tool_calls(
            "<function name=\"get_weather\"><param name=\"city\">1984</param><param name=\"days\"> 5 </param><param name=\"metric\">True</param><param name=\"zip\">75001</param><param name=\"hours\">['09:00', '12:00']</param><param name=\"opts\">{\"units\": \"si\", \"alerts\": false}</param></function>",
            Some(&tools),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({
                "city": "1984",            // string type: never a number
                "days": 5,                 // integer: JSON number
                "metric": true,            // Python literal True
                "zip": "75001",            // ["string","null"]: stays text
                "hours": ["09:00", "12:00"],
                "opts": {"units": "si", "alerts": false}
            })
        );
        // Order of the arguments string follows the call, not the schema.
        assert!(
            calls[0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .starts_with("{\"city\":\"1984\",\"days\":5")
        );
        // A non-string value that parses as neither stays a string.
        let (_, calls) = extract_tool_calls(
            "<function name=\"get_weather\"><param name=\"city\">X</param><param name=\"days\">three</param></function>",
            Some(&tools),
        );
        assert_eq!(args(&calls[0])["days"], "three");
    }

    #[test]
    fn minicpm5_invalid_blocks_stay_text() {
        let tools = weather_tools();
        for bad in [
            // undeclared tool
            "<function name=\"launch_rockets\"><param name=\"city\">X</param></function>",
            // undeclared parameter
            "<function name=\"get_weather\"><param name=\"city\">X</param><param name=\"color\">red</param></function>",
            // duplicated parameter
            "<function name=\"get_weather\"><param name=\"city\">X</param><param name=\"city\">Y</param></function>",
            // required parameter missing
            "<function name=\"get_weather\"><param name=\"days\">2</param></function>",
            // param without a name
            "<function name=\"get_weather\"><param>X</param></function>",
            // function without a name
            "<function id=\"x\"><param name=\"city\">X</param></function>",
        ] {
            let (text, calls) = extract_tool_calls(bad, Some(&tools));
            assert!(calls.is_empty(), "{bad}");
            assert_eq!(text, bad, "an invalid block is kept verbatim");
        }
        // Without declared tools nothing in this grammar is a call.
        let ok = "<function name=\"get_weather\"><param name=\"city\">X</param></function>";
        assert!(extract_tool_calls(ok, None).1.is_empty());
        // Unterminated (truncated output) stays text.
        let cut = "Sure. <function name=\"get_weather\"><param name=\"city\">Par";
        let (text, calls) = extract_tool_calls(cut, Some(&tools));
        assert!(calls.is_empty());
        assert_eq!(text, cut);
        // Prose that merely contains the word is not touched.
        let prose = "A <functional> programming note: <function=x> and <functions>.";
        assert_eq!(extract_tool_calls(prose, Some(&tools)).0, prose);
    }

    #[test]
    fn py_literals() {
        assert_eq!(py_literal("None"), Some(Value::Null));
        assert_eq!(
            py_literal("{'a': [1, 2.5, 'x\\'y'], \"b\": (True, False)}"),
            Some(serde_json::json!({"a": [1, 2.5, "x'y"], "b": [true, false]}))
        );
        assert_eq!(py_literal("[1, 2"), None);
        assert_eq!(py_literal("hello"), None);
    }

    // ── Streaming holdback ──

    fn stream(pieces: &[&str], tools: &[Value]) -> (String, Vec<Value>) {
        let mut h = ToolHoldback::new();
        let mut content = String::new();
        for p in pieces {
            content.push_str(&h.push(p));
        }
        let (rest, calls) = h.finish(Some(tools));
        content.push_str(&rest);
        (content, calls)
    }

    #[test]
    fn holdback_keeps_calls_out_of_content_even_split_across_tokens() {
        let tools = weather_tools();
        let (content, calls) = stream(
            &[
                "Let me ",
                "check.\n<fun",
                "ction",
                " name=\"get_weather\"><param name=\"city\">",
                "Paris</param></function>",
            ],
            &tools,
        );
        assert_eq!(content, "Let me check.\n");
        assert_eq!(calls.len(), 1);
        let (content, calls) = stream(
            &[
                "ok <tool",
                "_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
            ],
            &tools,
        );
        assert_eq!(content, "ok ");
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn holdback_ignores_markup_in_reasoning_and_never_drops_text() {
        let tools = weather_tools();
        let (content, calls) = stream(
            &[
                "<think>maybe <function name=\"get_weather\">",
                "…</function></think>",
                "\n\nNo call needed.",
            ],
            &tools,
        );
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "<think>maybe <function name=\"get_weather\">…</function></think>\n\nNo call needed."
        );
        // A held block that turns out not to be a call is flushed whole.
        let (content, calls) = stream(
            &["See <function name=\"nope\">", "x</function> end"],
            &tools,
        );
        assert!(calls.is_empty());
        assert_eq!(content, "See <function name=\"nope\">x</function> end");
        // Plain text passes through intact, multi-byte chars included.
        let (content, calls) = stream(&["Привет, ", "мир — ", "tools?"], &tools);
        assert!(calls.is_empty());
        assert_eq!(content, "Привет, мир — tools?");
    }
}
