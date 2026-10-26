//! Tool calls out of generated text — every grammar the embedded
//! templates teach, normalised to the OpenAI `tool_calls` shape.
//!
//! Two families:
//!
//! * **Hermes / Qwen**: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`,
//!   Nanbeige's XML body inside the same wrapper
//!   (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`),
//!   and the key/value body of Spark-X2.5 (GLM lineage):
//!   `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`.
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

// MiniMax-M3 deliberately uses a separate, namespaced wire grammar.  Its
// namespace is ordinary decoded text (rather than a tokenizer special token),
// so it must be parsed before it reaches an OpenAI-compatible client.
const MINIMAX_M3_NS: &str = "]<]minimax[>[";
const MINIMAX_M3_THINK_OPEN: &str = "<mm:think>";
const MINIMAX_M3_THINK_CLOSE: &str = "</mm:think>";
const MINIMAX_M3_TOOL_OPEN: &str = "]<]minimax[>[<tool_call>";
const MINIMAX_M3_TOOL_CLOSE: &str = "]<]minimax[>[</tool_call>";
const MINIMAX_M3_INVOKE_OPEN: &str = "]<]minimax[>[<invoke name=\"";
const MINIMAX_M3_INVOKE_CLOSE: &str = "]<]minimax[>[</invoke>";

/// Parse a MiniMax M3 completion into OpenAI response fields.
///
/// M3 emits adaptive reasoning as `<mm:think>…</mm:think>` and prefixes all
/// tool XML tags with `]<]minimax[>[`.  It also prefixes a non-thinking turn
/// with one bare `</mm:think>` marker.  The source M3/SGLang parser removes
/// that leading closer; doing the same here keeps it out of user-visible
/// `content`.  A malformed or undeclared-tool block is deliberately retained
/// verbatim as content rather than turning partial model output into an action.
pub fn extract_minimax_m3_response(
    text: &str,
    tools: Option<&[Value]>,
) -> (String, Vec<Value>, Option<String>) {
    let (content, reasoning) = split_minimax_m3_reasoning(text);
    let (content, calls) = extract_minimax_m3_tool_calls(&content, tools.unwrap_or(&[]));
    (content, calls, reasoning)
}

/// Incremental guard for an M3 server-sent response.
///
/// The final tool-call conversion remains deliberately whole-response (so
/// nested XML can be validated against the supplied schema), but normal final
/// text can be released as it arrives.  The guard never releases a possible
/// prefix of the M3 reasoning or namespaced tool protocol; that is important
/// because generated tokens may split a marker at arbitrary byte boundaries.
/// Reasoning is kept private in streaming mode because the OpenAI SSE delta
/// shape used by this server has no `reasoning_content` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MiniMaxM3StreamPhase {
    Leading,
    Reasoning,
    Content,
    Tool,
}

pub struct MiniMaxM3Stream {
    phase: MiniMaxM3StreamPhase,
    pending: String,
    emitted: String,
    /// The strict parser only normalizes namespace-tag whitespace when at
    /// least one tool was declared, and any later malformed block makes that
    /// normalization transactional.  Keep public text from the first
    /// namespace onward until final parsing in that case.
    has_declared_tools: bool,
}

impl Default for MiniMaxM3Stream {
    fn default() -> Self {
        Self {
            phase: MiniMaxM3StreamPhase::Leading,
            pending: String::new(),
            emitted: String::new(),
            has_declared_tools: false,
        }
    }
}

impl MiniMaxM3Stream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a guard for a request with declared tools.  M3's final tool
    /// parser is transactional: one malformed block restores the original
    /// wire text, while a fully valid response normalizes namespace-adjacent
    /// whitespace.  Deferring text from the first namespace preserves an
    /// exact final prefix under both outcomes.
    pub fn with_declared_tools() -> Self {
        Self {
            has_declared_tools: true,
            ..Self::default()
        }
    }

    /// Consume decoded model text and return only final user-visible content
    /// that is safe to send now.  M3 thinking and tool markup are never
    /// returned here.
    pub fn push(&mut self, token: &str) -> String {
        if self.phase == MiniMaxM3StreamPhase::Tool {
            return String::new();
        }
        self.pending.push_str(token);
        let mut out = String::new();
        loop {
            match self.phase {
                MiniMaxM3StreamPhase::Leading => {
                    // M3's adaptive no-think form begins with a bare close.
                    // Wait for a full marker before deciding a marker prefix
                    // is ordinary text, so a split closer never leaks.
                    let leading = self.pending.trim_start();
                    if leading.is_empty() {
                        break;
                    }
                    if MINIMAX_M3_THINK_CLOSE.starts_with(leading)
                        && leading.len() < MINIMAX_M3_THINK_CLOSE.len()
                    {
                        break;
                    }
                    if let Some(after) = leading.strip_prefix(MINIMAX_M3_THINK_CLOSE) {
                        self.pending = after.to_string();
                        self.phase = MiniMaxM3StreamPhase::Content;
                        continue;
                    }
                    // `split_minimax_m3_reasoning` only recognizes an opener
                    // at byte zero (unlike the adaptive leading closer above),
                    // so keep that distinction here.  Otherwise a malformed
                    // ` " <mm:think>...` response would be hidden live and
                    // duplicated only at the final safety flush.
                    if MINIMAX_M3_THINK_OPEN.starts_with(&self.pending)
                        && self.pending.len() < MINIMAX_M3_THINK_OPEN.len()
                    {
                        break;
                    }
                    if let Some(after) = self.pending.strip_prefix(MINIMAX_M3_THINK_OPEN) {
                        self.pending = after.to_string();
                        self.phase = MiniMaxM3StreamPhase::Reasoning;
                        continue;
                    }
                    self.phase = MiniMaxM3StreamPhase::Content;
                }
                MiniMaxM3StreamPhase::Reasoning => {
                    if let Some(end) = self.pending.find(MINIMAX_M3_THINK_CLOSE) {
                        self.pending =
                            self.pending[end + MINIMAX_M3_THINK_CLOSE.len()..].to_string();
                        self.phase = MiniMaxM3StreamPhase::Content;
                        continue;
                    }
                    // Nothing in a still-open reasoning span is public.  Keep
                    // just a suffix that could become the close marker.
                    let keep = minimax_m3_marker_suffix(&self.pending, MINIMAX_M3_THINK_CLOSE);
                    self.pending = self.pending[self.pending.len() - keep..].to_string();
                    break;
                }
                MiniMaxM3StreamPhase::Content => {
                    // `extract_minimax_m3_tool_calls` first normalizes every
                    // namespace followed by XML-looking whitespace, but only
                    // when tools are declared.  If any later tool block is
                    // malformed it atomically restores the original wire
                    // text.  Do not release bytes whose final spelling cannot
                    // yet be known under that transaction.
                    if self.has_declared_tools {
                        if let Some(first_namespace) = self.pending.find(MINIMAX_M3_NS) {
                            if first_namespace > 0 {
                                out.push_str(&self.pending[..first_namespace]);
                                self.pending = self.pending[first_namespace..].to_string();
                                break;
                            }
                            if minimax_m3_tool_start(&self.pending) == Some(0) {
                                self.pending.clear();
                                self.phase = MiniMaxM3StreamPhase::Tool;
                            }
                            // A namespace that is not yet a full tool marker
                            // might still be normalized or restored by the
                            // strict final parser, so keep it private for now.
                            break;
                        }
                    }
                    if let Some(start) = minimax_m3_tool_start(&self.pending) {
                        out.push_str(&self.pending[..start]);
                        self.pending.clear();
                        self.phase = MiniMaxM3StreamPhase::Tool;
                        break;
                    }
                    let flush = minimax_m3_content_flush_len(&self.pending);
                    if flush == 0 {
                        break;
                    }
                    out.push_str(&self.pending[..flush]);
                    self.pending = self.pending[flush..].to_string();
                    break;
                }
                MiniMaxM3StreamPhase::Tool => break,
            }
        }
        self.emitted.push_str(&out);
        out
    }

    /// Finish with the authoritative whole-response parse result.  Usually it
    /// is exactly the portion not yet released.  The fallback retains output
    /// rather than silently dropping it if a future wire variation makes the
    /// incremental guard disagree with the strict parser.
    pub fn finish(&mut self, content: &str) -> String {
        let tail = content
            .strip_prefix(&self.emitted)
            .map(str::to_string)
            .unwrap_or_else(|| content.to_string());
        self.pending.clear();
        self.emitted.push_str(&tail);
        tail
    }
}

/// Number of trailing bytes that may be the beginning of `marker`.
fn minimax_m3_marker_suffix(text: &str, marker: &str) -> usize {
    for len in (1..text.len().min(marker.len().saturating_sub(1)) + 1).rev() {
        let start = text.len() - len;
        if text.is_char_boundary(start) && marker.starts_with(&text[start..]) {
            return len;
        }
    }
    0
}

/// Start byte of a complete M3 tool-call marker, accepting the whitespace
/// spelling the reference detector normalizes after the namespace token.
fn minimax_m3_tool_start(text: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(relative) = text[from..].find(MINIMAX_M3_NS) {
        let start = from + relative;
        let after_ns = &text[start + MINIMAX_M3_NS.len()..];
        let spaces = after_ns
            .bytes()
            .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
            .count();
        if after_ns[spaces..].starts_with("<tool_call>") {
            return Some(start);
        }
        from = start + MINIMAX_M3_NS.len();
    }
    None
}

/// Bytes safe to release while in the public-content phase.  The retained
/// tail is either a partial namespace token or a full namespace followed by
/// whitespace/a partial `<tool_call>` marker.
fn minimax_m3_content_flush_len(text: &str) -> usize {
    let mut keep_from = text.len();
    let mut from = 0;
    while let Some(relative) = text[from..].find(MINIMAX_M3_NS) {
        let start = from + relative;
        let after_ns = &text[start + MINIMAX_M3_NS.len()..];
        let spaces = after_ns
            .bytes()
            .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
            .count();
        let after_space = &after_ns[spaces..];
        if after_space.is_empty() || "<tool_call>".starts_with(after_space) {
            keep_from = keep_from.min(start);
        }
        from = start + MINIMAX_M3_NS.len();
    }
    let partial = minimax_m3_marker_suffix(text, MINIMAX_M3_NS);
    if partial > 0 {
        keep_from = keep_from.min(text.len() - partial);
    }
    keep_from
}

/// M3's reasoning detector has one format-specific wrinkle: an adaptive
/// non-thinking response starts with a closing marker inherited from the
/// prompt.  Only that *leading* close is stripped.  A truncated reasoning
/// block remains available as `reasoning_content`, never as user content.
fn split_minimax_m3_reasoning(text: &str) -> (String, Option<String>) {
    let leading = text.trim_start();
    if let Some(content) = leading.strip_prefix(MINIMAX_M3_THINK_CLOSE) {
        return (content.to_string(), None);
    }

    let mut rest = text;
    let mut opened = false;
    while let Some(after_open) = rest.strip_prefix(MINIMAX_M3_THINK_OPEN) {
        opened = true;
        rest = after_open;
    }
    if !opened {
        return (text.to_string(), None);
    }

    match rest.find(MINIMAX_M3_THINK_CLOSE) {
        Some(end) => (
            rest[end + MINIMAX_M3_THINK_CLOSE.len()..].to_string(),
            Some(rest[..end].to_string()),
        ),
        None => (String::new(), Some(rest.to_string())),
    }
}

/// The M3 template permits harmless whitespace between its namespace token
/// and the following XML tag.  SGLang normalizes exactly that spelling before
/// parsing; do it without touching whitespace anywhere else.
fn normalize_minimax_m3_tag_spacing(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(MINIMAX_M3_NS) {
        let after_ns = &rest[at + MINIMAX_M3_NS.len()..];
        let spaces = after_ns
            .bytes()
            .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
            .count();
        out.push_str(&rest[..at + MINIMAX_M3_NS.len()]);
        if spaces > 0 && after_ns[spaces..].starts_with('<') {
            rest = &after_ns[spaces..];
        } else {
            rest = after_ns;
        }
    }
    out.push_str(rest);
    out
}

/// Extract complete M3 tool blocks.  This is intentionally transactional:
/// if one M3 block cannot be fully and safely decoded, return the original
/// text and no actions.  Clients can then see the model output, while no
/// malformed XML can be mistaken for an executable call.
fn extract_minimax_m3_tool_calls(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    if tools.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let normalized = normalize_minimax_m3_tag_spacing(text);
    let mut plain = String::with_capacity(normalized.len());
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(relative_start) = normalized[cursor..].find(MINIMAX_M3_TOOL_OPEN) {
        let start = cursor + relative_start;
        let body_start = start + MINIMAX_M3_TOOL_OPEN.len();
        let Some(relative_end) = normalized[body_start..].find(MINIMAX_M3_TOOL_CLOSE) else {
            return (text.to_string(), Vec::new());
        };
        let end = body_start + relative_end;
        let block = &normalized[body_start..end];
        let Some(block_calls) = parse_minimax_m3_tool_block(block, tools) else {
            return (text.to_string(), Vec::new());
        };
        if block_calls.is_empty() {
            return (text.to_string(), Vec::new());
        }
        plain.push_str(&normalized[cursor..start]);
        calls.extend(block_calls);
        cursor = end + MINIMAX_M3_TOOL_CLOSE.len();
    }
    plain.push_str(&normalized[cursor..]);
    (plain, calls)
}

/// Parse every `<invoke>` in one M3 `<tool_call>` block.  Extra non-whitespace
/// markup is rejected so malformed output remains ordinary content.
fn parse_minimax_m3_tool_block(block: &str, tools: &[Value]) -> Option<Vec<Value>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while cursor < block.len() {
        let relative_start = block[cursor..].find(MINIMAX_M3_INVOKE_OPEN);
        let Some(relative_start) = relative_start else {
            return block[cursor..].trim().is_empty().then_some(calls);
        };
        let start = cursor + relative_start;
        if !block[cursor..start].trim().is_empty() {
            return None;
        }
        let name_start = start + MINIMAX_M3_INVOKE_OPEN.len();
        let relative_name_end = block[name_start..].find("\">")?;
        let name_end = name_start + relative_name_end;
        let name = &block[name_start..name_end];
        if name.is_empty() {
            return None;
        }
        let function = minimax_m3_function(tools, name)?;
        let body_start = name_end + 2; // `">`
        let relative_end = block[body_start..].find(MINIMAX_M3_INVOKE_CLOSE)?;
        let end = body_start + relative_end;
        let args =
            parse_minimax_m3_parameters(&block[body_start..end], function.get("parameters"))?;
        calls.push(openai_call(name, &args));
        cursor = end + MINIMAX_M3_INVOKE_CLOSE.len();
    }
    Some(calls)
}

/// Locate an OpenAI tool declaration by name.  Accept the standard wrapper
/// (`{"type":"function","function":…}`) and the inner function object used
/// by a few clients' template tests.
fn minimax_m3_function<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools.iter().find_map(|tool| {
        let function = tool.get("function").unwrap_or(tool);
        (function.get("name").and_then(Value::as_str) == Some(name)).then_some(function)
    })
}

struct MiniMaxM3Frame<'a> {
    tag: String,
    schema: Option<&'a Value>,
    value: Value,
}

/// Decode M3's recursively-expanded XML arguments.  The template prefixes
/// each tag with the namespace token, so splitting on that token mirrors the
/// source parser and avoids treating arbitrary text as XML.
fn parse_minimax_m3_parameters(body: &str, schema: Option<&Value>) -> Option<Value> {
    let root = if minimax_m3_has_type(schema, &["array"]) && minimax_m3_body_starts_with_item(body)
    {
        Value::Array(Vec::new())
    } else {
        Value::Object(Map::new())
    };
    let mut stack = vec![MiniMaxM3Frame {
        tag: String::new(),
        schema,
        value: root,
    }];

    for raw_chunk in body.split(MINIMAX_M3_NS) {
        // This matches SGLang's parser: whitespace separating tags is layout,
        // while text after an opening tag is the actual leaf value.
        let chunk = raw_chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        if let Some(after_close) = chunk.strip_prefix("</") {
            let close_end = after_close.find('>')?;
            if !after_close[close_end + 1..].trim().is_empty() {
                return None;
            }
            let tag = after_close[..close_end].trim();
            if tag.is_empty() || stack.len() <= 1 || stack.last()?.tag != tag {
                return None;
            }
            let frame = stack.pop()?;
            if !minimax_m3_assign_child(&mut stack.last_mut()?.value, frame.tag, frame.value) {
                return None;
            }
            continue;
        }
        let Some(after_open) = chunk.strip_prefix('<') else {
            return None;
        };
        let open_end = after_open.find('>')?;
        let tag = after_open[..open_end].trim();
        if tag.is_empty() || tag.contains('/') {
            return None;
        }
        let leaf_text = &after_open[open_end + 1..];
        let parent = stack.last()?;
        let child_schema = minimax_m3_child_schema(parent.schema, tag, &parent.value);
        let value = if !leaf_text.is_empty() || minimax_m3_is_scalar_schema(child_schema) {
            minimax_m3_leaf_value(leaf_text, child_schema)
        } else {
            minimax_m3_new_container(child_schema)
        };
        stack.push(MiniMaxM3Frame {
            tag: tag.to_string(),
            schema: child_schema,
            value,
        });
    }

    (stack.len() == 1).then(|| stack.pop().expect("root frame").value)
}

fn minimax_m3_body_starts_with_item(body: &str) -> bool {
    body.split(MINIMAX_M3_NS)
        .map(str::trim)
        .find(|chunk| !chunk.is_empty())
        .is_some_and(|chunk| chunk.starts_with("<item>"))
}

fn minimax_m3_schema_types(schema: Option<&Value>) -> Vec<&str> {
    let Some(schema) = schema else {
        return Vec::new();
    };
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn minimax_m3_has_type(schema: Option<&Value>, wanted: &[&str]) -> bool {
    minimax_m3_schema_types(schema).into_iter().any(|kind| {
        wanted
            .iter()
            .any(|candidate| kind.eq_ignore_ascii_case(candidate))
    })
}

fn minimax_m3_is_scalar_schema(schema: Option<&Value>) -> bool {
    let kinds = minimax_m3_schema_types(schema);
    let scalar = kinds.iter().any(|kind| {
        [
            "string", "str", "integer", "int", "number", "float", "boolean", "bool", "null",
        ]
        .iter()
        .any(|candidate| kind.eq_ignore_ascii_case(candidate))
    });
    let container = kinds.iter().any(|kind| {
        ["object", "array"]
            .iter()
            .any(|candidate| kind.eq_ignore_ascii_case(candidate))
    });
    scalar && !container
}

fn minimax_m3_child_schema<'a>(
    parent_schema: Option<&'a Value>,
    tag: &str,
    parent_value: &Value,
) -> Option<&'a Value> {
    let schema = parent_schema?;
    if minimax_m3_has_type(Some(schema), &["array"]) && tag == "item" {
        return minimax_m3_array_item_schema(schema, parent_value);
    }
    if let Some(schema) = minimax_m3_schema_property(schema, tag) {
        return Some(schema);
    }
    schema
        .get("additionalProperties")
        .filter(|value| value.is_object())
}

fn minimax_m3_schema_property<'a>(schema: &'a Value, key: &str) -> Option<&'a Value> {
    if let Some(value) = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(key))
    {
        return Some(value);
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(value) = schema
            .get(keyword)
            .and_then(Value::as_array)
            .and_then(|branches| {
                branches
                    .iter()
                    .find_map(|branch| minimax_m3_schema_property(branch, key))
            })
        {
            return Some(value);
        }
    }
    None
}

fn minimax_m3_array_item_schema<'a>(schema: &'a Value, value: &Value) -> Option<&'a Value> {
    let index = value.as_array().map_or(0, Vec::len);
    if let Some(item) = schema
        .get("prefixItems")
        .and_then(Value::as_array)
        .and_then(|items| items.get(index))
    {
        return Some(item);
    }
    if let Some(item) = schema
        .get("additionalItems")
        .filter(|value| value.is_object())
    {
        return Some(item);
    }
    if let Some(item) = schema.get("items").filter(|value| value.is_object()) {
        return Some(item);
    }
    schema
        .get("prefixItems")
        .and_then(Value::as_array)
        .and_then(|items| items.last())
}

fn minimax_m3_new_container(schema: Option<&Value>) -> Value {
    if minimax_m3_has_type(schema, &["array"]) {
        Value::Array(Vec::new())
    } else {
        Value::Object(Map::new())
    }
}

/// M3 scalar coercion follows its reference detector: strings remain literal,
/// nullable strings turn only exact `null` spellings into JSON null, and typed
/// numeric/boolean fields are decoded from their XML text.
fn minimax_m3_leaf_value(raw: &str, schema: Option<&Value>) -> Value {
    let kinds = minimax_m3_schema_types(schema);
    let has = |wanted: &[&str]| {
        kinds.iter().any(|kind| {
            wanted
                .iter()
                .any(|candidate| kind.eq_ignore_ascii_case(candidate))
        })
    };
    let null_permitted = has(&["null"]);
    if has(&["string", "str"]) && !null_permitted {
        return Value::String(raw.to_string());
    }
    if null_permitted && (matches!(raw, "null" | "Null" | "NULL") || raw.is_empty()) {
        return Value::Null;
    }

    let parsed = raw.trim();
    if has(&["integer", "int"]) {
        if let Ok(value) = parsed.parse::<i64>() {
            return Value::Number(value.into());
        }
        if let Ok(value) = parsed.parse::<u64>() {
            return Value::Number(value.into());
        }
    }
    if has(&["number", "float"]) {
        if let Ok(value) = parsed.parse::<f64>() {
            if value.is_finite() {
                if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                    return Value::Number((value as i64).into());
                }
                if let Some(value) = serde_json::Number::from_f64(value) {
                    return Value::Number(value);
                }
            }
        }
    }
    if has(&["boolean", "bool"]) {
        match parsed.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => return Value::Bool(true),
            "false" | "0" | "no" | "off" => return Value::Bool(false),
            _ => {}
        }
    }
    Value::String(raw.to_string())
}

fn minimax_m3_assign_child(parent: &mut Value, tag: String, child: Value) -> bool {
    match parent {
        Value::Array(values) => {
            values.push(child);
            true
        }
        Value::Object(values) => {
            if let Some(existing) = values.get_mut(&tag) {
                if let Value::Array(items) = existing {
                    items.push(child);
                } else {
                    let old = std::mem::replace(existing, Value::Null);
                    *existing = Value::Array(vec![old, child]);
                }
            } else {
                values.insert(tag, child);
            }
            true
        }
        _ => false,
    }
}

/// Extract tool calls. Returns the text outside the calls (trimmed) and
/// the calls in OpenAI shape. `tools` is the request's declared tool list
/// (needed for the MiniCPM5 grammar's schema-typed values and name
/// validation; the Hermes grammar carries JSON and needs none).
pub fn extract_tool_calls(text: &str, tools: Option<&[Value]>) -> (String, Vec<Value>) {
    let (start, end) = content_region(text);
    let (plain_mid, mut calls) = extract_hermes(&text[start..end], tools.unwrap_or(&[]));
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
fn extract_hermes(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
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
            .or_else(|| parse_nanbeige_function(body))
            .or_else(|| parse_arg_key_call(body, tools));
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

/// Spark-X2.5's key/value grammar, normalised to the JSON shape:
/// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`. The template renders
/// a string value as is and anything else through `tojson`, so a value
/// stays a string when the tool's schema declares the parameter a string
/// (or when no schema is known and the text is not a JSON object, array,
/// number, boolean or null); otherwise it is parsed back as JSON.
fn parse_arg_key_call(body: &str, tools: &[Value]) -> Option<Value> {
    let t = body.trim();
    let first = t.find("<arg_key>").unwrap_or(t.len());
    let name = t[..first].trim();
    // a function name: an identifier, never prose that merely sits inside
    // a <tool_call> wrapper
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
    {
        return None;
    }
    let declared = tools
        .iter()
        .filter_map(|tool| tool.get("function").or(Some(tool)))
        .find(|f| f.get("name").and_then(|n| n.as_str()) == Some(name));
    if !tools.is_empty() && declared.is_none() {
        return None;
    }
    let props = declared
        .and_then(|f| f.get("parameters"))
        .and_then(|p| p.get("properties"));
    let mut args = Map::new();
    let mut rest = &t[first..];
    while let Some(ks) = rest.find("<arg_key>") {
        let k0 = ks + "<arg_key>".len();
        let k1 = rest[k0..].find("</arg_key>")? + k0;
        let key = rest[k0..k1].trim().to_string();
        let after = &rest[k1 + "</arg_key>".len()..];
        let v0 = after.find("<arg_value>")? + "<arg_value>".len();
        let v1 = after[v0..].find("</arg_value>")? + v0;
        let raw = &after[v0..v1];
        let declared_string = props
            .and_then(|p| p.get(&key))
            .and_then(|s| s.get("type"))
            .map(|ty| match ty {
                Value::String(s) => s == "string",
                Value::Array(a) => a.iter().any(|x| x.as_str() == Some("string")),
                _ => false,
            });
        let val = match declared_string {
            Some(true) => Value::String(raw.to_string()),
            _ => serde_json::from_str::<Value>(raw.trim())
                .ok()
                .filter(|v| !v.is_string() || declared_string == Some(false))
                .unwrap_or_else(|| Value::String(raw.to_string())),
        };
        args.insert(key, val);
        rest = &after[v1 + "</arg_value>".len()..];
    }
    if !rest.trim().is_empty() {
        return None;
    }
    Some(serde_json::json!({"name": name, "arguments": args}))
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

    #[test]
    fn spark_arg_key_calls_follow_the_schema() {
        let tools = vec![serde_json::json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "zip": {"type": "string"},
                "days": {"type": "integer"}, "units": {"type": "object"}}}}})];
        let text = "<think>let me check</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>zip</arg_key><arg_value>75001</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>units</arg_key><arg_value>{\"t\": \"C\"}</arg_value></tool_call>";
        let (plain, calls) = extract_tool_calls(text, Some(&tools));
        assert_eq!(calls.len(), 1);
        assert!(plain.ends_with("Sure."));
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather");
        let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["zip"], "75001");
        assert_eq!(args["days"], 3);
        assert_eq!(args["units"]["t"], "C");
        // no schema: plain words stay strings, JSON values parse
        let (_, calls) = extract_tool_calls(
            "<tool_call>lookup<arg_key>q</arg_key><arg_value>rust</arg_value><arg_key>n</arg_key><arg_value>5</arg_value></tool_call>",
            None,
        );
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["q"], "rust");
        assert_eq!(args["n"], 5);
        // a broken block stays text
        let (plain, calls) = extract_tool_calls(
            "<tool_call>lookup<arg_key>q</arg_key>rust</tool_call>",
            None,
        );
        assert!(calls.is_empty());
        assert!(plain.contains("<tool_call>"));
    }

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

    fn minimax_m3_tools() -> Vec<Value> {
        vec![
            serde_json::json!({"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "count": {"type": "integer"},
                    "ratio": {"type": "number"}, "verbose": {"type": "boolean"}
                }, "required": ["city"]}
            }}),
            serde_json::json!({"type": "function", "function": {
                "name": "create_event",
                "parameters": {"type": "object", "properties": {
                    "title": {"type": "string"},
                    "location": {"type": "object", "properties": {
                        "city": {"type": "string"}, "zip": {"type": "integer"}
                    }},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }}
            }}),
            serde_json::json!({"type": "function", "function": {
                "name": "configure",
                "parameters": {"type": "object", "properties": {
                    "mode": {"type": "string"},
                    "optional": {"type": ["string", "null"]}
                }}
            }}),
        ]
    }

    fn m3_wire(parts: &[&str]) -> String {
        parts
            .iter()
            .map(|part| format!("{MINIMAX_M3_NS}{part}"))
            .collect()
    }

    fn stream_m3(chunks: &[&str], tools: &[Value]) -> (String, Vec<Value>, Option<String>) {
        let text = chunks.concat();
        let mut stream = MiniMaxM3Stream::with_declared_tools();
        let mut content = String::new();
        for chunk in chunks {
            content.push_str(&stream.push(chunk));
        }
        let (plain, calls, reasoning) = extract_minimax_m3_response(&text, Some(tools));
        content.push_str(&stream.finish(&plain));
        (content, calls, reasoning)
    }

    // ── MiniMax M3 ──

    #[test]
    fn minimax_m3_splits_adaptive_reasoning_and_parses_namespaced_call() {
        let tools = minimax_m3_tools();
        let text = format!(
            "<mm:think>I should use the weather tool.</mm:think>Checking.{}",
            m3_wire(&[
                "<tool_call>",
                "<invoke name=\"get_weather\">",
                "<city>Beijing",
                "</city>",
                "<count>7",
                "</count>",
                "<ratio>0.5",
                "</ratio>",
                "<verbose>true",
                "</verbose>",
                "</invoke>",
                "</tool_call>",
            ])
        );
        let (plain, calls, reasoning) = extract_minimax_m3_response(&text, Some(&tools));
        assert_eq!(reasoning.as_deref(), Some("I should use the weather tool."));
        assert_eq!(plain, "Checking.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"city": "Beijing", "count": 7, "ratio": 0.5, "verbose": true})
        );
    }

    #[test]
    fn minimax_m3_recursively_decodes_object_array_and_nullable_scalars() {
        let tools = minimax_m3_tools();
        let event = m3_wire(&[
            "<tool_call>",
            "<invoke name=\"create_event\">",
            "<title>Standup",
            "</title>",
            "<location>",
            "<city>NYC",
            "</city>",
            "<zip>10001",
            "</zip>",
            "</location>",
            "<tags>",
            "<item>red",
            "</item>",
            "<item>blue",
            "</item>",
            "</tags>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (plain, calls, reasoning) = extract_minimax_m3_response(&event, Some(&tools));
        assert_eq!(plain, "");
        assert_eq!(reasoning, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({
                "title": "Standup",
                "location": {"city": "NYC", "zip": 10001},
                "tags": ["red", "blue"],
            })
        );

        let config = m3_wire(&[
            "<tool_call>",
            "<invoke name=\"configure\">",
            "<mode>null",
            "</mode>",
            "<optional>null",
            "</optional>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (_, calls, _) = extract_minimax_m3_response(&config, Some(&tools));
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"mode": "null", "optional": null})
        );
    }

    #[test]
    fn minimax_m3_strips_only_leading_close_and_keeps_malformed_output() {
        let tools = minimax_m3_tools();
        let (plain, calls, reasoning) =
            extract_minimax_m3_response(" \n</mm:think>direct answer", Some(&tools));
        assert_eq!(plain, "direct answer");
        assert!(calls.is_empty());
        assert_eq!(reasoning, None);

        let (plain, calls, reasoning) =
            extract_minimax_m3_response("<mm:think>truncated", Some(&tools));
        assert_eq!(plain, "");
        assert!(calls.is_empty());
        assert_eq!(reasoning.as_deref(), Some("truncated"));

        let malformed = m3_wire(&[
            "<tool_call>",
            "<invoke name=\"get_weather\">",
            "<city>Beijing",
            "</wrong>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (plain, calls, reasoning) = extract_minimax_m3_response(&malformed, Some(&tools));
        assert_eq!(plain, malformed);
        assert!(calls.is_empty());
        assert_eq!(reasoning, None);

        let unknown = m3_wire(&[
            "<tool_call>",
            "<invoke name=\"not_declared\">",
            "</invoke>",
            "</tool_call>",
        ]);
        let (plain, calls, _) = extract_minimax_m3_response(&unknown, Some(&tools));
        assert_eq!(plain, unknown);
        assert!(calls.is_empty());
    }

    #[test]
    fn minimax_m3_accepts_namespace_tag_spacing_without_rewriting_plain_text() {
        let tools = minimax_m3_tools();
        let text = format!(
            "answer{} \n<tool_call>{}\t<invoke name=\"get_weather\">{} <city>Rome{}</city>{}</invoke>{}</tool_call>",
            MINIMAX_M3_NS,
            MINIMAX_M3_NS,
            MINIMAX_M3_NS,
            MINIMAX_M3_NS,
            MINIMAX_M3_NS,
            MINIMAX_M3_NS,
        );
        let (plain, calls, _) = extract_minimax_m3_response(&text, Some(&tools));
        assert_eq!(plain, "answer");
        assert_eq!(calls.len(), 1);
        assert_eq!(args(&calls[0]), serde_json::json!({"city": "Rome"}));
    }

    #[test]
    fn minimax_m3_stream_hides_split_reasoning_and_tool_protocol() {
        let tools = minimax_m3_tools();
        let chunks = [
            "<mm:",
            "think>use the weather tool</mm:th",
            "ink>Checking.",
            "]<]min",
            "imax[>[<tool",
            "_call> ]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<city>Paris",
            "]<]minimax[>[</city> ]<]minimax[>[</invoke> ]<]minimax[>[</tool_call>",
        ];
        let (content, calls, reasoning) = stream_m3(&chunks, &tools);
        assert_eq!(content, "Checking.");
        assert!(!content.contains("minimax"));
        assert!(!content.contains("mm:think"));
        assert_eq!(reasoning.as_deref(), Some("use the weather tool"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(args(&calls[0]), serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn minimax_m3_stream_is_safe_at_every_marker_boundary() {
        let tools = minimax_m3_tools();
        let text = format!(
            "<mm:think>private</mm:think>public{}",
            m3_wire(&[
                "<tool_call>",
                "<invoke name=\"get_weather\">",
                "<city>Rome",
                "</city>",
                "</invoke>",
                "</tool_call>",
            ])
        );
        for split in 0..=text.len() {
            if !text.is_char_boundary(split) {
                continue;
            }
            let (content, calls, reasoning) = stream_m3(&[&text[..split], &text[split..]], &tools);
            assert_eq!(content, "public", "split at byte {split}");
            assert!(!content.contains("minimax"), "split at byte {split}");
            assert!(!content.contains("mm:think"), "split at byte {split}");
            assert_eq!(reasoning.as_deref(), Some("private"), "split at byte {split}");
            assert_eq!(calls.len(), 1, "split at byte {split}");
        }
    }

    #[test]
    fn minimax_m3_stream_handles_a_split_adaptive_close() {
        let tools = minimax_m3_tools();
        let (content, calls, reasoning) =
            stream_m3(&[" \n</mm:", "think>direct", " answer"], &tools);
        assert_eq!(content, "direct answer");
        assert!(calls.is_empty());
        assert_eq!(reasoning, None);
    }

    #[test]
    fn minimax_m3_stream_retains_malformed_tool_output_at_finish() {
        let tools = minimax_m3_tools();
        let chunks = [
            "Before.",
            "]<]minimax[>[<tool_call>",
            "]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<city>Paris",
            "]<]minimax[>[</wrong>] <]minimax[>[</invoke>] <]minimax[>[</tool_call>",
        ];
        let (content, calls, reasoning) = stream_m3(&chunks, &tools);
        assert_eq!(content, chunks.concat());
        assert!(calls.is_empty());
        assert_eq!(reasoning, None);
    }

    #[test]
    fn minimax_m3_stream_never_leaks_unterminated_reasoning() {
        let mut stream = MiniMaxM3Stream::new();
        for chunk in ["<mm:", "think>private ", "chain of thought"] {
            let emitted = stream.push(chunk);
            assert!(emitted.is_empty(), "reasoning leaked from {chunk:?}: {emitted:?}");
        }
        let text = "<mm:think>private chain of thought";
        let (plain, calls, reasoning) = extract_minimax_m3_response(text, None);
        assert!(plain.is_empty());
        assert!(calls.is_empty());
        assert_eq!(reasoning.as_deref(), Some("private chain of thought"));
        assert!(stream.finish(&plain).is_empty());
    }

    #[test]
    fn minimax_m3_stream_defers_namespace_text_for_transactional_parser() {
        let tools = minimax_m3_tools();
        // The strict parser normalizes whitespace after this namespace only
        // when all later tool blocks are valid.  A later malformed block must
        // therefore restore these exact wire bytes without duplicating an
        // already streamed prefix.
        let text = format!(
            "before{MINIMAX_M3_NS}  <ordinary>after{}",
            m3_wire(&[
                "<tool_call>",
                "<invoke name=\"get_weather\">",
                "<city>Paris",
                "</wrong>",
                "</invoke>",
                "</tool_call>",
            ])
        );
        let (content, calls, reasoning) = stream_m3(&[&text[..7], &text[7..]], &tools);
        assert_eq!(content, text);
        assert!(calls.is_empty());
        assert_eq!(reasoning, None);
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
