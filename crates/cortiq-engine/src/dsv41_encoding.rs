//! DeepSeek-V4.1 prompt encoding and completion parsing.
//!
//! The V4.1 checkpoint intentionally ships without a Jinja chat template.
//! This module is a Rust port of the official `encoding/encoding.py` at the
//! pinned model revision.  It keeps the protocol in one place for the CLI,
//! OpenAI server, and any embedding application: spaced DSML tags, numeric
//! reasoning effort, mid conversation system turns, tool result merging, and
//! image ordering all follow the reference implementation.

use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::io;
use thiserror::Error;

pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const DSML_TOKEN: &str = "｜DSML｜";
pub const USER_SP_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
pub const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
pub const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

const TOOL_CALLS_BLOCK_NAME: &str = " calls";
const TOOL_CALL_TAG_NAME: &str = " invoke";
const TOOL_PARAMETER_TAG_NAME: &str = " parameter";

/// Internal quick instruction tasks supported by the official formatter.
pub const TASK_TOKENS: &[(&str, &str)] = &[
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("invalid thinking mode '{0}', expected 'chat' or 'thinking'")]
    InvalidThinkingMode(String),
    #[error("invalid reasoning effort '{0}', expected an integer in 1..=100 or low/high/max")]
    InvalidReasoningEffort(String),
    #[error("message {index} has unsupported role '{role}'")]
    UnsupportedRole { index: usize, role: String },
    #[error("invalid message: {0}")]
    InvalidMessage(String),
    #[error("invalid image block: {0}")]
    InvalidImage(String),
    #[error("invalid DSML tool call: {0}")]
    InvalidToolCall(String),
    #[error("invalid completion: {0}")]
    InvalidCompletion(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn parse(value: &str) -> Result<Self, EncodingError> {
        match value {
            "chat" => Ok(Self::Chat),
            "thinking" => Ok(Self::Thinking),
            other => Err(EncodingError::InvalidThinkingMode(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Thinking => "thinking",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Budget(u8),
    Low,
    High,
    Max,
}

impl Default for ReasoningEffort {
    fn default() -> Self {
        Self::High
    }
}

impl ReasoningEffort {
    pub fn budget(self) -> u8 {
        match self {
            Self::Budget(v) => v,
            Self::Low => 50,
            Self::High => 75,
            Self::Max => 100,
        }
    }

    pub fn parse(value: &str) -> Result<Self, EncodingError> {
        match value {
            "low" => Ok(Self::Low),
            "high" => Ok(Self::High),
            "max" => Ok(Self::Max),
            other => other
                .parse::<u8>()
                .ok()
                .filter(|v| (1..=100).contains(v))
                .map(Self::Budget)
                .ok_or_else(|| EncodingError::InvalidReasoningEffort(other.to_string())),
        }
    }

    pub fn from_json(value: &Value) -> Result<Self, EncodingError> {
        if let Some(s) = value.as_str() {
            return match s {
                "low" => Ok(Self::Low),
                "high" => Ok(Self::High),
                "max" => Ok(Self::Max),
                _ => Err(EncodingError::InvalidReasoningEffort(value.to_string())),
            };
        }
        // bool is deliberately rejected even though serde_json represents it
        // as neither an integer nor a float.
        if let Some(v) = value.as_i64() {
            if (1..=100).contains(&v) {
                return Ok(Self::Budget(v as u8));
            }
        }
        Err(EncodingError::InvalidReasoningEffort(value.to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    pub thinking_mode: ThinkingMode,
    pub context: Vec<Value>,
    pub drop_thinking: bool,
    pub add_default_bos_token: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            thinking_mode: ThinkingMode::Chat,
            context: Vec::new(),
            drop_thinking: true,
            add_default_bos_token: true,
            reasoning_effort: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPrompt {
    pub prompt: String,
    /// Image records in the order their placeholders occur in the current
    /// messages. Records are intentionally JSON shaped so OpenAI, Anthropic,
    /// local paths, and data URLs survive the protocol boundary unchanged.
    pub images: Vec<Value>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

pub fn to_json(value: &Value) -> String {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonJsonFormatter);
    if value.serialize(&mut serializer).is_err() {
        return "null".to_string();
    }
    String::from_utf8(bytes).unwrap_or_else(|_| "null".to_string())
}

fn object(value: &Value) -> Result<&Map<String, Value>, EncodingError> {
    value
        .as_object()
        .ok_or_else(|| EncodingError::InvalidMessage("expected a JSON object".to_string()))
}

fn role(value: &Value, index: usize) -> Result<&str, EncodingError> {
    object(value)?
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| EncodingError::InvalidMessage(format!("message {index} has no string role")))
}

fn text(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or("").to_string()
}

/// Split a qualified tool name and validate explicit namespace metadata.
fn split_tool_name(
    name: &str,
    namespace: Option<&str>,
) -> Result<(Option<String>, String), EncodingError> {
    let (mut ns, bare) = if let Some((prefix, suffix)) = name.split_once("::") {
        if suffix.contains("::") {
            return Err(EncodingError::InvalidToolCall(format!(
                "tool name contains multiple '::': {name}"
            )));
        }
        if let Some(given) = namespace {
            if given != prefix {
                return Err(EncodingError::InvalidToolCall(format!(
                    "conflicting tool namespaces: {given} != {prefix}"
                )));
            }
        }
        (Some(prefix.to_string()), suffix.to_string())
    } else {
        (namespace.map(str::to_string), name.to_string())
    };
    if bare.contains("::") || ns.as_deref().is_some_and(|v| v.contains("::")) {
        return Err(EncodingError::InvalidToolCall(format!(
            "invalid qualified tool name: {name}"
        )));
    }
    if bare.is_empty() {
        return Err(EncodingError::InvalidToolCall(
            "tool name is empty".to_string(),
        ));
    }
    Ok((ns.take(), bare))
}

fn tool_name_for_encoding(tool: &Value) -> Result<String, EncodingError> {
    let map = object(tool)?;
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| EncodingError::InvalidToolCall("tool definition has no name".to_string()))?;
    let namespace = match map.get("namespace") {
        Some(Value::String(v)) => Some(v.as_str()),
        Some(Value::Object(v)) => v.get("name").and_then(Value::as_str),
        _ => None,
    };
    let (ns, bare) = split_tool_name(name, namespace)?;
    Ok(ns.map(|v| format!("{v}::{bare}")).unwrap_or(bare))
}

fn tools_from_openai_format(tools: &Value) -> Result<Vec<Value>, EncodingError> {
    let list = tools
        .as_array()
        .ok_or_else(|| EncodingError::InvalidToolCall("tools must be an array".to_string()))?;
    let mut out = Vec::with_capacity(list.len());
    for tool in list {
        let map = object(tool)?;
        let function = map
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                EncodingError::InvalidToolCall("OpenAI tool has no function object".to_string())
            })?;
        let mut f = function.clone();
        if let Some(ns) = map.get("namespace") {
            f.insert("namespace".to_string(), ns.clone());
        }
        let encoded = tool_name_for_encoding(&Value::Object(f.clone()))?;
        f.insert("name".to_string(), Value::String(encoded));
        if let Some(Value::Object(ns)) = f.get("namespace") {
            if let Some(description) = ns.get("description").and_then(Value::as_str) {
                let old = f.get("description").and_then(Value::as_str).unwrap_or("");
                f.insert(
                    "description".to_string(),
                    Value::String(format!("{description}\n{old}")),
                );
            }
        }
        // The internal schema carries namespace only while constructing the
        // qualified name.  The official formatter removes it from the
        // emitted function object regardless of whether the value was a
        // string or an object.
        f.remove("namespace");
        out.push(Value::Object(f));
    }
    Ok(out)
}

fn tool_calls_from_openai_format(value: &Value) -> Result<Vec<Value>, EncodingError> {
    let calls = value
        .as_array()
        .ok_or_else(|| EncodingError::InvalidToolCall("tool_calls must be an array".to_string()))?;
    let mut out = Vec::with_capacity(calls.len());
    for call in calls {
        let map = object(call)?;
        let function = map
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                EncodingError::InvalidToolCall("tool call has no function object".to_string())
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                EncodingError::InvalidToolCall("tool call has no function name".to_string())
            })?;
        let namespace = map
            .get("namespace")
            .and_then(Value::as_str)
            .or_else(|| function.get("namespace").and_then(Value::as_str));
        let (ns, bare) = split_tool_name(name, namespace)?;
        let mut result = Map::new();
        result.insert("name".to_string(), Value::String(bare));
        result.insert(
            "arguments".to_string(),
            function
                .get("arguments")
                .cloned()
                .unwrap_or(Value::String(String::new())),
        );
        if let Some(ns) = ns {
            result.insert("namespace".to_string(), Value::String(ns));
        }
        out.push(Value::Object(result));
    }
    Ok(out)
}

fn encode_arguments_to_dsml(tool_call: &Value) -> Result<String, EncodingError> {
    let map = object(tool_call)?;
    let mut arguments = map.get("arguments").cloned().unwrap_or(Value::Null);
    if !arguments.is_object() {
        for _ in 0..2 {
            if let Some(s) = arguments.as_str() {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    arguments = v;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    let mut fields = Vec::new();
    if let Some(args) = arguments.as_object() {
        for (key, value) in args {
            let is_string = value.is_string();
            let encoded = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| to_json(value));
            fields.push(format!(
                "<{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME} name=\"{key}\" string=\"{}\">{encoded}</{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}>",
                if is_string { "true" } else { "false" }
            ));
        }
    } else {
        fields.push(format!(
            "<{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME} name=\"arguments\" string=\"false\">{}</{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}>",
            to_json(&arguments)
        ));
    }
    Ok(fields.join("\n"))
}

fn render_tools(tools: &Value) -> Result<String, EncodingError> {
    let functions = tools_from_openai_format(tools)?;
    let schemas = functions.iter().map(to_json).collect::<Vec<_>>().join("\n");
    Ok(format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\" block like the following:\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n<{DSML_TOKEN}{TOOL_CALL_TAG_NAME} name=\"$TOOL_NAME\">\n<{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}>\n...\n</{DSML_TOKEN}{TOOL_CALL_TAG_NAME}>\n<{DSML_TOKEN}{TOOL_CALL_TAG_NAME} name=\"$TOOL_NAME2\">\n...\n</{DSML_TOKEN}{TOOL_CALL_TAG_NAME}>\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by {THINKING_START_TOKEN}), you MUST output your complete reasoning inside {THINKING_START_TOKEN}...{THINKING_END_TOKEN} BEFORE any tool calls or final response.\n\nOtherwise, output directly after {THINKING_END_TOKEN} with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    ))
}

fn decode_dsml_to_arguments(
    tool_name: &str,
    args: &[(String, String, String)],
) -> Result<Value, EncodingError> {
    let mut fields = Vec::with_capacity(args.len());
    for (key, value, string) in args {
        let encoded = if string == "true" {
            to_json(&Value::String(value.clone()))
        } else {
            value.clone()
        };
        fields.push(format!(
            "{}: {}",
            to_json(&Value::String(key.clone())),
            encoded
        ));
    }
    let arguments = format!("{{{}}}", fields.join(", "));
    let (namespace, name) = split_tool_name(tool_name, None)?;
    let mut out = Map::new();
    out.insert("name".to_string(), Value::String(name));
    out.insert("arguments".to_string(), Value::String(arguments));
    if let Some(namespace) = namespace {
        out.insert("namespace".to_string(), Value::String(namespace));
    }
    Ok(Value::Object(out))
}

/// Convert compact `<image>path</image>` input to OpenAI style content blocks.
pub fn parse_tagged_text(input: &str) -> Result<Value, EncodingError> {
    let mut blocks = Vec::new();
    let mut cursor = 0usize;
    let mut found = false;
    while let Some(rel) = input[cursor..].find("<image>") {
        let start = cursor + rel;
        if input[cursor..start].contains("</image>") {
            return Err(EncodingError::InvalidImage(
                "malformed <image>path</image> tag".to_string(),
            ));
        }
        let end_rel = input[start + 7..]
            .find("</image>")
            .ok_or_else(|| EncodingError::InvalidImage("malformed <image> tag".to_string()))?;
        let end = start + 7 + end_rel;
        if start > cursor {
            blocks.push(json!({"type":"text", "text": &input[cursor..start]}));
        }
        let path = &input[start + 7..end];
        if path.is_empty() {
            return Err(EncodingError::InvalidImage(
                "image path must not be empty".to_string(),
            ));
        }
        blocks.push(json!({"type":"image_url", "image_url":{"url":path}}));
        cursor = end + 8;
        found = true;
    }
    if input[cursor..].contains("<image>") || input[cursor..].contains("</image>") {
        return Err(EncodingError::InvalidImage(
            "malformed <image>path</image> tag".to_string(),
        ));
    }
    if !found {
        return Ok(Value::String(input.to_string()));
    }
    if cursor < input.len() {
        blocks.push(json!({"type":"text", "text": &input[cursor..]}));
    }
    Ok(Value::Array(blocks))
}

fn is_image_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("image" | "image_url")
    )
}

fn extract_image(block: &Value) -> Result<Value, EncodingError> {
    let map = object(block)?;
    let mut record = Map::new();
    record.insert("type".to_string(), Value::String("image".to_string()));
    if map.get("type").and_then(Value::as_str) == Some("image_url") {
        match map.get("image_url") {
            Some(Value::String(url)) => {
                record.insert("url".to_string(), Value::String(url.clone()));
            }
            Some(Value::Object(image_url)) => {
                record.insert(
                    "url".to_string(),
                    image_url
                        .get("url")
                        .cloned()
                        .unwrap_or(Value::String(String::new())),
                );
            }
            _ => {}
        }
    } else {
        for key in ["source", "url", "data"] {
            if let Some(value) = map.get(key) {
                record.insert(key.to_string(), value.clone());
            }
        }
    }
    if !["source", "url", "data"]
        .iter()
        .any(|key| record.get(*key).is_some_and(json_truthy))
    {
        return Err(EncodingError::InvalidImage(
            "image block has no source".to_string(),
        ));
    }
    Ok(Value::Object(record))
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn process_image_blocks(blocks: &[Value]) -> Result<(Vec<Value>, Vec<Value>), EncodingError> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut images = Vec::new();
    for block in blocks {
        if !block.is_object() {
            out.push(block.clone());
            continue;
        }
        if is_image_block(block) {
            out.push(json!({"type":"text", "text":IMAGE_PLACEHOLDER}));
            images.push(extract_image(block)?);
            continue;
        }
        if block.get("type").and_then(Value::as_str) == Some("tool_result")
            && block.get("content").and_then(Value::as_array).is_some()
        {
            let (nested, nested_images) =
                process_image_blocks(block.get("content").and_then(Value::as_array).unwrap())?;
            let mut copy = object(block)?.clone();
            copy.insert("content".to_string(), Value::Array(nested));
            out.push(Value::Object(copy));
            images.extend(nested_images);
            continue;
        }
        if block.get("type").and_then(Value::as_str) == Some("text") {
            let value = text(block.get("text"));
            if value.contains(IMAGE_PLACEHOLDER) {
                return Err(EncodingError::InvalidImage(
                    "text blocks must use image content blocks for the image placeholder"
                        .to_string(),
                ));
            }
        }
        out.push(block.clone());
    }
    Ok((out, images))
}

fn validate_no_image_tokens(message: &Value) -> Result<(), EncodingError> {
    let map = object(message)?;
    for key in ["content", "reasoning_content"] {
        if map
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains(IMAGE_PLACEHOLDER))
        {
            return Err(EncodingError::InvalidImage(format!(
                "message {key} contains {IMAGE_PLACEHOLDER}; use an image content block"
            )));
        }
    }
    Ok(())
}

/// Normalize image blocks and return image records in prompt order.
pub fn process_image_messages(
    messages: &[Value],
) -> Result<(Vec<Value>, Vec<Value>), EncodingError> {
    let mut processed = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for original in messages {
        validate_no_image_tokens(original)?;
        let mut message = object(original)?.clone();
        if message.get("content_blocks").is_none() {
            if let Some(Value::Array(blocks)) = message.get("content") {
                message.insert("content_blocks".to_string(), Value::Array(blocks.clone()));
                message.remove("content");
            }
        }
        if let Some(Value::Array(blocks)) = message
            .get("content_blocks")
            .filter(|blocks| !blocks.as_array().map_or(true, |items| items.is_empty()))
        {
            let (new_blocks, new_images) = process_image_blocks(blocks)?;
            let strings = new_blocks
                .iter()
                .filter_map(|block| {
                    (block.get("type").and_then(Value::as_str) == Some("text"))
                        .then(|| text(block.get("text")))
                })
                .collect::<Vec<_>>();
            message.insert("content_blocks".to_string(), Value::Array(new_blocks));
            if !message.get("content").is_some_and(Value::is_string) {
                message.insert("content".to_string(), Value::String(strings.join("\n\n")));
            }
            images.extend(new_images);
        }
        processed.push(Value::Object(message));
    }
    Ok((processed, images))
}

/// Merge standalone OpenAI `tool` messages into a preceding user content list.
pub fn merge_tool_messages(messages: &[Value]) -> Result<Vec<Value>, EncodingError> {
    let mut merged = Vec::new();
    for original in messages {
        let message = object(original)?.clone();
        let message_role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if message_role == "tool" {
            // A tool result may arrive as structured blocks (text and
            // images); keep them so the images are collected in prompt
            // order after the results are sorted.
            let content = match (message.get("content_blocks"), message.get("content")) {
                (Some(blocks), content)
                    if json_truthy(blocks) && !content.is_some_and(Value::is_string) =>
                {
                    blocks.clone()
                }
                (_, content) => content.cloned().unwrap_or(Value::String(String::new())),
            };
            let block = json!({
                "type":"tool_result",
                "tool_use_id": message.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                "content": content,
            });
            let can_append = merged.last().is_some_and(|m: &Value| {
                m.get("role").and_then(Value::as_str) == Some("user")
                    && m.get("content_blocks").is_some()
            });
            if can_append {
                merged.last_mut().unwrap()["content_blocks"]
                    .as_array_mut()
                    .unwrap()
                    .push(block);
            } else {
                merged.push(json!({"role":"user", "content_blocks":[block]}));
            }
        } else if message_role == "user" {
            let blocks = message
                .get("content_blocks")
                .filter(|value| !value.is_null())
                .cloned()
                .unwrap_or_else(|| match message.get("content") {
                    // OpenAI-style structured content is already a block list.
                    Some(Value::Array(blocks)) => Value::Array(blocks.clone()),
                    content => json!([{
                        "type":"text",
                        "text": content.cloned().unwrap_or(Value::String(String::new()))
                    }]),
                });
            let can_append = merged.last().is_some_and(|m: &Value| {
                m.get("role").and_then(Value::as_str) == Some("user")
                    && m.get("content_blocks").is_some()
                    && m.get("task").is_none()
            });
            if can_append {
                let dst = merged.last_mut().unwrap()["content_blocks"]
                    .as_array_mut()
                    .unwrap();
                if let Some(src) = blocks.as_array() {
                    dst.extend(src.iter().cloned());
                }
            } else {
                let mut copy = message;
                copy.insert("content_blocks".to_string(), blocks);
                merged.push(Value::Object(copy));
            }
        } else {
            merged.push(Value::Object(message));
        }
    }
    Ok(merged)
}

/// Stable tool result order follows the preceding assistant's tool call order.
pub fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut order: HashMap<String, usize> = HashMap::new();
    for message in messages {
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                if let Some(calls) = message
                    .get("tool_calls")
                    .filter(|value| json_truthy(value))
                    .and_then(Value::as_array)
                {
                    order.clear();
                    for (index, call) in calls.iter().enumerate() {
                        let id = call.get("id").and_then(Value::as_str).or_else(|| {
                            call.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(Value::as_str)
                        });
                        if let Some(id) = id {
                            order.insert(id.to_string(), index);
                        }
                    }
                }
            }
            Some("user") => {
                if let Some(blocks) = message
                    .get_mut("content_blocks")
                    .and_then(Value::as_array_mut)
                {
                    let mut tools = blocks
                        .iter()
                        .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                        .cloned()
                        .collect::<Vec<_>>();
                    if tools.len() > 1 && !order.is_empty() {
                        tools.sort_by_key(|b| {
                            order
                                .get(b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""))
                                .copied()
                                .unwrap_or(0)
                        });
                        let mut next = 0;
                        for block in blocks.iter_mut() {
                            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                                *block = tools[next].clone();
                                next += 1;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Index of the message that opened the current turn. Mid-conversation
/// system messages count as user turns; a user message holding only tool
/// results does not start a new turn, so the reasoning of the assistant
/// message that called the tools stays in the prompt.
fn last_user_index(messages: &[Value]) -> isize {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(idx, m)| {
            let role = m.get("role").and_then(Value::as_str);
            if !(role == Some("user") || (role == Some("system") && *idx > 0)) {
                return false;
            }
            match m.get("content_blocks").and_then(Value::as_array) {
                Some(blocks) if !blocks.is_empty() => blocks.iter().any(|block| {
                    matches!(
                        block.get("type").and_then(Value::as_str),
                        Some("text" | "image" | "image_url")
                    )
                }),
                _ => true,
            }
        })
        .map(|(i, _)| i as isize)
        .unwrap_or(-1)
}

fn render_reasoning_effort(
    index: usize,
    mode: ThinkingMode,
    effort: Option<ReasoningEffort>,
) -> String {
    if index == 0 && mode == ThinkingMode::Thinking {
        let budget = effort.unwrap_or_default().budget();
        format!(
            "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        )
    } else {
        String::new()
    }
}

fn render_content_blocks(blocks: &[Value]) -> String {
    blocks
        .iter()
        .map(|block| match block.get("type").and_then(Value::as_str) {
            Some("text") => text(block.get("text")),
            Some("tool_result") => {
                let content = block.get("content");
                let rendered = if let Some(parts) = content.and_then(Value::as_array) {
                    parts
                        .iter()
                        .map(|part| {
                            if part.get("type").and_then(Value::as_str) == Some("text") {
                                text(part.get("text"))
                            } else {
                                format!("[Unsupported {}]", python_type_label(part))
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                } else {
                    // Python's formatter inserts a string tool result as-is;
                    // JSON quoting here would change the prompt protocol.
                    content
                        .map(|value| {
                            value
                                .as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| to_json(value))
                        })
                        .unwrap_or_default()
                };
                format!("<tool_result>{rendered}</tool_result>")
            }
            _ => format!("[Unsupported {}]", python_type_label(block)),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `f"{block.get('type')}"` for the unsupported-block placeholder.
fn python_type_label(block: &Value) -> String {
    match block.get("type") {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(kind)) => kind.clone(),
        Some(other) => to_json(other),
    }
}

fn task_token(task: &str) -> Option<&'static str> {
    TASK_TOKENS
        .iter()
        .find(|(name, _)| *name == task)
        .map(|(_, token)| *token)
}

/// Render one message exactly as the official Python implementation does.
pub fn render_message(
    index: usize,
    messages: &[Value],
    options: &EncodeOptions,
) -> Result<String, EncodingError> {
    let message = object(messages.get(index).ok_or_else(|| {
        EncodingError::InvalidMessage(format!("message index {index} out of range"))
    })?)?;
    let message_value = Value::Object(message.clone());
    let message_role = role(&message_value, index)?;
    let last_user = last_user_index(messages);
    let effort = render_reasoning_effort(index, options.thinking_mode, options.reasoning_effort);
    let mut prompt = if index == 0 && (!effort.is_empty() || message_role == "system") {
        SYSTEM_SP_TOKEN.to_string()
    } else {
        String::new()
    };
    prompt.push_str(&effort);

    match message_role {
        "system" => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP_TOKEN);
            }
            prompt.push_str(&text(message.get("content")));
            if let Some(tools) = message.get("tools").filter(|value| json_truthy(value)) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools)?);
            }
            if let Some(schema) = message
                .get("response_format")
                .filter(|value| json_truthy(value))
            {
                prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                prompt.push_str(&to_json(schema));
            }
        }
        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            if let Some(blocks) = message
                .get("content_blocks")
                .and_then(Value::as_array)
                .filter(|blocks| !blocks.is_empty())
            {
                prompt.push_str(&render_content_blocks(blocks));
            } else {
                prompt.push_str(&text(message.get("content")));
            }
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(&text(message.get("content")));
        }
        "tool" => {
            return Err(EncodingError::UnsupportedRole {
                index,
                role: "tool".to_string(),
            });
        }
        "assistant" => {
            let mut reasoning = String::new();
            let mut tool_calls = String::new();
            if let Some(calls) = message.get("tool_calls") {
                let calls = tool_calls_from_openai_format(calls)?;
                if !calls.is_empty() {
                    let mut encoded = Vec::new();
                    for call in calls {
                        let cm = object(&call)?;
                        let name = tool_name_for_encoding(&call)?;
                        encoded.push(format!(
                            "<{DSML_TOKEN}{TOOL_CALL_TAG_NAME} name=\"{name}\">\n{}\n</{DSML_TOKEN}{TOOL_CALL_TAG_NAME}>",
                            encode_arguments_to_dsml(&Value::Object(cm.clone()))?
                        ));
                    }
                    tool_calls = format!(
                        "\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{}\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>",
                        encoded.join("\n")
                    );
                }
            }
            let previous_has_task = index > 0
                && messages[index - 1]
                    .get("task")
                    .is_some_and(|v| !v.is_null());
            if options.thinking_mode == ThinkingMode::Thinking && !previous_has_task {
                let keep = !options.drop_thinking || index as isize > last_user;
                if keep {
                    reasoning.push_str(&text(message.get("reasoning_content")));
                    reasoning.push_str(THINKING_END_TOKEN);
                }
            }
            prompt.push_str(&reasoning);
            prompt.push_str(&text(message.get("content")));
            prompt.push_str(&tool_calls);
            if !message
                .get("wo_eos")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                prompt.push_str(EOS_TOKEN);
            }
        }
        other => {
            return Err(EncodingError::UnsupportedRole {
                index,
                role: other.to_string(),
            });
        }
    }

    // The assistant generation header belongs after a user/mid-system turn
    // when the following message is an assistant (or when this is the final
    // message).  A non-assistant next turn is a completed transition and
    // leaves the header to that later turn, matching the Python formatter.
    let next_is_non_assistant = index + 1 < messages.len()
        && !matches!(
            messages[index + 1].get("role").and_then(Value::as_str),
            Some("assistant" | "latest_reminder")
        );
    if next_is_non_assistant {
        return Ok(prompt);
    }
    if let Some(task) = message.get("task").and_then(Value::as_str) {
        let token = task_token(task)
            .ok_or_else(|| EncodingError::InvalidMessage(format!("invalid task '{task}'")))?;
        if task != "action" {
            prompt.push_str(token);
        } else {
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if options.thinking_mode == ThinkingMode::Thinking {
                THINKING_START_TOKEN
            } else {
                THINKING_END_TOKEN
            });
            prompt.push_str(token);
        }
    } else if message_role == "user" || (message_role == "system" && index > 0) {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        if options.thinking_mode == ThinkingMode::Thinking
            && (!options.drop_thinking || index as isize >= last_user)
        {
            prompt.push_str(THINKING_START_TOKEN);
        } else {
            prompt.push_str(THINKING_END_TOKEN);
        }
    }
    Ok(prompt)
}

fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last = last_user_index(messages);
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let role = message.get("role").and_then(Value::as_str).unwrap_or("");
            let keep_role = matches!(
                role,
                "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
            );
            if keep_role || index as isize >= last {
                return Some(message.clone());
            }
            if role == "assistant" {
                let mut copy = object(message).ok()?.clone();
                copy.remove("reasoning_content");
                Some(Value::Object(copy))
            } else {
                None
            }
        })
        .collect()
}

/// Render already preprocessed messages (tool messages merged, tool
/// results sorted, image blocks replaced) after `context`.
fn encode_messages_text(
    messages: &[Value],
    options: &EncodeOptions,
    context: &[Value],
) -> Result<String, EncodingError> {
    let mut full = context.to_vec();
    full.extend(messages.iter().cloned());
    let mut prompt = if options.add_default_bos_token && context.is_empty() {
        BOS_TOKEN.to_string()
    } else {
        String::new()
    };
    // Any message carrying tools keeps every turn's reasoning.
    let effective_drop = options.drop_thinking
        && !full
            .iter()
            .any(|m| m.get("tools").is_some_and(json_truthy));
    let (render_messages, render_count, context_len) =
        if options.thinking_mode == ThinkingMode::Thinking && effective_drop {
            let reduced_full = drop_thinking_messages(&full);
            let reduced_context = drop_thinking_messages(context);
            let count = reduced_full.len().saturating_sub(reduced_context.len());
            let context_len = reduced_full.len() - count;
            (reduced_full, count, context_len)
        } else {
            (full, messages.len(), context.len())
        };
    let render_options = EncodeOptions {
        drop_thinking: effective_drop,
        ..options.clone()
    };
    for index in 0..render_count {
        prompt.push_str(&render_message(
            index + context_len,
            &render_messages,
            &render_options,
        )?);
    }
    Ok(prompt)
}

/// Encode current messages. `images` contains only records belonging to the
/// current turn, matching the official helper's context-cache contract.
pub fn encode_messages(
    messages: &[Value],
    options: &EncodeOptions,
) -> Result<EncodedPrompt, EncodingError> {
    // Validate before merging can discard message-level textual fields.
    for message in options.context.iter().chain(messages) {
        validate_no_image_tokens(message)?;
    }
    // Collect images only after tool results are in their final prompt
    // order. The reference sorts context and current turn as one list (its
    // dicts are shared, so the context is sorted in place as well) and then
    // slices the current turn back off by the merged context length.
    let context = merge_tool_messages(&options.context)?;
    let context_len = context.len();
    let mut all = context;
    all.extend(merge_tool_messages(messages)?);
    sort_tool_results_by_call_order(&mut all);
    let current = all.split_off(context_len);
    let (processed_context, _) = if all.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        process_image_messages(&all)?
    };
    let (processed, images) = process_image_messages(&current)?;
    let prompt = encode_messages_text(&processed, options, &processed_context)?;
    Ok(EncodedPrompt { prompt, images })
}

pub fn encode_messages_simple(
    messages: &[Value],
    thinking_mode: ThinkingMode,
) -> Result<EncodedPrompt, EncodingError> {
    encode_messages(
        messages,
        &EncodeOptions {
            thinking_mode,
            ..EncodeOptions::default()
        },
    )
}

/// Normalize the reference's JSON case files (`encoding/tests/*.json`): a
/// single case object, a bare message list, or a list of cases. A
/// case-level `tools` list moves onto the first message, as the official
/// `load_cases` does.
pub fn load_cases(data: &Value) -> Result<Vec<Value>, EncodingError> {
    let cases: Vec<Value> = match data {
        Value::Object(_) => vec![data.clone()],
        Value::Array(items)
            if items
                .first()
                .is_some_and(|first| first.get("role").is_some()) =>
        {
            vec![json!({ "messages": items })]
        }
        Value::Array(items) => items.clone(),
        _ => {
            return Err(EncodingError::InvalidMessage(
                "case file must hold an object or a list".to_string(),
            ));
        }
    };
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        let map = object(&case)?;
        let mut messages = map
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| {
                EncodingError::InvalidMessage("case has no messages array".to_string())
            })?;
        if let Some(tools) = map.get("tools") {
            let first = messages.first_mut().ok_or_else(|| {
                EncodingError::InvalidMessage(
                    "a case with tools must contain at least one message".to_string(),
                )
            })?;
            let first = first.as_object_mut().ok_or_else(|| {
                EncodingError::InvalidMessage("expected a JSON object".to_string())
            })?;
            first.insert("tools".to_string(), tools.clone());
        }
        let field = |key: &str| map.get(key).cloned().unwrap_or(Value::Null);
        out.push(json!({
            "messages": messages,
            "context": field("context"),
            "thinking_mode": field("thinking_mode"),
            "reasoning_effort": field("reasoning_effort"),
        }));
    }
    Ok(out)
}

pub fn encode_case(
    case: &Value,
    default_mode: ThinkingMode,
) -> Result<EncodedPrompt, EncodingError> {
    let map = object(case)?;
    let messages = map
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| EncodingError::InvalidMessage("case has no messages array".to_string()))?;
    // `case.get("thinking_mode") or thinking_mode`: an empty value falls back.
    let mode = map
        .get("thinking_mode")
        .and_then(Value::as_str)
        .filter(|mode| !mode.is_empty())
        .map(ThinkingMode::parse)
        .transpose()?
        .unwrap_or(default_mode);
    let effort = map
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
        .map(ReasoningEffort::from_json)
        .transpose()?;
    let context = map
        .get("context")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    encode_messages(
        messages,
        &EncodeOptions {
            thinking_mode: mode,
            reasoning_effort: effort,
            context,
            ..EncodeOptions::default()
        },
    )
}

fn read_until_stop(index: usize, input: &str, stops: &[&str]) -> (usize, String, Option<String>) {
    let mut position = input.len();
    let mut matched = None;
    for stop in stops {
        if let Some(relative) = input[index..].find(stop) {
            let candidate = index + relative;
            if candidate < position {
                position = candidate;
                matched = Some((*stop).to_string());
            }
        }
    }
    match matched {
        Some(stop) => (
            position + stop.len(),
            input[index..position].to_string(),
            Some(stop),
        ),
        None => (input.len(), input[index..].to_string(), None),
    }
}

pub fn parse_tool_calls(
    mut index: usize,
    input: &str,
) -> Result<(usize, Option<String>, Vec<Value>), EncodingError> {
    let calls_end = format!("</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>");
    let call_start = format!("<{DSML_TOKEN}{TOOL_CALL_TAG_NAME}");
    let call_end = format!("</{DSML_TOKEN}{TOOL_CALL_TAG_NAME}");
    let parameter_start = format!("<{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}");
    let parameter_end = format!("/{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}");
    let mut calls = Vec::new();
    let mut last_stop = None;
    while index < input.len() {
        let (next, content, stop) = read_until_stop(index, input, &[&call_start, &calls_end]);
        index = next;
        last_stop = stop.clone();
        if content != ">\n" {
            return Err(EncodingError::InvalidCompletion(format!(
                "tool call header expected '>\\n', got {content:?}"
            )));
        }
        if stop.as_deref() == Some(calls_end.as_str()) {
            break;
        }
        if stop.is_none() {
            return Err(EncodingError::InvalidCompletion(
                "missing DSML tool-call tag".to_string(),
            ));
        }
        let (next, name_content, mut stop) =
            read_until_stop(index, input, &[&parameter_start, &call_end]);
        index = next;
        let name = name_content
            .strip_prefix(" name=\"")
            .and_then(|s| s.strip_suffix("\">\n"))
            .ok_or_else(|| {
                EncodingError::InvalidCompletion(format!(
                    "invalid tool name header {name_content:?}"
                ))
            })?;
        let mut args = Vec::new();
        while stop.as_deref() == Some(parameter_start.as_str()) {
            let (next, parameter, matched) = read_until_stop(index, input, &[&parameter_end]);
            index = next;
            let body = parameter.strip_prefix(" name=\"").ok_or_else(|| {
                EncodingError::InvalidCompletion("invalid parameter header".to_string())
            })?;
            let (name_end, rest) = body.split_once("\" string=\"").ok_or_else(|| {
                EncodingError::InvalidCompletion(format!("invalid parameter {parameter:?}"))
            })?;
            let (string, value) = rest.split_once("\">").ok_or_else(|| {
                EncodingError::InvalidCompletion(format!("invalid parameter {parameter:?}"))
            })?;
            let (value, close) = value.rsplit_once("<").ok_or_else(|| {
                EncodingError::InvalidCompletion(format!("invalid parameter {parameter:?}"))
            })?;
            if close != "" || !matches!(string, "true" | "false") {
                return Err(EncodingError::InvalidCompletion(format!(
                    "invalid parameter {parameter:?}"
                )));
            }
            if args
                .iter()
                .any(|(key, _, _): &(String, String, String)| key == name_end)
            {
                return Err(EncodingError::InvalidCompletion(format!(
                    "duplicate parameter {name_end}"
                )));
            }
            let (next, content, next_stop) =
                read_until_stop(index, input, &[&parameter_start, &call_end]);
            index = next;
            if content != ">\n" {
                return Err(EncodingError::InvalidCompletion(
                    "parameter separator expected '>\\n'".to_string(),
                ));
            }
            args.push((name_end.to_string(), value.to_string(), string.to_string()));
            stop = next_stop;
            if matched.is_none() {
                return Err(EncodingError::InvalidCompletion(
                    "unterminated parameter".to_string(),
                ));
            }
        }
        calls.push(decode_dsml_to_arguments(name, &args)?);
    }
    Ok((index, last_stop, calls))
}

fn tool_calls_to_openai(calls: &[Value]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|call| {
                let name = call
                    .get("name")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                let arguments = call
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                let mut function = Map::new();
                function.insert("name".to_string(), name);
                function.insert("arguments".to_string(), arguments);
                let mut result = Map::new();
                result.insert("type".to_string(), Value::String("function".to_string()));
                result.insert("function".to_string(), Value::Object(function));
                if let Some(namespace) = call.get("namespace") {
                    result.insert("namespace".to_string(), namespace.clone());
                }
                Value::Object(result)
            })
            .collect(),
    )
}

/// Parse a generated V4.1 completion into an OpenAI style assistant object.
pub fn parse_message_from_completion_text(
    input: &str,
    mode: ThinkingMode,
) -> Result<Value, EncodingError> {
    let tool_start = format!("\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}");
    let (mut index, reasoning, mut stop) = if mode == ThinkingMode::Thinking {
        let (next, content, matched) =
            read_until_stop(0, input, &[THINKING_END_TOKEN, &tool_start]);
        if matched.as_deref() != Some(THINKING_END_TOKEN) {
            return Err(EncodingError::InvalidCompletion(
                "thinking completion is missing </think>".to_string(),
            ));
        }
        (next, content, matched)
    } else {
        (0, String::new(), None)
    };
    let (next, summary, matched) = read_until_stop(index, input, &[EOS_TOKEN, &tool_start]);
    index = next;
    stop = matched;
    let mut calls = Vec::new();
    if stop.as_deref() == Some(tool_start.as_str()) {
        let (next, _, parsed_calls) = parse_tool_calls(index, input)?;
        index = next;
        calls = parsed_calls;
        let (next, tail, matched) = read_until_stop(index, input, &[EOS_TOKEN]);
        if !tail.is_empty() {
            return Err(EncodingError::InvalidCompletion(
                "content follows DSML calls".to_string(),
            ));
        }
        index = next;
        stop = matched;
    } else if stop.as_deref() != Some(EOS_TOKEN) {
        return Err(EncodingError::InvalidCompletion(
            "completion is missing EOS token".to_string(),
        ));
    }
    if index != input.len() {
        return Err(EncodingError::InvalidCompletion(
            "unexpected bytes after completion".to_string(),
        ));
    }
    for special in [
        BOS_TOKEN,
        EOS_TOKEN,
        THINKING_START_TOKEN,
        THINKING_END_TOKEN,
        DSML_TOKEN,
    ] {
        if summary.contains(special) || reasoning.contains(special) {
            return Err(EncodingError::InvalidCompletion(format!(
                "special token {special} leaked into content"
            )));
        }
    }
    Ok(json!({
        "role":"assistant",
        "content":summary,
        "reasoning_content":reasoning,
        "tool_calls":tool_calls_to_openai(&calls),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(content: Value) -> Value {
        json!({"role":"user", "content":content})
    }

    #[test]
    fn text_chat_and_numeric_reasoning_match_reference() {
        let chat = encode_messages(
            &[user(Value::String("hello".into()))],
            &EncodeOptions::default(),
        )
        .unwrap();
        assert_eq!(
            chat.prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
        let thinking = encode_messages(
            &[user(Value::String("question".into()))],
            &EncodeOptions {
                thinking_mode: ThinkingMode::Thinking,
                reasoning_effort: Some(ReasoningEffort::Budget(42)),
                ..EncodeOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            thinking.prompt,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 42 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>question<｜Assistant｜><think>"
        );
    }

    #[test]
    fn images_and_tool_tags_are_ordered_and_spaced() {
        let prompt = encode_messages(
            &[user(json!([
                {"type":"text","text":"inspect"},
                {"type":"image_url","image_url":{"url":"/tmp/a.png"}},
            ]))],
            &EncodeOptions::default(),
        )
        .unwrap();
        assert_eq!(
            prompt.images,
            vec![json!({"type":"image","url":"/tmp/a.png"})]
        );
        assert_eq!(
            prompt.prompt,
            "<｜begin▁of▁sentence｜><｜User｜>inspect\n\n<｜deepseek_image｜><｜Assistant｜></think>"
        );
        assert!(!prompt.prompt.contains("<｜DSML｜tool_calls>"));
    }

    #[test]
    fn mid_system_fixture_keeps_prior_reasoning_and_new_header() {
        let encoded = encode_messages(
            &[
                user(Value::String("old".into())),
                json!({
                    "role":"assistant",
                    "content":"old answer",
                    "reasoning_content":"private",
                    "wo_eos":true
                }),
                json!({"role":"system", "content":"new policy"}),
                user(Value::String("now".into())),
            ],
            &EncodeOptions {
                thinking_mode: ThinkingMode::Thinking,
                reasoning_effort: Some(ReasoningEffort::Budget(75)),
                ..EncodeOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            encoded.prompt,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>old<｜Assistant｜></think>old answer<｜System｜>new policy<｜User｜>now<｜Assistant｜><think>"
        );
    }

    #[test]
    fn spaced_dsml_round_trip() {
        let completion = "reason </think>answer\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
        let parsed =
            parse_message_from_completion_text(completion, ThinkingMode::Thinking).unwrap();
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\": \"Paris\"}"
        );
    }

    /// The official golden fixtures (`encoding/tests/test_input_*.json` ->
    /// `test_output_*.txt`, DeepSeek-V4.1-Flash-1017), encoded the way
    /// `test_examples_encoding_golden_outputs` does: `load_cases`, first case,
    /// `encode_case(case, "chat")`, byte-exact.
    #[test]
    fn reference_golden_fixtures_encode_byte_exact() {
        let cases: [(&str, &str); 5] = [
            (
                include_str!("../tests/fixtures/dsv41/encoding/test_input_1.json"),
                include_str!("../tests/fixtures/dsv41/encoding/test_output_1.txt"),
            ),
            (
                include_str!("../tests/fixtures/dsv41/encoding/test_input_2.json"),
                include_str!("../tests/fixtures/dsv41/encoding/test_output_2.txt"),
            ),
            (
                include_str!("../tests/fixtures/dsv41/encoding/test_input_3.json"),
                include_str!("../tests/fixtures/dsv41/encoding/test_output_3.txt"),
            ),
            (
                include_str!("../tests/fixtures/dsv41/encoding/test_input_4.json"),
                include_str!("../tests/fixtures/dsv41/encoding/test_output_4.txt"),
            ),
            (
                include_str!("../tests/fixtures/dsv41/encoding/test_input_5.json"),
                include_str!("../tests/fixtures/dsv41/encoding/test_output_5.txt"),
            ),
        ];
        for (i, (input, output)) in cases.iter().enumerate() {
            let data: Value = serde_json::from_str(input).unwrap();
            let case = load_cases(&data).unwrap().remove(0);
            let encoded = encode_case(&case, ThinkingMode::Chat).unwrap();
            assert_eq!(encoded.prompt, *output, "test_output_{}.txt", i + 1);
        }
        // Case 5 carries two images in prompt order.
        let data: Value = serde_json::from_str(cases[4].0).unwrap();
        let case = load_cases(&data).unwrap().remove(0);
        let images = encode_case(&case, ThinkingMode::Chat).unwrap().images;
        assert_eq!(
            images,
            vec![
                json!({"type":"image","url":"examples/images/carrots.jpeg"}),
                json!({"type":"image","url":"examples/images/corn.jpeg"}),
            ]
        );
    }

    /// `test_v41_tool_result_images_follow_prompt_order` (reference commit
    /// 314113f8): tool results are sorted into call order before image
    /// records are collected, for results given as content blocks, as tool
    /// messages with list content, as tool messages with `content_blocks`,
    /// and across the context boundary.
    #[test]
    fn tool_result_images_follow_prompt_order() {
        let history = vec![
            json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"history.png"}}]}),
            json!({"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"a","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"b","arguments":"{}"}},
            ]}),
        ];
        let results: Vec<Value> = ["b", "a"]
            .iter()
            .map(|name| {
                json!({"type":"tool_result","tool_use_id":name,"content":[
                    {"type":"text","text":format!("result {name}")},
                    {"type":"image_url","image_url":{"url":format!("{name}.png")}},
                ]})
            })
            .collect();
        for format in [
            "content_blocks",
            "tool_messages",
            "tool_content_blocks",
            "context",
        ] {
            let (messages, context) = match format {
                "tool_messages" | "tool_content_blocks" => {
                    let field = if format == "tool_content_blocks" {
                        "content_blocks"
                    } else {
                        "content"
                    };
                    let mut messages = history.clone();
                    for block in &results {
                        let mut m = Map::new();
                        m.insert("role".into(), json!("tool"));
                        m.insert("tool_call_id".into(), block["tool_use_id"].clone());
                        m.insert(field.into(), block["content"].clone());
                        messages.push(Value::Object(m));
                    }
                    (messages, Vec::new())
                }
                "context" => (
                    vec![json!({"role":"user","content_blocks":results})],
                    history.clone(),
                ),
                _ => {
                    let mut messages = history.clone();
                    messages.push(json!({"role":"user","content_blocks":results}));
                    (messages, Vec::new())
                }
            };
            let original = (messages.clone(), context.clone());
            let encoded = encode_messages(
                &messages,
                &EncodeOptions {
                    context: context.clone(),
                    ..EncodeOptions::default()
                },
            )
            .unwrap();
            assert!(
                encoded.prompt.ends_with(&format!(
                    "<｜User｜><tool_result>result a\n\n{IMAGE_PLACEHOLDER}</tool_result>\n\n\
                     <tool_result>result b\n\n{IMAGE_PLACEHOLDER}</tool_result><｜Assistant｜></think>"
                )),
                "{format}: {}",
                encoded.prompt
            );
            let urls: &[&str] = if context.is_empty() {
                &["history.png", "a.png", "b.png"]
            } else {
                &["a.png", "b.png"]
            };
            let expected: Vec<Value> = urls
                .iter()
                .map(|url| json!({"type":"image","url":url}))
                .collect();
            assert_eq!(encoded.images, expected, "{format}");
            assert_eq!((messages, context), original, "{format}: inputs mutated");
        }
    }

    /// `test_v41_drop_thinking_without_tools` (314113f8): completed turns
    /// lose their reasoning, the current turn keeps it across tool results,
    /// and a following user message closes that turn.
    #[test]
    fn current_turn_thinking_survives_tool_results() {
        let messages = vec![
            json!({"role":"user","content":"q1"}),
            json!({"role":"assistant","content":"a1","reasoning_content":"r1"}),
            json!({"role":"user","content":"q2"}),
            json!({"role":"assistant","reasoning_content":"r2","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}
            ]}),
            json!({"role":"tool","tool_call_id":"c1","content":"value"}),
        ];
        let options = EncodeOptions {
            thinking_mode: ThinkingMode::Thinking,
            drop_thinking: true,
            ..EncodeOptions::default()
        };
        let prompt = encode_messages(&messages, &options).unwrap().prompt;
        assert!(prompt.contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"));
        assert!(!prompt.contains("r1"));
        assert!(prompt.contains("<think>r2</think>"), "{prompt}");
        assert!(
            prompt.ends_with("<tool_result>value</tool_result><｜Assistant｜><think>"),
            "{prompt}"
        );
        let mut next = messages.clone();
        next.push(json!({"role":"user","content":"q3"}));
        let next_turn = encode_messages(&next, &options).unwrap().prompt;
        assert!(!next_turn.contains("r2"), "{next_turn}");
        assert!(next_turn.ends_with("q3<｜Assistant｜><think>"));
    }

    /// The reference reference-tests around the same change: placeholders in
    /// user or tool text are refused before merging; tool messages merge
    /// into one user message of tool_result blocks; a task token replaces the
    /// assistant header.
    #[test]
    fn reference_preprocessing_rules() {
        for role in ["user", "tool"] {
            let err = encode_messages(
                &[json!({"role":role,"content":format!("hi {IMAGE_PLACEHOLDER}")})],
                &EncodeOptions::default(),
            );
            assert!(err.is_err(), "{role} placeholder accepted");
        }
        let merged = merge_tool_messages(&[
            json!({"role":"assistant","content":"","tool_calls":[]}),
            json!({"role":"tool","tool_call_id":"a","content":"r1"}),
            json!({"role":"tool","tool_call_id":"b","content":"r2"}),
        ])
        .unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[1]["role"], "user");
        let kinds: Vec<&str> = merged[1]["content_blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["tool_result", "tool_result"]);
        let prompt = encode_messages(
            &[json!({"role":"user","content":"classify me","task":"query"})],
            &EncodeOptions::default(),
        )
        .unwrap()
        .prompt;
        assert!(prompt.ends_with("classify me<｜query｜>"));
        assert!(!prompt.contains(ASSISTANT_SP_TOKEN));
        // A user turn holding only tool results does not open a new turn.
        let only_tools = [
            json!({"role":"user","content":"q"}),
            json!({"role":"user","content_blocks":[{"type":"tool_result","tool_use_id":"x","content":"v"}]}),
        ];
        assert_eq!(last_user_index(&only_tools), 0);
    }

    #[test]
    fn effort_rejects_float_and_bool_json() {
        assert!(ReasoningEffort::from_json(&json!(1.5)).is_err());
        assert!(ReasoningEffort::from_json(&json!(true)).is_err());
        assert!(ReasoningEffort::from_json(&json!(0)).is_err());
        assert!(ReasoningEffort::from_json(&json!("42")).is_err());
    }
}
