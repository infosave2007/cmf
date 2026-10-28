//! Deterministic, text-only building blocks for `cortiq teacher-data generate`.
//!
//! This module deliberately does **not** own a command or an append-only
//! writer.  It is the small, testable boundary between JSONL prompt input and
//! the `.tdata` writer: parse/normalise prompts without changing their text,
//! render exactly as `cortiq run` would, derive stable record seeds/splits,
//! and construct a record plus the exact token sequence that generated it.
//!
//! The first vertical slice has no trustworthy runtime hook for teacher top-k
//! logits.  Asking it for `--topk > 0` is therefore an error instead of
//! writing invented or post-sampling probabilities.

use anyhow::{ensure, Context, Result};
use cortiq_engine::{tokenizer::Tokenizer, Pipeline};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::tdata::{
    TdataMessage, TdataQuality, TdataRecord, TdataResponse, TdataSplit, TdataTokenSpan,
};

/// The v1 command's intentionally conservative minimum.  It is lower than
/// the historical `tube_gen` threshold (20), so short but meaningful prompts
/// are not silently lost.
pub const DEFAULT_MIN_PROMPT_CHARS: usize = 8;

/// Generation top-k is not exposed until the engine can capture raw logits at
/// the exact generated positions (including MTP verification positions).
pub const SUPPORTED_TOPK: u32 = 0;

/// Reject a request that the initial generator cannot represent faithfully.
pub fn require_topk_disabled(topk: u32) -> Result<()> {
    ensure!(
        topk == SUPPORTED_TOPK,
        "teacher-data generate: --topk {topk} is not available yet; use --topk 0. \
         Capturing post-sampling or reconstructed probabilities would make invalid KD data."
    );
    Ok(())
}

/// The initial record builder can preserve exact generated ids, but it cannot
/// honestly infer a reasoning-token boundary from `GenerateResult::text`.
/// The first CLI path must therefore force (or require) `--no-think` until the
/// engine exposes an explicit boundary hook.
pub fn require_direct_response_mode(no_think: bool) -> Result<()> {
    ensure!(
        no_think,
        "teacher-data generate currently requires --no-think: without an engine reasoning-boundary hook, \
         recording a hidden <think> span as answer tokens would corrupt answer-only losses"
    );
    Ok(())
}

/// The id of the token that closes a reasoning block (`</think>`) when the
/// vocabulary has it as one special token — the exact boundary marker.
pub fn think_end_id(tokenizer: &Tokenizer) -> Option<u32> {
    let ids = tokenizer.encode("</think>");
    (ids.len() == 1).then(|| ids[0])
}

/// Exact reasoning / answer split of generated ids: everything through the
/// first `</think>` id is reasoning, the rest is the answer. Without one the
/// whole generation is an unfinished thought (answer empty).
pub fn split_think_ids(ids: &[u32], end: u32) -> (usize, usize) {
    match ids.iter().position(|&t| t == end) {
        Some(i) => (i + 1, ids.len() - i - 1),
        None => (ids.len(), 0),
    }
}

/// Online repetition detector for a running generation: every 64 emitted
/// pieces it measures the distinct 5-grams of the last 512, and trips once
/// at least 256 pieces are in and fewer than `min_distinct` of them are new
/// (a healthy text keeps far above 30 %, a collapsed one sits at 0–5 %).
pub struct LoopGuard {
    pieces: Vec<u64>,
    min_distinct: f32,
}

impl LoopGuard {
    pub fn new(min_distinct: f32) -> Self {
        Self { pieces: Vec::new(), min_distinct }
    }

    /// Feed one decoded piece; false = stop the generation.
    pub fn feed(&mut self, piece: &str) -> bool {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        piece.hash(&mut h);
        self.pieces.push(h.finish());
        let n = self.pieces.len();
        if n < 256 || n % 64 != 0 {
            return true;
        }
        let tail = &self.pieces[n.saturating_sub(512)..];
        let grams: HashSet<[u64; 5]> = tail.windows(5).map(|w| [w[0], w[1], w[2], w[3], w[4]]).collect();
        (grams.len() as f32 / (tail.len() - 4) as f32) >= self.min_distinct
    }
}

/// One answered conversation from a foreign answer cache (`teacher-data
/// import`): the prompt as messages, the exact prefix text the answer
/// followed when the cache kept it (ChatML caches), and the answer text with
/// its reasoning.
pub struct ImportedRow {
    pub prompt: TeacherPrompt,
    pub prefix_text: Option<String>,
    pub response_text: String,
}

/// Split a rendered ChatML conversation into messages, the prefix up to and
/// including the last `<|im_start|>assistant\n`, and the answer after it.
pub fn parse_chatml(rendered: &str) -> Option<(Vec<TdataMessage>, String, String)> {
    const START: &str = "<|im_start|>";
    const END: &str = "<|im_end|>";
    let marker = format!("{START}assistant\n");
    let cut = rendered.rfind(&marker)? + marker.len();
    let (prefix, answer) = rendered.split_at(cut);
    let mut messages = Vec::new();
    for block in prefix[..cut - marker.len()].split(START).skip(1) {
        let (role, rest) = block.split_once('\n')?;
        let content = rest.split(END).next().unwrap_or("").trim_end_matches('\n');
        messages.push(TdataMessage { role: role.trim().to_string(), content: Value::String(content.to_string()) });
    }
    if messages.is_empty() {
        return None;
    }
    let answer = answer.trim_end().trim_end_matches(END).to_string();
    Some((messages, prefix.to_string(), answer))
}

/// Read a foreign answer cache: a `.json` array of `[prompt, rendered ChatML]`
/// pairs (the DTG-MA notebooks' cache) or JSONL rows with `prompt` or
/// `messages` and `response` (optional `id`, `category`).
pub fn parse_import_source(path: &Path) -> Result<Vec<ImportedRow>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    let mut make = |obj: Value, line: usize, prefix: Option<String>, response: String| -> Result<()> {
        let prompt = parse_prompt_value(obj, line)?;
        out.push(ImportedRow { prompt, prefix_text: prefix, response_text: response });
        Ok(())
    };
    if path.extension().and_then(|e| e.to_str()) == Some("json") {
        let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let arr = v.as_array().context("expected a JSON array of [prompt, rendered] pairs")?;
        for (i, item) in arr.iter().enumerate() {
            let pair = item.as_array().filter(|p| p.len() == 2).with_context(|| format!("item {i}: expected [prompt, rendered]"))?;
            let rendered = pair[1].as_str().with_context(|| format!("item {i}: rendered text is not a string"))?;
            let Some((messages, prefix, answer)) = parse_chatml(rendered) else {
                anyhow::bail!("item {i}: not a ChatML conversation with an assistant turn");
            };
            let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
            make(json!({ "messages": msgs }), i + 1, Some(prefix), answer)?;
        }
    } else {
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let mut v: Value = serde_json::from_str(line).with_context(|| format!("line {}", i + 1))?;
            let obj = v.as_object_mut().with_context(|| format!("line {}: expected an object", i + 1))?;
            let response = obj
                .remove("response")
                .or_else(|| obj.remove("answer"))
                .and_then(|r| r.as_str().map(str::to_owned))
                .with_context(|| format!("line {}: no `response` string", i + 1))?;
            make(v, i + 1, None, response)?;
        }
    }
    Ok(out)
}

/// Options that affect input validation only.  They never rewrite a retained
/// prompt, which is important because a `.tdata` token prefix must remain
/// reproducible from its stored structured messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptParseOptions {
    /// A prompt/message whose visible text is shorter than this is counted and
    /// skipped, not treated as a malformed JSONL line.
    pub min_prompt_chars: usize,
}

impl Default for PromptParseOptions {
    fn default() -> Self {
        Self {
            min_prompt_chars: DEFAULT_MIN_PROMPT_CHARS,
        }
    }
}

/// One accepted JSONL input row.  `messages` already includes the optional
/// row-level `system` message, so it is precisely the conversation passed to
/// the renderer and later written to `records.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherPrompt {
    /// Stable prompt id: user-provided byte-for-byte, or a deterministic
    /// SHA-256-derived id when absent.
    pub id: String,
    pub category: Option<String>,
    pub messages: Vec<TdataMessage>,
    /// Per-prompt overrides retained for the future command orchestrator.
    pub samples: Option<u32>,
    pub max_tokens: Option<usize>,
    /// Copied without interpretation into each generated record.
    pub meta: Value,
    /// One-based physical JSONL line for actionable duplicate/error reports.
    pub source_line: usize,
    identity: String,
}

impl TeacherPrompt {
    /// The normalised identity used for duplicate detection.  This is not the
    /// text fed to the tokenizer: the original content in `messages` is.
    pub fn normalized_identity(&self) -> &str {
        &self.identity
    }

    /// Return the only representation that can safely be put in raw
    /// completion mode.  Multi-turn/system/structured messages must go
    /// through a carried chat template; flattening them would silently change
    /// what the model sees.
    fn raw_completion_text(&self) -> Option<&str> {
        if self.messages.len() != 1 || self.messages[0].role != "user" {
            return None;
        }
        self.messages[0].content.as_str()
    }
}

/// Results of parsing one JSONL file.  `prompts` retain source order; no
/// implicit sort means `--limit` has the same intuitive meaning as the input
/// file while per-record seeds/splits stay order-independent.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptLoad {
    pub prompts: Vec<TeacherPrompt>,
    pub source_lines: usize,
    pub duplicates: usize,
    pub skipped_short: usize,
}

impl PromptLoad {
    pub fn accepted(&self) -> usize {
        self.prompts.len()
    }
}

/// Parse a JSONL prompt file with the command's default v1 input rules.
///
/// `limit` is applied only after validating and de-duplicating the complete
/// input.  Thus a duplicate/short early row never shifts which real prompts
/// are selected.  A caller that needs a custom minimum can use
/// [`load_jsonl_prompts_with_options`].
pub fn load_jsonl_prompts(path: impl AsRef<Path>, limit: Option<usize>) -> Result<PromptLoad> {
    let mut loaded = load_jsonl_prompts_with_options(path, PromptParseOptions::default())?;
    if let Some(limit) = limit {
        loaded.prompts.truncate(limit);
    }
    Ok(loaded)
}

/// Parse a JSONL prompt file with explicit validation settings.
pub fn load_jsonl_prompts_with_options(
    path: impl AsRef<Path>,
    options: PromptParseOptions,
) -> Result<PromptLoad> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening prompts {}", path.display()))?;
    parse_jsonl_prompts(BufReader::new(file), options)
        .with_context(|| format!("parsing prompts {}", path.display()))
}

/// Reader form used by unit tests and by future directory input support.
pub fn parse_jsonl_prompts<R: BufRead>(
    mut reader: R,
    options: PromptParseOptions,
) -> Result<PromptLoad> {
    let mut prompts = Vec::new();
    let mut identities = HashSet::new();
    let mut ids: HashMap<String, usize> = HashMap::new();
    let mut source_lines = 0usize;
    let mut duplicates = 0usize;
    let mut skipped_short = 0usize;
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .context("reading JSONL prompt input")?;
        if n == 0 {
            break;
        }
        source_lines += 1;
        let line_no = source_lines;
        let text = line.trim_end_matches(['\n', '\r']);
        ensure!(
            !text.trim().is_empty(),
            "prompts line {line_no}: blank JSONL rows are not allowed"
        );
        let value: Value = serde_json::from_str(text)
            .with_context(|| format!("prompts line {line_no}: invalid JSON"))?;
        let prompt = parse_prompt_value(value, line_no)?;

        if visible_message_chars(&prompt.messages) < options.min_prompt_chars {
            skipped_short += 1;
            continue;
        }

        // De-duplicate conversation identity before checking the id.  Two
        // byte-identical/whitespace-equivalent rows without explicit ids get
        // the same derived id and are a normal input duplicate, not an
        // accidental "duplicate id" failure.
        if !identities.insert(prompt.identity.clone()) {
            duplicates += 1;
            continue;
        }
        if let Some(previous) = ids.insert(prompt.id.clone(), line_no) {
            anyhow::bail!(
                "prompts line {line_no}: duplicate id {:?}; first seen at line {previous}",
                prompt.id
            );
        }
        prompts.push(prompt);
    }

    Ok(PromptLoad {
        prompts,
        source_lines,
        duplicates,
        skipped_short,
    })
}

fn parse_prompt_value(value: Value, source_line: usize) -> Result<TeacherPrompt> {
    let context = format!("prompts line {source_line}");
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{context}: expected a JSON object"))?;
    let has_prompt = obj.contains_key("prompt");
    let has_messages = obj.contains_key("messages");
    ensure!(
        has_prompt ^ has_messages,
        "{context}: requires exactly one of \"prompt\" or \"messages\""
    );

    let system = optional_nonempty_string(obj, "system", &context)?;
    let plain_prompt_identity = if has_prompt {
        Some(normalize_prompt_identity(required_string(
            obj, "prompt", &context,
        )?))
    } else {
        None
    };
    let messages = if has_prompt {
        let prompt = required_string(obj, "prompt", &context)?.to_owned();
        let mut values = Vec::with_capacity(if system.is_some() { 2 } else { 1 });
        if let Some(system) = &system {
            values.push(TdataMessage {
                role: "system".to_string(),
                content: Value::String(system.clone()),
            });
        }
        values.push(TdataMessage {
            role: "user".to_string(),
            content: Value::String(prompt),
        });
        values
    } else {
        let mut values = parse_messages(required_value(obj, "messages", &context)?, &context)?;
        if let Some(system) = system {
            ensure!(
                !values.iter().any(|m| m.role == "system"),
                "{context}: a row with messages already containing role=system must not also set \"system\""
            );
            values.insert(
                0,
                TdataMessage {
                    role: "system".to_string(),
                    content: Value::String(system),
                },
            );
        }
        values
    };
    ensure!(
        !messages.is_empty(),
        "{context}: messages must not be empty"
    );

    // The documented simple case is specifically sha256(normalized prompt),
    // even when that row also supplies a system instruction.  The duplicate
    // key remains the full structured conversation, so a changed system
    // instruction is not silently discarded; two such id-less rows correctly
    // collide on the documented derived id and require explicit ids.
    let id_material =
        plain_prompt_identity.unwrap_or_else(|| normalised_messages_identity(&messages));
    let identity = normalised_messages_identity(&messages);
    let id = match obj.get("id") {
        None | Some(Value::Null) => derived_prompt_id(&id_material),
        Some(value) => {
            let id = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("{context}: field \"id\" must be a string"))?;
            ensure!(!id.is_empty(), "{context}: field \"id\" must not be empty");
            id.to_owned()
        }
    };
    let category = optional_nonempty_string(obj, "category", &context)?;
    let samples = optional_positive_u32(obj, "samples", &context)?;
    let max_tokens = optional_positive_usize(obj, "max_tokens", &context)?;
    let meta = obj
        .get("meta")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));

    // `messages` is intentionally not normalised or re-serialised: templates
    // and tokenizers consume the original user-visible text exactly.
    Ok(TeacherPrompt {
        id,
        category,
        messages,
        samples,
        max_tokens,
        meta,
        source_line,
        identity,
    })
}

fn required_value<'a>(obj: &'a Map<String, Value>, name: &str, context: &str) -> Result<&'a Value> {
    obj.get(name)
        .ok_or_else(|| anyhow::anyhow!("{context}: missing required field {name:?}"))
}

fn required_string<'a>(obj: &'a Map<String, Value>, name: &str, context: &str) -> Result<&'a str> {
    required_value(obj, name, context)?
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("{context}: field {name:?} must be a string"))
}

fn optional_nonempty_string(
    obj: &Map<String, Value>,
    name: &str,
    context: &str,
) -> Result<Option<String>> {
    match obj.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let value = value.as_str().ok_or_else(|| {
                anyhow::anyhow!("{context}: field {name:?} must be a string or null")
            })?;
            ensure!(
                !value.is_empty(),
                "{context}: field {name:?} must not be an empty string"
            );
            Ok(Some(value.to_owned()))
        }
    }
}

fn optional_positive_u32(
    obj: &Map<String, Value>,
    name: &str,
    context: &str,
) -> Result<Option<u32>> {
    let Some(value) = obj.get(name) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("{context}: field {name:?} must be a positive integer"))?;
    let value = u32::try_from(value)
        .map_err(|_| anyhow::anyhow!("{context}: field {name:?} is too large"))?;
    ensure!(value > 0, "{context}: field {name:?} must be positive");
    Ok(Some(value))
}

fn optional_positive_usize(
    obj: &Map<String, Value>,
    name: &str,
    context: &str,
) -> Result<Option<usize>> {
    let Some(value) = obj.get(name) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("{context}: field {name:?} must be a positive integer"))?;
    let value = usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("{context}: field {name:?} is too large"))?;
    ensure!(value > 0, "{context}: field {name:?} must be positive");
    Ok(Some(value))
}

fn parse_messages(value: &Value, context: &str) -> Result<Vec<TdataMessage>> {
    let values = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{context}: field \"messages\" must be an array"))?;
    ensure!(
        !values.is_empty(),
        "{context}: field \"messages\" must not be empty"
    );
    let mut messages = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let item_context = format!("{context}: messages[{index}]");
        let obj = value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{item_context} must be an object"))?;
        let role = required_string(obj, "role", &item_context)?.to_owned();
        ensure!(!role.is_empty(), "{item_context}: role must not be empty");
        let content = required_value(obj, "content", &item_context)?.clone();
        ensure!(
            !content.is_null(),
            "{item_context}: content must not be null"
        );
        messages.push(TdataMessage { role, content });
    }
    Ok(messages)
}

/// Whitespace-normalise an identity key while leaving stored prompt content
/// untouched.  Unicode whitespace is handled by Rust's stable
/// `char::is_whitespace`; Unicode NFC/NFKC is intentionally not forced,
/// because changing a model prompt's codepoints would break token parity.
pub fn normalize_prompt_identity(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalised_messages_identity(messages: &[TdataMessage]) -> String {
    if messages.len() == 1 && messages[0].role == "user" {
        if let Some(text) = messages[0].content.as_str() {
            // Exact documented basic case: absent id is sha256(normalized
            // prompt)[:12].
            return normalize_prompt_identity(text);
        }
    }

    // Multi-message/structured content needs a separator-safe, map-order
    // independent identity.  It is only a duplicate/id derivation key; the
    // original JSON Values remain in `messages`.
    let messages = Value::Array(
        messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect(),
    );
    canonical_json(&messages)
}

fn visible_message_chars(messages: &[TdataMessage]) -> usize {
    // A long system instruction must not smuggle an otherwise-empty user
    // prompt past --min-prompt-chars.  For an unusual conversation without a
    // user role, fall back to all visible content so it remains reviewable.
    let user_chars = messages
        .iter()
        .filter(|message| message.role == "user")
        .map(|message| visible_value_text(&message.content).chars().count())
        .sum::<usize>();
    if user_chars > 0 {
        user_chars
    } else {
        messages
            .iter()
            .map(|message| visible_value_text(&message.content).chars().count())
            .sum()
    }
}

fn visible_value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(visible_value_text)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(obj) => obj
            .get("text")
            .map(visible_value_text)
            .unwrap_or_else(|| canonical_json(value)),
        _ => canonical_json(value),
    }
}

/// Return the documented 12-hex-character prompt id for a normalised identity
/// key.  The input is not re-normalised so callers can use it for structured
/// identities as well.
pub fn derived_prompt_id(normalized_identity: &str) -> String {
    let digest = Sha256::digest(normalized_identity.as_bytes());
    let mut out = String::with_capacity(12);
    for byte in digest[..6].iter() {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Stable record id used as the append/resume key in `.tdata`.
pub fn record_id(prompt_id: &str, sample: u32) -> String {
    format!("{prompt_id}#{sample}")
}

/// Deterministically derive a per-record seed from exactly `(base, id,
/// sample)`.  Length-prefixing avoids ambiguous concatenations such as
/// `("ab", "c")` vs. `("a", "bc")`; the fixed domain separator protects
/// it from accidental reuse by unrelated CMF hashes.
pub fn record_seed(base_seed: u64, prompt_id: &str, sample: u32) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"cmf-teacher-data-v1-record-seed\0");
    hasher.update(base_seed.to_le_bytes());
    hasher.update((prompt_id.len() as u64).to_le_bytes());
    hasher.update(prompt_id.as_bytes());
    hasher.update(sample.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 is 32 bytes"))
}

/// A rational train/validation/test split.  Integer parts permit exact,
/// platform-independent rounding instead of binary-float thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitRatios {
    pub train: u32,
    pub val: u32,
    pub test: u32,
}

impl SplitRatios {
    pub const DEFAULT: Self = Self {
        train: 90,
        val: 5,
        test: 5,
    };

    pub fn new(train: u32, val: u32, test: u32) -> Result<Self> {
        let total = train as u64 + val as u64 + test as u64;
        ensure!(total > 0, "split ratios must contain a positive part");
        Ok(Self { train, val, test })
    }

    fn total(self) -> u64 {
        self.train as u64 + self.val as u64 + self.test as u64
    }
}

/// Hash used to order one category before assigning split ranges.  It follows
/// the v1 `split-v1 || split_seed || id` rule, with an explicit little-endian
/// 64-bit seed representation so separate implementations have no textual
/// formatting ambiguity.
pub fn split_key(split_seed: u64, prompt_id: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"split-v1");
    hasher.update(split_seed.to_le_bytes());
    hasher.update(prompt_id.as_bytes());
    hasher.finalize().into()
}

/// Assign a stratified split in the exact input order.  Each category is
/// independently ordered by [`split_key`]; categories with fewer than three
/// prompts remain train-only, as specified for v1 — unless that leaves the
/// whole set without a val or a test prompt (a short prompt list, where
/// every category's share rounds to zero): then the prompts are split as
/// one stratum, at least one val and one test, so a held-out evaluation
/// exists.
///
/// The caller writes the returned split into both `prompts.jsonl` and every
/// sample record, rather than recalculating it in consumers.
pub fn assign_stratified_splits(
    prompts: &[TeacherPrompt],
    ratios: SplitRatios,
    split_seed: u64,
) -> Result<Vec<TdataSplit>> {
    ensure!(
        ratios.total() > 0,
        "split ratios must contain a positive part"
    );
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, prompt) in prompts.iter().enumerate() {
        // An absent category is a real shared stratum, not one singleton per
        // prompt.  A leading NUL avoids colliding with user categories.
        let category = prompt
            .category
            .as_ref()
            .map(|value| format!("c:{value}"))
            .unwrap_or_else(|| "\0uncategorized".to_string());
        groups.entry(category).or_default().push(index);
    }

    let mut out = vec![TdataSplit::Train; prompts.len()];
    for members in groups.values_mut() {
        if members.len() < 3 {
            continue;
        }
        members.sort_unstable_by_key(|&index| split_key(split_seed, &prompts[index].id));
        let n = members.len();
        let test = rounded_share(n, ratios.test, ratios.total());
        let val = rounded_share(n, ratios.val, ratios.total()).min(n.saturating_sub(test));
        for &index in &members[..test] {
            out[index] = TdataSplit::Test;
        }
        for &index in &members[test..test + val] {
            out[index] = TdataSplit::Val;
        }
    }
    let has = |s: TdataSplit| out.contains(&s);
    if ratios.val > 0 && ratios.test > 0 && !(has(TdataSplit::Val) && has(TdataSplit::Test)) {
        let mut pool: Vec<usize> = (0..prompts.len()).collect();
        out.iter_mut().for_each(|s| *s = TdataSplit::Train);
        if pool.len() >= 3 {
            pool.sort_unstable_by_key(|&index| split_key(split_seed, &prompts[index].id));
            let n = pool.len();
            let test = rounded_share(n, ratios.test, ratios.total()).max(1);
            let val = rounded_share(n, ratios.val, ratios.total()).max(1).min(n - test - 1);
            for &index in &pool[..test] {
                out[index] = TdataSplit::Test;
            }
            for &index in &pool[test..test + val] {
                out[index] = TdataSplit::Val;
            }
        }
    }
    Ok(out)
}

fn rounded_share(count: usize, part: u32, total: u64) -> usize {
    let numerator = (count as u128) * (part as u128);
    let rounded = (numerator + (total as u128 / 2)) / total as u128;
    usize::try_from(rounded).unwrap_or(count)
}

/// Result of rendering one prompt.  `token_ids` includes the model's declared
/// BOS exactly once, just like `cortiq run`; it is the immutable prefix that
/// must be concatenated with `GenerateResult::token_ids`, never reconstructed
/// from decoded text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub token_ids: Vec<u32>,
    pub template_applied: bool,
    pub thinking: Option<bool>,
}

/// Render a prompt using the exact text-only rules of `cortiq run`:
///
/// * use the carried template only when it exists and `raw` is false;
/// * pass `enable_thinking=false` only for `no_think`;
/// * raw/base-model completion gets `with_bos(encode(text))`.
///
/// Structured/multi-turn rows cannot be faithfully flattened for a raw/base
/// model and are rejected instead of silently changing teacher data.
pub fn render_prompt(
    tokenizer: &Tokenizer,
    prompt: &TeacherPrompt,
    raw: bool,
    no_think: bool,
) -> Result<RenderedPrompt> {
    let use_template = tokenizer.chat_template.is_some() && !raw;
    let thinking = no_think.then_some(false);
    let token_ids = if use_template {
        if prompt
            .messages
            .iter()
            .all(|message| message.content.is_string())
        {
            let pairs: Vec<(String, String)> = prompt
                .messages
                .iter()
                .map(|message| {
                    (
                        message.role.clone(),
                        message
                            .content
                            .as_str()
                            .expect("checked is_string")
                            .to_owned(),
                    )
                })
                .collect();
            // Same template renderer that cmd_run uses for text-only turns,
            // but teacher data must not silently switch to a hardcoded ChatML
            // fallback when the carried template itself is broken.
            let rendered = tokenizer.render_chat_opts(&pairs, thinking).ok_or_else(|| {
                anyhow::anyhow!(
                    "prompt {} (line {}): carried chat template failed to render; refusing to write a fallback training prefix",
                    prompt.id,
                    prompt.source_line
                )
            })?;
            tokenizer.with_bos(tokenizer.encode(&rendered))
        } else {
            let messages: Vec<Value> = prompt
                .messages
                .iter()
                .map(|message| json!({"role": message.role, "content": message.content}))
                .collect();
            let rendered = tokenizer
                .render_chat_json(&messages, None, thinking)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "prompt {} (line {}): carried chat template failed to render structured messages; refusing to write a fallback training prefix",
                        prompt.id,
                        prompt.source_line
                    )
                })?;
            tokenizer.with_bos(tokenizer.encode(&rendered))
        }
    } else {
        let text = prompt.raw_completion_text().ok_or_else(|| {
            anyhow::anyhow!(
                "prompt {} (line {}): raw/base-model completion supports only one text user message; \
                 provide a CMF with a chat template or use a plain \"prompt\" row without \"system\"",
                prompt.id,
                prompt.source_line
            )
        })?;
        tokenizer.with_bos(tokenizer.encode(text))
    };
    ensure!(
        !token_ids.is_empty(),
        "prompt {} (line {}): rendering produced an empty token prefix",
        prompt.id,
        prompt.source_line
    );
    Ok(RenderedPrompt {
        token_ids,
        template_applied: use_template,
        thinking: if use_template { thinking } else { None },
    })
}

/// Convenience API for CLI wiring: render directly from the generation
/// pipeline without exposing tokenizer details at the command dispatch site.
pub fn render_prompt_ids(
    pipeline: &Pipeline,
    prompt: &TeacherPrompt,
    raw: bool,
    no_think: bool,
) -> Result<Vec<u32>> {
    Ok(render_prompt(&pipeline.tokenizer, prompt, raw, no_think)?.token_ids)
}

/// Response boundaries supplied by the generator.  `think_tokens` and
/// `answer_tokens` are deliberately explicit: decoding and then re-tokenizing
/// a `<think>` string would not be exact across special-token vocabularies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseParts {
    pub think: Option<String>,
    pub content: String,
    pub think_tokens: u32,
    pub answer_tokens: u32,
}

impl ResponseParts {
    /// Use this only for a direct-answer generation (e.g. `--no-think`).  It
    /// preserves every generated token as answer tokens and never pretends to
    /// infer hidden reasoning boundaries from lossy decoded text.
    pub fn direct(content: String, generated_token_count: usize) -> Result<Self> {
        let answer_tokens = u32::try_from(generated_token_count)
            .map_err(|_| anyhow::anyhow!("generated token count exceeds u32"))?;
        Ok(Self {
            think: None,
            content,
            think_tokens: 0,
            answer_tokens,
        })
    }

    /// Construct an explicitly segmented response once an engine hook reports
    /// exact token boundaries.  This is intentionally not guessed from text.
    pub fn segmented(
        think: Option<String>,
        content: String,
        think_tokens: u32,
        answer_tokens: u32,
    ) -> Self {
        Self {
            think,
            content,
            think_tokens,
            answer_tokens,
        }
    }

    fn validate_against(&self, generated_tokens: usize) -> Result<()> {
        let actual = self
            .think_tokens
            .checked_add(self.answer_tokens)
            .ok_or_else(|| anyhow::anyhow!("response token counts overflow u32"))?
            as usize;
        ensure!(
            actual == generated_tokens,
            "response token partition says {actual} generated tokens, but generation returned {generated_tokens}; \
             do not reconstruct token boundaries from text"
        );
        if self.think_tokens > 0 {
            ensure!(
                self.think.as_deref().is_some(),
                "response has think_tokens > 0 but no structured think text"
            );
        }
        Ok(())
    }
}

/// Quality state with a one-to-one `.tdata` index representation.  The first
/// writer derives flags from this; it is never inferred from timing data.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordQuality {
    pub usable: bool,
    pub truncated: bool,
    pub loop_suspect: bool,
    pub check: Option<Value>,
}

/// Summary of the deterministic final 5-gram tail check.  It is an
/// after-generation diagnostic only; it does not claim to be the streaming
/// early-stop detector required by a later E2 implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailDistinct5 {
    pub tail_tokens: usize,
    pub windows: usize,
    pub distinct: usize,
    pub ratio: f32,
}

/// Calculate the TZ's final tail diversity signal over at most the last 512
/// generated token ids.  A tail shorter than five tokens has no repeated
/// 5-gram evidence and deliberately receives ratio 1.0.
pub fn tail_distinct_5gram(ids: &[u32]) -> TailDistinct5 {
    const WINDOW: usize = 512;
    const NGRAM: usize = 5;
    let tail = &ids[ids.len().saturating_sub(WINDOW)..];
    let windows = tail.len().saturating_sub(NGRAM - 1);
    if windows == 0 {
        return TailDistinct5 {
            tail_tokens: tail.len(),
            windows: 0,
            distinct: 0,
            ratio: 1.0,
        };
    }
    let mut unique = HashSet::with_capacity(windows);
    for gram in tail.windows(NGRAM) {
        unique.insert([gram[0], gram[1], gram[2], gram[3], gram[4]]);
    }
    let distinct = unique.len();
    TailDistinct5 {
        tail_tokens: tail.len(),
        windows,
        distinct,
        ratio: distinct as f32 / windows as f32,
    }
}

/// Final (not online) low-diversity flag from §7.5: fewer than 30% distinct
/// 5-grams in the generated tail is suspicious.
pub fn tail_has_low_distinct_5gram(ids: &[u32]) -> bool {
    tail_distinct_5gram(ids).ratio < 0.30
}

/// Translate the engine's current finish spelling into the `.tdata` v1
/// vocabulary.  `GenerateResult` uses `max_tokens`; v1 records use `length`.
/// A visually empty decoded answer is always an unusable `empty` record even
/// when the model happened to emit an EOS/other special token first.
pub fn normalize_finish_reason(engine_finish: &str, visible_content: &str) -> String {
    // A generation cut by --max-tokens inside its reasoning has no answer
    // yet; it is a truncated record (kept with --keep-truncated: its
    // reasoning is real teacher text), not an empty one.
    if visible_content.trim().is_empty() && !matches!(engine_finish, "max_tokens" | "length") {
        return "empty".to_owned();
    }
    match engine_finish {
        "max_tokens" => "length".to_owned(),
        "stop" | "length" | "loop" | "cancelled" | "error" => engine_finish.to_owned(),
        // Preserve an unknown engine reason for provenance, but it remains
        // unusable below unless a future policy explicitly whitelists it.
        other => other.to_owned(),
    }
}

/// Apply the initial post-generation quality policy.  Cancellation/errors are
/// retained for audit but never selected for training.  The tail flag is also
/// conservative: it marks the record unusable, but does not fabricate a
/// `finish: loop` because no online loop stop occurred.
pub fn final_record_quality(
    engine_finish: &str,
    visible_content: &str,
    generated_ids: &[u32],
    keep_truncated: bool,
) -> (String, RecordQuality) {
    let finish = normalize_finish_reason(engine_finish, visible_content);
    let mut quality = RecordQuality::from_finish(&finish, keep_truncated);
    if tail_has_low_distinct_5gram(generated_ids) {
        quality.loop_suspect = true;
        quality.usable = false;
    }
    (finish, quality)
}

impl RecordQuality {
    pub fn accepted() -> Self {
        Self {
            usable: true,
            truncated: false,
            loop_suspect: false,
            check: None,
        }
    }

    pub fn from_finish(finish: &str, keep_truncated: bool) -> Self {
        match finish {
            "length" => Self {
                usable: keep_truncated,
                truncated: true,
                loop_suspect: false,
                check: None,
            },
            "loop" => Self {
                usable: false,
                truncated: false,
                loop_suspect: true,
                check: None,
            },
            "stop" => Self::accepted(),
            // A new/unknown engine finish must not silently become training
            // data before its semantics are reviewed and mapped explicitly.
            _ => Self {
                usable: false,
                truncated: false,
                loop_suspect: false,
                check: None,
            },
        }
    }
}

/// Inputs used to build a logical `.tdata` record.  The append-only writer is
/// authoritative for `tok.index` and `tok.offset`; this builder sets both to
/// zero as a clear unpositioned placeholder and validates all other invariants.
pub struct CanonicalRecordInput<'a> {
    pub prompt: &'a TeacherPrompt,
    pub split: TdataSplit,
    pub sample: u32,
    pub seed: u64,
    pub rendered: &'a RenderedPrompt,
    /// Exact `GenerateResult::token_ids`, not `tokenizer.encode(result.text)`.
    pub generated_token_ids: &'a [u32],
    pub response: ResponseParts,
    pub finish: &'a str,
    pub quality: RecordQuality,
}

/// Logical record plus the exact `tokens.bin` payload for one append.  Pass
/// `record` and `tokens` to `TdataWriter::append_record`; the writer stamps
/// append offsets/ordinals and serializes the final JSONL line.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRecord {
    pub record: TdataRecord,
    pub tokens: Vec<u32>,
}

impl PreparedRecord {
    /// Return an independently positioned clone for tests or a writer that
    /// assigns locations before append.
    pub fn with_location(&self, index: u64, offset: u64) -> TdataRecord {
        let mut record = self.record.clone();
        record.tok.index = index;
        record.tok.offset = offset;
        record
    }

    /// Construct the stable v1 JSON object explicitly, without depending on
    /// serde derives or allowing volatile timing fields into canonical data.
    pub fn canonical_json(&self) -> Value {
        canonical_record_json(&self.record)
    }
}

/// Concatenate the actual rendered prefix and actual generated ids.  This is
/// intentionally a separate helper because `GenerateResult::text` excludes
/// special tokens and cannot be safely used to rebuild a calibration sequence.
pub fn exact_full_token_ids(prompt_ids: &[u32], generated_ids: &[u32]) -> Result<Vec<u32>> {
    let total = prompt_ids
        .len()
        .checked_add(generated_ids.len())
        .ok_or_else(|| anyhow::anyhow!("full token sequence length overflows usize"))?;
    ensure!(
        total <= u32::MAX as usize,
        "full token sequence has {total} tokens; .tdata v1 stores per-record counts as u32"
    );
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(prompt_ids);
    out.extend_from_slice(generated_ids);
    Ok(out)
}

/// Build the non-top-k logical record for one completed generation.
///
/// No elapsed time, tok/s, GPU name, MTP acceptance, or other machine-specific
/// measurement is included here.  Keeping those outside the canonical record
/// is what allows a killed/restarted deterministic job to produce the same
/// finalized `.tdata` bytes.
pub fn build_canonical_record(input: CanonicalRecordInput<'_>) -> Result<PreparedRecord> {
    input
        .response
        .validate_against(input.generated_token_ids.len())?;
    let prompt = u32::try_from(input.rendered.token_ids.len())
        .map_err(|_| anyhow::anyhow!("prompt token count exceeds u32"))?;
    let generated = u32::try_from(input.generated_token_ids.len())
        .map_err(|_| anyhow::anyhow!("generated token count exceeds u32"))?;
    let total = prompt
        .checked_add(generated)
        .ok_or_else(|| anyhow::anyhow!("total token count exceeds u32"))?;
    let tokens = exact_full_token_ids(&input.rendered.token_ids, input.generated_token_ids)?;
    let quality = TdataQuality {
        truncated: Some(input.quality.truncated),
        loop_suspect: Some(input.quality.loop_suspect),
        raw_check: input.quality.check,
    };
    let record = TdataRecord {
        rid: record_id(&input.prompt.id, input.sample),
        pid: input.prompt.id.clone(),
        sample: input.sample,
        category: input.prompt.category.clone(),
        split: input.split,
        usable: input.quality.usable,
        messages: input.prompt.messages.clone(),
        response: TdataResponse {
            think: input.response.think,
            content: input.response.content,
        },
        finish: Some(input.finish.to_owned()),
        tok: TdataTokenSpan {
            index: 0,
            offset: 0,
            prompt,
            think: input.response.think_tokens,
            answer: input.response.answer_tokens,
            total,
        },
        topk: None,
        quality: Some(quality),
        // Only output-deterministic provenance is canonical in stage 1.
        generation: json!({"seed": input.seed}),
        meta: input.prompt.meta.clone(),
    };
    Ok(PreparedRecord { record, tokens })
}

/// Serialize one accepted prompt for the immutable `prompts.jsonl` ledger.
///
/// The generated records keep the same `messages` verbatim.  This avoids
/// storing a second, potentially lossy rendering and lets a future import or
/// retokenisation command re-render the conversation with a different model.
/// Optional per-prompt generation limits are provenance, not model output.
pub fn canonical_prompt_json(prompt: &TeacherPrompt, split: TdataSplit) -> Value {
    let messages = prompt
        .messages
        .iter()
        .map(|message| json!({"role": message.role, "content": message.content}))
        .collect::<Vec<_>>();
    let mut object = Map::new();
    object.insert("id".to_owned(), Value::String(prompt.id.clone()));
    if let Some(category) = &prompt.category {
        object.insert("category".to_owned(), Value::String(category.clone()));
    }
    object.insert("split".to_owned(), Value::String(split.as_str().to_owned()));
    object.insert("messages".to_owned(), Value::Array(messages));
    if let Some(samples) = prompt.samples {
        object.insert("samples".to_owned(), json!(samples));
    }
    if let Some(max_tokens) = prompt.max_tokens {
        object.insert("max_tokens".to_owned(), json!(max_tokens));
    }
    object.insert("meta".to_owned(), prompt.meta.clone());
    Value::Object(object)
}

/// Serialize a parsed/built v1 record into the canonical field shape used by
/// the writer.  There is intentionally no `topk` member when top-k is absent.
pub fn canonical_record_json(record: &TdataRecord) -> Value {
    let messages = record
        .messages
        .iter()
        .map(|message| json!({"role": message.role, "content": message.content}))
        .collect::<Vec<_>>();
    let mut response = Map::new();
    if let Some(think) = &record.response.think {
        response.insert("think".to_owned(), Value::String(think.clone()));
    }
    response.insert(
        "content".to_owned(),
        Value::String(record.response.content.clone()),
    );
    let mut object = Map::new();
    object.insert("rid".to_owned(), Value::String(record.rid.clone()));
    object.insert("pid".to_owned(), Value::String(record.pid.clone()));
    object.insert("sample".to_owned(), json!(record.sample));
    if let Some(category) = &record.category {
        object.insert("category".to_owned(), Value::String(category.clone()));
    }
    object.insert(
        "split".to_owned(),
        Value::String(record.split.as_str().to_owned()),
    );
    object.insert("usable".to_owned(), Value::Bool(record.usable));
    object.insert("messages".to_owned(), Value::Array(messages));
    object.insert("response".to_owned(), Value::Object(response));
    if let Some(finish) = &record.finish {
        object.insert("finish".to_owned(), Value::String(finish.clone()));
    }
    object.insert(
        "tok".to_owned(),
        json!({
            "index": record.tok.index,
            "offset": record.tok.offset,
            "prompt": record.tok.prompt,
            "think": record.tok.think,
            "answer": record.tok.answer,
            "total": record.tok.total,
        }),
    );
    if let Some(topk) = record.topk {
        object.insert(
            "topk".to_owned(),
            json!({"offset": topk.offset, "positions": topk.positions}),
        );
    }
    if let Some(quality) = &record.quality {
        object.insert(
            "quality".to_owned(),
            json!({
                "truncated": quality.truncated,
                "loop_suspect": quality.loop_suspect,
                "check": quality.raw_check,
            }),
        );
    }
    object.insert("gen".to_owned(), record.generation.clone());
    object.insert("meta".to_owned(), record.meta.clone());
    Value::Object(object)
}

/// Stable JSON text for an identity key.  Object keys are sorted recursively;
/// strings use serde_json's escaping so control characters cannot alter the
/// key's boundaries.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("string JSON serialization"),
        Value::Array(values) => {
            let inner = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            let inner = entries
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("object key JSON serialization"),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(input: &str) -> PromptLoad {
        parse_jsonl_prompts(Cursor::new(input.as_bytes()), PromptParseOptions::default()).unwrap()
    }

    #[test]
    fn calibration_style_rows_are_preserved_byte_for_byte() {
        let line = r#"{"id":"algo_dp_knapsack","category":"algorithms","prompt":"Write a dynamic programming solution for the 0/1 knapsack problem in Python. Include type hints."}"#;
        let loaded = parse(line);
        assert_eq!(loaded.source_lines, 1);
        assert_eq!(loaded.accepted(), 1);
        let prompt = &loaded.prompts[0];
        assert_eq!(prompt.id, "algo_dp_knapsack");
        assert_eq!(prompt.category.as_deref(), Some("algorithms"));
        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.messages[0].role, "user");
        assert_eq!(
            prompt.messages[0].content.as_str(),
            Some("Write a dynamic programming solution for the 0/1 knapsack problem in Python. Include type hints.")
        );
    }

    #[test]
    fn duplicate_normalized_prompt_is_dropped_but_reused_id_is_not_an_error() {
        let loaded = parse(
            r#"{"prompt":"Explain   stable hashing."}
{"prompt":" Explain stable\thashing. "}
{"id":"other","prompt":"A different question with enough text."}"#,
        );
        assert_eq!(loaded.accepted(), 2);
        assert_eq!(loaded.duplicates, 1);
        assert_eq!(
            loaded.prompts[0].id,
            derived_prompt_id("Explain stable hashing.")
        );
    }

    #[test]
    fn duplicate_id_for_distinct_prompts_reports_both_lines() {
        let error = parse_jsonl_prompts(
            Cursor::new(
                b"{\"id\":\"same\",\"prompt\":\"First valid prompt text.\"}\n{\"id\":\"same\",\"prompt\":\"Second valid prompt text.\"}\n",
            ),
            PromptParseOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("line 2"), "{error}");
        assert!(error.contains("line 1"), "{error}");
    }

    #[test]
    fn structured_identity_is_independent_of_json_object_key_order() {
        let a = parse(
            r#"{"messages":[{"role":"user","content":{"text":"Explain consistent hashing.","kind":"text"}}]}"#,
        );
        let b = parse(
            r#"{"messages":[{"content":{"kind":"text","text":"Explain consistent hashing."},"role":"user"}]}"#,
        );
        assert_eq!(a.prompts[0].id, b.prompts[0].id);
        assert_eq!(
            a.prompts[0].normalized_identity(),
            b.prompts[0].normalized_identity()
        );
    }

    #[test]
    fn record_seed_is_stable_and_domain_separated_by_tuple_parts() {
        assert_eq!(record_seed(42, "a", 7), record_seed(42, "a", 7));
        assert_ne!(record_seed(42, "a", 7), record_seed(42, "a", 8));
        assert_ne!(record_seed(42, "ab", 7), record_seed(42, "ba", 7));
        assert_ne!(record_seed(42, "a", 7), record_seed(43, "a", 7));
    }

    #[test]
    fn one_off_categories_still_get_val_and_test() {
        let rows: Vec<String> = (0..12)
            .map(|i| format!("{{\"id\":\"p{i}\",\"category\":\"c{i}\",\"prompt\":\"Prompt p{i} has enough characters.\"}}"))
            .collect();
        let loaded = parse(&rows.join("\n"));
        let s = assign_stratified_splits(&loaded.prompts, SplitRatios::new(80, 10, 10).unwrap(), 42).unwrap();
        assert!(s.contains(&TdataSplit::Val) && s.contains(&TdataSplit::Test));
        assert!(s.iter().filter(|&&x| x == TdataSplit::Train).count() >= 8);
    }

    #[test]
    fn split_is_category_stratified_and_input_order_independent() {
        let loaded = parse(
            r#"{"id":"a0","category":"a","prompt":"Prompt a0 has enough characters."}
{"id":"a1","category":"a","prompt":"Prompt a1 has enough characters."}
{"id":"a2","category":"a","prompt":"Prompt a2 has enough characters."}
{"id":"a3","category":"a","prompt":"Prompt a3 has enough characters."}
{"id":"b0","category":"b","prompt":"Prompt b0 has enough characters."}
{"id":"b1","category":"b","prompt":"Prompt b1 has enough characters."}"#,
        );
        let split =
            assign_stratified_splits(&loaded.prompts, SplitRatios::new(0, 50, 50).unwrap(), 11)
                .unwrap();
        // a has four rows, yielding two test + two val; b is a small
        // category and remains fully train.
        assert_eq!(split.iter().filter(|&&s| s == TdataSplit::Test).count(), 2);
        assert_eq!(split.iter().filter(|&&s| s == TdataSplit::Val).count(), 2);
        assert_eq!(split[4], TdataSplit::Train);
        assert_eq!(split[5], TdataSplit::Train);

        let mut reversed = loaded.prompts.clone();
        reversed.reverse();
        let reverse_split =
            assign_stratified_splits(&reversed, SplitRatios::new(0, 50, 50).unwrap(), 11).unwrap();
        let forward_by_id: BTreeMap<_, _> = loaded
            .prompts
            .iter()
            .zip(split)
            .map(|(p, s)| (p.id.clone(), s))
            .collect();
        let reverse_by_id: BTreeMap<_, _> = reversed
            .iter()
            .zip(reverse_split)
            .map(|(p, s)| (p.id.clone(), s))
            .collect();
        assert_eq!(forward_by_id, reverse_by_id);
    }

    #[test]
    fn topk_and_hidden_reasoning_are_refused_in_the_initial_path() {
        require_topk_disabled(0).unwrap();
        assert!(require_topk_disabled(1).is_err());
        require_direct_response_mode(true).unwrap();
        assert!(require_direct_response_mode(false).is_err());
    }

    #[test]
    fn raw_rendering_never_flattens_a_structured_conversation() {
        let tokenizer = Tokenizer::byte_level();
        let simple = parse(r#"{"prompt":"A simple prompt with enough text."}"#)
            .prompts
            .remove(0);
        let rendered = render_prompt(&tokenizer, &simple, true, true).unwrap();
        assert!(!rendered.template_applied);
        assert_eq!(rendered.thinking, None);

        let structured =
            parse(r#"{"system":"System text","prompt":"A structured prompt with enough text."}"#)
                .prompts
                .remove(0);
        let error = render_prompt(&tokenizer, &structured, true, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("raw/base-model completion"), "{error}");
    }

    #[test]
    fn finish_and_final_tail_quality_are_conservative() {
        assert_eq!(normalize_finish_reason("max_tokens", "answer"), "length");
        assert_eq!(normalize_finish_reason("stop", "   "), "empty");
        assert_eq!(normalize_finish_reason("max_tokens", ""), "length");
        assert!(!RecordQuality::from_finish("cancelled", false).usable);
        assert!(!RecordQuality::from_finish("unreviewed-finish", false).usable);

        let repeated = vec![9_u32; 512];
        let stats = tail_distinct_5gram(&repeated);
        assert_eq!(stats.tail_tokens, 512);
        assert_eq!(stats.distinct, 1);
        assert!(tail_has_low_distinct_5gram(&repeated));
        let (finish, quality) = final_record_quality("stop", "answer", &repeated, false);
        assert_eq!(finish, "stop");
        assert!(quality.loop_suspect);
        assert!(!quality.usable);
        assert!(!tail_has_low_distinct_5gram(&[1, 2, 3, 4]));
    }

    #[test]
    fn canonical_prompt_ledger_keeps_messages_and_split() {
        let prompt = parse(
            r#"{"id":"p","category":"test","samples":2,"max_tokens":64,"meta":{"source":"unit"},"prompt":"A prompt with enough text."}"#,
        )
        .prompts
        .remove(0);
        let json = canonical_prompt_json(&prompt, TdataSplit::Test);
        assert_eq!(json["id"], "p");
        assert_eq!(json["split"], "test");
        assert_eq!(json["messages"][0]["content"], "A prompt with enough text.");
        assert_eq!(json["samples"], 2);
        assert_eq!(json["max_tokens"], 64);
        assert_eq!(json["meta"]["source"], "unit");
    }

    #[test]
    fn record_keeps_exact_generated_ids_and_no_volatile_measurements() {
        let prompt = parse(r#"{"id":"p","prompt":"A prompt with enough text."}"#)
            .prompts
            .remove(0);
        let rendered = RenderedPrompt {
            token_ids: vec![1, 2, 3],
            template_applied: true,
            thinking: Some(false),
        };
        let generated = [4, 5, 6, 7];
        let record = build_canonical_record(CanonicalRecordInput {
            prompt: &prompt,
            split: TdataSplit::Val,
            sample: 2,
            seed: record_seed(42, &prompt.id, 2),
            rendered: &rendered,
            generated_token_ids: &generated,
            response: ResponseParts::direct("answer".to_owned(), generated.len()).unwrap(),
            finish: "stop",
            quality: RecordQuality::accepted(),
        })
        .unwrap();
        assert_eq!(record.tokens, vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(record.record.tok.prompt, 3);
        assert_eq!(record.record.tok.answer, 4);
        assert_eq!(record.record.tok.total, 7);
        assert!(record.record.topk.is_none());
        let json = record.canonical_json();
        assert_eq!(json["gen"]["seed"], record.record.generation["seed"]);
        assert!(json["gen"].get("ms").is_none());
        assert!(json.get("topk").is_none());
    }

    #[test]
    fn think_split_is_exact_at_the_first_close_token() {
        assert_eq!(split_think_ids(&[1, 2, 99, 5, 6, 7], 99), (3, 3));
        assert_eq!(split_think_ids(&[1, 2, 3], 99), (3, 0), "no </think>: an unfinished thought");
        assert_eq!(split_think_ids(&[99, 4], 99), (1, 1));
    }

    #[test]
    fn loop_guard_trips_on_repetition_only() {
        let mut g = LoopGuard::new(0.05);
        let healthy: Vec<String> = (0..2000).map(|i| format!("w{}", (i * 7919) % 1543)).collect();
        assert!(healthy.iter().all(|p| g.feed(p)), "varied text must never be stopped");
        let mut g = LoopGuard::new(0.05);
        let mut stopped = false;
        for i in 0..2000 {
            if !g.feed(["a", "b", "c"][i % 3]) {
                stopped = true;
                assert!(i >= 255, "stopped before 256 pieces");
                break;
            }
        }
        assert!(stopped, "a 3-token cycle must be stopped");
    }

    #[test]
    fn chatml_cache_rows_split_into_messages_prefix_and_answer() {
        let r = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nAdd 2 and 2.<|im_end|>\n<|im_start|>assistant\n<think>\nsum\n</think>\n\n4<|im_end|>\n";
        let (msgs, prefix, answer) = parse_chatml(r).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].content, Value::String("Add 2 and 2.".into()));
        assert!(prefix.ends_with("<|im_start|>assistant\n"));
        assert_eq!(answer, "<think>\nsum\n</think>\n\n4");
        assert!(parse_chatml("no chat here").is_none());
    }

    #[test]
    fn response_partition_mismatch_is_refused() {
        let prompt = parse(r#"{"id":"p","prompt":"A prompt with enough text."}"#)
            .prompts
            .remove(0);
        let rendered = RenderedPrompt {
            token_ids: vec![1],
            template_applied: false,
            thinking: None,
        };
        let error = build_canonical_record(CanonicalRecordInput {
            prompt: &prompt,
            split: TdataSplit::Train,
            sample: 0,
            seed: 1,
            rendered: &rendered,
            generated_token_ids: &[2, 3],
            response: ResponseParts::direct("x".to_owned(), 1).unwrap(),
            finish: "stop",
            quality: RecordQuality::accepted(),
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("partition"), "{error}");
    }
}
