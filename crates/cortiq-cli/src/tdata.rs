//! Streaming reader for the version-1 CMF teacher-data (`.tdata`) directory.
//!
//! A `.tdata` data set deliberately keeps its generated token sequences as
//! *records*, rather than as one calibration-text stream.  [`TdataStream`]
//! therefore yields one [`TdataSample`] at a time and never joins tokens from
//! neighbouring records.  Consumers that need fixed-size windows must make
//! their windows inside `sample.tokens`; crossing a record boundary would
//! make answer-only loss masks and per-prompt train/val/test splits invalid.
//!
//! This is the read-only E1 foundation.  It validates the immutable v1
//! directory layout while streaming, so opening a multi-million-token set is
//! O(1) memory and records are checked only as they are consumed.

use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The manifest identity required by the v1 on-disk format.
pub const TDATA_FORMAT: &str = "cmf-tdata";
/// The only version this reader accepts.
pub const TDATA_VERSION: u32 = 1;
/// One `tokens.idx` entry is exactly this many bytes.
pub const TDATA_INDEX_BYTES: usize = 40;
/// Bound a malformed record before it can make the streaming reader allocate
/// unbounded memory.  Normal teacher records are limited by `--max-tokens`
/// (8192 by default), so 16 Mi tokens is intentionally generous.
pub const MAX_TOKENS_PER_RECORD: u32 = 16 * 1024 * 1024;
const MAX_RECORD_JSONL_BYTES: usize = 16 * 1024 * 1024;

/// Train/validation/test split stored in both a JSONL record and its index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TdataSplit {
    Train,
    Val,
    Test,
}

impl TdataSplit {
    /// Decode the one-byte representation in `tokens.idx`.
    pub fn from_index_byte(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Train),
            1 => Ok(Self::Val),
            2 => Ok(Self::Test),
            _ => anyhow::bail!("tokens.idx: invalid split byte {value}; expected 0, 1 or 2"),
        }
    }

    /// The stable one-byte representation in `tokens.idx`.
    pub const fn index_byte(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Val => 1,
            Self::Test => 2,
        }
    }

    /// Parse the spelling used in `records.jsonl` and selectors.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "train" => Ok(Self::Train),
            "val" => Ok(Self::Val),
            "test" => Ok(Self::Test),
            other => anyhow::bail!("invalid .tdata split {other:?}; expected train, val or test"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

impl fmt::Display for TdataSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A selector's split component.  `all` is a selector-only value and is not
/// written to records or `tokens.idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TdataSplitSelector {
    Train,
    Val,
    Test,
    All,
}

impl TdataSplitSelector {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "train" => Ok(Self::Train),
            "val" => Ok(Self::Val),
            "test" => Ok(Self::Test),
            "all" => Ok(Self::All),
            other => anyhow::bail!(
                "invalid .tdata selector split {other:?}; expected train, val, test or all"
            ),
        }
    }

    pub const fn matches(self, split: TdataSplit) -> bool {
        match self {
            Self::Train => split.index_byte() == 0,
            Self::Val => split.index_byte() == 1,
            Self::Test => split.index_byte() == 2,
            Self::All => true,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
            Self::All => "all",
        }
    }
}

impl From<TdataSplit> for TdataSplitSelector {
    fn from(value: TdataSplit) -> Self {
        match value {
            TdataSplit::Train => Self::Train,
            TdataSplit::Val => Self::Val,
            TdataSplit::Test => Self::Test,
        }
    }
}

/// An exact decimal fraction used by `PATH.tdata[:split][=SHARE]`.
///
/// Keeping the source decimal as an integer ratio avoids platform-dependent
/// floating-point thresholding when a job is resumed on a different machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TdataShare {
    numerator: u64,
    denominator: u64,
}

impl TdataShare {
    pub const FULL: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// Parse a finite decimal in `(0, 1]`, such as `0.8`, `.25`, or `1`.
    /// Scientific notation is intentionally not accepted: the selector is
    /// persisted in command provenance and must have one canonical meaning.
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        ensure!(!input.is_empty(), "empty .tdata share after '='");
        ensure!(
            !input.starts_with('+') && !input.starts_with('-'),
            "invalid .tdata share {input:?}; expected a decimal in (0, 1]"
        );
        let mut pieces = input.split('.');
        let whole = pieces.next().unwrap_or_default();
        let fraction = pieces.next();
        ensure!(
            pieces.next().is_none(),
            "invalid .tdata share {input:?}; expected one decimal point at most"
        );
        ensure!(
            !whole.is_empty() || fraction.is_some(),
            "invalid .tdata share {input:?}; expected a decimal in (0, 1]"
        );
        ensure!(
            whole.is_empty() || whole.bytes().all(|b| b.is_ascii_digit()),
            "invalid .tdata share {input:?}; expected decimal digits"
        );
        let whole = if whole.is_empty() {
            0_u64
        } else {
            whole
                .parse::<u64>()
                .with_context(|| format!("invalid .tdata share {input:?}"))?
        };
        let fraction = fraction.unwrap_or_default();
        ensure!(
            fraction.bytes().all(|b| b.is_ascii_digit()),
            "invalid .tdata share {input:?}; expected decimal digits"
        );
        ensure!(
            fraction.len() <= 18,
            "invalid .tdata share {input:?}; at most 18 fractional digits are supported"
        );
        let denominator = 10_u64.pow(fraction.len() as u32);
        let fractional = if fraction.is_empty() {
            0
        } else {
            fraction
                .parse::<u64>()
                .with_context(|| format!("invalid .tdata share {input:?}"))?
        };
        ensure!(
            whole <= 1 && (whole == 0 || fractional == 0),
            "invalid .tdata share {input:?}; expected a decimal in (0, 1]"
        );
        let numerator = whole
            .checked_mul(denominator)
            .and_then(|n| n.checked_add(fractional))
            .ok_or_else(|| anyhow::anyhow!("invalid .tdata share {input:?}"))?;
        ensure!(
            numerator > 0 && numerator <= denominator,
            "invalid .tdata share {input:?}; expected a decimal in (0, 1]"
        );
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub const fn is_full(self) -> bool {
        self.numerator == self.denominator
    }

    /// Test a stable record key against this fraction.  The SHA-256 domain
    /// separator makes the selected subset independent from other CMF hashes.
    pub fn contains_rid(self, rid: &str) -> bool {
        if self.is_full() {
            return true;
        }
        let mut hasher = Sha256::new();
        hasher.update(b"cmf-tdata-v1-share\0");
        hasher.update(rid.as_bytes());
        let digest = hasher.finalize();
        let value = u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has 32 bytes"));
        let threshold = ((self.numerator as u128) << 64) / self.denominator as u128;
        (value as u128) < threshold
    }

    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    pub const fn denominator(self) -> u64 {
        self.denominator
    }
}

impl fmt::Display for TdataShare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_full() {
            return f.write_str("1");
        }
        if self.denominator == 1 {
            return write!(f, "{}", self.numerator);
        }
        // This is only diagnostic output; preserving the input spelling is
        // not needed because equivalent decimals select the same records.
        let digits = self.denominator.ilog10() as usize;
        write!(f, "0.{:0digits$}", self.numerator, digits = digits)
    }
}

/// Parsed `PATH.tdata[:split][=SHARE]` selector (§8.7 of the teacher-data
/// specification).  It defaults to usable train records and the full share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdataSelector {
    pub root: PathBuf,
    pub split: TdataSplitSelector,
    pub share: TdataShare,
    /// All data-consuming commands use only usable records by default.
    /// `teacher-data inspect` can turn this off to see rejected rows too.
    pub usable_only: bool,
}

impl TdataSelector {
    pub fn parse(input: &str) -> Result<Self> {
        let marker = input.rfind(".tdata").ok_or_else(|| {
            anyhow::anyhow!(".tdata selector {input:?} has no '.tdata' directory")
        })?;
        let root_end = marker + ".tdata".len();
        let (root, suffix) = input.split_at(root_end);
        ensure!(
            !root.is_empty(),
            ".tdata selector {input:?} has an empty directory path"
        );

        let (split, share) = if suffix.is_empty() {
            (TdataSplitSelector::Train, TdataShare::FULL)
        } else if let Some(share) = suffix.strip_prefix('=') {
            (TdataSplitSelector::Train, TdataShare::parse(share)?)
        } else if let Some(rest) = suffix.strip_prefix(':') {
            let (split, share) = match rest.split_once('=') {
                Some((split, share)) => {
                    (TdataSplitSelector::parse(split)?, TdataShare::parse(share)?)
                }
                None => (TdataSplitSelector::parse(rest)?, TdataShare::FULL),
            };
            (split, share)
        } else {
            anyhow::bail!(
                "invalid .tdata selector suffix {suffix:?}; expected :train, :val, :test, :all, or =SHARE"
            );
        };
        Ok(Self {
            root: PathBuf::from(root),
            split,
            share,
            usable_only: true,
        })
    }

    pub fn include_rejected(mut self) -> Self {
        self.usable_only = false;
        self
    }

    fn matches(&self, record: &TdataRecord) -> bool {
        (!self.usable_only || record.usable)
            && self.split.matches(record.split)
            && self.share.contains_rid(&record.rid)
    }
}

/// The small manifest subset needed to safely open a v1 directory.  The
/// original JSON remains available for provenance and future fields.
#[derive(Debug, Clone)]
pub struct TdataManifest {
    pub format: String,
    pub version: u32,
    pub complete: bool,
    /// `stats.records` when the producer supplied it.  It is checked after a
    /// full stream reaches EOF, not at `open`, so `open` stays constant-memory.
    pub expected_records: Option<u64>,
    pub raw: Value,
}

impl TdataManifest {
    pub fn from_json(value: Value) -> Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("manifest.json: expected a JSON object"))?;
        let format = required_string(obj, "format", "manifest.json")?.to_owned();
        ensure!(
            format == TDATA_FORMAT,
            "manifest.json: format {format:?}; expected {TDATA_FORMAT:?}"
        );
        let version = required_u32(obj, "version", "manifest.json")?;
        ensure!(
            version == TDATA_VERSION,
            "manifest.json: version {version}; this reader supports only v{TDATA_VERSION}"
        );
        let complete = required_bool(obj, "complete", "manifest.json")?;
        ensure!(
            complete,
            "manifest.json: data set is incomplete; refuse to consume append-only data before its final manifest"
        );
        let expected_records = obj
            .get("stats")
            .and_then(Value::as_object)
            .and_then(|stats| stats.get("records"))
            .map(|value| json_u64(value, "manifest.json.stats.records"))
            .transpose()?;
        Ok(Self {
            format,
            version,
            complete,
            expected_records,
            raw: value,
        })
    }
}

/// Chat message stored structurally in `records.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub struct TdataMessage {
    pub role: String,
    /// Kept as JSON because future v1-compatible data may use structured
    /// multimodal content.  Text-only producers store a JSON string here.
    pub content: Value,
}

/// Generated response fields required by v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdataResponse {
    pub think: Option<String>,
    pub content: String,
}

/// The JSON token span attached to a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdataTokenSpan {
    pub index: u64,
    /// Offset in *tokens*, not bytes.
    pub offset: u64,
    pub prompt: u32,
    pub think: u32,
    pub answer: u32,
    pub total: u32,
}

impl TdataTokenSpan {
    fn validate(self, context: &str) -> Result<()> {
        let parts = (self.prompt as u64)
            .checked_add(self.think as u64)
            .and_then(|sum| sum.checked_add(self.answer as u64))
            .ok_or_else(|| anyhow::anyhow!("{context}: token counts overflow"))?;
        ensure!(
            parts == self.total as u64,
            "{context}: tok.total={} but prompt + think + answer={parts}",
            self.total
        );
        ensure!(
            self.total <= MAX_TOKENS_PER_RECORD,
            "{context}: tok.total={} exceeds MAX_TOKENS_PER_RECORD={MAX_TOKENS_PER_RECORD}",
            self.total
        );
        Ok(())
    }
}

/// Location of optional teacher top-k distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdataTopkRef {
    /// Offset in top-k positions, not bytes.
    pub offset: u64,
    pub positions: u32,
}

/// The quality booleans that have a one-to-one representation in index flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdataQuality {
    pub truncated: Option<bool>,
    pub loop_suspect: Option<bool>,
    pub raw_check: Option<Value>,
}

/// One parsed `records.jsonl` row.  Fields unrelated to safe streaming remain
/// in `meta`/`gen`; their detailed interpretation belongs to later commands.
#[derive(Debug, Clone, PartialEq)]
pub struct TdataRecord {
    pub rid: String,
    pub pid: String,
    pub sample: u32,
    pub category: Option<String>,
    pub split: TdataSplit,
    pub usable: bool,
    pub messages: Vec<TdataMessage>,
    pub response: TdataResponse,
    pub finish: Option<String>,
    pub tok: TdataTokenSpan,
    pub topk: Option<TdataTopkRef>,
    pub quality: Option<TdataQuality>,
    /// Raw optional `gen` object from the JSONL row.  `gen` is a reserved
    /// identifier in Rust 2024, hence the descriptive Rust field name.
    pub generation: Value,
    pub meta: Value,
}

impl TdataRecord {
    /// Parse a complete non-empty JSONL row.  `line_no` is only used to make
    /// a failed data set actionable to its operator.
    pub fn from_json_line(line: &str, line_no: usize) -> Result<Self> {
        let context = format!("records.jsonl line {line_no}");
        let value: Value = serde_json::from_str(line).with_context(|| context.clone())?;
        let obj = value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{context}: expected a JSON object"))?;
        let rid = required_string(obj, "rid", &context)?.to_owned();
        ensure!(!rid.is_empty(), "{context}: rid is empty");
        let pid = required_string(obj, "pid", &context)?.to_owned();
        ensure!(!pid.is_empty(), "{context}: pid is empty");
        let sample = required_u32(obj, "sample", &context)?;
        let category = optional_string(obj, "category", &context)?;
        let split = TdataSplit::parse(required_string(obj, "split", &context)?)
            .with_context(|| context.clone())?;
        let usable = required_bool(obj, "usable", &context)?;
        let messages = parse_messages(required_value(obj, "messages", &context)?, &context)?;
        let response = parse_response(required_value(obj, "response", &context)?, &context)?;
        let finish = optional_string(obj, "finish", &context)?;
        let tok = parse_token_span(required_value(obj, "tok", &context)?, &context)?;
        tok.validate(&context)?;
        let topk = parse_topk(obj.get("topk"), &context)?;
        let quality = parse_quality(obj.get("quality"), &context)?;
        Ok(Self {
            rid,
            pid,
            sample,
            category,
            split,
            usable,
            messages,
            response,
            finish,
            tok,
            topk,
            quality,
            generation: obj.get("gen").cloned().unwrap_or(Value::Null),
            meta: obj.get("meta").cloned().unwrap_or(Value::Null),
        })
    }
}

/// Bit flags stored in the `flags` u32 of a [`TdataTokenIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TdataIndexFlags(pub u32);

impl TdataIndexFlags {
    pub const USABLE: u32 = 1 << 0;
    pub const TRUNCATED: u32 = 1 << 1;
    pub const LOOP: u32 = 1 << 2;
    pub const CHECK_FAILED: u32 = 1 << 3;
    const KNOWN: u32 = Self::USABLE | Self::TRUNCATED | Self::LOOP | Self::CHECK_FAILED;

    pub const fn usable(self) -> bool {
        self.0 & Self::USABLE != 0
    }

    pub const fn truncated(self) -> bool {
        self.0 & Self::TRUNCATED != 0
    }

    pub const fn loop_suspect(self) -> bool {
        self.0 & Self::LOOP != 0
    }

    pub const fn check_failed(self) -> bool {
        self.0 & Self::CHECK_FAILED != 0
    }

    fn validate(self, context: &str) -> Result<()> {
        ensure!(
            self.0 & !Self::KNOWN == 0,
            "{context}: unknown flags 0x{:08x} in .tdata v1 index",
            self.0 & !Self::KNOWN
        );
        Ok(())
    }
}

/// Fixed-size `tokens.idx` entry.  Its wire representation is manually
/// encoded; it is never Rust-layout- or bincode-dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdataTokenIndex {
    pub tok_offset: u64,
    pub total: u32,
    pub prompt: u32,
    pub think: u32,
    pub answer: u32,
    pub topk_offset: u64,
    pub flags: TdataIndexFlags,
    pub split: TdataSplit,
}

impl TdataTokenIndex {
    pub const NO_TOPK: u64 = u64::MAX;

    /// Decode exactly one 40-byte v1 index entry.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == TDATA_INDEX_BYTES,
            "tokens.idx: expected {TDATA_INDEX_BYTES}-byte record, got {} bytes",
            bytes.len()
        );
        let tok_offset = u64::from_le_bytes(bytes[0..8].try_into().expect("fixed slice"));
        let total = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed slice"));
        let prompt = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed slice"));
        let think = u32::from_le_bytes(bytes[16..20].try_into().expect("fixed slice"));
        let answer = u32::from_le_bytes(bytes[20..24].try_into().expect("fixed slice"));
        let topk_offset = u64::from_le_bytes(bytes[24..32].try_into().expect("fixed slice"));
        let flags = TdataIndexFlags(u32::from_le_bytes(
            bytes[32..36].try_into().expect("fixed slice"),
        ));
        let split = TdataSplit::from_index_byte(bytes[36])?;
        ensure!(
            bytes[37..40] == [0, 0, 0],
            "tokens.idx: non-zero reserved bytes in v1 entry"
        );
        let index = Self {
            tok_offset,
            total,
            prompt,
            think,
            answer,
            topk_offset,
            flags,
            split,
        };
        index.validate("tokens.idx")?;
        Ok(index)
    }

    /// Encode exactly one stable 40-byte little-endian v1 index entry.  This
    /// is useful to the later append-only writer and lets tests pin the ABI.
    pub fn to_le_bytes(self) -> Result<[u8; TDATA_INDEX_BYTES]> {
        self.validate("tokens.idx")?;
        let mut bytes = [0_u8; TDATA_INDEX_BYTES];
        bytes[0..8].copy_from_slice(&self.tok_offset.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.total.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.prompt.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.think.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.answer.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.topk_offset.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.flags.0.to_le_bytes());
        bytes[36] = self.split.index_byte();
        Ok(bytes)
    }

    fn validate(self, context: &str) -> Result<()> {
        TdataTokenSpan {
            index: 0,
            offset: self.tok_offset,
            prompt: self.prompt,
            think: self.think,
            answer: self.answer,
            total: self.total,
        }
        .validate(context)?;
        self.flags.validate(context)
    }
}

/// A record plus its own token vector.  There is intentionally no flattened
/// `Iterator<Item = u32>` API: callers cannot accidentally build calibration
/// windows that straddle two different prompts.
#[derive(Debug, Clone, PartialEq)]
pub struct TdataSample {
    pub record: TdataRecord,
    pub index: TdataTokenIndex,
    pub tokens: Vec<u32>,
}

/// One split's counts as seen by [`TdataInspectSummary`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TdataSplitSummary {
    pub records: u64,
    pub usable: u64,
    pub rejected: u64,
    pub tokens: u64,
}

/// Streaming inspection result for a finalized data set.  `tokens` includes
/// complete rendered sequences (prompt + reasoning + answer), matching the
/// `tokens.bin` accounting.  `inspect` validates every token payload while
/// producing these counts; it does not trust manifest statistics alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TdataInspectSummary {
    pub records: u64,
    pub usable: u64,
    pub rejected: u64,
    pub tokens: u64,
    pub train: TdataSplitSummary,
    pub val: TdataSplitSummary,
    pub test: TdataSplitSummary,
}

impl TdataInspectSummary {
    fn add(&mut self, record: &TdataRecord, index: TdataTokenIndex) {
        self.records += 1;
        self.tokens += index.total as u64;
        let split = match record.split {
            TdataSplit::Train => &mut self.train,
            TdataSplit::Val => &mut self.val,
            TdataSplit::Test => &mut self.test,
        };
        split.records += 1;
        split.tokens += index.total as u64;
        if record.usable {
            self.usable += 1;
            split.usable += 1;
        } else {
            self.rejected += 1;
            split.rejected += 1;
        }
    }
}

/// Read-only opened `.tdata` directory.  Calling [`Self::stream`] opens fresh
/// file handles, allowing multiple independent streaming passes.
#[derive(Debug, Clone)]
pub struct TdataReader {
    root: PathBuf,
    pub manifest: TdataManifest,
    index_entries: u64,
    token_count: u64,
}

impl TdataReader {
    /// Open a finalized v1 directory.  An unfinished manifest is rejected:
    /// JSONL is the writer's commit marker and consuming it before finalizing
    /// would otherwise turn a crash tail into silent training data.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref();
        ensure!(
            root.is_dir(),
            "{}: expected a .tdata directory",
            root.display()
        );
        let manifest_path = root.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest_value: Value = serde_json::from_str(&manifest_text)
            .with_context(|| format!("parsing {}", manifest_path.display()))?;
        let manifest = TdataManifest::from_json(manifest_value)?;

        let records_path = root.join("records.jsonl");
        let index_path = root.join("tokens.idx");
        let tokens_path = root.join("tokens.bin");
        // Open every required file now.  This produces an actionable failure
        // before a costly conversion pipeline starts.
        let _ = File::open(&records_path)
            .with_context(|| format!("opening {}", records_path.display()))?;
        let index_len = fs::metadata(&index_path)
            .with_context(|| format!("stat {}", index_path.display()))?
            .len();
        ensure!(
            index_len % TDATA_INDEX_BYTES as u64 == 0,
            "{}: truncated/corrupt file: {index_len} bytes is not divisible by {TDATA_INDEX_BYTES}",
            index_path.display()
        );
        let token_bytes = fs::metadata(&tokens_path)
            .with_context(|| format!("stat {}", tokens_path.display()))?
            .len();
        ensure!(
            token_bytes % 4 == 0,
            "{}: truncated/corrupt file: {token_bytes} bytes is not divisible by 4",
            tokens_path.display()
        );

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            index_entries: index_len / TDATA_INDEX_BYTES as u64,
            token_count: token_bytes / 4,
        })
    }

    /// Parse and open a `PATH.tdata[:split][=SHARE]` expression in one step.
    pub fn open_selector(spec: &str) -> Result<(Self, TdataSelector)> {
        let selector = TdataSelector::parse(spec)?;
        let reader = Self::open(&selector.root)?;
        Ok((reader, selector))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_entries(&self) -> u64 {
        self.index_entries
    }

    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Stream selected records.  The records and 40-byte index are consumed
    /// in lockstep even when a row is filtered out, so malformed skipped rows
    /// cannot hide behind `:train` or `=SHARE`.
    pub fn stream(&self, selector: TdataSelector) -> Result<TdataStream> {
        let records_path = self.root.join("records.jsonl");
        let index_path = self.root.join("tokens.idx");
        let tokens_path = self.root.join("tokens.bin");
        Ok(TdataStream {
            records: BufReader::new(
                File::open(&records_path)
                    .with_context(|| format!("opening {}", records_path.display()))?,
            ),
            index: File::open(&index_path)
                .with_context(|| format!("opening {}", index_path.display()))?,
            tokens: File::open(&tokens_path)
                .with_context(|| format!("opening {}", tokens_path.display()))?,
            selector,
            index_entries: self.index_entries,
            token_count: self.token_count,
            expected_records: self.manifest.expected_records,
            ordinal: 0,
            expected_tok_offset: 0,
            done: false,
        })
    }

    /// Validate all rows and payloads.  It is deliberately a separate full
    /// pass: regular consumers stay streaming and only pay for selected
    /// token vectors.
    pub fn verify(&self) -> Result<()> {
        let selector = TdataSelector {
            root: self.root.clone(),
            split: TdataSplitSelector::All,
            share: TdataShare::FULL,
            usable_only: false,
        };
        for item in self.stream(selector)? {
            item?;
        }
        Ok(())
    }

    /// Scan every row (including rejected records) and return counts useful
    /// for `teacher-data inspect`.  Like [`Self::verify`], this is a full
    /// integrity pass and remains bounded by one record's token vector.
    pub fn inspect(&self) -> Result<TdataInspectSummary> {
        let selector = TdataSelector {
            root: self.root.clone(),
            split: TdataSplitSelector::All,
            share: TdataShare::FULL,
            usable_only: false,
        };
        let mut summary = TdataInspectSummary::default();
        for sample in self.stream(selector)? {
            let sample = sample?;
            summary.add(&sample.record, sample.index);
        }
        Ok(summary)
    }
}

// Concise names for CLI wiring.  The `Tdata*` spellings remain available to
// make public API searches unambiguous in a large command module.
pub type Split = TdataSplit;
pub type SplitSelector = TdataSplitSelector;
pub type Share = TdataShare;
pub type Selector = TdataSelector;
pub type Manifest = TdataManifest;
pub type Record = TdataRecord;
pub type TokenIndex = TdataTokenIndex;
pub type Sample = TdataSample;
pub type InspectSummary = TdataInspectSummary;
pub type Dataset = TdataReader;

/// Stateful O(1)-memory record reader returned by [`TdataReader::stream`].
pub struct TdataStream {
    records: BufReader<File>,
    index: File,
    tokens: File,
    selector: TdataSelector,
    index_entries: u64,
    token_count: u64,
    expected_records: Option<u64>,
    ordinal: u64,
    expected_tok_offset: u64,
    done: bool,
}

impl TdataStream {
    fn next_sample(&mut self) -> Result<Option<TdataSample>> {
        loop {
            let Some(line) = read_jsonl_line(&mut self.records)? else {
                ensure!(
                    self.ordinal == self.index_entries,
                    "records.jsonl ended after {} rows but tokens.idx contains {} entries",
                    self.ordinal,
                    self.index_entries
                );
                ensure!(
                    self.expected_tok_offset == self.token_count,
                    "tokens.bin has {} tokens but records/index account for {}",
                    self.token_count,
                    self.expected_tok_offset
                );
                if let Some(expected) = self.expected_records {
                    ensure!(
                        expected == self.ordinal,
                        "manifest.json stats.records={expected}, but records.jsonl has {} rows",
                        self.ordinal
                    );
                }
                return Ok(None);
            };
            ensure!(
                self.ordinal < self.index_entries,
                "records.jsonl has more rows than tokens.idx entries (first extra row {})",
                self.ordinal + 1
            );
            let line_no = self
                .ordinal
                .checked_add(1)
                .expect("u64 line number overflow");
            let record = TdataRecord::from_json_line(&line, line_no as usize)?;
            let index = read_index_entry(&mut self.index, self.ordinal)?;
            self.validate_pair(&record, index, line_no)?;
            self.ordinal += 1;
            self.expected_tok_offset = self
                .expected_tok_offset
                .checked_add(index.total as u64)
                .ok_or_else(|| {
                    anyhow::anyhow!("tokens.idx record {line_no}: token offset overflow")
                })?;

            if self.selector.matches(&record) {
                let tokens = self.read_tokens(index, line_no)?;
                return Ok(Some(TdataSample {
                    record,
                    index,
                    tokens,
                }));
            }
        }
    }

    fn validate_pair(
        &self,
        record: &TdataRecord,
        index: TdataTokenIndex,
        line_no: u64,
    ) -> Result<()> {
        let context = format!(
            "records.jsonl line {line_no} / tokens.idx entry {}",
            line_no - 1
        );
        ensure!(
            record.tok.index == line_no - 1,
            "{context}: tok.index={} but expected {}",
            record.tok.index,
            line_no - 1
        );
        ensure!(
            record.tok.offset == index.tok_offset,
            "{context}: JSON tok.offset={} differs from index {}",
            record.tok.offset,
            index.tok_offset
        );
        ensure!(
            index.tok_offset == self.expected_tok_offset,
            "{context}: tok_offset={} but append-only sequence expects {}",
            index.tok_offset,
            self.expected_tok_offset
        );
        ensure!(
            record.tok.total == index.total
                && record.tok.prompt == index.prompt
                && record.tok.think == index.think
                && record.tok.answer == index.answer,
            "{context}: JSON tok counts differ from tokens.idx"
        );
        ensure!(
            record.split == index.split,
            "{context}: JSON split={} differs from index split={}",
            record.split,
            index.split
        );
        ensure!(
            record.usable == index.flags.usable(),
            "{context}: JSON usable={} differs from index usable={}",
            record.usable,
            index.flags.usable()
        );
        if let Some(quality) = &record.quality {
            if let Some(truncated) = quality.truncated {
                ensure!(
                    truncated == index.flags.truncated(),
                    "{context}: quality.truncated={truncated} differs from index flag={} ",
                    index.flags.truncated()
                );
            }
            if let Some(loop_suspect) = quality.loop_suspect {
                ensure!(
                    loop_suspect == index.flags.loop_suspect(),
                    "{context}: quality.loop_suspect={loop_suspect} differs from index flag={}",
                    index.flags.loop_suspect()
                );
            }
        }
        match record.topk {
            Some(topk) => {
                ensure!(
                    index.topk_offset != TdataTokenIndex::NO_TOPK,
                    "{context}: JSON topk exists but index has NO_TOPK"
                );
                ensure!(
                    topk.offset == index.topk_offset,
                    "{context}: JSON topk.offset={} differs from index {}",
                    topk.offset,
                    index.topk_offset
                );
                ensure!(
                    topk.positions as u64 == index.think as u64 + index.answer as u64,
                    "{context}: topk.positions={} but generated think + answer={}",
                    topk.positions,
                    index.think as u64 + index.answer as u64
                );
            }
            None => ensure!(
                index.topk_offset == TdataTokenIndex::NO_TOPK,
                "{context}: index has topk_offset={} but JSON has no topk",
                index.topk_offset
            ),
        }
        let end = index
            .tok_offset
            .checked_add(index.total as u64)
            .ok_or_else(|| anyhow::anyhow!("{context}: token range overflows u64"))?;
        ensure!(
            end <= self.token_count,
            "{context}: token range {}..{end} exceeds tokens.bin length {}",
            index.tok_offset,
            self.token_count
        );
        Ok(())
    }

    fn read_tokens(&mut self, index: TdataTokenIndex, line_no: u64) -> Result<Vec<u32>> {
        let byte_offset = index.tok_offset.checked_mul(4).ok_or_else(|| {
            anyhow::anyhow!("tokens.idx entry {}: byte offset overflow", line_no - 1)
        })?;
        let byte_len = (index.total as usize).checked_mul(4).ok_or_else(|| {
            anyhow::anyhow!("tokens.idx entry {}: byte length overflow", line_no - 1)
        })?;
        self.tokens
            .seek(SeekFrom::Start(byte_offset))
            .with_context(|| format!("seeking tokens.bin for record {line_no}"))?;
        let mut bytes = vec![0_u8; byte_len];
        self.tokens
            .read_exact(&mut bytes)
            .with_context(|| format!("reading tokens.bin for record {line_no}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect())
    }
}

impl Iterator for TdataStream {
    type Item = Result<TdataSample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.next_sample() {
            Ok(Some(sample)) => Some(Ok(sample)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}

fn read_index_entry(file: &mut File, ordinal: u64) -> Result<TdataTokenIndex> {
    let mut bytes = [0_u8; TDATA_INDEX_BYTES];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading tokens.idx entry {ordinal}"))?;
    TdataTokenIndex::from_le_bytes(&bytes)
        .with_context(|| format!("decoding tokens.idx entry {ordinal}"))
}

/// Read exactly one JSONL row without allowing an arbitrary malformed line to
/// allocate the process.  Blank rows are an error: JSONL line order is the
/// index order, so silently skipping one would desynchronise the data set.
fn read_jsonl_line(reader: &mut BufReader<File>) -> Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let chunk = reader.fill_buf().context("reading records.jsonl")?;
        if chunk.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        let take = chunk
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(chunk.len(), |position| position + 1);
        ensure!(
            bytes.len().saturating_add(take) <= MAX_RECORD_JSONL_BYTES,
            "records.jsonl: one row exceeds {MAX_RECORD_JSONL_BYTES} bytes"
        );
        bytes.extend_from_slice(&chunk[..take]);
        let hit_newline = chunk[take - 1] == b'\n';
        reader.consume(take);
        if hit_newline {
            break;
        }
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    ensure!(
        !bytes.is_empty(),
        "records.jsonl: blank rows are not valid v1 records"
    );
    Ok(Some(
        String::from_utf8(bytes).context("records.jsonl: a row is not valid UTF-8")?,
    ))
}

fn required_value<'a>(
    obj: &'a Map<String, Value>,
    field: &str,
    context: &str,
) -> Result<&'a Value> {
    obj.get(field)
        .ok_or_else(|| anyhow::anyhow!("{context}: missing required field {field:?}"))
}

fn required_string<'a>(obj: &'a Map<String, Value>, field: &str, context: &str) -> Result<&'a str> {
    required_value(obj, field, context)?
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("{context}: field {field:?} must be a string"))
}

fn optional_string(obj: &Map<String, Value>, field: &str, context: &str) -> Result<Option<String>> {
    match obj.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| anyhow::anyhow!("{context}: field {field:?} must be a string or null")),
    }
}

fn required_bool(obj: &Map<String, Value>, field: &str, context: &str) -> Result<bool> {
    required_value(obj, field, context)?
        .as_bool()
        .ok_or_else(|| anyhow::anyhow!("{context}: field {field:?} must be a bool"))
}

fn required_u32(obj: &Map<String, Value>, field: &str, context: &str) -> Result<u32> {
    let number = json_u64(
        required_value(obj, field, context)?,
        &format!("{context}.{field}"),
    )?;
    u32::try_from(number)
        .map_err(|_| anyhow::anyhow!("{context}: field {field:?}={number} does not fit u32"))
}

fn json_u64(value: &Value, context: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("{context}: expected a non-negative integer"))
}

fn parse_messages(value: &Value, context: &str) -> Result<Vec<TdataMessage>> {
    let messages = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{context}: messages must be an array"))?;
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let message_context = format!("{context}: messages[{index}]");
        let obj = message
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{message_context} must be an object"))?;
        out.push(TdataMessage {
            role: required_string(obj, "role", &message_context)?.to_owned(),
            content: required_value(obj, "content", &message_context)?.clone(),
        });
    }
    Ok(out)
}

fn parse_response(value: &Value, context: &str) -> Result<TdataResponse> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{context}: response must be an object"))?;
    Ok(TdataResponse {
        think: optional_string(obj, "think", context)?,
        content: required_string(obj, "content", context)?.to_owned(),
    })
}

fn parse_token_span(value: &Value, context: &str) -> Result<TdataTokenSpan> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{context}: tok must be an object"))?;
    Ok(TdataTokenSpan {
        index: json_u64(
            required_value(obj, "index", context)?,
            &format!("{context}.tok.index"),
        )?,
        offset: json_u64(
            required_value(obj, "offset", context)?,
            &format!("{context}.tok.offset"),
        )?,
        prompt: required_u32(obj, "prompt", context)?,
        think: required_u32(obj, "think", context)?,
        answer: required_u32(obj, "answer", context)?,
        total: required_u32(obj, "total", context)?,
    })
}

fn parse_topk(value: Option<&Value>, context: &str) -> Result<Option<TdataTopkRef>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{context}: topk must be an object or null"))?;
    Ok(Some(TdataTopkRef {
        offset: json_u64(
            required_value(obj, "offset", context)?,
            &format!("{context}.topk.offset"),
        )?,
        positions: required_u32(obj, "positions", context)?,
    }))
}

fn parse_quality(value: Option<&Value>, context: &str) -> Result<Option<TdataQuality>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{context}: quality must be an object or null"))?;
    let parse_optional_bool = |field: &str| -> Result<Option<bool>> {
        match obj.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_bool().map(Some).ok_or_else(|| {
                anyhow::anyhow!("{context}: quality.{field} must be a bool or null")
            }),
        }
    };
    Ok(Some(TdataQuality {
        truncated: parse_optional_bool("truncated")?,
        loop_suspect: parse_optional_bool("loop_suspect")?,
        raw_check: obj.get("check").cloned(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTdata {
        root: PathBuf,
    }

    impl TempTdata {
        fn new(name: &str) -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "cmf-tdata-{name}-{}-{serial}.tdata",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(
            &self,
            rows: &[(Value, TdataTokenIndex, Vec<u32>)],
            expected_records: Option<usize>,
        ) {
            let manifest = json!({
                "format": TDATA_FORMAT,
                "version": TDATA_VERSION,
                "complete": true,
                "stats": {"records": expected_records.unwrap_or(rows.len())},
            });
            fs::write(
                self.root.join("manifest.json"),
                serde_json::to_vec_pretty(&manifest).unwrap(),
            )
            .unwrap();
            let mut records = File::create(self.root.join("records.jsonl")).unwrap();
            let mut index = File::create(self.root.join("tokens.idx")).unwrap();
            let mut tokens = File::create(self.root.join("tokens.bin")).unwrap();
            for (row, entry, ids) in rows {
                writeln!(records, "{}", serde_json::to_string(row).unwrap()).unwrap();
                index.write_all(&entry.to_le_bytes().unwrap()).unwrap();
                for id in ids {
                    tokens.write_all(&id.to_le_bytes()).unwrap();
                }
            }
        }
    }

    impl Drop for TempTdata {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn row(rid: &str, split: &str, usable: bool, index: u64, offset: u64, ids: &[u32]) -> Value {
        let total = ids.len() as u32;
        json!({
            "rid": rid,
            "pid": rid.split('#').next().unwrap_or(rid),
            "sample": 0,
            "category": "test",
            "split": split,
            "usable": usable,
            "messages": [{"role": "user", "content": "hello"}],
            "response": {"think": "", "content": "world"},
            "finish": "stop",
            "tok": {"index": index, "offset": offset, "prompt": 1, "think": 0, "answer": total - 1, "total": total},
            "quality": {"truncated": false, "loop_suspect": false, "check": null},
            "meta": {}
        })
    }

    fn index(offset: u64, total: u32, split: TdataSplit, usable: bool) -> TdataTokenIndex {
        TdataTokenIndex {
            tok_offset: offset,
            total,
            prompt: 1,
            think: 0,
            answer: total - 1,
            topk_offset: TdataTokenIndex::NO_TOPK,
            flags: TdataIndexFlags(if usable { TdataIndexFlags::USABLE } else { 0 }),
            split,
        }
    }

    #[test]
    fn streams_only_usable_selected_split_without_cross_record_join() {
        let data = TempTdata::new("split");
        let rows = vec![
            (
                row("train-ok#0", "train", true, 0, 0, &[10, 11]),
                index(0, 2, TdataSplit::Train, true),
                vec![10, 11],
            ),
            (
                row("train-bad#0", "train", false, 1, 2, &[20, 21]),
                index(2, 2, TdataSplit::Train, false),
                vec![20, 21],
            ),
            (
                row("val-ok#0", "val", true, 2, 4, &[30, 31, 32]),
                index(4, 3, TdataSplit::Val, true),
                vec![30, 31, 32],
            ),
            (
                row("test-ok#0", "test", true, 3, 7, &[40, 41]),
                index(7, 2, TdataSplit::Test, true),
                vec![40, 41],
            ),
        ];
        data.write(&rows, None);
        let reader = TdataReader::open(&data.root).unwrap();
        let train = reader
            .stream(TdataSelector::parse(&format!("{}:train", data.root.display())).unwrap())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(train.len(), 1);
        assert_eq!(train[0].record.rid, "train-ok#0");
        assert_eq!(train[0].tokens, vec![10, 11]);

        let all = reader
            .stream(TdataSelector::parse(&format!("{}:all", data.root.display())).unwrap())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            all.iter()
                .map(|item| item.record.rid.as_str())
                .collect::<Vec<_>>(),
            ["train-ok#0", "val-ok#0", "test-ok#0"]
        );
        assert_eq!(all[1].tokens, vec![30, 31, 32]);
        // The second sample stays exactly its own record; it was not joined
        // to the preceding train row or the following test row.
        assert_eq!(all[2].tokens, vec![40, 41]);
        reader.verify().unwrap();
        let inspection = reader.inspect().unwrap();
        assert_eq!(inspection.records, 4);
        assert_eq!(inspection.usable, 3);
        assert_eq!(inspection.rejected, 1);
        assert_eq!(inspection.tokens, 9);
        assert_eq!(inspection.train.records, 2);
        assert_eq!(inspection.val.usable, 1);
        assert_eq!(inspection.test.tokens, 2);
    }

    #[test]
    fn index_is_exactly_40_bytes_and_uses_le_u64_offsets() {
        let entry = TdataTokenIndex {
            tok_offset: 0x0102_0304_0506_0708,
            total: 3,
            prompt: 1,
            think: 1,
            answer: 1,
            topk_offset: 0x1112_1314_1516_1718,
            flags: TdataIndexFlags(TdataIndexFlags::USABLE | TdataIndexFlags::LOOP),
            split: TdataSplit::Val,
        };
        let bytes = entry.to_le_bytes().unwrap();
        assert_eq!(bytes.len(), TDATA_INDEX_BYTES);
        assert_eq!(&bytes[0..8], &0x0102_0304_0506_0708_u64.to_le_bytes());
        assert_eq!(&bytes[24..32], &0x1112_1314_1516_1718_u64.to_le_bytes());
        assert_eq!(TdataTokenIndex::from_le_bytes(&bytes).unwrap(), entry);

        let data = TempTdata::new("offsets");
        let rows = vec![
            (
                row("one#0", "train", true, 0, 0, &[1, 2]),
                index(0, 2, TdataSplit::Train, true),
                vec![1, 2],
            ),
            (
                row("two#0", "train", true, 1, 2, &[3, 4, 5]),
                index(2, 3, TdataSplit::Train, true),
                vec![3, 4, 5],
            ),
        ];
        data.write(&rows, None);
        let reader = TdataReader::open(&data.root).unwrap();
        let samples = reader
            .stream(TdataSelector::parse(&data.root.display().to_string()).unwrap())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(samples[1].index.tok_offset, 2);
        assert_eq!(samples[1].tokens, vec![3, 4, 5]);
    }

    #[test]
    fn rejects_truncated_index_and_malformed_or_truncated_payloads() {
        let bad_index = TempTdata::new("bad-index");
        fs::write(
            bad_index.root.join("manifest.json"),
            r#"{"format":"cmf-tdata","version":1,"complete":true}"#,
        )
        .unwrap();
        fs::write(bad_index.root.join("records.jsonl"), b"").unwrap();
        fs::write(bad_index.root.join("tokens.idx"), vec![0_u8; 39]).unwrap();
        fs::write(bad_index.root.join("tokens.bin"), b"").unwrap();
        assert!(TdataReader::open(&bad_index.root).is_err());

        let bad_payload = TempTdata::new("bad-payload");
        let good_row = row("broken#0", "train", true, 0, 0, &[1, 2]);
        bad_payload.write(
            &[(
                good_row,
                index(0, 2, TdataSplit::Train, true),
                // Deliberately one token short of the JSON/index promise.
                vec![1],
            )],
            None,
        );
        let reader = TdataReader::open(&bad_payload.root).unwrap();
        let error = reader
            .stream(TdataSelector::parse(&bad_payload.root.display().to_string()).unwrap())
            .unwrap()
            .next()
            .expect("one stream item")
            .expect_err("truncated tokens.bin must fail");
        assert!(error.to_string().contains("exceeds tokens.bin length"));

        let malformed = TempTdata::new("malformed");
        fs::write(
            malformed.root.join("manifest.json"),
            r#"{"format":"cmf-tdata","version":1,"complete":true,"stats":{"records":1}}"#,
        )
        .unwrap();
        fs::write(malformed.root.join("records.jsonl"), b"not json\n").unwrap();
        fs::write(
            malformed.root.join("tokens.idx"),
            index(0, 1, TdataSplit::Train, true).to_le_bytes().unwrap(),
        )
        .unwrap();
        fs::write(malformed.root.join("tokens.bin"), 1_u32.to_le_bytes()).unwrap();
        let reader = TdataReader::open(&malformed.root).unwrap();
        assert!(
            reader
                .stream(TdataSelector::parse(&malformed.root.display().to_string()).unwrap())
                .unwrap()
                .next()
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn selector_defaults_and_share_are_deterministic() {
        let selector = TdataSelector::parse("data.tdata").unwrap();
        assert_eq!(selector.split, TdataSplitSelector::Train);
        assert_eq!(selector.share, TdataShare::FULL);
        let half = TdataSelector::parse("data.tdata:all=.5").unwrap();
        assert_eq!(half.share.numerator(), 5);
        assert_eq!(half.share.denominator(), 10);
        assert_eq!(TdataShare::parse("1.0").unwrap().to_string(), "1");
        assert_eq!(
            half.share.contains_rid("stable-record"),
            half.share.contains_rid("stable-record")
        );
        assert!(TdataSelector::parse("data.tdata:test=0").is_err());
        assert!(TdataSelector::parse("data.tdata:other").is_err());
    }
}
