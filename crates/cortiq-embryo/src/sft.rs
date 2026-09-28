//! Native response-only SFT data preparation and batching.
//!
//! This module deliberately keeps the SFT corpus separate from the historical
//! flat continuation shards.  Each record is one independent causal example:
//! the input is exactly `seq` tokens and the target is exactly `seq` token ids,
//! with `u16::MAX` marking prompt/padding positions.  Records are sampled as
//! whole rows, so a batch never crosses a conversation boundary.

use crate::tokenizer::{Bpe, EOT};
use crate::train::Shard;
use anyhow::{ensure, Context};
use flate2::read::GzDecoder;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

pub const SFT_MAGIC: &[u8; 8] = b"CMFSFT1\0";
pub const IGNORE: u16 = u16::MAX;

/// A fixed-width response-only shard.  `tokens` has `(seq + 1)` values per
/// record (the extra value is the final causal target); `targets` has `seq`
/// values per record and uses `IGNORE` for prompt/padding positions.
#[derive(Clone, Debug)]
pub struct SftShard {
    pub seq: usize,
    pub records: usize,
    pub tokens: Vec<u16>,
    pub targets: Vec<u16>,
}

impl SftShard {
    pub fn new(seq: usize, records: usize, tokens: Vec<u16>, targets: Vec<u16>) -> SftShard {
        assert_eq!(tokens.len(), records.saturating_mul(seq + 1));
        assert_eq!(targets.len(), records.saturating_mul(seq));
        SftShard {
            seq,
            records,
            tokens,
            targets,
        }
    }

    pub fn valid_tokens(&self) -> usize {
        self.targets.iter().filter(|&&x| x != IGNORE).count()
    }

    pub fn record(&self, index: usize) -> (&[u16], &[u16]) {
        assert!(index < self.records);
        let to = index * (self.seq + 1);
        let yo = index * self.seq;
        (
            &self.tokens[to..to + self.seq + 1],
            &self.targets[yo..yo + self.seq],
        )
    }

    /// Fill one all-SFT batch using deterministic record order.  This is the
    /// validation path; the training sampler below mixes exactly 20% raw LM
    /// rows by batch slot.
    pub fn fixed_batch(
        &self,
        batch: usize,
        index: usize,
        tokens: &mut Vec<u32>,
        targets: &mut Vec<u32>,
    ) {
        ensure_batch_dims(self.seq, batch, self.records);
        tokens.clear();
        targets.clear();
        for row in 0..batch {
            let (tok, tgt) = self.record((index * batch + row) % self.records);
            tokens.extend(tok[..self.seq].iter().map(|&x| x as u32));
            targets.extend(
                tgt.iter()
                    .map(|&x| if x == IGNORE { u32::MAX } else { x as u32 }),
            );
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut f =
            File::create(path).with_context(|| format!("create SFT shard {}", path.display()))?;
        f.write_all(SFT_MAGIC)?;
        f.write_all(&(self.seq as u32).to_le_bytes())?;
        f.write_all(&(self.records as u64).to_le_bytes())?;
        for &x in self.tokens.iter().chain(&self.targets) {
            f.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<SftShard> {
        let mut f =
            File::open(path).with_context(|| format!("open SFT shard {}", path.display()))?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        ensure!(&magic == SFT_MAGIC, "{}: bad SFT magic", path.display());
        let mut seqb = [0u8; 4];
        f.read_exact(&mut seqb)?;
        let seq = u32::from_le_bytes(seqb) as usize;
        let mut rb = [0u8; 8];
        f.read_exact(&mut rb)?;
        let records = u64::from_le_bytes(rb) as usize;
        ensure!(seq >= 8, "{}: sequence too short", path.display());
        let count = records
            .checked_mul((seq + 1) + seq)
            .context("SFT shard element count overflow")?;
        let bytes = count
            .checked_mul(2)
            .context("SFT shard byte count overflow")?;
        let mut raw = vec![0u8; bytes];
        f.read_exact(&mut raw)?;
        let mut vals = Vec::with_capacity(count);
        for c in raw.chunks_exact(2) {
            vals.push(u16::from_le_bytes([c[0], c[1]]));
        }
        let tok_n = records * (seq + 1);
        let tokens = vals[..tok_n].to_vec();
        let targets = vals[tok_n..].to_vec();
        let mut extra = [0u8; 1];
        ensure!(
            f.read(&mut extra)? == 0,
            "{}: trailing bytes",
            path.display()
        );
        Ok(SftShard::new(seq, records, tokens, targets))
    }
}

fn ensure_batch_dims(seq: usize, batch: usize, records: usize) {
    assert!(seq > 0 && batch > 0 && records > 0);
}

/// Deterministic mixed SFT/replay sampler.  The first four fifths of every
/// batch are independent SFT records and the last fifth are raw-LM windows;
/// `batch % 5 == 0` is enforced by the constructor.
pub struct SftSampler {
    pub b: usize,
    pub t: usize,
    sft_rows: usize,
    state: u64,
}

impl SftSampler {
    pub fn new(b: usize, t: usize, seed: u64) -> SftSampler {
        assert!(b >= 5 && b % 5 == 0, "SFT batch must be a multiple of 5");
        SftSampler {
            b,
            t,
            sft_rows: b * 4 / 5,
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Advance exactly as many PRNG draws as `batch` consumes without
    /// materialising a batch.  A resumed run must use this to continue the
    /// original sampler stream rather than silently replaying its prefix.
    pub fn skip_batches(&mut self, batches: usize) {
        for _ in 0..batches {
            for _ in 0..self.b {
                let _ = self.next_u64();
            }
        }
    }

    pub fn batch(
        &mut self,
        sft: &SftShard,
        raw: &Shard,
        tokens: &mut Vec<u32>,
        targets: &mut Vec<u32>,
    ) {
        assert_eq!(sft.seq, self.t);
        ensure_batch_dims(self.t, self.b, sft.records);
        assert!(
            raw.tokens.len() > self.t + 1,
            "replay shard shorter than one window"
        );
        tokens.clear();
        targets.clear();
        for _ in 0..self.sft_rows {
            let index = (self.next_u64() % sft.records as u64) as usize;
            let (tok, tgt) = sft.record(index);
            tokens.extend(tok[..self.t].iter().map(|&x| x as u32));
            targets.extend(
                tgt.iter()
                    .map(|&x| if x == IGNORE { u32::MAX } else { x as u32 }),
            );
        }
        let n = raw.tokens.len();
        for _ in self.sft_rows..self.b {
            let start = (self.next_u64() % (n - self.t - 1) as u64) as usize;
            let w = &raw.tokens[start..start + self.t + 1];
            tokens.extend(w[..self.t].iter().map(|&x| x as u32));
            targets.extend(w[1..].iter().map(|&x| x as u32));
        }
        assert_eq!(tokens.len(), self.b * self.t);
        assert_eq!(targets.len(), self.b * self.t);
        assert!(targets.iter().any(|&x| x != u32::MAX));
    }
}

#[derive(Clone, Debug)]
struct Message {
    id: String,
    parent_id: Option<String>,
    tree_id: String,
    role: String,
    lang: String,
    text: String,
    review_count: u32,
    review_result: bool,
    deleted: bool,
    synthetic: bool,
    rank: i32,
    tree_state: String,
}

#[derive(Clone, Debug)]
struct Candidate {
    tree_id: String,
    message_id: String,
    split: Split,
    lang: String,
    prompt_key: String,
    pair_key: String,
    context: Vec<(String, String)>,
    answer: String,
    task: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
enum Split {
    Train,
    Dev,
    Final,
}

impl Split {
    fn name(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Dev => "dev",
            Split::Final => "final",
        }
    }
}

#[derive(Default, Serialize)]
struct SplitStats {
    trees: usize,
    examples: usize,
    valid_answer_tokens: usize,
    truncated_answers: usize,
    langs: BTreeMap<String, usize>,
    tasks: BTreeMap<String, usize>,
}

#[derive(Default, Serialize)]
struct PrepareStats {
    source_file: String,
    source_revision: String,
    source_sha256: String,
    license: String,
    template: String,
    sequence: usize,
    parsed_messages: usize,
    parsed_trees: usize,
    eligible_assistant_messages: usize,
    dropped_duplicate_pair: usize,
    dropped_duplicate_prompt: usize,
    dropped_missing_path: usize,
    dropped_empty_or_overlong: usize,
    splits: BTreeMap<String, SplitStats>,
    /// messages format only: how every record got its plant/tree group.
    #[serde(skip_serializing_if = "Option::is_none")]
    group_sources: Option<GroupSources>,
    /// messages format only: group → split, so a held-out subject can be
    /// looked up without re-running the converter.
    #[serde(skip_serializing_if = "Option::is_none")]
    groups: Option<BTreeMap<String, String>>,
}

#[derive(Default, Serialize, Debug, PartialEq, Eq)]
pub struct GroupSources {
    /// group read from a JSON field (`--group-field` or one of the defaults)
    pub by_field: usize,
    /// group inferred from a Latin binomial named in the record itself
    pub inferred: usize,
    /// no name in the record: group carried from the previous record
    pub carried: usize,
    /// records before the first named one: group of the first named record
    pub backfilled: usize,
    /// no record names anything: every record is its own group
    pub singleton: usize,
}

/// Prepare a provenance-sealed response-only corpus from the official flat
/// OASST1 message export.  Tree split and duplicate filtering happen on raw
/// text before any BPE tokenization.
pub fn prepare(
    input: &Path,
    tokenizer: &Path,
    seq: usize,
    train_out: &Path,
    dev_out: &Path,
    final_out: &Path,
    manifest_out: &Path,
) -> anyhow::Result<()> {
    ensure!(seq >= 64, "SFT sequence must be at least 64 tokens");
    let messages = read_messages(input)?;
    let parsed_trees: HashSet<&str> = messages.iter().map(|m| m.tree_id.as_str()).collect();
    let by_id: HashMap<&str, usize> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();
    let mut candidates = Vec::new();
    let mut eligible = 0usize;
    let mut missing_path = 0usize;
    for m in &messages {
        if !eligible_assistant(m) {
            continue;
        }
        eligible += 1;
        let mut chain = Vec::new();
        let mut cur = m.parent_id.as_deref();
        let mut seen = HashSet::new();
        while let Some(id) = cur {
            if !seen.insert(id) {
                break;
            }
            let Some(&i) = by_id.get(id) else { break };
            chain.push(i);
            cur = messages[i].parent_id.as_deref();
        }
        if chain.is_empty() || cur.is_some() {
            missing_path += 1;
            continue;
        }
        chain.reverse();
        if messages[chain[0]].role != "prompter"
            || chain
                .iter()
                .any(|&i| messages[i].deleted || messages[i].text.trim().is_empty())
        {
            missing_path += 1;
            continue;
        }
        let context: Vec<(String, String)> = chain
            .iter()
            .map(|&i| (messages[i].role.clone(), messages[i].text.clone()))
            .collect();
        let prompt_key = normalize(&context.last().map(|x| x.1.as_str()).unwrap_or(""));
        if prompt_key.is_empty() {
            missing_path += 1;
            continue;
        }
        let pair_key = format!("{prompt_key}\u{0}{}", normalize(&m.text));
        candidates.push(Candidate {
            tree_id: m.tree_id.clone(),
            message_id: m.id.clone(),
            split: split_for_tree(&m.tree_id),
            lang: m.lang.clone(),
            prompt_key,
            pair_key,
            task: classify_task(&context, &m.text),
            context,
            answer: m.text.clone(),
        });
    }
    // Stable tree/message order makes duplicate ownership reproducible even if
    // the upstream JSONL line order changes.
    candidates.sort_by(|a, b| {
        a.tree_id
            .cmp(&b.tree_id)
            .then_with(|| a.message_id.cmp(&b.message_id))
    });
    let mut seen_pair = HashSet::new();
    let mut seen_prompt = HashSet::new();
    let mut accepted = Vec::new();
    let mut dropped_pair = 0usize;
    let mut dropped_prompt = 0usize;
    for c in candidates {
        if !seen_pair.insert(c.pair_key.clone()) {
            dropped_pair += 1;
            continue;
        }
        if !seen_prompt.insert(c.prompt_key.clone()) {
            dropped_prompt += 1;
            continue;
        }
        accepted.push(c);
    }
    let bpe = Bpe::load(tokenizer).context("load native BPE tokenizer")?;
    let mut cache = HashMap::new();
    let mut out: HashMap<Split, (Vec<u16>, Vec<u16>)> = HashMap::new();
    let mut stats: BTreeMap<String, SplitStats> = BTreeMap::new();
    let mut split_trees: HashMap<String, HashSet<String>> = HashMap::new();
    let mut dropped_long = 0usize;
    for c in accepted {
        let Some((tokens, targets, truncated)) =
            encode_record(&bpe, &c.context, &c.answer, seq, &mut cache)?
        else {
            dropped_long += 1;
            continue;
        };
        let entry = out
            .entry(c.split)
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.extend_from_slice(&tokens);
        entry.1.extend_from_slice(&targets);
        let s = stats.entry(c.split.name().to_string()).or_default();
        s.examples += 1;
        s.valid_answer_tokens += targets.iter().filter(|&&x| x != IGNORE).count();
        s.truncated_answers += truncated as usize;
        *s.langs.entry(c.lang).or_default() += 1;
        *s.tasks.entry(c.task).or_default() += 1;
        split_trees
            .entry(c.split.name().to_string())
            .or_default()
            .insert(c.tree_id);
    }
    for (name, s) in &mut stats {
        s.trees = split_trees.get(name).map_or(0, HashSet::len);
    }
    let save_split = |split: Split, path: &Path| -> anyhow::Result<()> {
        let (tok, tgt) = out
            .get(&split)
            .cloned()
            .unwrap_or_else(|| (Vec::new(), Vec::new()));
        let records = stats.get(split.name()).map(|x| x.examples).unwrap_or(0);
        ensure!(records > 0, "{} split is empty", split.name());
        SftShard::new(seq, records, tok, tgt).save(path)
    };
    save_split(Split::Train, train_out)?;
    save_split(Split::Dev, dev_out)?;
    save_split(Split::Final, final_out)?;
    let manifest = PrepareStats {
        source_file: input.display().to_string(),
        source_revision: "fdf72ae0827c1cda404aff25b6603abec9e339b9".to_string(),
        source_sha256: "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c".to_string(),
        license: "Apache-2.0".to_string(),
        template: "cmf-im-v1: <|im_start|>role\\ntext<|im_end|>\\n; assistant answer + <|im_end|><|endoftext|> are valid targets".to_string(),
        sequence: seq,
        parsed_messages: messages.len(),
        parsed_trees: parsed_trees.len(),
        eligible_assistant_messages: eligible,
        dropped_duplicate_pair: dropped_pair,
        dropped_duplicate_prompt: dropped_prompt,
        dropped_missing_path: missing_path,
        dropped_empty_or_overlong: dropped_long,
        splits: stats,
        group_sources: None,
        groups: None,
    };
    // Keep the manifest self-contained and byte-stable across runs.
    let text = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(manifest_out, text)?;
    println!(
        "sft: {} messages, {} trees, train/dev/final={}/{}/{} records; manifest={}",
        manifest.parsed_messages,
        manifest.parsed_trees,
        manifest.splits.get("train").map_or(0, |s| s.examples),
        manifest.splits.get("dev").map_or(0, |s| s.examples),
        manifest.splits.get("final").map_or(0, |s| s.examples),
        manifest_out.display()
    );
    Ok(())
}

/// Build a fresh raw-LM replay shard from bounded prefixes of the original
/// native token shards.  This is intentionally not the frozen quality-v1
/// artifact used for the previous gate.
pub fn make_raw_replay(inputs: &[PathBuf], out: &Path, max_tokens: usize) -> anyhow::Result<()> {
    ensure!(!inputs.is_empty() && max_tokens >= inputs.len());
    let per = max_tokens / inputs.len();
    let mut tokens = Vec::with_capacity(max_tokens);
    for input in inputs {
        let mut f = File::open(input).with_context(|| format!("open {}", input.display()))?;
        let mut raw = vec![0u8; per * 2];
        let n = f.read(&mut raw)?;
        raw.truncate(n - (n % 2));
        tokens.extend(
            raw.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])),
        );
    }
    tokens.truncate(max_tokens);
    ensure!(tokens.len() > 1025, "raw replay is too short");
    let n_tokens = tokens.len();
    Shard { tokens }.save(out)?;
    println!("raw replay: {} tokens → {}", n_tokens, out.display());
    Ok(())
}

fn read_messages(path: &Path) -> anyhow::Result<Vec<Message>> {
    let file = File::open(path).with_context(|| format!("open OASST1 {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().and_then(|x| x.to_str()) == Some("gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut out = Vec::new();
    for (line_no, line) in BufReader::with_capacity(1 << 20, reader)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse OASST1 JSON line {}", line_no + 1))?;
        let Some(id) = v.get("message_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(tree_id) = v.get("message_tree_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(role) = v.get("role").and_then(Value::as_str) else {
            continue;
        };
        let Some(text) = v.get("text").and_then(Value::as_str) else {
            continue;
        };
        out.push(Message {
            id: id.to_string(),
            parent_id: v
                .get("parent_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            tree_id: tree_id.to_string(),
            role: role.to_string(),
            lang: v
                .get("lang")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            text: text.to_string(),
            review_count: v.get("review_count").and_then(Value::as_u64).unwrap_or(0) as u32,
            review_result: v
                .get("review_result")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            deleted: v.get("deleted").and_then(Value::as_bool).unwrap_or(false),
            synthetic: v.get("synthetic").and_then(Value::as_bool).unwrap_or(false),
            rank: v.get("rank").and_then(Value::as_i64).unwrap_or(0) as i32,
            tree_state: v
                .get("tree_state")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    Ok(out)
}

fn eligible_assistant(m: &Message) -> bool {
    m.role == "assistant"
        && matches!(m.lang.as_str(), "en" | "ru")
        && m.review_count > 0
        && m.review_result
        && !m.deleted
        && !m.synthetic
        && m.rank == 0
        && m.tree_state == "ready_for_export"
        && !m.text.trim().is_empty()
}

/// The split (`"train"` | `"dev"` | `"final"`) the group rule of
/// `sft-prepare` assigns a subject group / conversation tree
/// (FNV-1a(group) mod 10: 0–7 train, 8 dev, 9 final) — what a consumer
/// checks a manifest's `groups` map against.
pub fn group_split(group: &str) -> &'static str {
    split_for_tree(group).name()
}

fn split_for_tree(tree_id: &str) -> Split {
    let mut h = 0xcbf29ce484222325u64;
    for b in tree_id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    match h % 10 {
        0..=7 => Split::Train,
        8 => Split::Dev,
        _ => Split::Final,
    }
}

fn normalize(text: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in text.trim().to_lowercase().chars() {
        if ch.is_alphanumeric() {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(ch);
        } else {
            space = true;
        }
    }
    out.trim().to_string()
}

fn classify_task(context: &[(String, String)], answer: &str) -> String {
    let prompt = context.last().map(|x| x.1.as_str()).unwrap_or("");
    let text = format!("{} {}", prompt, answer).to_lowercase();
    if text.contains("```")
        || ["python", "rust", "javascript", "function", "code", "sql"]
            .iter()
            .any(|x| text.contains(x))
    {
        "code".to_string()
    } else if text.chars().filter(|c| c.is_ascii_digit()).count() >= 2
        && [
            "solve",
            "equation",
            "calculate",
            "math",
            "integral",
            "числ",
            "реши",
        ]
        .iter()
        .any(|x| text.contains(x))
    {
        "math".to_string()
    } else {
        "general".to_string()
    }
}

fn special(bpe: &Bpe, name: &str) -> anyhow::Result<u32> {
    bpe.special_id(name)
        .with_context(|| format!("tokenizer missing special {name}"))
}

fn push_text(bpe: &Bpe, text: &str, cache: &mut HashMap<String, Vec<u32>>, out: &mut Vec<u32>) {
    bpe.encode(text, cache, out);
}

fn encode_record(
    bpe: &Bpe,
    context: &[(String, String)],
    answer: &str,
    seq: usize,
    cache: &mut HashMap<String, Vec<u32>>,
) -> anyhow::Result<Option<(Vec<u16>, Vec<u16>, bool)>> {
    let im_start = special(bpe, "<|im_start|>")?;
    let im_end = special(bpe, "<|im_end|>")?;
    let eot = special(bpe, EOT)?;
    let pad = special(bpe, "<|pad|>")?;
    let mut prefix = Vec::new();
    for (role, text) in context {
        prefix.push(im_start);
        let role = match role.as_str() {
            "prompter" | "user" => "user",
            "system" => "system",
            _ => "assistant",
        };
        push_text(bpe, role, cache, &mut prefix);
        push_text(bpe, "\n", cache, &mut prefix);
        push_text(bpe, text, cache, &mut prefix);
        prefix.push(im_end);
        push_text(bpe, "\n", cache, &mut prefix);
    }
    prefix.push(im_start);
    push_text(bpe, "assistant\n", cache, &mut prefix);
    let mut answer_tokens = Vec::new();
    push_text(bpe, answer, cache, &mut answer_tokens);
    let mut truncated = false;
    // Reserve one causal prefix token and the two explicit end markers.
    let max_answer_text = seq.saturating_sub(3);
    if answer_tokens.len() > max_answer_text {
        answer_tokens.truncate(max_answer_text);
        truncated = true;
    }
    if answer_tokens.is_empty() {
        return Ok(None);
    }
    answer_tokens.push(im_end);
    answer_tokens.push(eot);
    let keep_prefix = (seq + 1).saturating_sub(answer_tokens.len()).max(1);
    if prefix.len() > keep_prefix {
        let start = prefix.len() - keep_prefix;
        prefix = prefix[start..].to_vec();
    }
    let answer_start = prefix.len();
    let mut full = prefix;
    full.extend_from_slice(&answer_tokens);
    if full.len() > seq + 1 || answer_start == 0 {
        return Ok(None);
    }
    full.resize(seq + 1, pad);
    let mut tokens = Vec::with_capacity(seq + 1);
    tokens.extend(full.iter().map(|&x| x as u16));
    let mut targets = vec![IGNORE; seq];
    for j in answer_start..answer_start + answer_tokens.len() {
        if j == 0 || j > seq {
            continue;
        }
        targets[j - 1] = full[j] as u16;
    }
    ensure!(
        targets.iter().any(|&x| x != IGNORE),
        "empty response target"
    );
    Ok(Some((tokens, targets, truncated)))
}


/// Default JSON fields consulted for the subject/plant group of a messages
/// record, in order.  A `--group-field` overrides the list.
const GROUP_FIELDS: &[&str] = &["src_title", "latin", "plant", "group", "tree_id", "title"];

/// Capitalised English words that open a "Capitalised lowercase" pair without
/// being a genus.  The inference is a heuristic for corpora whose records
/// carry no explicit subject field; the manifest reports how many records it
/// touched so the split can be audited.
const GENUS_STOP: &[&str] = &[
    "The", "This", "These", "Those", "There", "Then", "They", "Their", "When", "Where", "Which",
    "What", "While", "With", "Some", "Most", "Many", "Much", "Such", "Also", "Only", "Both",
    "Each", "Other", "Another", "Root", "Roots", "Leaf", "Leaves", "Seed", "Seeds", "Bark",
    "Fruit", "Flower", "Flowers", "Data", "Evidence", "Information", "Sources", "Source",
    "Traditional", "Modern", "Studies", "Research", "Clinical", "Cochrane", "Note", "Notes",
    "Dosage", "Dose", "Safety", "Efficacy", "Contraindications", "Adverse", "Side", "Effects",
    "Uses", "Usage", "Preparation", "Preparations", "According", "Because", "Although",
    "However", "Since", "Before", "After", "During", "From", "Into", "Over", "Under", "Between",
    "Among", "Native", "Southern", "Northern", "Eastern", "Western", "Central", "South", "North",
    "East", "West", "Asia", "Africa", "Europe", "America", "Australia", "India", "China", "Japan",
    "Russia", "Ukraine", "Mexico", "Brazil", "Latin", "Greek", "English", "Russian", "Chinese",
    "Indian", "African", "European", "American", "Wikipedia", "Plants", "Plant",
];

/// Typical endings of a Latin species epithet.
const EPITHET_ENDINGS: &[&str] = &[
    "a", "um", "us", "is", "es", "e", "ii", "ae", "on", "ens", "ans", "or", "ix", "ys", "as", "os",
];

/// Latin binomials named in `text`, in order of appearance: a capitalised
/// ASCII word of ≥4 letters followed by a lowercase ASCII word of ≥4 letters
/// with a Latin ending, not opening a sentence.
pub fn latin_binomials(text: &str) -> Vec<String> {
    let mut words: Vec<(bool, String)> = Vec::new(); // (sentence_start, word)
    let mut cur = String::new();
    let mut sep_terminal = false;
    for ch in text.chars() {
        if ch.is_alphabetic() {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                words.push((sep_terminal, std::mem::take(&mut cur)));
                sep_terminal = false;
            }
            if matches!(ch, '.' | '?' | '!') {
                sep_terminal = true;
            }
        }
    }
    if !cur.is_empty() {
        words.push((sep_terminal, cur));
    }
    let mut out = Vec::new();
    for w in words.windows(2) {
        let (start, genus) = (&w[0].0, w[0].1.as_str());
        let epithet = w[1].1.as_str();
        if *start || w[1].0 {
            continue;
        }
        let gb = genus.as_bytes();
        if gb.len() < 4
            || !gb[0].is_ascii_uppercase()
            || !gb[1..].iter().all(u8::is_ascii_lowercase)
            || GENUS_STOP.contains(&genus)
        {
            continue;
        }
        let eb = epithet.as_bytes();
        if eb.len() < 4
            || !eb.iter().all(u8::is_ascii_lowercase)
            || !EPITHET_ENDINGS.iter().any(|e| epithet.ends_with(e))
        {
            continue;
        }
        out.push(format!("{genus} {epithet}"));
    }
    out
}

/// Assign a subject group to every record of an ordered corpus whose records
/// come in consecutive blocks per subject.  A record keeps the current group
/// while it names it (or names nothing); it opens a new group only when it
/// names a binomial that also dominates the next `WINDOW` records (≥2 records),
/// which keeps look-alike species mentioned in passing from splitting a block.
pub fn infer_groups(candidates: &[Vec<String>]) -> (Vec<String>, GroupSources) {
    const WINDOW: usize = 12;
    let mut src = GroupSources::default();
    let mut cur: Option<String> = None;
    let mut groups: Vec<Option<String>> = Vec::with_capacity(candidates.len());
    let mut kinds = Vec::with_capacity(candidates.len());
    for (i, c) in candidates.iter().enumerate() {
        let keeps = cur.as_ref().is_some_and(|g| c.contains(g));
        let mut kind = if c.is_empty() { 1 } else { 2 };
        if !keeps && !c.is_empty() {
            let mut win: HashMap<&str, usize> = HashMap::new();
            for cc in &candidates[i..candidates.len().min(i + WINDOW)] {
                let mut seen = HashSet::new();
                for x in cc {
                    if seen.insert(x.as_str()) {
                        *win.entry(x.as_str()).or_default() += 1;
                    }
                }
            }
            // most frequent in the window, first mention breaks ties
            let mut best: Option<(&str, usize, usize)> = None;
            let mut seen = HashSet::new();
            for (pos, x) in c.iter().enumerate() {
                if !seen.insert(x.as_str()) {
                    continue;
                }
                let n = win[x.as_str()];
                if best.is_none_or(|(_, bn, bp)| n > bn || (n == bn && pos < bp)) {
                    best = Some((x.as_str(), n, pos));
                }
            }
            if let Some((name, n, _)) = best {
                if n >= 2 && cur.as_deref() != Some(name) {
                    cur = Some(name.to_string());
                } else if cur.is_none() {
                    // isolated mention before any dominant block: still a name
                    cur = Some(name.to_string());
                } else {
                    kind = 1;
                }
            }
        }
        if cur.is_none() {
            kind = 0;
        }
        kinds.push(kind);
        groups.push(cur.clone());
    }
    let first = groups.iter().find_map(|g| g.clone());
    let mut out = Vec::with_capacity(groups.len());
    for (i, g) in groups.into_iter().enumerate() {
        match g {
            Some(g) => {
                match kinds[i] {
                    2 => src.inferred += 1,
                    _ => src.carried += 1,
                }
                out.push(g);
            }
            None => match &first {
                Some(f) => {
                    src.backfilled += 1;
                    out.push(f.clone());
                }
                None => {
                    src.singleton += 1;
                    out.push(format!("record-{i}"));
                }
            },
        }
    }
    (out, src)
}

fn fnv1a64_file(path: &Path) -> anyhow::Result<String> {
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut h = 0xcbf29ce484222325u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for b in &buf[..n] {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("fnv1a64:{h:016x}"))
}

/// Prepare response-only shards from a plain messages JSONL
/// (`{"messages":[{"role","content"},...], "lang"?, <group field>?}`), one
/// record per assistant turn with the preceding turns as context, under the
/// same cmf-im-v1 template as the OASST pipeline.  Train/dev/final are
/// disjoint by subject group: the group comes from a JSON field when the
/// records have one, otherwise it is inferred from the Latin binomials named
/// in consecutive records (see `infer_groups`).  Exact duplicate
/// (prompt, answer) pairs are dropped; identical prompts with different
/// answers are kept, since templated question sets repeat them per subject.
#[allow(clippy::too_many_arguments)]
pub fn prepare_messages(
    input: &Path,
    tokenizer: &Path,
    seq: usize,
    train_out: &Path,
    dev_out: &Path,
    final_out: &Path,
    manifest_out: &Path,
    group_field: Option<&str>,
) -> anyhow::Result<()> {
    ensure!(seq >= 64, "SFT sequence must be at least 64 tokens");
    struct Rec {
        turns: Vec<(String, String)>,
        lang: String,
        group: Option<String>,
    }
    let file = File::open(input).with_context(|| format!("open messages JSONL {}", input.display()))?;
    let reader: Box<dyn Read> = if input.extension().and_then(|x| x.to_str()) == Some("gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut recs = Vec::new();
    let mut skipped_lines = 0usize;
    for (line_no, line) in BufReader::with_capacity(1 << 20, reader).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse messages JSON line {}", line_no + 1))?;
        let Some(msgs) = v.get("messages").and_then(Value::as_array) else {
            skipped_lines += 1;
            continue;
        };
        let mut turns = Vec::new();
        for m in msgs {
            let (Some(role), Some(content)) = (
                m.get("role").and_then(Value::as_str),
                m.get("content").and_then(Value::as_str),
            ) else {
                turns.clear();
                break;
            };
            turns.push((role.to_string(), content.to_string()));
        }
        if turns.is_empty() {
            skipped_lines += 1;
            continue;
        }
        let fields: Vec<&str> = match group_field {
            Some(f) => vec![f],
            None => GROUP_FIELDS.to_vec(),
        };
        let group = fields.iter().find_map(|f| {
            v.get(f)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(str::to_string)
        });
        recs.push(Rec {
            turns,
            lang: v
                .get("lang")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            group,
        });
    }
    ensure!(!recs.is_empty(), "no usable records in {}", input.display());
    // Subject groups: explicit field where present, inferred elsewhere.
    let by_field = recs.iter().filter(|r| r.group.is_some()).count();
    let (groups, mut sources) = if by_field == recs.len() {
        (
            recs.iter().map(|r| r.group.clone().unwrap()).collect::<Vec<_>>(),
            GroupSources::default(),
        )
    } else {
        let candidates: Vec<Vec<String>> = recs
            .iter()
            .map(|r| {
                let mut c = Vec::new();
                for (_, text) in &r.turns {
                    c.extend(latin_binomials(text));
                }
                c
            })
            .collect();
        let (inferred, src) = infer_groups(&candidates);
        (
            recs.iter()
                .zip(inferred)
                .map(|(r, g)| r.group.clone().unwrap_or(g))
                .collect(),
            src,
        )
    };
    sources.by_field = by_field;
    // One candidate per assistant turn, deduplicated on the exact pair.
    let mut seen_pair = HashSet::new();
    let mut accepted = Vec::new();
    let mut dropped_pair = 0usize;
    let mut missing = 0usize;
    for (r, group) in recs.iter().zip(&groups) {
        for j in 1..r.turns.len() {
            if r.turns[j].0 != "assistant" || r.turns[j].1.trim().is_empty() {
                continue;
            }
            if r.turns[..j].iter().any(|(_, t)| t.trim().is_empty())
                || !matches!(r.turns[0].0.as_str(), "user" | "prompter" | "system")
            {
                missing += 1;
                continue;
            }
            let context: Vec<(String, String)> = r.turns[..j].to_vec();
            let prompt_key = normalize(&context.last().map(|x| x.1.as_str()).unwrap_or(""));
            let pair_key = format!("{prompt_key}\u{0}{}", normalize(&r.turns[j].1));
            if !seen_pair.insert(pair_key.clone()) {
                dropped_pair += 1;
                continue;
            }
            accepted.push(Candidate {
                tree_id: group.clone(),
                message_id: format!("{}", accepted.len()),
                split: split_for_tree(group),
                lang: r.lang.clone(),
                prompt_key,
                pair_key,
                task: classify_task(&context, &r.turns[j].1),
                context,
                answer: r.turns[j].1.clone(),
            });
        }
    }
    let bpe = Bpe::load(tokenizer).context("load native BPE tokenizer")?;
    let mut cache = HashMap::new();
    let mut out: HashMap<Split, (Vec<u16>, Vec<u16>)> = HashMap::new();
    let mut stats: BTreeMap<String, SplitStats> = BTreeMap::new();
    let mut split_trees: HashMap<String, HashSet<String>> = HashMap::new();
    let mut group_split: BTreeMap<String, String> = BTreeMap::new();
    let mut dropped_long = 0usize;
    for c in accepted {
        let Some((tokens, targets, truncated)) =
            encode_record(&bpe, &c.context, &c.answer, seq, &mut cache)?
        else {
            dropped_long += 1;
            continue;
        };
        let entry = out.entry(c.split).or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.extend_from_slice(&tokens);
        entry.1.extend_from_slice(&targets);
        let s = stats.entry(c.split.name().to_string()).or_default();
        s.examples += 1;
        s.valid_answer_tokens += targets.iter().filter(|&&x| x != IGNORE).count();
        s.truncated_answers += truncated as usize;
        *s.langs.entry(c.lang).or_default() += 1;
        *s.tasks.entry(c.task).or_default() += 1;
        split_trees
            .entry(c.split.name().to_string())
            .or_default()
            .insert(c.tree_id.clone());
        group_split.insert(c.tree_id, c.split.name().to_string());
    }
    for (name, s) in &mut stats {
        s.trees = split_trees.get(name).map_or(0, HashSet::len);
    }
    let save_split = |split: Split, path: &Path| -> anyhow::Result<()> {
        let (tok, tgt) = out
            .get(&split)
            .cloned()
            .unwrap_or_else(|| (Vec::new(), Vec::new()));
        let records = stats.get(split.name()).map(|x| x.examples).unwrap_or(0);
        ensure!(records > 0, "{} split is empty", split.name());
        SftShard::new(seq, records, tok, tgt).save(path)
    };
    save_split(Split::Train, train_out)?;
    save_split(Split::Dev, dev_out)?;
    save_split(Split::Final, final_out)?;
    let distinct_groups = groups.iter().collect::<HashSet<_>>().len();
    let manifest = PrepareStats {
        source_file: input.display().to_string(),
        source_revision: "messages-jsonl".to_string(),
        source_sha256: fnv1a64_file(input)?,
        license: "unspecified".to_string(),
        template: "cmf-im-v1: <|im_start|>role\\ntext<|im_end|>\\n; assistant answer + <|im_end|><|endoftext|> are valid targets".to_string(),
        sequence: seq,
        parsed_messages: recs.iter().map(|r| r.turns.len()).sum(),
        parsed_trees: distinct_groups,
        eligible_assistant_messages: recs
            .iter()
            .map(|r| r.turns.iter().skip(1).filter(|t| t.0 == "assistant").count())
            .sum(),
        dropped_duplicate_pair: dropped_pair,
        dropped_duplicate_prompt: 0,
        dropped_missing_path: missing + skipped_lines,
        dropped_empty_or_overlong: dropped_long,
        splits: stats,
        group_sources: Some(sources),
        groups: Some(group_split),
    };
    let text = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(manifest_out, text)?;
    println!(
        "sft messages: {} records, {} groups ({:?}), train/dev/final={}/{}/{} records; manifest={}",
        recs.len(),
        distinct_groups,
        manifest.group_sources.as_ref().unwrap(),
        manifest.splits.get("train").map_or(0, |s| s.examples),
        manifest.splits.get("dev").map_or(0, |s| s.examples),
        manifest.splits.get("final").map_or(0, |s| s.examples),
        manifest_out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sft_round_trip_and_mask() {
        let shard = SftShard::new(
            8,
            1,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, 8, 9],
        );
        let path = std::env::temp_dir().join(format!("cortiq-sft-{}.bin", std::process::id()));
        shard.save(&path).unwrap();
        let got = SftShard::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(got.seq, 8);
        assert_eq!(got.records, 1);
        assert_eq!(got.valid_tokens(), 2);
        assert_eq!(got.record(0).1[0], IGNORE);
    }

    #[test]
    fn latin_binomials_skip_sentence_openers_and_stop_words() {
        let got = latin_binomials(
            "Как выглядит Pelargonium sidoides? Pelargonium reniforme похож. It forms a rosette (Salix lasiolepis).",
        );
        // the first name is mid-sentence, the second opens a sentence, "It forms"
        // is stopped by the ending rule, the parenthesised one counts
        assert_eq!(got, vec!["Pelargonium sidoides", "Salix lasiolepis"]);
        assert!(latin_binomials("The source says nothing. Root extract only.").is_empty());
        assert_eq!(latin_binomials("Where does arroyo willow grow?"), Vec::<String>::new());
    }

    #[test]
    fn infer_groups_keeps_blocks_and_ignores_passing_mentions() {
        let a = "Alpha prima".to_string();
        let b = "Beta secunda".to_string();
        let c = "Gamma tertia".to_string();
        // block A (6 records, one names only the look-alike C), then block B
        let cands = vec![
            vec![a.clone(), c.clone()],
            vec![],
            vec![c.clone()],
            vec![a.clone()],
            vec![],
            vec![a.clone()],
            vec![b.clone()],
            vec![],
            vec![b.clone()],
        ];
        let (groups, src) = infer_groups(&cands);
        assert_eq!(groups, vec![&a, &a, &a, &a, &a, &a, &b, &b, &b].into_iter().cloned().collect::<Vec<_>>());
        assert_eq!(src.inferred, 5);
        assert_eq!(src.carried, 4);
        assert_eq!(src.backfilled + src.singleton, 0);
        // records before the first named one join it; a nameless corpus is singletons
        let (g2, s2) = infer_groups(&[vec![], vec![a.clone()], vec![]]);
        assert_eq!(g2, vec![a.clone(), a.clone(), a.clone()]);
        assert_eq!(s2.backfilled, 1);
        let (g3, s3) = infer_groups(&[vec![], vec![]]);
        assert_eq!(g3, vec!["record-0", "record-1"]);
        assert_eq!(s3.singleton, 2);
    }

    #[test]
    fn split_and_normalization_are_stable() {
        assert_eq!(normalize("  WHAT?!  is\nthis? "), "what is this");
        assert_eq!(
            split_for_tree("00000000-0000-0000-0000-000000000000"),
            split_for_tree("00000000-0000-0000-0000-000000000000")
        );
    }

    #[test]
    fn sampler_skip_preserves_the_continuation_stream() {
        let sft = SftShard::new(8, 5, (0..45).collect(), vec![1; 40]);
        let raw = Shard {
            tokens: (0..128).map(|x| x as u16).collect(),
        };
        let mut continuous = SftSampler::new(5, 8, 17);
        let mut skipped = SftSampler::new(5, 8, 17);
        let mut tokens = Vec::new();
        let mut targets = Vec::new();
        continuous.batch(&sft, &raw, &mut tokens, &mut targets);
        continuous.batch(&sft, &raw, &mut tokens, &mut targets);
        let expected_tokens = tokens.clone();
        let expected_targets = targets.clone();
        skipped.skip_batches(1);
        skipped.batch(&sft, &raw, &mut tokens, &mut targets);
        assert_eq!(tokens, expected_tokens);
        assert_eq!(targets, expected_targets);
    }
}
