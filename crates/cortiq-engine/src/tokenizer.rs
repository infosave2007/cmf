//! Byte-level BPE tokenizer — HF tokenizer.json parity.
//!
//! Faithful pipeline (matches `tokenizers` for Qwen-style files):
//!   added-token split (raw text) → NFC → pre-tokenizer regex
//!   (GPT-2 style, needs lookahead) → byte-level mapping → ranked BPE
//!   merges → vocab ids. Decode reverses through the byte-level map,
//!   assembling UTF-8 across token boundaries.
//!
//! No silent corruption: a symbol that cannot be encoded is reported
//! (tracing::error), never dropped without a trace.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

/// GPT-2 pre-tokenizer pattern — used when tokenizer.json carries no
/// explicit Split regex (Qwen files carry their own; see `from_json`).
const DEFAULT_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A loaded BPE tokenizer.
pub struct Tokenizer {
    /// Token string → ID
    vocab: HashMap<String, u32>,
    /// ID → Token string
    id_to_token: Vec<String>,
    /// BPE merge ranks: (left, right) → rank (lower merges first)
    ranks: HashMap<(String, String), u32>,
    /// All added tokens (split during encode; emitted raw at decode)
    added: Vec<(String, u32)>,
    /// IDs of added tokens (decode: emit content raw, no byte-map)
    added_ids: HashSet<u32>,
    /// IDs of special tokens (skipped by `decode`)
    special_ids: HashSet<u32>,
    /// Pre-tokenizer split pattern (None = whitespace fallback for the
    /// synthetic `byte_level()` tokenizer)
    /// Applied in order; each subdivides the previous stage's pieces.
    split_res: Vec<fancy_regex::Regex>,
    /// SentencePiece Prepend("▁") normalizer present (llama family).
    /// Gemma replaces spaces with ▁ but does NOT prepend one.
    sp_prepend: bool,
    /// Metaspace `prepend_scheme: "first"` in the PRE-tokenizer (no
    /// Prepend normalizer): the ▁ goes on the very first section of the
    /// input only, so text after an added token gets none. Nanbeige 4.2
    /// is this shape — reading only the normalizer left every raw prompt
    /// short one leading ▁ ("Hello" instead of "▁Hello").
    sp_prepend_first: bool,
    /// SentencePiece family (TinyLlama/Llama-2/Mistral): metaspace ▁
    /// normalization + byte_fallback, no byte-level alphabet.
    metaspace: bool,
    /// NFC only when the file's normalizer declares it (Qwen does,
    /// TinyLlama does not — forcing it broke combining-accent parity).
    nfc: bool,
    /// byte → byte-level char (GPT-2 visible-alphabet mapping)
    byte_to_char: [char; 256],
    /// byte-level char → byte
    char_to_byte: HashMap<char, u8>,
    /// Special tokens
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    /// Chat template special tokens
    pub im_start_id: Option<u32>,
    pub im_end_id: Option<u32>,
    /// Jinja chat template carried by the container (spec §6.1);
    /// None → hardcoded ChatML fallback.
    pub chat_template: Option<String>,
    /// Extra stop ids from the container's generation config.
    pub extra_eos: HashSet<u32>,
    /// Generation prepends BOS (llama post_processor semantics).
    pub add_bos: bool,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab", &self.vocab.len())
            .field("merges", &self.ranks.len())
            .field("added", &self.added.len())
            .finish()
    }
}

/// GPT-2 byte↔unicode bijection: printable bytes map to themselves,
/// the rest get consecutive codepoints from U+0100 up.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0..=255u16 {
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let c = if printable {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        b2c[b as usize] = c;
        c2b.insert(c, b as u8);
    }
    (b2c, c2b)
}

/// HuggingFace tokenizer.json schema (the parts we execute).
#[derive(Deserialize)]
struct HfTokenizerJson {
    model: HfModel,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct HfModel {
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<HfMerge>,
    #[serde(default)]
    byte_fallback: bool,
}

/// Merge rules come in two HF flavours: legacy `"a b"` strings and
/// modern `["a", "b"]` pairs (Qwen3.5 tokenizer.json uses pairs).
#[derive(Deserialize)]
#[serde(untagged)]
enum HfMerge {
    Pair([String; 2]),
    Text(String),
}

#[derive(Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    special: bool,
}

/// Collect EVERY Split regex from a pre_tokenizer subtree, in order.
///
/// A Sequence applies its splits one after another, each subdividing what
/// the previous one produced — taking only the first is not an
/// approximation, it is a different tokenizer. DeepSeek-V4 puts the digit
/// rule first and the word rule third, so reading one pattern sent whole
/// sentences into BPE as a single piece and produced ids the model has
/// never seen.
fn collect_split_patterns(pt: &serde_json::Value, out: &mut Vec<String>) {
    if pt.get("type").and_then(|t| t.as_str()) == Some("Split") {
        if let Some(r) = pt
            .get("pattern")
            .and_then(|p| p.get("Regex"))
            .and_then(|r| r.as_str())
        {
            out.push(r.to_string());
        }
        return;
    }
    if let Some(list) = pt.get("pretokenizers").and_then(|l| l.as_array()) {
        for p in list {
            collect_split_patterns(p, out);
        }
    }
}

/// `prepend_scheme` of a Metaspace pre-tokenizer ("always" | "first" |
/// "never"), searched through a Sequence.
fn find_prepend_scheme(pt: &serde_json::Value) -> Option<String> {
    if pt.get("type").and_then(|t| t.as_str()) == Some("Metaspace") {
        return pt
            .get("prepend_scheme")
            .and_then(|p| p.as_str())
            .map(String::from);
    }
    if let Some(list) = pt.get("pretokenizers").and_then(|l| l.as_array()) {
        return list.iter().find_map(find_prepend_scheme);
    }
    None
}

impl Tokenizer {
    /// Load tokenizer from HuggingFace tokenizer.json file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let data = std::fs::read_to_string(path.as_ref())
            .map_err(|e| TokenizerError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    /// Load tokenizer from raw tokenizer.json bytes (CMF VOCAB section).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| TokenizerError::Parse(format!("vocab is not UTF-8: {e}")))?;
        Self::from_json(s)
    }

    /// Load tokenizer from JSON string.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let hf: HfTokenizerJson =
            serde_json::from_str(json).map_err(|e| TokenizerError::Parse(e.to_string()))?;

        let mut vocab = hf.model.vocab;
        let mut ranks = HashMap::new();
        for (rank, m) in hf.model.merges.into_iter().enumerate() {
            let (a, b) = match m {
                HfMerge::Pair([a, b]) => (a, b),
                HfMerge::Text(s) => {
                    let mut it = s.splitn(2, ' ');
                    match (it.next(), it.next()) {
                        (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                        _ => continue,
                    }
                }
            };
            ranks.insert((a, b), rank as u32);
        }

        // Family detection: SentencePiece carries byte_fallback and/or a
        // Prepend("▁") normalizer; byte-level BPE carries a Split regex.
        // Llama-family post_processor prepends <s> at add_special_tokens
        // time; generation must honor it (word salad without BOS).
        let mut saw_gemma_bos = false;
        let add_bos_detected = hf
            .post_processor
            .as_ref()
            .map(|p| {
                let pp = p.to_string();
                pp.contains("\"<s>\"") || pp.contains("\"<bos>\"")
            })
            .unwrap_or(false);
        let nfc = hf
            .normalizer
            .as_ref()
            .map(|n| n.to_string().contains("NFC"))
            .unwrap_or(false);
        let metaspace = hf.model.byte_fallback
            || hf
                .normalizer
                .as_ref()
                .map(|n| n.to_string().contains("\u{2581}") || n.to_string().contains("▁"))
                .unwrap_or(false);
        let sp_prepend = hf
            .normalizer
            .as_ref()
            .map(|n| n.to_string().contains("Prepend"))
            .unwrap_or(false);
        // Metaspace can also live in the pre-tokenizer, carrying its own
        // prepend_scheme: "always" behaves like the llama normalizer,
        // "first" only marks the head of the input (see `sp_prepend_first`).
        let (sp_prepend, sp_prepend_first) = if sp_prepend {
            (true, false)
        } else {
            match hf.pre_tokenizer.as_ref().and_then(find_prepend_scheme) {
                Some(s) if s == "always" => (true, false),
                Some(s) if s == "first" => (false, true),
                _ => (false, false),
            }
        };
        let split_res = if metaspace {
            Vec::new()
        } else {
            let mut pats = Vec::new();
            if let Some(pt) = hf.pre_tokenizer.as_ref() {
                collect_split_patterns(pt, &mut pats);
            }
            if pats.is_empty() {
                pats.push(DEFAULT_SPLIT.to_string());
            }
            pats.iter()
                .map(|p| {
                    fancy_regex::Regex::new(p)
                        .map_err(|e| TokenizerError::Parse(format!("pre-tokenizer regex: {e}")))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Added tokens: longest-first so overlapping contents match right.
        let mut bos_token_id = None;
        let mut eos_token_id = None;
        let mut pad_token_id = None;
        let mut im_start_id = None;
        let mut im_end_id = None;
        let mut special_ids = HashSet::new();
        let mut added_ids = HashSet::new();
        let mut added = Vec::new();

        for at in &hf.added_tokens {
            vocab.insert(at.content.clone(), at.id);
            added.push((at.content.clone(), at.id));
            added_ids.insert(at.id);
            if at.special {
                special_ids.insert(at.id);
            }
            match at.content.as_str() {
                "<|endoftext|>" | "</s>" | "[EOS]" => eos_token_id = Some(at.id),
                "<|im_start|>" => im_start_id = Some(at.id),
                "<|im_end|>" => im_end_id = Some(at.id),
                "<s>" | "[BOS]" => bos_token_id = Some(at.id),
                // Gemma spells BOS as literal "<bos>" — the family
                // REQUIRES it on every sequence, and newer tokenizers
                // (gemma-4) no longer say so in a post_processor.
                "<bos>" => {
                    bos_token_id = Some(at.id);
                    saw_gemma_bos = true;
                }
                "<pad>" => pad_token_id = Some(at.id),
                _ => {}
            }
        }
        added.sort_by_key(|(c, _)| std::cmp::Reverse(c.len()));

        // Gemma REQUIRES a leading <bos> on every sequence, but newer
        // tokenizers (gemma-4, 262k vocab) no longer spell it in a
        // post_processor template — the family marker <start_of_turn>
        // is the reliable tell. Without this, raw-text scoring runs
        // unanchored and the first ~30 positions read worse than
        // uniform (the chat path masked it: the template carries <bos>).
        let gemma_family = saw_gemma_bos
            || vocab.contains_key("<start_of_turn>")
            || added.iter().any(|(c, _)| c == "<start_of_turn>");

        // The post_processor template names the exact BOS content
        // (llama "<s>", gemma "<bos>" — gemma's vocab carries BOTH, so
        // added-token scan order must not decide).
        if let Some(pp) = hf.post_processor.as_ref() {
            let pp = pp.to_string();
            for name in ["<bos>", "<s>"] {
                if pp.contains(&format!("\"{name}\"")) {
                    if let Some(&id) = vocab.get(name) {
                        bos_token_id = Some(id);
                    }
                    break;
                }
            }
        }

        // Build reverse map
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
        }

        let (byte_to_char, char_to_byte) = bytes_to_unicode();

        tracing::info!(
            "Tokenizer loaded: {} vocab, {} merges, {} added, eos={:?}",
            vocab.len(),
            ranks.len(),
            added.len(),
            eos_token_id
        );

        Ok(Self {
            vocab,
            id_to_token,
            ranks,
            added,
            added_ids,
            special_ids,
            split_res,
            metaspace,
            sp_prepend,
            sp_prepend_first,
            nfc,
            byte_to_char,
            char_to_byte,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            im_start_id,
            im_end_id,
            chat_template: None,
            extra_eos: HashSet::new(),
            add_bos: add_bos_detected || gemma_family,
        })
    }

    /// Create a minimal tokenizer for testing (byte tokens, no merges).
    pub fn byte_level() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::with_capacity(256);
        for i in 0..256u32 {
            let tok = format!("<0x{:02X}>", i);
            vocab.insert(tok.clone(), i);
            id_to_token.push(tok);
        }
        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        Self {
            vocab,
            id_to_token,
            ranks: HashMap::new(),
            added: Vec::new(),
            added_ids: HashSet::new(),
            special_ids: HashSet::new(),
            split_res: Vec::new(),
            metaspace: false,
            sp_prepend: false,
            sp_prepend_first: false,
            nfc: false,
            byte_to_char,
            char_to_byte,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            im_start_id: None,
            im_end_id: None,
            chat_template: None,
            extra_eos: HashSet::new(),
            add_bos: false,
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        // Added tokens match on raw text (normalized: false), longest first.
        let mut rest = text;
        // `prepend_scheme: "first"` marks only the section that starts at
        // offset 0 — HF drops the ▁ for everything after an added token,
        // which is why a chat prompt opening with <|im_start|> tokenizes
        // the same either way and only raw prompts were wrong.
        let mut head = true;
        'outer: while !rest.is_empty() {
            let mut best: Option<(usize, usize, u32)> = None; // (pos, len, id)
            for (content, id) in &self.added {
                if let Some(pos) = rest.find(content.as_str()) {
                    let better = match best {
                        None => true,
                        Some((bp, bl, _)) => pos < bp || (pos == bp && content.len() > bl),
                    };
                    if better {
                        best = Some((pos, content.len(), *id));
                    }
                    if pos == 0 {
                        break; // earliest possible; added is longest-first
                    }
                }
            }
            match best {
                Some((pos, len, id)) => {
                    self.encode_segment_at(&rest[..pos], head, &mut ids);
                    ids.push(id);
                    rest = &rest[pos + len..];
                    head = false;
                }
                None => {
                    self.encode_segment_at(rest, head, &mut ids);
                    break 'outer;
                }
            }
        }
        ids
    }

    /// Encode one added-token-free segment: NFC → split → byte-map →
    /// BPE. `head` says whether this section starts at offset 0 of the
    /// input — only that one takes a `prepend_scheme: "first"` ▁.
    fn encode_segment_at(&self, segment: &str, head: bool, out: &mut Vec<u32>) {
        if segment.is_empty() {
            return;
        }
        let norm: String = if self.nfc {
            segment.nfc().collect()
        } else {
            segment.to_string()
        };
        if self.metaspace {
            // SentencePiece: [Prepend("▁") +] Replace(" "→"▁"), BPE over
            // chars of the whole span (no pre-tokenizer, no byte map).
            // Gemma's normalizer replaces only — no dummy prefix.
            let sp = if self.sp_prepend {
                // Normalizer order: Prepend THEN Replace, unguarded — so
                // " hello" really does become ▁▁hello on llama.
                format!("\u{2581}{}", norm).replace(' ', "\u{2581}")
            } else {
                // Pre-tokenizer Metaspace: Replace, then prepend only if
                // the span does not already start with ▁ (HF's guard, so
                // " hello" stays one ▁, not two).
                let replaced = norm.replace(' ', "\u{2581}");
                if self.sp_prepend_first && head && !replaced.starts_with('\u{2581}') {
                    format!("\u{2581}{replaced}")
                } else {
                    replaced
                }
            };
            self.bpe_piece_sp(&sp, out);
            return;
        }
        if !self.split_res.is_empty() {
            // Each stage subdivides the pieces the previous one left, with
            // Isolated behaviour: both the matches and the gaps survive.
            let mut pieces: Vec<(usize, usize)> = vec![(0, norm.len())];
            for re in &self.split_res {
                let mut next: Vec<(usize, usize)> = Vec::with_capacity(pieces.len() * 2);
                for (ps, pe) in pieces {
                    let seg = &norm[ps..pe];
                    let mut last = 0usize;
                    for m in re.find_iter(seg) {
                        let m = match m {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::error!("pre-tokenizer regex failed: {e}");
                                break;
                            }
                        };
                        if m.start() > last {
                            next.push((ps + last, ps + m.start()));
                        }
                        if m.end() > m.start() {
                            next.push((ps + m.start(), ps + m.end()));
                        }
                        last = m.end();
                    }
                    if last < seg.len() {
                        next.push((ps + last, pe));
                    }
                }
                pieces = next;
            }
            for (ps, pe) in pieces {
                self.bpe_piece(&norm[ps..pe], out);
            }
        } else {
            {
                // Synthetic byte_level() tokenizer: raw byte tokens.
                for b in norm.bytes() {
                    let tok = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.vocab.get(&tok) {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// SentencePiece BPE: symbols are chars (no byte-level alphabet);
    /// unknown symbols fall back to <0xNN> tokens per UTF-8 byte.
    fn bpe_piece_sp(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let mut sym: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", sym[i], sym[i + 1]);
            let (left, right) = (sym[i].clone(), sym[i + 1].clone());
            let mut j = 0;
            while j + 1 < sym.len() {
                if sym[j] == left && sym[j + 1] == right {
                    sym[j] = merged.clone();
                    sym.remove(j + 1);
                }
                j += 1;
            }
        }
        for t in &sym {
            if let Some(&id) = self.vocab.get(t) {
                out.push(id);
            } else {
                let mut ok = true;
                for byte in t.bytes() {
                    let tok = format!("<0x{:02X}>", byte);
                    match self.vocab.get(&tok) {
                        Some(&id) => out.push(id),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    tracing::error!("tokenizer: no id for SP symbol {t:?} — dropped");
                }
            }
        }
    }

    /// Byte-level map one pre-token piece, then ranked BPE merges.
    fn bpe_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let mapped: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();
        let mut sym = mapped;

        // Classic BPE: repeatedly merge the lowest-rank adjacent pair.
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", sym[i], sym[i + 1]);
            // Merge ALL occurrences of this exact pair, left to right.
            let (left, right) = (sym[i].clone(), sym[i + 1].clone());
            let mut j = 0;
            while j + 1 < sym.len() {
                if sym[j] == left && sym[j + 1] == right {
                    sym[j] = merged.clone();
                    sym.remove(j + 1);
                }
                j += 1;
            }
        }

        for s in &sym {
            if let Some(&id) = self.vocab.get(s) {
                out.push(id);
            } else {
                // Byte-fallback (synthetic vocabs); never drop silently.
                let mut ok = true;
                for ch in s.chars() {
                    let Some(&b) = self.char_to_byte.get(&ch) else {
                        ok = false;
                        break;
                    };
                    let tok = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.vocab.get(&tok) {
                        out.push(id);
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    tracing::error!("tokenizer: no id for symbol {s:?} — dropped");
                }
            }
        }
    }

    /// Decode token IDs back to text. Special tokens are skipped; added
    /// tokens are raw text; everything else reverses the byte-level map.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if self.special_ids.contains(&id) {
                continue;
            }
            let idx = id as usize;
            if idx >= self.id_to_token.len() {
                continue;
            }
            let tok = &self.id_to_token[idx];
            if self.added_ids.contains(&id) {
                // Gemma-3n declares its multi-space ▁-runs as ADDED
                // tokens — verbatim passthrough leaked ▁ into output.
                if self.metaspace && tok.contains('\u{2581}') {
                    bytes.extend_from_slice(tok.replace('\u{2581}', " ").as_bytes());
                } else {
                    bytes.extend_from_slice(tok.as_bytes());
                }
                continue;
            }
            // Byte-fallback / legacy byte tokens
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            if self.metaspace {
                // SP decoder: Replace(▁→" "); UTF-8 chars pass through.
                for ch in tok.chars() {
                    if ch == '\u{2581}' {
                        bytes.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                continue;
            }
            for ch in tok.chars() {
                match self.char_to_byte.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // Not a byte-level char (shouldn't happen for real
                    // vocabs) — pass the char through as UTF-8.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        if self.metaspace && (self.sp_prepend || self.sp_prepend_first) {
            // SP decoder Strip(start=1): one leading space from Prepend.
            if let Some(stripped) = text.strip_prefix(' ') {
                return stripped.to_string();
            }
        }
        text
    }

    /// Streaming decode of ONE token: no sequence-level Strip — a
    /// per-token strip would eat the ▁-spaces of every SP word.
    pub fn decode_token(&self, id: u32) -> String {
        if self.special_ids.contains(&id) {
            return String::new();
        }
        let idx = id as usize;
        if idx >= self.id_to_token.len() {
            return String::new();
        }
        let tok = &self.id_to_token[idx];
        if self.added_ids.contains(&id) {
            if self.metaspace && tok.contains('\u{2581}') {
                return tok.replace('\u{2581}', " ");
            }
            return tok.clone();
        }
        if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
            if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                return String::from_utf8_lossy(&[b]).into_owned();
            }
        }
        if self.metaspace {
            return tok.replace('\u{2581}', " ");
        }
        let mut bytes = Vec::new();
        for ch in tok.chars() {
            match self.char_to_byte.get(&ch) {
                Some(&b) => bytes.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Render the container's Jinja chat template (HF semantics:
    /// trim_blocks + lstrip_blocks + loop controls) and encode it.
    /// Falls back to hardcoded ChatML when the file carries none.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> Vec<u32> {
        self.apply_chat_template_opts(messages, None)
    }

    /// Like `apply_chat_template`, with an explicit `enable_thinking` value for
    /// reasoning-model templates (Qwen3/3.5 emit an empty <think> block when it
    /// is false, so the model answers directly). `None` leaves the variable
    /// undefined — the template's own default applies.

    /// Chat template with the FULL message shape and a tool list.
    ///
    /// The pair-based API below flattens every message to (role, text),
    /// which silently drops exactly what agentic use needs: the `tools`
    /// array, `role: "tool"` results, and `tool_calls` on assistant
    /// turns. The templates this format embeds — Qwen-family, Nanbeige —
    /// have carried a `{%- if tools %}` branch all along; this is the
    /// call that finally feeds it. Messages arrive as JSON objects in
    /// the OpenAI shape and pass through to minijinja unflattened, so a
    /// template sees the same fields a Python `apply_chat_template`
    /// would.
    pub fn apply_chat_template_json(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: Option<bool>,
    ) -> Vec<u32> {
        if let Some(tpl) = &self.chat_template {
            match self.render_template_json(tpl, messages, tools, enable_thinking) {
                Ok(text) => return self.with_bos(self.encode(&text)),
                Err(e) => {
                    tracing::error!("chat template render failed ({e}); ChatML fallback");
                }
            }
        }
        let pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| {
                (
                    m.get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("user")
                        .to_string(),
                    m.get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        self.with_bos(self.chatml_fallback_opts(&pairs, enable_thinking))
    }

    /// Render the template against JSON-shaped messages (parity surface).
    pub fn render_chat_json(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: Option<bool>,
    ) -> Option<String> {
        let tpl = self.chat_template.as_ref()?;
        match self.render_template_json(tpl, messages, tools, enable_thinking) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::error!("chat template render (json): {e:#}");
                eprintln!("chat template render (json): {e:#}");
                None
            }
        }
    }

    fn render_template_json(
        &self,
        tpl: &str,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: Option<bool>,
    ) -> Result<String, minijinja::Error> {
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // `visible_text` is a helper transformers injects into its
        // template env (it flattens multimodal content to its text).
        // Nanbeige's template calls it unconditionally in the tools
        // branch; without it the render errors and the fallback quietly
        // serves a TOOLLESS prompt.
        env.add_function("visible_text", |v: minijinja::Value| -> String {
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
        env.add_template("chat", tpl)?;
        let msgs: Vec<minijinja::Value> = messages
            .iter()
            .map(minijinja::Value::from_serialize)
            .collect();
        let tools_v: Option<Vec<minijinja::Value>> =
            tools.map(|ts| ts.iter().map(minijinja::Value::from_serialize).collect());
        let tpl = env.get_template("chat")?;
        // Three axes, each present only when meaningful: templates guard
        // with `is defined`, and an explicit null flips those guards.
        // `tool_call_format` picks the grammar in two-mode templates
        // (Nanbeige): undefined falls into their XML branch. The JSON
        // grammar is the one every parser downstream speaks, so choose
        // it explicitly; templates without the knob never read it.
        let rendered = match (tools_v, enable_thinking) {
            (Some(ts), Some(v)) => tpl.render(minijinja::context! {
                messages => msgs, tools => ts, add_generation_prompt => true, enable_thinking => v,
                tool_call_format => "json",
            })?,
            (Some(ts), None) => tpl.render(minijinja::context! {
                messages => msgs, tools => ts, add_generation_prompt => true,
                tool_call_format => "json",
            })?,
            (None, Some(v)) => tpl.render(minijinja::context! {
                messages => msgs, add_generation_prompt => true, enable_thinking => v,
                tool_call_format => "json",
            })?,
            (None, None) => tpl.render(minijinja::context! {
                messages => msgs, add_generation_prompt => true,
                tool_call_format => "json",
            })?,
        };
        Ok(rendered)
    }

    pub fn apply_chat_template_opts(
        &self,
        messages: &[(String, String)],
        enable_thinking: Option<bool>,
    ) -> Vec<u32> {
        if let Some(tpl) = &self.chat_template {
            match self.render_template(tpl, messages, enable_thinking) {
                Ok(text) => return self.with_bos(self.encode(&text)),
                Err(e) => {
                    tracing::error!("chat template render failed ({e}); ChatML fallback");
                }
            }
        }
        self.with_bos(self.chatml_fallback_opts(messages, enable_thinking))
    }

    /// Prepend BOS when the tokenizer declares it (llama family).
    pub fn with_bos(&self, mut ids: Vec<u32>) -> Vec<u32> {
        if self.add_bos {
            if let Some(b) = self.bos_token_id {
                if ids.first() != Some(&b) {
                    ids.insert(0, b);
                }
            }
        }
        ids
    }

    /// Render the carried template to text (parity-testable surface).
    pub fn render_chat(&self, messages: &[(String, String)]) -> Option<String> {
        self.render_chat_opts(messages, None)
    }

    /// Render the carried template to text with explicit thinking mode.
    pub fn render_chat_opts(
        &self,
        messages: &[(String, String)],
        enable_thinking: Option<bool>,
    ) -> Option<String> {
        let tpl = self.chat_template.as_ref()?;
        match self.render_template(tpl, messages, enable_thinking) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::error!("chat template render: {e:#}");
                None
            }
        }
    }

    fn render_template(
        &self,
        tpl: &str,
        messages: &[(String, String)],
        enable_thinking: Option<bool>,
    ) -> Result<String, minijinja::Error> {
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // HF templates use python string methods (.startswith, .strip…).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", tpl)?;
        let msgs: Vec<minijinja::Value> = messages
            .iter()
            .map(|(role, content)| {
                minijinja::context! { role => role, content => content }
            })
            .collect();
        // `enable_thinking` stays UNDEFINED when None — reasoning templates
        // check `enable_thinking is defined` and fall back to their default.
        let rendered = match enable_thinking {
            Some(v) => env.get_template("chat")?.render(minijinja::context! {
                messages => msgs,
                add_generation_prompt => true,
                enable_thinking => v,
            })?,
            None => env.get_template("chat")?.render(minijinja::context! {
                messages => msgs,
                add_generation_prompt => true,
            })?,
        };
        // Templates that ignore `enable_thinking` (e.g. Nanbeige/Qwen-legacy)
        // always emit a generation prompt. When thinking is explicitly disabled,
        // prefill an empty <think>…</think> block so the model answers directly.
        if enable_thinking == Some(false) && !rendered.contains("</think>") {
            if let Some(pos) = rendered.rfind("assistant") {
                let mut insert_at = pos + "assistant".len();
                if let Some(idx) = rendered[insert_at..].find('\n') {
                    insert_at += idx + 1;
                }
                let mut out = String::with_capacity(rendered.len() + 24);
                out.push_str(&rendered[..insert_at]);
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("<think>\n\n</think>\n\n");
                out.push_str(&rendered[insert_at..]);
                return Ok(out);
            }
        }
        Ok(rendered)
    }

    /// Hardcoded Qwen ChatML (pre-§6.1 files).
    fn chatml_fallback(&self, messages: &[(String, String)]) -> Vec<u32> {
        self.chatml_fallback_opts(messages, None)
    }

    /// Hardcoded Qwen ChatML (pre-§6.1 files) with optional thinking suppression.
    fn chatml_fallback_opts(
        &self,
        messages: &[(String, String)],
        enable_thinking: Option<bool>,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();

        for (role, content) in messages {
            // <|im_start|>role\ncontent<|im_end|>\n
            if let Some(start_id) = self.im_start_id {
                tokens.push(start_id);
            }
            tokens.extend(self.encode(&format!("{}\n{}", role, content)));
            if let Some(end_id) = self.im_end_id {
                tokens.push(end_id);
            }
            tokens.extend(self.encode("\n"));
        }

        // Add assistant prefix
        if let Some(start_id) = self.im_start_id {
            tokens.push(start_id);
        }
        tokens.extend(self.encode("assistant\n"));
        if enable_thinking == Some(false) {
            tokens.extend(self.encode("<think>\n\n</think>\n\n"));
        }

        tokens
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if token ID is EOS.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_token_id == Some(id) || self.im_end_id == Some(id) || self.extra_eos.contains(&id)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_unicode_bijection() {
        let (b2c, c2b) = bytes_to_unicode();
        for b in 0..=255u8 {
            assert_eq!(c2b[&b2c[b as usize]], b);
        }
        // GPT-2 well-known mappings: space → Ġ, newline → Ċ
        assert_eq!(b2c[b' ' as usize], 'Ġ');
        assert_eq!(b2c[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn byte_level_roundtrip_utf8() {
        let tok = Tokenizer::byte_level();
        let text = "hello 🌍 hi\n";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len()); // one id per byte
        assert_eq!(tok.decode(&ids), text);
    }

    /// A tiny real-format tokenizer.json exercising the full pipeline:
    /// GPT-2 regex, byte-level alphabet, one merge, an added token.
    fn mini_json() -> String {
        // vocab: byte-level chars for h,e,l,o,Ġ,w,r,d + merged "he"
        let vocab: Vec<(&str, u32)> = vec![
            ("h", 0),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("Ġ", 4),
            ("w", 5),
            ("r", 6),
            ("d", 7),
            ("he", 8),
            ("Ġw", 9),
        ];
        let vocab_json: String = vocab
            .iter()
            .map(|(t, i)| format!("\"{t}\": {i}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"{{
              "model": {{
                "type": "BPE",
                "vocab": {{ {vocab_json} }},
                "merges": [["h", "e"], ["Ġ", "w"]]
              }},
              "added_tokens": [
                {{"id": 10, "content": "<|eot|>", "special": true}}
              ]
            }}"#
        )
    }

    /// Parity against HuggingFace on a real tokenizer, run only when the
    /// file is present (CMF_TOK_PARITY=/path/to/tokenizer.json).
    #[test]
    fn real_tokenizer_parity_when_available() {
        let Ok(path) = std::env::var("CMF_TOK_PARITY") else {
            return;
        };
        let t = Tokenizer::from_file(&path).expect("load");
        for (text, want) in [
            (
                "The capital of France is",
                vec![671u32, 6102, 294, 8760, 344],
            ),
            ("2 + 2 =", vec![20, 940, 223, 20, 438]),
        ] {
            let got = t.encode(text);
            assert_eq!(got, want, "«{text}»");
        }
    }

    /// A Sequence of Splits applies ALL of them, in order. Reading only the
    /// first is not a near-miss: DeepSeek-V4 puts a digit rule first and the
    /// word rule third, so one-pattern behaviour hands BPE a whole sentence
    /// as a single piece and the ids that come back are ones the model was
    /// never trained on.
    #[test]
    fn every_split_in_a_sequence_is_applied() {
        let pt = serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "behavior": "Isolated",
                 "pattern": {"Regex": r"\p{N}{1,3}"}},
                {"type": "Split", "behavior": "Isolated",
                 "pattern": {"Regex": r" ?[\p{L}]+"}},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
        });
        let mut pats = Vec::new();
        collect_split_patterns(&pt, &mut pats);
        assert_eq!(
            pats.len(),
            2,
            "both Split stages must be collected: {pats:?}"
        );
        assert!(pats[0].contains("p{N}"), "digit rule first");
        assert!(pats[1].contains("p{L}"), "word rule second");

        // And the staged subdivision reaches the word boundaries. Building a
        // byte-level tokenizer over this pre_tokenizer, "ab cd" has to become
        // two pieces rather than one.
        let re: Vec<fancy_regex::Regex> = pats
            .iter()
            .map(|p| fancy_regex::Regex::new(p).unwrap())
            .collect();
        let norm = "ab cd12";
        let mut pieces: Vec<(usize, usize)> = vec![(0, norm.len())];
        for r in &re {
            let mut next = Vec::new();
            for (ps, pe) in pieces {
                let seg = &norm[ps..pe];
                let mut last = 0;
                for m in r.find_iter(seg).flatten() {
                    if m.start() > last {
                        next.push((ps + last, ps + m.start()));
                    }
                    if m.end() > m.start() {
                        next.push((ps + m.start(), ps + m.end()));
                    }
                    last = m.end();
                }
                if last < seg.len() {
                    next.push((ps + last, pe));
                }
            }
            pieces = next;
        }
        let got: Vec<&str> = pieces.iter().map(|(a, b)| &norm[*a..*b]).collect();
        assert_eq!(
            got,
            vec!["ab", " cd", "12"],
            "staged split produced {got:?}"
        );
    }

    #[test]
    fn full_pipeline_merges_and_added_tokens() {
        let tok = Tokenizer::from_json(&mini_json()).unwrap();
        // "hello world" → [he,l,l,o, Ġw,o,r,l,d]
        let ids = tok.encode("hello world");
        assert_eq!(ids, vec![8, 2, 2, 3, 9, 3, 6, 2, 7]);
        assert_eq!(tok.decode(&ids), "hello world");
        // Added token splits and is skipped at decode (special).
        let ids2 = tok.encode("he<|eot|>he");
        assert_eq!(ids2, vec![8, 10, 8]);
        assert_eq!(tok.decode(&ids2), "hehe");
    }

    #[test]
    fn non_ascii_is_never_silently_dropped() {
        let tok = Tokenizer::from_json(&mini_json()).unwrap();
        // A non-ASCII char is not encodable by the mini vocab (no byte tokens either):
        // the id list may be empty, but ASCII around it must survive.
        let ids = tok.encode("hello");
        assert!(!ids.is_empty());
    }
}
