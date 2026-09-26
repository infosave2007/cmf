//! HF-exact BERT normalizer, pre-tokenizer and WordPiece (spec §1.3).
//!
//! A copy of what HF `tokenizers` does with the encoder's `tokenizer.json`
//! (`Tokenizer::encode(text, add_special_tokens = true)`), step by step:
//!
//! 1. **Added tokens.** The special tokens (`normalized: false`, no
//!    `lstrip`/`rstrip`/`single_word`) are cut out of the *raw* text first,
//!    leftmost-longest, case-sensitive: `"a[UNK]b"` is `a`, `[UNK]`, `b`. They
//!    are the vocab entries at the record's special ids (pad, unk, cls, sep,
//!    mask); the export tool checks that `tokenizer.json` lists exactly those.
//! 2. **`BertNormalizer{clean_text, handle_chinese_chars, strip_accents: null,
//!    lowercase: true}`** on every other piece, in HF's order:
//!    * clean_text: drop U+0000, U+FFFD and `is_other` (Cc, Cf, Co — the
//!      `unicode_categories` definition, no Cn) except `\t\n\r`; map
//!      whitespace (`\t\n\r` and `char::is_whitespace`) to `' '`;
//!    * CJK ideographs (HF's eight blocks) get a space on each side;
//!    * strip accents (on, because `strip_accents` is null and `lowercase` is
//!      true): NFD, then drop Mn. NFD is `unicode-normalization` applied only
//!      to scalars that Unicode 9.0 assigns — HF's NFD tables are Unicode 9.0,
//!      so a later character is a starter without a decomposition there;
//!    * `char::to_lowercase` per character (no final-sigma context: `Σ` → `σ`).
//! 3. **`BertPreTokenizer`**: split on `char::is_whitespace` (removed), then
//!    every punctuation character (ASCII punctuation or Unicode P*) is its own
//!    piece; empty pieces vanish.
//! 4. **`WordPiece{unk, "##", max_input_chars_per_word}`**: greedy longest
//!    match from the left, continuation pieces prefixed; a word longer than
//!    `max_input_chars_per_word` characters, or one with an unmatched rest, is
//!    a single `[UNK]`.
//! 5. **Template and truncation**: the pieces are cut to `max_length − 2` on
//!    the right, then `[CLS] … [SEP]`.
//!
//! Category tables: [`crate::unicode_tables`] (generated from the crates HF
//! `tokenizers` 0.21.4 locks). `cortiq_engine::tokenizer::Tokenizer` is never
//! used: it reads a WordPiece `tokenizer.json` as BPE without merges.

use crate::manifest::{
    NormalizerRecord, PRE_TOKENIZER, SpecialIds, TEMPLATE, TOKENIZER_MODEL, TRUNCATION_DIRECTION,
    TokenizerRecord,
};
use crate::unicode_tables;
use anyhow::{Result, bail, ensure};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// The HF BERT whitespace test: `\t\n\r` or `char::is_whitespace`.
#[inline]
pub fn is_bert_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// The HF BERT control test: `is_other` (Cc, Cf, Co), except `\t\n\r`.
#[inline]
pub fn is_bert_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && unicode_tables::is_other(c)
}

/// HF's "chinese character": the CJK ideograph blocks it spaces out.
#[inline]
pub fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// `BertPreTokenizer`'s punctuation: ASCII punctuation or Unicode P*.
#[inline]
pub fn is_bert_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || unicode_tables::is_punctuation(c)
}

/// NFD of `s` without Mn, appended to `out`: `unicode-normalization` on runs of
/// Unicode 9.0-assigned scalars; any other scalar is copied as a starter.
fn push_nfd_without_mn(s: &str, out: &mut String) {
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if !unicode_tables::is_assigned(c) {
            out.extend(
                s[start..i]
                    .nfd()
                    .filter(|&d| !unicode_tables::is_mark_nonspacing(d)),
            );
            out.push(c);
            start = i + c.len_utf8();
        }
    }
    out.extend(
        s[start..]
            .nfd()
            .filter(|&d| !unicode_tables::is_mark_nonspacing(d)),
    );
}

/// `BertNormalizer{clean_text, handle_chinese_chars, strip_accents: null,
/// lowercase: true}` of one piece of text (no added tokens in it).
pub fn normalize(text: &str) -> String {
    // clean_text + handle_chinese_chars (both per character).
    let mut spaced = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if c == '\0' || c == '\u{FFFD}' || is_bert_control(c) {
            continue;
        }
        let c = if is_bert_whitespace(c) { ' ' } else { c };
        if is_chinese_char(c) {
            spaced.push(' ');
            spaced.push(c);
            spaced.push(' ');
        } else {
            spaced.push(c);
        }
    }
    // strip accents: NFD, drop Mn.
    let mut stripped = String::with_capacity(spaced.len());
    push_nfd_without_mn(&spaced, &mut stripped);
    // lowercase, character by character.
    let mut out = String::with_capacity(stripped.len());
    for c in stripped.chars() {
        out.extend(c.to_lowercase());
    }
    out
}

/// `BertPreTokenizer` of normalized text: whitespace removed, every
/// punctuation character isolated, empty pieces dropped.
pub fn pre_tokenize(normalized: &str) -> Vec<&str> {
    let mut words = Vec::new();
    for chunk in normalized.split(char::is_whitespace) {
        let mut start = 0;
        for (i, c) in chunk.char_indices() {
            if is_bert_punctuation(c) {
                if start < i {
                    words.push(&chunk[start..i]);
                }
                let end = i + c.len_utf8();
                words.push(&chunk[i..end]);
                start = end;
            }
        }
        if start < chunk.len() {
            words.push(&chunk[start..]);
        }
    }
    words
}

/// A piece of the raw text: an added (special) token or text to normalize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Piece<'a> {
    Special(u32),
    Text(&'a str),
}

/// Parse vocab.txt as HF `WordPiece::read_file` does: one token per line
/// (`\n` or `\r\n`), `trim_end`, id = line index.
pub fn parse_vocab(bytes: &[u8]) -> Result<Vec<String>> {
    let text =
        std::str::from_utf8(bytes).map_err(|e| anyhow::anyhow!("vocab is not UTF-8: {e}"))?;
    Ok(text.lines().map(|l| l.trim_end().to_string()).collect())
}

/// The WordPiece tokenizer of a decision encoder.
#[derive(Clone, Debug)]
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    unk: u32,
    cls: u32,
    sep: u32,
    prefix: String,
    max_input_chars_per_word: usize,
    max_length: usize,
    /// Added tokens, matched on the raw text (content, id), longest first.
    specials: Vec<(String, u32)>,
}

impl WordPiece {
    /// The tokenizer of a decision file: the vocab.txt bytes and the manifest's
    /// tokenizer record.
    pub fn from_record(vocab_bytes: &[u8], rec: &TokenizerRecord) -> Result<Self> {
        ensure!(
            rec.model == TOKENIZER_MODEL,
            "tokenizer model '{}' is not wordpiece",
            rec.model
        );
        ensure!(
            rec.normalizer == NormalizerRecord::bert_uncased(),
            "only the uncased BertNormalizer (clean_text, handle_chinese_chars, strip_accents null, lowercase) is implemented"
        );
        ensure!(
            rec.pre_tokenizer == PRE_TOKENIZER
                && rec.template == TEMPLATE
                && rec.truncation.direction == TRUNCATION_DIRECTION,
            "only the bert pre-tokenizer, the '[CLS] $A [SEP]' template and right truncation are implemented"
        );
        Self::new(
            vocab_bytes,
            &rec.unk,
            &rec.prefix,
            rec.max_input_chars_per_word as usize,
            &rec.ids,
            rec.truncation.max_length as usize,
        )
    }

    /// A tokenizer from its parts. Every vocab token must be distinct, so that
    /// the map equals the `tokenizer.json` vocab it was exported with.
    pub fn new(
        vocab_bytes: &[u8],
        unk: &str,
        prefix: &str,
        max_input_chars_per_word: usize,
        ids: &SpecialIds,
        max_length: usize,
    ) -> Result<Self> {
        let tokens = parse_vocab(vocab_bytes)?;
        ensure!(!tokens.is_empty(), "empty vocab");
        ensure!(
            tokens.len() <= u32::MAX as usize,
            "vocab larger than u32 ids"
        );
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            if vocab.insert(t.clone(), i as u32).is_some() {
                bail!("vocab token {t:?} appears twice (line {})", i + 1);
            }
        }
        let n = tokens.len() as u64;
        let special = |name: &str, id: u64| -> Result<u32> {
            ensure!(id < n, "special id {name} {id} outside the vocab of {n}");
            ensure!(
                !tokens[id as usize].is_empty(),
                "special token {name} (id {id}) is empty"
            );
            Ok(id as u32)
        };
        let pad = special("pad", ids.pad)?;
        let unk_id = special("unk", ids.unk)?;
        let cls = special("cls", ids.cls)?;
        let sep = special("sep", ids.sep)?;
        let mask = special("mask", ids.mask)?;
        ensure!(
            tokens[unk_id as usize] == unk,
            "unk token {unk:?} is not the vocab entry at id {unk_id} ({:?})",
            tokens[unk_id as usize]
        );
        ensure!(!prefix.is_empty(), "empty continuing-subword prefix");
        ensure!(
            max_input_chars_per_word >= 1,
            "max_input_chars_per_word must be positive"
        );
        ensure!(max_length >= 2, "max_length must hold [CLS] and [SEP]");
        let mut specials: Vec<(String, u32)> = Vec::new();
        for id in [pad, unk_id, cls, sep, mask] {
            let t = &tokens[id as usize];
            if !specials.iter().any(|(s, _)| s == t) {
                specials.push((t.clone(), id));
            }
        }
        // Longest first: the first hit at a position is the longest one there.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        Ok(Self {
            vocab,
            tokens,
            unk: unk_id,
            cls,
            sep,
            prefix: prefix.to_string(),
            max_input_chars_per_word,
            max_length,
            specials,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The vocab entry of an id.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// The id of a vocab entry.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Tokens of ids (an id outside the vocab is `"<?>"`).
    pub fn tokens(&self, ids: &[u32]) -> Vec<&str> {
        ids.iter()
            .map(|&i| self.token(i).unwrap_or("<?>"))
            .collect()
    }

    /// Longest `encode` output, `[CLS]` and `[SEP]` included.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// The added (special) tokens matched on the raw text, as (content, id).
    pub fn specials(&self) -> &[(String, u32)] {
        &self.specials
    }

    /// Split the raw text at the added tokens (leftmost-longest, case-sensitive).
    pub fn split_specials<'a>(&self, text: &'a str) -> Vec<Piece<'a>> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let (mut last, mut i) = (0, 0);
        while i < bytes.len() {
            let hit = self
                .specials
                .iter()
                .find(|(s, _)| bytes[i..].starts_with(s.as_bytes()));
            if let Some((s, id)) = hit {
                if last < i {
                    out.push(Piece::Text(&text[last..i]));
                }
                out.push(Piece::Special(*id));
                i += s.len();
                last = i;
            } else {
                // The next character boundary.
                i += 1;
                while i < bytes.len() && !text.is_char_boundary(i) {
                    i += 1;
                }
            }
        }
        if last < bytes.len() {
            out.push(Piece::Text(&text[last..]));
        }
        out
    }

    /// WordPiece of one pre-token, appended to `out`.
    pub fn word_ids(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_input_chars_per_word {
            out.push(self.unk);
            return;
        }
        let mark = out.len();
        let mut key = String::with_capacity(word.len() + self.prefix.len());
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let piece = &word[start..end];
                let id = if start > 0 {
                    key.clear();
                    key.push_str(&self.prefix);
                    key.push_str(piece);
                    self.vocab.get(key.as_str())
                } else {
                    self.vocab.get(piece)
                };
                if let Some(&id) = id {
                    found = Some(id);
                    break;
                }
                end -= piece.chars().next_back().map_or(1, char::len_utf8);
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(mark);
                    out.push(self.unk);
                    return;
                }
            }
        }
    }

    /// Token ids of the text, without the template and without truncation.
    pub fn encode_pieces(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in self.split_specials(text) {
            match piece {
                Piece::Special(id) => ids.push(id),
                Piece::Text(t) => {
                    let norm = normalize(t);
                    for w in pre_tokenize(&norm) {
                        self.word_ids(w, &mut ids);
                    }
                }
            }
        }
        ids
    }

    /// `encode(text, add_special_tokens = true)` with right truncation to
    /// `max_length`: `[CLS] pieces[..max_length − 2] [SEP]`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut pieces = self.encode_pieces(text);
        pieces.truncate(self.max_length - 2);
        let mut ids = Vec::with_capacity(pieces.len() + 2);
        ids.push(self.cls);
        ids.extend_from_slice(&pieces);
        ids.push(self.sep);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: SpecialIds = SpecialIds {
        pad: 0,
        unk: 1,
        cls: 2,
        sep: 3,
        mask: 4,
    };

    fn toy() -> WordPiece {
        let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nun\n##aff\n##able\na\nb\n!\n##b\nhello\n";
        WordPiece::new(vocab.as_bytes(), "[UNK]", "##", 9, &IDS, 6).unwrap()
    }

    #[test]
    fn normalizer_steps() {
        assert_eq!(normalize("Café crème naïve ÀÉÎ"), "cafe creme naive aei");
        assert_eq!(normalize("a\u{0}b\u{FFFD}c\u{200B}d"), "abcd");
        assert_eq!(normalize("a\tb\nc\r\u{A0}d"), "a b c  d");
        assert_eq!(normalize("野口 x"), " 野  口  x");
        assert_eq!(normalize("İΣΑΣ"), "iσασ");
        // Unicode 9.0 does not assign U+11938, so HF does not decompose it.
        assert_eq!(normalize("\u{11938}"), "\u{11938}");
        // Private use (Co) is dropped by clean_text.
        assert_eq!(normalize("x\u{E000}y"), "xy");
    }

    #[test]
    fn pre_tokenizer_splits() {
        assert_eq!(
            pre_tokenize("hey friend!   how are you?!?"),
            vec!["hey", "friend", "!", "how", "are", "you", "?", "!", "?"]
        );
        assert_eq!(pre_tokenize("$5 a+b"), vec!["$", "5", "a", "+", "b"]);
        assert_eq!(pre_tokenize("  "), Vec::<&str>::new());
        assert_eq!(pre_tokenize("«x»—y"), vec!["«", "x", "»", "—", "y"]);
    }

    #[test]
    fn wordpiece_greedy_and_unk() {
        let t = toy();
        assert_eq!(t.encode("unaffable"), vec![2, 5, 6, 7, 3]);
        assert_eq!(t.encode("unb hello"), vec![2, 5, 11, 12, 3]);
        // An unmatched rest makes the whole word [UNK].
        assert_eq!(t.encode("unx"), vec![2, 1, 3]);
        // Longer than max_input_chars_per_word (9 characters).
        assert_eq!(t.encode("unaffables"), vec![2, 1, 3]);
        assert_eq!(t.encode("ab!"), vec![2, 8, 11, 10, 3]);
        assert_eq!(t.encode(""), vec![2, 3]);
    }

    #[test]
    fn specials_and_truncation() {
        let t = toy();
        assert_eq!(t.encode("a[UNK]b"), vec![2, 8, 1, 9, 3]);
        assert_eq!(t.encode("[unk]"), vec![2, 1, 1, 1, 3]);
        assert_eq!(t.encode("[[PAD]]"), vec![2, 1, 0, 1, 3]);
        // max_length 6 keeps four pieces.
        assert_eq!(t.encode("a b a b a b"), vec![2, 8, 9, 8, 9, 3]);
    }

    #[test]
    fn duplicate_vocab_is_refused() {
        let vocab = b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\na \n";
        assert!(WordPiece::new(vocab, "[UNK]", "##", 100, &IDS, 512).is_err());
    }
}
