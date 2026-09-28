//! `lookup` records at run time (CMF_V2_SPEC §9.5.2; the lookup spec §3):
//! an explicit key → card table appended to a sealed genome, reached
//! through the request-level resonance router.
//!
//! * [`LookupTable`] — the four tensors `skill.{id}.lookup.*` of one
//!   record as the runtime reads them: sorted key hashes (binary search),
//!   entry per key, slot offsets, and the UTF-8 card blob left in the
//!   file's mapping (zero-copy).
//! * [`extract_key`] — O(len) key extraction from a user message: a Latin
//!   binomial in parentheses (the innermost group first), then a
//!   capitalised `Genus species` pair anywhere in the ORIGINAL message
//!   ([`capitalised_binomials`]), then every 1..=[`MAX_NGRAM`]-word n-gram
//!   of the `cmf-key-v2`-normalised message against the exact key index
//!   (the longest match wins, ties → the first occurrence), then the same
//!   n-grams STEMMED ([`stem_key`]: a light, deterministic Russian /
//!   English suffix stripper) against the table's [`StemIndex`]. A STRONG
//!   candidate ([`is_strong_key`]: ≥ 2 real words, no digits) wins over
//!   any weak one, in every path — the router's pick, `key_first`, the
//!   conversation memory; without one, the first weak candidate (a weak
//!   stem only when nothing matched exactly). The file stores only key
//!   hashes, so the stem index is built at open from the key texts the
//!   cards themselves spell ([`recover_key_texts`]): no format change.
//! * [`LookupTable::find_answer_turns`] — conversation memory: the key
//!   is taken from the most recent of up to [`MEMORY_TURNS`] user turns
//!   that holds one; the field and the language come from the LAST turn.
//! * [`select_field`] / [`pick_lang`] — the card field the question asks
//!   for (keyword rules) and the language of the answer (script).
//! * [`LookupMode`] — `answer` (the table answers, no generation),
//!   `context` (the card — or, when a field was selected, its first
//!   sentence plus that field — is prepended to the message and the
//!   backbone generates), `off` (the table is ignored); flag or
//!   `CMF_LOOKUP_MODE`.
//! * [`resolve_lookup`] — the one step every caller shares: a
//!   [`RouteDecision`] whose target is a lookup record becomes the lookup
//!   outcome; no key in the message, an empty card in every language, or
//!   mode `off` sends the request to the BACKBONE unchanged. A lookup
//!   record never has a lane of its own: the backbone pipeline — the same
//!   object F0 runs — serves it, so the logits of every non-hit request
//!   are bit-identical to F0's.
//! * [`LookupPolicy`] / [`resolve_lookup_gated`] — the record's routing
//!   policy (`LookupInfo.policy`): `router_and_key` (the default) looks a
//!   key up only in what the φ router sent; `key_first` lets a STRONG key
//!   of the message ([`is_strong_key`]) take a request the router sent to
//!   the backbone — only on a file whose router could pick a skill at all
//!   (no fail-closed state). A one-word key (`чай`, `мята`) still needs
//!   the router; a message without a strong key keeps the router's
//!   decision untouched. An unknown policy value reads as
//!   `router_and_key`.

use crate::router::{self, PromptFrame, RouteDecision, RouteOptions, RouteTarget};
use cortiq_core::knowledge::{
    LOOKUP_MAX_NGRAM, LookupInfo, lookup_leaf, lookup_policy, lookup_tensor_name, normalize_key,
    normalized_key_hash, read_u32_le, read_u64_le, skill_kind,
};
use cortiq_core::{CmfModel, SkillRecord};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

// ───────────────────────── mode ─────────────────────────

/// Environment default of [`LookupMode`] (`answer` | `context` | `off`).
pub const LOOKUP_MODE_ENV: &str = "CMF_LOOKUP_MODE";

/// What a request routed to a lookup record does with the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LookupMode {
    /// The table answers: the selected field's text (or the whole card);
    /// nothing is generated. The default.
    #[default]
    Answer,
    /// The card is prepended to the user message
    /// ([`context_prompt`]) and the BACKBONE generates.
    Context,
    /// The table is ignored: the backbone runs on the plain message.
    Off,
}

impl LookupMode {
    /// `answer` | `context` | `off` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "answer" => Ok(Self::Answer),
            "context" => Ok(Self::Context),
            "off" | "none" => Ok(Self::Off),
            other => Err(format!(
                "lookup mode '{other}': expected answer | context | off"
            )),
        }
    }

    /// [`LOOKUP_MODE_ENV`], `answer` when unset or empty.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(LOOKUP_MODE_ENV) {
            Ok(v) if !v.trim().is_empty() => {
                Self::parse(&v).map_err(|e| format!("{LOOKUP_MODE_ENV}: {e}"))
            }
            _ => Ok(Self::Answer),
        }
    }

    /// The flag when given, else the environment, else `answer`.
    pub fn resolve(flag: Option<&str>) -> Result<Self, String> {
        match flag {
            Some(f) => Self::parse(f).map_err(|e| format!("--lookup-mode: {e}")),
            None => Self::from_env(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Answer => "answer",
            Self::Context => "context",
            Self::Off => "off",
        }
    }
}

/// The header line of a `context`-mode prompt.
pub const CONTEXT_HEADER: &str = "Справочная карточка / Reference card:";

/// The user message the backbone generates from in `context` mode.
pub fn context_prompt(card: &str, user_text: &str) -> String {
    format!("{CONTEXT_HEADER}\n{card}\n\n{user_text}")
}

// ───────────────────────── routing policy ─────────────────────────

/// How a request reaches a lookup record (`LookupInfo.policy`, spec
/// §9.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LookupPolicy {
    /// `router_and_key` (the default, also when the field is absent): the
    /// φ router sends the request to the record, the table then looks the
    /// key up in it; a request the router sent to the backbone never
    /// reaches the table.
    #[default]
    RouterAndKey,
    /// `key_first`: a STRONG key in the message ([`is_strong_key`]) sends
    /// the request to the record even when the router picked the backbone
    /// (the backbone nearest, a novel input, the margin not beaten); a
    /// one-word key still needs the router's decision.
    KeyFirst,
}

impl LookupPolicy {
    /// `router_and_key` | `key_first`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            lookup_policy::ROUTER_AND_KEY => Ok(Self::RouterAndKey),
            lookup_policy::KEY_FIRST => Ok(Self::KeyFirst),
            other => Err(format!(
                "lookup policy '{other}': expected {}",
                lookup_policy::ALL.join(" | ")
            )),
        }
    }

    /// The policy a record declares ([`LookupInfo::policy_label`]). A
    /// value this reader does not know (a newer writer's policy, a hand
    /// edit) is `router_and_key` — the conservative reading, the one a
    /// reader from before the field existed applies — and never a reason
    /// to refuse the file (review KF-6; `CmfModel::open` warns once, the
    /// writers `lookup-build` / `lookup-policy` refuse such a value).
    pub fn of(info: &LookupInfo) -> Self {
        Self::parse(info.policy_label()).unwrap_or_default()
    }

    /// Does the record declare a policy this reader knows (absent = the
    /// default, known)?
    pub fn is_known(info: &LookupInfo) -> bool {
        Self::parse(info.policy_label()).is_ok()
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::RouterAndKey => lookup_policy::ROUTER_AND_KEY,
            Self::KeyFirst => lookup_policy::KEY_FIRST,
        }
    }
}

/// Who sent a request to a lookup record — `decided_by` in the route
/// summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecidedBy {
    /// The decision the lookup step was handed: the φ router's (or a
    /// pinned `--route <id>`).
    #[default]
    Router,
    /// The `key_first` policy: the router picked the backbone, a strong
    /// key of the message took the request.
    KeyFirst,
}

impl DecidedBy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::KeyFirst => "key_first",
        }
    }
}

// ───────────────────────── question → field / language ─────────────────────────

/// The card fields [`select_field`] can name, in rule order (the first
/// rule that matches wins): the specific ones before `uses`, which almost
/// every question mentions in passing (`safe to use`, `parts used`, `in
/// what form is it used`).
pub const FIELDS: &[&str] = &[
    "family",
    "parts",
    "compounds",
    "safety",
    "evidence",
    "preparations",
    "uses",
];

/// The field a question asks for, by keyword (lookup spec §3), in
/// [`FIELDS`] order:
/// `семейств|family → family`; `части|часть|частей|частям|частях|частью|
/// частями|part|parts → parts` (whole words: `частуха` is a plant,
/// `часто` an adverb); `веществ|соединен|состав|компонент|ингредиент|
/// compound|constituent|ingredient|chemical → compounds`; `противопоказ|
/// безопас|побочн|опасн|ядовит|токсич|contraindic|toxic|poison|safety|
/// safe|risk|risks|side effect → safety`; `доказ|исследован|evidence|
/// study|studies|research|clinical|trial|trials → evidence`; `форм|
/// препарат|дозиров|preparation|dosage|form|forms|dose|doses →
/// preparations`; `примен|use|uses|used|usage → uses`. Stems match a word
/// prefix, the short Latin words match whole words (so `because` is not
/// `use`, `information` is not `form`). `None` = no field named: the
/// whole card answers.
pub fn select_field(question: &str) -> Option<&'static str> {
    let norm = normalize_key(question);
    let words: Vec<&str> = norm.split(' ').filter(|w| !w.is_empty()).collect();
    let prefix = |p: &str| words.iter().any(|w| w.starts_with(p));
    let word = |x: &str| words.iter().any(|w| *w == x);
    let bigram = |a: &str, b: &str| words.windows(2).any(|w| w[0] == a && w[1].starts_with(b));
    if prefix("семейств") || word("family") || word("families") {
        return Some("family");
    }
    if ["части", "часть", "частей", "частям", "частях", "частью", "частями"]
        .iter()
        .any(|w| word(w))
        || word("part")
        || word("parts")
    {
        return Some("parts");
    }
    if prefix("веществ")
        || prefix("соединен")
        || prefix("состав")
        || prefix("компонент")
        || prefix("ингредиент")
        || prefix("compound")
        || prefix("constituent")
        || prefix("ingredient")
        || prefix("chemical")
    {
        return Some("compounds");
    }
    if prefix("противопоказ")
        || prefix("безопас")
        || prefix("побочн")
        || prefix("опасн")
        || prefix("ядовит")
        || prefix("токсич")
        || prefix("contraindic")
        || prefix("toxic")
        || prefix("poison")
        || word("safety")
        || word("safe")
        || word("risk")
        || word("risks")
        || bigram("side", "effect")
    {
        return Some("safety");
    }
    if prefix("доказ")
        || prefix("исследован")
        || word("evidence")
        || word("study")
        || word("studies")
        || word("research")
        || word("clinical")
        || word("trial")
        || word("trials")
    {
        return Some("evidence");
    }
    if prefix("форм")
        || prefix("препарат")
        || prefix("дозиров")
        || prefix("preparation")
        || prefix("dosage")
        || word("form")
        || word("forms")
        || word("dose")
        || word("doses")
    {
        return Some("preparations");
    }
    if prefix("примен") || word("use") || word("uses") || word("used") || word("usage") {
        return Some("uses");
    }
    None
}

/// Any Cyrillic letter in `s`.
pub fn has_cyrillic(s: &str) -> bool {
    s.chars().any(|c| ('\u{0400}'..='\u{052F}').contains(&c))
}

/// The slot language of the answer: `ru` for a Cyrillic question, else
/// `en`; a language the record lacks falls back to `en`, then to the
/// record's first language. Returns the index into `langs`.
pub fn pick_lang(question: &str, langs: &[String]) -> usize {
    let want = if has_cyrillic(question) { "ru" } else { "en" };
    langs
        .iter()
        .position(|l| l == want)
        .or_else(|| langs.iter().position(|l| l == "en"))
        .unwrap_or(0)
}

// ───────────────────────── key extraction ─────────────────────────

/// Longest n-gram (in words of the normalised message) tried as a key —
/// [`LOOKUP_MAX_NGRAM`] of the core, the number the builder drops longer
/// keys by.
pub const MAX_NGRAM: usize = LOOKUP_MAX_NGRAM;

/// Conversation memory of `serve` ([`LookupTable::find_answer_turns`]):
/// the key is searched in the last user message and up to this many
/// user turns in total, most recent first.
pub const MEMORY_TURNS: usize = 6;

/// Where a key hit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// A parenthesised Latin group, e.g. `(Abies balsamea)`.
    Parenthesised,
    /// A capitalised `Genus species` pair anywhere in the original
    /// message ([`capitalised_binomials`]).
    Binomial,
    /// An n-gram of the normalised message.
    Ngram,
}

impl KeySource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Parenthesised => "parenthesised",
            Self::Binomial => "binomial",
            Self::Ngram => "n-gram",
        }
    }
}

/// Which index answered: the exact `cmf-key-v2` hashes (primary), or the
/// stem index ([`StemIndex`], consulted only when the primary found
/// nothing). Reported as `lookup_match` in the route summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchVia {
    Exact,
    Stem,
}

impl MatchVia {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Stem => "stem",
        }
    }
}

/// A key found in a user message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyHit {
    /// The normalised text of the message that matched (`cmf-key-v2`): the
    /// key itself for an exact hit, the inflected n-gram for a stem hit.
    pub key: String,
    /// The hash that hit: of `key` for an exact match, of `stem` for a
    /// stem match.
    pub hash: u64,
    /// The entry the key names.
    pub entry: u32,
    /// Words in the key.
    pub words: usize,
    pub source: KeySource,
    pub via: MatchVia,
    /// The stemmed n-gram that hit the stem index (stem matches only).
    pub stem: Option<String>,
    /// How many user turns back the key was found: 0 = the message the
    /// request was routed on ([`LookupTable::find_answer_turns`]).
    pub turn: usize,
    /// A STRONG key ([`is_strong_key`]): the `key_first` policy acts on
    /// it, and every search prefers it over a weak one.
    pub strong: bool,
}

/// The candidate groups of every `( … )` of `s`, in order: each group
/// runs from a `(` to the first `)` after it; when the group holds
/// another `(` (a nested `(name (Binomial))`, `(syn. …)`) the text after
/// its LAST `(` — the innermost group — comes first, then the whole
/// group.
fn parenthesised(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(open) = rest.find('(') {
        let after = &rest[open + 1..];
        let Some(close) = after.find(')') else {
            break;
        };
        let group = &after[..close];
        if let Some(inner) = group.rfind('(') {
            out.push(&group[inner + 1..]);
        }
        out.push(group);
        rest = &after[close + 1..];
    }
    out
}

/// A letter of the Latin script: ASCII, or a Latin letter with a
/// diacritic the key normalisation cannot fold (`æ`, `ø`, `ß`, `ł`, …).
fn is_latin_char(c: char) -> bool {
    c.is_ascii_alphabetic()
        || (c.is_alphabetic() && matches!(c, '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}'))
}

/// A word of the Latin script ([`is_latin_char`] throughout). A binomial
/// in parentheses is Latin; a Cyrillic or numeric group is not tried as
/// one.
fn is_latin_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(is_latin_char)
}

/// A lowercase Latin word of at least `min` letters.
fn lower_latin_word(t: &str, min: usize) -> bool {
    t.chars().count() >= min && t.chars().all(|c| c.is_lowercase() && is_latin_char(c))
}

/// Every `Genus species` pair of the ORIGINAL (case-kept) message, in
/// order: a capitalised Latin word of ≥ 3 letters followed by a lowercase
/// Latin word of ≥ 3 letters, optionally hyphenated (`nux-vomica`: ≥ 3
/// and ≥ 2 letters around the hyphen); punctuation around the words is
/// ignored (`(Abies balsamea)?`). The pairs are tried as EXACT keys only,
/// before the n-grams; a genus alone is never tried (ambiguous). A
/// capitalised English pair (`What plant`) is a candidate too — harmless,
/// it is a key or it is not.
pub fn capitalised_binomials(s: &str) -> Vec<String> {
    let genus = |t: &str| {
        let mut cs = t.chars();
        matches!(cs.next(), Some(c) if c.is_uppercase() && is_latin_char(c))
            && lower_latin_word(cs.as_str(), 2)
    };
    let epithet = |t: &str| match t.split_once('-') {
        None => lower_latin_word(t, 3),
        Some((a, b)) => lower_latin_word(a, 3) && lower_latin_word(b, 2),
    };
    let toks: Vec<&str> = s
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !is_latin_char(c)))
        .collect();
    toks.windows(2)
        .filter(|w| genus(w[0]) && epithet(w[1]))
        .map(|w| format!("{} {}", w[0], w[1]))
        .collect()
}

// ───────────────────────── stemming ─────────────────────────

/// Russian inflectional endings [`stem_word`] strips, longest first
/// (adjective and noun endings of every case and number; `й` and `ь`
/// so that `зверобой` / `зверобоя` and `полынь` / `полыни` meet).
pub const RU_SUFFIXES: &[&str] = &[
    "ого", "его", "ому", "ему", "ыми", "ими", "ами", "ями", // 3
    "ая", "яя", "ой", "ей", "ий", "ый", "ое", "ее", "ые", "ых", "их", "ую", "юю", "ою", "ею",
    "ам", "ям", "ах", "ях", "ом", "ем", "ым", "им", "ов", "ев", "ии", "ия", "ие", "ье", "ья",
    "ью", // 2
    "а", "я", "ы", "и", "у", "ю", "о", "е", "ь", "й", // 1
];
/// A Russian stem keeps at least this many letters (`чай`, `вид`, `дуб`
/// are never shortened).
pub const RU_STEM_MIN: usize = 3;
/// An English stem keeps at least this many letters.
pub const EN_STEM_MIN: usize = 4;
/// A Latin-script word shorter than this is never stemmed (`sage`,
/// `uses`, `wort`, `Pinus` stay as they are).
pub const LATIN_STEM_MIN_WORD: usize = 5;
/// A ONE-word stem candidate shorter than this (in letters) is not looked
/// up: `виды` → `вид`, `чая` → `чая` never match by stem. Exact one-word
/// matches are unaffected.
pub const STEM_UNIGRAM_MIN: usize = 5;

/// A light, deterministic stem of one normalised word — the same function
/// on the keys (at open) and on the message (per query), so an inflected
/// form meets its nominative key. Cyrillic: the longest ending of
/// [`RU_SUFFIXES`] is stripped, repeatedly, while ≥ [`RU_STEM_MIN`]
/// letters remain (`тойона` → `тойон`, `магнолии` → `магнол`,
/// `лекарственной` → `лекарственн`, `зверобоя` → `зверобо` → `звероб`).
/// Latin script, ≥ [`LATIN_STEM_MIN_WORD`] letters: `ies` → `i`, `es`
/// after `ss`/`x`/`z`/`ch`/`sh`, or a final `s` (not after `ss`/`us`/`is`)
/// is dropped when ≥ [`EN_STEM_MIN`] letters remain; otherwise a final
/// `y` → `i` so that `berry` meets `berries`. Latin binomials mostly stay
/// as they are (`pinus`, `officinalis`, `balsamea`). A word with a digit
/// or another script is returned unchanged.
pub fn stem_word(w: &str) -> String {
    if w.is_empty() || !w.chars().all(char::is_alphabetic) {
        return w.to_string();
    }
    if has_cyrillic(w) {
        stem_cyrillic(w)
    } else {
        stem_latin(w)
    }
}

fn stem_cyrillic(w: &str) -> String {
    let mut s = w.to_string();
    let mut n = s.chars().count();
    'strip: loop {
        for suf in RU_SUFFIXES {
            let k = suf.chars().count();
            if n >= k + RU_STEM_MIN && s.ends_with(suf) {
                s.truncate(s.len() - suf.len());
                n -= k;
                continue 'strip;
            }
        }
        return s;
    }
}

fn stem_latin(w: &str) -> String {
    let n = w.chars().count();
    if n < LATIN_STEM_MIN_WORD {
        return w.to_string();
    }
    let mut s = w.to_string();
    if s.ends_with("ies") && n - 3 >= EN_STEM_MIN {
        s.truncate(s.len() - 3);
        s.push('i');
    } else if ["sses", "xes", "zes", "ches", "shes"]
        .iter()
        .any(|e| s.ends_with(e))
        && n - 2 >= EN_STEM_MIN
    {
        s.truncate(s.len() - 2);
    } else if s.ends_with('s')
        && !["ss", "us", "is"].iter().any(|e| s.ends_with(e))
        && n - 1 >= EN_STEM_MIN
    {
        s.pop();
    } else if s.ends_with('y') {
        s.pop();
        s.push('i');
    }
    s
}

/// [`stem_word`] of every word of a normalised key or message, joined by
/// single spaces (the form both sides of the stem index hash).
pub fn stem_key(norm: &str) -> String {
    let mut out = String::with_capacity(norm.len());
    for w in norm.split(' ').filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&stem_word(w));
    }
    out
}

/// Byte spans of the words of a normalised (single-spaced) string.
fn word_spans(norm: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in norm.char_indices() {
        if c == ' ' {
            if let Some(s) = start.take() {
                out.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, norm.len()));
    }
    out
}

/// The second, sorted hash index of a table: hashes of the STEMMED key
/// texts ([`stem_key`]), each mapped to its entry. A stem two keys of
/// DIFFERENT entries share is ambiguous and is left out (`ambiguous`);
/// the same stem from several keys of one entry (`ромашка аптечная`,
/// `ромашки аптечной`) is one row. Consulted only when the exact index
/// finds nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StemIndex {
    hashes: Vec<u64>,
    entry_of: Vec<u32>,
    /// Stems dropped because they named more than one entry.
    pub ambiguous: usize,
    /// Keys the index was built from.
    pub keys_in: usize,
}

impl StemIndex {
    /// From `(normalised key, entry)` pairs.
    pub fn from_keys<'a>(keys: impl IntoIterator<Item = (&'a str, u32)>) -> Self {
        let mut pairs: Vec<(u64, u32)> = keys
            .into_iter()
            .map(|(k, e)| (normalized_key_hash(&stem_key(k)), e))
            .collect();
        let keys_in = pairs.len();
        pairs.sort_unstable();
        pairs.dedup();
        let mut hashes = Vec::with_capacity(pairs.len());
        let mut entry_of = Vec::with_capacity(pairs.len());
        let mut ambiguous = 0;
        let mut i = 0;
        while i < pairs.len() {
            let mut j = i;
            while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                j += 1;
            }
            if j - i == 1 {
                hashes.push(pairs[i].0);
                entry_of.push(pairs[i].1);
            } else {
                ambiguous += 1;
            }
            i = j;
        }
        Self {
            hashes,
            entry_of,
            ambiguous,
            keys_in,
        }
    }

    /// The entry of stem hash `h` (binary search).
    pub fn find(&self, h: u64) -> Option<u32> {
        self.hashes
            .binary_search(&h)
            .ok()
            .map(|i| self.entry_of[i])
    }

    /// Stems in the index.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

/// The key texts a table's cards spell — the file stores only key hashes,
/// so this is where the stem index gets its words from: every
/// 1..=[`MAX_NGRAM`]-word n-gram of every slot's text (the raw JSON of the
/// card and its fields) whose hash the exact index knows, once per key,
/// with the entry the INDEX maps it to (a card that mentions another
/// entry's plant recovers that plant's key correctly). A key no card
/// spells — a synonym only the corpus knew — is not recovered and has no
/// stem; the exact index still serves it. O(blob) at open.
pub fn recover_key_texts(
    hashes: &[u64],
    entry_of: &[u32],
    blob: &[u8],
    offsets: &[u64],
) -> Vec<(String, u32)> {
    let find = |h: u64| hashes.binary_search(&h).ok().map(|i| entry_of[i]);
    let mut seen: BTreeMap<u64, (String, u32)> = BTreeMap::new();
    'slots: for w in offsets.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        if a > b || b > blob.len() {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&blob[a..b]) else {
            continue;
        };
        let norm = normalize_key(text);
        let spans = word_spans(&norm);
        for n in 1..=MAX_NGRAM.min(spans.len()) {
            for start in 0..=spans.len() - n {
                let g = &norm[spans[start].0..spans[start + n - 1].1];
                let h = normalized_key_hash(g);
                if seen.contains_key(&h) {
                    continue;
                }
                if let Some(e) = find(h) {
                    seen.insert(h, (g.to_string(), e));
                    if seen.len() == hashes.len() {
                        break 'slots;
                    }
                }
            }
        }
    }
    seen.into_values().collect()
}

/// [`extract_key_with`] against the exact index only (no stem index).
pub fn extract_key(message: &str, find: &dyn Fn(u64) -> Option<u32>) -> Option<KeyHit> {
    extract_key_with(message, find, None)
}

/// The key of `message`, O(len): the STRONGEST key when the message holds
/// one ([`extract_strong_key_with`]), else the first key of any strength.
/// One rule for every path — the φ router's pick of a table, the
/// `key_first` policy and the conversation memory resolve one message to
/// one entry (review KF-3: `чай из ромашки аптечной` is chamomile whoever
/// decided, not tea on the router's path and chamomile on key_first's).
///
/// The candidates, in order: a Latin group in parentheses (the binomial of
/// a plant name — the innermost group of a nested one, then the whole
/// group — 1..=[`MAX_NGRAM`] Latin words); a capitalised `Genus species`
/// pair anywhere in the original message ([`capitalised_binomials`]);
/// every n-gram of the `cmf-key-v2`-normalised message from the longest
/// down against `find` (the exact index; the first hit at the longest
/// length wins, ties → the first occurrence); then the STEMMED n-grams
/// ([`stem_key`]) against `find_stem` (the table's [`StemIndex`]; a
/// one-word stem must have ≥ [`STEM_UNIGRAM_MIN`] letters). The first
/// STRONG candidate in that order wins ([`is_strong_key_text`]); without
/// one, the first candidate of any strength — a weak stem candidate only
/// when nothing exact was found. `None` = no key: the backbone runs
/// unchanged.
pub fn extract_key_with(
    message: &str,
    find: &dyn Fn(u64) -> Option<u32>,
    find_stem: Option<&dyn Fn(u64) -> Option<u32>>,
) -> Option<KeyHit> {
    extract(message, find, find_stem, false)
}

/// A strong key has at least this many words of ≥
/// [`STRONG_WORD_MIN_LETTERS`] letters ([`is_strong_key_text`]).
pub const STRONG_KEY_MIN_WORDS: usize = 2;

/// Letters a word needs to count towards [`STRONG_KEY_MIN_WORDS`] (`st`,
/// `s`, `b`, `pb` never do).
pub const STRONG_WORD_MIN_LETTERS: usize = 3;

/// Is the normalised key text `norm` STRONG — a name the `key_first`
/// policy may act on without the router (review KF-5)? No word with a
/// digit (`sts 135`, `pti 2`, `a 41988`, `5f pb 22`), and at least
/// [`STRONG_KEY_MIN_WORDS`] purely alphabetic words of ≥
/// [`STRONG_WORD_MIN_LETTERS`] letters (`ромашки аптечной`, `pot
/// marigold`, `st john s wort`; not `thc b`, not `чай`). A general phrase
/// that happens to be a plant's common name (`scrambled eggs`, `gas
/// plant`, `live forever`) passes this rule — only the builder's general
/// probe and a stop list remove such a key (`lookup-build
/// --general-prompts`, `--drop-keys`).
pub fn is_strong_key_text(norm: &str) -> bool {
    let mut words = 0usize;
    for w in norm.split(' ').filter(|w| !w.is_empty()) {
        if w.chars().any(char::is_numeric) {
            return false;
        }
        if w.chars().count() >= STRONG_WORD_MIN_LETTERS && w.chars().all(char::is_alphabetic) {
            words += 1;
        }
    }
    words >= STRONG_KEY_MIN_WORDS
}

/// Is `hit` a STRONG key — one the `key_first` policy acts on without
/// the router, and one every search prefers over a weak key of the same
/// message? Its text passes [`is_strong_key_text`], whatever its source:
/// an exact or stem n-gram of ≥ 2 real words (`ромашки аптечной`, `pot
/// marigold`), a parenthesised Latin group, a capitalised `Genus species`
/// pair. The source does not make a key strong: every exact key a
/// capitalised pair finds is the same text as an n-gram of the message,
/// so a pair rule stricter than the n-gram rule would protect nothing
/// (`Common box` in `Why is Common box cutter so popular?` is the n-gram
/// key `common box` too — only a stop list removes it) and would only
/// reorder candidates (a binomial the cards spell over the common name
/// typed first: `Spring vetchling (Lathyrus vernus (L.) Bernh.)` would
/// then answer from a stub entry of the corpus). A one-word key — `чай`,
/// `мята`, `календулы`, a lone `(Calendula)` — never is strong: such
/// words occur in general chat, and only the router's decision may send
/// them to a table.
pub fn is_strong_key(hit: &KeyHit) -> bool {
    hit.strong
}

/// The STRONGEST key of `message` for the `key_first` policy: the same
/// search as [`extract_key_with`] (the parenthesised group, the
/// capitalised binomial, the exact n-grams, then the stem n-grams) over
/// strong candidates only ([`is_strong_key_text`]) — so a one-word exact
/// key never hides a two-word stem key of the same message (`чай из
/// ромашки аптечной`). `None` = no strong key; the result always passes
/// [`is_strong_key`].
pub fn extract_strong_key_with(
    message: &str,
    find: &dyn Fn(u64) -> Option<u32>,
    find_stem: Option<&dyn Fn(u64) -> Option<u32>>,
) -> Option<KeyHit> {
    extract(message, find, find_stem, true)
}

/// A strong candidate returns at once; a weak one is kept as the fallback
/// (the first in rule order) unless only strong keys are wanted.
fn offer(fallback: &mut Option<KeyHit>, hit: KeyHit, strong_only: bool) -> Option<KeyHit> {
    if hit.strong {
        return Some(hit);
    }
    if !strong_only && fallback.is_none() {
        *fallback = Some(hit);
    }
    None
}

/// The one extraction pass behind [`extract_key_with`] (`strong_only`
/// false: the first strong candidate, else the first of any strength) and
/// [`extract_strong_key_with`] (`strong_only` true). One scan of each
/// stage: a candidate that could no longer change the result (weak, with
/// a fallback already kept or not wanted) is not hashed.
fn extract(
    message: &str,
    find: &dyn Fn(u64) -> Option<u32>,
    find_stem: Option<&dyn Fn(u64) -> Option<u32>>,
    strong_only: bool,
) -> Option<KeyHit> {
    let mut fallback: Option<KeyHit> = None;
    let exact_hit = |key: String, hash: u64, entry: u32, words: usize, source, strong| KeyHit {
        key,
        hash,
        entry,
        words,
        source,
        via: MatchVia::Exact,
        stem: None,
        turn: 0,
        strong,
    };
    for group in parenthesised(message) {
        let norm = normalize_key(group);
        let words: Vec<&str> = norm.split(' ').filter(|w| !w.is_empty()).collect();
        if words.is_empty() || words.len() > MAX_NGRAM || !words.iter().all(|w| is_latin_word(w)) {
            continue;
        }
        let n_words = words.len();
        let strong = is_strong_key_text(&norm);
        if !strong && (strong_only || fallback.is_some()) {
            continue;
        }
        let hash = normalized_key_hash(&norm);
        if let Some(entry) = find(hash) {
            let hit = exact_hit(norm, hash, entry, n_words, KeySource::Parenthesised, strong);
            if let Some(h) = offer(&mut fallback, hit, strong_only) {
                return Some(h);
            }
        }
    }
    for pair in capitalised_binomials(message) {
        let norm = normalize_key(&pair);
        let n_words = norm.split(' ').filter(|w| !w.is_empty()).count();
        if n_words == 0 || n_words > MAX_NGRAM {
            continue;
        }
        let strong = is_strong_key_text(&norm);
        if !strong && (strong_only || fallback.is_some()) {
            continue;
        }
        let hash = normalized_key_hash(&norm);
        if let Some(entry) = find(hash) {
            let hit = exact_hit(norm, hash, entry, n_words, KeySource::Binomial, strong);
            if let Some(h) = offer(&mut fallback, hit, strong_only) {
                return Some(h);
            }
        }
    }
    let norm = normalize_key(message);
    let spans = word_spans(&norm);
    if spans.is_empty() {
        return fallback;
    }
    let text = |start: usize, n: usize| &norm[spans[start].0..spans[start + n - 1].1];
    // Exact n-grams, the longest first, ties → the first occurrence.
    for n in (1..=MAX_NGRAM.min(spans.len())).rev() {
        for start in 0..=spans.len() - n {
            let g = text(start, n);
            let strong = n >= STRONG_KEY_MIN_WORDS && is_strong_key_text(g);
            if !strong && (strong_only || fallback.is_some()) {
                continue;
            }
            let hash = normalized_key_hash(g);
            if let Some(entry) = find(hash) {
                let hit = exact_hit(g.to_string(), hash, entry, n, KeySource::Ngram, strong);
                if let Some(h) = offer(&mut fallback, hit, strong_only) {
                    return Some(h);
                }
            }
        }
    }
    let Some(find_stem) = find_stem else {
        return fallback;
    };
    let stemmed = stem_key(&norm);
    let sspans = word_spans(&stemmed);
    // A stem never empties a word (the minimum-letters floors), so the
    // words of the two strings are in one-to-one correspondence.
    debug_assert_eq!(sspans.len(), spans.len());
    if sspans.len() != spans.len() {
        return fallback;
    }
    // Stem n-grams: the strong ones always (a two-word stem key beats a
    // one-word exact key), a weak one only when nothing matched at all.
    for n in (1..=MAX_NGRAM.min(sspans.len())).rev() {
        for start in 0..=sspans.len() - n {
            // The strength of a stem candidate is the message's own words.
            let strong = n >= STRONG_KEY_MIN_WORDS && is_strong_key_text(text(start, n));
            if !strong && (strong_only || fallback.is_some()) {
                continue;
            }
            let g = &stemmed[sspans[start].0..sspans[start + n - 1].1];
            if n == 1 && g.chars().count() < STEM_UNIGRAM_MIN {
                continue;
            }
            let hash = normalized_key_hash(g);
            if let Some(entry) = find_stem(hash) {
                let hit = KeyHit {
                    key: text(start, n).to_string(),
                    hash,
                    entry,
                    words: n,
                    source: KeySource::Ngram,
                    via: MatchVia::Stem,
                    stem: Some(g.to_string()),
                    turn: 0,
                    strong,
                };
                if let Some(h) = offer(&mut fallback, hit, strong_only) {
                    return Some(h);
                }
            }
        }
    }
    fallback
}

// ───────────────────────── cards ─────────────────────────

/// One slot of the text blob: `{"card": "...", "fields": {name: text}}`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Card {
    pub card: String,
    pub fields: BTreeMap<String, String>,
}

impl Card {
    /// The slot's JSON object (the file validation guarantees an object;
    /// a missing `card` is empty, a non-string field value is rendered as
    /// JSON).
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("card is not JSON: {e}"))?;
        let obj = v.as_object().ok_or("card is not a JSON object")?;
        let card = obj
            .get("card")
            .map(|c| match c {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default();
        let mut fields = BTreeMap::new();
        if let Some(serde_json::Value::Object(f)) = obj.get("fields") {
            for (k, v) in f {
                let text = match v {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Null => continue,
                    other => other.to_string(),
                };
                fields.insert(k.clone(), text);
            }
        }
        Ok(Self { card, fields })
    }

    /// A slot the builder wrote for a language the entry does not carry
    /// (`{"card": "", "fields": {}}`), or a card with nothing in it.
    pub fn is_empty(&self) -> bool {
        self.card.trim().is_empty() && self.fields.values().all(|t| t.trim().is_empty())
    }

    /// The first sentence of the card (the name and its one-line
    /// identity): up to the first `.`, `!` or `?` followed by whitespace
    /// or the end, skipping a terminator that would leave fewer than
    /// [`FIRST_SENTENCE_MIN`] chars (`Hypericum perforatum L.`); without
    /// one, the card up to [`FIRST_SENTENCE_MAX`] chars at a word
    /// boundary.
    pub fn first_sentence(&self) -> &str {
        first_sentence(&self.card)
    }
}

/// [`Card::first_sentence`]: a terminator this early is an abbreviation.
pub const FIRST_SENTENCE_MIN: usize = 12;
/// [`Card::first_sentence`]: the cap when the card has no terminator.
pub const FIRST_SENTENCE_MAX: usize = 240;

fn first_sentence(card: &str) -> &str {
    let card = card.trim();
    let bytes = card.as_bytes();
    let mut chars_seen = 0usize;
    let mut word_start = 0usize;
    for (i, c) in card.char_indices() {
        chars_seen += 1;
        if c.is_whitespace() {
            word_start = i + c.len_utf8();
            continue;
        }
        if matches!(c, '.' | '!' | '?') {
            let next = bytes.get(i + 1).copied();
            let ends = next.is_none_or(|b| b.is_ascii_whitespace());
            // `L.`, `Mill.`-style author abbreviations of a binomial: a
            // one-letter word before the period is not a sentence end.
            let word = &card[word_start..i];
            let abbrev = word.chars().count() == 1 && word.chars().all(char::is_alphabetic);
            if ends && !abbrev && chars_seen >= FIRST_SENTENCE_MIN {
                return &card[..i + 1];
            }
        }
    }
    if card.chars().count() <= FIRST_SENTENCE_MAX {
        return card;
    }
    let cut = card
        .char_indices()
        .nth(FIRST_SENTENCE_MAX)
        .map_or(card.len(), |(i, _)| i);
    let head = &card[..cut];
    head.rfind(char::is_whitespace)
        .map_or(head, |w| &head[..w])
        .trim_end()
}

/// The human label of a card field in the slot language (`context` mode
/// writes `{label}: {text}`); an unknown field or language keeps the
/// field name.
pub fn field_label(field: &str, lang: &str) -> String {
    let ru = match field {
        "family" => "Семейство",
        "parts" => "Части",
        "compounds" => "Действующие вещества",
        "uses" => "Применение",
        "preparations" => "Формы и препараты",
        "safety" => "Безопасность",
        "evidence" => "Доказательства",
        _ => "",
    };
    let en = match field {
        "family" => "Family",
        "parts" => "Parts",
        "compounds" => "Compounds",
        "uses" => "Uses",
        "preparations" => "Preparations",
        "safety" => "Safety",
        "evidence" => "Evidence",
        _ => "",
    };
    match (lang, ru, en) {
        ("ru", r, _) if !r.is_empty() => r.to_string(),
        ("en", _, e) if !e.is_empty() => e.to_string(),
        _ => {
            let mut c = field.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        }
    }
}

/// The text `context` mode prepends for a card: the whole card when no
/// field was selected; otherwise its first sentence and the field as
/// `{label}: {text}` — a short excerpt that keeps the fact next to the
/// question. A bounded-attention genome (swa_sink, window 128) would
/// otherwise see the family line of a 1 000-char card hundreds of tokens
/// before the question, beyond its window.
pub fn context_excerpt(card: &Card, field: Option<&str>, lang: &str) -> String {
    match field.and_then(|f| card.fields.get(f).map(|t| (f, t))) {
        Some((f, text)) => {
            let head = card.first_sentence();
            let label = field_label(f, lang);
            if head.is_empty() {
                format!("{label}: {}", text.trim())
            } else {
                format!("{head}\n{label}: {}", text.trim())
            }
        }
        None => card.card.clone(),
    }
}

// ───────────────────────── the table ─────────────────────────

/// One mounted lookup record.
pub struct LookupTable {
    pub id: String,
    pub info: LookupInfo,
    /// `keys.hash`, sorted ascending, unique.
    hashes: Vec<u64>,
    /// `keys.entry`, parallel to `hashes`.
    entry_of: Vec<u32>,
    /// `entries.off`: slot `s` is `blob[off[s]..off[s+1]]`.
    offsets: Vec<u64>,
    /// The stem index over the key texts recovered from the cards
    /// ([`recover_key_texts`]); consulted when `hashes` finds nothing.
    stems: StemIndex,
    /// Key texts recovered from the cards (≤ `keys`).
    recovered: usize,
    model: Arc<CmfModel>,
    text_name: String,
}

impl LookupTable {
    /// The record `id` when it is a lookup record.
    pub fn record<'a>(model: &'a CmfModel, id: &str) -> Option<&'a SkillRecord> {
        model
            .header
            .skills
            .iter()
            .find(|s| s.id == id && s.kind.as_deref() == Some(skill_kind::LOOKUP))
    }

    pub fn is_lookup(model: &CmfModel, id: &str) -> bool {
        Self::record(model, id).is_some()
    }

    /// Ids of every lookup record of the file, in header order.
    pub fn lookup_ids(model: &CmfModel) -> Vec<String> {
        model
            .header
            .skills
            .iter()
            .filter(|s| s.kind.as_deref() == Some(skill_kind::LOOKUP))
            .map(|s| s.id.clone())
            .collect()
    }

    /// Read the record's four tensors (the file's `open()` validated
    /// their values; only the shapes are re-checked here). The blob stays
    /// in the mapping.
    pub fn open(model: &Arc<CmfModel>, id: &str) -> Result<Self, String> {
        let rec = Self::record(model, id).ok_or_else(|| {
            format!(
                "skill '{id}' is not a lookup record (header.skills: {:?})",
                model
                    .header
                    .skills
                    .iter()
                    .map(|s| format!("{}={}", s.id, s.kind.as_deref().unwrap_or("v1")))
                    .collect::<Vec<_>>()
            )
        })?;
        let info = rec
            .lookup
            .clone()
            .ok_or_else(|| format!("skill '{id}': lookup record without `lookup`"))?;
        fn bytes<'m>(model: &'m CmfModel, id: &str, leaf: &str) -> Result<&'m [u8], String> {
            let name = lookup_tensor_name(id, leaf);
            model
                .tensor_bytes(&name)
                .map_err(|e| format!("skill '{id}': tensor '{name}': {e}"))
        }
        let hashes = read_u64_le(bytes(model, id, lookup_leaf::KEYS_HASH)?);
        let entry_of = read_u32_le(bytes(model, id, lookup_leaf::KEYS_ENTRY)?);
        let offsets = read_u64_le(bytes(model, id, lookup_leaf::ENTRIES_OFF)?);
        let text_name = lookup_tensor_name(id, lookup_leaf::TEXT);
        let blob = bytes(model, id, lookup_leaf::TEXT)?;
        let blob_len = blob.len();
        if hashes.len() != info.keys || entry_of.len() != info.keys {
            return Err(format!(
                "skill '{id}': {} hashes / {} entries for lookup.keys {}",
                hashes.len(),
                entry_of.len(),
                info.keys
            ));
        }
        let slots = info
            .slots()
            .ok_or_else(|| format!("skill '{id}': entries × langs overflows"))?;
        if offsets.len() != slots + 1 {
            return Err(format!(
                "skill '{id}': {} offsets for {slots} slots (+1)",
                offsets.len()
            ));
        }
        if offsets.last().is_some_and(|&l| l as usize > blob_len) {
            return Err(format!(
                "skill '{id}': offsets end beyond the text blob ({blob_len} bytes)"
            ));
        }
        // The second index (no format change): the key texts the cards
        // spell, stemmed. Built once per open, O(blob).
        let recovered_keys = recover_key_texts(&hashes, &entry_of, blob, &offsets);
        let stems = StemIndex::from_keys(recovered_keys.iter().map(|(k, e)| (k.as_str(), *e)));
        Ok(Self {
            id: id.to_string(),
            info,
            hashes,
            entry_of,
            offsets,
            stems,
            recovered: recovered_keys.len(),
            model: model.clone(),
            text_name,
        })
    }

    pub fn keys(&self) -> usize {
        self.hashes.len()
    }

    /// Key texts recovered from the cards at open (the stem index is
    /// built from these; a key no card spells has no stem).
    pub fn recovered_keys(&self) -> usize {
        self.recovered
    }

    /// The stem index (stems, ambiguous stems dropped).
    pub fn stems(&self) -> &StemIndex {
        &self.stems
    }

    pub fn entries(&self) -> usize {
        self.info.entries
    }

    pub fn langs(&self) -> &[String] {
        &self.info.langs
    }

    pub fn fields(&self) -> &[String] {
        &self.info.fields
    }

    /// The UTF-8 card blob (zero-copy from the file's mapping).
    pub fn blob(&self) -> &[u8] {
        self.model
            .tensor_bytes(&self.text_name)
            .expect("lookup text tensor was read at open")
    }

    /// The entry of key hash `h` (binary search over the sorted hashes).
    pub fn find_hash(&self, h: u64) -> Option<u32> {
        self.hashes
            .binary_search(&h)
            .ok()
            .map(|i| self.entry_of[i])
    }

    /// The entry of stem hash `h` ([`StemIndex::find`]).
    pub fn find_stem_hash(&self, h: u64) -> Option<u32> {
        self.stems.find(h)
    }

    /// The raw slot text of `entry` in language index `lang`.
    pub fn slot_text(&self, entry: u32, lang: usize) -> Option<&str> {
        let s = self.info.slot(entry as usize, lang)?;
        let (a, b) = (self.offsets[s] as usize, self.offsets[s + 1] as usize);
        std::str::from_utf8(&self.blob()[a..b]).ok()
    }

    /// The parsed card of `entry` in language index `lang`.
    pub fn card(&self, entry: u32, lang: usize) -> Result<Card, String> {
        let text = self.slot_text(entry, lang).ok_or_else(|| {
            format!(
                "skill '{}': no slot for entry {entry}, lang {lang} (entries {}, langs {:?})",
                self.id, self.info.entries, self.info.langs
            )
        })?;
        Card::parse(text).map_err(|e| format!("skill '{}': entry {entry}: {e}", self.id))
    }

    /// [`extract_key_with`] against this table (the exact index, then the
    /// stem index): the strongest key of the message, else the first weak
    /// one — the rule every path shares (the router's pick, `key_first`,
    /// the conversation memory).
    pub fn find_key(&self, message: &str) -> Option<KeyHit> {
        extract_key_with(message, &|h| self.find_hash(h), Some(&|h| self.find_stem_hash(h)))
    }

    /// [`extract_strong_key_with`] against this table: the key the
    /// `key_first` policy acts on ([`is_strong_key`]).
    pub fn find_strong_key(&self, message: &str) -> Option<KeyHit> {
        extract_strong_key_with(message, &|h| self.find_hash(h), Some(&|h| self.find_stem_hash(h)))
    }

    /// The record's routing policy ([`LookupPolicy::of`]: a value this
    /// reader does not know is `router_and_key`).
    pub fn policy(&self) -> LookupPolicy {
        LookupPolicy::of(&self.info)
    }

    /// The table's answer to `message`: the key, the language by script
    /// (an empty slot — the entry does not carry that language — falls
    /// back to the first language whose card has text), the field by
    /// keywords — its text when the card has it, else the whole card.
    /// `None` = no key in the message, or a key whose card is empty in
    /// every language ([`Self::find_answer`] tells the two apart).
    pub fn answer(&self, message: &str) -> Result<Option<LookupAnswer>, String> {
        Ok(match self.find_answer(message)? {
            Found::Answer(a) => Some(a),
            Found::NoKey | Found::EmptyCard(_) => None,
        })
    }

    /// [`Self::answer`] with the miss reason: no key in the message, or a
    /// key found but no card text in any language.
    pub fn find_answer(&self, message: &str) -> Result<Found, String> {
        self.find_answer_turns(&[message])
    }

    /// [`Self::find_answer`] with conversation memory: `turns` are the
    /// user messages, the LAST one first (the message the request was
    /// routed on), then the earlier ones back in time (the caller passes
    /// at most [`MEMORY_TURNS`]). The key comes from the first turn that
    /// holds one (`KeyHit::turn` = how many turns back) — within a turn
    /// by [`Self::find_key`]'s rule, the strongest key first, the same
    /// rule `key_first` applies; the language and the field come from
    /// `turns[0]` — the question being asked now.
    /// An empty `turns` is [`Found::NoKey`].
    pub fn find_answer_turns(&self, turns: &[&str]) -> Result<Found, String> {
        let Some((turn, mut key)) = turns
            .iter()
            .enumerate()
            .find_map(|(i, t)| self.find_key(t).map(|k| (i, k)))
        else {
            return Ok(Found::NoKey);
        };
        key.turn = turn;
        self.answer_for_key(key, turns[0])
    }

    /// The card of a key already found (by [`Self::find_key`],
    /// [`Self::find_strong_key`], or in an earlier turn), answering
    /// `message`: the language by its script (an empty slot falls back to
    /// the first language with text), the field by its keywords.
    /// [`Found::EmptyCard`] when the entry has no text in any language.
    pub fn answer_for_key(&self, key: KeyHit, message: &str) -> Result<Found, String> {
        let want = pick_lang(message, &self.info.langs);
        let order = std::iter::once(want).chain((0..self.info.langs.len()).filter(|&l| l != want));
        let mut chosen = None;
        for lang_i in order {
            let card = self.card(key.entry, lang_i)?;
            if !card.is_empty() {
                chosen = Some((lang_i, card));
                break;
            }
        }
        let Some((lang_i, card)) = chosen else {
            return Ok(Found::EmptyCard(key));
        };
        let lang = self.info.langs[lang_i].clone();
        let field = select_field(message)
            .filter(|f| card.fields.get(*f).is_some_and(|t| !t.trim().is_empty()))
            .map(str::to_string);
        let text = match &field {
            Some(f) => card.fields[f].clone(),
            None => card.card.clone(),
        };
        let context = context_excerpt(&card, field.as_deref(), &lang);
        Ok(Found::Answer(LookupAnswer {
            id: self.id.clone(),
            key,
            lang,
            field,
            text,
            card: context,
            full_card: card.card,
            decided_by: DecidedBy::Router,
        }))
    }
}

/// [`LookupTable::find_answer`]: what the table found for a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Found {
    /// No key in the message.
    NoKey,
    /// A key, but its entry has no card text in any language (the
    /// builder stores `{"card": "", "fields": {}}` for a language the
    /// entry does not carry).
    EmptyCard(KeyHit),
    Answer(LookupAnswer),
}

/// What the table answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupAnswer {
    /// The record id.
    pub id: String,
    pub key: KeyHit,
    /// The slot language used.
    pub lang: String,
    /// The field chosen (`None` = the whole card).
    pub field: Option<String>,
    /// The answer text (`answer` mode).
    pub text: String,
    /// What `context` mode prepends ([`context_excerpt`]): the whole card
    /// when no field was selected, else its first sentence and the field.
    pub card: String,
    /// The whole card text.
    pub full_card: String,
    /// Who sent the request to the record: the router's decision, or the
    /// `key_first` policy over a backbone decision.
    pub decided_by: DecidedBy,
}

impl LookupAnswer {
    /// One human line: `lookup: <id> | key "…" → entry N (W words,
    /// <source>, <exact|stem>) [| N turns back] | <lang> | field … |
    /// decided by <router|key_first>`.
    pub fn describe(&self) -> String {
        format!(
            "lookup: {} | key {:?}{} → entry {} ({} words, {}, {}){} | {} | field {} | decided by {}",
            self.id,
            self.key.key,
            self.key
                .stem
                .as_deref()
                .map(|s| format!(" (stem {s:?})"))
                .unwrap_or_default(),
            self.key.entry,
            self.key.words,
            self.key.source.label(),
            self.key.via.label(),
            if self.key.turn > 0 {
                format!(" | {} turns back", self.key.turn)
            } else {
                String::new()
            },
            self.lang,
            self.field.as_deref().unwrap_or("— (whole card)"),
            self.decided_by.label()
        )
    }
}

/// The lookup tables of a file, opened on first use and kept
/// (`Sync`: `serve` shares one behind its router).
pub struct LookupTables {
    model: Arc<CmfModel>,
    tables: Mutex<BTreeMap<String, Arc<LookupTable>>>,
}

impl LookupTables {
    pub fn new(model: Arc<CmfModel>) -> Self {
        Self {
            model,
            tables: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn model(&self) -> &Arc<CmfModel> {
        &self.model
    }

    /// Does the file carry at least one lookup record?
    pub fn any(&self) -> bool {
        !LookupTable::lookup_ids(&self.model).is_empty()
    }

    /// Ids of the file's lookup records.
    pub fn ids(&self) -> Vec<String> {
        LookupTable::lookup_ids(&self.model)
    }

    /// The table of `id`; `None` when `id` is not a lookup record.
    pub fn get(&self, id: &str) -> Result<Option<Arc<LookupTable>>, String> {
        if !LookupTable::is_lookup(&self.model, id) {
            return Ok(None);
        }
        let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = tables.get(id) {
            return Ok(Some(t.clone()));
        }
        let t = Arc::new(LookupTable::open(&self.model, id)?);
        tables.insert(id.to_string(), t.clone());
        Ok(Some(t))
    }
}

// ───────────────────────── the routed outcome ─────────────────────────

/// What a routed request does once its target is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupOutcome {
    /// The target is the backbone or an ordinary skill: run its lane.
    NotLookup,
    /// The target is a lookup record but the mode is `off`: the backbone
    /// runs the plain message.
    Off { id: String },
    /// A lookup record, no key in the message: the backbone runs the
    /// plain message — unchanged, bit-identical to F0.
    Miss { id: String },
    /// `answer` mode: the table's text IS the answer; nothing runs.
    Answer(LookupAnswer),
    /// `context` mode: the backbone runs [`context_prompt`]`(card, message)`.
    Context(LookupAnswer),
}

impl LookupOutcome {
    /// The table's answer when a key was found (either mode).
    pub fn hit(&self) -> Option<&LookupAnswer> {
        match self {
            Self::Answer(a) | Self::Context(a) => Some(a),
            _ => None,
        }
    }

    pub fn is_hit(&self) -> bool {
        self.hit().is_some()
    }

    /// The text the backbone generates from (`context` mode prepends the
    /// card; everything else keeps `user_text`).
    pub fn generation_text(&self, user_text: &str) -> String {
        match self {
            Self::Context(a) => context_prompt(&a.card, user_text),
            _ => user_text.to_string(),
        }
    }

    /// One human line for a lookup target (`None` for [`Self::NotLookup`]).
    pub fn describe(&self) -> Option<String> {
        Some(match self {
            Self::NotLookup => return None,
            Self::Off { id } => format!("lookup: {id} | mode off — the backbone runs"),
            Self::Miss { id } => {
                format!("lookup: {id} | no key in the message — the backbone runs unchanged")
            }
            Self::Answer(a) => format!("{} | mode answer", a.describe()),
            Self::Context(a) => format!("{} | mode context", a.describe()),
        })
    }

    /// The lookup record the router decided on (`Off`, `Miss` and the
    /// hits), `None` for any other target.
    pub fn lookup_id(&self) -> Option<&str> {
        match self {
            Self::NotLookup => None,
            Self::Off { id } | Self::Miss { id } => Some(id),
            Self::Answer(a) | Self::Context(a) => Some(&a.id),
        }
    }

    /// Who sent the request to the lookup record (`None` for
    /// [`Self::NotLookup`]): a hit says, `Off` / `Miss` come only from the
    /// decision the lookup step was handed (the `key_first` policy takes a
    /// request only when the table answers it).
    pub fn decided_by(&self) -> Option<DecidedBy> {
        match self {
            Self::NotLookup => None,
            Self::Off { .. } | Self::Miss { .. } => Some(DecidedBy::Router),
            Self::Answer(a) | Self::Context(a) => Some(a.decided_by),
        }
    }

    /// Adds the lookup fields to a route summary
    /// ([`RouteDecision::summary_json`]): `lookup_hit`; whenever the
    /// target was a lookup record `lookup_mode`, `decided_target` (the
    /// record the decision chose — `target` is the lane that ran, the
    /// backbone after a miss) and `decided_by` (`router` | `key_first`);
    /// on a hit `lookup_key`, `lookup_entry`, `lookup_lang`, `field`,
    /// `lookup_match` (`exact` | `stem`), `lookup_turn` (how many user
    /// turns back the key was found; 0 = the message routed on),
    /// `lookup_key_words`.
    pub fn annotate(&self, summary: &mut serde_json::Value, mode: LookupMode) {
        let serde_json::Value::Object(m) = summary else {
            return;
        };
        m.insert("lookup_hit".into(), serde_json::json!(self.is_hit()));
        if let Some(id) = self.lookup_id() {
            m.insert("lookup_mode".into(), serde_json::json!(mode.label()));
            m.insert("decided_target".into(), serde_json::json!(id));
        }
        if let Some(by) = self.decided_by() {
            m.insert("decided_by".into(), serde_json::json!(by.label()));
        }
        if let Some(a) = self.hit() {
            m.insert("lookup_key".into(), serde_json::json!(a.key.key));
            m.insert("lookup_entry".into(), serde_json::json!(a.key.entry));
            m.insert("lookup_lang".into(), serde_json::json!(a.lang));
            m.insert("field".into(), serde_json::json!(a.field));
            m.insert("lookup_match".into(), serde_json::json!(a.key.via.label()));
            m.insert("lookup_turn".into(), serde_json::json!(a.key.turn));
            m.insert("lookup_key_words".into(), serde_json::json!(a.key.words));
        }
    }
}

/// The lookup step after the routing decision: a target that is a lookup
/// record becomes its outcome under `mode`. No key (or mode `off`) turns
/// the decision into the BACKBONE with the reason kept; a hit keeps the
/// skill target and notes the key. Any other target passes through as
/// [`LookupOutcome::NotLookup`]. One user message, no memory. The
/// decision is taken as given — the `key_first` policy needs to know how
/// it was made ([`resolve_lookup_gated`]).
pub fn resolve_lookup(
    tables: &LookupTables,
    decision: RouteDecision,
    user_text: &str,
    mode: LookupMode,
) -> Result<(RouteDecision, LookupOutcome), String> {
    resolve_lookup_turns(tables, decision, &[user_text], mode)
}

/// [`resolve_lookup`] with conversation memory: `turns` as
/// [`LookupTable::find_answer_turns`] takes them — the message routed on
/// first, then the earlier user turns (`serve` passes up to
/// [`MEMORY_TURNS`]).
pub fn resolve_lookup_turns(
    tables: &LookupTables,
    decision: RouteDecision,
    turns: &[&str],
    mode: LookupMode,
) -> Result<(RouteDecision, LookupOutcome), String> {
    let Some(id) = decision.skill().map(str::to_string) else {
        return Ok((decision, LookupOutcome::NotLookup));
    };
    let Some(table) = tables.get(&id)? else {
        return Ok((decision, LookupOutcome::NotLookup));
    };
    let to_backbone = |d: RouteDecision, why: &str| RouteDecision {
        target: RouteTarget::Backbone,
        routing: d.routing,
        reason: format!("{why} [decision was: {}]", d.reason),
    };
    if mode == LookupMode::Off {
        let d = to_backbone(
            decision,
            &format!("lookup '{id}' ignored (lookup mode off): the backbone runs"),
        );
        return Ok((d, LookupOutcome::Off { id }));
    }
    let answer = match table.find_answer_turns(turns)? {
        Found::Answer(a) => a,
        Found::NoKey => {
            let d = to_backbone(
                decision,
                &format!("lookup '{id}': no key in the message — the backbone runs unchanged"),
            );
            return Ok((d, LookupOutcome::Miss { id }));
        }
        Found::EmptyCard(key) => {
            let d = to_backbone(
                decision,
                &format!(
                    "lookup '{id}': key {:?} → entry {} has no card in any language — the \
                     backbone runs unchanged",
                    key.key, key.entry
                ),
            );
            return Ok((d, LookupOutcome::Miss { id }));
        }
    };
    let mut d = decision;
    d.reason = format!(
        "{}; lookup key {:?} → entry {}{}{}{}",
        d.reason,
        answer.key.key,
        answer.key.entry,
        match answer.key.via {
            MatchVia::Exact => "",
            MatchVia::Stem => " (stem)",
        },
        if answer.key.turn > 0 {
            format!(" ({} turns back)", answer.key.turn)
        } else {
            String::new()
        },
        answer
            .field
            .as_deref()
            .map(|f| format!(", field {f}"))
            .unwrap_or_default()
    );
    let outcome = match mode {
        LookupMode::Answer => LookupOutcome::Answer(answer),
        LookupMode::Context => LookupOutcome::Context(answer),
        LookupMode::Off => unreachable!("handled above"),
    };
    Ok((d, outcome))
}

/// What the `key_first` policy needs to take a decision
/// ([`resolve_lookup_gated`]): that the decision was the ROUTER's own (a
/// caller that pinned the target passes no gate — `--route backbone`,
/// `--skill none` and a forced record are never overridden) and the
/// conditions the router's own pick of the record would have met.
#[derive(Debug, Clone, Copy)]
pub struct KeyFirstGate {
    /// The routing options of the decision: a record the router could not
    /// pick under them ([`router::is_routable`] — `active` with a
    /// measured gate, or any non-retired class with `include_quarantine`)
    /// is not taken by `key_first` either.
    pub opts: RouteOptions,
    /// The frame the request generates under, and whether the caller can
    /// render the user text as cmf-im-v1: the record's `prompt_contract`
    /// is checked exactly as for the router's pick
    /// ([`router::enforce_prompt_contract`]).
    pub frame: PromptFrame,
    pub can_render: bool,
}

impl KeyFirstGate {
    /// A router decision under `opts` for a request framed as `frame`
    /// (no rendering).
    pub fn new(opts: RouteOptions, frame: PromptFrame) -> Self {
        Self {
            opts,
            frame,
            can_render: false,
        }
    }

    /// The caller can render a raw prompt as cmf-im-v1 (`run`).
    pub fn can_render(mut self, on: bool) -> Self {
        self.can_render = on;
        self
    }
}

/// [`resolve_lookup_turns`] with the records' routing policy. `key_first`
/// is `Some` when `decision` is the router's own (see [`KeyFirstGate`]).
///
/// When the router sent the request to the BACKBONE (for whatever reason:
/// the backbone nearest, a novel input, the margin not beaten, the
/// prompt contract of a record — but NOT the router's fail-closed state:
/// no router policy, no calibration, a stale `skills_hash`, no routable
/// class, [`router::no_candidate_reason`]), mode is not `off`, and a
/// lookup record whose policy is `key_first` — routable under
/// `gate.opts`, its prompt contract satisfied by `gate.frame` — finds a
/// STRONG key ([`LookupTable::find_strong_key`]) in the LAST user message
/// `turns[0]` (never in an earlier turn: a plant named three turns ago
/// does not pull an unrelated question into the table) whose card has
/// text, the request goes to that record: target the record, the
/// router's scores kept, reason `key_first: …`, `decided_by` = `key_first`
/// (records in header order, the first that answers wins). Every other
/// case is exactly [`resolve_lookup_turns`] — a message without a strong
/// key keeps the router's decision, and the backbone lane runs it
/// bit-identically to F0. A decision for a record (the router picked it)
/// resolves as always (any key, conversation memory), `decided_by` =
/// `router`; a decision for another skill is never overridden.
pub fn resolve_lookup_gated(
    tables: &LookupTables,
    decision: RouteDecision,
    key_first: Option<KeyFirstGate>,
    turns: &[&str],
    mode: LookupMode,
) -> Result<(RouteDecision, LookupOutcome), String> {
    if let (Some(gate), RouteTarget::Backbone, false, Some(message)) = (
        key_first,
        &decision.target,
        mode == LookupMode::Off,
        turns.first(),
    ) {
        if let Some(taken) = key_first_take(tables, &decision, gate, message, mode)? {
            return Ok(taken);
        }
    }
    resolve_lookup_turns(tables, decision, turns, mode)
}

/// The `key_first` step of [`resolve_lookup_gated`] on a backbone
/// decision: `Some` when a `key_first` record takes the request.
fn key_first_take(
    tables: &LookupTables,
    decision: &RouteDecision,
    gate: KeyFirstGate,
    message: &str,
    mode: LookupMode,
) -> Result<Option<(RouteDecision, LookupOutcome)>, String> {
    let model = tables.model().clone();
    // The router's own fail-closed state (review KF-4): on a file whose
    // router could not pick ANY skill — no policy, no calibration, a stale
    // `skills_hash`, no routable class — every request runs the backbone,
    // and key_first does not reopen the table behind it.
    let mut router_ok: Option<bool> = None;
    for id in tables.ids() {
        let Some(rec) = LookupTable::record(&model, &id) else {
            continue;
        };
        let Some(info) = rec.lookup.as_ref() else {
            continue;
        };
        if LookupPolicy::of(info) != LookupPolicy::KeyFirst {
            continue;
        }
        let ok = *router_ok.get_or_insert_with(|| key_first_router_ok(&model, gate.opts));
        if !ok {
            return Ok(None);
        }
        if !router::is_routable(rec, gate.opts) {
            continue;
        }
        let pick = RouteDecision::forced(RouteTarget::Skill(id.clone()), "key_first");
        let (pick, _) =
            router::enforce_prompt_contract(&model.header, pick, gate.frame, gate.can_render);
        if pick.skill().is_none() {
            continue;
        }
        let Some(table) = tables.get(&id)? else {
            continue;
        };
        let Some(key) = table.find_strong_key(message) else {
            continue;
        };
        let mut answer = match table.answer_for_key(key, message)? {
            Found::Answer(a) => a,
            Found::NoKey | Found::EmptyCard(_) => continue,
        };
        answer.decided_by = DecidedBy::KeyFirst;
        let reason = format!(
            "key_first: lookup '{id}' takes the request on the strong key {:?} ({} words, {}, \
             {}) [router decision: {}]; lookup key {:?} → entry {}{}",
            answer.key.key,
            answer.key.words,
            answer.key.source.label(),
            answer.key.via.label(),
            decision.reason,
            answer.key.key,
            answer.key.entry,
            answer
                .field
                .as_deref()
                .map(|f| format!(", field {f}"))
                .unwrap_or_default()
        );
        let d = RouteDecision {
            target: RouteTarget::Skill(id),
            routing: decision.routing.clone(),
            reason,
        };
        let outcome = match mode {
            LookupMode::Answer => LookupOutcome::Answer(answer),
            LookupMode::Context => LookupOutcome::Context(answer),
            LookupMode::Off => unreachable!("key_first never acts in mode off"),
        };
        return Ok(Some((d, outcome)));
    }
    Ok(None)
}

/// May `key_first` act on this file at all? Only where the router itself
/// could pick a skill: a router-v2 policy, a calibration whose
/// `skills_hash` binds, at least one class routable under `opts`
/// ([`router::no_candidate_reason`]). Anything else is the router's
/// fail-closed state — the backbone for every request.
fn key_first_router_ok(model: &CmfModel, opts: RouteOptions) -> bool {
    router::is_router_v2(model) && router::no_candidate_reason(&model.header, opts).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::knowledge::key_hash;

    /// A finder over `(key, entry)` pairs — what the table's binary search
    /// answers.
    fn finder(keys: &[(&str, u32)]) -> impl Fn(u64) -> Option<u32> {
        let m: BTreeMap<u64, u32> = keys.iter().map(|(k, e)| (key_hash(k), *e)).collect();
        move |h| m.get(&h).copied()
    }

    #[test]
    fn mode_parses_flag_and_env_spellings() {
        assert_eq!(LookupMode::parse("answer").unwrap(), LookupMode::Answer);
        assert_eq!(LookupMode::parse(" Context ").unwrap(), LookupMode::Context);
        assert_eq!(LookupMode::parse("OFF").unwrap(), LookupMode::Off);
        assert_eq!(LookupMode::parse("none").unwrap(), LookupMode::Off);
        assert!(LookupMode::parse("maybe").unwrap_err().contains("answer | context | off"));
        assert_eq!(LookupMode::resolve(Some("off")).unwrap(), LookupMode::Off);
        assert!(LookupMode::resolve(Some("x")).unwrap_err().starts_with("--lookup-mode"));
        assert_eq!(LookupMode::default().label(), "answer");
    }

    #[test]
    fn ngram_extraction_takes_the_longest_match_cyrillic_and_latin() {
        let find = finder(&[
            ("Пихта бальзамическая", 0),
            ("пихта", 7),
            ("Abies balsamea", 0),
            ("balsam fir", 0),
            ("Ромашка аптечная", 1),
            ("chamomile", 1),
        ]);
        // Cyrillic, 2 words beat the 1-word key of the same plant.
        let h = extract_key("Какое семейство у растения Пихта бальзамическая?", &find).unwrap();
        assert_eq!(h.key, "пихта бальзамическая");
        assert_eq!((h.entry, h.words, h.source), (0, 2, KeySource::Ngram));
        // The 1-word key alone.
        let h = extract_key("пихта — что это?", &find).unwrap();
        assert_eq!((h.key.as_str(), h.entry, h.words), ("пихта", 7, 1));
        // Latin, punctuation and case folded by the normalisation.
        let h = extract_key("What is BALSAM-FIR used for?", &find).unwrap();
        assert_eq!((h.key.as_str(), h.entry), ("balsam fir", 0));
        let h = extract_key("Tell me about chamomile.", &find).unwrap();
        assert_eq!((h.key.as_str(), h.entry), ("chamomile", 1));
        // No match: nothing.
        assert_eq!(extract_key("What is the capital of France?", &find), None);
        assert_eq!(extract_key("", &find), None);
        assert_eq!(extract_key("   ...  ", &find), None);
    }

    #[test]
    fn ties_go_to_the_first_occurrence() {
        let find = finder(&[("balsam fir", 0), ("chamomile", 1)]);
        let h = extract_key("chamomile or balsam fir?", &find).unwrap();
        assert_eq!(h.entry, 0, "the longer key wins over the earlier shorter one");
        let find = finder(&[("balsam fir", 0), ("red pine", 2)]);
        let h = extract_key("red pine and balsam fir", &find).unwrap();
        assert_eq!(h.entry, 2, "equal length: the first occurrence");
    }

    #[test]
    fn parenthesised_binomial_is_tried_first() {
        let find = finder(&[("Abies balsamea", 0), ("пихты", 3)]);
        // The declined Cyrillic name is also a key, but the binomial in
        // parentheses is resolved first.
        let h = extract_key("Какие части пихты (Abies balsamea) используют?", &find).unwrap();
        assert_eq!(h.key, "abies balsamea");
        assert_eq!((h.entry, h.source), (0, KeySource::Parenthesised));
        // A non-Latin or too long group is skipped; the n-gram scan follows.
        let h = extract_key("Какие части пихты (см. выше) используют?", &find).unwrap();
        assert_eq!((h.key.as_str(), h.source), ("пихты", KeySource::Ngram));
        // A group that is not a key falls through to the n-grams.
        let find = finder(&[("abies balsamea", 0)]);
        let h = extract_key("fir (Pinus sylvestris) vs abies balsamea", &find).unwrap();
        assert_eq!((h.entry, h.source), (0, KeySource::Ngram));
        assert_eq!(parenthesised("a (b) c (d e) (f"), vec!["b", "d e"]);
        // Nested parentheses: the innermost group is tried first, then
        // the whole group; the n-grams follow.
        assert_eq!(
            parenthesised("Репешок (репешок аптечный (Agrimonia eupatoria)?"),
            vec!["Agrimonia eupatoria", "репешок аптечный (Agrimonia eupatoria"]
        );
        assert_eq!(parenthesised("x (a (b (c)) d)"), vec!["c", "a (b (c"]);
    }

    #[test]
    fn nested_and_accented_binomials_resolve_through_the_parentheses() {
        // ru-wiki spells the binomial with stress marks; the table's key
        // (also accented, or not) normalises to the same plain letters.
        let find = finder(&[
            ("Oxycóccus", 0),
            ("клюква", 9),
            ("Agrimonia eupatoria", 1),
            ("репешок обыкновенный репешок аптечный", 8),
            ("Pinus", 2),
            ("сосна", 7),
            ("Strychnos nux-vomica", 3),
            ("чилибуха", 6),
        ]);
        let h = extract_key("К какому семейству относится Клюква (Oxycóccus)?", &find).unwrap();
        assert_eq!((h.entry, h.source), (0, KeySource::Parenthesised));
        assert_eq!(h.key, "oxycoccus");
        let h = extract_key("Сосна (Pínus) — семейство?", &find).unwrap();
        assert_eq!((h.entry, h.key.as_str()), (2, "pinus"));
        let h = extract_key(
            "К какому семейству относится Репешок обыкновенный (репешок аптечный (Agrimonia eupatoria)?",
            &find,
        )
        .unwrap();
        assert_eq!((h.entry, h.source), (1, KeySource::Parenthesised));
        let h = extract_key(
            "Чилибуха (Чилибуха обыкновенная (Strychnos nux-vomica) — что это?",
            &find,
        )
        .unwrap();
        assert_eq!((h.entry, h.key.as_str()), (3, "strychnos nux vomica"));
        // Latin letters the normalisation cannot fold are still Latin;
        // digits and Cyrillic are not a binomial.
        assert!(is_latin_word("æsculus") && is_latin_word("øst") && is_latin_word("straße"));
        assert!(!is_latin_word("клюква") && !is_latin_word("l2") && !is_latin_word(""));
        let find = finder(&[("2024", 5), ("клюква", 9)]);
        let h = extract_key("Клюква (2024)", &find).unwrap();
        assert_eq!((h.entry, h.source), (9, KeySource::Ngram));
    }

    #[test]
    fn field_selection_by_keywords() {
        assert_eq!(select_field("К какому семейству относится пихта?"), Some("family"));
        assert_eq!(select_field("What family is balsam fir in?"), Some("family"));
        assert_eq!(select_field("Какие части растения используют?"), Some("parts"));
        assert_eq!(select_field("Which part of the plant is used?"), Some("parts"));
        assert_eq!(select_field("Какие действующие вещества?"), Some("compounds"));
        assert_eq!(select_field("Main constituents?"), Some("compounds"));
        assert_eq!(select_field("Где применяется пихта?"), Some("uses"));
        assert_eq!(select_field("What is it used for?"), Some("uses"));
        assert_eq!(select_field("Какие формы препаратов бывают?"), Some("preparations"));
        assert_eq!(select_field("In what form is it taken?"), Some("preparations"));
        assert_eq!(select_field("Есть ли противопоказания?"), Some("safety"));
        assert_eq!(select_field("Any side effects?"), Some("safety"));
        assert_eq!(select_field("Is it safe?"), Some("safety"));
        assert_eq!(select_field("Какая доказательная база?"), Some("evidence"));
        assert_eq!(select_field("Is there a study on it?"), Some("evidence"));
        // Whole words only for the short Latin ones.
        assert_eq!(select_field("Because information matters"), None);
        assert_eq!(select_field("Что такое пихта?"), None);
        // Rule order: family before parts.
        assert_eq!(select_field("Семейство и части растения"), Some("family"));
        // Safety of a USE is safety, not uses; a form it is used in is a
        // preparation.
        assert_eq!(select_field("Is lungwort safe to use today?"), Some("safety"));
        assert_eq!(select_field("What safety measures apply to digoxin use?"), Some("safety"));
        assert_eq!(
            select_field("Is the plant safe, and is its use supported by evidence?"),
            Some("safety")
        );
        assert_eq!(select_field("Насколько безопасно применение жимолости?"), Some("safety"));
        assert_eq!(
            select_field("Каковы противопоказания и меры безопасности при применении алтея?"),
            Some("safety")
        );
        assert_eq!(select_field("In what form is it used?"), Some("preparations"));
        assert_eq!(select_field("Какие части растения используются?"), Some("parts"));
        // `част` is a whole-word rule: частуха is a plant, часто an adverb.
        assert_eq!(select_field("Как выглядит частуха обыкновенная?"), None);
        assert_eq!(select_field("Где растёт частуха обыкновенная?"), None);
        assert_eq!(
            select_field("Какие сведения о безопасности частухи приводятся?"),
            Some("safety")
        );
        assert_eq!(
            select_field("Каково научное название частухи обыкновенной и к какому семейству она относится?"),
            Some("family")
        );
        assert_eq!(select_field("Часто ли её применяют?"), Some("uses"));
        assert_eq!(select_field("Какую часть растения собирают?"), Some("parts"));
        // The dev-set spellings the first vocabulary missed.
        assert_eq!(select_field("Опасно ли это растение?"), Some("safety"));
        assert_eq!(select_field("Чем опасна наперстянка шерстистая?"), Some("safety"));
        assert_eq!(
            select_field("Каковы побочные эффекты и лекарственные взаимодействия галантамина?"),
            Some("safety")
        );
        assert_eq!(select_field("How toxic is Podophyllum peltatum?"), Some("safety"));
        assert_eq!(select_field("Is it poisonous to cats?"), Some("safety"));
        assert_eq!(select_field("What risks are mentioned?"), Some("safety"));
        assert_eq!(
            select_field("Какие биологически активные соединения найдены в растении?"),
            Some("compounds")
        );
        assert_eq!(select_field("Каков химический состав?"), Some("compounds"));
        assert_eq!(select_field("What are the active ingredients?"), Some("compounds"));
        assert_eq!(select_field("Which chemicals does it contain?"), Some("compounds"));
        assert_eq!(
            select_field("Какие данные исследований приводятся в источнике?"),
            Some("evidence")
        );
        assert_eq!(select_field("Are there clinical trials?"), Some("evidence"));
        assert_eq!(select_field("What does the research say?"), Some("evidence"));
        assert_eq!(select_field("Какая дозировка?"), Some("preparations"));
        assert_eq!(select_field("What is the usual dose?"), Some("preparations"));
        assert_eq!(select_field("Recommended dosage?"), Some("preparations"));
        assert_eq!(FIELDS, &["family", "parts", "compounds", "safety", "evidence", "preparations", "uses"]);
    }

    #[test]
    fn context_excerpt_keeps_the_first_sentence_and_the_field() {
        let c = Card::parse(
            r#"{"card":"Пихта бальзамическая (Abies balsamea) — хвойное дерево. Растёт в Канаде. Смола ароматна.","fields":{"family":"Сосновые (Pinaceae)","parts":"хвоя, смола"}}"#,
        )
        .unwrap();
        assert_eq!(c.first_sentence(), "Пихта бальзамическая (Abies balsamea) — хвойное дерево.");
        assert_eq!(
            context_excerpt(&c, Some("family"), "ru"),
            "Пихта бальзамическая (Abies balsamea) — хвойное дерево.\nСемейство: Сосновые (Pinaceae)"
        );
        assert_eq!(context_excerpt(&c, Some("parts"), "en"), format!("{}\nParts: хвоя, смола", c.first_sentence()));
        assert_eq!(context_excerpt(&c, Some("uses"), "ru"), c.card, "a field the card lacks: the whole card");
        assert_eq!(context_excerpt(&c, None, "ru"), c.card);
        assert_eq!(
            context_excerpt(&c, Some("family"), "de"),
            format!("{}\nFamily: Сосновые (Pinaceae)", c.first_sentence()),
            "an unknown slot language labels in English"
        );
        // An abbreviation's period is not a sentence end; no terminator →
        // the cap at a word boundary.
        assert_eq!(first_sentence("Hypericum perforatum L. is a herb. More."), "Hypericum perforatum L. is a herb.");
        assert_eq!(first_sentence("Abies balsamea?  yes"), "Abies balsamea?");
        assert_eq!(first_sentence("a.b.c end"), "a.b.c end");
        let long: String = "слово ".repeat(100);
        let head = first_sentence(&long);
        assert!(head.chars().count() <= FIRST_SENTENCE_MAX && head.ends_with("слово"), "{head:?}");
        assert_eq!(first_sentence("   "), "");
        assert_eq!(field_label("family", "ru"), "Семейство");
        assert_eq!(field_label("family", "en"), "Family");
        assert_eq!(field_label("origin", "ru"), "Origin");
        assert!(Card::parse(r#"{"card":"","fields":{}}"#).unwrap().is_empty());
        assert!(Card::parse(r#"{"card":" ","fields":{"family":""}}"#).unwrap().is_empty());
        assert!(!Card::parse(r#"{"card":"","fields":{"family":"x"}}"#).unwrap().is_empty());
    }

    #[test]
    fn language_by_script_with_fallbacks() {
        let ru_en = vec!["ru".to_string(), "en".to_string()];
        assert_eq!(pick_lang("Что такое пихта?", &ru_en), 0);
        assert_eq!(pick_lang("What is fir?", &ru_en), 1);
        let en_only = vec!["en".to_string()];
        assert_eq!(pick_lang("Что такое пихта?", &en_only), 0);
        let de_ru = vec!["de".to_string(), "ru".to_string()];
        assert_eq!(pick_lang("What is fir?", &de_ru), 0, "no en: the first language");
        assert_eq!(pick_lang("Пихта", &de_ru), 1);
        assert!(has_cyrillic("Ё"));
        assert!(!has_cyrillic("fir"));
    }

    #[test]
    fn card_parse_and_context_prompt() {
        let c = Card::parse(r#"{"card":"Abies balsamea — a fir.","fields":{"family":"Pinaceae","n":3,"x":null}}"#)
            .unwrap();
        assert_eq!(c.card, "Abies balsamea — a fir.");
        assert_eq!(c.fields["family"], "Pinaceae");
        assert_eq!(c.fields["n"], "3");
        assert!(!c.fields.contains_key("x"));
        assert!(Card::parse("[1]").is_err());
        assert_eq!(Card::parse("{}").unwrap(), Card::default());
        assert_eq!(
            context_prompt("CARD", "Q?"),
            "Справочная карточка / Reference card:\nCARD\n\nQ?"
        );
    }

    #[test]
    fn outcome_annotation_and_generation_text() {
        let a = LookupAnswer {
            id: "herbs".into(),
            key: KeyHit {
                key: "abies balsamea".into(),
                hash: 1,
                entry: 4,
                words: 2,
                source: KeySource::Ngram,
                via: MatchVia::Exact,
                stem: None,
                turn: 0,
                strong: true,
            },
            lang: "en".into(),
            field: Some("family".into()),
            text: "Pinaceae".into(),
            card: "CARD".into(),
            full_card: "CARD. MORE.".into(),
            decided_by: DecidedBy::Router,
        };
        let mut s = serde_json::json!({"target": "herbs"});
        LookupOutcome::Context(a.clone()).annotate(&mut s, LookupMode::Context);
        assert_eq!(s["lookup_hit"], true);
        assert_eq!(s["lookup_key"], "abies balsamea");
        assert_eq!(s["lookup_entry"], 4);
        assert_eq!(s["lookup_lang"], "en");
        assert_eq!(s["field"], "family");
        assert_eq!(s["lookup_mode"], "context");
        assert_eq!(s["decided_target"], "herbs");
        assert_eq!(s["decided_by"], "router");
        assert_eq!(s["lookup_key_words"], 2);
        assert_eq!(
            LookupOutcome::Context(a.clone()).generation_text("Q?"),
            context_prompt("CARD", "Q?")
        );
        assert_eq!(LookupOutcome::Answer(a.clone()).generation_text("Q?"), "Q?");
        let mut s = serde_json::json!({"target": "backbone"});
        LookupOutcome::NotLookup.annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["lookup_hit"], false);
        assert!(s.get("lookup_mode").is_none());
        assert!(s.get("decided_target").is_none());
        assert!(s.get("decided_by").is_none());
        // A key_first hit says so; a miss comes from the handed decision.
        let mut kf = a.clone();
        kf.decided_by = DecidedBy::KeyFirst;
        let mut s = serde_json::json!({"target": "herbs"});
        LookupOutcome::Answer(kf.clone()).annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["decided_by"], "key_first");
        assert!(kf.describe().ends_with("decided by key_first"), "{}", kf.describe());
        assert_eq!(
            LookupOutcome::Miss { id: "herbs".into() }.decided_by(),
            Some(DecidedBy::Router)
        );
        assert_eq!(LookupOutcome::NotLookup.decided_by(), None);
        let mut s = serde_json::json!({"target": "backbone"});
        LookupOutcome::Miss { id: "herbs".into() }.annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["lookup_hit"], false);
        assert_eq!(s["lookup_mode"], "answer");
        assert_eq!(s["decided_target"], "herbs", "the router's choice survives the miss");
        assert_eq!(s["decided_by"], "router");
        assert_eq!(s["target"], "backbone", "the lane that ran");
        assert_eq!(LookupOutcome::Off { id: "h".into() }.lookup_id(), Some("h"));
        assert_eq!(LookupOutcome::NotLookup.lookup_id(), None);
        assert!(LookupOutcome::NotLookup.describe().is_none());
        assert!(
            LookupOutcome::Miss { id: "herbs".into() }
                .describe()
                .unwrap()
                .contains("no key")
        );
        // The match kind and the turn are reported; a stem hit two turns
        // back says so in the human line too.
        let mut s = serde_json::json!({"target": "herbs"});
        let mut b = a.clone();
        b.key.via = MatchVia::Stem;
        b.key.stem = Some("abie balsamea".into());
        b.key.turn = 2;
        LookupOutcome::Answer(b.clone()).annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["lookup_match"], "stem");
        assert_eq!(s["lookup_turn"], 2);
        let line = b.describe();
        assert!(line.contains("stem \"abie balsamea\"") && line.contains("2 turns back"), "{line}");
        let mut s = serde_json::json!({});
        LookupOutcome::Answer(a).annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["lookup_match"], "exact");
        assert_eq!(s["lookup_turn"], 0);
    }

    /// A finder over the STEMS of `(key, entry)` pairs — what the table's
    /// stem index answers.
    fn stem_finder(keys: &[(&str, u32)]) -> (StemIndex, impl Fn(u64) -> Option<u32>) {
        let norm: Vec<(String, u32)> = keys.iter().map(|(k, e)| (normalize_key(k), *e)).collect();
        let idx = StemIndex::from_keys(norm.iter().map(|(k, e)| (k.as_str(), *e)));
        let idx2 = idx.clone();
        (idx, move |h| idx2.find(h))
    }

    #[test]
    fn stemmer_folds_russian_and_english_inflections() {
        // The dev-set misses: an inflected name meets its nominative key.
        assert_eq!(stem_word("тойона"), "тойон");
        assert_eq!(stem_word("тойон"), "тойон");
        assert_eq!(stem_key("магнолии лекарственной"), "магнол лекарственн");
        assert_eq!(stem_key("магнолия лекарственная"), "магнол лекарственн");
        assert_eq!(stem_key("кора магнолии лекарственной"), "кор магнол лекарственн");
        assert_eq!(stem_word("горопито"), "горопит");
        assert_eq!(stem_word("горопито"), stem_word("горопито"));
        // Every case of a noun and an adjective lands on one stem.
        for w in ["ромашка", "ромашки", "ромашке", "ромашку", "ромашкой", "ромашками", "ромашках"] {
            assert_eq!(stem_word(w), "ромашк", "{w}");
        }
        for w in ["лекарственный", "лекарственная", "лекарственное", "лекарственного", "лекарственному", "лекарственным", "лекарственными", "лекарственных", "лекарственные", "лекарственную"] {
            assert_eq!(stem_word(w), "лекарственн", "{w}");
        }
        // Nouns in -ой / -ей / -ь: the ending and the vowel before it go.
        assert_eq!(stem_word("зверобой"), stem_word("зверобоя"));
        assert_eq!(stem_word("зверобоя"), "звероб");
        assert_eq!(stem_word("шалфей"), stem_word("шалфея"));
        assert_eq!(stem_word("полынь"), stem_word("полыни"));
        assert_eq!(stem_word("растения"), "растен");
        // Short words keep ≥ 3 letters: `чай`, `вид`, `дуб` are never cut.
        assert_eq!(stem_word("чай"), "чай");
        assert_eq!(stem_word("чая"), "чая");
        assert_eq!(stem_word("вид"), "вид");
        assert_eq!(stem_word("виды"), "вид");
        assert_eq!(stem_word("дуба"), "дуб");
        // English plurals; short Latin words untouched; binomials mostly
        // untouched; `y` → `i` meets `ies` → `i`.
        assert_eq!(stem_word("herbs"), "herb");
        assert_eq!(stem_word("roses"), "rose");
        assert_eq!(stem_word("berries"), "berri");
        assert_eq!(stem_word("berry"), "berri");
        assert_eq!(stem_word("daisies"), stem_word("daisy"));
        assert_eq!(stem_word("grasses"), "grass");
        assert_eq!(stem_word("grass"), "grass");
        assert_eq!(stem_word("sage"), "sage");
        assert_eq!(stem_word("uses"), "uses");
        assert_eq!(stem_word("wort"), "wort");
        assert_eq!(stem_word("pinus"), "pinus");
        assert_eq!(stem_word("officinalis"), "officinalis");
        assert_eq!(stem_word("balsamea"), "balsamea");
        assert_eq!(stem_key("st john s wort"), "st john s wort");
        // Digits and the empty word are left alone.
        assert_eq!(stem_word("2024"), "2024");
        assert_eq!(stem_word("l2"), "l2");
        assert_eq!(stem_word(""), "");
        assert_eq!(stem_key("  "), "");
        // The suffix list is ordered longest first (the longest ending is
        // the one stripped).
        let lens: Vec<usize> = RU_SUFFIXES.iter().map(|s| s.chars().count()).collect();
        assert!(lens.windows(2).all(|w| w[0] >= w[1]), "{lens:?}");
    }

    #[test]
    fn stem_index_serves_inflected_names_after_the_exact_index_fails() {
        let keys = [
            ("тойон", 0u32),
            ("магнолия лекарственная", 1),
            ("магнолия", 2),
            ("чай", 3),
            ("вид", 4),
            ("укроп", 5),
            ("Abies balsamea", 6),
        ];
        let find = finder(&keys);
        let (idx, find_stem) = stem_finder(&keys);
        assert_eq!((idx.len(), idx.ambiguous, idx.keys_in), (7, 0, 7));
        // An inflected one-word name (stem ≥ 5 letters).
        let h = extract_key_with("традиционные применения тойона", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via, h.source, h.words), (0, MatchVia::Stem, KeySource::Ngram, 1));
        assert_eq!((h.key.as_str(), h.stem.as_deref()), ("тойона", Some("тойон")));
        assert_eq!(h.hash, key_hash("тойон"), "the stem's hash");
        // An inflected two-word name: the longest stem n-gram wins over
        // the one-word stem of the same message.
        let h = extract_key_with("кора магнолии лекарственной", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via, h.words), (1, MatchVia::Stem, 2));
        assert_eq!((h.key.as_str(), h.stem.as_deref()), ("магнолии лекарственной", Some("магнол лекарственн")));
        let h = extract_key_with("настойка магнолии", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via), (2, MatchVia::Stem));
        let h = extract_key_with("семена укропа", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via), (5, MatchVia::Stem));
        // A strong key first (review KF-3): the two-word stem n-gram beats
        // the exact one-word key of the same message — the rule key_first
        // uses, so every path names one entry.
        let h = extract_key_with("магнолия лекарственной", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via, h.words), (1, MatchVia::Stem, 2));
        assert!(is_strong_key(&h));
        // Among weak keys, exact before stem, as before.
        let h = extract_key_with("магнолия и тойона", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via, h.words), (2, MatchVia::Exact, 1));
        let h = extract_key_with("тойон — что это?", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via, h.stem), (0, MatchVia::Exact, None));
        // Never by stem: a one-word stem under 5 letters (`вид`, `чай`);
        // the exact forms still match.
        assert_eq!(extract_key_with("какие виды бывают?", &find, Some(&find_stem)), None);
        assert_eq!(extract_key_with("чашка чая и чаёв", &find, Some(&find_stem)), None);
        assert_eq!(extract_key_with("вида", &find, Some(&find_stem)), None);
        let h = extract_key_with("какой вид чая лучше?", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.via), (4, MatchVia::Exact));
        // Without a stem index the inflected forms are misses, as before.
        assert_eq!(extract_key("кора магнолии лекарственной", &find), None);
        assert_eq!(extract_key_with("кора магнолии лекарственной", &find, None), None);
        // Nothing at all.
        assert_eq!(extract_key_with("What is the capital of France?", &find, Some(&find_stem)), None);
        assert_eq!(extract_key_with("", &find, Some(&find_stem)), None);
        // A stem two entries share is dropped (ambiguous), the rest stay.
        let (idx, find_stem) = stem_finder(&[("пихта", 7), ("пихты", 3), ("ромашка", 1), ("ромашки", 1)]);
        assert_eq!((idx.len(), idx.ambiguous, idx.keys_in), (1, 1, 4));
        assert_eq!(find_stem(key_hash("пихт")), None);
        assert_eq!(find_stem(key_hash("ромашк")), Some(1));
        assert!(!idx.is_empty() && StemIndex::default().is_empty());
    }

    #[test]
    fn capitalised_binomial_anywhere_is_tried_before_the_ngrams() {
        let find = finder(&[
            ("Abies balsamea", 0),
            ("Strychnos nux-vomica", 3),
            ("пихта", 7),
            ("balsam fir", 8),
        ]);
        let h = extract_key("Чем полезна Abies balsamea?", &find).unwrap();
        assert_eq!((h.entry, h.source, h.via), (0, KeySource::Binomial, MatchVia::Exact));
        assert_eq!((h.key.as_str(), h.words), ("abies balsamea", 2));
        let h = extract_key("Is Strychnos nux-vomica toxic?", &find).unwrap();
        assert_eq!((h.entry, h.key.as_str(), h.words), (3, "strychnos nux vomica", 3));
        // Before the n-grams: the binomial beats the Cyrillic name that
        // comes first in the text; the parenthesised path still comes
        // before both.
        let h = extract_key("Пихта, то есть Abies balsamea, — семейство?", &find).unwrap();
        assert_eq!((h.entry, h.source), (0, KeySource::Binomial));
        let h = extract_key("Пихта (Abies balsamea) — семейство?", &find).unwrap();
        assert_eq!((h.entry, h.source), (0, KeySource::Parenthesised));
        // A genus alone is never a key candidate; a lowercase pair is an
        // ordinary n-gram; an English capitalised pair is harmless.
        assert_eq!(extract_key("What plant is Abies?", &find), None);
        let h = extract_key("what is abies balsamea", &find).unwrap();
        assert_eq!(h.source, KeySource::Ngram);
        let h = extract_key("Balsam fir — what is it?", &find).unwrap();
        assert_eq!((h.entry, h.source), (8, KeySource::Binomial));
        assert_eq!(
            capitalised_binomials("Tell me about Abies balsamea and Pinus sylvestris (Pínus)."),
            vec!["Abies balsamea", "Pinus sylvestris"],
            "`me` is too short for an epithet; a lowercase genus is none"
        );
        assert_eq!(capitalised_binomials("Strychnos nux-vomica) L."), vec!["Strychnos nux-vomica"]);
        assert_eq!(capitalised_binomials("St. John's wort; ABIES balsamea; Abies B."), Vec::<String>::new());
        assert_eq!(capitalised_binomials("Пихта бальзамическая"), Vec::<String>::new());
        assert_eq!(capitalised_binomials("Abies -balsamea- x"), vec!["Abies balsamea"]);
    }

    #[test]
    fn policy_parses_the_two_values_and_defaults_to_router_and_key() {
        assert_eq!(LookupPolicy::default(), LookupPolicy::RouterAndKey);
        assert_eq!(LookupPolicy::parse("router_and_key").unwrap(), LookupPolicy::RouterAndKey);
        assert_eq!(LookupPolicy::parse("key_first").unwrap(), LookupPolicy::KeyFirst);
        assert!(LookupPolicy::parse("Key_First").unwrap_err().contains("router_and_key | key_first"));
        assert!(LookupPolicy::parse("").is_err());
        let mut info = LookupInfo {
            entries: 1,
            keys: 1,
            key_norm: cortiq_core::knowledge::KEY_NORM.into(),
            langs: vec!["ru".into()],
            fields: Vec::new(),
            policy: None,
        };
        assert_eq!(LookupPolicy::of(&info), LookupPolicy::RouterAndKey);
        assert!(LookupPolicy::is_known(&info));
        info.policy = Some("key_first".into());
        assert_eq!(LookupPolicy::of(&info), LookupPolicy::KeyFirst);
        // A value this reader does not know (a newer writer, a hand edit):
        // the conservative router_and_key, never an error (review KF-6).
        for unknown in ["key_only", "key_first ", "KEY_FIRST", ""] {
            info.policy = Some(unknown.into());
            assert_eq!(LookupPolicy::of(&info), LookupPolicy::RouterAndKey, "{unknown:?}");
            assert!(!LookupPolicy::is_known(&info), "{unknown:?}");
        }
        assert_eq!(LookupPolicy::KeyFirst.label(), "key_first");
        assert_eq!(LookupPolicy::RouterAndKey.label(), "router_and_key");
        assert_eq!(DecidedBy::default().label(), "router");
        assert_eq!(DecidedBy::KeyFirst.label(), "key_first");
    }

    #[test]
    fn strong_keys_are_two_words_or_a_binomial_and_one_word_keys_never_are() {
        let keys = [
            ("ромашка аптечная", 0u32),
            ("ромашки аптечной", 0),
            ("Matricaria chamomilla", 0),
            ("chamomile", 0),
            ("календула", 1),
            ("календулы", 1),
            ("Calendula officinalis", 1),
            ("calendula", 1),
            ("pot marigold", 1),
            ("чай", 2),
            ("мята", 3),
            ("мята перечная", 3),
            ("магнолия лекарственная", 4),
            ("Strychnos nux-vomica", 5),
        ];
        let find = finder(&keys);
        let (_, find_stem) = stem_finder(&keys);
        let strong = |m: &str| extract_strong_key_with(m, &find, Some(&find_stem));
        let any = |m: &str| extract_key_with(m, &find, Some(&find_stem));
        // An exact n-gram of two words.
        let h = strong("Какие лечебные свойства у ромашки аптечной?").unwrap();
        assert_eq!((h.entry, h.words, h.via, h.source), (0, 2, MatchVia::Exact, KeySource::Ngram));
        assert!(is_strong_key(&h));
        let h = strong("Tell me about pot marigold tea").unwrap();
        assert_eq!((h.entry, h.words), (1, 2));
        // A stem n-gram of two words.
        let h = strong("Кора магнолии лекарственной — от чего?").unwrap();
        assert_eq!((h.entry, h.words, h.via), (4, 2, MatchVia::Stem));
        assert!(is_strong_key(&h));
        // A capitalised binomial anywhere, a parenthesised one, a
        // hyphenated epithet (three normalised words).
        let h = strong("What family is Matricaria chamomilla in?").unwrap();
        assert_eq!((h.entry, h.source, h.words), (0, KeySource::Binomial, 2));
        let h = strong("Ноготки (Calendula officinalis): семейство?").unwrap();
        assert_eq!((h.entry, h.source, h.words), (1, KeySource::Parenthesised, 2));
        let h = strong("Is Strychnos nux-vomica toxic?").unwrap();
        assert_eq!((h.entry, h.words), (5, 3));
        // One-word keys are never strong — exact, stemmed or in
        // parentheses — though the ordinary search still finds them.
        for m in [
            "Какое семейство у календулы?",
            "Хочу чай с мятой",
            "Налей чай",
            "мята",
            "What is calendula?",
            "A tea of chamomile, please",
            "Ноготки (Calendula) — что это?",
            "What is the capital of France?",
            "",
        ] {
            assert_eq!(strong(m), None, "{m:?}");
        }
        let h = any("Какое семейство у календулы?").unwrap();
        assert!(!is_strong_key(&h) && h.words == 1);
        let h = any("Ноготки (Calendula) — что это?").unwrap();
        assert!(!is_strong_key(&h));
        assert_eq!(any("Налей чай").unwrap().entry, 2);
        // A one-word exact key does not hide a two-word key of the same
        // message — in the strong search AND in the ordinary one (one rule
        // for the router's path and key_first's, review KF-3).
        for m in ["чай из ромашки аптечная", "Как заварить чай из ромашки аптечная?"] {
            let h = strong(m).unwrap();
            assert_eq!((h.entry, h.words, h.via), (0, 2, MatchVia::Stem), "{m}");
            assert_eq!(any(m), Some(h), "{m}");
        }
        let h = any("чай с мятой перечная").unwrap();
        assert_eq!((h.entry, h.via), (3, MatchVia::Stem), "not the exact `чай`");
        // The longest strong key wins, as in the ordinary search.
        let h = strong("мята перечная и ромашка аптечная").unwrap();
        assert_eq!((h.entry, h.key.as_str()), (3, "мята перечная"), "ties → first occurrence");
    }

    #[test]
    fn strong_keys_need_two_real_words_without_digits() {
        // Review KF-5: a digit, a one- or two-letter word does not make a
        // name — `STS-135`, `PTI-2`, `5F-PB-22`, `THC-B` split into
        // several normalised "words" but none is strong.
        for k in ["sts 135", "pti 2", "a 41988", "5f pb 22", "thc b", "чай", "b 12 vitamin"] {
            assert!(!is_strong_key_text(k), "{k}");
        }
        for k in ["st john s wort", "pot marigold", "ромашки аптечной", "strychnos nux vomica", "five finger"] {
            assert!(is_strong_key_text(k), "{k}");
        }
        let keys = [("STS-135", 0u32), ("PTI-2", 1), ("5F-PB-22", 2), ("THC-B", 3), ("St. John's wort", 4), ("мята", 5)];
        let find = finder(&keys);
        let (_, find_stem) = stem_finder(&keys);
        let msg = "Tell me about the STS-135 mission of Space Shuttle Atlantis";
        assert_eq!(extract_strong_key_with(msg, &find, Some(&find_stem)), None);
        let h = extract_key_with(msg, &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.key.as_str(), h.words, h.strong), (0, "sts 135", 2, false));
        for m in ["Is PTI-2 legal?", "5F-PB-22 effects", "What is THC-B?"] {
            assert_eq!(extract_strong_key_with(m, &find, Some(&find_stem)), None, "{m}");
            assert!(extract_key_with(m, &find, Some(&find_stem)).is_some_and(|h| !h.strong), "{m}");
        }
        let h = extract_strong_key_with("Is St. John's wort safe?", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.words, h.strong), (4, 4, true));
        // A parenthesised group of such words is strong as well.
        let h = extract_key_with("мята (St. John's wort)", &find, Some(&find_stem)).unwrap();
        assert_eq!((h.entry, h.source), (4, KeySource::Parenthesised));
    }

    #[test]
    fn a_capitalised_pair_is_judged_by_its_words_like_an_n_gram() {
        // The G5 row `Spring vetchling (Lathyrus vernus (L.) Bernh.)` of the
        // real corpus: the common name typed first names the full entry,
        // the binomial a stub duplicate. Both are strong by their words;
        // the first in rule order wins — on every path.
        let keys = [("Lathyrus vernus", 0u32), ("Spring vetchling", 1), ("Common box", 2)];
        let find = finder(&keys);
        let m = "Which plant family does Spring vetchling (Lathyrus vernus (L.) Bernh.) belong to?";
        let h = extract_strong_key_with(m, &find, None).unwrap();
        assert_eq!((h.entry, h.source, h.strong), (1, KeySource::Binomial, true));
        assert_eq!(extract_key_with(m, &find, None), Some(h));
        // `Common box`: a general phrase that is a common name — strong
        // as a pair and as an n-gram alike (review KF-5); the stop list of
        // the builder's general probe removes it (KF-1).
        let h = extract_strong_key_with("Why is Common box cutter so popular?", &find, None).unwrap();
        assert_eq!((h.entry, h.source), (2, KeySource::Binomial));
        let h = extract_strong_key_with("why is common box cutter so popular", &find, None).unwrap();
        assert_eq!((h.entry, h.source), (2, KeySource::Ngram));
        // A key with a one-letter word is weak, typed as a pair or not.
        let find = finder(&[("Vitamin b", 3)]);
        assert_eq!(extract_strong_key_with("Is Vitamin b safe?", &find, None), None);
        let h = extract_key_with("Is Vitamin b safe?", &find, None);
        assert_eq!(h.map(|k| (k.entry, k.strong, k.source)), Some((3, false, KeySource::Ngram)));
    }

    #[test]
    fn extraction_stays_linear_in_the_message() {
        use std::cell::Cell;
        let exact_calls = Cell::new(0usize);
        let stem_calls = Cell::new(0usize);
        let find = |_h: u64| {
            exact_calls.set(exact_calls.get() + 1);
            None
        };
        let find_stem = |_h: u64| {
            stem_calls.set(stem_calls.get() + 1);
            None
        };
        let words = 300usize;
        let msg: String = (0..words)
            .map(|i| if i % 3 == 0 { "Magnolia" } else { "лекарственной" })
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(extract_key_with(&msg, &find, Some(&find_stem)), None);
        // ≤ MAX_NGRAM hashes per word for the n-grams, plus one per
        // capitalised pair: linear, and the stem pass costs the same again.
        assert!(exact_calls.get() <= (MAX_NGRAM + 1) * words, "{}", exact_calls.get());
        assert!(stem_calls.get() <= MAX_NGRAM * words, "{}", stem_calls.get());
        assert!(stem_calls.get() >= words, "the stem pass ran: {}", stem_calls.get());
    }

    #[test]
    fn key_texts_are_recovered_from_the_cards() {
        // A table as the file holds it: sorted hashes, entries, the slot
        // blob. Keys 0 and 1 name entry 0; entry 1's card mentions entry
        // 0's plant; key `mint` is spelled by no card.
        let keys = [("ромашка аптечная", 0u32), ("chamomile", 0), ("шалфей", 1), ("mint", 2)];
        let mut pairs: Vec<(u64, u32)> = keys.iter().map(|(k, e)| (key_hash(k), *e)).collect();
        pairs.sort_unstable();
        let hashes: Vec<u64> = pairs.iter().map(|p| p.0).collect();
        let entry_of: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        let slots = [
            r#"{"card":"Ромашка аптечная — однолетник.","fields":{"uses":"чай"}}"#,
            r#"{"card":"Chamomile is an annual.","fields":{}}"#,
            r#"{"card":"Шалфей — полукустарник; сочетают с ромашкой аптечной.","fields":{}}"#,
            r#"{"card":"","fields":{}}"#,
            r#"{"card":"","fields":{}}"#,
            r#"{"card":"","fields":{}}"#,
        ];
        let mut blob = Vec::new();
        let mut offsets = vec![0u64];
        for s in slots {
            blob.extend_from_slice(s.as_bytes());
            offsets.push(blob.len() as u64);
        }
        let mut got = recover_key_texts(&hashes, &entry_of, &blob, &offsets);
        got.sort();
        assert_eq!(
            got,
            vec![
                ("chamomile".to_string(), 0),
                ("ромашка аптечная".to_string(), 0),
                ("шалфей".to_string(), 1),
            ]
        );
        let idx = StemIndex::from_keys(got.iter().map(|(k, e)| (k.as_str(), *e)));
        assert_eq!(idx.find(key_hash("ромашк аптечн")), Some(0));
        assert_eq!(idx.find(key_hash("шалф")), Some(1));
        assert_eq!(idx.find(key_hash("mint")), None, "spelled by no card: no stem");
        // A broken offset or a non-UTF-8 slot is skipped, not a panic.
        let bad = [0u64, 5, 3, blob.len() as u64 + 10];
        let _ = recover_key_texts(&hashes, &entry_of, &blob, &bad);
        assert_eq!(recover_key_texts(&hashes, &entry_of, &[0xFF, 0xFE], &[0, 2]), vec![]);
    }
}
