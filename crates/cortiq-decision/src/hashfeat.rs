//! Lexical hashing features of the original router: a port of cortiq-router
//! `src/distill.rs::feat` (the `φ_H` signal of the v3 spec).
//!
//! * `original::fnv1a` and `original::feat` are VERBATIM copies of the original file
//!   (the text between the `---- verbatim` markers is a byte-exact substring of
//!   `distill.rs`; `tests::original_sources_unchanged_and_copies_verbatim` proves it
//!   against the file and records its sha256 in [`DISTILL_RS_SHA256`]).
//! * [`dense`] is the form the v2 feature contract ships: the same tokens, weights,
//!   hash, index and sign, accumulated in the same order into a dense `dim` vector,
//!   then L2 with the sum of squares taken in ascending index order. The original
//!   sums the squares in `HashMap` iteration order (random per process), so its
//!   normalised values are not reproducible to the bit even against itself; the
//!   per-index accumulations are, and `tests::parity_with_the_original` checks
//!   both facts on > 600 strings (Cyrillic, CJK, emoji, punctuation, empty, 1 char).
//!
//! Definition (`HASHING_DEFINITION` in contract_v2 repeats it for the record):
//! `lower = text.to_lowercase()`; words = `lower.split(|c| !c.is_alphanumeric())`
//! without empties, weight 1.0; word bigrams `"{a}_{b}"`, weight 0.7; byte
//! 3-, 4- and 5-grams of `lower`'s UTF-8, weight 0.5; `h = fnv1a64(token, seed 0)`;
//! index `h % dim`; sign `+1` when bit 33 of `h` is set, else `-1`; the signed weight
//! is added (f32) to the index in token order: words, bigrams, 3-grams, 4-grams,
//! 5-grams. Then `norm = sqrt(Σ x²)` (f32, ascending index); when `norm > 1e-9`
//! every value is multiplied by `1.0 / norm` (f32); a zero vector stays zero.

/// sha256 of the original sources at the time of the port (recorded, and verified
/// by the test when the files are present).
pub const DISTILL_RS_SHA256: &str = "a751e51a6dbae0630e9d476046c3df1eb0500dc3d60cb02f3ebaac71e687e035";
#[cfg_attr(not(test), allow(dead_code))]
pub const EMBED_RS_SHA256: &str = "96ac6a89ffd1d340efb6e847ceae887199a0f6b540c8b5ef934d922d19fd9faf";
#[cfg_attr(not(test), allow(dead_code))]
pub const ORIGINAL_SRC_DIR: &str = "/Users/oleg/Documents/cortiq-bot/cortiq-router/src";
pub const ORIGINAL_FILE: &str = "cortiq-router/src/distill.rs";
/// The constants of the verbatim `feat` (the record documents them; `parse` refuses
/// any other values because nothing else is implemented).
pub const WORD_WEIGHT: f64 = 1.0;
pub const BIGRAM_WEIGHT: f64 = 0.7;
pub const NGRAM_WEIGHT: f64 = 0.5;
pub const NGRAMS: [u64; 3] = [3, 4, 5];
pub const SEED: u64 = 0;
pub const SIGN_BIT: u64 = 33;

// The verbatim copy is dead in the binary that never emits sparse pairs; `dense`
// lives beside it because the original keeps `fnv1a` private.
#[allow(dead_code)]
mod original {
    // ---- verbatim: cortiq-router src/distill.rs (fnv1a + feat) ----
    #[inline]
    fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
        let mut h = 0xcbf29ce484222325u64 ^ seed;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Sparse signed hashed features of `text`: word tokens + char 3/4-grams.
    /// Returns `(index, value)` pairs, L2-normalized.
    pub fn feat(text: &str, fdim: usize) -> Vec<(u32, f32)> {
        use std::collections::HashMap;
        let mut acc: HashMap<u32, f32> = HashMap::new();
        let mut add = |tok: &[u8], w: f32| {
            let h = fnv1a(tok, 0);
            let idx = (h % fdim as u64) as u32;
            let sign = if (h >> 33) & 1 == 1 { 1.0 } else { -1.0 };
            *acc.entry(idx).or_insert(0.0) += sign * w;
        };
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        for tok in &words {
            add(tok.as_bytes(), 1.0);
        }
        // word bigrams (light syntactic signal)
        for pair in words.windows(2) {
            add(format!("{}_{}", pair[0], pair[1]).as_bytes(), 0.7);
        }
        // character 3/4/5-grams (morphology, symbols, robustness)
        let b = lower.as_bytes();
        for n in [3usize, 4, 5] {
            if b.len() >= n {
                for w in b.windows(n) {
                    add(w, 0.5);
                }
            }
        }
        let mut v: Vec<(u32, f32)> = acc.into_iter().filter(|&(_, x)| x != 0.0).collect();
        let norm: f32 = v.iter().map(|&(_, x)| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            let inv = 1.0 / norm;
            for (_, x) in v.iter_mut() {
                *x *= inv;
            }
        }
        v
    }
    // ---- end verbatim ----

    /// Pre-normalisation accumulator of `feat` as a dense vector: per index, the same
    /// f32 additions in the same order as `feat`'s `HashMap` entry.
    pub fn accumulate(text: &str, dim: usize) -> Vec<f32> {
        assert!(dim > 0, "hashing dim must be positive");
        let mut v = vec![0.0f32; dim];
        let mut add = |tok: &[u8], w: f32| {
            let h = fnv1a(tok, 0);
            let idx = (h % dim as u64) as usize;
            let sign = if (h >> 33) & 1 == 1 { 1.0 } else { -1.0 };
            v[idx] += sign * w;
        };
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        for tok in &words {
            add(tok.as_bytes(), 1.0);
        }
        for pair in words.windows(2) {
            add(format!("{}_{}", pair[0], pair[1]).as_bytes(), 0.7);
        }
        let b = lower.as_bytes();
        for n in [3usize, 4, 5] {
            if b.len() >= n {
                for w in b.windows(n) {
                    add(w, 0.5);
                }
            }
        }
        v
    }

    /// Dense, deterministic `feat`: `accumulate`, then the original's normalisation
    /// with the sum of squares in ascending index order (zeros add exactly nothing).
    pub fn dense(text: &str, dim: usize) -> Vec<f32> {
        let mut v = accumulate(text, dim);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            let inv = 1.0 / norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
        v
    }
}
pub use original::dense;
// The sparse original is emitted only by the `hashfeat` CLI (unused in the other binary).
#[allow(unused_imports)]
pub use original::feat;

// ==== cortiq-hashfeat-v1 contract (decision-v4 spec §1.5, §2.6) ====
// Added by the release crate. Everything outside this section is byte-identical
// to tools/cortiq-decision-embryo/src/hashfeat.rs (sha256 [`EMBRYO_COPY_SHA256`]);
// tests/hashfeat.rs removes this section and checks that sha256. The runtime calls
// only [`dense`] with [`DIM`]; the sparse `feat` normalises in `HashMap` order and
// is kept only as the verbatim record of the original.

/// Name of the hashing contract recorded in a decision file (spec §2.6).
pub const KIND: &str = "cortiq-hashfeat-v1";
/// The hashing dimension of the PH signal (`x = [φ_P ; 0.5·φ_H]`, φ_H = `dense(text, DIM)`).
pub const DIM: usize = 4096;
/// Hash function of the contract.
pub const HASH: &str = "fnv1a64";
/// `HASHING_DEFINITION`, verbatim from tools/cortiq-decision-embryo/src/contract_v2.rs:44-46.
pub const DEFINITION: &str = "cortiq-router distill.rs::feat: lower = text.to_lowercase(); words = lower.split(|c| !char::is_alphanumeric(c)) without empties, weight 1.0; word bigrams \"{a}_{b}\" weight 0.7; byte 3/4/5-grams of lower's UTF-8 weight 0.5; h = fnv1a64(token bytes, seed 0); index = h % dim; sign = +1 if bit 33 of h else -1; signed weights added (f32) in token order: words, bigrams, 3-grams, 4-grams, 5-grams; dense vector of dim";
/// `HASHING_NORMALIZATION` of contract_v2.rs: the normalisation [`dense`] applies.
pub const NORMALIZATION: &str = "l2-f32-index-order-mul-inv;zero-stays-zero";
/// sha256 of tools/cortiq-decision-embryo/src/hashfeat.rs, the file this one copies.
pub const EMBRYO_COPY_SHA256: &str = "b34b49dde233f807dce7a5914ac06f239be09a749f91588d2019cc8e944d0c4b";

/// The 32 golden texts of the contract (spec §2.6): English, Cyrillic, CJK, Hangul,
/// Greek sigma, Turkish dotted İ, Devanagari, circled letters, emoji, NBSP, control
/// characters, empty and one-character strings. A loader recomputes [`dense`] of each
/// and compares the sha256 of its little-endian f32 bytes with [`GOLDEN_DENSE_SHA256`].
pub const GOLDEN_TEXTS: [&str; 32] = [
    "I still have not received my new card",
    "My statement has not shown my refund.",
    "card payment declined",
    "How do I top up my account with a cheque?",
    "PIN blocked!!!",
    "what's the exchange rate for EUR -> USD?",
    "",
    "a",
    "Привет, мир! Как мне пополнить счёт?",
    "ЩЁЛКНИТЕ ЗДЕСЬ",
    "日本語のテキスト 口座の残高を教えて",
    "中文测试 银行 卡",
    "한국어 문장",
    "ΣΊΣΥΦΟΣ ΟΔΟΣ σ ς",
    "İstanbul ıi İI",
    "हिन्दी भाषा",
    "Ⓐⓑ circled ①②③",
    "emoji 😀 test 🎉🎉",
    "🇬🇧 flag",
    "café crème naïve façade",
    "Straße STRASSE ẞ ﬁ",
    "€50 £20 … ’quoted’",
    "x^2 + y^2 = z^2",
    "tab\tnew\nline\r\u{0}\u{7}end",
    "\u{a0}nbsp\u{a0} and\u{2003}em space",
    "MiXeD cAsE 12345 3.14159",
    "email@example.com http://example.com/path?q=1&r=2",
    "very_long_identifier_with_underscores_2026",
    "Repeat repeat REPEAT repeat",
    "hyphen-ated words, double  space",
    "½ ⅷ ² ³ Ⅻ ǅ",
    "The quick brown fox jumps over the lazy dog while my transfer is still pending and I would like to know why it takes so long",
];

/// sha256 of `dense(GOLDEN_TEXTS[i], DIM)` as little-endian f32 bytes (16,384 bytes each).
pub const GOLDEN_DENSE_SHA256: [&str; 32] = [
    "dcec52e17ee4a0f44f0577452d81e035abff4915eeac3d26bef3d3b19e299480",
    "0a5e8559cdecff463f42f17ac517613c103ae3865fad123e7d3f37c6d18c5c3d",
    "756078d4bf2ce9e791c336927b148f4a8f9f91cacc6ddf6c9eb8395aa655f3ad",
    "7d83f843522bfe97007c67c0486634f5d1095affc1596a60bb2fd6e45779b803",
    "52133cc0efe2b8ed857ef914e590a387f3f724dc0946eb93f8e5ad087795174e",
    "80fd6826f2b5671e160a1c1d39eb1172cb7be517c6b0c4b8abc868b17c46d5e9",
    "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
    "8d026661e4f19786c5650deb10f11d0677c406f59bfc82c6e54c161fcf59fc69",
    "090bf95e401db166297581a4109ed8f64282f80633614eecb3a70e497676990e",
    "bc43883ad999ba74ccc549c725663f099a540cb3ab1f10706f4d19e8fe9c317d",
    "586cffa3daa4acd86d8c8d42345c1daad9c0d16a9a018d370c86b428ad1e2350",
    "92bcd526c5344b6be980346dc25dce4496fd9fc99076cba97aada369bb926506",
    "1b123baf11c2c40bbd6ebd821a41cefe791dd623c1a21d94cb43f845c6af3cd9",
    "18deac2058df925d234973d53f722898df2d8dd5eae52ea86a81bd393ccadb3f",
    "6489119fc352279045c524f2dcc4739311b8a787fadc781bc85bffe335087b70",
    "2d637caa4f57a95a56ab94e4d4a0e96e6ac1e916381da500a92802131a81d616",
    "bcc3395d838c1c66cc7d04e6b42deee0c3b9c0d92f3a000e671aee183f06b0f3",
    "58ce6c67d413bda085d2ed7b30a59505731f2315807f9b481ca86fd81ff21542",
    "73cf5c0ef7f11e74e35f89b629f96ea0108cb942b3d1aab7721aed95b72be845",
    "c6346027057c1a92fd2facebe3c9d81477903a0cc6a8e7f297e62198eae4618a",
    "70404ef793a4377685a775dfa91dbff9c1045a14c0f8c799ffce6f5cfb2044ab",
    "9c8f29fb6e425e26232095479182212c0378be29b734ffad36f397e3d351d6c7",
    "a27778e340a080c34f6cf52056eda53b4f7cb13110202cc9a3c26b16f1fe750c",
    "261cf001b7182456efc7c77c2fcd7112703a9e44d1fab16e9ff4cc02c9d672b1",
    "6b3e2995a593f395d0dc6ba04dde77f065a9acfad0e04939788173b0741d63ba",
    "0d970130d2e067dca0f338ffa521a7dd6bc30f2f877cdc22dfcad453749c1a71",
    "5631739f6d3785eb196ba9362fc252159531f4f091546df53850edb4ce18954d",
    "ba20f730ddec63497d2377f9fe9518d1986190bbe8b3f15f2fc76668c7d2d837",
    "4d5f7e09bab3d50edc6e828575bf626a3f8b61940d76d4900ec84b167437e72c",
    "00513a780f2464e538d52a4d9dd9a4e5cfff9df29760ff293b5fc06115281f3d",
    "f83e8e7d7ec92fdfe883d22bd4a69ad4c8151935f25798abaa6d6ca7c130cceb",
    "e136b0f6d9c649cfb994899783218dfd83afb5535f3871b0e5782d2487ce4ab3",
];

/// Little-endian bytes of `dense(text, dim)`.
pub fn dense_f32le(text: &str, dim: usize) -> Vec<u8> {
    dense(text, dim).iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// sha256 (lowercase hex) of [`dense_f32le`].
pub fn dense_f32le_sha256(text: &str, dim: usize) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(dense_f32le(text, dim)))
}

/// Indices of the golden texts whose [`dense`] differs from the pinned sha256 with
/// this build (empty = the contract holds; spec §2.6 "hashing contract mismatch").
pub fn golden_mismatches() -> Vec<usize> {
    (0..GOLDEN_TEXTS.len())
        .filter(|&i| dense_f32le_sha256(GOLDEN_TEXTS[i], DIM) != GOLDEN_DENSE_SHA256[i])
        .collect()
}

/// `core::char::UNICODE_VERSION` of the compiler that built this crate, as "major.minor.patch":
/// `char::to_lowercase` and `char::is_alphanumeric` follow it.
pub fn unicode_version() -> String {
    let (a, b, c) = char::UNICODE_VERSION;
    format!("{a}.{b}.{c}")
}

/// The hashing record of a decision file (spec §2.6), with the golden texts and the
/// sha256 values recomputed by this build (they equal [`GOLDEN_DENSE_SHA256`] exactly
/// when [`golden_mismatches`] is empty).
pub fn contract_record() -> serde_json::Value {
    let golden: Vec<serde_json::Value> = GOLDEN_TEXTS
        .iter()
        .map(|t| serde_json::json!({"text": t, "dense_f32le_sha256": dense_f32le_sha256(t, DIM)}))
        .collect();
    serde_json::json!({
        "kind": KIND,
        "dim": DIM,
        "definition": DEFINITION,
        "normalization": NORMALIZATION,
        "tokens": {
            "lowercase": "rust char::to_lowercase",
            "word_split": "!char::is_alphanumeric, empties dropped",
            "word_weight": WORD_WEIGHT,
            "bigram": "{a}_{b}",
            "bigram_weight": BIGRAM_WEIGHT,
            "byte_ngrams": NGRAMS,
            "byte_ngram_weight": NGRAM_WEIGHT
        },
        "hash": HASH,
        "seed": SEED,
        "index": "h % dim",
        "sign": "+1 if bit 33 set else -1",
        "accumulate": "f32, hashfeat.rs::accumulate token order",
        "source": {"file": ORIGINAL_FILE, "sha256": DISTILL_RS_SHA256, "embed_rs_sha256": EMBED_RS_SHA256},
        "unicode_version": unicode_version(),
        "golden": golden
    })
}
// ==== end cortiq-hashfeat-v1 contract ====

#[cfg(test)]
mod tests {
    use super::original::accumulate;
    use super::*;
    use sha2::{Digest, Sha256};

    /// A second, independent verbatim copy of the original: the reference the port
    /// is compared with (also checked as a substring of the file when present).
    mod reference {
        #![allow(dead_code)]
        // ---- verbatim: cortiq-router src/distill.rs (fnv1a + feat) ----
        #[inline]
        fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
            let mut h = 0xcbf29ce484222325u64 ^ seed;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        }

        /// Sparse signed hashed features of `text`: word tokens + char 3/4-grams.
        /// Returns `(index, value)` pairs, L2-normalized.
        pub fn feat(text: &str, fdim: usize) -> Vec<(u32, f32)> {
            use std::collections::HashMap;
            let mut acc: HashMap<u32, f32> = HashMap::new();
            let mut add = |tok: &[u8], w: f32| {
                let h = fnv1a(tok, 0);
                let idx = (h % fdim as u64) as u32;
                let sign = if (h >> 33) & 1 == 1 { 1.0 } else { -1.0 };
                *acc.entry(idx).or_insert(0.0) += sign * w;
            };
            let lower = text.to_lowercase();
            let words: Vec<&str> = lower
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| !t.is_empty())
                .collect();
            for tok in &words {
                add(tok.as_bytes(), 1.0);
            }
            // word bigrams (light syntactic signal)
            for pair in words.windows(2) {
                add(format!("{}_{}", pair[0], pair[1]).as_bytes(), 0.7);
            }
            // character 3/4/5-grams (morphology, symbols, robustness)
            let b = lower.as_bytes();
            for n in [3usize, 4, 5] {
                if b.len() >= n {
                    for w in b.windows(n) {
                        add(w, 0.5);
                    }
                }
            }
            let mut v: Vec<(u32, f32)> = acc.into_iter().filter(|&(_, x)| x != 0.0).collect();
            let norm: f32 = v.iter().map(|&(_, x)| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                let inv = 1.0 / norm;
                for (_, x) in v.iter_mut() {
                    *x *= inv;
                }
            }
            v
        }
        // ---- end verbatim ----
    }

    const BEGIN: &str = "// ---- verbatim: cortiq-router src/distill.rs (fnv1a + feat) ----\n";
    const END: &str = "// ---- end verbatim ----";
    /// Every verbatim span of this file (module and test copies), markers excluded,
    /// with the indentation of the enclosing module removed.
    fn verbatim_spans() -> Vec<String> {
        let me = include_str!("hashfeat.rs");
        let mut spans = Vec::new();
        let mut rest = me;
        while let Some(b) = rest.find(BEGIN) {
            let after = &rest[b + BEGIN.len()..];
            let e = after.find(END).expect("unterminated verbatim span");
            let mut lines: Vec<&str> = after[..e].split('\n').collect();
            // the last element is the END marker's own indentation
            assert!(lines.pop().unwrap().trim().is_empty());
            let indent = lines[0].len() - lines[0].trim_start_matches(' ').len();
            let span: Vec<&str> = lines
                .iter()
                .map(|l| {
                    assert!(
                        l.is_empty() || l.starts_with(&" ".repeat(indent)),
                        "verbatim line is not indented uniformly: {l:?}"
                    );
                    if l.is_empty() { "" } else { &l[indent..] }
                })
                .collect();
            spans.push(span.join("\n"));
            rest = &after[e + END.len()..];
        }
        spans
    }

    #[test]
    fn original_sources_unchanged_and_copies_verbatim() {
        let spans = verbatim_spans();
        assert_eq!(spans.len(), 2, "module copy and test reference copy");
        assert_eq!(spans[0], spans[1], "the two copies must be identical text");
        assert!(spans[0].contains("pub fn feat(text: &str, fdim: usize) -> Vec<(u32, f32)>"));
        assert!(spans[0].contains("fn fnv1a(bytes: &[u8], seed: u64) -> u64"));
        let dir = std::path::Path::new(ORIGINAL_SRC_DIR);
        let distill = dir.join("distill.rs");
        if !distill.exists() {
            eprintln!("original router source absent at {}; sha256/substring checks skipped", distill.display());
            return;
        }
        let distill_src = std::fs::read(&distill).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&distill_src)), DISTILL_RS_SHA256, "distill.rs changed since the port");
        let embed_src = std::fs::read(dir.join("embed.rs")).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&embed_src)), EMBED_RS_SHA256, "embed.rs changed since the port");
        let text = String::from_utf8(distill_src).unwrap();
        assert!(text.contains(&spans[0]), "the copied fnv1a + feat is not a byte-exact substring of distill.rs");
    }

    #[test]
    fn fnv1a_known_vectors() {
        // FNV-1a 64 published test vectors (seed 0 = the plain offset basis).
        let d = dense("", 8);
        assert!(d.iter().all(|&x| x == 0.0), "empty text is the zero vector");
        assert_eq!(reference::feat("", 4096), Vec::<(u32, f32)>::new());
        let one = accumulate("a", 4096);
        // fnv1a64("a") = 0xaf63dc4c8601ec8c: index = h % 4096 = 0xc8c, bit 33 of h is 0 -> -1.0
        let h: u64 = 0xaf63dc4c8601ec8c;
        assert_eq!((h >> 33) & 1, 0);
        assert_eq!(one[(h % 4096) as usize], -1.0);
        assert_eq!(one.iter().filter(|&&x| x != 0.0).count(), 1);
        assert_eq!(dense("a", 4096)[(h % 4096) as usize], -1.0);
    }

    fn corpus() -> Vec<String> {
        let parts: [&str; 48] = [
            "My statement has not shown my refund.", "Why isn't a refund showing on my statement?",
            "card payment declined", "I need to top up", "how do I get a virtual card?", "PIN blocked!!!",
            "Привет, мир!", "Как мне пополнить счёт?", "Перевод не пришёл", "ЩЁЛКНИТЕ ЗДЕСЬ",
            "日本語のテキスト", "口座の残高を教えて", "中文测试 银行 卡", "한국어 문장",
            "emoji 😀 test 🎉🎉", "😀", "🇬🇧 flag", "café crème naïve façade", "Straße STRASSE ẞ",
            "İstanbul ıi", "ΣΊΣΥΦΟΣ ΟΔΟΣ σ", "हिन्दी भाषा", "①②③ ½ ⅷ", "€50 £20 … ’quoted’",
            "x^2 + y^2 = z^2", "a_b c-d e.f", "tab\tnew\nline", "\u{a0}nbsp\u{a0}", "ALL CAPS TEXT",
            "MiXeD cAsE", "12345", "3.14159", "!!!", "??", "..", "a", "Z", "é", "ж", "字", "_",
            " leading and trailing ", "double  space", "hyphen-ated words", "email@example.com",
            "http://example.com/path?q=1&r=2", "very_long_identifier_with_underscores_2026",
            "Repeat repeat REPEAT repeat",
        ];
        let mut out: Vec<String> = vec![String::new(), " ".into(), "\n".into(), "ab".into(), "abc".into()];
        out.extend(parts.iter().map(|s| s.to_string()));
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        while out.len() < 640 {
            let n = (next() % 12) as usize;
            let s: Vec<&str> = (0..n).map(|_| parts[(next() % parts.len() as u64) as usize]).collect();
            out.push(s.join(if next() % 2 == 0 { " " } else { ", " }));
        }
        out
    }

    #[test]
    fn parity_with_the_original() {
        let texts = corpus();
        assert!(texts.len() >= 600);
        for dim in [4096usize, 512, 64] {
            let (mut exact_rows, mut max_ulp) = (0usize, 0u32);
            for t in &texts {
                let o = reference::feat(t, dim);
                let raw = accumulate(t, dim);
                let d = dense(t, dim);
                assert_eq!(d.len(), dim);
                // (1) identical support and sign
                let support: std::collections::BTreeSet<u32> = o.iter().map(|&(i, _)| i).collect();
                assert_eq!(support.len(), o.len(), "original repeats an index");
                for (i, &x) in d.iter().enumerate() {
                    assert_eq!(support.contains(&(i as u32)), x != 0.0, "support differs at {i} for {t:?}");
                    if x != 0.0 {
                        let ox = o.iter().find(|p| p.0 == i as u32).unwrap().1;
                        assert_eq!(ox.is_sign_negative(), x.is_sign_negative(), "sign differs at {i} for {t:?}");
                    }
                }
                // (2) bit-exact: the original's values are the port's accumulator normalised
                // with the sum of squares taken in the original's (HashMap) order.
                let norm: f32 = o.iter().map(|&(i, _)| raw[i as usize] * raw[i as usize]).sum::<f32>().sqrt();
                for &(i, ox) in &o {
                    let mut want = raw[i as usize];
                    if norm > 1e-9 {
                        want *= 1.0 / norm;
                    }
                    assert_eq!(want.to_bits(), ox.to_bits(), "accumulator differs at {i} for {t:?}");
                }
                // (3) the shipped dense vector: within a few ulp (only the norm order differs)
                let mut row_exact = true;
                for &(i, ox) in &o {
                    let x = d[i as usize];
                    let ulp = (x.to_bits() as i64 - ox.to_bits() as i64).unsigned_abs() as u32;
                    if ulp != 0 {
                        row_exact = false;
                        max_ulp = max_ulp.max(ulp);
                    }
                }
                exact_rows += usize::from(row_exact);
            }
            eprintln!("dim {dim}: {}/{} rows bit-identical to the original's own normalisation, max ulp {max_ulp}", exact_rows, texts.len());
            assert!(max_ulp <= 8, "dense differs from the original by {max_ulp} ulp at dim {dim}");
        }
    }

    #[test]
    fn dense_is_deterministic_and_unit_length() {
        for t in corpus().iter().take(200) {
            let a = dense(t, 4096);
            let b = dense(t, 4096);
            assert_eq!(a, b);
            let n: f32 = a.iter().map(|x| x * x).sum::<f32>();
            assert!(n == 0.0 || (n - 1.0).abs() < 1e-5, "norm {n} for {t:?}");
        }
    }
}
