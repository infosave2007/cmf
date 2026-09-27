//! WordPiece golden cases (spec §1.3, §6.1): the release tokenizer (vocab.txt
//! of the release encoder) against ids produced offline by HF `tokenizers`
//! 0.22.2 (`tools/mk_decision_toy.py`), including the stress set (Cyrillic,
//! Greek sigma, Turkish İ, Devanagari, circled letters, emoji, NBSP, control
//! characters, special tokens in the text, words over 100 characters,
//! truncation at 512). A case may differ only if the fixture lists it under
//! `exceptions` with its reason.

use cortiq_decision::manifest::SpecialIds;
use cortiq_decision::unicode_tables;
use cortiq_decision::wordpiece::{self, WordPiece};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy")
}

fn load() -> (Value, WordPiece) {
    let dir = fixture_dir();
    let g: Value = serde_json::from_slice(
        &std::fs::read(dir.join("wordpiece_golden.json")).expect("wordpiece_golden.json"),
    )
    .unwrap();
    let vocab = std::fs::read(dir.join(g["vocab_file"].as_str().unwrap())).expect("vocab");
    assert_eq!(
        format!("{:x}", Sha256::digest(&vocab)),
        g["vocab_sha256"].as_str().unwrap(),
        "fixture vocab changed"
    );
    let ids: SpecialIds = serde_json::from_value(g["ids"].clone()).unwrap();
    let wp = WordPiece::new(
        &vocab,
        g["unk"].as_str().unwrap(),
        g["prefix"].as_str().unwrap(),
        g["max_input_chars_per_word"].as_u64().unwrap() as usize,
        &ids,
        g["max_length"].as_u64().unwrap() as usize,
    )
    .unwrap();
    (g, wp)
}

#[test]
fn hf_golden_cases_are_equal() {
    let (g, wp) = load();
    assert_eq!(
        g["tokenizers"], "0.22.2",
        "golden ids come from HF tokenizers 0.22.2"
    );
    let cases = g["cases"].as_array().unwrap();
    assert!(
        cases.len() >= 200,
        "at least 200 golden cases, found {}",
        cases.len()
    );
    let exceptions = g["exceptions"].as_object().unwrap();
    let (mut equal, mut listed, mut stress) = (0usize, 0usize, 0usize);
    let mut failures = Vec::new();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if c["kind"] == "stress" {
            stress += 1;
        }
        let got = wp.encode(text);
        if got == want {
            equal += 1;
        } else if exceptions.contains_key(text) {
            listed += 1;
        } else {
            failures.push(format!(
                "{:?} ({}): native {:?} vs HF {:?}",
                text,
                c["kind"],
                wp.tokens(&got),
                wp.tokens(&want)
            ));
        }
    }
    assert!(stress >= 50, "the stress set has {stress} cases");
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from HF:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!(
        "wordpiece golden: {equal}/{} equal, {listed} listed exceptions, {stress} stress cases",
        cases.len()
    );
}

#[test]
fn truncation_keeps_512_with_specials() {
    let (_, wp) = load();
    let long = "card ".repeat(700);
    let ids = wp.encode(&long);
    assert_eq!(ids.len(), 512);
    assert_eq!(ids[0], 101);
    assert_eq!(ids[511], 102);
    assert_eq!(wp.encode("").as_slice(), [101, 102]);
}

#[test]
fn special_tokens_are_cut_from_the_raw_text() {
    let (_, wp) = load();
    let specials: Vec<(&str, u32)> = wp
        .specials()
        .iter()
        .map(|(s, i)| (s.as_str(), *i))
        .collect();
    assert_eq!(specials.len(), 5);
    for (s, id) in [
        ("[PAD]", 0),
        ("[UNK]", 100),
        ("[CLS]", 101),
        ("[SEP]", 102),
        ("[MASK]", 103),
    ] {
        assert!(specials.contains(&(s, id)), "{s} → {id}");
        assert_eq!(wp.encode(s), vec![101, id, 102]);
    }
    // Case-sensitive: the lowercase form is ordinary text.
    assert_eq!(
        wp.tokens(&wp.encode("[mask]")),
        ["[CLS]", "[", "mask", "]", "[SEP]"]
    );
}

#[test]
fn unicode_tables_follow_the_hf_crates() {
    // unicode_categories 0.1.1 is Unicode 9.0: later characters are unassigned.
    assert_eq!(unicode_tables::UNICODE_VERSION, (9, 0, 0));
    assert!(unicode_tables::is_assigned('a') && unicode_tables::is_assigned('\u{17000}'));
    assert!(!unicode_tables::is_assigned('\u{1F972}')); // 🥲, Unicode 13
    assert!(!unicode_tables::is_assigned('\u{0378}'));
    // is_other = Cc | Cf | Co (no Cn).
    assert!(unicode_tables::is_other('\u{7}') && unicode_tables::is_other('\u{200B}'));
    assert!(unicode_tables::is_other('\u{E000}') && unicode_tables::is_other('\u{10FFFD}'));
    assert!(!unicode_tables::is_other('\u{0378}'));
    // Punctuation (P*), not symbols.
    assert!(unicode_tables::is_punctuation('«') && unicode_tables::is_punctuation('—'));
    assert!(!unicode_tables::is_punctuation('$') && !unicode_tables::is_punctuation('+'));
    assert!(wordpiece::is_bert_punctuation('$') && wordpiece::is_bert_punctuation('+'));
    // Mn, including the variation selectors.
    assert!(unicode_tables::is_mark_nonspacing('\u{301}'));
    assert!(unicode_tables::is_mark_nonspacing('\u{FE0F}'));
    assert!(!unicode_tables::is_mark_nonspacing('\u{903}')); // Mc
    assert!(unicode_tables::SOURCE.contains("unicode_categories 0.1.1"));
}
