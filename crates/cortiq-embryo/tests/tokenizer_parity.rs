//! Our BPE trained on the repo docs, saved as tokenizer.json, loaded by the
//! RUNTIME's tokenizer: encodings must be identical (and round-trip).
use cortiq_embryo::tokenizer::{Bpe, SPLIT, count_words, train};
use std::collections::HashMap;

fn utf8_prefix(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[test]
fn runtime_encodes_our_tokenizer_identically() {
    let mut text = String::new();
    for f in [
        "../../README.md",
        "../../README.ru.md",
        "../../docs/CMF_V2_SPEC.md",
        "../../docs/SKILLS.ru.md",
    ] {
        if let Ok(s) = std::fs::read_to_string(f) {
            text.push_str(&s);
            text.push('\n');
        }
    }
    assert!(text.len() > 10_000, "need some corpus text");
    let re = fancy_regex::Regex::new(SPLIT).unwrap();
    let mut counts = HashMap::new();
    count_words(&text, &re, &mut counts);
    let bpe = train(&counts, 2048, false);
    assert_eq!(bpe.vocab_size(), 2048);
    let json = bpe.to_hf_json();
    let dir = std::env::temp_dir().join(format!("embryo_tok_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, &json).unwrap();
    // reload ours
    let ours = Bpe::load(&path).unwrap();
    // runtime
    let rt = cortiq_engine::tokenizer::Tokenizer::from_json(&json)
        .expect("runtime loads our tokenizer.json");
    let samples = [
        "Hello, world! Cortiq Embryo — саморазвивающееся ядро на нашем формате.",
        "fn main() {\n    println!(\"{}\", 42);\n}\n",
        "   Leading spaces, tabs\t and 12345 numbers, e-mail@example.com; 日本語テキスト.",
        utf8_prefix(&text, 20_000),
    ];
    let mut cache = HashMap::new();
    for s in samples {
        let mut a = Vec::new();
        ours.encode(s, &mut cache, &mut a);
        let b = rt.encode(s);
        assert_eq!(a, b, "encoding differs on {:?}", utf8_prefix(s, 60));
        assert_eq!(ours.decode(&a), s, "round trip");
    }
    let bytes: usize = samples.iter().map(|s| s.len()).sum();
    let mut all = Vec::new();
    for s in samples {
        ours.encode(s, &mut cache, &mut all);
    }
    eprintln!(
        "vocab 2048 on docs: {:.2} bytes/token",
        bytes as f64 / all.len() as f64
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn heldout_text_encodes_identically_in_the_runtime() {
    let Ok(tok_path) = std::env::var("EMBRYO_TOK") else {
        return;
    };
    let text = std::fs::read_to_string("/Users/oleg/embryo-data/heldout-en.txt").unwrap();
    let json = std::fs::read_to_string(&tok_path).unwrap();
    let ours = Bpe::load(std::path::Path::new(&tok_path)).unwrap();
    let rt = cortiq_engine::tokenizer::Tokenizer::from_json(&json).unwrap();
    let mut cache = HashMap::new();
    let mut a = Vec::new();
    ours.encode(&text, &mut cache, &mut a);
    let b = rt.encode(&text);
    eprintln!("ours {} tokens, runtime {} tokens", a.len(), b.len());
    let first_diff = a.iter().zip(&b).position(|(x, y)| x != y);
    if let Some(i) = first_diff {
        eprintln!(
            "first diff at {i}: ours {:?} runtime {:?}",
            &a[i.saturating_sub(3)..(i + 5).min(a.len())],
            &b[i.saturating_sub(3)..(i + 5).min(b.len())]
        );
        eprintln!(
            "ours decode: {:?}",
            ours.decode(&a[i.saturating_sub(3)..(i + 5).min(a.len())])
        );
    }
    assert_eq!(a, b);
}
