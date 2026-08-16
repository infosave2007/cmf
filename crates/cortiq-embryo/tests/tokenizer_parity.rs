//! Our BPE trained on the repo docs, saved as tokenizer.json, loaded by the
//! RUNTIME's tokenizer: encodings must be identical (and round-trip).
use cortiq_embryo::tokenizer::{Bpe, SPLIT, count_words, train};
use std::collections::HashMap;

#[test]
fn runtime_encodes_our_tokenizer_identically() {
    let mut text = String::new();
    for f in ["../../README.md", "../../README.ru.md", "../../docs/CMF_V2_SPEC.md", "../../docs/SKILLS.ru.md"] {
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
    let rt = cortiq_engine::tokenizer::Tokenizer::from_json(&json).expect("runtime loads our tokenizer.json");
    let samples = [
        "Hello, world! Cortiq Embryo — саморазвивающееся ядро на нашем формате.",
        "fn main() {\n    println!(\"{}\", 42);\n}\n",
        "   Leading spaces, tabs\t and 12345 numbers, e-mail@example.com; 日本語テキスト.",
        &text[..20_000],
    ];
    let mut cache = HashMap::new();
    for s in samples {
        let mut a = Vec::new();
        ours.encode(s, &mut cache, &mut a);
        let b = rt.encode(s);
        assert_eq!(a, b, "encoding differs on {:?}", &s[..s.len().min(60)]);
        assert_eq!(ours.decode(&a), s, "round trip");
    }
    let bytes: usize = samples.iter().map(|s| s.len()).sum();
    let mut all = Vec::new();
    for s in samples {
        ours.encode(s, &mut cache, &mut all);
    }
    eprintln!("vocab 2048 on docs: {:.2} bytes/token", bytes as f64 / all.len() as f64);
    let _ = std::fs::remove_dir_all(&dir);
}
