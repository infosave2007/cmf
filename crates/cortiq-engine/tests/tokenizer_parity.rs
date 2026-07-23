//! Bit-exact tokenizer parity vs the reference HF `tokenizers` library.
//! Env-gated (needs a real tokenizer.json): run via tests/tokenizer_parity.sh
//!   CMF_TOK_JSON=path/to/tokenizer.json CMF_TOK_CASES=path/to/cases.json

use cortiq_engine::tokenizer::Tokenizer;

#[test]
fn tokenizer_matches_hf_reference() {
    let (Ok(tok_path), Ok(cases_path)) = (
        std::env::var("CMF_TOK_JSON"),
        std::env::var("CMF_TOK_CASES"),
    ) else {
        eprintln!("skipped: set CMF_TOK_JSON and CMF_TOK_CASES");
        return;
    };
    let tok = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();

    let mut failures = 0;
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let expect: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text);
        if got != expect {
            failures += 1;
            eprintln!(
                "case {i} ENCODE mismatch\n  text: {text:?}\n  want: {expect:?}\n  got:  {got:?}"
            );
        }
        let decoded = tok.decode(&expect);
        let want_decoded = case["decoded"].as_str().unwrap();
        if decoded != want_decoded {
            failures += 1;
            eprintln!("case {i} DECODE mismatch\n  want: {want_decoded:?}\n  got:  {decoded:?}");
        }
    }
    assert_eq!(failures, 0, "{failures} tokenizer parity failures");
    println!("tokenizer parity: all cases bit-exact");
}
