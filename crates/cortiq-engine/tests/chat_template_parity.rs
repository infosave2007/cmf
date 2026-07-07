//! Chat-template parity: minijinja render vs the reference jinja2
//! (transformers semantics). Env-gated: CMF_CHAT_CASES=path/to/cases.json
//! (built by tests/gen_chat_cases.py).

use cortiq_engine::tokenizer::Tokenizer;

#[test]
fn chat_template_matches_jinja2_reference() {
    let Ok(cases_path) = std::env::var("CMF_CHAT_CASES") else {
        eprintln!("skipped: set CMF_CHAT_CASES");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();

    let mut tok = Tokenizer::byte_level();
    tok.chat_template = Some(fx["template"].as_str().unwrap().to_string());

    let mut failures = 0;
    for (i, case) in fx["cases"].as_array().unwrap().iter().enumerate() {
        let messages: Vec<(String, String)> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let m = m.as_array().unwrap();
                (
                    m[0].as_str().unwrap().to_string(),
                    m[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let want = case["rendered"].as_str().unwrap();
        match tok.render_chat(&messages) {
            Some(got) if got == want => {}
            Some(got) => {
                failures += 1;
                eprintln!("case {i} RENDER mismatch\n--- want ---\n{want}\n--- got ---\n{got}");
            }
            None => {
                failures += 1;
                eprintln!("case {i}: render failed");
            }
        }
    }
    assert_eq!(failures, 0, "{failures} chat template parity failures");
    println!("chat template parity: all cases identical");
}
