//! Token-id parity of a chat template against transformers'
//! `apply_chat_template`, through the exact call the server makes
//! (`try_apply_chat_template_json`: messages as JSON, tools, thinking).
//!
//! Env-gated — it needs a real tokenizer and a reference file:
//!   CMF_TPL_REF=cases.json        {"cases": [{messages, tools?, enable_thinking?, hf_ids}]}
//!   CMF_TPL_TOKENIZER=tokenizer.json
//!   CMF_TPL_TEMPLATE=chat_template.jinja   (default: the MiniCPM5 fixture)
//! The reference comes from `transformers` on the ORIGINAL template, e.g.
//! `at(at.apply_chat_template(msgs, tools=…, add_generation_prompt=True,
//! tokenize=False), add_special_tokens=False).input_ids`.

use cortiq_engine::tokenizer::Tokenizer;

#[test]
fn chat_template_ids_match_transformers() {
    let (Ok(ref_path), Ok(tok_path)) = (
        std::env::var("CMF_TPL_REF"),
        std::env::var("CMF_TPL_TOKENIZER"),
    ) else {
        eprintln!("skipped: set CMF_TPL_REF and CMF_TPL_TOKENIZER");
        return;
    };
    let template = match std::env::var("CMF_TPL_TEMPLATE") {
        Ok(p) => std::fs::read_to_string(p).unwrap(),
        Err(_) => include_str!("fixtures/minicpm5_chat_template.jinja").to_string(),
    };
    let mut tok = Tokenizer::from_file(&tok_path).expect("tokenizer");
    tok.chat_template = Some(template);
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).unwrap()).unwrap();

    let mut failures = 0;
    let cases = fx["cases"].as_array().unwrap();
    for (i, case) in cases.iter().enumerate() {
        let msgs: Vec<serde_json::Value> = case["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<serde_json::Value>> =
            case.get("tools").and_then(|t| t.as_array()).cloned();
        let think = case.get("enable_thinking").and_then(|v| v.as_bool());
        let want: Vec<u32> = case["hf_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        match tok.try_apply_chat_template_json(&msgs, tools.as_deref(), think) {
            Ok(got) if got == want => println!("case {i}: ids identical ({})", got.len()),
            Ok(got) => {
                failures += 1;
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                eprintln!(
                    "case {i}: MISMATCH got {} want {} first diff at {at:?}\n--- got text ---\n{}",
                    got.len(),
                    want.len(),
                    tok.render_chat_json(&msgs, tools.as_deref(), think)
                        .unwrap_or_default()
                );
            }
            Err(e) => {
                failures += 1;
                eprintln!("case {i}: render error: {e}");
            }
        }
    }
    assert_eq!(failures, 0, "{failures} of {} cases differ", cases.len());
}
