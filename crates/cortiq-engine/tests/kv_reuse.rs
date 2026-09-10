//! Cross-turn KV reuse gate (needs the local MiniCPM3 .cmf; skips
//! otherwise): a second generate whose ids extend the first turn must
//! produce EXACTLY what a fresh pipeline produces for the same ids.
#[test]
fn kv_reuse_matches_fresh_prefill() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/minicpm3-q4t.cmf";
    if !std::path::Path::new(p).exists() {
        return;
    }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let cfg = || cortiq_engine::SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut warm = cortiq_engine::Pipeline::from_model(&m, cfg()).unwrap();
    let msgs1 = vec![("user".to_string(), "Name one primary color.".to_string())];
    let ids1 = warm.tokenizer.apply_chat_template(&msgs1);
    let r1 = warm.generate_from_ids(&ids1, 24, None, None).unwrap();

    // Turn 2 = turn 1 + the reply + a follow-up — the chat pattern.
    let mut ids2 = ids1.clone();
    ids2.extend(r1.token_ids.iter().copied());
    ids2.extend(warm.tokenizer.encode("\nAnd one more?"));
    let r2_warm = warm.generate_from_ids(&ids2, 24, None, None).unwrap();

    let mut fresh = cortiq_engine::Pipeline::from_model(&m, cfg()).unwrap();
    let r2_fresh = fresh.generate_from_ids(&ids2, 24, None, None).unwrap();

    assert_eq!(
        r2_warm.token_ids, r2_fresh.token_ids,
        "reused-prefix generation must equal fresh-prefill generation"
    );
    eprintln!(
        "turn2 ids: {} (cached prefix would be ~{}), output {:?}",
        ids2.len(),
        ids1.len() + r1.token_ids.len().saturating_sub(1),
        r2_warm.text
    );
}
