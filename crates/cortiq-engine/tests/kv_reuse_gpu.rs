//! Cross-turn KV reuse on the wgpu whole-token graph.
//!
//! The graph decodes into a DEVICE K/V mirror; the chunked prefill of a
//! pure-attention model reads and appends to the HOST cache. Before the fix a
//! reused turn found the host cache ending at the previous prompt, so its
//! tail prefill attended without the model's own answer and wrote its rows at
//! the wrong index (MiniCPM5-2B on an RTX 3090 repeated its tool call instead
//! of reading the tool result: 1 of 6 against 6 of 6 fresh).
//!
//! Needs a pure-attention .cmf the token graph takes (q8_2f / q4tp) and a
//! wgpu adapter; skips otherwise:
//!
//!     CMF_REUSE_MODEL=/path/model.cmf cargo test -p cortiq-engine \
//!         --features gpu --release --test kv_reuse_gpu -- --nocapture

#![cfg(feature = "gpu")]

use std::sync::Arc;

fn greedy() -> cortiq_engine::SamplerConfig {
    cortiq_engine::SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    }
}

fn chat(p: &cortiq_engine::Pipeline, msgs: &[(&str, &str)]) -> Vec<u32> {
    let m: Vec<(String, String)> = msgs
        .iter()
        .map(|(r, c)| (r.to_string(), c.to_string()))
        .collect();
    p.tokenizer.apply_chat_template(&m)
}

/// Leading tokens two greedy continuations share.
fn agree(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[test]
fn reused_turn_on_the_token_graph_matches_a_fresh_prefill() {
    let Ok(path) = std::env::var("CMF_REUSE_MODEL") else {
        eprintln!("CMF_REUSE_MODEL unset — skip");
        return;
    };
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu selected but the context did not come up"),
        Some(true) => {}
    }
    let model = Arc::new(cortiq_core::CmfModel::open_sharded(&path).unwrap());
    let mk = || cortiq_engine::Pipeline::from_model(&model, greedy()).unwrap();
    let n = 24usize;

    // Turn 1 decodes enough tokens on the graph that the answer matters.
    let mut warm = mk();
    let ids1 = chat(
        &warm,
        &[(
            "user",
            "Invent a name for a pet dragon and describe its colour in two sentences.",
        )],
    );
    let r1 = warm.generate_from_ids(&ids1, 48, None, None).unwrap();
    assert!(r1.token_ids.len() > 8, "turn 1 generated too little to test");

    // (a) The chat pattern: turn 2 = turn 1 + the model's own answer + a
    // question about that answer. It strictly extends the cache, so it is
    // reused — and must equal a fresh computation of the same ids.
    let follow = warm
        .tokenizer
        .encode("\nWhat was the dragon's name? Answer with the name only.");
    let mut ids2 = ids1.clone();
    ids2.extend(r1.token_ids.iter().copied());
    ids2.extend(follow.iter().copied());
    let r2_warm = warm.generate_from_ids(&ids2, n, None, None).unwrap();
    let r2_fresh = mk().generate_from_ids(&ids2, n, None, None).unwrap();
    eprintln!(
        "reused turn: {:?}\nfresh turn:  {:?}",
        r2_warm.text, r2_fresh.text
    );
    let same = agree(&r2_warm.token_ids, &r2_fresh.token_ids);
    assert!(
        same >= n.min(r2_fresh.token_ids.len()),
        "reused turn diverged from the fresh prefill after {same} token(s)"
    );

    // (b) The new prompt diverges INSIDE the previous generation (an edited
    // answer): no reuse — the stale device rows past the divergence must
    // not leak into the fresh sequence.
    let r3 = warm.generate_from_ids(&ids1, 48, None, None).unwrap();
    let cut = r3.token_ids.len() / 2;
    let mut ids3 = ids1.clone();
    ids3.extend(r3.token_ids[..cut].iter().copied());
    ids3.extend(warm.tokenizer.encode(" Actually, call it Ember instead."));
    ids3.extend(follow.iter().copied());
    let r3_warm = warm.generate_from_ids(&ids3, n, None, None).unwrap();
    let r3_fresh = mk().generate_from_ids(&ids3, n, None, None).unwrap();
    let same = agree(&r3_warm.token_ids, &r3_fresh.token_ids);
    assert!(
        same >= n.min(r3_fresh.token_ids.len()),
        "diverged turn differs from the fresh prefill after {same} token(s)"
    );
}
