//! Cross-turn KV reuse on the wgpu whole-token graph.
//!
//! The graph decodes into a DEVICE K/V mirror; the chunked prefill of a
//! pure-attention model reads and appends to the HOST cache. Before the fix a
//! reused turn found the host cache ending at the previous prompt, so its
//! tail prefill attended without the model's own answer and wrote its rows at
//! the wrong index (MiniCPM5-2B on an RTX 3090 repeated its tool call instead
//! of reading the tool result: 1 of 6 against 6 of 6 fresh).
//!
//! Needs a ChatML pure-attention .cmf the token graph takes (MiniCPM5 or
//! Qwen3 at q8_2f / q4tp) and a wgpu adapter; skips otherwise:
//!
//!     CMF_GPU=wgpu CMF_REUSE_MODEL=/path/model.cmf cargo test -p cortiq-engine \
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

/// Turn 1 of a ChatML chat with thinking off.
fn first_turn(p: &cortiq_engine::Pipeline, user: &str) -> Vec<u32> {
    p.tokenizer
        .apply_chat_template_opts(&[("user".to_string(), user.to_string())], Some(false))
}

/// The ids a chat client sends next: the previous prompt, the model's own
/// answer (closed with `<|im_end|>` if the cap cut it), then a new user turn
/// and an open, non-thinking assistant turn — ChatML, as the templates render.
fn next_turn(p: &cortiq_engine::Pipeline, prev: &[u32], answer: &[u32], user: &str) -> Vec<u32> {
    let mut ids = prev.to_vec();
    ids.extend_from_slice(answer);
    let end = p.tokenizer.encode("<|im_end|>");
    if !answer.ends_with(&end) {
        ids.extend_from_slice(&end);
    }
    ids.extend(p.tokenizer.encode(&format!(
        "\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )));
    ids
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
    let ids1 = first_turn(
        &warm,
        "Invent an unusual name for a pet dragon and describe its colour in two sentences.",
    );
    let r1 = warm.generate_from_ids(&ids1, 96, None, None).unwrap();
    eprintln!("turn 1 ({} tokens): {:?}", r1.token_ids.len(), r1.text);
    assert!(r1.token_ids.len() > 8, "turn 1 generated too little to test");

    // (a) The chat pattern: turn 2 = turn 1 + the model's own answer + a
    // question only that answer can settle. It strictly extends the cache,
    // so it is reused — and must equal a fresh computation of the same ids.
    let q2 = "What was the dragon's name? Answer with the name only.";
    let ids2 = next_turn(&warm, &ids1, &r1.token_ids, q2);
    let r2_warm = warm.generate_from_ids(&ids2, n, None, None).unwrap();
    let r2_fresh = mk().generate_from_ids(&ids2, n, None, None).unwrap();
    eprintln!("reused turn: {:?}\nfresh turn:  {:?}", r2_warm.text, r2_fresh.text);
    assert!(r2_fresh.token_ids.len() >= 2, "fresh turn 2 said nothing");
    let same = agree(&r2_warm.token_ids, &r2_fresh.token_ids);
    assert!(
        same >= n.min(r2_fresh.token_ids.len()),
        "reused turn diverged from the fresh prefill after {same} token(s)"
    );

    // (b) The new prompt diverges INSIDE the previous generation (the
    // client edited the answer): no reuse — the device rows of the old
    // answer must not leak into the fresh sequence.
    let cut = r1.token_ids.len() / 2;
    let mut edited = r1.token_ids[..cut].to_vec();
    edited.extend(warm.tokenizer.encode(" Actually, its name is Ember."));
    let ids3 = next_turn(&warm, &ids1, &edited, q2);
    let r3_warm = warm.generate_from_ids(&ids3, n, None, None).unwrap();
    let r3_fresh = mk().generate_from_ids(&ids3, n, None, None).unwrap();
    eprintln!("edited turn: {:?}\nfresh turn:  {:?}", r3_warm.text, r3_fresh.text);
    let same = agree(&r3_warm.token_ids, &r3_fresh.token_ids);
    assert!(
        same >= n.min(r3_fresh.token_ids.len()),
        "edited turn differs from the fresh prefill after {same} token(s)"
    );
}
