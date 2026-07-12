//! O(1) Nyström attention — runtime integration gates.
//!
//! The kernel itself is golden-parity tested (nystrom_parity.rs); these
//! tests cover the RUNTIME plumbing: config resolution, the exact
//! prompt pass + seal + step lifecycle, GQA head mapping through the
//! pipeline, the short-prompt guard, and the memory accounting.

use cortiq_engine::nystrom::{O1Cfg, O1Layers};
use cortiq_engine::pipeline::create_test_pipeline;

fn o1(layers: O1Layers, m: usize, w: usize, sink: usize) -> Option<O1Cfg> {
    Some(O1Cfg { layers, m, w, sink })
}

#[test]
fn config_spec_parsing() {
    assert_eq!(O1Cfg::parse_layers("all"), Some(O1Layers::All));
    assert_eq!(O1Cfg::parse_layers("deep6"), Some(O1Layers::Deep(6)));
    assert_eq!(
        O1Cfg::parse_layers("1, 3,5"),
        Some(O1Layers::List(vec![1, 3, 5]))
    );
    assert_eq!(O1Cfg::parse_layers("off"), None);
    assert_eq!(O1Cfg::parse_layers("deepX"), None);
    assert_eq!(O1Cfg::parse_layers("1,x"), None);

    // deep-N flags = the N deepest layers; out-of-range list indices drop.
    let cfg = O1Cfg::from_spec("deep2", None, None, None).unwrap();
    assert_eq!(cfg.layer_flags(4), vec![false, false, true, true]);
    assert_eq!((cfg.m, cfg.w, cfg.sink), (32, 128, 4), "validated defaults");
    let cfg = O1Cfg::from_spec("1,99", Some(8), Some(16), Some(0)).unwrap();
    assert_eq!(cfg.layer_flags(3), vec![false, true, false]);
    assert_eq!((cfg.m, cfg.w, cfg.sink), (8, 16, 0));
    assert!(O1Cfg::from_spec("off", None, None, None).is_none());

    // Header-hint JSON: string spec and explicit index array.
    let j = serde_json::json!({"layers": "all", "m": 8, "w": 32, "sink": 2});
    let cfg = O1Cfg::from_json(&j).unwrap();
    assert_eq!(cfg.layers, O1Layers::All);
    assert_eq!((cfg.m, cfg.w, cfg.sink), (8, 32, 2));
    let j = serde_json::json!({"layers": [0, 2]});
    let cfg = O1Cfg::from_json(&j).unwrap();
    assert_eq!(cfg.layers, O1Layers::List(vec![0, 2]));
    assert_eq!((cfg.m, cfg.w, cfg.sink), (32, 128, 4));
}

/// With a window wider than the whole run the kernel stays in
/// exact-only mode, so the o1 pipeline must reproduce the baseline
/// greedy sequence — this validates the projection/RoPE/GQA plumbing
/// end-to-end, independent of the skeleton approximation.
#[test]
fn o1_exact_window_matches_baseline_greedy() {
    let run = |o1_cfg: Option<O1Cfg>| {
        let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
        p.sampler_config.temperature = 0.0;
        p.sampler_config.repetition_penalty = 1.0;
        p.set_o1(o1_cfg);
        p.generate("abcdef", 12, None, None).unwrap().token_ids
    };
    let baseline = run(None);
    let o1_ids = run(o1(O1Layers::All, 4, 64, 4));
    assert_eq!(
        baseline, o1_ids,
        "exact-only o1 must reproduce the baseline greedy sequence"
    );
}

/// Long generation across the window boundary: the ring evicts into the
/// far accumulators every step, the layer stores nothing per position,
/// and the state is counted in memory_bytes.
#[test]
fn o1_long_generation_crosses_window_and_stays_o1() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.sampler_config.temperature = 0.0;
    p.sampler_config.repetition_penalty = 1.0;
    // 36-token prompt > w + sink + 8 = 18 → skeleton mode for real.
    p.set_o1(o1(O1Layers::All, 4, 8, 2));
    let prompt = "abcdefghijklmnopqrstuvwxyz0123456789";
    let r = p.generate(prompt, 40, None, None).unwrap();
    assert_eq!(r.prompt_tokens, 36);
    assert!(r.tokens_generated > 0);
    for &c in &r.token_confidence {
        assert!(c.is_finite() && (0.0..=1.0).contains(&c), "confidence {c}");
    }
    for (li, layer) in p.kv_cache.layers.iter().enumerate() {
        assert!(layer.o1_sealed(), "layer {li} must be sealed");
        assert_eq!(
            layer.head_keys(0).len(),
            0,
            "layer {li}: sealed layer must hold no per-position KV"
        );
        let o1_mem = layer.o1_memory_bytes();
        assert!(o1_mem > 0, "layer {li}: nystrom state must be accounted");
        assert!(
            layer.memory_bytes() >= o1_mem,
            "layer {li}: memory_bytes must include the o1 state"
        );
    }
    // O(1) claim: the state does not grow with generated tokens.
    let before: usize = p.kv_cache.layers.iter().map(|l| l.o1_memory_bytes()).sum();
    let _ = p.generate(prompt, 80, None, None).unwrap();
    let after: usize = p.kv_cache.layers.iter().map(|l| l.o1_memory_bytes()).sum();
    assert_eq!(before, after, "sealed state must be constant in context");
}

/// Prompt shorter than the window (the §5-guard regime): the kernel
/// runs exact-only with a growing buffer — the runtime must not assume
/// skeleton state exists.
#[test]
fn o1_short_prompt_does_not_crash() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.sampler_config.temperature = 0.0;
    p.sampler_config.repetition_penalty = 1.0;
    p.set_o1(o1(O1Layers::All, 32, 128, 4));
    let r = p.generate("ab", 8, None, None).unwrap();
    assert_eq!(r.prompt_tokens, 2);
    assert!(r.tokens_generated > 0);
    assert!(p.kv_cache.layers[0].o1_sealed());
}

/// Per-layer override is really per-layer: an un-flagged layer keeps
/// growing its exact KV while the flagged one runs O(1).
#[test]
fn o1_mixed_layers_split_exact_and_o1() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.sampler_config.temperature = 0.0;
    p.sampler_config.repetition_penalty = 1.0;
    p.set_o1(o1(O1Layers::List(vec![1]), 4, 8, 2));
    let prompt = "abcdefghijklmnopqrstuvwxyz0123456789";
    let r = p.generate(prompt, 20, None, None).unwrap();
    let expect_positions = 36 + r.tokens_generated - 1; // see pipeline KV test
    let l0 = &p.kv_cache.layers[0];
    let l1 = &p.kv_cache.layers[1];
    assert!(!l0.o1_sealed() && l1.o1_sealed());
    assert_eq!(l0.head_keys(0).len() / 4, expect_positions);
    assert_eq!(l1.head_keys(0).len(), 0);
    assert_eq!(l0.seq_len, l1.seq_len, "both layers track the same depth");
}
