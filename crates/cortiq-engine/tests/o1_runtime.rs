//! O(1) Nyström attention — runtime integration gates.
//!
//! The kernel itself is golden-parity tested (nystrom_parity.rs); these
//! tests cover the RUNTIME plumbing: config resolution, the exact
//! prompt pass + seal + step lifecycle, GQA head mapping through the
//! pipeline, the short-prompt guard, and the memory accounting.

use cortiq_engine::kv_cache::{EvictionPolicy, KvCache, KvMode, LayerKvCache, O1State};
use cortiq_engine::nystrom::{
    O1_DEFAULT_M, O1_DEFAULT_RECT, O1_DEFAULT_SINK, O1_DEFAULT_W, O1Cfg, O1Layers, O1Rect,
};
use cortiq_engine::pipeline::create_test_pipeline;

fn o1(layers: O1Layers, m: usize, w: usize, sink: usize) -> Option<O1Cfg> {
    Some(O1Cfg {
        layers,
        m,
        w,
        sink,
        rect: O1_DEFAULT_RECT,
    })
}

#[test]
fn config_spec_parsing() {
    let defaults = (O1_DEFAULT_M, O1_DEFAULT_W, O1_DEFAULT_SINK);
    assert_eq!(O1Cfg::parse_layers("all"), Some(O1Layers::All));
    assert_eq!(O1Cfg::parse_layers("deep6"), Some(O1Layers::Deep(6)));
    assert_eq!(
        O1Cfg::parse_layers("1, 3,5"),
        Some(O1Layers::List(vec![1, 3, 5]))
    );
    assert_eq!(O1Cfg::parse_layers("off"), None);
    assert_eq!(O1Cfg::parse_layers("deepX"), None);
    assert_eq!(O1Cfg::parse_layers("1,x"), None);

    assert_eq!(O1Cfg::parse_rect("agg"), Some(O1Rect::Aggregate));
    assert_eq!(O1Cfg::parse_rect("aggregate"), Some(O1Rect::Aggregate));
    assert_eq!(O1Cfg::parse_rect("fm"), Some(O1Rect::Fm));
    assert_eq!(O1Cfg::parse_rect("clamp"), None);

    // deep-N flags = the N deepest layers; out-of-range list indices drop.
    let cfg = O1Cfg::from_spec("deep2", None, None, None, None).unwrap();
    assert_eq!(cfg.layer_flags(4), vec![false, false, true, true]);
    assert_eq!((cfg.m, cfg.w, cfg.sink), defaults, "validated defaults");
    assert_eq!(cfg.rect, O1_DEFAULT_RECT);
    let cfg =
        O1Cfg::from_spec("1,99", Some(8), Some(16), Some(0), Some(O1Rect::Aggregate)).unwrap();
    assert_eq!(cfg.layer_flags(3), vec![false, true, false]);
    assert_eq!((cfg.m, cfg.w, cfg.sink), (8, 16, 0));
    assert_eq!(cfg.rect, O1Rect::Aggregate, "explicit rect wins");
    assert!(O1Cfg::from_spec("off", None, None, None, None).is_none());

    // Header-hint JSON: string spec and explicit index array.
    let j = serde_json::json!({"layers": "all", "m": 8, "w": 32, "sink": 2});
    let cfg = O1Cfg::from_json(&j).unwrap();
    assert_eq!(cfg.layers, O1Layers::All);
    assert_eq!((cfg.m, cfg.w, cfg.sink), (8, 32, 2));
    let j = serde_json::json!({"layers": [0, 2]});
    let cfg = O1Cfg::from_json(&j).unwrap();
    assert_eq!(cfg.layers, O1Layers::List(vec![0, 2]));
    assert_eq!((cfg.m, cfg.w, cfg.sink), defaults);
    // The rectifier is a runtime knob: a file hint cannot pin it.
    assert_eq!(cfg.rect, O1_DEFAULT_RECT);
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

    // A long sealed request must not leak its skeleton into a later short
    // request: reset/reuse starts a fresh collecting state and defers again.
    let short = p.generate("ab", 8, None, None).unwrap();
    assert!(short.tokens_generated > 0);
    assert!(matches!(
        p.kv_cache.layers[0].o1,
        Some(O1State::Collecting { .. })
    ));
    assert!(p.kv_cache.layers[0].seq_len < 19);
}

/// Prompt shorter than the window (the §5-guard regime): the runtime keeps
/// the exact trace only until the first skeleton-safe boundary, then seals.
/// A short request must remain bounded by that one deferred boundary.
#[test]
fn o1_short_prompt_does_not_crash() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.sampler_config.temperature = 0.0;
    p.sampler_config.repetition_penalty = 1.0;
    p.set_o1(o1(O1Layers::All, 32, 128, 4));
    let r = p.generate("ab", 8, None, None).unwrap();
    assert_eq!(r.prompt_tokens, 2);
    assert!(r.tokens_generated > 0);
    assert!(!p.kv_cache.layers[0].o1_sealed());
    assert!(p.kv_cache.layers[0].seq_len < 32 + 128 + 4 + 8 + 1);
    assert!(matches!(
        p.kv_cache.layers[0].o1,
        Some(O1State::Collecting { .. })
    ));
}

/// The layer-level barrier keeps the full trace/KV through B-1, converts at
/// B, and makes repeated sealing idempotent. A malformed trace is rejected
/// before conversion and cannot resume ordinary KV growth if ignored.
#[test]
fn o1_deferred_layer_boundary_and_error_are_terminal() {
    let mut layer = LayerKvCache::new(1, 4);
    layer.o1_begin(4, 8, 2, O1_DEFAULT_RECT); // B = 19
    let q = vec![0.1f32; 8];
    let k = vec![0.2f32; 4];
    let v = vec![0.3f32; 4];
    for _ in 0..18 {
        layer.o1_push_q(&q);
        layer.append(&k, &v, &[]);
    }
    assert!(!layer.o1_seal(2));
    assert!(matches!(
        layer.o1.as_ref(),
        Some(O1State::Collecting {
            seal_at: Some(19),
            ..
        })
    ));
    assert_eq!(layer.head_keys(0).len(), 18 * 4);

    layer.o1_push_q(&q);
    layer.append(&k, &v, &[]);
    assert!(layer.o1_seal(2));
    let bounded = layer.o1_memory_bytes();
    assert!(layer.o1_sealed());
    assert!(layer.head_keys(0).is_empty());
    assert!(layer.o1_seal(2));
    assert_eq!(layer.o1_memory_bytes(), bounded);
    for _ in 0..64 {
        layer.o1_step(&q, &k, &v, 2);
    }
    assert_eq!(layer.head_keys(0).len(), 0);
    assert_eq!(layer.o1_memory_bytes(), bounded);

    let mut bad = LayerKvCache::new(1, 4);
    bad.o1_begin(4, 8, 2, O1_DEFAULT_RECT);
    for _ in 0..19 {
        bad.append(&k, &v, &[]);
    }
    assert!(!bad.o1_seal(2));
    assert!(bad.o1.is_none());
    assert_eq!(bad.seq_len, 0);
    bad.append(&k, &v, &[]);
    assert_eq!(bad.seq_len, 0, "failed seal must not resume KV growth");
}

/// A collecting layer owns the exact prefix needed for its deferred boundary.
/// Both cache eviction policies must leave its Q/K/V rows aligned until the
/// one conversion at B, even when the ordinary cache limit is much smaller.
#[test]
fn o1_deferred_prefix_survives_both_eviction_policies() {
    const B: usize = 19;
    let q = vec![0.1f32; 8];
    let k = vec![0.2f32; 4];
    let v = vec![0.3f32; 4];

    for policy in [EvictionPolicy::Recent, EvictionPolicy::Born { sink: 2 }] {
        let mut cache = KvCache::new(1, 1, 4, 6);
        cache.policy = policy;
        cache.layers[0].mode = KvMode::F32;
        cache.layers[0].o1_begin(4, 8, 2, O1_DEFAULT_RECT);

        cache.layers[0].o1_push_q(&q);
        cache.layers[0].append(&k, &v, &[]);
        assert!(!cache.layers[0].o1_seal(2));

        for _ in 1..B {
            cache.layers[0].o1_push_q(&q);
            cache.layers[0].append(&k, &v, &[]);
            cache.evict(3);
        }
        let layer = &cache.layers[0];
        assert_eq!(layer.seq_len, B, "policy {policy:?} must preserve depth");
        assert_eq!(layer.head_keys(0).len(), B * 4, "policy {policy:?} K rows");
        assert_eq!(
            layer.head_values(0).len(),
            B * 4,
            "policy {policy:?} V rows"
        );
        assert!(
            layer.o1_memory_bytes() >= B * 2 * 4 * 4,
            "policy {policy:?} must retain the full Q trace"
        );

        assert!(cache.layers[0].o1_seal(2));
        assert!(cache.layers[0].o1_sealed());
        assert!(cache.layers[0].head_keys(0).is_empty());
        assert!(cache.layers[0].head_values(0).is_empty());
    }
}

/// A short prompt eventually reaches B = w + sink + slack + 1. The exact
/// rows are retained through B-1; the completed Bth row seals and all later
/// steps use the bounded state without restoring ordinary KV.
#[test]
fn o1_short_prompt_seals_at_deferred_boundary() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.sampler_config.temperature = 0.0;
    p.sampler_config.repetition_penalty = 1.0;
    p.set_o1(o1(O1Layers::All, 4, 8, 2)); // B = 19
    let r = p.generate("ab", 32, None, None).unwrap();
    assert!(r.tokens_generated > 18, "test needs to cross B");
    for (li, layer) in p.kv_cache.layers.iter().enumerate() {
        assert!(layer.o1_sealed(), "layer {li} must seal at deferred B");
        assert!(layer.k_heads().iter().all(Vec::is_empty));
    }
}

/// The completed-row barrier is equivalent to an explicit seal over the
/// identical exact prefix: rows through B-1 stay exact, row B completes the
/// conversion, and row B+1 is the first streaming step.
#[test]
fn o1_boundary_matches_explicit_seal_prefix() {
    const B: usize = 19;
    let ids: Vec<u32> = (0..=B as u32).collect();
    let mut deferred = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    deferred.set_o1(o1(O1Layers::All, 4, 8, 2));
    deferred.o1_begin();
    for (pos, &id) in ids[..B - 1].iter().enumerate() {
        deferred
            .forward_span(
                &deferred.embed_id(id),
                pos,
                0,
                deferred.num_layers - 1,
                None,
            )
            .unwrap();
    }
    assert!(!deferred.o1_seal_checked().unwrap());
    assert!(matches!(
        deferred.kv_cache.layers[0].o1,
        Some(O1State::Collecting {
            seal_at: Some(B),
            ..
        })
    ));
    let b_minus_one = deferred.kv_cache.layers[0].head_len(0);
    assert_eq!(b_minus_one, B - 1);
    deferred
        .forward_span(
            &deferred.embed_id(ids[B - 1]),
            B - 1,
            0,
            deferred.num_layers - 1,
            None,
        )
        .unwrap();
    assert!(deferred.kv_cache.layers[0].o1_sealed());
    assert_eq!(deferred.kv_cache.layers[0].seq_len, B);
    assert!(deferred.kv_cache.layers[0].head_keys(0).is_empty());
    let deferred_first_stream = deferred
        .forward_span(
            &deferred.embed_id(ids[B]),
            B,
            0,
            deferred.num_layers - 1,
            None,
        )
        .unwrap();

    let mut explicit = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    explicit.set_o1(o1(O1Layers::All, 4, 8, 2));
    explicit.o1_begin();
    for (pos, &id) in ids[..B].iter().enumerate() {
        explicit
            .forward_span(
                &explicit.embed_id(id),
                pos,
                0,
                explicit.num_layers - 1,
                None,
            )
            .unwrap();
    }
    assert!(explicit.o1_seal_checked().unwrap());
    let explicit_first_stream = explicit
        .forward_span(
            &explicit.embed_id(ids[B]),
            B,
            0,
            explicit.num_layers - 1,
            None,
        )
        .unwrap();
    assert_eq!(
        deferred_first_stream, explicit_first_stream,
        "first compressed row must match explicit seal of the same exact prefix"
    );
}

/// The public batched span must publish the transition before a following
/// serial span consumes the first compressed row. This is the handoff used by
/// the network split after a prefix batch.
#[test]
fn o1_batched_span_handoffs_to_serial_span() {
    const B: usize = 19;
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.set_o1(o1(O1Layers::All, 4, 8, 2));
    p.o1_begin_with_prefix(Some(B));
    let ids: Vec<u32> = (0..B as u32).collect();
    let last = p.num_layers - 1;

    p.prefill_span_ids(&ids, 0, last, None)
        .expect("batch prefix should complete and seal");
    assert!(p.kv_cache.layers.iter().all(|l| l.o1_sealed()));

    let next = p.embed_id(B as u32);
    let hidden = p
        .forward_span(&next, B, 0, last, None)
        .expect("serial row after batch seal");
    assert_eq!(hidden.len(), p.hidden_size);
    assert!(p.kv_cache.layers.iter().all(|l| l.head_keys(0).is_empty()));
}

/// A conversion failure raised inside a batched span is an error at the
/// public boundary, never successful boundary hiddens. Reset/reuse then
/// clears the terminal latch and permits a fresh bounded request.
#[test]
fn o1_batched_failure_returns_err_and_reset_reuses() {
    const B: usize = 19;
    let ids: Vec<u32> = (0..B as u32).collect();
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.set_o1(o1(O1Layers::All, 4, 8, 2));
    p.o1_begin_with_prefix(Some(B));
    p.kv_cache.layers[0].mode = KvMode::Q8 { k: true, v: true };
    let last = p.num_layers - 1;

    let err = p
        .prefill_span_ids(&ids, 0, last, None)
        .expect_err("a malformed O(1) batch boundary must fail");
    assert!(err.contains("deferred O(1)"));
    assert_eq!(
        p.kv_cache.seq_len(),
        0,
        "failed batch must clear the request"
    );

    p.kv_cache.layers[0].mode = KvMode::F32;
    p.reset_session();
    p.o1_begin_with_prefix(Some(B));
    p.prefill_span_ids(&ids, 0, last, None)
        .expect("reset must make the bounded path reusable");
    assert!(p.kv_cache.layers.iter().all(|l| l.o1_sealed()));
}

/// The deferred exact lead-in must not change the caller's shifted target
/// range: a requested prefix of three still scores every target from three
/// onward, including exact rows before B.
#[test]
fn o1_nll_preserves_requested_range_under_deferred_boundary() {
    let mut p = create_test_pipeline(8, 16, 2, 1, 4, 2, 260);
    p.set_o1(o1(O1Layers::All, 4, 8, 2)); // B = 19
    let ids: Vec<u32> = (0..25).collect();
    let (_, count) = p.nll_ids_o1(&ids, 3).unwrap();
    assert_eq!(count, ids.len() - 1 - 3);
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

/// The speculative-burst contract: snapshot -> k steps -> restore must
/// leave the state BIT-identical, so a replay of the same tokens (the
/// accepted prefix of a rejected draft) reproduces the same outputs.
/// This is the mechanism Patent 16 says cannot exist ("insertion is
/// irreversible"): the bounded state makes it a memcpy.
#[test]
fn snapshot_restore_bit_exact() {
    let (m, w, sink, d, dv, heads, t_pre) =
        (8usize, 16usize, 2usize, 32usize, 32usize, 2usize, 96usize);
    let mut st = cortiq_engine::nystrom::NystromState::new_group(m, w, sink, heads);
    let det = |i: usize, j: usize, salt: u64| -> f32 {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
            .wrapping_add(salt);
        ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let q0: Vec<f32> = (0..t_pre * d).map(|i| det(i, 1, 7)).collect();
    let q1: Vec<f32> = (0..t_pre * d).map(|i| det(i, 1, 53)).collect();
    let ks: Vec<f32> = (0..t_pre * d).map(|i| det(i, 2, 11)).collect();
    let vs: Vec<f32> = (0..t_pre * dv).map(|i| det(i, 3, 13)).collect();
    st.prefill_group(&[&q0, &q1], &ks, &vs, t_pre, d, dv);

    // a few committed decode steps so the ring is live
    let mut out = vec![0f32; heads * dv];
    for s in 0..4usize {
        let q: Vec<f32> = (0..heads * d).map(|i| det(i, 4 + s, 17)).collect();
        let k: Vec<f32> = (0..d).map(|i| det(i, 40 + s, 19)).collect();
        let v: Vec<f32> = (0..dv).map(|i| det(i, 80 + s, 23)).collect();
        st.step_group(&q, &k, &v, &mut out);
    }

    let snap = st.snapshot();
    let mut out_a = vec![0f32; heads * dv];
    // the burst that will be "rejected"
    for s in 0..3usize {
        let q: Vec<f32> = (0..heads * d).map(|i| det(i, 200 + s, 29)).collect();
        let k: Vec<f32> = (0..d).map(|i| det(i, 240 + s, 31)).collect();
        let v: Vec<f32> = (0..dv).map(|i| det(i, 280 + s, 37)).collect();
        st.step_group(&q, &k, &v, &mut out_a);
    }
    st.restore(&snap);
    // replay a DIFFERENT continuation twice: restored state must give
    // bit-identical outputs to a second restore+replay
    let replay = |st: &mut cortiq_engine::nystrom::NystromState| -> Vec<f32> {
        let mut acc = Vec::new();
        let mut o = vec![0f32; heads * dv];
        for s in 0..3usize {
            let q: Vec<f32> = (0..heads * d).map(|i| det(i, 300 + s, 41)).collect();
            let k: Vec<f32> = (0..d).map(|i| det(i, 340 + s, 43)).collect();
            let v: Vec<f32> = (0..dv).map(|i| det(i, 380 + s, 47)).collect();
            st.step_group(&q, &k, &v, &mut o);
            acc.extend_from_slice(&o);
        }
        acc
    };
    let a = replay(&mut st);
    st.restore(&snap);
    let b = replay(&mut st);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(x.to_bits() == y.to_bits(), "restore is not bit-exact");
    }
}
