//! DeepSeek-V4.1 regression fixture on the CPU path.
//!
//!     cargo test -p cortiq-engine --test dsv41_fixture -- --nocapture
//!
//! A five-layer synthetic V4.1 checkpoint (see `common/dsv41_fixture.rs`)
//! covers ratio-0/2/1 layers, two CSA2 KV sources, an index consumer,
//! candidate pre-filtering, Engram, q4tp and q2tp gate/up experts and F32
//! linears. It runs through the same loader, `QTensor` kernels, pool and
//! shared DSV4 helpers as the released model. Since 0.6.7 those shared
//! paths changed by tens of thousands of lines without a V4.1 run; these
//! tests make such drift visible without the 510 GB checkpoint.
//!
//! "Strict" below is the exact-activation CPU path (`CMF_SDOT=0`, or
//! `qtensor::float_activations_scope`); "default" is what a CPU decode
//! runs, with A8W8 int8 activations in the quantized matvecs on ARM SDOT
//! and x86 AVX2 hosts.

// The fixture's `json!` config nests past the default macro depth.
#![recursion_limit = "256"]

#[path = "common/dsv41_fixture.rs"]
mod dsv41_fixture;

use cortiq_core::CmfModel;
use cortiq_engine::dsv41::{self, Dsv41Cfg, Dsv41State};
use cortiq_engine::pool::Pool;
use cortiq_engine::qtensor::float_activations_scope;
use dsv41_fixture::*;
use std::sync::Arc;

struct Loaded {
    model: Arc<CmfModel>,
    cfg: Dsv41Cfg,
    n_layers: usize,
}

fn load_fixture(fx: &Fixture, tag: &str) -> Loaded {
    let path = fx.write(tag);
    let model = Arc::new(CmfModel::open(&path).expect("open fixture"));
    let source = model
        .arch()
        .deepseek_v41
        .clone()
        .expect("fixture keeps its source config");
    let (cfg, n_layers) = Dsv41Cfg::from_source(&source, model.arch()).expect("parse fixture cfg");
    Loaded {
        model,
        cfg,
        n_layers,
    }
}

fn run(l: &Loaded, token_map: Vec<u32>, tokens: &[u32], pool: Option<&Pool>) -> Vec<Vec<f32>> {
    let (globals, layers, hash) =
        dsv41::load(&l.model, &l.cfg, l.n_layers, token_map).expect("load fixture layers");
    let mut st = Dsv41State::new(&l.cfg, hash);
    let mut out = Vec::with_capacity(tokens.len());
    for (p, &t) in tokens.iter().enumerate() {
        let mut logits = Vec::new();
        dsv41::forward_token(&globals, &layers, &l.cfg, &mut st, t, p, pool, &mut logits);
        out.push(logits);
    }
    out
}

fn per_position(got: &[Vec<f32>], want: &[Vec<f64>]) -> Vec<f64> {
    got.iter()
        .zip(want)
        .map(|(g, w)| worst_rel(std::slice::from_ref(g), std::slice::from_ref(w)).0)
        .collect()
}

fn fmt_errs(errs: &[f64]) -> String {
    errs.iter()
        .map(|e| format!("{e:.1e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn argmax_agree(got: &[Vec<f32>], want: &[Vec<f64>]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| argmax32(g) == argmax64(w))
        .count()
}

fn bits_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits())
        })
}

/// The parsed geometry is the fixture's: a parse regression would make the
/// comparisons below meaningless, so check it first and loudly.
#[test]
fn fixture_config_parses_to_its_geometry() {
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let l = load_fixture(&fx, "cfg");
    let c = &l.cfg;
    assert_eq!(l.n_layers, N_LAYERS);
    assert_eq!(
        (c.dim, c.n_heads, c.head_dim, c.rope_head_dim),
        (DIM, N_HEADS, HEAD_DIM, ROPE_DIM)
    );
    assert_eq!(
        (c.q_lora_rank, c.o_lora_rank, c.o_groups),
        (Q_LORA, O_LORA, O_GROUPS)
    );
    assert_eq!(
        (c.n_routed_experts, c.top_k, c.moe_inter),
        (EXPERTS, TOP_K, INTER)
    );
    assert_eq!(c.window, WINDOW);
    assert_eq!(c.compress_ratios, RATIOS.to_vec());
    assert_eq!(c.kv_sources, KV_SOURCES.to_vec());
    assert_eq!(c.index_sources, INDEX_SOURCES.to_vec());
    assert_eq!(
        (c.index_heads, c.index_head_dim, c.index_topk),
        (IH, ID, INDEX_TOPK)
    );
    assert_eq!(
        (c.candidate_source, c.candidate_topk_blocks, c.candidate_block_size),
        (Some(CANDIDATE_SOURCE), CAND_BLOCKS, CAND_BS)
    );
    assert_eq!(c.engram_layers, vec![ENGRAM_LAYER]);
    assert_eq!(c.engram_embeddings, vec![ENGRAM_ROWS]);
    assert_eq!(c.engram_compressed_vocab, RELEASE_COMPRESSED_VOCAB);
    assert_eq!(c.swiglu_limit as f64, SWIGLU);
    assert_eq!(c.route_scale as f64, ROUTE_SCALE);
    assert_eq!(c.rope_factor as f64, ROPE_FACTOR);
    assert_eq!(c.original_seq_len as f64, ORIG_SEQ);
}

/// Strict CPU logits against the independent f64 reference.
///
/// When no BF16 rounding boundary falls between the f32 and the f64
/// accumulator, the runtime and the reference agree to ~1e-7 relative:
/// every BF16 boundary, quantizer, rope pair, hc transpose, top-k and
/// candidate rule is the same. Where one value does land on the other side
/// of a boundary, the FP4/FP8 caches can carry the one-ulp flip into a
/// visible logit change (measured ~3e-3). A structural regression moves
/// every position, so the test asks most positions to be exact and bounds
/// the rest.
#[test]
fn strict_cpu_forward_matches_f64_reference() {
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let l = load_fixture(&fx, "ref");
    let got = float_activations_scope(|| run(&l, fx.token_map(), &TOKENS, None));
    let want = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let errs = per_position(&got, &want);
    let exact = errs.iter().filter(|&&e| e < 1e-5).count();
    let (worst, at) = worst_rel(&got, &want);
    let (kern, kern_at) = worst_rel(&got, &reference_logits(&fx, &TOKENS, RefOpts::kernels()));
    println!(
        "dsv41 fixture strict: per-position relative logit error [{}]; \
         {exact}/{} exact, worst {worst:.3e} at {at}, argmax {}/{}; \
         against the official kernel semantics (BF16 attention probabilities \
         and output, act_quant(o)) worst {kern:.3e} at {kern_at}",
        fmt_errs(&errs),
        TOKENS.len(),
        argmax_agree(&got, &want),
        TOKENS.len()
    );
    assert!(got.iter().flatten().all(|v| v.is_finite()), "non-finite logits");
    assert!(
        exact * 3 >= TOKENS.len() * 2,
        "only {exact}/{} positions agree with the reference to 1e-5: [{}]",
        TOKENS.len(),
        fmt_errs(&errs)
    );
    assert!(
        worst < 0.15,
        "strict CPU V4.1 forward drifted from the reference: {worst:.3e} at position {at}"
    );
}

/// `candidate_source_layer_id: -1` turns candidate pre-filtering off (it
/// used to map to the release layer 20). With it off, the index consumer
/// L3 ranks every compressed row, and the logits follow the reference
/// without candidates, not the one with them.
#[test]
fn negative_candidate_source_runs_without_prefiltering() {
    let fx = Fixture::build_with(RELEASE_COMPRESSED_VOCAB, None);
    let l = load_fixture(&fx, "nocand");
    assert_eq!(l.cfg.candidate_source, None);
    let got = float_activations_scope(|| run(&l, fx.token_map(), &TOKENS, None));
    let want = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let errs = per_position(&got, &want);
    let exact = errs.iter().filter(|&&e| e < 1e-5).count();
    let with = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let (apart, _) = worst_rel(&got, &reference_logits(&with, &TOKENS, RefOpts::cortiq()));
    println!(
        "dsv41 fixture without candidates: [{}]; {exact}/{} exact; the run with \
         candidates differs by up to {apart:.3e}",
        fmt_errs(&errs),
        TOKENS.len()
    );
    assert!(exact * 3 >= TOKENS.len() * 2, "[{}]", fmt_errs(&errs));
    assert!(apart > 1e-3, "candidate pre-filtering never changed a pick");
}

/// The default CPU path (A8W8 activations where the host has the int8
/// dot units) stays close to the reference; it is not exact by design.
#[test]
fn default_cpu_forward_tracks_f64_reference() {
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let l = load_fixture(&fx, "default");
    let got = run(&l, fx.token_map(), &TOKENS, None);
    let want = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let (worst, at) = worst_rel(&got, &want);
    let agree = argmax_agree(&got, &want);
    println!(
        "dsv41 fixture default: worst relative logit error {worst:.3e} at {at}, \
         argmax {agree}/{}",
        TOKENS.len()
    );
    assert!(worst < 0.25, "default CPU forward: {worst:.3e} at {at}");
    assert!(
        agree + 2 >= TOKENS.len(),
        "argmax agrees at {agree}/{}",
        TOKENS.len()
    );
}

/// Two fresh states give bit-identical logits, and the multi-thread pool
/// gives the same bits as the single-thread walk, in both activation
/// modes: the decode is a pure function of the weights and the tokens.
#[test]
fn cpu_forward_is_deterministic_and_pool_invariant() {
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let l = load_fixture(&fx, "det");
    let pool = Pool::new(4);
    let a = run(&l, fx.token_map(), &TOKENS, None);
    let b = run(&l, fx.token_map(), &TOKENS, None);
    let c = run(&l, fx.token_map(), &TOKENS, Some(&pool));
    assert!(bits_equal(&a, &b), "two default runs differ");
    assert!(bits_equal(&a, &c), "default: pool and single-thread differ");
    let s1 = float_activations_scope(|| run(&l, fx.token_map(), &TOKENS, None));
    let s2 = float_activations_scope(|| run(&l, fx.token_map(), &TOKENS, Some(&pool)));
    assert!(bits_equal(&s1, &s2), "strict: pool and single-thread differ");
}

/// `Pipeline::from_model` reaches the same stack: the loader's V4.1 branch
/// parses the same geometry and its serial logits equal the direct walk.
/// The container carries no tokenizer, so every id normalizes to one
/// Engram key and the config states a compressed vocabulary of one.
#[test]
fn pipeline_loads_fixture_and_matches_direct_walk() {
    let fx = Fixture::build(1);
    let l = load_fixture(&fx, "pipe");
    let direct = run(&l, fx.token_map(), &TOKENS, None);
    let mut pipeline = cortiq_engine::pipeline::Pipeline::from_model(
        &l.model,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .expect("pipeline loads the V4.1 fixture");
    let loaded_cfg = pipeline
        .dsv41
        .as_ref()
        .map(|b| b.2.clone())
        .expect("pipeline selected the V4.1 stack");
    assert_eq!(loaded_cfg, l.cfg, "loader parsed a different geometry");
    let serial = pipeline
        .dsv41_serial_logits(&TOKENS)
        .expect("serial logits");
    // The pipeline owns a worker pool; the pool-invariance test above is
    // what makes a bitwise comparison with the pool-free walk meaningful.
    assert!(
        bits_equal(&serial, &direct),
        "pipeline and direct walk differ"
    );
}
