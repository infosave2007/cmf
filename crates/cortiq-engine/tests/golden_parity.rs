//! Golden parity: the Rust engine must reproduce the numpy reference
//! on the same .cmf file — greedy token-for-token, logits within
//! accumulation-order tolerance.
//!
//! Driven by tests/golden_parity.sh which sets:
//!   CMF_GOLDEN_FILE=<tiny.cmf>  CMF_GOLDEN_REF=<reference.json>
//! Without the env vars the test is skipped (unit runs stay hermetic).

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

#[derive(serde::Deserialize)]
struct Reference {
    quant: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    first_logits: Vec<f32>,
}

#[test]
fn engine_matches_numpy_reference() {
    let (Ok(model_path), Ok(ref_path)) = (
        std::env::var("CMF_GOLDEN_FILE"),
        std::env::var("CMF_GOLDEN_REF"),
    ) else {
        eprintln!("golden parity skipped: CMF_GOLDEN_FILE/CMF_GOLDEN_REF not set");
        return;
    };

    let reference: Reference =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).unwrap()).unwrap();
    let model = Arc::new(CmfModel::open(&model_path).unwrap());
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();

    // First-step logits: element-wise against numpy.
    let logits = pipeline
        .forward_ids(&reference.prompt_ids, None)
        .expect("forward");
    assert_eq!(logits.len(), reference.first_logits.len());
    let mut max_diff = 0f32;
    for (a, b) in logits.iter().zip(&reference.first_logits) {
        max_diff = max_diff.max((a - b).abs());
    }
    // CMF_GOLDEN_LOOSE=1: A8W8 SDOT pass — activation quantization noise
    // is expected (bounded); the greedy sequence below stays strict.
    let tol = if std::env::var("CMF_GOLDEN_LOOSE").is_ok() {
        0.05
    } else {
        1e-3
    };
    assert!(
        max_diff < tol,
        "[{}] first-step logits diverge: max|Δ| = {max_diff} (tol {tol})",
        reference.quant
    );

    // Greedy decode: token-for-token.
    let mut ids = reference.prompt_ids.clone();
    let mut greedy = Vec::new();
    for _ in 0..reference.greedy_ids.len() {
        let logits = pipeline.forward_ids(&ids, None).unwrap();
        let next = cortiq_engine::sampler::argmax(&logits);
        greedy.push(next);
        ids.push(next);
    }
    assert_eq!(
        greedy, reference.greedy_ids,
        "[{}] greedy sequence diverged from numpy reference",
        reference.quant
    );

    println!(
        "golden parity [{}]: greedy {:?} ok, max|Δ|logits = {max_diff:.2e}",
        reference.quant, greedy
    );
}
