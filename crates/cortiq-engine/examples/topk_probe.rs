//! Next-token top-k probe over a RAW token prefix (scratch diagnostic).
//!
//! `explain` applies the chat template and only ever reports position 0,
//! so it cannot answer "what does the model think comes after token N of
//! its own answer". This feeds an arbitrary raw string (added tokens are
//! honoured) and prints the top-k logits AND softmax mass of the next
//! token — the margin is what separates "quantization flipped a near-tie"
//! from "the model really believes this".
//!
//! Usage:
//!   cargo run --release --example topk_probe -- <model.cmf> <raw text> [k]

use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: topk_probe <model.cmf> <raw text> [k]");
        std::process::exit(2);
    }
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    let model = Arc::new(CmfModel::open_sharded(&args[1]).expect("open model"));
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default()).expect("pipeline");
    let ids = pipeline.tokenizer.encode(&args[2]);
    eprintln!("prompt tokens: {}", ids.len());
    eprintln!("last 8 ids: {:?}", &ids[ids.len().saturating_sub(8)..]);

    let logits = pipeline.prefill_next_logits(&ids, None);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    for &i in order.iter().take(k) {
        let p = ((logits[i] - max) as f64).exp() / sum;
        println!(
            "{:>8}  logit {:>9.4}  p {:.6}  {:?}",
            i,
            logits[i],
            p,
            pipeline.tokenizer.decode_token(i as u32)
        );
    }
}
