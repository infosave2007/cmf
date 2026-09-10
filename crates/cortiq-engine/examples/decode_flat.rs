//! Per-token decode-latency probe (scratch diagnostic, not a product).
//!
//! `bench` reports ONE aggregate decode tok/s, which cannot distinguish
//! "the operator is algorithmically slower at depth" from "the machine
//! throttled during the long prefill that precedes the decode". This
//! prints the per-token wall time so the shape is visible: a thermal
//! effect DECAYS across the run, an algorithmic cost is flat-but-higher
//! from the first token.
//!
//! Usage:
//!   cargo run --release --example decode_flat -- <model.cmf> <ctx> <tokens> [o1spec]

use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: decode_flat <model.cmf> <ctx> <tokens> [o1spec|off]");
        std::process::exit(2);
    }
    let path = &args[1];
    let ctx: usize = args[2].parse().expect("ctx");
    let tokens: usize = args[3].parse().expect("tokens");
    let o1spec = args.get(4).cloned().unwrap_or_else(|| "off".to_string());

    let model = Arc::new(CmfModel::open_sharded(path).expect("open model"));
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default()).expect("pipeline");
    pipeline.set_o1(cortiq_engine::nystrom::O1Cfg::from_spec(
        &o1spec, None, None, None, None,
    ));
    eprintln!("o1_active = {}", pipeline.o1_active());

    // Same synthetic prompt shape as `cortiq bench --ctx`.
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(ctx / 8 + 2);
    let mut ids = pipeline.tokenizer.encode(&prompt);
    ids.truncate(ctx);
    assert_eq!(ids.len(), ctx, "prompt too short for ctx");

    // Warm the mmap so the first token is not billed for page faults.
    let _ = pipeline.forward_ids(&ids[..2], None).expect("warm");

    let stamps: Arc<std::sync::Mutex<Vec<Instant>>> = Arc::default();
    let st = stamps.clone();
    let cb: cortiq_engine::TokenCallback = Box::new(move |_t| {
        st.lock().unwrap().push(Instant::now());
        true
    });
    let t0 = Instant::now();
    let _ = pipeline
        .generate_from_ids(&ids, tokens, None, Some(cb))
        .expect("generate");

    let stamps = stamps.lock().unwrap();
    // stamps[0] fires after generation's prefill + the one-off o1 seal:
    // that gap is TTFT, not a decode step.
    println!(
        "ctx={ctx} o1={o1spec} ttft_s={:.3}",
        stamps[0].duration_since(t0).as_secs_f64()
    );
    println!("# idx  ms");
    for i in 1..stamps.len() {
        println!(
            "{:4}  {:7.2}",
            i,
            (stamps[i] - stamps[i - 1]).as_secs_f64() * 1e3
        );
    }
    // Mean of the last half — past any startup transient.
    let half = stamps.len() / 2;
    if stamps.len() > 2 && half >= 1 {
        let dt = (stamps[stamps.len() - 1] - stamps[half]).as_secs_f64();
        let n = (stamps.len() - 1 - half) as f64;
        println!(
            "last-half: {:.2} ms/tok = {:.2} tok/s",
            dt / n * 1e3,
            n / dt
        );
    }
}
