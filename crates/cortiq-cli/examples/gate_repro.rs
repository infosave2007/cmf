//! Repro for the cross-model GPU cache poisoning the bake gate exposed:
//! open and score several models back-to-back in ONE process. Each score
//! must match what a fresh process reports for the same file — if any
//! resident cache keys device buffers by host address alone, a later
//! model inherits the earlier one's bytes wherever its mmap landed on
//! the freed one, and the perplexity explodes.
//!
//!   cargo run --release -p cortiq-cli --example gate_repro -- text.txt A.cmf [B.cmf ...]
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let txt = args.next().expect("text file first");
    let text = std::fs::read_to_string(&txt)?;
    let mut ids: Option<Vec<u32>> = None;
    for path in args {
        let m = Arc::new(CmfModel::open(&path)?);
        let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
        let ids = ids.get_or_insert_with(|| {
            let mut v = p.tokenizer.encode(&text);
            v.truncate(512);
            v
        });
        let (nll, n) = p.nll_ids_from(ids, 0);
        println!("{path}: PPL = {:.3}", (nll / n.max(1) as f64).exp());
    }
    Ok(())
}
