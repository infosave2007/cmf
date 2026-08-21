//! Calibration pass that accumulates the AWNP refit statistics online.
//!
//!   CMF_GPU=0 CMF_FFN_REFIT=<dir> [CMF_FFN_REFIT_FROM=a CMF_FFN_REFIT_TO=b] \
//!     cargo run --release -p cortiq-cli --example tube_refit_probe -- \
//!       model.cmf corpus_dir [tokens_per_file]
//!
//! `<dir>/support.<L>.u32` must already hold each layer's kept-neuron
//! list (written by the planner). The pass writes `gss.<L>.f32` and
//! `ya.<L>.f32` — everything the refit solve needs, without ever putting
//! a layer's activations on disk.
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let corpus_d = a.next().expect("corpus dir");
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
    let mut files: Vec<_> = std::fs::read_dir(&corpus_d)?.flatten().collect();
    files.sort_by_key(|e| e.file_name());
    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    for e in files {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let ids = p.tokenizer.encode(&std::fs::read_to_string(&path)?);
        let mut done = 0usize;
        for chunk in ids.chunks(256) {
            if done >= ntok || chunk.len() < 8 {
                break;
            }
            let _ = p.nll_ids_masked(chunk, 0, None);
            done += chunk.len();
            // One corpus file means the print below fires once, at the end,
            // and a pass that takes an hour looks identical to a hung one.
            if done % (256 * 40) == 0 {
                println!(
                    "  {done}/{ntok} tokens ({:.0}s, {:.0} tok/s)",
                    t0.elapsed().as_secs_f64(),
                    done as f64 / t0.elapsed().as_secs_f64().max(1e-9)
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }
        total += done;
        println!(
            "{}: {done} tokens ({total} total, {:.0}s)",
            path.file_stem().unwrap().to_str().unwrap(),
            t0.elapsed().as_secs_f64()
        );
    }
    let n = cortiq_engine::pipeline::refit_flush();
    println!("flushed {n} layer(s) after {total} tokens");
    Ok(())
}
