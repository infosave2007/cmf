//! Per-task FFN activation mass over EXISTING text (no generation) —
//! the probe for models too slow to answer their own prompts first.
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_mass -- \
//!       model.cmf corpus_dir out_dir [tokens_per_task]
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let corpus_d = a.next().expect("corpus dir");
    let out_d = a.next().expect("out dir");
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    std::fs::create_dir_all(&out_d)?;
    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let (nl, inter) = (m.arch().num_layers, m.arch().intermediate_size);
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
    let mut files: Vec<_> = std::fs::read_dir(&corpus_d)?.flatten().collect();
    files.sort_by_key(|e| e.file_name());
    for e in files {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let task = path.file_stem().unwrap().to_str().unwrap().to_string();
        let t0 = std::time::Instant::now();
        let ids = p.tokenizer.encode(&std::fs::read_to_string(&path)?);
        let mut mass = vec![vec![0f64; inter]; nl];
        let mut done = 0usize;
        // `CMF_PROBE_BATCH=1` rides the prefill sweep instead of one
        // forward per token — the only affordable way on a big model.
        let batched = std::env::var("CMF_PROBE_BATCH").is_ok();
        for chunk in ids.chunks(if batched { 4096 } else { 256 }) {
            if done >= ntok {
                break;
            }
            let probe = if batched {
                p.probe_ffn_mass_batch(chunk)
            } else {
                p.probe_ffn_mass(chunk)
            };
            for (li, row) in probe.into_iter().enumerate() {
                if li >= nl {
                    break;
                }
                for (acc, v) in mass[li].iter_mut().zip(row) {
                    *acc += v;
                }
            }
            done += chunk.len();
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out_d}/{task}.mass"))?);
        f.write_all(&(nl as u32).to_le_bytes())?;
        f.write_all(&(inter as u32).to_le_bytes())?;
        for row in &mass {
            for v in row {
                f.write_all(&(*v as f32).to_le_bytes())?;
            }
        }
        f.flush()?;
        println!("{task}: {done} tokens, {:.0}s", t0.elapsed().as_secs_f64());
    }
    Ok(())
}
