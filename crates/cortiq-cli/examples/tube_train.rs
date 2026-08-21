//! Train ONE DTG-MA mask per task (Patent 2, phase A only) and dump the
//! trained per-neuron logits — the importance field the tube planner
//! orders neurons by. Raw top-K activation mass is measurably the wrong
//! criterion (see docs/SKILLS.md: untrained top-K collapses quality);
//! the L1-trained mask is the repo's certified one.
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_train -- \
//!       model.cmf corpus_dir out_dir [steps_a] [target_sparsity] [chunks]
//!
//! `corpus_dir/<task>.txt` → `out_dir/<task>.mass` (σ(logit) per neuron,
//! the mass-dump layout) + `out_dir/<task>.keep` (hard flags, one byte
//! per neuron) + a one-line report per task.
use cortiq_core::CmfModel;
use cortiq_engine::skillbake::{BakeHyper, skill_bake};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let corpus_d = a.next().expect("corpus dir");
    let out_d = a.next().expect("out dir");
    let steps_a: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(120);
    let target: f64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let max_chunks: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let chunk_len: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
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
        let text = std::fs::read_to_string(&path)?;
        let ids = p.tokenizer.encode(&text);
        let chunks: Vec<Vec<u32>> = ids
            .chunks(chunk_len)
            .filter(|c| c.len() == chunk_len)
            .take(max_chunks)
            .map(|c| c.to_vec())
            .collect();
        let held = (chunks.len() / 5).clamp(3, 12);
        if chunks.len() < held + 12 {
            eprintln!("{task}: only {} chunks — skipped", chunks.len());
            continue;
        }
        let hy = BakeHyper {
            steps_a,
            steps_b: 0,
            fcd_layers: 0,
            target_sparsity: target,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let (rep, arts) = match skill_bake(&m, &chunks, held, &hy, |l| {
            println!("  [{task}] {l}");
        }) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("{task}: bake failed: {err}");
                continue;
            }
        };
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out_d}/{task}.mass"))?);
        f.write_all(&(nl as u32).to_le_bytes())?;
        f.write_all(&(inter as u32).to_le_bytes())?;
        for li in 0..nl {
            let row = arts.logits.get(li);
            for n in 0..inter {
                let v = row.and_then(|r| r.get(n)).copied().unwrap_or(0.0);
                f.write_all(&(1.0 / (1.0 + (-v).exp())).to_le_bytes())?;
            }
        }
        f.flush()?;
        let mut k = std::io::BufWriter::new(std::fs::File::create(format!("{out_d}/{task}.keep"))?);
        for li in 0..nl {
            for n in 0..inter {
                k.write_all(&[arts.keep[li][n] as u8])?;
            }
        }
        k.flush()?;
        println!(
            "{task}: backbone {:.3} → masked {:.3} | pruned {:.1}% | {} chunks | {:.0}s (total {:.0}s)",
            rep.backbone,
            rep.masked,
            rep.pruned_ratio * 100.0,
            chunks.len(),
            rep.sec,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
