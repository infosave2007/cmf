//! Per-task FFN activation-mass probe — the raw material for task tubes.
//!
//! For every task file in `<seeds>` (either `<seeds>/<task>.txt` or
//! `<seeds>/<task>/*.txt`, one prompt per line) the model answers each
//! prompt greedily, then the prompt+answer token stream is run back
//! through the DTG-MA calibration pass (`probe_ffn_mass`) so the mass
//! reflects what the model actually does on that task — prefill AND
//! decode, not just the question.
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_probe -- \
//!       model.cmf seeds_dir out_dir [gen_tokens] [max_prompts]
//!
//! Writes `<out>/<task>.mass` — `u32 layers, u32 inter, f32[layers*inter]`
//! (little-endian) — plus `<out>/<task>.txt`, the corpus the mass came
//! from (the same text the quality gate later scores).
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::Write;
use std::sync::Arc;

fn task_files(seeds: &str) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let Ok(rd) = std::fs::read_dir(seeds) else {
        return out;
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() || name.starts_with('.') {
            continue;
        }
        let mut lines: Vec<String> = Vec::new();
        if path.is_dir() {
            let mut inner: Vec<_> = std::fs::read_dir(&path).into_iter().flatten().flatten().collect();
            inner.sort_by_key(|e| e.file_name());
            for f in inner {
                if let Ok(t) = std::fs::read_to_string(f.path()) {
                    lines.extend(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from));
                }
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            if let Ok(t) = std::fs::read_to_string(&path) {
                lines.extend(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from));
            }
        }
        if !lines.is_empty() {
            out.push((name, lines));
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let seeds = a.next().expect("seeds dir");
    let out_dir = a.next().expect("out dir");
    let gen_n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(96);
    let max_p: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    std::fs::create_dir_all(&out_dir)?;

    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let (nl, inter) = (m.arch().num_layers, m.arch().intermediate_size);
    let cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(1),
        suppress_tokens: Vec::new(),
    };
    let mut p = Pipeline::from_model(&m, cfg)?;

    for (task, prompts) in task_files(&seeds) {
        let t0 = std::time::Instant::now();
        let mut mass = vec![vec![0f64; inter]; nl];
        let mut corpus = String::new();
        let mut tok_total = 0usize;
        for prompt in prompts.iter().take(max_p) {
            let ids = p
                .tokenizer
                .apply_chat_template(&[("user".to_string(), prompt.clone())]);
            let r = match p.generate_from_ids(&ids, gen_n, None, None) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  [{task}] generate failed: {e}");
                    continue;
                }
            };
            corpus.push_str(prompt);
            corpus.push('\n');
            corpus.push_str(r.text.trim());
            corpus.push_str("\n\n");
            // Probe the FULL stream the model saw: template + answer.
            let mut full = ids.clone();
            full.extend(r.token_ids.iter().skip(r.prompt_tokens).copied());
            tok_total += full.len();
            for (li, row) in p.probe_ffn_mass(&full).into_iter().enumerate() {
                if li >= nl {
                    break;
                }
                for (acc, v) in mass[li].iter_mut().zip(row) {
                    *acc += v;
                }
            }
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!(
            "{out_dir}/{task}.mass"
        ))?);
        f.write_all(&(nl as u32).to_le_bytes())?;
        f.write_all(&(inter as u32).to_le_bytes())?;
        for row in &mass {
            for v in row {
                f.write_all(&(*v as f32).to_le_bytes())?;
            }
        }
        f.flush()?;
        std::fs::write(format!("{out_dir}/{task}.txt"), &corpus)?;
        println!(
            "{task}: {} prompt(s), {tok_total} tokens probed, {:.1}s",
            prompts.len().min(max_p),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
