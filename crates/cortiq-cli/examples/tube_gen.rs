//! Build a calibration corpus of the model's OWN answers.
//!
//!   cortiq … --example tube_gen -- model.cmf prompts.jsonl out.txt \
//!            [max_tokens] [take] [skip]
//!
//! Calibrating on text another model wrote measures the wrong
//! activations: what the refit needs is what THIS model does when it
//! works. The prompts can come from anywhere — they are only the
//! stimulus — but the continuation has to be the model's own.
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let prompts_p = a.next().expect("prompts.jsonl");
    let out_p = a.next().expect("out.txt");
    let max_tokens: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let take: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let skip: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
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
    let text = std::fs::read_to_string(&prompts_p)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_p)?);
    let t0 = std::time::Instant::now();
    let (mut n, mut toks) = (0usize, 0usize);
    for line in text.lines().skip(skip).take(take) {
        // one JSON object per line; the prompt field is all we need
        let Some(i) = line.find("\"prompt\":") else {
            continue;
        };
        let rest = &line[i + 9..];
        let Some(s) = rest.find('"') else { continue };
        let mut prompt = String::new();
        let mut esc = false;
        for c in rest[s + 1..].chars() {
            if esc {
                prompt.push(match c {
                    'n' => '\n',
                    't' => '\t',
                    other => other,
                });
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                break;
            } else {
                prompt.push(c);
            }
        }
        if prompt.len() < 20 {
            continue;
        }
        let ids = p
            .tokenizer
            .apply_chat_template(&[("user".to_string(), prompt.clone())]);
        let r = match p.generate_from_ids(&ids, max_tokens, None, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("generate failed: {e}");
                continue;
            }
        };
        writeln!(out, "{prompt}\n{}\n", r.text.trim())?;
        n += 1;
        toks += r.tokens_generated;
        if n % 10 == 0 {
            println!(
                "{n} answers, {toks} tokens, {:.0}s ({:.1} tok/s)",
                t0.elapsed().as_secs_f64(),
                toks as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    out.flush()?;
    println!("wrote {out_p}: {n} answers, {toks} generated tokens");
    Ok(())
}
