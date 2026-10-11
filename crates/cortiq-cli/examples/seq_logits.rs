//! Teacher-forced next-token logits over RAW token-id sequences, through
//! the generation path (prefill of the first id, then one decode step per
//! position) — the bridge to an external reference (HF transformers,
//! llama.cpp) and the quant-quality gate (KL / top-1 against a reference
//! CMF on the same ids).
//!
//! Input: JSONL, one `{"name": "...", "ids": [..], "greedy": N}` per line
//! (`greedy` optional: after the sequence, decode N greedy tokens and
//! record them with their logits rows too; `score_from` optional: write
//! rows from this index on only; `text` / `chat` optional: tokenizer and
//! chat-template parity against `ids`, `parity_only` skips the forward).
//! Output, per sequence, in <out_dir>:
//!   <name>.f32  — rows × vocab little-endian f32, row p = logits after
//!                 position p (row 0 predicts ids[1]); greedy rows follow
//!                 (rows before `score_from` are not written)
//!   <name>.json — {"n_rows", "vocab", "n_prompt", "greedy_ids", "secs"}
//! With `--top1` only the argmax ids and the per-row logsumexp are kept
//! (no .f32), for quick checks of large sets.
//!
//!     cargo run --release -p cortiq-cli --example seq_logits -- m.cmf seqs.jsonl out/

use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::{BufRead, Write};
use std::sync::Arc;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: seq_logits <model.cmf> <seqs.jsonl> <out_dir> [--top1]");
        std::process::exit(2);
    }
    let top1_only = args.iter().any(|a| a == "--top1");
    let out_dir = std::path::PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let model = Arc::new(CmfModel::open_sharded(&args[1]).expect("open model"));
    let mut sampler = SamplerConfig::default();
    sampler.temperature = 0.0;
    sampler.repetition_penalty = 1.0;
    let mut p = Pipeline::from_model(&model, sampler).expect("pipeline");
    let f = std::fs::File::open(&args[2]).expect("seqs");
    for line in std::io::BufReader::new(f).lines() {
        let line = line.expect("line");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("json");
        let name = v["name"].as_str().expect("name").to_string();
        let ids: Vec<u32> = v["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let greedy = v.get("greedy").and_then(|g| g.as_u64()).unwrap_or(0) as usize;
        // Tokenizer parity: `text` = the raw prompt string (BOS added as the
        // runtime adds it), `chat` = OpenAI-shaped messages rendered through
        // the file's chat template. Both must reproduce `ids` exactly.
        let mut parity = serde_json::Map::new();
        if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
            let mine = p.tokenizer.with_bos(p.tokenizer.encode(t));
            parity.insert("text_ids_equal".into(), (mine == ids).into());
            if mine != ids {
                let d = mine.iter().zip(&ids).position(|(a, b)| a != b);
                eprintln!("{name}: raw-text ids differ (len {} vs {}, first diff {d:?})", mine.len(), ids.len());
            }
        }
        if let Some(msgs) = v.get("chat").and_then(|m| m.as_array()) {
            let mine = p.tokenizer.apply_chat_template_json(msgs, None, None);
            parity.insert("chat_ids_equal".into(), (mine == ids).into());
            if mine != ids {
                let d = mine.iter().zip(&ids).position(|(a, b)| a != b);
                eprintln!("{name}: chat-template ids differ (len {} vs {}, first diff {d:?})", mine.len(), ids.len());
            }
        }
        if v.get("parity_only").and_then(|b| b.as_bool()).unwrap_or(false) {
            println!("{name} {}", serde_json::Value::Object(parity));
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(ids.len() + greedy);
        let first = p.forward_ids(&ids[..1], None).expect("forward_ids");
        let vocab = first.len();
        rows.push(first);
        for (pos, &id) in ids.iter().enumerate().skip(1) {
            rows.push(p.decode_step_logits(id, pos));
        }
        let mut greedy_ids = Vec::new();
        let mut pos = ids.len();
        for _ in 0..greedy {
            let tok = argmax(rows.last().unwrap()) as u32;
            greedy_ids.push(tok);
            if p.tokenizer.is_eos(tok) {
                break;
            }
            rows.push(p.decode_step_logits(tok, pos));
            pos += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        let top1: Vec<usize> = rows.iter().map(|r| argmax(r)).collect();
        let lse: Vec<f64> = rows
            .iter()
            .map(|r| {
                let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                r.iter().map(|&x| ((x - m) as f64).exp()).sum::<f64>().ln() + m as f64
            })
            .collect();
        // `score_from`: only rows from this index on are written (the
        // prompt rows of a quality corpus are computed, not stored).
        let score_from = v.get("score_from").and_then(|g| g.as_u64()).unwrap_or(0) as usize;
        if !top1_only {
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(out_dir.join(format!("{name}.f32"))).expect("create"),
            );
            for r in rows.iter().skip(score_from) {
                let bytes: Vec<u8> = r.iter().flat_map(|x| x.to_le_bytes()).collect();
                w.write_all(&bytes).expect("write");
            }
        }
        let meta = serde_json::json!({
            "n_rows": rows.len(),
            "score_from": score_from,
            "vocab": vocab,
            "n_prompt": ids.len(),
            "greedy_ids": greedy_ids,
            "greedy_text": p.tokenizer.decode(&greedy_ids),
            "top1": top1,
            "lse": lse,
            "secs": secs,
            "parity": serde_json::Value::Object(parity),
        });
        std::fs::write(
            out_dir.join(format!("{name}.json")),
            serde_json::to_vec(&meta).unwrap(),
        )
        .expect("meta");
        eprintln!(
            "{name}: {} rows ({} prompt + {} greedy) in {secs:.1}s",
            rows.len(),
            ids.len(),
            greedy_ids.len()
        );
    }
}
