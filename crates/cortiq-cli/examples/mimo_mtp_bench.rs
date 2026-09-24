//! MiMo-V2 speculative decode with the MTP draft stack: plain greedy vs
//! speculative greedy on ONE loaded model, in-process timers, token-identity
//! check, per-depth acceptance.
//!
//!   cargo run --release -p cortiq-cli --example mimo_mtp_bench -- \
//!       model.cmf prompt.txt [-n 128] [--raw] [--think] [--modes plain,spec] \
//!       [--ids-out ids.json] [--k 3] [--ids prompt_ids.json]
//!
//! `--ids` feeds a JSON id array verbatim (the prompt file is then only
//! a placeholder).
//! The draft stack loads from `<stem>.mtp.cmf` beside the model. Greedy,
//! no repetition penalty (the llama-bench contract), EOS ignored so every
//! arm decodes exactly `n` tokens. Decode rate = (n − 1) tokens over the
//! time from the first streamed token to the last (prefill excluded); the
//! first token's latency is reported separately. `--ids-out` writes
//! {"prompt": ids, "continuation": ids} (the plain arm's) for the oracle.
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        anyhow::bail!(
            "usage: mimo_mtp_bench model.cmf prompt.txt [-n N] [--raw] [--think] \
             [--modes plain,spec] [--ids-out F] [--k K]"
        );
    }
    let mut n = 128usize;
    let mut raw = false;
    let mut think = false;
    let mut modes = vec!["plain".to_string(), "spec".to_string()];
    let mut ids_out: Option<String> = None;
    let mut ids_in: Option<String> = None;
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "-n" => {
                n = a[i + 1].parse()?;
                i += 1;
            }
            "--raw" => raw = true,
            "--ids" => {
                ids_in = Some(a[i + 1].clone());
                i += 1;
            }
            "--think" => think = true,
            "--modes" => {
                modes = a[i + 1].split(',').map(|s| s.to_string()).collect();
                i += 1;
            }
            "--ids-out" => {
                ids_out = Some(a[i + 1].clone());
                i += 1;
            }
            "--k" => {
                // read by MimoMtp::from_layers at load
                unsafe { std::env::set_var("CMF_MIMO_MTP_K", &a[i + 1]) };
                i += 1;
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
        i += 1;
    }
    let t_load = Instant::now();
    let m = Arc::new(CmfModel::open_sharded(&a[1])?);
    let cfg = SamplerConfig {
        temperature: 0.0,
        seed: Some(42),
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut p = Pipeline::from_model(&m, cfg)?;
    p.ignore_eos = true;
    p.set_confidence(false);
    eprintln!(
        "loaded {} in {:.1}s; MTP draft stack: {}",
        a[1],
        t_load.elapsed().as_secs_f64(),
        match &p.mimo_mtp {
            Some(st) => format!("{} layers, depth {}", st.layers.len(), st.depth),
            None => "absent".into(),
        }
    );
    let text = std::fs::read_to_string(&a[2])?;
    let ids: Vec<u32> = if let Some(f) = &ids_in {
        serde_json::from_str(&std::fs::read_to_string(f)?)?
    } else if raw {
        p.tokenizer.encode(&text)
    } else {
        p.tokenizer
            .apply_chat_template_opts(&[("user".to_string(), text.clone())], Some(think))
    };
    eprintln!("prompt: {} tokens ({})", ids.len(), if raw { "raw" } else { "chat" });

    let mut first_ids: Option<Vec<u32>> = None;
    let mut rows = Vec::new();
    for mode in &modes {
        let spec = match mode.as_str() {
            "plain" => false,
            "spec" => true,
            other => anyhow::bail!("mode {other}: plain | spec"),
        };
        p.speculative = spec;
        p.reset_session();
        let stamps: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
        let s2 = stamps.clone();
        let t0 = Instant::now();
        let r = p
            .generate_from_ids(
                &ids,
                n,
                None,
                Some(Box::new(move |_t: &str| {
                    s2.lock().unwrap().push(Instant::now());
                    true
                })),
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        let st = stamps.lock().unwrap().clone();
        let ttft = st.first().map(|t| t.duration_since(t0).as_secs_f64()).unwrap_or(0.0);
        let decode_s = match (st.first(), st.last()) {
            (Some(f), Some(l)) if st.len() > 1 => l.duration_since(*f).as_secs_f64(),
            _ => 0.0,
        };
        let tps = if decode_s > 0.0 {
            (st.len() - 1) as f64 / decode_s
        } else {
            0.0
        };
        let stats = p.mimo_mtp.as_ref().map(|s| s.stats.clone());
        let same = match &first_ids {
            None => {
                first_ids = Some(r.token_ids.clone());
                true
            }
            Some(f) => *f == r.token_ids,
        };
        let text_out = p.tokenizer.decode(&r.token_ids);
        eprintln!(
            "[{mode}] {} tokens, TTFT {ttft:.2}s, decode {tps:.2} tok/s, identical to first arm: {same}",
            r.token_ids.len()
        );
        if let (true, Some(s)) = (spec, &stats) {
            eprintln!("[{mode}] {}", s.line());
        }
        eprintln!("[{mode}] text: {:?}", text_out.chars().take(400).collect::<String>());
        rows.push(serde_json::json!({
            "mode": mode,
            "prompt_tokens": ids.len(),
            "tokens": r.token_ids.len(),
            "ttft_s": ttft,
            "decode_tok_s": tps,
            "decode_s": decode_s,
            "identical_to_first": same,
            "mtp_drafted": r.mtp_drafted,
            "mtp_accepted": r.mtp_accepted,
            "rounds": stats.as_ref().filter(|_| spec).map(|s| s.rounds),
            "tokens_per_round": stats.as_ref().filter(|_| spec).map(|s| s.tokens_per_round()),
            "depth_drafted": stats.as_ref().filter(|_| spec).map(|s| s.depth_drafted.clone()),
            "depth_accepted": stats.as_ref().filter(|_| spec).map(|s| s.depth_accepted.clone()),
            "accept_hist": stats.as_ref().filter(|_| spec).map(|s| s.accept_hist.clone()),
            "draft_ms_per_round": stats.as_ref().filter(|_| spec)
                .map(|s| s.draft_ns as f64 / 1e6 / s.rounds.max(1) as f64),
            "verify_ms_per_round": stats.as_ref().filter(|_| spec)
                .map(|s| s.verify_ns as f64 / 1e6 / s.rounds.max(1) as f64),
        }));
        if let (Some(path), false) = (&ids_out, spec) {
            std::fs::write(
                path,
                serde_json::json!({"prompt": ids, "continuation": r.token_ids}).to_string(),
            )?;
        }
    }
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
