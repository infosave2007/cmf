//! `cortiq-embryo loglik`: lm-eval `loglikelihood` requests scored on the GPU
//! by the trainer's evaluation forward — the numbers `cortiq loglik --split
//! hflm --frame raw` computes on the runtime's CPU path (same tokenizer, same
//! pair encoding, the backbone only), at the card's speed: one row per
//! request, rows bucketed by length, every GEMM in full f32 (no TF32), the
//! served routing (dropless, served anchor window, frozen descriptors).
//!
//! Output lines: the request minus `context` / `continuation`, plus `i`,
//! `logprob` (sum over the continuation), `is_greedy` (null: the
//! multiple-choice tasks read only the log-likelihood), `n_ctx`, `n_cont`.

use crate::model::EmbryoGpu;
use crate::train::load_checkpoint;
use cortiq_core::format::CmfModel;
use std::io::Write as _;
use std::path::Path;

/// lm-eval `HFLM._encode_pair` on strings: the context without its trailing
/// whitespace, the continuation with it in front (as `cortiq loglik`).
fn hflm_move_spaces(context: &str, continuation: &str) -> (String, String) {
    let trimmed = context.trim_end();
    let spaces = &context[trimmed.len()..];
    (trimmed.to_string(), format!("{spaces}{continuation}"))
}

/// Context and continuation ids: `whole = enc(ctx + cont)`, `ctx = enc(ctx)`,
/// `cont = whole[len(ctx)..]`; an empty context is the end-of-text token.
fn encode_pair(
    encode: &dyn Fn(&str) -> Vec<u32>,
    context: &str,
    continuation: &str,
    eot: Option<u32>,
    separate: bool,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    if context.is_empty() {
        let e = eot.ok_or_else(|| anyhow::anyhow!("an empty context needs an end-of-text token"))?;
        return Ok((vec![e], encode(continuation)));
    }
    if separate {
        return Ok((encode(context), encode(continuation)));
    }
    let (ctx, cont) = hflm_move_spaces(context, continuation);
    let whole = encode(&format!("{ctx}{cont}"));
    let ctx_ids = encode(&ctx);
    Ok((ctx_ids.clone(), whole.get(ctx_ids.len()..).unwrap_or(&[]).to_vec()))
}

/// Scores `requests` (JSONL, `context` + `continuation`) with the weights of
/// `ckpt` and the tokenizer of `vocab` (a `.cmf` with the same vocabulary),
/// `rows` requests per forward.
/// `split`: `hflm` (lm-eval's pair encoding) or `separate` (the two strings encoded apart), as `cortiq loglik`.
pub fn run(ckpt: &Path, vocab: &Path, requests: &Path, out: &Path, rows: usize, split: &str) -> anyhow::Result<()> {
    let separate = match split {
        "hflm" | "lm-eval" => false,
        "separate" => true,
        other => anyhow::bail!("--split {other}: expected hflm | separate"),
    };
    let t0 = std::time::Instant::now();
    let ck = load_checkpoint(ckpt)?;
    let model = CmfModel::open(vocab)?;
    let tok = crate::skill::runtime_tokenizer(
        model.vocab.as_ref().ok_or_else(|| anyhow::anyhow!("{} has no VOCAB section", vocab.display()))?,
    )?;
    let eot = tok.token_to_id("<|endoftext|>").or(tok.eos_token_id);
    let mut reqs = Vec::new();
    for (n, line) in std::fs::read_to_string(requests)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut o: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("requests line {}: {e}", n + 1))?;
        let take = |o: &mut serde_json::Map<String, serde_json::Value>, k: &str| -> anyhow::Result<String> {
            match o.remove(k) {
                Some(serde_json::Value::String(s)) => Ok(s),
                _ => anyhow::bail!("requests line {}: no string `{k}`", n + 1),
            }
        };
        let c = take(&mut o, "context")?;
        let k = take(&mut o, "continuation")?;
        anyhow::ensure!(!k.is_empty(), "requests line {}: empty continuation", n + 1);
        let (ci, ki) = encode_pair(&|s| tok.encode(s), &c, &k, eot, separate)?;
        anyhow::ensure!(!ci.is_empty() && !ki.is_empty(), "request {n}: empty context or continuation ids");
        reqs.push((ci, ki, o));
    }
    let cfg = ck.cfg.clone();
    let mut logprob = vec![0f64; reqs.len()];
    // length buckets of 64: one instance per bucket, `rows` requests a forward
    let mut by_t: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, (c, k, _)) in reqs.iter().enumerate() {
        let len = c.len() + k.len();
        anyhow::ensure!(len <= 2048, "request {i}: {len} tokens, more than the scorer's 2048");
        by_t.entry(len.div_ceil(64) * 64).or_default().push(i);
    }
    for (&t, idx) in &by_t {
        let b = rows.min(idx.len()).max(1);
        let gpu = EmbryoGpu::new_eval_dropless(cfg.clone(), b, t, &ck.params)
            .ok_or_else(|| anyhow::anyhow!("no GPU device for the scorer (Metal / Vulkan / CUDA)"))?;
        gpu.set_desc(&ck.extras);
        gpu.desc_updates.set(false);
        gpu.anchor_fixed_window.set(true);
        let _exact = gpu.ctx().exact_f32();
        for chunk in idx.chunks(b) {
            let mut tokens = vec![0u32; b * t];
            let mut targets = vec![u32::MAX; b * t];
            for (r, &i) in chunk.iter().enumerate() {
                let (c, k, _) = &reqs[i];
                let row = &mut tokens[r * t..(r + 1) * t];
                row[..c.len()].copy_from_slice(c);
                row[c.len()..c.len() + k.len()].copy_from_slice(k);
                let last = row[c.len() + k.len() - 1];
                row[c.len() + k.len()..].fill(last);
                // position p predicts token p + 1: the continuation is predicted from c.len() − 1 on
                for (j, &id) in k.iter().enumerate() {
                    targets[r * t + c.len() - 1 + j] = id;
                }
            }
            let _ = gpu.eval_loss(&tokens, &targets);
            let per = gpu.per_position_loss();
            for (r, &i) in chunk.iter().enumerate() {
                let (c, k, _) = &reqs[i];
                let s = r * t + c.len() - 1;
                logprob[i] = -per[s..s + k.len()].iter().map(|&x| x as f64).sum::<f64>();
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    for (i, (c, k, echo)) in reqs.iter().enumerate() {
        let mut o = echo.clone();
        o.insert("i".into(), serde_json::json!(i));
        o.insert("logprob".into(), serde_json::json!(logprob[i]));
        o.insert("is_greedy".into(), serde_json::Value::Null);
        o.insert("n_ctx".into(), serde_json::json!(c.len()));
        o.insert("n_cont".into(), serde_json::json!(k.len()));
        writeln!(f, "{}", serde_json::Value::Object(o))?;
    }
    f.flush()?;
    eprintln!(
        "loglik: {} requests in {} length buckets on {} in {:.1} s ({})",
        reqs.len(),
        by_t.len(),
        if cfg!(target_os = "macos") { "Metal" } else if cfg!(feature = "cuda") { "CUDA" } else { "Vulkan" },
        t0.elapsed().as_secs_f64(),
        ckpt.display()
    );
    Ok(())
}
