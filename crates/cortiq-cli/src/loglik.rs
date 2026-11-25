//! `cortiq loglik` — the `loglikelihood` request of lm-evaluation-harness
//! served by the engine: for every `(context, continuation)` pair, the
//! summed log-prob of the continuation tokens given the context and
//! whether greedy decoding would have produced exactly those tokens
//! (`is_greedy`).
//!
//! Input: JSONL `{"context": str, "continuation": str}` per line (any
//! other field, e.g. an `id`, is echoed back). Output: one JSONL line per
//! request, in input order: `{"i", "logprob", "is_greedy", "n_ctx",
//! "n_cont", …echo}`.
//!
//! Tokenization follows lm-eval's `HFLM._encode_pair` (`--split hflm`,
//! the default): trailing whitespace of the context moves to the front of
//! the continuation, `whole = enc(context + continuation)`,
//! `ctx = enc(context)`, `cont = whole[len(ctx)..]`; an empty context is
//! the end-of-text token. `--split separate` encodes the two strings
//! independently. No BOS (the Embryo tokenizer declares none; `--bos`
//! prepends one when the tokenizer has it).
//!
//! `--frame chat` renders the context as a cmf-im-v1 user turn and scores
//! the continuation (leading whitespace trimmed) as the assistant answer;
//! the default `raw` scores plain text, as lm-eval does without
//! `--apply_chat_template`.
//!
//! Scoring. `--scorer device`: every `context + continuation` runs on the
//! CUDA or wgpu batch graph (`Pipeline::score_ids_device`): the context as
//! prompt chunks, the continuation rows in short verify windows whose final
//! norm and lm_head run on the card; the host reads back only those rows'
//! logits. `--scorer shared`: consecutive requests with the same context
//! tokens form a group; the context is forwarded once, the continuations
//! are laid out as a token trie and walked depth-first with the host KV
//! state rolled back between siblings — the walk of `probe-choice`,
//! extended with the greedy check (per-op on the host route: the KV
//! snapshot lives on the host). `--scorer auto`, the default: `device`
//! when the backend and the model admit it, `shared` otherwise.
//! `--scorer full` re-forwards `context + continuation` per request
//! through the engine's `nll_ids_from` (the slow cross-check; `is_greedy`
//! is then null).
//!
//! Always the BACKBONE pipeline (no skill lane, no router, no lookup
//! table); `CMF_GROWTH=off` keeps growth records out of the forward.

use crate::choice::{log_sum_exp, Trie};
use anyhow::Context;
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::io::Write as _;
use std::sync::Arc;

pub struct LoglikArgs<'a> {
    pub model: &'a str,
    pub requests_jsonl: &'a str,
    pub out: Option<&'a str>,
    /// `hflm` | `separate`.
    pub split: &'a str,
    /// `raw` | `chat`.
    pub frame: &'a str,
    /// `auto` | `device` | `shared` | `full`.
    pub scorer: &'a str,
    pub bos: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub context: String,
    pub continuation: String,
    /// The input object minus `context` / `continuation`, echoed back.
    pub echo: serde_json::Map<String, serde_json::Value>,
}

pub fn parse_requests(text: &str) -> anyhow::Result<Vec<Request>> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("requests jsonl line {}", ln + 1))?;
        let mut obj = match v {
            serde_json::Value::Object(o) => o,
            _ => anyhow::bail!("requests jsonl line {}: expected an object", ln + 1),
        };
        let take = |o: &mut serde_json::Map<String, serde_json::Value>, k: &str| match o.remove(k) {
            Some(serde_json::Value::String(s)) => Ok(s),
            _ => Err(anyhow::anyhow!(
                "requests jsonl line {}: no string \"{k}\"",
                ln + 1
            )),
        };
        let context = take(&mut obj, "context")?;
        let continuation = take(&mut obj, "continuation")?;
        anyhow::ensure!(
            !continuation.is_empty(),
            "requests jsonl line {}: empty continuation",
            ln + 1
        );
        out.push(Request {
            context,
            continuation,
            echo: obj,
        });
    }
    anyhow::ensure!(!out.is_empty(), "request set is empty");
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Hflm,
    Separate,
}

impl Split {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hflm" | "lm-eval" => Ok(Self::Hflm),
            "separate" => Ok(Self::Separate),
            other => anyhow::bail!("--split {other}: expected hflm | separate"),
        }
    }
}

/// lm-eval `HFLM._encode_pair` on strings: the context without its
/// trailing whitespace and the continuation with it in front.
pub fn hflm_move_spaces(context: &str, continuation: &str) -> (String, String) {
    let trimmed = context.trim_end();
    let spaces = &context[trimmed.len()..];
    (trimmed.to_string(), format!("{spaces}{continuation}"))
}

/// Context and continuation token ids of one request.
pub fn encode_pair(
    encode: &dyn Fn(&str) -> Vec<u32>,
    context: &str,
    continuation: &str,
    split: Split,
    eot: Option<u32>,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    if context.is_empty() {
        let e = eot.ok_or_else(|| {
            anyhow::anyhow!("an empty context needs an end-of-text token; the tokenizer has none")
        })?;
        return Ok((vec![e], encode(continuation)));
    }
    match split {
        Split::Separate => Ok((encode(context), encode(continuation))),
        Split::Hflm => {
            let (ctx, cont) = hflm_move_spaces(context, continuation);
            let whole = encode(&format!("{ctx}{cont}"));
            let ctx_ids = encode(&ctx);
            let cont_ids = whole.get(ctx_ids.len()..).unwrap_or(&[]).to_vec();
            Ok((ctx_ids, cont_ids))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub logprob: f64,
    pub is_greedy: Option<bool>,
}

/// Index of the first maximum (torch `argmax` on ties).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Depth-first trie walk from `node`, whose next-token `logits` were just
/// produced at `pos`: every edge adds its log-prob to the continuations
/// below it and clears their greedy flag unless it is the argmax token.
#[allow(clippy::too_many_arguments)]
fn walk(
    p: &mut Pipeline,
    trie: &Trie,
    node: usize,
    logits: &[f32],
    pos: usize,
    vocab: usize,
    lp: &mut [f64],
    greedy: &mut [bool],
) {
    let kids = &trie.children[node];
    if kids.is_empty() {
        return;
    }
    let lg = &logits[..vocab.min(logits.len())];
    let lse = log_sum_exp(lg);
    let top = argmax(lg) as u32;
    let mut snap = None;
    let mut dirty = false;
    for &(tok, child) in kids {
        let l = lg[tok as usize] as f64 - lse;
        for &m in &trie.members[child] {
            lp[m] += l;
            if tok != top {
                greedy[m] = false;
            }
        }
        if trie.children[child].is_empty() {
            continue;
        }
        if snap.is_none() {
            snap = Some(p.kv_cache.layers.clone());
        }
        if dirty {
            p.kv_cache
                .layers
                .clone_from(snap.as_ref().expect("snapshot taken"));
        }
        let next = p.decode_step_logits(tok, pos);
        dirty = true;
        walk(p, trie, child, &next, pos + 1, vocab, lp, greedy);
    }
}

/// One context forward, then every continuation off the shared state.
pub fn score_group(
    p: &mut Pipeline,
    ctx: &[u32],
    conts: &[Vec<u32>],
) -> anyhow::Result<Vec<Scored>> {
    let trie = Trie::build(conts);
    p.reset_session();
    let logits = p
        .forward_ids(ctx, None)
        .map_err(|e| anyhow::anyhow!("forward_ids: {e}"))?;
    let vocab = p.vocab_size;
    let mut lp = vec![0f64; conts.len()];
    let mut greedy = vec![true; conts.len()];
    walk(p, &trie, 0, &logits, ctx.len(), vocab, &mut lp, &mut greedy);
    Ok(lp
        .into_iter()
        .zip(greedy)
        .map(|(logprob, g)| Scored {
            logprob,
            is_greedy: Some(g),
        })
        .collect())
}

/// `context + continuation` through `nll_ids_from` (one fresh forward).
pub fn score_full(p: &mut Pipeline, ctx: &[u32], cont: &[u32]) -> anyhow::Result<Scored> {
    let mut all = Vec::with_capacity(ctx.len() + cont.len());
    all.extend_from_slice(ctx);
    all.extend_from_slice(cont);
    p.reset_session();
    let (nll, cnt) = p
        .nll_ids_from(&all, ctx.len() - 1)
        .map_err(|e| anyhow::anyhow!(e))?;
    anyhow::ensure!(
        cnt == cont.len(),
        "nll_ids_from scored {cnt} positions for a {}-token continuation",
        cont.len()
    );
    Ok(Scored {
        logprob: -nll,
        is_greedy: None,
    })
}

/// `context + continuation` on the device batch graph (one fresh
/// sequence); `Ok(None)` when the device route does not take the pipeline.
pub fn score_device(p: &mut Pipeline, ctx: &[u32], cont: &[u32]) -> anyhow::Result<Option<Scored>> {
    let mut all = Vec::with_capacity(ctx.len() + cont.len());
    all.extend_from_slice(ctx);
    all.extend_from_slice(cont);
    let Some(rows) = p
        .score_ids_device(&all, ctx.len() - 1)
        .map_err(|e| anyhow::anyhow!(e))?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        rows.len() == cont.len(),
        "the device scorer scored {} positions for a {}-token continuation",
        rows.len(),
        cont.len()
    );
    Ok(Some(Scored {
        logprob: rows.iter().map(|r| r.logprob).sum(),
        is_greedy: Some(rows.iter().all(|r| r.greedy)),
    }))
}

/// Runs of consecutive equal contexts: `(start, end)` index ranges.
pub fn context_groups(ctxs: &[Vec<u32>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0usize;
    for i in 1..=ctxs.len() {
        if i == ctxs.len() || ctxs[i] != ctxs[s] {
            out.push((s, i));
            s = i;
        }
    }
    out
}

pub fn cmd_loglik(a: LoglikArgs<'_>) -> anyhow::Result<()> {
    let split = Split::parse(a.split)?;
    let chat = match a.frame.trim().to_ascii_lowercase().as_str() {
        "raw" => false,
        "chat" => true,
        other => anyhow::bail!("--frame {other}: expected raw | chat"),
    };
    let mut scorer = a.scorer.trim().to_ascii_lowercase();
    anyhow::ensure!(
        matches!(scorer.as_str(), "auto" | "device" | "shared" | "full"),
        "--scorer {scorer}: expected auto | device | shared | full"
    );
    let reqs = parse_requests(
        &std::fs::read_to_string(a.requests_jsonl).with_context(|| a.requests_jsonl.to_string())?,
    )?;
    let model = Arc::new(CmfModel::open_sharded(a.model).with_context(|| a.model.to_string())?);
    let mut p = Pipeline::from_model_with_skill(&model, SamplerConfig::default(), None)
        .context("loading the backbone pipeline")?;
    // `auto` takes the device batch graph where the pipeline admits it; the
    // host scorers refuse the token graph (their KV snapshot is the host's)
    if matches!(scorer.as_str(), "auto" | "device") {
        match p.device_score_decline() {
            None => scorer = "device".into(),
            Some(why) if scorer == "device" => anyhow::bail!("--scorer device: {why}"),
            Some(why) => {
                eprintln!("loglik: the device scorer does not take this pipeline ({why}); host trie walk");
                scorer = "shared".into();
            }
        }
    }
    if scorer != "device" {
        p.mark_graph_refused();
    }
    let tok = p.tokenizer.clone();
    let eot = tok.token_to_id("<|endoftext|>").or(tok.eos_token_id);
    let encode = |s: &str| tok.encode(s);

    let mut ctxs = Vec::with_capacity(reqs.len());
    let mut conts = Vec::with_capacity(reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        let (mut c, k) = if chat {
            let framed = crate::utility::chat_prefix(r.context.trim_end());
            (encode(&framed), encode(r.continuation.trim_start()))
        } else {
            encode_pair(&encode, &r.context, &r.continuation, split, eot)?
        };
        if a.bos {
            c = tok.with_bos(c);
        }
        anyhow::ensure!(
            !c.is_empty(),
            "request {i}: empty context after tokenization"
        );
        anyhow::ensure!(
            !k.is_empty(),
            "request {i}: continuation tokenizes to nothing"
        );
        if let Some(&bad) = c.iter().chain(&k).find(|&&x| x as usize >= p.vocab_size) {
            anyhow::bail!(
                "request {i}: token {bad} outside the vocab ({})",
                p.vocab_size
            );
        }
        ctxs.push(c);
        conts.push(k);
    }
    let groups = context_groups(&ctxs);
    let growth = cortiq_engine::loader::growth_mode().label();
    eprintln!(
        "loglik: {} | {} requests, {} context groups | split={} frame={} scorer={} bos={} | growth={growth} CMF_GPU={}",
        a.model,
        reqs.len(),
        groups.len(),
        a.split,
        a.frame,
        scorer,
        a.bos,
        std::env::var("CMF_GPU").unwrap_or_else(|_| "unset".into())
    );

    let mut sink: Box<dyn std::io::Write> = match a.out {
        Some(path) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(path).with_context(|| path.to_string())?,
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    let t0 = std::time::Instant::now();
    let mut last_report = 0usize;
    for &(s, e) in &groups {
        let mut scored: Vec<Scored> = Vec::with_capacity(e - s);
        if scorer == "device" {
            for i in s..e {
                match score_device(&mut p, &ctxs[i], &conts[i])? {
                    Some(sc) => scored.push(sc),
                    // the batch graph refused the stack before anything ran:
                    // `auto` keeps the host walk from the first request on
                    None if i == 0 && a.scorer.trim().eq_ignore_ascii_case("auto") => {
                        eprintln!("loglik: the batch graph declined this model; host trie walk");
                        scorer = "shared".into();
                        p.mark_graph_refused();
                        break;
                    }
                    None => anyhow::bail!("request {i}: the device batch graph declined"),
                }
            }
        }
        if scorer != "device" {
            scored = match scorer.as_str() {
                "full" => (s..e)
                    .map(|i| score_full(&mut p, &ctxs[i], &conts[i]))
                    .collect::<anyhow::Result<_>>()?,
                _ => score_group(&mut p, &ctxs[s], &conts[s..e])?,
            };
        }
        for (j, sc) in scored.into_iter().enumerate() {
            let i = s + j;
            let mut o = reqs[i].echo.clone();
            o.insert("i".into(), serde_json::json!(i));
            o.insert("logprob".into(), serde_json::json!(sc.logprob));
            o.insert("is_greedy".into(), serde_json::json!(sc.is_greedy));
            o.insert("n_ctx".into(), serde_json::json!(ctxs[i].len()));
            o.insert("n_cont".into(), serde_json::json!(conts[i].len()));
            writeln!(sink, "{}", serde_json::Value::Object(o))?;
        }
        let done = e;
        if done - last_report >= 500 || done == reqs.len() {
            last_report = done;
            eprintln!(
                "loglik: {done}/{} requests, {:.1} s",
                reqs.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    sink.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hflm_moves_trailing_context_space_into_the_continuation() {
        assert_eq!(
            hflm_move_spaces("Question: x\nAnswer: ", "yes"),
            ("Question: x\nAnswer:".to_string(), " yes".to_string())
        );
        assert_eq!(
            hflm_move_spaces("Answer:", " yes"),
            ("Answer:".to_string(), " yes".to_string())
        );
    }

    #[test]
    fn encode_pair_slices_the_whole_encoding_after_the_context() {
        // A toy byte tokenizer: one id per byte.
        let enc = |s: &str| s.bytes().map(|b| b as u32).collect::<Vec<u32>>();
        let (c, k) = encode_pair(&enc, "ab ", "cd", Split::Hflm, Some(0)).unwrap();
        assert_eq!(c, vec![b'a' as u32, b'b' as u32]);
        assert_eq!(k, vec![b' ' as u32, b'c' as u32, b'd' as u32]);
        let (c, k) = encode_pair(&enc, "", "cd", Split::Hflm, Some(7)).unwrap();
        assert_eq!(c, vec![7]);
        assert_eq!(k, vec![b'c' as u32, b'd' as u32]);
        assert!(encode_pair(&enc, "", "cd", Split::Hflm, None).is_err());
    }

    #[test]
    fn groups_are_runs_of_equal_contexts() {
        let c = vec![vec![1], vec![1], vec![2], vec![1], vec![1], vec![1]];
        assert_eq!(context_groups(&c), vec![(0, 2), (2, 3), (3, 6)]);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 2.0, 1.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn requests_echo_extra_fields() {
        let r = parse_requests("{\"context\":\"a\",\"continuation\":\" b\",\"id\":3}\n\n").unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].continuation, " b");
        assert_eq!(r[0].echo["id"], 3);
        assert!(parse_requests("{\"context\":\"a\",\"continuation\":\"\"}").is_err());
    }
}
