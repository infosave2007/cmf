//! In-process A/B of the prefill chunk width: one loaded model, the chunk
//! switched between runs through `CMF_PREFILL_CHUNK` (read per call), the
//! arms rotated every round so drift on a shared card lands on all of them.
//!
//!   cargo run --release -p cortiq-cli --example prefill_ab -- \
//!       model.cmf text.txt <ctx> <chunks,comma,separated> [rounds] [gen]
//!
//! Per round and arm: `forward_ids` over a `ctx`-token prompt (what
//! `bench` reports as prefill) and the time to the first token of
//! `generate_from_ids` (what `serve` pays as TTFT). Chunk arm 0 = no
//! override (the engine's built-in default). With `gen` > 0 the
//! last round also decodes `gen` greedy tokens per arm and prints whether
//! each arm's continuation equals the first arm's.
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        anyhow::bail!("usage: prefill_ab model.cmf text.txt ctx chunks [rounds] [gen]");
    }
    let ctx: usize = a[3].parse()?;
    let chunks: Vec<usize> = a[4].split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
    let rounds: usize = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let n_gen: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let m = Arc::new(CmfModel::open_sharded(&a[1])?);
    let cfg = SamplerConfig {
        temperature: 0.0,
        seed: Some(42),
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut p = Pipeline::from_model(&m, cfg)?;
    let text = std::fs::read_to_string(&a[2])?;
    let mut ids = p.tokenizer.encode(&text);
    if ids.len() < ctx {
        anyhow::bail!("text tokenizes to {} < ctx {ctx}", ids.len());
    }
    ids.truncate(ctx);
    // Arm 0 = no override: the engine's own default for this model/card.
    let set = |c: usize| unsafe {
        if c == 0 {
            std::env::remove_var("CMF_PREFILL_CHUNK")
        } else {
            std::env::set_var("CMF_PREFILL_CHUNK", c.to_string())
        }
    };
    set(0);
    println!("built-in default chunk for this model here: {}", p.prefill_chunk());
    // Warm-up: shader compile + weight upload, every arm once.
    for &c in &chunks {
        set(c);
        p.reset_session();
        p.forward_ids(&ids[..ctx.min(1100)], None).map_err(anyhow::Error::msg)?;
    }
    let mut fwd: Vec<Vec<f64>> = vec![Vec::new(); chunks.len()];
    let mut ttft: Vec<Vec<f64>> = vec![Vec::new(); chunks.len()];
    for r in 0..rounds {
        for k in 0..chunks.len() {
            let i = (k + r) % chunks.len();
            set(chunks[i]);
            p.reset_session();
            let t0 = Instant::now();
            p.forward_ids(&ids, None).map_err(anyhow::Error::msg)?;
            let f = ctx as f64 / t0.elapsed().as_secs_f64();
            p.reset_session();
            let first: Arc<Mutex<Option<Instant>>> = Arc::default();
            let fc = first.clone();
            let t0 = Instant::now();
            p.generate_from_ids(
                &ids,
                1,
                None,
                Some(Box::new(move |_| {
                    fc.lock().unwrap().get_or_insert_with(Instant::now);
                    true
                })),
            )
            .map_err(anyhow::Error::msg)?;
            let t = first.lock().unwrap().map(|x| (x - t0).as_secs_f64()).unwrap_or(f64::NAN);
            println!(
                "round {r} chunk {:>5}: forward {f:7.1} tok/s | generate ttft {:6.3} s | resident {} MiB",
                chunks[i],
                t,
                cortiq_engine::gpu::resident_bytes() >> 20
            );
            fwd[i].push(f);
            ttft[i].push(t);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for i in 0..chunks.len() {
        println!(
            "SUMMARY ctx {ctx} chunk {:>5}: forward median {:7.1} tok/s | ttft median {:6.3} s",
            chunks[i],
            med(&mut fwd[i]),
            med(&mut ttft[i])
        );
    }
    // PREFILL_AB_DUMP=<dir>: the prompt ids, and per arm the greedy ids and
    // the last prompt position's logits (f32 LE), for an outside reference.
    let dump = std::env::var("PREFILL_AB_DUMP").ok();
    if let Some(d) = &dump {
        std::fs::create_dir_all(d)?;
        std::fs::write(format!("{d}/ids.json"), serde_json::to_string(&ids)?)?;
    }
    if n_gen > 0 {
        let mut base: Option<Vec<u32>> = None;
        for &c in &chunks {
            set(c);
            p.reset_session();
            let r = p.generate_from_ids(&ids, n_gen, None, None).map_err(anyhow::Error::msg)?;
            let out = r.token_ids.clone();
            if let Some(d) = &dump {
                p.reset_session();
                let lg = p.forward_ids(&ids, None).map_err(anyhow::Error::msg)?;
                let bytes: Vec<u8> = lg.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{d}/logits_{c}.f32"), bytes)?;
                std::fs::write(format!("{d}/greedy_{c}.json"), serde_json::to_string(&out)?)?;
            }
            let same = match &base {
                None => {
                    base = Some(out.clone());
                    "reference".to_string()
                }
                Some(b) => {
                    let n = b.iter().zip(&out).take_while(|(x, y)| x == y).count();
                    if *b == out {
                        "IDENTICAL".to_string()
                    } else {
                        format!("DIFFERS after {n} of {} tokens", b.len())
                    }
                }
            };
            println!(
                "GREEDY chunk {c:>5}: {same} | {:?}",
                r.text.chars().take(160).collect::<String>()
            );
        }
    }
    Ok(())
}
