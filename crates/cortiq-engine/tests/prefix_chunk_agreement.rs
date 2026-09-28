//! A 6000-token prefix scored on the CPU path must not depend on how the
//! prefix is chunked: per-position logits from the per-token walk
//! (`forward_span`) and from chunked prefills of 1024 / 2048 / 4096
//! positions (`prefill_span_ids`), and the `ppl` path's own 128-position
//! chunks (`nll_ids_from`), on a synthetic vmf_phase + bounded genome at
//! the Embryo-0 width.  Written while separating the ~4k perplexity cliff
//! of the 500-step exports from the runtime: a chunk hand-off that lost
//! the vmf state, the conv ring, the bounded ring or the position would
//! show up here as a chunk-size-dependent difference.  Beyond the
//! trained window the only difference allowed is float accumulation
//! order of the batched GEMM panels.

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

#[path = "common/embryo_synth.rs"]
mod embryo_synth;

fn per_token(p: &mut Pipeline, ids: &[u32], upto: usize) -> Vec<Vec<f32>> {
    p.kv_cache.clear();
    p.clear_history();
    let mut out = Vec::with_capacity(ids.len());
    for (pos, &id) in ids.iter().enumerate() {
        let e = p.embed_id(id);
        let h = p.forward_span(&e, pos, 0, upto, None).unwrap();
        out.push(p.logits_from_hidden(&h));
    }
    out
}

fn chunked(p: &mut Pipeline, ids: &[u32], upto: usize, chunk: usize) -> Vec<Vec<f32>> {
    p.kv_cache.clear();
    p.clear_history();
    let hs = p.hidden_size;
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    while pos < ids.len() {
        let end = (pos + chunk).min(ids.len());
        let hb = p.prefill_span_ids(&ids[pos..end], pos, upto, None).unwrap();
        for k in 0..end - pos {
            out.push(p.logits_from_hidden(&hb[k * hs..(k + 1) * hs]));
        }
        pos = end;
    }
    out
}

/// (max |Δ logit| over all positions, position of that max)
fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, usize) {
    assert_eq!(a.len(), b.len());
    let mut m = (0f64, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        for (u, v) in x.iter().zip(y) {
            let d = ((*u as f64) - (*v as f64)).abs();
            if d > m.0 {
                m = (d, i);
            }
        }
    }
    m
}

fn nll_of(logits: &[Vec<f32>], ids: &[u32]) -> f64 {
    let mut nll = 0f64;
    for (pos, lg) in logits.iter().enumerate().take(ids.len() - 1) {
        let max = lg.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let lse = lg.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() + max as f64;
        nll += lse - lg[ids[pos + 1] as usize] as f64;
    }
    nll / (ids.len() - 1) as f64
}

#[test]
fn six_thousand_token_prefix_agrees_across_chunk_sizes() {
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::set_var("CMF_MAX_SEQ", "8192");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
    }
    let g = embryo_synth::SynthGeom::bounded_small();
    let path = embryo_synth::synth_genome_path(&g);
    let model = Arc::new(CmfModel::open(&path).expect("open synthetic genome"));
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("load");
    assert!(p.bounded_native());
    let upto = p.num_layers - 1;
    let ids = embryo_synth::synth_ids(6000, 41, g.vocab);
    let t0 = std::time::Instant::now();
    let base = per_token(&mut p, &ids, upto);
    let t_tok = t0.elapsed().as_secs_f64();
    let base_nll = nll_of(&base, &ids);
    let mut worst = 0f64;
    for chunk in [1024usize, 2048, 4096] {
        let t1 = std::time::Instant::now();
        let c = chunked(&mut p, &ids, upto, chunk);
        let (d, at) = max_abs_diff(&base, &c);
        let nll = nll_of(&c, &ids);
        eprintln!(
            "chunk {chunk}: max|Δ logit| {d:.3e} at pos {at}, |Δ mean nll| {:.3e} ({:.1}s vs per-token {t_tok:.1}s)",
            (nll - base_nll).abs(),
            t1.elapsed().as_secs_f64()
        );
        assert!(d <= 1e-4, "chunk {chunk}: per-position logits diverge: {d:.3e} at {at}");
        assert!((nll - base_nll).abs() <= 1e-5, "chunk {chunk}: mean nll {nll} vs {base_nll}");
        worst = worst.max(d);
    }
    // The ppl path itself (128-position chunks, batched head).
    let (n, cnt) = p.nll_ids_from(&ids, 0).expect("nll scoring");
    assert_eq!(cnt, ids.len() - 1);
    let ppl_nll = n / cnt as f64;
    eprintln!(
        "ppl path: mean nll {ppl_nll:.6} vs per-token {base_nll:.6} (|Δ| {:.3e}); worst chunk Δ logit {worst:.3e}",
        (ppl_nll - base_nll).abs()
    );
    assert!((ppl_nll - base_nll).abs() <= 1e-5, "ppl path {ppl_nll} vs per-token {base_nll}");
    // No position-keyed cliff in the operator itself: the mean NLL of the
    // last 1000 positions is within the noise of the first 1000 on random
    // weights (a runtime step at ~4k would show as a jump).
    let head = nll_of(&base[..1001], &ids[..1001]);
    let tail = {
        let lg = &base[5000..];
        let id = &ids[5000..];
        nll_of(lg, id)
    };
    eprintln!("mean nll positions 0..1000 {head:.4}, 5000..6000 {tail:.4}");
    assert!((tail - head).abs() < 1.0, "position-keyed step: {head} vs {tail}");
}
