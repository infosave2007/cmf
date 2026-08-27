//! The genome exported to .cmf and run by the RUNTIME (CPU pipeline:
//! vmf_phase mixer with κ, GQA anchor, resonance-routed experts + shared,
//! hierarchical head) must produce the trainer's log-probabilities.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::lcg_vec;
use cortiq_embryo::train::Checkpoint;

#[test]
fn runtime_matches_trainer_logprobs() {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    let (b, t) = (1usize, 64usize);
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 5);
    let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    let tokens: Vec<u32> = lcg_vec(21, t)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    // seed the descriptors from this batch and read them back — routing state
    // is part of the model; run one training forward with updates on
    let targets: Vec<u32> = tokens.iter().cycle().skip(1).take(t).cloned().collect();
    let mut gpu = gpu;
    let _ = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1e9); // lr 0: params unchanged, descriptors seeded
    gpu.desc_updates.set(false);
    let params = gpu.params_host();
    let extras: Vec<(String, Vec<f32>)> = gpu
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    let xf = gpu.forward_hidden(&tokens);
    // trainer-side hierarchical log-probs from xf
    let (h, v, ncl) = (cfg.hidden, cfg.vocab, cfg.head_clusters);
    let cs = v / ncl;
    let e = &params[lay.embed..lay.embed + v * h];
    let cm = &params[lay.head_clusters..lay.head_clusters + ncl * h];
    let logprobs = |x: &[f32]| -> Vec<f32> {
        let mut lc: Vec<f32> = (0..ncl)
            .map(|c| (0..h).map(|j| cm[c * h + j] * x[j]).sum())
            .collect();
        let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + lc.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
        for z in &mut lc {
            *z -= lse;
        }
        let mut out = vec![0.0f32; v];
        for c in 0..ncl {
            let lg: Vec<f32> = (0..cs)
                .map(|s| (0..h).map(|j| e[(c * cs + s) * h + j] * x[j]).sum())
                .collect();
            let bm = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bl = bm + lg.iter().map(|z| (z - bm).exp()).sum::<f32>().ln();
            for s in 0..cs {
                out[c * cs + s] = lc[c] + lg[s] - bl;
            }
        }
        out
    };
    // export
    let tok_json = {
        // a minimal byte-level tokenizer.json (vocab must cover the ids we feed)
        let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
        let mut counts = std::collections::HashMap::new();
        cortiq_embryo::tokenizer::count_words("hello world hello embryo", &re, &mut counts);
        cortiq_embryo::tokenizer::train(&counts, cfg.vocab, false).to_hf_json()
    };
    let ck = Checkpoint {
        cfg: cfg.clone(),
        step: 0,
        params: params.clone(),
        m: None,
        v: None,
        extras,
    };
    let dir = std::env::temp_dir().join(format!("embryo_parity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.cmf");
    cortiq_embryo::export::export(&ck, tok_json.as_bytes(), &path).expect("export");
    // runtime
    let model = std::sync::Arc::new(cortiq_core::format::CmfModel::open(&path).expect("open cmf"));
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
        &model,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .expect("pipeline");
    let mut worst = 0.0f32;
    for pos in [0usize, 1, 5, 17, 40, 63] {
        let want = logprobs(&xf[pos * h..(pos + 1) * h]);
        let got = pipe.prefill_next_logits(&tokens[..=pos], None);
        assert_eq!(got.len(), v);
        let d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "pos {pos}: max|Δ logprob| = {d:.3e}  (argmax trainer {} runtime {})",
            want.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0,
            got.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        );
        worst = worst.max(d);
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        worst < 2e-3,
        "runtime vs trainer log-probs differ: max {worst}"
    );
}
