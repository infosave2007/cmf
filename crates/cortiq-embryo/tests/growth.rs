//! Growth as records: E → E+1 experts per layer, only the new experts
//! train, the export of the grown genome keeps every old tensor byte-
//! identical (new records only), and the runtime loads and runs it.
#![cfg(target_os = "macos")]

use cortiq_embryo::growth::{GrowArgs, grow_experts, train_new_experts};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::train::{Checkpoint, Shard};

#[test]
fn growth_appends_experts_without_touching_old_records() {
    let Some(_) = cortiq_embryo::metal::ctx() else { return };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 2;
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 21);
    let mut text = Vec::new();
    for f in ["../../README.md", "../../docs/CMF_V2_SPEC.md", "../../docs/SKILLS.md", "../../docs/COMPARISON.md"] {
        if let Ok(b) = std::fs::read(f) {
            text.extend_from_slice(&b);
        }
    }
    let corpus = Shard::from_bytes(&text);
    // a few training steps to seed descriptors and give the routing shape
    let mut gpu = EmbryoGpu::new(cfg.clone(), 4, 64, &params).unwrap();
    for s in 0..3 {
        let tk: Vec<u32> = corpus.tokens[s * 256..s * 256 + 256].iter().map(|&x| x as u32).collect();
        let tg: Vec<u32> = corpus.tokens[s * 256 + 1..s * 256 + 257].iter().map(|&x| x as u32).collect();
        gpu.train_step(&tk, &tg, 1e-3, 0.0, 1.0);
    }
    let params = gpu.params_host();
    let extras: Vec<(String, Vec<f32>)> = gpu.desc_host().into_iter().map(|(n, x)| (n.to_string(), x)).collect();
    drop(gpu);
    let ck = Checkpoint { cfg: cfg.clone(), step: 3, params, m: None, v: None, extras };
    // grow
    let (grown, sources) = grow_experts(&ck, 1e-3, 0.1, 5);
    assert_eq!(grown.cfg.experts, 3);
    assert_eq!(sources.len(), cfg.layers);
    // old tensors identical inside the arena (by name)
    let lay1 = Layout::new(&grown.cfg);
    let by0: std::collections::HashMap<&str, (usize, usize)> = lay.names.iter().map(|(n, o, l)| (n.as_str(), (*o, *l))).collect();
    for (name, o1, l1) in &lay1.names {
        if let Some((o0, l0)) = by0.get(name.as_str()) {
            assert_eq!(&ck.params[*o0..*o0 + l0], &grown.params[*o1..*o1 + l1], "{name} changed by growth");
        }
    }
    // train only the new experts
    let a = GrowArgs { steps: 20, lr: 1e-3, batch: 4, seq: 64, eval_every: 10, seed: 3 };
    let (trained, l0, l1) = train_new_experts(&grown, &corpus, &a, &|| false).unwrap();
    eprintln!("growth: held-out {l0:.4} → {l1:.4}; sources {sources:?}");
    for (name, o1, l1n) in &lay1.names {
        if let Some((o0, l0n)) = by0.get(name.as_str()) {
            assert_eq!(&ck.params[*o0..*o0 + l0n], &trained.params[*o1..*o1 + l1n], "{name} changed by training the new experts");
        }
    }
    // export both, compare tensor bytes, load the grown one in the runtime
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    cortiq_embryo::tokenizer::count_words("hello world", &re, &mut counts);
    let tok_json = cortiq_embryo::tokenizer::train(&counts, cfg.vocab, false).to_hf_json();
    let dir = std::env::temp_dir().join(format!("embryo_grow_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p0 = dir.join("before.cmf");
    let p1 = dir.join("after.cmf");
    cortiq_embryo::export::export(&ck, tok_json.as_bytes(), &p0).unwrap();
    cortiq_embryo::export::export(&trained, tok_json.as_bytes(), &p1).unwrap();
    let m0 = cortiq_core::format::CmfModel::open(&p0).unwrap();
    let m1 = cortiq_core::format::CmfModel::open(&p1).unwrap();
    let mut same = 0;
    for t in &m0.tensors {
        let _t1 = m1.tensor(&t.name).expect("old tensor kept");
        assert_eq!(m0.tensor_bytes(&t.name).unwrap(), m1.tensor_bytes(&t.name).unwrap(), "{}: bytes changed", t.name);
        same += 1;
    }
    let added = m1.tensors.len() - m0.tensors.len();
    eprintln!("export: {same} old tensors byte-identical, {added} new records (experts.2.* + descriptors)");
    assert!(m1.tensor("model.layers.0.mlp.experts.2.gate_proj.weight").is_some());
    assert!(m1.tensor("model.layers.0.mlp.experts.2.desc.mu").is_some());
    assert_eq!(m1.header.arch.moe.as_ref().unwrap().num_experts, 3);
    let m1 = std::sync::Arc::new(m1);
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(&m1, cortiq_engine::sampler::SamplerConfig::default()).unwrap();
    let ids: Vec<u32> = corpus.tokens[100..132].iter().map(|&x| x as u32).collect();
    let lg = pipe.prefill_next_logits(&ids, None);
    assert!(lg.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(&dir);
}
