//! Skill bake end to end on a tiny genome: masks + FCD on a byte-level
//! corpus, appended to the exported .cmf — old tensors byte-identical, the
//! runtime loads the skill overlay, held-out loss improves.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};
use cortiq_embryo::skill::{BakeArgs, append_to_cmf, bake};
use cortiq_embryo::train::{Checkpoint, Shard};

#[test]
fn bake_appends_a_byte_identical_skill_the_runtime_loads() {
    let Some(_) = cortiq_embryo::metal::ctx() else { return };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    cfg.vocab = 4096;
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 9);
    // corpus: the repo docs as bytes (ids < 256 ⊂ vocab)
    let mut text = Vec::new();
    for f in ["../../README.md", "../../docs/CMF_V2_SPEC.md", "../../docs/SKILLS.md", "../../docs/COMPARISON.md"] {
        if let Ok(b) = std::fs::read(f) {
            text.extend_from_slice(&b);
        }
    }
    let corpus = Shard::from_bytes(&text);
    assert!(corpus.tokens.len() > 30_000);
    // seed descriptors with one training step (lr 0), as a birth would have
    let mut gpu = cortiq_embryo::model::EmbryoGpu::new(cfg.clone(), 4, 64, &params).unwrap();
    let tk: Vec<u32> = corpus.tokens[..256].iter().map(|&x| x as u32).collect();
    let tg: Vec<u32> = corpus.tokens[1..257].iter().map(|&x| x as u32).collect();
    let _ = gpu.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    let extras: Vec<(String, Vec<f32>)> = gpu.desc_host().into_iter().map(|(n, x)| (n.to_string(), x)).collect();
    drop(gpu);
    let ck = Checkpoint { cfg: cfg.clone(), step: 0, params, m: None, v: None, extras };
    // export the base
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    cortiq_embryo::tokenizer::count_words("hello world", &re, &mut counts);
    let tok_json = cortiq_embryo::tokenizer::train(&counts, cfg.vocab, false).to_hf_json();
    let dir = std::env::temp_dir().join(format!("embryo_skill_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base.cmf");
    cortiq_embryo::export::export(&ck, tok_json.as_bytes(), &base).unwrap();
    // bake
    let a = BakeArgs {
        id: "docs".into(),
        layers: vec![0, 1],
        steps_a: 24,
        steps_b: 12,
        lr_a: 3e-2,
        lr_b: 1e-3,
        l1: 2e-3,
        tau: 0.5,
        eval_every: 12,
        batch: 4,
        seq: 64,
        phi_layer: 1,
        phi_len: 32,
        rank: 4,
        seed: 3,
    };
    let (tensors, sel, kept, (l0, la, lb)) = bake(&ck, &corpus, &a).expect("bake");
    eprintln!("held-out: base {l0:.4} → mask {la:.4} → mask+fcd {lb:.4}; kept {kept:?}");
    assert_eq!(tensors.len(), 6);
    assert!(lb <= l0 + 1e-3, "the skill must not be worse than the base on its own held-out ({lb} vs {l0})");
    assert_eq!(sel.rank, 4);
    // append
    let out = dir.join("skilled.cmf");
    let unchanged = append_to_cmf(&base, &out, "docs", &[0, 1], &tensors, sel, serde_json::json!({})).unwrap();
    let m0 = cortiq_core::format::CmfModel::open(&base).unwrap();
    let m1 = cortiq_core::format::CmfModel::open(&out).unwrap();
    assert_eq!(unchanged, m0.tensors.len());
    // every base tensor: same bytes (hash + payload)
    for t in &m0.tensors {
        let t1 = m1.tensor(&t.name).expect("base tensor kept");
        assert_eq!(t.hash, t1.hash, "{}: hash changed", t.name);
        assert_eq!(m0.tensor_bytes(&t.name).unwrap(), m1.tensor_bytes(&t.name).unwrap(), "{}: bytes changed", t.name);
    }
    assert!(m1.tensor("skill.docs.model.layers.1.mlp.shared_expert.down_proj.weight").is_some());
    assert_eq!(m1.header.skills.len(), 1);
    // runtime: base vs skill overlay produce different logits (the skill is live)
    let m1 = std::sync::Arc::new(m1);
    let sc = cortiq_engine::sampler::SamplerConfig::default();
    let mut p_base = cortiq_engine::pipeline::Pipeline::from_model(&m1, sc.clone()).unwrap();
    let mut p_skill = cortiq_engine::pipeline::Pipeline::from_model_with_skill(&m1, sc, Some("docs")).unwrap();
    let ids: Vec<u32> = corpus.tokens[1000..1032].iter().map(|&x| x as u32).collect();
    let lb0 = p_base.prefill_next_logits(&ids, None);
    let lb1 = p_skill.prefill_next_logits(&ids, None);
    let d = lb0.iter().zip(&lb1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("runtime: max|Δ logits| base vs skill = {d:.3e}");
    assert!(d > 1e-4, "the skill overlay must change the runtime's output");
    let _ = std::fs::remove_dir_all(&dir);
}
