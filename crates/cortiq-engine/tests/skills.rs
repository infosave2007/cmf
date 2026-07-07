//! Skills (spec §9, Patent 15 claims 1/2/15): shared backbone stored
//! once + per-skill replacement tensors read IN PLACE OF backbone
//! tensors. Gate for B1: substitution changes the forward, storage
//! scales as backbone + Σ deltas, unknown skill refuses loudly.

use base64::Engine as _;
use cortiq_core::format::{CmfHeader, CmfModel, SelectionDescriptor, SkillRecord, TensorSpec};
use cortiq_core::quant::f32_to_f16;
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType};
use cortiq_core::CMF_VERSION;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

const H: usize = 8;
const FFN: usize = 16;
const NH: usize = 2;
const HD: usize = 4;
const VOCAB: usize = 32;

fn arch() -> ModelArch {
    ModelArch {
        arch_name: "tiny".into(),
        hidden_size: H,
        intermediate_size: FFN,
        num_layers: 2,
        num_attention_heads: NH,
        num_kv_heads: 1,
        head_dim: HD,
        vocab_size: VOCAB,
        layer_types: vec![LayerType::FullAttention; 2],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 64,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
    }
}

fn synth(rows: usize, cols: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i * 13 + salt * 7) % 97) as f32 / 97.0 - 0.5) * scale)
        .collect()
}

fn spec(name: &str, rows: usize, cols: usize, data: Vec<f32>) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: cortiq_core::TensorDtype::F32,
        shape: vec![rows, cols],
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn spec1d(name: &str, n: usize, v: f32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: cortiq_core::TensorDtype::F32,
        shape: vec![n],
        data: std::iter::repeat(v).take(n).flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// Backbone + one skill ("boost") that replaces layer-0 gate_proj with
/// a ×2 copy. Returns (path, backbone-only bytes) for the scaling check.
fn write_swarm(path: &std::path::Path, with_skill: bool) -> u64 {
    let mut tensors = vec![
        spec("model.embed_tokens.weight", VOCAB, H, synth(VOCAB, H, 1, 0.4)),
        spec("lm_head.weight", VOCAB, H, synth(VOCAB, H, 2, 0.4)),
        spec1d("model.norm.weight", H, 1.0),
    ];
    for li in 0..2 {
        let p = format!("model.layers.{li}.");
        tensors.push(spec1d(&format!("{p}input_layernorm.weight"), H, 1.0));
        tensors.push(spec1d(&format!("{p}post_attention_layernorm.weight"), H, 1.0));
        tensors.push(spec(&format!("{p}self_attn.q_proj.weight"), NH * HD, H, synth(NH * HD, H, 10 + li, 0.3)));
        tensors.push(spec(&format!("{p}self_attn.k_proj.weight"), HD, H, synth(HD, H, 20 + li, 0.3)));
        tensors.push(spec(&format!("{p}self_attn.v_proj.weight"), HD, H, synth(HD, H, 30 + li, 0.3)));
        tensors.push(spec(&format!("{p}self_attn.o_proj.weight"), H, NH * HD, synth(H, NH * HD, 40 + li, 0.3)));
        tensors.push(spec(&format!("{p}mlp.gate_proj.weight"), FFN, H, synth(FFN, H, 50 + li, 0.3)));
        tensors.push(spec(&format!("{p}mlp.up_proj.weight"), FFN, H, synth(FFN, H, 60 + li, 0.3)));
        tensors.push(spec(&format!("{p}mlp.down_proj.weight"), H, FFN, synth(H, FFN, 70 + li, 0.3)));
    }
    let mut skills = Vec::new();
    if with_skill {
        let boosted: Vec<f32> = synth(FFN, H, 50, 0.3).iter().map(|v| v * 2.0).collect();
        tensors.push(spec("skill.boost.model.layers.0.mlp.gate_proj.weight", FFN, H, boosted));
        skills.push(SkillRecord {
            id: "boost".into(),
            name: Some("×2 gate".into()),
            layers: vec![0],
            selection: None,
            input_mask_task: None,
            quality: None,
        });
    }
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch: arch(),
        quant_type: QuantType::F32,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills,
        shard: None,
        calibration: None,
    };
    CmfModel::write(path, &header, &tensors, None, None).unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn greedy_logit_argmax(p: &mut Pipeline, ids: &[u32]) -> Vec<f32> {
    p.forward_ids(ids, None).unwrap()
}

#[test]
fn skill_replaces_in_place_and_scales_storage() {
    let dir = std::env::temp_dir().join(format!("cmf-skills-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base_path = dir.join("base.cmf");
    let swarm_path = dir.join("swarm.cmf");
    let base_len = write_swarm(&base_path, false);
    let swarm_len = write_swarm(&swarm_path, true);

    // Claim 15: storage = backbone + Σ deltas, not K × full model.
    let delta_bytes = (FFN * H * 4) as u64;
    let overhead = swarm_len - base_len;
    assert!(
        overhead >= delta_bytes && overhead < delta_bytes + 4096,
        "skill must cost ~its tensor bytes (+dir/header), got {overhead} vs {delta_bytes}"
    );

    let model = Arc::new(CmfModel::open(&swarm_path).unwrap());
    assert!(model.verify().is_empty());
    assert_eq!(model.skill_tensors("boost").count(), 1);

    let ids = [3u32, 7, 11];
    let mut backbone =
        Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let logits_backbone = greedy_logit_argmax(&mut backbone, &ids);
    drop(backbone);

    let mut overlaid =
        Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some("boost"))
            .unwrap();
    let logits_skill = greedy_logit_argmax(&mut overlaid, &ids);

    // Claim 1/3: the replacement is READ — the forward must change.
    let diff: f32 = logits_backbone
        .iter()
        .zip(&logits_skill)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff > 1e-4, "skill overlay must change logits (max diff {diff})");

    // Unknown skill: loud refusal, no silent backbone fallback.
    assert!(
        Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some("nope"))
            .is_err()
    );
}


fn b64f16(v: &[f32]) -> String {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Recon-argmin routing (B3): descriptors built from real probe φ of
/// two prompts must route each prompt back to its own skill.
#[test]
fn routing_picks_the_matching_skill() {
    let dir = std::env::temp_dir().join(format!("cmf-route-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("route.cmf");
    write_swarm(&path, true);

    // φ references from the backbone itself (phi_layer 1 = last layer).
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let seq_a = [3u32, 7, 11, 5];
    let seq_b = [20u32, 25, 30, 31];
    let phi_a = p.probe_phi(&seq_a, 1);
    let phi_b = p.probe_phi(&seq_b, 1);
    drop(p);

    // Shared rank-1 basis (any unit vector); means separate the skills.
    let mut basis = vec![0f32; H];
    basis[0] = 1.0;
    let sel = |mean: &[f32]| SelectionDescriptor {
        metric: "mse".into(),
        phi_layer: 1,
        mean: b64f16(mean),
        basis: b64f16(&basis),
        rank: 1,
    };
    let mk = |id: &str, mean: &[f32]| SkillRecord {
        id: id.into(),
        name: None,
        layers: vec![0],
        selection: Some(sel(mean)),
        input_mask_task: None,
        quality: None,
    };

    // Rewrite the file with routable skills (registry only differs).
    let mut header = model.header.clone();
    header.skills = vec![mk("alpha", &phi_a), mk("beta", &phi_b)];
    drop(model);
    let bytes_path = dir.join("route2.cmf");
    // Reuse write_swarm tensors by re-writing with the new header.
    let _ = write_swarm(&bytes_path, true);
    // Patch: simplest — write a fresh file with the same tensors + header.
    // (write_swarm builds its own header; easier to just re-open and route
    // against a file written with our header via CmfModel::write.)
    let model2 = Arc::new(CmfModel::open(&bytes_path).unwrap());
    let mut tensors = Vec::new();
    for t in &model2.tensors {
        tensors.push(TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model2.entry_bytes(t).to_vec(),
        });
    }
    let final_path = dir.join("route3.cmf");
    CmfModel::write(&final_path, &header, &tensors, None, None).unwrap();

    let model3 = Arc::new(CmfModel::open(&final_path).unwrap());
    let mut p3 = Pipeline::from_model(&model3, SamplerConfig::default()).unwrap();

    let ra = cortiq_engine::router::route(&model3, &mut p3, &seq_a);
    assert_eq!(ra[0].id, "alpha", "seq_a must route to alpha: {ra:?}");
    let rb = cortiq_engine::router::route(&model3, &mut p3, &seq_b);
    assert_eq!(rb[0].id, "beta", "seq_b must route to beta: {rb:?}");
    assert!(ra[0].error < ra[1].error && rb[0].error < rb[1].error);
}

/// Dynamic per-token skill switching (spec §9 runtime). Two invariants:
/// (1) `set_active_skill(Some(i))` gives BIT-IDENTICAL logits to loading
/// the pipeline statically with that skill; (2) switching back to None
/// restores the backbone exactly. This is the whole correctness claim
/// of the hysteresis router — the routing decision is separate.
#[test]
fn dynamic_skill_switch_matches_static_overlay() {
    let dir = std::env::temp_dir().join(format!("cmf-dyn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("swarm.cmf");
    write_swarm(&path, true); // one skill "boost" replacing layer-0 gate
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let boost_idx = model
        .header
        .skills
        .iter()
        .position(|s| s.id == "boost")
        .unwrap();

    let ids = [3u32, 7, 11, 5, 9];

    // Static references.
    let mut base = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let logits_base = greedy_logit_argmax(&mut base, &ids);
    let mut stat =
        Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some("boost")).unwrap();
    let logits_static = greedy_logit_argmax(&mut stat, &ids);

    // Dynamic: backbone → switch to boost → switch back.
    let mut dyn_p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    assert_eq!(dyn_p.active_skill(), None);
    let d0 = greedy_logit_argmax(&mut dyn_p, &ids);
    assert_eq!(d0, logits_base, "dynamic backbone must equal static backbone");

    dyn_p.set_active_skill(Some(boost_idx)).unwrap();
    assert_eq!(dyn_p.active_skill(), Some(boost_idx));
    let d1 = greedy_logit_argmax(&mut dyn_p, &ids);
    assert_eq!(
        d1, logits_static,
        "dynamic-overlaid logits must be BIT-IDENTICAL to static overlay"
    );

    dyn_p.set_active_skill(None).unwrap();
    let d2 = greedy_logit_argmax(&mut dyn_p, &ids);
    assert_eq!(d2, logits_base, "switch back must restore backbone exactly");

    // boost is FFN-eligible → switchable, but has no selection descriptor
    // → NOT auto-routable. dynamic_skills() requires both (routing needs φ).
    assert!(
        dyn_p.dynamic_skills().is_empty(),
        "no-selection skill must not be auto-routable"
    );

    // Idempotent no-op switch.
    dyn_p.set_active_skill(None).unwrap();
    assert_eq!(dyn_p.active_skill(), None);
}

/// Regression (adversarial review): a pipeline loaded WITH a static
/// skill must know it (dyn_active set at load), so switching back to
/// backbone actually reverts — not a silent no-op leaving the skill
/// overlaid. This is the `--skill X --route-dynamic` corruption path.
#[test]
fn static_skill_load_reverts_to_backbone_on_switch() {
    let dir = std::env::temp_dir().join(format!("cmf-revert-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("swarm.cmf");
    write_swarm(&path, true);
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let boost_idx = model.header.skills.iter().position(|s| s.id == "boost").unwrap();
    let ids = [3u32, 7, 11, 5];

    let mut base = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let logits_base = greedy_logit_argmax(&mut base, &ids);

    // Load STATICALLY with the skill; the pipeline must record dyn_active.
    let mut p =
        Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some("boost")).unwrap();
    assert_eq!(
        p.active_skill(),
        Some(boost_idx),
        "static skill load must record dyn_active"
    );
    let logits_overlaid = greedy_logit_argmax(&mut p, &ids);
    assert_ne!(logits_overlaid, logits_base);

    // Switch to backbone — must be a REAL revert, bit-identical to base.
    p.set_active_skill(None).unwrap();
    let logits_reverted = greedy_logit_argmax(&mut p, &ids);
    assert_eq!(
        logits_reverted, logits_base,
        "set_active_skill(None) after a static load must revert to true backbone"
    );
}
