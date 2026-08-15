//! Standalone skill files at the container level: the identity keys
//! round-trip, the feature bit raises itself from the record, and a
//! reader treats the file as data — not as a network missing most of
//! its organs.

use cortiq_core::format::features;
use cortiq_core::types::{LayerType, ModelArch, NormStyle};
use cortiq_core::{CmfHeader, CmfModel, SkillRecord, TensorDtype, TensorSpec};

fn tiny_header() -> CmfHeader {
    // The loop_masks fixture's arch, wrapped in the smallest header the
    // writer accepts.
    let arch = ModelArch {
        arch_name: "tiny-looped".into(),
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 10,
        layer_types: vec![LayerType::FullAttention; 2],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 64,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: false,
    };
    let mut h: CmfHeader = serde_json::from_value(serde_json::json!({
        "version": 2,
        "arch": serde_json::to_value(&arch).unwrap(),
        "quant_type": "F16",
    }))
    .expect("tiny header");
    h.arch = arch;
    h
}

fn t(name: &str, fill: u8) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::F16,
        shape: vec![4, 4],
        data: vec![fill; 32],
    }
}

#[test]
fn skill_keys_round_trip_and_raise_the_feature_bit() {
    let dir = std::env::temp_dir().join(format!("cmf-skill-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base_p = dir.join("base.cmf");
    let skill_p = dir.join("gfx.skill.cmf");

    // A base with three tensors.
    let header = tiny_header();
    CmfModel::write(
        &base_p,
        &header,
        &[
            t("model.layers.0.mlp.down_proj.weight", 1),
            t("model.layers.1.mlp.down_proj.weight", 2),
            t("lm_head.weight", 3),
        ],
        None,
        None,
    )
    .unwrap();
    let base = CmfModel::open(&base_p).unwrap();
    assert_eq!(
        base.required_features & features::SKILL_FILE,
        0,
        "a plain model must not carry the skill bit"
    );

    // A skill file: one replaced tensor + the identity keys.
    let mut sh = tiny_header();
    sh.skills = vec![SkillRecord {
        id: "gfx".into(),
        name: Some("graphics".into()),
        layers: vec![1],
        selection: None,
        input_mask_task: None,
        quality: None,
        base_dir_hash: Some(format!("{:016x}", base.dir_hash())),
        base_arch: Some("tiny-looped".into()),
        task: Some("specialist".into()),
        provenance: Some(serde_json::json!({"corpus": "test"})),
    }];
    CmfModel::write(
        &skill_p,
        &sh,
        &[t("model.layers.1.mlp.down_proj.weight", 9)],
        None,
        None,
    )
    .unwrap();

    let skill = CmfModel::open(&skill_p).unwrap();
    assert_ne!(
        skill.required_features & features::SKILL_FILE,
        0,
        "a bound skill record must raise the SKILL_FILE bit"
    );
    let rec = &skill.header.skills[0];
    assert_eq!(
        rec.base_dir_hash.as_deref(),
        Some(format!("{:016x}", base.dir_hash()).as_str())
    );
    assert_eq!(rec.base_arch.as_deref(), Some("tiny-looped"));
    assert_eq!(rec.task.as_deref(), Some("specialist"));
    assert_eq!(rec.layers, vec![1]);
    // The partial tensor set is exactly what was cut.
    assert_eq!(skill.tensors.len(), 1);
    assert_eq!(skill.tensors[0].name, "model.layers.1.mlp.down_proj.weight");

    std::fs::remove_dir_all(&dir).ok();
}
