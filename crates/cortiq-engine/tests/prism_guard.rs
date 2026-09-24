//! Regression: a file WITHOUT a Prism (signed-Hadamard) header must never
//! be treated as one. 0.6.9 answered the embedding's name before the header
//! gate, so `row_f32` on any non-Prism q4t/q4tp embedding walked into
//! `inverse_embedding` and panicked on the missing descriptor.

use cortiq_core::CMF_VERSION;
use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType};

fn arch() -> ModelArch {
    ModelArch {
        arch_name: "tiny".into(),
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 1,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        layer_types: vec![LayerType::FullAttention; 1],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        qwen4_exp: None,
        deepseek_v41: None,
        linear_core: None,
        head_clusters: None,
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
        qk_norm_after_rope: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: false,
        prism_hadamard: None,
        kv_heads_per_layer: None,
        v_head_dim: None,
    }
}

#[test]
fn non_prism_embedding_is_not_an_inverse_matrix() {
    let dir = std::env::temp_dir().join(format!("cmf-prism-guard-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.cmf");
    let embed: Vec<f32> = (0..32 * 8).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let tensors = vec![TensorSpec {
        name: "model.embed_tokens.weight".into(),
        dtype: cortiq_core::TensorDtype::F32,
        shape: vec![32, 8],
        data: embed.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }];
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch: arch(),
        quant_type: QuantType::F32,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(&path, &header, &tensors, None, None).unwrap();
    let model = CmfModel::open(&path).unwrap();
    assert!(!cortiq_engine::prism::has_contract(&model));
    // The exact call that panicked in 0.6.9 for every non-Prism file.
    assert!(!cortiq_engine::prism::is_inverse_embedding(
        &model,
        "model.embed_tokens.weight"
    ));
    assert!(!cortiq_engine::prism::is_forward_weight(
        &model,
        "model.layers.0.self_attn.q_proj.weight"
    ));
    let _ = std::fs::remove_dir_all(&dir);
}
