//! MiMo-V2 attention geometry at LOAD time: per-layer KV heads, V heads
//! narrower than Q/K, learned sinks. A file whose k/v/o shapes disagree
//! with its header must fail to load with a message naming the tensor —
//! the mapped matvec would otherwise read a short V or a wide o_proj
//! silently. A consistent file loads, carries the geometry, gets the 32k
//! context cap despite a 1M `max_position_embeddings`, declines the GPU
//! graphs and runs.

use std::sync::Arc;

use cortiq_core::CMF_VERSION;
use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType};
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;

const HS: usize = 16;
const INTER: usize = 24;
const NH: usize = 4;
const HD: usize = 8;
const VD: usize = 4;
const VOCAB: usize = 48;
/// KV heads of layer 0 (full attention) and layer 1 (sliding).
const KVH: [usize; 2] = [1, 2];

fn arch(kv_heads_per_layer: Option<Vec<usize>>, v_head_dim: Option<usize>) -> ModelArch {
    ModelArch {
        arch_name: "mimo_v2".into(),
        hidden_size: HS,
        intermediate_size: INTER,
        num_layers: 2,
        num_attention_heads: NH,
        num_kv_heads: KVH[0],
        head_dim: HD,
        vocab_size: VOCAB,
        layer_types: vec![LayerType::FullAttention, LayerType::SlidingAttention],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000_000.0,
        tie_word_embeddings: true,
        partial_rotary_factor: 0.5,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        qwen4_exp: None,
        deepseek_v41: None,
        linear_core: None,
        head_clusters: None,
        max_position_embeddings: 1 << 20,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: Some(3),
        sliding_window_pattern: None,
        rope_local_base_freq: Some(10_000.0),
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
        kv_heads_per_layer,
        v_head_dim,
    }
}

fn f32_tensor(name: String, shape: &[usize], salt: usize) -> TensorSpec {
    let n: usize = shape.iter().product();
    let data: Vec<u8> = (0..n)
        .map(|i| (((i * 29 + salt * 13 + 5) % 89) as f32 / 89.0 - 0.5) * 0.4)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    TensorSpec {
        name,
        dtype: cortiq_core::TensorDtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

/// The tensors of a consistent two-layer MiMo-shaped file; `edit` may
/// replace any of them (by name) before writing.
fn tensors(edit: &dyn Fn(&str) -> Option<Vec<usize>>) -> Vec<TensorSpec> {
    let mut specs: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![VOCAB, HS]),
        ("model.norm.weight".into(), vec![HS]),
    ];
    for li in 0..2 {
        let p = format!("model.layers.{li}.");
        let kv = KVH[li];
        specs.extend([
            (format!("{p}input_layernorm.weight"), vec![HS]),
            (format!("{p}post_attention_layernorm.weight"), vec![HS]),
            (format!("{p}self_attn.q_proj.weight"), vec![NH * HD, HS]),
            (format!("{p}self_attn.k_proj.weight"), vec![kv * HD, HS]),
            (format!("{p}self_attn.v_proj.weight"), vec![kv * VD, HS]),
            (format!("{p}self_attn.o_proj.weight"), vec![HS, NH * VD]),
            (format!("{p}mlp.gate_proj.weight"), vec![INTER, HS]),
            (format!("{p}mlp.up_proj.weight"), vec![INTER, HS]),
            (format!("{p}mlp.down_proj.weight"), vec![HS, INTER]),
        ]);
    }
    // The sliding layer carries learned sinks, one per Q head.
    specs.push(("model.layers.1.self_attn.sinks".into(), vec![NH]));
    specs
        .into_iter()
        .enumerate()
        .filter_map(|(i, (name, shape))| {
            let shape = match edit(&name) {
                Some(s) if s.is_empty() => return None, // drop the tensor
                Some(s) => s,
                None => shape,
            };
            Some(f32_tensor(name, &shape, i))
        })
        .collect()
}

fn write_model(
    tag: &str,
    arch: ModelArch,
    tensors: &[TensorSpec],
) -> (std::path::PathBuf, Arc<CmfModel>) {
    let dir = std::env::temp_dir().join(format!("cmf-mimo-geometry-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.cmf");
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::F32,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(&path, &header, tensors, None, None).unwrap();
    (dir, Arc::new(CmfModel::open(&path).unwrap()))
}

fn mimo_arch() -> ModelArch {
    arch(Some(KVH.to_vec()), Some(VD))
}

fn load_err(tag: &str, arch: ModelArch, edit: &dyn Fn(&str) -> Option<Vec<usize>>) -> String {
    let (dir, model) = write_model(tag, arch, &tensors(edit));
    let res = Pipeline::from_model(&model, SamplerConfig::default());
    let _ = std::fs::remove_dir_all(&dir);
    match res {
        Ok(_) => panic!("{tag}: a mis-shaped file loaded"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn consistent_mimo_file_loads_with_its_geometry_and_runs() {
    let (dir, model) = write_model("ok", mimo_arch(), &tensors(&|_| None));
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("load");
    assert_eq!(p.kv_heads_per_layer, Some(vec![1, 2]));
    assert_eq!(p.v_head_dim, Some(VD));
    let geo: Vec<(usize, usize)> = p
        .kv_cache
        .layers
        .iter()
        .map(|l| (l.num_kv_heads, l.head_dim))
        .collect();
    assert_eq!(geo, vec![(1, HD), (2, HD)]);
    assert!(p.kv_cache.layers[0].sinks.is_none());
    assert_eq!(p.kv_cache.layers[1].sinks.as_ref().map(Vec::len), Some(NH));
    // A 1M-position header does not size a 1M-row KV: the default cap.
    if std::env::var_os("CMF_MAX_SEQ").is_none() {
        assert_eq!(p.kv_cache.max_seq_len, 32_768);
    }
    assert_eq!(
        p.graph_attn_decline_reason(),
        Some("per-layer KV head counts")
    );
    p.layer_dump = None;
    let ids: Vec<u32> = (0..9u32).map(|i| (i * 5 + 1) % VOCAB as u32).collect();
    let logits = p.forward_ids(&ids, None).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mis_shaped_attention_tensors_fail_at_load() {
    let cases: [(&str, &str, Vec<usize>, &str); 4] = [
        // Layer 1 has 2 KV heads: 1·head_dim K rows is the full layers' shape.
        (
            "k",
            "model.layers.1.self_attn.k_proj.weight",
            vec![HD, HS],
            "k_proj rows",
        ),
        // V padded to head_dim in the FILE is not what the header says.
        (
            "v",
            "model.layers.1.self_attn.v_proj.weight",
            vec![2 * HD, HS],
            "v_proj rows",
        ),
        // o_proj reading nh·head_dim instead of nh·v_head_dim.
        (
            "o",
            "model.layers.0.self_attn.o_proj.weight",
            vec![HS, NH * HD],
            "o_proj cols",
        ),
        // One sink per Q head.
        (
            "sinks",
            "model.layers.1.self_attn.sinks",
            vec![NH - 1],
            "sinks",
        ),
    ];
    for (tag, name, shape, want) in cases {
        let edit = |n: &str| (n == name).then(|| shape.clone());
        let msg = load_err(tag, mimo_arch(), &edit);
        assert!(
            msg.contains(want),
            "{tag}: error does not name {want:?}: {msg}"
        );
    }
    // A header that forgets the per-layer KV heads: layer 1's 2-head K no
    // longer matches num_kv_heads = 1.
    let msg = load_err("no-kv-map", arch(None, Some(VD)), &|_| None);
    assert!(msg.contains("k_proj rows"), "{msg}");
    // A header that forgets v_head_dim: V and O are narrower than head_dim.
    let msg = load_err("no-vd", arch(Some(KVH.to_vec()), None), &|_| None);
    assert!(msg.contains("v_proj rows"), "{msg}");
    // A per-layer map of the wrong length.
    let msg = load_err("kv-len", arch(Some(vec![1, 2, 2]), Some(VD)), &|_| None);
    assert!(msg.contains("kv_heads_per_layer"), "{msg}");
}
