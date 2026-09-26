//! A decision-profile CMF (DECISION feature bit, `cortiq-decision-` arch)
//! must never enter the generative pipeline: every `Pipeline::from_model*`
//! entry point refuses it with the DECISION message, before any tensor is
//! read as a layer.

use cortiq_core::format::features;
use cortiq_core::{CmfError, CmfHeader, CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::loader::DECISION_MODEL_REFUSAL;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::path::PathBuf;
use std::sync::Arc;

fn decision_header() -> CmfHeader {
    serde_json::from_value(serde_json::json!({
        "format": "cmf",
        "version": 2,
        "quant_type": "F32",
        "arch": {
            "arch_name": "cortiq-decision-ph-v1",
            "hidden_size": 4480,
            "intermediate_size": 0,
            "num_layers": 0,
            "num_attention_heads": 0,
            "num_kv_heads": 0,
            "head_dim": 0,
            "vocab_size": 0,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 512
        },
        "provenance": {"tool": "cortiq decision", "version": "0.7.8"}
    }))
    .expect("decision header")
}

fn write_decision_toy() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cmf-decision-refusal-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("decision.cmf");
    let mean: Vec<u8> = (0..4480)
        .flat_map(|i| (i as f32 / 4480.0).to_le_bytes())
        .collect();
    CmfModel::write(
        &path,
        &decision_header(),
        &[
            TensorSpec {
                name: "decision.manifest".into(),
                dtype: TensorDtype::U8,
                shape: vec![2],
                data: b"{}".to_vec(),
            },
            TensorSpec {
                name: "decision.skill.toy.task.0.mean".into(),
                dtype: TensorDtype::F32,
                shape: vec![4480],
                data: mean,
            },
        ],
        None,
        None,
    )
    .unwrap();
    path
}

fn refusal<T>(r: Result<T, CmfError>) -> String {
    match r {
        Ok(_) => panic!("a DECISION model was accepted by the generative pipeline"),
        Err(CmfError::Parse(msg)) => msg,
        Err(e) => panic!("wrong refusal kind: {e}"),
    }
}

#[test]
fn decision_profile_is_not_a_language_model() {
    let path = write_decision_toy();
    let model = Arc::new(CmfModel::open(&path).unwrap());
    assert_ne!(model.required_features & features::DECISION, 0);

    let plain = refusal(Pipeline::from_model(&model, SamplerConfig::default()));
    assert_eq!(plain, DECISION_MODEL_REFUSAL);
    assert!(plain.contains("DECISION model"), "{plain}");
    assert!(plain.contains("cortiq decide"), "{plain}");
    assert!(plain.contains("cortiq serve"), "{plain}");
    // Display of the error (what run/chat print) carries the same text.
    let shown = Pipeline::from_model(&model, SamplerConfig::default())
        .err()
        .unwrap()
        .to_string();
    assert!(shown.contains(DECISION_MODEL_REFUSAL), "{shown}");

    // The skill and blend entry points go through the same guard.
    let skill = refusal(Pipeline::from_model_with_skill(
        &model,
        SamplerConfig::default(),
        Some("toy"),
    ));
    assert_eq!(skill, DECISION_MODEL_REFUSAL);
    let blend = refusal(Pipeline::from_model_with_blend(
        &model,
        SamplerConfig::default(),
        &[("toy".to_string(), 1.0)],
    ));
    assert_eq!(blend, DECISION_MODEL_REFUSAL);

    drop(model);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn decision_message_names_the_decision_commands() {
    assert_eq!(
        DECISION_MODEL_REFUSAL,
        "this is a DECISION model; use `cortiq decide` or `cortiq serve`"
    );
}
