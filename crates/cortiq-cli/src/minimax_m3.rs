//! Fail-closed source contract for MiniMax-M3.1-Flash.
//!
//! This deliberately validates only the published **text backbone** geometry.
//! It is not a converter or a runtime route yet: the M3 source uses MXFP8 and
//! NVFP4 encodings, sparse attention, and a DSpark sidecar, each of which needs
//! an exact implementation before a CMF can be emitted.  Keeping this contract
//! separate prevents a future generic decoder path from silently accepting an
//! incompatible M3 checkpoint.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub(crate) const MINIMAX_M3_MODEL_TYPE: &str = "minimax_m3_vl";
const MINIMAX_M3_ARCHITECTURE: &str = "MiniMaxM3SparseForConditionalGeneration";

/// Frozen geometry of the public MiniMax-M3.1-Flash text checkpoint.
///
/// Fields are intentionally explicit rather than inferred from a model name:
/// conversion code must bind every special execution invariant to the source
/// configuration that supplied its tensors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MiniMaxM3Layout {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub sparse_index_dim: usize,
    pub sparse_num_index_heads: usize,
    pub sparse_topk_blocks: usize,
    pub sparse_block_size: usize,
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .with_context(|| format!("MiniMax-M3.1 contract: {label} must be an object"))
}

fn field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
) -> Result<&'a Value> {
    object
        .get(key)
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} is missing"))
}

fn usize_field(object: &serde_json::Map<String, Value>, key: &str, scope: &str) -> Result<usize> {
    let raw = field(object, key, scope)?.as_u64().with_context(|| {
        format!("MiniMax-M3.1 contract: {scope}.{key} must be an unsigned integer")
    })?;
    usize::try_from(raw)
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} does not fit usize"))
}

fn bool_field(object: &serde_json::Map<String, Value>, key: &str, scope: &str) -> Result<bool> {
    field(object, key, scope)?
        .as_bool()
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} must be a boolean"))
}

fn str_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
) -> Result<&'a str> {
    field(object, key, scope)?
        .as_str()
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} must be a string"))
}

fn f64_field(object: &serde_json::Map<String, Value>, key: &str, scope: &str) -> Result<f64> {
    field(object, key, scope)?
        .as_f64()
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} must be a number"))
}

fn expect_usize(
    object: &serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
    expected: usize,
) -> Result<usize> {
    let got = usize_field(object, key, scope)?;
    ensure!(
        got == expected,
        "MiniMax-M3.1 contract: {scope}.{key} is {got}, expected {expected}"
    );
    Ok(got)
}

fn expect_bool(
    object: &serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
    expected: bool,
) -> Result<()> {
    let got = bool_field(object, key, scope)?;
    ensure!(
        got == expected,
        "MiniMax-M3.1 contract: {scope}.{key} is {got}, expected {expected}"
    );
    Ok(())
}

fn expect_str(
    object: &serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
    expected: &str,
) -> Result<()> {
    let got = str_field(object, key, scope)?;
    ensure!(
        got == expected,
        "MiniMax-M3.1 contract: {scope}.{key} is {got:?}, expected {expected:?}"
    );
    Ok(())
}

fn expect_f64(
    object: &serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
    expected: f64,
) -> Result<()> {
    let got = f64_field(object, key, scope)?;
    ensure!(
        (got - expected).abs() <= f64::EPSILON,
        "MiniMax-M3.1 contract: {scope}.{key} is {got}, expected {expected}"
    );
    Ok(())
}

fn expect_i64_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
    scope: &str,
    expected: impl Iterator<Item = i64>,
) -> Result<()> {
    let values = field(object, key, scope)?
        .as_array()
        .with_context(|| format!("MiniMax-M3.1 contract: {scope}.{key} must be an array"))?;
    let expected: Vec<i64> = expected.collect();
    ensure!(
        values.len() == expected.len(),
        "MiniMax-M3.1 contract: {scope}.{key} has {} values, expected {}",
        values.len(),
        expected.len()
    );
    for (index, (value, expected)) in values.iter().zip(expected).enumerate() {
        let got = value.as_i64().with_context(|| {
            format!("MiniMax-M3.1 contract: {scope}.{key}[{index}] must be an integer")
        })?;
        ensure!(
            got == expected,
            "MiniMax-M3.1 contract: {scope}.{key}[{index}] is {got}, expected {expected}"
        );
    }
    Ok(())
}

/// Parse the exact published MiniMax-M3.1-Flash text-backbone contract.
///
/// The function fails closed.  A new upstream revision must add a deliberately
/// reviewed contract rather than being packed by a generic Llama/MoE path.
#[allow(dead_code)] // wired into `convert` only after exact source decoders land.
pub(crate) fn parse_minimax_m3_layout(config: &Value) -> Result<MiniMaxM3Layout> {
    let root = object(config, "root config")?;
    expect_str(root, "model_type", "root", MINIMAX_M3_MODEL_TYPE)?;

    let architectures = field(root, "architectures", "root")?
        .as_array()
        .context("MiniMax-M3.1 contract: root.architectures must be an array")?;
    ensure!(
        architectures
            .iter()
            .any(|value| value.as_str() == Some(MINIMAX_M3_ARCHITECTURE)),
        "MiniMax-M3.1 contract: root.architectures must include {MINIMAX_M3_ARCHITECTURE}"
    );

    let text = object(field(root, "text_config", "root")?, "root.text_config")?;
    const TEXT: &str = "text_config";

    let hidden_size = expect_usize(text, "hidden_size", TEXT, 6144)?;
    let intermediate_size = expect_usize(text, "intermediate_size", TEXT, 3072)?;
    let dense_intermediate_size = expect_usize(text, "dense_intermediate_size", TEXT, 12288)?;
    let shared_intermediate_size = expect_usize(text, "shared_intermediate_size", TEXT, 3072)?;
    let num_hidden_layers = expect_usize(text, "num_hidden_layers", TEXT, 60)?;
    let num_attention_heads = expect_usize(text, "num_attention_heads", TEXT, 64)?;
    let num_key_value_heads = expect_usize(text, "num_key_value_heads", TEXT, 4)?;
    let head_dim = expect_usize(text, "head_dim", TEXT, 128)?;
    let vocab_size = expect_usize(text, "vocab_size", TEXT, 200_064)?;
    let max_position_embeddings = expect_usize(text, "max_position_embeddings", TEXT, 1_048_576)?;
    expect_f64(text, "rms_norm_eps", TEXT, 1e-6)?;
    expect_bool(text, "use_gemma_norm", TEXT, true)?;
    expect_bool(text, "use_qk_norm", TEXT, true)?;
    expect_str(text, "qk_norm_type", TEXT, "per_head")?;
    expect_str(text, "hidden_act", TEXT, "swigluoai")?;
    expect_f64(text, "swiglu_alpha", TEXT, 1.702)?;
    expect_f64(text, "swiglu_limit", TEXT, 7.0)?;
    expect_bool(text, "tie_word_embeddings", TEXT, false)?;

    let num_local_experts = expect_usize(text, "num_local_experts", TEXT, 128)?;
    let num_experts_per_tok = expect_usize(text, "num_experts_per_tok", TEXT, 4)?;
    let n_shared_experts = expect_usize(text, "n_shared_experts", TEXT, 1)?;
    expect_str(text, "scoring_func", TEXT, "sigmoid")?;
    expect_bool(text, "use_routing_bias", TEXT, true)?;
    expect_f64(text, "routed_scaling_factor", TEXT, 2.0)?;
    expect_i64_array(
        text,
        "moe_layer_freq",
        TEXT,
        std::iter::repeat_n(0, 3).chain(std::iter::repeat_n(1, 57)),
    )?;

    let sparse = object(
        field(text, "sparse_attention_config", TEXT)?,
        "text_config.sparse_attention_config",
    )?;
    const SPARSE: &str = "text_config.sparse_attention_config";
    expect_bool(sparse, "use_sparse_attention", SPARSE, true)?;
    let sparse_index_dim = expect_usize(sparse, "sparse_index_dim", SPARSE, 128)?;
    let sparse_num_index_heads = expect_usize(sparse, "sparse_num_index_heads", SPARSE, 4)?;
    let sparse_topk_blocks = expect_usize(sparse, "sparse_topk_blocks", SPARSE, 16)?;
    let sparse_block_size = expect_usize(sparse, "sparse_block_size", SPARSE, 128)?;
    expect_str(sparse, "sparse_score_type", SPARSE, "max")?;
    expect_usize(sparse, "sparse_init_block", SPARSE, 0)?;
    expect_usize(sparse, "sparse_local_block", SPARSE, 1)?;
    expect_i64_array(
        sparse,
        "sparse_attention_freq",
        SPARSE,
        std::iter::repeat_n(1, num_hidden_layers),
    )?;
    expect_i64_array(
        sparse,
        "sparse_disable_index_value",
        SPARSE,
        std::iter::repeat_n(1, num_hidden_layers),
    )?;

    Ok(MiniMaxM3Layout {
        hidden_size,
        intermediate_size,
        dense_intermediate_size,
        shared_intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        max_position_embeddings,
        num_local_experts,
        num_experts_per_tok,
        n_shared_experts,
        sparse_index_dim,
        sparse_num_index_heads,
        sparse_topk_blocks,
        sparse_block_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn published_config() -> Value {
        let mut moe_layer_freq = vec![0i64; 3];
        moe_layer_freq.extend(std::iter::repeat_n(1, 57));
        let all_sparse_layers = vec![1i64; 60];
        json!({
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "max_position_embeddings": 1048576,
                "rms_norm_eps": 0.000001,
                "use_gemma_norm": true,
                "use_qk_norm": true,
                "qk_norm_type": "per_head",
                "hidden_act": "swigluoai",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "tie_word_embeddings": false,
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "routed_scaling_factor": 2.0,
                "moe_layer_freq": moe_layer_freq,
                "sparse_attention_config": {
                    "use_sparse_attention": true,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_score_type": "max",
                    "sparse_init_block": 0,
                    "sparse_local_block": 1,
                    "sparse_attention_freq": all_sparse_layers,
                    "sparse_disable_index_value": vec![1i64; 60]
                }
            }
        })
    }

    #[test]
    fn accepts_published_text_backbone_contract() {
        let layout = parse_minimax_m3_layout(&published_config()).unwrap();
        assert_eq!(layout.hidden_size, 6144);
        assert_eq!(layout.num_hidden_layers, 60);
        assert_eq!(layout.num_local_experts, 128);
        assert_eq!(layout.sparse_topk_blocks, 16);
    }

    #[test]
    fn rejects_changed_moe_schedule() {
        let mut config = published_config();
        config["text_config"]["moe_layer_freq"][3] = json!(0);
        let err = parse_minimax_m3_layout(&config).unwrap_err().to_string();
        assert!(err.contains("moe_layer_freq[3]"), "{err}");
    }

    #[test]
    fn rejects_changed_sparse_geometry() {
        let mut config = published_config();
        config["text_config"]["sparse_attention_config"]["sparse_topk_blocks"] = json!(8);
        let err = parse_minimax_m3_layout(&config).unwrap_err().to_string();
        assert!(err.contains("sparse_topk_blocks"), "{err}");
    }

    #[test]
    fn rejects_generic_swiglu_route() {
        let mut config = published_config();
        config["text_config"]["hidden_act"] = json!("silu");
        let err = parse_minimax_m3_layout(&config).unwrap_err().to_string();
        assert!(err.contains("hidden_act"), "{err}");
    }
}
