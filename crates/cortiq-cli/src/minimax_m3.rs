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
use std::collections::HashMap;

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

/// Header-only view of a source safetensors entry.
///
/// This deliberately does not borrow the converter's `TensorMeta` type.  The
/// contract remains usable with a streamed shard index (and unit tests) while
/// `convert` can create these cheap views directly from its mmap metadata.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MiniMaxM3TensorRef<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: &'a str,
    pub(crate) shape: &'a [usize],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorSignature {
    dtype: &'static str,
    shape: Vec<usize>,
}

fn add_expected(
    expected: &mut HashMap<String, TensorSignature>,
    name: impl Into<String>,
    dtype: &'static str,
    shape: impl Into<Vec<usize>>,
) {
    let name = name.into();
    assert!(
        expected
            .insert(
                name.clone(),
                TensorSignature {
                    dtype,
                    shape: shape.into(),
                },
            )
            .is_none(),
        "MiniMax-M3.1 built duplicate source contract entry {name}"
    );
}

fn add_mxfp8_matrix(
    expected: &mut HashMap<String, TensorSignature>,
    name: &str,
    rows: usize,
    cols: usize,
) {
    // The public M3 checkpoint uses MXFP8 E4M3 with a [1, 32] UE8M0 scale
    // grid.  Keep this next to the name contract: accepting an ordinary FP8
    // matrix here would make the converter decode silently incorrect values.
    assert_eq!(cols % 32, 0, "MXFP8 source columns must be 32-aligned");
    add_expected(expected, name, "F8_E4M3", [rows, cols]);
    add_expected(
        expected,
        format!("{name}_scale_inv"),
        "U8",
        [rows, cols / 32],
    );
}

fn add_ct_nvfp4_matrix(
    expected: &mut HashMap<String, TensorSignature>,
    name: &str,
    rows: usize,
    cols: usize,
) {
    // `name` is the source prefix ending in `.w1` / `.w2` / `.w3`.  The
    // compressed-tensors NVFP4 payload is packed two E2M1 values/byte and
    // uses one E4M3 scale per 16 logical values.  The global/input scales are
    // intentionally scalar F32 tensors, not one-element vectors.
    assert_eq!(cols % 16, 0, "NVFP4 source columns must be 16-aligned");
    add_expected(
        expected,
        format!("{name}.weight_packed"),
        "U8",
        [rows, cols / 2],
    );
    add_expected(
        expected,
        format!("{name}.weight_scale"),
        "F8_E4M3",
        [rows, cols / 16],
    );
    add_expected(expected, format!("{name}.weight_global_scale"), "F32", []);
    add_expected(expected, format!("{name}.input_global_scale"), "F32", []);
}

fn expected_text_tensors(layout: &MiniMaxM3Layout) -> HashMap<String, TensorSignature> {
    let mut expected = HashMap::with_capacity(89_109);
    let h = layout.hidden_size;
    let attn_q = layout.num_attention_heads * layout.head_dim;
    let attn_kv = layout.num_key_value_heads * layout.head_dim;
    let index_q = layout.sparse_num_index_heads * layout.sparse_index_dim;
    // MiniMax's source contains one index-K head, not one per index-Q head.
    let index_k = layout.sparse_index_dim;

    add_expected(
        &mut expected,
        "language_model.lm_head.weight",
        "BF16",
        [layout.vocab_size, h],
    );
    add_expected(
        &mut expected,
        "language_model.model.embed_tokens.weight",
        "BF16",
        [layout.vocab_size, h],
    );
    add_expected(
        &mut expected,
        "language_model.model.norm.weight",
        "BF16",
        [h],
    );

    for layer in 0..layout.num_hidden_layers {
        let prefix = format!("language_model.model.layers.{layer}");
        add_expected(
            &mut expected,
            format!("{prefix}.input_layernorm.weight"),
            "BF16",
            [h],
        );
        add_expected(
            &mut expected,
            format!("{prefix}.post_attention_layernorm.weight"),
            "BF16",
            [h],
        );

        // Per-head Gemma RMSNorm stores a single head-width vector, rather
        // than the concatenated Q/K projection width.
        for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
            add_expected(
                &mut expected,
                format!("{prefix}.self_attn.{norm}.weight"),
                "BF16",
                [layout.head_dim],
            );
        }

        for (projection, rows) in [
            ("q_proj", attn_q),
            ("k_proj", attn_kv),
            ("v_proj", attn_kv),
            ("o_proj", h),
            ("index_q_proj", index_q),
            ("index_k_proj", index_k),
        ] {
            let cols = if projection == "o_proj" { attn_q } else { h };
            add_mxfp8_matrix(
                &mut expected,
                &format!("{prefix}.self_attn.{projection}.weight"),
                rows,
                cols,
            );
        }

        if layer < 3 {
            for (projection, rows, cols) in [
                ("gate_proj", layout.dense_intermediate_size, h),
                ("up_proj", layout.dense_intermediate_size, h),
                ("down_proj", h, layout.dense_intermediate_size),
            ] {
                add_mxfp8_matrix(
                    &mut expected,
                    &format!("{prefix}.mlp.{projection}.weight"),
                    rows,
                    cols,
                );
            }
        } else {
            let moe = format!("{prefix}.block_sparse_moe");
            add_expected(
                &mut expected,
                format!("{moe}.gate.weight"),
                "F32",
                [layout.num_local_experts, h],
            );
            add_expected(
                &mut expected,
                format!("{moe}.e_score_correction_bias"),
                "F32",
                [layout.num_local_experts],
            );
            for (projection, rows, cols) in [
                ("gate_proj", layout.shared_intermediate_size, h),
                ("up_proj", layout.shared_intermediate_size, h),
                ("down_proj", h, layout.shared_intermediate_size),
            ] {
                add_mxfp8_matrix(
                    &mut expected,
                    &format!("{moe}.shared_experts.{projection}.weight"),
                    rows,
                    cols,
                );
            }
            for expert in 0..layout.num_local_experts {
                let expert_prefix = format!("{moe}.experts.{expert}");
                add_ct_nvfp4_matrix(
                    &mut expected,
                    &format!("{expert_prefix}.w1"),
                    layout.intermediate_size,
                    h,
                );
                add_ct_nvfp4_matrix(
                    &mut expected,
                    &format!("{expert_prefix}.w2"),
                    h,
                    layout.intermediate_size,
                );
                add_ct_nvfp4_matrix(
                    &mut expected,
                    &format!("{expert_prefix}.w3"),
                    layout.intermediate_size,
                    h,
                );
            }
        }
    }

    debug_assert_eq!(expected.len(), 89_109);
    expected
}

/// Exact non-text tensor contract for the published M3.1 visual wrapper.
///
/// The current conversion route deliberately emits only the text backbone,
/// but that is not a license to accept an arbitrary collection of tensors
/// under the three visual roots.  The source identity matters even for a
/// text-only CMF: a changed patch geometry or projector can indicate a model
/// revision whose prompt/media contract has changed as well.  Keep this
/// signature table here (rather than a count-only guard) so full VL support
/// has a checked source inventory to build on later.
fn expected_visual_tensors() -> HashMap<String, TensorSignature> {
    const VISION_HIDDEN: usize = 1280;
    const VISION_INTERMEDIATE: usize = 5120;
    const VISION_LAYERS: usize = 32;
    const TEXT_HIDDEN: usize = 6144;
    let mut expected = HashMap::with_capacity(523);

    add_expected(
        &mut expected,
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "F32",
        [VISION_HIDDEN, 3, 2, 14, 14],
    );
    for param in ["weight", "bias"] {
        add_expected(
            &mut expected,
            format!("vision_tower.vision_model.pre_layrnorm.{param}"),
            "BF16",
            [VISION_HIDDEN],
        );
    }

    for layer in 0..VISION_LAYERS {
        let prefix = format!("vision_tower.vision_model.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            for param in ["weight", "bias"] {
                add_expected(
                    &mut expected,
                    format!("{prefix}.{norm}.{param}"),
                    "BF16",
                    [VISION_HIDDEN],
                );
            }
        }
        for (projection, rows, cols) in [
            ("fc1", VISION_INTERMEDIATE, VISION_HIDDEN),
            ("fc2", VISION_HIDDEN, VISION_INTERMEDIATE),
        ] {
            add_expected(
                &mut expected,
                format!("{prefix}.mlp.{projection}.weight"),
                "BF16",
                [rows, cols],
            );
            add_expected(
                &mut expected,
                format!("{prefix}.mlp.{projection}.bias"),
                "BF16",
                [rows],
            );
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add_expected(
                &mut expected,
                format!("{prefix}.self_attn.{projection}.weight"),
                "BF16",
                [VISION_HIDDEN, VISION_HIDDEN],
            );
            add_expected(
                &mut expected,
                format!("{prefix}.self_attn.{projection}.bias"),
                "BF16",
                [VISION_HIDDEN],
            );
        }
    }

    for (root, input) in [
        ("multi_modal_projector", VISION_HIDDEN),
        ("patch_merge_mlp", TEXT_HIDDEN * 4),
    ] {
        add_expected(
            &mut expected,
            format!("{root}.linear_1.weight"),
            "BF16",
            [TEXT_HIDDEN, input],
        );
        add_expected(
            &mut expected,
            format!("{root}.linear_1.bias"),
            "BF16",
            [TEXT_HIDDEN],
        );
        add_expected(
            &mut expected,
            format!("{root}.linear_2.weight"),
            "BF16",
            [TEXT_HIDDEN, TEXT_HIDDEN],
        );
        add_expected(
            &mut expected,
            format!("{root}.linear_2.bias"),
            "BF16",
            [TEXT_HIDDEN],
        );
    }

    debug_assert_eq!(expected.len(), 523);
    expected
}

/// Validate the full published MiniMax-M3.1 safetensors *header index*.
///
/// This is intentionally run before conversion opens/consumes a large source
/// shard.  It rejects a checkpoint with a changed text or visual tensor name,
/// dtype, shape, companion, expert, or duplicate.  The converter may choose
/// not to emit vision tensors until their runtime is implemented, but it must
/// never silently accept an unreviewed source layout.
#[allow(dead_code)] // invoked by `convert` when the dedicated M3 route lands.
pub(crate) fn validate_minimax_m3_source_index<'a>(
    layout: &MiniMaxM3Layout,
    tensors: impl IntoIterator<Item = MiniMaxM3TensorRef<'a>>,
) -> Result<()> {
    let mut expected_text = expected_text_tensors(layout);
    let mut expected_visual = expected_visual_tensors();
    let mut total = 0usize;

    for tensor in tensors {
        total += 1;
        if tensor.name.starts_with("language_model.") {
            let Some(signature) = expected_text.remove(tensor.name) else {
                anyhow::bail!(
                    "MiniMax-M3.1 source contract: unexpected or duplicate text tensor {}",
                    tensor.name
                );
            };
            ensure!(
                tensor.dtype == signature.dtype,
                "MiniMax-M3.1 source contract: {} dtype is {}, expected {}",
                tensor.name,
                tensor.dtype,
                signature.dtype
            );
            ensure!(
                tensor.shape == signature.shape.as_slice(),
                "MiniMax-M3.1 source contract: {} shape is {:?}, expected {:?}",
                tensor.name,
                tensor.shape,
                signature.shape
            );
        } else {
            let Some(signature) = expected_visual.remove(tensor.name) else {
                anyhow::bail!(
                    "MiniMax-M3.1 source contract: unexpected or duplicate visual tensor {}",
                    tensor.name
                );
            };
            ensure!(
                tensor.dtype == signature.dtype,
                "MiniMax-M3.1 source contract: {} dtype is {}, expected {}",
                tensor.name,
                tensor.dtype,
                signature.dtype
            );
            ensure!(
                tensor.shape == signature.shape.as_slice(),
                "MiniMax-M3.1 source contract: {} shape is {:?}, expected {:?}",
                tensor.name,
                tensor.shape,
                signature.shape
            );
        }
    }

    ensure!(
        expected_text.is_empty(),
        "MiniMax-M3.1 source contract: missing {} expected text tensor(s), first: {}",
        expected_text.len(),
        expected_text
            .keys()
            .next()
            .expect("non-empty map has first key")
    );
    ensure!(
        expected_visual.is_empty(),
        "MiniMax-M3.1 source contract: missing {} expected visual tensor(s), first: {}",
        expected_visual.len(),
        expected_visual
            .keys()
            .next()
            .expect("non-empty map has first key")
    );
    ensure!(
        total == 89_632,
        "MiniMax-M3.1 source contract: checkpoint has {total} tensors, expected 89632"
    );
    Ok(())
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

    #[derive(Clone, Debug)]
    struct OwnedTensor {
        name: String,
        dtype: String,
        shape: Vec<usize>,
    }

    impl OwnedTensor {
        fn as_ref(&self) -> MiniMaxM3TensorRef<'_> {
            MiniMaxM3TensorRef {
                name: &self.name,
                dtype: &self.dtype,
                shape: &self.shape,
            }
        }
    }

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

    fn published_source_index() -> Vec<OwnedTensor> {
        let layout = parse_minimax_m3_layout(&published_config()).unwrap();
        let mut tensors: Vec<_> = expected_text_tensors(&layout)
            .into_iter()
            .map(|(name, signature)| OwnedTensor {
                name,
                dtype: signature.dtype.to_owned(),
                shape: signature.shape,
            })
            .collect();
        tensors.extend(
            expected_visual_tensors()
                .into_iter()
                .map(|(name, signature)| OwnedTensor {
                    name,
                    dtype: signature.dtype.to_owned(),
                    shape: signature.shape,
                }),
        );
        assert_eq!(tensors.len(), 89_632);
        tensors
    }

    fn validate_owned_source(tensors: &[OwnedTensor]) -> Result<()> {
        let layout = parse_minimax_m3_layout(&published_config())?;
        validate_minimax_m3_source_index(&layout, tensors.iter().map(OwnedTensor::as_ref))
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

    #[test]
    fn accepts_exact_published_source_index_taxonomy() {
        let tensors = published_source_index();
        validate_owned_source(&tensors).unwrap();
    }

    #[test]
    fn source_index_rejects_wrong_mxfp8_companion_shape() {
        let mut tensors = published_source_index();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| {
                tensor.name == "language_model.model.layers.0.self_attn.q_proj.weight_scale_inv"
            })
            .unwrap();
        tensor.shape = vec![8192, 191];
        let err = validate_owned_source(&tensors).unwrap_err().to_string();
        assert!(err.contains("q_proj.weight_scale_inv"), "{err}");
        assert!(err.contains("shape"), "{err}");
    }

    #[test]
    fn source_index_rejects_changed_visual_signature() {
        let mut tensors = published_source_index();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| {
                tensor.name == "vision_tower.vision_model.embeddings.patch_embedding.weight"
            })
            .unwrap();
        tensor.shape = vec![1280, 3, 1, 14, 14];
        let err = validate_owned_source(&tensors).unwrap_err().to_string();
        assert!(err.contains("patch_embedding.weight"), "{err}");
        assert!(err.contains("shape"), "{err}");
    }

    #[test]
    fn source_index_rejects_changed_visual_dtype() {
        let mut tensors = published_source_index();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "multi_modal_projector.linear_1.weight")
            .unwrap();
        tensor.dtype = "F32".to_owned();
        let err = validate_owned_source(&tensors).unwrap_err().to_string();
        assert!(
            err.contains("multi_modal_projector.linear_1.weight"),
            "{err}"
        );
        assert!(err.contains("dtype"), "{err}");
    }

    #[test]
    fn source_index_rejects_missing_routed_expert_companion() {
        let mut tensors = published_source_index();
        tensors.retain(|tensor| {
            tensor.name
                != "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_global_scale"
        });
        let err = validate_owned_source(&tensors).unwrap_err().to_string();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn source_index_rejects_unknown_text_tensor() {
        let mut tensors = published_source_index();
        tensors[0].name = "language_model.model.layers.0.unreviewed.weight".to_owned();
        let err = validate_owned_source(&tensors).unwrap_err().to_string();
        assert!(err.contains("unexpected"), "{err}");
    }
}
