//! Recursive Flint compatibility repair.
//!
//! The first public CMF exports of RecursiveSI/flint predated its complete
//! attention profile.  Their quantized tensors are sound, but their headers
//! lack the sliding-window, no-RoPE, learned-sink, and sigmoid-router
//! declarations.  Re-exporting from the 80 GB BF16 source is wasteful and
//! unnecessary: this module validates the exact released tensor catalogue and
//! atomically appends a corrected header.  The directory and every payload
//! byte remain untouched.

use anyhow::{Context, bail, ensure};
use cortiq_core::{CmfHeader, CmfModel, LayerType, TensorDtype};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;

const FLINT_SOURCE_REVISION: &str = "2a247be0c0aec895ccb0ac7ca92229ec6df37288";
pub(crate) const FLINT_LAYERS: usize = 49;
const FLINT_TENSORS: usize = 36_734;
pub(crate) const FLINT_HIDDEN: usize = 2_048;
const FLINT_Q_HEADS: usize = 32;
const FLINT_KV_HEADS: usize = 4;
const FLINT_HEAD_DIM: usize = 128;
const FLINT_VOCAB: usize = 150_144;
pub(crate) const FLINT_EXPERTS: usize = 256;
const FLINT_TOP_K: usize = 6;
pub(crate) const FLINT_EXPERT_WIDTH: usize = 512;
const FLINT_SHARED_WIDTH: usize = 1_024;
const FLINT_WINDOW: usize = 513;
const FLINT_NO_ROPE: [usize; 14] = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48];

/// Recursive's release settings.  These belong in the artifact rather than
/// only in its model card: `run` and `serve` consume `provenance.sampling`,
/// while callers can still explicitly override any field per request.
fn flint_sampling() -> Value {
    json!({
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantProfile {
    Q4tp,
    Q2tp,
    /// Whole routed gate/up layers may be q2tp or q4tp, but a layer may not
    /// mix them.  This is the only mixed layout the token/decode graph can
    /// dispatch without crossing a per-layer kernel ABI.
    Hybrid,
}

impl QuantProfile {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Q4tp => "q4tp",
            Self::Q2tp => "q2tp",
            Self::Hybrid => "q2tp-q4tp-layer-hybrid",
        }
    }
}

#[derive(Clone, Debug)]
struct RepairPlan {
    profile: QuantProfile,
    provenance: Value,
    already_current: bool,
}

fn expected_no_rope() -> Vec<bool> {
    (0..FLINT_LAYERS)
        .map(|li| FLINT_NO_ROPE.contains(&li))
        .collect()
}

fn expected_layer_types() -> Vec<LayerType> {
    expected_no_rope()
        .into_iter()
        .map(|no_rope| {
            if no_rope {
                LayerType::FullAttention
            } else {
                LayerType::SlidingAttention
            }
        })
        .collect()
}

/// Validate the architecture fields shared by both the old and corrected
/// public exports.  This deliberately happens before an in-place update: a
/// similarly named model must never receive Flint semantics by accident.
fn validate_flint_header(header: &CmfHeader) -> anyhow::Result<()> {
    let arch = &header.arch;
    ensure!(
        arch.arch_name == "flint",
        "repair-flint only accepts arch_name=flint (got '{}')",
        arch.arch_name
    );
    ensure!(
        (
            arch.num_layers,
            arch.hidden_size,
            arch.num_attention_heads,
            arch.num_kv_heads,
            arch.head_dim,
            arch.vocab_size
        ) == (
            FLINT_LAYERS,
            FLINT_HIDDEN,
            FLINT_Q_HEADS,
            FLINT_KV_HEADS,
            FLINT_HEAD_DIM,
            FLINT_VOCAB
        ),
        "Flint geometry is not the released RecursiveSI/flint profile"
    );
    ensure!(
        arch.layer_types == expected_layer_types(),
        "Flint layer schedule is not the released 35 sliding / 14 full schedule"
    );
    let moe = arch
        .moe
        .as_ref()
        .context("Flint header has no MoE descriptor")?;
    ensure!(
        (
            moe.num_experts,
            moe.top_k,
            moe.moe_intermediate_size,
            moe.shared_expert_intermediate_size
        ) == (
            FLINT_EXPERTS,
            FLINT_TOP_K,
            FLINT_EXPERT_WIDTH,
            Some(FLINT_SHARED_WIDTH)
        ),
        "Flint MoE geometry is not 256 experts / top-6 / 512 + shared-1024"
    );
    ensure!(
        moe.norm_topk_prob,
        "Flint must normalize its selected sigmoid routing scores"
    );
    Ok(())
}

pub(crate) fn expert_name(li: usize, expert: usize, leaf: &str) -> String {
    format!("model.layers.{li}.mlp.experts.{expert}.{leaf}.weight")
}

/// Remove one exactly-shaped directory entry from a catalogue under
/// validation.  Removing every expected name and rejecting leftovers means a
/// same-length but structurally different CMF cannot be blessed as Flint.
fn take_catalogue_entry<'a>(
    remaining: &mut HashMap<&'a str, &'a cortiq_core::TensorEntry>,
    name: &str,
    shape: &[usize],
    allowed_dtypes: &[TensorDtype],
) -> anyhow::Result<&'a cortiq_core::TensorEntry> {
    let entry = remaining
        .remove(name)
        .with_context(|| format!("missing Flint tensor '{name}'"))?;
    ensure!(
        entry.shape.as_slice() == shape,
        "{name} has shape {:?}, expected {shape:?}",
        entry.shape
    );
    ensure!(
        allowed_dtypes.contains(&entry.dtype),
        "{name} has dtype {:?}, expected one of {:?}",
        entry.dtype,
        allowed_dtypes
    );
    Ok(entry)
}

/// Validate every name, shape, and non-profile dtype in Flint's released
/// catalogue.  The two routed gate/up matrices are deliberately the only
/// entries allowed to be q2tp or q4tp; their per-layer profile is returned.
fn validate_flint_catalogue(
    tensors: &[cortiq_core::TensorEntry],
) -> anyhow::Result<Vec<TensorDtype>> {
    ensure!(
        tensors.len() == FLINT_TENSORS,
        "Flint tensor catalogue has {} entries, expected {FLINT_TENSORS}",
        tensors.len()
    );
    let mut remaining = HashMap::with_capacity(tensors.len());
    for entry in tensors {
        ensure!(
            remaining.insert(entry.name.as_str(), entry).is_none(),
            "Flint tensor catalogue has duplicate entry '{}'",
            entry.name
        );
    }

    let f16 = [TensorDtype::F16];
    let q4tp = [TensorDtype::Q4TiledP];
    let routed = [TensorDtype::Q2TiledP, TensorDtype::Q4TiledP];

    take_catalogue_entry(
        &mut remaining,
        "model.embed_tokens.weight",
        &[FLINT_VOCAB, FLINT_HIDDEN],
        &q4tp,
    )?;
    take_catalogue_entry(&mut remaining, "model.norm.weight", &[FLINT_HIDDEN], &f16)?;
    take_catalogue_entry(
        &mut remaining,
        "lm_head.weight",
        &[FLINT_VOCAB, FLINT_HIDDEN],
        &q4tp,
    )?;

    let mut layer_profiles = Vec::with_capacity(FLINT_LAYERS.saturating_sub(2));
    for li in 0..FLINT_LAYERS {
        let prefix = format!("model.layers.{li}");
        for (suffix, shape, allowed_dtypes) in [
            ("input_layernorm.weight", vec![FLINT_HIDDEN], &f16[..]),
            (
                "post_attention_layernorm.weight",
                vec![FLINT_HIDDEN],
                &f16[..],
            ),
            ("self_attn.q_norm.weight", vec![FLINT_HEAD_DIM], &f16[..]),
            ("self_attn.k_norm.weight", vec![FLINT_HEAD_DIM], &f16[..]),
            (
                "self_attn.q_proj.weight",
                vec![FLINT_Q_HEADS * FLINT_HEAD_DIM, FLINT_HIDDEN],
                &q4tp[..],
            ),
            (
                "self_attn.k_proj.weight",
                vec![FLINT_KV_HEADS * FLINT_HEAD_DIM, FLINT_HIDDEN],
                &q4tp[..],
            ),
            (
                "self_attn.v_proj.weight",
                vec![FLINT_KV_HEADS * FLINT_HEAD_DIM, FLINT_HIDDEN],
                &q4tp[..],
            ),
            (
                "self_attn.o_proj.weight",
                vec![FLINT_HIDDEN, FLINT_Q_HEADS * FLINT_HEAD_DIM],
                &q4tp[..],
            ),
            ("self_attn.softmax_offset", vec![FLINT_Q_HEADS], &f16[..]),
        ] {
            take_catalogue_entry(
                &mut remaining,
                &format!("{prefix}.{suffix}"),
                &shape,
                allowed_dtypes,
            )?;
        }

        if li < 2 {
            for (suffix, shape) in [
                ("mlp.gate_proj.weight", vec![6_144, FLINT_HIDDEN]),
                ("mlp.up_proj.weight", vec![6_144, FLINT_HIDDEN]),
                ("mlp.down_proj.weight", vec![FLINT_HIDDEN, 6_144]),
            ] {
                take_catalogue_entry(&mut remaining, &format!("{prefix}.{suffix}"), &shape, &q4tp)?;
            }
            continue;
        }

        take_catalogue_entry(
            &mut remaining,
            &format!("{prefix}.mlp.gate.weight"),
            &[FLINT_EXPERTS, FLINT_HIDDEN],
            &f16,
        )?;
        for (suffix, shape) in [
            ("gate_proj.weight", vec![FLINT_SHARED_WIDTH, FLINT_HIDDEN]),
            ("up_proj.weight", vec![FLINT_SHARED_WIDTH, FLINT_HIDDEN]),
            ("down_proj.weight", vec![FLINT_HIDDEN, FLINT_SHARED_WIDTH]),
        ] {
            take_catalogue_entry(
                &mut remaining,
                &format!("{prefix}.mlp.shared_expert.{suffix}"),
                &shape,
                &q4tp,
            )?;
        }

        let mut layer_dtype = None;
        for expert in 0..FLINT_EXPERTS {
            for leaf in ["gate_proj", "up_proj"] {
                let name = expert_name(li, expert, leaf);
                let entry = take_catalogue_entry(
                    &mut remaining,
                    &name,
                    &[FLINT_EXPERT_WIDTH, FLINT_HIDDEN],
                    &routed,
                )?;
                if let Some(expected) = layer_dtype {
                    ensure!(
                        entry.dtype == expected,
                        "Flint routed layer {li} mixes q4tp and q2tp gate/up tensors; \
                         {name} is {:?}, expected {:?}",
                        entry.dtype,
                        expected
                    );
                } else {
                    layer_dtype = Some(entry.dtype);
                }
            }
            let name = expert_name(li, expert, "down_proj");
            take_catalogue_entry(
                &mut remaining,
                &name,
                &[FLINT_HIDDEN, FLINT_EXPERT_WIDTH],
                &q4tp,
            )?;
        }
        layer_profiles.push(layer_dtype.expect("each routed Flint layer has gate/up tensors"));
    }

    ensure!(
        remaining.is_empty(),
        "Flint tensor catalogue contains unexpected entry '{}'{}, not part of the released profile",
        remaining.keys().next().expect("checked non-empty"),
        if remaining.len() > 1 {
            " (and others)"
        } else {
            ""
        }
    );
    Ok(layer_profiles)
}

/// Verify the actual release catalogue, then derive its profile from tensor
/// dtypes rather than trusting stale header provenance.
pub(crate) fn inspect_quant_profile(model: &CmfModel) -> anyhow::Result<QuantProfile> {
    validate_flint_header(&model.header)?;
    profile_from_layer_dtypes(&validate_flint_catalogue(&model.tensors)?)
}

/// Classify an already uniform routed layer sequence.  Keeping this separate
/// makes it explicit that Hybrid is a *layer* property, not permission for an
/// arbitrary mixture of individual expert tensors.
fn profile_from_layer_dtypes(layer_dtypes: &[TensorDtype]) -> anyhow::Result<QuantProfile> {
    let mut seen_q4 = false;
    let mut seen_q2 = false;
    for &dtype in layer_dtypes {
        match dtype {
            TensorDtype::Q4TiledP => seen_q4 = true,
            TensorDtype::Q2TiledP => seen_q2 = true,
            other => bail!("Flint routed layer must be q4tp or q2tp, got {other:?}"),
        }
    }
    match (seen_q4, seen_q2) {
        (true, false) => Ok(QuantProfile::Q4tp),
        (false, true) => Ok(QuantProfile::Q2tp),
        (true, true) => Ok(QuantProfile::Hybrid),
        (false, false) => bail!("Flint contains no routed gate/up tensors"),
    }
}

fn repaired_provenance(old: Option<&Value>, profile: QuantProfile) -> anyhow::Result<Value> {
    let mut provenance = old.cloned().unwrap_or_else(|| json!({}));
    let object = provenance
        .as_object_mut()
        .context("Flint provenance must be a JSON object when present")?;
    object.insert("source_model".into(), json!("RecursiveSI/flint"));
    object.insert("source_revision".into(), json!(FLINT_SOURCE_REVISION));
    object.insert("weight_quant".into(), json!(profile.name()));
    object.insert("sampling".into(), flint_sampling());
    object.insert(
        "flint_repair".into(),
        json!({
            "kind": "header-only",
            "reason": "restore sliding-window, no-RoPE, learned-sink, and sigmoid-router metadata",
            "source_revision": FLINT_SOURCE_REVISION,
            "tensor_payloads": "unchanged",
        }),
    );
    Ok(provenance)
}

fn has_current_metadata(header: &CmfHeader, profile: QuantProfile) -> bool {
    let arch = &header.arch;
    let Some(moe) = arch.moe.as_ref() else {
        return false;
    };
    let provenance_ok = header
        .provenance
        .as_ref()
        .and_then(Value::as_object)
        .is_some_and(|p| {
            p.get("source_model").and_then(Value::as_str) == Some("RecursiveSI/flint")
                && p.get("source_revision").and_then(Value::as_str) == Some(FLINT_SOURCE_REVISION)
                && p.get("weight_quant").and_then(Value::as_str) == Some(profile.name())
                && p.get("sampling") == Some(&flint_sampling())
        });
    arch.sliding_window == Some(FLINT_WINDOW)
        && arch.no_rope_layers.as_deref() == Some(expected_no_rope().as_slice())
        && arch.attention_sink
        && moe.router_sigmoid
        && provenance_ok
}

fn apply_repair(header: &mut CmfHeader, provenance: &Value) {
    header.arch.sliding_window = Some(FLINT_WINDOW);
    header.arch.no_rope_layers = Some(expected_no_rope());
    header.arch.attention_sink = true;
    // Flint ranks and mixes with sigmoid router scores, then renormalizes
    // the selected top-6.  Treating it as softmax changes both selection and
    // output even when every tensor byte is correct.
    header
        .arch
        .moe
        .as_mut()
        .expect("validate_flint_header checked MoE")
        .router_sigmoid = true;
    header.provenance = Some(provenance.clone());
}

fn make_plan(model: &CmfModel) -> anyhow::Result<RepairPlan> {
    let profile = inspect_quant_profile(model)?;
    Ok(RepairPlan {
        profile,
        provenance: repaired_provenance(model.header.provenance.as_ref(), profile)?,
        already_current: has_current_metadata(&model.header, profile),
    })
}

/// A header append only changes the file passed to `repair-flint`.  A sharded
/// export has sibling headers which would retain stale semantics, so require
/// the exact standalone distribution instead of producing a split-brain set.
fn require_standalone_repair(model: &CmfModel) -> anyhow::Result<()> {
    ensure!(
        model.header.shard.is_none(),
        "repair-flint accepts a standalone CMF only; '{}' has shard metadata",
        model.path.display()
    );
    ensure!(
        model.tensors.iter().all(|entry| entry.shard == 0),
        "repair-flint accepts a standalone CMF only; '{}' has external shard-backed tensor entries",
        model.path.display()
    );
    Ok(())
}

/// Return a fresh header with the released Flint execution semantics and
/// provenance for `profile`.  Rewriters use this rather than assuming an
/// input produced after `repair-flint`; source payloads can remain legacy
/// while the newly emitted artifact is self-describing.
pub(crate) fn repaired_flint_header(
    source: &CmfHeader,
    profile: QuantProfile,
) -> anyhow::Result<CmfHeader> {
    validate_flint_header(source)?;
    let provenance = repaired_provenance(source.provenance.as_ref(), profile)?;
    let mut header = source.clone();
    apply_repair(&mut header, &provenance);
    Ok(header)
}

/// Validate both the released Flint catalogue and its already-correct
/// execution metadata.  Useful after a whole-file rewrite, where accepting a
/// stale header would make a byte-correct tensor payload execute incorrectly.
pub(crate) fn ensure_current_flint_metadata(model: &CmfModel) -> anyhow::Result<QuantProfile> {
    let profile = inspect_quant_profile(model)?;
    ensure!(
        has_current_metadata(&model.header, profile),
        "Flint tensor catalogue is valid but its semantic metadata is stale; run repair-flint"
    );
    Ok(profile)
}

/// Append the missing Flint semantic metadata in place.
///
/// `update_header_append` writes and fsyncs a new header at EOF before it
/// switches the envelope, so a cancelled repair leaves the previous model
/// readable.  It does not touch the tensor directory, data range, or vocab.
pub fn cmd_repair_flint(model_path: &str, dry_run: bool) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    let model = CmfModel::open(path).with_context(|| format!("open {}", path.display()))?;
    require_standalone_repair(&model)?;
    let dir_hash = model.dir_hash();
    let plan = make_plan(&model)?;
    if plan.already_current {
        println!(
            "{}: Flint metadata is already current ({}) — no changes",
            path.display(),
            plan.profile.name()
        );
        return Ok(());
    }
    println!(
        "{}: verified Recursive Flint catalogue; profile {}; restoring window={}, 14 no-RoPE layers, 49 learned sinks, and sigmoid top-6 routing",
        path.display(),
        plan.profile.name(),
        FLINT_WINDOW
    );
    if dry_run {
        println!("dry run: header was not changed");
        return Ok(());
    }
    drop(model);
    let provenance = plan.provenance;
    let report =
        CmfModel::update_header_append(path, move |header| apply_repair(header, &provenance))
            .with_context(|| format!("append repaired Flint header to {}", path.display()))?;
    ensure!(
        report.tensors_added == 0 && report.data_len_before == report.data_len_after,
        "header repair unexpectedly changed tensor data"
    );
    let repaired = CmfModel::open(path).with_context(|| format!("re-open {}", path.display()))?;
    let verified = make_plan(&repaired)?;
    ensure!(
        verified.already_current,
        "repaired Flint header did not persist"
    );
    ensure!(
        repaired.dir_hash() == dir_hash,
        "header repair changed the tensor directory hash"
    );
    println!(
        "repaired {}: appended {} header bytes; tensor directory and payload bytes unchanged",
        path.display(),
        report.new_len - report.old_len
    );
    let sig = format!("{}.sig", path.display());
    if Path::new(&sig).exists() {
        eprintln!(
            "warning: {sig} signs the old complete file; re-sign it after this header update"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::{CmfHeader, ModelArch, MoeConfig, NormStyle, QuantType};

    fn old_header() -> CmfHeader {
        CmfHeader {
            format: "cmf".into(),
            version: 2,
            arch: ModelArch {
                arch_name: "flint".into(),
                hidden_size: FLINT_HIDDEN,
                intermediate_size: 6_144,
                num_layers: FLINT_LAYERS,
                num_attention_heads: FLINT_Q_HEADS,
                num_kv_heads: FLINT_KV_HEADS,
                head_dim: FLINT_HEAD_DIM,
                vocab_size: FLINT_VOCAB,
                layer_types: expected_layer_types(),
                rms_norm_eps: 1e-6,
                norm_style: NormStyle::Qwen,
                rope_theta: 10_000.0,
                tie_word_embeddings: false,
                partial_rotary_factor: 1.0,
                yarn: None,
                attention_heads_per_layer: None,
                kv_heads_per_layer: None,
                v_head_dim: None,
                hidden_act: "silu".into(),
                embed_multiplier: 1.0,
                query_pre_attn_scalar: None,
                sliding_window: None,
                sliding_window_pattern: None,
                rope_local_base_freq: None,
                local_partial_rotary_factor: None,
                global_head_dim: None,
                num_global_kv_heads: None,
                global_partial_rotary_factor: None,
                final_logit_softcapping: None,
                attn_logit_softcapping: None,
                mla: None,
                no_rope_layers: None,
                attention_sink: false,
                activation_situ_beta: None,
                activation_situ_linear_beta: None,
                attn_v_norm: false,
                qk_norm_after_rope: false,
                mtp: None,
                moe: Some(MoeConfig {
                    num_experts: FLINT_EXPERTS,
                    top_k: FLINT_TOP_K,
                    moe_intermediate_size: FLINT_EXPERT_WIDTH,
                    norm_topk_prob: true,
                    shared_expert_intermediate_size: Some(FLINT_SHARED_WIDTH),
                    router_sigmoid: false,
                    routed_scaling_factor: None,
                    router_resonance: false,
                }),
                qwen4_exp: None,
                block_diffusion: None,
                deepseek_v41: None,
                linear_core: None,
                anchor_core: None,
                head_clusters: None,
                max_position_embeddings: 262_144,
                linear_conv_kernel_dim: None,
                linear_num_key_heads: None,
                linear_num_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                num_loops: 1,
                kda_gate_lower_bound: None,
                g3n: None,
                rope_freq_factors: None,
                logit_multiplier: None,
                loop_final_norm: false,
                prism_hadamard: None,
            },
            quant_type: QuantType::Q4Block,
            provenance: Some(json!({"source_model":"/root/flint", "weight_quant":"q4tp"})),
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: Vec::new(),
            router: None,
            segments: Vec::new(),
        }
    }

    #[test]
    fn exact_flint_schedule_is_preserved() {
        assert_eq!(
            FLINT_NO_ROPE.iter().copied().collect::<Vec<_>>(),
            expected_no_rope()
                .iter()
                .enumerate()
                .filter_map(|(li, &no_rope)| no_rope.then_some(li))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            expected_layer_types()
                .iter()
                .filter(|ty| matches!(ty, LayerType::SlidingAttention))
                .count(),
            35
        );
    }

    #[test]
    fn repair_sets_the_four_execution_semantics_and_corrects_provenance() {
        let mut header = old_header();
        validate_flint_header(&header).unwrap();
        let provenance =
            repaired_provenance(header.provenance.as_ref(), QuantProfile::Q2tp).unwrap();
        apply_repair(&mut header, &provenance);
        assert_eq!(header.arch.sliding_window, Some(FLINT_WINDOW));
        assert_eq!(header.arch.no_rope_layers, Some(expected_no_rope()));
        assert!(header.arch.attention_sink);
        assert!(header.arch.moe.as_ref().unwrap().router_sigmoid);
        assert_eq!(
            header.provenance.as_ref().unwrap()["weight_quant"],
            json!("q2tp")
        );
        assert_eq!(
            header.provenance.as_ref().unwrap()["source_revision"],
            json!(FLINT_SOURCE_REVISION)
        );
        assert_eq!(
            header.provenance.as_ref().unwrap()["sampling"],
            flint_sampling()
        );
    }
}
