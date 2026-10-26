//! `cortiq tube bake` — defragment a dense FFN into task tubes.
//!
//! A task mask is only a quality device as long as the neurons it keeps
//! are scattered: the kernel still streams every weight byte and zeroes
//! the dead activations afterwards. Reordering the FFN's intermediate
//! axis so that each task's neurons land in ONE contiguous run turns the
//! same mask into a *smaller matrix* — and a smaller matrix is fewer
//! bytes off the bus, which is what decode time is made of.
//!
//! The rewrite is exact: permuting the intermediate axis of an FFN
//! (gate/up rows and down columns together) is an identity on the
//! layer's function, because the activation is element-wise and the sum
//! over neurons is commutative. Nothing is trained here; the quality
//! delta of a tube file at full width is only the requantization of the
//! re-cut segments.
//!
//! Layout per layer: `[ core | tube 1 | tube 2 | … ]` — the core is the
//! ordinary `mlp.*_proj.weight` triple (so any older reader still opens
//! the file and sees a valid, narrower model), each tube an extra
//! `mlp.*_proj.tube{k}.weight` triple. A task is a first-class task mask
//! whose bits are on for the core and for its own tubes.

use anyhow::{Context, bail};
use cortiq_core::mask::{MaskCatalog, MaskPriority, TaskMask};
use cortiq_core::{CmfModel, ModelArch, TensorDtype, TensorEntry, TensorSpec};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::convert::{Quant, quantize_2d};

/// Normalize an existing file, or normalize the parent of a prospective
/// output.  The latter catches `model.cmf` versus `./model.cmf` before a
/// writer has created the output leaf.
fn collision_key(path: &Path) -> PathBuf {
    if let Ok(full) = std::fs::canonicalize(path) {
        return full;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    match (std::fs::canonicalize(parent), path.file_name()) {
        (Ok(parent), Some(name)) => parent.join(name),
        _ => path.to_path_buf(),
    }
}

/// Whether two paths name the same existing file, including a symlink or a
/// hard-link alias on Unix.  Canonical path comparison alone cannot detect
/// hard links, which is especially relevant because sidecar workflows often
/// use hard links to avoid duplicating a large backbone.
pub(crate) fn paths_collide(a: &Path, b: &Path) -> bool {
    if collision_key(a) == collision_key(b) {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(a), Ok(b)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return a.dev() == b.dev() && a.ino() == b.ino();
        }
    }
    false
}

fn ensure_distinct_bake_paths(
    model_path: &str,
    plan_path: &str,
    output: &str,
) -> anyhow::Result<()> {
    let output = Path::new(output);
    if paths_collide(output, Path::new(model_path)) {
        bail!("tube bake: --output must differ from the source model");
    }
    if paths_collide(output, Path::new(plan_path)) {
        bail!("tube bake: --output must differ from the input plan");
    }
    Ok(())
}

fn quant_of(d: TensorDtype) -> Option<Quant> {
    Some(match d {
        TensorDtype::Q8Row => Quant::Q8Row,
        TensorDtype::Q8_2f => Quant::Q8_2f,
        TensorDtype::Q4Block => Quant::Q4Block,
        TensorDtype::Q4Tiled => Quant::Q4Tiled,
        TensorDtype::Q4TiledP => Quant::Q4TiledP,
        TensorDtype::Q2TiledP => Quant::Q2TiledP,
        TensorDtype::F16 => Quant::F16,
        TensorDtype::Vbit | TensorDtype::VbitRo => Quant::Vbit,
        _ => return None,
    })
}

/// One layer's plan: the new order of the neurons that survive, and the
/// widths of the segments that order is cut into.
#[derive(Clone, Debug)]
struct LayerPlan {
    order: Vec<usize>,
    widths: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MtpMode {
    Preserve,
    Compact,
    Drop,
}

impl MtpMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Preserve => "preserve",
            Self::Compact => "compact",
            Self::Drop => "drop",
        }
    }
}

#[derive(Clone, Debug)]
struct MtpPlan {
    mode: MtpMode,
    layers: Vec<LayerPlan>,
}

#[derive(Clone, Debug)]
struct CanonicalMtp {
    widths: Vec<usize>,
    has_down_t: Vec<bool>,
}

fn parse_layer_plan(value: &Value, what: &str) -> anyhow::Result<LayerPlan> {
    let parse_numbers = |key: &str| -> anyhow::Result<Vec<usize>> {
        value
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("{what}.{key} must be an array"))?
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let raw = item.as_u64().with_context(|| {
                    format!("{what}.{key}[{index}] must be a non-negative integer")
                })?;
                usize::try_from(raw).with_context(|| {
                    format!("{what}.{key}[{index}]={raw} does not fit this platform")
                })
            })
            .collect()
    };
    Ok(LayerPlan {
        order: parse_numbers("order")?,
        widths: parse_numbers("widths")?,
    })
}

fn parse_layer_plans(value: &Value, what: &str) -> anyhow::Result<Vec<LayerPlan>> {
    value
        .as_array()
        .with_context(|| format!("{what} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, item)| parse_layer_plan(item, &format!("{what}[{index}]")))
        .collect()
}

fn validate_layer_plan(plan: &LayerPlan, source_width: usize, what: &str) -> anyhow::Result<()> {
    let sum = plan
        .widths
        .iter()
        .enumerate()
        .try_fold(0usize, |sum, (segment, &width)| {
            if width == 0 {
                bail!("{what}: segment {segment} has zero width");
            }
            sum.checked_add(width)
                .with_context(|| format!("{what}: segment widths overflow usize"))
        })?;
    if sum != plan.order.len() {
        bail!("{what}: widths sum {sum} != order len {}", plan.order.len());
    }
    if plan.widths.is_empty() {
        bail!("{what}: the core segment must be non-empty");
    }
    let mut seen = HashSet::with_capacity(plan.order.len());
    for &channel in &plan.order {
        if channel >= source_width {
            bail!("{what}: channel {channel} is out of range (width {source_width})");
        }
        if !seen.insert(channel) {
            bail!("{what}: channel {channel} occurs more than once");
        }
    }
    Ok(())
}

// Plan-v2 MTP behavior is explicit.  In particular, old plans which used
// drop_mtp cannot silently remove a speculative draft.
fn parse_mtp_plan(plan: &Value) -> anyhow::Result<MtpPlan> {
    if plan.get("drop_mtp").and_then(Value::as_bool) == Some(true) {
        bail!("legacy drop_mtp is unsupported; use the explicit mtp.mode=drop plan-v2 field");
    }
    let Some(raw) = plan.get("mtp") else {
        return Ok(MtpPlan {
            mode: MtpMode::Preserve,
            layers: Vec::new(),
        });
    };
    let object = raw.as_object().context("plan.mtp must be an object")?;
    let mode = match object
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("preserve")
    {
        "preserve" => MtpMode::Preserve,
        "compact" => MtpMode::Compact,
        "drop" => MtpMode::Drop,
        other => bail!("plan.mtp.mode must be preserve, compact, or drop (got {other:?})"),
    };
    let layers = match mode {
        MtpMode::Compact => parse_layer_plans(
            object
                .get("layers")
                .context("plan.mtp.layers is required for mtp.mode=compact")?,
            "plan.mtp.layers",
        )?,
        MtpMode::Preserve | MtpMode::Drop => {
            if object.contains_key("layers") {
                bail!("plan.mtp.layers is valid only for mtp.mode=compact");
            }
            Vec::new()
        }
    };
    Ok(MtpPlan { mode, layers })
}

// A compact/drop MTP plan with no legacy top-level trunk fields is an MTP-only
// operation.  This is also spelled explicitly by trunk.mode=preserve.
fn trunk_is_preserved(plan: &Value, mtp: &MtpPlan) -> anyhow::Result<bool> {
    let has_trunk_fields = ["tasks", "layers", "active"]
        .iter()
        .any(|key| plan.get(*key).is_some());
    if let Some(raw) = plan.get("trunk") {
        let object = raw.as_object().context("plan.trunk must be an object")?;
        match object
            .get("mode")
            .and_then(Value::as_str)
            .context("plan.trunk.mode is required")?
        {
            "preserve" => {
                if has_trunk_fields {
                    bail!(
                        "plan.trunk.mode=preserve cannot be combined with top-level tasks/layers/active"
                    );
                }
                Ok(true)
            }
            "bake" | "compact" => Ok(false),
            other => bail!("unknown plan.trunk.mode {other:?}"),
        }
    } else {
        Ok(!has_trunk_fields && mtp.mode != MtpMode::Preserve)
    }
}

fn parse_active(
    value: &Value,
    task_count: usize,
    layer_count: usize,
) -> anyhow::Result<Vec<Vec<Vec<bool>>>> {
    let rows = value.as_array().context("plan.active must be an array")?;
    if rows.len() != task_count {
        bail!("plan.active has {} rows, {task_count} tasks", rows.len());
    }
    let per_layer = rows
        .first()
        .and_then(Value::as_array)
        .and_then(|task| task.first())
        .is_some_and(Value::is_array);
    rows.iter()
        .enumerate()
        .map(|(task_index, task)| -> anyhow::Result<Vec<Vec<bool>>> {
            let task = task
                .as_array()
                .with_context(|| format!("plan.active[{task_index}] must be an array"))?;
            let parsed = if per_layer {
                task.iter()
                    .enumerate()
                    .map(|(layer_index, layer)| -> anyhow::Result<Vec<bool>> {
                        layer
                            .as_array()
                            .with_context(|| {
                                format!(
                                    "plan.active[{task_index}][{layer_index}] must be an array"
                                )
                            })?
                            .iter()
                            .enumerate()
                            .map(|(segment, bit)| {
                                bit.as_bool().with_context(|| {
                                    format!(
                                        "plan.active[{task_index}][{layer_index}][{segment}] must be a bool"
                                    )
                                })
                            })
                            .collect()
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
            } else {
                let one = task
                    .iter()
                    .enumerate()
                    .map(|(segment, bit)| {
                        bit.as_bool().with_context(|| {
                            format!("plan.active[{task_index}][{segment}] must be a bool")
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                vec![one; layer_count]
            };
            if parsed.len() != layer_count {
                bail!(
                    "plan.active[{task_index}] has {} layer rows, model has {layer_count}",
                    parsed.len()
                );
            }
            Ok(parsed)
        })
        .collect()
}

fn tensor<'a>(model: &'a CmfModel, name: &str) -> anyhow::Result<&'a TensorEntry> {
    model
        .tensors
        .iter()
        .find(|entry| entry.name == name)
        .with_context(|| format!("missing tensor {name}"))
}

fn dequant(model: &CmfModel, name: &str) -> anyhow::Result<Vec<f32>> {
    let entry = tensor(model, name)?;
    let mut out = vec![0.0; entry.shape.iter().product()];
    cortiq_core::quant::dequant_tensor(entry, model.tensor_bytes(name)?, &mut out)
        .map_err(|error| anyhow::anyhow!(error))?;
    Ok(out)
}

fn expect_shape(model: &CmfModel, name: &str, expected: &[usize]) -> anyhow::Result<()> {
    let actual = &tensor(model, name)?.shape;
    if actual.as_slice() != expected {
        bail!("{name}: expected {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn mtp_mlp_names(layer: usize) -> [String; 4] {
    let prefix = format!("model.mtp.layers.{layer}.mlp.");
    [
        format!("{prefix}gate_proj.weight"),
        format!("{prefix}up_proj.weight"),
        format!("{prefix}down_proj.weight"),
        format!("{prefix}down_proj.t.weight"),
    ]
}

fn mtp_payload_present(model: &CmfModel) -> bool {
    model
        .tensors
        .iter()
        .any(|entry| entry.name.starts_with("model.mtp."))
}

// The compactor only transforms canonical dense local MLPs.  This works for
// the generic one-block MTP and for multi-block MiMo sidecars because their
// differing surrounding tensors are copied byte-for-byte.  Routed experts,
// old MTP tubes, DSV4's different namespace, and header/tensor disagreement
// are rejected rather than guessed at.
fn validate_canonical_dense_mtp(
    model: &CmfModel,
    arch: &ModelArch,
) -> anyhow::Result<CanonicalMtp> {
    let config = arch
        .mtp
        .as_ref()
        .context("MTP tensors are present but header.arch.mtp is absent")?;
    if config.num_layers == 0 {
        bail!("header.arch.mtp.num_layers must be positive for MTP compact");
    }
    // The generic executor deliberately implements one canonical block;
    // MiMo owns its distinct multi-block draft executor.  Do not let a
    // direct `tube bake` manufacture a file the runtime cannot load.
    if config.num_layers != 1 && arch.arch_name != "mimo_v2" {
        bail!(
            "MTP compact supports multiple draft layers only for mimo_v2; {} declares {}",
            arch.arch_name,
            config.num_layers
        );
    }
    if model
        .tensors
        .iter()
        .any(|entry| entry.name.starts_with("model.mtp.0."))
    {
        bail!("MTP compact does not support DeepSeek-V4 model.mtp.N layouts");
    }

    let mut widths = Vec::with_capacity(config.num_layers);
    let mut has_down_t = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let [gate_name, up_name, down_name, down_t_name] = mtp_mlp_names(layer);
        let hidden = arch.hidden_size;
        let gate_shape = &tensor(model, &gate_name)?.shape;
        if gate_shape.len() != 2 || gate_shape[0] == 0 || gate_shape[1] != hidden {
            bail!("{gate_name}: expected non-empty [intermediate, {hidden}], got {gate_shape:?}");
        }
        let width = gate_shape[0];
        expect_shape(model, &up_name, &[width, hidden])?;
        expect_shape(model, &down_name, &[hidden, width])?;
        let down_t = model.tensors.iter().any(|entry| entry.name == down_t_name);
        if down_t {
            expect_shape(model, &down_t_name, &[width, hidden])?;
        }
        let prefix = format!("model.mtp.layers.{layer}.mlp.");
        for entry in model
            .tensors
            .iter()
            .filter(|entry| entry.name.starts_with(&prefix))
        {
            if entry.name != gate_name
                && entry.name != up_name
                && entry.name != down_name
                && entry.name != down_t_name
            {
                bail!(
                    "MTP compact only supports an unsplit dense MLP triple; unsupported tensor {}",
                    entry.name
                );
            }
        }
        widths.push(width);
        has_down_t.push(down_t);
    }
    for entry in &model.tensors {
        if let Some(rest) = entry.name.strip_prefix("model.mtp.layers.") {
            let layer = rest.split('.').next().unwrap_or_default();
            let index = layer.parse::<usize>().with_context(|| {
                format!("MTP compact found malformed layer namespace {}", entry.name)
            })?;
            if index >= config.num_layers {
                bail!(
                    "MTP compact header declares {} layers but found tensor {}",
                    config.num_layers,
                    entry.name
                );
            }
        }
    }
    if let Some(declared) = config.ffn_intermediate_sizes.as_deref() {
        if declared != widths.as_slice() {
            bail!(
                "header.arch.mtp.ffn_intermediate_sizes={declared:?} disagrees with MTP widths {widths:?}"
            );
        }
    }
    Ok(CanonicalMtp { widths, has_down_t })
}

/// A calibration-generated MTP plan is tied to the exact draft directory it
/// measured.  Unlike the main-file hash (which is only audit metadata for a
/// sidecar), this hash can be checked by `tube bake` because its input is the
/// draft file itself.  Requiring it prevents a saved calibrated plan from
/// being silently applied to a same-shaped but different draft.
fn validate_mtp_compact_source(plan: &Value, model: &CmfModel) -> anyhow::Result<()> {
    let selection = plan
        .get("selection")
        .context("MTP compact plan requires selection provenance")?
        .as_object()
        .context("MTP compact plan.selection must be an object")?;
    let recorded = selection
        .get("source_draft_directory_hash")
        .and_then(Value::as_str)
        .context("MTP compact plan.selection.source_draft_directory_hash is required")?;
    let parsed = u64::from_str_radix(recorded, 16).with_context(
        || "MTP compact plan.selection.source_draft_directory_hash must be a hexadecimal u64",
    )?;
    let actual = model.dir_hash();
    if parsed != actual {
        bail!(
            "MTP compact plan targets draft directory hash {recorded}, but input is {:016x}",
            actual
        );
    }
    Ok(())
}

fn is_mtp_compacted_tensor(name: &str, layer_count: usize) -> bool {
    (0..layer_count).any(|layer| {
        mtp_mlp_names(layer)
            .iter()
            .any(|candidate| candidate == name)
    })
}

fn compact_rows(values: &[f32], selected: &[usize], source_rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), source_rows * cols);
    let mut output = Vec::with_capacity(selected.len() * cols);
    for &row in selected {
        output.extend_from_slice(&values[row * cols..(row + 1) * cols]);
    }
    output
}

fn compact_down_columns(
    values: &[f32],
    rows: usize,
    source_cols: usize,
    selected: &[usize],
) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * source_cols);
    let mut output = Vec::with_capacity(rows * selected.len());
    for row in 0..rows {
        for &column in selected {
            output.push(values[row * source_cols + column]);
        }
    }
    output
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut output = Vec::with_capacity(values.len());
    for column in 0..cols {
        for row in 0..rows {
            output.push(values[row * cols + column]);
        }
    }
    output
}

// MTP stays one physical dense triple per layer.  A draft has no task mask,
// and tube tensors would both slow it down and disqualify the fast speculative
// GPU graph.  The plan's widths remain in provenance for reproducibility.
fn append_compacted_mtp_layer(
    model: &CmfModel,
    layer: usize,
    plan: &LayerPlan,
    source_width: usize,
    has_down_t: bool,
    hidden: usize,
    tensors: &mut Vec<TensorSpec>,
) -> anyhow::Result<()> {
    let [gate_name, up_name, down_name, down_t_name] = mtp_mlp_names(layer);
    let gate = dequant(model, &gate_name)?;
    let up = dequant(model, &up_name)?;
    let down = dequant(model, &down_name)?;
    let gate_q =
        quant_of(tensor(model, &gate_name)?.dtype).context("unsupported MTP gate dtype")?;
    let up_q = quant_of(tensor(model, &up_name)?.dtype).context("unsupported MTP up dtype")?;
    let down_q =
        quant_of(tensor(model, &down_name)?.dtype).context("unsupported MTP down dtype")?;
    let width = plan.order.len();
    let gate = compact_rows(&gate, &plan.order, source_width, hidden);
    let up = compact_rows(&up, &plan.order, source_width, hidden);
    let down = compact_down_columns(&down, hidden, source_width, &plan.order);
    // Tiled low-bit codecs group the input axis, so a hand-written plan
    // with an unaligned compact width must fall back.  Plain f16/q8 rows do
    // not have that restriction and must not lose precision merely because
    // their selected draft width is not a codec-group multiple.
    let down_q = if matches!(down_q, Quant::F16 | Quant::Q8Row | Quant::Q8_2f) || width % 32 == 0 {
        down_q
    } else {
        Quant::Q8_2f
    };
    for (name, values, rows, cols, quant) in [
        (&gate_name, &gate, width, hidden, gate_q),
        (&up_name, &up, width, hidden, up_q),
        (&down_name, &down, hidden, width, down_q),
    ] {
        let (dtype, data) = quantize_2d(quant, values, rows, cols);
        tensors.push(TensorSpec {
            name: name.to_string(),
            dtype,
            shape: vec![rows, cols],
            data,
        });
    }
    if has_down_t {
        // Rebuild instead of slicing the old cached transpose.  This keeps it
        // mathematically coherent with a permutation as well as a subset.
        let t_q = quant_of(tensor(model, &down_t_name)?.dtype)
            .context("unsupported MTP down_proj.t dtype")?;
        let down_t = transpose(&down, hidden, width);
        let (dtype, data) = quantize_2d(t_q, &down_t, width, hidden);
        tensors.push(TensorSpec {
            name: down_t_name,
            dtype,
            shape: vec![width, hidden],
            data,
        });
    }
    Ok(())
}

fn cmd_mtp_only_bake(
    model: &Arc<CmfModel>,
    plan_document: &Value,
    mtp: &MtpPlan,
    output: &str,
) -> anyhow::Result<()> {
    let arch = model.arch().clone();
    let canonical = if mtp.mode == MtpMode::Compact {
        validate_mtp_compact_source(plan_document, model)?;
        if !mtp_payload_present(model) {
            bail!("plan requests MTP compact, but source carries no model.mtp tensors");
        }
        let canonical = validate_canonical_dense_mtp(model, &arch)?;
        if mtp.layers.len() != canonical.widths.len() {
            bail!(
                "MTP compact plan has {} layers, source header declares {}",
                mtp.layers.len(),
                canonical.widths.len()
            );
        }
        for (layer, (plan, &width)) in mtp.layers.iter().zip(&canonical.widths).enumerate() {
            validate_layer_plan(plan, width, &format!("plan.mtp.layers[{layer}]"))?;
        }
        Some(canonical)
    } else {
        None
    };

    let layer_count = canonical.as_ref().map_or(0, |item| item.widths.len());
    let mut tensors = Vec::with_capacity(model.tensors.len() + layer_count * 4);
    for entry in &model.tensors {
        let omit = match mtp.mode {
            MtpMode::Preserve => false,
            MtpMode::Drop => entry.name.starts_with("model.mtp."),
            MtpMode::Compact => is_mtp_compacted_tensor(&entry.name, layer_count),
        };
        if !omit {
            tensors.push(TensorSpec {
                name: entry.name.clone(),
                dtype: entry.dtype,
                shape: entry.shape.clone(),
                data: model.tensor_bytes(&entry.name)?.to_vec(),
            });
        }
    }
    if let Some(canonical) = &canonical {
        for (layer, plan) in mtp.layers.iter().enumerate() {
            append_compacted_mtp_layer(
                model,
                layer,
                plan,
                canonical.widths[layer],
                canonical.has_down_t[layer],
                arch.hidden_size,
                &mut tensors,
            )?;
        }
    }

    let mut header = model.header.clone();
    match mtp.mode {
        MtpMode::Preserve => {}
        MtpMode::Drop => header.arch.mtp = None,
        MtpMode::Compact => {
            let config = header
                .arch
                .mtp
                .as_mut()
                .context("validated MTP source lost header.arch.mtp")?;
            config.ffn_intermediate_sizes =
                Some(mtp.layers.iter().map(|plan| plan.order.len()).collect());
        }
    }
    let mut provenance = header.provenance.take().unwrap_or_else(|| json!({}));
    let mut tubes = provenance
        .get("tubes")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if !tubes.is_object() {
        tubes = json!({});
    }
    tubes["mtp"] = match mtp.mode {
        MtpMode::Preserve => json!({"mode": mtp.mode.as_str(), "rewritten": false}),
        MtpMode::Drop => json!({"plan_version": 2, "mode": mtp.mode.as_str(), "rewritten": true}),
        MtpMode::Compact => {
            let source = canonical.as_ref().context("missing validated MTP")?;
            json!({
                "plan_version": 2,
                "mode": mtp.mode.as_str(),
                "layout": "canonical-dense-mtp-ffn-v1",
                "source_widths": source.widths,
                "post_widths": mtp.layers.iter().map(|plan| plan.order.len()).collect::<Vec<_>>(),
                "logical_segments": mtp.layers.iter().map(|plan| plan.widths.clone()).collect::<Vec<_>>(),
                "physical_segments_per_layer": 1,
                "down_proj_t_rebuilt": source.has_down_t,
                "non_ffn_tensors": "copied-bytewise",
            })
        }
    };
    if let Some(selection) = plan_document.get("selection") {
        tubes["mtp"]["selection"] = selection.clone();
    }
    provenance["tubes"] = tubes;
    header.provenance = Some(provenance);
    CmfModel::write(
        output,
        &header,
        &tensors,
        (!model.masks.masks.is_empty()).then_some(&model.masks),
        model.vocab.as_deref(),
    )?;
    match mtp.mode {
        MtpMode::Preserve => println!("MTP: copied bytewise"),
        MtpMode::Drop => {
            println!("MTP: explicitly dropped model.mtp tensors and header declaration")
        }
        MtpMode::Compact => println!(
            "MTP: compacted dense draft widths {:?} -> {:?}; kept unsplit for speculative GPU graphs",
            canonical.as_ref().expect("validated MTP").widths,
            mtp.layers
                .iter()
                .map(|plan| plan.order.len())
                .collect::<Vec<_>>()
        ),
    }
    Ok(())
}

pub fn cmd_tube_bake(model_path: &str, plan_path: &str, output: &str) -> anyhow::Result<()> {
    ensure_distinct_bake_paths(model_path, plan_path, output)?;
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "tube-bake")?;
    let arch = model.arch().clone();
    let (nl, hidden, inter) = (arch.num_layers, arch.hidden_size, arch.intermediate_size);
    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(plan_path).with_context(|| format!("reading {plan_path}"))?,
    )?;
    let mtp = parse_mtp_plan(&plan)?;
    if trunk_is_preserved(&plan, &mtp)? {
        return cmd_mtp_only_bake(&model, &plan, &mtp, output);
    }
    let canonical_mtp = if mtp.mode == MtpMode::Compact {
        validate_mtp_compact_source(&plan, &model)?;
        if !mtp_payload_present(&model) {
            bail!("plan requests MTP compact, but source carries no model.mtp tensors");
        }
        let canonical = validate_canonical_dense_mtp(&model, &arch)?;
        if mtp.layers.len() != canonical.widths.len() {
            bail!(
                "MTP compact plan has {} layers, source header declares {}",
                mtp.layers.len(),
                canonical.widths.len()
            );
        }
        for (layer, (item, &width)) in mtp.layers.iter().zip(&canonical.widths).enumerate() {
            validate_layer_plan(item, width, &format!("plan.mtp.layers[{layer}]"))?;
        }
        Some(canonical)
    } else {
        None
    };
    let tasks: Vec<String> = plan["tasks"]
        .as_array()
        .context("plan.tasks")?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("plan.tasks[{index}] must be a string"))
        })
        .collect::<anyhow::Result<_>>()?;
    let layers = parse_layer_plans(&plan["layers"], "plan.layers")?;
    if layers.len() != nl {
        bail!("plan has {} layers, model has {nl}", layers.len());
    }
    let active = parse_active(&plan["active"], tasks.len(), nl)?;
    for (li, l) in layers.iter().enumerate() {
        validate_layer_plan(l, inter, &format!("plan.layers[{li}]"))?;
    }
    let nseg = layers.iter().map(|l| l.widths.len()).max().unwrap_or(1);

    let deq = |name: &str| -> anyhow::Result<Vec<f32>> {
        let e = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("missing tensor {name}"))?;
        let mut out = vec![0f32; e.shape.iter().product()];
        cortiq_core::quant::dequant_tensor(e, model.tensor_bytes(name)?, &mut out)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(out)
    };

    // The stack FFNs are rebuilt below.  MTP is copied bytewise unless an
    // explicit v2 compact/drop mode says otherwise.
    let is_stack_ffn = |name: &str| -> bool {
        name.strip_prefix("model.layers.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(li, rest)| {
                li.parse::<usize>().is_ok()
                    && (rest.starts_with("mlp.gate_proj.")
                        || rest.starts_with("mlp.up_proj.")
                        || rest.starts_with("mlp.down_proj."))
            })
    };
    let compact_mtp_layers = canonical_mtp.as_ref().map_or(0, |item| item.widths.len());
    let mut tensors: Vec<TensorSpec> = Vec::new();
    for t in &model.tensors {
        let omit_mtp = match mtp.mode {
            MtpMode::Preserve => false,
            MtpMode::Drop => t.name.starts_with("model.mtp."),
            MtpMode::Compact => is_mtp_compacted_tensor(&t.name, compact_mtp_layers),
        };
        if is_stack_ffn(&t.name) || omit_mtp {
            continue;
        }
        tensors.push(TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model.tensor_bytes(&t.name)?.to_vec(),
        });
    }

    let mut max_width = 0usize;
    let mut kept_total = 0usize;
    for (li, lp) in layers.iter().enumerate() {
        let gate = deq(&format!("model.layers.{li}.mlp.gate_proj.weight"))?;
        let up = deq(&format!("model.layers.{li}.mlp.up_proj.weight"))?;
        let down = deq(&format!("model.layers.{li}.mlp.down_proj.weight"))?;
        let src_dtype = model
            .tensors
            .iter()
            .find(|t| t.name == format!("model.layers.{li}.mlp.gate_proj.weight"))
            .map(|t| t.dtype)
            .context("gate tensor missing")?;
        let q = quant_of(src_dtype).context("unsupported ffn dtype")?;
        max_width = max_width.max(lp.order.len());
        kept_total += lp.order.len();
        let mut at = 0usize;
        // Written tubes are numbered densely (the loader stops at the
        // first gap), so a zero-width segment must not consume a number.
        let mut tube_no = 0usize;
        for (k, &w) in lp.widths.iter().enumerate() {
            let rows = &lp.order[at..at + w];
            at += w;
            if w == 0 {
                continue; // a tube this layer does not need
            }
            if k > 0 {
                tube_no += 1;
            }
            let mut gate_k = Vec::with_capacity(w * hidden);
            let mut up_k = Vec::with_capacity(w * hidden);
            for &r in rows {
                gate_k.extend_from_slice(&gate[r * hidden..(r + 1) * hidden]);
                up_k.extend_from_slice(&up[r * hidden..(r + 1) * hidden]);
            }
            let mut down_k = Vec::with_capacity(hidden * w);
            for r in 0..hidden {
                for &c in rows {
                    down_k.push(down[r * inter + c]);
                }
            }
            // Grouped codecs need the IN dim on a group boundary; the
            // planner aligns tube widths to 32 for exactly this reason.
            let q_down = if w % 32 == 0 { q } else { Quant::Q8_2f };
            let suffix = if k == 0 {
                String::new()
            } else {
                format!(".tube{tube_no}")
            };
            for (name, vals, r, c, qq) in [
                (
                    format!("model.layers.{li}.mlp.gate_proj{suffix}.weight"),
                    &gate_k,
                    w,
                    hidden,
                    q,
                ),
                (
                    format!("model.layers.{li}.mlp.up_proj{suffix}.weight"),
                    &up_k,
                    w,
                    hidden,
                    q,
                ),
                (
                    format!("model.layers.{li}.mlp.down_proj{suffix}.weight"),
                    &down_k,
                    hidden,
                    w,
                    q_down,
                ),
            ] {
                let (dtype, data) = quantize_2d(qq, vals, r, c);
                tensors.push(TensorSpec {
                    name,
                    dtype,
                    shape: vec![r, c],
                    data,
                });
            }
        }
    }

    if let Some(canonical) = &canonical_mtp {
        for (layer, plan) in mtp.layers.iter().enumerate() {
            append_compacted_mtp_layer(
                &model,
                layer,
                plan,
                canonical.widths[layer],
                canonical.has_down_t[layer],
                hidden,
                &mut tensors,
            )?;
        }
    }

    let mut header = model.header.clone();
    header.arch.intermediate_size = max_width;
    match mtp.mode {
        MtpMode::Preserve => {}
        MtpMode::Drop => header.arch.mtp = None,
        MtpMode::Compact => {
            let config = header
                .arch
                .mtp
                .as_mut()
                .context("validated MTP source lost header.arch.mtp")?;
            config.ffn_intermediate_sizes =
                Some(mtp.layers.iter().map(|plan| plan.order.len()).collect());
        }
    }
    // Task masks over the NEW index space: the core plus this task's
    // tubes, every neuron of an active segment set.
    let mut catalog = MaskCatalog::empty();
    for (ti, task) in tasks.iter().enumerate() {
        let mut ffn_masks = Vec::with_capacity(nl);
        let mut alive = 0usize;
        let mut total = 0usize;
        for (li, lp) in layers.iter().enumerate() {
            let width: usize = lp.order.len();
            let mut bits = vec![0u8; width.div_ceil(8)];
            let mut at = 0usize;
            let row = &active[ti][li];
            for (k, &w) in lp.widths.iter().enumerate() {
                if k == 0 || row.get(k).copied().unwrap_or(false) {
                    for n in at..at + w {
                        bits[n / 8] |= 1 << (n % 8);
                    }
                    alive += w;
                }
                at += w;
            }
            total += width;
            ffn_masks.push(bits);
        }
        let mut hb = vec![0u8; arch.num_attention_heads.div_ceil(8)];
        for h in 0..arch.num_attention_heads {
            hb[h / 8] |= 1 << (h % 8);
        }
        catalog.masks.push(TaskMask {
            task_id: ti as u32 + 1,
            name: task.clone(),
            description: Some(format!(
                "tube: {:.1}% of the stored FFN active",
                (1.0 - (1.0 - alive as f64 / total.max(1) as f64)) * 100.0
            )),
            sparsity: 1.0 - alive as f32 / total.max(1) as f32,
            quality: None,
            ffn_masks,
            head_masks: vec![hb.clone(); nl],
            layer_gates: vec![true; nl],
            expert_masks: Vec::new(),
            parent: None,
            priority: MaskPriority::Normal,
            has_hot_pack: false,
        });
    }
    if let Some(first) = tasks.first() {
        catalog.default_task = first.clone();
    }
    let mut prov = header
        .provenance
        .take()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut tube_provenance = serde_json::json!({
        "recipe": "task-tube defrag (permutation + segment cut)",
        "pre_intermediate": inter,
        "post_intermediate_max": max_width,
        "segments": nseg,
        "widths": layers.iter().map(|l| l.widths.clone()).collect::<Vec<_>>(),
        "tasks": tasks,
        "active": active,
    });
    tube_provenance["mtp"] = match mtp.mode {
        MtpMode::Preserve => json!({"mode": mtp.mode.as_str(), "rewritten": false}),
        MtpMode::Drop => json!({"plan_version": 2, "mode": mtp.mode.as_str(), "rewritten": true}),
        MtpMode::Compact => {
            let source = canonical_mtp.as_ref().context("missing validated MTP")?;
            json!({
                "plan_version": 2,
                "mode": mtp.mode.as_str(),
                "layout": "canonical-dense-mtp-ffn-v1",
                "source_widths": source.widths,
                "post_widths": mtp.layers.iter().map(|item| item.order.len()).collect::<Vec<_>>(),
                "logical_segments": mtp.layers.iter().map(|item| item.widths.clone()).collect::<Vec<_>>(),
                "physical_segments_per_layer": 1,
                "down_proj_t_rebuilt": source.has_down_t,
                "non_ffn_tensors": "copied-bytewise",
            })
        }
    };
    if let Some(selection) = plan.get("selection") {
        tube_provenance["mtp"]["selection"] = selection.clone();
    }
    prov["tubes"] = tube_provenance;
    header.provenance = Some(prov);
    CmfModel::write(
        output,
        &header,
        &tensors,
        Some(&catalog),
        model.vocab.as_deref(),
    )?;
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let out_sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    println!(
        "tube bake: {nseg} segment(s), widths {:?}, kept {kept_total}/{} neurons\n\
         {model_path} {in_sz:.2} GB -> {output} {out_sz:.2} GB ({:+.1}%)",
        layers[0].widths,
        nl * inter,
        (out_sz / in_sz - 1.0) * 100.0
    );
    for (ti, task) in tasks.iter().enumerate() {
        let w: usize = layers
            .iter()
            .enumerate()
            .map(|(li, lp)| -> usize {
                lp.widths
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k == 0 || active[ti][li].get(*k).copied().unwrap_or(false))
                    .map(|(_, w)| *w)
                    .sum()
            })
            .sum();
        println!(
            "  {task:18} {:6} / {inter} neurons per layer ({:.1}% of the dense FFN)",
            w / nl,
            w as f64 / (nl * inter) as f64 * 100.0
        );
    }
    match mtp.mode {
        MtpMode::Preserve => {}
        MtpMode::Drop => {
            println!("MTP: explicitly dropped model.mtp tensors and header declaration")
        }
        MtpMode::Compact => println!(
            "MTP: compacted dense draft widths {:?} -> {:?}; kept unsplit for speculative GPU graphs",
            canonical_mtp.as_ref().expect("validated MTP").widths,
            mtp.layers
                .iter()
                .map(|item| item.order.len())
                .collect::<Vec<_>>()
        ),
    }
    Ok(())
}

#[cfg(test)]
mod mtp_plan_tests {
    use super::*;
    use cortiq_engine::{Pipeline, SamplerConfig};

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| cortiq_core::quant::f32_to_f16(*value).to_le_bytes())
            .collect()
    }

    fn f16_tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TensorSpec {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F16,
            shape: shape.to_vec(),
            data: f16_bytes(&values),
        }
    }

    /// A small sidecar header.  It need not contain a trunk: `tube bake`
    /// must be able to transform a MiMo-like MTP sidecar by itself.
    fn sidecar_header() -> cortiq_core::CmfHeader {
        use cortiq_core::{
            CMF_VERSION, CmfHeader, LayerType, ModelArch, MtpConfig, NormStyle, QuantType,
        };

        CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
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
            arch: ModelArch {
                arch_name: "mimo_v2".into(),
                hidden_size: 4,
                // This differs deliberately from the draft FFN width below:
                // an MTP-only rewrite must not alter trunk geometry.
                intermediate_size: 6,
                num_layers: 1,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                vocab_size: 8,
                layer_types: vec![LayerType::FullAttention],
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
                mla: None,
                no_rope_layers: None,
                attention_sink: false,
                activation_situ_beta: None,
                activation_situ_linear_beta: None,
                attn_logit_softcapping: None,
                attn_v_norm: false,
                qk_norm_after_rope: false,
                mtp: Some(MtpConfig {
                    num_layers: 1,
                    ffn_intermediate_sizes: Some(vec![4]),
                    share_lm_head: true,
                    share_embed: true,
                }),
                moe: None,
                qwen4_exp: None,
                deepseek_v41: None,
                linear_core: None,
                anchor_core: None,
                head_clusters: None,
                max_position_embeddings: 64,
                linear_conv_kernel_dim: None,
                linear_num_key_heads: None,
                linear_num_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                rope_freq_factors: None,
                logit_multiplier: None,
                g3n: None,
                kda_gate_lower_bound: None,
                num_loops: 1,
                loop_final_norm: false,
                prism_hadamard: None,
            },
            quant_type: QuantType::F16,
            provenance: None,
        }
    }

    /// Minimal runnable one-block model with a canonical MTP head.  This
    /// lets the physical writer test load its output back through the real
    /// loader and speculative verifier rather than only inspecting tensors.
    fn canonical_mtp_header() -> cortiq_core::CmfHeader {
        let mut header = sidecar_header();
        header.arch.arch_name = "qwen2".into();
        header.arch.intermediate_size = 4;
        header.arch.vocab_size = 16;
        header
    }

    fn generated_f16_tensor(name: &str, shape: &[usize], salt: usize) -> TensorSpec {
        let len = shape.iter().product();
        let values = (0..len)
            .map(|index| (((index * 17 + salt * 11 + 3) % 37) as f32 - 18.0) / 128.0)
            .collect();
        f16_tensor(name, shape, values)
    }

    fn append_canonical_mtp_block(tensors: &mut Vec<TensorSpec>, prefix: &str, salt: usize) {
        tensors.extend([
            f16_tensor(
                &format!("{prefix}input_layernorm.weight"),
                &[4],
                vec![1.0; 4],
            ),
            f16_tensor(
                &format!("{prefix}post_attention_layernorm.weight"),
                &[4],
                vec![1.0; 4],
            ),
            generated_f16_tensor(&format!("{prefix}self_attn.q_proj.weight"), &[4, 4], salt),
            generated_f16_tensor(
                &format!("{prefix}self_attn.k_proj.weight"),
                &[4, 4],
                salt + 1,
            ),
            generated_f16_tensor(
                &format!("{prefix}self_attn.v_proj.weight"),
                &[4, 4],
                salt + 2,
            ),
            generated_f16_tensor(
                &format!("{prefix}self_attn.o_proj.weight"),
                &[4, 4],
                salt + 3,
            ),
            generated_f16_tensor(&format!("{prefix}mlp.gate_proj.weight"), &[4, 4], salt + 4),
            generated_f16_tensor(&format!("{prefix}mlp.up_proj.weight"), &[4, 4], salt + 5),
            generated_f16_tensor(&format!("{prefix}mlp.down_proj.weight"), &[4, 4], salt + 6),
        ]);
    }

    fn canonical_mtp_tensors() -> Vec<TensorSpec> {
        let mut tensors = vec![
            generated_f16_tensor("model.embed_tokens.weight", &[16, 4], 1),
            f16_tensor("model.norm.weight", &[4], vec![1.0; 4]),
            generated_f16_tensor("lm_head.weight", &[16, 4], 2),
        ];
        append_canonical_mtp_block(&mut tensors, "model.layers.0.", 10);
        tensors.extend([
            f16_tensor("model.mtp.enorm.weight", &[4], vec![1.0; 4]),
            f16_tensor("model.mtp.hnorm.weight", &[4], vec![1.0; 4]),
            generated_f16_tensor("model.mtp.eh_proj.weight", &[4, 8], 30),
            f16_tensor("model.mtp.norm.weight", &[4], vec![1.0; 4]),
        ]);
        append_canonical_mtp_block(&mut tensors, "model.mtp.layers.0.", 40);
        tensors
    }

    #[test]
    fn mtp_defaults_to_preserve() {
        let plan = serde_json::json!({"tasks": [], "layers": [], "active": []});
        let mtp = parse_mtp_plan(&plan).unwrap();
        assert_eq!(mtp.mode, MtpMode::Preserve);
        assert!(mtp.layers.is_empty());
    }

    #[test]
    fn mtp_compact_requires_independent_layer_plan() {
        let error = parse_mtp_plan(&serde_json::json!({"mtp": {"mode": "compact"}}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("plan.mtp.layers"));
    }

    #[test]
    fn mtp_only_compaction_preserves_trunk() {
        let document = serde_json::json!({
            "mtp": {
                "mode": "compact",
                "layers": [{"order": [0, 1], "widths": [2]}]
            }
        });
        let mtp = parse_mtp_plan(&document).unwrap();
        assert!(trunk_is_preserved(&document, &mtp).unwrap());
    }

    #[test]
    fn plan_rejects_duplicate_channels() {
        let plan = LayerPlan {
            order: vec![0, 2, 2],
            widths: vec![3],
        };
        assert!(validate_layer_plan(&plan, 4, "mtp").is_err());
    }

    #[test]
    fn plan_rejects_zero_or_overflowing_segment_widths() {
        let zero = LayerPlan {
            order: vec![0],
            widths: vec![1, 0],
        };
        assert!(validate_layer_plan(&zero, 1, "mtp").is_err());

        let overflowing = LayerPlan {
            order: Vec::new(),
            widths: vec![usize::MAX, 1],
        };
        assert!(validate_layer_plan(&overflowing, 1, "mtp").is_err());
    }

    #[test]
    fn tube_bake_refuses_source_or_plan_as_output() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.cmf");
        let plan = dir.path().join("plan.json");
        std::fs::write(&source, b"cmf").unwrap();
        std::fs::write(&plan, b"{}").unwrap();

        assert!(
            ensure_distinct_bake_paths(
                source.to_str().unwrap(),
                plan.to_str().unwrap(),
                source.to_str().unwrap(),
            )
            .is_err()
        );
        assert!(
            ensure_distinct_bake_paths(
                source.to_str().unwrap(),
                plan.to_str().unwrap(),
                plan.to_str().unwrap(),
            )
            .is_err()
        );

        let source_hard_link = dir.path().join("source-hard-link.cmf");
        std::fs::hard_link(&source, &source_hard_link).unwrap();
        assert!(
            ensure_distinct_bake_paths(
                source.to_str().unwrap(),
                plan.to_str().unwrap(),
                source_hard_link.to_str().unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn selected_down_and_transpose_are_coherent() {
        // Original down is [[0, 1, 2], [3, 4, 5]]; selected order is [2, 0].
        let original = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let compact = compact_down_columns(&original, 2, 3, &[2, 0]);
        assert_eq!(compact, vec![2.0, 0.0, 5.0, 3.0]);
        assert_eq!(transpose(&compact, 2, 2), vec![2.0, 5.0, 0.0, 3.0]);
    }

    #[test]
    fn mtp_only_bake_rewrites_dense_ffn_and_preserves_draft_projection_and_norm() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.mtp.cmf");
        let plan = dir.path().join("plan.json");
        let output = dir.path().join("compact.mtp.cmf");

        let gate = vec![
            0.0, 1.0, 2.0, 3.0, // row 0
            10.0, 11.0, 12.0, 13.0, // row 1
            20.0, 21.0, 22.0, 23.0, // row 2
            30.0, 31.0, 32.0, 33.0, // row 3
        ];
        let up: Vec<f32> = gate.iter().map(|value| value + 100.0).collect();
        let down = vec![
            10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0, 40.0, 41.0,
            42.0, 43.0,
        ];
        let down_t = transpose(&down, 4, 4);
        let eh: Vec<f32> = (0..32).map(|value| 1000.0 + value as f32).collect();
        let enorm = vec![0.5, 1.0, 1.5, 2.0];
        let tensors = vec![
            f16_tensor("model.mtp.layers.0.mlp.gate_proj.weight", &[4, 4], gate),
            f16_tensor("model.mtp.layers.0.mlp.up_proj.weight", &[4, 4], up),
            f16_tensor("model.mtp.layers.0.mlp.down_proj.weight", &[4, 4], down),
            f16_tensor("model.mtp.layers.0.mlp.down_proj.t.weight", &[4, 4], down_t),
            f16_tensor("model.mtp.layers.0.eh_proj.weight", &[4, 8], eh.clone()),
            f16_tensor("model.mtp.layers.0.enorm.weight", &[4], enorm.clone()),
        ];
        CmfModel::write(&source, &sidecar_header(), &tensors, None, None).unwrap();
        let source_model = CmfModel::open(&source).unwrap();
        assert!(
            validate_mtp_compact_source(
                &serde_json::json!({
                    "selection": {"source_draft_directory_hash": "0000000000000000"}
                }),
                &source_model,
            )
            .is_err()
        );
        std::fs::write(
            &plan,
            serde_json::to_vec(&serde_json::json!({
                "format": "cmf-mtp-ffn-plan",
                "version": 2,
                "trunk": {"mode": "preserve"},
                "mtp": {
                    "mode": "compact",
                    "layers": [{"order": [3, 1], "widths": [2]}]
                },
                "selection": {
                    "test": "synthetic",
                    "source_draft_directory_hash": format!("{:016x}", source_model.dir_hash())
                }
            }))
            .unwrap(),
        )
        .unwrap();

        cmd_tube_bake(
            source.to_str().unwrap(),
            plan.to_str().unwrap(),
            output.to_str().unwrap(),
        )
        .unwrap();

        let compact = CmfModel::open(&output).unwrap();
        assert!(compact.verify().is_empty(), "{:?}", compact.verify());
        assert_eq!(compact.arch().intermediate_size, 6, "trunk width changed");
        assert_eq!(
            compact
                .arch()
                .mtp
                .as_ref()
                .and_then(|config| config.ffn_intermediate_sizes.as_deref()),
            Some(&[2][..])
        );
        assert_eq!(
            compact
                .tensor("model.mtp.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .shape,
            vec![2, 4]
        );
        assert_eq!(
            compact
                .tensor("model.mtp.layers.0.mlp.up_proj.weight")
                .unwrap()
                .shape,
            vec![2, 4]
        );
        assert_eq!(
            compact
                .tensor("model.mtp.layers.0.mlp.down_proj.weight")
                .unwrap()
                .shape,
            vec![4, 2]
        );
        assert_eq!(
            compact
                .tensor("model.mtp.layers.0.mlp.down_proj.t.weight")
                .unwrap()
                .shape,
            vec![2, 4]
        );
        assert_eq!(
            compact
                .tensor_bytes("model.mtp.layers.0.mlp.gate_proj.weight")
                .unwrap(),
            f16_bytes(&[30.0, 31.0, 32.0, 33.0, 10.0, 11.0, 12.0, 13.0])
        );
        assert_eq!(
            compact
                .tensor_bytes("model.mtp.layers.0.mlp.down_proj.weight")
                .unwrap(),
            f16_bytes(&[13.0, 11.0, 23.0, 21.0, 33.0, 31.0, 43.0, 41.0])
        );
        assert_eq!(
            compact
                .tensor_bytes("model.mtp.layers.0.mlp.down_proj.t.weight")
                .unwrap(),
            f16_bytes(&[13.0, 23.0, 33.0, 43.0, 11.0, 21.0, 31.0, 41.0])
        );
        // These projection/norm tensors are hidden-width, not FFN-axis
        // tensors, so they must stay byte-for-byte intact.
        assert_eq!(
            compact
                .tensor_bytes("model.mtp.layers.0.eh_proj.weight")
                .unwrap(),
            f16_bytes(&eh)
        );
        assert_eq!(
            compact
                .tensor_bytes("model.mtp.layers.0.enorm.weight")
                .unwrap(),
            f16_bytes(&enorm)
        );
        assert_eq!(
            compact.header.provenance.as_ref().unwrap()["tubes"]["mtp"]["selection"]["test"],
            "synthetic"
        );
    }

    #[test]
    fn compacted_serialized_mtp_loads_and_speculative_greedy_stays_exact() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.cmf");
        let plan = dir.path().join("plan.json");
        let output = dir.path().join("compact.cmf");
        CmfModel::write(
            &source,
            &canonical_mtp_header(),
            &canonical_mtp_tensors(),
            None,
            None,
        )
        .unwrap();
        let source_model = CmfModel::open(&source).unwrap();
        std::fs::write(
            &plan,
            serde_json::to_vec(&serde_json::json!({
                "format": "cmf-mtp-ffn-plan",
                "version": 2,
                "trunk": {"mode": "preserve"},
                "mtp": {
                    "mode": "compact",
                    "layers": [{"order": [3, 1], "widths": [2]}]
                },
                "selection": {
                    "source_draft_directory_hash": format!("{:016x}", source_model.dir_hash())
                }
            }))
            .unwrap(),
        )
        .unwrap();
        cmd_tube_bake(
            source.to_str().unwrap(),
            plan.to_str().unwrap(),
            output.to_str().unwrap(),
        )
        .unwrap();

        let model = Arc::new(CmfModel::open(&output).unwrap());
        let mut sampler = SamplerConfig::default();
        sampler.temperature = 0.0;
        let mut plain = Pipeline::from_model(&model, sampler.clone()).unwrap();
        plain.ignore_eos = true;
        plain.speculative = false;
        assert_eq!(plain.probe_mtp_ffn(&[1, 2, 3, 4]).unwrap()[0].mass.len(), 2);
        let plain = plain
            .generate_from_ids(&[1, 2, 3, 4], 8, None, None)
            .unwrap();

        let mut speculative = Pipeline::from_model(&model, sampler).unwrap();
        speculative.ignore_eos = true;
        speculative.speculative = true;
        let speculative = speculative
            .generate_from_ids(&[1, 2, 3, 4], 8, None, None)
            .unwrap();
        assert!(speculative.mtp_drafted > 0, "MTP draft did not run");
        assert!(speculative.mtp_accepted <= speculative.mtp_drafted);
        assert_eq!(plain.token_ids, speculative.token_ids);
    }
}

/// Store each dense layer's `down_proj` a second time, transposed.
///
/// `[hidden, inter]` puts one neuron's down weights in a COLUMN, and a
/// column is strided — per-token neuron selection then saves arithmetic
/// and not one byte. The transpose `[inter, hidden]` makes that neuron a
/// contiguous row, which is what `dense_ffn_dynamic` reads. The original
/// stays (the dense path still wants it), so the file grows by a third
/// of its FFN.
pub fn cmd_ffn_transpose(model_path: &str, output: &str) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "ffn-transpose")?;
    let arch = model.arch().clone();
    let (nl, hidden, inter) = (arch.num_layers, arch.hidden_size, arch.intermediate_size);
    let mut tensors: Vec<TensorSpec> = model
        .tensors
        .iter()
        .map(|t| -> anyhow::Result<TensorSpec> {
            Ok(TensorSpec {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                data: model.tensor_bytes(&t.name)?.to_vec(),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    let mut added = 0usize;
    for li in 0..nl {
        let name = format!("model.layers.{li}.mlp.down_proj.weight");
        let Some(e) = model.tensors.iter().find(|t| t.name == name) else {
            continue; // MoE layer, or a tube layer whose core is elsewhere
        };
        let (r, c) = (e.shape[0], e.shape[1]);
        if r != hidden {
            bail!("{name}: shape [{r}, {c}] is not [hidden, inter]");
        }
        let q = quant_of(e.dtype).context("unsupported down dtype")?;
        let mut w = vec![0f32; r * c];
        cortiq_core::quant::dequant_tensor(e, model.tensor_bytes(&name)?, &mut w)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut t = vec![0f32; c * r];
        for i in 0..r {
            for j in 0..c {
                t[j * r + i] = w[i * c + j];
            }
        }
        let (dtype, data) = quantize_2d(q, &t, c, r);
        tensors.push(TensorSpec {
            name: format!("model.layers.{li}.mlp.down_proj.t.weight"),
            dtype,
            shape: vec![c, r],
            data,
        });
        added += 1;
    }
    let mut header = model.header.clone();
    let mut prov = header
        .provenance
        .take()
        .unwrap_or_else(|| serde_json::json!({}));
    prov["down_t"] = serde_json::json!({
        "layers": added,
        "why": "contiguous per-neuron down rows for per-token sparsity",
    });
    header.provenance = Some(prov);
    CmfModel::write(
        output,
        &header,
        &tensors,
        (!model.masks.masks.is_empty()).then_some(&model.masks),
        model.vocab.as_deref(),
    )?;
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let out_sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    println!(
        "down-t: {added} layer(s) transposed (inter {inter} x hidden {hidden})\n\
         {model_path} {in_sz:.2} GB -> {output} {out_sz:.2} GB ({:+.1}%)",
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}
