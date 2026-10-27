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

use anyhow::{Context, bail, ensure};
use cortiq_core::format::CmfStreamWriter;
use cortiq_core::mask::{MaskCatalog, MaskPriority, TaskMask};
use cortiq_core::quant::{
    Q2TP_CHUNK, Q4TP_NIB, bf16_to_f32, dequant_q8_2f, dequant_q8_row, expected_nbytes,
    f16_to_f32,
    q2tp_ladder, q2tp_sections, q4tp_code, q4tp_ladder, q4tp_sections,
};
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

/// Validate every property of a dense MTP source that the physical compact
/// writer needs, without allocating a plan or touching an output.  The
/// calibration command calls this immediately after opening its verified
/// draft, so an unsupported MTP layout or codec cannot consume a large
/// `.tdata` scan and hours of activation collection before failing in the
/// writer.
pub(crate) fn preflight_mtp_compact_source(model: &CmfModel) -> anyhow::Result<Vec<usize>> {
    if !mtp_payload_present(model) {
        bail!("MTP compact source carries no model.mtp tensors");
    }
    let canonical = validate_canonical_dense_mtp(model, model.arch())?;
    for layer in 0..canonical.widths.len() {
        let [gate_name, up_name, down_name, down_t_name] = mtp_mlp_names(layer);
        for name in [&gate_name, &up_name, &down_name] {
            let dtype = tensor(model, name)?.dtype;
            ensure!(
                is_mtp_materializable_dtype(dtype),
                "MTP compact: {name} uses {}, which the physical compact writer cannot materialize",
                dtype.name()
            );
        }
        if canonical.has_down_t[layer] {
            let dtype = tensor(model, &down_t_name)?.dtype;
            ensure!(
                is_mtp_materializable_dtype(dtype),
                "MTP compact: {down_t_name} uses {}, which the physical compact writer cannot rebuild",
                dtype.name()
            );
        }
    }
    Ok(canonical.widths)
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
    // New plans additionally bind the merged tensor directory. `dir_hash()`
    // is necessarily the first physical shard's directory, whereas this
    // identity covers all shard entries and their payload hashes. Keep this
    // optional so historical unsharded plans remain readable.
    if let Some(recorded) = selection
        .get("source_draft_trunk_hash")
        .and_then(Value::as_str)
    {
        let parsed = u64::from_str_radix(recorded, 16).with_context(
            || "MTP compact plan.selection.source_draft_trunk_hash must be a hexadecimal u64",
        )?;
        let actual = model.trunk_hash();
        if parsed != actual {
            bail!(
                "MTP compact plan targets draft trunk hash {recorded}, but input is {:016x}",
                actual
            );
        }
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

/// Byte width of a raw, row-major scalar encoding whose axes can be selected
/// without changing any quantization parameters.
fn raw_scalar_width(dtype: TensorDtype) -> Option<usize> {
    match dtype {
        TensorDtype::F32 => Some(4),
        TensorDtype::F16 | TensorDtype::Bf16 => Some(2),
        _ => None,
    }
}

fn is_mtp_materializable_dtype(dtype: TensorDtype) -> bool {
    raw_scalar_width(dtype).is_some() || quant_of(dtype).is_some()
}

/// Round an f32 to bfloat16, using round-to-nearest-even. Values that
/// originated from a bfloat16 payload have zero low bits and therefore retain
/// their exact source representation on a transpose rebuild.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let round_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(round_bias).wrapping_shr(16) as u16
}

/// Retain selected complete rows of the simple, row-addressable dense codecs
/// without first expanding their entire source panel.  Unlike the tiled
/// formats below, these layouts have raw f32/bf16/f16 values or an independent
/// row scale, so selecting a neuron keeps exactly the same encoded values.
/// `q8_2f` additionally has one unchanged column-scale plane because the
/// input axis of gate/up is unchanged.
fn compact_encoded_basic_rows(
    dtype: TensorDtype,
    source: &[u8],
    source_rows: usize,
    cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<Vec<u8>>> {
    if !matches!(
        dtype,
        TensorDtype::F32
            | TensorDtype::F16
            | TensorDtype::Bf16
            | TensorDtype::Q8Row
            | TensorDtype::Q8_2f
    ) {
        return Ok(None);
    }
    let source_shape = [source_rows, cols];
    let expected_source = expected_nbytes(dtype, &source_shape)
        .context("MTP compact: encoded basic row-selection source shape overflows")?;
    ensure!(
        source.len() == expected_source,
        "MTP compact: encoded basic row-selection source payload has {} bytes, expected {expected_source}",
        source.len()
    );
    for &row in selected {
        ensure!(
            row < source_rows,
            "MTP compact: encoded basic row-selection index {row} is out of range for {source_rows} rows"
        );
    }
    let destination_shape = [selected.len(), cols];
    let expected_destination = expected_nbytes(dtype, &destination_shape)
        .context("MTP compact: encoded basic row-selection destination shape overflows")?;
    let source_values = source_rows
        .checked_mul(cols)
        .context("MTP compact: encoded basic row-selection source element count overflows")?;
    let destination_values = selected
        .len()
        .checked_mul(cols)
        .context("MTP compact: encoded basic row-selection destination element count overflows")?;
    let mut destination = vec![0u8; expected_destination];
    match dtype {
        TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16 => {
            let scalar_width = raw_scalar_width(dtype)
                .expect("raw scalar dtype was accepted by basic row selection");
            for (destination_row, &source_row) in selected.iter().enumerate() {
                let source_start = source_row
                    .checked_mul(cols)
                    .and_then(|offset| offset.checked_mul(scalar_width))
                    .context("MTP compact: encoded raw row source offset overflows")?;
                let destination_start = destination_row
                    .checked_mul(cols)
                    .and_then(|offset| offset.checked_mul(scalar_width))
                    .context("MTP compact: encoded raw row destination offset overflows")?;
                let row_bytes = cols
                    .checked_mul(scalar_width)
                    .context("MTP compact: encoded raw row byte width overflows")?;
                destination[destination_start..destination_start + row_bytes]
                    .copy_from_slice(&source[source_start..source_start + row_bytes]);
            }
        }
        TensorDtype::Q8Row => {
            let source_scales = &source[source_values..];
            let destination_scales = destination_values;
            for (destination_row, &source_row) in selected.iter().enumerate() {
                let source_start = source_row
                    .checked_mul(cols)
                    .context("MTP compact: encoded q8 row source offset overflows")?;
                let destination_start = destination_row
                    .checked_mul(cols)
                    .context("MTP compact: encoded q8 row destination offset overflows")?;
                destination[destination_start..destination_start + cols]
                    .copy_from_slice(&source[source_start..source_start + cols]);
                destination[destination_scales + destination_row * 2
                    ..destination_scales + destination_row * 2 + 2]
                    .copy_from_slice(&source_scales[source_row * 2..source_row * 2 + 2]);
            }
        }
        TensorDtype::Q8_2f => {
            let source_row_scales_end = source_values
                .checked_add(
                    source_rows
                        .checked_mul(2)
                        .context("MTP compact: encoded q8_2f source row-scale width overflows")?,
                )
                .context("MTP compact: encoded q8_2f source scale offset overflows")?;
            let (source_quant, source_tail) = source.split_at(source_values);
            let (source_row_scales, source_column_scales) =
                source_tail.split_at(source_row_scales_end - source_values);
            let destination_row_scales = destination_values;
            let destination_column_scales = destination_row_scales
                .checked_add(
                    selected
                        .len()
                        .checked_mul(2)
                        .context("MTP compact: encoded q8_2f destination row-scale width overflows")?,
                )
                .context("MTP compact: encoded q8_2f destination scale offset overflows")?;
            for (destination_row, &source_row) in selected.iter().enumerate() {
                let source_start = source_row
                    .checked_mul(cols)
                    .context("MTP compact: encoded q8_2f row source offset overflows")?;
                let destination_start = destination_row
                    .checked_mul(cols)
                    .context("MTP compact: encoded q8_2f row destination offset overflows")?;
                destination[destination_start..destination_start + cols]
                    .copy_from_slice(&source_quant[source_start..source_start + cols]);
                destination[destination_row_scales + destination_row * 2
                    ..destination_row_scales + destination_row * 2 + 2]
                    .copy_from_slice(&source_row_scales[source_row * 2..source_row * 2 + 2]);
            }
            destination[destination_column_scales..]
                .copy_from_slice(source_column_scales);
        }
        _ => unreachable!("non-basic dtype returned above"),
    }
    Ok(Some(destination))
}

/// Select the input axis of a dense MTP `down_proj` from simple encodings
/// exactly. The output needs no requantization: raw f32/bf16/f16 values remain
/// raw, q8 rows keep their per-row scale, and q8_2f keeps both the row scale and
/// the matching (possibly re-ordered) column scale.  This is especially
/// important for a quality-focused q8_2f draft: re-fitting scales after a
/// column subset would introduce avoidable error.
fn compact_encoded_basic_columns(
    dtype: TensorDtype,
    source: &[u8],
    rows: usize,
    source_cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<Vec<u8>>> {
    if !matches!(
        dtype,
        TensorDtype::F32
            | TensorDtype::F16
            | TensorDtype::Bf16
            | TensorDtype::Q8Row
            | TensorDtype::Q8_2f
    ) {
        return Ok(None);
    }
    let source_shape = [rows, source_cols];
    let expected_source = expected_nbytes(dtype, &source_shape)
        .context("MTP compact: encoded basic column-selection source shape overflows")?;
    ensure!(
        source.len() == expected_source,
        "MTP compact: encoded basic column-selection source payload has {} bytes, expected {expected_source}",
        source.len()
    );
    for &column in selected {
        ensure!(
            column < source_cols,
            "MTP compact: encoded basic column-selection index {column} is out of range for {source_cols} columns"
        );
    }
    let destination_cols = selected.len();
    let destination_shape = [rows, destination_cols];
    let expected_destination = expected_nbytes(dtype, &destination_shape)
        .context("MTP compact: encoded basic column-selection destination shape overflows")?;
    let source_values = rows
        .checked_mul(source_cols)
        .context("MTP compact: encoded basic column-selection source element count overflows")?;
    let destination_values = rows
        .checked_mul(destination_cols)
        .context("MTP compact: encoded basic column-selection destination element count overflows")?;
    let mut destination = vec![0u8; expected_destination];
    match dtype {
        TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16 => {
            let scalar_width = raw_scalar_width(dtype)
                .expect("raw scalar dtype was accepted by basic column selection");
            for row in 0..rows {
                for (destination_column, &source_column) in selected.iter().enumerate() {
                    let source_start = row
                        .checked_mul(source_cols)
                        .and_then(|offset| offset.checked_add(source_column))
                        .and_then(|offset| offset.checked_mul(scalar_width))
                        .context("MTP compact: encoded raw column source offset overflows")?;
                    let destination_start = row
                        .checked_mul(destination_cols)
                        .and_then(|offset| offset.checked_add(destination_column))
                        .and_then(|offset| offset.checked_mul(scalar_width))
                        .context("MTP compact: encoded raw column destination offset overflows")?;
                    destination[destination_start..destination_start + scalar_width]
                        .copy_from_slice(&source[source_start..source_start + scalar_width]);
                }
            }
        }
        TensorDtype::Q8Row => {
            let source_scales = &source[source_values..];
            let destination_scales = destination_values;
            for row in 0..rows {
                let source_row = row
                    .checked_mul(source_cols)
                    .context("MTP compact: encoded q8 column source row offset overflows")?;
                let destination_row = row
                    .checked_mul(destination_cols)
                    .context("MTP compact: encoded q8 column destination row offset overflows")?;
                for (destination_column, &source_column) in selected.iter().enumerate() {
                    destination[destination_row + destination_column] =
                        source[source_row + source_column];
                }
                destination[destination_scales + row * 2..destination_scales + row * 2 + 2]
                    .copy_from_slice(&source_scales[row * 2..row * 2 + 2]);
            }
        }
        TensorDtype::Q8_2f => {
            let source_row_scales_end = source_values
                .checked_add(
                    rows.checked_mul(2)
                        .context("MTP compact: encoded q8_2f source row-scale width overflows")?,
                )
                .context("MTP compact: encoded q8_2f source scale offset overflows")?;
            let (source_quant, source_tail) = source.split_at(source_values);
            let (source_row_scales, source_column_scales) =
                source_tail.split_at(source_row_scales_end - source_values);
            let destination_row_scales = destination_values;
            let destination_column_scales = destination_row_scales
                .checked_add(
                    rows.checked_mul(2)
                        .context("MTP compact: encoded q8_2f destination row-scale width overflows")?,
                )
                .context("MTP compact: encoded q8_2f destination scale offset overflows")?;
            for row in 0..rows {
                let source_row = row
                    .checked_mul(source_cols)
                    .context("MTP compact: encoded q8_2f column source row offset overflows")?;
                let destination_row = row
                    .checked_mul(destination_cols)
                    .context("MTP compact: encoded q8_2f column destination row offset overflows")?;
                for (destination_column, &source_column) in selected.iter().enumerate() {
                    destination[destination_row + destination_column] =
                        source_quant[source_row + source_column];
                }
                destination[destination_row_scales + row * 2
                    ..destination_row_scales + row * 2 + 2]
                    .copy_from_slice(&source_row_scales[row * 2..row * 2 + 2]);
            }
            for (destination_column, &source_column) in selected.iter().enumerate() {
                destination[destination_column_scales + destination_column * 2
                    ..destination_column_scales + destination_column * 2 + 2]
                    .copy_from_slice(
                        &source_column_scales[source_column * 2..source_column * 2 + 2],
                    );
            }
        }
        _ => unreachable!("non-basic dtype returned above"),
    }
    Ok(Some(destination))
}

/// Decode the compact basic payload only when rebuilding an optional
/// `down_proj.t` cache.  The source panel is never expanded: this consumes
/// the already-small selected matrix and exactly matches the core codecs.
fn dequant_compact_basic_matrix(
    dtype: TensorDtype,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> anyhow::Result<Vec<f32>> {
    let expected = expected_nbytes(dtype, &[rows, cols])
        .context("MTP compact: compact basic matrix shape overflows")?;
    ensure!(
        bytes.len() == expected,
        "MTP compact: compact basic matrix has {} bytes, expected {expected}",
        bytes.len()
    );
    let values = rows
        .checked_mul(cols)
        .context("MTP compact: compact basic matrix element count overflows")?;
    let mut output = vec![0.0f32; values];
    match dtype {
        TensorDtype::F32 => {
            for (slot, packed) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
            }
        }
        TensorDtype::F16 => {
            for (slot, packed) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
            }
        }
        TensorDtype::Bf16 => {
            for (slot, packed) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = bf16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
            }
        }
        TensorDtype::Q8Row => dequant_q8_row(bytes, rows, cols, &mut output),
        TensorDtype::Q8_2f => dequant_q8_2f(bytes, rows, cols, &mut output),
        _ => bail!("MTP compact: {dtype:?} is not a basic exact matrix codec"),
    }
    Ok(output)
}

/// Encode a rebuilt compact matrix using the source tensor's physical dtype.
/// Raw f32/bf16 source tensors have no `Quant` equivalent, but preserving the
/// dtype matters for a `down_proj.t` cache: silently falling back to f16 can
/// overflow bf16 values or add a needless quality change.
fn encode_compact_mtp_matrix(
    dtype: TensorDtype,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> anyhow::Result<(TensorDtype, Vec<u8>)> {
    let expected_values = rows
        .checked_mul(cols)
        .context("MTP compact: rebuilt matrix element count overflows")?;
    ensure!(
        values.len() == expected_values,
        "MTP compact: rebuilt matrix has {} values, expected {expected_values}",
        values.len()
    );
    match dtype {
        TensorDtype::F32 => {
            let byte_len = expected_values
                .checked_mul(4)
                .context("MTP compact: rebuilt f32 matrix byte count overflows")?;
            let mut data = Vec::new();
            data.try_reserve_exact(byte_len).map_err(|error| {
                anyhow::anyhow!("MTP compact: reserving rebuilt f32 matrix: {error}")
            })?;
            for &value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            Ok((dtype, data))
        }
        TensorDtype::Bf16 => {
            let byte_len = expected_values
                .checked_mul(2)
                .context("MTP compact: rebuilt bf16 matrix byte count overflows")?;
            let mut data = Vec::new();
            data.try_reserve_exact(byte_len).map_err(|error| {
                anyhow::anyhow!("MTP compact: reserving rebuilt bf16 matrix: {error}")
            })?;
            for &value in values {
                data.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
            }
            Ok((dtype, data))
        }
        _ => {
            let quant = quant_of(dtype)
                .with_context(|| format!("unsupported MTP matrix dtype {}", dtype.name()))?;
            Ok(quantize_2d(quant, values, rows, cols))
        }
    }
}

enum CompactMtpColumns {
    /// Exact selected bytes for an encoding whose axes remain independently
    /// addressable after compaction.
    Exact {
        dtype: TensorDtype,
        data: Vec<u8>,
    },
    /// q4tp/q2tp need a new group layout after their input columns change,
    /// so they are decoded only for the selected columns and re-encoded.
    Decoded(Vec<f32>),
}

/// Losslessly retain selected *rows* of the row-addressable q4tp/q2tp
/// payloads. Gate/up MTP matrices keep their input axis unchanged, so a
/// selected neuron is a complete encoded row: copying its weight, ladder and
/// code planes is both exact relative to the source CMF and much cheaper than
/// dequantizing/re-quantizing it. `down_proj` is deliberately excluded: its
/// selected axis is columns, which changes groups and requires requantization.
fn compact_encoded_tiled_rows(
    dtype: TensorDtype,
    source: &[u8],
    source_rows: usize,
    cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<Vec<u8>>> {
    let tile_bytes = match dtype {
        TensorDtype::Q4TiledP => Q4TP_NIB,
        TensorDtype::Q2TiledP => Q2TP_CHUNK,
        _ => return Ok(None),
    };
    ensure!(
        cols != 0 && cols % 32 == 0,
        "MTP compact: {:?} row selection needs a positive 32-aligned input width, got {cols}",
        dtype
    );
    let source_shape = [source_rows, cols];
    let expected_source = expected_nbytes(dtype, &source_shape)
        .context("MTP compact: encoded row-selection source shape overflows")?;
    ensure!(
        source.len() == expected_source,
        "MTP compact: encoded row-selection source payload has {} bytes, expected {expected_source}",
        source.len()
    );
    let destination_shape = [selected.len(), cols];
    let expected_destination = expected_nbytes(dtype, &destination_shape)
        .context("MTP compact: encoded row-selection destination shape overflows")?;
    let groups_per_row = cols / 32;
    let row_weight_bytes = groups_per_row
        .checked_mul(tile_bytes)
        .context("MTP compact: encoded row-selection row width overflows")?;
    let (source_params, source_codes, source_code_stride) = match dtype {
        TensorDtype::Q4TiledP => q4tp_sections(source_rows, cols),
        TensorDtype::Q2TiledP => q2tp_sections(source_rows, cols),
        _ => unreachable!("non-tiled dtype returned above"),
    };
    let (destination_params, destination_codes, destination_code_stride) = match dtype {
        TensorDtype::Q4TiledP => q4tp_sections(selected.len(), cols),
        TensorDtype::Q2TiledP => q2tp_sections(selected.len(), cols),
        _ => unreachable!("non-tiled dtype returned above"),
    };
    ensure!(
        source_code_stride == destination_code_stride,
        "MTP compact: encoded row-selection changed code stride unexpectedly"
    );
    ensure!(
        source_params == source_rows.checked_mul(row_weight_bytes).context("MTP compact: encoded source plane offset overflows")?
            && destination_params == selected.len().checked_mul(row_weight_bytes).context("MTP compact: encoded destination plane offset overflows")?,
        "MTP compact: encoded row-selection plane layout is inconsistent"
    );

    let mut destination = vec![0u8; expected_destination];
    for (destination_row, &source_row) in selected.iter().enumerate() {
        ensure!(
            source_row < source_rows,
            "MTP compact: encoded row-selection index {source_row} is out of range for {source_rows} rows"
        );
        let source_weight = source_row * row_weight_bytes;
        let destination_weight = destination_row * row_weight_bytes;
        destination[destination_weight..destination_weight + row_weight_bytes]
            .copy_from_slice(&source[source_weight..source_weight + row_weight_bytes]);
        let source_param = source_params + source_row * 4;
        let destination_param = destination_params + destination_row * 4;
        destination[destination_param..destination_param + 4]
            .copy_from_slice(&source[source_param..source_param + 4]);
        let source_code = source_codes + source_row * source_code_stride;
        let destination_code = destination_codes + destination_row * destination_code_stride;
        destination[destination_code..destination_code + destination_code_stride]
            .copy_from_slice(&source[source_code..source_code + source_code_stride]);
    }
    Ok(Some(destination))
}

fn compact_encoded_mtp_rows(
    model: &CmfModel,
    name: &str,
    source_rows: usize,
    cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<(TensorDtype, Vec<u8>)>> {
    let entry = tensor(model, name)?;
    ensure!(
        entry.shape.as_slice() == [source_rows, cols],
        "MTP compact: {name} has shape {:?}, expected [{source_rows}, {cols}]",
        entry.shape
    );
    let source = model.tensor_bytes(name)?;
    let data = if let Some(data) =
        compact_encoded_basic_rows(entry.dtype, source, source_rows, cols, selected)?
    {
        Some(data)
    } else {
        compact_encoded_tiled_rows(entry.dtype, source, source_rows, cols, selected)?
    };
    Ok(data.map(|data| (entry.dtype, data)))
}

/// Decode only the selected columns of a q4tp/q2tp matrix.  Unlike gate/up,
/// `down_proj` cannot keep its encoded source rows because compaction changes
/// its input axis and therefore its group boundaries.  It still need not
/// expand the entire source matrix first: both predicted-scale codecs are
/// independently addressable by source row and 32-wide group.  The returned
/// f32 values use precisely the same scalar operations as `dequant_q4tp` and
/// `dequant_q2tp`, then the ordinary writer re-encodes the compact matrix.
fn compact_encoded_tiled_columns(
    dtype: TensorDtype,
    source: &[u8],
    rows: usize,
    source_cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<Vec<f32>>> {
    let tile_bytes = match dtype {
        TensorDtype::Q4TiledP => Q4TP_NIB,
        TensorDtype::Q2TiledP => Q2TP_CHUNK,
        _ => return Ok(None),
    };
    ensure!(
        source_cols != 0 && source_cols % 32 == 0,
        "MTP compact: {:?} column selection needs a positive 32-aligned input width, got {source_cols}",
        dtype
    );
    let source_shape = [rows, source_cols];
    let expected_source = expected_nbytes(dtype, &source_shape)
        .context("MTP compact: encoded column-selection source shape overflows")?;
    ensure!(
        source.len() == expected_source,
        "MTP compact: encoded column-selection source payload has {} bytes, expected {expected_source}",
        source.len()
    );
    // Channel ranking produces a non-monotonic order.  Group the requested
    // columns once so each source 32-wide tile obtains its predicted scale
    // code only once per row, while `destination_column` preserves that
    // ranking order in the compact output.
    let groups_per_row = source_cols / 32;
    let mut selected_by_group = vec![Vec::<(usize, usize)>::new(); groups_per_row];
    for (destination_column, &column) in selected.iter().enumerate() {
        ensure!(
            column < source_cols,
            "MTP compact: encoded column-selection index {column} is out of range for {source_cols} columns"
        );
        selected_by_group[column / 32].push((destination_column, column % 32));
    }
    let (params_offset, codes_offset, code_stride) = match dtype {
        TensorDtype::Q4TiledP => q4tp_sections(rows, source_cols),
        TensorDtype::Q2TiledP => q2tp_sections(rows, source_cols),
        _ => unreachable!("non-tiled dtype returned above"),
    };
    let row_weight_bytes = groups_per_row
        .checked_mul(tile_bytes)
        .context("MTP compact: encoded column-selection row width overflows")?;
    ensure!(
        params_offset
            == rows
                .checked_mul(row_weight_bytes)
                .context("MTP compact: encoded column-selection plane offset overflows")?,
        "MTP compact: encoded column-selection plane layout is inconsistent"
    );
    let capacity = rows
        .checked_mul(selected.len())
        .context("MTP compact: encoded column-selection output shape overflows")?;
    let params_end = params_offset
        .checked_add(rows.checked_mul(4).context("MTP compact: encoded column-selection parameter plane overflows")?)
        .context("MTP compact: encoded column-selection parameter plane end overflows")?;
    ensure!(
        params_end <= source.len()
            && codes_offset
                .checked_add(rows.checked_mul(code_stride).context("MTP compact: encoded column-selection code plane overflows")?)
                .is_some_and(|end| end == source.len()),
        "MTP compact: encoded column-selection plane bounds are inconsistent"
    );
    let params = &source[params_offset..params_end];
    let mut output = vec![0.0f32; capacity];
    for row in 0..rows {
        let codes = &source[codes_offset + row * code_stride..codes_offset + (row + 1) * code_stride];
        let destination = &mut output[row * selected.len()..(row + 1) * selected.len()];
        match dtype {
            TensorDtype::Q4TiledP => {
                let ladder = q4tp_ladder(params, row);
                for (group, requested) in selected_by_group.iter().enumerate() {
                    if requested.is_empty() {
                        continue;
                    }
                    let scale = ladder[q4tp_code(codes, group)];
                    let tile_offset = row * row_weight_bytes + group * Q4TP_NIB;
                    for &(destination_column, within_group) in requested {
                        let byte = source[tile_offset + within_group / 2];
                        let quant = if within_group % 2 == 0 {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        };
                        destination[destination_column] = (quant as f32 - 8.0) * scale;
                    }
                }
            }
            TensorDtype::Q2TiledP => {
                let ladder = q2tp_ladder(params, row);
                for (group, requested) in selected_by_group.iter().enumerate() {
                    if requested.is_empty() {
                        continue;
                    }
                    let scale = ladder[q4tp_code(codes, group)];
                    for &(destination_column, within_group) in requested {
                        let byte = source[row * row_weight_bytes
                            + group * Q2TP_CHUNK
                            + within_group / 4];
                        let quant = (byte >> (2 * (within_group % 4))) & 0x03;
                        destination[destination_column] = (quant as f32 - 1.5) * scale;
                    }
                }
            }
            _ => unreachable!("non-tiled dtype returned above"),
        }
    }
    debug_assert_eq!(output.len(), capacity);
    Ok(Some(output))
}

fn compact_encoded_mtp_columns(
    model: &CmfModel,
    name: &str,
    rows: usize,
    source_cols: usize,
    selected: &[usize],
) -> anyhow::Result<Option<CompactMtpColumns>> {
    let entry = tensor(model, name)?;
    ensure!(
        entry.shape.as_slice() == [rows, source_cols],
        "MTP compact: {name} has shape {:?}, expected [{rows}, {source_cols}]",
        entry.shape
    );
    let source = model.tensor_bytes(name)?;
    if let Some(data) =
        compact_encoded_basic_columns(entry.dtype, source, rows, source_cols, selected)?
    {
        return Ok(Some(CompactMtpColumns::Exact {
            dtype: entry.dtype,
            data,
        }));
    }
    Ok(compact_encoded_tiled_columns(entry.dtype, source, rows, source_cols, selected)?
        .map(CompactMtpColumns::Decoded))
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

// Copy large mmap payloads in bounded chunks. The only allocated dense f32
// buffers in an MTP-only rewrite are one compacted draft layer at a time.
const STREAM_COPY_CHUNK_BYTES: usize = 16 * 1024 * 1024;

fn stream_writer_for(
    model: &CmfModel,
    output: &Path,
    additional_tensors: usize,
    no_clobber: bool,
) -> anyhow::Result<CmfStreamWriter> {
    let average_name = model
        .tensors
        .iter()
        .map(|entry| entry.name.len())
        .sum::<usize>()
        / model.tensors.len().max(1);
    let tensor_count = model
        .tensors
        .len()
        .checked_add(additional_tensors)
        .context("MTP compact: output tensor count overflow")?;
    let head = CmfStreamWriter::head_reserve_for(tensor_count, average_name);
    let writer = if no_clobber {
        CmfStreamWriter::new_no_clobber(output, head)
    } else {
        CmfStreamWriter::new(output, head)
    };
    Ok(writer?)
}

/// Copy a merged source model to one valid, unsharded CMF without making a
/// `Vec<u8>` for every payload. MiMo pair publishing uses this for a sharded
/// backbone; MTP-only baking below uses the same streaming contract while
/// replacing the draft FFN tensors.
fn stream_copy_as_unsharded_impl(
    model: &CmfModel,
    output: &Path,
    no_clobber: bool,
) -> anyhow::Result<()> {
    let mut writer = stream_writer_for(model, output, 0, no_clobber)?;
    for entry in &model.tensors {
        if no_clobber {
            // This private-stage route is reached only from `mtp-compact`,
            // which verifies both source and final candidate. Reuse the
            // source entry hash to avoid an otherwise redundant mmap pass.
            writer.push_bounded_prehashed(
                &entry.name,
                entry.dtype,
                &entry.shape,
                model.entry_bytes(entry),
                entry.hash,
                STREAM_COPY_CHUNK_BYTES,
            )?;
        } else {
            // The public helper retains its historic standalone semantics:
            // callers are not required to have verified the source or to
            // perform a final candidate scan.
            writer.push_bounded(
                &entry.name,
                entry.dtype,
                &entry.shape,
                model.entry_bytes(entry),
                STREAM_COPY_CHUNK_BYTES,
            )?;
        }
    }
    let mut header = model.header.clone();
    let source_shards_coalesced = header.shard.take().is_some();
    if source_shards_coalesced {
        let mut provenance = header
            .provenance
            .take()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        provenance["stream_copy"] = json!({"source_shards_coalesced": true});
        header.provenance = Some(provenance);
    }
    writer.finish(
        &header,
        (!model.masks.masks.is_empty()).then_some(&model.masks),
        model.vocab.as_deref(),
    )?;
    Ok(())
}

pub(crate) fn stream_copy_as_unsharded(model: &CmfModel, output: &Path) -> anyhow::Result<()> {
    stream_copy_as_unsharded_impl(model, output, false)
}

/// Same streaming coalescer, but safe for a private stage: an unexpected
/// existing file is an error rather than a truncation target.
pub(crate) fn stream_copy_as_unsharded_new(model: &CmfModel, output: &Path) -> anyhow::Result<()> {
    stream_copy_as_unsharded_impl(model, output, true)
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
    let gate_q = quant_of(tensor(model, &gate_name)?.dtype);
    let up_q = quant_of(tensor(model, &up_name)?.dtype);
    let source_down_q = quant_of(tensor(model, &down_name)?.dtype);
    let width = plan.order.len();
    let gate_encoded = compact_encoded_mtp_rows(
        model,
        &gate_name,
        source_width,
        hidden,
        &plan.order,
    )?;
    let up_encoded = compact_encoded_mtp_rows(
        model,
        &up_name,
        source_width,
        hidden,
        &plan.order,
    )?;
    // Tiled low-bit codecs group the input axis, so a hand-written plan
    // with an unaligned compact width must fall back. Plain raw/f16/q8 rows do
    // not have that restriction and must not lose precision merely because
    // their selected draft width is not a codec-group multiple.
    let down_q = source_down_q.map(|quant| {
        if matches!(quant, Quant::F16 | Quant::Q8Row | Quant::Q8_2f) || width % 32 == 0 {
            quant
        } else {
            Quant::Q8_2f
        }
    });
    for (name, encoded, quant) in [
        (&gate_name, gate_encoded, gate_q),
        (&up_name, up_encoded, up_q),
    ] {
        if let Some((dtype, data)) = encoded {
            tensors.push(TensorSpec {
                name: name.to_string(),
                dtype,
                shape: vec![width, hidden],
                data,
            });
        } else {
            let quant = quant.with_context(|| format!("unsupported MTP dtype for {name}"))?;
            let original = dequant(model, name)?;
            let compacted = compact_rows(&original, &plan.order, source_width, hidden);
            let (dtype, data) = quantize_2d(quant, &compacted, width, hidden);
            tensors.push(TensorSpec {
                name: name.to_string(),
                dtype,
                shape: vec![width, hidden],
                data,
            });
        }
    }
    let down_for_transpose = match compact_encoded_mtp_columns(
        model,
        &down_name,
        hidden,
        source_width,
        &plan.order,
    )? {
        Some(CompactMtpColumns::Exact { dtype, data }) => {
            // Raw/f16/q8/q8_2f retain exactly the selected source values and
            // scales, so do not introduce a second quantization merely
            // because the MTP intermediate axis became narrower.
            let compacted = has_down_t
                .then(|| dequant_compact_basic_matrix(dtype, &data, hidden, width))
                .transpose()?;
            tensors.push(TensorSpec {
                name: down_name.to_string(),
                dtype,
                shape: vec![hidden, width],
                data,
            });
            compacted
        }
        Some(CompactMtpColumns::Decoded(down)) => {
            let down_q = down_q.context("unsupported MTP down dtype")?;
            let (dtype, data) = quantize_2d(down_q, &down, hidden, width);
            tensors.push(TensorSpec {
                name: down_name.to_string(),
                dtype,
                shape: vec![hidden, width],
                data,
            });
            has_down_t.then_some(down)
        }
        None => {
            let down_q = down_q.context("unsupported MTP down dtype")?;
            let original = dequant(model, &down_name)?;
            let down = compact_down_columns(&original, hidden, source_width, &plan.order);
            drop(original);
            let (dtype, data) = quantize_2d(down_q, &down, hidden, width);
            tensors.push(TensorSpec {
                name: down_name.to_string(),
                dtype,
                shape: vec![hidden, width],
                data,
            });
            has_down_t.then_some(down)
        }
    };
    if has_down_t {
        // Rebuild instead of slicing the old cached transpose.  This keeps it
        // mathematically coherent with a permutation as well as a subset.
        let down_t_dtype = tensor(model, &down_t_name)?.dtype;
        let down = down_for_transpose.expect("down projection retained for transpose");
        let down_t = transpose(&down, hidden, width);
        let (dtype, data) = encode_compact_mtp_matrix(down_t_dtype, &down_t, width, hidden)?;
        tensors.push(TensorSpec {
            name: down_t_name,
            dtype,
            shape: vec![width, hidden],
            data,
        });
    }
    Ok(())
}

/// Rewrite one compacted MTP layer directly into the stream writer.  This is
/// intentionally projection-at-a-time: keeping the original and compacted
/// gate, up and down matrices together would make a large draft rewrite need
/// several extra GiB of transient f32 RAM.  `down_proj.t`, when present, is
/// reconstructed only after the compacted `down_proj` was emitted.
fn stream_compacted_mtp_layer(
    model: &CmfModel,
    layer: usize,
    plan: &LayerPlan,
    source_width: usize,
    has_down_t: bool,
    hidden: usize,
    writer: &mut CmfStreamWriter,
) -> anyhow::Result<()> {
    let [gate_name, up_name, down_name, down_t_name] = mtp_mlp_names(layer);
    let width = plan.order.len();

    for name in [&gate_name, &up_name] {
        if let Some((dtype, data)) =
            compact_encoded_mtp_rows(model, name, source_width, hidden, &plan.order)?
        {
            // q4tp/q2tp gate/up rows are copied byte-for-byte. This avoids
            // a second PTQ pass and keeps the selected source rows exactly
            // intact, including their per-row predicted-scale ladder.
            writer.push_bounded(
                name,
                dtype,
                &[width, hidden],
                &data,
                STREAM_COPY_CHUNK_BYTES,
            )?;
            continue;
        }
        let quant = quant_of(tensor(model, name)?.dtype)
            .with_context(|| format!("unsupported MTP dtype for {name}"))?;
        let original = dequant(model, name)?;
        let compacted = compact_rows(&original, &plan.order, source_width, hidden);
        drop(original);
        let (dtype, data) = quantize_2d(quant, &compacted, width, hidden);
        // `data` no longer borrows the f32 panel after quantization.  Do not
        // keep both representations resident while a large streamed write is
        // in progress: a multi-layer MTP draft can otherwise retain another
        // wide projection per layer longer than necessary.
        drop(compacted);
        writer.push_bounded(
            name,
            dtype,
            &[width, hidden],
            &data,
            STREAM_COPY_CHUNK_BYTES,
        )?;
    }

    let source_down_quant = quant_of(tensor(model, &down_name)?.dtype);
    // Tiled low-bit codecs group the input axis.  A hand-written unaligned
    // plan must fall back rather than serialize an invalid encoded tensor.
    let down_quant = source_down_quant.map(|quant| {
        if matches!(quant, Quant::F16 | Quant::Q8Row | Quant::Q8_2f) || width % 32 == 0 {
            quant
        } else {
            Quant::Q8_2f
        }
    });
    let compacted_down_for_transpose = match compact_encoded_mtp_columns(
        model,
        &down_name,
        hidden,
        source_width,
        &plan.order,
    )? {
        Some(CompactMtpColumns::Exact { dtype, data }) => {
            // Preserve the selected raw/f16/q8/q8_2f values and scales exactly.
            // `down_proj.t`, when present, is derived only from this compact
            // payload below; no full source down projection is materialized.
            let compacted = has_down_t
                .then(|| dequant_compact_basic_matrix(dtype, &data, hidden, width))
                .transpose()?;
            writer.push_bounded(
                &down_name,
                dtype,
                &[hidden, width],
                &data,
                STREAM_COPY_CHUNK_BYTES,
            )?;
            compacted
        }
        Some(CompactMtpColumns::Decoded(compacted)) => {
            let down_quant = down_quant.context("unsupported MTP down dtype")?;
            let (dtype, data) = quantize_2d(down_quant, &compacted, hidden, width);
            writer.push_bounded(
                &down_name,
                dtype,
                &[hidden, width],
                &data,
                STREAM_COPY_CHUNK_BYTES,
            )?;
            has_down_t.then_some(compacted)
        }
        None => {
            let down_quant = down_quant.context("unsupported MTP down dtype")?;
            let original = dequant(model, &down_name)?;
            let compacted = compact_down_columns(&original, hidden, source_width, &plan.order);
            drop(original);
            let (dtype, data) = quantize_2d(down_quant, &compacted, hidden, width);
            writer.push_bounded(
                &down_name,
                dtype,
                &[hidden, width],
                &data,
                STREAM_COPY_CHUNK_BYTES,
            )?;
            has_down_t.then_some(compacted)
        }
    };

    if has_down_t {
        let down_t_dtype = tensor(model, &down_t_name)?.dtype;
        let compacted_down = compacted_down_for_transpose
            .expect("down projection retained for transpose");
        let compacted_down_t = transpose(&compacted_down, hidden, width);
        // The transpose owns every value required by the cached layout. Free
        // the original compacted orientation before re-encoding it so this
        // optional cache does not make peak f32 RAM grow to two panels plus
        // an encoded payload.
        drop(compacted_down);
        let (dtype, data) =
            encode_compact_mtp_matrix(down_t_dtype, &compacted_down_t, width, hidden)?;
        drop(compacted_down_t);
        writer.push_bounded(
            &down_t_name,
            dtype,
            &[width, hidden],
            &data,
            STREAM_COPY_CHUNK_BYTES,
        )?;
    }
    Ok(())
}

fn cmd_mtp_only_bake(
    model: &Arc<CmfModel>,
    plan_document: &Value,
    mtp: &MtpPlan,
    output: &str,
    no_clobber: bool,
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
    // MTP-only baking must work for a main model far larger than RAM. Stream
    // all preserved mmap payloads directly to the candidate; only one small
    // MTP layer is dequantized/requantized at a time below. This also turns a
    // valid sharded input into one independently runnable unsharded CMF.
    let replacement_tensor_capacity = layer_count
        .checked_mul(4)
        .context("MTP compact: replacement tensor count overflow")?;
    let mut writer = stream_writer_for(
        model,
        Path::new(output),
        replacement_tensor_capacity,
        no_clobber,
    )?;
    for entry in &model.tensors {
        let omit = match mtp.mode {
            MtpMode::Preserve => false,
            MtpMode::Drop => entry.name.starts_with("model.mtp."),
            MtpMode::Compact => is_mtp_compacted_tensor(&entry.name, layer_count),
        };
        if !omit {
            if no_clobber {
                // The private `mtp-compact` route verified this source and
                // will verify the final candidate. Preserve the known hash to
                // avoid an extra full read of every untouched payload.
                writer.push_bounded_prehashed(
                    &entry.name,
                    entry.dtype,
                    &entry.shape,
                    model.entry_bytes(entry),
                    entry.hash,
                    STREAM_COPY_CHUNK_BYTES,
                )?;
            } else {
                // A public `tube bake` is intentionally usable standalone;
                // it must compute fresh hashes rather than rely on an
                // unverified input directory.
                writer.push_bounded(
                    &entry.name,
                    entry.dtype,
                    &entry.shape,
                    model.entry_bytes(entry),
                    STREAM_COPY_CHUNK_BYTES,
                )?;
            }
        }
    }
    if let Some(canonical) = &canonical {
        for (layer, plan) in mtp.layers.iter().enumerate() {
            stream_compacted_mtp_layer(
                model,
                layer,
                plan,
                canonical.widths[layer],
                canonical.has_down_t[layer],
                arch.hidden_size,
                &mut writer,
            )?;
        }
    }

    let mut header = model.header.clone();
    let source_shards_coalesced = header.shard.take().is_some();
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
    let mut provenance = header
        .provenance
        .take()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
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
                "gate_up_q4tp_q2tp": "selected encoded rows copied bytewise; other projections are re-encoded as needed",
                "down_q4tp_q2tp": "only selected source columns are decoded before compact output re-encoding",
                "raw_f32_bf16_f16_q8_q8_2f": "gate/up selected rows and down selected columns retain source encoded values/scales exactly; compact down is decoded only when rebuilding down_proj.t",
                "down_proj_t_rebuilt": source.has_down_t,
                "non_ffn_tensors": "copied-bytewise",
            })
        }
    };
    if let Some(selection) = plan_document.get("selection") {
        tubes["mtp"]["selection"] = selection.clone();
    }
    if source_shards_coalesced {
        tubes["mtp"]["source_shards_coalesced"] = json!(true);
    }
    provenance["tubes"] = tubes;
    header.provenance = Some(provenance);
    writer.finish(
        &header,
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

fn cmd_tube_bake_impl(
    model_path: &str,
    plan_path: &str,
    output: &str,
    no_clobber: bool,
) -> anyhow::Result<()> {
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
        return cmd_mtp_only_bake(&model, &plan, &mtp, output, no_clobber);
    }
    ensure!(
        !no_clobber,
        "MTP compact: private no-clobber bake only supports trunk.mode=preserve"
    );
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
                "raw_f32_bf16_f16_q8_q8_2f": "gate/up selected rows and down selected columns retain source encoded values/scales exactly; compact down is decoded only when rebuilding down_proj.t",
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

/// Rewrite a model according to a tube/MTP plan. This legacy public command
/// keeps its established overwrite semantics; production `mtp-compact` uses
/// the private no-clobber variant below for nonce stages.
pub fn cmd_tube_bake(model_path: &str, plan_path: &str, output: &str) -> anyhow::Result<()> {
    cmd_tube_bake_impl(model_path, plan_path, output, false)
}

/// Safe private-stage entry point used by `mtp-compact`. It only accepts the
/// MTP-only path, whose writer opens the destination with `create_new`.
pub(crate) fn cmd_tube_bake_new(
    model_path: &str,
    plan_path: &str,
    output: &str,
) -> anyhow::Result<()> {
    cmd_tube_bake_impl(model_path, plan_path, output, true)
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
    fn q4tp_and_q2tp_gate_up_row_selection_is_lossless() {
        let (rows, cols) = (7usize, 64usize);
        let values = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                ((row * 31 + col * 17 + 5) % 71) as f32 / 29.0 - 1.2
            })
            .collect::<Vec<_>>();
        // Non-monotonic order proves that every plane is copied with the same
        // row permutation rather than merely truncating its prefix.
        let selected = [6usize, 1, 4, 0];
        for quant in [Quant::Q4TiledP, Quant::Q2TiledP] {
            let (dtype, source) = quantize_2d(quant, &values, rows, cols);
            let compact = compact_encoded_tiled_rows(dtype, &source, rows, cols, &selected)
                .unwrap()
                .expect("tiled codec must use encoded row copy");
            assert!(
                cortiq_core::quant::validate_payload(dtype, &[selected.len(), cols], &compact)
                    .is_ok(),
                "selected {dtype:?} payload is invalid"
            );
            let mut decoded_source = vec![0.0f32; rows * cols];
            let mut decoded_compact = vec![0.0f32; selected.len() * cols];
            match dtype {
                TensorDtype::Q4TiledP => {
                    cortiq_core::quant::dequant_q4tp(&source, rows, cols, &mut decoded_source);
                    cortiq_core::quant::dequant_q4tp(
                        &compact,
                        selected.len(),
                        cols,
                        &mut decoded_compact,
                    );
                }
                TensorDtype::Q2TiledP => {
                    cortiq_core::quant::dequant_q2tp(&source, rows, cols, &mut decoded_source);
                    cortiq_core::quant::dequant_q2tp(
                        &compact,
                        selected.len(),
                        cols,
                        &mut decoded_compact,
                    );
                }
                _ => panic!("q4tp/q2tp encoder returned {dtype:?}"),
            }
            for (destination_row, &source_row) in selected.iter().enumerate() {
                assert_eq!(
                    &decoded_compact[destination_row * cols..(destination_row + 1) * cols],
                    &decoded_source[source_row * cols..(source_row + 1) * cols],
                    "{dtype:?} selected row {source_row} changed"
                );
            }
        }
    }

    #[test]
    fn q4tp_and_q2tp_down_column_selection_matches_full_dequant() {
        let (rows, cols) = (5usize, 64usize);
        let values = (0..rows * cols)
            .map(|index| ((index * 29 % 113) as f32 - 56.0) / 37.0)
            .collect::<Vec<_>>();
        // A non-monotonic order is the actual compaction case: channel score
        // rank, rather than source position, determines output columns.
        let selected = [63usize, 2, 32, 31, 1, 47];
        for quant in [Quant::Q4TiledP, Quant::Q2TiledP] {
            let (dtype, source) = quantize_2d(quant, &values, rows, cols);
            let direct = compact_encoded_tiled_columns(dtype, &source, rows, cols, &selected)
                .unwrap()
                .expect("tiled codec must decode selected columns directly");
            let mut full = vec![0.0f32; rows * cols];
            match dtype {
                TensorDtype::Q4TiledP => {
                    cortiq_core::quant::dequant_q4tp(&source, rows, cols, &mut full)
                }
                TensorDtype::Q2TiledP => {
                    cortiq_core::quant::dequant_q2tp(&source, rows, cols, &mut full)
                }
                _ => panic!("q4tp/q2tp encoder returned {dtype:?}"),
            }
            assert_eq!(
                direct,
                compact_down_columns(&full, rows, cols, &selected),
                "{dtype:?} direct selected-column decode changed a down_proj value"
            );
        }
    }

    #[test]
    fn f16_and_q8_mtp_selection_preserves_encoded_values_exactly() {
        // Unlike tiled low-bit payloads, these codecs retain independent
        // scalar/scale fields after an arbitrary input-column reorder.  The
        // direct path must therefore preserve the original decoded values,
        // not merely produce a close re-quantization.
        let (rows, cols) = (5usize, 37usize);
        let values = (0..rows * cols)
            .map(|index| ((index * 43 % 173) as f32 - 86.0) / 53.0)
            .collect::<Vec<_>>();
        let selected_rows = [4usize, 1, 3];
        let selected_columns = [36usize, 2, 0, 19, 7];
        for quant in [Quant::F16, Quant::Q8Row, Quant::Q8_2f] {
            let (dtype, source) = quantize_2d(quant, &values, rows, cols);
            let selected_row_bytes =
                compact_encoded_basic_rows(dtype, &source, rows, cols, &selected_rows)
                    .unwrap()
                    .expect("basic codec must use direct row selection");
            let selected_column_bytes =
                compact_encoded_basic_columns(dtype, &source, rows, cols, &selected_columns)
                    .unwrap()
                    .expect("basic codec must use direct column selection");
            assert!(
                cortiq_core::quant::validate_payload(
                    dtype,
                    &[selected_rows.len(), cols],
                    &selected_row_bytes,
                )
                .is_ok(),
                "{dtype:?} selected rows are not a valid payload"
            );
            assert!(
                cortiq_core::quant::validate_payload(
                    dtype,
                    &[rows, selected_columns.len()],
                    &selected_column_bytes,
                )
                .is_ok(),
                "{dtype:?} selected columns are not a valid payload"
            );
            let decoded_source = dequant_compact_basic_matrix(dtype, &source, rows, cols).unwrap();
            let decoded_rows = dequant_compact_basic_matrix(
                dtype,
                &selected_row_bytes,
                selected_rows.len(),
                cols,
            )
            .unwrap();
            let decoded_columns = dequant_compact_basic_matrix(
                dtype,
                &selected_column_bytes,
                rows,
                selected_columns.len(),
            )
            .unwrap();
            assert_eq!(
                decoded_rows,
                compact_rows(&decoded_source, &selected_rows, rows, cols),
                "{dtype:?} direct selected rows changed decoded values"
            );
            assert_eq!(
                decoded_columns,
                compact_down_columns(&decoded_source, rows, cols, &selected_columns),
                "{dtype:?} direct selected columns changed decoded values"
            );
        }
    }

    #[test]
    fn mtp_only_bake_uses_lossless_tiled_gate_up_and_selected_down_decode() {
        // Exercise the actual MTP-only stream writer, not only the helper:
        // q4tp/q2tp gate/up must retain selected source rows byte-for-byte,
        // and down must be equivalent to selecting those source columns
        // before its one necessary output re-quantization.
        let dir = tempfile::tempdir().unwrap();
        let (hidden, source_width) = (32usize, 64usize);
        let order = vec![
            63usize, 2, 32, 31, 1, 47, 8, 59, 17, 40, 4, 55, 23, 36, 9, 60, 14, 42, 6, 52,
            27, 34, 12, 57, 19, 45, 3, 62, 24, 38, 11, 50,
        ];
        let values = |salt: usize, rows: usize, cols: usize| {
            (0..rows * cols)
                .map(|index| ((index * 31 + salt * 19) % 127) as f32 / 41.0 - 1.5)
                .collect::<Vec<_>>()
        };
        for (suffix, quant) in [("q4tp", Quant::Q4TiledP), ("q2tp", Quant::Q2TiledP)] {
            let source_path = dir.path().join(format!("source-{suffix}.cmf"));
            let plan_path = dir.path().join(format!("plan-{suffix}.json"));
            let output_path = dir.path().join(format!("output-{suffix}.cmf"));
            let gate = values(1, source_width, hidden);
            let up = values(2, source_width, hidden);
            let down = values(3, hidden, source_width);
            let (gate_dtype, gate_data) = quantize_2d(quant, &gate, source_width, hidden);
            let (up_dtype, up_data) = quantize_2d(quant, &up, source_width, hidden);
            let (down_dtype, down_data) = quantize_2d(quant, &down, hidden, source_width);
            let mut header = sidecar_header();
            header.arch.hidden_size = hidden;
            header.arch.intermediate_size = hidden;
            header.arch.num_attention_heads = 1;
            header.arch.num_kv_heads = 1;
            header.arch.head_dim = hidden;
            header.arch.v_head_dim = Some(hidden);
            header
                .arch
                .mtp
                .as_mut()
                .expect("sidecar header has MTP")
                .ffn_intermediate_sizes = Some(vec![source_width]);
            let prefix = "model.mtp.layers.0.mlp.";
            CmfModel::write(
                &source_path,
                &header,
                &[
                    TensorSpec {
                        name: format!("{prefix}gate_proj.weight"),
                        dtype: gate_dtype,
                        shape: vec![source_width, hidden],
                        data: gate_data,
                    },
                    TensorSpec {
                        name: format!("{prefix}up_proj.weight"),
                        dtype: up_dtype,
                        shape: vec![source_width, hidden],
                        data: up_data,
                    },
                    TensorSpec {
                        name: format!("{prefix}down_proj.weight"),
                        dtype: down_dtype,
                        shape: vec![hidden, source_width],
                        data: down_data,
                    },
                ],
                None,
                None,
            )
            .unwrap();
            let source = CmfModel::open(&source_path).unwrap();
            assert!(
                preflight_mtp_compact_source(&source).is_ok(),
                "{suffix} must be accepted before expensive calibration"
            );
            std::fs::write(
                &plan_path,
                serde_json::to_vec(&json!({
                    "format": "cmf-mtp-ffn-plan",
                    "version": 2,
                    "trunk": {"mode": "preserve"},
                    "mtp": {"mode": "compact", "layers": [{"order": order, "widths": [32]}]},
                    "selection": {
                        "source_draft_directory_hash": format!("{:016x}", source.dir_hash())
                    }
                }))
                .unwrap(),
            )
            .unwrap();

            let expected_gate = compact_encoded_tiled_rows(
                gate_dtype,
                source
                    .tensor_bytes(&format!("{prefix}gate_proj.weight"))
                    .unwrap(),
                source_width,
                hidden,
                &order,
            )
            .unwrap()
            .unwrap();
            let expected_up = compact_encoded_tiled_rows(
                up_dtype,
                source
                    .tensor_bytes(&format!("{prefix}up_proj.weight"))
                    .unwrap(),
                source_width,
                hidden,
                &order,
            )
            .unwrap()
            .unwrap();
            let expected_down = compact_encoded_tiled_columns(
                down_dtype,
                source
                    .tensor_bytes(&format!("{prefix}down_proj.weight"))
                    .unwrap(),
                hidden,
                source_width,
                &order,
            )
            .unwrap()
            .unwrap();
            let (expected_down_dtype, expected_down_data) =
                quantize_2d(quant, &expected_down, hidden, order.len());

            cmd_tube_bake(
                source_path.to_str().unwrap(),
                plan_path.to_str().unwrap(),
                output_path.to_str().unwrap(),
            )
            .unwrap();
            let output = CmfModel::open(&output_path).unwrap();
            assert!(output.verify().is_empty(), "{suffix}: {:?}", output.verify());
            assert_eq!(
                output
                    .tensor_bytes(&format!("{prefix}gate_proj.weight"))
                    .unwrap(),
                expected_gate,
                "{suffix} gate lost encoded source rows"
            );
            assert_eq!(
                output
                    .tensor_bytes(&format!("{prefix}up_proj.weight"))
                    .unwrap(),
                expected_up,
                "{suffix} up lost encoded source rows"
            );
            assert_eq!(
                output
                    .tensor(&format!("{prefix}down_proj.weight"))
                    .unwrap()
                    .dtype,
                expected_down_dtype
            );
            assert_eq!(
                output
                    .tensor_bytes(&format!("{prefix}down_proj.weight"))
                    .unwrap(),
                expected_down_data,
                "{suffix} down did not encode direct selected-column values"
            );
        }
    }

    #[test]
    fn f32_and_bf16_mtp_selection_preserves_encoded_values_exactly() {
        // Raw scalar codecs have no scale plane at all. Their selected
        // payload must therefore be literal byte-for-byte gathers, while the
        // bfloat decode check also catches an accidental f16 fallback.
        let (rows, cols) = (5usize, 37usize);
        let values = (0..rows * cols)
            .map(|index| ((index * 43 % 173) as f32 - 86.0) / 53.0)
            .collect::<Vec<_>>();
        let selected_rows = [4usize, 1, 3];
        let selected_columns = [36usize, 2, 0, 19, 7];
        for dtype in [TensorDtype::F32, TensorDtype::Bf16] {
            let (_, source) = encode_compact_mtp_matrix(dtype, &values, rows, cols).unwrap();
            let selected_row_bytes =
                compact_encoded_basic_rows(dtype, &source, rows, cols, &selected_rows)
                    .unwrap()
                    .expect("raw codec must use direct row selection");
            let selected_column_bytes =
                compact_encoded_basic_columns(dtype, &source, rows, cols, &selected_columns)
                    .unwrap()
                    .expect("raw codec must use direct column selection");
            assert!(
                cortiq_core::quant::validate_payload(
                    dtype,
                    &[selected_rows.len(), cols],
                    &selected_row_bytes,
                )
                .is_ok(),
                "{dtype:?} selected rows are not a valid payload"
            );
            assert!(
                cortiq_core::quant::validate_payload(
                    dtype,
                    &[rows, selected_columns.len()],
                    &selected_column_bytes,
                )
                .is_ok(),
                "{dtype:?} selected columns are not a valid payload"
            );
            let decoded_source = dequant_compact_basic_matrix(dtype, &source, rows, cols).unwrap();
            let decoded_rows = dequant_compact_basic_matrix(
                dtype,
                &selected_row_bytes,
                selected_rows.len(),
                cols,
            )
            .unwrap();
            let decoded_columns = dequant_compact_basic_matrix(
                dtype,
                &selected_column_bytes,
                rows,
                selected_columns.len(),
            )
            .unwrap();
            assert_eq!(
                decoded_rows,
                compact_rows(&decoded_source, &selected_rows, rows, cols),
                "{dtype:?} selected rows changed values"
            );
            assert_eq!(
                decoded_columns,
                compact_down_columns(&decoded_source, rows, cols, &selected_columns),
                "{dtype:?} selected columns changed values"
            );
        }
    }

    #[test]
    fn rebuilt_bf16_transpose_uses_nearest_even_rounding() {
        // Exact halfway cases cover the only nontrivial part of rebuilding a
        // bf16 `down_proj.t`: the lower retained bit decides which neighbor
        // wins. This also prevents an accidental f16 encoder fallback.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        let (dtype, bytes) = encode_compact_mtp_matrix(
            TensorDtype::Bf16,
            &[f32::from_bits(0x3f80_8000), f32::from_bits(0x3f81_8000)],
            1,
            2,
        )
        .unwrap();
        assert_eq!(dtype, TensorDtype::Bf16);
        assert_eq!(bytes, vec![0x80, 0x3f, 0x82, 0x3f]);
    }

    #[test]
    fn mtp_only_bake_keeps_raw_f16_and_q8_selected_values_exact() {
        // Exercise the streaming writer's exact-code branch, including the
        // optional transpose cache. The output down matrix must retain the
        // selected source raw/f16/q8 values exactly; only down_proj.t is
        // rebuilt from that compact matrix.
        let dir = tempfile::tempdir().unwrap();
        let (hidden, source_width) = (32usize, 64usize);
        let order = vec![
            63usize, 2, 32, 31, 1, 47, 8, 59, 17, 40, 4, 55, 23, 36, 9, 60, 14, 42, 6, 52,
            27, 34, 12, 57, 19, 45, 3, 62, 24, 38, 11, 50,
        ];
        let values = |salt: usize, rows: usize, cols: usize| {
            (0..rows * cols)
                .map(|index| ((index * 31 + salt * 19) % 127) as f32 / 41.0 - 1.5)
                .collect::<Vec<_>>()
        };
        for (suffix, dtype) in [
            ("f32", TensorDtype::F32),
            ("bf16", TensorDtype::Bf16),
            ("f16", TensorDtype::F16),
            ("q8", TensorDtype::Q8Row),
            ("q8_2f", TensorDtype::Q8_2f),
        ] {
            let source_path = dir.path().join(format!("basic-source-{suffix}.cmf"));
            let plan_path = dir.path().join(format!("basic-plan-{suffix}.json"));
            let output_path = dir.path().join(format!("basic-output-{suffix}.cmf"));
            let gate = values(1, source_width, hidden);
            let up = values(2, source_width, hidden);
            let down = values(3, hidden, source_width);
            let down_t = transpose(&down, hidden, source_width);
            let (gate_dtype, gate_data) =
                encode_compact_mtp_matrix(dtype, &gate, source_width, hidden).unwrap();
            let (up_dtype, up_data) =
                encode_compact_mtp_matrix(dtype, &up, source_width, hidden).unwrap();
            let (down_dtype, down_data) =
                encode_compact_mtp_matrix(dtype, &down, hidden, source_width).unwrap();
            let (down_t_dtype, down_t_data) =
                encode_compact_mtp_matrix(dtype, &down_t, source_width, hidden).unwrap();
            let mut header = sidecar_header();
            header.arch.hidden_size = hidden;
            header.arch.intermediate_size = hidden;
            header.arch.num_attention_heads = 1;
            header.arch.num_kv_heads = 1;
            header.arch.head_dim = hidden;
            header.arch.v_head_dim = Some(hidden);
            header
                .arch
                .mtp
                .as_mut()
                .expect("sidecar header has MTP")
                .ffn_intermediate_sizes = Some(vec![source_width]);
            let prefix = "model.mtp.layers.0.mlp.";
            CmfModel::write(
                &source_path,
                &header,
                &[
                    TensorSpec {
                        name: format!("{prefix}gate_proj.weight"),
                        dtype: gate_dtype,
                        shape: vec![source_width, hidden],
                        data: gate_data,
                    },
                    TensorSpec {
                        name: format!("{prefix}up_proj.weight"),
                        dtype: up_dtype,
                        shape: vec![source_width, hidden],
                        data: up_data,
                    },
                    TensorSpec {
                        name: format!("{prefix}down_proj.weight"),
                        dtype: down_dtype,
                        shape: vec![hidden, source_width],
                        data: down_data,
                    },
                    TensorSpec {
                        name: format!("{prefix}down_proj.t.weight"),
                        dtype: down_t_dtype,
                        shape: vec![source_width, hidden],
                        data: down_t_data,
                    },
                ],
                None,
                None,
            )
            .unwrap();
            let source = CmfModel::open(&source_path).unwrap();
            std::fs::write(
                &plan_path,
                serde_json::to_vec(&json!({
                    "format": "cmf-mtp-ffn-plan",
                    "version": 2,
                    "trunk": {"mode": "preserve"},
                    "mtp": {"mode": "compact", "layers": [{"order": order, "widths": [32]}]},
                    "selection": {
                        "source_draft_directory_hash": format!("{:016x}", source.dir_hash())
                    }
                }))
                .unwrap(),
            )
            .unwrap();

            let expected_gate = compact_encoded_basic_rows(
                gate_dtype,
                source
                    .tensor_bytes(&format!("{prefix}gate_proj.weight"))
                    .unwrap(),
                source_width,
                hidden,
                &order,
            )
            .unwrap()
            .unwrap();
            let expected_up = compact_encoded_basic_rows(
                up_dtype,
                source
                    .tensor_bytes(&format!("{prefix}up_proj.weight"))
                    .unwrap(),
                source_width,
                hidden,
                &order,
            )
            .unwrap()
            .unwrap();
            let expected_down = compact_encoded_basic_columns(
                down_dtype,
                source
                    .tensor_bytes(&format!("{prefix}down_proj.weight"))
                    .unwrap(),
                hidden,
                source_width,
                &order,
            )
            .unwrap()
            .unwrap();
            let expected_down_values =
                dequant_compact_basic_matrix(down_dtype, &expected_down, hidden, order.len())
                    .unwrap();
            let (expected_down_t_dtype, expected_down_t) = encode_compact_mtp_matrix(
                down_t_dtype,
                &transpose(&expected_down_values, hidden, order.len()),
                order.len(),
                hidden,
            )
            .unwrap();

            // The ordinary tube-bake route uses the older collecting writer
            // rather than the private streaming route. Exercise its shared
            // exact raw path too, so F32/BF16 cannot regress behind the
            // MTP-only fast path.
            let mut collected = Vec::new();
            append_compacted_mtp_layer(
                &source,
                0,
                &LayerPlan {
                    order: order.clone(),
                    widths: vec![order.len()],
                },
                source_width,
                true,
                hidden,
                &mut collected,
            )
            .unwrap();
            for (name, expected_dtype, expected) in [
                (format!("{prefix}gate_proj.weight"), gate_dtype, &expected_gate),
                (format!("{prefix}up_proj.weight"), up_dtype, &expected_up),
                (format!("{prefix}down_proj.weight"), down_dtype, &expected_down),
                (
                    format!("{prefix}down_proj.t.weight"),
                    expected_down_t_dtype,
                    &expected_down_t,
                ),
            ] {
                let actual = collected
                    .iter()
                    .find(|tensor| tensor.name == name)
                    .expect("collecting MTP bake emitted every projection");
                assert_eq!(actual.dtype, expected_dtype, "{suffix} {name} dtype");
                assert_eq!(&actual.data, expected, "{suffix} {name} collecting bake");
            }

            cmd_tube_bake(
                source_path.to_str().unwrap(),
                plan_path.to_str().unwrap(),
                output_path.to_str().unwrap(),
            )
            .unwrap();
            let output = CmfModel::open(&output_path).unwrap();
            assert!(output.verify().is_empty(), "{suffix}: {:?}", output.verify());
            for (name, expected) in [
                (format!("{prefix}gate_proj.weight"), expected_gate),
                (format!("{prefix}up_proj.weight"), expected_up),
                (format!("{prefix}down_proj.weight"), expected_down),
            ] {
                assert_eq!(
                    output.tensor_bytes(&name).unwrap(),
                    expected,
                    "{suffix} {name} did not retain exact selected encoded values"
                );
            }
            let down_t_entry = output
                .tensor(&format!("{prefix}down_proj.t.weight"))
                .unwrap();
            assert_eq!(down_t_entry.dtype, expected_down_t_dtype, "{suffix} down_t dtype");
            assert_eq!(
                output
                    .tensor_bytes(&format!("{prefix}down_proj.t.weight"))
                    .unwrap(),
                expected_down_t,
                "{suffix} down_t was not rebuilt from exact compact down"
            );
        }
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
        let bad_trunk = validate_mtp_compact_source(
            &serde_json::json!({
                "selection": {
                    "source_draft_directory_hash": format!("{:016x}", source_model.dir_hash()),
                    "source_draft_trunk_hash": "0000000000000000"
                }
            }),
            &source_model,
        )
        .unwrap_err()
        .to_string();
        assert!(bad_trunk.contains("trunk hash"), "{bad_trunk}");
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
        assert!(speculative.mtp_accepted <= speculative.mtp_drafted);
        assert_eq!(plain.token_ids, speculative.token_ids);
    }

    #[test]
    fn sharded_mtp_bake_coalesces_a_runnable_unsharded_candidate() {
        use cortiq_core::format::ShardInfo;

        let dir = tempfile::tempdir().unwrap();
        let shard1 = dir.path().join("source-00001-of-00002.cmf");
        let shard2 = dir.path().join("source-00002-of-00002.cmf");
        let plan = dir.path().join("plan.json");
        let copied = dir.path().join("copied.cmf");
        let output = dir.path().join("compact.cmf");
        let tensors = canonical_mtp_tensors();
        let split = tensors.len() / 2;
        let mut header1 = canonical_mtp_header();
        header1.shard = Some(ShardInfo { no: 1, count: 2 });
        let mut header2 = canonical_mtp_header();
        header2.shard = Some(ShardInfo { no: 2, count: 2 });
        CmfModel::write(&shard1, &header1, &tensors[..split], None, None).unwrap();
        CmfModel::write(&shard2, &header2, &tensors[split..], None, None).unwrap();

        let source = CmfModel::open_sharded(&shard1).unwrap();
        assert!(source.verify().is_empty(), "{:?}", source.verify());
        // This is the path used by a sharded MiMo backbone before its
        // separately compacted sidecar is admitted and published.
        stream_copy_as_unsharded(&source, &copied).unwrap();
        let copied_model = CmfModel::open(&copied).unwrap();
        assert!(copied_model.header.shard.is_none());
        assert!(copied_model.verify().is_empty(), "{:?}", copied_model.verify());
        assert_eq!(
            copied_model
                .tensor_bytes("model.layers.0.mlp.gate_proj.weight")
                .unwrap(),
            source
                .tensor_bytes("model.layers.0.mlp.gate_proj.weight")
                .unwrap()
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
                    "source_draft_directory_hash": format!("{:016x}", source.dir_hash()),
                    "source_draft_trunk_hash": format!("{:016x}", source.trunk_hash())
                }
            }))
            .unwrap(),
        )
        .unwrap();

        cmd_tube_bake(
            shard1.to_str().unwrap(),
            plan.to_str().unwrap(),
            output.to_str().unwrap(),
        )
        .unwrap();

        let candidate = Arc::new(CmfModel::open(&output).unwrap());
        assert!(candidate.header.shard.is_none());
        assert!(candidate.verify().is_empty(), "{:?}", candidate.verify());
        assert_eq!(
            candidate
                .arch()
                .mtp
                .as_ref()
                .and_then(|config| config.ffn_intermediate_sizes.as_deref()),
            Some(&[2][..])
        );
        let mut pipeline = Pipeline::from_model(&candidate, SamplerConfig::default()).unwrap();
        assert_eq!(pipeline.probe_mtp_ffn(&[1, 2, 3, 4]).unwrap()[0].mass.len(), 2);
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
