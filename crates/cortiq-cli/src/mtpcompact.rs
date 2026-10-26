//! Calibration-driven physical compaction of a dense MTP draft FFN.
//!
//! This is deliberately a small, auditable operation rather than a hidden
//! branch of trunk FFN compaction: an MTP draft has its own activation
//! distribution and may have a width independent of the backbone.  The
//! command therefore obtains statistics through `Pipeline::probe_mtp_ffn`,
//! writes an explicit v2 tube plan with `trunk.mode = preserve`, then hands
//! the physical gather to the common CMF writer.

use anyhow::{Context, Result, ensure};
use cortiq_core::quant::dequant_tensor;
use cortiq_core::{CmfModel, mtp_sidecar_path};
use cortiq_engine::{MtpFfnProbe, Pipeline, SamplerConfig};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::tdata::{TdataReader, TdataSplitSelector};
use crate::tube::paths_collide;

/// The physical codecs used by CMF dense FFNs have 32-wide groups.  Keeping
/// the draft plan on that boundary prevents a supposedly compact q4tp/q8_2f
/// MTP matrix from falling back to a less suitable layout in the writer.
const MTP_COMPACT_ALIGNMENT: usize = 32;

/// Keep the backbone and independently stored draft provenance separate.
/// MiMo puts its draft in a sidecar, so recording only the main directory
/// hash would let a stale or swapped sidecar look like the calibrated source.
fn source_hash_fields(main_hash: u64, draft_hash: u64) -> Map<String, Value> {
    let mut fields = Map::with_capacity(2);
    fields.insert(
        "source_directory_hash".into(),
        json!(format!("{main_hash:016x}")),
    );
    fields.insert(
        "source_draft_directory_hash".into(),
        json!(format!("{draft_hash:016x}")),
    );
    fields
}

fn tensor_f32(model: &CmfModel, name: &str) -> Result<Vec<f32>> {
    let entry = model
        .tensor(name)
        .with_context(|| format!("MTP compact: missing tensor {name}"))?;
    let mut values = vec![0.0f32; entry.n_elems()];
    dequant_tensor(entry, model.tensor_bytes(name)?, &mut values)
        .map_err(|error| anyhow::anyhow!("MTP compact: dequantizing {name}: {error}"))?;
    Ok(values)
}

fn tensor_shape(model: &CmfModel, name: &str) -> Result<Vec<usize>> {
    Ok(model
        .tensor(name)
        .with_context(|| format!("MTP compact: missing tensor {name}"))?
        .shape
        .clone())
}

fn aligned_keep(width: usize, keep_ratio: f64) -> Result<usize> {
    ensure!(
        keep_ratio.is_finite() && keep_ratio > 0.0 && keep_ratio <= 1.0,
        "MTP compact: --keep-ratio must be finite and in (0, 1]"
    );
    ensure!(
        width >= MTP_COMPACT_ALIGNMENT && width % MTP_COMPACT_ALIGNMENT == 0,
        "MTP compact: draft FFN width {width} is not a positive {MTP_COMPACT_ALIGNMENT}-channel codec group"
    );
    let keep = ((width as f64 * keep_ratio).floor() as usize / MTP_COMPACT_ALIGNMENT)
        * MTP_COMPACT_ALIGNMENT;
    ensure!(
        keep >= MTP_COMPACT_ALIGNMENT,
        "MTP compact: --keep-ratio {keep_ratio} leaves fewer than one {MTP_COMPACT_ALIGNMENT}-channel group from width {width}"
    );
    Ok(keep)
}

/// Rank independently measured draft channels.  Kept separate from CMF I/O
/// so the score and deterministic tie break have a small, direct test.
fn rank_channel_scores(
    squared_mass: &[f64],
    down: &[f32],
    hidden: usize,
    width: usize,
) -> Result<Vec<(usize, f64)>> {
    ensure!(
        squared_mass.len() == width && down.len() == hidden.saturating_mul(width),
        "MTP compact: score inputs do not match a {hidden} × {width} down projection"
    );
    let mut ranked: Vec<(usize, f64)> = (0..width)
        .map(|channel| {
            let col_norm_sq = (0..hidden)
                .map(|row| {
                    let value = down[row * width + channel] as f64;
                    value * value
                })
                .sum::<f64>();
            let activation = squared_mass[channel];
            ensure!(
                activation.is_finite() && activation >= 0.0 && col_norm_sq.is_finite(),
                "MTP compact: draft channel {channel} has invalid score inputs"
            );
            let score = activation * col_norm_sq;
            ensure!(
                score.is_finite(),
                "MTP compact: draft channel {channel} score overflow"
            );
            Ok((channel, score))
        })
        .collect::<Result<_>>()?;
    ranked.sort_by(|(ia, a), (ib, b)| {
        b.partial_cmp(a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });
    Ok(ranked)
}

fn validate_probe(probe: &[MtpFfnProbe]) -> Result<()> {
    ensure!(
        !probe.is_empty(),
        "MTP compact: draft probe returned no layers"
    );
    for (layer, row) in probe.iter().enumerate() {
        ensure!(
            row.mass.len() == row.squared_mass.len()
                && row
                    .mass
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && row
                    .squared_mass
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
            "MTP compact: draft layer {layer} returned invalid activation statistics"
        );
    }
    Ok(())
}

/// Add calibration records without allowing a finite individual record to
/// overflow into an invalid plan statistic.
fn accumulate_probe(total: &mut [MtpFfnProbe], probe: &[MtpFfnProbe]) -> Result<()> {
    ensure!(
        total.len() == probe.len(),
        "MTP compact: draft layer count changed from {} to {} during calibration",
        total.len(),
        probe.len()
    );
    for (layer, (sum, row)) in total.iter_mut().zip(probe.iter()).enumerate() {
        ensure!(
            sum.mass.len() == row.mass.len() && sum.squared_mass.len() == row.squared_mass.len(),
            "MTP compact: draft layer {layer} width changed during calibration"
        );
        for (channel, (a, b)) in sum.mass.iter_mut().zip(&row.mass).enumerate() {
            let next = *a + *b;
            ensure!(
                next.is_finite() && next >= 0.0,
                "MTP compact: draft layer {layer} mass overflow at channel {channel}"
            );
            *a = next;
        }
        for (channel, (a, b)) in sum
            .squared_mass
            .iter_mut()
            .zip(&row.squared_mass)
            .enumerate()
        {
            let next = *a + *b;
            ensure!(
                next.is_finite() && next >= 0.0,
                "MTP compact: draft layer {layer} squared mass overflow at channel {channel}"
            );
            *a = next;
        }
        sum.observed_positions = sum
            .observed_positions
            .checked_add(row.observed_positions)
            .ok_or_else(|| anyhow::anyhow!("MTP compact: observed-position counter overflow"))?;
    }
    Ok(())
}

fn rank_draft_channels(
    model: &CmfModel,
    layer: usize,
    squared_mass: &[f64],
    keep_ratio: f64,
) -> Result<(usize, Vec<usize>, Value)> {
    let prefix = format!("model.mtp.layers.{layer}.mlp.");
    let gate_name = format!("{prefix}gate_proj.weight");
    let up_name = format!("{prefix}up_proj.weight");
    let down_name = format!("{prefix}down_proj.weight");
    let gate_shape = tensor_shape(model, &gate_name)?;
    let up_shape = tensor_shape(model, &up_name)?;
    let down_shape = tensor_shape(model, &down_name)?;
    ensure!(
        gate_shape.len() == 2 && up_shape.len() == 2 && down_shape.len() == 2,
        "MTP compact: draft layer {layer} FFN tensors must be matrices"
    );
    let (width, hidden) = (gate_shape[0], gate_shape[1]);
    ensure!(
        up_shape == gate_shape && down_shape == vec![hidden, width],
        "MTP compact: draft layer {layer} has mismatched FFN shapes gate={gate_shape:?}, up={up_shape:?}, down={down_shape:?}"
    );
    ensure!(
        squared_mass.len() == width,
        "MTP compact: draft layer {layer} probe width {} != tensor width {width}",
        squared_mass.len()
    );
    ensure!(
        hidden == model.arch().hidden_size,
        "MTP compact: draft layer {layer} hidden width {hidden} != header hidden_size {}",
        model.arch().hidden_size
    );
    let keep = aligned_keep(width, keep_ratio)?;
    let down = tensor_f32(model, &down_name)?;
    let ranked = rank_channel_scores(squared_mass, &down, hidden, width)
        .with_context(|| format!("MTP compact: draft layer {layer}"))?;
    let total: f64 = ranked.iter().map(|(_, score)| *score).sum();
    ensure!(
        total.is_finite() && total >= 0.0,
        "MTP compact: invalid score total"
    );
    let selected_score: f64 = ranked.iter().take(keep).map(|(_, score)| *score).sum();
    let order = ranked
        .iter()
        .take(keep)
        .map(|(channel, _)| *channel)
        .collect();
    Ok((
        width,
        order,
        json!({
            "layer": layer,
            "source_width": width,
            "kept_width": keep,
            "score_total": total,
            "score_kept": selected_score,
            "score_kept_fraction": if total > 0.0 { Some(selected_score / total) } else { None },
        }),
    ))
}

fn draft_source(main: &Arc<CmfModel>, main_path: &str) -> Result<(PathBuf, Arc<CmfModel>)> {
    if main
        .tensors
        .iter()
        .any(|tensor| tensor.name.starts_with("model.mtp."))
    {
        return Ok((PathBuf::from(main_path), main.clone()));
    }
    let sidecar = mtp_sidecar_path(Path::new(main_path));
    ensure!(
        sidecar.exists(),
        "MTP compact: {} carries no model.mtp.* tensors and its expected sidecar {} is absent",
        main_path,
        sidecar.display()
    );
    let side = Arc::new(
        CmfModel::open_sharded(&sidecar)
            .with_context(|| format!("MTP compact: opening sidecar {}", sidecar.display()))?,
    );
    ensure!(
        side.tensors
            .iter()
            .any(|tensor| tensor.name.starts_with("model.mtp.")),
        "MTP compact: sidecar {} carries no model.mtp.* payload",
        sidecar.display()
    );
    Ok((sidecar, side))
}

/// Build an explicit independent-draft plan from train-only `.tdata`
/// activations and immediately bake it.  The source model is never edited;
/// when its draft lives in `<model>.mtp.cmf`, `output` is the new sidecar and
/// the backbone is left byte-for-byte untouched.
pub fn cmd_mtp_compact(
    model_path: &str,
    data_selector: &str,
    output: &str,
    keep_ratio: f64,
    plan_out: Option<&str>,
    max_records: usize,
) -> Result<()> {
    ensure!(
        !paths_collide(Path::new(output), Path::new(model_path)),
        "MTP compact: --output must differ from the source model"
    );
    let main = Arc::new(
        CmfModel::open_sharded(model_path)
            .with_context(|| format!("MTP compact: opening source {model_path:?}"))?,
    );
    let integrity = main.verify();
    ensure!(
        integrity.is_empty(),
        "MTP compact: source integrity failed: {}",
        integrity
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ")
    );
    let (draft_path, draft) = draft_source(&main, model_path)?;
    ensure!(
        !paths_collide(Path::new(output), &draft_path),
        "MTP compact: --output must differ from the source draft {}",
        draft_path.display()
    );
    if !paths_collide(Path::new(model_path), &draft_path) {
        let integrity = draft.verify();
        ensure!(
            integrity.is_empty(),
            "MTP compact: draft sidecar integrity failed: {}",
            integrity
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    let (reader, selector) = TdataReader::open_selector(data_selector)
        .with_context(|| format!("MTP compact: opening data selector {data_selector:?}"))?;
    ensure!(
        selector.split == TdataSplitSelector::Train,
        "MTP compact: data selector must be the train split; use PATH.tdata or PATH.tdata:train so validation/test remain untouched"
    );
    reader
        .verify()
        .with_context(|| format!("MTP compact: verifying {data_selector:?}"))?;

    let mut pipeline = Pipeline::from_model(&main, SamplerConfig::default())
        .map_err(|error| anyhow::anyhow!("MTP compact: loading pipeline: {error}"))?;
    if !paths_collide(Path::new(model_path), &draft_path) {
        ensure!(
            pipeline.mimo_mtp.is_some(),
            "MTP compact: an external .mtp.cmf sidecar is supported only for the MiMo draft executor; the main model did not load it (check CMF_MIMO_MTP)"
        );
    }
    let mut aggregate: Option<Vec<MtpFfnProbe>> = None;
    let mut records = 0usize;
    let mut tokens = 0usize;
    for item in reader.stream(selector)? {
        if max_records != 0 && records >= max_records {
            break;
        }
        let sample = item?;
        if sample.tokens.len() < 2 {
            continue;
        }
        let probe = pipeline.probe_mtp_ffn(&sample.tokens).map_err(|error| {
            anyhow::anyhow!(
                "MTP compact: probing record {:?}: {error}",
                sample.record.rid
            )
        })?;
        validate_probe(&probe).with_context(|| {
            format!(
                "MTP compact: invalid draft activation statistics for record {:?}",
                sample.record.rid
            )
        })?;
        if let Some(total) = aggregate.as_mut() {
            accumulate_probe(total, &probe).with_context(|| {
                format!(
                    "MTP compact: aggregating draft activation statistics for record {:?}",
                    sample.record.rid
                )
            })?;
        } else {
            aggregate = Some(probe);
        }
        records += 1;
        tokens = tokens
            .checked_add(sample.tokens.len())
            .ok_or_else(|| anyhow::anyhow!("MTP compact: token counter overflow"))?;
    }
    let aggregate = aggregate.ok_or_else(|| {
        anyhow::anyhow!(
            "MTP compact: no usable calibration record with at least two tokens was selected"
        )
    })?;

    let mut mtp_layers = Vec::with_capacity(aggregate.len());
    let mut report_layers = Vec::with_capacity(aggregate.len());
    for (layer, probe) in aggregate.iter().enumerate() {
        ensure!(
            probe.observed_positions > 0,
            "MTP compact: draft layer {layer} saw no teacher-forced positions"
        );
        let (source_width, order, mut report) =
            rank_draft_channels(&draft, layer, &probe.squared_mass, keep_ratio)?;
        let obj = report
            .as_object_mut()
            .expect("locally constructed JSON object");
        obj.insert("observed_positions".into(), json!(probe.observed_positions));
        obj.insert(
            "activation_mass".into(),
            json!(probe.mass.iter().sum::<f64>()),
        );
        obj.insert(
            "activation_squared_mass".into(),
            json!(probe.squared_mass.iter().sum::<f64>()),
        );
        let kept_width = order.len();
        mtp_layers.push(json!({"order": order, "widths": [kept_width]}));
        // Keep the source width in this independent report: it makes a plan
        // edited by hand auditable before tube bake sees it.
        report_layers.push(json!({"source_width": source_width, "report": report}));
    }

    let mut selection = json!({
        "kind": "draft-activation-weighted-down-column-energy-v1",
        "formula": "sum_teacher_forced((activation(gate, up))^2) * down_proj_column_l2_squared",
        "activation_contract": draft.arch().hidden_act,
        "keep_ratio": keep_ratio,
        "alignment": MTP_COMPACT_ALIGNMENT,
        "calibration_records": records,
        "calibration_tokens": tokens,
        "data_selector": data_selector,
        "source_main": model_path,
        "source_draft": draft_path.display().to_string(),
        "layers": report_layers,
    });
    selection
        .as_object_mut()
        .expect("locally constructed JSON object")
        .extend(source_hash_fields(main.dir_hash(), draft.dir_hash()));
    let plan = json!({
        "format": "cmf-mtp-ffn-plan",
        "version": 2,
        "trunk": {"mode": "preserve"},
        "mtp": {"mode": "compact", "layers": mtp_layers},
        "selection": selection,
    });
    let default_plan = format!("{output}.mtp-plan.json");
    let plan_path = plan_out.unwrap_or(&default_plan);
    ensure!(
        !paths_collide(Path::new(plan_path), Path::new(model_path))
            && !paths_collide(Path::new(plan_path), &draft_path)
            && !paths_collide(Path::new(plan_path), Path::new(output)),
        "MTP compact: --plan-out must differ from source and output CMFs"
    );
    std::fs::write(plan_path, serde_json::to_vec_pretty(&plan)?)
        .with_context(|| format!("MTP compact: writing plan {plan_path:?}"))?;
    crate::tube::cmd_tube_bake(
        draft_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("MTP compact: non-UTF-8 draft path"))?,
        plan_path,
        output,
    )?;
    println!(
        "mtp compact: {} calibration record(s), {} token(s), {} draft layer(s)\n  plan: {}\n  source draft: {}\n  compact draft: {}",
        records,
        tokens,
        aggregate.len(),
        plan_path,
        draft_path.display(),
        output,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_probe, aligned_keep, paths_collide, rank_channel_scores, source_hash_fields,
        validate_probe,
    };
    use cortiq_engine::MtpFfnProbe;
    use std::fs;

    #[test]
    fn compact_width_stays_codec_aligned_and_fail_closed() {
        assert_eq!(aligned_keep(128, 1.0).unwrap(), 128);
        assert_eq!(aligned_keep(128, 0.76).unwrap(), 96);
        assert_eq!(aligned_keep(128, 0.25).unwrap(), 32);
        assert!(aligned_keep(128, 0.24).is_err());
        assert!(aligned_keep(127, 0.9).is_err());
        assert!(aligned_keep(128, f64::NAN).is_err());
    }

    #[test]
    fn ranking_uses_draft_activation_and_down_column_energy() {
        // Four columns, two output rows. Their column L2² values are
        // [1, 1, 9, 100]; combined with the activation energy below this
        // makes the order [2, 1, 3, 0].
        let down = vec![1.0, 1.0, 3.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let ranked = rank_channel_scores(&[1.0, 16.0, 4.0, 0.1], &down, 2, 4).unwrap();
        assert_eq!(
            ranked.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![2, 1, 3, 0]
        );
        // Equal scores must be deterministic: lower original index first.
        let tied = rank_channel_scores(&[1.0, 1.0], &[1.0, 1.0], 1, 2).unwrap();
        assert_eq!(
            tied.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(rank_channel_scores(&[f64::NAN], &[1.0], 1, 1).is_err());
        assert!(rank_channel_scores(&[f64::MAX], &[f32::MAX], 1, 1).is_err());
    }

    #[test]
    fn selection_records_independent_main_and_draft_hashes() {
        let fields = source_hash_fields(0xabc, 0xdef);
        assert_eq!(
            fields
                .get("source_directory_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000abc")
        );
        assert_eq!(
            fields
                .get("source_draft_directory_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000def")
        );
    }

    #[test]
    fn probe_aggregation_fails_closed_on_invalid_or_overflowed_statistics() {
        let invalid = vec![MtpFfnProbe {
            mass: vec![1.0],
            squared_mass: vec![f64::NAN],
            observed_positions: 1,
        }];
        assert!(validate_probe(&invalid).is_err());

        let mut total = vec![MtpFfnProbe {
            mass: vec![f64::MAX],
            squared_mass: vec![1.0],
            observed_positions: 1,
        }];
        let next = vec![MtpFfnProbe {
            mass: vec![f64::MAX],
            squared_mass: vec![1.0],
            observed_positions: 1,
        }];
        assert!(accumulate_probe(&mut total, &next).is_err());
    }

    #[test]
    fn source_and_alias_output_are_detected_as_the_same_path() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("draft.cmf");
        fs::write(&source, b"cmf").unwrap();
        let alias = temp.path().join(".").join("draft.cmf");
        assert!(paths_collide(&source, &alias));
        assert!(!paths_collide(&source, &temp.path().join("new-output.cmf")));
    }
}
