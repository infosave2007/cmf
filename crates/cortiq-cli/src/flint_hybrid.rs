//! Build a layer-aligned q2tp/q4tp Recursive Flint hybrid without an
//! unquantized checkpoint.
//!
//! The operation is deliberately a whole-file rewrite rather than a recode:
//! selected routed gate/up payloads are borrowed directly from a verified q4
//! CMF mmap, and every other payload is borrowed from a verified q2 CMF mmap.
//! That retains the q4 source bytes exactly and avoids a large RAM allocation
//! or a lossy q2 -> float -> q4 round trip.

use crate::flint;
use anyhow::{Context, bail, ensure};
use cortiq_core::{CmfHeader, CmfModel, TensorDtype, TensorEntry, TensorSpecRef};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const FIRST_ROUTED_LAYER: usize = 2;
// The default WGPU Flint live-set guard admits a complete graph through two
// q4-routed upgrades on the known 24 GiB profile.  The third exceeds that
// default budget; do not turn a layout warning into advice to defeat it.
const DEFAULT_FULL_GRAPH_MAX_UPGRADED_LAYERS: usize = 2;

fn routed_layer_count() -> usize {
    flint::FLINT_LAYERS - FIRST_ROUTED_LAYER
}

fn all_routed_gate_up_names() -> HashSet<String> {
    let mut names = HashSet::with_capacity(routed_layer_count() * flint::FLINT_EXPERTS * 2);
    for layer in FIRST_ROUTED_LAYER..flint::FLINT_LAYERS {
        for expert in 0..flint::FLINT_EXPERTS {
            for leaf in ["gate_proj", "up_proj"] {
                names.insert(flint::expert_name(layer, expert, leaf));
            }
        }
    }
    names
}

fn selected_routed_gate_up_names(layers: &[usize]) -> HashSet<String> {
    let mut names = HashSet::with_capacity(layers.len() * flint::FLINT_EXPERTS * 2);
    for &layer in layers {
        for expert in 0..flint::FLINT_EXPERTS {
            for leaf in ["gate_proj", "up_proj"] {
                names.insert(flint::expert_name(layer, expert, leaf));
            }
        }
    }
    names
}

/// Parse a deliberately explicit routed-layer selection.  `all` and `none`
/// are accepted only as the complete value, so an invocation can never
/// silently inherit an unsafe default selection.
fn parse_layers(spec: &str) -> anyhow::Result<Vec<usize>> {
    let spec = spec.trim();
    if spec.eq_ignore_ascii_case("all") {
        return Ok((FIRST_ROUTED_LAYER..flint::FLINT_LAYERS).collect());
    }
    if spec.eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    ensure!(
        !spec.is_empty(),
        "--layers is required; use e.g. --layers 2,7-10 (or explicit 'all'/'none')"
    );

    let mut selected = std::collections::BTreeSet::new();
    for item in spec.split(',') {
        let item = item.trim();
        ensure!(!item.is_empty(), "empty item in --layers '{spec}'");
        ensure!(
            !item.eq_ignore_ascii_case("all") && !item.eq_ignore_ascii_case("none"),
            "'all' and 'none' must be the complete --layers value"
        );
        let (first, last) = if let Some((first, last)) = item.split_once('-') {
            ensure!(
                !last.contains('-'),
                "invalid layer range '{item}' in --layers"
            );
            (parse_layer(first)?, parse_layer(last)?)
        } else {
            let layer = parse_layer(item)?;
            (layer, layer)
        };
        ensure!(first <= last, "descending layer range '{item}' in --layers");
        for layer in first..=last {
            ensure!(
                selected.insert(layer),
                "layer {layer} appears more than once in --layers"
            );
        }
    }
    Ok(selected.into_iter().collect())
}

fn parse_layer(raw: &str) -> anyhow::Result<usize> {
    let layer = raw
        .trim()
        .parse::<usize>()
        .with_context(|| format!("invalid Flint layer '{raw}' in --layers"))?;
    ensure!(
        (FIRST_ROUTED_LAYER..flint::FLINT_LAYERS).contains(&layer),
        "Flint routed layers are {FIRST_ROUTED_LAYER}..{}; got {layer}",
        flint::FLINT_LAYERS - 1
    );
    Ok(layer)
}

fn profile_for_selection(layers: &[usize]) -> flint::QuantProfile {
    if layers.is_empty() {
        flint::QuantProfile::Q2tp
    } else if layers.len() == routed_layer_count() {
        flint::QuantProfile::Q4tp
    } else {
        flint::QuantProfile::Hybrid
    }
}

fn fits_default_full_graph_budget(layers: &[usize]) -> bool {
    layers.len() <= DEFAULT_FULL_GRAPH_MAX_UPGRADED_LAYERS
}

fn require_standalone_source(model: &CmfModel, role: &str) -> anyhow::Result<()> {
    ensure!(
        model.header.shard.is_none(),
        "{role} '{}' is sharded; flint-hybrid accepts standalone CMFs only",
        model.path.display()
    );
    ensure!(
        model.tensors.iter().all(|entry| entry.shard == 0),
        "{role} '{}' has external shard-backed tensor entries; flint-hybrid accepts standalone CMFs only",
        model.path.display()
    );
    crate::knowledge::refuse_genome_rewrite(model, "flint-hybrid")
}

fn verify_integrity(model: &CmfModel, role: &str) -> anyhow::Result<()> {
    let problems = model.verify();
    ensure!(
        problems.is_empty(),
        "{role} '{}' failed CMF integrity verification:\n  {}",
        model.path.display(),
        problems.join("\n  ")
    );
    Ok(())
}

/// Verify that the candidates differ *only* at the known q2/q4 routed
/// gate/up payloads.  Directory order is checked as well as names, shapes,
/// dtypes, byte lengths, and recorded hashes so borrowing by paired position
/// cannot accidentally blend related-but-different exports.
fn validate_pair_catalogues(
    q2: &[TensorEntry],
    q4: &[TensorEntry],
    routed_gate_up: &HashSet<String>,
) -> anyhow::Result<()> {
    ensure!(
        q2.len() == q4.len(),
        "q2 base has {} tensors but q4 source has {}",
        q2.len(),
        q4.len()
    );
    let mut seen_targets = HashSet::with_capacity(routed_gate_up.len());
    for (index, (base, donor)) in q2.iter().zip(q4).enumerate() {
        ensure!(
            base.name == donor.name,
            "tensor directory differs at entry {index}: q2 has '{}', q4 has '{}'",
            base.name,
            donor.name
        );
        ensure!(
            base.shape == donor.shape,
            "tensor '{}' shape differs: q2 {:?}, q4 {:?}",
            base.name,
            base.shape,
            donor.shape
        );
        if routed_gate_up.contains(&base.name) {
            ensure!(
                seen_targets.insert(base.name.as_str()),
                "routed gate/up tensor '{}' appears more than once in q2 directory",
                base.name
            );
            ensure!(
                base.dtype == TensorDtype::Q2TiledP,
                "q2 routed tensor '{}' is {:?}, expected q2tp",
                base.name,
                base.dtype
            );
            ensure!(
                donor.dtype == TensorDtype::Q4TiledP,
                "q4 routed tensor '{}' is {:?}, expected q4tp",
                donor.name,
                donor.dtype
            );
            ensure!(
                base.shape.as_slice() == [flint::FLINT_EXPERT_WIDTH, flint::FLINT_HIDDEN],
                "routed tensor '{}' has shape {:?}, expected [{}, {}]",
                base.name,
                base.shape,
                flint::FLINT_EXPERT_WIDTH,
                flint::FLINT_HIDDEN
            );
        } else {
            ensure!(
                base.dtype == donor.dtype,
                "non-routed tensor '{}' dtype differs: q2 {:?}, q4 {:?}",
                base.name,
                base.dtype,
                donor.dtype
            );
            ensure!(
                base.nbytes == donor.nbytes,
                "non-routed tensor '{}' byte length differs: q2 {}, q4 {}",
                base.name,
                base.nbytes,
                donor.nbytes
            );
            ensure!(
                base.hash == donor.hash,
                "non-routed tensor '{}' payload hash differs: q2 {:016x}, q4 {:016x}",
                base.name,
                base.hash,
                donor.hash
            );
        }
    }
    ensure!(
        seen_targets.len() == routed_gate_up.len(),
        "Flint candidate pair is missing {} expected routed gate/up tensor(s)",
        routed_gate_up.len().saturating_sub(seen_targets.len())
    );
    Ok(())
}

fn payload_delta(q2: &CmfModel, q4: &CmfModel, selected: &HashSet<String>) -> anyhow::Result<u64> {
    q2.tensors
        .iter()
        .zip(&q4.tensors)
        .filter(|(base, _)| selected.contains(&base.name))
        .try_fold(0u64, |total, (base, donor)| {
            let delta = donor.nbytes.checked_sub(base.nbytes).ok_or_else(|| {
                anyhow::anyhow!(
                    "selected q4 tensor '{}' is smaller than its q2 base ({} < {})",
                    donor.name,
                    donor.nbytes,
                    base.nbytes
                )
            })?;
            total
                .checked_add(delta)
                .ok_or_else(|| anyhow::anyhow!("selected q4 payload delta overflows u64"))
        })
}

fn payload_bytes(model: &CmfModel) -> anyhow::Result<u64> {
    model.tensors.iter().try_fold(0u64, |total, entry| {
        total
            .checked_add(entry.nbytes)
            .ok_or_else(|| anyhow::anyhow!("CMF tensor payload bytes overflow u64"))
    })
}

fn hybrid_header(
    q2: &CmfModel,
    q4: &CmfModel,
    layers: &[usize],
    selected_tensor_count: usize,
    delta: u64,
) -> anyhow::Result<CmfHeader> {
    let profile = profile_for_selection(layers);
    let mut header = flint::repaired_flint_header(&q2.header, profile)?;
    // Full rewrites are single-file artifacts.  The source checks above make
    // this assignment documentary as well as defensive.
    header.shard = None;
    ensure!(
        header.genome.is_none(),
        "flint-hybrid refuses frozen genome input"
    );
    let provenance = header
        .provenance
        .as_mut()
        .and_then(Value::as_object_mut)
        .context("Flint hybrid provenance must be a JSON object")?;
    provenance.remove("flint_repair");
    provenance.insert("weight_quant".into(), json!(profile.name()));
    provenance.insert(
        "flint_hybrid".into(),
        json!({
            "kind": "raw-q4tp-routed-gate-up-copy",
            "base_directory_hash": format!("{:016x}", q2.dir_hash()),
            "q4_source_directory_hash": format!("{:016x}", q4.dir_hash()),
            "upgraded_layers": layers,
            "upgraded_tensor_count": selected_tensor_count,
            "routed_experts_per_layer": flint::FLINT_EXPERTS,
            "payload_delta_bytes": delta,
            "tensor_payloads": "selected routed gate/up q4tp payloads copied verbatim; no dequantization or requantization",
        }),
    );
    Ok(header)
}

fn validate_staged_output(
    temp_path: &Path,
    q2: &CmfModel,
    q4: &CmfModel,
    selected: &HashSet<String>,
    expected_profile: flint::QuantProfile,
    layers: &[usize],
) -> anyhow::Result<()> {
    let output = CmfModel::open(temp_path)
        .with_context(|| format!("open staged Flint hybrid {}", temp_path.display()))?;
    verify_integrity(&output, "staged Flint hybrid")?;
    ensure!(
        output.header.shard.is_none(),
        "staged Flint hybrid unexpectedly has shard metadata"
    );
    ensure!(
        output.header.genome.is_none(),
        "staged Flint hybrid unexpectedly has genome metadata"
    );
    let profile = flint::ensure_current_flint_metadata(&output)?;
    ensure!(
        profile == expected_profile,
        "staged Flint hybrid profile is {}, expected {}",
        profile.name(),
        expected_profile.name()
    );
    let hybrid = output
        .header
        .provenance
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|provenance| provenance.get("flint_hybrid"))
        .and_then(Value::as_object)
        .context("staged Flint hybrid is missing flint_hybrid provenance")?;
    ensure!(
        hybrid.get("upgraded_layers") == Some(&json!(layers)),
        "staged Flint hybrid provenance has the wrong upgraded_layers"
    );
    ensure!(
        output.tensors.len() == q2.tensors.len() && output.tensors.len() == q4.tensors.len(),
        "staged Flint hybrid tensor count differs from its sources"
    );
    for (index, ((written, base), donor)) in output
        .tensors
        .iter()
        .zip(&q2.tensors)
        .zip(&q4.tensors)
        .enumerate()
    {
        let expected = if selected.contains(&base.name) {
            donor
        } else {
            base
        };
        ensure!(
            written.name == expected.name
                && written.dtype == expected.dtype
                && written.shape == expected.shape
                && written.hash == expected.hash,
            "staged Flint hybrid tensor {index} ('{}') does not match its {} source",
            written.name,
            if selected.contains(&base.name) {
                "q4"
            } else {
                "q2"
            }
        );
    }
    Ok(())
}

fn canonical_input(path: &Path, role: &str) -> anyhow::Result<PathBuf> {
    fs::canonicalize(path).with_context(|| format!("canonicalize {role} input {}", path.display()))
}

fn refuse_existing_output(output: &Path) -> anyhow::Result<()> {
    match fs::symlink_metadata(output) {
        Ok(_) => bail!(
            "refusing to overwrite existing output '{}'; choose a new path",
            output.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect output {}", output.display())),
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Build a q2-base/q4-source Flint artifact whose selected *complete routed
/// layers* use the q4 donor's gate/up payloads.  Inputs are always verified;
/// the output is written to a same-directory temporary file, verified again,
/// synced, then atomically installed.
pub fn cmd_flint_hybrid(
    q2_base: &str,
    q4_source: &str,
    output: &str,
    layers_spec: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let layers = parse_layers(layers_spec)?;
    let q2_path = Path::new(q2_base);
    let q4_path = Path::new(q4_source);
    let output_path = Path::new(output);
    refuse_existing_output(output_path)?;
    let q2_canonical = canonical_input(q2_path, "q2 base")?;
    let q4_canonical = canonical_input(q4_path, "q4 source")?;
    ensure!(
        q2_canonical != q4_canonical,
        "q2 base and q4 source resolve to the same file"
    );

    let q2 =
        CmfModel::open(q2_path).with_context(|| format!("open q2 base {}", q2_path.display()))?;
    let q4 =
        CmfModel::open(q4_path).with_context(|| format!("open q4 source {}", q4_path.display()))?;
    require_standalone_source(&q2, "q2 base")?;
    require_standalone_source(&q4, "q4 source")?;

    let q2_profile = flint::inspect_quant_profile(&q2)?;
    ensure!(
        q2_profile == flint::QuantProfile::Q2tp,
        "q2 base has Flint profile {}, expected q2tp",
        q2_profile.name()
    );
    let q4_profile = flint::inspect_quant_profile(&q4)?;
    ensure!(
        q4_profile == flint::QuantProfile::Q4tp,
        "q4 source has Flint profile {}, expected q4tp",
        q4_profile.name()
    );

    let all_targets = all_routed_gate_up_names();
    validate_pair_catalogues(&q2.tensors, &q4.tensors, &all_targets)?;
    // The q4 donor supplies weight bytes only, but those bytes must still
    // mean the same token IDs and optional masks as the q2 base.  Do not
    // infer that from equal vocab *shape* in the embedding / head.
    ensure!(
        q2.vocab == q4.vocab,
        "q2 base and q4 source embed different tokenizer payloads"
    );
    let q2_tokenizer = serde_json::to_vec(&q2.header.tokenizer_config)
        .context("serialize q2 base tokenizer metadata for Flint hybrid validation")?;
    let q4_tokenizer = serde_json::to_vec(&q4.header.tokenizer_config)
        .context("serialize q4 source tokenizer metadata for Flint hybrid validation")?;
    ensure!(
        q2_tokenizer == q4_tokenizer,
        "q2 base and q4 source have different tokenizer metadata"
    );
    let q2_masks = serde_json::to_vec(&q2.masks)
        .context("serialize q2 base masks for Flint hybrid validation")?;
    let q4_masks = serde_json::to_vec(&q4.masks)
        .context("serialize q4 source masks for Flint hybrid validation")?;
    ensure!(
        q2_masks == q4_masks,
        "q2 base and q4 source have different task masks"
    );
    verify_integrity(&q2, "q2 base")?;
    verify_integrity(&q4, "q4 source")?;

    let selected = selected_routed_gate_up_names(&layers);
    let selected_tensor_count = selected.len();
    let expected_selected_count = layers.len() * flint::FLINT_EXPERTS * 2;
    ensure!(
        selected_tensor_count == expected_selected_count,
        "internal Flint layer selection has {selected_tensor_count} tensors, expected {expected_selected_count}"
    );
    let delta = payload_delta(&q2, &q4, &selected)?;
    let base_payload = payload_bytes(&q2)?;
    let output_payload = base_payload
        .checked_add(delta)
        .ok_or_else(|| anyhow::anyhow!("hybrid payload size overflows u64"))?;
    let expected_profile = profile_for_selection(&layers);

    println!(
        "Flint hybrid plan: {} routed layer(s), {} q4 gate/up tensor(s), profile {}",
        layers.len(),
        selected_tensor_count,
        expected_profile.name()
    );
    println!(
        "payload bytes: q2 base {base_payload} ({:.3} GiB) + q4 upgrades {delta} ({:.3} GiB) = {output_payload} ({:.3} GiB)",
        gib(base_payload),
        gib(delta),
        gib(output_payload)
    );
    if expected_profile == flint::QuantProfile::Hybrid {
        println!(
            "layout: complete routed gate/up layers only; this is valid for Flint's WGPU token/decode mixed-q2/q4 dispatch (actual residency still depends on CMF_GPU_VRAM_MB)"
        );
    }
    if fits_default_full_graph_budget(&layers) {
        println!(
            "WGPU default live-budget note: {} upgraded layer(s) is within the N=0..2 full-graph profile on the validated 24 GiB configuration",
            layers.len()
        );
    } else {
        eprintln!(
            "warning: N={} upgraded routed layers conflicts with the default WGPU live budget on the validated 24 GiB configuration; only N=0..2 preserves the default full graph. This builder does not recommend overriding that guard.",
            layers.len()
        );
    }
    if dry_run {
        println!("dry run: sources were verified; no output was written");
        return Ok(());
    }

    let header = hybrid_header(&q2, &q4, &layers, selected_tensor_count, delta)?;
    let specs: Vec<TensorSpecRef<'_>> = q2
        .tensors
        .iter()
        .zip(&q4.tensors)
        .map(|(base, donor)| {
            let source = if selected.contains(&base.name) {
                donor
            } else {
                base
            };
            let bytes = if selected.contains(&base.name) {
                q4.entry_bytes(donor)
            } else {
                q2.entry_bytes(base)
            };
            TensorSpecRef {
                name: source.name.clone(),
                dtype: source.dtype,
                shape: source.shape.clone(),
                data: bytes,
            }
        })
        .collect();

    // `write_ref` truncates its target.  It is therefore never given the
    // requested final path; qwen_image_temp_path reserves a unique sibling
    // so the final rename remains on the same filesystem.
    let temp_path = crate::gguf::qwen_image_temp_path(output_path)?;
    let build_result: anyhow::Result<()> = (|| {
        CmfModel::write_ref(
            &temp_path,
            &header,
            &specs,
            Some(&q2.masks),
            q2.vocab.as_deref(),
        )
        .with_context(|| format!("write staged Flint hybrid {}", temp_path.display()))?;
        fs::File::open(&temp_path)
            .with_context(|| format!("open staged Flint hybrid {}", temp_path.display()))?
            .sync_all()
            .with_context(|| format!("sync staged Flint hybrid {}", temp_path.display()))?;
        validate_staged_output(&temp_path, &q2, &q4, &selected, expected_profile, &layers)?;
        Ok(())
    })();
    if let Err(error) = build_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    // Close all mappings before installing.  This is important on platforms
    // that do not permit rename over an open mapped file, and is harmless on
    // Unix where the sources are distinct from the destination.
    drop(specs);
    drop(q4);
    drop(q2);
    if let Err(error) = refuse_existing_output(output_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temp_path, output_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(anyhow::Error::from(error).context(format!(
            "atomically install Flint hybrid {}",
            output_path.display()
        )));
    }
    crate::gguf::qwen_image_sync_parent(output_path)
        .with_context(|| format!("sync parent directory for {}", output_path.display()))?;
    let output_len = fs::metadata(output_path)
        .with_context(|| format!("stat Flint hybrid {}", output_path.display()))?
        .len();
    println!(
        "wrote Flint hybrid {}: {} bytes ({:.3} GiB), {} upgraded routed layer(s)",
        output_path.display(),
        output_len,
        gib(output_len),
        layers.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        name: String,
        dtype: TensorDtype,
        shape: Vec<usize>,
        nbytes: u64,
        hash: u64,
    ) -> TensorEntry {
        TensorEntry {
            name,
            dtype,
            shape,
            off: 0,
            nbytes,
            shard: 0,
            hash,
        }
    }

    fn routed_catalog(dtype: TensorDtype, nbytes: u64, hash_base: u64) -> Vec<TensorEntry> {
        let mut entries = Vec::with_capacity(routed_layer_count() * flint::FLINT_EXPERTS * 2 + 1);
        let mut i = 0u64;
        for layer in FIRST_ROUTED_LAYER..flint::FLINT_LAYERS {
            for expert in 0..flint::FLINT_EXPERTS {
                for leaf in ["gate_proj", "up_proj"] {
                    entries.push(entry(
                        flint::expert_name(layer, expert, leaf),
                        dtype,
                        vec![flint::FLINT_EXPERT_WIDTH, flint::FLINT_HIDDEN],
                        nbytes,
                        hash_base + i,
                    ));
                    i += 1;
                }
            }
        }
        entries.push(entry(
            "model.embed_tokens.weight".into(),
            TensorDtype::F16,
            vec![1],
            2,
            0xfeed_beef,
        ));
        entries
    }

    #[test]
    fn parse_layers_requires_an_explicit_valid_nonoverlapping_selection() {
        assert_eq!(parse_layers("2, 5-7, 48").unwrap(), vec![2, 5, 6, 7, 48]);
        assert_eq!(parse_layers("none").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_layers("all").unwrap().len(), routed_layer_count());
        for bad in ["", "1", "49", "7-5", "2,2", "2-4,4", "all,2", "2--3"] {
            assert!(parse_layers(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn selection_is_entire_routed_gate_up_layers_only() {
        let selected = selected_routed_gate_up_names(&[2, 48]);
        assert_eq!(selected.len(), 2 * flint::FLINT_EXPERTS * 2);
        assert!(selected.contains(&flint::expert_name(2, 0, "gate_proj")));
        assert!(selected.contains(&flint::expert_name(48, 255, "up_proj")));
        assert!(!selected.contains(&flint::expert_name(2, 0, "down_proj")));
    }

    #[test]
    fn paired_catalogue_requires_only_the_known_q2_q4_routed_delta() {
        let targets = all_routed_gate_up_names();
        let q2 = routed_catalog(TensorDtype::Q2TiledP, 284_672, 0x1000);
        let mut q4 = routed_catalog(TensorDtype::Q4TiledP, 546_816, 0x2000);
        validate_pair_catalogues(&q2, &q4, &targets).unwrap();

        // A shared/non-routed payload mismatch is never silently inherited.
        q4.last_mut().unwrap().hash ^= 1;
        assert!(validate_pair_catalogues(&q2, &q4, &targets).is_err());
    }

    #[test]
    fn selection_profile_is_hybrid_only_for_a_proper_subset() {
        assert_eq!(profile_for_selection(&[]), flint::QuantProfile::Q2tp);
        assert_eq!(
            profile_for_selection(&(FIRST_ROUTED_LAYER..flint::FLINT_LAYERS).collect::<Vec<_>>()),
            flint::QuantProfile::Q4tp
        );
        assert_eq!(profile_for_selection(&[2]), flint::QuantProfile::Hybrid);
        assert!(fits_default_full_graph_budget(&[2, 3]));
        assert!(!fits_default_full_graph_budget(&[2, 3, 4]));
    }
}
