//! Physical MoE expert defrag (DTG-MA over routed experts, claims 9/10
//! applied to MoE): drop the experts a task never routes to, renumber
//! the kept ones into a contiguous per-layer prefix, and slice the
//! router's rows to match — the specialist neither stores nor considers
//! them. Selection statistics come from a claim-12 B-field dump
//! (`CMF_MOE_STATS` counts over a task-representative run); per layer
//! the smallest top set reaching `--cover` of the recorded routing mass
//! is kept. Runtime semantics equal the `CMF_MOE_MASK` runtime mask
//! (softmax renormalizes over the kept set) — gate on the same ppl A/B.

use anyhow::{Context, bail};
use cortiq_core::{CmfModel, TensorDtype, TensorSpecRef};
use std::collections::HashMap;
use std::sync::Arc;

/// "model.layers.L.mlp.experts.E.<rest>" → (L, E, rest).
fn expert_parts(name: &str) -> Option<(usize, usize, &str)> {
    let t = name.strip_prefix("model.layers.")?;
    let (li, t) = t.split_once('.')?;
    let t = t.strip_prefix("mlp.experts.")?;
    let (e, rest) = t.split_once('.')?;
    Some((li.parse().ok()?, e.parse().ok()?, rest))
}

/// "model.layers.L.mlp.expert_bias" → L (noaux_tc selection bias —
/// one scalar per expert, sliced with the experts it scores).
fn expert_bias_layer(name: &str) -> Option<usize> {
    let t = name.strip_prefix("model.layers.")?;
    let (li, t) = t.split_once('.')?;
    (t == "mlp.expert_bias").then(|| li.parse().ok())?
}

/// "model.layers.L.mlp.gate.weight" → L.
fn router_layer(name: &str) -> Option<usize> {
    let t = name.strip_prefix("model.layers.")?;
    let (li, t) = t.split_once('.')?;
    (t == "mlp.gate.weight").then(|| li.parse().ok())?
}

/// Plain container rewrite: reclaims the dead directory/header tails
/// left by append-only skill growth (spec §9) and re-packs tensors
/// tightly in directory order. Payloads stream from the source mmap
/// (write_ref), so a 19 GB file compacts without RAM spikes.
pub fn cmd_compact(model_path: &str, output: &str) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let specs: Vec<TensorSpecRef> = model
        .tensors
        .iter()
        .map(|entry| TensorSpecRef {
            name: entry.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.to_vec(),
            data: model.entry_bytes(entry),
        })
        .collect();
    CmfModel::write_ref(
        output,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let out_sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    println!(
        "compact: {} tensors\n{model_path} {in_sz:.2} GB -> {output} {out_sz:.2} GB ({:+.1}%)",
        specs.len(),
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}

/// Bake a task's expert restriction as a FIRST-CLASS task mask (spec
/// §5 expert fields) instead of cutting the file: the full expert set
/// stays on disk, `run --task <name>` narrows routing to the mask's
/// experts at inference — one MoE file, many switchable specialists.
pub fn cmd_moe_mask(
    model_path: &str,
    stats_path: &str,
    cover: f64,
    name: &str,
    output: &str,
) -> anyhow::Result<()> {
    if !(cover > 0.0 && cover <= 1.0) {
        bail!("--cover must be in (0, 1]");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let arch = model.arch().clone();
    let Some(moe) = arch.moe.as_ref() else {
        bail!("{model_path}: not a MoE model (no arch.moe block)");
    };
    let ne = moe.num_experts;
    let stats: HashMap<String, Vec<u64>> = serde_json::from_str(
        &std::fs::read_to_string(stats_path).with_context(|| format!("reading {stats_path}"))?,
    )
    .with_context(|| format!("parsing {stats_path}"))?;

    let expert_b = ne.div_ceil(8);
    let mut expert_masks: Vec<Vec<u8>> = vec![Vec::new(); arch.num_layers];
    let mut kept_total = 0usize;
    let mut masked_layers = 0usize;
    for (k, counts) in &stats {
        let Ok(li) = k.parse::<usize>() else { continue };
        if li >= arch.num_layers || counts.len() != ne {
            continue;
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            continue;
        }
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_unstable_by_key(|&e| std::cmp::Reverse(counts[e]));
        let mut bits = vec![0u8; expert_b];
        let mut acc = 0u64;
        for &e in &order {
            bits[e / 8] |= 1 << (e % 8);
            kept_total += 1;
            acc += counts[e];
            if (acc as f64) >= cover * (total as f64) {
                break;
            }
        }
        expert_masks[li] = bits;
        masked_layers += 1;
    }
    if masked_layers == 0 {
        bail!("no usable layer stats in {stats_path}");
    }

    let mut catalog = model.masks.clone();
    if catalog.masks.iter().any(|m| m.name == name) {
        bail!("mask '{name}' already exists in {model_path}");
    }
    let task_id = catalog
        .masks
        .iter()
        .map(|m| m.task_id + 1)
        .max()
        .unwrap_or(1);
    let sparsity = 1.0 - kept_total as f32 / (masked_layers * ne) as f32;
    catalog.masks.push(cortiq_core::TaskMask {
        task_id,
        name: name.to_string(),
        description: Some(format!(
            "MoE expert mask (cover {:.0}%, {} layers)",
            cover * 100.0,
            masked_layers
        )),
        sparsity,
        quality: None, // measure with `cortiq ppl --task` before trusting
        ffn_masks: vec![vec![0xFF; arch.ffn_mask_bytes()]; arch.num_layers],
        head_masks: vec![vec![0xFF; arch.head_mask_bytes()]; arch.num_layers],
        layer_gates: vec![true; arch.num_layers],
        expert_masks,
        parent: None,
        has_hot_pack: false,
        priority: cortiq_core::MaskPriority::Normal,
    });

    let specs: Vec<TensorSpecRef> = model
        .tensors
        .iter()
        .map(|entry| TensorSpecRef {
            name: entry.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.to_vec(),
            data: model.entry_bytes(entry),
        })
        .collect();
    CmfModel::write_ref(
        output,
        &model.header,
        &specs,
        Some(&catalog),
        model.vocab.as_deref(),
    )?;
    println!(
        "moe-mask: '{name}' added ({masked_layers} layers, expert sparsity {:.0}%)\nactivate with: cortiq run {output} --task {name} …",
        sparsity * 100.0
    );
    Ok(())
}

pub fn cmd_moe_defrag(
    model_path: &str,
    stats_path: Option<&str>,
    cover: f64,
    output: &str,
) -> anyhow::Result<()> {
    if !(cover > 0.0 && cover <= 1.0) {
        bail!("--cover must be in (0, 1]");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    // Stats: an explicit CMF_MOE_STATS dump, or — for re-defragging an
    // already-cut specialist tighter — the routing counts embedded in
    // the source file's provenance by a previous moe-defrag.
    let (stats_text, stats_origin): (String, String) = match stats_path {
        Some(p) => (
            std::fs::read_to_string(p).with_context(|| format!("reading {p}"))?,
            p.to_string(),
        ),
        None => {
            let counts = model
                .header
                .provenance
                .as_ref()
                .and_then(|p| p.get("moe_defrag"))
                .and_then(|d| d.get("routing_counts"))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "--stats not given and the file carries no embedded \
                         routing counts (provenance.moe_defrag.routing_counts)"
                    )
                })?;
            (counts.to_string(), "embedded provenance".to_string())
        }
    };
    let stats: HashMap<String, Vec<u64>> =
        serde_json::from_str(&stats_text).with_context(|| format!("parsing {stats_origin}"))?;

    // Per-layer plan: kept old indices (ascending — renumbering keeps
    // relative order) and old → new contiguous index.
    let mut remap: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (k, counts) in &stats {
        let li: usize = match k.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let total: u64 = counts.iter().sum();
        if total == 0 {
            continue;
        }
        let mut order: Vec<usize> = (0..counts.len()).collect();
        order.sort_unstable_by_key(|&e| std::cmp::Reverse(counts[e]));
        let mut acc = 0u64;
        let mut kept = Vec::new();
        for &e in &order {
            kept.push(e);
            acc += counts[e];
            if (acc as f64) >= cover * (total as f64) {
                break;
            }
        }
        kept.sort_unstable();
        remap.insert(
            li,
            kept.iter()
                .enumerate()
                .map(|(new, &old)| (old, new))
                .collect(),
        );
    }
    if remap.is_empty() {
        bail!("no usable layer stats in {stats_origin}");
    }

    // Pass 1: gather each masked layer's router rows into owned buffers
    // (small: kept×hidden f32); experts borrow the source mmap directly.
    let mut routers: HashMap<usize, (Vec<u8>, Vec<usize>)> = HashMap::new(); // tensor idx → (data, shape)
    for (ti, entry) in model.tensors.iter().enumerate() {
        // Router rows [ne, hidden] and the noaux_tc selection bias
        // [ne] (Kimi/DeepSeek-V3/LFM2) slice by the same kept set —
        // a renumbered expert must keep ITS bias or selection breaks.
        let (li, hidden) = if let Some(li) = router_layer(&entry.name) {
            (li, Some(entry.shape[1]))
        } else if let Some(li) = expert_bias_layer(&entry.name) {
            (li, None)
        } else {
            continue;
        };
        let Some(map) = remap.get(&li) else { continue };
        let elem = match entry.dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            other => bail!("{}: dtype {other:?} — expected F32/F16", entry.name),
        };
        let ne = entry.shape[0];
        let src = model.entry_bytes(entry);
        let row = hidden.unwrap_or(1) * elem;
        let mut kept: Vec<usize> = map.keys().copied().collect();
        kept.sort_unstable();
        let mut data = Vec::with_capacity(kept.len() * row);
        for &old in &kept {
            if old >= ne {
                bail!("{}: stats index {old} >= {ne} rows", entry.name);
            }
            data.extend_from_slice(&src[old * row..(old + 1) * row]);
        }
        let shape = match hidden {
            Some(h) => vec![kept.len(), h],
            None => vec![kept.len()],
        };
        routers.insert(ti, (data, shape));
    }

    // Pass 2: the spec list — dropped experts skipped, kept ones
    // renumbered, routers swapped for their sliced rows.
    let mut specs: Vec<TensorSpecRef> = Vec::new();
    let mut dropped = 0usize;
    let mut kept_experts = 0usize;
    for (ti, entry) in model.tensors.iter().enumerate() {
        if let Some((data, shape)) = routers.get(&ti) {
            specs.push(TensorSpecRef {
                name: entry.name.clone(),
                dtype: entry.dtype,
                shape: shape.clone(),
                data,
            });
            continue;
        }
        if let Some((li, e, rest)) = expert_parts(&entry.name) {
            if let Some(map) = remap.get(&li) {
                match map.get(&e) {
                    Some(&new) => {
                        kept_experts += 1;
                        specs.push(TensorSpecRef {
                            name: format!("model.layers.{li}.mlp.experts.{new}.{rest}"),
                            dtype: entry.dtype,
                            shape: entry.shape.to_vec(),
                            data: model.entry_bytes(entry),
                        });
                    }
                    None => dropped += 1,
                }
                continue;
            }
        }
        specs.push(TensorSpecRef {
            name: entry.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.to_vec(),
            data: model.entry_bytes(entry),
        });
    }
    if dropped == 0 {
        bail!("nothing to drop — check the stats file / --cover");
    }

    // Honest provenance contract (spec §11.1, mirroring §11's defrag
    // block): what was cut, from what evidence, at what cover.
    let mut header = model.header.clone();
    let mut kept_per_layer: Vec<(usize, usize)> =
        remap.iter().map(|(&li, m)| (li, m.len())).collect();
    kept_per_layer.sort_unstable();
    // The claim-12 B-field rides INSIDE the specialist, remapped to the
    // new expert numbering — the file explains its own cut, and a later
    // `moe-defrag` without --stats can tighten it further.
    let mut routing_counts = serde_json::Map::new();
    for (k, counts) in &stats {
        let Ok(li) = k.parse::<usize>() else { continue };
        let Some(map) = remap.get(&li) else { continue };
        let mut remapped = vec![0u64; map.len()];
        for (&old, &new) in map {
            remapped[new] = counts[old];
        }
        routing_counts.insert(k.clone(), serde_json::json!(remapped));
    }
    let prov = serde_json::json!({
        "tool": format!("cortiq moe-defrag {}", env!("CARGO_PKG_VERSION")),
        "cover": cover,
        "stats_hash64": format!("{:016x}", cortiq_core::hash64(stats_text.as_bytes())),
        "num_experts_pre": header
            .arch
            .moe
            .as_ref()
            .map(|m| m.num_experts)
            .unwrap_or(0),
        "kept_per_layer": kept_per_layer
            .iter()
            .map(|&(_, k)| k)
            .collect::<Vec<_>>(),
        "routing_counts": serde_json::Value::Object(routing_counts),
    });
    match header.provenance.as_mut() {
        Some(serde_json::Value::Object(map)) => {
            map.insert("moe_defrag".into(), prov);
        }
        _ => {
            header.provenance = Some(serde_json::json!({ "moe_defrag": prov }));
        }
    }

    CmfModel::write_ref(
        output,
        &header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let out_sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    println!(
        "moe-defrag: kept {kept_experts} expert tensors, dropped {dropped} ({} MoE layers, cover {:.0}%)\n{model_path} {in_sz:.1} GB -> {output} {out_sz:.1} GB ({:+.0}%)",
        remap.len(),
        cover * 100.0,
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}
