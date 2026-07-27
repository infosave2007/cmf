//! Physical MoE expert defrag (DTG-MA over routed experts, claims 9/10
//! applied to MoE): drop the experts a task never routes to, renumber
//! the kept ones into a contiguous per-layer prefix, and slice the
//! router's rows to match — the specialist neither stores nor considers
//! them. Selection statistics come from a claim-12 B-field dump
//! (`CMF_MOE_STATS` counts over a task-representative run); per layer
//! the smallest top set reaching `--cover` of the recorded routing mass
//! is kept. Runtime semantics equal the `CMF_MOE_MASK` runtime mask
//! (softmax renormalizes over the kept set) — gate on the same ppl A/B.

use anyhow::{bail, Context};
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
            shape: entry.shape.iter().map(|&d| d as usize).collect(),
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

pub fn cmd_moe_defrag(
    model_path: &str,
    stats_path: &str,
    cover: f64,
    output: &str,
) -> anyhow::Result<()> {
    if !(cover > 0.0 && cover <= 1.0) {
        bail!("--cover must be in (0, 1]");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let stats: HashMap<String, Vec<u64>> = serde_json::from_str(
        &std::fs::read_to_string(stats_path).with_context(|| format!("reading {stats_path}"))?,
    )
    .with_context(|| format!("parsing {stats_path}"))?;

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
            kept.iter().enumerate().map(|(new, &old)| (old, new)).collect(),
        );
    }
    if remap.is_empty() {
        bail!("no usable layer stats in {stats_path}");
    }

    // Pass 1: gather each masked layer's router rows into owned buffers
    // (small: kept×hidden f32); experts borrow the source mmap directly.
    let mut routers: HashMap<usize, (Vec<u8>, Vec<usize>)> = HashMap::new(); // tensor idx → (data, shape)
    for (ti, entry) in model.tensors.iter().enumerate() {
        let Some(li) = router_layer(&entry.name) else {
            continue;
        };
        let Some(map) = remap.get(&li) else { continue };
        if entry.dtype != TensorDtype::F32 {
            bail!("{}: router dtype {:?} != F32", entry.name, entry.dtype);
        }
        let ne = entry.shape[0] as usize;
        let hidden = entry.shape[1] as usize;
        let src = model.entry_bytes(entry);
        let row = hidden * 4;
        let mut kept: Vec<usize> = map.keys().copied().collect();
        kept.sort_unstable();
        let mut data = Vec::with_capacity(kept.len() * row);
        for &old in &kept {
            if old >= ne {
                bail!("{}: stats index {old} >= {ne} rows", entry.name);
            }
            data.extend_from_slice(&src[old * row..(old + 1) * row]);
        }
        routers.insert(ti, (data, vec![kept.len(), hidden]));
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
                            shape: entry.shape.iter().map(|&d| d as usize).collect(),
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
            shape: entry.shape.iter().map(|&d| d as usize).collect(),
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
    let stats_bytes = std::fs::read(stats_path)?;
    let prov = serde_json::json!({
        "tool": format!("cortiq moe-defrag {}", env!("CARGO_PKG_VERSION")),
        "cover": cover,
        "stats_hash64": format!("{:016x}", cortiq_core::hash64(&stats_bytes)),
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
