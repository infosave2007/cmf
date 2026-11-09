//! Physical MoE expert defrag (DTG-MA over routed experts, claims 9/10
//! applied to MoE): drop the experts a task never routes to, renumber
//! the kept ones into a contiguous per-layer prefix, and slice the
//! router's rows to match — the specialist neither stores nor considers
//! them. Selection statistics come from a claim-12 B-field dump
//! (`CMF_MOE_STATS` counts over a task-representative run); per layer
//! the smallest top set reaching `--cover` of the recorded routing mass
//! is kept. Runtime semantics equal the `CMF_MOE_MASK` runtime mask
//! (softmax renormalizes over the kept set) — gate on the same ppl A/B.

use anyhow::{Context, bail, ensure};
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
/// tightly in directory order by default. `execution_order` instead puts
/// weights in decode traversal order, with every MoE expert's gate/up/down
/// projections together. Payloads stream from the source mmap (`write_ref`),
/// so a 19 GB file compacts without RAM spikes.
pub fn cmd_compact(
    model_path: &str,
    output: &str,
    execution_order: bool,
) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "compact")?;
    // Capture source metadata before writing: `output` may name this same
    // file (or an alias of it), and a successful atomic install replaces that
    // directory entry only after all mmap-backed payloads have been copied.
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let mut specs: Vec<TensorSpecRef> = model
        .tensors
        .iter()
        .map(|entry| TensorSpecRef {
            name: entry.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.to_vec(),
            data: model.entry_bytes(entry),
        })
        .collect();
    if execution_order {
        sort_specs_for_execution_order(&mut specs);
    }
    let tensor_count = specs.len();

    // `open_sharded` merges all source shards into one logical model, while
    // `write_ref` emits exactly one standalone file. Retaining the first
    // source shard's coordinates here would make the output look for sibling
    // files that do not exist.
    let mut header = model.header.clone();
    header.shard = None;

    // Never hand `write_ref` the final path: it truncates its destination
    // before streaming its borrowed mmap payloads. Staging beside the final
    // output makes the replacement atomic and also keeps it on the same
    // filesystem. This protects same-path, symlink, and hard-link aliases of
    // the source as well as pre-existing outputs on failure.
    let output_path = std::path::Path::new(output);
    let temp_path = crate::gguf::qwen_image_temp_path(output_path)?;
    let write_result: anyhow::Result<()> = (|| {
        CmfModel::write_ref(
            &temp_path,
            &header,
            &specs,
            Some(&model.masks),
            model.vocab.as_deref(),
        )?;
        std::fs::File::open(&temp_path)
            .with_context(|| format!("open staged compact output {}", temp_path.display()))?
            .sync_all()
            .with_context(|| format!("sync staged compact output {}", temp_path.display()))?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(err);
    }

    // Close every source mmap before replacing the output. This is required
    // for same-path compaction on platforms that do not permit renaming over
    // an open mapped file (and is harmless on Unix).
    drop(specs);
    drop(model);
    if let Err(err) = std::fs::rename(&temp_path, output_path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(anyhow::Error::from(err).context(format!(
            "atomically install compact output {}",
            output_path.display()
        )));
    }
    crate::gguf::qwen_image_sync_parent(output_path)?;

    let out_sz = std::fs::metadata(output_path)?.len() as f64 / 1e9;
    let layout = if execution_order {
        " (execution order)"
    } else {
        ""
    };
    println!(
        "compact{layout}: {tensor_count} tensors\n{model_path} {in_sz:.2} GB -> {output} {out_sz:.2} GB ({:+.1}%)",
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}

/// Order a compacted payload stream in the same deterministic traversal order
/// used by conversion. Keep this separate from collection so the ordinary
/// `compact` path preserves the source directory order byte-for-byte.
fn sort_specs_for_execution_order(specs: &mut [TensorSpecRef<'_>]) {
    specs.sort_by(|a, b| {
        crate::convert::exec_order_key(&a.name)
            .cmp(&crate::convert::exec_order_key(&b.name))
            .then_with(|| a.name.cmp(&b.name))
    });
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
    crate::knowledge::refuse_genome_rewrite(&model, "moe-mask")?;
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
    cmd_moe_defrag_with(model_path, stats_path, &DefragOpts { cover, keep: None, merge: None }, output)
}

/// How `moe-defrag` picks the kept experts and what else it writes.
pub struct DefragOpts<'a> {
    /// Per-layer routing-mass fraction to keep (ignored with `keep`).
    pub cover: f64,
    /// Keep exactly this many experts in every MoE layer, the most routed
    /// first: one bank geometry for all layers (the Qwen3.8-Flash-Next
    /// device path sizes every layer's expert table alike).
    pub keep: Option<usize>,
    /// Another CMF whose tensors the output also carries (the names the
    /// source lacks), e.g. the `.mtp.cmf` draft head: one file instead of two.
    pub merge: Option<&'a str>,
}

/// Magic of the qwen4 device path's routing profile (`CMF_QWEN_PROFILE_SAVE`):
/// `[magic 8][layers u64][experts u64][tokens u64][counts u32 × layers·experts]`.
const QWEN_PROFILE_MAGIC: &[u8; 8] = b"CMFQ4PF\0";

/// A qwen4 routing profile as `{layer: counts}` JSON text, `None` when the
/// bytes are not one. The token count rides along for the provenance.
fn qwen_profile_as_stats(bytes: &[u8]) -> Option<(String, u64)> {
    if bytes.len() < 32 || &bytes[..8] != QWEN_PROFILE_MAGIC {
        return None;
    }
    let rd = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let (nl, ne, tokens) = (rd(8) as usize, rd(16) as usize, rd(24));
    if bytes.len() < 32 + nl * ne * 4 {
        return None;
    }
    let mut map = serde_json::Map::new();
    for l in 0..nl {
        let counts: Vec<u64> = (0..ne)
            .map(|e| {
                let o = 32 + (l * ne + e) * 4;
                u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as u64
            })
            .collect();
        map.insert(l.to_string(), serde_json::json!(counts));
    }
    Some((serde_json::Value::Object(map).to_string(), tokens))
}

/// The kept experts of one layer, ascending (renumbering keeps relative
/// order): the top `keep` by count, or the smallest top set reaching `cover`
/// of the layer's routing mass. Ties go to the lower index.
/// The checkpoint expert behind every slot of a `moe-defrag` cut, `{layer: [source expert
/// of each slot]}`; `None` for a file that was never cut. Read from the cut's provenance
/// (`kept_ids`), or, for a cut written before it recorded them, recomputed from the routing
/// stats that made the cut from an uncut source (`stats_path`, checked against its hash).
pub(crate) fn kept_ids_of(model: &CmfModel, stats_path: Option<&str>) -> anyhow::Result<Option<HashMap<usize, Vec<usize>>>> {
    let Some(prov) = model.header.provenance.as_ref().and_then(|p| p.get("moe_defrag")) else {
        return Ok(None);
    };
    let stats = match stats_path {
        Some(p) => Some((p, std::fs::read(p).with_context(|| format!("reading {p}"))?)),
        None => None,
    };
    kept_ids_from(prov, stats.as_ref().map(|(p, b)| (*p, b.as_slice()))).map(Some)
}

/// [`kept_ids_of`] over a cut's `moe_defrag` provenance and the bytes of the stats that made it.
fn kept_ids_from(prov: &serde_json::Value, stats: Option<(&str, &[u8])>) -> anyhow::Result<HashMap<usize, Vec<usize>>> {
    if let Some(ids) = prov.get("kept_ids").and_then(|v| v.as_object()) {
        let mut out = HashMap::new();
        for (k, v) in ids {
            let li: usize = k.parse().with_context(|| format!("kept_ids: layer {k}"))?;
            out.insert(li, serde_json::from_value(v.clone()).with_context(|| format!("kept_ids of layer {k}"))?);
        }
        return Ok(out);
    }
    let Some((p, bytes)) = stats else {
        bail!("the file is a moe-defrag cut that does not record its source experts: pass the routing stats that made the cut (--defrag-stats)");
    };
    let text = match qwen_profile_as_stats(bytes) {
        Some((text, _)) => text,
        None => String::from_utf8(bytes.to_vec()).with_context(|| format!("{p}: not UTF-8 JSON"))?,
    };
    let want = prov.get("stats_hash64").and_then(|v| v.as_str()).unwrap_or("");
    let got = format!("{:016x}", cortiq_core::hash64(text.as_bytes()));
    ensure!(got == want, "{p}: stats hash {got}, the cut was made from {want}");
    let cover = prov.get("cover").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let keep = prov.get("keep").and_then(|v| v.as_u64()).map(|k| k as usize);
    let stats: HashMap<String, Vec<u64>> = serde_json::from_str(&text).with_context(|| format!("parsing {p}"))?;
    let mut out = HashMap::new();
    for (k, counts) in &stats {
        let Ok(li) = k.parse::<usize>() else { continue };
        if counts.iter().sum::<u64>() > 0 {
            out.insert(li, pick_kept(counts, cover, keep));
        }
    }
    Ok(out)
}

fn pick_kept(counts: &[u64], cover: f64, keep: Option<usize>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    let mut kept = match keep {
        Some(k) => order[..k.min(order.len())].to_vec(),
        None => {
            let total: u64 = counts.iter().sum();
            let mut acc = 0u64;
            let mut kept = Vec::new();
            for &e in &order {
                kept.push(e);
                acc += counts[e];
                if (acc as f64) >= cover * (total as f64) {
                    break;
                }
            }
            kept
        }
    };
    kept.sort_unstable();
    kept
}

pub fn cmd_moe_defrag_with(
    model_path: &str,
    stats_path: Option<&str>,
    opts: &DefragOpts<'_>,
    output: &str,
) -> anyhow::Result<()> {
    let cover = opts.cover;
    if !(cover > 0.0 && cover <= 1.0) {
        bail!("--cover must be in (0, 1]");
    }
    if opts.keep == Some(0) {
        bail!("--keep must be at least 1");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "moe-defrag")?;
    // Stats: an explicit CMF_MOE_STATS dump (or a qwen4 device routing
    // profile), or — for re-defragging an already-cut specialist tighter —
    // the routing counts embedded in the source file's provenance by a
    // previous moe-defrag.
    let mut profile_tokens: Option<u64> = None;
    let (stats_text, stats_origin): (String, String) = match stats_path {
        Some(p) => {
            let bytes = std::fs::read(p).with_context(|| format!("reading {p}"))?;
            match qwen_profile_as_stats(&bytes) {
                Some((text, tokens)) => {
                    profile_tokens = Some(tokens);
                    (text, format!("{p} (qwen4 routing profile, {tokens} tokens)"))
                }
                None => (
                    String::from_utf8(bytes).with_context(|| format!("{p}: not UTF-8 JSON"))?,
                    p.to_string(),
                ),
            }
        }
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
        let kept = pick_kept(counts, cover, opts.keep);
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

    // `--merge`: the other file's tensors the source lacks ride along (the
    // qwen4 MTP draft head reads them from the main file when present).
    let merged = match opts.merge {
        Some(p) => Some(Arc::new(CmfModel::open(p).with_context(|| format!("opening --merge {p}"))?)),
        None => None,
    };
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
    let mut merged_n = 0usize;
    if let Some(m) = merged.as_ref() {
        let have: std::collections::HashSet<&str> = model.tensors.iter().map(|e| e.name.as_str()).collect();
        for entry in &m.tensors {
            if have.contains(entry.name.as_str()) {
                continue;
            }
            merged_n += 1;
            specs.push(TensorSpecRef {
                name: entry.name.clone(),
                dtype: entry.dtype,
                shape: entry.shape.to_vec(),
                data: m.entry_bytes(entry),
            });
        }
        if merged_n == 0 {
            bail!("--merge {}: every tensor already exists in the source", opts.merge.unwrap_or(""));
        }
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
    // The checkpoint expert behind every kept slot, so a later bake from the original
    // checkpoint (`hot-bake --experts`) reads each slot's own weights. A cut of a cut
    // composes with its source's list; a source cut that records none passes none on.
    let src_ids = kept_ids_of(&model, None).ok().flatten();
    let src_cut = model.header.provenance.as_ref().and_then(|p| p.get("moe_defrag")).is_some();
    let mut kept_ids = serde_json::Map::new();
    if !src_cut || src_ids.is_some() {
        for (&li, m) in &remap {
            let mut ids = vec![0usize; m.len()];
            for (&old, &new) in m {
                ids[new] = match &src_ids {
                    Some(s) => match s.get(&li).and_then(|v| v.get(old)) {
                        Some(&id) => id,
                        None => bail!("layer {li}: the source cut lists no expert {old}"),
                    },
                    None => old,
                };
            }
            kept_ids.insert(li.to_string(), serde_json::json!(ids));
        }
    }
    let mut prov = serde_json::json!({
        "tool": format!("cortiq moe-defrag {}", env!("CARGO_PKG_VERSION")),
        "cover": cover,
        "keep": opts.keep,
        "profile_tokens": profile_tokens,
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
        "routing_counts": serde_json::Value::Object(routing_counts.clone()),
    });
    if !kept_ids.is_empty() {
        prov["kept_ids"] = serde_json::Value::Object(kept_ids);
    }
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
    // A qwen4 routing profile in, a profile in the new numbering out
    // (`<output stem>.profile`): it warms the cut model's expert arena
    // (`CMF_QWEN_PROFILE`) exactly as the source profile warmed the source.
    let mut profile_out = None;
    if let (Some(tokens), Some(k)) = (profile_tokens, opts.keep) {
        let nl = remap.keys().max().map_or(0, |&l| l + 1);
        if remap.len() == nl && remap.values().all(|m| m.len() == k) {
            let mut out = Vec::with_capacity(32 + nl * k * 4);
            out.extend_from_slice(QWEN_PROFILE_MAGIC);
            out.extend_from_slice(&(nl as u64).to_le_bytes());
            out.extend_from_slice(&(k as u64).to_le_bytes());
            out.extend_from_slice(&tokens.to_le_bytes());
            for li in 0..nl {
                let counts = &routing_counts[&li.to_string()];
                for e in 0..k {
                    let c = counts[e].as_u64().unwrap_or(0).min(u32::MAX as u64) as u32;
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            let stem = output.strip_suffix(".cmf").unwrap_or(output);
            let path = format!("{stem}.profile");
            std::fs::write(&path, out).with_context(|| format!("writing {path}"))?;
            profile_out = Some(path);
        }
    }
    let in_sz = std::fs::metadata(model_path)?.len() as f64 / 1e9;
    let out_sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    let how = match opts.keep {
        Some(k) => format!("top {k} a layer"),
        None => format!("cover {:.0}%", cover * 100.0),
    };
    println!(
        "moe-defrag: kept {kept_experts} expert tensors, dropped {dropped} ({} MoE layers, {how})\n{model_path} {in_sz:.1} GB -> {output} {out_sz:.1} GB ({:+.0}%)",
        remap.len(),
        (out_sz / in_sz - 1.0) * 100.0
    );
    if merged_n > 0 {
        println!("moe-defrag: {merged_n} tensors merged from {}", opts.merge.unwrap_or(""));
    }
    if let Some(p) = profile_out {
        println!("moe-defrag: routing profile in the new numbering -> {p}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> TensorSpecRef<'static> {
        TensorSpecRef {
            name: name.to_string(),
            dtype: TensorDtype::F32,
            shape: Vec::new(),
            data: &[],
        }
    }

    #[test]
    fn keep_takes_the_most_routed_and_ties_go_low() {
        let counts = [5u64, 9, 0, 9, 1, 7];
        assert_eq!(pick_kept(&counts, 0.95, Some(3)), vec![1, 3, 5]);
        assert_eq!(pick_kept(&counts, 0.95, Some(4)), vec![0, 1, 3, 5]);
        // cover: the smallest top set reaching the routing mass (31 of 31 at 1.0)
        assert_eq!(pick_kept(&counts, 0.5, None), vec![1, 3]);
        assert_eq!(pick_kept(&counts, 1.0, None), vec![0, 1, 3, 4, 5]);
    }

    #[test]
    fn qwen_profile_reads_as_layer_counts() {
        let (nl, ne) = (2usize, 3usize);
        let mut b = QWEN_PROFILE_MAGIC.to_vec();
        for v in [nl as u64, ne as u64, 77] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for c in [4u32, 0, 2, 1, 1, 9] {
            b.extend_from_slice(&c.to_le_bytes());
        }
        let (text, tokens) = qwen_profile_as_stats(&b).unwrap();
        assert_eq!(tokens, 77);
        let v: HashMap<String, Vec<u64>> = serde_json::from_str(&text).unwrap();
        assert_eq!(v["0"], vec![4, 0, 2]);
        assert_eq!(v["1"], vec![1, 1, 9]);
        assert!(qwen_profile_as_stats(b"{\"0\": [1]}").is_none());
    }

    #[test]
    fn a_cut_names_its_source_experts() {
        let stats = br#"{"0": [5, 9, 0, 9, 1, 7], "1": [0, 0, 3, 1, 2, 8]}"#;
        let hash = format!("{:016x}", cortiq_core::hash64(stats));
        // A cut that predates kept_ids: recomputed from the stats that made it.
        let prov = serde_json::json!({"cover": 0.95, "keep": 3, "stats_hash64": hash});
        let ids = kept_ids_from(&prov, Some(("s.json", stats))).unwrap();
        assert_eq!(ids[&0], vec![1, 3, 5]);
        assert_eq!(ids[&1], vec![2, 4, 5]);
        assert!(kept_ids_from(&prov, None).is_err());
        assert!(kept_ids_from(&prov, Some(("t.json", br#"{"0": [1, 2, 3, 4, 5, 6]}"#))).is_err());
        // A cut that records them: read as written.
        let prov = serde_json::json!({"kept_ids": {"0": [4, 0, 7], "3": [1]}});
        let ids = kept_ids_from(&prov, None).unwrap();
        assert_eq!(ids[&0], vec![4, 0, 7]);
        assert_eq!(ids[&3], vec![1]);
    }

    #[test]
    fn compact_execution_order_groups_moe_expert_projections() {
        let mut specs = vec![
            spec("lm_head.weight"),
            spec("model.layers.10.mlp.experts.1.up_proj.weight"),
            spec("model.layers.2.mlp.experts.1.down_proj.weight"),
            spec("model.layers.2.mlp.experts.0.up_proj.weight"),
            spec("model.layers.2.mlp.experts.1.gate_proj.weight"),
            spec("model.layers.2.mlp.experts.0.down_proj.weight"),
            spec("model.layers.2.mlp.experts.0.gate_proj.weight"),
            spec("model.layers.2.mlp.gate.weight"),
            spec("model.embed_tokens.weight"),
            spec("model.norm.weight"),
        ];

        sort_specs_for_execution_order(&mut specs);

        let names: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "model.embed_tokens.weight",
                "model.layers.2.mlp.gate.weight",
                "model.layers.2.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.0.up_proj.weight",
                "model.layers.2.mlp.experts.0.down_proj.weight",
                "model.layers.2.mlp.experts.1.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
                "model.layers.10.mlp.experts.1.up_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ]
        );
    }
}
