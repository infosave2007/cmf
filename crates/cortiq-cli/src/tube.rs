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
use cortiq_core::{CmfModel, TensorDtype, TensorSpec};
use serde_json::Value;
use std::sync::Arc;

use crate::convert::{Quant, quantize_2d};

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
struct LayerPlan {
    order: Vec<usize>,
    widths: Vec<usize>,
}

pub fn cmd_tube_bake(model_path: &str, plan_path: &str, output: &str) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let arch = model.arch().clone();
    let (nl, hidden, inter) = (arch.num_layers, arch.hidden_size, arch.intermediate_size);
    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(plan_path).with_context(|| format!("reading {plan_path}"))?,
    )?;
    let tasks: Vec<String> = plan["tasks"]
        .as_array()
        .context("plan.tasks")?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    let layers: Vec<LayerPlan> = plan["layers"]
        .as_array()
        .context("plan.layers")?
        .iter()
        .map(|l| LayerPlan {
            order: l["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            widths: l["widths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
        })
        .collect();
    if layers.len() != nl {
        bail!("plan has {} layers, model has {nl}", layers.len());
    }
    // `active` is either [task][segment] — one segment table for the
    // whole stack — or [task][layer][segment], which lets a tube open in
    // the layers where the task needs it and stay shut elsewhere.
    let raw_active = plan["active"].as_array().context("plan.active")?;
    if raw_active.len() != tasks.len() {
        bail!(
            "plan.active has {} rows, {} tasks",
            raw_active.len(),
            tasks.len()
        );
    }
    let per_layer_active = raw_active
        .first()
        .and_then(|r| r.as_array())
        .and_then(|r| r.first())
        .is_some_and(|v| v.is_array());
    let active: Vec<Vec<Vec<bool>>> = raw_active
        .iter()
        .map(|t| {
            let rows = t.as_array().unwrap();
            if per_layer_active {
                rows.iter()
                    .map(|l| {
                        l.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_bool().unwrap_or(false))
                            .collect()
                    })
                    .collect()
            } else {
                let one: Vec<bool> = rows.iter().map(|v| v.as_bool().unwrap_or(false)).collect();
                vec![one; nl]
            }
        })
        .collect();
    for (li, l) in layers.iter().enumerate() {
        let sum: usize = l.widths.iter().sum();
        if sum != l.order.len() {
            bail!("layer {li}: widths sum {sum} != order len {}", l.order.len());
        }
        if l.widths.is_empty() || l.widths[0] == 0 {
            bail!("layer {li}: the core segment must be non-empty");
        }
        if l.order.iter().any(|&n| n >= inter) {
            bail!("layer {li}: neuron index out of range (inter {inter})");
        }
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

    // Only the stack's own FFN is rebuilt — an MTP block carries its
    // own `model.mtp.layers.*.mlp.*` triple and must copy through.
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
    let mut tensors: Vec<TensorSpec> = Vec::new();
    for t in &model.tensors {
        if is_stack_ffn(&t.name) {
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

    let mut header = model.header.clone();
    header.arch.intermediate_size = max_width;
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
    prov["tubes"] = serde_json::json!({
        "recipe": "task-tube defrag (permutation + segment cut)",
        "pre_intermediate": inter,
        "post_intermediate_max": max_width,
        "segments": nseg,
        "widths": layers.iter().map(|l| l.widths.clone()).collect::<Vec<_>>(),
        "tasks": tasks,
        "active": active,
    });
    header.provenance = Some(prov);
    CmfModel::write(output, &header, &tensors, Some(&catalog), model.vocab.as_deref())?;
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
    Ok(())
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
