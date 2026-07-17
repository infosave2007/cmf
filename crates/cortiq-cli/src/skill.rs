//! `cortiq skill` — bake, list and route-fit swarm skills (spec §9).
//!
//! A skill is a set of full-shape replacement tensors appended to the
//! container (`skill.{id}.{tensor}`), plus a registry record with the
//! recon-argmin selection subspace and the honest quality contract.
//! `skill add` grafts them from a REAL donor checkpoint of the same
//! architecture (any HF repo or local dir): the donor's chosen tensors
//! are quantized with the backbone's own per-tensor encoding and the
//! file is rewritten append-style — backbone bytes never change,
//! storage scales as |backbone| + Σ|deltas|.

use crate::convert::{
    canon_name, hf_download, looks_like_repo, open_model, parse_quant, quantize_2d, to_f32, Quant,
};
use base64::Engine as _;
use cortiq_core::quant::f32_to_f16;
use cortiq_core::mask::{MaskPriority, TaskMask};
use cortiq_core::{CmfModel, SelectionDescriptor, SkillRecord, TensorDtype, TensorSpec};
use cortiq_engine::{Pipeline, SamplerConfig};
use anyhow::Context as _;
use std::path::Path;
use std::sync::Arc;

/// Bitfield bytes for `n` attention heads.
fn nh_bytes(n: usize) -> usize {
    n.div_ceil(8)
}

/// Which tensor families a skill replaces.
#[derive(Clone, Copy, PartialEq)]
pub enum Families {
    Ffn,
    Attn,
    All,
}

impl Families {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "ffn" => Self::Ffn,
            "attn" => Self::Attn,
            "all" => Self::All,
            other => anyhow::bail!("unknown --tensors '{other}' (ffn | attn | all)"),
        })
    }

    fn suffixes(self) -> &'static [&'static str] {
        const FFN: &[&str] = &[
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        const ATTN: &[&str] = &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ];
        const ALL: &[&str] = &[
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ];
        match self {
            Self::Ffn => FFN,
            Self::Attn => ATTN,
            Self::All => ALL,
        }
    }
}

/// Parse `--layers`: `all`, `A-B`, or `i,j,k`.
pub fn parse_layers(spec: &str, num_layers: usize) -> anyhow::Result<Vec<usize>> {
    if spec == "all" {
        return Ok((0..num_layers).collect());
    }
    if let Some((a, b)) = spec.split_once('-') {
        let (a, b): (usize, usize) = (a.trim().parse()?, b.trim().parse()?);
        anyhow::ensure!(a <= b && b < num_layers, "--layers {spec}: out of 0..{num_layers}");
        return Ok((a..=b).collect());
    }
    let mut v = Vec::new();
    for part in spec.split(',') {
        let i: usize = part.trim().parse()?;
        anyhow::ensure!(i < num_layers, "--layers {spec}: layer {i} out of 0..{num_layers}");
        v.push(i);
    }
    anyhow::ensure!(!v.is_empty(), "--layers {spec}: empty");
    Ok(v)
}

fn b64_f16(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for &x in v {
        bytes.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Fit the recon-argmin selection subspace from example prompts: the
/// φ(x) mean plus a rank-K PCA basis (power iteration + deflation) of
/// the centered φ cloud. Rank is clamped to N−1 — with one prompt the
/// subspace degenerates to pure distance-to-mean, which still routes.
fn fit_selection(
    pipeline: &mut Pipeline,
    prompts: &[String],
    phi_layer: usize,
    rank: usize,
) -> SelectionDescriptor {
    let hidden = pipeline.hidden_size;
    let phis: Vec<Vec<f32>> = prompts
        .iter()
        .map(|p| {
            let ids = pipeline.tokenizer.encode(p);
            pipeline.probe_phi(&ids, phi_layer)
        })
        .collect();
    let n = phis.len();
    let mut mean = vec![0f32; hidden];
    for phi in &phis {
        for (m, v) in mean.iter_mut().zip(phi) {
            *m += v / n as f32;
        }
    }
    let mut centered: Vec<Vec<f32>> = phis
        .iter()
        .map(|phi| phi.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let rank = rank.min(n.saturating_sub(1)).min(8);
    let mut basis: Vec<f32> = Vec::with_capacity(rank * hidden);
    for _ in 0..rank {
        // Power iteration on Σ ccᵀ (implicitly, via the N vectors).
        let mut v = vec![1f32; hidden];
        for _ in 0..50 {
            let mut next = vec![0f32; hidden];
            for c in &centered {
                let dot: f32 = c.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (nx, cv) in next.iter_mut().zip(c) {
                    *nx += dot * cv;
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in next.iter_mut() {
                *x /= norm;
            }
            v = next;
        }
        // Deflate: remove the found component from every sample.
        for c in centered.iter_mut() {
            let dot: f32 = c.iter().zip(&v).map(|(a, b)| a * b).sum();
            for (cv, bv) in c.iter_mut().zip(&v) {
                *cv -= dot * bv;
            }
        }
        basis.extend_from_slice(&v);
    }
    SelectionDescriptor {
        metric: "mse".into(),
        phi_layer,
        mean: b64_f16(&mean),
        basis: b64_f16(&basis),
        rank,
    }
}

fn dtype_to_quant(d: TensorDtype) -> Option<Quant> {
    Some(match d {
        TensorDtype::Q8Row => Quant::Q8Row,
        TensorDtype::Q8_2f => Quant::Q8_2f,
        TensorDtype::Q4Block => Quant::Q4Block,
        TensorDtype::Q4Tiled => Quant::Q4Tiled,
        TensorDtype::F16 => Quant::F16,
        TensorDtype::Vbit | TensorDtype::VbitRo => Quant::Vbit,
        TensorDtype::Q1 => Quant::Q1,
        _ => return None,
    })
}

/// PPL of a text file through an (optionally overlaid) pipeline —
/// the claim-16 quality gate, same math as `cortiq ppl`.
fn ppl_of(
    model: &Arc<CmfModel>,
    skill: Option<&str>,
    text: &str,
    max_tokens: usize,
) -> anyhow::Result<f64> {
    let mut p = Pipeline::from_model_with_skill(model, SamplerConfig::default(), skill)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut ids = p.tokenizer.with_bos(p.tokenizer.encode(text));
    ids.truncate(max_tokens);
    Ok(p.ppl_ids(&ids))
}

#[allow(clippy::too_many_arguments)]
pub fn run_skill_add(
    model_path: &str,
    from: &str,
    id: &str,
    name: Option<&str>,
    layers_spec: &str,
    families: Families,
    prompts_file: Option<&str>,
    phi_layer: Option<usize>,
    rank: usize,
    quality_file: Option<&str>,
    quality_tokens: usize,
    min_delta: f32,
    skill_quant: Option<&str>,
    mean_bits: Option<f32>,
    sparse: Option<f32>,
    output: Option<&str>,
    hf_token: Option<&str>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "skill id must be [A-Za-z0-9_-]"
    );
    if let Some(k) = sparse {
        anyhow::ensure!(
            (0.05..=0.95).contains(&k),
            "--sparse {k}: keep fraction must be within 0.05..=0.95"
        );
        anyhow::ensure!(
            prompts_file.is_some(),
            "--sparse needs --prompts: the DTG-MA mask is derived from the task's activations"
        );
    }
    if let Some(b) = mean_bits {
        crate::convert::set_vbit_mean_bits(b);
    }
    let model = Arc::new(CmfModel::open(model_path)?);
    let num_layers = model.arch().num_layers;
    let layers = parse_layers(layers_spec, num_layers)?;

    // ── donor: local dir or HF repo (cached download) ──
    let donor_dir = if looks_like_repo(from) && !Path::new(from).exists() {
        hf_download(from, hf_token)?
    } else {
        Path::new(from).to_path_buf()
    };
    let shards = open_model(&donor_dir)?;
    println!("donor: {} ({} shard(s))", donor_dir.display(), shards.len());

    // ── graft: donor tensors for the chosen layers/families, quantized
    //    with the backbone's own per-tensor encoding ──
    let mut wanted: Vec<String> = Vec::new();
    for &li in &layers {
        for suf in families.suffixes() {
            wanted.push(format!("model.layers.{li}.{suf}"));
        }
    }
    let mut new_tensors: Vec<TensorSpec> = Vec::new();
    let mut ffn_vals: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut unchanged = 0usize;
    let mut unchanged_bytes = 0u64;
    let mut deltas: Vec<(String, f32)> = Vec::new();
    for want in &wanted {
        let Some(entry) = model.tensors.iter().find(|t| &t.name == want) else {
            skipped.push(format!("{want} (not in backbone)"));
            continue;
        };
        let mut found = false;
        'shards: for sh in &shards {
            for m in &sh.tensors {
                if canon_name(&m.name).as_deref() != Some(want.as_str()) {
                    continue;
                }
                anyhow::ensure!(
                    m.shape == entry.shape,
                    "{want}: donor shape {:?} != backbone {:?} — different architecture?",
                    m.shape,
                    entry.shape
                );
                let vals = to_f32(&m.dtype, sh.bytes(m))?;
                found = true;
                if sparse.is_some() && want.contains(".mlp.") {
                    ffn_vals.push((want.clone(), entry.shape.clone(), vals.clone()));
                }
                // Delta gate: neurons the fine-tune never touched are
                // not worth bytes. Compare the donor against the
                // backbone through the reference decoder and drop
                // tensors whose relative change is below --min-delta —
                // the runtime reads the backbone entry for them, which
                // is exactly what the donor holds there anyway.
                if min_delta > 0.0 {
                    let n: usize = entry.shape.iter().product();
                    let mut base = vec![0f32; n];
                    cortiq_core::quant::dequant_tensor(
                        entry,
                        model.tensor_bytes(want)?,
                        &mut base,
                    )
                    .map_err(|e| anyhow::anyhow!("{want}: dequant: {e}"))?;
                    let mut dd = 0f64;
                    let mut bb = 0f64;
                    for (d, b) in vals.iter().zip(&base) {
                        let diff = (d - b) as f64;
                        dd += diff * diff;
                        bb += (*b as f64) * (*b as f64);
                    }
                    let rel = (dd / bb.max(1e-30)).sqrt() as f32;
                    deltas.push((want.clone(), rel));
                    if rel < min_delta {
                        unchanged += 1;
                        unchanged_bytes += entry.nbytes;
                        break 'shards;
                    }
                }
                let (out_dim, in_dim) = (entry.shape[0], entry.shape[1]);
                // A skill may live in a cheaper encoding than the
                // backbone (--skill-quant, spec §3 per-tensor dtypes):
                // the overlay is small next to the backbone, so its
                // bytes are often better spent halved.
                let q = match skill_quant {
                    Some(sq) => parse_quant(sq)?,
                    None => match dtype_to_quant(entry.dtype) {
                        Some(q) => q,
                        None => {
                            anyhow::bail!("{want}: backbone dtype {:?} unsupported", entry.dtype)
                        }
                    },
                };
                let (dtype, data) = quantize_2d(q, &vals, out_dim, in_dim);
                new_tensors.push(TensorSpec {
                    name: format!("skill.{id}.{want}"),
                    dtype,
                    shape: entry.shape.clone(),
                    data,
                });
                break 'shards;
            }
        }
        if !found {
            skipped.push(format!("{want} (not in donor)"));
        }
    }
    anyhow::ensure!(
        !new_tensors.is_empty(),
        "no matching donor tensors{} — wrong --from{}?",
        if unchanged > 0 { " above --min-delta" } else { "" },
        if unchanged > 0 { " or threshold too high" } else { "" }
    );
    if !skipped.is_empty() {
        for s in &skipped {
            println!("  skipped: {s}");
        }
    }
    if min_delta > 0.0 && !deltas.is_empty() {
        let mut sorted: Vec<f32> = deltas.iter().map(|(_, d)| *d).collect();
        sorted.sort_by(f32::total_cmp);
        println!(
            "delta gate ≥ {min_delta}: kept {} / dropped {} unchanged tensor(s) (−{:.1} MB); \
             rel-delta min {:.4} / median {:.4} / max {:.4}",
            new_tensors.len(),
            unchanged,
            unchanged_bytes as f64 / 1e6,
            sorted.first().unwrap(),
            sorted[sorted.len() / 2],
            sorted.last().unwrap()
        );
    }
    // The registry's layer list reflects what is actually stored.
    let layers: Vec<usize> = layers
        .into_iter()
        .filter(|li| {
            new_tensors.iter().any(|t| {
                t.name
                    .strip_prefix(&format!("skill.{id}.model.layers.{li}."))
                    .is_some()
            })
        })
        .collect();
    let delta_bytes: usize = new_tensors.iter().map(|t| t.data.len()).sum();
    println!(
        "skill '{id}': {} tensors over {} layer(s), +{:.1} MB",
        new_tensors.len(),
        layers.len(),
        delta_bytes as f64 / 1e6
    );

    // ── selection subspace from example prompts (recon-argmin routing) ──
    let selection = match prompts_file {
        Some(pf) => {
            let text = std::fs::read_to_string(pf)?;
            let prompts: Vec<String> =
                text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
            anyhow::ensure!(!prompts.is_empty(), "--prompts {pf}: no prompts");
            let phi_layer = phi_layer.unwrap_or(num_layers * 2 / 3);
            let mut p = Pipeline::from_model(&model, SamplerConfig::default())
                .map_err(|e| anyhow::anyhow!(e))?;
            let sel = fit_selection(&mut p, &prompts, phi_layer, rank);
            println!(
                "selection: φ-layer {phi_layer}, rank {} from {} prompt(s)",
                sel.rank,
                prompts.len()
            );
            Some(sel)
        }
        None => {
            println!("selection: none (no --prompts) — `route`/`--route-dynamic` will skip this skill");
            None
        }
    };

    // ── rebuild the container: old tensors byte-for-byte + the skill ──
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len() + new_tensors.len());
    for t in &model.tensors {
        if t.name.starts_with(&format!("skill.{id}.")) {
            continue; // re-baking the same id replaces its tensors
        }
        tensors.push(TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model.tensor_bytes(&t.name)?.to_vec(),
        });
    }
    tensors.extend(new_tensors);

    let mut header = model.header.clone();
    header.skills.retain(|s| s.id != id);
    header.skills.push(SkillRecord {
        id: id.to_string(),
        name: name.map(String::from),
        layers: layers.clone(),
        selection,
        input_mask_task: None,
        quality: None, // measured below, on the REBUILT file
    });

    let out_path = output.unwrap_or(model_path).to_string();
    let tmp = format!("{out_path}.tmp");
    let mut catalog = model.masks.clone();
    CmfModel::write(
        &tmp,
        &header,
        &tensors,
        if catalog.masks.is_empty() { None } else { Some(&catalog) },
        model.vocab.as_deref(),
    )?;

    // ── DTG-MA sparse bake (Patent 2): derive the task-guided FFN mask
    //    from the skill's own prompts run through the OVERLAID model,
    //    zero the dead neurons in the stored skill tensors and let vbit
    //    water-filling sink them to its bit floor. With the mask active
    //    the zeroed neurons are never read — mathematically identical
    //    to the donor (a dead neuron contributes act·0). ──
    if let Some(keep) = sparse {
        let prompts_text = std::fs::read_to_string(prompts_file.unwrap())?;
        let prompts: Vec<String> = prompts_text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        let probe_model = Arc::new(CmfModel::open(&tmp)?);
        let mut p =
            Pipeline::from_model_with_skill(&probe_model, SamplerConfig::default(), Some(id))
                .map_err(|e| anyhow::anyhow!(e))?;
        let mut mass = vec![vec![0f64; model.arch().intermediate_size]; num_layers];
        for prompt in &prompts {
            let ids = p.tokenizer.encode(prompt);
            for (li, row) in p.probe_ffn_mass(&ids).into_iter().enumerate() {
                for (a, v) in mass[li].iter_mut().zip(row) {
                    *a += v;
                }
            }
        }
        drop(p);
        drop(probe_model);
        let inter = model.arch().intermediate_size;
        let keep_n = ((inter as f32 * keep).ceil() as usize).clamp(1, inter);
        let mut ffn_bits: Vec<Vec<u8>> = Vec::with_capacity(num_layers);
        let mut keep_sets: Vec<Vec<bool>> = Vec::with_capacity(num_layers);
        for row in &mass {
            let mut order: Vec<usize> = (0..inter).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let mut alive = vec![false; inter];
            for &n in order.iter().take(keep_n) {
                alive[n] = true;
            }
            let mut bits = vec![0u8; inter.div_ceil(8)];
            for (n, &a) in alive.iter().enumerate() {
                if a {
                    bits[n / 8] |= 1 << (n % 8);
                }
            }
            keep_sets.push(alive);
            ffn_bits.push(bits);
        }
        // Re-encode the skill's FFN tensors with dead neurons zeroed:
        // gate/up rows and down columns. vbit gives zero rows its bit
        // floor, so the dead neurons cost ~3 bits instead of 8.
        let vq = match skill_quant {
            Some(sq) => parse_quant(sq)?,
            None => Quant::Vbit,
        };
        if skill_quant.is_none() && mean_bits.is_none() {
            // Live rows deserve full precision; the mean budget is what
            // water-filling needs so they float to ~8 bits while the
            // zeroed rows sink to the floor.
            crate::convert::set_vbit_mean_bits((3.0 + 5.0 * keep).clamp(3.0, 8.0));
        }
        let mut saved = 0usize;
        for (name, shape, vals) in &ffn_vals {
            let li: usize = name
                .strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse().ok())
                .context("ffn tensor without layer index")?;
            let alive = &keep_sets[li];
            let (rows, cols) = (shape[0], shape[1]);
            let mut z = vals.clone();
            if name.ends_with("down_proj.weight") {
                // [hidden, inter]: the neuron axis is the columns.
                for r in 0..rows {
                    for (c, a) in alive.iter().enumerate() {
                        if !a {
                            z[r * cols + c] = 0.0;
                        }
                    }
                }
            } else {
                // gate/up [inter, hidden]: the neuron axis is the rows.
                for (r, a) in alive.iter().enumerate() {
                    if !a {
                        z[r * cols..(r + 1) * cols].fill(0.0);
                    }
                }
            }
            let (dtype, data) = quantize_2d(vq, &z, rows, cols);
            let skill_name = format!("skill.{id}.{name}");
            if let Some(t) = tensors.iter_mut().find(|t| t.name == skill_name) {
                saved += t.data.len().saturating_sub(data.len());
                t.dtype = dtype;
                t.shape = shape.clone();
                t.data = data;
            }
        }
        let sparsity = 1.0 - keep_n as f32 / inter as f32;
        println!(
            "sparse bake: keep {keep_n}/{inter} neurons/layer (sparsity {:.0}%), −{:.1} MB",
            sparsity * 100.0,
            saved as f64 / 1e6
        );
        // The mask is an ordinary task in the catalog, linked to the
        // skill via input_mask_task — `run --skill` activates it.
        catalog.masks.retain(|m| m.name != id);
        let task_id = catalog.masks.iter().map(|m| m.task_id + 1).max().unwrap_or(1);
        catalog.masks.push(TaskMask {
            task_id,
            name: id.to_string(),
            description: Some(format!("DTG-MA mask of skill '{id}' (keep {keep:.2})")),
            sparsity,
            quality: None,
            ffn_masks: ffn_bits,
            head_masks: vec![vec![0xffu8; nh_bytes(model.arch().num_attention_heads)]; num_layers],
            layer_gates: vec![true; num_layers],
            parent: None,
            priority: MaskPriority::Normal,
            has_hot_pack: false,
        });
        if let Some(rec) = header.skills.iter_mut().find(|s| s.id == id) {
            rec.input_mask_task = Some(id.to_string());
        }
        CmfModel::write(&tmp, &header, &tensors, Some(&catalog), model.vocab.as_deref())?;
    }

    // ── claim-16 quality gate: overlaid vs backbone on held-out text,
    //    measured through the rebuilt file and recorded in the registry.
    //    A sparse skill is measured WITH its mask active — that is how
    //    it runs. ──
    if let Some(qf) = quality_file {
        let text = std::fs::read_to_string(qf)?;
        let probe = Arc::new(CmfModel::open(&tmp)?);
        let backbone = ppl_of(&probe, None, &text, quality_tokens)?;
        let overlaid = if sparse.is_some() {
            let mask = probe.masks.get(id).context("sparse bake lost its mask")?.clone();
            let mut p =
                Pipeline::from_model_with_skill(&probe, SamplerConfig::default(), Some(id))
                    .map_err(|e| anyhow::anyhow!(e))?;
            let mut ids = p.tokenizer.with_bos(p.tokenizer.encode(&text));
            ids.truncate(quality_tokens);
            p.ppl_ids_masked(&ids, &mask)
        } else {
            ppl_of(&probe, Some(id), &text, quality_tokens)?
        };
        println!(
            "quality ({qf}): backbone PPL {backbone:.3} → skill PPL {overlaid:.3} ({:+.1}%)",
            (overlaid / backbone - 1.0) * 100.0
        );
        drop(probe);
        let mut header2 = header.clone();
        if let Some(rec) = header2.skills.iter_mut().find(|s| s.id == id) {
            rec.quality = Some(serde_json::json!({
                "metric": "ppl",
                "backbone": (backbone * 1000.0).round() / 1000.0,
                "overlaid": (overlaid * 1000.0).round() / 1000.0,
                "file": Path::new(qf).file_name().map(|f| f.to_string_lossy().into_owned()),
                "tokens": quality_tokens,
                "masked": sparse.is_some(),
            }));
        }
        CmfModel::write(
            &tmp,
            &header2,
            &tensors,
            if catalog.masks.is_empty() { None } else { Some(&catalog) },
            model.vocab.as_deref(),
        )?;
    }

    // Verify before replacing anything.
    let check = CmfModel::open(&tmp)?;
    anyhow::ensure!(
        check.skill_tensors(id).count() > 0,
        "rebuilt file lost the skill tensors — refusing"
    );
    drop(check);
    drop(model);
    std::fs::rename(&tmp, &out_path)?;
    println!("✓ wrote {out_path}");
    Ok(())
}

pub fn run_skill_list(model_path: &str) -> anyhow::Result<()> {
    let model = CmfModel::open(model_path)?;
    if model.header.skills.is_empty() {
        println!("no skills — a flat backbone");
        return Ok(());
    }
    println!("{} skill(s):", model.header.skills.len());
    for s in &model.header.skills {
        let bytes: u64 = model
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&format!("skill.{}.", s.id)))
            .map(|t| t.nbytes)
            .sum();
        let routable = if s.selection.is_some() { "routable" } else { "no selection" };
        println!(
            "  {:<10} {:<24} {} tensor(s), {:.1} MB, layers {:?}, {}",
            s.id,
            s.name.as_deref().unwrap_or("—"),
            model.skill_tensors(&s.id).count(),
            bytes as f64 / 1e6,
            s.layers,
            routable
        );
        if let Some(q) = &s.quality {
            println!("      quality: {q}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_specs() {
        assert_eq!(parse_layers("all", 4).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_layers("1-2", 4).unwrap(), vec![1, 2]);
        assert_eq!(parse_layers("0,3", 4).unwrap(), vec![0, 3]);
        assert!(parse_layers("2-9", 4).is_err());
        assert!(parse_layers("9", 4).is_err());
        assert!(parse_layers("", 4).is_err());
    }

    #[test]
    fn families_parse() {
        assert!(Families::parse("ffn").is_ok());
        assert!(Families::parse("attn").is_ok());
        assert!(Families::parse("all").is_ok());
        assert!(Families::parse("norms").is_err());
        assert_eq!(Families::Ffn.suffixes().len(), 3);
        assert_eq!(Families::All.suffixes().len(), 7);
    }
}
