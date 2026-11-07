//! `cortiq hot-probe` and `cortiq hot-bake` — a low-bit FFN whose busiest neurons stay at 8 bits.
//!
//! A dense FFN neuron is a gate row, an up row and a down column, and a layer does not
//! use its neurons evenly. Patent 2's probe (`Pipeline::probe_ffn_mass_span`) measures
//! that on a text: the energy of `act(gate)·up` per neuron, or how often a neuron is
//! among a position's loudest.
//!
//! `hot-bake` takes two files of one model, a q2tp or q4tp one and an 8-bit one, and cuts
//! every such layer in two: the ordinary triple keeps the cold neurons (their gate and up
//! rows copied bit for bit, the down columns re-cut from the 8-bit file), and a tube
//! (`…_proj.tube1.weight`) holds the hot neurons in q8_2f exactly as the 8-bit file has
//! them. Permuting an FFN's intermediate axis is an identity on the layer, so the model
//! computes the same function; only the hot neurons' codec changes.

use anyhow::{Context, bail, ensure};
use cortiq_core::format::CmfStreamWriter;
use cortiq_core::{CmfModel, TensorDtype};
use cortiq_engine::{Pipeline, SamplerConfig};
use serde_json::json;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::convert::quantize_2d;
use crate::tube::{compact_encoded_basic_columns, compact_encoded_basic_rows, compact_encoded_tiled_rows, quant_of};

const COPY_CHUNK: usize = 16 << 20;

/// A probe's result on disk: `u32 layers, u32 width, f32[layers·width]`.
fn write_stat(path: &str, rows: &[Vec<f64>]) -> anyhow::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("creating {path}"))?);
    f.write_all(&(rows.len() as u32).to_le_bytes())?;
    f.write_all(&(rows.first().map_or(0, Vec::len) as u32).to_le_bytes())?;
    for row in rows {
        for v in row {
            f.write_all(&(*v as f32).to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

fn read_stat(path: &str) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let b = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    ensure!(b.len() >= 8, "{path}: not a probe file");
    let layers = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let width = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    ensure!(b.len() == 8 + layers * width * 4, "{path}: {} bytes, expected {layers} x {width} values", b.len());
    let vals = b[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((layers, width, vals))
}

/// The share of a layer's statistic held by its top `frac` of neurons.
fn top_share(row: &[f64], frac: f64) -> f64 {
    let mut v = row.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = ((v.len() as f64 * frac).round() as usize).clamp(1, v.len());
    let total: f64 = v.iter().sum();
    if total > 0.0 { v[..n].iter().sum::<f64>() / total } else { 0.0 }
}

/// `cortiq hot-probe`.
pub fn cmd_hot_probe(model_path: &str, text: &str, out: &str, tokens: usize, window: usize, skip: usize, stat: &str, topk: usize) -> anyhow::Result<()> {
    match stat {
        // SAFETY: set before the first probe reads them, on the only thread running.
        "energy" => unsafe { std::env::set_var("CMF_FFN_PROBE_SQ", "1") },
        "hits" => unsafe { std::env::set_var("CMF_FFN_PROBE_TOPK", topk.max(1).to_string()) },
        "mass" => {}
        other => bail!("--stat must be energy, hits or mass (got {other:?})"),
    }
    let m = Arc::new(CmfModel::open_sharded(model_path)?);
    let (nl, inter) = (m.arch().num_layers, m.arch().intermediate_size);
    let mut p = Pipeline::from_model(&m, SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = p.tokenizer.encode(&std::fs::read_to_string(text).with_context(|| format!("reading {text}"))?);
    ensure!(ids.len() > skip + 2, "{text}: {} tokens, --skip {skip} leaves none", ids.len());
    let ids = &ids[skip..(skip + tokens).min(ids.len())];
    let window = window.max(2);
    let mut acc = vec![vec![0f64; inter]; nl];
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for w in ids.chunks(window) {
        if w.len() < 2 {
            continue;
        }
        let rows = p.probe_ffn_mass_span(w, w.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        for (a, r) in acc.iter_mut().zip(rows) {
            for (x, v) in a.iter_mut().zip(r) {
                *x += v;
            }
        }
        done += w.len();
        eprintln!("hot-probe: {done}/{} tokens, {:.0} s", ids.len(), t0.elapsed().as_secs_f64());
    }
    ensure!(acc.iter().any(|r| r.iter().any(|&v| v > 0.0)), "hot-probe: the statistic is all zero (a fused device path hid the FFN from the probe)");
    write_stat(out, &acc)?;
    let mean = |f: f64| acc.iter().map(|r| top_share(r, f)).sum::<f64>() / nl.max(1) as f64;
    println!(
        "hot-probe: {stat} over {done} tokens, {nl} layers x {inter} neurons -> {out}\n  share of the statistic in a layer's top 1 % / 5 % / 10 % / 25 % of neurons (mean over layers): {:.1} % / {:.1} % / {:.1} % / {:.1} %",
        mean(0.01) * 100.0,
        mean(0.05) * 100.0,
        mean(0.10) * 100.0,
        mean(0.25) * 100.0
    );
    for li in [0, nl / 4, nl / 2, 3 * nl / 4, nl.saturating_sub(1)] {
        println!("  layer {li:3}: top 5 % holds {:.1} %", top_share(&acc[li], 0.05) * 100.0);
    }
    Ok(())
}

fn entry<'a>(model: &'a CmfModel, name: &str) -> anyhow::Result<&'a cortiq_core::TensorEntry> {
    model.tensors.iter().find(|e| e.name == name).with_context(|| format!("missing tensor {name}"))
}

/// A layer's hot neurons, ascending: the `n` largest of `score`, or `n` at random.
fn pick(score: &[f64], n: usize, random: Option<u64>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    match random {
        Some(seed) => {
            let mut s = seed | 1;
            for i in (1..idx.len()).rev() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                idx.swap(i, (s % (i as u64 + 1)) as usize);
            }
        }
        None => idx.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))),
    }
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

/// `cortiq hot-bake`.
#[allow(clippy::too_many_arguments)]
pub fn cmd_hot_bake(cold_path: &str, hot_path: &str, stat_path: &str, output: &str, frac: f64, pick_mode: &str, readout: bool, seed: u64, take: &[String]) -> anyhow::Result<()> {
    ensure!(!crate::tube::paths_collide(Path::new(output), Path::new(cold_path)) && !crate::tube::paths_collide(Path::new(output), Path::new(hot_path)), "hot-bake: the output must differ from both inputs");
    ensure!(matches!(pick_mode, "top" | "random"), "--pick must be top or random");
    let cold = Arc::new(CmfModel::open_sharded(cold_path)?);
    let hot = Arc::new(CmfModel::open_sharded(hot_path)?);
    let arch = cold.arch().clone();
    let (nl, hidden, inter) = (arch.num_layers, arch.hidden_size, arch.intermediate_size);
    ensure!(
        hot.arch().num_layers == nl && hot.arch().hidden_size == hidden && hot.arch().intermediate_size == inter,
        "hot-bake: the two files are not one model"
    );
    let (sl, sw, stat) = read_stat(stat_path)?;
    ensure!(sl == nl && sw == inter, "{stat_path}: {sl} x {sw}, the model is {nl} x {inter}");
    let n_hot = ((frac * inter as f64 / 32.0).round() as usize * 32).min(inter - 32);
    ensure!((n_hot >= 32 || frac == 0.0) && inter % 32 == 0, "hot-bake: --frac {frac} of {inter} neurons is under one group of 32");
    ensure!(n_hot > 0 || !take.is_empty(), "hot-bake: --frac 0 without --take changes nothing");
    let name = |li: usize, part: &str, tube: bool| format!("model.layers.{li}.mlp.{part}_proj{}.weight", if tube { ".tube1" } else { "" });
    // The layers cut: a 2-bit or 4-bit tiled gate, no tubes yet, an 8-bit triple in the other file.
    let mut layers = Vec::new();
    for li in 0..nl {
        let g = entry(&cold, &name(li, "gate", false))?;
        if n_hot == 0 || !matches!(g.dtype, TensorDtype::Q2TiledP | TensorDtype::Q4TiledP) {
            continue;
        }
        ensure!(cold.tensor(&name(li, "gate", true)).is_none(), "hot-bake: layer {li} already has tubes");
        for part in ["gate", "up", "down"] {
            let h = entry(&hot, &name(li, part, false))?;
            ensure!(h.dtype == TensorDtype::Q8_2f, "hot-bake: {} is {:?} in {hot_path}, expected q8_2f", h.name, h.dtype);
        }
        layers.push(li);
    }
    ensure!(!layers.is_empty() || n_hot == 0, "hot-bake: {cold_path} has no q2tp or q4tp FFN layer");
    let avg = cold.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cold.tensors.len().max(1);
    let mut w = CmfStreamWriter::new(output, CmfStreamWriter::head_reserve_for(cold.tensors.len() + 3 * layers.len(), avg + 6))?;
    let is_cut = |n: &str| -> Option<(usize, bool)> {
        let rest = n.strip_prefix("model.layers.")?;
        let (li, rest) = rest.split_once('.')?;
        let li: usize = li.parse().ok()?;
        if !layers.contains(&li) {
            return None;
        }
        match rest {
            "mlp.gate_proj.weight" => Some((li, true)),
            "mlp.up_proj.weight" | "mlp.down_proj.weight" => Some((li, false)),
            _ => None,
        }
    };
    let t0 = std::time::Instant::now();
    let (mut covered, mut cut, mut taken) = (0f64, 0usize, 0usize);
    for e in &cold.tensors {
        let Some((li, first)) = is_cut(&e.name) else {
            // a tensor asked for by name comes whole from the 8-bit file
            match hot.tensor(&e.name).filter(|_| take.iter().any(|t| e.name.contains(t.as_str()))) {
                Some(h) => {
                    ensure!(h.shape == e.shape, "hot-bake: {} is {:?} in {hot_path} and {:?} in {cold_path}", e.name, h.shape, e.shape);
                    w.push_bounded(&h.name, h.dtype, &h.shape, hot.entry_bytes(h), COPY_CHUNK)?;
                    taken += 1;
                }
                None => w.push_bounded(&e.name, e.dtype, &e.shape, cold.entry_bytes(e), COPY_CHUNK)?,
            }
            continue;
        };
        if !first {
            continue; // written with the layer's gate
        }
        // The layer's score per neuron: the probe's statistic, weighed by the
        // neuron's read-out (its down column's energy) when asked.
        let dn = entry(&hot, &name(li, "down", false))?;
        let mut down = vec![0f32; hidden * inter];
        cortiq_core::quant::dequant_tensor(dn, hot.tensor_bytes(&dn.name)?, &mut down).map_err(|e| anyhow::anyhow!(e))?;
        let mut score: Vec<f64> = stat[li * inter..(li + 1) * inter].iter().map(|&v| v as f64).collect();
        if readout {
            let mut col = vec![0f64; inter];
            for r in 0..hidden {
                for (c, &v) in col.iter_mut().zip(&down[r * inter..(r + 1) * inter]) {
                    *c += (v as f64) * (v as f64);
                }
            }
            for (s, c) in score.iter_mut().zip(col) {
                *s *= c;
            }
        }
        let hot_idx = pick(&score, n_hot, (pick_mode == "random").then_some(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(li as u64)));
        let mut is_hot = vec![false; inter];
        for &j in &hot_idx {
            is_hot[j] = true;
        }
        let cold_idx: Vec<usize> = (0..inter).filter(|&j| !is_hot[j]).collect();
        let total: f64 = score.iter().sum();
        covered += if total > 0.0 { hot_idx.iter().map(|&j| score[j]).sum::<f64>() / total } else { 0.0 };
        let nc = cold_idx.len();
        // Cold gate and up: the file's own rows as they are.
        for part in ["gate", "up"] {
            let ce = entry(&cold, &name(li, part, false))?;
            ensure!(ce.shape.as_slice() == [inter, hidden], "hot-bake: {} is {:?}", ce.name, ce.shape);
            let data = compact_encoded_tiled_rows(ce.dtype, cold.tensor_bytes(&ce.name)?, inter, hidden, &cold_idx)?
                .with_context(|| format!("{}: rows of {:?} cannot be selected", ce.name, ce.dtype))?;
            w.push(&ce.name, ce.dtype, &[nc, hidden], &data)?;
        }
        // Cold down: its columns from the 8-bit file, in the cold file's codec.
        let cd = entry(&cold, &name(li, "down", false))?;
        let q_down = quant_of(cd.dtype).with_context(|| format!("{}: unsupported codec {:?}", cd.name, cd.dtype))?;
        let mut sel = Vec::with_capacity(hidden * nc);
        for r in 0..hidden {
            let row = &down[r * inter..(r + 1) * inter];
            sel.extend(cold_idx.iter().map(|&j| row[j]));
        }
        drop(down);
        let (dt, data) = quantize_2d(q_down, &sel, hidden, nc);
        drop(sel);
        w.push(&cd.name, dt, &[hidden, nc], &data)?;
        // The hot tube: the 8-bit file's own rows and columns.
        for part in ["gate", "up"] {
            let he = entry(&hot, &name(li, part, false))?;
            let data = compact_encoded_basic_rows(he.dtype, hot.tensor_bytes(&he.name)?, inter, hidden, &hot_idx)?
                .with_context(|| format!("{}: rows of {:?} cannot be selected", he.name, he.dtype))?;
            w.push(&name(li, part, true), he.dtype, &[n_hot, hidden], &data)?;
        }
        let data = compact_encoded_basic_columns(dn.dtype, hot.tensor_bytes(&dn.name)?, hidden, inter, &hot_idx)?
            .with_context(|| format!("{}: columns of {:?} cannot be selected", dn.name, dn.dtype))?;
        w.push(&name(li, "down", true), dn.dtype, &[hidden, n_hot], &data)?;
        cut += 1;
        if cut % 8 == 0 || cut == layers.len() {
            eprintln!("hot-bake: {cut}/{} layers, {:.0} s", layers.len(), t0.elapsed().as_secs_f64());
        }
    }
    let mut header = cold.header.clone();
    let mut prov = header.provenance.take().filter(serde_json::Value::is_object).unwrap_or_else(|| json!({}));
    prov["hot_neurons"] = json!({
        "width": n_hot,
        "of": inter,
        "layers": layers,
        "pick": pick_mode,
        "readout": readout,
        "stat": Path::new(stat_path).file_name().and_then(|s| s.to_str()),
        "codec": "q8_2f",
        "taken_whole": take,
    });
    header.provenance = Some(prov);
    w.finish(&header, (!cold.masks.masks.is_empty()).then_some(&cold.masks), cold.vocab.as_deref())?;
    let gb = |p: &str| std::fs::metadata(p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
    println!(
        "hot-bake: {} layers, {n_hot} of {inter} neurons a layer in q8_2f ({pick_mode}), holding {:.1} % of the score on average; {taken} tensors taken whole\n  {cold_path} {:.2} GB -> {output} {:.2} GB ({:+.1} %)",
        layers.len(),
        covered / layers.len().max(1) as f64 * 100.0,
        gb(cold_path),
        gb(output),
        (gb(output) / gb(cold_path) - 1.0) * 100.0
    );
    Ok(())
}
