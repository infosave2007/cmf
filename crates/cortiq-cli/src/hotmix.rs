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

// ---- MoE experts: `hot-probe --experts`, `hot-bake --experts` ----------------------------------
//
// A routed expert is a small FFN of its own, and the busiest few per cent of its neurons
// carry most of what it adds. The experts' layout is what the device arena is built
// around, so their matrices are not cut: the hot neurons' gate and up rows are set to an
// exact zero in place (a q2tp group at rung 0, a q4tp nibble at 8), which silences them
// in the cold expert, and one q8_2f tube per expert (`mlp.experts.{e}.*_proj.tube1.weight`)
// carries them from the bf16 checkpoint. The shared expert is left as it is.

/// The routed experts of a file: `(layers with experts, experts a layer, inter)`.
fn expert_layout(m: &CmfModel) -> anyhow::Result<(Vec<usize>, usize, usize)> {
    let nl = m.arch().num_layers;
    let mut layers = Vec::new();
    let (mut ne, mut inter) = (0usize, 0usize);
    for li in 0..nl {
        let Some(g0) = m.tensor(&format!("model.layers.{li}.mlp.experts.0.gate_proj.weight")) else {
            continue;
        };
        let n = (1..).take_while(|e| m.tensor(&format!("model.layers.{li}.mlp.experts.{e}.gate_proj.weight")).is_some()).count() + 1;
        ensure!(g0.shape.len() == 2, "{}: not a matrix", g0.name);
        ensure!(layers.is_empty() || (n == ne && g0.shape[0] == inter), "layer {li}: {n} experts of {} neurons, layer {} has {ne} of {inter}", g0.shape[0], layers[0]);
        (ne, inter) = (n, g0.shape[0]);
        layers.push(li);
    }
    ensure!(!layers.is_empty(), "the file has no routed experts (model.layers.N.mlp.experts.E.*)");
    Ok((layers, ne, inter))
}

/// `cortiq hot-probe --experts`: the statistic per layer is `[expert · inter + neuron]`.
pub fn cmd_hot_probe_experts(model_path: &str, text: &str, out: &str, tokens: usize, window: usize, skip: usize) -> anyhow::Result<()> {
    // SAFETY: set before the engine's backends initialise, on the only thread running.
    // The probe reads the host path: the device frames and the fused expert kernels
    // do not report activations.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::set_var("CMF_QWEN_DEVICE", "0");
    }
    let m = Arc::new(CmfModel::open_sharded(model_path)?);
    let (layers, ne, inter) = expert_layout(&m)?;
    let nl = m.arch().num_layers;
    let mut p = Pipeline::from_model(&m, SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = p.tokenizer.encode(&std::fs::read_to_string(text).with_context(|| format!("reading {text}"))?);
    ensure!(ids.len() > skip + 2, "{text}: {} tokens, --skip {skip} leaves none", ids.len());
    let ids = &ids[skip..(skip + tokens).min(ids.len())];
    let window = window.max(2);
    let mut acc = vec![vec![0f64; ne * inter]; nl];
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for w in ids.chunks(window) {
        if w.len() < 2 {
            continue;
        }
        let rows = p.probe_moe_mass(w).map_err(|e| anyhow::anyhow!("{e}"))?;
        for (a, r) in acc.iter_mut().zip(rows) {
            for (x, v) in a.iter_mut().zip(r) {
                *x += v;
            }
        }
        done += w.len();
        eprintln!("hot-probe: {done}/{} tokens, {:.0} s", ids.len(), t0.elapsed().as_secs_f64());
    }
    write_stat(out, &acc)?;
    // Per expert: the share of its energy in its top 5 % of neurons, over the experts
    // the text reached.
    let (mut share, mut reached) = (0f64, 0usize);
    for &li in &layers {
        for e in 0..ne {
            let row = &acc[li][e * inter..(e + 1) * inter];
            if row.iter().any(|&v| v > 0.0) {
                share += top_share(row, 0.05);
                reached += 1;
            }
        }
    }
    println!(
        "hot-probe: Σ(w·act)² over {done} tokens, {} MoE layers x {ne} experts x {inter} neurons -> {out}\n  {reached} of {} experts reached; in a reached expert its top 5 % of neurons hold {:.1} % of the energy on average",
        layers.len(),
        layers.len() * ne,
        share / reached.max(1) as f64 * 100.0
    );
    Ok(())
}

/// Set a 5-bit rung code (the encoder's `q4tp_put_code` assumes a zeroed field).
fn set_code(codes: &mut [u8], g: usize, code: usize) {
    let bit = g * 5;
    let (b, sh) = (bit / 8, bit % 8);
    codes[b] &= !((0x1Fu16 << sh) as u8);
    if sh > 3 {
        codes[b + 1] &= !((0x1Fu16 >> (8 - sh)) as u8);
    }
    cortiq_core::quant::q4tp_put_code(codes, g, code);
}

/// Zero `rows` of a `[rows_n, cols]` q2tp or q4tp matrix in place: q2tp groups go to
/// rung 0 (the format's exact zero), q4tp nibbles to 8. Other rows are untouched.
pub(crate) fn zero_tiled_rows(dtype: TensorDtype, bytes: &mut [u8], rows_n: usize, cols: usize, rows: &[usize]) -> anyhow::Result<()> {
    use cortiq_core::quant::{GROUP_SIZE, Q2TP_CHUNK, Q4TP_NIB, q2tp_sections, q4tp_sections};
    ensure!(cols % GROUP_SIZE == 0, "{cols} columns are not whole groups of {GROUP_SIZE}");
    let gpr = cols / GROUP_SIZE;
    match dtype {
        TensorDtype::Q2TiledP => {
            let (_, codes_off, stride) = q2tp_sections(rows_n, cols);
            for &r in rows {
                ensure!(r < rows_n, "row {r} of {rows_n}");
                bytes[r * gpr * Q2TP_CHUNK..(r + 1) * gpr * Q2TP_CHUNK].fill(0);
                let codes = &mut bytes[codes_off + r * stride..codes_off + (r + 1) * stride];
                for g in 0..gpr {
                    set_code(codes, g, 0);
                }
            }
        }
        TensorDtype::Q4TiledP => {
            let _ = q4tp_sections(rows_n, cols);
            for &r in rows {
                ensure!(r < rows_n, "row {r} of {rows_n}");
                bytes[r * gpr * Q4TP_NIB..(r + 1) * gpr * Q4TP_NIB].fill(0x88);
            }
        }
        other => bail!("{other:?}: only q2tp and q4tp rows can be zeroed in place"),
    }
    Ok(())
}

/// `cortiq hot-bake --experts`.
pub fn cmd_hot_bake_experts(cold_path: &str, hf: &str, stat_path: &str, output: &str, frac: f64, readout: bool, scratch: Option<&str>) -> anyhow::Result<()> {
    ensure!(!crate::tube::paths_collide(Path::new(output), Path::new(cold_path)), "hot-bake: the output must differ from the input");
    let cold = Arc::new(CmfModel::open_sharded(cold_path)?);
    let hidden = cold.arch().hidden_size;
    let (layers, ne, inter) = expert_layout(&cold)?;
    let (sl, sw, stat) = read_stat(stat_path)?;
    ensure!(sl == cold.arch().num_layers && sw == ne * inter, "{stat_path}: {sl} x {sw}, the model is {} x {ne}·{inter}", cold.arch().num_layers);
    // The device tube kernels take the neurons four at a time.
    let h = ((frac * inter as f64 / 4.0).round() as usize * 4).clamp(4, inter - 4);
    let name = |li: usize, e: usize, part: &str, tube: bool| format!("model.layers.{li}.mlp.experts.{e}.{part}_proj{}.weight", if tube { ".tube1" } else { "" });
    for &li in &layers {
        for part in ["gate", "up"] {
            let t = entry(&cold, &name(li, 0, part, false))?;
            ensure!(matches!(t.dtype, TensorDtype::Q2TiledP | TensorDtype::Q4TiledP), "hot-bake: {} is {:?}, expected q2tp or q4tp", t.name, t.dtype);
            ensure!(t.shape.as_slice() == [inter, hidden], "hot-bake: {} is {:?}", t.name, t.shape);
        }
        ensure!(cold.tensor(&name(li, 0, "gate", true)).is_none(), "hot-bake: layer {li} already has expert tubes");
    }
    let scratch_dir = scratch.map(str::to_string).unwrap_or_else(|| format!("{output}.src"));
    let mut source = crate::moegptq::ExpertSource::open(hf, &scratch_dir)?;
    let avg = cold.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cold.tensors.len().max(1);
    let mut w = CmfStreamWriter::new(output, CmfStreamWriter::head_reserve_for(cold.tensors.len() + 3 * layers.len() * ne, avg + 12))?;
    // `(layer, expert, part)` of a routed expert's gate or up matrix in a cut layer.
    let routed = |n: &str| -> Option<(usize, usize, &str)> {
        let rest = n.strip_prefix("model.layers.")?;
        let (li, rest) = rest.split_once('.')?;
        let li: usize = li.parse().ok()?;
        let rest = rest.strip_prefix("mlp.experts.")?;
        let (e, rest) = rest.split_once('.')?;
        let e: usize = e.parse().ok()?;
        let part = match rest {
            "gate_proj.weight" => "gate",
            "up_proj.weight" => "up",
            _ => return None,
        };
        layers.contains(&li).then_some((li, e, part))
    };
    let t0 = std::time::Instant::now();
    let mut hot: std::collections::HashMap<(usize, usize), Vec<usize>> = std::collections::HashMap::new();
    let (mut covered, mut done) = (0f64, 0usize);
    for t in &cold.tensors {
        let Some((li, e, part)) = routed(&t.name) else {
            w.push_bounded(&t.name, t.dtype, &t.shape, cold.entry_bytes(t), COPY_CHUNK)?;
            continue;
        };
        if e == 0 && part == "gate" {
            // The layer's first expert tensor: its bf16 experts in, the tubes out.
            let mut tubes = Vec::with_capacity(ne);
            source.routed_layer(li, ne, hidden, inter, |e, x| {
                let mut score: Vec<f64> = stat[li * sw + e * inter..li * sw + (e + 1) * inter].iter().map(|&v| v as f64).collect();
                let mut col = vec![0f64; inter];
                for r in 0..hidden {
                    for (c, &v) in col.iter_mut().zip(&x.d[r * inter..(r + 1) * inter]) {
                        *c += (v as f64) * (v as f64);
                    }
                }
                let total: f64 = score.iter().sum();
                if readout || total <= 0.0 {
                    // The read-out weighs the statistic; an expert the probe never reached
                    // is ranked by its read-out alone.
                    for (s, c) in score.iter_mut().zip(&col) {
                        *s = if total > 0.0 { *s * c } else { *c };
                    }
                }
                let idx = pick(&score, h, None);
                let total: f64 = score.iter().sum();
                covered += if total > 0.0 { idx.iter().map(|&j| score[j]).sum::<f64>() / total } else { 0.0 };
                let mut g = Vec::with_capacity(h * hidden);
                let mut u = Vec::with_capacity(h * hidden);
                for &j in &idx {
                    g.extend_from_slice(&x.g[j * hidden..(j + 1) * hidden]);
                    u.extend_from_slice(&x.u[j * hidden..(j + 1) * hidden]);
                }
                let mut d = Vec::with_capacity(hidden * h);
                for r in 0..hidden {
                    d.extend(idx.iter().map(|&j| x.d[r * inter + j]));
                }
                tubes.push((e, idx, cortiq_core::quant::encode_q8_2f(&g, h, hidden), cortiq_core::quant::encode_q8_2f(&u, h, hidden), cortiq_core::quant::encode_q8_2f(&d, hidden, h)));
                Ok(())
            })?;
            for (e, idx, g, u, d) in tubes {
                w.push(&name(li, e, "gate", true), TensorDtype::Q8_2f, &[h, hidden], &g)?;
                w.push(&name(li, e, "up", true), TensorDtype::Q8_2f, &[h, hidden], &u)?;
                w.push(&name(li, e, "down", true), TensorDtype::Q8_2f, &[hidden, h], &d)?;
                hot.insert((li, e), idx);
            }
            done += 1;
            eprintln!("hot-bake: {done}/{} layers, {:.0} s", layers.len(), t0.elapsed().as_secs_f64());
        }
        let idx = hot.get(&(li, e)).with_context(|| format!("{}: expert {e} comes before expert 0 in the file", t.name))?;
        let mut bytes = cold.tensor_bytes(&t.name)?.to_vec();
        zero_tiled_rows(t.dtype, &mut bytes, inter, hidden, idx).with_context(|| t.name.clone())?;
        w.push(&t.name, t.dtype, &t.shape, &bytes)?;
    }
    let mut header = cold.header.clone();
    let mut prov = header.provenance.take().filter(serde_json::Value::is_object).unwrap_or_else(|| json!({}));
    prov["hot_experts"] = json!({
        "width": h,
        "of": inter,
        "experts": ne,
        "layers": layers,
        "readout": readout,
        "stat": Path::new(stat_path).file_name().and_then(|s| s.to_str()),
        "codec": "q8_2f",
        "source": hf,
    });
    header.provenance = Some(prov);
    w.finish(&header, (!cold.masks.masks.is_empty()).then_some(&cold.masks), cold.vocab.as_deref())?;
    drop(source);
    if scratch.is_none() {
        let _ = std::fs::remove_dir_all(&scratch_dir);
    }
    let gb = |p: &str| std::fs::metadata(p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
    println!(
        "hot-bake: {} layers x {ne} experts, {h} of {inter} neurons an expert in q8_2f, holding {:.1} % of the score on average\n  {cold_path} {:.2} GB -> {output} {:.2} GB ({:+.1} %)",
        layers.len(),
        covered / (layers.len() * ne).max(1) as f64 * 100.0,
        gb(cold_path),
        gb(output),
        (gb(output) / gb(cold_path) - 1.0) * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod expert_tests {
    use super::*;
    use cortiq_core::quant::{dequant_q2tp, dequant_q4tp};

    fn tiled(q: crate::convert::Quant, vals: &[f32], rows: usize, cols: usize) -> (TensorDtype, Vec<u8>) {
        quantize_2d(q, vals, rows, cols)
    }

    #[test]
    fn zeroed_rows_read_back_as_zero_and_the_others_are_untouched() {
        let (rows, cols) = (6, 64);
        let vals: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 0.3).collect();
        for q in [crate::convert::Quant::Q2TiledP, crate::convert::Quant::Q4TiledP] {
            let (dt, before) = tiled(q, &vals, rows, cols);
            let mut after = before.clone();
            zero_tiled_rows(dt, &mut after, rows, cols, &[1, 4]).unwrap();
            let mut a = vec![0f32; rows * cols];
            let mut b = vec![0f32; rows * cols];
            match dt {
                TensorDtype::Q2TiledP => {
                    dequant_q2tp(&before, rows, cols, &mut a);
                    dequant_q2tp(&after, rows, cols, &mut b);
                }
                _ => {
                    dequant_q4tp(&before, rows, cols, &mut a);
                    dequant_q4tp(&after, rows, cols, &mut b);
                }
            }
            for r in 0..rows {
                let (x, y) = (&a[r * cols..(r + 1) * cols], &b[r * cols..(r + 1) * cols]);
                if r == 1 || r == 4 {
                    assert!(y.iter().all(|&v| v == 0.0), "{dt:?} row {r} not zero: {:?}", &y[..4]);
                } else {
                    assert_eq!(x, y, "{dt:?} row {r} changed");
                }
            }
        }
    }
}
