//! In-place recoding of an existing `.cmf` between quantization layouts.
//!
//! The motivating case is `q4_tiled → q4tp`: the nibbles are unchanged in
//! meaning, only the per-tile scale is re-expressed as a rung on a per-row
//! ladder, so a published model can shed ~7% of its bytes without anyone
//! re-downloading the original checkpoint. That matters — the checkpoints
//! behind the published CMF files are tens of gigabytes, and re-converting
//! them is the expensive path this command exists to avoid.
//!
//! The second case is `q2tp-draft`: the MTP draft stack's expert gate/up
//! planes to q2tp — the layout a fresh conversion produces since the
//! converter's q2tp profile covered them. These weights only ever draft
//! (a trunk pass verifies every token), so the grid change is a fidelity
//! trade confined to acceptance rate, never to output. The trunk's own
//! experts are deliberately NOT eligible: recoding them 4→2 bit without
//! the original checkpoint would degrade the model itself.
//!
//! Everything the recoder cannot improve is copied verbatim, so the output
//! is byte-identical outside the tensors it deliberately rewrites. With
//! `--in-place` the rewrite happens inside the source file itself (new
//! payload ≤ old slot; the slack goes dark) — for machines whose disk
//! cannot hold the model twice.

use anyhow::{Context, bail};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_core::quant::{GROUP_SIZE, dequant_tensor};
use cortiq_core::types::TensorDtype;
use std::sync::Arc;

use crate::convert::{
    Quant, encode_q2tp, encode_q4tp, encode_q8_row, parse_quant, q2tp_expert_gate_or_up, quantize_2d,
};

enum Mode {
    /// Whole-file scale re-expression, lossless in the 4-bit grid.
    FullQ4tp,
    /// REAL quantization of a float container (f32/f16/bf16 2-D weight
    /// tensors → q4tp; the value grid changes). Embeddings, the head, the
    /// tiny routing/gate tensors and 1-D tensors stay as they are — the
    /// Embryo trainer's f32 export is the source this exists for.
    QuantizeQ4tp,
    DenseQ4tp,
    /// Explicit, experimental Embryo tied vocabulary profile. The trunk and
    /// hierarchical cluster head remain untouched; quality needs a new gate.
    EmbryoQ8Head,
    /// Draft-only expert inputs to the 2-bit grid.
    DraftQ2tp,
    /// The draft's whole expert bodies — inputs AND the down plane. Still
    /// draft-only for the grid, but EXPERIMENTAL and measured harmful: on
    /// the release the drafts collapsed to 13% acceptance and the host
    /// q2 path corrupts on down-shaped tensors — do not run this against
    /// a file you cannot re-download until both are resolved.
    DraftQ2tpFull,
}

/// Tensors of the Embryo-O1 operators (`docs/EMBRYO_BOUNDED_ANCHOR.md`,
/// plan §2.5) that never leave f32 whatever their shape: the anchor's
/// trained sinks, the GDN `a`/`b` projections and far-field landmarks.
pub fn embryo_keep_f32(name: &str) -> bool {
    name.contains("in_proj_a")
        || name.contains("in_proj_b")
        || name.contains("sink_")
        || name.contains("landmarks_")
}

/// Which dtypes this mode recodes, and into what.
fn target_dtype(mode: &Mode, name: &str, src: TensorDtype, shape: &[usize]) -> Option<TensorDtype> {
    let two_d = shape.len() == 2 && shape[1] % GROUP_SIZE == 0;
    match mode {
        Mode::EmbryoQ8Head
            if two_d && name == "model.embed_tokens.weight" && src == TensorDtype::F32 =>
        {
            Some(TensorDtype::Q8Row)
        }
        // q4tp reads the same 4-bit grid as q4_tiled/q4_block, so recoding
        // either is lossless in the grid and only re-expresses the scale.
        Mode::FullQ4tp if two_d && matches!(src, TensorDtype::Q4Tiled | TensorDtype::Q4Block) => {
            Some(TensorDtype::Q4TiledP)
        }
        // Embryo-O1 operator tensors that must stay f32 by NAME, not by
        // shape: the bounded anchor's trained sinks (`sink_k`/`sink_v`,
        // 3-D today, but the guarantee must not rest on that), the GDN
        // mixer's decay/write-strength projections (`in_proj_a`/`in_proj_b`,
        // nv rows — below the 64-row floor today, likewise not a contract)
        // and any trained far-field landmarks.
        Mode::QuantizeQ4tp
            if two_d
                && shape[0] >= 64
                && matches!(src, TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16)
                && !name.contains("embed_tokens")
                && !name.contains("lm_head")
                && !name.contains(".desc.")
                && !name.contains("mlp.gate.weight")
                && !name.contains("k_gate")
                // MiMo audio towers: the RVQ codebooks are nearest-neighbour
                // tables (a nudged codeword changes codes) and the speech
                // embeddings are gathers, not matmuls.
                && !name.contains("._codebook.")
                && !name.starts_with("speech_embeddings.")
                && !embryo_keep_f32(name) =>
        {
            Some(TensorDtype::Q4TiledP)
        }
        // The dense trunk's 8-bit matrices at 4 bits (Hy4-flash: attention,
        // indexer query, latent projections, shared experts, the dense FFN):
        // a decode reads half the bytes and the q4tp kernels. The head, the
        // embeddings, the router, the routed experts and Engram stay.
        Mode::DenseQ4tp
            if two_d
                && shape[0] >= 64
                && src == TensorDtype::Q8_2f
                && !name.contains("embed_tokens")
                && !name.contains("lm_head")
                && !name.contains(".experts.")
                && !name.contains("mlp.gate.weight")
                && !name.contains("engram") =>
        {
            Some(TensorDtype::Q4TiledP)
        }
        // The draft's SwiGLU inputs, and only the draft's: `.mtp.` scopes
        // the match to the speculative stack.
        Mode::DraftQ2tp
            if two_d
                && src == TensorDtype::Q4TiledP
                && name.contains(".mtp.")
                && q2tp_expert_gate_or_up(name) =>
        {
            Some(TensorDtype::Q2TiledP)
        }
        // …and with the down plane too, for the residency the 2-bit body
        // buys. Same scope: `.mtp.` only, the trunk never qualifies.
        Mode::DraftQ2tpFull
            if two_d
                && src == TensorDtype::Q4TiledP
                && name.contains(".mtp.")
                && (q2tp_expert_gate_or_up(name)
                    || ((name.contains(".experts.")
                        || name.contains(".shared_experts.")
                        || name.contains(".shared_expert."))
                        && name.ends_with(".w2.weight"))) =>
        {
            Some(TensorDtype::Q2TiledP)
        }
        _ => None,
    }
}

pub fn cmd_requant(
    model_path: &str,
    output: Option<&str>,
    quant: &str,
    in_place: bool,
) -> anyhow::Result<()> {
    let mode = if quant == "embryo-q8-head" {
        Mode::EmbryoQ8Head
    } else if quant == "q4tp-quantize" {
        Mode::QuantizeQ4tp
    } else if quant == "q4tp-dense" {
        Mode::DenseQ4tp
    } else if quant == "q2tp-draft" {
        Mode::DraftQ2tp
    } else if quant == "q2tp-draft-full" {
        Mode::DraftQ2tpFull
    } else if matches!(parse_quant(quant)?, Quant::Q4TiledP) {
        Mode::FullQ4tp
    } else {
        bail!(
            "requant targets q4tp or q2tp-draft (got '{quant}'); other layouts \
             change the value grid, which needs the original checkpoint"
        );
    };
    if !in_place && output.is_none() {
        bail!("either --output or --in-place is required");
    }
    if in_place && matches!(mode, Mode::DraftQ2tpFull) {
        bail!(
            "q2tp-draft-full is experimental and measured harmful (13% acceptance, \
             and the host 2-bit path corrupts on down-shaped tensors); refusing to \
             rewrite a file in place — use --output so the original survives"
        );
    }

    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;

    if matches!(mode, Mode::EmbryoQ8Head) {
        anyhow::ensure!(
            !in_place,
            "embryo-q8-head is experimental: in-place rewriting is forbidden"
        );
        anyhow::ensure!(
            !std::path::Path::new(output.unwrap()).exists(),
            "embryo-q8-head requires a new output path"
        );
        anyhow::ensure!(
            model.header.arch.arch_name == "cortiq_embryo"
                && model.header.arch.tie_word_embeddings
                && model.header.skills.is_empty()
                && model.tensor("lm_head.weight").is_none(),
            "embryo-q8-head requires a bare Embryo with a tied vocabulary and no separate lm_head"
        );
        let e = model
            .tensor("model.embed_tokens.weight")
            .context("missing Embryo tied embedding")?;
        anyhow::ensure!(
            e.shape == [model.header.arch.vocab_size, model.header.arch.hidden_size]
                && e.shape.iter().all(|&n| n > 0)
                && target_dtype(&mode, &e.name, e.dtype, &e.shape).is_some(),
            "embryo-q8-head requires an original F32 [vocab, hidden] embedding, hidden divisible by 32"
        );
    }

    let recode =
        |entry: &cortiq_core::format::TensorEntry, dt: TensorDtype| -> anyhow::Result<Vec<u8>> {
            let (rows, cols) = (entry.shape[0], entry.shape[1]);
            let mut buf = vec![0.0f32; rows * cols];
            dequant_tensor(entry, model.entry_bytes(entry), &mut buf)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("dequantizing '{}'", entry.name))?;
            Ok(match dt {
                TensorDtype::Q8Row => {
                    anyhow::ensure!(buf.iter().all(|v| v.is_finite()), "non-finite embedding");
                    encode_q8_row(&buf, rows, cols)
                }
                TensorDtype::Q2TiledP => encode_q2tp(&buf, rows, cols),
                _ => encode_q4tp(&buf, rows, cols),
            })
        };

    if in_place {
        // The new payload rides into the old slot; the directory entry keeps
        // its offset. Only shard-0 tensors are reachable this way.
        let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::new();
        let (mut before, mut after) = (0u64, 0u64);
        for (i, entry) in model.tensors.iter().enumerate() {
            let Some(dt) = target_dtype(&mode, &entry.name, entry.dtype, &entry.shape) else {
                continue;
            };
            if entry.shard != 0 {
                bail!(
                    "'{}' lives in a sibling shard; in-place needs shard 0",
                    entry.name
                );
            }
            let data = recode(entry, dt)?;
            if data.len() as u64 > entry.nbytes {
                bail!(
                    "'{}' would grow {} → {} bytes; in-place cannot move tensors — \
                     use --output",
                    entry.name,
                    entry.nbytes,
                    data.len()
                );
            }
            before += entry.nbytes;
            after += data.len() as u64;
            patches.push((i, dt, data));
        }
        let n = patches.len();
        drop(model); // release the mmap before rewriting under it
        CmfModel::recode_entries_in_place(model_path, &patches)?;
        println!(
            "requant → {quant} (in place): {n} tensors recoded, \
             {:.2} GB → {:.2} GB payload (file length unchanged; slack goes dark)",
            before as f64 / 1e9,
            after as f64 / 1e9,
        );
        return Ok(());
    }

    let output = output.unwrap();
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
    let (mut recoded, mut copied, mut before, mut after) = (0usize, 0usize, 0u64, 0u64);

    for entry in &model.tensors {
        let src = model.entry_bytes(entry);
        before += src.len() as u64;

        let (dtype, data) = match target_dtype(&mode, &entry.name, entry.dtype, &entry.shape) {
            Some(dt) => {
                recoded += 1;
                (dt, recode(entry, dt)?)
            }
            None => {
                copied += 1;
                (entry.dtype, src.to_vec())
            }
        };
        after += data.len() as u64;
        specs.push(TensorSpec {
            name: entry.name.clone(),
            dtype,
            shape: entry.shape.clone(),
            data,
        });
    }

    CmfModel::write(
        output,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;

    let (a, b) = (before as f64, after as f64);
    println!(
        "requant → {quant}: {recoded} tensors recoded, {copied} copied verbatim\n\
         payload {:.2} GB → {:.2} GB ({:+.1}%)",
        a / 1e9,
        b / 1e9,
        (b / a - 1.0) * 100.0
    );
    let (in_sz, out_sz) = (
        std::fs::metadata(model_path)?.len() as f64 / 1e9,
        std::fs::metadata(output)?.len() as f64 / 1e9,
    );
    println!(
        "file {in_sz:.2} GB → {out_sz:.2} GB ({:+.1}%)",
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}

/// `q4tp-dense` with a calibration text: the dense trunk's q8_2f matrices
/// GPTQ-rounded into q4tp instead of rounded to the nearest level — the
/// same layout and kernels, the nibbles chosen so each matrix's OUTPUT over
/// the calibration rows survives (`gptq::gptq_quantize_q4tp`). The
/// calibration forward is the model's own host stack with the Hessian hook
/// on (`gptq_capture`); `--calib-head` takes the head too. A model whose
/// stack folds no inputs (no captured Hessian) is refused: its matrices
/// would only be rounded, which the plain `q4tp-dense` already does.
///
/// Inkling 2's loop (`inkling2_loop`) reads a local layer's q, k and v from
/// one row in the first pass and from three after it: its projections keep a
/// Hessian each (`gptq_capture::set_alias(false)`), every pass folded in.
pub fn cmd_requant_dense_gptq(model_path: &str, output: &str, calib: &[String], tokens: usize, window: usize, head: bool, lambda: f64) -> anyhow::Result<()> {
    use cortiq_engine::{Pipeline, SamplerConfig};
    use std::collections::{HashMap, HashSet};
    anyhow::ensure!(!calib.is_empty(), "--calib names no text");
    anyhow::ensure!(tokens >= GROUP_SIZE && window >= 2, "--tokens {tokens} / --calib-window {window}: too few");
    // the calibration forward is the host stack (its inputs are what the hook folds): the
    // card stays out until the folds
    // SAFETY: set before any pipeline thread starts
    unsafe {
        std::env::set_var("CMF_GPU", "0");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;
    let mode = Mode::DenseQ4tp;
    // `CMF_REQUANT_HEAD_OUT=<path>`: a second file with the head GPTQ-rounded too, from
    // the same calibration pass (the two against each other, one forward)
    let head_out = std::env::var("CMF_REQUANT_HEAD_OUT").ok().filter(|p| !p.is_empty() && !head);
    let want_head = head || head_out.is_some();
    // the MTP layer's matrices (`mtp.*`: Maxwell) stay as they are: the calibration forward
    // never runs it, and it only drafts
    let is_target = |e: &cortiq_core::format::TensorEntry| {
        (!e.name.starts_with("mtp.") && target_dtype(&mode, &e.name, e.dtype, &e.shape).is_some())
            || (want_head && e.name == "lm_head.weight" && e.dtype == TensorDtype::Q8_2f && e.shape.len() == 2 && e.shape[1] % GROUP_SIZE == 0)
    };
    let names: HashSet<String> = model.tensors.iter().filter(|e| is_target(e)).map(|e| e.name.clone()).collect();
    anyhow::ensure!(!names.is_empty(), "no q8_2f matrix of a dense trunk in {model_path}");

    let mut pipe = Pipeline::from_model(&model, SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    let share = tokens / calib.len();
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for path in calib {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        // a share of the tokens from the head of each text (eight bytes a token is ample)
        let mut cut = text.len().min(share.saturating_mul(8).max(4096));
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut ids = pipe.tokenizer.encode(&text[..cut]);
        ids.truncate(share);
        windows.extend(ids.chunks(window).filter(|w| w.len() >= 2).map(|w| w.to_vec()));
    }
    let total: usize = windows.iter().map(Vec::len).sum();
    anyhow::ensure!(total >= GROUP_SIZE, "the calibration texts gave {total} tokens");
    eprintln!("calibrating {} matrices on {total} tokens ({} windows) …", names.len(), windows.len());
    let t0 = std::time::Instant::now();
    let keep = names.clone();
    cortiq_engine::gptq_capture::set_alias(model.header.arch.arch_name != "inkling2_loop");
    cortiq_engine::gptq_capture::begin_filtered(true, move |n| keep.contains(n));
    for (i, w) in windows.iter().enumerate() {
        if let Err(e) = pipe.ppl_ids(w) {
            let _ = cortiq_engine::gptq_capture::end_shared();
            bail!("calibration window {i}: {e}");
        }
    }
    let cap = cortiq_engine::gptq_capture::end_shared();
    drop(pipe);
    eprintln!("captured {} Hessians ({} aliases, {} invalid) in {:.0} s", cap.hess.len(), cap.alias.len(), cap.broken.len(), t0.elapsed().as_secs_f64());
    anyhow::ensure!(
        !cap.hess.is_empty(),
        "the calibration forward folded no inputs: this model's stack has no capture (plain `--quant q4tp-dense` rounds without one)"
    );

    // the folds' O(n³) algebra on the card when the build has one
    #[cfg(feature = "cuda")]
    // SAFETY: the pipeline is gone; the workers left idle read no environment
    unsafe {
        std::env::set_var("CMF_GPU", "cuda");
    }
    let owner_of: HashMap<&str, &str> = cap.alias.iter().map(|(a, o)| (a.as_str(), o.as_str())).collect();
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, e) in model.tensors.iter().enumerate() {
        if !names.contains(&e.name) || cap.broken.contains(&e.name) {
            continue;
        }
        let owner = owner_of.get(e.name.as_str()).copied().unwrap_or(e.name.as_str());
        if cap.hess.get(owner).is_some_and(|h| h.cols == e.shape[1] && h.h.len() == h.cols * h.cols) {
            groups.entry(owner.to_string()).or_default().push(i);
        }
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut hess = cap.hess;
    let mut done: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut order: Vec<String> = groups.keys().cloned().collect();
    order.sort();
    let t1 = std::time::Instant::now();
    for (gi, owner) in order.iter().enumerate() {
        let acc = hess.remove(owner).unwrap();
        let n = acc.cols;
        let hdiag: Vec<f32> = (0..n).map(|i| acc.h[i * n + i] as f32).collect();
        let (u, dead) = crate::gptq::gptq_fold_operator_par(acc.h, n, lambda, threads);
        for &i in &groups[owner] {
            let e = &model.tensors[i];
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let mut w = vec![0f32; rows * cols];
            dequant_tensor(e, model.entry_bytes(e), &mut w).map_err(anyhow::Error::msg).with_context(|| format!("dequantizing '{}'", e.name))?;
            done.insert(i, crate::gptq::gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, threads));
        }
        if (gi + 1) % 20 == 0 || gi + 1 == order.len() {
            eprintln!("  {} / {} input groups folded and swept, {:.0} s", gi + 1, order.len(), t1.elapsed().as_secs_f64());
        }
    }

    let write = |path: &str, with_head: bool| -> anyhow::Result<()> {
        let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
        let (mut gptq, mut rounded, mut copied) = (0usize, 0usize, 0usize);
        for (i, entry) in model.tensors.iter().enumerate() {
            let is_head = entry.name == "lm_head.weight";
            let (dtype, data) = if let Some(b) = done.get(&i).filter(|_| with_head || !is_head) {
                gptq += 1;
                (TensorDtype::Q4TiledP, b.clone())
            } else if names.contains(&entry.name) && !is_head {
                // a matrix whose input the pass did not see (an invalid capture): rounded
                rounded += 1;
                let (rows, cols) = (entry.shape[0], entry.shape[1]);
                let mut buf = vec![0.0f32; rows * cols];
                dequant_tensor(entry, model.entry_bytes(entry), &mut buf).map_err(anyhow::Error::msg)?;
                (TensorDtype::Q4TiledP, encode_q4tp(&buf, rows, cols))
            } else {
                copied += 1;
                (entry.dtype, model.entry_bytes(entry).to_vec())
            };
            specs.push(TensorSpec { name: entry.name.clone(), dtype, shape: entry.shape.clone(), data });
        }
        CmfModel::write(path, &model.header, &specs, Some(&model.masks), model.vocab.as_deref())?;
        println!(
            "requant → q4tp-dense (GPTQ, {total} calibration tokens{}): {gptq} matrices GPTQ-rounded, {rounded} rounded without a Hessian, {copied} copied verbatim",
            if with_head { ", the head too" } else { "" }
        );
        let (in_sz, out_sz) = (std::fs::metadata(model_path)?.len() as f64 / 1e9, std::fs::metadata(path)?.len() as f64 / 1e9);
        println!("{path}: {in_sz:.2} GB → {out_sz:.2} GB ({:+.1}%)", (out_sz / in_sz - 1.0) * 100.0);
        Ok(())
    };
    write(output, head)?;
    if let Some(p) = &head_out {
        write(p, true)?;
    }
    Ok(())
}

/// `q1 → q1g`: repack the 1-bit tensors of a model whose native scale group
/// is 128 (Bonsai class). Such a `q1` file stores every group scale four
/// times, once per 32-tile; `q1g` keeps the sign bits byte for byte and
/// one scale per group — 1.5 → 1.125 bits/weight with nothing re-quantized.
///
/// Lossless by default: a 128-group whose four tile scales are not
/// byte-identical refuses the whole run (nothing is written) and the
/// counts say where. `--allow-tol` accepts groups that agree within `tol`
/// (relative; they take the f16 nearest the mean of their four scales, the
/// L2-optimal level). Tensors with a group beyond `tol`, or with
/// `cols % 128 != 0`, stay `q1` byte for byte. Everything else in the file
/// is copied verbatim.
pub fn cmd_requant_q1g(
    model_path: &str,
    output: Option<&str>,
    in_place: bool,
    allow_tol: bool,
    tol: f64,
) -> anyhow::Result<()> {
    use cortiq_core::quant::{Q1G_GROUP, repack_q1_to_q1g};
    if !in_place && output.is_none() {
        bail!("either --output or --in-place is required");
    }
    anyhow::ensure!(
        tol.is_finite() && tol >= 0.0,
        "--tol must be a finite, non-negative ratio (got {tol})"
    );
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;

    // Plan every tensor before writing anything: the run is all or nothing.
    let mut plans: Vec<(usize, Vec<u8>)> = Vec::new();
    let (mut groups, mut differing, mut accepted_tol) = (0usize, 0usize, 0usize);
    let mut max_rel = 0f64;
    let mut kept_shape: Vec<&str> = Vec::new();
    let mut kept_tol: Vec<&str> = Vec::new();
    for (i, e) in model.tensors.iter().enumerate() {
        if e.dtype != TensorDtype::Q1 {
            continue;
        }
        if e.shape.len() != 2 || e.shape[1] % Q1G_GROUP != 0 {
            kept_shape.push(&e.name);
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let rp = repack_q1_to_q1g(model.entry_bytes(e), rows, cols, tol);
        groups += rp.groups;
        differing += rp.differing;
        if rp.differing > 0 {
            max_rel = max_rel.max(rp.max_rel);
            println!(
                "  {}: {} of {} groups differ (max rel {:.3e}){}",
                e.name,
                rp.differing,
                rp.groups,
                rp.max_rel,
                if rp.over_tol > 0 {
                    format!(" — {} beyond tol {tol}, stays q1", rp.over_tol)
                } else {
                    String::new()
                }
            );
        }
        if rp.over_tol > 0 {
            kept_tol.push(&e.name);
            continue;
        }
        accepted_tol += rp.differing;
        plans.push((i, rp.data));
    }
    if accepted_tol > 0 && !allow_tol {
        bail!(
            "{accepted_tol} of {groups} q1 128-groups have four tile scales that are not \
             byte-identical (all within --tol {tol}, max rel {max_rel:.3e}: f16 rounding of the \
             native scale). Repacking would move those groups' weights by up to that much; \
             re-run with --allow-tol to accept. Nothing was written."
        );
    }
    if plans.is_empty() {
        bail!(
            "no q1 tensor qualifies for q1g ({} with cols % 128 != 0, {} with a group beyond \
             --tol {tol}); nothing was written",
            kept_shape.len(),
            kept_tol.len()
        );
    }
    let exact = groups - differing;
    let summary = format!(
        "{} q1 tensors → q1g; 128-groups: {groups} total, {exact} exact, {accepted_tol} within \
         tol (max rel {max_rel:.3e}); kept q1: {} (cols % 128), {} (beyond tol)",
        plans.len(),
        kept_shape.len(),
        kept_tol.len()
    );

    if in_place {
        let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::with_capacity(plans.len());
        let (mut before, mut after) = (0u64, 0u64);
        for (i, data) in plans {
            let entry = &model.tensors[i];
            if entry.shard != 0 {
                bail!(
                    "'{}' lives in a sibling shard; in-place needs shard 0",
                    entry.name
                );
            }
            before += entry.nbytes;
            after += data.len() as u64;
            patches.push((i, TensorDtype::Q1G, data));
        }
        drop(model); // release the mmap before rewriting under it
        CmfModel::recode_entries_in_place(model_path, &patches)?;
        println!(
            "requant → q1g (in place): {summary}\n\
             q1 payload {:.1} MB → {:.1} MB (file length unchanged; slack goes dark)",
            before as f64 / 1e6,
            after as f64 / 1e6,
        );
        return Ok(());
    }

    let output = output.unwrap();
    anyhow::ensure!(
        std::path::Path::new(output) != std::path::Path::new(model_path),
        "--output must differ from the source (use --in-place to rewrite it)"
    );
    let mut by_idx: std::collections::HashMap<usize, Vec<u8>> = plans.into_iter().collect();
    let (mut before, mut after) = (0u64, 0u64);
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
    for (i, entry) in model.tensors.iter().enumerate() {
        let (dtype, data) = match by_idx.remove(&i) {
            Some(d) => {
                before += entry.nbytes;
                after += d.len() as u64;
                (TensorDtype::Q1G, d)
            }
            None => (entry.dtype, model.entry_bytes(entry).to_vec()),
        };
        specs.push(TensorSpec {
            name: entry.name.clone(),
            dtype,
            shape: entry.shape.clone(),
            data,
        });
    }
    CmfModel::write(
        output,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;
    let (in_sz, out_sz) = (
        std::fs::metadata(model_path)?.len() as f64,
        std::fs::metadata(output)?.len() as f64,
    );
    println!(
        "requant → q1g: {summary}\n\
         q1 payload {:.1} MB → {:.1} MB ({:+.1}%), file {:.1} MB → {:.1} MB ({:+.1}%)",
        before as f64 / 1e6,
        after as f64 / 1e6,
        (after as f64 / before as f64 - 1.0) * 100.0,
        in_sz / 1e6,
        out_sz / 1e6,
        (out_sz / in_sz - 1.0) * 100.0
    );
    Ok(())
}

/// `q1g → q1`: write every q1g tensor back as q1 tiles (the group scale in
/// each of its four tiles) — exact, so a runtime that predates dtype 20
/// reads the same weights. The file grows back to 1.5 bits/weight, so
/// there is no in-place form.
pub fn cmd_requant_q1(
    model_path: &str,
    output: Option<&str>,
    in_place: bool,
) -> anyhow::Result<()> {
    use cortiq_core::quant::unpack_q1g_to_q1;
    anyhow::ensure!(
        !in_place,
        "q1g → q1 grows every tensor by a third; it needs --output, not --in-place"
    );
    let output = output.context("--output is required")?;
    anyhow::ensure!(
        std::path::Path::new(output) != std::path::Path::new(model_path),
        "--output must differ from the source"
    );
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;
    let mut n = 0usize;
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len());
    for entry in &model.tensors {
        let src = model.entry_bytes(entry);
        let (dtype, data) = if entry.dtype == TensorDtype::Q1G {
            n += 1;
            (
                TensorDtype::Q1,
                unpack_q1g_to_q1(src, entry.shape[0], entry.shape[1]),
            )
        } else {
            (entry.dtype, src.to_vec())
        };
        specs.push(TensorSpec {
            name: entry.name.clone(),
            dtype,
            shape: entry.shape.clone(),
            data,
        });
    }
    anyhow::ensure!(n > 0, "no q1g tensor in {model_path}; nothing was written");
    CmfModel::write(
        output,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )?;
    println!("requant → q1: {n} q1g tensors written back as q1 tiles (exact)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embryo_q8_head_is_explicit_and_preserves_other_tensors() {
        let dir = std::env::temp_dir().join(format!("cmf-embryo-q8-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.cmf");
        let output = dir.join("output.cmf");
        let absent = dir.join("refused.cmf");
        // Owned test outputs only, so a previously interrupted test is safe.
        for p in [&output, &absent] {
            if p.exists() {
                std::fs::remove_file(p).unwrap();
            }
        }
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf", "version": cortiq_core::CMF_VERSION,
            "arch": {"arch_name": "cortiq_embryo", "hidden_size": 32,
                "intermediate_size": 64, "num_layers": 1, "num_attention_heads": 1,
                "num_kv_heads": 1, "head_dim": 32, "vocab_size": 64,
                "layer_types": ["FullAttention"], "tie_word_embeddings": true,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 64},
            "quant_type": "F32"
        }))
        .unwrap();
        let weights: Vec<f32> = (0..64 * 32).map(|i| (i as f32 * 0.17).sin()).collect();
        let raw: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let specs = vec![
            TensorSpec {
                name: "model.embed_tokens.weight".into(),
                dtype: TensorDtype::F32,
                shape: vec![64, 32],
                data: raw.clone(),
            },
            TensorSpec {
                name: "lm_head.clusters.weight".into(),
                dtype: TensorDtype::F32,
                shape: vec![64, 32],
                data: raw.clone(),
            },
        ];
        CmfModel::write(input.to_str().unwrap(), &header, &specs, None, None).unwrap();
        let before = std::fs::read(&input).unwrap();
        assert!(cmd_requant(input.to_str().unwrap(), None, "embryo-q8-head", true).is_err());
        assert_eq!(std::fs::read(&input).unwrap(), before);
        cmd_requant(
            input.to_str().unwrap(),
            Some(output.to_str().unwrap()),
            "embryo-q8-head",
            false,
        )
        .unwrap();
        let result = CmfModel::open(output.to_str().unwrap()).unwrap();
        assert!(result.verify().is_empty());
        let embedding = result.tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(embedding.dtype, TensorDtype::Q8Row);
        let mut decoded = vec![0.0; weights.len()];
        dequant_tensor(embedding, result.entry_bytes(embedding), &mut decoded).unwrap();
        assert!(
            decoded
                .iter()
                .zip(&weights)
                .all(|(a, b)| (a - b).abs() < 0.005)
        );
        let cluster = result.tensor("lm_head.clusters.weight").unwrap();
        assert_eq!(cluster.dtype, TensorDtype::F32);
        assert_eq!(result.entry_bytes(cluster), raw.as_slice());
        // Requantizing an already quantized vocabulary is not allowed.
        assert!(
            cmd_requant(
                output.to_str().unwrap(),
                Some(absent.to_str().unwrap()),
                "embryo-q8-head",
                false
            )
            .is_err()
        );
        assert!(!absent.exists());
        // Refuse overwrites even when a caller asks for the same input path.
        assert!(
            cmd_requant(
                input.to_str().unwrap(),
                Some(input.to_str().unwrap()),
                "embryo-q8-head",
                false
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&input).unwrap(), before);
        drop(result);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn q4tp_dense_recodes_only_the_dense_trunk() {
        let q8 = TensorDtype::Q8_2f;
        for name in [
            "model.layers.5.self_attn.o_proj.weight",
            "model.layers.5.self_attn.q_b_proj.weight",
            "model.layers.5.mlp.latent_up.weight",
            "model.layers.5.mlp.shared_experts.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ] {
            assert_eq!(target_dtype(&Mode::DenseQ4tp, name, q8, &[3072, 1536]), Some(TensorDtype::Q4TiledP), "{name}");
        }
        for name in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.5.mlp.gate.weight",
            "model.layers.5.mlp.experts.7.down_proj.weight",
            "model.layers.2.engram.table",
        ] {
            assert!(target_dtype(&Mode::DenseQ4tp, name, q8, &[3072, 1536]).is_none(), "{name}");
        }
        // only q8_2f sources, only matrices of 64 rows and more
        assert!(target_dtype(&Mode::DenseQ4tp, "model.layers.5.self_attn.o_proj.weight", TensorDtype::F32, &[3072, 1536]).is_none());
        assert!(target_dtype(&Mode::DenseQ4tp, "model.layers.5.attn_hc.fn", q8, &[8, 12288]).is_none());
    }

    #[test]
    fn embryo_q8_head_does_not_change_default_policy() {
        for name in [
            "lm_head.weight",
            "lm_head.clusters.weight",
            "model.layers.0.mlp.gate.weight",
        ] {
            assert!(target_dtype(&Mode::EmbryoQ8Head, name, TensorDtype::F32, &[64, 32]).is_none());
        }
        assert!(
            target_dtype(
                &Mode::QuantizeQ4tp,
                "model.embed_tokens.weight",
                TensorDtype::F32,
                &[64, 32]
            )
            .is_none()
        );
        assert!(
            target_dtype(
                &Mode::EmbryoQ8Head,
                "model.embed_tokens.weight",
                TensorDtype::F16,
                &[64, 32]
            )
            .is_none()
        );
    }

    /// End-to-end through the real command: a mini CMF with a draft expert,
    /// a trunk expert and a draft down-projection — only the draft's w1
    /// changes layout, in place, and the file stays verifiable.
    #[test]
    fn in_place_q2tp_draft_recodes_the_file_it_was_pointed_at() {
        let dir = std::env::temp_dir().join(format!("cmf-rq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.cmf");
        let path_s = path.to_str().unwrap();

        let (rows, cols) = (8usize, 256usize);
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.377).sin()) * 0.5)
            .collect();
        // Pin the encoder switches: `convert`'s tests steer the same
        // process-global variables, and the harness runs both on
        // threads of ONE process.
        let _g = crate::convert::tests::env_guard("1", "1");
        let q4 = encode_q4tp(&vals, rows, cols);
        let spec = |name: &str| TensorSpec {
            name: name.into(),
            dtype: TensorDtype::Q4TiledP,
            shape: vec![rows, cols],
            data: q4.clone(),
        };
        let tensors = vec![
            spec("model.mtp.0.ffn.experts.0.w1.weight"),
            spec("model.mtp.0.ffn.experts.0.w2.weight"),
            spec("model.layers.0.mlp.experts.0.gate_proj.weight"),
        ];
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf",
            "version": cortiq_core::CMF_VERSION,
            "arch": {
                "arch_name": "tiny-test",
                "hidden_size": cols,
                "intermediate_size": rows,
                "num_layers": 1,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "layer_types": ["FullAttention"],
                "rms_norm_eps": 1e-6,
                "norm_style": "qwen",
                "hidden_act": "silu",
                "max_position_embeddings": 64
            },
            "quant_type": "Q4_BLOCK"
        }))
        .expect("header json");
        CmfModel::write(path_s, &header, &tensors, None, None).unwrap();

        cmd_requant(path_s, None, "q2tp-draft", true).unwrap();

        let m = CmfModel::open(path_s).unwrap();
        assert!(m.verify().is_empty(), "verify(): {:?}", m.verify());
        let by = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap();
        assert_eq!(
            by("model.mtp.0.ffn.experts.0.w1.weight").dtype,
            TensorDtype::Q2TiledP
        );
        assert_eq!(
            by("model.mtp.0.ffn.experts.0.w2.weight").dtype,
            TensorDtype::Q4TiledP
        );
        let trunk = by("model.layers.0.mlp.experts.0.gate_proj.weight");
        assert_eq!(trunk.dtype, TensorDtype::Q4TiledP);
        assert_eq!(m.entry_bytes(trunk), &q4[..], "trunk bytes changed");

        // The recoded plane must dequant close to the source values — the
        // exact bytes the pure-conversion path would produce.
        let e = by("model.mtp.0.ffn.experts.0.w1.weight");
        let mut back = vec![0.0f32; rows * cols];
        dequant_tensor(e, m.entry_bytes(e), &mut back).unwrap();
        let q2_direct = encode_q2tp(&vals_dequant(&q4, rows, cols), rows, cols);
        assert_eq!(
            m.entry_bytes(e),
            &q2_direct[..],
            "not the conversion-path bytes"
        );
    }

    /// The heap corruption on the stand pointed at 2-bit tensors shaped
    /// like a DOWN projection (rows ≫ cols). Encode one, run the host
    /// matvec and matmat against a dequant reference, and hold the line —
    /// whatever the stand's crash was, this shape must be provably sound.
    #[test]
    fn host_q2_survives_down_shaped_tensors() {
        let (rows, cols) = (4096usize, 2048usize);
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.7311).sin()) * 0.3)
            .collect();
        let _g = crate::convert::tests::env_guard("1", "1");
        let q2 = encode_q2tp(&vals, rows, cols);
        assert_eq!(
            q2.len(),
            cortiq_core::quant::expected_nbytes(TensorDtype::Q2TiledP, &[rows, cols]).unwrap()
        );
        let entry = cortiq_core::format::TensorEntry {
            name: "x".into(),
            dtype: TensorDtype::Q2TiledP,
            shape: vec![rows, cols],
            off: 0,
            nbytes: q2.len() as u64,
            hash: 0,
            shard: 0,
        };
        let mut back = vec![0.0f32; rows * cols];
        dequant_tensor(&entry, &q2, &mut back).unwrap();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).cos()).collect();
        let mut got = vec![0.0f32; rows];
        cortiq_engine::qtensor::q2tp_matvec_for_test(&q2, &x, rows, cols, &mut got);
        for r in (0..rows).step_by(511) {
            let want: f32 = back[r * cols..(r + 1) * cols]
                .iter()
                .zip(&x)
                .map(|(w, xv)| w * xv)
                .sum();
            assert!(
                (got[r] - want).abs() <= want.abs().max(1.0) * 1e-4,
                "row {r}: {} vs {}",
                got[r],
                want
            );
        }
        // The batched path walks the same rows for two columns at once.
        let xs: Vec<f32> = x.iter().chain(x.iter()).copied().collect();
        let mut got2 = vec![0.0f32; 2 * rows];
        cortiq_engine::qtensor::q2tp_matmat_for_test(&q2, &xs, 2, rows, cols, &mut got2);
        for r in (0..rows).step_by(777) {
            assert!((got2[r] - got[r]).abs() <= got[r].abs().max(1.0) * 1e-5);
            assert!((got2[rows + r] - got[r]).abs() <= got[r].abs().max(1.0) * 1e-5);
        }
    }

    /// A q1 payload with ONE scale per 128-group written into all four of
    /// its tiles (the Bonsai shape), deterministic signs.
    fn uniform_q1(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        use cortiq_core::quant::{Q1_TILE, f32_to_f16};
        let tiles = rows * cols / GROUP_SIZE;
        let mut q = vec![0u8; tiles * Q1_TILE];
        for t in 0..tiles {
            let s = 0.01 + 0.0005 * ((t / 4) as u32 ^ seed) as f32 % 0.02;
            let h = f32_to_f16(s).to_le_bytes();
            q[t * Q1_TILE..t * Q1_TILE + 2].copy_from_slice(&h);
            let bits = (t as u32 ^ seed).wrapping_mul(2_654_435_761);
            q[t * Q1_TILE + 2..t * Q1_TILE + 6].copy_from_slice(&bits.to_le_bytes());
        }
        q
    }

    fn dequant_of(m: &CmfModel, name: &str) -> Vec<f32> {
        let e = m.tensor(name).unwrap();
        let mut out = vec![0f32; e.shape.iter().product()];
        dequant_tensor(e, m.entry_bytes(e), &mut out).unwrap();
        out
    }

    /// End to end through the real command: uniform q1 tensors become q1g
    /// with identical decoded weights; a group whose tiles disagree by an
    /// f16 ulp refuses the run unless `--allow-tol`; a group far outside
    /// the tolerance and a `cols % 128 != 0` tensor stay q1 byte for byte;
    /// the in-place path writes the same thing.
    #[test]
    fn q1g_repack_is_lossless_and_refuses_what_it_cannot_keep() {
        use cortiq_core::quant::Q1_TILE;
        let dir = std::env::temp_dir().join(format!("cmf-q1g-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.cmf");
        let out = dir.join("out.cmf");
        let refused = dir.join("refused.cmf");
        for p in [&out, &refused] {
            if p.exists() {
                std::fs::remove_file(p).unwrap();
            }
        }
        let (rows, cols) = (8usize, 256usize);
        let exact = uniform_q1(rows, cols, 7);
        let mut ulp = uniform_q1(rows, cols, 9);
        // group 5, tile 1: one f16 ulp up
        let o = (5 * 4 + 1) * Q1_TILE;
        let h = u16::from_le_bytes([ulp[o], ulp[o + 1]]) + 1;
        ulp[o..o + 2].copy_from_slice(&h.to_le_bytes());
        let mut wild = uniform_q1(rows, cols, 11);
        // group 2, tile 3: twice the scale — not one group at all
        let o = (2 * 4 + 3) * Q1_TILE;
        let s = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([wild[o], wild[o + 1]]));
        wild[o..o + 2].copy_from_slice(&cortiq_core::quant::f32_to_f16(2.0 * s).to_le_bytes());
        let narrow = uniform_q1(rows, 96, 13);
        let norm: Vec<u8> = (0..cols)
            .flat_map(|i| cortiq_core::quant::f32_to_f16(1.0 + i as f32 * 1e-3).to_le_bytes())
            .collect();
        let q1 = |name: &str, c: usize, data: &Vec<u8>| TensorSpec {
            name: name.into(),
            dtype: TensorDtype::Q1,
            shape: vec![rows, c],
            data: data.clone(),
        };
        let tensors = vec![
            q1("model.layers.0.mlp.gate_proj.weight", cols, &exact),
            q1("model.layers.0.mlp.up_proj.weight", cols, &ulp),
            q1("model.layers.0.mlp.down_proj.weight", cols, &wild),
            q1("model.layers.0.self_attn.o_proj.weight", 96, &narrow),
            TensorSpec {
                name: "model.norm.weight".into(),
                dtype: TensorDtype::F16,
                shape: vec![cols],
                data: norm.clone(),
            },
        ];
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf",
            "version": cortiq_core::CMF_VERSION,
            "arch": {
                "arch_name": "tiny-test", "hidden_size": cols, "intermediate_size": rows,
                "num_layers": 1, "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": 4,
                "vocab_size": 16, "layer_types": ["FullAttention"], "rms_norm_eps": 1e-6,
                "norm_style": "qwen", "hidden_act": "silu", "max_position_embeddings": 64
            },
            "quant_type": "VBIT"
        }))
        .expect("header json");
        let (src_s, out_s) = (src.to_str().unwrap(), out.to_str().unwrap());
        CmfModel::write(src_s, &header, &tensors, None, None).unwrap();
        let src_bytes = std::fs::read(&src).unwrap();

        // Strict: the ulp group refuses everything, nothing written.
        let err = cmd_requant_q1g(
            src_s,
            Some(refused.to_str().unwrap()),
            false,
            false,
            2f64.powi(-8),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("--allow-tol") && err.contains("1 of"), "{err}");
        assert!(!refused.exists());
        // --allow-tol only works with q1g … and an output equal to the source is refused.
        assert!(cmd_requant_q1g(src_s, Some(src_s), false, true, 2f64.powi(-8)).is_err());
        assert_eq!(std::fs::read(&src).unwrap(), src_bytes);

        cmd_requant_q1g(src_s, Some(out_s), false, true, 2f64.powi(-8)).unwrap();
        let (s, m) = (
            CmfModel::open(src_s).unwrap(),
            CmfModel::open(out_s).unwrap(),
        );
        assert!(m.verify().is_empty(), "verify(): {:?}", m.verify());
        let dt = |n: &str| m.tensor(n).unwrap().dtype;
        assert_eq!(dt("model.layers.0.mlp.gate_proj.weight"), TensorDtype::Q1G);
        assert_eq!(dt("model.layers.0.mlp.up_proj.weight"), TensorDtype::Q1G);
        assert_eq!(dt("model.layers.0.mlp.down_proj.weight"), TensorDtype::Q1);
        assert_eq!(
            dt("model.layers.0.self_attn.o_proj.weight"),
            TensorDtype::Q1
        );
        // exact: bit for bit; kept tensors: byte for byte.
        let g = "model.layers.0.mlp.gate_proj.weight";
        assert!(
            dequant_of(&s, g)
                .iter()
                .zip(dequant_of(&m, g))
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(
            m.entry_bytes(m.tensor(g).unwrap()).len() * 4,
            s.entry_bytes(s.tensor(g).unwrap()).len() * 3
        );
        for n in [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.norm.weight",
        ] {
            assert_eq!(
                m.entry_bytes(m.tensor(n).unwrap()),
                s.entry_bytes(s.tensor(n).unwrap())
            );
        }
        // the ulp group moved, nothing else in that tensor did.
        let (a, b) = (
            dequant_of(&s, "model.layers.0.mlp.up_proj.weight"),
            dequant_of(&m, "model.layers.0.mlp.up_proj.weight"),
        );
        let moved: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        assert!(
            !moved.is_empty() && moved.iter().all(|&i| i / 128 == 5),
            "{moved:?}"
        );

        // In place: same tensors, same bytes as the --output file.
        let ip = dir.join("inplace.cmf");
        std::fs::write(&ip, &src_bytes).unwrap();
        cmd_requant_q1g(ip.to_str().unwrap(), None, true, true, 2f64.powi(-8)).unwrap();
        let mi = CmfModel::open(ip.to_str().unwrap()).unwrap();
        assert!(mi.verify().is_empty(), "verify(): {:?}", mi.verify());
        for e in &m.tensors {
            let ei = mi.tensor(&e.name).unwrap();
            assert_eq!(ei.dtype, e.dtype, "{}", e.name);
            assert_eq!(mi.entry_bytes(ei), m.entry_bytes(e), "{}", e.name);
        }
        // And back to q1: the exact source bytes wherever the repack was
        // exact; the q1g weights (not the source's) in the ulp tensor.
        let back = dir.join("back.cmf");
        if back.exists() {
            std::fs::remove_file(&back).unwrap();
        }
        assert!(cmd_requant_q1(out_s, None, true).is_err());
        cmd_requant_q1(out_s, Some(back.to_str().unwrap()), false).unwrap();
        let mb = CmfModel::open(back.to_str().unwrap()).unwrap();
        assert!(mb.verify().is_empty(), "verify(): {:?}", mb.verify());
        for e in &s.tensors {
            let eb = mb.tensor(&e.name).unwrap();
            assert_eq!(eb.dtype, e.dtype, "{}", e.name);
            if e.name != "model.layers.0.mlp.up_proj.weight" {
                assert_eq!(mb.entry_bytes(eb), s.entry_bytes(e), "{}", e.name);
            }
        }
        assert_eq!(dequant_of(&mb, "model.layers.0.mlp.up_proj.weight"), b);
        drop((s, m, mi, mb));
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn vals_dequant(q4: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let entry = cortiq_core::format::TensorEntry {
            name: "x".into(),
            dtype: TensorDtype::Q4TiledP,
            shape: vec![rows, cols],
            off: 0,
            nbytes: q4.len() as u64,
            hash: 0,
            shard: 0,
        };
        let mut out = vec![0.0f32; rows * cols];
        dequant_tensor(&entry, q4, &mut out).unwrap();
        out
    }

    /// The 2-bit recode must reach ONLY the draft stack's SwiGLU inputs:
    /// trunk experts came from the original checkpoint and a 4→2 bit hop
    /// through dequant would degrade the model itself, not just the draft.
    #[test]
    fn q2tp_draft_recode_is_scoped_to_the_mtp_stack() {
        let t =
            |name: &str| target_dtype(&Mode::DraftQ2tp, name, TensorDtype::Q4TiledP, &[64, 256]);
        assert_eq!(
            t("model.mtp.0.ffn.experts.42.w1.weight"),
            Some(TensorDtype::Q2TiledP)
        );
        assert_eq!(
            t("model.mtp.2.ffn.experts.7.w3.weight"),
            Some(TensorDtype::Q2TiledP)
        );
        assert_eq!(
            t("model.mtp.1.ffn.shared_experts.w1.weight"),
            Some(TensorDtype::Q2TiledP)
        );
        // the draft's own down/skeleton stay 4-bit
        assert_eq!(t("model.mtp.0.ffn.experts.42.w2.weight"), None);
        assert_eq!(t("model.mtp.0.self_attn.wkv.weight"), None);
        // the trunk is untouchable, expert or not
        assert_eq!(t("model.layers.7.mlp.experts.42.gate_proj.weight"), None);
        assert_eq!(t("model.layers.7.mlp.experts.42.up_proj.weight"), None);
        // already 2-bit, or the wrong rank — nothing to do
        assert_eq!(
            target_dtype(
                &Mode::DraftQ2tp,
                "model.mtp.0.ffn.experts.1.w1.weight",
                TensorDtype::Q2TiledP,
                &[64, 256]
            ),
            None
        );
        assert_eq!(
            target_dtype(
                &Mode::DraftQ2tp,
                "model.mtp.0.ffn.experts.1.w1.weight",
                TensorDtype::Q4TiledP,
                &[64]
            ),
            None
        );
    }
}

/// `requant --quant tiled-q8_2f`: every `q4tp`/`q2tp` tensor dequantized and
/// stored as q8_2f; everything else copied. An evaluation bridge — the CPU
/// prefill of the tiled codecs is scalar (~18 tok/s on a 2B model), the q8_2f
/// one is a GEMM; the q8 rounding is ~0.4 % of a weight against the 2-bit
/// step's 50 %.
pub fn cmd_requant_tiled_q8(model_path: &str, output: &str) -> anyhow::Result<()> {
    use cortiq_core::format::CmfStreamWriter;
    let t0 = std::time::Instant::now();
    let model = CmfModel::open_sharded(model_path)?;
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;
    let avg = model.tensors.iter().map(|e| e.name.len()).sum::<usize>() / model.tensors.len().max(1);
    let mut out = CmfStreamWriter::new(output, CmfStreamWriter::head_reserve_for(model.tensors.len(), avg))?;
    let mut n = 0usize;
    for e in &model.tensors {
        let bytes = model.entry_bytes(e);
        if !matches!(e.dtype, TensorDtype::Q4TiledP | TensorDtype::Q2TiledP) {
            out.push_bounded(&e.name, e.dtype, &e.shape, bytes, 1 << 28)?;
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_tensor(e, bytes, &mut w).map_err(|m| anyhow::anyhow!("{}: {m}", e.name))?;
        let (dt, enc) = quantize_2d(Quant::Q8_2f, &w, rows, cols);
        out.push(&e.name, dt, &e.shape, &enc)?;
        n += 1;
    }
    let mut header = model.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    prov["tiled_expanded"] = serde_json::json!({ "to": "q8_2f", "tensors": n, "source": std::path::Path::new(model_path).file_name().map(|f| f.to_string_lossy().into_owned()) });
    header.provenance = Some(prov);
    out.finish(&header, None, model.vocab.as_deref())?;
    eprintln!("requant tiled-q8_2f: {n} q4tp/q2tp tensors expanded to q8_2f → {output} in {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// `requant --quant blr-q8_2f`: every `blr` tensor expanded to its dense
/// weight `Σ_p (S_U·diag h·diag ℓ)(S_V·diag g)ᵀ` (one GEMM per tensor, rows
/// split over the cores) and stored as q8_2f; every other tensor copied. The
/// result runs on the ordinary dense paths — what the perplexity tables of a
/// `blr` model use, since the `blr` kernels serve decode, not long prefills.
pub fn cmd_requant_blr_q8(model_path: &str, output: &str) -> anyhow::Result<()> {
    use cortiq_core::blr::{blr_f16s, blr_path_offsets, validate_blr};
    use cortiq_core::format::CmfStreamWriter;
    let t0 = std::time::Instant::now();
    let model = CmfModel::open_sharded(model_path)?;
    crate::knowledge::refuse_genome_rewrite(&model, "requant")?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let bit = |b: &[u8], off: usize, i: usize| (b[off + i / 8] >> (i % 8)) & 1 == 1;
    let avg = model.tensors.iter().map(|e| e.name.len()).sum::<usize>() / model.tensors.len().max(1);
    let mut out = CmfStreamWriter::new(output, CmfStreamWriter::head_reserve_for(model.tensors.len(), avg))?;
    let mut n_blr = 0usize;
    for e in &model.tensors {
        let bytes = model.entry_bytes(e);
        if e.dtype != TensorDtype::BinaryLowRank {
            out.push_bounded(&e.name, e.dtype, &e.shape, bytes, 1 << 28)?;
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let h = validate_blr(&e.shape, bytes).map_err(anyhow::Error::msg)?;
        let (r, np) = (h.rank, h.paths);
        let kk = r * np;
        // A = [S_U,p·diag(h_p)·diag(ℓ_p)]_p (rows × P·r), B = [diag(g_p)·S_V,p]_p stacked (P·r × cols)
        let mut a = vec![0f32; rows * kk];
        let mut b = vec![0f32; kk * cols];
        for p in 0..np {
            let o = blr_path_offsets(&h, p);
            let (hv, gv, lv) = (blr_f16s(bytes, o.h, rows), blr_f16s(bytes, o.g, cols), blr_f16s(bytes, o.l, r));
            for i in 0..rows {
                for k in 0..r {
                    let s = if bit(bytes, o.u + i * (r / 8), k) { 1.0 } else { -1.0 };
                    a[i * kk + p * r + k] = s * hv[i] * lv[k];
                }
            }
            for k in 0..r {
                for j in 0..cols {
                    let s = if bit(bytes, o.v + k * (cols / 8), j) { 1.0 } else { -1.0 };
                    b[(p * r + k) * cols + j] = s * gv[j];
                }
            }
        }
        let mut w = vec![0f32; rows * cols];
        let per = rows.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for (ci, chunk) in w.chunks_mut(per * cols).enumerate() {
                let m = chunk.len() / cols;
                let src = &a[ci * per * kk..(ci * per + m) * kk];
                let b = &b;
                s.spawn(move || cortiq_engine::qtensor::sgemm_public(m, cols, kk, 1.0, src, kk, b, cols, false, chunk, cols));
            }
        });
        drop((a, b));
        let (dt, enc) = quantize_2d(Quant::Q8_2f, &w, rows, cols);
        out.push(&e.name, dt, &e.shape, &enc)?;
        n_blr += 1;
    }
    let mut header = model.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    prov["blr_expanded"] = serde_json::json!({ "to": "q8_2f", "tensors": n_blr, "source": std::path::Path::new(model_path).file_name().map(|f| f.to_string_lossy().into_owned()) });
    header.provenance = Some(prov);
    out.finish(&header, None, model.vocab.as_deref())?;
    eprintln!("requant blr-q8_2f: {n_blr} blr tensors expanded to q8_2f → {output} in {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}
