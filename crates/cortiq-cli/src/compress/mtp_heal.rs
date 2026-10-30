//! MTP draft trained on its own trunk after a structural compress.
//!
//! The draft predicts the token after next from the trunk's last hidden
//! state; the fold and the heal change that trunk, and the source's draft
//! stops agreeing with it (Qwen3.8 27B → 14B: 59–80 % of drafts accepted on
//! the source, 51–54 % after). Neither a ridge map of its input onto the
//! source's hidden rows nor a least-squares refit of its read-out moved the
//! acceptance (measured, 56/48 and 48/52 %): the draft has to be trained on
//! the compressed trunk's own predictions, the token its verify accepts.
//!
//! The trunk runs once on the trainer: per window its last hidden rows and
//! its next-token top-k. The draft's block is loaded as a one-layer model
//! (its attention and FFN as layer 0, its final norm, the shared head) that
//! starts from the block's input `eh_proj·[enorm(e_{p+1}); hnorm(h_p)]`, and
//! learns the trunk's distribution at `p+1` by KL — self-distillation. The
//! best held-out checkpoint is written back into `model.mtp.*`.

use super::cmfio::{self, Encoding};
use super::data::Data;
use crate::tdata::TdataSplit;
use anyhow::{Context, Result, anyhow, ensure};
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::types::{LayerType, NormStyle, TensorDtype};
use cortiq_engine::gpu_wgpu::nvg_dev::{self as dv, DMat};
use cortiq_engine::nvg_train as nt;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

pub struct Cfg {
    /// Chain steps trained (the runtime drafts 3–5 deep)
    pub chain: usize,
    /// Steps between rebuilds of the chain's inputs from the current weights
    pub refresh: usize,
    pub tokens: usize,
    pub window: usize,
    pub epochs: f64,
    pub lr: f64,
    pub topk: usize,
    pub batch: usize,
    pub eval_every: usize,
    pub patience: usize,
}

pub struct Report {
    pub steps: usize,
    pub best_step: usize,
    pub loss_start: f64,
    pub loss_best: f64,
}

const LAYER: [&str; 11] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// `x / rms(x)` times the gain (`w` or `1 + w`), in place.
fn rms_into(x: &[f32], w: &[f32], style: NormStyle, eps: f64, out: &mut [f32]) {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt() as f32;
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * if matches!(style, NormStyle::Gemma) { 1.0 + g } else { g };
    }
}

/// The draft's block as a one-layer model: its layer as `model.layers.0`,
/// its final norm as `model.norm`, the trunk's embedding and head. `init`
/// supplies the layer's starting weights (the source's own, in their
/// precision, when given).
fn draft_model(student: &CmfModel, init: &CmfModel, path: &Path) -> Result<()> {
    let tmp = path.with_extension("cmf.partial");
    let _ = std::fs::remove_file(&tmp);
    let mut w = CmfStreamWriter::new(&tmp, CmfStreamWriter::head_reserve_for(16, 48))?;
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        let e = student.tensor(name).with_context(|| format!("mtp heal: no {name}"))?;
        w.push_bounded(name, e.dtype, &e.shape, student.entry_bytes(e), 1 << 28)?;
    }
    let e = init.tensor("model.mtp.norm.weight").context("mtp heal: no model.mtp.norm.weight")?;
    w.push("model.norm.weight", e.dtype, &e.shape, init.entry_bytes(e))?;
    let mut inter = 0;
    for t in LAYER {
        let src = format!("model.mtp.layers.0.{t}");
        let e = init.tensor(&src).with_context(|| format!("mtp heal: no {src}"))?;
        if t == "mlp.gate_proj.weight" {
            inter = e.shape[0];
        }
        let name = format!("model.layers.0.{t}");
        // the trainer exports f16/f32 only: a quantized draft starts from
        // its dequantized f16 copy
        if matches!(e.dtype, TensorDtype::F16 | TensorDtype::F32) {
            w.push(&name, e.dtype, &e.shape, init.entry_bytes(e))?;
        } else {
            let v = cmfio::tensor_f32(init, e)?;
            let b: Vec<u8> = v.iter().flat_map(|&x| cortiq_core::quant::f32_to_f16(x).to_le_bytes()).collect();
            w.push(&name, TensorDtype::F16, &e.shape, &b)?;
        }
    }
    let mut h = student.header.clone();
    h.section_hashes = None;
    h.segments.clear();
    h.arch.num_layers = 1;
    h.arch.layer_types = vec![LayerType::FullAttention];
    h.arch.intermediate_size = inter;
    h.arch.mtp = None;
    h.arch.attention_heads_per_layer = None;
    h.arch.kv_heads_per_layer = None;
    h.arch.no_rope_layers = None;
    w.finish(&h, None, student.vocab.as_deref())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// One window: its padded ids, its real length, the trunk's top-k at every
/// position, and per chain step the block's input rows (f16) — step 1 from
/// the trunk's hidden rows, step j > 1 from the draft's own output at step
/// j − 1, the way a speculative round chains the head on itself.
struct Win {
    ids: Vec<u32>,
    len: usize,
    ti: Vec<u32>,
    tp: Vec<f32>,
    h0: Vec<Vec<u16>>,
}

fn to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect()
}

fn from_f16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| cortiq_core::quant::f16_to_f32(x)).collect()
}

/// Train `student`'s draft on its own trunk; write the student with the
/// trained draft to `out`. `init` (the source, before compress) gives the
/// draft's starting weights when it carries them in a higher precision.
///
/// The round chains the draft: its first step reads the trunk's hidden
/// row, every later one the block's own output. Trained on the first step
/// alone (Qwen3.8 14B) it accepted 88–90 % at depth 1 and still 54 % at the
/// runtime's depth 3–5, so every step of the chain is trained: step j's
/// input is rebuilt from the current weights' step j − 1 output every
/// `refresh` steps, its target the trunk's distribution j positions ahead.
pub fn heal(student_path: &Path, init_path: Option<&Path>, data_spec: &str, cfg: &Cfg, out: &Path, threads: usize) -> Result<Report> {
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("mtp heal: no wgpu GPU")?;
    let student = CmfModel::open_sharded(student_path).with_context(|| format!("opening {}", student_path.display()))?;
    super::heal_gpu::usable(&student).map_err(|e| anyhow!("mtp heal: {e}"))?;
    let init = match init_path {
        Some(p) => CmfModel::open_sharded(p).with_context(|| format!("opening {}", p.display()))?,
        None => CmfModel::open_sharded(student_path)?,
    };
    let arch = student.arch().clone();
    let (hs, t, k, nb, chain) = (arch.hidden_size, cfg.window, cfg.topk, cfg.batch.max(1), cfg.chain.max(1));
    for n in ["model.mtp.eh_proj.weight", "model.mtp.enorm.weight", "model.mtp.hnorm.weight"] {
        ensure!(student.tensor(n).is_some(), "mtp heal: {} carries no canonical MTP draft ({n})", student_path.display());
    }
    let pad = student.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);

    // windows: train records cut to `t`, every tenth held out
    let data = Data::load(data_spec)?;
    let mut wins_ids: Vec<Vec<u32>> = Vec::new();
    let mut left = cfg.tokens;
    'records: for r in data.split(TdataSplit::Train) {
        for ch in r.ids.chunks(t) {
            if left == 0 {
                break 'records;
            }
            if ch.len() < 16 + chain {
                continue;
            }
            let ch = &ch[..ch.len().min(left)];
            left -= ch.len();
            wins_ids.push(ch.to_vec());
        }
    }
    ensure!(wins_ids.len() >= 10 * nb, "mtp heal: {} windows in {data_spec}, need {} (every tenth is held out)", wins_ids.len(), 10 * nb);
    let total: usize = wins_ids.iter().map(Vec::len).sum();
    eprintln!("mtp heal: {} windows, {total} tokens of {data_spec}; chain depth {chain}", wins_ids.len());

    let embed = cmfio::tensor_f32(&student, student.tensor("model.embed_tokens.weight").context("mtp heal: no embedding")?)?;
    let enorm = cmfio::tensor_f32(&student, student.tensor("model.mtp.enorm.weight").unwrap())?;
    let hnorm = cmfio::tensor_f32(&student, student.tensor("model.mtp.hnorm.weight").unwrap())?;
    // the block input through the projection the draft will run with
    let eh = DMat::from_f32(hs, 2 * hs, &cmfio::tensor_f32(&student, student.tensor("model.mtp.eh_proj.weight").unwrap())?);
    // step j's input rows: row p reads token p + j and `hid`'s row p (the
    // trunk's for j = 1, the block's own step j − 1 output after)
    let block_in = |ids: &[u32], len: usize, j: usize, hid: &[f32]| -> Vec<u16> {
        let mut cat = vec![0f32; t * 2 * hs];
        for p in 0..len.saturating_sub(j) {
            let tok = ids[p + j] as usize;
            let (ce, ch) = cat[p * 2 * hs..(p + 1) * 2 * hs].split_at_mut(hs);
            rms_into(&embed[tok * hs..(tok + 1) * hs], &enorm, arch.norm_style, arch.rms_norm_eps, ce);
            rms_into(&hid[p * hs..(p + 1) * hs], &hnorm, arch.norm_style, arch.rms_norm_eps, ch);
        }
        to_f16(&dv::mm(&DMat::from_f32(t, 2 * hs, &cat), false, &eh, true).to_f32())
    };

    // ── the trunk once: hidden rows and the next-token top-k per window ──
    let mut wins: Vec<Win> = Vec::with_capacity(wins_ids.len());
    {
        let tnet = nt::Net::load(c, &student, true, false, None).map_err(|e| anyhow!("mtp heal: trunk: {e}"))?;
        let tw = nt::Work::new(c, &tnet, 1, t);
        let t0 = Instant::now();
        let mut done = 0usize;
        for (i, ids) in wins_ids.iter().enumerate() {
            let len = ids.len();
            let mut pids = ids.clone();
            pids.resize(t, pad);
            let (ti, tp, h) = nt::teacher_topk_hidden(c, &tnet, &tw, &pids, k).with_context(|| format!("mtp heal: window {i}: trunk pass failed"))?;
            let h1 = block_in(&pids, len, 1, &h);
            wins.push(Win { ids: pids, len, ti, tp, h0: vec![h1] });
            done += len;
            eprint!("\r  trunk: {done} tokens · {:.0} tok/s   ", done as f64 / t0.elapsed().as_secs_f64().max(1e-9));
        }
        eprintln!();
    }

    // ── the draft as a one-layer student ──
    let dpath = out.with_extension("draft.cmf");
    draft_model(&student, &init, &dpath)?;
    let dm = CmfModel::open_sharded(&dpath)?;
    let snet = nt::Net::load_heal(c, &dm, nt::HealCfg { train_from: 0, ffn_only: false, prefix_cached: false }).map_err(|e| anyhow!("mtp heal: draft: {e}"))?;
    let sw = nt::Work::new(c, &snet, nb, t);
    let rw = nt::Work::new(c, &snet, 1, t);
    // the chain's later steps from the current weights
    let refresh = |wins: &mut Vec<Win>| -> Result<()> {
        let t0 = Instant::now();
        for w in wins.iter_mut() {
            w.h0.truncate(1);
            for j in 2..=chain {
                let o = nt::hidden_out(c, &snet, &rw, &w.ids, Some(&from_f16(&w.h0[j - 2]))).context("mtp heal: chain forward failed")?;
                let x = block_in(&w.ids, w.len, j, &o);
                w.h0.push(x);
            }
        }
        if chain > 1 {
            eprintln!("  chain steps 2..{chain} rebuilt from the current draft ({:.0}s)", t0.elapsed().as_secs_f64());
        }
        Ok(())
    };
    refresh(&mut wins)?;
    let (val, train): (Vec<usize>, Vec<usize>) = (0..wins.len()).partition(|i| i % 10 == 9);
    let batches = |set: &[usize]| -> Vec<Vec<usize>> { set.chunks(nb).filter(|b| b.len() == nb).map(<[usize]>::to_vec).collect() };
    let (tb, vb) = (batches(&train), batches(&val));
    // a batch of step j: rows p with p + j < len, the trunk's top-k at p + j
    let join = |wins: &[Win], b: &[usize], j: usize| -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>, f64) {
        let mut o = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0f64);
        for &i in b {
            let w = &wins[i];
            o.0.extend_from_slice(&w.ids);
            o.1.extend(from_f16(&w.h0[j - 1]));
            let (mut tid, mut tp) = (vec![pad; t * k], vec![0f32; t * k]);
            let rows = w.len.saturating_sub(j);
            for p in 0..rows {
                tid[p * k..(p + 1) * k].copy_from_slice(&w.ti[(p + j) * k..(p + j + 1) * k]);
                tp[p * k..(p + 1) * k].copy_from_slice(&w.tp[(p + j) * k..(p + j + 1) * k]);
            }
            o.2.extend(tid);
            o.3.extend(tp);
            o.4 += rows as f64;
        }
        o
    };
    // held-out KL per chain step; checkpoints by their mean
    let val_loss = |wins: &[Win], net: &nt::Net| -> Result<Vec<f64>> {
        (1..=chain)
            .map(|j| {
                let (mut s, mut n) = (0f64, 0f64);
                for b in &vb {
                    let (ids, h0, tid, tp, tot) = join(wins, b, j);
                    s += nt::loss_sum_h(c, net, &sw, &ids, Some(&h0), &tid, &tp, k, true).context("mtp heal: eval failed")?;
                    n += tot;
                }
                Ok(s / n.max(1e-9))
            })
            .collect()
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" / ");
    let v0 = val_loss(&wins, &snet)?;
    let e0 = mean(&v0);
    eprintln!("  held-out KL to the trunk before training, per chain step: {}", fmt(&v0));
    let jobs: Vec<(usize, usize)> = (0..tb.len()).flat_map(|b| (1..=chain).map(move |j| (b, j))).collect();
    let total_steps = ((cfg.epochs * jobs.len() as f64).ceil() as usize).max(1);
    let warm = (total_steps / 10).max(1);
    let shape = |d: usize| -> f64 {
        if d <= warm {
            d as f64 / warm as f64
        } else {
            let pr = (d - warm) as f64 / (total_steps - warm).max(1) as f64;
            0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * pr.min(1.0)).cos())
        }
    };
    let mut best = (e0, nt::snapshot(c, &snet, false).context("snapshot")?, 0usize);
    let (mut done, mut skipped, mut gscale) = (0usize, 0usize, 4096f32);
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    let t0 = Instant::now();
    'outer: loop {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        for &oi in &order {
            if done >= total_steps {
                break 'outer;
            }
            let (bi, j) = jobs[oi];
            let (ids, h0, tid, tp, tot) = join(&wins, &tb[bi], j);
            let st = nt::step_wh(c, &snet, &sw, &ids, Some(&h0), &tid, &tp, k, gscale, None, Some(tot)).context("mtp heal: step failed")?;
            let gn = nt::grad_norm(c, &snet).unwrap_or(f64::NAN);
            if !st.finite || !gn.is_finite() {
                nt::zero_grads(c, &snet);
                gscale = (gscale / 2.0).max(1.0);
                skipped += 1;
                ensure!(skipped < 20, "mtp heal: 20 non-finite steps");
                continue;
            }
            done += 1;
            let lr = cfg.lr * shape(done);
            let gmul = if gn > 1.0 { (1.0 / gn) as f32 } else { 1.0 };
            nt::adam(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul).context("mtp heal: adam failed")?;
            let sps = t0.elapsed().as_secs_f64() / done as f64;
            eprint!("\r  step {done}/{total_steps}  loss {:.4}  lr {lr:.2e}  {sps:.2} s/step · ~{} left   ", st.loss, crate::fmt_eta(sps * (total_steps - done) as f64));
            if chain > 1 && cfg.refresh > 0 && done % cfg.refresh == 0 && done < total_steps {
                eprintln!();
                refresh(&mut wins)?;
            }
            if done % cfg.eval_every == 0 || done == total_steps {
                let v = val_loss(&wins, &snet)?;
                let e = mean(&v);
                eprintln!("\n  step {done}: held-out KL {e:.5} (per step {}; best {:.5})", fmt(&v), best.0.min(e));
                if e < best.0 {
                    best = (e, nt::snapshot(c, &snet, false).context("snapshot")?, done);
                } else if cfg.patience > 0 && done - best.2 >= cfg.patience * cfg.eval_every && done < total_steps {
                    eprintln!("  no new best in {} evaluations: stopping at step {done} of {total_steps}", cfg.patience);
                    break 'outer;
                }
            }
        }
    }
    drop(embed);
    nt::restore(c, &snet, &best.1);
    let trained: std::collections::HashMap<String, Vec<f32>> = nt::export(c, &snet, &dm)
        .map_err(|e| anyhow!("mtp heal: export: {e}"))?
        .into_iter()
        .map(|(name, bytes)| {
            let v: Vec<f32> = match dm.tensor(&name).map(|e| e.dtype) {
                Some(TensorDtype::F32) => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                _ => bytes.chunks_exact(2).map(|b| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            };
            let back = if name == "model.norm.weight" { "model.mtp.norm.weight".to_string() } else { name.replacen("model.layers.0.", "model.mtp.layers.0.", 1) };
            (back, v)
        })
        .collect();
    drop(snet);
    let _ = std::fs::remove_file(&dpath);
    let prov = json!({
        "data": data_spec, "tokens": total, "windows": wins.len(), "steps": done, "best_step": best.2,
        "lr": cfg.lr, "topk": k, "heldout_kl_start": e0, "heldout_kl_best": best.0,
        "init": init_path.map(|p| p.display().to_string()),
    });
    cmfio::rewrite(
        &student,
        out,
        Encoding::Keep,
        &|name| Ok(trained.get(name).map(|v| (student.tensor(name).map(|e| e.shape.clone()).unwrap_or_default(), v.clone()))),
        &|h| cmfio::provenance_set(h, "mtp_heal", prov.clone()),
        threads,
    )?;
    Ok(Report { steps: done, best_step: best.2, loss_start: e0, loss_best: best.0 })
}
