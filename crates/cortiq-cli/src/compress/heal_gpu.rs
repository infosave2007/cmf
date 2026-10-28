//! S4 on the device: the KD heal run by the nvg-train kernels (wgpu, the
//! matrix units of an NVIDIA card). The fold's layers from `train_from` up
//! train in full — f32 master weights with Adam, f16 planes refreshed every
//! step — against the source's top-k next-token distribution; the loss
//! covers the scored tokens only (`--loss-on`), the layers below are frozen
//! and the backward stops at the lowest trained one. The held-out records
//! pick the best checkpoint.

use super::data::{Data, LossOn, Stream};
use crate::tdata::TdataSplit;
use anyhow::{Context, Result};
use cortiq_core::CmfModel;
use cortiq_core::format::TensorSpec;
use cortiq_engine::nvg_train as nt;
use std::path::Path;
use std::time::Instant;

/// Whether this process can heal on the device: a wgpu card with the
/// trainer's matrix-unit GEMMs, and a model the trainer's geometry takes.
pub fn usable(model: &CmfModel) -> Result<(), String> {
    cortiq_engine::gpu_wgpu::train_ops::dev().ok_or("no wgpu GPU")?;
    if !cortiq_engine::gpu_wgpu::train_mm::matrix_units() {
        return Err("the card has no cooperative-matrix GEMMs (the trainer needs NVIDIA Vulkan)".into());
    }
    let g = nt::Geom::from_model(model)?;
    // the trainer's layer set: GatedDeltaNet / gated full attention + a
    // dense SwiGLU MLP per layer (the Qwen3.5 family)
    if model.arch().moe.is_some() {
        return Err("a MoE model (the trainer runs dense MLPs)".into());
    }
    for l in 0..g.nl {
        let mixer = if g.gdn_layer[l] { "linear_attn.in_proj_qkv" } else { "self_attn.q_proj" };
        for name in [format!("model.layers.{l}.{mixer}.weight"), format!("model.layers.{l}.mlp.down_proj.weight")] {
            if model.tensor(&name).is_none() {
                return Err(format!("{} layers are not the trainer's (no {name})", model.arch().arch_name));
            }
        }
    }
    Ok(())
}

pub struct Cfg {
    pub train_from: usize,
    pub ffn_only: bool,
    pub epochs: f64,
    pub lr: f64,
    pub window: usize,
    pub batch: usize,
    pub topk: usize,
    pub eval_every: usize,
    pub clip: f64,
    /// Fraction of the steps the learning rate ramps up over.
    pub warm: f64,
    /// Weight of the hard target (the teacher data's own next token) in
    /// the loss: 0 = KL to the source's top-k only, 1 = plain cross-entropy
    /// on the task text (distillation-free healing, Patent 12 §4.7), in
    /// between their mixture `α·CE + (1−α)·KL`.
    pub ce: f64,
    /// The notebook's Genesis schedule: the trained window widens epoch by
    /// epoch from the last layers down (12 → 24 → 40 → all of 64, as
    /// fractions), each layer's rate decays with its distance above the
    /// window's lowest layer (exp(−κ·d), κ = 0.1/(epoch+1)), and the rate
    /// follows (1 − e^(−t/τ))·e^(−t/2T) with τ = T/3.
    pub genesis: bool,
}

pub struct Report {
    pub kl_start: f64,
    pub kl_best: f64,
    pub best_step: usize,
    pub steps: usize,
    pub sec_per_step: f64,
    pub teacher_secs: f64,
    pub trained: usize,
}

/// Windows of `t` tokens cut from one stream of whole records, each with
/// the weight of its rows: row j predicts token j+1, scored when that token
/// is (the last row looks at the next window's first token).
struct Windows {
    ids: Vec<Vec<u32>>,
    rows: Vec<Vec<f32>>,
    /// Each row's next token (its hard target).
    next: Vec<Vec<u32>>,
}

fn windows(s: &Stream, t: usize, pad_id: u32) -> Windows {
    let (mut ids, mut rows, mut next) = (Vec::new(), Vec::new(), Vec::new());
    let n = s.ids.len();
    let mut at = 0;
    while at < n {
        let mut w: Vec<u32> = s.ids[at..(at + t).min(n)].to_vec();
        let mut r: Vec<f32> = (0..w.len()).map(|j| s.weight.get(at + j + 1).copied().unwrap_or(0.0)).collect();
        if r.iter().all(|&x| x == 0.0) {
            at += t;
            continue;
        }
        let mut nx: Vec<u32> = (0..w.len()).map(|j| s.ids.get(at + j + 1).copied().unwrap_or(pad_id)).collect();
        w.resize(t, pad_id);
        r.resize(t, 0.0);
        nx.resize(t, pad_id);
        ids.push(w);
        rows.push(r);
        next.push(nx);
        at += t;
    }
    Windows { ids, rows, next }
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Heal `student` (an f16/f32 fold) against `teacher`, write the result to `out`.
#[allow(clippy::too_many_arguments)]
pub fn heal(data: &Data, on: LossOn, student_path: &Path, teacher_path: &Path, out: &Path, cfg: &Cfg, log: &dyn Fn(&str)) -> Result<Report> {
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("no wgpu GPU")?;
    let student = CmfModel::open_sharded(student_path)?;
    let pad_id = student.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);
    // train records in a fixed shuffled order (categories interleave)
    let mut tr: Vec<_> = data.split(TdataSplit::Train).collect();
    tr.sort_by_key(|r| fnv(&r.rid));
    let trw = windows(&Stream::build(tr.into_iter(), on, pad_id, 0), cfg.window, pad_id);
    let vaw = windows(&Stream::build(data.split(TdataSplit::Val), on, pad_id, 0), cfg.window, pad_id);
    let nb = cfg.batch;
    let n_train_b = trw.ids.len() / nb;
    let n_val_b = (vaw.ids.len() / nb).max(usize::from(!vaw.ids.is_empty()));
    anyhow::ensure!(n_train_b > 0, "heal: {} train windows of {} tokens, fewer than one batch of {nb}", trw.ids.len(), cfg.window);
    // a short val split still gets one batch (its tail windows repeated)
    let batch_of = |w: &Windows, b: usize| -> (Vec<u32>, Vec<f32>) {
        let (mut ids, mut rows) = (Vec::new(), Vec::new());
        for i in 0..nb {
            let j = (b * nb + i) % w.ids.len();
            ids.extend_from_slice(&w.ids[j]);
            rows.extend_from_slice(&w.rows[j]);
        }
        (ids, rows)
    };
    let next_of = |w: &Windows, b: usize| -> Vec<u32> { (0..nb).flat_map(|i| w.next[(b * nb + i) % w.ids.len()].iter().copied()).collect() };
    let alpha = cfg.ce.clamp(0.0, 1.0);
    // slots per row: the source's top-k, plus one for the hard target when mixed
    let k = if alpha >= 1.0 { 1 } else if alpha > 0.0 { cfg.topk + 1 } else { cfg.topk };
    // ── targets per row, scaled by the row's weight: the source's top-k
    // (renormalized) and/or the data's own next token ──
    let t0 = Instant::now();
    let (mut tids, mut tps, mut tots) = (Vec::new(), Vec::new(), Vec::new());
    let all = n_train_b + n_val_b;
    let tnet_model = if alpha < 1.0 { Some(CmfModel::open_sharded(teacher_path)?) } else { None };
    {
        let tnet = match &tnet_model {
            Some(m) => Some(nt::Net::load(c, m, true, false, None).map_err(|e| anyhow::anyhow!("teacher: {e}"))?),
            None => None,
        };
        let tw = tnet.as_ref().map(|n| nt::Work::new(c, n, nb, cfg.window));
        for b in 0..all {
            let (ids, rows) = if b < n_train_b { batch_of(&trw, b) } else { batch_of(&vaw, b - n_train_b) };
            let nx = if b < n_train_b { next_of(&trw, b) } else { next_of(&vaw, b - n_train_b) };
            let (mut ti, mut tp) = match (&tnet, &tw) {
                (Some(n), Some(w)) => nt::teacher_topk(c, n, w, &ids, k).context("teacher pass failed")?,
                _ => (nx.clone(), vec![1f32; nx.len()]),
            };
            for (r, &wr) in rows.iter().enumerate() {
                let (ri, rp) = (&mut ti[r * k..(r + 1) * k], &mut tp[r * k..(r + 1) * k]);
                if alpha > 0.0 && alpha < 1.0 {
                    // the target takes its own slot: its own if the source
                    // ranked it, else the source's weakest entry gives way
                    // (ids stay distinct — the kernel writes per id)
                    let slot = match ri.iter().position(|&x| x == nx[r]) {
                        Some(q) => q,
                        None => {
                            ri[k - 1] = nx[r];
                            rp[k - 1] = 0.0;
                            k - 1
                        }
                    };
                    let s: f32 = rp.iter().sum::<f32>().max(1e-12);
                    for (q, p) in rp.iter_mut().enumerate() {
                        *p = (1.0 - alpha as f32) * *p / s + if q == slot { alpha as f32 } else { 0.0 };
                    }
                }
                for p in rp.iter_mut() {
                    *p *= wr;
                }
            }
            tots.push(rows.iter().map(|&x| x as f64).sum::<f64>());
            tids.push(ti);
            tps.push(tp);
            if tnet.is_some() {
                eprint!("\r  teacher {}/{all}  {:.0} tok/s   ", b + 1, ((b + 1) * nb * cfg.window) as f64 / t0.elapsed().as_secs_f64());
            }
        }
        eprintln!();
    }
    let teacher_secs = t0.elapsed().as_secs_f64();
    // ── frozen prefix: its output once per batch, so its layers never sit
    // on the device next to the trained tail ──
    let nb_all = n_train_b + n_val_b;
    let hs = student.arch().hidden_size;
    // kept in f32 when that fits a third of the host budget, else in f16
    let bytes32 = (nb_all * nb * cfg.window * hs * 4) as f64;
    let room = super::stages::mem_budget() as f64 * 0.3;
    let cached = cfg.train_from > 0 && bytes32 / 2.0 <= room;
    let half = bytes32 > room;
    let (mut h32, mut h16): (Vec<Vec<f32>>, Vec<Vec<u16>>) = (Vec::new(), Vec::new());
    if cached {
        let tp = Instant::now();
        let pnet = nt::Net::load_prefix(c, &student, cfg.train_from).map_err(|e| anyhow::anyhow!("prefix: {e}"))?;
        let pw = nt::Work::new(c, &pnet, nb, cfg.window);
        for b in 0..nb_all {
            let (ids, _) = if b < n_train_b { batch_of(&trw, b) } else { batch_of(&vaw, b - n_train_b) };
            let h = nt::prefix_hidden(c, &pnet, &pw, &ids).context("prefix pass failed")?;
            if half {
                h16.push(h.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect());
            } else {
                h32.push(h);
            }
        }
        log(&format!(
            "S4 (device): frozen layers 0..{} run once ({:.0}s, {:.1} GB of {} activations on the host); the student holds only the tail",
            cfg.train_from,
            tp.elapsed().as_secs_f64(),
            bytes32 / if half { 2e9 } else { 1e9 },
            if half { "f16" } else { "f32" }
        ));
    }
    let h0 = |b: usize| -> Option<std::borrow::Cow<'_, [f32]>> {
        if !cached {
            return None;
        }
        Some(if half {
            std::borrow::Cow::Owned(h16[b].iter().map(|&x| cortiq_core::quant::f16_to_f32(x)).collect())
        } else {
            std::borrow::Cow::Borrowed(h32[b].as_slice())
        })
    };
    // ── student ──
    let snet = nt::Net::load_heal(c, &student, nt::HealCfg { train_from: cfg.train_from, ffn_only: cfg.ffn_only, prefix_cached: cached })
        .map_err(|e| anyhow::anyhow!("student: {e}"))?;
    let sw = nt::Work::new(c, &snet, nb, cfg.window);
    let trained = snet.trained_entries();
    let total_steps = ((cfg.epochs * n_train_b as f64).ceil() as usize).max(1);
    log(&format!(
        "S4 (device): layers {}..{} trained{} ({:.1} M weights), {} train windows + {} val of {} tokens, batch {nb}, top-{k}, {total_steps} steps, lr {}; teacher pass {:.0}s",
        cfg.train_from,
        snet.geom.nl - 1,
        if cfg.ffn_only { " (MLP only)" } else { "" },
        trained as f64 / 1e6,
        trw.ids.len(),
        vaw.ids.len(),
        cfg.window,
        cfg.lr,
        teacher_secs
    ));
    let val = |net: &nt::Net| -> Result<f64> {
        let (mut s, mut t) = (0.0, 0.0);
        for e in 0..n_val_b {
            let b = n_train_b + e;
            let (ids, _) = batch_of(&vaw, e);
            s += nt::loss_sum_h(c, net, &sw, &ids, h0(b).as_deref(), &tids[b], &tps[b], k, true).context("eval failed")?;
            t += tots[b];
        }
        Ok(s / t.max(1e-9))
    };
    let e0 = val(&snet)?;
    eprintln!("  held-out loss before training: {e0:.5}");
    // a long warm-up: Adam's first steps move every trained weight by ~lr
    // at once (all 24 layers of the 0.8B at 1e-5 after 3 warm-up steps
    // raised the held-out KL 0.160 → 0.196 before it fell)
    let warm = ((total_steps as f64 * cfg.warm) as usize).max(1);
    let shape = |d: usize| -> f64 {
        if cfg.genesis {
            let (t, tt) = (d as f64, total_steps as f64);
            return (1.0 - (-t / (tt / 3.0)).exp()) * (-t / (2.0 * tt)).exp();
        }
        if d <= warm {
            d as f64 / warm as f64
        } else {
            let pr = (d - warm) as f64 / (total_steps - warm).max(1) as f64;
            0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * pr.min(1.0)).cos())
        }
    };
    let mut best = (e0, nt::snapshot(c, &snet, false).context("snapshot")?, 0usize);
    let mut gscale = 4096f32;
    let (mut done, mut skipped, mut step_secs) = (0usize, 0usize, 0f64);
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut order: Vec<usize> = (0..n_train_b).collect();
    'outer: loop {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        for &b in &order {
            if done >= total_steps {
                break 'outer;
            }
            if cfg.genesis {
                let epoch = done / n_train_b.max(1);
                let frac = [12.0 / 64.0, 24.0 / 64.0, 40.0 / 64.0, 1.0][epoch.min(3)];
                let nl = snet.geom.nl;
                let from = (nl - ((frac * nl as f64).ceil() as usize).min(nl)).max(cfg.train_from);
                if snet.active_from.swap(from, std::sync::atomic::Ordering::Relaxed) != from && done > 0 {
                    eprintln!("\n  epoch {}: training layers {from}..{}", epoch + 1, nl - 1);
                }
            }
            let ts = Instant::now();
            let (ids, _) = batch_of(&trw, b);
            let st = nt::step_wh(c, &snet, &sw, &ids, h0(b).as_deref(), &tids[b], &tps[b], k, gscale, None, Some(tots[b])).context("step failed")?;
            let gn = nt::grad_norm(c, &snet).unwrap_or(f64::NAN);
            if !st.finite || !gn.is_finite() {
                nt::zero_grads(c, &snet);
                gscale = (gscale / 2.0).max(1.0);
                skipped += 1;
                anyhow::ensure!(skipped < 20, "heal: 20 non-finite steps");
                continue;
            }
            done += 1;
            let lr = cfg.lr * shape(done);
            let gmul = if gn > cfg.clip { (cfg.clip / gn) as f32 } else { 1.0 };
            if cfg.genesis {
                let kappa = 0.1 / (done / n_train_b.max(1) + 1) as f64;
                let from = snet.active_from.load(std::sync::atomic::Ordering::Relaxed);
                let decay = move |l: usize| (-kappa * l.saturating_sub(from) as f64).exp() as f32;
                nt::adam_scaled(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul, Some(&decay)).context("adam failed")?;
            } else {
                nt::adam(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul).context("adam failed")?;
            }
            step_secs += ts.elapsed().as_secs_f64();
            let sps = step_secs / done as f64;
            eprint!(
                "\r  step {done}/{total_steps}  loss {:.4}  |g| {gn:.3e}  lr {lr:.2e}  {sps:.2} s/step · ~{} left   ",
                st.loss,
                crate::fmt_eta(sps * (total_steps - done) as f64 * 1.1)
            );
            if done % cfg.eval_every == 0 || done == total_steps {
                let e = val(&snet)?;
                eprintln!("\n  step {done}: held-out loss {e:.5} (best {:.5})", best.0.min(e));
                if e < best.0 {
                    best = (e, nt::snapshot(c, &snet, false).context("snapshot")?, done);
                }
            }
        }
    }
    log(&format!(
        "S4 (device): held-out {} {e0:.5} → {:.5} (best step {} of {done}, {:.2} s/step, {skipped} non-finite steps skipped)",
        if alpha >= 1.0 { "NLL" } else if alpha > 0.0 { "mixed loss" } else { "KL" },
        best.0,
        best.2,
        step_secs / done.max(1) as f64
    ));
    nt::restore(c, &snet, &best.1);
    let patched: std::collections::HashMap<String, Vec<u8>> = nt::export(c, &snet, &student).map_err(|e| anyhow::anyhow!(e))?.into_iter().collect();
    let specs: Vec<TensorSpec> = student
        .tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: patched.get(&e.name).cloned().unwrap_or_else(|| student.entry_bytes(e).to_vec()),
        })
        .collect();
    let mut header = student.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    prov["heal"] = serde_json::json!({
        "device": "wgpu", "train_from": cfg.train_from, "ffn_only": cfg.ffn_only, "steps": done,
        "best_step": best.2, "lr": cfg.lr, "window": cfg.window, "batch": nb, "topk": k,
        "heldout_loss_start": e0, "heldout_loss_best": best.0, "ce": alpha, "genesis": cfg.genesis,
    });
    header.provenance = Some(prov);
    CmfModel::write(out, &header, &specs, None, student.vocab.as_deref()).map_err(|e| anyhow::anyhow!("write {}: {e}", out.display()))?;
    Ok(Report { kl_start: e0, kl_best: best.0, best_step: best.2, steps: done, sec_per_step: step_secs / done.max(1) as f64, teacher_secs, trained })
}

/// Teacher-forced NLL of the scored tokens of `recs` on the device: the
/// trainer's forward (weights dequantized to f16 planes), each record its
/// own window from position 0 (padding rows unscored), the loss a KL
/// against the one-hot next token — which is exactly its NLL.
pub fn test_nll(path: &Path, recs: &[&super::data::Rec], on: LossOn) -> Result<(f64, usize)> {
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("no wgpu GPU")?;
    let model = CmfModel::open_sharded(path)?;
    let pad_id = model.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);
    let net = nt::Net::load(c, &model, true, false, None).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    anyhow::ensure!(!recs.is_empty(), "no records");
    let t = recs.iter().map(|r| r.ids.len()).max().unwrap_or(1).div_ceil(128) * 128;
    let w = nt::Work::new(c, &net, 1, t);
    let (mut sum, mut cnt) = (0f64, 0usize);
    for r in recs {
        let first = on.first_target(r).max(1);
        let mut ids = r.ids.clone();
        ids.resize(t, pad_id);
        let (mut tid, mut tp) = (vec![0u32; t], vec![0f32; t]);
        for j in 0..r.ids.len().saturating_sub(1) {
            tid[j] = r.ids[j + 1];
            if j + 1 >= first {
                tp[j] = 1.0;
                cnt += 1;
            }
        }
        sum += nt::loss_sum(c, &net, &w, &ids, &tid, &tp, 1, true).with_context(|| format!("{}: forward failed", r.rid))?;
    }
    Ok((sum, cnt))
}

/// Illumination on the device: the trainer's forward of the f16 source,
/// the MLP down-projection input Grams accumulated on the matrix units —
/// the same `HessianAcc` the runtime's capture hook returns.
pub struct DevCapture {
    net: nt::Net,
    work: nt::Work,
    t: usize,
    pad_id: u32,
}

impl DevCapture {
    pub fn new(path: &Path, window: usize) -> Result<Self> {
        let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("no wgpu GPU")?;
        let model = CmfModel::open_sharded(path)?;
        let pad_id = model.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);
        let net = nt::Net::load(c, &model, true, false, None).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let t = window.div_ceil(128) * 128;
        let work = nt::Work::new(c, &net, 1, t);
        Ok(Self { net, work, t, pad_id })
    }

    /// Grams of `layers` over `windows` (each its own context from position 0).
    pub fn capture(&self, windows: &[Vec<u32>], layers: &[usize], label: &str) -> Result<std::collections::BTreeMap<usize, cortiq_engine::gptq_capture::HessianAcc>> {
        let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("no wgpu GPU")?;
        let to = cortiq_engine::gpu_wgpu::train_ops::zeros;
        let width = |l: usize| self.net.layers[l].gate.m();
        let wmax = layers.iter().map(|&l| width(l)).max().unwrap_or(0);
        let kp = self.t.div_ceil(32) * 32;
        let xt = cortiq_engine::gpu_wgpu::train_mm::f16_buf(c, cortiq_engine::gpu_wgpu::train_mm::pad_rows(wmax) * kp);
        let tmp = to(c, wmax * wmax);
        // accumulators within half the free weight budget; more layers take more passes
        let room = cortiq_engine::gpu_wgpu::dsv4_vram_budget().unwrap_or(8 << 30) as usize / 2;
        let mut parts: Vec<Vec<usize>> = vec![Vec::new()];
        let mut used = 0usize;
        for &l in layers {
            let b = 4 * width(l) * width(l);
            if used + b > room && !parts.last().unwrap().is_empty() {
                parts.push(Vec::new());
                used = 0;
            }
            parts.last_mut().unwrap().push(l);
            used += b;
        }
        let total: usize = windows.iter().map(Vec::len).sum();
        let mut out = std::collections::BTreeMap::new();
        for part in &parts {
            let accs: Vec<cortiq_engine::wgpu::Buffer> = part.iter().map(|&l| to(c, width(l) * width(l))).collect();
            let grams: Vec<(usize, &cortiq_engine::wgpu::Buffer)> = part.iter().copied().zip(accs.iter()).collect();
            let t0 = Instant::now();
            let mut done = 0usize;
            for (i, w) in windows.iter().enumerate() {
                anyhow::ensure!(w.len() <= self.t, "{label}: window {i} has {} tokens > {}", w.len(), self.t);
                let mut ids = w.clone();
                ids.resize(self.t, self.pad_id);
                nt::capture_grams(c, &self.net, &self.work, &ids, w.len(), &grams, &xt, &tmp).with_context(|| format!("{label}: window {i}: device forward failed"))?;
                done += w.len();
                if i % 16 == 15 || i + 1 == windows.len() {
                    let rate = done as f64 / t0.elapsed().as_secs_f64().max(1e-9);
                    eprint!(
                        "\r  {label} (device): {done}/{total} tokens · {rate:.0} tok/s · ~{} left   ",
                        crate::fmt_eta((total - done) as f64 / rate.max(1e-9))
                    );
                }
            }
            eprintln!();
            for (&l, acc) in part.iter().zip(&accs) {
                let n = width(l);
                let h: Vec<f64> = cortiq_engine::gpu_wgpu::train_ops::download(c, acc, n * n).context("Gram read-back")?.into_iter().map(f64::from).collect();
                let sumsq: Vec<f64> = (0..n).map(|j| h[j * n + j]).collect();
                anyhow::ensure!(sumsq.iter().any(|&v| v > 0.0), "{label}: layer {l}: captured energy is all zero");
                out.insert(l, cortiq_engine::gptq_capture::HessianAcc { cols: n, h, sumsq, count: total });
            }
        }
        Ok(out)
    }
}
