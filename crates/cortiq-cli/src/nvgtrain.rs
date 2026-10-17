//! `cortiq nvg-train`: NVG Ť — light distillation training of a compressed
//! model's metric against its full-precision teacher, entirely on the GPU
//! (`cortiq_engine::nvg_train`).
//!
//! The teacher pass runs first (its top-k next-token distribution per token
//! is kept on the host, the teacher is then freed); the student trains its
//! binary low-rank metric (h, ℓ, g of every path) and every RMSNorm gain
//! with Adam, a cosine schedule after a short warm-up, global gradient-norm
//! clipping, and the f16 gradient scale halved on a non-finite step. The
//! held-out windows pick the best checkpoint, which is written into a copy
//! of the student file (same layout and sizes: only payload values change).

use crate::gptq::{CalibSource, calib_source_tokens, split_calib_budget};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_engine::nvg_train as nt;
use std::time::Instant;

pub struct NvgTrainOpts<'a> {
    pub student: &'a str,
    pub teacher: &'a str,
    pub calib: &'a [String],
    pub tokens: usize,
    pub window: usize,
    pub batch: usize,
    pub epochs: usize,
    pub lr: f64,
    pub topk: usize,
    pub eval_windows: usize,
    pub eval_every: usize,
    pub clip: f64,
    pub seed: u64,
    pub output: &'a str,
}

pub fn run(o: &NvgTrainOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().ok_or_else(|| anyhow::anyhow!("nvg-train needs a wgpu GPU"))?;
    let tmodel = CmfModel::open_sharded(o.teacher)?;
    let smodel = CmfModel::open_sharded(o.student)?;
    let vocab = tmodel.vocab.as_deref().ok_or_else(|| anyhow::anyhow!("teacher has no embedded tokenizer"))?;
    let tok = cortiq_engine::tokenizer::Tokenizer::from_bytes(vocab).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let srcs: Vec<CalibSource> = o.calib.iter().map(|s| CalibSource::parse(s)).collect::<anyhow::Result<_>>()?;
    let want = o.tokens + o.eval_windows * o.window;
    let budgets = split_calib_budget(&srcs, want);
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for (src, &n) in srcs.iter().zip(&budgets) {
        let ids = calib_source_tokens(&src.path, n, &|t| tok.encode_plain(t))?;
        windows.extend(ids.chunks(o.window).filter(|w| w.len() == o.window).map(<[u32]>::to_vec));
    }
    anyhow::ensure!(windows.len() > o.eval_windows + o.batch, "too few {}-token windows ({})", o.window, windows.len());
    // interleave the sources deterministically, then hold out the tail
    let mut rng = o.seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for i in (1..windows.len()).rev() {
        windows.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    let eval: Vec<Vec<u32>> = windows.split_off(windows.len() - o.eval_windows);
    let eval_n = eval.len() / o.batch * o.batch;
    let train = windows;
    eprintln!(
        "nvg-train: {} training windows + {} held out of {} tokens, batch {}, top-{} teacher",
        train.len(),
        eval_n,
        o.window,
        o.batch,
        o.topk
    );

    // ── teacher pass ──
    let k = o.topk;
    let mut t_ids: Vec<Vec<u32>> = Vec::new();
    let mut t_p: Vec<Vec<f32>> = Vec::new();
    {
        let tnet = nt::Net::load(c, &tmodel, true, false).map_err(|e| anyhow::anyhow!(e))?;
        let tw = nt::Work::new(c, &tnet, o.batch, o.window);
        let all: Vec<&Vec<u32>> = train.iter().chain(eval.iter().take(eval_n)).collect();
        let tt = Instant::now();
        for (bi, chunk) in all.chunks(o.batch).enumerate() {
            if chunk.len() < o.batch {
                break;
            }
            let ids: Vec<u32> = chunk.iter().flat_map(|w| w.iter().copied()).collect();
            let (ti, tp) = nt::teacher_topk(c, &tnet, &tw, &ids, k).ok_or_else(|| anyhow::anyhow!("teacher pass failed"))?;
            t_ids.push(ti);
            t_p.push(tp);
            eprint!("\r  teacher {}/{}  {:.0} tok/s   ", bi + 1, all.len() / o.batch, ((bi + 1) * o.batch * o.window) as f64 / tt.elapsed().as_secs_f64());
        }
        eprintln!();
    }
    let n_train_b = train.len() / o.batch;
    let batch_ids = |b: usize| -> Vec<u32> {
        let src = if b < n_train_b { &train[b * o.batch..(b + 1) * o.batch] } else { &eval[(b - n_train_b) * o.batch..(b - n_train_b + 1) * o.batch] };
        src.iter().flat_map(|w| w.iter().copied()).collect()
    };

    // ── student ──
    let snet = nt::Net::load(c, &smodel, true, true).map_err(|e| anyhow::anyhow!(e))?;
    let sw = nt::Work::new(c, &snet, o.batch, o.window);
    let n_eval_b = eval_n / o.batch;
    let eval_loss = |net: &nt::Net| -> anyhow::Result<f64> {
        let mut s = 0.0;
        for e in 0..n_eval_b {
            let b = n_train_b + e;
            s += nt::loss_only(c, net, &sw, &batch_ids(b), &t_ids[b], &t_p[b], k).ok_or_else(|| anyhow::anyhow!("eval failed"))?;
        }
        Ok(s / n_eval_b.max(1) as f64)
    };
    let e0 = eval_loss(&snet)?;
    eprintln!("held-out KL before training: {e0:.5}");
    let total_steps = n_train_b * o.epochs;
    let warm = (total_steps / 50).max(1);
    let mut best = (e0, nt::snapshot(c, &snet).ok_or_else(|| anyhow::anyhow!("snapshot"))?, 0usize);
    let mut gscale = 4096f32;
    let mut done = 0usize;
    let mut skipped = 0usize;
    let ts = Instant::now();
    let mut order: Vec<usize> = (0..n_train_b).collect();
    for ep in 0..o.epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        for &b in &order {
            let st = nt::step(c, &snet, &sw, &batch_ids(b), &t_ids[b], &t_p[b], k, gscale).ok_or_else(|| anyhow::anyhow!("step failed"))?;
            let gn = nt::grad_norm(c, &snet).unwrap_or(f64::NAN);
            if !st.finite || !gn.is_finite() {
                nt::zero_grads(c, &snet);
                gscale = (gscale / 2.0).max(1.0);
                skipped += 1;
                continue;
            }
            done += 1;
            let lr = if done <= warm {
                o.lr * done as f64 / warm as f64
            } else {
                let pr = (done - warm) as f64 / (total_steps - warm).max(1) as f64;
                o.lr * 0.5 * (1.0 + (std::f64::consts::PI * pr.min(1.0)).cos())
            };
            let gmul = if gn > o.clip { (o.clip / gn) as f32 } else { 1.0 };
            nt::adam(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul).ok_or_else(|| anyhow::anyhow!("adam failed"))?;
            eprint!(
                "\r  epoch {} step {done}/{total_steps}  loss {:.4}  |g| {:.3e}  lr {:.2e}  {:.1} s/step   ",
                ep + 1,
                st.loss,
                gn,
                lr,
                ts.elapsed().as_secs_f64() / done as f64
            );
            if done % o.eval_every == 0 || done == total_steps {
                let e = eval_loss(&snet)?;
                eprintln!("\n  step {done}: held-out KL {e:.5} (best {:.5} at {})", best.0.min(e), if e < best.0 { done } else { best.2 });
                if e < best.0 {
                    best = (e, nt::snapshot(c, &snet).ok_or_else(|| anyhow::anyhow!("snapshot"))?, done);
                }
            }
        }
    }
    eprintln!("\nbest held-out KL {:.5} at step {} (start {e0:.5}); {skipped} non-finite steps skipped", best.0, best.2);
    nt::restore(c, &snet, &best.1);
    let patched = nt::export(c, &snet, &smodel).map_err(|e| anyhow::anyhow!(e))?;
    let map: std::collections::HashMap<String, Vec<u8>> = patched.into_iter().collect();
    let specs: Vec<TensorSpec> = smodel
        .tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: map.get(&e.name).cloned().unwrap_or_else(|| smodel.entry_bytes(e).to_vec()),
        })
        .collect();
    let mut header = smodel.header.clone();
    header.section_hashes = None;
    let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
    prov["nvg_train"] = serde_json::json!({
        "teacher": std::path::Path::new(o.teacher).file_name().map(|f| f.to_string_lossy().into_owned()),
        "tokens": train.len() * o.window * o.epochs,
        "window": o.window,
        "batch": o.batch,
        "lr": o.lr,
        "topk": k,
        "steps": done,
        "best_step": best.2,
        "heldout_kl_start": e0,
        "heldout_kl_best": best.0,
    });
    header.provenance = Some(prov);
    CmfModel::write(o.output, &header, &specs, None, smodel.vocab.as_deref()).map_err(|e| anyhow::anyhow!("write {}: {e}", o.output))?;
    eprintln!("wrote {} in {:.0}s", o.output, t0.elapsed().as_secs_f64());
    Ok(())
}
