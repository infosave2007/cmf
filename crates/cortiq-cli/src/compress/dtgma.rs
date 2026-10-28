//! Width by DTG-MA (Patent 2): instead of ranking neurons once by their
//! illumination, a mask σ(m) over every FFN neuron is TRAINED with the task's
//! LM loss through the whole model under a growing L1 pressure. On an NVIDIA
//! card the gates sit in the device trainer (`heal_gpu::learn_masks`, the
//! nvg-train kernels); elsewhere `cortiq_engine::skillbake` (the engine of
//! `cortiq skill bake`, Phase A only) trains them on the host. The trained
//! logits then decide how many neurons each layer keeps and which ones; the
//! fold (refit) and the heal run on that selection as they do on the
//! illumination's.

use super::data::{Data, LossOn, Stream};
use super::Run;
use crate::tdata::TdataSplit;
use anyhow::Result;
use cortiq_core::CmfModel;
use std::path::Path;
use std::sync::Arc;

/// Tokens per mask-training chunk.
const CHUNK: usize = 1024;
/// Held-out chunks the bake checkpoints on.
const HELD: usize = 24;

/// A tuning knob of the device mask training (`CMF_DTGMA_<NAME>`).
fn knob<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(format!("CMF_DTGMA_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Trained mask logits per layer (higher = more needed by the task).
pub fn learn(run: &Run, data: &Data, model_path: &Path, keep_frac: f64) -> Result<Vec<Vec<f32>>> {
    let model = super::stages::open(model_path)?;
    #[cfg(feature = "gpu")]
    if run.a.heal_device != "cpu" {
        match super::heal_gpu::usable(&model) {
            Ok(()) => {
                let cfg = super::heal_gpu::MaskCfg {
                    window: run.a.heal_window,
                    batch: 2,
                    keep_frac,
                    lr: knob("LR", 0.1),
                    l1_init: knob("L1", 0.01),
                    l1_step: knob("L1_STEP", 0.005),
                    l1_grow: knob("L1_GROW", 1.5),
                    eval_every: knob("EVAL", 25),
                    max_steps: knob("MAX_STEPS", 3000),
                    mask_init: knob("INIT", 2.0),
                    tau: 0.5,
                    settle: knob("SETTLE", 0),
                };
                let on = LossOn::parse(&run.a.loss_on)?;
                return Ok(super::heal_gpu::learn_masks(data, on, model_path, &cfg, &|l| run.log(l))?.logits);
            }
            Err(e) if run.a.heal_device == "gpu" => anyhow::bail!("--heal-device gpu: {e}"),
            Err(e) => run.log(&format!("S2: DTG-MA on the host ({e})")),
        }
    }
    learn_host(run, data, &model, keep_frac)
}

/// [`learn`] through `skillbake` Phase A on the host.
fn learn_host(run: &Run, data: &Data, model: &Arc<CmfModel>, keep_frac: f64) -> Result<Vec<Vec<f32>>> {
    let pad = model.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);
    let on = LossOn::All;
    let cut = |s: &Stream| -> Vec<Vec<u32>> { s.ids.chunks(CHUNK).filter(|c| c.len() == CHUNK).map(<[u32]>::to_vec).collect() };
    let mut tr: Vec<&super::data::Rec> = data.split(TdataSplit::Train).collect();
    tr.sort_by_cached_key(|r| crate::knowledge::sha256_hex(r.rid.as_bytes()));
    let held = cut(&Stream::build(data.split(TdataSplit::Val), on, pad, 0));
    let calib = cut(&Stream::build(tr.into_iter(), on, pad, 0));
    anyhow::ensure!(held.len() >= 2 && calib.len() >= 12, "DTG-MA needs ≥ 2 val and ≥ 12 train chunks of {CHUNK} tokens (have {} / {})", held.len(), calib.len());
    let held_n = held.len().min(HELD);
    let mut chunks: Vec<Vec<u32>> = held.into_iter().take(held_n).collect();
    chunks.extend(calib);
    let n_calib = chunks.len() - held_n;
    let batch = 2usize;
    // one pass over the train chunks, at least 200 steps
    let steps = (n_calib / batch).max(200);
    let hy = cortiq_engine::skillbake::BakeHyper {
        steps_a: steps,
        steps_b: 0,
        lr_a: 0.1,
        lr_b: 0.0,
        eval_every: (steps / 6).max(20),
        fcd_layers: 0,
        batch,
        fcd_batch: 1,
        target_sparsity: (1.0 - keep_frac).clamp(0.0, 0.95),
        l1_init: 0.01,
        l1_step: 0.005,
        tau: 0.5,
        l1_mult: 1.0,
        mask_init: 2.0,
        align: 1,
        uniform_inter: false,
        ..Default::default()
    };
    run.log(&format!(
        "S2: DTG-MA mask training — {n_calib} train + {held_n} held chunks of {CHUNK} tokens, {steps} steps, target {:.0} % of the FFN neurons pruned",
        hy.target_sparsity * 100.0
    ));
    let t0 = std::time::Instant::now();
    let (rep, arts) = cortiq_engine::skillbake::skill_bake(model, &chunks, held_n, &hy, |line| {
        eprintln!("  {line} · {} since the mask training began", crate::fmt_eta(t0.elapsed().as_secs_f64()));
    })
    .map_err(|e| anyhow::anyhow!("DTG-MA: {e}"))?;
    run.log(&format!(
        "S2: DTG-MA mask — held-out ppl {:.3} (backbone {:.3}), {:.1} % pruned at the selected step {} ({:.0}s)",
        rep.masked,
        rep.backbone,
        rep.pruned_ratio * 100.0,
        rep.selected_step,
        rep.sec
    ));
    Ok(arts.logits)
}

/// Per-layer keeps from the logits: one global threshold so the total kept
/// equals `total`, each layer rounded to `align` within [`floor`, width].
pub fn keeps_from(logits: &[Vec<f32>], widths: &[usize], total: usize, align: usize, floor: &[usize]) -> Vec<usize> {
    let mut all: Vec<f32> = logits.iter().flatten().copied().collect();
    all.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let count = |t: f32| -> Vec<usize> {
        logits
            .iter()
            .zip(widths)
            .zip(floor)
            .map(|((lg, &w), &f)| {
                let n = lg.iter().filter(|&&x| x >= t).count();
                let a = align.max(1);
                (n.div_ceil(a) * a).clamp(f.min(w), w)
            })
            .collect()
    };
    // the deepest threshold rank whose rounded keeps still fit the total
    let ok = |r: usize| count(all[r]).iter().sum::<usize>() <= total;
    if all.is_empty() || !ok(0) {
        return floor.iter().zip(widths).map(|(&f, &w)| f.min(w)).collect();
    }
    let (mut lo, mut hi) = (0usize, all.len() - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    count(all[lo])
}

/// Each layer's neuron order, most needed first.
pub fn orders_from(logits: &[Vec<f32>]) -> Vec<Vec<usize>> {
    logits
        .iter()
        .map(|lg| {
            let mut o: Vec<usize> = (0..lg.len()).collect();
            o.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            o
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_follow_the_mask_and_meet_the_total() {
        // layer 0's neurons score high, layer 1's low
        let lg = vec![(0..256).map(|i| 5.0 + i as f32 * 0.01).collect::<Vec<_>>(), (0..256).map(|i| i as f32 * 0.01).collect()];
        let k = keeps_from(&lg, &[256, 256], 256, 128, &[0, 0]);
        assert_eq!(k.iter().sum::<usize>(), 256);
        assert_eq!(k, vec![256, 0]);
        let k = keeps_from(&lg, &[256, 256], 384, 128, &[128, 128]);
        assert_eq!(k, vec![256, 128]);
        let o = orders_from(&lg);
        assert_eq!(o[0][0], 255);
    }
}
