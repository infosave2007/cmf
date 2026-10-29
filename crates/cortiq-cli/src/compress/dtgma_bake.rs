//! `cortiq dtgma-bake`: the task's skill baked into the weights before a
//! compress (DTG-MA, Patent 2). Soft gates σ(m) on the FFN neurons of the
//! deep layers are trained with the task's LM loss at a small constant L1
//! — no sparsity target, the held-out best is kept — and folded into
//! `down_proj` (Soft-Fold: exact, the gate scales the activation its column
//! reads). The file keeps its width; `compress` then folds and heals it as
//! usual, its illumination ranking the neurons through the baked gains.

use super::cmfio::{self, Encoding};
use super::data::{Data, LossOn};
use super::heal_gpu::{MaskCfg, learn_masks};
use anyhow::{Context, Result, ensure};
use cortiq_core::CmfModel;
use serde_json::json;
use std::path::Path;

pub struct Cfg {
    pub from: usize,
    pub steps: usize,
    pub eval_every: usize,
    pub lr: f64,
    pub l1: f64,
    pub init: f32,
    pub window: usize,
    pub batch: usize,
    pub max_batches: usize,
}

pub struct Report {
    pub nll_open: f64,
    pub nll_best: f64,
    pub step: usize,
    pub steps: usize,
}

pub fn bake(model_path: &Path, data_spec: &str, loss_on: &str, cfg: &Cfg, out: &Path, threads: usize, log: &dyn Fn(&str)) -> Result<Report> {
    let model = CmfModel::open_sharded(model_path).with_context(|| format!("opening {}", model_path.display()))?;
    super::heal_gpu::usable(&model).map_err(|e| anyhow::anyhow!("dtgma bake: {e}"))?;
    let nl = model.arch().num_layers;
    ensure!(cfg.from < nl, "dtgma bake: --from-layer {} past the model's {nl} layers", cfg.from);
    let data = Data::load(data_spec)?;
    let mcfg = MaskCfg {
        window: cfg.window,
        batch: cfg.batch,
        keep_frac: 1.0,
        lr: cfg.lr,
        l1_init: cfg.l1,
        l1_step: 0.0,
        l1_grow: 1.0,
        eval_every: cfg.eval_every,
        max_steps: cfg.steps,
        mask_init: cfg.init,
        tau: 0.5,
        // no target: every checkpoint qualifies, the run goes to its last step
        settle: cfg.steps / cfg.eval_every.max(1) + 1,
        train_from: cfg.from,
        soft: true,
        max_batches: cfg.max_batches,
    };
    let rep = learn_masks(&data, LossOn::parse(loss_on)?, model_path, &mcfg, log)?;
    ensure!(rep.logits.len() == nl - cfg.from, "dtgma bake: {} gated layers, expected {}", rep.logits.len(), nl - cfg.from);
    let gains: Vec<Vec<f32>> = rep.logits.iter().map(|l| l.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()).collect();
    let (lo, mean) = gains.iter().flatten().fold((f32::MAX, 0f64), |(m, s), &g| (m.min(g), s + g as f64));
    let n: usize = gains.iter().map(Vec::len).sum();
    log(&format!("dtgma bake: gains on layers {}..{}: mean {:.3}, min {lo:.3}", cfg.from, nl - 1, mean / n as f64));
    let prov = json!({
        "data": data_spec, "from_layer": cfg.from, "steps": rep.steps, "best_step": rep.step,
        "lr": cfg.lr, "l1": cfg.l1, "init": cfg.init, "heldout_open": rep.nll_open, "heldout_best": rep.nll_hard,
    });
    cmfio::rewrite(
        &model,
        out,
        Encoding::KeepF16,
        &|name| {
            let Some(l) = name.strip_prefix("model.layers.").and_then(|r| r.strip_suffix(".mlp.down_proj.weight")).and_then(|l| l.parse::<usize>().ok()) else {
                return Ok(None);
            };
            if l < cfg.from {
                return Ok(None);
            }
            let e = model.tensor(name).context("down_proj")?;
            let g = &gains[l - cfg.from];
            let (rows, cols) = (e.shape[0], e.shape[1]);
            ensure!(cols == g.len(), "{name}: {cols} columns, {} gates", g.len());
            let mut v = cmfio::tensor_f32(&model, e)?;
            for r in 0..rows {
                for (x, &s) in v[r * cols..(r + 1) * cols].iter_mut().zip(g) {
                    *x *= s;
                }
            }
            Ok(Some((e.shape.clone(), v)))
        },
        &|h| cmfio::provenance_set(h, "dtgma_bake", prov.clone()),
        threads,
    )?;
    Ok(Report { nll_open: rep.nll_open, nll_best: rep.nll_hard, step: rep.step, steps: rep.steps })
}
