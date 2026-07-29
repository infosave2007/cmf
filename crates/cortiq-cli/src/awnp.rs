//! Activation-Weighted Nullspace Projection over a MoE model's expert inputs.
//!
//! The point of AWNP is not the pruning, it is the projection. Dropping the
//! weakest input channels of an expert costs, on Kimi-Linear-48B layer 19,
//! 25.0% of the layer's output energy if the survivors are left alone — and
//! 4.3% if they are refitted to absorb what was removed. This pass does the
//! refit, so the question "what does that cost in perplexity" can be answered
//! by running the result instead of argued from a norm.
//!
//! For each MoE layer with a calibration dump (`CMF_ACT_DUMP`):
//!   C  = XᵀX/n over the FFN input,          [hidden, hidden]
//!   S  = the kept channels, ranked by their weight energy across experts,
//!   Q  = C[:,S]·C[S,S]⁻¹,                    [hidden, |S|]
//!   W' = W·Q scattered back to full width, zeros outside S.
//!
//! The output keeps the original tensor shapes: a column of zeros is exactly
//! what a physically narrowed expert computes, and this way the result runs on
//! today's runtime and can be measured before anyone commits to a format
//! change. Physical narrowing is a separate step and a separate risk.

use anyhow::{Context, bail};
use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_core::quant::dequant_tensor;
use cortiq_core::types::TensorDtype;
use cortiq_engine::qtensor::sgemm_public;
use std::collections::HashMap;
use std::sync::Arc;

/// In-place Cholesky (lower). Returns false if the matrix is not positive
/// definite at the working precision — with a ridge applied that means the
/// calibration sample was too small for the channel count, which is the one
/// way this whole measurement silently lies.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                a[i * n + i] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    true
}

/// Solve `L Lᵀ x = b` in place for `nrhs` right-hand sides stored row-major
/// as [nrhs, n].
fn chol_solve(l: &[f64], n: usize, b: &mut [f64], nrhs: usize) {
    for r in 0..nrhs {
        let x = &mut b[r * n..(r + 1) * n];
        for i in 0..n {
            let mut s = x[i];
            for k in 0..i {
                s -= l[i * n + k] * x[k];
            }
            x[i] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s -= l[k * n + i] * x[k];
            }
            x[i] = s / l[i * n + i];
        }
    }
}

fn read_acts(path: &str, hidden: usize) -> anyhow::Result<Vec<f32>> {
    let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    if raw.len() % (hidden * 4) != 0 {
        bail!("{path}: {} bytes is not a whole number of {hidden}-wide rows", raw.len());
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn cmd_awnp(
    model_path: &str,
    acts_prefix: &str,
    output: &str,
    drop: f64,
    ridge: f64,
    rescale: bool,
) -> anyhow::Result<()> {
    if !(0.0..0.9).contains(&drop) {
        bail!("--drop must be in [0, 0.9)");
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);

    // Group expert tensors by layer: only gate/up read the residual stream on
    // their input side, so only those two get projected. down_proj's input is
    // the intermediate axis and is untouched by channel pruning.
    let mut by_layer: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, e) in model.tensors.iter().enumerate() {
        let n = &e.name;
        if !(n.contains(".mlp.") && (n.ends_with("gate_proj.weight") || n.ends_with("up_proj.weight")))
        {
            continue;
        }
        let Some(rest) = n.strip_prefix("model.layers.") else {
            continue;
        };
        let Some((li, _)) = rest.split_once('.') else {
            continue;
        };
        if let Ok(li) = li.parse::<usize>() {
            by_layer.entry(li).or_default().push(i);
        }
    }

    let mut projected: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut layers_done = 0usize;
    let mut kept_report = Vec::new();

    let mut lids: Vec<usize> = by_layer.keys().copied().collect();
    lids.sort_unstable();
    for li in lids {
        let path = format!("{acts_prefix}.{li}.f32");
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let idxs = &by_layer[&li];
        let hidden = *model.tensors[idxs[0]].shape.get(1).context("2-D expert weight")? as usize;
        let x = read_acts(&path, hidden)?;
        let n_rows = x.len() / hidden;
        if n_rows < 2 * hidden {
            bail!(
                "layer {li}: {n_rows} calibration rows for {hidden} channels. The covariance \
                 needs n well above the channel count — below it the matrix is rank-deficient, \
                 any large enough subset reconstructs exactly, and the projection error reads a \
                 flat 0% that measures the sample, not the model."
            );
        }

        // C = XᵀX / n. Transposing X once beats a scalar triple loop by two
        // orders of magnitude here (45 GFLOP per layer).
        let mut xt = vec![0f32; hidden * n_rows];
        for t in 0..n_rows {
            for j in 0..hidden {
                xt[j * n_rows + t] = x[t * hidden + j];
            }
        }
        let mut c32 = vec![0f32; hidden * hidden];
        sgemm_public(
            hidden, hidden, n_rows, 1.0 / n_rows as f32,
            &xt, n_rows, &x, hidden, false, &mut c32, hidden,
        );
        let c: Vec<f64> = c32.iter().map(|&v| v as f64).collect();

        // Channel energy across every expert in the layer (the ranking that
        // measured best after compensation — the SVD form of the criterion
        // ranks identically once the projection is applied).
        let mut energy = vec![0f64; hidden];
        let mut buf = Vec::new();
        for &ti in idxs {
            let e = &model.tensors[ti];
            let (rows, cols) = (e.shape[0] as usize, e.shape[1] as usize);
            buf.clear();
            buf.resize(rows * cols, 0.0);
            dequant_tensor(e, model.entry_bytes(e), &mut buf).map_err(anyhow::Error::msg)?;
            for r in 0..rows {
                for j in 0..cols {
                    let v = buf[r * cols + j] as f64;
                    energy[j] += v * v;
                }
            }
        }
        let mut order: Vec<usize> = (0..hidden).collect();
        order.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());
        let keep_n = hidden - (hidden as f64 * drop).round() as usize;
        let mut keep: Vec<usize> = order[..keep_n].to_vec();
        keep.sort_unstable();

        // Q = C[:,S]·C[S,S]⁻¹, via Cholesky with a small ridge.
        let mut css = vec![0f64; keep_n * keep_n];
        for (a, &ia) in keep.iter().enumerate() {
            for (b, &ib) in keep.iter().enumerate() {
                css[a * keep_n + b] = c[ia * hidden + ib];
            }
        }
        // The ridge is the difference between a projection that generalizes
        // and one that memorizes the calibration text. At 1e-8 it is decorative:
        // a least-squares refit with ~1500 predictors will happily fit the
        // sample and fall apart on held-out data (measured: PPL 7.3 baseline
        // vs 38 at 12.5% dropped, on a narrow calibration).
        let scale = (0..keep_n).map(|i| css[i * keep_n + i]).sum::<f64>() / keep_n as f64;
        for i in 0..keep_n {
            css[i * keep_n + i] += scale * ridge;
        }
        if !cholesky(&mut css, keep_n) {
            bail!("layer {li}: C[S,S] not positive definite even with a ridge");
        }
        // Solve for Qᵀ: rows are the hidden channels, each a right-hand side.
        let mut qt = vec![0f64; hidden * keep_n];
        for i in 0..hidden {
            for (a, &ia) in keep.iter().enumerate() {
                qt[i * keep_n + a] = c[i * hidden + ia];
            }
        }
        chol_solve(&css, keep_n, &mut qt, hidden);
        let q32: Vec<f32> = qt.iter().map(|&v| v as f32).collect();

        // W' = W·Q, scattered back to full width.
        for &ti in idxs {
            let e = &model.tensors[ti];
            let (rows, cols) = (e.shape[0] as usize, e.shape[1] as usize);
            buf.clear();
            buf.resize(rows * cols, 0.0);
            dequant_tensor(e, model.entry_bytes(e), &mut buf).map_err(anyhow::Error::msg)?;
            let mut small = vec![0f32; rows * keep_n];
            sgemm_public(
                rows, keep_n, cols, 1.0, &buf, cols, &q32, keep_n, false, &mut small, keep_n,
            );
            let mut wide = vec![0f32; rows * cols];
            for r in 0..rows {
                for (a, &ia) in keep.iter().enumerate() {
                    wide[r * cols + ia] = small[r * keep_n + a];
                }
            }
            // Variance-preserving rescale (patent 12). A least-squares refit
            // shrinks the weights — it minimizes error, and part of the input
            // it can no longer see is simply given up rather than guessed at.
            // Restoring the original spread puts the activations back on the
            // scale the rest of the network was trained against.
            if rescale {
                let mean_sq = |v: &[f32]| -> f64 {
                    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64
                };
                let (before, after) = (mean_sq(&buf), mean_sq(&wide));
                if after > 0.0 {
                    let g = (before / after).sqrt() as f32;
                    for v in wide.iter_mut() {
                        *v *= g;
                    }
                }
            }
            projected.insert(ti, crate::convert::encode_for(e.dtype, &wide, rows, cols));
        }
        layers_done += 1;
        kept_report.push((li, keep_n, hidden, n_rows));
    }

    if layers_done == 0 {
        bail!("no layer had a calibration dump at '{acts_prefix}.<layer>.f32'");
    }

    let n_rewritten = projected.len();
    let specs: Vec<TensorSpec> = model
        .tensors
        .iter()
        .enumerate()
        .map(|(i, e)| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: projected
                .remove(&i)
                .unwrap_or_else(|| model.entry_bytes(e).to_vec()),
        })
        .collect();
    CmfModel::write(output, &model.header, &specs, Some(&model.masks), model.vocab.as_deref())?;

    let (li, k, h, n) = kept_report[0];
    println!(
        "AWNP: {layers_done} layers projected, {rewritten} tensors rewritten\n\
         dropped {:.1}% of input channels (layer {li}: {k}/{h} kept from {n} rows)\n\
         → {output}",
        drop * 100.0,
        rewritten = n_rewritten,
    );
    Ok(())
}
