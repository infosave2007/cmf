//! Growth as records (docs §2 "Рост", §4.3): the genome gains an expert
//! slot in every layer — a copy of that layer's hottest expert (the most
//! negative balancing bias) plus noise, its descriptor shifted along the
//! source subspace's first direction so the two split the source's
//! cluster; the network's function is unchanged at insertion. Only the new
//! experts (and their descriptors) then train on the growth corpus, gated
//! on held-out loss against the pre-growth genome. Every old tensor and
//! descriptor stays byte-identical; the export appends `mlp.experts.{E}.*`.

use crate::model::{EmbryoCfg, EmbryoGpu, LayerOffs, Layout, MOE_K, gauss_vec};
use crate::train::{Checkpoint, Sampler, Shard};
use std::time::Instant;

/// Host-side surgery: E → E+1 experts in every layer. Returns the grown
/// checkpoint and, per layer, the source expert copied.
pub fn grow_experts(
    ck: &Checkpoint,
    noise: f32,
    shift: f32,
    seed: u64,
) -> (Checkpoint, Vec<usize>) {
    let cfg0 = ck.cfg.clone();
    let mut cfg = cfg0.clone();
    cfg.experts += 1;
    let (e0, e1) = (cfg0.experts, cfg.experts);
    let (h, i, k) = (cfg.hidden, cfg.inter, MOE_K);
    let lay0 = Layout::new(&cfg0);
    let lay1 = Layout::new(&cfg);
    let mut p = vec![0.0f32; lay1.total];
    // copy every named tensor by name (offsets differ; names are stable)
    let old_by_name: std::collections::HashMap<&str, (usize, usize)> = lay0
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    for (name, off, len) in &lay1.names {
        if let Some((o0, l0)) = old_by_name.get(name.as_str()) {
            debug_assert_eq!(l0, len);
            p[*off..*off + len].copy_from_slice(&ck.params[*o0..*o0 + l0]);
        }
    }
    let ex = |name: &str| {
        ck.extras
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, x)| x.clone())
    };
    let mu0 = ex("desc.mu").unwrap_or_else(|| vec![0.0; cfg0.layers * e0 * h]);
    let u0 = ex("desc.u").unwrap_or_else(|| vec![0.0; cfg0.layers * e0 * k * h]);
    let b0 = ex("desc.bias").unwrap_or_else(|| vec![0.0; cfg0.layers * e0]);
    let mut mu1 = vec![0.0f32; cfg.layers * e1 * h];
    let mut u1 = vec![0.0f32; cfg.layers * e1 * k * h];
    let mut b1 = vec![0.0f32; cfg.layers * e1];
    let mut sources = Vec::new();
    let mut m1 = ck.m.as_ref().map(|_| vec![0.0f32; lay1.total]);
    let mut v1 = ck.v.as_ref().map(|_| vec![0.0f32; lay1.total]);
    if let (Some(m), Some(mo)) = (m1.as_mut(), ck.m.as_ref()) {
        for (name, off, len) in &lay1.names {
            if let Some((o0, _)) = old_by_name.get(name.as_str()) {
                m[*off..*off + len].copy_from_slice(&mo[*o0..*o0 + len]);
            }
        }
    }
    if let (Some(v), Some(vo)) = (v1.as_mut(), ck.v.as_ref()) {
        for (name, off, len) in &lay1.names {
            if let Some((o0, _)) = old_by_name.get(name.as_str()) {
                v[*off..*off + len].copy_from_slice(&vo[*o0..*o0 + len]);
            }
        }
    }
    let ew = 3 * h * i;
    for l in 0..cfg.layers {
        // old descriptors copied verbatim
        mu1[l * e1 * h..l * e1 * h + e0 * h].copy_from_slice(&mu0[l * e0 * h..(l + 1) * e0 * h]);
        u1[l * e1 * k * h..l * e1 * k * h + e0 * k * h]
            .copy_from_slice(&u0[l * e0 * k * h..(l + 1) * e0 * k * h]);
        b1[l * e1..l * e1 + e0].copy_from_slice(&b0[l * e0..(l + 1) * e0]);
        // hottest expert = most negative balancing bias
        let src = (0..e0)
            .min_by(|&a, &b| b0[l * e0 + a].partial_cmp(&b0[l * e0 + b]).unwrap())
            .unwrap_or(0);
        sources.push(src);
        let ffn = match &lay1.layers[l] {
            LayerOffs::Mixer { ffn, .. } | LayerOffs::Anchor { ffn, .. } => ffn,
        };
        let (s_off, n_off) = (ffn.experts + src * ew, ffn.experts + e0 * ew);
        let noise_v = gauss_vec(seed.wrapping_add(l as u64), ew);
        for j in 0..ew {
            p[n_off + j] = p[s_off + j] + noise * noise_v[j];
        }
        // descriptor: μ_new = μ_src + shift·‖μ_src‖·u₀ (u₀: first subspace row, or a random unit)
        let mu_src = &mu0[(l * e0 + src) * h..(l * e0 + src + 1) * h];
        let mut dir: Vec<f32> = u0[(l * e0 + src) * k * h..(l * e0 + src) * k * h + h].to_vec();
        let dn: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if dn < 1e-6 {
            dir = gauss_vec(seed.wrapping_add(1000 + l as u64), h);
            let n2: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for d in &mut dir {
                *d /= n2;
            }
        }
        let mn: f32 = mu_src.iter().map(|x| x * x).sum::<f32>().sqrt();
        let base = (l * e1 + e0) * h;
        for j in 0..h {
            mu1[base + j] = mu_src[j] + shift * mn * dir[j];
        }
        let ub = (l * e1 + e0) * k * h;
        u1[ub..ub + k * h].copy_from_slice(&u0[(l * e0 + src) * k * h..(l * e0 + src + 1) * k * h]);
        b1[l * e1 + e0] = b0[l * e0 + src];
    }
    let extras = vec![
        ("desc.mu".to_string(), mu1),
        ("desc.u".to_string(), u1),
        ("desc.bias".to_string(), b1),
    ];
    (
        Checkpoint {
            cfg,
            step: ck.step,
            params: p,
            m: m1,
            v: v1,
            extras,
        },
        sources,
    )
}

pub struct GrowArgs {
    pub steps: usize,
    pub lr: f32,
    pub batch: usize,
    pub seq: usize,
    pub eval_every: usize,
    pub seed: u64,
}

/// Train only the newest expert of every layer (index E−1) on `corpus`;
/// returns (trained checkpoint, held-out loss before, after).
pub fn train_new_experts(
    ck: &Checkpoint,
    corpus: &Shard,
    a: &GrowArgs,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<(Checkpoint, f32, f32)> {
    let cfg: EmbryoCfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let (h, i) = (cfg.hidden, cfg.inter);
    let e_new = cfg.experts - 1;
    let mut gpu = EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &ck.params)
        .ok_or_else(|| anyhow::anyhow!("no Metal"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_frozen_below.set(e_new);
    let n = corpus.tokens.len();
    anyhow::ensure!(n > 20 * (a.seq + 2), "growth corpus too small: {n} tokens");
    let cut = n - n / 10;
    let train = Shard {
        tokens: corpus.tokens[..cut].to_vec(),
    };
    let held = Shard {
        tokens: corpus.tokens[cut..].to_vec(),
    };
    let m = a.batch * a.seq;
    let nval = (held.tokens.len() / (m + 1)).clamp(1, 4);
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let eval = |gpu: &EmbryoGpu, tk: &mut Vec<u32>, tg: &mut Vec<u32>| -> f32 {
        let mut s = 0.0;
        for kk in 0..nval {
            Sampler::fixed_batch(&held, a.batch, a.seq, kk, tk, tg);
            s += gpu.eval_loss(tk, tg);
        }
        s / nval as f32
    };
    let l_before = eval(&gpu, &mut tk, &mut tg);
    // trainable ranges: the new expert in every layer
    let ew = 3 * h * i;
    let ranges: Vec<(usize, usize)> = lay
        .layers
        .iter()
        .map(|lo| {
            let ffn = match lo {
                LayerOffs::Mixer { ffn, .. } | LayerOffs::Anchor { ffn, .. } => ffn,
            };
            (ffn.experts + e_new * ew, ew)
        })
        .collect();
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed);
    let t0 = Instant::now();
    let mut best = (l_before, gpu.params_host(), gpu.desc_host());
    for step in 0..a.steps {
        if should_stop() {
            anyhow::bail!("preempted");
        }
        sampler.batch(&train, &mut tk, &mut tg);
        let lr =
            a.lr * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / a.steps.max(1) as f32).cos());
        let (loss, gn) = gpu.train_step_ranges(&tk, &tg, lr, 0.0, 1.0, &ranges);
        if (step + 1) % a.eval_every == 0 || step + 1 == a.steps {
            let vl = eval(&gpu, &mut tk, &mut tg);
            eprintln!(
                "  grow step {:>4} loss {loss:.4} |g| {gn:.3} lr {lr:.2e} held-out {vl:.4} (before {l_before:.4}) [{:.0} s]",
                step + 1,
                t0.elapsed().as_secs_f64()
            );
            if vl < best.0 {
                best = (vl, gpu.params_host(), gpu.desc_host());
            }
        }
    }
    let extras: Vec<(String, Vec<f32>)> = best
        .2
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    Ok((
        Checkpoint {
            cfg,
            step: ck.step,
            params: best.1,
            m: None,
            v: None,
            extras,
        },
        l_before,
        best.0,
    ))
}
