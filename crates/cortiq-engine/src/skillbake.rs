//! Native DTG-MA skill bake (Patent 2) — no Python, no torch.
//!
//! The certified recipe of `converter/make_skill_l1fcd.py`, in Rust on
//! the `FcdModel` f32 replica:
//!
//! - **Phase A** — a trainable L1 mask over FFN neurons (one logit per
//!   neuron, applied to the input of down_proj as σ(m)): pure LM loss
//!   on the task corpus + a progressive L1 penalty. Every 30 steps the
//!   binarized mask (σ>τ) is scored on held-out chunks; the best
//!   checkpoint — the *denoising bottom* — is restored at the end.
//!   Pruning noise neurons IMPROVES the model before it starts to hurt.
//! - **Phase B** — FCD: the FFN of the last N layers trains against the
//!   same LM loss with the hard mask active (cosine LR), held-out
//!   gated, best checkpoint restored.
//!
//! Attention (softmax and GDN alike) is FROZEN and carries no gradient
//! — exactly like the reference recipe (`torch.no_grad()` around the
//! attention branch): the backward walks the residual stream through
//! the FFN chain only, which is what makes a pure-Rust backward small.

use crate::fcd::{FcdModel, LnFfn};
use crate::fcd_ops as ops;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Hyper-parameters — defaults are the certified recipe.
#[derive(Clone, Debug)]
pub struct BakeHyper {
    pub steps_a: usize,
    pub steps_b: usize,
    pub l1_init: f64,
    pub l1_step: f64,
    pub eval_every: usize,
    pub lr_a: f64,
    pub lr_b: f64,
    pub tau: f32,
    pub fcd_layers: usize,
    pub seed: u64,
}

impl Default for BakeHyper {
    fn default() -> Self {
        Self {
            steps_a: 240,
            steps_b: 120,
            l1_init: 0.01,
            l1_step: 0.005,
            eval_every: 30,
            lr_a: 0.1,
            lr_b: 1e-5,
            tau: 0.5,
            fcd_layers: 4,
            seed: 0,
        }
    }
}

/// What the bake measured and produced.
pub struct BakeReport {
    /// Held-out PPL of the untouched backbone.
    pub backbone: f64,
    /// Held-out PPL with the best hard mask (the denoising bottom).
    pub masked: f64,
    /// Held-out PPL after FCD (the final specialist).
    pub overlaid: f64,
    pub pruned_ratio: f64,
    pub kept_per_layer: Vec<usize>,
    pub sec: f64,
}

/// The trained artifacts: everything the defrag writer needs, f32.
pub struct BakeArtifacts {
    /// Per-layer live-neuron flags (true = keep).
    pub keep: Vec<Vec<bool>>,
    /// Per-layer down_proj `[hidden, inter]` with dead columns zeroed
    /// (FCD layers: the trained weights; others: the backbone's).
    pub down: Vec<Vec<f32>>,
    /// Trained gate/up for the FCD layers (`None` elsewhere).
    pub gate_up: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Which layers went through Phase B.
    pub fcd_layers: Vec<usize>,
}

const CLIP: f64 = 1.0;
const B1: f64 = 0.9;
const B2: f64 = 0.999;
const EPS: f64 = 1e-8;

/// Plain Adam over a set of f32 tensors (masks are tiny, FFN mid-size).
struct Adam {
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: i32,
    lr: f64,
}

impl Adam {
    fn new(sizes: &[usize], lr: f64) -> Self {
        Self {
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            t: 0,
            lr,
        }
    }

    /// Global-norm clip + Adam step. `params[i].len() == grads[i].len()`.
    fn step(&mut self, params: &mut [&mut [f32]], grads: &[Vec<f64>], lr_scale: f64) {
        let gn: f64 = grads
            .iter()
            .flat_map(|g| g.iter().map(|x| x * x))
            .sum::<f64>()
            .sqrt();
        let clip = if gn > CLIP { CLIP / gn } else { 1.0 };
        self.t += 1;
        let (bc1, bc2) = (1.0 - B1.powi(self.t), 1.0 - B2.powi(self.t));
        for (pi, p) in params.iter_mut().enumerate() {
            for j in 0..p.len() {
                let g = grads[pi][j] * clip;
                let m = &mut self.m[pi][j];
                let v = &mut self.v[pi][j];
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                let upd = (*m / bc1) / ((*v / bc2).sqrt() + EPS);
                p[j] -= (self.lr * lr_scale * upd) as f32;
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One forward + CE(+optionally backward through the FFN chain).
/// Returns (nll_sum, tokens). `dmask`/`dffn` accumulate when given.
struct Pass<'a> {
    fm: &'a FcdModel,
    tau: f32,
    /// σ(m) per layer when soft; binarized when `hard`.
    logits: &'a [Vec<f32>],
    hard: bool,
    /// Phase-B replacement FFN weights per layer (trained copies).
    ffn: &'a [Option<(Vec<f32>, Vec<f32>, Vec<f32>)>],
}

impl Pass<'_> {
    fn gates(&self, li: usize) -> Vec<f32> {
        self.logits[li]
            .iter()
            .map(|&l| {
                let s = sigmoid(l);
                if self.hard {
                    if s > self.tau { 1.0 } else { 0.0 }
                } else {
                    s
                }
            })
            .collect()
    }

    fn wts<'b>(&'b self, li: usize) -> LnFfn<'b> {
        let l = &self.fm.layers[li];
        match &self.ffn[li] {
            Some((g, u, d)) => LnFfn {
                iln: &l.iln,
                pln: &l.pln,
                gate: g,
                up: u,
                down: d,
            },
            None => LnFfn {
                iln: &l.iln,
                pln: &l.pln,
                gate: &l.gate,
                up: &l.up,
                down: &l.down,
            },
        }
    }

    /// Teacher-forced NLL over one chunk; when `grad` is set, backprop
    /// through the FFN chain into the mask grads (and FFN grads for
    /// Phase-B layers).
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        &self,
        ids: &[u32],
        grad: Option<(
            &mut [Vec<f64>],
            &mut [Option<(Vec<f64>, Vec<f64>, Vec<f64>)>],
        )>,
    ) -> (f64, usize) {
        let fm = self.fm;
        let (t, hsz) = (ids.len(), fm.hidden);
        let nl = fm.layers.len();
        // Embed.
        let mut h = vec![0f32; t * hsz];
        for (r, &id) in ids.iter().enumerate() {
            h[r * hsz..(r + 1) * hsz]
                .copy_from_slice(&fm.embed[id as usize * hsz..(id as usize + 1) * hsz]);
        }
        // Forward, keeping per-layer inputs + activations.
        let mut h_ins = Vec::with_capacity(nl);
        let mut acts = Vec::with_capacity(nl);
        let mut masks = Vec::with_capacity(nl);
        for li in 0..nl {
            let g = self.gates(li);
            let wts = self.wts(li);
            let want = grad.is_some();
            let (h2, a) = fm.layer_forward_scaled(li, &h, 1, t, &wts, false, want, Some(&g));
            h_ins.push(if want { h } else { Vec::new() });
            acts.push(a);
            masks.push(g);
            h = h2;
        }
        // Final norm + tied LM head, CE summed over positions 1..t.
        let mut hn = vec![0f32; t * hsz];
        let mut inv = vec![0f32; t];
        ops::rmsnorm_fwd(&h, &fm.final_norm, fm.eps, fm.gemma, &mut hn, &mut inv);
        let lm: &[f32] = fm.lm_head.as_deref().unwrap_or(&fm.embed);
        let vocab = lm.len() / hsz;
        let pool = fm.pool.as_deref();
        let mut nll = 0f64;
        let mut dh_n = vec![0f32; t * hsz]; // dL/d hn
        // Chunk the vocab matmul over positions to bound the logits buf.
        const POS_CHUNK: usize = 32;
        let scored = t - 1;
        let mut p0 = 0usize;
        while p0 < scored {
            let pc = POS_CHUNK.min(scored - p0);
            let mut logits = vec![0f32; pc * vocab];
            ops::gemm_nt(
                &hn[p0 * hsz..(p0 + pc) * hsz],
                lm,
                &mut logits,
                pc,
                hsz,
                vocab,
                pool,
            );
            for r in 0..pc {
                let target = ids[p0 + r + 1] as usize;
                let row = &mut logits[r * vocab..(r + 1) * vocab];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let mut sum = 0f64;
                for v in row.iter() {
                    sum += ((*v as f64) - mx).exp();
                }
                nll += mx + sum.ln() - row[target] as f64;
                if grad.is_some() {
                    // dCE/dlogit = softmax − onehot, scaled by 1/scored.
                    let inv_n = 1.0 / scored as f64;
                    for v in row.iter_mut() {
                        *v = ((((*v as f64) - mx).exp() / sum) * inv_n) as f32;
                    }
                    row[target] -= inv_n as f32;
                }
            }
            if grad.is_some() {
                ops::gemm_dx(
                    &logits,
                    lm,
                    &mut dh_n[p0 * hsz..(p0 + pc) * hsz],
                    pc,
                    hsz,
                    vocab,
                    pool,
                );
            }
            p0 += pc;
        }
        let Some((dmask, dffn)) = grad else {
            return (nll, scored);
        };
        // Backward: final norm, then the FFN chain layer by layer.
        let mut dh = vec![0f32; t * hsz];
        ops::rmsnorm_bwd(&h, &fm.final_norm, &inv, &dh_n, fm.gemma, &mut dh, None);
        for li in (0..nl).rev() {
            let a = acts[li].as_ref().expect("acts saved in grad mode");
            let g = &masks[li];
            let inter = fm.layers[li].inter;
            let wts = self.wts(li);
            // h2 = h1 + act2 @ downᵀ  →  dact2 = dh @ down.
            let mut dact2 = vec![0f32; t * inter];
            ops::gemm_dx(&dh, wts.down, &mut dact2, t, inter, hsz, fm.pool.as_deref());
            if let Some((_, _, dd)) = dffn[li].as_mut() {
                // dW_down += dhᵀ · act2 (act2 = act·g).
                let mut act2 = a.act.clone();
                for r in 0..t {
                    for (x, &gv) in act2[r * inter..(r + 1) * inter].iter_mut().zip(g) {
                        *x *= gv;
                    }
                }
                let mut dw = vec![0f32; hsz * inter];
                ops::gemm_dw(&dh, &act2, &mut dw, t, inter, hsz, fm.pool.as_deref());
                for (o, &x) in dd.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
            }
            // Mask grad: dm = Σ_t dact2·act · σ'(m)  (soft; STE-equal).
            {
                let dm = &mut dmask[li];
                for r in 0..t {
                    let da = &dact2[r * inter..(r + 1) * inter];
                    let aa = &a.act[r * inter..(r + 1) * inter];
                    for j in 0..inter {
                        dm[j] += da[j] as f64 * aa[j] as f64;
                    }
                }
                // σ'(m) folded in once per chunk (constant per neuron).
                for (j, d) in dm.iter_mut().enumerate() {
                    let _ = j;
                    let _ = d;
                }
            }
            // dact = dact2 · g;  silu·mul backward.
            let mut dg_pre = vec![0f32; t * inter];
            let mut du_pre = vec![0f32; t * inter];
            for r in 0..t {
                for j in 0..inter {
                    let i = r * inter + j;
                    let da = dact2[i] * g[j];
                    let sg = ops::silu(a.gpre[i]);
                    dg_pre[i] = da * a.upre[i] * ops::silu_bwd(a.gpre[i]);
                    du_pre[i] = da * sg;
                }
            }
            // dn2 = dg_pre @ gate + du_pre @ up.
            let mut dn2 = vec![0f32; t * hsz];
            ops::gemm_dx(
                &dg_pre,
                wts.gate,
                &mut dn2,
                t,
                hsz,
                inter,
                fm.pool.as_deref(),
            );
            let mut dn2b = vec![0f32; t * hsz];
            ops::gemm_dx(
                &du_pre,
                wts.up,
                &mut dn2b,
                t,
                hsz,
                inter,
                fm.pool.as_deref(),
            );
            for (x, &y) in dn2.iter_mut().zip(&dn2b) {
                *x += y;
            }
            if let Some((dgw, duw, _)) = dffn[li].as_mut() {
                let mut dw = vec![0f32; inter * hsz];
                ops::gemm_dw(&dg_pre, &a.n2, &mut dw, t, hsz, inter, fm.pool.as_deref());
                for (o, &x) in dgw.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
                dw.fill(0.0);
                ops::gemm_dw(&du_pre, &a.n2, &mut dw, t, hsz, inter, fm.pool.as_deref());
                for (o, &x) in duw.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
            }
            // Post-norm backward into h1; the attention branch carries
            // no gradient (frozen), so dh1 flows straight to dh_in.
            let mut dh1 = dh.clone(); // residual h2 = h1 + ffn
            ops::rmsnorm_bwd(&a.h1, wts.pln, &a.inv2, &dn2, fm.gemma, &mut dh1, None);
            dh = dh1;
            let _ = &h_ins[li];
        }
        (nll, scored)
    }
}

/// Held-out PPL with the hard mask (and Phase-B weights when present).
fn held_ppl(pass: &Pass, held: &[Vec<u32>]) -> f64 {
    let mut nll = 0f64;
    let mut n = 0usize;
    for c in held {
        let (l, k) = pass.chunk(c, None);
        nll += l;
        n += k;
    }
    (nll / n.max(1) as f64).exp()
}

/// The whole recipe. `log` receives progress lines.
pub fn skill_bake(
    model: &Arc<CmfModel>,
    chunks: &[Vec<u32>],
    held_n: usize,
    hy: &BakeHyper,
    mut log: impl FnMut(&str),
) -> Result<(BakeReport, BakeArtifacts), String> {
    let t0 = std::time::Instant::now();
    let o1_off = crate::nystrom::O1Cfg {
        layers: crate::nystrom::O1Layers::List(Vec::new()),
        m: 4,
        w: 8,
        sink: 1,
        rect: crate::nystrom::O1_DEFAULT_RECT,
    };
    let fm = FcdModel::from_cmf(model, &o1_off)?;
    let nl = fm.layers.len();
    let inter = fm.layers.iter().map(|l| l.inter).max().unwrap_or(0);
    if fm.layers.iter().any(|l| l.inter != inter) {
        return Err("skill bake: non-uniform FFN widths".into());
    }
    let held: Vec<Vec<u32>> = chunks[..held_n.min(chunks.len())].to_vec();
    let calib: Vec<Vec<u32>> = chunks[held_n.min(chunks.len())..].to_vec();
    if calib.len() < 12 {
        return Err(format!(
            "skill bake: corpus too small ({} calib chunks)",
            calib.len()
        ));
    }
    let fcd: Vec<usize> = (nl.saturating_sub(hy.fcd_layers)..nl).collect();
    let _rng = SplitMix64::new(hy.seed);

    // Trainables.
    let mut logits: Vec<Vec<f32>> = vec![vec![2.0; inter]; nl];
    let mut ffn: Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> = vec![None; nl];

    // Baseline (no mask): σ(2.0)≈0.88 is NOT identity, so measure with
    // gates forced open via hard mask over +∞… simplest: logits +50.
    let open: Vec<Vec<f32>> = vec![vec![50.0; inter]; nl];
    let base_pass = Pass {
        fm: &fm,
        tau: hy.tau,
        logits: &open,
        hard: true,
        ffn: &ffn,
    };
    let backbone = held_ppl(&base_pass, &held);
    log(&format!("baseline (full): {backbone:.3}"));

    // ── Phase A: mask training ──
    let mut adam_a = Adam::new(&vec![inter; nl], hy.lr_a);
    let mut l1 = hy.l1_init;
    let mut best: (f64, Option<Vec<Vec<f32>>>, f64) = (backbone, None, 0.0);
    for step in 0..hy.steps_a {
        let chunk = &calib[step % calib.len()];
        let mut dmask: Vec<Vec<f64>> = vec![vec![0.0; inter]; nl];
        let mut dffn: Vec<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> = vec![None; nl];
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: false,
            ffn: &ffn,
        };
        let _ = pass.chunk(chunk, Some((&mut dmask, &mut dffn)));
        // Fold σ'(m) into the mask grads + add the L1 term.
        let l1_per = l1 / (inter as f64 * nl as f64);
        for li in 0..nl {
            for j in 0..inter {
                let s = sigmoid(logits[li][j]) as f64;
                dmask[li][j] = dmask[li][j] * s * (1.0 - s) + l1_per * s * (1.0 - s);
            }
        }
        let mut params: Vec<&mut [f32]> = logits.iter_mut().map(|v| v.as_mut_slice()).collect();
        adam_a.step(&mut params, &dmask, 1.0);
        if (step + 1) % hy.eval_every == 0 {
            l1 += hy.l1_step;
            let pass = Pass {
                fm: &fm,
                tau: hy.tau,
                logits: &logits,
                hard: true,
                ffn: &ffn,
            };
            let hp = held_ppl(&pass, &held);
            let alive: usize = logits
                .iter()
                .map(|l| l.iter().filter(|&&x| sigmoid(x) > hy.tau).count())
                .sum();
            let sp = 1.0 - alive as f64 / (nl * inter) as f64;
            if hp < best.0 {
                best = (hp, Some(logits.clone()), sp);
            }
            log(&format!(
                "  [A] step {}: L1={l1:.3} pruned={:.0}% hard-PPL={hp:.3} (bottom {:.3}@{:.0}%)",
                step + 1,
                sp * 100.0,
                best.0,
                best.2 * 100.0
            ));
        }
    }
    if let Some(b) = best.1.take() {
        logits = b;
    }
    let pass = Pass {
        fm: &fm,
        tau: hy.tau,
        logits: &logits,
        hard: true,
        ffn: &ffn,
    };
    let masked = held_ppl(&pass, &held);
    log(&format!(
        "[A] {:.0}s: masked-PPL {masked:.3}",
        t0.elapsed().as_secs_f64()
    ));

    // ── Phase B: FCD of the last N layers' FFN (hard mask active) ──
    for &li in &fcd {
        let l = &fm.layers[li];
        ffn[li] = Some((l.gate.clone(), l.up.clone(), l.down.clone()));
    }
    let sizes: Vec<usize> = fcd
        .iter()
        .flat_map(|&li| {
            let l = &fm.layers[li];
            [l.gate.len(), l.up.len(), l.down.len()]
        })
        .collect();
    let mut adam_b = Adam::new(&sizes, hy.lr_b);
    let mut best_b: (f64, Option<Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>>>) = (masked, None);
    for step in 0..hy.steps_b {
        let chunk = &calib[step % calib.len()];
        let mut dmask: Vec<Vec<f64>> = vec![vec![0.0; inter]; nl];
        let mut dffn: Vec<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> = (0..nl)
            .map(|li| {
                ffn[li]
                    .as_ref()
                    .map(|(g, u, d)| (vec![0.0; g.len()], vec![0.0; u.len()], vec![0.0; d.len()]))
            })
            .collect();
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: true,
            ffn: &ffn,
        };
        let _ = pass.chunk(chunk, Some((&mut dmask, &mut dffn)));
        // Cosine LR.
        let lr_scale = 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / hy.steps_b as f64).cos());
        let first_fcd = fcd[0];
        let mut params: Vec<&mut [f32]> = Vec::new();
        let mut grads: Vec<Vec<f64>> = Vec::new();
        for (off, slot) in ffn[first_fcd..].iter_mut().enumerate() {
            let li = first_fcd + off;
            let Some((g, u, d)) = slot.as_mut() else {
                continue;
            };
            let (dg, du, dd) = dffn[li].take().unwrap();
            params.push(g.as_mut_slice());
            grads.push(dg);
            params.push(u.as_mut_slice());
            grads.push(du);
            params.push(d.as_mut_slice());
            grads.push(dd);
        }
        adam_b.step(&mut params, &grads, lr_scale);
        if (step + 1) % hy.eval_every == 0 {
            let pass = Pass {
                fm: &fm,
                tau: hy.tau,
                logits: &logits,
                hard: true,
                ffn: &ffn,
            };
            let cur = held_ppl(&pass, &held);
            if cur < best_b.0 {
                best_b = (cur, Some(ffn.clone()));
            }
            log(&format!(
                "  [B] step {}: held-PPL {cur:.3} (best {:.3})",
                step + 1,
                best_b.0
            ));
        }
    }
    if let Some(b) = best_b.1.take() {
        ffn = b;
    }
    let overlaid = best_b.0;

    // ── Export artifacts ──
    let mut keep = Vec::with_capacity(nl);
    let mut down_out = Vec::with_capacity(nl);
    let mut gate_up = Vec::with_capacity(nl);
    let mut kept_per_layer = Vec::with_capacity(nl);
    for li in 0..nl {
        let alive: Vec<bool> = logits[li].iter().map(|&x| sigmoid(x) > hy.tau).collect();
        kept_per_layer.push(alive.iter().filter(|&&a| a).count());
        let l = &fm.layers[li];
        let mut down = match &ffn[li] {
            Some((_, _, d)) => d.clone(),
            None => l.down.clone(),
        };
        let hsz = fm.hidden;
        for r in 0..hsz {
            for (c, &a) in alive.iter().enumerate() {
                if !a {
                    down[r * inter + c] = 0.0;
                }
            }
        }
        gate_up.push(ffn[li].as_ref().map(|(g, u, _)| (g.clone(), u.clone())));
        down_out.push(down);
        keep.push(alive);
    }
    let total: usize = kept_per_layer.iter().sum();
    let report = BakeReport {
        backbone,
        masked,
        overlaid,
        pruned_ratio: 1.0 - total as f64 / (nl * inter) as f64,
        kept_per_layer,
        sec: t0.elapsed().as_secs_f64(),
    };
    let arts = BakeArtifacts {
        keep,
        down: down_out,
        gate_up,
        fcd_layers: fcd,
    };
    Ok((report, arts))
}
