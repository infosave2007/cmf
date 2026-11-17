//! The skill mask over the routed experts (`skill-bake --mask-experts`) on a
//! tiny GDN + bounded-anchor genome with a shared expert and 4 routed experts:
//!  - the all-on hard mask over the shared FFN and every expert is the exact
//!    genome;
//!  - the gradient of every mask group (the shared FFN and each expert of
//!    each selected layer) equals the finite difference of the loss along it;
//!  - Soft-Fold: scaling the down_proj columns of the shared FFN and of every
//!    expert by σ(m) gives the loss the mask gives.
//!
//! The same file runs on the native Metal backend on macOS, on the Vulkan
//! backend in the Linux `--features vulkan` build (`CMF_GPU_SOFTWARE=1`: the
//! software adapter, a CPU) and on CUDA in the `--features cuda` build;
//! skipped when no device exists.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, LayerOffs, Layout, Mixer, SkillState, init_params};
use cortiq_embryo::skill::bake_gpu;

fn genome() -> EmbryoCfg {
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 4;
    cfg.anchor_layers = Some(vec![1, 3]);
    cfg.mixer = Mixer::Gdn;
    cfg.gdn_heads = 2;
    cfg.gdn_dk = 32;
    cfg.gdn_dv = 32;
    cfg.anchor_window = 16;
    cfg.anchor_sink = 2;
    cfg.experts = 4;
    cfg
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// A response-only batch: the first third of every row is prompt (no target).
fn batch(cfg: &EmbryoCfg, b: usize, t: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let tokens: Vec<u32> = (0..b * t).map(|_| (lcg(&mut s) % cfg.vocab as u64) as u32).collect();
    let targets: Vec<u32> = (0..b * t)
        .map(|k| {
            let next = (lcg(&mut s) % cfg.vocab as u64) as u32;
            if k % t < t / 3 { u32::MAX } else { next }
        })
        .collect();
    (tokens, targets)
}

/// The dropless bake instance over a seeded genome, as `bake_v2_train` sets it up.
fn instance(cfg: &EmbryoCfg, b: usize, t: usize) -> Option<(EmbryoGpu, Vec<f32>)> {
    cortiq_embryo::metal::ctx()?;
    let lay = Layout::new(cfg);
    let params = init_params(cfg, &lay, 11);
    let mut seed = EmbryoGpu::new(cfg.clone(), b, t, &params).unwrap();
    let (tk, tg) = batch(cfg, b, t, 5);
    let _ = seed.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    let extras: Vec<(String, Vec<f32>)> = seed
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    drop(seed);
    let mut gpu = bake_gpu(cfg.clone(), b, t, &params)?;
    gpu.set_desc(&extras);
    gpu.desc_updates.set(false);
    gpu.anchor_fixed_window.set(true);
    Some((gpu, params))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn expert_masks_start_at_the_genome_and_their_gradients_match_finite_differences() {
    let cfg = genome();
    let (b, t) = (4usize, 64usize);
    let Some((mut gpu, _)) = instance(&cfg, b, t) else {
        return;
    };
    let c = cortiq_embryo::metal::ctx().unwrap();
    let (tk, tg) = batch(&cfg, b, t, 23);
    let plain = gpu.eval_loss(&tk, &tg);
    let layers = vec![1usize, 2, 3];
    let (ne, i) = (cfg.experts, cfg.inter);
    // σ(0) = 0.5: far from the flat ends, so a finite step is well inside the linear range
    gpu.skill = Some(SkillState::with_experts(
        c,
        layers.clone(),
        i,
        0.0,
        0.25,
        ne,
        gpu.moe_cap,
    ));
    gpu.skill.as_ref().unwrap().hard.set(true);
    let hard = gpu.eval_loss(&tk, &tg);
    assert!(
        (hard - plain).abs() <= 1e-6 * plain.abs().max(1.0),
        "the all-on hard mask is not the genome: {hard} vs {plain}"
    );
    gpu.skill.as_ref().unwrap().hard.set(false);
    let soft = gpu.eval_loss(&tk, &tg);
    let (loss, _) = gpu.train_step_skill(&tk, &tg, 0.0, 0.0, 1e9, false);
    assert!(
        (loss - soft).abs() <= 2e-5 * soft.abs().max(1.0),
        "the training forward and the evaluation disagree under the soft mask: {loss} vs {soft}"
    );
    let (g, base) = {
        let sk = gpu.skill.as_ref().unwrap();
        (sk.g.to_vec(), sk.logits.to_vec())
    };
    assert_eq!(g.len(), layers.len() * (1 + ne) * i);
    let eps = 0.25f32;
    let (mut checked, mut worst) = (0usize, 0f64);
    for mi in 0..layers.len() {
        for grp in 0..=ne {
            let o = gpu.skill.as_ref().unwrap().off(mi, grp);
            let norm = g[o..o + i].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            if norm < 1e-6 {
                continue; // an expert that got no token of this batch
            }
            let shifted = |sign: f32| -> Vec<f32> {
                let mut v = base.clone();
                for j in 0..i {
                    v[o + j] += sign * eps * (g[o + j] as f64 / norm) as f32;
                }
                v
            };
            let sk = gpu.skill.as_ref().unwrap();
            sk.logits.write_from(&shifted(1.0));
            let up = gpu.eval_loss(&tk, &tg) as f64;
            sk.logits.write_from(&shifted(-1.0));
            let down = gpu.eval_loss(&tk, &tg) as f64;
            sk.logits.write_from(&base);
            let fd = (up - down) / (2.0 * eps as f64);
            let rel = (fd - norm).abs() / norm;
            eprintln!(
                "layer {} group {grp}: |g| {norm:.5e}, finite difference {fd:.5e} (rel {rel:.3})",
                layers[mi]
            );
            worst = worst.max(rel);
            checked += 1;
        }
    }
    assert!(checked > layers.len(), "only {checked} mask groups carried a gradient");
    assert!(worst < 0.08, "a mask group's gradient is off by {worst:.3} of its norm");
}

#[test]
fn soft_fold_of_shared_and_expert_masks_is_the_function_the_mask_computes() {
    let cfg = genome();
    let (b, t) = (4usize, 64usize);
    let Some((mut gpu, params)) = instance(&cfg, b, t) else {
        return;
    };
    let c = cortiq_embryo::metal::ctx().unwrap();
    let lay = Layout::new(&cfg);
    let (tk, tg) = batch(&cfg, b, t, 41);
    let layers = vec![0usize, 2, 3];
    let (ne, i, h) = (cfg.experts, cfg.inter, cfg.hidden);
    let sk = SkillState::with_experts(c, layers.clone(), i, 0.0, 0.25, ne, gpu.moe_cap);
    let mut s = 77u64;
    let logits: Vec<f32> = (0..layers.len() * (1 + ne) * i)
        .map(|_| (lcg(&mut s) % 4001) as f32 / 1000.0 - 2.0)
        .collect();
    sk.logits.write_from(&logits);
    let off = |mi: usize, grp: usize| sk.off(mi, grp);
    let mut folded = params.clone();
    for (mi, &l) in layers.iter().enumerate() {
        let ffn = match &lay.layers[l] {
            LayerOffs::Mixer { ffn, .. } | LayerOffs::Anchor { ffn, .. } | LayerOffs::Gdn { ffn, .. } => ffn,
        };
        for grp in 0..=ne {
            let wd = if grp == 0 { ffn.wd } else { ffn.experts + (grp - 1) * 3 * h * i + 2 * h * i };
            for j in 0..i {
                let gain = sigmoid(logits[off(mi, grp) + j]);
                for r in 0..h {
                    folded[wd + r * i + j] *= gain;
                }
            }
        }
    }
    gpu.skill = Some(sk);
    let masked = gpu.eval_loss(&tk, &tg);
    gpu.skill = None;
    let plain = gpu.eval_loss(&tk, &tg);
    gpu.set_params(&folded);
    let by_weights = gpu.eval_loss(&tk, &tg);
    eprintln!("genome {plain:.6}, under the mask {masked:.6}, folded into down_proj {by_weights:.6}");
    assert!(
        (masked - plain).abs() > 1e-4,
        "the mask does not change the loss ({masked} vs {plain}): the test proves nothing"
    );
    assert!(
        (masked - by_weights).abs() <= 3e-5 * masked.abs().max(1.0),
        "the fold is not the masked function: {masked} vs {by_weights}"
    );
}
