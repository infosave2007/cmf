//! Whole-model Vulkan finite-difference gate.
//!
//! This is intentionally a micro genome rather than a quality run: it keeps
//! one legacy hybrid layer, one anchor, and one selected Phase-Delta layer in
//! the same graph, with conv4, dropless custom routing, and the hierarchical
//! tied head enabled.  Every named arena tensor gets a directional finite
//! difference check, so a forward/backward seam can not pass by testing only
//! one operator in isolation.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, MOE_K, init_params};
use cortiq_embryo::ops::lcg_vec;

fn routed_descriptors(cfg: &EmbryoCfg) -> (Vec<f32>, Vec<f32>) {
    let h = cfg.hidden;
    let n = cfg.layers * cfg.experts;
    let mut u = vec![0.0f32; n * MOE_K * h];
    // Keep the descriptor projection nonzero without making the route
    // discontinuous for this finite-difference witness.
    for (i, x) in u.iter_mut().enumerate() {
        *x = 0.0025 * (((i * 17 + 3) % 23) as f32 - 11.0);
    }
    let mut bias = vec![0.0f32; n];
    for (i, x) in bias.iter_mut().enumerate() {
        *x = 0.01 * (i as f32 - (n as f32 - 1.0) * 0.5);
    }
    (u, bias)
}

/// Pick deterministic, distributed coordinates whose analytic gradient is
/// actually active.  The old witness always used the first 64 values of every
/// tensor, which made sparse embedding/expert tensors appear checked while
/// differentiating zeros.  Filtering against the observed gradient keeps the
/// route-frozen finite difference about the graph rather than inactive rows.
fn active_witnesses(grads: &[f32], off: usize, n: usize) -> Vec<usize> {
    let max_abs = (0..n).map(|i| grads[off + i].abs()).fold(0.0f32, f32::max);
    let cutoff = (max_abs * 1e-3).max(1e-7);
    let active: Vec<usize> = (0..n).filter(|&i| grads[off + i].abs() >= cutoff).collect();
    // Include every active coordinate.  The signs still come from the
    // analytic gradient, so this remains one normalized directional check
    // rather than a bulk norm assertion; sparse tensors simply contribute
    // their active rows.  Using the complete active direction keeps tiny
    // expert-gate gradients above the f32 per-position-loss readback floor.
    let take = active.len();
    if take == 0 {
        return Vec::new();
    }
    if take == 1 {
        return vec![active[active.len() / 2]];
    }
    (0..take)
        .map(|j| active[j * (active.len() - 1) / (take - 1)])
        .collect()
}

#[test]
fn whole_graph_named_tensor_finite_difference() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 3;
    cfg.anchor_every = 2; // legacy mixer, anchor, selected Phase-Delta mixer
    cfg.phase_delta_layer = Some(2);
    cfg.conv_k = 4;
    cfg.experts = 4;
    cfg.head_clusters = 64;
    let (b, t) = (4usize, 128usize); // crosses the 64-token recurrence boundary
    let m = b * t;
    let layout = Layout::new(&cfg);
    let p0 = init_params(&cfg, &layout, 0x51_7f);
    let mut gpu = EmbryoGpu::new_eval_dropless(cfg.clone(), b, t, &p0)
        .expect("native Vulkan adapter and supported micro graph");
    gpu.desc_updates.set(false);
    let (u, bias) = routed_descriptors(&cfg);
    gpu.set_desc(&[("desc.u".into(), u), ("desc.bias".into(), bias)]);

    let tokens: Vec<u32> = lcg_vec(0x1001, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    let targets: Vec<u32> = lcg_vec(0x1002, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();

    // The first pass records the piecewise custom routes.  Holding those
    // assignments fixed makes the central differences test the differentiable
    // graph rather than an unrelated argmin tie crossing.
    let (loss0, grad_norm, _) = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1.0);
    assert!(loss0.is_finite() && grad_norm.is_finite());
    gpu.route_frozen.set(true);
    let grads = gpu.grads_host();
    assert!(grads.iter().all(|x| x.is_finite()));

    let mut worst = 0.0f64;
    let mut checked = 0usize;
    let mut active_tensors = 0usize;
    let mut zero_tensors = 0usize;
    for (name, off, n) in &layout.names {
        if *n == 0 {
            continue;
        }
        let picks = active_witnesses(&grads, *off, *n);
        if picks.is_empty() {
            zero_tensors += 1;
            eprintln!("{name:<32} active=0 (all sampled gradients below cutoff)");
            continue;
        }
        active_tensors += 1;
        let width = picks.len();
        // Align the finite-difference direction with the observed gradient
        // signs.  This avoids a weak near-zero directional projection while
        // remaining a genuine multi-coordinate central difference.
        let inv_width = 1.0 / (width as f64).sqrt();
        let delta: Vec<f64> = picks
            .iter()
            .map(|&i| {
                if grads[*off + i].is_sign_negative() {
                    -inv_width
                } else {
                    inv_width
                }
            })
            .collect();
        let analytic: f64 = picks
            .iter()
            .zip(&delta)
            .map(|(&i, &d)| grads[*off + i] as f64 * d)
            .sum();
        assert!(analytic.is_finite() && analytic.abs() > 1e-6);
        // Keep perturbations below the old 5e-2 ceiling.  The target loss
        // movement is set in f64, but eval_loss itself is the native f32 GPU
        // loss; eps/2 convergence separates f32 quantisation from truncation.
        let eps = (2e-3 / analytic.abs()).clamp(5e-4, 5e-2);
        let mut eval_fd = |step: f64| -> f64 {
            let mut pp = p0.clone();
            for (&i, &d) in picks.iter().zip(&delta) {
                pp[*off + i] = (p0[*off + i] as f64 + step * d) as f32;
            }
            gpu.set_params(&pp);
            gpu.eval_loss(&tokens, &targets);
            let lp = gpu.read_loss_f64();
            for (&i, &d) in picks.iter().zip(&delta) {
                pp[*off + i] = (p0[*off + i] as f64 - step * d) as f32;
            }
            gpu.set_params(&pp);
            gpu.eval_loss(&tokens, &targets);
            let lm = gpu.read_loss_f64();
            (lp - lm) / (2.0 * step)
        };
        let fd = eval_fd(eps);
        let fd_half = eval_fd(eps * 0.5);
        let scale = analytic.abs().max(1e-4);
        let rel = (fd_half - analytic).abs() / scale;
        let conv = (fd - fd_half).abs() / scale;
        worst = worst.max(rel.max(conv));
        checked += 1;
        eprintln!(
            "{name:<32} active={width:<2} eps={eps:.2e} fd={fd_half:+.4e} analytic={analytic:+.4e} rel={rel:.3e} conv={conv:.3e}"
        );
        assert!(rel < 0.15, "{name}: finite-difference mismatch {rel:e}");
        assert!(
            conv < 0.15,
            "{name}: finite-difference not converged {conv:e}"
        );
    }
    gpu.set_params(&p0);
    assert!(
        checked == layout.names.iter().filter(|(_, _, n)| *n > 0).count()
            && checked >= 30
            && active_tensors == checked
            && zero_tensors == 0,
        "micro graph active witness coverage too small: checked={checked}, active_tensors={active_tensors}"
    );
    eprintln!(
        "whole-model Vulkan finite-difference: tensors={checked} active={active_tensors} zero={zero_tensors} worst={worst:.3e} (f32 GPU loss, f64 finite-difference arithmetic)"
    );
}
