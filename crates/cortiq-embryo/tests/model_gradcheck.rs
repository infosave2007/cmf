//! Whole-graph gradcheck on the GPU: for every named tensor of a tiny
//! genome, the directional derivative of the loss along (a) the gradient
//! direction and (b) a random direction, by central differences of the
//! f32 forward, against g·δ from the hand-rolled backward.
#![cfg(target_os = "macos")]

use cortiq_embryo::metal::{Cmd, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, gauss_vec, init_params};
use cortiq_embryo::ops::lcg_vec;

#[test]
fn every_tensor_matches_finite_differences() {
    let Some(c) = ctx() else { return };
    let cfg = EmbryoCfg::tiny();
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 7);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    let tokens: Vec<u32> = lcg_vec(11, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    let targets: Vec<u32> = lcg_vec(12, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    // analytic gradient
    unsafe {
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
    }
    let cmd = Cmd::new(c);
    gpu.encode_fwd_bwd(&cmd);
    cmd.commit();
    let g = gpu.grads_host();
    let l0 = gpu.eval_loss(&tokens, &targets);
    eprintln!("loss {l0:.5} (ln V = {:.5})", (cfg.vocab as f64).ln());
    let mut worst_all = 0.0f64;
    let mut seed = 100u64;
    for (name, off, n) in &lay.names {
        if name.contains("hk.kappa") {
            // padded rows: only the real ones are trained; still checked below via full-slice δ (pad grads are 0 by construction)
        }
        let gs = &g[*off..*off + n];
        let gnorm = gs.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        for dir in 0..2 {
            seed += 1;
            let delta: Vec<f64> = if dir == 0 {
                gs.iter().map(|x| *x as f64 / gnorm.max(1e-30)).collect()
            } else {
                let r = gauss_vec(seed, *n);
                let rn = r.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                r.iter().map(|x| *x as f64 / rn).collect()
            };
            let analytic: f64 = gs.iter().zip(&delta).map(|(a, d)| *a as f64 * d).sum();
            // step so that the loss moves ~1e-2 along the gradient direction
            let eps = (2e-3 / gnorm.max(1e-6)).clamp(1e-3, 0.1);
            let mut pp = p0.clone();
            for i in 0..*n {
                pp[off + i] = (p0[off + i] as f64 + eps * delta[i]) as f32;
            }
            gpu.set_params(&pp);
            let lp = gpu.eval_loss(&tokens, &targets) as f64;
            for i in 0..*n {
                pp[off + i] = (p0[off + i] as f64 - eps * delta[i]) as f32;
            }
            gpu.set_params(&pp);
            let lm = gpu.eval_loss(&tokens, &targets) as f64;
            let fd = (lp - lm) / (2.0 * eps);
            let denom = gnorm.max(1e-6);
            let rel = (fd - analytic).abs() / denom;
            worst_all = worst_all.max(rel);
            eprintln!(
                "{name:<24} n={n:<7} |g|={gnorm:.3e} dir={} fd={fd:+.4e} an={analytic:+.4e} err/|g|={rel:.2e}",
                if dir == 0 { "grad" } else { "rand" }
            );
            assert!(rel < 3e-2, "{name} dir {dir}: fd {fd} vs analytic {analytic} (|g| {gnorm})");
        }
    }
    gpu.set_params(&p0);
    eprintln!("worst err/|g| = {worst_all:.2e}");
}
