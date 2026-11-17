//! Logit distillation into the hierarchical head (`birth --kd-from`,
//! `EmbryoGpu::enable_kd` + `kd_gather`): with the student as its own teacher
//! p_teacher = q, so the head gradient scale·(q − (1−α)·1ₜ − α·p) is
//! (1−α)·scale·(q − 1ₜ): the whole gradient scales by 1−α. The teacher runs
//! the batch in micro-batches of half its size, so the rows, the grouped rows
//! of both head levels and the 1/valid rescaling are all exercised; the
//! reported loss stays the plain CE.
//!
//! Runs on Metal (macOS), Vulkan and CUDA (`--features vulkan` / `cuda`);
//! skipped when no device exists.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Random tokens and targets; the first third of every row has no target.
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

#[test]
fn distilling_from_itself_scales_the_gradient_by_one_minus_alpha() {
    if cortiq_embryo::metal::ctx().is_none() {
        return;
    }
    let cfg = EmbryoCfg::tiny();
    assert!(cfg.head_clusters > 0);
    let (b, t) = (4usize, 64usize);
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 7);
    let mut student = EmbryoGpu::new(cfg.clone(), b, t, &params).unwrap();
    let teacher = EmbryoGpu::new(cfg.clone(), b / 2, t, &params).unwrap();
    let (tk, tg) = batch(&cfg, b, t, 3);
    // lr 0 and no decay: the parameters stay, every step sees the same function
    let (l0, g0, _) = student.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    assert!(g0 > 1e-3, "the plain gradient is too small to compare: {g0}");
    for &alpha in &[1.0f32, 0.5, 0.25] {
        student.enable_kd(alpha, &teacher);
        let tl = student.kd_gather(&teacher, &tk, &tg, 0);
        assert!((tl - l0).abs() <= 1e-4 * l0, "the teacher (the same weights) has another CE: {tl} vs {l0}");
        let (l, g, _) = student.train_step(&tk, &tg, 0.0, 0.0, 1e9);
        eprintln!("alpha {alpha}: CE {l:.6} (plain {l0:.6}), |g| {g:.6e} (plain {g0:.6e}, ratio {:.6})", g / g0);
        assert!((l - l0).abs() <= 1e-5 * l0, "the logged loss is not the CE: {l} vs {l0}");
        let want = 1.0 - alpha;
        assert!(
            (g / g0 - want).abs() <= 2e-3,
            "alpha {alpha}: |g| {g} is {} of the plain gradient, not {want}",
            g / g0
        );
    }
    // a step without kd_gather is a plain step again (the buffers disarm after one use)
    let (_, g, _) = student.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    assert!((g / g0 - 1.0).abs() <= 1e-4, "stale distillation buffers: {} of the plain gradient", g / g0);
    // the first positions of every window keep the plain CE: with alpha 1 only they carry a gradient
    student.enable_kd(1.0, &teacher);
    let _ = student.kd_gather(&teacher, &tk, &tg, t / 2);
    let (_, g, _) = student.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    eprintln!("alpha 1, plain CE before position {}: |g| {:.6} of the plain gradient", t / 2, g / g0);
    assert!(g / g0 > 0.05 && g / g0 < 0.95, "skip: {} of the plain gradient", g / g0);
}
