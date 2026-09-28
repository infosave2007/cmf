#![cfg(all(feature = "vulkan", not(target_os = "macos")))]
use cortiq_embryo::model::{init_params, EmbryoCfg, EmbryoGpu, Layout};
#[test]
fn desc_tmp() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    let lay = Layout::new(&cfg);
    let p = init_params(&cfg, &lay, 29);
    let mut gpu = EmbryoGpu::new(cfg.clone(), 1, 64, &p).expect("gpu");
    let tok: Vec<u32> = (0..64).map(|i| (i * 17 % cfg.vocab) as u32).collect();
    let tgt: Vec<u32> = (0..64).map(|i| ((i * 17 + 1) % cfg.vocab) as u32).collect();
    let (l, g, ms) = gpu.train_step(&tok, &tgt, 1e-4, 0.01, 1.0);
    assert!(l.is_finite() && g.is_finite() && ms > 0.0);
}
