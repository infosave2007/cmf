#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};

/// This is deliberately a required native-GPU gate: a software adapter or a
/// missing Vulkan device must fail rather than silently exercise a CPU path.
#[test]
fn tiny_forward_backward_adamw_is_device_resident() {
    let cfg = EmbryoCfg::tiny();
    let layout = Layout::new(&cfg);
    let params = init_params(&cfg, &layout, 7);
    let mut gpu = EmbryoGpu::new(cfg, 1, 64, &params).expect("native Vulkan adapter");
    gpu.desc_updates.set(false);
    let tokens: Vec<u32> = (0..64).map(|i| (i * 17 % 4096) as u32).collect();
    let targets: Vec<u32> = (0..64).map(|i| ((i * 17 + 1) % 4096) as u32).collect();
    let before = gpu.p.to_vec();
    let eval = gpu.eval_loss(&tokens, &targets);
    assert!(eval.is_finite() && eval.abs() < 100.0, "eval={eval}");
    let (loss, grad, ms) = gpu.train_step(&tokens, &targets, 1e-4, 0.01, 1.0);
    assert!(loss.is_finite(), "loss={loss}");
    assert!(grad.is_finite(), "grad={grad}");
    assert!(ms.is_finite() && ms > 0.0, "gpu_ms={ms}");
    let after = gpu.p.to_vec();
    assert!(after.iter().all(|x| x.is_finite()));
    assert!(
        after
            .iter()
            .zip(before)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
    assert!(gpu.m.to_vec().iter().all(|x| x.is_finite()));
    assert!(gpu.v.to_vec().iter().all(|x| x.is_finite()));
}

#[test]
fn response_only_all_masked_batch_is_finite_and_a_noop() {
    let cfg = EmbryoCfg::tiny();
    let layout = Layout::new(&cfg);
    let params = init_params(&cfg, &layout, 19);
    let mut gpu = EmbryoGpu::new(cfg, 1, 64, &params).expect("native Vulkan adapter");
    gpu.desc_updates.set(false);
    let tokens: Vec<u32> = (0..64).map(|i| (i * 23 % 4096) as u32).collect();
    let targets = vec![u32::MAX; 64];
    let before_p = gpu.p.to_vec();
    let before_m = gpu.m.to_vec();
    let before_v = gpu.v.to_vec();
    let before_step = gpu.step;

    let eval = gpu.eval_loss(&tokens, &targets);
    assert_eq!(eval, 0.0, "all-masked eval must be zero, got {eval}");
    let (loss, grad, ms) = gpu.train_step(&tokens, &targets, 1e-4, 0.01, 1.0);
    assert_eq!(loss, 0.0, "all-masked train loss must be zero, got {loss}");
    assert_eq!(grad, 0.0, "all-masked gradient norm must be zero, got {grad}");
    assert!(ms.is_finite() && ms > 0.0, "gpu_ms={ms}");
    assert_eq!(gpu.step, before_step, "all-masked batch advanced AdamW clock");
    assert_eq!(gpu.p.to_vec(), before_p, "all-masked batch changed params");
    assert_eq!(gpu.m.to_vec(), before_m, "all-masked batch changed Adam moments");
    assert_eq!(gpu.v.to_vec(), before_v, "all-masked batch changed Adam moments");
}

#[test]
fn response_only_partial_mask_uses_answer_rows_and_updates() {
    let cfg = EmbryoCfg::tiny();
    let layout = Layout::new(&cfg);
    let params = init_params(&cfg, &layout, 23);
    let mut gpu = EmbryoGpu::new(cfg, 1, 64, &params).expect("native Vulkan adapter");
    gpu.desc_updates.set(false);
    let tokens: Vec<u32> = (0..64).map(|i| (i * 29 % 4096) as u32).collect();
    let mut targets = vec![u32::MAX; 64];
    targets[17] = (tokens[17] + 1) % 4096;
    let before = gpu.p.to_vec();
    let eval = gpu.eval_loss(&tokens, &targets);
    assert!(eval.is_finite() && eval > 0.0, "partial-mask eval={eval}");
    assert_eq!(gpu.head_valid.get(), 1);
    let (loss, grad, ms) = gpu.train_step(&tokens, &targets, 1e-4, 0.01, 1.0);
    assert!(loss.is_finite() && loss > 0.0, "partial-mask loss={loss}");
    assert!(grad.is_finite() && grad > 0.0, "partial-mask grad={grad}");
    assert!(ms.is_finite() && ms > 0.0, "gpu_ms={ms}");
    assert_eq!(gpu.step, 1);
    assert!(gpu
        .p
        .to_vec()
        .iter()
        .zip(before)
        .any(|(a, b)| a.to_bits() != b.to_bits()));
}

#[test]
fn response_only_partial_mask_is_valid_token_normalized_and_finite_difference() {
    // Duplicate the same independent sequence in two batch rows.  With one
    // valid answer row and then two, valid-token normalization must keep the
    // mean loss unchanged; the old B*T denominator made the one-row result
    // half as large.  The same one-row mask is then checked against the
    // native backward with a route-frozen central difference.
    let cfg = EmbryoCfg::tiny();
    let layout = Layout::new(&cfg);
    let params = init_params(&cfg, &layout, 29);
    let (b, t) = (2usize, 64usize);
    let mut gpu = EmbryoGpu::new(cfg, b, t, &params).expect("native Vulkan adapter");
    gpu.desc_updates.set(false);
    let row: Vec<u32> = (0..t).map(|i| (i * 31 % 4096) as u32).collect();
    let mut tokens = row.clone();
    tokens.extend_from_slice(&row);
    let answer = (row[17] + 1) % 4096;
    let mut one = vec![u32::MAX; b * t];
    one[17] = answer;
    let mut two = one.clone();
    two[t + 17] = answer;

    let loss_one = gpu.eval_loss(&tokens, &one);
    let loss_two = gpu.eval_loss(&tokens, &two);
    assert!(loss_one.is_finite() && loss_one > 0.0);
    assert!(loss_two.is_finite() && loss_two > 0.0);
    let loss_scale = loss_one.abs().max(loss_two.abs()).max(1e-6);
    assert!(
        (loss_one - loss_two).abs() / loss_scale < 5e-4,
        "valid-token mean changed with duplicated row: one={loss_one} two={loss_two}"
    );

    let base = gpu.params_host();
    let (loss, grad_norm, ms) = gpu.train_step(&tokens, &one, 0.0, 0.0, 1.0);
    assert!(loss.is_finite() && loss > 0.0);
    assert!(grad_norm.is_finite() && grad_norm > 0.0);
    assert!(ms.is_finite() && ms > 0.0);
    assert_eq!(gpu.head_valid.get(), 1);
    let grads = gpu.grads_host();
    let (index, &analytic_f32) = grads
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .expect("finite Vulkan gradient")
        })
        .expect("non-empty parameter gradient");
    let analytic = analytic_f32 as f64;
    assert!(analytic.is_finite() && analytic.abs() > 1e-6);

    // Keep the discrete route fixed while perturbing one arena coordinate;
    // read_loss_f64 preserves the small signal lost by the public f32 API.
    gpu.route_frozen.set(true);
    let eps = (2e-3 / analytic.abs()).clamp(5e-4, 5e-2);
    let mut plus = base.clone();
    plus[index] = (base[index] as f64 + eps) as f32;
    gpu.set_params(&plus);
    gpu.eval_loss(&tokens, &one);
    let lp = gpu.read_loss_f64();
    let mut minus = base.clone();
    minus[index] = (base[index] as f64 - eps) as f32;
    gpu.set_params(&minus);
    gpu.eval_loss(&tokens, &one);
    let lm = gpu.read_loss_f64();
    let fd = (lp - lm) / (2.0 * eps);
    let rel = (fd - analytic).abs() / analytic.abs().max(1e-4);
    assert!(
        rel < 0.15,
        "partial valid-token finite difference mismatch: fd={fd} analytic={analytic} rel={rel}"
    );
    gpu.set_params(&base);
}

#[test]
fn terminal_sft_final_holdout_eval() {
    // This is deliberately evaluation-only: the selected terminal checkpoint
    // is loaded with frozen descriptors and no optimizer step is issued.
    let ckpt_path = std::env::var("CMF_SFT_FINAL_CKPT").expect("CMF_SFT_FINAL_CKPT");
    let final_path = std::env::var("CMF_SFT_FINAL_SHARD").expect("CMF_SFT_FINAL_SHARD");
    let batch = std::env::var("CMF_SFT_FINAL_BATCH")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(10usize);
    let max_batches = std::env::var("CMF_SFT_FINAL_BATCHES")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(8usize);
    let ck = cortiq_embryo::train::load_checkpoint(std::path::Path::new(&ckpt_path))
        .expect("load terminal SFT checkpoint");
    let shard = cortiq_embryo::sft::SftShard::load(std::path::Path::new(&final_path))
        .expect("load terminal SFT shard");
    assert_eq!(shard.seq, 512, "terminal SFT shard sequence");
    assert!(shard.records >= batch && shard.valid_tokens() > 0);
    let lay = Layout::new(&ck.cfg);
    assert_eq!(ck.params.len(), lay.total);
    let gpu = EmbryoGpu::new_eval_dropless(ck.cfg.clone(), batch, shard.seq, &ck.params)
        .expect("native Vulkan adapter");
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false);
    let batches = max_batches.min((shard.records / batch).max(1));
    let mut tokens = Vec::new();
    let mut targets = Vec::new();
    let nll = (0..batches)
        .map(|i| {
            shard.fixed_batch(batch, i, &mut tokens, &mut targets);
            let value = gpu.eval_loss(&tokens, &targets);
            assert!(value.is_finite() && value > 0.0);
            value as f64
        })
        .sum::<f64>()
        / batches as f64;
    println!(
        "terminal_sft_final_response_nll={nll:.6} batches={batches} records={} valid_tokens={}",
        shard.records,
        shard.valid_tokens()
    );
}
