//! Complete resident SUMSQ reduction, including short/tail tiles and offsets.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};

#[test]
fn sumsq_covers_every_element_and_preserves_offsets() {
    let c = ctx().expect("native Vulkan adapter");
    for &(n, off, poff) in &[
        (1usize, 3usize, 2usize),
        (255, 5, 1),
        (256, 7, 3),
        (257, 11, 4),
        (4096, 13, 5),
        (4097, 17, 6),
    ] {
        let xlen = off + n + 9;
        let x_host: Vec<f32> = (0..xlen)
            .map(|i| (((i * 17 + 5) % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let x = GBuf::from_slice(c, &x_host);
        let groups = n.div_ceil(256);
        let part = GBuf::zeros(c, poff + groups + 3);
        let cmd = Cmd::new(c);
        assert_eq!(cmd.sumsq_at(&x, off, n, &part, poff), groups);
        cmd.commit();
        let got = part.to_vec();
        let mut worst = 0.0f64;
        for g in 0..groups {
            let begin = off + g * 256;
            let end = (begin + 256).min(off + n);
            let want = x_host[begin..end]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>();
            let err = (got[poff + g] as f64 - want).abs();
            worst = worst.max(err);
            assert!(
                err <= 2.0e-5 * want.abs().max(1.0),
                "n={n} off={off} poff={poff} group={g}: got={} want={want} err={err}",
                got[poff + g]
            );
        }
        assert!(
            got[..poff].iter().all(|&x| x == 0.0) && got[poff + groups..].iter().all(|&x| x == 0.0),
            "SUMSQ wrote outside part offset (n={n}, off={off}, poff={poff})"
        );
        eprintln!("SUMSQ n={n} off={off} poff={poff} groups={groups} worst={worst:.3e}");
    }

    // Arena-sized reduction: the host clamps `groups` to 4096, so every
    // element past 4096*256 = 1_048_576 is reachable only through the
    // grid-stride over tiles.  Before that stride existed the printed
    // gradient norm of the 56M-parameter genome covered 1.87% of the arena.
    {
        let (n, off, poff) = (2_500_000usize, 5usize, 2usize);
        let x_host: Vec<f32> = (0..off + n + 3)
            .map(|i| (((i * 7 + 3) % 23) as f32 - 11.0) * 0.0625)
            .collect();
        let want = x_host[off..off + n]
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>();
        let x = GBuf::from_slice(c, &x_host);
        let part = GBuf::zeros(c, poff + 4096 + 2);
        let cmd = Cmd::new(c);
        let groups = cmd.sumsq_at(&x, off, n, &part, poff);
        assert_eq!(groups, 4096, "host clamp changed: groups={groups}");
        cmd.commit();
        let got: f64 = part.to_vec()[poff..poff + groups]
            .iter()
            .map(|&v| v as f64)
            .sum();
        let rel = (got - want).abs() / want;
        assert!(rel <= 1.0e-5, "arena SUMSQ n={n}: got={got} want={want} rel={rel:.3e}");
        eprintln!("SUMSQ arena n={n} groups={groups} rel={rel:.3e}");

        // A NaN in the LAST element of the arena — the region the old
        // kernel never read — must reach the reduction.
        let mut poisoned = x_host.clone();
        poisoned[off + n - 1] = f32::NAN;
        let x = GBuf::from_slice(c, &poisoned);
        let part = GBuf::zeros(c, poff + 4096 + 2);
        let cmd = Cmd::new(c);
        let groups = cmd.sumsq_at(&x, off, n, &part, poff);
        cmd.commit();
        let total: f64 = part.to_vec()[poff..poff + groups]
            .iter()
            .map(|&v| v as f64)
            .sum();
        assert!(total.is_nan(), "SUMSQ missed a NaN at arena index n-1={} (total={total})", n - 1);
    }

    // A non-finite value in a formerly skipped lane must reach the reduction
    // result.  This is deliberately not an aggregate CPU-side witness: the
    // train path relies on the device reduction to reject the update before
    // AdamW advances its clock.
    let off = 19usize;
    let n = 257usize;
    let poff = 3usize;
    let mut x_host = vec![1.0f32; off + n + 1];
    x_host[off + 255] = f32::NAN;
    let x = GBuf::from_slice(c, &x_host);
    let groups = n.div_ceil(256);
    let part = GBuf::zeros(c, poff + groups + 1);
    let cmd = Cmd::new(c);
    assert_eq!(cmd.sumsq_at(&x, off, n, &part, poff), groups);
    cmd.commit();
    let got = part.to_vec();
    assert!(
        got[poff].is_nan(),
        "SUMSQ omitted NaN in the interior of a 256-element tile: {:?}",
        &got[poff..poff + groups]
    );
}

#[test]
fn nonfinite_reduction_rejects_before_adamw() {
    let cfg = EmbryoCfg::tiny();
    let lay = Layout::new(&cfg);
    let mut params = init_params(&cfg, &lay, 0x51_7f);
    // A single parameter poisons a complete forward/backward lane.  The
    // reduction must observe it and train_step must panic before incrementing
    // the optimizer clock or dispatching AdamW.
    params[lay.embed] = f32::NAN;
    let mut gpu = EmbryoGpu::new(cfg.clone(), 1, 64, &params).expect("native Vulkan adapter");
    let before = gpu.params_host();
    let tokens = vec![1u32; 64];
    let targets = vec![2u32; 64];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = gpu.train_step(&tokens, &targets, 1.0e-5, 0.0, 1.0);
    }));
    assert!(result.is_err(), "non-finite train step was accepted");
    assert_eq!(gpu.step, 0, "optimizer clock advanced after rejection");
    let after = gpu.params_host();
    assert_eq!(before.len(), after.len());
    assert!(
        before
            .iter()
            .zip(after.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "AdamW changed parameters before non-finite rejection"
    );
}
