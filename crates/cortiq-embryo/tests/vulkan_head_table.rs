//! S6c item 3b: the Vulkan training head's per-cluster GEMM loops (up to
//! `head_clusters` dispatches per GEMM, three GEMMs) as one table-batched
//! dispatch each (`Cmd::gemm_table`). Gate: loss and every gradient
//! bit-identical to the loop path on the tiny genome (same per-element
//! sums, only the dispatch shape changes), and the encode time of the head
//! at the B8/T512 Embryo-0 geometry for both paths.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, HEAD_TABLE, Layout, init_params};
use cortiq_embryo::ops::lcg_vec;
use std::sync::atomic::Ordering;

fn toks(seed: u64, n: usize, vocab: usize) -> Vec<u32> {
    lcg_vec(seed, n)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * vocab as f32) as u32 % vocab as u32)
        .collect()
}

#[test]
fn table_batched_head_is_bit_identical_to_the_loops() {
    let _ = ctx().expect("native Vulkan adapter is required");
    for variant in 0..3 {
        let mut cfg = EmbryoCfg::tiny();
        if variant == 1 {
            cfg.anchor_every = 1; // all anchors: a different embedding gradient path mix
        }
        assert!(cfg.head_clusters > 0);
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 7);
        let tokens = toks(1000, m, cfg.vocab);
        // targets with repeats so several clusters get more than one row;
        // variant 2: natural-text skew — 70% of the rows in cluster 0, so
        // the widest group is many tiles taller than the typical one
        let targets: Vec<u32> = toks(2000, m, cfg.vocab)
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if variant == 2 && i % 10 < 7 {
                    x % 64
                } else if i % 3 == 0 {
                    x % 700
                } else {
                    *x
                }
            })
            .collect();
        HEAD_TABLE.store(false, Ordering::Relaxed);
        let mut loops = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("loops");
        loops.desc_updates.set(false);
        let (l0, g0, _) = loops.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
        let grads0 = loops.grads_host();
        HEAD_TABLE.store(true, Ordering::Relaxed);
        let mut table = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("table");
        table.desc_updates.set(false);
        let (l1, g1, _) = table.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
        let grads1 = table.grads_host();
        let pads: Vec<usize> = table.head_groups.borrow().iter().map(|g| g.2).collect();
        eprintln!(
            "variant {variant}: loss 0x{:08x} vs 0x{:08x} ({l1:.5} vs {l0:.5}), |g| 0x{:08x} vs 0x{:08x}; groups {} pad max {} min {}",
            l1.to_bits(),
            l0.to_bits(),
            g1.to_bits(),
            g0.to_bits(),
            pads.len(),
            pads.iter().max().unwrap_or(&0),
            pads.iter().min().unwrap_or(&0)
        );
        assert_eq!(l1.to_bits(), l0.to_bits(), "loss bits");
        assert_eq!(g1.to_bits(), g0.to_bits(), "grad norm bits");
        for (name, off, n) in &lay.names {
            assert_eq!(&grads1[*off..*off + n], &grads0[*off..*off + n], "{name} gradients differ");
        }
    }
}

#[test]
fn table_batched_head_timing_b8_t512() {
    let c = ctx().expect("native Vulkan adapter is required");
    let cfg = EmbryoCfg::embryo0();
    let (b, t) = (8usize, 512usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 1);
    let tokens = toks(7001, m, cfg.vocab);
    let targets = toks(7002, m, cfg.vocab);
    let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("embryo0");
    gpu.desc_updates.set(false);
    unsafe {
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
    }
    gpu.prepare_head(&targets);
    let groups = gpu.head_groups.borrow().len();
    let time = |on: bool| -> f64 {
        HEAD_TABLE.store(on, Ordering::Relaxed);
        let mut v = Vec::new();
        for _ in 0..4 {
            let cmd = Cmd::new(c);
            gpu.encode_head(&cmd, true);
            v.push(cmd.commit());
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let t_loops = time(false);
    let t_table = time(true);
    eprintln!("B8/T512 head fwd+bwd ({groups} cluster groups): loops {t_loops:.1} ms, table-batched {t_table:.1} ms");
    HEAD_TABLE.store(true, Ordering::Relaxed);
}
