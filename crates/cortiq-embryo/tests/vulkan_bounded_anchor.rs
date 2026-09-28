//! Vulkan twin of tests/anchor_bounded.rs for the bounded anchor
//! `swa_sink_v1` (WGSL op 19 band+sink softmax, op 20 backward over the
//! `[T, LD]` rows, op 52 sink gradient fold, and the whole attention core
//! through `anchor_core_probe`):
//!  - kernels vs the f64 reference (≤ 1e-6 abs) and vs the Metal outputs
//!    of the same fixture (bits captured on the M4: max|Δ| ≤ 1e-6 is the
//!    Metal ↔ WGSL gate of the contract);
//!  - the core vs the f64 operator reference (≤ 1e-5 rel, every gradient);
//!  - `window ≥ T` without sinks bit-identical to the legacy path.
//! A missing/non-NVIDIA adapter fails the process (repo rule for the
//! native Vulkan gates).
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::{
    BoundedAnchorDims, SOFTMAX_FIXTURE, bounded_anchor_ref, lcg_vec, softmax_reference,
    softmax_samples, softmax_scores, softmax_upstream,
};

fn toks(seed: u64, n: usize, vocab: usize) -> Vec<u32> {
    lcg_vec(seed, n)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * vocab as f32) as u32 % vocab as u32)
        .collect()
}

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w).abs())
        .fold(0.0, f64::max)
        / scale
}

/// Metal outputs of the softmax fixture at `softmax_samples` (bits), from
/// `softmax_kernels_match_f64_reference_and_print_fingerprint` on the M4.
const METAL_P_BITS: [u32; 16] = [
    0x3e8475a9, 0x3cf2bfe9, 0x3e7d85f0, 0x00000000, 0x3cc55055, 0x3d7ff12a, 0x3e791939, 0x00000000,
    0x3dcc6300, 0x3ec6ff4e, 0x3acfae60, 0x00000000, 0x3d84570a, 0x3bee9f70, 0x3ca867e7, 0x00000000,
];
const METAL_DS_BITS: [u32; 16] = [
    0xbc2e6eec, 0xbd4a9ec5, 0xbabfa928, 0x80000000, 0xbc5d8245, 0xbc237ece, 0xbd6e7442, 0x00000000,
    0x3d1b5ff4, 0xbe581f4d, 0x39b68cd6, 0x80000000, 0x3b053533, 0x3c154e2e, 0x3c69a6a8, 0x00000000,
];
/// Metal `sink_grad_accum` outputs at `x == 0` of every (g, s) row.
const METAL_ACC_BITS: [u32; 6] = [
    0x3f3ed28a, 0x3f654879, 0x3f40af53, 0x3f81002e, 0x3dd00a2e, 0xbede2786,
];

#[test]
fn wgsl_softmax_kernels_match_reference_and_metal() {
    let c = ctx().expect("native Vulkan adapter is required");
    let f = &SOFTMAX_FIXTURE;
    let scores = softmax_scores(f);
    let upstream = softmax_upstream(f);
    let (want_p, want_ds) = softmax_reference(f, &scores, &upstream);
    let p = GBuf::from_slice(c, &scores);
    let cmd = Cmd::new(c);
    cmd.banded_softmax_blocks(&p, f.off, f.t, f.ld, f.sink, f.sink_pad, f.window, f.blocks);
    cmd.commit();
    let got_p = p.to_vec();
    assert_eq!(got_p[..f.off], scores[..f.off], "offset sentinel modified");
    let e_p = got_p[f.off..]
        .iter()
        .zip(&want_p[f.off..])
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0, f64::max);
    eprintln!("op 19 band+sink softmax: max|Δ| vs f64 = {e_p:.2e}");
    assert!(e_p < 1e-6);
    for block in 0..f.blocks {
        for row in 0..f.t {
            let base = f.off + block * f.t * f.ld + row * f.ld;
            let s: f32 = got_p[base..base + f.ld].iter().sum();
            assert!((s - 1.0).abs() < 2e-6, "row sum {s} at {block}:{row}");
        }
    }
    let d = GBuf::from_slice(c, &upstream);
    let cmd = Cmd::new(c);
    cmd.softmax_bwd_blocks_ld(&p, f.off, &d, f.off, f.t, f.ld, f.blocks);
    cmd.commit();
    let got_ds = d.to_vec();
    assert_eq!(got_ds[..f.off], upstream[..f.off]);
    let e_ds = got_ds[f.off..]
        .iter()
        .zip(&want_ds[f.off..])
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0, f64::max);
    eprintln!("op 20 softmax bwd [T,LD]: max|Δ| vs f64 = {e_ds:.2e}");
    assert!(e_ds < 1e-6);
    let mut dm_p = 0.0f32;
    let mut dm_ds = 0.0f32;
    for (i, &(block, row, col)) in softmax_samples(f).iter().enumerate() {
        let idx = f.off + block * f.t * f.ld + row * f.ld + col;
        dm_p = dm_p.max((got_p[idx] - f32::from_bits(METAL_P_BITS[i])).abs());
        dm_ds = dm_ds.max((got_ds[idx] - f32::from_bits(METAL_DS_BITS[i])).abs());
    }
    eprintln!("Metal ↔ WGSL at 16 sampled cells: max|ΔP| = {dm_p:.2e}, max|ΔdS| = {dm_ds:.2e}");
    assert!(dm_p <= 1e-6 && dm_ds <= 1e-6, "Metal/WGSL disagreement");
    // op 52: sink gradient fold
    let (qh, kvh, sink, hd, dst_off) = (4usize, 2usize, 3usize, 32usize, 5usize);
    let group = qh / kvh;
    let src = lcg_vec(4444, qh * 64 * hd);
    let dst0 = lcg_vec(4545, dst_off + kvh * sink * hd);
    let sb = GBuf::from_slice(c, &src);
    let db = GBuf::from_slice(c, &dst0);
    let cmd = Cmd::new(c);
    cmd.sink_grad_accum(&sb, &db, dst_off, 1, kvh, group, sink, hd, 0.125);
    cmd.commit();
    let got = db.to_vec();
    assert_eq!(got[..dst_off], dst0[..dst_off]);
    let mut e = 0.0f64;
    let mut dm = 0.0f32;
    for g in 0..kvh {
        for s in 0..sink {
            for x in 0..hd {
                let mut acc = 0.0f64;
                for j in 0..group {
                    acc += src[((g * group + j) * 64 + s) * hd + x] as f64;
                }
                let i = dst_off + (g * sink + s) * hd + x;
                let want = dst0[i] as f64 + 0.125 * acc;
                e = e.max((got[i] as f64 - want).abs());
                if x == 0 {
                    dm = dm.max((got[i] - f32::from_bits(METAL_ACC_BITS[g * sink + s])).abs());
                }
            }
        }
    }
    eprintln!("op 52 sink_grad_accum: max|Δ| vs f64 = {e:.2e}, vs Metal = {dm:.2e}");
    assert!(e < 1e-6 && dm <= 1e-6);
}

fn first_anchor(cfg: &EmbryoCfg) -> usize {
    (0..cfg.layers).find(|&l| cfg.is_anchor(l)).expect("an anchor layer")
}

fn probe_vs_reference(cfg: &EmbryoCfg, b: usize, t: usize, w: usize, seed: u64, tol: f64) -> f64 {
    let lay = Layout::new(cfg);
    let mut p0 = init_params(cfg, &lay, 7);
    let l = first_anchor(cfg);
    let (qh, kvh, hd, h) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd, cfg.hidden);
    let m = b * t;
    let ns = kvh * cfg.anchor_sink * hd;
    let q = lcg_vec(seed + 1, m * qh * hd);
    let k = lcg_vec(seed + 2, m * kvh * hd);
    let v = lcg_vec(seed + 3, m * kvh * hd);
    let dy = lcg_vec(seed + 4, m * h);
    let sink_k = lcg_vec(seed + 5, ns);
    let sink_v = lcg_vec(seed + 6, ns);
    let wo: Vec<f32> = lcg_vec(seed + 7, h * qh * hd).iter().map(|x| x * 0.125).collect();
    let (wo_off, sk_off, sv_off) = match &lay.layers[l] {
        cortiq_embryo::model::LayerOffs::Anchor {
            wo, sink_k, sink_v, ..
        } => (*wo, *sink_k, *sink_v),
        _ => unreachable!(),
    };
    p0[wo_off..wo_off + wo.len()].copy_from_slice(&wo);
    if cfg.anchor_sink > 0 {
        p0[sk_off..sk_off + ns].copy_from_slice(&sink_k);
        p0[sv_off..sv_off + ns].copy_from_slice(&sink_v);
    }
    let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("vulkan model");
    let got = gpu.anchor_core_probe(l, &q, &k, &v, &dy, w);
    let d = BoundedAnchorDims {
        b,
        t,
        qh,
        kvh,
        hd,
        h,
        s: cfg.anchor_sink,
        w,
        base: cfg.rope_base as f64,
    };
    let want = bounded_anchor_ref(
        &d,
        &to64(&q),
        &to64(&k),
        &to64(&v),
        &to64(&sink_k),
        &to64(&sink_v),
        &to64(&wo),
        Some(&to64(&dy)),
    );
    let mut worst = 0.0f64;
    let mut check = |name: &str, g: &[f32], r: &[f64]| {
        let e = rel_err(g, r);
        eprintln!("  S={} W={w}: {name:<8} max|Δ|/max|ref| = {e:.2e}", cfg.anchor_sink);
        assert!(e <= tol, "{name}: rel err {e:.3e} > {tol:e}");
        worst = worst.max(e);
    };
    check("y", &got.y, &want.y);
    check("dq", &got.dq, &want.dq);
    check("dk", &got.dk, &want.dk);
    check("dv", &got.dv, &want.dv);
    check("dWo", &got.dwo, &want.dwo);
    if cfg.anchor_sink > 0 {
        check("dsink_k", &got.dsink_k, &want.dsink_k);
        check("dsink_v", &got.dsink_v, &want.dsink_v);
    }
    worst
}

#[test]
fn vulkan_bounded_core_matches_f64_reference() {
    let _ = ctx().expect("native Vulkan adapter is required");
    let (b, t) = (2usize, 64usize);
    let mut worst = 0.0f64;
    for (s, w) in [(2usize, 3usize), (4, 16), (0, 8), (4, 64)] {
        let mut cfg = EmbryoCfg::tiny();
        cfg.anchor_window = w;
        cfg.anchor_sink = s;
        eprintln!("=== vulkan bounded core S={s} W={w} ===");
        worst = worst.max(probe_vs_reference(&cfg, b, t, w, 500 + s as u64 * 10 + w as u64, 1e-5));
    }
    eprintln!("=== vulkan legacy core (W=0) ===");
    worst = worst.max(probe_vs_reference(&EmbryoCfg::tiny(), b, t, 0, 600, 1e-5));
    eprintln!("worst rel err over all cases: {worst:.2e}");
}

#[test]
fn vulkan_full_window_without_sinks_matches_legacy_bits() {
    let _ = ctx().expect("native Vulkan adapter is required");
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let legacy = EmbryoCfg::tiny();
    let lay = Layout::new(&legacy);
    let p0 = init_params(&legacy, &lay, 7);
    let tokens = toks(1000, m, legacy.vocab);
    let targets = toks(2000, m, legacy.vocab);
    let mut lhs = EmbryoGpu::new(legacy.clone(), b, t, &p0).expect("legacy");
    lhs.desc_updates.set(false);
    let (ll, gl, _) = lhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    let gll = lhs.grads_host();
    for w in [t, cortiq_embryo::model::MAX_SINK_PLUS_WINDOW] {
        let mut cfg = legacy.clone();
        cfg.anchor_window = w;
        let mut rhs = EmbryoGpu::new(cfg, b, t, &p0).expect("bounded");
        rhs.desc_updates.set(false);
        let (lr, gr, _) = rhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
        eprintln!("W={w}: loss 0x{:08x} vs legacy 0x{:08x}, |g| 0x{:08x} vs 0x{:08x}", lr.to_bits(), ll.to_bits(), gr.to_bits(), gl.to_bits());
        assert_eq!(lr.to_bits(), ll.to_bits(), "W={w} loss bits");
        assert_eq!(gr.to_bits(), gl.to_bits(), "W={w} grad norm bits");
        assert_eq!(rhs.grads_host(), gll, "W={w} gradients");
    }
    // a bounded step with sinks is finite and different from the legacy one
    let mut cfg = legacy.clone();
    cfg.anchor_window = 16;
    cfg.anchor_sink = 4;
    let lay_b = Layout::new(&cfg);
    let mut p1 = init_params(&cfg, &lay_b, 7);
    for (name, off, n) in &lay_b.names {
        if name.contains("attn.sink_") {
            p1[*off..*off + n].copy_from_slice(&lcg_vec(*off as u64, *n));
        }
    }
    let mut bnd = EmbryoGpu::new(cfg, b, t, &p1).expect("bounded with sinks");
    bnd.desc_updates.set(false);
    let (lb, gb, _) = bnd.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
    eprintln!("W=16 S=4: loss {lb:.5} |g| {gb:.4} (legacy loss {ll:.5})");
    assert!(lb.is_finite() && gb.is_finite());
    assert_ne!(lb.to_bits(), ll.to_bits());
}


/// S6c item 3a: the anchor backward batched over the sequences vs the
/// per-sequence dispatch shape, same process, Embryo-0 anchor geometry at
/// B8/T512 (legacy full causal and bounded W128/S4): gradients identical
/// to 1e-6 rel and both timings (median of 3 probes: fwd + bwd + readback).
#[test]
fn vulkan_anchor_backward_batched_vs_per_sequence() {
    let _ = ctx().expect("native Vulkan adapter is required");
    use cortiq_embryo::model::ANCHOR_BWD_BATCH;
    use std::sync::atomic::Ordering;
    let (b, t) = (8usize, 512usize);
    for (w, s) in [(0usize, 0usize), (128, 4)] {
        let mut cfg = EmbryoCfg::embryo0();
        cfg.anchor_window = w;
        cfg.anchor_sink = s;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 3);
        let l = first_anchor(&cfg);
        let (qh, kvh, hd, h) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd, cfg.hidden);
        let m = b * t;
        let inp_q = lcg_vec(701, m * qh * hd);
        let inp_k = lcg_vec(702, m * kvh * hd);
        let inp_v = lcg_vec(703, m * kvh * hd);
        let inp_dy = lcg_vec(704, m * h);
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("embryo0");
        let mut res = Vec::new();
        for batched in [true, false] {
            ANCHOR_BWD_BATCH.store(batched, Ordering::Relaxed);
            let mut ms = Vec::new();
            let mut last = None;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                last = Some(gpu.anchor_core_probe(l, &inp_q, &inp_k, &inp_v, &inp_dy, w));
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(|a, b| a.total_cmp(b));
            res.push((ms[1], last.unwrap()));
        }
        ANCHOR_BWD_BATCH.store(false, Ordering::Relaxed);
        let (tb, gb) = &res[0];
        let (ts, gs) = &res[1];
        let rel = |a: &[f32], c: &[f32]| {
            let sc = c.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30);
            a.iter().zip(c).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / sc
        };
        let worst = [rel(&gb.dq, &gs.dq), rel(&gb.dk, &gs.dk), rel(&gb.dv, &gs.dv), rel(&gb.dsink_k, &gs.dsink_k), rel(&gb.dsink_v, &gs.dsink_v)]
            .into_iter()
            .fold(0.0f32, f32::max);
        eprintln!("anchor bwd B8/T512 W={w} S={s}: batched {tb:.1} ms vs per-sequence {ts:.1} ms (probe = fwd+bwd+readback); max rel Δ grads {worst:.2e}");
        assert!(worst <= 1e-6, "batched vs per-sequence gradients differ: {worst:e}");
    }
}

/// Row 0 of the attention core must not depend on how many rows share the
/// forward.  Production shape of the S4 bounded arm (8/2 GQA heads of 128,
/// T=1024, window 128, 4 sinks): the model-level witness showed rows 0-1
/// shifting between B≤3 and B=4 (M=4096) on Vulkan, in the anchor layer only.
/// `CMF_ANCHOR_ROWS_TEST=b1,b2,t,w` overrides the shapes.
#[test]
fn vulkan_anchor_core_row0_independent_of_batch() {
    let Some(_) = ctx() else { panic!("no Vulkan adapter") };
    let spec: Vec<usize> = std::env::var("CMF_ANCHOR_ROWS_TEST")
        .ok()
        .map(|x| x.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![3, 4, 1024, 128]);
    let (b1, b2, t, w) = (spec[0], spec[1], spec[2], spec[3]);
    let mut cfg = EmbryoCfg::tiny();
    cfg.hidden = 384;
    cfg.layers = 2;
    cfg.anchor_every = 2;
    cfg.heads = 8;
    cfg.nphase = 32;
    cfg.dv = 128;
    let heads: Vec<usize> = std::env::var("CMF_ANCHOR_ROWS_HEADS")
        .ok()
        .map(|x| x.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 2, 128]);
    cfg.anchor_q_heads = heads[0];
    cfg.anchor_kv_heads = heads[1];
    cfg.anchor_hd = heads[2];
    cfg.experts = 4;
    cfg.inter = 768;
    cfg.head_clusters = 64;
    cfg.seq = t;
    cfg.conv_k = 4;
    cfg.anchor_window = w;
    cfg.anchor_sink = if w > 0 { 4 } else { 0 };
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 7);
    let l = first_anchor(&cfg);
    let (qh, kvh, hd, h) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd, cfg.hidden);
    let bmax = b1.max(b2);
    let q = lcg_vec(11, bmax * t * qh * hd);
    let k = lcg_vec(12, bmax * t * kvh * hd);
    let v = lcg_vec(13, bmax * t * kvh * hd);
    let dy = lcg_vec(14, bmax * t * h);
    // stage dumps first (scores, softmax, P·V of sequence 0), then the output
    for stage in [1usize, 2, 3] {
        unsafe { std::env::set_var("CMF_ANCHOR_STAGE", stage.to_string()) };
        let mut dumps: Vec<Vec<f32>> = Vec::new();
        for b in [b1, b2] {
            let m = b * t;
            let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("vulkan model");
            let got = gpu.anchor_core_probe(
                l,
                &q[..m * qh * hd],
                &k[..m * kvh * hd],
                &v[..m * kvh * hd],
                &dy[..m * h],
                w,
            );
            dumps.push(if stage == 3 { got.o0 } else { got.p0 });
        }
        let scale = dumps[0].iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-30);
        let mut worst = 0.0f32;
        let mut first = None;
        let mut n = 0usize;
        for (i, (a, b)) in dumps[0].iter().zip(&dumps[1]).enumerate() {
            let d = (a - b).abs();
            if d > 0.0 {
                n += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
            worst = worst.max(d);
        }
        eprintln!(
            "  stage {stage} ({}): B={b1} vs B={b2} max rel {:.2e}, {n} elements differ, first at {first:?}",
            ["", "scores", "softmax", "P·V"][stage],
            worst / scale
        );
    }
    unsafe { std::env::remove_var("CMF_ANCHOR_STAGE") };
    let mut ys: Vec<Vec<f32>> = Vec::new();
    for b in [b1, b2] {
        let m = b * t;
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("vulkan model");
        let got = gpu.anchor_core_probe(
            l,
            &q[..m * qh * hd],
            &k[..m * kvh * hd],
            &v[..m * kvh * hd],
            &dy[..m * h],
            w,
        );
        ys.push(got.y[..t * h].to_vec());
    }
    let scale = ys[0].iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-30);
    let mut worst = 0.0f32;
    let mut first: Option<(usize, usize)> = None;
    for (i, (a, b)) in ys[0].iter().zip(&ys[1]).enumerate() {
        let d = (a - b).abs();
        if d > 0.0 && first.is_none() {
            first = Some((i / h, i % h));
        }
        worst = worst.max(d);
    }
    let mut bins = vec![0.0f32; t / 64];
    let mut npos = 0usize;
    for pos in 0..t {
        let d = (0..h)
            .map(|j| (ys[0][pos * h + j] - ys[1][pos * h + j]).abs())
            .fold(0.0f32, f32::max);
        if d > 0.0 {
            npos += 1;
        }
        bins[pos / 64] = bins[pos / 64].max(d / scale);
    }
    eprintln!(
        "anchor core y row 0, B={b1} vs B={b2}, T={t} W={w} heads {heads:?}: max|Δ|/max|y| = {:.2e}, first differing (pos, dim) = {first:?}, {npos}/{t} positions differ; per-64 max rel: {}",
        worst / scale,
        bins.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" ")
    );
    assert!(worst / scale <= 1e-6, "row 0 of the anchor core depends on the batch: {:e}", worst / scale);
}
