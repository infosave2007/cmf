//! Bounded anchor `swa_sink_v1` in the trainer (plan S2, contract
//! docs/EMBRYO_BOUNDED_ANCHOR.md):
//!  - the GPU attention core (band + trained NoPE sinks, absolute rope on
//!    q/k, `[T, SINK_PAD + T]` score rows) against an f64 reference of the
//!    served operator, forward and every gradient;
//!  - `anchor_window = 0` bit-identical to the pre-S2 trainer (golden bits
//!    captured on this M4 before the change) and to the "window ≥ T, no
//!    sink" branch;
//!  - whole-graph f32 finite differences on bounded genomes;
//!  - export round-trip (BoundedAttention + anchor_core + sink tensors) and
//!    a byte-identical legacy export;
//!  - continuing a legacy checkpoint under the mask (name-based arena
//!    extension, old parameters/moments bit-identical);
//!  - the stochastic window schedule and the step-time cost.
#![cfg(target_os = "macos")]

use cortiq_embryo::metal::{Cmd, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, gauss_vec, init_params};
use cortiq_embryo::ops::{
    BoundedAnchorDims, SOFTMAX_FIXTURE, bounded_anchor_ref, lcg_vec, softmax_reference, softmax_samples,
    softmax_scores, softmax_upstream,
};
use cortiq_embryo::train::{Checkpoint, append_anchor_sinks_checkpoint, load_checkpoint, save_checkpoint};

fn toks(seed: u64, n: usize, vocab: usize) -> Vec<u32> {
    lcg_vec(seed, n)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * vocab as f32) as u32 % vocab as u32)
        .collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

/// max|got − want| / max|want|
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w).abs())
        .fold(0.0, f64::max)
        / scale
}

fn tok_json(vocab: usize) -> String {
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    cortiq_embryo::tokenizer::count_words("hello world hello embryo", &re, &mut counts);
    cortiq_embryo::tokenizer::train(&counts, vocab, false).to_hf_json()
}

/// Which anchor layer of `cfg` to probe (the first).
fn first_anchor(cfg: &EmbryoCfg) -> usize {
    (0..cfg.layers).find(|&l| cfg.is_anchor(l)).expect("an anchor layer")
}

/// Random operator inputs of O(1) magnitude (sinks/Wo overwritten in the
/// arena so their terms are not negligible next to the window).
struct ProbeInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dy: Vec<f32>,
    sink_k: Vec<f32>,
    sink_v: Vec<f32>,
    wo: Vec<f32>,
}

fn probe_inputs(cfg: &EmbryoCfg, b: usize, t: usize, seed: u64) -> ProbeInputs {
    let (qh, kvh, hd, h) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd, cfg.hidden);
    let m = b * t;
    let ns = kvh * cfg.anchor_sink * hd;
    ProbeInputs {
        q: lcg_vec(seed + 1, m * qh * hd),
        k: lcg_vec(seed + 2, m * kvh * hd),
        v: lcg_vec(seed + 3, m * kvh * hd),
        dy: lcg_vec(seed + 4, m * h),
        sink_k: lcg_vec(seed + 5, ns),
        sink_v: lcg_vec(seed + 6, ns),
        wo: lcg_vec(seed + 7, h * qh * hd).iter().map(|x| x * 0.125).collect(),
    }
}

/// Run the GPU probe on `cfg` (window `w`) and compare with the reference.
/// Returns the worst relative error over every output.
fn probe_vs_reference(cfg: &EmbryoCfg, b: usize, t: usize, w: usize, seed: u64, tol: f64) -> f64 {
    let lay = Layout::new(cfg);
    let mut p0 = init_params(cfg, &lay, 7);
    let l = first_anchor(cfg);
    let inp = probe_inputs(cfg, b, t, seed);
    let (wo_off, sk_off, sv_off) = match &lay.layers[l] {
        cortiq_embryo::model::LayerOffs::Anchor {
            wo, sink_k, sink_v, ..
        } => (*wo, *sink_k, *sink_v),
        _ => unreachable!(),
    };
    p0[wo_off..wo_off + inp.wo.len()].copy_from_slice(&inp.wo);
    if cfg.anchor_sink > 0 {
        p0[sk_off..sk_off + inp.sink_k.len()].copy_from_slice(&inp.sink_k);
        p0[sv_off..sv_off + inp.sink_v.len()].copy_from_slice(&inp.sink_v);
    }
    let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    let got = gpu.anchor_core_probe(l, &inp.q, &inp.k, &inp.v, &inp.dy, w);
    let d = BoundedAnchorDims {
        b,
        t,
        qh: cfg.anchor_q_heads,
        kvh: cfg.anchor_kv_heads,
        hd: cfg.anchor_hd,
        h: cfg.hidden,
        s: cfg.anchor_sink,
        w,
        base: cfg.rope_base as f64,
    };
    let want = bounded_anchor_ref(
        &d,
        &to64(&inp.q),
        &to64(&inp.k),
        &to64(&inp.v),
        &to64(&inp.sink_k),
        &to64(&inp.sink_v),
        &to64(&inp.wo),
        Some(&to64(&inp.dy)),
    );
    let mut worst = 0.0f64;
    let mut check = |name: &str, g: &[f32], r: &[f64]| {
        let e = rel_err(g, r);
        let scale = r.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        eprintln!("  S={} W={w}: {name:<8} max|Δ|/max|ref| = {e:.2e} (max|ref| {scale:.3e})", cfg.anchor_sink);
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

/// Gate (contract §2/§4): forward ≤ 1e-5 rel vs f64, every gradient (dq,
/// dk, dv, dsink_k, dsink_v, dWo) ≤ 1e-5 rel on `tiny()` with S = 2, W = 3.
/// T = 64 instead of the contract's 8: the GEMM tile contract needs T % 64
/// == 0 (`EmbryoGpu::new` asserts it); a 3-wide band over 64 rows masks
/// far more than a band over 8 rows would, so the check is stricter, not
/// weaker. S = 2 also exercises the pad columns 2..64 of the sink tile.
#[test]
fn bounded_core_matches_f64_reference() {
    let Some(_) = ctx() else { return };
    let (b, t) = (2usize, 64usize);
    let mut worst = 0.0f64;
    for (s, w) in [(2usize, 3usize), (4, 16), (0, 8), (4, 64), (4, 1)] {
        let mut cfg = EmbryoCfg::tiny();
        cfg.anchor_window = w;
        cfg.anchor_sink = s;
        eprintln!("=== bounded core S={s} W={w} ===");
        worst = worst.max(probe_vs_reference(&cfg, b, t, w, 500 + s as u64 * 10 + w as u64, 1e-5));
    }
    // legacy full causal (window 0, no sinks) through the same probe
    let cfg = EmbryoCfg::tiny();
    eprintln!("=== legacy core (W=0) ===");
    worst = worst.max(probe_vs_reference(&cfg, b, t, 0, 600, 1e-5));
    eprintln!("worst rel err over all cases: {worst:.2e}");
}

/// The f64 reference's analytic backward against central differences of
/// its own forward (so the oracle above is itself checked), on a handful
/// of coordinates of every input.
#[test]
fn f64_reference_backward_matches_finite_differences() {
    let d = BoundedAnchorDims {
        b: 1,
        t: 8,
        qh: 2,
        kvh: 1,
        hd: 8,
        h: 6,
        s: 2,
        w: 3,
        base: 10000.0,
    };
    let (qd, kd, m) = (d.qh * d.hd, d.kvh * d.hd, d.b * d.t);
    let q = to64(&lcg_vec(1, m * qd));
    let k = to64(&lcg_vec(2, m * kd));
    let v = to64(&lcg_vec(3, m * kd));
    let sk = to64(&lcg_vec(4, d.kvh * d.s * d.hd));
    let sv = to64(&lcg_vec(5, d.kvh * d.s * d.hd));
    let wo = to64(&lcg_vec(6, d.h * qd));
    let dy = to64(&lcg_vec(7, m * d.h));
    let loss = |q: &[f64], k: &[f64], v: &[f64], sk: &[f64], sv: &[f64], wo: &[f64]| -> f64 {
        let o = bounded_anchor_ref(&d, q, k, v, sk, sv, wo, None);
        o.y.iter().zip(&dy).map(|(a, b)| a * b).sum()
    };
    let an = bounded_anchor_ref(&d, &q, &k, &v, &sk, &sv, &wo, Some(&dy));
    let eps = 1e-6;
    let mut worst = 0.0f64;
    let mut check = |name: &str, x: &[f64], g: &[f64], f: &dyn Fn(&[f64]) -> f64| {
        for i in (0..x.len()).step_by((x.len() / 12).max(1)) {
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
            let err = (fd - g[i]).abs() / (1.0 + fd.abs());
            worst = worst.max(err);
            assert!(err < 1e-7, "{name}[{i}]: fd {fd} vs analytic {}", g[i]);
        }
    };
    check("dq", &q, &an.dq, &|x| loss(x, &k, &v, &sk, &sv, &wo));
    check("dk", &k, &an.dk, &|x| loss(&q, x, &v, &sk, &sv, &wo));
    check("dv", &v, &an.dv, &|x| loss(&q, &k, x, &sk, &sv, &wo));
    check("dsink_k", &sk, &an.dsink_k, &|x| loss(&q, &k, &v, x, &sv, &wo));
    check("dsink_v", &sv, &an.dsink_v, &|x| loss(&q, &k, &v, &sk, x, &wo));
    check("dwo", &wo, &an.dwo, &|x| loss(&q, &k, &v, &sk, &sv, x));
    eprintln!("f64 reference: worst FD rel err {worst:.2e}");
}

// ---------------------------------------------------------------------
// Regression: the legacy anchor is untouched
// ---------------------------------------------------------------------

/// Loss / gradient-norm bits of the pre-S2 trainer (commit f31fee53 tree,
/// Apple M4, `cargo test` dev profile) on three fixed batches: eval
/// forward and `train_step(lr = 0)`. Both `tiny()` (one anchor) and the
/// all-anchor `anchor_every = 1` twin.
const GOLDEN: [[(u32, u32); 3]; 2] = [
    [
        (0x4104a97c, 0x4024303f),
        (0x410549dc, 0x402aab75),
        (0x4105053b, 0x402e7bf8),
    ],
    [
        (0x4104e408, 0x3fde4326),
        (0x41055f75, 0x3fd4ee0b),
        (0x4105cace, 0x3fd69eb0),
    ],
];

#[test]
fn legacy_anchor_is_bit_identical_to_pre_s2_goldens() {
    let Some(_) = ctx() else { return };
    for variant in 0..2 {
        let mut cfg = EmbryoCfg::tiny();
        if variant == 1 {
            cfg.anchor_every = 1;
        }
        assert_eq!(cfg.anchor_window, 0);
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 7);
        let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
        gpu.desc_updates.set(false);
        for k in 0..3u64 {
            let tokens = toks(1000 + k, m, cfg.vocab);
            let targets = toks(2000 + k, m, cfg.vocab);
            let e = gpu.eval_loss(&tokens, &targets);
            let (l, g, _) = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
            let (want_l, want_g) = GOLDEN[variant][k as usize];
            eprintln!(
                "variant {variant} batch {k}: loss {l:.7} (0x{:08x}, golden 0x{want_l:08x}) |g| {g:.7} (0x{:08x}, golden 0x{want_g:08x})",
                l.to_bits(),
                g.to_bits()
            );
            assert_eq!(e.to_bits(), want_l, "eval loss bits drifted from master");
            assert_eq!(l.to_bits(), want_l, "train loss bits drifted from master");
            assert_eq!(g.to_bits(), want_g, "grad norm bits drifted from master");
        }
    }
}

/// `window ≥ T` without sinks is mathematically the legacy operator; the
/// kernels must agree bit-for-bit (loss, gradient norm, every gradient).
#[test]
fn full_window_without_sinks_matches_legacy_bits() {
    let Some(_) = ctx() else { return };
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
        assert_eq!(Layout::new(&cfg).total, lay.total, "no sinks: same arena");
        let mut rhs = EmbryoGpu::new(cfg, b, t, &p0).expect("bounded");
        rhs.desc_updates.set(false);
        let (lr, gr, _) = rhs.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
        eprintln!("W={w}: loss 0x{:08x} vs legacy 0x{:08x}", lr.to_bits(), ll.to_bits());
        assert_eq!(lr.to_bits(), ll.to_bits(), "W={w} loss bits");
        assert_eq!(gr.to_bits(), gl.to_bits(), "W={w} grad norm bits");
        // Every tensor bit-identical except `embed`, whose tied-head scatter
        // uses float atomics on Metal (run-to-run 1-ulp noise, independent
        // of the anchor: two identical legacy models show it too).
        let grr = rhs.grads_host();
        for (name, off, n) in &lay.names {
            let (a, c) = (&grr[*off..*off + n], &gll[*off..*off + n]);
            if name == "embed" {
                let mx = a.iter().zip(c).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                assert!(mx <= 1e-7, "W={w} embed grad max|Δ| {mx:e}");
            } else {
                assert_eq!(a, c, "W={w} {name} gradients");
            }
        }
    }
}

/// Legacy cfg serialization gained no keys: an old checkpoint header and
/// the export provenance stay byte-identical, and old headers load.
#[test]
fn sink_plus_window_ceiling_matches_the_format() {
    assert_eq!(
        cortiq_embryo::model::MAX_SINK_PLUS_WINDOW,
        cortiq_core::types::AnchorCoreConfig::MAX_SINK_PLUS_WINDOW
    );
}

#[test]
fn legacy_cfg_serializes_without_anchor_keys_and_loads() {
    let js = serde_json::to_string(&EmbryoCfg::tiny()).unwrap();
    assert!(!js.contains("anchor_window"), "{js}");
    assert!(!js.contains("anchor_sink"), "{js}");
    assert!(!js.contains("anchor_train_windows"), "{js}");
    assert!(!js.contains("anchor_layers"), "{js}");
    let back: EmbryoCfg = serde_json::from_str(&js).unwrap();
    assert_eq!(back.anchor_window, 0);
    assert_eq!(back.anchor_sink, 0);
    assert!(back.anchor_train_windows.is_empty());
    assert!(back.anchor_layers.is_none());
    let mut bounded = EmbryoCfg::tiny();
    bounded.anchor_window = 16;
    bounded.anchor_sink = 4;
    bounded.anchor_train_windows = vec![8, 16];
    bounded.anchor_layers = Some(vec![1]);
    let js = serde_json::to_string(&bounded).unwrap();
    let back: EmbryoCfg = serde_json::from_str(&js).unwrap();
    assert_eq!(back.anchor_window, 16);
    assert_eq!(back.anchor_sink, 4);
    assert_eq!(back.anchor_train_windows, vec![8, 16]);
    assert_eq!(back.anchor_layers, Some(vec![1]));
}

// ---------------------------------------------------------------------
// Whole-graph f32 finite differences on bounded genomes (repo precedent:
// model_gradcheck.rs, 3e-2 on err/|g| with the f32 forward as the oracle)
// ---------------------------------------------------------------------

#[test]
fn bounded_genome_whole_graph_finite_differences() {
    let Some(c) = ctx() else { return };
    for variant in 0..4 {
        let mut cfg = EmbryoCfg::tiny();
        match variant {
            0 => {
                cfg.anchor_window = 16;
                cfg.anchor_sink = 4;
            }
            1 => {
                cfg.anchor_window = 3;
                cfg.anchor_sink = 2;
            }
            2 => {
                cfg.anchor_window = 8;
                cfg.anchor_sink = 0;
            }
            _ => {
                // explicit schedule: both tiny layers are bounded anchors
                cfg.anchor_layers = Some(vec![0, 1]);
                cfg.anchor_window = 16;
                cfg.anchor_sink = 4;
                cfg.anchor_train_windows = vec![4, 16];
            }
        }
        eprintln!("=== variant {variant}: W={} S={} layers={:?} ===", cfg.anchor_window, cfg.anchor_sink, cfg.anchor_layers);
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let mut p0 = init_params(&cfg, &lay, 7);
        // sinks at O(1) so their gradient paths carry signal
        for (name, off, n) in &lay.names {
            if name.contains("attn.sink_") {
                let r = lcg_vec(fnv1a(name.as_bytes()), *n);
                p0[*off..*off + n].copy_from_slice(&r);
            }
        }
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
        gpu.desc_updates.set(false);
        // the analytic pass and the FD forwards must share one window
        gpu.anchor_fixed_window.set(true);
        let tokens = toks(11, m, cfg.vocab);
        let targets = toks(12, m, cfg.vocab);
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
            std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
        }
        gpu.prepare_head(&targets);
        let cmd = Cmd::new(c);
        gpu.encode_fwd_bwd(&cmd);
        cmd.commit();
        assert_eq!(gpu.anchor_window_t.get(), cfg.anchor_window);
        let g = gpu.grads_host();
        gpu.route_frozen.set(true);
        let l0 = gpu.eval_loss(&tokens, &targets);
        eprintln!("loss {l0:.5}");
        let mut worst_all = 0.0f64;
        let mut seed = 100u64;
        for (name, off, n) in &lay.names {
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
                let denom = gnorm.max(5e-4);
                let rel = (fd - analytic).abs() / denom;
                worst_all = worst_all.max(rel);
                if name.contains("attn.") {
                    eprintln!(
                        "{name:<24} n={n:<6} |g|={gnorm:.3e} dir={} fd={fd:+.4e} an={analytic:+.4e} err/|g|={rel:.2e}",
                        if dir == 0 { "grad" } else { "rand" }
                    );
                }
                assert!(rel < 3e-2, "{name} dir {dir}: fd {fd} vs analytic {analytic} (|g| {gnorm})");
            }
        }
        gpu.set_params(&p0);
        eprintln!("variant {variant}: worst err/|g| = {worst_all:.2e}");
    }
}

// ---------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------

/// Bytes of the legacy tiny export (seed 5, step 0, the byte-level test
/// tokenizer): f32 / f16 storage profiles. Re-pinned 24.09.2026 for format
/// v2 only because the provenance no longer hardcodes `"genome":
/// "embryo-0"` (an export without `--genome-id` names no genome); lengths,
/// directory and every tensor byte are unchanged (the pre-v2 values were
/// 0xae73390d50313db7 / 0x315d37bdc6631092 with that key present).
const LEGACY_EXPORT_FNV: [(usize, u64); 2] = [
    (1_599_043, 0xa39fbbbbeef46ace),
    (865_859, 0x03f589854d3e3e70),
];

#[test]
fn export_round_trip_and_legacy_bytes() {
    let dir = std::env::temp_dir().join(format!("embryo_bounded_export_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vocab = EmbryoCfg::tiny().vocab;
    let tj = tok_json(vocab);
    // legacy: byte-identical to the pre-S2 exporter
    let cfg = EmbryoCfg::tiny();
    let lay = Layout::new(&cfg);
    let ck = Checkpoint {
        cfg: cfg.clone(),
        step: 0,
        params: init_params(&cfg, &lay, 5),
        m: None,
        v: None,
        extras: Vec::new(),
    };
    for (i, dtype) in [cortiq_core::TensorDtype::F32, cortiq_core::TensorDtype::F16]
        .into_iter()
        .enumerate()
    {
        let path = dir.join(format!("legacy-{i}.cmf"));
        cortiq_embryo::export::export_with_dtype(&ck, tj.as_bytes(), &path, dtype).expect("export");
        let bytes = std::fs::read(&path).unwrap();
        let (want_len, want_fnv) = LEGACY_EXPORT_FNV[i];
        eprintln!("legacy {dtype:?}: {} bytes fnv 0x{:016x} (golden {want_len} 0x{want_fnv:016x})", bytes.len(), fnv1a(&bytes));
        assert_eq!(bytes.len(), want_len, "legacy export size changed");
        assert_eq!(fnv1a(&bytes), want_fnv, "legacy export bytes changed");
        let model = cortiq_core::format::CmfModel::open(&path).expect("open legacy");
        assert!(model.header.arch.anchor_core.is_none());
        assert!(model.header.arch.layer_types.iter().all(|t| !matches!(t, cortiq_core::LayerType::BoundedAttention)));
        assert_eq!(model.required_features & cortiq_core::format::features::BOUNDED_STATE, 0);
        assert!(model.tensor("model.layers.1.self_attn.sink_k.weight").is_none());
    }
    // bounded: BoundedAttention + anchor_core + f32 sink tensors in both profiles
    let mut cfg = EmbryoCfg::tiny();
    cfg.anchor_window = 16;
    cfg.anchor_sink = 4;
    cfg.anchor_train_windows = vec![8, 16];
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 5);
    let ck = Checkpoint {
        cfg: cfg.clone(),
        step: 3,
        params: params.clone(),
        m: None,
        v: None,
        extras: Vec::new(),
    };
    let (kvh, hd) = (cfg.anchor_kv_heads, cfg.anchor_hd);
    let (sk_off, sv_off) = match &lay.layers[1] {
        cortiq_embryo::model::LayerOffs::Anchor { sink_k, sink_v, .. } => (*sink_k, *sink_v),
        _ => unreachable!(),
    };
    for dtype in [cortiq_core::TensorDtype::F32, cortiq_core::TensorDtype::F16] {
        let path = dir.join(format!("bounded-{dtype:?}.cmf"));
        cortiq_embryo::export::export_with_dtype(&ck, tj.as_bytes(), &path, dtype).expect("export bounded");
        let model = cortiq_core::format::CmfModel::open(&path).expect("open bounded");
        let arch = &model.header.arch;
        assert_eq!(arch.layer_types.len(), 2);
        assert!(matches!(arch.layer_types[0], cortiq_core::LayerType::LinearAttention));
        assert!(matches!(arch.layer_types[1], cortiq_core::LayerType::BoundedAttention));
        let ac = arch.anchor_core.as_ref().expect("anchor_core record");
        assert_eq!(ac.kind, "swa_sink_v1");
        assert_eq!(ac.window, 16);
        assert_eq!(ac.sink, 4);
        assert_eq!(ac.rope, "relative_in_window");
        assert_eq!(ac.sink_scores, "nope");
        assert_eq!(ac.train_windows, vec![8, 16]);
        assert!(ac.far.is_none());
        assert_ne!(model.required_features & cortiq_core::format::features::BOUNDED_STATE, 0);
        for (name, off) in [("sink_k", sk_off), ("sink_v", sv_off)] {
            let full = format!("model.layers.1.self_attn.{name}.weight");
            let e = model.tensor(&full).unwrap_or_else(|| panic!("{full} missing"));
            assert_eq!(e.shape, vec![kvh, 4, hd]);
            assert_eq!(e.dtype, cortiq_core::TensorDtype::F32, "{full} must stay f32 in {dtype:?}");
            let bytes = model.tensor_bytes(&full).unwrap();
            let got: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            assert_eq!(got, params[off..off + kvh * 4 * hd].to_vec(), "{full} payload");
        }
        // the q/k/v/o projections follow the profile as before
        let q = model.tensor("model.layers.1.self_attn.q_proj.weight").unwrap();
        assert_eq!(q.dtype, dtype);
        eprintln!("bounded {dtype:?}: {} tensors, anchor_core {:?}", model.tensors.len(), ac);
    }
    // refusals: sink + window > 160, sinks without a window, windows outside 1..=W
    for (w, s, tw) in [(157usize, 4usize, vec![]), (0, 4, vec![]), (16, 4, vec![32]), (16, 4, vec![0])] {
        let mut bad = EmbryoCfg::tiny();
        bad.anchor_window = w;
        bad.anchor_sink = s;
        bad.anchor_train_windows = tw.clone();
        // the checkpoint's params are irrelevant: validation runs first
        let ck = Checkpoint {
            cfg: bad,
            step: 0,
            params: Vec::new(),
            m: None,
            v: None,
            extras: Vec::new(),
        };
        let path = dir.join(format!("bad-{w}-{s}-{}.cmf", tw.len()));
        let err = match cortiq_embryo::export::export(&ck, tj.as_bytes(), &path) {
            Ok(()) => panic!("export must refuse W={w} S={s} tw={tw:?}"),
            Err(e) => e,
        };
        eprintln!("refused W={w} S={s} tw={tw:?}: {err}");
        assert!(!path.exists(), "a refused export must not leave a file");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------
// Resume-extend (S4 probe): continue a legacy checkpoint under the mask
// ---------------------------------------------------------------------

#[test]
fn resume_extends_legacy_checkpoint_with_sinks() {
    let Some(_) = ctx() else { return };
    let cfg = EmbryoCfg::tiny();
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 9);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    gpu.desc_updates.set(false);
    for k in 0..2u64 {
        let (loss, _, _) = gpu.train_step(&toks(300 + k, m, cfg.vocab), &toks(400 + k, m, cfg.vocab), 1e-3, 0.1, 1.0);
        assert!(loss.is_finite());
    }
    let legacy = Checkpoint {
        cfg: cfg.clone(),
        step: gpu.step,
        params: gpu.params_host(),
        m: Some(gpu.m.to_vec()),
        v: Some(gpu.v.to_vec()),
        extras: Vec::new(),
    };
    assert_eq!(legacy.step, 2);
    let dir = std::env::temp_dir().join(format!("embryo_bounded_resume_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ck_path = dir.join("legacy.ckpt");
    save_checkpoint(
        &ck_path,
        &legacy.cfg,
        legacy.step,
        &legacy.params,
        legacy.m.as_deref(),
        legacy.v.as_deref(),
        &[],
    )
    .unwrap();
    let loaded = load_checkpoint(&ck_path).unwrap();
    assert_eq!(loaded.cfg.anchor_window, 0);
    assert_eq!(loaded.params, legacy.params);
    // what `birth --resume legacy.ckpt --anchor-window 16 --anchor-sink 4 --anchor-train-windows 8,16` does
    let grown = append_anchor_sinks_checkpoint(&loaded, 16, 4, &[8, 16], 1).expect("append sinks");
    assert_eq!(grown.step, 2);
    assert_eq!(grown.cfg.anchor_window, 16);
    assert_eq!(grown.cfg.anchor_sink, 4);
    assert_eq!(grown.cfg.anchor_train_windows, vec![8, 16]);
    let glay = Layout::new(&grown.cfg);
    assert_eq!(grown.params.len(), glay.total);
    let added = 2 * cfg.anchor_kv_heads * 4 * cfg.anchor_hd; // one anchor layer in tiny
    assert_eq!(glay.total, lay.total + added);
    let old_map: std::collections::HashMap<&str, (usize, usize)> =
        lay.names.iter().map(|(n, o, l)| (n.as_str(), (*o, *l))).collect();
    let (gm, gv) = (grown.m.as_ref().unwrap(), grown.v.as_ref().unwrap());
    let (lm, lv) = (legacy.m.as_ref().unwrap(), legacy.v.as_ref().unwrap());
    let mut legacy_seen = 0usize;
    for (name, off, n) in &glay.names {
        if let Some((oo, ol)) = old_map.get(name.as_str()) {
            assert_eq!(ol, n);
            assert_eq!(&grown.params[*off..*off + n], &legacy.params[*oo..*oo + n], "{name} params");
            assert_eq!(&gm[*off..*off + n], &lm[*oo..*oo + n], "{name} m");
            assert_eq!(&gv[*off..*off + n], &lv[*oo..*oo + n], "{name} v");
            legacy_seen += 1;
        } else {
            assert!(name.ends_with("attn.sink_k") || name.ends_with("attn.sink_v"), "unexpected new tensor {name}");
            assert!(gm[*off..*off + n].iter().all(|x| *x == 0.0), "{name} m must start at 0");
            assert!(gv[*off..*off + n].iter().all(|x| *x == 0.0), "{name} v must start at 0");
            let x = &grown.params[*off..*off + n];
            if name.ends_with("sink_v") {
                assert!(x.iter().all(|v| *v == 0.0), "sink_v init must be 0");
            } else {
                let std = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / *n as f64).sqrt();
                eprintln!("{name}: n={n} std {std:.4}");
                assert!((std - 0.02).abs() < 0.006, "sink_k std {std}");
            }
        }
    }
    assert_eq!(legacy_seen, lay.names.len(), "every legacy tensor survives by name");
    // the grown checkpoint round-trips through the file format and trains
    let g_path = dir.join("bounded.ckpt");
    save_checkpoint(&g_path, &grown.cfg, grown.step, &grown.params, grown.m.as_deref(), grown.v.as_deref(), &[]).unwrap();
    let back = load_checkpoint(&g_path).unwrap();
    assert_eq!(back.cfg.anchor_sink, 4);
    assert_eq!(back.params, grown.params);
    assert_eq!(back.m, grown.m);
    let mut gpu2 = EmbryoGpu::new(back.cfg.clone(), b, t, &back.params).expect("gpu bounded");
    gpu2.m.write_from(back.m.as_ref().unwrap());
    gpu2.v.write_from(back.v.as_ref().unwrap());
    gpu2.step = back.step;
    gpu2.desc_updates.set(false);
    let (loss, gn, _) = gpu2.train_step(&toks(302, m, cfg.vocab), &toks(402, m, cfg.vocab), 1e-3, 0.1, 1.0);
    eprintln!("bounded continuation step 3: loss {loss:.4} |g| {gn:.3} window {}", gpu2.anchor_window_t.get());
    assert!(loss.is_finite() && gn.is_finite());
    assert!([8usize, 16].contains(&gpu2.anchor_window_t.get()));
    // a checkpoint with sinks cannot change its sink count on resume
    let err = match append_anchor_sinks_checkpoint(&back, 16, 2, &[], 1) {
        Ok(_) => panic!("changing the sink count on resume must be refused"),
        Err(e) => e,
    };
    eprintln!("refused: {err}");
    // but may keep it (a pure window change)
    let same = append_anchor_sinks_checkpoint(&back, 32, 4, &[16, 32], 1).expect("window change");
    assert_eq!(same.params, back.params);
    assert_eq!(same.cfg.anchor_window, 32);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------
// Stochastic window schedule
// ---------------------------------------------------------------------

#[test]
fn stochastic_window_is_deterministic_and_served_on_eval() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.anchor_window = 16;
    cfg.anchor_sink = 4;
    cfg.anchor_train_windows = vec![4, 8, 16];
    let draws: Vec<usize> = (0..64).map(|s| cfg.anchor_window_at_step(s)).collect();
    assert!(draws.iter().all(|w| cfg.anchor_train_windows.contains(w)));
    for w in &cfg.anchor_train_windows {
        assert!(draws.contains(w), "window {w} never drawn in 64 steps: {draws:?}");
    }
    let again: Vec<usize> = (0..64).map(|s| cfg.anchor_window_at_step(s)).collect();
    assert_eq!(draws, again);
    eprintln!("first 16 draws: {:?}", &draws[..16]);
    let fixed = EmbryoCfg { anchor_train_windows: vec![], ..cfg.clone() };
    assert!((0..8).all(|s| fixed.anchor_window_at_step(s) == 16));
    let Some(_) = ctx() else { return };
    let (b, t) = (1usize, 64usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 3);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    gpu.desc_updates.set(false);
    let tokens = toks(1, m, cfg.vocab);
    let targets = toks(2, m, cfg.vocab);
    for step in 0..6u64 {
        let _ = gpu.train_step(&tokens, &targets, 1e-4, 0.0, 1.0);
        assert_eq!(gpu.anchor_window_t.get(), cfg.anchor_window_at_step(step), "step {step}");
    }
    assert_eq!(gpu.anchor_window_for(false), 16, "eval serves the fixed window");
    gpu.anchor_fixed_window.set(true);
    let _ = gpu.train_step(&tokens, &targets, 1e-4, 0.0, 1.0);
    assert_eq!(gpu.anchor_window_t.get(), 16, "forced served window");
    // eval with the served window is a different function from a narrower one
    let mut narrow = cfg.clone();
    narrow.anchor_window = 4;
    narrow.anchor_train_windows = vec![];
    let p_now = gpu.params_host();
    let wide = EmbryoGpu::new(EmbryoCfg { anchor_train_windows: vec![], ..cfg.clone() }, b, t, &p_now).unwrap();
    let nar = EmbryoGpu::new(narrow, b, t, &p_now).unwrap();
    let (lw, ln) = (wide.eval_loss(&tokens, &targets), nar.eval_loss(&tokens, &targets));
    eprintln!("eval loss W=16 {lw:.5} vs W=4 {ln:.5}");
    assert_ne!(lw.to_bits(), ln.to_bits());
}

// ---------------------------------------------------------------------
// Step time
// ---------------------------------------------------------------------

/// tiny B2/T64 always (legacy vs bounded in one process: both fit).
/// The Embryo-0 B8/T1024 measurement runs ONE configuration per process
/// (`EMBRYO_BENCH_FULL=W,S`, e.g. `0,0` / `128,4` / `128,0`): two ~17 GiB
/// arenas allocated back to back in one process put the second one under
/// memory pressure (measured 1348 → 2629 ms for identical work), so the
/// A/B is taken across processes from the printed medians and the
/// anchor-layer profile (`profile_step`), never from one process.
#[test]
fn step_time_legacy_vs_bounded() {
    let Some(_) = ctx() else { return };
    let run = |cfg: EmbryoCfg, b: usize, t: usize, warmup: usize, samples: usize, profile: bool| {
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 1);
        let m = b * t;
        let tokens = toks(7001, m, cfg.vocab);
        let targets = toks(7002, m, cfg.vocab);
        let tag = format!("W={} S={}", cfg.anchor_window, cfg.anchor_sink);
        let mut gpu = EmbryoGpu::new(cfg, b, t, &p0).expect("gpu");
        let (mut wall, mut dev) = (Vec::new(), Vec::new());
        for i in 0..warmup + samples {
            let t0 = std::time::Instant::now();
            let (loss, _, gpu_ms) = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1.0);
            assert!(loss.is_finite());
            if i >= warmup {
                wall.push(t0.elapsed().as_secs_f64() * 1e3);
                dev.push(gpu_ms);
            }
        }
        wall.sort_by(|a, b| a.total_cmp(b));
        dev.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "{tag} B{b}/T{t}: median wall {:.1} ms, median gpu {:.1} ms (n={samples})",
            wall[wall.len() / 2],
            dev[dev.len() / 2]
        );
        if profile {
            unsafe {
                std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
                std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
            }
            gpu.prepare_head(&targets);
            for (name, ms) in gpu.profile_step() {
                if name.contains("anchor") {
                    eprintln!("{tag} profile {name:<28} {ms:>8.1} ms");
                }
            }
        }
        wall[wall.len() / 2]
    };
    let tiny = EmbryoCfg::tiny();
    let mut tiny_b = tiny.clone();
    tiny_b.anchor_window = 16;
    tiny_b.anchor_sink = 4;
    let c = run(tiny, 2, 64, 3, 9, false);
    let bnd = run(tiny_b, 2, 64, 3, 9, false);
    eprintln!("tiny B2/T64: legacy {c:.2} ms, bounded(W16,S4) {bnd:.2} ms, ratio {:.3}", bnd / c);
    if let Ok(spec) = std::env::var("EMBRYO_BENCH_FULL") {
        let mut it = spec.split(',').map(|x| x.trim().parse::<usize>().expect("EMBRYO_BENCH_FULL=W,S"));
        let (w, s) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
        let mut cfg = EmbryoCfg::embryo0();
        cfg.anchor_window = w;
        cfg.anchor_sink = s;
        if w > 0 {
            cfg.anchor_train_windows = vec![w / 2, w];
        }
        run(cfg, 8, 1024, 2, 5, true);
    }
}

// ---------------------------------------------------------------------
// Kernel-level: band+sink softmax fwd/bwd and the sink gradient fold vs
// an f64 CPU reference, plus a fingerprint of sampled outputs that the
// Vulkan twin (tests/vulkan_bounded_anchor.rs) compares against, so the
// Metal ↔ WGSL agreement is a measured number and not an inference.
// ---------------------------------------------------------------------

#[test]
fn softmax_kernels_match_f64_reference_and_print_fingerprint() {
    let Some(c) = ctx() else { return };
    let f = &SOFTMAX_FIXTURE;
    let scores = softmax_scores(f);
    let upstream = softmax_upstream(f);
    let (want_p, want_ds) = softmax_reference(f, &scores, &upstream);
    let p = cortiq_embryo::metal::GBuf::from_slice(c, &scores);
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
    eprintln!("band+sink softmax fwd: max|Δ| vs f64 = {e_p:.2e}");
    assert!(e_p < 1e-6);
    for block in 0..f.blocks {
        for row in 0..f.t {
            let base = f.off + block * f.t * f.ld + row * f.ld;
            let s: f32 = got_p[base..base + f.ld].iter().sum();
            assert!((s - 1.0).abs() < 2e-6, "row sum {s} at {block}:{row}");
        }
    }
    let d = cortiq_embryo::metal::GBuf::from_slice(c, &upstream);
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
    eprintln!("band+sink softmax bwd: max|Δ| vs f64 = {e_ds:.2e}");
    assert!(e_ds < 1e-6);
    let samples = softmax_samples(f);
    let mut line_p = String::new();
    let mut line_ds = String::new();
    for &(block, row, col) in &samples {
        let i = f.off + block * f.t * f.ld + row * f.ld + col;
        line_p.push_str(&format!("0x{:08x}, ", got_p[i].to_bits()));
        line_ds.push_str(&format!("0x{:08x}, ", got_ds[i].to_bits()));
    }
    eprintln!("METAL_P_BITS: [{line_p}]");
    eprintln!("METAL_DS_BITS: [{line_ds}]");
    // sink gradient fold: per-head tiles [qh][64][hd] → arena [kvh][S][hd]
    let (qh, kvh, sink, hd, dst_off) = (4usize, 2usize, 3usize, 32usize, 5usize);
    let group = qh / kvh;
    let src = lcg_vec(4444, qh * 64 * hd);
    let dst0 = lcg_vec(4545, dst_off + kvh * sink * hd);
    let sb = cortiq_embryo::metal::GBuf::from_slice(c, &src);
    let db = cortiq_embryo::metal::GBuf::from_slice(c, &dst0);
    let cmd = Cmd::new(c);
    cmd.sink_grad_accum(&sb, &db, dst_off, 1, kvh, group, sink, hd, 0.125);
    cmd.commit();
    let got = db.to_vec();
    assert_eq!(got[..dst_off], dst0[..dst_off]);
    let mut e = 0.0f64;
    let mut line = String::new();
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
                    line.push_str(&format!("0x{:08x}, ", got[i].to_bits()));
                }
            }
        }
    }
    eprintln!("sink_grad_accum: max|Δ| vs f64 = {e:.2e}");
    eprintln!("METAL_ACC_BITS: [{line}]");
    assert!(e < 1e-6);
}


/// S6c item 3a: the anchor backward batched over the sequences vs the
/// per-sequence dispatch shape, same process, Embryo-0 anchor geometry at
/// B8/T512 (legacy full causal and bounded W128/S4): gradients identical
/// to 1e-6 rel and both timings (median of 3 probes: fwd + bwd + readback).
#[test]
fn anchor_backward_batched_vs_per_sequence() {
    let Some(_) = ctx() else { return };
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
        let inp = probe_inputs(&cfg, b, t, 700);
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("embryo0");
        let mut res = Vec::new();
        for batched in [true, false] {
            ANCHOR_BWD_BATCH.store(batched, Ordering::Relaxed);
            let mut ms = Vec::new();
            let mut last = None;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                last = Some(gpu.anchor_core_probe(l, &inp.q, &inp.k, &inp.v, &inp.dy, w));
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
