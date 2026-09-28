//! State carry-over across windows (plan S6b, `--carry`): the hybrid_k
//! kernels from a non-zero initial state against the f64 oracle, two
//! carried windows against one pass (forward, per-position NLL), and the
//! whole-graph finite-difference check with a carried (non-zero) state.
//! Runs on Metal (macOS) and Vulkan (`--features vulkan`).
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::metal::{Cmd, GBuf, HkGrads, HkScratch, HkWork, ctx, hk_pow_table};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, gauss_vec, init_params};
use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd_s0, hk_ref_fwd_s0, lcg_vec};

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max)
        / scale
}

/// hybrid_k forward/backward from a carried S_0 ≠ 0 (checkpoint slot 0 +
/// `set_hk_carry`) vs the f64 oracle, both kernel paths (SIMT scan and the
/// GEMM form).
#[test]
fn hk_kernels_from_carried_state_match_f64() {
    let Some(c) = ctx() else { return };
    // the GEMM form needs 2·nph and dv to be tile multiples (production: 32 / 128)
    let d = HkDims {
        b: 2,
        t: 128,
        nh: 2,
        nph: 32,
        dv: 64,
    };
    let rows = d.b * d.t;
    let p2 = d.p2();
    let thq: Vec<f32> = lcg_vec(501, rows * d.nh * d.nph).iter().map(|x| x * 2.0).collect();
    let thk: Vec<f32> = lcg_vec(502, rows * d.nh * d.nph).iter().map(|x| x * 2.0).collect();
    let v = lcg_vec(503, rows * d.nh * d.dv);
    let kappa: Vec<f32> = lcg_vec(504, rows * d.nh).iter().map(|x| 0.25 + 0.5 * (x + 1.0) / 2.0).collect();
    let dout = lcg_vec(505, rows * d.nh * d.dv);
    let s0: Vec<f32> = lcg_vec(506, d.b * d.nh * p2 * d.dv).iter().map(|x| 0.5 * x).collect();
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let (want_o, want_s) = hk_ref_fwd_s0(&d, &to64(&thq), &to64(&thk), &to64(&v), &to64(&kappa), &to64(&decay), &to64(&s0));
    let (want_q, want_k, want_v, want_kap) = hk_ref_bwd_s0(
        &d, &to64(&thq), &to64(&thk), &to64(&v), &to64(&kappa), &to64(&decay), &to64(&s0), &to64(&dout),
    );
    let z = |n: usize| GBuf::zeros(c, n);
    let nch = d.t / 64;
    let nst = d.b * d.nh * (nch + 1) * p2 * d.dv;
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kappa),
    );
    let gpow = GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, d.nph));
    let (gphq, gphk, gkv, gout) = (z(rows * d.nh * p2), z(rows * d.nh * p2), z(rows * d.nh * d.dv), z(rows * d.nh * d.dv));
    let states = z(nst);
    let chunk = z(d.b * d.nh * d.t * (p2 + 1) * d.dv);
    let partial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + p2));
    let cl = HkScratch::chunk_len(&d);
    let sc: Vec<GBuf> = (0..8).map(|_| z(cl)).chain(std::iter::once(z(HkScratch::a_len(&d)))).collect();
    let (gdst, gdkv, gdphq, gdphk) = (z(nst), z(rows * d.nh * d.dv), z(rows * d.nh * p2), z(rows * d.nh * p2));
    let (gdthq, gdthk, gdv, gdkap) = (z(rows * d.nh * d.nph), z(rows * d.nh * d.nph), z(rows * d.nh * d.dv), z(rows * d.nh));
    let gdout = GBuf::from_slice(c, &dout);
    // S_0 into checkpoint slot 0 of every (b, h)
    let seed_slot0 = || {
        let mut st = vec![0.0f32; nst];
        for bh in 0..d.b * d.nh {
            st[bh * (nch + 1) * p2 * d.dv..bh * (nch + 1) * p2 * d.dv + p2 * d.dv]
                .copy_from_slice(&s0[bh * p2 * d.dv..(bh + 1) * p2 * d.dv]);
        }
        states.write_from(&st);
    };
    for gemm in [false, true] {
        seed_slot0();
        gout.fill(0.0);
        let w = HkWork {
            thq: &gthq,
            thk: &gthk,
            v: &gv,
            kappa: &gkap,
            pow: &gpow,
            pow_off: 0,
            phq: &gphq,
            phk: &gphk,
            kv: &gkv,
            states: &states,
            out: &gout,
            phase_chunk: Some(&chunk),
            phase_partial: Some(&partial),
        };
        let gr = HkGrads {
            dout: &gdout,
            dstates: &gdst,
            dkv: &gdkv,
            dphq: &gdphq,
            dphk: &gdphk,
            dthq: &gdthq,
            dthk: &gdthk,
            dv: &gdv,
            dkappa: &gdkap,
        };
        let hs = HkScratch {
            qt: &sc[0],
            kt: &sc[1],
            qp: &sc[2],
            kh: &sc[3],
            dqt: &sc[4],
            dkt: &sc[5],
            dqi: &sc[6],
            dki: &sc[7],
            a: &sc[8],
        };
        let cmd = Cmd::new(c);
        cmd.set_hk_carry(true);
        if gemm {
            cmd.hk_forward_gemm(&d, &w, &hs);
        } else {
            cmd.hk_forward(&d, &w);
        }
        cmd.set_hk_carry(false);
        if gemm {
            cmd.hk_backward_gemm(&d, &w, &gr, &hs, 0.0);
        } else {
            cmd.hk_backward(&d, &w, &gr, 0.0);
        }
        cmd.commit();
        // final state = checkpoint slot nch
        let st = states.to_vec();
        let s_end: Vec<f32> = (0..d.b * d.nh)
            .flat_map(|bh| st[(bh * (nch + 1) + nch) * p2 * d.dv..(bh * (nch + 1) + nch + 1) * p2 * d.dv].to_vec())
            .collect();
        let e = [
            ("out", rel(&gout.to_vec(), &want_o)),
            ("S_T", rel(&s_end, &want_s)),
            ("dthq", rel(&gdthq.to_vec(), &want_q)),
            ("dthk", rel(&gdthk.to_vec(), &want_k)),
            ("dv", rel(&gdv.to_vec(), &want_v)),
            ("dkappa", rel(&gdkap.to_vec(), &want_kap)),
        ];
        eprintln!("hk from S_0 ≠ 0 ({}): {:?}", if gemm { "GEMM form" } else { "scan" }, e);
        for (name, v) in e {
            assert!(v <= 1e-5, "hk carry ({}): {name} rel {v:e}", if gemm { "gemm" } else { "scan" });
        }
    }
}

fn tiny_carry_cfg(gdn: bool) -> EmbryoCfg {
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    cfg.anchor_window = 16;
    cfg.anchor_sink = 2;
    if gdn {
        cfg.mixer = Mixer::Gdn;
        cfg.gdn_heads = 2;
        cfg.gdn_dk = 32;
        cfg.gdn_dv = 32;
    } else {
        cfg.conv_k = 4;
    }
    cfg
}

fn tokens_for(cfg: &EmbryoCfg, seed: u64, n: usize) -> Vec<u32> {
    lcg_vec(seed, n)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect()
}

/// Two carried windows of 64 ≡ one pass of 128 on the second window's
/// per-position NLL (hybrid_k + conv + bounded anchor; GDN + bounded anchor).
#[test]
fn two_carried_windows_match_one_pass() {
    let Some(_) = ctx() else { return };
    for gdn in [false, true] {
        let cfg = tiny_carry_cfg(gdn);
        let (b, t) = (2usize, 64usize);
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 41);
        let tokens = tokens_for(&cfg, 4101, b * 2 * t);
        let targets = tokens_for(&cfg, 4102, b * 2 * t);
        // one pass of 2T (rows of 128); dropless routing on both models —
        // the capacity (2M/E per expert) would otherwise drop different tokens
        let one = EmbryoGpu::new_eval_dropless(cfg.clone(), b, 2 * t, &p0).expect("one-pass model");
        one.desc_updates.set(false);
        let _ = one.eval_loss(&tokens, &targets);
        let per_one = one.per_position_loss();
        // two carried windows of T
        let mut two = EmbryoGpu::new_carry(cfg.clone(), b, t, &p0, true).expect("carried model");
        two.desc_updates.set(false);
        let split = |x: &[u32], w: usize| -> Vec<u32> {
            (0..b).flat_map(|bi| x[bi * 2 * t + w * t..bi * 2 * t + (w + 1) * t].to_vec()).collect()
        };
        two.carry_begin(&[true, true]);
        let _ = two.eval_loss(&split(&tokens, 0), &split(&targets, 0));
        let per_w1 = two.per_position_loss();
        two.carry_commit();
        // a held-out evaluation between the windows (as in training) must
        // neither read nor disturb the carried state
        let _ = two.eval_loss(&split(&targets, 1), &split(&tokens, 1));
        two.carry_begin(&[false, false]);
        let _ = two.eval_loss(&split(&tokens, 1), &split(&targets, 1));
        let per_w2 = two.per_position_loss();
        two.carry_commit();
        let mut worst1 = 0.0f64;
        let mut worst2 = 0.0f64;
        for bi in 0..b {
            for i in 0..t {
                let a1 = per_one[bi * 2 * t + i] as f64;
                let c1 = per_w1[bi * t + i] as f64;
                worst1 = worst1.max((a1 - c1).abs() / a1.abs().max(1e-6));
                let a2 = per_one[bi * 2 * t + t + i] as f64;
                let c2 = per_w2[bi * t + i] as f64;
                worst2 = worst2.max((a2 - c2).abs() / a2.abs().max(1e-6));
            }
        }
        eprintln!(
            "carry continuation ({}): window 1 vs one-pass[0..64] rel {worst1:.2e}; window 2 (carried) vs one-pass[64..128] rel {worst2:.2e}",
            if gdn { "GDN + bounded anchor" } else { "hybrid_k + conv4 + bounded anchor" }
        );
        assert!(worst1 <= 2e-5, "window 1 differs from the one-pass prefix: {worst1:e}");
        assert!(worst2 <= 2e-5, "carried window 2 differs from the one-pass suffix: {worst2:e}");
        // and a fresh (reset) second window must NOT match the one-pass suffix
        two.carry_begin(&[true, true]);
        let _ = two.eval_loss(&split(&tokens, 1), &split(&targets, 1));
        let per_fresh = two.per_position_loss();
        two.carry_commit();
        let diff = (0..b * t)
            .map(|i| (per_fresh[i] - per_w2[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-3, "reset window equals the carried one ({diff:e}) — the carry is a no-op");
    }
}

/// Whole-graph FD on a window that starts from a carried (non-zero) state:
/// the analytic backward against central differences (f32 forward, the
/// precedent bar of tests/model_gradcheck.rs).
#[test]
fn carried_window_every_tensor_matches_finite_differences() {
    let Some(c) = ctx() else { return };
    for gdn in [false, true] {
        let cfg = tiny_carry_cfg(gdn);
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 43);
        let mut gpu = EmbryoGpu::new_carry(cfg.clone(), b, t, &p0, false).expect("gpu");
        gpu.desc_updates.set(false);
        let w1 = tokens_for(&cfg, 4301, m);
        let w1t = tokens_for(&cfg, 4302, m);
        let tokens = tokens_for(&cfg, 4303, m);
        let targets = tokens_for(&cfg, 4304, m);
        // window 1 (fresh) → carried state; window 2 is the checked one
        gpu.carry_begin(&[true, true]);
        let _ = gpu.train_step(&w1, &w1t, 0.0, 0.0, 1e9);
        gpu.carry_commit();
        gpu.carry_begin(&[false, false]);
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), gpu.tok.buf.contents() as *mut u32, m);
            std::ptr::copy_nonoverlapping(targets.as_ptr(), gpu.tgt.buf.contents() as *mut u32, m);
        }
        gpu.prepare_head(&targets);
        let cmd = Cmd::new(c);
        gpu.encode_fwd_bwd(&cmd);
        cmd.commit();
        let g = gpu.grads_host();
        gpu.route_frozen.set(true);
        let l0 = gpu.eval_loss(&tokens, &targets);
        let mut worst = 0.0f64;
        let mut seed = 500u64;
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
                let r = (fd - analytic).abs() / gnorm.max(5e-4);
                worst = worst.max(r);
                assert!(r < 3e-2, "{name} dir {dir}: fd {fd} vs analytic {analytic} (|g| {gnorm})");
            }
        }
        gpu.set_params(&p0);
        gpu.carry_commit();
        eprintln!(
            "carried-window FD ({}): loss {l0:.5}, worst err/|g| = {worst:.2e}",
            if gdn { "GDN" } else { "hybrid_k" }
        );
    }
}

/// The sampler: consecutive windows per row, resets at the start, after
/// `reset_every` windows and after an end-of-text window.
#[test]
fn stream_sampler_windows_are_consecutive_and_reset() {
    use cortiq_embryo::train::{Mix, Shard, StreamSampler};
    let toks: Vec<u16> = (0..4000u16).collect();
    let mut sh = Shard { tokens: toks.clone() };
    sh.tokens[1000] = 9999; // an "EOT"
    let mix = Mix {
        shards: vec![sh],
        weights: vec![1.0],
    };
    let mut ss = StreamSampler::new(2, 16, 7, 3, Some(9999));
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let r0 = ss.batch_mix(&mix, &mut tk, &mut tg);
    assert_eq!(r0, vec![true, true]);
    let first: Vec<u32> = tk.clone();
    let r1 = ss.batch_mix(&mix, &mut tk, &mut tg);
    assert_eq!(r1, vec![false, false]);
    // consecutive: window 2 starts where window 1 ended (+16) unless it hit the EOT
    for row in 0..2 {
        let a = first[row * 16];
        let b = tk[row * 16];
        assert!(b == a + 16 || (a..a + 16).contains(&9999) || (a + 16..a + 32).contains(&9999), "row {row}: {a} → {b}");
    }
    let r2 = ss.batch_mix(&mix, &mut tk, &mut tg);
    let r3 = ss.batch_mix(&mix, &mut tk, &mut tg);
    // reset_every = 3 → the 4th window of an uninterrupted row restarts
    assert!(r3.iter().any(|&x| x) || r2.iter().any(|&x| x));
    assert_eq!(tg[..15], tk[1..16], "targets are the next tokens");
}

/// Rows of a batch are independent sequences: row 0's per-position NLL must
/// not depend on how many other rows share the forward (B=1 vs 2 vs 4, same
/// T), nor on the row-major pool size M.  The production witness (bounded-500,
/// Vulkan) showed a 2e-2 nat shift of rows 0-1 between M=2048 and M=4096
/// forwards, entirely in the anchor layer.
#[test]
fn batch_rows_are_independent() {
    let Some(_) = ctx() else { return };
    // CMF_ROWS_TEST_BIG=1: the production shape of the S4 bounded arm
    // (hidden 384, 8/2 GQA heads of 128, window 128, 4 sinks, T=1024) with two
    // layers, the anchor on the last one, random parameters
    let big = std::env::var("CMF_ROWS_TEST_BIG").is_ok();
    let only = std::env::var("CMF_ROWS_TEST_ONLY").unwrap_or_default();
    for gdn in [false, true] {
        if (only == "gdn" && !gdn) || (only == "hk" && gdn) {
            continue;
        }
        let (cfg, t) = if big {
            let mut cfg = EmbryoCfg::tiny();
            cfg.hidden = 384;
            cfg.layers = 2;
            cfg.anchor_every = 2;
            cfg.heads = 8;
            cfg.nphase = 32;
            cfg.dv = 128;
            cfg.anchor_q_heads = 8;
            cfg.anchor_kv_heads = 2;
            cfg.anchor_hd = 128;
            cfg.experts = 4;
            cfg.inter = 768;
            // tiny vocab 4096 = 64 clusters × 64 (the hierarchical head wants
            // both multiples of 64)
            cfg.head_clusters = 64;
            cfg.mtp_heads = 2;
            cfg.seq = 1024;
            cfg.conv_k = if gdn { 0 } else { 4 };
            // bisection knobs: CMF_ROWS_TEST_ANCHOR=0 removes the anchor
            // layer, _WINDOW (default 128; 0 = legacy full-causal anchor),
            // _SINK (default 4)
            let knob = |k: &str, d: usize| -> usize {
                std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d)
            };
            if knob("CMF_ROWS_TEST_ANCHOR", 1) == 0 {
                cfg.anchor_layers = Some(Vec::new());
            }
            cfg.anchor_window = knob("CMF_ROWS_TEST_WINDOW", 128);
            cfg.anchor_sink = knob("CMF_ROWS_TEST_SINK", 4);
            cfg.anchor_train_windows = if cfg.anchor_window > 0 { vec![64, 128] } else { Vec::new() };
            if gdn {
                cfg.mixer = Mixer::Gdn;
                cfg.gdn_heads = 4;
                cfg.gdn_dk = 128;
                cfg.gdn_dv = 128;
            }
            (cfg, 1024usize)
        } else {
            (tiny_carry_cfg(gdn), 64usize)
        };
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 43);
        let batches: Vec<usize> = std::env::var("CMF_ROWS_TEST_BATCHES")
            .ok()
            .map(|x| x.split(',').filter_map(|v| v.parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 4]);
        let bmax = batches.iter().copied().max().unwrap_or(4).max(4);
        let tokens = tokens_for(&cfg, 4301, bmax * t);
        let targets = tokens_for(&cfg, 4302, bmax * t);
        let mut per: Vec<Vec<f32>> = Vec::new();
        for b in batches.iter().copied() {
            let m = EmbryoGpu::new_eval_dropless(cfg.clone(), b, t, &p0).expect("eval model");
            m.desc_updates.set(false);
            let _ = m.eval_loss(&tokens[..b * t], &targets[..b * t]);
            per.push(m.per_position_loss()[..t].to_vec());
        }
        let worst = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs() as f64 / (a.abs() as f64).max(1e-6))
                .fold(0.0f64, f64::max)
        };
        let ds: Vec<String> = (1..per.len())
            .map(|i| format!("B={} vs B={} rel {:.2e}", batches[0], batches[i], worst(&per[0], &per[i])))
            .collect();
        let worst_all = (1..per.len()).map(|i| worst(&per[0], &per[i])).fold(0.0f64, f64::max);
        eprintln!(
            "row 0 NLL, {} ({}{})",
            ds.join(", "),
            if gdn { "GDN" } else { "hybrid_k + conv4" },
            if cfg.is_anchor(cfg.layers - 1) { " + bounded anchor" } else { ", no anchor" }
        );
        assert!(worst_all <= 2e-5, "row 0 depends on the batch: {worst_all:e}");
    }
}
