//! GDN mixer (plan S7, variant B): the token-scan kernels against the f64
//! reference, continuation ≡ one pass, bit-exact determinism, and the
//! whole-graph finite-difference check of a tiny GDN genome. The same file
//! runs on Metal (macOS) and on Vulkan (`--features vulkan`, Linux): the
//! backend is whatever `cortiq_embryo::metal::ctx()` resolves to.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::metal::{Cmd, GBuf, GdnScanDims, ctx};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, GDN_AB_PAD, Layout, Mixer, gauss_vec, init_params};
use cortiq_embryo::ops::{gdn_scan_ref_bwd, gdn_scan_ref_fwd, lcg_vec};

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}
fn to32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

/// max|got − want| / max|want|
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max)
        / scale
}

struct Fix {
    d: GdnScanDims,
    qkv_cv: Vec<f32>,
    a_pre: Vec<f32>,
    b_pre: Vec<f32>,
    alog: Vec<f32>,
    dt: Vec<f32>,
    s0: Vec<f32>,
    doo: Vec<f32>,
    ds_end: Vec<f32>,
}

fn fixture(b: usize, t: usize, nv: usize, dk: usize, dv: usize, seed: u64) -> Fix {
    let c_dim = 2 * nv * dk + nv * dv;
    let d = GdnScanDims {
        b,
        t,
        nv,
        dk,
        dv,
        c_dim,
        ab_ld: GDN_AB_PAD,
    };
    let rows = b * t;
    let qkv_cv = lcg_vec(seed + 1, rows * c_dim);
    // a/b pre-activations around 0 (β ≈ ½, decay near its horizon)
    let a_pre: Vec<f32> = lcg_vec(seed + 2, rows * GDN_AB_PAD).iter().map(|x| 1.5 * x).collect();
    let b_pre: Vec<f32> = lcg_vec(seed + 3, rows * GDN_AB_PAD).iter().map(|x| 2.0 * x).collect();
    // horizons 4..64 → A_log = −ln H
    let alog: Vec<f32> = (0..nv)
        .map(|h| -((4.0f64 * 16f64.powf(h as f64 / (nv.max(2) - 1) as f64)).ln()) as f32)
        .collect();
    let dt = vec![(std::f32::consts::E - 1.0).ln(); nv];
    let s0: Vec<f32> = lcg_vec(seed + 4, b * nv * dk * dv).iter().map(|x| 0.3 * x).collect();
    let doo = lcg_vec(seed + 5, rows * nv * dv);
    let ds_end: Vec<f32> = lcg_vec(seed + 6, b * nv * dk * dv).iter().map(|x| 0.5 * x).collect();
    Fix {
        d,
        qkv_cv,
        a_pre,
        b_pre,
        alog,
        dt,
        s0,
        doo,
        ds_end,
    }
}

/// Reference gradients, checked against central finite differences of the
/// reference forward (loss = Σ raw_o·doo + Σ S_T·dS_T) — the oracle itself.
#[test]
fn scan_reference_backward_matches_finite_differences() {
    let f = fixture(1, 5, 2, 4, 3, 300);
    let d = f.d;
    let (qkv, a, b, alog, dt, s0, doo, dse) = (
        to64(&f.qkv_cv),
        to64(&f.a_pre),
        to64(&f.b_pre),
        to64(&f.alog),
        to64(&f.dt),
        to64(&f.s0),
        to64(&f.doo),
        to64(&f.ds_end),
    );
    let loss = |qkv: &[f64], a: &[f64], b: &[f64], alog: &[f64], dt: &[f64], s0: &[f64]| -> f64 {
        let r = gdn_scan_ref_fwd(&d, qkv, a, b, alog, dt, s0);
        r.raw_o.iter().zip(&doo).map(|(x, y)| x * y).sum::<f64>()
            + r.s_end.iter().zip(&dse).map(|(x, y)| x * y).sum::<f64>()
    };
    let g = gdn_scan_ref_bwd(&d, &qkv, &a, &b, &alog, &dt, &s0, &doo, Some(&dse));
    let eps = 1e-5;
    let check = |name: &str, x: &[f64], grad: &[f64], f: &dyn Fn(&[f64]) -> f64, idx: &dyn Fn(usize) -> Option<usize>| {
        let mut worst = 0.0f64;
        for i in 0..x.len() {
            let Some(gi) = idx(i) else { continue };
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
            let err = (fd - grad[gi]).abs() / (1.0 + fd.abs());
            worst = worst.max(err);
        }
        eprintln!("{name}: worst rel err {worst:.2e}");
        assert!(worst < 1e-6, "{name}: worst rel err {worst:e}");
    };
    check("dcv", &qkv, &g.dcv, &|x| loss(x, &a, &b, &alog, &dt, &s0), &|i| Some(i));
    // a/b: reference grads are [rows, nv] (head columns of the padded rows)
    let ab_idx = |i: usize| {
        let (row, col) = (i / GDN_AB_PAD, i % GDN_AB_PAD);
        (col < d.nv).then_some(row * d.nv + col)
    };
    check("da", &a, &g.da, &|x| loss(&qkv, x, &b, &alog, &dt, &s0), &ab_idx);
    check("db", &b, &g.db, &|x| loss(&qkv, &a, x, &alog, &dt, &s0), &ab_idx);
    check("dalog", &alog, &g.dalog, &|x| loss(&qkv, &a, &b, x, &dt, &s0), &|i| Some(i));
    check("ddt", &dt, &g.ddt, &|x| loss(&qkv, &a, &b, &alog, x, &s0), &|i| Some(i));
    check("ds0", &s0, &g.ds0, &|x| loss(&qkv, &a, &b, &alog, &dt, x), &|i| Some(i));
}

struct Arena {
    qkv_cv: GBuf,
    a_pre: GBuf,
    b_pre: GBuf,
    p: GBuf,
    raw_o: GBuf,
    states: GBuf,
    live: GBuf,
    doo: GBuf,
    chunk: GBuf,
    dlive: GBuf,
    dcv: GBuf,
    da: GBuf,
    db: GBuf,
    part: GBuf,
    g: GBuf,
}

/// `p` = [alog | dt] (alog at 0, dt at nv); `g` the same layout for the fold.
fn arena(c: &'static cortiq_embryo::metal::Ctx, f: &Fix) -> Arena {
    let d = &f.d;
    let z = |n: usize| GBuf::zeros(c, n);
    let mut p = f.alog.clone();
    p.extend_from_slice(&f.dt);
    let nst = d.b * d.nv * (d.nch() + 1) * d.state();
    Arena {
        qkv_cv: GBuf::from_slice(c, &f.qkv_cv),
        a_pre: GBuf::from_slice(c, &f.a_pre),
        b_pre: GBuf::from_slice(c, &f.b_pre),
        p: GBuf::from_slice(c, &p),
        raw_o: z(d.rows() * d.nv * d.dv),
        states: z(nst),
        live: z(d.b * d.nv * d.state()),
        doo: GBuf::from_slice(c, &f.doo),
        chunk: z(d.b * d.nv * 65 * d.state()),
        dlive: z(d.b * d.nv * d.state()),
        dcv: z(d.rows() * d.c_dim),
        da: z(d.rows() * GDN_AB_PAD),
        db: z(d.rows() * GDN_AB_PAD),
        part: z(d.b * d.nv * 2),
        g: z(2 * d.nv),
    }
}

/// Write S_0 into checkpoint slot 0 of every (b, head) and dS_T into dlive.
fn seed_states(a: &Arena, f: &Fix) {
    let d = &f.d;
    let ss = d.state();
    let mut st = a.states.to_vec();
    for bh in 0..d.b * d.nv {
        let base = bh * (d.nch() + 1) * ss;
        st[base..base + ss].copy_from_slice(&f.s0[bh * ss..(bh + 1) * ss]);
    }
    a.states.write_from(&st);
    a.dlive.write_from(&f.ds_end);
}

fn run_fwd_bwd(c: &'static cortiq_embryo::metal::Ctx, a: &Arena, f: &Fix, s0: bool, ds: bool) -> f64 {
    let d = &f.d;
    let cmd = Cmd::new(c);
    cmd.gdn_scan_fwd(d, &a.qkv_cv, &a.a_pre, &a.b_pre, &a.p, 0, d.nv, &a.raw_o, &a.states, &a.live, s0, false);
    cmd.gdn_scan_bwd(
        d, &a.qkv_cv, &a.a_pre, &a.b_pre, &a.p, 0, d.nv, &a.states, &a.doo, &a.chunk, &a.dlive, ds, false,
        &a.dcv, &a.da, &a.db, &a.part,
    );
    cmd.axpby(0.0, &a.g, 0.0, &a.g, 2 * d.nv);
    cmd.gdn_scan_fold(&a.part, d.b, d.nv, &a.g, 0, d.nv);
    cmd.commit()
}

/// Kernels vs the f64 reference with S_0 ≠ 0 and dS_T ≠ 0 (tiny geometry,
/// two chunk lengths).
#[test]
fn scan_kernels_match_f64_reference() {
    let Some(c) = ctx() else { return };
    for &(t, nv, dk, dv) in &[(64usize, 2usize, 32usize, 32usize), (128, 2, 32, 32), (64, 3, 16, 24)] {
        let f = fixture(2, t, nv, dk, dv, 700 + t as u64);
        let d = f.d;
        let want = gdn_scan_ref_fwd(&d, &to64(&f.qkv_cv), &to64(&f.a_pre), &to64(&f.b_pre), &to64(&f.alog), &to64(&f.dt), &to64(&f.s0));
        let wg = gdn_scan_ref_bwd(
            &d, &to64(&f.qkv_cv), &to64(&f.a_pre), &to64(&f.b_pre), &to64(&f.alog), &to64(&f.dt), &to64(&f.s0), &to64(&f.doo), Some(&to64(&f.ds_end)),
        );
        let a = arena(c, &f);
        seed_states(&a, &f);
        run_fwd_bwd(c, &a, &f, true, true);
        let e_o = rel(&a.raw_o.to_vec(), &want.raw_o);
        let e_s = rel(&a.live.to_vec(), &want.s_end);
        let e_cv = rel(&a.dcv.to_vec(), &wg.dcv);
        // da/db: head columns of the padded rows
        let pick = |x: &[f32]| -> Vec<f32> {
            (0..d.rows()).flat_map(|r| (0..nv).map(move |h| x[r * GDN_AB_PAD + h])).collect()
        };
        let e_a = rel(&pick(&a.da.to_vec()), &wg.da);
        let e_b = rel(&pick(&a.db.to_vec()), &wg.db);
        let g = a.g.to_vec();
        let e_al = rel(&g[..nv], &wg.dalog);
        let e_dt = rel(&g[nv..2 * nv], &wg.ddt);
        let e_s0 = rel(&a.dlive.to_vec(), &wg.ds0);
        // pad columns of da/db must stay exactly zero
        let da = a.da.to_vec();
        assert!((0..d.rows()).all(|r| (nv..GDN_AB_PAD).all(|h| da[r * GDN_AB_PAD + h] == 0.0)));
        eprintln!(
            "scan T={t} nv={nv} dk={dk} dv={dv}: raw_o {e_o:.2e} S_T {e_s:.2e} dcv {e_cv:.2e} da {e_a:.2e} db {e_b:.2e} dA_log {e_al:.2e} ddt {e_dt:.2e} dS_0 {e_s0:.2e}"
        );
        for (name, e) in [("raw_o", e_o), ("S_T", e_s), ("dcv", e_cv), ("da", e_a), ("db", e_b), ("dA_log", e_al), ("ddt", e_dt), ("dS_0", e_s0)] {
            assert!(e < 1e-5, "T={t}: {name} rel {e:e}");
        }
    }
}

/// One pass over T ≡ two continued passes (S carried through checkpoint 0
/// forward, dS_T through dlive backward): T = 128 = 64 + 64 and
/// T = 1088 = 1024 + 64.
#[test]
fn scan_continuation_matches_one_pass() {
    let Some(c) = ctx() else { return };
    for &(t1, t2) in &[(64usize, 64usize), (1024, 64)] {
        let t = t1 + t2;
        let (nv, dk, dv) = (2, 32, 32);
        let f = fixture(1, t, nv, dk, dv, 900 + t as u64);
        let d = f.d;
        let a = arena(c, &f);
        seed_states(&a, &f);
        run_fwd_bwd(c, &a, &f, true, true);
        let (o_all, cv_all, da_all, db_all, g_all, ds0_all) = (
            a.raw_o.to_vec(), a.dcv.to_vec(), a.da.to_vec(), a.db.to_vec(), a.g.to_vec(), a.dlive.to_vec(),
        );
        // segment helper
        let seg = |lo: usize, hi: usize, s0: &[f32], dse: &[f32]| -> Fix {
            let rows = |x: &[f32], w: usize| x[lo * w..hi * w].to_vec();
            Fix {
                d: GdnScanDims { t: hi - lo, ..d },
                qkv_cv: rows(&f.qkv_cv, d.c_dim),
                a_pre: rows(&f.a_pre, GDN_AB_PAD),
                b_pre: rows(&f.b_pre, GDN_AB_PAD),
                alog: f.alog.clone(),
                dt: f.dt.clone(),
                s0: s0.to_vec(),
                doo: rows(&f.doo, nv * dv),
                ds_end: dse.to_vec(),
            }
        };
        // first segment forward (state out), second segment fwd+bwd, first segment bwd with dS from the second
        let f1 = seg(0, t1, &f.s0, &f.ds_end);
        let a1 = arena(c, &f1);
        seed_states(&a1, &f1);
        let cmd = Cmd::new(c);
        cmd.gdn_scan_fwd(&f1.d, &a1.qkv_cv, &a1.a_pre, &a1.b_pre, &a1.p, 0, nv, &a1.raw_o, &a1.states, &a1.live, true, false);
        cmd.commit();
        let s_mid = a1.live.to_vec();
        let f2 = seg(t1, t, &s_mid, &f.ds_end);
        let a2 = arena(c, &f2);
        seed_states(&a2, &f2);
        run_fwd_bwd(c, &a2, &f2, true, true);
        let ds_mid = a2.dlive.to_vec();
        a1.dlive.write_from(&ds_mid);
        let cmd = Cmd::new(c);
        cmd.gdn_scan_bwd(
            &f1.d, &a1.qkv_cv, &a1.a_pre, &a1.b_pre, &a1.p, 0, nv, &a1.states, &a1.doo, &a1.chunk, &a1.dlive, true, false,
            &a1.dcv, &a1.da, &a1.db, &a1.part,
        );
        cmd.axpby(0.0, &a1.g, 0.0, &a1.g, 2 * nv);
        cmd.gdn_scan_fold(&a1.part, 1, nv, &a1.g, 0, nv);
        cmd.commit();
        let cat = |x1: Vec<f32>, x2: Vec<f32>| -> Vec<f32> { x1.into_iter().chain(x2).collect() };
        let o_two = cat(a1.raw_o.to_vec(), a2.raw_o.to_vec());
        let cv_two = cat(a1.dcv.to_vec(), a2.dcv.to_vec());
        let da_two = cat(a1.da.to_vec(), a2.da.to_vec());
        let db_two = cat(a1.db.to_vec(), a2.db.to_vec());
        let g1 = a1.g.to_vec();
        let g2 = a2.g.to_vec();
        let g_two: Vec<f32> = g1.iter().zip(&g2).map(|(x, y)| x + y).collect();
        let ds0_two = a1.dlive.to_vec();
        let e = [
            ("raw_o", rel(&o_two, &to64(&o_all))),
            ("dcv", rel(&cv_two, &to64(&cv_all))),
            ("da", rel(&da_two, &to64(&da_all))),
            ("db", rel(&db_two, &to64(&db_all))),
            ("dA_log/ddt", rel(&g_two, &to64(&g_all))),
            ("dS_0", rel(&ds0_two, &to64(&ds0_all))),
        ];
        eprintln!("continuation T={t}={t1}+{t2}: {:?}", e);
        for (name, v) in e {
            // per-token outputs and dS_0 are the same f32 operations in the
            // same order (bit-exact); the per-(b,head) dA_log/ddt scalars
            // are one f32 running sum over T tokens vs two partial sums
            // added (measured 6.8e-7 Metal / 1.7e-6 Vulkan at T = 1088)
            let bound = if name == "dA_log/ddt" { 4e-6 } else { 1e-6 };
            assert!(v <= bound, "T={t}: {name} continuation vs one pass rel {v:e}");
        }
    }
}

/// Two identical fwd+bwd passes are bit-identical.
#[test]
fn scan_is_bit_exact_deterministic() {
    let Some(c) = ctx() else { return };
    let f = fixture(3, 128, 4, 32, 32, 1200);
    let a = arena(c, &f);
    let snap = |a: &Arena| -> Vec<Vec<u32>> {
        [&a.raw_o, &a.live, &a.dcv, &a.da, &a.db, &a.part, &a.dlive]
            .iter()
            .map(|b| b.to_vec().iter().map(|x| x.to_bits()).collect())
            .collect()
    };
    seed_states(&a, &f);
    run_fwd_bwd(c, &a, &f, true, true);
    let s1 = snap(&a);
    // scribble the outputs, run again
    a.raw_o.fill(7.0);
    a.dcv.fill(7.0);
    a.chunk.fill(7.0);
    seed_states(&a, &f);
    run_fwd_bwd(c, &a, &f, true, true);
    let s2 = snap(&a);
    assert_eq!(s1, s2, "GDN scan is not bit-exact across runs");
    eprintln!("determinism: {} outputs bit-identical", s1.iter().map(|v| v.len()).sum::<usize>());
}

fn tiny_gdn(experts: usize) -> EmbryoCfg {
    let mut cfg = EmbryoCfg::tiny();
    cfg.mixer = Mixer::Gdn;
    cfg.gdn_heads = 2;
    cfg.gdn_dk = 32;
    cfg.gdn_dv = 32;
    cfg.experts = experts;
    cfg
}

/// Whole-graph FD over every named tensor of a tiny GDN genome (the
/// precedent of tests/model_gradcheck.rs: f32 forward, err/|g| < 3e-2).
#[test]
fn gdn_genome_every_tensor_matches_finite_differences() {
    let Some(c) = ctx() else { return };
    for variant in 0..2 {
        let mut cfg = tiny_gdn(if variant == 1 { 4 } else { 0 });
        if variant == 1 {
            // anchor 1 of 2 is the bounded operator too
            cfg.anchor_window = 16;
            cfg.anchor_sink = 2;
        }
        let (b, t) = (2usize, 64usize);
        let m = b * t;
        let lay = Layout::new(&cfg);
        let p0 = init_params(&cfg, &lay, 7);
        let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
        gpu.desc_updates.set(false);
        let tokens: Vec<u32> = lcg_vec(11, m)
            .iter()
            .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
            .collect();
        let targets: Vec<u32> = lcg_vec(12, m)
            .iter()
            .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
            .collect();
        let (l0, _, _) = {
            let mut g2 = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
            g2.desc_updates.set(false);
            g2.train_step(&tokens, &targets, 0.0, 0.0, 1e9)
        };
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
        eprintln!("=== variant {variant}: experts {} loss {l0:.5} ===", cfg.experts);
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
                let r = (fd - analytic).abs() / denom;
                worst_all = worst_all.max(r);
                eprintln!(
                    "{name:<26} n={n:<7} |g|={gnorm:.3e} dir={} fd={fd:+.4e} an={analytic:+.4e} err/|g|={r:.2e}",
                    if dir == 0 { "grad" } else { "rand" }
                );
                assert!(r < 3e-2, "{name} dir {dir}: fd {fd} vs analytic {analytic} (|g| {gnorm})");
            }
        }
        gpu.set_params(&p0);
        eprintln!("variant {variant}: worst err/|g| = {worst_all:.2e}");
    }
}

/// Legacy configs serialize byte-identically (no mixer fields) and a GDN
/// config round-trips; a legacy JSON without the fields deserializes to
/// HybridK with the default geometry.
#[test]
fn mixer_config_serde_is_backward_compatible() {
    let legacy = EmbryoCfg::embryo0();
    let js = serde_json::to_string(&legacy).unwrap();
    assert!(
        !js.contains("mixer") && !js.contains("gdn_heads") && !js.contains("gdn_dk") && !js.contains("gdn_dv"),
        "{js}"
    );
    let back: EmbryoCfg = serde_json::from_str(&js).unwrap();
    assert_eq!(back.mixer, Mixer::HybridK);
    assert_eq!((back.gdn_heads, back.gdn_dk, back.gdn_dv), (4, 128, 128));
    let mut g = tiny_gdn(0);
    g.gdn_dk = 64;
    let js = serde_json::to_string(&g).unwrap();
    assert!(js.contains("\"mixer\":\"gdn\"") && js.contains("\"gdn_dk\":64"), "{js}");
    let back: EmbryoCfg = serde_json::from_str(&js).unwrap();
    assert_eq!(back.mixer, Mixer::Gdn);
    assert_eq!((back.gdn_heads, back.gdn_dk, back.gdn_dv), (2, 64, 32));
    // the plan's parameter count: 992 392 per GDN layer at H=384, 4×128×128
    let mut e0 = EmbryoCfg::embryo0();
    e0.mixer = Mixer::Gdn;
    let lay = Layout::new(&e0);
    let per_layer: usize = lay
        .names
        .iter()
        .filter(|(n, _, _)| n.starts_with("layers.0.gdn.") || n == "layers.0.ln1" || n == "layers.0.ln2")
        .map(|(n, _, l)| if n.ends_with(".in_a") || n.ends_with(".in_b") { 4 * 384 } else if n.ends_with(".alog") || n.ends_with(".dt_bias") { 4 } else { *l })
        .sum::<usize>()
        - 2 * 384;
    assert_eq!(per_layer, 992_392, "GDN mixer parameters per layer");
}
