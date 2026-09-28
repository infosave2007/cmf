//! GDN chunked WY/UT form (Vulkan only) — the f64 witness the plan demands
//! BEFORE any use: the production tile (B8/T1024, nv 4, dk = dv = 128) with
//! S_0 ≠ 0 against the f64 token reference (`ops::gdn_scan_ref_*`) on O,
//! the chunk-boundary states, dQ/dK/dV (through the q/k norms), dβ (db),
//! dA_log / d dt_bias and dS_0, ≤ 5e-3 rel; plus the same tile against the
//! token-scan kernels (every checkpoint state), and a small tile. The
//! per-layer scan-vs-WY timings are printed (they are only meaningful on an
//! idle GPU).
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::gdn_wy::{GdnWyKeep, GdnWyScratch, wy_bwd, wy_fwd};
use cortiq_embryo::metal::{Cmd, GBuf, GdnScanDims, ctx};
use cortiq_embryo::model::GDN_AB_PAD;
use cortiq_embryo::ops::{gdn_scan_ref_bwd, gdn_scan_ref_fwd, lcg_vec};

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
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
fn rel32(got: &[f32], want: &[f32]) -> f64 {
    rel(got, &to64(want))
}

struct Fix {
    d: GdnScanDims,
    qkv_cv: Vec<f32>,
    a_pre: Vec<f32>,
    b_pre: Vec<f32>,
    p: Vec<f32>, // [alog | dt]
    s0: Vec<f32>,
    doo: Vec<f32>,
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
    let a_pre: Vec<f32> = lcg_vec(seed + 2, rows * GDN_AB_PAD).iter().map(|x| 1.5 * x).collect();
    let b_pre: Vec<f32> = lcg_vec(seed + 3, rows * GDN_AB_PAD).iter().map(|x| 2.0 * x).collect();
    // horizons 8..2048 on the log grid (the plan's init)
    let mut p: Vec<f32> = (0..nv)
        .map(|h| -((8.0f64 * 256f64.powf(h as f64 / (nv.max(2) - 1) as f64)).ln()) as f32)
        .collect();
    p.extend(std::iter::repeat((std::f32::consts::E - 1.0).ln()).take(nv));
    let s0: Vec<f32> = lcg_vec(seed + 4, b * nv * dk * dv).iter().map(|x| 0.3 * x).collect();
    let doo = lcg_vec(seed + 5, rows * nv * dv);
    Fix {
        d,
        qkv_cv,
        a_pre,
        b_pre,
        p,
        s0,
        doo,
    }
}

struct Bufs {
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

fn bufs(c: &'static cortiq_embryo::metal::Ctx, f: &Fix) -> Bufs {
    let d = &f.d;
    let z = |n: usize| GBuf::zeros(c, n);
    let nst = d.b * d.nv * (d.nch() + 1) * d.state();
    Bufs {
        qkv_cv: GBuf::from_slice(c, &f.qkv_cv),
        a_pre: GBuf::from_slice(c, &f.a_pre),
        b_pre: GBuf::from_slice(c, &f.b_pre),
        p: GBuf::from_slice(c, &f.p),
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

fn seed_s0(a: &Bufs, f: &Fix) {
    let d = &f.d;
    let ss = d.state();
    let mut st = a.states.to_vec();
    for bh in 0..d.b * d.nv {
        let base = bh * (d.nch() + 1) * ss;
        st[base..base + ss].copy_from_slice(&f.s0[bh * ss..(bh + 1) * ss]);
    }
    a.states.write_from(&st);
}

struct Outs {
    raw_o: Vec<f32>,
    states: Vec<f32>,
    dcv: Vec<f32>,
    da: Vec<f32>,
    db: Vec<f32>,
    g: Vec<f32>,
    ds0: Vec<f32>,
    ms_fwd: f64,
    ms_bwd: f64,
}

fn run_scan(c: &'static cortiq_embryo::metal::Ctx, a: &Bufs, f: &Fix) -> Outs {
    let d = &f.d;
    seed_s0(a, f);
    let cmd = Cmd::new(c);
    cmd.gdn_scan_fwd(d, &a.qkv_cv, &a.a_pre, &a.b_pre, &a.p, 0, d.nv, &a.raw_o, &a.states, &a.live, true, false);
    let ms_fwd = cmd.commit();
    let cmd = Cmd::new(c);
    cmd.gdn_scan_bwd(
        d, &a.qkv_cv, &a.a_pre, &a.b_pre, &a.p, 0, d.nv, &a.states, &a.doo, &a.chunk, &a.dlive, false, false,
        &a.dcv, &a.da, &a.db, &a.part,
    );
    cmd.axpby(0.0, &a.g, 0.0, &a.g, 2 * d.nv);
    cmd.gdn_scan_fold(&a.part, d.b, d.nv, &a.g, 0, d.nv);
    let ms_bwd = cmd.commit();
    Outs {
        raw_o: a.raw_o.to_vec(),
        states: a.states.to_vec(),
        dcv: a.dcv.to_vec(),
        da: a.da.to_vec(),
        db: a.db.to_vec(),
        g: a.g.to_vec(),
        ds0: a.dlive.to_vec(),
        ms_fwd,
        ms_bwd,
    }
}

fn run_wy(c: &'static cortiq_embryo::metal::Ctx, a: &Bufs, w: &GdnWyKeep, x: &GdnWyScratch, f: &Fix) -> Outs {
    let d = &f.d;
    seed_s0(a, f);
    a.raw_o.fill(0.0);
    a.dcv.fill(0.0);
    a.da.fill(0.0);
    a.db.fill(0.0);
    let cmd = Cmd::new(c);
    wy_fwd(&cmd, d, w, &a.qkv_cv, &a.a_pre, &a.b_pre, &a.p, 0, d.nv, &a.raw_o, &a.states, true, false);
    let ms_fwd = cmd.commit();
    let cmd = Cmd::new(c);
    wy_bwd(
        &cmd, d, w, x, &a.qkv_cv, &a.a_pre, &a.p, 0, d.nv, &a.states, &a.doo, &a.dlive, false, false, &a.dcv,
        &a.da, &a.db, &a.part,
    );
    cmd.axpby(0.0, &a.g, 0.0, &a.g, 2 * d.nv);
    cmd.gdn_scan_fold(&a.part, d.b, d.nv, &a.g, 0, d.nv);
    let ms_bwd = cmd.commit();
    Outs {
        raw_o: a.raw_o.to_vec(),
        states: a.states.to_vec(),
        dcv: a.dcv.to_vec(),
        da: a.da.to_vec(),
        db: a.db.to_vec(),
        g: a.g.to_vec(),
        ds0: a.dlive.to_vec(),
        ms_fwd,
        ms_bwd,
    }
}

fn pick_ab(x: &[f32], rows: usize, nv: usize) -> Vec<f32> {
    (0..rows).flat_map(|r| (0..nv).map(move |h| x[r * GDN_AB_PAD + h])).collect()
}

fn witness(b: usize, t: usize, nv: usize, dk: usize, dv: usize, tol: f64) {
    let Some(c) = ctx() else { return };
    let f = fixture(b, t, nv, dk, dv, 4000 + t as u64);
    let d = f.d;
    let a = bufs(c, &f);
    let w = GdnWyKeep::new(c, &d);
    let x = GdnWyScratch::new(c, &d);
    // warm-up + timing (scan, then WY)
    let sc = run_scan(c, &a, &f);
    let wy = run_wy(c, &a, &w, &x, &f);
    let sc2 = run_scan(c, &a, &f);
    let wy2 = run_wy(c, &a, &w, &x, &f);
    eprintln!(
        "tile B{b}/T{t} nv{nv} dk{dk} dv{dv}: scan fwd {:.1} ms bwd {:.1} ms | WY fwd {:.1} ms bwd {:.1} ms (GPU ms, meaningful on an idle GPU only)",
        sc2.ms_fwd, sc2.ms_bwd, wy2.ms_fwd, wy2.ms_bwd
    );
    assert_eq!(wy.raw_o.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), wy2.raw_o.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "WY forward not deterministic");
    let _ = sc;
    // f64 reference
    let alog = to64(&f.p[..nv]);
    let dt = to64(&f.p[nv..2 * nv]);
    let want = gdn_scan_ref_fwd(&d, &to64(&f.qkv_cv), &to64(&f.a_pre), &to64(&f.b_pre), &alog, &dt, &to64(&f.s0));
    let wg = gdn_scan_ref_bwd(&d, &to64(&f.qkv_cv), &to64(&f.a_pre), &to64(&f.b_pre), &alog, &dt, &to64(&f.s0), &to64(&f.doo), None);
    let rows = d.rows();
    let ss = d.state();
    // final states of every (b, h) from the checkpoint array
    let s_end: Vec<f32> = (0..b * nv)
        .flat_map(|bh| wy2.states[(bh * (d.nch() + 1) + d.nch()) * ss..(bh * (d.nch() + 1) + d.nch() + 1) * ss].to_vec())
        .collect();
    let cd = d.c_dim;
    let cols = |x: &[f32], lo: usize, hi: usize| -> Vec<f32> {
        (0..rows).flat_map(|r| x[r * cd + lo..r * cd + hi].to_vec()).collect()
    };
    let cols64 = |x: &[f64], lo: usize, hi: usize| -> Vec<f64> {
        (0..rows).flat_map(|r| x[r * cd + lo..r * cd + hi].to_vec()).collect()
    };
    let e = [
        ("O", rel(&wy2.raw_o, &want.raw_o)),
        ("S_T", rel(&s_end, &want.s_end)),
        ("dQ", rel(&cols(&wy2.dcv, 0, nv * dk), &cols64(&wg.dcv, 0, nv * dk))),
        ("dK", rel(&cols(&wy2.dcv, nv * dk, 2 * nv * dk), &cols64(&wg.dcv, nv * dk, 2 * nv * dk))),
        ("dV", rel(&cols(&wy2.dcv, 2 * nv * dk, cd), &cols64(&wg.dcv, 2 * nv * dk, cd))),
        ("da", rel(&pick_ab(&wy2.da, rows, nv), &wg.da)),
        ("dβ(db)", rel(&pick_ab(&wy2.db, rows, nv), &wg.db)),
        ("dA_log", rel(&wy2.g[..nv], &wg.dalog)),
        ("ddt", rel(&wy2.g[nv..2 * nv], &wg.ddt)),
        ("dS_0", rel(&wy2.ds0, &wg.ds0)),
    ];
    eprintln!("WY vs f64 (B{b}/T{t} nv{nv} dk{dk} dv{dv}): {:?}", e);
    // WY vs the token-scan kernels: every checkpoint state
    let es = [
        ("states(all checkpoints)", rel32(&wy2.states, &sc2.states)),
        ("O", rel32(&wy2.raw_o, &sc2.raw_o)),
        ("dcv", rel32(&wy2.dcv, &sc2.dcv)),
        ("da", rel32(&wy2.da, &sc2.da)),
        ("db", rel32(&wy2.db, &sc2.db)),
        ("dA_log/ddt", rel32(&wy2.g, &sc2.g)),
    ];
    eprintln!("WY vs scan kernels: {:?}", es);
    for (name, v) in e.iter().chain(es.iter()) {
        assert!(*v <= tol, "B{b}/T{t}: {name} rel {v:e} > {tol:e}");
    }
}

/// The production tile of the plan's gate (dthk 1.40 / dv 2.68 / beta 1.11
/// rel broke the previous WY here): ≤ 5e-3 rel on everything.
#[test]
fn wy_production_tile_matches_f64_and_scan() {
    witness(8, 1024, 4, 128, 128, 5e-3);
}

/// Small tile (two chunks, sub-tile head dims → the 16×16 GEMM path).
#[test]
fn wy_small_tile_matches_f64_and_scan() {
    witness(2, 128, 2, 32, 32, 5e-3);
}

/// Model-level regression: a genome with TWO GDN layers (the bug the
/// step-bench exposed — one shared intermediates arena handed layer 0's
/// backward layer 1's tables): WY vs scan gradients of every tensor.
#[test]
fn wy_two_gdn_layers_match_scan_gradients() {
    use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, init_params};
    let Some(_) = ctx() else { return };
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 3;
    cfg.anchor_every = 3; // layers 0, 1 = GDN, layer 2 = anchor
    cfg.mixer = Mixer::Gdn;
    cfg.gdn_heads = 2;
    cfg.gdn_dk = 32;
    cfg.gdn_dv = 32;
    cfg.experts = 4;
    let (b, t) = (2usize, 128usize);
    let m = b * t;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 23);
    let tokens: Vec<u32> = lcg_vec(31, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    let targets: Vec<u32> = lcg_vec(32, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    let run = |wy: bool| -> (f32, f32, Vec<f32>) {
        if wy {
            unsafe { std::env::remove_var("CMF_GDN_WY") };
        } else {
            unsafe { std::env::set_var("CMF_GDN_WY", "0") };
        }
        let mut g = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
        g.desc_updates.set(false);
        assert_eq!(g.gdn_wy.is_some(), wy, "CMF_GDN_WY switch");
        let (l, gn, _) = g.train_step(&tokens, &targets, 0.0, 0.0, 1e9);
        (l, gn, g.grads_host())
    };
    let (l_wy, gn_wy, g_wy) = run(true);
    let (l_sc, gn_sc, g_sc) = run(false);
    unsafe { std::env::remove_var("CMF_GDN_WY") };
    eprintln!("two GDN layers: loss WY {l_wy} scan {l_sc}; |g| WY {gn_wy} scan {gn_sc}");
    assert!((l_wy - l_sc).abs() <= 1e-5 * l_sc.abs(), "loss differs");
    let mut worst = 0.0f64;
    for (name, off, n) in &lay.names {
        let a = &g_wy[*off..*off + n];
        let c = &g_sc[*off..*off + n];
        let scale = c.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12) as f64;
        let e = a.iter().zip(c).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max) / scale;
        worst = worst.max(e);
        assert!(e <= 2e-5, "{name}: WY vs scan gradient rel {e:e}");
    }
    eprintln!("two GDN layers: worst per-tensor rel {worst:.2e}; |g| rel {:.2e}", ((gn_wy - gn_sc) / gn_sc).abs());
}
