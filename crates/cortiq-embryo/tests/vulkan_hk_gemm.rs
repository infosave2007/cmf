//! GEMM-form hybrid_k on the Vulkan backend (S6c item 1): the port of
//! Metal's chunked `hk_forward_gemm` / `hk_backward_gemm` (ops 53-57 +
//! batched GEMMs) against (a) the f64 CPU oracle of the literal recurrence
//! and (b) the resident value-parallel scan (`hk_forward` / `hk_backward`,
//! the bit-exact GPU reference), on a small cross-chunk geometry and on the
//! production geometry B8/T512 (nh 8, nph 32, dv 128) where both paths are
//! also timed. A missing/non-NVIDIA adapter fails the process.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, HkGrads, HkScratch, HkWork, ctx, hk_pow_table};
use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd, hk_ref_fwd, lcg_vec};

fn as_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| *v as f64).collect()
}

/// max|got − want| / max|want|
fn rel_err64(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(x, y)| (*x as f64 - *y).abs())
        .fold(0.0, f64::max)
        / scale
}

fn rel_err32(got: &[f32], want: &[f32]) -> f64 {
    rel_err64(got, &as_f64(want))
}

struct Arena {
    d: HkDims,
    thq: GBuf,
    thk: GBuf,
    v: GBuf,
    kappa: GBuf,
    pow: GBuf,
    phq: GBuf,
    phk: GBuf,
    kv: GBuf,
    states: GBuf,
    out: GBuf,
    chunk: GBuf,
    partial: GBuf,
    dout: GBuf,
    dstates: GBuf,
    dkv: GBuf,
    dphq: GBuf,
    dphk: GBuf,
    dthq: GBuf,
    dthk: GBuf,
    dv: GBuf,
    dkappa: GBuf,
    sc: [GBuf; 9],
    // host copies of the inputs for the oracle
    h_thq: Vec<f32>,
    h_thk: Vec<f32>,
    h_v: Vec<f32>,
    h_kappa: Vec<f32>,
    h_dout: Vec<f32>,
    decay: Vec<f32>,
}

impl Arena {
    fn new(c: &'static cortiq_embryo::metal::Ctx, d: HkDims, seed: u64) -> Arena {
        let rows = d.b * d.t;
        let p2 = d.p2();
        let thq: Vec<f32> = lcg_vec(seed + 1, rows * d.nh * d.nph).iter().map(|x| x * 2.0).collect();
        let thk: Vec<f32> = lcg_vec(seed + 2, rows * d.nh * d.nph).iter().map(|x| x * 2.0).collect();
        let v = lcg_vec(seed + 3, rows * d.nh * d.dv);
        let kappa: Vec<f32> = lcg_vec(seed + 4, rows * d.nh)
            .iter()
            .map(|x| 0.25 + 0.5 * (x + 1.0) / 2.0)
            .collect();
        let dout = lcg_vec(seed + 5, rows * d.nh * d.dv);
        let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
        let z = |n: usize| GBuf::zeros(c, n);
        let nst = d.b * d.nh * (d.t / 64 + 1) * p2 * d.dv;
        let cl = HkScratch::chunk_len(&d);
        Arena {
            thq: GBuf::from_slice(c, &thq),
            thk: GBuf::from_slice(c, &thk),
            v: GBuf::from_slice(c, &v),
            kappa: GBuf::from_slice(c, &kappa),
            pow: GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, d.nph)),
            phq: z(rows * d.nh * p2),
            phk: z(rows * d.nh * p2),
            kv: z(rows * d.nh * d.dv),
            states: z(nst),
            out: z(rows * d.nh * d.dv),
            chunk: z(d.b * d.nh * d.t * (p2 + 1) * d.dv),
            partial: z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + p2)),
            dout: GBuf::from_slice(c, &dout),
            dstates: z(nst),
            dkv: z(rows * d.nh * d.dv),
            dphq: z(rows * d.nh * p2),
            dphk: z(rows * d.nh * p2),
            dthq: z(rows * d.nh * d.nph),
            dthk: z(rows * d.nh * d.nph),
            dv: z(rows * d.nh * d.dv),
            dkappa: z(rows * d.nh),
            sc: [
                z(cl),
                z(cl),
                z(cl),
                z(cl),
                z(cl),
                z(cl),
                z(cl),
                z(cl),
                z(HkScratch::a_len(&d)),
            ],
            h_thq: thq,
            h_thk: thk,
            h_v: v,
            h_kappa: kappa,
            h_dout: dout,
            decay,
            d,
        }
    }
    fn work(&self) -> HkWork<'_> {
        HkWork {
            thq: &self.thq,
            thk: &self.thk,
            v: &self.v,
            kappa: &self.kappa,
            pow: &self.pow,
            pow_off: 0,
            phq: &self.phq,
            phk: &self.phk,
            kv: &self.kv,
            states: &self.states,
            out: &self.out,
            phase_chunk: Some(&self.chunk),
            phase_partial: Some(&self.partial),
        }
    }
    fn grads(&self) -> HkGrads<'_> {
        HkGrads {
            dout: &self.dout,
            dstates: &self.dstates,
            dkv: &self.dkv,
            dphq: &self.dphq,
            dphk: &self.dphk,
            dthq: &self.dthq,
            dthk: &self.dthk,
            dv: &self.dv,
            dkappa: &self.dkappa,
        }
    }
    fn scratch(&self) -> HkScratch<'_> {
        HkScratch {
            qt: &self.sc[0],
            kt: &self.sc[1],
            qp: &self.sc[2],
            kh: &self.sc[3],
            dqt: &self.sc[4],
            dkt: &self.sc[5],
            dqi: &self.sc[6],
            dki: &self.sc[7],
            a: &self.sc[8],
        }
    }
    fn zero_outputs(&self) {
        for x in [
            &self.out,
            &self.states,
            &self.dstates,
            &self.dkv,
            &self.dphq,
            &self.dphk,
            &self.dthq,
            &self.dthk,
            &self.dv,
            &self.dkappa,
        ] {
            x.write_from(&vec![0.0f32; x.len]);
        }
    }
}

struct Outs {
    out: Vec<f32>,
    states: Vec<f32>,
    dstates: Vec<f32>,
    dv: Vec<f32>,
    dthq: Vec<f32>,
    dthk: Vec<f32>,
    dkappa: Vec<f32>,
}

fn read(a: &Arena) -> Outs {
    Outs {
        out: a.out.to_vec(),
        states: a.states.to_vec(),
        dstates: a.dstates.to_vec(),
        dv: a.dv.to_vec(),
        dthq: a.dthq.to_vec(),
        dthk: a.dthk.to_vec(),
        dkappa: a.dkappa.to_vec(),
    }
}

/// Run both paths on `a`; returns (scan outputs, gemm outputs, scan fwd ms,
/// scan bwd ms, gemm fwd ms, gemm bwd ms) — medians of `reps` timed runs.
fn run_both(c: &'static cortiq_embryo::metal::Ctx, a: &Arena, reps: usize) -> (Outs, Outs, [f64; 4]) {
    let d = &a.d;
    let (w, g, sc) = (a.work(), a.grads(), a.scratch());
    let mut t_scan_f = Vec::new();
    let mut t_scan_b = Vec::new();
    let mut t_gemm_f = Vec::new();
    let mut t_gemm_b = Vec::new();
    a.zero_outputs();
    for _ in 0..reps {
        let cmd = Cmd::new(c);
        cmd.hk_forward(d, &w);
        t_scan_f.push(cmd.commit());
        let cmd = Cmd::new(c);
        cmd.hk_backward(d, &w, &g, 0.0);
        t_scan_b.push(cmd.commit());
    }
    let scan = read(a);
    a.zero_outputs();
    for _ in 0..reps {
        let cmd = Cmd::new(c);
        cmd.hk_forward_gemm(d, &w, &sc);
        t_gemm_f.push(cmd.commit());
        let cmd = Cmd::new(c);
        cmd.hk_backward_gemm(d, &w, &g, &sc, 0.0);
        t_gemm_b.push(cmd.commit());
    }
    let gemm = read(a);
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.total_cmp(y));
        v[v.len() / 2]
    };
    (
        scan,
        gemm,
        [
            med(&mut t_scan_f),
            med(&mut t_scan_b),
            med(&mut t_gemm_f),
            med(&mut t_gemm_b),
        ],
    )
}

#[test]
fn gemm_form_matches_oracle_and_scan_small() {
    let c = ctx().expect("native Vulkan adapter is required");
    // two chunks, two heads, 48 value channels (two value blocks)
    let d = HkDims {
        b: 2,
        t: 128,
        nh: 2,
        nph: 5,
        dv: 48,
    };
    let a = Arena::new(c, d, 300);
    let dec = as_f64(&a.decay);
    let (td, kd, vd, kapd, dod) = (
        as_f64(&a.h_thq),
        as_f64(&a.h_thk),
        as_f64(&a.h_v),
        as_f64(&a.h_kappa),
        as_f64(&a.h_dout),
    );
    let want_o = hk_ref_fwd(&d, &td, &kd, &vd, &kapd, &dec);
    let (want_q, want_k, want_v, want_kap) = hk_ref_bwd(&d, &td, &kd, &vd, &kapd, &dec, &dod);
    let (scan, gemm, _) = run_both(c, &a, 1);
    for (name, got, want) in [
        ("forward", &gemm.out, &want_o),
        ("dtheta-q", &gemm.dthq, &want_q),
        ("dtheta-k", &gemm.dthk, &want_k),
        ("dv", &gemm.dv, &want_v),
        ("dkappa", &gemm.dkappa, &want_kap),
    ] {
        let e = rel_err64(got, want);
        let e_scan = rel_err64(
            match name {
                "forward" => &scan.out,
                "dtheta-q" => &scan.dthq,
                "dtheta-k" => &scan.dthk,
                "dv" => &scan.dv,
                _ => &scan.dkappa,
            },
            want,
        );
        eprintln!("small {name:<9} gemm vs f64 {e:.2e}   (scan vs f64 {e_scan:.2e})");
        assert!(e < 5e-4, "gemm-form {name} vs f64 oracle: {e:e}");
    }
    for (name, g, s) in [
        ("states", &gemm.states, &scan.states),
        ("dstates", &gemm.dstates, &scan.dstates),
        ("dv", &gemm.dv, &scan.dv),
        ("dtheta-q", &gemm.dthq, &scan.dthq),
        ("dtheta-k", &gemm.dthk, &scan.dthk),
        ("dkappa", &gemm.dkappa, &scan.dkappa),
        ("forward", &gemm.out, &scan.out),
    ] {
        let e = rel_err32(g, s);
        eprintln!("small {name:<9} gemm vs scan max|Δ|/max|scan| = {e:.2e}");
        assert!(e < 1e-4, "gemm-form {name} vs scan: {e:e}");
    }
}

#[test]
fn gemm_form_matches_scan_on_b8_t512_and_is_faster() {
    let c = ctx().expect("native Vulkan adapter is required");
    let d = HkDims {
        b: 8,
        t: 512,
        nh: 8,
        nph: 32,
        dv: 128,
    };
    let a = Arena::new(c, d, 400);
    let (scan, gemm, ms) = run_both(c, &a, 3);
    let mut worst = 0.0f64;
    for (name, g, s) in [
        ("states", &gemm.states, &scan.states),
        ("dstates", &gemm.dstates, &scan.dstates),
        ("dv", &gemm.dv, &scan.dv),
        ("dtheta-q", &gemm.dthq, &scan.dthq),
        ("dtheta-k", &gemm.dthk, &scan.dthk),
        ("dkappa", &gemm.dkappa, &scan.dkappa),
        ("forward", &gemm.out, &scan.out),
    ] {
        let e = rel_err32(g, s);
        worst = worst.max(e);
        eprintln!("B8/T512 {name:<9} gemm vs scan max|Δ|/max|scan| = {e:.2e}");
        assert!(e < 1e-4, "gemm-form {name} vs scan: {e:e}");
    }
    eprintln!(
        "B8/T512 nh8 nph32 dv128: scan fwd {:.1} ms, scan bwd {:.1} ms; gemm fwd {:.1} ms, gemm bwd {:.1} ms; bwd speedup {:.1}x; worst rel {worst:.2e}",
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        ms[1] / ms[3].max(1e-9)
    );
    assert!(ms[3] < ms[1], "GEMM-form backward must be faster than the scan");
}
