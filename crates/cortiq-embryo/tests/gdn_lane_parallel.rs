//! Parallel GDN scan parity and repeated-run race witness.
#![cfg(target_os = "macos")]

use cortiq_embryo::metal::{Cmd, GBuf, ctx};
use cortiq_embryo::ops::lcg_vec;

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn gdn_parallel_matches_serial_and_repeats() {
    let Some(c) = ctx() else { return };
    const B: usize = 2;
    const T: usize = 64;
    const D: usize = 64;
    const CD: usize = 192;
    let q: Vec<f32> = lcg_vec(1, B * T * D).into_iter().map(|x| x * 0.2).collect();
    let k: Vec<f32> = lcg_vec(2, B * T * D).into_iter().map(|x| x * 0.2).collect();
    let v: Vec<f32> = lcg_vec(3, B * T * D).into_iter().map(|x| x * 0.2).collect();
    let z: Vec<f32> = lcg_vec(4, B * T * D).into_iter().map(|x| x * 0.2).collect();
    let mut ab = vec![0.0; B * T * D];
    for i in 0..B * T {
        ab[i * D] = i as f32 * 0.01 - 0.3;
        ab[i * D + 1] = i as f32 * 0.007 - 0.2;
    }
    let conv: Vec<f32> = lcg_vec(5, CD * 4).into_iter().map(|x| x * 0.1).collect();
    let norm = vec![1.0; D];
    let mut p = vec![0.0; CD * 4 + D + 2];
    p[..CD * 4].copy_from_slice(&conv);
    p[CD * 4..CD * 4 + D].copy_from_slice(&norm);
    p[CD * 4 + D] = -1.2;
    p[CD * 4 + D + 1] = -0.3;
    let qb = GBuf::from_slice(c, &q);
    let kb = GBuf::from_slice(c, &k);
    let vb = GBuf::from_slice(c, &v);
    let zb = GBuf::from_slice(c, &z);
    let abb = GBuf::from_slice(c, &ab);
    let pb = GBuf::from_slice(c, &p);
    let make = || {
        (
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * (T + 1) * D * D),
        )
    };
    let (q0, k0, v0, b0, r0, i0, o0, s0) = make();
    let forward = |parallel: bool,
                   qx: &GBuf,
                   kx: &GBuf,
                   vx: &GBuf,
                   bx: &GBuf,
                   rx: &GBuf,
                   ix: &GBuf,
                   ox: &GBuf,
                   sx: &GBuf| {
        let cmd = Cmd::new(c);
        if parallel {
            cmd.gdn_forward_parallel(
                &qb,
                &kb,
                &vb,
                &zb,
                &abb,
                &pb,
                0,
                CD * 4,
                CD * 4 + D,
                CD * 4 + D + 1,
                qx,
                kx,
                vx,
                bx,
                rx,
                ix,
                ox,
                sx,
                B,
                T,
                1e-6,
            );
        } else {
            cmd.gdn_forward(
                &qb,
                &kb,
                &vb,
                &zb,
                &abb,
                &pb,
                0,
                CD * 4,
                CD * 4 + D,
                CD * 4 + D + 1,
                qx,
                kx,
                vx,
                bx,
                rx,
                ix,
                ox,
                sx,
                B,
                T,
                1e-6,
            );
        }
        cmd.commit();
    };
    forward(false, &q0, &k0, &v0, &b0, &r0, &i0, &o0, &s0);
    let (q1, k1, v1, b1, r1, i1, o1, s1) = make();
    forward(true, &q1, &k1, &v1, &b1, &r1, &i1, &o1, &s1);
    let (q2, k2, v2, b2, r2, i2, o2, s2) = make();
    forward(true, &q2, &k2, &v2, &b2, &r2, &i2, &o2, &s2);
    assert!(max_err(&q0.to_vec(), &q1.to_vec()) == 0.0);
    assert!(max_err(&k0.to_vec(), &k1.to_vec()) == 0.0);
    assert!(max_err(&v0.to_vec(), &v1.to_vec()) == 0.0);
    assert!(max_err(&b0.to_vec(), &b1.to_vec()) == 0.0);
    assert!(max_err(&r0.to_vec(), &r1.to_vec()) < 1e-5);
    assert!(max_err(&i0.to_vec(), &i1.to_vec()) == 0.0);
    assert!(max_err(&o0.to_vec(), &o1.to_vec()) < 5e-5);
    assert!(max_err(&s0.to_vec(), &s1.to_vec()) < 5e-6);
    assert!(max_err(&r1.to_vec(), &r2.to_vec()) < 5e-6);
    assert!(max_err(&o1.to_vec(), &o2.to_vec()) < 5e-5);
    assert!(max_err(&s1.to_vec(), &s2.to_vec()) < 5e-6);

    let dn: Vec<f32> = lcg_vec(22, B * T * D)
        .into_iter()
        .map(|x| x * 0.3)
        .collect();
    let dnb = GBuf::from_slice(c, &dn);
    let dzb = GBuf::zeros(c, B * T * D);
    let make_bwd = || {
        (
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, B * T * D),
            GBuf::zeros(c, p.len()),
        )
    };
    let (dq0, dk0, dv0, da0, g0) = make_bwd();
    let cmd = Cmd::new(c);
    cmd.gdn_backward(
        &qb,
        &kb,
        &vb,
        &q0,
        &k0,
        &v0,
        &abb,
        &b0,
        &r0,
        &i0,
        &pb,
        &s0,
        &dnb,
        &dzb,
        &dq0,
        &dk0,
        &dv0,
        &da0,
        &pb,
        0,
        CD * 4,
        CD * 4 + D,
        CD * 4 + D + 1,
        &g0,
        0,
        CD * 4,
        CD * 4 + D,
        CD * 4 + D + 1,
        B,
        T,
    );
    cmd.commit();
    let (dq1, dk1, dv1, da1, g1) = make_bwd();
    let cmd = Cmd::new(c);
    cmd.gdn_backward_parallel(
        &qb,
        &kb,
        &vb,
        &q1,
        &k1,
        &v1,
        &abb,
        &b1,
        &r1,
        &i1,
        &pb,
        &s1,
        &dnb,
        &dzb,
        &dq1,
        &dk1,
        &dv1,
        &da1,
        &pb,
        0,
        CD * 4,
        CD * 4 + D,
        CD * 4 + D + 1,
        &g1,
        0,
        CD * 4,
        CD * 4 + D,
        CD * 4 + D + 1,
        B,
        T,
    );
    cmd.commit();
    assert!(max_err(&dq0.to_vec(), &dq1.to_vec()) < 5e-5);
    assert!(max_err(&dk0.to_vec(), &dk1.to_vec()) < 5e-5);
    assert!(max_err(&dv0.to_vec(), &dv1.to_vec()) < 5e-5);
    assert!(max_err(&da0.to_vec(), &da1.to_vec()) < 5e-5);
    assert!(max_err(&g0.to_vec(), &g1.to_vec()) < 5e-5);
}
