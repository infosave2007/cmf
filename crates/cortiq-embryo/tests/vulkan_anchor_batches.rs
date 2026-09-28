//! The anchor's batched GEMMs in isolation, at the production shape of the
//! S4 bounded arm (T=1024, 8/2 GQA heads of 128, P rows of ld = 8 + T):
//! batch 0's output must not depend on how many batches share the dispatch
//! (nb = 3 vs 4 sequences → 24 vs 32 (b, kv-group, head) batches), and must
//! match the host.  Written for the Vulkan witness that showed row 0 of the
//! anchor core shifting by 30 % at B=4.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, GemmBatch, Op, ctx};
use cortiq_embryo::ops::lcg_vec;

fn run(nb: usize, t: usize, pv: bool) -> Vec<f32> {
    run_sm(nb, t, pv, false)
}

fn run_sm(nb: usize, t: usize, pv: bool, softmax: bool) -> Vec<f32> {
    let c = ctx().expect("vulkan");
    let (qh, kvh, hd) = (8usize, 2usize, 128usize);
    let group = qh / kvh;
    let (qd, kd) = (qh * hd, kvh * hd);
    let sp = 64usize; // SINK_PAD
    let ld = sp + t;
    let m = nb * t;
    let q = GBuf::from_slice(c, &lcg_vec(1, m * qd));
    let k = GBuf::from_slice(c, &lcg_vec(2, m * kd));
    let p = GBuf::zeros(c, nb * qh * t * ld);
    let o = GBuf::zeros(c, m * qd);
    let bt = |sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]| GemmBatch { nb, nh: kvh, nc: group, sa, sb, sc };
    let sq = [t * qd, group * hd, hd];
    let skv = [t * kd, hd, 0];
    let spb = [qh * t * ld, group * t * ld, t * ld];
    let cmd = Cmd::new(c);
    // S = Q·Kᵀ·scale at column offset sp
    cmd.gemm_ex(Op::N, Op::T, t, t, hd, 0.088, &q, 0, qd, &k, 0, kd, 0.0, &p, sp, ld, &bt(sq, skv, spb), false);
    if softmax {
        // band + sink softmax over the [t, ld] rows of every block (op 19)
        cmd.banded_softmax_carry(&p, 0, t, ld, 4, sp, 128, nb * qh, 0, qh, None);
    }
    if pv {
        // O = P·V with P as the A operand (rows of ld)
        cmd.gemm_ex(Op::N, Op::N, t, hd, ld, 1.0, &p, 0, ld, &k, 0, kd, 0.0, &o, 0, qd, &bt(spb, skv, sq), false);
    }
    cmd.commit();
    if pv { o.to_vec()[..t * qd].to_vec() } else { p.to_vec()[..qh * t * ld].to_vec() }
}

fn worst_rel(a: &[f32], b: &[f32]) -> (f64, Option<usize>) {
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30) as f64;
    let mut w = 0.0f64;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (*x as f64 - *y as f64).abs();
        if d > 0.0 && first.is_none() {
            first = Some(i);
        }
        w = w.max(d);
    }
    (w / scale, first)
}

#[test]
fn anchor_score_gemm_batch0_independent_of_batch_count() {
    for t in [1024usize] {
        let a = run(3, t, false);
        let b = run(4, t, false);
        let (w, first) = worst_rel(&a, &b);
        eprintln!("S GEMM T={t}: batch-0 block nb=3 vs nb=4 rel {w:.2e}, first diff at {first:?}");
        assert!(w <= 1e-6, "S GEMM batch 0 depends on nb: {w:e}");
    }
}

#[test]
fn anchor_pv_gemm_batch0_independent_of_batch_count() {
    for t in [1024usize] {
        let a = run(3, t, true);
        let b = run(4, t, true);
        let (w, first) = worst_rel(&a, &b);
        eprintln!("P·V GEMM T={t}: sequence-0 output nb=3 vs nb=4 rel {w:.2e}, first diff at {first:?}");
        assert!(w <= 1e-6, "P·V GEMM sequence 0 depends on nb: {w:e}");
    }
}

#[test]
fn anchor_softmax_block0_independent_of_block_count() {
    let t = 1024usize;
    let a = run_sm(3, t, false, true);
    let b = run_sm(4, t, false, true);
    let (w, first) = worst_rel(&a, &b);
    eprintln!("band softmax T={t}: block-0 rows 24 vs 32 blocks rel {w:.2e}, first diff at {first:?}");
    assert!(w <= 1e-6, "softmax block 0 depends on the block count: {w:e}");
}

/// RoPE at the production shapes against the host formula, including the
/// shape whose row·t·heads·hd product is exactly 2^32 (B=8, T=1024, 8 heads
/// of 128): the old WGSL guard wrapped to 0 there and skipped the rotation.
#[test]
fn rope_matches_host_at_production_shapes() {
    let c = ctx().expect("vulkan");
    for (rows, t, heads, hd) in [(8 * 1024usize, 1024usize, 8usize, 128usize), (4 * 1024, 1024, 8, 128), (2 * 1024, 1024, 2, 128), (2 * 2048, 2048, 8, 128), (256, 64, 2, 32)] {
        let n = rows * heads * hd;
        let x = lcg_vec(5, n);
        let g = GBuf::from_slice(c, &x);
        let cmd = Cmd::new(c);
        cmd.rope_at(&g, 0, rows, t, heads, hd, 10000.0, false, 0);
        cmd.commit();
        let got = g.to_vec();
        let mut worst = 0.0f32;
        for row in 0..rows {
            let pos = (row % t) as f32;
            for h in 0..heads {
                for j in 0..hd / 2 {
                    let ang = pos * 10000f32.powf(-((2 * j) as f32) / hd as f32);
                    let (s, co) = ang.sin_cos();
                    let ix = row * heads * hd + h * hd + j;
                    let iy = ix + hd / 2;
                    let (a, b) = (x[ix], x[iy]);
                    let (wx, wy) = (a * co - b * s, a * s + b * co);
                    worst = worst.max((got[ix] - wx).abs().max((got[iy] - wy).abs()));
                }
            }
        }
        // f32 sin/cos of angles up to ~2·10³ rad: ≤ 3e-4; a skipped rotation is O(1)
        eprintln!("rope rows={rows} t={t} heads={heads} hd={hd}: max|Δ| vs host {worst:.2e}");
        assert!(worst <= 2e-3, "rope wrong at rows={rows} t={t} heads={heads} hd={hd}: max|Δ| {worst:e}");
    }
}
