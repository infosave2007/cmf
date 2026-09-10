//! TRAIN/SERVE PARITY: the FCD trainer's matrix forward vs the streaming
//! kernel that actually ships.
//!
//! The polish optimizes LN/FFN tensors against `fcd_ops::nystrom_head_fwd`,
//! but what serves the request is `nystrom::NystromState`. If those two
//! compute different functions, the trainer is polishing a model nobody
//! runs — a train/serve skew that no amount of training fixes and that
//! the gradcheck suite cannot see (a backward can be a perfect derivative
//! of the WRONG forward).
//!
//! This file pins the equivalence. It is the reason the trainer seals
//! landmarks from a prompt PREFIX and rectifies with the AGGREGATE guard:
//! both were chosen to match the kernel below, not for their own sake.
//!
//! Historical skew this test would have caught (fixed together with it):
//!   - landmarks sealed from the FULL window (the runtime only ever sees
//!     the prompt) — `seg_means(q, t, ..)` vs `seg_means(q[..tp], tp, ..)`;
//!   - a per-(t,j) clamp `a.max(0)` (impossible to stream: the weights
//!     are never materialized) vs the aggregate far-denominator guard;
//!   - `m_eff` derived from the window rather than the sealed prompt;
//!   - pre-seal rows run through the skeleton instead of exact attention.

use cortiq_engine::fcd_ops as ops;
use cortiq_engine::nystrom::{NystromState, O1Rect};

/// Deterministic pseudo-random values in [-0.5, 0.5) (same generator as
/// the gradcheck suite).
fn synth(n: usize, salt: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
            let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// Trainer forward (f64 matrix form) vs runtime kernel (f32 streaming),
/// over the SERVED rows. Returns (max_abs_diff, max_abs_out).
fn compare(t: usize, d: usize, dv: usize, p: usize, m: usize, w: usize, sink: usize) -> (f64, f64) {
    let cfg = ops::NysCfg {
        m,
        w,
        sink,
        prefill: Some(p),
    };
    // Logit spread ×2 so the far field carries real (and, for a healthy
    // minority of keys, NEGATIVE) mass — a flat-logit fixture would let
    // a broken rectifier pass.
    let mut q = synth(t * d, 20);
    let mut k = synth(t * d, 21);
    let v = synth(t * dv, 22);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 2.0;
    }

    let mut want = vec![0f64; t * dv];
    ops::nystrom_head_fwd(&q, &k, &v, t, d, dv, &cfg, &mut want);

    // The runtime: seal on the prompt prefix, then stream every later
    // position through step() exactly as `Pipeline::nll_ids_o1` does.
    let f32s = |x: &[f64]| x.iter().map(|&a| a as f32).collect::<Vec<f32>>();
    let (q32, k32, v32) = (f32s(&q), f32s(&k), f32s(&v));
    let mut st = NystromState::new(m, w, sink).with_rect(O1Rect::Aggregate);
    st.prefill(&q32[..p * d], &k32[..p * d], &v32[..p * dv], p, d, dv);

    let mut got = vec![0.0f32; dv];
    let (mut max_diff, mut max_out) = (0.0f64, 0.0f64);
    for i in p..t {
        st.step(
            &q32[i * d..(i + 1) * d],
            &k32[i * d..(i + 1) * d],
            &v32[i * dv..(i + 1) * dv],
            &mut got,
        );
        for c in 0..dv {
            let expect = want[i * dv + c];
            max_diff = max_diff.max((expect - got[c] as f64).abs());
            max_out = max_out.max(expect.abs());
        }
    }
    (max_diff, max_out)
}

/// The trainer's forward IS the runtime's operator, on the rows the
/// runtime actually serves (post-seal).
///
/// Tolerance 2e-5 absolute: the two sides are deliberately different
/// arithmetic — the trainer runs the joint weight matrix in f64 with a
/// per-row shift, the kernel streams f32 accumulators with per-landmark
/// flash rescales and a different joint shift (`c_all`). Agreement is
/// therefore limited by f32 epsilon on values of order 0.1–1, not by the
/// operator. Anything at 1e-2 or above is a semantic mismatch, not
/// numerics — which is the failure this test exists to catch.
#[test]
fn trainer_forward_matches_runtime_kernel() {
    for &(t, p, m, w, sink) in &[
        (160usize, 80usize, 8usize, 32usize, 4usize),
        (160, 80, 16, 32, 0), // sink-free kernel
        (200, 100, 8, 48, 4), // wider window
        (120, 60, 4, 16, 2),  // minimum landmark budget
    ] {
        let (d, dv) = (6usize, 5usize);
        assert!(p > w + sink + 8, "fixture must seal a real skeleton");
        let (diff, out) = compare(t, d, dv, p, m, w, sink);
        println!(
            "t={t} p={p} m={m} w={w} sink={sink}: max|trainer-runtime| = {diff:.3e} \
             (max|out| {out:.3e})"
        );
        assert!(
            diff < 2e-5,
            "train/serve skew: t={t} p={p} m={m} w={w} sink={sink} max diff {diff:.3e}"
        );
    }
}

/// Guard the guard: a fixture where the AGGREGATE rectifier and the old
/// per-(t,j) clamp genuinely disagree, so this suite cannot go green on a
/// trainer that silently reintroduces the clamp.
///
/// The clamp is not a refinement of the guard — it is a different
/// operator (it deletes negative per-key mass the shipped kernel keeps),
/// so on real logits the two forwards must measurably diverge. If this
/// test ever fails, the fixture stopped exercising negative far mass and
/// `trainer_forward_matches_runtime_kernel` above lost its teeth.
#[test]
fn aggregate_guard_differs_from_per_key_clamp() {
    let (t, d, dv, p, m, w, sink) = (160usize, 6usize, 5usize, 80usize, 8usize, 32usize, 4usize);
    let cfg = ops::NysCfg {
        m,
        w,
        sink,
        prefill: Some(p),
    };
    let mut q = synth(t * d, 20);
    let mut k = synth(t * d, 21);
    let v = synth(t * dv, 22);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 2.0;
    }
    let mut out = vec![0f64; t * dv];
    ops::nystrom_head_fwd(&q, &k, &v, t, d, dv, &cfg, &mut out);

    // The runtime kernel agrees with the trainer (checked above); what
    // must NOT agree is the strictly-more-conservative per-key rectifier,
    // whose streaming stand-in is O1Rect::Fm.
    let f32s = |x: &[f64]| x.iter().map(|&a| a as f32).collect::<Vec<f32>>();
    let (q32, k32, v32) = (f32s(&q), f32s(&k), f32s(&v));
    let mut st = NystromState::new(m, w, sink).with_rect(O1Rect::Fm);
    st.prefill(&q32[..p * d], &k32[..p * d], &v32[..p * dv], p, d, dv);
    let mut got = vec![0.0f32; dv];
    let mut max_diff = 0.0f64;
    for i in p..t {
        st.step(
            &q32[i * d..(i + 1) * d],
            &k32[i * d..(i + 1) * d],
            &v32[i * dv..(i + 1) * dv],
            &mut got,
        );
        for c in 0..dv {
            max_diff = max_diff.max((out[i * dv + c] - got[c] as f64).abs());
        }
    }
    println!("aggregate vs per-key(Fm) rectifier: max diff {max_diff:.3e}");
    assert!(
        max_diff > 1e-3,
        "fixture no longer exercises negative far mass ({max_diff:.3e}): the parity \
         test above would pass under a per-key clamp too"
    );
}

/// A window whose PROMPT is too short to seal a skeleton is served by
/// exact attention all the way (`NystromState::exact_only`), so the
/// trainer must fall back too — keyed on the prompt length, not on how
/// long the window happens to run.
#[test]
fn short_prompt_falls_back_to_exact_both_sides() {
    let (t, d, dv, p, m, w, sink) = (160usize, 6usize, 5usize, 20usize, 8usize, 32usize, 4usize);
    assert!(
        p <= w + sink + 8,
        "fixture must be in the degenerate regime"
    );
    let cfg = ops::NysCfg {
        m,
        w,
        sink,
        prefill: Some(p),
    };
    let q = synth(t * d, 30);
    let k = synth(t * d, 31);
    let v = synth(t * dv, 32);

    let mut got = vec![0f64; t * dv];
    ops::nystrom_head_fwd(&q, &k, &v, t, d, dv, &cfg, &mut got);

    // Plain causal softmax attention over the whole window.
    let scale = 1.0 / (d as f64).sqrt();
    for ti in 0..t {
        let mut lg = vec![0f64; ti + 1];
        let mut c = f64::NEG_INFINITY;
        for (j, l) in lg.iter_mut().enumerate() {
            *l = (0..d).map(|x| q[ti * d + x] * k[j * d + x]).sum::<f64>() * scale;
            c = c.max(*l);
        }
        let mut den = 0f64;
        let mut acc = vec![0f64; dv];
        for (j, &l) in lg.iter().enumerate() {
            let e = (l - c).exp();
            den += e;
            for x in 0..dv {
                acc[x] += e * v[j * dv + x];
            }
        }
        for x in 0..dv {
            let want = acc[x] / den;
            assert!(
                (want - got[ti * dv + x]).abs() < 1e-12,
                "row {ti}: short-prompt window must be exact attention"
            );
        }
    }
}
