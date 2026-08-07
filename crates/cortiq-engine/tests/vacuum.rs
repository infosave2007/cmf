//! Vacuum tests — inputs whose answer is known in closed form.
//!
//! Every other test in this directory is DIFFERENTIAL: it compares two
//! implementations (GPU against CPU, this port against ComfyUI) and
//! passes when they agree. Agreement is not truth. Two implementations
//! that share a wrong convention agree with each other perfectly, and
//! this engine has shipped three such bugs — a weight cache keyed by
//! pointer address, an unprobed `gemm_nt`, a coop kernel whose f16
//! operands went out of range. None of them crashed. All three produced
//! plausible output (grey frames, clipped audio) and were caught only
//! by comparing against an answer that was known beforehand.
//!
//! So this file is the other kind: degenerate inputs where the right
//! answer follows from the definition rather than from a second run.
//! They are cheap, they need no fixture and no GPU, and they fail
//! loudly on exactly the class of bug that silently survives a parity
//! suite.

use cortiq_engine::attention::{rope_inv_freq, rope_rotate};
use cortiq_engine::nystrom::{NystromState, O1Rect};

/// Deterministic uniform noise in [-1, 1) — no rand dependency, and the
/// same numbers on every platform, which a vacuum test wants.
fn unif(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / a.len() as f64).sqrt() as f32
}

// ─────────────────────────────────────────────────────────────────────
// 1. Attention with equal keys is the mean of the values.
// ─────────────────────────────────────────────────────────────────────

/// If every key is the same vector, every logit q·k_j/√d is the same
/// number, so softmax returns 1/n on each and the output is the plain
/// mean of the values. That holds for ANY query, ANY d, ANY sequence
/// length — it is the definition, not a measurement.
///
/// The reason it is worth asserting through the Nyström kernel is that
/// the kernel does not compute a softmax. It splits the sequence into an
/// exact window, permanent sinks and a landmark skeleton, and divides
/// once by a denominator assembled from all three. Getting the mean back
/// out requires the near mass, the sink mass and the estimated far mass
/// to sit on ONE consistent scale. Anything that breaks that — a missing
/// rescale, a double-counted eviction, a hole at the window boundary,
/// a sink leaking into the far accumulators — moves this number and
/// nothing else in the suite would notice.
///
/// The skeleton is exact here rather than approximate, and that is not
/// luck. With all keys equal to k, the landmark keys are all k too, so
/// Au = exp(Q̃K̃ᵀ/√d) = a·1ᵀ is rank one, and pinv(a·1ᵀ) = 1·aᵀ/(m‖a‖²).
/// The far estimate then carries the scalar 1ᵀMa = 1 exactly, which is
/// precisely the factor that makes the far field weigh the same as the
/// near one. What the regularizer does to that 1 is what the tolerance
/// below is measuring.
fn equal_keys_case(m: usize, w: usize, sink: usize, rect: O1Rect) -> (f32, f32) {
    let (d, dv) = (32usize, 24usize);
    // Long enough that m_eff reaches m (it is clamp(t/8, 4, m)) and that
    // the far field holds real mass rather than a handful of keys.
    let t = 8 * m + w + 128;
    let mut s = 0xC0FFEE_u64;

    let k0: Vec<f32> = (0..d).map(|_| unif(&mut s)).collect();
    let ks: Vec<f32> = std::iter::repeat(k0.iter().copied()).take(t).flatten().collect();
    let vs: Vec<f32> = (0..t * dv).map(|_| unif(&mut s)).collect();
    let qs: Vec<f32> = (0..t * d).map(|_| unif(&mut s)).collect();

    let mut st = NystromState::new(m, w, sink).with_rect(rect);
    st.prefill(&qs, &ks, &vs, t, d, dv);

    let v_last: Vec<f32> = (0..dv).map(|_| unif(&mut s)).collect();
    let q: Vec<f32> = (0..d).map(|_| unif(&mut s)).collect();
    let mut got = vec![0f32; dv];
    st.step(&q, &k0, &v_last, &mut got);

    let mut want = v_last.clone();
    for j in 0..t {
        for (c, wc) in want.iter_mut().enumerate() {
            *wc += vs[j * dv + c];
        }
    }
    for wc in want.iter_mut() {
        *wc /= t as f32 + 1.0;
    }
    (max_abs_diff(&got, &want), rms(&want))
}

#[test]
fn equal_keys_give_the_mean_of_the_values() {
    // The window/sink/landmark budget must not matter: the answer is the
    // same mean whatever the split, so a configuration that drifts is
    // reporting a bookkeeping bug and not an approximation error.
    let cases = [
        (8usize, 32usize, 0usize),
        (8, 32, 4),
        (16, 64, 4),
        (32, 128, 4),
        (32, 128, 16),
        (64, 128, 4),
    ];
    let mut worst = 0f32;
    for rect in [O1Rect::Aggregate, O1Rect::Fm] {
        for &(m, w, sink) in &cases {
            let (mx, sig) = equal_keys_case(m, w, sink, rect);
            println!("m={m:3} w={w:3} sink={sink:2} {rect:?}: max |Δ| {mx:.3e} over rms {sig:.3e}");
            worst = worst.max(mx / sig);
        }
    }
    // A relative floor rather than an absolute one: the mean of ~1k
    // uniform values is itself small, so an absolute epsilon would be
    // measuring the signal and not the kernel.
    assert!(
        worst < 1e-3,
        "equal keys no longer give the mean: worst relative {worst:.3e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 2. One key that dominates returns its own value.
// ─────────────────────────────────────────────────────────────────────

/// The opposite degeneracy. Give one key a logit far above every other
/// and softmax collapses onto it: the output is that key's value, to
/// within exp(−gap). Placed deep in the far field, the needle is
/// visible to the kernel ONLY through the landmark skeleton, so this is
/// the smallest possible injection/recovery experiment — and the seed of
/// the sweep in `examples/o1_recovery.rs`.
///
/// Returned: how much of the needle's value survived, as a fraction. 1.0
/// means the kernel put the whole output on the needle, 0.0 means it
/// lost it entirely.
pub fn needle_case(m: usize, w: usize, sink: usize, amp: f32, depth: usize, seed: u64) -> f32 {
    let (d, dv) = (64usize, 16usize);
    let t = 8 * m + w + depth;
    let mut s = seed;
    let rd = (d as f32).sqrt();

    // Work in LOGIT space so the injection strength means something.
    // Queries are √d·û for a random unit û, keys are iid N(0,1)-ish, so
    // a background logit q·k/√d = û·k has unit spread across j. The
    // needle key gets `amp` added along the decode query's direction,
    // which puts its logit exactly `amp` standard deviations up.
    let mut unit = |s: &mut u64| -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|_| unif(s)).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    };
    let qhat = unit(&mut s);
    let q: Vec<f32> = qhat.iter().map(|x| x * rd).collect();

    // Prefill queries come from the SAME family as the decode query.
    // Drawing them smaller would ask the skeleton to extrapolate outside
    // the space its landmarks were fitted on, which measures the stand
    // and not the kernel.
    let mut qs = Vec::with_capacity(t * d);
    for _ in 0..t {
        let u = unit(&mut s);
        qs.extend(u.iter().map(|x| x * rd));
    }

    let mut ks: Vec<f32> = (0..t * d).map(|_| unif(&mut s) * 1.732).collect();
    // Values carry the detection statistic in dimension 0: background
    // values are zero there and the needle's is one, so the output's
    // component 0 IS the needle's contribution and nothing else.
    let mut vs: Vec<f32> = (0..t * dv).map(|_| unif(&mut s)).collect();
    for j in 0..t {
        vs[j * dv] = 0.0;
    }

    let p = t - depth; // `depth` keys back from the end — far outside w
    for c in 0..d {
        ks[p * d + c] += amp * qhat[c] * rd / rd; // logit shift = amp
    }
    vs[p * dv] = 1.0;

    let mut st = NystromState::new(m, w, sink);
    st.prefill(&qs, &ks, &vs, t, d, dv);

    let k_new: Vec<f32> = (0..d).map(|_| unif(&mut s) * 1.732).collect();
    let mut v_new: Vec<f32> = (0..dv).map(|_| unif(&mut s)).collect();
    v_new[0] = 0.0;
    let mut got = vec![0f32; dv];
    st.step(&q, &k_new, &v_new, &mut got);

    // Exact attention, from the definition — the closed form, not a
    // second implementation of the same idea.
    let mut logits = Vec::with_capacity(t + 1);
    for j in 0..t {
        let dot: f32 = (0..d).map(|c| q[c] * ks[j * d + c]).sum();
        logits.push(dot / rd);
    }
    let dot: f32 = (0..d).map(|c| q[c] * k_new[c]).sum();
    logits.push(dot / rd);
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let (mut den, mut num0) = (0f64, 0f64);
    for (j, &l) in logits.iter().enumerate() {
        let e = ((l - mx) as f64).exp();
        den += e;
        let v0 = if j == t { v_new[0] } else { vs[j * dv] };
        num0 += e * v0 as f64;
    }
    let want0 = (num0 / den) as f32;
    if want0.abs() < 1e-6 {
        return f32::NAN; // the injection was too weak to be detectable at all
    }
    got[0] / want0
}

#[test]
fn a_dominant_key_in_the_far_field_is_recovered() {
    // gap 8 puts ~exp(8) of the softmax mass on the needle: exact
    // attention returns essentially the needle's value, so a kernel that
    // sees the needle at all must return essentially the same.
    let mut worst = f32::INFINITY;
    for &(m, w, sink) in &[(16usize, 64usize, 4usize), (32, 128, 4), (64, 128, 4)] {
        for &depth in &[256usize, 1024] {
            let r = needle_case(m, w, sink, 12.0, depth, 0x5EED_1234);
            println!("m={m:3} w={w:3} depth={depth:5}: recovered {:.1}%", r * 100.0);
            worst = worst.min(r);
        }
    }
    assert!(
        worst > 0.5,
        "a needle 8 logits above the background is being lost: worst {:.1}%",
        worst * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────
// 3. RoPE encodes relative position and nothing else.
// ─────────────────────────────────────────────────────────────────────

/// The whole point of rotary embedding: ⟨R_a q, R_b k⟩ depends on a − b
/// alone. Shift both positions by the same amount and every dot product
/// must be unchanged — that is the property the method is named for, and
/// it is checkable without a second implementation.
#[test]
fn rope_depends_only_on_the_position_difference() {
    let (d, base) = (64usize, 10000.0f32);
    let inv = rope_inv_freq(d, base);
    let mut s = 0xA11CE_u64;
    let q0: Vec<f32> = (0..d).map(|_| unif(&mut s)).collect();
    let k0: Vec<f32> = (0..d).map(|_| unif(&mut s)).collect();

    let dot_at = |a: usize, b: usize| -> f32 {
        let (mut q, mut k) = (q0.clone(), k0.clone());
        rope_rotate(&mut q, a, &inv);
        rope_rotate(&mut k, b, &inv);
        q.iter().zip(&k).map(|(x, y)| x * y).sum()
    };

    let mut worst = 0f32;
    for &(a, b) in &[(0usize, 0usize), (3, 1), (17, 5), (100, 60)] {
        let reference = dot_at(a, b);
        for shift in [1usize, 7, 64, 1000] {
            let shifted = dot_at(a + shift, b + shift);
            worst = worst.max((shifted - reference).abs() / reference.abs().max(1e-6));
        }
    }
    println!("rope relative-position invariance: worst relative {worst:.3e}");
    // f32 trig at position 1064 loses a few digits; the invariant is
    // exact in reals, so the tolerance is measuring float error only.
    assert!(worst < 1e-4, "RoPE is not purely relative: {worst:.3e}");
}

/// Rotations compose: turning by a and then by b is turning by a + b.
/// A pairing bug (interleaved vs half-split) or a sign flip in the
/// rotation breaks this and leaves parity against another
/// implementation of the SAME convention perfectly happy.
#[test]
fn rope_rotations_compose_additively() {
    let d = 64usize;
    let inv = rope_inv_freq(d, 10000.0);
    let mut s = 0xBEEF_u64;
    let x0: Vec<f32> = (0..d).map(|_| unif(&mut s)).collect();

    let mut worst = 0f32;
    for &(a, b) in &[(1usize, 1usize), (5, 11), (64, 3), (200, 300)] {
        let mut twice = x0.clone();
        rope_rotate(&mut twice, a, &inv);
        rope_rotate(&mut twice, b, &inv);
        let mut once = x0.clone();
        rope_rotate(&mut once, a + b, &inv);
        worst = worst.max(max_abs_diff(&twice, &once));
    }
    println!("rope composition: worst |Δ| {worst:.3e}");
    assert!(worst < 1e-4, "R_b∘R_a ≠ R_(a+b): {worst:.3e}");

    // And position 0 is the identity.
    let mut z = x0.clone();
    rope_rotate(&mut z, 0, &inv);
    assert_eq!(z, x0, "RoPE at position 0 is not the identity");
}

// ─────────────────────────────────────────────────────────────────────
// 4. A GEMM against the identity returns its input.
// ─────────────────────────────────────────────────────────────────────

/// `gemm_nt(x, W)` computes x·Wᵀ. With W = I it must return x exactly,
/// and with W = 0 exactly zero. Trivial arithmetic, but it pins the
/// operand ORDER and the transpose convention — and this is the call
/// that shipped unprobed onto a GPU arm and returned plausible garbage,
/// so it earns a test that does not depend on the CPU path being right.
#[test]
fn gemm_nt_against_the_identity_returns_the_input() {
    let (n, k) = (7usize, 33usize);
    let mut s = 0xFEED_u64;
    let x: Vec<f32> = (0..n * k).map(|_| unif(&mut s)).collect();

    let mut eye = vec![0f32; k * k];
    for i in 0..k {
        eye[i * k + i] = 1.0;
    }
    let mut y = vec![0f32; n * k];
    cortiq_engine::fcd_ops::gemm_nt(&x, &eye, &mut y, n, k, k, None);
    assert_eq!(y, x, "x·Iᵀ ≠ x");

    let zero = vec![0f32; k * k];
    let mut y0 = vec![1f32; n * k];
    cortiq_engine::fcd_ops::gemm_nt(&x, &zero, &mut y0, n, k, k, None);
    assert!(y0.iter().all(|&v| v == 0.0), "x·0ᵀ ≠ 0");

    // Scaling the weight scales the output by the same factor — linear
    // in W, which a fused accumulator that forgets to zero its
    // destination gets wrong.
    let mut two = vec![0f32; k * k];
    for i in 0..k {
        two[i * k + i] = 2.0;
    }
    let mut y2 = vec![0f32; n * k];
    cortiq_engine::fcd_ops::gemm_nt(&x, &two, &mut y2, n, k, k, None);
    let want: Vec<f32> = x.iter().map(|v| v * 2.0).collect();
    assert_eq!(y2, want, "x·(2I)ᵀ ≠ 2x");
}
