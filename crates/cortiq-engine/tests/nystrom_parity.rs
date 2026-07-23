//! Nyström kernel golden parity: the f32 streaming kernel vs the
//! validated fp64 matrix math (nystrom_ppl3_06b.py head_out, transposed
//! to streaming semantics).  Fixture: tools/gen_nystrom_golden.py.
//!
//! These tests pin `O1Rect::Aggregate` DELIBERATELY, whatever the
//! runtime default is.  The matrix reference rectifies per-(t, j);
//! `Aggregate` is the streaming operator meant to track it, so it is the
//! one this fixture can hold to account.  `Fm` is a deliberately
//! different (strictly more conservative) operator — on this very
//! fixture 34.6% of the FM entries are negative even though the probe's
//! per-(t, j) clamp never fires, so FM outputs are EXPECTED to differ
//! from the matrix math and are gated on end-to-end ppl instead
//! (`fm_rect_is_nonnegative_far_mass` below covers its contract).

use cortiq_engine::nystrom::{NystromState, O1Rect};

fn rows(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn flat(rows: &[Vec<f32>], range: std::ops::Range<usize>) -> Vec<f32> {
    rows[range].iter().flatten().copied().collect()
}

/// Run the kernel over the golden decode range against the fixture
/// output set named `out_key`, with `sink` permanent exact keys.
fn run_golden(sink: usize, out_key: &str) {
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("data/nystrom_golden.json")).unwrap();
    let g = |k: &str| fx[k].as_u64().unwrap() as usize;
    let (d, dv, m, w, t, p) = (g("d"), g("dv"), g("m"), g("w"), g("t"), g("p"));
    let q = rows(&fx["q"]);
    let k = rows(&fx["k"]);
    let v = rows(&fx["v"]);
    let expect = rows(&fx[out_key]);

    let mut st = NystromState::new(m, w, sink).with_rect(O1Rect::Aggregate);
    st.prefill(&flat(&q, 0..p), &flat(&k, 0..p), &flat(&v, 0..p), p, d, dv);

    let mut out = vec![0.0f32; dv];
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut cnt = 0usize;
    for i in p..t {
        st.step(&q[i], &k[i], &v[i], &mut out);
        for (a, b) in out.iter().zip(&expect[i]) {
            let e = (*a as f64 - *b as f64).abs();
            max_abs = max_abs.max(e);
            sum_abs += e;
            cnt += 1;
        }
    }
    let mean_abs = sum_abs / cnt as f64;
    println!("nystrom parity (sink={sink}): max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
    assert!(max_abs < 1e-3, "max abs error {max_abs:.3e} >= 1e-3");
    assert!(mean_abs < 1e-4, "mean abs error {mean_abs:.3e} >= 1e-4");
}

#[test]
fn nystrom_matches_python_golden() {
    // Default operating mode: sink tokens on (spec §5b).
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("data/nystrom_golden.json")).unwrap();
    let sink = fx["sink"].as_u64().unwrap() as usize;
    assert!(sink > 0, "fixture must carry a nonzero sink count");
    run_golden(sink, "out");
}

#[test]
fn nystrom_sink0_matches_legacy_golden() {
    // Regression: sink = 0 must reproduce the original sink-free
    // kernel against the sink-free golden outputs.
    run_golden(0, "out_sink0");
}

/// Deterministic pseudo-random f32 in [-1, 1) (LCG) — the short test
/// needs no fixture.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // >> 32 keeps 32 bits, so the ratio spans [0, 2) and the result
        // is centred. (A >> 33 here yields [-1, 0): every component
        // negative, hence q·k a large positive sum and near-rank-1 data
        // — which silently defeats any adversarial-conditioning test.)
        ((self.0 >> 32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Plain causal softmax over all positions, f64 — the exact-only
/// reference.
fn softmax_row(q: &[f32], ks: &[Vec<f32>], vs: &[Vec<f32>], upto: usize, dv: usize) -> Vec<f32> {
    let d = q.len();
    let scale = 1.0 / (d as f64).sqrt();
    let lg: Vec<f64> = (0..=upto)
        .map(|j| {
            q.iter()
                .zip(&ks[j])
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>()
                * scale
        })
        .collect();
    let c = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ws: Vec<f64> = lg.iter().map(|x| (x - c).exp()).collect();
    let den: f64 = ws.iter().sum();
    let mut out = vec![0.0f32; dv];
    for (j, wj) in ws.iter().enumerate() {
        for (o, &x) in out.iter_mut().zip(&vs[j]) {
            *o += (wj / den * x as f64) as f32;
        }
    }
    out
}

/// The FM rectifier's contract: every far weight is non-negative, so
/// the joint row is a CONVEX combination of the causal values and each
/// output component must land inside that value range. IID gaussian
/// data is the adversarial regime for any Nyström skeleton (the probe's
/// per-(t,j) clamp fires constantly here) — exactly where the aggregate
/// guard is known to leak negative per-key mass, so it is the regime
/// that separates the two operators.
#[test]
fn fm_rect_is_nonnegative_far_mass() {
    let (d, dv, m, w, sink, t, p) = (
        16usize, 4usize, 32usize, 16usize, 4usize, 320usize, 256usize,
    );
    let mut rng = Lcg(777);
    // Scale 3 puts the landmark logits at the magnitude real post-RoPE
    // heads reach; exp(Q̃K̃ᵀ/√d) is then violently ill-conditioned, M is
    // strongly indefinite, and the skeleton estimates negative weights —
    // the regime the whole rectifier question lives in.
    let q: Vec<Vec<f32>> = (0..t)
        .map(|_| rng.vec(d).iter().map(|x| x * 3.0).collect())
        .collect();
    let k: Vec<Vec<f32>> = (0..t)
        .map(|_| rng.vec(d).iter().map(|x| x * 3.0).collect())
        .collect();
    // Values in [0, 2): a negative output component is then proof of
    // negative weight mass, no bookkeeping needed.
    let v: Vec<Vec<f32>> = (0..t)
        .map(|_| rng.vec(dv).iter().map(|x| x + 1.0).collect())
        .collect();

    let fl = |rows: &[Vec<f32>], hi: usize| -> Vec<f32> {
        rows[..hi].iter().flatten().copied().collect()
    };
    // All weights ≥ 0 ⇒ the row is a CONVEX combination of the causal
    // values ⇒ every output component lands inside the value range.
    // Negative weight mass breaks that bound in EITHER direction (an
    // over-shoot above max(v) is the likelier witness: the near field
    // keeps the sum positive while a negative far key inflates the
    // complementary keys' share), so measure the worst excursion out of
    // [min v, max v] rather than only the sign.
    let (vlo, vhi) = v
        .iter()
        .flatten()
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let run = |rect: O1Rect| -> f32 {
        let mut st = NystromState::new(m, w, sink).with_rect(rect);
        st.prefill(&fl(&q, p), &fl(&k, p), &fl(&v, p), p, d, dv);
        let mut out = vec![0.0f32; dv];
        let mut worst = 0.0f32; // worst distance outside [vlo, vhi]
        for i in p..t {
            st.step(&q[i], &k[i], &v[i], &mut out);
            for &o in out.iter() {
                worst = worst.max(vlo - o).max(o - vhi);
            }
        }
        worst
    };
    let fm = run(O1Rect::Fm);
    let agg = run(O1Rect::Aggregate);
    println!("worst excursion outside v∈[{vlo:.3}, {vhi:.3}]: fm={fm:.4e} aggregate={agg:.4e}");
    // FM: every weight ≥ 0 ⇒ the row cannot leave the value hull.
    assert!(
        fm < 1e-3,
        "FM rect left the value hull by {fm:.3e} — negative far mass"
    );
    // …and the aggregate guard gives no such guarantee: it only inspects
    // the row SUM, so per-key negative mass survives whenever the sum
    // stays positive. If this ever stops holding, the fixture drifted out
    // of the ill-conditioned regime and the assertion above went vacuous —
    // fail loudly rather than keep a green test that proves nothing.
    assert!(
        agg > 1e-2,
        "aggregate guard stayed inside the value hull ({agg:.3e}) — the regime is \
         no longer adversarial, so the FM assertion above is vacuous"
    );
}

/// GQA sharing is an ARITHMETIC NO-OP: a group of Q heads sharing one
/// window/sink/K̃ must produce, bit for bit, what the same heads produce
/// as fully independent single-head states. This is the whole claim of
/// the per-group split — less memory, identical output — so it is tested
/// on `to_bits()`, not on a tolerance.
#[test]
fn group_output_is_bit_identical_to_independent_heads() {
    let (d, dv, m, w, sink, t, p, hpk) = (
        16usize, 8usize, 8usize, 16usize, 4usize, 200usize, 120usize, 4usize,
    );
    let mut rng = Lcg(4242);
    // One k/v stream (the KV head), hpk DIFFERENT query streams — the
    // real GQA shape. Identical queries would hide a head-indexing bug.
    let k: Vec<Vec<f32>> = (0..t)
        .map(|_| rng.vec(d).iter().map(|x| x * 2.0).collect())
        .collect();
    let v: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(dv)).collect();
    let q: Vec<Vec<Vec<f32>>> = (0..hpk)
        .map(|_| {
            (0..t)
                .map(|_| rng.vec(d).iter().map(|x| x * 2.0).collect())
                .collect()
        })
        .collect();

    let fl = |rows: &[Vec<f32>], hi: usize| -> Vec<f32> {
        rows[..hi].iter().flatten().copied().collect()
    };
    let (kp, vp) = (fl(&k, p), fl(&v, p));

    // Reference: hpk independent single-head states (the pre-split shape).
    let mut solo: Vec<NystromState> = (0..hpk)
        .map(|h| {
            let mut st = NystromState::new(m, w, sink).with_rect(O1Rect::Aggregate);
            st.prefill(&fl(&q[h], p), &kp, &vp, p, d, dv);
            st
        })
        .collect();

    // Under test: one shared-window group.
    let qs: Vec<Vec<f32>> = (0..hpk).map(|h| fl(&q[h], p)).collect();
    let qrefs: Vec<&[f32]> = qs.iter().map(|x| x.as_slice()).collect();
    let mut grp = NystromState::new_group(m, w, sink, hpk).with_rect(O1Rect::Aggregate);
    grp.prefill_group(&qrefs, &kp, &vp, p, d, dv);

    let mut got = vec![0.0f32; hpk * dv];
    let mut want = vec![0.0f32; dv];
    for i in p..t {
        let q_all: Vec<f32> = (0..hpk).flat_map(|h| q[h][i].iter().copied()).collect();
        grp.step_group(&q_all, &k[i], &v[i], &mut got);
        for h in 0..hpk {
            solo[h].step(&q[h][i], &k[i], &v[i], &mut want);
            for c in 0..dv {
                assert_eq!(
                    got[h * dv + c].to_bits(),
                    want[c].to_bits(),
                    "pos {i} head {h} chan {c}: group {} vs solo {} — sharing the \
                     window changed the arithmetic",
                    got[h * dv + c],
                    want[c]
                );
            }
        }
    }
    // …and the sharing actually saved something.
    let solo_bytes: usize = solo.iter().map(|s| s.memory_bytes()).sum();
    println!(
        "group {} B vs {} independent heads {} B (÷{:.2})",
        grp.memory_bytes(),
        hpk,
        solo_bytes,
        solo_bytes as f64 / grp.memory_bytes() as f64
    );
    assert!(
        grp.memory_bytes() < solo_bytes,
        "shared group must be smaller"
    );
}

/// Delayed insertion stays EXACTLY once per position per head after the
/// split. The window is shared, so eviction is one GROUP event per
/// position; the far accumulators are per head, so each head absorbs the
/// evicted key once. The bug this pins: driving `far_insert` from a
/// per-head loop over a per-group eviction (each head would absorb the
/// key hpk times — far_len ×hpk, mass double-counted), or evicting per
/// head (the window would advance hpk positions per token — a hole).
#[test]
fn group_far_insert_runs_once_per_evicted_position_per_head() {
    let (d, dv, m, w, sink, t, p, hpk) = (
        8usize, 4usize, 4usize, 16usize, 4usize, 90usize, 64usize, 3usize,
    );
    let mut rng = Lcg(99);
    let k: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(d)).collect();
    let v: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(dv)).collect();
    let q: Vec<Vec<Vec<f32>>> = (0..hpk)
        .map(|_| (0..t).map(|_| rng.vec(d)).collect())
        .collect();
    let fl = |rows: &[Vec<f32>], hi: usize| -> Vec<f32> {
        rows[..hi].iter().flatten().copied().collect()
    };

    let qs: Vec<Vec<f32>> = (0..hpk).map(|h| fl(&q[h], p)).collect();
    let qrefs: Vec<&[f32]> = qs.iter().map(|x| x.as_slice()).collect();
    let mut grp = NystromState::new_group(m, w, sink, hpk).with_rect(O1Rect::Aggregate);
    grp.prefill_group(&qrefs, &fl(&k, p), &fl(&v, p), p, d, dv);

    // Prefill replays positions sink..p through the window: the first w
    // fill it, every later one evicts exactly one key.
    let expect_prefill = p - sink - w;
    for h in 0..hpk {
        assert_eq!(
            grp.far_len(h),
            expect_prefill,
            "head {h}: prefill absorbed {} keys, expected {expect_prefill} \
             (one per evicted position, not per head)",
            grp.far_len(h)
        );
    }
    // Then one eviction per decode step — the window is already full.
    let mut out = vec![0.0f32; hpk * dv];
    for i in p..t {
        let q_all: Vec<f32> = (0..hpk).flat_map(|h| q[h][i].iter().copied()).collect();
        grp.step_group(&q_all, &k[i], &v[i], &mut out);
        for h in 0..hpk {
            assert_eq!(
                grp.far_len(h),
                expect_prefill + (i - p + 1),
                "head {h} at pos {i}: far field grew by more than one key per step"
            );
        }
    }
    // Every key is accounted for exactly once: far + window + sink = t.
    assert_eq!(grp.far_len(0) + w + sink, t, "no double count, no hole");
}

#[test]
fn nystrom_short_prompt_is_exact_softmax() {
    // t = 20 < w = 32 → exact-only mode: no skeleton, and the window
    // still covers every position, so the kernel must equal plain
    // causal softmax attention (sink setting is irrelevant here — every
    // key is already permanent-exact).
    let (d, dv, m, w, sink, t, p) = (16usize, 8usize, 4usize, 32usize, 4usize, 20usize, 12usize);
    let mut rng = Lcg(20260712);
    let q: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(d)).collect();
    let k: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(d)).collect();
    let v: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(dv)).collect();

    let mut st = NystromState::new(m, w, sink);
    let fl = |rows: &[Vec<f32>], hi: usize| -> Vec<f32> {
        rows[..hi].iter().flatten().copied().collect()
    };
    st.prefill(&fl(&q, p), &fl(&k, p), &fl(&v, p), p, d, dv);

    let mut out = vec![0.0f32; dv];
    let mut max_abs = 0.0f32;
    for i in p..t {
        st.step(&q[i], &k[i], &v[i], &mut out);
        let expect = softmax_row(&q[i], &k, &v, i, dv);
        for (a, b) in out.iter().zip(&expect) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    println!("nystrom exact-only parity: max_abs={max_abs:.3e}");
    assert!(
        max_abs < 1e-5,
        "exact-only max abs error {max_abs:.3e} >= 1e-5"
    );
}
