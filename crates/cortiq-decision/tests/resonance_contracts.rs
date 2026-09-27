//! Resonance contracts (spec §3.5, §6.1), ported from the reference crate's
//! `tests/contracts.rs` without its ORT/Metal/Vulkan/packaging parts: the packed
//! scorer equals the reference error bit for bit on non-orthogonal bases of
//! every rank 0..16 and on dimension tails; the decision equals the reference
//! runtime (`Seed::from_errors`, kept below verbatim as the oracle) bit for bit;
//! ties are stable; an empty active set abstains; invalid inputs are refused.
use cortiq_decision::packed::{LANES, Packed};
use cortiq_decision::resonance::{
    self, Decision, ErrStats, TaskView, Topology, decide, errors_reference, reference_error,
};

/// The reference runtime (tools/cortiq-decision/src/lib.rs:40-50, 207-298),
/// verbatim apart from the fields this comparison does not need.
mod reference {
    #[derive(Clone, Debug)]
    pub struct Task {
        pub id: usize,
        pub mean: Vec<f32>,
        pub basis: Vec<Vec<f32>>,
        pub err_mean: f32,
        pub err_std: f32,
    }
    #[derive(Debug)]
    pub struct Score {
        pub task_id: usize,
        pub score: f32,
        pub probability: f32,
        pub reconstruction_error: f32,
    }
    #[derive(Debug)]
    pub struct Prediction {
        pub task_id: usize,
        pub confidence: f32,
        pub raw_confidence: f32,
        pub margin: f32,
        pub is_novel: bool,
        pub novelty_score: f32,
        pub scores: Vec<Score>,
    }
    pub fn from_errors(
        tasks: &[Task],
        errors: &[f32],
        temperature: f32,
        novelty_theta: f32,
    ) -> Prediction {
        if errors.is_empty() {
            return Prediction {
                task_id: 0,
                confidence: 0.,
                raw_confidence: 0.,
                margin: 0.,
                is_novel: true,
                novelty_score: 1.,
                scores: vec![],
            };
        }
        let mut scored: Vec<_> = tasks
            .iter()
            .zip(errors)
            .map(|(t, &e)| {
                (
                    Score {
                        task_id: t.id,
                        score: 1.0 / (1.0 + e),
                        probability: 0.,
                        reconstruction_error: e,
                    },
                    (e - t.err_mean) / t.err_std,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap());
        let raw_confidence = scored[0].0.score;
        let margin = if scored.len() > 1 {
            raw_confidence - scored[1].0.score
        } else {
            raw_confidence
        };
        let mut logits: Vec<_> = scored
            .iter()
            .map(|(s, _)| -s.reconstruction_error / temperature.max(1e-3))
            .collect();
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for p in &mut logits {
            *p = (*p - mx).exp();
            sum += *p;
        }
        let inv = 1.0 / sum;
        for (i, (s, _)) in scored.iter_mut().enumerate() {
            logits[i] *= inv;
            s.probability = logits[i];
        }
        let confidence = logits[0];
        let novelty_score = 0.5 * (1.0 / (1.0 + (-scored[0].1).exp()))
            + 0.25 * (1.0 / (1.0 + margin * 8.0))
            + 0.25 * (1.0 - confidence);
        Prediction {
            task_id: scored[0].0.task_id,
            confidence,
            raw_confidence,
            margin,
            is_novel: novelty_score > novelty_theta,
            novelty_score,
            scores: scored.into_iter().map(|s| s.0).collect(),
        }
    }
    /// Source algorithm is sequential residual projection, NOT ||x-μ||²-||Bᵀ(x-μ)||².
    pub fn reference_error(x: &[f32], t: &Task) -> f32 {
        let mut r: Vec<f32> = x.iter().zip(&t.mean).map(|(a, b)| a - b).collect();
        for b in &t.basis {
            let mut c = 0.;
            #[allow(clippy::needless_range_loop)]
            for i in 0..r.len() {
                c += r[i] * b[i];
            }
            #[allow(clippy::needless_range_loop)]
            for i in 0..r.len() {
                r[i] -= c * b[i];
            }
        }
        let mut e = 0.;
        for v in r {
            e += v * v;
        }
        e
    }
}

/// xorshift64*: deterministic test data without a rand dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Deliberately NOT orthonormal bases, ranks cycling 0..=16.
fn topologies(rng: &mut Rng, n: usize, d: usize, rank_offset: usize) -> Vec<Topology> {
    (0..n)
        .map(|i| {
            let rank = (i + rank_offset) % 17;
            Topology {
                mean: (0..d).map(|_| rng.f() * 0.2).collect(),
                basis: (0..rank * d).map(|_| rng.f() * 0.3).collect(),
            }
        })
        .collect()
}

fn views(ts: &[Topology]) -> Vec<TaskView<'_>> {
    ts.iter().map(|t| t.view()).collect()
}

fn ref_tasks(ts: &[Topology], stats: &[ErrStats]) -> Vec<reference::Task> {
    ts.iter()
        .zip(stats)
        .enumerate()
        .map(|(i, (t, s))| reference::Task {
            id: i,
            mean: t.mean.clone(),
            basis: t.basis.chunks(t.dim()).map(|r| r.to_vec()).collect(),
            err_mean: s.err_mean,
            err_std: s.err_std,
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn packed_exact_nonorthogonal_variable_rank_and_tail() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for d in [1usize, 3, 7, 17, 37, 384, 4480] {
        for n in [1usize, 5, LANES, LANES + 3, 2 * LANES + 1, 17] {
            let ts = topologies(&mut rng, n, d, n);
            let packed = Packed::new(&views(&ts)).unwrap();
            assert_eq!(packed.tasks(), n);
            let rows = if d > 1000 { 4 } else { 20 };
            let mut scratch = packed.scratch();
            for _ in 0..rows {
                let x: Vec<f32> = (0..d).map(|_| rng.f()).collect();
                let expected: Vec<f32> = ts
                    .iter()
                    .zip(ref_tasks(
                        &ts,
                        &vec![
                            ErrStats {
                                err_mean: 0.1,
                                err_std: 0.02
                            };
                            n
                        ],
                    ))
                    .map(|(t, rt)| {
                        let e = reference::reference_error(&x, &rt);
                        assert_eq!(
                            e.to_bits(),
                            reference_error(&x, &t.mean, &t.basis).to_bits()
                        );
                        e
                    })
                    .collect();
                let mut out = vec![0.0; n];
                packed.errors_with(&x, &mut out, &mut scratch).unwrap();
                assert_eq!(bits(&out), bits(&expected), "d={d} n={n}");
                assert_eq!(
                    bits(&errors_reference(&x, &views(&ts)).unwrap()),
                    bits(&expected)
                );
            }
        }
    }
}

#[test]
fn every_rank_from_0_to_16_in_one_block() {
    let mut rng = Rng(7);
    let d = 29;
    let ts: Vec<Topology> = (0..=16)
        .map(|rank| Topology {
            mean: (0..d).map(|_| rng.f()).collect(),
            basis: (0..rank * d).map(|_| rng.f()).collect(),
        })
        .collect();
    let packed = Packed::new(&views(&ts)).unwrap();
    assert_eq!(packed.ranks(), (0..=16).collect::<Vec<_>>().as_slice());
    for _ in 0..50 {
        let x: Vec<f32> = (0..d).map(|_| rng.f()).collect();
        let want: Vec<f32> = ts
            .iter()
            .map(|t| reference_error(&x, &t.mean, &t.basis))
            .collect();
        assert_eq!(bits(&packed.errors(&x).unwrap()), bits(&want));
    }
}

#[test]
fn nonorthogonal_basis_is_not_the_energy_shortcut() {
    // Two equal basis rows: the sequential projection removes the direction
    // twice (second coefficient 0), the shortcut ‖r‖² − Σc² would subtract twice.
    let mean = [0.0f32; 3];
    let basis = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let x = [2.0f32, 1.0, 0.0];
    let e = reference_error(&x, &mean, &basis);
    assert_eq!(e, 1.0);
    let packed = Packed::new(&[TaskView::new(&mean, &basis).unwrap()]).unwrap();
    assert_eq!(packed.errors(&x).unwrap(), vec![1.0]);
}

#[test]
fn decision_is_bit_identical_to_the_reference_runtime() {
    let mut rng = Rng(0xdead_beef);
    for n in [1usize, 2, 3, 9, 77, 150] {
        for _ in 0..40 {
            let errors: Vec<f32> = (0..n).map(|_| rng.f().abs() * 2.0).collect();
            let stats: Vec<ErrStats> = (0..n)
                .map(|_| ErrStats {
                    err_mean: rng.f().abs(),
                    err_std: rng.f().abs() * 0.1 + 1e-4,
                })
                .collect();
            let temperature = [0.0001f32, 0.02, 0.0245571, 0.5, 1.0][(rng.next() % 5) as usize];
            let theta = rng.f().abs();
            let ts: Vec<Topology> = (0..n)
                .map(|_| Topology {
                    mean: vec![0.0],
                    basis: vec![],
                })
                .collect();
            let want = reference::from_errors(&ref_tasks(&ts, &stats), &errors, temperature, theta);
            let got = decide(&errors, &stats, temperature).unwrap();
            assert_eq!(got.winner, Some(want.task_id));
            assert_eq!(got.p_top.to_bits(), want.confidence.to_bits());
            assert_eq!(got.raw_confidence.to_bits(), want.raw_confidence.to_bits());
            assert_eq!(got.margin.to_bits(), want.margin.to_bits());
            assert_eq!(got.novelty.to_bits(), want.novelty_score.to_bits());
            assert_eq!(got.is_novel(theta), want.is_novel);
            assert_eq!(got.ranked.len(), want.scores.len());
            for (a, b) in got.ranked.iter().zip(&want.scores) {
                assert_eq!(a.index, b.task_id);
                assert_eq!(a.score.to_bits(), b.score.to_bits());
                assert_eq!(a.probability.to_bits(), b.probability.to_bits());
                assert_eq!(a.error.to_bits(), b.reconstruction_error.to_bits());
            }
        }
    }
}

#[test]
fn empty_active_set_abstains_and_stable_ties() {
    let d = decide(&[], &[], 0.1).unwrap();
    assert_eq!(d, Decision::abstain());
    assert!(d.winner.is_none() && d.is_novel(0.999) && d.is_novel(1.0));
    assert!(!d.accepted(0.0, 1.0));
    // Packed with no tasks yields no errors, and the decision abstains.
    let p = Packed::new(&[]).unwrap();
    assert!(
        decide(&p.errors(&[]).unwrap(), &[], 0.1)
            .unwrap()
            .winner
            .is_none()
    );

    // Identical topologies: the first one wins.
    let mean = vec![0.1f32; 17];
    let basis = vec![0.2f32; 34];
    let ts = [
        Topology {
            mean: mean.clone(),
            basis: basis.clone(),
        },
        Topology { mean, basis },
    ];
    let e = Packed::new(&views(&ts))
        .unwrap()
        .errors(&[0.0; 17])
        .unwrap();
    assert_eq!(e[0].to_bits(), e[1].to_bits());
    let s = [ErrStats {
        err_mean: 0.1,
        err_std: 0.02,
    }; 2];
    let dec = decide(&e, &s, 0.12).unwrap();
    assert_eq!(dec.winner, Some(0));
    assert_eq!(dec.margin, 0.0);
    assert_eq!(dec.ranked[0].probability, dec.ranked[1].probability);

    // Distinct errors that round to one f32 score keep the candidate order
    // (the reference sorts by score, not by error).
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    assert_eq!(resonance::score(a), resonance::score(b));
    let dec = decide(&[b, a], &s, 0.5).unwrap();
    assert_eq!(
        dec.winner,
        Some(0),
        "tie on the score keeps the first candidate"
    );
}

#[test]
fn gate_semantics() {
    let s = [ErrStats {
        err_mean: 0.2,
        err_std: 0.05,
    }; 3];
    let d = decide(&[0.1, 0.4, 0.9], &s, 0.05).unwrap();
    assert_eq!(d.winner, Some(0));
    assert_eq!(d.top_error(), Some(0.1));
    // Boundaries: p_top ≥ τ accepts at equality, novelty ≤ θ accepts at equality.
    assert!(d.accepted(d.p_top, d.novelty));
    assert!(!d.accepted(f32::from_bits(d.p_top.to_bits() + 1), d.novelty));
    assert!(!d.accepted(0.0, f32::from_bits(d.novelty.to_bits() - 1)));
    assert!(d.is_novel(f32::from_bits(d.novelty.to_bits() - 1)));
    let total: f32 = d.ranked.iter().map(|r| r.probability).sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert_eq!(d.probability_of(0), Some(d.p_top));
    assert!(d.ranked.windows(2).all(|w| w[0].score >= w[1].score));
    // T is clamped at 1e-3 (a smaller T gives the same bits).
    let a = decide(&[0.1, 0.1004], &s[..2], 1e-6).unwrap();
    let b = decide(&[0.1, 0.1004], &s[..2], 1e-3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn sign_flip_of_a_basis_row_changes_no_error_bit() {
    let mut rng = Rng(99);
    let d = 64;
    for _ in 0..50 {
        let mean: Vec<f32> = (0..d).map(|_| rng.f()).collect();
        let basis: Vec<f32> = (0..5 * d).map(|_| rng.f()).collect();
        let mut flipped = basis.clone();
        for v in &mut flipped[2 * d..3 * d] {
            *v = -*v;
        }
        let x: Vec<f32> = (0..d).map(|_| rng.f()).collect();
        assert_eq!(
            reference_error(&x, &mean, &basis).to_bits(),
            reference_error(&x, &mean, &flipped).to_bits()
        );
    }
}

#[test]
fn rejects_invalid_shapes_numbers_and_statistics() {
    let t = Topology {
        mean: vec![0.0; 17],
        basis: vec![0.1; 34],
    };
    let p = Packed::new(&[t.view()]).unwrap();
    assert!(p.errors(&[1.0]).is_err());
    assert!(p.errors(&[f32::NAN; 17]).is_err());
    assert!(p.errors(&[f32::INFINITY; 17]).is_err());
    let mut out = vec![0.0; 2];
    assert!(p.errors_into(&[0.0; 17], &mut out).is_err());
    // Overflow to a non-finite error is refused.
    assert!(p.errors(&[3.0e38; 17]).is_err());
    assert!(errors_reference(&[3.0e38; 17], &[t.view()]).is_err());
    // Mixed dimensions, ragged bases and non-finite parameters.
    let u = Topology {
        mean: vec![0.0; 16],
        basis: vec![],
    };
    assert!(Packed::new(&[t.view(), u.view()]).is_err());
    assert!(TaskView::new(&[0.0; 17], &[0.1; 33]).is_err());
    assert!(TaskView::new(&[f32::NAN; 17], &[]).is_err());
    // Error statistics and errors.
    let ok = ErrStats {
        err_mean: 0.1,
        err_std: 0.02,
    };
    assert!(decide(&[0.5], &[ErrStats { err_std: 0.0, ..ok }], 0.1).is_err());
    assert!(
        decide(
            &[0.5],
            &[ErrStats {
                err_mean: -1.0,
                ..ok
            }],
            0.1
        )
        .is_err()
    );
    assert!(
        decide(
            &[0.5],
            &[ErrStats {
                err_mean: f32::NAN,
                ..ok
            }],
            0.1
        )
        .is_err()
    );
    assert!(decide(&[-0.5], &[ok], 0.1).is_err());
    assert!(decide(&[0.5, 0.2], &[ok], 0.1).is_err());
    assert!(decide(&[0.5], &[ok], f32::INFINITY).is_err());
    assert!(decide(&[0.5], &[ok], -1.0).is_err());
}

#[test]
fn packed_is_shareable_across_threads() {
    let mut rng = Rng(5);
    let ts = topologies(&mut rng, 21, 97, 3);
    let packed = Packed::new(&views(&ts)).unwrap();
    let xs: Vec<Vec<f32>> = (0..16)
        .map(|_| (0..97).map(|_| rng.f()).collect())
        .collect();
    let serial: Vec<Vec<f32>> = xs.iter().map(|x| packed.errors(x).unwrap()).collect();
    let parallel: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = xs
            .iter()
            .map(|x| s.spawn(|| packed.errors(x).unwrap()))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (a, b) in serial.iter().zip(&parallel) {
        assert_eq!(bits(a), bits(b));
    }
}
