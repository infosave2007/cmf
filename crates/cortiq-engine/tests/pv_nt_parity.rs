//! Is PV-as-NT the same product as PV-as-NN, and which one is right?
//!
//! A render cannot answer this. The A/B on the real pipeline showed the patched
//! arm differing from the unpatched one by relative L2 0.070 -- and cortiq's own
//! gpu lane differs from its own cpu lane, with no patch at all, by 0.086. A
//! diffusion latent amplifies any perturbation, so a deviation the size of the
//! engine's existing spread proves nothing in either direction.
//!
//! So: fixed inputs, both arms, and an f64 reference computed here. The f64 arm
//! is what makes this decisive -- without it the test can only say the two
//! disagree, not which one is wrong.
//!
//!     cargo test --release -p cortiq-engine --test pv_nt_parity -- --nocapture
//!     set CMF_GPU=wgpu & set CMF_GPU_ADAPTER=1 & cargo test --release ...
//!
//! Run it BOTH ways. Without CMF_GPU the host kernels are exercised; with it the
//! device arms are. The bug hunted here was hypothesised to live in one and not
//! the other, and a single run cannot tell those apart.

use cortiq_engine::fcd_ops::{gemm_dx, gemm_nt};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
}

/// O = P·V in f64, row-major, the answer both arms are trying to produce.
fn reference(p: &[f32], v: &[f32], n: usize, m: usize, dh: usize) -> Vec<f64> {
    let mut o = vec![0f64; n * dh];
    for i in 0..n {
        for j in 0..m {
            let pij = p[i * m + j] as f64;
            if pij == 0.0 {
                continue;
            }
            for d in 0..dh {
                o[i * dh + d] += pij * v[j * dh + d] as f64;
            }
        }
    }
    o
}

fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want) {
        let d = *g as f64 - *w;
        num += d * d;
        den += w * w;
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
fn pv_nt_matches_pv_nn_and_the_f64_reference() {
    // The shapes a real 512x512x49 LTX render actually asks for. `m` is the
    // token count of the thing being attended to, and 49 is the audio one --
    // the case where the two arms take different branches, because 49 % 4 != 0
    // fails the cooperative filter on both but the fallbacks differ.
    let cases: [(usize, usize, usize, &str); 4] = [
        (1792, 1792, 128, "self-attn video"),
        (1792, 1024, 128, "cross-attn to context"),
        (1792, 49, 128, "video attending audio (m=49, not %4)"),
        (49, 1792, 128, "audio attending video (n=49)"),
    ];

    let gpu = std::env::var("CMF_GPU").is_ok() && cortiq_engine::gpu::enabled_here();
    eprintln!("device lane: {}", if gpu { "ON (CMF_GPU set, backend up)" } else { "off -- host kernels only" });
    eprintln!();
    eprintln!("{:<38} {:>12} {:>12} {:>12}", "case", "NN vs f64", "NT vs f64", "NN vs NT");

    let mut bad = Vec::new();
    for (n, m, dh, name) in cases {
        let mut seed = 0xdead_beef_1234_5678u64;
        // P is a softmax output in the real thing: non-negative, rows sum to 1.
        // Feeding signed noise would hide a sign or transpose error behind
        // cancellation, so build a real row-stochastic matrix.
        let mut p = vec![0f32; n * m];
        for row in p.chunks_exact_mut(m) {
            let mut s = 0f32;
            for x in row.iter_mut() {
                *x = lcg(&mut seed) + 0.5 + 1e-3;
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        let v: Vec<f32> = (0..m * dh).map(|_| lcg(&mut seed)).collect();

        // Arm A: what the engine does today.
        let mut o_nn = vec![0f32; n * dh];
        gemm_dx(&p, &v, &mut o_nn, n, dh, m, None);

        // Arm B: v gathered as [dh, tokens], PV as an NT product.
        let mut vt = vec![0f32; dh * m];
        for j in 0..m {
            for d in 0..dh {
                vt[d * m + j] = v[j * dh + d];
            }
        }
        let mut o_nt = vec![0f32; n * dh];
        gemm_nt(&p, &vt, &mut o_nt, n, m, dh, None);

        let want = reference(&p, &v, n, m, dh);
        let e_nn = rel_l2(&o_nn, &want);
        let e_nt = rel_l2(&o_nt, &want);
        let want_nn: Vec<f64> = o_nn.iter().map(|&x| x as f64).collect();
        let e_ab = rel_l2(&o_nt, &want_nn);

        eprintln!("{name:<38} {e_nn:12.3e} {e_nt:12.3e} {e_ab:12.3e}");

        // WHAT IS ASSERTED, and why not an absolute tolerance.
        //
        // The first version asserted both arms within 1e-4 of f64 and failed --
        // on the EXISTING arm, in three of four shapes. Measured on the device
        // lane: NN sits at 2.6e-4 to 2.9e-4 while NT reaches 1.5e-7 to 2.1e-7,
        // the f32 floor. So an absolute bar either fails code this test was not
        // written to judge, or has to be loosened until it stops meaning
        // anything.
        //
        // The invariant that matters for the change under test is comparative:
        // the NT arm must be NO WORSE than the arm it would replace. A little
        // slack (2x) keeps a shape where both are already at the f32 floor from
        // failing on which one happens to round better.
        if e_nt > e_nn.max(1e-7) * 2.0 {
            bad.push(format!(
                "{name}: NT is {e_nt:.3e} off f64 against NN's {e_nn:.3e} -- the \
                 replacement is LESS accurate than what it replaces"
            ));
        }
        // Not a failure, but the reason this file exists twice over: report it.
        if e_nn > e_nt * 10.0 {
            eprintln!(
                "  note: {name}: the existing NN arm is {:.0}x further from f64 than NT",
                e_nn / e_nt.max(f64::MIN_POSITIVE)
            );
        }
    }

    eprintln!();
    eprintln!("NOT covered: the device lane unless CMF_GPU was set for this run,");
    eprintln!("and any shape not in the list above. A pass here is about PV only --");
    eprintln!("it says nothing about the gather that feeds it in ltxdit.rs.");

    assert!(bad.is_empty(), "\n{}", bad.join("\n"));
}
