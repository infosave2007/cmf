//! The device 1D convolution against the loop it replaces.
//!
//! It is an implicit GEMM — the column matrix is never built — so an
//! indexing slip shows up as plausible-looking audio rather than a
//! crash. The first Metal cut of it rendered a song 245% off and the
//! only reason anyone knew was a parity check on the whole render,
//! which says "wrong" without saying where. This says where.

/// `yt[t·oc + o] = Σ_i Σ_j w[(o·ic + i)·k + j] · x[i, t + j·dil - pad]`,
/// zero outside the signal — the contract `gpu::conv1d_gemm` fills.
fn reference(
    x: &[f32],
    w: &[f32],
    ic: usize,
    oc: usize,
    n: usize,
    k: usize,
    pad: usize,
    dil: usize,
    out_n: usize,
) -> Vec<f32> {
    let mut yt = vec![0f32; out_n * oc];
    for t in 0..out_n {
        for o in 0..oc {
            let mut acc = 0f32;
            for i in 0..ic {
                for j in 0..k {
                    let p = (t + j * dil) as isize - pad as isize;
                    if p >= 0 && (p as usize) < n {
                        acc += w[(o * ic + i) * k + j] * x[i * n + p as usize];
                    }
                }
            }
            yt[t * oc + o] = acc;
        }
    }
    yt
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
}

/// Shapes the vocoder actually asks for: k=7 with pad 3, and the
/// dilated k=3 taps of a residual unit. Sized past the kernel's own
/// `out_n·ic·k·oc < 4M` floor so the device arm does not simply refuse.
#[test]
fn device_conv1d_matches_the_host_loop() {
    if !cortiq_engine::gpu::enabled_here() {
        eprintln!("no device backend up — skipping");
        return;
    }
    let cases: &[(usize, usize, usize, usize, usize, usize)] = &[
        // ic,  oc,   n,    k, pad, dil
        (64, 128, 512, 7, 3, 1),
        (128, 128, 512, 3, 1, 1),
        (128, 128, 512, 3, 3, 3),
        (128, 128, 512, 3, 9, 9),
        // A ragged tile on both axes: out_n not a multiple of 32, oc not
        // a multiple of 64 — the epilogue takes its slow path there.
        (96, 96, 501, 7, 3, 1),
    ];
    let mut seed = 0x2026_0814u64;
    let mut failures = Vec::new();
    for &(ic, oc, n, k, pad, dil) in cases {
        let out_n = n + 2 * pad - dil * (k - 1);
        let x: Vec<f32> = (0..ic * n).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..oc * ic * k).map(|_| lcg(&mut seed) * 0.1).collect();
        let want = reference(&x, &w, ic, oc, n, k, pad, dil, out_n);
        let mut got = vec![0f32; out_n * oc];
        if !cortiq_engine::gpu::conv1d_gemm(&x, &w, ic, oc, n, k, pad, dil, out_n, &mut got) {
            eprintln!("({ic}x{oc} k={k} dil={dil}) refused — host arm covers it");
            continue;
        }
        // The device stages through f16, so this is a tolerance check,
        // not an equality one. Relative to the reference's own scale.
        let den = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let err = (want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / want.len() as f32)
            .sqrt();
        let rel = err / den.max(1e-9);
        eprintln!(
            "({ic}x{oc} n={n} k={k} pad={pad} dil={dil}) rel err {:.5}",
            rel
        );
        if rel > 0.01 {
            let (i, _) = want
                .iter()
                .zip(&got)
                .enumerate()
                .max_by(|a, b| {
                    ((a.1.0 - a.1.1).abs())
                        .partial_cmp(&(b.1.0 - b.1.1).abs())
                        .unwrap()
                })
                .map(|(i, v)| (i, v))
                .unwrap();
            failures.push(format!(
                "ic={ic} oc={oc} n={n} k={k} pad={pad} dil={dil}: rel {rel:.4}, \
                 worst at t={} o={} want {} got {}",
                i / oc,
                i % oc,
                want[i],
                got[i]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "device conv1d disagrees:\n{}",
        failures.join("\n")
    );
}
