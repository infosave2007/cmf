//! `gemm_nt_f32`'s w-side cache against a REUSED buffer.
//!
//! Every batched attention in this engine allocates one `k`/`v` scratch
//! pair per call and refills it per head, so the device-side cache sees
//! the same address holding a different matrix over and over. Keyed on
//! the address alone it returns head 0's keys for every head — silently,
//! and only on the GPU, which is exactly the shape of bug that survives
//! a CPU test suite. Run with the `gpu` feature on a machine that has a
//! device; without one `gemm_nt_f32` declines and the test skips.

#![cfg(feature = "gpu")]

use cortiq_engine::gpu_wgpu::gemm_nt_f32;

/// y = x · wᵀ, row-major, the definition.
fn reference(x: &[f32], w: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0f64;
            for l in 0..k {
                acc += (x[i * k + l] as f64) * (w[j * k + l] as f64);
            }
            y[i * m + j] = acc as f32;
        }
    }
    y
}

fn fill(buf: &mut [f32], seed: u64) {
    let mut s = seed | 1;
    for v in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s >> 40) as f32 / 8_388_608.0) - 1.0;
    }
}

#[test]
fn refilling_the_w_buffer_is_not_a_cache_hit() {
    // Past gemm_nt_f32's own size floor, or it declines and runs on the
    // host — which would pass whatever the cache did.
    let (n, k, m) = (256usize, 128usize, 256usize);
    let mut x = vec![0f32; n * k];
    let mut w = vec![0f32; m * k];
    fill(&mut x, 0x1234_5678);
    fill(&mut w, 0xabcd_ef01);

    let mut y = vec![0f32; n * m];
    if !gemm_nt_f32(&x, &w, &mut y, n, k, m) {
        eprintln!("no GPU (or CMF_BAKE_GPU=0) — skipping");
        return;
    }
    let want = reference(&x, &w, n, k, m);
    // RELATIVE, not absolute: on a card with cooperative matrices this
    // GEMM feeds the units f16 operands (f32 accumulator), so ~1e-3 of
    // the result's own magnitude is the floor, not a defect — an
    // absolute 2e-3 passed on the scalar arm and failed the tensor-core
    // one at 4.4e-3 while being numerically fine. What this test is
    // actually about is the CACHE: a stale weight gives O(1) relative
    // error, which 5e-3 still catches by three orders of magnitude.
    let worst = |a: &[f32], b: &[f32]| {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter()
            .zip(b)
            .fold(0f32, |acc, (&p, &q)| acc.max((p - q).abs()))
            / scale
    };
    let d0 = worst(&y, &want);
    assert!(d0 < 5e-3, "first call already wrong: rel {d0:.3e}");

    // The same allocation, a different matrix — one head later.
    fill(&mut w, 0x5555_aaaa);
    let mut y2 = vec![0f32; n * m];
    assert!(gemm_nt_f32(&x, &w, &mut y2, n, k, m));
    let want2 = reference(&x, &w, n, k, m);
    let d1 = worst(&y2, &want2);
    println!("first {d0:.3e}, after refill {d1:.3e}");
    assert!(
        d1 < 5e-3,
        "stale w served from the cache: rel {d1:.3e} (a hit on the address, not the contents)"
    );

    // And the original contents again: a real hit, still correct.
    fill(&mut w, 0xabcd_ef01);
    let mut y3 = vec![0f32; n * m];
    assert!(gemm_nt_f32(&x, &w, &mut y3, n, k, m));
    let d2 = worst(&y3, &want);
    assert!(d2 < 5e-3, "cache hit returned the wrong matrix: rel {d2:.3e}");
}
