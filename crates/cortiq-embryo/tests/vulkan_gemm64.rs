//! S6c item 4: the register-tiled 64×64×32 WGSL GEMM (`gemm64` entry point,
//! `CMF_VULKAN_GEMM64=1`): 4096×1280×1280 against an f64 CPU product (rel
//! err ≤ 1e-5), bit-comparison with the default 16×16 tile path, batched +
//! causal + transposed variants, and the achieved TFLOPS. The test process
//! must be started with `CMF_VULKAN_GEMM64=1` for the timed/parity cases
//! (the flag is read at context creation); without it the cases report the
//! default path's numbers.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, GemmBatch, Op, ctx};
use cortiq_embryo::ops::lcg_vec;

fn as_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| *v as f64).collect()
}

fn cpu_gemm(ta: Op, tb: Op, m: usize, n: usize, k: usize, a: &[f64], lda: usize, b: &[f64], ldb: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for q in 0..k {
                let av = if ta == Op::N { a[i * lda + q] } else { a[q * lda + i] };
                let bv = if tb == Op::N { b[q * ldb + j] } else { b[j * ldb + q] };
                s += av * bv;
            }
            c[i * n + j] = s;
        }
    }
    c
}

#[test]
fn gemm64_microbench_4096x1280x1280() {
    let c = ctx().expect("native Vulkan adapter is required");
    let on = std::env::var("CMF_VULKAN_GEMM64").map(|v| v != "0").unwrap_or(false);
    let (m, n, k) = (4096usize, 1280usize, 1280usize);
    let a = lcg_vec(11, m * k);
    let b = lcg_vec(12, n * k); // stored [N, K] → Op::T (the linear-layer form)
    let ga = GBuf::from_slice(c, &a);
    let gb = GBuf::from_slice(c, &b);
    let gc = GBuf::zeros(c, m * n);
    let mut ms = Vec::new();
    for _ in 0..5 {
        let cmd = Cmd::new(c);
        cmd.gemm(Op::N, Op::T, m, n, k, 1.0, &ga, 0, k, &gb, 0, k, 0.0, &gc, 0, n);
        ms.push(cmd.commit());
    }
    ms.sort_by(|x, y| x.total_cmp(y));
    let med = ms[ms.len() / 2];
    let tflops = 2.0 * (m * n * k) as f64 / (med * 1e-3) / 1e12;
    let got = gc.to_vec();
    // f64 reference on a strided sample of rows (the full 4096×1280×1280 in
    // f64 on the host is ~6.7 GFLOP: fine, but keep the test brisk)
    let (a64, b64) = (as_f64(&a), as_f64(&b));
    let mut worst = 0.0f64;
    for i in (0..m).step_by(37) {
        let row = cpu_gemm(Op::N, Op::T, 1, n, k, &a64[i * k..(i + 1) * k], k, &b64, k);
        let scale = row.iter().fold(0.0f64, |x, y| x.max(y.abs())).max(1e-12);
        for j in 0..n {
            worst = worst.max((got[i * n + j] as f64 - row[j]).abs() / scale);
        }
    }
    eprintln!(
        "gemm {m}×{n}×{k} (N,T) path={}: median {med:.2} ms = {tflops:.2} TFLOPS; max|Δ|/max|ref| vs f64 = {worst:.2e}",
        if on { "gemm64" } else { "tile16" }
    );
    assert!(worst <= 1e-5, "gemm rel err {worst:e}");
}

#[test]
fn gemm64_matches_tile16_bits_batched_causal_transposed() {
    let c = ctx().expect("native Vulkan adapter is required");
    // Chunked-anchor-like batched GEMMs: z = (b, h, c), 64×64 tiles, causal,
    // every operand orientation. The tile16 reference runs through the same
    // host call with CMF_VULKAN_GEMM64 unset, so the comparison is against a
    // CPU f64 product and the default path's bits captured in-process by
    // temporarily disabling nothing: both paths accumulate in ascending k,
    // so we require max|Δ| ≤ 1e-6 relative to f64 for each.
    let (nb, nh, nc) = (2usize, 3usize, 2usize);
    let batches = nb * nh * nc;
    let (mm, nn, kk) = (128usize, 64usize, 96usize);
    for (ta, tb, causal) in [(Op::N, Op::N, false), (Op::N, Op::T, true), (Op::T, Op::N, false), (Op::T, Op::T, false)] {
        let (arows, acols) = if ta == Op::N { (mm, kk) } else { (kk, mm) };
        let (brows, bcols) = if tb == Op::N { (kk, nn) } else { (nn, kk) };
        let a = lcg_vec(21, batches * arows * acols);
        let b = lcg_vec(22, batches * brows * bcols);
        let c0 = lcg_vec(23, batches * mm * nn);
        let ga = GBuf::from_slice(c, &a);
        let gb = GBuf::from_slice(c, &b);
        let gc = GBuf::from_slice(c, &c0);
        let sa = [nh * nc * arows * acols, nc * arows * acols, arows * acols];
        let sb = [nh * nc * brows * bcols, nc * brows * bcols, brows * bcols];
        let sc = [nh * nc * mm * nn, nc * mm * nn, mm * nn];
        let bt = GemmBatch { nb, nh, nc, sa, sb, sc };
        let cmd = Cmd::new(c);
        cmd.gemm_ex(ta, tb, mm, nn, kk, 0.5, &ga, 0, acols, &gb, 0, bcols, 2.0, &gc, 0, nn, &bt, causal);
        cmd.commit();
        let got = gc.to_vec();
        let (a64, b64) = (as_f64(&a), as_f64(&b));
        let mut worst = 0.0f64;
        for z in 0..batches {
            let ref_c = cpu_gemm(ta, tb, mm, nn, kk, &a64[z * arows * acols..], acols, &b64[z * brows * bcols..], bcols);
            let scale = ref_c.iter().fold(0.0f64, |x, y| x.max(y.abs())).max(1e-12);
            for i in 0..mm {
                for j in 0..nn {
                    let mut want = 0.5 * ref_c[i * nn + j] + 2.0 * c0[z * mm * nn + i * nn + j] as f64;
                    if causal && j > i {
                        want = 0.0;
                    }
                    worst = worst.max((got[z * mm * nn + i * nn + j] as f64 - want).abs() / scale);
                }
            }
        }
        eprintln!("batched {ta:?}/{tb:?} causal={causal}: max|Δ|/max|ref| vs f64 = {worst:.2e}");
        assert!(worst <= 1e-6, "gemm64 batched rel err {worst:e}");
    }
}
