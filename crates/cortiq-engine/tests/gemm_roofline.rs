//! What the device's f32 coop GEMM actually delivers, in GFLOP/s, on
//! the shapes a DiT asks for.
//!
//! The q4tp arm measures 414 GFLOP/s on a 3090 — about a per cent of
//! the card — and its kernel is line-for-line the same tiling as
//! `gemm_nt_coop`, differing only in an activation scale and in reading
//! the weight as packed f16 pairs. So either that one kernel is wrong
//! or the whole cooperative-matrix path is slow here, and those two
//! have completely different fixes. This measures the known-good arm on
//! the same shape and settles it.
//!
//! Run: `cargo test --release -p cortiq-engine --test gemm_roofline -- --nocapture`

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
}

#[test]
fn f32_coop_gemm_roofline() {
    if !cortiq_engine::gpu::enabled_here() {
        eprintln!("no device backend up — skipping");
        return;
    }
    // The DiT's own shapes: ff_in, ff_out, qkv, out — n = one 689-frame
    // window rounded to the tile.
    let cases: &[(usize, usize, usize)] = &[
        // n(batch), k(cols), m(rows)
        (690, 2048, 16384),
        (690, 8192, 2048),
        (690, 2048, 6144),
        (690, 2048, 2048),
    ];
    let mut seed = 0x5eed_0814u64;
    for &(n, k, m) in cases {
        let x: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 0.05).collect();
        let mut y = vec![0f32; n * m];
        // One untimed call: shader compile and the weight upload belong
        // to warmup, not to the number.
        if !cortiq_engine::gpu::gemm_nt_f32(&x, &w, &mut y, n, k, m) {
            eprintln!("({n}x{k}x{m}) refused");
            continue;
        }
        let reps = 3;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            cortiq_engine::gpu::gemm_nt_f32(&x, &w, &mut y, n, k, m);
        }
        let per = t.elapsed().as_secs_f64() / reps as f64;
        let flops = 2.0 * n as f64 * k as f64 * m as f64;
        eprintln!(
            "n={n:<5} k={k:<6} m={m:<6} {:>8.1} ms  {:>8.0} GFLOP/s",
            per * 1e3,
            flops / per / 1e9
        );
    }
}
