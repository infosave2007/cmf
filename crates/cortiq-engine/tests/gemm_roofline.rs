//! What the device's f32 coop GEMM actually delivers, in GFLOP/s, on
//! the shapes a DiT asks for.
//!
//! This measures the f32 cooperative arm — the known-good one — so a
//! number for any other kernel has something to be compared against.
//!
//! It used to open by quoting 414 GFLOP/s for the q4tp arm on a 3090 and
//! asking whether that kernel or the whole cooperative path was at fault.
//! Do not carry that figure anywhere: it does not reproduce. Measured on
//! a 3090 with this methodology the q4tp arm runs 2842-2926 GFLOP/s —
//! seven times the quoted number, and 1.5x FASTER than its f32 sibling
//! rather than 4.6x behind it, because it reads ~0.52 bytes per weight
//! against f32's four and these shapes are weight-read bound. The f32
//! arm here measures 842-2016 GFLOP/s on an A100 (contended) and
//! 1310-1957 on a 3090.
//!
//! The lesson is the reason this paragraph is long: a benchmark figure
//! with no machine, no shape and no date attached propagates as a fact,
//! and a work plan got built on this one before it was checked.
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
