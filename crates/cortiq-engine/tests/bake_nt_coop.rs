//! The bake's forward GEMM against a CPU reference — the contract that
//! lets `gemm_nt_f32` swap its scalar arm for the tensor-core one without
//! anyone above noticing. The coop arm works in f16 operands, so it is
//! NOT bit-equal: the assertion is same-arithmetic-within-half-precision,
//! and never a plausible wrong answer (a tile masked wrongly at an edge,
//! a stride crossed at the vocabulary head).
//!
//! Skips itself, loudly, where no GPU comes up; on a device without
//! cooperative matrices it still pins the scalar arm to the reference.

#![cfg(feature = "gpu")]

fn cpu_ref(x: &[f32], w: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0f64;
            for p in 0..k {
                acc += x[i * k + p] as f64 * w[j * k + p] as f64;
            }
            y[i * m + j] = acc as f32;
        }
    }
    y
}

#[test]
fn bake_gemm_matches_cpu_reference_on_tile_edges() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    // (n, k, m): all past the size gate; 63 and 321 are deliberately not
    // multiples of the 64-wide tile, and 66_048 is past the scalar arm's
    // dispatch ceiling — the vocabulary-head case only the coop arm takes.
    let shapes: [(usize, usize, usize); 4] =
        [(64, 512, 320), (63, 512, 321), (256, 768, 1024), (64, 512, 66_048)];
    let mut ran = 0;
    for &(n, k, m) in &shapes {
        let x: Vec<f32> = (0..n * k)
            .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 41) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut y = vec![0f32; n * m];
        if !cortiq_engine::gpu_wgpu::gemm_nt_f32(&x, &w, &mut y, n, k, m) {
            eprintln!("({n},{k},{m}): the GPU arm declined — skipping");
            continue;
        }
        ran += 1;
        let r = cpu_ref(&x, &w, n, k, m);
        let scale = r.iter().fold(0f32, |a, v| a.max(v.abs()));
        let mut worst = 0f32;
        let mut at = 0;
        for (i, (a, b)) in y.iter().zip(&r).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        // Half-precision operands over k terms: the observed error class is
        // ~1e-3 relative to the result scale; 1e-2 is the alarm line that
        // only a real layout bug crosses (a wrong tile is off by O(scale)).
        assert!(
            worst <= 1e-2 * scale.max(1.0),
            "({n},{k},{m}): worst |Δ| {worst} at {at} (y {} vs ref {}, scale {scale})",
            y[at],
            r[at]
        );
        println!("({n},{k},{m}): worst |Δ| {worst:.2e} against scale {scale:.2}");
    }
    if ran == 0 {
        eprintln!("no GPU arm engaged anywhere — skipped");
    }
}
