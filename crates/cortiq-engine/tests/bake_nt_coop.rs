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

fn cpu_ref_nn(dy: &[f32], w: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut dx = vec![0f32; n * k];
    for i in 0..n {
        for kk in 0..k {
            let mut acc = 0f64;
            for j in 0..m {
                acc += dy[i * m + j] as f64 * w[j * k + kk] as f64;
            }
            dx[i * k + kk] = acc as f32;
        }
    }
    dx
}

/// The backward twin reads w down its columns — the staging pattern the
/// forward kernel never exercises, and the one a stride bug would hide in.
#[test]
fn bake_gemm_dx_matches_cpu_reference() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    let shapes: [(usize, usize, usize); 3] = [(64, 320, 512), (63, 321, 512), (256, 1024, 768)];
    let mut ran = 0;
    for &(n, k, m) in &shapes {
        let dy: Vec<f32> = (0..n * m)
            .map(|i| ((i * 31 + 7) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..m * k)
            .map(|i| ((i * 23 + 5) % 89) as f32 / 89.0 - 0.5)
            .collect();
        let mut dx = vec![0f32; n * k];
        if !cortiq_engine::gpu_wgpu::gemm_dx_f32(&dy, &w, &mut dx, n, k, m) {
            eprintln!("dx ({n},{k},{m}): the GPU arm declined — skipping");
            continue;
        }
        ran += 1;
        let r = cpu_ref_nn(&dy, &w, n, k, m);
        let scale = r.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = dx
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-2 * scale.max(1.0),
            "dx ({n},{k},{m}): worst |Δ| {worst} (scale {scale})"
        );
        println!("dx ({n},{k},{m}): worst |Δ| {worst:.2e} against scale {scale:.2}");
    }
    if ran == 0 {
        eprintln!("no GPU arm engaged for dx — skipped");
    }
}

/// The whole frozen-FFN chain against its own composition on the host:
/// gate+up GEMM → silu·mul(·scale) → down GEMM, with and without the
/// mask gate, with and without the training readback of the middle plane.
#[test]
fn bake_ffn_chain_matches_host_composition() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    let (n, hsz, inter) = (64usize, 256usize, 512usize);
    let n2: Vec<f32> = (0..n * hsz)
        .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
        .collect();
    let gu: Vec<f32> = (0..2 * inter * hsz)
        .map(|i| ((i * 17 + 41) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let down: Vec<f32> = (0..hsz * inter)
        .map(|i| ((i * 23 + 5) % 89) as f32 / 89.0 - 0.5)
        .collect();
    let scale: Vec<f32> = (0..inter).map(|j| 0.5 + (j % 7) as f32 * 0.1).collect();
    let silu = |x: f32| x / (1.0 + (-x).exp());
    for (sc, want_both) in [(None, false), (Some(&scale[..]), true)] {
        let mut ffn = vec![0f32; n * hsz];
        let mut both = vec![0f32; n * 2 * inter];
        let got = cortiq_engine::gpu_wgpu::ffn_chain_f32(
            &n2,
            &gu,
            &down,
            sc,
            want_both.then_some(&mut both[..]),
            &mut ffn,
            n,
            hsz,
            inter,
        );
        if !got {
            eprintln!("chain declined (no cooperative arm here) — skipped");
            return;
        }
        // Host composition in f64-free f32, the reference the scalar
        // path computes.
        let mut r_both = vec![0f32; n * 2 * inter];
        for i in 0..n {
            for j in 0..2 * inter {
                let mut a = 0f64;
                for p in 0..hsz {
                    a += n2[i * hsz + p] as f64 * gu[j * hsz + p] as f64;
                }
                r_both[i * 2 * inter + j] = a as f32;
            }
        }
        let mut r_ffn = vec![0f32; n * hsz];
        for i in 0..n {
            for o in 0..hsz {
                let mut a = 0f64;
                for j in 0..inter {
                    let g = r_both[i * 2 * inter + j];
                    let u = r_both[i * 2 * inter + inter + j];
                    let mut v = silu(g) * u;
                    if let Some(s) = sc {
                        v *= s[j];
                    }
                    a += v as f64 * down[o * inter + j] as f64;
                }
                r_ffn[i * hsz + o] = a as f32;
            }
        }
        let fscale = r_ffn.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = ffn
            .iter()
            .zip(&r_ffn)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 2e-2 * fscale.max(1.0),
            "chain(scale={}): worst |Δ| {worst} (scale {fscale})",
            sc.is_some()
        );
        if want_both {
            let bscale = r_both.iter().fold(0f32, |a, v| a.max(v.abs()));
            let bworst = both
                .iter()
                .zip(&r_both)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                bworst <= 1e-2 * bscale.max(1.0),
                "chain middle plane: worst |Δ| {bworst} (scale {bscale})"
            );
        }
        println!("chain(scale={}): ffn worst |Δ| {worst:.2e}", sc.is_some());
    }
}
