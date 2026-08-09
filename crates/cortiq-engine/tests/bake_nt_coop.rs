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
            None,
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

/// The whole attention forward chain against a host reference that
/// mirrors the replica's math: qkv GEMM → bias + gate split → qk-RMSNorm
/// → RoPE (f64 angles) → causal max-softmax·V per head (GQA) → σ(gate)
/// → wo GEMM. Gated + normed + biased — every branch the kernel has.
#[test]
fn bake_attn_chain_matches_host_reference() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    let (b, t, nh, nkv, hd, hsz) = (2usize, 64usize, 4usize, 2usize, 64usize, 256usize);
    let n = b * t;
    let (qdim, kvdim) = (nh * hd, nkv * hd);
    let qrows = 2 * qdim; // output gate on
    let fused = qrows + 2 * kvdim;
    let half = hd / 2;
    let g = |i: usize, m: usize, s: f32| ((i * 29 + m) % 97) as f32 / 97.0 * s - s / 2.0;
    let n1: Vec<f32> = (0..n * hsz).map(|i| g(i, 13, 1.0)).collect();
    let wqkv: Vec<f32> = (0..fused * hsz).map(|i| g(i, 17, 0.2)).collect();
    let wo: Vec<f32> = (0..hsz * qdim).map(|i| g(i, 23, 0.2)).collect();
    let qn: Vec<f32> = (0..hd).map(|i| 0.8 + g(i, 5, 0.2)).collect();
    let kn: Vec<f32> = (0..hd).map(|i| 0.9 + g(i, 7, 0.2)).collect();
    let bias: Vec<f32> = (0..fused).map(|i| g(i, 11, 0.1)).collect();
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / 10_000f64.powf(2.0 * i as f64 / hd as f64))
        .collect();
    let mut rope = Vec::with_capacity(t * half * 2);
    for pos in 0..t {
        for &f in &inv_freq {
            let a = pos as f64 * f;
            rope.push(a.cos() as f32);
            rope.push(a.sin() as f32);
        }
    }
    let eps = 1e-6f32;
    let cfg = cortiq_engine::gpu_wgpu::AttnChainCfg {
        wqkv: &wqkv,
        wo: &wo,
        q_norm: Some(&qn),
        k_norm: Some(&kn),
        bias: Some(&bias),
        output_gate: true,
        gemma: false,
        eps,
        rotary_half: half,
        rope: &rope,
        b,
        t,
        nh,
        nkv,
        hd,
        hsz,
    };
    let mut got = vec![0f32; n * hsz];
    let Some(acts) = cortiq_engine::gpu_wgpu::attn_chain_f32(&n1, &cfg, &mut got, true) else {
        eprintln!("attention chain declined (no cooperative arm here) — skipped");
        return;
    };
    // ── host reference ──
    let mut plane = vec![0f32; n * fused];
    for r in 0..n {
        for j in 0..fused {
            let mut a = 0f64;
            for p in 0..hsz {
                a += n1[r * hsz + p] as f64 * wqkv[j * hsz + p] as f64;
            }
            plane[r * fused + j] = a as f32 + bias[j];
        }
    }
    let silu_sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut qrot = vec![0f32; n * qdim];
    let mut krot = vec![0f32; n * kvdim];
    let mut vproj = vec![0f32; n * kvdim];
    let mut gate_pre = vec![0f32; n * qdim];
    let norm_rope = |head: &mut [f32], w: &[f32], pos: usize| {
        let ss: f64 = head.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let inv = (1.0 / (ss / hd as f64 + eps as f64).sqrt()) as f32;
        for (j, x) in head.iter_mut().enumerate() {
            *x *= inv * w[j];
        }
        for i in 0..half {
            let a = pos as f64 * inv_freq[i];
            let (c, s) = (a.cos() as f32, a.sin() as f32);
            let (x0, x1) = (head[i], head[i + half]);
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    };
    for r in 0..n {
        let pos = r % t;
        let row = &plane[r * fused..(r + 1) * fused];
        for h in 0..nh {
            let mut hb: Vec<f32> = row[2 * h * hd..2 * h * hd + hd].to_vec();
            norm_rope(&mut hb, &qn, pos);
            qrot[r * qdim + h * hd..r * qdim + (h + 1) * hd].copy_from_slice(&hb);
            gate_pre[r * qdim + h * hd..r * qdim + (h + 1) * hd]
                .copy_from_slice(&row[2 * h * hd + hd..2 * h * hd + 2 * hd]);
        }
        for gi in 0..nkv {
            let mut hb: Vec<f32> = row[qrows + gi * hd..qrows + (gi + 1) * hd].to_vec();
            norm_rope(&mut hb, &kn, pos);
            krot[r * kvdim + gi * hd..r * kvdim + (gi + 1) * hd].copy_from_slice(&hb);
            vproj[r * kvdim + gi * hd..r * kvdim + (gi + 1) * hd]
                .copy_from_slice(&row[qrows + kvdim + gi * hd..qrows + kvdim + (gi + 1) * hd]);
        }
    }
    let rep = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut ao = vec![0f32; n * qdim];
    for bi in 0..b {
        for h in 0..nh {
            let gi = h / rep;
            for ti in 0..t {
                let qb = (bi * t + ti) * qdim + h * hd;
                let mut sc = vec![0f64; ti + 1];
                let mut mx = f64::NEG_INFINITY;
                for j in 0..=ti {
                    let kb = (bi * t + j) * kvdim + gi * hd;
                    let mut s = 0f64;
                    for c in 0..hd {
                        s += qrot[qb + c] as f64 * krot[kb + c] as f64;
                    }
                    sc[j] = s * scale;
                    mx = mx.max(sc[j]);
                }
                let mut den = 0f64;
                for v in sc.iter_mut() {
                    *v = (*v - mx).exp();
                    den += *v;
                }
                for c in 0..hd {
                    let mut acc = 0f64;
                    for j in 0..=ti {
                        acc += sc[j] / den
                            * vproj[(bi * t + j) * kvdim + gi * hd + c] as f64;
                    }
                    ao[(bi * t + ti) * qdim + h * hd + c] = acc as f32;
                }
            }
        }
    }
    let mut r_out = vec![0f32; n * hsz];
    for r in 0..n {
        for o in 0..hsz {
            let mut a = 0f64;
            for j in 0..qdim {
                a += (ao[r * qdim + j] * silu_sig(gate_pre[r * qdim + j])) as f64
                    * wo[o * qdim + j] as f64;
            }
            r_out[r * hsz + o] = a as f32;
        }
    }
    let cmp = |name: &str, a: &[f32], r: &[f32], tol: f32| {
        let scale = r.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(r).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(
            worst <= tol * scale.max(1.0),
            "{name}: worst |Δ| {worst} (scale {scale})"
        );
        println!("attn chain {name}: worst |Δ| {worst:.2e} / scale {scale:.2}");
    };
    cmp("attn_out", &got, &r_out, 2e-2);
    cmp("qrot", &acts.qrot, &qrot, 1e-2);
    cmp("krot", &acts.krot, &krot, 1e-2);
    cmp("ao", &acts.ao, &ao, 1e-2);
}

/// The backward chain against its host composition, fed by the plane a
/// forward call parked — the resident-graph handshake end to end.
#[test]
fn bake_ffn_bwd_chain_matches_host_composition() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    let (n, hsz, inter, li) = (64usize, 256usize, 512usize, 7usize);
    let n2: Vec<f32> = (0..n * hsz)
        .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
        .collect();
    let gu: Vec<f32> = (0..2 * inter * hsz)
        .map(|i| ((i * 17 + 41) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let down: Vec<f32> = (0..hsz * inter)
        .map(|i| ((i * 23 + 5) % 89) as f32 / 89.0 - 0.5)
        .collect();
    let dh2: Vec<f32> = (0..n * hsz)
        .map(|i| ((i * 13 + 3) % 71) as f32 / 71.0 - 0.5)
        .collect();
    // Park the plane exactly as the checkpointed recompute does.
    let mut ffn = vec![0f32; n * hsz];
    let mut both = vec![0f32; n * 2 * inter];
    if !cortiq_engine::gpu_wgpu::ffn_chain_f32(
        &n2,
        &gu,
        &down,
        None,
        Some(&mut both[..]),
        Some(li),
        &mut ffn,
        n,
        hsz,
        inter,
    ) {
        eprintln!("forward chain declined (no cooperative arm) — skipped");
        return;
    }
    let mut dn2 = vec![0f32; n * hsz];
    if !cortiq_engine::gpu_wgpu::ffn_bwd_chain_f32(&dh2, &down, &gu, li, &mut dn2, n, hsz, inter)
    {
        // The forward parked a plane; on a cooperative device the
        // backward must take it (a non-discrete adapter may not).
        eprintln!("bwd chain declined (uma adapter?) — skipped");
        return;
    }
    // Host composition from the DEVICE plane — the same numbers the
    // trainer's host arm would read out of `both`.
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let silu_bwd = |x: f32| {
        let s = 1.0 / (1.0 + (-x).exp());
        s * (1.0 + x * (1.0 - s))
    };
    let mut r_dn2 = vec![0f32; n * hsz];
    for i in 0..n {
        // dact = dh2 · down
        let mut dact = vec![0f32; inter];
        for j in 0..inter {
            let mut a = 0f64;
            for o in 0..hsz {
                a += dh2[i * hsz + o] as f64 * down[o * inter + j] as f64;
            }
            dact[j] = a as f32;
        }
        for o in 0..hsz {
            let mut a = 0f64;
            for j in 0..inter {
                let g = both[i * 2 * inter + j];
                let u = both[i * 2 * inter + inter + j];
                let dg = dact[j] * u * silu_bwd(g);
                let du = dact[j] * silu(g);
                a += dg as f64 * gu[j * hsz + o] as f64
                    + du as f64 * gu[(inter + j) * hsz + o] as f64;
            }
            r_dn2[i * hsz + o] = a as f32;
        }
    }
    let scale = r_dn2.iter().fold(0f32, |a, v| a.max(v.abs()));
    let worst = dn2
        .iter()
        .zip(&r_dn2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 2e-2 * scale.max(1.0),
        "bwd chain: worst |Δ| {worst} (scale {scale})"
    );
    println!("bwd chain: dn2 worst |Δ| {worst:.2e} against scale {scale:.2}");
}
