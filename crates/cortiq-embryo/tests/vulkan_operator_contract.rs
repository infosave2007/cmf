//! Required native Vulkan operator-contract regressions.
//!
//! These gates compare resident WGSL operators against small f64 witnesses.
//! They intentionally use nonzero offsets/accumulators and cross the legacy
//! hybrid-k chunk boundary; finite-only smoke is not a substitute.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, HkDims, HkGrads, HkWork, RouteDims, ctx, hk_pow_table};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::{hk_decay_grid, hk_ref_bwd, hk_ref_fwd, lcg_vec};

fn assert_close(name: &str, got: &[f32], want: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), want.len(), "{name} length");
    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w).abs();
        let lim = atol + rtol * w.abs();
        if err > worst {
            worst = err;
            worst_i = i;
        }
        assert!(
            err <= lim,
            "{name}[{i}] got {g:?}, want {w:?}, err {err:e} > {lim:e}"
        );
    }
    let _ = (worst, worst_i);
}

fn as_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

fn adamw_witness_step(
    p: &mut [f64],
    m: &mut [f64],
    v: &mut [f64],
    g: &[f32],
    off: usize,
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: u32,
    gscale: f32,
) {
    let t = step.max(1) as f64;
    let bc1 = (1.0 / (1.0 - (beta1 as f64).powf(t))) as f32 as f64;
    let bc2 = (1.0 / (1.0 - (beta2 as f64).powf(t))) as f32 as f64;
    let b1 = beta1 as f64;
    let b2 = beta2 as f64;
    let one_minus_b1 = (1.0f32 - beta1) as f64;
    let one_minus_b2 = (1.0f32 - beta2) as f64;
    for i in off..off + n {
        let gr = (g[i] * gscale) as f64;
        let mm = b1 * m[i] + one_minus_b1 * gr;
        let vv = b2 * v[i] + one_minus_b2 * gr * gr;
        m[i] = (mm as f32) as f64;
        v[i] = (vv as f32) as f64;
        let update = (m[i] * bc1) / ((v[i] * bc2).sqrt() + eps as f64);
        p[i] = (p[i] - (lr as f64) * (update + (wd as f64) * p[i])) as f32 as f64;
    }
}

#[test]
fn rmsnorm_backward_honors_existing_dx_and_dw() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let rows = 3usize;
    let d = 5usize;
    let woff = 2usize;
    let dwoff = 3usize;
    let x: Vec<f32> = vec![
        0.25, -1.5, 2.0, 0.75, -0.5, 1.25, 0.5, -0.75, 1.75, -2.0, 0.5, 1.0, -1.25, 0.25, 2.25,
    ];
    let dy: Vec<f32> = vec![
        -0.75, 0.25, 1.5, -1.25, 0.5, 0.625, -1.75, 0.375, 1.125, -0.25, 1.0, -0.5, 0.875, -1.25,
        0.75,
    ];
    let inv = vec![0.8f32, 1.1, 0.65];
    let mut w = vec![99.0f32, -77.0];
    w.extend([0.5, -1.25, 0.75, 1.5, -0.25]);
    let mut dx0 = vec![0.25f32; rows * d];
    dx0[3] = -0.5;
    dx0[11] = 0.75;
    let mut dw0 = vec![7.0f32, -3.0, 5.0];
    dw0.extend([0.125, -0.25, 0.5, -0.75, 1.0]);
    let beta = 0.35f32;

    let gx = GBuf::from_slice(c, &x);
    let gw = GBuf::from_slice(c, &w);
    let gdy = GBuf::from_slice(c, &dy);
    let ginv = GBuf::from_slice(c, &inv);
    let gdx = GBuf::from_slice(c, &dx0);
    let gdw = GBuf::from_slice(c, &dw0);
    let cmd = Cmd::new(c);
    cmd.rmsnorm_bwd_at(
        &gx, &gw, woff, &gdy, &ginv, &gdx, beta, &gdw, dwoff, rows, d,
    );
    cmd.commit();

    let mut want_dx = vec![0.0f64; rows * d];
    let mut want_dw = dw0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for row in 0..rows {
        let mut dot = 0.0f64;
        for col in 0..d {
            dot += dy[row * d + col] as f64 * w[woff + col] as f64 * x[row * d + col] as f64;
        }
        let c = inv[row] as f64 * inv[row] as f64 * inv[row] as f64 * dot / d as f64;
        for col in 0..d {
            let i = row * d + col;
            let xx = x[i] as f64;
            let ww = w[woff + col] as f64;
            let yy = dy[i] as f64;
            want_dx[i] = beta as f64 * dx0[i] as f64 + inv[row] as f64 * yy * ww - c * xx;
            want_dw[dwoff + col] += dy[i] as f64 * xx * inv[row] as f64;
        }
    }
    assert_close("rmsnorm dx", &gdx.to_vec(), &want_dx, 3e-6, 3e-6);
    assert_close("rmsnorm dw", &gdw.to_vec(), &want_dw, 3e-6, 3e-6);
}

#[test]
fn kappa_backward_honors_rows_and_padded_ld() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let rows = 4usize;
    let nh = 3usize;
    let ld = 5usize;
    let bias = 0.37f32;
    let mut pre = vec![0.0f32; rows * ld];
    for row in 0..rows {
        for col in 0..ld {
            pre[row * ld + col] = (row as f32 * 0.7) - (col as f32 * 0.31) + 0.1;
        }
    }
    let dkap: Vec<f32> = (0..rows * nh)
        .map(|i| 0.25 + (i % 5) as f32 * 0.4)
        .collect();
    let gpre = GBuf::from_slice(c, &pre);
    let gkap = GBuf::zeros(c, rows * nh);
    let gdkap = GBuf::from_slice(c, &dkap);
    let gdpre = GBuf::from_slice(c, &vec![17.0f32; rows * ld]);
    let cmd = Cmd::new(c);
    cmd.kappa_fwd(&gpre, &gkap, rows, nh, ld, bias);
    cmd.kappa_bwd(&gkap, &gdkap, &gdpre, rows, nh, ld);
    cmd.commit();

    let mut want_kap = vec![0.0f64; rows * nh];
    let mut want_pre = vec![0.0f64; rows * ld];
    for row in 0..rows {
        for col in 0..ld {
            if col < nh {
                let k = 1.0f64 / (1.0f64 + (-(pre[row * ld + col] as f64 + bias as f64)).exp());
                want_kap[row * nh + col] = k;
                want_pre[row * ld + col] = dkap[row * nh + col] as f64 * k * (1.0 - k);
            }
        }
    }
    assert_close("kappa", &gkap.to_vec(), &want_kap, 3e-6, 3e-6);
    assert_close("kappa dpre", &gdpre.to_vec(), &want_pre, 3e-6, 3e-6);
}

#[test]
fn conv4_is_causal_with_asymmetric_taps_in_both_directions() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let (b, t, h, k) = (2usize, 7usize, 3usize, 4usize);
    let woff = 2usize;
    let dwoff = 5usize;
    let x: Vec<f32> = (0..b * t * h)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.07)
        .collect();
    let dy: Vec<f32> = (0..b * t * h)
        .map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.11)
        .collect();
    let mut w = vec![99.0f32; woff + h * k];
    for col in 0..h {
        for tap in 0..k {
            w[woff + col * k + tap] = (0.2 + col as f32 * 0.13) * (tap as f32 + 1.0);
        }
    }
    let mut dw0 = vec![3.0f32; dwoff + h * k];
    dw0[..dwoff].fill(-7.0);
    let gx = GBuf::from_slice(c, &x);
    let gw = GBuf::from_slice(c, &w);
    let gdy = GBuf::from_slice(c, &dy);
    let gy = GBuf::zeros(c, b * t * h);
    let gdx = GBuf::zeros(c, b * t * h);
    let gdw = GBuf::from_slice(c, &dw0);
    let cmd = Cmd::new(c);
    cmd.conv1d_fwd_at(&gx, &gw, woff, &gy, b, t, h, k);
    cmd.conv1d_bwd_at(&gx, &gw, woff, &gdy, &gdx, &gdw, dwoff, b, t, h, k);
    cmd.commit();

    let mut want_y = vec![0.0f64; b * t * h];
    for bi in 0..b {
        for ti in 0..t {
            for col in 0..h {
                let out = (bi * t + ti) * h + col;
                for tap in 0..k {
                    let back = k - 1 - tap;
                    if ti >= back {
                        let src = (bi * t + ti - back) * h + col;
                        want_y[out] += w[woff + col * k + tap] as f64 * x[src] as f64;
                    }
                }
            }
        }
    }
    let mut want_dx = vec![0.0f64; b * t * h];
    for bi in 0..b {
        for ti in 0..t {
            for col in 0..h {
                let out = (bi * t + ti) * h + col;
                for tap in 0..k {
                    let dst = ti + (k - 1 - tap);
                    if dst < t {
                        want_dx[out] +=
                            w[woff + col * k + tap] as f64 * dy[(bi * t + dst) * h + col] as f64;
                    }
                }
            }
        }
    }
    let mut want_dw = dw0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for col in 0..h {
        for tap in 0..k {
            let back = k - 1 - tap;
            let mut sum = 0.0f64;
            for bi in 0..b {
                for ti in back..t {
                    sum += dy[(bi * t + ti) * h + col] as f64
                        * x[(bi * t + ti - back) * h + col] as f64;
                }
            }
            want_dw[dwoff + col * k + tap] += sum;
        }
    }
    assert_close("conv forward", &gy.to_vec(), &want_y, 3e-6, 3e-6);
    assert_close("conv dx", &gdx.to_vec(), &want_dx, 3e-6, 3e-6);
    assert_close("conv dw", &gdw.to_vec(), &want_dw, 3e-6, 3e-6);
}

#[test]
fn legacy_hybrid_k_crosses_64_token_state_boundary() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let d = HkDims {
        b: 1,
        t: 128,
        nh: 2,
        nph: 3,
        dv: 5,
    };
    let rows = d.b * d.t;
    let thq: Vec<f32> = lcg_vec(0x1001, rows * d.nh * d.nph)
        .into_iter()
        .map(|x| x * 0.7)
        .collect();
    let thk: Vec<f32> = lcg_vec(0x1002, rows * d.nh * d.nph)
        .into_iter()
        .map(|x| x * 0.8)
        .collect();
    let v: Vec<f32> = lcg_vec(0x1003, rows * d.nh * d.dv)
        .into_iter()
        .map(|x| x * 0.4)
        .collect();
    let kappa: Vec<f32> = lcg_vec(0x1004, rows * d.nh)
        .into_iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect();
    let dout: Vec<f32> = lcg_vec(0x1005, rows * d.nh * d.dv)
        .into_iter()
        .map(|x| x * 0.5)
        .collect();
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let want_o = hk_ref_fwd(
        &d,
        &as_f64(&thq),
        &as_f64(&thk),
        &as_f64(&v),
        &as_f64(&kappa),
        &as_f64(&decay),
    );
    let (want_q, want_k, want_v, want_kap) = hk_ref_bwd(
        &d,
        &as_f64(&thq),
        &as_f64(&thk),
        &as_f64(&v),
        &as_f64(&kappa),
        &as_f64(&decay),
        &as_f64(&dout),
    );

    let z = |n: usize| GBuf::zeros(c, n);
    let gthq = GBuf::from_slice(c, &thq);
    let gthk = GBuf::from_slice(c, &thk);
    let gv = GBuf::from_slice(c, &v);
    let gkap = GBuf::from_slice(c, &kappa);
    let gphq = z(rows * d.nh * d.p2());
    let gphk = z(rows * d.nh * d.p2());
    let gkv = z(rows * d.nh * d.dv);
    let states = z(d.b * d.nh * (d.t / 64 + 1) * d.p2() * d.dv);
    let gout = z(rows * d.nh * d.dv);
    let gpow = GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, d.nph));
    let w = HkWork {
        thq: &gthq,
        thk: &gthk,
        v: &gv,
        kappa: &gkap,
        pow: &gpow,
        pow_off: 0,
        phq: &gphq,
        phk: &gphk,
        kv: &gkv,
        states: &states,
        out: &gout,
        phase_chunk: None,
        phase_partial: None,
    };
    let gdst = z(states.len());
    let gdkv = z(rows * d.nh * d.dv);
    let gdphq = z(rows * d.nh * d.p2());
    let gdphk = z(rows * d.nh * d.p2());
    let gdthq = z(rows * d.nh * d.nph);
    let gdthk = z(rows * d.nh * d.nph);
    let gdv = z(rows * d.nh * d.dv);
    let gdkap = z(rows * d.nh);
    let gdout = GBuf::from_slice(c, &dout);
    let gr = HkGrads {
        dout: &gdout,
        dstates: &gdst,
        dkv: &gdkv,
        dphq: &gdphq,
        dphk: &gdphk,
        dthq: &gdthq,
        dthk: &gdthk,
        dv: &gdv,
        dkappa: &gdkap,
    };
    let cmd = Cmd::new(c);
    cmd.hk_forward(&d, &w);
    cmd.hk_backward(&d, &w, &gr, 0.0);
    cmd.commit();

    assert_close("legacy HK forward", &gout.to_vec(), &want_o, 2e-4, 2e-4);
    assert_close("legacy HK dtheta-q", &gdthq.to_vec(), &want_q, 2e-4, 2e-4);
    assert_close("legacy HK dtheta-k", &gdthk.to_vec(), &want_k, 2e-4, 2e-4);
    assert_close("legacy HK dv", &gdv.to_vec(), &want_v, 2e-4, 2e-4);
    assert_close("legacy HK dkappa", &gdkap.to_vec(), &want_kap, 2e-4, 2e-4);
}

#[test]
fn resonance_route_uses_projection_bias_and_offsets() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let r = RouteDims {
        rows: 4,
        h: 4,
        e: 3,
        k: 2,
        cap: 8,
    };
    let mu_off = 5usize;
    let u_off = 7usize;
    let bias_off = 3usize;
    let x = vec![
        0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0,
    ];
    let mut mu = vec![71.0f32; mu_off + r.e * r.h];
    let means = [
        [0.0f32, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ];
    for (e, mean) in means.iter().enumerate() {
        mu[mu_off + e * r.h..mu_off + (e + 1) * r.h].copy_from_slice(mean);
    }
    let mut u = vec![83.0f32; u_off + r.e * r.k * r.h];
    let basis = [
        [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[0.0f32, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[1.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    ];
    for e in 0..r.e {
        for q in 0..r.k {
            let off = u_off + (e * r.k + q) * r.h;
            u[off..off + r.h].copy_from_slice(&basis[e][q]);
        }
    }
    let mut bias = vec![61.0f32; bias_off + r.e];
    bias[bias_off..bias_off + r.e].copy_from_slice(&[0.0, -0.25, 0.5]);
    let gx = GBuf::from_slice(c, &x);
    let gmu = GBuf::from_slice(c, &mu);
    let gu = GBuf::from_slice(c, &u);
    let gbias = GBuf::from_slice(c, &bias);
    let gassign = GBuf::from_u32(c, &vec![u32::MAX; r.rows]);
    let gres = GBuf::from_slice(c, &vec![-1.0f32; r.rows]);
    let cmd = Cmd::new(c);
    cmd.route(
        &r, &gx, &gmu, mu_off, &gu, u_off, &gbias, bias_off, &gassign, &gres,
    );
    cmd.commit();

    let mut want_assign = vec![0u32; r.rows];
    let mut want_res = vec![0.0f64; r.rows];
    for row in 0..r.rows {
        let mut best = f64::INFINITY;
        let mut best_e = 0usize;
        for e in 0..r.e {
            let mut d2 = 0.0f64;
            let mut proj = 0.0f64;
            for col in 0..r.h {
                let delta = x[row * r.h + col] as f64 - mu[mu_off + e * r.h + col] as f64;
                d2 += delta * delta;
            }
            for q in 0..r.k {
                let mut p = 0.0f64;
                for col in 0..r.h {
                    let delta = x[row * r.h + col] as f64 - mu[mu_off + e * r.h + col] as f64;
                    p += delta * u[u_off + (e * r.k + q) * r.h + col] as f64;
                }
                proj += p * p;
            }
            let residual = d2 - proj;
            let score = residual - bias[bias_off + e] as f64;
            if score < best {
                best = score;
                best_e = e;
                want_res[row] = residual;
            }
        }
        want_assign[row] = best_e as u32;
    }
    let got_assign = unsafe { gassign.as_u32_slice().to_vec() };
    assert_eq!(got_assign, want_assign, "router assignment");
    assert_close("router resonance", &gres.to_vec(), &want_res, 3e-6, 3e-6);
}

#[test]
fn descriptor_init_honors_layer_offset() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let r = RouteDims {
        rows: 4,
        h: 3,
        e: 2,
        k: 1,
        cap: 4,
    };
    let off = 5usize;
    let x: Vec<f32> = (0..r.rows * r.h).map(|i| (i as f32 - 4.0) * 0.25).collect();
    let seed_rows = vec![3u32, 1];
    let mu0 = vec![91.0f32; off + r.e * r.h];
    let gx = GBuf::from_slice(c, &x);
    let grows = GBuf::from_u32(c, &seed_rows);
    let gmu = GBuf::from_slice(c, &mu0);
    let cmd = Cmd::new(c);
    cmd.moe_init_mu(&r, &gx, &grows, &gmu, off);
    cmd.commit();

    let mut want = mu0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for (e, &row) in seed_rows.iter().enumerate() {
        let src = row as usize * r.h;
        let dst = off + e * r.h;
        for col in 0..r.h {
            want[dst + col] = x[src + col] as f64;
        }
    }
    assert_close("descriptor mu seed", &gmu.to_vec(), &want, 3e-6, 3e-6);
}

#[test]
fn embedding_scatter_add_accumulates_repeated_tokens() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let rows = 1024usize;
    let d = 4usize;
    let vocab = 1024usize;
    let de_off = 7usize;
    let mut tok = vec![0u32; rows];
    let mut dx = vec![0.0f32; rows * d];
    for row in 0..rows {
        let token = if row % 4 == 0 { 17 } else { 13 };
        tok[row] = token;
        let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
        let base = row * d;
        dx[base] = sign;
        dx[base + 1] = if token == 13 { 0.125 } else { -0.25 };
        dx[base + 2] = if row % 8 == 0 { 1.0 } else { -1.0 };
        dx[base + 3] = 0.003 * (row % 11) as f32;
    }
    let mut de0 = vec![0.0f32; de_off + vocab * d];
    de0[de_off + 13 * d] = 0.5;
    de0[de_off + 13 * d + 1] = -0.75;
    de0[de_off + 17 * d] = -1.25;
    let gde = GBuf::from_slice(c, &de0);
    let gt = GBuf::from_u32(c, &tok);
    let gdx = GBuf::from_slice(c, &dx);
    let cmd = Cmd::new(c);
    cmd.embed_scatter_add(&gde, de_off, &gt, &gdx, rows, d);
    cmd.commit();

    let mut want = de0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for row in 0..rows {
        let off = de_off + tok[row] as usize * d;
        for col in 0..d {
            want[off + col] += dx[row * d + col] as f64;
        }
    }
    assert_close("tied embedding gradient", &gde.to_vec(), &want, 2e-5, 2e-6);
}

#[test]
fn sumsq_at_reduces_every_element_with_offsets_and_sentinels() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    // The WGSL reduction emits one partial per 256-element block.  Keep the
    // input and partial offsets nonzero, and make the old one-sample-per-block
    // implementation observably wrong by leaving block-leading entries zero.
    let specs = [
        (1usize, 0x7101u64, false),
        (255, 0x7102, true),
        (256, 0x7103, false),
        (257, 0x7104, true),
        (4096, 0x7105, false),
        (4097, 0x7106, true),
        (8193, 0x7107, false),
    ];
    let x_guard = 4usize;
    let part_guard = 7usize;
    let mut cases = Vec::with_capacity(specs.len());
    let mut part_len = 0usize;
    for (case_id, &(n, seed, adversarial)) in specs.iter().enumerate() {
        let mut values = vec![0.0f32; n];
        if n == 1 {
            values[0] = 3.25;
        } else {
            let random = lcg_vec(seed, n);
            for i in 0..n {
                if i % 256 == 0 {
                    continue;
                }
                values[i] = if adversarial {
                    let base = match i % 5 {
                        0 => 3.0,
                        1 => 0.03125,
                        2 => 1.75,
                        3 => 0.5,
                        _ => 2.25,
                    };
                    base + random[i] * 0.003
                } else {
                    0.125 + random[i].abs() * 0.875
                };
            }
        }
        let x_off = x_guard + case_id;
        let mut input = vec![7000.0 + case_id as f32; x_off + n + 3];
        input[x_off..x_off + n].copy_from_slice(&values);
        let groups = n.div_ceil(256);
        let part_off = part_guard + case_id * 40;
        part_len = part_len.max(part_off + groups + 2);
        cases.push((n, x_off, part_off, values, GBuf::from_slice(c, &input)));
    }
    let partial0 = vec![9137.0f32; part_len];
    let gpartial = GBuf::from_slice(c, &partial0);
    let cmd = Cmd::new(c);
    for (n, x_off, part_off, _, gx) in &cases {
        cmd.sumsq_at(gx, *x_off, *n, &gpartial, *part_off);
    }
    cmd.commit();

    let mut want = partial0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for (n, _, part_off, values, _) in &cases {
        let groups = n.div_ceil(256);
        for group in 0..groups {
            let first = group * 256;
            let last = (*n).min(first + 256);
            let mut sum = 0.0f64;
            for &value in &values[first..last] {
                sum += (value as f64) * (value as f64);
            }
            want[*part_off + group] = sum;
        }
    }
    assert_close("sumsq partials", &gpartial.to_vec(), &want, 3e-3, 3e-6);
}

#[test]
fn adamw_at_updates_all_outputs_with_offset_and_resume() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let off = 5usize;
    let n = 259usize;
    let total = off + n + 7;
    let mut p0 = vec![0.0f32; total];
    let mut g0 = vec![0.0f32; total];
    let mut m0 = vec![0.0f32; total];
    let mut v0 = vec![0.0f32; total];
    for i in 0..total {
        p0[i] = -700.0 - i as f32;
        g0[i] = 900.0 + i as f32;
        m0[i] = 500.0 + i as f32;
        v0[i] = 300.0 + i as f32;
    }
    for i in off..off + n {
        let j = i - off;
        p0[i] = -1.2 + (j % 11) as f32 * 0.19;
        g0[i] = -0.8 + (j % 13) as f32 * 0.11;
        m0[i] = 0.15 + (j % 7) as f32 * 0.07;
        v0[i] = 0.25 + (j % 9) as f32 * 0.13;
    }
    let lr = 0.007f32;
    let beta1 = 0.81f32;
    let beta2 = 0.93f32;
    let eps = 1e-4f32;
    let wd = 0.031f32;
    let gscale = 0.63f32;
    let gp = GBuf::from_slice(c, &p0);
    let gg = GBuf::from_slice(c, &g0);
    let gm = GBuf::from_slice(c, &m0);
    let gv = GBuf::from_slice(c, &v0);
    let mut want_p = as_f64(&p0);
    let mut want_m = as_f64(&m0);
    let mut want_v = as_f64(&v0);
    let cmd = Cmd::new(c);
    cmd.adamw_at(
        &gp, &gg, &gm, &gv, off, n, lr, beta1, beta2, eps, wd, 1, gscale,
    );
    cmd.commit();
    adamw_witness_step(
        &mut want_p,
        &mut want_m,
        &mut want_v,
        &g0,
        off,
        n,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        1,
        gscale,
    );
    assert_close("AdamW p step1", &gp.to_vec(), &want_p, 3e-5, 3e-6);
    assert_close("AdamW m step1", &gm.to_vec(), &want_m, 3e-5, 3e-6);
    assert_close("AdamW v step1", &gv.to_vec(), &want_v, 3e-5, 3e-6);
    assert_eq!(gg.to_vec(), g0, "AdamW must not mutate gradients");

    let cmd = Cmd::new(c);
    cmd.adamw_at(
        &gp, &gg, &gm, &gv, off, n, lr, beta1, beta2, eps, wd, 7, gscale,
    );
    cmd.commit();
    adamw_witness_step(
        &mut want_p,
        &mut want_m,
        &mut want_v,
        &g0,
        off,
        n,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        7,
        gscale,
    );
    assert_close("AdamW p resumed", &gp.to_vec(), &want_p, 3e-5, 3e-6);
    assert_close("AdamW m resumed", &gm.to_vec(), &want_m, 3e-5, 3e-6);
    assert_close("AdamW v resumed", &gv.to_vec(), &want_v, 3e-5, 3e-6);
    assert_eq!(
        gg.to_vec(),
        g0,
        "AdamW must preserve gradients after resume"
    );
}

#[test]
fn row_scatter_add_accumulates_float_bits_for_repeated_targets() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let rows = 5usize;
    let d = 3usize;
    let idx = vec![2u32, 1, 2, u32::MAX, 1];
    let src = vec![
        0.25f32, -1.5, 2.0, 1.0, 0.5, -0.75, -0.25, 2.5, 1.25, 9.0, 9.0, 9.0, 0.75, -0.5, 0.125,
    ];
    let dst0 = vec![
        31.0f32, 31.0, 31.0, 0.5, -0.25, 1.0, -2.0, 0.75, 4.0, 17.0, 17.0, 17.0,
    ];
    let gdst = GBuf::from_slice(c, &dst0);
    let gidx = GBuf::from_u32(c, &idx);
    let gsrc = GBuf::from_slice(c, &src);
    let cmd = Cmd::new(c);
    cmd.scatter_add_rows(&gdst, &gidx, &gsrc, rows, d);
    cmd.commit();

    let mut want = dst0.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for row in 0..rows {
        let target = idx[row] as i32;
        if target >= 0 {
            for col in 0..d {
                want[target as usize * d + col] += src[row * d + col] as f64;
            }
        }
    }
    assert_close("row scatter add", &gdst.to_vec(), &want, 3e-6, 3e-6);
}

#[test]
fn unsupported_vulkan_optional_tails_are_rejected_before_training() {
    let c = ctx().expect("native Vulkan adapter is required for this test");
    let mut gdn_cfg = EmbryoCfg::tiny();
    gdn_cfg.gdn_lane = true;
    let gdn_layout = Layout::new(&gdn_cfg);
    let gdn_params = init_params(&gdn_cfg, &gdn_layout, 0x55aa);
    assert!(
        EmbryoGpu::new(gdn_cfg, 1, 64, &gdn_params).is_none(),
        "unsupported Vulkan GDN must fail at construction, not panic during train"
    );

    let mut decay_cfg = EmbryoCfg::tiny();
    decay_cfg.learn_decay = true;
    let decay_layout = Layout::new(&decay_cfg);
    let decay_params = init_params(&decay_cfg, &decay_layout, 0x55ab);
    assert!(
        EmbryoGpu::new(decay_cfg, 1, 64, &decay_params).is_none(),
        "unsupported Vulkan learned decay must fail at construction"
    );

    let mut smooth_cfg = EmbryoCfg::tiny();
    smooth_cfg.router_smooth_k4 = true;
    let smooth_layout = Layout::new(&smooth_cfg);
    let smooth_params = init_params(&smooth_cfg, &smooth_layout, 0x55ac);
    assert!(
        EmbryoGpu::new(smooth_cfg, 1, 64, &smooth_params).is_none(),
        "unsupported Vulkan smoothed routing must fail at construction"
    );

    let mut top2_cfg = EmbryoCfg::tiny();
    top2_cfg.router_top2_margin = Some(0.1);
    let top2_layout = Layout::new(&top2_cfg);
    let top2_params = init_params(&top2_cfg, &top2_layout, 0x55ad);
    assert!(
        EmbryoGpu::new(top2_cfg, 1, 64, &top2_params).is_none(),
        "unsupported Vulkan top-2 routing must fail at construction"
    );

    let _ = c;
}
