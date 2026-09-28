//! Every Metal training kernel against the CPU reference (f64 accumulate).
//! Runs only where a Metal device exists (macOS); skips otherwise.
#![cfg(target_os = "macos")]

use cortiq_embryo::metal::{Cmd, GBuf, Op, ctx};
use cortiq_embryo::ops::{gemm_ref, lcg_vec};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn gemm_all_layouts_and_beta() {
    let Some(c) = ctx() else {
        eprintln!("no Metal — skip");
        return;
    };
    // (m, n, k) beyond one tile in every dimension, K crossing tile edges.
    for &(m, n, k) in &[
        (64usize, 64usize, 32usize),
        (128, 192, 96),
        (256, 64, 160),
        (64, 320, 32),
    ] {
        for &(ta, tb) in &[
            (Op::N, Op::N),
            (Op::N, Op::T),
            (Op::T, Op::N),
            (Op::T, Op::T),
        ] {
            let (arows, acols) = if ta == Op::N { (m, k) } else { (k, m) };
            let (brows, bcols) = if tb == Op::N { (k, n) } else { (n, k) };
            let a = lcg_vec(11 + m as u64, arows * acols);
            let b = lcg_vec(23 + n as u64, brows * bcols);
            let c0 = lcg_vec(37, m * n);
            for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.5, 1.0), (-2.0, 0.25)] {
                let mut want = c0.clone();
                gemm_ref(
                    ta == Op::T,
                    tb == Op::T,
                    m,
                    n,
                    k,
                    alpha,
                    &a,
                    acols,
                    &b,
                    bcols,
                    beta,
                    &mut want,
                    n,
                );
                let ga = GBuf::from_slice(c, &a);
                let gb = GBuf::from_slice(c, &b);
                let gc = GBuf::from_slice(c, &c0);
                let cmd = Cmd::new(c);
                cmd.gemm(
                    ta, tb, m, n, k, alpha, &ga, 0, acols, &gb, 0, bcols, beta, &gc, 0, n,
                );
                cmd.commit();
                let got = gc.to_vec();
                let d = max_abs_diff(&got, &want);
                assert!(
                    d < 2e-4,
                    "gemm {ta:?}{tb:?} m={m} n={n} k={k} alpha={alpha} beta={beta}: max|Δ|={d}"
                );
            }
        }
    }
}

#[test]
fn gemm_offsets_and_leading_dims() {
    let Some(c) = ctx() else { return };
    // A is a column block of a wider matrix (lda > K), C written into a
    // column block of a wider output (ldc > N) at an offset.
    let (m, n, k) = (128usize, 64usize, 64usize);
    let lda = 96;
    let ldc = 192;
    let a_full = lcg_vec(5, m * lda);
    let w = lcg_vec(6, n * k);
    let c_full = vec![7.0f32; m * ldc];
    let mut want = c_full.clone();
    // reference on the sub-block: A[:, 32..96], C[:, 64..128]
    let mut a_sub = vec![0.0f32; m * k];
    for i in 0..m {
        a_sub[i * k..(i + 1) * k].copy_from_slice(&a_full[i * lda + 32..i * lda + 96]);
    }
    let mut c_sub = vec![0.0f32; m * n];
    gemm_ref(
        false, true, m, n, k, 1.0, &a_sub, k, &w, k, 0.0, &mut c_sub, n,
    );
    for i in 0..m {
        want[i * ldc + 64..i * ldc + 128].copy_from_slice(&c_sub[i * n..(i + 1) * n]);
    }
    let ga = GBuf::from_slice(c, &a_full);
    let gw = GBuf::from_slice(c, &w);
    let gc = GBuf::from_slice(c, &c_full);
    let cmd = Cmd::new(c);
    cmd.gemm(
        Op::N,
        Op::T,
        m,
        n,
        k,
        1.0,
        &ga,
        32,
        lda,
        &gw,
        0,
        k,
        0.0,
        &gc,
        64,
        ldc,
    );
    cmd.commit();
    let d = max_abs_diff(&gc.to_vec(), &want);
    assert!(d < 1e-4, "offset gemm max|Δ|={d}");
}

#[test]
fn rmsnorm_swiglu_ce_adamw_match_cpu() {
    let Some(c) = ctx() else { return };
    let (rows, d) = (37usize, 384usize);
    let x = lcg_vec(1, rows * d);
    let w: Vec<f32> = lcg_vec(2, d).iter().map(|v| 1.0 + 0.1 * v).collect();
    let dy = lcg_vec(3, rows * d);
    let eps = 1e-6f32;
    // ---- rmsnorm fwd/bwd reference ----
    let mut y_ref = vec![0.0f32; rows * d];
    let mut inv_ref = vec![0.0f32; rows];
    let mut dx_ref = vec![0.0f32; rows * d];
    let mut dw_ref = vec![0.0f32; d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let ss: f64 = xr.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ss + eps as f64).sqrt();
        inv_ref[r] = inv as f32;
        for j in 0..d {
            y_ref[r * d + j] = (xr[j] as f64 * inv * w[j] as f64) as f32;
        }
        let dyr = &dy[r * d..(r + 1) * d];
        let dot: f64 = (0..d)
            .map(|j| dyr[j] as f64 * w[j] as f64 * xr[j] as f64)
            .sum();
        for j in 0..d {
            let g = dyr[j] as f64 * w[j] as f64;
            dx_ref[r * d + j] = (inv * g - inv * inv * inv * dot / d as f64 * xr[j] as f64) as f32;
            dw_ref[j] += (dyr[j] as f64 * xr[j] as f64 * inv) as f32;
        }
    }
    let gx = GBuf::from_slice(c, &x);
    let gw = GBuf::from_slice(c, &w);
    let gy = GBuf::zeros(c, rows * d);
    let ginv = GBuf::zeros(c, rows);
    let gdy = GBuf::from_slice(c, &dy);
    let gdx = GBuf::zeros(c, rows * d);
    let gdw = GBuf::zeros(c, d);
    let cmd = Cmd::new(c);
    cmd.rmsnorm_fwd(&gx, &gw, &gy, &ginv, rows, d, eps);
    cmd.rmsnorm_bwd(&gx, &gw, &gdy, &ginv, &gdx, 0.0, &gdw, rows, d);
    cmd.commit();
    assert!(max_abs_diff(&gy.to_vec(), &y_ref) < 1e-5, "rmsnorm fwd");
    assert!(max_abs_diff(&ginv.to_vec(), &inv_ref) < 1e-5, "rmsnorm inv");
    assert!(max_abs_diff(&gdx.to_vec(), &dx_ref) < 1e-4, "rmsnorm dx");
    assert!(max_abs_diff(&gdw.to_vec(), &dw_ref) < 1e-3, "rmsnorm dw");

    // ---- swiglu ----
    let n = 1000;
    let gate = lcg_vec(4, n);
    let up = lcg_vec(5, n);
    let dh = lcg_vec(6, n);
    let mut h_ref = vec![0.0f32; n];
    let mut dg_ref = vec![0.0f32; n];
    let mut du_ref = vec![0.0f32; n];
    for i in 0..n {
        let g = gate[i] as f64;
        let s = 1.0 / (1.0 + (-g).exp());
        h_ref[i] = (g * s * up[i] as f64) as f32;
        dg_ref[i] = (dh[i] as f64 * up[i] as f64 * (s * (1.0 + g * (1.0 - s)))) as f32;
        du_ref[i] = (dh[i] as f64 * g * s) as f32;
    }
    let gg = GBuf::from_slice(c, &gate);
    let gu = GBuf::from_slice(c, &up);
    let gh = GBuf::zeros(c, n);
    let gdh = GBuf::from_slice(c, &dh);
    let gdg = GBuf::zeros(c, n);
    let gdu = GBuf::zeros(c, n);
    let cmd = Cmd::new(c);
    cmd.swiglu_fwd(&gg, &gu, &gh, n);
    cmd.swiglu_bwd(&gg, &gu, &gdh, &gdg, &gdu, n);
    cmd.commit();
    assert!(max_abs_diff(&gh.to_vec(), &h_ref) < 1e-5, "swiglu fwd");
    assert!(max_abs_diff(&gdg.to_vec(), &dg_ref) < 1e-5, "swiglu dgate");
    assert!(max_abs_diff(&gdu.to_vec(), &du_ref) < 1e-5, "swiglu dup");

    // ---- softmax CE ----
    let (rows, v) = (13usize, 3000usize);
    let logits: Vec<f32> = lcg_vec(7, rows * v).iter().map(|x| x * 8.0).collect();
    let target: Vec<u32> = (0..rows as u32).map(|r| (r * 977) % v as u32).collect();
    let mut loss_ref = vec![0.0f32; rows];
    let mut dl_ref = vec![0.0f32; rows * v];
    let scale = 1.0 / rows as f32;
    for r in 0..rows {
        let lr = &logits[r * v..(r + 1) * v];
        let mx = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = lr.iter().map(|x| ((*x as f64) - mx).exp()).sum();
        let lse = mx + sum.ln();
        loss_ref[r] = (lse - lr[target[r] as usize] as f64) as f32;
        for j in 0..v {
            let p = ((lr[j] as f64) - lse).exp();
            dl_ref[r * v + j] =
                ((p - if j == target[r] as usize { 1.0 } else { 0.0 }) * scale as f64) as f32;
        }
    }
    let gl = GBuf::from_slice(c, &logits);
    let gt = GBuf::from_u32(c, &target);
    let gloss = GBuf::zeros(c, rows);
    let cmd = Cmd::new(c);
    cmd.softmax_ce(&gl, &gt, &gloss, rows, v, scale);
    cmd.commit();
    assert!(max_abs_diff(&gloss.to_vec(), &loss_ref) < 1e-4, "ce loss");
    assert!(max_abs_diff(&gl.to_vec(), &dl_ref) < 1e-6, "ce dlogits");

    // ---- adamw + sumsq ----
    let n = 5000;
    let p0 = lcg_vec(8, n);
    let g = lcg_vec(9, n);
    let m0: Vec<f32> = lcg_vec(10, n).iter().map(|x| x * 0.1).collect();
    let v0: Vec<f32> = lcg_vec(11, n).iter().map(|x| x.abs() * 0.01).collect();
    let (lr, b1, b2, eps, wd, step, gscale) =
        (1e-3f32, 0.9f32, 0.95f32, 1e-8f32, 0.1f32, 7u32, 0.5f32);
    let mut p_ref = p0.clone();
    let bc1 = 1.0 / (1.0 - (b1 as f64).powi(step as i32));
    let bc2 = 1.0 / (1.0 - (b2 as f64).powi(step as i32));
    for i in 0..n {
        let gr = g[i] as f64 * gscale as f64;
        let m = b1 as f64 * m0[i] as f64 + (1.0 - b1 as f64) * gr;
        let v = b2 as f64 * v0[i] as f64 + (1.0 - b2 as f64) * gr * gr;
        let upd = (m * bc1) / ((v * bc2).sqrt() + eps as f64);
        p_ref[i] = (p_ref[i] as f64 - lr as f64 * (upd + wd as f64 * p_ref[i] as f64)) as f32;
    }
    let gp = GBuf::from_slice(c, &p0);
    let gg = GBuf::from_slice(c, &g);
    let gm = GBuf::from_slice(c, &m0);
    let gv = GBuf::from_slice(c, &v0);
    let part = GBuf::zeros(c, 4096);
    let cmd = Cmd::new(c);
    let groups = cmd.sumsq(&gg, n, &part);
    cmd.adamw(&gp, &gg, &gm, &gv, n, lr, b1, b2, eps, wd, step, gscale);
    cmd.commit();
    assert!(max_abs_diff(&gp.to_vec(), &p_ref) < 1e-6, "adamw");
    let ss_gpu: f64 = part.as_slice()[..groups].iter().map(|x| *x as f64).sum();
    let ss_ref: f64 = g.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    assert!(
        (ss_gpu - ss_ref).abs() / ss_ref < 1e-5,
        "sumsq {ss_gpu} vs {ss_ref}"
    );

    // ---- embed gather ----
    let (vocab, d, rows) = (300usize, 64usize, 20usize);
    let e = lcg_vec(12, vocab * d);
    let tok: Vec<u32> = (0..rows as u32)
        .map(|r| (r * 37 + 5) % vocab as u32)
        .collect();
    let mut want = vec![0.0f32; rows * d];
    for r in 0..rows {
        want[r * d..(r + 1) * d]
            .copy_from_slice(&e[tok[r] as usize * d..(tok[r] as usize + 1) * d]);
    }
    let ge = GBuf::from_slice(c, &e);
    let gt = GBuf::from_u32(c, &tok);
    let go = GBuf::zeros(c, rows * d);
    let cmd = Cmd::new(c);
    cmd.embed_gather(&ge, &gt, &go, rows, d);
    cmd.commit();
    assert_eq!(go.to_vec(), want, "embed gather");
}

#[test]
fn hybrid_k_chunk_scan_matches_cpu_oracle() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd, hk_ref_fwd};
    let Some(c) = ctx() else { return };
    // 3 chunks, 2 sequences, 2 heads, nph 5 (p2 = 10), dv 32: exercises the
    // inter-chunk state path (states + dstates) and non-full register arrays.
    for &(nph, dv) in &[(5usize, 32usize), (32, 128)] {
        let d = HkDims {
            b: 2,
            t: 192,
            nh: 2,
            nph,
            dv,
        };
        let rows = d.b * d.t;
        let thq: Vec<f32> = lcg_vec(1, rows * d.nh * nph)
            .iter()
            .map(|x| x * 3.0)
            .collect();
        let thk: Vec<f32> = lcg_vec(2, rows * d.nh * nph)
            .iter()
            .map(|x| x * 3.0)
            .collect();
        let v = lcg_vec(3, rows * d.nh * dv);
        let kappa: Vec<f32> = lcg_vec(4, rows * d.nh)
            .iter()
            .map(|x| 0.5 + 0.4 * x)
            .collect();
        let dout = lcg_vec(5, rows * d.nh * dv);
        let decay = hk_decay_grid(d.nh, nph, 8.0, 2.0 * d.t as f64);
        let to64 = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let o_ref = hk_ref_fwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kappa),
            &to64(&decay),
        );
        let (dthq_ref, dthk_ref, dv_ref, dkap_ref) = hk_ref_bwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kappa),
            &to64(&decay),
            &to64(&dout),
        );

        let p2 = 2 * nph;
        let g = |n: usize| GBuf::zeros(c, n);
        let (gthq, gthk, gv, gkap) = (
            GBuf::from_slice(c, &thq),
            GBuf::from_slice(c, &thk),
            GBuf::from_slice(c, &v),
            GBuf::from_slice(c, &kappa),
        );
        let gpow = GBuf::from_slice(c, &hk_pow_table(&decay, d.nh, nph));
        let (gphq, gphk, gkv, gout) = (
            g(rows * d.nh * p2),
            g(rows * d.nh * p2),
            g(rows * d.nh * dv),
            g(rows * d.nh * dv),
        );
        let nst = d.b * d.nh * (d.t / 64 + 1) * p2 * dv;
        let gstates = g(nst);
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
            states: &gstates,
            out: &gout,
            phase_chunk: None,
            phase_partial: None,
        };
        let gdout = GBuf::from_slice(c, &dout);
        let (gdst, gdkv, gdphq, gdphk) = (
            g(nst),
            g(rows * d.nh * dv),
            g(rows * d.nh * p2),
            g(rows * d.nh * p2),
        );
        let (gdthq, gdthk, gdv, gdkap) = (
            g(rows * d.nh * nph),
            g(rows * d.nh * nph),
            g(rows * d.nh * dv),
            g(rows * d.nh),
        );
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
        let ms = cmd.commit();
        let rel = |got: &[f32], want: &[f64]| -> f64 {
            let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
            got.iter()
                .zip(want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max)
                / scale
        };
        let e_o = rel(&gout.to_vec(), &o_ref);
        let e_q = rel(&gdthq.to_vec(), &dthq_ref);
        let e_k = rel(&gdthk.to_vec(), &dthk_ref);
        let e_v = rel(&gdv.to_vec(), &dv_ref);
        let e_kap = rel(&gdkap.to_vec(), &dkap_ref);
        eprintln!(
            "hk nph={nph} dv={dv}: {ms:.2} ms; rel err out {e_o:.2e} dthq {e_q:.2e} dthk {e_k:.2e} dv {e_v:.2e} dkappa {e_kap:.2e}"
        );
        for (name, e) in [
            ("out", e_o),
            ("dthq", e_q),
            ("dthk", e_k),
            ("dv", e_v),
            ("dkappa", e_kap),
        ] {
            assert!(
                e < 5e-4,
                "hybrid_k {name}: rel err {e:e} (nph={nph} dv={dv})"
            );
        }
    }
}

#[test]
fn phase_delta_metal_matches_f64_oracle_and_repeats() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid, phase_delta_ref_bwd, phase_delta_ref_fwd};
    let Some(c) = ctx() else { return };
    let d = HkDims {
        b: 1,
        t: 65,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let rows = d.b * d.t;
    let thq: Vec<f32> = lcg_vec(101, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 2.0)
        .collect();
    let thk: Vec<f32> = lcg_vec(102, rows * d.nh * d.nph)
        .iter()
        .map(|x| x * 2.0)
        .collect();
    let v = lcg_vec(103, rows * d.nh * d.dv);
    let kappa: Vec<f32> = lcg_vec(104, rows * d.nh)
        .iter()
        .map(|x| 0.25 + 0.5 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(105, rows * d.nh * d.dv);
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 128.0);
    let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let want_o = phase_delta_ref_fwd(
        &d,
        &to64(&thq),
        &to64(&thk),
        &to64(&v),
        &to64(&kappa),
        &to64(&decay),
    );
    let (want_q, want_k, want_v, want_kap) = phase_delta_ref_bwd(
        &d,
        &to64(&thq),
        &to64(&thk),
        &to64(&v),
        &to64(&kappa),
        &to64(&decay),
        &to64(&dout),
    );
    let z = |n| GBuf::zeros(c, n);
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kappa),
    );
    let (gphq, gphk) = (z(rows * d.nh * 2 * d.nph), z(rows * d.nh * 2 * d.nph));
    let gkv = z(rows * d.nh * d.dv);
    let gout = z(rows * d.nh * d.dv);
    let nst = d.b * d.nh * (d.t.div_ceil(64) + 1) * 2 * d.nph * d.dv;
    let states = z(nst);
    let gchunk = z(d.b * d.nh * 65 * 2 * d.nph * d.dv);
    let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + 2 * d.nph));
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
        phase_chunk: Some(&gchunk),
        phase_partial: Some(&gpartial),
    };
    let (gdst, gdkv, gdphq, gdphk) = (
        z(nst),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * 2 * d.nph),
        z(rows * d.nh * 2 * d.nph),
    );
    let (gdthq, gdthk, gdv, gdkap) = (
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.dv),
        z(rows * d.nh),
    );
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
    cmd.phase_delta_forward(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();
    let first_o = gout.to_vec();
    let first_q = gdthq.to_vec();
    let first_k = gdthk.to_vec();
    let first_v = gdv.to_vec();
    let first_kap = gdkap.to_vec();
    let cmd = Cmd::new(c);
    cmd.phase_delta_forward(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();
    assert_eq!(
        gout.to_vec(),
        first_o,
        "Phase-Delta forward is not deterministic"
    );
    assert_eq!(
        gdthq.to_vec(),
        first_q,
        "Phase-Delta q gradient is not deterministic"
    );
    assert_eq!(
        gdthk.to_vec(),
        first_k,
        "Phase-Delta k gradient is not deterministic"
    );
    assert_eq!(
        gdv.to_vec(),
        first_v,
        "Phase-Delta v gradient is not deterministic"
    );
    assert_eq!(
        gdkap.to_vec(),
        first_kap,
        "Phase-Delta gate gradient is not deterministic"
    );
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-9);
        got.iter()
            .zip(want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max)
            / scale
    };
    eprintln!(
        "phase delta rel: o {:.3e}, q {:.3e}, k {:.3e}, v {:.3e}, kap {:.3e}",
        rel(&gout.to_vec(), &want_o),
        rel(&gdthq.to_vec(), &want_q),
        rel(&gdthk.to_vec(), &want_k),
        rel(&gdv.to_vec(), &want_v),
        rel(&gdkap.to_vec(), &want_kap)
    );
    assert!(rel(&gout.to_vec(), &want_o) < 2e-4, "forward parity");
    assert!(rel(&gdthq.to_vec(), &want_q) < 2e-3, "q gradient parity");
    assert!(rel(&gdthk.to_vec(), &want_k) < 2e-3, "k gradient parity");
    assert!(rel(&gdv.to_vec(), &want_v) < 2e-4, "v gradient parity");
    assert!(
        rel(&gdkap.to_vec(), &want_kap) < 2e-3,
        "kappa gradient parity"
    );
}

#[test]
fn phase_delta_chunked_backward_production_geometry_linear_work() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{
        HkDims, hk_decay_grid, phase_delta_ref_bwd, phase_delta_ref_bwd_state, phase_delta_ref_fwd,
        phase_delta_step,
    };
    let Some(c) = ctx() else { return };
    // Production geometry with a non-multiple-of-64 tail exercises chunk
    // entry recovery, reverse carry, and the bounded one-grid dispatch.
    let d = HkDims {
        b: 2,
        t: 129,
        nh: 8,
        nph: 32,
        dv: 128,
    };
    let rows = d.b * d.t;
    let p2 = d.p2();
    let thq = lcg_vec(301, rows * d.nh * d.nph);
    let thk = lcg_vec(302, rows * d.nh * d.nph);
    let v = lcg_vec(303, rows * d.nh * d.dv);
    let kappa: Vec<f32> = lcg_vec(304, rows * d.nh)
        .iter()
        .map(|x| 0.15 + 0.7 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(305, rows * d.nh * d.dv);
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let want_o = phase_delta_ref_fwd(
        &d,
        &to64(&thq),
        &to64(&thk),
        &to64(&v),
        &to64(&kappa),
        &to64(&decay),
    );
    let (want_q, want_k, want_v, want_kap) = phase_delta_ref_bwd(
        &d,
        &to64(&thq),
        &to64(&thk),
        &to64(&v),
        &to64(&kappa),
        &to64(&decay),
        &to64(&dout),
    );
    let z = |n| GBuf::zeros(c, n);
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kappa),
    );
    let (gphq, gphk) = (z(rows * d.nh * p2), z(rows * d.nh * p2));
    let gkv = z(rows * d.nh * d.dv);
    let gout = z(rows * d.nh * d.dv);
    let nchunks = d.t.div_ceil(64);
    let nst = d.b * d.nh * (nchunks + 1) * p2 * d.dv;
    let states = z(nst);
    let gchunk = z(d.b * d.nh * 65 * p2 * d.dv);
    let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + 2 * d.nph));
    // The activation scratch is fixed at 65 states regardless of T; the
    // production boundary table is only (ceil(T/64)+1) states.
    assert_eq!(gchunk.len, d.b * d.nh * 65 * p2 * d.dv);
    assert_eq!(states.len, d.b * d.nh * (nchunks + 1) * p2 * d.dv);
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
        phase_chunk: Some(&gchunk),
        phase_partial: Some(&gpartial),
    };
    let (gdst, gdkv, gdphq, gdphk) = (
        z(nst),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
    );
    let (gdthq, gdthk, gdv, gdkap) = (
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.dv),
        z(rows * d.nh),
    );
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
    cmd.phase_delta_forward(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-9);
        got.iter()
            .zip(want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max)
            / scale
    };
    eprintln!(
        "phase delta production rel: o {:.3e}, q {:.3e}, k {:.3e}, v {:.3e}, kap {:.3e}",
        rel(&gout.to_vec(), &want_o),
        rel(&gdthq.to_vec(), &want_q),
        rel(&gdthk.to_vec(), &want_k),
        rel(&gdv.to_vec(), &want_v),
        rel(&gdkap.to_vec(), &want_kap)
    );
    assert!(
        rel(&gout.to_vec(), &want_o) < 3e-4,
        "production forward parity"
    );
    assert!(
        rel(&gdthq.to_vec(), &want_q) < 4e-3,
        "production q gradient parity"
    );
    assert!(
        rel(&gdthk.to_vec(), &want_k) < 4e-3,
        "production k gradient parity"
    );
    assert!(
        rel(&gdv.to_vec(), &want_v) < 4e-4,
        "production v gradient parity"
    );
    assert!(
        rel(&gdkap.to_vec(), &want_kap) < 4e-3,
        "production kappa gradient parity"
    );
    let first = (
        gout.to_vec(),
        gdthq.to_vec(),
        gdthk.to_vec(),
        gdv.to_vec(),
        gdkap.to_vec(),
    );
    let cmd = Cmd::new(c);
    cmd.phase_delta_forward(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();
    assert_eq!(
        gout.to_vec(),
        first.0,
        "production forward is not deterministic"
    );
    assert_eq!(
        gdthq.to_vec(),
        first.1,
        "production q gradient is not deterministic"
    );
    assert_eq!(
        gdthk.to_vec(),
        first.2,
        "production k gradient is not deterministic"
    );
    assert_eq!(
        gdv.to_vec(),
        first.3,
        "production v gradient is not deterministic"
    );
    assert_eq!(
        gdkap.to_vec(),
        first.4,
        "production gate gradient is not deterministic"
    );

    // A nonzero entry state is supported by the non-reset forward dispatch.
    // Compare that path against the one-token f64 continuation oracle.
    let init = lcg_vec(306, d.nh * p2 * d.dv);
    states.fill(0.0);
    for bh in 0..d.b * d.nh {
        let dst = &mut states.as_mut_slice()[bh * (nchunks + 1) * p2 * d.dv..];
        dst[..p2 * d.dv]
            .copy_from_slice(&init[(bh % d.nh) * p2 * d.dv..(bh % d.nh + 1) * p2 * d.dv]);
    }
    let mut want_nonzero = vec![0.0f64; rows * d.nh * d.dv];
    for b in 0..d.b {
        for h in 0..d.nh {
            let mut s = init[h * p2 * d.dv..(h + 1) * p2 * d.dv]
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<_>>();
            for t in 0..d.t {
                let row = b * d.t + t;
                let q0 = row * d.nh * d.nph + h * d.nph;
                let v0 = row * d.nh * d.dv + h * d.dv;
                let o = phase_delta_step(
                    &mut s,
                    &to64(&thq[q0..q0 + d.nph]),
                    &to64(&thk[q0..q0 + d.nph]),
                    &to64(&v[v0..v0 + d.dv]),
                    kappa[row * d.nh + h] as f64,
                    &to64(&decay[h * p2..(h + 1) * p2]),
                );
                want_nonzero[row * d.nh * d.dv + h * d.dv..row * d.nh * d.dv + (h + 1) * d.dv]
                    .copy_from_slice(&o);
            }
        }
    }
    let cmd = Cmd::new(c);
    cmd.phase_delta_forward(&d, &w);
    cmd.commit();
    assert!(
        rel(&gout.to_vec(), &want_nonzero) < 3e-4,
        "nonzero entry-state forward parity"
    );

    // The reverse must also propagate a nonzero incoming state into boundary
    // zero.  Compare Metal's dS0 against the independent f64 reverse oracle,
    // not merely against the zero-state parameter gradients above.
    let mut init_full = Vec::with_capacity(d.b * d.nh * p2 * d.dv);
    for _ in 0..d.b {
        init_full.extend_from_slice(&init);
    }
    let init64 = init_full.iter().map(|x| *x as f64).collect::<Vec<_>>();
    let (_q0, _k0, _v0, _kap0, want_dstate0) = phase_delta_ref_bwd_state(
        &d,
        &to64(&thq),
        &to64(&thk),
        &to64(&v),
        &to64(&kappa),
        &to64(&decay),
        &to64(&dout),
        &init64,
    );
    let cmd = Cmd::new(c);
    cmd.phase_delta_forward(&d, &w);
    cmd.phase_delta_backward(&d, &w, &gr);
    cmd.commit();
    let got_dstate0 = gdst.to_vec();
    let dstate0_len = d.p2() * d.dv;
    let want_dstate0_first = &want_dstate0[..dstate0_len];
    let got_dstate0_first = &got_dstate0[..dstate0_len];
    let state_scale = want_dstate0_first
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(1e-9);
    let state_rel = got_dstate0_first
        .iter()
        .zip(want_dstate0_first)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max)
        / state_scale;
    eprintln!("phase delta nonzero dS0 rel: {state_rel:.3e}");
    assert!(state_rel < 5e-4, "nonzero entry-state backward parity");
}

#[test]
fn selected_mode_six_unselected_legacy_hk_bit_identity() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid};
    let Some(c) = ctx() else { return };
    // A selected-layer model still allocates shared Phase-Delta scratch, but
    // every unselected hybrid must take the literal legacy path.  Running the
    // legacy kernels with and without those optional buffers is a direct
    // output/gradient bit-identity witness for each of the six layers.
    let d = HkDims {
        b: 1,
        t: 64,
        nh: 2,
        nph: 3,
        dv: 4,
    };
    let rows = d.b * d.t;
    let p2 = d.p2();
    let thq = lcg_vec(1201, rows * d.nh * d.nph);
    let thk = lcg_vec(1202, rows * d.nh * d.nph);
    let v = lcg_vec(1203, rows * d.nh * d.dv);
    let kap: Vec<f32> = lcg_vec(1204, rows * d.nh)
        .iter()
        .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(1205, rows * d.nh * d.dv);
    let pow = GBuf::from_slice(
        c,
        &hk_pow_table(&hk_decay_grid(d.nh, d.nph, 4.0, 32.0), d.nh, d.nph),
    );
    let run = |with_optional_scratch: bool| {
        let gthq = GBuf::from_slice(c, &thq);
        let gthk = GBuf::from_slice(c, &thk);
        let gv = GBuf::from_slice(c, &v);
        let gkap = GBuf::from_slice(c, &kap);
        let phq = GBuf::zeros(c, rows * d.nh * p2);
        let phk = GBuf::zeros(c, rows * d.nh * p2);
        let kv = GBuf::zeros(c, rows * d.nh * d.dv);
        let out = GBuf::zeros(c, rows * d.nh * d.dv);
        let states = GBuf::zeros(c, d.b * d.nh * (d.t / 64 + 1) * p2 * d.dv);
        let chunk = GBuf::zeros(c, d.b * d.nh * 65 * p2 * d.dv);
        let partial = GBuf::zeros(c, d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + 2 * d.nph));
        let w = HkWork {
            thq: &gthq,
            thk: &gthk,
            v: &gv,
            kappa: &gkap,
            pow: &pow,
            pow_off: 0,
            phq: &phq,
            phk: &phk,
            kv: &kv,
            states: &states,
            out: &out,
            phase_chunk: with_optional_scratch.then_some(&chunk),
            phase_partial: with_optional_scratch.then_some(&partial),
        };
        let cmd = Cmd::new(c);
        cmd.hk_forward(&d, &w);
        cmd.commit();
        let gdout = GBuf::from_slice(c, &dout);
        let gdstates = GBuf::zeros(c, states.len);
        let gdkv = GBuf::zeros(c, rows * d.nh * d.dv);
        let gdphq = GBuf::zeros(c, rows * d.nh * p2);
        let gdphk = GBuf::zeros(c, rows * d.nh * p2);
        let gdthq = GBuf::zeros(c, rows * d.nh * d.nph);
        let gdthk = GBuf::zeros(c, rows * d.nh * d.nph);
        let gdv = GBuf::zeros(c, rows * d.nh * d.dv);
        let gdkap = GBuf::zeros(c, rows * d.nh);
        let gr = HkGrads {
            dout: &gdout,
            dstates: &gdstates,
            dkv: &gdkv,
            dphq: &gdphq,
            dphk: &gdphk,
            dthq: &gdthq,
            dthk: &gdthk,
            dv: &gdv,
            dkappa: &gdkap,
        };
        let cmd = Cmd::new(c);
        cmd.hk_backward(&d, &w, &gr, 0.0);
        cmd.commit();
        (
            out.to_vec(),
            states.to_vec(),
            phq.to_vec(),
            phk.to_vec(),
            kv.to_vec(),
            gdthq.to_vec(),
            gdthk.to_vec(),
            gdv.to_vec(),
            gdkap.to_vec(),
        )
    };
    let legacy = run(false);
    let selected_mode_legacy = run(true);
    for layer in [0usize, 1, 2, 4, 5, 6] {
        assert_eq!(
            legacy, selected_mode_legacy,
            "legacy witness drift at layer {layer}"
        );
    }
}

#[test]
fn phase_delta_b8_t1024_kernel_timing_smoke() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid};
    let Some(c) = ctx() else { return };
    // Direct operator smoke only (no model birth/training): production
    // geometry, one Phase-Delta forward+backward command per repetition.
    let d = HkDims {
        b: 8,
        t: 1024,
        nh: 8,
        nph: 32,
        dv: 128,
    };
    let rows = d.b * d.t;
    let p2 = d.p2();
    let thq = lcg_vec(901, rows * d.nh * d.nph);
    let thk = lcg_vec(902, rows * d.nh * d.nph);
    let v = lcg_vec(903, rows * d.nh * d.dv);
    let kap: Vec<f32> = lcg_vec(904, rows * d.nh)
        .iter()
        .map(|x| 0.25 + 0.5 * (x + 1.0) / 2.0)
        .collect();
    let dout = lcg_vec(905, rows * d.nh * d.dv);
    let decay = hk_decay_grid(d.nh, d.nph, 8.0, 2048.0);
    let z = |n: usize| GBuf::zeros(c, n);
    let (gthq, gthk, gv, gkap) = (
        GBuf::from_slice(c, &thq),
        GBuf::from_slice(c, &thk),
        GBuf::from_slice(c, &v),
        GBuf::from_slice(c, &kap),
    );
    let (gphq, gphk, gkv, gout) = (
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * d.dv),
    );
    let nchunks = d.t.div_ceil(64);
    let states = z(d.b * d.nh * (nchunks + 1) * p2 * d.dv);
    let gchunk = z(d.b * d.nh * 65 * p2 * d.dv);
    let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + 2 * d.nph));
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
        phase_chunk: Some(&gchunk),
        phase_partial: Some(&gpartial),
    };
    let (gdst, gdkv, gdphq, gdphk) = (
        z(states.len),
        z(rows * d.nh * d.dv),
        z(rows * d.nh * p2),
        z(rows * d.nh * p2),
    );
    let (gdthq, gdthk, gdv, gdkap) = (
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.nph),
        z(rows * d.nh * d.dv),
        z(rows * d.nh),
    );
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
    let mut timings = Vec::new();
    let mut forward_timings = Vec::new();
    let mut backward_timings = Vec::new();
    let mut fold_timings = Vec::new();
    let mut combined_timings = Vec::new();
    // One warmup plus three measured direct B8/T1024 samples.
    for _ in 0..4 {
        let cmd = Cmd::new(c);
        cmd.phase_delta_forward_reset(&d, &w);
        let forward = cmd.commit();
        forward_timings.push(forward);
        let cmd = Cmd::new(c);
        cmd.phase_delta_backward_blocks(&d, &w, &gr);
        let backward = cmd.commit();
        backward_timings.push(backward);
        let cmd = Cmd::new(c);
        cmd.phase_delta_fold(&d, &w, &gr);
        let fold = cmd.commit();
        fold_timings.push(fold);
        timings.push(forward + backward + fold);
        let cmd = Cmd::new(c);
        cmd.phase_delta_forward_reset(&d, &w);
        cmd.phase_delta_backward(&d, &w, &gr);
        combined_timings.push(cmd.commit());
    }
    assert!(gout.to_vec().iter().all(|x| x.is_finite()));
    eprintln!(
        "phase_delta B=8 T=1024 dispatch_groups={} lanes={} scratch={} bytes partial_scratch={} bytes timings_ms={timings:?} combined_ms={combined_timings:?} forward_ms={forward_timings:?} backward_blocks_ms={backward_timings:?} fold_ms={fold_timings:?}",
        d.b * d.nh * d.dv.div_ceil(32),
        d.dv,
        gchunk.len * std::mem::size_of::<f32>(),
        gpartial.len * std::mem::size_of::<f32>(),
    );
}

#[test]
fn phase_delta_metal_boundary_lengths_parity() {
    use cortiq_embryo::metal::{HkGrads, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid, phase_delta_ref_bwd, phase_delta_ref_fwd};
    let Some(c) = ctx() else { return };
    // Exact tail/boundary lengths that historically expose chunk carry bugs.
    for &t in &[1usize, 63, 64, 65, 129] {
        let d = HkDims {
            b: 1,
            t,
            nh: 2,
            nph: 3,
            dv: 4,
        };
        let rows = d.b * d.t;
        let p2 = d.p2();
        let thq = lcg_vec(920 + t as u64, rows * d.nh * d.nph);
        let thk = lcg_vec(930 + t as u64, rows * d.nh * d.nph);
        let v = lcg_vec(940 + t as u64, rows * d.nh * d.dv);
        let kap: Vec<f32> = lcg_vec(950 + t as u64, rows * d.nh)
            .iter()
            .map(|x| 0.2 + 0.6 * (x + 1.0) / 2.0)
            .collect();
        let dout = lcg_vec(960 + t as u64, rows * d.nh * d.dv);
        let decay = hk_decay_grid(d.nh, d.nph, 8.0, 256.0);
        let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let want_o = phase_delta_ref_fwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kap),
            &to64(&decay),
        );
        let (want_q, want_k, want_v, want_kap) = phase_delta_ref_bwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kap),
            &to64(&decay),
            &to64(&dout),
        );
        let z = |n: usize| GBuf::zeros(c, n);
        let (gthq, gthk, gv, gkap) = (
            GBuf::from_slice(c, &thq),
            GBuf::from_slice(c, &thk),
            GBuf::from_slice(c, &v),
            GBuf::from_slice(c, &kap),
        );
        let (gphq, gphk, gkv, gout) = (
            z(rows * d.nh * p2),
            z(rows * d.nh * p2),
            z(rows * d.nh * d.dv),
            z(rows * d.nh * d.dv),
        );
        let nchunks = d.t.div_ceil(64);
        let states = z(d.b * d.nh * (nchunks + 1) * p2 * d.dv);
        let gchunk = z(d.b * d.nh * 65 * p2 * d.dv);
        let gpartial = z(d.b * d.nh * d.dv.div_ceil(32) * d.t * (1 + 2 * d.nph));
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
            phase_chunk: Some(&gchunk),
            phase_partial: Some(&gpartial),
        };
        let (gdst, gdkv, gdphq, gdphk) = (
            z(states.len),
            z(rows * d.nh * d.dv),
            z(rows * d.nh * p2),
            z(rows * d.nh * p2),
        );
        let (gdthq, gdthk, gdv, gdkap) = (
            z(rows * d.nh * d.nph),
            z(rows * d.nh * d.nph),
            z(rows * d.nh * d.dv),
            z(rows * d.nh),
        );
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
        cmd.phase_delta_forward(&d, &w);
        cmd.phase_delta_backward(&d, &w, &gr);
        cmd.commit();
        let rel = |got: &[f32], want: &[f64]| -> f64 {
            let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-9);
            got.iter()
                .zip(want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max)
                / scale
        };
        assert!(rel(&gout.to_vec(), &want_o) < 5e-4, "T={t} forward");
        assert!(rel(&gdthq.to_vec(), &want_q) < 5e-3, "T={t} q");
        assert!(rel(&gdthk.to_vec(), &want_k) < 5e-3, "T={t} k");
        assert!(rel(&gdv.to_vec(), &want_v) < 5e-4, "T={t} v");
        assert!(rel(&gdkap.to_vec(), &want_kap) < 5e-3, "T={t} gate");
        eprintln!("phase delta boundary T={t}: parity ok; chunks={nchunks}");
    }
}

#[test]
fn gemm_batched_with_causal_mask() {
    use cortiq_embryo::metal::GemmBatch;
    let Some(c) = ctx() else { return };
    // 2×3×4 batches of [64×64]·[64×128]ᵀ... use NT: A [64,32] (lda 32), B [64,32] → C [64,64], causal.
    let (nb, nh, nc) = (2usize, 3usize, 4usize);
    let (m, n, k) = (64usize, 64usize, 32usize);
    let sa = [nh * nc * m * k, nc * m * k, m * k];
    let sb = [nh * nc * n * k, nc * n * k, n * k];
    let sc = [nh * nc * m * n, nc * m * n, m * n];
    let a = lcg_vec(1, nb * nh * nc * m * k);
    let b = lcg_vec(2, nb * nh * nc * n * k);
    let ga = GBuf::from_slice(c, &a);
    let gb = GBuf::from_slice(c, &b);
    let gc = GBuf::zeros(c, nb * nh * nc * m * n);
    let cmd = Cmd::new(c);
    cmd.gemm_ex(
        Op::N,
        Op::T,
        m,
        n,
        k,
        1.0,
        &ga,
        0,
        k,
        &gb,
        0,
        k,
        0.0,
        &gc,
        0,
        n,
        &GemmBatch {
            nb,
            nh,
            nc,
            sa,
            sb,
            sc,
        },
        true,
    );
    cmd.commit();
    let got = gc.to_vec();
    for bi in 0..nb {
        for hi in 0..nh {
            for ci in 0..nc {
                let ao = bi * sa[0] + hi * sa[1] + ci * sa[2];
                let bo = bi * sb[0] + hi * sb[1] + ci * sb[2];
                let co = bi * sc[0] + hi * sc[1] + ci * sc[2];
                let mut want = vec![0.0f32; m * n];
                gemm_ref(
                    false,
                    true,
                    m,
                    n,
                    k,
                    1.0,
                    &a[ao..ao + m * k],
                    k,
                    &b[bo..bo + n * k],
                    k,
                    0.0,
                    &mut want,
                    n,
                );
                for i in 0..m {
                    for j in 0..n {
                        if j > i {
                            want[i * n + j] = 0.0;
                        }
                    }
                }
                let d = max_abs_diff(&got[co..co + m * n], &want);
                assert!(d < 1e-4, "batch ({bi},{hi},{ci}) max|Δ|={d}");
            }
        }
    }
}

#[test]
fn hybrid_k_gemm_formulation_matches_oracle_and_simt() {
    use cortiq_embryo::metal::{HkGrads, HkScratch, HkWork, hk_pow_table};
    use cortiq_embryo::ops::{HkDims, hk_decay_grid, hk_ref_bwd, hk_ref_fwd};
    let Some(c) = ctx() else { return };
    for &(b, t, nh, nph, dv, hmin) in &[
        (2usize, 192usize, 2usize, 32usize, 128usize, 8.0f64),
        (1, 256, 3, 32, 64, 1.5),
    ] {
        let d = HkDims { b, t, nh, nph, dv };
        let rows = b * t;
        let p2 = 2 * nph;
        let thq: Vec<f32> = lcg_vec(1, rows * nh * nph)
            .iter()
            .map(|x| x * 3.0)
            .collect();
        let thk: Vec<f32> = lcg_vec(2, rows * nh * nph)
            .iter()
            .map(|x| x * 3.0)
            .collect();
        let v = lcg_vec(3, rows * nh * dv);
        let kappa: Vec<f32> = lcg_vec(4, rows * nh)
            .iter()
            .map(|x| 0.5 + 0.4 * x)
            .collect();
        let dout = lcg_vec(5, rows * nh * dv);
        let decay = hk_decay_grid(nh, nph, hmin, 2.0 * t as f64);
        let to64 = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let o_ref = hk_ref_fwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kappa),
            &to64(&decay),
        );
        let (dthq_ref, dthk_ref, dv_ref, dkap_ref) = hk_ref_bwd(
            &d,
            &to64(&thq),
            &to64(&thk),
            &to64(&v),
            &to64(&kappa),
            &to64(&decay),
            &to64(&dout),
        );
        let g = |n: usize| GBuf::zeros(c, n);
        let (gthq, gthk, gv, gkap) = (
            GBuf::from_slice(c, &thq),
            GBuf::from_slice(c, &thk),
            GBuf::from_slice(c, &v),
            GBuf::from_slice(c, &kappa),
        );
        let gpow = GBuf::from_slice(c, &hk_pow_table(&decay, nh, nph));
        let (gphq, gphk, gkv, gout) = (
            g(rows * nh * p2),
            g(rows * nh * p2),
            g(rows * nh * dv),
            g(rows * nh * dv),
        );
        let nst = b * nh * (t / 64 + 1) * p2 * dv;
        let gstates = g(nst);
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
            states: &gstates,
            out: &gout,
            phase_chunk: None,
            phase_partial: None,
        };
        let gdout = GBuf::from_slice(c, &dout);
        let (gdst, gdkv, gdphq, gdphk) = (
            g(nst),
            g(rows * nh * dv),
            g(rows * nh * p2),
            g(rows * nh * p2),
        );
        let (gdthq, gdthk, gdv, gdkap) = (
            g(rows * nh * nph),
            g(rows * nh * nph),
            g(rows * nh * dv),
            g(rows * nh),
        );
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
        let cl = HkScratch::chunk_len(&d);
        let al = HkScratch::a_len(&d);
        let (qt, kt, qp, kh, dqt, dkt, dqi, dki, a) = (
            g(cl),
            g(cl),
            g(cl),
            g(cl),
            g(cl),
            g(cl),
            g(cl),
            g(cl),
            g(al),
        );
        let sc = HkScratch {
            qt: &qt,
            kt: &kt,
            qp: &qp,
            kh: &kh,
            dqt: &dqt,
            dkt: &dkt,
            dqi: &dqi,
            dki: &dki,
            a: &a,
        };
        let cmd = Cmd::new(c);
        cmd.hk_forward_gemm(&d, &w, &sc);
        cmd.hk_backward_gemm(&d, &w, &gr, &sc, 0.0);
        let ms = cmd.commit();
        let rel = |got: &[f32], want: &[f64]| -> f64 {
            let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
            got.iter()
                .zip(want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max)
                / scale
        };
        let e_o = rel(&gout.to_vec(), &o_ref);
        let e_q = rel(&gdthq.to_vec(), &dthq_ref);
        let e_k = rel(&gdthk.to_vec(), &dthk_ref);
        let e_v = rel(&gdv.to_vec(), &dv_ref);
        let e_kap = rel(&gdkap.to_vec(), &dkap_ref);
        eprintln!(
            "hk-gemm b={b} t={t} nh={nh} nph={nph} dv={dv} hmin={hmin}: {ms:.2} ms; rel err out {e_o:.2e} dthq {e_q:.2e} dthk {e_k:.2e} dv {e_v:.2e} dkappa {e_kap:.2e}"
        );
        for (name, e) in [
            ("out", e_o),
            ("dthq", e_q),
            ("dthk", e_k),
            ("dv", e_v),
            ("dkappa", e_kap),
        ] {
            assert!(e < 5e-4, "hybrid_k gemm {name}: rel err {e:e}");
        }
    }
}

#[test]
fn top_eigenvectors_recover_a_planted_subspace() {
    use cortiq_embryo::model::top_eigenvectors;
    // A = Σ_i λ_i v_i v_iᵀ with 3 strong planted directions in R^32
    let n = 32;
    let dirs = [3usize, 11, 27];
    let mut a = vec![0.0f32; n * n];
    for (i, &d) in dirs.iter().enumerate() {
        let lam = 10.0 - i as f32;
        a[d * n + d] += lam;
    }
    // small noise on the rest of the diagonal
    for i in 0..n {
        if !dirs.contains(&i) {
            a[i * n + i] += 0.01 * (i as f32);
        }
    }
    let u = top_eigenvectors(&a, n, 3, 30, 1);
    // each recovered row must lie in the planted span (projection energy > 0.999)
    for r in 0..3 {
        let row = &u[r * n..(r + 1) * n];
        let energy: f32 = dirs.iter().map(|&d| row[d] * row[d]).sum();
        assert!(
            energy > 0.999,
            "row {r}: energy in the planted span {energy}"
        );
    }
}
