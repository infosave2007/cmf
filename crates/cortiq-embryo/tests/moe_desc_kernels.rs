//! The descriptor kernels of the trainer against the host formula, on the
//! backend at hand (Metal on macOS, Vulkan with `--features vulkan`):
//!  - `moe_update`: the μ EMA divides the capped sum of `moe_stats`
//!    (only `min(count, cap)` slots exist) by that same `min(count, cap)`,
//!    never by the full count — an over-capacity expert's μ was pulled
//!    toward the origin by cap/count every step (T1);
//!  - `route`: the reconstruction error starts at exactly 0.0, so a row
//!    equal to μ reports `res == 0.0` bit-for-bit like the host and Metal
//!    (the WGSL port initialised it from the op-code word, F6).
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::growth::resonance_scores;
use cortiq_embryo::metal::{Cmd, GBuf, RouteDims, ctx};

#[test]
fn moe_update_divides_the_capped_sum_by_the_capped_count() {
    let Some(c) = ctx() else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let r = RouteDims {
        rows: 8,
        h: 4,
        e: 2,
        k: 0,
        cap: 4,
    };
    let (e, h, cap) = (r.e, r.h, r.cap);
    // expert 0 received 6 tokens for 4 slots; expert 1 received 2
    let count = [6u32, 2];
    let mut hg = vec![0.0f32; e * cap * h];
    for ee in 0..e {
        for s in 0..cap {
            for j in 0..h {
                hg[(ee * cap + s) * h + j] = 1.0 + ee as f32 * 10.0 + s as f32 + 0.25 * j as f32;
            }
        }
    }
    let mu0: Vec<f32> = (0..e * h).map(|i| -2.0 + 0.5 * i as f32).collect();
    let bias0 = [0.125f32, -0.375];
    let res = vec![1.0f32; r.rows];
    let alpha = 0.5f32;
    let gh = GBuf::from_slice(c, &hg);
    let gcount = GBuf::from_u32(c, &count);
    let gsums = GBuf::zeros(c, e * h);
    let gmu = GBuf::from_slice(c, &mu0);
    let gbias = GBuf::from_slice(c, &bias0);
    let gres = GBuf::from_slice(c, &res);
    let cmd = Cmd::new(c);
    cmd.moe_stats(&r, &gh, &gcount, 0, &gsums, 0);
    cmd.moe_update(&r, &gmu, 0, &gbias, 0, &gsums, 0, &gcount, 0, &gres, alpha, 0.0, 0, usize::MAX);
    cmd.commit();
    let sums = gsums.to_vec();
    let mu = gmu.to_vec();
    for ee in 0..e {
        let n_slots = (count[ee] as usize).min(cap);
        for j in 0..h {
            let want_sum: f32 = (0..n_slots).map(|s| hg[(ee * cap + s) * h + j]).sum();
            assert!((sums[ee * h + j] - want_sum).abs() < 1e-5, "sums[{ee}][{j}] = {} want {want_sum}", sums[ee * h + j]);
            let mean_capped = want_sum / n_slots as f32;
            let want = (1.0 - alpha) * mu0[ee * h + j] + alpha * mean_capped;
            let wrong = (1.0 - alpha) * mu0[ee * h + j] + alpha * want_sum / count[ee] as f32;
            let got = mu[ee * h + j];
            assert!((got - want).abs() < 1e-5, "mu[{ee}][{j}] = {got}, want {want} (capped mean)");
            if count[ee] as usize > cap {
                assert!((want - wrong).abs() > 1e-3, "fixture: the two formulas must differ");
                assert!((got - wrong).abs() > 1e-4, "mu[{ee}][{j}] = {got}: divided the capped sum by the full count {}", count[ee]);
            }
        }
    }
    // eta = 0: the balancing bias is untouched
    assert_eq!(gbias.to_vec(), bias0.to_vec());
}

#[test]
fn route_error_of_a_row_at_mu_is_exactly_zero() {
    let Some(c) = ctx() else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let r = RouteDims {
        rows: 2,
        h: 4,
        e: 2,
        k: 1,
        cap: 8,
    };
    let mu = vec![0.3f32, -1.2, 2.5, 0.7, 1.0, 1.0, -1.0, 0.0];
    let u = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let bias = vec![0.0f32, -0.5];
    // row 0 == μ_0 exactly; row 1 elsewhere
    let x = vec![0.3f32, -1.2, 2.5, 0.7, 0.1, 0.2, 0.3, 0.4];
    let gx = GBuf::from_slice(c, &x);
    let gmu = GBuf::from_slice(c, &mu);
    let gu = GBuf::from_slice(c, &u);
    let gbias = GBuf::from_slice(c, &bias);
    let gassign = GBuf::from_u32(c, &[u32::MAX; 2]);
    let gres = GBuf::from_slice(c, &[-1.0f32; 2]);
    let cmd = Cmd::new(c);
    cmd.route(&r, &gx, &gmu, 0, &gu, 0, &gbias, 0, &gassign, &gres);
    cmd.commit();
    let assign = unsafe { gassign.as_u32_slice().to_vec() };
    let res = gres.to_vec();
    let mut scores = vec![0.0f32; 2];
    let mut errs = vec![0.0f32; 2];
    resonance_scores(&x[..4], &mu, &u, 1, &bias, None, &mut scores, &mut errs);
    assert_eq!(assign[0], 0);
    assert_eq!(errs[0].to_bits(), 0.0f32.to_bits(), "host");
    assert_eq!(
        res[0].to_bits(),
        errs[0].to_bits(),
        "row at μ: the kernel's error {:e} (bits {:#x}) is not the host's exact 0.0",
        res[0],
        res[0].to_bits()
    );
    resonance_scores(&x[4..], &mu, &u, 1, &bias, None, &mut scores, &mut errs);
    let w = (0..2).max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap()).unwrap();
    assert_eq!(assign[1] as usize, w);
    assert!((res[1] - errs[w]).abs() < 1e-5, "row 1: {} vs host {}", res[1], errs[w]);
}
