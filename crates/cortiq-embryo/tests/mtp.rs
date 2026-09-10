//! MTP heads on the frozen trunk: gradcheck of the head parameters (the
//! projection and its norm) by finite differences of the head loss, and the
//! trunk arena stays byte-identical.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, gauss_vec, init_params};
use cortiq_embryo::mtp::{MtpState, mtp_step};
use cortiq_embryo::train::Shard;

#[test]
fn mtp_head_gradients_match_finite_differences() {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 2;
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 33);
    let mut text = Vec::new();
    for f in ["../../README.md", "../../docs/CMF_V2_SPEC.md"] {
        if let Ok(b) = std::fs::read(f) {
            text.extend_from_slice(&b);
        }
    }
    let corpus = Shard::from_bytes(&text);
    let (b, t) = (2usize, 64usize);
    let m = b * t;
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &params).unwrap();
    let tk: Vec<u32> = corpus.tokens[..m].iter().map(|&x| x as u32).collect();
    let tg: Vec<u32> = corpus.tokens[1..m + 1].iter().map(|&x| x as u32).collect();
    gpu.train_step(&tk, &tg, 1e-3, 0.0, 1.0); // seeds descriptors
    let trunk = gpu.params_host();
    gpu.desc_updates.set(false);
    let mut st = MtpState::new(&gpu, 1, 7);
    let toks: Vec<u32> = corpus.tokens[1000..1000 + m]
        .iter()
        .map(|&x| x as u32)
        .collect();
    let valid = (t - 2) * b; // head 1 (t+2): positions with a target
    // analytic gradient (lr 0 → params untouched, grads left in gp/gw)
    let l = mtp_step(&mut gpu, &mut st, &toks, 0.0, true)[0] * valid as f32 / m as f32;
    let gp = st.gp.to_vec();
    let gw = st.gw.to_vec();
    let p0 = st.p.to_vec();
    let w0 = st.w.to_vec();
    let h = cfg.hidden;
    for (name, is_p, base, g) in [("proj", true, &p0, &gp), ("norm", false, &w0, &gw)] {
        let gn = g
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        for dir in 0..2 {
            let delta: Vec<f64> = if dir == 0 {
                g.iter().map(|x| *x as f64 / gn.max(1e-30)).collect()
            } else {
                let r = gauss_vec(11 + dir, g.len());
                let rn = r
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                r.iter().map(|x| *x as f64 / rn).collect()
            };
            let an: f64 = g.iter().zip(&delta).map(|(a, d)| *a as f64 * d).sum();
            let eps = (2e-3 / gn.max(1e-6)).clamp(1e-3, 0.1);
            let mut pp = base.clone();
            for i in 0..pp.len() {
                pp[i] = (base[i] as f64 + eps * delta[i]) as f32;
            }
            if is_p {
                st.p.write_from(&pp)
            } else {
                st.w.write_from(&pp)
            }
            let lp =
                mtp_step(&mut gpu, &mut st, &toks, 0.0, false)[0] as f64 * valid as f64 / m as f64;
            for i in 0..pp.len() {
                pp[i] = (base[i] as f64 - eps * delta[i]) as f32;
            }
            if is_p {
                st.p.write_from(&pp)
            } else {
                st.w.write_from(&pp)
            }
            let lm =
                mtp_step(&mut gpu, &mut st, &toks, 0.0, false)[0] as f64 * valid as f64 / m as f64;
            if is_p {
                st.p.write_from(base)
            } else {
                st.w.write_from(base)
            }
            let fd = (lp - lm) / (2.0 * eps);
            let err = (fd - an).abs() / gn.max(5e-4);
            eprintln!(
                "mtp {name} dir {dir}: |g| {gn:.3e} fd {fd:+.4e} an {an:+.4e} err/|g| {err:.2e} (loss {l:.4}, h {h})"
            );
            assert!(err < 3e-2, "mtp {name}: fd {fd} vs analytic {an}");
        }
    }
    assert_eq!(gpu.params_host(), trunk, "the trunk must not move");
}
