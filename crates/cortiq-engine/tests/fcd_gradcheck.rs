//! Gradient checks for the FCD polish operators (fcd_ops).
//!
//! Every hand-rolled backward is verified against CENTRAL finite
//! differences of its own forward, in f64 (h = 1e-5 → truncation
//! O(h²) ≈ 1e-10, roundoff ≈ 1e-10; the 1e-4 assertions leave two
//! orders of headroom under the 1e-3 requirement). The measured error
//! is printed per op — run with `--nocapture` to collect the table.
//!
//! Scheme: scalar loss L(inputs) = Σ out ⊙ R for a fixed pseudo-random
//! R; the analytic gradient is backward(dout = R); FD perturbs each
//! input element by ±h.

use cortiq_engine::fcd_ops as ops;

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn synth(n: usize, salt: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
            let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// L2 relative error between analytic and FD gradients.
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let mut d2 = 0f64;
    let mut n2 = 0f64;
    for (x, y) in a.iter().zip(b) {
        d2 += (x - y) * (x - y);
        n2 += x * x + y * y;
    }
    (d2.sqrt()) / (n2.sqrt() + 1e-30)
}

const H: f64 = 1e-5;
const TOL: f64 = 1e-4;

/// Central finite differences of `f` w.r.t. `x`, in place.
fn fd_grad(x: &mut [f64], mut f: impl FnMut(&[f64]) -> f64) -> Vec<f64> {
    let mut g = vec![0f64; x.len()];
    for i in 0..x.len() {
        let x0 = x[i];
        x[i] = x0 + H;
        let lp = f(x);
        x[i] = x0 - H;
        let lm = f(x);
        x[i] = x0;
        g[i] = (lp - lm) / (2.0 * H);
    }
    g
}

#[test]
fn gradcheck_matmul_nt_dx_dw() {
    let (n, k, m) = (3usize, 5usize, 4usize);
    let mut x = synth(n * k, 1);
    let mut w = synth(m * k, 2);
    let r = synth(n * m, 3);
    let loss = |x: &[f64], w: &[f64]| -> f64 {
        let mut y = vec![0f64; n * m];
        ops::matmul_nt(x, w, &mut y, n, k, m);
        y.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    // analytic
    let mut dx = vec![0f64; n * k];
    ops::matmul_nt_dx(&r, &w, &mut dx, n, k, m);
    let mut dw = vec![0f64; m * k];
    ops::matmul_nt_dw(&r, &x, &mut dw, n, k, m);
    // fd
    let wc = w.clone();
    let fx = fd_grad(&mut x, |x| loss(x, &wc));
    let xc = x.clone();
    let fw = fd_grad(&mut w, |w| loss(&xc, w));
    let (ex, ew) = (rel_err(&dx, &fx), rel_err(&dw, &fw));
    println!("gradcheck matmul_nt: dX rel err {ex:.2e}, dW rel err {ew:.2e}");
    assert!(ex < TOL && ew < TOL);
}

#[test]
fn gradcheck_silu() {
    let mut x = synth(64, 4);
    for v in x.iter_mut() {
        *v *= 6.0; // cover both tails
    }
    let r = synth(64, 5);
    let loss = |x: &[f64]| -> f64 { x.iter().zip(&r).map(|(&v, b)| ops::silu(v) * b).sum() };
    let dx: Vec<f64> = x.iter().zip(&r).map(|(&v, b)| ops::silu_bwd(v) * b).collect();
    let fx = fd_grad(&mut x, loss);
    let e = rel_err(&dx, &fx);
    println!("gradcheck silu: rel err {e:.2e}");
    assert!(e < TOL);
}

#[test]
fn gradcheck_swiglu_mul() {
    // c = silu(g) ⊙ u — the SwiGLU joint, both branches.
    let n = 32usize;
    let mut g = synth(n, 6);
    let mut u = synth(n, 7);
    let r = synth(n, 8);
    let loss = |g: &[f64], u: &[f64]| -> f64 {
        (0..n).map(|i| ops::silu(g[i]) * u[i] * r[i]).sum()
    };
    let dg: Vec<f64> = (0..n).map(|i| r[i] * u[i] * ops::silu_bwd(g[i])).collect();
    let du: Vec<f64> = (0..n).map(|i| r[i] * ops::silu(g[i])).collect();
    let uc = u.clone();
    let fg = fd_grad(&mut g, |g| loss(g, &uc));
    let gc = g.clone();
    let fu = fd_grad(&mut u, |u| loss(&gc, u));
    let (eg, eu) = (rel_err(&dg, &fg), rel_err(&du, &fu));
    println!("gradcheck swiglu mul: dGate rel err {eg:.2e}, dUp rel err {eu:.2e}");
    assert!(eg < TOL && eu < TOL);
}

fn check_rmsnorm(gemma: bool) -> (f64, f64) {
    let (n, d) = (3usize, 7usize);
    let eps = 1e-6;
    let mut x = synth(n * d, 9);
    let mut w = synth(d, 10);
    let r = synth(n * d, 11);
    let loss = |x: &[f64], w: &[f64]| -> f64 {
        let mut y = vec![0f64; n * d];
        let mut inv = vec![0f64; n];
        ops::rmsnorm_fwd(x, w, eps, gemma, &mut y, &mut inv);
        y.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut y = vec![0f64; n * d];
    let mut inv = vec![0f64; n];
    ops::rmsnorm_fwd(&x, &w, eps, gemma, &mut y, &mut inv);
    let mut dx = vec![0f64; n * d];
    let mut dw = vec![0f64; d];
    ops::rmsnorm_bwd(&x, &w, &inv, &r, gemma, &mut dx, Some(&mut dw));
    let wc = w.clone();
    let fx = fd_grad(&mut x, |x| loss(x, &wc));
    let xc = x.clone();
    let fw = fd_grad(&mut w, |w| loss(&xc, w));
    (rel_err(&dx, &fx), rel_err(&dw, &fw))
}

#[test]
fn gradcheck_rmsnorm_qwen_and_gemma() {
    let (ex, ew) = check_rmsnorm(false);
    println!("gradcheck rmsnorm x̂·w (qwen): dX rel err {ex:.2e}, dW rel err {ew:.2e}");
    assert!(ex < TOL && ew < TOL);
    let (ex, ew) = check_rmsnorm(true);
    println!("gradcheck rmsnorm x̂·(1+w) (gemma): dX rel err {ex:.2e}, dW rel err {ew:.2e}");
    assert!(ex < TOL && ew < TOL);
}

#[test]
fn gradcheck_rope_full_and_partial() {
    for (hd, rd, tag) in [(8usize, 8usize, "full"), (8, 4, "partial")] {
        let inv_freq: Vec<f64> = (0..rd / 2)
            .map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / rd as f64))
            .collect();
        let pos = 7usize;
        let mut x = synth(hd, 12);
        let r = synth(hd, 13);
        let loss = |x: &[f64]| -> f64 {
            let mut y = x.to_vec();
            ops::rope_fwd(&mut y[..rd], pos, &inv_freq);
            y.iter().zip(&r).map(|(a, b)| a * b).sum()
        };
        let mut dx = r.clone();
        ops::rope_bwd(&mut dx[..rd], pos, &inv_freq);
        let fx = fd_grad(&mut x, loss);
        let e = rel_err(&dx, &fx);
        println!("gradcheck rope ({tag} rotary): rel err {e:.2e}");
        assert!(e < TOL);
    }
}

#[test]
fn gradcheck_seg_means() {
    let (t, d, m) = (11usize, 3usize, 4usize);
    let mut x = synth(t * d, 14);
    let r = synth(m * d, 15);
    let loss = |x: &[f64]| -> f64 {
        let mut l = vec![0f64; m * d];
        ops::seg_means(x, t, d, m, &mut l);
        l.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dx = vec![0f64; t * d];
    ops::seg_means_bwd(&r, t, d, m, &mut dx);
    let fx = fd_grad(&mut x, loss);
    let e = rel_err(&dx, &fx);
    println!("gradcheck seg_means: rel err {e:.2e}");
    assert!(e < TOL);
}

#[test]
fn gradcheck_exact_attention_head() {
    let (t, d, dv) = (7usize, 4usize, 3usize);
    let mut q = synth(t * d, 16);
    let mut k = synth(t * d, 17);
    let mut v = synth(t * dv, 18);
    // Spread the logits so softmax is non-degenerate.
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 2.0;
    }
    let r = synth(t * dv, 19);
    let loss = |q: &[f64], k: &[f64], v: &[f64]| -> f64 {
        let mut o = vec![0f64; t * dv];
        ops::attn_head_fwd(q, k, v, t, d, dv, &mut o);
        o.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dq = vec![0f64; t * d];
    let mut dk = vec![0f64; t * d];
    let mut dv_ = vec![0f64; t * dv];
    ops::attn_head_bwd(&q, &k, &v, &r, t, d, dv, &mut dq, &mut dk, &mut dv_);
    let (kc, vc) = (k.clone(), v.clone());
    let fq = fd_grad(&mut q, |q| loss(q, &kc, &vc));
    let qc = q.clone();
    let fk = fd_grad(&mut k, |k| loss(&qc, k, &vc));
    let kc = k.clone();
    let fv = fd_grad(&mut v, |v| loss(&qc, &kc, v));
    let (eq, ek, ev) = (rel_err(&dq, &fq), rel_err(&dk, &fk), rel_err(&dv_, &fv));
    println!("gradcheck exact attention: dQ {eq:.2e}, dK {ek:.2e}, dV {ev:.2e}");
    assert!(eq < TOL && ek < TOL && ev < TOL);
}

#[test]
fn gradcheck_nystrom_joint_head_frozen_mu() {
    // Skeleton actually active: t > w + sink + 8, many evicted keys.
    // The DEFINING convention of the certified recipe is that M is
    // CONSTANT in backward — so the gradcheck freezes M in the FD
    // functional too (via the _mu hooks) and demands tight agreement.
    // This validates every OTHER far-field chain: Fu/E exponentials,
    // the two skeleton matmuls, the clamp mask, the joint denominator,
    // the c-shift invariance, and the landmark segment-mean scatter.
    let (t, d, dv) = (40usize, 6usize, 5usize);
    let cfg = ops::NysCfg { m: 4, w: 8, sink: 2 };
    let mut q = synth(t * d, 20);
    let mut k = synth(t * d, 21);
    let mut v = synth(t * dv, 22);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 2.0;
    }
    let r = synth(t * dv, 23);
    let mu = ops::nystrom_mu_for_test(&q, &k, t, d, &cfg);
    let loss = |q: &[f64], k: &[f64], v: &[f64]| -> f64 {
        let mut o = vec![0f64; t * dv];
        ops::nystrom_head_fwd_mu(q, k, v, t, d, dv, &cfg, Some(&mu), &mut o);
        o.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dq = vec![0f64; t * d];
    let mut dk = vec![0f64; t * d];
    let mut dv_ = vec![0f64; t * dv];
    ops::nystrom_head_bwd_mu(
        &q, &k, &v, &r, t, d, dv, &cfg, Some(&mu), &mut dq, &mut dk, &mut dv_,
    );
    let (kc, vc) = (k.clone(), v.clone());
    let fq = fd_grad(&mut q, |q| loss(q, &kc, &vc));
    let qc = q.clone();
    let fk = fd_grad(&mut k, |k| loss(&qc, k, &vc));
    let kc = k.clone();
    let fv = fd_grad(&mut v, |v| loss(&qc, &kc, v));
    let (eq, ek, ev) = (rel_err(&dq, &fq), rel_err(&dk, &fk), rel_err(&dv_, &fv));
    println!("gradcheck nystrom joint (frozen M): dQ {eq:.2e}, dK {ek:.2e}, dV {ev:.2e}");
    assert!(eq < TOL && ek < TOL && ev < TOL, "dQ {eq:.2e} dK {ek:.2e} dV {ev:.2e}");
}

#[test]
fn gradcheck_nystrom_joint_head_full_functional() {
    // Same configuration WITHOUT freezing M in the FD: the analytic
    // backward deliberately omits the pinv chain (torch reference runs
    // it under no_grad), so FD sees extra gradient through M. This
    // check documents the size of that intentional gap — it must be
    // moderate (the convention is workable), not tiny.
    let (t, d, dv) = (40usize, 6usize, 5usize);
    let cfg = ops::NysCfg { m: 4, w: 8, sink: 2 };
    let mut q = synth(t * d, 20);
    let mut k = synth(t * d, 21);
    let v = synth(t * dv, 22);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 2.0;
    }
    let r = synth(t * dv, 23);
    let loss = |q: &[f64], k: &[f64]| -> f64 {
        let mut o = vec![0f64; t * dv];
        ops::nystrom_head_fwd(q, k, &v, t, d, dv, &cfg, &mut o);
        o.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dq = vec![0f64; t * d];
    let mut dk = vec![0f64; t * d];
    let mut dv_ = vec![0f64; t * dv];
    ops::nystrom_head_bwd(&q, &k, &v, &r, t, d, dv, &cfg, &mut dq, &mut dk, &mut dv_);
    let kc = k.clone();
    let fq = fd_grad(&mut q, |q| loss(q, &kc));
    let qc = q.clone();
    let fk = fd_grad(&mut k, |k| loss(&qc, k));
    let (eq, ek) = (rel_err(&dq, &fq), rel_err(&dk, &fk));
    println!(
        "gradcheck nystrom joint (full functional, M moves in FD): \
         dQ gap {eq:.2e}, dK gap {ek:.2e} — the intentional no-grad-pinv gap"
    );
    assert!(eq < 0.5 && ek < 0.5, "M-freeze gap unexpectedly large");
}

/// The M-constant convention: with q,k FIXED (so M cannot move), FD
/// over V must match analytically to f64 precision, and FD over q,k
/// through everything EXCEPT the pinv is checked by freezing M — here
/// approximated by verifying the near-field-only configuration (w
/// large ⇒ no skeleton ⇒ no M at all) to tight tolerance.
#[test]
fn gradcheck_nystrom_near_only_tight() {
    let (t, d, dv) = (40usize, 6usize, 5usize);
    // w ≥ t → every key is near → joint == exact softmax path but still
    // goes through the skeleton-free branch of the SAME kernel code.
    let cfg = ops::NysCfg { m: 4, w: 64, sink: 0 };
    let mut q = synth(t * d, 24);
    let mut k = synth(t * d, 25);
    let v = synth(t * dv, 26);
    let r = synth(t * dv, 27);
    let loss = |q: &[f64], k: &[f64]| -> f64 {
        let mut o = vec![0f64; t * dv];
        ops::nystrom_head_fwd(q, k, &v, t, d, dv, &cfg, &mut o);
        o.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dq = vec![0f64; t * d];
    let mut dk = vec![0f64; t * d];
    let mut dv_ = vec![0f64; t * dv];
    ops::nystrom_head_bwd(&q, &k, &v, &r, t, d, dv, &cfg, &mut dq, &mut dk, &mut dv_);
    let kc = k.clone();
    let fq = fd_grad(&mut q, |q| loss(q, &kc));
    let qc = q.clone();
    let fk = fd_grad(&mut k, |k| loss(&qc, k));
    let (eq, ek) = (rel_err(&dq, &fq), rel_err(&dk, &fk));
    println!("gradcheck nystrom near-only: dQ {eq:.2e}, dK {ek:.2e}");
    assert!(eq < TOL && ek < TOL);
}

#[test]
fn gradcheck_ce_kl_loss() {
    let vocab = 11usize;
    let target = 3usize;
    let kl_w = 0.7;
    let mut s = synth(vocab, 28);
    let t_log = synth(vocab, 29);
    for x in s.iter_mut() {
        *x *= 3.0;
    }
    let inv_n = 1.0; // single position
    let loss = |s: &[f64]| -> f64 {
        let mut d = vec![0f64; vocab];
        let (ce, kl) = ops::ce_kl_position(s, &t_log, target, kl_w, inv_n, &mut d);
        (1.0 - kl_w) * ce + kl_w * kl
    };
    let mut dl = vec![0f64; vocab];
    let _ = ops::ce_kl_position(&s, &t_log, target, kl_w, inv_n, &mut dl);
    let fs = fd_grad(&mut s, loss);
    let e = rel_err(&dl, &fs);
    println!("gradcheck ce+kl loss: dLogits rel err {e:.2e}");
    assert!(e < TOL);
}

/// Tied lm_head backward is matmul_nt_dx with the embedding as W —
/// checked here in the exact configuration the trainer uses (dX only,
/// embedding frozen).
#[test]
fn gradcheck_tied_head_dx_only() {
    let (n, h, vocab) = (2usize, 5usize, 9usize);
    let mut x = synth(n * h, 30);
    let emb = synth(vocab * h, 31);
    let r = synth(n * vocab, 32);
    let loss = |x: &[f64]| -> f64 {
        let mut lg = vec![0f64; n * vocab];
        ops::matmul_nt(x, &emb, &mut lg, n, h, vocab);
        lg.iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let mut dx = vec![0f64; n * h];
    ops::matmul_nt_dx(&r, &emb, &mut dx, n, h, vocab);
    let fx = fd_grad(&mut x, loss);
    let e = rel_err(&dx, &fx);
    println!("gradcheck tied head (dX only): rel err {e:.2e}");
    assert!(e < TOL);
}

/// GQA repeat_interleave backward = sum-reduce over the group's Q
/// heads. Checked standalone (the trainer's group-accumulation form).
#[test]
fn gradcheck_gqa_repeat_sum_reduce() {
    let (t, d, rep) = (5usize, 3usize, 4usize);
    let mut k = synth(t * d, 33);
    let r = synth(rep * t * d, 34);
    // forward: broadcast k to rep heads, loss = Σ_h Σ k ⊙ r_h
    let loss = |k: &[f64]| -> f64 {
        let mut s = 0f64;
        for h in 0..rep {
            for i in 0..t * d {
                s += k[i] * r[h * t * d + i];
            }
        }
        s
    };
    // analytic: dk = Σ_h r_h (the sum-reduce)
    let mut dk = vec![0f64; t * d];
    for h in 0..rep {
        for i in 0..t * d {
            dk[i] += r[h * t * d + i];
        }
    }
    let fk = fd_grad(&mut k, loss);
    let e = rel_err(&dk, &fk);
    println!("gradcheck gqa sum-reduce: rel err {e:.2e}");
    assert!(e < TOL);
}

/// The pooled f32 GEMMs must agree with the generic serial reference —
/// the hot path is a reorganization, not different math.
#[test]
fn pooled_gemms_match_generic() {
    let (n, k, m) = (37usize, 19usize, 23usize);
    let xf: Vec<f32> = synth(n * k, 40).iter().map(|&v| v as f32).collect();
    let wf: Vec<f32> = synth(m * k, 41).iter().map(|&v| v as f32).collect();
    let dyf: Vec<f32> = synth(n * m, 42).iter().map(|&v| v as f32).collect();

    let mut y0 = vec![0f32; n * m];
    ops::matmul_nt(&xf, &wf, &mut y0, n, k, m);
    let mut y1 = vec![0f32; n * m];
    ops::gemm_nt(&xf, &wf, &mut y1, n, k, m, None);
    for (a, b) in y0.iter().zip(&y1) {
        assert!((a - b).abs() < 1e-5, "gemm_nt {a} vs {b}");
    }

    let mut d0 = vec![0f32; n * k];
    ops::matmul_nt_dx(&dyf, &wf, &mut d0, n, k, m);
    let mut d1 = vec![0f32; n * k];
    ops::gemm_dx(&dyf, &wf, &mut d1, n, k, m, None);
    for (a, b) in d0.iter().zip(&d1) {
        assert!((a - b).abs() < 1e-5, "gemm_dx {a} vs {b}");
    }

    let mut w0 = vec![0f32; m * k];
    ops::matmul_nt_dw(&dyf, &xf, &mut w0, n, k, m);
    let mut w1 = vec![0f32; m * k];
    ops::gemm_dw(&dyf, &xf, &mut w1, n, k, m, None);
    for (a, b) in w0.iter().zip(&w1) {
        assert!((a - b).abs() < 1e-5, "gemm_dw {a} vs {b}");
    }
}

// ─────────────────── GatedDeltaNet through-backward ───────────────────

/// Small GDN geometry with rep = nv/nk = 2 so the SHARED q/k channels
/// (GQA) accumulate across the group's v-heads — the bug-prone path.
struct GdnFix {
    nv: usize,
    nk: usize,
    dk: usize,
    dv: usize,
    kk: usize,
    t: usize,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

impl GdnFix {
    fn new() -> Self {
        let (nv, nk, dk, dv, kk, t) = (2usize, 1usize, 3usize, 2usize, 4usize, 10usize);
        let c_dim = 2 * nk * dk + nv * dv;
        GdnFix {
            nv,
            nk,
            dk,
            dv,
            kk,
            t,
            conv: synth(c_dim * kk, 50).iter().map(|&v| v as f32).collect(),
            a_log: synth(nv, 51).iter().map(|&v| v as f32).collect(),
            dt_bias: synth(nv, 52).iter().map(|&v| v as f32).collect(),
            norm: synth(nv.max(dv), 53).iter().take(dv).map(|&v| 1.0 + v as f32).collect(),
        }
    }
    fn cfg(&self) -> ops::GdnSeqCfg<'_> {
        ops::GdnSeqCfg {
            nv: self.nv,
            nk: self.nk,
            dk: self.dk,
            dv: self.dv,
            kk: self.kk,
            rms_eps: 1e-6,
            conv: &self.conv,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
        }
    }
}

/// BPTT gradcheck over a T=10 window: FD over ALL FOUR projection
/// streams (qkv through conv+SiLU+l2norm+delta-rule state chain; z
/// through the output gate; a through g = exp(−e^A·softplus(a+bias));
/// b through β = σ(b)).
#[test]
fn gradcheck_gdn_bptt_all_streams() {
    let f = GdnFix::new();
    let cfg = f.cfg();
    let (t, nv, dv) = (f.t, f.nv, f.dv);
    let c_dim = cfg.c_dim();
    let vd = nv * dv;
    let mut qkv = synth(t * c_dim, 54);
    let mut z = synth(t * vd, 55);
    let mut a = synth(t * nv, 56);
    let mut b = synth(t * nv, 57);
    let r = synth(t * vd, 58);
    let loss = |qkv: &[f64], z: &[f64], a: &[f64], b: &[f64]| -> f64 {
        let mut o = vec![0f64; t * vd];
        ops::gdn_seq_fwd(qkv, z, a, b, t, &f.cfg(), &mut o);
        o.iter().zip(&r).map(|(x, y)| x * y).sum()
    };
    let mut dqkv = vec![0f64; t * c_dim];
    let mut dz = vec![0f64; t * vd];
    let mut da = vec![0f64; t * nv];
    let mut db = vec![0f64; t * nv];
    ops::gdn_seq_bwd(&qkv, &z, &a, &b, t, &cfg, &r, &mut dqkv, &mut dz, &mut da, &mut db);

    let (zc, ac, bc) = (z.clone(), a.clone(), b.clone());
    let qc0 = qkv.clone();
    let fq = fd_grad(&mut qkv, |x| loss(x, &zc, &ac, &bc));
    let fz = fd_grad(&mut z, |x| loss(&qc0, x, &ac, &bc));
    let fa = fd_grad(&mut a, |x| loss(&qc0, &zc, x, &bc));
    let fb = fd_grad(&mut b, |x| loss(&qc0, &zc, &ac, x));
    let (eq, ez, ea, eb) = (
        rel_err(&dqkv, &fq),
        rel_err(&dz, &fz),
        rel_err(&da, &fa),
        rel_err(&db, &fb),
    );
    println!("gradcheck gdn bptt (T=10): dQKV {eq:.2e}, dZ {ez:.2e}, dA {ea:.2e}, dB {eb:.2e}");
    assert!(eq < TOL && ez < TOL && ea < TOL && eb < TOL);
}

/// Parity: the fcd GDN sequence forward must reproduce the RUNTIME
/// operator (`linear_core::gdn_forward`) — the hybrid teacher path is
/// only honest if it computes the same function the runtime serves.
#[test]
fn gdn_seq_fwd_matches_runtime_operator() {
    use cortiq_engine::linear_core::{gdn_forward, GdnCfg, GdnWeights};
    use cortiq_engine::qtensor::QTensor;
    let f = GdnFix::new();
    let cfg = f.cfg();
    let hidden = 8usize;
    let (t, nv, dv) = (f.t, f.nv, f.dv);
    let c_dim = cfg.c_dim();
    let vd = nv * dv;
    let wqkv: Vec<f32> = synth(c_dim * hidden, 60).iter().map(|&v| v as f32).collect();
    let wz: Vec<f32> = synth(vd * hidden, 61).iter().map(|&v| v as f32).collect();
    let wa: Vec<f32> = synth(nv * hidden, 62).iter().map(|&v| v as f32).collect();
    let wb: Vec<f32> = synth(nv * hidden, 63).iter().map(|&v| v as f32).collect();
    let wout: Vec<f32> = synth(hidden * vd, 64).iter().map(|&v| v as f32).collect();
    let xs: Vec<f32> = synth(t * hidden, 65).iter().map(|&v| v as f32).collect();

    // Runtime path: per-position gdn_forward over its own state.
    let gw = GdnWeights {
        in_proj_qkv: QTensor::from_f32(wqkv.clone(), c_dim, hidden),
        in_proj_z: QTensor::from_f32(wz.clone(), vd, hidden),
        in_proj_a: QTensor::from_f32(wa.clone(), nv, hidden),
        in_proj_b: QTensor::from_f32(wb.clone(), nv, hidden),
        conv1d: f.conv.clone(),
        a_log: f.a_log.clone(),
        dt_bias: f.dt_bias.clone(),
        norm: f.norm.clone(),
        out_proj: QTensor::from_f32(wout.clone(), hidden, vd),
    };
    let gc = GdnCfg {
        num_v_heads: nv,
        num_k_heads: f.nk,
        key_head_dim: f.dk,
        value_head_dim: dv,
        conv_kernel: f.kk,
        hidden_size: hidden,
        rms_eps: 1e-6,
    };
    let mut state = Vec::new();
    let mut runtime_out = Vec::new();
    for p in 0..t {
        let o = gdn_forward(&xs[p * hidden..(p + 1) * hidden], &gw, &gc, &mut state, None);
        runtime_out.extend_from_slice(&o);
    }

    // fcd path: batched projections → gdn_seq_fwd → out_proj.
    let proj = |w: &[f32], m: usize| -> Vec<f64> {
        let mut y = vec![0f64; t * m];
        for p in 0..t {
            for o in 0..m {
                let mut s = 0f64;
                for i in 0..hidden {
                    s += w[o * hidden + i] as f64 * xs[p * hidden + i] as f64;
                }
                y[p * m + o] = s;
            }
        }
        y
    };
    let (qkv, z, a, b) = (proj(&wqkv, c_dim), proj(&wz, vd), proj(&wa, nv), proj(&wb, nv));
    let mut of = vec![0f64; t * vd];
    ops::gdn_seq_fwd(&qkv, &z, &a, &b, t, &cfg, &mut of);
    let mut fcd_out = vec![0f64; t * hidden];
    for p in 0..t {
        for o in 0..hidden {
            let mut s = 0f64;
            for j in 0..vd {
                s += wout[o * vd + j] as f64 * of[p * vd + j];
            }
            fcd_out[p * hidden + o] = s;
        }
    }
    let mut worst = 0f64;
    for i in 0..t * hidden {
        let d = (runtime_out[i] as f64 - fcd_out[i]).abs();
        worst = worst.max(d);
        assert!(
            d < 1e-4,
            "pos {} dim {}: runtime {} vs fcd {}",
            i / hidden,
            i % hidden,
            runtime_out[i],
            fcd_out[i]
        );
    }
    println!("gdn parity vs runtime operator: max |Δ| {worst:.2e} over {t} positions");
}
