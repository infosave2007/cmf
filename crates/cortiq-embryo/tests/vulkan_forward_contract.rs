//! Forward attention contract checks for the native Vulkan seam.
//!
//! These are deliberately tiny, direct operator tests.  They catch a class
//! of bugs that a whole-model backward finite difference can miss when both
//! sides accidentally consume the same unnormalised buffer.
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, GemmBatch, GemmDyn, HkDims, HkWork, Op, ctx, hk_pow_table};

fn softmax_row(xs: &[f32]) -> Vec<f32> {
    let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = xs.iter().map(|x| (x - mx).exp()).collect();
    let den: f32 = out.iter().sum();
    for x in &mut out {
        *x /= den;
    }
    out
}

#[test]
fn causal_softmax_is_in_place_and_backward_uses_slot_one() {
    let c = ctx().expect("native Vulkan adapter is required");
    let t = 4usize;
    let blocks = 2usize;
    let off = 3usize;
    let n = off + blocks * t * t;
    let mut logits = vec![91.0f32; n];
    for block in 0..blocks {
        for row in 0..t {
            for col in 0..t {
                logits[off + block * t * t + row * t + col] =
                    (block * 10 + row * 3 + col) as f32 - 4.0;
            }
        }
    }
    let p = GBuf::from_slice(c, &logits);
    let cmd = Cmd::new(c);
    cmd.causal_softmax_blocks(&p, off, t, blocks);
    cmd.commit();
    let probs = p.to_vec();
    for block in 0..blocks {
        for row in 0..t {
            let base = off + block * t * t + row * t;
            let want = softmax_row(&logits[base..base + row + 1]);
            let sum: f32 = probs[base..base + t].iter().sum();
            assert!((sum - 1.0).abs() < 2e-6, "row sum {sum} at {block}:{row}");
            for col in 0..t {
                let got = probs[base + col];
                let expected = if col <= row { want[col] } else { 0.0 };
                assert!(
                    (got - expected).abs() < 2e-6,
                    "softmax[{block},{row},{col}] got {got} want {expected}"
                );
            }
        }
    }
    assert_eq!(probs[..off], logits[..off], "offset sentinel was modified");

    let mut upstream = vec![77.0f32; n];
    for block in 0..blocks {
        for row in 0..t {
            let base = off + block * t * t + row * t;
            for col in 0..t {
                upstream[base + col] = (col + 1 + row) as f32 * 0.25;
            }
        }
    }
    let d = GBuf::from_slice(c, &upstream);
    let cmd = Cmd::new(c);
    cmd.softmax_bwd_blocks(&p, off, &d, off, t, blocks);
    cmd.commit();
    let got = d.to_vec();
    for block in 0..blocks {
        for row in 0..t {
            let base = off + block * t * t + row * t;
            let mut dot = 0.0;
            for col in 0..t {
                dot += probs[base + col] * upstream[base + col];
            }
            for col in 0..t {
                let want = probs[base + col] * (upstream[base + col] - dot);
                assert!(
                    (got[base + col] - want).abs() < 2e-6,
                    "softmax backward[{block},{row},{col}] got {} want {want}",
                    got[base + col]
                );
            }
        }
    }
    assert_eq!(
        got[..off],
        upstream[..off],
        "backward offset sentinel changed"
    );
}

#[test]
fn hk_kv_materializes_kappa_times_v() {
    let c = ctx().expect("native Vulkan adapter is required");
    let d = HkDims {
        b: 1,
        t: 64,
        nh: 1,
        nph: 1,
        dv: 4,
    };
    let rows = d.b * d.t;
    let thq = GBuf::from_slice(c, &vec![0.0; rows * d.nh * d.nph]);
    let thk = GBuf::from_slice(c, &vec![0.0; rows * d.nh * d.nph]);
    let mut vv = vec![0.0f32; rows * d.nh * d.dv];
    for (i, x) in vv.iter_mut().enumerate() {
        *x = (i as f32 * 0.03125).sin();
    }
    let v = GBuf::from_slice(c, &vv);
    let mut kk = vec![0.0f32; rows * d.nh];
    for (i, x) in kk.iter_mut().enumerate() {
        *x = 0.2 + i as f32 * 0.003;
    }
    let kappa = GBuf::from_slice(c, &kk);
    let phq = GBuf::zeros(c, rows * d.nh * d.p2());
    let phk = GBuf::zeros(c, rows * d.nh * d.p2());
    let kv = GBuf::zeros(c, vv.len());
    let states = GBuf::zeros(c, d.b * d.nh * (d.t / 64 + 1) * d.p2() * d.dv);
    let out = GBuf::zeros(c, vv.len());
    let pow = GBuf::from_slice(c, &hk_pow_table(&[0.9, 0.9], d.nh, d.nph));
    let work = HkWork {
        thq: &thq,
        thk: &thk,
        v: &v,
        kappa: &kappa,
        pow: &pow,
        pow_off: 0,
        phq: &phq,
        phk: &phk,
        kv: &kv,
        states: &states,
        out: &out,
        phase_chunk: None,
        phase_partial: None,
    };
    let cmd = Cmd::new(c);
    cmd.hk_forward(&d, &work);
    cmd.commit();
    let got = kv.to_vec();
    for i in 0..got.len() {
        let want = vv[i] * kk[i / d.dv];
        assert!(
            (got[i] - want).abs() < 2e-6,
            "kv[{i}] got {} want {want}",
            got[i]
        );
    }
}

#[test]
fn indirect_expert_count_reads_count_binding() {
    let c = ctx().expect("native Vulkan adapter is required");
    let counts = GBuf::from_u32(c, &[1]);
    let indir = GBuf::from_u32(c, &[0; 6]);
    let cmd = Cmd::new(c);
    cmd.moe_indirect_args(&counts, 0, &indir, 0, 1, 64, 64, 64);
    cmd.commit();
    let got = unsafe { indir.as_u32_slice().to_vec() };
    assert_eq!(&got[..3], &[1, 1, 1], "forward expert dispatch record");

    let mut aa = vec![0.0f32; 64 * 64];
    for col in 0..64 {
        aa[col] = col as f32 * 0.125 - 2.0;
    }
    let mut bb = vec![0.0f32; 64 * 64];
    for i in 0..64 {
        bb[i * 64 + i] = 1.0;
    }
    let a = GBuf::from_slice(c, &aa);
    let b = GBuf::from_slice(c, &bb);
    let out = GBuf::zeros(c, aa.len());
    let dyn_args = GemmDyn {
        indirect: Some((&indir, 0)),
        kcount: None,
    };
    let cmd = Cmd::new(c);
    cmd.gemm_dyn(
        Op::N,
        Op::T,
        64,
        64,
        64,
        1.0,
        &a,
        0,
        64,
        &b,
        0,
        64,
        0.0,
        &out,
        0,
        64,
        &GemmBatch::none(),
        false,
        &dyn_args,
    );
    cmd.commit();
    let got = out.to_vec();
    for col in 0..64 {
        assert!((got[col] - aa[col]).abs() < 2e-6, "active expert row {col}");
    }
    assert!(
        got[64..].iter().all(|x| x.abs() < 2e-6),
        "padded expert rows were not zero"
    );
}
