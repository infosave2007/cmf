//! Focused identity/provenance/gradient/round-trip proof for the layer-4
//! additive donor GQA lane.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use cortiq_embryo::ops::lcg_vec;
use cortiq_embryo::train::{
    Checkpoint, append_gqa_lane_checkpoint, load_checkpoint, save_checkpoint,
};

#[test]
fn gqa_lane_appends_exact_identity_and_engages_gradients() {
    let Some(c) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut student_cfg = EmbryoCfg::tiny();
    student_cfg.layers = 4;
    student_cfg.anchor_every = 8;
    student_cfg.head_clusters = 0;
    let mut donor_cfg = student_cfg.clone();
    donor_cfg.anchor_every = 1;
    let sl = Layout::new(&student_cfg);
    let dl = Layout::new(&donor_cfg);
    let sp = init_params(&student_cfg, &sl, 7);
    let dp = init_params(&donor_cfg, &dl, 11);
    let ck = Checkpoint {
        cfg: student_cfg.clone(),
        step: 41,
        params: sp.clone(),
        m: Some(sp.iter().map(|x| x * 0.1).collect()),
        v: Some(sp.iter().map(|x| x * 0.2).collect()),
        extras: Vec::new(),
    };
    let dk = Checkpoint {
        cfg: donor_cfg.clone(),
        step: 99,
        params: dp.clone(),
        m: None,
        v: None,
        extras: Vec::new(),
    };
    let graft = append_gqa_lane_checkpoint(&ck, &dk).unwrap();
    assert_eq!(graft.donor_tensors.len(), 3);
    assert!(
        graft
            .donor_tensors
            .iter()
            .all(|x| x.source == "donor.layer4.gqa")
    );
    let cand = graft.checkpoint;
    assert!(cand.cfg.gqa_lane);
    assert_eq!(cand.step, ck.step);
    assert_eq!(&cand.params[..sp.len()], &sp[..]);
    assert_eq!(
        &cand.m.as_ref().unwrap()[..sp.len()],
        &ck.m.as_ref().unwrap()[..]
    );
    assert_eq!(
        &cand.v.as_ref().unwrap()[..sp.len()],
        &ck.v.as_ref().unwrap()[..]
    );
    let cl = Layout::new(&cand.cfg);
    let go = cl.gqa[3].as_ref().unwrap();
    let qn = cand.cfg.anchor_q_heads * cand.cfg.anchor_hd * cand.cfg.hidden;
    let kn = cand.cfg.anchor_kv_heads * cand.cfg.anchor_hd * cand.cfg.hidden;
    let dq = dl
        .names
        .iter()
        .find(|(n, _, _)| n == "layers.3.attn.q")
        .unwrap();
    let dkoff = dl
        .names
        .iter()
        .find(|(n, _, _)| n == "layers.3.attn.k")
        .unwrap();
    let dvoff = dl
        .names
        .iter()
        .find(|(n, _, _)| n == "layers.3.attn.v")
        .unwrap();
    assert_eq!(&cand.params[go.q..go.q + qn], &dp[dq.1..dq.1 + qn]);
    assert_eq!(&cand.params[go.k..go.k + kn], &dp[dkoff.1..dkoff.1 + kn]);
    assert_eq!(&cand.params[go.v..go.v + kn], &dp[dvoff.1..dvoff.1 + kn]);
    assert!(
        cand.params[go.wo..go.wo + cand.cfg.hidden * cand.cfg.anchor_q_heads * cand.cfg.anchor_hd]
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(cand.m.as_ref().unwrap()[go.q..].iter().all(|&x| x == 0.0));

    let path = std::env::temp_dir().join(format!("gqa-lane-{}.ckpt", std::process::id()));
    let ex = cand
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect::<Vec<_>>();
    save_checkpoint(
        &path,
        &cand.cfg,
        cand.step,
        &cand.params,
        cand.m.as_deref(),
        cand.v.as_deref(),
        &ex,
    )
    .unwrap();
    let rt = load_checkpoint(&path).unwrap();
    assert_eq!(rt.params, cand.params);
    assert_eq!(rt.m, cand.m);
    assert_eq!(rt.v, cand.v);
    let _ = std::fs::remove_file(&path);

    let old = EmbryoGpu::new(student_cfg, 1, 64, &sp).unwrap();
    let mut candidate = EmbryoGpu::new(cand.cfg.clone(), 1, 64, &cand.params).unwrap();
    let m = 64;
    let tok: Vec<u32> = lcg_vec(31, m)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cand.cfg.vocab as f32) as u32 % cand.cfg.vocab as u32)
        .collect();
    let a = old.forward_hidden(&tok);
    let b = candidate.forward_hidden(&tok);
    let maxe = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(maxe <= 1e-5, "identity max error {maxe}");
    let tgt = tok.clone();
    unsafe {
        std::ptr::copy_nonoverlapping(tok.as_ptr(), candidate.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(tgt.as_ptr(), candidate.tgt.buf.contents() as *mut u32, m);
    }
    candidate.prepare_head(&tgt);
    let cmd = cortiq_embryo::metal::Cmd::new(c);
    candidate.encode_fwd_bwd(&cmd);
    cmd.commit();
    let wo_grad = candidate.grads_host()
        [go.wo..go.wo + cand.cfg.hidden * cand.cfg.anchor_q_heads * cand.cfg.anchor_hd]
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f32::max);
    assert!(
        wo_grad > 1e-8,
        "zero-output Wo gradient suppressed: {wo_grad:.3e}"
    );
    let mut pp = cand.params.clone();
    for (i, x) in pp[go.wo..go.wo + cand.cfg.hidden * cand.cfg.anchor_q_heads * cand.cfg.anchor_hd]
        .iter_mut()
        .enumerate()
    {
        *x = 0.002 * ((i % 13) as f32 - 6.0) / 6.0;
    }
    candidate.set_params(&pp);
    unsafe {
        std::ptr::copy_nonoverlapping(tok.as_ptr(), candidate.tok.buf.contents() as *mut u32, m);
        std::ptr::copy_nonoverlapping(tgt.as_ptr(), candidate.tgt.buf.contents() as *mut u32, m);
    }
    candidate.prepare_head(&tgt);
    let cmd = cortiq_embryo::metal::Cmd::new(c);
    candidate.encode_fwd_bwd(&cmd);
    cmd.commit();
    let q_grad = candidate.grads_host()[go.q..go.q + qn]
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f32::max);
    assert!(
        q_grad > 1e-8,
        "donor q gradient did not engage: {q_grad:.3e}"
    );
}
