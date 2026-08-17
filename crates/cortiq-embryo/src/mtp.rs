//! MTP heads (docs §3.4): head k predicts token t+1+k from the trunk's
//! final hidden through its own RMSNorm + linear projection and the TIED
//! hierarchical head. Trained after birth on the FROZEN trunk (only the
//! heads' norm + projection move) — the drafts of the verify graph.
//! Exported as `model.mtp.{k}.norm.weight` [H], `model.mtp.{k}.proj.weight`
//! [H,H]; the runtime's speculative decode does not consume them yet.

use crate::metal::{Cmd, GBuf, Op};
use crate::model::EmbryoGpu;
use crate::train::{Checkpoint, Sampler, Shard};
use std::time::Instant;

pub struct MtpState {
    pub k: usize,
    pub h: usize,
    /// [K, H] norm gains, [K, H, H] projections (+ grads, AdamW state)
    pub w: GBuf,
    pub p: GBuf,
    pub gw: GBuf,
    pub gp: GBuf,
    pub mw: GBuf,
    pub vw: GBuf,
    pub mp: GBuf,
    pub vp: GBuf,
    // scratch
    pub n: GBuf,   // [M, H] normed trunk hidden
    pub inv: GBuf, // [M]
    pub hk: GBuf,  // [M, H] projected
    pub dhk: GBuf, // [M, H]
    pub dn: GBuf,  // [M, H]
    pub dx: GBuf,  // [M, H] (trunk grad sink, unused)
    pub tgt: GBuf, // [M] shifted targets
    pub step: u32,
}

impl MtpState {
    pub fn new(gpu: &EmbryoGpu, k: usize, seed: u64) -> MtpState {
        let c = gpu.ctx();
        let h = gpu.cfg.hidden;
        let m = gpu.b * gpu.t;
        // projections start near identity so head k begins as "predict the
        // next token" and moves toward t+1+k
        let mut p = vec![0.0f32; k * h * h];
        let noise = crate::model::gauss_vec(seed, k * h * h);
        for kk in 0..k {
            for i in 0..h {
                for j in 0..h {
                    p[(kk * h + i) * h + j] = if i == j { 1.0 } else { 0.0 } + 0.01 * noise[(kk * h + i) * h + j];
                }
            }
        }
        MtpState {
            k,
            h,
            w: GBuf::from_slice(c, &vec![1.0f32; k * h]),
            p: GBuf::from_slice(c, &p),
            gw: GBuf::zeros(c, k * h),
            gp: GBuf::zeros(c, k * h * h),
            mw: GBuf::zeros(c, k * h),
            vw: GBuf::zeros(c, k * h),
            mp: GBuf::zeros(c, k * h * h),
            vp: GBuf::zeros(c, k * h * h),
            n: GBuf::zeros(c, m * h),
            inv: GBuf::zeros(c, m),
            hk: GBuf::zeros(c, m * h),
            dhk: GBuf::zeros(c, m * h),
            dn: GBuf::zeros(c, m * h),
            dx: GBuf::zeros(c, m * h),
            tgt: GBuf::from_u32(c, &vec![u32::MAX; m.max(1)]),
            step: 0,
        }
    }
}

/// Targets shifted by k within each sequence (u32::MAX where none).
pub fn shifted_targets(tokens: &[u32], b: usize, t: usize, k: usize) -> Vec<u32> {
    let mut out = vec![u32::MAX; b * t];
    for bi in 0..b {
        for pos in 0..t {
            // next-token targets are tokens[pos+1]; head k predicts tokens[pos+1+k]
            if pos + 1 + k < t {
                out[bi * t + pos] = tokens[bi * t + pos + 1 + k];
            }
        }
    }
    out
}

/// One MTP training step on a batch: trunk forward (frozen), then for each
/// head k: norm → proj → tied head loss on the k-shifted targets → grads of
/// the head only → AdamW. Returns per-head mean loss (over valid positions).
pub fn mtp_step(gpu: &mut EmbryoGpu, st: &mut MtpState, tokens: &[u32], lr: f32, train: bool) -> Vec<f32> {
    let m = gpu.b * gpu.t;
    let h = st.h;
    let cfg_eps = gpu.cfg.norm_eps;
    let _ = gpu.forward_hidden(tokens); // xf ready on the device
    let c = gpu.ctx();
    let mut losses = Vec::new();
    for kk in 0..st.k {
        let shifted = shifted_targets(tokens, gpu.b, gpu.t, kk + 1);
        let valid = shifted.iter().filter(|&&x| x != u32::MAX).count().max(1);
        unsafe { std::ptr::copy_nonoverlapping(shifted.as_ptr(), st.tgt.buf.contents() as *mut u32, m) };
        gpu.prepare_head(&shifted);
        let cmd = Cmd::new(c);
        if train {
            cmd.axpby(0.0, &gpu.g, 0.0, &gpu.g, gpu.lay.total); // head grads land here (ignored)
        }
        cmd.rmsnorm_fwd_at(&gpu.xf, &st.w, kk * h, &st.n, &st.inv, m, h, cfg_eps);
        cmd.gemm(Op::N, Op::T, m, h, h, 1.0, &st.n, 0, h, &st.p, kk * h * h, h, 0.0, &st.hk, 0, h);
        gpu.encode_head_on(&cmd, train, &st.hk, &st.dhk, &st.tgt);
        if train {
            // dP_k = dhkᵀ·n ; dn = dhk·P_k ; norm backward → dw_k (dx into a sink)
            cmd.axpby(0.0, &st.gp, 0.0, &st.gp, st.gp.len);
            cmd.axpby(0.0, &st.gw, 0.0, &st.gw, st.gw.len);
            cmd.gemm(Op::T, Op::N, h, h, m, 1.0, &st.dhk, 0, h, &st.n, 0, h, 0.0, &st.gp, kk * h * h, h);
            cmd.gemm(Op::N, Op::N, m, h, h, 1.0, &st.dhk, 0, h, &st.p, kk * h * h, h, 0.0, &st.dn, 0, h);
            cmd.rmsnorm_bwd_at(&gpu.xf, &st.w, kk * h, &st.dn, &st.inv, &st.dx, 0.0, &st.gw, kk * h, m, h);
            st.step += 1;
            cmd.adamw_at(&st.p, &st.gp, &st.mp, &st.vp, kk * h * h, h * h, lr, 0.9, 0.95, 1e-8, 0.0, st.step, 1.0);
            cmd.adamw_at(&st.w, &st.gw, &st.mw, &st.vw, kk * h, h, lr, 0.9, 0.95, 1e-8, 0.0, st.step, 1.0);
        }
        cmd.commit();
        // read_loss divides by m; rescale to valid positions
        losses.push(gpu.read_loss() * m as f32 / valid as f32);
    }
    losses
}

/// Train K heads on `corpus` for `steps`; returns (state, per-head held-out losses).
pub fn train_mtp(ck: &Checkpoint, corpus: &Shard, k: usize, steps: usize, lr: f32, batch: usize, seq: usize, seed: u64) -> anyhow::Result<(EmbryoGpu, MtpState, Vec<f32>)> {
    let mut gpu = EmbryoGpu::new(ck.cfg.clone(), batch, seq, &ck.params).ok_or_else(|| anyhow::anyhow!("no Metal"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false);
    let mut st = MtpState::new(&gpu, k, seed);
    let n = corpus.tokens.len();
    let cut = n - n / 10;
    let train = Shard { tokens: corpus.tokens[..cut].to_vec() };
    let held = Shard { tokens: corpus.tokens[cut..].to_vec() };
    let mut sampler = Sampler::new(batch, seq, seed);
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let t0 = Instant::now();
    for step in 0..steps {
        sampler.batch(&train, &mut tk, &mut tg);
        let lr_s = lr * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / steps.max(1) as f32).cos());
        let l = mtp_step(&mut gpu, &mut st, &tk, lr_s, true);
        if (step + 1) % 25 == 0 || step + 1 == steps {
            eprintln!("  mtp step {:>4} losses {:?} [{:.0} s]", step + 1, l.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>(), t0.elapsed().as_secs_f64());
        }
    }
    // held-out
    let mut acc = vec![0.0f32; k];
    let nval = 4;
    for i in 0..nval {
        Sampler::fixed_batch(&held, batch, seq, i, &mut tk, &mut tg);
        let l = mtp_step(&mut gpu, &mut st, &tk, 0.0, false);
        for (a, b) in acc.iter_mut().zip(&l) {
            *a += b / nval as f32;
        }
    }
    Ok((gpu, st, acc))
}

/// Append the trained heads to a .cmf as `model.mtp.{k}.{norm,proj}.weight`
/// (base tensors byte-identical; the runtime ignores unknown tensors).
pub fn append_to_cmf(base: &std::path::Path, out: &std::path::Path, st: &MtpState) -> anyhow::Result<usize> {
    use cortiq_core::format::{CmfModel, TensorSpec};
    use cortiq_core::types::TensorDtype;
    let model = CmfModel::open(base)?;
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut kept = 0usize;
    for t in &model.tensors {
        if t.name.starts_with("model.mtp.") {
            continue;
        }
        specs.push(TensorSpec { name: t.name.clone(), dtype: t.dtype, shape: t.shape.clone(), data: model.tensor_bytes(&t.name)?.to_vec() });
        kept += 1;
    }
    let (w, p, h) = (st.w.to_vec(), st.p.to_vec(), st.h);
    let f32b = |x: &[f32]| {
        let mut v = Vec::with_capacity(x.len() * 4);
        for f in x {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    };
    for k in 0..st.k {
        specs.push(TensorSpec { name: format!("model.mtp.{}.norm.weight", k + 1), dtype: TensorDtype::F32, shape: vec![h], data: f32b(&w[k * h..(k + 1) * h]) });
        specs.push(TensorSpec { name: format!("model.mtp.{}.proj.weight", k + 1), dtype: TensorDtype::F32, shape: vec![h, h], data: f32b(&p[k * h * h..(k + 1) * h * h]) });
    }
    let mut header = model.header.clone();
    let mut prov = header.provenance.clone().unwrap_or(serde_json::json!({}));
    prov["mtp_heads"] = serde_json::json!({"k": st.k, "layout": "model.mtp.{k}.{norm,proj}.weight, tied hierarchical head; k = 1..K predicts t+1+k"});
    header.provenance = Some(prov);
    let vocab = model.vocab.clone();
    CmfModel::write(out, &header, &specs, if model.masks.masks.is_empty() { None } else { Some(&model.masks) }, vocab.as_deref())?;
    Ok(kept)
}
