//! The Lumina Next-DiT block (Lumina-Image 2.0) on CUDA, one block a call:
//! the hidden state goes up, the block runs on the card — the sandwich RMS
//! norms, the q/k/v products, per-head q/k norm with the interleaved
//! rotation, grouped-query attention per segment (`h3_flash`), the output
//! product, the gated post-normed residual, then the SwiGLU feed-forward the
//! same way — and comes back. The weights are the H3 chain's store: q4tp on
//! the card in their codec, expanded to an f16 plane for their product.

use super::*;
use crate::gpu::DitBlockArgs;

const LUMINA_SRC: &str = include_str!("lumina.cu");
/// The per-head q/k norm's epsilon (the reference's 1e-5).
const QK_EPS: f32 = 1e-5;

#[derive(Clone, Copy)]
pub(super) struct LbFuncs {
    norm: cs::CUfunction,
    res: cs::CUfunction,
    heads: cs::CUfunction,
    swiglu: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<LbFuncs> {
    if let Some(f) = st.h3.lumina {
        return Ok(f);
    }
    let module = compile(LUMINA_SRC, Vec::new(), c.arch)?;
    let f = LbFuncs {
        norm: func(module, "lb_norm")?,
        res: func(module, "lb_res")?,
        heads: func(module, "lb_heads")?,
        swiglu: func(module, "lb_swiglu")?,
    };
    st.h3.lumina = Some(f);
    Ok(f)
}

/// One block over `x` `[n][hidden]` (independent sequences of `segs` rows
/// each), in place. `false`: nothing was written.
pub fn lumina_block(model: &Arc<CmfModel>, a: &DitBlockArgs<'_>, segs: &[usize], x: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || a.resident_in || a.resident_out || std::env::var("CMF_LUMINA_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { block(c, &mut st, model, a, segs, x) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("lumina block", &e);
            false
        }
    }
}

unsafe fn block(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, a: &DitBlockArgs<'_>, segs: &[usize], x: &mut [f32]) -> Res<()> {
    if x.len() != a.n * a.hidden {
        return Err("a block of the wrong shape".into());
    }
    let xb = st.h3.buf(c, "lb x", a.n * a.hidden * 4)?;
    unsafe {
        cu::memcpy_htod_async(xb, x, c.stream).map_err(derr)?;
        block_dev(c, st, model, a, segs, xb)?;
    }
    let mut out = vec![0f32; a.n * a.hidden];
    unsafe { cu::memcpy_dtoh_async(&mut out, xb, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the block produced a value that is not finite".into());
    }
    x.copy_from_slice(&out);
    Ok(())
}

/// One block over the rows `xb` (`[n][hidden]` f32 on the card), in place
/// on the card: the Lumina Next-DiT block, and Z-Image's (`zimage.rs`), whose
/// block is the same arithmetic. The gates arrive `tanh`-ed.
pub(super) unsafe fn block_dev(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, a: &DitBlockArgs<'_>, segs: &[usize], xb: u64) -> Res<()> {
    let (n, hs, inter, nh, nkv, hd) = (a.n, a.hidden, a.inter, a.nh, a.nkv, a.hd);
    let (qd, kd) = (nh * hd, nkv * hd);
    if n == 0 || segs.iter().sum::<usize>() != n || nkv == 0 || !nh.is_multiple_of(nkv) {
        return Err("a block of the wrong shape".into());
    }
    if !hd.is_multiple_of(32) || a.rope_cos.len() != n * hd / 2 || a.rope_sin.len() != n * hd / 2 {
        return Err("a head width or rotation table outside the kernels".into());
    }
    let f = funcs(st, c)?;
    let h3f = st.h3.funcs(c)?;
    let fl = st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;

    // constants: norms, modulation, rotation
    let mut cv: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        let at = cv.len();
        cv.extend_from_slice(v);
        at
    };
    let (o_n1, o_n2, o_f1, o_f2) = (put(a.norm1), put(a.norm2), put(a.ffn_norm1), put(a.ffn_norm2));
    let (o_qn, o_kn) = (put(a.norm_q), put(a.norm_k));
    let (o_sm, o_gm, o_sf, o_gf) = (put(a.s_msa), put(a.gate_msa), put(a.s_mlp), put(a.gate_mlp));
    let (o_cos, o_sin) = (put(a.rope_cos), put(a.rope_sin));
    let cb = st.h3.buf(c, "lb c", cv.len() * 4 + 64)?;
    let at = |o: usize| cb + (o * 4) as u64;

    let ns_max = segs.iter().map(|s| s.next_multiple_of(16)).max().unwrap_or(16);
    let xn = st.h3.buf(c, "lb xn", n * hs.max(inter) * 2)?;
    let y = st.h3.buf(c, "lb y", n * qd.max(inter) * 4)?;
    let y2 = st.h3.buf(c, "lb y2", n * kd.max(inter) * 4)?;
    let y3 = st.h3.buf(c, "lb y3", n * kd * 4)?;
    let hq = st.h3.buf(c, "lb hq", n * qd * 4)?;
    let qp = st.h3.buf(c, "lb q", n * qd * 2)?;
    let kp = st.h3.buf(c, "lb k", n * kd * 2)?;
    let vp = st.h3.buf(c, "lb v", nkv * ns_max * hd * 2 + 64)?;
    let ao = st.h3.buf(c, "lb ao", n * qd * 2)?;
    let w = st.h3.buf(c, "lb w", hs.max(inter) * qd.max(inter).max(hs) * 2)?;
    let w2 = st.h3.buf(c, "lb w2", hs.max(inter) * hs.max(inter) * 2)?;
    unsafe { cu::memcpy_htod_async(cb, &cv, c.stream).map_err(derr)? };
    let eps = a.eps;
    let rows = (n as u32, 1);
    let hsi = hs as i32;

    // ── attention ──
    unsafe {
        c.launch(f.norm, rows, 256, 0, kargs!(xb, xn, at(o_n1), at(o_sm), hsi, eps, 1.0f32))?;
        expand(c, st, model, a.wq, w, true, true)?;
        c.gemm(true, false, qd, n, hs, 1.0, w, hs, xn, hs, 0.0, y, qd, Gemm::Half)?;
        expand(c, st, model, a.wk, w2, true, true)?;
        c.gemm(true, false, kd, n, hs, 1.0, w2, hs, xn, hs, 0.0, y2, kd, Gemm::Half)?;
        expand(c, st, model, a.wv, w, true, true)?;
        c.gemm(true, false, kd, n, hs, 1.0, w, hs, xn, hs, 0.0, y3, kd, Gemm::Half)?;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let (nhi, hdi, hpk) = (nh as i32, hd as i32, (nh / nkv) as i32);
    let mut r0 = 0usize;
    for &sn in segs {
        if sn == 0 {
            continue;
        }
        let ns = sn.next_multiple_of(16);
        let (sni, nsi) = (sn as i32, ns as i32);
        let cs_ = at(o_cos) + (r0 * hd / 2 * 4) as u64;
        let sn_ = at(o_sin) + (r0 * hd / 2 * 4) as u64;
        unsafe {
            // q: norm, rotation, head-major, then the flash kernel's layouts
            let (qi, ki) = (nh as i32, nkv as i32);
            c.launch(f.heads, (sn as u32, nh as u32), 32, 0, kargs!(y + (r0 * qd * 4) as u64, hq, at(o_qn), cs_, sn_, sni, qi, hdi, QK_EPS))?;
            let (m0, m1, m2) = (0i32, 1i32, 2i32);
            c.launch(h3f.pack, ((nh * ns * hd).div_ceil(256) as u32, 1), 256, 0, kargs!(hq, qp, sni, nsi, qi, hdi, hdi, m0))?;
            c.launch(f.heads, (sn as u32, nkv as u32), 32, 0, kargs!(y2 + (r0 * kd * 4) as u64, hq, at(o_kn), cs_, sn_, sni, ki, hdi, QK_EPS))?;
            c.launch(h3f.pack, ((nkv * ns * hd).div_ceil(256) as u32, 1), 256, 0, kargs!(hq, kp, sni, nsi, ki, hdi, hdi, m1))?;
            let nul = 0u64;
            c.launch(f.heads, (sn as u32, nkv as u32), 32, 0, kargs!(y3 + (r0 * kd * 4) as u64, hq, nul, nul, nul, sni, ki, hdi, QK_EPS))?;
            c.launch(h3f.pack, ((nkv * ns * hd).div_ceil(256) as u32, 1), 256, 0, kargs!(hq, vp, sni, nsi, ki, hdi, hdi, m2))?;
            let aor = ao + (r0 * qd * 2) as u64;
            c.launch(fl, (sn.div_ceil(64) as u32, nh as u32), 128, 0, kargs!(qp, kp, vp, aor, sni, nsi, nhi, scale, hpk))?;
        }
        r0 += sn;
    }
    unsafe {
        expand(c, st, model, a.wo, w, true, true)?;
        c.gemm(true, false, hs, n, qd, 1.0, w, qd, ao, qd, 0.0, y, hs, Gemm::Half)?;
        c.launch(f.res, rows, 256, 0, kargs!(xb, y, at(o_n2), at(o_gm), hsi, eps))?;
    }

    // ── the feed-forward ──
    const INV: f32 = 1.0 / 64.0;
    unsafe {
        c.launch(f.norm, rows, 256, 0, kargs!(xb, xn, at(o_f1), at(o_sf), hsi, eps, 1.0f32))?;
        expand(c, st, model, a.w1, w, true, true)?;
        c.gemm(true, false, inter, n, hs, 1.0, w, hs, xn, hs, 0.0, y, inter, Gemm::Half)?;
        expand(c, st, model, a.w3, w2, true, true)?;
        c.gemm(true, false, inter, n, hs, 1.0, w2, hs, xn, hs, 0.0, y2, inter, Gemm::Half)?;
        let total = (n * inter) as u64;
        c.launch(f.swiglu, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(y, y2, xn, INV, total))?;
        expand(c, st, model, a.w2, w, true, true)?;
        c.gemm(true, false, hs, n, inter, 1.0 / INV, w, inter, xn, inter, 0.0, y, hs, Gemm::Half)?;
        c.launch(f.res, rows, 256, 0, kargs!(xb, y, at(o_f2), at(o_gf), hsi, eps))?;
    }
    Ok(())
}
