//! The Qwen-Image transformer (Qwen-Image-Edit-2509) on CUDA: every block of a
//! step as one resident chain. Both streams go up once and come back once;
//! between them a block runs on the card — each stream's LayerNorm with its
//! shift and scale, its q/k/v products, per-head RMS norm of q and k with the
//! interleaved rotation, joint attention over [text | image] (`h3_flash`),
//! each stream's output product and gated residual, then each stream's
//! tanh-GELU MLP the same way. The weights are the H3 chain's store: q4tp on
//! the card in their codec under `CMF_GPU_VRAM_MB`, expanded to an f16 plane
//! for their product.

use super::*;
use crate::gpu::{QwenImageChainArgs, QwenImageChainBlock};

const QI_SRC: &str = include_str!("qwen_image.cu");
const LN_EPS: f32 = 1e-6;
/// The MLP's middle is stored divided by this before its second product.
const MID_INV: f32 = 1.0 / 64.0;

#[derive(Clone, Copy)]
pub(super) struct QiFuncs {
    lnmod: cs::CUfunction,
    heads: cs::CUfunction,
    addg: cs::CUfunction,
    gelu: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<QiFuncs> {
    if let Some(f) = st.h3.qwen_image {
        return Ok(f);
    }
    let module = compile(QI_SRC, Vec::new(), c.arch)?;
    let f = QiFuncs {
        lnmod: func(module, "qi_lnmod")?,
        heads: func(module, "qi_heads")?,
        addg: func(module, "qi_addg")?,
        gelu: func(module, "qi_gelu")?,
    };
    st.h3.qwen_image = Some(f);
    Ok(f)
}

pub fn qwen_image_chain(model: &Arc<CmfModel>, args: &mut QwenImageChainArgs<'_>) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_QWEN_IMAGE_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { chain(c, &mut st, model, args) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("qwen image chain", &e);
            false
        }
    }
}

/// A block's twelve q4tp tensors, in the order the chain reads them.
fn tensors(b: &QwenImageChainBlock<'_>) -> [usize; 12] {
    [
        b.image_q,
        b.image_k,
        b.image_v,
        b.text_q,
        b.text_k,
        b.text_v,
        b.image_out,
        b.text_out,
        b.image_mlp_in,
        b.image_mlp_out,
        b.text_mlp_in,
        b.text_mlp_out,
    ]
}

unsafe fn chain(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, a: &mut QwenImageChainArgs<'_>) -> Res<()> {
    let (ni, nt, heads, hd) = (a.image_tokens, a.text_tokens, a.heads, a.head_dim);
    let hs = heads * hd;
    let total = ni + nt;
    if ni == 0 || nt == 0 || a.blocks.is_empty() || a.image.len() != ni * hs || a.text.len() != nt * hs {
        return Err("empty stack or a stream of the wrong size".into());
    }
    if a.image_cos.len() != ni * hd / 2 || a.text_cos.len() != nt * hd / 2 || a.image_sin.len() != a.image_cos.len() || a.text_sin.len() != a.text_cos.len() {
        return Err("rotation tables of the wrong size".into());
    }
    let mut inter = 0usize;
    for b in a.blocks {
        if tensors(b).contains(&usize::MAX) {
            return Err("a projection that is not a q4tp matrix".into());
        }
        if b.image_mod.len() != 6 * hs || b.text_mod.len() != 6 * hs {
            return Err("a modulation of the wrong width".into());
        }
        let (_, r, _, _) = packed(model, b.image_mlp_in)?;
        inter = inter.max(r);
    }
    let f = funcs(st, c)?;
    let h3f = st.h3.funcs(c)?;
    let fl = st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;
    let cap = ceiling();
    let panel: usize = match cap {
        Some(b) if b < (9 << 30) => 192 << 20,
        _ => 512 << 20,
    };
    let cq = (panel / (inter.max(hs) * 6)).clamp(16, 4096);

    // ── constants: one upload ──
    let mut cv: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        if v.is_empty() {
            return usize::MAX;
        }
        let at = cv.len();
        cv.extend_from_slice(v);
        at
    };
    struct Off {
        im: usize,
        tm: usize,
        b: [usize; 12],
        n: [usize; 4],
    }
    let mut lay: Vec<Off> = Vec::with_capacity(a.blocks.len());
    for b in a.blocks {
        lay.push(Off {
            im: put(b.image_mod),
            tm: put(b.text_mod),
            b: [
                put(b.image_q_bias),
                put(b.image_k_bias),
                put(b.image_v_bias),
                put(b.text_q_bias),
                put(b.text_k_bias),
                put(b.text_v_bias),
                put(b.image_out_bias),
                put(b.text_out_bias),
                put(b.image_mlp_in_bias),
                put(b.image_mlp_out_bias),
                put(b.text_mlp_in_bias),
                put(b.text_mlp_out_bias),
            ],
            n: [put(b.image_q_norm), put(b.image_k_norm), put(b.text_q_norm), put(b.text_k_norm)],
        });
    }
    let (o_ic, o_is, o_tc, o_ts) = (put(a.image_cos), put(a.image_sin), put(a.text_cos), put(a.text_sin));

    // ── the working set ──
    let ns = total.next_multiple_of(16);
    let xi = st.h3.buf(c, "qi xi", ni * hs * 4)?;
    let xt = st.h3.buf(c, "qi xt", nt * hs * 4)?;
    let xn = st.h3.buf(c, "qi xn", ni.max(nt) * hs * 2)?;
    let y = st.h3.buf(c, "qi y", cq * inter.max(hs) * 4)?;
    let hq = st.h3.buf(c, "qi hq", total * hs * 4)?;
    let hk = st.h3.buf(c, "qi hk", total * hs * 4)?;
    let hv = st.h3.buf(c, "qi hv", total * hs * 4)?;
    let qp = st.h3.buf(c, "qi q", total * hs * 2)?;
    let kp = st.h3.buf(c, "qi k", total * hs * 2)?;
    let vp = st.h3.buf(c, "qi v", heads * ns * hd * 2 + 64)?;
    let ao = st.h3.buf(c, "qi ao", total * hs * 2)?;
    let act = st.h3.buf(c, "qi act", cq * inter * 2)?;
    let w = st.h3.buf(c, "qi w", inter.max(hs) * hs * 2)?;
    let w2 = st.h3.buf(c, "qi w2", inter.max(hs) * hs * 2)?;
    let cb = st.h3.buf(c, "qi c", cv.len() * 4 + 64)?;
    let at = |o: usize| -> u64 { if o == usize::MAX { 0 } else { cb + (o * 4) as u64 } };

    // ── which blocks stay on the card ──
    let uid = model.uid();
    let (free, total_mem) = cu::mem_get_info().map_err(derr)?;
    let reserve: i64 = (160 << 20) + (128 << 20);
    let mut room: i64 = match cap {
        Some(b) => b as i64 - (total_mem - free) as i64,
        None => free as i64,
    } - reserve;
    let mut keep = vec![false; a.blocks.len()];
    let mut all_before = true;
    for (i, b) in a.blocks.iter().enumerate() {
        let mut need = 0u64;
        for ix in tensors(b) {
            if !st.h3.mats.contains_key(&(uid, ix)) {
                let (kind, r, co, _) = packed(model, ix)?;
                need += packed_dev_bytes(kind, r, co);
            }
        }
        if need == 0 {
            keep[i] = true;
        } else if all_before && room >= need as i64 {
            keep[i] = true;
            room -= need as i64;
        } else {
            all_before = false;
        }
    }
    unsafe {
        cu::memcpy_htod_async(xi, &*a.image, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(xt, &*a.text, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(cb, &cv, c.stream).map_err(derr)?;
    }

    // `src` f16 [n][k] through tensor `ix` ([out][k]), a chunk of rows at a
    // time into `y`; `after(r0, rows)` takes each chunk from there
    let project = |st: &mut State, ix: usize, keep: bool, src: u64, n: usize, k: usize, alpha: f32, after: &mut dyn FnMut(usize, usize) -> Res<()>| -> Res<usize> {
        let (out, kk) = unsafe { expand(c, st, model, ix, w, keep, true)? };
        if kk != k {
            return Err("a projection of the wrong width".into());
        }
        let mut r0 = 0usize;
        while r0 < n {
            let rows = cq.min(n - r0);
            unsafe { c.gemm(true, false, out, rows, k, alpha, w, k, src + (r0 * k * 2) as u64, k, 0.0, y, out, Gemm::Half)? };
            after(r0, rows)?;
            r0 += rows;
        }
        Ok(out)
    };
    let (hsi, hdi, hi, toti) = (hs as i32, hd as i32, heads as i32, total as i32);
    let nul = 0u64;
    for (bi, b) in a.blocks.iter().enumerate() {
        let (l, k) = (&lay[bi], keep[bi]);
        let t = tensors(b);
        // ── attention: each stream's q/k/v into the joint head-major buffers ──
        // (stream state, rows, mod, first row, q/k/v tensors and biases, norms, rotation)
        let streams = [
            (xt, nt, l.tm, 0usize, [t[3], t[4], t[5]], [l.b[3], l.b[4], l.b[5]], [l.n[2], l.n[3]], (o_tc, o_ts)),
            (xi, ni, l.im, nt, [t[0], t[1], t[2]], [l.b[0], l.b[1], l.b[2]], [l.n[0], l.n[1]], (o_ic, o_is)),
        ];
        for &(x, n, m, r0, ix, bs, nw, (oc, os)) in &streams {
            // the stream's normed, modulated rows (shift chunk 0, scale chunk 1)
            unsafe { c.launch(f.lnmod, (n as u32, 1), 256, 0, kargs!(x, xn, at(m), at(m + hs), hsi, LN_EPS, 1.0f32))? };
            for (part, dst) in [hq, hk, hv].into_iter().enumerate() {
                let (wn, cs_, sn_) = if part < 2 { (at(nw[part]), at(oc), at(os)) } else { (nul, nul, nul) };
                let bias = at(bs[part]);
                project(st, ix[part], k, xn, n, hs, 1.0, &mut |q0, rows| {
                    let (q0i, rr) = ((r0 + q0) as i32, rows as u32);
                    let (cso, sno) = if cs_ == 0 { (0, 0) } else { (cs_ + (q0 * hd / 2 * 4) as u64, sn_ + (q0 * hd / 2 * 4) as u64) };
                    unsafe { c.launch(f.heads, (rr, heads as u32), 32, 0, kargs!(y, bias, dst, wn, cso, sno, hi, hdi, toti, q0i, LN_EPS)) }
                })?;
            }
        }
        // the flash kernel's layouts of the joint q, k and v
        let nsi = ns as i32;
        for (mode, (src, dst)) in [(hq, qp), (hk, kp), (hv, vp)].into_iter().enumerate() {
            let mode = mode as i32;
            unsafe { c.launch(h3f.pack, ((heads * ns * hd).div_ceil(256) as u32, 1), 256, 0, kargs!(src, dst, toti, nsi, hi, hdi, hdi, mode))? };
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let one = 1i32;
        unsafe { c.launch(fl, (total.div_ceil(64) as u32, heads as u32), 128, 0, kargs!(qp, kp, vp, ao, toti, nsi, hi, scale, one))? };
        // ── each stream's output product and gated residual (gate chunk 2) ──
        for &(x, n, m, r0, ox, ob) in &[(xt, nt, l.tm, 0usize, t[7], l.b[7]), (xi, ni, l.im, nt, t[6], l.b[6])] {
            let gate = at(m + 2 * hs);
            let bias = at(ob);
            project(st, ox, k, ao + (r0 * hs * 2) as u64, n, hs, 1.0, &mut |q0, rows| {
                let tot = (rows * hs) as u64;
                let xr = x + (q0 * hs * 4) as u64;
                unsafe { c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y, bias, gate, hsi, tot)) }
            })?;
        }
        // ── each stream's MLP (shift chunk 3, scale 4, gate 5) ──
        for &(x, n, m, ix_in, b_in, ix_out, b_out) in &[(xi, ni, l.im, t[8], l.b[8], t[9], l.b[9]), (xt, nt, l.tm, t[10], l.b[10], t[11], l.b[11])] {
            unsafe { c.launch(f.lnmod, (n as u32, 1), 256, 0, kargs!(x, xn, at(m + 3 * hs), at(m + 4 * hs), hsi, LN_EPS, 1.0f32))? };
            let gate = at(m + 5 * hs);
            let (bi_, bo_) = (at(b_in), at(b_out));
            // both planes once, then a chunk of rows through both products
            let (mid, _) = unsafe { expand(c, st, model, ix_in, w, k, true)? };
            let (o, _) = unsafe { expand(c, st, model, ix_out, w2, k, true)? };
            if o != hs {
                return Err("an MLP output of the wrong width".into());
            }
            let mut r0 = 0usize;
            while r0 < n {
                let rows = cq.min(n - r0);
                unsafe {
                    c.gemm(true, false, mid, rows, hs, 1.0, w, hs, xn + (r0 * hs * 2) as u64, hs, 0.0, y, mid, Gemm::Half)?;
                    let tot = (rows * mid) as u64;
                    let midi = mid as i32;
                    c.launch(f.gelu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(y, bi_, act, midi, MID_INV, tot))?;
                }
                unsafe {
                    c.gemm(true, false, hs, rows, mid, 1.0 / MID_INV, w2, mid, act, mid, 0.0, y, hs, Gemm::Half)?;
                    let tot = (rows * hs) as u64;
                    let xr = x + (r0 * hs * 4) as u64;
                    c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y, bo_, gate, hsi, tot))?;
                }
                r0 += rows;
            }
        }
    }
    let (mut oi, mut ot) = (vec![0f32; ni * hs], vec![0f32; nt * hs]);
    unsafe {
        cu::memcpy_dtoh_async(&mut oi, xi, c.stream).map_err(derr)?;
        cu::memcpy_dtoh_async(&mut ot, xt, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if oi.iter().chain(ot.iter()).any(|v| !v.is_finite()) {
        return Err("a stream came back with a value that is not finite".into());
    }
    a.image.copy_from_slice(&oi);
    a.text.copy_from_slice(&ot);
    Ok(())
}
