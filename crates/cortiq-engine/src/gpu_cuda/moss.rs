//! The MOSS-Image transformer on CUDA: every block of a forward as one
//! resident chain. The joint [image | text] rows go up once and come back
//! once; between them a block runs on the card — the RMS norm with its weight
//! and the step's shift and scale, the q/k/v products, per-head RMS norm of q
//! and k with the split-half rotation, attention over all rows (`h3_flash`),
//! the output product and gated residual, then the fused up/gate product, the
//! SiTU and the down product the same way. The weights are the H3 chain's
//! store: q4tp or q8 on the card in their codec under `CMF_GPU_VRAM_MB`,
//! expanded to an f16 plane for their product.

use super::*;
use crate::gpu::{MossChainArgs, MossChainBlock};

const MOSS_SRC: &str = include_str!("moss.cu");
const EPS: f32 = 1e-6;
/// The feed-forward's middle is stored times this before its second product
/// (the SiTU is bounded by 100: no range guard).
const MID_INV: f32 = 1.0;

#[derive(Clone, Copy)]
pub(super) struct MoFuncs {
    norm: cs::CUfunction,
    heads: cs::CUfunction,
    situ: cs::CUfunction,
    addg: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<MoFuncs> {
    if let Some(f) = st.h3.moss {
        return Ok(f);
    }
    let module = compile(MOSS_SRC, Vec::new(), c.arch)?;
    let f = MoFuncs {
        norm: func(module, "mo_norm")?,
        heads: func(module, "mo_heads")?,
        situ: func(module, "mo_situ")?,
        addg: func(module, "mo_addg")?,
    };
    st.h3.moss = Some(f);
    Ok(f)
}

pub fn moss_chain(model: &Arc<CmfModel>, args: &mut MossChainArgs<'_>) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_MOSS_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { chain(c, &mut st, model, args) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("moss chain", &e);
            false
        }
    }
}

/// A block's six tensors, in the order the chain reads them.
fn tensors(b: &MossChainBlock<'_>) -> [usize; 6] {
    [b.q, b.k, b.v, b.o, b.up_gate, b.down]
}

unsafe fn chain(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, a: &mut MossChainArgs<'_>) -> Res<()> {
    let (n, heads, hd) = (a.tokens, a.heads, a.head_dim);
    let d = heads * hd;
    if n == 0 || a.blocks.is_empty() || a.x.len() != n * d || a.modulation.len() != 6 * d {
        return Err("empty stack or rows of the wrong size".into());
    }
    if a.cos.len() != n * hd / 2 || a.sin.len() != a.cos.len() {
        return Err("rotation tables of the wrong size".into());
    }
    let mut inter = 0usize;
    for b in a.blocks {
        if b.norm1.len() != d || b.norm2.len() != d || b.q_norm.len() != hd || b.k_norm.len() != hd {
            return Err("a norm of the wrong width".into());
        }
        let (_, r, _, _) = packed(model, b.up_gate)?;
        inter = inter.max(r / 2);
    }
    let f = funcs(st, c)?;
    let h3f = st.h3.funcs(c)?;
    let fl = st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;
    let cap = ceiling();
    let panel: usize = match cap {
        Some(b) if b < (9 << 30) => 192 << 20,
        _ => 512 << 20,
    };
    let cq = (panel / (2 * inter.max(d) * 6)).clamp(16, 4096);

    // ── constants: one upload ──
    let mut cv: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        let at = cv.len();
        cv.extend_from_slice(v);
        at
    };
    let o_mod = put(a.modulation);
    let lay: Vec<[usize; 4]> = a.blocks.iter().map(|b| [put(b.norm1), put(b.norm2), put(b.q_norm), put(b.k_norm)]).collect();
    let (o_cos, o_sin) = (put(a.cos), put(a.sin));

    // ── the working set ──
    let ns = n.next_multiple_of(16);
    let xb = st.h3.buf(c, "mo x", n * d * 4)?;
    let xn = st.h3.buf(c, "mo xn", n * d * 2)?;
    let y = st.h3.buf(c, "mo y", cq * 2 * inter.max(d) * 4)?;
    let hq = st.h3.buf(c, "mo hq", n * d * 4)?;
    let hk = st.h3.buf(c, "mo hk", n * d * 4)?;
    let hv = st.h3.buf(c, "mo hv", n * d * 4)?;
    let qp = st.h3.buf(c, "mo q", n * d * 2)?;
    let kp = st.h3.buf(c, "mo k", n * d * 2)?;
    let vp = st.h3.buf(c, "mo v", heads * ns * hd * 2 + 64)?;
    let ao = st.h3.buf(c, "mo ao", n * d * 2)?;
    let act = st.h3.buf(c, "mo act", cq * inter * 2)?;
    let w = st.h3.buf(c, "mo w", 2 * inter.max(d) * d * 2)?;
    let w2 = st.h3.buf(c, "mo w2", inter.max(d) * d * 2)?;
    let cb = st.h3.buf(c, "mo c", cv.len() * 4 + 64)?;
    let at = |o: usize| -> u64 { cb + (o * 4) as u64 };

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
        cu::memcpy_htod_async(xb, &*a.x, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(cb, &cv, c.stream).map_err(derr)?;
    }

    // `src` f16 [n][k] through tensor `ix` ([out][k]), a chunk of rows at a
    // time into `y`; `after(r0, rows)` takes each chunk from there
    let project = |st: &mut State, ix: usize, keep: bool, src: u64, k: usize, alpha: f32, after: &mut dyn FnMut(usize, usize) -> Res<()>| -> Res<usize> {
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
    let (di, hdi, hi, ni) = (d as i32, hd as i32, heads as i32, n as i32);
    let nul = 0u64;
    let rows_grid = (n as u32, 1);
    for (bi, b) in a.blocks.iter().enumerate() {
        let (l, k) = (&lay[bi], keep[bi]);
        let t = tensors(b);
        // ── attention (shift chunk 0, scale 1, gate 2) ──
        unsafe { c.launch(f.norm, rows_grid, 256, 0, kargs!(xb, xn, at(l[0]), at(o_mod), at(o_mod + d), di, EPS))? };
        for (part, dst) in [hq, hk, hv].into_iter().enumerate() {
            let wn = if part < 2 { at(l[2 + part]) } else { nul };
            project(st, t[part], k, xn, d, 1.0, &mut |q0, rows| {
                let (q0i, rr) = (q0 as i32, rows as u32);
                let (cso, sno) = (at(o_cos) + (q0 * hd / 2 * 4) as u64, at(o_sin) + (q0 * hd / 2 * 4) as u64);
                unsafe { c.launch(f.heads, (rr, heads as u32), 32, 0, kargs!(y, dst, wn, cso, sno, hi, hdi, ni, q0i, EPS)) }
            })?;
        }
        let nsi = ns as i32;
        for (mode, (src, dst)) in [(hq, qp), (hk, kp), (hv, vp)].into_iter().enumerate() {
            let mode = mode as i32;
            unsafe { c.launch(h3f.pack, ((heads * ns * hd).div_ceil(256) as u32, 1), 256, 0, kargs!(src, dst, ni, nsi, hi, hdi, hdi, mode))? };
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let one = 1i32;
        unsafe { c.launch(fl, (n.div_ceil(64) as u32, heads as u32), 128, 0, kargs!(qp, kp, vp, ao, ni, nsi, hi, scale, one))? };
        let gate = at(o_mod + 2 * d);
        project(st, t[3], k, ao, d, 1.0, &mut |q0, rows| {
            let tot = (rows * d) as u64;
            let xr = xb + (q0 * d * 4) as u64;
            unsafe { c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y, gate, di, tot)) }
        })?;
        // ── the feed-forward (shift chunk 3, scale 4, gate 5) ──
        unsafe { c.launch(f.norm, rows_grid, 256, 0, kargs!(xb, xn, at(l[1]), at(o_mod + 3 * d), at(o_mod + 4 * d), di, EPS))? };
        let gate = at(o_mod + 5 * d);
        let (ug, _) = unsafe { expand(c, st, model, t[4], w, k, true)? };
        let (o, mid) = unsafe { expand(c, st, model, t[5], w2, k, true)? };
        if o != d || ug != 2 * mid {
            return Err("a feed-forward of the wrong width".into());
        }
        let mut r0 = 0usize;
        while r0 < n {
            let rows = cq.min(n - r0);
            unsafe {
                c.gemm(true, false, ug, rows, d, 1.0, w, d, xn + (r0 * d * 2) as u64, d, 0.0, y, ug, Gemm::Half)?;
                let tot = (rows * mid) as u64;
                let midi = mid as i32;
                c.launch(f.situ, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(y, act, midi, MID_INV, tot))?;
                c.gemm(true, false, d, rows, mid, 1.0 / MID_INV, w2, mid, act, mid, 0.0, y, d, Gemm::Half)?;
                let tot = (rows * d) as u64;
                let xr = xb + (r0 * d * 4) as u64;
                c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y, gate, di, tot))?;
            }
            r0 += rows;
        }
    }
    let mut out = vec![0f32; n * d];
    unsafe { cu::memcpy_dtoh_async(&mut out, xb, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the rows came back with a value that is not finite".into());
    }
    a.x.copy_from_slice(&out);
    Ok(())
}
