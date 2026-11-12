//! The Qwen-Image-2.1 denoiser on CUDA, the `gpu::qi21_*` contract the Metal
//! and Vulkan modules serve: [`qi21_prefill`] runs the prompt prefix (text
//! and condition-image rows, the t = 0 modulation, the block-causal mask)
//! through every block once and keeps each layer's post-rotation keys and
//! values on the card, already in the flash kernel's layouts of the whole
//! sequence `[prefix | target rows]`; [`qi21_step`] writes a denoiser
//! call's own keys and values after them and runs `h3_flash` over the lot.
//!
//! A block: the affine-free LayerNorm times `1 + s1`, the q/k/v products,
//! per-head RMS norm of q and k with the interleaved rotation, attention,
//! the output product and `x += tanh(g1) · y`, the LayerNorm times `1 + s2`,
//! the SwiGLU (`down(silu(gate) · up)`) and `x += tanh(g2) · y`. The weights
//! are the H3 chain's store: in their codec on the card under
//! `CMF_GPU_VRAM_MB`, expanded to an f16 plane for their product. The
//! prefill's attention takes TF32 products around a softmax over each row's
//! visible keys (a block-causal mask the flash kernel does not carry).
//! `CMF_QI21_CUDA=0` keeps the host path.

use super::*;
use crate::gpu::{Qi21Geom, Qi21PrefillArgs};

const Q21_SRC: &str = include_str!("qi21.cu");
/// The SwiGLU middle is stored times this before the down product.
const MID_INV: f32 = 1.0 / 64.0;
/// Programs kept at once (the positive and the negative prompt).
const MAX_PROGS: usize = 2;
/// The bytes of one chunk of products' rows or of the prefill's scores.
const PANEL: usize = 256 << 20;

#[derive(Clone, Copy)]
pub(super) struct Q21Funcs {
    lnmod: cs::CUfunction,
    heads: cs::CUfunction,
    pack: cs::CUfunction,
    addg: cs::CUfunction,
    swiglu: cs::CUfunction,
    softmax: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<Q21Funcs> {
    if let Some(f) = st.h3.q21_funcs {
        return Ok(f);
    }
    let module = compile(Q21_SRC, Vec::new(), c.arch)?;
    let f = Q21Funcs {
        lnmod: func(module, "q21_lnmod")?,
        heads: func(module, "q21_heads")?,
        pack: func(module, "q21_pack")?,
        addg: func(module, "q21_addg")?,
        swiglu: func(module, "q21_swiglu")?,
        softmax: func(module, "q21_softmax_vis")?,
    };
    st.h3.q21_funcs = Some(f);
    Ok(f)
}

/// One prompt's program: its prefix cache and what a step needs.
pub(super) struct Q21Prog {
    key: u64,
    model: Arc<CmfModel>,
    geom: Qi21Geom,
    /// per block: q, k, v, o, gate, up, down
    w: Vec<[usize; 7]>,
    lp: usize,
    n: usize,
    /// the sequence of a step, `lp + n` rows, padded to `ns`
    t: usize,
    ns: usize,
    /// per layer: keys `[heads][t][hd]` in lane order and values in lane
    /// order `[heads][ns][hd]`, f16; the prefix rows written by the prefill
    kv: Vec<(Buf, Buf)>,
    /// norms `[layer][q | k]`, the target rows' cos and sin, img_in, proj_out
    consts: Buf,
    o_cos: usize,
    o_sin: usize,
    o_in: usize,
    o_out: usize,
    keep: Vec<bool>,
}

/// Which blocks keep their weights on the card (the first that fit).
fn keep_plan(st: &State, model: &Arc<CmfModel>, w: &[[usize; 7]]) -> Res<Vec<bool>> {
    let uid = model.uid();
    let (free, total_mem) = cu::mem_get_info().map_err(derr)?;
    let reserve: i64 = (160 << 20) + (128 << 20);
    let mut room: i64 = match ceiling() {
        Some(b) => b as i64 - (total_mem - free) as i64,
        None => free as i64,
    } - reserve;
    let mut keep = vec![false; w.len()];
    let mut all_before = true;
    for (i, b) in w.iter().enumerate() {
        let mut need = 0u64;
        for &ix in b {
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
    Ok(keep)
}

/// `[s1 | tanh g1 | s2 | tanh g2]`, the host path's own f32 `tanh`.
fn mods_dev(mods: &[f32], h: usize) -> Vec<f32> {
    mods[..4 * h].iter().enumerate().map(|(i, &v)| if (h..2 * h).contains(&i) || i >= 3 * h { v.tanh() } else { v }).collect()
}

pub fn qi21_prefill(a: &Qi21PrefillArgs) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_QI21_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    st.h3.q21.retain(|p| p.key != a.key);
    match unsafe { prefill(c, &mut st, a) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            st.h3.q21.retain(|p| p.key != a.key);
            decline("qwen-image-2.1 prefill", &e);
            false
        }
    }
}

pub fn qi21_step(key: u64, xtok: &[f32], mods: &[f32], fs: &[f32], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    let Some(i) = st.h3.q21.iter().position(|p| p.key == key) else { return false };
    let prog = st.h3.q21.remove(i);
    let r = unsafe { step(c, &mut st, &prog, xtok, mods, fs, out) };
    st.h3.q21.push(prog);
    match r {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("qwen-image-2.1 step", &e);
            false
        }
    }
}

pub fn qi21_release_key(key: u64) {
    if let Some(c) = ctx()
        && let Ok(mut st) = c.st.lock()
    {
        let _ = c.sync();
        st.h3.q21.retain(|p| p.key != key);
    }
}

pub fn qi21_release() {
    if let Some(c) = ctx()
        && let Ok(mut st) = c.st.lock()
    {
        let _ = c.sync();
        st.h3.q21.clear();
    }
}

/// The working buffers of a pass over `rows` rows of a sequence `t` long.
struct Work {
    x: u64,
    xn: u64,
    y: u64,
    y2: u64,
    hq: u64,
    hk: u64,
    hv: u64,
    act: u64,
    w: [u64; 3],
    mods: u64,
    cq: usize,
}

fn work(c: &Ctx, st: &mut State, g: &Qi21Geom, rows: usize) -> Res<Work> {
    let (h, inter) = (g.hidden, g.inter);
    let cq = (PANEL / (inter.max(h) * 4)).clamp(16, 4096).min(rows.max(1));
    Ok(Work {
        x: st.h3.buf(c, "q21 x", rows * h * 4)?,
        xn: st.h3.buf(c, "q21 xn", rows * h * 2)?,
        y: st.h3.buf(c, "q21 y", cq * inter.max(h) * 4)?,
        y2: st.h3.buf(c, "q21 y2", cq * inter * 4)?,
        hq: st.h3.buf(c, "q21 hq", rows * h * 4)?,
        hk: st.h3.buf(c, "q21 hk", rows * h * 4)?,
        hv: st.h3.buf(c, "q21 hv", rows * h * 4)?,
        act: st.h3.buf(c, "q21 act", cq * inter * 2)?,
        w: [
            st.h3.buf(c, "q21 w", inter.max(h) * h * 2)?,
            st.h3.buf(c, "q21 w2", inter.max(h) * h * 2)?,
            st.h3.buf(c, "q21 w3", inter.max(h) * h * 2)?,
        ],
        mods: st.h3.buf(c, "q21 mods", 4 * h * 4 + 64)?,
        cq,
    })
}

/// What a block reads besides the work buffers.
struct BlockIo<'a> {
    rows: usize,
    /// the block's q/k norm weights on the card
    norm: u64,
    /// cos and sin of these rows `[rows][hd / 2]`
    cos: u64,
    sin: u64,
    /// the flash layouts' sequence: rows written at `r0`
    r0: usize,
    t: usize,
    ns: usize,
    kv: &'a (Buf, Buf),
}

/// The first half of a block: its normed rows, q/k/v into the head-major
/// planes (q, k normed and rotated), and k, v into the layer's cache at row
/// `r0` (q into `qp` too when given).
unsafe fn qkv(c: &Ctx, st: &mut State, f: &Q21Funcs, model: &Arc<CmfModel>, g: &Qi21Geom, w: &[usize; 7], keep: bool, wk: &Work, io: &BlockIo, qp: Option<u64>) -> Res<()> {
    let (h, nh, hd) = (g.hidden, g.nh, g.hd);
    let rows = io.rows;
    unsafe {
        c.launch(f.lnmod, (rows as u32, 1), 256, 0, kargs!(wk.x, wk.xn, 0u64, wk.mods, 1i32, h as i32, g.eps, 1.0f32))?;
    }
    for (part, dst) in [wk.hq, wk.hk, wk.hv].into_iter().enumerate() {
        let (out, kk) = unsafe { expand(c, st, model, w[part], wk.w[0], keep, true)? };
        if out != h || kk != h {
            return Err("a q/k/v projection of the wrong shape".into());
        }
        let wn = if part < 2 { io.norm + (part * hd * 4) as u64 } else { 0 };
        let mut r0 = 0usize;
        while r0 < rows {
            let n = wk.cq.min(rows - r0);
            let off = (r0 * hd / 2 * 4) as u64;
            let (cs_, sn_) = if part < 2 { (io.cos + off, io.sin + off) } else { (0, 0) };
            unsafe {
                c.gemm(true, false, h, n, h, 1.0, wk.w[0], h, wk.xn + (r0 * h * 2) as u64, h, 0.0, wk.y, h, Gemm::Half)?;
                let (ti, r0i) = (rows as i32, r0 as i32);
                c.launch(f.heads, (n as u32, nh as u32), 32, 0, kargs!(wk.y, dst, wn, cs_, sn_, ti, r0i, nh as i32, hd as i32, g.eps))?;
            }
            r0 += n;
        }
    }
    // into the flash layouts of the whole sequence
    let total = (nh * rows * hd) as u64;
    let grid = (total.div_ceil(256) as u32, 1);
    let (ri, r0i, ti, nsi, nhi, hdi) = (rows as i32, io.r0 as i32, io.t as i32, io.ns as i32, nh as i32, hd as i32);
    unsafe {
        let (kp, vp) = (io.kv.0.ptr, io.kv.1.ptr);
        c.launch(f.pack, grid, 256, 0, kargs!(wk.hk, kp, ri, r0i, ti, nsi, nhi, hdi, 1i32))?;
        c.launch(f.pack, grid, 256, 0, kargs!(wk.hv, vp, ri, r0i, ti, nsi, nhi, hdi, 2i32))?;
        if let Some(qp) = qp {
            c.launch(f.pack, grid, 256, 0, kargs!(wk.hq, qp, ri, r0i, ti, nsi, nhi, hdi, 0i32))?;
        }
    }
    Ok(())
}

/// The second half: the output product from `ao` (f16 `[rows][H]`), the
/// gated residual, the SwiGLU and its gated residual.
unsafe fn tail(c: &Ctx, st: &mut State, f: &Q21Funcs, model: &Arc<CmfModel>, g: &Qi21Geom, w: &[usize; 7], keep: bool, wk: &Work, rows: usize, ao: u64) -> Res<()> {
    let (h, inter) = (g.hidden, g.inter);
    let hi = h as i32;
    let (g1, s2, g2) = (wk.mods + (h * 4) as u64, wk.mods + (2 * h * 4) as u64, wk.mods + (3 * h * 4) as u64);
    let (out, kk) = unsafe { expand(c, st, model, w[3], wk.w[0], keep, true)? };
    if out != h || kk != h {
        return Err("an output projection of the wrong shape".into());
    }
    let mut r0 = 0usize;
    while r0 < rows {
        let n = wk.cq.min(rows - r0);
        unsafe {
            c.gemm(true, false, h, n, h, 1.0, wk.w[0], h, ao + (r0 * h * 2) as u64, h, 0.0, wk.y, h, Gemm::Half)?;
            let tot = (n * h) as u64;
            c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(wk.x + (r0 * h * 4) as u64, wk.y, g1, hi, tot))?;
        }
        r0 += n;
    }
    unsafe { c.launch(f.lnmod, (rows as u32, 1), 256, 0, kargs!(wk.x, wk.xn, 0u64, s2, 1i32, hi, g.eps, 1.0f32))? };
    let (gi, _) = unsafe { expand(c, st, model, w[4], wk.w[0], keep, true)? };
    let (ui, _) = unsafe { expand(c, st, model, w[5], wk.w[1], keep, true)? };
    let (dn, di) = unsafe { expand(c, st, model, w[6], wk.w[2], keep, true)? };
    if gi != inter || ui != inter || dn != h || di != inter {
        return Err("a feed-forward of the wrong shape".into());
    }
    let mut r0 = 0usize;
    while r0 < rows {
        let n = wk.cq.min(rows - r0);
        let xs = wk.xn + (r0 * h * 2) as u64;
        unsafe {
            c.gemm(true, false, inter, n, h, 1.0, wk.w[0], h, xs, h, 0.0, wk.y, inter, Gemm::Half)?;
            c.gemm(true, false, inter, n, h, 1.0, wk.w[1], h, xs, h, 0.0, wk.y2, inter, Gemm::Half)?;
            let tot = (n * inter) as u64;
            c.launch(f.swiglu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(wk.y, wk.y2, wk.act, MID_INV, tot))?;
            c.gemm(true, false, h, n, inter, 1.0 / MID_INV, wk.w[2], inter, wk.act, inter, 0.0, wk.y, h, Gemm::Half)?;
            let tot = (n * h) as u64;
            c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(wk.x + (r0 * h * 4) as u64, wk.y, g2, hi, tot))?;
        }
        r0 += n;
    }
    Ok(())
}

unsafe fn prefill(c: &Ctx, st: &mut State, a: &Qi21PrefillArgs) -> Res<()> {
    let g = a.geom;
    let (h, nh, hd, lp, n) = (g.hidden, g.nh, g.hd, a.lp, a.n);
    if nh * hd != h || !hd.is_multiple_of(32) || g.in_ch != 64 || a.blocks.is_empty() || lp == 0 || n == 0 {
        return Err(format!("geometry {g:?} outside the chain"));
    }
    if a.x.len() != lp * h || a.vis.len() != lp || a.mods0.len() < 4 * h || a.img_in.len() != h * 64 || a.proj_out.len() != 64 * h {
        return Err("prefix rows, mask, modulation or projections of the wrong size".into());
    }
    if a.rope_p.0.len() != lp * hd / 2 || a.rope_p.1.len() != lp * hd / 2 || a.rope_t.0.len() != n * hd / 2 || a.rope_t.1.len() != n * hd / 2 {
        return Err("rotation tables of the wrong size".into());
    }
    if a.blocks.iter().any(|b| b.norm_q.len() != hd || b.norm_k.len() != hd) {
        return Err("a q/k norm of the wrong width".into());
    }
    let f = funcs(st, c)?;
    st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;
    let model = a.model.clone();
    let w: Vec<[usize; 7]> = a.blocks.iter().map(|b| b.w).collect();
    let keep = keep_plan(st, &model, &w)?;
    let (t, ns) = (lp + n, (lp + n).next_multiple_of(16));

    // ── the program ──
    while st.h3.q21.len() >= MAX_PROGS {
        st.h3.q21.remove(0);
    }
    let mut cv: Vec<f32> = Vec::new();
    for b in a.blocks {
        cv.extend_from_slice(b.norm_q);
        cv.extend_from_slice(b.norm_k);
    }
    let o_cos = cv.len();
    cv.extend_from_slice(a.rope_t.0);
    let o_sin = cv.len();
    cv.extend_from_slice(a.rope_t.1);
    let o_in = cv.len();
    cv.extend_from_slice(a.img_in);
    let o_out = cv.len();
    cv.extend_from_slice(a.proj_out);
    let consts = Buf::direct(cv.len() * 4)?;
    let mut kv = Vec::with_capacity(w.len());
    for _ in 0..w.len() {
        let k = Buf::direct(nh * t * hd * 2)?;
        let v = Buf::direct(nh * ns * hd * 2 + 64)?;
        unsafe { cu::memset_d8_async(v.ptr, 0, v.bytes, c.stream).map_err(derr)? };
        kv.push((k, v));
    }
    unsafe { cu::memcpy_htod_async(consts.ptr, &cv, c.stream).map_err(derr)? };
    let prog = Q21Prog { key: a.key, model: model.clone(), geom: g, w: w.clone(), lp, n, t, ns, kv, consts, o_cos, o_sin, o_in, o_out, keep };

    // ── the prefix through every block ──
    let wk = work(c, st, &g, lp)?;
    let ropep = st.h3.buf(c, "q21 ropep", lp * hd * 4)?;
    let vis = st.h3.buf(c, "q21 vis", lp * 4 + 64)?;
    let aof = st.h3.buf(c, "q21 aof", lp * h * 4)?;
    let ao = st.h3.buf(c, "q21 ao", lp * h * 2)?;
    let cqs = (PANEL / (lp * 4)).clamp(1, lp);
    let sb = st.h3.buf(c, "q21 sc", cqs * lp * 4)?;
    let to_half = st.h3.funcs(c)?.to_half;
    unsafe {
        cu::memcpy_htod_async(wk.x, a.x, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(wk.mods, &mods_dev(a.mods0, h), c.stream).map_err(derr)?;
        cu::memcpy_htod_async(ropep, a.rope_p.0, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(ropep + (lp * hd / 2 * 4) as u64, a.rope_p.1, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(vis, a.vis, c.stream).map_err(derr)?;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let nb = w.len();
    for (l, wl) in w.iter().enumerate() {
        let io = BlockIo { rows: lp, norm: prog.consts.ptr + (l * 2 * hd * 4) as u64, cos: ropep, sin: ropep + (lp * hd / 2 * 4) as u64, r0: 0, t, ns, kv: &prog.kv[l] };
        unsafe { qkv(c, st, &f, &model, &g, wl, prog.keep[l], &wk, &io, None)? };
        if l + 1 == nb {
            break;
        }
        // attention over each row's visible keys, a head at a time
        for hh in 0..nh {
            let (qh, kh, vh) = (wk.hq + (hh * lp * hd * 4) as u64, wk.hk + (hh * lp * hd * 4) as u64, wk.hv + (hh * lp * hd * 4) as u64);
            let mut q0 = 0usize;
            while q0 < lp {
                let rows = cqs.min(lp - q0);
                unsafe {
                    c.gemm(true, false, lp, rows, hd, scale, kh, hd, qh + (q0 * hd * 4) as u64, hd, 0.0, sb, lp, Gemm::Tf32)?;
                    c.launch(f.softmax, (rows as u32, 1), 256, 0, kargs!(sb, lp as i32, vis, q0 as i32))?;
                    c.gemm(false, false, hd, rows, lp, 1.0, vh, hd, sb, lp, 0.0, aof + ((q0 * h + hh * hd) * 4) as u64, h, Gemm::Tf32)?;
                }
                q0 += rows;
            }
        }
        let nv = (lp * h) as u64;
        unsafe {
            c.launch(to_half, (nv.div_ceil(256) as u32, 1), 256, 0, kargs!(aof, ao, nv))?;
            tail(c, st, &f, &model, &g, wl, prog.keep[l], &wk, lp, ao)?;
        }
    }
    c.sync()?;
    st.h3.q21.push(prog);
    Ok(())
}

unsafe fn step(c: &Ctx, st: &mut State, p: &Q21Prog, xtok: &[f32], mods: &[f32], fs: &[f32], out: &mut [f32]) -> Res<()> {
    let g = p.geom;
    let (h, nh, hd, n) = (g.hidden, g.nh, g.hd, p.n);
    if xtok.len() != n * 64 || out.len() != n * 64 || mods.len() < 4 * h || fs.len() != h {
        return Err("tokens, modulation or output of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let fl = st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;
    let wk = work(c, st, &g, n)?;
    let tok = st.h3.buf(c, "q21 tok", n * 64 * 4)?;
    let fsb = st.h3.buf(c, "q21 fs", h * 4)?;
    let qp = st.h3.buf(c, "q21 qp", nh * p.t * hd * 2)?;
    let ao = st.h3.buf(c, "q21 ao", p.t * h * 2)?;
    let xf = st.h3.buf(c, "q21 xf", n * h * 4)?;
    let ob = st.h3.buf(c, "q21 out", n * 64 * 4)?;
    let at = |o: usize| p.consts.ptr + (o * 4) as u64;
    unsafe {
        cu::memcpy_htod_async(tok, xtok, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(wk.mods, &mods_dev(mods, h), c.stream).map_err(derr)?;
        cu::memcpy_htod_async(fsb, fs, c.stream).map_err(derr)?;
        // the prefix rows' queries are never read back; zero them once
        cu::memset_d8_async(qp, 0, nh * p.t * hd * 2, c.stream).map_err(derr)?;
        c.gemm(true, false, h, n, 64, 1.0, at(p.o_in), 64, tok, 64, 0.0, wk.x, h, Gemm::F32)?;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let (ti, nsi, nhi) = (p.t as i32, p.ns as i32, nh as i32);
    for (l, wl) in p.w.iter().enumerate() {
        let io = BlockIo { rows: n, norm: p.consts.ptr + (l * 2 * hd * 4) as u64, cos: at(p.o_cos), sin: at(p.o_sin), r0: p.lp, t: p.t, ns: p.ns, kv: &p.kv[l] };
        unsafe {
            qkv(c, st, &f, &p.model, &g, wl, p.keep[l], &wk, &io, Some(qp))?;
            c.launch(fl, (p.t.div_ceil(64) as u32, nh as u32), 128, 0, kargs!(qp, p.kv[l].0.ptr, p.kv[l].1.ptr, ao, ti, nsi, nhi, scale, 1i32))?;
            tail(c, st, &f, &p.model, &g, wl, p.keep[l], &wk, n, ao + (p.lp * h * 2) as u64)?;
        }
    }
    unsafe {
        c.launch(f.lnmod, (n as u32, 1), 256, 0, kargs!(wk.x, 0u64, xf, fsb, 0i32, h as i32, g.eps, 1.0f32))?;
        c.gemm(true, false, 64, n, h, 1.0, at(p.o_out), h, xf, h, 0.0, ob, 64, Gemm::F32)?;
        cu::memcpy_dtoh_async(out, ob, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the step produced a value that is not finite".into());
    }
    Ok(())
}
