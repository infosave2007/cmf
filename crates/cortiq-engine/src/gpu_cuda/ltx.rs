//! LTX-2.5 on CUDA: the 48 blocks of a denoising step as one resident chain.
//!
//! Both residual streams (video and audio) go up once and come back once.
//! Between them every block runs on the card: the timestep-modulated norms
//! into f16, the projections as cuBLAS GEMMs on the tensor cores (f16
//! operands, f32 accumulation), q/k norm with the split rotation into
//! head-major planes, attention as two batched GEMMs a chunk of query rows
//! with a softmax between them, the per-head gate, the gated residual, and
//! the GELU feed-forward the same way. A block is six attentions — self and
//! prompt for each stream, and the two directions of their fusion, which
//! both read the streams as they were before it.
//!
//! Weights use the H3 chain's store (`expand`): on the card in their codec
//! (q4tp, q8_2f) and expanded to a dense f16 plane only while their GEMM
//! runs; blocks that do not fit under `CMF_GPU_VRAM_MB` pass through the
//! stream slot every step.

use super::*;
use crate::gpu::{LtxAttn, LtxChain, LtxLin, LtxStream, Vae3dStep};

const LTX_SRC: &str = include_str!("ltx.cu");
/// The norms' epsilon (the reference's 1e-6).
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
pub(super) struct LxFuncs {
    ada: cs::CUfunction,
    qk: cs::CUfunction,
    v: cs::CUfunction,
    softmax: cs::CUfunction,
    join: cs::CUfunction,
    addg: cs::CUfunction,
    gelu: cs::CUfunction,
    copy: cs::CUfunction,
    norm_silu: cs::CUfunction,
    im2col3d: cs::CUfunction,
    bias: cs::CUfunction,
    d2s: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<LxFuncs> {
    if let Some(f) = st.h3.ltx {
        return Ok(f);
    }
    let module = compile(LTX_SRC, Vec::new(), c.arch)?;
    let f = LxFuncs {
        ada: func(module, "lx_ada")?,
        qk: func(module, "lx_qk")?,
        v: func(module, "lx_v")?,
        softmax: func(module, "lx_softmax")?,
        join: func(module, "lx_join")?,
        addg: func(module, "lx_addg")?,
        gelu: func(module, "lx_gelu")?,
        copy: func(module, "lx_copy")?,
        norm_silu: func(module, "vx_norm_silu")?,
        im2col3d: func(module, "vx_im2col3d")?,
        bias: func(module, "vx_bias")?,
        d2s: func(module, "vx_d2s")?,
    };
    st.h3.ltx = Some(f);
    Ok(f)
}

/// Where an attention's constants sit in the one buffer they are uploaded
/// in (offsets in floats; `usize::MAX`: none).
#[derive(Clone, Copy)]
struct AOff {
    qb: usize,
    kb: usize,
    vb: usize,
    ob: usize,
    qn: usize,
    kn: usize,
    gb: usize,
}

#[derive(Clone, Copy)]
struct SOff {
    a1: AOff,
    a2: AOff,
    fb1: usize,
    fb2: usize,
    mods: usize,
    ctx: usize,
}

#[derive(Clone, Copy)]
struct BOff {
    v: SOff,
    a: SOff,
    a2v: AOff,
    v2a: AOff,
    xv: usize,
    xa: usize,
    ga: usize,
    gv: usize,
}

/// The small arrays of a call (biases, norm weights, tables), in the order
/// they are uploaded.
struct Consts(Vec<f32>);

impl Consts {
    fn put(&mut self, x: &[f32]) -> usize {
        if x.is_empty() {
            return usize::MAX;
        }
        let at = self.0.len();
        self.0.extend_from_slice(x);
        at
    }

    fn attn(&mut self, t: &LtxAttn<'_>) -> AOff {
        AOff {
            qb: self.put(t.q.bias),
            kb: self.put(t.k.bias),
            vb: self.put(t.v.bias),
            ob: self.put(t.o.bias),
            qn: self.put(t.q_norm),
            kn: self.put(t.k_norm),
            gb: t.gate.as_ref().map_or(usize::MAX, |g| self.put(g.bias)),
        }
    }

    fn stream(&mut self, s: &LtxStream<'_>) -> SOff {
        SOff {
            a1: self.attn(&s.attn1),
            a2: self.attn(&s.attn2),
            fb1: self.put(s.ff_in.bias),
            fb2: self.put(s.ff_out.bias),
            mods: self.put(s.mods),
            ctx: self.put(s.ctx_mod),
        }
    }
}

/// A rotation table on the card: cos, sin and the half width of a group.
#[derive(Clone, Copy)]
struct Pe {
    cos: u64,
    sin: u64,
    half: usize,
}

/// Where an attention's output goes: `x += g * (out + bias)`, the gate a
/// chunk of the row's table row (or one).
#[derive(Clone, Copy)]
struct Add {
    x: u64,
    tab: u64,
    rowidx: u64,
    width: usize,
    stride: usize,
    go: usize,
}

/// The buffers every stage of a call shares.
#[derive(Clone, Copy)]
struct Work {
    /// The plane buffer: one weight, or the two of a feed-forward.
    w: u64,
    /// A chunk of a product's result.
    y: u64,
    /// The constants.
    cb: u64,
    /// Rows a chunk of a product, and bytes a chunk of score rows.
    cq: usize,
    panel: usize,
}

pub fn ltx_chain(job: &LtxChain<'_>, vx: &mut [f32], ax: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { chain(c, &mut st, job, vx, ax) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("ltx chain", &e);
            false
        }
    }
}

fn lins<'a, 'b>(a: &'b LtxAttn<'a>) -> impl Iterator<Item = &'b LtxLin<'a>> {
    [&a.q, &a.k, &a.v, &a.o].into_iter().chain(a.gate.as_ref())
}

fn stream_lins<'a, 'b>(s: &'b LtxStream<'a>) -> impl Iterator<Item = &'b LtxLin<'a>> {
    lins(&s.attn1).chain(lins(&s.attn2)).chain([&s.ff_in, &s.ff_out])
}

fn check_attn(a: &LtxAttn<'_>, qcols: usize, kcols: usize, out: usize) -> Res<()> {
    let inner = a.heads * a.dh;
    let ok = a.q.rows == inner
        && a.k.rows == inner
        && a.v.rows == inner
        && a.q.cols == qcols
        && a.k.cols == kcols
        && a.v.cols == kcols
        && a.o.cols == inner
        && a.o.rows == out
        && a.q_norm.len() == inner
        && a.k_norm.len() == inner
        && a.gate.as_ref().is_none_or(|g| g.rows == a.heads && g.cols == qcols);
    if !ok || a.heads == 0 || a.dh == 0 {
        return Err("an attention of the wrong shape".into());
    }
    for l in lins(a) {
        if !l.bias.is_empty() && l.bias.len() != l.rows {
            return Err("a bias of the wrong width".into());
        }
    }
    Ok(())
}

/// A projection's weight as a dense f16 plane: a container matrix expands
/// into `w`, a dense one is kept on the card as it is. Returns where the
/// plane is.
unsafe fn plane(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, l: &LtxLin<'_>, keep: bool, w: u64) -> Res<u64> {
    match l.w {
        Some(idx) => {
            let (r, co) = unsafe { expand(c, st, model, idx, w, keep, true)? };
            if r != l.rows || co != l.cols {
                return Err("a projection of the wrong shape".into());
            }
            Ok(w)
        }
        None => {
            if l.dense.len() != l.rows * l.cols {
                return Err("a dense projection of the wrong size".into());
            }
            unsafe { host_half(c, st, l.dense) }
        }
    }
}

/// `src` f16 `[n][cols]` through a projection, a chunk of rows at a time:
/// each chunk's result lands in `wk.y` (f32 `[rows][l.rows]`) and `after`
/// takes it from there.
#[allow(clippy::too_many_arguments)]
unsafe fn project(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    l: &LtxLin<'_>,
    keep: bool,
    wk: &Work,
    src: u64,
    n: usize,
    mut after: impl FnMut(usize, usize) -> Res<()>,
) -> Res<()> {
    let w = unsafe { plane(c, st, model, l, keep, wk.w)? };
    let mut r0 = 0usize;
    while r0 < n {
        let rows = wk.cq.min(n - r0);
        unsafe {
            c.gemm(true, false, l.rows, rows, l.cols, 1.0, w, l.cols, src + (r0 * l.cols * 2) as u64, l.cols, 0.0, wk.y, l.rows, Gemm::Half)?;
        }
        after(r0, rows)?;
        r0 += rows;
    }
    Ok(())
}

/// `rows` rows of `x` (f32, `width` wide) into f16 at `out`: the RMS norm
/// where `rms`, then `* (1 + scale) + shift` from chunks `sc` and `sh` of
/// each row's table row.
#[allow(clippy::too_many_arguments)]
unsafe fn ada(
    c: &Ctx,
    f: &LxFuncs,
    x: u64,
    out: u64,
    rows: usize,
    width: usize,
    tab: u64,
    rowidx: u64,
    stride: usize,
    sh: usize,
    sc: usize,
    rms: bool,
) -> Res<()> {
    let (wi, si, shi, sci, ri) = (width as i32, stride as i32, sh as i32, sc as i32, rms as i32);
    unsafe { c.launch(f.ada, (rows as u32, 1), 256, 0, kargs!(x, out, tab, rowidx, wi, si, shi, sci, ri, EPS)) }
}

/// One attention: queries from `xq` (f16 `[nq][q.cols]`), keys and values
/// from `xk` (f16 `[nk][k.cols]`), the result added into the stream `add`
/// names.
#[allow(clippy::too_many_arguments)]
unsafe fn attention(
    c: &Ctx,
    st: &mut State,
    f: &LxFuncs,
    model: &Arc<CmfModel>,
    keep: bool,
    a: &LtxAttn<'_>,
    o: &AOff,
    wk: &Work,
    xq: u64,
    nq: usize,
    xk: u64,
    nk: usize,
    pe_q: Option<Pe>,
    pe_k: Option<Pe>,
    mask: u64,
    add: &Add,
) -> Res<()> {
    let (heads, dh) = (a.heads, a.dh);
    let inner = heads * dh;
    let at = |off: usize| -> u64 { if off == usize::MAX { 0 } else { wk.cb + (off * 4) as u64 } };
    let qp = st.h3.buf(c, "lx q", nq * inner * 2)?;
    let kp = st.h3.buf(c, "lx k", nk * inner * 2)?;
    let vp = st.h3.buf(c, "lx v", nk * inner * 2 + 64)?;
    let (ii, dhi) = (inner as i32, dh as i32);
    let y = wk.y;

    // ── q and k: norm, rotation, head-major planes; v: dim-major planes ──
    let qk = |pe: Option<Pe>, bias: u64, nw: u64, planes: u64, n: usize, r0: usize, rows: usize| -> Res<()> {
        let (cos, sin, half) = match pe {
            Some(p) => (p.cos, p.sin, p.half as i32),
            None => (0u64, 0u64, 1i32),
        };
        let (r0i, ni) = (r0 as i32, n as i32);
        unsafe { c.launch(f.qk, (rows as u32, 1), 256, 0, kargs!(y, bias, nw, cos, sin, planes, ii, half, dhi, r0i, ni, EPS)) }
    };
    unsafe {
        let (qb, qn, kb, kn) = (at(o.qb), at(o.qn), at(o.kb), at(o.kn));
        project(c, st, model, &a.q, keep, wk, xq, nq, |r0, rows| qk(pe_q, qb, qn, qp, nq, r0, rows))?;
        project(c, st, model, &a.k, keep, wk, xk, nk, |r0, rows| qk(pe_k, kb, kn, kp, nk, r0, rows))?;
        let vb = at(o.vb);
        let nki = nk as i32;
        project(c, st, model, &a.v, keep, wk, xk, nk, |r0, rows| {
            let (r0i, total) = (r0 as i32, (rows * inner) as u64);
            c.launch(f.v, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(y, vb, vp, ii, dhi, r0i, nki, total))
        })?;
    }

    // ── the head gates' logits, for every query row ──
    let gl = match &a.gate {
        Some(g) => {
            let gl = st.h3.buf(c, "lx gl", nq * heads * 4)?;
            let gw = unsafe { plane(c, st, model, g, keep, wk.w)? };
            let mut r0 = 0usize;
            while r0 < nq {
                let rows = wk.cq.min(nq - r0);
                unsafe {
                    c.gemm(
                        true,
                        false,
                        heads,
                        rows,
                        g.cols,
                        1.0,
                        gw,
                        g.cols,
                        xq + (r0 * g.cols * 2) as u64,
                        g.cols,
                        0.0,
                        gl + (r0 * heads * 4) as u64,
                        heads,
                        Gemm::Half,
                    )?;
                }
                r0 += rows;
            }
            gl
        }
        None => 0,
    };

    // ── scores, weights, values, the output projection, the residual ──
    let wo = unsafe { plane(c, st, model, &a.o, keep, wk.w)? };
    let ca = (wk.panel / (heads * nk * 6)).clamp(8, 4096).min(nq);
    let sp = st.h3.buf(c, "lx s", heads * ca * nk * 4)?;
    let pp = st.h3.buf(c, "lx p", heads * ca * nk * 2)?;
    let op = st.h3.buf(c, "lx o", heads * ca * dh * 4)?;
    let zp = st.h3.buf(c, "lx z", heads * ca * 4 + 64)?;
    let ao = st.h3.buf(c, "lx ao", ca * inner * 2)?;
    let out = a.o.rows;
    let y2 = st.h3.buf(c, "lx y2", ca * out * 4)?;
    let scale = 1.0 / (dh as f32).sqrt();
    let (hi, nki, wi, si, goi) = (heads as i32, nk as i32, add.width as i32, add.stride as i32, add.go as i32);
    let (ob, gb) = (at(o.ob), at(o.gb));
    let mut a0 = 0usize;
    while a0 < nq {
        let rows = ca.min(nq - a0);
        unsafe {
            // S [heads][rows][nk] = scale * Q · Kᵀ
            gemm_many(c, st, true, nk, rows, dh, scale, kp, dh, nk * dh, qp + (a0 * dh * 2) as u64, dh, nq * dh, sp, nk, rows * nk, heads)?;
            c.launch(f.softmax, ((heads * rows) as u32, 1), 256, 0, kargs!(sp, pp, zp, mask, nki, P_SCALE))?;
            // O [heads][rows][dh] = P · V (the values dim-major: [dh][nk] a plane)
            gemm_many(c, st, true, dh, rows, nk, 1.0 / P_SCALE, vp, nk, dh * nk, pp, nk, rows * nk, op, dh, rows * dh, heads)?;
            let total = (rows * inner) as u64;
            let glr = if gl == 0 { 0 } else { gl + (a0 * heads * 4) as u64 };
            let ri = rows as i32;
            c.launch(f.join, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(op, zp, glr, gb, ao, ri, hi, dhi, total))?;
            c.gemm(true, false, out, rows, inner, 1.0, wo, inner, ao, inner, 0.0, y2, out, Gemm::Half)?;
            let total = (rows * out) as u64;
            let xr = add.x + (a0 * add.width * 4) as u64;
            let rr = if add.rowidx == 0 { 0 } else { add.rowidx + (a0 * 4) as u64 };
            c.launch(f.addg, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y2, ob, add.tab, rr, wi, si, goi, total))?;
        }
        a0 += rows;
    }
    Ok(())
}

/// A stream's feed-forward over its `n` rows at `x`.
#[allow(clippy::too_many_arguments)]
unsafe fn feed_forward(
    c: &Ctx,
    st: &mut State,
    f: &LxFuncs,
    model: &Arc<CmfModel>,
    keep: bool,
    s: &LtxStream<'_>,
    o: &SOff,
    wk: &Work,
    x: u64,
    n: usize,
    dim: usize,
    rowidx: u64,
) -> Res<()> {
    let wide = s.ff_in.rows;
    if s.ff_in.cols != dim || s.ff_out.cols != wide || s.ff_out.rows != dim {
        return Err("a feed-forward of the wrong shape".into());
    }
    let at = |off: usize| -> u64 { if off == usize::MAX { 0 } else { wk.cb + (off * 4) as u64 } };
    let (w1, w2) = unsafe {
        (
            plane(c, st, model, &s.ff_in, keep, wk.w)?,
            plane(c, st, model, &s.ff_out, keep, wk.w + (wide * dim * 2) as u64)?,
        )
    };
    let xn = st.h3.buf(c, "lx xn", wk.cq * dim * 2)?;
    let act = st.h3.buf(c, "lx act", wk.cq * wide * 2)?;
    let y2 = st.h3.buf(c, "lx y2f", wk.cq * dim * 4)?;
    let tab = at(o.mods);
    let (di, wdi, si, goi) = (dim as i32, wide as i32, (9 * dim) as i32, (5 * dim) as i32);
    let (b1, b2) = (at(o.fb1), at(o.fb2));
    let mut r0 = 0usize;
    while r0 < n {
        let rows = wk.cq.min(n - r0);
        let xr = x + (r0 * dim * 4) as u64;
        let rr = rowidx + (r0 * 4) as u64;
        unsafe {
            ada(c, f, xr, xn, rows, dim, tab, rr, 9 * dim, 3 * dim, 4 * dim, true)?;
            c.gemm(true, false, wide, rows, dim, 1.0, w1, dim, xn, dim, 0.0, wk.y, wide, Gemm::Half)?;
            let total = (rows * wide) as u64;
            c.launch(f.gelu, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(wk.y, b1, act, wdi, ACT_INV, total))?;
            c.gemm(true, false, dim, rows, wide, 1.0 / ACT_INV, w2, wide, act, wide, 0.0, y2, dim, Gemm::Half)?;
            let total = (rows * dim) as u64;
            c.launch(f.addg, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y2, b2, tab, rr, di, si, goi, total))?;
        }
        r0 += rows;
    }
    Ok(())
}

unsafe fn chain(c: &Ctx, st: &mut State, job: &LtxChain<'_>, vx: &mut [f32], ax: &mut [f32]) -> Res<()> {
    let (v, a) = (&job.video, &job.audio);
    let (n, m, dim, adim) = (v.n, a.n, v.dim, a.dim);
    if n == 0 || m == 0 || job.blocks.is_empty() || vx.len() != n * dim || ax.len() != m * adim {
        return Err("empty stack or a stream of the wrong size".into());
    }
    let side_ok = |s: &crate::gpu::LtxSide<'_>, xw: usize| {
        s.rows.len() == s.n
            && s.xrows.len() == s.n
            && s.pe.0.len() == s.n * s.dim / 2
            && s.pe.1.len() == s.pe.0.len()
            && s.xpe.0.len() == s.n * xw / 2
            && s.xpe.1.len() == s.xpe.0.len()
            && s.ctx.len() == s.ctx_len * s.dim
            && s.ctx_len > 0
            && (s.mask.is_empty() || s.mask.len() == s.ctx_len)
    };
    let b0 = &job.blocks[0];
    let xw = b0.a2v.heads * b0.a2v.dh;
    if !side_ok(v, xw) || !side_ok(a, xw) {
        return Err("a stream's tables are of the wrong size".into());
    }
    let (dv, da) = (b0.video.mods.len() / (9 * dim), b0.audio.mods.len() / (9 * adim));
    let (xdv, xda) = (b0.xmods_v.len() / (4 * dim), b0.xmods_a.len() / (4 * adim));
    let rows_ok = |r: &[u32], d: usize| d > 0 && r.iter().all(|&i| (i as usize) < d);
    if !rows_ok(v.rows, dv) || !rows_ok(a.rows, da) || !rows_ok(v.xrows, xdv) || !rows_ok(a.xrows, xda) {
        return Err("a token names a modulation row past its table".into());
    }
    let mut plane_max = 0usize;
    let mut out_max = 0usize;
    for b in job.blocks {
        check_attn(&b.video.attn1, dim, dim, dim)?;
        check_attn(&b.video.attn2, dim, dim, dim)?;
        check_attn(&b.audio.attn1, adim, adim, adim)?;
        check_attn(&b.audio.attn2, adim, adim, adim)?;
        check_attn(&b.a2v, dim, adim, dim)?;
        check_attn(&b.v2a, adim, dim, adim)?;
        let tabs_ok = b.video.mods.len() == dv * 9 * dim
            && b.audio.mods.len() == da * 9 * adim
            && b.video.ctx_mod.len() == 2 * dim
            && b.audio.ctx_mod.len() == 2 * adim
            && b.xmods_v.len() == xdv * 4 * dim
            && b.xmods_a.len() == xda * 4 * adim
            && b.gate_a2v.len() == dim
            && b.gate_v2a.len() == adim;
        if !tabs_ok {
            return Err("a block's tables are of the wrong size".into());
        }
        for l in stream_lins(&b.video).chain(stream_lins(&b.audio)).chain(lins(&b.a2v)).chain(lins(&b.v2a)) {
            plane_max = plane_max.max(l.rows * l.cols);
            out_max = out_max.max(l.rows);
        }
    }
    let f = funcs(st, c)?;
    let cap = ceiling();
    let panel: usize = match cap {
        Some(b) if b < (5 << 30) => 96 << 20,
        Some(b) if b < (9 << 30) => 192 << 20,
        _ => 384 << 20,
    };
    let cq = (panel / (out_max * 6)).clamp(16, 4096);

    // ── constants: one upload ──
    let mut consts = Consts(Vec::new());
    let mut lay: Vec<BOff> = Vec::with_capacity(job.blocks.len());
    for b in job.blocks {
        lay.push(BOff {
            v: consts.stream(&b.video),
            a: consts.stream(&b.audio),
            a2v: consts.attn(&b.a2v),
            v2a: consts.attn(&b.v2a),
            xv: consts.put(b.xmods_v),
            xa: consts.put(b.xmods_a),
            ga: consts.put(b.gate_a2v),
            gv: consts.put(b.gate_v2a),
        });
    }
    let consts = consts.0;

    // ── the working set that lives through the call ──
    let xv = st.h3.buf(c, "lx xv", n * dim * 4)?;
    let xa = st.h3.buf(c, "lx xa", m * adim * 4)?;
    let pv = st.h3.buf(c, "lx pv", n * dim * 4)?;
    let pa = st.h3.buf(c, "lx pa", m * adim * 4)?;
    let nq = st.h3.buf(c, "lx nq", (n * dim).max(m * adim) * 2)?;
    let nk = st.h3.buf(c, "lx nk", (v.ctx_len * dim).max(a.ctx_len * adim).max(n * dim).max(m * adim) * 2)?;
    let w = st.h3.buf(c, "lx w", plane_max * 2 * 2)?;
    let y = st.h3.buf(c, "lx y", cq * out_max * 4)?;
    let cb = st.h3.buf(c, "lx c", consts.len() * 4 + 64)?;
    let rowb = st.h3.buf(c, "lx rows", (2 * n + 2 * m) * 4 + 64)?;
    let pe_len = v.pe.0.len() + a.pe.0.len() + v.xpe.0.len() + a.xpe.0.len();
    let peb = st.h3.buf(c, "lx pe", pe_len * 2 * 4 + 64)?;
    let ctxb = st.h3.buf(c, "lx ctx", (v.ctx.len() + a.ctx.len()) * 4 + 64)?;
    let maskb = st.h3.buf(c, "lx mask", (v.ctx_len + a.ctx_len) * 4 + 64)?;
    let wk = Work { w, y, cb, cq, panel };

    // ── which blocks stay on the card ──
    let uid = job.model.uid();
    let (free, total) = cu::mem_get_info().map_err(derr)?;
    let reserve: i64 = (160 << 20) + (128 << 20);
    let mut room: i64 = match cap {
        Some(b) => b as i64 - (total - free) as i64,
        None => free as i64,
    } - reserve;
    let mut keep = vec![false; job.blocks.len()];
    let mut all_before = true;
    for (i, b) in job.blocks.iter().enumerate() {
        let mut need = 0u64;
        for l in stream_lins(&b.video).chain(stream_lins(&b.audio)).chain(lins(&b.a2v)).chain(lins(&b.v2a)) {
            if let Some(ix) = l.w {
                if !st.h3.mats.contains_key(&(uid, ix)) {
                    let (kind, r, co, _) = packed(job.model, ix)?;
                    need += packed_dev_bytes(kind, r, co);
                }
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
    if debug() {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            eprintln!(
                "cuda ltx chain: {n} video and {m} audio rows, chunk {cq}, {} of {} blocks resident, card free {:.1} GB",
                keep.iter().filter(|k| **k).count(),
                job.blocks.len(),
                free as f64 / 1e9
            );
        }
    }

    // ── up ──
    let f4 = |len: usize| (len * 4) as u64;
    let (vr, vxr, ar, axr) = (rowb, rowb + f4(n), rowb + f4(2 * n), rowb + f4(2 * n + m));
    let mut pe_at = peb;
    let mut next_pe = |t: (&[f32], &[f32], usize)| -> Res<Pe> {
        let p = Pe { cos: pe_at, sin: pe_at + f4(t.0.len()), half: t.2 };
        unsafe {
            cu::memcpy_htod_async(p.cos, t.0, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(p.sin, t.1, c.stream).map_err(derr)?;
        }
        pe_at += f4(2 * t.0.len());
        Ok(p)
    };
    let (v_pe, a_pe, v_xpe, a_xpe) = (next_pe(v.pe)?, next_pe(a.pe)?, next_pe(v.xpe)?, next_pe(a.xpe)?);
    for p in [(v_pe, dim), (a_pe, adim), (v_xpe, xw), (a_xpe, xw)] {
        if p.0.half == 0 || p.1 % (2 * p.0.half) != 0 {
            return Err("a rotation group that does not divide its width".into());
        }
    }
    let (vctx, actx) = (ctxb, ctxb + f4(v.ctx.len()));
    let (vmask, amask) = (
        if v.mask.is_empty() { 0 } else { maskb },
        if a.mask.is_empty() { 0 } else { maskb + f4(v.ctx_len) },
    );
    unsafe {
        cu::memcpy_htod_async(xv, vx, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(xa, ax, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(cb, &consts, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(vr, v.rows, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(vxr, v.xrows, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(ar, a.rows, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(axr, a.xrows, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(vctx, v.ctx, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(actx, a.ctx, c.stream).map_err(derr)?;
        if vmask != 0 {
            cu::memcpy_htod_async(vmask, v.mask, c.stream).map_err(derr)?;
        }
        if amask != 0 {
            cu::memcpy_htod_async(amask, a.mask, c.stream).map_err(derr)?;
        }
    }

    let at = |off: usize| -> u64 { if off == usize::MAX { 0 } else { cb + (off * 4) as u64 } };
    for (bi, blk) in job.blocks.iter().enumerate() {
        let (l, k) = (lay[bi], keep[bi]);
        // a stream's self-attention, then its attention to the prompt
        let streams = [
            (&blk.video, &l.v, xv, n, dim, vr, v_pe, vctx, v.ctx_len, vmask),
            (&blk.audio, &l.a, xa, m, adim, ar, a_pe, actx, a.ctx_len, amask),
        ];
        for (s, o, x, rows, width, ridx, pe, ctx_p, ctx_len, mask) in streams {
            let tab = at(o.mods);
            let add = |go: usize| Add { x, tab, rowidx: ridx, width, stride: 9 * width, go };
            unsafe {
                ada(c, &f, x, nq, rows, width, tab, ridx, 9 * width, 0, width, true)?;
                attention(c, st, &f, job.model, k, &s.attn1, &o.a1, &wk, nq, rows, nq, rows, Some(pe), Some(pe), 0, &add(2 * width))?;
                ada(c, &f, x, nq, rows, width, tab, ridx, 9 * width, 6 * width, 7 * width, true)?;
                ada(c, &f, ctx_p, nk, ctx_len, width, at(o.ctx), 0, 2 * width, 0, width, false)?;
                attention(c, st, &f, job.model, k, &s.attn2, &o.a2, &wk, nq, rows, nk, ctx_len, None, None, mask, &add(8 * width))?;
            }
        }
        // the fusion, both directions off the streams as they are now
        unsafe {
            let (tv, ta) = ((n * dim) as u64, (m * adim) as u64);
            c.launch(f.copy, (tv.div_ceil(256) as u32, 1), 256, 0, kargs!(xv, pv, tv))?;
            c.launch(f.copy, (ta.div_ceil(256) as u32, 1), 256, 0, kargs!(xa, pa, ta))?;
            let (xvt, xat) = (at(l.xv), at(l.xa));
            // audio to video: (scale, shift) are rows 0 and 1 of the tables
            ada(c, &f, pv, nq, n, dim, xvt, vxr, 4 * dim, dim, 0, true)?;
            ada(c, &f, pa, nk, m, adim, xat, axr, 4 * adim, adim, 0, true)?;
            let add = Add { x: xv, tab: at(l.ga), rowidx: 0, width: dim, stride: dim, go: 0 };
            attention(c, st, &f, job.model, k, &blk.a2v, &l.a2v, &wk, nq, n, nk, m, Some(v_xpe), Some(a_xpe), 0, &add)?;
            // video to audio: rows 2 and 3
            ada(c, &f, pa, nq, m, adim, xat, axr, 4 * adim, 3 * adim, 2 * adim, true)?;
            ada(c, &f, pv, nk, n, dim, xvt, vxr, 4 * dim, 3 * dim, 2 * dim, true)?;
            let add = Add { x: xa, tab: at(l.gv), rowidx: 0, width: adim, stride: adim, go: 0 };
            attention(c, st, &f, job.model, k, &blk.v2a, &l.v2a, &wk, nq, m, nk, n, Some(a_xpe), Some(v_xpe), 0, &add)?;
            feed_forward(c, st, &f, job.model, k, &blk.video, &l.v, &wk, xv, n, dim, vr)?;
            feed_forward(c, st, &f, job.model, k, &blk.audio, &l.a, &wk, xa, m, adim, ar)?;
        }
    }
    // down into copies: a refused call must leave the caller's streams alone
    let (mut ov, mut oa) = (vec![0f32; vx.len()], vec![0f32; ax.len()]);
    unsafe {
        cu::memcpy_dtoh_async(&mut ov, xv, c.stream).map_err(derr)?;
        cu::memcpy_dtoh_async(&mut oa, xa, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if ov.iter().chain(oa.iter()).any(|x| !x.is_finite()) {
        return Err("a stream came back with a value that is not finite".into());
    }
    vx.copy_from_slice(&ov);
    ax.copy_from_slice(&oa);
    Ok(())
}

/// A host f32 weight as an f16 plane on the card, kept by the address and
/// length of the host slice until the stage is released.
unsafe fn host_half(c: &Ctx, st: &mut State, v: &[f32]) -> Res<u64> {
    let key = (v.as_ptr() as usize, v.len());
    if let Some(bf) = st.h3.host_w.get(&key) {
        return Ok(bf.ptr);
    }
    let to_half = st.h3.funcs(c)?.to_half;
    let tmp = st.h3.buf(c, "cv wup", v.len() * 4)?;
    let bf = Buf::new(v.len() * 2 + 64)?;
    let p = bf.ptr;
    let n = v.len() as u64;
    unsafe {
        cu::memcpy_htod_async(tmp, v, c.stream).map_err(derr)?;
        c.launch(to_half, (v.len().div_ceil(256) as u32, 1), 256, 0, kargs!(tmp, p, n))?;
    }
    // the staging buffer is reused by the next weight
    c.sync()?;
    st.h3.host_w.insert(key, bf);
    Ok(p)
}

/// The LTX-2.5 video VAE's stages over a volume `[c][f][h][w]` on the card:
/// the activation goes up once and the result comes back once. Returns the
/// result and its `(c, f, h, w)`.
pub fn ltx_vae_chain(
    steps: &[Vae3dStep<'_>],
    x: &[f32],
    c0: usize,
    f0: usize,
    h0: usize,
    w0: usize,
) -> Option<(Vec<f32>, usize, usize, usize, usize)> {
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { vae_chain(c, &mut st, steps, x, c0, f0, h0, w0) } {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("ltx vae chain", &e);
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn vae_chain(
    c: &Ctx,
    st: &mut State,
    steps: &[Vae3dStep<'_>],
    x: &[f32],
    c0: usize,
    f0: usize,
    h0: usize,
    w0: usize,
) -> Res<(Vec<f32>, usize, usize, usize, usize)> {
    if steps.is_empty() || x.len() != c0 * f0 * h0 * w0 || x.is_empty() {
        return Err("no stages, or an input of the wrong size".into());
    }
    // the shapes every stage leaves, and the largest activation among them
    let (mut ch, mut fr, mut hh, mut ww) = (c0, f0, h0, w0);
    let mut act_max = x.len();
    for s in steps {
        match s {
            Vae3dStep::Conv { w, b, in_ch, out_ch } => {
                if *in_ch != ch || w.len() != out_ch * in_ch * 27 || b.len() != *out_ch {
                    return Err(format!("a convolution takes {in_ch} channels, the activation has {ch}"));
                }
                ch = *out_ch;
            }
            Vae3dStep::DepthToSpace { p } => {
                let q = p.0 * p.1 * p.2;
                if q == 0 || ch % q != 0 || (p.0 == 2 && fr * 2 < 2) {
                    return Err("a depth-to-space that does not divide the channels".into());
                }
                ch /= q;
                fr = fr * p.0 - (p.0 == 2) as usize;
                hh *= p.1;
                ww *= p.2;
            }
            Vae3dStep::NormSilu { .. } | Vae3dStep::Save | Vae3dStep::AddSaved => {}
        }
        act_max = act_max.max(ch * fr * hh * ww);
    }
    // three activations and an im2col chunk, inside what is free and inside
    // `CMF_GPU_VRAM_MB` when it is set
    let (free, total) = cu::mem_get_info().map_err(derr)?;
    let room = match ceiling() {
        Some(cap) => (cap as i64 - (total - free) as i64).min(free as i64),
        None => free as i64,
    };
    if (act_max as i64) * 4 * 3 + (512 << 20) > room {
        return Err(format!("three activations of {} MB do not fit the card", act_max * 4 >> 20));
    }
    let f = funcs(st, c)?;
    let a = st.h3.buf(c, "vx a", act_max * 4)?;
    let b = st.h3.buf(c, "vx b", act_max * 4)?;
    let sv = st.h3.buf(c, "vx s", act_max * 4)?;
    let (mut cur, mut nxt) = (a, b);
    let (mut ch, mut fr, mut hh, mut ww) = (c0, f0, h0, w0);
    unsafe { cu::memcpy_htod_async(cur, x, c.stream).map_err(derr)? };
    let mut saved = 0usize;
    for s in steps {
        let p = fr * hh * ww;
        match s {
            Vae3dStep::NormSilu { eps } => unsafe {
                let (ci, pu) = (ch as i32, p as u64);
                c.launch(f.norm_silu, (p.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nxt, ci, pu, *eps))?;
                std::mem::swap(&mut cur, &mut nxt);
            },
            Vae3dStep::Save => unsafe {
                let n = (ch * p) as u64;
                c.launch(f.copy, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, sv, n))?;
                saved = ch * p;
            },
            Vae3dStep::AddSaved => unsafe {
                if saved != ch * p {
                    return Err("the saved activation is of another size".into());
                }
                let (pu, n, nul) = (p as u64, (ch * p) as u64, 0u64);
                c.launch(f.bias, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nul, sv, pu, n))?;
            },
            Vae3dStep::Conv { w, b, in_ch, out_ch } => unsafe {
                let kk = in_ch * 27;
                let wp = host_half(c, st, w)?;
                let bp = st.h3.buf(c, "vx bias", out_ch * 4 + 64)?;
                cu::memcpy_htod_async(bp, b, c.stream).map_err(derr)?;
                // an im2col chunk of at most 256 MB
                let chunk = ((256usize << 20) / (kk * 2)).clamp(1, p);
                let cols = st.h3.buf(c, "vx cols", chunk * kk * 2)?;
                let (ci, fi, hi, wi) = (*in_ch as i32, fr as i32, hh as i32, ww as i32);
                let mut p0 = 0usize;
                while p0 < p {
                    let np = chunk.min(p - p0);
                    let (p0u, total) = (p0 as u64, (np * in_ch) as u64);
                    c.launch(f.im2col3d, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, cols, ci, fi, hi, wi, p0u, total))?;
                    // y [out_ch][p] at positions p0..: column-major (positions x out_ch)
                    c.gemm(true, false, np, *out_ch, kk, 1.0, cols, kk, wp, kk, 0.0, nxt + (p0 * 4) as u64, p, Gemm::Half)?;
                    p0 += np;
                }
                let (pu, n, nul) = (p as u64, (out_ch * p) as u64, 0u64);
                c.launch(f.bias, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(nxt, bp, nul, pu, n))?;
                // the bias buffer is reused by the next convolution
                c.sync()?;
                std::mem::swap(&mut cur, &mut nxt);
                ch = *out_ch;
            },
            Vae3dStep::DepthToSpace { p: (p1, p2, p3) } => unsafe {
                let drop = (*p1 == 2) as usize;
                let (oc, of, oh, ow) = (ch / (p1 * p2 * p3), fr * p1 - drop, hh * p2, ww * p3);
                let total = (oc * of * oh * ow) as u64;
                let (fi, hi, wi, a1, a2, a3, di) = (fr as i32, hh as i32, ww as i32, *p1 as i32, *p2 as i32, *p3 as i32, drop as i32);
                c.launch(f.d2s, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nxt, fi, hi, wi, a1, a2, a3, di, total))?;
                std::mem::swap(&mut cur, &mut nxt);
                (ch, fr, hh, ww) = (oc, of, oh, ow);
            },
        }
    }
    let mut out = vec![0f32; ch * fr * hh * ww];
    unsafe { cu::memcpy_dtoh_async(&mut out, cur, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the decoder came back with a value that is not finite".into());
    }
    Ok((out, ch, fr, hh, ww))
}
