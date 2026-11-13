//! A vision tower's encoder on CUDA (`gpu::vit_encode`, `gpu::sam_encode`):
//! the pre-norm ViT blocks of a CLIP-class tower and SAM's blocks and neck
//! with the activations on the card from the first block to the last, one
//! upload and one readback, as `gpu_wgpu/vit.rs` does on wgpu. The weights go
//! up once per tower (`key`); the products are cuBLAS f32 GEMMs, attention a
//! head at a time (scores, a softmax with SAM's decomposed relative bias on
//! its global blocks, values), SAM's windowed blocks one kernel over the
//! windows. Without it a page's towers ran on the host (40 s of a one-page
//! OCR on an RTX A5000's pod).

use super::*;
use std::sync::Mutex;

const VT_SRC: &str = include_str!("vit.cu");

#[derive(Clone, Copy)]
struct VtFuncs {
    ln: cs::CUfunction,
    act: cs::CUfunction,
    add: cs::CUfunction,
    split: cs::CUfunction,
    part: cs::CUfunction,
    unpart: cs::CUfunction,
    win: cs::CUfunction,
    softmax: cs::CUfunction,
    conv: cs::CUfunction,
}

/// A ViT block's weights: ln1 w b, qkv w b, proj w b, ln2 w b, fc1 w b, fc2 w b.
type VitW = [Buf; 12];
/// A SAM block's: the same with the relative table after proj's.
type SamW = [Buf; 13];

struct VtState {
    f: Option<VtFuncs>,
    vit: Option<(u64, Vec<VitW>)>,
    sam: Option<SamDev>,
}

struct SamDev {
    key: u64,
    layers: Vec<SamW>,
    convs: Vec<Buf>,
    norms: Vec<Buf>,
    /// (grid, the global blocks' relative tables resized to it)
    rel: Option<(usize, Vec<Buf>)>,
}

// SAFETY: the functions and allocations are used under the context's state lock.
unsafe impl Send for VtState {}

static STATE: Mutex<VtState> = Mutex::new(VtState { f: None, vit: None, sam: None });

fn funcs(c: &Ctx, s: &mut VtState) -> Res<VtFuncs> {
    if let Some(f) = s.f {
        return Ok(f);
    }
    let m = compile(VT_SRC, Vec::new(), c.arch)?;
    let f = VtFuncs {
        ln: func(m, "vt_ln")?,
        act: func(m, "vt_bias_act")?,
        add: func(m, "vt_add")?,
        split: func(m, "vt_split")?,
        part: func(m, "sam_part")?,
        unpart: func(m, "sam_unpart")?,
        win: func(m, "sam_win")?,
        softmax: func(m, "vt_softmax")?,
        conv: func(m, "sam_conv")?,
    };
    s.f = Some(f);
    Ok(f)
}

fn blocks(total: usize) -> (u32, u32) {
    (total.div_ceil(256).max(1) as u32, 1)
}

fn rows_grid(rows: usize) -> (u32, u32) {
    (rows.clamp(1, 65_535) as u32, rows.div_ceil(65_535).max(1) as u32)
}

/// The kernels and products both towers share, over one context.
struct Ops<'a> {
    c: &'a Ctx,
    f: VtFuncs,
    eps: f32,
}

impl Ops<'_> {
    unsafe fn ln(&self, src: u64, w: &Buf, b: &Buf, dst: u64, rows: usize, cols: usize) -> Res<()> {
        let (r_i, c_i) = (rows as i32, cols as i32);
        unsafe { self.c.launch(self.f.ln, rows_grid(rows), 256, 0, kargs!(src, w.ptr, b.ptr, dst, r_i, c_i, self.eps)) }
    }
    /// y [rows][m] = x [rows][k] · w [m][k]ᵀ
    unsafe fn mm(&self, w: &Buf, x: u64, y: u64, k: usize, m: usize, rows: usize) -> Res<()> {
        unsafe { self.c.gemm(true, false, m, rows, k, 1.0, w.ptr, k, x, k, 0.0, y, m, Gemm::F32) }
    }
    unsafe fn bias_act(&self, y: u64, b: &Buf, rows: usize, cols: usize, act: i32) -> Res<()> {
        let (tot, c_i) = ((rows * cols) as u64, cols as i32);
        unsafe { self.c.launch(self.f.act, blocks(rows * cols), 256, 0, kargs!(y, b.ptr, tot, c_i, act)) }
    }
    unsafe fn add(&self, x: u64, y: u64, b: &Buf, rows: usize, cols: usize) -> Res<()> {
        let (tot, c_i) = ((rows * cols) as u64, cols as i32);
        unsafe { self.c.launch(self.f.add, blocks(rows * cols), 256, 0, kargs!(x, y, b.ptr, tot, c_i)) }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn split(&self, qkv: u64, cs: u64, sn: u64, q: u64, k: u64, v: u64, n: usize, nh: usize, hd: usize, halves: bool) -> Res<()> {
        let (n_i, nh_i, hd_i, h_i) = (n as i32, nh as i32, hd as i32, halves as i32);
        unsafe { self.c.launch(self.f.split, blocks(nh * n * hd), 256, 0, kargs!(qkv, cs, sn, q, k, v, n_i, nh_i, hd_i, h_i)) }
    }
    /// Full attention over head-major planes [nh][n][hd] into ab [n][nh hd],
    /// a head at a time through `s` ([n][n] scores); `rel` (SAM's global
    /// blocks): the relative table and the grid side.
    #[allow(clippy::too_many_arguments)]
    unsafe fn attend(&self, q: u64, k: u64, v: u64, ab: u64, s: u64, n: usize, nh: usize, hd: usize, scale: f32, rel: Option<(u64, usize)>) -> Res<()> {
        let (n_i, hd_i) = (n as i32, hd as i32);
        let plane = (n * hd * 4) as u64;
        for h in 0..nh {
            let (qh, kh, vh) = (q + h as u64 * plane, k + h as u64 * plane, v + h as u64 * plane);
            let (rp, side) = rel.unwrap_or((0, 1));
            let s_i = side as i32;
            unsafe {
                self.c.gemm(true, false, n, n, hd, scale, kh, hd, qh, hd, 0.0, s, n, Gemm::F32)?;
                self.c.launch(self.f.softmax, rows_grid(n), 256, 0, kargs!(s, n_i, qh, rp, s_i, hd_i))?;
                self.c.gemm(false, false, hd, n, n, 1.0, vh, hd, s, n, 0.0, ab + (h * hd * 4) as u64, nh * hd, Gemm::F32)?;
            }
        }
        Ok(())
    }
}

/// Bytes of the readback check: a tower that came back with a value that is
/// not finite declines (the host runs it).
fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// The ViT blocks over `x` (`[n, d]`, embedded rows) in place; `false` =
/// declined (said once), the caller runs its per-op path.
pub fn vit_encode(a: &crate::gpu::VitArgs, x: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    match unsafe { vit_run(c, &mut s, a, x) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("vit encode", &e);
            false
        }
    }
}

unsafe fn vit_run(c: &Ctx, s: &mut VtState, a: &crate::gpu::VitArgs, x: &mut [f32]) -> Res<()> {
    let (n, d, nh, hd, inter) = (a.n, a.d, a.heads, a.head_dim, a.inter);
    if n == 0 || nh * hd != d || x.len() != n * d || a.cos.len() != n * hd || a.sin.len() != n * hd {
        return Err("a tower shape outside the kernels".into());
    }
    let f = funcs(c, s)?;
    if s.vit.as_ref().is_none_or(|(k, l)| *k != a.key || l.len() != a.layers.len()) {
        s.vit = None;
        let mut layers = Vec::with_capacity(a.layers.len());
        for l in &a.layers {
            let up = Buf::from_slice;
            layers.push([
                up(l.ln1_w)?,
                up(l.ln1_b)?,
                up(l.qkv)?,
                up(l.qkv_b)?,
                up(l.proj)?,
                up(l.proj_b)?,
                up(l.ln2_w)?,
                up(l.ln2_b)?,
                up(l.fc1)?,
                up(l.fc1_b)?,
                up(l.fc2)?,
                up(l.fc2_b)?,
            ]);
        }
        s.vit = Some((a.key, layers));
    }
    let ops = Ops { c, f, eps: a.eps as f32 };
    let t0 = std::time::Instant::now();
    let xb = Buf::from_slice(x)?;
    let xn = Buf::new(n * d * 4)?;
    let wide = Buf::new(n * (3 * d).max(inter) * 4)?;
    let tmp = Buf::new(n * d * 4)?;
    let (q, k, v) = (Buf::new(n * d * 4)?, Buf::new(n * d * 4)?, Buf::new(n * d * 4)?);
    let sc = Buf::new(n * n * 4)?;
    let (cs, sn) = (Buf::from_slice(a.cos)?, Buf::from_slice(a.sin)?);
    let scale = (hd as f32).powf(-0.5);
    let act = if a.quick_gelu { 3 } else if a.gelu_tanh { 2 } else { 1 };
    let (_, layers) = s.vit.as_ref().expect("set above");
    for w in layers {
        unsafe {
            ops.ln(xb.ptr, &w[0], &w[1], xn.ptr, n, d)?;
            ops.mm(&w[2], xn.ptr, wide.ptr, d, 3 * d, n)?;
            ops.bias_act(wide.ptr, &w[3], n, 3 * d, 0)?;
            ops.split(wide.ptr, cs.ptr, sn.ptr, q.ptr, k.ptr, v.ptr, n, nh, hd, a.rope_halves)?;
            ops.attend(q.ptr, k.ptr, v.ptr, xn.ptr, sc.ptr, n, nh, hd, scale, None)?;
            ops.mm(&w[4], xn.ptr, tmp.ptr, d, d, n)?;
            ops.add(xb.ptr, tmp.ptr, &w[5], n, d)?;
            ops.ln(xb.ptr, &w[6], &w[7], xn.ptr, n, d)?;
            ops.mm(&w[8], xn.ptr, wide.ptr, d, inter, n)?;
            ops.bias_act(wide.ptr, &w[9], n, inter, act)?;
            ops.mm(&w[10], wide.ptr, tmp.ptr, inter, d, n)?;
            ops.add(xb.ptr, tmp.ptr, &w[11], n, d)?;
        }
    }
    unsafe { cu::memcpy_dtoh_async(x, xb.ptr, c.stream).map_err(derr)? };
    c.sync()?;
    if !finite(x) {
        return Err("the tower came back with a value that is not finite".into());
    }
    if a.prof {
        eprintln!("vit (cuda, resident): {} layers over {n} rows, {:.3} s", layers.len(), t0.elapsed().as_secs_f64());
    }
    Ok(())
}

/// SAM's blocks, neck and strided convs over the embedded grid `x`
/// (`[g², d]`); `out` the final channels. `false` = declined.
pub fn sam_encode(a: &crate::gpu::SamArgs, x: &[f32], out: &mut Vec<f32>) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    match unsafe { sam_run(c, &mut s, a, x, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("sam encode", &e);
            false
        }
    }
}

unsafe fn sam_run(c: &Ctx, s: &mut VtState, a: &crate::gpu::SamArgs, x: &[f32], out: &mut Vec<f32>) -> Res<()> {
    let (g, d, nh, hd, inter) = (a.g, a.d, a.heads, a.head_dim, a.inter);
    let n = g * g;
    if x.len() != n * d || nh * hd != d || g > 64 || a.layers.iter().any(|l| l.window > 16) {
        return Err("a SAM shape outside the kernels".into());
    }
    let n_global = a.layers.iter().filter(|l| l.window == 0).count();
    if a.rel_global.len() != n_global {
        return Err("a global block without its relative table".into());
    }
    let f = funcs(c, s)?;
    if s.sam.as_ref().is_none_or(|m| m.key != a.key || m.layers.len() != a.layers.len()) {
        s.sam = None;
        let up = Buf::from_slice;
        let mut layers = Vec::with_capacity(a.layers.len());
        for l in &a.layers {
            layers.push([
                up(l.ln1_w)?,
                up(l.ln1_b)?,
                up(l.qkv)?,
                up(l.qkv_b)?,
                up(l.proj)?,
                up(l.proj_b)?,
                up(l.rel)?,
                up(l.ln2_w)?,
                up(l.ln2_b)?,
                up(l.fc1)?,
                up(l.fc1_b)?,
                up(l.fc2)?,
                up(l.fc2_b)?,
            ]);
        }
        let convs = a.convs.iter().map(|v| up(v.w)).collect::<Res<Vec<_>>>()?;
        let norms = a.norms.iter().map(|v| up(v)).collect::<Res<Vec<_>>>()?;
        s.sam = Some(SamDev { key: a.key, layers, convs, norms, rel: None });
    }
    let m = s.sam.as_mut().expect("set above");
    if m.rel.as_ref().is_none_or(|r| r.0 != g) {
        m.rel = Some((g, a.rel_global.iter().map(|v| Buf::from_slice(v)).collect::<Res<Vec<_>>>()?));
    }
    let ops = Ops { c, f, eps: a.eps as f32 };
    // windows pad up to the window grid; the largest activation sizes it
    let w0 = a.layers.iter().map(|l| l.window).max().unwrap_or(1).max(1);
    let wrows = g.div_ceil(w0) * g.div_ceil(w0) * w0 * w0;
    let rows_max = wrows.max(n);
    let xb = Buf::from_slice(x)?;
    let xn = Buf::new(n * d * 4)?;
    let wn = Buf::new(rows_max * d * 4)?;
    let wide = Buf::new(rows_max * (3 * d).max(inter) * 4)?;
    let tmp = Buf::new(rows_max * d * 4)?;
    let wo = Buf::new(rows_max * d * 4)?;
    let (q, k, v) = (Buf::new(rows_max * d * 4)?, Buf::new(rows_max * d * 4)?, Buf::new(rows_max * d * 4)?);
    let sc = Buf::new(n * n * 4)?;
    let scale = (hd as f32).powf(-0.5);
    let (d_i, g_i, nh_i, hd_i) = (d as i32, g as i32, nh as i32, hd as i32);
    let mut gi = 0usize;
    for (w, spec) in m.layers.iter().zip(&a.layers) {
        unsafe {
            ops.ln(xb.ptr, &w[0], &w[1], xn.ptr, n, d)?;
            if spec.window > 0 {
                let win = spec.window;
                let rows = g.div_ceil(win) * g.div_ceil(win) * win * win;
                let (r_i, w_i) = (rows as i32, win as i32);
                c.launch(f.part, blocks(rows * d), 256, 0, kargs!(xn.ptr, wn.ptr, r_i, d_i, g_i, w_i))?;
                ops.mm(&w[2], wn.ptr, wide.ptr, d, 3 * d, rows)?;
                ops.bias_act(wide.ptr, &w[3], rows, 3 * d, 0)?;
                ops.split(wide.ptr, 0, 0, q.ptr, k.ptr, v.ptr, rows, nh, hd, false)?;
                c.launch(f.win, blocks(nh * rows), 256, 0, kargs!(q.ptr, k.ptr, v.ptr, w[6].ptr, wo.ptr, r_i, g_i, w_i, nh_i, hd_i, scale))?;
                ops.mm(&w[4], wo.ptr, tmp.ptr, d, d, rows)?;
                let n_i = n as i32;
                c.launch(f.unpart, blocks(n * d), 256, 0, kargs!(xb.ptr, tmp.ptr, w[5].ptr, n_i, d_i, g_i, w_i))?;
            } else {
                ops.mm(&w[2], xn.ptr, wide.ptr, d, 3 * d, n)?;
                ops.bias_act(wide.ptr, &w[3], n, 3 * d, 0)?;
                ops.split(wide.ptr, 0, 0, q.ptr, k.ptr, v.ptr, n, nh, hd, false)?;
                let rel = m.rel.as_ref().expect("set above").1[gi].ptr;
                gi += 1;
                ops.attend(q.ptr, k.ptr, v.ptr, wo.ptr, sc.ptr, n, nh, hd, scale, Some((rel, g)))?;
                ops.mm(&w[4], wo.ptr, tmp.ptr, d, d, n)?;
                ops.add(xb.ptr, tmp.ptr, &w[5], n, d)?;
            }
            ops.ln(xb.ptr, &w[7], &w[8], xn.ptr, n, d)?;
            ops.mm(&w[9], xn.ptr, wide.ptr, d, inter, n)?;
            ops.bias_act(wide.ptr, &w[10], n, inter, 1)?;
            ops.mm(&w[11], wide.ptr, tmp.ptr, inter, d, n)?;
            ops.add(xb.ptr, tmp.ptr, &w[12], n, d)?;
        }
    }
    // the neck and the strided convs
    let mut cur: Option<Buf> = None;
    let mut cg = g;
    let mut norm_at = 0usize;
    let mut last = (0usize, 0usize);
    for (cv, wbuf) in a.convs.iter().zip(&m.convs) {
        let g2 = (cg + 2 * cv.pad - cv.k) / cv.stride + 1;
        let y = Buf::new(g2 * g2 * cv.co * 4)?;
        let src = cur.as_ref().map_or(xb.ptr, |b| b.ptr);
        let (cg_i, ci_i, co_i, k_i, s_i, p_i, g2_i) = (cg as i32, cv.ci as i32, cv.co as i32, cv.k as i32, cv.stride as i32, cv.pad as i32, g2 as i32);
        unsafe { c.launch(f.conv, blocks(g2 * g2 * cv.co), 256, 0, kargs!(wbuf.ptr, src, y.ptr, cg_i, ci_i, co_i, k_i, s_i, p_i, g2_i))? };
        cur = Some(if cv.norm {
            let (nw, nb) = (&m.norms[norm_at], &m.norms[norm_at + 1]);
            norm_at += 2;
            let yn = Buf::new(g2 * g2 * cv.co * 4)?;
            unsafe { ops.ln(y.ptr, nw, nb, yn.ptr, g2 * g2, cv.co)? };
            yn
        } else {
            y
        });
        cg = g2;
        last = (g2 * g2, cv.co);
    }
    let res = cur.as_ref().map_or(xb.ptr, |b| b.ptr);
    let len = if a.convs.is_empty() { n * d } else { last.0 * last.1 };
    out.clear();
    out.resize(len, 0.0);
    unsafe { cu::memcpy_dtoh_async(out.as_mut_slice(), res, c.stream).map_err(derr)? };
    c.sync()?;
    if !finite(out) {
        return Err("the encoder came back with a value that is not finite".into());
    }
    Ok(())
}
