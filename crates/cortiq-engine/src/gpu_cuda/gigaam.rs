//! The GigaAM encoder on CUDA (`gpu::gigaam_encode`): the three Zapformer
//! stacks of one segment with the activations on the card from the first
//! projection to the output norm — one upload, one readback. The weights go
//! up once a model (`key`) as f32 planes; the products are cuBLAS GEMMs on
//! the tensor cores (TF32 inside, plain f32 under `CMF_CUDA_PREC=exact`),
//! attention two batched GEMMs over the heads with a softmax between, and
//! everything else the kernels of `gigaam.cu`.

use super::*;
use crate::gpu::{GaLin, GigaamArgs};
use std::sync::Mutex;

const GA_SRC: &str = include_str!("gigaam.cu");

#[derive(Clone, Copy)]
struct GaFuncs {
    bias_act: cs::CUfunction,
    scale: cs::CUfunction,
    axpy: cs::CUfunction,
    copy_pad: cs::CUfunction,
    rms: cs::CUfunction,
    split: cs::CUfunction,
    softmax: cs::CUfunction,
    gate: cs::CUfunction,
    mean: cs::CUfunction,
    conv: cs::CUfunction,
}

/// A product on the card: the weight plane and its bias (None: no bias).
struct LinDev {
    w: Buf,
    b: Option<Buf>,
    rows: usize,
    cols: usize,
}

struct ConvDev {
    inp: LinDev,
    taps: Buf,
    bias: Buf,
    mean_w: Buf,
    kernel: usize,
    out: LinDev,
}

struct LayerDev {
    ff1: (LinDev, LinDev),
    in_norm: Buf,
    qk: LinDev,
    vg: LinDev,
    attn_out: LinDev,
    conv: Option<ConvDev>,
    ff2: Option<(LinDev, LinDev)>,
    norm: Buf,
}

struct StackDev {
    proj: LinDev,
    proj_t: LinDev,
    layers: Vec<LayerDev>,
}

struct GaState {
    f: Option<GaFuncs>,
    model: Option<(u64, Vec<StackDev>, Buf, Buf)>,
    /// working buffers by name; they only grow
    work: Option<std::collections::HashMap<&'static str, Buf>>,
}

/// The working buffer `name` of at least `bytes`.
fn work(c: &Ctx, bufs: &mut std::collections::HashMap<&'static str, Buf>, name: &'static str, bytes: usize) -> Res<u64> {
    if let Some(b) = bufs.get(name) {
        if b.bytes >= bytes {
            return Ok(b.ptr);
        }
        // queued work may still read the old one
        c.sync()?;
        bufs.remove(name);
    }
    let b = Buf::new(bytes.next_multiple_of(1 << 20))?;
    let p = b.ptr;
    bufs.insert(name, b);
    Ok(p)
}

// SAFETY: the functions and allocations are used under the context's state lock.
unsafe impl Send for GaState {}

static STATE: Mutex<GaState> = Mutex::new(GaState { f: None, model: None, work: None });

fn funcs(c: &Ctx, s: &mut GaState) -> Res<GaFuncs> {
    if let Some(f) = s.f {
        return Ok(f);
    }
    let m = compile(GA_SRC, Vec::new(), c.arch)?;
    let f = GaFuncs {
        bias_act: func(m, "ga_bias_act")?,
        scale: func(m, "ga_scale")?,
        axpy: func(m, "ga_axpy")?,
        copy_pad: func(m, "ga_copy_pad")?,
        rms: func(m, "ga_rms")?,
        split: func(m, "ga_split")?,
        softmax: func(m, "ga_softmax")?,
        gate: func(m, "ga_gate")?,
        mean: func(m, "ga_mean")?,
        conv: func(m, "ga_conv")?,
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

fn up_lin(l: &GaLin, scratch: &mut Vec<f32>) -> Res<LinDev> {
    scratch.clear();
    scratch.resize(l.rows * l.cols, 0.0);
    l.weight.fill(scratch);
    Ok(LinDev {
        w: Buf::from_slice(scratch.as_slice())?,
        b: if l.bias.is_empty() { None } else { Some(Buf::from_slice(l.bias)?) },
        rows: l.rows,
        cols: l.cols,
    })
}

fn upload(a: &GigaamArgs) -> Res<(Vec<StackDev>, Buf, Buf)> {
    let mut scratch = Vec::new();
    let mut stacks = Vec::with_capacity(a.stacks.len());
    for st in &a.stacks {
        let mut layers = Vec::with_capacity(st.layers.len());
        for l in &st.layers {
            let conv = match &l.conv {
                Some(cv) => Some(ConvDev {
                    inp: up_lin(&cv.inp, &mut scratch)?,
                    taps: Buf::from_slice(cv.taps)?,
                    bias: Buf::from_slice(cv.bias)?,
                    mean_w: Buf::from_slice(cv.mean_w)?,
                    kernel: cv.kernel,
                    out: up_lin(&cv.out, &mut scratch)?,
                }),
                None => None,
            };
            let ff2 = match &l.ff2 {
                Some((i, o)) => Some((up_lin(i, &mut scratch)?, up_lin(o, &mut scratch)?)),
                None => None,
            };
            layers.push(LayerDev {
                ff1: (up_lin(&l.ff1_in, &mut scratch)?, up_lin(&l.ff1_out, &mut scratch)?),
                in_norm: Buf::from_slice(l.in_norm)?,
                qk: up_lin(&l.qk, &mut scratch)?,
                vg: up_lin(&l.vg, &mut scratch)?,
                attn_out: up_lin(&l.attn_out, &mut scratch)?,
                conv,
                ff2,
                norm: Buf::from_slice(l.norm)?,
            });
        }
        stacks.push(StackDev { proj: up_lin(&st.proj, &mut scratch)?, proj_t: up_lin(&st.proj_t, &mut scratch)?, layers });
    }
    Ok((stacks, Buf::from_slice(a.cos)?, Buf::from_slice(a.sin)?))
}

struct Ops<'a> {
    c: &'a Ctx,
    f: GaFuncs,
    mode: Gemm,
}

impl Ops<'_> {
    /// y [n][rows] = x [n][cols] · wᵀ, then the bias and the activation.
    unsafe fn lin(&self, l: &LinDev, x: u64, y: u64, n: usize, act: i32) -> Res<()> {
        let mode = if n < 32 { Gemm::F32 } else { self.mode };
        unsafe { self.c.gemm(true, false, l.rows, n, l.cols, 1.0, l.w.ptr, l.cols, x, l.cols, 0.0, y, l.rows, mode)? };
        if l.b.is_some() || act != 0 {
            let (tot, c_i, hb) = ((n * l.rows) as u64, l.rows as i32, l.b.is_some() as i32);
            let bp = l.b.as_ref().map_or(y, |b| b.ptr);
            unsafe { self.c.launch(self.f.bias_act, blocks(n * l.rows), 256, 0, kargs!(y, bp, tot, c_i, hb, act))? };
        }
        Ok(())
    }
    unsafe fn axpy(&self, dst: u64, src: u64, total: usize, a: f32) -> Res<()> {
        let tot = total as u64;
        unsafe { self.c.launch(self.f.axpy, blocks(total), 256, 0, kargs!(dst, src, tot, a)) }
    }
    unsafe fn scale(&self, dst: u64, src: u64, total: usize, a: f32) -> Res<()> {
        let tot = total as u64;
        unsafe { self.c.launch(self.f.scale, blocks(total), 256, 0, kargs!(dst, src, tot, a)) }
    }
    unsafe fn copy_pad(&self, dst: u64, src: u64, total: usize, valid: usize) -> Res<()> {
        let (tot, val) = (total as u64, valid as u64);
        unsafe { self.c.launch(self.f.copy_pad, blocks(total), 256, 0, kargs!(dst, src, tot, val)) }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn rms(&self, res: u64, s: u64, w: u64, dst: u64, rows: usize, cols: usize, eps: f32, mix: f32, scalar_w: bool) -> Res<()> {
        let (r_i, c_i, sw) = (rows as i32, cols as i32, scalar_w as i32);
        unsafe { self.c.launch(self.f.rms, rows_grid(rows), 256, 0, kargs!(res, s, w, dst, r_i, c_i, eps, mix, sw)) }
    }
}

/// The encoder stacks over `x` (`[t, width]`) in place; `false` = declined
/// (said once), the caller runs its own path.
pub fn gigaam_encode(a: &GigaamArgs, x: &mut [f32], t: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    match unsafe { run(c, &mut s, a, x, t) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gigaam encode", &e);
            false
        }
    }
}

unsafe fn run(c: &Ctx, s: &mut GaState, a: &GigaamArgs, x: &mut [f32], t: usize) -> Res<()> {
    let (w, hd) = (a.width, a.head_dim);
    if t == 0 || x.len() != t * w || hd % 2 != 0 || a.cos.len() < t * hd / 2 || a.sin.len() != a.cos.len() {
        return Err("an encoder shape outside the kernels".into());
    }
    if a.stacks.iter().any(|st| st.fold == 0 || st.fold > 2 || st.residual_scales.len() != st.layers.len() + 1) {
        return Err("a stack shape outside the kernels".into());
    }
    let f = funcs(c, s)?;
    if s.model.as_ref().is_none_or(|m| m.0 != a.key || m.1.len() != a.stacks.len()) {
        s.model = None;
        let (stacks, cos, sin) = upload(a)?;
        s.model = Some((a.key, stacks, cos, sin));
    }
    let GaState { model, work: bufs, .. } = s;
    let (_, stacks, cos, sin) = model.as_ref().expect("set above");
    let bufs = bufs.get_or_insert_with(Default::default);
    let mode = if super::super::prec().1 == Gemm::F32 { Gemm::F32 } else { Gemm::Tf32 };
    let ops = Ops { c, f, mode };
    let t0 = std::time::Instant::now();
    // the widest rows of any stack size the working buffers
    let max_dim = a.stacks.iter().map(|st| st.dim).max().unwrap_or(w);
    let max_in = a.stacks.iter().map(|st| st.fold * w).max().unwrap_or(w);
    let max_heads = a.stacks.iter().map(|st| st.heads).max().unwrap_or(1);
    let wide_cols = a
        .stacks
        .iter()
        .flat_map(|st| st.layers.iter())
        .map(|l| l.ff1_in.rows.max(l.vg.rows).max(l.conv.as_ref().map_or(0, |cv| cv.inp.rows)))
        .max()
        .unwrap_or(max_dim);
    let mut b = |name: &'static str, bytes: usize| work(c, bufs, name, bytes);
    let xb = b("x", (t + 1) * max_in * 4)?;
    let (xin, xout) = (b("xin", (t + 1) * max_in * 4)?, b("xout", (t + 1) * max_in * 4)?);
    let rows_d = (t + 1) * max_dim * 4;
    let (h, res, sbuf, tmp, bypass) = (b("h", rows_d)?, b("res", rows_d)?, b("s", rows_d)?, b("tmp", rows_d)?, b("bypass", rows_d)?);
    let wide = b("wide", (t + 1) * wide_cols * 4)?;
    let planes = (t + 1) * max_heads * hd * 4;
    let qkb = b("qk", 2 * planes)?;
    let (q, k, v, o) = (b("q", planes)?, b("k", planes)?, b("v", planes)?, b("o", planes)?);
    let sc = b("scores", max_heads * t * t * 4)?;
    let mean = b("mean", max_dim * 4)?;
    let out_w = b("out_w", 64)?;
    let out_scale = [a.out_scale];
    unsafe {
        cu::memcpy_htod_async(xb, &*x, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(out_w, &out_scale, c.stream).map_err(derr)?;
    }
    let scale = (hd as f32).powf(-0.5);
    for (st, dev) in a.stacks.iter().zip(stacks) {
        let (n, d, nh) = (t.div_ceil(st.fold), st.dim, st.heads);
        let in_w = st.fold * w;
        let hw = nh * hd;
        let (n_i, nh_i, hd_i, d_i) = (n as i32, nh as i32, hd as i32, d as i32);
        unsafe {
            // rows of this stack: the frames, or pairs of them
            let src = if st.fold == 1 {
                xb
            } else {
                ops.copy_pad(xin, xb, n * in_w, t * w)?;
                xin
            };
            ops.lin(&dev.proj, src, h, n, 0)?;
            ops.scale(bypass, h, n * d, st.residual_scales[0])?;
            ops.scale(h, h, n * d, st.input_scale)?;
            for (li, l) in dev.layers.iter().enumerate() {
                // the layer input stays in h; the stream is sbuf
                ops.lin(&l.ff1.0, h, wide, n, 1)?;
                ops.lin(&l.ff1.1, wide, tmp, n, 0)?;
                ops.scale(sbuf, h, n * d, 1.0)?;
                ops.axpy(sbuf, tmp, n * d, 1.0)?;
                // attention
                ops.rms(h, h, l.in_norm.ptr, res, n, d, 1e-6, 1.0, false)?;
                ops.lin(&l.qk, res, qkb, n, 0)?;
                ops.lin(&l.vg, sbuf, wide, n, 0)?;
                c.launch(f.split, blocks(nh * n * hd / 2), 256, 0, kargs!(qkb, wide, cos.ptr, sin.ptr, q, k, v, n_i, nh_i, hd_i))?;
                c.gemm_batched(true, n, n, hd, scale, k, hd, n * hd, q, hd, n * hd, sc, n, n * n, nh, mode)?;
                c.launch(f.softmax, rows_grid(nh * n), 256, 0, kargs!(sc, (nh * n) as i32, n_i))?;
                c.gemm_batched(false, hd, n, n, 1.0, v, hd, n * hd, sc, n, n * n, o, hd, n * hd, nh, mode)?;
                c.launch(f.gate, blocks(n * hw), 256, 0, kargs!(o, wide, res, n_i, nh_i, hd_i))?;
                ops.lin(&l.attn_out, res, tmp, n, 0)?;
                ops.axpy(sbuf, tmp, n * d, 1.0)?;
                if let Some(cv) = &l.conv {
                    let k_i = cv.kernel as i32;
                    ops.lin(&cv.inp, sbuf, wide, n, 0)?;
                    c.launch(f.mean, (d as u32, 1), 256, 0, kargs!(wide, cv.mean_w.ptr, mean, n_i, d_i))?;
                    c.launch(f.conv, blocks(n * d), 256, 0, kargs!(wide, cv.taps.ptr, cv.bias.ptr, mean, res, n_i, d_i, k_i))?;
                    ops.lin(&cv.out, res, tmp, n, 0)?;
                    ops.axpy(sbuf, tmp, n * d, 1.0)?;
                }
                if let Some(ff2) = &l.ff2 {
                    ops.lin(&ff2.0, sbuf, wide, n, 1)?;
                    ops.lin(&ff2.1, wide, tmp, n, 0)?;
                    ops.axpy(sbuf, tmp, n * d, 1.0)?;
                }
                // h = norm(h + ¼ (stream − h))
                ops.rms(h, sbuf, l.norm.ptr, tmp, n, d, 1e-6, 0.25, false)?;
                ops.scale(h, tmp, n * d, 1.0)?;
                ops.axpy(bypass, h, n * d, st.residual_scales[li + 1])?;
            }
            ops.lin(&dev.proj_t, bypass, xout, n, 0)?;
            ops.axpy(xout, src, n * in_w, 1.0)?;
            ops.copy_pad(xb, xout, t * w, t * w)?;
        }
    }
    // the output norm: one scale for every lane
    unsafe {
        ops.rms(xb, xb, out_w, xin, t, w, a.out_eps * a.out_eps, 1.0, true)?;
        cu::memcpy_dtoh_async(x, xin, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if !x.iter().all(|v| v.is_finite()) {
        return Err("the encoder came back with a value that is not finite".into());
    }
    if a.prof {
        eprintln!("gigaam (cuda, resident): {t} frames, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}
