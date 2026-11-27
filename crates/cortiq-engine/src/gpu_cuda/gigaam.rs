//! The GigaAM encoder on CUDA (`gpu::gigaam_encode`): the three Zapformer
//! stacks of one segment with the activations on the card from the first
//! projection to the output norm — one upload, one readback. The weights go
//! up once a model (`key`) as f32 planes; the products are cuBLAS GEMMs on
//! the tensor cores (TF32 inside, plain f32 under `CMF_CUDA_PREC=exact`),
//! attention two batched GEMMs over the heads with a softmax between, and
//! everything else the kernels of `gigaam.cu`.

use super::*;
use crate::gpu::{GaFront, GaLin, GaVadArgs, GigaamArgs};
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
    conv2d: cs::CUfunction,
    flat: cs::CUfunction,
    ln: cs::CUfunction,
    patch1d: cs::CUfunction,
    rot_half: cs::CUfunction,
    planes: cs::CUfunction,
    glu_conv: cs::CUfunction,
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

/// The front end on the card: the three convolutions (weight, bias, channels
/// out), the depthwise one, the two pointwise ones as convolutions, the projection.
struct FrontDev {
    conv: Vec<(Buf, Buf, usize)>,
    dw: (Buf, Buf),
    pw1: LinDev,
    pw2: LinDev,
    out: LinDev,
}

fn up_front(fr: &GaFront) -> Res<FrontDev> {
    let mut scratch = Vec::new();
    let mut conv = Vec::with_capacity(3);
    for (w, b, co) in &fr.conv {
        conv.push((Buf::from_slice(w)?, Buf::from_slice(b)?, *co));
    }
    Ok(FrontDev {
        conv,
        dw: (Buf::from_slice(fr.dw.0)?, Buf::from_slice(fr.dw.1)?),
        pw1: up_lin(&fr.pw1, &mut scratch)?,
        pw2: up_lin(&fr.pw2, &mut scratch)?,
        out: up_lin(&fr.out, &mut scratch)?,
    })
}

/// A LayerNorm's weight and bias on the card.
type NormDev = (Buf, Buf);

struct VadLayerDev {
    norm_ff1: NormDev,
    ff1: (LinDev, LinDev),
    norm_att: NormDev,
    q: LinDev,
    k: LinDev,
    v: LinDev,
    o: LinDev,
    norm_conv: NormDev,
    pw1: LinDev,
    taps: Buf,
    dw_b: Buf,
    conv_norm: NormDev,
    pw2: LinDev,
    norm_ff2: NormDev,
    ff2: (LinDev, LinDev),
    norm_out: NormDev,
}

struct VadDev {
    pre: Vec<(Buf, Buf, usize, usize)>,
    layers: Vec<VadLayerDev>,
    head: LinDev,
    cos: Buf,
    sin: Buf,
}

fn up_vad(a: &GaVadArgs) -> Res<VadDev> {
    let mut sc = Vec::new();
    let norm = |n: (&[f32], &[f32])| -> Res<NormDev> { Ok((Buf::from_slice(n.0)?, Buf::from_slice(n.1)?)) };
    let mut pre = Vec::with_capacity(2);
    for (w, b, co, ci) in &a.pre {
        pre.push((Buf::from_slice(w)?, Buf::from_slice(b)?, *co, *ci));
    }
    let mut layers = Vec::with_capacity(a.layers.len());
    for l in &a.layers {
        layers.push(VadLayerDev {
            norm_ff1: norm(l.norm_ff1)?,
            ff1: (up_lin(&l.ff1.0, &mut sc)?, up_lin(&l.ff1.1, &mut sc)?),
            norm_att: norm(l.norm_att)?,
            q: up_lin(&l.q, &mut sc)?,
            k: up_lin(&l.k, &mut sc)?,
            v: up_lin(&l.v, &mut sc)?,
            o: up_lin(&l.o, &mut sc)?,
            norm_conv: norm(l.norm_conv)?,
            pw1: up_lin(&l.pw1, &mut sc)?,
            taps: Buf::from_slice(l.taps)?,
            dw_b: Buf::from_slice(l.dw_b)?,
            conv_norm: norm(l.conv_norm)?,
            pw2: up_lin(&l.pw2, &mut sc)?,
            norm_ff2: norm(l.norm_ff2)?,
            ff2: (up_lin(&l.ff2.0, &mut sc)?, up_lin(&l.ff2.1, &mut sc)?),
            norm_out: norm(l.norm_out)?,
        });
    }
    Ok(VadDev { pre, layers, head: up_lin(&a.head, &mut sc)?, cos: Buf::from_slice(a.cos)?, sin: Buf::from_slice(a.sin)? })
}

struct GaState {
    f: Option<GaFuncs>,
    vad: Option<(u64, VadDev)>,
    model: Option<(u64, Vec<StackDev>, Buf, Buf)>,
    front: Option<(u64, FrontDev)>,
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

static STATE: Mutex<GaState> = Mutex::new(GaState { f: None, vad: None, model: None, front: None, work: None });

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
        conv2d: func(m, "ga_conv2d")?,
        flat: func(m, "ga_flat")?,
        ln: func(m, "ga_ln")?,
        patch1d: func(m, "ga_patch1d")?,
        rot_half: func(m, "ga_rot_half")?,
        planes: func(m, "ga_planes")?,
        glu_conv: func(m, "ga_glu_conv")?,
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
    match unsafe { run(c, &mut s, a, x, t, None) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gigaam encode", &e);
            false
        }
    }
}

/// The front end and the stacks: `feats` (`[frames, n_mel]`) in, `out`
/// (`[t, width]`) the encoder output; `false` = declined.
pub fn gigaam_encode_feats(a: &GigaamArgs, feats: &[f32], frames: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || a.front.is_none() {
        return false;
    }
    let t = frames.div_ceil(2).div_ceil(2);
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    match unsafe { run(c, &mut s, a, out, t, Some((feats, frames))) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gigaam front end", &e);
            false
        }
    }
}

/// The front end over `feats` into the working buffer `x` (`[t, width]`).
unsafe fn front_run(c: &Ctx, f: GaFuncs, ops: &Ops, fr: &GaFront, dev: &FrontDev, bufs: &mut std::collections::HashMap<&'static str, Buf>, feats: &[f32], frames: usize, xb: u64) -> Res<usize> {
    let n_mel = fr.n_mel;
    if frames == 0 || feats.len() != frames * n_mel || dev.conv.len() != 3 {
        return Err("a front-end shape outside the kernels".into());
    }
    // sizes after each convolution: (channels, frames, bins)
    let s1 = (dev.conv[0].2, frames, n_mel);
    let s2 = (dev.conv[1].2, frames.div_ceil(2), (n_mel - 3) / 2 + 1);
    let s3 = (dev.conv[2].2, s2.1.div_ceil(2), (s2.2 - 3) / 2 + 1);
    let vol = |s: (usize, usize, usize)| s.0 * s.1 * s.2 * 4;
    let fin = work(c, bufs, "f_in", frames * n_mel * 4)?;
    let (a1, a2, a3, a4) = (work(c, bufs, "f_1", vol(s1))?, work(c, bufs, "f_2", vol(s2))?, work(c, bufs, "f_3", vol(s3))?, work(c, bufs, "f_4", vol(s3))?);
    let up = work(c, bufs, "f_up", dev.pw1.rows * s3.1 * s3.2 * 4)?;
    let conv = |x: u64, w: u64, b: u64, add: u64, y: u64, i: (usize, usize, usize), o: (usize, usize, usize), k: i32, st: i32, pt: i32, pf: i32, dw: i32, act: i32, has_add: i32| -> Res<()> {
        let (ci, t_i, f_i, co, to, fo) = (i.0 as i32, i.1 as i32, i.2 as i32, o.0 as i32, o.1 as i32, o.2 as i32);
        unsafe { c.launch(f.conv2d, blocks(o.0 * o.1 * o.2), 256, 0, kargs!(x, w, b, add, y, ci, co, t_i, f_i, to, fo, k, st, pt, pf, dw, act, has_add)) }
    };
    unsafe {
        cu::memcpy_htod_async(fin, feats, c.stream).map_err(derr)?;
        conv(fin, dev.conv[0].0.ptr, dev.conv[0].1.ptr, fin, a1, (1, frames, n_mel), s1, 3, 1, 2, 1, 0, 1, 0)?;
        conv(a1, dev.conv[1].0.ptr, dev.conv[1].1.ptr, a1, a2, s1, s2, 3, 2, 1, 0, 0, 1, 0)?;
        conv(a2, dev.conv[2].0.ptr, dev.conv[2].1.ptr, a2, a3, s2, s3, 3, 2, 1, 0, 0, 1, 0)?;
        // ConvNeXt: depthwise 3×3, 1×1 up with SiLU, 1×1 down added to the block input
        conv(a3, dev.dw.0.ptr, dev.dw.1.ptr, a3, a4, s3, s3, 3, 1, 2, 1, 1, 0, 0)?;
        let up_b = dev.pw1.b.as_ref().ok_or("the front end's pointwise convolutions carry a bias")?;
        let dn_b = dev.pw2.b.as_ref().ok_or("the front end's pointwise convolutions carry a bias")?;
        conv(a4, dev.pw1.w.ptr, up_b.ptr, a4, up, s3, (dev.pw1.rows, s3.1, s3.2), 1, 1, 0, 0, 0, 1, 0)?;
        conv(up, dev.pw2.w.ptr, dn_b.ptr, a3, a4, (dev.pw1.rows, s3.1, s3.2), s3, 1, 1, 0, 0, 0, 0, 1)?;
        let (c_i, t_i, f_i) = (s3.0 as i32, s3.1 as i32, s3.2 as i32);
        c.launch(f.flat, blocks(s3.0 * s3.1 * s3.2), 256, 0, kargs!(a4, a3, c_i, t_i, f_i))?;
        ops.lin(&dev.out, a3, xb, s3.1, 0)?;
    }
    Ok(s3.1)
}

unsafe fn run(c: &Ctx, s: &mut GaState, a: &GigaamArgs, x: &mut [f32], t: usize, feats: Option<(&[f32], usize)>) -> Res<()> {
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
    if let (Some(fr), true) = (&a.front, feats.is_some()) {
        if s.front.as_ref().is_none_or(|m| m.0 != a.key) {
            s.front = None;
            s.front = Some((a.key, up_front(fr)?));
        }
    }
    let GaState { model, work: bufs, front: front_dev, .. } = s;
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
    unsafe { cu::memcpy_htod_async(out_w, &out_scale, c.stream).map_err(derr)? };
    match (feats, &a.front, front_dev.as_ref()) {
        (Some((fe, frames)), Some(fr), Some((_, dev))) => {
            let got = unsafe { front_run(c, f, &ops, fr, dev, bufs, fe, frames, xb)? };
            if got != t {
                return Err(format!("the front end gave {got} frames for {t}"));
            }
        }
        (Some(_), _, _) => return Err("no front end on the card".into()),
        (None, _, _) => unsafe { cu::memcpy_htod_async(xb, &*x, c.stream).map_err(derr)? },
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

/// The voice-activity logits of `windows` windows of `frames` mel frames
/// (`feats` is `[windows·frames, n_mel]`) into `out`; `false` = declined.
pub fn gigaam_vad(a: &GaVadArgs, feats: &[f32], windows: usize, frames: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    match unsafe { vad_run(c, &mut s, a, feats, windows, frames, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gigaam vad", &e);
            false
        }
    }
}

unsafe fn vad_run(c: &Ctx, s: &mut GaState, a: &GaVadArgs, feats: &[f32], windows: usize, frames: usize, out: &mut [f32]) -> Res<()> {
    let (d, nh) = (a.dim, a.heads);
    let hd = d / nh.max(1);
    let half = |l: usize| (l - 1) / 2 + 1;
    if windows == 0 || frames == 0 || feats.len() != windows * frames * a.n_mel || nh * hd != d || hd % 2 != 0 {
        return Err("a voice-activity shape outside the kernels".into());
    }
    let (l1, steps) = (half(frames), half(half(frames)));
    if out.len() != windows * steps || a.cos.len() < steps * hd / 2 || a.pre[0].3 != a.n_mel || a.pre[1].2 != d {
        return Err("a voice-activity shape outside the kernels".into());
    }
    let f = funcs(c, s)?;
    if s.vad.as_ref().is_none_or(|m| m.0 != a.key || m.1.layers.len() != a.layers.len()) {
        s.vad = None;
        s.vad = Some((a.key, up_vad(a)?));
    }
    let GaState { vad, work: bufs, .. } = s;
    let dev = &vad.as_ref().expect("set above").1;
    let bufs = bufs.get_or_insert_with(Default::default);
    // every product in plain f32: a probability near a threshold must not move
    let ops = Ops { c, f, mode: Gemm::F32 };
    let n = windows * steps;
    let (c0, c1) = (a.pre[0].2, a.pre[1].2);
    let mut b = |name: &'static str, bytes: usize| work(c, bufs, name, bytes);
    let fin = b("v_in", feats.len() * 4)?;
    let patch = b("v_patch", (windows * l1 * 3 * a.n_mel).max(n * 3 * c0) * 4)?;
    let y1 = b("v_y1", windows * l1 * c0 * 4)?;
    let wide = a.layers.iter().map(|l| l.ff1.0.rows.max(l.pw1.rows)).max().unwrap_or(2 * d);
    let (r, xn, tmp, rot) = (b("v_r", n * c1.max(d) * 4)?, b("v_xn", n * d * 4)?, b("v_tmp", n * d * 4)?, b("v_rot", n * d * 4)?);
    let up = b("v_wide", n * wide * 4)?;
    let (q, k, v, qp, kp, vp) = (b("v_q", n * d * 4)?, b("v_k", n * d * 4)?, b("v_v", n * d * 4)?, b("v_qp", n * d * 4)?, b("v_kp", n * d * 4)?, b("v_vp", n * d * 4)?);
    let sc = b("v_scores", windows * nh * steps * steps * 4)?;
    let ln = |src: u64, w: &NormDev, dst: u64| -> Res<()> {
        let (r_i, c_i, eps) = (n as i32, d as i32, 1e-5f32);
        unsafe { c.launch(f.ln, rows_grid(n), 256, 0, kargs!(src, w.0.ptr, w.1.ptr, dst, r_i, c_i, eps)) }
    };
    let scale = (hd as f32).powf(-0.5);
    let (win_i, nh_i, hd_i, d_i, k_i) = (steps as i32, nh as i32, hd as i32, d as i32, a.kernel as i32);
    unsafe {
        cu::memcpy_htod_async(fin, feats, c.stream).map_err(derr)?;
        // the two strided convolutions: patches, a product, bias and ReLU
        let (tot, wi, wo, ci) = ((windows * l1 * 3 * a.n_mel) as u64, frames as i32, l1 as i32, a.n_mel as i32);
        c.launch(f.patch1d, blocks(tot as usize), 256, 0, kargs!(fin, patch, tot, wi, wo, ci))?;
        let (w0, b0) = (&dev.pre[0].0, &dev.pre[0].1);
        c.gemm(true, false, c0, windows * l1, 3 * a.n_mel, 1.0, w0.ptr, 3 * a.n_mel, patch, 3 * a.n_mel, 0.0, y1, c0, Gemm::F32)?;
        let (t1, c0_i, one, relu) = ((windows * l1 * c0) as u64, c0 as i32, 1i32, 2i32);
        c.launch(f.bias_act, blocks(t1 as usize), 256, 0, kargs!(y1, b0.ptr, t1, c0_i, one, relu))?;
        let (tot, wi, wo, ci) = ((n * 3 * c0) as u64, l1 as i32, steps as i32, c0 as i32);
        c.launch(f.patch1d, blocks(tot as usize), 256, 0, kargs!(y1, patch, tot, wi, wo, ci))?;
        let (w1, b1) = (&dev.pre[1].0, &dev.pre[1].1);
        c.gemm(true, false, c1, n, 3 * c0, 1.0, w1.ptr, 3 * c0, patch, 3 * c0, 0.0, r, c1, Gemm::F32)?;
        let (t2, c1_i) = ((n * c1) as u64, c1 as i32);
        c.launch(f.bias_act, blocks(t2 as usize), 256, 0, kargs!(r, b1.ptr, t2, c1_i, one, relu))?;
        for l in &dev.layers {
            // feed-forward, half weight
            ln(r, &l.norm_ff1, xn)?;
            ops.lin(&l.ff1.0, xn, up, n, 1)?;
            ops.lin(&l.ff1.1, up, tmp, n, 0)?;
            ops.axpy(r, tmp, n * d, 0.5)?;
            // attention inside each window
            ln(r, &l.norm_att, xn)?;
            let pairs = (n * nh * hd / 2) as u64;
            c.launch(f.rot_half, blocks(pairs as usize), 256, 0, kargs!(xn, dev.cos.ptr, dev.sin.ptr, rot, pairs, win_i, nh_i, hd_i))?;
            ops.lin(&l.q, rot, q, n, 0)?;
            ops.lin(&l.k, rot, k, n, 0)?;
            ops.lin(&l.v, xn, v, n, 0)?;
            let (tot, fwd, back) = ((n * d) as u64, 0i32, 1i32);
            for (src, dst) in [(q, qp), (k, kp), (v, vp)] {
                c.launch(f.planes, blocks(n * d), 256, 0, kargs!(src, dst, tot, win_i, nh_i, hd_i, fwd))?;
            }
            let count = windows * nh;
            c.gemm_batched(true, steps, steps, hd, scale, kp, hd, steps * hd, qp, hd, steps * hd, sc, steps, steps * steps, count, Gemm::F32)?;
            c.launch(f.softmax, rows_grid(count * steps), 256, 0, kargs!(sc, (count * steps) as i32, win_i))?;
            c.gemm_batched(false, hd, steps, steps, 1.0, vp, hd, steps * hd, sc, steps, steps * steps, qp, hd, steps * hd, count, Gemm::F32)?;
            c.launch(f.planes, blocks(n * d), 256, 0, kargs!(qp, xn, tot, win_i, nh_i, hd_i, back))?;
            ops.lin(&l.o, xn, tmp, n, 0)?;
            ops.axpy(r, tmp, n * d, 1.0)?;
            // convolution module
            ln(r, &l.norm_conv, xn)?;
            ops.lin(&l.pw1, xn, up, n, 0)?;
            c.launch(f.glu_conv, blocks(n * d), 256, 0, kargs!(up, l.taps.ptr, l.dw_b.ptr, tmp, tot, win_i, d_i, k_i))?;
            ln(tmp, &l.conv_norm, xn)?;
            let (zero, silu) = (0i32, 1i32);
            c.launch(f.bias_act, blocks(n * d), 256, 0, kargs!(xn, xn, tot, d_i, zero, silu))?;
            ops.lin(&l.pw2, xn, tmp, n, 0)?;
            ops.axpy(r, tmp, n * d, 1.0)?;
            // feed-forward, half weight; the output norm
            ln(r, &l.norm_ff2, xn)?;
            ops.lin(&l.ff2.0, xn, up, n, 1)?;
            ops.lin(&l.ff2.1, up, tmp, n, 0)?;
            ops.axpy(r, tmp, n * d, 0.5)?;
            ln(r, &l.norm_out, xn)?;
            ops.scale(r, xn, n * d, 1.0)?;
        }
        ops.lin(&dev.head, r, tmp, n, 0)?;
        cu::memcpy_dtoh_async(out, tmp, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if !out.iter().all(|v| v.is_finite()) {
        return Err("the voice-activity model came back with a value that is not finite".into());
    }
    Ok(())
}
