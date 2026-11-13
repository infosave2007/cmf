//! The HunyuanVideo VAE decoder (Kandinsky S2V's) on CUDA: each temporal tile
//! of `s2v_vae::tiled` resident on the card, one upload and one readback. The
//! stream is f16 channel-major `[C][F·H·W]` (the reference decodes under bf16
//! autocast; at 512² the last stages hold 128-256 channels over millions of
//! positions). A 3-D causal conv is an im2col with replicate padding into f16
//! and an f16 cuBLAS product a chunk of positions at a time, stored with its
//! bias and residual; GroupNorm three passes with SiLU; the mid attention one
//! query frame at a time over the keys of its frame and the ones before it
//! (TF32 products around a softmax). The weights go up once as f16.
//! `CMF_S2V_VAE_CUDA=0` keeps the host decoder.

use super::*;
use crate::ltxvae::Vol;
use crate::s2v_vae::{Conv, GroupNorm, Mid, Res as VRes, S2vVae, S2vVaeEncoder};

const HV_SRC: &str = include_str!("s2v_vae.cu");
const GN_EPS: f32 = 1e-6;
const GN_CHUNK: usize = 1 << 16;
/// The bytes of one im2col panel or product chunk.
const PANEL: usize = 192 << 20;

#[derive(Clone, Copy)]
pub(super) struct HvFuncs {
    im2col: cs::CUfunction,
    store: cs::CUfunction,
    part: cs::CUfunction,
    fin: cs::CUfunction,
    apply: cs::CUfunction,
    up: cs::CUfunction,
    to32: cs::CUfunction,
    to16: cs::CUfunction,
    softmax: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<HvFuncs> {
    if let Some(f) = st.h3.hv_funcs {
        return Ok(f);
    }
    let m = compile(HV_SRC, Vec::new(), c.arch)?;
    let f = HvFuncs {
        im2col: func(m, "hv_im2col")?,
        store: func(m, "hv_store")?,
        part: func(m, "hv_gn_part")?,
        fin: func(m, "hv_gn_fin")?,
        apply: func(m, "hv_gn_apply")?,
        up: func(m, "hv_up")?,
        to32: func(m, "hv_to_f32")?,
        to16: func(m, "hv_to_f16")?,
        softmax: func(m, "hv_softmax")?,
    };
    st.h3.hv_funcs = Some(f);
    Ok(f)
}

pub(super) struct HvConv {
    w: Buf,
    b: Buf,
    cin: usize,
    cout: usize,
    k: [usize; 3],
    stride: [usize; 3],
}

pub(super) struct HvNorm {
    w: Buf,
    b: Buf,
    c: usize,
}

pub(super) struct HvRes {
    n1: HvNorm,
    c1: HvConv,
    n2: HvNorm,
    c2: HvConv,
    short: Option<HvConv>,
}

pub(super) struct HvMid {
    r0: HvRes,
    gn: HvNorm,
    qkvo: [HvConv; 4],
    r1: HvRes,
}

pub(super) struct HvDev {
    post_quant: HvConv,
    conv_in: HvConv,
    mid: HvMid,
    ups: Vec<(Vec<HvRes>, Option<(HvConv, [usize; 3])>)>,
    norm_out: HvNorm,
    conv_out: HvConv,
}

unsafe fn up_conv(c: &Ctx, to_half: cs::CUfunction, stage: u64, w: &[f32], b: &[f32], cin: usize, cout: usize, k: [usize; 3]) -> Res<HvConv> {
    if w.len() != cout * cin * k[0] * k[1] * k[2] || b.len() != cout {
        return Err("a VAE conv of an unexpected shape".into());
    }
    let wb = Buf::direct(w.len() * 2)?;
    let bb = Buf::direct(b.len() * 4)?;
    let n = w.len() as u64;
    unsafe {
        cu::memcpy_htod_async(stage, w, c.stream).map_err(derr)?;
        c.launch(to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(stage, wb.ptr, n))?;
        cu::memcpy_htod_async(bb.ptr, b, c.stream).map_err(derr)?;
    }
    c.sync()?;
    Ok(HvConv { w: wb, b: bb, cin, cout, k, stride: [1, 1, 1] })
}

unsafe fn conv_of(c: &Ctx, th: cs::CUfunction, stage: u64, cv: &Conv) -> Res<HvConv> {
    if cv.stride != [1, 1, 1] {
        return Err("a strided VAE conv".into());
    }
    unsafe { up_conv(c, th, stage, &cv.w, &cv.b, cv.cin, cv.cout, cv.k) }
}

/// A conv of any stride (the encoder's downsamplers).
unsafe fn conv_any(c: &Ctx, th: cs::CUfunction, stage: u64, cv: &Conv) -> Res<HvConv> {
    let mut h = unsafe { up_conv(c, th, stage, &cv.w, &cv.b, cv.cin, cv.cout, cv.k)? };
    h.stride = cv.stride;
    Ok(h)
}

unsafe fn mid_of(c: &Ctx, th: cs::CUfunction, stage: u64, m: &Mid) -> Res<HvMid> {
    let lin = |l: &(Vec<f32>, Vec<f32>)| -> Res<HvConv> {
        let n = l.1.len();
        unsafe { up_conv(c, th, stage, &l.0, &l.1, n, n, [1, 1, 1]) }
    };
    unsafe {
        Ok(HvMid {
            r0: res_of(c, th, stage, &m.r0)?,
            gn: norm_of(c, &m.gn)?,
            qkvo: [lin(&m.q)?, lin(&m.k)?, lin(&m.v)?, lin(&m.o)?],
            r1: res_of(c, th, stage, &m.r1)?,
        })
    }
}

fn norm_of(c: &Ctx, n: &GroupNorm) -> Res<HvNorm> {
    let w = Buf::direct(n.w.len() * 4)?;
    let b = Buf::direct(n.b.len() * 4)?;
    unsafe {
        cu::memcpy_htod_async(w.ptr, &n.w, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(b.ptr, &n.b, c.stream).map_err(derr)?;
    }
    Ok(HvNorm { w, b, c: n.w.len() })
}

unsafe fn res_of(c: &Ctx, th: cs::CUfunction, stage: u64, r: &VRes) -> Res<HvRes> {
    unsafe {
        Ok(HvRes {
            n1: norm_of(c, &r.n1)?,
            c1: conv_of(c, th, stage, &r.c1)?,
            n2: norm_of(c, &r.n2)?,
            c2: conv_of(c, th, stage, &r.c2)?,
            short: match &r.short {
                Some(s) => Some(conv_of(c, th, stage, s)?),
                None => None,
            },
        })
    }
}

unsafe fn build(c: &Ctx, st: &mut State, vae: &S2vVae) -> Res<HvDev> {
    let th = st.h3.funcs(c)?.to_half;
    let mut most = 0usize;
    let mut see = |cv: &Conv| most = most.max(cv.w.len());
    see(&vae.post_quant);
    see(&vae.conv_in);
    see(&vae.conv_out);
    for r in [&vae.mid.r0, &vae.mid.r1].into_iter().chain(vae.ups.iter().flat_map(|u| u.res.iter())) {
        see(&r.c1);
        see(&r.c2);
        if let Some(s) = &r.short {
            see(s);
        }
    }
    for u in &vae.ups {
        if let Some((cv, _)) = &u.conv {
            see(cv);
        }
    }
    let stage = st.h3.buf(c, "hv stage", most.max(vae.mid.q.0.len()) * 4)?;
    unsafe {
        let mut ups = Vec::with_capacity(vae.ups.len());
        for u in &vae.ups {
            let res = u.res.iter().map(|r| res_of(c, th, stage, r)).collect::<Res<Vec<_>>>()?;
            let conv = match &u.conv {
                Some((cv, f)) => Some((conv_of(c, th, stage, cv)?, *f)),
                None => None,
            };
            ups.push((res, conv));
        }
        Ok(HvDev {
            post_quant: conv_of(c, th, stage, &vae.post_quant)?,
            conv_in: conv_of(c, th, stage, &vae.conv_in)?,
            mid: mid_of(c, th, stage, &vae.mid)?,
            ups,
            norm_out: norm_of(c, &vae.norm_out)?,
            conv_out: conv_of(c, th, stage, &vae.conv_out)?,
        })
    }
}

/// The whole tiled decode on the card, or `None` (said once) for the host's.
pub fn s2v_vae_decode(vae: &S2vVae, z: &Vol) -> Option<Vol> {
    if std::env::var("CMF_S2V_VAE_CUDA").as_deref() == Ok("0") {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    // the denoiser's working set is not needed while the decoder runs (the
    // weights stay, the next chunk's steps reuse them)
    let _ = c.sync();
    st.h3.bufs.retain(|k, _| !k.starts_with("s2 ") && !k.starts_with("s2s "));
    if st.h3.hv.is_none() {
        match unsafe { build(c, &mut st, vae) } {
            Ok(d) => st.h3.hv = Some(d),
            Err(e) => {
                decline("hunyuan vae", &e);
                return None;
            }
        }
    }
    let dev = st.h3.hv.take().unwrap();
    let mut failed: Option<String> = None;
    let out = crate::s2v_vae::tiled(z, &mut |t| {
        if failed.is_some() {
            return Vol::zeros(3, 1, 1, 1);
        }
        match unsafe { tile(c, &mut st, &dev, t) } {
            Ok(v) => v,
            Err(e) => {
                failed = Some(e);
                Vol::zeros(3, 1, 1, 1)
            }
        }
    });
    st.h3.hv = Some(dev);
    match failed {
        None => Some(out),
        Some(e) => {
            let _ = c.sync();
            decline("hunyuan vae", &e);
            None
        }
    }
}

/// Drop the decoder's weights (the render is done).
pub fn s2v_vae_release() {
    if let Some(c) = ctx()
        && let Ok(mut st) = c.st.lock()
    {
        let _ = c.sync();
        st.h3.hv = None;
    }
}

type Dims = (usize, usize, usize);

/// The output's (F, H, W) of a causal conv over `d` (replicate padding: kt−1
/// frames in front, k/2 in space).
fn out_dims(cv: &HvConv, d: Dims) -> Dims {
    let o = |n: usize, k: usize, s: usize, front: usize, back: usize| (n + front + back - k) / s + 1;
    (
        o(d.0, cv.k[0], cv.stride[0], cv.k[0] - 1, 0),
        o(d.1, cv.k[1], cv.stride[1], cv.k[1] / 2, cv.k[1] / 2),
        o(d.2, cv.k[2], cv.stride[2], cv.k[2] / 2, cv.k[2] / 2),
    )
}

/// One pass over the four f16 stream buffers: convs, norms, resnets and the
/// mid attention, with `CMF_S2V_CUDA_PROF=1` seconds per class.
struct Run<'a> {
    c: &'a Ctx,
    f: HvFuncs,
    b: u64,
    s: u64,
    prof: bool,
    times: [std::cell::Cell<f64>; 3],
}

impl Run<'_> {
    fn timed(&self, k: usize, t0: std::time::Instant) {
        if self.prof {
            let _ = self.c.sync();
            self.times[k].set(self.times[k].get() + t0.elapsed().as_secs_f64());
        }
    }

    /// `dst` [cout][F'·H'·W'] = conv(src [cin][F·H·W]) + bias (+ res); the output's dims.
    unsafe fn conv(&self, st: &mut State, cv: &HvConv, src: u64, d: Dims, dst: u64, res: u64) -> Res<Dims> {
        let (c, f) = (self.c, &self.f);
        let t0 = std::time::Instant::now();
        let od = out_dims(cv, d);
        let p = od.0 * od.1 * od.2;
        let taps = cv.k[0] * cv.k[1] * cv.k[2];
        let kk = cv.cin * taps;
        let np = (PANEL / (kk * 2)).min(PANEL / (cv.cout * 4)).clamp(1, p);
        let cols = st.h3.buf(c, "hv cols", np * kk * 2)?;
        let yc = st.h3.buf(c, "hv yc", np * cv.cout * 4)?;
        let (ci, fi, hi, wi) = (cv.cin as i32, d.0 as i32, d.1 as i32, d.2 as i32);
        let (kt, kh, kw, co) = (cv.k[0] as i32, cv.k[1] as i32, cv.k[2] as i32, cv.cout as i32);
        let (ho, wo) = (od.1 as i32, od.2 as i32);
        let (s0, s1, s2) = (cv.stride[0] as i32, cv.stride[1] as i32, cv.stride[2] as i32);
        let pu = p as u64;
        let mut p0 = 0usize;
        while p0 < p {
            let n = np.min(p - p0);
            let (p0u, total, nu) = (p0 as u64, (n * cv.cin) as u64, n as u64);
            unsafe {
                c.launch(
                    f.im2col,
                    (total.div_ceil(256) as u32, 1),
                    256,
                    0,
                    kargs!(src, cols, ci, fi, hi, wi, kt, kh, kw, ho, wo, s0, s1, s2, p0u, nu),
                )?;
                // y [cout][n] = cols ([K][n], K-major) through the weights [cout][K]
                c.gemm(false, false, n, cv.cout, kk, 1.0, cols, n, cv.w.ptr, kk, 0.0, yc, n, Gemm::Half)?;
                let tot = (n * cv.cout) as u64;
                c.launch(f.store, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(yc, dst, cv.b.ptr, res, co, nu, pu, p0u))?;
            }
            p0 += n;
        }
        self.timed(0, t0);
        Ok(od)
    }

    /// GroupNorm (32 groups) of `src` [cc][P] into `dst`, SiLU under `silu`.
    unsafe fn gn(&self, st: &mut State, nr: &HvNorm, src: u64, dst: u64, p: usize, silu: bool) -> Res<()> {
        let (c, f) = (self.c, &self.f);
        let g = 32usize;
        if !nr.c.is_multiple_of(g) {
            return Err("a GroupNorm of an unexpected width".into());
        }
        let t0 = std::time::Instant::now();
        let span = nr.c / g * p;
        let nchunk = span.div_ceil(GN_CHUNK);
        let part = st.h3.buf(c, "hv part", g * nchunk * 16)?;
        let stats = st.h3.buf(c, "hv stats", g * 8 + 64)?;
        let (spanu, chu, nci, cnt) = (span as u64, GN_CHUNK as u64, nchunk as i32, span as f64);
        let (pu, peri, si, tot) = (p as u64, (nr.c / g) as i32, silu as i32, (nr.c * p) as u64);
        unsafe {
            c.launch(f.part, (nchunk as u32, g as u32), 256, 0, kargs!(src, part, spanu, chu, nci))?;
            c.launch(f.fin, (g as u32, 1), 32, 0, kargs!(part, stats, nci, cnt, GN_EPS))?;
            c.launch(f.apply, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(src, dst, stats, nr.w.ptr, nr.b.ptr, pu, peri, si, tot))?;
        }
        self.timed(1, t0);
        Ok(())
    }

    /// A resnet over `a` [cin][P] (scratch `t`, `b`, `s`): the result lands in `a`.
    unsafe fn resnet(&self, st: &mut State, r: &HvRes, a: &mut u64, t: &mut u64, d: Dims) -> Res<()> {
        let p = d.0 * d.1 * d.2;
        unsafe {
            self.gn(st, &r.n1, *a, *t, p, true)?;
            self.conv(st, &r.c1, *t, d, self.b, 0)?;
            self.gn(st, &r.n2, self.b, self.b, p, true)?;
            let res = match &r.short {
                Some(sc) => {
                    self.conv(st, sc, *a, d, self.s, 0)?;
                    self.s
                }
                None => *a,
            };
            self.conv(st, &r.c2, self.b, d, *t, res)?;
        }
        std::mem::swap(a, t);
        Ok(())
    }

    /// The mid block: resnet, the attention of each query frame over the keys
    /// of its frame and the ones before it, resnet; over `a` in place.
    unsafe fn mid(&self, st: &mut State, m: &HvMid, a: &mut u64, t: &mut u64, d: Dims) -> Res<()> {
        let (c, f, b) = (self.c, &self.f, self.b);
        unsafe {
            self.resnet(st, &m.r0, a, t, d)?;
            let cc = m.gn.c;
            let per = d.1 * d.2;
            let p = d.0 * per;
            self.gn(st, &m.gn, *a, *t, p, false)?;
            let q32 = st.h3.buf(c, "hv q32", cc * p * 4)?;
            let k32 = st.h3.buf(c, "hv k32", cc * p * 4)?;
            let v32 = st.h3.buf(c, "hv v32", cc * p * 4)?;
            let o32 = st.h3.buf(c, "hv o32", cc * p * 4)?;
            let sc = st.h3.buf(c, "hv sc", per * p * 4)?;
            let n = (cc * p) as u64;
            for (i, dst) in [q32, k32, v32].into_iter().enumerate() {
                self.conv(st, &m.qkvo[i], *t, d, b, 0)?;
                c.launch(f.to32, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(b, dst, n))?;
            }
            let scale = 1.0 / (cc as f32).sqrt();
            let t0 = std::time::Instant::now();
            for fr in 0..d.0 {
                let live = (fr + 1) * per;
                let q0 = fr * per;
                let (li, ni) = (live as i32, live as i32);
                c.gemm(false, true, live, per, cc, scale, k32, p, q32 + (q0 * 4) as u64, p, 0.0, sc, live, Gemm::Tf32)?;
                c.launch(f.softmax, (per as u32, 1), 256, 0, kargs!(sc, ni, li))?;
                c.gemm(true, false, per, cc, live, 1.0, sc, live, v32, p, 0.0, o32 + (q0 * 4) as u64, p, Gemm::Tf32)?;
            }
            c.launch(f.to16, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(o32, b, n))?;
            self.timed(2, t0);
            self.conv(st, &m.qkvo[3], b, d, *t, *a)?;
            std::mem::swap(a, t);
            self.resnet(st, &m.r1, a, t, d)
        }
    }

    fn report(&self, what: &str, d: Dims, t_all: std::time::Instant) {
        if self.prof {
            eprintln!(
                "s2v cuda vae {what} {}x{}x{}: {:.2} s (conv {:.2}, norm {:.2}, attention {:.2})",
                d.0,
                d.1,
                d.2,
                t_all.elapsed().as_secs_f64(),
                self.times[0].get(),
                self.times[1].get(),
                self.times[2].get()
            );
        }
    }
}

/// The four stream buffers sized for `most` values, the input `x` uploaded
/// as f16 into the third.
unsafe fn start<'a>(c: &'a Ctx, st: &mut State, most: usize, x: &Vol) -> Res<(Run<'a>, [u64; 2], u64)> {
    let f = funcs(st, c)?;
    let a = st.h3.buf(c, "hv a", most * 2)?;
    let b = st.h3.buf(c, "hv b", most * 2)?;
    let t = st.h3.buf(c, "hv t", most * 2)?;
    let s = st.h3.buf(c, "hv s", most * 2)?;
    let zin = st.h3.buf(c, "hv z", x.data.len() * 4)?;
    unsafe {
        cu::memcpy_htod_async(zin, &x.data, c.stream).map_err(derr)?;
        let n = x.data.len() as u64;
        c.launch(f.to16, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(zin, t, n))?;
    }
    let prof = std::env::var("CMF_S2V_CUDA_PROF").as_deref() == Ok("1");
    let z = std::cell::Cell::new(0f64);
    Ok((Run { c, f, b, s, prof, times: [z.clone(), z.clone(), z] }, [a, t], t))
}

/// The f16 stream `src` [ch][F·H·W] back as f32 on the host.
unsafe fn finish(c: &Ctx, st: &mut State, f: &HvFuncs, src: u64, ch: usize, d: Dims) -> Res<Vol> {
    let n = (ch * d.0 * d.1 * d.2) as u64;
    let o32 = st.h3.buf(c, "hv out", n as usize * 4)?;
    let mut out = Vol::zeros(ch, d.0, d.1, d.2);
    unsafe {
        c.launch(f.to32, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(src, o32, n))?;
        cu::memcpy_dtoh_async(&mut out.data, o32, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if out.data.iter().any(|v| !v.is_finite()) {
        return Err("the VAE came back with a value that is not finite".into());
    }
    Ok(out)
}

/// The largest stream (values) of a chain of convs from `d`: each conv's input
/// and output, a resnet's widths at its dims.
struct Most {
    most: usize,
    d: Dims,
}

impl Most {
    fn conv(&mut self, cv: &HvConv) {
        let p = self.d.0 * self.d.1 * self.d.2;
        let od = out_dims(cv, self.d);
        self.most = self.most.max(cv.cin * p).max(cv.cout * od.0 * od.1 * od.2);
        self.d = od;
    }
    fn res(&mut self, r: &HvRes) {
        let p = self.d.0 * self.d.1 * self.d.2;
        self.most = self.most.max(r.c1.cin.max(r.c1.cout).max(r.c2.cout) * p);
    }
}

unsafe fn tile(c: &Ctx, st: &mut State, d: &HvDev, z: &Vol) -> Res<Vol> {
    // the shapes of the pass: the largest stream
    let mut m = Most { most: z.c * z.f * z.h * z.w, d: (z.f, z.h, z.w) };
    m.conv(&d.post_quant);
    m.conv(&d.conv_in);
    m.res(&d.mid.r0);
    for (res, up) in &d.ups {
        for r in res {
            m.res(r);
        }
        if let Some((cv, fct)) = up {
            let ch = res.last().map_or(cv.cin, |r| r.c2.cout);
            m.d = (if fct[0] == 2 { 1 + (m.d.0 - 1) * 2 } else { m.d.0 }, m.d.1 * fct[1], m.d.2 * fct[2]);
            m.most = m.most.max(ch * m.d.0 * m.d.1 * m.d.2);
            m.conv(cv);
        }
    }
    let od = m.d;
    let t_all = std::time::Instant::now();
    unsafe {
        let (run, [mut a, mut t], zin) = start(c, st, m.most, z)?;
        let f = run.f;
        let b = run.b;
        let mut dims = (z.f, z.h, z.w);
        dims = run.conv(st, &d.post_quant, zin, dims, b, 0)?;
        dims = run.conv(st, &d.conv_in, b, dims, a, 0)?;
        run.mid(st, &d.mid, &mut a, &mut t, dims)?;
        for (res, up) in &d.ups {
            for r in res {
                run.resnet(st, r, &mut a, &mut t, dims)?;
            }
            if let Some((cv, fct)) = up {
                let (fo, ho, wo) = (if fct[0] == 2 { 1 + (dims.0 - 1) * 2 } else { dims.0 }, dims.1 * fct[1], dims.2 * fct[2]);
                let tot = (cv.cin * fo * ho * wo) as u64;
                let (ci, fi, hi, wi, fti, fsi) = (cv.cin as i32, dims.0 as i32, dims.1 as i32, dims.2 as i32, fct[0] as i32, fct[1] as i32);
                c.launch(f.up, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(a, b, ci, fi, hi, wi, fti, fsi, tot))?;
                dims = run.conv(st, cv, b, (fo, ho, wo), a, 0)?;
            }
        }
        if dims != od {
            return Err("the decoder's shapes went astray".into());
        }
        let p = dims.0 * dims.1 * dims.2;
        run.gn(st, &d.norm_out, a, b, p, true)?;
        dims = run.conv(st, &d.conv_out, b, dims, t, 0)?;
        let out = finish(c, st, &f, t, d.conv_out.cout, dims)?;
        run.report("tile", dims, t_all);
        Ok(out)
    }
}

/// The encoder's weights on the card.
pub(super) struct HvEnc {
    conv_in: HvConv,
    downs: Vec<(Vec<HvRes>, Option<HvConv>)>,
    mid: HvMid,
    norm_out: HvNorm,
    conv_out: HvConv,
}

unsafe fn build_enc(c: &Ctx, st: &mut State, e: &S2vVaeEncoder) -> Res<HvEnc> {
    let th = st.h3.funcs(c)?.to_half;
    let mut most = e.conv_in.w.len().max(e.conv_out.w.len()).max(e.mid.q.0.len());
    for (res, dn) in &e.downs {
        for r in res {
            most = most.max(r.c1.w.len()).max(r.c2.w.len()).max(r.short.as_ref().map_or(0, |s| s.w.len()));
        }
        if let Some(cv) = dn {
            most = most.max(cv.w.len());
        }
    }
    for r in [&e.mid.r0, &e.mid.r1] {
        most = most.max(r.c1.w.len()).max(r.c2.w.len()).max(r.short.as_ref().map_or(0, |s| s.w.len()));
    }
    let stage = st.h3.buf(c, "hv stage", most * 4)?;
    unsafe {
        let mut downs = Vec::with_capacity(e.downs.len());
        for (res, dn) in &e.downs {
            let rs = res.iter().map(|r| res_of(c, th, stage, r)).collect::<Res<Vec<_>>>()?;
            let d = match dn {
                Some(cv) => Some(conv_any(c, th, stage, cv)?),
                None => None,
            };
            downs.push((rs, d));
        }
        Ok(HvEnc {
            conv_in: conv_of(c, th, stage, &e.conv_in)?,
            downs,
            mid: mid_of(c, th, stage, &e.mid)?,
            norm_out: norm_of(c, &e.norm_out)?,
            conv_out: conv_of(c, th, stage, &e.conv_out)?,
        })
    }
}

/// The reference frames `[3, F, H, W]` through the encoder on the card, up to
/// `conv_out`; the quant conv and the posterior's split stay on the host.
pub fn s2v_vae_encode(enc: &S2vVaeEncoder, img: &Vol) -> Option<(Vol, Vol)> {
    if std::env::var("CMF_S2V_VAE_CUDA").as_deref() == Ok("0") {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    let r = (|| -> Res<Vol> {
        let e = unsafe { build_enc(c, &mut st, enc)? };
        let mut m = Most { most: img.c * img.f * img.h * img.w, d: (img.f, img.h, img.w) };
        m.conv(&e.conv_in);
        for (res, dn) in &e.downs {
            for r in res {
                m.res(r);
            }
            if let Some(cv) = dn {
                m.conv(cv);
            }
        }
        m.res(&e.mid.r0);
        let t_all = std::time::Instant::now();
        unsafe {
            let (run, [mut a, mut t], xin) = start(c, &mut st, m.most, img)?;
            let mut dims = (img.f, img.h, img.w);
            dims = run.conv(&mut st, &e.conv_in, xin, dims, a, 0)?;
            for (res, dn) in &e.downs {
                for r in res {
                    run.resnet(&mut st, r, &mut a, &mut t, dims)?;
                }
                if let Some(cv) = dn {
                    dims = run.conv(&mut st, cv, a, dims, t, 0)?;
                    std::mem::swap(&mut a, &mut t);
                }
            }
            run.mid(&mut st, &e.mid, &mut a, &mut t, dims)?;
            let p = dims.0 * dims.1 * dims.2;
            run.gn(&mut st, &e.norm_out, a, run.b, p, true)?;
            dims = run.conv(&mut st, &e.conv_out, run.b, dims, t, 0)?;
            let out = finish(c, &mut st, &run.f, t, e.conv_out.cout, dims)?;
            run.report("encode", dims, t_all);
            Ok(out)
        }
    })();
    // the encoder runs once a render: its weights and buffers go
    let _ = c.sync();
    st.h3.bufs.retain(|k, _| !k.starts_with("hv "));
    match r {
        Ok(h) => Some(enc.finish_host(&h)),
        Err(e) => {
            decline("hunyuan vae encoder", &e);
            None
        }
    }
}
