//! The FLUX-class VAE decoder (diffusers `AutoencoderKL`: Lumina-Image 2.0,
//! Z-Image, MOSS-Image's Flux2 decoder) on CUDA as one resident chain: the
//! latent goes up once and the picture comes back once. Between them every
//! convolution is the LTX chain's im2col (nearest ×2 upsampling folded into
//! the gather) with an f16 cuBLAS product, GroupNorm three passes of its own
//! (`flux_vae.cu`), the mid-block's single wide head TF32 products around an
//! f32 softmax; residuals land as the second product's accumulator. The
//! weights come from the host slices on every call (a decoder runs once a
//! picture). `CMF_FLUX_VAE_CUDA=0` keeps the per-operation path.

use super::*;
use crate::vae::{VaeChainArgs, VaeConvRef, VaeNormRef, VaeResnetRef};

const FV_SRC: &str = include_str!("flux_vae.cu");
const GN_EPS: f32 = 1e-6;
/// Values of a group's span one block of the first GroupNorm pass sums.
const GN_CHUNK: usize = 1 << 16;
/// The bytes of one im2col panel or one chunk of attention scores.
const PANEL: usize = 256 << 20;

#[derive(Clone, Copy)]
pub(super) struct FvFuncs {
    part: cs::CUfunction,
    fin: cs::CUfunction,
    apply: cs::CUfunction,
    bias_rows: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<FvFuncs> {
    if let Some(f) = st.h3.flux_vae {
        return Ok(f);
    }
    let module = compile(FV_SRC, Vec::new(), c.arch)?;
    let f = FvFuncs {
        part: func(module, "fv_gn_part")?,
        fin: func(module, "fv_gn_fin")?,
        apply: func(module, "fv_gn_apply")?,
        bias_rows: func(module, "fv_bias_rows")?,
    };
    st.h3.flux_vae = Some(f);
    Ok(f)
}

/// The decoder over the de-normalised latent `z` `[latent_channels][h][w]`
/// into `out` `[3][H][W]`. `false`: nothing was written.
pub fn flux_vae_chain(a: &VaeChainArgs<'_>, z: &[f32], h: usize, w: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_FLUX_VAE_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { run(c, &mut st, a, z, h, w, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("flux vae chain", &e);
            false
        }
    }
}

/// The largest tensor of the pass, in values, and the picture's size.
fn plan(a: &VaeChainArgs<'_>, h: usize, w: usize) -> (usize, usize, usize) {
    let (mut hh, mut ww) = (h, w);
    let mut most = a.latent_channels.max(a.conv_in.oc) * h * w;
    let res = |r: &VaeResnetRef<'_>, hw: usize, most: &mut usize| *most = (*most).max(r.conv1.ic.max(r.conv1.oc).max(r.conv2.oc) * hw);
    res(&a.mid_res1, h * w, &mut most);
    res(&a.mid_res2, h * w, &mut most);
    for up in &a.ups {
        for r in &up.resnets {
            res(r, hh * ww, &mut most);
        }
        if let Some(u) = &up.upsample {
            hh *= 2;
            ww *= 2;
            most = most.max(u.ic.max(u.oc) * hh * ww);
        }
    }
    most = most.max(a.conv_out.ic.max(a.conv_out.oc) * hh * ww);
    (most, hh, ww)
}

unsafe fn run(c: &Ctx, st: &mut State, a: &VaeChainArgs<'_>, z: &[f32], h: usize, w: usize, out: &mut [f32]) -> Res<()> {
    let (most, oh, ow) = plan(a, h, w);
    if z.len() != a.latent_channels * h * w || a.conv_in.ic != a.latent_channels || out.len() != a.conv_out.oc * oh * ow {
        return Err("a latent or picture of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let lx = super::ltx::funcs(st, c)?;
    let to_half = st.h3.funcs(c)?.to_half;
    let bufs = [st.h3.buf(c, "fv x", most * 4)?, st.h3.buf(c, "fv t", most * 4)?, st.h3.buf(c, "fv u", most * 4)?, st.h3.buf(c, "fv s", most * 4)?];
    let (mut x, mut t, u, s) = (bufs[0], bufs[1], bufs[2], bufs[3]);
    let nul = 0u64;
    unsafe { cu::memcpy_htod_async(x, z, c.stream).map_err(derr)? };

    // ── the parts ──
    // `y` [oc][oh·ow] (= or +=, `acc`) of the conv over `src` [ic][h][w],
    // nearest ×2 first under `up`; then the bias
    let conv = |st: &mut State, cv: &VaeConvRef<'_>, src: u64, h: usize, w: usize, up: bool, y: u64, acc: bool| -> Res<()> {
        let (ci, co, k) = (cv.ic, cv.oc, cv.k);
        let (oh, ow) = if up { (2 * h, 2 * w) } else { (h, w) };
        let (p, kk) = (oh * ow, ci * k * k);
        if k.is_multiple_of(2) || cv.w.len() != co * kk || cv.b.len() != co {
            return Err("a convolution of an unexpected shape".into());
        }
        let wf = st.h3.buf(c, "fv wf", co * kk * 4)?;
        let wh = st.h3.buf(c, "fv wh", co * kk * 2 + 64)?;
        let bb = st.h3.buf(c, "fv b", co * 4 + 64)?;
        let chunk = (PANEL / (kk * 2)).clamp(1, p);
        let cols = st.h3.buf(c, "fv cols", chunk * kk * 2)?;
        let nw = (co * kk) as u64;
        let (cii, hi, wi, owi, ohi, ki, upi) = (ci as i32, h as i32, w as i32, ow as i32, oh as i32, k as i32, up as i32);
        unsafe {
            cu::memcpy_htod_async(wf, cv.w, c.stream).map_err(derr)?;
            c.launch(to_half, (nw.div_ceil(256) as u32, 1), 256, 0, kargs!(wf, wh, nw))?;
            cu::memcpy_htod_async(bb, cv.b, c.stream).map_err(derr)?;
            let beta = if acc { 1.0 } else { 0.0 };
            let mut p0 = 0usize;
            while p0 < p {
                let np = chunk.min(p - p0);
                let (p0u, total) = (p0 as u64, (np * ci) as u64);
                c.launch(lx.im2col2d, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(src, cols, cii, hi, wi, owi, ohi, ki, upi, p0u, total))?;
                c.gemm(true, false, np, co, kk, 1.0, cols, kk, wh, kk, beta, y + (p0 * 4) as u64, p, Gemm::Half)?;
                p0 += np;
            }
            let (pu, n) = (p as u64, (co * p) as u64);
            c.launch(lx.bias, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(y, bb, nul, pu, n))
        }
    };
    // GroupNorm of `src` [ch][hw] into `dst` (may be `src`), SiLU under `silu`
    let gn = |st: &mut State, nr: &VaeNormRef<'_>, src: u64, dst: u64, ch: usize, hw: usize, silu: bool| -> Res<()> {
        let g = nr.groups;
        if g == 0 || !ch.is_multiple_of(g) || nr.w.len() != ch || nr.b.len() != ch {
            return Err("a GroupNorm of an unexpected shape".into());
        }
        let span = ch / g * hw;
        let nchunk = span.div_ceil(GN_CHUNK);
        let part = st.h3.buf(c, "fv part", g * nchunk * 16)?;
        let stats = st.h3.buf(c, "fv stats", g * 8 + 64)?;
        let nw = st.h3.buf(c, "fv nw", 2 * ch * 4)?;
        let (spanu, chunku, nci) = (span as u64, GN_CHUNK as u64, nchunk as i32);
        let (count, peri, silui, hwu, total) = (span as f64, (ch / g) as i32, silu as i32, hw as u64, (ch * hw) as u64);
        unsafe {
            cu::memcpy_htod_async(nw, nr.w, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(nw + (ch * 4) as u64, nr.b, c.stream).map_err(derr)?;
            c.launch(f.part, (nchunk as u32, g as u32), 256, 0, kargs!(src, part, spanu, chunku, nci))?;
            c.launch(f.fin, (g as u32, 1), 32, 0, kargs!(part, stats, nci, count, GN_EPS))?;
            let (wo, bo) = (nw, nw + (ch * 4) as u64);
            c.launch(f.apply, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(src, dst, stats, wo, bo, hwu, peri, silui, total))
        }
    };
    // a resnet over `x` [ic][hw]: the result lands in `x` (or, through a
    // shortcut, in `s`, which then swaps with `x`)
    let resnet = |st: &mut State, r: &VaeResnetRef<'_>, x: &mut u64, s: &mut u64, t: u64, h: usize, w: usize| -> Res<()> {
        let hw = h * w;
        gn(st, &r.norm1, *x, t, r.conv1.ic, hw, true)?;
        conv(st, &r.conv1, t, h, w, false, u, false)?;
        gn(st, &r.norm2, u, u, r.conv1.oc, hw, true)?;
        match &r.shortcut {
            Some(sc) => {
                conv(st, sc, *x, h, w, false, *s, false)?;
                conv(st, &r.conv2, u, h, w, false, *s, true)?;
                std::mem::swap(x, s);
            }
            None => conv(st, &r.conv2, u, h, w, false, *x, true)?,
        }
        Ok(())
    };

    // ── the pass ──
    let mut sx = s;
    conv(st, &a.conv_in, x, h, w, false, t, false)?;
    std::mem::swap(&mut x, &mut t);
    resnet(st, &a.mid_res1, &mut x, &mut sx, t, h, w)?;
    // the mid-block's attention: one head as wide as the channels
    {
        let at = &a.mid_attn;
        let (ch, n) = (at.c, h * w);
        gn(st, &at.norm, x, t, ch, n, false)?;
        let wq = st.h3.buf(c, "fv aw", 4 * ch * ch * 4)?;
        let ab = st.h3.buf(c, "fv ab", 4 * ch * 4)?;
        let qkv = st.h3.buf(c, "fv qkv", 4 * n * ch * 4)?;
        let cq = (PANEL / (n * 4)).clamp(1, n);
        let sb = st.h3.buf(c, "fv sc", cq * n * 4)?;
        let (q, k, v, o) = (qkv, qkv + (n * ch * 4) as u64, qkv + (2 * n * ch * 4) as u64, qkv + (3 * n * ch * 4) as u64);
        let mats = [at.q, at.k, at.v, at.out];
        if mats.iter().any(|(m, b)| m.len() != ch * ch || b.len() != ch) {
            return Err("a mid-block attention of an unexpected shape".into());
        }
        let wm = |i: usize| wq + (i * ch * ch * 4) as u64;
        let bm = |i: usize| ab + (i * ch * 4) as u64;
        let (ci, ni, total) = (ch as i32, n as i32, (n * ch) as u64);
        unsafe {
            for (i, (m, b)) in mats.iter().enumerate() {
                cu::memcpy_htod_async(wm(i), m, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(bm(i), b, c.stream).map_err(derr)?;
            }
            // q, k, v token-major [n][ch] from the channel-major normed panel
            for (i, dst) in [q, k, v].into_iter().enumerate() {
                c.gemm(true, true, ch, n, ch, 1.0, wm(i), ch, t, n, 0.0, dst, ch, Gemm::Tf32)?;
                c.launch(f.bias_rows, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(dst, bm(i), ci, total))?;
            }
            let scale = 1.0 / (ch as f32).sqrt();
            let mut q0 = 0usize;
            while q0 < n {
                let rows = cq.min(n - q0);
                c.gemm(true, false, n, rows, ch, scale, k, ch, q + (q0 * ch * 4) as u64, ch, 0.0, sb, n, Gemm::Tf32)?;
                c.launch(lx.softmax_f32, (rows as u32, 1), 256, 0, kargs!(sb, ni))?;
                c.gemm(false, false, ch, rows, n, 1.0, v, ch, sb, n, 0.0, o + (q0 * ch * 4) as u64, ch, Gemm::Tf32)?;
                q0 += rows;
            }
            // the output product straight onto the channel-major residual
            c.gemm(true, false, n, ch, ch, 1.0, o, ch, wm(3), ch, 1.0, x, n, Gemm::Tf32)?;
            let nu = n as u64;
            c.launch(lx.bias, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(x, bm(3), nul, nu, total))?;
        }
    }
    resnet(st, &a.mid_res2, &mut x, &mut sx, t, h, w)?;
    let (mut hh, mut ww) = (h, w);
    for up in &a.ups {
        for r in &up.resnets {
            resnet(st, r, &mut x, &mut sx, t, hh, ww)?;
        }
        if let Some(cv) = &up.upsample {
            conv(st, cv, x, hh, ww, true, t, false)?;
            std::mem::swap(&mut x, &mut t);
            hh *= 2;
            ww *= 2;
        }
    }
    gn(st, &a.norm_out, x, t, a.norm_out.w.len(), hh * ww, true)?;
    conv(st, &a.conv_out, t, hh, ww, false, u, false)?;
    unsafe { cu::memcpy_dtoh_async(out, u, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the picture came back with a value that is not finite".into());
    }
    Ok(())
}
