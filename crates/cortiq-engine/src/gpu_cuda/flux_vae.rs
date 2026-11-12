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
    rms_ch: cs::CUfunction,
    dupup: cs::CUfunction,
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
        rms_ch: func(module, "fv_rms_ch")?,
        dupup: func(module, "fv_dupup_add")?,
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

/// `y` [co][oh·ow] (= or +=, `acc`) of a k×k same-padding convolution
/// `(weights [co][ci][k][k], bias [co], ci, co, k)` over `src` [ci][h][w],
/// nearest ×2 first under `up`; then the bias. Im2col a panel of positions
/// at a time into f16, an f16 product against the weights in f16.
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d(
    c: &Ctx,
    st: &mut State,
    lx: &super::ltx::LxFuncs,
    to_half: cs::CUfunction,
    cv: (&[f32], &[f32], usize, usize, usize),
    src: u64,
    h: usize,
    w: usize,
    up: bool,
    y: u64,
    acc: bool,
) -> Res<()> {
    let (wt, bias, ci, co, k) = cv;
    let (oh, ow) = if up { (2 * h, 2 * w) } else { (h, w) };
    let (p, kk) = (oh * ow, ci * k * k);
    if k.is_multiple_of(2) || wt.len() != co * kk || bias.len() != co {
        return Err("a convolution of an unexpected shape".into());
    }
    let wf = st.h3.buf(c, "fv wf", co * kk * 4)?;
    let wh = st.h3.buf(c, "fv wh", co * kk * 2 + 64)?;
    let bb = st.h3.buf(c, "fv b", co * 4 + 64)?;
    let chunk = (PANEL / (kk * 2)).clamp(1, p);
    let cols = st.h3.buf(c, "fv cols", chunk * kk * 2)?;
    let nw = (co * kk) as u64;
    let (cii, hi, wi, owi, ohi, ki, upi) = (ci as i32, h as i32, w as i32, ow as i32, oh as i32, k as i32, up as i32);
    let nul = 0u64;
    unsafe {
        cu::memcpy_htod_async(wf, wt, c.stream).map_err(derr)?;
        c.launch(to_half, (nw.div_ceil(256) as u32, 1), 256, 0, kargs!(wf, wh, nw))?;
        cu::memcpy_htod_async(bb, bias, c.stream).map_err(derr)?;
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
    let conv = |st: &mut State, cv: &VaeConvRef<'_>, src: u64, h: usize, w: usize, up: bool, y: u64, acc: bool| -> Res<()> {
        unsafe { conv2d(c, st, &lx, to_half, (cv.w, cv.b, cv.ic, cv.oc, cv.k), src, h, w, up, y, acc) }
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

// ───────────────────────── Qwen-Image-2.1 ─────────────────────────

/// The Qwen-Image-2.1 (Wan-class) VAE decoder as one resident chain:
/// RMS norm over channels with SiLU, the convolutions as above, the mid
/// block's single head over every position (TF32 products around an f32
/// softmax), each up block's DupUp shortcut added onto its upsampled
/// output. `z` raw latent planes `[z_dim][h][w]` into `out`
/// `[out_channels][H][W]` before the clamp. `false`: nothing was written.
/// `CMF_QI21_VAE_CUDA=0` keeps the per-convolution path.
pub fn qi21_vae_decode(a: &crate::gpu::Qi21VaeDecodeArgs<'_>, z: &[f32], h: usize, w: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_QI21_VAE_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { run_qi21(c, &mut st, a, z, h, w, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("qwen-image-2.1 vae chain", &e);
            false
        }
    }
}

unsafe fn run_qi21(c: &Ctx, st: &mut State, a: &crate::gpu::Qi21VaeDecodeArgs<'_>, z: &[f32], h: usize, w: usize, out: &mut [f32]) -> Res<()> {
    use crate::gpu::{Qi21VaeConvRef, Qi21VaeResRef};
    // the largest tensor, and the picture's size
    let (mut hh, mut ww) = (h, w);
    let mut most = a.post_quant.ci.max(a.post_quant.co).max(a.conv_in.co) * h * w;
    let res_most = |r: &Qi21VaeResRef<'_>, hw: usize| r.c1.ci.max(r.c1.co).max(r.c2.co) * hw;
    for r in &a.mid_res {
        most = most.max(res_most(r, h * w));
    }
    most = most.max(3 * a.attn_qkv.ci * h * w);
    for up in &a.ups {
        for r in &up.resnets {
            most = most.max(res_most(r, hh * ww));
        }
        if let Some((u, _)) = &up.up {
            hh *= 2;
            ww *= 2;
            most = most.max(u.ci.max(u.co).max(up.out_dim) * hh * ww);
        }
    }
    most = most.max(a.conv_out.ci.max(a.conv_out.co) * hh * ww);
    if z.len() != a.post_quant.ci * h * w || out.len() != a.conv_out.co * hh * ww {
        return Err("a latent or picture of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let lx = super::ltx::funcs(st, c)?;
    let to_half = st.h3.funcs(c)?.to_half;
    let b = [
        st.h3.buf(c, "fv x", most * 4)?,
        st.h3.buf(c, "fv t", most * 4)?,
        st.h3.buf(c, "fv u", most * 4)?,
        st.h3.buf(c, "fv s", most * 4)?,
        st.h3.buf(c, "fv copy", most * 4)?,
    ];
    let (mut x, mut t, u, mut s, cp) = (b[0], b[1], b[2], b[3], b[4]);
    let gam = st.h3.buf(c, "fv gamma", 4096 * 4)?;
    unsafe { cu::memcpy_htod_async(t, z, c.stream).map_err(derr)? };

    let conv = |st: &mut State, cv: &Qi21VaeConvRef<'_>, src: u64, h: usize, w: usize, up: bool, y: u64, acc: bool| -> Res<()> {
        unsafe { conv2d(c, st, &lx, to_half, (cv.w, cv.b, cv.ci, cv.co, cv.k), src, h, w, up, y, acc) }
    };
    // the RMS norm over channels of `src` [ch][hw] into `dst`, SiLU under `silu`
    let rms = |gamma: &[f32], src: u64, dst: u64, ch: usize, hw: usize, silu: bool| -> Res<()> {
        if gamma.len() != ch || ch > 4096 {
            return Err("an RMS norm of an unexpected width".into());
        }
        let (chi, hwu, si) = (ch as i32, hw as u64, silu as i32);
        unsafe {
            cu::memcpy_htod_async(gam, gamma, c.stream).map_err(derr)?;
            c.launch(f.rms_ch, (hwu.div_ceil(256) as u32, 1), 256, 0, kargs!(src, dst, gam, chi, hwu, si))
        }
    };
    // a resnet over `x` [ci][hw]: the result lands in `x` (or, through a
    // shortcut, in `s`, which then swaps with `x`)
    let resnet = |st: &mut State, r: &Qi21VaeResRef<'_>, x: &mut u64, s: &mut u64, t: u64, h: usize, w: usize| -> Res<()> {
        let hw = h * w;
        rms(r.g1, *x, t, r.c1.ci, hw, true)?;
        conv(st, &r.c1, t, h, w, false, u, false)?;
        rms(r.g2, u, u, r.c1.co, hw, true)?;
        match &r.shortcut {
            Some(sc) => {
                conv(st, sc, *x, h, w, false, *s, false)?;
                conv(st, &r.c2, u, h, w, false, *s, true)?;
                std::mem::swap(x, s);
            }
            None => conv(st, &r.c2, u, h, w, false, *x, true)?,
        }
        Ok(())
    };

    conv(st, &a.post_quant, t, h, w, false, x, false)?;
    conv(st, &a.conv_in, x, h, w, false, t, false)?;
    std::mem::swap(&mut x, &mut t);
    resnet(st, &a.mid_res[0], &mut x, &mut s, t, h, w)?;
    // the mid attention: one head as wide as the channels
    {
        let (ch, n) = (a.attn_qkv.ci, h * w);
        if a.attn_qkv.co != 3 * ch {
            return Err("a mid attention of an unexpected shape".into());
        }
        rms(a.attn_gamma, x, u, ch, n, false)?;
        conv(st, &a.attn_qkv, u, h, w, false, t, false)?;
        let (q, k, v) = (t, t + (ch * n * 4) as u64, t + (2 * ch * n * 4) as u64);
        let cq = (PANEL / (n * 4)).clamp(1, n);
        let sb = st.h3.buf(c, "fv sc", cq * n * 4)?;
        let scale = 1.0 / (ch as f32).sqrt();
        let ni = n as i32;
        let mut q0 = 0usize;
        while q0 < n {
            let rows = cq.min(n - q0);
            unsafe {
                // S [rows][n] = scale · Qᵀ K over the channel-major planes
                c.gemm(false, true, n, rows, ch, scale, k, n, q + (q0 * 4) as u64, n, 0.0, sb, n, Gemm::Tf32)?;
                c.launch(lx.softmax_f32, (rows as u32, 1), 256, 0, kargs!(sb, ni))?;
                // O [ch][q0 + i] = Σ_j P[i][j] V[ch][j], channel-major
                c.gemm(true, false, rows, ch, n, 1.0, sb, n, v, n, 0.0, u + (q0 * 4) as u64, n, Gemm::Tf32)?;
            }
            q0 += rows;
        }
        conv(st, &a.attn_proj, u, h, w, false, x, true)?;
    }
    resnet(st, &a.mid_res[1], &mut x, &mut s, t, h, w)?;
    let (mut hh, mut ww) = (h, w);
    for up in &a.ups {
        if up.up.is_some() {
            unsafe { cu::memcpy_dtod_async(cp, x, up.in_dim * hh * ww * 4, c.stream).map_err(derr)? };
        }
        for r in &up.resnets {
            resnet(st, r, &mut x, &mut s, t, hh, ww)?;
        }
        if let Some((cv, ft)) = &up.up {
            conv(st, cv, x, hh, ww, true, t, false)?;
            if cv.co != up.out_dim || up.in_dim == 0 || !(up.out_dim * ft * 4).is_multiple_of(up.in_dim) {
                return Err("an up block's shortcut of an unexpected shape".into());
            }
            let total = (up.out_dim * 4 * hh * ww) as u64;
            let (ci, co, fti, hi, wi) = (up.in_dim as i32, up.out_dim as i32, *ft as i32, hh as i32, ww as i32);
            unsafe { c.launch(f.dupup, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(t, cp, ci, co, fti, hi, wi, total))? };
            std::mem::swap(&mut x, &mut t);
            hh *= 2;
            ww *= 2;
        }
    }
    rms(a.norm_out, x, t, a.conv_out.ci, hh * ww, true)?;
    conv(st, &a.conv_out, t, hh, ww, false, u, false)?;
    unsafe { cu::memcpy_dtoh_async(out, u, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the picture came back with a value that is not finite".into());
    }
    Ok(())
}
