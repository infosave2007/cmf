//! The Primal VAE on CUDA: a `primal_vae::PvPlan` (encode or decode) run
//! whole on the card — one upload, the steps, one readback — as the Metal
//! and Vulkan modules run it. A 3×3 convolution is an im2col with its
//! dilation into f16 and an f16 cuBLAS product, the residual landing as the
//! product's accumulator; the RMS norm with SiLU and pixel (un)shuffle are
//! kernels of their own (`primal.cu`). The weights go up as f16 once per
//! model (keyed by the caller's key). `CMF_PRIMAL_VAE_CUDA=0` keeps the
//! host steps.

use super::*;
use crate::primal_vae::{PvPlan, PvStep, PvWeights};

const PV_SRC: &str = include_str!("primal.cu");
const EPS: f32 = 1e-6;
/// The bytes of one im2col panel.
const PANEL: usize = 128 << 20;

#[derive(Clone, Copy)]
pub(super) struct PvFuncs {
    im2col: cs::CUfunction,
    norm_silu: cs::CUfunction,
    unshuffle: cs::CUfunction,
    shuffle: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<PvFuncs> {
    if let Some(f) = st.h3.pv_funcs {
        return Ok(f);
    }
    let module = compile(PV_SRC, Vec::new(), c.arch)?;
    let f = PvFuncs {
        im2col: func(module, "pv_im2col")?,
        norm_silu: func(module, "pv_norm_silu")?,
        unshuffle: func(module, "pv_unshuffle")?,
        shuffle: func(module, "pv_shuffle")?,
    };
    st.h3.pv_funcs = Some(f);
    Ok(f)
}

/// A model's weights on the card: per conv the f16 weights `[cout][cin·9]`
/// and the f32 bias, then every norm's gamma.
pub(super) struct PvDev {
    key: u64,
    convs: Vec<(Buf, Buf)>,
    gammas: Vec<Buf>,
}

pub fn primal_vae_run(key: u64, wts: &PvWeights, plan: &PvPlan, input: &[f32], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || std::env::var("CMF_PRIMAL_VAE_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { run(c, &mut st, key, wts, plan, input, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("primal vae", &e);
            false
        }
    }
}

unsafe fn upload(c: &Ctx, st: &mut State, key: u64, wts: &PvWeights) -> Res<PvDev> {
    let to_half = st.h3.funcs(c)?.to_half;
    let most = wts.convs.iter().map(|v| v.w.len()).max().unwrap_or(1);
    let stage = st.h3.buf(c, "pv stage", most * 4)?;
    let mut convs = Vec::with_capacity(wts.convs.len());
    for cv in &wts.convs {
        if cv.w.len() != cv.cout * cv.cin * 9 || cv.b.len() != cv.cout {
            return Err("a convolution of an unexpected shape".into());
        }
        let w = Buf::direct(cv.w.len() * 2)?;
        let b = Buf::direct(cv.b.len() * 4)?;
        let n = cv.w.len() as u64;
        unsafe {
            cu::memcpy_htod_async(stage, &cv.w, c.stream).map_err(derr)?;
            c.launch(to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(stage, w.ptr, n))?;
            cu::memcpy_htod_async(b.ptr, &cv.b, c.stream).map_err(derr)?;
        }
        convs.push((w, b));
    }
    let mut gammas = Vec::with_capacity(wts.gammas.len());
    for g in &wts.gammas {
        let b = Buf::direct(g.len() * 4)?;
        unsafe { cu::memcpy_htod_async(b.ptr, g, c.stream).map_err(derr)? };
        gammas.push(b);
    }
    c.sync()?;
    Ok(PvDev { key, convs, gammas })
}

unsafe fn run(c: &Ctx, st: &mut State, key: u64, wts: &PvWeights, plan: &PvPlan, input: &[f32], out: &mut [f32]) -> Res<()> {
    if out.len() != plan.out_elems || input.len() > plan.buf_elems[plan.input as usize] {
        return Err("an input or output of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let lx = super::ltx::funcs(st, c)?;
    if !st.h3.pv.as_ref().is_some_and(|d| d.key == key) {
        st.h3.pv = None;
        let d = unsafe { upload(c, st, key, wts)? };
        st.h3.pv = Some(d);
    }
    let most = plan.buf_elems.iter().copied().max().unwrap_or(1);
    let bufs: Vec<Buf> = plan.buf_elems.iter().map(|&n| Buf::direct(n.max(1) * 4)).collect::<Res<_>>()?;
    let tmp = Buf::direct(most * 4)?;
    let dev = st.h3.pv.take().unwrap();
    let r = (|| -> Res<()> {
        unsafe { cu::memcpy_htod_async(bufs[plan.input as usize].ptr, input, c.stream).map_err(derr)? };
        let at = |i: u8| bufs[i as usize].ptr;
        for s in &plan.steps {
            match *s {
                PvStep::Conv { wi, src, dst, res, dil, h, w } => {
                    let cv = &wts.convs[wi];
                    let (cin, cout, hw) = (cv.cin, cv.cout, h * w);
                    let kk = cin * 9;
                    // the product's target: a scratch buffer when it would
                    // overwrite what its own gather still reads
                    let o = if src == dst { tmp.ptr } else { at(dst) };
                    let beta = match res {
                        Some(r) => {
                            if at(r) != o {
                                unsafe { cu::memcpy_dtod_async(o, at(r), cout * hw * 4, c.stream).map_err(derr)? };
                            }
                            1.0
                        }
                        None => 0.0,
                    };
                    let chunk = (PANEL / (kk * 2)).clamp(1, hw);
                    let cols = st.h3.buf(c, "pv cols", chunk * kk * 2)?;
                    let (ci, hi, wi_, di) = (cin as i32, h as i32, w as i32, dil as i32);
                    let mut p0 = 0usize;
                    while p0 < hw {
                        let np = chunk.min(hw - p0);
                        let (p0u, total) = (p0 as u64, (np * cin) as u64);
                        unsafe {
                            c.launch(f.im2col, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(at(src), cols, ci, hi, wi_, di, p0u, total))?;
                            c.gemm(true, false, np, cout, kk, 1.0, cols, kk, dev.convs[wi].0.ptr, kk, beta, o + (p0 * 4) as u64, hw, Gemm::Half)?;
                        }
                        p0 += np;
                    }
                    let (hwu, n, nul) = (hw as u64, (cout * hw) as u64, 0u64);
                    unsafe {
                        c.launch(lx.bias, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(o, dev.convs[wi].1.ptr, nul, hwu, n))?;
                        if o != at(dst) {
                            cu::memcpy_dtod_async(at(dst), o, cout * hw * 4, c.stream).map_err(derr)?;
                        }
                    }
                }
                PvStep::NormSilu { gi, src, dst, c: ch, h, w } => {
                    let hw = (h * w) as u64;
                    unsafe { c.launch(f.norm_silu, (hw.div_ceil(256) as u32, 1), 256, 0, kargs!(at(src), at(dst), dev.gammas[gi].ptr, ch as i32, hw, EPS))? };
                }
                PvStep::Unshuffle { src, dst, c: ch, h, w } => {
                    let total = (ch * h * w) as u64;
                    unsafe { c.launch(f.unshuffle, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(at(src), at(dst), ch as i32, h as i32, w as i32, total))? };
                }
                PvStep::Shuffle { src, dst, add, c: ch, h, w } => {
                    let total = (ch * h * w) as u64;
                    let a = add.map_or(0u64, at);
                    unsafe { c.launch(f.shuffle, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(at(src), at(dst), a, ch as i32, h as i32, w as i32, total))? };
                }
            }
        }
        unsafe { cu::memcpy_dtoh_async(out, at(plan.output), c.stream).map_err(derr)? };
        c.sync()
    })();
    st.h3.pv = Some(dev);
    r?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the forward produced a value that is not finite".into());
    }
    Ok(())
}
