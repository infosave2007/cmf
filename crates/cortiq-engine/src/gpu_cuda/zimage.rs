//! Z-Image (Turbo and the base model) on CUDA: the `gpu::zimage_prepare` /
//! `zimage_step` contract the Metal and Vulkan modules serve, one DiT
//! forward a step on the card. The rows go up once (the patch tokens) and
//! come back once (the image rows of the velocity): the embed with its pad
//! rows, the two noise refiners over the image rows, the refined caption
//! after them, the thirty layers over [image | caption], the final
//! LayerNorm times the step's scale and its linear. A block is the Lumina
//! Next-DiT block's arithmetic (`lumina.rs::block_dev`: the sandwich RMS
//! norms with the step's scale, q/k/v products, per-head q/k norm with the
//! interleaved rotation, `h3_flash`, `tanh`-gated post-normed residuals, the
//! SwiGLU), the weights in the H3 store. The caption's context refiner and
//! a CFG pair's batch of two stay with the caller (host refiner, the items
//! stepped one by one). `CMF_ZIMAGE_CUDA=0` keeps the host path.

use super::*;
use crate::gpu::{DitBlockArgs, ZBlockRef, ZGeom, ZPrepareArgs, ZStepArgs};

const ZI_SRC: &str = include_str!("zimage.cu");
/// Programs kept at once (a CFG pair is prepared as two).
const MAX_PROGS: usize = 2;

#[derive(Clone, Copy)]
pub(super) struct ZiFuncs {
    bias_pad: cs::CUfunction,
    lnscale: cs::CUfunction,
    bias: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<ZiFuncs> {
    if let Some(f) = st.h3.zi_funcs {
        return Ok(f);
    }
    let module = compile(ZI_SRC, Vec::new(), c.arch)?;
    let f = ZiFuncs {
        bias_pad: func(module, "zi_bias_pad")?,
        lnscale: func(module, "zi_lnscale")?,
        bias: func(module, "zi_bias")?,
    };
    st.h3.zi_funcs = Some(f);
    Ok(f)
}

/// A block's weights and norms (host copies: the caller's slices end with
/// the prepare call).
pub(super) struct ZiBlock {
    w: [usize; 7],
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    ffn_norm1: Vec<f32>,
    ffn_norm2: Vec<f32>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
}

impl ZiBlock {
    fn of(b: &ZBlockRef<'_>) -> ZiBlock {
        ZiBlock {
            w: [b.wq, b.wk, b.wv, b.wo, b.w1, b.w3, b.w2],
            norm1: b.norm1.to_vec(),
            norm2: b.norm2.to_vec(),
            ffn_norm1: b.ffn_norm1.to_vec(),
            ffn_norm2: b.ffn_norm2.to_vec(),
            norm_q: b.norm_q.to_vec(),
            norm_k: b.norm_k.to_vec(),
        }
    }
}

/// One (prompt, resolution)'s program.
pub(super) struct ZiProg {
    key: u64,
    model: Arc<CmfModel>,
    geom: ZGeom,
    n_img: usize,
    n_img_p: usize,
    n_cap_p: usize,
    rope_img: (Vec<f32>, Vec<f32>),
    rope_joint: (Vec<f32>, Vec<f32>),
    /// the refined caption `[n_cap_p][hidden]`, then x_emb_w `[hidden][64]`,
    /// x_emb_b, x_pad, final_w `[64][hidden]`, final_b
    consts: Buf,
    o_emb_w: usize,
    o_emb_b: usize,
    o_pad: usize,
    o_fin_w: usize,
    o_fin_b: usize,
    refiner: Vec<ZiBlock>,
    layers: Vec<ZiBlock>,
}

pub fn zimage_prepare(a: &ZPrepareArgs) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || a.neg.is_some() || std::env::var("CMF_ZIMAGE_CUDA").as_deref() == Ok("0") {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    st.h3.zi.retain(|p| p.key != a.key);
    match prepare(c, &mut st, a) {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("z-image prepare", &e);
            false
        }
    }
}

pub fn zimage_step(a: &mut ZStepArgs) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || a.out_neg.is_some() {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    let Some(i) = st.h3.zi.iter().position(|p| p.key == a.key) else { return false };
    let prog = st.h3.zi.remove(i);
    let r = unsafe { step(c, &mut st, &prog, a) };
    st.h3.zi.push(prog);
    match r {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("z-image step", &e);
            false
        }
    }
}

/// Drop every program (the DiT stage is over).
pub fn zimage_release() {
    if let Some(c) = ctx()
        && let Ok(mut st) = c.st.lock()
    {
        let _ = c.sync();
        st.h3.zi.clear();
    }
}

fn prepare(c: &Ctx, st: &mut State, a: &ZPrepareArgs) -> Res<()> {
    let g = a.geom;
    let (h, hd) = (g.hidden, g.hd);
    let pairs = hd / 2;
    let s = a.n_img_p + a.n_cap_p;
    if g.nh * hd != h || !hd.is_multiple_of(32) || g.patch_dim != 64 || a.n_img == 0 || a.n_img > a.n_img_p {
        return Err(format!("geometry {g:?} outside the chain"));
    }
    if a.cap.len() != a.n_cap_p * h || a.rope_img.0.len() != a.n_img_p * pairs || a.rope_joint.0.len() != s * pairs {
        return Err("a caption or rotation table of the wrong size".into());
    }
    if a.x_emb_w.len() != h * 64 || a.x_emb_b.len() != h || a.x_pad.len() != h || a.final_w.len() != 64 * h || a.final_b.len() != 64 {
        return Err("an embedder or final layer of the wrong size".into());
    }
    st.h3.flash(c, hd, false).ok_or("no flash attention kernel for this head width")?;
    while st.h3.zi.len() >= MAX_PROGS {
        st.h3.zi.remove(0);
    }
    let mut cv: Vec<f32> = Vec::with_capacity(a.cap.len() + 130 * h + 64);
    cv.extend_from_slice(a.cap);
    let mut put = |v: &[f32]| {
        let at = cv.len();
        cv.extend_from_slice(v);
        at
    };
    let (o_emb_w, o_emb_b, o_pad, o_fin_w, o_fin_b) = (put(a.x_emb_w), put(a.x_emb_b), put(a.x_pad), put(a.final_w), put(a.final_b));
    let consts = Buf::direct(cv.len() * 4)?;
    unsafe { cu::memcpy_htod_async(consts.ptr, &cv, c.stream).map_err(derr)? };
    c.sync()?;
    st.h3.zi.push(ZiProg {
        key: a.key,
        model: a.model.clone(),
        geom: g,
        n_img: a.n_img,
        n_img_p: a.n_img_p,
        n_cap_p: a.n_cap_p,
        rope_img: (a.rope_img.0.to_vec(), a.rope_img.1.to_vec()),
        rope_joint: (a.rope_joint.0.to_vec(), a.rope_joint.1.to_vec()),
        consts,
        o_emb_w,
        o_emb_b,
        o_pad,
        o_fin_w,
        o_fin_b,
        refiner: a.noise_refiner.iter().map(ZiBlock::of).collect(),
        layers: a.layers.iter().map(ZiBlock::of).collect(),
    });
    Ok(())
}

unsafe fn step(c: &Ctx, st: &mut State, p: &ZiProg, a: &mut ZStepArgs) -> Res<()> {
    let g = p.geom;
    let h = g.hidden;
    let nb = p.refiner.len() + p.layers.len();
    let s = p.n_img_p + p.n_cap_p;
    if a.x_tok.len() != p.n_img_p * 64 || a.mods.len() != nb * 4 * h || a.final_scale.len() != h || a.out.len() != p.n_img * 64 {
        return Err("tokens, modulation or output of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let x = st.h3.buf(c, "zi x", s * h * 4)?;
    let tok = st.h3.buf(c, "zi tok", p.n_img_p * 64 * 4)?;
    let xf = st.h3.buf(c, "zi xf", p.n_img * h * 4)?;
    let fsb = st.h3.buf(c, "zi fs", h * 4)?;
    let ob = st.h3.buf(c, "zi out", p.n_img * 64 * 4)?;
    let at = |o: usize| p.consts.ptr + (o * 4) as u64;
    let hi = h as i32;
    unsafe {
        cu::memcpy_htod_async(tok, a.x_tok, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(fsb, a.final_scale, c.stream).map_err(derr)?;
        // embed the image rows, the pad rows := x_pad, the caption after them
        c.gemm(true, false, h, p.n_img_p, 64, 1.0, at(p.o_emb_w), 64, tok, 64, 0.0, x, h, Gemm::F32)?;
        let tot = (p.n_img_p * h) as u64;
        let ni = p.n_img as i32;
        c.launch(f.bias_pad, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, at(p.o_emb_b), at(p.o_pad), hi, ni, tot))?;
        cu::memcpy_dtod_async(x + (p.n_img_p * h * 4) as u64, p.consts.ptr, p.n_cap_p * h * 4, c.stream).map_err(derr)?;
    }
    let tanh = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| x.tanh()).collect() };
    let run = |st: &mut State, b: &ZiBlock, m: &[f32], n: usize, rope: &(Vec<f32>, Vec<f32>)| -> Res<()> {
        let (gm, gf) = (tanh(&m[h..2 * h]), tanh(&m[3 * h..4 * h]));
        let args = DitBlockArgs {
            n,
            hidden: h,
            inter: g.inter,
            nh: g.nh,
            nkv: g.nh,
            hd: g.hd,
            eps: g.eps,
            rope_cos: &rope.0[..n * g.hd / 2],
            rope_sin: &rope.1[..n * g.hd / 2],
            norm1: &b.norm1,
            norm2: &b.norm2,
            ffn_norm1: &b.ffn_norm1,
            ffn_norm2: &b.ffn_norm2,
            norm_q: &b.norm_q,
            norm_k: &b.norm_k,
            s_msa: &m[..h],
            gate_msa: &gm,
            s_mlp: &m[2 * h..3 * h],
            gate_mlp: &gf,
            wq: b.w[0],
            wk: b.w[1],
            wv: b.w[2],
            wo: b.w[3],
            w1: b.w[4],
            w3: b.w[5],
            w2: b.w[6],
            q4tp: true,
            resident_in: false,
            resident_out: false,
        };
        unsafe { super::lumina::block_dev(c, st, &p.model, &args, &[n], x) }
    };
    for (i, b) in p.refiner.iter().enumerate() {
        run(st, b, &a.mods[i * 4 * h..(i + 1) * 4 * h], p.n_img_p, &p.rope_img)?;
    }
    let nr = p.refiner.len();
    for (i, b) in p.layers.iter().enumerate() {
        run(st, b, &a.mods[(nr + i) * 4 * h..(nr + i + 1) * 4 * h], s, &p.rope_joint)?;
    }
    unsafe {
        c.launch(f.lnscale, (p.n_img as u32, 1), 256, 0, kargs!(x, xf, fsb, hi, g.final_eps))?;
        c.gemm(true, false, 64, p.n_img, h, 1.0, at(p.o_fin_w), h, xf, h, 0.0, ob, 64, Gemm::F32)?;
        let tot = (p.n_img * 64) as u64;
        c.launch(f.bias, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(ob, at(p.o_fin_b), 64i32, tot))?;
        cu::memcpy_dtoh_async(&mut *a.out, ob, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if a.out.iter().any(|v| !v.is_finite()) {
        return Err("the step produced a value that is not finite".into());
    }
    Ok(())
}
