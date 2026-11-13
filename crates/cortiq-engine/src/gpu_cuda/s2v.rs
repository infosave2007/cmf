//! Kandinsky S2V on CUDA: the `gpu::s2v_prepare` stepper the Metal and Vulkan
//! modules serve, every DiT forward on the card. Per render the modulations
//! of every step are computed once (as the Vulkan program does); per step the
//! four prompt blocks run once and are kept, per speech chunk the four audio
//! blocks and the clean rows (memory and reference) once, then the visual
//! stream `[clean | noisy]` goes through the sixty blocks: the LayerNorm with
//! each row's modulation (clean rows at t = 0, the noisy tail at t), the q/k/v
//! products with the per-head RMS norm and the rotation, attention over every
//! row (`h3_flash`), the gated residual; the prompt cross-attention (TF32
//! products around a softmax, at most 512 keys); in the speech blocks the
//! affine LayerNorm of the noisy tail and the frame-aligned speech attention;
//! the erf-GELU feed-forward. The weights are the H3 store's: in their codec
//! on the card under `CMF_GPU_VRAM_MB`, expanded to an f16 plane for their
//! product. `CMF_S2V_CUDA=0` keeps the host path.

use super::*;
use crate::s2v::{self, Attn, Lin, S2vDit, S2vGeom};
use crate::s2vgen::S2vCond;

const S2V_SRC: &str = include_str!("s2v.cu");
const EPS: f32 = 1e-5;
/// The feed-forward's middle is stored times this before its second product.
const MID_INV: f32 = 1.0 / 16.0;
/// The bytes of one chunk of product rows or of attention scores.
const PANEL: usize = 256 << 20;

#[derive(Clone, Copy)]
pub(super) struct S2Funcs {
    ln: cs::CUfunction,
    heads: cs::CUfunction,
    addg: cs::CUfunction,
    gelu: cs::CUfunction,
    gattn: cs::CUfunction,
    addvec: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<S2Funcs> {
    if let Some(f) = st.h3.s2v_funcs {
        return Ok(f);
    }
    let module = compile(S2V_SRC, Vec::new(), c.arch)?;
    let f = S2Funcs {
        ln: func(module, "s2v_ln")?,
        heads: func(module, "s2v_heads")?,
        addg: func(module, "s2v_addg")?,
        gelu: func(module, "s2v_gelu")?,
        gattn: func(module, "s2v_gattn")?,
        addvec: func(module, "s2v_addvec")?,
    };
    st.h3.s2v_funcs = Some(f);
    Ok(f)
}

/// An attention's tensors and its constants (offsets in floats).
#[derive(Clone, Copy)]
struct AttnD {
    w: [usize; 4],
    b: [usize; 4],
    qn: usize,
    kn: usize,
}

#[derive(Clone, Copy)]
struct FfnD {
    w1: usize,
    w2: usize,
    b1: usize,
    b2: usize,
}

struct VBlk {
    sa: AttnD,
    ca: AttnD,
    /// the speech adapter: its LayerNorm's weight and bias, its attention
    v2a: Option<(usize, usize, AttnD)>,
    ffn: FfnD,
    keep: bool,
}

struct SBlk {
    /// unmodulated (audio): the two affine LayerNorms (weight, bias)
    n1: Option<(usize, usize)>,
    n2: Option<(usize, usize)>,
    attn: AttnD,
    ffn: FfnD,
    keep: bool,
}

pub struct S2vCuda {
    dit: *const S2vDit,
    model: Arc<CmfModel>,
    g: S2vGeom,
    h: usize,
    ff: usize,
    nh: usize,
    vb: Vec<VBlk>,
    tb: Vec<SBlk>,
    ab: Vec<SBlk>,
    consts: Buf,
    o_clean: usize,
    o_vcos: usize,
    o_vsin: usize,
    o_tcos: usize,
    o_tsin: usize,
    o_emb: usize,
    o_emb_b: usize,
    o_mask0: usize,
    o_out_b: usize,
    out_w16: Buf,
    vm: Vec<Vec<f32>>,
    tm: Vec<Vec<f32>>,
    fm: Vec<Vec<f32>>,
    text0: Vec<f32>,
    nt: usize,
    tcache: Vec<Option<Buf>>,
    audio_feats: Vec<Vec<f32>>,
    reference: Vec<f32>,
    /// the speech chunk the audio stream and the clean rows belong to
    chunk: Option<usize>,
    acond: Option<Buf>,
    agroups: Option<Buf>,
    na: usize,
    x0: Vec<f32>,
    cfg_in_visual: usize,
    cfg_out_visual: usize,
}

fn lin_idx(l: &Lin) -> Option<usize> {
    l.w.graph_w().map(|(_, g)| g.idx)
}

fn lin_bias(l: &Lin) -> Vec<f32> {
    l.b.clone().unwrap_or_else(|| vec![0.0; l.rows()])
}

/// The device program, or `None` (said once) to run on the host.
pub fn s2v_prepare(dit: &S2vDit, g: &S2vGeom, cond: &S2vCond, ts: &[f32], sf: [f32; 3]) -> Option<S2vCuda> {
    if std::env::var("CMF_S2V_CUDA").as_deref() == Ok("0") {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { prepare(c, &mut st, dit, g, cond, ts, sf) } {
        Ok(p) => Some(p),
        Err(e) => {
            let _ = c.sync();
            decline("kandinsky s2v prepare", &e);
            None
        }
    }
}

unsafe fn prepare(c: &Ctx, st: &mut State, dit: &S2vDit, g: &S2vGeom, cond: &S2vCond, ts: &[f32], sf: [f32; 3]) -> Res<S2vCuda> {
    let cfg = dit.cfg.clone();
    let (h, ff, nh) = (cfg.dim, cfg.ff, cfg.heads());
    let m = g.total();
    let nt = cond.text.len() / cfg.in_text;
    if nt == 0 || nt > 4096 || m == 0 || h != nh * s2v::HEAD_DIM || cond.audio.is_empty() {
        return Err(format!("{m} rows and {nt} prompt tokens outside the program"));
    }
    funcs(st, c)?;
    st.h3.flash(c, s2v::HEAD_DIM, false).ok_or("no flash attention kernel for this head width")?;

    // ── modulations of the whole render (the Vulkan program's) ──
    let pe = dit.pooled_embed(&cond.pooled);
    let te_all: Vec<Vec<f32>> = ts.iter().map(|&t| dit.time_embed(t, &pe)).collect();
    let te0 = dit.time_embed(0.0, &pe);
    let si = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| s2v::silu(x)).collect() };
    let n_t = ts.len();
    let mut rows_all = si(&te0);
    for te in &te_all {
        rows_all.extend(si(te));
    }
    let mut clean_mods = Vec::with_capacity(cfg.n_visual * 9 * h);
    let mut vm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_visual * 9 * h); n_t];
    let vnames: Vec<String> = (0..cfg.n_visual).map(|b| format!("dit.visual_transformer_blocks.{b}.visual_modulation.out_layer")).collect();
    for y in dit.modulate_all(&vnames, &rows_all, n_t + 1)? {
        clean_mods.extend_from_slice(&y[..9 * h]);
        for (s, v) in vm.iter_mut().enumerate() {
            v.extend_from_slice(&y[(s + 1) * 9 * h..(s + 2) * 9 * h]);
        }
    }
    let mut tm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_text * 6 * h); n_t];
    let t_rows = &rows_all[cfg.time_dim..];
    for b in 0..cfg.n_text {
        let y = dit.modulate(&format!("dit.text_transformer_blocks.{b}.text_modulation.out_layer"), t_rows, n_t)?;
        for (s, v) in tm.iter_mut().enumerate() {
            v.extend_from_slice(&y[s * 6 * h..(s + 1) * 6 * h]);
        }
    }
    let y = dit.modulate("dit.out_layer.modulation.out_layer", t_rows, n_t)?;
    let fm: Vec<Vec<f32>> = (0..n_t).map(|s| y[s * 2 * h..(s + 1) * 2 * h].to_vec()).collect();

    // ── constants and the tensors' indices ──
    let mut cv: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        let at = cv.len();
        cv.extend_from_slice(v);
        at
    };
    let attn = |a: &Attn, put: &mut dyn FnMut(&[f32]) -> usize| -> Res<AttnD> {
        let w = [lin_idx(&a.q), lin_idx(&a.k), lin_idx(&a.v), lin_idx(&a.o)];
        if w.iter().any(|i| i.is_none()) {
            return Err("an attention projection that is not a quantized tensor".into());
        }
        Ok(AttnD {
            w: w.map(|i| i.unwrap()),
            b: [put(&lin_bias(&a.q)), put(&lin_bias(&a.k)), put(&lin_bias(&a.v)), put(&lin_bias(&a.o))],
            qn: put(&a.qn),
            kn: put(&a.kn),
        })
    };
    let ffn = |f: &s2v::Ffn, put: &mut dyn FnMut(&[f32]) -> usize| -> Res<FfnD> {
        match (lin_idx(&f.w1), lin_idx(&f.w2)) {
            (Some(w1), Some(w2)) => Ok(FfnD { w1, w2, b1: put(&lin_bias(&f.w1)), b2: put(&lin_bias(&f.w2)) }),
            _ => Err("a feed-forward that is not a quantized tensor".into()),
        }
    };
    let o_clean = put(&clean_mods);
    let mut vb = Vec::with_capacity(dit.visual.len());
    for b in &dit.visual {
        let sa = attn(&b.sa, &mut put)?;
        let ca = attn(&b.ca, &mut put)?;
        let v2a = match &b.v2a {
            Some((n, a)) => {
                let (nw, nb) = (put(&n.w), put(&n.b));
                Some((nw, nb, attn(a, &mut put)?))
            }
            None => None,
        };
        let ffn = ffn(&b.ffn, &mut put)?;
        vb.push(VBlk { sa, ca, v2a, ffn, keep: false });
    }
    let mut tb = Vec::with_capacity(dit.text.len());
    for b in &dit.text {
        tb.push(SBlk { n1: None, n2: None, attn: attn(&b.attn, &mut put)?, ffn: ffn(&b.ffn, &mut put)?, keep: false });
    }
    let mut ab = Vec::with_capacity(dit.audio.len());
    for b in &dit.audio {
        let n1 = (put(&b.sa_norm.w), put(&b.sa_norm.b));
        let at = attn(&b.attn, &mut put)?;
        let n2 = (put(&b.ff_norm.w), put(&b.ff_norm.b));
        ab.push(SBlk { n1: Some(n1), n2: Some(n2), attn: at, ffn: ffn(&b.ffn, &mut put)?, keep: false });
    }
    let (vc, vs) = s2v::rope_visual(&cfg, g, sf);
    let (o_vcos, o_vsin) = (put(&vc), put(&vs));
    let (tc, tsn) = s2v::rope_1d(&(0..nt as i32).collect::<Vec<_>>());
    let (o_tcos, o_tsin) = (put(&tc), put(&tsn));
    let emb_w = match &dit.emb1.w {
        crate::dit::Proj::F32 { w, .. } => w.clone(),
        _ => return Err("an embedder that is not f32".into()),
    };
    let o_emb = put(&emb_w);
    let o_emb_b = put(&lin_bias(&dit.emb1));
    let o_mask0 = put(&dit.mask[..h]);
    let o_out_b = put(&lin_bias(&dit.out_w));
    let out_w = match &dit.out_w.w {
        crate::dit::Proj::F32 { w, .. } => w.clone(),
        _ => return Err("an output layer that is not f32".into()),
    };
    let consts = Buf::direct(cv.len() * 4)?;
    let out_w16 = Buf::direct(out_w.len() * 2)?;
    let to_half = st.h3.funcs(c)?.to_half;
    let stage = st.h3.buf(c, "s2 stage", out_w.len() * 4)?;
    unsafe {
        cu::memcpy_htod_async(consts.ptr, &cv, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(stage, &out_w, c.stream).map_err(derr)?;
        let n = out_w.len() as u64;
        c.launch(to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(stage, out_w16.ptr, n))?;
    }
    c.sync()?;

    // ── which blocks keep their weights on the card (prompt, speech, visual) ──
    let model = dit.model.clone();
    let uid = model.uid();
    let (free, total_mem) = cu::mem_get_info().map_err(derr)?;
    // the activations of a step at these rows
    let acts = (m * h * (4 + 2 + 3 * 4 + 3 * 2 + 2) + 2 * (PANEL + (ff.max(h) * h * 2))) as i64;
    let mut room: i64 = match ceiling() {
        Some(b) => b as i64 - (total_mem - free) as i64,
        None => free as i64,
    } - acts
        - (512 << 20);
    let mut fits = |ws: &[usize], st: &State| -> Res<bool> {
        let mut need = 0u64;
        for &ix in ws {
            if !st.h3.mats.contains_key(&(uid, ix)) {
                let (kind, r, co, _) = packed(&model, ix)?;
                need += packed_dev_bytes(kind, r, co);
            }
        }
        if room >= need as i64 {
            room -= need as i64;
            Ok(true)
        } else {
            Ok(false)
        }
    };
    for b in tb.iter_mut().chain(ab.iter_mut()) {
        let ws = [b.attn.w[0], b.attn.w[1], b.attn.w[2], b.attn.w[3], b.ffn.w1, b.ffn.w2];
        b.keep = fits(&ws, st)?;
    }
    for b in vb.iter_mut() {
        let mut ws = vec![b.sa.w[0], b.sa.w[1], b.sa.w[2], b.sa.w[3], b.ca.w[0], b.ca.w[1], b.ca.w[2], b.ca.w[3], b.ffn.w1, b.ffn.w2];
        if let Some((_, _, a)) = &b.v2a {
            ws.extend_from_slice(&a.w);
        }
        b.keep = fits(&ws, st)?;
    }
    Ok(S2vCuda {
        dit: dit as *const S2vDit,
        model,
        g: *g,
        h,
        ff,
        nh,
        vb,
        tb,
        ab,
        consts,
        o_clean,
        o_vcos,
        o_vsin,
        o_tcos,
        o_tsin,
        o_emb,
        o_emb_b,
        o_mask0,
        o_out_b,
        out_w16,
        vm,
        tm,
        fm,
        text0: dit.embed_text(&cond.text),
        nt,
        tcache: (0..n_t).map(|_| None).collect(),
        audio_feats: cond.audio.clone(),
        reference: cond.reference.clone(),
        chunk: None,
        acond: None,
        agroups: None,
        na: 0,
        x0: Vec::new(),
        cfg_in_visual: cfg.in_visual,
        cfg_out_visual: cfg.out_visual,
    })
}

/// The working buffers of a stream of `rows` rows (the visual one, or the
/// short prompt and speech streams under their own names).
struct Work {
    xn: u64,
    y: u64,
    hq: u64,
    hk: u64,
    hv: u64,
    ao: u64,
    ao32: u64,
    act: u64,
    sc: u64,
    w: u64,
    cq: usize,
}

fn work(c: &Ctx, st: &mut State, rows: usize, h: usize, ff: usize, short: bool) -> Res<Work> {
    let cq = (PANEL / (ff.max(h) * 4)).clamp(16, 8192);
    let n = |a: &'static str, b: &'static str| if short { a } else { b };
    Ok(Work {
        xn: st.h3.buf(c, n("s2s xn", "s2 xn"), rows * h * 2)?,
        y: st.h3.buf(c, "s2 y", cq * ff.max(h) * 4)?,
        hq: st.h3.buf(c, n("s2s hq", "s2 hq"), rows * h * 4)?,
        hk: st.h3.buf(c, n("s2s hk", "s2 hk"), rows * h * 4)?,
        hv: st.h3.buf(c, n("s2s hv", "s2 hv"), rows * h * 4)?,
        ao: st.h3.buf(c, n("s2s ao", "s2 ao"), rows * h * 2)?,
        ao32: st.h3.buf(c, n("s2s ao32", "s2 ao32"), rows * h * 4)?,
        act: st.h3.buf(c, "s2 act", cq * ff * 2)?,
        sc: st.h3.buf(c, "s2 sc", PANEL)?,
        w: st.h3.buf(c, "s2 w", ff.max(h) * h * 2)?,
        cq,
    })
}

impl S2vCuda {
    fn at(&self, o: usize) -> u64 {
        self.consts.ptr + (o * 4) as u64
    }

    /// `src` f16 [rows][k] through tensor `ix`, a chunk of rows at a time into
    /// `wk.y`; `after(r0, n)` takes each chunk from there.
    unsafe fn project(&self, c: &Ctx, st: &mut State, wk: &Work, ix: usize, keep: bool, src: u64, rows: usize, k: usize, alpha: f32, after: &mut dyn FnMut(&mut State, usize, usize) -> Res<()>) -> Res<usize> {
        let (out, kk) = unsafe { expand(c, st, &self.model, ix, wk.w, keep, true)? };
        if kk != k {
            return Err("a projection of the wrong width".into());
        }
        let mut r0 = 0usize;
        while r0 < rows {
            let n = wk.cq.min(rows - r0);
            unsafe { c.gemm(true, false, out, n, k, alpha, wk.w, k, src + (r0 * k * 2) as u64, k, 0.0, wk.y, out, Gemm::Half)? };
            after(st, r0, n)?;
            r0 += n;
        }
        Ok(out)
    }

    /// q, k, v of `rows` rows through an attention's projections with their
    /// bias, q and k normed (and rotated by `rope` at its row offset): into
    /// wk.hq / hk / hv, head-major under `hmaj`. `kv_src` (rows `kv_rows`)
    /// gives the keys and values when they come from another stream.
    #[allow(clippy::too_many_arguments)]
    unsafe fn qkv(&self, c: &Ctx, st: &mut State, f: &S2Funcs, wk: &Work, a: &AttnD, keep: bool, src: u64, rows: usize, kv: Option<(u64, usize)>, rope: Option<(u64, u64)>, hmaj: bool) -> Res<()> {
        let (h, nh, hd) = (self.h, self.nh, s2v::HEAD_DIM);
        let (kv_src, kv_rows) = kv.unwrap_or((src, rows));
        for (part, dst) in [wk.hq, wk.hk, wk.hv].into_iter().enumerate() {
            let (s, n) = if part == 0 { (src, rows) } else { (kv_src, kv_rows) };
            let wn = match part {
                0 => self.at(a.qn),
                1 => self.at(a.kn),
                _ => 0,
            };
            let bias = self.at(a.b[part]);
            let rot = if part < 2 && kv.is_none() { rope } else { None };
            let (ti, hi, hdi, hm) = (n as i32, nh as i32, hd as i32, hmaj as i32);
            unsafe {
                self.project(c, st, wk, a.w[part], keep, s, n, h, 1.0, &mut |_, r0, rr| {
                    let (cs_, sn_) = match rot {
                        Some((cs, sn)) => (cs + (r0 * hd / 2 * 4) as u64, sn + (r0 * hd / 2 * 4) as u64),
                        None => (0, 0),
                    };
                    let r0i = r0 as i32;
                    c.launch(f.heads, (rr as u32, nh as u32), 32, 0, kargs!(wk.y, bias, dst, wn, cs_, sn_, ti, r0i, hi, hdi, EPS, hm))
                })?;
            }
        }
        Ok(())
    }

    /// Attention of `nq` token-major query rows (wk.hq) over `nk` keys and
    /// values (wk.hk, wk.hv), all visible, into wk.ao (f16 token-major): TF32
    /// products around a softmax, a head at a time.
    unsafe fn attn_small(&self, c: &Ctx, st: &mut State, wk: &Work, ao32: u64, nq: usize, nk: usize) -> Res<()> {
        let (h, nh, hd) = (self.h, self.nh, s2v::HEAD_DIM);
        let lx = super::ltx::funcs(st, c)?;
        let to_half = st.h3.funcs(c)?.to_half;
        let cqs = (PANEL / (nk * 4)).clamp(1, nq);
        let scale = 1.0 / (hd as f32).sqrt();
        let nki = nk as i32;
        for hh in 0..nh {
            let off = (hh * hd * 4) as u64;
            let mut q0 = 0usize;
            while q0 < nq {
                let rows = cqs.min(nq - q0);
                unsafe {
                    c.gemm(true, false, nk, rows, hd, scale, wk.hk + off, h, wk.hq + off + (q0 * h * 4) as u64, h, 0.0, wk.sc, nk, Gemm::Tf32)?;
                    c.launch(lx.softmax_f32, (rows as u32, 1), 256, 0, kargs!(wk.sc, nki))?;
                    c.gemm(false, false, hd, rows, nk, 1.0, wk.hv + off, h, wk.sc, nk, 0.0, ao32 + off + (q0 * h * 4) as u64, h, Gemm::Tf32)?;
                }
                q0 += rows;
            }
        }
        let n = (nq * h) as u64;
        unsafe { c.launch(to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(ao32, wk.ao, n)) }
    }

    /// The output product of wk.ao (rows `rows`) with its bias, gated (by the
    /// clean / noisy gate chunks, or none) onto x at row `row0`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn out_add(&self, c: &Ctx, st: &mut State, f: &S2Funcs, wk: &Work, ix: usize, b: usize, keep: bool, x: u64, rows: usize, row0: usize, gates: Option<(u64, u64, usize)>) -> Res<()> {
        let h = self.h;
        let (gc, gn, ncl) = gates.unwrap_or((0, 0, 0));
        let (hi, ncli) = (h as i32, ncl as i32);
        let bias = self.at(b);
        unsafe {
            self.project(c, st, wk, ix, keep, wk.ao, rows, h, 1.0, &mut |_, r0, n| {
                let tot = (n * h) as u64;
                let r0i = (row0 + r0) as i32;
                c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x + ((row0 + r0) * h * 4) as u64, wk.y, bias, gc, gn, ncli, r0i, hi, tot))
            })?;
        }
        Ok(())
    }

    /// The feed-forward of wk.xn (rows `rows`), gated onto x.
    unsafe fn ffn(&self, c: &Ctx, st: &mut State, f: &S2Funcs, wk: &Work, d: &FfnD, keep: bool, x: u64, rows: usize, gates: Option<(u64, u64, usize)>) -> Res<()> {
        let (h, ff) = (self.h, self.ff);
        let (gc, gn, ncl) = gates.unwrap_or((0, 0, 0));
        let w2 = st.h3.buf(c, "s2 w2", ff.max(h) * h * 2)?;
        let (mid, k1) = unsafe { expand(c, st, &self.model, d.w1, wk.w, keep, true)? };
        let (o, k2) = unsafe { expand(c, st, &self.model, d.w2, w2, keep, true)? };
        if mid != ff || k1 != h || o != h || k2 != ff {
            return Err("a feed-forward of the wrong shape".into());
        }
        let (b1, b2) = (self.at(d.b1), self.at(d.b2));
        let (hi, ffi, ncli) = (h as i32, ff as i32, ncl as i32);
        let mut r0 = 0usize;
        while r0 < rows {
            let n = wk.cq.min(rows - r0);
            unsafe {
                c.gemm(true, false, ff, n, h, 1.0, wk.w, h, wk.xn + (r0 * h * 2) as u64, h, 0.0, wk.y, ff, Gemm::Half)?;
                let tot = (n * ff) as u64;
                c.launch(f.gelu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(wk.y, b1, wk.act, MID_INV, ffi, tot))?;
                c.gemm(true, false, h, n, ff, 1.0 / MID_INV, w2, ff, wk.act, ff, 0.0, wk.y, h, Gemm::Half)?;
                let tot = (n * h) as u64;
                let r0i = r0 as i32;
                c.launch(f.addg, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x + (r0 * h * 4) as u64, wk.y, b2, gc, gn, ncli, r0i, hi, tot))?;
            }
            r0 += n;
        }
        Ok(())
    }

    /// A short stream's block (prompt: modulated by `m` [6h]; speech: affine).
    #[allow(clippy::too_many_arguments)]
    unsafe fn short_block(&self, c: &Ctx, st: &mut State, f: &S2Funcs, wk: &Work, b: &SBlk, x: u64, n: usize, m: Option<u64>, rope: (u64, u64)) -> Res<()> {
        let h = self.h;
        let ln = |src: u64, sh: u64, sc: u64, aff: bool| -> Res<()> {
            let (z, hi, a) = (0i32, h as i32, aff as i32);
            unsafe { c.launch(f.ln, (n as u32, 1), 256, 0, kargs!(src, wk.xn, sh, sc, sh, sc, z, z, hi, EPS, 1.0f32, a)) }
        };
        let fl = |o: usize| self.at(o);
        let mh = |k: usize| m.unwrap_or(0) + (k * h * 4) as u64;
        match (m, b.n1) {
            (Some(_), _) => ln(x, mh(0), mh(1), false)?,
            (None, Some((w, bb))) => ln(x, fl(bb), fl(w), true)?,
            _ => return Err("a short block without its norm".into()),
        }
        let ao32 = wk.ao32;
        unsafe {
            self.qkv(c, st, f, wk, &b.attn, b.keep, wk.xn, n, None, Some(rope), false)?;
            self.attn_small(c, st, wk, ao32, n, n)?;
            let g = m.map(|_| (mh(2), mh(2), 0usize));
            self.out_add(c, st, f, wk, b.attn.w[3], b.attn.b[3], b.keep, x, n, 0, g)?;
        }
        match (m, b.n2) {
            (Some(_), _) => ln(x, mh(3), mh(4), false)?,
            (None, Some((w, bb))) => ln(x, fl(bb), fl(w), true)?,
            _ => {}
        }
        let g = m.map(|_| (mh(5), mh(5), 0usize));
        unsafe { self.ffn(c, st, f, wk, &b.ffn, b.keep, x, n, g) }
    }

    unsafe fn step_inner(&mut self, c: &Ctx, st: &mut State, step: usize, chunk: usize, x_in: &[f32], memory: &[Vec<f32>; 3]) -> Res<Vec<f32>> {
        let f = funcs(st, c)?;
        let (h, nh, hd) = (self.h, self.nh, s2v::HEAD_DIM);
        let g = self.g;
        let (m, clean, nvis) = (g.total(), g.clean(), g.nvis);
        // SAFETY: the DiT outlives the stepper (`generate` drops the stepper first).
        let dit = unsafe { &*self.dit };
        if step >= self.tm.len() {
            return Err("a step past the render's schedule".into());
        }
        let mods = st.h3.buf(c, "s2 mods", self.vb.len() * 9 * h * 4 + 64)?;
        let tmods = st.h3.buf(c, "s2 tmods", self.tb.len() * 6 * h * 4 + 64)?;
        let fmods = st.h3.buf(c, "s2 fmods", 2 * h * 4)?;
        unsafe {
            cu::memcpy_htod_async(mods, &self.vm[step], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(tmods, &self.tm[step], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(fmods, &self.fm[step], c.stream).map_err(derr)?;
        }

        // ── the speech stream and the clean rows, once a chunk ──
        if self.chunk != Some(chunk) {
            let a0 = dit.embed_audio(self.audio_feats.get(chunk).ok_or("a speech chunk out of range")?);
            let na = a0.len() / h;
            let ab = Buf::direct(na * h * 4)?;
            let (ac, asn) = s2v::rope_1d(&s2v::audio_rope_pos(na, g.lt));
            let arope = st.h3.buf(c, "s2 arope", 2 * ac.len() * 4)?;
            let grp: Vec<u32> = s2v::audio_groups(na, g.lt).into_iter().flat_map(|(a, b)| [a as u32, b as u32]).collect();
            if grp.chunks(2).any(|p| p[1] > 64) {
                return Err("a frame with more than 64 speech tokens".into());
            }
            let gb = Buf::direct(grp.len() * 4)?;
            unsafe {
                cu::memcpy_htod_async(ab.ptr, &a0, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(arope, &ac, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(arope + (ac.len() * 4) as u64, &asn, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(gb.ptr, &grp, c.stream).map_err(derr)?;
            }
            let wk = work(c, st, na.max(self.nt), h, self.ff, true)?;
            for b in &self.ab {
                unsafe { self.short_block(c, st, &f, &wk, b, ab.ptr, na, None, (arope, arope + (ac.len() * 4) as u64))? };
            }
            self.acond = Some(ab);
            self.agroups = Some(gb);
            self.na = na;
            self.x0 = dit.embed_clean(&self.reference, memory, &g);
            self.chunk = Some(chunk);
        }
        // ── the prompt stream, once a step ──
        if self.tcache[step].is_none() {
            let tb = Buf::direct(self.nt * h * 4)?;
            unsafe { cu::memcpy_htod_async(tb.ptr, &self.text0, c.stream).map_err(derr)? };
            let wk = work(c, st, self.na.max(self.nt), h, self.ff, true)?;
            for (bi, b) in self.tb.iter().enumerate() {
                let mb = tmods + (bi * 6 * h * 4) as u64;
                unsafe { self.short_block(c, st, &f, &wk, b, tb.ptr, self.nt, Some(mb), (self.at(self.o_tcos), self.at(self.o_tsin)))? };
            }
            self.tcache[step] = Some(tb);
        }
        // the conditioning streams as f16 product inputs
        let to_half = st.h3.funcs(c)?.to_half;
        let t16 = st.h3.buf(c, "s2 t16", self.nt * h * 2)?;
        let a16 = st.h3.buf(c, "s2 a16", self.na * h * 2)?;
        unsafe {
            let (nt, na) = ((self.nt * h) as u64, (self.na * h) as u64);
            c.launch(to_half, (nt.div_ceil(256) as u32, 1), 256, 0, kargs!(self.tcache[step].as_ref().unwrap().ptr, t16, nt))?;
            c.launch(to_half, (na.div_ceil(256) as u32, 1), 256, 0, kargs!(self.acond.as_ref().unwrap().ptr, a16, na))?;
        }

        // ── the visual stream ──
        let x = st.h3.buf(c, "s2 x", m * h * 4)?;
        let tok_h = s2v::patchify(x_in, self.cfg_in_visual, 1, 2, g.lh, g.lw);
        if tok_h.len() != nvis * 64 || self.x0.len() != clean * h {
            return Err("a latent or clean rows of the wrong size".into());
        }
        let tok = st.h3.buf(c, "s2 tok", nvis * 64 * 4)?;
        let xt = x + (clean * h * 4) as u64;
        let hi = h as i32;
        unsafe {
            cu::memcpy_htod_async(x, &self.x0, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(tok, &tok_h, c.stream).map_err(derr)?;
            c.gemm(true, false, h, nvis, 64, 1.0, self.at(self.o_emb), 64, tok, 64, 0.0, xt, h, Gemm::F32)?;
            let tot = (nvis * h) as u64;
            c.launch(f.addvec, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xt, self.at(self.o_emb_b), self.at(self.o_mask0), hi, tot))?;
        }
        let wk = work(c, st, m, h, self.ff, false)?;
        let h3f = st.h3.funcs(c)?;
        let fl = st.h3.flash(c, hd, false).ok_or("no flash attention kernel")?;
        let ns = m.next_multiple_of(16);
        let qp = st.h3.buf(c, "s2 qp", nh * m * hd * 2)?;
        let kp = st.h3.buf(c, "s2 kp", nh * m * hd * 2)?;
        let vp = st.h3.buf(c, "s2 vp", nh * ns * hd * 2 + 64)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let (mi, nsi, nhi, hdi, ncli) = (m as i32, ns as i32, nh as i32, hd as i32, clean as i32);
        let ln = |src: u64, rows: usize, row0: usize, cm: u64, nm: u64, k: usize, aff: bool| -> Res<()> {
            let (r0i, a) = (row0 as i32, aff as i32);
            let off = (k * h * 4) as u64;
            let (shc, scc, shn, scn) = if aff { (cm, nm, cm, nm) } else { (cm + off, cm + off + (h * 4) as u64, nm + off, nm + off + (h * 4) as u64) };
            unsafe { c.launch(f.ln, (rows as u32, 1), 256, 0, kargs!(src, wk.xn, shc, scc, shn, scn, ncli, r0i, hi, EPS, 1.0f32, a)) }
        };
        let prof = std::env::var("CMF_S2V_CUDA_PROF").as_deref() == Ok("1");
        let mut times = [0f64; 8];
        let mut lap = |k: usize, t: &mut std::time::Instant| {
            if prof {
                let _ = c.sync();
                times[k] += t.elapsed().as_secs_f64();
                *t = std::time::Instant::now();
            }
        };
        for (bi, b) in self.vb.iter().enumerate() {
            let mut tt = std::time::Instant::now();
            let (cm, nm) = (self.at(self.o_clean + bi * 9 * h), mods + (bi * 9 * h * 4) as u64);
            let gate = |k: usize| Some((cm + (k * h * 4) as u64, nm + (k * h * 4) as u64, clean));
            // self-attention over every row
            ln(x, m, 0, cm, nm, 0, false)?;
            lap(0, &mut tt);
            unsafe {
                self.qkv(c, st, &f, &wk, &b.sa, b.keep, wk.xn, m, None, Some((self.at(self.o_vcos), self.at(self.o_vsin))), true)?;
                lap(1, &mut tt);
                let grid = ((nh * ns * hd).div_ceil(256) as u32, 1);
                c.launch(h3f.pack, grid, 256, 0, kargs!(wk.hq, qp, mi, nsi, nhi, hdi, hdi, 0i32))?;
                c.launch(h3f.pack, grid, 256, 0, kargs!(wk.hk, kp, mi, nsi, nhi, hdi, hdi, 1i32))?;
                c.launch(h3f.pack, grid, 256, 0, kargs!(wk.hv, vp, mi, nsi, nhi, hdi, hdi, 2i32))?;
                lap(2, &mut tt);
                c.launch(fl, (m.div_ceil(64) as u32, nh as u32), 128, 0, kargs!(qp, kp, vp, wk.ao, mi, nsi, nhi, scale, 1i32))?;
                lap(3, &mut tt);
                self.out_add(c, st, &f, &wk, b.sa.w[3], b.sa.b[3], b.keep, x, m, 0, gate(2))?;
                lap(4, &mut tt);
            }
            // the prompt cross-attention (queries token-major in hq)
            ln(x, m, 0, cm, nm, 3, false)?;
            unsafe {
                self.qkv(c, st, &f, &wk, &b.ca, b.keep, wk.xn, m, Some((t16, self.nt)), None, false)?;
                self.attn_small(c, st, &wk, wk.ao32, m, self.nt)?;
                self.out_add(c, st, &f, &wk, b.ca.w[3], b.ca.b[3], b.keep, x, m, 0, gate(5))?;
            }
            lap(5, &mut tt);
            // the speech adapter over the noisy tail
            if let Some((nw, nb, a)) = &b.v2a {
                ln(xt, nvis, clean, self.at(*nb), self.at(*nw), 0, true)?;
                unsafe {
                    self.qkv(c, st, &f, &wk, a, b.keep, wk.xn, nvis, Some((a16, self.na)), None, false)?;
                    let per = g.per_frame() as i32;
                    c.launch(f.gattn, (nvis as u32, nh as u32), 32, 0, kargs!(wk.hq, wk.hk, wk.hv, wk.ao, self.agroups.as_ref().unwrap().ptr, per, nhi, hdi, scale))?;
                    self.out_add(c, st, &f, &wk, a.w[3], a.b[3], b.keep, x, nvis, clean, None)?;
                }
            }
            lap(6, &mut tt);
            // the feed-forward
            ln(x, m, 0, cm, nm, 6, false)?;
            unsafe { self.ffn(c, st, &f, &wk, &b.ffn, b.keep, x, m, gate(8))? };
            lap(7, &mut tt);
        }
        if prof {
            let names = ["norm", "qkv", "pack", "flash", "o", "prompt", "speech", "ffn"];
            let s: Vec<String> = names.iter().zip(times).map(|(n, t)| format!("{n} {t:.2}")).collect();
            eprintln!("s2v cuda step {step}, 60 blocks, s: {}", s.join(" · "));
        }
        // ── the head over the noisy tail ──
        let ob = st.h3.buf(c, "s2 out", nvis * 320 * 4)?;
        unsafe {
            let z = 0i32;
            c.launch(f.ln, (nvis as u32, 1), 256, 0, kargs!(xt, wk.xn, fmods, fmods + (h * 4) as u64, fmods, fmods + (h * 4) as u64, z, z, hi, EPS, 1.0f32, z))?;
            c.gemm(true, false, 320, nvis, h, 1.0, self.out_w16.ptr, h, wk.xn, h, 0.0, ob, 320, Gemm::Half)?;
            let tot = (nvis * 320) as u64;
            let nul = 0u64;
            c.launch(f.addvec, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(ob, self.at(self.o_out_b), nul, 320i32, tot))?;
        }
        let mut y = vec![0f32; nvis * 320];
        unsafe { cu::memcpy_dtoh_async(&mut y, ob, c.stream).map_err(derr)? };
        c.sync()?;
        if y.iter().any(|v| !v.is_finite()) {
            return Err("the forward produced a value that is not finite".into());
        }
        Ok(s2v::unpatchify_out(&y, &g, self.cfg_out_visual))
    }
}

impl crate::gpu::S2vStepper for S2vCuda {
    fn step(&mut self, step: usize, chunk: usize, x: &[f32], memory: &[Vec<f32>; 3]) -> Result<Vec<f32>, String> {
        let c = ctx().ok_or("no CUDA context")?;
        let mut st = c.st.lock().unwrap();
        let r = unsafe { self.step_inner(c, &mut st, step, chunk, x, memory) };
        if r.is_err() {
            let _ = c.sync();
        }
        r
    }
}
