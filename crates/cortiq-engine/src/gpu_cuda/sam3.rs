//! SAM3's ViT trunk on CUDA: the patch rows up once, the 32 layers resident,
//! the trunk features back once. The patch embedding is an f32 product, the
//! tiled position table and `ln_pre` follow; a layer is LayerNorm into an f16
//! product input, the fused q|k|v product, its bias and axial rotation into
//! the flash planes (`sm_pack`: the 24 x 24 windows as groups of 576 rows, or
//! the four global layers over all 5184), `h3_flash2` at head width 64, the
//! output product added back in raster order, then LayerNorm, the erf-GELU
//! MLP and its residual. The weights are f16 planes expanded from the file's
//! codec once per model. `CMF_SAM3_CUDA=0` keeps the host trunk.

use super::*;
use crate::sam3 as s3;

const SM_SRC: &str = include_str!("sam3.cu");

#[derive(Clone, Copy)]
pub(super) struct SmFuncs {
    ln: cs::CUfunction,
    pack: cs::CUfunction,
    addrows: cs::CUfunction,
    gelu: cs::CUfunction,
    add: cs::CUfunction,
    im2col: cs::CUfunction,
    bias_ch: cs::CUfunction,
    convt: cs::CUfunction,
    softmax: cs::CUfunction,
    bias_rows: cs::CUfunction,
}

fn funcs(st: &mut State, c: &Ctx) -> Res<SmFuncs> {
    if let Some(f) = st.h3.sm_funcs {
        return Ok(f);
    }
    let m = compile(SM_SRC, Vec::new(), c.arch)?;
    let f = SmFuncs {
        ln: func(m, "sm_ln")?,
        pack: func(m, "sm_pack")?,
        addrows: func(m, "sm_addrows")?,
        gelu: func(m, "sm_gelu")?,
        add: func(m, "sm_add")?,
        im2col: func(m, "sm_im2col")?,
        bias_ch: func(m, "sm_bias_ch")?,
        convt: func(m, "sm_convt_scatter")?,
        softmax: func(m, "sm_softmax")?,
        bias_rows: func(m, "sm_bias_rows")?,
    };
    st.h3.sm_funcs = Some(f);
    Ok(f)
}

struct SmLayer {
    global: bool,
    n1: (Buf, Buf),
    n2: (Buf, Buf),
    qkv: Buf,
    qkv_b: Buf,
    proj: Buf,
    proj_b: Buf,
    fc1: Buf,
    fc1_b: Buf,
    fc2: Buf,
    fc2_b: Buf,
    rope: Buf,
}

/// One model's trunk weights on the card.
pub(super) struct SmDev {
    uid: u64,
    patch: Buf,
    pos: Buf,
    ln_pre: (Buf, Buf),
    layers: Vec<SmLayer>,
}

fn up_f32(c: &Ctx, v: &[f32]) -> Res<Buf> {
    let b = Buf::direct(v.len().max(1) * 4)?;
    unsafe { cu::memcpy_htod_async(b.ptr, v, c.stream).map_err(derr)? };
    Ok(b)
}

fn vec_of(m: &Arc<CmfModel>, name: &str) -> Res<Vec<f32>> {
    s3::load_vec(m, name).map_err(|e| e.to_string())
}

/// A [rows][cols] weight as an f16 plane: expanded from the file's codec, or
/// (a dense tensor) converted from f32.
unsafe fn plane(c: &Ctx, st: &mut State, m: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Res<Buf> {
    let idx = m.tensor_index(name).ok_or_else(|| format!("{name}: missing"))?;
    let b = Buf::direct(rows * cols * 2)?;
    unsafe {
        if packed(m, idx).is_ok() {
            let (r, k) = expand(c, st, m, idx, b.ptr, false, true)?;
            if (r, k) != (rows, cols) {
                return Err(format!("{name}: {r} x {k}, expected {rows} x {cols}"));
            }
        } else {
            let v = vec_of(m, name)?;
            if v.len() != rows * cols {
                return Err(format!("{name}: {} values, expected {}", v.len(), rows * cols));
            }
            let th = st.h3.funcs(c)?.to_half;
            let stage = st.h3.buf(c, "sm stage", v.len() * 4)?;
            cu::memcpy_htod_async(stage, &v, c.stream).map_err(derr)?;
            let n = v.len() as u64;
            c.launch(th, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(stage, b.ptr, n))?;
        }
    }
    Ok(b)
}

unsafe fn build(c: &Ctx, st: &mut State, m: &Arc<CmfModel>, pos_skip_cls: bool) -> Res<SmDev> {
    let (td, ti, grid) = (s3::TD, s3::TI, s3::GRID);
    let np = grid * grid;
    let patch = up_f32(c, &vec_of(m, "trunk.patch_embed.proj.weight")?)?;
    // the position table [1, 577, 1024] (Meta's cls row first) tiled 24 x 24 -> 72 x 72
    let pt = vec_of(m, "trunk.pos_embed")?;
    let base = if pos_skip_cls { 1 } else { 0 };
    if pt.len() < (base + 576) * td {
        return Err("pos_embed too small".into());
    }
    let mut pos = vec![0f32; np * td];
    for p in 0..np {
        let (py, px) = (p / grid, p % grid);
        let src = (base + (py % 24) * 24 + px % 24) * td;
        pos[p * td..(p + 1) * td].copy_from_slice(&pt[src..src + td]);
    }
    let pos = up_f32(c, &pos)?;
    let ln_pre = (up_f32(c, &vec_of(m, "trunk.ln_pre.weight")?)?, up_f32(c, &vec_of(m, "trunk.ln_pre.bias")?)?);
    let mut layers = Vec::with_capacity(s3::TL);
    for l in 0..s3::TL {
        let b = format!("trunk.blocks.{l}.");
        let v = |n: &str| -> Res<Buf> { up_f32(c, &vec_of(m, &format!("{b}{n}"))?) };
        let global = s3::GLOBAL.contains(&l);
        let rope = vec_of(m, &format!("{b}attn.freqs_cis"))?;
        let rows = if global { np } else { s3::WIN * s3::WIN };
        if rope.len() != rows * 64 {
            return Err(format!("layer {l}: a rotation table of {} values", rope.len()));
        }
        unsafe {
            layers.push(SmLayer {
                global,
                n1: (v("norm1.weight")?, v("norm1.bias")?),
                n2: (v("norm2.weight")?, v("norm2.bias")?),
                qkv: plane(c, st, m, &format!("{b}attn.qkv.weight"), 3 * td, td)?,
                qkv_b: v("attn.qkv.bias")?,
                proj: plane(c, st, m, &format!("{b}attn.proj.weight"), td, td)?,
                proj_b: v("attn.proj.bias")?,
                fc1: plane(c, st, m, &format!("{b}mlp.fc1.weight"), ti, td)?,
                fc1_b: v("mlp.fc1.bias")?,
                fc2: plane(c, st, m, &format!("{b}mlp.fc2.weight"), td, ti)?,
                fc2_b: v("mlp.fc2.bias")?,
                rope: up_f32(c, &rope)?,
            });
        }
    }
    c.sync()?;
    st.h3.bufs.remove("sm stage");
    Ok(SmDev { uid: m.uid(), patch, pos, ln_pre, layers })
}

/// The trunk features `[5184][1024]` of the patch rows `pv` (`[5184][588]`,
/// channel-major 14 x 14 patches), or `None` (said once) for the host trunk.
pub fn sam3_trunk(m: &Arc<CmfModel>, pv: &[f32], pos_skip_cls: bool) -> Option<Vec<f32>> {
    if std::env::var("CMF_SAM3_CUDA").as_deref() == Ok("0") {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { run(c, &mut st, m, pv, pos_skip_cls) } {
        Ok(x) => Some(x),
        Err(e) => {
            let _ = c.sync();
            decline("sam3 trunk", &e);
            None
        }
    }
}

unsafe fn run(c: &Ctx, st: &mut State, m: &Arc<CmfModel>, pv: &[f32], pos_skip_cls: bool) -> Res<Vec<f32>> {
    let (td, ti, grid, th) = (s3::TD, s3::TI, s3::GRID, s3::TH);
    let np = grid * grid;
    let kc = 3 * s3::PATCH * s3::PATCH;
    if pv.len() != np * kc || td != th * 64 {
        return Err("patch rows of the wrong size".into());
    }
    let f = funcs(st, c)?;
    let fl = st.h3.flash(c, 64, false).ok_or("no flash attention kernel for head width 64")?;
    if st.h3.sm.as_ref().is_none_or(|d| d.uid != m.uid()) {
        st.h3.sm = None;
        let d = unsafe { build(c, st, m, pos_skip_cls)? };
        st.h3.sm = Some(d);
    }
    let d = st.h3.sm.take().unwrap();
    let r = unsafe { layers(c, st, &f, fl, &d, pv, np, kc, td, ti) };
    st.h3.sm = Some(d);
    r
}

#[allow(clippy::too_many_arguments)]
unsafe fn layers(
    c: &Ctx,
    st: &mut State,
    f: &SmFuncs,
    fl: cs::CUfunction,
    d: &SmDev,
    pv: &[f32],
    np: usize,
    kc: usize,
    td: usize,
    ti: usize,
) -> Res<Vec<f32>> {
    let prof = std::env::var("SAM3_STAGES").is_ok_and(|v| v != "0");
    let t0 = std::time::Instant::now();
    let x = st.h3.buf(c, "sm x", np * td * 4)?;
    let pvb = st.h3.buf(c, "sm pv", np * kc * 4)?;
    let xn = st.h3.buf(c, "sm xn", np * ti * 2)?;
    let y = st.h3.buf(c, "sm y", np * ti * 4)?;
    let q = st.h3.buf(c, "sm q", np * td * 2)?;
    let k = st.h3.buf(c, "sm k", np * td * 2)?;
    let v = st.h3.buf(c, "sm v", np * td * 2)?;
    let ao = st.h3.buf(c, "sm ao", np * td * 2)?;
    let (tdi, tii, npi) = (td as i32, ti as i32, np as i32);
    let ln = |src: u64, dst: u64, w: &(Buf, Buf)| -> Res<()> {
        unsafe { c.launch(f.ln, (np as u32, 1), 256, 0, kargs!(src, dst, w.0.ptr, w.1.ptr, tdi, s3::T_EPS)) }
    };
    unsafe {
        cu::memcpy_htod_async(pvb, pv, c.stream).map_err(derr)?;
        // x [np][td] = pv [np][kc] · patchᵀ, plus the positions, then ln_pre in place
        c.gemm(true, false, td, np, kc, 1.0, d.patch.ptr, kc, pvb, kc, 0.0, x, td, Gemm::F32)?;
        let tot = (np * td) as u64;
        c.launch(f.add, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, d.pos.ptr, tot))?;
        ln(x, 0, &d.ln_pre)?;
        for l in &d.layers {
            let win = (!l.global) as i32;
            let (groups, n) = if l.global { (1usize, np) } else { (9usize, s3::WIN * s3::WIN) };
            ln(x, xn, &l.n1)?;
            // y [np][3 td] = xn · qkvᵀ
            c.gemm(true, false, 3 * td, np, td, 1.0, l.qkv.ptr, td, xn, td, 0.0, y, 3 * td, Gemm::Half)?;
            let pt = (np * s3::TH * 32) as u64;
            c.launch(f.pack, (pt.div_ceil(256) as u32, 1), 256, 0, kargs!(y, l.qkv_b.ptr, l.rope.ptr, q, k, v, npi, win))?;
            let (ni, nhi, one) = (n as i32, s3::TH as i32, 1i32);
            let scale = 0.125f32;
            c.launch(fl, (n.div_ceil(64) as u32, (groups * s3::TH) as u32), 128, 0, kargs!(q, k, v, ao, ni, ni, nhi, scale, one))?;
            // y [np][td] = ao · projᵀ (group order), added back in raster order
            c.gemm(true, false, td, np, td, 1.0, l.proj.ptr, td, ao, td, 0.0, y, td, Gemm::Half)?;
            c.launch(f.addrows, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, y, l.proj_b.ptr, npi, tdi, win))?;
            ln(x, xn, &l.n2)?;
            c.gemm(true, false, ti, np, td, 1.0, l.fc1.ptr, td, xn, td, 0.0, y, ti, Gemm::Half)?;
            let ht = (np * ti) as u64;
            c.launch(f.gelu, (ht.div_ceil(256) as u32, 1), 256, 0, kargs!(y, l.fc1_b.ptr, xn, tii, ht))?;
            c.gemm(true, false, td, np, ti, 1.0, l.fc2.ptr, ti, xn, ti, 0.0, y, td, Gemm::Half)?;
            let zero = 0i32;
            c.launch(f.addrows, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, y, l.fc2_b.ptr, npi, tdi, zero))?;
        }
    }
    let mut out = vec![0f32; np * td];
    unsafe { cu::memcpy_dtoh_async(&mut out, x, c.stream).map_err(derr)? };
    c.sync()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err("the trunk came back with a value that is not finite".into());
    }
    if prof {
        eprintln!("[sam3-prof] cuda trunk {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(out)
}

/// The bytes of one im2col panel or score block.
const PANEL: usize = 192 << 20;

fn with_ctx<T>(what: &str, f: impl FnOnce(&Ctx, &mut State) -> Res<T>) -> Option<T> {
    let c = ctx()?;
    if c.current().is_err() || std::env::var("CMF_SAM3_CUDA").as_deref() == Ok("0") {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    match f(c, &mut st) {
        Ok(v) => Some(v),
        Err(e) => {
            let _ = c.sync();
            decline(what, &e);
            None
        }
    }
}

/// `[co][oh][ow]` = conv(x `[ci][h][w]`) + b with zero padding, the products
/// in f32 (`sam3::conv2d`'s arithmetic, TF32 operands).
#[allow(clippy::too_many_arguments)]
pub fn sam3_conv2d(w: &[f32], b: &[f32], x: &[f32], ci: usize, h: usize, wid: usize, kh: usize, kw: usize, stride: usize, pad: usize) -> Option<Vec<f32>> {
    let co = b.len();
    if w.len() != co * ci * kh * kw || x.len() != ci * h * wid || stride == 0 || h + 2 * pad < kh || wid + 2 * pad < kw {
        return None;
    }
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (wid + 2 * pad - kw) / stride + 1;
    with_ctx("sam3 conv", |c, st| {
        let f = funcs(st, c)?;
        let p = oh * ow;
        let kk = ci * kh * kw;
        let xb = st.h3.buf(c, "sm cx", x.len() * 4)?;
        let wb = st.h3.buf(c, "sm cw", w.len() * 4)?;
        let bb = st.h3.buf(c, "sm cb", co * 4)?;
        let yb = st.h3.buf(c, "sm cy", co * p * 4)?;
        let direct = kh == 1 && kw == 1 && stride == 1 && pad == 0;
        unsafe {
            cu::memcpy_htod_async(xb, x, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(wb, w, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(bb, b, c.stream).map_err(derr)?;
            if direct {
                // y [co][p] = w [co][ci] · x [ci][p]
                c.gemm(false, false, p, co, ci, 1.0, xb, p, wb, ci, 0.0, yb, p, Gemm::Tf32)?;
            } else {
                let np = (PANEL / (kk * 4)).clamp(1, p);
                let cols = st.h3.buf(c, "sm cols", np * kk * 4)?;
                let (cii, hi, wi, khi, kwi, si, pi, owi) = (ci as i32, h as i32, wid as i32, kh as i32, kw as i32, stride as i32, pad as i32, ow as i32);
                let mut p0 = 0usize;
                while p0 < p {
                    let n = np.min(p - p0);
                    let (p0u, nu, tot) = (p0 as u64, n as u64, (n * kk) as u64);
                    c.launch(f.im2col, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xb, cols, cii, hi, wi, khi, kwi, si, pi, owi, p0u, nu))?;
                    // y [co][p0..p0+n] = w [co][kk] · cols [kk][n]
                    c.gemm(false, false, n, co, kk, 1.0, cols, n, wb, kk, 0.0, yb + (p0 * 4) as u64, p, Gemm::Tf32)?;
                    p0 += n;
                }
            }
            let (pu, tot) = (p as u64, (co * p) as u64);
            c.launch(f.bias_ch, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(yb, bb, pu, tot))?;
            let mut out = vec![0f32; co * p];
            cu::memcpy_dtoh_async(&mut out, yb, c.stream).map_err(derr)?;
            c.sync()?;
            Ok(out)
        }
    })
}

/// ConvTranspose2d(k = 2, s = 2): w `[ci][co][2][2]`, x `[ci][h][w]` →
/// `[co][2h][2w]`, one product of the four taps and a scatter.
pub fn sam3_convt2x2(w: &[f32], b: &[f32], x: &[f32], ci: usize, h: usize, wid: usize) -> Option<Vec<f32>> {
    let co = b.len();
    if w.len() != ci * co * 4 || x.len() != ci * h * wid {
        return None;
    }
    with_ctx("sam3 transposed conv", |c, st| {
        let f = funcs(st, c)?;
        let p = h * wid;
        let xb = st.h3.buf(c, "sm cx", x.len() * 4)?;
        let wb = st.h3.buf(c, "sm cw", w.len() * 4)?;
        let bb = st.h3.buf(c, "sm cb", co * 4)?;
        let yt = st.h3.buf(c, "sm cols", co * 4 * p * 4)?;
        let yb = st.h3.buf(c, "sm cy", co * 4 * p * 4)?;
        unsafe {
            cu::memcpy_htod_async(xb, x, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(wb, w, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(bb, b, c.stream).map_err(derr)?;
            // yt [co·4][p] = wᵀ ([ci][co·4] stored) · x [ci][p]
            c.gemm(false, true, p, co * 4, ci, 1.0, xb, p, wb, co * 4, 0.0, yt, p, Gemm::Tf32)?;
            let tot = (co * 4 * p) as u64;
            let (coi, hi, wi) = (co as i32, h as i32, wid as i32);
            c.launch(f.convt, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(yt, bb, yb, coi, hi, wi))?;
            let mut out = vec![0f32; co * 4 * p];
            cu::memcpy_dtoh_async(&mut out, yb, c.stream).map_err(derr)?;
            c.sync()?;
            Ok(out)
        }
    })
}

/// The attention core of `sam3::run_attn` on the card: q, k, v are the q|k|v
/// thirds of fused rows (`[n][3 dim]`, k at `dim`, v at `2 dim`), `mode` the
/// mask (0 none, 1 key `[n_k]`, 2 additive `[n_q][n_k]`, 3 per head
/// `[heads][n_q][n_k]`); `out` `[n_q][dim]`. Plain f32 products.
#[allow(clippy::too_many_arguments)]
pub fn sam3_attn_core(q: &[f32], k: &[f32], v: &[f32], n_q: usize, n_k: usize, dim: usize, heads: usize, mode: i32, mask: &[f32], scale: f32, out: &mut [f32]) -> bool {
    let h3 = dim * 3;
    let ops = [Op { x: q, ld: h3, off: 0 }, Op { x: k, ld: h3, off: dim }, Op { x: v, ld: h3, off: 2 * dim }];
    attn(ops, n_q, n_k, dim, heads, mode, mask, scale, out)
}

/// The same over separate rows: q `[n_q][dim]`, k and v `[n_k][dim]`, no mask.
#[allow(clippy::too_many_arguments)]
pub fn sam3_attn_sep(q: &[f32], k: &[f32], v: &[f32], n_q: usize, n_k: usize, dim: usize, heads: usize, scale: f32, out: &mut [f32]) -> bool {
    let ops = [Op { x: q, ld: dim, off: 0 }, Op { x: k, ld: dim, off: 0 }, Op { x: v, ld: dim, off: 0 }];
    attn(ops, n_q, n_k, dim, heads, 0, &[], scale, out)
}

/// An attention operand: rows `ld` wide, the heads from column `off`.
#[derive(Clone, Copy)]
struct Op<'a> {
    x: &'a [f32],
    ld: usize,
    off: usize,
}

#[allow(clippy::too_many_arguments)]
fn attn(ops: [Op<'_>; 3], n_q: usize, n_k: usize, dim: usize, heads: usize, mode: i32, mask: &[f32], scale: f32, out: &mut [f32]) -> bool {
    let hd = dim / heads.max(1);
    let need = match mode {
        0 => 0,
        1 => n_k,
        2 => n_q * n_k,
        _ => heads * n_q * n_k,
    };
    let rows_of = [n_q, n_k, n_k];
    if hd * heads != dim || n_k == 0 || mask.len() < need || out.len() < n_q * dim || ops.iter().zip(rows_of).any(|(o, r)| o.x.len() < r * o.ld || o.off + dim > o.ld) {
        return false;
    }
    with_ctx("sam3 attention", |c, st| {
        let f = funcs(st, c)?;
        // an operand that is the same host rows as an earlier one goes up once
        let names = ["sm aq", "sm ak", "sm av"];
        let mut dev = [0u64; 3];
        for i in 0..3 {
            let same = (0..i).find(|&j| std::ptr::eq(ops[j].x.as_ptr(), ops[i].x.as_ptr()) && ops[j].x.len() == ops[i].x.len());
            dev[i] = match same {
                Some(j) => dev[j],
                None => {
                    let b = st.h3.buf(c, names[i], ops[i].x.len() * 4)?;
                    unsafe { cu::memcpy_htod_async(b, ops[i].x, c.stream).map_err(derr)? };
                    b
                }
            };
        }
        let mb = st.h3.buf(c, "sm am", need.max(1) * 4)?;
        let ob = st.h3.buf(c, "sm ao32", n_q * dim * 4)?;
        let rows = (PANEL / (n_k * 4)).clamp(1, n_q);
        let sb = st.h3.buf(c, "sm as", rows * n_k * 4)?;
        let ([q, k, v], [qo, ko, vo]) = (dev, [ops[0], ops[1], ops[2]]);
        unsafe {
            if need > 0 {
                cu::memcpy_htod_async(mb, &mask[..need], c.stream).map_err(derr)?;
            }
            let nki = n_k as i32;
            for t in 0..heads {
                let hoff = t * hd;
                let mt = if mode == 3 { mb + (t * n_q * n_k * 4) as u64 } else { mb };
                let mut i0 = 0usize;
                while i0 < n_q {
                    let nr = rows.min(n_q - i0);
                    // S [nr][n_k] = Q_t [nr][hd] · K_tᵀ
                    let (kp, qp) = (k + ((ko.off + hoff) * 4) as u64, q + ((i0 * qo.ld + qo.off + hoff) * 4) as u64);
                    c.gemm(true, false, n_k, nr, hd, 1.0, kp, ko.ld, qp, qo.ld, 0.0, sb, n_k, Gemm::F32)?;
                    let r0 = i0 as i32;
                    c.launch(f.softmax, (nr as u32, 1), 256, 0, kargs!(sb, mt, nki, mode, scale, r0))?;
                    // O_t [nr][hd] = S · V_t
                    let vp = v + ((vo.off + hoff) * 4) as u64;
                    c.gemm(false, false, hd, nr, n_k, 1.0, vp, vo.ld, sb, n_k, 0.0, ob + ((i0 * dim + hoff) * 4) as u64, dim, Gemm::F32)?;
                    i0 += nr;
                }
            }
            cu::memcpy_dtoh_async(&mut out[..n_q * dim], ob, c.stream).map_err(derr)?;
            c.sync()?;
        }
        Ok(())
    })
    .is_some()
}

struct TxLayer {
    n1: (Buf, Buf),
    n2: (Buf, Buf),
    qkv: Buf,
    qkv_b: Buf,
    out: Buf,
    out_b: Buf,
    fc: Buf,
    fc_b: Buf,
    proj: Buf,
    proj_b: Buf,
}

/// One model's CLIP text tower on the card.
pub(super) struct TxDev {
    uid: u64,
    layers: Vec<TxLayer>,
    lnf: (Buf, Buf),
}

unsafe fn build_text(c: &Ctx, st: &mut State, m: &Arc<CmfModel>) -> Res<TxDev> {
    let (sd, si) = (s3::SD, s3::SI);
    let mut layers = Vec::with_capacity(s3::SL);
    for l in 0..s3::SL {
        let b = format!("text.encoder.transformer.resblocks.{l}.");
        let v = |n: &str| -> Res<Buf> { up_f32(c, &vec_of(m, &format!("{b}{n}"))?) };
        unsafe {
            layers.push(TxLayer {
                n1: (v("ln_1.weight")?, v("ln_1.bias")?),
                n2: (v("ln_2.weight")?, v("ln_2.bias")?),
                qkv: plane(c, st, m, &format!("{b}attn.in_proj_weight"), 3 * sd, sd)?,
                qkv_b: v("attn.in_proj_bias")?,
                out: plane(c, st, m, &format!("{b}attn.out_proj.weight"), sd, sd)?,
                out_b: v("attn.out_proj.bias")?,
                fc: plane(c, st, m, &format!("{b}mlp.c_fc.weight"), si, sd)?,
                fc_b: v("mlp.c_fc.bias")?,
                proj: plane(c, st, m, &format!("{b}mlp.c_proj.weight"), sd, si)?,
                proj_b: v("mlp.c_proj.bias")?,
            });
        }
    }
    let lnf = (up_f32(c, &vec_of(m, "text.encoder.ln_final.weight")?)?, up_f32(c, &vec_of(m, "text.encoder.ln_final.bias")?)?);
    c.sync()?;
    st.h3.bufs.remove("sm stage");
    Ok(TxDev { uid: m.uid(), layers, lnf })
}

/// The CLIP text tower over the embedded tokens `x` `[32][1024]` (in place,
/// through the final LayerNorm), `cm` the additive mask `[32][32]`.
pub fn sam3_text(m: &Arc<CmfModel>, x: &mut [f32], cm: &[f32]) -> bool {
    let (ss, sd, si) = (s3::SS, s3::SD, s3::SI);
    if x.len() != ss * sd || cm.len() != ss * ss || sd != 16 * 64 {
        return false;
    }
    let t_all = std::time::Instant::now();
    with_ctx("sam3 text tower", |c, st| {
        let f = funcs(st, c)?;
        let th = st.h3.funcs(c)?.to_half;
        if st.h3.sm_text.as_ref().is_none_or(|d| d.uid != m.uid()) {
            st.h3.sm_text = None;
            let d = unsafe { build_text(c, st, m)? };
            st.h3.sm_text = Some(d);
        }
        let d = st.h3.sm_text.take().unwrap();
        let t_run = std::time::Instant::now();
        let r = (|| -> Res<()> {
            let xb = st.h3.buf(c, "tx x", ss * sd * 4)?;
            let mb = st.h3.buf(c, "tx m", ss * ss * 4)?;
            let xn = st.h3.buf(c, "tx xn", ss * si * 2)?;
            let y = st.h3.buf(c, "tx y", ss * si * 4)?;
            let ao = st.h3.buf(c, "tx ao", ss * sd * 4)?;
            let sc = st.h3.buf(c, "tx s", ss * ss * 4)?;
            let (sdi, sii, ssi) = (sd as i32, si as i32, ss as i32);
            let (tot, h3) = ((ss * sd) as u64, 3 * sd);
            let zero = 0i32;
            unsafe {
                cu::memcpy_htod_async(xb, &*x, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(mb, cm, c.stream).map_err(derr)?;
                for l in &d.layers {
                    c.launch(f.ln, (ss as u32, 1), 256, 0, kargs!(xb, xn, l.n1.0.ptr, l.n1.1.ptr, sdi, s3::S_EPS))?;
                    c.gemm(true, false, h3, ss, sd, 1.0, l.qkv.ptr, sd, xn, sd, 0.0, y, h3, Gemm::Half)?;
                    let qt = (ss * h3) as u64;
                    let h3i = h3 as i32;
                    c.launch(f.bias_rows, (qt.div_ceil(256) as u32, 1), 256, 0, kargs!(y, l.qkv_b.ptr, h3i, qt))?;
                    for t in 0..16usize {
                        let off = (t * 64 * 4) as u64;
                        c.gemm(true, false, ss, ss, 64, 1.0, y + (sd * 4) as u64 + off, h3, y + off, h3, 0.0, sc, ss, Gemm::F32)?;
                        let (mode, scale) = (2i32, 0.125f32);
                        c.launch(f.softmax, (ss as u32, 1), 32, 0, kargs!(sc, mb, ssi, mode, scale, zero))?;
                        c.gemm(false, false, 64, ss, ss, 1.0, y + (2 * sd * 4) as u64 + off, h3, sc, ss, 0.0, ao + off, sd, Gemm::F32)?;
                    }
                    c.launch(th, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(ao, xn, tot))?;
                    c.gemm(true, false, sd, ss, sd, 1.0, l.out.ptr, sd, xn, sd, 0.0, y, sd, Gemm::Half)?;
                    c.launch(f.addrows, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xb, y, l.out_b.ptr, ssi, sdi, zero))?;
                    c.launch(f.ln, (ss as u32, 1), 256, 0, kargs!(xb, xn, l.n2.0.ptr, l.n2.1.ptr, sdi, s3::S_EPS))?;
                    c.gemm(true, false, si, ss, sd, 1.0, l.fc.ptr, sd, xn, sd, 0.0, y, si, Gemm::Half)?;
                    let ht = (ss * si) as u64;
                    c.launch(f.gelu, (ht.div_ceil(256) as u32, 1), 256, 0, kargs!(y, l.fc_b.ptr, xn, sii, ht))?;
                    c.gemm(true, false, sd, ss, si, 1.0, l.proj.ptr, si, xn, si, 0.0, y, sd, Gemm::Half)?;
                    c.launch(f.addrows, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xb, y, l.proj_b.ptr, ssi, sdi, zero))?;
                }
                let nul = 0u64;
                c.launch(f.ln, (ss as u32, 1), 256, 0, kargs!(xb, nul, d.lnf.0.ptr, d.lnf.1.ptr, sdi, s3::S_EPS))?;
                cu::memcpy_dtoh_async(&mut *x, xb, c.stream).map_err(derr)?;
                c.sync()?;
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err("the text tower came back with a value that is not finite".into());
            }
            Ok(())
        })();
        st.h3.sm_text = Some(d);
        if std::env::var("SAM3_STAGES").is_ok_and(|v| v != "0") {
            eprintln!("[sam3-prof] cuda text {:.1} ms (build {:.1} ms)", t_all.elapsed().as_secs_f64() * 1e3, (t_run - t_all).as_secs_f64() * 1e3);
        }
        r
    })
    .is_some()
}

/// `c[m][n] = a[m][k] · b[k][n]`, f32 products.
pub fn sam3_mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if a.len() != m * k || b.len() != k * n {
        return None;
    }
    with_ctx("sam3 product", |c, st| {
        let ab = st.h3.buf(c, "sm ma", a.len() * 4)?;
        let bb = st.h3.buf(c, "sm mb", b.len() * 4)?;
        let cb = st.h3.buf(c, "sm mc", m * n * 4)?;
        let mut out = vec![0f32; m * n];
        unsafe {
            cu::memcpy_htod_async(ab, a, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(bb, b, c.stream).map_err(derr)?;
            // column-major: cᵀ [n][m] = bᵀ · aᵀ
            c.gemm(false, false, n, m, k, 1.0, bb, n, ab, k, 0.0, cb, n, Gemm::F32)?;
            cu::memcpy_dtoh_async(&mut out, cb, c.stream).map_err(derr)?;
            c.sync()?;
        }
        Ok(out)
    })
}
