//! The skill bake's GDN layers on the card (`fcd.rs` `gdn_attn_fwd` /
//! `gdn_attn_bwd`): conv + SiLU, the delta-rule recurrence and its BPTT in
//! `bake.cu`, f32, the operator of `fcd_ops::gdn_*` (which runs it in f64 on
//! the host: the hot spot of a Qwen3.5-class bake — 18 of 24 layers of the
//! 0.8B, ~90 % of a step on a 7-core host). The projections stay where they
//! are (`gemm_nt` / `gemm_dx`: cuBLAS on this backend).

use super::*;
use std::collections::HashMap;
use std::sync::Mutex;

const BK_SRC: &str = include_str!("bake.cu");

#[derive(Clone, Copy)]
struct BkFuncs {
    conv_fwd: cs::CUfunction,
    conv_bwd: cs::CUfunction,
    gdn_fwd: cs::CUfunction,
    gdn_bwd: cs::CUfunction,
    ffn_act: cs::CUfunction,
    ffn_act_bwd: cs::CUfunction,
    attn_perm: cs::CUfunction,
    attn_unperm: cs::CUfunction,
    softmax_fwd: cs::CUfunction,
    softmax_bwd: cs::CUfunction,
    ce_rows: cs::CUfunction,
    embed: cs::CUfunction,
    rmsnorm: cs::CUfunction,
    rmsnorm_bwd: cs::CUfunction,
    add: cs::CUfunction,
    attn_prep: cs::CUfunction,
    attn_prep_bwd: cs::CUfunction,
    gate_mul: cs::CUfunction,
    gate_bwd: cs::CUfunction,
    gather: cs::CUfunction,
    scatter_add: cs::CUfunction,
    to_bf16: cs::CUfunction,
    col_dot_part: cs::CUfunction,
    col_dot_sum: cs::CUfunction,
    scale_rows: cs::CUfunction,
    scale_rows_bf16: cs::CUfunction,
}

#[derive(Default)]
struct BkState {
    /// By (dk, dv): the module's functions, or why it did not build.
    f: HashMap<(usize, usize), Result<BkFuncs, String>>,
    /// `bake_gemm_dx`'s frozen matrices on the card, by (address, values):
    /// a fingerprint of the values beside them; `w_bytes` held.
    w: HashMap<(usize, usize), (u64, Buf)>,
    /// The same by bf16 copy, for the resident step's `Gemm::Bf16` products
    /// (counted in `w_bytes` too).
    wbf: HashMap<(usize, usize), (u64, Buf)>,
    w_bytes: usize,
    /// A training forward's gate/up planes `[n][2·inter]` by layer, for its
    /// backward (`bake_ffn_bwd` takes them).
    planes: HashMap<usize, (usize, Buf)>,
    /// Scratch by name, grown to the largest call (a step's buffers are the
    /// same every call: no cuMemAlloc / cuMemFree, which synchronizes, per op).
    scr: HashMap<&'static str, Buf>,
    /// The resident step's per-layer saves (h1, inv2, the FFN plane), grown to
    /// the largest step.
    saves: Vec<(Buf, Buf, Buf)>,
    /// With `BkStep::through_attn`: each layer's input h, for the backward
    /// through its attention branch (recomputed from it).
    saves_in: Vec<Buf>,
    /// With output masks (`BkLayer::lin`) and a gradient: each layer's
    /// attention-output and FFN-down outputs before their gains.
    saves_lin: Vec<(Buf, Buf)>,
    /// With output masks: each projection's weight copy with its rows scaled
    /// by the step's gains, by (address, len, mode) of the host matrix.
    wsc: HashMap<(usize, usize, bool), Buf>,
}

impl BkState {
    /// A scratch buffer of at least `bytes` (its contents undefined).
    fn scr(&mut self, name: &'static str, bytes: usize) -> Res<u64> {
        let bytes = bytes.max(4);
        if !self.scr.get(name).is_some_and(|b| b.bytes >= bytes) {
            self.scr.remove(name);
            self.scr.insert(name, Buf::new(bytes.next_multiple_of(1 << 20))?);
        }
        Ok(self.scr[name].ptr)
    }

    /// `data` uploaded into scratch `name`.
    fn up(&mut self, c: &Ctx, name: &'static str, data: &[f32]) -> Res<u64> {
        let p = self.scr(name, data.len() * 4)?;
        unsafe { cu::memcpy_htod_async(p, data, c.stream).map_err(derr)? };
        Ok(p)
    }

    /// A frozen matrix on the card (kept for the next step, up to `w_cap`).
    fn weight(&mut self, w: &[f32]) -> Res<u64> {
        let fp = crate::gpu::fp_f32(w);
        let key = (w.as_ptr() as usize, w.len());
        if !self.w.get(&key).is_some_and(|(f, _)| *f == fp) {
            if let Some((_, old)) = self.w.remove(&key) {
                self.w_bytes -= old.bytes;
            }
            // `CMF_BAKE_W_GB` (default 24): a 4B replica is ~16 GB of f32
            if self.w_bytes + w.len() * 4 > w_cap() {
                self.w.clear();
                self.wbf.clear();
                self.w_bytes = 0;
            }
            let b = Buf::from_slice(w)?;
            self.w_bytes += b.bytes;
            self.w.insert(key, (fp, b));
        }
        Ok(self.w[&key].1.ptr)
    }

    /// A frozen matrix as the A operand of a `mode` product: its bf16 copy
    /// under `Gemm::Bf16` (rounded on the host once, kept like `weight`),
    /// else the f32 one.
    fn wm(&mut self, w: &[f32], mode: Gemm) -> Res<u64> {
        if mode != Gemm::Bf16 {
            return self.weight(w);
        }
        let fp = crate::gpu::fp_f32(w);
        let key = (w.as_ptr() as usize, w.len());
        if !self.wbf.get(&key).is_some_and(|(f, _)| *f == fp) {
            if let Some((_, old)) = self.wbf.remove(&key) {
                self.w_bytes -= old.bytes;
            }
            if self.w_bytes + w.len() * 2 > w_cap() {
                self.w.clear();
                self.wbf.clear();
                self.w_bytes = 0;
            }
            let h: Vec<u16> = w.iter().map(|&v| bf16_bits(v)).collect();
            let b = Buf::from_slice(&h)?;
            self.w_bytes += b.bytes;
            self.wbf.insert(key, (fp, b));
        }
        Ok(self.wbf[&key].1.ptr)
    }
}

/// f32 → bf16 bits, rounded to nearest even (as `bk_to_bf16`).
fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

// SAFETY: the functions are used under the context's state lock.
unsafe impl Send for BkState {}

static STATE: Mutex<Option<BkState>> = Mutex::new(None);

/// How many bytes of frozen f32 matrices the bake keeps on the card.
fn w_cap() -> usize {
    std::env::var("CMF_BAKE_W_GB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(24) << 30
}

/// The products' precision: plain f32 in the bake's strict phase (the mask's
/// gradient signal; cuBLAS SGEMM costs little here), TF32 otherwise.
fn gemm_mode() -> Gemm {
    if crate::gpu_wgpu::bake_f32_strict() {
        Gemm::F32
    } else {
        Gemm::Tf32
    }
}

/// The resident step's weight products (projections, FFN, their
/// transposes): `CMF_BAKE_RES_BF16=1` gives them bf16 operands with f32
/// accumulation — the precision torch's bf16 autocast trains a LoRA in, at
/// twice TF32's tensor rate on a GeForce card. The attention products, the
/// head and the GDN gates keep `gemm_mode()` / f32. Default: `gemm_mode()`.
fn res_mode() -> Gemm {
    static BF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *BF.get_or_init(|| std::env::var("CMF_BAKE_RES_BF16").as_deref() == Ok("1")) {
        Gemm::Bf16
    } else {
        gemm_mode()
    }
}

/// `x` (`cnt` f32 values on the card) as the B operand of a `mode` product:
/// converted into scratch `name` under `Gemm::Bf16`, else `x` itself.
unsafe fn bop(c: &Ctx, s: &mut BkState, f: &BkFuncs, x: u64, cnt: usize, name: &'static str, mode: Gemm) -> Res<u64> {
    if mode != Gemm::Bf16 {
        return Ok(x);
    }
    let y = s.scr(name, cnt * 2)?;
    let n = cnt as i64;
    unsafe { c.launch(f.to_bf16, ((cnt as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(x, y, n))? };
    Ok(y)
}

/// The geometry and frozen constants of one GDN layer.
pub struct BakeGdn<'a> {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub eps: f32,
    pub conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
}

impl BakeGdn<'_> {
    fn c_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
    fn fits(&self) -> bool {
        self.dv % 32 == 0
            && (32..=256).contains(&self.dv)
            && (1..=256).contains(&self.dk)
            && self.nk > 0
            && self.nv % self.nk == 0
            && self.kk >= 1
            && self.conv.len() == self.c_dim() * self.kk
            && self.a_log.len() == self.nv
            && self.dt_bias.len() == self.nv
            && self.norm.len() == self.dv
    }
}

fn funcs(c: &Ctx, s: &mut BkState, dk: usize, dv: usize) -> Res<BkFuncs> {
    s.f.entry((dk, dv))
        .or_insert_with(|| {
            let m = compile(BK_SRC, vec![format!("-DDK={dk}"), format!("-DDV={dv}")], c.arch)?;
            Ok(BkFuncs {
                conv_fwd: func(m, "bk_conv_fwd")?,
                conv_bwd: func(m, "bk_conv_bwd")?,
                gdn_fwd: func(m, "bk_gdn_fwd")?,
                gdn_bwd: func(m, "bk_gdn_bwd")?,
                ffn_act: func(m, "bk_ffn_act")?,
                ffn_act_bwd: func(m, "bk_ffn_act_bwd")?,
                attn_perm: func(m, "bk_attn_perm")?,
                attn_unperm: func(m, "bk_attn_unperm")?,
                softmax_fwd: func(m, "bk_softmax_fwd")?,
                softmax_bwd: func(m, "bk_softmax_bwd")?,
                ce_rows: func(m, "bk_ce_rows")?,
                embed: func(m, "bk_embed")?,
                rmsnorm: func(m, "bk_rmsnorm")?,
                rmsnorm_bwd: func(m, "bk_rmsnorm_bwd")?,
                add: func(m, "bk_add")?,
                attn_prep: func(m, "bk_attn_prep")?,
                attn_prep_bwd: func(m, "bk_attn_prep_bwd")?,
                gate_mul: func(m, "bk_gate_mul")?,
                gate_bwd: func(m, "bk_gate_bwd")?,
                gather: func(m, "bk_gather")?,
                scatter_add: func(m, "bk_scatter_add")?,
                to_bf16: func(m, "bk_to_bf16")?,
                col_dot_part: func(m, "bk_col_dot_part")?,
                col_dot_sum: func(m, "bk_col_dot_sum")?,
                scale_rows: func(m, "bk_scale_rows")?,
                scale_rows_bf16: func(m, "bk_scale_rows_bf16")?,
            })
        })
        .clone()
}

/// Is the bake's GDN on the card here (CUDA up, the layer's geometry one the
/// kernels take)? `CMF_BAKE_CUDA_GDN=0` keeps the host operator.
pub fn bake_gdn_ok(g: &BakeGdn) -> bool {
    std::env::var("CMF_BAKE_CUDA_GDN").as_deref() != Ok("0") && g.fits() && ctx().is_some()
}

fn dtoh(c: &Ctx, ptr: u64, n: usize) -> Res<Vec<f32>> {
    let mut v = vec![0f32; n];
    unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), ptr, c.stream).map_err(derr)? };
    Ok(v)
}

/// Sequences per BPTT launch: the state history is `(t + 1)·dk·dv` floats a
/// (sequence, v-head); `CMF_BAKE_GDN_HIST_MB` (default 2048) caps a launch's.
fn seqs_per_launch(g: &BakeGdn, t: usize, b: usize) -> usize {
    let cap = std::env::var("CMF_BAKE_GDN_HIST_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048) << 20;
    let per = (t + 1) * g.dk * g.dv * 4 * g.nv;
    (cap / per.max(1)).clamp(1, b)
}

/// GDN forward of `b` sequences of `t` positions: the raw projections in,
/// the gated-norm output `[b·t, nv·dv]` out. None: the host runs it.
pub fn bake_gdn_fwd(qkv: &[f32], z: &[f32], a: &[f32], bstr: &[f32], b: usize, t: usize, g: &BakeGdn) -> Option<Vec<f32>> {
    let c = ctx()?;
    let (n, cdim, vd) = (b * t, g.c_dim(), g.nv * g.dv);
    if c.current().is_err() || qkv.len() != n * cdim || z.len() != n * vd || a.len() != n * g.nv || bstr.len() != n * g.nv {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    let s = s.get_or_insert_with(BkState::default);
    let mut run = || -> Res<Vec<f32>> {
        let f = funcs(c, s, g.dk, g.dv)?;
        let (rq, zz, aa, bb) = (Buf::from_slice(qkv)?, Buf::from_slice(z)?, Buf::from_slice(a)?, Buf::from_slice(bstr)?);
        let (taps, alog, dtb, nw) = (Buf::from_slice(g.conv)?, Buf::from_slice(g.a_log)?, Buf::from_slice(g.dt_bias)?, Buf::from_slice(g.norm)?);
        let pre = Buf::new(n * cdim * 4)?;
        let cq = Buf::new(n * cdim * 4)?;
        let out = Buf::new(n * vd * 4)?;
        let (n_i, t_i, cd_i, kk_i, nv_i, nk_i) = (n as i32, t as i32, cdim as i32, g.kk as i32, g.nv as i32, g.nk as i32);
        let eps = g.eps;
        unsafe {
            let tot = (n * cdim) as u64;
            c.launch(f.conv_fwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(rq.ptr, taps.ptr, pre.ptr, cq.ptr, n_i, t_i, cd_i, kk_i))?;
            let seq0 = 0i32;
            c.launch(
                f.gdn_fwd,
                (g.nv as u32, b as u32),
                g.dv as u32,
                0,
                kargs!(cq.ptr, zz.ptr, aa.ptr, bb.ptr, alog.ptr, dtb.ptr, nw.ptr, out.ptr, t_i, nv_i, nk_i, eps, seq0),
            )?;
        }
        let o = dtoh(c, out.ptr, n * vd)?;
        c.sync()?;
        Ok(o)
    };
    match run() {
        Ok(o) => Some(o),
        Err(e) => {
            let _ = c.sync();
            decline("bake gdn forward", &e);
            None
        }
    }
}

/// GDN backward (through-grads, frozen weights): (dqkv, dz, da, db). None:
/// the host runs it.
#[allow(clippy::type_complexity)]
pub fn bake_gdn_bwd(
    qkv: &[f32],
    z: &[f32],
    a: &[f32],
    bstr: &[f32],
    dout: &[f32],
    b: usize,
    t: usize,
    g: &BakeGdn,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let c = ctx()?;
    let (n, cdim, vd) = (b * t, g.c_dim(), g.nv * g.dv);
    if c.current().is_err() || qkv.len() != n * cdim || z.len() != n * vd || a.len() != n * g.nv || bstr.len() != n * g.nv || dout.len() != n * vd {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut s = STATE.lock().unwrap();
    let s = s.get_or_insert_with(BkState::default);
    let mut run = || -> Res<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let f = funcs(c, s, g.dk, g.dv)?;
        let (rq, zz, aa, bb, dd) = (Buf::from_slice(qkv)?, Buf::from_slice(z)?, Buf::from_slice(a)?, Buf::from_slice(bstr)?, Buf::from_slice(dout)?);
        let (taps, alog, dtb, nw) = (Buf::from_slice(g.conv)?, Buf::from_slice(g.a_log)?, Buf::from_slice(g.dt_bias)?, Buf::from_slice(g.norm)?);
        let pre = Buf::new(n * cdim * 4)?;
        let cq = Buf::new(n * cdim * 4)?;
        let dcq = Buf::new(n * cdim * 4)?;
        let draw = Buf::new(n * cdim * 4)?;
        let dz = Buf::new(n * vd * 4)?;
        let da = Buf::new(n * g.nv * 4)?;
        let db = Buf::new(n * g.nv * 4)?;
        let per = seqs_per_launch(g, t, b);
        let hist = Buf::new(per * g.nv * (t + 1) * g.dk * g.dv * 4)?;
        let kvh = Buf::new(per * g.nv * t * g.dv * 4)?;
        let oh = Buf::new(per * g.nv * t * g.dv * 4)?;
        let (n_i, t_i, cd_i, kk_i, nv_i, nk_i) = (n as i32, t as i32, cdim as i32, g.kk as i32, g.nv as i32, g.nk as i32);
        let eps = g.eps;
        unsafe {
            cu::memset_d8_async(dcq.ptr, 0, n * cdim * 4, c.stream).map_err(derr)?;
            let tot = (n * cdim) as u64;
            c.launch(f.conv_fwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(rq.ptr, taps.ptr, pre.ptr, cq.ptr, n_i, t_i, cd_i, kk_i))?;
            let mut s0 = 0usize;
            while s0 < b {
                let nb = per.min(b - s0);
                let seq0 = s0 as i32;
                c.launch(
                    f.gdn_bwd,
                    (g.nv as u32, nb as u32),
                    g.dv as u32,
                    0,
                    kargs!(
                        cq.ptr, zz.ptr, aa.ptr, bb.ptr, alog.ptr, dtb.ptr, nw.ptr, dd.ptr, dcq.ptr, dz.ptr, da.ptr, db.ptr, hist.ptr, kvh.ptr,
                        oh.ptr, t_i, nv_i, nk_i, eps, seq0
                    ),
                )?;
                s0 += nb;
            }
            c.launch(f.conv_bwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(pre.ptr, dcq.ptr, taps.ptr, draw.ptr, n_i, t_i, cd_i, kk_i))?;
        }
        let r = (dtoh(c, draw.ptr, n * cdim)?, dtoh(c, dz.ptr, n * vd)?, dtoh(c, da.ptr, n * g.nv)?, dtoh(c, db.ptr, n * g.nv)?);
        c.sync()?;
        Ok(r)
    };
    match run() {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("bake gdn backward", &e);
            None
        }
    }
}

/// `dX[n,k] += dY[n,m] · W[m,k]` on cuBLAS (TF32, as `h3_gemm_nt`), the
/// backward twin of `gemm_nt` for a bake on this backend. False: the host.
pub fn bake_gemm_dx(dy: &[f32], w: &[f32], dx: &mut [f32], n: usize, k: usize, m: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || dy.len() < n * m || w.len() < m * k || dx.len() < n * k {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut run = || -> Res<()> {
        let wp = s.weight(&w[..m * k])?;
        let db = s.up(c, "gx", &dy[..n * m])?;
        let xb = s.up(c, "gy", &dx[..n * k])?;
        // column-major: dXᵀ[k×n] += Wᵀ[k×m] · dYᵀ[m×n]
        unsafe {
            c.gemm(false, false, k, n, m, 1.0, wp, k, db, m, 1.0, xb, k, gemm_mode())?;
            cu::memcpy_dtoh_async(&mut dx[..n * k], xb, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake gemm dx", &e);
            false
        }
    }
}

/// The bake's frozen gated FFN as one chain on the card (the CUDA twin of
/// `gpu_wgpu::ffn_chain_f32`): both = n2·guᵀ (gate | up), act =
/// act(g)·u·scale, ffn = act·downᵀ, cuBLAS TF32 for the products. `both_out`
/// gets the plane on the host; `plane` keeps it on the card for that layer's
/// `bake_ffn_bwd`. False: the caller's host path.
#[allow(clippy::too_many_arguments)]
pub fn bake_ffn_fwd(
    n2: &[f32],
    gu: &[f32],
    down: &[f32],
    scale: Option<&[f32]>,
    both_out: Option<&mut [f32]>,
    plane: Option<usize>,
    ffn: &mut [f32],
    n: usize,
    hsz: usize,
    inter: usize,
    act: crate::fcd_ops::FfnAct,
) -> bool {
    let Some(c) = ctx() else { return false };
    if std::env::var("CMF_BAKE_GPU").as_deref() == Ok("0")
        || c.current().is_err()
        || n2.len() < n * hsz
        || gu.len() < 2 * inter * hsz
        || down.len() < hsz * inter
        || ffn.len() < n * hsz
        || scale.is_some_and(|s| s.len() < inter)
        || both_out.as_ref().is_some_and(|b| b.len() < n * 2 * inter)
    {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut both_out = both_out;
    let mut run = || -> Res<()> {
        let f = funcs(c, s, 128, 128)?;
        let wgu = s.weight(&gu[..2 * inter * hsz])?;
        let wdn = s.weight(&down[..hsz * inter])?;
        let x = s.up(c, "x", &n2[..n * hsz])?;
        // a plane kept for the backward is its own allocation
        let kept = match plane {
            Some(_) => Some(Buf::new(n * 2 * inter * 4)?),
            None => None,
        };
        let both = match &kept {
            Some(b) => b.ptr,
            None => s.scr("both", n * 2 * inter * 4)?,
        };
        let actb = s.scr("act", n * inter * 4)?;
        let y = s.scr("y", n * hsz * 4)?;
        let sc = match scale {
            Some(sv) => s.up(c, "scale", &sv[..inter])?,
            None => s.scr("scale", 4)?,
        };
        let (n_i, in_i, gelu, has) = (n as i32, inter as i32, (act == crate::fcd_ops::FfnAct::GeluErf) as i32, scale.is_some() as i32);
        unsafe {
            c.gemm(true, false, 2 * inter, n, hsz, 1.0, wgu, hsz, x, hsz, 0.0, both, 2 * inter, gemm_mode())?;
            let tot = (n * inter) as u64;
            c.launch(f.ffn_act, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(both, sc, actb, n_i, in_i, gelu, has))?;
            c.gemm(true, false, hsz, n, inter, 1.0, wdn, inter, actb, inter, 0.0, y, hsz, gemm_mode())?;
            cu::memcpy_dtoh_async(&mut ffn[..n * hsz], y, c.stream).map_err(derr)?;
            if let Some(b) = both_out.as_deref_mut() {
                cu::memcpy_dtoh_async(&mut b[..n * 2 * inter], both, c.stream).map_err(derr)?;
            }
        }
        c.sync()?;
        if let (Some(li), Some(b)) = (plane, kept) {
            s.planes.insert(li, (n * 2 * inter, b));
        }
        Ok(())
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake ffn chain", &e);
            false
        }
    }
}

/// The backward of `bake_ffn_fwd` from the plane its training forward kept
/// for layer `li`: dact2 = dh2·down, dgu (and the mask gradient Σ dact2·act
/// into `dmask`, overwritten), dn2 += dgu·gu. False (the plane absent or the
/// card refusing): the caller's host path.
#[allow(clippy::too_many_arguments)]
pub fn bake_ffn_bwd(
    dh2: &[f32],
    down: &[f32],
    gu: &[f32],
    li: usize,
    dn2: &mut [f32],
    n: usize,
    hsz: usize,
    inter: usize,
    act: crate::fcd_ops::FfnAct,
    scale: Option<&[f32]>,
    dmask: Option<&mut [f32]>,
) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || dh2.len() < n * hsz || dn2.len() < n * hsz || scale.is_some_and(|s| s.len() < inter) || dmask.as_ref().is_some_and(|d| d.len() < inter) {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    if !s.planes.get(&li).is_some_and(|(len, _)| *len == n * 2 * inter) {
        return false;
    }
    let mut dmask = dmask;
    let mut run = || -> Res<()> {
        let f = funcs(c, s, 128, 128)?;
        let wgu = s.weight(&gu[..2 * inter * hsz])?;
        let wdn = s.weight(&down[..hsz * inter])?;
        let both = s.planes[&li].1.ptr;
        let dy = s.up(c, "x", &dh2[..n * hsz])?;
        let dact2 = s.scr("act", n * inter * 4)?;
        let dgu = s.scr("both", n * 2 * inter * 4)?;
        let dm = s.scr("dm", inter * 4)?;
        let dx = s.up(c, "y", &dn2[..n * hsz])?;
        let sc = match scale {
            Some(sv) => s.up(c, "scale", &sv[..inter])?,
            None => s.scr("scale", 4)?,
        };
        let rows = 64usize;
        let (n_i, in_i, gelu, has, rows_i) = (n as i32, inter as i32, (act == crate::fcd_ops::FfnAct::GeluErf) as i32, scale.is_some() as i32, rows as i32);
        unsafe {
            cu::memset_d8_async(dm, 0, inter * 4, c.stream).map_err(derr)?;
            // dact2[n×inter] = dh2[n×hsz] · down[hsz×inter]
            c.gemm(false, false, inter, n, hsz, 1.0, wdn, inter, dy, hsz, 0.0, dact2, inter, gemm_mode())?;
            c.launch(f.ffn_act_bwd, (inter.div_ceil(256) as u32, n.div_ceil(rows) as u32), 256, 0, kargs!(both, dact2, sc, dgu, dm, n_i, in_i, gelu, has, rows_i))?;
            // dn2[n×hsz] += dgu[n×2·inter] · gu[2·inter×hsz]
            c.gemm(false, false, hsz, n, 2 * inter, 1.0, wgu, hsz, dgu, 2 * inter, 1.0, dx, hsz, gemm_mode())?;
            cu::memcpy_dtoh_async(&mut dn2[..n * hsz], dx, c.stream).map_err(derr)?;
            if let Some(d) = dmask.as_deref_mut() {
                cu::memcpy_dtoh_async(&mut d[..inter], dm, c.stream).map_err(derr)?;
            }
        }
        c.sync()
    };
    let r = run();
    s.planes.remove(&li);
    match r {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake ffn backward", &e);
            false
        }
    }
}

/// Drop every FFN plane a training forward left on the card (a step whose
/// backward did not consume them).
pub fn bake_planes_clear() {
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.planes.clear();
    }
}

/// `y[n,m] = x[n,k] · w[m,k]ᵀ` for the bake on cuBLAS, the frozen matrix kept
/// on the card, in `gemm_mode` precision. False: the caller's next arm.
pub fn bake_gemm_nt(x: &[f32], w: &[f32], y: &mut [f32], n: usize, k: usize, m: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || x.len() < n * k || w.len() < m * k || y.len() < n * m {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut run = || -> Res<()> {
        let wp = s.weight(&w[..m * k])?;
        let xb = s.up(c, "gx", &x[..n * k])?;
        let yb = s.scr("gy", n * m * 4)?;
        unsafe {
            c.gemm(true, false, m, n, k, 1.0, wp, k, xb, k, 0.0, yb, m, gemm_mode())?;
            cu::memcpy_dtoh_async(&mut y[..n * m], yb, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake gemm nt", &e);
            false
        }
    }
}

/// Is the bake's exact attention core on the card for this geometry?
fn attn_fits(b: usize, t: usize, nh: usize, nkv: usize, hd: usize) -> bool {
    b > 0 && t > 0 && t <= 4096 && nkv > 0 && nh % nkv == 0 && (1..=512).contains(&hd) && b * nh * t * t <= 1 << 30
}

/// The permuted planes of one attention call: q, k, v as [b·nh][t][hd] (k/v
/// repeated over their group's q-heads).
unsafe fn attn_planes(c: &Ctx, f: &BkFuncs, qrot: &[f32], krot: &[f32], vproj: &[f32], b: usize, t: usize, nh: usize, nkv: usize, hd: usize) -> Res<(Buf, Buf, Buf)> {
    let (n, rep) = (b * t, nh / nkv);
    let per = b * nh * t * hd;
    let (q, k, v) = (Buf::from_slice(&qrot[..n * nh * hd])?, Buf::from_slice(&krot[..n * nkv * hd])?, Buf::from_slice(&vproj[..n * nkv * hd])?);
    let (qp, kp, vp) = (Buf::new(per * 4)?, Buf::new(per * 4)?, Buf::new(per * 4)?);
    let (b_i, t_i, nh_i, nkv_i, hd_i, one, rep_i) = (b as i32, t as i32, nh as i32, nkv as i32, hd as i32, 1i32, rep as i32);
    let grid = ((per as u64).div_ceil(256) as u32, 1);
    unsafe {
        c.launch(f.attn_perm, grid, 256, 0, kargs!(q.ptr, qp.ptr, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(k.ptr, kp.ptr, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(v.ptr, vp.ptr, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
    }
    Ok((qp, kp, vp))
}

/// P = softmax(QKᵀ/√hd) over the causal (windowed) span, and Pᵀ.
unsafe fn attn_probs(c: &Ctx, f: &BkFuncs, qp: &Buf, kp: &Buf, b: usize, t: usize, nh: usize, hd: usize, window: Option<usize>) -> Res<(Buf, Buf)> {
    let cnt = b * nh;
    let s = Buf::new(cnt * t * t * 4)?;
    let p = Buf::new(cnt * t * t * 4)?;
    let pt = Buf::new(cnt * t * t * 4)?;
    let scale = 1.0 / (hd as f32).sqrt();
    let (t_i, w_i) = (t as i32, window.map_or(0, |w| w as i32));
    unsafe {
        // S[p][q] = Σ_c K[q][c]·Q[p][c] (column-major C = Kᵀᵀ·Qᵀ)
        c.gemm_batched(true, t, t, hd, scale, kp.ptr, hd, t * hd, qp.ptr, hd, t * hd, s.ptr, t, t * t, cnt, gemm_mode())?;
        c.launch(f.softmax_fwd, ((cnt * t) as u32, 1), 128, 0, kargs!(s.ptr, p.ptr, pt.ptr, t_i, w_i))?;
    }
    Ok((p, pt))
}

/// The bake's exact attention core on the card: `ao[n, nh·hd]` from the
/// rotated q, k and the v projection (`fcd_ops::attn_head_fwd_window` per
/// (sequence, head)). False: the host loop.
#[allow(clippy::too_many_arguments)]
pub fn bake_attn_fwd(qrot: &[f32], krot: &[f32], vproj: &[f32], b: usize, t: usize, nh: usize, nkv: usize, hd: usize, window: Option<usize>, ao: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let n = b * t;
    if c.current().is_err() || !attn_fits(b, t, nh, nkv, hd) || qrot.len() < n * nh * hd || krot.len() < n * nkv * hd || vproj.len() < n * nkv * hd || ao.len() < n * nh * hd {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut run = || -> Res<()> {
        let f = funcs(c, s, 128, 128)?;
        let cnt = b * nh;
        unsafe {
            let (qp, kp, vp) = attn_planes(c, &f, qrot, krot, vproj, b, t, nh, nkv, hd)?;
            let (p, _pt) = attn_probs(c, &f, &qp, &kp, b, t, nh, hd, window)?;
            let op = Buf::new(cnt * t * hd * 4)?;
            // O = P·V (column-major Oᵀ = Vᵀ·Pᵀ)
            c.gemm_batched(false, hd, t, t, 1.0, vp.ptr, hd, t * hd, p.ptr, t, t * t, op.ptr, hd, t * hd, cnt, gemm_mode())?;
            let out = Buf::new(n * nh * hd * 4)?;
            let (b_i, t_i, nh_i, hd_i, one) = (b as i32, t as i32, nh as i32, hd as i32, 1i32);
            c.launch(f.attn_unperm, (((n * nh * hd) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(op.ptr, out.ptr, b_i, t_i, nh_i, nh_i, hd_i, one))?;
            cu::memcpy_dtoh_async(&mut ao[..n * nh * hd], out.ptr, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake attention", &e);
            false
        }
    }
}

/// Its backward: dq [n, nh·hd], dk, dv [n, nkv·hd] (overwritten) from the same
/// inputs and dao (`fcd_ops::attn_head_bwd_window`, probabilities recomputed).
#[allow(clippy::too_many_arguments)]
pub fn bake_attn_bwd(
    qrot: &[f32],
    krot: &[f32],
    vproj: &[f32],
    dao: &[f32],
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    window: Option<usize>,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let n = b * t;
    if c.current().is_err()
        || !attn_fits(b, t, nh, nkv, hd)
        || qrot.len() < n * nh * hd
        || krot.len() < n * nkv * hd
        || vproj.len() < n * nkv * hd
        || dao.len() < n * nh * hd
        || dq.len() < n * nh * hd
        || dk.len() < n * nkv * hd
        || dv.len() < n * nkv * hd
    {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut run = || -> Res<()> {
        let f = funcs(c, s, 128, 128)?;
        let (cnt, rep) = (b * nh, nh / nkv);
        let scale = 1.0 / (hd as f32).sqrt();
        unsafe {
            let (qp, kp, vp) = attn_planes(c, &f, qrot, krot, vproj, b, t, nh, nkv, hd)?;
            let (p, pt) = attn_probs(c, &f, &qp, &kp, b, t, nh, hd, window)?;
            let (dop, _, _) = attn_planes(c, &f, dao, krot, vproj, b, t, nh, nkv, hd)?;
            let dp = Buf::new(cnt * t * t * 4)?;
            let ds = Buf::new(cnt * t * t * 4)?;
            let dst = Buf::new(cnt * t * t * 4)?;
            // dP[p][q] = Σ_c V[q][c]·dO[p][c]
            c.gemm_batched(true, t, t, hd, 1.0, vp.ptr, hd, t * hd, dop.ptr, hd, t * hd, dp.ptr, t, t * t, cnt, gemm_mode())?;
            let t_i = t as i32;
            c.launch(f.softmax_bwd, ((cnt * t) as u32, 1), 128, 0, kargs!(p.ptr, dp.ptr, ds.ptr, dst.ptr, t_i))?;
            let (dqp, dkp, dvp) = (Buf::new(cnt * t * hd * 4)?, Buf::new(cnt * t * hd * 4)?, Buf::new(cnt * t * hd * 4)?);
            // dQ = dS·K/√hd; dK = dSᵀ·Q/√hd; dV = Pᵀ·dO
            c.gemm_batched(false, hd, t, t, scale, kp.ptr, hd, t * hd, ds.ptr, t, t * t, dqp.ptr, hd, t * hd, cnt, gemm_mode())?;
            c.gemm_batched(false, hd, t, t, scale, qp.ptr, hd, t * hd, dst.ptr, t, t * t, dkp.ptr, hd, t * hd, cnt, gemm_mode())?;
            c.gemm_batched(false, hd, t, t, 1.0, dop.ptr, hd, t * hd, pt.ptr, t, t * t, dvp.ptr, hd, t * hd, cnt, gemm_mode())?;
            let (oq, ok, ov) = (Buf::new(n * nh * hd * 4)?, Buf::new(n * nkv * hd * 4)?, Buf::new(n * nkv * hd * 4)?);
            let (b_i, nh_i, nkv_i, hd_i, one, rep_i) = (b as i32, nh as i32, nkv as i32, hd as i32, 1i32, rep as i32);
            c.launch(f.attn_unperm, (((n * nh * hd) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(dqp.ptr, oq.ptr, b_i, t_i, nh_i, nh_i, hd_i, one))?;
            let kvg = (((n * nkv * hd) as u64).div_ceil(256) as u32, 1);
            c.launch(f.attn_unperm, kvg, 256, 0, kargs!(dkp.ptr, ok.ptr, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
            c.launch(f.attn_unperm, kvg, 256, 0, kargs!(dvp.ptr, ov.ptr, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
            cu::memcpy_dtoh_async(&mut dq[..n * nh * hd], oq.ptr, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(&mut dk[..n * nkv * hd], ok.ptr, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(&mut dv[..n * nkv * hd], ov.ptr, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("bake attention backward", &e);
            false
        }
    }
}

/// The bake's LM-head cross-entropy over `rows` scored hidden rows (`hsrc`
/// [rows × hsz], final-normed): the logits on cuBLAS a band of rows at a time,
/// each row's logsumexp, argmax and (with `grad`) softmax − onehot scaled by
/// `scale`, then dsrc = dlogits · lm. Returns (nll per row, argmax per row,
/// dsrc [rows × hsz] — empty without `grad`). None: the host loop.
pub fn bake_lm_loss(hsrc: &[f32], lm: &[f32], targets: &[u32], hsz: usize, scale: f32, grad: bool) -> Option<(Vec<f32>, Vec<u32>, Vec<f32>)> {
    let c = ctx()?;
    let rows = targets.len();
    if c.current().is_err() || hsz == 0 || lm.len() % hsz != 0 || hsrc.len() < rows * hsz || rows == 0 {
        return None;
    }
    let vocab = lm.len() / hsz;
    if targets.iter().any(|&t| t as usize >= vocab) {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(BkState::default);
    let mut run = || -> Res<(Vec<f32>, Vec<u32>, Vec<f32>)> {
        let f = funcs(c, s, 128, 128)?;
        let wl = s.weight(lm)?;
        let band = ((512usize << 20) / (vocab * 4)).clamp(1, rows);
        let x = s.up(c, "x", &hsrc[..rows * hsz])?;
        let tg: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let tgb = Buf::from_slice(&tg)?;
        let logits = s.scr("logits", band * vocab * 4)?;
        let nll = s.scr("nll", rows * 4)?;
        let arg = s.scr("arg", rows * 4)?;
        let dx = s.scr("y", if grad { rows * hsz * 4 } else { 4 })?;
        let (v_i, gr) = (vocab as i32, grad as i32);
        let mut r0 = 0usize;
        while r0 < rows {
            let nb = band.min(rows - r0);
            unsafe {
                let xp = x + (r0 * hsz * 4) as u64;
                // logits[nb × vocab] = x·lmᵀ
                c.gemm(true, false, vocab, nb, hsz, 1.0, wl, hsz, xp, hsz, 0.0, logits, vocab, gemm_mode())?;
                let (tp, np, ap) = (tgb.ptr + (r0 * 4) as u64, nll + (r0 * 4) as u64, arg + (r0 * 4) as u64);
                c.launch(f.ce_rows, (nb as u32, 1), 1024, 0, kargs!(logits, tp, np, ap, v_i, scale, gr))?;
                if grad {
                    // dx[nb × hsz] = dlogits · lm
                    let dp = dx + (r0 * hsz * 4) as u64;
                    c.gemm(false, false, hsz, nb, vocab, 1.0, wl, hsz, logits, vocab, 0.0, dp, hsz, gemm_mode())?;
                }
            }
            r0 += nb;
        }
        let nl = dtoh(c, nll, rows)?;
        let mut ai = vec![0i32; rows];
        unsafe { cu::memcpy_dtoh_async(ai.as_mut_slice(), arg, c.stream).map_err(derr)? };
        let d = if grad { dtoh(c, dx, rows * hsz)? } else { Vec::new() };
        c.sync()?;
        Ok((nl, ai.into_iter().map(|a| a as u32).collect(), d))
    };
    match run() {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("bake lm loss", &e);
            None
        }
    }
}

/// A GDN layer's frozen projections (f32 replica): qkv [c_dim × hsz], z, a,
/// b [· × hsz], out [hsz × nv·dv].
pub struct BakeGdnW<'a> {
    pub wqkv: &'a [f32],
    pub wz: &'a [f32],
    pub wa: &'a [f32],
    pub wb: &'a [f32],
    pub wout: &'a [f32],
    pub hsz: usize,
}

/// A whole GDN layer forward on the card from its normed input `n1`
/// [n × hsz]: the four projections, conv + recurrence, out_proj — only n1 up
/// and attn_out down (the projection streams too when `want_streams`, for a
/// backward through the layer). None: the host path.
#[allow(clippy::type_complexity)]
pub fn bake_gdn_layer_fwd(n1: &[f32], w: &BakeGdnW, g: &BakeGdn, b: usize, t: usize, want_streams: bool) -> Option<(Vec<f32>, Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>)> {
    let c = ctx()?;
    let (n, cdim, vd, hsz, nv) = (b * t, g.c_dim(), g.nv * g.dv, w.hsz, g.nv);
    if c.current().is_err()
        || n1.len() < n * hsz
        || w.wqkv.len() != cdim * hsz
        || w.wz.len() != vd * hsz
        || w.wa.len() != nv * hsz
        || w.wb.len() != nv * hsz
        || w.wout.len() != hsz * vd
    {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut gs = STATE.lock().unwrap();
    let s = gs.get_or_insert_with(BkState::default);
    let mut run = || -> Res<(Vec<f32>, Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>)> {
        let f = funcs(c, s, g.dk, g.dv)?;
        let (wq, wz, wa, wb, wo) = (s.weight(w.wqkv)?, s.weight(w.wz)?, s.weight(w.wa)?, s.weight(w.wb)?, s.weight(w.wout)?);
        let x = s.up(c, "x", &n1[..n * hsz])?;
        let (qkv, z, a, bb) = (s.scr("qkv", n * cdim * 4)?, s.scr("z", n * vd * 4)?, s.scr("a", n * nv * 4)?, s.scr("b", n * nv * 4)?);
        let (taps, alog, dtb, nw) = (s.up(c, "taps", g.conv)?, s.up(c, "alog", g.a_log)?, s.up(c, "dtb", g.dt_bias)?, s.up(c, "nw", g.norm)?);
        let pre = s.scr("pre", n * cdim * 4)?;
        let cq = s.scr("cq", n * cdim * 4)?;
        let of = s.scr("of", n * vd * 4)?;
        let y = s.scr("y", n * hsz * 4)?;
        let (n_i, t_i, cd_i, kk_i, nv_i, nk_i) = (n as i32, t as i32, cdim as i32, g.kk as i32, nv as i32, g.nk as i32);
        let (eps, seq0) = (g.eps, 0i32);
        let mode = gemm_mode();
        unsafe {
            c.gemm(true, false, cdim, n, hsz, 1.0, wq, hsz, x, hsz, 0.0, qkv, cdim, mode)?;
            c.gemm(true, false, vd, n, hsz, 1.0, wz, hsz, x, hsz, 0.0, z, vd, mode)?;
            c.gemm(true, false, nv, n, hsz, 1.0, wa, hsz, x, hsz, 0.0, a, nv, Gemm::F32)?;
            c.gemm(true, false, nv, n, hsz, 1.0, wb, hsz, x, hsz, 0.0, bb, nv, Gemm::F32)?;
            let tot = (n * cdim) as u64;
            c.launch(f.conv_fwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(qkv, taps, pre, cq, n_i, t_i, cd_i, kk_i))?;
            c.launch(f.gdn_fwd, (nv as u32, b as u32), g.dv as u32, 0, kargs!(cq, z, a, bb, alog, dtb, nw, of, t_i, nv_i, nk_i, eps, seq0))?;
            c.gemm(true, false, hsz, n, vd, 1.0, wo, vd, of, vd, 0.0, y, hsz, mode)?;
        }
        let out = dtoh(c, y, n * hsz)?;
        let streams = if want_streams {
            Some((dtoh(c, qkv, n * cdim)?, dtoh(c, z, n * vd)?, dtoh(c, a, n * nv)?, dtoh(c, bb, n * nv)?))
        } else {
            None
        };
        c.sync()?;
        Ok((out, streams))
    };
    match run() {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("bake gdn layer", &e);
            None
        }
    }
}

/// A full-attention layer of the resident step (no bias, no projected head
/// gate: those keep the host path).
pub struct BkFull<'a> {
    pub wqkv: &'a [f32],
    pub wo: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub gated: bool,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    /// (cos, sin) per position and frequency, `[t][half][2]`.
    pub rope: &'a [f32],
    pub half: usize,
    pub window: Option<usize>,
}

pub enum BkAttn<'a> {
    Gdn(BakeGdn<'a>, BakeGdnW<'a>),
    Full(BkFull<'a>),
}

pub struct BkLayer<'a> {
    pub iln: &'a [f32],
    pub pln: &'a [f32],
    pub attn: BkAttn<'a>,
    /// `[gate; up]` [2·inter × hsz], down [hsz × inter].
    pub gu: &'a [f32],
    pub down: &'a [f32],
    pub inter: usize,
    /// The mask's gain σ(m) per neuron (the act's scale).
    pub gain: &'a [f32],
    /// `--mask-scope linear`: a gain σ(m) per output row of every projection
    /// of the layer, laid out as [`lin_layout`] gives; None: no output masks.
    pub lin: Option<&'a [f32]>,
}

/// Where each projection's output gains sit in a layer's `BkLayer::lin`
/// (and its gradient in `BkOut::dlin`): GDN `[qkv | z | a | b | out | gate |
/// up | down]`, full attention `[fused qkv | o | gate | up | down]` (z, a, b
/// empty). Offsets and lengths in floats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinOff {
    pub qkv: (usize, usize),
    pub z: (usize, usize),
    pub a: (usize, usize),
    pub b: (usize, usize),
    pub out: (usize, usize),
    pub gate: (usize, usize),
    pub up: (usize, usize),
    pub down: (usize, usize),
    pub len: usize,
}

/// The output-gain layout of a layer (see [`LinOff`]).
pub fn lin_layout(attn: &BkAttn, inter: usize, hsz: usize) -> LinOff {
    let (qkv, z, ab) = match attn {
        BkAttn::Gdn(g, _) => (g.c_dim(), g.nv * g.dv, g.nv),
        BkAttn::Full(a) => {
            let (qdim, kvdim) = (a.nh * a.hd, a.nkv * a.hd);
            ((if a.gated { 2 * qdim } else { qdim }) + 2 * kvdim, 0, 0)
        }
    };
    let mut at = 0usize;
    let mut take = |len: usize| {
        let r = (at, len);
        at += len;
        r
    };
    let (qkv, z, a, b, out, gate, up, down) = (take(qkv), take(z), take(ab), take(ab), take(hsz), take(inter), take(inter), take(hsz));
    LinOff { qkv, z, a, b, out, gate, up, down, len: at }
}

/// The `mode` copy of the frozen matrix `w` (`rows` × the rest) with output
/// row r scaled by the gain at `g + 4r` — the output mask in the weight:
/// `(diag(g) W) x` is the gained output, and its transpose carries the
/// gradient straight back to x.
unsafe fn wscaled(c: &Ctx, s: &mut BkState, f: &BkFuncs, w: &[f32], rows: usize, g: u64, mode: Gemm) -> Res<u64> {
    let cols = w.len() / rows;
    let src = s.wm(w, mode)?;
    let half = mode == Gemm::Bf16;
    let key = (w.as_ptr() as usize, w.len(), half);
    let bytes = w.len() * if half { 2 } else { 4 };
    if !s.wsc.get(&key).is_some_and(|b| b.bytes >= bytes) {
        s.wsc.insert(key, Buf::new(bytes)?);
    }
    let dst = s.wsc[&key].ptr;
    let cols_i = cols as i32;
    let grid = (cols.div_ceil(256) as u32, rows as u32);
    unsafe { c.launch(if half { f.scale_rows_bf16 } else { f.scale_rows }, grid, 256, 0, kargs!(src, g, dst, cols_i))? };
    Ok(dst)
}

/// `dg[j] += Σ_r y[r][j]·dy[r][j]` (÷ g[j] with `div`: y is the gained
/// output), in a fixed order.
unsafe fn col_dot(c: &Ctx, s: &mut BkState, f: &BkFuncs, y: u64, dy: u64, g: u64, dg: u64, n: usize, m: usize, div: bool) -> Res<()> {
    const ROWS: usize = 128;
    let chunks = n.div_ceil(ROWS);
    let part = s.scr("lindot", chunks * m * 4)?;
    let (n_i, m_i, rows_i, ch_i, div_i) = (n as i64, m as i32, ROWS as i32, chunks as i32, div as i32);
    let gx = m.div_ceil(256) as u32;
    unsafe {
        c.launch(f.col_dot_part, (gx, chunks as u32), 256, 0, kargs!(y, dy, part, n_i, m_i, rows_i))?;
        c.launch(f.col_dot_sum, (gx, 1), 256, 0, kargs!(part, g, dg, ch_i, m_i, div_i))
    }
}

/// One Phase-A step (or a held-out pass without `grad`) of the skill bake.
pub struct BkStep<'a> {
    pub layers: Vec<BkLayer<'a>>,
    pub ids: &'a [u32],
    pub b: usize,
    pub t: usize,
    pub hsz: usize,
    pub eps: f32,
    pub gemma: bool,
    pub act: crate::fcd_ops::FfnAct,
    pub embed: &'a [f32],
    pub final_norm: &'a [f32],
    pub lm: &'a [f32],
    /// Scored target rows (the token predicted from row − 1).
    pub targets: &'a [usize],
    /// The exact classifier (`--focus-tokens`): `lm` holds only the class
    /// rows, and this is each target's class (a row of `lm`); the
    /// cross-entropy and argmax are over the classes. None: the targets are
    /// the token ids and `lm` the whole head.
    pub classes: Option<&'a [u32]>,
    pub scale: f32,
    pub grad: bool,
    /// The backward also runs through every frozen attention (and GDN)
    /// branch (`BakeHyper::through_attn`); each layer's input is kept and its
    /// attention recomputed from it.
    pub through_attn: bool,
}

/// What a step returns: nll and argmax per scored target, the mask gradient
/// Σ dact·act per layer (empty without `grad`).
pub struct BkOut {
    pub nll: Vec<f32>,
    pub argmax: Vec<u32>,
    pub dmask: Vec<Vec<f32>>,
    /// With output masks and `grad`: the loss gradient per output gain of
    /// each layer, laid out as `BkLayer::lin` (empty otherwise).
    pub dlin: Vec<Vec<f32>>,
}

/// The full-attention layer forward on the card: `x` (normed rows) → `y`.
unsafe fn full_attn_dev(c: &Ctx, s: &mut BkState, f: &BkFuncs, a: &BkFull, x: u64, y: u64, b: usize, t: usize, hsz: usize, eps: f32, gemma: bool, gw: Option<(u64, u64)>) -> Res<()> {
    let (n, nh, nkv, hd) = (b * t, a.nh, a.nkv, a.hd);
    let (qdim, kvdim, rep) = (nh * hd, nkv * hd, nh / nkv);
    let qrows = if a.gated { 2 * qdim } else { qdim };
    let fused = qrows + 2 * kvdim;
    if a.wqkv.len() != fused * hsz || a.wo.len() != hsz * qdim || hd > 512 || a.rope.len() < t * a.half * 2 {
        return Err("full attention geometry".into());
    }
    let (mode, amode) = (res_mode(), gemm_mode());
    // with output masks the weights come row-scaled (`wscaled`)
    let (wq, wo) = match gw {
        Some(w) => w,
        None => (s.wm(a.wqkv, mode)?, s.wm(a.wo, mode)?),
    };
    let rope = s.up(c, "rope", a.rope)?;
    let (qn, kn) = (match a.q_norm { Some(w) => s.up(c, "qn", w)?, None => s.scr("qn", 4)? }, match a.k_norm { Some(w) => s.up(c, "kn", w)?, None => s.scr("kn", 4)? });
    let qkv = s.scr("qkv", n * fused * 4)?;
    let (qr, kr, vp, gt) = (s.scr("qrot", n * qdim * 4)?, s.scr("krot", n * kvdim * 4)?, s.scr("vproj", n * kvdim * 4)?, s.scr("gate", n * qdim * 4)?);
    let cnt = b * nh;
    let per = cnt * t * hd;
    let (qp, kp, vpp, op) = (s.scr("qp", per * 4)?, s.scr("kp", per * 4)?, s.scr("vp", per * 4)?, s.scr("op", per * 4)?);
    let (sm, pm, ptm) = (s.scr("S", cnt * t * t * 4)?, s.scr("P", cnt * t * t * 4)?, s.scr("PT", cnt * t * t * 4)?);
    let ao = s.scr("ao", n * qdim * 4)?;
    let (t_i, nh_i, nkv_i, hd_i, half_i, gated_i, gm_i, hq, hk) =
        (t as i32, nh as i32, nkv as i32, hd as i32, a.half as i32, a.gated as i32, gemma as i32, a.q_norm.is_some() as i32, a.k_norm.is_some() as i32);
    let (b_i, one, rep_i, w_i) = (b as i32, 1i32, rep as i32, a.window.map_or(0, |w| w as i32));
    let scale = 1.0 / (hd as f32).sqrt();
    unsafe {
        let xo = bop(c, s, f, x, n * hsz, "bfx", mode)?;
        c.gemm(true, false, fused, n, hsz, 1.0, wq, hsz, xo, hsz, 0.0, qkv, fused, mode)?;
        c.launch(f.attn_prep, (n as u32, (nh + nkv) as u32), hd.min(512) as u32, 0, kargs!(qkv, qn, kn, rope, qr, kr, vp, gt, t_i, nh_i, nkv_i, hd_i, half_i, gated_i, eps, gm_i, hq, hk))?;
        let grid = ((per as u64).div_ceil(256) as u32, 1);
        c.launch(f.attn_perm, grid, 256, 0, kargs!(qr, qp, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(kr, kp, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(vp, vpp, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.gemm_batched(true, t, t, hd, scale, kp, hd, t * hd, qp, hd, t * hd, sm, t, t * t, cnt, amode)?;
        c.launch(f.softmax_fwd, ((cnt * t) as u32, 1), 128, 0, kargs!(sm, pm, ptm, t_i, w_i))?;
        c.gemm_batched(false, hd, t, t, 1.0, vpp, hd, t * hd, pm, t, t * t, op, hd, t * hd, cnt, amode)?;
        c.launch(f.attn_unperm, (((n * qdim) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(op, ao, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        if a.gated {
            let tot = (n * qdim) as i64;
            c.launch(f.gate_mul, (((n * qdim) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(ao, gt, tot))?;
        }
        let aoo = bop(c, s, f, ao, n * qdim, "bfy", mode)?;
        c.gemm(true, false, hsz, n, qdim, 1.0, wo, qdim, aoo, qdim, 0.0, y, hsz, mode)?;
    }
    Ok(())
}

/// The full-attention branch's input gradient on the card: `dy` (at the
/// branch's output, [n × hsz]) → `dx` (at its normed input `x`), frozen
/// weights. The forward is recomputed from `x`: projections, q/k norm and
/// RoPE, P; then o_proj, the output gate, the core, RoPE and the q/k norm
/// backward, the projections' transpose.
#[allow(clippy::too_many_arguments)]
unsafe fn full_attn_bwd_dev(c: &Ctx, s: &mut BkState, f: &BkFuncs, a: &BkFull, x: u64, dy: u64, dx: u64, b: usize, t: usize, hsz: usize, eps: f32, gemma: bool, gq: Option<(u64, u64, u64, u64)>) -> Res<()> {
    let (n, nh, nkv, hd) = (b * t, a.nh, a.nkv, a.hd);
    let (qdim, kvdim, rep) = (nh * hd, nkv * hd, nh / nkv);
    let qrows = if a.gated { 2 * qdim } else { qdim };
    let fused = qrows + 2 * kvdim;
    if a.wqkv.len() != fused * hsz || a.wo.len() != hsz * qdim || hd > 512 || a.rope.len() < t * a.half * 2 {
        return Err("full attention geometry".into());
    }
    let (mode, amode) = (res_mode(), gemm_mode());
    // gq = (row-scaled wqkv, row-scaled wo, the qkv gains, their gradient)
    let (wq, wo) = match gq {
        Some((wq, wo, _, _)) => (wq, wo),
        None => (s.wm(a.wqkv, mode)?, s.wm(a.wo, mode)?),
    };
    let rope = s.up(c, "rope", a.rope)?;
    let (qn, kn) = (match a.q_norm { Some(w) => s.up(c, "qn", w)?, None => s.scr("qn", 4)? }, match a.k_norm { Some(w) => s.up(c, "kn", w)?, None => s.scr("kn", 4)? });
    let qkv = s.scr("qkv", n * fused * 4)?;
    let (qr, kr, vp, gt) = (s.scr("qrot", n * qdim * 4)?, s.scr("krot", n * kvdim * 4)?, s.scr("vproj", n * kvdim * 4)?, s.scr("gate", n * qdim * 4)?);
    let cnt = b * nh;
    let per = cnt * t * hd;
    let (qp, kp, vpp, op) = (s.scr("qp", per * 4)?, s.scr("kp", per * 4)?, s.scr("vp", per * 4)?, s.scr("op", per * 4)?);
    let (pm, ptm) = (s.scr("P", cnt * t * t * 4)?, s.scr("PT", cnt * t * t * 4)?);
    let sm = s.scr("S", cnt * t * t * 4)?;
    let ao = s.scr("ao", n * qdim * 4)?;
    let dao = s.scr("bdao", n * qdim * 4)?;
    let dgt = s.scr("bdgate", n * qdim * 4)?;
    let dop = s.scr("bdop", per * 4)?;
    let (dp, ds, dst) = (s.scr("bdP", cnt * t * t * 4)?, s.scr("bdS", cnt * t * t * 4)?, s.scr("bdST", cnt * t * t * 4)?);
    let (dqp, dkp, dvp) = (s.scr("bdqp", per * 4)?, s.scr("bdkp", per * 4)?, s.scr("bdvp", per * 4)?);
    let (dq, dk, dv) = (s.scr("bdq", n * qdim * 4)?, s.scr("bdk", n * kvdim * 4)?, s.scr("bdv", n * kvdim * 4)?);
    let dqkv = s.scr("bdqkv", n * fused * 4)?;
    let (t_i, nh_i, nkv_i, hd_i, half_i, gated_i, gm_i, hq, hk) =
        (t as i32, nh as i32, nkv as i32, hd as i32, a.half as i32, a.gated as i32, gemma as i32, a.q_norm.is_some() as i32, a.k_norm.is_some() as i32);
    let (b_i, one, rep_i, w_i) = (b as i32, 1i32, rep as i32, a.window.map_or(0, |w| w as i32));
    let scale = 1.0 / (hd as f32).sqrt();
    let grid = ((per as u64).div_ceil(256) as u32, 1);
    let qg = (((n * qdim) as u64).div_ceil(256) as u32, 1);
    let kvg = (((n * kvdim) as u64).div_ceil(256) as u32, 1);
    unsafe {
        // the forward again, to P and the pre-gate head outputs
        let xo = bop(c, s, f, x, n * hsz, "bfx", mode)?;
        c.gemm(true, false, fused, n, hsz, 1.0, wq, hsz, xo, hsz, 0.0, qkv, fused, mode)?;

        c.launch(f.attn_prep, (n as u32, (nh + nkv) as u32), hd.min(512) as u32, 0, kargs!(qkv, qn, kn, rope, qr, kr, vp, gt, t_i, nh_i, nkv_i, hd_i, half_i, gated_i, eps, gm_i, hq, hk))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(qr, qp, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(kr, kp, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.launch(f.attn_perm, grid, 256, 0, kargs!(vp, vpp, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.gemm_batched(true, t, t, hd, scale, kp, hd, t * hd, qp, hd, t * hd, sm, t, t * t, cnt, amode)?;
        c.launch(f.softmax_fwd, ((cnt * t) as u32, 1), 128, 0, kargs!(sm, pm, ptm, t_i, w_i))?;
        if a.gated {
            c.gemm_batched(false, hd, t, t, 1.0, vpp, hd, t * hd, pm, t, t * t, op, hd, t * hd, cnt, amode)?;
            c.launch(f.attn_unperm, qg, 256, 0, kargs!(op, ao, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        }
        // dao_eff = dy·Wo; through the output gate
        let dyo = bop(c, s, f, dy, n * hsz, "bfy", mode)?;
        c.gemm(false, false, qdim, n, hsz, 1.0, wo, qdim, dyo, hsz, 0.0, dao, qdim, mode)?;
        if a.gated {
            let tot = (n * qdim) as i64;
            c.launch(f.gate_bwd, qg, 256, 0, kargs!(dao, ao, gt, dgt, tot))?;
        }
        // the core: dP = dO·Vᵀ, dS, dQ = dS·K/√hd, dK = dSᵀ·Q/√hd, dV = Pᵀ·dO
        c.launch(f.attn_perm, grid, 256, 0, kargs!(dao, dop, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        c.gemm_batched(true, t, t, hd, 1.0, vpp, hd, t * hd, dop, hd, t * hd, dp, t, t * t, cnt, amode)?;
        c.launch(f.softmax_bwd, ((cnt * t) as u32, 1), 128, 0, kargs!(pm, dp, ds, dst, t_i))?;
        c.gemm_batched(false, hd, t, t, scale, kp, hd, t * hd, ds, t, t * t, dqp, hd, t * hd, cnt, amode)?;
        c.gemm_batched(false, hd, t, t, scale, qp, hd, t * hd, dst, t, t * t, dkp, hd, t * hd, cnt, amode)?;
        c.gemm_batched(false, hd, t, t, 1.0, dop, hd, t * hd, ptm, t, t * t, dvp, hd, t * hd, cnt, amode)?;
        c.launch(f.attn_unperm, qg, 256, 0, kargs!(dqp, dq, b_i, t_i, nh_i, nh_i, hd_i, one))?;
        c.launch(f.attn_unperm, kvg, 256, 0, kargs!(dkp, dk, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        c.launch(f.attn_unperm, kvg, 256, 0, kargs!(dvp, dv, b_i, t_i, nh_i, nkv_i, hd_i, rep_i))?;
        // RoPE and the q/k norm back to the projections, then their transpose
        c.launch(f.attn_prep_bwd, (n as u32, (nh + nkv) as u32), hd.min(512) as u32, 0, kargs!(qkv, qn, kn, rope, dq, dk, dv, dgt, dqkv, t_i, nh_i, nkv_i, hd_i, half_i, gated_i, eps, gm_i, hq, hk))?;
        if let Some((_, _, g, dg)) = gq {
            // qkv holds the gained projection (the row-scaled weight's
            // output): its gain gradient; the scaled weight carries dqkv back
            col_dot(c, s, f, qkv, dqkv, g, dg, n, fused, true)?;
        }
        let dqo = bop(c, s, f, dqkv, n * fused, "bfx", mode)?;
        c.gemm(false, false, hsz, n, fused, 1.0, wq, hsz, dqo, fused, 0.0, dx, hsz, mode)?;
    }
    Ok(())
}

/// The GDN branch's input gradient on the card: `dy` → `dx` at its normed
/// input `x`, frozen weights. The four projections and the conv are
/// recomputed from `x`; then out_proj's transpose, the recurrence's BPTT,
/// the conv's backward and the projections' transpose.
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_bwd_dev(c: &'static Ctx, s: &mut BkState, f: &BkFuncs, g: &BakeGdn, w: &BakeGdnW, x: u64, dy: u64, dx: u64, b: usize, t: usize, hsz: usize, lin: Option<(u64, u64, LinOff)>) -> Res<()> {
    let (n, cdim, vd, nv) = (b * t, g.c_dim(), g.nv * g.dv, g.nv);
    let mode = res_mode();
    let (wq, wz, wa, wb, wo) = match lin {
        // with output masks: the row-scaled weights (the gains live in them)
        Some((gl, _, o)) => {
            let at = |r: (usize, usize)| gl + (r.0 * 4) as u64;
            unsafe {
                (
                    wscaled(c, s, f, w.wqkv, cdim, at(o.qkv), mode)?,
                    wscaled(c, s, f, w.wz, vd, at(o.z), mode)?,
                    wscaled(c, s, f, w.wa, nv, at(o.a), Gemm::F32)?,
                    wscaled(c, s, f, w.wb, nv, at(o.b), Gemm::F32)?,
                    wscaled(c, s, f, w.wout, hsz, at(o.out), mode)?,
                )
            }
        }
        None => (s.wm(w.wqkv, mode)?, s.wm(w.wz, mode)?, s.weight(w.wa)?, s.weight(w.wb)?, s.wm(w.wout, mode)?),
    };
    let (qkv, z, a, bb) = (s.scr("qkv", n * cdim * 4)?, s.scr("z", n * vd * 4)?, s.scr("a", n * nv * 4)?, s.scr("b", n * nv * 4)?);
    let (taps, alog, dtb, nw) = (s.up(c, "taps", g.conv)?, s.up(c, "alog", g.a_log)?, s.up(c, "dtb", g.dt_bias)?, s.up(c, "nw", g.norm)?);
    let dof = s.scr("bdof", n * vd * 4)?;
    let draw = s.scr("bdraw", n * cdim * 4)?;
    let (dz, da, db) = (s.scr("bdz", n * vd * 4)?, s.scr("bda", n * nv * 4)?, s.scr("bdb", n * nv * 4)?);
    // The NVG trainer's operator (states at chunk starts, the chunked reverse
    // sweep) for 128-wide heads; `CMF_BAKE_GDN_NVG=0` keeps the bake's own
    // per-position BPTT below.
    let nvg = g.dk == 128 && g.dv == 128 && std::env::var("CMF_BAKE_GDN_NVG").as_deref() != Ok("0");
    let (n_i, t_i, cd_i, kk_i, nv_i, nk_i) = (n as i32, t as i32, cdim as i32, g.kk as i32, nv as i32, g.nk as i32);
    let eps = g.eps;
    let tot = (n * cdim) as u64;
    unsafe {
        let xo = bop(c, s, f, x, n * hsz, "bfx", mode)?;
        c.gemm(true, false, cdim, n, hsz, 1.0, wq, hsz, xo, hsz, 0.0, qkv, cdim, mode)?;
        c.gemm(true, false, vd, n, hsz, 1.0, wz, hsz, xo, hsz, 0.0, z, vd, mode)?;
        c.gemm(true, false, nv, n, hsz, 1.0, wa, hsz, x, hsz, 0.0, a, nv, Gemm::F32)?;
        c.gemm(true, false, nv, n, hsz, 1.0, wb, hsz, x, hsz, 0.0, bb, nv, Gemm::F32)?;
        // dof = dy·Wout
        let dyo = bop(c, s, f, dy, n * hsz, "bfy", mode)?;
        c.gemm(false, false, vd, n, hsz, 1.0, wo, vd, dyo, hsz, 0.0, dof, vd, mode)?;
        if nvg {
            super::nvgt::bake_gdn_bwd(c, (b, t, nv, g.nk, g.kk, eps), [alog, dtb, nw, taps], qkv, z, a, bb, dof, draw, dz, da, db).map_err(|e| format!("nvg gdn: {e}"))?;
        } else {
            let (pre, cq) = (s.scr("pre", n * cdim * 4)?, s.scr("cq", n * cdim * 4)?);
            let dcq = s.scr("bdcq", n * cdim * 4)?;
            let per = seqs_per_launch(g, t, b);
            let hist = s.scr("bhist", per * g.nv * (t + 1) * g.dk * g.dv * 4)?;
            let kvh = s.scr("bkvh", per * g.nv * t * g.dv * 4)?;
            let oh = s.scr("boh", per * g.nv * t * g.dv * 4)?;
            c.launch(f.conv_fwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(qkv, taps, pre, cq, n_i, t_i, cd_i, kk_i))?;
            cu::memset_d8_async(dcq, 0, n * cdim * 4, c.stream).map_err(derr)?;
            let mut s0 = 0usize;
            while s0 < b {
                let nb = per.min(b - s0);
                let seq0 = s0 as i32;
                c.launch(f.gdn_bwd, (nv as u32, nb as u32), g.dv as u32, 0, kargs!(cq, z, a, bb, alog, dtb, nw, dof, dcq, dz, da, db, hist, kvh, oh, t_i, nv_i, nk_i, eps, seq0))?;
                s0 += nb;
            }
            c.launch(f.conv_bwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(pre, dcq, taps, draw, n_i, t_i, cd_i, kk_i))?;
        }
        if let Some((gl, dl, o)) = lin {
            // the inputs hold the gained projections: the gain gradients (the
            // row-scaled weights carry each gradient back to x)
            let at = |base: u64, r: (usize, usize)| base + (r.0 * 4) as u64;
            for (y, d, r, m) in [(qkv, draw, o.qkv, cdim), (z, dz, o.z, vd), (a, da, o.a, nv), (bb, db, o.b, nv)] {
                col_dot(c, s, f, y, d, at(gl, r), at(dl, r), n, m, true)?;
            }
        }
        // dx = draw·Wqkv + dz·Wz + da·Wa + db·Wb
        let dro = bop(c, s, f, draw, n * cdim, "bfx", mode)?;
        c.gemm(false, false, hsz, n, cdim, 1.0, wq, hsz, dro, cdim, 0.0, dx, hsz, mode)?;
        let dzo = bop(c, s, f, dz, n * vd, "bfy", mode)?;
        c.gemm(false, false, hsz, n, vd, 1.0, wz, hsz, dzo, vd, 1.0, dx, hsz, mode)?;
        c.gemm(false, false, hsz, n, nv, 1.0, wa, hsz, da, nv, 1.0, dx, hsz, Gemm::F32)?;
        c.gemm(false, false, hsz, n, nv, 1.0, wb, hsz, db, nv, 1.0, dx, hsz, Gemm::F32)?;
    }
    Ok(())
}

/// `CMF_BAKE_RES_PROF=1`: the step synchronizes after each section and
/// prints the wall time per section (the syncs remove the overlap between
/// sections, so the sum runs a little over the plain step).
struct StepProf {
    on: bool,
    t: std::time::Instant,
    secs: Vec<(&'static str, f64)>,
}

impl StepProf {
    fn new() -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("CMF_BAKE_RES_PROF").as_deref() == Ok("1"));
        Self { on, t: std::time::Instant::now(), secs: Vec::new() }
    }

    fn mark(&mut self, c: &Ctx, name: &'static str) {
        if !self.on {
            return;
        }
        let _ = c.sync();
        let dt = self.t.elapsed().as_secs_f64() * 1e3;
        match self.secs.iter_mut().find(|(n, _)| *n == name) {
            Some(x) => x.1 += dt,
            None => self.secs.push((name, dt)),
        }
        self.t = std::time::Instant::now();
    }

    fn report(&self) {
        if self.on {
            let total: f64 = self.secs.iter().map(|x| x.1).sum();
            let parts: Vec<String> = self.secs.iter().map(|(n, ms)| format!("{n} {ms:.1}")).collect();
            eprintln!("[bake-res-prof cuda] {total:.1} ms: {}", parts.join(" | "));
        }
    }
}

/// A whole Phase-A step of the skill bake on the card: embeddings, every
/// layer (frozen attention — GDN or full —, the masked FFN), the final norm,
/// the cross-entropy over the scored rows and, with `grad`, the backward of
/// the FFN path down to each layer's mask gradient — through each frozen
/// attention (or GDN) branch too with `through_attn`, else dh1 flows to dh_in. Only the ids go up
/// and the per-target numbers and the mask gradients come down. None: the
/// layer-by-layer host-orchestrated path.
pub fn bake_phase_a(st: &BkStep) -> Option<BkOut> {
    let c = ctx()?;
    let (b, t, hsz) = (st.b, st.t, st.hsz);
    let n = b * t;
    let nl = st.layers.len();
    if c.current().is_err() || st.ids.len() != n || hsz == 0 || st.lm.len() % hsz != 0 || st.embed.len() % hsz != 0 || st.targets.is_empty() || st.targets.iter().any(|&r| r == 0 || r >= n) {
        return None;
    }
    let vocab = st.lm.len() / hsz;
    let target_of = |k: usize| -> usize { st.classes.map_or(st.ids[st.targets[k]] as usize, |c| c[k] as usize) };
    if st.ids.iter().any(|&i| i as usize >= st.embed.len() / hsz)
        || st.classes.is_some_and(|c| c.len() != st.targets.len())
        || (0..st.targets.len()).any(|k| target_of(k) >= vocab)
    {
        return None;
    }
    let _g = c.st.lock().unwrap();
    let mut gs = STATE.lock().unwrap();
    let s = gs.get_or_insert_with(BkState::default);
    let mut prof = StepProf::new();
    let mut run = |s: &mut BkState| -> Res<BkOut> {
        let gdn_dims = st.layers.iter().find_map(|l| match &l.attn {
            BkAttn::Gdn(g, _) => Some((g.dk, g.dv)),
            _ => None,
        });
        let f = match gdn_dims {
            Some((dk, dv)) => funcs(c, s, dk, dv)?,
            None => funcs(c, s, 128, 128)?,
        };
        let (mode, amode) = (res_mode(), gemm_mode());
        let (eps, gm) = (st.eps, st.gemma as i32);
        let hsz_i = hsz as i32;
        let gelu = (st.act == crate::fcd_ops::FfnAct::GeluErf) as i32;
        let max_inter = st.layers.iter().map(|l| l.inter).max().unwrap_or(0);
        // per-layer saves
        while s.saves.len() < nl {
            s.saves.push((Buf::new(4)?, Buf::new(4)?, Buf::new(4)?));
        }
        for (li, l) in st.layers.iter().enumerate() {
            let need = [n * hsz * 4, n * 4, n * 2 * l.inter * 4];
            let sv = &mut s.saves[li];
            if sv.0.bytes < need[0] { sv.0 = Buf::new(need[0])?; }
            if sv.1.bytes < need[1] { sv.1 = Buf::new(need[1])?; }
            if st.grad && sv.2.bytes < need[2] { sv.2 = Buf::new(need[2])?; }
        }
        // output masks (`BkLayer::lin`): every layer's gains in one buffer,
        // their gradient beside, the output projections' pre-gain outputs
        let lin_on = st.layers.iter().any(|l| l.lin.is_some());
        let offs: Vec<LinOff> = st.layers.iter().map(|l| lin_layout(&l.attn, l.inter, hsz)).collect();
        let lmax = offs.iter().map(|o| o.len).max().unwrap_or(0);
        let (glin, dlin) = if lin_on {
            if st.layers.iter().zip(&offs).any(|(l, o)| l.lin.map(<[f32]>::len) != Some(o.len)) {
                return Err("output masks: every layer needs its gains, laid out as lin_layout gives".into());
            }
            if st.grad && !st.through_attn {
                return Err("output masks train through the attention branches (--through-attn)".into());
            }
            let mut flat = vec![0f32; nl * lmax];
            for (li, l) in st.layers.iter().enumerate() {
                flat[li * lmax..li * lmax + offs[li].len].copy_from_slice(l.lin.expect("checked above"));
            }
            let g = s.up(c, "rlin", &flat)?;
            let d = if st.grad {
                let d = s.scr("rdlin", nl * lmax * 4)?;
                unsafe { cu::memset_d8_async(d, 0, nl * lmax * 4, c.stream).map_err(derr)? };
                d
            } else {
                0
            };
            if st.grad {
                while s.saves_lin.len() < nl {
                    s.saves_lin.push((Buf::new(4)?, Buf::new(4)?));
                }
                for sv in s.saves_lin.iter_mut().take(nl) {
                    if sv.0.bytes < n * hsz * 4 { sv.0 = Buf::new(n * hsz * 4)?; }
                    if sv.1.bytes < n * hsz * 4 { sv.1 = Buf::new(n * hsz * 4)?; }
                }
            }
            (g, d)
        } else {
            (0, 0)
        };
        // a layer's gains (or their gradient) for one projection
        let gat = |base: u64, li: usize, r: (usize, usize)| base + ((li * lmax + r.0) * 4) as u64;
        let emb = s.weight(st.embed)?;
        let wl = s.weight(st.lm)?;
        let ids: Vec<i32> = st.ids.iter().map(|&i| i as i32).collect();
        let idb = Buf::from_slice(&ids)?;
        let h = s.scr("rh", n * hsz * 4)?;
        let nrm = s.scr("rn", n * hsz * 4)?;
        let ao = s.scr("rao", n * hsz * 4)?;
        let inv1 = s.scr("rinv1", n * 4)?;
        let act = s.scr("ract", n * max_inter * 4)?;
        let (n_i, tot_h) = (n as i32, (n * hsz) as i64);
        let gh = (((n * hsz) as u64).div_ceil(256) as u32, 1);
        unsafe {
            c.launch(f.embed, gh, 256, 0, kargs!(emb, idb.ptr, h, n_i, hsz_i))?;
        }
        let keep_in = st.grad && st.through_attn;
        if keep_in {
            while s.saves_in.len() < nl {
                s.saves_in.push(Buf::new(4)?);
            }
            for li in 0..nl {
                if s.saves_in[li].bytes < n * hsz * 4 {
                    s.saves_in[li] = Buf::new(n * hsz * 4)?;
                }
            }
        }
        for (li, l) in st.layers.iter().enumerate() {
            let iln = s.up(c, "riln", l.iln)?;
            let pln = s.up(c, "rpln", l.pln)?;
            let (h1, inv2) = (s.saves[li].0.ptr, s.saves[li].1.ptr);
            unsafe {
                if keep_in {
                    cu::memcpy_dtod_async(s.saves_in[li].ptr, h, n * hsz * 4, c.stream).map_err(derr)?;
                }
                c.launch(f.rmsnorm, (n as u32, 1), 256, 0, kargs!(h, iln, nrm, inv1, hsz_i, eps, gm))?;
            }
            match &l.attn {
                BkAttn::Gdn(g, w) => {
                    let (cdim, vd, nv) = (g.c_dim(), g.nv * g.dv, g.nv);
                    let (wq, wz, wa, wb, wo) = if lin_on {
                        // the output masks ride in row-scaled weights
                        let o = offs[li];
                        unsafe {
                            (
                                wscaled(c, s, &f, w.wqkv, cdim, gat(glin, li, o.qkv), mode)?,
                                wscaled(c, s, &f, w.wz, vd, gat(glin, li, o.z), mode)?,
                                wscaled(c, s, &f, w.wa, nv, gat(glin, li, o.a), Gemm::F32)?,
                                wscaled(c, s, &f, w.wb, nv, gat(glin, li, o.b), Gemm::F32)?,
                                wscaled(c, s, &f, w.wout, hsz, gat(glin, li, o.out), mode)?,
                            )
                        }
                    } else {
                        (s.wm(w.wqkv, mode)?, s.wm(w.wz, mode)?, s.weight(w.wa)?, s.weight(w.wb)?, s.wm(w.wout, mode)?)
                    };
                    let (qkv, z, a, bb) = (s.scr("qkv", n * cdim * 4)?, s.scr("z", n * vd * 4)?, s.scr("a", n * nv * 4)?, s.scr("b", n * nv * 4)?);
                    let (taps, alog, dtb, nw) = (s.up(c, "taps", g.conv)?, s.up(c, "alog", g.a_log)?, s.up(c, "dtb", g.dt_bias)?, s.up(c, "nw", g.norm)?);
                    let (pre, cq, of) = (s.scr("pre", n * cdim * 4)?, s.scr("cq", n * cdim * 4)?, s.scr("of", n * vd * 4)?);
                    let (t_i, cd_i, kk_i, nv_i, nk_i, seq0) = (t as i32, cdim as i32, g.kk as i32, nv as i32, g.nk as i32, 0i32);
                    let geps = g.eps;
                    unsafe {
                        let xo = bop(c, s, &f, nrm, n * hsz, "bfx", mode)?;
                        c.gemm(true, false, cdim, n, hsz, 1.0, wq, hsz, xo, hsz, 0.0, qkv, cdim, mode)?;
                        c.gemm(true, false, vd, n, hsz, 1.0, wz, hsz, xo, hsz, 0.0, z, vd, mode)?;
                        c.gemm(true, false, nv, n, hsz, 1.0, wa, hsz, nrm, hsz, 0.0, a, nv, Gemm::F32)?;
                        c.gemm(true, false, nv, n, hsz, 1.0, wb, hsz, nrm, hsz, 0.0, bb, nv, Gemm::F32)?;
                        if g.dk == 128 && g.dv == 128 && std::env::var("CMF_BAKE_GDN_NVG").as_deref() != Ok("0") {
                            // the NVG trainer's forward (the backward runs on its operator too)
                            super::nvgt::bake_gdn_fwd_dev(c, (b, t, nv, g.nk, g.kk, geps), [alog, dtb, nw, taps], qkv, z, a, bb, pre, cq, of)
                                .map_err(|e| format!("nvg gdn forward: {e}"))?;
                        } else {
                            let tot = (n * cdim) as u64;
                            c.launch(f.conv_fwd, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(qkv, taps, pre, cq, n_i, t_i, cd_i, kk_i))?;
                            c.launch(f.gdn_fwd, (nv as u32, b as u32), g.dv as u32, 0, kargs!(cq, z, a, bb, alog, dtb, nw, of, t_i, nv_i, nk_i, geps, seq0))?;
                        }
                        let oo = bop(c, s, &f, of, n * vd, "bfy", mode)?;
                        c.gemm(true, false, hsz, n, vd, 1.0, wo, vd, oo, vd, 0.0, ao, hsz, mode)?;
                    }
                    prof.mark(c, "fwd.gdn");
                }
                BkAttn::Full(fa) => {
                    let gw = if lin_on {
                        let (qdim, kvdim) = (fa.nh * fa.hd, fa.nkv * fa.hd);
                        let fused = (if fa.gated { 2 * qdim } else { qdim }) + 2 * kvdim;
                        unsafe { Some((wscaled(c, s, &f, fa.wqkv, fused, gat(glin, li, offs[li].qkv), mode)?, wscaled(c, s, &f, fa.wo, hsz, gat(glin, li, offs[li].out), mode)?)) }
                    } else {
                        None
                    };
                    unsafe { full_attn_dev(c, s, &f, fa, nrm, ao, b, t, hsz, eps, st.gemma, gw)? };
                    prof.mark(c, "fwd.full");
                }
            }
            let inter = l.inter;
            let (wgu, wdn) = if lin_on {
                unsafe { (wscaled(c, s, &f, l.gu, 2 * inter, gat(glin, li, offs[li].gate), mode)?, wscaled(c, s, &f, l.down, hsz, gat(glin, li, offs[li].down), mode)?) }
            } else {
                (s.wm(l.gu, mode)?, s.wm(l.down, mode)?)
            };
            let gain = s.up(c, "rgain", l.gain)?;
            let both = if st.grad { s.saves[li].2.ptr } else { s.scr("rboth", n * 2 * inter * 4)? };
            let (in_i, one) = (inter as i32, 1i32);
            unsafe {
                if lin_on && st.grad {
                    // the attention output (gained already) kept for its gain's gradient
                    cu::memcpy_dtod_async(s.saves_lin[li].0.ptr, ao, n * hsz * 4, c.stream).map_err(derr)?;
                }
                c.launch(f.add, gh, 256, 0, kargs!(h, ao, h1, tot_h))?;
                c.launch(f.rmsnorm, (n as u32, 1), 256, 0, kargs!(h1, pln, nrm, inv2, hsz_i, eps, gm))?;
                let xo = bop(c, s, &f, nrm, n * hsz, "bfx", mode)?;
                c.gemm(true, false, 2 * inter, n, hsz, 1.0, wgu, hsz, xo, hsz, 0.0, both, 2 * inter, mode)?;
                let tot = (n * inter) as u64;
                c.launch(f.ffn_act, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(both, gain, act, n_i, in_i, gelu, one))?;
                let actb = bop(c, s, &f, act, n * inter, "bfy", mode)?;
                c.gemm(true, false, hsz, n, inter, 1.0, wdn, inter, actb, inter, 0.0, ao, hsz, mode)?;
                if lin_on && st.grad {
                    cu::memcpy_dtod_async(s.saves_lin[li].1.ptr, ao, n * hsz * 4, c.stream).map_err(derr)?;
                }
                c.launch(f.add, gh, 256, 0, kargs!(h1, ao, h, tot_h))?;
            }
            prof.mark(c, "fwd.ffn");
        }
        // final norm, the scored rows' cross-entropy
        let fnw = s.up(c, "rfn", st.final_norm)?;
        let finv = s.scr("rfinv", n * 4)?;
        let k = st.targets.len();
        let src: Vec<i32> = st.targets.iter().map(|&r| (r - 1) as i32).collect();
        let tg: Vec<i32> = (0..k).map(|i| target_of(i) as i32).collect();
        let (srcb, tgb) = (Buf::from_slice(&src)?, Buf::from_slice(&tg)?);
        let hs = s.scr("rhs", k * hsz * 4)?;
        let band = ((512usize << 20) / (vocab * 4)).clamp(1, k);
        let logits = s.scr("logits", band * vocab * 4)?;
        let (nll, arg) = (s.scr("nll", k * 4)?, s.scr("arg", k * 4)?);
        let dhs = s.scr("rdhs", k * hsz * 4)?;
        let (k_i, v_i, gr) = (k as i32, vocab as i32, st.grad as i32);
        let scale = st.scale;
        unsafe {
            c.launch(f.rmsnorm, (n as u32, 1), 256, 0, kargs!(h, fnw, nrm, finv, hsz_i, eps, gm))?;
            c.launch(f.gather, (((k * hsz) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(nrm, srcb.ptr, hs, k_i, hsz_i))?;
            let mut r0 = 0usize;
            while r0 < k {
                let nb = band.min(k - r0);
                let xp = hs + (r0 * hsz * 4) as u64;
                c.gemm(true, false, vocab, nb, hsz, 1.0, wl, hsz, xp, hsz, 0.0, logits, vocab, amode)?;
                let (tp, np, ap) = (tgb.ptr + (r0 * 4) as u64, nll + (r0 * 4) as u64, arg + (r0 * 4) as u64);
                c.launch(f.ce_rows, (nb as u32, 1), 1024, 0, kargs!(logits, tp, np, ap, v_i, scale, gr))?;
                if st.grad {
                    let dp = dhs + (r0 * hsz * 4) as u64;
                    c.gemm(false, false, hsz, nb, vocab, 1.0, wl, hsz, logits, vocab, 0.0, dp, hsz, amode)?;
                }
                r0 += nb;
            }
        }
        let nl_v = dtoh(c, nll, k)?;
        let mut am = vec![0i32; k];
        unsafe { cu::memcpy_dtoh_async(am.as_mut_slice(), arg, c.stream).map_err(derr)? };
        prof.mark(c, "head");
        let mut dmask = Vec::new();
        let mut dlin_out: Vec<Vec<f32>> = Vec::new();
        if st.grad {
            // dhn (scattered) → dh through the final norm
            let dhn = s.scr("rdhn", n * hsz * 4)?;
            let dh = s.scr("rdh", n * hsz * 4)?;
            let dn2 = s.scr("rdn2", n * hsz * 4)?;
            let dact = s.scr("ract", n * max_inter * 4)?;
            let dgu = s.scr("rdgu", n * 2 * max_inter * 4)?;
            let dm = s.scr("rdm", nl * max_inter * 4)?;
            unsafe {
                cu::memset_d8_async(dhn, 0, n * hsz * 4, c.stream).map_err(derr)?;
                cu::memset_d8_async(dm, 0, nl * max_inter * 4, c.stream).map_err(derr)?;
                c.launch(f.scatter_add, (((k * hsz) as u64).div_ceil(256) as u32, 1), 256, 0, kargs!(dhs, srcb.ptr, dhn, k_i, hsz_i))?;
                let zero = 0i32;
                c.launch(f.rmsnorm_bwd, (n as u32, 1), 256, 0, kargs!(h, fnw, finv, dhn, dh, hsz_i, gm, zero))?;
            }
            for (li, l) in st.layers.iter().enumerate().rev() {
                let inter = l.inter;
                let (wgu, wdn) = if lin_on {
                    unsafe { (wscaled(c, s, &f, l.gu, 2 * inter, gat(glin, li, offs[li].gate), mode)?, wscaled(c, s, &f, l.down, hsz, gat(glin, li, offs[li].down), mode)?) }
                } else {
                    (s.wm(l.gu, mode)?, s.wm(l.down, mode)?)
                };
                let gain = s.up(c, "rgain", l.gain)?;
                let pln = s.up(c, "rpln", l.pln)?;
                let (h1, inv2, both) = (s.saves[li].0.ptr, s.saves[li].1.ptr, s.saves[li].2.ptr);
                let (in_i, one, rows) = (inter as i32, 1i32, 64i32);
                let dml = dm + (li * max_inter * 4) as u64;
                let acc = 1i32;
                unsafe {
                    // dact2 = dh·down; mask grad, dgu; dn2 = dgu·gu
                    if lin_on {
                        // the down gain's gradient from the kept (gained) FFN output
                        let (g, d) = (gat(glin, li, offs[li].down), gat(dlin, li, offs[li].down));
                        let kept = s.saves_lin[li].1.ptr;
                        col_dot(c, s, &f, kept, dh, g, d, n, hsz, true)?;
                    }
                    let dho = bop(c, s, &f, dh, n * hsz, "bfx", mode)?;
                    c.gemm(false, false, inter, n, hsz, 1.0, wdn, inter, dho, hsz, 0.0, dact, inter, mode)?;
                    c.launch(f.ffn_act_bwd, (inter.div_ceil(256) as u32, n.div_ceil(64) as u32), 256, 0, kargs!(both, dact, gain, dgu, dml, n_i, in_i, gelu, one, rows))?;
                    if lin_on {
                        // both holds the gained gate/up: their gains' gradient
                        let (g, d) = (gat(glin, li, offs[li].gate), gat(dlin, li, offs[li].gate));
                        col_dot(c, s, &f, both, dgu, g, d, n, 2 * inter, true)?;
                    }
                    let dguo = bop(c, s, &f, dgu, n * 2 * inter, "bfy", mode)?;
                    c.gemm(false, false, hsz, n, 2 * inter, 1.0, wgu, hsz, dguo, 2 * inter, 0.0, dn2, hsz, mode)?;
                    // dh1 = dh + post-norm backward (dh accumulates in place);
                    // without through_attn the frozen attention passes dh1
                    // straight to dh_in
                    c.launch(f.rmsnorm_bwd, (n as u32, 1), 256, 0, kargs!(h1, pln, inv2, dn2, dh, hsz_i, gm, acc))?;
                }
                prof.mark(c, "bwd.ffn");
                if st.through_attn {
                    // dh_in = dh1 + input-norm backward of the branch's dX
                    let hin = s.saves_in[li].ptr;
                    let iln = s.up(c, "riln", l.iln)?;
                    let dn1 = s.scr("rdn1", n * hsz * 4)?;
                    unsafe {
                        c.launch(f.rmsnorm, (n as u32, 1), 256, 0, kargs!(hin, iln, nrm, inv1, hsz_i, eps, gm))?;
                        // with output masks: the attention output gain's gradient, then dh at the branch's own output
                        if lin_on {
                            // the attention output gain's gradient from the kept (gained) output
                            let (g, d) = (gat(glin, li, offs[li].out), gat(dlin, li, offs[li].out));
                            let kept = s.saves_lin[li].0.ptr;
                            col_dot(c, s, &f, kept, dh, g, d, n, hsz, true)?;
                        }
                        let dya = dh;
                        match &l.attn {
                            BkAttn::Full(fa) => {
                                let gq = if lin_on {
                                    let (qdim, kvdim) = (fa.nh * fa.hd, fa.nkv * fa.hd);
                                    let fused = (if fa.gated { 2 * qdim } else { qdim }) + 2 * kvdim;
                                    Some((
                                        wscaled(c, s, &f, fa.wqkv, fused, gat(glin, li, offs[li].qkv), mode)?,
                                        wscaled(c, s, &f, fa.wo, hsz, gat(glin, li, offs[li].out), mode)?,
                                        gat(glin, li, offs[li].qkv),
                                        gat(dlin, li, offs[li].qkv),
                                    ))
                                } else {
                                    None
                                };
                                full_attn_bwd_dev(c, s, &f, fa, nrm, dya, dn1, b, t, hsz, eps, st.gemma, gq)?
                            }
                            BkAttn::Gdn(g, w) => {
                                let lb = lin_on.then(|| (glin + (li * lmax * 4) as u64, dlin + (li * lmax * 4) as u64, offs[li]));
                                gdn_bwd_dev(c, s, &f, g, w, nrm, dya, dn1, b, t, hsz, lb)?
                            }
                        }
                        c.launch(f.rmsnorm_bwd, (n as u32, 1), 256, 0, kargs!(hin, iln, inv1, dn1, dh, hsz_i, gm, acc))?;
                    }
                    prof.mark(c, if matches!(l.attn, BkAttn::Full(_)) { "bwd.full" } else { "bwd.gdn" });
                }
            }
            let all = dtoh(c, dm, nl * max_inter)?;
            dmask = st.layers.iter().enumerate().map(|(li, l)| all[li * max_inter..li * max_inter + l.inter].to_vec()).collect();
            if lin_on {
                let all = dtoh(c, dlin, nl * lmax)?;
                dlin_out = offs.iter().enumerate().map(|(li, o)| all[li * lmax..li * lmax + o.len].to_vec()).collect();
            }
        }
        c.sync()?;
        prof.mark(c, "readback");
        Ok(BkOut { nll: nl_v, argmax: am.into_iter().map(|a| a as u32).collect(), dmask, dlin: dlin_out })
    };
    // CMF_BAKE_KT=<n>: the n-th training step on the card (0-based; the
    // evaluations do not count) runs with every kernel launch timed, printed
    // per kernel after it
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = if st.grad { CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) } else { usize::MAX };
    let kt = std::env::var("CMF_BAKE_KT").ok().and_then(|v| v.parse::<usize>().ok()) == Some(call);
    if kt {
        super::kt_start();
    }
    let mut res = run(s);
    if kt {
        let _ = c.sync();
        super::kt_stop_as(if st.grad { "bake step" } else { "bake eval" });
    }
    if matches!(&res, Err(e) if e.contains("OUT_OF_MEMORY")) {
        // A pass of another shape (the held evaluation's batch, a longer
        // bucket) left its scratch and saves on the card: free them — the
        // frozen weights stay — and run the step again before the layer
        // path takes over (silently ~10x slower for the rest of the bake).
        let _ = c.sync();
        s.scr.clear();
        s.saves.clear();
        s.saves_in.clear();
        s.saves_lin.clear();
        s.planes.clear();
        res = run(s);
        if res.is_ok() {
            tracing::info!("cuda bake resident step: out of memory, scratch freed, step resumed on the card");
        }
    }
    prof.report();
    match res {
        Ok(o) => Some(o),
        Err(e) => {
            let _ = c.sync();
            decline("bake resident step", &e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host rounding of the bf16 weight copies: nearest, ties to even,
    /// NaN stays NaN (the kernel `bk_to_bf16` is the same expression).
    #[test]
    fn bf16_bits_round_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(-2.5), 0xc020);
        assert_eq!(bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80); // tie, even stays
        assert_eq!(bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82); // tie, odd rounds up
        assert_eq!(bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81); // above the tie
        assert_eq!(bf16_bits(f32::INFINITY), 0x7f80);
        assert!(bf16_bits(f32::NAN) & 0x7fff > 0x7f80);
        assert_eq!(bf16_bits(f32::from_bits(0x7f7f_ffff)), 0x7f80); // the largest f32 rounds to inf
    }

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
            })
            .collect()
    }

    fn rel(a: &[f32], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-30);
        (num / den).sqrt()
    }

    /// The card's GDN forward and BPTT answer what `fcd_ops` answers in f64
    /// (two sequences, GQA 4 v-heads over 2 k-heads, dk 32 / dv 64).
    #[test]
    fn bake_gdn_matches_host() {
        if std::env::var("CMF_GPU").as_deref() != Ok("cuda") {
            eprintln!("CMF_GPU=cuda not set — skip");
            return;
        }
        let (b, t, nv, nk, dk, dv, kk) = (2usize, 37usize, 4usize, 2usize, 32usize, 64usize, 4usize);
        let cdim = 2 * nk * dk + nv * dv;
        let n = b * t;
        let qkv = rnd(n * cdim, 1, 1.0);
        let z = rnd(n * nv * dv, 2, 1.0);
        let a = rnd(n * nv, 3, 1.0);
        let bs = rnd(n * nv, 4, 1.0);
        let dout = rnd(n * nv * dv, 5, 1.0);
        let conv = rnd(cdim * kk, 6, 0.5);
        let a_log = rnd(nv, 7, 0.5);
        let dt_bias = rnd(nv, 8, 0.5);
        let norm: Vec<f32> = rnd(dv, 9, 0.2).iter().map(|v| 1.0 + v).collect();
        let g = BakeGdn { nv, nk, dk, dv, kk, eps: 1e-6, conv: &conv, a_log: &a_log, dt_bias: &dt_bias, norm: &norm };
        if !bake_gdn_ok(&g) {
            eprintln!("no CUDA device — skip");
            return;
        }
        let cfg = crate::fcd_ops::GdnSeqCfg { nv, nk, dk, dv, kk, rms_eps: 1e-6, conv: &conv, a_log: &a_log, dt_bias: &dt_bias, norm: &norm };
        let f64v = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let (vd, c) = (nv * dv, cdim);
        let mut want_o = vec![0f64; n * vd];
        let (mut wq, mut wz, mut wa, mut wb) = (vec![0f64; n * c], vec![0f64; n * vd], vec![0f64; n * nv], vec![0f64; n * nv]);
        for bi in 0..b {
            let (rc, rv, rh) = (bi * t * c..(bi + 1) * t * c, bi * t * vd..(bi + 1) * t * vd, bi * t * nv..(bi + 1) * t * nv);
            crate::fcd_ops::gdn_seq_fwd(&f64v(&qkv[rc.clone()]), &f64v(&z[rv.clone()]), &f64v(&a[rh.clone()]), &f64v(&bs[rh.clone()]), t, &cfg, &mut want_o[rv.clone()]);
            crate::fcd_ops::gdn_seq_bwd(
                &f64v(&qkv[rc.clone()]),
                &f64v(&z[rv.clone()]),
                &f64v(&a[rh.clone()]),
                &f64v(&bs[rh.clone()]),
                t,
                &cfg,
                &f64v(&dout[rv.clone()]),
                &mut wq[rc],
                &mut wz[rv],
                &mut wa[rh.clone()],
                &mut wb[rh],
            );
        }
        let o = bake_gdn_fwd(&qkv, &z, &a, &bs, b, t, &g).expect("forward ran");
        let e = rel(&o, &want_o);
        eprintln!("forward rel {e:.2e}");
        assert!(e < 1e-4, "GDN forward vs host rel {e:e}");
        let (dq, dz, da, db) = bake_gdn_bwd(&qkv, &z, &a, &bs, &dout, b, t, &g).expect("backward ran");
        for (name, got, want) in [("dqkv", &dq, &wq), ("dz", &dz, &wz), ("da", &da, &wa), ("db", &db, &wb)] {
            let e = rel(got, want);
            eprintln!("{name} rel {e:.2e}");
            assert!(e < 1e-3, "GDN backward {name} vs host rel {e:e}");
        }
    }


    /// The card's exact attention core (forward and backward, full and a
    /// sliding window, GQA 4 q-heads over 2 kv-heads) against `fcd_ops` in f64.
    #[test]
    fn bake_attn_matches_host() {
        if std::env::var("CMF_GPU").as_deref() != Ok("cuda") {
            eprintln!("CMF_GPU=cuda not set — skip");
            return;
        }
        if ctx().is_none() {
            eprintln!("no CUDA device — skip");
            return;
        }
        let (b, t, nh, nkv, hd) = (2usize, 33usize, 4usize, 2usize, 64usize);
        let (n, qdim, kvdim, rep) = (b * t, nh * hd, nkv * hd, nh / nkv);
        let q = rnd(n * qdim, 11, 1.0);
        let k = rnd(n * kvdim, 12, 1.0);
        let v = rnd(n * kvdim, 13, 1.0);
        let dao = rnd(n * qdim, 14, 1.0);
        for window in [None, Some(8usize)] {
            let (mut wao, mut wdq, mut wdk, mut wdv) = (vec![0f64; n * qdim], vec![0f64; n * qdim], vec![0f64; n * kvdim], vec![0f64; n * kvdim]);
            for bi in 0..b {
                for h in 0..nh {
                    let g = h / rep;
                    let pick = |src: &[f32], w: usize, hh: usize| -> Vec<f64> {
                        (0..t).flat_map(|p| (0..hd).map(move |c| (p, c))).map(|(p, c)| src[(bi * t + p) * w + hh * hd + c] as f64).collect()
                    };
                    let (q64, k64, v64, do64) = (pick(&q, qdim, h), pick(&k, kvdim, g), pick(&v, kvdim, g), pick(&dao, qdim, h));
                    let mut o = vec![0f64; t * hd];
                    crate::fcd_ops::attn_head_fwd_window(&q64, &k64, &v64, t, hd, hd, window, &mut o);
                    let (mut dq, mut dk, mut dv) = (vec![0f64; t * hd], vec![0f64; t * hd], vec![0f64; t * hd]);
                    crate::fcd_ops::attn_head_bwd_window(&q64, &k64, &v64, &do64, t, hd, hd, window, &mut dq, &mut dk, &mut dv);
                    for p in 0..t {
                        for c in 0..hd {
                            wao[(bi * t + p) * qdim + h * hd + c] = o[p * hd + c];
                            wdq[(bi * t + p) * qdim + h * hd + c] = dq[p * hd + c];
                            wdk[(bi * t + p) * kvdim + g * hd + c] += dk[p * hd + c];
                            wdv[(bi * t + p) * kvdim + g * hd + c] += dv[p * hd + c];
                        }
                    }
                }
            }
            let mut ao = vec![0f32; n * qdim];
            assert!(bake_attn_fwd(&q, &k, &v, b, t, nh, nkv, hd, window, &mut ao), "forward ran");
            let e = rel(&ao, &wao);
            eprintln!("window {window:?}: ao rel {e:.2e}");
            assert!(e < 1e-4, "attention forward rel {e:e}");
            let (mut dq, mut dk, mut dv) = (vec![0f32; n * qdim], vec![0f32; n * kvdim], vec![0f32; n * kvdim]);
            assert!(bake_attn_bwd(&q, &k, &v, &dao, b, t, nh, nkv, hd, window, &mut dq, &mut dk, &mut dv), "backward ran");
            for (name, got, want) in [("dq", &dq, &wdq), ("dk", &dk, &wdk), ("dv", &dv, &wdv)] {
                let e = rel(got, want);
                eprintln!("window {window:?}: {name} rel {e:.2e}");
                assert!(e < 1e-3, "attention backward {name} rel {e:e}");
            }
        }
    }


    /// The bake's dX and plain products on cuBLAS against the host's.
    #[test]
    fn bake_gemms_match_host() {
        if std::env::var("CMF_GPU").as_deref() != Ok("cuda") || ctx().is_none() {
            eprintln!("no CUDA — skip");
            return;
        }
        let (n, k, m) = (37usize, 96usize, 160usize);
        let dy = rnd(n * m, 21, 1.0);
        let w = rnd(m * k, 22, 1.0);
        let base = rnd(n * k, 23, 1.0);
        let mut want = base.clone();
        crate::fcd_ops::gemm_dx(&dy, &w, &mut want, n, k, m, None);
        let mut got = base.clone();
        assert!(bake_gemm_dx(&dy, &w, &mut got, n, k, m));
        let e = rel(&got, &want.iter().map(|&v| v as f64).collect::<Vec<_>>());
        eprintln!("gemm_dx rel {e:.2e}");
        assert!(e < 1e-3, "gemm_dx rel {e:e}");
        let x = rnd(n * k, 24, 1.0);
        let mut want = vec![0f32; n * m];
        crate::fcd_ops::gemm_nt_host(&x, &w, &mut want, n, k, m, None);
        let mut got = vec![0f32; n * m];
        assert!(bake_gemm_nt(&x, &w, &mut got, n, k, m));
        let e = rel(&got, &want.iter().map(|&v| v as f64).collect::<Vec<_>>());
        eprintln!("gemm_nt rel {e:.2e}");
        assert!(e < 1e-3, "gemm_nt rel {e:e}");
    }
}
