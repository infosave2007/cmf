//! NVG Ť on CUDA: the layer-streamed distillation trainer of `nvg_train.rs`
//! (Qwen3.5 family: GatedDeltaNet + gated full attention + SwiGLU) on this
//! backend, and the forward the NVG conversion calibrates with.
//!
//! The step is the wgpu trainer's — teacher top-k, student forward with
//! checkpointed layer inputs, `mean_t KL(p_T,topk ‖ p_S)` through the frozen
//! head, a reverse sweep that recomputes each layer from its checkpoint,
//! Adam on the metric (binary low-rank h, ℓ, g and every RMSNorm gain) and,
//! with sign training, one straight-through latent per sign updated right
//! after its path's backward. What differs is how it runs:
//!
//! * every product is a cuBLAS GEMM on the tensor cores over f16 operands
//!   with f32 accumulation — activations staged by one elementwise launch
//!   (column factor and gradient scale folded in), sign bitmaps expanded to
//!   ±1 f16 planes right before use (no resident plane: the bits and the
//!   latents are the only per-sign state), frozen weights as f16 planes;
//! * the GDN recurrence and its BPTT are the bake's kernels (`bake.cu`) with
//!   ordered reductions, attention is batched GEMMs and a causal softmax;
//! * every trainable parameter lives in one concatenated buffer with its
//!   gradient and Adam moments: the optimizer and the gradient norm are one
//!   launch each;
//! * column reductions run in two ordered passes and the sign kernels are
//!   built without fused multiply-adds, so a run is bit-reproducible and a
//!   sign update is bit for bit `train_sign`'s host mirror.
//!
//! Layout, snapshot (`nvg_train::Snap`) and state-file order are the wgpu
//! trainer's, so a snapshot or a saved state moves between the backends.

use super::*;
use crate::gpu_wgpu::train_sign::{self as ts, SignInit, SignOpt, SignStepCfg};
use crate::nvg_train::{Geom, Snap, StepStats, deq, load_threads};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use std::collections::HashMap;
use std::sync::Mutex;

const NT_SRC: &str = include_str!("nvgt.cu");
const NT_SIGN_SRC: &str = include_str!("nvgt_sign.cu");
const NT_BK_SRC: &str = include_str!("bake.cu");

/// GDN head width the kernels are compiled for (Qwen3.5/3.8).
const DKV: usize = 128;

#[derive(Clone, Copy)]
struct F {
    conv_fwd: cs::CUfunction,
    conv_bwd: cs::CUfunction,
    gdn_fwd: cs::CUfunction,
    attn_perm: cs::CUfunction,
    attn_unperm: cs::CUfunction,
    softmax_fwd: cs::CUfunction,
    softmax_bwd: cs::CUfunction,
    rmsnorm: cs::CUfunction,
    rmsnorm_bwd: cs::CUfunction,
    stage: cs::CUfunction,
    expand_s: cs::CUfunction,
    g_fold: cs::CUfunction,
    h2f_n: cs::CUfunction,
    colsum_part: cs::CUfunction,
    colsum_fin: cs::CUfunction,
    swiglu_fwd: cs::CUfunction,
    swiglu_bwd: cs::CUfunction,
    gate_fwd: cs::CUfunction,
    gate_bwd: cs::CUfunction,
    split_qg: cs::CUfunction,
    add: cs::CUfunction,
    qkr_fwd: cs::CUfunction,
    qkr_bwd: cs::CUfunction,
    gdn_rev: cs::CUfunction,
    gdn_qk_sum: cs::CUfunction,
    kl_rows: cs::CUfunction,
    topk_rows: cs::CUfunction,
    adam: cs::CUfunction,
    sumsq: cs::CUfunction,
    lat_init: cs::CUfunction,
    lat_init_f32: cs::CUfunction,
    g_rowsq: cs::CUfunction,
    loss_flag: cs::CUfunction,
    g_colsq: cs::CUfunction,
    g_colfin: cs::CUfunction,
    g_cmean: cs::CUfunction,
    sign_upd: cs::CUfunction,
    sign_upd_ref: cs::CUfunction,
    bits_diff: cs::CUfunction,
}

struct FBox(Result<F, String>);
// SAFETY: the function handles are used under the context's lock.
unsafe impl Send for FBox {}

static FUNCS: Mutex<Option<FBox>> = Mutex::new(None);

fn build(c: &Ctx) -> Res<F> {
    let src = format!("{NT_BK_SRC}\n{NT_SRC}");
    let m = compile(&src, vec![format!("-DDK={DKV}"), format!("-DDV={DKV}"), format!("-DGC={}", gdn_gc())], c.arch)?;
    // the sign kernels' arithmetic is the host mirrors': no contraction
    let ms = compile(NT_SIGN_SRC, vec!["-fmad=false".into()], c.arch)?;
    let f = |n: &str| func(m, n);
    let g = |n: &str| func(ms, n);
    Ok(F {
        conv_fwd: f("nt_conv_fwd")?,
        conv_bwd: f("nt_conv_bwd")?,
        gdn_fwd: f("nt_gdn_fwd")?,
        attn_perm: f("nt_attn_perm")?,
        attn_unperm: f("nt_attn_unperm")?,
        softmax_fwd: f("bk_softmax_fwd")?,
        softmax_bwd: f("bk_softmax_bwd")?,
        rmsnorm: f("bk_rmsnorm")?,
        rmsnorm_bwd: f("bk_rmsnorm_bwd")?,
        stage: f("nt_stage")?,
        expand_s: f("nt_expand_s")?,
        g_fold: f("nt_g_fold")?,
        h2f_n: f("nt_h2f_n")?,
        colsum_part: f("nt_colsum_part")?,
        colsum_fin: f("nt_colsum_fin")?,
        swiglu_fwd: f("nt_swiglu_fwd")?,
        swiglu_bwd: f("nt_swiglu_bwd")?,
        gate_fwd: f("nt_gate_fwd")?,
        gate_bwd: f("nt_gate_bwd")?,
        split_qg: f("nt_split_qg")?,
        add: f("nt_add")?,
        qkr_fwd: f("nt_qkr_fwd")?,
        qkr_bwd: f("nt_qkr_bwd")?,
        gdn_rev: f("nt_gdn_rev")?,
        gdn_qk_sum: f("nt_gdn_qk_sum")?,
        kl_rows: f("nt_kl_rows")?,
        topk_rows: f("nt_topk_rows")?,
        adam: f("nt_adam")?,
        sumsq: f("nt_sumsq")?,
        lat_init: g("nt_lat_init")?,
        lat_init_f32: g("nt_lat_init_f32")?,
        g_rowsq: g("nt_g_rowsq")?,
        loss_flag: g("nt_loss_flag")?,
        g_colsq: g("nt_g_colsq")?,
        g_colfin: g("nt_g_colfin")?,
        g_cmean: g("nt_g_cmean")?,
        sign_upd: g("nt_sign_upd")?,
        sign_upd_ref: g("nt_sign_upd_ref")?,
        bits_diff: g("nt_bits_diff")?,
    })
}

/// The context made current, its lock held, the kernels built.
struct Dev {
    c: &'static Ctx,
    f: F,
    _g: std::sync::MutexGuard<'static, State>,
}

fn enter() -> Result<Dev, String> {
    let c = ctx().ok_or("no CUDA device")?;
    let g = c.st.lock().unwrap();
    c.current()?;
    let mut fg = FUNCS.lock().unwrap();
    let f = fg.get_or_insert_with(|| FBox(build(c))).0.clone()?;
    Ok(Dev { c, f, _g: g })
}

/// The card is up and the trainer's kernels compile on it.
pub fn available() -> bool {
    match enter() {
        Ok(_) => true,
        Err(e) => {
            decline("nvg trainer", &e);
            false
        }
    }
}

// ───────────────────────────── profiler ─────────────────────────────

/// `CMF_NVG_PROF=1`: wall time per phase, the stream drained at each phase
/// change (the numbers are the card's, not the queue's).
pub mod prof {
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("CMF_NVG_PROF").is_some())
    }

    pub(super) static ACC: Mutex<Option<HashMap<&'static str, (f64, u64)>>> = Mutex::new(None);
    static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub(super) fn add(name: &'static str, secs: f64) {
        let mut g = ACC.lock().unwrap();
        let e = g.get_or_insert_with(HashMap::new).entry(name).or_insert((0.0, 0));
        e.0 += secs;
        e.1 += 1;
    }

    /// A training step ended (the per-step averages' denominator).
    pub fn step_end() {
        if on() {
            STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Forget what was measured so far (the teacher pass, the first evaluation).
    pub fn reset() {
        if on() {
            *ACC.lock().unwrap() = None;
            STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Print the phases' time per step.
    pub fn report() {
        if !on() {
            return;
        }
        let steps = STEPS.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
        let g = ACC.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let mut rows: Vec<(&&'static str, &(f64, u64))> = m.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        eprintln!("nvg-train (cuda) profile: {:.3} s/step in the phases below, {} steps", total / steps, steps as u64);
        for (name, (s, n)) in rows {
            eprintln!("  {name:<18} {:8.4} s/step  {:5.1} %  ({:.1} calls/step)", s / steps, 100.0 * s / total.max(1e-30), *n as f64 / steps);
        }
    }
}

struct Phase<'a> {
    d: &'a Dev,
    name: &'static str,
    t: std::time::Instant,
}

impl Dev {
    /// Time the launches queued until the guard drops (profiling only).
    fn phase(&self, name: &'static str) -> Option<Phase<'_>> {
        if !prof::on() {
            return None;
        }
        let _ = self.c.sync();
        Some(Phase { d: self, name, t: std::time::Instant::now() })
    }
}

impl Drop for Phase<'_> {
    fn drop(&mut self) {
        let _ = self.d.c.sync();
        prof::add(self.name, self.t.elapsed().as_secs_f64());
    }
}

// ───────────────────────────── device buffers ─────────────────────────────

/// A device allocation of this trainer (freed on drop, the stream drained first).
pub struct DBuf {
    ptr: u64,
    bytes: usize,
}

// SAFETY: a device address; every use goes through the context's lock.
unsafe impl Send for DBuf {}
unsafe impl Sync for DBuf {}

impl DBuf {
    /// `bytes` of zeroed device memory (the context current).
    fn zeroed(c: &Ctx, bytes: usize) -> Res<DBuf> {
        let bytes = bytes.max(64);
        let ptr = unsafe { cu::malloc_sync(bytes) }.map_err(|e| format!("device allocation of {bytes} bytes: {e}"))?;
        unsafe { cu::memset_d8_async(ptr, 0, bytes, c.stream).map_err(derr)? };
        Ok(DBuf { ptr, bytes })
    }
    fn floats(c: &Ctx, n: usize) -> Res<DBuf> {
        Self::zeroed(c, n * 4)
    }
    fn halves(c: &Ctx, n: usize) -> Res<DBuf> {
        Self::zeroed(c, n * 2)
    }
    fn from<T: Copy>(data: &[T]) -> Res<DBuf> {
        let bytes = std::mem::size_of_val(data).max(64);
        let ptr = unsafe { cu::malloc_sync(bytes) }.map_err(|e| format!("device allocation of {bytes} bytes: {e}"))?;
        if !data.is_empty() {
            unsafe { htod(ptr, data) }?;
        }
        Ok(DBuf { ptr, bytes })
    }
    fn at(&self, floats: usize) -> u64 {
        self.ptr + (floats * 4) as u64
    }
}

impl Drop for DBuf {
    fn drop(&mut self) {
        if let Some(c) = ctx() {
            let _ = c.current();
            let _ = c.sync();
            unsafe {
                let _ = cu::free_sync(self.ptr);
            }
        }
    }
}

/// Host memory into a device buffer other launches may still be reading:
/// the stream drained first.
fn put<T: Copy>(c: &Ctx, ptr: u64, data: &[T]) -> Res<()> {
    if data.is_empty() {
        return Ok(());
    }
    c.sync()?;
    unsafe { htod(ptr, data) }
}

fn get_f32(c: &Ctx, ptr: u64, n: usize) -> Res<Vec<f32>> {
    let mut v = vec![0f32; n];
    if n > 0 {
        unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), ptr, c.stream).map_err(derr)? };
    }
    c.sync()?;
    Ok(v)
}

fn get_u32(c: &Ctx, ptr: u64, n: usize) -> Res<Vec<u32>> {
    let mut v = vec![0u32; n];
    if n > 0 {
        unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), ptr, c.stream).map_err(derr)? };
    }
    c.sync()?;
    Ok(v)
}

fn copy_f(c: &Ctx, src: u64, dst: u64, floats: usize) -> Res<()> {
    if floats == 0 {
        return Ok(());
    }
    unsafe { cs::cuMemcpyDtoDAsync_v2(dst, src, floats * 4, c.stream) }.result().map_err(derr)
}

fn zero_f(c: &Ctx, ptr: u64, floats: usize) -> Res<()> {
    if floats == 0 {
        return Ok(());
    }
    unsafe { cu::memset_d8_async(ptr, 0, floats * 4, c.stream).map_err(derr) }
}

/// f32 → f16 bits over the load's threads.
fn to_f16(v: &[f32]) -> Vec<u16> {
    let mut h = vec![0u16; v.len()];
    let per = v.len().div_ceil(load_threads()).max(1 << 16);
    std::thread::scope(|s| {
        for (d, src) in h.chunks_mut(per).zip(v.chunks(per)) {
            s.spawn(move || d.iter_mut().zip(src).for_each(|(o, &f)| *o = f32_to_f16(f)));
        }
    });
    h
}

// ───────────────────────────── kernel launches ─────────────────────────────

impl Dev {
    /// An elementwise launch over `n` elements, 256 threads a block.
    unsafe fn el(&self, f: cs::CUfunction, n: usize, args: &mut [*mut c_void]) -> Res<()> {
        unsafe { self.c.launch(f, (n.div_ceil(256).max(1) as u32, 1), 256, 0, args) }
    }

    /// A launch over a `rows × cols` matrix: the row from the grid's y, the
    /// column from x (no division per element).
    unsafe fn el2(&self, f: cs::CUfunction, rows: usize, cols: usize, args: &mut [*mut c_void]) -> Res<()> {
        if rows > 65535 {
            return Err(format!("{rows} rows in one launch (at most 65535)"));
        }
        unsafe { self.c.launch(f, (cols.div_ceil(256).max(1) as u32, rows.max(1) as u32), 256, 0, args) }
    }

    /// `C[m][n] = α·op(A)·op(B) + β·C`, row-major operands (`t`: element (i,k)
    /// of `op(M)` is `M[k·ld + i]`). Read column-major, a row-major array is
    /// its transpose: the call is `Cᵀ = op(B)ᵀ·op(A)ᵀ`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mm(&self, m: usize, n: usize, k: usize, alpha: f32, a: u64, lda: usize, ta: bool, b: u64, ldb: usize, tb: bool, beta: f32, cp: u64, ldc: usize, mode: Gemm) -> Res<()> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let ev = self.kt_begin();
        let r = unsafe { self.c.gemm(tb, ta, n, m, k, alpha, b, ldb, a, lda, beta, cp, ldc, mode) };
        self.kt_end(ev, if mode == Gemm::Half { (1, "cublas gemm f16") } else { (2, "cublas gemm f32") });
        r
    }

    /// `count` attention GEMMs of one shape (`Ctx::gemm_batched`), timed under `CMF_NVG_KT`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mmb(&self, ta: bool, m: usize, n: usize, k: usize, alpha: f32, a: u64, lda: usize, sa: usize, b: u64, ldb: usize, sb: usize, cp: u64, ldc: usize, sc: usize, count: usize) -> Res<()> {
        let ev = self.kt_begin();
        let r = unsafe { self.c.gemm_batched(ta, m, n, k, alpha, a, lda, sa, b, ldb, sb, cp, ldc, sc, count, attn_mode()) };
        self.kt_end(ev, (3, "cublas gemm batched"));
        r
    }

    /// An event before a cuBLAS call while the per-kernel timer is on.
    fn kt_begin(&self) -> Option<cs::CUevent> {
        if !KT_ON.load(Ordering::Relaxed) {
            return None;
        }
        let a = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT).ok()?;
        let _ = unsafe { cu::event::record(a, self.c.stream) };
        Some(a)
    }

    fn kt_end(&self, a: Option<cs::CUevent>, key: (usize, &str)) {
        let Some(a) = a else { return };
        if let Ok(b) = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT) {
            let _ = unsafe { cu::event::record(b, self.c.stream) };
            KT_NAMES.lock().unwrap().get_or_insert_with(HashMap::new).entry(key.0).or_insert_with(|| key.1.to_string());
            KT_EVENTS.lock().unwrap().push(KtEvent(key.0, a, b));
        }
    }

    /// `dst` (f16 `[m][k]`) = `src` (f32) ⊙ `col[k]` · `scale`.
    unsafe fn stage(&self, src: u64, col: Option<u64>, dst: u64, m: usize, k: usize, scale: f32) -> Res<()> {
        let (ki, hc, colp) = (k as u32, col.is_some() as u32, col.unwrap_or(src));
        unsafe { self.el2(self.f.stage, m, k, kargs!(src, colp, dst, ki, scale, hc)) }
    }

    /// A sign bitmap `[rows][rp/32]` as the f16 plane `[rows][rp]` of
    /// `±sv[row]·mul` (the row's scale folded in).
    unsafe fn expand_s(&self, bits: u64, pl: u64, rows: usize, rp: usize, sv: u64, mul: f32) -> Res<()> {
        let (rpi, wpr) = (rp as u32, (rp / 32) as u32);
        unsafe { self.el2(self.f.expand_s, rows, rp, kargs!(bits, pl, sv, rpi, wpr, mul)) }
    }

    /// `acc[row] += Σ_k ±G[row][k]` (the row scale's gradient), then
    /// `G[row][k] *= sv[row]` (`G`: `[rows][rp]`, the signs' gradient before
    /// their row scale).
    unsafe fn g_fold(&self, g: u64, bits: u64, sv: u64, acc: u64, rows: usize, rp: usize) -> Res<()> {
        let (rpi, wpr) = (rp as u32, (rp / 32) as u32);
        unsafe { self.c.launch(self.f.g_fold, (rows as u32, 1), 256, 0, kargs!(g, bits, sv, acc, rpi, wpr)) }
    }

    /// `out[k] (+)= Σ_rows a·b` (mode 0), `a·b·f[row]` (1) or `a` (2), two
    /// ordered passes through `part` (≥ 256·k floats).
    #[allow(clippy::too_many_arguments)]
    unsafe fn colsum(&self, a: u64, b: u64, f: u64, out: u64, m: usize, k: usize, mode: u32, acc: bool, part: u64) -> Res<()> {
        let rpb = 64usize.max(m.div_ceil(256));
        let nblk = m.div_ceil(rpb).max(1);
        let (mi, ki, rpbi, nblki, acci) = (m as u32, k as u32, rpb as u32, nblk as u32, acc as u32);
        unsafe {
            self.c.launch(self.f.colsum_part, (k.div_ceil(256).max(1) as u32, nblk as u32), 256, 0, kargs!(a, b, f, part, mi, ki, rpbi, mode))?;
            self.c.launch(self.f.colsum_fin, (k.div_ceil(256).max(1) as u32, 1), 256, 0, kargs!(part, out, ki, nblki, acci))
        }
    }

    unsafe fn add(&self, a: u64, b: u64, y: u64, n: usize) -> Res<()> {
        let ni = n as u64;
        unsafe { self.el(self.f.add, n, kargs!(a, b, y, ni)) }
    }

    /// RMSNorm of `n` rows of width `d`: y, and 1/rms per row into `inv`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rms_fwd(&self, x: u64, w: u64, y: u64, inv: u64, n: usize, d: usize, eps: f32, gemma: bool) -> Res<()> {
        let (di, gi) = (d as i32, gemma as i32);
        unsafe { self.c.launch(self.f.rmsnorm, (n as u32, 1), 256, 0, kargs!(x, w, y, inv, di, eps, gi)) }
    }

    /// Its backward into `dx` (overwritten) and the gain gradient into `dw` (accumulated).
    #[allow(clippy::too_many_arguments)]
    unsafe fn rms_bwd(&self, x: u64, w: u64, inv: u64, dy: u64, dx: u64, dw: u64, n: usize, d: usize, gemma: bool, part: u64) -> Res<()> {
        let (di, gi, zero) = (d as i32, gemma as i32, 0i32);
        unsafe {
            self.c.launch(self.f.rmsnorm_bwd, (n as u32, 1), 256, 0, kargs!(x, w, inv, dy, dx, di, gi, zero))?;
            self.colsum(dy, x, inv, dw, n, d, 1, true, part)
        }
    }
}

// ───────────────────────────── the net ─────────────────────────────

/// A trainable f32 vector: its range in the concatenated parameter buffers.
#[derive(Clone, Copy, Debug)]
pub struct Param {
    pub off: usize,
    pub n: usize,
}

/// Optimizer state of one sign matrix (`rows × rp`, live ranks `r_true`).
pub struct SignMat {
    pub rows: usize,
    pub r_true: usize,
    pub rp: usize,
    pub id: u32,
    pub opt: SignOpt,
    lat: DBuf,
    mom: Option<DBuf>,
    v: Option<DBuf>,
    vrow: DBuf,
    vcol: DBuf,
    bits0: Option<DBuf>,
}

impl SignMat {
    fn wpr(&self) -> usize {
        self.rp / 32
    }
    pub fn live(&self) -> u64 {
        (self.rows * self.r_true) as u64
    }
    fn lat_words(&self) -> usize {
        match self.opt {
            SignOpt::Ref => self.rows * self.rp,
            _ => self.rows * self.rp / 2,
        }
    }
}

struct SignPath {
    u: SignMat,
    v: SignMat,
}

/// One binary low-rank path: the sign bitmaps and the indices of its h, ℓ, g
/// parameters. `mh`, `mg`: the powers of two its h and g ride the sign planes
/// with ([`plane_mul`], from the values at load).
struct Path {
    bits_u: DBuf,
    bits_v: DBuf,
    h: usize,
    l: usize,
    g: usize,
    mh: f32,
    mg: f32,
    sign: Option<SignPath>,
}

/// The power of two that brings the largest magnitude of `v` into (0.5, 1]:
/// a scale vector is folded into an f16 sign plane times this (far from the
/// f16 range's ends whatever the tensor's units; training moves a scale by a
/// small factor of its start).
fn plane_mul(v: &[f32]) -> f32 {
    let mx = v.iter().fold(0f32, |a, x| if x.is_finite() { a.max(x.abs()) } else { a });
    if mx > 0.0 { 2f32.powi(-(mx.log2().ceil() as i32).clamp(-60, 60)) } else { 1.0 }
}

/// A linear layer: binary low-rank (trainable metric), a frozen f16 plane,
/// a small frozen f32 matrix, or not on the device (outside the net's range).
enum Lin {
    Blr { m: usize, n: usize, r: usize, role: &'static str, paths: Vec<Path> },
    Plane { w: DBuf, m: usize, k: usize },
    Dense { w: DBuf, m: usize, k: usize },
    Absent { m: usize },
}

impl Lin {
    fn m(&self) -> usize {
        match self {
            Lin::Blr { m, .. } | Lin::Plane { m, .. } | Lin::Dense { m, .. } | Lin::Absent { m } => *m,
        }
    }
}

struct GdnPrm {
    a_log: DBuf,
    dt_bias: DBuf,
    norm: DBuf,
    conv: DBuf,
}

impl GdnPrm {
    fn ptrs(&self) -> GdnP {
        GdnP { a_log: self.a_log.ptr, dt_bias: self.dt_bias.ptr, norm: self.norm.ptr, conv: self.conv.ptr }
    }
}

enum Mix {
    Gdn { qkv: Lin, z: Lin, a: Lin, b: Lin, out: Lin, prm: Option<GdnPrm> },
    Attn { q: Lin, k: Lin, v: Lin, o: Lin, qn: usize, kn: usize },
}

struct Layer {
    ln1: usize,
    ln2: usize,
    mix: Mix,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// The model on the card.
pub struct Net {
    pub geom: Geom,
    /// The layers on the device (`first..end`).
    pub first: usize,
    pub end: usize,
    layers: Vec<Layer>,
    final_norm: usize,
    /// The frozen head as an f16 plane `[vocab][hs]` (a net that runs to the last layer).
    head: Option<DBuf>,
    /// Embedding rows on the host (f16 bits), gathered per step (a net that starts at layer 0).
    embed: Vec<u16>,
    params: Vec<Param>,
    total: usize,
    /// Values, gradients, Adam moments, mask (0 = frozen entry) and relative
    /// floor (0 = plain steps) of every parameter, concatenated.
    pw: DBuf,
    pg: DBuf,
    pm1: DBuf,
    pm2: DBuf,
    pmask: DBuf,
    prelf: DBuf,
    /// Tensor names of the norm gains, by parameter index (export).
    norm_names: Vec<(String, usize)>,
    pub n_sign_mats: u32,
    bwd: bool,
}

/// Host side of the concatenated parameters while a net loads.
#[derive(Default)]
struct PB {
    w: Vec<f32>,
    mask: Vec<f32>,
    relf: Vec<f32>,
    params: Vec<Param>,
}

impl PB {
    /// A parameter; `metric`: relative steps floored at 5 % of the live values' RMS.
    fn add(&mut self, v: &[f32], mask: Option<&[f32]>, metric: bool) -> usize {
        let mut rel = 0f32;
        if metric {
            let (mut ss, mut n) = (0f64, 0usize);
            for (i, &x) in v.iter().enumerate() {
                if mask.is_none_or(|m| m[i] != 0.0) {
                    ss += (x as f64) * (x as f64);
                    n += 1;
                }
            }
            let rms = (ss / n.max(1) as f64).sqrt() as f32;
            rel = if rms > 0.0 { 0.05 * rms } else { 0.0 };
        }
        let off = self.w.len();
        self.w.extend_from_slice(v);
        match mask {
            Some(m) => self.mask.extend_from_slice(m),
            None => self.mask.extend(std::iter::repeat_n(1.0, v.len())),
        }
        self.relf.extend(std::iter::repeat_n(rel, v.len()));
        self.params.push(Param { off, n: v.len() });
        self.params.len() - 1
    }
}

/// The kernels' sign parameters (`NtSP` of `nvgt_sign.cu`).
#[repr(C)]
#[derive(Clone, Copy)]
struct SP {
    rows: u32,
    r: u32,
    rp: u32,
    wpr: u32,
    id: u32,
    seed: u32,
    step: u32,
    slot: u32,
    eta: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    clampv: f32,
    hyst: f32,
    kss: f32,
    bc2: f32,
    bc1: f32,
    tau: f32,
    chunks: u32,
    pad0: u32,
}

fn sp(mat: &SignMat, cfg: Option<&SignStepCfg>, slot: u32, tau: f32, seed: u32) -> SP {
    let (eta, step, b1, b2, eps, clampv, hyst, kss, bc1, bc2, seed) = match cfg {
        Some(c) => {
            let bc1 = 1.0 - (c.b1 as f64).powi(c.step as i32);
            let bc2 = 1.0 - (c.b2 as f64).powi(c.step as i32);
            (c.eta, c.step, c.b1, c.b2, c.eps, c.clamp, c.hyst, c.k_ss, bc1 as f32, bc2 as f32, c.seed)
        }
        None => (0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, seed),
    };
    SP {
        rows: mat.rows as u32,
        r: mat.r_true as u32,
        rp: mat.rp as u32,
        wpr: mat.wpr() as u32,
        id: mat.id,
        seed,
        step,
        slot,
        eta,
        b1,
        b2,
        eps,
        clampv,
        hyst,
        kss,
        bc2,
        bc1,
        tau,
        chunks: mat.rows.div_ceil(512) as u32,
        pad0: 0,
    }
}

/// Sign state for the bitmap `bits` (`[rows][rp/32]`), latents drawn from it.
fn init_mat(d: &Dev, bits: &DBuf, rows: usize, r_true: usize, rp: usize, init: &SignInit, id: u32) -> Res<SignMat> {
    let c = d.c;
    let opt = init.opt;
    let lat_words = if opt == SignOpt::Ref { rows * rp } else { rows * rp / 2 };
    let words = rows * rp / 32;
    let mat = SignMat {
        rows,
        r_true,
        rp,
        id,
        opt,
        lat: DBuf::floats(c, lat_words)?,
        mom: if opt != SignOpt::Fact { Some(DBuf::floats(c, lat_words)?) } else { None },
        v: if opt == SignOpt::Ref { Some(DBuf::floats(c, rows * rp)?) } else { None },
        vrow: DBuf::floats(c, rows)?,
        vcol: DBuf::floats(c, rp)?,
        bits0: if init.keep_bits0 {
            let b = DBuf::floats(c, words)?;
            copy_f(c, bits.ptr, b.ptr, words)?;
            Some(b)
        } else {
            None
        },
    };
    let p = sp(&mat, None, 0, init.tau, init.seed);
    let f = if opt == SignOpt::Ref { d.f.lat_init_f32 } else { d.f.lat_init };
    unsafe { d.el(f, words, kargs!(p, bits.ptr, mat.lat.ptr))? };
    Ok(mat)
}

/// Parse a binary low-rank payload (dtype 19, docs/BLR_NVG.md layout) into
/// device paths, the rank padded to a multiple of 128 with ℓ = 0 (masked) —
/// the wgpu trainer's layout.
fn load_blr(d: &Dev, bytes: &[u8], role: &'static str, sign: Option<&SignInit>, next_id: &mut u32, pb: &mut PB) -> Res<Lin> {
    if bytes.len() < 16 || bytes[0] != 0x4C {
        return Err(format!("{role}: not a binary low-rank payload"));
    }
    let np = bytes[2] as usize;
    let r = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let rp = r.div_ceil(128) * 128;
    let wpr = rp / 32;
    let mut off = 16;
    let f16s = |b: &[u8], k: usize| -> Vec<f32> { (0..k).map(|i| f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))).collect() };
    let mut paths = Vec::new();
    for _ in 0..np {
        let h = f16s(&bytes[off..], m);
        off += 2 * m;
        let g = f16s(&bytes[off..], n);
        off += 2 * n;
        let mut l = f16s(&bytes[off..], r);
        off += 2 * r;
        let ub = &bytes[off..off + m * r / 8];
        off += m * r / 8;
        let vb = &bytes[off..off + r * n / 8];
        off += r * n / 8;
        let mut bu = vec![0u32; m * wpr];
        for i in 0..m {
            let row = &ub[i * r / 8..(i + 1) * r / 8];
            for (b, &byte) in row.iter().enumerate() {
                bu[i * wpr + b / 4] |= (byte as u32) << (8 * (b % 4));
            }
        }
        let mut bv = vec![0u32; n * wpr];
        for k in 0..r {
            let row = &vb[k * n / 8..(k + 1) * n / 8];
            let (w, bit) = (k / 32, 1u32 << (k % 32));
            for (jb, &byte) in row.iter().enumerate() {
                if byte == 0 {
                    continue;
                }
                for q in 0..8 {
                    if (byte >> q) & 1 == 1 {
                        bv[(jb * 8 + q) * wpr + w] |= bit;
                    }
                }
            }
        }
        l.resize(rp, 0.0);
        let mask: Vec<f32> = (0..rp).map(|k| if k < r { 1.0 } else { 0.0 }).collect();
        let (bits_u, bits_v) = (DBuf::from(&bu)?, DBuf::from(&bv)?);
        let (ph, pl, pg) = (pb.add(&h, None, true), pb.add(&l, Some(&mask), true), pb.add(&g, None, true));
        let sp = match sign {
            Some(init) => {
                let u = init_mat(d, &bits_u, m, r, rp, init, *next_id)?;
                let v = init_mat(d, &bits_v, n, r, rp, init, *next_id + 1)?;
                *next_id += 2;
                Some(SignPath { u, v })
            }
            None => None,
        };
        paths.push(Path { bits_u, bits_v, h: ph, l: pl, g: pg, mh: plane_mul(&h), mg: plane_mul(&g), sign: sp });
    }
    Ok(Lin::Blr { m, n, r: rp, role, paths })
}

fn load_lin(d: &Dev, model: &CmfModel, name: &str, role: &'static str, small: bool, sign: Option<&SignInit>, next_id: &mut u32, pb: &mut PB) -> Res<Lin> {
    let e = model.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.dtype as u8 == 19 {
        return load_blr(d, model.entry_bytes(e), role, sign, next_id, pb);
    }
    let (v, shape) = deq(model, name)?;
    let (rows, cols) = (shape[0], shape[1]);
    if small {
        Ok(Lin::Dense { w: DBuf::from(&v)?, m: rows, k: cols })
    } else {
        Ok(Lin::Plane { w: DBuf::from(&to_f16(&v))?, m: rows, k: cols })
    }
}

impl Net {
    /// Device copy of `model`: its binary low-rank tensors as such (trainable
    /// metric), every other linear a frozen f16 plane. `bwd`: the net is
    /// stepped (the workspace gets the backward's buffers). `sign`: give
    /// every binary low-rank path sign latents.
    pub fn load(model: &CmfModel, bwd: bool, sign: Option<&SignInit>) -> Result<Self, String> {
        let nl = model.arch().num_layers;
        Self::load_range(model, bwd, sign, 0..nl)
    }

    /// Layers `range` only, forward, without the head: the calibration
    /// forward over the layers a conversion is working on.
    pub fn load_calib(model: &CmfModel, range: std::ops::Range<usize>) -> Result<Self, String> {
        Self::load_with(model, false, None, range, false)
    }

    /// Layers `range` only (the others stay off the card).
    pub fn load_range(model: &CmfModel, bwd: bool, sign: Option<&SignInit>, range: std::ops::Range<usize>) -> Result<Self, String> {
        Self::load_with(model, bwd, sign, range, true)
    }

    fn load_with(model: &CmfModel, bwd: bool, sign: Option<&SignInit>, range: std::ops::Range<usize>, with_head: bool) -> Result<Self, String> {
        let d = enter()?;
        let mut geom = Geom::from_model(model)?;
        let mut pb = PB::default();
        let mut norm_names = Vec::new();
        let mut norm = |name: String, pb: &mut PB| -> Result<usize, String> {
            let i = pb.add(&deq(model, &name)?.0, None, false);
            norm_names.push((name, i));
            Ok(i)
        };
        let final_norm = norm("model.norm.weight".into(), &mut pb)?;
        let mut layers = Vec::with_capacity(geom.nl);
        let mut next_id = 0u32;
        for li in 0..geom.nl {
            let p = format!("model.layers.{li}.");
            let here = range.contains(&li);
            let absent = |n: &str| -> Result<Lin, String> {
                let e = model.tensor(&format!("{p}{n}")).ok_or_else(|| format!("missing tensor {p}{n}"))?;
                Ok(Lin::Absent { m: e.shape[0] })
            };
            let mut lin = |n: &str, role: &'static str, small: bool, pb: &mut PB| -> Result<Lin, String> {
                if !here {
                    return absent(n);
                }
                load_lin(&d, model, &format!("{p}{n}"), role, small, sign, &mut next_id, pb)
            };
            let ln1 = norm(format!("{p}input_layernorm.weight"), &mut pb)?;
            let ln2 = norm(format!("{p}post_attention_layernorm.weight"), &mut pb)?;
            let mix = if geom.gdn_layer[li] {
                let qkv = lin("linear_attn.in_proj_qkv.weight", "gdn.qkv", false, &mut pb)?;
                let z = lin("linear_attn.in_proj_z.weight", "gdn.z", false, &mut pb)?;
                let a = lin("linear_attn.in_proj_a.weight", "gdn.ab", true, &mut pb)?;
                let b = lin("linear_attn.in_proj_b.weight", "gdn.ab", true, &mut pb)?;
                let out = lin("linear_attn.out_proj.weight", "gdn.out", false, &mut pb)?;
                let prm = if here {
                    Some(GdnPrm {
                        a_log: DBuf::from(&deq(model, &format!("{p}linear_attn.A_log"))?.0)?,
                        dt_bias: DBuf::from(&deq(model, &format!("{p}linear_attn.dt_bias"))?.0)?,
                        norm: DBuf::from(&deq(model, &format!("{p}linear_attn.norm.weight"))?.0)?,
                        conv: DBuf::from(&deq(model, &format!("{p}linear_attn.conv1d.weight"))?.0)?,
                    })
                } else {
                    None
                };
                Mix::Gdn { qkv, z, a, b, out, prm }
            } else {
                let q = lin("self_attn.q_proj.weight", "attn.q", false, &mut pb)?;
                let k = lin("self_attn.k_proj.weight", "attn.k", false, &mut pb)?;
                let v = lin("self_attn.v_proj.weight", "attn.v", false, &mut pb)?;
                let o = lin("self_attn.o_proj.weight", "attn.o", false, &mut pb)?;
                let qn = norm(format!("{p}self_attn.q_norm.weight"), &mut pb)?;
                let kn = norm(format!("{p}self_attn.k_norm.weight"), &mut pb)?;
                Mix::Attn { q, k, v, o, qn, kn }
            };
            let gate = lin("mlp.gate_proj.weight", "mlp.gate", false, &mut pb)?;
            let up = lin("mlp.up_proj.weight", "mlp.up", false, &mut pb)?;
            let down = lin("mlp.down_proj.weight", "mlp.down", false, &mut pb)?;
            layers.push(Layer { ln1, ln2, mix, gate, up, down });
        }
        // per-layer MLP widths (a pruned model): the workspace takes the widest
        geom.inter = layers.iter().map(|l| l.gate.m()).max().unwrap_or(geom.inter);
        let embed = if range.start == 0 { to_f16(&deq(model, "model.embed_tokens.weight")?.0) } else { Vec::new() };
        let head = if with_head && range.end == geom.nl {
            let name = if model.tensor("lm_head.weight").is_some() { "lm_head.weight" } else { "model.embed_tokens.weight" };
            Some(DBuf::from(&to_f16(&deq(model, name)?.0))?)
        } else {
            None
        };
        let total = pb.w.len();
        let c = d.c;
        let net = Net {
            geom,
            first: range.start,
            end: range.end,
            layers,
            final_norm,
            head,
            embed,
            params: pb.params,
            total,
            pw: DBuf::from(&pb.w)?,
            pg: DBuf::floats(c, total)?,
            pm1: DBuf::floats(c, total)?,
            pm2: DBuf::floats(c, total)?,
            pmask: DBuf::from(&pb.mask)?,
            prelf: DBuf::from(&pb.relf)?,
            norm_names,
            n_sign_mats: next_id,
            bwd,
        };
        c.sync()?;
        Ok(net)
    }

    fn w(&self, i: usize) -> u64 {
        self.pw.at(self.params[i].off)
    }
    fn g(&self, i: usize) -> u64 {
        self.pg.at(self.params[i].off)
    }

    /// Every binary low-rank linear with its tensor name, in the wgpu
    /// trainer's order (`nvg_train::Net::blr_lins`).
    fn blr_lins(&self) -> Vec<(String, &Lin)> {
        let mut v: Vec<(String, &Lin)> = Vec::new();
        for (li, ly) in self.layers.iter().enumerate() {
            let mut list: Vec<(&str, &Lin)> = match &ly.mix {
                Mix::Gdn { qkv, z, out, .. } => vec![("linear_attn.in_proj_qkv.weight", qkv), ("linear_attn.in_proj_z.weight", z), ("linear_attn.out_proj.weight", out)],
                Mix::Attn { q, k, v: vv, o, .. } => vec![("self_attn.q_proj.weight", q), ("self_attn.k_proj.weight", k), ("self_attn.v_proj.weight", vv), ("self_attn.o_proj.weight", o)],
            };
            list.extend([("mlp.gate_proj.weight", &ly.gate), ("mlp.up_proj.weight", &ly.up), ("mlp.down_proj.weight", &ly.down)]);
            for (n, l) in list {
                if matches!(l, Lin::Blr { .. }) {
                    v.push((format!("model.layers.{li}.{n}"), l));
                }
            }
        }
        v
    }

    /// Every sign matrix with its bitmap and counter slot, in [`Net::blr_lins`] order.
    fn sign_mats(&self) -> Vec<(&SignMat, &DBuf, u32)> {
        let mut v = Vec::new();
        for (_, lin) in self.blr_lins() {
            if let Lin::Blr { role, paths, .. } = lin {
                let rid = ts::role_id(role);
                for p in paths {
                    if let Some(sp) = &p.sign {
                        v.push((&sp.u, &p.bits_u, 2 * rid));
                        v.push((&sp.v, &p.bits_v, 2 * rid + 1));
                    }
                }
            }
        }
        v
    }

    /// Every path's bitmaps (U then V) with their word counts, in [`Net::blr_lins`] order.
    fn bitmaps(&self) -> Vec<(&DBuf, usize)> {
        let mut v = Vec::new();
        for (_, lin) in self.blr_lins() {
            if let Lin::Blr { m, n, r, paths, .. } = lin {
                for p in paths {
                    v.push((&p.bits_u, m * r / 32));
                    v.push((&p.bits_v, n * r / 32));
                }
            }
        }
        v
    }

    /// Width of layer `l`'s input of a calibration role (see [`CALIB_ROLES`]).
    pub fn calib_width(&self, l: usize, role: &str) -> Result<usize, String> {
        let g = &self.geom;
        match role {
            "in" | "mlp" => Ok(g.hs),
            "mixout" => Ok(if g.gdn_layer[l] { vd(g) } else { g.nh * g.hd }),
            "down" => Ok(self.layers[l].gate.m()),
            other => Err(format!("unknown calibration role {other}")),
        }
    }

    /// Live signs under training (Σ rows · true rank over every sign matrix).
    pub fn sign_live(&self) -> u64 {
        self.sign_mats().iter().map(|(m, _, _)| m.live()).sum()
    }

    /// Entries trained (the metric and the norm gains).
    pub fn trained_entries(&self) -> usize {
        self.total
    }

    /// Device bytes of the sign state (latents, moments, statistics, starting bits).
    pub fn sign_bytes(&self) -> u64 {
        let mut s = 0u64;
        for (m, _, _) in self.sign_mats() {
            s += (m.lat.bytes + m.vrow.bytes + m.vcol.bytes) as u64;
            s += m.mom.as_ref().map_or(0, |b| b.bytes) as u64 + m.v.as_ref().map_or(0, |b| b.bytes) as u64 + m.bits0.as_ref().map_or(0, |b| b.bytes) as u64;
        }
        s
    }
}

fn vd(g: &Geom) -> usize {
    g.nv * DKV
}
fn c_dim(g: &Geom) -> usize {
    2 * g.nk * DKV + vd(g)
}

// ───────────────────────────── workspace ─────────────────────────────

/// Top-k entries per token the head's scratch holds.
const K_MAX: usize = 256;

/// Activation and gradient buffers for `nb` windows of `t` tokens, shared by
/// every layer (one layer is live at a time), plus the checkpoints.
struct BlrScratch {
    a: DBuf,
    pu: DBuf,
    pv: DBuf,
}

/// A layer input staged in f16 once for every linear and path that reads it
/// (`done` is reset by each layer forward).
struct Xs {
    buf: u64,
    done: std::cell::Cell<bool>,
}

impl Xs {
    fn new(buf: u64) -> Xs {
        Xs { buf, done: std::cell::Cell::new(false) }
    }
}

pub struct Work {
    pub nb: usize,
    pub t: usize,
    ntok: usize,
    ckpt: Vec<DBuf>,
    b: HashMap<&'static str, DBuf>,
    /// Per role, per path: a `[ntok][rp]` and c `[ntok][m]` of its last forward.
    /// Per role and path: a (`[ntok][r]`, the forward's rank activations) and
    /// the two sign planes with their scales folded in (`[m][r]`, `[n][r]`
    /// f16), kept from a layer's forward for its backward.
    blr: HashMap<&'static str, Vec<BlrScratch>>,
    /// The f16 stagings of a layer's inputs: the first norm's output, the
    /// mixer core's output, the second norm's output, the MLP's activation.
    xs_n1: Xs,
    xs_mx: Xs,
    xs_n2: Xs,
    xs_hh: Xs,
    rope: DBuf,
    /// u32[64] sign counters (the CNT_* slots of `train_sign`).
    cnt: DBuf,
    head_chunk: usize,
    /// Sequences a GDN backward launch holds the state history of.
    gdn_per: usize,
    /// Device bytes of the sign scratch.
    pub sign_scratch: u64,
}

impl Work {
    pub fn new(net: &Net, nb: usize, t: usize) -> Result<Self, String> {
        let d = enter()?;
        let c = d.c;
        let g = &net.geom;
        let ntok = nb * t;
        let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
        let any_attn = (net.first..net.end).any(|l| !g.gdn_layer[l]);
        let any_gdn = (net.first..net.end).any(|l| g.gdn_layer[l]);
        let bwd = net.bwd;
        let mut b = HashMap::new();
        let mut z = |name: &'static str, n: usize| -> Res<()> {
            b.insert(name, DBuf::floats(c, n.max(1))?);
            Ok(())
        };
        for n in ["x", "n1", "x2", "n2", "mix"] {
            z(n, ntok * g.hs)?;
        }
        for n in ["inv1", "inv2", "invf", "loss"] {
            z(n, ntok)?;
        }
        for n in ["gg", "uu", "hh"] {
            z(n, ntok * g.inter)?;
        }
        if bwd {
            for n in ["dx", "dn1", "dx2", "dn2", "dmix"] {
                z(n, ntok * g.hs)?;
            }
            for n in ["dgg", "duu", "dhh"] {
                z(n, ntok * g.inter)?;
            }
        }
        if any_attn {
            let cnt = nb * g.nh;
            for (n, w) in [("qg", 2 * qd), ("q", qd), ("gate", qd), ("k", kd), ("v", kd), ("qr", qd), ("kr", kd), ("o", qd), ("og", qd)] {
                z(n, ntok * w)?;
            }
            z("qinv", ntok * g.nh)?;
            z("kinv", ntok * g.nkv)?;
            for n in ["qp", "kp", "vp", "op"] {
                z(n, cnt * t * g.hd)?;
            }
            for n in ["as", "ap", "apt"] {
                z(n, cnt * t * t)?;
            }
            if bwd {
                for (n, w) in [("dqg", 2 * qd), ("dq", qd), ("dgate", qd), ("dk", kd), ("dv", kd), ("dqr", qd), ("dkr", kd), ("do", qd), ("dog", qd), ("dwq", qd), ("dwk", kd)] {
                    z(n, ntok * w)?;
                }
                for n in ["dop", "dqp", "dkp", "dvp"] {
                    z(n, cnt * t * g.hd)?;
                }
                for n in ["adp", "ads", "adst"] {
                    z(n, cnt * t * t)?;
                }
            }
        }
        let mut gdn_per = 0usize;
        if any_gdn {
            let (cd, v) = (c_dim(g), vd(g));
            for (n, w) in [("raw", cd), ("pre", cd), ("cq", cd), ("zz", v), ("aa", g.nv), ("bb", g.nv), ("gout", v)] {
                z(n, ntok * w)?;
            }
            if bwd {
                for (n, w) in [("dgout", v), ("dcq", cd), ("draw", cd), ("dz", v), ("da", g.nv), ("db", g.nv), ("dqk", g.nv * 2 * DKV)] {
                    z(n, ntok * w)?;
                }
                // the BPTT's state history: t·dk·dv halves a (sequence, v-head)
                let cap = std::env::var("CMF_NVG_GDN_HIST_GB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16) << 30;
                let per_seq = t * DKV * DKV * 2 * g.nv;
                gdn_per = (cap / per_seq.max(1)).clamp(1, nb);
                z("hist", (gdn_per * g.nv * t * DKV * DKV).div_ceil(2))?;
                z("kvh", gdn_per * g.nv * t * DKV)?;
                z("oh", gdn_per * g.nv * t * DKV)?;
            }
        }
        // the binary low-rank scratch: a and the two planes per role and path,
        // the staging buffers for the largest instance
        let mut dims: HashMap<&'static str, (usize, usize, usize, usize)> = HashMap::new();
        let (mut mx_m, mut mx_n, mut mx_r) = (1usize, 1usize, 1usize);
        let (mut sg_rows, mut sg_rp) = (0usize, 0usize);
        let mut act_w = g.hs.max(g.inter).max(2 * qd).max(c_dim(g)).max(vd(g));
        for l in &net.layers[net.first..net.end] {
            let mut add = |lin: &Lin| {
                match lin {
                    Lin::Blr { m, n, r, role, paths, .. } => {
                        let e = dims.entry(*role).or_insert((0, 0, 0, 0));
                        *e = (e.0.max(*m), e.1.max(*r), e.2.max(paths.len()), e.3.max(*n));
                        mx_m = mx_m.max(*m);
                        mx_n = mx_n.max(*n);
                        mx_r = mx_r.max(*r);
                        // the signs' gradient: the row scales' gradients come out of it too
                        sg_rows = sg_rows.max(*m).max(*n);
                        sg_rp = sg_rp.max(*r);
                    }
                    Lin::Plane { m, k, .. } | Lin::Dense { m, k, .. } => act_w = act_w.max(*m).max(*k),
                    Lin::Absent { .. } => {}
                }
            };
            match &l.mix {
                Mix::Gdn { qkv, z, a, b, out, .. } => {
                    for x in [qkv, z, a, b, out] {
                        add(x);
                    }
                }
                Mix::Attn { q, k, v, o, .. } => {
                    for x in [q, k, v, o] {
                        add(x);
                    }
                }
            }
            add(&l.gate);
            add(&l.up);
            add(&l.down);
        }
        let mut blr = HashMap::new();
        for (role, (m, r, np, n)) in dims {
            let mut v = Vec::with_capacity(np);
            for _ in 0..np {
                v.push(BlrScratch { a: DBuf::floats(c, ntok * r)?, pu: DBuf::halves(c, m * r)?, pv: DBuf::halves(c, n * r)? });
            }
            blr.insert(role, v);
        }
        let wide = act_w.max(mx_m).max(mx_n).max(mx_r);
        let mut h = |name: &'static str, n: usize| -> Res<()> {
            b.insert(name, DBuf::halves(c, n.max(2))?);
            Ok(())
        };
        // f16 staging: s1 the widest operand (an output gradient), s2 / s3
        // rank-wide; a layer's four inputs (the two norms' outputs, the
        // mixer's core output, the MLP's activation) staged once each
        h("s1", ntok * wide)?;
        h("s2", ntok * mx_r)?;
        h("xs_n1", ntok * g.hs)?;
        h("xs_n2", ntok * g.hs)?;
        h("xs_mx", ntok * wide)?;
        h("xs_hh", ntok * wide)?;
        if bwd {
            h("s3", ntok * mx_r)?;
        }
        let mut z = |name: &'static str, n: usize| -> Res<()> {
            b.insert(name, DBuf::floats(c, n.max(1))?);
            Ok(())
        };
        // column-reduction partials: ≤ 256 row blocks of the widest reduction
        z("part", 256 * wide.max(g.hd))?;
        let mut sign_scratch = 0u64;
        if bwd {
            z("tmp_h", ntok * wide)?;
            z("t2", ntok * mx_r)?;
            if sg_rows > 0 {
                z("sg", sg_rows * sg_rp)?;
                z("spart", sg_rows.div_ceil(512) * sg_rp)?;
                z("scal", 4)?;
                sign_scratch = ((sg_rows * sg_rp + sg_rows.div_ceil(512) * sg_rp) * 4) as u64;
            }
        }
        // the head a chunk of rows at a time (the logits of 1024 rows are 1 GB at a 248k vocabulary)
        let head_chunk = 1024.min(ntok);
        if net.head.is_some() {
            z("logits", head_chunk * g.vocab)?;
            z("tid", ntok * K_MAX)?;
            z("tp", ntok * K_MAX)?;
            b.insert("dl16", DBuf::halves(c, if bwd { head_chunk * g.vocab } else { 2 })?);
        }
        b.insert("emb16", DBuf::halves(c, ntok * g.hs)?);
        let mut tab = Vec::with_capacity(t * g.half * 2);
        for pos in 0..t {
            for &f in &g.inv_freq {
                let a = pos as f64 * f;
                tab.push(a.cos() as f32);
                tab.push(a.sin() as f32);
            }
        }
        let xs = |n: &str| Xs::new(b[n].ptr);
        let (xs_n1, xs_mx, xs_n2, xs_hh) = (xs("xs_n1"), xs("xs_mx"), xs("xs_n2"), xs("xs_hh"));
        let w = Work {
            nb,
            t,
            ntok,
            xs_n1,
            xs_mx,
            xs_n2,
            xs_hh,
            ckpt: (0..=g.nl).map(|l| if l >= net.first && l <= net.end { DBuf::floats(c, ntok * g.hs) } else { DBuf::floats(c, 1) }).collect::<Res<Vec<_>>>()?,
            b,
            blr,
            rope: DBuf::from(&tab)?,
            cnt: DBuf::floats(c, ts::CNT_LEN)?,
            head_chunk,
            gdn_per,
            sign_scratch,
        };
        c.sync()?;
        Ok(w)
    }

    fn p(&self, n: &str) -> u64 {
        self.b[n].ptr
    }
}

// ───────────────────────────── linear ops ─────────────────────────────

/// `y = lin(x)` (`acc`: added — a small dense linear only). `xs`: the f16
/// staging of `x`, filled by the first linear that needs it.
unsafe fn lin_fwd(d: &Dev, net: &Net, w: &Work, lin: &Lin, x: u64, xs: &Xs, y: u64, acc: bool) -> Res<()> {
    let ntok = w.ntok;
    let staged = |k: usize| -> Res<u64> {
        if !xs.done.get() {
            unsafe { d.stage(x, None, xs.buf, ntok, k, 1.0)? };
            xs.done.set(true);
        }
        Ok(xs.buf)
    };
    unsafe {
        match lin {
            Lin::Blr { m, n, r, role, paths, .. } => {
                if acc {
                    return Err("accumulating forward of a binary low-rank linear".into());
                }
                let (m, n, r) = (*m, *n, *r);
                let sc = &w.blr[role];
                let xh = staged(n)?;
                for (pi, p) in paths.iter().enumerate() {
                    let (a, pu, pv) = (sc[pi].a.ptr, sc[pi].pu.ptr, sc[pi].pv.ptr);
                    // a = x·(diag(g) S_V);  y (+)= (a ⊙ ℓ)·(diag(h) S_U)ᵀ
                    d.expand_s(p.bits_v.ptr, pv, n, r, net.w(p.g), p.mg)?;
                    d.mm(ntok, r, n, 1.0 / p.mg, xh, n, false, pv, r, false, 0.0, a, r, Gemm::Half)?;
                    d.stage(a, Some(net.w(p.l)), w.p("s2"), ntok, r, 1.0)?;
                    d.expand_s(p.bits_u.ptr, pu, m, r, net.w(p.h), p.mh)?;
                    d.mm(ntok, m, r, 1.0 / p.mh, w.p("s2"), r, false, pu, r, true, if pi > 0 { 1.0 } else { 0.0 }, y, m, Gemm::Half)?;
                }
                Ok(())
            }
            Lin::Plane { w: pl, m, k } => {
                if acc {
                    return Err("accumulating forward of a plane linear".into());
                }
                let xh = staged(*k)?;
                d.mm(ntok, *m, *k, 1.0, xh, *k, false, pl.ptr, *k, true, 0.0, y, *m, Gemm::Half)
            }
            Lin::Dense { w: dw, m, k } => d.mm(ntok, *m, *k, 1.0, x, *k, false, dw.ptr, *k, true, if acc { 1.0 } else { 0.0 }, y, *m, Gemm::F32),
            Lin::Absent { .. } => Err("a linear outside the net's range".into()),
        }
    }
}

/// The sign update of one matrix from the gradient in `g` (`[rows][rp]`).
unsafe fn sign_update(d: &Dev, w: &Work, cfg: &SignStepCfg, mat: &SignMat, g: u64, bits: u64, slot: u32) -> Res<()> {
    let p = sp(mat, Some(cfg), slot, 0.0, 0);
    let (rows, rp) = (mat.rows, mat.rp);
    let words = rows * rp / 32;
    let cnt = w.cnt.ptr;
    unsafe {
        if mat.opt == SignOpt::Ref {
            let (mom, v) = (mat.mom.as_ref().ok_or("sign moments")?.ptr, mat.v.as_ref().ok_or("sign second moments")?.ptr);
            return d.el(d.f.sign_upd_ref, words, kargs!(p, g, mat.lat.ptr, mom, v, bits, cnt));
        }
        let (part, scal) = (w.p("spart"), w.p("scal"));
        d.c.launch(d.f.g_rowsq, (rows as u32, 1), 256, 0, kargs!(p, g, mat.vrow.ptr, cnt))?;
        d.c.launch(d.f.g_colsq, (rp.div_ceil(256) as u32, rows.div_ceil(512) as u32), 256, 0, kargs!(p, g, part))?;
        d.c.launch(d.f.g_colfin, (rp.div_ceil(256) as u32, 1), 256, 0, kargs!(p, part, mat.vcol.ptr, cnt))?;
        d.c.launch(d.f.g_cmean, (1, 1), 256, 0, kargs!(p, mat.vcol.ptr, scal, cnt))?;
        let (mom, has) = match &mat.mom {
            Some(m) => (m.ptr, 1u32),
            None => (mat.lat.ptr, 0u32),
        };
        d.el(d.f.sign_upd, words, kargs!(p, g, mat.lat.ptr, mom, mat.vrow.ptr, mat.vcol.ptr, bits, scal, cnt, has))
    }
}

/// Input gradient into `dx` (overwritten, or accumulated with `acc`), metric
/// gradients accumulated. A binary low-rank linear needs the scratch of its
/// own last forward (its rank activations, its two planes) and that
/// forward's staged input `xs`; with `sign` its sign latents and bits are
/// updated in the same pass, after its input gradient is formed from the old
/// bits.
///
/// With y = Σ_p diag(h)·S_U·diag(ℓ)·S_Vᵀ·diag(g)·x and t2 = (dy ⊙ h)·S_U:
/// dx = Σ_p (t2 ⊙ ℓ)·(diag(g) S_V)ᵀ, dℓ = Σ_rows a ⊙ t2, and with
/// G'_U = dyᵀ·(a ⊙ ℓ), G'_V = xᵀ·(t2 ⊙ ℓ): dh_i = Σ_k S_U[i][k]·G'_U[i][k],
/// dg_j = Σ_k S_V[j][k]·G'_V[j][k], and the signs' gradients are
/// diag(h)·G'_U and diag(g)·G'_V — the scales' gradients are row sums of the
/// matrices the sign training needs anyway, not passes over the activations.
#[allow(clippy::too_many_arguments)]
unsafe fn lin_bwd(d: &Dev, net: &Net, w: &Work, lin: &Lin, xs: &Xs, dy: u64, dx: u64, acc: bool, gs: f32, sign: Option<&SignStepCfg>) -> Res<()> {
    let ntok = w.ntok;
    unsafe {
        match lin {
            Lin::Blr { m, n, r, role, paths, .. } => {
                let (m, n, r) = (*m, *n, *r);
                if !xs.done.get() {
                    return Err("backward of a binary low-rank linear without its forward".into());
                }
                let sc = &w.blr[role];
                let part = w.p("part");
                let rid = ts::role_id(role);
                let (s1, s2, s3, t2, g) = (w.p("s1"), w.p("s2"), w.p("s3"), w.p("t2"), w.p("sg"));
                d.stage(dy, None, s1, ntok, m, gs)?;
                for (pi, p) in paths.iter().enumerate() {
                    let (a, pu, pv) = (sc[pi].a.ptr, sc[pi].pu.ptr, sc[pi].pv.ptr);
                    let (ph, pl, pg) = (net.w(p.h), net.w(p.l), net.w(p.g));
                    // t2 = dy·(diag(h) S_U);  dℓ += Σ a ⊙ t2
                    d.mm(ntok, r, m, 1.0 / (gs * p.mh), s1, m, false, pu, r, false, 0.0, t2, r, Gemm::Half)?;
                    d.colsum(a, t2, a, net.g(p.l), ntok, r, 0, true, part)?;
                    // dx (+)= (t2 ⊙ ℓ)·(diag(g) S_V)ᵀ
                    d.stage(t2, Some(pl), s2, ntok, r, gs)?;
                    d.mm(ntok, n, r, 1.0 / (gs * p.mg), s2, r, false, pv, r, true, if acc || pi > 0 { 1.0 } else { 0.0 }, dx, n, Gemm::Half)?;
                    let _ph = d.phase("sign");
                    // G'_U = dyᵀ·(a ⊙ ℓ): dh, then the U signs
                    d.stage(a, Some(pl), s3, ntok, r, 1.0)?;
                    d.mm(m, r, ntok, 1.0 / gs, s1, m, true, s3, r, false, 0.0, g, r, Gemm::Half)?;
                    d.g_fold(g, p.bits_u.ptr, ph, net.g(p.h), m, r)?;
                    if let (Some(cfg), Some(spth)) = (sign, p.sign.as_ref()) {
                        sign_update(d, w, cfg, &spth.u, g, p.bits_u.ptr, 2 * rid)?;
                    }
                    // G'_V = xᵀ·(t2 ⊙ ℓ): dg, then the V signs
                    d.mm(n, r, ntok, 1.0 / gs, xs.buf, n, true, s2, r, false, 0.0, g, r, Gemm::Half)?;
                    d.g_fold(g, p.bits_v.ptr, pg, net.g(p.g), n, r)?;
                    if let (Some(cfg), Some(spth)) = (sign, p.sign.as_ref()) {
                        sign_update(d, w, cfg, &spth.v, g, p.bits_v.ptr, 2 * rid + 1)?;
                    }
                }
                Ok(())
            }
            Lin::Plane { w: pl, m, k } => {
                let target = if acc { w.p("tmp_h") } else { dx };
                d.stage(dy, None, w.p("s1"), ntok, *m, gs)?;
                d.mm(ntok, *k, *m, 1.0 / gs, w.p("s1"), *m, false, pl.ptr, *k, false, 0.0, target, *k, Gemm::Half)?;
                if acc {
                    d.add(dx, target, dx, ntok * k)?;
                }
                Ok(())
            }
            Lin::Dense { w: dw, m, k } => d.mm(ntok, *k, *m, 1.0, dy, *m, false, dw.ptr, *k, false, if acc { 1.0 } else { 0.0 }, dx, *k, Gemm::F32),
            Lin::Absent { .. } => Err("a linear outside the net's range".into()),
        }
    }
}

// ───────────────────────────── attention core ─────────────────────────────

#[derive(Clone, Copy)]
struct AttnG {
    nb: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
}

/// The attention core's scratch: q, k, v, o and their gradients as
/// `[nb·nh][t][hd]`, the scores, probabilities and their gradients as
/// `[nb·nh][t][t]`.
struct AttnS {
    qp: u64,
    kp: u64,
    vp: u64,
    op: u64,
    sc: u64,
    p: u64,
    pt: u64,
    dop: u64,
    dqp: u64,
    dkp: u64,
    dvp: u64,
    dp: u64,
    ds: u64,
    dst: u64,
}

impl AttnS {
    fn of(w: &Work) -> AttnS {
        let g = |n: &str| w.b.get(n).map_or(0, |b| b.ptr);
        AttnS {
            qp: g("qp"),
            kp: g("kp"),
            vp: g("vp"),
            op: g("op"),
            sc: g("as"),
            p: g("ap"),
            pt: g("apt"),
            dop: g("dop"),
            dqp: g("dqp"),
            dkp: g("dkp"),
            dvp: g("dvp"),
            dp: g("adp"),
            ds: g("ads"),
            dst: g("adst"),
        }
    }
}

/// Exact causal GQA attention of rotated q `[n][nh·hd]`, k and v
/// `[n][nkv·hd]` into o `[n][nh·hd]` (n = nb·t; windows never mix): the
/// planes permuted to `[nb·nh][t][hd]` (k/v repeated over their group's
/// q-heads), `S = Q·Kᵀ/√hd`, a causal softmax, `O = P·V`. P and Pᵀ stay in
/// the scratch for the backward.
unsafe fn attn_core_fwd(d: &Dev, g: AttnG, qr: u64, kr: u64, v: u64, o: u64, s: &AttnS) -> Res<()> {
    let (t, hd, cnt, rep, n) = (g.t, g.hd, g.nb * g.nh, g.nh / g.nkv, g.nb * g.t);
    let (t_i, tu, nhu, nkvu, hdu, one, repu) = (t as i32, t as u32, g.nh as u32, g.nkv as u32, hd as u32, 1u32, rep as u32);
    let scale = 1.0 / (hd as f32).sqrt();
    let win = 0i32;
    unsafe {
        let pg = (hd.div_ceil(256) as u32, t as u32, cnt as u32);
        d.c.launch_3d(d.f.attn_perm, pg, 256, 0, kargs!(qr, s.qp, tu, nhu, nhu, hdu, one))?;
        d.c.launch_3d(d.f.attn_perm, pg, 256, 0, kargs!(kr, s.kp, tu, nhu, nkvu, hdu, repu))?;
        d.c.launch_3d(d.f.attn_perm, pg, 256, 0, kargs!(v, s.vp, tu, nhu, nkvu, hdu, repu))?;
        // S[p][q] = Σ_c K[q][c]·Q[p][c]; P = causal softmax; O = P·V
        d.mmb(true, t, t, hd, scale, s.kp, hd, t * hd, s.qp, hd, t * hd, s.sc, t, t * t, cnt)?;
        d.c.launch(d.f.softmax_fwd, ((cnt * t) as u32, 1), 128, 0, kargs!(s.sc, s.p, s.pt, t_i, win))?;
        d.mmb(false, hd, t, t, 1.0, s.vp, hd, t * hd, s.p, t, t * t, s.op, hd, t * hd, cnt)?;
        d.el2(d.f.attn_unperm, n, g.nh * hd, kargs!(s.op, o, tu, nhu, nhu, hdu, one))
    }
}

/// Its backward from `dout` `[n][nh·hd]` (the forward's planes and
/// probabilities still in the scratch): dq `[n][nh·hd]`, dk and dv
/// `[n][nkv·hd]` (the q-heads of a group summed), overwritten.
unsafe fn attn_core_bwd(d: &Dev, g: AttnG, dout: u64, dq: u64, dk: u64, dv: u64, s: &AttnS) -> Res<()> {
    let (t, hd, cnt, rep, n) = (g.t, g.hd, g.nb * g.nh, g.nh / g.nkv, g.nb * g.t);
    let (t_i, tu, nhu, nkvu, hdu, one, repu) = (t as i32, t as u32, g.nh as u32, g.nkv as u32, hd as u32, 1u32, rep as u32);
    let scale = 1.0 / (hd as f32).sqrt();
    unsafe {
        d.c.launch_3d(d.f.attn_perm, (hd.div_ceil(256) as u32, t as u32, cnt as u32), 256, 0, kargs!(dout, s.dop, tu, nhu, nhu, hdu, one))?;
        // dP[p][q] = Σ_c V[q][c]·dO[p][c];  dS = P ⊙ (dP − Σ P·dP)
        d.mmb(true, t, t, hd, 1.0, s.vp, hd, t * hd, s.dop, hd, t * hd, s.dp, t, t * t, cnt)?;
        d.c.launch(d.f.softmax_bwd, ((cnt * t) as u32, 1), 128, 0, kargs!(s.p, s.dp, s.ds, s.dst, t_i))?;
        // dQ = dS·K/√hd; dK = dSᵀ·Q/√hd; dV = Pᵀ·dO
        d.mmb(false, hd, t, t, scale, s.kp, hd, t * hd, s.ds, t, t * t, s.dqp, hd, t * hd, cnt)?;
        d.mmb(false, hd, t, t, scale, s.qp, hd, t * hd, s.dst, t, t * t, s.dkp, hd, t * hd, cnt)?;
        d.mmb(false, hd, t, t, 1.0, s.dop, hd, t * hd, s.pt, t, t * t, s.dvp, hd, t * hd, cnt)?;
        d.el2(d.f.attn_unperm, n, g.nh * hd, kargs!(s.dqp, dq, tu, nhu, nhu, hdu, one))?;
        d.el2(d.f.attn_unperm, n, g.nkv * hd, kargs!(s.dkp, dk, tu, nhu, nkvu, hdu, repu))?;
        d.el2(d.f.attn_unperm, n, g.nkv * hd, kargs!(s.dvp, dv, tu, nhu, nkvu, hdu, repu))
    }
}

// ───────────────────────────── GDN core ─────────────────────────────

#[derive(Clone, Copy)]
struct GdnG {
    nb: usize,
    t: usize,
    nv: usize,
    nk: usize,
    kk: usize,
    eps: f32,
}

impl GdnG {
    fn vd(&self) -> usize {
        self.nv * DKV
    }
    fn cd(&self) -> usize {
        2 * self.nk * DKV + self.vd()
    }
}

/// The device addresses of a GDN layer's frozen parameters.
#[derive(Clone, Copy)]
struct GdnP {
    a_log: u64,
    dt_bias: u64,
    norm: u64,
    conv: u64,
}

/// Row chunks of the state in the recurrence's kernels (`GC` of nvgt.cu): a
/// block is `DV·GC` threads, each holding `DK / GC` rows of a state column.
/// `CMF_NVG_GDN_GC` = 1, 2, 4 (default) or 8.
fn gdn_gc() -> usize {
    static G: OnceLock<usize> = OnceLock::new();
    *G.get_or_init(|| std::env::var("CMF_NVG_GDN_GC").ok().and_then(|v| v.parse::<usize>().ok()).filter(|v| [1, 2, 4, 8].contains(v)).unwrap_or(4))
}

/// Threads of a block of the recurrence's kernels.
fn gdn_block() -> u32 {
    (DKV * gdn_gc()) as u32
}

/// GDN forward from the raw projections (qkv `[n][c_dim]`, z `[n][vd]`, a
/// and b `[n][nv]`): conv + SiLU into `pre` / `cq`, the delta-rule
/// recurrence and the gated RMSNorm into `out` `[n][vd]`. With `keep` whose
/// history holds every window, the states stay there for [`gdn_core_bwd`].
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_core_fwd(d: &Dev, g: GdnG, p: GdnP, raw: u64, z: u64, a: u64, b: u64, pre: u64, cq: u64, out: u64, keep: Option<&GdnS>) -> Res<()> {
    let (cd, n) = (g.cd(), g.nb * g.t);
    let (ti, cdi, kki, nvi, nki, seq0, eps) = (g.t as i32, cd as i32, g.kk as i32, g.nv as i32, g.nk as i32, 0i32, g.eps);
    let (hist, kvh, oh) = match keep {
        Some(s) if s.per >= g.nb => (s.hist, s.kvh, s.oh),
        _ => (0u64, 0u64, 0u64),
    };
    unsafe {
        d.el2(d.f.conv_fwd, n, cd, kargs!(raw, p.conv, pre, cq, ti, cdi, kki))?;
        d.c.launch(d.f.gdn_fwd, (g.nv as u32, g.nb as u32), gdn_block(), 0, kargs!(cq, z, a, b, p.a_log, p.dt_bias, p.norm, out, hist, kvh, oh, ti, nvi, nki, eps, seq0))
    }
}

/// The BPTT's scratch: per-head q̂/k̂ through-gradients, and the state
/// history (f16), kv and o of the `per` sequences one launch holds.
struct GdnS {
    dcq: u64,
    dqk: u64,
    hist: u64,
    kvh: u64,
    oh: u64,
    per: usize,
}

impl GdnS {
    /// The workspace's (a net loaded for the backward).
    fn of(w: &Work) -> GdnS {
        GdnS { dcq: w.p("dcq"), dqk: w.p("dqk"), hist: w.p("hist"), kvh: w.p("kvh"), oh: w.p("oh"), per: w.gdn_per }
    }
}

/// GDN backward from `dout` `[n][vd]` (the forward's `pre` / `cq` and its
/// inputs): draw `[n][c_dim]`, dz `[n][vd]`, da and db `[n][nv]`, overwritten.
/// `kept`: the forward was run with this scratch and left its states there
/// (when the history holds every window); otherwise the windows are replayed
/// in groups of `s.per`.
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_core_bwd(d: &Dev, g: GdnG, p: GdnP, z: u64, a: u64, b: u64, pre: u64, cq: u64, dout: u64, draw: u64, dz: u64, da: u64, db: u64, s: &GdnS, kept: bool) -> Res<()> {
    let (cd, n) = (g.cd(), g.nb * g.t);
    let (ti, nvi, nki, cdi, kki, eps) = (g.t as i32, g.nv as i32, g.nk as i32, cd as i32, g.kk as i32, g.eps);
    let kept = kept && s.per >= g.nb;
    unsafe {
        let mut s0 = 0usize;
        while s0 < g.nb {
            let nbs = if kept { g.nb } else { s.per.max(1).min(g.nb - s0) };
            let seq0 = s0 as i32;
            let grid = (g.nv as u32, nbs as u32);
            if !kept {
                let no_out = 0u64;
                d.c.launch(d.f.gdn_fwd, grid, gdn_block(), 0, kargs!(cq, z, a, b, p.a_log, p.dt_bias, p.norm, no_out, s.hist, s.kvh, s.oh, ti, nvi, nki, eps, seq0))?;
            }
            d.c.launch(
                d.f.gdn_rev,
                grid,
                gdn_block(),
                0,
                kargs!(cq, z, a, b, p.a_log, p.dt_bias, p.norm, dout, s.dcq, dz, da, db, s.dqk, s.hist, s.kvh, s.oh, ti, nvi, nki, eps, seq0),
            )?;
            s0 += nbs;
        }
        d.el2(d.f.gdn_qk_sum, n, 2 * g.nk * DKV, kargs!(s.dqk, s.dcq, nvi, nki))?;
        d.el2(d.f.conv_bwd, n, cd, kargs!(pre, s.dcq, p.conv, draw, ti, cdi, kki))
    }
}

// ───────────────────────────── layer forward / backward ─────────────────────────────

/// The precision of the attention core's batched products (`CMF_NVG_CUDA_ATTN=f32`: plain SGEMM).
fn attn_mode() -> Gemm {
    static M: OnceLock<bool> = OnceLock::new();
    if *M.get_or_init(|| std::env::var("CMF_NVG_CUDA_ATTN").as_deref() == Ok("f32")) { Gemm::F32 } else { Gemm::Tf32 }
}

/// Forward of layer `l` from `x` to `y`; intermediates stay in the workspace
/// for the backward. `bptt`: the backward follows — a GDN layer keeps its
/// state history ([`layer_bwd`] is told the same).
unsafe fn layer_fwd(d: &Dev, net: &Net, w: &Work, l: usize, x: u64, y: u64, bptt: bool) -> Res<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    for xs in [&w.xs_n1, &w.xs_mx, &w.xs_n2, &w.xs_hh] {
        xs.done.set(false);
    }
    unsafe {
        d.rms_fwd(x, net.w(ly.ln1), w.p("n1"), w.p("inv1"), n, g.hs, g.eps, g.gemma)?;
        match &ly.mix {
            Mix::Gdn { qkv, z, a, b, out, prm } => {
                let prm = prm.as_ref().ok_or("GDN layer outside the net's range")?;
                lin_fwd(d, net, w, qkv, w.p("n1"), &w.xs_n1, w.p("raw"), false)?;
                lin_fwd(d, net, w, z, w.p("n1"), &w.xs_n1, w.p("zz"), false)?;
                lin_fwd(d, net, w, a, w.p("n1"), &w.xs_n1, w.p("aa"), false)?;
                lin_fwd(d, net, w, b, w.p("n1"), &w.xs_n1, w.p("bb"), false)?;
                let gg = GdnG { nb: w.nb, t: w.t, nv: g.nv, nk: g.nk, kk: g.kk, eps: g.eps };
                let sc = bptt.then(|| GdnS::of(w));
                gdn_core_fwd(d, gg, prm.ptrs(), w.p("raw"), w.p("zz"), w.p("aa"), w.p("bb"), w.p("pre"), w.p("cq"), w.p("gout"), sc.as_ref())?;
                lin_fwd(d, net, w, out, w.p("gout"), &w.xs_mx, w.p("mix"), false)?;
            }
            Mix::Attn { q, k, v, o, qn, kn } => {
                let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
                lin_fwd(d, net, w, q, w.p("n1"), &w.xs_n1, w.p("qg"), false)?;
                lin_fwd(d, net, w, k, w.p("n1"), &w.xs_n1, w.p("k"), false)?;
                lin_fwd(d, net, w, v, w.p("n1"), &w.xs_n1, w.p("v"), false)?;
                let (nhi, nkvi, hdi, halfi, ti, zero) = (g.nh as u32, g.nkv as u32, g.hd as u32, g.half as u32, w.t as u32, 0u32);
                let (eps, gem) = (g.eps, g.gemma as u32);
                d.el2(d.f.split_qg, n, qd, kargs!(w.p("qg"), w.p("q"), w.p("gate"), nhi, hdi, zero))?;
                let blk = g.hd.next_multiple_of(32) as u32;
                d.c.launch(d.f.qkr_fwd, ((n * g.nh) as u32, 1), blk, 0, kargs!(w.p("q"), net.w(*qn), w.p("qr"), w.p("qinv"), w.rope.ptr, nhi, hdi, halfi, ti, eps, gem))?;
                d.c.launch(d.f.qkr_fwd, ((n * g.nkv) as u32, 1), blk, 0, kargs!(w.p("k"), net.w(*kn), w.p("kr"), w.p("kinv"), w.rope.ptr, nkvi, hdi, halfi, ti, eps, gem))?;
                let ag = AttnG { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
                attn_core_fwd(d, ag, w.p("qr"), w.p("kr"), w.p("v"), w.p("o"), &AttnS::of(w))?;
                let nq = (n * qd) as u64;
                d.el(d.f.gate_fwd, n * qd, kargs!(w.p("o"), w.p("gate"), w.p("og"), nq))?;
                let _ = kd;
                lin_fwd(d, net, w, o, w.p("og"), &w.xs_mx, w.p("mix"), false)?;
            }
        }
        d.add(x, w.p("mix"), w.p("x2"), n * g.hs)?;
        d.rms_fwd(w.p("x2"), net.w(ly.ln2), w.p("n2"), w.p("inv2"), n, g.hs, g.eps, g.gemma)?;
        lin_fwd(d, net, w, &ly.gate, w.p("n2"), &w.xs_n2, w.p("gg"), false)?;
        lin_fwd(d, net, w, &ly.up, w.p("n2"), &w.xs_n2, w.p("uu"), false)?;
        let inter = ly.gate.m();
        let ni = (n * inter) as u64;
        d.el(d.f.swiglu_fwd, n * inter, kargs!(w.p("gg"), w.p("uu"), w.p("hh"), ni))?;
        lin_fwd(d, net, w, &ly.down, w.p("hh"), &w.xs_hh, w.p("mix"), false)?;
        d.add(w.p("x2"), w.p("mix"), y, n * g.hs)
    }
}

/// Backward of layer `l` (its forward just recomputed from `x`): `dy` is the
/// gradient at the layer output; `dx` gets the gradient at its input.
#[allow(clippy::too_many_arguments)]
unsafe fn layer_bwd(d: &Dev, net: &Net, w: &Work, l: usize, x: u64, dy: u64, dx: u64, gs: f32, sg: Option<&SignStepCfg>) -> Res<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    let part = w.p("part");
    unsafe {
        // MLP: y = x2 + down(swiglu(gate(n2), up(n2)))
        lin_bwd(d, net, w, &ly.down, &w.xs_hh, dy, w.p("dhh"), false, gs, sg)?;
        let inter = ly.gate.m();
        let ni = (n * inter) as u64;
        d.el(d.f.swiglu_bwd, n * inter, kargs!(w.p("gg"), w.p("uu"), w.p("dhh"), w.p("dgg"), w.p("duu"), ni))?;
        lin_bwd(d, net, w, &ly.gate, &w.xs_n2, w.p("dgg"), w.p("dn2"), false, gs, sg)?;
        lin_bwd(d, net, w, &ly.up, &w.xs_n2, w.p("duu"), w.p("dn2"), true, gs, sg)?;
        d.rms_bwd(w.p("x2"), net.w(ly.ln2), w.p("inv2"), w.p("dn2"), w.p("dx2"), net.g(ly.ln2), n, g.hs, g.gemma, part)?;
        d.add(w.p("dx2"), dy, w.p("dx2"), n * g.hs)?;
        // mixer: x2 = x + mix(n1)
        match &ly.mix {
            Mix::Gdn { qkv, z, a, b, out, prm } => {
                let prm = prm.as_ref().ok_or("GDN layer outside the net's range")?;
                lin_bwd(d, net, w, out, &w.xs_mx, w.p("dx2"), w.p("dgout"), false, gs, sg)?;
                {
                    let _ph = d.phase("gdn.bptt");
                    let gg = GdnG { nb: w.nb, t: w.t, nv: g.nv, nk: g.nk, kk: g.kk, eps: g.eps };
                    gdn_core_bwd(d, gg, prm.ptrs(), w.p("zz"), w.p("aa"), w.p("bb"), w.p("pre"), w.p("cq"), w.p("dgout"), w.p("draw"), w.p("dz"), w.p("da"), w.p("db"), &GdnS::of(w), true)?;
                }
                lin_bwd(d, net, w, qkv, &w.xs_n1, w.p("draw"), w.p("dn1"), false, gs, sg)?;
                lin_bwd(d, net, w, z, &w.xs_n1, w.p("dz"), w.p("dn1"), true, gs, sg)?;
                lin_bwd(d, net, w, a, &w.xs_n1, w.p("da"), w.p("dn1"), true, gs, sg)?;
                lin_bwd(d, net, w, b, &w.xs_n1, w.p("db"), w.p("dn1"), true, gs, sg)?;
            }
            Mix::Attn { q, k, v, o, qn, kn } => {
                let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
                lin_bwd(d, net, w, o, &w.xs_mx, w.p("dx2"), w.p("dog"), false, gs, sg)?;
                let nq = (n * qd) as u64;
                d.el(d.f.gate_bwd, n * qd, kargs!(w.p("o"), w.p("gate"), w.p("dog"), w.p("do"), w.p("dgate"), nq))?;
                let ag = AttnG { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
                let _ = kd;
                attn_core_bwd(d, ag, w.p("do"), w.p("dqr"), w.p("dkr"), w.p("dv"), &AttnS::of(w))?;
                let (nhi, nkvi, hdi, halfi, ti, gem) = (g.nh as u32, g.nkv as u32, g.hd as u32, g.half as u32, w.t as u32, g.gemma as u32);
                let blk = g.hd.next_multiple_of(32) as u32;
                d.c.launch(d.f.qkr_bwd, ((n * g.nh) as u32, 1), blk, 0, kargs!(w.p("q"), net.w(*qn), w.p("qinv"), w.rope.ptr, w.p("dqr"), w.p("dq"), w.p("dwq"), nhi, hdi, halfi, ti, gem))?;
                d.c.launch(d.f.qkr_bwd, ((n * g.nkv) as u32, 1), blk, 0, kargs!(w.p("k"), net.w(*kn), w.p("kinv"), w.rope.ptr, w.p("dkr"), w.p("dk"), w.p("dwk"), nkvi, hdi, halfi, ti, gem))?;
                // gain grads (summed over rows·heads) into the norms' gradients
                d.colsum(w.p("dwq"), w.p("dwq"), w.p("dwq"), net.g(*qn), n * g.nh, g.hd, 2, true, part)?;
                d.colsum(w.p("dwk"), w.p("dwk"), w.p("dwk"), net.g(*kn), n * g.nkv, g.hd, 2, true, part)?;
                let one_u = 1u32;
                d.el2(d.f.split_qg, n, qd, kargs!(w.p("dqg"), w.p("dq"), w.p("dgate"), nhi, hdi, one_u))?;
                lin_bwd(d, net, w, q, &w.xs_n1, w.p("dqg"), w.p("dn1"), false, gs, sg)?;
                lin_bwd(d, net, w, k, &w.xs_n1, w.p("dk"), w.p("dn1"), true, gs, sg)?;
                lin_bwd(d, net, w, v, &w.xs_n1, w.p("dv"), w.p("dn1"), true, gs, sg)?;
            }
        }
        d.rms_bwd(x, net.w(ly.ln1), w.p("inv1"), w.p("dn1"), dx, net.g(ly.ln1), n, g.hs, g.gemma, part)?;
        d.add(dx, w.p("dx2"), dx, n * g.hs)
    }
}

// ───────────────────────────── forward / step ─────────────────────────────

/// The embedded `ids` (f16 rows gathered on the host, converted on the card)
/// into the first checkpoint.
fn embed(d: &Dev, net: &Net, w: &Work, ids: &[u32]) -> Res<()> {
    let _p = d.phase("embed");
    let hs = net.geom.hs;
    if net.embed.is_empty() {
        return Err("this net does not start at layer 0".into());
    }
    let mut rows = vec![0u16; ids.len() * hs];
    let per = ids.len().div_ceil(16).max(1);
    std::thread::scope(|sc| {
        for (dst, chunk) in rows.chunks_mut(per * hs).zip(ids.chunks(per)) {
            let emb = &net.embed;
            sc.spawn(move || {
                for (o, &id) in dst.chunks_mut(hs).zip(chunk) {
                    o.copy_from_slice(&emb[id as usize * hs..(id as usize + 1) * hs]);
                }
            });
        }
    });
    put(d.c, w.p("emb16"), &rows)?;
    let n = (ids.len() * hs) as u64;
    unsafe { d.el(d.f.h2f_n, ids.len() * hs, kargs!(w.p("emb16"), w.ckpt[0].ptr, n)) }
}

/// Forward over layers `net.first..upto` from `h0` (`[ntok][hs]`: the input
/// of layer `net.first`) or the embedding of `ids`; with `head` and `upto`
/// the last layer, the final norm into `n1` (a calibration keeps `n1` as the
/// last layer's own input instead).
fn forward_to(d: &Dev, net: &Net, w: &Work, ids: &[u32], h0: Option<&[f32]>, upto: usize, head: bool) -> Res<()> {
    let g = &net.geom;
    if ids.len() != w.ntok && h0.is_none() {
        return Err(format!("{} ids for a workspace of {} tokens", ids.len(), w.ntok));
    }
    match h0 {
        Some(h) => {
            if h.len() != w.ntok * g.hs {
                return Err("hidden input of the wrong size".into());
            }
            put(d.c, w.ckpt[net.first].ptr, h)?;
        }
        None if net.first == 0 => embed(d, net, w, ids)?,
        None => return Err("a net starting past layer 0 needs its input".into()),
    }
    let upto = upto.min(net.end);
    for l in net.first..upto {
        let _p = d.phase(if g.gdn_layer[l] { "fwd.gdn" } else { "fwd.attn" });
        unsafe { layer_fwd(d, net, w, l, w.ckpt[l].ptr, w.ckpt[l + 1].ptr, false)? };
    }
    if head && upto == g.nl {
        let _p = d.phase("fwd.final_norm");
        unsafe { d.rms_fwd(w.ckpt[g.nl].ptr, net.w(net.final_norm), w.p("n1"), w.p("invf"), w.ntok, g.hs, g.eps, g.gemma)? };
    }
    Ok(())
}

/// Logits of `rows` normed hidden rows from row `r0` into the head scratch.
unsafe fn head_logits(d: &Dev, net: &Net, w: &Work, r0: usize, rows: usize) -> Res<()> {
    let (v, hs) = (net.geom.vocab, net.geom.hs);
    let head = net.head.as_ref().ok_or("this net has no head")?;
    unsafe {
        d.stage(w.b["n1"].at(r0 * hs), None, w.p("s1"), rows, hs, 1.0)?;
        d.mm(rows, v, hs, 1.0, w.p("s1"), hs, false, head.ptr, hs, true, 0.0, w.p("logits"), v, Gemm::Half)
    }
}

/// Teacher top-k for `ids` (`[nb·t]`): ids and renormalized probabilities
/// (`[ntok][k]` each, k ≤ 256).
pub fn teacher_topk(net: &Net, w: &Work, ids: &[u32], k: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
    let d = enter()?;
    if k == 0 || k > K_MAX {
        return Err(format!("top-{k}: 1..={K_MAX}"));
    }
    forward_to(&d, net, w, ids, None, net.geom.nl, true)?;
    topk_n1(&d, net, w, k)
}

/// Top-k of the normed hidden rows in `n1` through the head.
fn topk_n1(d: &Dev, net: &Net, w: &Work, k: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
    let _p = d.phase("head.topk");
    let (v, n) = (net.geom.vocab as u32, w.ntok);
    let ki = k as u32;
    let mut r0 = 0;
    while r0 < n {
        let rows = (n - r0).min(w.head_chunk);
        unsafe {
            head_logits(d, net, w, r0, rows)?;
            d.c.launch(d.f.topk_rows, (rows as u32, 1), 256, 0, kargs!(w.p("logits"), w.b["tid"].at(r0 * k), w.b["tp"].at(r0 * k), v, ki))?;
        }
        r0 += rows;
    }
    Ok((get_u32(d.c, w.p("tid"), n * k)?, get_f32(d.c, w.p("tp"), n * k)?))
}

/// The head's loss per token into `loss` and, with `bwd`, the gradient at the
/// normed hidden into `dn1` (`gscale` the f16 staging scale of the logits'
/// gradient). `total`: the loss's normalizer.
#[allow(clippy::too_many_arguments)]
unsafe fn head_kl(d: &Dev, net: &Net, w: &Work, tid: &[u32], tp: &[f32], k: usize, total: f32, gs: f32, weighted: bool, bwd: bool) -> Res<()> {
    let (v, hs, n) = (net.geom.vocab, net.geom.hs, w.ntok);
    if k == 0 || k > K_MAX || tid.len() != n * k || tp.len() != n * k {
        return Err("teacher rows of the wrong shape".into());
    }
    let head = net.head.as_ref().ok_or("this net has no head")?;
    put(d.c, w.p("tid"), tid)?;
    put(d.c, w.p("tp"), tp)?;
    let (vi, ki, coef, wt) = (v as u32, k as u32, gs / total, weighted as u32);
    let mut r0 = 0;
    while r0 < n {
        let rows = (n - r0).min(w.head_chunk);
        unsafe {
            head_logits(d, net, w, r0, rows)?;
            d.c.launch(d.f.kl_rows, (rows as u32, 1), 1024, 0, kargs!(w.p("logits"), w.b["tid"].at(r0 * k), w.b["tp"].at(r0 * k), w.b["loss"].at(r0), vi, ki, coef, wt))?;
            if bwd {
                // dh = dlogits · W
                d.stage(w.p("logits"), None, w.p("dl16"), rows, v, 1.0)?;
                d.mm(rows, hs, v, 1.0 / gs, w.p("dl16"), v, false, head.ptr, hs, false, 0.0, w.b["dn1"].at(r0 * hs), hs, Gemm::Half)?;
            }
        }
        r0 += rows;
    }
    Ok(())
}

fn loss_read(d: &Dev, w: &Work) -> Res<f64> {
    Ok(get_f32(d.c, w.p("loss"), w.ntok)?.iter().map(|&x| x as f64).sum())
}

/// Mean KL of `ids` against the teacher top-k (forward only).
pub fn loss_only(net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize) -> Result<f64, String> {
    let d = enter()?;
    forward_to(&d, net, w, ids, None, net.geom.nl, true)?;
    let _p = d.phase("head");
    unsafe { head_kl(&d, net, w, tid, tp, k, w.ntok as f32, 1024.0, false, false)? };
    Ok(loss_read(&d, w)? / w.ntok as f64)
}

/// One forward + backward on `ids` against the teacher top-k; gradients
/// accumulate into the parameters' gradient buffer (cleared by [`adam`] or
/// [`zero_grads`]). With `sign`, every sign matrix is updated during the
/// reverse sweep; a non-finite loss then switches every sign-state write off
/// on the card (the step is reported without a backward).
#[allow(clippy::too_many_arguments)]
pub fn step(net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize, gscale: f32, sign: Option<&SignStepCfg>) -> Result<StepStats, String> {
    let d = enter()?;
    let g = &net.geom;
    let n = w.ntok;
    // `CMF_NVG_KT=N`: the time of every kernel and GEMM of the N-th step
    static STEP_NO: AtomicU64 = AtomicU64::new(0);
    static KT_AT: OnceLock<u64> = OnceLock::new();
    let kt = STEP_NO.fetch_add(1, Ordering::Relaxed) + 1 == *KT_AT.get_or_init(|| std::env::var("CMF_NVG_KT").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if kt {
        d.c.sync()?;
        kt_start();
    }
    if !net.bwd {
        return Err("this net was loaded without a backward".into());
    }
    let sign = if net.n_sign_mats > 0 { sign } else { None };
    if sign.is_some() {
        unsafe { cu::memset_d8_async(w.cnt.ptr, 0, ts::CNT_LEN * 4, d.c.stream).map_err(derr)? };
    }
    forward_to(&d, net, w, ids, None, g.nl, true)?;
    {
        let _p = d.phase("head");
        unsafe {
            head_kl(&d, net, w, tid, tp, k, n as f32, gscale, false, true)?;
            if sign.is_some() {
                let ni = n as u32;
                d.c.launch(d.f.loss_flag, (1, 1), 256, 0, kargs!(w.p("loss"), w.cnt.ptr, ni))?;
            }
        }
    }
    let (mut cur, mut nxt) = (w.p("dx"), w.p("dmix"));
    {
        let _p = d.phase("bwd.final_norm");
        unsafe { d.rms_bwd(w.ckpt[g.nl].ptr, net.w(net.final_norm), w.p("invf"), w.p("dn1"), cur, net.g(net.final_norm), n, g.hs, g.gemma, w.p("part"))? };
    }
    for l in (net.first..g.nl).rev() {
        let gdn = g.gdn_layer[l];
        {
            let _p = d.phase(if gdn { "recomp.gdn" } else { "recomp.attn" });
            unsafe { layer_fwd(&d, net, w, l, w.ckpt[l].ptr, w.p("x"), true)? };
        }
        let _p = d.phase(if gdn { "bwd.gdn" } else { "bwd.attn" });
        unsafe { layer_bwd(&d, net, w, l, w.ckpt[l].ptr, cur, nxt, gscale, sign)? };
        std::mem::swap(&mut cur, &mut nxt);
    }
    let mut st = StepStats { backward: true, ..Default::default() };
    if sign.is_some() {
        let cnt = get_u32(d.c, w.cnt.ptr, ts::CNT_LEN)?;
        for (i, f) in st.flips_role.iter_mut().enumerate() {
            *f = cnt[i] as u64;
        }
        st.flips = st.flips_role.iter().sum();
        st.g_nonfinite = cnt[ts::CNT_NONFINITE] as u64;
        st.saturated = cnt[ts::CNT_SATURATED] as u64;
        st.clamped = cnt[ts::CNT_CLAMPED] as u64;
    }
    st.loss = loss_read(&d, w)? / n as f64;
    st.finite = st.loss.is_finite();
    if sign.is_some() && !st.finite {
        st.backward = false;
    }
    if kt {
        kt_stop_as("nvg-train cuda");
    }
    Ok(st)
}

// ───────────────────────────── optimizer ─────────────────────────────

/// Global gradient norm over every parameter (NaN/inf when a gradient is):
/// 1024 partial sums of squares summed on the host in f64.
pub fn grad_norm(net: &Net) -> Result<f64, String> {
    let d = enter()?;
    let _p = d.phase("opt.grad_norm");
    const NB: usize = 1024;
    let part = DBuf::floats(d.c, NB)?;
    let (n, nb) = (net.total as u64, NB as u32);
    unsafe { d.c.launch(d.f.sumsq, (NB as u32, 1), 256, 0, kargs!(net.pg.ptr, part.ptr, n, nb))? };
    Ok(get_f32(d.c, part.ptr, NB)?.iter().map(|&x| x as f64).sum::<f64>().sqrt())
}

/// Clear every gradient (a skipped step).
pub fn zero_grads(net: &Net) -> Result<(), String> {
    let d = enter()?;
    zero_f(d.c, net.pg.ptr, net.total)
}

/// Adam over every parameter (masked entries stay put), the gradients
/// multiplied by `gmul` (clipping); clears the gradients.
#[allow(clippy::too_many_arguments)]
pub fn adam(net: &Net, lr: f32, step_no: u32, b1: f32, b2: f32, eps: f32, gmul: f32) -> Result<(), String> {
    let d = enter()?;
    let _p = d.phase("opt.adam");
    let bc1 = (1.0 - (b1 as f64).powi(step_no as i32)) as f32;
    let bc2 = (1.0 - (b2 as f64).powi(step_no as i32)) as f32;
    let n = net.total as u64;
    unsafe { d.el(d.f.adam, net.total, kargs!(net.pw.ptr, net.pg.ptr, net.pm1.ptr, net.pm2.ptr, net.pmask.ptr, net.prelf.ptr, n, lr, bc1, bc2, b1, b2, eps, gmul)) }
}

// ───────────────────────────── snapshots, state, export ─────────────────────────────

/// Host copy of every parameter's values and (with `with_bits`) every sign
/// bitmap — `nvg_train::Snap`, the wgpu trainer's order.
pub fn snapshot(net: &Net, with_bits: bool) -> Result<Snap, String> {
    let d = enter()?;
    let all = get_f32(d.c, net.pw.ptr, net.total)?;
    let params = net.params.iter().map(|p| all[p.off..p.off + p.n].to_vec()).collect();
    let bits = if with_bits { net.bitmaps().iter().map(|(b, n)| get_u32(d.c, b.ptr, *n)).collect::<Res<Vec<_>>>()? } else { Vec::new() };
    Ok(Snap { params, bits })
}

/// Restore a [`snapshot`].
pub fn restore(net: &Net, snap: &Snap) -> Result<(), String> {
    let d = enter()?;
    if snap.params.len() != net.params.len() {
        return Err("snapshot of another net".into());
    }
    let mut all = vec![0f32; net.total];
    for (p, v) in net.params.iter().zip(&snap.params) {
        if v.len() != p.n {
            return Err("snapshot of another net".into());
        }
        all[p.off..p.off + p.n].copy_from_slice(v);
    }
    put(d.c, net.pw.ptr, &all)?;
    for ((b, n), v) in net.bitmaps().iter().zip(&snap.bits) {
        if v.len() != *n {
            return Err("snapshot bitmaps of another net".into());
        }
        put(d.c, b.ptr, v)?;
    }
    Ok(())
}

/// Distinct flips against the starting bits per role and side:
/// (`role.U` / `role.V`, flipped, live). Needs `SignInit::keep_bits0`.
pub fn sign_stats(net: &Net) -> Result<Vec<(String, u64, u64)>, String> {
    let mats = net.sign_mats();
    if mats.is_empty() {
        return Ok(Vec::new());
    }
    let d = enter()?;
    let diff = DBuf::floats(d.c, net.n_sign_mats as usize)?;
    for (mat, bits, _) in &mats {
        let b0 = mat.bits0.as_ref().ok_or("sign statistics need the starting bits")?;
        let p = sp(mat, None, mat.id, 0.0, 0);
        unsafe { d.el(d.f.bits_diff, mat.rows * mat.wpr(), kargs!(p, bits.ptr, b0.ptr, diff.ptr))? };
    }
    let dv = get_u32(d.c, diff.ptr, net.n_sign_mats as usize)?;
    let mut acc = [(0u64, 0u64); 20];
    for (mat, _, slot) in &mats {
        acc[*slot as usize].0 += dv[mat.id as usize] as u64;
        acc[*slot as usize].1 += mat.live();
    }
    Ok(acc
        .iter()
        .enumerate()
        .filter(|(_, (_, live))| *live > 0)
        .map(|(i, &(f, live))| (format!("{}.{}", ts::ROLES[i / 2], ["U", "V"][i % 2]), f, live))
        .collect())
}

/// Every device buffer a resumed run needs with its word count, in the wgpu
/// trainer's order: each parameter with its Adam moments, every bitmap,
/// every sign matrix's latents, moments, factored statistics and starting bits.
fn state_bufs(net: &Net) -> Vec<(u64, usize)> {
    let mut v = Vec::new();
    for p in &net.params {
        v.push((net.pw.at(p.off), p.n));
        v.push((net.pm1.at(p.off), p.n));
        v.push((net.pm2.at(p.off), p.n));
    }
    v.extend(net.bitmaps().iter().map(|(b, n)| (b.ptr, *n)));
    for (mat, _, _) in net.sign_mats() {
        v.push((mat.lat.ptr, mat.lat_words()));
        if let Some(m) = &mat.mom {
            v.push((m.ptr, mat.lat_words()));
        }
        if let Some(x) = &mat.v {
            v.push((x.ptr, mat.rows * mat.rp));
        }
        v.push((mat.vrow.ptr, mat.rows));
        v.push((mat.vcol.ptr, mat.rp));
        if let Some(b0) = &mat.bits0 {
            v.push((b0.ptr, mat.rows * mat.wpr()));
        }
    }
    v
}

/// Write the training state to `path` (atomic rename; the wgpu trainer's
/// file format and order). Returns the bytes written.
pub fn save_state(net: &Net, path: &std::path::Path) -> Result<u64, String> {
    let d = enter()?;
    let bufs = state_bufs(net);
    let sizes: Vec<usize> = bufs.iter().map(|b| b.1).collect();
    let mut it = bufs.iter().map(|(p, n)| get_u32(d.c, *p, *n));
    crate::nvg_train::write_words(path, &mut it, &sizes)
}

/// Load a [`save_state`] file into a net of the same layout and options.
pub fn load_state(net: &Net, path: &std::path::Path) -> Result<(), String> {
    let d = enter()?;
    let bufs = state_bufs(net);
    let sizes: Vec<usize> = bufs.iter().map(|b| b.1).collect();
    let mut err = None;
    crate::nvg_train::read_words(path, &sizes, |i, v| {
        if let Err(e) = put(d.c, bufs[i].0, v) {
            err.get_or_insert(e);
        }
    })?;
    err.map_or(Ok(()), Err)
}

/// Copy the trained metric and the sign bits of every binary low-rank tensor
/// and every norm gain back into payloads of `model` (same layout and
/// sizes): (tensor name, new bytes) for every changed entry.
pub fn export(net: &Net, model: &CmfModel) -> Result<Vec<(String, Vec<u8>)>, String> {
    let d = enter()?;
    let mut out = Vec::new();
    let to_bytes_f16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect() };
    let all = get_f32(d.c, net.pw.ptr, net.total)?;
    let val = |i: usize| -> &[f32] { &all[net.params[i].off..net.params[i].off + net.params[i].n] };
    for (name, i) in &net.norm_names {
        let v = val(*i);
        let e = model.tensor(name).ok_or_else(|| format!("missing {name}"))?;
        let bytes = match e.dtype as u8 {
            1 => to_bytes_f16(v),
            0 => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            other => return Err(format!("{name}: norm dtype {other} not writable")),
        };
        out.push((name.clone(), bytes));
    }
    for (name, lin) in net.blr_lins() {
        let Lin::Blr { m, n, r: rp, paths, .. } = lin else { continue };
        let (m, n, rp) = (*m, *n, *rp);
        let e = model.tensor(&name).ok_or_else(|| format!("missing {name}"))?;
        let mut bytes = model.entry_bytes(e).to_vec();
        let r = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if r % 8 != 0 || n % 8 != 0 {
            return Err(format!("{name}: rank {r} / width {n} not byte-aligned"));
        }
        let wpr = rp / 32;
        let mut off = 16;
        for p in paths {
            bytes[off..off + 2 * m].copy_from_slice(&to_bytes_f16(val(p.h)));
            off += 2 * m;
            bytes[off..off + 2 * n].copy_from_slice(&to_bytes_f16(val(p.g)));
            off += 2 * n;
            bytes[off..off + 2 * r].copy_from_slice(&to_bytes_f16(&val(p.l)[..r]));
            off += 2 * r;
            // signs: U [m][r/8] (rank-minor), V [r][n/8] (rank-major); pad ranks never written
            let bu = get_u32(d.c, p.bits_u.ptr, m * wpr)?;
            crate::nvg_train::u_bits_to_file(&bu, m, r, wpr, &mut bytes[off..off + m * r / 8]);
            off += m * r / 8;
            let bv = get_u32(d.c, p.bits_v.ptr, n * wpr)?;
            crate::nvg_train::v_bits_to_file(&bv, n, r, wpr, &mut bytes[off..off + r * n / 8]);
            off += r * n / 8;
        }
        out.push((name, bytes));
    }
    Ok(out)
}

// ───────────────────────────── calibration ─────────────────────────────

/// A device Hessian accumulator `H += XᵀX` of one linear input (`k` wide).
pub struct Gram {
    buf: DBuf,
    pub k: usize,
    pub count: usize,
}

impl Gram {
    pub fn new(k: usize) -> Result<Gram, String> {
        let d = enter()?;
        Ok(Gram { buf: DBuf::floats(d.c, k * k)?, k, count: 0 })
    }
    /// The accumulated `Σ x xᵀ` as a full symmetric f64 matrix.
    pub fn read(&self) -> Result<Vec<f64>, String> {
        let d = enter()?;
        Ok(get_f32(d.c, self.buf.ptr, self.k * self.k)?.into_iter().map(|x| x as f64).collect())
    }
}

/// The inputs of a layer's linears a calibration folds into its Hessians:
/// `in` (q/k/v, or GDN qkv/z), `mixout` (o_proj or GDN out_proj), `mlp`
/// (gate/up), `down`.
pub const CALIB_ROLES: [&str; 4] = ["in", "mixout", "mlp", "down"];

/// One calibration pass of `nb` windows: forward over layers `net.first..=l`
/// from `h0` (the input of layer `net.first`) or the embedding of `ids`, then
/// `H += XᵀX` of layer `l`'s inputs for every listed (role, accumulator).
/// With `keep`, returns the input of layer `l` (`[ntok][hs]`: where the next
/// layer's pass resumes, so each layer runs once over a conversion).
pub fn calib_pass(net: &Net, w: &Work, ids: &[u32], h0: Option<&[f32]>, l: usize, grams: &mut [(&str, &mut Gram)], keep: bool) -> Result<Option<Vec<f32>>, String> {
    let d = enter()?;
    let g = &net.geom;
    if l < net.first || l >= net.end {
        return Err(format!("layer {l} outside the net's range {}..{}", net.first, net.end));
    }
    forward_to(&d, net, w, ids, h0, l + 1, false)?;
    let n = w.ntok;
    for (role, acc) in grams.iter_mut() {
        let (x, k) = match *role {
            "in" => (w.p("n1"), g.hs),
            "mixout" => {
                if g.gdn_layer[l] {
                    (w.p("gout"), vd(g))
                } else {
                    (w.p("og"), g.nh * g.hd)
                }
            }
            "mlp" => (w.p("n2"), g.hs),
            "down" => (w.p("hh"), net.layers[l].gate.m()),
            other => return Err(format!("unknown calibration role {other}")),
        };
        if acc.k != k {
            return Err(format!("{role}: a {}-wide accumulator for a {k}-wide input", acc.k));
        }
        // H[i][j] += Σ_t x[t][i]·x[t][j], plain f32
        unsafe { d.mm(k, k, n, 1.0, x, k, true, x, k, false, 1.0, acc.buf.ptr, k, Gemm::F32)? };
        acc.count += n;
    }
    if keep { Ok(Some(get_f32(d.c, w.ckpt[l].ptr, n * g.hs)?)) } else { Ok(None) }
}

#[cfg(test)]
mod tests {
    //! The trainer's kernels against f64 host references, the sign update
    //! against `train_sign`'s host mirror (bit for bit), the GDN and
    //! attention cores against the bake's entries. They skip without a card.
    use super::*;

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale
            })
            .collect()
    }

    fn rel(name: &str, got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
        let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-30);
        let e = (num / den).sqrt();
        eprintln!("{name}: relative error {e:e}");
        assert!(e < tol, "{name}: relative error {e:e}");
    }

    fn geom(hs: usize, inter: usize, vocab: usize) -> Geom {
        Geom {
            hs,
            nl: 1,
            nh: 1,
            nkv: 1,
            hd: 32,
            half: 8,
            inter,
            vocab,
            eps: 1e-6,
            gemma: false,
            gdn_layer: vec![false],
            nv: 0,
            nk: 0,
            kk: 0,
            inv_freq: (0..8).map(|i| 1.0 / 10000f64.powf(i as f64 / 8.0)).collect(),
        }
    }

    /// A one-layer net whose MLP gate is `gate` (every other linear absent),
    /// with the head `w` (`[vocab][hs]`) when given.
    fn net_of(d: &Dev, mut pb: PB, gate: Lin, hs: usize, inter: usize, head: Option<(&[f32], usize)>, n_sign: u32) -> Net {
        let final_norm = pb.add(&vec![1f32; hs], None, false);
        let ln1 = pb.add(&vec![1f32; hs], None, false);
        let ln2 = pb.add(&vec![1f32; hs], None, false);
        let qn = pb.add(&[1f32; 32], None, false);
        let kn = pb.add(&[1f32; 32], None, false);
        let total = pb.w.len();
        Net {
            geom: geom(hs, inter, head.map_or(64, |h| h.1)),
            first: 0,
            end: 1,
            layers: vec![Layer {
                ln1,
                ln2,
                mix: Mix::Attn { q: Lin::Absent { m: 64 }, k: Lin::Absent { m: 32 }, v: Lin::Absent { m: 32 }, o: Lin::Absent { m: hs }, qn, kn },
                gate,
                up: Lin::Absent { m: inter },
                down: Lin::Absent { m: hs },
            }],
            final_norm,
            head: head.map(|(w, _)| DBuf::from(&to_f16(w)).unwrap()),
            embed: Vec::new(),
            params: pb.params,
            total,
            pw: DBuf::from(&pb.w).unwrap(),
            pg: DBuf::floats(d.c, total).unwrap(),
            pm1: DBuf::floats(d.c, total).unwrap(),
            pm2: DBuf::floats(d.c, total).unwrap(),
            pmask: DBuf::from(&pb.mask).unwrap(),
            prelf: DBuf::from(&pb.relf).unwrap(),
            norm_names: Vec::new(),
            n_sign_mats: n_sign,
            bwd: true,
        }
    }

    type HostPath = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// A random binary low-rank linear (`np` paths, true rank `r` padded to
    /// 128) on the card and its host copy (S_U, S_V, h, ℓ, g per path).
    fn blr_of(d: &Dev, pb: &mut PB, m: usize, n: usize, r: usize, np: usize, sign: Option<&SignInit>) -> (Lin, Vec<HostPath>, u32) {
        let rp = r.div_ceil(128) * 128;
        let wpr = rp / 32;
        let mut host = Vec::new();
        let mut paths = Vec::new();
        let mut next_id = 0u32;
        for pi in 0..np {
            let su: Vec<f32> = rnd(m * r, 10 + pi as u64, 1.0).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let sv: Vec<f32> = rnd(n * r, 20 + pi as u64, 1.0).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let pack = |s: &[f32], rows: usize| -> Vec<u32> {
                let mut b = vec![0u32; rows * wpr];
                for row in 0..rows {
                    for j in 0..r {
                        if s[row * r + j] > 0.0 {
                            b[row * wpr + j / 32] |= 1 << (j % 32);
                        }
                    }
                }
                b
            };
            let h: Vec<f32> = rnd(m, 30 + pi as u64, 0.5).iter().map(|v| 0.02 + v.abs() * 0.05).collect();
            let l: Vec<f32> = rnd(r, 40 + pi as u64, 0.5).iter().map(|v| 0.5 + v.abs()).collect();
            let g: Vec<f32> = rnd(n, 50 + pi as u64, 0.5).iter().map(|v| 0.1 + v.abs() * 0.2).collect();
            let (bits_u, bits_v) = (DBuf::from(&pack(&su, m)).unwrap(), DBuf::from(&pack(&sv, n)).unwrap());
            let mut lp = l.clone();
            lp.resize(rp, 0.0);
            let mask: Vec<f32> = (0..rp).map(|k| if k < r { 1.0 } else { 0.0 }).collect();
            let (ph, pl, pg) = (pb.add(&h, None, true), pb.add(&lp, Some(&mask), true), pb.add(&g, None, true));
            let sp = sign.map(|init| {
                let u = init_mat(d, &bits_u, m, r, rp, init, next_id).unwrap();
                let v = init_mat(d, &bits_v, n, r, rp, init, next_id + 1).unwrap();
                next_id += 2;
                SignPath { u, v }
            });
            paths.push(Path { bits_u, bits_v, h: ph, l: pl, g: pg, mh: plane_mul(&h), mg: plane_mul(&g), sign: sp });
            host.push((su, sv, h, l, g));
        }
        (Lin::Blr { m, n, r: rp, role: "mlp.gate", paths }, host, next_id)
    }

    /// Two-path binary low-rank forward and backward against an f64
    /// reference (tolerance set by the f16 operands: ~1e-3 relative), with a
    /// true rank below the padded one.
    #[test]
    fn cuda_blr_layer_matches_host() {
        let Ok(d) = enter() else {
            eprintln!("no CUDA device — skipped");
            return;
        };
        let (m, n, r, ntok, np) = (256usize, 384usize, 120usize, 70usize, 2usize);
        let mut pb = PB::default();
        let (gate, host, _) = blr_of(&d, &mut pb, m, n, r, np, None);
        let net = net_of(&d, pb, gate, n, m, None, 0);
        drop(d);
        let w = Work::new(&net, 1, ntok).unwrap();
        let d = enter().unwrap();
        let x = rnd(ntok * n, 60, 2.0);
        let dy = rnd(ntok * m, 61, 1e-3);
        let (bx, bdy) = (DBuf::from(&x).unwrap(), DBuf::from(&dy).unwrap());
        let (by, bdx) = (DBuf::floats(d.c, ntok * m).unwrap(), DBuf::floats(d.c, ntok * n).unwrap());
        let lin = &net.layers[0].gate;
        unsafe {
            lin_fwd(&d, &net, &w, lin, bx.ptr, &w.xs_n2, by.ptr, false).unwrap();
            lin_bwd(&d, &net, &w, lin, &w.xs_n2, bdy.ptr, bdx.ptr, false, 1024.0, None).unwrap();
        }
        let mut y = vec![0f64; ntok * m];
        let mut dx = vec![0f64; ntok * n];
        let mut grads = Vec::new();
        for (su, sv, h, l, g) in &host {
            let (mut dh, mut dl, mut dg) = (vec![0f64; m], vec![0f64; r], vec![0f64; n]);
            for t in 0..ntok {
                let a: Vec<f64> = (0..r).map(|k| (0..n).map(|j| x[t * n + j] as f64 * g[j] as f64 * sv[j * r + k] as f64).sum()).collect();
                let cc: Vec<f64> = (0..m).map(|i| (0..r).map(|k| a[k] * l[k] as f64 * su[i * r + k] as f64).sum()).collect();
                for i in 0..m {
                    y[t * m + i] += h[i] as f64 * cc[i];
                    dh[i] += dy[t * m + i] as f64 * cc[i];
                }
                let db: Vec<f64> = (0..r).map(|k| (0..m).map(|i| dy[t * m + i] as f64 * h[i] as f64 * su[i * r + k] as f64).sum()).collect();
                for k in 0..r {
                    dl[k] += a[k] * db[k];
                }
                for j in 0..n {
                    let dxp: f64 = (0..r).map(|k| db[k] * l[k] as f64 * sv[j * r + k] as f64).sum();
                    dg[j] += x[t * n + j] as f64 * dxp;
                    dx[t * n + j] += g[j] as f64 * dxp;
                }
            }
            grads.push((dh, dl, dg));
        }
        rel("y", &get_f32(d.c, by.ptr, ntok * m).unwrap(), &y, 3e-3);
        rel("dx", &get_f32(d.c, bdx.ptr, ntok * n).unwrap(), &dx, 3e-3);
        let pg = get_f32(d.c, net.pg.ptr, net.total).unwrap();
        let Lin::Blr { paths, .. } = lin else { unreachable!() };
        for (pi, (dh, dl, dg)) in grads.iter().enumerate() {
            let at = |i: usize, n: usize| &pg[net.params[i].off..net.params[i].off + n];
            rel(&format!("dh{pi}"), at(paths[pi].h, m), dh, 3e-3);
            rel(&format!("dl{pi}"), at(paths[pi].l, r), dl, 3e-3);
            rel(&format!("dg{pi}"), at(paths[pi].g, n), dg, 3e-3);
        }
    }

    /// The fact-mom sign update: the factored statistics against the host's
    /// (f32 sums in another order), then the latent, moment and bitmap step
    /// bit for bit against `train_sign::host_step_fact` from the card's own
    /// statistics — two steps, non-finite gradient entries, pad ranks.
    #[test]
    fn cuda_sign_update_matches_host_mirror() {
        let Ok(d) = enter() else {
            eprintln!("no CUDA device — skipped");
            return;
        };
        let (m, n, r, ntok) = (300usize, 160usize, 120usize, 40usize);
        let rp = 128;
        let init = SignInit { opt: SignOpt::FactMom, tau: 0.25, seed: 42, keep_bits0: true, defer: false };
        let mut pb = PB::default();
        let (gate, _, ns) = blr_of(&d, &mut pb, m, n, r, 1, Some(&init));
        let net = net_of(&d, pb, gate, n, m, None, ns);
        drop(d);
        let w = Work::new(&net, 1, ntok).unwrap();
        let d = enter().unwrap();
        let Lin::Blr { paths, .. } = &net.layers[0].gate else { unreachable!() };
        let (mat, bits) = (&paths[0].sign.as_ref().unwrap().u, &paths[0].bits_u);
        let (rows, wpr) = (m, rp / 32);
        let mom_of = |words: &[u32]| -> Vec<u32> { (0..rows * rp).map(|e| (words[e / 2] >> (16 * (e % 2))) & 0xffff).collect() };
        for step in 1..=2u32 {
            let mut g = rnd(rows * rp, 70 + step as u64, 2e-3);
            g[5] = f32::NAN;
            g[rp + 7] = f32::INFINITY;
            let cfg = SignStepCfg { opt: SignOpt::FactMom, eta: 0.02, step, b1: 0.9, b2: 0.999, eps: 1e-3, clamp: 4.0, hyst: 0.0, k_ss: 8.0, seed: 42, defer: false };
            let bg = DBuf::from(&g).unwrap();
            let lat0 = get_u32(d.c, mat.lat.ptr, rows * rp / 2).unwrap();
            let mom0 = get_u32(d.c, mat.mom.as_ref().unwrap().ptr, rows * rp / 2).unwrap();
            let bits0 = get_u32(d.c, bits.ptr, rows * wpr).unwrap();
            let (mut vrow_h, mut vcol_h) = (get_f32(d.c, mat.vrow.ptr, rows).unwrap(), get_f32(d.c, mat.vcol.ptr, rp).unwrap());
            unsafe {
                cu::memset_d8_async(w.cnt.ptr, 0, ts::CNT_LEN * 4, d.c.stream).unwrap();
                sign_update(&d, &w, &cfg, mat, bg.ptr, bits.ptr, 5).unwrap();
            }
            let (vrow, vcol) = (get_f32(d.c, mat.vrow.ptr, rows).unwrap(), get_f32(d.c, mat.vcol.ptr, rp).unwrap());
            let mm = get_f32(d.c, w.p("scal"), 1).unwrap()[0];
            let cnt = get_u32(d.c, w.cnt.ptr, ts::CNT_LEN).unwrap();
            // the statistics: the host sums in f64, the card in f32
            let (mm_h, nonfin) = ts::host_stats(&cfg, rows, r, rp, &g, &mut vrow_h, &mut vcol_h);
            assert_eq!(cnt[ts::CNT_NONFINITE] as u64, nonfin, "step {step}: non-finite count");
            rel("vrow", &vrow, &vrow_h.iter().map(|&v| v as f64).collect::<Vec<_>>(), 1e-5);
            rel("vcol", &vcol[..r], &vcol_h[..r].iter().map(|&v| v as f64).collect::<Vec<_>>(), 1e-5);
            assert!((mm - mm_h).abs() <= 1e-5 * mm_h.abs(), "step {step}: column mean {mm} vs {mm_h}");
            // the step from the card's statistics: exact
            let mut lat_h = ts::unpack_i16(&lat0);
            let mut mom_h = mom_of(&mom0);
            let mut bits_h = bits0.clone();
            let (flips, sat, clp) = ts::host_step_fact(&cfg, mat.id, rows, r, rp, &g, &mut lat_h, Some(&mut mom_h), &vrow, &vcol, mm, &mut bits_h);
            let lat1 = ts::unpack_i16(&get_u32(d.c, mat.lat.ptr, rows * rp / 2).unwrap());
            let mom1 = mom_of(&get_u32(d.c, mat.mom.as_ref().unwrap().ptr, rows * rp / 2).unwrap());
            let bits1 = get_u32(d.c, bits.ptr, rows * wpr).unwrap();
            let dl = lat1.iter().zip(&lat_h).filter(|(a, b)| a != b).count();
            let dm = mom1.iter().zip(&mom_h).filter(|(a, b)| a != b).count();
            let db = bits1.iter().zip(&bits_h).filter(|(a, b)| a != b).count();
            eprintln!("step {step}: {flips} flips, {sat} saturated, {clp} clamped; differing latents {dl}, moments {dm}, bitmap words {db}");
            assert_eq!((dl, dm, db), (0, 0, 0), "step {step}: the card's update differs from the host mirror");
            assert_eq!((cnt[5] as u64, cnt[ts::CNT_SATURATED] as u64, cnt[ts::CNT_CLAMPED] as u64), (flips, sat, clp), "step {step}: counters");
            assert!(flips > 0, "step {step}: the test moved no sign");
        }
        // distinct flips against the starting bits
        drop(d);
        let stats = sign_stats(&net).unwrap();
        assert!(stats.iter().any(|(name, f, live)| name == "mlp.gate.U" && *f > 0 && *live == (m * r) as u64), "{stats:?}");
    }

    /// The GDN core (conv, recurrence, the reverse sweep over an f16 state
    /// history — kept by the forward, and replayed one sequence a launch — the
    /// group sum of the q̂/k̂ through-gradients) against the bake's entries.
    #[test]
    fn cuda_gdn_core_matches_bake() {
        if enter().is_err() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let (nb, t, nv, nk, kk) = (2usize, 9usize, 2usize, 1usize, 4usize);
        let g = GdnG { nb, t, nv, nk, kk, eps: 1e-6 };
        let (n, cd, v) = (nb * t, g.cd(), g.vd());
        let (qkv, z, a, b, dout) = (rnd(n * cd, 1, 0.5), rnd(n * v, 2, 1.0), rnd(n * nv, 3, 1.0), rnd(n * nv, 4, 1.0), rnd(n * v, 5, 1.0));
        let (conv, a_log, dt_bias) = (rnd(cd * kk, 6, 0.3), rnd(nv, 7, 0.5), rnd(nv, 8, 0.5));
        let norm: Vec<f32> = rnd(DKV, 9, 0.1).iter().map(|x| 1.0 + x).collect();
        let bg = BakeGdn { nv, nk, dk: DKV, dv: DKV, kk, eps: 1e-6, conv: &conv, a_log: &a_log, dt_bias: &dt_bias, norm: &norm };
        let out_ref = bake_gdn_fwd(&qkv, &z, &a, &b, nb, t, &bg).expect("bake forward");
        let (draw_ref, dz_ref, da_ref, db_ref) = bake_gdn_bwd(&qkv, &z, &a, &b, &dout, nb, t, &bg).expect("bake backward");
        let d = enter().unwrap();
        let up = |v: &[f32]| DBuf::from(v).unwrap();
        let fl = |n: usize| DBuf::floats(d.c, n).unwrap();
        let (braw, bz, ba, bb, bdout) = (up(&qkv), up(&z), up(&a), up(&b), up(&dout));
        let (bconv, balog, bdtb, bnorm) = (up(&conv), up(&a_log), up(&dt_bias), up(&norm));
        let p = GdnP { a_log: balog.ptr, dt_bias: bdtb.ptr, norm: bnorm.ptr, conv: bconv.ptr };
        let (pre, cq, out, draw, dz, da, db) = (fl(n * cd), fl(n * cd), fl(n * v), fl(n * cd), fl(n * v), fl(n * nv), fl(n * nv));
        let (dcq, dqk, hist, kvh, oh) = (fl(n * cd), fl(n * nv * 2 * DKV), fl(nb * nv * t * DKV * DKV / 2), fl(nb * nv * t * DKV), fl(nb * nv * t * DKV));
        let f64s = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        // per = nb: the forward keeps the states; per = 1: a replay a sequence
        for per in [nb, 1] {
            let sc = GdnS { dcq: dcq.ptr, dqk: dqk.ptr, hist: hist.ptr, kvh: kvh.ptr, oh: oh.ptr, per };
            unsafe {
                gdn_core_fwd(&d, g, p, braw.ptr, bz.ptr, ba.ptr, bb.ptr, pre.ptr, cq.ptr, out.ptr, Some(&sc)).unwrap();
                gdn_core_bwd(&d, g, p, bz.ptr, ba.ptr, bb.ptr, pre.ptr, cq.ptr, bdout.ptr, draw.ptr, dz.ptr, da.ptr, db.ptr, &sc, true).unwrap();
            }
            rel("gdn out", &get_f32(d.c, out.ptr, n * v).unwrap(), &f64s(&out_ref), 1e-5);
            // the reverse sweep reads the states as f16
            rel("gdn draw", &get_f32(d.c, draw.ptr, n * cd).unwrap(), &f64s(&draw_ref), 1e-3);
            rel("gdn dz", &get_f32(d.c, dz.ptr, n * v).unwrap(), &f64s(&dz_ref), 1e-4);
            rel("gdn da", &get_f32(d.c, da.ptr, n * nv).unwrap(), &f64s(&da_ref), 1e-3);
            rel("gdn db", &get_f32(d.c, db.ptr, n * nv).unwrap(), &f64s(&db_ref), 1e-4);
        }
    }

    /// Times of the recurrence's kernels at the 0.8B trainer's size (8 windows
    /// of 512, 16 v-heads): `--ignored --nocapture`; `CMF_NVG_GDN_GC` picks
    /// the block.
    #[test]
    #[ignore]
    fn cuda_gdn_core_throughput() {
        if enter().is_err() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let nvs: usize = std::env::var("GDN_BENCH_NV").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let (nb, t, nv, nk, kk) = (8usize, 512usize, nvs, 16usize, 4usize);
        let g = GdnG { nb, t, nv, nk, kk, eps: 1e-6 };
        let (n, cd, v) = (nb * t, g.cd(), g.vd());
        let (qkv, z, a, b, dout) = (rnd(n * cd, 1, 0.5), rnd(n * v, 2, 1.0), rnd(n * nv, 3, 1.0), rnd(n * nv, 4, 1.0), rnd(n * v, 5, 1.0));
        let (conv, a_log, dt_bias) = (rnd(cd * kk, 6, 0.3), rnd(nv, 7, 0.5), rnd(nv, 8, 0.5));
        let norm: Vec<f32> = rnd(DKV, 9, 0.1).iter().map(|x| 1.0 + x).collect();
        let d = enter().unwrap();
        let up = |v: &[f32]| DBuf::from(v).unwrap();
        let fl = |n: usize| DBuf::floats(d.c, n).unwrap();
        let (braw, bz, ba, bb, bdout) = (up(&qkv), up(&z), up(&a), up(&b), up(&dout));
        let (bconv, balog, bdtb, bnorm) = (up(&conv), up(&a_log), up(&dt_bias), up(&norm));
        let p = GdnP { a_log: balog.ptr, dt_bias: bdtb.ptr, norm: bnorm.ptr, conv: bconv.ptr };
        let (pre, cq, out, draw, dz, da, db) = (fl(n * cd), fl(n * cd), fl(n * v), fl(n * cd), fl(n * v), fl(n * nv), fl(n * nv));
        let (dcq, dqk, hist, kvh, oh) = (fl(n * cd), fl(n * nv * 2 * DKV), fl(nb * nv * t * DKV * DKV / 2), fl(nb * nv * t * DKV), fl(nb * nv * t * DKV));
        let sc = GdnS { dcq: dcq.ptr, dqk: dqk.ptr, hist: hist.ptr, kvh: kvh.ptr, oh: oh.ptr, per: nb };
        let time = |f: &dyn Fn()| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..4 {
                d.c.sync().unwrap();
                let t0 = std::time::Instant::now();
                f();
                d.c.sync().unwrap();
                best = best.min(t0.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        unsafe {
            let fwd = time(&|| gdn_core_fwd(&d, g, p, braw.ptr, bz.ptr, ba.ptr, bb.ptr, pre.ptr, cq.ptr, out.ptr, None).unwrap());
            let keep = time(&|| gdn_core_fwd(&d, g, p, braw.ptr, bz.ptr, ba.ptr, bb.ptr, pre.ptr, cq.ptr, out.ptr, Some(&sc)).unwrap());
            let rev = time(&|| gdn_core_bwd(&d, g, p, bz.ptr, ba.ptr, bb.ptr, pre.ptr, cq.ptr, bdout.ptr, draw.ptr, dz.ptr, da.ptr, db.ptr, &sc, true).unwrap());
            eprintln!("gdn core, {nb}×{t}, {nv} v-heads, block {}: forward {fwd:.2} ms, forward keeping the states {keep:.2} ms, reverse {rev:.2} ms (conv and the group sum included)", gdn_block());
        }
    }

    /// The attention core (GQA 4/2, two windows) against the bake's entries.
    #[test]
    fn cuda_attn_core_matches_bake() {
        if enter().is_err() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let g = AttnG { nb: 2, t: 17, nh: 4, nkv: 2, hd: 32 };
        let n = g.nb * g.t;
        let (qd, kd, cnt) = (g.nh * g.hd, g.nkv * g.hd, g.nb * g.nh);
        let (qr, kr, v, dao) = (rnd(n * qd, 1, 1.0), rnd(n * kd, 2, 1.0), rnd(n * kd, 3, 1.0), rnd(n * qd, 4, 1.0));
        let mut ao = vec![0f32; n * qd];
        assert!(bake_attn_fwd(&qr, &kr, &v, g.nb, g.t, g.nh, g.nkv, g.hd, None, &mut ao));
        let (mut dq, mut dk, mut dv) = (vec![0f32; n * qd], vec![0f32; n * kd], vec![0f32; n * kd]);
        assert!(bake_attn_bwd(&qr, &kr, &v, &dao, g.nb, g.t, g.nh, g.nkv, g.hd, None, &mut dq, &mut dk, &mut dv));
        let d = enter().unwrap();
        let up = |v: &[f32]| DBuf::from(v).unwrap();
        let fl = |n: usize| DBuf::floats(d.c, n).unwrap();
        let (bq, bk, bv, bdo) = (up(&qr), up(&kr), up(&v), up(&dao));
        let (o, bdq, bdk, bdv) = (fl(n * qd), fl(n * qd), fl(n * kd), fl(n * kd));
        let pl: Vec<DBuf> = (0..7).map(|_| fl(cnt * g.t * g.hd)).collect();
        let sq: Vec<DBuf> = (0..7).map(|_| fl(cnt * g.t * g.t)).collect();
        let s = AttnS {
            qp: pl[0].ptr,
            kp: pl[1].ptr,
            vp: pl[2].ptr,
            op: pl[3].ptr,
            dop: pl[4].ptr,
            dqp: pl[5].ptr,
            dkp: pl[6].ptr,
            dvp: sq[6].ptr,
            sc: sq[0].ptr,
            p: sq[1].ptr,
            pt: sq[2].ptr,
            dp: sq[3].ptr,
            ds: sq[4].ptr,
            dst: sq[5].ptr,
        };
        unsafe {
            attn_core_fwd(&d, g, bq.ptr, bk.ptr, bv.ptr, o.ptr, &s).unwrap();
            attn_core_bwd(&d, g, bdo.ptr, bdq.ptr, bdk.ptr, bdv.ptr, &s).unwrap();
        }
        let f64s = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        rel("attention out", &get_f32(d.c, o.ptr, n * qd).unwrap(), &f64s(&ao), 2e-3);
        rel("attention dq", &get_f32(d.c, bdq.ptr, n * qd).unwrap(), &f64s(&dq), 5e-3);
        rel("attention dk", &get_f32(d.c, bdk.ptr, n * kd).unwrap(), &f64s(&dk), 5e-3);
        rel("attention dv", &get_f32(d.c, bdv.ptr, n * kd).unwrap(), &f64s(&dv), 5e-3);
    }

    /// The head: teacher top-k (ids from the true top of the logits,
    /// probabilities renormalized over them) and the KL loss with its
    /// hidden-state gradient against an f64 reference.
    #[test]
    fn cuda_head_topk_and_kl_match_host() {
        let Ok(d) = enter() else {
            eprintln!("no CUDA device — skipped");
            return;
        };
        let (v, hs, ntok, k) = (300usize, 64usize, 20usize, 8usize);
        let hw = rnd(v * hs, 1, 0.3);
        let net = net_of(&d, PB::default(), Lin::Absent { m: 64 }, hs, 64, Some((&hw, v)), 0);
        drop(d);
        let w = Work::new(&net, 1, ntok).unwrap();
        let d = enter().unwrap();
        let logits = |x: &[f32]| -> Vec<f64> {
            let mut l = vec![0f64; ntok * v];
            for t in 0..ntok {
                for j in 0..v {
                    l[t * v + j] = (0..hs).map(|c| x[t * hs + c] as f64 * hw[j * hs + c] as f64).sum();
                }
            }
            l
        };
        // teacher top-k
        let x = rnd(ntok * hs, 2, 1.0);
        put(d.c, w.p("n1"), &x).unwrap();
        let (tid, tp) = topk_n1(&d, &net, &w, k).unwrap();
        let lt = logits(&x);
        for t in 0..ntok {
            let mut row: Vec<f64> = lt[t * v..(t + 1) * v].to_vec();
            row.sort_by(|a, b| b.total_cmp(a));
            let (ids, ps) = (&tid[t * k..(t + 1) * k], &tp[t * k..(t + 1) * k]);
            assert!((ps.iter().map(|&p| p as f64).sum::<f64>() - 1.0).abs() < 1e-4, "row {t}: probabilities sum");
            assert!(ps.windows(2).all(|p| p[0] >= p[1]), "row {t}: not sorted");
            for &id in ids {
                assert!(lt[t * v + id as usize] >= row[k - 1] - 2e-2, "row {t}: token {id} is not in the top {k}");
            }
            let z: f64 = ids.iter().map(|&id| (lt[t * v + id as usize] - row[0]).exp()).sum();
            for (q, &id) in ids.iter().enumerate() {
                let want = (lt[t * v + id as usize] - row[0]).exp() / z;
                assert!((ps[q] as f64 - want).abs() < 5e-3, "row {t}: p[{q}] {} vs {want}", ps[q]);
            }
        }
        // the student's KL to those rows and its gradient
        let x2: Vec<f32> = x.iter().zip(rnd(ntok * hs, 3, 0.3)).map(|(a, b)| a + b).collect();
        put(d.c, w.p("n1"), &x2).unwrap();
        unsafe { head_kl(&d, &net, &w, &tid, &tp, k, ntok as f32, 1024.0, false, true).unwrap() };
        let ls = logits(&x2);
        let (mut loss, mut dx) = (vec![0f64; ntok], vec![0f64; ntok * hs]);
        for t in 0..ntok {
            let row = &ls[t * v..(t + 1) * v];
            let mx = row.iter().cloned().fold(f64::MIN, f64::max);
            let lse = mx + row.iter().map(|s| (s - mx).exp()).sum::<f64>().ln();
            let mut dl: Vec<f64> = row.iter().map(|s| (s - lse).exp() / ntok as f64).collect();
            for q in 0..k {
                let (id, pt) = (tid[t * k + q] as usize, tp[t * k + q] as f64);
                loss[t] += pt * (pt.ln() - (row[id] - lse));
                dl[id] -= pt / ntok as f64;
            }
            for c in 0..hs {
                dx[t * hs + c] = (0..v).map(|j| dl[j] * hw[j * hs + c] as f64).sum();
            }
        }
        rel("KL per token", &get_f32(d.c, w.p("loss"), ntok).unwrap(), &loss, 5e-3);
        rel("dKL/dh", &get_f32(d.c, w.p("dn1"), ntok * hs).unwrap(), &dx, 5e-3);
    }
}
