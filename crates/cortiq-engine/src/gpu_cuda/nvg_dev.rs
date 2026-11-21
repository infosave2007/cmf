//! Device linear algebra for the NVG `blr` encoder on CUDA (docs/BLR_NVG.md
//! §4.1): the API of `gpu_wgpu::nvg_dev`, f32 on the card, so the encoder's
//! driver (`cortiq-cli` `nvg_gpu.rs`) runs on either backend through
//! `crate::nvg_dev`.
//!
//! * dense products — cuBLAS SGEMM in plain f32 over sub-matrix views (the
//!   row-major operands read column-major are their transposes, so no copy
//!   is made); a Hadamard epilogue is one more elementwise launch;
//! * SPD solves — an equilibrated blocked Cholesky (64- or 128-wide diagonal blocks
//!   factored on the host in f64, the panel by STRSM, the trailing matrix by
//!   SSYRK), triangular solves and inverses by STRSV / STRSM;
//! * the elementwise / reduction kernels, the blocked output-side flips, the
//!   input-side column search of one 128-column block and the Householder
//!   steps are `nvg_dev.cu` (NVRTC), the arithmetic of the WGSL kernels;
//! * the r largest eigenpairs — Householder tridiagonalisation on the card,
//!   bisection + inverse iteration of the tridiagonal on the host in f64
//!   (`gpu_wgpu::nvg_dev::tridiag_top`), a compact-WY back-transform on the card.
//!
//! Every entry takes the context's lock and queues on the backend's one
//! stream, so calls from several encoder threads interleave in a fixed order
//! and a buffer returned to the pool is reused only by work queued after its
//! last reader. Readbacks wait for the stream.

use super::*;
use std::sync::Mutex;

const ND_SRC: &str = include_str!("nvg_dev.cu");

/// Scaling flags of [`scale_rc`] (the values of `gpu_wgpu::nvg_dev`).
pub const ROW: u32 = 1;
pub const COL: u32 = 2;
pub const ACC: u32 = 4;
pub const LOWER: u32 = 8;
pub const SIGN: u32 = 16;
pub const ABS: u32 = 32;
pub const INPLACE: u32 = 64;

/// The largest rank [`gv_block`] takes (4 lanes per thread × 1024).
pub const GV_MAX_RANK: usize = 4096;

/// Diagonal block of the blocked Cholesky of an `n×n` matrix
/// (`CMF_NVG_CUDA_CHOL_NB`; default 64 up to n = 8192, 128 past it). The
/// host factors a block in f64 while the card waits, so the block is small:
/// on an RTX PRO 6000 n = 5120 took 8.3 / 11.8 / 33.2 ms and n = 17408
/// 98.9 / 86.3 / 157.5 ms with 64 / 128 / 256.
fn chol_nb(n: usize) -> usize {
    static NB: OnceLock<Option<usize>> = OnceLock::new();
    NB.get_or_init(|| std::env::var("CMF_NVG_CUDA_CHOL_NB").ok().and_then(|v| v.parse::<usize>().ok()).filter(|v| (16..=1024).contains(v)))
        .unwrap_or(if n <= 8192 { 64 } else { 128 })
}

#[derive(Clone, Copy)]
struct NdFuncs {
    scale_rc: cs::CUfunction,
    rowdot: cs::CUfunction,
    rownorm: cs::CUfunction,
    colsum_part: cs::CUfunction,
    colsum_red: cs::CUfunction,
    frob_part: cs::CUfunction,
    diag: cs::CUfunction,
    equil: cs::CUfunction,
    make_uc: cs::CUfunction,
    permute_rows: cs::CUfunction,
    transpose: cs::CUfunction,
    had: cs::CUfunction,
    eye: cs::CUfunction,
    diag_ridge: cs::CUfunction,
    equil_rg: cs::CUfunction,
    div_vec: cs::CUfunction,
    hsolve: cs::CUfunction,
    scatter_vec: cs::CUfunction,
    sum_into: cs::CUfunction,
    scale_rsqrt: cs::CUfunction,
    flipblk: cs::CUfunction,
    flipu: cs::CUfunction,
    hh_vec: cs::CUfunction,
    hh_matvec: cs::CUfunction,
    hh_w: cs::CUfunction,
    hh_rank2: cs::CUfunction,
    hh_last: cs::CUfunction,
    vblock: cs::CUfunction,
    wy_t: cs::CUfunction,
}

struct NdState {
    f: NdFuncs,
    /// `nd_gv_block` by lanes per thread (the module is compiled per NQ).
    gv: Mutex<std::collections::HashMap<usize, cs::CUfunction>>,
}

// SAFETY: function handles of the backend's context — launched with that
// context current, on the context's stream under its lock or on a thread's
// private lane.
unsafe impl Send for NdState {}
unsafe impl Sync for NdState {}

static STATE: OnceLock<Result<NdState, String>> = OnceLock::new();

/// The stream and the cuBLAS handle a thread's encoder calls queue on.
#[derive(Clone, Copy)]
struct Lane {
    stream: cs::CUstream,
    blas: bl::cublasHandle_t,
}

// SAFETY: handles of the backend's context; a lane is one thread's at a time.
unsafe impl Send for Lane {}

/// Lanes not held by a thread (a cuBLAS handle takes a while to create: they
/// are made once and handed on).
static LANES: Mutex<Vec<Lane>> = Mutex::new(Vec::new());

thread_local! {
    /// The calling thread's private lane, while it holds a [`LaneGuard`].
    static MY_LANE: std::cell::Cell<Option<Lane>> = const { std::cell::Cell::new(None) };
    /// Free buffers of the thread's lane, by size: a buffer a thread freed may
    /// still be read by work queued on that thread's stream, so only that
    /// thread takes it again (its own stream keeps the order).
    static MY_POOL: std::cell::RefCell<std::collections::HashMap<usize, Vec<u64>>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// What an encoder call runs on: the backend's context with the stream and
/// the cuBLAS handle of the calling thread — the context's own (every thread
/// in one order, under the context's lock), or the thread's private lane.
struct E {
    c: &'static Ctx,
    stream: cs::CUstream,
    blas: bl::cublasHandle_t,
    arch: Option<&'static str>,
}

impl E {
    unsafe fn launch(&self, f: cs::CUfunction, grid: (u32, u32), block: u32, shared: usize, params: &mut [*mut c_void]) -> Res<()> {
        unsafe { self.c.launch_on(self.stream, f, grid, block, shared, params) }
    }

    fn sync(&self) -> Res<()> {
        unsafe { cu::stream::synchronize(self.stream) }.map_err(derr)
    }

    /// `C = α·op(A)·op(B) + β·C`, f32, column-major as cuBLAS reads it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sgemm(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize, alpha: f32, a: u64, lda: usize, b: u64, ldb: usize, beta: f32, cp: u64, ldc: usize) -> Res<()> {
        let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
        let s = unsafe {
            bl::cublasSgemm_v2(self.blas, op(ta), op(tb), m as i32, n as i32, k as i32, &alpha, a as *const f32, lda as i32, b as *const f32, ldb as i32, &beta, cp as *mut f32, ldc as i32)
        };
        if s as u32 != 0 {
            return Err(format!("cuBLAS gemm {m}x{n}x{k}: {s:?}"));
        }
        Ok(())
    }
}

/// A private lane for the calling thread until the guard drops: its encoder
/// calls queue on a stream and a cuBLAS handle of their own and take no
/// lock, so the tensors of several threads overlap on the card (on the
/// context's one stream a tensor's column search — one block, one
/// multiprocessor, tens of milliseconds a launch — holds every other tensor
/// back). Matrices made before are visible (the context's stream is drained
/// first); the thread that made them must not use the encoder until the
/// lanes are gone (the driver's thread waits for its job threads). `None`:
/// no card, a lane already held, or `CMF_NVG_CUDA_LANES=0`.
pub fn private_lane() -> Option<LaneGuard> {
    if std::env::var("CMF_NVG_CUDA_LANES").as_deref() == Ok("0") || MY_LANE.with(|l| l.get().is_some()) {
        return None;
    }
    let c = ctx()?;
    let _st = c.st.lock().unwrap();
    c.current().ok()?;
    // everything queued on the context's stream is done: the shared pool's
    // buffers are free to be written from another stream
    c.sync().ok()?;
    let lane = match LANES.lock().unwrap().pop() {
        Some(l) => l,
        None => {
            let stream = cu::stream::create(cu::stream::StreamKind::NonBlocking).ok()?;
            let mut blas: bl::cublasHandle_t = std::ptr::null_mut();
            if unsafe { bl::cublasCreate_v2(&mut blas) } as u32 != 0 || unsafe { bl::cublasSetStream_v2(blas, stream.cast()) } as u32 != 0 {
                return None;
            }
            Lane { stream, blas }
        }
    };
    MY_LANE.with(|l| l.set(Some(lane)));
    Some(LaneGuard { _thread: std::marker::PhantomData })
}

/// See [`private_lane`]. Dropped on the thread that took it.
pub struct LaneGuard {
    _thread: std::marker::PhantomData<*const ()>,
}

impl Drop for LaneGuard {
    fn drop(&mut self) {
        let Some(lane) = MY_LANE.with(|l| l.take()) else { return };
        if let Some(c) = ctx() {
            let _ = c.current();
        }
        // the lane's work is done before its buffers go to the shared pool
        let _ = unsafe { cu::stream::synchronize(lane.stream) };
        let mine = MY_POOL.with(|m| std::mem::take(&mut *m.borrow_mut()));
        let mut p = POOL.lock().unwrap();
        for (bytes, v) in mine {
            p.held += bytes * v.len();
            p.free.entry(bytes).or_default().extend(v);
        }
        drop(p);
        LANES.lock().unwrap().push(lane);
    }
}

fn build(c: &Ctx) -> Res<NdState> {
    let m = compile(ND_SRC, vec!["-DNQ=1".into()], c.arch)?;
    let f = |n: &str| func(m, n);
    Ok(NdState {
        f: NdFuncs {
            scale_rc: f("nd_scale_rc")?,
            rowdot: f("nd_rowdot")?,
            rownorm: f("nd_rownorm")?,
            colsum_part: f("nd_colsum_part")?,
            colsum_red: f("nd_colsum_red")?,
            frob_part: f("nd_frob_part")?,
            diag: f("nd_diag")?,
            equil: f("nd_equil")?,
            make_uc: f("nd_make_uc")?,
            permute_rows: f("nd_permute_rows")?,
            transpose: f("nd_transpose")?,
            had: f("nd_had")?,
            eye: f("nd_eye")?,
            diag_ridge: f("nd_diag_ridge")?,
            equil_rg: f("nd_equil_rg")?,
            div_vec: f("nd_div_vec")?,
            hsolve: f("nd_hsolve")?,
            scatter_vec: f("nd_scatter_vec")?,
            sum_into: f("nd_sum_into")?,
            scale_rsqrt: f("nd_scale_rsqrt")?,
            flipblk: f("nd_flipblk")?,
            flipu: f("nd_flipu")?,
            hh_vec: f("nd_hh_vec")?,
            hh_matvec: f("nd_hh_matvec")?,
            hh_w: f("nd_hh_w")?,
            hh_rank2: f("nd_hh_rank2")?,
            hh_last: f("nd_hh_last")?,
            vblock: f("nd_vblock")?,
            wy_t: f("nd_wy_t")?,
        },
        gv: Mutex::new(std::collections::HashMap::new()),
    })
}

/// Run `f` with the context current and the kernels built, on the calling
/// thread's lane: the context's own stream under its lock, or the thread's
/// private lane with no lock ([`private_lane`]).
/// A device error is a panic with its message (the wgpu twin `expect`s too):
/// the encoder has no partial result to fall back on.
fn with<R>(f: impl FnOnce(&E, &NdState) -> Res<R>) -> R {
    let c = ctx().expect("nvg_dev: no CUDA device");
    let lane = MY_LANE.with(|l| l.get());
    let _st = if lane.is_none() { Some(c.st.lock().unwrap()) } else { None };
    c.current().expect("nvg_dev: CUDA context");
    let st = match STATE.get_or_init(|| build(c)) {
        Ok(s) => s,
        Err(e) => panic!("nvg_dev (CUDA) kernels: {e}"),
    };
    let e = match lane {
        Some(l) => E { c, stream: l.stream, blas: l.blas, arch: c.arch },
        None => E { c, stream: c.stream, blas: c.blas, arch: c.arch },
    };
    match f(&e, st) {
        Ok(v) => v,
        Err(err) => {
            let _ = e.sync();
            panic!("nvg_dev (CUDA): {err}");
        }
    }
}

/// The card is up and the encoder's kernels compile on it.
pub fn available() -> bool {
    let Some(c) = ctx() else { return false };
    let _st = c.st.lock().unwrap();
    if c.current().is_err() {
        return false;
    }
    match STATE.get_or_init(|| build(c)) {
        Ok(_) => true,
        Err(e) => {
            decline("nvg encoder kernels", e);
            false
        }
    }
}

// ───────────────────────────── buffers ─────────────────────────────

/// Freed device buffers by size, reused by later allocations of that size
/// (the encoder asks for the same shapes over and over; `cuMemAlloc` /
/// `cuMemFree` per temporary would synchronize the card each time).
struct PoolState {
    free: std::collections::HashMap<usize, Vec<u64>>,
    held: usize,
}

static POOL: std::sync::LazyLock<Mutex<PoolState>> = std::sync::LazyLock::new(|| Mutex::new(PoolState { free: std::collections::HashMap::new(), held: 0 }));

/// Bytes of free buffers the pool keeps (`CMF_NVG_CUDA_POOL_GB`, default 24).
fn pool_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| std::env::var("CMF_NVG_CUDA_POOL_GB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(24) << 30)
}

fn round_bytes(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(256)
}

/// Free every pooled buffer — the shared pool's and the calling lane's own
/// (the lane's stream drained first: queued work may read them; the shared
/// pool's buffers have no work queued while private lanes are out).
fn pool_trim(c: &E) {
    let _ = c.sync();
    let mine = MY_POOL.with(|m| std::mem::take(&mut *m.borrow_mut()));
    for ptr in mine.into_values().flatten() {
        unsafe {
            let _ = cu::free_sync(ptr);
        }
    }
    let mut p = POOL.lock().unwrap();
    for (_, v) in p.free.drain() {
        for ptr in v {
            unsafe {
                let _ = cu::free_sync(ptr);
            }
        }
    }
    p.held = 0;
}

/// `bytes` of device memory, contents undefined. Context current.
fn raw_alloc(c: &E, bytes: usize) -> Res<u64> {
    let bytes = round_bytes(bytes);
    if MY_LANE.with(|l| l.get().is_some())
        && let Some(ptr) = MY_POOL.with(|m| m.borrow_mut().get_mut(&bytes).and_then(Vec::pop))
    {
        return Ok(ptr);
    }
    {
        let mut p = POOL.lock().unwrap();
        if let Some(ptr) = p.free.get_mut(&bytes).and_then(Vec::pop) {
            p.held -= bytes;
            return Ok(ptr);
        }
    }
    if POOL.lock().unwrap().held > pool_cap() {
        pool_trim(c);
    }
    match unsafe { cu::malloc_sync(bytes) } {
        Ok(p) => Ok(p),
        Err(_) => {
            pool_trim(c);
            unsafe { cu::malloc_sync(bytes) }.map_err(|e| format!("device allocation of {bytes} bytes: {e}"))
        }
    }
}

fn raw_release(ptr: u64, bytes: usize) {
    let bytes = round_bytes(bytes);
    // a private lane keeps what it frees (see `MY_POOL`); a thread past its
    // thread-locals' end falls through to the shared pool
    if MY_LANE.try_with(|l| l.get().is_some()).unwrap_or(false) && MY_POOL.try_with(|m| m.borrow_mut().entry(bytes).or_default().push(ptr)).is_ok() {
        return;
    }
    let mut p = POOL.lock().unwrap();
    p.free.entry(bytes).or_default().push(ptr);
    p.held += bytes;
}

/// A scoped temporary (returned to the pool on drop).
struct Tmp {
    ptr: u64,
    bytes: usize,
}

impl Tmp {
    fn new(c: &E, floats: usize) -> Res<Tmp> {
        let bytes = floats.max(4) * 4;
        Ok(Tmp { ptr: raw_alloc(c, bytes)?, bytes })
    }
    fn zeroed(c: &E, floats: usize) -> Res<Tmp> {
        let t = Tmp::new(c, floats)?;
        unsafe { cu::memset_d8_async(t.ptr, 0, t.bytes, c.stream).map_err(derr)? };
        Ok(t)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        raw_release(self.ptr, self.bytes);
    }
}

/// A row-major f32 matrix on the card.
pub struct DMat {
    ptr: u64,
    bytes: usize,
    pub r: usize,
    pub c: usize,
}

// SAFETY: a device address; every use goes through the context's lock.
unsafe impl Send for DMat {}
unsafe impl Sync for DMat {}

impl Drop for DMat {
    fn drop(&mut self) {
        raw_release(self.ptr, self.bytes);
    }
}

/// A u32 device vector (permutations).
pub struct U32Buf {
    ptr: u64,
    bytes: usize,
}

unsafe impl Send for U32Buf {}
unsafe impl Sync for U32Buf {}

impl Drop for U32Buf {
    fn drop(&mut self) {
        raw_release(self.ptr, self.bytes);
    }
}

fn mat_raw(c: &E, r: usize, cc: usize) -> Res<DMat> {
    let bytes = (r * cc).max(4) * 4;
    Ok(DMat { ptr: raw_alloc(c, bytes)?, bytes, r, c: cc })
}

fn mat_zero(c: &E, r: usize, cc: usize) -> Res<DMat> {
    let m = mat_raw(c, r, cc)?;
    unsafe { cu::memset_d8_async(m.ptr, 0, m.bytes, c.stream).map_err(derr)? };
    Ok(m)
}

/// Host memory into a (possibly reused) device buffer: the stream drained
/// first — work queued earlier may still read the buffer's previous contents.
fn upload<T: Copy>(c: &E, ptr: u64, data: &[T]) -> Res<()> {
    if data.is_empty() {
        return Ok(());
    }
    c.sync()?;
    unsafe { htod(ptr, data) }
}

fn download_f32(c: &E, ptr: u64, n: usize) -> Res<Vec<f32>> {
    let mut v = vec![0f32; n];
    if n > 0 {
        unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), ptr, c.stream).map_err(derr)? };
    }
    c.sync()?;
    Ok(v)
}

fn download_u32(c: &E, ptr: u64, n: usize) -> Res<Vec<u32>> {
    let mut v = vec![0u32; n];
    if n > 0 {
        unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), ptr, c.stream).map_err(derr)? };
    }
    c.sync()?;
    Ok(v)
}

/// f64 → f32 over all threads (a Gram at n = 17408 is 303 M elements).
fn to_f32_par(v: &[f64]) -> Vec<f32> {
    let mut f = vec![0f32; v.len()];
    let per = v.len().div_ceil(crate::pool::Pool::effective_threads().max(1)).max(1 << 16);
    std::thread::scope(|s| {
        for (d, src) in f.chunks_mut(per).zip(v.chunks(per)) {
            s.spawn(move || d.iter_mut().zip(src).for_each(|(o, &x)| *o = x as f32));
        }
    });
    f
}

impl DMat {
    pub fn zeros(r: usize, c: usize) -> DMat {
        with(|cx, _| mat_zero(cx, r, c))
    }
    pub fn from_f32(r: usize, c: usize, v: &[f32]) -> DMat {
        assert_eq!(v.len(), r * c);
        with(|cx, _| {
            let m = if v.is_empty() { mat_zero(cx, r, c)? } else { mat_raw(cx, r, c)? };
            upload(cx, m.ptr, v)?;
            Ok(m)
        })
    }
    pub fn from_f64(r: usize, c: usize, v: &[f64]) -> DMat {
        DMat::from_f32(r, c, &to_f32_par(v))
    }
    pub fn write_f64(&self, v: &[f64]) {
        assert!(v.len() <= self.r * self.c);
        let f = to_f32_par(v);
        with(|cx, _| upload(cx, self.ptr, &f));
    }
    pub fn len(&self) -> usize {
        self.r * self.c
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn to_f32(&self) -> Vec<f32> {
        with(|cx, _| download_f32(cx, self.ptr, self.len()))
    }
    pub fn to_f64(&self) -> Vec<f64> {
        self.to_f32().into_iter().map(|x| x as f64).collect()
    }
    /// A device copy.
    pub fn dup(&self) -> DMat {
        with(|cx, _| {
            let o = mat_raw(cx, self.r, self.c)?;
            copy_raw(cx, self.ptr, o.ptr, self.len())?;
            Ok(o)
        })
    }
}

fn copy_raw(c: &E, src: u64, dst: u64, floats: usize) -> Res<()> {
    if floats == 0 {
        return Ok(());
    }
    unsafe { cs::cuMemcpyDtoDAsync_v2(dst, src, floats * 4, c.stream) }.result().map_err(derr)
}

/// `dst ← src` (same element count).
pub fn copy(src: &DMat, dst: &DMat) {
    assert_eq!(src.len(), dst.len());
    with(|c, _| copy_raw(c, src.ptr, dst.ptr, src.len()));
}

/// Copy `len` floats `src[so..] → dst[do_..]`.
pub fn copy_range(src: &DMat, so: usize, dst: &DMat, do_: usize, len: usize) {
    with(|c, _| copy_raw(c, src.ptr + (so * 4) as u64, dst.ptr + (do_ * 4) as u64, len));
}

/// Wait until everything queued so far has finished (stage timing).
pub fn sync() {
    with(|c, _| c.sync());
}

/// A u32 device vector.
pub fn upload_u32(v: &[u32]) -> U32Buf {
    with(|c, _| {
        let bytes = v.len().max(4) * 4;
        let b = U32Buf { ptr: raw_alloc(c, bytes)?, bytes };
        upload(c, b.ptr, v)?;
        Ok(b)
    })
}

// ───────────────────────────── launches ─────────────────────────────

/// Grid of an elementwise kernel over `n` elements (256 threads a block).
fn egrid(n: usize) -> (u32, u32) {
    (n.div_ceil(256).max(1) as u32, 1)
}

/// Grid of a kernel over a `rows × cols` matrix: its row from y, its column
/// from x (256 threads a block).
fn mgrid(rows: usize, cols: usize) -> (u32, u32) {
    assert!(rows <= 65535, "nvg_dev (CUDA): {rows} rows in one launch");
    (cols.div_ceil(256).max(1) as u32, rows.max(1) as u32)
}

fn fl(off: usize) -> u64 {
    (off * 4) as u64
}

// ───────────────────────────── GEMM ─────────────────────────────

/// An operand view: element (i, k) of `op(M)` is `M[off + i·ld + k]`, or
/// `M[off + k·ld + i]` when `t`.
#[derive(Clone, Copy)]
pub struct V<'a> {
    pub m: &'a DMat,
    pub off: usize,
    pub ld: usize,
    pub t: bool,
}

impl<'a> V<'a> {
    /// The whole matrix, as stored.
    pub fn n(m: &'a DMat) -> V<'a> {
        V { m, off: 0, ld: m.c, t: false }
    }
    /// The whole matrix, transposed.
    pub fn t(m: &'a DMat) -> V<'a> {
        V { m, off: 0, ld: m.c, t: true }
    }
}

/// `C[i·ldc + j] = α·Σ_k op(A)(i,k)·op(B)(k,j) + β·C` over device addresses.
/// Read column-major, a row-major operand is its transpose: the product is
/// `Cᵀ = op(B)ᵀ·op(A)ᵀ` with each stored array taken as it lies.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_raw(c: &E, m: usize, n: usize, k: usize, alpha: f32, a: u64, lda: usize, ta: bool, b: u64, ldb: usize, tb: bool, beta: f32, cp: u64, ldc: usize) -> Res<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        return Err("gemm with an empty inner dimension".into());
    }
    unsafe { c.sgemm(tb, ta, n, m, k, alpha, b, ldb, a, lda, beta, cp, ldc) }
}

/// `C[oc + i·ldc + j] = α·Σ_k op(A)(i,k)·op(B)(k,j) [⊙ E] + β·C` for an
/// `m×n` output (`k` the inner dimension). `E` (same layout as C) when given.
#[allow(clippy::too_many_arguments)]
pub fn gemm_v(m: usize, n: usize, k: usize, alpha: f32, a: V, b: V, beta: f32, c: &DMat, oc: usize, ldc: usize, e: Option<&DMat>) {
    if m == 0 || n == 0 {
        return;
    }
    with(|cx, s| unsafe {
        let (ap, bp) = (a.m.ptr + fl(a.off), b.m.ptr + fl(b.off));
        match e {
            None => gemm_raw(cx, m, n, k, alpha, ap, a.ld, a.t, bp, b.ld, b.t, beta, c.ptr + fl(oc), ldc),
            Some(e) => {
                let t = Tmp::new(cx, m * n)?;
                gemm_raw(cx, m, n, k, alpha, ap, a.ld, a.t, bp, b.ld, b.t, 0.0, t.ptr, n)?;
                let (mi, ni, ldi, oi, acc) = (m as u32, n as u32, ldc as u32, oc as u32, (beta != 0.0) as u32);
                cx.launch(s.f.had, mgrid(m, n), 256, 0, kargs!(t.ptr, e.ptr, c.ptr, mi, ni, ldi, oi, beta, acc))
            }
        }
    });
}

/// `C = α·op(A)·op(B) + β·C` on whole matrices.
pub fn gemm(alpha: f32, a: &DMat, ta: bool, b: &DMat, tb: bool, beta: f32, c: &DMat) {
    let (m, k) = if ta { (a.c, a.r) } else { (a.r, a.c) };
    let (k2, n) = if tb { (b.c, b.r) } else { (b.r, b.c) };
    assert_eq!(k, k2, "nvg gemm inner dims");
    assert!(c.r == m && c.c == n, "nvg gemm output dims");
    let va = V { m: a, off: 0, ld: a.c, t: ta };
    let vb = V { m: b, off: 0, ld: b.c, t: tb };
    gemm_v(m, n, k, alpha, va, vb, beta, c, 0, n, None);
}

/// `op(A)·op(B)` into a new matrix.
pub fn mm(a: &DMat, ta: bool, b: &DMat, tb: bool) -> DMat {
    let (m, k) = if ta { (a.c, a.r) } else { (a.r, a.c) };
    let (k2, n) = if tb { (b.c, b.r) } else { (b.r, b.c) };
    assert_eq!(k, k2, "nvg gemm inner dims");
    with(|cx, _| unsafe {
        let c = mat_raw(cx, m, n)?;
        gemm_raw(cx, m, n, k, 1.0, a.ptr, a.c, ta, b.ptr, b.c, tb, 0.0, c.ptr, n)?;
        Ok(c)
    })
}

// ───────────────────────────── elementwise / reductions ─────────────────────────────

#[allow(clippy::too_many_arguments)]
unsafe fn scale_rc_raw(cx: &E, f: &NdFuncs, rows: usize, cols: usize, a: u64, lda: usize, oa: usize, c: u64, ldc: usize, oc: usize, r: Option<u64>, s: Option<u64>, flags: u32, alpha: f32, beta: f32) -> Res<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let flags = flags | if r.is_some() { ROW } else { 0 } | if s.is_some() { COL } else { 0 };
    let (rp, sp) = (r.unwrap_or(c), s.unwrap_or(c));
    let (ri, ci, ldai, oai, ldci, oci) = (rows as u32, cols as u32, lda as u32, oa as u32, ldc as u32, oc as u32);
    unsafe { cx.launch(f.scale_rc, mgrid(rows, cols), 256, 0, kargs!(a, rp, sp, c, ri, ci, ldai, oai, ldci, oci, flags, alpha, beta)) }
}

/// `C[oc + i·ldc + j] = α·f(X(i,j))·R[i]·S[j] (+ β·C)` over `rows × cols`,
/// `X(i,j) = A[oa + i·lda + j]` (or C itself with [`INPLACE`]).
#[allow(clippy::too_many_arguments)]
pub fn scale_rc(rows: usize, cols: usize, a: Option<(&DMat, usize, usize)>, c: (&DMat, usize, usize), r: Option<&DMat>, s: Option<&DMat>, flags: u32, alpha: f32, beta: f32) {
    if rows == 0 || cols == 0 {
        return;
    }
    with(|cx, st| unsafe {
        let (ap, oa, lda) = match a {
            Some((m, off, ld)) => (m.ptr, off, ld),
            None => (c.0.ptr, 0, 0),
        };
        scale_rc_raw(cx, &st.f, rows, cols, ap, lda, oa, c.0.ptr, c.2, c.1, r.map(|x| x.ptr), s.map(|x| x.ptr), flags, alpha, beta)
    });
}

/// Whole-matrix `C = α·f(A)·diag-scalings`.
pub fn scale_into(a: &DMat, c: &DMat, r: Option<&DMat>, s: Option<&DMat>, flags: u32, alpha: f32) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    scale_rc(a.r, a.c, Some((a, 0, a.c)), (c, 0, c.c), r, s, flags, alpha, 0.0);
}

/// `C ← α·A + β·C` (whole matrices).
pub fn axpby(alpha: f32, a: &DMat, beta: f32, c: &DMat) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    scale_rc(a.r, a.c, Some((a, 0, a.c)), (c, 0, c.c), None, None, ACC, alpha, beta);
}

/// `O[i] = Σ_j A[oa+i·lda+j]·B[ob+i·ldb+j]` for `rows` rows of `cols` columns.
#[allow(clippy::too_many_arguments)]
pub fn rowdot(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize, out: &DMat) {
    if rows == 0 {
        return;
    }
    with(|cx, s| unsafe {
        let (ci, ldai, oai, ldbi, obi) = (cols as u32, lda as u32, oa as u32, ldb as u32, ob as u32);
        cx.launch(s.f.rowdot, (rows as u32, 1), 256, 0, kargs!(a.ptr, b.ptr, out.ptr, ci, ldai, oai, ldbi, obi))
    });
}

/// Row-normalise `rows × cols` of `c` (at `oc`, leading dimension `ldc`) in place.
pub fn rownorm(rows: usize, cols: usize, c: &DMat, oc: usize, ldc: usize) {
    if rows == 0 {
        return;
    }
    with(|cx, s| unsafe {
        let (ci, ldci, oci) = (cols as u32, ldc as u32, oc as u32);
        cx.launch(s.f.rownorm, (rows as u32, 1), 256, 0, kargs!(c.ptr, ci, ldci, oci))
    });
}

#[allow(clippy::too_many_arguments)]
unsafe fn colsum_part_raw(cx: &E, f: &NdFuncs, rows: usize, cols: usize, a: u64, oa: usize, lda: usize, b: u64, ob: usize, ldb: usize) -> Res<(Tmp, usize)> {
    let per = 64usize.max(rows.div_ceil(256));
    let nch = rows.div_ceil(per).max(1);
    let part = Tmp::new(cx, nch * cols)?;
    let (ri, ci, ldai, oai, ldbi, obi, peri) = (rows as u32, cols as u32, lda as u32, oa as u32, ldb as u32, ob as u32, per as u32);
    unsafe { cx.launch(f.colsum_part, (cols.div_ceil(256).max(1) as u32, nch as u32), 256, 0, kargs!(a, b, part.ptr, ri, ci, ldai, oai, ldbi, obi, peri))? };
    Ok((part, nch))
}

/// `Σ_i A[oa+i·lda+j]·B[ob+i·ldb+j]` per column, summed in f64 on the host.
#[allow(clippy::too_many_arguments)]
pub fn colsum_prod(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize) -> Vec<f64> {
    with(|cx, s| unsafe {
        let (part, nch) = colsum_part_raw(cx, &s.f, rows, cols, a.ptr, oa, lda, b.ptr, ob, ldb)?;
        let v = download_f32(cx, part.ptr, nch * cols)?;
        let mut out = vec![0f64; cols];
        for ch in 0..nch {
            for j in 0..cols {
                out[j] += v[ch * cols + j] as f64;
            }
        }
        Ok(out)
    })
}

/// `out[oo + j] = Σ_i A[oa+i·lda+j]·B[ob+i·ldb+j]` on the device (no readback).
#[allow(clippy::too_many_arguments)]
pub fn colsum_dev(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize, out: &DMat, oo: usize) {
    with(|cx, s| unsafe {
        let (part, nch) = colsum_part_raw(cx, &s.f, rows, cols, a.ptr, oa, lda, b.ptr, ob, ldb)?;
        let (nchi, ci, ooi) = (nch as u32, cols as u32, oo as u32);
        cx.launch(s.f.colsum_red, (cols.div_ceil(256).max(1) as u32, 1), 256, 0, kargs!(part.ptr, out.ptr, nchi, ci, ooi))
    });
}

/// Workgroups of a Frobenius reduction over `len` elements.
fn frob_nwg(len: usize) -> usize {
    512usize.min(len.div_ceil(256)).max(1)
}

unsafe fn frob_part_raw(cx: &E, f: &NdFuncs, a: u64, b: u64, len: usize, out: u64) -> Res<usize> {
    let nwg = frob_nwg(len);
    let (leni, nwgi) = (len as u64, nwg as u32);
    unsafe { cx.launch(f.frob_part, (nwg as u32, 1), 256, 0, kargs!(a, b, out, leni, nwgi))? };
    Ok(nwg)
}

/// `Σ A ⊙ B` over the first `len` elements, f64 on the host.
pub fn frob(a: &DMat, b: &DMat, len: usize) -> f64 {
    with(|cx, s| unsafe {
        let part = Tmp::new(cx, frob_nwg(len))?;
        let nwg = frob_part_raw(cx, &s.f, a.ptr, b.ptr, len, part.ptr)?;
        Ok(download_f32(cx, part.ptr, nwg)?.iter().map(|&x| x as f64).sum())
    })
}

/// Several `Σ A⊙B` read back together: `add` queues one (no readback),
/// `read` returns them all in f64 after one wait.
pub struct FrobBatch {
    parts: Vec<(Tmp, usize)>,
}

impl FrobBatch {
    pub fn new() -> Self {
        FrobBatch { parts: Vec::new() }
    }
    /// Queue `Σ_{i<len} a[i]·b[i]`; returns its slot.
    pub fn add(&mut self, a: &DMat, b: &DMat, len: usize) -> usize {
        let p = with(|cx, s| unsafe {
            let part = Tmp::new(cx, frob_nwg(len))?;
            let nwg = frob_part_raw(cx, &s.f, a.ptr, b.ptr, len, part.ptr)?;
            Ok((part, nwg))
        });
        self.parts.push(p);
        self.parts.len() - 1
    }
    pub fn read(self) -> Vec<f64> {
        with(|cx, _| {
            let mut vs: Vec<Vec<f32>> = Vec::with_capacity(self.parts.len());
            for (p, n) in &self.parts {
                let mut v = vec![0f32; *n];
                unsafe { cu::memcpy_dtoh_async(v.as_mut_slice(), p.ptr, cx.stream).map_err(derr)? };
                vs.push(v);
            }
            cx.sync()?;
            Ok(vs.iter().map(|v| v.iter().map(|&x| x as f64).sum()).collect())
        })
    }
}

impl Default for FrobBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagonal of the square `n×n` view at `oa` with leading dimension `lda`.
pub fn diag(a: &DMat, n: usize, oa: usize, lda: usize) -> Vec<f64> {
    with(|cx, s| unsafe {
        let out = Tmp::new(cx, n)?;
        let (ni, ldai, oai) = (n as u32, lda as u32, oa as u32);
        cx.launch(s.f.diag, egrid(n), 256, 0, kargs!(a.ptr, out.ptr, ni, ldai, oai))?;
        Ok(download_f32(cx, out.ptr, n)?.into_iter().map(|x| x as f64).collect())
    })
}

/// `F = (S + ridge·I) / (d dᵀ)` (`d` = `dv`), optionally in reversed act
/// order (`perm`: F[i][j] uses S[perm[n-1-i]][perm[n-1-j]]).
pub fn equil(s: &DMat, f: &DMat, dv: &DMat, ridge: f32, perm: Option<&U32Buf>) {
    let n = s.r;
    with(|cx, st| unsafe {
        let (ni, rev) = (n as u32, perm.is_some() as u32);
        let pp = perm.map_or(dv.ptr, |p| p.ptr);
        cx.launch(st.f.equil, mgrid(n, n), 256, 0, kargs!(s.ptr, dv.ptr, pp, f.ptr, ni, rev, ridge))
    });
}

/// `Uc[j][k] = X[n-1-j][n-1-k] / dp[k]` (k ≥ j).
pub fn make_uc(x: &DMat, dp_: &DMat, uc: &DMat) {
    let n = x.r;
    with(|cx, s| unsafe {
        let ni = n as u32;
        cx.launch(s.f.make_uc, mgrid(n, n), 256, 0, kargs!(x.ptr, dp_.ptr, uc.ptr, ni))
    });
}

/// Row gather `C[j] = A[perm[j]]` or scatter `C[perm[j]] = A[j]`.
pub fn permute_rows(a: &DMat, c: &DMat, perm: &U32Buf, scatter: bool) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    with(|cx, s| unsafe {
        let (ri, ci, sc) = (a.r as u32, a.c as u32, scatter as u32);
        cx.launch(s.f.permute_rows, mgrid(a.r, a.c), 256, 0, kargs!(a.ptr, perm.ptr, c.ptr, ri, ci, sc))
    });
}

/// `Aᵀ` into a new matrix.
pub fn transpose(a: &DMat) -> DMat {
    with(|cx, s| unsafe {
        let c = mat_raw(cx, a.c, a.r)?;
        let (ri, ci) = (a.r as u32, a.c as u32);
        cx.launch(s.f.transpose, mgrid(a.c, a.r), 256, 0, kargs!(a.ptr, c.ptr, ri, ci))?;
        Ok(c)
    })
}

/// Per-row h solve on the device (see the kernel); `h1` unused for P = 1.
#[allow(clippy::too_many_arguments)]
pub fn hsolve(m: usize, np: usize, rh0: &DMat, rh1: &DMat, g00: &DMat, g01: &DMat, g11: &DMat, h0: &DMat, h1: &DMat) {
    with(|cx, s| unsafe {
        let (mi, npi) = (m as u32, np as u32);
        cx.launch(s.f.hsolve, egrid(m), 256, 0, kargs!(rh0.ptr, rh1.ptr, g00.ptr, g01.ptr, g11.ptr, h0.ptr, h1.ptr, mi, npi))
    });
}

/// `dst[perm[j]] = src[so + j]` for `n` entries.
pub fn scatter_vec(src: &DMat, so: usize, dst: &DMat, perm: &U32Buf, n: usize) {
    with(|cx, s| unsafe {
        let (ni, soi) = (n as u32, so as u32);
        cx.launch(s.f.scatter_vec, egrid(n), 256, 0, kargs!(src.ptr, perm.ptr, dst.ptr, ni, soi))
    });
}

/// `out = a / ‖a‖_F` on the device (no readback).
pub fn normalize_frob(a: &DMat, out: &DMat) {
    let len = a.len();
    with(|cx, s| unsafe {
        let part = Tmp::new(cx, frob_nwg(len))?;
        let rg = Tmp::zeroed(cx, 4)?;
        let nwg = frob_part_raw(cx, &s.f, a.ptr, a.ptr, len, part.ptr)?;
        let (nwgi, slot, leni) = (nwg as u32, 0u32, len as u64);
        cx.launch(s.f.sum_into, (1, 1), 256, 0, kargs!(part.ptr, rg.ptr, nwgi, slot))?;
        cx.launch(s.f.scale_rsqrt, egrid(len), 256, 0, kargs!(a.ptr, rg.ptr, out.ptr, leni, slot))
    });
}

// ───────────────────────────── SPD solves ─────────────────────────────

fn blas_ok(s: bl::cublasStatus_t, what: &str) -> Res<()> {
    if s == bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(format!("cuBLAS {what}: {s:?}")) }
}

/// Lower Cholesky of the row-major `n×n` matrix at `a`, in place: read
/// column-major it is the upper factor `A = UᵀU`, right-looking in blocks of
/// [`chol_nb`] — a diagonal block on the host in f64 (a pivot that is not
/// finite and positive fails), its row panel by STRSM, the trailing matrix
/// by SSYRK. The strict upper triangle (row-major) is left as it was.
unsafe fn chol_dev(c: &E, a: u64, n: usize) -> Res<bool> {
    use bl::{cublasDiagType_t as Dg, cublasFillMode_t as Fm, cublasOperation_t as Op, cublasSideMode_t as Sd};
    let at = |r: usize, col: usize| a + ((r + col * n) * 4) as u64;
    let nb = chol_nb(n);
    let mut blk = vec![0f32; nb * nb];
    let mut d = vec![0f64; nb * nb];
    let mut k = 0;
    while k < n {
        let kb = nb.min(n - k);
        // the block's trailing updates are done (the stream is in order)
        c.sync()?;
        unsafe {
            blas_ok(bl::cublasGetMatrix(kb as i32, kb as i32, 4, at(k, k) as *const c_void, n as i32, blk.as_mut_ptr().cast(), kb as i32), "get block")?;
        }
        for (o, &v) in d.iter_mut().zip(&blk[..kb * kb]) {
            *o = v as f64;
        }
        // U11ᵀU11 = A11, column-major, upper: u(i, j) = d[i + j·kb], i ≤ j
        for j in 0..kb {
            let mut p = d[j + j * kb];
            for i in 0..j {
                p -= d[i + j * kb] * d[i + j * kb];
            }
            if !(p > 0.0) || !p.is_finite() {
                return Ok(false);
            }
            let djj = p.sqrt();
            d[j + j * kb] = djj;
            for col in j + 1..kb {
                let mut v = d[j + col * kb];
                for i in 0..j {
                    v -= d[i + j * kb] * d[i + col * kb];
                }
                d[j + col * kb] = v / djj;
            }
        }
        for (o, &v) in blk[..kb * kb].iter_mut().zip(&d[..kb * kb]) {
            *o = v as f32;
        }
        unsafe {
            blas_ok(bl::cublasSetMatrix(kb as i32, kb as i32, 4, blk.as_ptr().cast(), kb as i32, at(k, k) as *mut c_void, n as i32), "set block")?;
            // a copy from pageable memory returns once it is staged: its DMA
            // runs on the legacy stream, which this backend's non-blocking
            // stream does not wait for — the panel solve queued next read the
            // block as it was before (a rare "not positive definite" at a
            // well-conditioned Gram)
            cu::stream::synchronize(std::ptr::null_mut()).map_err(derr)?;
        }
        let rest = n - k - kb;
        if rest > 0 {
            let (one, m_one) = (1.0f32, -1.0f32);
            unsafe {
                // A12 ← U11⁻ᵀ A12
                blas_ok(
                    bl::cublasStrsm_v2(c.blas, Sd::CUBLAS_SIDE_LEFT, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_T, Dg::CUBLAS_DIAG_NON_UNIT,
                        kb as i32, rest as i32, &one, at(k, k) as *const f32, n as i32, at(k, k + kb) as *mut f32, n as i32),
                    "strsm panel",
                )?;
                // A22 ← A22 − A12ᵀ A12
                blas_ok(
                    bl::cublasSsyrk_v2(c.blas, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_T, rest as i32, kb as i32, &m_one,
                        at(k, k + kb) as *const f32, n as i32, &one, at(k + kb, k + kb) as *mut f32, n as i32),
                    "ssyrk trailing",
                )?;
            }
        }
        k += kb;
    }
    Ok(true)
}

/// Lower Cholesky of the square `a` in place (`Some(false)`: a pivot failed).
/// The strict upper triangle is left undefined.
pub fn chol_in_place(a: &DMat) -> Option<bool> {
    Some(with(|c, _| unsafe { chol_dev(c, a.ptr, a.r) }))
}

/// `x = L⁻¹` (`l` as [`chol_in_place`] leaves it): zero above the diagonal.
pub fn tri_inv(l: &DMat, x: &DMat) -> Option<()> {
    use bl::{cublasDiagType_t as Dg, cublasFillMode_t as Fm, cublasOperation_t as Op, cublasSideMode_t as Sd};
    let n = l.r;
    with(|c, s| unsafe {
        let ni = n as u32;
        c.launch(s.f.eye, mgrid(n, n), 256, 0, kargs!(x.ptr, ni))?;
        // column-major: X ← U⁻¹ (upper) — read row-major, L⁻¹ (lower)
        let one = 1.0f32;
        blas_ok(
            bl::cublasStrsm_v2(c.blas, Sd::CUBLAS_SIDE_LEFT, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_N, Dg::CUBLAS_DIAG_NON_UNIT,
                n as i32, n as i32, &one, l.ptr as *const f32, n as i32, x.ptr as *mut f32, n as i32),
            "strsm inverse",
        )
    });
    Some(())
}

/// Solve `L Lᵀ x = b` in place over device addresses (`l` the lower factor).
unsafe fn chol_solve_raw(c: &E, l: u64, n: usize, y: u64) -> Res<()> {
    use bl::{cublasDiagType_t as Dg, cublasFillMode_t as Fm, cublasOperation_t as Op};
    unsafe {
        // UᵀU x = b with U = Lᵀ (column-major upper)
        blas_ok(bl::cublasStrsv_v2(c.blas, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_T, Dg::CUBLAS_DIAG_NON_UNIT, n as i32, l as *const f32, n as i32, y as *mut f32, 1), "strsv forward")?;
        blas_ok(bl::cublasStrsv_v2(c.blas, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_N, Dg::CUBLAS_DIAG_NON_UNIT, n as i32, l as *const f32, n as i32, y as *mut f32, 1), "strsv backward")
    }
}

/// Solve `L Lᵀ x = b` in place (`y` holds b; `l` the dense lower factor).
pub fn chol_solve_dev(l: &DMat, y: &DMat) {
    with(|c, _| unsafe { chol_solve_raw(c, l.ptr, l.r, y.ptr) });
}

/// `(A + rel·tr(A)/n·I) x = b` for the SPD `a` (left untouched), the
/// right-hand side and the solution on the device: Jacobi equilibration and
/// the ridge there, the ridge ×100 on a failed pivot (the host `solve_spd`
/// rule), one readback per failed attempt. Zeros when the ridge runs past 10·tr.
pub fn spd_solve_d(a: &DMat, b: &DMat, rel: f64) -> DMat {
    let n = a.r;
    with(|c, s| unsafe {
        let dvec = Tmp::new(c, n)?;
        let rg = Tmp::zeroed(c, 4)?;
        let f = Tmp::new(c, n * n)?;
        let ni = n as u32;
        for attempt in 0..12u32 {
            let (retry, relf) = ((attempt > 0) as u32, rel as f32);
            c.launch(s.f.diag_ridge, (1, 1), 256, 0, kargs!(a.ptr, dvec.ptr, rg.ptr, ni, retry, relf))?;
            c.launch(s.f.equil_rg, mgrid(n, n), 256, 0, kargs!(a.ptr, dvec.ptr, rg.ptr, f.ptr, ni))?;
            if chol_dev(c, f.ptr, n)? {
                let y = Tmp::new(c, n)?;
                let zero = 0u32;
                c.launch(s.f.div_vec, egrid(n), 256, 0, kargs!(b.ptr, dvec.ptr, y.ptr, ni, zero, zero))?;
                chol_solve_raw(c, f.ptr, n, y.ptr)?;
                let x = mat_raw(c, 1, n)?;
                c.launch(s.f.div_vec, egrid(n), 256, 0, kargs!(y.ptr, dvec.ptr, x.ptr, ni, zero, zero))?;
                return Ok(x);
            }
            let r = download_f32(c, rg.ptr, 2)?;
            if r[0] as f64 > (r[1].abs() as f64) * 10.0 + 1e-30 {
                break;
            }
        }
        mat_zero(c, 1, n)
    })
}

// ───────────────────────────── discrete knots ─────────────────────────────

/// Output-side flips by blocks of 32 ranks (see `nd_flipblk`): `su`, `qs`,
/// `whb` row-major m×r (`su` and `qs` updated), `q` = diag(ℓ)·K·diag(ℓ), `h`,
/// `l`. At most four sweeps, stopping after one without a flip. Returns the
/// flip count.
#[allow(clippy::too_many_arguments)]
pub fn flipu_blocked(su: &DMat, qs: &DMat, whb: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> Option<u64> {
    Some(with(|c, s| unsafe {
        let f = Tmp::new(c, m * 32)?;
        let cnt = Tmp::new(c, 4)?;
        let mut total = 0u64;
        let (mi, ri) = (m as u32, r as u32);
        for _ in 0..4 {
            cu::memset_d8_async(cnt.ptr, 0, 16, c.stream).map_err(derr)?;
            let mut b0 = 0;
            while b0 < r {
                let bs = 32.min(r - b0);
                let b0i = b0 as u32;
                c.launch(s.f.flipblk, (m.div_ceil(64) as u32, 1), 64, 0, kargs!(whb.ptr, q.ptr, h.ptr, l.ptr, su.ptr, qs.ptr, f.ptr, cnt.ptr, mi, ri, b0i, tol))?;
                // qs −= F · Q[b0..b0+bs, :]
                gemm_raw(c, m, r, bs, -1.0, f.ptr, 32, false, q.ptr + fl(b0 * r), r, false, 1.0, qs.ptr, r)?;
                b0 += bs;
            }
            let n = download_u32(c, cnt.ptr, 1)?[0] as u64;
            total += n;
            if n == 0 {
                break;
            }
        }
        Ok(total)
    }))
}

/// Output-side flips, one thread per row over transposed operands (see
/// `nd_flipu`). Returns the flip count.
#[allow(clippy::too_many_arguments)]
pub fn flipu(sut: &DMat, qst: &DMat, whbt: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> u64 {
    with(|c, s| unsafe {
        let cnt = Tmp::zeroed(c, 4)?;
        let (mi, ri) = (m as u32, r as u32);
        c.launch(s.f.flipu, egrid(m), 256, 0, kargs!(whbt.ptr, q.ptr, h.ptr, l.ptr, sut.ptr, qst.ptr, cnt.ptr, mi, ri, tol))?;
        Ok(download_u32(c, cnt.ptr, 1)?[0] as u64)
    })
}

/// Relative convergence of the column scale between flip rounds
/// (`CMF_NVG_GJ_TOL`, default 1e-8; the f64 host uses 1e-9).
fn gj_tol() -> f32 {
    std::env::var("CMF_NVG_GJ_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8)
}

/// Threads of the column search's block (`CMF_NVG_GV_WS`: 64 … 1024, default
/// 256). A flip is one barrier, after which every thread merges the warps'
/// partial winners itself: eight warps keep that merge short.
fn gv_ws() -> usize {
    std::env::var("CMF_NVG_GV_WS").ok().and_then(|v| v.parse().ok()).filter(|w| [64usize, 128, 256, 512, 1024].contains(w)).unwrap_or(256)
}

/// One 128-column block of the input-side search (see `nd_gv_block`), then
/// the trailing feedback `BV[b1:] −= Uc[b0:b1, b1:]ᵀ · E` as a GEMM.
/// `cnt` accumulates the flips (a u32 in its first word).
#[allow(clippy::too_many_arguments)]
pub fn gv_block(bv: &DMat, s: &DMat, ks0: &DMat, k: &DMat, uc: &DMat, eg: &DMat, cnt: &DMat, b0: usize, b1: usize, max_flips: usize, tol: f32) {
    let (n, r) = (bv.r, bv.c);
    let ws = gv_ws();
    let nq = r.div_ceil(ws).max(1);
    with(|c, st| unsafe {
        let f = {
            let mut gv = st.gv.lock().unwrap();
            match gv.get(&nq) {
                Some(f) => *f,
                None => {
                    let m = compile(ND_SRC, vec![format!("-DNQ={nq}")], c.arch)?;
                    let f = func(m, "nd_gv_block")?;
                    gv.insert(nq, f);
                    f
                }
            }
        };
        let (ni, ri, b0i, b1i, mf, gjt) = (n as u32, r as u32, b0 as u32, b1 as u32, max_flips as u32, gj_tol());
        c.launch(f, (1, 1), ws as u32, 0, kargs!(ks0.ptr, k.ptr, uc.ptr, bv.ptr, s.ptr, eg.ptr, cnt.ptr, ni, ri, b0i, b1i, mf, tol, gjt))?;
        if b1 < n {
            gemm_raw(c, n - b1, r, b1 - b0, -1.0, uc.ptr + fl(b0 * n + b1), n, true, eg.ptr, r, false, 1.0, bv.ptr + fl(b1 * r), r)?;
        }
        Ok(())
    });
}

/// The flips a [`gv_block`] counter holds.
pub fn count_read(cnt: &DMat) -> u64 {
    with(|c, _| Ok(download_u32(c, cnt.ptr, 1)?[0] as u64))
}

// ───────────────────────────── symmetric eigensolver ─────────────────────────────

/// Householder tridiagonalisation of the symmetric `a` (k×k) in place: the
/// reflectors stay below the subdiagonal of `a`; returns `[d | e | tau]`
/// (3k values, f64) — `e[i]` couples i and i+1, `tau[i]` is reflector i's.
pub fn tridiag(a: &DMat) -> Vec<f64> {
    let k = a.r;
    if k < 2 {
        return a.to_f64().into_iter().chain([0.0, 0.0]).collect();
    }
    with(|c, s| unsafe {
        let de = Tmp::zeroed(c, 3 * k)?;
        let (vs, pv, wv) = (Tmp::new(c, k)?, Tmp::new(c, k)?, Tmp::new(c, k)?);
        let ki = k as u32;
        for i in 0..k - 1 {
            let l = k - i - 1;
            let ii = i as u32;
            c.launch(s.f.hh_vec, (1, 1), 1024, 0, kargs!(a.ptr, vs.ptr, de.ptr, ki, ii))?;
            c.launch(s.f.hh_matvec, (l as u32, 1), 256, 0, kargs!(a.ptr, vs.ptr, pv.ptr, de.ptr, ki, ii))?;
            c.launch(s.f.hh_w, (1, 1), 1024, 0, kargs!(pv.ptr, vs.ptr, wv.ptr, de.ptr, ki, ii))?;
            c.launch(s.f.hh_rank2, mgrid(l, l), 256, 0, kargs!(a.ptr, vs.ptr, wv.ptr, ki, ii))?;
        }
        c.launch(s.f.hh_last, (1, 1), 32, 0, kargs!(a.ptr, de.ptr, ki))?;
        Ok(download_f32(c, de.ptr, 3 * k)?.into_iter().map(|x| x as f64).collect())
    })
}

/// The `r` largest eigenpairs of the symmetric `a` (k×k, destroyed):
/// eigenvalues descending (f64) and the eigenvectors as a k×r device matrix.
pub fn sym_eig_top(a: &DMat, r: usize, pool: Option<&crate::pool::Pool>) -> (Vec<f64>, DMat) {
    let k = a.r;
    let r = r.min(k);
    let trace = std::env::var("CMF_NVG_EIG_TRACE").as_deref() == Ok("1");
    let t0 = std::time::Instant::now();
    let de = tridiag(a);
    let t1 = t0.elapsed().as_secs_f64();
    let (d, e) = (&de[..k], &de[k..2 * k - 1]);
    let (lam, z) = crate::gpu_wgpu::nvg_dev::tridiag_top(d, e, r, pool);
    let t2 = t0.elapsed().as_secs_f64();
    let zd = DMat::from_f64(k, r, &z);
    let tau: Vec<f32> = de[2 * k..].iter().map(|&x| x as f32).collect();
    // x = H_0 H_1 ⋯ H_{k-2} z, blocks of reflectors applied last-first:
    // per block V, VᵀV, T, then z −= V (T (Vᵀ z))
    with(|c, s| unsafe {
        let bsz = 32usize;
        let nref = k - 1;
        let taud = Tmp::new(c, k)?;
        upload(c, taud.ptr, &tau)?;
        let (vb, vtv, td, w, w2) = (Tmp::new(c, k * bsz)?, Tmp::new(c, bsz * bsz)?, Tmp::new(c, bsz * bsz)?, Tmp::new(c, bsz * r)?, Tmp::new(c, bsz * r)?);
        let (ki, bi) = (k as u32, bsz as u32);
        let mut blocks: Vec<usize> = (0..nref).step_by(bsz).collect();
        blocks.reverse();
        for i0 in blocks {
            let b = bsz.min(nref - i0);
            let (i0i, bl_) = (i0 as u32, b as u32);
            c.launch(s.f.vblock, mgrid(k, bsz), 256, 0, kargs!(a.ptr, vb.ptr, ki, bi, i0i))?;
            gemm_raw(c, bsz, bsz, k, 1.0, vb.ptr, bsz, true, vb.ptr, bsz, false, 0.0, vtv.ptr, bsz)?;
            c.launch(s.f.wy_t, (1, 1), 64, 0, kargs!(vtv.ptr, taud.ptr, td.ptr, bi, bl_, i0i))?;
            gemm_raw(c, bsz, r, k, 1.0, vb.ptr, bsz, true, zd.ptr, r, false, 0.0, w.ptr, r)?;
            gemm_raw(c, bsz, r, bsz, 1.0, td.ptr, bsz, false, w.ptr, r, false, 0.0, w2.ptr, r)?;
            gemm_raw(c, k, r, bsz, -1.0, vb.ptr, bsz, false, w2.ptr, r, false, 1.0, zd.ptr, r)?;
        }
        if trace {
            c.sync()?;
            eprintln!(
                "sym_eig_top (cuda) k={k} r={r}: tridiag {t1:.3}s, host bisection+inverse iteration {:.3}s, back-transform {:.3}s",
                t2 - t1,
                t0.elapsed().as_secs_f64() - t2
            );
        }
        Ok(())
    });
    (lam, zd)
}

#[cfg(test)]
mod tests {
    //! The CUDA twin against f64 host references (the tests of
    //! `gpu_wgpu::nvg_dev`); they skip without a card.
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn host_mm(a: &[f64], ar: usize, ac: usize, ta: bool, b: &[f64], br: usize, bc: usize, tb: bool) -> (Vec<f64>, usize, usize) {
        let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if tb { (bc, br) } else { (br, bc) };
        assert_eq!(k, k2);
        let mut c = vec![0f64; m * n];
        for i in 0..m {
            for t in 0..k {
                let av = if ta { a[t * ac + i] } else { a[i * ac + t] };
                for j in 0..n {
                    let bv = if tb { b[j * bc + t] } else { b[t * bc + j] };
                    c[i * n + j] += av * bv;
                }
            }
        }
        (c, m, n)
    }

    fn relerr(got: &[f64], want: &[f64]) -> f64 {
        let num: f64 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
        let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-300);
        (num / den).sqrt()
    }

    /// Every transpose variant at ragged shapes, sub-matrix views, α/β and
    /// the Hadamard epilogue, against an f64 reference.
    #[test]
    fn cuda_gemm_variants_match_host() {
        if !available() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        for &(m, n, k) in &[(1usize, 1usize, 1usize), (37, 129, 70), (200, 64, 300), (130, 257, 17)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
                let (ar, ac) = if ta { (k, m) } else { (m, k) };
                let (br, bc) = if tb { (n, k) } else { (k, n) };
                let a = rnd(ar * ac, 1 + m as u64);
                let b = rnd(br * bc, 2 + n as u64);
                let c0 = rnd(m * n, 3);
                let e = rnd(m * n, 4);
                let (ab, bb) = (DMat::from_f64(ar, ac, &a), DMat::from_f64(br, bc, &b));
                let cb = DMat::from_f64(m, n, &c0);
                let eb = DMat::from_f64(m, n, &e);
                gemm_v(m, n, k, 0.5, V { m: &ab, off: 0, ld: ac, t: ta }, V { m: &bb, off: 0, ld: bc, t: tb }, -1.5, &cb, 0, n, Some(&eb));
                let (h, _, _) = host_mm(&a, ar, ac, ta, &b, br, bc, tb);
                let want: Vec<f64> = (0..m * n).map(|i| 0.5 * h[i] * e[i] - 1.5 * c0[i]).collect();
                let err = relerr(&cb.to_f64(), &want);
                assert!(err < 1e-5, "gemm {m}x{n}x{k} ta {ta} tb {tb}: {err:e}");
                // without the epilogue, into an offset view of a wider output
                let wide = DMat::from_f64(m + 2, n + 3, &rnd((m + 2) * (n + 3), 5));
                let before = wide.to_f64();
                gemm_v(m, n, k, 1.0, V { m: &ab, off: 0, ld: ac, t: ta }, V { m: &bb, off: 0, ld: bc, t: tb }, 0.0, &wide, (n + 3) + 1, n + 3, None);
                let got = wide.to_f64();
                let mut worst = 0f64;
                for i in 0..m + 2 {
                    for j in 0..n + 3 {
                        let g = got[i * (n + 3) + j];
                        let want = if (1..=m).contains(&i) && (1..=n).contains(&j) { h[(i - 1) * n + j - 1] } else { before[i * (n + 3) + j] };
                        worst = worst.max((g - want).abs());
                    }
                }
                assert!(worst < 1e-3, "gemm view {m}x{n}x{k} ta {ta} tb {tb}: {worst:e}");
            }
        }
    }

    /// The elementwise and reduction kernels against their definitions.
    #[test]
    fn cuda_elementwise_match_host() {
        if !available() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let (rows, cols) = (70usize, 300usize);
        let a = rnd(rows * cols, 11);
        let b = rnd(rows * cols, 12);
        let r = rnd(rows, 13);
        let s = rnd(cols, 14);
        let (ad, bd) = (DMat::from_f64(rows, cols, &a), DMat::from_f64(rows, cols, &b));
        let (rd, sd) = (DMat::from_f64(1, rows, &r), DMat::from_f64(1, cols, &s));
        // scale_rc: sign, row and column scale, accumulate
        let c = DMat::from_f64(rows, cols, &b);
        scale_rc(rows, cols, Some((&ad, 0, cols)), (&c, 0, cols), Some(&rd), Some(&sd), SIGN | ACC, 2.0, 0.5);
        let want: Vec<f64> = (0..rows * cols).map(|i| 2.0 * (if a[i] >= 0.0 { 1.0 } else { -1.0 }) * r[i / cols] * s[i % cols] + 0.5 * b[i]).collect();
        assert!(relerr(&c.to_f64(), &want) < 1e-6, "scale_rc");
        // rowdot, colsum, frob, diag, transpose
        let out = DMat::zeros(1, rows);
        rowdot(rows, cols, &ad, 0, cols, &bd, 0, cols, &out);
        let want: Vec<f64> = (0..rows).map(|i| (0..cols).map(|j| a[i * cols + j] * b[i * cols + j]).sum()).collect();
        assert!(relerr(&out.to_f64(), &want) < 1e-5, "rowdot");
        let got = colsum_prod(rows, cols, &ad, 0, cols, &bd, 0, cols);
        let want: Vec<f64> = (0..cols).map(|j| (0..rows).map(|i| a[i * cols + j] * b[i * cols + j]).sum()).collect();
        assert!(relerr(&got, &want) < 1e-5, "colsum");
        let f = frob(&ad, &bd, rows * cols);
        let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((f - want).abs() < 1e-3 * want.abs().max(1.0), "frob {f} vs {want}");
        let t = transpose(&ad).to_f64();
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(t[j * rows + i] as f32, a[i * cols + j] as f32);
            }
        }
        // row gather and scatter by a permutation
        let perm: Vec<u32> = (0..rows as u32).rev().collect();
        let pb = upload_u32(&perm);
        let g = DMat::zeros(rows, cols);
        permute_rows(&ad, &g, &pb, false);
        let back = DMat::zeros(rows, cols);
        permute_rows(&g, &back, &pb, true);
        assert_eq!(back.to_f32(), ad.to_f32());
        assert_eq!(g.to_f32()[..cols], ad.to_f32()[(rows - 1) * cols..]);
    }

    /// Top-r eigenpairs (device tridiagonalisation + host bisection /
    /// inverse iteration + device back-transform) of a matrix with a known
    /// graded spectrum.
    #[test]
    fn cuda_sym_eig_top_matches_known_spectrum() {
        if !available() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let (k, r) = (300usize, 40usize);
        let mu: Vec<f64> = (0..k).map(|i| 1000.0 * 0.96f64.powi(i as i32) + 1e-3 * (i % 3) as f64).collect();
        let mut q = vec![0f64; k * k];
        for i in 0..k {
            q[i * k + i] = 1.0;
        }
        for s in 0..3u64 {
            let w = rnd(k, 50 + s);
            let ww: f64 = w.iter().map(|x| x * x).sum();
            let mut q2 = q.clone();
            for i in 0..k {
                let dot: f64 = (0..k).map(|j| q[i * k + j] * w[j]).sum();
                for j in 0..k {
                    q2[i * k + j] -= 2.0 * dot * w[j] / ww;
                }
            }
            q = q2;
        }
        let mut a = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|t| q[i * k + t] * mu[t] * q[j * k + t]).sum();
            }
        }
        let ad = DMat::from_f64(k, k, &a);
        let (lam, z) = sym_eig_top(&ad, r, None);
        let z = z.to_f64();
        for j in 0..r {
            assert!((lam[j] - mu[j]).abs() <= 2e-5 * mu[0], "lambda {j}: {} vs {}", lam[j], mu[j]);
            let mut res = 0f64;
            for i in 0..k {
                let az: f64 = (0..k).map(|t| a[i * k + t] * z[t * r + j]).sum();
                res += (az - lam[j] * z[i * r + j]).powi(2);
            }
            assert!(res.sqrt() <= 2e-4 * mu[0], "residual {j}: {}", res.sqrt());
            for jj in 0..=j {
                let d: f64 = (0..k).map(|i| z[i * r + j] * z[i * r + jj]).sum();
                assert!((d - if j == jj { 1.0 } else { 0.0 }).abs() < 1e-3, "orthogonality {j},{jj}: {d}");
            }
        }
    }

    /// Dense SPD solve (equilibrated blocked Cholesky past one diagonal block
    /// + triangular solves) and the act-order upper factor of the inverse.
    #[test]
    fn cuda_spd_solve_and_uc_match_host() {
        if !available() {
            eprintln!("no CUDA device — skipped");
            return;
        }
        let n = 600usize;
        let x = rnd(n * (n + 20), 7);
        let (mut a, _, _) = host_mm(&x, n, n + 20, false, &x, n, n + 20, true);
        for i in 0..n {
            a[i * n + i] += 0.5 + 40.0 * (i % 7 == 0) as u8 as f64;
        }
        let b = rnd(n, 9);
        let ad = DMat::from_f64(n, n, &a);
        let xs = spd_solve_d(&ad, &DMat::from_f64(1, n, &b), 1e-10).to_f64();
        let ax: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * xs[j]).sum()).collect();
        let err = relerr(&ax, &b);
        assert!(err < 1e-3, "spd residual {err:e}");
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&p, &q| a[q * n + q].partial_cmp(&a[p * n + p]).unwrap());
        let pb = upload_u32(&perm.iter().map(|&v| v as u32).collect::<Vec<_>>());
        let d: Vec<f64> = (0..n).map(|i| a[i * n + i].sqrt()).collect();
        let dv = DMat::from_f64(1, n, &d);
        let f = DMat::zeros(n, n);
        equil(&ad, &f, &dv, 0.0, Some(&pb));
        assert_eq!(chol_in_place(&f), Some(true));
        let xi = DMat::zeros(n, n);
        tri_inv(&f, &xi).unwrap();
        let dpv: Vec<f64> = perm.iter().map(|&p| d[p]).collect();
        let uc = DMat::zeros(n, n);
        make_uc(&xi, &DMat::from_f64(1, n, &dpv), &uc);
        let u = uc.to_f64();
        let (utu, _, _) = host_mm(&u, n, n, true, &u, n, n, false);
        let mut worst = 0f64;
        for i in (0..n).step_by(7) {
            for j in 0..n {
                let v: f64 = (0..n).map(|t| utu[i * n + t] * a[perm[t] * n + perm[j]]).sum();
                worst = worst.max((v - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst < 5e-3, "Ucᵀ Uc Hp − I: {worst:e}");
        // a matrix that is not positive definite is reported, not factored
        let mut bad = a.clone();
        bad[5 * n + 5] = -1.0;
        assert_eq!(chol_in_place(&DMat::from_f64(n, n, &bad)), Some(false));
    }

    /// Throughput of the GEMM and of the dense Cholesky at encoder sizes
    /// (`--ignored --nocapture`).
    #[test]
    #[ignore]
    fn cuda_gemm_and_chol_throughput() {
        if !available() {
            return;
        }
        for &(m, n, k, ta, tb) in &[(4096usize, 4096usize, 4096usize, false, false), (17408, 1064, 5120, false, false), (5120, 5120, 17408, false, true)] {
            let (ar, ac) = if ta { (k, m) } else { (m, k) };
            let (br, bc) = if tb { (n, k) } else { (k, n) };
            let a = DMat::from_f64(ar, ac, &rnd(ar * ac, 1));
            let b = DMat::from_f64(br, bc, &rnd(br * bc, 2));
            let c = DMat::zeros(m, n);
            gemm(1.0, &a, ta, &b, tb, 0.0, &c);
            sync();
            let t = std::time::Instant::now();
            let reps = 5;
            for _ in 0..reps {
                gemm(1.0, &a, ta, &b, tb, 0.0, &c);
            }
            sync();
            let s = t.elapsed().as_secs_f64() / reps as f64;
            eprintln!("cuda gemm {m}x{n}x{k} ta {ta} tb {tb}: {:.2} ms, {:.1} TFLOP/s", s * 1e3, 2.0 * (m * n * k) as f64 / s / 1e12);
        }
        for &n in &[2152usize, 5120, 17408] {
            let x = DMat::from_f64(n, 64, &rnd(n * 64, 3));
            let a = DMat::zeros(n, n);
            gemm(1.0, &x, false, &x, true, 0.0, &a);
            let id: Vec<f64> = (0..n).map(|_| 1.0).collect();
            let dv = DMat::from_f64(1, n, &id);
            let f = DMat::zeros(n, n);
            equil(&a, &f, &dv, 1.0, None);
            sync();
            let t = std::time::Instant::now();
            let ok = chol_in_place(&f);
            sync();
            let tc = t.elapsed().as_secs_f64();
            let xi = DMat::zeros(n, n);
            let t = std::time::Instant::now();
            tri_inv(&f, &xi).unwrap();
            sync();
            eprintln!("cuda chol n={n}: {ok:?} {:.1} ms ({:.1} TFLOP/s), tri_inv {:.1} ms", tc * 1e3, (n as f64).powi(3) * 2.0 / 3.0 / tc / 1e12, t.elapsed().as_secs_f64() * 1e3);
        }
        // the metric's solve by parts, best of five
        for &n in &[1024usize, 3584, 5120, 17408] {
            let x = DMat::from_f64(n, 64, &rnd(n * 64, 3));
            let a = DMat::zeros(n, n);
            gemm(1.0, &x, false, &x, true, 0.0, &a);
            let id: Vec<f64> = (0..n).map(|_| 1.0).collect();
            let dv = DMat::from_f64(1, n, &id);
            let b = DMat::from_f64(1, n, &rnd(n, 5));
            let (mut tc, mut ts, mut ta) = (f64::MAX, f64::MAX, f64::MAX);
            for _ in 0..5 {
                let f = DMat::zeros(n, n);
                equil(&a, &f, &dv, 1.0, None);
                sync();
                let t = std::time::Instant::now();
                chol_in_place(&f);
                sync();
                tc = tc.min(t.elapsed().as_secs_f64());
                let y = b.dup();
                sync();
                let t = std::time::Instant::now();
                chol_solve_dev(&f, &y);
                sync();
                ts = ts.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                let _x = spd_solve_d(&a, &b, 1.0);
                sync();
                ta = ta.min(t.elapsed().as_secs_f64());
            }
            eprintln!("cuda spd n={n}: chol {:.2} ms, two triangular solves {:.2} ms, spd_solve_d {:.2} ms", tc * 1e3, ts * 1e3, ta * 1e3);
        }
    }
}
