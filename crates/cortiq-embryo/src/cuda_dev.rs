// The CUDA device layer of the trainer (`--features cuda`), included by
// `vulkan.rs` in place of its wgpu layer: the same `Cmd` recording, the
// same kernels (`EMBRYO_WGSL` through `wgsl_cuda::translate`, compiled by
// NVRTC once per process and kept on disk as PTX), one launch per recorded
// operation on one stream. A buffer is a device allocation with the same
// demand-driven host mirror as the wgpu layer's.

use cudarc::cublas::sys as bl;
use cudarc::driver::{result as cu, sys as cs};

/// `OpRec::entry` of a GEMM that runs on cuBLAS (`Cmd::cublas_gemm`).
const ENTRY_CUBLAS: u32 = 100;

/// A device allocation's address (the split-K pool's element).
type RawBuf = u64;

struct BufInner {
    ptr: u64,
    len: usize,
    /// A view (the dummy, a split-K scratch) frees nothing.
    owned: bool,
    host: UnsafeCell<Vec<u32>>,
    host_dirty: std::sync::atomic::AtomicBool,
    gpu_dirty: std::sync::atomic::AtomicBool,
    ctx: Arc<CtxInner>,
}

impl Drop for BufInner {
    fn drop(&mut self) {
        if self.owned {
            self.ctx.current();
            unsafe {
                let _ = cu::free_sync(self.ptr);
            }
        }
    }
}

/// The entry points by `OpRec::entry`: kernel name and block size.
const ENTRIES: [(&str, u32); 11] = [
    ("emb_main", 256),
    ("emb_gemm64", 256),
    ("emb_gdn_fwd", 128),
    ("emb_gdn_bwd", 128),
    ("emb_gemm128", 256),
    ("emb_rms_bwd_dx", 256),
    ("emb_rms_dw", 256),
    ("emb_wy_norm_rows", 128),
    ("emb_wy_dnorm_rows", 128),
    ("emb_band_softmax_rows", 256),
    ("emb_softmax_bwd_rows", 256),
];

/// The parameter blocks of one commit, uploaded together.
struct ParArena {
    ptr: u64,
    words: usize,
}

struct CtxInner {
    cuctx: cs::CUcontext,
    stream: cs::CUstream,
    funcs: [cs::CUfunction; 11],
    big_gpu: bool,
    gemm128: bool,
    gemm64: bool,
    tile_gemm: bool,
    profile_ops: bool,
    op_ts: bool,
    /// cuBLAS for the plain GEMMs (`CMF_CUDA_CUBLAS=1`; else the translated kernels).
    blas: Option<bl::cublasHandle_t>,
    /// 64 KB of zeros: the slots an operation leaves unbound.
    dummy: u64,
    adapter_name: String,
    splitk_pool: Mutex<Vec<(RawBuf, usize)>>,
    /// One commit at a time owns the parameter arena.
    commit: Mutex<ParArena>,
}

// SAFETY: the context, stream and functions are driver handles valid on any
// thread once the context is current there (`current`), and every use of the
// stream is serialised by `commit`.
unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

impl CtxInner {
    fn current(&self) {
        unsafe {
            let _ = cu::ctx::set_current(self.cuctx);
        }
    }
}

pub struct Ctx {
    inner: Arc<CtxInner>,
}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// Process-wide CUDA context. `None` is an honest capability refusal; the
/// caller must not run a CPU implementation under the GPU name.
pub fn ctx() -> Option<&'static Ctx> {
    match CTX.get_or_init(init) {
        Ok(c) => {
            c.inner.current();
            Some(c)
        }
        Err(e) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| eprintln!("cortiq-embryo: CUDA unavailable: {e}"));
            None
        }
    }
}

fn derr(e: cu::DriverError) -> String {
    format!("{e:?}")
}

/// Where the kernels' PTX is kept: `CMF_CUDA_CACHE` (`0` = no cache), else
/// `$XDG_CACHE_HOME/cortiq/cuda` or `~/.cache/cortiq/cuda`.
fn ptx_cache(src: &str, arch: Option<&str>) -> Option<std::path::PathBuf> {
    let dir = match std::env::var("CMF_CUDA_CACHE") {
        Ok(v) if v == "0" => return None,
        Ok(v) if !v.is_empty() => std::path::PathBuf::from(v),
        _ => std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?
            .join("cortiq")
            .join("cuda"),
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.bytes().chain(arch.unwrap_or("").bytes()).chain(env!("CARGO_PKG_VERSION").bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(dir.join(format!("embryo-train-{h:016x}.ptx")))
}

fn init() -> Result<Ctx, String> {
    unsafe {
        if !cs::is_culib_present() {
            return Err("the NVIDIA driver library (libcuda) was not found".into());
        }
        if !cudarc::nvrtc::sys::is_culib_present() {
            return Err("libnvrtc (CUDA toolkit) was not found: it compiles the kernels".into());
        }
    }
    cu::init().map_err(derr)?;
    let ordinal: i32 = std::env::var("CMF_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = cu::device::get(ordinal).map_err(derr)?;
    let cuctx = unsafe { cu::primary_ctx::retain(dev) }.map_err(derr)?;
    unsafe { cu::ctx::set_current(cuctx) }.map_err(derr)?;
    let name = cu::device::get_name(dev).unwrap_or_default();
    let attr = |a| unsafe { cu::device::get_attribute(dev, a) }.unwrap_or(0);
    let major = attr(cs::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    let minor = attr(cs::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    let arch = match (major, minor) {
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 7) => Some("compute_87"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        (10, 0) => Some("compute_100"),
        (12, 0) => Some("compute_120"),
        _ => None,
    };
    // A blocking stream: the synchronous copies of the host mirrors (the
    // legacy stream) are ordered with the launches.
    let stream = cu::stream::create(cu::stream::StreamKind::Default).map_err(derr)?;
    let src = crate::wgsl_cuda::translate(EMBRYO_WGSL);
    let cached = ptx_cache(&src, arch);
    let ptx = match cached.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(t) if t.contains(".entry") => t,
        _ => {
            let opts = cudarc::nvrtc::CompileOptions { arch, ..Default::default() };
            let t = cudarc::nvrtc::compile_ptx_with_opts(&src, opts)
                .map_err(|e| format!("NVRTC: {e:?}"))?
                .to_src();
            if let Some(p) = &cached {
                if let Some(dir) = p.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let tmp = p.with_extension("tmp");
                if std::fs::write(&tmp, &t).is_ok() {
                    let _ = std::fs::rename(&tmp, p);
                }
            }
            t
        }
    };
    let text = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
    let module = unsafe { cu::module::load_data(text.as_ptr().cast()) }.map_err(derr)?;
    let mut funcs = [std::ptr::null_mut(); 11];
    for (f, (n, _)) in funcs.iter_mut().zip(ENTRIES) {
        *f = unsafe { cu::module::get_function(module, std::ffi::CString::new(n).unwrap()) }
            .map_err(|e| format!("kernel {n}: {e:?}"))?;
    }
    let on = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true);
    // Opt-in: on an RTX A5000 cuBLAS left an Embryo-0 step where the kernels
    // had it (335 against 339 ms) and lost on a GDN genome (561 against 350),
    // and it breaks the bit identity of a GEMM's loop and table forms.
    let blas = if std::env::var("CMF_CUDA_CUBLAS").as_deref() == Ok("1") && unsafe { bl::is_culib_present() } {
        let mut h: bl::cublasHandle_t = std::ptr::null_mut();
        let ok = unsafe { bl::cublasCreate_v2(&mut h) } as u32 == 0
            && unsafe { bl::cublasSetStream_v2(h, stream.cast()) } as u32 == 0;
        ok.then_some(h)
    } else {
        None
    };
    let dummy = unsafe { cu::malloc_sync(64 << 10) }.map_err(derr)?;
    unsafe { cu::memset_d8_sync(dummy, 0, 64 << 10) }.map_err(derr)?;
    Ok(Ctx {
        inner: Arc::new(CtxInner {
            cuctx,
            stream,
            funcs,
            big_gpu: name.contains("RTX PRO 6000") || name.contains("RTX 5090"),
            gemm128: on("CMF_VULKAN_GEMM128"),
            gemm64: on("CMF_VULKAN_GEMM64"),
            tile_gemm: on("CMF_VULKAN_TILE_GEMM"),
            profile_ops: std::env::var("CMF_VULKAN_PROFILE_OPS").ok().as_deref() == Some("1"),
            op_ts: std::env::var("CMF_VULKAN_OP_TS").ok().as_deref() == Some("1"),
            blas,
            dummy,
            adapter_name: name,
            splitk_pool: Mutex::new(Vec::new()),
            commit: Mutex::new(ParArena { ptr: 0, words: 0 }),
        }),
    })
}

impl DeviceBuf {
    /// Return the host-mirror pointer after synchronising a device write;
    /// the contract of the wgpu layer's `contents`.
    ///
    /// # Safety
    /// The pointer borrows this buffer's mirror: no command may run and no
    /// mirror method be called while it is dereferenced.
    pub unsafe fn contents(&self) -> *mut c_void {
        self.sync_host();
        self.0.host_dirty.store(true, std::sync::atomic::Ordering::Release);
        unsafe { (*self.0.host.get()).as_mut_ptr() as *mut c_void }
    }
    fn sync_host(&self) {
        if !self.0.gpu_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        self.0.ctx.current();
        let host = unsafe { &mut *self.0.host.get() };
        let n = self.0.len.min(host.len());
        unsafe { cu::memcpy_dtoh_sync(&mut host[..n], self.0.ptr) }.expect("CUDA readback");
    }
    fn upload_host(&self) {
        if !self.0.host_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        self.0.ctx.current();
        let host = unsafe { &*self.0.host.get() };
        unsafe { cu::memcpy_htod_sync(self.0.ptr, &host[..]) }.expect("CUDA upload");
    }
    fn mark_gpu(&self) {
        self.0.gpu_dirty.store(true, std::sync::atomic::Ordering::Release);
    }
}

fn new_buf(ctx: &Arc<CtxInner>, len: usize, init: Option<&[u32]>) -> Arc<BufInner> {
    let n = len.max(1);
    ctx.current();
    // a little past the end: a kernel's guarded tail may still form the address
    let bytes = n * 4 + 256;
    let ptr = unsafe { cu::malloc_sync(bytes) }.unwrap_or_else(|e| panic!("CUDA allocation of {bytes} bytes: {e:?}"));
    let mut host = vec![0u32; n];
    unsafe { cu::memset_d8_sync(ptr, 0, bytes) }.expect("CUDA clear");
    if let Some(x) = init {
        host[..x.len()].copy_from_slice(x);
        unsafe { cu::memcpy_htod_sync(ptr, &host[..]) }.expect("CUDA upload");
    }
    Arc::new(BufInner {
        ptr,
        len,
        owned: true,
        host: UnsafeCell::new(host),
        host_dirty: std::sync::atomic::AtomicBool::new(false),
        gpu_dirty: std::sync::atomic::AtomicBool::new(false),
        ctx: Arc::clone(ctx),
    })
}

impl Ctx {
    fn inner_dummy(&self) -> DeviceBuf {
        DeviceBuf(Arc::new(BufInner {
            ptr: self.inner.dummy,
            len: 1,
            owned: false,
            host: UnsafeCell::new(vec![0]),
            host_dirty: std::sync::atomic::AtomicBool::new(false),
            gpu_dirty: std::sync::atomic::AtomicBool::new(false),
            ctx: Arc::clone(&self.inner),
        }))
    }
}

impl Cmd<'_> {
    /// Record a GEMM for cuBLAS when it is a plain one (no causal mask, no
    /// dynamic K or rows, no table, no band, K > 0): the op-1 parameter block
    /// with `entry` = `ENTRY_CUBLAS`. False = the caller records the kernels.
    #[allow(clippy::too_many_arguments)]
    fn cublas_gemm(
        &self,
        p: &[u32],
        a: &GBuf,
        b: &GBuf,
        c: &GBuf,
        causal: bool,
        dynamic: &GemmDyn<'_>,
        table: &GemmTable<'_>,
        band: Option<&GemmBand>,
    ) -> bool {
        if self.c.inner.blas.is_none()
            || causal
            || dynamic.kcount.is_some()
            || dynamic.indirect.is_some()
            || table.is_some()
            || band.is_some_and(|b| b.mode != 0)
            || p[4] == 0
        {
            return false;
        }
        let mut q = p.to_vec();
        q[PARAM_WORDS - 2] = ENTRY_CUBLAS;
        self.rec(OP_GEMM, &[(a, 0), (b, 1), (c, 2)], q, 256);
        true
    }

    /// The split-K partial buffer of this stream (≥ `n` floats): borrowed
    /// from the context pool on first use, kept until `commit`.
    fn splitk_scratch(&self, n: usize) -> GBuf {
        let mut mine = self.splitk.borrow_mut();
        if !matches!(*mine, Some((_, len)) if len >= n) {
            let mut pool = self.c.inner.splitk_pool.lock().unwrap();
            if let Some(b) = mine.take() {
                pool.push(b);
            }
            *mine = match pool.iter().position(|(_, len)| *len >= n) {
                Some(i) => Some(pool.swap_remove(i)),
                None => {
                    let len = n.max(1 << 22);
                    self.c.inner.current();
                    // kept for the process: the pool lends it to every later stream
                    let raw = unsafe { cu::malloc_sync(len * 4) }.expect("CUDA split-K scratch");
                    Some((raw, len))
                }
            };
        }
        let (raw, len) = mine.as_ref().unwrap();
        // device-only scratch: no host mirror (never uploaded or read back)
        GBuf {
            buf: DeviceBuf(Arc::new(BufInner {
                ptr: *raw,
                len: *len,
                owned: false,
                host: UnsafeCell::new(Vec::new()),
                host_dirty: std::sync::atomic::AtomicBool::new(false),
                gpu_dirty: std::sync::atomic::AtomicBool::new(false),
                ctx: Arc::clone(&self.c.inner),
            })),
            len: *len,
        }
    }

    /// Launch every recorded operation in order on the stream and wait for
    /// the card; the milliseconds from the first launch to the end.
    pub fn commit(self) -> f64 {
        let splitk = self.splitk.borrow_mut().take();
        let ops = self.ops.into_inner();
        let inner = &self.c.inner;
        inner.current();
        let mut arena = inner.commit.lock().unwrap();
        for op in &ops {
            for s in &op.slots {
                s.upload_host();
            }
        }
        let encode_start = std::time::Instant::now();
        let words = ops.len() * PARAM_WORDS;
        if arena.words < words {
            if arena.ptr != 0 {
                unsafe {
                    let _ = cu::free_sync(arena.ptr);
                }
            }
            let want = words.next_power_of_two().max(PARAM_WORDS * 1024);
            arena.ptr = unsafe { cu::malloc_sync(want * 4) }.expect("CUDA parameter arena");
            arena.words = want;
        }
        let mut all = Vec::with_capacity(words);
        for op in &ops {
            all.extend_from_slice(&op.params);
        }
        if !all.is_empty() {
            unsafe { cu::memcpy_htod_sync(arena.ptr, &all) }.expect("CUDA parameter upload");
        }
        let t = std::time::Instant::now();
        let mut events = Vec::new();
        let ev = || cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT).expect("CUDA event");
        if inner.op_ts {
            let e = ev();
            unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
            events.push(e);
        }
        for (i, op) in ops.iter().enumerate() {
            if op.entry == ENTRY_CUBLAS {
                unsafe { blas_gemm(inner, op) };
                if inner.op_ts {
                    let e = ev();
                    unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
                    events.push(e);
                }
                for s in &op.slots {
                    s.mark_gpu();
                }
                continue;
            }
            let entry = (op.entry as usize).min(ENTRIES.len() - 1);
            let (f, block) = (inner.funcs[entry], ENTRIES[entry].1);
            let mut vals = [0u64; SLOTS + 1];
            for (v, s) in vals.iter_mut().zip(&op.slots) {
                *v = s.0.ptr;
            }
            vals[SLOTS] = arena.ptr + (i * PARAM_WORDS * 4) as u64;
            let mut args: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
            let gx = op.groups.clamp(1, 65_535);
            let gy = op.groups.div_ceil(gx).max(1);
            unsafe { cu::launch_kernel(f, (gx, gy, 1), (block, 1, 1), 0, inner.stream, &mut args) }
                .unwrap_or_else(|e| panic!("CUDA launch of op {} (entry {entry}): {e:?}", op.params[0]));
            if inner.op_ts {
                let e = ev();
                unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
                events.push(e);
            }
            for s in &op.slots {
                s.mark_gpu();
            }
        }
        if inner.profile_ops {
            eprintln!("cuda encode ops={} ms={:.3}", ops.len(), encode_start.elapsed().as_secs_f64() * 1000.0);
        }
        unsafe { cu::stream::synchronize(inner.stream) }.expect("CUDA commit");
        if inner.op_ts && events.len() == ops.len() + 1 {
            let mut totals = HashMap::<u32, (f64, usize)>::new();
            let mut gemms = HashMap::<[u32; 7], (f64, usize)>::new();
            let mut span = 0.0;
            for (i, op) in ops.iter().enumerate() {
                let ms = unsafe { cu::event::elapsed(events[i], events[i + 1]) }.unwrap_or(0.0) as f64;
                span += ms;
                let e = totals.entry(op.params[0]).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
                if op.params[0] == OP_GEMM {
                    let p = &op.params;
                    let key = [op.entry, p[5] * 2 + p[6], p[2], p[3], p[4], p[15] * p[16] * p[17], p[31]];
                    let g = gemms.entry(key).or_insert((0.0, 0));
                    g.0 += ms;
                    g.1 += 1;
                }
            }
            eprintln!("cuda commit-ts ops={} span_ms={span:.3}", ops.len());
            let mut rows: Vec<_> = totals.into_iter().collect();
            rows.sort_unstable_by_key(|(op, _)| *op);
            for (op, (ms, count)) in rows {
                eprintln!("cuda op-ts op={op} count={count} gpu_ms={ms:.3}");
            }
            for (k, (ms, count)) in gemms {
                eprintln!(
                    "cuda gemm-ts entry={} t={} m={} n={} k={} batches={} tile={} count={count} gpu_ms={ms:.3}",
                    k[0], k[1], k[2], k[3], k[4], k[5], k[6]
                );
            }
        }
        for e in events {
            unsafe {
                let _ = cu::event::destroy(e);
            }
        }
        drop(arena);
        if let Some(b) = splitk {
            inner.splitk_pool.lock().unwrap().push(b);
        }
        t.elapsed().as_secs_f64() * 1000.0
    }
}

/// The op-1 GEMM of `op` (row-major, three batch levels with their strides)
/// as cuBLAS calls: row-major C = op(A)·op(B) is column-major
/// Cᵀ = op(B)ᵀ·op(A)ᵀ over the same storage, so B goes first; the innermost
/// batch level is one strided call, the outer two a loop.
unsafe fn blas_gemm(inner: &CtxInner, op: &OpRec) {
    let p = &op.params;
    let (m, n, k) = (p[2] as i32, p[3] as i32, p[4] as i32);
    let (ta, tb) = (p[5] == 1, p[6] == 1);
    let (lda, ldb, ldc) = (p[7] as i32, p[8] as i32, p[9] as i32);
    let (al, be) = (f32::from_bits(p[13]), f32::from_bits(p[14]));
    let (nb, nh, nc) = (p[15] as u64, p[16] as u64, p[17] as u64);
    let opr = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
    let h = inner.blas.expect("recorded only with a handle");
    let base = |slot: usize, off: u32| op.slots[slot].0.ptr + 4 * off as u64;
    let (a0, b0, c0) = (base(0, p[10]), base(1, p[11]), base(2, p[12]));
    for cb in 0..nb {
        for ch in 0..nh {
            let a = a0 + 4 * (cb * p[18] as u64 + ch * p[19] as u64);
            let b = b0 + 4 * (cb * p[21] as u64 + ch * p[22] as u64);
            let c = c0 + 4 * (cb * p[24] as u64 + ch * p[25] as u64);
            let s = unsafe {
                if nc == 1 {
                    bl::cublasSgemm_v2(h, opr(tb), opr(ta), n, m, k, &al, b as *const f32, ldb, a as *const f32, lda, &be, c as *mut f32, ldc)
                } else {
                    bl::cublasSgemmStridedBatched(
                        h,
                        opr(tb),
                        opr(ta),
                        n,
                        m,
                        k,
                        &al,
                        b as *const f32,
                        ldb,
                        p[23] as i64,
                        a as *const f32,
                        lda,
                        p[20] as i64,
                        &be,
                        c as *mut f32,
                        ldc,
                        p[26] as i64,
                        nc as i32,
                    )
                }
            };
            assert_eq!(s as u32, 0, "cuBLAS GEMM m={m} n={n} k={k}: {s:?}");
        }
    }
}
