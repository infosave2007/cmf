//! The `qwen4_exp` stack on CUDA: a chunk of tokens goes through every layer
//! on the card, one cuBLAS GEMM per projection and one kernel per glue step
//! (`qwen4.cu`). Sequence state (GDN matrices, conv rings, QSA keys/values,
//! indexer keys, PLE history) lives on the card between chunks.
//!
//! Experts stay in their file form (q4tp) in one arena allocation and are
//! expanded per sub-batch for a strided batched GEMM; the arena is an LRU
//! over (layer, expert) that never evicts what the current layer needs.
//!
//! A speculation round runs as a *window* (`Dev::spec`): the chunk (at most
//! `tape` rows: eight, or 256 with the copy frames) goes through the stack with
//! the recurrent state (GDN matrix and conv ring, PLE history) left aside and
//! the inputs of those updates kept on a tape, and the most likely token of
//! every row comes back. `Dev::settle` then makes the rows that stand current
//! (the tape replays a shorter prefix when drafts were refused). Attention
//! state is positional and needs no rollback.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;

use super::{
    Buf, Ctx, DevMat, Gemm, MvJob, Res, bl, compile, cs, ctx, cu, derr, dev_mat, func, mv_shared_bytes, prec, push_job, rows_per_block,
    vram_budget,
};
use crate::linear_core::GdnWeights;
use crate::pipeline::{Act, DenseFfn, MoeFfn};
use crate::qtensor::QTensor;
use crate::qwen4_exp::{Cfg, GatedResidual, Globals, Layer, Mixer, PleWeights, QsaWeights};

/// Rows of one expert sub-batch (experts x the longest token list among them).
const XR: usize = 16384;
/// Experts of one sub-batch.
const XB: usize = 64;
/// Attention score scratch.
const SBUF: usize = 512 << 20;
/// Indexer score scratch (a part of the attention scratch).
const DS_BUDGET: usize = 256 << 20;
/// Largest chunk of tokens a call takes.
pub(crate) const MAX_CHUNK: usize = 8192;
/// Token rows the small path takes: a decode step or a verify round.
const SMALL_ROWS: usize = 8;
/// Token rows of one unit of the expert kernels (`Q4_UR` of `qwen4.cu`), the
/// words of a unit: the slot address, then the rows.
const UNIT_ROWS: usize = 4;
const UNIT_WORDS: usize = 2 + UNIT_ROWS;
/// Victims a scan of the arena keeps for the next misses, and the largest
/// miss count that keeps them (a decode step, not a prompt).
const SPARE: usize = 64;
const SPARE_NEED: usize = 8;

macro_rules! qfuncs {
    ($($name:ident),* $(,)?) => {
        struct QFuncs { $($name: cs::CUfunction,)* sm: QSmall }
        impl QFuncs {
            fn load(c: &Ctx) -> Res<Self> {
                let module = compile(include_str!("qwen4.cu"), Vec::new(), c.arch)?;
                Ok(Self { $($name: func(module, concat!("q4_", stringify!($name)))?,)* sm: QSmall::load(module)? })
            }
        }
    };
}
qfuncs!(
    tile, grms, hc_low, hc_fold, inject, hist, gdn_conv, gdn_prep, gdn_gb, gdn_scan, gdn_out, norm_rope, idx_reduce,
    idx_topk, qsa_softmax, qsa_merge, ple_gate, ple_conv, route, expand_exp, expand_q8, to_wt, zero, gather, act,
    scatter, shared_add, argmax, mtp_fuse, units,
);
unsafe impl Send for QFuncs {}
unsafe impl Sync for QFuncs {}

/// The kernels of the small path: up to `SMALL_ROWS` token rows straight
/// from the stored weights (int8 bodies, f32 planes, raw expert slots), in
/// f32, with no dense plane and no cuBLAS call.
struct QSmall {
    /// By the number of rows, 1..=SMALL_ROWS.
    mvb: [cs::CUfunction; SMALL_ROWS],
    hc_di: [cs::CUfunction; SMALL_ROWS],
    hc_up: [cs::CUfunction; SMALL_ROWS],
    exp_gu: cs::CUfunction,
    exp_d: cs::CUfunction,
    exp_sum: cs::CUfunction,
    /// Dynamic shared memory a block of these kernels may take.
    shared_max: usize,
}

impl QSmall {
    fn load(module: cs::CUmodule) -> Res<Self> {
        let set = |pre: &str| -> Res<[cs::CUfunction; SMALL_ROWS]> {
            let mut v = [std::ptr::null_mut(); SMALL_ROWS];
            for (i, f) in v.iter_mut().enumerate() {
                *f = func(module, &format!("q4_{pre}{}", i + 1))?;
            }
            Ok(v)
        };
        let (mvb, hc_di, hc_up) = (set("mvb")?, set("hc_di")?, set("hc_up")?);
        let (exp_gu, exp_d) = (func(module, "q4_exp_gu")?, func(module, "q4_exp_d")?);
        // a block stages every row of the round: ask for all a block may have
        let mut shared_max = 48 << 10;
        for want in [200usize << 10, 96 << 10] {
            let ok = mvb.iter().chain(&hc_di).chain(&hc_up).chain([&exp_gu, &exp_d]).all(|&f| {
                unsafe {
                    cu::function::set_function_attribute(
                        f,
                        cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        want as i32,
                    )
                }
                .is_ok()
            });
            if ok {
                shared_max = want;
                break;
            }
        }
        Ok(Self { mvb, hc_di, hc_up, exp_gu, exp_d, exp_sum: func(module, "q4_exp_sum")?, shared_max })
    }
}

/// One matvec of a small-path launch: the weights (kind 7 = q8_2f with its
/// row scale and column field, kind 4 = an f32 plane) and where the rows go.
#[derive(Clone, Copy)]
struct SOp {
    w: u64,
    aux0: u64,
    aux1: u64,
    rows: usize,
    kind: i32,
    y: u64,
}

impl SOp {
    fn q8(m: &DevMat, y: u64) -> Self {
        Self { w: m.w.ptr, aux0: m.aux0(), aux1: m.aux1(), rows: m.rows, kind: 7, y }
    }
    fn f32(w: u64, rows: usize, y: u64) -> Self {
        Self { w, aux0: 0, aux1: 0, rows, kind: 4, y }
    }
}

/// Launch shapes of the small path's kernels. They are bound by the latency
/// of a load more than by the bus, so what counts is how many warps have a
/// load in flight: `CMF_QWEN_CUDA_HC` = "chunk columns, threads of the down
/// kernel, of the up kernel", `CMF_QWEN_CUDA_EXP` = "threads and rows a block
/// of gate/up, of down"; `CMF_QWEN_CUDA_SHARE=0` gives every pick its own
/// unit.
#[derive(Clone, Copy)]
struct Tune {
    cw: usize,
    di_t: usize,
    up_t: usize,
    g_t: usize,
    g_rb: usize,
    d_t: usize,
    d_rb: usize,
    share: bool,
}

impl Tune {
    fn from_env() -> Self {
        let list = |name: &str, def: &[usize]| -> Vec<usize> {
            let mut v = def.to_vec();
            if let Ok(s) = std::env::var(name) {
                for (slot, part) in v.iter_mut().zip(s.split(',')) {
                    if let Some(x) = part.trim().parse::<usize>().ok().filter(|&x| x > 0) {
                        *slot = x;
                    }
                }
            }
            v
        };
        let threads = |t: usize| (t / 32 * 32).clamp(32, 1024);
        let hc = list("CMF_QWEN_CUDA_HC", &[1280, 512, 512]);
        let ex = list("CMF_QWEN_CUDA_EXP", &[512, 32, 512, 64]);
        let (g_t, d_t) = (threads(ex[0]), threads(ex[2]));
        Self {
            cw: (hc[0] / 8 * 8).max(8),
            di_t: threads(hc[1]),
            up_t: threads(hc[2]),
            g_t,
            g_rb: ex[1].min(g_t / 2),
            d_t,
            d_rb: ex[3].min(d_t),
            share: !std::env::var("CMF_QWEN_CUDA_SHARE").is_ok_and(|v| v.trim() == "0"),
        }
    }
}

static QF: OnceLock<Result<QFuncs, String>> = OnceLock::new();

fn qf(c: &Ctx) -> Res<&'static QFuncs> {
    QF.get_or_init(|| {
        c.current()?;
        QFuncs::load(c)
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Launch a kernel of `qwen4.cu`: grid, block size, then its arguments in the
/// kernel's own types (`i32`, `u64`, `f32`, `f64`).
macro_rules! go {
    ($c:expr, $f:expr, $grid:expr, $block:expr; $($a:expr),* $(,)?) => {
        unsafe { $c.launch($f, $grid, $block as u32, 0, kargs!($($a),*)) }?
    };
}

/// Blocks of 256 threads that cover `n` items.
fn blocks(n: usize) -> u32 {
    n.div_ceil(256).max(1) as u32
}

/// `count` GEMMs of one shape with f16 operands and f16 results (f32 inside):
/// cuBLAS takes a strided batch only with one type on all three operands.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_batched_half(
    c: &Ctx,
    ta: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: u64,
    lda: usize,
    stride_a: usize,
    b: u64,
    ldb: usize,
    stride_b: usize,
    cptr: u64,
    ldc: usize,
    stride_c: usize,
    count: usize,
) -> Res<()> {
    let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
    let beta = 0.0f32;
    let ty = bl::cudaDataType::CUDA_R_16F;
    let s = unsafe {
        bl::cublasGemmStridedBatchedEx(
            c.blas,
            op(ta),
            op(false),
            m as i32,
            n as i32,
            k as i32,
            std::ptr::from_ref(&alpha).cast(),
            a as *const c_void,
            ty,
            lda as i32,
            stride_a as i64,
            b as *const c_void,
            ty,
            ldb as i32,
            stride_b as i64,
            std::ptr::from_ref(&beta).cast(),
            cptr as *mut c_void,
            ty,
            ldc as i32,
            stride_c as i64,
            count as i32,
            bl::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if s as u32 != 0 {
        return Err(format!("cuBLAS batched f16 gemm {count} x {m}x{n}x{k}: {s:?}"));
    }
    Ok(())
}

/// An f16 value as f32 (host side, for the range report).
fn h2f(h: u16) -> f32 {
    let (e, m) = (((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if h >> 15 == 1 { -v } else { v }
}

/// Smallest normal f16.
const F16_MIN: f32 = 6.103_515_6e-5;

/// Fill `out` with the rows of `t` (any storage), on a few threads.
fn fill_rows(t: &QTensor, out: &mut [f32]) {
    let cols = t.cols();
    if let Some(d) = t.as_f32()
        && d.len() >= out.len()
    {
        out.copy_from_slice(&d[..out.len()]);
        return;
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = t.rows().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * cols).enumerate() {
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(cols).enumerate() {
                    t.row_f32(ci * per + i, row);
                }
            });
        }
    });
}

/// The tensors' rows one under another, as f32.
fn host_rows(ts: &[&QTensor]) -> Vec<f32> {
    let cols = ts[0].cols();
    let rows: usize = ts.iter().map(|t| t.rows()).sum();
    let mut out = vec![0.0f32; rows * cols];
    let mut r0 = 0;
    for t in ts {
        fill_rows(t, &mut out[r0 * cols..(r0 + t.rows()) * cols]);
        r0 += t.rows();
    }
    out
}

/// Bytes of a q4tp matrix in the file.
fn q4_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / 32) * 16 + rows * 4 + rows * (cols / 32 * 5).div_ceil(8)
}

/// A projection on the card: its q8_2f file form (expanded into the scratch
/// plane right before its GEMM) or a resident plane in the GEMM operand type.
enum DW {
    Q8(DevMat),
    Plane(Buf),
}

impl DW {
    /// The q8_2f file form, when the projection is held in it.
    fn q8(&self) -> Option<&DevMat> {
        match self {
            DW::Q8(m) if m.kind == 7 => Some(m),
            _ => None,
        }
    }
}

/// One hyper-connection mix.
struct Hc {
    norm: Buf,
    /// `[r][w]`: the down projection, then the inject rows.
    di: Buf,
    /// `[w][lr]`.
    up: Buf,
    r: usize,
    lr: usize,
}

struct GdnW {
    qkv: DW,
    z: DW,
    out: DW,
    /// `[2 nv][h]` f32: a rows, then b rows.
    ab: Buf,
    conv: Buf,
    a_log: Buf,
    dt_bias: Buf,
    norm: Buf,
    /// `[kk - 1][c_dim]` raw qkv rows, newest last.
    ring: Buf,
    /// `[nv][dk][dv]`.
    s: Buf,
    /// The next ring and state: a pass writes them and the two are swapped
    /// (a window keeps them aside until it is settled).
    ring2: Buf,
    s2: Buf,
    /// A window's tape: the raw qkv rows `[tape][c_dim]`, their conv outputs,
    /// the gates `[tape][2 nv]`.
    t_s1: Buf,
    t_mix: Buf,
    t_gb: Buf,
    /// A window pass left its result in `s2` / `ring2`.
    dirty: bool,
}

struct QsaW {
    q: DW,
    k: DW,
    v: DW,
    o: DW,
    iqk: DW,
    q_norm: Buf,
    k_norm: Buf,
    iq_norm: Buf,
    ik_norm: Buf,
    /// `[cap][kvh hd]` normalized, rotated keys.
    keys: Buf,
    /// `[cap][kvh hd]`.
    vals: Buf,
    /// `[cap][idim]` raw index keys.
    ikraw: Buf,
    /// `[cap / cr][idim]` block keys.
    bkey: Buf,
    cap: usize,
}

struct PleW {
    key: DW,
    value: DW,
    norm_key: Buf,
    norm_query: Buf,
    norm_conv: Buf,
    conv: Buf,
    /// `[(kernel - 1) dil][w]` normalized values, newest last.
    hist: Buf,
    pin: usize,
    /// The next history, and the window's tape `[tape][w]` of normalized values.
    hist2: Buf,
    t_normed: Buf,
    dirty: bool,
}

enum Mix {
    Gdn(GdnW),
    Qsa(Box<QsaW>),
}

struct MoeW {
    /// `[r][h]` f32: router rows, then the shared expert's gate row.
    router: Buf,
    /// The shared expert in the arena's slot layout.
    shared: Option<Buf>,
    /// Column of the shared gate in the logits, -1 without one.
    gcol: i32,
    r: usize,
    topk: usize,
    /// Routed experts of the layer (the MTP head has 512, the backbone's cut layers fewer).
    ne: usize,
}

struct LayerW {
    attn: Hc,
    mlp: Hc,
    mix: Mix,
    ple: Option<PleW>,
    moe: MoeW,
}

#[derive(Clone, Copy)]
struct Dims {
    h: usize,
    hc: usize,
    w: usize,
    eps: f64,
    /// The GDN output norm has its own epsilon.
    gdn_eps: f64,
    nv: usize,
    nk: usize,
    dk: usize,
    dv: usize,
    kk: usize,
    c_dim: usize,
    nh: usize,
    kvh: usize,
    hd: usize,
    groups: usize,
    ih: usize,
    idim: usize,
    keep: usize,
    cr: usize,
    rot: usize,
    nfreq: usize,
    ne: usize,
    inter: usize,
    /// Bytes of an expert's gate (and up) matrix, of its down matrix.
    sgu: usize,
    sd: usize,
    vocab: usize,
    ple_k: usize,
    ple_dil: usize,
}

struct Scratch {
    emb: Buf,
    hyper: Buf,
    normed: Buf,
    normed_h: Buf,
    low: Buf,
    lowh: Buf,
    /// The small path's chunk sums of the low-rank rows `[chunks][n][r]`.
    hcp: Buf,
    iw: Buf,
    mix: Buf,
    s1: Buf,
    folded: Buf,
    folded_h: Buf,
    blk: Buf,
    z: Buf,
    qo: Buf,
    qq: Buf,
    ofh: Buf,
    ab: Buf,
    gb: Buf,
    kraw: Buf,
    iq: Buf,
    sbuf: Buf,
    sc: Buf,
    ple_in: Buf,
    ple_h: Buf,
    logits: Buf,
    picks: Buf,
    wts: Buf,
    rowtok: Buf,
    tokrow: Buf,
    slots: Buf,
    xg: Buf,
    g: Buf,
    u: Buf,
    ah: Buf,
    d: Buf,
    wg: Buf,
    wu: Buf,
    wd: Buf,
    wq: Buf,
    out: Buf,
    /// Token ids of a window's rows.
    ids: Buf,
}

/// Page-locked host memory the expert uploads are staged in: a copy from it
/// is a plain DMA transfer, and the staging itself runs on several threads.
struct Pinned {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the block is owned by one `Arena` and reached through `&mut` only.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Pinned {
    fn new(len: usize) -> Option<Self> {
        let ptr = unsafe { cu::malloc_host(len, 0) }.ok()? as *mut u8;
        (!ptr.is_null()).then_some(Self { ptr, len })
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let _ = unsafe { cu::free_host(self.ptr as *mut c_void) };
    }
}

/// The file pages of the experts the arena keeps losing, pinned where they are
/// (`cuMemHostRegister`, read-only): an upload is one DMA straight from the
/// mapped file, with no copy through the staging first (RTX 5090: 19 us for
/// 880 KB against 54 us from pageable memory, and none of the 160 us copy).
/// Pinned pages cannot be dropped from the page cache, so only the experts
/// outside the prefilled arena are taken. A thread pins layer after layer;
/// `ready[li]` says when a layer's are pinned.
struct Pins {
    /// Per layer, per expert: pinned once the layer is ready.
    sel: Vec<Vec<bool>>,
    ready: Arc<Vec<AtomicBool>>,
    done: Arc<Mutex<Vec<(usize, usize)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Pins {
    /// `ranges[li]` = the page-aligned, disjoint (address, bytes) of layer `li`.
    fn start(c: &'static Ctx, sel: Vec<Vec<bool>>, ranges: Vec<Vec<(usize, usize)>>) -> Self {
        let ready: Arc<Vec<AtomicBool>> = Arc::new((0..ranges.len()).map(|_| AtomicBool::new(false)).collect());
        let done = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (r2, d2, s2) = (ready.clone(), done.clone(), stop.clone());
        let run = move || {
            if c.current().is_err() {
                return;
            }
            let t0 = Instant::now();
            let (mut bytes, mut failed) = (0usize, 0usize);
            'layers: for (li, rs) in ranges.iter().enumerate() {
                let mut ok = true;
                for &(at, len) in rs {
                    if s2.load(Ordering::Acquire) {
                        break 'layers;
                    }
                    // flag 8: the pages are read-only to the card
                    let rc = unsafe { cs::cuMemHostRegister_v2(at as *mut c_void, len, 8) };
                    if rc == cs::cudaError_enum::CUDA_SUCCESS {
                        d2.lock().unwrap().push((at, len));
                        bytes += len;
                    } else {
                        ok = false;
                        failed += 1;
                        if failed == 1 {
                            eprintln!("qwen4 cuda: pinning experts of layer {li} failed ({rc:?}); they go through the staging");
                        }
                    }
                }
                // a layer with a failed range stays on the staging as a whole
                r2[li].store(ok, Ordering::Release);
            }
            if std::env::var_os("CMF_QWEN_PROF").is_some() {
                eprintln!("qwen4 cuda: pinned {:.2} GiB of expert pages in {:.1} s", bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
            }
        };
        let thread = std::thread::Builder::new().name("cmf-qwen-pin".into()).spawn(run).ok();
        Self { sel, ready, done, stop, thread }
    }

    fn has(&self, li: usize, e: usize) -> bool {
        self.ready.get(li).is_some_and(|f| f.load(Ordering::Acquire)) && self.sel[li].get(e).copied().unwrap_or(false)
    }
}

impl Drop for Pins {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        for &(at, _) in self.done.lock().unwrap().iter() {
            let _ = unsafe { cs::cuMemHostUnregister(at as *mut c_void) };
        }
    }
}

/// Threads that stage a small upload wave. A decode step misses an expert
/// or two a layer; one thread copies an expert in about a quarter of a
/// millisecond, and threads spawned for it would cost as much, so these stay
/// parked on their queues between waves.
struct CopyPool {
    tx: Vec<mpsc::Sender<(usize, usize, usize)>>,
    done: Mutex<mpsc::Receiver<()>>,
}

impl CopyPool {
    /// `CMF_QWEN_CUDA_COPY` = the number of threads (0: the caller alone).
    fn new() -> Option<Self> {
        let n = std::env::var("CMF_QWEN_CUDA_COPY")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()).min(8).saturating_sub(1));
        let (ack, done) = mpsc::channel::<()>();
        let mut tx = Vec::new();
        for _ in 0..n.min(32) {
            let (t, rx) = mpsc::channel::<(usize, usize, usize)>();
            let ack = ack.clone();
            let run = move || {
                for (dst, src, len) in rx {
                    // SAFETY: `run` keeps both ranges alive and to this piece
                    // alone until it has read the piece's answer.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                    if ack.send(()).is_err() {
                        break;
                    }
                }
            };
            if std::thread::Builder::new().name("cmf-qwen-copy".into()).spawn(run).is_ok() {
                tx.push(t);
            }
        }
        (!tx.is_empty()).then_some(Self { tx, done: Mutex::new(done) })
    }

    /// Copy the spans (to, from, bytes), the caller one share of them.
    /// `false`: a thread is gone and the spans are to be copied again.
    fn run(&self, spans: &[(usize, usize, usize)]) -> bool {
        let lanes = self.tx.len() + 1;
        let total: usize = spans.iter().map(|s| s.2).sum();
        let step = (total / lanes).next_multiple_of(4096).max(128 << 10);
        let done = self.done.lock().unwrap();
        let (mut sent, mut i) = (0usize, 0usize);
        let mut own: Vec<(usize, usize, usize)> = Vec::new();
        for &(dst, src, len) in spans {
            let mut at = 0;
            while at < len {
                let piece = (dst + at, src + at, step.min(len - at));
                at += piece.2;
                let w = i % lanes;
                i += 1;
                if w + 1 < lanes && self.tx[w].send(piece).is_ok() {
                    sent += 1;
                } else {
                    own.push(piece);
                }
            }
        }
        for &(dst, src, len) in &own {
            // SAFETY: a piece of the caller's spans, given to no thread.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
        }
        (0..sent).all(|_| done.recv().is_ok())
    }
}

/// The expert arena: raw q4tp slots (gate, up, down) of (layer, expert).
///
/// A pass (one `forward` call) visits the layers in order, so the next use of
/// a resident expert is known up to whether its layer picks it again: an
/// expert of a layer ahead is due in this pass, one of a layer behind in the
/// next. A victim is the expert unused for the most passes, and among those
/// the one whose layer comes up last. A prompt that touches more experts than
/// the arena holds then re-reads only the overflow, where a plain LRU would
/// evict each layer just before its turn and re-read everything.
struct Arena {
    buf: Buf,
    slot: usize,
    n: usize,
    used: usize,
    key: Vec<u32>,
    /// Pass of the slot's last use.
    last: Vec<u64>,
    map: HashMap<u32, u32>,
    /// Layers of the model, the pass under way.
    nl: u64,
    pass: u64,
    /// The upload staging, the experts it holds, whether a copy from it may
    /// still be in flight.
    stage: Option<Pinned>,
    stage_cap: usize,
    pending: bool,
    pool: Option<CopyPool>,
    /// Victims the last scan found and did not use: (slot, its expert, its
    /// last use), the best last. One holds while its slot still has that
    /// expert, unused since.
    spare: Vec<(u32, u32, u64)>,
    /// Experts uploaded and evicted, the host time of the staging, the units
    /// of the small path and the picks they served (since the last report).
    admitted: u64,
    evicted: u64,
    stage_ms: f64,
    units: u64,
    pairs: u64,
    /// The pinned file pages of the experts (`CMF_QWEN_CUDA_REG`).
    pins: Option<Pins>,
    /// Experts of the layers from this index on (the MTP head's) are never evicted.
    sticky_from: u32,
    /// Slots whose expert a refill chose for the request: evicted only when nothing else can be.
    prot: Vec<bool>,
    /// How often each (layer, expert) was picked since the sequence began, while `hist_on`
    /// (the prompt's routing: `Dev::retarget` refills the arena from it).
    hist: Vec<u32>,
    hist_ne: usize,
    hist_on: bool,
    /// Decoding routes among the experts the arena holds (`CMF_QWEN_MASK=1`, set by `retarget`).
    masked: bool,
    /// The slot address of every (layer, expert) on the card, 0 for not in the arena
    /// (`[row][tbl_stride]`, a row a layer, the MTP head's after them): what the masked
    /// route and the lean path's units read. Rebuilt when `epoch` (a count of changes of
    /// the arena's contents) is ahead of `tbl_epoch`.
    tbl: Buf,
    tbl_stride: usize,
    tbl_ne: Vec<usize>,
    tbl_held: Vec<usize>,
    tbl_epoch: u64,
    epoch: u64,
}

/// Copy the experts' file bytes `parts` into the slot-sized `dst` slices, on
/// several threads for a wave and on the copy pool's for a few.
fn copy_jobs(pool: Option<&CopyPool>, jobs: &mut [(&mut [u8], [&[u8]; 3])], sgu: usize) {
    let wave = jobs.len();
    let copy = |group: &mut [(&mut [u8], [&[u8]; 3])]| {
        for (dst, parts) in group.iter_mut() {
            dst[..sgu].copy_from_slice(parts[0]);
            dst[sgu..2 * sgu].copy_from_slice(parts[1]);
            dst[2 * sgu..].copy_from_slice(parts[2]);
        }
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(24).min(wave.div_ceil(4));
    if threads <= 1 {
        let spans: Vec<(usize, usize, usize)> = jobs
            .iter_mut()
            .flat_map(|(dst, parts)| {
                let to = dst.as_mut_ptr() as usize;
                [(to, parts[0].as_ptr() as usize, sgu), (to + sgu, parts[1].as_ptr() as usize, sgu), (
                    to + 2 * sgu,
                    parts[2].as_ptr() as usize,
                    parts[2].len(),
                )]
            })
            .collect();
        if !pool.is_some_and(|p| p.run(&spans)) {
            copy(jobs);
        }
    } else {
        let per = wave.div_ceil(threads);
        std::thread::scope(|sc| {
            for group in jobs.chunks_mut(per) {
                sc.spawn(|| copy(group));
            }
        });
    }
}

impl Arena {
    /// The table of slot addresses, rebuilt if the arena changed since.
    fn refresh_table(&mut self, c: &Ctx) -> Res<()> {
        if self.tbl_epoch == self.epoch {
            return Ok(());
        }
        let mut host = vec![0u64; self.tbl_ne.len() * self.tbl_stride];
        for (li, &ne) in self.tbl_ne.iter().enumerate() {
            let mut held = 0usize;
            for e in 0..ne {
                if let Some(&s) = self.map.get(&(((li as u32) << 16) | e as u32)) {
                    host[li * self.tbl_stride + e] = self.buf.ptr + (s as usize * self.slot) as u64;
                    held += 1;
                }
            }
            self.tbl_held[li] = held;
        }
        unsafe { cu::memcpy_htod_async(self.tbl.ptr, &host, c.stream) }.map_err(derr)?;
        self.tbl_epoch = self.epoch;
        if std::env::var_os("CMF_QWEN_PROF").is_some() && self.masked {
            let main = &self.tbl_held[..self.main_layers().min(self.tbl_held.len())];
            let (lo, hi) = (main.iter().min().copied().unwrap_or(0), main.iter().max().copied().unwrap_or(0));
            eprintln!(
                "qwen4 cuda: address table at epoch {}: experts held a layer {lo}..{hi} (layers with 10 or fewer: {})",
                self.epoch,
                main.iter().filter(|&&h| h <= 10).count()
            );
        }
        Ok(())
    }

    /// The table row of layer `li` as a device address, when the route may use it: the
    /// route is masked and the layer holds more than `topk` experts, or `force` (the MTP
    /// head, whose experts all stay). 0: no.
    fn tbl_ptr(&mut self, c: &Ctx, li: usize, topk: usize, force: bool) -> Res<u64> {
        if li >= self.tbl_ne.len() || !(self.masked || force) {
            return Ok(0);
        }
        self.refresh_table(c)?;
        if !force && self.tbl_held[li] <= topk {
            return Ok(0);
        }
        Ok(self.tbl.ptr + (li * self.tbl_stride * 8) as u64)
    }

    /// Layers of the backbone (the rows before the MTP head's).
    fn main_layers(&self) -> usize {
        self.hist.len() / self.hist_ne.max(1)
    }

    /// Count the picks of a layer's step.
    fn note(&mut self, li: usize, picks: &[i32]) {
        if !self.hist_on || (li + 1) * self.hist_ne > self.hist.len() {
            return;
        }
        let row = &mut self.hist[li * self.hist_ne..(li + 1) * self.hist_ne];
        for &e in picks {
            if e >= 0 && (e as usize) < row.len() {
                row[e as usize] += 1;
            }
        }
    }

    /// The slots of the layer's experts `order` into `slot_of`, uploading the
    /// absent ones. The copies go through the stream, behind the queued
    /// kernels: a slot is never overwritten under a kernel that reads it.
    fn admit_layer(&mut self, li: usize, order: &[usize], experts: &[DenseFfn], d: &Dims, c: &Ctx, slot_of: &mut [u64]) -> Res<()> {
        let (pass, slot) = (self.pass, self.slot);
        let mut miss: Vec<usize> = Vec::new();
        for &e in order {
            match self.map.get(&(((li as u32) << 16) | e as u32)) {
                Some(&s) => {
                    self.last[s as usize] = pass;
                    slot_of[e] = self.buf.ptr + (s as usize * slot) as u64;
                }
                None => miss.push(e),
            }
        }
        if miss.is_empty() {
            return Ok(());
        }
        self.epoch += 1;
        let mut slots: Vec<u32> = Vec::with_capacity(miss.len());
        while slots.len() < miss.len() && self.used < self.n {
            slots.push(self.used as u32);
            self.used += 1;
        }
        let mut need = miss.len() - slots.len();
        while need > 0 {
            let Some((s, key, last)) = self.spare.pop() else { break };
            if self.key[s as usize] != key || self.last[s as usize] != last {
                continue;
            }
            self.map.remove(&key);
            self.key[s as usize] = u32::MAX;
            self.evicted += 1;
            slots.push(s);
            need -= 1;
        }
        if need > 0 {
            let (nl, lu) = (self.nl, li as u64);
            let mut cand: Vec<(u64, u32)> = Vec::with_capacity(self.n);
            // the experts a refill chose for the request (`prot`) go last
            for allow_prot in [false, true] {
                cand.clear();
                for s in 0..self.n {
                    let key = self.key[s];
                    if key == u32::MAX || (self.prot[s] && !allow_prot) {
                        continue;
                    }
                    let (j, lp) = ((key >> 16) as u64, self.last[s]);
                    if (key >> 16) >= self.sticky_from || (j == lu && lp == pass) {
                        continue;
                    }
                    // passes the expert went unused when its layer came up
                    let age = if j <= lu { pass - lp } else { (pass - lp).saturating_sub(1) };
                    let dist = if j > lu { j - lu } else { nl - lu + j };
                    cand.push((age * nl + dist, s as u32));
                }
                if cand.len() >= need {
                    break;
                }
            }
            if cand.len() < need {
                return Err("qwen4 cuda: the expert arena is smaller than one layer's experts".into());
            }
            // a decode step: the runners-up of this scan serve the next misses
            let take = if need <= SPARE_NEED { (need + SPARE).min(cand.len()) } else { need };
            if take < cand.len() {
                cand.select_nth_unstable_by(take - 1, |a, b| b.cmp(a));
            }
            cand.truncate(take);
            if take > need {
                cand.sort_unstable_by(|a, b| b.cmp(a));
            }
            for &(_, s) in &cand[..need] {
                self.map.remove(&self.key[s as usize]);
                self.key[s as usize] = u32::MAX;
                self.evicted += 1;
                slots.push(s);
            }
            self.spare.clear();
            for &(_, s) in cand[need..].iter().rev() {
                self.spare.push((s, self.key[s as usize], self.last[s as usize]));
            }
        }
        let t0 = Instant::now();
        let sgu = d.sgu;
        // (expert, arena slot) that go through the upload staging; the experts
        // whose file pages are pinned are one DMA per matrix, straight from the file
        let mut rest: Vec<(usize, u32)> = Vec::with_capacity(miss.len());
        for (&e, &s) in miss.iter().zip(&slots) {
            if self.pins.as_ref().is_some_and(|p| p.has(li, e)) {
                let parts = expert_parts(&experts[e], d)?;
                let base = self.buf.ptr + (s as usize * slot) as u64;
                unsafe {
                    cu::memcpy_htod_async(base, parts[0], c.stream).map_err(derr)?;
                    cu::memcpy_htod_async(base + sgu as u64, parts[1], c.stream).map_err(derr)?;
                    cu::memcpy_htod_async(base + 2 * sgu as u64, parts[2], c.stream).map_err(derr)?;
                }
            } else {
                rest.push((e, s));
            }
        }
        let mut done = 0usize;
        while done < rest.len() {
            let Some(pin) = self.stage.as_mut() else {
                // no staging: pageable copies, staged by the driver one by one
                for &(e, s) in &rest {
                    let parts = expert_parts(&experts[e], d)?;
                    let base = self.buf.ptr + (s as usize * slot) as u64;
                    unsafe {
                        cu::memcpy_htod_async(base, parts[0], c.stream).map_err(derr)?;
                        cu::memcpy_htod_async(base + sgu as u64, parts[1], c.stream).map_err(derr)?;
                        cu::memcpy_htod_async(base + 2 * sgu as u64, parts[2], c.stream).map_err(derr)?;
                    }
                }
                break;
            };
            if self.pending {
                // the last wave is still read from the staging
                c.sync()?;
            }
            let wave = (rest.len() - done).min(self.stage_cap);
            // SAFETY: the block is `stage_cap` slots long and this is its only user.
            let st = unsafe { std::slice::from_raw_parts_mut(pin.ptr, pin.len) };
            // the wave goes in pieces: a piece is on its way over the bus while the next is copied
            const PIECE: usize = 24;
            let mut c0 = 0;
            while c0 < wave {
                let c1 = (c0 + PIECE).min(wave);
                {
                    let mut jobs: Vec<(&mut [u8], [&[u8]; 3])> = Vec::with_capacity(c1 - c0);
                    for (dst, &(e, _)) in st[c0 * slot..c1 * slot].chunks_mut(slot).zip(&rest[done + c0..done + c1]) {
                        jobs.push((dst, expert_parts(&experts[e], d)?));
                    }
                    copy_jobs(self.pool.as_ref(), &mut jobs, sgu);
                }
                for i in c0..c1 {
                    let base = self.buf.ptr + (rest[done + i].1 as usize * slot) as u64;
                    unsafe { cu::memcpy_htod_async(base, &st[i * slot..(i + 1) * slot], c.stream) }.map_err(derr)?;
                }
                c0 = c1;
            }
            self.pending = true;
            done += wave;
        }
        self.stage_ms += t0.elapsed().as_secs_f64() * 1e3;
        for (&e, &s) in miss.iter().zip(&slots) {
            let key = ((li as u32) << 16) | e as u32;
            self.key[s as usize] = key;
            self.last[s as usize] = pass;
            self.prot[s as usize] = false;
            self.map.insert(key, s);
            self.admitted += 1;
            slot_of[e] = self.buf.ptr + (s as usize * slot) as u64;
        }
        Ok(())
    }
}

/// Bytes of one expert in the file.
fn slot_bytes(d: &Dims) -> usize {
    2 * d.sgu + d.sd
}

/// The three matrices of an expert as the arena stores them.
fn expert_parts<'a>(f: &'a DenseFfn, d: &Dims) -> Res<[&'a [u8]; 3]> {
    if !matches!(f.act, Act::Silu) {
        return Err("qwen4 cuda: an expert without SiLU".into());
    }
    if !f.segs.is_empty() {
        return Err("qwen4 cuda: an expert with task tubes".into());
    }
    let part = |t: &'a QTensor, rows: usize, cols: usize, len: usize| -> Res<&'a [u8]> {
        if t.rows() != rows || t.cols() != cols || t.mapped_q4tp().is_none() {
            return Err(format!("qwen4 cuda: an expert matrix is not q4tp {rows}x{cols}"));
        }
        match t.mapped_bytes() {
            Some(b) if b.len() == len => Ok(b),
            Some(b) => Err(format!("qwen4 cuda: an expert matrix of {} bytes, {len} expected", b.len())),
            None => Err("qwen4 cuda: an expert matrix without file bytes".into()),
        }
    };
    Ok([
        part(&f.gate_proj, d.inter, d.h, d.sgu)?,
        part(&f.up_proj, d.inter, d.h, d.sgu)?,
        part(&f.down_proj, d.h, d.inter, d.sd)?,
    ])
}

/// The routing this path reproduces: softmax over all experts, top-k,
/// renormalised, no bias, no temperature, unit scale, no mask, no grown experts.
fn moe_ok(m: &MoeFfn) -> bool {
    !m.router_sigmoid
        && m.norm_topk_prob
        && m.route_tau.is_none()
        && m.expert_bias.is_none()
        && (m.routed_scaling - 1.0).abs() <= 1e-6
        && m.mask.is_none()
        && m.per_expert_scale.is_none()
        && !m.router_input_norm
        && m.resonance.is_none()
        && m.grown.is_empty()
}

/// Weight placement while the device state is built.
struct Build {
    c: &'static Ctx,
    q: &'static QFuncs,
    half: bool,
    held: u64,
    /// Elements of the largest q8 projection (the scratch plane's size).
    wq_elems: usize,
    /// `CMF_QWEN_CUDA_RANGE`: report how the planes sit in the f16 range.
    range: bool,
    li: usize,
    /// Rows of a window's tape.
    tape: usize,
}

impl Build {
    fn buf(&mut self, bytes: usize) -> Res<Buf> {
        let b = Buf::new(bytes)?;
        self.held += b.bytes as u64;
        Ok(b)
    }

    /// `elems` zeroed f32.
    fn zeroed(&mut self, elems: usize) -> Res<Buf> {
        let b = self.buf(elems * 4)?;
        go!(self.c, self.q.zero, (blocks(elems), 1), 256; b.ptr, elems as u64);
        Ok(b)
    }

    fn f32buf(&mut self, d: &[f32]) -> Res<Buf> {
        let b = Buf::from_slice(d)?;
        self.held += b.bytes as u64;
        Ok(b)
    }

    /// The range report of a plane `[rows][cols]`.
    fn wstat(&self, name: &str, d: &[f32], cols: usize) {
        if !self.range || d.is_empty() || cols == 0 {
            return;
        }
        let (mut max, mut sub, mut e_all, mut e_sub) = (0f32, 0usize, 0f64, 0f64);
        let mut cmax = vec![0f32; cols];
        for row in d.chunks(cols) {
            for (c, &v) in cmax.iter_mut().zip(row) {
                let a = v.abs();
                max = max.max(a);
                *c = c.max(a);
                e_all += (a as f64) * (a as f64);
                if a > 0.0 && a < F16_MIN {
                    sub += 1;
                    e_sub += (a as f64) * (a as f64);
                }
            }
        }
        let small = cmax.iter().filter(|&&m| m < 8.0 * F16_MIN).count();
        let cmin = cmax.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "qwen4 cuda range L{} w {name}: max {max:.3e}, under the f16 normals {:.3} % of the values and {:.3e} of the energy; {small} of {cols} columns under 4.9e-4, smallest column max {cmin:.3e}",
            self.li,
            100.0 * sub as f64 / d.len() as f64,
            e_sub / e_all.max(1e-300),
        );
    }

    /// A plane in the GEMM operand type (f16 under `fast`, else f32).
    fn plane_wt(&mut self, d: &[f32]) -> Res<Buf> {
        if !self.half {
            return self.f32buf(d);
        }
        let src = Buf::from_slice(d)?;
        let dst = self.buf(d.len() * 2)?;
        go!(self.c, self.q.to_wt, (blocks(d.len()), 1), 256; src.ptr, dst.ptr, d.len() as u64, 1i32);
        self.c.sync()?;
        Ok(dst)
    }

    fn dw(&mut self, t: &QTensor, name: &str) -> Res<DW> {
        if self.range {
            self.wstat(name, &host_rows(&[t]), t.cols());
        }
        if t.mapped_q8_2f().is_some()
            && t.cols().is_multiple_of(16)
            && let Some(bytes) = t.mapped_bytes()
        {
            let m = dev_mat(7, bytes, t.rows(), t.cols())?;
            self.held += m.held;
            self.wq_elems = self.wq_elems.max(t.rows() * t.cols());
            return Ok(DW::Q8(m));
        }
        Ok(DW::Plane(self.plane_wt(&host_rows(&[t]))?))
    }

    fn hc(&mut self, g: &GatedResidual, d: &Dims, inject: bool) -> Res<Hc> {
        let lr = g.down.rows();
        if g.down.cols() != d.w || g.up.rows() != d.w || g.up.cols() != lr || g.norm.len() != d.w {
            return Err("qwen4 cuda: unexpected hyper-connection shapes".into());
        }
        let mut ts = vec![&g.down];
        match (&g.inject, inject) {
            (Some(i), true) if i.rows() == d.hc && i.cols() == d.w => ts.push(i),
            (None, false) => {}
            _ => return Err("qwen4 cuda: unexpected hyper-connection inject".into()),
        }
        let r = ts.iter().map(|t| t.rows()).sum();
        let (di, up) = (host_rows(&ts), host_rows(&[&g.up]));
        self.wstat("hc.down", &di, d.w);
        self.wstat("hc.up", &up, lr);
        self.wstat("hc.norm", &g.norm, d.w);
        Ok(Hc { norm: self.f32buf(&g.norm)?, di: self.plane_wt(&di)?, up: self.plane_wt(&up)?, r, lr })
    }

    fn gdn(&mut self, g: &GdnWeights, d: &Dims) -> Res<GdnW> {
        let vd = d.nv * d.dv;
        let ok = g.in_proj_qkv.rows() == d.c_dim
            && g.in_proj_qkv.cols() == d.h
            && g.in_proj_z.rows() == vd
            && g.in_proj_z.cols() == d.h
            && g.in_proj_a.rows() == d.nv
            && g.in_proj_b.rows() == d.nv
            && g.in_proj_a.cols() == d.h
            && g.in_proj_b.cols() == d.h
            && g.out_proj.rows() == d.h
            && g.out_proj.cols() == vd
            && g.conv1d.len() == d.c_dim * d.kk
            && g.a_log.len() == d.nv
            && g.dt_bias.len() == d.nv
            && g.norm.len() == d.dv;
        if !ok {
            return Err("qwen4 cuda: unexpected GDN shapes".into());
        }
        Ok(GdnW {
            qkv: self.dw(&g.in_proj_qkv, "gdn.qkv")?,
            z: self.dw(&g.in_proj_z, "gdn.z")?,
            out: self.dw(&g.out_proj, "gdn.out")?,
            ab: self.f32buf(&host_rows(&[&g.in_proj_a, &g.in_proj_b]))?,
            conv: self.f32buf(&g.conv1d)?,
            a_log: self.f32buf(&g.a_log)?,
            dt_bias: self.f32buf(&g.dt_bias)?,
            norm: self.f32buf(&g.norm)?,
            ring: self.zeroed((d.kk - 1).max(1) * d.c_dim)?,
            s: self.zeroed(d.nv * d.dk * d.dv)?,
            ring2: self.zeroed((d.kk - 1).max(1) * d.c_dim)?,
            s2: self.zeroed(d.nv * d.dk * d.dv)?,
            t_s1: self.buf(self.tape * d.c_dim * 4)?,
            t_mix: self.buf(self.tape * d.c_dim * 4)?,
            t_gb: self.buf(self.tape * 2 * d.nv * 4)?,
            dirty: false,
        })
    }

    fn mtp(&mut self, p: &MtpParts, d: &Dims, li: usize, _globals: &Globals) -> Res<MtpW> {
        let Mixer::Qsa(qw) = &p.layer.mixer else {
            return Err("the MTP layer is not QSA".into());
        };
        if p.layer.ple.is_some() || p.enorm.len() != d.h || p.hnorm.len() != d.w {
            return Err("unexpected MTP head shapes".into());
        }
        let ne = p.layer.moe.experts.len();
        let attn = self.hc(&p.layer.attn_hc, d, true)?;
        let mlp = self.hc(&p.layer.mlp_hc, d, true)?;
        let mix = Mix::Qsa(Box::new(self.qsa(qw, d)?));
        let moe = self.moe(&p.layer.moe, d, ne)?;
        let mixer = self.hc(p.mixer, d, false)?;
        let mut mat = |t: &QTensor| -> Res<DevMat> {
            if t.rows() != d.h || t.cols() != d.h || t.mapped_q8_2f().is_none() {
                return Err("an MTP projection is not q8_2f hidden x hidden".into());
            }
            let bytes = t.mapped_bytes().ok_or_else(|| "an MTP projection has no file bytes".to_string())?;
            let m = dev_mat(7, bytes, t.rows(), t.cols())?;
            self.held += m.held;
            Ok(m)
        };
        let (fc_e, fc_h) = (mat(p.fc_e)?, mat(p.fc_h)?);
        Ok(MtpW {
            lw: LayerW { attn, mlp, mix, ple: None, moe },
            li,
            mixer,
            fc_e,
            fc_h,
            enorm: self.f32buf(p.enorm)?,
            hnorm: self.f32buf(p.hnorm)?,
            r_last: self.zeroed(d.w)?,
            r_pos: None,
            en: self.zeroed(d.h)?,
            e: self.zeroed(d.h)?,
            rn: self.zeroed(d.w)?,
            hs: self.zeroed(d.w)?,
        })
    }

    fn qsa(&mut self, a: &QsaWeights, d: &Dims) -> Res<QsaW> {
        let kv = d.kvh * d.hd;
        let ok = a.q_proj.rows() == d.nh * 2 * d.hd
            && a.q_proj.cols() == d.h
            && a.k_proj.rows() == kv
            && a.k_proj.cols() == d.h
            && a.v_proj.rows() == kv
            && a.v_proj.cols() == d.h
            && a.o_proj.rows() == d.h
            && a.o_proj.cols() == d.nh * d.hd
            && a.index_qk.rows() == d.ih * d.idim + d.idim
            && a.index_qk.cols() == d.h
            && a.q_norm.len() == d.hd
            && a.k_norm.len() == d.hd
            && a.index_q_norm.len() == d.idim
            && a.index_k_norm.len() == d.idim;
        if !ok {
            return Err("qwen4 cuda: unexpected QSA shapes".into());
        }
        let cap = MAX_CHUNK;
        Ok(QsaW {
            q: self.dw(&a.q_proj, "qsa.q")?,
            k: self.dw(&a.k_proj, "qsa.k")?,
            v: self.dw(&a.v_proj, "qsa.v")?,
            o: self.dw(&a.o_proj, "qsa.o")?,
            iqk: self.dw(&a.index_qk, "qsa.iqk")?,
            q_norm: self.f32buf(&a.q_norm)?,
            k_norm: self.f32buf(&a.k_norm)?,
            iq_norm: self.f32buf(&a.index_q_norm)?,
            ik_norm: self.f32buf(&a.index_k_norm)?,
            keys: self.zeroed(cap * kv)?,
            vals: self.zeroed(cap * kv)?,
            ikraw: self.zeroed(cap * d.idim)?,
            bkey: self.zeroed((cap / d.cr + 1) * d.idim)?,
            cap,
        })
    }

    fn ple(&mut self, p: &PleWeights, d: &Dims) -> Res<PleW> {
        let pin = p.key_proj.cols();
        let ok = p.key_proj.rows() == d.w
            && p.value_proj.rows() == d.h
            && p.value_proj.cols() == pin
            && p.norm_key.len() == d.w
            && p.norm_query.len() == d.w
            && p.norm_conv.len() == d.w
            && p.conv.len() == d.w * d.ple_k;
        if !ok {
            return Err("qwen4 cuda: unexpected PLE shapes".into());
        }
        Ok(PleW {
            key: self.dw(&p.key_proj, "ple.key")?,
            value: self.dw(&p.value_proj, "ple.value")?,
            norm_key: self.f32buf(&p.norm_key)?,
            norm_query: self.f32buf(&p.norm_query)?,
            norm_conv: self.f32buf(&p.norm_conv)?,
            conv: self.f32buf(&p.conv)?,
            hist: self.zeroed(((d.ple_k - 1) * d.ple_dil).max(1) * d.w)?,
            pin,
            hist2: self.zeroed(((d.ple_k - 1) * d.ple_dil).max(1) * d.w)?,
            t_normed: self.buf(self.tape * d.w * 4)?,
            dirty: false,
        })
    }

    fn moe(&mut self, m: &MoeFfn, d: &Dims, ne: usize) -> Res<MoeW> {
        if !moe_ok(m) || m.experts.len() != ne || m.router.rows() != ne || m.router.cols() != d.h {
            return Err("qwen4 cuda: a router this path does not reproduce".into());
        }
        for e in &m.experts {
            expert_parts(e, d)?;
        }
        if self.range {
            for (name, f) in m.experts.iter().take(1).map(|e| ("expert0", e)).chain(m.shared.iter().map(|(f, _)| ("shared", f))) {
                self.wstat(&format!("{name}.gate"), &host_rows(&[&f.gate_proj]), d.h);
                self.wstat(&format!("{name}.up"), &host_rows(&[&f.up_proj]), d.h);
                self.wstat(&format!("{name}.down"), &host_rows(&[&f.down_proj]), d.inter);
            }
        }
        let mut ts = vec![&m.router];
        let (shared, gcol) = match &m.shared {
            Some((ffn, gate)) => {
                let parts = expert_parts(ffn, d)?;
                let mut raw = Vec::with_capacity(2 * d.sgu + d.sd);
                for p in parts {
                    raw.extend_from_slice(p);
                }
                let b = Buf::from_slice(&raw)?;
                self.held += b.bytes as u64;
                let gcol = match gate {
                    Some(g) if g.rows() == 1 && g.cols() == d.h => {
                        ts.push(g);
                        ne as i32
                    }
                    Some(_) => return Err("qwen4 cuda: unexpected shared-expert gate".into()),
                    None => -1,
                };
                (Some(b), gcol)
            }
            None => (None, -1),
        };
        let r = ts.iter().map(|t| t.rows()).sum();
        Ok(MoeW { router: self.f32buf(&host_rows(&ts))?, shared, gcol, r, topk: m.top_k.min(ne), ne })
    }
}

/// Time by stage (with `CMF_QWEN_PROF`): each lap waits for the stream.
struct Prof {
    t0: Instant,
    acc: [f64; STAGES.len()],
    /// Single-token calls since the last report.
    steps: usize,
}

const STAGES: [&str; 10] = ["io", "ple", "hc", "gdn", "qsa", "route", "admit", "experts", "shared", "head"];

impl Prof {
    /// Print the stage times gathered since the last report, and start over.
    fn report(&mut self, arena: &mut Arena, what: String) {
        let line = STAGES.iter().zip(&self.acc).map(|(name, ms)| format!("{name} {ms:.1}")).collect::<Vec<_>>().join(" ");
        eprintln!(
            "qwen4 cuda: {what}: {line} ms; arena +{} -{}, staged in {:.1} ms, {} units for {} picks",
            arena.admitted, arena.evicted, arena.stage_ms, arena.units, arena.pairs
        );
        self.acc = [0.0; STAGES.len()];
        self.steps = 0;
        arena.admitted = 0;
        arena.evicted = 0;
        arena.stage_ms = 0.0;
        arena.units = 0;
        arena.pairs = 0;
    }
}

/// What the MTP draft head takes from the host: its hybrid layer (QSA mixer,
/// its own routed experts and shared expert), the mixer hyper-connection that
/// feeds the shared LM head, the two input projections and their norms.
pub(crate) struct MtpParts<'a> {
    pub layer: &'a Layer,
    pub mixer: &'a GatedResidual,
    pub enorm: &'a [f32],
    pub hnorm: &'a [f32],
    pub fc_e: &'a QTensor,
    pub fc_h: &'a QTensor,
}

/// The draft head on the card. A cell pairs the main model's final hyper
/// state at position p with the token at p + 1 and drafts the token at p + 2;
/// its attention state (a zeroed QSA cache of its own) is written by the cells
/// alone.
struct MtpW {
    lw: LayerW,
    /// The layer's index in the expert arena's keys.
    li: usize,
    mixer: Hc,
    fc_e: DevMat,
    fc_h: DevMat,
    enorm: Buf,
    hnorm: Buf,
    /// The main model's final hyper state of the last position `[hc][hidden]`, the token count it was taken at.
    r_last: Buf,
    r_pos: Option<usize>,
    en: Buf,
    e: Buf,
    rn: Buf,
    hs: Buf,
}

/// Everything a layer step needs besides the layer's own state.
struct Core {
    c: &'static Ctx,
    q: &'static QFuncs,
    d: Dims,
    hi: i32,
    es: usize,
    gw: Gemm,
    ga: Gemm,
    n_cap: usize,
    s: Scratch,
    invf: Buf,
    prof: Option<Mutex<Prof>>,
    /// `CMF_QWEN_CUDA_RANGE`: report the range of the GEMM operands.
    range: bool,
    /// The small path is on (`CMF_QWEN_CUDA_SMALL=0` turns it off).
    small: bool,
    tune: Tune,
    /// `CMF_QWEN_ROUTE_TAIL`: picks lighter than this share of the picked
    /// weight are dropped from a decode step's routing (not the exact model).
    tail: f32,
    /// Rows of a window's tape.
    tape: usize,
    /// Rows of an expert sub-batch, experts of one, bytes of the attention score scratch and
    /// of the indexer's: smaller on a card that cannot spare them (`lean`).
    xr: usize,
    xb: usize,
    sbuf: usize,
    ds: usize,
    /// Decode steps and windows of a masked route (and the MTP head) build their expert units on
    /// the card and never wait for the host between the layers (`CMF_QWEN_CUDA_LEAN_ROUTE=0` off).
    /// 0: never, 1: always, 2: every other round (`CMF_QWEN_CUDA_LEAN_ROUTE=alt`: both ways in one process, for A/B timing).
    lean_mode: u8,
    lean_flip: std::sync::atomic::AtomicBool,
    /// A window of more than `SMALL_ROWS` rows runs its experts on the unit kernels too
    /// (`CMF_QWEN_CUDA_MID=0` gives it the batched GEMMs).
    mid: bool,
}

/// The device state of one sequence.
pub(crate) struct Dev {
    k: Core,
    layers: Vec<LayerW>,
    head: Hc,
    lm: DevMat,
    arena: Arena,
    /// Indexer masks `[rows][mwords]`, grown on demand.
    maskb: Buf,
    held: u64,
    /// Keeps the mapped model (the pinned pages) alive until the arena is dropped.
    _keep: Option<Arc<cortiq_core::CmfModel>>,
    /// Tokens the device state holds.
    pub(crate) pos: usize,
    /// The window (position, rows) a `spec` pass left to settle.
    pending: Option<(usize, usize)>,
    /// Host time (ms) the last pass took to queue its layers, before it waited for the card.
    pub(crate) launch_ms: f64,
    /// The MTP draft head (`CMF_QWEN_MTP=0` leaves it out).
    mtp: Option<MtpW>,
    /// Keep the stage rows of each chunk's last token (check mode).
    pub(crate) want_trace: bool,
    pub(crate) trace: Vec<(&'static str, usize, Vec<f32>)>,
}

impl Drop for Dev {
    fn drop(&mut self) {
        let _ = self.k.c.current();
        let _ = self.k.c.sync();
        self.k.c.resident.fetch_sub(self.held, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Dev {
    /// Place the model on the card. An `Err` leaves the model on the host path.
    pub(crate) fn new(globals: &Globals, layers: &[Layer], cfg: &Cfg, inv_freq: &[f32], n_cap: usize, mtp: Option<&MtpParts>) -> Res<Self> {
        let c = ctx().ok_or_else(|| "qwen4 cuda: no CUDA device".to_string())?;
        // one user of the card at a time: the generic entry points take the same lock
        let _st = c.st.lock().unwrap();
        c.current()?;
        let q = qf(c)?;
        let (gw, ga) = prec();
        let ga = if ga == Gemm::Half { Gemm::Tf32 } else { ga };
        let half = gw == Gemm::Half;
        let es = if half { 2 } else { 4 };
        let g = &cfg.gdn;
        let first = layers.first().ok_or_else(|| "qwen4 cuda: no layers".to_string())?;
        let ne = first.moe.experts.len();
        let inter = first.moe.experts.first().map_or(0, |e| e.gate_proj.rows());
        let h = cfg.hidden;
        if ne == 0 || inter == 0 || !h.is_multiple_of(32) || !inter.is_multiple_of(32) || cfg.compress_ratio == 0 {
            return Err("qwen4 cuda: unexpected expert shapes".into());
        }
        let d = Dims {
            h,
            hc: cfg.hc,
            w: h * cfg.hc,
            eps: cfg.eps,
            gdn_eps: g.rms_eps,
            nv: g.num_v_heads,
            nk: g.num_k_heads,
            dk: g.key_head_dim,
            dv: g.value_head_dim,
            kk: g.conv_kernel,
            c_dim: g.conv_dim(),
            nh: cfg.n_heads,
            kvh: cfg.n_kv_heads,
            hd: cfg.head_dim,
            groups: cfg.n_heads / cfg.n_kv_heads.max(1),
            ih: cfg.index_heads,
            idim: cfg.index_dim,
            keep: cfg.index_budget / cfg.compress_ratio,
            cr: cfg.compress_ratio,
            rot: cfg.rotary_dim,
            nfreq: inv_freq.len(),
            ne,
            inter,
            sgu: q4_bytes(inter, h),
            sd: q4_bytes(h, inter),
            vocab: globals.lm_head.rows(),
            ple_k: cfg.ple_kernel.max(1),
            ple_dil: cfg.ple_dilation.max(1),
        };
        let topk = layers.iter().map(|l| l.moe.top_k.min(ne)).max().unwrap_or(1).max(1);
        let refuse = [
            (d.dk != 128, "GDN key head dim is not 128"),
            (d.dv > 256 || d.hd > 256 || d.idim > 256, "a head wider than 256"),
            (!g.output_gate_sigmoid, "GDN output gate is not a sigmoid"),
            (cfg.index_kv_heads != 1, "more than one indexer key head"),
            (d.kvh == 0 || !d.nh.is_multiple_of(d.kvh.max(1)), "query heads do not divide by key heads"),
            (ne > 1024 || topk > 32, "more than 1024 experts or top-k above 32"),
            (n_cap == 0 || n_cap > MAX_CHUNK, "chunk size out of range"),
            (d.c_dim != 2 * d.nk * d.dk + d.nv * d.dv, "GDN conv width"),
            (d.keep == 0 || d.ih == 0 || d.idim == 0, "indexer budget"),
            (globals.lm_head.cols() != h, "LM head width"),
            (inv_freq.is_empty(), "no rotary frequencies"),
        ];
        if let Some((_, why)) = refuse.iter().find(|(bad, _)| *bad) {
            return Err(format!("qwen4 cuda: {why}"));
        }

        let range = std::env::var_os("CMF_QWEN_CUDA_RANGE").is_some();
        let tape = tape_rows(n_cap);
        let mut b = Build { c, q, half, held: 0, wq_elems: 16, range, li: 0, tape };
        let mut lws = Vec::with_capacity(layers.len());
        let (mut r_max, mut lr_max, mut pin_max, mut hist_max, mut rl_max) = (1usize, 1usize, 16usize, 1usize, 1usize);
        for (li, l) in layers.iter().enumerate() {
            b.li = li;
            let attn = b.hc(&l.attn_hc, &d, true)?;
            let mlp = b.hc(&l.mlp_hc, &d, true)?;
            let mix = match &l.mixer {
                Mixer::Gdn(w) => Mix::Gdn(b.gdn(w, &d)?),
                Mixer::Qsa(w) => Mix::Qsa(Box::new(b.qsa(w, &d)?)),
            };
            let ple = match &l.ple {
                Some(p) => Some(b.ple(p, &d)?),
                None => None,
            };
            let moe = b.moe(&l.moe, &d, d.ne)?;
            r_max = r_max.max(attn.r).max(mlp.r);
            lr_max = lr_max.max(attn.lr).max(mlp.lr);
            rl_max = rl_max.max(moe.r);
            if let Some(p) = &ple {
                pin_max = pin_max.max(p.pin);
                hist_max = hist_max.max((d.ple_k - 1) * d.ple_dil * d.w);
            }
            lws.push(LayerW { attn, mlp, mix, ple, moe });
        }
        let head = b.hc(&globals.head_hc, &d, false)?;
        r_max = r_max.max(head.r);
        lr_max = lr_max.max(head.lr);
        let lm = {
            let t = &globals.lm_head;
            let kind = if t.mapped_q8_2f().is_some() {
                7u8
            } else if t.mapped_q4tp().is_some() {
                6u8
            } else {
                return Err("qwen4 cuda: the LM head is neither q8_2f nor q4tp".into());
            };
            let bytes = t.mapped_bytes().ok_or_else(|| "qwen4 cuda: the LM head has no file bytes".to_string())?;
            let m = dev_mat(kind, bytes, t.rows(), t.cols())?;
            b.held += m.held;
            m
        };
        let invf = b.f32buf(inv_freq)?;
        // the MTP draft head, if the file carries one that fits this path
        let mtp_w = match mtp {
            Some(p) => match b.mtp(p, &d, layers.len(), globals) {
                Ok(m) => {
                    r_max = r_max.max(m.lw.attn.r).max(m.lw.mlp.r).max(m.mixer.r);
                    lr_max = lr_max.max(m.lw.attn.lr).max(m.lw.mlp.lr).max(m.mixer.lr);
                    rl_max = rl_max.max(m.lw.moe.r);
                    Some(m)
                }
                Err(e) => {
                    eprintln!("qwen4 cuda: the MTP head stays off: {e}");
                    None
                }
            },
            None => None,
        };
        let topk = topk.max(mtp_w.as_ref().map_or(1, |m| m.lw.moe.topk));
        // the lean path's units: two of four rows for every expert of the widest layer, and the shared expert's
        let head_ne = mtp_w.as_ref().map_or(0, |m| m.lw.moe.ne);
        let lean_units = SMALL_ROWS * (topk + 1) + 2;

        // a card that cannot spare the prompt scratch: sub-batches, score buffers and chunks shrink
        // (`CMF_QWEN_CUDA_LEAN=1` forces, `=0` forbids)
        let lean = match std::env::var("CMF_QWEN_CUDA_LEAN").ok().as_deref().map(str::trim) {
            Some("1") => true,
            Some("0") => false,
            _ => lean_card(),
        };
        let (xr, xb, sbuf_b, ds_b) = if lean { (XR / 4, XB / 4, SBUF / 4, DS_BUDGET / 4) } else { (XR, XB, SBUF, DS_BUDGET) };
        let n = n_cap;
        let ow = (d.nv * d.dv).max(d.nh * d.hd);
        let s1w = d.c_dim.max(d.nh * 2 * d.hd).max(d.w);
        let mixw = d.w.max(d.c_dim).max(d.ih * d.idim);
        let tune = Tune::from_env();
        let s = Scratch {
            emb: b.buf(n * d.h * 4)?,
            hyper: b.buf(n * d.w * 4)?,
            normed: b.buf(n * d.w * 4)?,
            normed_h: b.buf(n * d.w * es)?,
            low: b.buf(n * r_max * 4)?,
            lowh: b.buf((n * lr_max * es).max(n.min(SMALL_ROWS) * lr_max * 4))?,
            hcp: b.buf(d.w.div_ceil(tune.cw.min(d.w)) * n.min(SMALL_ROWS) * r_max * 4)?,
            iw: b.buf(n * d.hc * 4)?,
            mix: b.buf(n * mixw * 4)?,
            s1: b.buf(n * s1w * 4)?,
            folded: b.buf(n * d.h * 4)?,
            folded_h: b.buf(n * d.h * es)?,
            blk: b.buf(n * d.h * 4)?,
            z: b.buf(n * ow * 4)?,
            qo: b.buf(n * ow * 4)?,
            qq: b.buf(n * ow * 4)?,
            ofh: b.buf((n * ow * es).max(n.min(SMALL_ROWS) * ow * 4))?,
            ab: b.buf(n * 2 * d.nv * 4)?,
            gb: b.buf(n * 2 * d.nv * 4)?,
            kraw: b.buf(n * d.kvh * d.hd * 4)?,
            iq: b.buf(n * d.ih * d.idim * 4)?,
            sbuf: b.buf(sbuf_b)?,
            sc: b.buf(ds_b / d.ih + 4096)?,
            ple_in: b.buf(n * pin_max * 4)?,
            ple_h: b.buf(n * pin_max * es)?,
            logits: b.buf(n * rl_max * 4)?,
            picks: b.buf(n * topk * 4)?,
            wts: b.buf(n * topk * 4)?,
            rowtok: b.buf(xr * 4)?,
            tokrow: b.buf(n * topk * 4)?,
            slots: b.buf((xb * 8).max(tape.max(SMALL_ROWS) * (topk + 1) * (UNIT_WORDS + 1) * 4).max((lean_units * UNIT_WORDS + SMALL_ROWS * (topk + 1)) * 4))?,
            xg: b.buf(xr * d.h * es)?,
            g: b.buf((xr * d.inter * es).max(n.max(tape.max(SMALL_ROWS) * (topk + 1) * UNIT_ROWS).max(lean_units * UNIT_ROWS) * d.inter * 4))?,
            u: b.buf((xr * d.inter * es).max(n * d.inter * 4))?,
            ah: b.buf(xr.max(n) * d.inter * es)?,
            d: b.buf((xr * d.h * es).max(n.max(tape.max(SMALL_ROWS) * (topk + 1) * UNIT_ROWS).max(lean_units * UNIT_ROWS) * d.h * 4))?,
            wg: b.buf(xb * d.inter * d.h * es)?,
            wu: b.buf(xb * d.inter * d.h * es)?,
            wd: b.buf(xb * d.inter * d.h * es)?,
            wq: b.buf(b.wq_elems * es)?,
            out: b.buf(SMALL_ROWS * d.vocab * 4)?,
            ids: b.buf(n.max(SMALL_ROWS) * 4)?,
        };
        let maskb = b.buf(n * 64 * 4)?;

        // The arena takes what the budget leaves after the resident weights,
        // the scratch and a reserve for the attention state of a long prompt.
        let slot = 2 * d.sgu + d.sd;
        let reserve = 1536u64 << 20;
        let taken = c.resident.load(std::sync::atomic::Ordering::Relaxed) + b.held + reserve;
        let mut slots = (vram_budget().saturating_sub(taken) / slot as u64) as usize;
        if let Some(mb) = std::env::var("CMF_QWEN_CUDA_ARENA_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            slots = slots.min((mb << 20) / slot);
        }
        slots = slots.min(ne * layers.len());
        if slots < ne.max(256).min(ne * layers.len()) {
            return Err(format!("qwen4 cuda: room for {slots} experts only, a layer has {ne}"));
        }
        // one layer's experts of staging (`CMF_QWEN_CUDA_STAGE=0`: the driver's own)
        let stage_cap = ne.min(slots);
        let stage = if std::env::var("CMF_QWEN_CUDA_STAGE").is_ok_and(|v| v.trim() == "0") { None } else { Pinned::new(stage_cap * slot) };
        let tbl_rows = layers.len() + usize::from(mtp_w.is_some());
        let tbl_stride = ne.max(head_ne);
        let mut tbl_ne = vec![ne; layers.len()];
        if mtp_w.is_some() {
            tbl_ne.push(head_ne);
        }
        let tbl = b.buf(tbl_rows * tbl_stride * 8)?;
        let arena = Arena {
            buf: b.buf(slots * slot)?,
            slot,
            n: slots,
            used: 0,
            key: vec![u32::MAX; slots],
            last: vec![0; slots],
            map: HashMap::new(),
            nl: (layers.len() + usize::from(mtp_w.is_some())) as u64,
            pass: 0,
            stage,
            stage_cap,
            pending: false,
            pool: CopyPool::new(),
            spare: Vec::new(),
            admitted: 0,
            evicted: 0,
            stage_ms: 0.0,
            units: 0,
            pairs: 0,
            pins: None,
            sticky_from: u32::MAX,
            prot: vec![false; slots],
            hist: vec![0; layers.len() * ne],
            hist_ne: ne,
            hist_on: true,
            masked: false,
            tbl,
            tbl_stride,
            tbl_ne,
            tbl_held: vec![0; tbl_rows],
            tbl_epoch: 0,
            epoch: 1,
        };
        // the pinned expert pages (`Dev::prefill`) need the mapped model to outlive them
        let keep = layers.first().and_then(|l| l.moe.experts.first()).and_then(|f| f.gate_proj.model_arc());
        c.sync()?;
        c.resident.fetch_add(b.held, std::sync::atomic::Ordering::Relaxed);
        if super::debug() || std::env::var_os("CMF_QWEN_PROF").is_some() {
            eprintln!(
                "qwen4 cuda: {} layers on {}, {:.1} GiB held, arena {} of {} experts, chunk {}, weights {}",
                layers.len(),
                c.name,
                b.held as f64 / (1u64 << 30) as f64,
                slots,
                ne * layers.len(),
                n,
                if half { "f16" } else { "f32" },
            );
        }
        let prof = std::env::var_os("CMF_QWEN_PROF").map(|_| Mutex::new(Prof { t0: Instant::now(), acc: [0.0; STAGES.len()], steps: 0 }));
        Ok(Self {
            k: Core {
                c,
                q,
                d,
                hi: half as i32,
                es,
                gw,
                ga,
                n_cap: n,
                s,
                invf,
                prof,
                range,
                small: std::env::var("CMF_QWEN_CUDA_SMALL").map(|v| v != "0").unwrap_or(true),
                tune,
                lean_mode: match std::env::var("CMF_QWEN_CUDA_LEAN_ROUTE").ok().as_deref().map(str::trim) {
                    Some("0") => 0,
                    Some("alt") => 2,
                    _ => 1,
                },
                lean_flip: std::sync::atomic::AtomicBool::new(false),
                xr,
                xb,
                sbuf: sbuf_b,
                ds: ds_b,
                mid: !std::env::var("CMF_QWEN_CUDA_MID").is_ok_and(|v| v.trim() == "0"),
                tail: std::env::var("CMF_QWEN_ROUTE_TAIL").ok().and_then(|v| v.trim().parse::<f32>().ok()).map_or(0.0, |t| t.clamp(0.0, 0.5)),
                tape,
            },
            layers: lws,
            head,
            lm,
            arena,
            maskb,
            held: b.held,
            _keep: keep,
            pos: 0,
            pending: None,
            launch_ms: 0.0,
            mtp: mtp_w,
            want_trace: false,
            trace: Vec::new(),
        })
    }

    /// Largest chunk a call takes.
    pub(crate) fn chunk(&self) -> usize {
        self.k.n_cap
    }

    /// Rows a window can take back in part (the tape).
    pub(crate) fn tape(&self) -> usize {
        self.k.tape
    }

    /// How often each (layer, expert) was picked since the sequence began (`[layer][expert]`).
    pub(crate) fn route_hist(&self) -> &[u32] {
        &self.arena.hist
    }

    /// Whether the lean path is on for the round that has just run (the other way next round with `alt`).
    pub(crate) fn lean_was(&self) -> bool {
        match self.k.lean_mode {
            0 => false,
            1 => true,
            _ => !self.k.lean_flip.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// `CMF_QWEN_NCU=<round>`: the profiler's range around one speculation round (for
    /// `ncu --profile-from-start off`).
    pub(crate) fn profiler_range(&self, on: bool) {
        let _ = self.k.c.sync();
        // SAFETY: plain driver calls; they only mark a range for an attached profiler.
        unsafe {
            if on {
                let _ = cs::cuProfilerStart();
            } else {
                let _ = cs::cuProfilerStop();
            }
        }
    }

    /// The MTP head is on the card.
    pub(crate) fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// The prompt has run: refill the arena with what this request needs. Per layer
    /// the experts rank by `mix` of the prompt's own routing share and `1 - mix` of
    /// `pack` (a per-layer share, `[layer][expert]`, of the task's routing); the
    /// layers take the arena in turns by rank; what is not in it yet comes in, at most
    /// `max_swap` of them, for the experts the arena holds least worth. Returns (out, in).
    pub(crate) fn retarget(&mut self, layers: &[Layer], pack: Option<&[f32]>, mix: f32, max_swap: usize) -> Res<(u64, u64)> {
        let Self { k, arena, mtp, .. } = self;
        let (d, c) = (&k.d, k.c);
        let (nl, ne) = (layers.len(), d.ne);
        if arena.hist.len() != nl * ne || !arena.hist_on {
            return Ok((0, 0));
        }
        arena.hist_on = false;
        if arena.hist.iter().all(|&h| h == 0) {
            return Ok((0, 0));
        }
        let sticky = mtp.as_ref().map_or(0, |m| layers.get(0).map_or(0, |_| m.lw.moe.ne));
        let cap = arena.n.saturating_sub(sticky);
        let per_layer: Vec<Vec<(f32, usize)>> = (0..nl)
            .map(|l| {
                let h = &arena.hist[l * ne..(l + 1) * ne];
                let hs = h.iter().map(|&x| x as f32).sum::<f32>().max(1.0);
                let mut v: Vec<(f32, usize)> = (0..ne)
                    .map(|e| (mix * h[e] as f32 / hs + (1.0 - mix) * pack.map_or(0.0, |p| p[l * ne + e]), e))
                    .filter(|&(sc, _)| sc > 0.0)
                    .collect();
                v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                v
            })
            .collect();
        // the layers take the arena in turns, by rank
        let mut ranked: Vec<(usize, usize)> = Vec::with_capacity(cap);
        'fill: for r in 0..ne {
            for (l, v) in per_layer.iter().enumerate() {
                if let Some(&(_, e)) = v.get(r) {
                    if ranked.len() >= cap {
                        break 'fill;
                    }
                    ranked.push((l, e));
                }
            }
        }
        let _st = c.st.lock().unwrap();
        c.current()?;
        arena.pass += 1;
        let pass = arena.pass;
        let (adm0, ev0) = (arena.admitted, arena.evicted);
        let mut missing: Vec<Vec<usize>> = vec![Vec::new(); nl];
        let mut swaps = 0usize;
        for &(l, e) in &ranked {
            match arena.map.get(&(((l as u32) << 16) | e as u32)) {
                Some(&s) => arena.last[s as usize] = pass,
                None if swaps < max_swap => {
                    missing[l].push(e);
                    swaps += 1;
                }
                None => {}
            }
        }
        let mut slot_of = vec![0u64; ne];
        for (li, order) in missing.iter().enumerate() {
            if order.is_empty() {
                continue;
            }
            arena.admit_layer(li, order, &layers[li].moe.experts, d, c, &mut slot_of)?;
            c.sync()?;
            arena.pending = false;
        }
        arena.prot.iter_mut().for_each(|p| *p = false);
        for &(l, e) in &ranked {
            if let Some(&s) = arena.map.get(&(((l as u32) << 16) | e as u32)) {
                arena.prot[s as usize] = true;
            }
        }
        // every layer must hold more than its top-k for a masked route
        if std::env::var("CMF_QWEN_MASK").as_deref() == Ok("1") {
            arena.masked = true;
        }
        let out = (arena.evicted - ev0, arena.admitted - adm0);
        arena.admitted = 0;
        arena.evicted = 0;
        arena.stage_ms = 0.0;
        Ok(out)
    }

    /// Put all the head's experts into the arena and keep them there: a draft
    /// cell then never waits for an upload. `head` is the head's host layer.
    pub(crate) fn warm_mtp(&mut self, head: &Layer) -> Res<usize> {
        let Self { k, arena, mtp, .. } = self;
        let Some(m) = mtp.as_ref() else { return Ok(0) };
        let (d, c) = (&k.d, k.c);
        let _st = c.st.lock().unwrap();
        c.current()?;
        let ne = head.moe.experts.len();
        let order: Vec<usize> = (0..ne).collect();
        let mut slot_of = vec![0u64; ne];
        arena.admit_layer(m.li, &order, &head.moe.experts, d, c, &mut slot_of)?;
        c.sync()?;
        arena.pending = false;
        arena.sticky_from = m.li as u32;
        arena.admitted = 0;
        arena.evicted = 0;
        arena.stage_ms = 0.0;
        Ok(ne)
    }

    /// Leave the MTP head out from now on.
    pub(crate) fn mtp_off(&mut self) {
        self.mtp = None;
    }

    /// Up to `kmax` greedy drafts of the MTP head for the tokens after `tok0`
    /// (the token at `pos`): cell `j` pairs the hyper state of position
    /// `pos - 1 + j` (the main model's for the first, the previous cell's after)
    /// with the token it has just been given. `head` is the head's host layer
    /// (its experts' file bytes). Ok(false): no head, or its state is not at `pos`.
    pub(crate) fn draft(&mut self, head: &Layer, embed: &dyn Fn(u32, &mut [f32]), tok0: u32, pos: usize, kmax: usize, out: &mut Vec<u32>) -> Res<bool> {
        let Self { k, arena, maskb, held, lm, mtp, pos: dpos, .. } = self;
        let k = &*k;
        let Some(m) = mtp.as_mut() else { return Ok(false) };
        if pos == 0 || *dpos != pos || m.r_pos != Some(pos) {
            return Ok(false);
        }
        let (d, s, c, q) = (&k.d, &k.s, k.c, k.q);
        let _st = c.st.lock().unwrap();
        c.current()?;
        arena.pass += 1;
        out.clear();
        let (mut tok, mut emb) = (tok0, vec![0.0f32; d.h]);
        for j in 0..kmax {
            let cell = pos - 1 + j;
            emb.iter_mut().for_each(|v| *v = 0.0);
            embed(tok, &mut emb);
            k.put(s.emb.ptr, &emb)?;
            let src = if j == 0 { m.r_last.ptr } else { s.hyper.ptr };
            // e = fc_e(rms(embed) (1 + w_e)); h = fc_h per stream of rms(R) (1 + w_h); R' = h + e
            go!(c, q.grms, (1u32, 1), 256; s.emb.ptr, m.enorm.ptr, m.en.ptr, 0u64, d.h as i32, 1i32, d.eps, 0i32);
            k.mvb(&[SOp::q8(&m.fc_e, m.e.ptr)], m.en.ptr, d.h, d.h, 1)?;
            go!(c, q.grms, (1u32, 1), 256; src, m.hnorm.ptr, m.rn.ptr, 0u64, d.w as i32, 1i32, d.eps, 0i32);
            k.mvb(&[SOp::q8(&m.fc_h, m.hs.ptr)], m.rn.ptr, d.h, d.h, d.hc)?;
            go!(c, q.mtp_fuse, (blocks(d.w), 1), 256; m.hs.ptr, m.e.ptr, s.hyper.ptr, d.h as i32, d.hc as i32);
            // the layer, one row at the cell's position
            k.hc_mix(&m.lw.attn, s.hyper.ptr, 1)?;
            if let Mix::Qsa(a) = &mut m.lw.mix {
                let grown = k.qsa(a, maskb, cell, 1)?;
                *held += grown;
                c.resident.fetch_add(grown, std::sync::atomic::Ordering::Relaxed);
            }
            k.inject(1)?;
            k.hc_mix(&m.lw.mlp, s.hyper.ptr, 1)?;
            k.moe(&m.lw.moe, head, m.li, arena, 1, false)?;
            k.inject(1)?;
            let mut ids: Vec<u32> = Vec::new();
            k.head_rows(&m.mixer, lm, 1, &mut ids)?;
            tok = ids.first().copied().unwrap_or(0);
            out.push(tok);
        }
        Ok(true)
    }

    /// Fill the arena with the experts a profile ranks hottest, `(layer,
    /// expert)`: the first tokens meet a warm arena. Returns how many went in.
    pub(crate) fn prefill(&mut self, layers: &[Layer], ranked: &[(usize, usize)]) -> Res<usize> {
        let Self { k, arena, .. } = self;
        let (d, c) = (&k.d, k.c);
        if layers.len() != self.layers.len() {
            return Err("qwen4 cuda: prefill with other layers".into());
        }
        let _st = c.st.lock().unwrap();
        c.current()?;
        let mut by_layer: Vec<Vec<usize>> = vec![Vec::new(); layers.len()];
        for &(l, e) in ranked.iter().take(arena.n) {
            if l < layers.len() && e < d.ne {
                by_layer[l].push(e);
            }
        }
        let mut slot_of = vec![0u64; d.ne];
        let mut n = 0;
        for (li, order) in by_layer.iter().enumerate() {
            if order.is_empty() {
                continue;
            }
            arena.admit_layer(li, order, &layers[li].moe.experts, d, c, &mut slot_of)?;
            c.sync()?;
            arena.pending = false;
            n += order.len();
        }
        arena.admitted = 0;
        arena.evicted = 0;
        arena.stage_ms = 0.0;
        // `CMF_QWEN_CUDA_REG=1`: the experts outside the prefilled arena (the ones it keeps
        // missing) get their file pages pinned, up to `CMF_QWEN_CUDA_REG_MB` (default: all of them)
        if std::env::var("CMF_QWEN_CUDA_REG").is_ok_and(|v| v.trim() == "1") && arena.stage.is_some() && arena.pins.is_none() {
            let budget = std::env::var("CMF_QWEN_CUDA_REG_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(usize::MAX, |mb| mb << 20);
            let mut sel: Vec<Vec<bool>> = vec![vec![true; d.ne]; layers.len()];
            // `CMF_QWEN_CUDA_REG_ALL=1`: the arena's own experts too (a host with the page cache for it)
            if !std::env::var("CMF_QWEN_CUDA_REG_ALL").is_ok_and(|v| v.trim() == "1") {
                for &(l, e) in ranked.iter().take(arena.n) {
                    if l < sel.len() && e < d.ne {
                        sel[l][e] = false;
                    }
                }
            }
            let (mut ranges, mut total) = (Vec::with_capacity(layers.len()), 0usize);
            for (li, l) in layers.iter().enumerate() {
                // page runs of the layer's chosen experts, merged where they touch
                let mut runs: Vec<(usize, usize)> = Vec::new();
                for (e, f) in l.moe.experts.iter().enumerate() {
                    if !sel[li][e] {
                        continue;
                    }
                    let Ok(parts) = expert_parts(f, d) else {
                        sel[li][e] = false;
                        continue;
                    };
                    if total >= budget {
                        sel[li][e] = false;
                        continue;
                    }
                    for p in parts {
                        let a = p.as_ptr() as usize;
                        runs.push((a & !4095, (a + p.len() + 4095) & !4095));
                    }
                    total += slot_bytes(d);
                }
                runs.sort_unstable();
                let mut merged: Vec<(usize, usize)> = Vec::new();
                for (a, b) in runs {
                    match merged.last_mut() {
                        Some(last) if a <= last.1 => last.1 = last.1.max(b),
                        _ => merged.push((a, b)),
                    }
                }
                ranges.push(merged.into_iter().map(|(a, b)| (a, b - a)).collect::<Vec<_>>());
            }
            arena.pins = Some(Pins::start(c, sel, ranges));
        }
        Ok(n)
    }

    /// Forget the sequence: recurrent state to zero, attention state empty.
    pub(crate) fn reset(&mut self) -> Res<()> {
        let k = &self.k;
        let _st = k.c.st.lock().unwrap();
        k.c.current()?;
        self.pending = None;
        self.arena.hist.iter_mut().for_each(|c| *c = 0);
        self.arena.hist_on = true;
        self.arena.masked = false;
        self.arena.prot.iter_mut().for_each(|p| *p = false);
        if let Some(m) = &mut self.mtp {
            m.r_pos = None;
            if let Mix::Qsa(a) = &m.lw.mix {
                for (b, bytes) in [(&a.keys, a.keys.bytes), (&a.vals, a.vals.bytes), (&a.ikraw, a.ikraw.bytes), (&a.bkey, a.bkey.bytes)] {
                    go!(k.c, k.q.zero, (blocks(bytes / 4), 1), 256; b.ptr, (bytes / 4) as u64);
                }
            }
        }
        for l in &mut self.layers {
            if let Mix::Gdn(g) = &mut l.mix {
                g.dirty = false;
                let (ring, st) = ((k.d.kk - 1).max(1) * k.d.c_dim, k.d.nv * k.d.dk * k.d.dv);
                go!(k.c, k.q.zero, (blocks(ring), 1), 256; g.ring.ptr, ring as u64);
                go!(k.c, k.q.zero, (blocks(st), 1), 256; g.s.ptr, st as u64);
            }
            if let Some(p) = &mut l.ple {
                p.dirty = false;
                let hist = ((k.d.ple_k - 1) * k.d.ple_dil).max(1) * k.d.w;
                go!(k.c, k.q.zero, (blocks(hist), 1), 256; p.hist.ptr, hist as u64);
            }
        }
        self.pos = 0;
        Ok(())
    }
}

/// A card whose budget leaves little beyond the weights: below 26 GiB.
pub(crate) fn lean_card() -> bool {
    vram_budget() < (26u64 << 30)
}

/// Rows of a window's tape: eight (a verify round), or `CMF_QWEN_CUDA_TAPE`
/// (up to 256; 256 when the copy frames are on).
fn tape_rows(n_cap: usize) -> usize {
    let frames = std::env::var("CMF_QWEN_COPY_FRAME").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
    let want = std::env::var("CMF_QWEN_CUDA_TAPE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(if frames { 256 } else { SMALL_ROWS });
    want.clamp(SMALL_ROWS, 256).min(n_cap).max(1)
}

impl Core {
    /// `out [n][rows]` (f32) = `x [n][cols]` times the transpose of the
    /// row-major `w [rows][cols]`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_w(&self, w: u64, rows: usize, cols: usize, x: u64, n: usize, out: u64, mode: Gemm) -> Res<()> {
        unsafe { self.c.gemm(true, false, rows, n, cols, 1.0, w, cols, x, cols, 0.0, out, rows, mode) }
    }

    /// The projection as a dense plane in the GEMM operand type. A q8 weight
    /// is expanded into the one scratch plane (valid until the next q8 one).
    fn plane(&self, w: &DW, rows: usize, cols: usize) -> Res<u64> {
        match w {
            DW::Plane(b) => Ok(b.ptr),
            DW::Q8(m) => {
                go!(self.c, self.q.expand_q8, (blocks(rows * cols / 4), 1), 256;
                    m.w.ptr, m.aux0(), m.aux1(), self.s.wq.ptr, rows as i32, cols as i32, self.hi);
                Ok(self.s.wq.ptr)
            }
        }
    }

    /// Close stage `k` of the profile (waits for the stream).
    fn lap(&self, k: usize) -> Res<()> {
        if let Some(p) = &self.prof {
            self.c.sync()?;
            let mut p = p.lock().unwrap();
            p.acc[k] += p.t0.elapsed().as_secs_f64() * 1e3;
            p.t0 = Instant::now();
        }
        Ok(())
    }

    fn grab(&self, ptr: u64, elems: usize) -> Res<Vec<f32>> {
        let mut v = vec![0.0f32; elems];
        unsafe { cu::memcpy_dtoh_async(&mut v, ptr, self.c.stream) }.map_err(derr)?;
        self.c.sync()?;
        Ok(v)
    }

    /// The range report of a GEMM operand buffer (debug).
    fn range(&self, what: &str, li: usize, ptr: u64, elems: usize) -> Res<()> {
        if !self.range || elems == 0 {
            return Ok(());
        }
        let v: Vec<f32> = if self.hi != 0 {
            let mut h = vec![0u16; elems];
            unsafe { cu::memcpy_dtoh_async(&mut h, ptr, self.c.stream) }.map_err(derr)?;
            self.c.sync()?;
            h.iter().map(|&x| h2f(x)).collect()
        } else {
            self.grab(ptr, elems)?
        };
        let (mut max, mut sat, mut sub, mut ss) = (0f32, 0usize, 0usize, 0f64);
        for &x in &v {
            let a = x.abs();
            max = max.max(a);
            sat += (a >= 65504.0) as usize;
            sub += (a > 0.0 && a < F16_MIN) as usize;
            ss += (a as f64) * (a as f64);
        }
        eprintln!(
            "qwen4 cuda range L{li} x {what}: max {max:.3e} rms {:.3e}, saturated {sat}, under the f16 normals {sub} of {elems}",
            (ss / elems as f64).sqrt()
        );
        Ok(())
    }

    /// Upload in stream order: a pageable copy waits for the queued kernels
    /// and is staged before the call returns.
    fn put<T: Copy>(&self, dst: u64, src: &[T]) -> Res<()> {
        unsafe { cu::memcpy_htod_async(dst, src, self.c.stream) }.map_err(derr)
    }

    /// Whether `n` token rows go the small path.
    fn small(&self, n: usize) -> bool {
        self.small && n <= SMALL_ROWS
    }

    /// The small path's matvecs of one input: `ops[j].y [n][rows_j]` =
    /// `W_j x`, `x [n]` rows `xstride` values apart and `cols` wide. Up to
    /// five jobs a launch; the rows are split by what a block can stage.
    fn mvb(&self, ops: &[SOp], x: u64, xstride: usize, cols: usize, n: usize) -> Res<()> {
        let sm = &self.q.sm;
        let per = 16 * ((cols >> 4) | 1) * 4;
        let cap = (sm.shared_max / per).min(SMALL_ROWS);
        if cap == 0 || ops.is_empty() || ops.len() > 5 {
            return Err(format!("qwen4 cuda: a small matvec of {} jobs over {cols} columns", ops.len()));
        }
        let mut jobs = [MvJob { w: 0, aux0: 0, aux1: 0, y: 0, res: 0, rows: 0, kind: 0, blk0: 0, rpb: 1 }; 5];
        let mut t0 = 0;
        while t0 < n {
            let nb = (n - t0).min(cap);
            let mut blk = 0u32;
            for (j, o) in jobs.iter_mut().zip(ops) {
                let rpb = rows_per_block(o.rows);
                *j = MvJob {
                    w: o.w,
                    aux0: o.aux0,
                    aux1: o.aux1,
                    y: o.y + (t0 * o.rows * 4) as u64,
                    res: 0,
                    rows: o.rows as i32,
                    kind: o.kind,
                    blk0: blk as i32,
                    rpb: rpb as i32,
                };
                blk += o.rows.div_ceil(rpb) as u32;
            }
            let (j0, j1, j2, j3, j4) = (jobs[0], jobs[1], jobs[2], jobs[3], jobs[4]);
            let xp = x + (t0 * xstride * 4) as u64;
            unsafe {
                self.c.launch(
                    sm.mvb[nb - 1],
                    (blk, 1),
                    256,
                    nb * per,
                    kargs!(j0, j1, j2, j3, j4, ops.len() as i32, xp, xstride as i32, cols as i32),
                )
            }?;
            t0 += nb;
        }
        Ok(())
    }

    /// The hyper-connection fold of `n` tokens: `folded` (and its GEMM-typed
    /// copy), the inject weights in `iw`.
    fn hc_mix(&self, hc: &Hc, hyper: u64, n: usize) -> Res<()> {
        let (d, s) = (&self.d, &self.s);
        go!(self.c, self.q.grms, ((n * d.hc) as u32, 1), 256;
            hyper, hc.norm.ptr, s.normed.ptr, s.normed_h.ptr, d.h as i32, d.hc as i32, d.eps, self.hi);
        if self.small(n) {
            // two launches, the low-rank values kept in f32: the down rows by
            // column chunks, then their sums and the fold
            let (sm, t) = (&self.q.sm, &self.tune);
            let per = if self.hi != 0 { 8 } else { 4 };
            let cw = t.cw.min(d.w);
            let nch = d.w.div_ceil(cw);
            let (dsh, ush) = (n * per * ((cw / per) | 1) * 4, n * per * ((hc.lr / per) | 1) * 4);
            let fits = d.w.is_multiple_of(per)
                && cw.is_multiple_of(per)
                && hc.lr.is_multiple_of(per)
                && hc.r == hc.lr + d.hc
                && dsh.max(ush) <= sm.shared_max
                && nch * n * hc.r * 4 <= s.hcp.bytes;
            if fits {
                unsafe {
                    self.c.launch(
                        sm.hc_di[n - 1],
                        (hc.r.div_ceil(t.di_t / 32) as u32, nch as u32),
                        t.di_t as u32,
                        dsh,
                        kargs!(hc.di.ptr, s.normed.ptr, s.hcp.ptr, hc.r as i32, d.w as i32, cw as i32, self.hi),
                    )
                }?;
                unsafe {
                    self.c.launch(
                        sm.hc_up[n - 1],
                        (d.h.div_ceil(t.up_t / 32) as u32, 1),
                        t.up_t as u32,
                        ush,
                        kargs!(
                            hc.up.ptr,
                            s.hcp.ptr,
                            s.iw.ptr,
                            s.normed.ptr,
                            s.folded.ptr,
                            s.folded_h.ptr,
                            d.h as i32,
                            d.hc as i32,
                            hc.lr as i32,
                            hc.r as i32,
                            nch as i32,
                            self.hi
                        ),
                    )
                }?;
                return Ok(());
            }
        }
        self.gemm_w(hc.di.ptr, hc.r, d.w, s.normed_h.ptr, n, s.low.ptr, self.gw)?;
        go!(self.c, self.q.hc_low, (blocks(n * hc.r), 1), 256;
            s.low.ptr, s.lowh.ptr, s.iw.ptr, n as i32, hc.r as i32, hc.lr as i32, d.hc as f32, self.hi);
        self.gemm_w(hc.up.ptr, d.w, hc.lr, s.lowh.ptr, n, s.mix.ptr, self.gw)?;
        go!(self.c, self.q.hc_fold, (blocks(n * d.h), 1), 256;
            s.mix.ptr, s.normed.ptr, s.folded.ptr, s.folded_h.ptr, n as i32, d.h as i32, d.hc as i32, self.hi);
        Ok(())
    }

    fn inject(&self, n: usize) -> Res<()> {
        let (d, s) = (&self.d, &self.s);
        go!(self.c, self.q.inject, (blocks(n * d.w), 1), 256;
            s.hyper.ptr, s.blk.ptr, s.iw.ptr, n as i32, d.h as i32, d.hc as i32);
        Ok(())
    }

    /// The gated delta rule over `n` tokens: `blk` = the mixer's output.
    /// With `hold` (a window of at most `tape` rows) the pass keeps its inputs
    /// on the tape and leaves the new state aside: `Dev::settle` makes it (or a
    /// shorter prefix of it) current.
    fn gdn(&self, g: &mut GdnW, n: usize, hold: bool) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let vd = d.nv * d.dv;
        let small = self.small(n);
        let tape = hold && n <= self.tape;
        let (s1, mix, gb) = if tape { (g.t_s1.ptr, g.t_mix.ptr, g.t_gb.ptr) } else { (s.s1.ptr, s.mix.ptr, s.gb.ptr) };
        if small && let (Some(mq), Some(mz)) = (g.qkv.q8(), g.z.q8()) {
            let ops = [SOp::q8(mq, s1), SOp::q8(mz, s.z.ptr), SOp::f32(g.ab.ptr, 2 * d.nv, s.ab.ptr)];
            self.mvb(&ops, s.folded.ptr, d.h, d.h, n)?;
        } else {
            let w = self.plane(&g.qkv, d.c_dim, d.h)?;
            self.gemm_w(w, d.c_dim, d.h, s.folded_h.ptr, n, s1, self.gw)?;
            let w = self.plane(&g.z, vd, d.h)?;
            self.gemm_w(w, vd, d.h, s.folded_h.ptr, n, s.z.ptr, self.gw)?;
            self.gemm_w(g.ab.ptr, 2 * d.nv, d.h, s.folded.ptr, n, s.ab.ptr, Gemm::F32)?;
        }
        go!(c, q.gdn_conv, (blocks(n * d.c_dim), 1), 256;
            s1, g.ring.ptr, g.conv.ptr, mix, n as i32, d.c_dim as i32, d.kk as i32);
        if d.kk > 1 {
            let cap = d.kk - 1;
            go!(c, q.hist, (blocks(cap * d.c_dim), 1), 256;
                g.ring.ptr, s1, g.ring2.ptr, cap as i32, d.c_dim as i32, n as i32);
        }
        go!(c, q.gdn_prep, ((n * 2 * d.nk) as u32, 1), 128; mix, d.c_dim as i32, d.nk as i32, d.dk as i32);
        go!(c, q.gdn_gb, (blocks(n * d.nv), 1), 256;
            s.ab.ptr, g.a_log.ptr, g.dt_bias.ptr, gb, n as i32, d.nv as i32);
        go!(c, q.gdn_scan, (vd.div_ceil(8) as u32, 1), 256;
            mix, gb, g.s.ptr, g.s2.ptr, s.qo.ptr, n as i32, d.c_dim as i32, d.nv as i32, d.nk as i32, d.dk as i32, d.dv as i32);
        if hold {
            g.dirty = true;
        } else {
            std::mem::swap(&mut g.s, &mut g.s2);
            if d.kk > 1 {
                std::mem::swap(&mut g.ring, &mut g.ring2);
            }
        }
        // the small path keeps the gated output in f32
        let direct = if small { g.out.q8() } else { None };
        let oh = if direct.is_some() { 0i32 } else { self.hi };
        go!(c, q.gdn_out, ((n * d.nv) as u32, 1), 128;
            s.qo.ptr, s.z.ptr, g.norm.ptr, s.ofh.ptr, d.dv as i32, d.gdn_eps, oh);
        if let Some(m) = direct {
            return self.mvb(&[SOp::q8(m, s.blk.ptr)], s.ofh.ptr, vd, vd, n);
        }
        let w = self.plane(&g.out, d.h, vd)?;
        self.gemm_w(w, d.h, vd, s.ofh.ptr, n, s.blk.ptr, self.gw)
    }

    /// Sparse attention over `n` tokens at `pos0..`: `blk` = the mixer's
    /// output. Returns the bytes its state grew by.
    fn qsa(&self, a: &mut QsaW, maskb: &mut Buf, pos0: usize, n: usize) -> Res<u64> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let (kv, qi, es) = (d.kvh * d.hd, d.ih * d.idim, self.es);
        let mut grown = 0u64;
        if pos0 + n > a.cap {
            let cap = (pos0 + n).max(a.cap + a.cap / 2).next_multiple_of(d.cr);
            let mut regrow = |old: &mut Buf, bytes: usize, keep: usize| -> Res<()> {
                let nb = Buf::new(bytes)?;
                go!(c, q.zero, (blocks(bytes / 4), 1), 256; nb.ptr, (bytes / 4) as u64);
                if keep > 0 {
                    unsafe { cu::memcpy_dtod_async(nb.ptr, old.ptr, keep, c.stream) }.map_err(derr)?;
                }
                c.sync()?;
                grown += nb.bytes.saturating_sub(old.bytes) as u64;
                *old = nb;
                Ok(())
            };
            regrow(&mut a.keys, cap * kv * 4, pos0 * kv * 4)?;
            regrow(&mut a.vals, cap * kv * 4, pos0 * kv * 4)?;
            regrow(&mut a.ikraw, cap * d.idim * 4, pos0 * d.idim * 4)?;
            regrow(&mut a.bkey, (cap / d.cr + 1) * d.idim * 4, (pos0 / d.cr) * d.idim * 4)?;
            a.cap = cap;
        }
        // projections: s1 = [q | gate] per head (kept until the merge)
        let small = self.small(n);
        if small && let (Some(mq), Some(mk), Some(mv), Some(mi)) = (a.q.q8(), a.k.q8(), a.v.q8(), a.iqk.q8()) {
            // all five in one launch; the indexer's matrix is two jobs: its
            // query rows, then its key rows
            let iq = SOp { rows: qi, ..SOp::q8(mi, s.mix.ptr) };
            let ik = SOp {
                w: mi.w.ptr + (qi * d.h) as u64,
                aux0: mi.aux0() + (qi * 4) as u64,
                rows: d.idim,
                ..SOp::q8(mi, a.ikraw.ptr + (pos0 * d.idim * 4) as u64)
            };
            let ops = [SOp::q8(mq, s.s1.ptr), SOp::q8(mk, s.kraw.ptr), SOp::q8(mv, a.vals.ptr + (pos0 * kv * 4) as u64), iq, ik];
            self.mvb(&ops, s.folded.ptr, d.h, d.h, n)?;
        } else {
            let w = self.plane(&a.q, d.nh * 2 * d.hd, d.h)?;
            self.gemm_w(w, d.nh * 2 * d.hd, d.h, s.folded_h.ptr, n, s.s1.ptr, self.gw)?;
            let w = self.plane(&a.k, kv, d.h)?;
            self.gemm_w(w, kv, d.h, s.folded_h.ptr, n, s.kraw.ptr, self.gw)?;
            let w = self.plane(&a.v, kv, d.h)?;
            self.gemm_w(w, kv, d.h, s.folded_h.ptr, n, a.vals.ptr + (pos0 * kv * 4) as u64, self.gw)?;
            let w = self.plane(&a.iqk, qi + d.idim, d.h)?;
            self.gemm_w(w, qi, d.h, s.folded_h.ptr, n, s.mix.ptr, self.gw)?;
            self.gemm_w(w + (qi * d.h * es) as u64, d.idim, d.h, s.folded_h.ptr, n, a.ikraw.ptr + (pos0 * d.idim * 4) as u64, self.gw)?;
        }
        let (bq, bi) = (d.hd.next_power_of_two() as u32, d.idim.next_power_of_two() as u32);
        // query heads -> qq[((kvh n + t) groups + g) hd + d]
        go!(c, q.norm_rope, ((n * d.nh) as u32, 1), bq;
            s.s1.ptr, s.qq.ptr, a.q_norm.ptr, self.invf.ptr, d.nh as i32, d.hd as i32,
            (d.nh * 2 * d.hd) as u64, (2 * d.hd) as u64, (d.groups * d.hd) as u64, d.hd as u64, d.groups as i32,
            (n * d.groups * d.hd) as u64, d.rot as i32, d.nfreq as i32, pos0 as i32, 1i32, 1i32, d.eps);
        // key heads -> the key rows of these positions
        go!(c, q.norm_rope, ((n * d.kvh) as u32, 1), bq;
            s.kraw.ptr, a.keys.ptr + (pos0 * kv * 4) as u64, a.k_norm.ptr, self.invf.ptr, d.kvh as i32, d.hd as i32,
            kv as u64, d.hd as u64, kv as u64, d.hd as u64, d.kvh as i32, 0u64,
            d.rot as i32, d.nfreq as i32, pos0 as i32, 1i32, 1i32, d.eps);
        // indexer queries
        go!(c, q.norm_rope, ((n * d.ih) as u32, 1), bi;
            s.mix.ptr, s.iq.ptr, a.iq_norm.ptr, self.invf.ptr, d.ih as i32, d.idim as i32,
            qi as u64, d.idim as u64, qi as u64, d.idim as u64, d.ih as i32, 0u64,
            d.rot as i32, d.nfreq as i32, pos0 as i32, 1i32, 1i32, d.eps);
        // keys of the blocks this chunk completes: the mean of a block's raw keys
        let (b0, b1) = (pos0 / d.cr, (pos0 + n) / d.cr);
        if b1 > b0 {
            go!(c, q.norm_rope, ((b1 - b0) as u32, 1), bi;
                a.ikraw.ptr + (b0 * d.cr * d.idim * 4) as u64, a.bkey.ptr + (b0 * d.idim * 4) as u64, a.ik_norm.ptr,
                self.invf.ptr, 1i32, d.idim as i32, d.idim as u64, 0u64, d.idim as u64, d.idim as u64, 1i32, 0u64,
                d.rot as i32, d.nfreq as i32, (b0 * d.cr) as i32, d.cr as i32, d.cr as i32, d.eps);
        }
        // the indexer, for the tokens that see more complete blocks than the
        // budget keeps (every pass before the score passes: both use `sbuf`)
        let t_act0 = ((d.keep + 1) * d.cr - 1).saturating_sub(pos0);
        let (mut mask_p0, mut mwords) = (0usize, 0usize);
        if t_act0 < n {
            let nb = (pos0 + n) / d.cr;
            mwords = nb.div_ceil(32);
            mask_p0 = pos0 + t_act0;
            let per = nb.div_ceil(256).next_multiple_of(32);
            let nact = n - t_act0;
            if nact * mwords * 4 > maskb.bytes {
                c.sync()?;
                let nbuf = Buf::new(nact * mwords * 4 * 3 / 2)?;
                grown += nbuf.bytes.saturating_sub(maskb.bytes) as u64;
                *maskb = nbuf;
            }
            let rmax = (self.ds / (d.ih * nb * 4)).max(1);
            let inv = 1.0 / (d.idim as f32).sqrt();
            let mut row = 0usize;
            while row < nact {
                let r = rmax.min(nact - row);
                let t = t_act0 + row;
                unsafe {
                    c.gemm(true, false, nb, r * d.ih, d.idim, 1.0, a.bkey.ptr, d.idim, s.iq.ptr + (t * qi * 4) as u64, d.idim, 0.0, s.sbuf.ptr, nb, Gemm::F32)
                }?;
                go!(c, q.idx_reduce, (blocks(r * nb), 1), 256; s.sbuf.ptr, s.sc.ptr, nb as i32, d.ih as i32, r as i32, inv);
                go!(c, q.idx_topk, (r as u32, 1), 256;
                    s.sc.ptr, maskb.ptr + (row * mwords * 4) as u64, nb as i32, mwords as i32, d.keep as i32,
                    (pos0 + t) as i32, d.cr as i32, per as i32);
                row += r;
            }
        }
        // scores, softmax and values per key head and run of tokens
        let budget = self.sbuf / (4 * d.groups);
        let scale = 1.0 / (d.hd as f32).sqrt();
        for kh in 0..d.kvh {
            let mut t0 = 0usize;
            while t0 < n {
                let p = pos0 + t0;
                let mut r = (n - t0).min(2048);
                while r > 1 && r * (p + r) > budget {
                    r -= r.div_ceil(8);
                }
                if r * (p + r) > budget {
                    return Err("qwen4 cuda: the prompt is longer than the attention scratch".into());
                }
                let l = p + r;
                let qoff = ((kh * n + t0) * d.groups * d.hd * 4) as u64;
                unsafe {
                    c.gemm(true, false, l, r * d.groups, d.hd, scale, a.keys.ptr + (kh * d.hd * 4) as u64, kv, s.qq.ptr + qoff, d.hd, 0.0, s.sbuf.ptr, l, self.ga)
                }?;
                go!(c, q.qsa_softmax, ((r * d.groups) as u32, 1), 256;
                    s.sbuf.ptr, maskb.ptr, l as i32, d.groups as i32, p as i32, mask_p0 as i32, mwords as i32,
                    d.keep as i32, d.cr as i32);
                unsafe {
                    c.gemm(false, false, d.hd, r * d.groups, l, 1.0, a.vals.ptr + (kh * d.hd * 4) as u64, kv, s.sbuf.ptr, l, 0.0, s.qo.ptr + qoff, d.hd, self.ga)
                }?;
                t0 += r;
            }
        }
        let direct = if small { a.o.q8() } else { None };
        let oh = if direct.is_some() { 0i32 } else { self.hi };
        go!(c, q.qsa_merge, (blocks(n * d.nh * d.hd), 1), 256;
            s.qo.ptr, s.s1.ptr, s.ofh.ptr, n as i32, d.nh as i32, d.groups as i32, d.hd as i32, oh);
        if let Some(m) = direct {
            self.mvb(&[SOp::q8(m, s.blk.ptr)], s.ofh.ptr, d.nh * d.hd, d.nh * d.hd, n)?;
            return Ok(grown);
        }
        let w = self.plane(&a.o, d.h, d.nh * d.hd)?;
        self.gemm_w(w, d.h, d.nh * d.hd, s.ofh.ptr, n, s.blk.ptr, self.gw)?;
        Ok(grown)
    }

    /// The n-gram side input of a layer, added to `hyper`; `rows[token]` are
    /// the table rows the host read.
    fn ple(&self, p: &mut PleW, rows: &[Vec<f32>], n: usize, hold: bool) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        if rows.len() != n || rows.iter().any(|r| r.len() != p.pin) {
            return Err("qwen4 cuda: unexpected PLE rows".into());
        }
        let mut flat = Vec::with_capacity(n * p.pin);
        for r in rows {
            flat.extend_from_slice(r);
        }
        self.put(s.ple_in.ptr, &flat)?;
        go!(c, q.to_wt, (blocks(n * p.pin), 1), 256; s.ple_in.ptr, s.ple_h.ptr, (n * p.pin) as u64, self.hi);
        let w = self.plane(&p.key, d.w, p.pin)?;
        self.gemm_w(w, d.w, p.pin, s.ple_h.ptr, n, s.mix.ptr, self.gw)?;
        go!(c, q.grms, ((n * d.hc) as u32, 1), 256;
            s.mix.ptr, p.norm_key.ptr, s.s1.ptr, 0u64, d.h as i32, d.hc as i32, d.eps, 0i32);
        go!(c, q.grms, ((n * d.hc) as u32, 1), 256;
            s.hyper.ptr, p.norm_query.ptr, s.normed.ptr, 0u64, d.h as i32, d.hc as i32, d.eps, 0i32);
        let w = self.plane(&p.value, d.h, p.pin)?;
        self.gemm_w(w, d.h, p.pin, s.ple_h.ptr, n, s.blk.ptr, self.gw)?;
        // mix = the gated value per stream
        go!(c, q.ple_gate, ((n * d.hc) as u32, 1), 256; s.s1.ptr, s.normed.ptr, s.blk.ptr, s.mix.ptr, d.h as i32, d.hc as i32);
        let normed = if hold && n <= self.tape { p.t_normed.ptr } else { s.normed.ptr };
        go!(c, q.grms, ((n * d.hc) as u32, 1), 256;
            s.mix.ptr, p.norm_conv.ptr, normed, 0u64, d.h as i32, d.hc as i32, d.eps, 0i32);
        go!(c, q.ple_conv, (blocks(n * d.w), 1), 256;
            normed, p.hist.ptr, p.conv.ptr, s.mix.ptr, s.hyper.ptr, n as i32, d.w as i32, d.ple_k as i32, d.ple_dil as i32);
        let cap = (d.ple_k - 1) * d.ple_dil;
        if cap > 0 {
            go!(c, q.hist, (blocks(cap * d.w), 1), 256; p.hist.ptr, normed, p.hist2.ptr, cap as i32, d.w as i32, n as i32);
            if hold {
                p.dirty = true;
            } else {
                std::mem::swap(&mut p.hist, &mut p.hist2);
            }
        }
        Ok(())
    }

    /// The routed experts and the shared one over `n` tokens: `blk` = the
    /// block's output. Experts are grouped into strided batches by the number
    /// of tokens that picked them.
    fn moe(&self, lw: &MoeW, layer: &Layer, li: usize, arena: &mut Arena, n: usize, hold: bool) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let topk = lw.topk;
        // a window of a few rows, or of a frame's many (`hold`), takes the unit kernels
        let small = self.small(n);
        if small || (hold && self.mid && n > SMALL_ROWS && n <= self.tape) {
            // an expert of the round once for all the rows that picked it,
            // straight from its raw slot: gate and up, down, the weighted sums
            let (sm, t) = (&q.sm, &self.tune);
            let kk = topk + lw.shared.is_some() as usize;
            let (gg, gd) = (d.h >> 5, d.inter >> 5);
            let room = sm.shared_max / 4;
            // rows of a unit by what a block can stage
            let ur = [
                if t.share { UNIT_ROWS } else { 1 },
                room.saturating_sub(2 * t.g_rb * 32) / (32 * (gg | 1) + gg),
                room.saturating_sub(t.d_rb * 32) / (32 * (gd | 1) + gd),
            ]
            .into_iter()
            .min()
            .unwrap_or(0);
            let fits = d.h.is_multiple_of(32)
                && d.inter.is_multiple_of(32)
                && d.sgu.is_multiple_of(16)
                && arena.slot.is_multiple_of(16)
                && ur > 0
                && n * kk * (UNIT_WORDS + 1) * 4 <= s.slots.bytes
                && n * kk * UNIT_ROWS * d.inter * 4 <= s.g.bytes
                && n * kk * UNIT_ROWS * d.h * 4 <= s.d.bytes;
            if fits {
                if small {
                    self.mvb(&[SOp::f32(lw.router.ptr, lw.r, s.logits.ptr)], s.folded.ptr, d.h, d.h, n)?;
                } else {
                    self.gemm_w(lw.router.ptr, lw.r, d.h, s.folded.ptr, n, s.logits.ptr, Gemm::F32)?;
                }
                let head = li >= arena.main_layers();
                let tbl_p = arena.tbl_ptr(c, li, topk, head)?;
                // the route is masked by the table in the backbone's layers; the head routes over all its experts
                let mask_p = if head { 0u64 } else { tbl_p };
                go!(c, q.route, (n as u32, 1), 256; s.logits.ptr, s.picks.ptr, s.wts.ptr, lw.ne as i32, lw.r as i32, topk as i32, self.tail, mask_p);
                let mut lean_saved: Option<Vec<f32>> = None;
                if small && self.lean_on() && tbl_p != 0 && ur == UNIT_ROWS && lw.ne <= arena.tbl_stride {
                    // every pick is in the arena: the units are built on the card and nothing waits for the host
                    let nsh = if lw.shared.is_some() { n.div_ceil(UNIT_ROWS) } else { 0 };
                    // at most one unit a pick, and the shared expert's
                    let nu = n * topk + nsh;
                    let refs_at = s.slots.ptr + (nu * UNIT_WORDS * 4) as u64;
                    if refs_at + (n * kk * 4) as u64 <= s.slots.ptr + s.slots.bytes as u64
                        && nu * UNIT_ROWS * d.inter * 4 <= s.g.bytes
                        && nu * UNIT_ROWS * d.h * 4 <= s.d.bytes
                    {
                        let shared_at = lw.shared.as_ref().map_or(0u64, |b| b.ptr);
                        go!(c, q.units, (1u32, 1), 256;
                            s.picks.ptr, tbl_p, shared_at, s.slots.ptr, refs_at, n as i32, topk as i32, kk as i32, lw.ne as i32, nu as i32, nsh as i32);
                        let gsh = (UNIT_ROWS * (32 * (gg | 1) + gg) + 2 * t.g_rb * 32) * 4;
                        let dsh = (UNIT_ROWS * (32 * (gd | 1) + gd) + t.d_rb * 32) * 4;
                        unsafe {
                            c.launch(
                                sm.exp_gu,
                                (d.inter.div_ceil(t.g_rb) as u32, nu as u32),
                                t.g_t as u32,
                                gsh,
                                kargs!(s.slots.ptr, s.folded.ptr, s.g.ptr, d.inter as i32, d.h as i32, t.g_rb as i32),
                            )
                        }?;
                        unsafe {
                            c.launch(
                                sm.exp_d,
                                (d.h.div_ceil(t.d_rb) as u32, nu as u32),
                                t.d_t as u32,
                                dsh,
                                kargs!(s.slots.ptr, s.g.ptr, s.d.ptr, d.inter as i32, d.h as i32, t.d_rb as i32),
                            )
                        }?;
                        go!(c, sm.exp_sum, (blocks(n * d.h), 1), 256;
                            refs_at, s.d.ptr, s.wts.ptr, s.logits.ptr, s.blk.ptr,
                            n as i32, kk as i32, topk as i32, d.h as i32, lw.r as i32, lw.gcol);
                        if std::env::var_os("CMF_QWEN_LEAN_CHECK").is_none() {
                            return self.lap(7);
                        }
                        // check mode: keep the lean result, the host path runs too and the two are compared
                        c.sync()?;
                        lean_saved = Some(self.grab(s.blk.ptr, n * d.h)?);
                    }
                }
                let mut picks = vec![0i32; n * topk];
                unsafe { cu::memcpy_dtoh_async(&mut picks, s.picks.ptr, c.stream) }.map_err(derr)?;
                c.sync()?;
                arena.pending = false;
                arena.note(li, &picks);
                self.lap(5)?;
                let valid = |e: i32| e >= 0 && (e as usize) < lw.ne;
                // the experts of the round in order of first pick, and the places each took
                let mut order: Vec<usize> = Vec::with_capacity(picks.len().min(lw.ne));
                let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); lw.ne];
                for (i, &e) in picks.iter().enumerate() {
                    if valid(e) {
                        let e = e as usize;
                        if by_expert[e].is_empty() {
                            order.push(e);
                        }
                        by_expert[e].push(i / topk * kk + i % topk);
                    }
                }
                let mut slot_of = vec![0u64; lw.ne];
                arena.admit_layer(li, &order, &layer.moe.experts, d, c, &mut slot_of)?;
                self.lap(6)?;
                // the units, then for each (token, place) the row of the
                // partial outputs it takes
                let mut units: Vec<u32> = Vec::with_capacity(n * kk * (UNIT_WORDS + 1));
                let mut refs = vec![-1i32; n * kk];
                let (mut cmax, mut pairs) = (0usize, 0usize);
                let mut add = |slot: u64, places: &[usize]| {
                    for group in places.chunks(ur) {
                        let u = units.len() / UNIT_WORDS;
                        units.extend([slot as u32, (slot >> 32) as u32]);
                        for j in 0..UNIT_ROWS {
                            match group.get(j) {
                                Some(&p) => {
                                    units.push((p / kk) as u32);
                                    refs[p] = (u * UNIT_ROWS + j) as i32;
                                }
                                None => units.push(u32::MAX),
                            }
                        }
                        cmax = cmax.max(group.len());
                        pairs += group.len();
                    }
                };
                for &e in &order {
                    add(slot_of[e], &by_expert[e]);
                }
                if let Some(sh) = &lw.shared {
                    let places: Vec<usize> = (0..n).map(|t| t * kk + topk).collect();
                    add(sh.ptr, &places);
                }
                let nu = units.len() / UNIT_WORDS;
                if nu == 0 {
                    go!(c, q.zero, (blocks(n * d.h), 1), 256; s.blk.ptr, (n * d.h) as u64);
                    return self.lap(7);
                }
                arena.units += nu as u64;
                arena.pairs += pairs as u64;
                let refs_at = s.slots.ptr + (units.len() * 4) as u64;
                units.extend(refs.iter().map(|&r| r as u32));
                self.put(s.slots.ptr, &units)?;
                let gsh = (cmax * (32 * (gg | 1) + gg) + 2 * t.g_rb * 32) * 4;
                let dsh = (cmax * (32 * (gd | 1) + gd) + t.d_rb * 32) * 4;
                unsafe {
                    c.launch(
                        sm.exp_gu,
                        (d.inter.div_ceil(t.g_rb) as u32, nu as u32),
                        t.g_t as u32,
                        gsh,
                        kargs!(s.slots.ptr, s.folded.ptr, s.g.ptr, d.inter as i32, d.h as i32, t.g_rb as i32),
                    )
                }?;
                unsafe {
                    c.launch(
                        sm.exp_d,
                        (d.h.div_ceil(t.d_rb) as u32, nu as u32),
                        t.d_t as u32,
                        dsh,
                        kargs!(s.slots.ptr, s.g.ptr, s.d.ptr, d.inter as i32, d.h as i32, t.d_rb as i32),
                    )
                }?;
                go!(c, sm.exp_sum, (blocks(n * d.h), 1), 256;
                    refs_at, s.d.ptr, s.wts.ptr, s.logits.ptr, s.blk.ptr,
                    n as i32, kk as i32, topk as i32, d.h as i32, lw.r as i32, lw.gcol);
                if let Some(a) = lean_saved {
                    c.sync()?;
                    let b = self.grab(s.blk.ptr, n * d.h)?;
                    let (mut worst, mut at) = (0f32, 0usize);
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        if (x - y).abs() > worst {
                            (worst, at) = ((x - y).abs(), i);
                        }
                    }
                    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    if worst > 1e-4 * scale.max(1e-6) {
                        eprintln!("qwen4 cuda: lean check layer {li} rows {n}: max diff {worst:e} at {at} (lean {} host {}), scale {scale:e}", a[at], b[at]);
                    }
                }
                return self.lap(7);
            }
        }
        let half = self.hi != 0;
        // f16 activations: scaled down before the down projection, back up in it
        let (inv, alpha) = if half { (1.0f32 / 16.0, 16.0f32) } else { (1.0f32, 1.0f32) };
        self.gemm_w(lw.router.ptr, lw.r, d.h, s.folded.ptr, n, s.logits.ptr, Gemm::F32)?;
        let allow_p = if hold && li < arena.main_layers() { arena.tbl_ptr(c, li, topk, false)? } else { 0u64 };
        go!(c, q.route, (n as u32, 1), 256; s.logits.ptr, s.picks.ptr, s.wts.ptr, lw.ne as i32, lw.r as i32, topk as i32, 0.0f32, allow_p);
        let mut picks = vec![0i32; n * topk];
        unsafe { cu::memcpy_dtoh_async(&mut picks, s.picks.ptr, c.stream) }.map_err(derr)?;
        c.sync()?;
        // every copy out of the upload staging is through
        arena.pending = false;
        arena.note(li, &picks);
        self.lap(5)?;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); lw.ne];
        for (i, &e) in picks.iter().enumerate() {
            if e >= 0 && (e as usize) < lw.ne {
                lists[e as usize].push(i as u32);
            }
        }
        let mut order: Vec<usize> = (0..lw.ne).filter(|&e| !lists[e].is_empty()).collect();
        order.sort_by(|&x, &y| lists[y].len().cmp(&lists[x].len()).then(x.cmp(&y)));
        let mut slot_of = vec![0u64; lw.ne];
        arena.admit_layer(li, &order, &layer.moe.experts, d, c, &mut slot_of)?;
        self.lap(6)?;
        let (mut i, mut first) = (0usize, true);
        while i < order.len() {
            let ne_max = lists[order[i]].len();
            if ne_max > self.xr {
                return Err("qwen4 cuda: an expert with more rows than the sub-batch".into());
            }
            let mut b = 1usize;
            while i + b < order.len() && b < self.xb && (b + 1) * ne_max <= self.xr {
                if b >= 8 && lists[order[i + b]].len() * 2 < ne_max {
                    break;
                }
                b += 1;
            }
            let rows = b * ne_max;
            let mut slots = vec![0u64; b];
            let mut rowtok = vec![-1i32; rows];
            let mut tokrow = vec![-1i32; n * topk];
            for j in 0..b {
                let e = order[i + j];
                slots[j] = slot_of[e];
                for (r, &entry) in lists[e].iter().enumerate() {
                    rowtok[j * ne_max + r] = (entry as usize / topk) as i32;
                    tokrow[entry as usize] = (j * ne_max + r) as i32;
                }
            }
            self.put(s.slots.ptr, &slots)?;
            self.put(s.rowtok.ptr, &rowtok)?;
            self.put(s.tokrow.ptr, &tokrow)?;
            go!(c, q.expand_exp, (blocks(d.inter * d.h / 8), (3 * b) as u32), 256;
                s.slots.ptr, s.wg.ptr, s.wu.ptr, s.wd.ptr, d.inter as i32, d.h as i32, self.hi);
            go!(c, q.gather, (blocks(rows * d.h), 1), 256; s.folded_h.ptr, s.rowtok.ptr, s.xg.ptr, rows as i32, d.h as i32, self.hi);
            let (gs, xs, as_, ds) = (d.inter * d.h, ne_max * d.h, ne_max * d.inter, ne_max * d.h);
            unsafe {
                if half {
                    gemm_batched_half(c, true, d.inter, ne_max, d.h, 1.0, s.wg.ptr, d.h, gs, s.xg.ptr, d.h, xs, s.g.ptr, d.inter, as_, b)?;
                    gemm_batched_half(c, true, d.inter, ne_max, d.h, 1.0, s.wu.ptr, d.h, gs, s.xg.ptr, d.h, xs, s.u.ptr, d.inter, as_, b)?;
                } else {
                    c.gemm_batched(true, d.inter, ne_max, d.h, 1.0, s.wg.ptr, d.h, gs, s.xg.ptr, d.h, xs, s.g.ptr, d.inter, as_, b, self.gw)?;
                    c.gemm_batched(true, d.inter, ne_max, d.h, 1.0, s.wu.ptr, d.h, gs, s.xg.ptr, d.h, xs, s.u.ptr, d.inter, as_, b, self.gw)?;
                }
            }
            go!(c, q.act, (blocks(rows * d.inter), 1), 256; s.g.ptr, s.u.ptr, s.ah.ptr, (rows * d.inter) as u64, self.hi, self.hi, inv);
            unsafe {
                if half {
                    gemm_batched_half(c, true, d.h, ne_max, d.inter, alpha, s.wd.ptr, d.inter, gs, s.ah.ptr, d.inter, as_, s.d.ptr, d.h, ds, b)?;
                } else {
                    c.gemm_batched(true, d.h, ne_max, d.inter, alpha, s.wd.ptr, d.inter, gs, s.ah.ptr, d.inter, as_, s.d.ptr, d.h, ds, b, self.gw)?;
                }
            }
            go!(c, q.scatter, (blocks(n * d.h), 1), 256;
                s.d.ptr, s.tokrow.ptr, s.wts.ptr, s.blk.ptr, n as i32, d.h as i32, topk as i32, self.hi, first as i32);
            if first && self.range {
                self.range("expert x", li, s.xg.ptr, rows * d.h)?;
                self.range("expert gate", li, s.g.ptr, rows * d.inter)?;
                self.range("expert up", li, s.u.ptr, rows * d.inter)?;
                self.range("expert act", li, s.ah.ptr, rows * d.inter)?;
                self.range("expert down", li, s.d.ptr, rows * d.h)?;
                self.range("expert w gate", li, s.wg.ptr, b * d.inter * d.h)?;
                self.range("expert w down", li, s.wd.ptr, b * d.inter * d.h)?;
            }
            first = false;
            i += b;
        }
        if first {
            go!(c, q.zero, (blocks(n * d.h), 1), 256; s.blk.ptr, (n * d.h) as u64);
        }
        self.lap(7)?;
        if let Some(sh) = &lw.shared {
            self.put(s.slots.ptr, &[sh.ptr])?;
            go!(c, q.expand_exp, (blocks(d.inter * d.h / 8), 3), 256;
                s.slots.ptr, s.wg.ptr, s.wu.ptr, s.wd.ptr, d.inter as i32, d.h as i32, self.hi);
            // one GEMM per projection over all tokens, f32 results
            self.gemm_w(s.wg.ptr, d.inter, d.h, s.folded_h.ptr, n, s.g.ptr, self.gw)?;
            self.gemm_w(s.wu.ptr, d.inter, d.h, s.folded_h.ptr, n, s.u.ptr, self.gw)?;
            go!(c, q.act, (blocks(n * d.inter), 1), 256; s.g.ptr, s.u.ptr, s.ah.ptr, (n * d.inter) as u64, 0i32, self.hi, inv);
            unsafe { c.gemm(true, false, d.h, n, d.inter, alpha, s.wd.ptr, d.inter, s.ah.ptr, d.inter, 0.0, s.d.ptr, d.h, self.gw) }?;
            go!(c, q.shared_add, (blocks(n * d.h), 1), 256; s.blk.ptr, s.d.ptr, s.logits.ptr, n as i32, d.h as i32, lw.r as i32, lw.gcol);
            self.lap(8)?;
        }
        Ok(())
    }

    /// Whether the lean path is taken now.
    fn lean_on(&self) -> bool {
        match self.lean_mode {
            0 => false,
            1 => true,
            _ => self.lean_flip.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Keep the final hyper state of row `row` for the MTP head, taken at `pos` tokens.
    fn keep_r(&self, m: &mut MtpW, row: usize, pos: usize) -> Res<()> {
        let (d, s, c) = (&self.d, &self.s, self.c);
        unsafe { cu::memcpy_dtod_async(m.r_last.ptr, s.hyper.ptr + (row * d.w * 4) as u64, d.w * 4, c.stream) }.map_err(derr)?;
        m.r_pos = Some(pos);
        Ok(())
    }

    /// The logits of the chunk's last token.
    fn head(&self, head: &Hc, lm: &DevMat, n: usize, logits: &mut Vec<f32>) -> Res<()> {
        let (d, s, c) = (&self.d, &self.s, self.c);
        // the head's mix has no inject rows: nothing is written to `iw`
        self.hc_mix(head, s.hyper.ptr + ((n - 1) * d.w * 4) as u64, 1)?;
        let (mut jobs, mut blk) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk, lm, s.out.ptr, 0);
        let jb = Buf::from_slice(&jobs)?;
        let (nullp, no_act, cols_i, eps) = (0u64, -1i32, lm.cols as i32, 0f32);
        logits.clear();
        logits.resize(d.vocab, 0.0);
        unsafe {
            c.launch(
                c.f.mv_multi,
                (blk, 1),
                256,
                mv_shared_bytes(lm.cols),
                kargs!(jb.ptr, 1i32, s.folded.ptr, nullp, no_act, nullp, eps, 0i32, cols_i),
            )?;
            cu::memcpy_dtoh_async(logits.as_mut_slice(), s.out.ptr, c.stream).map_err(derr)?;
        }
        c.sync()?;
        drop(jb);
        Ok(())
    }

    /// The most likely token of every row of the chunk, `n` of them: the
    /// logits of eight rows at a time go through `s.out` and a block-wide argmax.
    fn head_rows(&self, head: &Hc, lm: &DevMat, n: usize, ids: &mut Vec<u32>) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        self.hc_mix(head, s.hyper.ptr, n)?;
        let mut held: Vec<Buf> = Vec::new();
        let mut t0 = 0;
        while t0 < n {
            let rows = (n - t0).min(SMALL_ROWS);
            let x = s.folded.ptr + (t0 * d.h * 4) as u64;
            if lm.kind == 7 && self.small {
                self.mvb(&[SOp::q8(lm, s.out.ptr)], x, d.h, d.h, rows)?;
            } else {
                for r in 0..rows {
                    let (mut jobs, mut blk) = (Vec::new(), 0u32);
                    push_job(&mut jobs, &mut blk, lm, s.out.ptr + (r * d.vocab * 4) as u64, 0);
                    let jb = Buf::from_slice(&jobs)?;
                    let (nullp, no_act, cols_i, eps) = (0u64, -1i32, lm.cols as i32, 0f32);
                    unsafe {
                        c.launch(
                            c.f.mv_multi,
                            (blk, 1),
                            256,
                            mv_shared_bytes(lm.cols),
                            kargs!(jb.ptr, 1i32, x + (r * d.h * 4) as u64, nullp, no_act, nullp, eps, 0i32, cols_i),
                        )?;
                    }
                    held.push(jb);
                }
            }
            go!(c, q.argmax, (rows as u32, 1), 1024; s.out.ptr, s.ids.ptr + (t0 * 4) as u64, d.vocab as i32);
            t0 += rows;
        }
        let mut got = vec![0i32; n];
        unsafe { cu::memcpy_dtoh_async(got.as_mut_slice(), s.ids.ptr, c.stream) }.map_err(derr)?;
        c.sync()?;
        drop(held);
        ids.clear();
        ids.extend(got.iter().map(|&i| i.max(0) as u32));
        Ok(())
    }
}

/// What a pass brings back.
enum Out<'a> {
    None,
    /// The last row's logits.
    Last(&'a mut Vec<f32>),
    /// Every row's most likely token; the pass is a window: its state changes
    /// wait for `Dev::settle`.
    Rows(&'a mut Vec<u32>),
}

impl Dev {
    /// Run `n` tokens (their embedding rows `[n][h]`) at positions `pos0..`.
    /// `ple[layer][token]` are the layer's n-gram table rows (empty = the
    /// layer's side input is skipped). With `logits`, the last token's logits
    /// come back. After an `Err` the state is undefined: drop the device.
    pub(crate) fn forward(
        &mut self,
        layers: &[Layer],
        emb: &[f32],
        ple: &[Vec<Vec<f32>>],
        pos0: usize,
        logits: Option<&mut Vec<f32>>,
    ) -> Res<()> {
        let eager = |li: usize| ple.get(li).cloned().unwrap_or_default();
        self.run(layers, emb, &eager, pos0, match logits {
            Some(l) => Out::Last(l),
            None => Out::None,
        })
    }

    /// `forward` with the n-gram rows of a layer asked for when the pass reaches it: the host
    /// gathers them (page faults in the table) while the card works on the layers before.
    pub(crate) fn forward_lazy(&mut self, layers: &[Layer], emb: &[f32], ple: &dyn Fn(usize) -> Vec<Vec<f32>>, pos0: usize, logits: Option<&mut Vec<f32>>) -> Res<()> {
        self.run(layers, emb, ple, pos0, match logits {
            Some(l) => Out::Last(l),
            None => Out::None,
        })
    }

    /// A verify window: the chunk (at most `tape` rows) runs with the state
    /// left as it was; `ids` get every row's most likely token. `settle`
    /// then keeps the rows that stand.
    pub(crate) fn spec(&mut self, layers: &[Layer], emb: &[f32], ple: &dyn Fn(usize) -> Vec<Vec<f32>>, pos0: usize, ids: &mut Vec<u32>) -> Res<()> {
        self.run(layers, emb, ple, pos0, Out::Rows(ids))
    }

    fn run(&mut self, layers: &[Layer], emb: &[f32], ple: &dyn Fn(usize) -> Vec<Vec<f32>>, pos0: usize, out: Out) -> Res<()> {
        let Self { k, layers: lws, head, lm, arena, maskb, held, pos, pending, want_trace, trace, mtp, launch_ms, .. } = self;
        let t_run = Instant::now();
        let k = &*k;
        let (d, s, c, q) = (&k.d, &k.s, k.c, k.q);
        let n = emb.len() / d.h;
        if n == 0 || n > k.n_cap || emb.len() != n * d.h || layers.len() != lws.len() || pos0 != *pos {
            return Err(format!("qwen4 cuda: a chunk of {n} tokens at {pos0} (state at {})", *pos));
        }
        let hold = matches!(out, Out::Rows(_));
        if hold && n > k.tape {
            return Err(format!("qwen4 cuda: a window of {n} rows, the tape holds {}", k.tape));
        }
        let _st = c.st.lock().unwrap();
        c.current()?;
        if pending.take().is_some() {
            // a window nobody settled: its results are void
            for lw in lws.iter_mut() {
                if let Mix::Gdn(g) = &mut lw.mix {
                    g.dirty = false;
                }
                if let Some(p) = &mut lw.ple {
                    p.dirty = false;
                }
            }
        }
        arena.pass += 1;
        if let Some(p) = &k.prof {
            let mut p = p.lock().unwrap();
            if n > 1 && p.steps > 0 {
                // the single tokens before this chunk get their own line
                let what = format!("{} single tokens to {pos0}", p.steps);
                p.report(arena, what);
            }
            p.t0 = Instant::now();
        }
        trace.clear();
        let tr = *want_trace;
        // a stage's row of the chunk's last token
        let last = |ptr: u64, width: usize| k.grab(ptr + ((n - 1) * width * 4) as u64, width);
        k.put(s.emb.ptr, emb)?;
        go!(c, q.tile, (blocks(n * d.h), 1), 256; s.emb.ptr, s.hyper.ptr, n as i32, d.h as i32, d.hc as i32);
        if tr {
            trace.push(("embedding", 0, last(s.emb.ptr, d.h)?));
        }
        k.lap(0)?;
        for (li, (lw, layer)) in lws.iter_mut().zip(layers).enumerate() {
            if let Some(p) = &mut lw.ple
                && let rows = ple(li)
                && !rows.is_empty()
            {
                k.ple(p, &rows, n, hold)?;
                if tr {
                    trace.push(("post_ple", li, last(s.hyper.ptr, d.w)?));
                }
                k.lap(1)?;
            }
            k.hc_mix(&lw.attn, s.hyper.ptr, n)?;
            if k.range {
                k.range("attn normed", li, s.normed_h.ptr, n * d.w)?;
                if !k.small(n) {
                    k.range("attn low", li, s.lowh.ptr, n * lw.attn.lr)?;
                }
                k.range("attn folded", li, s.folded_h.ptr, n * d.h)?;
            }
            if tr {
                trace.push(("attn_in", li, last(s.folded.ptr, d.h)?));
            }
            k.lap(2)?;
            match &mut lw.mix {
                Mix::Gdn(g) => {
                    k.gdn(g, n, hold)?;
                    if !k.small(n) {
                        k.range("gdn out", li, s.ofh.ptr, n * d.nv * d.dv)?;
                    }
                    k.lap(3)?;
                }
                Mix::Qsa(a) => {
                    let grown = k.qsa(a, maskb, pos0, n)?;
                    *held += grown;
                    c.resident.fetch_add(grown, std::sync::atomic::Ordering::Relaxed);
                    if !k.small(n) {
                        k.range("qsa out", li, s.ofh.ptr, n * d.nh * d.hd)?;
                    }
                    k.lap(4)?;
                }
            }
            if tr {
                trace.push(("attn_out", li, last(s.blk.ptr, d.h)?));
            }
            k.inject(n)?;
            if tr {
                trace.push(("post_attn", li, last(s.hyper.ptr, d.w)?));
            }
            k.hc_mix(&lw.mlp, s.hyper.ptr, n)?;
            if k.range {
                k.range("moe normed", li, s.normed_h.ptr, n * d.w)?;
                if !k.small(n) {
                    k.range("moe low", li, s.lowh.ptr, n * lw.mlp.lr)?;
                }
                k.range("moe folded", li, s.folded_h.ptr, n * d.h)?;
            }
            if tr {
                trace.push(("moe_in", li, last(s.folded.ptr, d.h)?));
            }
            k.lap(2)?;
            k.moe(&lw.moe, layer, li, arena, n, hold)?;
            if tr {
                trace.push(("moe_out", li, last(s.blk.ptr, d.h)?));
            }
            k.inject(n)?;
            if tr {
                trace.push(("post_moe", li, last(s.hyper.ptr, d.w)?));
            }
        }
        *launch_ms = t_run.elapsed().as_secs_f64() * 1e3;
        match out {
            Out::Last(lg) => {
                k.head(head, lm, n, lg)?;
                if tr {
                    trace.push(("head_in", layers.len(), k.grab(s.folded.ptr, d.h)?));
                    trace.push(("logits", layers.len(), lg.clone()));
                }
            }
            Out::Rows(ids) => k.head_rows(head, lm, n, ids)?,
            Out::None => c.sync()?,
        }
        k.lap(9)?;
        if hold {
            *pending = Some((pos0, n));
        } else {
            *pos = pos0 + n;
            if let Some(m) = mtp.as_mut() {
                k.keep_r(m, n - 1, pos0 + n)?;
            }
            k.lean_flip.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(p) = &k.prof {
            let mut p = p.lock().unwrap();
            if hold {
                p.report(arena, format!("{n} rows at {pos0} (window)"));
            } else if n > 1 {
                p.report(arena, format!("{n} tokens at {pos0}"));
            } else {
                p.steps += 1;
                if p.steps >= 64 {
                    let what = format!("{} single tokens to {}", p.steps, pos0 + 1);
                    p.report(arena, what);
                }
            }
        }
        Ok(())
    }
}

impl Dev {
    /// Keep the first `m` rows of the window `spec` ran: the recurrent state
    /// and the histories move on by `m` tokens, and the logits of row `m - 1`
    /// come back. `Ok(false)`: the window is too long for a partial settle
    /// (nothing changed; run the `m` tokens again with `forward`).
    pub(crate) fn settle(&mut self, m: usize, logits: &mut Vec<f32>) -> Res<bool> {
        let Self { k, layers: lws, head, lm, pending, pos, mtp, .. } = self;
        let k = &*k;
        let (d, s, c, q) = (&k.d, &k.s, k.c, k.q);
        let Some((pos0, n)) = pending.take() else {
            return Err("qwen4 cuda: no window to settle".into());
        };
        if m == 0 || m > n || pos0 != *pos {
            return Err(format!("qwen4 cuda: settle {m} of {n} rows at {pos0} (state at {})", *pos));
        }
        let _st = c.st.lock().unwrap();
        c.current()?;
        let part = m < n;
        if part && n > k.tape {
            return Ok(false);
        }
        for lw in lws.iter_mut() {
            if let Mix::Gdn(g) = &mut lw.mix
                && g.dirty
            {
                if part {
                    if d.kk > 1 {
                        let cap = d.kk - 1;
                        go!(c, q.hist, (blocks(cap * d.c_dim), 1), 256;
                            g.ring.ptr, g.t_s1.ptr, g.ring2.ptr, cap as i32, d.c_dim as i32, m as i32);
                    }
                    go!(c, q.gdn_scan, ((d.nv * d.dv).div_ceil(8) as u32, 1), 256;
                        g.t_mix.ptr, g.t_gb.ptr, g.s.ptr, g.s2.ptr, s.qo.ptr, m as i32, d.c_dim as i32, d.nv as i32, d.nk as i32, d.dk as i32, d.dv as i32);
                }
                std::mem::swap(&mut g.s, &mut g.s2);
                if d.kk > 1 {
                    std::mem::swap(&mut g.ring, &mut g.ring2);
                }
                g.dirty = false;
            }
            if let Some(p) = &mut lw.ple
                && p.dirty
            {
                let cap = (d.ple_k - 1) * d.ple_dil;
                if cap > 0 {
                    if part {
                        go!(c, q.hist, (blocks(cap * d.w), 1), 256;
                            p.hist.ptr, p.t_normed.ptr, p.hist2.ptr, cap as i32, d.w as i32, m as i32);
                    }
                    std::mem::swap(&mut p.hist, &mut p.hist2);
                }
                p.dirty = false;
            }
        }
        // the logits of row m - 1
        if n <= SMALL_ROWS {
            logits.clear();
            logits.resize(d.vocab, 0.0);
            unsafe { cu::memcpy_dtoh_async(logits.as_mut_slice(), s.out.ptr + ((m - 1) * d.vocab * 4) as u64, c.stream) }.map_err(derr)?;
            c.sync()?;
        } else {
            k.head(head, lm, m, logits)?;
        }
        *pos = pos0 + m;
        if let Some(h) = mtp.as_mut() {
            k.keep_r(h, m - 1, pos0 + m)?;
        }
        k.lean_flip.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }
}
