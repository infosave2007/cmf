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

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::time::Instant;

use super::{
    Buf, Ctx, DevMat, Gemm, MvJob, Res, bl, compile, cs, ctx, cu, derr, dev_mat, func, mv_shared_bytes, prec, push_job, rows_per_block,
    vram_budget,
};
use crate::linear_core::GdnWeights;
use crate::pipeline::{Act, DenseFfn, MoeFfn};
use crate::qtensor::QTensor;
use crate::qwen4_exp::{Cfg, GatedResidual, Globals, Latent, Layer, Mixer, PleWeights, QsaWeights};

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
/// Picks (token rows x top-k) a latent layer still computes straight from the NVFP4 slots; past
/// them a chunk expands each sub-batch's experts for the batched GEMMs.
const LAT_PICKS: usize = 512;
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
                // CMF_QWEN_DEFS="-DQ4_GW=6 -DQ4_DW=8 ...": compile-time constants of the kernels (see the #ifndef blocks of qwen4.cu)
                // in the fast precision (the default) the expert kernels quantize their inputs to int8 per group of 32
                // and use __dp4a (EXP_DP4A); exact and tf32 keep the f32 dot
                let mut defs: Vec<String> = Vec::new();
                if prec().0 == Gemm::Half {
                    defs.push("-DEXP_DP4A=1".to_string());
                    // and the norms, gates and conv in f32 (F64OPS; CMF_QWEN_DEFS="-DF64OPS=1" keeps f64)
                    defs.push("-DF64OPS=0".to_string());
                }
                if let Ok(v) = std::env::var("CMF_QWEN_DEFS") {
                    defs.extend(v.split_whitespace().map(str::to_string));
                }
                let module = compile(include_str!("qwen4.cu"), defs, c.arch)?;
                Ok(Self { $($name: func(module, concat!("q4_", stringify!($name)))?,)* sm: QSmall::load(module)? })
            }
        }
    };
}
qfuncs!(
    tile, grms, grms_inj, grms_inj4, hc_low, hc_fold, inject, hist, gdn_conv, gdn_prep, gdn_gb, gdn_scan, gdn_scan_c, gdn_out, norm_rope, idx_reduce,
    idx_topk, qsa_softmax, qsa_merge, ple_gate, ple_conv, route, expand_exp, expand_q8, to_wt, zero, gather, act,
    scatter, shared_add, argmax, mtp_fuse, units, lat_gu, lat_dn, lat_sum, lat_expand, lat_act, lat_gmask, lat_mv, lat_put, lat_red,
);
unsafe impl Send for QFuncs {}
unsafe impl Sync for QFuncs {}

/// The kernels of the small path: up to `SMALL_ROWS` token rows straight
/// from the stored weights (int8 bodies, f32 planes, raw expert slots), in
/// f32, with no dense plane and no cuBLAS call.
struct QSmall {
    /// By the number of rows, 1..=SMALL_ROWS.
    mvb: [cs::CUfunction; SMALL_ROWS],
    /// The same matvecs with int8-quantized inputs (`__dp4a`), the fast precision.
    mvq: [cs::CUfunction; SMALL_ROWS],
    /// Whether the matvec of b + 1 rows (f32 / int8) can run 512 threads (its registers allow it).
    mv512: [bool; SMALL_ROWS],
    mvq512: [bool; SMALL_ROWS],
    hc_di: [cs::CUfunction; SMALL_ROWS],
    hc_up: [cs::CUfunction; SMALL_ROWS],
    /// The fold with four streams at once (`hc == 4`, rows of at most 48 words).
    hc_up4: [cs::CUfunction; SMALL_ROWS],
    exp_gu: cs::CUfunction,
    /// The same for q2tp gate/up.
    exp_gu2: cs::CUfunction,
    exp_d: cs::CUfunction,
    /// The down projection of the int8 path, four lanes a row.
    exp_d4: cs::CUfunction,
    /// The 8-bit tubes of a hot file (gate/up, down).
    tube_gu: cs::CUfunction,
    tube_d: cs::CUfunction,
    exp_sum: cs::CUfunction,
    /// Dynamic shared memory a block of the matvec kernels, of the hyper-connection kernels, of the
    /// expert kernels may take (each family has the most its kernels accept).
    mvb_max: usize,
    hc_max: usize,
    ex_max: usize,
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
        let (mvb, mvq, hc_di, hc_up, hc_up4) = (set("mvb")?, set("mvq")?, set("hc_di")?, set("hc_up")?, set("hc_up4_")?);
        let fits512 = |fs: &[cs::CUfunction; SMALL_ROWS]| -> [bool; SMALL_ROWS] {
            let mut v = [false; SMALL_ROWS];
            for (b, &f) in v.iter_mut().zip(fs) {
                *b = unsafe { cu::function::get_function_attribute(f, cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) }
                    .is_ok_and(|t| t >= 512);
            }
            v
        };
        let (mv512, mvq512) = (fits512(&mvb), fits512(&mvq));
        let (exp_gu, exp_gu2, exp_d) = (func(module, "q4_exp_gu")?, func(module, "q4_exp_gu2")?, func(module, "q4_exp_d")?);
        let exp_d4 = func(module, "q4_exp_d4")?;
        let (tube_gu, tube_d) = (func(module, "q4_tube_gu")?, func(module, "q4_tube_d")?);
        // a block stages every row of the round: ask for all a block may have
        let family = |fs: &[cs::CUfunction]| -> usize {
            // the opt-in limit of a block is 227 KB on a datacenter part and 99 KB on a consumer Blackwell
            for want in [200usize << 10, 99 << 10, 96 << 10] {
                let ok = fs.iter().all(|&f| {
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
                    return want;
                }
            }
            48 << 10
        };
        let (mvb_max, ex_max) = (family(&[mvb.as_slice(), mvq.as_slice()].concat()), family(&[exp_gu, exp_gu2, exp_d, exp_d4, tube_gu, tube_d]));
        let hc_max = family(&[hc_di.as_slice(), hc_up.as_slice(), hc_up4.as_slice()].concat());
        Ok(Self { mvb, mvq, mv512, mvq512, hc_di, hc_up, hc_up4, exp_gu, exp_gu2, exp_d, exp_d4, tube_gu, tube_d, exp_sum: func(module, "q4_exp_sum")?, mvb_max, hc_max, ex_max })
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
        let ex = list("CMF_QWEN_CUDA_EXP", &[512, 32, 256, 32]);
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

/// `CMF_QWEN_MV_INT8=1`: the small path's q8_2f matvecs on int8-quantized inputs (`q4_mvq*`, fast precision
/// only; the vocabulary head keeps the f32 dot). Off by default: unlike the expert kernels these matvecs are not
/// bound by their arithmetic, and the window measured the same (12.9 vs 12.9 ms) with 3.13 instead of 3.18
/// tokens a round.
fn mv_int8() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| prec().0 == Gemm::Half && std::env::var("CMF_QWEN_MV_INT8").is_ok_and(|v| v.trim() == "1"))
}

fn grms4() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("CMF_QWEN_GRMS4").is_ok_and(|v| v.trim() == "0"))
}

fn exp_d4_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| prec().0 == Gemm::Half && !std::env::var("CMF_QWEN_EXP_D4").is_ok_and(|v| v.trim() == "0"))
}

/// `CMF_QWEN_MV_WIDE=0`: every small-path matvec on 256 threads.
fn mv_wide() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("CMF_QWEN_MV_WIDE").is_ok_and(|v| v.trim() == "0"))
}

/// Shared memory a q4_mvb* block takes per input row: the side-by-side layout (MV_CONTIG, columns up to MV_CMAX)
/// keeps 20 floats per 16 columns, the transposed one 16 per chunk plus an odd stride.
fn mv_row_bytes(cols: usize) -> usize {
    if cols <= 4096 { (cols >> 4) * 20 * 4 } else { 16 * ((cols >> 4) | 1) * 4 }
}

/// `CMF_QWEN_ZERO_COPY=1`: the experts outside the arena are mapped into the card's address space and read in
/// place over the bus, so the full route runs on the lean path with no host wait (same text as the host-admitted
/// route). Off by default: on a 5090 + EPYC host the bus (52 GB/s) carries about a fifth of the routing of the
/// 256-expert cut and the window took 33-35 ms against 27 ms with host admissions (aquarium 81 vs 86 tok/s);
/// the first request of a server was faster (66 vs 40 tok/s), and a slow host CPU would favour it.
fn zero_copy() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_QWEN_ZERO_COPY").is_ok_and(|v| v.trim() == "1"))
}

/// In a decode step or a verify window, the experts the arena does not hold are computed by the host's cores
/// (the q4tp int8 kernels of the CPU path) while the card runs the ones it holds; only their inputs and outputs
/// cross the bus, a few KB instead of 2.6 MB an expert, and the arena takes only the admissions of
/// `CMF_QWEN_ADMIT`. On by default where the host has the fast kernel (AVX-512 VNNI) and eight threads or
/// more; `CMF_QWEN_CPU_EXPERTS=1` forces it, `=0` keeps every miss an admission.
fn cpu_experts() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("CMF_QWEN_CPU_EXPERTS").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => QTensor::moe_expert_rows_fast_available() && std::thread::available_parallelism().map_or(1, |n| n.get()) >= 8,
    })
}

/// The host's expert workers: `CMF_QWEN_CPU_EXPERT_THREADS`, by default every core the process may use but one
/// (the thread that drives the card), but three from sixteen cores up: a container's CPU quota rounds up to
/// whole cores, and thirty workers on a quota of 31.1 were throttled at every window. They spin `CMF_QWEN_CPU_EXPERT_SPIN` rounds (default 400 000, a few
/// milliseconds) before parking: a layer's misses come every few hundred microseconds, and waking thirty
/// parked threads twice a layer cost more than the experts' arithmetic (83 experts of a window in 37.6 ms).
fn cpu_pool() -> &'static crate::pool::Pool {
    static P: OnceLock<crate::pool::Pool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("CMF_QWEN_CPU_EXPERT_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or_else(|| {
            let n = std::thread::available_parallelism().map_or(1, |n| n.get());
            n.saturating_sub(if n >= 16 { 3 } else { 1 }).max(1)
        });
        let spin = std::env::var("CMF_QWEN_CPU_EXPERT_SPIN").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(400_000);
        eprintln!("qwen4 cuda: {n} host threads compute the experts outside the arena (spin {spin})");
        crate::pool::Pool::with_spin(n, spin)
    })
}

/// `CMF_QWEN_CPU_FAST=0`: the host's experts go through the two row kernels, not the fast kernel (the
/// reference of a measurement).
fn cpu_fast() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("CMF_QWEN_CPU_FAST").is_ok_and(|v| v.trim() == "0"))
}

/// `CMF_QWEN_ADMIT=<n>[:<heat>[:<decay>]]`: with the host computing the arena's misses, up to `n` of a layer's
/// misses a window move into the arena for the windows to come (default 1; 0: the arena stays as the request
/// filled it), the ones picked most lately first: an expert's heat is one a window that picked it, times
/// `decay` (default 0.95) a pass since; `heat` (default 0) is the least that is admitted.
fn admit_policy() -> (usize, f32, f32) {
    static V: OnceLock<(usize, f32, f32)> = OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("CMF_QWEN_ADMIT").unwrap_or_default();
        let mut it = v.split(':').map(str::trim);
        // one: the copies ride the stream the window waits on, and the aquarium within 12 / 16 / 24 GB ran at
        // 145 / 175 / 207 tok/s with one, 139 / 171 / 186 with two, 128 / 142 / 177 with four
        let n = it.next().and_then(|x| x.parse::<usize>().ok()).unwrap_or(1);
        let heat = it.next().and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.0);
        let decay = it.next().and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.95).clamp(0.0, 1.0);
        (n, heat, decay)
    })
}

/// Wait for the stream's queued work. `CMF_QWEN_WAIT=poll` asks the driver in a loop instead of entering its
/// wait: the host path waits once a layer, for work of a few hundred microseconds.
fn wait_stream(c: &Ctx) -> Res<()> {
    static POLL: OnceLock<bool> = OnceLock::new();
    if !*POLL.get_or_init(|| std::env::var("CMF_QWEN_WAIT").is_ok_and(|v| v.trim() == "poll")) {
        return c.sync();
    }
    loop {
        match unsafe { cs::cuStreamQuery(c.stream) }.result() {
            Ok(()) => return Ok(()),
            Err(e) if e.0 == cs::CUresult::CUDA_ERROR_NOT_READY => std::hint::spin_loop(),
            Err(e) => return Err(derr(e)),
        }
    }
}

/// The unweighted down outputs of the experts `miss` for the places `by_expert[e]` (token = place / `kk`),
/// on the host, into `out`: `[place in that order][hid]`. Whole experts in one dispatch at the speed of the
/// memory bus where the cores have AVX-512 VNNI, the two row kernels otherwise. False: a matrix the CPU kernels
/// do not cover.
fn cpu_expert_rows(moe: &MoeFfn, miss: &[usize], by_expert: &[Vec<usize>], kk: usize, xs: &[f32], hid: usize, out: &mut [f32]) -> bool {
    let groups: Vec<Vec<usize>> = miss.iter().map(|&e| by_expert[e].iter().map(|&p| p / kk).collect()).collect();
    let n_pairs: usize = groups.iter().map(|g| g.len()).sum();
    let Some(&first) = miss.first() else { return false };
    if out.len() != n_pairs * hid {
        return false;
    }
    let pool = Some(cpu_pool());
    if cpu_fast() {
        let experts: Vec<(&QTensor, &QTensor, &QTensor)> = miss
            .iter()
            .map(|&e| {
                let x = &moe.experts[e];
                (&x.gate_proj, &x.up_proj, &x.down_proj)
            })
            .collect();
        if QTensor::moe_expert_rows_fast(&experts, &groups, xs, out, pool) {
            return true;
        }
    }
    let pairs: Vec<(&QTensor, &QTensor)> = miss.iter().map(|&e| (&moe.experts[e].gate_proj, &moe.experts[e].up_proj)).collect();
    let inter = moe.experts[first].gate_proj.rows();
    let mut gs: Vec<Vec<f32>> = (0..n_pairs).map(|_| vec![0f32; inter]).collect();
    if !QTensor::moe_gate_up_rows(&pairs, &groups, xs, &mut gs, pool) {
        return false;
    }
    let downs: Vec<&QTensor> = miss.iter().map(|&e| &moe.experts[e].down_proj).collect();
    let lens: Vec<usize> = groups.iter().map(|g| g.len()).collect();
    let mut ds: Vec<Vec<f32>> = (0..n_pairs).map(|_| vec![0f32; hid]).collect();
    if !QTensor::moe_down_rows(&downs, &lens, &gs, &mut ds, pool) {
        return false;
    }
    for (o, d) in out.chunks_exact_mut(hid).zip(&ds) {
        o.copy_from_slice(d);
    }
    true
}

/// How many of the first token ids the draft chooses among. `CMF_QWEN_DRAFT_VOCAB=<n>` fixes it (0 = all). By
/// default it adapts: the first 65 536 ids (the draft's vocabulary matvec reads a quarter of the head) while at
/// most 10 % of the recent tokens lie beyond them (English prose and code: 3-5 %), all ids otherwise (text whose
/// tokens sit high in the vocabulary, Cyrillic or CJK, would never be drafted right).
fn draft_vocab(recent: &[u32], vocab: usize) -> usize {
    const PREFIX: usize = 65536;
    static V: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(n) = *V.get_or_init(|| std::env::var("CMF_QWEN_DRAFT_VOCAB").ok().and_then(|v| v.trim().parse().ok())) {
        return n;
    }
    // the chat template's special tokens sit at the top of the vocabulary and are not text: they do not count
    let top = vocab.saturating_sub(1024);
    let out = recent.iter().filter(|&&t| (PREFIX..top).contains(&(t as usize))).count();
    if recent.len() >= 16 && out * 10 <= recent.len() { PREFIX } else { 0 }
}

/// The gated delta rule over `n` tokens from state `st` into `st2`: a state column per thread when the key
/// width is 128 (`q4_gdn_scan_c`; `CMF_QWEN_SCAN_C=0` keeps the lane-per-four-rows kernel).
#[allow(clippy::too_many_arguments)]
fn scan(c: &Ctx, q: &QFuncs, d: &Dims, mix: u64, gb: u64, st: u64, st2: u64, out: u64, n: usize) -> Res<()> {
    static COL: OnceLock<bool> = OnceLock::new();
    let col = *COL.get_or_init(|| !std::env::var("CMF_QWEN_SCAN_C").is_ok_and(|v| v.trim() == "0"));
    let bt = d.dv.min(128);
    if col && d.dk == 128 && d.dv.is_multiple_of(bt) && d.nk > 0 && d.nv.is_multiple_of(d.nk) {
        go!(c, q.gdn_scan_c, (d.nv as u32, (d.dv / bt) as u32), bt;
            mix, gb, st, st2, out, n as i32, d.c_dim as i32, d.nv as i32, d.nk as i32, d.dk as i32, d.dv as i32);
    } else {
        go!(c, q.gdn_scan, ((d.nv * d.dv).div_ceil(8) as u32, 1), 256;
            mix, gb, st, st2, out, n as i32, d.c_dim as i32, d.nv as i32, d.nk as i32, d.dk as i32, d.dv as i32);
    }
    Ok(())
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

/// Bytes of a q2tp matrix: 8 bytes of weights a group of 32, the rest as q4tp.
fn q2_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / 32) * 8 + rows * 4 + rows * (cols / 32 * 5).div_ceil(8)
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
    /// Per routed expert the index of its 8-bit tube, -1 for none; the shared expert's. A hot expert's
    /// own q2tp triple is zero in the rows the tube holds, so it is not read at all.
    tubes: Vec<i32>,
    shared_tube: i32,
    has_tubes: bool,
}

struct LayerW {
    attn: Hc,
    mlp: Hc,
    mix: Mix,
    ple: Option<PleW>,
    moe: MoeW,
    /// A latent layer's MoE (Edge0); `moe` is then a placeholder.
    lat: Option<LatW>,
}

/// A latent MoE layer on the card: the router (and the shared expert's gate row), D and U, the shared
/// expert as q8 planes, every routed expert resident as an NVFP4 slot, the experts' groups.
struct LatW {
    router: Buf,
    r: usize,
    gcol: i32,
    topk: usize,
    ne: usize,
    dmat: DevMat,
    umat: DevMat,
    sh: [DevMat; 3],
    labels: Buf,
    /// Per expert the address of its slot.
    tbl: Buf,
    tbl_host: Vec<u64>,
    /// Per expert the globals of gate, up and down.
    glob: Vec<[f32; 3]>,
    labels_host: Vec<i32>,
    _experts: Buf,
    l: usize,
    groups: usize,
    ff: usize,
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
    /// Bits of the gate and up weights of an expert: 2 (q2tp) or 4 (q4tp); the down matrix is q4tp.
    gbits: usize,
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
    tick: Buf,
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
    /// Latent layers: D's output `[n][G·L]` (f32 and the GEMM type), the experts' sums `[n][G·L]` (f32 and
    /// the GEMM type), the small path's per-pick act `[rows·topk][ff]` and down `[rows·topk][L]`, the
    /// sub-batch factors.
    lxl: Buf,
    lxlh: Buf,
    lacc: Buf,
    lacch: Buf,
    lact: Buf,
    ly: Buf,
    lf: Buf,
    /// The groups a small round picked (one word).
    lmask: Buf,
    /// U's partial sums by group `[G][n][hidden]`.
    lpart: Buf,
    /// A chunk's per-pick down rows `[n·topk][L]`, and the pick of each sub-batch row.
    lyc: Buf,
    lrp: Buf,
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

/// One expert on its way into the arena off the main stream (`Stagers`).
struct StageJob {
    key: u32,
    slot: u32,
    /// The slot's address on the card.
    dst: u64,
    /// Gate, up and down in the mapped file: address and bytes.
    parts: [(usize, usize); 3],
    /// The file pages are pinned: the card reads them in place.
    direct: bool,
}

struct StageShared {
    q: Mutex<VecDeque<StageJob>>,
    cv: Condvar,
    /// (key, slot, copied) of the experts whose bytes are on the card.
    done: Mutex<Vec<(u32, u32, bool)>>,
    stop: AtomicBool,
    busy_us: AtomicU64,
    ring_jobs: AtomicU64,
}

/// Threads that carry experts into the arena on streams of their own (`CMF_QWEN_STAGERS=<n>`; 0, the
/// default: the admissions ride the main stream, which gave the same aquarium speed within the noise and
/// the same answer from run to run). With the host computing the misses, an expert the windows keep
/// missing is copied behind the decode's back and joins the arena once every byte of it is on the card: the
/// main stream carries no upload and waits for none.
struct Stagers {
    sh: Arc<StageShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// The most experts on their way at once (`CMF_QWEN_STAGE_QUEUE`, default 64).
    cap: usize,
}

impl Stagers {
    fn start(c: &'static Ctx, slot: usize, sgu: usize) -> Option<Self> {
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok());
        let n = env("CMF_QWEN_STAGERS").unwrap_or(0).min(8);
        if n == 0 {
            return None;
        }
        let cap = env("CMF_QWEN_STAGE_QUEUE").unwrap_or(64).max(1);
        let sh = Arc::new(StageShared {
            q: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            done: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            busy_us: AtomicU64::new(0),
            ring_jobs: AtomicU64::new(0),
        });
        let mut threads = Vec::with_capacity(n);
        for _ in 0..n {
            let sh = sh.clone();
            let t = std::thread::Builder::new().name("cmf-qwen-stage".into()).spawn(move || stage_loop(c, sh, slot, sgu)).ok()?;
            threads.push(t);
        }
        Some(Self { sh, threads, cap })
    }
}

impl Drop for Stagers {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Release);
        {
            // under the queue's lock: a stager between its check and its wait must not miss the call
            let _q = self.sh.q.lock().unwrap();
            self.sh.cv.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// A stager: up to `RING` experts a turn go to the card on the thread's own stream. An expert on pinned file
/// pages is read by the card in place; any other passes through a page-locked ring of the thread.
fn stage_loop(c: &'static Ctx, sh: Arc<StageShared>, slot: usize, sgu: usize) {
    const RING: usize = 4;
    if c.current().is_err() {
        return;
    }
    let Ok(stream) = cu::stream::create(cu::stream::StreamKind::NonBlocking) else { return };
    let ring = Pinned::new(RING * slot);
    let offs = [0usize, sgu, 2 * sgu];
    let mut batch: Vec<StageJob> = Vec::with_capacity(RING);
    loop {
        {
            let mut q = sh.q.lock().unwrap();
            while q.is_empty() && !sh.stop.load(Ordering::Acquire) {
                q = sh.cv.wait(q).unwrap();
            }
            if sh.stop.load(Ordering::Acquire) {
                break;
            }
            while batch.len() < RING {
                match q.pop_front() {
                    Some(j) => batch.push(j),
                    None => break,
                }
            }
        }
        let t0 = Instant::now();
        let mut ok = [true; RING];
        for (i, j) in batch.iter().enumerate() {
            // SAFETY: the bytes are the mapped model's, which outlives the stagers (`Dev` stops them first)
            let parts = j.parts.map(|(a, n)| unsafe { std::slice::from_raw_parts(a as *const u8, n) });
            let fits = parts[0].len() <= sgu && parts[1].len() <= sgu && 2 * sgu + parts[2].len() <= slot;
            match &ring {
                Some(r) if !j.direct && fits => {
                    // SAFETY: slot `i` of the ring is this turn's alone, and the stream is drained before the next turn
                    let src = unsafe {
                        let at = r.ptr.add(i * slot);
                        for (p, o) in parts.iter().zip(offs) {
                            std::ptr::copy_nonoverlapping(p.as_ptr(), at.add(o), p.len());
                        }
                        std::slice::from_raw_parts(at as *const u8, 2 * sgu + parts[2].len())
                    };
                    ok[i] = unsafe { cu::memcpy_htod_async(j.dst, src, stream) }.is_ok();
                    sh.ring_jobs.fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    for (p, o) in parts.iter().zip(offs) {
                        ok[i] &= unsafe { cu::memcpy_htod_async(j.dst + o as u64, *p, stream) }.is_ok();
                    }
                }
            }
        }
        let landed = stage_wait(stream);
        sh.busy_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        let mut done = sh.done.lock().unwrap();
        for (i, j) in batch.drain(..).enumerate() {
            done.push((j.key, j.slot, ok[i] && landed));
        }
    }
    let _ = unsafe { cu::stream::synchronize(stream) };
    drop(ring);
    let _ = unsafe { cu::stream::destroy(stream) };
}

/// Wait for a stager's stream by polling: the thread sleeps between looks and spins no core.
fn stage_wait(stream: cs::CUstream) -> bool {
    loop {
        match unsafe { cs::cuStreamQuery(stream) }.result() {
            Ok(()) => return true,
            Err(e) if e.0 == cs::CUresult::CUDA_ERROR_NOT_READY => std::thread::sleep(std::time::Duration::from_micros(50)),
            Err(_) => return false,
        }
    }
}

/// Windows and drafts an expert must have sat out, when its layer came up, before a background admission
/// takes its slot (`CMF_QWEN_STAGE_AGE`, default 2: the layer's last visit went without it).
fn stage_age() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_QWEN_STAGE_AGE").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(2).max(1))
}

/// The host's scratch of a layer (`CMF_QWEN_CPU_EXPERTS`): the picks and the experts' input come down into it
/// and the missed experts' rows go up from it. Page-locked when the driver gives the memory (a transfer is then
/// a plain DMA the calling thread does not wait for), ordinary memory otherwise.
struct HostScratch {
    pin: Option<Pinned>,
    plain: Vec<u64>,
    /// Bytes of the three parts: picks, input rows, output rows.
    cap: [usize; 3],
}

impl HostScratch {
    fn new() -> Self {
        Self { pin: None, plain: Vec::new(), cap: [0; 3] }
    }

    /// Room for `picks` ids, `xs` input and `rows` output values: where the three parts start. `drain` runs
    /// before a block in use is replaced (a transfer from it may still be under way).
    fn parts(&mut self, picks: usize, xs: usize, rows: usize, drain: impl FnOnce() -> Res<()>) -> Res<(*mut i32, *mut f32, *mut f32)> {
        let need = [picks, xs, rows].map(|n| (n * 4).next_multiple_of(64));
        if need.iter().zip(&self.cap).any(|(n, c)| n > c) {
            drain()?;
            let cap = [need[0].max(self.cap[0]), need[1].max(self.cap[1]), need[2].max(self.cap[2])];
            let total: usize = cap.iter().sum();
            self.pin = Pinned::new(total);
            self.plain = if self.pin.is_none() { vec![0u64; total / 8] } else { Vec::new() };
            self.cap = cap;
        }
        let base = match &self.pin {
            Some(p) => p.ptr,
            None => self.plain.as_mut_ptr() as *mut u8,
        };
        // SAFETY: the three parts lie inside the block of `cap[0] + cap[1] + cap[2]` bytes
        Ok(unsafe { (base as *mut i32, base.add(self.cap[0]) as *mut f32, base.add(self.cap[0] + self.cap[1]) as *mut f32) })
    }
}

/// The file pages of the experts the arena keeps losing, pinned where they are
/// (`cuMemHostRegister`, read-only): an upload is one DMA straight from the
/// mapped file, with no copy through the staging first (RTX 5090: 19 us for
/// 880 KB against 54 us from pageable memory, and none of the 160 us copy).
/// Pinned pages cannot be dropped from the page cache, so only the experts
/// outside the prefilled arena are taken. A thread pins layer after layer;
/// `ready[li]` says when a layer's are pinned.
/// Host pages that stay in RAM for good: page-aligned spans of a read-only mapping registered with the driver
/// (`cuMemHostRegister`, read-only) from a background thread, in pieces of 1 GiB. A container that refuses
/// `mlock` (RLIMIT_MEMLOCK, no CAP_IPC_LOCK) still lets the driver pin. Returns at once; never unregistered.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
pub(crate) fn pin_host_for_good(name: &'static str, spans: Vec<(usize, usize)>) {
    let Some(c) = ctx() else { return };
    let _ = std::thread::Builder::new().name("pin-host".into()).spawn(move || {
        if c.current().is_err() {
            return;
        }
        let t0 = Instant::now();
        let (mut pinned, mut failed) = (0usize, 0usize);
        for (a, len) in spans {
            let mut at = a;
            let end = a + len;
            while at < end {
                let n = (end - at).min(1 << 30);
                // flag 8: the pages are read-only to the card
                let rc = unsafe { cs::cuMemHostRegister_v2(at as *mut c_void, n, 8) };
                if rc == cs::cudaError_enum::CUDA_SUCCESS {
                    pinned += n;
                } else {
                    failed += n;
                }
                at += n;
            }
        }
        eprintln!(
            "qwen4 cuda: {name}: {:.1} GB pinned in RAM ({:.1} GB refused) in {:.1} s",
            pinned as f64 / 1e9,
            failed as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
    });
}

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
    fn start(c: &'static Ctx, sel: Vec<Vec<bool>>, ranges: Vec<Vec<(usize, usize)>>, wait: bool) -> Self {
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
                    // flag 8: the pages are read-only to the card; 2: mapped into its address space (zero copy)
                    let rc = unsafe { cs::cuMemHostRegister_v2(at as *mut c_void, len, if zero_copy() { 8 | 2 } else { 8 }) };
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
            if wait || std::env::var_os("CMF_QWEN_PROF").is_some() {
                eprintln!("qwen4 cuda: pinned {:.2} GiB of expert pages in {:.1} s", bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
            }
        };
        let thread = std::thread::Builder::new().name("cmf-qwen-pin".into()).spawn(run).ok();
        // the thread binds the context itself; joined here, the pins are in place before the first request
        let thread = if wait {
            if let Some(t) = thread {
                let _ = t.join();
            }
            None
        } else {
            thread
        };
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
    /// The background admissions' threads; dropped first, ahead of the arena's memory.
    stagers: Option<Stagers>,
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
    /// The heat of every backbone expert the host computed, `[layer][expert]`, as of the pass `heat_at`
    /// (`admit_hot`), and the hot misses left on the host for want of pinned pages.
    heat: Vec<f32>,
    heat_at: Vec<u64>,
    unpinned: u64,
    /// Experts on their way in through the stagers (their slots hold no key yet), the slots a failed or
    /// overtaken copy gave back, and the stagers' own list of runners-up for eviction.
    inflight: HashSet<u32>,
    free: Vec<u32>,
    bg_spare: Vec<(u32, u32, u64)>,
    /// (pass, layer): no slot was to be had; the layers before this one do not look again in that pass.
    bg_dry: (u64, usize),
    staged: u64,
    landed: u64,
    queue_full: u64,
    /// The MTP head's experts take slots by use as the backbone's do, the host computing the absent ones
    /// (`CMF_QWEN_HEAD_HOST=1`).
    head_host: bool,
    /// Zero-copy experts (`CMF_QWEN_ZERO_COPY`, on by default with pinned experts): the host address of every
    /// expert outside the arena whose three matrices lie in the file as a slot does (0 otherwise), and, once its
    /// layer is pinned, the device's address for it. The address table then names them, so a route over all the
    /// experts runs on the lean path and the kernels read the missing ones over the bus: nothing waits for the host.
    zc: Vec<Vec<[u64; 3]>>,
    zc_dev: Vec<Vec<u64>>,
    /// The three device addresses {gate, up, down} of each zero-copy expert, `[layer][expert]`; a table entry
    /// names its triple with bit 63 set (`unit_parts` in the kernels).
    zc_dsc: Option<Buf>,
    zc_ready: Vec<bool>,
    /// Every backbone layer pinned and mapped: the route covers all experts through the table.
    zc_full: bool,
    /// Host-computed expert rows, experts and milliseconds (`CMF_QWEN_CPU_EXPERTS`), since the last report.
    cpu_rows: u64,
    cpu_experts: u64,
    cpu_ms: f64,
    /// The host path's time since the last report, by phase: the launches from the layer before to the route,
    /// the wait for the picks, the units and their launches, the admissions, the rows' upload and the sum.
    host_ms: [f64; 5],
    host_layers: u64,
    /// The layer whose host path returned last, and when.
    host_at: Option<(usize, Instant)>,
    /// Where the host's share of a layer meets the card.
    host: HostScratch,
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
        let zc_changed = self.zc_poll();
        if self.tbl_epoch == self.epoch && !zc_changed {
            return Ok(());
        }
        let mut host = vec![0u64; self.tbl_ne.len() * self.tbl_stride];
        for (li, &ne) in self.tbl_ne.iter().enumerate() {
            let mut held = 0usize;
            for e in 0..ne {
                if let Some(&s) = self.map.get(&(((li as u32) << 16) | e as u32)) {
                    host[li * self.tbl_stride + e] = self.buf.ptr + (s as usize * self.slot) as u64;
                    held += 1;
                } else if let Some(&a) = self.zc_dev.get(li).and_then(|v| v.get(e))
                    && a != 0
                {
                    host[li * self.tbl_stride + e] = a;
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

    /// Map the experts of the layers pinned since the last call (their device addresses); true when any changed.
    fn zc_poll(&mut self) -> bool {
        let Some(pins) = self.pins.as_ref() else { return false };
        let mut changed = false;
        let (mut want, mut got, mut first_err) = (0usize, 0usize, None);
        for li in 0..self.zc.len() {
            if self.zc_ready[li] || !pins.ready.get(li).is_some_and(|f| f.load(Ordering::Acquire)) {
                continue;
            }
            let ne = self.zc[li].len();
            let mut triples = vec![0u64; ne * 3];
            for (e, parts) in self.zc[li].iter().enumerate() {
                if parts[0] == 0 || !pins.sel[li].get(e).copied().unwrap_or(false) {
                    continue;
                }
                want += 1;
                let mut ok = true;
                for (k, &a) in parts.iter().enumerate() {
                    let mut dp: cs::CUdeviceptr = 0;
                    // SAFETY: a host address inside a range registered with the driver (DEVICEMAP)
                    let rc = unsafe { cs::cuMemHostGetDevicePointer_v2(&mut dp, a as *mut c_void, 0) };
                    if rc == cs::cudaError_enum::CUDA_SUCCESS && dp != 0 {
                        triples[e * 3 + k] = dp;
                    } else {
                        ok = false;
                        if first_err.is_none() {
                            first_err = Some(rc);
                        }
                    }
                }
                if ok {
                    got += 1;
                } else {
                    triples[e * 3..e * 3 + 3].fill(0);
                }
            }
            if let Some(dsc) = self.zc_dsc.as_ref() {
                let at = dsc.ptr + (li * ne * 24) as u64;
                // SAFETY: the triples of layer li go to their rows of the descriptor buffer; the copy is synchronous
                if unsafe { cu::memcpy_htod_sync(at, &triples) }.is_ok() {
                    for e in 0..ne {
                        if triples[e * 3] != 0 {
                            self.zc_dev[li][e] = (at + (e * 24) as u64) | (1u64 << 63);
                        }
                    }
                }
            }
            self.zc_ready[li] = true;
            changed = true;
        }
        if want > 0 && got < want {
            eprintln!("qwen4 cuda: zero copy: {got} of {want} experts mapped for the card (first error {first_err:?})");
        }
        if changed {
            let main = self.main_layers().min(self.zc_ready.len());
            // only a layer whose every expert is on the card or mapped makes a full route
            let full = main > 0
                && self.zc_ready[..main].iter().all(|&r| r)
                && (0..main).all(|li| {
                    (0..self.zc_dev[li].len()).all(|e| self.zc_dev[li][e] != 0 || self.map.contains_key(&(((li as u32) << 16) | e as u32)))
                });
            if full && !self.zc_full {
                eprintln!("qwen4 cuda: every expert outside the arena is mapped for the card: the route covers them all on the lean path");
            }
            self.zc_full = full;
        }
        changed
    }

    /// The table row of layer `li` as a device address, when the route may use it: the
    /// route is masked and the layer holds more than `topk` experts, or `force` (the MTP
    /// head, whose experts all stay). 0: no.
    fn tbl_ptr(&mut self, c: &Ctx, li: usize, topk: usize, force: bool) -> Res<u64> {
        if !self.zc.is_empty() && !self.zc_full {
            self.refresh_table(c)?;
        }
        if li >= self.tbl_ne.len() || !(self.masked || force || self.zc_full) {
            return Ok(0);
        }
        self.refresh_table(c)?;
        if !force && self.tbl_held[li] <= topk {
            return Ok(0);
        }
        Ok(self.tbl.ptr + (li * self.tbl_stride * 8) as u64)
    }

    /// With every expert outside the arena mapped for the card (zero copy), the slots of `order` without an
    /// admission: an arena slot, or the expert's descriptor (the card reads it from its pinned file pages). The
    /// arena then stays as the profile filled it. False when not in that mode or an expert is neither: then
    /// `admit_layer`, which rewrites every entry of `order`.
    fn resolve_layer(&mut self, li: usize, order: &[usize], slot_of: &mut [u64]) -> bool {
        if !self.zc_full {
            return false;
        }
        let pass = self.pass;
        for &e in order {
            match self.map.get(&(((li as u32) << 16) | e as u32)) {
                Some(&s) => {
                    self.last[s as usize] = pass;
                    slot_of[e] = self.buf.ptr + (s as usize * self.slot) as u64;
                }
                None => match self.zc_dev.get(li).and_then(|v| v.get(e)) {
                    Some(&a) if a != 0 => slot_of[e] = a,
                    _ => return false,
                },
            }
        }
        true
    }

    /// The slot of expert `e` of layer `li` when the arena holds it (marked used this pass); None: a miss.
    fn hit(&mut self, li: usize, e: usize) -> Option<u64> {
        let &s = self.map.get(&(((li as u32) << 16) | e as u32))?;
        self.last[s as usize] = self.pass;
        Some(self.buf.ptr + (s as usize * self.slot) as u64)
    }

    /// Whether the arena may give up the expert in a slot (`key`): always, unless the route covers every expert
    /// through zero copy, which holds only while each evicted expert stays mapped for the card.
    fn evictable(&self, key: u32) -> bool {
        if !self.zc_full {
            return true;
        }
        let (l, e) = ((key >> 16) as usize, (key & 0xffff) as usize);
        self.zc_dev.get(l).and_then(|v| v.get(e)).is_some_and(|&a| a != 0)
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

    /// The experts the stagers have put on the card join the arena. One the main stream admitted meanwhile
    /// keeps its slot, and the copy's slot goes back to the free ones.
    fn land(&mut self) {
        let Some(st) = &self.stagers else { return };
        if self.inflight.is_empty() {
            return;
        }
        let done = std::mem::take(&mut *st.sh.done.lock().unwrap());
        if done.is_empty() {
            return;
        }
        for (key, s, ok) in done {
            self.inflight.remove(&key);
            let su = s as usize;
            if ok && !self.map.contains_key(&key) {
                self.key[su] = key;
                self.last[su] = self.pass;
                self.prot[su] = false;
                self.map.insert(key, s);
                self.admitted += 1;
                self.landed += 1;
            } else {
                self.free.push(s);
            }
        }
        self.epoch += 1;
    }

    /// A slot for an expert on its way in at layer `li`: a free one, else the slot of the expert unused the
    /// longest among those that sat out `min_age` visits of their layer. Its expert leaves the arena at once;
    /// the slot holds no key until the copy lands. None: the windows use every slot.
    fn victim(&mut self, li: usize, min_age: u64) -> Option<u32> {
        if let Some(s) = self.free.pop() {
            return Some(s);
        }
        if self.used < self.n {
            self.used += 1;
            return Some(self.used as u32 - 1);
        }
        while let Some((s, key, last)) = self.bg_spare.pop() {
            let su = s as usize;
            if self.key[su] == key && self.last[su] == last && !self.prot[su] && self.evictable(key) {
                self.map.remove(&key);
                self.key[su] = u32::MAX;
                self.evicted += 1;
                self.epoch += 1;
                return Some(s);
            }
        }
        let pass = self.pass;
        if self.bg_dry.0 == pass && li < self.bg_dry.1 {
            return None;
        }
        let (nl, lu) = (self.nl, li as u64);
        let mut cand: Vec<(u64, u32)> = Vec::new();
        for s in 0..self.n {
            let key = self.key[s];
            if key == u32::MAX || self.prot[s] || (key >> 16) >= self.sticky_from || !self.evictable(key) {
                continue;
            }
            let (j, lp) = ((key >> 16) as u64, self.last[s]);
            // passes the expert went unused when its layer came up: one this pass has used (a queued kernel
            // may still read it) counts none
            let age = if j <= lu { pass - lp } else { (pass - lp).saturating_sub(1) };
            if age < min_age {
                continue;
            }
            let dist = if j > lu { j - lu } else { nl - lu + j };
            cand.push((age * nl + dist, s as u32));
        }
        if cand.is_empty() {
            self.bg_dry = (pass, li + 4);
            return None;
        }
        let take = (1 + SPARE).min(cand.len());
        if take < cand.len() {
            cand.select_nth_unstable_by(take - 1, |a, b| b.cmp(a));
        }
        cand.truncate(take);
        cand.sort_unstable_by(|a, b| b.cmp(a));
        self.bg_spare.clear();
        for &(_, t) in cand[1..].iter().rev() {
            self.bg_spare.push((t, self.key[t as usize], self.last[t as usize]));
        }
        let su = cand[0].1 as usize;
        self.map.remove(&self.key[su]);
        self.key[su] = u32::MAX;
        self.evicted += 1;
        self.epoch += 1;
        Some(cand[0].1)
    }

    /// Hand the layer's experts `order` to the stagers, each with the slot it will land in.
    fn admit_bg(&mut self, li: usize, order: &[usize], experts: &[DenseFfn], d: &Dims) -> Res<()> {
        let Some(cap) = self.stagers.as_ref().map(|s| s.cap) else { return Ok(()) };
        let min_age = stage_age();
        let mut jobs: Vec<StageJob> = Vec::new();
        for &e in order {
            if self.inflight.len() >= cap {
                self.queue_full += 1;
                break;
            }
            let key = ((li as u32) << 16) | e as u32;
            if self.inflight.contains(&key) || self.map.contains_key(&key) {
                continue;
            }
            let Some(f) = experts.get(e) else { continue };
            let parts = expert_parts(f, d)?;
            let direct = self.pins.as_ref().is_some_and(|p| p.has(li, e));
            let Some(s) = self.victim(li, min_age) else { break };
            self.inflight.insert(key);
            self.staged += 1;
            jobs.push(StageJob { key, slot: s, dst: self.buf.ptr + (s as usize * self.slot) as u64, parts: parts.map(|p| (p.as_ptr() as usize, p.len())), direct });
        }
        if jobs.is_empty() {
            return Ok(());
        }
        let Some(st) = &self.stagers else { return Ok(()) };
        let many = jobs.len() > 1;
        st.sh.q.lock().unwrap().extend(jobs);
        if many {
            st.sh.cv.notify_all();
        } else {
            st.sh.cv.notify_one();
        }
        Ok(())
    }

    /// With the host computing the misses: the hottest of the layer's misses `miss`, up to the budget of
    /// `admit_policy`, go into the arena for the windows to come, in place of the experts unused the longest.
    /// The stagers carry them on streams of their own (`admit_bg`). Without the stagers the copies ride the
    /// main stream behind the layer's queued kernels while the host computes this window's rows, so only
    /// experts whose file pages are pinned are taken: one of them costs this thread three calls.
    fn admit_hot(&mut self, li: usize, miss: &[usize], experts: &[DenseFfn], d: &Dims, c: &Ctx) -> Res<()> {
        let (budget, least, decay) = admit_policy();
        let ne = self.hist_ne;
        let main = self.main_layers();
        // the heat's rows: the backbone's layers, then the MTP head's
        let (off, lne) = if li < main {
            (li * ne, ne)
        } else if li == main {
            (main * ne, self.tbl_ne.get(li).copied().unwrap_or(0))
        } else {
            return Ok(());
        };
        if budget == 0 || self.zc_full || off + lne > self.heat.len() {
            return Ok(());
        }
        let pass = self.pass;
        let mut cand: Vec<(f32, usize)> = Vec::with_capacity(miss.len());
        for &e in miss.iter().filter(|&&e| e < lne) {
            let i = off + e;
            let idle = pass.saturating_sub(self.heat_at[i]).min(512) as i32;
            let h = self.heat[i] * decay.powi(idle) + 1.0;
            (self.heat[i], self.heat_at[i]) = (h, pass);
            if h >= least {
                cand.push((h, e));
            }
        }
        // the head's experts stay; what is left has to outnumber a layer's picks by far
        let sticky = if self.sticky_from == u32::MAX { 0 } else { self.tbl_ne.get(main).copied().unwrap_or(0) };
        if cand.is_empty() || self.n < sticky + 4 * ne.min(64) + budget {
            return Ok(());
        }
        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        if self.stagers.is_some() {
            let order: Vec<usize> = cand.iter().take(budget).map(|&(_, e)| e).collect();
            return self.admit_bg(li, &order, experts, d);
        }
        let mut order: Vec<usize> = Vec::with_capacity(budget);
        for &(_, e) in &cand {
            if order.len() == budget {
                break;
            }
            if self.pins.as_ref().is_some_and(|p| p.has(li, e)) {
                order.push(e);
            } else {
                self.unpinned += 1;
            }
        }
        if order.is_empty() {
            return Ok(());
        }
        let mut slot_of = vec![0u64; lne];
        self.admit_layer(li, &order, experts, d, c, &mut slot_of)
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
        // the slots the background admissions gave back
        while slots.len() < miss.len() {
            let Some(s) = self.free.pop() else { break };
            slots.push(s);
        }
        while slots.len() < miss.len() && self.used < self.n {
            slots.push(self.used as u32);
            self.used += 1;
        }
        let mut need = miss.len() - slots.len();
        while need > 0 {
            let Some((s, key, last)) = self.spare.pop() else { break };
            if self.key[s as usize] != key || self.last[s as usize] != last || !self.evictable(key) {
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
                    if (key >> 16) >= self.sticky_from || (j == lu && lp == pass) || !self.evictable(key) {
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
            if cand.len() < need && self.zc_full {
                // zero copy: what finds no slot stays where the card reads it in place
                for &e in &miss[slots.len() + cand.len()..] {
                    slot_of[e] = self.zc_dev[li][e];
                }
                miss.truncate(slots.len() + cand.len());
                need = cand.len();
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
    if f.segs.len() > 1 {
        return Err("qwen4 cuda: an expert with task tubes".into());
    }
    let part = |t: &'a QTensor, rows: usize, cols: usize, len: usize, bits: usize| -> Res<&'a [u8]> {
        let kind = if bits == 2 { t.mapped_q2tp().is_some() } else { t.mapped_q4tp().is_some() };
        if t.rows() != rows || t.cols() != cols || !kind {
            return Err(format!("qwen4 cuda: an expert matrix is not q{bits}tp {rows}x{cols}"));
        }
        match t.mapped_bytes() {
            Some(b) if b.len() == len => Ok(b),
            Some(b) => Err(format!("qwen4 cuda: an expert matrix of {} bytes, {len} expected", b.len())),
            None => Err("qwen4 cuda: an expert matrix without file bytes".into()),
        }
    };
    Ok([
        part(&f.gate_proj, d.inter, d.h, d.sgu, d.gbits)?,
        part(&f.up_proj, d.inter, d.h, d.sgu, d.gbits)?,
        part(&f.down_proj, d.h, d.inter, d.sd, 4)?,
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
    /// The 8-bit tubes of a hot file: nine words each (body, row scale, column field of gate, up, down),
    /// and the matrices they point into.
    tube_desc: Vec<u64>,
    tube_keep: Vec<DevMat>,
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

    /// `elems` zeroed f32 in an allocation of the driver's own: the attention
    /// state, replaced by a larger one as a session grows.
    fn zeroed_direct(&mut self, elems: usize) -> Res<Buf> {
        let b = Buf::direct(elems * 4)?;
        self.held += b.bytes as u64;
        go!(self.c, self.q.zero, (blocks(elems), 1), 256; b.ptr, elems as u64);
        Ok(b)
    }

    fn f32buf_u64(&mut self, d: &[u64]) -> Res<Buf> {
        let b = Buf::from_slice(d)?;
        self.held += b.bytes as u64;
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
            lw: LayerW { attn, mlp, mix, ple: None, moe, lat: None },
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
            keys: self.zeroed_direct(cap * kv)?,
            vals: self.zeroed_direct(cap * kv)?,
            ikraw: self.zeroed_direct(cap * d.idim)?,
            bkey: self.zeroed_direct((cap / d.cr + 1) * d.idim)?,
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

    /// The expert's 8-bit tube on the card (`hot-bake`: gate, up and down at eight bits, the expert's own
    /// rows zero): its index, -1 for an expert without one.
    fn tube_of(&mut self, f: &DenseFfn, d: &Dims) -> Res<i32> {
        match f.segs.as_slice() {
            [] => Ok(-1),
            [sg] => {
                if sg.width != d.inter || sg.gate.rows() != d.inter || sg.up.rows() != d.inter || sg.gate.cols() != d.h || sg.up.cols() != d.h
                    || sg.down.rows() != d.h || sg.down.cols() != d.inter
                {
                    return Err("qwen4 cuda: a tube that is not a whole expert".into());
                }
                let mut desc = [0u64; 9];
                for (i, t) in [&sg.gate, &sg.up, &sg.down].into_iter().enumerate() {
                    if t.mapped_q8_2f().is_none() {
                        return Err("qwen4 cuda: a tube that is not q8_2f".into());
                    }
                    let bytes = t.mapped_bytes().ok_or_else(|| "qwen4 cuda: a tube without file bytes".to_string())?;
                    let m = dev_mat(7, bytes, t.rows(), t.cols())?;
                    self.held += m.held;
                    desc[3 * i] = m.w.ptr;
                    desc[3 * i + 1] = m.aux0();
                    desc[3 * i + 2] = m.aux1();
                    self.tube_keep.push(m);
                }
                self.tube_desc.extend(desc);
                Ok((self.tube_desc.len() / 9 - 1) as i32)
            }
            _ => Err("qwen4 cuda: an expert with more than one tube".into()),
        }
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
        let mut tubes = vec![-1i32; ne];
        for (e, f) in m.experts.iter().enumerate() {
            tubes[e] = self.tube_of(f, d)?;
        }
        let shared_tube = match &m.shared {
            Some((f, _)) => self.tube_of(f, d)?,
            None => -1,
        };
        let has_tubes = shared_tube >= 0 || tubes.iter().any(|&t| t >= 0);
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
        Ok(MoeW { router: self.f32buf(&host_rows(&ts))?, shared, gcol, r, topk: m.top_k.min(ne), ne, tubes, shared_tube, has_tubes })
    }

    /// A latent layer: every routed expert goes to the card whole, NVFP4 as the file has it, each matrix
    /// padded to 16 bytes in its slot.
    fn moe_lat(&mut self, m: &MoeFfn, lat: &Latent, d: &Dims) -> Res<(LatW, MoeW)> {
        let ne = m.experts.len();
        if !moe_ok(m) || m.router.rows() != ne || m.router.cols() != d.h {
            return Err("qwen4 cuda: a latent router this path does not reproduce".into());
        }
        let (l, groups) = (lat.dim, lat.groups);
        let ff = m.experts.first().map_or(0, |e| e.gate_proj.rows());
        if ff == 0 || !ff.is_multiple_of(16) || !l.is_multiple_of(16) {
            return Err("qwen4 cuda: latent expert shapes".into());
        }
        let q8 = |t: &QTensor, what: &str| -> Res<DevMat> {
            if t.mapped_q8_2f().is_none() || !t.cols().is_multiple_of(16) {
                return Err(format!("qwen4 cuda: latent {what} is not q8_2f"));
            }
            let bytes = t.mapped_bytes().ok_or_else(|| format!("qwen4 cuda: latent {what} has no file bytes"))?;
            dev_mat(7, bytes, t.rows(), t.cols())
        };
        let dmat = q8(&lat.down, "D")?;
        let umat = q8(&lat.up, "U")?;
        let (shf, gate) = m.shared.as_ref().ok_or_else(|| "qwen4 cuda: a latent layer without a shared expert".to_string())?;
        let sh = [q8(&shf.gate_proj, "shared gate")?, q8(&shf.up_proj, "shared up")?, q8(&shf.down_proj, "shared down")?];
        for mm in [&dmat, &umat].into_iter().chain(sh.iter()) {
            self.held += mm.held;
            self.wq_elems = self.wq_elems.max(mm.rows * mm.cols);
        }
        let pad = |n: usize| n.div_ceil(16) * 16;
        let pg = cortiq_core::nvfp4::nbytes(ff, l).ok_or("qwen4 cuda: latent expert size")?;
        let pd = cortiq_core::nvfp4::nbytes(l, ff).ok_or("qwen4 cuda: latent expert size")?;
        let slot = 2 * pad(pg) + pad(pd);
        let buf = self.buf(ne * slot)?;
        let mut stage = vec![0u8; slot * 64];
        let mut glob = Vec::with_capacity(ne);
        for (c0, chunk) in m.experts.chunks(64).enumerate() {
            stage.iter_mut().for_each(|b| *b = 0);
            for (j, e) in chunk.iter().enumerate() {
                let mut gl = [0f32; 3];
                for (k, (t, rows, cols, len)) in
                    [(&e.gate_proj, ff, l, pg), (&e.up_proj, ff, l, pg), (&e.down_proj, l, ff, pd)].into_iter().enumerate()
                {
                    let ok = t.rows() == rows && t.cols() == cols && t.model_dtype() == Some(cortiq_core::TensorDtype::Nvfp4);
                    let bytes = t.mapped_bytes().filter(|b| ok && b.len() == len).ok_or_else(|| {
                        format!("qwen4 cuda: latent expert {} matrix {k} is not nvfp4 {rows}x{cols}", c0 * 64 + j)
                    })?;
                    let at = j * slot + k * pad(pg);
                    stage[at..at + len].copy_from_slice(bytes);
                    gl[k] = cortiq_core::nvfp4::global(bytes, rows, cols);
                }
                glob.push(gl);
            }
            let dst = buf.ptr + (c0 * 64 * slot) as u64;
            unsafe { cu::memcpy_htod_sync(dst, &stage[..chunk.len() * slot]) }.map_err(derr)?;
        }
        let tbl_host: Vec<u64> = (0..ne).map(|e| buf.ptr + (e * slot) as u64).collect();
        let labels_host: Vec<i32> = lat.labels.iter().map(|&g| g as i32).collect();
        let tbl = self.f32buf_u64(&tbl_host)?;
        let labels = Buf::from_slice(&labels_host)?;
        let mut ts = vec![&m.router];
        let gcol = match gate {
            Some(g) if g.rows() == 1 && g.cols() == d.h => {
                ts.push(g);
                ne as i32
            }
            Some(_) => return Err("qwen4 cuda: unexpected shared-expert gate".into()),
            None => -1,
        };
        let r = ts.iter().map(|t| t.rows()).sum();
        let router = self.f32buf(&host_rows(&ts))?;
        let topk = m.top_k.min(ne);
        let lw = LatW {
            router,
            r,
            gcol,
            topk,
            ne,
            dmat,
            umat,
            sh,
            labels,
            tbl,
            tbl_host,
            glob,
            labels_host,
            _experts: buf,
            l,
            groups,
            ff,
        };
        // the placeholder of the layer's MoeW: nothing of it is read
        let ph = MoeW { router: self.buf(16)?, shared: None, gcol: -1, r, topk, ne, tubes: vec![-1; ne], shared_tube: -1, has_tubes: false };
        Ok((lw, ph))
    }
}

/// Time by stage (with `CMF_QWEN_PROF`): each lap waits for the stream, unless `sync` is off.
struct Prof {
    t0: Instant,
    acc: [f64; STAGES.len()],
    /// Single-token calls since the last report.
    steps: usize,
    /// Wait for the stream at every stage (`CMF_QWEN_PROF=2` does not: the window then runs as it does
    /// without the profile, and only the host path's own clock means anything).
    sync: bool,
}

const STAGES: [&str; 10] = ["io", "ple", "hc", "gdn", "qsa", "route", "admit", "experts", "shared", "head"];

impl Prof {
    /// Print the stage times gathered since the last report, and start over.
    fn report(&mut self, arena: &mut Arena, what: String) {
        let line = STAGES.iter().zip(&self.acc).map(|(name, ms)| format!("{name} {ms:.1}")).collect::<Vec<_>>().join(" ");
        let host = if arena.cpu_rows > 0 {
            format!("; host experts {} ({} rows) in {:.1} ms", arena.cpu_experts, arena.cpu_rows, arena.cpu_ms)
        } else {
            String::new()
        };
        eprintln!(
            "qwen4 cuda: {what}: {line} ms; arena +{} -{} ({} hot misses without pinned pages), staged in {:.1} ms, {} units for {} picks{host}",
            arena.admitted, arena.evicted, arena.unpinned, arena.stage_ms, arena.units, arena.pairs
        );
        if arena.host_layers > 0 {
            let h = arena.host_ms;
            eprintln!(
                "qwen4 cuda: host path, {} layers: launches up to the route {:.2}, wait for the picks {:.2}, units and launches {:.2}, admissions {:.2}, kernel {:.2}, rows up and sum {:.2} ms",
                arena.host_layers, h[0], h[1], h[2], h[3], arena.cpu_ms, h[4]
            );
        }
        if let Some(st) = &arena.stagers {
            let (busy, ring) = (st.sh.busy_us.swap(0, Ordering::Relaxed), st.sh.ring_jobs.swap(0, Ordering::Relaxed));
            eprintln!(
                "qwen4 cuda: host path admissions off the stream: {} sent, {} landed ({} through the ring), {} on their way, {} layers met a full queue, stagers busy {:.1} ms",
                arena.staged,
                arena.landed,
                ring,
                arena.inflight.len(),
                arena.queue_full,
                busy as f64 / 1e3
            );
        }
        (arena.staged, arena.landed, arena.queue_full) = (0, 0, 0);
        if let Some(k) = QTensor::moe_expert_rows_fast_prof_take() {
            eprintln!("qwen4 cuda: host kernel: {k}");
        }
        (arena.cpu_rows, arena.cpu_experts, arena.cpu_ms) = (0, 0, 0.0);
        (arena.host_ms, arena.host_layers) = ([0.0; 5], 0);
        self.acc = [0.0; STAGES.len()];
        self.steps = 0;
        arena.admitted = 0;
        arena.evicted = 0;
        arena.stage_ms = 0.0;
        arena.unpinned = 0;
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
    /// Address of the tubes' descriptors (nine words each) on the card, and their copy on the host.
    tube_desc: u64,
    tube_host: Vec<u64>,
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
    /// The tubes' matrices and descriptors.
    _tubes: (Vec<DevMat>, Option<Buf>),
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
        // the stagers read the mapped model and write the arena: they stop before either goes
        self.arena.stagers = None;
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
        // what the card had free before this model, and one slab pool for its many small buffers
        let free0 = super::mem_free();
        let slabs = super::SlabScope::enter();
        let res0 = c.resident.load(std::sync::atomic::Ordering::Relaxed);
        super::pdl_set(std::env::var("CMF_QWEN_PDL").is_ok_and(|v| v == "1"));
        let q = qf(c)?;
        if std::env::var("CMF_QWEN_NOP").is_ok_and(|v| v == "1") {
            super::nop_set(q.zero);
        }
        let (gw, ga) = prec();
        let ga = if ga == Gemm::Half { Gemm::Tf32 } else { ga };
        let half = gw == Gemm::Half;
        let es = if half { 2 } else { 4 };
        let g = &cfg.gdn;
        let first = layers.first().ok_or_else(|| "qwen4 cuda: no layers".to_string())?;
        // latent layers (Edge0) keep their experts out of the arena, which then serves the MTP head alone:
        // its geometry is the head's (or Qwen's 640-wide q4tp experts without a head)
        let latent = layers.iter().any(|l| l.latent.is_some());
        if latent && !layers.iter().all(|l| l.latent.is_some()) {
            return Err("qwen4 cuda: latent and full-width layers mixed".into());
        }
        let arena_moe = if latent { mtp.map(|p| &p.layer.moe) } else { Some(&first.moe) };
        let ne = first.moe.experts.len();
        let inter = arena_moe.and_then(|m| m.experts.first()).map_or(if latent { 640 } else { 0 }, |e| e.gate_proj.rows());
        let h = cfg.hidden;
        if ne == 0 || inter == 0 || !h.is_multiple_of(32) || !inter.is_multiple_of(32) || cfg.compress_ratio == 0 {
            return Err("qwen4 cuda: unexpected expert shapes".into());
        }
        // the experts' gate and up: q2tp (the 2-bit codec) or q4tp
        let gbits = if arena_moe.and_then(|m| m.experts.first()).is_some_and(|e| e.gate_proj.mapped_q2tp().is_some()) { 2 } else { 4 };
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
            sgu: if gbits == 2 { q2_bytes(inter, h) } else { q4_bytes(inter, h) },
            sd: q4_bytes(h, inter),
            gbits,
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
        let mut b = Build { c, q, half, held: 0, wq_elems: 16, range, li: 0, tape, tube_desc: Vec::new(), tube_keep: Vec::new() };
        let mut lws = Vec::with_capacity(layers.len());
        let (mut r_max, mut lr_max, mut pin_max, mut hist_max, mut rl_max) = (1usize, 1usize, 16usize, 1usize, 1usize);
        // what the card holds by kind, for the load report: hyper-connections, GDN, QSA, PLE, routers and
        // shared experts, LM head, MTP head, prompt scratch, tables; and the GDN and QSA layers there are
        let mut mem = [0u64; 9];
        let (mut n_gdn, mut n_qsa) = (0usize, 0usize);
        let mut lat_dims: Option<(usize, usize, usize)> = None;
        for (li, l) in layers.iter().enumerate() {
            b.li = li;
            let h0 = b.held;
            let attn = b.hc(&l.attn_hc, &d, true)?;
            let mlp = b.hc(&l.mlp_hc, &d, true)?;
            mem[0] += b.held - h0;
            let h0 = b.held;
            let (mix, kind) = match &l.mixer {
                Mixer::Gdn(w) => (Mix::Gdn(b.gdn(w, &d)?), 1),
                Mixer::Qsa(w) => (Mix::Qsa(Box::new(b.qsa(w, &d)?)), 2),
            };
            mem[kind] += b.held - h0;
            if kind == 1 {
                n_gdn += 1;
            } else {
                n_qsa += 1;
            }
            let h0 = b.held;
            let ple = match &l.ple {
                Some(p) => Some(b.ple(p, &d)?),
                None => None,
            };
            mem[3] += b.held - h0;
            let h0 = b.held;
            let (moe, lat) = match &l.latent {
                Some(lt) => {
                    let (lw, ph) = b.moe_lat(&l.moe, lt, &d)?;
                    lat_dims = Some((lw.l, lw.groups, lw.ff));
                    (ph, Some(lw))
                }
                None => (b.moe(&l.moe, &d, d.ne)?, None),
            };
            mem[4] += b.held - h0;
            r_max = r_max.max(attn.r).max(mlp.r);
            lr_max = lr_max.max(attn.lr).max(mlp.lr);
            rl_max = rl_max.max(moe.r);
            if let Some(p) = &ple {
                pin_max = pin_max.max(p.pin);
                hist_max = hist_max.max((d.ple_k - 1) * d.ple_dil * d.w);
            }
            lws.push(LayerW { attn, mlp, mix, ple, moe, lat });
        }
        let h0 = b.held;
        let head = b.hc(&globals.head_hc, &d, false)?;
        mem[0] += b.held - h0;
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
            mem[5] += m.held;
            m
        };
        let invf = b.f32buf(inv_freq)?;
        let h0 = b.held;
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
        mem[6] += b.held - h0;
        let topk = topk.max(mtp_w.as_ref().map_or(1, |m| m.lw.moe.topk));
        // the 8-bit tubes of a hot file: the descriptors the units point at
        let tube_host = b.tube_desc.clone();
        let tube_buf = if tube_host.is_empty() { None } else { Some(b.f32buf_u64(&tube_host)?) };
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
        let h0 = b.held;
        let s = Scratch {
            emb: b.buf(n * d.h * 4)?,
            hyper: b.buf(n * d.w * 4)?,
            normed: b.buf(n * d.w * 4)?,
            normed_h: b.buf(n * d.w * es)?,
            low: b.buf(n * r_max * 4)?,
            lowh: b.buf((n * lr_max * es).max(n.min(SMALL_ROWS) * lr_max * 4))?,
            hcp: b.buf(d.w.div_ceil(tune.cw.min(d.w)) * n.min(SMALL_ROWS) * r_max * 4)?,
            iw: b.buf(n * d.hc * 4)?,
            tick: b.zeroed(1024)?,
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
            lxl: b.buf(lat_dims.map_or(16, |(l, g, _)| n * g * l * 4))?,
            lxlh: b.buf(lat_dims.map_or(16, |(l, g, _)| n * g * l * es))?,
            lacc: b.buf(lat_dims.map_or(16, |(l, g, _)| n * g * l * 4))?,
            lacch: b.buf(lat_dims.map_or(16, |(l, g, _)| n * g * l * es))?,
            lact: b.buf(lat_dims.map_or(16, |(_, _, ff)| LAT_PICKS.max(tape.max(SMALL_ROWS) * topk) * ff * 4))?,
            ly: b.buf(lat_dims.map_or(16, |(l, _, _)| LAT_PICKS.max(tape.max(SMALL_ROWS) * topk) * l * 4))?,
            lf: b.buf(3 * xb * 4 + 64)?,
            lmask: b.buf(64)?,
            lpart: b.buf(lat_dims.map_or(16, |(_, g, _)| g * SMALL_ROWS * d.h * 4))?,
            lyc: b.buf(lat_dims.map_or(16, |(l, _, _)| n * topk * l * 4))?,
            lrp: b.buf(xr * 4)?,
        };
        mem[7] += b.held - h0;
        let h0 = b.held;
        let maskb = b.buf(n * 64 * 4)?;

        // The arena takes what the card has left, as the driver reports it,
        // less what a session grows into (the attention state of a long
        // prompt, `CMF_QWEN_CUDA_GROW_MB`); under `CMF_GPU_VRAM_MB` no more
        // than that budget less the driver's context and what is taken so far.
        let slot = 2 * d.sgu + d.sd;
        let mib = |v: u64| v as f64 / (1u64 << 20) as f64;
        let grow = std::env::var("CMF_QWEN_CUDA_GROW_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024) << 20;
        let free1 = super::mem_free();
        let room = match (free0, free1) {
            (Some(f0), Some(f1)) => {
                let spare = f1.saturating_sub((96 << 20) + grow);
                match std::env::var("CMF_GPU_VRAM_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
                    Some(mb) => spare.min((mb << 20).saturating_sub(CTX_BYTES + (32 << 20) + res0 + f0.saturating_sub(f1) + grow)),
                    None => spare,
                }
            }
            _ => vram_budget().saturating_sub(c.resident.load(std::sync::atomic::Ordering::Relaxed) + b.held + (512 << 20) + grow),
        };
        let mut slots = (room / slot as u64) as usize;
        if let Some(mb) = std::env::var("CMF_QWEN_CUDA_ARENA_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            slots = slots.min((mb << 20) / slot);
        }
        slots = slots.min(ne * layers.len());
        if latent {
            // the arena holds the MTP head's experts only
            slots = slots.min(2 * ne.max(head_ne));
        }
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
        mem[8] += b.held - h0;
        let buf = b.buf(slots * slot)?;
        let stagers = if cpu_experts() { Stagers::start(c, slot, d.sgu) } else { None };
        let arena = Arena {
            stagers,
            buf,
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
            heat: vec![0.0; layers.len() * ne + head_ne],
            heat_at: vec![0; layers.len() * ne + head_ne],
            unpinned: 0,
            inflight: HashSet::new(),
            free: Vec::new(),
            bg_spare: Vec::new(),
            bg_dry: (u64::MAX, 0),
            staged: 0,
            landed: 0,
            queue_full: 0,
            head_host: false,
            zc: Vec::new(),
            zc_dev: Vec::new(),
            zc_dsc: None,
            zc_ready: Vec::new(),
            zc_full: false,
            cpu_rows: 0,
            cpu_experts: 0,
            cpu_ms: 0.0,
            host_ms: [0.0; 5],
            host_layers: 0,
            host_at: None,
            host: HostScratch::new(),
            epoch: 1,
        };
        // the pinned expert pages (`Dev::prefill`) need the mapped model to outlive them
        let keep = layers.first().and_then(|l| l.moe.experts.first()).and_then(|f| f.gate_proj.model_arc());
        c.sync()?;
        c.resident.fetch_add(b.held, std::sync::atomic::Ordering::Relaxed);
        if super::debug() || std::env::var_os("CMF_QWEN_PROF").is_some() {
            eprintln!(
                "qwen4 cuda: {} layers on {}, {:.1} GiB held, arena {} of {} experts, chunk {}, weights {}, small path shared memory (matvec/hc/experts) {} KB",
                layers.len(),
                c.name,
                b.held as f64 / (1u64 << 30) as f64,
                slots,
                ne * layers.len(),
                n,
                if half { "f16" } else { "f32" },
                format!("{}/{}/{}", q.sm.mvb_max >> 10, q.sm.hc_max >> 10, q.sm.ex_max >> 10),
            );
            eprintln!(
                "qwen4 cuda: card free {:.0} MiB at the start, {:.0} before the arena, {:.0} now; arena {:.0} MiB, {:.0} MiB left to grow into; {}",
                mib(free0.unwrap_or(0)),
                mib(free1.unwrap_or(0)),
                mib(super::mem_free().unwrap_or(0)),
                mib((slots * slot) as u64),
                mib(grow),
                slabs.report(),
            );
            // the recurrent state and tape of a GDN layer, the attention cache of a QSA layer (the MTP head's too)
            let gdn_state = (4 * (2 * (d.kk - 1).max(1) * d.c_dim + 2 * d.nv * d.dk * d.dv + tape * (2 * d.c_dim + 2 * d.nv))) as u64;
            let kv_cells = MAX_CHUNK;
            let qsa_cache = (4 * (2 * kv_cells * d.kvh * d.hd + kv_cells * d.idim + (kv_cells / d.cr + 1) * d.idim)) as u64;
            let (gs, qc) = (gdn_state * n_gdn as u64, qsa_cache * n_qsa as u64);
            let mtp_kv = if mtp_w.is_some() { qsa_cache } else { 0 };
            let named: u64 = mem.iter().sum();
            let lost = free0.zip(free1).map_or(0, |(f0, f1)| f0.saturating_sub(f1));
            eprintln!(
                "qwen4 cuda: card memory before the arena, MiB: hyper-connections {:.0}, GDN weights {:.0} and state {:.0}, QSA weights {:.0} and attention cache {:.0} ({kv_cells} cells, f32), PLE {:.0}, routers and shared experts {:.0}, LM head {:.0}, MTP head {:.0} (its cache {:.0}), prompt scratch {:.0} (chunk {n}, tape {tape}), tables {:.0}; of the {:.0} MiB the card lost, {:.0} are the driver's (kernels, libraries, rounding)",
                mib(mem[0]),
                mib(mem[1].saturating_sub(gs)),
                mib(gs),
                mib(mem[2].saturating_sub(qc)),
                mib(qc),
                mib(mem[3]),
                mib(mem[4]),
                mib(mem[5]),
                mib(mem[6]),
                mib(mtp_kv),
                mib(mem[7]),
                mib(mem[8]),
                mib(lost),
                mib(lost.saturating_sub(named)),
            );
        }
        // `CMF_QWEN_PROF=2`: no wait at the stages (their times then say nothing), the host path's own clock only
        let prof = std::env::var("CMF_QWEN_PROF").ok().map(|v| {
            QTensor::moe_expert_rows_fast_prof(true);
            Mutex::new(Prof { t0: Instant::now(), acc: [0.0; STAGES.len()], steps: 0, sync: v.trim() != "2" })
        });
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
                tube_desc: tube_buf.as_ref().map_or(0, |t| t.ptr),
                tube_host,
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
            _tubes: (std::mem::take(&mut b.tube_keep), tube_buf),
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
        if on {
            super::kt_start();
        } else {
            super::kt_stop();
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
        let sticky = if arena.sticky_from == u32::MAX { 0 } else { mtp.as_ref().map_or(0, |m| layers.get(0).map_or(0, |_| m.lw.moe.ne)) };
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
                None if swaps < max_swap && layers[l].moe.experts[e].segs.is_empty() => {
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

    /// The host's expert workers on two NUMA nodes, the rows of every expert matrix split between them: the
    /// lower rows' weights in the memory of the node the process sits on, the upper rows' in the other
    /// node's, each computed by the threads of its node (`CMF_QWEN_CPU_NODES=1` keeps one node).
    ///
    /// WHY: a node has one memory bus. On 2 x EPYC 9554 a plain reader gets 220-230 GB/s on one node and
    /// 393-421 GB/s with 14 + 14 threads over both, each on weights of its own node; the expert kernel itself
    /// is bound by its arithmetic at 28 threads (273 us of a thread an expert at an idle bus), so the split
    /// did not move the aquarium's speed, and it keeps the far threads off the near bus. The weights are page-cache pages of the model
    /// file, which stay on the node that read them first: `Pool::place_file_pages` has the upper half of
    /// every matrix read again by the far node's threads where it is not there yet. It runs before anything
    /// pins the pages (a pinned page cannot leave the page cache), and a second start finds them placed.
    /// Where the pages will not go (another process maps the file) the experts stay on one node.
    pub(crate) fn host_nodes(&self, layers: &[Layer], ranked: &[(usize, usize)], main: Option<&crate::pool::Pool>) {
        if !cpu_experts() || !cpu_fast() || std::env::var("CMF_QWEN_CPU_NODES").is_ok_and(|v| v.trim() == "1") {
            return;
        }
        let d = &self.k.d;
        let total = layers.len() * d.ne;
        // the kernel halves a matrix by its 16-row blocks; an arena that holds every expert leaves the host none
        if d.gbits != 4 || d.inter % 32 != 0 || d.h % 32 != 0 || self.arena.n.saturating_sub(self.arena.used) >= total {
            return;
        }
        // the nibble planes are the bytes a row costs (96 % of a matrix), row after row from the start
        let nib = [cortiq_core::quant::q4tp_sections(d.inter, d.h).0, cortiq_core::quant::q4tp_sections(d.h, d.inter).0];
        // hottest first, as the pins take them: what the RAM does not hold is not read for this
        let mut seen = vec![false; total];
        let by_heat = ranked.iter().copied().filter(|&(l, e)| l < layers.len() && e < d.ne);
        let by_index = (0..layers.len()).flat_map(|l| (0..d.ne).map(move |e| (l, e)));
        let (mut lo, mut hi): (Vec<&[u8]>, Vec<&[u8]>) = (Vec::new(), Vec::new());
        for (l, e) in by_heat.chain(by_index) {
            if std::mem::replace(&mut seen[l * d.ne + e], true) {
                continue;
            }
            let Some(f) = layers[l].moe.experts.get(e).filter(|f| f.segs.is_empty()) else {
                continue;
            };
            let Ok(parts) = expert_parts(f, d) else {
                continue;
            };
            for (b, nb) in [(parts[0], nib[0]), (parts[1], nib[0]), (parts[2], nib[1])] {
                if nb <= b.len() {
                    lo.push(&b[..nb / 2]);
                    hi.push(&b[nb / 2..nb]);
                }
            }
        }
        if lo.is_empty() {
            return;
        }
        let pool = cpu_pool();
        // the node most of the lower halves are on already is the home: a second start moves nothing
        let Some((home, far, nh, nf)) = pool.bind_two_nodes(crate::pool::file_pages_node(&lo)) else {
            return;
        };
        if let Some(p) = main {
            p.bind_node(home);
        }
        // as many experts as the pins will hold: every one of them when they are forced
        let room = if std::env::var("CMF_QWEN_CUDA_REG").is_ok_and(|v| v.trim() == "1") {
            total
        } else {
            crate::qwen4_exp::mem_available().saturating_sub(crate::qwen4_exp::ple_table_bytes(layers) + (4 << 30)) / slot_bytes(d).max(1)
        };
        let t0 = Instant::now();
        let placed = pool.place_file_pages(&lo, &hi, (home, far), 3 * room, &|bytes| {
            eprintln!(
                "qwen4 cuda: host experts: {:.1} GB of expert weights leave the page cache to be read again by the node that computes them",
                bytes as f64 / 1e9
            )
        });
        match placed {
            Some(p) if p.home >= 0.8 && p.far >= 0.8 => eprintln!(
                "qwen4 cuda: host experts on two NUMA nodes: {nh} threads on node {home} with the lower rows of every expert matrix ({:.0} % of a sample there), {nf} on node {far} with the upper rows ({:.0} %); {:.1} GB read again and {:.1} GB read first for it in {:.1}s",
                p.home * 100.0,
                p.far * 100.0,
                p.moved as f64 / 1e9,
                p.filled as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            ),
            p => {
                pool.unbind_two_nodes();
                eprintln!(
                    "qwen4 cuda: host experts stay on one node: {}",
                    p.map_or("the expert weights are no file pages this process can place".to_string(), |p| format!(
                        "{:.0} % of the lower rows on node {home} and {:.0} % of the upper rows on node {far} after {:.1} GB read again",
                        p.home * 100.0,
                        p.far * 100.0,
                        p.moved as f64 / 1e9
                    ))
                );
            }
        }
    }

    /// `CMF_QWEN_CPU_BENCH=<calls>`: time the host kernel alone on this model's own experts, the ones the
    /// arena does not hold. A call is 14 experts for 21 (expert, token) pairs over a window's 4 rows, a layer
    /// after a layer: a pass over the layers reads 1.6 GB, so every call meets cold weights as a window's
    /// does. Every setting of `CMF_QWEN_CPU_BENCH_SET` (`<prefetch bytes>:<gate/up claim>:<down claim>`, comma
    /// separated) runs `calls` calls three times, the settings interleaved; a line a setting says the median
    /// and the best round and where the threads' time went.
    pub(crate) fn cpu_bench(&self, layers: &[Layer]) {
        let Some(calls) = std::env::var("CMF_QWEN_CPU_BENCH").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0) else {
            return;
        };
        let d = &self.k.d;
        let (hid, ne) = (d.h, d.ne);
        if !QTensor::moe_expert_rows_fast_available() || layers.is_empty() || ne < 64 {
            eprintln!("qwen4 cuda: host kernel bench: no host kernel here");
            return;
        }
        const EXPERTS: usize = 14;
        const ROWS: usize = 4;
        // the pages of the experts outside the arena are pinned by a thread of their own: wait for it
        if let Some(pins) = self.arena.pins.as_ref() {
            let t0 = Instant::now();
            while !pins.ready.iter().all(|f| f.load(Ordering::Acquire)) && t0.elapsed().as_secs() < 120 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let mut seed = 0x9E37_79B9u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as usize
        };
        let mut sets: Vec<Vec<usize>> = Vec::with_capacity(layers.len());
        for (li, l) in layers.iter().enumerate() {
            let mut set: Vec<usize> = Vec::with_capacity(EXPERTS);
            for _ in 0..4096 {
                if set.len() == EXPERTS {
                    break;
                }
                let e = next() % ne.min(l.moe.experts.len());
                if !set.contains(&e) && l.moe.experts[e].segs.is_empty() && !self.arena.map.contains_key(&(((li as u32) << 16) | e as u32)) {
                    set.push(e);
                }
            }
            if set.len() < EXPERTS {
                eprintln!("qwen4 cuda: host kernel bench: layer {li} has no {EXPERTS} experts outside the arena");
                return;
            }
            sets.push(set);
        }
        // 14 experts, 21 pairs: one token, two tokens, one, two...
        let groups: Vec<Vec<usize>> = (0..EXPERTS).map(|i| if i % 2 == 0 { vec![i % ROWS] } else { vec![i % ROWS, (i + 2) % ROWS] }).collect();
        let n_pairs: usize = groups.iter().map(|g| g.len()).sum();
        let xs: Vec<f32> = (0..ROWS * hid).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut out = vec![0f32; n_pairs * hid];
        let pool = cpu_pool();
        let bytes: usize = {
            let x = &layers[0].moe.experts[sets[0][0]];
            EXPERTS * [&x.gate_proj, &x.up_proj, &x.down_proj].iter().map(|t| t.mapped_bytes().map_or(0, |b| b.len())).sum::<usize>()
        };
        let mut run = |n: usize, at: &mut usize| -> Option<f64> {
            let t0 = Instant::now();
            for _ in 0..n {
                let li = *at % layers.len();
                *at += 1;
                let ex = &layers[li].moe.experts;
                let experts: Vec<(&QTensor, &QTensor, &QTensor)> = sets[li].iter().map(|&e| (&ex[e].gate_proj, &ex[e].up_proj, &ex[e].down_proj)).collect();
                if !QTensor::moe_expert_rows_fast(&experts, &groups, &xs, &mut out, Some(pool)) {
                    return None;
                }
            }
            Some(t0.elapsed().as_secs_f64() / n as f64)
        };
        let set_env = std::env::var("CMF_QWEN_CPU_BENCH_SET").unwrap_or_else(|_| "0:2:8,1280:2:8,2560:2:8,5120:2:8,10240:2:8,0:1:4,0:4:16,0:1:8,0:2:16".into());
        let settings: Vec<[usize; 3]> = set_env
            .split(',')
            .filter_map(|v| {
                let p: Vec<usize> = v.split(':').filter_map(|x| x.trim().parse().ok()).collect();
                (p.len() == 3).then(|| [p[0], p[1].max(1), p[2].max(1)])
            })
            .collect();
        if settings.is_empty() {
            return;
        }
        let prof_was = QTensor::moe_expert_rows_fast_prof(true);
        let before: Vec<usize> = (0..3).map(|k| QTensor::moe_expert_rows_fast_tune(k, settings[0][k])).collect();
        let mut at = 0usize;
        // every expert's pages once, untimed
        if run(layers.len(), &mut at).is_none() {
            eprintln!("qwen4 cuda: host kernel bench: the kernel declined these experts");
        } else {
            let _ = QTensor::moe_expert_rows_fast_prof_take();
            let mut secs: Vec<Vec<f64>> = vec![Vec::new(); settings.len()];
            let mut lines: Vec<String> = vec![String::new(); settings.len()];
            for round in 0..3 {
                for (i, st) in settings.iter().enumerate() {
                    for (k, &v) in st.iter().enumerate() {
                        QTensor::moe_expert_rows_fast_tune(k, v);
                    }
                    if let Some(t) = run(calls, &mut at) {
                        secs[i].push(t);
                    }
                    let line = QTensor::moe_expert_rows_fast_prof_take().unwrap_or_default();
                    if round == 2 {
                        lines[i] = line;
                    }
                }
            }
            for (i, st) in settings.iter().enumerate() {
                let mut t = secs[i].clone();
                t.sort_by(|a, b| a.total_cmp(b));
                let (Some(&best), Some(&mid)) = (t.first(), t.get(t.len() / 2)) else {
                    continue;
                };
                eprintln!(
                    "qwen4 cuda: host kernel bench: prefetch {} claims {}:{}: {:.0} us a call of {EXPERTS} experts ({:.0} GB/s), best round {:.0} us; {}",
                    st[0],
                    st[1],
                    st[2],
                    mid * 1e6,
                    bytes as f64 / mid / 1e9,
                    best * 1e6,
                    lines[i].split("; ").skip(1).collect::<Vec<_>>().join("; "),
                );
            }
        }
        for (k, &v) in before.iter().enumerate() {
            QTensor::moe_expert_rows_fast_tune(k, v);
        }
        QTensor::moe_expert_rows_fast_prof(prof_was);
        let _ = QTensor::moe_expert_rows_fast_prof_take();
    }

    /// Put all the head's experts into the arena and keep them there: a draft
    /// cell then never waits for an upload. `head` is the head's host layer.
    pub(crate) fn warm_mtp(&mut self, head: &Layer) -> Res<usize> {
        let Self { k, arena, mtp, .. } = self;
        let Some(m) = mtp.as_ref() else { return Ok(0) };
        let (d, c) = (&k.d, k.c);
        // `CMF_QWEN_HEAD_HOST=1`: the head keeps no slots of its own. Its experts take them by use, as the
        // backbone's do, and the host computes the absent ones: a small arena gets the head's share back
        // for the backbone, and a draft cell pays a trip to the host.
        if std::env::var("CMF_QWEN_HEAD_HOST").is_ok_and(|v| v.trim() == "1") && cpu_experts() && d.gbits == 4 && !m.lw.moe.has_tubes {
            arena.head_host = true;
            return Ok(0);
        }
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
    /// `recent`: the last accepted tokens, which pick the draft's vocabulary (see `draft_vocab`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft(
        &mut self,
        head: &Layer,
        embed: &dyn Fn(u32, &mut [f32]),
        tok0: u32,
        pos: usize,
        kmax: usize,
        recent: &[u32],
        out: &mut Vec<u32>,
    ) -> Res<bool> {
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
            k.hc_mix_inj(&m.lw.mlp, s.hyper.ptr, 1, true)?;
            k.moe(&m.lw.moe, head, m.li, arena, 1, false)?;
            k.inject(1)?;
            let mut ids: Vec<u32> = Vec::new();
            k.head_rows_v(&m.mixer, lm, 1, &mut ids, draft_vocab(recent, d.vocab), true)?;
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
        // the free slots only: the MTP head's experts already hold theirs, and an admission past the room
        // would evict a hotter expert of an earlier layer
        let room = arena.n.saturating_sub(arena.used);
        for &(l, e) in ranked.iter().filter(|&&(l, e)| l < layers.len() && e < d.ne && layers[l].moe.experts[e].segs.is_empty()).take(room) {
            by_layer[l].push(e);
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
        // missing) get their file pages pinned, up to `CMF_QWEN_CUDA_REG_MB` (default: all of them).
        // Unset, they are pinned when the RAM holds them beside the n-gram table with 8 GB to spare (a prompt's
        // misses then never wait on the disk: on a 62 GB container the first requests ran 1.5-2x faster); `0`
        // never pins.
        let reg = std::env::var("CMF_QWEN_CUDA_REG").ok().map(|v| v.trim().to_string());
        if reg.as_deref() != Some("0") && arena.stage.is_some() && arena.pins.is_none() {
            let budget = std::env::var("CMF_QWEN_CUDA_REG_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(usize::MAX, |mb| mb << 20);
            let mut sel: Vec<Vec<bool>> = vec![vec![true; d.ne]; layers.len()];
            // `CMF_QWEN_CUDA_REG_ALL=1`: the arena's own experts too (a host with the page cache for it)
            if !std::env::var("CMF_QWEN_CUDA_REG_ALL").is_ok_and(|v| v.trim() == "1") {
                // the experts the arena holds now, not the profile's head: these two differ by what did not fit
                for &key in arena.map.keys() {
                    let (l, e) = ((key >> 16) as usize, (key & 0xffff) as usize);
                    if l < sel.len() && e < d.ne {
                        sel[l][e] = false;
                    }
                }
                // zero copy: `CMF_QWEN_ZC_SPARE=<n>` pins up to n of the arena's coldest experts too, as far as the
                // RAM holds them beside the n-gram table with 8 GB to spare, so a refill (`CMF_QWEN_SELF_PACK=1`)
                // may trade them for what a request routes to. Off by default: the refill from the prompt's
                // routing left the aquarium's window as it was (35.4 vs 35.0 ms)
                if zero_copy() && !layers.iter().any(|l| l.moe.experts.iter().any(|f| !f.segs.is_empty())) {
                    let outside: usize = sel.iter().map(|v| v.iter().filter(|&&b| b).count()).sum();
                    let (avail, table) = (crate::qwen4_exp::mem_available(), crate::qwen4_exp::ple_table_bytes(layers));
                    let room = avail.saturating_sub(table + (8 << 30) + outside * slot_bytes(d)) / slot_bytes(d).max(1);
                    let cap = std::env::var("CMF_QWEN_ZC_SPARE").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
                    let mut extra = 0usize;
                    let held: Vec<(usize, usize)> = ranked
                        .iter()
                        .copied()
                        .filter(|&(l, e)| l < sel.len() && e < d.ne && arena.map.contains_key(&(((l as u32) << 16) | e as u32)))
                        .collect();
                    for &(l, e) in held.iter().rev().take(room.min(cap)) {
                        sel[l][e] = true;
                        extra += 1;
                    }
                    if extra > 0 {
                        eprintln!("qwen4 cuda: zero copy: {extra} of the arena's coldest experts pinned too (the arena trades them per request)");
                    }
                }
            }
            // with the host computing the misses the pinned pages only feed the admissions (`admit_hot` takes
            // none without them: at 12 GB that was 87.9 against 98.4 tok/s), so the unset default pins what the
            // RAM holds beside the n-gram table with 4 GB to spare, the profile's hottest experts first
            let mut host_pins = false;
            if reg.is_none() && cpu_experts() {
                let (avail, table) = (crate::qwen4_exp::mem_available(), crate::qwen4_exp::ple_table_bytes(layers));
                let outside: usize = sel.iter().map(|v| v.iter().filter(|&&b| b).count()).sum();
                let room = avail.saturating_sub(table + (4 << 30)) / slot_bytes(d).max(1);
                if room < outside {
                    let mut keep: Vec<Vec<bool>> = vec![vec![false; d.ne]; layers.len()];
                    let mut left = room;
                    let by_heat = ranked.iter().copied().filter(|&(l, e)| l < sel.len() && e < d.ne);
                    let by_index = (0..sel.len()).flat_map(|l| (0..d.ne).map(move |e| (l, e)));
                    for (l, e) in by_heat.chain(by_index) {
                        if left == 0 {
                            break;
                        }
                        if sel[l][e] && !keep[l][e] {
                            keep[l][e] = true;
                            left -= 1;
                        }
                    }
                    sel = keep;
                }
                host_pins = room > 0;
                eprintln!(
                    "qwen4 cuda: {} of {outside} experts outside the arena pinned for the admissions, hottest first (the n-gram table {:.1} GB, {:.1} GB of RAM available)",
                    room.min(outside),
                    table as f64 / 1e9,
                    avail as f64 / 1e9
                );
            }
            let (mut ranges, mut total) = (Vec::with_capacity(layers.len()), 0usize);
            let mut zc: Vec<Vec<[u64; 3]>> = vec![vec![[0u64; 3]; d.ne]; layers.len()];
            for (li, l) in layers.iter().enumerate() {
                // page runs of the layer's chosen experts, merged where they touch
                let mut runs: Vec<(usize, usize)> = Vec::new();
                for (e, f) in l.moe.experts.iter().enumerate() {
                    if !sel[li][e] {
                        continue;
                    }
                    // a hot expert is read from its tube, not from the file's pages
                    let Ok(parts) = expert_parts(f, d).and_then(|p| if f.segs.is_empty() { Ok(p) } else { Err(String::new()) }) else {
                        sel[li][e] = false;
                        continue;
                    };
                    if total >= budget {
                        sel[li][e] = false;
                        continue;
                    }
                    // the three matrices in the file: the card reads them in place through a descriptor
                    zc[li][e] = [parts[0].as_ptr() as u64, parts[1].as_ptr() as u64, parts[2].as_ptr() as u64];
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
            let fits = || {
                let (avail, table) = (crate::qwen4_exp::mem_available(), crate::qwen4_exp::ple_table_bytes(layers));
                let ok = avail >= total + table + (8 << 30);
                eprintln!(
                    "qwen4 cuda: {:.1} GB of experts outside the arena, the n-gram table {:.1} GB, {:.1} GB of RAM available: {}",
                    total as f64 / 1e9,
                    table as f64 / 1e9,
                    avail as f64 / 1e9,
                    if ok { "pinning them" } else { "left to the page cache" }
                );
                ok
            };
            if reg.as_deref() == Some("1") || host_pins || fits() {
                let zc_on = zero_copy() && !layers.iter().any(|l| l.moe.experts.iter().any(|f| !f.segs.is_empty()));
                // zero copy waits for the pins: a prompt that came first would evict experts nobody maps
                arena.pins = Some(Pins::start(c, sel, ranges, zc_on));
                if zc_on {
                    let n = zc.len();
                    let cand: usize = zc.iter().map(|v| v.iter().filter(|p| p[0] != 0).count()).sum();
                    arena.zc_dsc = Buf::new(n * d.ne * 24).ok();
                    eprintln!("qwen4 cuda: zero copy: {cand} experts can be read in place");
                    arena.zc_dev = vec![vec![0u64; d.ne]; n];
                    arena.zc_ready = vec![false; n];
                    arena.zc = zc;
                    // mapped now, so the first prompt's experts are read in place instead of evicting the arena's
                    arena.zc_poll();
                    if !arena.zc_full {
                        eprintln!("qwen4 cuda: zero copy: the route does not cover every expert; misses stay on the staging");
                    }
                }
            }
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

/// What the driver's context takes of a card before the first buffer.
const CTX_BYTES: u64 = 512 << 20;

/// A card whose budget leaves little beyond the weights: below 26 GiB.
pub(crate) fn lean_card() -> bool {
    vram_budget() < (26u64 << 30)
}

/// Rows of a window's tape: eight (a verify round), or `CMF_QWEN_CUDA_TAPE`
/// (up to 256; with the copy frames on, 256, or 64 on a lean card).
fn tape_rows(n_cap: usize) -> usize {
    let frames = std::env::var("CMF_QWEN_COPY_FRAME").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
    let want = std::env::var("CMF_QWEN_CUDA_TAPE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(match frames {
            // a lean card keeps the short frames only: 192 rows of tape less is 500 MiB of experts more
            true if lean_card() => 64,
            true => 256,
            false => SMALL_ROWS,
        });
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

    /// A q8 matrix expanded into the one scratch plane (valid until the next q8 one).
    fn plane_q8(&self, m: &DevMat) -> Res<u64> {
        go!(self.c, self.q.expand_q8, (blocks(m.rows * m.cols / 4), 1), 256;
            m.w.ptr, m.aux0(), m.aux1(), self.s.wq.ptr, m.rows as i32, m.cols as i32, self.hi);
        Ok(self.s.wq.ptr)
    }

    /// Close stage `k` of the profile (waits for the stream).
    fn lap(&self, k: usize) -> Res<()> {
        if let Some(p) = &self.prof {
            let mut p = p.lock().unwrap();
            if p.sync {
                self.c.sync()?;
            }
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
        self.mvb_x(ops, x, xstride, cols, n, false)
    }

    /// `mvb`; with `exact` (the vocabulary head) the f32 dot even in the fast precision.
    fn mvb_x(&self, ops: &[SOp], x: u64, xstride: usize, cols: usize, n: usize, exact: bool) -> Res<()> {
        let sm = &self.q.sm;
        let (fam, fam512) = if !exact && mv_int8() { (&sm.mvq, &sm.mvq512) } else { (&sm.mvb, &sm.mv512) };
        let per = mv_row_bytes(cols);
        let cap = (sm.mvb_max / per).min(SMALL_ROWS);
        if cap == 0 || ops.is_empty() || ops.len() > 5 {
            return Err(format!("qwen4 cuda: a small matvec of {} jobs over {cols} columns", ops.len()));
        }
        let mut jobs = [MvJob { w: 0, aux0: 0, aux1: 0, y: 0, res: 0, rows: 0, kind: 0, blk0: 0, rpb: 1 }; 5];
        // a launch with a big matrix runs 512 threads and about rows / 128 rows a block (up to 80): the inputs are
        // staged once for twice the rows (microbenchmark on the 5090: 10240 x 2560 23.7 -> 20.4 us, the vocabulary
        // head 429 -> 392 us); small matrices keep 256 threads
        let big = mv_wide() && ops.iter().any(|o| o.rows >= 4096);
        let mut t0 = 0;
        while t0 < n {
            let nb = (n - t0).min(cap);
            // the kernels of many rows hold too many registers for 512 threads
            let wide = big && fam512[nb - 1];
            let threads = if wide { 512u32 } else { 256 };
            let mut blk = 0u32;
            for (j, o) in jobs.iter_mut().zip(ops) {
                let rpb = if wide { (o.rows / 128).next_multiple_of(16).clamp(16, 80) } else { rows_per_block(o.rows) };
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
                    fam[nb - 1],
                    (blk, 1),
                    threads,
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
        self.hc_mix_inj(hc, hyper, n, false)
    }

    /// `hc_mix`, with the previous block's `inject` folded into its first launch when `inj` is set.
    fn hc_mix_inj(&self, hc: &Hc, hyper: u64, n: usize, inj: bool) -> Res<()> {
        let (d, s) = (&self.d, &self.s);
        // one float4 a thread: rows of up to 4096 values, aligned (CMF_QWEN_GRMS4=0 keeps the 256-thread kernels)
        let t4 = d.h.div_ceil(4).next_multiple_of(32);
        if d.h.is_multiple_of(4) && t4 <= 1024 && grms4() {
            let blk = if inj { s.blk.ptr } else { 0u64 };
            go!(self.c, self.q.grms_inj4, ((n * d.hc) as u32, 1), t4;
                hyper, hc.norm.ptr, s.normed.ptr, s.normed_h.ptr, d.h as i32, d.hc as i32, d.eps, self.hi, blk, s.iw.ptr);
        } else if inj {
            go!(self.c, self.q.grms_inj, ((n * d.hc) as u32, 1), 256;
                hyper, hc.norm.ptr, s.normed.ptr, s.normed_h.ptr, d.h as i32, d.hc as i32, d.eps, self.hi, s.blk.ptr, s.iw.ptr);
        } else {
            go!(self.c, self.q.grms, ((n * d.hc) as u32, 1), 256;
                hyper, hc.norm.ptr, s.normed.ptr, s.normed_h.ptr, d.h as i32, d.hc as i32, d.eps, self.hi);
        }
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
                && dsh.max(ush) <= sm.hc_max
                && nch <= 32
                && n * hc.r <= s.low.bytes / 4
                && nch * n * hc.r * 4 <= s.hcp.bytes;
            if fits {
                unsafe {
                    self.c.launch(
                        sm.hc_di[n - 1],
                        (hc.r.div_ceil(t.di_t / 32) as u32, nch as u32),
                        t.di_t as u32,
                        dsh,
                        kargs!(hc.di.ptr, s.normed.ptr, s.hcp.ptr, hc.r as i32, d.w as i32, cw as i32, self.hi, s.low.ptr, s.iw.ptr, s.tick.ptr, hc.lr as i32, d.hc as i32),
                    )
                }?;
                let nl = hc.lr / per;
                if d.hc == 4 && nl <= 48 && !std::env::var("CMF_QWEN_HC_UP4").is_ok_and(|v| v.trim() == "0") {
                    unsafe {
                        self.c.launch(
                            sm.hc_up4[n - 1],
                            (d.h.div_ceil(t.up_t / 32) as u32, 1),
                            t.up_t as u32,
                            ush,
                            kargs!(
                                hc.up.ptr,
                                s.low.ptr,
                                s.normed.ptr,
                                s.folded.ptr,
                                s.folded_h.ptr,
                                d.h as i32,
                                hc.lr as i32,
                                hc.r as i32,
                                self.hi
                            ),
                        )
                    }?;
                    return Ok(());
                }
                unsafe {
                    self.c.launch(
                        sm.hc_up[n - 1],
                        (d.h.div_ceil(t.up_t / 32) as u32, 1),
                        t.up_t as u32,
                        ush,
                        kargs!(
                            hc.up.ptr,
                            s.low.ptr,
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

    /// The down projections of `nu` units (`s.g` -> `s.d`): in the fast precision the int8 kernel with four lanes
    /// a row (`q4_exp_d4`, rows of up to 32 groups; `CMF_QWEN_EXP_D4=0` keeps `q4_exp_d`).
    fn exp_down(&self, nu: usize, dsh: usize) -> Res<()> {
        let (d, s, c, sm, t) = (&self.d, &self.s, self.c, &self.q.sm, &self.tune);
        let gd = d.inter / 32;
        if exp_d4_on() && d.inter.is_multiple_of(32) && gd.div_ceil(4) <= 8 {
            let rb = 64usize;
            let sh4 = (UNIT_ROWS * gd * 12 + rb * 32) * 4;
            if sh4 <= sm.ex_max {
                return unsafe {
                    c.launch(
                        sm.exp_d4,
                        (d.h.div_ceil(rb) as u32, nu as u32),
                        256,
                        sh4,
                        kargs!(s.slots.ptr, s.g.ptr, s.d.ptr, d.inter as i32, d.h as i32, rb as i32, d.sgu as u64),
                    )
                };
            }
        }
        unsafe {
            c.launch(
                sm.exp_d,
                (d.h.div_ceil(t.d_rb) as u32, nu as u32),
                t.d_t as u32,
                dsh,
                kargs!(s.slots.ptr, s.g.ptr, s.d.ptr, d.inter as i32, d.h as i32, t.d_rb as i32, d.sgu as u64),
            )
        }
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
        scan(c, q, d, mix, gb, g.s.ptr, g.s2.ptr, s.qo.ptr, n)?;
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
            // the cells before `pos0` that the old buffers hold: the MTP head keeps a few cells and is asked
            // for a position far past them after a long prompt, where a copy of `pos0` cells read past the old
            // buffers' end and the head was lost for the session
            let kept = pos0.min(a.cap);
            regrow(&mut a.keys, cap * kv * 4, kept * kv * 4)?;
            regrow(&mut a.vals, cap * kv * 4, kept * kv * 4)?;
            regrow(&mut a.ikraw, cap * d.idim * 4, kept * d.idim * 4)?;
            regrow(&mut a.bkey, (cap / d.cr + 1) * d.idim * 4, (kept / d.cr) * d.idim * 4)?;
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
            let room = sm.ex_max / 4;
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
                // the host computes the misses: the route stays whole and the host path runs, with no table
                let cpu_on = small && cpu_experts() && (!head || arena.head_host) && d.gbits == 4 && !lw.has_tubes;
                if !cpu_on {
                    arena.land();
                }
                let tbl_p = if cpu_on { 0 } else { arena.tbl_ptr(c, li, topk, head)? };
                // the route is masked by the table in the backbone's layers; the head routes over all its experts
                let mask_p = if head { 0u64 } else { tbl_p };
                go!(c, q.route, (n as u32, 1), 256; s.logits.ptr, s.picks.ptr, s.wts.ptr, lw.ne as i32, lw.r as i32, topk as i32, self.tail, mask_p);
                let mut lean_saved: Option<Vec<f32>> = None;
                if small && self.lean_on() && !lw.has_tubes && tbl_p != 0 && ur == UNIT_ROWS && lw.ne <= arena.tbl_stride {
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
                                if d.gbits == 2 { sm.exp_gu2 } else { sm.exp_gu },
                                (d.inter.div_ceil(t.g_rb) as u32, nu as u32),
                                t.g_t as u32,
                                gsh,
                                kargs!(s.slots.ptr, s.folded.ptr, s.g.ptr, d.inter as i32, d.h as i32, t.g_rb as i32),
                            )
                        }?;
                        self.exp_down(nu, dsh)?;
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
                let mut picks_own: Vec<i32> = Vec::new();
                let (mut xs_host, mut rows_host): (&[f32], &mut [f32]) = (&[], &mut []);
                let picks: &[i32] = if cpu_on {
                    // the host's share: the picks and the experts' input come down in the same wait, into the
                    // scratch the rows later go up from
                    let (pp, xp, rp) = arena.host.parts(n * topk, n * d.h, n * topk * d.h, || c.sync())?;
                    // SAFETY: the scratch lives as long as the arena and only this call uses it; the upload of
                    // the rows of the layer before ended by the wait below, ahead of any write here
                    let (pk, xs, rows) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(pp, n * topk),
                            std::slice::from_raw_parts_mut(xp, n * d.h),
                            std::slice::from_raw_parts_mut(rp, n * topk * d.h),
                        )
                    };
                    unsafe { cu::memcpy_dtoh_async(&mut *pk, s.picks.ptr, c.stream) }.map_err(derr)?;
                    unsafe { cu::memcpy_dtoh_async(&mut *xs, s.folded.ptr, c.stream) }.map_err(derr)?;
                    (xs_host, rows_host) = (xs, rows);
                    pk
                } else {
                    picks_own.resize(n * topk, 0);
                    unsafe { cu::memcpy_dtoh_async(&mut picks_own, s.picks.ptr, c.stream) }.map_err(derr)?;
                    &picks_own
                };
                let t_wait = Instant::now();
                if let Some((l0, t)) = arena.host_at
                    && cpu_on
                    && l0 + 1 == li
                {
                    arena.host_ms[0] += (t_wait - t).as_secs_f64() * 1e3;
                }
                wait_stream(c)?;
                let t_picks = Instant::now();
                if cpu_on {
                    arena.host_ms[1] += (t_picks - t_wait).as_secs_f64() * 1e3;
                    arena.host_layers += 1;
                }
                arena.pending = false;
                // what the stagers have brought in since the last layer is a hit from here on
                arena.land();
                arena.note(li, picks);
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
                // a hot expert is an 8-bit tube on the card: its own q2tp triple is not read
                let (hot, cold): (Vec<usize>, Vec<usize>) = order.iter().copied().partition(|&e| lw.tubes[e] >= 0);
                // the arena's misses for the host (`cpu_on`), or admitted
                let mut cpu_miss: Vec<usize> = Vec::new();
                let cold: Vec<usize> = if cpu_on {
                    let mut held = Vec::with_capacity(cold.len());
                    for &e in &cold {
                        match arena.hit(li, e) {
                            Some(a) => {
                                slot_of[e] = a;
                                held.push(e);
                            }
                            None => cpu_miss.push(e),
                        }
                    }
                    held
                } else {
                    if !arena.resolve_layer(li, &cold, &mut slot_of) {
                        arena.admit_layer(li, &cold, &layer.moe.experts, d, c, &mut slot_of)?;
                    }
                    cold
                };
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
                let mut nu_q = 0usize;
                for &e in &cold {
                    add(slot_of[e], &by_expert[e]);
                    nu_q += by_expert[e].len().div_ceil(ur);
                }
                let shared_places: Vec<usize> = (0..n).map(|t| t * kk + topk).collect();
                if let (Some(sh), true) = (&lw.shared, lw.shared_tube < 0) {
                    add(sh.ptr, &shared_places);
                    nu_q += n.div_ceil(ur);
                }
                // the tubes' units follow the q-units' (the tube kernels start at unit `nu_q`)
                for &e in &hot {
                    add(self.tube_desc + lw.tubes[e] as u64 * 72, &by_expert[e]);
                }
                if lw.shared_tube >= 0 {
                    add(self.tube_desc + lw.shared_tube as u64 * 72, &shared_places);
                }
                let nu = units.len() / UNIT_WORDS;
                // the host's rows of the partial outputs follow the units' rows
                let cpu_base = nu * UNIT_ROWS;
                let mut cpu_rows = 0usize;
                for &e in &cpu_miss {
                    for &p in &by_expert[e] {
                        refs[p] = (cpu_base + cpu_rows) as i32;
                        cpu_rows += 1;
                    }
                }
                if (cpu_base + cpu_rows) * d.h * 4 > s.d.bytes {
                    return Err("qwen4 cuda: the host's expert rows do not fit the partial outputs".into());
                }
                if nu == 0 && cpu_rows == 0 {
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
                if nu_q > 0 {
                    unsafe {
                        c.launch(
                            if d.gbits == 2 { sm.exp_gu2 } else { sm.exp_gu },
                            (d.inter.div_ceil(t.g_rb) as u32, nu_q as u32),
                            t.g_t as u32,
                            gsh,
                            kargs!(s.slots.ptr, s.folded.ptr, s.g.ptr, d.inter as i32, d.h as i32, t.g_rb as i32),
                        )
                    }?;
                    self.exp_down(nu_q, dsh)?;
                }
                if nu > nu_q {
                    // the tubes: 16 rows a block, both matrices staged for every row of a unit
                    let (rb, tu) = (16usize, s.slots.ptr + (nu_q * UNIT_WORDS * 4) as u64);
                    let tsh_gu = 2 * UNIT_ROWS * 16 * ((d.h / 16) | 1) * 4;
                    let tsh_d = UNIT_ROWS * 16 * ((d.inter / 16) | 1) * 4;
                    unsafe {
                        c.launch(
                            sm.tube_gu,
                            (d.inter.div_ceil(rb) as u32, (nu - nu_q) as u32),
                            256,
                            tsh_gu,
                            kargs!(tu, s.folded.ptr, s.g.ptr, d.inter as i32, d.h as i32, rb as i32, nu_q as i32),
                        )
                    }?;
                    unsafe {
                        c.launch(
                            sm.tube_d,
                            (d.h.div_ceil(rb) as u32, (nu - nu_q) as u32),
                            256,
                            tsh_d,
                            kargs!(tu, s.g.ptr, s.d.ptr, d.inter as i32, d.h as i32, rb as i32, nu_q as i32),
                        )
                    }?;
                }
                let t_units = Instant::now();
                if cpu_on && !cpu_miss.is_empty() {
                    // behind the layer's kernels, under the host's work on this window's rows
                    arena.admit_hot(li, &cpu_miss, &layer.moe.experts, d, c)?;
                }
                let t_admit = Instant::now();
                if cpu_on {
                    arena.host_ms[2] += (t_units - t_picks).as_secs_f64() * 1e3;
                    arena.host_ms[3] += (t_admit - t_units).as_secs_f64() * 1e3;
                }
                let mut t_rows = t_admit;
                if cpu_rows > 0 {
                    // the card runs the held experts meanwhile
                    let t0 = Instant::now();
                    let rows = &mut rows_host[..cpu_rows * d.h];
                    if !cpu_expert_rows(&layer.moe, &cpu_miss, &by_expert, kk, xs_host, d.h, rows) {
                        return Err("qwen4 cuda: the CPU expert kernels do not cover this layer (CMF_QWEN_CPU_EXPERTS)".into());
                    }
                    t_rows = Instant::now();
                    arena.cpu_ms += (t_rows - t0).as_secs_f64() * 1e3;
                    self.put(s.d.ptr + (cpu_base * d.h * 4) as u64, rows)?;
                    arena.cpu_rows += cpu_rows as u64;
                    arena.cpu_experts += cpu_miss.len() as u64;
                }
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
                if cpu_on {
                    let now = Instant::now();
                    arena.host_ms[4] += (now - t_rows).as_secs_f64() * 1e3;
                    arena.host_at = Some((li, now));
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
        // a hot expert's weights are its tube: no slot (the expansion skips slot 0), the tube is expanded instead
        let cold: Vec<usize> = order.iter().copied().filter(|&e| lw.tubes[e] < 0).collect();
        // `hold` (the MTP head's step) keeps its experts in the arena already
        if !(li < arena.main_layers() && arena.resolve_layer(li, &cold, &mut slot_of)) {
            arena.admit_layer(li, &cold, &layer.moe.experts, d, c, &mut slot_of)?;
        }
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
                s.slots.ptr, s.wg.ptr, s.wu.ptr, s.wd.ptr, d.inter as i32, d.h as i32, self.hi, d.gbits as i32);
            for j in 0..b {
                let t = lw.tubes[order[i + j]];
                if t >= 0 {
                    self.expand_tube(t as usize, j)?;
                }
            }
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
            if lw.shared_tube >= 0 {
                self.expand_tube(lw.shared_tube as usize, 0)?;
            } else {
                self.put(s.slots.ptr, &[sh.ptr])?;
                go!(c, q.expand_exp, (blocks(d.inter * d.h / 8), 3), 256;
                    s.slots.ptr, s.wg.ptr, s.wu.ptr, s.wd.ptr, d.inter as i32, d.h as i32, self.hi, d.gbits as i32);
            }
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

    /// A latent layer's MoE (Edge0): the route as the base model's, D, the routed experts in their
    /// groups' latent rows, U, the shared expert. A few rows take the per-pick kernels straight from the
    /// NVFP4 slots; a chunk expands each sub-batch's experts for the batched GEMMs.
    fn moe_lat(&self, lw: &LatW, li: usize, n: usize) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let (l, g, ff, ne, topk) = (lw.l, lw.groups, lw.ff, lw.ne, lw.topk);
        let gl = g * l;
        let small = self.small(n);
        if small {
            self.mvb(&[SOp::f32(lw.router.ptr, lw.r, s.logits.ptr)], s.folded.ptr, d.h, d.h, n)?;
        } else {
            self.gemm_w(lw.router.ptr, lw.r, d.h, s.folded.ptr, n, s.logits.ptr, Gemm::F32)?;
        }
        let tail = if small { self.tail } else { 0.0f32 };
        go!(c, q.route, (n as u32, 1), 256; s.logits.ptr, s.picks.ptr, s.wts.ptr, ne as i32, lw.r as i32, topk as i32, tail, 0u64);
        self.lap(5)?;
        let half = self.hi != 0;
        let (inv, alpha) = if half { (1.0f32 / 16.0, 16.0f32) } else { (1.0f32, 1.0f32) };
        if !small && n * topk <= LAT_PICKS {
            // a short chunk: the dense parts as GEMMs, the experts straight from their slots
            let dp = self.plane_q8(&lw.dmat)?;
            self.gemm_w(dp, gl, d.h, s.folded_h.ptr, n, s.lxl.ptr, self.gw)?;
            self.lat_picks(lw, n)?;
            let acc_g = if half {
                go!(c, q.to_wt, (blocks(n * gl), 1), 256; s.lacc.ptr, s.lacch.ptr, (n * gl) as u64, 1i32);
                s.lacch.ptr
            } else {
                s.lacc.ptr
            };
            let up = self.plane_q8(&lw.umat)?;
            self.gemm_w(up, d.h, gl, acc_g, n, s.blk.ptr, self.gw)?;
            self.lap(7)?;
            return self.lat_shared_chunk(lw, n, inv, alpha);
        }
        if small {
            // D for the groups picked, the picks' gate/up and down straight from their slots, the groups'
            // sums, U over the groups picked (each matrix read once for every row of the round)
            go!(c, q.lat_gmask, (1u32, 1), 32; s.picks.ptr, lw.labels.ptr, s.lmask.ptr, n as i32, topk as i32);
            self.lat_mv(&lw.dmat, s.folded.ptr, d.h, s.lxl.ptr, gl, n, l, 0, 1280)?;
            self.lat_picks(lw, n)?;
            self.lat_mv(&lw.umat, s.lacc.ptr, gl, s.lpart.ptr, d.h, n, l, 1, l / 2)?;
            go!(c, q.lat_red, (blocks(n * d.h), 1), 256; s.lpart.ptr, s.lmask.ptr, s.blk.ptr, d.h as i32, d.h as i32, n as i32, g as i32);
            self.lap(7)?;
            // the shared expert at full width
            self.mvb(&[SOp::q8(&lw.sh[0], s.g.ptr), SOp::q8(&lw.sh[1], s.u.ptr)], s.folded.ptr, d.h, d.h, n)?;
            go!(c, q.act, (blocks(n * ff), 1), 256; s.g.ptr, s.u.ptr, s.ah.ptr, (n * ff) as u64, 0i32, 0i32, 1.0f32);
            self.mvb(&[SOp::q8(&lw.sh[2], s.d.ptr)], s.ah.ptr, ff, ff, n)?;
            go!(c, q.shared_add, (blocks(n * d.h), 1), 256; s.blk.ptr, s.d.ptr, s.logits.ptr, n as i32, d.h as i32, lw.r as i32, lw.gcol);
            return self.lap(8);
        }
        // D over the chunk, its GEMM-typed copy for the gathers
        let dp = self.plane_q8(&lw.dmat)?;
        self.gemm_w(dp, gl, d.h, s.folded_h.ptr, n, s.lxl.ptr, self.gw)?;
        let xl_g = if half {
            go!(c, q.to_wt, (blocks(n * gl), 1), 256; s.lxl.ptr, s.lxlh.ptr, (n * gl) as u64, 1i32);
            s.lxlh.ptr
        } else {
            s.lxl.ptr
        };
        let mut picks = vec![0i32; n * topk];
        unsafe { cu::memcpy_dtoh_async(&mut picks, s.picks.ptr, c.stream) }.map_err(derr)?;
        c.sync()?;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
        for (i, &e) in picks.iter().enumerate() {
            if e >= 0 && (e as usize) < ne {
                lists[e as usize].push(i as u32);
            }
        }
        let mut order: Vec<usize> = (0..ne).filter(|&e| !lists[e].is_empty()).collect();
        order.sort_by(|&x, &y| lists[y].len().cmp(&lists[x].len()).then(x.cmp(&y)));
        let mut i = 0usize;
        while i < order.len() {
            let ne_max = lists[order[i]].len();
            if ne_max > self.xr {
                return Err("qwen4 cuda: a latent expert with more rows than the sub-batch".into());
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
            let mut rowpick = vec![-1i32; rows];
            // the sub-batch's GEMM alphas (the mean globals) and each expert's factor against them
            let (mut ag, mut au, mut ad) = (0f32, 0f32, 0f32);
            for j in 0..b {
                let gl3 = lw.glob[order[i + j]];
                (ag, au, ad) = (ag + gl3[0].abs(), au + gl3[1].abs(), ad + gl3[2].abs());
            }
            let nb = b as f32;
            let (ag, au, ad) = ((ag / nb).max(1e-30), (au / nb).max(1e-30), (ad / nb).max(1e-30));
            let mut f = vec![0f32; 3 * b];
            for j in 0..b {
                let e = order[i + j];
                slots[j] = lw.tbl_host[e];
                let gl3 = lw.glob[e];
                f[2 * j] = gl3[0] / ag;
                f[2 * j + 1] = gl3[1] / au;
                f[2 * b + j] = gl3[2] / ad;
                let grp = lw.labels_host[e];
                for (r, &entry) in lists[e].iter().enumerate() {
                    rowtok[j * ne_max + r] = (entry as usize / topk) as i32 * g as i32 + grp;
                    rowpick[j * ne_max + r] = entry as i32;
                }
            }
            self.put(s.slots.ptr, &slots)?;
            self.put(s.rowtok.ptr, &rowtok)?;
            self.put(s.lrp.ptr, &rowpick)?;
            self.put(s.lf.ptr, &f)?;
            go!(c, q.lat_expand, (blocks(ff * l / 8), (3 * b) as u32), 256;
                s.slots.ptr, s.wg.ptr, s.wu.ptr, s.wd.ptr, ff as i32, l as i32, self.hi);
            go!(c, q.gather, (blocks(rows * l), 1), 256; xl_g, s.rowtok.ptr, s.xg.ptr, rows as i32, l as i32, self.hi);
            let (gs, xs, as_, ds) = (ff * l, ne_max * l, ne_max * ff, ne_max * l);
            unsafe {
                if half {
                    gemm_batched_half(c, true, ff, ne_max, l, ag, s.wg.ptr, l, gs, s.xg.ptr, l, xs, s.g.ptr, ff, as_, b)?;
                    gemm_batched_half(c, true, ff, ne_max, l, au, s.wu.ptr, l, gs, s.xg.ptr, l, xs, s.u.ptr, ff, as_, b)?;
                } else {
                    c.gemm_batched(true, ff, ne_max, l, ag, s.wg.ptr, l, gs, s.xg.ptr, l, xs, s.g.ptr, ff, as_, b, self.gw)?;
                    c.gemm_batched(true, ff, ne_max, l, au, s.wu.ptr, l, gs, s.xg.ptr, l, xs, s.u.ptr, ff, as_, b, self.gw)?;
                }
            }
            go!(c, q.lat_act, (blocks(rows * ff), 1), 256; s.g.ptr, s.u.ptr, s.ah.ptr, s.lf.ptr, rows as i32, ne_max as i32, ff as i32, self.hi, self.hi, inv);
            unsafe {
                if half {
                    gemm_batched_half(c, true, l, ne_max, ff, alpha * ad, s.wd.ptr, ff, gs, s.ah.ptr, ff, as_, s.d.ptr, l, ds, b)?;
                } else {
                    c.gemm_batched(true, l, ne_max, ff, alpha * ad, s.wd.ptr, ff, gs, s.ah.ptr, ff, as_, s.d.ptr, l, ds, b, self.gw)?;
                }
            }
            go!(c, q.lat_put, (blocks(rows * l), 1), 256;
                s.d.ptr, s.lrp.ptr, s.lf.ptr + (2 * b * 4) as u64, s.lyc.ptr, rows as i32, l as i32, ne_max as i32, self.hi);
            i += b;
        }
        // the groups' sums over every pick of the chunk, in pick order
        go!(c, q.lat_sum, (blocks(n * gl), 1), 256; s.picks.ptr, lw.labels.ptr, s.wts.ptr, s.lyc.ptr, s.lacc.ptr, n as i32, g as i32, l as i32, topk as i32);
        // U over the groups' sums
        let acc_g = if half {
            go!(c, q.to_wt, (blocks(n * gl), 1), 256; s.lacc.ptr, s.lacch.ptr, (n * gl) as u64, 1i32);
            s.lacch.ptr
        } else {
            s.lacc.ptr
        };
        let up = self.plane_q8(&lw.umat)?;
        self.gemm_w(up, d.h, gl, acc_g, n, s.blk.ptr, self.gw)?;
        self.lap(7)?;
        let _ = li;
        self.lat_shared_chunk(lw, n, inv, alpha)
    }

    /// `y [n][rows]` = W x over the groups the round picked (`q4_lat_mv`), W q8_2f.
    #[allow(clippy::too_many_arguments)]
    fn lat_mv(&self, w: &DevMat, x: u64, xs: usize, y: u64, ys: usize, n: usize, glen: usize, mode: i32, ch: usize) -> Res<()> {
        let (c, q, s) = (self.c, self.q, &self.s);
        let rb = 16usize;
        if n > 8 || !w.cols.is_multiple_of(ch) || !ch.is_multiple_of(16) || (mode == 1 && !glen.is_multiple_of(ch)) || (mode == 0 && !glen.is_multiple_of(rb)) {
            return Err(format!("qwen4 cuda: latent matvec shape {}x{} chunk {ch} group {glen}", w.rows, w.cols));
        }
        // mode 1: one row of blocks per group of columns
        let gy = if mode == 1 { (w.cols / glen) as u32 } else { 1 };
        unsafe {
            c.launch(
                q.lat_mv,
                (w.rows.div_ceil(rb) as u32, gy),
                256,
                n * ch * 4,
                kargs!(w.w.ptr, w.aux0(), w.aux1(), x, xs as i32, y, ys as i32, w.rows as i32, w.cols as i32, n as i32, glen as i32, s.lmask.ptr, mode, ch as i32),
            )
        }?;
        Ok(())
    }

    /// The routed experts of `n` rows straight from their NVFP4 slots: every pick's gate/up and
    /// down, then the groups' sums into `lacc`.
    fn lat_picks(&self, lw: &LatW, n: usize) -> Res<()> {
        let (s, c, q) = (&self.s, self.c, self.q);
        let (l, g, ff, topk) = (lw.l, lw.groups, lw.ff, lw.topk);
        let (rb, routes) = (8usize, n * topk);
        unsafe {
            c.launch(
                q.lat_gu,
                (ff.div_ceil(rb) as u32, routes as u32),
                256,
                l * 4,
                kargs!(s.picks.ptr, lw.labels.ptr, lw.tbl.ptr, s.lxl.ptr, s.lact.ptr, ff as i32, l as i32, g as i32, topk as i32, rb as i32),
            )
        }?;
        unsafe {
            c.launch(
                q.lat_dn,
                (l.div_ceil(rb) as u32, routes as u32),
                256,
                ff * 4,
                kargs!(s.picks.ptr, lw.tbl.ptr, s.lact.ptr, s.ly.ptr, ff as i32, l as i32, rb as i32),
            )
        }?;
        go!(c, q.lat_sum, (blocks(n * g * l), 1), 256; s.picks.ptr, lw.labels.ptr, s.wts.ptr, s.ly.ptr, s.lacc.ptr, n as i32, g as i32, l as i32, topk as i32);
        Ok(())
    }

    /// The shared expert of a chunk: one GEMM per projection over all tokens, gated into `blk`.
    fn lat_shared_chunk(&self, lw: &LatW, n: usize, inv: f32, alpha: f32) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let ff = lw.ff;
        let sg = self.plane_q8(&lw.sh[0])?;
        self.gemm_w(sg, ff, d.h, s.folded_h.ptr, n, s.g.ptr, self.gw)?;
        let su = self.plane_q8(&lw.sh[1])?;
        self.gemm_w(su, ff, d.h, s.folded_h.ptr, n, s.u.ptr, self.gw)?;
        go!(c, q.act, (blocks(n * ff), 1), 256; s.g.ptr, s.u.ptr, s.ah.ptr, (n * ff) as u64, 0i32, self.hi, inv);
        let sd = self.plane_q8(&lw.sh[2])?;
        unsafe { c.gemm(true, false, d.h, n, ff, alpha, sd, ff, s.ah.ptr, ff, 0.0, s.d.ptr, d.h, self.gw) }?;
        go!(c, q.shared_add, (blocks(n * d.h), 1), 256; s.blk.ptr, s.d.ptr, s.logits.ptr, n as i32, d.h as i32, lw.r as i32, lw.gcol);
        self.lap(8)
    }

    /// Tube `t` (a hot expert's gate, up and down at eight bits) as dense planes in the batched path's
    /// expert scratch, at the place of expert `j` of the sub-batch.
    fn expand_tube(&self, t: usize, j: usize) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let w = &self.tube_host[t * 9..t * 9 + 9];
        let plane = (j * d.inter * d.h * self.es) as u64;
        for (m, dst) in [s.wg.ptr, s.wu.ptr, s.wd.ptr].into_iter().enumerate() {
            let (rows, cols) = if m == 2 { (d.h, d.inter) } else { (d.inter, d.h) };
            go!(c, q.expand_q8, (blocks(rows * cols / 4), 1), 256; w[3 * m], w[3 * m + 1], w[3 * m + 2], dst + plane, rows as i32, cols as i32, self.hi);
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
        self.head_rows_v(head, lm, n, ids, 0, false)
    }

    /// `head_rows`, choosing among the first `vp` tokens of the vocabulary when `vp` is set (a draft: the
    /// frequent ids carry nearly all of its picks, and the head's matvec is a tenth of the bytes).
    /// A `draft` pass may use the int8 head too: what it picks is checked by the verify window.
    fn head_rows_v(&self, head: &Hc, lm: &DevMat, n: usize, ids: &mut Vec<u32>, vp: usize, draft: bool) -> Res<()> {
        let (d, s, c, q) = (&self.d, &self.s, self.c, self.q);
        let vp = if vp > 0 && vp < d.vocab && lm.kind == 7 && self.small { vp } else { d.vocab };
        self.hc_mix(head, s.hyper.ptr, n)?;
        let mut held: Vec<Buf> = Vec::new();
        let mut t0 = 0;
        while t0 < n {
            let rows = (n - t0).min(SMALL_ROWS);
            let x = s.folded.ptr + (t0 * d.h * 4) as u64;
            if lm.kind == 7 && self.small {
                let mut op = SOp::q8(lm, s.out.ptr);
                op.rows = vp;
                self.mvb_x(&[op], x, d.h, d.h, rows, !draft)?;
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
            go!(c, q.argmax, (rows as u32, 1), 1024; s.out.ptr, s.ids.ptr + (t0 * 4) as u64, vp as i32);
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
        // the inject that closes a block is folded into the norm launch that opens the next one
        // (CMF_QWEN_FUSE_INJ=0 keeps them apart); a layer's PLE reads the hyper state, so it flushes first
        let fuse = !tr && std::env::var("CMF_QWEN_FUSE_INJ").map(|v| v != "0").unwrap_or(true);
        let mut pend = false;
        for (li, (lw, layer)) in lws.iter_mut().zip(layers).enumerate() {
            if let Some(p) = &mut lw.ple
                && let rows = ple(li)
                && !rows.is_empty()
            {
                if pend {
                    k.inject(n)?;
                    pend = false;
                }
                k.ple(p, &rows, n, hold)?;
                if tr {
                    trace.push(("post_ple", li, last(s.hyper.ptr, d.w)?));
                }
                k.lap(1)?;
            }
            k.hc_mix_inj(&lw.attn, s.hyper.ptr, n, pend)?;
            pend = false;
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
            if !fuse {
                k.inject(n)?;
            }
            if tr {
                trace.push(("post_attn", li, last(s.hyper.ptr, d.w)?));
            }
            k.hc_mix_inj(&lw.mlp, s.hyper.ptr, n, fuse)?;
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
            match &lw.lat {
                Some(lt) => k.moe_lat(lt, li, n)?,
                None => k.moe(&lw.moe, layer, li, arena, n, hold)?,
            }
            if tr {
                trace.push(("moe_out", li, last(s.blk.ptr, d.h)?));
            }
            if fuse {
                pend = true;
            } else {
                k.inject(n)?;
            }
            if tr {
                trace.push(("post_moe", li, last(s.hyper.ptr, d.w)?));
            }
        }
        if pend {
            k.inject(n)?;
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
                    scan(c, q, d, g.t_mix.ptr, g.t_gb.ptr, g.s.ptr, g.s2.ptr, s.qo.ptr, m)?;
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
