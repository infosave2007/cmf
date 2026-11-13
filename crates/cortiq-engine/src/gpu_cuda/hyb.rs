//! The Qwen3.5-family hybrid on CUDA (Qwen3.5 / 3.6 / 3.8 dense: GDN layers
//! between gated full-attention layers with Q/K norms and a partial rotary
//! turn), behind the same token and batch graph contracts the dense arm of
//! `gpu_cuda.rs` serves. The pipeline's MTP rounds, prompt lookup and server
//! run on it unchanged.
//!
//! Two paths:
//! - **small** (a decode step, a speculative verify window, an MTP warm-up:
//!   up to `NMAX` rows): weights read in their codec by `hy_mvn_N`, which
//!   reads a matrix once for all the rows of the window; every row goes
//!   through the arithmetic of a one-row step in the same order, so a verify
//!   window's rows agree to the bit with the decode steps they stand for, and
//!   a greedy answer is the same with drafts and without;
//! - **big** (prompt chunks): each weight expanded to an f16 plane per chunk,
//!   cuBLAS on the tensor cores, attention as two batched GEMMs a KV group.
//!
//! GDN state (the delta-rule matrix and the conv ring) lives on the card per
//! (sequence, layer). A verify window keeps the state it started from and a
//! tape of its rows' inputs; `gdn_spec_restore` puts the start back and
//! replays the accepted rows.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use cortiq_core::CmfModel;

use super::{Buf, Ctx, DevMat, Gemm, KvMirror, Res, State, compile, cs, cu, debug, decline, derr, func, prec};
use crate::gpu::{BatchGraphOutcome, GraphAct, GraphAttn, GraphFfn, GraphLayer, GraphW, SpecTail, TokenGraphOutcome};

const HYB_SRC: &str = include_str!("hyb.cu");

/// The DFlash 2 block on the card, on this arm's kernels.
pub(super) mod dflash;
pub(super) mod dory;
/// Rows of the small path.
pub(super) const NMAX: usize = 16;
/// Rows of a prompt chunk read past a seal (`big` over sealed layers): the far
/// field's mirror keeps room for them after its near rows.
pub(super) const SEAL_ROWS: usize = 1024;
/// A prompt chunk's weight expands to f16 in bands of at most this many bytes
/// (`CMF_HYB_WD_MB`): the gate/up pair of a 27B layer is 356 MB whole.
fn wd_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| std::env::var("CMF_HYB_WD_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64) << 20)
}
/// Rows of a chunk one attention GEMM pass takes at most (`CMF_HYB_ATT_ROWS`):
/// its score and weight buffers grow with rows x keys.
fn att_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_HYB_ATT_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 16).unwrap_or(128))
}
/// Stages of `CMF_CUDA_PROF=1` on the small path (each closed by a stream
/// sync, so the numbers are the card's).
const HSTAGES: [&str; 12] = [
    "norms", "gdn in", "gdn core", "gdn out", "attn in", "attn core", "attn out", "gate+up", "silu", "down",
    "head", "copies",
];
static HSTAGE_NS: [std::sync::atomic::AtomicU64; 12 * 17] = [const { std::sync::atomic::AtomicU64::new(0) }; 12 * 17];
static HCALLS_N: [std::sync::atomic::AtomicU64; 17] = [const { std::sync::atomic::AtomicU64::new(0) }; 17];
static HCALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
thread_local! {
    /// The row count of the small-path call being timed (`hst`).
    static HROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}
/// Small-path graphs recorded and replayed (`CMF_CUDA_DEBUG=1` reports them).
static GRAPH_REC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GRAPH_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Stages of the prompt path (`CMF_CUDA_PROF=1`), summed over its chunks.
const BSTAGES: [&str; 8] = ["norms", "expand", "gemm", "gdn glue", "gdn scan", "attention", "act", "copies"];
static BSTAGE_NS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
static BROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn bst(c: &Ctx, s: usize, t0: &std::cell::Cell<std::time::Instant>) -> Res<()> {
    if super::prof_on() {
        c.sync()?;
        BSTAGE_NS[s].fetch_add(t0.get().elapsed().as_nanos() as u64, Ordering::Relaxed);
        t0.set(std::time::Instant::now());
    }
    Ok(())
}

fn hst(c: &Ctx, s: usize, t0: &mut std::time::Instant) -> Res<()> {
    if super::prof_on() {
        c.sync()?;
        let n = HROWS.with(|r| r.get()).min(16);
        HSTAGE_NS[n * 12 + s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        *t0 = std::time::Instant::now();
    }
    Ok(())
}

/// The small path's stage times, per call (`CMF_CUDA_PROF=1`).
pub(super) fn report() {
    if debug() {
        eprintln!(
            "cuda hybrid graphs: {} recorded, {} replays",
            GRAPH_REC.load(Ordering::Relaxed),
            GRAPH_RUN.load(Ordering::Relaxed)
        );
    }
    let calls = HCALLS.load(Ordering::Relaxed);
    if calls == 0 || !super::prof_on() {
        return;
    }
    eprintln!("cuda hybrid small path: {calls} calls");
    for n in 1..=16 {
        let k = HCALLS_N[n].load(Ordering::Relaxed);
        if k == 0 {
            continue;
        }
        let row: Vec<String> = HSTAGES
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{name} {:.3}", HSTAGE_NS[n * 12 + i].load(Ordering::Relaxed) as f64 / 1e6 / k as f64))
            .collect();
        let tot: f64 = (0..12).map(|i| HSTAGE_NS[n * 12 + i].load(Ordering::Relaxed) as f64 / 1e6 / k as f64).sum();
        eprintln!("  {n:>2} rows, {k} calls, {tot:.2} ms a call: {}", row.join(" | "));
    }
    let rows = BROWS.load(Ordering::Relaxed);
    if rows > 0 {
        eprintln!("cuda hybrid prompt path: {rows} rows");
        for (name, ns) in BSTAGES.iter().zip(&BSTAGE_NS) {
            eprintln!("  {name:>10}: {:8.3} ms per 1000 rows", ns.load(Ordering::Relaxed) as f64 / 1e6 / rows as f64 * 1000.0);
        }
    }
}

/// `struct HyJob` of `hyb.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct HyJob {
    w: u64,
    aux0: u64,
    aux1: u64,
    y: u64,
    res: u64,
    rows: i32,
    kind: i32,
    blk0: i32,
    ystride: i32,
    /// Elements into the input planes this job's rows start at (a MoE slot's own input).
    xoff: i32,
    /// > 0: this job's input rows, fewer than the launch's (a grouped expert's).
    nr: i32,
}

pub(super) struct HFuncs {
    /// `hy_mmp_{NT}_{TB}_{KS}` by [NT 1, 2, 4 (windows of 4, 8, 16 rows)][TB - 1][KS == 16].
    mmp: [[[cs::CUfunction; 2]; 2]; 3],
    /// One codec's windows, `hy_mmpk_{NT}_{TB}_{KS}_{KD}` by [NT as in
    /// `mmp`][TB - 1][KS == 16][`KDS` index of KD].
    mmpk: [[[[cs::CUfunction; 6]; 2]; 2]; 3],
    /// The grouped experts' launches: `hy_mmp_8_2_8` (32 rows a job), `hy_mmp_16_1_8` (64).
    mmp_g: [cs::CUfunction; 2],

    split: cs::CUfunction,
    expand_p: cs::CUfunction,
    rmsnorm: cs::CUfunction,
    to_half: cs::CUfunction,
    ring: cs::CUfunction,
    scan: cs::CUfunction,
    front: cs::CUfunction,
    out32: cs::CUfunction,
    rmsnorm2: cs::CUfunction,
    conv32: cs::CUfunction,
    prep32: cs::CUfunction,
    gb32: cs::CUfunction,
    rs_scan: cs::CUfunction,
    rs_ring: cs::CUfunction,
    scan4: cs::CUfunction,
    rs_scan4: cs::CUfunction,
    gather: cs::CUfunction,
    argmax: cs::CUfunction,
    silu_pair: cs::CUfunction,
    field_planes: cs::CUfunction,
    dash_route: cs::CUfunction,
    dash_close: cs::CUfunction,
    dash_attend: cs::CUfunction,
    repack: cs::CUfunction,
    repack8: cs::CUfunction,
    expand_i8: cs::CUfunction,
    repack1: cs::CUfunction,
    expand_q1: cs::CUfunction,
    repack1t: cs::CUfunction,
    expand_q1t: cs::CUfunction,
    repack2: cs::CUfunction,
    expand_q2: cs::CUfunction,
    repack4t: cs::CUfunction,
    expand_q4g: cs::CUfunction,
    mhc_enter: cs::CUfunction,
    mhc_leave: cs::CUfunction,
    mhc_route: cs::CUfunction,
    mhc_logits: cs::CUfunction,
    mhc_merge: cs::CUfunction,
    prism_p: cs::CUfunction,
    prism_h: cs::CUfunction,
    qk: cs::CUfunction,
    scores: cs::CUfunction,
    values: cs::CUfunction,
    finish: cs::CUfunction,
    attn_out: cs::CUfunction,
    softmax_h: cs::CUfunction,
    far_insert: cs::CUfunction,
    far_prep: cs::CUfunction,
    far_prep_b: cs::CUfunction,
    expand_pr: cs::CUfunction,
    far_rows: cs::CUfunction,
    softmax_hf: cs::CUfunction,
    attn_out_f: cs::CUfunction,
    far_insert_n: cs::CUfunction,
    moe_router: cs::CUfunction,
    moe_select: cs::CUfunction,
    moe_combine: cs::CUfunction,
    moe_topk: cs::CUfunction,
    moe_group: cs::CUfunction,
    moe_gather: cs::CUfunction,
    moe_gcombine: cs::CUfunction,
    norm_add: cs::CUfunction,
    scale: cs::CUfunction,
    sconv: cs::CUfunction,
    head_gate: cs::CUfunction,
}
unsafe impl Send for HFuncs {}
unsafe impl Sync for HFuncs {}

fn load_funcs(c: &Ctx, hpk: usize, hd: usize) -> Res<HFuncs> {
    if hd % 32 != 0 || hd > 512 || hpk == 0 || hpk > 16 {
        return Err(format!("attention geometry {hpk} heads per KV head x {hd} outside the kernels"));
    }
    let module = compile(HYB_SRC, vec![format!("-DHPK={hpk}"), format!("-DHD={hd}")], c.arch)?;
    let mut mmp = [[[std::ptr::null_mut(); 2]; 2]; 3];
    for (ni, nt) in [1usize, 2, 4].into_iter().enumerate() {
        for tb in 1..=2usize {
            for (ki, ks) in [8usize, 16].into_iter().enumerate() {
                mmp[ni][tb - 1][ki] = func(module, &format!("hy_mmp_{nt}_{tb}_{ks}"))?;
            }
        }
    }
    let mut mmpk = [[[[std::ptr::null_mut(); 6]; 2]; 2]; 3];
    for (ni, nt) in [1usize, 2, 4].into_iter().enumerate() {
        for tb in 1..=2usize {
            for (ki, ks) in [8usize, 16].into_iter().enumerate() {
                for (di, kd) in KDS.into_iter().enumerate() {
                    mmpk[ni][tb - 1][ki][di] = func(module, &format!("hy_mmpk_{nt}_{tb}_{ks}_{kd}"))?;
                }
            }
        }
    }
    Ok(HFuncs {
        mmp,
        mmpk,
        mmp_g: [func(module, "hy_mmp_8_2_8")?, func(module, "hy_mmp_16_1_8")?],

        moe_topk: func(module, "hy_moe_topk")?,
        moe_group: func(module, "hy_moe_group")?,
        moe_gather: func(module, "hy_moe_gather")?,
        moe_gcombine: func(module, "hy_moe_gcombine")?,
        split: func(module, "hy_split")?,
        expand_p: func(module, "hy_expand_p")?,
        rmsnorm: func(module, "hy_rmsnorm")?,
        to_half: func(module, "hy_to_half")?,
        ring: func(module, "hy_gdn_ring")?,
        scan: func(module, "hy_gdn_scan")?,
        front: func(module, "hy_gdn_front")?,
        out32: func(module, "hy_gdn_out32")?,
        rmsnorm2: func(module, "hy_rmsnorm2")?,
        conv32: func(module, "hy_gdn_conv32")?,
        prep32: func(module, "hy_gdn_prep32")?,
        gb32: func(module, "hy_gdn_gb32")?,
        rs_scan: func(module, "hy_gdn_restore_scan")?,
        rs_ring: func(module, "hy_gdn_restore_ring")?,
        scan4: func(module, "hy_gdn_scan4")?,
        rs_scan4: func(module, "hy_gdn_restore_scan4")?,
        gather: func(module, "hy_embed_gather")?,
        argmax: func(module, "hy_argmax")?,
        silu_pair: func(module, "hy_silu_pair_half")?,
        field_planes: func(module, "hy_field_planes")?,
        dash_route: func(module, "hy_dash_route")?,
        dash_close: func(module, "hy_dash_close")?,
        dash_attend: func(module, "hy_dash_attend")?,
        repack: func(module, "hy_repack")?,
        repack8: func(module, "hy_repack8")?,
        expand_i8: func(module, "hy_expand_i8r")?,
        repack1: func(module, "hy_repack1")?,
        expand_q1: func(module, "hy_expand_q1r")?,
        repack1t: func(module, "hy_repack1t")?,
        expand_q1t: func(module, "hy_expand_q1tr")?,
        repack2: func(module, "hy_repack2")?,
        repack4t: func(module, "hy_repack4t")?,
        mhc_enter: func(module, "hy_mhc_enter")?,
        mhc_leave: func(module, "hy_mhc_leave")?,
        mhc_route: func(module, "hy_mhc_route")?,
        mhc_logits: func(module, "hy_mhc_logits")?,
        mhc_merge: func(module, "hy_mhc_merge")?,
        expand_q4g: func(module, "hy_expand_q4gr")?,
        expand_q2: func(module, "hy_expand_q2r")?,
        prism_p: func(module, "hy_prism_p")?,
        prism_h: func(module, "hy_prism_h")?,
        qk: func(module, "hy_qk")?,
        scores: func(module, "hy_scores")?,
        values: func(module, "hy_values")?,
        finish: func(module, "hy_finish")?,
        attn_out: func(module, "hy_attn_out")?,
        softmax_h: func(module, "hy_softmax_h")?,
        far_insert: func(module, "hy_far_insert")?,
        far_prep: func(module, "hy_far_prep")?,
        far_prep_b: func(module, "hy_far_prep_b")?,
        expand_pr: func(module, "hy_expand_pr")?,
        far_rows: func(module, "hy_far_rows")?,
        softmax_hf: func(module, "hy_softmax_hf")?,
        attn_out_f: func(module, "hy_attn_out_f")?,
        far_insert_n: func(module, "hy_far_insert_n")?,
        moe_router: func(module, "hy_moe_router")?,
        moe_select: func(module, "hy_moe_select")?,
        moe_combine: func(module, "hy_moe_combine")?,
        norm_add: func(module, "hy_norm_add")?,
        scale: func(module, "hy_scale")?,
        sconv: func(module, "hy_sconv")?,
        head_gate: func(module, "hy_head_gate")?,
    })
}

/// (matrix, output, residual or 0, output row stride, rows to compute)
type JobSpec<'a> = (&'a DevMat, u64, u64, usize, usize);

/// The codecs with a launch of their own (`hy_mmpk_*`, KD in hyb.cu):
/// q4tp, int8 tiles, q1, ternary, q2tp (both forms), q4_tiled.
const KDS: [i32; 6] = [6, 5, 3, 2, 1, 10];

/// The `KDS` index of a tile kind (q2tp's affine 9 rides 1, the two-field
/// int8 pair 15 rides 5).
fn kd_of(kind: i32) -> Option<usize> {
    let k = match kind {
        9 => 1,
        15 => 5,
        k => k,
    };
    KDS.iter().position(|&d| d == k)
}

/// One launch's q4tp matrices for the tensor-core kernel: job tables for one
/// and two tiles a block, and the split of the groups (`KS` warps) the
/// arithmetic of every output depends on.
struct MmaSet {
    bufs: [Buf; 2],
    blocks: [u32; 2],
    n: i32,
    /// 16: a matrix of under 16 384 rows (more warps a tile); 8 otherwise.
    ks16: bool,
    /// One tile a block when the launch's matrices are short (K/V).
    tb1: bool,
    /// One tile a block for a one-row call over 12 288 rows or more: the
    /// tail of a launch costs less than half the tiles' parallelism there
    /// (3090: gate/up 801 against 745 GB/s, GDN input 746 / 679, head
    /// 863 / 839; 5120-row matrices keep two). The arithmetic does not
    /// depend on it.
    tb1_n1: bool,
    /// The `KDS` index of the one codec of every job, if they share one.
    kd: Option<usize>,
    /// int8 jobs under a column field: the launch's rows go through the
    /// fields once a call (`hy_field_planes`) instead of once a tile.
    pre: Option<Prescale>,
}

/// A launch's rows under its jobs' column fields: each job's region of
/// `fields` (0: none, the rows as they are) x its columns, side by side in
/// rows `width` long, up to `NMAX` rows, two planes.
struct Prescale {
    p0: Buf,
    p1: Buf,
    width: usize,
    /// (field(s) or 0, fields, columns, region at)
    ents: Vec<(u64, usize, usize, usize)>,
}

fn mmaset(list: &[JobSpec]) -> Res<MmaSet> {
    mmaset_kind(list, 6)
}

/// One gate/up pair (see `hmat_pair`) whose output is silu(gate) * up,
/// written as the two bf16 planes `p0`, `p1` (`pstride` a row).
fn mmaset_pair(m: &DevMat, p0: u64, p1: u64, pstride: usize) -> Res<MmaSet> {
    mmaset_kind(&[(m, p0, p1, pstride, m.rows)], 6 | 0x100)
}

fn mmaset_kind(list: &[JobSpec], kind: i32) -> Res<MmaSet> {
    let t = std::time::Instant::now();
    let r = mmaset_kind_(list, kind);
    SETS_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

fn mmaset_kind_(list: &[JobSpec], kind: i32) -> Res<MmaSet> {
    if list.iter().any(|(m, ..)| !matches!(m.kind, 1 | 2 | 3 | 5 | 6 | 9 | 10 | 15) || m.cols % 32 != 0) {
        return Err("a tensor-core job over a matrix that is not repacked (q4tp, q2tp, int8, q1 or ternary tiles)".into());
    }
    let max_rows = list.iter().map(|&(_, _, _, _, r)| r).max().unwrap_or(0);
    // a q8_2f job (one field; a pair's two) puts the whole launch on
    // prescaled rows: A5000, one row, 34816 x 5120 under one field 312 us,
    // none 263, a pair's two 404
    let fields_of = |m: &DevMat| if matches!(m.kind, 5 | 15) && m.aux0.is_some() { if m.kind == 15 { 2 } else { 1 } } else { 0 };
    let pre = if list.iter().any(|(m, ..)| fields_of(m) > 0) {
        let mut at = 0usize;
        let ents: Vec<_> = list
            .iter()
            .map(|(m, ..)| {
                let e = (m.aux0(), fields_of(m), m.cols, at);
                at += fields_of(m).max(1) * m.cols;
                e
            })
            .collect();
        Some(Prescale { p0: Buf::new(NMAX * at * 2 + 64)?, p1: Buf::new(NMAX * at * 2 + 64)?, width: at, ents })
    } else {
        None
    };
    let mut bufs = Vec::with_capacity(2);
    let mut blocks = [0u32; 2];
    for tb in 1..=2usize {
        let mut jobs = Vec::with_capacity(list.len());
        let mut blk = 0u32;
        for (i, &(m, y, res, ystride, rows)) in list.iter().enumerate() {
            jobs.push(HyJob {
                w: m.w.ptr,
                aux0: m.aux0(),
                aux1: m.aux1(),
                y,
                res,
                rows: rows as i32,
                // the codec from the matrix (q4tp 6, int8 tiles 5 / 15 two
                // fields, q1 tiles 3, ternary 2, q2tp 1 / 9 affine), the
                // epilogue bits from the call, 0x800 prescaled rows
                kind: (m.kind as i32) | (kind & !0xFF) | if pre.is_some() { 0x800 } else { 0 },
                blk0: blk as i32,
                ystride: ystride as i32,
                xoff: pre.as_ref().map_or(0, |p| p.ents[i].3 as i32),
                nr: 0,
            });
            blk += rows.div_ceil(16).div_ceil(tb) as u32;
        }
        blocks[tb - 1] = blk;
        bufs.push(Buf::from_slice(&jobs)?);
    }
    let b1 = bufs.pop().unwrap();
    let b0 = bufs.pop().unwrap();
    let total_rows: usize = list.iter().map(|&(_, _, _, _, r)| r).sum();
    let kd = list
        .first()
        .and_then(|(m, ..)| kd_of(m.kind as i32))
        .filter(|&d| list.iter().all(|(m, ..)| kd_of(m.kind as i32) == Some(d)));
    // KS 16 under 4096 rows: one codec's decode row on an A5000 (q1 / q4tp,
    // us at KS 8 against 16): 5120 x 17408 49.8 / 52.7, 78.5 / 86.7;
    // 10240 x 5120 25.5 / 27.8, 48.9 / 53.2; 2048 x 2048 4.9 / 4.3, 5.9 / 5.4
    // (the all-codec loop's 16 384 came from a 3090)
    Ok(MmaSet { bufs: [b0, b1], blocks, n: list.len() as i32, ks16: max_rows < 4096, tb1: max_rows <= 2048, tb1_n1: total_rows >= 12288, kd, pre })
}

/// The row planes a tensor-core launch reads: two bf16 parts of each f32
/// (`put3` in hyb.cu), `stride` elements a row.
#[derive(Clone, Copy)]
struct Planes {
    p0: u64,
    p1: u64,
    p2: u64,
    stride: usize,
}

impl MmaSet {
    /// The window's rows of x (as split planes) through every matrix.
    unsafe fn run(&self, c: &Ctx, fx: &HFuncs, p: &Planes, cols: usize, n: usize) -> Res<()> {
        // one codec's decode row takes one tile a block at any height (A5000,
        // q4tp 5120 x 17408: 76.7 against 80.5 us; 10240 x 5120 49.3 / 56.3)
        let tb = if self.tb1 || (n == 1 && (self.tb1_n1 || self.kd.is_some())) { 1 } else { 2 };
        let ks = if self.ks16 { 16 } else { 8 };
        let f = match self.kd {
            Some(d) => fx.mmpk[nt_of(n)][tb - 1][self.ks16 as usize][d],
            None => fx.mmp[nt_of(n)][tb - 1][self.ks16 as usize],
        };
        // the rows under the fields first, every job's region
        let (x0, x1, xs) = match &self.pre {
            Some(pre) => {
                if n > NMAX {
                    return Err(format!("a prescaled tile launch of {n} rows"));
                }
                let (ps_i, n_i, os_i) = (p.stride as i32, n as i32, pre.width as i32);
                for &(cf, fields, cols_j, at) in &pre.ents {
                    let (f_i, c_i, at_i) = (fields as i32, cols_j as i32, at as i32);
                    let tot = (n * fields.max(1) * cols_j) as u64;
                    unsafe {
                        c.launch(fx.field_planes, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(p.p0, p.p1, ps_i, cf, f_i, c_i, n_i, pre.p0.ptr, pre.p1.ptr, os_i, at_i))?;
                    }
                }
                (pre.p0.ptr, pre.p1.ptr, pre.width)
            }
            None => (p.p0, p.p1, p.stride),
        };
        let (nj, xs_i, n_i, cols_i) = (self.n, xs as i32, n as i32, cols as i32);
        unsafe { c.launch(f, (self.blocks[tb - 1], 1), (32 * ks) as u32, 0, kargs!(self.bufs[tb - 1].ptr, nj, x0, x1, x1, xs_i, n_i, cols_i)) }
    }
}

/// The tile launch of a window of `n` rows (at most 16): n tiles of 4 rows.
fn nt_of(n: usize) -> usize {
    debug_assert!(n <= 16, "a tile window of {n} rows");
    if n <= 4 { 0 } else if n <= 8 { 1 } else { 2 }
}

/// x [n][xstride] f32 into the planes, the first `cols` columns (for an
/// input no kernel of this arm writes as planes itself).
unsafe fn split(c: &Ctx, fx: &HFuncs, p: &Planes, x: u64, xstride: usize, cols: usize, n: usize) -> Res<()> {
    let (xs_i, cols_i, os_i, n_i) = (xstride as i32, cols as i32, p.stride as i32, n as i32);
    let tot = (n * cols) as u64;
    unsafe { c.launch(fx.split, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, xs_i, cols_i, p.p0, p.p1, p.p2, os_i, n_i)) }
}

/// A q4tp tensor in the tensor-core layout (see `hyb.cu`), on the host: the
/// reference `hy_repack` is tested against. The nibbles as
/// [tile][group][lane] pairs of words, the codes [tile][group][16 rows], each
/// row's ladder as its two numbers (lo, step). Rows are padded to a multiple
/// of 16 with zero weights.
#[cfg(test)]
fn repack_q4tp(bytes: &[u8], rows: usize, cols: usize) -> Res<(Vec<u32>, Vec<u8>, Vec<f32>)> {
    repack_rows(&[(bytes, rows)], rows, cols, &|r| (0, r))
}

/// `repack_q4tp` over rows drawn from several q4tp payloads of one width:
/// logical row r is row `map(r).1` of payload `srcs[map(r).0]` (a payload
/// and its row count).
#[cfg(test)]
fn repack_rows(srcs: &[(&[u8], usize)], rows: usize, cols: usize, map: &(dyn Fn(usize) -> (usize, usize) + Sync)) -> Res<(Vec<u32>, Vec<u8>, Vec<f32>)> {
    use cortiq_core::quant::{GROUP_SIZE, f16_to_f32, q4tp_code, q4tp_sections};
    if cols % GROUP_SIZE != 0 {
        return Err("q4tp width not a multiple of 32".into());
    }
    let gpr = cols / GROUP_SIZE;
    // per payload: (nibble plane, params, codes, code stride)
    let mut planes = Vec::with_capacity(srcs.len());
    for &(bytes, srows) in srcs {
        let (params_off, codes_off, stride) = q4tp_sections(srows, cols);
        if bytes.len() < codes_off + srows * stride {
            return Err("q4tp payload shorter than its planes".into());
        }
        planes.push((&bytes[..params_off], &bytes[params_off..params_off + srows * 4], &bytes[codes_off..], stride, srows));
    }
    for r in 0..rows {
        let (i, sr) = map(r);
        if i >= planes.len() || sr >= planes[i].4 {
            return Err("repack: a row past its payload".into());
        }
    }
    let ntile = rows.div_ceil(16);
    let mut nib = vec![0u32; ntile * gpr * 64];
    let mut codes = vec![0u8; ntile * gpr * 16];
    // a row's ladder is exp2(lo + code * step) (q4tp_ladder's rungs, up to
    // rounding): the kernels take its two numbers
    let mut lad = vec![0f32; rows * 2];
    for r in 0..rows {
        let (i, sr) = map(r);
        let p = planes[i].1;
        lad[r * 2] = f16_to_f32(u16::from_le_bytes([p[sr * 4], p[sr * 4 + 1]]));
        lad[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[sr * 4 + 2], p[sr * 4 + 3]]));
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 32);
    let per = ntile.div_ceil(threads).max(1);
    let planes = &planes;
    std::thread::scope(|sc| {
        for (ci, (nch, cch)) in nib.chunks_mut(per * gpr * 64).zip(codes.chunks_mut(per * gpr * 16)).enumerate() {
            sc.spawn(move || {
                let t0 = ci * per;
                // the tile's 16 rows: their nibble planes and code rows
                let mut rsrc: [Option<(&[u8], &[u8])>; 16] = [None; 16];
                for (ti, (tn, tc)) in nch.chunks_mut(gpr * 64).zip(cch.chunks_mut(gpr * 16)).enumerate() {
                    let t = t0 + ti;
                    for (i, slot) in rsrc.iter_mut().enumerate() {
                        let r = t * 16 + i;
                        *slot = (r < rows).then(|| {
                            let (pi, sr) = map(r);
                            let (nbp, _, cp, stride, _) = planes[pi];
                            (&nbp[sr * gpr * 16..(sr + 1) * gpr * 16], &cp[sr * stride..(sr + 1) * stride])
                        });
                    }
                    for g in 0..gpr {
                        // the group's four words of each of the 16 rows (a
                        // missing row: nibbles 8, a zero weight)
                        let mut w = [[0x8888_8888u32; 4]; 16];
                        for (i, row) in rsrc.iter().enumerate() {
                            if let Some((nb, _)) = row {
                                let b = &nb[g * 16..g * 16 + 16];
                                for q in 0..4 {
                                    w[i][q] = u32::from_le_bytes([b[q * 4], b[q * 4 + 1], b[q * 4 + 2], b[q * 4 + 3]]);
                                }
                            }
                        }
                        for l in 0..32 {
                            let (gid, tig) = (l >> 2, l & 3);
                            for s in 0..2 {
                                let mut x = 0u32;
                                for m in 0..4 {
                                    // a_m: row gid (+8 for odd m), k = 16s + 2tig (+8 for m >= 2):
                                    // word 2s + (m >> 1) of the row, its byte tig
                                    let byte = (w[gid + (m & 1) * 8][2 * s + (m >> 1)] >> (8 * tig)) & 0xFF;
                                    x |= (byte & 0xF) << (4 * m);
                                    x |= (byte >> 4) << (4 * m + 16);
                                }
                                tn[(g * 32 + l) * 2 + s] = x;
                            }
                        }
                        for (i, row) in rsrc.iter().enumerate() {
                            tc[g * 16 + i] = row.map_or(0, |(_, cr)| q4tp_code(cr, g) as u8);
                        }
                    }
                }
            });
        }
    });
    Ok((nib, codes, lad))
}

/// `count` GEMMs of one shape: A and B f16 (A transposed when `ta`), C f32,
/// f32 accumulation; operand i sits `stride_*` elements after i - 1.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_batched_h(
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
    use super::bl;
    let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
    let beta = 0.0f32;
    let st = unsafe {
        bl::cublasGemmStridedBatchedEx(
            c.blas,
            op(ta),
            op(false),
            m as i32,
            n as i32,
            k as i32,
            std::ptr::from_ref(&alpha).cast(),
            a as *const c_void,
            bl::cudaDataType::CUDA_R_16F,
            lda as i32,
            stride_a as i64,
            b as *const c_void,
            bl::cudaDataType::CUDA_R_16F,
            ldb as i32,
            stride_b as i64,
            std::ptr::from_ref(&beta).cast(),
            cptr as *mut c_void,
            bl::cudaDataType::CUDA_R_32F,
            ldc as i32,
            stride_c as i64,
            count as i32,
            bl::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if st as u32 != 0 {
        return Err(format!("cuBLAS f16 batched gemm {count} x {m}x{n}x{k}: {st:?}"));
    }
    Ok(())
}

/// A q4tp matrix (one payload, or the gate/up pair of two) uploaded as it
/// is stored and repacked on the card (`hy_repack`): (nibbles, codes, row
/// params) of `rows` logical rows.
fn repack_on_card(c: &Ctx, fx: &HFuncs, srcs: &[&[u8]], src_rows: usize, rows: usize, cols: usize) -> Res<(Buf, Buf, Buf)> {
    use cortiq_core::quant::{GROUP_SIZE, f16_to_f32, q4tp_sections};
    if cols % GROUP_SIZE != 0 || srcs.is_empty() || srcs.len() > 2 {
        return Err("repack: a q4tp payload (or a pair) 32 columns wide".into());
    }
    let pair = srcs.len() == 2;
    let gpr = cols / GROUP_SIZE;
    let (params_off, codes_off, stride) = q4tp_sections(src_rows, cols);
    for b in srcs {
        if b.len() < codes_off + src_rows * stride {
            return Err("q4tp payload shorter than its planes".into());
        }
    }
    let map = |r: usize| -> (usize, usize) { if pair { (((r & 15) >= 8) as usize, (r >> 4) * 8 + (r & 7)) } else { (0, r) } };
    let mut lad = vec![0f32; rows * 2];
    for r in 0..rows {
        let (i, sr) = map(r);
        let p = &srcs[i][params_off + sr * 4..params_off + sr * 4 + 4];
        lad[r * 2] = f16_to_f32(u16::from_le_bytes([p[0], p[1]]));
        lad[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[2], p[3]]));
    }
    // the stored planes on the card for the kernel's reads
    let up = |b: &[u8]| -> Res<(Buf, Buf)> {
        Ok((Buf::from_slice(&b[..params_off])?, Buf::from_slice(&b[codes_off..codes_off + src_rows * stride])?))
    };
    let (na, ca) = up(srcs[0])?;
    let second = if pair { Some(up(srcs[1])?) } else { None };
    let (nb_p, cb_p) = second.as_ref().map_or((na.ptr, ca.ptr), |(n, c)| (n.ptr, c.ptr));
    let ntile = rows.div_ceil(16);
    let (t, cc, ld) = mat_planes(ntile * gpr * 64 * 4, ntile * gpr * 16, &lad)?;
    let (stride_i, pair_i, rows_i, cols_i) = (stride as i32, pair as i32, rows as i32, cols as i32);
    let threads = (ntile * gpr * 32) as u64;
    unsafe {
        c.launch(fx.repack, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(na.ptr, ca.ptr, nb_p, cb_p, stride_i, pair_i, t.ptr, cc.ptr, rows_i, cols_i))?;
    }
    // the stored planes are freed on return: the repack must be done
    c.sync()?;
    Ok((t, cc, ld))
}

thread_local! {
    /// A plan build's weight arena (`build`): base, capacity, used. Matrices
    /// come out of it while it has room, so the card's rounding of each
    /// allocation (to 2 MiB) costs once instead of once a matrix.
    static ARENA: std::cell::Cell<(u64, usize, usize)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// The bytes `mat_planes` takes for a rows x cols q4tp matrix.
fn planes_bytes(rows: usize, cols: usize) -> usize {
    let (ntile, gpr) = (rows.div_ceil(16), cols / 32);
    let cc_at = (ntile * gpr * 256).next_multiple_of(256);
    let lad_at = (cc_at + ntile * gpr * 16).next_multiple_of(256);
    (lad_at + rows * 8 + 64).next_multiple_of(256)
}

fn arena_view(bytes: usize) -> Option<Buf> {
    ARENA.with(|a| {
        let (p, cap, used) = a.get();
        if p == 0 || used + bytes > cap {
            return None;
        }
        a.set((p, cap, used + bytes.next_multiple_of(256)));
        Some(Buf { ptr: p + used as u64, bytes, owned: false, slab: None })
    })
}

/// A repacked matrix's three planes (tiles, rung codes, ladders) in one
/// allocation (the plan's arena when it has room): the tiles hold it, the
/// other two are views into it.
fn mat_planes(t_bytes: usize, cc_bytes: usize, lad: &[f32]) -> Res<(Buf, Buf, Buf)> {
    let cc_at = t_bytes.next_multiple_of(256);
    let lad_at = (cc_at + cc_bytes).next_multiple_of(256);
    let lad_bytes = std::mem::size_of_val(lad) + 64;
    let all = match arena_view(lad_at + lad_bytes) {
        Some(v) => v,
        None => Buf::new(lad_at + lad_bytes)?,
    };
    let (cc, ld) = (all.view(cc_at, cc_bytes), all.view(lad_at, lad_bytes));
    unsafe { super::htod(ld.ptr, lad) }?;
    Ok((all, cc, ld))
}

/// `repack_on_card` for the leading `rows` rows of one q4tp payload, its
/// three sections given apart (nibbles, ladders, codes of those rows), so
/// they go to the card straight from the file's pages.
fn repack_rows_on_card(c: &Ctx, fx: &HFuncs, nib: &[u8], params: &[u8], codes: &[u8], stride: usize, rows: usize, cols: usize) -> Res<(Buf, Buf, Buf)> {
    use cortiq_core::quant::{GROUP_SIZE, f16_to_f32};
    let gpr = cols / GROUP_SIZE;
    if cols % GROUP_SIZE != 0 || nib.len() < rows * gpr * 16 || params.len() < rows * 4 || codes.len() < rows * stride {
        return Err("repack: q4tp sections shorter than their rows".into());
    }
    let mut lad = vec![0f32; rows * 2];
    for r in 0..rows {
        let p = &params[r * 4..r * 4 + 4];
        lad[r * 2] = f16_to_f32(u16::from_le_bytes([p[0], p[1]]));
        lad[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[2], p[3]]));
    }
    let (na, ca) = (Buf::from_slice(&nib[..rows * gpr * 16])?, Buf::from_slice(&codes[..rows * stride])?);
    let ntile = rows.div_ceil(16);
    let (t, cc, ld) = mat_planes(ntile * gpr * 64 * 4, ntile * gpr * 16, &lad)?;
    let (stride_i, pair_i, rows_i, cols_i) = (stride as i32, 0i32, rows as i32, cols as i32);
    let threads = (ntile * gpr * 32) as u64;
    unsafe {
        c.launch(fx.repack, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(na.ptr, ca.ptr, na.ptr, ca.ptr, stride_i, pair_i, t.ptr, cc.ptr, rows_i, cols_i))?;
    }
    c.sync()?;
    Ok((t, cc, ld))
}

/// A q8_2f / q8_row payload's planes: (int8 body, f16 row scales, f16
/// column field — none for q8_row).
fn q8_parts(dtype: cortiq_core::TensorDtype, bytes: &[u8], rows: usize, cols: usize) -> Res<(&[u8], &[u8], Option<&[u8]>)> {
    let body = rows * cols;
    let field = dtype == cortiq_core::TensorDtype::Q8_2f;
    if bytes.len() < body + rows * 2 + if field { cols * 2 } else { 0 } {
        return Err("q8 payload shorter than its planes".into());
    }
    Ok((&bytes[..body], &bytes[body..body + rows * 2], field.then(|| &bytes[body + rows * 2..body + rows * 2 + cols * 2])))
}

/// Where an int8-tile matrix's three parts sit in its one allocation:
/// (row scales at, column field at, bytes in all); `fields` column fields
/// (two: a q8_2f gate/up pair's, gate then up).
fn i8_layout(rows: usize, cols: usize, fields: usize) -> (usize, usize, usize) {
    let rs_at = (rows.div_ceil(16) * (cols / 32) * 512).next_multiple_of(256);
    let cf_at = (rs_at + rows * 4).next_multiple_of(256);
    (rs_at, cf_at, (cf_at + fields * cols * 4 + 64).next_multiple_of(256))
}

/// int8 matrices (one, or a gate/up pair laid out as `hy_repack` lays a
/// q4tp pair) as int8 tiles of the tensor-core kernel (kind 5,
/// `hy_repack8`): the bodies go to the card and are tiled there, the f16
/// row scales (pair rows mapped) and the column field as f32 beside them.
/// A q8_2f pair keeps both fields, gate then up (kind 15: a tile's up rows
/// read the input under the up field). (tiles, row scales, column fields
/// — none for q8_row) in one allocation.
fn repack8_on_card(c: &Ctx, fx: &HFuncs, srcs: &[(&[u8], &[u8], Option<&[u8]>)], rows: usize, cols: usize) -> Res<(Buf, Buf, Option<Buf>)> {
    use cortiq_core::quant::f16_to_f32;
    if cols % 32 != 0 || srcs.is_empty() || srcs.len() > 2 {
        return Err("repack: an int8 matrix (or a pair) 32 columns wide".into());
    }
    let pair = srcs.len() == 2;
    if pair && srcs[0].2.is_some() != srcs[1].2.is_some() {
        return Err("a gate/up pair of a q8_2f and a q8_row matrix".into());
    }
    let f16 = |b: &[u8], i: usize| f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
    let rs: Vec<f32> = (0..rows)
        .map(|r| if pair { f16(srcs[((r & 15) >= 8) as usize].1, (r >> 4) * 8 + (r & 7)) } else { f16(srcs[0].1, r) })
        .collect();
    let cf: Option<Vec<f32>> = srcs[0].2.map(|_| srcs.iter().flat_map(|s| (0..cols).map(|k| f16(s.2.unwrap(), k))).collect());
    let fields = cf.as_ref().map_or(1, |v| v.len() / cols);
    let (rs_at, cf_at, bytes) = i8_layout(rows, cols, fields);
    let all = match arena_view(bytes) {
        Some(v) => v,
        None => Buf::new(bytes)?,
    };
    let rsb = all.view(rs_at, rows * 4);
    unsafe { super::htod(rsb.ptr, &rs) }?;
    let cfb = match &cf {
        Some(v) => {
            let b = all.view(cf_at, v.len() * 4);
            unsafe { super::htod(b.ptr, v) }?;
            Some(b)
        }
        None => None,
    };
    let ba = Buf::from_slice(srcs[0].0)?;
    let bb = if pair { Some(Buf::from_slice(srcs[1].0)?) } else { None };
    let bb_p = bb.as_ref().map_or(ba.ptr, |b| b.ptr);
    let (pair_i, rows_i, cols_i) = (pair as i32, rows as i32, cols as i32);
    let threads = (rows.div_ceil(16) * (cols / 32) * 32) as u64;
    unsafe {
        c.launch(fx.repack8, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(ba.ptr, bb_p, pair_i, all.ptr, rows_i, cols_i))?;
    }
    // the int8 bodies are freed on return
    c.sync()?;
    Ok((all, rsb, cfb))
}

/// Where a q1-tile matrix's scales sit in its one allocation: (scales at,
/// bytes in all). `ternary`: two-bit codes, 4 bytes a lane instead of 2.
fn q1_layout_of(rows: usize, cols: usize, ternary: bool) -> (usize, usize) {
    low_layout(rows, cols, if ternary { 2 } else { 3 })
}

/// A per-group-scale tile kind's sizes: (bytes a lane a group, bytes a
/// source group tile) — q1 (3), q1t (2), q4_tiled (10).
fn low_sizes(kind: u8) -> (usize, usize) {
    use cortiq_core::quant::{Q1_TILE, Q1T_TILE, Q4_TILE};
    match kind {
        2 => (4, Q1T_TILE),
        10 => (8, Q4_TILE),
        _ => (2, Q1_TILE),
    }
}

/// (scales at, bytes in all) of a per-group-scale tile matrix of `kind`.
fn low_layout(rows: usize, cols: usize, kind: u8) -> (usize, usize) {
    let tg = rows.div_ceil(16) * (cols / 32);
    let s_at = (tg * 32 * low_sizes(kind).0).next_multiple_of(256);
    (s_at, (s_at + tg * 32 + 64).next_multiple_of(256))
}

fn q1_layout(rows: usize, cols: usize) -> (usize, usize) {
    q1_layout_of(rows, cols, false)
}

/// A q1t payload's ternary tiles, `None` when it carries outlier rows (the
/// overlay after the tiles: `[u32 row_ptr[rows + 1]]` then the entries).
fn q1t_base(bytes: &[u8], rows: usize, cols: usize) -> Option<&[u8]> {
    let base = rows * (cols / 32) * cortiq_core::quant::Q1T_TILE;
    if bytes.len() < base {
        return None;
    }
    let rest = &bytes[base..];
    let clean = rest.len() <= (rows + 1) * 4 && rest.chunks_exact(4).all(|w| w == [0, 0, 0, 0]);
    clean.then_some(&bytes[..base])
}

/// q1 matrices (one, or a gate/up pair laid out as `hy_repack` lays a q4tp
/// pair) as q1 tiles of the tensor-core kernel (kind 3, `hy_repack1`): the
/// payloads go to the card and are tiled there. (tiles, scales) in one
/// allocation.
fn repack1_on_card(c: &Ctx, fx: &HFuncs, srcs: &[&[u8]], src_rows: usize, rows: usize, cols: usize) -> Res<(Buf, Buf)> {
    repack1x_on_card(c, fx, srcs, src_rows, rows, cols, 3)
}

/// `repack1_on_card` for a per-group-scale tile `kind`: q1 (3), q1t (2,
/// `hy_repack1t`; the payloads without their empty overlay) or q4_tiled (10,
/// `hy_repack4t`).
fn repack1x_on_card(c: &Ctx, fx: &HFuncs, srcs: &[&[u8]], src_rows: usize, rows: usize, cols: usize, kind: u8) -> Res<(Buf, Buf)> {
    if cols % 32 != 0 || srcs.is_empty() || srcs.len() > 2 {
        return Err("repack: a q1 matrix (or a pair) 32 columns wide".into());
    }
    let need = src_rows * (cols / 32) * low_sizes(kind).1;
    if srcs.iter().any(|b| b.len() < need) {
        return Err("q1 payload shorter than its tiles".into());
    }
    let (s_at, bytes) = low_layout(rows, cols, kind);
    let all = match arena_view(bytes) {
        Some(v) => v,
        None => Buf::new(bytes)?,
    };
    let tg = rows.div_ceil(16) * (cols / 32);
    let sv = all.view(s_at, tg * 32);
    let a = Buf::from_slice(&srcs[0][..need])?;
    let b = if srcs.len() == 2 { Some(Buf::from_slice(&srcs[1][..need])?) } else { None };
    let b_p = b.as_ref().map_or(a.ptr, |x| x.ptr);
    let (pair_i, rows_i, cols_i) = (b.is_some() as i32, rows as i32, cols as i32);
    let threads = (tg * 32) as u64;
    let f = match kind {
        2 => fx.repack1t,
        10 => fx.repack4t,
        _ => fx.repack1,
    };
    unsafe {
        c.launch(f, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(a.ptr, b_p, pair_i, all.ptr, sv.ptr, rows_i, cols_i))?;
    }
    // the payloads are freed on return
    c.sync()?;
    Ok((all, sv))
}

/// The bytes `repack2_on_card` takes for a rows x cols q2tp matrix.
fn q2_bytes(rows: usize, cols: usize) -> usize {
    let tg = rows.div_ceil(16) * (cols / 32);
    let cc_at = (tg * 128).next_multiple_of(256);
    let lad_at = (cc_at + tg * 16).next_multiple_of(256);
    (lad_at + rows * 8 + 64).next_multiple_of(256)
}

/// A q2tp matrix (one payload, or a gate/up pair laid out as `hy_repack`
/// lays a q4tp pair) as q2tp tiles (`hy_repack2`): (tiles, rung codes, row
/// ladders) as `repack_on_card` gives them for q4tp.
fn repack2_on_card(c: &Ctx, fx: &HFuncs, srcs: &[&[u8]], src_rows: usize, rows: usize, cols: usize) -> Res<(Buf, Buf, Buf)> {
    use cortiq_core::quant::{f16_to_f32, q2tp_sections};
    if cols % 32 != 0 || srcs.is_empty() || srcs.len() > 2 {
        return Err("repack: a q2tp payload (or a pair) 32 columns wide".into());
    }
    let pair = srcs.len() == 2;
    let gpr = cols / 32;
    let (params_off, codes_off, stride) = q2tp_sections(src_rows, cols);
    if srcs.iter().any(|b| b.len() < codes_off + src_rows * stride) {
        return Err("q2tp payload shorter than its planes".into());
    }
    let map = |r: usize| -> (usize, usize) { if pair { (((r & 15) >= 8) as usize, (r >> 4) * 8 + (r & 7)) } else { (0, r) } };
    let mut lad = vec![0f32; rows * 2];
    for r in 0..rows {
        let (i, sr) = map(r);
        let p = &srcs[i][params_off + sr * 4..params_off + sr * 4 + 4];
        lad[r * 2] = f16_to_f32(u16::from_le_bytes([p[0], p[1]]));
        lad[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[2], p[3]]));
    }
    let up = |b: &[u8]| -> Res<(Buf, Buf)> { Ok((Buf::from_slice(&b[..params_off])?, Buf::from_slice(&b[codes_off..codes_off + src_rows * stride])?)) };
    let (ca, ka) = up(srcs[0])?;
    let second = if pair { Some(up(srcs[1])?) } else { None };
    let (cb_p, kb_p) = second.as_ref().map_or((ca.ptr, ka.ptr), |(x, y)| (x.ptr, y.ptr));
    let ntile = rows.div_ceil(16);
    let (t, cc, ld) = mat_planes(ntile * gpr * 128, ntile * gpr * 16, &lad)?;
    let (stride_i, pair_i, rows_i, cols_i) = (stride as i32, pair as i32, rows as i32, cols as i32);
    let threads = (ntile * gpr * 32) as u64;
    unsafe {
        c.launch(fx.repack2, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(ca.ptr, ka.ptr, cb_p, kb_p, stride_i, pair_i, t.ptr, cc.ptr, rows_i, cols_i))?;
    }
    c.sync()?;
    Ok((t, cc, ld))
}

/// Prism's activation boundary for a stack (`prism::forward`): each input
/// width's sign vector on the card, the FWHT block, the f16 cast.
struct PrismPlan {
    blk: usize,
    f16: bool,
    signs: Vec<(usize, Buf)>,
}

/// A stack's Prism plan: none without a Prism header (and then no Prism
/// weight); with one, every matrix an input site feeds must be in the
/// forward basis (the GDN gates a / b in none), the head too, since the
/// transform is made once at the site for all of them.
fn prism_plan(model: &CmfModel, layers: &[GraphLayer], lm: Option<(&GraphW, usize)>) -> Res<Option<PrismPlan>> {
    use crate::gpu::GraphPrismOp as P;
    let fwd = |w: &GraphW| w.prism == P::Forward;
    let none = |w: &GraphW| w.prism == P::None;
    let Some(cfg) = model.header.arch.prism_hadamard.as_ref() else {
        let mut any = lm.is_some_and(|(w, _)| !none(w));
        for l in layers {
            match &l.attn {
                GraphAttn::Gdn { qkv, z, out, .. } => any |= !none(qkv) || !none(z) || !none(out),
                GraphAttn::Full { wq, wk, wv, wo, .. } => any |= !none(wq) || !none(wk) || !none(wv) || !none(wo),
                _ => {}
            }
            if let GraphFfn::Dense { gate, up, down, .. } = &l.ffn {
                any |= !none(gate) || !none(up) || !none(down);
            }
        }
        return if any { Err("a Prism weight without a Prism header".into()) } else { Ok(None) };
    };
    let mixed = || -> Res<Option<PrismPlan>> { Err("a Prism stack whose matrices are not all in the forward basis".into()) };
    for l in layers {
        match &l.attn {
            GraphAttn::Gdn { qkv, z, a, b, out, .. } => {
                if !(fwd(qkv) && fwd(z) && fwd(out) && none(a) && none(b)) {
                    return mixed();
                }
            }
            GraphAttn::Full { wq, wk, wv, wo, head_gate, .. } => {
                if !(fwd(wq) && fwd(wk) && fwd(wv) && fwd(wo)) || head_gate.is_some() {
                    return mixed();
                }
            }
            _ => return Err("a Prism stack with a token mixer the transform is not wired for".into()),
        }
        match &l.ffn {
            GraphFfn::Dense { gate, up, down, .. } if fwd(gate) && fwd(up) && fwd(down) => {}
            _ => return mixed(),
        }
    }
    if lm.is_some_and(|(w, _)| !fwd(w)) {
        return mixed();
    }
    let blk = cfg.block_size;
    if blk == 0 || !blk.is_power_of_two() || blk > 4096 || cfg.widths.iter().any(|w| w % blk != 0) {
        return Err(format!("Prism block {blk} over widths {:?}", cfg.widths));
    }
    let mut signs = Vec::with_capacity(cfg.widths.len());
    let mut off = 0usize;
    for &w in &cfg.widths {
        let s = cfg.signs.get(off..off + w).ok_or("Prism signs shorter than their widths")?;
        signs.push((w, Buf::from_slice(s)?));
        off += w;
    }
    Ok(Some(PrismPlan { blk, f16: cfg.activation_f16, signs }))
}

impl PrismPlan {
    fn signs(&self, width: usize) -> Res<u64> {
        self.signs.iter().find(|(w, _)| *w == width).map(|(_, b)| b.ptr).ok_or_else(|| format!("a Prism input of width {width} without signs"))
    }
}

impl HPlan {
    /// Floats into `tab` (or the prompt path's copy) where table `local`'s
    /// rows of `n` positions start: the call's table first, then each other
    /// one at its own width (`plan_tab`'s order).
    fn tab_off(&self, local: usize, n: usize) -> usize {
        if local == 0 {
            return 0;
        }
        let before: usize = self.invf_local[..local - 1].iter().map(|(f, _)| f.len()).sum();
        n * 2 * (self.half + before)
    }
}

/// The Prism boundary on `n` rows of planes (`p0`, `p1`, `stride` a row) of
/// `width` columns, in place; nothing without a Prism plan.
unsafe fn prism_planes(c: &Ctx, plan: &HPlan, p0: u64, p1: u64, stride: usize, width: usize, n: usize) -> Res<()> {
    let Some(pr) = &plan.prism else { return Ok(()) };
    let s = pr.signs(width)?;
    let (st_i, blk_i, f_i) = (stride as i32, pr.blk as i32, pr.f16 as i32);
    unsafe { c.launch(plan.fx.prism_p, ((width / pr.blk) as u32, n as u32), 256, pr.blk * 4, kargs!(p0, p1, st_i, s, blk_i, f_i)) }
}

/// The Prism boundary on `n` f16 rows `src` (`stride` a row) into `dst`.
unsafe fn prism_rows(c: &Ctx, plan: &HPlan, src: u64, dst: u64, stride: usize, width: usize, n: usize) -> Res<()> {
    let Some(pr) = &plan.prism else { return Ok(()) };
    let s = pr.signs(width)?;
    let (st_i, blk_i, f_i) = (stride as i32, pr.blk as i32, pr.f16 as i32);
    unsafe { c.launch(plan.fx.prism_h, ((width / pr.blk) as u32, n as u32), 256, pr.blk * 4, kargs!(src, dst, st_i, s, blk_i, f_i)) }
}

/// Plan-build time spent repacking and uploading weights (ns).
static REPACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static UPLOAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HMAT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SETS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A graph weight resolved for this arm: q4tp repacked for the tensor-core
/// kernels (cached here, apart from the dense arm's layout), f32 as the
/// dense arm keeps it.
fn hmat(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, gw: &GraphW, rows: usize, cols: usize) -> Res<Arc<DevMat>> {
    let t = std::time::Instant::now();
    let r = hmat_(c, st, model, gw, rows, cols);
    HMAT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

fn hmat_(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, gw: &GraphW, rows: usize, cols: usize) -> Res<Arc<DevMat>> {
    use cortiq_core::TensorDtype;
    if gw.kind == 4 {
        return st.mat(c, model, gw, rows, cols);
    }
    // the codec from the file (MoE experts come with a nominal kind)
    let dtype = model.tensors.get(gw.idx).map(|e| e.dtype);
    let q8 = matches!(dtype, Some(TensorDtype::Q8_2f | TensorDtype::Q8Row));
    let q1 = dtype == Some(TensorDtype::Q1);
    let q1t = dtype == Some(TensorDtype::Q1T);
    let q2 = dtype == Some(TensorDtype::Q2TiledP);
    let q4g = dtype == Some(TensorDtype::Q4Tiled);
    if dtype != Some(TensorDtype::Q4TiledP) && !q8 && !q1 && !q1t && !q2 && !q4g {
        let name = model.tensors.get(gw.idx).map_or("?", |e| e.name.as_str());
        return Err(format!("weight codec {dtype:?} has no hybrid kernel ({name})"));
    }
    // a Prism matrix's input is transformed at its site (`prism_plan` checks
    // the stack); the affine correction is a q2tp codec's
    if gw.prism == crate::gpu::GraphPrismOp::InverseEmbedding || (gw.affine && dtype != Some(TensorDtype::Q2TiledP)) {
        return Err("an inverse-basis or affine weight outside a q2tp matrix".into());
    }
    let key = (model.uid(), gw.idx);
    if let Some(m) = st.hyb.mats.get(&key) {
        if m.rows == rows && m.cols == cols {
            return Ok(m.clone());
        }
    }
    let entry = model.tensors.get(gw.idx).ok_or("tensor index out of range")?;
    if entry.shape.first() != Some(&rows) || entry.shape.get(1) != Some(&cols) {
        return Err(format!("tensor {} is {:?}, the graph wants {rows}x{cols}", entry.name, entry.shape));
    }
    let bytes = model.entry_bytes(entry);
    let budget = super::vram_budget();
    let held = c.resident.load(Ordering::Relaxed);
    if held + bytes.len() as u64 > budget {
        return Err(format!(
            "weights do not fit the card ({:.1} GB held, budget {:.1} GB)",
            held as f64 / 1e9,
            budget as f64 / 1e9
        ));
    }
    let t0 = std::time::Instant::now();
    let fx = st.hyb.funcs.values().next().cloned().ok_or("repack before the hybrid kernels")?;
    let m = if q8 {
        let (w, rs, cf) = repack8_on_card(c, &fx, &[q8_parts(entry.dtype, bytes, rows, cols)?], rows, cols)?;
        DevMat { kind: 5, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: cf, aux1: Some(rs) }
    } else if q1 {
        let (w, s) = repack1_on_card(c, &fx, &[bytes], rows, rows, cols)?;
        DevMat { kind: 3, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else if q4g {
        let (w, s) = repack1x_on_card(c, &fx, &[bytes], rows, rows, cols, 10)?;
        DevMat { kind: 10, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else if q2 {
        let (w, cc, ld) = repack2_on_card(c, &fx, &[bytes], rows, rows, cols)?;
        DevMat { kind: if gw.affine { 9 } else { 1 }, rows, cols, w: Arc::new(w), aux0: Some(cc), aux1: Some(ld), held: bytes.len() as u64 }
    } else if q1t {
        let base = q1t_base(bytes, rows, cols).ok_or_else(|| format!("{}: a q1t matrix with outlier rows", entry.name))?;
        let (w, s) = repack1x_on_card(c, &fx, &[base], rows, rows, cols, 2)?;
        DevMat { kind: 2, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else {
        let (w, aux0, aux1) = repack_on_card(c, &fx, &[bytes], rows, rows, cols)?;
        DevMat { kind: 6, rows, cols, w: Arc::new(w), aux0: Some(aux0), aux1: Some(aux1), held: bytes.len() as u64 }
    };
    REPACK_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    c.resident.fetch_add(m.held, Ordering::Relaxed);
    let m = Arc::new(m);
    st.hyb.mats.insert(key, m.clone());
    Ok(m)
}

/// A dense FFN's gate and up as one repacked matrix of 2 inter rows, its
/// 16-row tiles 8 gate rows then the same 8 up rows: an mma tile holds a
/// row's gate and up in one lane, and the kernel's epilogue makes
/// silu(gate) * up there.
fn hmat_pair(c: &Ctx, st: &mut State, model: &Arc<CmfModel>, gw: &GraphW, uw: &GraphW, inter: usize, cols: usize) -> Res<Arc<DevMat>> {
    use cortiq_core::TensorDtype;
    use cortiq_core::quant::GROUP_SIZE;
    let dt = |w: &GraphW| model.tensors.get(w.idx).map(|e| e.dtype);
    let q8 = |d: Option<TensorDtype>| matches!(d, Some(TensorDtype::Q8_2f | TensorDtype::Q8Row));
    let (dg, du) = (dt(gw), dt(uw));
    let both8 = q8(dg) && q8(du);
    let both1 = dg == Some(TensorDtype::Q1) && du == Some(TensorDtype::Q1);
    let both1t = dg == Some(TensorDtype::Q1T) && du == Some(TensorDtype::Q1T);
    let both2 = dg == Some(TensorDtype::Q2TiledP) && du == Some(TensorDtype::Q2TiledP) && gw.affine == uw.affine;
    let both4g = dg == Some(TensorDtype::Q4Tiled) && du == Some(TensorDtype::Q4Tiled);
    if !(both8 || both1 || both1t || both2 || both4g || (dg == Some(TensorDtype::Q4TiledP) && du == Some(TensorDtype::Q4TiledP))) || inter % 8 != 0 || cols % GROUP_SIZE != 0 {
        return Err(format!("a gate/up pair of one codec (q4tp, int8 or q1) with rows a multiple of 8, not {dg:?} / {du:?}"));
    }
    let key = (model.uid(), gw.idx, uw.idx);
    if let Some(m) = st.hyb.pairs.get(&key) {
        return Ok(m.clone());
    }
    let ge = model.tensors.get(gw.idx).ok_or("tensor index out of range")?;
    let ue = model.tensors.get(uw.idx).ok_or("tensor index out of range")?;
    for e in [ge, ue] {
        if e.shape.first() != Some(&inter) || e.shape.get(1) != Some(&cols) {
            return Err(format!("tensor {} is {:?}, the pair wants {inter}x{cols}", e.name, e.shape));
        }
    }
    let t0 = std::time::Instant::now();
    let (gb, ub) = (model.entry_bytes(ge), model.entry_bytes(ue));
    let held = c.resident.load(Ordering::Relaxed);
    if held + (gb.len() + ub.len()) as u64 > super::vram_budget() {
        return Err("weights do not fit the card".into());
    }
    // the pair's rows straight from the two payloads, on the card
    let rows = 2 * inter;
    let fx = st.hyb.funcs.values().next().cloned().ok_or("repack before the hybrid kernels")?;
    let m = if both8 {
        let (w, rs, cf) = repack8_on_card(c, &fx, &[q8_parts(ge.dtype, gb, inter, cols)?, q8_parts(ue.dtype, ub, inter, cols)?], rows, cols)?;
        // two column fields: the up rows' own (kind 15)
        let kind = if cf.is_some() { 15 } else { 5 };
        DevMat { kind, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: cf, aux1: Some(rs) }
    } else if both1 {
        let (w, s) = repack1_on_card(c, &fx, &[gb, ub], inter, rows, cols)?;
        DevMat { kind: 3, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else if both4g {
        let (w, s) = repack1x_on_card(c, &fx, &[gb, ub], inter, rows, cols, 10)?;
        DevMat { kind: 10, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else if both2 {
        let (w, cc, ld) = repack2_on_card(c, &fx, &[gb, ub], inter, rows, cols)?;
        DevMat { kind: if gw.affine { 9 } else { 1 }, rows, cols, w: Arc::new(w), aux0: Some(cc), aux1: Some(ld), held: (gb.len() + ub.len()) as u64 }
    } else if both1t {
        let (g0, u0) = (q1t_base(gb, inter, cols), q1t_base(ub, inter, cols));
        let (Some(g0), Some(u0)) = (g0, u0) else {
            return Err(format!("{} / {}: a q1t pair with outlier rows", ge.name, ue.name));
        };
        let (w, s) = repack1x_on_card(c, &fx, &[g0, u0], inter, rows, cols, 2)?;
        DevMat { kind: 2, rows, cols, held: w.bytes as u64, w: Arc::new(w), aux0: Some(s), aux1: None }
    } else {
        let (w, aux0, aux1) = repack_on_card(c, &fx, &[gb, ub], inter, rows, cols)?;
        DevMat { kind: 6, rows, cols, w: Arc::new(w), aux0: Some(aux0), aux1: Some(aux1), held: (gb.len() + ub.len()) as u64 }
    };
    REPACK_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    c.resident.fetch_add(m.held, Ordering::Relaxed);
    let m = Arc::new(m);
    st.hyb.pairs.insert(key, m.clone());
    Ok(m)
}

struct GdnL {
    qkv: Arc<DevMat>,
    z: Arc<DevMat>,
    /// Held for the job tables' pointers into them.
    #[allow(dead_code)]
    a: Arc<DevMat>,
    #[allow(dead_code)]
    b: Arc<DevMat>,
    out: Arc<DevMat>,
    /// f16 copies of the f32 a/b projections (the prompt's GEMM operands).
    a_h: Buf,
    b_h: Buf,
    taps: Arc<Buf>,
    a_log: Arc<Buf>,
    dt_bias: Arc<Buf>,
    norm: Arc<Buf>,
    nv: usize,
    nk: usize,
    dk: usize,
    dv: usize,
    kk: usize,
    cdim: usize,
    /// qkv and z (tensor cores); a and b run in the front kernel.
    ms_in: MmaSet,
    ms_out: MmaSet,
}

/// `hy_qk`'s flag word (gemma, bit 1 the late norm) and its bias pointers.
fn qk_flags(a: &AttnL, gemma_i: i32) -> (i32, (u64, u64, u64)) {
    let b = a.bias.as_ref().map_or((0, 0, 0), |(q, k, v)| (q.ptr, k.ptr, v.ptr));
    (gemma_i | ((a.late as i32) << 1) | ((a.v_norm as i32) << 2), b)
}

struct AttnL {
    q: Arc<DevMat>,
    k: Arc<DevMat>,
    v: Arc<DevMat>,
    o: Arc<DevMat>,
    qn: Option<Arc<Buf>>,
    kn: Option<Arc<Buf>>,
    gated: bool,
    /// q/k norm after the rotation (HunYuan)
    late: bool,
    /// Gemma-4's scale-less V norm before the mirror
    v_norm: bool,
    /// (bq, bk, bv) projection biases (Qwen2-class), added before norm and rotation
    bias: Option<(Arc<Buf>, Arc<Buf>, Arc<Buf>)>,
    /// A head-wise output gate: its matrix (the dense arm's, any codec) and a
    /// job table a row, into `hgv`; `hy_head_gate` scales the heads with it.
    hg: Option<(Arc<DevMat>, Vec<Buf>, u32)>,
    /// A sliding window (positions a query sees, its own included).
    window: Option<usize>,
    /// Learned per-Q-head sink logits (in the softmax, no value row).
    sink: Option<Arc<Buf>>,
    /// The rotation table of the layer in `tab`: 0 the call's, k the k-th other.
    local: usize,
    /// The layer's rotary pairs (its table's width).
    half: usize,
    /// The layer's KV heads and head width (the call's, or Gemma-4's global
    /// layers' own), and the attention kernels compiled for them.
    nkv: usize,
    hd: usize,
    fx: Arc<HFuncs>,
    ms_in: MmaSet,
    ms_out: MmaSet,
    /// Unlimited-OCR's DASH router (the attend over its read set).
    dash: Option<DashL>,
}

/// A DASH layer's router (`hy_dash_*`): the q / k index rows, their norms,
/// the block summaries' head, and its constants.
struct DashL {
    w: Arc<Buf>,
    qn: Arc<Buf>,
    kn: Arc<Buf>,
    hc: Arc<Buf>,
    k: usize,
    block: usize,
    local_blocks: usize,
    idx_scale: f32,
    floor: f32,
    sigma: f32,
}

/// Rows a DASH call takes (`gpu::DASH_CHUNK_MAX`), blocks a sequence holds.
const DASH_ROWS: usize = crate::gpu::DASH_CHUNK_MAX;
const DASH_BLOCKS: usize = 4096;

/// A plan's DASH call buffers (the host's candidate lists, row offsets and
/// rotation table of the call's rows, and its record: first position, block
/// count), and the router's scratch; shared by the layers.
struct DashBufs {
    cands: Buf,
    coff: Buf,
    tab: Buf,
    rec: Buf,
    sc: Buf,
    sel: Buf,
    bias: Buf,
}

/// Can the DASH kernels hold this router (index width 128, a head of 128)?
fn dash_admits(d: &crate::gpu::GraphDash<'_>, hidden: usize, hd: usize, rd: usize) -> bool {
    d.q_norm.len() == 128
        && d.k_norm.len() == 128
        && d.head_cls.len() == 128
        && d.qk_proj.len() == 256 * hidden
        && hidden % 4 == 0
        && hd == 128
        && rd == hd
        && d.block > 0
        && d.block <= 64
        && d.k <= 16
        && d.block * d.k + d.block * d.local_blocks <= 512
}

enum Mix {
    Gdn(GdnL),
    Attn(AttnL),
    Conv(ConvL),
}

/// LFM2's gated short convolution: the fused B | C | x projection into the
/// GDN rows (`qkv`), `hy_sconv` over the layer's ring (kept as a GDN state of
/// ring only), the out projection.
struct ConvL {
    inp: Arc<DevMat>,
    out: Arc<DevMat>,
    taps: Arc<Buf>,
    kernel: usize,
    ms_in: MmaSet,
    ms_out: MmaSet,
}

/// A layer's mHC routers (`sensenova::Mhc`): the attention's and the FFN's
/// weights [out][ns·hidden] and biases, whether the layer enters / leaves the
/// streams.
struct MhcL {
    aw: Arc<Buf>,
    ab: Arc<Buf>,
    fw: Arc<Buf>,
    fb: Arc<Buf>,
    out: usize,
    enter: bool,
    leave: bool,
}

/// The stack's mHC streams: their count, Sinkhorn iterations, branch scale,
/// and the call's rows of streams and merge coefficients.
struct MhcPlan {
    ns: usize,
    iters: usize,
    pre_scale: f32,
    r: Buf,
    coef: Buf,
    /// The router's logits [row][24] (`hy_mhc_logits`).
    lg: Buf,
}

struct HLayer {
    mix: Mix,
    /// gate and up as one pair (`hmat_pair`); a routed FFN's shared expert
    gu: Arc<DevMat>,
    down: Arc<DevMat>,
    in_norm: Arc<Buf>,
    post_norm: Arc<Buf>,
    inter: usize,
    /// GeGLU (tanh GELU, the Gemma family) instead of SwiGLU
    gelu: bool,
    ms_gu: MmaSet,
    /// into the stream, or (a routed FFN) the shared expert's output rows
    ms_dn: MmaSet,
    moe: Option<MoeL>,
    /// SenseNova-Loop's mHC routers around both sub-layers.
    mhc: Option<MhcL>,
    /// Gemma's sandwich norms (the mixer's and the FFN's output normed before
    /// their residual adds: the products then write `yo`) and layer scalar.
    attn_out: Option<Arc<Buf>>,
    ffn_out: Option<Arc<Buf>>,
    layer_scale: Option<f32>,
}

/// The vocabulary head: q4tp on the tensor cores, or another codec (a q8_2f
/// head or tied embedding) through the dense arm's `mv_multi`, one launch a
/// row with the final norm folded into its staging (a job table a row).
enum Head {
    Mma(Arc<DevMat>, usize, MmaSet),
    Dense(Arc<DevMat>, usize, Vec<Buf>, u32),
}

impl Head {
    fn rows(&self) -> usize {
        match self {
            Head::Mma(_, r, _) | Head::Dense(_, r, _, _) => *r,
        }
    }
}

/// A routed FFN: the router, every routed expert's gate/up pair and down as
/// job templates (`hy_moe_select` points the picked ones at the launch's
/// slots); the shared expert is the layer's `gu` / `down`.
struct MoeL {
    router: Arc<DevMat>,
    sgw: Option<Arc<DevMat>>,
    bias: Option<Arc<Buf>>,
    gu_t: Buf,
    dn_t: Buf,
    e: usize,
    k: usize,
    inter: usize,
    flags: i32,
    route_scale: f32,
    gated: bool,
    bpj_gu: u32,
    bpj_dn: u32,
    /// The `KDS` index of the experts' one codec (gate/up, down), if they share one.
    kd: (Option<usize>, Option<usize>),
    /// Whether a shared expert rides beside the routed ones (the layer's pair).
    shared: bool,
    _mats: Vec<Arc<DevMat>>,
}

/// The routed FFN's per-call rows (`NMAX` rows, `smax` slots), one set a plan.
struct MoeBufs {
    logits: Buf,
    sg: Buf,
    wsel: Buf,
    gu_a: Buf,
    dn_a: Buf,
    qq: [Buf; 2],
    dns: Buf,
    dsh: Buf,
    smax: usize,
}

/// A hybrid stack resolved for the card.
pub(super) struct HPlan {
    /// Unique among the plans of the process (graph keys hold it).
    id: u64,
    /// The call records of the attention layers (8 words each).
    ci: Buf,
    sig: Vec<usize>,
    consts: Vec<(usize, usize, u64)>,
    fx: Arc<HFuncs>,
    layers: Vec<HLayer>,
    lm: Option<Head>,
    final_norm: Option<Arc<Buf>>,
    invf: Vec<f32>,
    /// The other rotations (other layers' frequencies and YaRN amplitude; their
    /// rows follow the call's in `tab`, a table after the other).
    invf_local: Vec<(Vec<f32>, f32)>,
    /// Prism's activation boundary at every input site (a Prism file).
    prism: Option<PrismPlan>,
    /// The widest head of the stack (a layer of its own geometry).
    hdmax: usize,
    /// SenseNova-Loop's mHC streams (the stack's geometry and rows).
    mhc: Option<MhcPlan>,
    dash: Option<DashBufs>,
    /// A Looped Transformer: the layers after which the final norm is applied
    /// to the stream (`loop_norm_at`), and its weights.
    loop_norm: Vec<usize>,
    loop_w: Option<Arc<Buf>>,
    /// Some layer's head geometry is its own (its kernels a module of their own).
    own_geom: bool,
    half: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    hpk: usize,
    hidden: usize,
    gemma: bool,
    eps: f32,
    scale: f32,
    // the small path's rows (NMAX each), fixed: the job tables hold them
    h: Buf,
    xn: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    qg: Buf,
    gate: Buf,
    #[allow(dead_code)]
    ao: Buf,
    qkv: Buf,
    z: Buf,
    o: Buf,
    of: Buf,
    logits: Buf,
    tab: Buf,
    #[allow(dead_code)]
    ids: Buf,
    /// The bf16 planes every tensor-core launch reads (`NMAX` rows of
    /// `pstride` each).
    pp: [Buf; 3],
    /// The FFN middle's two planes.
    qq: [Buf; 2],
    pstride: usize,
    qrow: usize,
    cdmax: usize,
    vdmax: usize,
    nvmax: usize,
    moe: Option<MoeBufs>,
    yo: Buf,
    hgv: Buf,
}

/// One GDN layer's recurrent state on the card, with what a verify window
/// needs to step back.
struct GdnState {
    dims: (usize, usize, usize, usize, usize),
    /// The allocation the buffers below are views into (one a layer: the card
    /// rounds each allocation up, and five of the seven are small).
    _all: Buf,
    s: Buf,
    ring: Buf,
    next_pos: usize,
    // the window's tape: raw rows (`t_stride` floats apart), conv rows,
    // gates (NMAX rows each)
    t_qkv: Buf,
    t_stride: usize,
    t_cq: Buf,
    t_gb: Buf,
    s_save: Buf,
    ring_save: Buf,
    /// (first position, rows) of the last window that kept a tape.
    tape: Option<(usize, usize)>,
    /// A copy of `[s | ring]` at a prompt position (`gdn_snapshot`): the
    /// next turn of a chat goes back there instead of reading the
    /// conversation again.
    snap: Option<(usize, Buf, Buf)>,
}

#[derive(Default)]
pub(super) struct HState {
    funcs: HashMap<(usize, usize), Arc<HFuncs>>,
    /// q4tp weights in the tensor-core layout.
    mats: HashMap<(u64, usize), Arc<DevMat>>,
    /// Gate/up pairs (`hmat_pair`) by (model, gate, up).
    pairs: HashMap<(u64, usize, usize), Arc<DevMat>>,
    /// The far fields of sealed layers (turbo files), by sequence and layer.
    o1: HashMap<(u64, usize), HO1>,
    /// K/V mirrors this arm keeps in f16 (by sequence and layer): a mirror's
    /// `row` counts f32 slots, half its f16 values.
    pub(super) half: std::collections::HashSet<(u64, usize)>,
    /// `rows_matvec`'s job tables, by tensor, with the output they point at.
    rowsets: HashMap<(u64, usize), (MmaSet, u64)>,
    /// The restore tables of `spec_restore`, by sequence: the layers' state
    /// pointers (7 a layer) and the keys they were made for.
    rtabs: HashMap<u64, (Buf, Vec<(u64, usize)>)>,
    /// Recorded small-path graphs by (plan, sequence, layer base, rows, tape,
    /// head).
    graphs: HashMap<(u64, u64, usize, usize, bool, bool, u64), GraphRec>,
    plans: HashMap<(u64, usize, usize, bool), Arc<HPlan>>,
    gdn: HashMap<(u64, usize), GdnState>,
    /// DASH layers' open blocks (three, the router's keys) and block
    /// summaries, by sequence and layer.
    dash: HashMap<(u64, usize), (Buf, Buf)>,
    /// The MTP draft chain's table and norms (`mtp_chain`).
    chain: Option<ChainRes>,
    /// Plan builds' weight arenas (the matrices in them are views).
    arenas: Vec<Buf>,
    /// The chain's eh_proj job tables by (plan, tensor): their output is
    /// that plan's `h`.
    chain_sets: HashMap<(u64, usize), MmaSet>,
}

/// The draft chain's resources for one model and shortlist: the shortlist
/// rows of the embedding (a draft's token is always one of them), repacked
/// for the tensor-core layout, and the block input's two norms.
struct ChainRes {
    key: (u64, usize, usize, u64, u64),
    emb: DevMat,
    enorm: Buf,
    hnorm: Buf,
}

/// Does this stack belong to the hybrid arm? (A GDN layer, or full attention
/// with Q/K norms or the element-wise output gate.) An empty stack with a
/// head goes here when the dense arm's attention kernels cannot take the
/// geometry.
/// A Clover block preview is active over this stack (its GDN layers' preview
/// operator, which the arm has no kernels for; a cached plan is causal).
fn previewing(layers: &[GraphLayer]) -> bool {
    crate::gpu::block_preview().is_some() && layers.iter().any(|l| matches!(&l.attn, GraphAttn::Gdn { preview: Some(_), .. }))
}

/// Every stack tries the hybrid arm first (the dense arm's plain attention
/// stacks included); `CMF_CUDA_HYB_ALL=0` keeps those on the dense arm.
pub(super) fn all_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_CUDA_HYB_ALL").as_deref() != Ok("0"))
}

pub(super) fn wants(layers: &[GraphLayer], nh: usize, nkv: usize, hd: usize) -> bool {
    if layers.is_empty() {
        let hpk = if nkv > 0 { nh / nkv } else { 0 };
        return 8 * hpk * hd * 4 > 44_000;
    }
    layers.iter().any(|l| match &l.attn {
        GraphAttn::Gdn { .. } => true,
        GraphAttn::Full { q_norm, k_norm, output_gate, .. } => *output_gate || q_norm.is_some() || k_norm.is_some(),
        _ => false,
    })
}

fn rows_of(model: &CmfModel, gw: &GraphW) -> Option<usize> {
    if gw.kind == 4 {
        return None;
    }
    model.tensors.get(gw.idx).and_then(|e| e.shape.first().copied())
}

#[allow(clippy::too_many_arguments)]
fn hsig(
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    gemma: bool,
    invf: &[f32],
) -> Vec<usize> {
    let mut s = Vec::with_capacity(layers.len() * 24 + 16);
    let w = |s: &mut Vec<usize>, gw: &GraphW| s.extend([gw.idx, gw.kind as usize, gw.data.as_ptr() as usize, gw.data.len()]);
    let f = |s: &mut Vec<usize>, d: &[f32]| s.extend([d.as_ptr() as usize, d.len()]);
    s.extend([model.uid() as usize, nh, nkv, hd, rd, hidden, gemma as usize]);
    f(&mut s, invf);
    // by content: a verify passes a fresh copy of the final norm every round
    s.extend([final_norm.len(), crate::gpu::fp_f32(final_norm) as usize]);
    if let Some((gw, rows)) = lm {
        w(&mut s, gw);
        s.push(rows);
    }
    for l in layers {
        f(&mut s, l.input_norm);
        f(&mut s, l.post_norm);
        match &l.attn {
            GraphAttn::Gdn { qkv, z, a, b, out, conv1d, a_log, dt_bias, norm, nv, nk, dk, dv, kk, .. } => {
                s.push(1);
                for g in [qkv, z, a, b, out] {
                    w(&mut s, g);
                }
                for d in [*conv1d, *a_log, *dt_bias, *norm] {
                    f(&mut s, d);
                }
                s.extend([*nv, *nk, *dk, *dv, *kk]);
            }
            GraphAttn::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, late_qk_norm, bias, head_gate, geom, .. } => {
                s.push(2);
                if let Some(g) = geom {
                    s.extend([8, g.window.unwrap_or(0)]);
                    // a DASH router: its weights (by address) and constants; the
                    // call's candidates and angles go up a call at a time
                    if let Some(d) = &g.dash {
                        s.extend([9, d.block, d.k, d.local_blocks, d.idx_scale.to_bits() as usize, d.floor.to_bits() as usize, d.sigma.to_bits() as usize]);
                        for w in [d.qk_proj, d.q_norm, d.k_norm, d.head_cls] {
                            f(&mut s, w);
                        }
                    }
                }
                if let Some(g) = head_gate {
                    s.push(7);
                    w(&mut s, g);
                }
                for g in [wq, wk, wv, wo] {
                    w(&mut s, g);
                }
                f(&mut s, q_norm.unwrap_or(&[]));
                f(&mut s, k_norm.unwrap_or(&[]));
                s.push(*output_gate as usize);
                s.push(*late_qk_norm as usize);
                for d in bias.map_or([&[][..]; 3], |(a, b, c)| [a, b, c]) {
                    f(&mut s, d);
                }
            }
            GraphAttn::ShortConv { inp, out, taps, kernel, .. } => {
                s.push(6);
                w(&mut s, inp);
                w(&mut s, out);
                f(&mut s, taps);
                s.push(*kernel);
            }
            _ => s.push(3),
        }
        match &l.ffn {
            GraphFfn::Dense { gate, up, down, .. } => {
                for g in [gate, up, down] {
                    w(&mut s, g);
                }
            }
            GraphFfn::Moe { router, shared_gate, experts, n_exp, top_k, inter, norm_topk, sigmoid, bias, detached_shared, shared_gated, route_scale, .. } => {
                s.extend([4, *n_exp, *top_k, *inter, *norm_topk as usize, *sigmoid as usize, *shared_gated as usize, route_scale.to_bits() as usize]);
                w(&mut s, router);
                w(&mut s, shared_gate);
                for &(g, u, d) in experts.iter() {
                    s.extend([g, u, d]);
                }
                if let Some(d) = detached_shared {
                    s.extend([d.gate.idx, d.up.idx, d.down.idx, d.width]);
                }
                f(&mut s, bias.unwrap_or(&[]));
            }
            _ => s.push(5),
        }
        f(&mut s, l.attn_out_norm.unwrap_or(&[]));
        f(&mut s, l.ffn_out_norm.unwrap_or(&[]));
        s.push(l.layer_scale.map_or(0, |v| v.to_bits() as usize + 1));
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn build(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    gemma: bool,
    eps: f32,
    scale: f32,
    invf: &[f32],
    sig: Vec<usize>,
    loop_norm: &[usize],
) -> Res<HPlan> {
    if nkv == 0 || nh % nkv != 0 || (layers.is_empty() && lm.is_none()) {
        return Err("empty stack or head geometry".into());
    }
    let prism = prism_plan(model, layers, lm)?;
    if hidden % 32 != 0 {
        return Err("hidden width not a multiple of 32".into());
    }
    let hpk = nh / nkv;
    // a layer-less head pass turns nothing
    let rd = if layers.is_empty() { 0 } else { rd };
    if rd % 2 != 0 || rd > hd || invf.len() < rd / 2 {
        return Err(format!("rotary width {rd}"));
    }
    let half = rd / 2;
    let fx = match st.hyb.funcs.get(&(hpk, hd)) {
        Some(f) => f.clone(),
        None => {
            let t = std::time::Instant::now();
            let f = Arc::new(load_funcs(c, hpk, hd)?);
            tracing::info!("cuda hybrid kernels compiled in {:.1} s", t.elapsed().as_secs_f64());
            st.hyb.funcs.insert((hpk, hd), f.clone());
            f
        }
    };
    let mut consts = Vec::new();
    let mut note = |d: &[f32]| consts.push((d.as_ptr() as usize, d.len(), crate::gpu::fp_f32(d)));
    // geometry first: the row buffers the job tables point into
    let mut qrow = nh * hd;
    // the widest attention a layer of its own geometry (Gemma-4's global
    // layers: one KV head of 512) takes: K/V rows, head width
    let (mut kvmax, mut hdmax) = (nkv * hd, hd);
    let (mut cdmax, mut vdmax, mut nvmax, mut intermax) = (0usize, 0usize, 0usize, 0usize);
    let (mut moe_e, mut moe_k) = (0usize, 0usize);
    let mut invf_local: Vec<(Vec<f32>, f32)> = Vec::new();
    // SenseNova-Loop's mHC: the streams' count (from the router's shape), its
    // Sinkhorn iterations and the branch scale, one for the stack
    let mut mhc_geo: Option<(usize, usize, f32)> = None;
    for (li, l) in layers.iter().enumerate() {
        if let Some(m) = &l.mhc {
            let out = m.attn_b.len();
            let ns = if out > 0 { m.attn_w.len() / (out * hidden).max(1) } else { 0 };
            let ok = (1..=4).contains(&ns)
                && (out == 2 * ns + ns * ns || out == 2 * ns)
                && out <= 24
                && m.attn_w.len() == out * ns * hidden
                && m.ffn_b.len() == out
                && m.ffn_w.len() == out * ns * hidden
                && !matches!(l.ffn, GraphFfn::Moe { .. })
                && l.attn_out_norm.is_none()
                && l.ffn_out_norm.is_none();
            let geo = (ns, m.iters, m.pre_scale);
            if !ok || mhc_geo.is_some_and(|g| g != geo) {
                return Err(format!("layer {li}: mHC streams of a shape the kernels do not take"));
            }
            mhc_geo = Some(geo);
        }
        let sandwich = l.attn_out_norm.is_some() || l.ffn_out_norm.is_some() || l.layer_scale.is_some();
        if sandwich && !matches!(l.ffn, GraphFfn::Dense { .. }) {
            return Err(format!("layer {li}: sandwich norms beside a routed FFN"));
        }
        if l.attn_out_norm.is_some_and(|w| w.len() != hidden) || l.ffn_out_norm.is_some_and(|w| w.len() != hidden) {
            return Err(format!("layer {li}: sandwich norm width"));
        }
        if l.input_norm.len() != hidden || l.post_norm.len() != hidden {
            return Err(format!("layer {li}: norm width"));
        }
        match &l.attn {
            GraphAttn::Gdn { nv, nk, dk, dv, kk, preview, .. } => {
                // Clover carries its preview weights on every call; only an
                // active block preview (`enter_block_preview`) reads them
                if preview.is_some() && crate::gpu::block_preview().is_some() {
                    return Err(format!("layer {li}: block preview"));
                }
                if *dk != 128 || *dv != *dk || *nk == 0 || nv % nk != 0 || *kk < 1 || *kk > 9 {
                    return Err(format!("layer {li}: GDN geometry nv {nv} nk {nk} dk {dk} dv {dv}"));
                }
                cdmax = cdmax.max(2 * nk * dk + nv * dv);
                vdmax = vdmax.max(nv * dv);
                nvmax = nvmax.max(*nv);
            }
            GraphAttn::Full { bias, output_gate, geom, head_gate, .. } => {
                // a geometry equal to the call-wide one (a stack where only some
                // layers differ) is the call-wide one, a sliding window aside
                // a head geometry of the layer's own: its kernels a module of their own
                let own = |g: &crate::gpu::GraphAttnGeom| g.hd % 32 == 0 && g.hd <= 512 && g.nkv > 0 && nh % g.nkv == 0 && nh / g.nkv <= 16;
                let same = geom.as_ref().is_none_or(|g| {
                    ((g.nkv == nkv && g.hd == hd) || own(g))
                        && g.dv == g.hd
                        && g.rd % 2 == 0
                        && g.window.is_none_or(|w| w > 0 && w < i32::MAX as usize)
                        && g.sink.is_none_or(|s| s.len() == nh)
                        && g.dash.as_ref().is_none_or(|d| dash_admits(d, hidden, g.hd, g.rd) && g.window.is_none() && g.sink.is_none())
                        && g.vres.is_none()
                        && g.invf.len() >= g.rd / 2
                        && invf.len() >= rd / 2
                        // a DASH layer rotates by its rows' own angles (the call's table)
                        && (g.dash.is_some() || (g.rd == rd && g.invf[..rd / 2] == invf[..rd / 2] && g.rope_scale == 1.0) || {
                            // up to two more tables besides the call's: other layers'
                            // bases (local ones), a YaRN amplitude baked into the rows,
                            // a rotary width of the layer's own (flint: none on its
                            // global layers, 128 on the sliding ones), each table as
                            // wide as its layer's rotation
                            let key = (g.invf[..g.rd / 2].to_vec(), g.rope_scale);
                            invf_local.contains(&key) || {
                                invf_local.push(key);
                                invf_local.len() <= 2
                            }
                        })
                });
                if let (false, Some(g)) = (same, geom) {
                    return Err(format!(
                        "layer {li}: an attention geometry of its own (nkv {} hd {} dv {} rd {} window {:?} sink {} v-norm {} rope x{} of call nkv {nkv} hd {hd} rd {rd})",
                        g.nkv,
                        g.hd,
                        g.dv,
                        g.rd,
                        g.window,
                        g.sink.is_some(),
                        g.v_norm,
                        g.rope_scale
                    ));
                }
                if head_gate.is_some() && super::mv_shared_bytes(hidden) > c.max_shared {
                    return Err(format!("layer {li}: a head-wise output gate over a row wider than shared memory"));
                }
                if bias.is_some() && *output_gate {
                    return Err(format!("layer {li}: projection biases beside a gated query"));
                }
                let (lnkv, lhd) = geom.as_ref().map_or((nkv, hd), |g| (g.nkv, g.hd));
                qrow = qrow.max(nh * lhd * (1 + *output_gate as usize));
                kvmax = kvmax.max(lnkv * lhd);
                hdmax = hdmax.max(lhd);
            }
            GraphAttn::ShortConv { kernel, taps, cpu_state, .. } => {
                if *kernel == 0 || *kernel > 9 || taps.len() != hidden * kernel || !(cpu_state.is_empty() || cpu_state.len() == (kernel - 1) * hidden) {
                    return Err(format!("layer {li}: a short convolution of kernel {kernel}"));
                }
                cdmax = cdmax.max(3 * hidden);
            }
            _ => return Err(format!("layer {li}: a token mixer the hybrid arm has no kernels for")),
        }
        match &l.ffn {
            GraphFfn::Dense { gate, act, limit, hot, .. } => {
                if !matches!(act, GraphAct::Silu | GraphAct::GeluTanh) || *limit != 0.0 || hot.is_some() {
                    return Err(format!("layer {li}: an FFN other than plain SwiGLU / GeGLU"));
                }
                intermax = intermax.max(rows_of(model, gate).ok_or("f32 FFN")?);
            }
            GraphFfn::Moe { act, q4tp, gu_q2, q82, n_exp, top_k, inter, experts, has_shared, detached_shared, router, .. } => {
                // q2tp gate/up beside q4tp down (`gu_q2`) load by their file codec;
                // q8_2f experts (two column fields a pair) have no tiles
                let _ = (gu_q2, q4tp);
                if *act != GraphAct::Silu || *q82 {
                    return Err(format!("layer {li}: a routed FFN other than q4tp / q2tp SwiGLU"));
                }
                if *n_exp == 0 || *n_exp > 1024 || *top_k == 0 || *top_k > 64 || experts.len() != n_exp + *has_shared as usize || router.kind != 4 {
                    return Err(format!("layer {li}: a routed FFN of {n_exp} experts, top {top_k}"));
                }
                if *has_shared && detached_shared.is_some() {
                    return Err(format!("layer {li}: a routed FFN with two shared experts"));
                }
                let sh = match detached_shared {
                    Some(d) => d.width,
                    None if *has_shared => model.tensors.get(experts[*n_exp].0).and_then(|e| e.shape.first().copied()).ok_or("shared expert shape")?,
                    None => *inter,
                };
                intermax = intermax.max(*inter).max(sh);
                moe_e = moe_e.max(*n_exp);
                moe_k = moe_k.max(*top_k);
            }
            _ => return Err(format!("layer {li}: an FFN other than a dense or a routed one")),
        }
    }
    let lm_rows = lm.map_or(0, |(_, r)| r);
    let f4 = |n: usize| Buf::new(n.max(1) * 4);
    let h = f4(NMAX * hidden)?;
    let xn = f4(NMAX * hidden)?;
    let q = f4(NMAX * qrow)?;
    let k = f4(NMAX * kvmax)?;
    let v = f4(NMAX * kvmax)?;
    let qg = f4(NMAX * nh * hdmax)?;
    let gate = f4(NMAX * nh * hdmax)?;
    let ao = f4(NMAX * nh * hdmax)?;
    let qkv = f4(NMAX * cdmax)?;
    let z = f4(NMAX * vdmax)?;
    let o = f4(NMAX * vdmax)?;
    let of = f4(NMAX * vdmax)?;
    // the call's rotation rows, then (a stack with other rotations) theirs
    // the call's rotation rows, then each other table's at its own width
    let tab = f4((NMAX * 2 * (half + invf_local.iter().map(|(f, _)| f.len()).sum::<usize>())).max(1))?;
    let ids = f4(NMAX)?;
    let pstride = hidden.max(intermax).max(vdmax).max(nh * hdmax).next_multiple_of(32);
    // the FFN middle's planes (the gate/up pair writes them, down reads them)
    let qq = [Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?];
    // a sandwich-normed product's rows before their norm; a head gate's logits
    let yo = f4(NMAX * hidden)?;
    let hgv = f4(NMAX * nh)?;
    // a routed FFN's rows: at most NMAX x top_k distinct experts a call
    let moe_bufs = if moe_e > 0 {
        let smax = (NMAX * moe_k).min(moe_e);
        let job = std::mem::size_of::<HyJob>();
        Some(MoeBufs {
            logits: f4(NMAX * moe_e)?,
            sg: f4(NMAX)?,
            wsel: f4(NMAX * smax)?,
            gu_a: Buf::new(smax * job)?,
            dn_a: Buf::new(smax * job)?,
            qq: [Buf::new(smax * NMAX * pstride * 2)?, Buf::new(smax * NMAX * pstride * 2)?],
            dns: f4(smax * NMAX * hidden)?,
            dsh: f4(NMAX * hidden)?,
            smax,
        })
    } else {
        None
    };
    let mut out = Vec::with_capacity(layers.len());
    mem_report(c, st, "a plan's row buffers");
    // the q4tp matrices this plan uploads, in one arena
    let uid = model.uid();
    let q6 = |w: &GraphW| w.kind == 6 && w.prism == crate::gpu::GraphPrismOp::None && !w.affine;
    let is1 = |w: &GraphW| {
        w.prism == crate::gpu::GraphPrismOp::None
            && !w.affine
            && matches!(model.tensors.get(w.idx).map(|e| e.dtype), Some(cortiq_core::TensorDtype::Q1 | cortiq_core::TensorDtype::Q1T))
    };
    let tern = |idx: usize| model.tensors.get(idx).map(|e| e.dtype) == Some(cortiq_core::TensorDtype::Q1T);
    let is2 = |w: &GraphW| model.tensors.get(w.idx).map(|e| e.dtype) == Some(cortiq_core::TensorDtype::Q2TiledP);
    let is8 = |w: &GraphW| {
        w.prism == crate::gpu::GraphPrismOp::None
            && !w.affine
            && matches!(model.tensors.get(w.idx).map(|e| e.dtype), Some(cortiq_core::TensorDtype::Q8_2f | cortiq_core::TensorDtype::Q8Row))
    };
    let mut need = 0usize;
    for l in layers {
        let ffn: Vec<(usize, usize, usize, usize)> = match &l.ffn {
            GraphFfn::Dense { gate: gw, up: uw, down: dw, .. } if (q6(gw) && q6(uw) && q6(dw)) || (is1(gw) && is1(uw) && is1(dw)) || (is2(gw) && is2(uw) && is2(dw)) => {
                vec![(gw.idx, uw.idx, dw.idx, rows_of(model, gw).unwrap_or(0))]
            }
            GraphFfn::Moe { experts, inter, n_exp, detached_shared, .. } => {
                let mut v: Vec<_> = experts[..*n_exp].iter().map(|&(g, u, d)| (g, u, d, *inter)).collect();
                match detached_shared {
                    Some(d) => v.push((d.gate.idx, d.up.idx, d.down.idx, d.width)),
                    None if experts.len() > *n_exp => {
                        let (g, u, d) = experts[*n_exp];
                        v.push((g, u, d, model.tensors.get(g).and_then(|e| e.shape.first().copied()).unwrap_or(0)));
                    }
                    None => {}
                }
                v
            }
            _ => Vec::new(),
        };
        let sized = |idx: usize, rows: usize, cols: usize| match model.tensors.get(idx).map(|e| e.dtype) {
            Some(cortiq_core::TensorDtype::Q1) => q1_layout(rows, cols).1,
            Some(cortiq_core::TensorDtype::Q1T) => q1_layout_of(rows, cols, true).1,
            Some(cortiq_core::TensorDtype::Q2TiledP) => q2_bytes(rows, cols),
            Some(cortiq_core::TensorDtype::Q4Tiled) => low_layout(rows, cols, 10).1,
            Some(cortiq_core::TensorDtype::Q8_2f | cortiq_core::TensorDtype::Q8Row) => i8_layout(rows, cols, 2).2,
            _ => planes_bytes(rows, cols),
        };
        for (g, u, d, inter) in ffn {
            if !st.hyb.pairs.contains_key(&(uid, g, u)) {
                need += sized(g, 2 * inter, hidden);
            }
            if !st.hyb.mats.contains_key(&(uid, d)) {
                need += sized(d, hidden, inter);
            }
        }
        let mut one = |w: &GraphW, rows: usize, cols: usize| {
            if st.hyb.mats.contains_key(&(uid, w.idx)) {
                return;
            }
            if is8(w) {
                need += i8_layout(rows, cols, 1).2;
            } else if is1(w) {
                need += q1_layout_of(rows, cols, tern(w.idx)).1;
            } else if is2(w) {
                need += q2_bytes(rows, cols);
            } else if q6(w) {
                need += planes_bytes(rows, cols);
            }
        };
        match &l.attn {
            GraphAttn::Gdn { qkv: qw, z: zw, out: ow, nv, nk, dk, dv, .. } => {
                one(qw, 2 * nk * dk + nv * dv, hidden);
                one(zw, nv * dv, hidden);
                one(ow, hidden, nv * dv);
            }
            GraphAttn::Full { wq, wk, wv, wo, output_gate, .. } => {
                one(wq, nh * hd * (1 + *output_gate as usize), hidden);
                one(wk, nkv * hd, hidden);
                one(wv, nkv * hd, hidden);
                one(wo, hidden, nh * hd);
            }
            GraphAttn::ShortConv { inp, out, .. } => {
                one(inp, 3 * hidden, hidden);
                one(out, hidden, hidden);
            }
            _ => {}
        }
    }
    if let Some((gw, rows)) = lm {
        if q6(gw) && !st.hyb.mats.contains_key(&(uid, gw.idx)) {
            need += planes_bytes(rows, hidden);
        } else if is1(gw) && !st.hyb.mats.contains_key(&(uid, gw.idx)) {
            need += q1_layout_of(rows_of(model, gw).unwrap_or(rows), hidden, tern(gw.idx)).1;
        } else if is2(gw) && !st.hyb.mats.contains_key(&(uid, gw.idx)) {
            need += q2_bytes(rows_of(model, gw).unwrap_or(rows), hidden);
        }
    }
    if need > 0 && std::env::var("CMF_HYB_ARENA").as_deref() != Ok("0") {
        let a = Buf::new(need)?;
        ARENA.with(|x| x.set((a.ptr, need, 0)));
        st.hyb.arenas.push(a);
    }
    struct ArenaEnd;
    impl Drop for ArenaEnd {
        fn drop(&mut self) {
            ARENA.with(|x| x.set((0, 0, 0)));
        }
    }
    let _arena_end = ArenaEnd;
    for (li, l) in layers.iter().enumerate() {
        if li % 16 == 1 {
            mem_report(c, st, &format!("{li} layers of a plan"));
        }
        let gw6 = |idx: usize| GraphW { idx, kind: 6, row_scale: &[], data: &[], prism: crate::gpu::GraphPrismOp::None, affine: false };
        let (gum, dm, inter, moe) = match &l.ffn {
            GraphFfn::Dense { gate: gw, up: uw, down: dw, .. } => {
                let inter = rows_of(model, gw).ok_or("f32 FFN")?;
                (hmat_pair(c, st, model, gw, uw, inter, hidden)?, hmat(c, st, model, dw, hidden, inter)?, inter, None)
            }
            GraphFfn::Moe {
                router, shared_gate, experts, n_exp, top_k, inter, norm_topk, sigmoid, sigmoid_denom_eps, bias, has_shared,
                detached_shared, shared_gated, route_scale, ..
            } => {
                let (e, k, inter) = (*n_exp, *top_k, *inter);
                let mut mats = Vec::with_capacity(2 * e);
                let (mut gu_t, mut dn_t) = (Vec::with_capacity(e), Vec::with_capacity(e));
                let job = |m: &DevMat, rows: usize, kind: i32| HyJob {
                    w: m.w.ptr,
                    aux0: m.aux0(),
                    aux1: m.aux1(),
                    y: 0,
                    res: 0,
                    rows: rows as i32,
                    kind,
                    blk0: 0,
                    ystride: 0,
                    xoff: 0,
                    nr: 0,
                };
                for &(g, u, d) in &experts[..e] {
                    let pm = hmat_pair(c, st, model, &gw6(g), &gw6(u), inter, hidden)?;
                    let dmx = hmat(c, st, model, &gw6(d), hidden, inter)?;
                    gu_t.push(job(&pm, 2 * inter, pm.kind as i32 | 0x100));
                    dn_t.push(job(&dmx, hidden, dmx.kind as i32));
                    mats.push(pm);
                    mats.push(dmx);
                }
                // the shared expert: the last entry, one of its own width, or
                // none (the first routed expert then stands in, never launched)
                let shared = *has_shared || detached_shared.is_some();
                let (sg_, su_, sd_, sh) = match detached_shared {
                    Some(d) => (d.gate.idx, d.up.idx, d.down.idx, d.width),
                    None if *has_shared => {
                        let (g, u, d) = experts[e];
                        (g, u, d, model.tensors.get(g).and_then(|t| t.shape.first().copied()).ok_or("shared expert shape")?)
                    }
                    None => (experts[0].0, experts[0].1, experts[0].2, inter),
                };
                let gated = *has_shared && *shared_gated;
                let pm = hmat_pair(c, st, model, &gw6(sg_), &gw6(su_), sh, hidden)?;
                let dmx = hmat(c, st, model, &gw6(sd_), hidden, sh)?;
                let routerm = st.mat(c, model, router, e, hidden)?;
                let sgw = if gated { Some(st.mat(c, model, shared_gate, 1, hidden)?) } else { None };
                let biasb = match bias {
                    Some(b) if b.len() == e => {
                        note(b);
                        Some(st.f32_buf(b)?)
                    }
                    Some(_) => return Err(format!("layer {li}: a selection bias of the wrong width")),
                    None => None,
                };
                let flags = (*norm_topk as i32) | ((*sigmoid as i32) << 1) | ((biasb.is_some() as i32) << 2) | (((*sigmoid && *sigmoid_denom_eps < 1e-10) as i32) << 5);
                let one_kd = |t: &[HyJob]| {
                    let d = t.first().and_then(|j| kd_of(j.kind & 0xFF))?;
                    t.iter().all(|j| kd_of(j.kind & 0xFF) == Some(d)).then_some(d)
                };
                let kd = (one_kd(&gu_t), one_kd(&dn_t));
                let ml = MoeL {
                    router: routerm,
                    sgw,
                    bias: biasb,
                    gu_t: Buf::from_slice(&gu_t)?,
                    dn_t: Buf::from_slice(&dn_t)?,
                    e,
                    k,
                    inter,
                    flags,
                    route_scale: *route_scale,
                    gated,
                    bpj_gu: (2 * inter).div_ceil(16) as u32,
                    bpj_dn: hidden.div_ceil(16) as u32,
                    kd,
                    shared,
                    _mats: mats,
                };
                (pm, dmx, sh, Some(ml))
            }
            _ => unreachable!(),
        };
        note(l.input_norm);
        note(l.post_norm);
        let in_norm = st.f32_buf(l.input_norm)?;
        let post_norm = st.f32_buf(l.post_norm)?;
        let mix = match &l.attn {
            GraphAttn::Gdn { qkv: qw, z: zw, a: aw, b: bw, out: ow, conv1d, a_log, dt_bias, norm, nv, nk, dk, dv, kk, .. } => {
                let (nv, nk, dk, dv, kk) = (*nv, *nk, *dk, *dv, *kk);
                let cdim = 2 * nk * dk + nv * dv;
                if conv1d.len() != cdim * kk || a_log.len() != nv || dt_bias.len() != nv || norm.len() != dv {
                    return Err(format!("layer {li}: GDN constants of unexpected lengths"));
                }
                let qm = hmat(c, st, model, qw, cdim, hidden)?;
                let zm = hmat(c, st, model, zw, nv * dv, hidden)?;
                let am = hmat(c, st, model, aw, nv, hidden)?;
                let bm = hmat(c, st, model, bw, nv, hidden)?;
                let om = hmat(c, st, model, ow, hidden, nv * dv)?;
                for d in [*conv1d, *a_log, *dt_bias, *norm] {
                    note(d);
                }
                let half_of = |m: &DevMat, st: &mut State| -> Res<Buf> {
                    let n = m.rows * m.cols;
                    let hb = st.slab_buf(n * 2 + 64)?;
                    let src = if m.kind == 4 {
                        m.w.ptr
                    } else {
                        return Err("a GDN gate projection that is not f32".into());
                    };
                    let n64 = n as u64;
                    unsafe { c.launch(fx.to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(src, hb.ptr, n64))? };
                    Ok(hb)
                };
                let a_h = half_of(&am, st)?;
                let b_h = half_of(&bm, st)?;
                let ms_in = mmaset(&[(&qm, qkv.ptr, 0, cdmax, cdim), (&zm, z.ptr, 0, vdmax, nv * dv)])?;
                // a sandwich norm or the mHC streams take the product apart from the stream
                let ms_out = match l.attn_out_norm.is_some() || l.mhc.is_some() {
                    true => mmaset(&[(&om, yo.ptr, 0, hidden, hidden)])?,
                    false => mmaset(&[(&om, h.ptr, h.ptr, hidden, hidden)])?,
                };
                Mix::Gdn(GdnL {
                    qkv: qm,
                    z: zm,
                    a: am,
                    b: bm,
                    out: om,
                    a_h,
                    b_h,
                    taps: st.f32_buf(conv1d)?,
                    a_log: st.f32_buf(a_log)?,
                    dt_bias: st.f32_buf(dt_bias)?,
                    norm: st.f32_buf(norm)?,
                    nv,
                    nk,
                    dk,
                    dv,
                    kk,
                    cdim,
                    ms_in,
                    ms_out,
                })
            }
            GraphAttn::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, late_qk_norm, bias, head_gate, geom, .. } => {
                // the layer's head geometry and its kernels (the call's module for the call's)
                let (lnkv, lhd) = geom.as_ref().map_or((nkv, hd), |g| (g.nkv, g.hd));
                let lhpk = nh / lnkv;
                let lfx = if (lnkv, lhd) == (nkv, hd) {
                    fx.clone()
                } else {
                    match st.hyb.funcs.get(&(lhpk, lhd)) {
                        Some(f) => f.clone(),
                        None => {
                            let f = Arc::new(load_funcs(c, lhpk, lhd)?);
                            st.hyb.funcs.insert((lhpk, lhd), f.clone());
                            f
                        }
                    }
                };
                let (nkv, hd) = (lnkv, lhd);
                let qr = nh * hd * (1 + *output_gate as usize);
                let window = geom.as_ref().and_then(|g| g.window);
                let sink = match geom.as_ref().and_then(|g| g.sink) {
                    Some(s) => {
                        note(s);
                        Some(st.f32_buf(s)?)
                    }
                    None => None,
                };
                let local = geom
                    .as_ref()
                    .filter(|g| g.dash.is_none())
                    .filter(|g| g.invf.len() >= g.rd / 2 && !(g.rd == rd && invf.len() >= rd / 2 && g.invf[..rd / 2] == invf[..rd / 2] && g.rope_scale == 1.0))
                    .and_then(|g| invf_local.iter().position(|(f, s)| f[..] == g.invf[..g.rd / 2] && *s == g.rope_scale))
                    .map_or(0, |i| i + 1);
                // this layer's rotation width (pairs)
                let lhalf = geom.as_ref().map_or(rd / 2, |g| g.rd / 2);
                let hg = match head_gate {
                    Some(gw) => {
                        let m = st.mat(c, model, gw, nh, hidden)?;
                        let (mut tabs, mut blk) = (Vec::with_capacity(NMAX), 0u32);
                        for r in 0..NMAX {
                            let (mut jobs, mut b) = (Vec::new(), 0u32);
                            super::push_job(&mut jobs, &mut b, &m, hgv.ptr + (r * nh * 4) as u64, 0);
                            blk = b;
                            tabs.push(Buf::from_slice(&jobs)?);
                        }
                        Some((m, tabs, blk))
                    }
                    None => None,
                };
                let qm = hmat(c, st, model, wq, qr, hidden)?;
                let km = hmat(c, st, model, wk, nkv * hd, hidden)?;
                let vm = hmat(c, st, model, wv, nkv * hd, hidden)?;
                let om = hmat(c, st, model, wo, hidden, nh * hd)?;
                let nrm = |d: Option<&[f32]>, st: &mut State| -> Res<Option<Arc<Buf>>> {
                    match d {
                        Some(d) if d.len() == hd => Ok(Some(st.f32_buf(d)?)),
                        Some(_) => Err(format!("layer {li}: Q/K norm width")),
                        None => Ok(None),
                    }
                };
                if let Some(d) = q_norm {
                    note(d);
                }
                if let Some(d) = k_norm {
                    note(d);
                }
                let qn = nrm(*q_norm, st)?;
                let kn = nrm(*k_norm, st)?;
                let bias = match bias {
                    Some((bq, bk, bv)) => {
                        if bq.len() != nh * hd || bk.len() != nkv * hd || bv.len() != nkv * hd {
                            return Err(format!("layer {li}: projection bias widths"));
                        }
                        for d in [*bq, *bk, *bv] {
                            note(d);
                        }
                        Some((st.f32_buf(bq)?, st.f32_buf(bk)?, st.f32_buf(bv)?))
                    }
                    None => None,
                };
                let ms_in = mmaset(&[
                    (&qm, q.ptr, 0, qrow, qr),
                    (&km, k.ptr, 0, nkv * hd, nkv * hd),
                    (&vm, v.ptr, 0, nkv * hd, nkv * hd),
                ])?;
                // a sandwich norm or the mHC streams take the product apart from the stream
                let ms_out = match l.attn_out_norm.is_some() || l.mhc.is_some() {
                    true => mmaset(&[(&om, yo.ptr, 0, hidden, hidden)])?,
                    false => mmaset(&[(&om, h.ptr, h.ptr, hidden, hidden)])?,
                };
                let dash = match geom.as_ref().and_then(|g| g.dash.as_ref()) {
                    Some(d) => {
                        for w in [d.qk_proj, d.q_norm, d.k_norm, d.head_cls] {
                            note(w);
                        }
                        Some(DashL {
                            w: st.f32_buf(d.qk_proj)?,
                            qn: st.f32_buf(d.q_norm)?,
                            kn: st.f32_buf(d.k_norm)?,
                            hc: st.f32_buf(d.head_cls)?,
                            k: d.k,
                            block: d.block,
                            local_blocks: d.local_blocks,
                            idx_scale: d.idx_scale,
                            floor: d.floor,
                            sigma: d.sigma,
                        })
                    }
                    None => None,
                };
                Mix::Attn(AttnL { q: qm, k: km, v: vm, o: om, qn, kn, gated: *output_gate, late: *late_qk_norm, v_norm: geom.as_ref().is_some_and(|g| g.v_norm), bias, hg, window, sink, local, half: lhalf, nkv, hd, fx: lfx, ms_in, ms_out, dash })
            }
            GraphAttn::ShortConv { inp, out: ow, taps, kernel, .. } => {
                let im = hmat(c, st, model, inp, 3 * hidden, hidden)?;
                let om = hmat(c, st, model, ow, hidden, hidden)?;
                note(taps);
                let taps = st.f32_buf(taps)?;
                let ms_in = mmaset(&[(&im, qkv.ptr, 0, cdmax, 3 * hidden)])?;
                // a sandwich norm or the mHC streams take the product apart from the stream
                let ms_out = match l.attn_out_norm.is_some() || l.mhc.is_some() {
                    true => mmaset(&[(&om, yo.ptr, 0, hidden, hidden)])?,
                    false => mmaset(&[(&om, h.ptr, h.ptr, hidden, hidden)])?,
                };
                Mix::Conv(ConvL { inp: im, out: om, taps, kernel: *kernel, ms_in, ms_out })
            }
            _ => unreachable!(),
        };
        let gelu = matches!(&l.ffn, GraphFfn::Dense { act: GraphAct::GeluTanh, .. });
        let ms_gu = mmaset_kind(&[(&gum, qq[0].ptr, qq[1].ptr, pstride, gum.rows)], 6 | 0x100 | if gelu { 0x200 } else { 0 })?;
        // a routed FFN's shared expert goes to its own rows, mixed in after the routed ones
        let ms_dn = match (&moe, &moe_bufs) {
            (Some(_), Some(mb)) => mmaset(&[(&dm, mb.dsh.ptr, 0, hidden, hidden)])?,
            _ if l.ffn_out_norm.is_some() || l.mhc.is_some() => mmaset(&[(&dm, yo.ptr, 0, hidden, hidden)])?,
            _ => mmaset(&[(&dm, h.ptr, h.ptr, hidden, hidden)])?,
        };
        let mut sw = |w: Option<&[f32]>, st: &mut State| -> Res<Option<Arc<Buf>>> {
            match w {
                Some(w) => {
                    note(w);
                    Ok(Some(st.f32_buf(w)?))
                }
                None => Ok(None),
            }
        };
        let attn_out = sw(l.attn_out_norm, st)?;
        let ffn_out = sw(l.ffn_out_norm, st)?;
        let mhc = match &l.mhc {
            Some(m) => {
                for d in [m.attn_w, m.attn_b, m.ffn_w, m.ffn_b] {
                    note(d);
                }
                Some(MhcL {
                    aw: st.f32_buf(m.attn_w)?,
                    ab: st.f32_buf(m.attn_b)?,
                    fw: st.f32_buf(m.ffn_w)?,
                    fb: st.f32_buf(m.ffn_b)?,
                    out: m.attn_b.len(),
                    enter: m.enter,
                    leave: m.leave,
                })
            }
            None => None,
        };
        out.push(HLayer { mix, gu: gum, down: dm, in_norm, post_norm, inter, gelu, ms_gu, ms_dn, moe, mhc, attn_out, ffn_out, layer_scale: l.layer_scale });
    }
    let loop_w = if loop_norm.is_empty() {
        None
    } else {
        if final_norm.len() != hidden || loop_norm.iter().any(|&l| l >= layers.len()) {
            return Err("a mid-stack norm past the stack or of the wrong width".into());
        }
        note(final_norm);
        Some(st.f32_buf(final_norm)?)
    };
    let (lm, logits, final_norm_buf) = match lm {
        Some((gw, rows)) => {
            if final_norm.len() != hidden {
                return Err("final norm width".into());
            }
            let logits = f4(NMAX * rows)?;
            let head_q1 = matches!(
                model.tensors.get(gw.idx).map(|e| e.dtype),
                Some(cortiq_core::TensorDtype::Q1 | cortiq_core::TensorDtype::Q1T | cortiq_core::TensorDtype::Q2TiledP | cortiq_core::TensorDtype::Q4Tiled)
            );
            let head = if gw.kind == 6 || head_q1 {
                let m = hmat(c, st, model, gw, rows_of(model, gw).ok_or("an f32 head")?, hidden)?;
                if rows == 0 || rows > m.rows {
                    return Err(format!("head of {rows} rows over a {}-row matrix", m.rows));
                }
                let js = mmaset(&[(&m, logits.ptr, 0, rows, rows)])?;
                Head::Mma(m, rows, js)
            } else {
                // a q8_2f (or f32) head: the dense arm's matrix and matvec
                let full = model.tensors.get(gw.idx).and_then(|e| e.shape.first().copied()).ok_or("head shape")?;
                if rows == 0 || rows > full || super::mv_shared_bytes(hidden) > c.max_shared {
                    return Err(format!("head of {rows} rows over a {full}-row matrix"));
                }
                let m = st.mat(c, model, gw, full, hidden)?;
                let mut tabs = Vec::with_capacity(NMAX);
                let mut blk = 0u32;
                for r in 0..NMAX {
                    let (mut jobs, mut b) = (Vec::new(), 0u32);
                    super::push_job(&mut jobs, &mut b, &m, logits.ptr + (r * rows * 4) as u64, 0);
                    // only the head's first `rows` rows (a trimmed vocabulary)
                    jobs[0].rows = rows as i32;
                    b = rows.div_ceil(jobs[0].rpb as usize) as u32;
                    blk = b;
                    tabs.push(Buf::from_slice(&jobs)?);
                }
                Head::Dense(m, rows, tabs, blk)
            };
            // (in the signature by content: no pointer of it is kept)
            (Some(head), logits, Some(st.f32_buf(final_norm)?))
        }
        None => (None, f4(1)?, None),
    };
    let _ = lm_rows;
    let pp = [Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?];
    static PLAN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let n_attn = layers.iter().filter(|l| matches!(l.attn, GraphAttn::Full { .. })).count();
    let has_dash = out.iter().any(|l| matches!(&l.mix, Mix::Attn(a) if a.dash.is_some()));
    Ok(HPlan {
        id: PLAN_ID.fetch_add(1, Ordering::Relaxed),
        ci: Buf::new(n_attn.max(1) * 8 * 4)?,
        sig,
        consts,
        fx,
        layers: out,
        lm,
        final_norm: final_norm_buf,
        invf: invf[..half].to_vec(),
        invf_local,
        prism,
        hdmax,
        loop_norm: loop_norm.to_vec(),
        loop_w,
        mhc: match mhc_geo {
            Some((ns, iters, pre_scale)) => Some(MhcPlan { ns, iters, pre_scale, r: Buf::new(NMAX * ns * hidden * 4)?, coef: Buf::new(NMAX * 20 * 4)?, lg: Buf::new(NMAX * 24 * 4)? }),
            None => None,
        },
        dash: if has_dash {
            let cand = DASH_ROWS * DASH_BLOCKS;
            Some(DashBufs {
                cands: Buf::new(cand * 4)?,
                coff: Buf::new((DASH_ROWS + 1) * 4)?,
                tab: Buf::new(DASH_ROWS * 64 * 2 * 4)?,
                rec: Buf::new(4 * 4)?,
                sc: Buf::new(cand * 4)?,
                sel: Buf::new(DASH_ROWS * 17 * 4)?,
                bias: Buf::new(DASH_ROWS * 16 * 4)?,
            })
        } else {
            None
        },
        own_geom: hdmax != hd || kvmax != nkv * hd,
        half,
        nh,
        nkv,
        hd,
        hpk,
        hidden,
        gemma,
        eps,
        scale,
        h,
        xn,
        q,
        k,
        v,
        qg,
        gate,
        ao,
        qkv,
        z,
        o,
        of,
        logits,
        tab,
        ids,
        pp,
        qq,
        pstride,
        qrow,
        cdmax,
        vdmax,
        nvmax,
        moe: moe_bufs,
        yo,
        hgv,
    })
}

#[allow(clippy::too_many_arguments)]
fn plan_for(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    layer_base: usize,
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    gemma: bool,
    eps: f32,
    scale: f32,
    invf: &[f32],
    first_pos: usize,
    loop_norm: &[usize],
) -> Res<Arc<HPlan>> {
    let key = (model.uid(), layer_base, layers.len(), lm.is_some());
    let mut sig = hsig(model, layers, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, invf);
    // a Looped Transformer's mid-stack norms are part of the plan
    sig.push(usize::MAX);
    sig.extend_from_slice(loop_norm);
    if let Some(p) = st.hyb.plans.get(&key) {
        if p.sig == sig && p.eps == eps && p.scale == scale {
            // SAFETY: the signature holds these addresses and lengths, and the
            // caller's slices are alive for the call.
            let same = first_pos != 0
                || p.consts.iter().all(|&(ptr, len, fp)| {
                    crate::gpu::fp_f32(unsafe { std::slice::from_raw_parts(ptr as *const f32, len) }) == fp
                });
            if same {
                return Ok(p.clone());
            }
        }
    }
    c.sync()?;
    let t0 = std::time::Instant::now();
    let p = Arc::new(build(c, st, model, layers, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, scale, invf, sig, loop_norm)?);
    mem_report(c, st, &format!("a plan of {} layers", p.layers.len()));
    tracing::info!(
        "cuda hybrid graph: {} layers resident, {:.2} GB of weights, built in {:.1} s (repack {:.1} s, upload {:.1} s, single weights {:.1} s, job tables {:.1} s)",
        p.layers.len(),
        c.resident.load(Ordering::Relaxed) as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        REPACK_NS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        UPLOAD_NS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        HMAT_NS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        SETS_NS.swap(0, Ordering::Relaxed) as f64 / 1e9
    );
    st.hyb.plans.insert(key, p.clone());
    Ok(p)
}

/// (sin, cos) per row and frequency, `[row][pair][2]`, as the host's table.
/// The call's rotation rows, followed by the local table's when the plan has one.
fn plan_tab(plan: &HPlan, positions: impl Iterator<Item = usize> + Clone) -> Vec<f32> {
    let mut t = rope_rows(&plan.invf, positions.clone());
    for (l, sc) in &plan.invf_local {
        // the amplitude multiplies the rotated pair: both sin and cos
        t.extend(rope_rows(l, positions.clone()).into_iter().map(|v| v * sc));
    }
    t
}

fn rope_rows(invf: &[f32], positions: impl Iterator<Item = usize>) -> Vec<f32> {
    let mut tab = Vec::new();
    for p in positions {
        for &f in invf {
            let (s, co) = (p as f32 * f).sin_cos();
            tab.push(s);
            tab.push(co);
        }
    }
    tab
}

/// `KvMirror::sync_to` for an f16 mirror: rewind past rows of another
/// continuation, or take the rows the host holds (as f16) and the card does
/// not. Ok(false): the host does not hold them either.
#[allow(clippy::too_many_arguments)]
fn sync_half(c: &Ctx, m: &mut KvMirror, cpu_k: &[Vec<f32>], cpu_v: &[Vec<f32>], cpu_base: usize, nkv: usize, hd: usize, upto: usize) -> Res<bool> {
    if m.stored > upto {
        if upto >= m.base {
            m.stored = upto;
        } else {
            m.base = 0;
            m.stored = 0;
        }
    }
    if m.stored == upto {
        return Ok(true);
    }
    let from = m.stored;
    let n = upto - from;
    let have = cpu_k.first().map_or(0, |k| k.len() / hd);
    if cpu_k.len() != nkv || cpu_v.len() != nkv || from < cpu_base || upto > cpu_base + have {
        return Ok(false);
    }
    m.reserve(c, n, 0)?;
    let row = nkv * hd;
    let mut kb = vec![0u16; n * row];
    let mut vb = vec![0u16; n * row];
    for r in 0..n {
        let src = (from - cpu_base + r) * hd;
        for g in 0..nkv {
            for d in 0..hd {
                kb[r * row + g * hd + d] = cortiq_core::quant::f32_to_f16(cpu_k[g][src + d]);
                vb[r * row + g * hd + d] = cortiq_core::quant::f32_to_f16(cpu_v[g][src + d]);
            }
        }
    }
    let off = ((from - m.base) * row * 2) as u64;
    unsafe {
        cu::memcpy_htod_async(m.kptr() + off, &kb, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(m.vptr() + off, &vb, c.stream).map_err(derr)?;
    }
    // the host vectors go when this returns
    c.sync()?;
    m.stored = upto;
    Ok(true)
}

/// Bring every layer's state to `pos0` rows before anything is launched:
/// the K/V mirrors (from the host's rows where the card lacks them) and the
/// GDN states (fresh at a sequence start, seeded from the host's state after
/// a host prefill). Err(true): the card holds state this call cannot use.
#[allow(clippy::too_many_arguments)]
fn admit(
    c: &Ctx,
    st: &mut State,
    plan: &HPlan,
    layers: &[GraphLayer],
    kv_id: u64,
    layer_base: usize,
    pos0: usize,
    n: usize,
    hint: usize,
    sealed: &[bool],
) -> Result<(), (bool, String)> {
    let soft = |e: String| (false, e);
    for (li, (pl, l)) in plan.layers.iter().zip(layers).enumerate() {
        let key = (kv_id, layer_base + li);
        if sealed.get(li).copied().unwrap_or(false) {
            continue;
        }
        match (&pl.mix, &l.attn) {
            (Mix::Attn(a), GraphAttn::Full { cpu_k, cpu_v, cpu_base, .. }) => {
                // f16 rows: `row` counts f32 slots of a mirror, so half the values
                // (the layer's own head geometry)
                let row = a.nkv * a.hd / 2;
                let fresh_half = !st.hyb.half.contains(&key);
                let m = st.kv.entry(key).or_insert_with(|| KvMirror::new(row, 0));
                if m.row != row || m.window != 0 || fresh_half {
                    let _ = c.sync();
                    c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
                    *m = KvMirror::new(row, 0);
                }
                st.hyb.half.insert(key);
                let m = st.kv.get_mut(&key).unwrap();
                match sync_half(c, m, cpu_k, cpu_v, *cpu_base, a.nkv, a.hd, pos0) {
                    Ok(true) => m.reserve(c, n, hint).map_err(soft)?,
                    Ok(false) if m.stored < pos0 && m.stored >= m.base => {
                        // Rows neither on the card nor on the host: positions
                        // a draft block never saw (the plain tokens of a
                        // speculation trial). The wgpu graph reads whatever
                        // its mirror holds there; here they are zero rows,
                        // which a draft's attention sees as keys of score 0.
                        let gap = pos0 - m.stored;
                        m.reserve(c, gap + n, hint).map_err(soft)?;
                        let off = ((m.stored - m.base) * m.row * 4) as u64;
                        unsafe {
                            cu::memset_d8_async(m.kptr() + off, 0, gap * m.row * 4, c.stream).map_err(|e| soft(derr(e)))?;
                            cu::memset_d8_async(m.vptr() + off, 0, gap * m.row * 4, c.stream).map_err(|e| soft(derr(e)))?;
                        }
                        if debug() {
                            eprintln!("cuda hybrid: layer {li} of sequence {kv_id}: {gap} K/V rows to {pos0} filled with zeros");
                        }
                        m.stored = pos0;
                    }
                    Ok(false) => return Err(soft("a layer's K/V rows are neither on the card nor on the host".into())),
                    Err(e) => return Err(soft(e)),
                }
            }
            (Mix::Gdn(g), GraphAttn::Gdn { cpu_state, .. }) => {
                let dims = (g.nv, g.nk, g.dk, g.dv, g.kk);
                let ring_n = (g.kk - 1) * g.cdim;
                let s_n = g.nv * g.dk * g.dv;
                if let Some(gs) = st.hyb.gdn.get_mut(&key) {
                    if gs.dims == dims && gs.next_pos == pos0 {
                        continue;
                    }
                    if gs.dims == dims && pos0 == 0 {
                        // a new sequence on the same id: no history
                        unsafe {
                            cu::memset_d8_async(gs.s.ptr, 0, s_n * 4, c.stream).map_err(|e| soft(derr(e)))?;
                            cu::memset_d8_async(gs.ring.ptr, 0, ring_n.max(1) * 4, c.stream).map_err(|e| soft(derr(e)))?;
                        }
                        gs.next_pos = 0;
                        gs.tape = None;
                        continue;
                    }
                    return Err((true, format!("GDN state of layer {li} is at row {}, the call at {pos0}", gs.next_pos)));
                }
                if pos0 > 0 && cpu_state.len() != ring_n + s_n {
                    return Err(soft("GDN host seed missing for a nonzero position".into()));
                }
                // the seven buffers of a layer in one allocation, zeroed
                let sizes = [s_n, ring_n, NMAX * plan.cdmax, NMAX * g.cdim, NMAX * 2 * g.nv, s_n, ring_n].map(|n| (n.max(1) * 4).next_multiple_of(256));
                let alloc = || -> Res<GdnState> {
                    let all = Buf::new(sizes.iter().sum())?;
                    unsafe { cu::memset_d8_async(all.ptr, 0, all.bytes, c.stream).map_err(derr)? };
                    let mut at = 0usize;
                    let mut v = Vec::with_capacity(sizes.len());
                    for &n in &sizes {
                        v.push(all.view(at, n));
                        at += n;
                    }
                    let mut v = v.into_iter();
                    let mut next = || v.next().ok_or("GDN state views");
                    let (s, ring, t_qkv, t_cq, t_gb, s_save, ring_save) = (next()?, next()?, next()?, next()?, next()?, next()?, next()?);
                    if pos0 > 0 {
                        unsafe {
                            if ring_n > 0 {
                                cu::memcpy_htod_async(ring.ptr, &cpu_state[..ring_n], c.stream).map_err(derr)?;
                            }
                            cu::memcpy_htod_async(s.ptr, &cpu_state[ring_n..], c.stream).map_err(derr)?;
                        }
                    }
                    Ok(GdnState {
                        dims,
                        _all: all,
                        s,
                        ring,
                        next_pos: pos0,
                        t_qkv,
                        t_stride: plan.cdmax,
                        t_cq,
                        t_gb,
                        s_save,
                        ring_save,
                        tape: None,
                        snap: None,
                    })
                };
                let gs = alloc().map_err(soft)?;
                c.kv_bytes.fetch_add(((s_n * 2 + ring_n * 2 + NMAX * (2 * g.cdim + 2 * g.nv)) * 4) as u64, Ordering::Relaxed);
                st.hyb.gdn.insert(key, gs);
            }
            (Mix::Conv(cv), GraphAttn::ShortConv { cpu_state, .. }) => {
                // the ring rides a GDN state with no recurrent matrix: dims whose
                // conv width is the hidden one (nv 1, dv hidden, no keys), so the
                // state helpers size its ring right ([channel][k - 1], the host's)
                let dims = (1, 0, 0, plan.hidden, cv.kernel);
                let ring_n = (cv.kernel - 1) * plan.hidden;
                if let Some(gs) = st.hyb.gdn.get_mut(&key) {
                    if gs.dims == dims && gs.next_pos == pos0 {
                        continue;
                    }
                    if gs.dims == dims && pos0 == 0 {
                        unsafe { cu::memset_d8_async(gs.ring.ptr, 0, ring_n.max(1) * 4, c.stream).map_err(|e| soft(derr(e)))? };
                        gs.next_pos = 0;
                        gs.tape = None;
                        continue;
                    }
                    return Err((true, format!("short-conv ring of layer {li} is at row {}, the call at {pos0}", gs.next_pos)));
                }
                if pos0 > 0 && cpu_state.len() != ring_n {
                    return Err(soft("short-conv host seed missing for a nonzero position".into()));
                }
                let sizes = [1usize, ring_n, 1, 1, 1, 1, ring_n].map(|n| (n.max(1) * 4).next_multiple_of(256));
                let alloc = || -> Res<GdnState> {
                    let all = Buf::new(sizes.iter().sum())?;
                    unsafe { cu::memset_d8_async(all.ptr, 0, all.bytes, c.stream).map_err(derr)? };
                    let mut at = 0usize;
                    let mut v = Vec::with_capacity(sizes.len());
                    for &n in &sizes {
                        v.push(all.view(at, n));
                        at += n;
                    }
                    let mut v = v.into_iter();
                    let mut next = || v.next().ok_or("short-conv state views");
                    let (s, ring, t_qkv, t_cq, t_gb, s_save, ring_save) = (next()?, next()?, next()?, next()?, next()?, next()?, next()?);
                    if pos0 > 0 && ring_n > 0 {
                        unsafe { cu::memcpy_htod_async(ring.ptr, &cpu_state[..ring_n], c.stream).map_err(derr)? };
                    }
                    Ok(GdnState { dims, _all: all, s, ring, next_pos: pos0, t_qkv, t_stride: plan.cdmax, t_cq, t_gb, s_save, ring_save, tape: None, snap: None })
                };
                let gs = alloc().map_err(soft)?;
                // the count `reset` takes back for these dims
                c.kv_bytes.fetch_add(((ring_n * 2 + NMAX * (2 * plan.hidden + 2)) * 4) as u64, Ordering::Relaxed);
                st.hyb.gdn.insert(key, gs);
            }
            _ => return Err(soft(format!("layer {li}: descriptors changed under the plan"))),
        }
    }
    Ok(())
}

/// A sealed layer's far field on the card (attn.cu's `O1Dev` with f16 near
/// rows: the sinks and the rows kept exact, then the ring of the window, laid
/// out as a K/V mirror; the landmarks and each head's far accumulators). It
/// goes up once per seal epoch and moves with every row from there.
struct HO1 {
    epoch: u64,
    /// The position of the next row that may move this state; None until the
    /// first one has.
    next_pos: Option<usize>,
    bytes: u64,
    nk: Buf,
    nv: Buf,
    kt: Buf,
    qt: Buf,
    mu: Buf,
    mz: Buf,
    th: Buf,
    ns: usize,
    w: usize,
    m: usize,
    win_len: usize,
    win_head: usize,
    far_len: usize,
    scale: f32,
    rect_fm: bool,
    /// A verify window's rows (the first row's index, rows): written after the
    /// near rows, not yet in the window; `spec_commit` moves the accepted ones.
    spec: Option<(usize, usize)>,
    /// The layer's geometry: KV heads, query heads a KV head, head width.
    nkv: usize,
    hpk: usize,
    hd: usize,
}

/// A sealed layer's far field up for this epoch (unless it is already).
#[allow(clippy::too_many_arguments)]
fn o1_ensure(c: &Ctx, st: &mut State, key: (u64, usize), views: &[crate::nystrom::O1DeviceView<'_>], epoch: u64, hpk: usize, nkv: usize, hd: usize) -> Res<()> {
    if st.hyb.o1.get(&key).is_some_and(|d| d.epoch == epoch) {
        return Ok(());
    }
    let g0 = views.first().ok_or("far field: a layer without groups")?;
    let (m, ns, w) = (g0.m_eff, g0.sink_len, g0.w);
    let rect_fm = g0.heads.first().is_some_and(|h| h.rect_fm);
    if views.len() != nkv
        || views.iter().any(|v| {
            v.exact_only
                || v.d != hd
                || v.dv != hd
                || v.m_eff != m
                || v.sink_len != ns
                || v.w != w
                || v.heads.len() != hpk
                || v.win_len != g0.win_len
                || v.win_head != g0.win_head
                || v.far_len != g0.far_len
                || v.scale != g0.scale
                || v.heads.iter().any(|h| h.rect_fm != rect_fm)
        })
    {
        return Err("far field: the groups of a layer disagree, or one is exact-only".into());
    }
    if m == 0 || m > 64 || hpk * m > 1024 || w == 0 || g0.win_len > w {
        return Err(format!("far field: {m} landmarks x {hpk} heads a group, window {w}: outside the kernels"));
    }
    let row = nkv * hd;
    let h16 = cortiq_core::quant::f32_to_f16;
    // room after the near rows for a verify window's rows, or a prompt chunk's
    let extra = NMAX.max(SEAL_ROWS);
    let mut nk = vec![0u16; (ns + w + extra) * row];
    let mut nv = vec![0u16; (ns + w + extra) * row];
    let (mut kt, mut qt, mut mu, mut mz, mut th) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (g, v) in views.iter().enumerate() {
        let have = (v.win_k.len() / hd).min(v.win_v.len() / hd).min(w);
        if v.sink_k.len() != ns * hd || v.sink_v.len() != ns * hd || v.k_tilde.len() != m * hd || have < v.win_len {
            return Err("far field: sink, window or landmark rows of an unexpected length".into());
        }
        for r in 0..ns {
            for d in 0..hd {
                nk[(r * nkv + g) * hd + d] = h16(v.sink_k[r * hd + d]);
                nv[(r * nkv + g) * hd + d] = h16(v.sink_v[r * hd + d]);
            }
        }
        for r in 0..have {
            for d in 0..hd {
                nk[((ns + r) * nkv + g) * hd + d] = h16(v.win_k[r * hd + d]);
                nv[((ns + r) * nkv + g) * hd + d] = h16(v.win_v[r * hd + d]);
            }
        }
        kt.extend_from_slice(v.k_tilde);
        for h in &v.heads {
            if h.q_tilde.len() != m * hd || h.mu.len() != m * m || h.m_max.len() != m || h.z_hat.len() != m || h.t_hat.len() != m * hd {
                return Err("far field: a head's skeleton of an unexpected length".into());
            }
            qt.extend_from_slice(h.q_tilde);
            mu.extend_from_slice(h.mu);
            mz.extend_from_slice(h.m_max);
            mz.extend_from_slice(h.z_hat);
            th.extend_from_slice(h.t_hat);
        }
    }
    // queued work may still read the epoch being replaced
    c.sync()?;
    let bytes = ((nk.len() + nv.len()) * 2 + (kt.len() + qt.len() + mu.len() + mz.len() + th.len()) * 4) as u64;
    let d = HO1 {
        epoch,
        next_pos: None,
        bytes,
        nk: Buf::from_slice(&nk)?,
        nv: Buf::from_slice(&nv)?,
        kt: Buf::from_slice(&kt)?,
        qt: Buf::from_slice(&qt)?,
        mu: Buf::from_slice(&mu)?,
        mz: Buf::from_slice(&mz)?,
        th: Buf::from_slice(&th)?,
        ns,
        w,
        m,
        win_len: g0.win_len,
        win_head: g0.win_head,
        far_len: g0.far_len,
        scale: g0.scale,
        rect_fm,
        spec: None,
        nkv,
        hpk,
        hd,
    };
    if debug() {
        eprintln!(
            "cuda hybrid: far field of layer {} on the card (epoch {epoch}): {ns} exact rows, window {} of {w}, {m} landmarks, {} keys far, {:.1} MB",
            key.1,
            d.win_len,
            d.far_len,
            bytes as f64 / 1e6
        );
    }
    c.kv_bytes.fetch_add(bytes, Ordering::Relaxed);
    if let Some(old) = st.hyb.o1.insert(key, d) {
        c.kv_bytes.fetch_sub(old.bytes, Ordering::Relaxed);
    }
    Ok(())
}

/// One small-path call's launch inputs, made by `book`, which also moves
/// the host-side cursors (mirror rows, GDN positions and tapes).
struct Book {
    /// Per layer: the GDN state's buffers, or the mirror's K and V.
    layers: Vec<[u64; 7]>,
    /// The call record of every attention layer, 8 words each:
    /// n0 (mirror rows before the call), the mirror row of the first row,
    /// rb, ncap, nbcap.
    ci: Vec<i32>,
    /// The attention scratch (S, Mx, P, E).
    scratch: [u64; 4],
    /// Per layer: 0 its exact mirror, 1 a far field, 2 a far field whose ring
    /// row goes into the far sums first.
    far: Vec<u8>,
    /// Per layer: a far field's landmarks, scale and rectification.
    farp: Vec<(usize, f32, bool)>,
}

/// Rows a key block of the decode attention takes for `keys` keys (64 up to
/// 8192 keys, so a window's rows match their one-row steps there).
fn attn_rb(keys: usize) -> usize {
    64 * keys.div_ceil(8192).max(1)
}

/// Key blocks the attention grid covers: every count `attn_rb` allows.
const ATTN_NBLK: usize = 129;

#[allow(clippy::too_many_arguments)]
fn book(
    c: &Ctx,
    st: &mut State,
    plan: &HPlan,
    kv_id: u64,
    layer_base: usize,
    pos0: usize,
    n: usize,
    tape: bool,
    kcap: usize,
    sealed: &[bool],
) -> Res<Book> {
    let (nh, hd) = (plan.nh, plan.hd);
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut far = Vec::with_capacity(plan.layers.len());
    let mut farp = Vec::with_capacity(plan.layers.len());
    let mut ci = Vec::new();
    let mut keys_max = 0usize;
    for (li, pl) in plan.layers.iter().enumerate() {
        let key = (kv_id, layer_base + li);
        if sealed.get(li).copied().unwrap_or(false) && matches!(&pl.mix, Mix::Attn(a) if a.nkv != plan.nkv || a.hd != plan.hd) {
            return Err("a sealed far field over a layer head geometry of its own".into());
        }
        if !matches!(&pl.mix, Mix::Attn(_)) || !sealed.get(li).copied().unwrap_or(false) || !st.hyb.o1.contains_key(&key) {
            far.push(0);
            farp.push((0, 0.0, false));
        }
        match &pl.mix {
            Mix::Gdn(_) | Mix::Conv(_) => {
                if tape && matches!(pl.mix, Mix::Conv(_)) {
                    return Err("a verify window over a short convolution".into());
                }
                let gs = st.hyb.gdn.get_mut(&key).ok_or("GDN state missing")?;
                layers.push([gs.s.ptr, gs.ring.ptr, gs.t_qkv.ptr, gs.t_cq.ptr, gs.t_gb.ptr, gs.s_save.ptr, gs.ring_save.ptr]);
                gs.next_pos = pos0 + n;
                gs.tape = if tape { Some((pos0, n)) } else { None };
            }
            Mix::Attn(_) if st.hyb.o1.contains_key(&key) && sealed.get(li).copied().unwrap_or(false) && (tape || n > 1) => {
                // a verify window over the far field: its rows after the near
                // ones, read by the rows after them; the state moves over the
                // accepted ones only (`spec_commit`)
                if n > NMAX {
                    return Err("a verify window over a far field of more than 16 rows".into());
                }
                let d = st.hyb.o1.get_mut(&key).unwrap();
                let base = d.ns + d.win_len;
                d.spec = Some((base, n));
                let keys = base + n;
                let rb = attn_rb(keys);
                let ncap = keys.next_multiple_of(4096);
                let nbcap = ncap.div_ceil(rb) + 1;
                ci.extend([base as i32, base as i32, rb as i32, ncap as i32, nbcap as i32, 1, base as i32, (d.far_len > 0) as i32]);
                layers.push([d.nk.ptr, d.nv.ptr, d.kt.ptr, d.qt.ptr, d.mu.ptr, d.mz.ptr, d.th.ptr]);
                far.push(1);
                farp.push((d.m, d.scale, d.rect_fm));
                keys_max = keys_max.max(keys);
                continue;
            }
            Mix::Attn(_) if st.hyb.o1.contains_key(&key) && sealed.get(li).copied().unwrap_or(false) => {
                // the far field (`nystrom.rs` advance): the ring row the new
                // key replaces goes into the far sums first, once the ring is full
                let d = st.hyb.o1.get_mut(&key).unwrap();
                let full = d.win_len == d.w;
                let row = d.ns + if full { d.win_head } else { d.win_len };
                if full {
                    d.far_len += 1;
                    d.win_head = (d.win_head + 1) % d.w;
                } else {
                    d.win_len += 1;
                }
                d.next_pos = Some(pos0 + 1);
                let keys = d.ns + d.win_len;
                let rb = attn_rb(keys);
                let ncap = keys.next_multiple_of(4096);
                let nbcap = ncap.div_ceil(rb) + 1;
                ci.extend([(keys - 1) as i32, row as i32, rb as i32, ncap as i32, nbcap as i32, 1, row as i32, (d.far_len > 0) as i32]);
                layers.push([d.nk.ptr, d.nv.ptr, d.kt.ptr, d.qt.ptr, d.mu.ptr, d.mz.ptr, d.th.ptr]);
                far.push(if full { 2 } else { 1 });
                farp.push((d.m, d.scale, d.rect_fm));
                keys_max = keys_max.max(keys);
                continue;
            }
            Mix::Attn(a) => {
                // a DASH layer: its open blocks and block summaries beside the mirror
                let (dopen, dbar) = match &a.dash {
                    Some(dl) => {
                        let e = match st.hyb.dash.entry(key) {
                            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                            std::collections::hash_map::Entry::Vacant(v) => v.insert((Buf::new(3 * dl.block * 128 * 4)?, Buf::new(DASH_BLOCKS * 128 * 4)?)),
                        };
                        (e.0.ptr, e.1.ptr)
                    }
                    None => (0, 0),
                };
                let m = st.kv.get_mut(&key).ok_or("K/V mirror missing")?;
                let n0 = m.stored - m.base;
                layers.push([m.kptr(), m.vptr(), dopen, dbar, 0, 0, 0]);
                m.stored += n;
                let keys = n0 + n;
                let rb = attn_rb(keys);
                let ncap = keys.next_multiple_of(4096);
                let nbcap = ncap.div_ceil(rb);
                let win = match &pl.mix {
                    Mix::Attn(a) => a.window.unwrap_or(0) as i32,
                    _ => 0,
                };
                ci.extend([n0 as i32, n0 as i32, rb as i32, ncap as i32, nbcap as i32, 0, 0, win]);
                keys_max = keys_max.max(keys);
            }
        }
    }
    // the attention scratch sized for this call's records: it grows (by 4096
    // keys at a time) as the rows do, and a graph whose pointers moved is
    // recorded again (its signature holds them). `CMF_HYB_KCAP=1` sizes it for
    // the context cap from the start.
    let kc = if std::env::var("CMF_HYB_KCAP").as_deref() == Ok("1") { kcap.max(keys_max) } else { keys_max }.next_multiple_of(4096).max(4096);
    let nbc = ci.chunks(8).map(|r| r[4].max(0) as usize).max().unwrap_or(0).max(1).min(ATTN_NBLK + 64);
    let scratch = [
        st.scr(c, "hy_S", NMAX * nh * kc * 4)?,
        st.scr(c, "hy_Mx", NMAX * nh * nbc * 4)?,
        st.scr(c, "hy_P", NMAX * nbc * nh * hd.max(plan.hdmax) * 4)?,
        st.scr(c, "hy_E", NMAX * nbc * nh * 4)?,
    ];
    Ok(Book { layers, ci, scratch, far, farp })
}

/// The small path's launches over the rows in `plan.h` (`n <= NMAX`), as
/// `book` laid them out: kernels and device copies only, nothing on the
/// host, so a stream capture can record it. `tape`: keep what a verify
/// needs to step back.
#[allow(clippy::too_many_arguments)]
unsafe fn launch(c: &Ctx, plan: &HPlan, bk: &Book, n: usize, tape: bool, cib: u64, tabp: u64, taps: &[(usize, u64)]) -> Res<()> {
    let fx = &plan.fx;
    let (hidden, nh) = (plan.hidden, plan.nh);
    let (eps, gemma_i, n_i) = (plan.eps, plan.gemma as i32, n as i32);
    let hid_i = hidden as i32;
    let zero = 0i32;
    HCALLS.fetch_add(1, Ordering::Relaxed);
    HCALLS_N[n.min(16)].fetch_add(1, Ordering::Relaxed);
    HROWS.with(|r| r.set(n));
    let pl_ = Planes { p0: plan.pp[0].ptr, p1: plan.pp[1].ptr, p2: plan.pp[2].ptr, stride: plan.pstride };
    let planes = &pl_;
    let (pp0, pp1, pp2, ps_i) = (pl_.p0, pl_.p1, pl_.p2, plan.pstride as i32);
    let [s_p, mx_p, p_p, e_p] = bk.scratch;
    let mut ai = 0usize;
    let mut t0 = std::time::Instant::now();
    hst(c, 11, &mut t0)?;
    for (li, pl) in plan.layers.iter().enumerate() {
        let lb = bk.layers[li];
        unsafe {
            // SenseNova-Loop: the attention's input from the mHC streams
            sub_in(c, fx, plan, pl, false, n)?;
            c.launch(
                fx.rmsnorm2,
                (hidden.div_ceil(1024) as u32, n as u32),
                256,
                0,
                kargs!(plan.h.ptr, pl.in_norm.ptr, plan.xn.ptr, hid_i, eps, gemma_i, pp0, pp1, pp2, ps_i),
            )?;
            // a Prism file: the projections' input in their basis (the GDN
            // gates read the f32 rows in `xn`, untouched)
            prism_planes(c, plan, pp0, pp1, plan.pstride, hidden, n)?;
        }
        hst(c, 0, &mut t0)?;
        match &pl.mix {
            Mix::Conv(cv) => {
                let ring = lb[1];
                let (cd_i, k_i, n_i2) = (plan.cdmax as i32, cv.kernel as i32, n as i32);
                unsafe {
                    cv.ms_in.run(c, fx, planes, hidden, n)?;
                    hst(c, 1, &mut t0)?;
                    c.launch(fx.sconv, (hidden.div_ceil(256) as u32, 1), 256, 0, kargs!(plan.qkv.ptr, cd_i, cv.taps.ptr, ring, hid_i, k_i, n_i2, pp0, pp1, pp2, ps_i))?;
                    hst(c, 2, &mut t0)?;
                    cv.ms_out.run(c, fx, planes, hidden, n)?;
                    sub_out(c, fx, plan, pl, false, n)?;
                    hst(c, 3, &mut t0)?;
                }
            }
            Mix::Gdn(g) => {
                let [gs_s, gs_ring, t_qkv, t_cq, t_gb, s_save, ring_save] = lb;
                if plan.nvmax != g.nv {
                    return Err("GDN layers of different widths in one stack".into());
                }
                let (cd_i, kk_i, nv_i, nk_i, dk_i, dv_i) =
                    (g.cdim as i32, g.kk as i32, g.nv as i32, g.nk as i32, g.dk as i32, g.dv as i32);
                let qs_i = plan.cdmax as i32;
                let s_bytes = g.nv * g.dk * g.dv * 4;
                let r_bytes = (g.kk - 1) * g.cdim * 4;
                unsafe {
                    g.ms_in.run(c, fx, planes, hidden, n)?;
                    hst(c, 1, &mut t0)?;
                    // dk = 128 (every Qwen3.5-family file): the column scan,
                    // which keeps the tape's state copy itself
                    let col_scan = g.dk == 128 && g.dv % 32 == 0;
                    if tape {
                        if !col_scan {
                            cu::memcpy_dtod_async(s_save, gs_s, s_bytes, c.stream).map_err(derr)?;
                        }
                        if r_bytes > 0 {
                            cu::memcpy_dtod_async(ring_save, gs_ring, r_bytes, c.stream).map_err(derr)?;
                        }
                        // the raw rows (the ring replays from them), at the
                        // projection's own row stride
                        cu::memcpy_dtod_async(t_qkv, plan.qkv.ptr, n * plan.cdmax * 4, c.stream).map_err(derr)?;
                    }
                    let units = (g.cdim / g.dk) as u32;
                    c.launch(
                        fx.front,
                        (units + 2 * g.nv as u32, n as u32),
                        g.dk.max(64) as u32,
                        0,
                        kargs!(plan.qkv.ptr, qs_i, gs_ring, g.taps.ptr, t_cq, n_i, cd_i, kk_i, nk_i, dk_i, plan.xn.ptr, hid_i, g.a.w.ptr, g.b.w.ptr, g.a_log.ptr, g.dt_bias.ptr, t_gb, nv_i),
                    )?;
                    let cols = g.nv * g.dv;
                    // the scan, with blocks after it that move the conv ring
                    let ring_p = if g.kk > 1 { gs_ring } else { 0 };
                    if col_scan {
                        let ring_blocks = if g.kk > 1 { (g.cdim as u32).div_ceil(128) } else { 0 };
                        let save = if tape { s_save } else { 0 };
                        c.launch(
                            fx.scan4,
                            ((g.nv * g.dv / 32) as u32 + ring_blocks, 1),
                            128,
                            0,
                            kargs!(t_cq, t_gb, gs_s, plan.o.ptr, n_i, cd_i, nv_i, nk_i, dv_i, ring_p, plan.qkv.ptr, qs_i, kk_i, save),
                        )?;
                    } else {
                        let ring_blocks = if g.kk > 1 { (g.cdim as u32).div_ceil(256) } else { 0 };
                        c.launch(
                            fx.scan,
                            ((cols as u32).div_ceil(8) + ring_blocks, 1),
                            256,
                            0,
                            kargs!(t_cq, t_gb, gs_s, plan.o.ptr, n_i, cd_i, nv_i, nk_i, dk_i, dv_i, ring_p, plan.qkv.ptr, qs_i, kk_i),
                        )?;
                    }
                    let zs_i = plan.vdmax as i32;
                    let sig0 = 0i32;
                    c.launch(
                        fx.out32,
                        ((n * g.nv) as u32, 1),
                        g.dv as u32,
                        0,
                        kargs!(plan.o.ptr, plan.z.ptr, zs_i, g.norm.ptr, plan.of.ptr, nv_i, dv_i, eps, sig0, zero, pp0, pp1, pp2, ps_i),
                    )?;
                    hst(c, 2, &mut t0)?;
                    prism_planes(c, plan, pp0, pp1, plan.pstride, cols, n)?;
                    g.ms_out.run(c, fx, planes, cols, n)?;
                    sub_out(c, fx, plan, pl, false, n)?;
                    hst(c, 3, &mut t0)?;
                }
            }
            Mix::Attn(a) => {
                // the layer's head geometry and its kernels
                let (nkv, hd, fx) = (a.nkv, a.hd, &*a.fx);
                let hpk_l = nh / nkv;
                let (kp, vp) = (lb[0], lb[1]);
                let far = bk.far[li];
                let (fm, fscale, frect) = bk.farp[li];
                let cip = cib + (ai * 8 * 4) as u64;
                let w = &bk.ci[ai * 8..ai * 8 + 8];
                ai += 1;
                let (n0_i, slot0, rb_i, ncap_i, nbcap_i, r0) = (w[0], w[1], w[2], w[3], w[4], 0i32);
                let (nh_i, nkv_i, half_i, gated_i) = (nh as i32, nkv as i32, a.half as i32, a.gated as i32);
                let (qs_i, kvs_i) = (plan.qrow as i32, (nkv * hd) as i32);
                let (qn, kn) = (a.qn.as_ref().map_or(0, |b| b.ptr), a.kn.as_ref().map_or(0, |b| b.ptr));
                let (qk_g, (bqp, bkp, bvp)) = qk_flags(a, gemma_i);
                let tabl = tabp + (plan.tab_off(a.local, n) * 4) as u64;
                let gate_p = if a.gated { plan.gate.ptr } else { 0 };
                let nul16 = 0u64;
                unsafe {
                    a.ms_in.run(c, fx, planes, hidden, n)?;
                    hst(c, 4, &mut t0)?;
                    if let (Some(dl), Some(db)) = (&a.dash, &plan.dash) {
                        // DASH: the rows rotated by their own angles and appended,
                        // the router a row, the closing blocks' summaries, the
                        // attend over the read set and the local window
                        let (dopen, dbar) = (lb[2], lb[3]);
                        c.launch(
                            fx.qk,
                            ((nh + nkv) as u32, n as u32),
                            hd as u32,
                            0,
                            kargs!(
                                plan.q.ptr, qs_i, plan.k.ptr, plan.v.ptr, kvs_i, qn, kn, plan.qg.ptr, plan.gate.ptr, kp, vp,
                                db.tab.ptr, nh_i, nkv_i, half_i, eps, qk_g, gated_i, slot0, n_i, cip, nul16, bqp, bkp, bvp
                            ),
                        )?;
                        let (blk_i, k_i, win_i) = (dl.block as i32, dl.k as i32, (dl.block * dl.local_blocks) as i32);
                        c.launch(
                            fx.dash_route,
                            (n as u32, 1),
                            256,
                            0,
                            kargs!(
                                plan.xn.ptr, hid_i, dl.w.ptr, dl.qn.ptr, dl.kn.ptr, db.tab.ptr, dopen, dbar, db.coff.ptr, db.cands.ptr,
                                db.sc.ptr, db.sel.ptr, db.bias.ptr, db.rec.ptr, n_i, blk_i, k_i, eps, dl.idx_scale, dl.floor, dl.sigma
                            ),
                        )?;
                        c.launch(fx.dash_close, (1, 1), 128, 0, kargs!(dopen, dl.hc.ptr, dbar, db.rec.ptr, n_i, blk_i))?;
                        c.launch(
                            fx.dash_attend,
                            (nh as u32, n as u32),
                            256,
                            0,
                            kargs!(
                                plan.qg.ptr, kp, vp, db.sel.ptr, db.bias.ptr, gate_p, pp0, pp1, ps_i, nul16, db.rec.ptr, cip, slot0,
                                nh_i, nkv_i, n_i, blk_i, win_i, plan.scale
                            ),
                        )?;
                        hst(c, 5, &mut t0)?;
                        prism_planes(c, plan, pp0, pp1, plan.pstride, nh * hd, n)?;
                        a.ms_out.run(c, fx, planes, nh * hd, n)?;
                        sub_out(c, fx, plan, pl, false, n)?;
                        hst(c, 6, &mut t0)?;
                    } else {
                    if far == 2 {
                        let _ = hpk_l;
                        // the ring row this key replaces into the far sums,
                        // before the key takes its place
                        let (fm_i, nkv_i2) = (fm as i32, nkv as i32);
                        c.launch(
                            fx.far_insert,
                            ((nkv * hpk_l * fm).div_ceil(8) as u32, 1),
                            256,
                            0,
                            kargs!(kp, vp, lb[3], lb[5], lb[6], cip, fm_i, nkv_i2, fscale),
                        )?;
                    }
                    c.launch(
                        fx.qk,
                        ((nh + nkv) as u32, n as u32),
                        hd as u32,
                        0,
                        kargs!(
                            plan.q.ptr, qs_i, plan.k.ptr, plan.v.ptr, kvs_i, qn, kn, plan.qg.ptr, plan.gate.ptr, kp, vp,
                            tabl, nh_i, nkv_i, half_i, eps, qk_g, gated_i, slot0, n_i, cip, nul16, bqp, bkp, bvp
                        ),
                    )?;
                    // a grid for any record: blocks past the call's keys leave
                    let nblk = ATTN_NBLK as u32;
                    c.launch_3d(
                        fx.scores,
                        (nblk, nkv as u32, n as u32),
                        256,
                        0,
                        kargs!(plan.qg.ptr, kp, s_p, mx_p, n0_i, r0, nkv_i, plan.scale, rb_i, ncap_i, nbcap_i, n_i, cip),
                    )?;
                    if far != 0 {
                        // the far part: one more block of P and E, and its maximum
                        // (each row of a verify window its own)
                        let (fm_i, frect_i) = (fm as i32, frect as i32);
                        if n == 1 {
                            c.launch(
                                fx.far_prep,
                                (nh as u32, 1),
                                256,
                                0,
                                kargs!(plan.qg.ptr, lb[2], lb[4], lb[5], lb[6], mx_p, p_p, e_p, cip, fm_i, nh_i, fscale, frect_i),
                            )?;
                        } else {
                            c.launch(
                                fx.far_prep_b,
                                (nh as u32, n as u32),
                                256,
                                0,
                                kargs!(plan.qg.ptr, lb[2], lb[4], lb[5], lb[6], mx_p, p_p, e_p, cip, fm_i, nh_i, fscale, frect_i, n_i),
                            )?;
                        }
                    }
                    let sinkp = a.sink.as_ref().map_or(0u64, |b| b.ptr);
                    c.launch_3d(
                        fx.values,
                        (nblk, nkv as u32, n as u32),
                        256,
                        0,
                        kargs!(s_p, mx_p, vp, p_p, e_p, n0_i, r0, nkv_i, rb_i, ncap_i, nbcap_i, n_i, cip, sinkp),
                    )?;
                    c.launch(
                        fx.finish,
                        (nh as u32, n as u32),
                        256,
                        0,
                        kargs!(p_p, e_p, gate_p, pp0, pp1, pp2, ps_i, n0_i, rb_i, nh_i, nbcap_i, cip),
                    )?;
                    hst(c, 5, &mut t0)?;
                    if let Some((_, tabs, blk)) = &a.hg {
                        let (nullp, no_act, zero_g) = (0u64, -1i32, 0i32);
                        for (r, jt) in tabs.iter().enumerate().take(n) {
                            let x = plan.xn.ptr + (r * hidden * 4) as u64;
                            c.launch(c.f.mv_multi, (*blk, 1), 256, super::mv_shared_bytes(hidden), kargs!(jt.ptr, 1i32, x, nullp, no_act, nullp, eps, zero_g, hid_i))?;
                        }
                        let (hd_i2, nh_i2, n_i3) = (hd as i32, nh as i32, n as i32);
                        c.launch(fx.head_gate, ((nh * hd).div_ceil(256) as u32, n as u32), 256, 0, kargs!(pp0, pp1, ps_i, plan.hgv.ptr, nh_i2, hd_i2, n_i3))?;
                    }
                    prism_planes(c, plan, pp0, pp1, plan.pstride, nh * hd, n)?;
                    a.ms_out.run(c, fx, planes, nh * hd, n)?;
                    sub_out(c, fx, plan, pl, false, n)?;
                    hst(c, 6, &mut t0)?;
                    }
                }
            }
        }
        unsafe {
            let nullp = 0u64;
            // a routed FFN also wants the normed rows in f32 (the router's input)
            let xn32 = if pl.moe.is_some() { plan.xn.ptr } else { nullp };
            sub_in(c, fx, plan, pl, true, n)?;
            c.launch(
                fx.rmsnorm2,
                (hidden.div_ceil(1024) as u32, n as u32),
                256,
                0,
                kargs!(plan.h.ptr, pl.post_norm.ptr, xn32, hid_i, eps, gemma_i, pp0, pp1, pp2, ps_i),
            )?;
            prism_planes(c, plan, pp0, pp1, plan.pstride, hidden, n)?;
            hst(c, 0, &mut t0)?;
            if let (Some(m), Some(mb)) = (&pl.moe, &plan.moe) {
                moe_routed(c, fx, plan, m, mb, n)?;
            }
            // gate/up with silu(gate) * up made in its epilogue, into the
            // middle's planes; down reads them (a routed FFN: its shared expert,
            // when it has one)
            let dense_pair = pl.moe.as_ref().is_none_or(|m| m.shared);
            if dense_pair {
                pl.ms_gu.run(c, fx, planes, hidden, n)?;
            }
            hst(c, 7, &mut t0)?;
            let mid = Planes { p0: plan.qq[0].ptr, p1: plan.qq[1].ptr, p2: plan.qq[1].ptr, stride: plan.pstride };
            if dense_pair {
                prism_planes(c, plan, mid.p0, mid.p1, mid.stride, pl.inter, n)?;
                pl.ms_dn.run(c, fx, &mid, pl.inter, n)?;
            }
            sub_out(c, fx, plan, pl, true, n)?;
            if let Some(s) = pl.layer_scale {
                let tot = (n * hidden) as u64;
                c.launch(fx.scale, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(plan.h.ptr, s, tot))?;
            }
            if let (Some(m), Some(mb)) = (&pl.moe, &plan.moe) {
                // the mixture and the shared expert into the stream
                let s_n = (n * m.k).min(m.e).min(mb.smax);
                let (gi, ni, si) = (m.gated as i32, n as i32, s_n as i32);
                let dsh = if m.shared { mb.dsh.ptr } else { 0u64 };
                c.launch(
                    fx.moe_combine,
                    (hidden.div_ceil(256) as u32, n as u32),
                    256,
                    0,
                    kargs!(plan.h.ptr, mb.dns.ptr, mb.wsel.ptr, dsh, mb.sg.ptr, gi, ni, hid_i, si),
                )?;
            }
            hst(c, 9, &mut t0)?;
            // a Looped Transformer: the final norm on the stream after a loop's last layer
            if let (true, Some(w)) = (plan.loop_norm.contains(&li), plan.loop_w.as_ref()) {
                let (nullp, zero_h) = (0u64, 0i32);
                c.launch(fx.rmsnorm, (n as u32, 1), 256, 0, kargs!(plan.h.ptr, w.ptr, plan.h.ptr, hid_i, eps, gemma_i, zero_h, nullp, nullp, nullp, zero_h))?;
            }
            // a DFlash drafter's taps: the residual stream after this layer
            for &(tl, dst) in taps {
                if tl == li {
                    cu::memcpy_dtod_async(dst, plan.h.ptr, n * hidden * 4, c.stream).map_err(derr)?;
                }
            }
        }
    }
    Ok(())
}

/// A sub-layer's product in `yo` into the residual: a sandwich norm's add, or
/// the mHC streams' merge (and, after the range's last FFN, the stream back
/// as their mean); nothing when the product went to the stream.
unsafe fn sub_out(c: &Ctx, fx: &HFuncs, plan: &HPlan, pl: &HLayer, ffn: bool, n: usize) -> Res<()> {
    match (&pl.mhc, &plan.mhc) {
        (Some(m), Some(mp)) => unsafe { mhc_out(c, fx, plan.hidden, m, mp, MhcRows::small(plan, mp), ffn, n) },
        _ => unsafe { sandwich_add(c, fx, plan, if ffn { pl.ffn_out.as_deref() } else { pl.attn_out.as_deref() }, n) },
    }
}

/// The rows an mHC sub-layer reads and writes: its input rows (`h`), its
/// product (`yo`), the streams, the merge coefficients and the router's
/// logits; the small path's plan buffers or the prompt path's own.
#[derive(Clone, Copy)]
struct MhcRows {
    h: u64,
    yo: u64,
    r: u64,
    coef: u64,
    lg: u64,
}

impl MhcRows {
    fn small(plan: &HPlan, mp: &MhcPlan) -> Self {
        MhcRows { h: plan.h.ptr, yo: plan.yo.ptr, r: mp.r.ptr, coef: mp.coef.ptr, lg: mp.lg.ptr }
    }
}

/// After an mHC sub-layer: its product merged into the streams; the range's
/// last sub-layer hands their mean on as the stream.
unsafe fn mhc_out(c: &Ctx, fx: &HFuncs, hidden: usize, m: &MhcL, mp: &MhcPlan, rows: MhcRows, ffn: bool, n: usize) -> Res<()> {
    let (ns_i, hid_i) = (mp.ns as i32, hidden as i32);
    let grid = (hidden.div_ceil(256) as u32, n as u32);
    unsafe {
        c.launch(fx.mhc_merge, grid, 256, 0, kargs!(rows.r, rows.yo, rows.coef, ns_i, hid_i))?;
        if ffn && m.leave {
            c.launch(fx.mhc_leave, grid, 256, 0, kargs!(rows.r, rows.h, ns_i, hid_i))?;
        }
    }
    Ok(())
}

/// The mHC routing before a sub-layer: entering the range (the attention of
/// its first layer) the streams take the stream's rows, then the router
/// writes the sub-layer's input rows into the stream and the merge
/// coefficients.
unsafe fn sub_in(c: &Ctx, fx: &HFuncs, plan: &HPlan, pl: &HLayer, ffn: bool, n: usize) -> Res<()> {
    let (Some(m), Some(mp)) = (&pl.mhc, &plan.mhc) else { return Ok(()) };
    unsafe { mhc_in(c, fx, plan.hidden, m, mp, MhcRows::small(plan, mp), ffn, n) }
}

/// Before an mHC sub-layer: entering the range the streams take the stream's
/// rows; the router's logits, its coefficients and the sub-layer's input rows.
unsafe fn mhc_in(c: &Ctx, fx: &HFuncs, hidden: usize, m: &MhcL, mp: &MhcPlan, rows: MhcRows, ffn: bool, n: usize) -> Res<()> {
    let (ns_i, hid_i) = (mp.ns as i32, hidden as i32);
    unsafe {
        if !ffn && m.enter {
            c.launch(fx.mhc_enter, (hidden.div_ceil(256) as u32, n as u32), 256, 0, kargs!(rows.h, rows.r, ns_i, hid_i))?;
        }
        let (w, b) = if ffn { (&m.fw, &m.fb) } else { (&m.aw, &m.ab) };
        let (out_i, it_i, fixed_i) = (m.out as i32, mp.iters as i32, (m.out == 2 * mp.ns) as i32);
        // the logits a block each, then the coefficients and the input rows
        c.launch(fx.mhc_logits, (m.out as u32, n as u32), 256, 0, kargs!(rows.r, w.ptr, rows.lg, ns_i, hid_i))?;
        c.launch(fx.mhc_route, (n as u32, 1), 256, 0, kargs!(rows.r, rows.lg, b.ptr, rows.h, rows.coef, ns_i, hid_i, out_i, it_i, fixed_i, mp.pre_scale))
    }
}

/// A sandwich norm's add: the product's rows in `yo`, normed with `w`, onto
/// the stream (nothing without a norm: the product went to the stream).
unsafe fn sandwich_add(c: &Ctx, fx: &HFuncs, plan: &HPlan, w: Option<&Buf>, n: usize) -> Res<()> {
    let Some(w) = w else { return Ok(()) };
    let (hi, gi) = (plan.hidden as i32, plan.gemma as i32);
    unsafe { c.launch(fx.norm_add, (n as u32, 1), 256, 0, kargs!(plan.h.ptr, plan.yo.ptr, w.ptr, hi, plan.eps, gi)) }
}

/// A routed FFN's experts for the call's `n` rows: the router's logits (and
/// the shared gate), the picks and their union as slots, every slot's gate/up
/// and down on the tensor cores. The mixture is added by `hy_moe_combine`.
unsafe fn moe_routed(c: &Ctx, fx: &HFuncs, plan: &HPlan, m: &MoeL, mb: &MoeBufs, n: usize) -> Res<()> {
    let hidden = plan.hidden;
    let s_n = (n * m.k).min(m.e).min(mb.smax);
    let (ni, ei, hi, ki, fl, si) = (n as i32, m.e as i32, hidden as i32, m.k as i32, m.flags, s_n as i32);
    let sgw = m.sgw.as_ref().map_or(0, |d| d.w.ptr);
    let bias = m.bias.as_ref().map_or(0, |b| b.ptr);
    let (ps_i, inter_i) = (plan.pstride as i32, m.inter as i32);
    // two tiles a block past one row: every block reads the whole input, and a
    // window's slots are many (CMF_HYB_MOE_TB=1 keeps one)
    let tb = if n > 1 && std::env::var("CMF_HYB_MOE_TB").as_deref() != Ok("1") { 2u32 } else { 1u32 };
    let (bpg, bpd) = (m.bpj_gu.div_ceil(tb), m.bpj_dn.div_ceil(tb));
    let (bg, bd) = (bpg as i32, bpd as i32);
    unsafe {
        c.launch(fx.moe_router, ((m.e + 1).div_ceil(8) as u32, n as u32), 256, 0, kargs!(plan.xn.ptr, m.router.w.ptr, sgw, mb.logits.ptr, mb.sg.ptr, ni, ei, hi))?;
        c.launch(
            fx.moe_select,
            (1, 1),
            (32 * n.max(1)) as u32,
            0,
            kargs!(
                mb.logits.ptr, bias, ni, ei, ki, fl, m.route_scale, m.gu_t.ptr, m.dn_t.ptr, mb.gu_a.ptr, mb.dn_a.ptr, mb.wsel.ptr, si,
                mb.qq[0].ptr, mb.qq[1].ptr, ps_i, mb.dns.ptr, hi, bg, bd
            ),
        )?;
        let (pp0, pp1, pp2) = (plan.pp[0].ptr, plan.pp[1].ptr, plan.pp[2].ptr);
        let t = (tb - 1) as usize;
        // experts of one codec: that codec's launch
        let pick = |kd: Option<usize>| match kd {
            Some(d) => fx.mmpk[nt_of(n)][t][1][d],
            None => fx.mmp[nt_of(n)][t][1],
        };
        // every slot's gate/up from the normed rows
        c.launch(pick(m.kd.0), (s_n as u32 * bpg, 1), 32 * 16, 0, kargs!(mb.gu_a.ptr, si, pp0, pp1, pp2, ps_i, ni, hi))?;
        // every slot's down from its own planes (the job's `xoff`)
        let (q0, q1) = (mb.qq[0].ptr, mb.qq[1].ptr);
        c.launch(pick(m.kd.1), (s_n as u32 * bpd, 1), 32 * 16, 0, kargs!(mb.dn_a.ptr, si, q0, q1, q1, ps_i, ni, inter_i))?;
    }
    Ok(())
}

/// A layer the prompt path has no kernels for (a routed FFN's sandwich, mHC
/// beside a sandwich or a routed FFN, a short convolution or a head-wise
/// gate): such a stack's long prompt runs the small path a chunk at a time.
fn small_only(l: &HLayer) -> bool {
    // sandwich norms and a layer scale run on the prompt path too, a routed
    // FFN's sandwich aside
    (l.moe.is_some() && l.ffn_out.is_some())
        || (l.mhc.is_some() && (l.moe.is_some() || l.attn_out.is_some() || l.ffn_out.is_some()))
        || matches!(l.mix, Mix::Conv(_))
        || matches!(&l.mix, Mix::Attn(a) if a.hg.is_some())
}

/// `CMF_HYB_MOE_BIG=0`: a routed FFN's long prompt goes the small path
/// NMAX rows at a time instead of the prompt path's grouped experts.
fn moe_big_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_HYB_MOE_BIG").as_deref() != Ok("0"))
}

/// Rows a grouped expert job takes (`CMF_HYB_MOE_RMAX`: 32 or 64).
fn moe_rmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| if std::env::var("CMF_HYB_MOE_RMAX").as_deref() == Ok("64") { 64 } else { 32 })
}

/// A replayable small-path call: the instantiated graph and the pointers it
/// was recorded with.
pub(super) struct GraphRec {
    exec: cs::CUgraphExec,
    sig: Vec<u64>,
}
unsafe impl Send for GraphRec {}

impl Drop for GraphRec {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cuGraphExecDestroy(self.exec);
        }
    }
}

fn graphs_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_CUDA_GRAPH").as_deref() != Ok("0")) && !super::prof_on() && !KT_TOKEN.with(|k| k.get())
}

thread_local! {
    /// This token's kernels are timed (`CMF_HYB_KT=N`): launched one by one.
    static KT_TOKEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn curesult(r: cs::CUresult, what: &str) -> Res<()> {
    if r == cs::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(format!("{what}: {r:?}")) }
}

/// Record `f`'s launches on the stream into an instantiated graph.
unsafe fn capture(c: &Ctx, f: impl FnOnce() -> Res<()>) -> Res<cs::CUgraphExec> {
    unsafe {
        curesult(
            cs::cuStreamBeginCapture_v2(c.stream, cs::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
            "begin capture",
        )?;
        let r = f();
        let mut g: cs::CUgraph = std::ptr::null_mut();
        let e = cs::cuStreamEndCapture(c.stream, &mut g);
        r?;
        curesult(e, "end capture")?;
        let mut ex: cs::CUgraphExec = std::ptr::null_mut();
        let ie = cs::cuGraphInstantiateWithFlags(&mut ex, g, 0);
        let _ = cs::cuGraphDestroy(g);
        curesult(ie, "graph instantiate")?;
        Ok(ex)
    }
}

thread_local! {
    /// Set while a long prompt goes through the small path a chunk at a time.
    static DIRECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A DASH stack's call lists for rows [r0, r0 + n) of the call (its first
/// row at `pos`): the candidates (the call's), those rows' offsets into them,
/// their rotation table (sin and cos of the rows' angles, `hy_qk`'s rows) and
/// the record (first position, the router's block count). Synchronous: the
/// lists are the caller's.
fn dash_upload(plan: &HPlan, layers: &[GraphLayer], pos: usize, r0: usize, n: usize) -> Res<()> {
    let Some(db) = &plan.dash else { return Ok(()) };
    let d = layers
        .iter()
        .find_map(|l| match &l.attn {
            GraphAttn::Full { geom: Some(g), .. } => g.dash.as_ref(),
            _ => None,
        })
        .ok_or("DASH layers without their router")?;
    if n == 0
        || n > DASH_ROWS
        || d.coff.len() < r0 + n + 1
        || d.cands.len() > DASH_ROWS * DASH_BLOCKS
        || d.angles.len() < (r0 + n) * 64
        || d.nblk > DASH_BLOCKS
        || (pos + r0 + n) / d.block >= DASH_BLOCKS
    {
        return Err(format!("DASH lists of {n} rows past the kernels ({} candidates, {} blocks)", d.cands.len(), d.nblk));
    }
    let tab: Vec<f32> = d.angles[r0 * 64..(r0 + n) * 64].iter().flat_map(|&a| [a.sin(), a.cos()]).collect();
    let rec = [(pos + r0) as i32, d.nblk as i32, 0, 0];
    unsafe {
        if !d.cands.is_empty() {
            super::htod(db.cands.ptr, d.cands)?;
        }
        super::htod(db.coff.ptr, &d.coff[r0..r0 + n + 1])?;
        super::htod(db.tab.ptr, &tab)?;
        super::htod(db.rec.ptr, &rec)
    }
}

/// The small path, with the head when `head`: book, then the launches —
/// replayed from a recorded graph when one with the same pointers is there,
/// else recorded now (or launched directly, `CMF_CUDA_GRAPH=0` and under
/// `CMF_CUDA_PROF=1`). The rows in `plan.h` and the RoPE table in
/// `plan.tab` must be on the stream already.
#[allow(clippy::too_many_arguments)]
unsafe fn run_small(
    c: &Ctx,
    st: &mut State,
    plan: &HPlan,
    kv_id: u64,
    layer_base: usize,
    pos0: usize,
    n: usize,
    tape: bool,
    head: bool,
    kcap: usize,
    sealed: &[bool],
    taps: &[(usize, u64)],
) -> Res<usize> {
    let bk = book(c, st, plan, kv_id, layer_base, pos0, n, tape, kcap, sealed)?;
    let rows = plan.lm.as_ref().map_or(0, |l| l.rows());
    unsafe {
        if !bk.ci.is_empty() {
            cu::memcpy_htod_async(plan.ci.ptr, &bk.ci, c.stream).map_err(derr)?;
        }
        // a prompt read through this path a chunk at a time launches directly:
        // capturing a graph per chunk shape cost more than the chunk
        if !graphs_on() || DIRECT.with(|d| d.get()) {
            launch(c, plan, &bk, n, tape, plan.ci.ptr, plan.tab.ptr, taps)?;
            if head {
                head_launch(c, plan, n)?;
            }
            return Ok(if head { rows } else { 0 });
        }
        let mut far_key = 0u64;
        for (i, &f) in bk.far.iter().enumerate() {
            far_key ^= (f as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(i as u32 % 64);
        }
        // a call with taps records copies a call without them has not
        for &(tl, _) in taps {
            far_key ^= ((tl as u64 + 1) << 40).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        }
        let key = (plan.id, kv_id, layer_base, n, tape, head, far_key);
        let mut sig: Vec<u64> = bk.layers.iter().flatten().copied().collect();
        sig.extend(bk.scratch);
        sig.extend(bk.far.iter().map(|&f| f as u64));
        sig.extend(bk.farp.iter().map(|&(m, sc, r)| (m as u64) << 33 | (sc.to_bits() as u64) << 1 | r as u64));
        sig.extend(taps.iter().flat_map(|&(tl, dst)| [tl as u64, dst]));
        let hit = st.hyb.graphs.get(&key).is_some_and(|g| g.sig == sig);
        GRAPH_RUN.fetch_add(1, Ordering::Relaxed);
        if !hit {
            GRAPH_REC.fetch_add(1, Ordering::Relaxed);
            if debug() {
                let why = match st.hyb.graphs.get(&key) {
                    None => "new key".to_string(),
                    Some(g) => format!("{} of {} pointers moved", g.sig.iter().zip(&sig).filter(|(a, b)| a != b).count(), sig.len()),
                };
                eprintln!("cuda hybrid: recording a graph for {n} rows (tape {tape}, head {head}): {why}");
            }
            let exec = capture(c, || {
                launch(c, plan, &bk, n, tape, plan.ci.ptr, plan.tab.ptr, taps)?;
                if head {
                    head_launch(c, plan, n)?;
                }
                Ok(())
            })?;
            st.hyb.graphs.insert(key, GraphRec { exec, sig });
        }
        let g = &st.hyb.graphs[&key];
        curesult(cs::cuGraphLaunch(g.exec, c.stream), "graph launch")?;
    }
    Ok(if head { rows } else { 0 })
}

/// The DFlash taps a small-path call keeps (`super::layer_taps_arm`): for
/// every asked layer of a call over the whole stack, its plan index and a
/// device slot of `NMAX` rows. None when nothing was asked or the call does
/// not reach every asked layer.
fn tap_slots(c: &Ctx, st: &mut State, nlayers: usize, layer_base: usize, hidden: usize) -> Res<Option<(Vec<(usize, u64)>, u64)>> {
    let want = super::taps_wanted();
    if want.is_empty() || layer_base != 0 || want.iter().any(|&l| l >= nlayers) {
        return Ok(None);
    }
    let buf = st.scr(c, "hy_taps", want.len() * NMAX * hidden * 4)?;
    let slots = want.iter().enumerate().map(|(s, &l)| (l, buf + (s * NMAX * hidden * 4) as u64)).collect();
    Ok(Some((slots, buf)))
}

/// The kept tap rows (`n` a slot) to the host as `[slot][row][hidden]`, for
/// `super::layer_taps_take` once the stream has run.
unsafe fn taps_read(c: &Ctx, slots: &[(usize, u64)], n: usize, hidden: usize) -> Res<Vec<f32>> {
    let mut raw = vec![0f32; slots.len() * n * hidden];
    for (s, &(_, dst)) in slots.iter().enumerate() {
        unsafe { cu::memcpy_dtoh_async(&mut raw[s * n * hidden..(s + 1) * n * hidden], dst, c.stream).map_err(derr)? };
    }
    Ok(raw)
}

/// The head over the rows in `plan.h`: logits [n][rows] into `plan.logits`.
unsafe fn head_launch(c: &Ctx, plan: &HPlan, n: usize) -> Res<usize> {
    let Some(head) = &plan.lm else { return Ok(0) };
    let fx = &plan.fx;
    let (hid_i, eps, gemma_i) = (plan.hidden as i32, plan.eps, plan.gemma as i32);
    let fnorm = plan.final_norm.as_ref().unwrap().ptr;
    let mut t0 = std::time::Instant::now();
    let (rows, js) = match head {
        Head::Mma(_, rows, js) => (rows, js),
        Head::Dense(_, rows, jobs, blk) => {
            let (nullp, no_act) = (0u64, -1i32);
            for (r, jt) in jobs.iter().enumerate().take(n) {
                let x = plan.h.ptr + (r * plan.hidden * 4) as u64;
                unsafe {
                    c.launch(c.f.mv_multi, (*blk, 1), 256, super::mv_shared_bytes(plan.hidden), kargs!(jt.ptr, 1i32, x, nullp, no_act, fnorm, eps, gemma_i, hid_i))?;
                }
            }
            hst(c, 10, &mut t0)?;
            return Ok(*rows);
        }
    };
    let planes = Planes { p0: plan.pp[0].ptr, p1: plan.pp[1].ptr, p2: plan.pp[2].ptr, stride: plan.pstride };
    let (pp0, pp1, pp2, ps_i, nullp) = (planes.p0, planes.p1, planes.p2, plan.pstride as i32, 0u64);
    unsafe {
        c.launch(
            fx.rmsnorm2,
            (plan.hidden.div_ceil(1024) as u32, n as u32),
            256,
            0,
            kargs!(plan.h.ptr, fnorm, nullp, hid_i, eps, gemma_i, pp0, pp1, pp2, ps_i),
        )?;
        prism_planes(c, plan, pp0, pp1, plan.pstride, plan.hidden, n)?;
        js.run(c, fx, &planes, plan.hidden, n)?;
    }
    hst(c, 10, &mut t0)?;
    Ok(*rows)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn token(
    c: &'static Ctx,
    model: &Arc<CmfModel>,
    kv_id: u64,
    layers: &[GraphLayer],
    o1: &[Option<Vec<crate::nystrom::O1DeviceView<'_>>>],
    o1_epoch: u64,
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    attn_scale: f32,
    rd: usize,
    hidden: usize,
    position: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    lm_head: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    logits: &mut Vec<f32>,
    loop_norm_at: &[usize],
    steps: usize,
    layers_run: Option<&mut usize>,
    layer_base: usize,
    hidden_too: bool,
) -> TokenGraphOutcome {
    if previewing(layers) {
        decline("hybrid token", "an active block preview");
        return TokenGraphOutcome::Declined;
    }
    if steps != 1 || (!o1.is_empty() && o1.len() != layers.len()) {
        decline("hybrid token", "a multi-step burst or far-field views short of the layers");
        return TokenGraphOutcome::Declined;
    }
    let sealed: Vec<bool> = (0..layers.len()).map(|li| o1.get(li).is_some_and(Option::is_some)).collect();
    if position >= cap || h.len() != hidden {
        decline("hybrid token", "position past the context cap");
        return TokenGraphOutcome::Declined;
    }
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return TokenGraphOutcome::Declined;
    }
    let plan = match plan_for(c, &mut st, model, layers, layer_base, lm_head, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, attn_scale, invf, position, loop_norm_at) {
        Ok(p) => p,
        Err(e) => {
            decline("hybrid token", &e);
            return TokenGraphOutcome::Declined;
        }
    };
    // the far fields first: a state the card has moved has no host copy to
    // go back to, so from there a refusal is a failure
    let moved = sealed.iter().enumerate().any(|(li, &s)| s && st.hyb.o1.get(&(kv_id, layer_base + li)).is_some_and(|d| d.next_pos.is_some()));
    for (li, views) in o1.iter().enumerate() {
        let Some(views) = views else { continue };
        let key = (kv_id, layer_base + li);
        let at = st.hyb.o1.get(&key).filter(|d| d.epoch == o1_epoch).and_then(|d| d.next_pos);
        let res = match at {
            Some(next) if next != position => Err(format!("far field: the card is at row {next}, the call at {position}")),
            _ => o1_ensure(c, &mut st, key, views, o1_epoch, plan.hpk, nkv, hd),
        };
        if let Err(e) = res {
            decline("hybrid token", &e);
            return if moved { TokenGraphOutcome::Failed } else { TokenGraphOutcome::Declined };
        }
        // a sealed layer reads no mirror again
        if let Some(m) = st.kv.remove(&key) {
            let _ = c.sync();
            c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
        }
    }
    if let Err((hard, e)) = admit(c, &mut st, &plan, layers, kv_id, layer_base, position, 1, 0, &sealed) {
        decline("hybrid token", &e);
        return if hard || moved { TokenGraphOutcome::Failed } else { TokenGraphOutcome::Declined };
    }
    let t0 = std::time::Instant::now();
    // `CMF_HYB_KT=N`: the kernel times of the N-th token
    static TOKENS_SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let kt = {
        let n = TOKENS_SEEN.fetch_add(1, Ordering::Relaxed);
        std::env::var("CMF_HYB_KT").ok().and_then(|v| v.parse::<usize>().ok()) == Some(n)
    };
    if kt {
        KT_TOKEN.with(|k| k.set(true));
        super::kt_start();
    }
    let mut run = || -> Res<()> {
        unsafe {
            cu::memcpy_htod_async(plan.h.ptr, h, c.stream).map_err(derr)?;
            let tab = plan_tab(&plan, std::iter::once(position));
            if !tab.is_empty() {
                cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
            }
            let taps = tap_slots(c, &mut st, layers.len(), layer_base, hidden)?;
            let tl = taps.as_ref().map_or(&[][..], |t| &t.0[..]);
            dash_upload(&plan, layers, position, 0, 1)?;
            let rows = run_small(c, &mut st, &plan, kv_id, layer_base, position, 1, false, plan.lm.is_some(), cap, &sealed, tl)?;
            if taps.is_some() {
                // a pageable read: the rows are there when it returns
                super::taps_put(taps_read(c, tl, 1, hidden)?);
            }
            if rows > 0 {
                logits.resize(rows, 0.0);
                cu::memcpy_dtoh_async(logits.as_mut_slice(), plan.logits.ptr, c.stream).map_err(derr)?;
                if hidden_too {
                    cu::memcpy_dtoh_async(h, plan.h.ptr, c.stream).map_err(derr)?;
                }
            } else {
                cu::memcpy_dtoh_async(h, plan.h.ptr, c.stream).map_err(derr)?;
            }
        }
        c.sync()
    };
    let res = run();
    if kt {
        super::kt_stop();
        KT_TOKEN.with(|k| k.set(false));
    }
    match res {
        Ok(()) => {
            if let Some(lr) = layers_run {
                *lr = layers.len();
            }
            if debug() {
                super::TOKEN_US.lock().unwrap().push(t0.elapsed().as_micros() as u32);
            }
            if super::TOKENS.fetch_add(1, Ordering::Relaxed) >= 16 {
                super::TOKEN_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            if position % 256 == 0 {
                mem_report(c, &st, &format!("a token at {position}"));
            }
            TokenGraphOutcome::Completed
        }
        Err(e) => {
            tracing::error!("cuda hybrid token graph failed: {e}");
            let _ = c.sync();
            TokenGraphOutcome::Failed
        }
    }
}

/// The round's `k` greedy drafts of an MTP block in ONE graph: per step,
/// the block input (the token's embedding and the hidden row, each RMS-normed,
/// through eh_proj), the block over its K/V mirror at `position + j`, the head
/// over the shortlist and its argmax, whose token the next step's embedding
/// is gathered for on the card. One upload, one launch, the k ids back.
/// `e0` is the embedding of `t0` (the host's); `embed_row(id, out)` fills a
/// shortlist row the first time the table is built. Err(false): declined
/// before any state changed; Err(true): failed after the mirror moved.
#[allow(clippy::too_many_arguments)]
pub(super) fn mtp_chain(
    c: &'static Ctx,
    model: &Arc<CmfModel>,
    kv_id: u64,
    layer: &GraphLayer,
    invf: &[f32],
    h0: &[f32],
    e0: &[f32],
    t0: u32,
    k: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    attn_scale: f32,
    rd: usize,
    hidden: usize,
    position: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    lm_head: (&GraphW, usize),
    final_norm: &[f32],
    eh: &GraphW,
    enorm: &[f32],
    hnorm: &[f32],
    embed: &GraphW,
    layer_base: usize,
) -> Result<Vec<u32>, bool> {
    let rows = lm_head.1;
    if k == 0 || k > NMAX || h0.len() != hidden || e0.len() != hidden || enorm.len() != hidden || hnorm.len() != hidden {
        return Err(false);
    }
    if position + k >= cap || eh.kind != 6 || embed.kind != 6 || embed.prism != crate::gpu::GraphPrismOp::None || hidden % 1024 != 0 {
        return Err(false);
    }
    let layers = std::slice::from_ref(layer);
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return Err(false);
    }
    let st = &mut *st;
    let plan = match plan_for(c, st, model, layers, layer_base, Some(lm_head), final_norm, nh, nkv, hd, rd, hidden, gemma, eps, attn_scale, invf, position, &[]) {
        Ok(p) => p,
        Err(e) => {
            decline("hybrid draft chain", &e);
            return Err(false);
        }
    };
    if plan.pstride < 2 * hidden || plan.layers.iter().any(|l| !matches!(l.mix, Mix::Attn(_))) {
        decline("hybrid draft chain", "planes narrower than the block input, or a GDN layer");
        return Err(false);
    }
    // the table, the norms and eh_proj's job tables (built once)
    let key = (model.uid(), embed.idx, rows, crate::gpu::fp_f32(enorm), crate::gpu::fp_f32(hnorm));
    let prep = |st: &mut State| -> Res<()> {
        if st.hyb.chain.as_ref().is_none_or(|r| r.key != key) {
            if let Some(old) = st.hyb.chain.take() {
                c.resident.fetch_sub(old.emb.held, Ordering::Relaxed);
            }
            use cortiq_core::quant::q4tp_sections;
            let entry = model.tensors.get(embed.idx).ok_or("embedding index out of range")?;
            let erows = *entry.shape.first().ok_or("embedding without a shape")?;
            if entry.shape.get(1) != Some(&hidden) || rows > erows || rows % 16 != 0 {
                return Err(format!("embedding {:?} against a {rows}-row shortlist of width {hidden}", entry.shape));
            }
            // the first `rows` rows: each section's leading rows, from the file
            let bytes = model.entry_bytes(entry);
            let (po, co, stride) = q4tp_sections(erows, hidden);
            if bytes.len() < co + erows * stride {
                return Err("embedding payload shorter than its planes".into());
            }
            let fx = st.hyb.funcs.values().next().cloned().ok_or("repack before the hybrid kernels")?;
            let t0 = std::time::Instant::now();
            let (w, aux0, aux1) = repack_rows_on_card(c, &fx, &bytes[..po], &bytes[po..co], &bytes[co..], stride, rows, hidden)?;
            if debug() {
                eprintln!("cuda hybrid draft chain: {rows} embedding rows on the card in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            let held = (rows * (hidden / 32) * 16 + rows * (4 + stride)) as u64;
            let emb = DevMat { kind: 6, rows, cols: hidden, w: Arc::new(w), aux0: Some(aux0), aux1: Some(aux1), held };
            c.resident.fetch_add(emb.held, Ordering::Relaxed);
            st.hyb.chain = Some(ChainRes { key, emb, enorm: Buf::from_slice(enorm)?, hnorm: Buf::from_slice(hnorm)? });
        }
        if !st.hyb.chain_sets.contains_key(&(plan.id, eh.idx)) {
            let m = hmat(c, st, model, eh, hidden, 2 * hidden)?;
            let ms = mmaset(&[(&m, plan.h.ptr, 0, hidden, hidden)])?;
            st.hyb.chain_sets.insert((plan.id, eh.idx), ms);
        }
        Ok(())
    };
    if !plan.invf_local.is_empty() || plan.own_geom || plan.mhc.is_some() {
        decline("hybrid draft chain", "a stack with local rotations or a layer head geometry of its own");
        return Err(false);
    }
    if let Err(e) = prep(st) {
        decline("hybrid draft chain", &e);
        return Err(false);
    }
    if let Err((hard, e)) = admit(c, st, &plan, layers, kv_id, layer_base, position, k, 0, &[false]) {
        decline("hybrid draft chain", &e);
        return Err(hard);
    }
    // from here the mirror moves: a failure is terminal
    let run = |st: &mut State| -> Res<Vec<u32>> {
        let mut books = Vec::with_capacity(k);
        for j in 0..k {
            books.push(book(c, st, &plan, kv_id, layer_base, position + j, 1, false, cap, &[false])?);
        }
        let cilen = books[0].ci.len();
        let tabrow = plan.invf.len() * 2;
        let ci_b = st.scr(c, "hy_chain_ci", NMAX * cilen.max(1) * 4)?;
        let tab_b = st.scr(c, "hy_chain_tab", NMAX * tabrow.max(1) * 4)?;
        let e_b = st.scr(c, "hy_chain_e", NMAX * hidden * 4)?;
        let h0_b = st.scr(c, "hy_chain_h0", hidden * 4)?;
        let tok_b = st.scr(c, "hy_chain_tok", (NMAX + 1) * 4)?;
        let ci: Vec<i32> = books.iter().flat_map(|b| b.ci.iter().copied()).collect();
        // a launch a position here: the rows of one table (a stack with local
        // layers is refused before the mirror moves)
        let tab = rope_rows(&plan.invf, position..position + k);
        let ch = st.hyb.chain.as_ref().unwrap();
        let (etab, ecod, epar, enp, hnp) = (ch.emb.w.ptr, ch.emb.aux0(), ch.emb.aux1.as_ref().map_or(0, |b| b.ptr), ch.enorm.ptr, ch.hnorm.ptr);
        let tok0 = [t0 as i32];
        unsafe {
            if !ci.is_empty() {
                cu::memcpy_htod_async(ci_b, &ci, c.stream).map_err(derr)?;
            }
            if !tab.is_empty() {
                cu::memcpy_htod_async(tab_b, &tab, c.stream).map_err(derr)?;
            }
            cu::memcpy_htod_async(e_b, e0, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(h0_b, h0, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(tok_b, &tok0, c.stream).map_err(derr)?;
        }
        let ms = &st.hyb.chain_sets[&(plan.id, eh.idx)];
        let fx = &plan.fx;
        let (hid_i, gemma_i, rows_i) = (hidden as i32, gemma as i32, rows as i32);
        let ps_i = plan.pstride as i32;
        let (p0, p1, p2) = (plan.pp[0].ptr, plan.pp[1].ptr, plan.pp[2].ptr);
        let off = (hidden * 2) as u64;
        let planes = Planes { p0, p1, p2, stride: plan.pstride };
        let body = || -> Res<()> {
            let nullp = 0u64;
            for (j, bk) in books.iter().enumerate() {
                let ej = e_b + (j * hidden * 4) as u64;
                let hsrc = if j == 0 { h0_b } else { plan.h.ptr };
                let (j_i, j1_i) = (j as i32, j as i32 + 1);
                let (q0, q1, q2) = (p0 + off, p1 + off, p2 + off);
                unsafe {
                    if j > 0 {
                        c.launch(fx.gather, (hidden.div_ceil(256) as u32, 1), 256, 0, kargs!(etab, ecod, epar, rows_i, hid_i, tok_b, j_i, ej))?;
                    }
                    c.launch(fx.rmsnorm2, (hidden.div_ceil(1024) as u32, 1), 256, 0, kargs!(ej, enp, nullp, hid_i, eps, gemma_i, p0, p1, p2, ps_i))?;
                    c.launch(fx.rmsnorm2, (hidden.div_ceil(1024) as u32, 1), 256, 0, kargs!(hsrc, hnp, nullp, hid_i, eps, gemma_i, q0, q1, q2, ps_i))?;
                    ms.run(c, fx, &planes, 2 * hidden, 1)?;
                    launch(c, &plan, bk, 1, false, ci_b + (j * cilen * 4) as u64, tab_b + (j * tabrow * 4) as u64, &[])?;
                    head_launch(c, &plan, 1)?;
                    c.launch(fx.argmax, (1, 1), 1024, 0, kargs!(plan.logits.ptr, rows_i, rows_i, tok_b, j1_i))?;
                }
            }
            Ok(())
        };
        unsafe {
            if graphs_on() {
                let gkey = (plan.id, kv_id, layer_base, k, true, true, 0xC4A1_0000u64);
                let mut sig: Vec<u64> = books[0].layers.iter().flatten().copied().collect();
                sig.extend(books[0].scratch);
                sig.extend([etab, ecod, epar, enp, hnp, ci_b, tab_b, e_b, h0_b, tok_b, ms.bufs[0].ptr, ms.bufs[1].ptr]);
                if st.hyb.graphs.get(&gkey).is_none_or(|g| g.sig != sig) {
                    GRAPH_REC.fetch_add(1, Ordering::Relaxed);
                    let exec = capture(c, body)?;
                    st.hyb.graphs.insert(gkey, GraphRec { exec, sig });
                }
                GRAPH_RUN.fetch_add(1, Ordering::Relaxed);
                curesult(cs::cuGraphLaunch(st.hyb.graphs[&gkey].exec, c.stream), "graph launch")?;
            } else {
                body()?;
            }
            let mut ids = vec![0i32; k + 1];
            cu::memcpy_dtoh_async(ids.as_mut_slice(), tok_b, c.stream).map_err(derr)?;
            c.sync()?;
            Ok(ids[1..].iter().map(|&t| t as u32).collect())
        }
    };
    match run(st) {
        Ok(ids) => Ok(ids),
        Err(e) => {
            tracing::error!("cuda hybrid draft chain failed: {e}");
            let _ = c.sync();
            Err(true)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch(
    c: &'static Ctx,
    model: &Arc<CmfModel>,
    kv_id: u64,
    layer_base: usize,
    layers: &[GraphLayer],
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    positions: &[usize],
    cap: usize,
    gemma: bool,
    eps: f32,
    attn_scale: f32,
    k: usize,
    o1: &[Option<Vec<crate::nystrom::O1DeviceView<'_>>>],
    o1_epoch: u64,
    spec: Option<SpecTail<'_>>,
    layers_run: Option<&mut usize>,
    defer: bool,
) -> BatchGraphOutcome {
    let b = k;
    if previewing(layers) {
        decline("hybrid batch", "an active block preview");
        return BatchGraphOutcome::Declined;
    }
    // a sealed layer takes a verify window over its far field (the state on
    // the card at this position), or a prompt chunk read past the seal
    let sealed: Vec<bool> = (0..layers.len()).map(|li| o1.get(li).is_some_and(Option::is_some)).collect();
    let any_sealed = sealed.iter().any(|&s| s);
    let prompt_far = any_sealed && spec.is_none();
    if any_sealed && spec.is_some() && (b > NMAX || !far_spec_on()) {
        decline("hybrid batch", "a sealed far field under a verify window longer than the small path");
        return BatchGraphOutcome::Declined;
    }
    if prompt_far && !far_chunks_on() {
        decline("hybrid batch", "a sealed far field under a prompt chunk (CMF_O1_CHUNKS=0)");
        return BatchGraphOutcome::Declined;
    }
    if b == 0
        || positions.len() != b
        || h.len() != b * hidden
        || positions.windows(2).any(|w| w[1] != w[0] + 1)
        || positions[b - 1] >= cap
    {
        decline("hybrid batch", "rows not contiguous or past the context cap");
        return BatchGraphOutcome::Declined;
    }
    if spec.as_ref().is_some_and(|s| s.pick.is_some()) || (spec.is_some() && b > NMAX) {
        decline("hybrid batch", "a verify window longer than the small path, or a device pick");
        return BatchGraphOutcome::Declined;
    }
    let pos0 = positions[0];
    let t_all = std::time::Instant::now();
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return BatchGraphOutcome::Declined;
    }
    let lm = spec.as_ref().map(|s| (&s.lm, s.lm_rows));
    // a Looped Transformer's loop norms (the final norm, as the head's)
    let looped = crate::gpu::batch_loop_norm();
    let loop_at: &[usize] = looped.as_ref().map_or(&[], |(at, _)| at.as_slice());
    let final_norm: &[f32] = match (&spec, &looped) {
        (Some(s), _) => s.final_norm,
        (None, Some((_, w))) => w.as_slice(),
        (None, None) => &[],
    };
    let plan = match plan_for(c, &mut st, model, layers, layer_base, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, attn_scale, invf, pos0, loop_at) {
        Ok(p) => p,
        Err(e) => {
            decline("hybrid batch", &e);
            return BatchGraphOutcome::Declined;
        }
    };
    // a routed FFN runs on the small path only (its rows' experts as slots):
    // a longer prompt goes through it NMAX rows at a time
    // what the prompt path has no kernels for (a routed FFN, sandwich norms, a
    // layer scale) runs the small path
    // a routed FFN's long prompt takes the prompt path (its experts grouped)
    // unless the stack has a layer only the small path runs
    let grouped = moe_big_on() && spec.is_none() && !prompt_far && b > NMAX && !plan.layers.iter().any(small_only);
    let moe = !grouped && plan.layers.iter().any(|l| l.moe.is_some() || small_only(l));
    if spec.is_some() && plan.layers.iter().any(|l| matches!(l.mix, Mix::Conv(_))) {
        decline("hybrid batch", "a verify window over a short convolution");
        return BatchGraphOutcome::Declined;
    }
    // CMF_HYB_MOE_CHUNKS=0: such a long prompt declines (the per-op path)
    let chunks_on = std::env::var("CMF_HYB_MOE_CHUNKS").as_deref() != Ok("0");
    if moe && (prompt_far || (b > NMAX && (spec.is_some() || !chunks_on))) {
        decline("hybrid batch", "a routed FFN past a seal or in a long verify window");
        return BatchGraphOutcome::Declined;
    }
    if prompt_far {
        // a prompt chunk past a seal: the epoch's far field goes up when it
        // is new (its rows are this call's first position), and the layer's
        // exact mirror goes
        let moved = sealed.iter().enumerate().any(|(li, &s)| s && st.hyb.o1.get(&(kv_id, layer_base + li)).is_some_and(|d| d.epoch == o1_epoch && d.next_pos.is_some()));
        for (li, views) in o1.iter().enumerate() {
            let Some(views) = views else { continue };
            let key = (kv_id, layer_base + li);
            let at = st.hyb.o1.get(&key).filter(|d| d.epoch == o1_epoch).map(|d| d.next_pos);
            let rows = views.first().map_or(0, |v| v.sink_len + v.win_len + v.far_len);
            let res = match at {
                Some(Some(next)) if next != pos0 => Err(format!("far field: the card is at row {next}, the call at {pos0}")),
                Some(Some(_)) => Ok(()),
                _ if rows != pos0 => Err(format!("far field: the seal holds {rows} rows, the call is at {pos0}")),
                _ => o1_ensure(c, &mut st, key, views, o1_epoch, plan.hpk, nkv, hd),
            };
            if let Err(e) = res {
                decline("hybrid batch", &e);
                return if moved { BatchGraphOutcome::Failed } else { BatchGraphOutcome::Declined };
            }
            if let Some(m) = st.kv.remove(&key) {
                let _ = c.sync();
                c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
            }
        }
    } else if any_sealed {
        for (li, &s) in sealed.iter().enumerate() {
            let at = st.hyb.o1.get(&(kv_id, layer_base + li)).and_then(|d| d.next_pos);
            if s && at != Some(pos0) {
                decline("hybrid batch", "a far field the card has not moved to this row");
                return BatchGraphOutcome::Declined;
            }
        }
    }
    let hint = st.kv_hint;
    if let Err((hard, e)) = admit(c, &mut st, &plan, layers, kv_id, layer_base, pos0, b, hint, &sealed) {
        decline("hybrid batch", &e);
        return if hard { BatchGraphOutcome::Failed } else { BatchGraphOutcome::Declined };
    }
    let mut spec = spec;
    let mut run = || -> Res<()> {
        if moe && b > NMAX {
            struct Direct;
            impl Drop for Direct {
                fn drop(&mut self) {
                    DIRECT.with(|d| d.set(false));
                }
            }
            DIRECT.with(|d| d.set(true));
            let _direct = Direct;
            let mut at = 0usize;
            while at < b {
                let n = NMAX.min(b - at);
                unsafe {
                    cu::memcpy_htod_async(plan.h.ptr, &h[at * hidden..(at + n) * hidden], c.stream).map_err(derr)?;
                    let tab = plan_tab(&plan, positions[at..at + n].iter().copied());
                    if !tab.is_empty() {
                        cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
                    }
                    dash_upload(&plan, layers, pos0, at, n)?;
                    run_small(c, &mut st, &plan, kv_id, layer_base, pos0 + at, n, false, false, cap, &sealed, &[])?;
                    if !defer {
                        cu::memcpy_dtoh_async(&mut h[at * hidden..(at + n) * hidden], plan.h.ptr, c.stream).map_err(derr)?;
                    }
                }
                at += n;
            }
            return c.sync();
        }
        if b <= NMAX && !prompt_far {
            unsafe {
                cu::memcpy_htod_async(plan.h.ptr, h, c.stream).map_err(derr)?;
                let tab = plan_tab(&plan, positions.iter().copied());
                if !tab.is_empty() {
                    cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
                }
                let taps = tap_slots(c, &mut st, layers.len(), layer_base, hidden)?;
                let tl = taps.as_ref().map_or(&[][..], |t| &t.0[..]);
                dash_upload(&plan, layers, pos0, 0, b)?;
                let rows = run_small(c, &mut st, &plan, kv_id, layer_base, pos0, b, spec.is_some(), spec.is_some(), cap, &sealed, tl)?;
                if taps.is_some() {
                    super::taps_put(taps_read(c, tl, b, hidden)?);
                }
                // a greedy verify the pipeline armed: each row's argmax on
                // the card, the ids back instead of the logits
                let pick = spec.is_some() && super::verify_argmax_armed();
                if pick {
                    let ids_b = st.scr(c, "hy_vids", NMAX * 4)?;
                    let (rows_i, zero_i) = (rows as i32, 0i32);
                    c.launch(plan.fx.argmax, (b as u32, 1), 1024, 0, kargs!(plan.logits.ptr, rows_i, rows_i, ids_b, zero_i))?;
                    let mut ids = vec![0i32; b];
                    cu::memcpy_dtoh_async(ids.as_mut_slice(), ids_b, c.stream).map_err(derr)?;
                    c.sync()?;
                    super::verify_argmax_put(ids.iter().map(|&t| t as u32).collect());
                } else if let Some(sp) = spec.as_mut() {
                    sp.logits_out.resize(b * rows, 0.0);
                    cu::memcpy_dtoh_async(sp.logits_out.as_mut_slice(), plan.logits.ptr, c.stream).map_err(derr)?;
                }
                if !defer || spec.is_some() {
                    cu::memcpy_dtoh_async(h, plan.h.ptr, c.stream).map_err(derr)?;
                } else {
                    // nothing comes back: the stream runs on
                    return Ok(());
                }
            }
            c.sync()
        } else {
            if prompt_far && b > SEAL_ROWS {
                // past a seal the far field's mirror has room for SEAL_ROWS rows
                // after its near ones: the chunk in pieces of that many
                let mut at = 0usize;
                while at < b {
                    let n = SEAL_ROWS.min(b - at);
                    let last = at + n == b;
                    unsafe { big(c, &mut st, &plan, kv_id, layer_base, &positions[at..at + n], &mut h[at * hidden..(at + n) * hidden], defer && last, &sealed)? };
                    at += n;
                }
                return Ok(());
            }
            dash_upload(&plan, layers, pos0, 0, b)?;
            unsafe { big(c, &mut st, &plan, kv_id, layer_base, positions, h, defer, &sealed) }
        }
    };
    match run() {
        Ok(()) => {
            if let Some(lr) = layers_run {
                *lr = layers.len();
            }
            super::BATCH_ROWS.fetch_add(b as u64, Ordering::Relaxed);
            super::BATCH_NS.fetch_add(t_all.elapsed().as_nanos() as u64, Ordering::Relaxed);
            BatchGraphOutcome::Completed
        }
        Err(e) => {
            tracing::error!("cuda hybrid batch graph failed: {e}");
            let _ = c.sync();
            BatchGraphOutcome::Failed
        }
    }
}

/// A sealed layer's far sums on the card (`kv_cache`'s seal without the
/// host's replay): `hy_far_insert_n` over the far rows a tile at a time, the
/// running maxima, masses and numerators staying on the card between tiles.
/// Row r of group g is `ks[g][r * stride..][..hd]`: `stride` is `hd` for a
/// group's own rows, the whole row's width for rows that hold every KV head
/// (Dory's, `ks[g]` then starting at the group's first value).
#[allow(clippy::too_many_arguments)]
pub(super) fn far_sums(c: &Ctx, nkv: usize, hpk: usize, hd: usize, m: usize, scale: f32, qt: &[f32], idx: &[u32], ks: &[&[f32]], vs: &[&[f32]], stride: usize) -> Res<(Vec<f32>, Vec<f32>)> {
    let nh = nkv * hpk;
    let n = idx.len();
    let top = idx.iter().max().map_or(0, |&r| r as usize + 1);
    if m == 0 || m > 64 || stride < hd || ks.len() != nkv || vs.len() != nkv || qt.len() != nh * m * hd || ks.iter().chain(vs).any(|x| top > 0 && x.len() < (top - 1) * stride + hd) {
        return Err("far sums: unexpected geometry".into());
    }
    let t0 = std::time::Instant::now();
    let mut st = c.st.lock().unwrap();
    c.current()?;
    let fx = match st.hyb.funcs.get(&(hpk, hd)) {
        Some(f) => f.clone(),
        None => {
            let f = Arc::new(load_funcs(c, hpk, hd)?);
            st.hyb.funcs.insert((hpk, hd), f.clone());
            f
        }
    };
    let mut mz0 = vec![0f32; nh * 2 * m];
    for h in 0..nh {
        mz0[h * 2 * m..h * 2 * m + m].fill(f32::NEG_INFINITY);
    }
    let qtb = Buf::from_slice(qt)?;
    let mzb = Buf::from_slice(&mz0)?;
    let thb = Buf::new(nh * m * hd * 4)?;
    unsafe { cu::memset_d8_async(thb.ptr, 0, nh * m * hd * 4, c.stream).map_err(derr)? };
    // a tile of rows [row][kv head][hd] in f16, as the far field's mirror keeps them
    const TILE: usize = 8192;
    let tile = TILE.min(n.max(1));
    let row = nkv * hd;
    let kb = st.scr(c, "fs_k", tile * row * 2)?;
    let vb = st.scr(c, "fs_v", tile * row * 2)?;
    // two host tiles: one fills on the threads while the card sums the other
    let mut hk = [vec![0u16; tile * row], vec![0u16; tile * row]];
    let mut hv = [vec![0u16; tile * row], vec![0u16; tile * row]];
    let h16 = cortiq_core::quant::f32_to_f16;
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get()).clamp(1, 16);
    let mut r0 = 0usize;
    let mut slot = 0usize;
    while r0 < n {
        let nr = tile.min(n - r0);
        let (kh, vh) = (&mut hk[slot], &mut hv[slot]);
        let per = nr.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (ti, (kc, vc)) in kh[..nr * row].chunks_mut(per * row).zip(vh[..nr * row].chunks_mut(per * row)).enumerate() {
                sc.spawn(move || {
                    for (j, (kd, vd)) in kc.chunks_mut(row).zip(vc.chunks_mut(row)).enumerate() {
                        let at = idx[r0 + ti * per + j] as usize * stride;
                        for g in 0..nkv {
                            let (ksrc, vsrc) = (&ks[g][at..at + hd], &vs[g][at..at + hd]);
                            for d in 0..hd {
                                kd[g * hd + d] = h16(ksrc[d]);
                                vd[g * hd + d] = h16(vsrc[d]);
                            }
                        }
                    }
                });
            }
        });
        let (m_i, nkv_i, zero_i, w_i, e_i) = (m as i32, nkv as i32, 0i32, nr as i32, nr as i32);
        let warps = nh * m;
        unsafe {
            // stream order: this tile's copy waits for the last tile's sums
            cu::memcpy_htod_async(kb, &kh[..nr * row], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(vb, &vh[..nr * row], c.stream).map_err(derr)?;
            c.launch(fx.far_insert_n, ((warps * 32).div_ceil(256) as u32, 1), 256, 0, kargs!(kb, vb, qtb.ptr, mzb.ptr, thb.ptr, m_i, nkv_i, scale, zero_i, w_i, zero_i, e_i))?;
        }
        r0 += nr;
        slot ^= 1;
        if slot == 0 {
            // both host tiles are queued: the older must be on the card before it is written again
            c.sync()?;
        }
    }
    let mut mz = vec![0f32; nh * 2 * m];
    let mut th = vec![0f32; nh * m * hd];
    unsafe {
        cu::memcpy_dtoh_async(mz.as_mut_slice(), mzb.ptr, c.stream).map_err(derr)?;
        cu::memcpy_dtoh_async(th.as_mut_slice(), thb.ptr, c.stream).map_err(derr)?;
    }
    c.sync()?;
    if debug() {
        eprintln!("cuda hybrid: far sums of {n} rows x {nkv} groups on the card in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok((mz, th))
}

/// `CMF_CUDA_MEM=1`: what the card holds, on stderr, after `what`.
pub(super) fn mem_report(c: &Ctx, st: &State, what: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("CMF_CUDA_MEM").as_deref() == Ok("1")) {
        return;
    }
    let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
    let (free, total) = cu::mem_get_info().unwrap_or((0, 0));
    let scratch: u64 = st.scratch.values().map(|b| b.bytes as u64).sum();
    let mut big: Vec<(&str, usize)> = st.scratch.iter().map(|(k, b)| (*k, b.bytes)).collect();
    big.sort_by_key(|x| std::cmp::Reverse(x.1));
    let gdn: u64 = st
        .hyb
        .gdn
        .values()
        .map(|g| {
            let snap = g.snap.as_ref().map_or(0, |s| s.1.bytes + s.2.bytes);
            (g.s.bytes + g.ring.bytes + g.t_qkv.bytes + g.t_cq.bytes + g.t_gb.bytes + g.s_save.bytes + g.ring_save.bytes + snap) as u64
        })
        .sum();
    let o1: u64 = st.hyb.o1.values().map(|d| d.bytes).sum();
    let mirrors: Vec<(usize, usize)> = st.kv.values().map(|m| (m.cap, m.stored - m.base)).collect();
    let mcap: usize = mirrors.iter().map(|m| m.0).max().unwrap_or(0);
    eprintln!(
        "cuda mem after {what}: used {:.0} of {:.0} MiB; weights {:.0}, K/V {:.0} (far fields {:.0}, {} mirrors of up to {mcap} rows), GDN states {:.0} in {}, scratch {:.0} (largest {}), graphs {}",
        mb((total - free) as u64),
        mb(total as u64),
        mb(c.resident.load(Ordering::Relaxed)),
        mb(c.kv_bytes.load(Ordering::Relaxed)),
        mb(o1),
        mirrors.len(),
        mb(gdn),
        st.hyb.gdn.len(),
        mb(scratch),
        big.iter().take(6).map(|(k, b)| format!("{k} {:.0}", mb(*b as u64))).collect::<Vec<_>>().join(", "),
        st.hyb.graphs.len()
    );
}

/// The prompt path: a chunk of `b > NMAX` rows on cuBLAS.
#[allow(clippy::too_many_arguments)]
/// A prompt chunk's routed FFN, its experts grouped (see `hy_moe_group`):
/// the normed rows in f32, the router and the shared gate as f32 GEMMs,
/// every row's picks, the (row, pick) entries sorted by expert and laid out
/// as input planes, a tensor-core launch over every expert's gate/up and one
/// over its down, the mixture (and `dsh`, the shared expert's rows) into h.
#[allow(clippy::too_many_arguments)]
unsafe fn moe_grouped(c: &Ctx, st: &mut State, fx: &HFuncs, m: &MoeL, bh: u64, dsh: u64, b: usize, hidden: usize, eps: f32, gemma_i: i32, norm: u64) -> Res<()> {
    let (e, k, inter) = (m.e, m.k, m.inter);
    if e > 1024 || k > 64 || inter % 32 != 0 {
        return Err(format!("a grouped routed FFN of {e} experts, top {k}, width {inter}"));
    }
    let nk = b * k;
    let rmax = moe_rmax();
    let jmax = e + nk.div_ceil(rmax);
    let (nul, zero_i, hid_i) = (0u64, 0i32, hidden as i32);
    let x32 = st.scr(c, "hb_mx32", b * hidden * 4)?;
    let logits = st.scr(c, "hb_mlog", b * e * 4)?;
    let sg = if m.sgw.is_some() { st.scr(c, "hb_msg", b * 4)? } else { 0 };
    let pidx = st.scr(c, "hb_mpidx", nk * 4)?;
    let pwt = st.scr(c, "hb_mpwt", nk * 4)?;
    let pos = st.scr(c, "hb_mpos", nk * 4)?;
    let src = st.scr(c, "hb_msrc", nk * 4)?;
    let xg0 = st.scr(c, "hb_mxg0", nk * hidden * 2)?;
    let xg1 = st.scr(c, "hb_mxg1", nk * hidden * 2)?;
    let qq0 = st.scr(c, "hb_mqq0", nk * inter * 2)?;
    let qq1 = st.scr(c, "hb_mqq1", nk * inter * 2)?;
    let ys = st.scr(c, "hb_mys", nk * hidden * 4)?;
    let gu_a = st.scr(c, "hb_mgua", jmax * std::mem::size_of::<HyJob>())?;
    let dn_a = st.scr(c, "hb_mdna", jmax * std::mem::size_of::<HyJob>())?;
    let (gi, tb) = if rmax == 64 { (1usize, 1u32) } else { (0, 2) };
    let (bpg, bpd) = (m.bpj_gu.div_ceil(tb), m.bpj_dn.div_ceil(tb));
    let (n_i, e_i, k_i, nk_i, rmax_i, jmax_i, inter_i) = (b as i32, e as i32, k as i32, nk as i32, rmax as i32, jmax as i32, inter as i32);
    let (bg_i, bd_i) = (bpg as i32, bpd as i32);
    let bias = m.bias.as_ref().map_or(0, |x| x.ptr);
    unsafe {
        c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, norm, x32, hid_i, eps, gemma_i, zero_i, nul, nul, nul, zero_i))?;
        c.gemm(true, false, e, b, hidden, 1.0, m.router.w.ptr, hidden, x32, hidden, 0.0, logits, e, Gemm::F32)?;
        if let Some(g) = &m.sgw {
            c.gemm(true, false, 1, b, hidden, 1.0, g.w.ptr, hidden, x32, hidden, 0.0, sg, 1, Gemm::F32)?;
        }
        c.launch(fx.moe_topk, (b.div_ceil(8) as u32, 1), 256, 0, kargs!(logits, bias, n_i, e_i, k_i, m.flags, m.route_scale, pidx, pwt))?;
        c.launch(
            fx.moe_group,
            (1, 1),
            1024,
            0,
            kargs!(pidx, nk_i, k_i, e_i, rmax_i, jmax_i, m.gu_t.ptr, m.dn_t.ptr, gu_a, dn_a, pos, src, qq0, qq1, hid_i, inter_i, ys, bg_i, bd_i),
        )?;
        let tot = (nk * hidden) as u64;
        c.launch(fx.moe_gather, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x32, src, nk_i, hid_i, xg0, xg1, xg1))?;
        let f = fx.mmp_g[gi];
        c.launch(f, (jmax as u32 * bpg, 1), 32 * 8, 0, kargs!(gu_a, jmax_i, xg0, xg1, xg1, hid_i, rmax_i, hid_i))?;
        c.launch(f, (jmax as u32 * bpd, 1), 32 * 8, 0, kargs!(dn_a, jmax_i, qq0, qq1, qq1, inter_i, rmax_i, inter_i))?;
        let gated_i = (m.gated && sg != 0) as i32;
        c.launch(fx.moe_gcombine, (hidden.div_ceil(256) as u32, b as u32), 256, 0, kargs!(bh, ys, pos, pwt, dsh, sg, gated_i, n_i, k_i, hid_i))?;
    }
    Ok(())
}

/// The prompt path's residual add of a product `y` [b][hidden] under a
/// sandwich norm `w` (Gemma): h += rmsnorm(y) * w, a block a row.
unsafe fn sandwich_rows(c: &Ctx, plan: &HPlan, w: &Buf, h: u64, y: u64, b: usize) -> Res<()> {
    let (hi, gi) = (plan.hidden as i32, plan.gemma as i32);
    unsafe { c.launch(plan.fx.norm_add, (b as u32, 1), 256, 0, kargs!(h, y, w.ptr, hi, plan.eps, gi)) }
}

unsafe fn big(
    c: &Ctx,
    st: &mut State,
    plan: &HPlan,
    kv_id: u64,
    layer_base: usize,
    positions: &[usize],
    h: &mut [f32],
    defer: bool,
    sealed: &[bool],
) -> Res<()> {
    let fx = &plan.fx;
    let b = positions.len();
    let pos0 = positions[0];
    let (hidden, nh, nkv, hd, hpk) = (plan.hidden, plan.nh, plan.nkv, plan.hd, plan.hpk);
    let sealed_window = plan.layers.iter().zip(sealed).any(|(l, &s)| {
        s && matches!(&l.mix, Mix::Attn(a) if a.window.is_some() || a.sink.is_some() || a.nkv != plan.nkv || a.hd != plan.hd)
    });
    if plan.layers.iter().any(small_only) || sealed_window || (plan.layers.iter().any(|l| l.moe.is_some()) && !sealed.iter().all(|&s| !s)) {
        return Err("sandwich norms, a short convolution or a sealed routed FFN on the prompt path".into());
    }
    let (gw, _) = prec();
    if gw != Gemm::Half {
        return Err("the hybrid prompt path runs f16 weight GEMMs only".into());
    }
    let (eps, gemma_i, b_i, one) = (plan.eps, plan.gemma as i32, b as i32, 1i32);
    let hid_i = hidden as i32;
    let (nul, zero_i) = (0u64, 0i32);
    let bh = st.scr(c, "hb_h", b * hidden * 4)?;
    let bxn = st.scr(c, "hb_xn", b * hidden * 2)?;
    let wmax = plan
        .layers
        .iter()
        .flat_map(|l| {
            let mut v = vec![(l.gu.rows, l.gu.cols), (l.down.rows, l.down.cols)];
            match &l.mix {
                Mix::Gdn(g) => v.extend([(g.qkv.rows, g.qkv.cols), (g.z.rows, g.z.cols), (g.out.rows, g.out.cols)]),
                Mix::Attn(a) => v.extend([(a.q.rows, a.q.cols), (a.k.rows, a.k.cols), (a.v.rows, a.v.cols), (a.o.rows, a.o.cols)]),
                Mix::Conv(cv) => v.extend([(cv.inp.rows, cv.inp.cols), (cv.out.rows, cv.out.cols)]),
            }
            v
        })
        .map(|(r, c)| r * c)
        .max()
        .unwrap_or(0);
    // a large weight expands a band of rows at a time
    let wd = st.scr(c, "hb_wd", (wmax * 2).min(wd_cap().max(1 << 20)))?;
    // a DFlash drafter's taps: the chunk's rows after each asked layer
    let want = super::taps_wanted();
    let tapping = !want.is_empty() && layer_base == 0 && want.iter().all(|&l| l < plan.layers.len());
    let mut tapped = if tapping { vec![0f32; want.len() * b * hidden] } else { Vec::new() };
    // the call's rotation rows, then each extra table's (a local layer's)
    let tab = plan_tab(plan, positions.iter().copied());
    let btab = st.scr(c, "hb_tab", tab.len().max(1) * 4)?;
    unsafe {
        cu::memcpy_htod_async(bh, h, c.stream).map_err(derr)?;
        if !tab.is_empty() {
            cu::memcpy_htod_async(btab, &tab, c.stream).map_err(derr)?;
        }
    }
    let tm_ = std::cell::Cell::new(std::time::Instant::now());
    let tm = &tm_;
    BROWS.fetch_add(b as u64, Ordering::Relaxed);
    bst(c, 7, tm)?;
    // y [b][ldy] (+)= alpha W x: the weight expanded to f16, a GEMM a band
    // of rows (the whole matrix when it fits the band)
    let wd_rows = (wd_cap().max(1 << 20) / 2).max(1);
    let mul = |m: &DevMat, x: u64, y: u64, ldy: usize, alpha: f32, beta: f32| -> Res<()> {
        let (rows_i, cols_i) = (m.rows as i32, m.cols as i32);
        if !matches!(m.kind, 1 | 2 | 3 | 5 | 6 | 9 | 10 | 15) {
            return Err("a weight other than repacked q4tp, q2tp, int8, q1 or ternary tiles on the prompt path".into());
        }
        if m.kind == 6 && m.rows * m.cols <= wd_rows {
            let threads = (m.rows.div_ceil(16) * (m.cols / 32) * 32) as u64;
            unsafe {
                c.launch(fx.expand_p, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i))?;
                bst(c, 1, tm)?;
                c.gemm(true, false, m.rows, b, m.cols, alpha, wd, m.cols, x, m.cols, beta, y, ldy, Gemm::Half)?;
                return bst(c, 2, tm);
            }
        }
        let tiles = m.rows.div_ceil(16);
        let band = (wd_rows / m.cols / 16).max(1);
        let mut t0 = 0usize;
        while t0 < tiles {
            let nt = band.min(tiles - t0);
            let r0 = t0 * 16;
            let nr = (nt * 16).min(m.rows - r0);
            let threads = (nt * (m.cols / 32) * 32) as u64;
            let (t0_i, nt_i) = (t0 as i32, nt as i32);
            unsafe {
                if m.kind == 10 {
                    c.launch(fx.expand_q4g, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), wd, rows_i, cols_i, t0_i, nt_i))?;
                } else if m.kind == 1 || m.kind == 9 {
                    let aff_i = (m.kind == 9) as i32;
                    c.launch(fx.expand_q2, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i, t0_i, nt_i, aff_i))?;
                } else if m.kind == 2 {
                    c.launch(fx.expand_q1t, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), wd, rows_i, cols_i, t0_i, nt_i))?;
                } else if m.kind == 3 {
                    c.launch(fx.expand_q1, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), wd, rows_i, cols_i, t0_i, nt_i))?;
                } else if m.kind == 5 || m.kind == 15 {
                    let dual_i = (m.kind == 15) as i32;
                    c.launch(fx.expand_i8, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux1(), m.aux0(), wd, rows_i, cols_i, t0_i, nt_i, dual_i))?;
                } else {
                    c.launch(fx.expand_pr, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i, t0_i, nt_i))?;
                }
                bst(c, 1, tm)?;
                c.gemm(true, false, nr, b, m.cols, alpha, wd, m.cols, x, m.cols, beta, y + (r0 * 4) as u64, ldy, Gemm::Half)?;
                bst(c, 2, tm)?;
            }
            t0 += nt;
        }
        Ok(())
    };
    // mHC (SenseNova-Loop): the chunk's streams, coefficients and logits;
    // a sub-layer reads its input rows from bh and leaves its product in
    // "hb_yo" for the merge
    let mrows = match &plan.mhc {
        Some(mp) => Some(MhcRows {
            h: bh,
            yo: st.scr(c, "hb_yo", b * hidden * 4)?,
            r: st.scr(c, "hb_mr", b * mp.ns * hidden * 4)?,
            coef: st.scr(c, "hb_mc", b * 20 * 4)?,
            lg: st.scr(c, "hb_ml", b * 24 * 4)?,
        }),
        None => None,
    };
    for (li, pl) in plan.layers.iter().enumerate() {
        let key = (kv_id, layer_base + li);
        let mh = match (&pl.mhc, &plan.mhc, mrows) {
            (Some(m), Some(mp), Some(r)) => Some((m, mp, r)),
            _ => None,
        };
        unsafe {
            if let Some((m, mp, r)) = mh {
                mhc_in(c, fx, hidden, m, mp, r, false, b)?;
            }
            c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, pl.in_norm.ptr, bxn, hid_i, eps, gemma_i, one, nul, nul, nul, zero_i))?;
        }
        // a Prism file: the projections read the rows in their basis, the GDN
        // gates the untouched ones
        let bxf = if plan.prism.is_some() {
            let bxp = st.scr(c, "hb_xp", b * hidden * 2)?;
            unsafe { prism_rows(c, plan, bxn, bxp, hidden, hidden, b)? };
            bxp
        } else {
            bxn
        };
        bst(c, 0, tm)?;
        match &pl.mix {
            // the prompt path is not entered with such a layer (see its guard)
            Mix::Conv(_) => return Err("a short convolution on the prompt path".into()),
            Mix::Gdn(g) => {
                let (cd_i, kk_i, nv_i, nk_i, dk_i, dv_i) =
                    (g.cdim as i32, g.kk as i32, g.nv as i32, g.nk as i32, g.dk as i32, g.dv as i32);
                let vd = g.nv * g.dv;
                let yq = st.scr(c, "hb_qkv", b * g.cdim * 4)?;
                let yz = st.scr(c, "hb_z", b * vd * 4)?;
                let yab = st.scr(c, "hb_ab", b * 2 * g.nv * 4)?;
                let cq = st.scr(c, "hb_cq", b * g.cdim * 4)?;
                let gbp = st.scr(c, "hb_gb", b * 2 * g.nv * 4)?;
                let yo = st.scr(c, "hb_o", b * vd * 4)?;
                let yof = st.scr(c, "hb_of", b * vd * 2)?;
                mul(&g.qkv, bxf, yq, g.cdim, 1.0, 0.0)?;
                mul(&g.z, bxf, yz, vd, 1.0, 0.0)?;
                unsafe {
                    c.gemm(true, false, g.nv, b, hidden, 1.0, g.a_h.ptr, hidden, bxn, hidden, 0.0, yab, 2 * g.nv, Gemm::Half)?;
                    c.gemm(true, false, g.nv, b, hidden, 1.0, g.b_h.ptr, hidden, bxn, hidden, 0.0, yab + (g.nv * 4) as u64, 2 * g.nv, Gemm::Half)?;
                }
                let gs = st.hyb.gdn.get_mut(&key).ok_or("GDN state missing")?;
                unsafe {
                    let tot = (b * g.cdim) as u64;
                    c.launch(fx.conv32, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(yq, cd_i, gs.ring.ptr, g.taps.ptr, cq, b_i, cd_i, kk_i))?;
                    if g.kk > 1 {
                        c.launch(fx.ring, ((g.cdim as u32).div_ceil(256), 1), 256, 0, kargs!(gs.ring.ptr, yq, cd_i, b_i, cd_i, kk_i))?;
                    }
                    c.launch(fx.prep32, ((b * 2 * g.nk) as u32, 1), g.dk as u32, 0, kargs!(cq, cd_i, nk_i, dk_i))?;
                    let nvt = (b * g.nv) as u64;
                    c.launch(fx.gb32, (nvt.div_ceil(256) as u32, 1), 256, 0, kargs!(yab, g.a_log.ptr, g.dt_bias.ptr, gbp, b_i, nv_i))?;
                    bst(c, 3, tm)?;
                    c.launch(fx.scan, ((vd as u32).div_ceil(8), 1), 256, 0, kargs!(cq, gbp, gs.s.ptr, yo, b_i, cd_i, nv_i, nk_i, dk_i, dv_i, nul, nul, zero_i, kk_i))?;
                    bst(c, 4, tm)?;
                    let (vd_i, sig0) = (vd as i32, 0i32);
                    c.launch(fx.out32, ((b * g.nv) as u32, 1), g.dv as u32, 0, kargs!(yo, yz, vd_i, g.norm.ptr, yof, nv_i, dv_i, eps, sig0, one, nul, nul, nul, zero_i))?;
                }
                bst(c, 3, tm)?;
                gs.next_pos = pos0 + b;
                gs.tape = None;
                unsafe { prism_rows(c, plan, yof, yof, vd, vd, b)? };
                match (pl.attn_out.as_deref(), mh) {
                    (_, Some((m, mp, r))) => {
                        mul(&g.out, yof, r.yo, hidden, 1.0, 0.0)?;
                        unsafe { mhc_out(c, fx, hidden, m, mp, r, false, b)? };
                    }
                    (Some(w), None) => {
                        let by = st.scr(c, "hb_yo", b * hidden * 4)?;
                        mul(&g.out, yof, by, hidden, 1.0, 0.0)?;
                        unsafe { sandwich_rows(c, plan, w, bh, by, b)? };
                    }
                    (None, None) => mul(&g.out, yof, bh, hidden, 1.0, 1.0)?,
                }
            }
            Mix::Attn(a) if sealed.get(li).copied().unwrap_or(false) => {
                // past the seal: the chunk's rows after the far field's near
                // rows, their attention over those and the landmarks, then
                // the rows into the window (and the window's oldest into the
                // far sums)
                let qr = a.q.rows;
                let yq = st.scr(c, "hb_q", b * qr * 4)?;
                let yk = st.scr(c, "hb_k", b * nkv * hd * 4)?;
                let yv = st.scr(c, "hb_v", b * nkv * hd * 4)?;
                let qg = st.scr(c, "hb_qg", b * nh * hd * 4)?;
                let qgh = st.scr(c, "hb_qgh", b * nh * hd * 2)?;
                let gt = st.scr(c, "hb_gate", b * nh * hd * 4)?;
                let aog = st.scr(c, "hb_aog", b * nh * hd * 4)?;
                let bao = st.scr(c, "hb_ao", b * nh * hd * 2)?;
                let bis = st.scr(c, "hb_is", b * nh * 4)?;
                let bfa = st.scr(c, "hb_fa", b * nh * 4)?;
                let bfc = st.scr(c, "hb_fc", b * nh * 4)?;
                let bfm = st.scr(c, "hb_fm", b * nh * 4)?;
                let bfz = st.scr(c, "hb_fz", b * nh * 4)?;
                let bfn = st.scr(c, "hb_fn", b * nh * hd * 4)?;
                mul(&a.q, bxf, yq, qr, 1.0, 0.0)?;
                mul(&a.k, bxf, yk, nkv * hd, 1.0, 0.0)?;
                mul(&a.v, bxf, yv, nkv * hd, 1.0, 0.0)?;
                let (kp, vp, kt, qt, mu, mz, th, ns, w, m, win_len, win_head, far_len, fscale, rect_fm) = {
                    let d = st.hyb.o1.get(&key).ok_or("far field missing on the prompt path")?;
                    if (d.nkv, d.hpk, d.hd) != (nkv, hpk, hd) {
                        return Err("far field of another geometry".into());
                    }
                    if d.next_pos.is_some_and(|p| p != pos0) {
                        return Err(format!("far field at row {:?}, the chunk at {pos0}", d.next_pos));
                    }
                    (d.nk.ptr, d.nv.ptr, d.kt.ptr, d.qt.ptr, d.mu.ptr, d.mz.ptr, d.th.ptr, d.ns, d.w, d.m, d.win_len, d.win_head, d.far_len, d.scale, d.rect_fm)
                };
                if b > SEAL_ROWS.max(NMAX) || b > w {
                    return Err(format!("a chunk of {b} rows past a seal (window {w})"));
                }
                // the near rows before the chunk: the sinks with the kept rows,
                // then the window (unwrapped until it is full)
                let n0 = ns + win_len;
                let (nh_i, nkv_i, half_i, gated_i) = (nh as i32, nkv as i32, a.half as i32, a.gated as i32);
                let (qs_i, kvs_i, n0_i) = (qr as i32, (nkv * hd) as i32, n0 as i32);
                let (qn, kn) = (a.qn.as_ref().map_or(0, |b| b.ptr), a.kn.as_ref().map_or(0, |b| b.ptr));
                let (qk_g, (bqp, bkp, bvp)) = qk_flags(a, gemma_i);
                // a local layer's rotation rows follow the call's (`plan_tab`)
                let tabl = btab + (plan.tab_off(a.local, b) * 4) as u64;
                unsafe {
                    c.launch(
                        fx.qk,
                        ((nh + nkv) as u32, b as u32),
                        hd as u32,
                        0,
                        kargs!(yq, qs_i, yk, yv, kvs_i, qn, kn, qg, gt, kp, vp, tabl, nh_i, nkv_i, half_i, eps, qk_g, gated_i, n0_i, b_i, nul, qgh, bqp, bkp, bvp),
                    )?;
                }
                // the archive of a prompt past its first seal: the chunk's rows
                // and one query row in the trace's stride off the card
                super::acap_note(c, layer_base + li, kp, vp, n0, b, nkv * hd, qg, nh, nkv, hd, pos0)?;
                let far_on = far_len > 0;
                if far_on {
                    let (m_i, sc, rf, bb_i) = (m as i32, fscale, rect_fm as i32, b as i32);
                    unsafe {
                        c.launch(fx.far_rows, (nh as u32, b as u32), 256, 0, kargs!(qg, kt, mu, mz, th, bfm, bfz, bfn, m_i, nh_i, sc, rf, bb_i))?;
                    }
                }
                let lda = nkv * hd;
                let grow = hpk * hd;
                // a key r of the near rows sits at position col_base + r for the mask:
                // all the rows before the chunk are visible, the chunk's causally
                let col_base = pos0 - n0;
                let mut b0 = 0usize;
                while b0 < b {
                    let bn = (b - b0).min(att_rows());
                    let p0 = pos0 + b0;
                    let l_rows = n0 + b0 + bn;
                    let cols = bn * hpk;
                    let bs = st.scr(c, "hb_s", nkv * l_rows * cols * 4)?;
                    let bp = st.scr(c, "hb_p", nkv * l_rows * cols * 2)?;
                    let (l_i, p0_i, cb_i, hpk_i) = (l_rows as i32, p0 as i32, col_base as i32, hpk as i32);
                    let (cols_i, istride) = (cols as i32, (b * hpk) as i32);
                    let off = (b0 * hpk * 4) as u64;
                    let (fmp, fzp) = if far_on { (bfm + (b0 * nh * 4) as u64, bfz + (b0 * nh * 4) as u64) } else { (0u64, 0u64) };
                    unsafe {
                        gemm_batched_h(c, true, l_rows, cols, hd, plan.scale, kp, lda, hd, qgh + (b0 * grow * 2) as u64, hd, b * grow, bs, l_rows, l_rows * cols, nkv)?;
                        c.launch(fx.softmax_hf, ((nkv * cols) as u32, 1), 256, 0, kargs!(bs, bp, bis + off, bfa + off, bfc + off, l_i, p0_i, cb_i, hpk_i, cols_i, istride, fmp, fzp, nh_i))?;
                        gemm_batched_h(c, false, hd, cols, l_rows, 1.0, vp, lda, hd, bp, l_rows, l_rows * cols, aog + (b0 * grow * 4) as u64, hd, b * grow, nkv)?;
                    }
                    b0 += bn;
                }
                let n_ao = (b * nh * hd) as u64;
                let gate_p = if a.gated { gt } else { 0u64 };
                let fnp = if far_on { bfn } else { 0u64 };
                unsafe {
                    c.launch(fx.attn_out_f, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(aog, bis, bfa, bfc, fnp, gate_p, bao, nh_i, b_i, one))?;
                }
                // the commit: the first rows fill the window where they are; past
                // a full window each row takes the oldest's slot, which goes into
                // the far sums first
                let fill = b.min(w - win_len);
                let rest = b - fill;
                let head0 = if win_len == w { win_head } else { 0 };
                if rest > 0 {
                    let (m_i, ns_i, w_i, h0_i, e_i) = (m as i32, ns as i32, w as i32, head0 as i32, rest as i32);
                    let warps = nkv * hpk * m;
                    let row_b = (nkv * hd * 2) as u64;
                    unsafe {
                        c.launch(fx.far_insert_n, ((warps * 32).div_ceil(256) as u32, 1), 256, 0, kargs!(kp, vp, qt, mz, th, m_i, nkv_i, fscale, ns_i, w_i, h0_i, e_i))?;
                        // the rows past the window's room into the freed slots: two runs at most
                        let src = (n0 + fill) as u64;
                        let r1 = rest.min(w - head0);
                        for (dst, from, n) in [((ns + head0) as u64, src, r1), (ns as u64, src + r1 as u64, rest - r1)] {
                            if n == 0 {
                                continue;
                            }
                            cu::memcpy_dtod_async(kp + dst * row_b, kp + from * row_b, n * row_b as usize, c.stream).map_err(derr)?;
                            cu::memcpy_dtod_async(vp + dst * row_b, vp + from * row_b, n * row_b as usize, c.stream).map_err(derr)?;
                        }
                    }
                }
                {
                    let d = st.hyb.o1.get_mut(&key).ok_or("far field missing on the prompt path")?;
                    d.win_len = (win_len + b).min(w);
                    if rest > 0 {
                        d.win_head = (head0 + rest) % w;
                        d.far_len += rest;
                    }
                    d.next_pos = Some(pos0 + b);
                    d.spec = None;
                }
                bst(c, 5, tm)?;
                unsafe { prism_rows(c, plan, bao, bao, nh * hd, nh * hd, b)? };
                match (pl.attn_out.as_deref(), mh) {
                    (_, Some((m, mp, r))) => {
                        mul(&a.o, bao, r.yo, hidden, 1.0, 0.0)?;
                        unsafe { mhc_out(c, fx, hidden, m, mp, r, false, b)? };
                    }
                    (Some(w), None) => {
                        let by = st.scr(c, "hb_yo", b * hidden * 4)?;
                        mul(&a.o, bao, by, hidden, 1.0, 0.0)?;
                        unsafe { sandwich_rows(c, plan, w, bh, by, b)? };
                    }
                    (None, None) => mul(&a.o, bao, bh, hidden, 1.0, 1.0)?,
                }
            }
            Mix::Attn(a) => {
                // the layer's head geometry and its kernels
                let (nkv, hd, fx) = (a.nkv, a.hd, &*a.fx);
                let hpk = nh / nkv;
                let qr = a.q.rows;
                let yq = st.scr(c, "hb_q", b * qr * 4)?;
                let yk = st.scr(c, "hb_k", b * nkv * hd * 4)?;
                let yv = st.scr(c, "hb_v", b * nkv * hd * 4)?;
                let qg = st.scr(c, "hb_qg", b * nh * hd * 4)?;
                let qgh = st.scr(c, "hb_qgh", b * nh * hd * 2)?;
                let gt = st.scr(c, "hb_gate", b * nh * hd * 4)?;
                let aog = st.scr(c, "hb_aog", b * nh * hd * 4)?;
                let bao = st.scr(c, "hb_ao", b * nh * hd * 2)?;
                let bis = st.scr(c, "hb_is", b * nh * 4)?;
                mul(&a.q, bxf, yq, qr, 1.0, 0.0)?;
                mul(&a.k, bxf, yk, nkv * hd, 1.0, 0.0)?;
                mul(&a.v, bxf, yv, nkv * hd, 1.0, 0.0)?;
                let hint = st.kv_hint;
                let (slot0, kp, vp, m_base) = {
                    let m = st.kv.get_mut(&key).ok_or("K/V mirror missing")?;
                    let out = ((m.stored - m.base) as i32, m.kptr(), m.vptr(), m.base);
                    m.stored += b;
                    out
                };
                let (nh_i, nkv_i, half_i, gated_i) = (nh as i32, nkv as i32, a.half as i32, a.gated as i32);
                let (qs_i, kvs_i) = (qr as i32, (nkv * hd) as i32);
                let (qn, kn) = (a.qn.as_ref().map_or(0, |b| b.ptr), a.kn.as_ref().map_or(0, |b| b.ptr));
                let (qk_g, (bqp, bkp, bvp)) = qk_flags(a, gemma_i);
                // a local layer's rotation rows follow the call's (`plan_tab`);
                // a DASH layer's are its rows' own (`dash_upload`)
                let tabl = match (&a.dash, &plan.dash) {
                    (Some(_), Some(db)) => db.tab.ptr,
                    _ => btab + (plan.tab_off(a.local, b) * 4) as u64,
                };
                unsafe {
                    c.launch(
                        fx.qk,
                        ((nh + nkv) as u32, b as u32),
                        hd as u32,
                        0,
                        kargs!(yq, qs_i, yk, yv, kvs_i, qn, kn, qg, gt, kp, vp, tabl, nh_i, nkv_i, half_i, eps, qk_g, gated_i, slot0, b_i, nul, qgh, bqp, bkp, bvp),
                    )?;
                }
                // a far-field layer of a prompt read for its seal: the rotated
                // queries its trace keeps wait for `o1_qcap_drain`
                let cap_stride = super::QCAP.with(|q| {
                    q.borrow().as_ref().filter(|q| q.layers.contains(&(layer_base + li))).map(|q| q.stride.max(1))
                });
                if let Some(stride) = cap_stride {
                    let first = (stride - pos0 % stride) % stride;
                    let kept = if first < b { (b - first).div_ceil(stride) } else { 0 };
                    if kept > 0 {
                        let floats = kept * nh * hd;
                        let buf = Buf::new(floats * 4)?;
                        let (first_i, stride_i, hd_i) = (first as i32, stride as i32, hd as i32);
                        unsafe {
                            c.launch(c.f.q_rows_strided, (kept as u32, 1), 256, 0, kargs!(qg, buf.ptr, nh_i, nkv_i, hd_i, b_i, first_i, stride_i))?;
                        }
                        super::QCAP.with(|q| {
                            if let Some(q) = q.borrow_mut().as_mut() {
                                q.bufs.push((layer_base + li, floats, buf));
                            }
                        });
                    }
                }
                if let (Some(dl), Some(db)) = (&a.dash, &plan.dash) {
                    // DASH: the router over the rows normed in f32, the closing
                    // blocks' summaries, the attend over the read set into bao
                    let bx32 = st.scr(c, "hb_x32", b * hidden * 4)?;
                    let (dopen, dbar) = match st.hyb.dash.entry(key) {
                        std::collections::hash_map::Entry::Occupied(e) => (e.get().0.ptr, e.get().1.ptr),
                        std::collections::hash_map::Entry::Vacant(v) => {
                            let e = v.insert((Buf::new(3 * dl.block * 128 * 4)?, Buf::new(DASH_BLOCKS * 128 * 4)?));
                            (e.0.ptr, e.1.ptr)
                        }
                    };
                    let (blk_i, k_i, win_i) = (dl.block as i32, dl.k as i32, (dl.block * dl.local_blocks) as i32);
                    let gate_p = if a.gated { gt } else { 0u64 };
                    unsafe {
                        c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, pl.in_norm.ptr, bx32, hid_i, eps, gemma_i, zero_i, nul, nul, nul, zero_i))?;
                        c.launch(
                            fx.dash_route,
                            (b as u32, 1),
                            256,
                            0,
                            kargs!(
                                bx32, hid_i, dl.w.ptr, dl.qn.ptr, dl.kn.ptr, db.tab.ptr, dopen, dbar, db.coff.ptr, db.cands.ptr, db.sc.ptr,
                                db.sel.ptr, db.bias.ptr, db.rec.ptr, b_i, blk_i, k_i, eps, dl.idx_scale, dl.floor, dl.sigma
                            ),
                        )?;
                        c.launch(fx.dash_close, (1, 1), 128, 0, kargs!(dopen, dl.hc.ptr, dbar, db.rec.ptr, b_i, blk_i))?;
                        c.launch(
                            fx.dash_attend,
                            (nh as u32, b as u32),
                            256,
                            0,
                            kargs!(qg, kp, vp, db.sel.ptr, db.bias.ptr, gate_p, nul, nul, zero_i, bao, db.rec.ptr, nul, slot0, nh_i, nkv_i, b_i, blk_i, win_i, plan.scale),
                        )?;
                    }
                } else {
                let lda = nkv * hd;
                let grow = hpk * hd;
                let mut b0 = 0usize;
                // a sliding window: the block's keys start where its first row's window does
                let win = a.window.unwrap_or(0);
                while b0 < b {
                    let p0 = pos0 + b0;
                    let col_base = if win > 0 { m_base.max((p0 + 1).saturating_sub(win)) } else { m_base };
                    let (kpw, vpw) = (kp + ((col_base - m_base) * lda * 2) as u64, vp + ((col_base - m_base) * lda * 2) as u64);
                    let fit = |n: usize| (p0 + n - col_base) * n * nh * 4 <= 384 << 20;
                    let mut bn = (b - b0).min(att_rows());
                    while bn > 64 && !fit(bn) {
                        bn = bn.div_ceil(2);
                    }
                    let l_rows = p0 + bn - col_base;
                    let cols = bn * hpk;
                    let whole = (hint.max(pos0 + b) * b.min(att_rows()) * nh * 4).min(400 << 20);
                    let bs = st.scr(c, "hb_s", (nkv * l_rows * cols * 4).max(whole))?;
                    let bp = st.scr(c, "hb_p", (nkv * l_rows * cols * 2).max(whole / 2))?;
                    let (l_i, p0_i, cb_i, hpk_i) = (l_rows as i32, p0 as i32, col_base as i32, hpk as i32);
                    let (cols_i, istride, win_i) = (cols as i32, (b * hpk) as i32, win as i32);
                    let isum = bis + (b0 * hpk * 4) as u64;
                    // the mirrors hold f16 rows: both GEMMs on f16 operands
                    // (the query copy and the exponentials), f32 out
                    unsafe {
                        gemm_batched_h(c, true, l_rows, cols, hd, plan.scale, kpw, lda, hd, qgh + (b0 * grow * 2) as u64, hd, b * grow, bs, l_rows, l_rows * cols, nkv)?;
                        let sinkp = a.sink.as_ref().map_or(0u64, |b| b.ptr);
                        c.launch(fx.softmax_h, ((nkv * cols) as u32, 1), 256, 0, kargs!(bs, bp, isum, l_i, p0_i, cb_i, hpk_i, cols_i, istride, win_i, sinkp))?;
                        gemm_batched_h(c, false, hd, cols, l_rows, 1.0, vpw, lda, hd, bp, l_rows, l_rows * cols, aog + (b0 * grow * 4) as u64, hd, b * grow, nkv)?;
                    }
                    b0 += bn;
                }
                let n_ao = (b * nh * hd) as u64;
                let gate_p = if a.gated { gt } else { 0u64 };
                unsafe {
                    c.launch(fx.attn_out, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(aog, bis, gate_p, bao, nh_i, b_i, one))?;
                }
                }
                bst(c, 5, tm)?;
                unsafe { prism_rows(c, plan, bao, bao, nh * hd, nh * hd, b)? };
                match (pl.attn_out.as_deref(), mh) {
                    (_, Some((m, mp, r))) => {
                        mul(&a.o, bao, r.yo, hidden, 1.0, 0.0)?;
                        unsafe { mhc_out(c, fx, hidden, m, mp, r, false, b)? };
                    }
                    (Some(w), None) => {
                        let by = st.scr(c, "hb_yo", b * hidden * 4)?;
                        mul(&a.o, bao, by, hidden, 1.0, 0.0)?;
                        unsafe { sandwich_rows(c, plan, w, bh, by, b)? };
                    }
                    (None, None) => mul(&a.o, bao, bh, hidden, 1.0, 1.0)?,
                }
            }
        }
        let inter = pl.inter;
        let bgu = st.scr(c, "hb_gu", b * 2 * inter * 4)?;
        let bmh = st.scr(c, "hb_mh", b * inter * 2)?;
        unsafe {
            if let Some((m, mp, r)) = mh {
                mhc_in(c, fx, hidden, m, mp, r, true, b)?;
            }
            c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, pl.post_norm.ptr, bxn, hid_i, eps, gemma_i, one, nul, nul, nul, zero_i))?;
        }
        bst(c, 0, tm)?;
        // the shared expert, or the dense FFN: the gate/up pair as one GEMM,
        // its rows in the pair's order
        let dense_pair = pl.moe.as_ref().is_none_or(|m| m.shared);
        let dsh = match &pl.moe {
            Some(_) if dense_pair => st.scr(c, "hb_dsh", b * hidden * 4)?,
            _ => 0,
        };
        if dense_pair {
            unsafe { prism_rows(c, plan, bxn, bxn, hidden, hidden, b)? };
            mul(&pl.gu, bxn, bgu, 2 * inter, 1.0, 0.0)?;
            let tot = (b * inter) as u64;
            let (in_i, bb_i) = (inter as i32, b as i32);
            unsafe {
                let act_i = pl.gelu as i32;
                c.launch(fx.silu_pair, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(bgu, bmh, in_i, bb_i, act_i))?;
            }
            bst(c, 6, tm)?;
            unsafe { prism_rows(c, plan, bmh, bmh, inter, inter, b)? };
            if pl.moe.is_some() {
                mul(&pl.down, bmh, dsh, hidden, 16.0, 0.0)?;
            } else if let Some((m, mp, r)) = mh {
                mul(&pl.down, bmh, r.yo, hidden, 16.0, 0.0)?;
                unsafe { mhc_out(c, fx, hidden, m, mp, r, true, b)? };
            } else if let Some(w) = pl.ffn_out.as_deref() {
                let by = st.scr(c, "hb_yo", b * hidden * 4)?;
                mul(&pl.down, bmh, by, hidden, 16.0, 0.0)?;
                unsafe { sandwich_rows(c, plan, w, bh, by, b)? };
            } else {
                mul(&pl.down, bmh, bh, hidden, 16.0, 1.0)?;
            }
        }
        if let Some(m) = &pl.moe {
            unsafe { moe_grouped(c, st, fx, m, bh, dsh, b, hidden, eps, gemma_i, pl.post_norm.ptr)? };
            bst(c, 6, tm)?;
        }
        if let Some(s) = pl.layer_scale {
            // Gemma-4's layer scalar on the whole residual row
            let tot = (b * hidden) as u64;
            unsafe { c.launch(fx.scale, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(bh, s, tot))? };
        }
        // a Looped Transformer's loop boundary: the final norm on the rows
        if let (true, Some(w)) = (plan.loop_norm.contains(&li), plan.loop_w.as_ref()) {
            let (nullp, zero_h) = (0u64, 0i32);
            unsafe { c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, w.ptr, bh, hid_i, eps, gemma_i, zero_h, nullp, nullp, nullp, zero_h))? };
        }
        if tapping {
            if let Some(s) = want.iter().position(|&l| l == li) {
                // pageable: done when it returns
                unsafe { cu::memcpy_dtoh_async(&mut tapped[s * b * hidden..(s + 1) * b * hidden], bh, c.stream).map_err(derr)? };
            }
        }
    }
    if tapping {
        super::taps_put(tapped);
    }
    if !defer {
        unsafe { cu::memcpy_dtoh_async(h, bh, c.stream).map_err(derr)? };
    }
    let r = c.sync();
    mem_report(c, st, &format!("a prompt chunk of {b} rows at {pos0}"));
    r
}

/// After a verify window accepted rows `..= slot`: every GDN layer of the
/// sequence back to the window's start, the accepted rows replayed.
pub(super) fn spec_restore(kv_id: u64, slot: usize, base_pos: usize, expected: usize) -> bool {
    let Some(c) = super::ctx() else { return false };
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return false;
    }
    let st = &mut *st;
    let mut keys: Vec<(u64, usize)> = st.hyb.gdn.keys().filter(|k| k.0 == kv_id).copied().collect();
    keys.sort_unstable();
    let run = |st: &mut State| -> Res<usize> {
        let Some(fx) = st.hyb.funcs.values().next().cloned() else { return Ok(0) };
        // every layer with this window's tape, of one geometry
        let mut dims = None;
        let mut live = Vec::with_capacity(keys.len());
        for k in &keys {
            let gs = &st.hyb.gdn[k];
            let Some((p0, rows)) = gs.tape else { continue };
            if p0 != base_pos || slot >= rows {
                continue;
            }
            if dims.is_some_and(|d| d != (gs.dims, gs.t_stride)) {
                return Err("GDN layers of different geometry in one restore".into());
            }
            dims = Some((gs.dims, gs.t_stride));
            live.push(*k);
        }
        let Some(((nv, nk, dk, dv, kk), tstride)) = dims else { return Ok(0) };
        let cdim = 2 * nk * dk + nv * dv;
        let keep = slot + 1;
        // the table of the layers' pointers, made again when they change
        let fresh = st.hyb.rtabs.get(&kv_id).is_none_or(|(_, ks)| *ks != live);
        if fresh {
            let mut t = Vec::with_capacity(live.len() * 7);
            for k in &live {
                let gs = &st.hyb.gdn[k];
                t.extend([gs.s_save.ptr, gs.s.ptr, gs.t_cq.ptr, gs.t_gb.ptr, gs.ring_save.ptr, gs.ring.ptr, gs.t_qkv.ptr]);
            }
            st.hyb.rtabs.insert(kv_id, (Buf::from_slice(&t)?, live.clone()));
        }
        let tab = st.hyb.rtabs[&kv_id].0.ptr;
        let nl = live.len() as u32;
        let (keep_i, cd_i, nv_i, nk_i, dk_i, dv_i, kk_i, qs_i) =
            (keep as i32, cdim as i32, nv as i32, nk as i32, dk as i32, dv as i32, kk as i32, tstride as i32);
        unsafe {
            if dk == 128 && dv % 32 == 0 {
                c.launch_3d(fx.rs_scan4, ((nv * dv / 32) as u32, nl, 1), 128, 0, kargs!(tab, keep_i, cd_i, nv_i, nk_i, dv_i))?;
            } else {
                c.launch_3d(fx.rs_scan, (((nv * dv) as u32).div_ceil(8), nl, 1), 256, 0, kargs!(tab, keep_i, cd_i, nv_i, nk_i, dk_i, dv_i))?;
            }
            if kk > 1 {
                c.launch_3d(fx.rs_ring, ((cdim as u32).div_ceil(256), nl, 1), 256, 0, kargs!(tab, keep_i, qs_i, cd_i, kk_i))?;
            }
        }
        for k in &live {
            let gs = st.hyb.gdn.get_mut(k).unwrap();
            gs.next_pos = base_pos + keep;
            gs.tape = None;
        }
        Ok(live.len())
    };
    match run(st) {
        Ok(done) => done == expected,
        Err(e) => {
            tracing::error!("cuda GDN restore failed: {e}");
            false
        }
    }
}

/// The arm's kernels are compiled (a trunk plan was built).
/// A verify window over the far fields of `kv_id` kept `rows` rows: they move
/// into the window in order, each pushing the ring's oldest row into the far
/// sums once the ring is full (`nystrom.rs` advance, as a decode step does).
/// Drafts over a far field (`CMF_O1_SPEC=0` turns them off).
pub(crate) fn far_spec_on() -> bool {
    std::env::var("CMF_O1_SPEC").as_deref() != Ok("0")
}

/// A prompt read past a seal in chunks on the card (`CMF_O1_CHUNKS=0`: none).
pub(crate) fn far_chunks_on() -> bool {
    std::env::var("CMF_O1_CHUNKS").as_deref() != Ok("0")
}

pub(super) fn spec_commit(kv_id: u64, pos0: usize, rows: usize) -> bool {
    let Some(c) = super::ctx() else { return false };
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return false;
    }
    let r = (|| -> Res<()> {
        let mut recs: Vec<i32> = Vec::new();
        // (layer key, record offset, the row to move, the ring slot it takes)
        let mut steps: Vec<((u64, usize), usize, usize, usize)> = Vec::new();
        let mut keys: Vec<(u64, usize)> = st.hyb.o1.keys().filter(|k| k.0 == kv_id).copied().collect();
        keys.sort_unstable();
        for key in keys {
            let d = st.hyb.o1.get_mut(&key).unwrap();
            let Some((base, n)) = d.spec.take() else { continue };
            for j in 0..rows.min(n) {
                let row = base + j;
                if d.win_len < d.w {
                    // the row is where the window's next slot is
                    debug_assert_eq!(row, d.ns + d.win_len);
                    d.win_len += 1;
                    continue;
                }
                let slot = d.ns + d.win_head;
                let keys_n = d.ns + d.w;
                let rb = attn_rb(keys_n);
                let ncap = keys_n.next_multiple_of(4096);
                let at = recs.len();
                recs.extend([(keys_n - 1) as i32, slot as i32, rb as i32, ncap as i32, (ncap.div_ceil(rb) + 1) as i32, 1, slot as i32, 1]);
                steps.push((key, at, row, slot));
                d.far_len += 1;
                d.win_head = (d.win_head + 1) % d.w;
            }
            d.next_pos = Some(pos0 + rows);
        }
        if steps.is_empty() {
            return Ok(());
        }
        let rb = st.scr(c, "hy_spec_ci", recs.len() * 4)?;
        unsafe { cu::memcpy_htod_async(rb, &recs, c.stream).map_err(derr)? };
        for (key, at, row, slot) in steps {
            let d = &st.hyb.o1[&key];
            let fx = st.hyb.funcs.get(&(d.hpk, d.hd)).ok_or("far field commit before the arm's kernels")?;
            let row_b = (d.nkv * d.hd * 2) as u64;
            let cip = rb + (at * 4) as u64;
            let (fm_i, nkv_i) = (d.m as i32, d.nkv as i32);
            unsafe {
                // the oldest row into the far sums, then the kept row in its slot
                c.launch(fx.far_insert, ((d.nkv * d.hpk * d.m).div_ceil(8) as u32, 1), 256, 0, kargs!(d.nk.ptr, d.nv.ptr, d.qt.ptr, d.mz.ptr, d.th.ptr, cip, fm_i, nkv_i, d.scale))?;
                cu::memcpy_dtod_async(d.nk.ptr + slot as u64 * row_b, d.nk.ptr + row as u64 * row_b, row_b as usize, c.stream).map_err(derr)?;
                cu::memcpy_dtod_async(d.nv.ptr + slot as u64 * row_b, d.nv.ptr + row as u64 * row_b, row_b as usize, c.stream).map_err(derr)?;
            }
        }
        Ok(())
    })();
    match r {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("cuda hybrid: far field commit: {e}");
            false
        }
    }
}

pub(super) fn kernels_ready(st: &State) -> bool {
    !st.hyb.funcs.is_empty()
}

/// A recurrent state of this sequence lives on the card.
pub(super) fn state_resident(st: &State, kv_id: u64, layer: usize) -> bool {
    st.hyb.gdn.contains_key(&(kv_id, layer))
}

/// A GDN layer's state back to the host, `[ring | S]` as the host keeps it.
pub(super) fn gdn_state_read(c: &Ctx, st: &State, kv_id: u64, layer: usize) -> Option<Vec<f32>> {
    let gs = st.hyb.gdn.get(&(kv_id, layer))?;
    let (nv, nk, dk, dv, kk) = gs.dims;
    let cdim = 2 * nk * dk + nv * dv;
    let ring_n = (kk - 1) * cdim;
    let mut out = vec![0f32; ring_n + nv * dk * dv];
    unsafe {
        if ring_n > 0 {
            cu::memcpy_dtoh_async(&mut out[..ring_n], gs.ring.ptr, c.stream).ok()?;
        }
        cu::memcpy_dtoh_async(&mut out[ring_n..], gs.s.ptr, c.stream).ok()?;
    }
    c.sync().ok()?;
    Some(out)
}

/// Drop a sequence's recurrent states.
pub(super) fn reset(c: &Ctx, st: &mut State, kv_id: u64) {
    st.hyb.rtabs.remove(&kv_id);
    st.hyb.o1.retain(|&(id, _), d| {
        if id == kv_id {
            c.kv_bytes.fetch_sub(d.bytes, Ordering::Relaxed);
        }
        id != kv_id
    });
    st.hyb.gdn.retain(|&(id, _), g| {
        if id == kv_id {
            let (nv, nk, dk, dv, kk) = g.dims;
            let cdim = 2 * nk * dk + nv * dv;
            let ring_n = (kk - 1) * cdim;
            let s_n = nv * dk * dv;
            c.kv_bytes.fetch_sub(((s_n * 2 + ring_n * 2 + NMAX * (2 * cdim + 2 * nv)) * 4) as u64, Ordering::Relaxed);
            if g.snap.is_some() {
                c.kv_bytes.fetch_sub(((s_n + ring_n.max(1)) * 4) as u64, Ordering::Relaxed);
            }
        }
        id != kv_id
    });
}

/// Copy every GDN state of the sequence, all at prompt position `pos`, to
/// its snapshot (a device-to-device copy on the stream, after the prompt
/// rows before it). False when a layer is elsewhere or none is resident.
pub(super) fn gdn_snapshot(kv_id: u64, pos: usize) -> bool {
    let Some(c) = super::ctx() else { return false };
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return false;
    }
    let run = |st: &mut State| -> Res<bool> {
        let keys: Vec<(u64, usize)> = st.hyb.gdn.keys().filter(|k| k.0 == kv_id).copied().collect();
        if keys.is_empty() || keys.iter().any(|k| st.hyb.gdn[k].next_pos != pos) {
            return Ok(false);
        }
        for k in keys {
            let gs = st.hyb.gdn.get_mut(&k).unwrap();
            let (nv, nk, dk, dv, kk) = gs.dims;
            let ring_n = (kk - 1) * (2 * nk * dk + nv * dv);
            let s_n = nv * dk * dv;
            if gs.snap.is_none() {
                let (bs, br) = (Buf::new(s_n * 4)?, Buf::new(ring_n.max(1) * 4)?);
                c.kv_bytes.fetch_add(((s_n + ring_n.max(1)) * 4) as u64, Ordering::Relaxed);
                gs.snap = Some((0, bs, br));
            }
            let (at, bs, br) = gs.snap.as_mut().unwrap();
            unsafe {
                cu::memcpy_dtod_async(bs.ptr, gs.s.ptr, s_n * 4, c.stream).map_err(derr)?;
                if ring_n > 0 {
                    cu::memcpy_dtod_async(br.ptr, gs.ring.ptr, ring_n * 4, c.stream).map_err(derr)?;
                }
            }
            *at = pos;
        }
        Ok(true)
    };
    run(&mut st).unwrap_or_else(|e| {
        tracing::warn!("cuda GDN snapshot failed: {e}");
        false
    })
}

/// Every GDN state of the sequence back to its snapshot at `pos`; the
/// attention mirrors rewind themselves when the next call starts there.
/// False (nothing changed) unless every layer holds a snapshot at `pos`.
pub(super) fn gdn_snapshot_restore(kv_id: u64, pos: usize) -> bool {
    let Some(c) = super::ctx() else { return false };
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return false;
    }
    let run = |st: &mut State| -> Res<bool> {
        let keys: Vec<(u64, usize)> = st.hyb.gdn.keys().filter(|k| k.0 == kv_id).copied().collect();
        if keys.is_empty() || keys.iter().any(|k| st.hyb.gdn[k].snap.as_ref().is_none_or(|s| s.0 != pos)) {
            return Ok(false);
        }
        for k in keys {
            let gs = st.hyb.gdn.get_mut(&k).unwrap();
            let (nv, nk, dk, dv, kk) = gs.dims;
            let ring_n = (kk - 1) * (2 * nk * dk + nv * dv);
            let s_n = nv * dk * dv;
            let (_, bs, br) = gs.snap.as_ref().unwrap();
            unsafe {
                cu::memcpy_dtod_async(gs.s.ptr, bs.ptr, s_n * 4, c.stream).map_err(derr)?;
                if ring_n > 0 {
                    cu::memcpy_dtod_async(gs.ring.ptr, br.ptr, ring_n * 4, c.stream).map_err(derr)?;
                }
            }
            gs.next_pos = pos;
            gs.tape = None;
        }
        Ok(true)
    };
    run(&mut st).unwrap_or_else(|e| {
        tracing::warn!("cuda GDN snapshot restore failed: {e}");
        false
    })
}

/// y [n][rows] = W x [n][cols] on the card for a q4tp graph weight (the
/// tensor-core kernel, NMAX rows a launch): an MTP draft block's input
/// projection, done where the rest of the draft runs.
pub(super) fn rows_matvec(model: &Arc<CmfModel>, gw: &GraphW, rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32]) -> Res<()> {
    let c = super::ctx().ok_or("no CUDA context")?;
    if gw.kind != 6 || x.len() < n * cols || y.len() < n * rows || cols % 32 != 0 {
        return Err("rows_matvec: a q4tp weight and rows of its width".into());
    }
    let mut st = c.st.lock().unwrap();
    c.current()?;
    let st = &mut *st;
    let m = hmat(c, st, model, gw, rows, cols)?;
    let fx = match st.hyb.funcs.values().next() {
        Some(f) => f.clone(),
        None => return Err("rows_matvec before the first hybrid plan".into()),
    };
    let xin = st.scr(c, "hy_rx", NMAX * cols * 4)?;
    let yout = st.scr(c, "hy_ry", NMAX * rows * 4)?;
    let p0 = st.scr(c, "hy_rp0", NMAX * cols * 2)?;
    let p1 = st.scr(c, "hy_rp1", NMAX * cols * 2)?;
    let p2 = st.scr(c, "hy_rp2", NMAX * cols * 2)?;
    let key = (model.uid(), gw.idx);
    if st.hyb.rowsets.get(&key).is_none_or(|(_, yp)| *yp != yout) {
        let ms = mmaset(&[(&m, yout, 0, rows, rows)])?;
        st.hyb.rowsets.insert(key, (ms, yout));
    }
    let (ms, _) = &st.hyb.rowsets[&key];
    let planes = Planes { p0, p1, p2, stride: cols };
    let mut done = 0;
    while done < n {
        let k = (n - done).min(NMAX);
        unsafe {
            cu::memcpy_htod_async(xin, &x[done * cols..(done + k) * cols], c.stream).map_err(derr)?;
            split(c, &fx, &planes, xin, cols, cols, k)?;
            ms.run(c, &fx, &planes, cols, k)?;
            cu::memcpy_dtoh_async(&mut y[done * rows..(done + k) * rows], yout, c.stream).map_err(derr)?;
        }
        c.sync()?;
        done += k;
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// A q4tp payload of random nibbles, codes and row params.
    fn payload(rows: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
        let (_, codes_off, stride) = cortiq_core::quant::q4tp_sections(rows, cols);
        let mut b: Vec<u8> = (0..codes_off + rows * stride).map(|_| lcg(seed) as u8).collect();
        let (params_off, _, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        for r in 0..rows {
            // finite params: f16 lo in [-12, -4], step in [0, 0.25]
            let lo = cortiq_core::quant::f32_to_f16(-12.0 + 8.0 * (lcg(seed) % 1000) as f32 / 1000.0);
            let st = cortiq_core::quant::f32_to_f16(0.25 * (lcg(seed) % 1000) as f32 / 1000.0);
            b[params_off + r * 4..params_off + r * 4 + 2].copy_from_slice(&lo.to_le_bytes());
            b[params_off + r * 4 + 2..params_off + r * 4 + 4].copy_from_slice(&st.to_le_bytes());
        }
        b
    }

    /// The tile matvec's speed by codec (q1, q4tp), shape, window rows and
    /// launch shape: the numbers behind `MmaSet`'s KS / TB choices
    /// (`bench_tile_matvec -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_tile_matvec() {
        use cortiq_core::quant::f32_to_f16;
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 5u64;
        for (rows, cols) in [(34816usize, 5120usize), (5120, 17408), (10240, 5120), (4096, 2048), (2048, 2048)] {
            let q1b: Vec<u8> = (0..rows * cols / 32)
                .flat_map(|_| {
                    let mut v = f32_to_f16(0.002 + (lcg(&mut seed) % 1000) as f32 * 1e-5).to_le_bytes().to_vec();
                    v.extend((0..4).map(|_| lcg(&mut seed) as u8));
                    v
                })
                .collect();
            let (w1, s1) = repack1_on_card(c, &fx, &[&q1b], rows, rows, cols).unwrap();
            let m1 = DevMat { kind: 3, rows, cols, held: (w1.bytes + s1.bytes) as u64, w: Arc::new(w1.view(0, w1.bytes)), aux0: Some(s1.view(0, s1.bytes)), aux1: None };
            let q4b = payload(rows, cols, &mut seed);
            let (w4, a0, a1) = repack_on_card(c, &fx, &[&q4b], rows, rows, cols).unwrap();
            let m4 = DevMat { kind: 6, rows, cols, held: (w4.bytes + a0.bytes + a1.bytes) as u64, w: Arc::new(w4.view(0, w4.bytes)), aux0: Some(a0.view(0, a0.bytes)), aux1: Some(a1.view(0, a1.bytes)) };
            let x: Vec<f32> = (0..8 * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 5.0).collect();
            let xb = Buf::from_slice(&x).unwrap();
            let pl = |_: ()| Buf::new(8 * cols * 2 + 64).unwrap();
            let (p0, p1, p2) = (pl(()), pl(()), pl(()));
            let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
            let y = Buf::new(8 * rows * 4 + 64).unwrap();
            unsafe { split(c, &fx, &planes, xb.ptr, cols, cols, 8).unwrap() };
            // int8 tiles with a column field (q8_2f), without (q8_row), and a
            // q8_2f gate/up pair (two fields)
            let i8b = |field: bool, seed: &mut u64| {
                let body: Vec<u8> = (0..rows * cols).map(|_| lcg(seed) as u8).collect();
                let rs: Vec<u8> = (0..rows).flat_map(|_| f32_to_f16(0.001).to_le_bytes()).collect();
                let cf: Option<Vec<u8>> = field.then(|| (0..cols).flat_map(|k| f32_to_f16(0.5 + 0.01 * (k % 7) as f32).to_le_bytes()).collect());
                (body, rs, cf)
            };
            let (f8, r8) = (i8b(true, &mut seed), i8b(false, &mut seed));
            // the matrix owns the allocation its scales and fields sit in
            let mk8 = |t: (Buf, Buf, Option<Buf>), kind: u8| DevMat { kind, rows, cols, held: t.0.bytes as u64, aux0: t.2, aux1: Some(t.1), w: Arc::new(t.0) };
            let m8f = mk8(repack8_on_card(c, &fx, &[(&f8.0, &f8.1, f8.2.as_deref())], rows, cols).unwrap(), 5);
            let m8r = mk8(repack8_on_card(c, &fx, &[(&r8.0, &r8.1, None)], rows, cols).unwrap(), 5);
            let half = |v: &[u8], each: usize| v[..rows / 2 * each].to_vec();
            let (gb, gr, gc) = (half(&f8.0, cols), half(&f8.1, 2), f8.2.clone());
            let m8d = mk8(
                repack8_on_card(c, &fx, &[(&gb, &gr, gc.as_deref()), (&gb, &gr, gc.as_deref())], rows, cols).unwrap(),
                15,
            );
            for (name, m) in [("q1", &m1), ("q4tp", &m4), ("q8_2f", &m8f), ("q8_row", &m8r), ("q8_2f-pair", &m8d)] {
                for n in [1usize, 4, 8] {
                    for (ks16, tb1) in [(false, true), (true, true), (false, false), (true, false)] {
                        let mut set = mmaset(&[(m, y.ptr, 0, rows, rows)]).unwrap();
                        set.ks16 = ks16;
                        set.tb1 = tb1;
                        let it = 40;
                        unsafe { set.run(c, &fx, &planes, cols, n).unwrap() };
                        c.sync().unwrap();
                        let t = std::time::Instant::now();
                        for _ in 0..it {
                            unsafe { set.run(c, &fx, &planes, cols, n).unwrap() };
                        }
                        c.sync().unwrap();
                        let us = t.elapsed().as_secs_f64() * 1e6 / it as f64;
                        println!(
                            "BENCH {name} {rows}x{cols} n {n} ks {} tb {}: {us:.1} us, {:.0} GB/s",
                            if ks16 { 16 } else { 8 },
                            if tb1 { 1 } else { 2 },
                            m.held as f64 / us / 1e3
                        );
                    }
                }
            }
        }
    }

    fn down<T: Copy + Default>(c: &Ctx, b: &Buf, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        unsafe { cu::memcpy_dtoh_sync(&mut v, b.ptr) }.unwrap();
        let _ = c;
        v
    }

    /// A seal's far sums on the card (`far_sums`, several tiles of rows) are
    /// the host's replay of the same rows: the keys and values exact in f16,
    /// so only the dot products' order differs.
    #[test]
    fn card_far_sums_match_the_host_replay() {
        let Some(c) = super::super::ctx() else { return };
        let (nkv, hpk, hd, m, w, sink) = (2usize, 2usize, 64usize, 8usize, 32usize, 4usize);
        let t = 9000usize;
        let nh = nkv * hpk;
        let mut seed = 5u64;
        let mut f = || cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16((lcg(&mut seed) % 2000) as f32 / 1000.0 - 1.0));
        let ks: Vec<Vec<f32>> = (0..nkv).map(|_| (0..t * hd).map(|_| f()).collect()).collect();
        let vs: Vec<Vec<f32>> = (0..nkv).map(|_| (0..t * hd).map(|_| f()).collect()).collect();
        let qtr: Vec<f32> = (0..t * nh * hd).map(|_| f()).collect();
        let order: Vec<u32> = (0..t as u32).collect();
        let mut host = Vec::new();
        let mut qt = Vec::new();
        let mut range = 0..0;
        for g in 0..nkv {
            let mut st = crate::nystrom::NystromState::new_group(m, w, sink, hpk);
            range = st.seal_gathered(&ks[g], &vs[g], &qtr, 1, nh, g * hpk, &order, t, hd, hd).unwrap();
            for h in 0..hpk {
                qt.extend_from_slice(st.far_q_tilde(h));
            }
            st.replay_far_idx(&ks[g], &vs[g], &order[range.clone()]);
            host.push(st);
        }
        let (kr, vr): (Vec<&[f32]>, Vec<&[f32]>) = (ks.iter().map(|k| &k[..]).collect(), vs.iter().map(|v| &v[..]).collect());
        let scale = 1.0 / (hd as f32).sqrt();
        let (mz, th) = far_sums(c, nkv, hpk, hd, m, scale, &qt, &order[range], &kr, &vr, hd).unwrap();
        let mut worst = 0f32;
        for (g, st) in host.iter().enumerate() {
            let view = st.device_view();
            for (j, hv) in view.heads.iter().enumerate() {
                let h = g * hpk + j;
                for i in 0..m {
                    let (mm, z) = (mz[h * 2 * m + i], mz[h * 2 * m + m + i]);
                    assert!((mm - hv.m_max[i]).abs() < 1e-4, "largest logit {mm} against {}", hv.m_max[i]);
                    // the sums at the two scales, brought to one
                    let r = (hv.m_max[i] - mm).exp();
                    worst = worst.max((z - hv.z_hat[i] * r).abs() / z);
                    for d in 0..hd {
                        let (a, b) = (th[(h * m + i) * hd + d], hv.t_hat[i * hd + d] * r);
                        worst = worst.max((a - b).abs() / z);
                    }
                }
            }
        }
        assert!(worst < 1e-4, "far sums off the host's by {worst}");
    }

    /// The card's repack (one matrix and a gate/up pair) is the host's,
    /// word for word: the layout the tensor-core kernels read.
    #[test]
    fn card_repack_matches_the_host_repack() {
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 7u64;
        for (rows, cols) in [(48usize, 96usize), (40, 64)] {
            let a = payload(rows, cols, &mut seed);
            let (hn, hc, hl) = repack_q4tp(&a, rows, cols).unwrap();
            let (dn, dc, dl) = repack_on_card(c, &fx, &[&a], rows, rows, cols).unwrap();
            assert_eq!(down::<u32>(c, &dn, hn.len()), hn, "nibbles {rows}x{cols}");
            assert_eq!(down::<u8>(c, &dc, hc.len()), hc, "codes {rows}x{cols}");
            assert_eq!(down::<f32>(c, &dl, hl.len()), hl, "params {rows}x{cols}");
        }
        let (inter, cols) = (24usize, 64usize);
        let g = payload(inter, cols, &mut seed);
        let u = payload(inter, cols, &mut seed);
        let map = |r: usize| (((r & 15) >= 8) as usize, (r >> 4) * 8 + (r & 7));
        let (hn, hc, hl) = repack_rows(&[(&g, inter), (&u, inter)], 2 * inter, cols, &map).unwrap();
        let (dn, dc, dl) = repack_on_card(c, &fx, &[&g, &u], inter, 2 * inter, cols).unwrap();
        assert_eq!(down::<u32>(c, &dn, hn.len()), hn, "pair nibbles");
        assert_eq!(down::<u8>(c, &dc, hc.len()), hc, "pair codes");
        assert_eq!(down::<f32>(c, &dl, hl.len()), hl, "pair params");
    }

    /// q1 matrices as q1 tiles: expanded back they are `dequant_q1`'s
    /// values word for word (one matrix and a gate/up pair), and the
    /// tensor-core matvec over them is the host's product.
    #[test]
    fn card_q1_tiles_match_the_host() {
        use cortiq_core::quant::{dequant_q1, f32_to_f16};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 23u64;
        let q1 = |rows: usize, cols: usize, seed: &mut u64| -> Vec<u8> {
            let mut b = Vec::with_capacity(rows * cols / 32 * 6);
            for _ in 0..rows * cols / 32 {
                b.extend(f32_to_f16(0.002 + (lcg(seed) % 1000) as f32 * 1e-5).to_le_bytes());
                for _ in 0..4 {
                    b.push(lcg(seed) as u8);
                }
            }
            b
        };
        let host = |b: &[u8], rows: usize, cols: usize| -> Vec<f32> {
            let mut w = vec![0f32; rows * cols];
            dequant_q1(b, &mut w);
            w
        };
        let expand = |t: &(Buf, Buf), rows: usize, cols: usize| -> Vec<u16> {
            let o = Buf::new(rows * cols * 2 + 64).unwrap();
            let ntile = rows.div_ceil(16);
            let (r_i, c_i, t0, nt) = (rows as i32, cols as i32, 0i32, ntile as i32);
            let threads = (ntile * (cols / 32) * 32) as u64;
            unsafe { c.launch(fx.expand_q1, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(t.0.ptr, t.1.ptr, o.ptr, r_i, c_i, t0, nt)).unwrap() };
            c.sync().unwrap();
            down::<u16>(c, &o, rows * cols)
        };
        for (rows, cols) in [(40usize, 512usize), (2048, 2048), (6144, 2048)] {
            let b = q1(rows, cols, &mut seed);
            let w = host(&b, rows, cols);
            let t = repack1_on_card(c, &fx, &[&b], rows, rows, cols).unwrap();
            let got = expand(&t, rows, cols);
            for i in 0..rows * cols {
                assert_eq!(got[i], f32_to_f16(w[i]), "{rows}x{cols} at {},{}", i / cols, i % cols);
            }
            for n in [1usize, 11] {
                let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 5.0).collect();
                let xb = Buf::from_slice(&x).unwrap();
                let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                let y = Buf::new(n * rows * 4 + 64).unwrap();
                let dm = DevMat { kind: 3, rows, cols, held: 0, w: Arc::new(t.0.view(0, t.0.bytes)), aux0: Some(t.1.view(0, t.1.bytes)), aux1: None };
                let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                unsafe {
                    split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                    set.run(c, &fx, &planes, cols, n).unwrap();
                }
                c.sync().unwrap();
                let yd = down::<f32>(c, &y, n * rows);
                for j in 0..n {
                    for r in 0..rows {
                        let (mut s, mut a) = (0f64, 0f64);
                        for k in 0..cols {
                            let p = w[r * cols + k] as f64 * x[j * cols + k] as f64;
                            s += p;
                            a += p.abs();
                        }
                        let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                        assert!(e < 1e-4, "{rows}x{cols} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                    }
                }
            }
        }
        let (inter, cols) = (24usize, 64usize);
        let (g, u) = (q1(inter, cols, &mut seed), q1(inter, cols, &mut seed));
        let (wg, wu) = (host(&g, inter, cols), host(&u, inter, cols));
        let t = repack1_on_card(c, &fx, &[&g, &u], inter, 2 * inter, cols).unwrap();
        let got = expand(&t, 2 * inter, cols);
        for r in 0..2 * inter {
            let (src, sr) = if (r & 15) >= 8 { (&wu, (r >> 4) * 8 + (r & 7)) } else { (&wg, (r >> 4) * 8 + (r & 7)) };
            for k in 0..cols {
                assert_eq!(got[r * cols + k], f32_to_f16(src[sr * cols + k]), "pair at {r},{k}");
            }
        }
    }

    /// q1t matrices (no outlier rows) as ternary tiles: expanded back they
    /// are `dequant_q1t`'s values word for word (one matrix and a gate/up
    /// pair), and the tensor-core matvec over them is the host's product.
    #[test]
    fn card_q1t_tiles_match_the_host() {
        use cortiq_core::quant::{dequant_q1t, f32_to_f16, q1t_pack};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 31u64;
        let q1t = |rows: usize, cols: usize, seed: &mut u64| -> Vec<u8> {
            let mut b = Vec::with_capacity(rows * cols / 32 * 9 + (rows + 1) * 4);
            for _ in 0..rows * cols / 32 {
                b.extend(f32_to_f16(0.002 + (lcg(seed) % 1000) as f32 * 1e-5).to_le_bytes());
                let mut codes = [0u8; 7];
                for k in 0..32 {
                    q1t_pack(&mut codes, k, (lcg(seed) % 3) as u8);
                }
                b.extend(codes);
            }
            b.extend(std::iter::repeat_n(0u8, (rows + 1) * 4));
            b
        };
        let host = |b: &[u8], rows: usize, cols: usize| -> Vec<f32> {
            let mut w = vec![0f32; rows * cols];
            dequant_q1t(b, rows, cols, &mut w);
            w
        };
        let expand = |t: &(Buf, Buf), rows: usize, cols: usize| -> Vec<u16> {
            let o = Buf::new(rows * cols * 2 + 64).unwrap();
            let ntile = rows.div_ceil(16);
            let (r_i, c_i, t0, nt) = (rows as i32, cols as i32, 0i32, ntile as i32);
            let threads = (ntile * (cols / 32) * 32) as u64;
            unsafe { c.launch(fx.expand_q1t, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(t.0.ptr, t.1.ptr, o.ptr, r_i, c_i, t0, nt)).unwrap() };
            c.sync().unwrap();
            down::<u16>(c, &o, rows * cols)
        };
        for (rows, cols) in [(40usize, 512usize), (4096, 4096)] {
            let b = q1t(rows, cols, &mut seed);
            let w = host(&b, rows, cols);
            let base = q1t_base(&b, rows, cols).unwrap();
            let t = repack1x_on_card(c, &fx, &[base], rows, rows, cols, 2).unwrap();
            let got = expand(&t, rows, cols);
            for i in 0..rows * cols {
                assert_eq!(got[i], f32_to_f16(w[i]), "{rows}x{cols} at {},{}", i / cols, i % cols);
            }
            for n in [1usize, 11] {
                let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 5.0).collect();
                let xb = Buf::from_slice(&x).unwrap();
                let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                let y = Buf::new(n * rows * 4 + 64).unwrap();
                let dm = DevMat { kind: 2, rows, cols, held: 0, w: Arc::new(t.0.view(0, t.0.bytes)), aux0: Some(t.1.view(0, t.1.bytes)), aux1: None };
                let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                unsafe {
                    split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                    set.run(c, &fx, &planes, cols, n).unwrap();
                }
                c.sync().unwrap();
                let yd = down::<f32>(c, &y, n * rows);
                for j in 0..n {
                    for r in 0..rows {
                        let (mut s, mut a) = (0f64, 0f64);
                        for k in 0..cols {
                            let p = w[r * cols + k] as f64 * x[j * cols + k] as f64;
                            s += p;
                            a += p.abs();
                        }
                        let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                        assert!(e < 1e-4, "{rows}x{cols} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                    }
                }
            }
        }
        let (inter, cols) = (24usize, 64usize);
        let (g, u) = (q1t(inter, cols, &mut seed), q1t(inter, cols, &mut seed));
        let (wg, wu) = (host(&g, inter, cols), host(&u, inter, cols));
        let t = repack1x_on_card(c, &fx, &[q1t_base(&g, inter, cols).unwrap(), q1t_base(&u, inter, cols).unwrap()], inter, 2 * inter, cols, 2).unwrap();
        let got = expand(&t, 2 * inter, cols);
        for r in 0..2 * inter {
            let (src, sr) = if (r & 15) >= 8 { (&wu, (r >> 4) * 8 + (r & 7)) } else { (&wg, (r >> 4) * 8 + (r & 7)) };
            for k in 0..cols {
                assert_eq!(got[r * cols + k], f32_to_f16(src[sr * cols + k]), "pair at {r},{k}");
            }
        }
        // an overlay with outliers is refused
        let mut o = q1t(16, 64, &mut seed);
        let n = o.len();
        o[n - 4] = 1;
        assert!(q1t_base(&o, 16, 64).is_none());
    }

    /// q2tp matrices (mid-rise and affine) as q2tp tiles: expanded back they
    /// are the host's f16 of (c - 1.5) s / (c - 1) s word for word, and the
    /// tensor-core matvec over them is the host's product.
    #[test]
    fn card_q2tp_tiles_match_the_host() {
        use cortiq_core::quant::{f32_to_f16, q2tp_ladder, q2tp_sections, q4tp_code};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 41u64;
        let q2 = |rows: usize, cols: usize, seed: &mut u64| -> Vec<u8> {
            let (po, co, stride) = q2tp_sections(rows, cols);
            let mut b: Vec<u8> = (0..co + rows * stride).map(|_| lcg(seed) as u8).collect();
            for r in 0..rows {
                b[po + r * 4..po + r * 4 + 2].copy_from_slice(&f32_to_f16(-9.0 + (r % 5) as f32 * 0.1).to_le_bytes());
                b[po + r * 4 + 2..po + r * 4 + 4].copy_from_slice(&f32_to_f16(0.05).to_le_bytes());
            }
            b
        };
        let host = |b: &[u8], rows: usize, cols: usize, aff: bool| -> Vec<f32> {
            let (po, co, stride) = q2tp_sections(rows, cols);
            let gpr = cols / 32;
            let mut w = vec![0f32; rows * cols];
            for r in 0..rows {
                let tab = q2tp_ladder(&b[po..po + rows * 4], r);
                for g in 0..gpr {
                    let sc = tab[q4tp_code(&b[co + r * stride..co + (r + 1) * stride], g)];
                    for k in 0..32 {
                        let cv = ((b[(r * gpr + g) * 8 + k / 4] >> (2 * (k % 4))) & 3) as f32;
                        w[r * cols + g * 32 + k] = (cv - if aff { 1.0 } else { 1.5 }) * sc;
                    }
                }
            }
            w
        };
        let expand = |t: &(Buf, Buf, Buf), rows: usize, cols: usize, aff: bool| -> Vec<u16> {
            let o = Buf::new(rows * cols * 2 + 64).unwrap();
            let ntile = rows.div_ceil(16);
            let (r_i, c_i, t0, nt, a_i) = (rows as i32, cols as i32, 0i32, ntile as i32, aff as i32);
            let threads = (ntile * (cols / 32) * 32) as u64;
            unsafe { c.launch(fx.expand_q2, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(t.0.ptr, t.1.ptr, t.2.ptr, o.ptr, r_i, c_i, t0, nt, a_i)).unwrap() };
            c.sync().unwrap();
            down::<u16>(c, &o, rows * cols)
        };
        for (rows, cols) in [(40usize, 512usize), (2048, 2048)] {
            let b = q2(rows, cols, &mut seed);
            let t = repack2_on_card(c, &fx, &[&b], rows, rows, cols).unwrap();
            for aff in [false, true] {
                let w = host(&b, rows, cols, aff);
                let got = expand(&t, rows, cols, aff);
                for i in 0..rows * cols {
                    assert_eq!(got[i], f32_to_f16(w[i]), "{rows}x{cols} aff {aff} at {},{}", i / cols, i % cols);
                }
                for n in [1usize, 11] {
                    let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 5.0).collect();
                    let xb = Buf::from_slice(&x).unwrap();
                    let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                    let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                    let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                    let y = Buf::new(n * rows * 4 + 64).unwrap();
                    let kind = if aff { 9 } else { 1 };
                    let dm = DevMat {
                        kind,
                        rows,
                        cols,
                        held: 0,
                        w: Arc::new(t.0.view(0, t.0.bytes)),
                        aux0: Some(t.1.view(0, t.1.bytes)),
                        aux1: Some(t.2.view(0, t.2.bytes)),
                    };
                    let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                    unsafe {
                        split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                        set.run(c, &fx, &planes, cols, n).unwrap();
                    }
                    c.sync().unwrap();
                    let yd = down::<f32>(c, &y, n * rows);
                    for j in 0..n {
                        for r in 0..rows {
                            let (mut s, mut a) = (0f64, 0f64);
                            for k in 0..cols {
                                let p = w[r * cols + k] as f64 * x[j * cols + k] as f64;
                                s += p;
                                a += p.abs();
                            }
                            let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                            assert!(e < 1e-4, "{rows}x{cols} aff {aff} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                        }
                    }
                }
            }
        }
    }

    /// The Prism boundary on planes and on f16 rows is the host's
    /// `signed_fwht_forward` with the f16 cast, word for word.
    #[test]
    fn card_prism_matches_the_host() {
        use cortiq_core::hadamard::{prism_f32_to_f16_rne, signed_fwht_forward};
        use cortiq_core::quant::{f16_to_f32, f32_to_f16};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 53u64;
        let (n, width, blk) = (3usize, 3072usize, 1024usize);
        let signs: Vec<f32> = (0..width).map(|_| if lcg(&mut seed) & 1 == 1 { 1.0 } else { -1.0 }).collect();
        let sb = Buf::from_slice(&signs).unwrap();
        let x: Vec<f32> = (0..n * width).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 3.0).collect();
        // planes: the host takes their sum as the input
        let xb = Buf::from_slice(&x).unwrap();
        let pl = |_: ()| Buf::new(n * width * 2 + 64).unwrap();
        let (p0, p1, p2) = (pl(()), pl(()), pl(()));
        let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: width };
        unsafe { split(c, &fx, &planes, xb.ptr, width, width, n).unwrap() };
        c.sync().unwrap();
        let bf = |h: u16| f32::from_bits((h as u32) << 16);
        let (a0, a1) = (down::<u16>(c, &p0, n * width), down::<u16>(c, &p1, n * width));
        let mut want = Vec::with_capacity(n * width);
        for j in 0..n {
            let mut row: Vec<f32> = (0..width).map(|i| bf(a0[j * width + i]) + bf(a1[j * width + i])).collect();
            signed_fwht_forward(&mut row, &signs, blk).unwrap();
            want.extend(row.iter().map(|&v| f16_to_f32(prism_f32_to_f16_rne(v))));
        }
        let (st_i, blk_i, f_i) = (width as i32, blk as i32, 1i32);
        unsafe { c.launch(fx.prism_p, ((width / blk) as u32, n as u32), 256, blk * 4, kargs!(p0.ptr, p1.ptr, st_i, sb.ptr, blk_i, f_i)).unwrap() };
        c.sync().unwrap();
        let (b0, b1) = (down::<u16>(c, &p0, n * width), down::<u16>(c, &p1, n * width));
        for i in 0..n * width {
            assert_eq!(bf(b0[i]) + bf(b1[i]), want[i], "planes at {i}");
        }
        // f16 rows, in place
        let h: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
        let hb = Buf::from_slice(&h).unwrap();
        let mut want = Vec::with_capacity(n * width);
        for j in 0..n {
            let mut row: Vec<f32> = (0..width).map(|i| f16_to_f32(h[j * width + i])).collect();
            signed_fwht_forward(&mut row, &signs, blk).unwrap();
            want.extend(row.iter().map(|&v| prism_f32_to_f16_rne(v)));
        }
        unsafe { c.launch(fx.prism_h, ((width / blk) as u32, n as u32), 256, blk * 4, kargs!(hb.ptr, hb.ptr, st_i, sb.ptr, blk_i, f_i)).unwrap() };
        c.sync().unwrap();
        assert_eq!(down::<u16>(c, &hb, n * width), want, "f16 rows");
    }

    /// q4_tiled matrices as q4_tiled tiles: expanded back they are
    /// `dequant_q4_tiled`'s values word for word (one matrix and a gate/up
    /// pair), and the tensor-core matvec over them is the host's product.
    #[test]
    fn card_q4_tiled_tiles_match_the_host() {
        use cortiq_core::quant::{dequant_q4_tiled, f32_to_f16};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 61u64;
        let q4 = |rows: usize, cols: usize, seed: &mut u64| -> Vec<u8> {
            let mut b = Vec::with_capacity(rows * cols / 32 * 18);
            for _ in 0..rows * cols / 32 {
                b.extend(f32_to_f16(0.002 + (lcg(seed) % 1000) as f32 * 1e-5).to_le_bytes());
                for _ in 0..16 {
                    b.push(lcg(seed) as u8);
                }
            }
            b
        };
        let host = |b: &[u8], rows: usize, cols: usize| -> Vec<f32> {
            let mut w = vec![0f32; rows * cols];
            dequant_q4_tiled(b, &mut w);
            w
        };
        let expand = |t: &(Buf, Buf), rows: usize, cols: usize| -> Vec<u16> {
            let o = Buf::new(rows * cols * 2 + 64).unwrap();
            let ntile = rows.div_ceil(16);
            let (r_i, c_i, t0, nt) = (rows as i32, cols as i32, 0i32, ntile as i32);
            let threads = (ntile * (cols / 32) * 32) as u64;
            unsafe { c.launch(fx.expand_q4g, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(t.0.ptr, t.1.ptr, o.ptr, r_i, c_i, t0, nt)).unwrap() };
            c.sync().unwrap();
            down::<u16>(c, &o, rows * cols)
        };
        for (rows, cols) in [(40usize, 512usize), (2048, 2048)] {
            let b = q4(rows, cols, &mut seed);
            let w = host(&b, rows, cols);
            let t = repack1x_on_card(c, &fx, &[&b], rows, rows, cols, 10).unwrap();
            let got = expand(&t, rows, cols);
            for i in 0..rows * cols {
                assert_eq!(got[i], f32_to_f16(w[i]), "{rows}x{cols} at {},{}", i / cols, i % cols);
            }
            for n in [1usize, 11] {
                let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 5.0).collect();
                let xb = Buf::from_slice(&x).unwrap();
                let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                let y = Buf::new(n * rows * 4 + 64).unwrap();
                let dm = DevMat { kind: 10, rows, cols, held: 0, w: Arc::new(t.0.view(0, t.0.bytes)), aux0: Some(t.1.view(0, t.1.bytes)), aux1: None };
                let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                unsafe {
                    split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                    set.run(c, &fx, &planes, cols, n).unwrap();
                }
                c.sync().unwrap();
                let yd = down::<f32>(c, &y, n * rows);
                for j in 0..n {
                    for r in 0..rows {
                        let (mut s, mut a) = (0f64, 0f64);
                        for k in 0..cols {
                            let p = w[r * cols + k] as f64 * x[j * cols + k] as f64;
                            s += p;
                            a += p.abs();
                        }
                        let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                        assert!(e < 1e-4, "{rows}x{cols} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                    }
                }
            }
        }
        let (inter, cols) = (24usize, 64usize);
        let (g, u) = (q4(inter, cols, &mut seed), q4(inter, cols, &mut seed));
        let (wg, wu) = (host(&g, inter, cols), host(&u, inter, cols));
        let t = repack1x_on_card(c, &fx, &[&g, &u], inter, 2 * inter, cols, 10).unwrap();
        let got = expand(&t, 2 * inter, cols);
        for r in 0..2 * inter {
            let (src, sr) = if (r & 15) >= 8 { (&wu, (r >> 4) * 8 + (r & 7)) } else { (&wg, (r >> 4) * 8 + (r & 7)) };
            for k in 0..cols {
                assert_eq!(got[r * cols + k], f32_to_f16(src[sr * cols + k]), "pair at {r},{k}");
            }
        }
    }

    /// int8 matrices (q8_2f with its column field, q8_row without) as int8
    /// tiles: expanded back they are the host's f16 of int8 * row scale *
    /// field, word for word (one matrix and a q8_row gate/up pair), and the
    /// tensor-core matvec over them is the host's product.
    #[test]
    fn card_int8_tiles_match_the_host() {
        use cortiq_core::quant::{f16_to_f32, f32_to_f16};
        let Some(c) = super::super::ctx() else { return };
        c.current().unwrap();
        let fx = load_funcs(c, 4, 64).unwrap();
        let mut seed = 11u64;
        type Q8 = (Vec<u8>, Vec<u8>, Option<Vec<u8>>);
        let q8 = |rows: usize, cols: usize, field: bool, seed: &mut u64| -> Q8 {
            let body: Vec<u8> = (0..rows * cols).map(|_| lcg(seed) as u8).collect();
            let h = |v: f32| f32_to_f16(v).to_le_bytes();
            let rs: Vec<u8> = (0..rows).flat_map(|r| h(0.001 + 0.0001 * (r % 7) as f32)).collect();
            let fo = (lcg(seed) % 7) as usize;
            let cf = field.then(|| (0..cols).flat_map(|k| h(0.5 + 0.03 * ((k + fo) % 11) as f32)).collect());
            (body, rs, cf)
        };
        // the weight as the card's expand makes it: (int8 * rs) * cf in f32
        let w = |m: &Q8, r: usize, k: usize, cols: usize| -> f32 {
            let f = |b: &[u8], i: usize| f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
            let cf = m.2.as_ref().map_or(1.0, |b| f(b, k));
            (m.0[r * cols + k] as i8) as f32 * f(&m.1, r) * cf
        };
        let expand = |t: &(Buf, Buf, Option<Buf>), rows: usize, cols: usize| -> Vec<u16> {
            let o = Buf::new(rows * cols * 2 + 64).unwrap();
            let ntile = rows.div_ceil(16);
            let (r_i, c_i, t0, nt) = (rows as i32, cols as i32, 0i32, ntile as i32);
            let cf = t.2.as_ref().map_or(0, |b| b.ptr);
            // two fields (a q8_2f pair's): the up rows under the second
            let dual_i = t.2.as_ref().is_some_and(|b| b.bytes == 2 * cols * 4) as i32;
            let threads = (ntile * (cols / 32) * 32) as u64;
            unsafe { c.launch(fx.expand_i8, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(t.0.ptr, t.1.ptr, cf, o.ptr, r_i, c_i, t0, nt, dual_i)).unwrap() };
            c.sync().unwrap();
            down::<u16>(c, &o, rows * cols)
        };
        fn parts(m: &(Vec<u8>, Vec<u8>, Option<Vec<u8>>)) -> (&[u8], &[u8], Option<&[u8]>) {
            (m.0.as_slice(), m.1.as_slice(), m.2.as_deref())
        }
        // the last three: Mellum2.1's q, k and o shapes (two tiles a block past 2048 rows)
        for (rows, cols, field) in [(40usize, 512usize, true), (48, 96, false), (4096, 2304, true), (512, 2304, true), (2304, 4096, true)] {
            let m = q8(rows, cols, field, &mut seed);
            let t = repack8_on_card(c, &fx, &[parts(&m)], rows, cols).unwrap();
            let got = expand(&t, rows, cols);
            for r in 0..rows {
                for k in 0..cols {
                    assert_eq!(got[r * cols + k], f32_to_f16(w(&m, r, k, cols)), "{rows}x{cols} at {r},{k}");
                }
            }
            // y = W x over 1, 3 and 11 rows through the tensor-core kernel
            for n in [1usize, 3, 11] {
                let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 37.0).collect();
                let xb = Buf::from_slice(&x).unwrap();
                let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                let y = Buf::new(n * rows * 4 + 64).unwrap();
                let (tw, trs, tcf) = (t.0.view(0, t.0.bytes), t.1.view(0, t.1.bytes), t.2.as_ref().map(|b| b.view(0, b.bytes)));
                let dm = DevMat { kind: 5, rows, cols, held: 0, w: Arc::new(tw), aux0: tcf, aux1: Some(trs) };
                let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                unsafe {
                    split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                    set.run(c, &fx, &planes, cols, n).unwrap();
                }
                c.sync().unwrap();
                let yd = down::<f32>(c, &y, n * rows);
                for j in 0..n {
                    for r in 0..rows {
                        let (mut s, mut a) = (0f64, 0f64);
                        for k in 0..cols {
                            let p = w(&m, r, k, cols) as f64 * x[j * cols + k] as f64;
                            s += p;
                            a += p.abs();
                        }
                        let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                        assert!(e < 1e-4, "{rows}x{cols} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                    }
                }
            }
        }
        // a q8_row gate/up pair: 8 gate rows, then the same 8 up rows, a tile
        let (inter, cols) = (24usize, 64usize);
        let g = q8(inter, cols, false, &mut seed);
        let u = q8(inter, cols, false, &mut seed);
        let t = repack8_on_card(c, &fx, &[parts(&g), parts(&u)], 2 * inter, cols).unwrap();
        let got = expand(&t, 2 * inter, cols);
        for r in 0..2 * inter {
            let (src, sr) = if (r & 15) >= 8 { (&u, (r >> 4) * 8 + (r & 7)) } else { (&g, (r >> 4) * 8 + (r & 7)) };
            for k in 0..cols {
                assert_eq!(got[r * cols + k], f32_to_f16(w(src, sr, k, cols)), "pair at {r},{k}");
            }
        }
        // a q8_2f pair: two column fields, the up rows under their own
        // (kind 15), expanded and through the tensor-core kernel (1 and 3 rows)
        for (inter, cols) in [(24usize, 64usize), (1024, 2048)] {
            let g2 = q8(inter, cols, true, &mut seed);
            let u2 = q8(inter, cols, true, &mut seed);
            let rows = 2 * inter;
            let t = repack8_on_card(c, &fx, &[parts(&g2), parts(&u2)], rows, cols).unwrap();
            let src = |r: usize| if (r & 15) >= 8 { (&u2, (r >> 4) * 8 + (r & 7)) } else { (&g2, (r >> 4) * 8 + (r & 7)) };
            let got = expand(&t, rows, cols);
            for r in 0..rows {
                let (m, sr) = src(r);
                for k in 0..cols {
                    assert_eq!(got[r * cols + k], f32_to_f16(w(m, sr, k, cols)), "q8_2f pair at {r},{k}");
                }
            }
            for n in [1usize, 3] {
                let x: Vec<f32> = (0..n * cols).map(|_| ((lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 37.0).collect();
                let xb = Buf::from_slice(&x).unwrap();
                let pl = |_: ()| Buf::new(n * cols * 2 + 64).unwrap();
                let (p0, p1, p2) = (pl(()), pl(()), pl(()));
                let planes = Planes { p0: p0.ptr, p1: p1.ptr, p2: p2.ptr, stride: cols };
                let y = Buf::new(n * rows * 4 + 64).unwrap();
                let (tw, trs, tcf) = (t.0.view(0, t.0.bytes), t.1.view(0, t.1.bytes), t.2.as_ref().map(|b| b.view(0, b.bytes)));
                let dm = DevMat { kind: 15, rows, cols, held: 0, w: Arc::new(tw), aux0: tcf, aux1: Some(trs) };
                let set = mmaset(&[(&dm, y.ptr, 0, rows, rows)]).unwrap();
                unsafe {
                    split(c, &fx, &planes, xb.ptr, cols, cols, n).unwrap();
                    set.run(c, &fx, &planes, cols, n).unwrap();
                }
                c.sync().unwrap();
                let yd = down::<f32>(c, &y, n * rows);
                for j in 0..n {
                    for r in 0..rows {
                        let (m, sr) = src(r);
                        let (mut s, mut a) = (0f64, 0f64);
                        for k in 0..cols {
                            let p = w(m, sr, k, cols) as f64 * x[j * cols + k] as f64;
                            s += p;
                            a += p.abs();
                        }
                        let e = (yd[j * rows + r] as f64 - s).abs() / a.max(1e-30);
                        assert!(e < 1e-4, "q8_2f pair {rows}x{cols} n {n} y[{j}][{r}] {} against {s} ({e})", yd[j * rows + r]);
                    }
                }
            }
        }
    }
}
