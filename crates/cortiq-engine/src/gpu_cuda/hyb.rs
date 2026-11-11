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
/// Rows of the small path.
pub(super) const NMAX: usize = 16;
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
}

pub(super) struct HFuncs {
    /// `hy_mmp_{NT}_{TB}_{KS}` by [NT - 1][TB - 1][KS == 16].
    mmp: [[[cs::CUfunction; 2]; 2]; 2],
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
    silu_pair: cs::CUfunction,
    repack: cs::CUfunction,
    qk: cs::CUfunction,
    scores: cs::CUfunction,
    values: cs::CUfunction,
    finish: cs::CUfunction,
    attn_out: cs::CUfunction,
}
unsafe impl Send for HFuncs {}
unsafe impl Sync for HFuncs {}

fn load_funcs(c: &Ctx, hpk: usize, hd: usize) -> Res<HFuncs> {
    if hd % 32 != 0 || hd > 512 || hpk == 0 || hpk > 16 {
        return Err(format!("attention geometry {hpk} heads per KV head x {hd} outside the kernels"));
    }
    let module = compile(HYB_SRC, vec![format!("-DHPK={hpk}"), format!("-DHD={hd}")], c.arch)?;
    let mut mmp = [[[std::ptr::null_mut(); 2]; 2]; 2];
    for nt in 1..=2usize {
        for tb in 1..=2usize {
            for (ki, ks) in [8usize, 16].into_iter().enumerate() {
                mmp[nt - 1][tb - 1][ki] = func(module, &format!("hy_mmp_{nt}_{tb}_{ks}"))?;
            }
        }
    }
    Ok(HFuncs {
        mmp,
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
        silu_pair: func(module, "hy_silu_pair_half")?,
        repack: func(module, "hy_repack")?,
        qk: func(module, "hy_qk")?,
        scores: func(module, "hy_scores")?,
        values: func(module, "hy_values")?,
        finish: func(module, "hy_finish")?,
        attn_out: func(module, "hy_attn_out")?,
    })
}

/// (matrix, output, residual or 0, output row stride, rows to compute)
type JobSpec<'a> = (&'a DevMat, u64, u64, usize, usize);

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
    if list.iter().any(|(m, ..)| m.kind != 6 || m.cols % 32 != 0) {
        return Err("a tensor-core job over a matrix that is not repacked q4tp".into());
    }
    let max_rows = list.iter().map(|&(_, _, _, _, r)| r).max().unwrap_or(0);
    let mut bufs = Vec::with_capacity(2);
    let mut blocks = [0u32; 2];
    for tb in 1..=2usize {
        let mut jobs = Vec::with_capacity(list.len());
        let mut blk = 0u32;
        for &(m, y, res, ystride, rows) in list {
            jobs.push(HyJob {
                w: m.w.ptr,
                aux0: m.aux0(),
                aux1: m.aux1(),
                y,
                res,
                rows: rows as i32,
                kind,
                blk0: blk as i32,
                ystride: ystride as i32,
            });
            blk += rows.div_ceil(16).div_ceil(tb) as u32;
        }
        blocks[tb - 1] = blk;
        bufs.push(Buf::from_slice(&jobs)?);
    }
    let b1 = bufs.pop().unwrap();
    let b0 = bufs.pop().unwrap();
    // measured on a 3090 (q4tp, 1/8/16 rows): KS 8 wins from ~16k rows
    // (gate/up 67.5/71.6/124 us against 69.4/76.3/146), KS 16 below it
    // (down 5120 x 17408: 61.1/66.8/113 against 76.3/79.3/117)
    Ok(MmaSet { bufs: [b0, b1], blocks, n: list.len() as i32, ks16: max_rows < 16384, tb1: max_rows <= 2048 })
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
        let nt = if n <= 8 { 1 } else { 2 };
        let tb = if self.tb1 { 1 } else { 2 };
        let ks = if self.ks16 { 16 } else { 8 };
        let f = fx.mmp[nt - 1][tb - 1][self.ks16 as usize];
        let (nj, xs_i, n_i, cols_i) = (self.n, p.stride as i32, n as i32, cols as i32);
        unsafe {
            c.launch(
                f,
                (self.blocks[tb - 1], 1),
                (32 * ks) as u32,
                0,
                kargs!(self.bufs[tb - 1].ptr, nj, p.p0, p.p1, p.p2, xs_i, n_i, cols_i),
            )
        }
    }
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
    let t = Buf::new(ntile * gpr * 64 * 4)?;
    let cc = Buf::new(ntile * gpr * 16)?;
    let (stride_i, pair_i, rows_i, cols_i) = (stride as i32, pair as i32, rows as i32, cols as i32);
    let threads = (ntile * gpr * 32) as u64;
    unsafe {
        c.launch(fx.repack, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(na.ptr, ca.ptr, nb_p, cb_p, stride_i, pair_i, t.ptr, cc.ptr, rows_i, cols_i))?;
    }
    // the stored planes are freed on return: the repack must be done
    c.sync()?;
    Ok((t, cc, Buf::from_slice(&lad)?))
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
    if gw.kind != 6 {
        if gw.kind == 4 {
            return st.mat(c, model, gw, rows, cols);
        }
        return Err(format!("weight codec kind {} has no hybrid kernel", gw.kind));
    }
    if gw.prism != crate::gpu::GraphPrismOp::None || gw.affine {
        return Err("Prism / affine weight".into());
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
    let (w, aux0, aux1) = repack_on_card(c, &fx, &[bytes], rows, rows, cols)?;
    let m = DevMat { kind: 6, rows, cols, w: Arc::new(w), aux0: Some(aux0), aux1: Some(aux1), held: bytes.len() as u64 };
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
    use cortiq_core::quant::GROUP_SIZE;
    if gw.kind != 6 || uw.kind != 6 || inter % 8 != 0 || cols % GROUP_SIZE != 0 {
        return Err("a gate/up pair of q4tp matrices with rows a multiple of 8".into());
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
    let (w, aux0, aux1) = repack_on_card(c, &fx, &[gb, ub], inter, rows, cols)?;
    let m = DevMat { kind: 6, rows, cols, w: Arc::new(w), aux0: Some(aux0), aux1: Some(aux1), held: (gb.len() + ub.len()) as u64 };
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

struct AttnL {
    q: Arc<DevMat>,
    k: Arc<DevMat>,
    v: Arc<DevMat>,
    o: Arc<DevMat>,
    qn: Option<Arc<Buf>>,
    kn: Option<Arc<Buf>>,
    gated: bool,
    ms_in: MmaSet,
    ms_out: MmaSet,
}

enum Mix {
    Gdn(GdnL),
    Attn(AttnL),
}

struct HLayer {
    mix: Mix,
    /// gate and up as one pair (`hmat_pair`)
    gu: Arc<DevMat>,
    down: Arc<DevMat>,
    in_norm: Arc<Buf>,
    post_norm: Arc<Buf>,
    inter: usize,
    ms_gu: MmaSet,
    ms_dn: MmaSet,
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
    lm: Option<(Arc<DevMat>, usize, MmaSet)>,
    final_norm: Option<Arc<Buf>>,
    invf: Vec<f32>,
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
}

/// One GDN layer's recurrent state on the card, with what a verify window
/// needs to step back.
struct GdnState {
    dims: (usize, usize, usize, usize, usize),
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
}

#[derive(Default)]
pub(super) struct HState {
    funcs: HashMap<(usize, usize), Arc<HFuncs>>,
    /// q4tp weights in the tensor-core layout.
    mats: HashMap<(u64, usize), Arc<DevMat>>,
    /// Gate/up pairs (`hmat_pair`) by (model, gate, up).
    pairs: HashMap<(u64, usize, usize), Arc<DevMat>>,
    /// `rows_matvec`'s job tables, by tensor, with the output they point at.
    rowsets: HashMap<(u64, usize), (MmaSet, u64)>,
    /// The restore tables of `spec_restore`, by sequence: the layers' state
    /// pointers (7 a layer) and the keys they were made for.
    rtabs: HashMap<u64, (Buf, Vec<(u64, usize)>)>,
    /// Recorded small-path graphs by (plan, sequence, layer base, rows, tape,
    /// head).
    graphs: HashMap<(u64, u64, usize, usize, bool, bool), GraphRec>,
    plans: HashMap<(u64, usize, usize, bool), Arc<HPlan>>,
    gdn: HashMap<(u64, usize), GdnState>,
}

/// Does this stack belong to the hybrid arm? (A GDN layer, or full attention
/// with Q/K norms or the element-wise output gate.) An empty stack with a
/// head goes here when the dense arm's attention kernels cannot take the
/// geometry.
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
            GraphAttn::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, .. } => {
                s.push(2);
                for g in [wq, wk, wv, wo] {
                    w(&mut s, g);
                }
                f(&mut s, q_norm.unwrap_or(&[]));
                f(&mut s, k_norm.unwrap_or(&[]));
                s.push(*output_gate as usize);
            }
            _ => s.push(3),
        }
        if let GraphFfn::Dense { gate, up, down, .. } = &l.ffn {
            for g in [gate, up, down] {
                w(&mut s, g);
            }
        }
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
) -> Res<HPlan> {
    if nkv == 0 || nh % nkv != 0 || (layers.is_empty() && lm.is_none()) {
        return Err("empty stack or head geometry".into());
    }
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
    let (mut cdmax, mut vdmax, mut nvmax, mut intermax) = (0usize, 0usize, 0usize, 0usize);
    for (li, l) in layers.iter().enumerate() {
        if l.mhc.is_some() || l.attn_out_norm.is_some() || l.ffn_out_norm.is_some() || l.layer_scale.is_some() {
            return Err(format!("layer {li}: mHC streams, sandwich norms or a layer scale"));
        }
        if l.input_norm.len() != hidden || l.post_norm.len() != hidden {
            return Err(format!("layer {li}: norm width"));
        }
        match &l.attn {
            GraphAttn::Gdn { nv, nk, dk, dv, kk, preview, .. } => {
                if preview.is_some() {
                    return Err(format!("layer {li}: block preview"));
                }
                if *dk != 128 || *dv != *dk || *nk == 0 || nv % nk != 0 || *kk < 1 || *kk > 9 {
                    return Err(format!("layer {li}: GDN geometry nv {nv} nk {nk} dk {dk} dv {dv}"));
                }
                cdmax = cdmax.max(2 * nk * dk + nv * dv);
                vdmax = vdmax.max(nv * dv);
                nvmax = nvmax.max(*nv);
            }
            GraphAttn::Full { late_qk_norm, bias, output_gate, geom, head_gate, .. } => {
                if *late_qk_norm || bias.is_some() || geom.is_some() || head_gate.is_some() {
                    return Err(format!("layer {li}: late Q/K norm, biases, own geometry or a head-wise gate"));
                }
                qrow = qrow.max(nh * hd * (1 + *output_gate as usize));
            }
            _ => return Err(format!("layer {li}: a token mixer the hybrid arm has no kernels for")),
        }
        let GraphFfn::Dense { gate, act, limit, hot, .. } = &l.ffn else {
            return Err(format!("layer {li}: an FFN other than a dense one"));
        };
        if *act != GraphAct::Silu || *limit != 0.0 || hot.is_some() {
            return Err(format!("layer {li}: an FFN other than plain SwiGLU"));
        }
        intermax = intermax.max(rows_of(model, gate).ok_or("f32 FFN")?);
    }
    let lm_rows = lm.map_or(0, |(_, r)| r);
    let f4 = |n: usize| Buf::new(n.max(1) * 4);
    let h = f4(NMAX * hidden)?;
    let xn = f4(NMAX * hidden)?;
    let q = f4(NMAX * qrow)?;
    let k = f4(NMAX * nkv * hd)?;
    let v = f4(NMAX * nkv * hd)?;
    let qg = f4(NMAX * nh * hd)?;
    let gate = f4(NMAX * nh * hd)?;
    let ao = f4(NMAX * nh * hd)?;
    let qkv = f4(NMAX * cdmax)?;
    let z = f4(NMAX * vdmax)?;
    let o = f4(NMAX * vdmax)?;
    let of = f4(NMAX * vdmax)?;
    let tab = f4(NMAX * half * 2)?;
    let ids = f4(NMAX)?;
    let pstride = hidden.max(intermax).max(vdmax).max(nh * hd).next_multiple_of(32);
    // the FFN middle's planes (the gate/up pair writes them, down reads them)
    let qq = [Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?];
    let mut out = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let GraphFfn::Dense { gate: gw, up: uw, down: dw, .. } = &l.ffn else { unreachable!() };
        let inter = rows_of(model, gw).ok_or("f32 FFN")?;
        let gum = hmat_pair(c, st, model, gw, uw, inter, hidden)?;
        let dm = hmat(c, st, model, dw, hidden, inter)?;
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
                let half_of = |m: &DevMat| -> Res<Buf> {
                    let n = m.rows * m.cols;
                    let hb = Buf::new(n * 2)?;
                    let src = if m.kind == 4 {
                        m.w.ptr
                    } else {
                        return Err("a GDN gate projection that is not f32".into());
                    };
                    let n64 = n as u64;
                    unsafe { c.launch(fx.to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(src, hb.ptr, n64))? };
                    Ok(hb)
                };
                let a_h = half_of(&am)?;
                let b_h = half_of(&bm)?;
                let ms_in = mmaset(&[(&qm, qkv.ptr, 0, cdmax, cdim), (&zm, z.ptr, 0, vdmax, nv * dv)])?;
                let ms_out = mmaset(&[(&om, h.ptr, h.ptr, hidden, hidden)])?;
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
            GraphAttn::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, .. } => {
                let qr = nh * hd * (1 + *output_gate as usize);
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
                let ms_in = mmaset(&[
                    (&qm, q.ptr, 0, qrow, qr),
                    (&km, k.ptr, 0, nkv * hd, nkv * hd),
                    (&vm, v.ptr, 0, nkv * hd, nkv * hd),
                ])?;
                let ms_out = mmaset(&[(&om, h.ptr, h.ptr, hidden, hidden)])?;
                Mix::Attn(AttnL { q: qm, k: km, v: vm, o: om, qn, kn, gated: *output_gate, ms_in, ms_out })
            }
            _ => unreachable!(),
        };
        let ms_gu = mmaset_pair(&gum, qq[0].ptr, qq[1].ptr, pstride)?;
        let ms_dn = mmaset(&[(&dm, h.ptr, h.ptr, hidden, hidden)])?;
        out.push(HLayer { mix, gu: gum, down: dm, in_norm, post_norm, inter, ms_gu, ms_dn });
    }
    let (lm, logits, final_norm_buf) = match lm {
        Some((gw, rows)) => {
            let m = hmat(c, st, model, gw, rows_of(model, gw).ok_or("an f32 head")?, hidden)?;
            if rows == 0 || rows > m.rows {
                return Err(format!("head of {rows} rows over a {}-row matrix", m.rows));
            }
            if final_norm.len() != hidden {
                return Err("final norm width".into());
            }
            // (in the signature by content: no pointer of it is kept)
            let logits = f4(NMAX * rows)?;
            let js = mmaset(&[(&m, logits.ptr, 0, rows, rows)])?;
            (Some((m, rows, js)), logits, Some(st.f32_buf(final_norm)?))
        }
        None => (None, f4(1)?, None),
    };
    let _ = lm_rows;
    let pp = [Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?, Buf::new(NMAX * pstride * 2)?];
    static PLAN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let n_attn = layers.iter().filter(|l| matches!(l.attn, GraphAttn::Full { .. })).count();
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
) -> Res<Arc<HPlan>> {
    let key = (model.uid(), layer_base, layers.len(), lm.is_some());
    let sig = hsig(model, layers, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, invf);
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
    let p = Arc::new(build(c, st, model, layers, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, scale, invf, sig)?);
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
) -> Result<(), (bool, String)> {
    let soft = |e: String| (false, e);
    for (li, (pl, l)) in plan.layers.iter().zip(layers).enumerate() {
        let key = (kv_id, layer_base + li);
        match (&pl.mix, &l.attn) {
            (Mix::Attn(_), GraphAttn::Full { cpu_k, cpu_v, cpu_base, .. }) => {
                let row = plan.nkv * plan.hd;
                let m = st.kv.entry(key).or_insert_with(|| KvMirror::new(row, 0));
                if m.row != row || m.window != 0 {
                    let _ = c.sync();
                    c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
                    *m = KvMirror::new(row, 0);
                }
                match m.sync_to(c, cpu_k, cpu_v, *cpu_base, plan.nkv, plan.hd, pos0) {
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
                let mk = |n: usize| -> Res<Buf> {
                    let b = Buf::new(n.max(1) * 4)?;
                    unsafe { cu::memset_d8_async(b.ptr, 0, n.max(1) * 4, c.stream).map_err(derr)? };
                    Ok(b)
                };
                let alloc = || -> Res<GdnState> {
                    let s = mk(s_n)?;
                    let ring = mk(ring_n)?;
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
                        s,
                        ring,
                        next_pos: pos0,
                        t_qkv: mk(NMAX * plan.cdmax)?,
                        t_stride: plan.cdmax,
                        t_cq: mk(NMAX * g.cdim)?,
                        t_gb: mk(NMAX * 2 * g.nv)?,
                        s_save: mk(s_n)?,
                        ring_save: mk(ring_n)?,
                        tape: None,
                    })
                };
                let gs = alloc().map_err(soft)?;
                c.kv_bytes.fetch_add(((s_n * 2 + ring_n * 2 + NMAX * (2 * g.cdim + 2 * g.nv)) * 4) as u64, Ordering::Relaxed);
                st.hyb.gdn.insert(key, gs);
            }
            _ => return Err(soft(format!("layer {li}: descriptors changed under the plan"))),
        }
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
) -> Res<Book> {
    let (nh, hd) = (plan.nh, plan.hd);
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut ci = Vec::new();
    let mut keys_max = 0usize;
    for (li, pl) in plan.layers.iter().enumerate() {
        let key = (kv_id, layer_base + li);
        match &pl.mix {
            Mix::Gdn(_) => {
                let gs = st.hyb.gdn.get_mut(&key).ok_or("GDN state missing")?;
                layers.push([gs.s.ptr, gs.ring.ptr, gs.t_qkv.ptr, gs.t_cq.ptr, gs.t_gb.ptr, gs.s_save.ptr, gs.ring_save.ptr]);
                gs.next_pos = pos0 + n;
                gs.tape = if tape { Some((pos0, n)) } else { None };
            }
            Mix::Attn(_) => {
                let m = st.kv.get_mut(&key).ok_or("K/V mirror missing")?;
                let n0 = m.stored - m.base;
                layers.push([m.kptr(), m.vptr(), 0, 0, 0, 0, 0]);
                m.stored += n;
                let keys = n0 + n;
                let rb = attn_rb(keys);
                let ncap = keys.next_multiple_of(4096);
                let nbcap = ncap.div_ceil(rb);
                ci.extend([n0 as i32, n0 as i32, rb as i32, ncap as i32, nbcap as i32, 0, 0, 0]);
                keys_max = keys_max.max(keys);
            }
        }
    }
    // the attention scratch sized for the context cap, so a replayed graph's
    // pointers hold from one call to the next
    let kc = kcap.max(keys_max).next_multiple_of(4096);
    let nbc = ATTN_NBLK + 64;
    let scratch = [
        st.scr(c, "hy_S", NMAX * nh * kc * 4)?,
        st.scr(c, "hy_Mx", NMAX * nh * nbc * 4)?,
        st.scr(c, "hy_P", NMAX * nbc * nh * hd * 4)?,
        st.scr(c, "hy_E", NMAX * nbc * nh * 4)?,
    ];
    Ok(Book { layers, ci, scratch })
}

/// The small path's launches over the rows in `plan.h` (`n <= NMAX`), as
/// `book` laid them out: kernels and device copies only, nothing on the
/// host, so a stream capture can record it. `tape`: keep what a verify
/// needs to step back.
#[allow(clippy::too_many_arguments)]
unsafe fn launch(c: &Ctx, plan: &HPlan, bk: &Book, n: usize, tape: bool) -> Res<()> {
    let fx = &plan.fx;
    let (hidden, nh, nkv, hd) = (plan.hidden, plan.nh, plan.nkv, plan.hd);
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
            c.launch(
                fx.rmsnorm2,
                (hidden.div_ceil(1024) as u32, n as u32),
                256,
                0,
                kargs!(plan.h.ptr, pl.in_norm.ptr, plan.xn.ptr, hid_i, eps, gemma_i, pp0, pp1, pp2, ps_i),
            )?;
        }
        hst(c, 0, &mut t0)?;
        match &pl.mix {
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
                    if tape {
                        cu::memcpy_dtod_async(s_save, gs_s, s_bytes, c.stream).map_err(derr)?;
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
                        (units + g.nv as u32, n as u32),
                        g.dk.max(64) as u32,
                        0,
                        kargs!(plan.qkv.ptr, qs_i, gs_ring, g.taps.ptr, t_cq, n_i, cd_i, kk_i, nk_i, dk_i, plan.xn.ptr, hid_i, g.a.w.ptr, g.b.w.ptr, g.a_log.ptr, g.dt_bias.ptr, t_gb, nv_i),
                    )?;
                    let cols = g.nv * g.dv;
                    // the scan, with blocks after it that move the conv ring
                    let ring_blocks = if g.kk > 1 { (g.cdim as u32).div_ceil(256) } else { 0 };
                    let ring_p = if g.kk > 1 { gs_ring } else { 0 };
                    c.launch(
                        fx.scan,
                        ((cols as u32).div_ceil(8) + ring_blocks, 1),
                        256,
                        0,
                        kargs!(t_cq, t_gb, gs_s, plan.o.ptr, n_i, cd_i, nv_i, nk_i, dk_i, dv_i, ring_p, plan.qkv.ptr, qs_i, kk_i),
                    )?;
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
                    g.ms_out.run(c, fx, planes, cols, n)?;
                    hst(c, 3, &mut t0)?;
                }
            }
            Mix::Attn(a) => {
                let (kp, vp) = (lb[0], lb[1]);
                let cip = plan.ci.ptr + (ai * 8 * 4) as u64;
                let w = &bk.ci[ai * 8..ai * 8 + 8];
                ai += 1;
                let (n0_i, slot0, rb_i, ncap_i, nbcap_i, r0) = (w[0], w[1], w[2], w[3], w[4], 0i32);
                let (nh_i, nkv_i, half_i, gated_i) = (nh as i32, nkv as i32, plan.half as i32, a.gated as i32);
                let (qs_i, kvs_i) = (plan.qrow as i32, (nkv * hd) as i32);
                let (qn, kn) = (a.qn.as_ref().map_or(0, |b| b.ptr), a.kn.as_ref().map_or(0, |b| b.ptr));
                let gate_p = if a.gated { plan.gate.ptr } else { 0 };
                unsafe {
                    a.ms_in.run(c, fx, planes, hidden, n)?;
                    hst(c, 4, &mut t0)?;
                    c.launch(
                        fx.qk,
                        ((nh + nkv) as u32, n as u32),
                        hd as u32,
                        0,
                        kargs!(
                            plan.q.ptr, qs_i, plan.k.ptr, plan.v.ptr, kvs_i, qn, kn, plan.qg.ptr, plan.gate.ptr, kp, vp,
                            plan.tab.ptr, nh_i, nkv_i, half_i, eps, gemma_i, gated_i, slot0, n_i, cip
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
                    c.launch_3d(
                        fx.values,
                        (nblk, nkv as u32, n as u32),
                        256,
                        0,
                        kargs!(s_p, mx_p, vp, p_p, e_p, n0_i, r0, nkv_i, rb_i, ncap_i, nbcap_i, n_i, cip),
                    )?;
                    c.launch(
                        fx.finish,
                        (nh as u32, n as u32),
                        256,
                        0,
                        kargs!(p_p, e_p, gate_p, pp0, pp1, pp2, ps_i, n0_i, rb_i, nh_i, nbcap_i, cip),
                    )?;
                    hst(c, 5, &mut t0)?;
                    a.ms_out.run(c, fx, planes, nh * hd, n)?;
                    hst(c, 6, &mut t0)?;
                }
            }
        }
        unsafe {
            let nullp = 0u64;
            c.launch(
                fx.rmsnorm2,
                (hidden.div_ceil(1024) as u32, n as u32),
                256,
                0,
                kargs!(plan.h.ptr, pl.post_norm.ptr, nullp, hid_i, eps, gemma_i, pp0, pp1, pp2, ps_i),
            )?;
            hst(c, 0, &mut t0)?;
            // gate/up with silu(gate) * up made in its epilogue, into the
            // middle's planes; down reads them
            pl.ms_gu.run(c, fx, planes, hidden, n)?;
            hst(c, 7, &mut t0)?;
            let mid = Planes { p0: plan.qq[0].ptr, p1: plan.qq[1].ptr, p2: plan.qq[1].ptr, stride: plan.pstride };
            pl.ms_dn.run(c, fx, &mid, pl.inter, n)?;
            hst(c, 9, &mut t0)?;
        }
    }
    Ok(())
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
    *ON.get_or_init(|| std::env::var("CMF_CUDA_GRAPH").as_deref() != Ok("0")) && !super::prof_on()
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
) -> Res<usize> {
    let bk = book(c, st, plan, kv_id, layer_base, pos0, n, tape, kcap)?;
    let rows = plan.lm.as_ref().map_or(0, |l| l.1);
    unsafe {
        if !bk.ci.is_empty() {
            cu::memcpy_htod_async(plan.ci.ptr, &bk.ci, c.stream).map_err(derr)?;
        }
        if !graphs_on() {
            launch(c, plan, &bk, n, tape)?;
            if head {
                head_launch(c, plan, n)?;
            }
            return Ok(if head { rows } else { 0 });
        }
        let key = (plan.id, kv_id, layer_base, n, tape, head);
        let mut sig: Vec<u64> = bk.layers.iter().flatten().copied().collect();
        sig.extend(bk.scratch);
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
                launch(c, plan, &bk, n, tape)?;
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

/// The head over the rows in `plan.h`: logits [n][rows] into `plan.logits`.
unsafe fn head_launch(c: &Ctx, plan: &HPlan, n: usize) -> Res<usize> {
    let Some((_, rows, js)) = &plan.lm else { return Ok(0) };
    let fx = &plan.fx;
    let (hid_i, eps, gemma_i) = (plan.hidden as i32, plan.eps, plan.gemma as i32);
    let fnorm = plan.final_norm.as_ref().unwrap().ptr;
    let mut t0 = std::time::Instant::now();
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
    if steps != 1 || !loop_norm_at.is_empty() || o1.iter().any(Option::is_some) {
        decline("hybrid token", "a multi-step burst, a mid-stack norm or a sealed far field");
        return TokenGraphOutcome::Declined;
    }
    if position >= cap || h.len() != hidden {
        decline("hybrid token", "position past the context cap");
        return TokenGraphOutcome::Declined;
    }
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return TokenGraphOutcome::Declined;
    }
    let plan = match plan_for(c, &mut st, model, layers, layer_base, lm_head, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, attn_scale, invf, position) {
        Ok(p) => p,
        Err(e) => {
            decline("hybrid token", &e);
            return TokenGraphOutcome::Declined;
        }
    };
    if let Err((hard, e)) = admit(c, &mut st, &plan, layers, kv_id, layer_base, position, 1, 0) {
        decline("hybrid token", &e);
        return if hard { TokenGraphOutcome::Failed } else { TokenGraphOutcome::Declined };
    }
    let t0 = std::time::Instant::now();
    let mut run = || -> Res<()> {
        unsafe {
            cu::memcpy_htod_async(plan.h.ptr, h, c.stream).map_err(derr)?;
            let tab = rope_rows(&plan.invf, std::iter::once(position));
            if !tab.is_empty() {
                cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
            }
            let rows = run_small(c, &mut st, &plan, kv_id, layer_base, position, 1, false, plan.lm.is_some(), cap)?;
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
    match run() {
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
            TokenGraphOutcome::Completed
        }
        Err(e) => {
            tracing::error!("cuda hybrid token graph failed: {e}");
            let _ = c.sync();
            TokenGraphOutcome::Failed
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
    spec: Option<SpecTail<'_>>,
    layers_run: Option<&mut usize>,
    defer: bool,
) -> BatchGraphOutcome {
    let b = k;
    if o1.iter().any(|v| v.is_some()) {
        decline("hybrid batch", "a sealed far field");
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
    let final_norm: &[f32] = spec.as_ref().map_or(&[], |s| s.final_norm);
    let plan = match plan_for(c, &mut st, model, layers, layer_base, lm, final_norm, nh, nkv, hd, rd, hidden, gemma, eps, attn_scale, invf, pos0) {
        Ok(p) => p,
        Err(e) => {
            decline("hybrid batch", &e);
            return BatchGraphOutcome::Declined;
        }
    };
    let hint = st.kv_hint;
    if let Err((hard, e)) = admit(c, &mut st, &plan, layers, kv_id, layer_base, pos0, b, hint) {
        decline("hybrid batch", &e);
        return if hard { BatchGraphOutcome::Failed } else { BatchGraphOutcome::Declined };
    }
    let mut spec = spec;
    let mut run = || -> Res<()> {
        if b <= NMAX {
            unsafe {
                cu::memcpy_htod_async(plan.h.ptr, h, c.stream).map_err(derr)?;
                let tab = rope_rows(&plan.invf, positions.iter().copied());
                if !tab.is_empty() {
                    cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
                }
                let rows = run_small(c, &mut st, &plan, kv_id, layer_base, pos0, b, spec.is_some(), spec.is_some(), cap)?;
                if let Some(sp) = spec.as_mut() {
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
            unsafe { big(c, &mut st, &plan, kv_id, layer_base, positions, h, defer) }
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

/// The prompt path: a chunk of `b > NMAX` rows on cuBLAS.
#[allow(clippy::too_many_arguments)]
unsafe fn big(
    c: &Ctx,
    st: &mut State,
    plan: &HPlan,
    kv_id: u64,
    layer_base: usize,
    positions: &[usize],
    h: &mut [f32],
    defer: bool,
) -> Res<()> {
    let fx = &plan.fx;
    let f = &c.f;
    let b = positions.len();
    let pos0 = positions[0];
    let (hidden, nh, nkv, hd, hpk) = (plan.hidden, plan.nh, plan.nkv, plan.hd, plan.hpk);
    let (gw, ga) = prec();
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
            }
            v
        })
        .map(|(r, c)| r * c)
        .max()
        .unwrap_or(0);
    let wd = st.scr(c, "hb_wd", wmax * 2)?;
    let tab = rope_rows(&plan.invf, positions.iter().copied());
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
    // y [b][ldy] (+)= alpha W x: the weight expanded to f16, one GEMM
    let mul = |m: &DevMat, x: u64, y: u64, ldy: usize, alpha: f32, beta: f32| -> Res<()> {
        let (rows_i, cols_i) = (m.rows as i32, m.cols as i32);
        if m.kind != 6 {
            return Err("a weight other than repacked q4tp on the prompt path".into());
        }
        let threads = (m.rows.div_ceil(16) * (m.cols / 32) * 32) as u64;
        unsafe {
            c.launch(fx.expand_p, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i))?;
            bst(c, 1, tm)?;
            c.gemm(true, false, m.rows, b, m.cols, alpha, wd, m.cols, x, m.cols, beta, y, ldy, Gemm::Half)?;
            bst(c, 2, tm)
        }
    };
    for (li, pl) in plan.layers.iter().enumerate() {
        let key = (kv_id, layer_base + li);
        unsafe {
            c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, pl.in_norm.ptr, bxn, hid_i, eps, gemma_i, one, nul, nul, nul, zero_i))?;
        }
        bst(c, 0, tm)?;
        match &pl.mix {
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
                mul(&g.qkv, bxn, yq, g.cdim, 1.0, 0.0)?;
                mul(&g.z, bxn, yz, vd, 1.0, 0.0)?;
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
                mul(&g.out, yof, bh, hidden, 1.0, 1.0)?;
            }
            Mix::Attn(a) => {
                let qr = a.q.rows;
                let yq = st.scr(c, "hb_q", b * qr * 4)?;
                let yk = st.scr(c, "hb_k", b * nkv * hd * 4)?;
                let yv = st.scr(c, "hb_v", b * nkv * hd * 4)?;
                let qg = st.scr(c, "hb_qg", b * nh * hd * 4)?;
                let gt = st.scr(c, "hb_gate", b * nh * hd * 4)?;
                let aog = st.scr(c, "hb_aog", b * nh * hd * 4)?;
                let bao = st.scr(c, "hb_ao", b * nh * hd * 2)?;
                let bis = st.scr(c, "hb_is", b * nh * 4)?;
                mul(&a.q, bxn, yq, qr, 1.0, 0.0)?;
                mul(&a.k, bxn, yk, nkv * hd, 1.0, 0.0)?;
                mul(&a.v, bxn, yv, nkv * hd, 1.0, 0.0)?;
                let hint = st.kv_hint;
                let (slot0, kp, vp, m_base) = {
                    let m = st.kv.get_mut(&key).ok_or("K/V mirror missing")?;
                    let out = ((m.stored - m.base) as i32, m.kptr(), m.vptr(), m.base);
                    m.stored += b;
                    out
                };
                let (nh_i, nkv_i, half_i, gated_i) = (nh as i32, nkv as i32, plan.half as i32, a.gated as i32);
                let (qs_i, kvs_i) = (qr as i32, (nkv * hd) as i32);
                let (qn, kn) = (a.qn.as_ref().map_or(0, |b| b.ptr), a.kn.as_ref().map_or(0, |b| b.ptr));
                unsafe {
                    c.launch(
                        fx.qk,
                        ((nh + nkv) as u32, b as u32),
                        hd as u32,
                        0,
                        kargs!(yq, qs_i, yk, yv, kvs_i, qn, kn, qg, gt, kp, vp, btab, nh_i, nkv_i, half_i, eps, gemma_i, gated_i, slot0, b_i, nul),
                    )?;
                }
                let lda = nkv * hd;
                let grow = hpk * hd;
                let mut b0 = 0usize;
                while b0 < b {
                    let p0 = pos0 + b0;
                    let col_base = m_base;
                    let fit = |n: usize| (p0 + n - col_base) * n * nh * 4 <= 384 << 20;
                    let mut bn = b - b0;
                    while bn > 64 && !fit(bn) {
                        bn = bn.div_ceil(2);
                    }
                    let l_rows = p0 + bn - col_base;
                    let cols = bn * hpk;
                    let whole = (hint.max(pos0 + b) * b * nh * 4).min(400 << 20);
                    let bs = st.scr(c, "hb_s", (nkv * l_rows * cols * 4).max(whole))?;
                    let (l_i, p0_i, cb_i, win_i, hpk_i) = (l_rows as i32, p0 as i32, col_base as i32, 0i32, hpk as i32);
                    let (cols_i, istride) = (cols as i32, (b * hpk) as i32);
                    let goff = (b0 * grow * 4) as u64;
                    let isum = bis + (b0 * hpk * 4) as u64;
                    unsafe {
                        c.gemm_batched(true, l_rows, cols, hd, plan.scale, kp, lda, hd, qg + goff, hd, b * grow, bs, l_rows, l_rows * cols, nkv, ga)?;
                        c.launch(f.softmax_rows, ((nkv * cols) as u32, 1), 256, 0, kargs!(bs, isum, l_i, p0_i, cb_i, win_i, hpk_i, cols_i, istride))?;
                        c.gemm_batched(false, hd, cols, l_rows, 1.0, vp, lda, hd, bs, l_rows, l_rows * cols, aog + goff, hd, b * grow, nkv, ga)?;
                    }
                    b0 += bn;
                }
                let n_ao = (b * nh * hd) as u64;
                let gate_p = if a.gated { gt } else { 0u64 };
                unsafe {
                    c.launch(fx.attn_out, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(aog, bis, gate_p, bao, nh_i, b_i, one))?;
                }
                bst(c, 5, tm)?;
                mul(&a.o, bao, bh, hidden, 1.0, 1.0)?;
            }
        }
        let inter = pl.inter;
        let bgu = st.scr(c, "hb_gu", b * 2 * inter * 4)?;
        let bmh = st.scr(c, "hb_mh", b * inter * 2)?;
        unsafe {
            c.launch(fx.rmsnorm, (b as u32, 1), 256, 0, kargs!(bh, pl.post_norm.ptr, bxn, hid_i, eps, gemma_i, one, nul, nul, nul, zero_i))?;
        }
        bst(c, 0, tm)?;
        // the gate/up pair as one GEMM, its rows in the pair's order
        mul(&pl.gu, bxn, bgu, 2 * inter, 1.0, 0.0)?;
        let tot = (b * inter) as u64;
        let (in_i, bb_i) = (inter as i32, b as i32);
        unsafe {
            c.launch(fx.silu_pair, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(bgu, bmh, in_i, bb_i))?;
        }
        bst(c, 6, tm)?;
        mul(&pl.down, bmh, bh, hidden, 16.0, 1.0)?;
    }
    if !defer {
        unsafe { cu::memcpy_dtoh_async(h, bh, c.stream).map_err(derr)? };
    }
    c.sync()
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
            c.launch_3d(fx.rs_scan, (((nv * dv) as u32).div_ceil(8), nl, 1), 256, 0, kargs!(tab, keep_i, cd_i, nv_i, nk_i, dk_i, dv_i))?;
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
    st.hyb.gdn.retain(|&(id, _), g| {
        if id == kv_id {
            let (nv, nk, dk, dv, kk) = g.dims;
            let cdim = 2 * nk * dk + nv * dv;
            let ring_n = (kk - 1) * cdim;
            let s_n = nv * dk * dv;
            c.kv_bytes.fetch_sub(((s_n * 2 + ring_n * 2 + NMAX * (2 * cdim + 2 * nv)) * 4) as u64, Ordering::Relaxed);
        }
        id != kv_id
    });
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

    fn down<T: Copy + Default>(c: &Ctx, b: &Buf, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        unsafe { cu::memcpy_dtoh_sync(&mut v, b.ptr) }.unwrap();
        let _ = c;
        v
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
}
