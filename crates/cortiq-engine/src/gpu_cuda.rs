//! Native CUDA backend (`--features cuda`, selected at run time with
//! `CMF_GPU=cuda`). NVIDIA only; every other machine keeps wgpu / Metal.
//!
//! It serves the two graph contracts of `gpu.rs` for dense stacks with full
//! or sliding-window attention (the Spark-X2.x shape): the whole-token decode
//! graph (`forward_token_graph`) and the batched prompt graph
//! (`forward_batch_graph_at`), over K/V mirrors that live on the card. What
//! it does not carry — MoE, GDN, sealed O(1) far fields, sandwich norms,
//! output gates other than Spark's head-wise one — it declines, and the
//! caller's existing path runs.
//!
//! Stack: `cudarc` for the driver API (context, memory, module load, kernel
//! launch), NVRTC for the kernels (CUDA C in `gpu_cuda/*.cu`, compiled on
//! first use for the card's architecture and the model's head geometry), and
//! cuBLAS for the prompt's GEMMs. Nothing is linked: the libraries are opened
//! at run time, and a machine without them falls back to the CPU.
//!
//! Decode keeps weights in their codec and reads each matrix once per token
//! (`mv_multi`: a warp per row, the input row staged in shared memory with
//! the norm / activation / column field folded in). The prompt expands each
//! weight to a dense f32 plane once per chunk and multiplies with cuBLAS;
//! attention is two GEMMs and a causal softmax per head.
//!
//! Knobs: `CMF_CUDA_DEVICE=N` (ordinal), `CMF_CUDA_TF32=1` (TF32 tensor
//! cores for the prompt's weight GEMMs, `=2` for its attention too),
//! `CMF_CUDA_BATCH=N` (prompt rows per chunk), `CMF_CUDA_PROF=1` (stage
//! timers, synchronous), `CMF_CUDA_DEBUG=1` (say why a graph declined).

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cortiq_core::CmfModel;
use cudarc::cublas::sys as bl;
use cudarc::driver::{result as cu, sys as cs};

use crate::gpu::{
    BatchGraphOutcome, GraphAct, GraphAttn, GraphFfn, GraphLayer, GraphPrismOp, GraphW,
    TokenGraphOutcome,
};

const KERNELS_SRC: &str = include_str!("gpu_cuda/kernels.cu");
const ATTN_SRC: &str = include_str!("gpu_cuda/attn.cu");

type Res<T> = Result<T, String>;

fn derr(e: cu::DriverError) -> String {
    format!("CUDA driver: {e}")
}

/// `CMF_GPU=cuda` selects this backend (the facade asks before `enabled()`).
pub fn selected() -> bool {
    matches!(std::env::var("CMF_GPU").as_deref(), Ok("cuda"))
}

/// Selected and the context came up.
pub fn enabled() -> bool {
    selected() && ctx().is_some()
}

fn debug() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_CUDA_DEBUG").is_some())
}

fn prof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_CUDA_PROF").is_some())
}

/// How a prompt GEMM runs on the tensor cores.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gemm {
    /// f32 in, f32 out, no tensor cores: the reference.
    F32,
    /// f32 in and out, operands rounded to TF32 inside (10-bit mantissa).
    Tf32,
    /// f16 operands, f32 accumulation and output: twice TF32's rate.
    Half,
}

/// `CMF_CUDA_PREC`: the prompt's precision, as (weight GEMMs, attention GEMMs).
/// `fast` (default): weights expanded to f16, attention on TF32 — the class
/// of the wgpu matrix-unit kernels. `tf32`: both on TF32. `exact`: plain f32.
/// Decode is f32 in every mode.
fn prec() -> (Gemm, Gemm) {
    static P: OnceLock<(Gemm, Gemm)> = OnceLock::new();
    *P.get_or_init(|| match std::env::var("CMF_CUDA_PREC").as_deref() {
        Ok("exact") | Ok("f32") => (Gemm::F32, Gemm::F32),
        Ok("tf32") => (Gemm::Tf32, Gemm::Tf32),
        _ => (Gemm::Half, Gemm::Tf32),
    })
}

/// Prompt rows per batched-graph chunk on this backend.
pub fn batch_rows() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CMF_CUDA_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|k| *k > 0)
            .unwrap_or(1024)
    })
}

/// The f32 constants' slab size.
const SLAB: usize = 8 << 20;

/// One driver allocation carved into buffers (`SlabScope`).
struct Slab {
    ptr: cs::CUdeviceptr,
    /// The free extents (offset, length), by offset.
    free: Mutex<Vec<(usize, usize)>>,
}

impl Slab {
    fn carve(&self, need: usize) -> Option<usize> {
        let mut f = self.free.lock().unwrap();
        let i = f.iter().position(|&(_, len)| len >= need)?;
        let (off, len) = f[i];
        if len == need {
            f.remove(i);
        } else {
            f[i] = (off + need, len - need);
        }
        Some(off)
    }

    fn release(&self, off: usize, len: usize) {
        let mut f = self.free.lock().unwrap();
        let i = f.partition_point(|&(o, _)| o < off);
        f.insert(i, (off, len));
        if i + 1 < f.len() && f[i].0 + f[i].1 == f[i + 1].0 {
            f[i].1 += f[i + 1].1;
            f.remove(i + 1);
        }
        if i > 0 && f[i - 1].0 + f[i - 1].1 == f[i].0 {
            f[i - 1].1 += f[i].1;
            f.remove(i);
        }
    }

    fn left(&self) -> usize {
        self.free.lock().unwrap().iter().map(|&(_, len)| len).sum()
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::free_sync(self.ptr);
        }
    }
}

/// The slabs a `SlabScope` carves this thread's buffers from.
struct SlabPool {
    slabs: Vec<Arc<Slab>>,
    /// Buffers past this size stay the driver's own.
    max_piece: usize,
    pieces: usize,
    carved: u64,
    direct: u64,
    direct_n: usize,
}

/// Bytes of a slab: a multiple of what the driver rounds an allocation to.
const SLAB_BYTES: usize = 64 << 20;
/// The largest buffer a slab holds.
const SLAB_PIECE: usize = 16 << 20;

impl SlabPool {
    fn carve(&mut self, bytes: usize) -> Option<Buf> {
        if bytes > self.max_piece {
            self.direct += bytes as u64;
            self.direct_n += 1;
            return None;
        }
        let need = bytes.next_multiple_of(256);
        let hit = self.slabs.iter().find_map(|s| s.carve(need).map(|off| (s.clone(), off)));
        let (slab, off) = match hit {
            Some(h) => h,
            None => {
                let ptr = unsafe { cu::malloc_sync(SLAB_BYTES) }.ok()?;
                // the last page stays out: a kernel's last vector load may read past a buffer
                let slab = Arc::new(Slab { ptr, free: Mutex::new(vec![(need, SLAB_BYTES - 4096 - need)]) });
                self.slabs.push(slab.clone());
                (slab, 0)
            }
        };
        self.pieces += 1;
        self.carved += need as u64;
        Some(Buf { ptr: slab.ptr + off as u64, bytes, owned: true, slab: Some((slab, need)) })
    }
}

thread_local! {
    static SLABS: std::cell::RefCell<Option<SlabPool>> = const { std::cell::RefCell::new(None) };
}

/// While it lives, the buffers this thread allocates are pieces of shared
/// slabs: the driver rounds an allocation above a megabyte up to a multiple of
/// two, and a model of a thousand buffers loses a gigabyte of the card to that.
/// A buffer that is dropped gives its piece back to its slab; a slab goes back
/// to the driver with its last buffer. Buffers that are replaced as a session
/// runs (`Buf::direct`) and the large ones stay the driver's own.
struct SlabScope;

impl SlabScope {
    fn enter() -> Self {
        let off = std::env::var("CMF_CUDA_SLAB").is_ok_and(|v| v.trim() == "0");
        if !off {
            SLABS.with(|s| {
                *s.borrow_mut() = Some(SlabPool { slabs: Vec::new(), max_piece: SLAB_PIECE, pieces: 0, carved: 0, direct: 0, direct_n: 0 })
            });
        }
        Self
    }

    /// What the scope took so far, for the load report.
    fn report(&self) -> String {
        SLABS.with(|s| match s.borrow().as_ref() {
            Some(p) => {
                let left: usize = p.slabs.iter().map(|s| s.left()).sum();
                format!(
                    "{} slabs of {} MiB hold {} buffers ({:.0} MiB, {:.0} MiB unused), {} buffers of {:.0} MiB are the driver's own",
                    p.slabs.len(),
                    SLAB_BYTES >> 20,
                    p.pieces,
                    p.carved as f64 / (1u64 << 20) as f64,
                    left as f64 / (1u64 << 20) as f64,
                    p.direct_n,
                    p.direct as f64 / (1u64 << 20) as f64,
                )
            }
            None => "no slabs".into(),
        })
    }
}

impl Drop for SlabScope {
    fn drop(&mut self) {
        SLABS.with(|s| s.borrow_mut().take());
    }
}

/// Free bytes of the card as the driver reports them (every process counted).
fn mem_free() -> Option<u64> {
    cu::mem_get_info().ok().map(|(free, _)| free as u64)
}

/// A device allocation, freed on drop (the context is current wherever one
/// is dropped: inside an entry point, under the state lock).
struct Buf {
    ptr: cs::CUdeviceptr,
    bytes: usize,
    /// A view into another allocation (`view`) frees nothing.
    owned: bool,
    /// The slab it is a piece of, and the piece's length (`SlabScope`).
    slab: Option<(Arc<Slab>, usize)>,
}

impl Buf {
    fn new(bytes: usize) -> Res<Self> {
        let bytes = bytes.max(64);
        if let Some(b) = SLABS.with(|s| s.borrow_mut().as_mut().and_then(|p| p.carve(bytes))) {
            return Ok(b);
        }
        Self::direct(bytes)
    }

    /// An allocation of the driver's own, whatever the scope: for a buffer
    /// that is replaced by a larger one as a session runs.
    fn direct(bytes: usize) -> Res<Self> {
        let bytes = bytes.max(64);
        let ptr = unsafe { cu::malloc_sync(bytes) }
            .map_err(|e| format!("device allocation of {bytes} bytes: {e}"))?;
        Ok(Self { ptr, bytes, owned: true, slab: None })
    }

    /// `bytes` of this allocation from `off`, freed with it: the holder keeps
    /// the owner alive at least as long (the card rounds each allocation up,
    /// so a matrix's small planes go in one with its big one).
    fn view(&self, off: usize, bytes: usize) -> Buf {
        debug_assert!(off + bytes <= self.bytes);
        Buf { ptr: self.ptr + off as u64, bytes, owned: false, slab: None }
    }

    fn from_slice<T: Copy>(data: &[T]) -> Res<Self> {
        // 64 spare bytes: a kernel's last vector load may read past the body.
        let b = Self::new(std::mem::size_of_val(data) + 64)?;
        if !data.is_empty() {
            unsafe { cu::memcpy_htod_sync(b.ptr, data) }.map_err(derr)?;
        }
        Ok(b)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        match self.slab.take() {
            Some((slab, len)) => slab.release((self.ptr - slab.ptr) as usize, len),
            None if self.owned => unsafe {
                let _ = cu::free_sync(self.ptr);
            },
            None => {}
        }
    }
}

/// One resident weight in its codec (see `gpu_cuda/kernels.cu` for the
/// planes of each kind).
struct DevMat {
    kind: u8,
    rows: usize,
    cols: usize,
    w: Arc<Buf>,
    aux0: Option<Buf>,
    aux1: Option<Buf>,
    /// Bytes of the weight budget it holds (its size in the file; 0 for an
    /// f32 weight, which the f32 cache owns).
    held: u64,
}

impl DevMat {
    fn aux0(&self) -> u64 {
        self.aux0.as_ref().map_or(0, |b| b.ptr)
    }
    fn aux1(&self) -> u64 {
        self.aux1.as_ref().map_or(0, |b| b.ptr)
    }
}

/// A codec tensor's bytes as the planes the kernels read (the layouts at the
/// top of `kernels.cu`): q4tp keeps its nibbles and gets its 5-bit rung codes
/// as bytes and every row's 32-scale ladder expanded; q8_2f keeps its int8
/// body and gets its two f16 scale planes as f32.
fn dev_mat(kind: u8, bytes: &[u8], rows: usize, cols: usize) -> Res<DevMat> {
    match kind {
        6 => {
            use cortiq_core::quant::{GROUP_SIZE, q4tp_code, q4tp_ladder, q4tp_sections};
            if cols % GROUP_SIZE != 0 {
                return Err("q4tp width not a multiple of 32".into());
            }
            let gpr = cols / GROUP_SIZE;
            let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
            if bytes.len() < codes_off + rows * stride {
                return Err("q4tp payload shorter than its planes".into());
            }
            let params = &bytes[params_off..params_off + rows * 4];
            let mut lad = vec![0f32; rows * 32];
            let mut codes = vec![0u8; rows * gpr];
            for r in 0..rows {
                lad[r * 32..r * 32 + 32].copy_from_slice(&q4tp_ladder(params, r));
                let cr = &bytes[codes_off + r * stride..codes_off + (r + 1) * stride];
                for g in 0..gpr {
                    codes[r * gpr + g] = q4tp_code(cr, g) as u8;
                }
            }
            Ok(DevMat {
                kind: 6,
                rows,
                cols,
                w: Arc::new(Buf::from_slice(&bytes[..params_off])?),
                aux0: Some(Buf::from_slice(&codes)?),
                aux1: Some(Buf::from_slice(&lad)?),
                held: bytes.len() as u64,
            })
        }
        7 => {
            use cortiq_core::quant::f16_to_f32;
            if cols % 16 != 0 {
                return Err("q8_2f width not a multiple of 16".into());
            }
            let body = rows * cols;
            if bytes.len() < body + rows * 2 + cols * 2 {
                return Err("q8_2f payload shorter than its planes".into());
            }
            let f16s = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(2).map(|p| f16_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect()
            };
            let rs = f16s(&bytes[body..body + rows * 2]);
            let col = f16s(&bytes[body + rows * 2..body + rows * 2 + cols * 2]);
            Ok(DevMat {
                kind: 7,
                rows,
                cols,
                w: Arc::new(Buf::from_slice(&bytes[..body])?),
                aux0: Some(Buf::from_slice(&rs)?),
                aux1: Some(Buf::from_slice(&col)?),
                held: bytes.len() as u64,
            })
        }
        k => Err(format!("weight codec kind {k} has no CUDA kernel")),
    }
}

/// The kernels' job record (`struct MvJob` in `kernels.cu`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MvJob {
    w: u64,
    aux0: u64,
    aux1: u64,
    y: u64,
    res: u64,
    rows: i32,
    kind: i32,
    blk0: i32,
    rpb: i32,
}

/// Shared memory a block of `mv_multi` stages a `cols`-wide row in (the
/// layout at the top of the kernel).
fn mv_shared_bytes(cols: usize) -> usize {
    let cpad = cols.next_multiple_of(32);
    4 * (16 + cpad + 64 + cpad / 32 + 8)
}

/// Rows a block of `mv_multi` owns: about 256 blocks a matrix, 8 warps each.
fn rows_per_block(rows: usize) -> usize {
    (rows / 256).max(1).next_multiple_of(8).clamp(8, 64)
}

fn push_job(jobs: &mut Vec<MvJob>, blk: &mut u32, m: &DevMat, y: u64, res: u64) {
    let rpb = rows_per_block(m.rows);
    jobs.push(MvJob {
        w: m.w.ptr,
        aux0: m.aux0(),
        aux1: m.aux1(),
        y,
        res,
        rows: m.rows as i32,
        kind: m.kind as i32,
        blk0: *blk as i32,
        rpb: rpb as i32,
    });
    *blk += m.rows.div_ceil(rpb) as u32;
}

struct PlanLayer {
    q: Arc<DevMat>,
    k: Arc<DevMat>,
    v: Arc<DevMat>,
    o: Arc<DevMat>,
    hg: Option<Arc<DevMat>>,
    /// The f16 copy of an f32 head gate (the f16 prompt path's operand).
    hg_h: Option<Buf>,
    gate: Arc<DevMat>,
    up: Arc<DevMat>,
    down: Arc<DevMat>,
    in_norm: Arc<Buf>,
    post_norm: Arc<Buf>,
    act: i32,
    inter: usize,
    nkv: usize,
    hd: usize,
    /// Rotary pairs (`rd / 2`).
    half: usize,
    /// Positions a query sees, 0 = the whole context.
    window: usize,
    rope_scale: f32,
    /// Offset of this layer's table inside one row of the call's RoPE table.
    rope_off: usize,
    jobs_qkv: Buf,
    n_qkv: i32,
    blk_qkv: u32,
    jobs_o: Buf,
    blk_o: u32,
    jobs_gu: Buf,
    blk_gu: u32,
    jobs_dn: Buf,
    blk_dn: u32,
}

/// A layer stack resolved for the device: weights, norms, job tables and the
/// decode work rows. Built once, reused while the descriptors stay the same.
struct Plan {
    layers: Vec<PlanLayer>,
    sig: Vec<usize>,
    /// (address, length, fingerprint) of every f32 slice the plan uploaded.
    consts: Vec<(usize, usize, u64)>,
    /// Distinct RoPE frequency tables, and floats per row of the call table.
    rope: Vec<Vec<f32>>,
    rope_row: usize,
    hpk: usize,
    final_norm: Option<Arc<Buf>>,
    lm: Option<(Arc<DevMat>, Buf, u32)>,
    // decode work rows
    h: Buf,
    q: Buf,
    /// q after the rotary kernel (what the attend reads).
    qr: Buf,
    k: Buf,
    v: Buf,
    g16: Buf,
    ao: Buf,
    gg: Buf,
    uu: Buf,
    logits: Buf,
    tab: Buf,
    qmax: usize,
    kvmax: usize,
    intermax: usize,
}

/// A layer's K/V rows on the card: `[row][kv head][hd]` f32, row 0 at
/// absolute position `base`. A sliding layer lets go of rows no later query
/// reads when it needs room.
struct KvMirror {
    k: Option<Buf>,
    v: Option<Buf>,
    cap: usize,
    base: usize,
    stored: usize,
    row: usize,
    window: usize,
}

impl KvMirror {
    fn new(row: usize, window: usize) -> Self {
        Self { k: None, v: None, cap: 0, base: 0, stored: 0, row, window }
    }

    fn kptr(&self) -> u64 {
        self.k.as_ref().map_or(0, |b| b.ptr)
    }
    fn vptr(&self) -> u64 {
        self.v.as_ref().map_or(0, |b| b.ptr)
    }

    /// Room for `add` rows after `stored`; `hint` = rows a full layer is
    /// about to hold (the prompt's length).
    ///
    /// No allocation on the decode path where it can be helped: a device
    /// allocation is a driver call of anything from a fraction of a
    /// millisecond to tens of them, and a mirror that grew for the second
    /// decoded token made one run in four of a 60-token benchmark 15 %
    /// slower (89 against 107 tok/s, every token's graph time the same).
    /// So a full layer grows with room for the next thousand rows, and a
    /// sliding layer moves its window to the front of the buffer it has.
    fn reserve(&mut self, c: &Ctx, add: usize, hint: usize) -> Res<()> {
        if self.stored - self.base + add <= self.cap {
            return Ok(());
        }
        let keep_from = if self.window > 0 {
            self.base.max((self.stored + 1).saturating_sub(self.window))
        } else {
            self.base
        };
        let keep = self.stored - keep_from;
        let off = ((keep_from - self.base) * self.row * 4) as u64;
        let bytes = keep * self.row * 4;
        if self.window > 0 && keep + add <= self.cap {
            // the window's rows to the front of the same buffer, through a
            // scratch where source and destination overlap; stream-ordered,
            // nothing to wait for
            if keep > 0 {
                let overlap = keep_from - self.base < keep;
                let mut tmp = c.move_tmp.lock().unwrap();
                if overlap && tmp.as_ref().is_none_or(|t| t.bytes < bytes) {
                    c.sync()?;
                    *tmp = Some(Buf::new(bytes)?);
                }
                for p in [self.kptr(), self.vptr()] {
                    unsafe {
                        if overlap {
                            let t = tmp.as_ref().unwrap().ptr;
                            cu::memcpy_dtod_async(t, p + off, bytes, c.stream).map_err(derr)?;
                            cu::memcpy_dtod_async(p, t, bytes, c.stream).map_err(derr)?;
                        } else {
                            cu::memcpy_dtod_async(p, p + off, bytes, c.stream).map_err(derr)?;
                        }
                    }
                }
            }
            self.base = keep_from;
            return Ok(());
        }
        let new_cap = if self.window > 0 {
            (self.window - 1 + add.max(256)).max(keep + add)
        } else {
            // a hint that covers the rows is the size (the prompt's rows, and its
            // first token); growth past it keeps some slack
            if hint >= keep + add { hint } else { (keep + add) + (self.cap / 8).max(1024) }
        };
        let nk = Buf::new(new_cap * self.row * 4)?;
        let nv = Buf::new(new_cap * self.row * 4)?;
        if keep > 0 {
            unsafe {
                cu::memcpy_dtod_async(nk.ptr, self.kptr() + off, bytes, c.stream).map_err(derr)?;
                cu::memcpy_dtod_async(nv.ptr, self.vptr() + off, bytes, c.stream).map_err(derr)?;
            }
        }
        // the old rows may still be read by queued work
        c.sync()?;
        c.kv_bytes.fetch_add((new_cap * self.row * 8) as u64, Ordering::Relaxed);
        c.kv_bytes.fetch_sub((self.cap * self.row * 8) as u64, Ordering::Relaxed);
        self.k = Some(nk);
        self.v = Some(nv);
        self.cap = new_cap;
        self.base = keep_from;
        Ok(())
    }

    /// Bring the mirror to exactly `upto` rows: rewind past rows of another
    /// continuation, or take the rows the host holds and the card does not.
    /// Ok(false): the host does not hold them either.
    #[allow(clippy::too_many_arguments)]
    fn sync_to(
        &mut self,
        c: &Ctx,
        cpu_k: &[Vec<f32>],
        cpu_v: &[Vec<f32>],
        cpu_base: usize,
        nkv: usize,
        hd: usize,
        upto: usize,
    ) -> Res<bool> {
        if self.stored > upto {
            if upto >= self.base {
                self.stored = upto;
            } else {
                self.base = 0;
                self.stored = 0;
            }
        }
        if self.stored == upto {
            return Ok(true);
        }
        let from = if self.window > 0 {
            self.stored.max((upto + 1).saturating_sub(self.window))
        } else {
            self.stored
        };
        if from > self.stored {
            self.base = from;
            self.stored = from;
        }
        let n = upto - from;
        if n == 0 {
            return Ok(true);
        }
        let have = cpu_k.first().map_or(0, |k| k.len() / hd);
        if cpu_k.len() != nkv || cpu_v.len() != nkv || from < cpu_base || upto > cpu_base + have {
            return Ok(false);
        }
        self.reserve(c, n, 0)?;
        let row = nkv * hd;
        let mut kb = vec![0f32; n * row];
        let mut vb = vec![0f32; n * row];
        for r in 0..n {
            let src = (from - cpu_base + r) * hd;
            for g in 0..nkv {
                kb[r * row + g * hd..][..hd].copy_from_slice(&cpu_k[g][src..src + hd]);
                vb[r * row + g * hd..][..hd].copy_from_slice(&cpu_v[g][src..src + hd]);
            }
        }
        let off = ((from - self.base) * row * 4) as u64;
        unsafe {
            cu::memcpy_htod_async(self.kptr() + off, &kb, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(self.vptr() + off, &vb, c.stream).map_err(derr)?;
        }
        self.stored = upto;
        Ok(true)
    }
}

struct Funcs {
    mv_multi: cs::CUfunction,
    rmsnorm_rows: cs::CUfunction,
    expand_q4tp: cs::CUfunction,
    expand_q8: cs::CUfunction,
    expand_q4tp_h: cs::CUfunction,
    expand_q8_h: cs::CUfunction,
    to_half: cs::CUfunction,
    act_half: cs::CUfunction,
    act_mul: cs::CUfunction,
    rope_kv_rows: cs::CUfunction,
    softmax_rows: cs::CUfunction,
    attn_out_rows: cs::CUfunction,
    q_rows_strided: cs::CUfunction,
}

#[derive(Clone, Copy)]
struct AttnFuncs {
    scores: cs::CUfunction,
    values: cs::CUfunction,
    finish: cs::CUfunction,
    far_insert: cs::CUfunction,
    far_prep: cs::CUfunction,
}

/// A sealed layer's far field on the card (`nystrom.rs`, `attn.cu`): the
/// near rows (the sinks with the rows kept exact, then the ring of the
/// window) laid out as a K/V mirror, the landmarks and each head's far
/// accumulators. It goes up once per seal epoch and then moves with every
/// row the graph runs; the host's copy is stale from there on, as a
/// mirrored layer's rows are.
struct O1Dev {
    epoch: u64,
    /// The position of the next row that may move this state; None until
    /// the first one has.
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
}

/// The rotated queries of the far-field layers of a prompt read on the batch
/// graph (`o1_qcap_begin` … `o1_qcap_take`): the seal's trace. A captured
/// layer's rows of a chunk — every one, or one in `stride` by position — are
/// gathered into a buffer that waits here for the drain.
struct QCap {
    layers: Vec<usize>,
    stride: usize,
    /// (layer, floats, the kept rows [row][head][hd])
    bufs: Vec<(usize, usize, Buf)>,
}

thread_local! {
    static QCAP: std::cell::RefCell<Option<QCap>> = const { std::cell::RefCell::new(None) };
    static ACAP: std::cell::RefCell<Option<ACap>> = const { std::cell::RefCell::new(None) };
}

/// The rows of sealed prompt chunks on their way to the archive
/// (`o1_acap_begin` … `o1_acap_end`): a sealed layer's chunk leaves its K/V
/// rows (f16, as the far field's mirror holds them) and one query row in
/// `stride` by position, until `o1_acap_take` reads them.
struct ACap {
    stride: usize,
    /// (layer, rows, K f16, V f16, kept query floats, queries [row][head][hd])
    bufs: Vec<CapBufs>,
}

type CapBufs = (usize, usize, Buf, Buf, usize, Option<Buf>);

/// Copy sealed prompt chunks' rows off the card until `o1_acap_end`; their
/// query rows one in `stride` by position.
pub fn o1_acap_begin(stride: usize) {
    if ctx().is_some() {
        ACAP.with(|a| *a.borrow_mut() = Some(ACap { stride: stride.max(1), bufs: Vec::new() }));
    }
}

pub fn o1_acap_end() {
    let a = ACAP.with(|a| a.borrow_mut().take());
    if let (Some(a), Some(c)) = (a, ctx()) {
        // freed with the context current, after the queued copies
        let _st = c.st.lock().unwrap();
        if c.current().is_ok() {
            let _ = c.sync();
        }
        drop(a);
    }
}

/// A layer's captured rows: (layer, queries, keys, values).
type CapRows = (usize, Vec<f32>, Vec<f32>, Vec<f32>);

/// The rows captured since the last call: per layer, (layer, queries one row
/// in the stride, keys, values), keys and values `[row][kv head][hd]` in f32.
pub fn o1_acap_take() -> Option<Vec<CapRows>> {
    let bufs = ACAP.with(|a| a.borrow_mut().as_mut().map(|a| std::mem::take(&mut a.bufs)))?;
    let c = ctx()?;
    let _st = c.st.lock().unwrap();
    c.current().ok()?;
    c.sync().ok()?;
    let mut out: Vec<CapRows> = Vec::new();
    // every buffer off the card, then the f16 rows widened a buffer a thread
    let mut raw: Vec<(usize, Vec<u16>, Vec<u16>, Vec<f32>)> = Vec::with_capacity(bufs.len());
    for (layer, _rows, k, v, qf, q) in &bufs {
        let mut kh = vec![0u16; k.bytes / 2];
        let mut vh = vec![0u16; v.bytes / 2];
        let mut qv = vec![0f32; *qf];
        unsafe {
            cu::memcpy_dtoh_async(kh.as_mut_slice(), k.ptr, c.stream).ok()?;
            cu::memcpy_dtoh_async(vh.as_mut_slice(), v.ptr, c.stream).ok()?;
            if let Some(q) = q {
                cu::memcpy_dtoh_async(qv.as_mut_slice(), q.ptr, c.stream).ok()?;
            }
        }
        c.sync().ok()?;
        raw.push((*layer, kh, vh, qv));
    }
    let wide: Vec<CapRows> = std::thread::scope(|sc| {
        let w = |h: &[u16]| h.iter().map(|&x| cortiq_core::quant::f16_to_f32(x)).collect::<Vec<f32>>();
        let hs: Vec<_> = raw.into_iter().map(|(l, kh, vh, qv)| sc.spawn(move || (l, qv, w(&kh), w(&vh)))).collect();
        hs.into_iter().map(|h| h.join().expect("archive widening")).collect()
    });
    for (layer, qv, kf, vf) in wide {
        let layer = &layer;
        match out.iter_mut().find(|o| o.0 == *layer) {
            Some(o) => {
                o.1.extend_from_slice(&qv);
                o.2.extend_from_slice(&kf);
                o.3.extend_from_slice(&vf);
            }
            None => out.push((*layer, qv, kf, vf)),
        }
    }
    drop(bufs);
    Some(out)
}

#[derive(Default)]
struct State {
    mats: HashMap<(u64, usize), Arc<DevMat>>,
    f32s: HashMap<(usize, usize), (Arc<Buf>, u64)>,
    plans: HashMap<(u64, usize, usize, bool), Arc<Plan>>,
    kv: HashMap<(u64, usize), KvMirror>,
    o1: HashMap<(u64, usize), O1Dev>,
    scratch: HashMap<&'static str, Buf>,
    /// Slabs the small f32 constants live in (`f32_buf`), the last one filling.
    slabs: Vec<Buf>,
    slab_used: usize,
    attn: HashMap<(usize, usize), AttnFuncs>,
    kv_hint: usize,
    /// The Qwen3.5-family hybrid arm's plans and recurrent states.
    hyb: hyb::HState,
    /// The MiniMax-H3 chains' resident weights and working buffers.
    h3: h3::H3State,
}

struct Ctx {
    cuctx: cs::CUcontext,
    stream: cs::CUstream,
    blas: bl::cublasHandle_t,
    f: Funcs,
    name: String,
    arch: Option<&'static str>,
    total_mem: u64,
    max_shared: usize,
    st: Mutex<State>,
    resident: AtomicU64,
    kv_bytes: AtomicU64,
    /// Scratch a sliding layer's window passes through on its way to the
    /// front of its mirror.
    move_tmp: Mutex<Option<Buf>>,
}

// SAFETY: the raw handles are used only under `st`'s lock, with the context
// made current on the calling thread first.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

macro_rules! kargs {
    ($($a:expr),* $(,)?) => {
        &mut [$(std::ptr::from_ref(&$a) as *mut c_void),*]
    };
}

/// The qwen4_exp stack (hyper-connection mixes, GDN, sparse attention, MoE)
/// as whole-chunk GEMMs; declared after `kargs!`, which it uses.
pub(crate) mod qwen4;

/// The Qwen3.5-family hybrid (GDN + gated full attention) behind the graph
/// contracts; declared after `kargs!`, which it uses.
mod hyb;

/// MiniMax-H3 stacks (the DiT, the VAE's ViT decoder) as resident chains
/// under a memory ceiling; declared after `kargs!`, which it uses.
mod h3;
pub use h3::{
    h3_attention, h3_chain, h3_conv1d, h3_conv_chain, h3_gemm_nt, h3_matmat, h3_prof_report, h3_release, h3_resident_bytes,
    flux_vae_chain, h3_te_chain, h3_vae_conv2d, lumina_block, ltx_chain, ltx_vae_chain, moss_chain, qi21_prefill, qi21_release,
    qi21_release_key, qi21_step, qi21_vae_decode, qwen_image_chain, zimage_prepare, zimage_release, zimage_step,
};

fn compile(src: &str, defs: Vec<String>, arch: Option<&'static str>) -> Res<cs::CUmodule> {
    // NVRTC takes seconds on a large module (the hybrid arm's: ~11 s on a
    // 3090's host); its PTX is kept on disk by (source, defines, arch) and
    // the driver's own cache keeps what it builds from that.
    let cached = ptx_cache_path(src, &defs, arch);
    let ptx = match cached.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(t) if t.contains(".entry") => t,
        _ => {
            let opts = cudarc::nvrtc::CompileOptions { options: defs, arch, ..Default::default() };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts).map_err(|e| format!("NVRTC: {e:?}"))?;
            let t = ptx.to_src();
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
    let text = CString::new(ptx).map_err(|e| e.to_string())?;
    unsafe { cu::module::load_data(text.as_ptr().cast()) }.map_err(derr)
}

/// Where a module's PTX is kept: `CMF_CUDA_CACHE` (a directory; `0` = no
/// cache), else `$XDG_CACHE_HOME/cortiq/cuda` or `~/.cache/cortiq/cuda`.
fn ptx_cache_path(src: &str, defs: &[String], arch: Option<&str>) -> Option<std::path::PathBuf> {
    let dir = match std::env::var("CMF_CUDA_CACHE") {
        Ok(v) if v == "0" => return None,
        Ok(v) if !v.is_empty() => std::path::PathBuf::from(v),
        _ => {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
            base.join("cortiq").join("cuda")
        }
    };
    // FNV-1a over everything the PTX depends on, and the crate version
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    eat(src.as_bytes());
    for d in defs {
        eat(d.as_bytes());
    }
    eat(arch.unwrap_or("").as_bytes());
    eat(env!("CARGO_PKG_VERSION").as_bytes());
    Some(dir.join(format!("{h:016x}.ptx")))
}

fn func(module: cs::CUmodule, name: &str) -> Res<cs::CUfunction> {
    let f = unsafe { cu::module::get_function(module, CString::new(name).unwrap()) }
        .map_err(|e| format!("kernel {name}: {e}"))?;
    KT_NAMES.lock().unwrap().get_or_insert_with(HashMap::new).insert(f as usize, name.to_string());
    if name.starts_with("q4_") {
        PDL_FUNCS.lock().unwrap().get_or_insert_with(std::collections::HashSet::new).insert(f as usize);
    }
    Ok(f)
}

/// Programmatic dependent launch (`CMF_QWEN_PDL=1`): the kernels of the qwen4 module (each starts with
/// `griddepcontrol.wait`) are launched so that one may begin to schedule while its predecessor drains.
static PDL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `CMF_QWEN_NOP=1` (a measurement): an empty kernel after every qwen4 launch, to see what a launch costs.
static NOP_F: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) fn nop_set(f: cs::CUfunction) {
    NOP_F.store(f as usize, Ordering::SeqCst);
}
static PDL_FUNCS: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(crate) fn pdl_set(on: bool) {
    PDL_ON.store(on, Ordering::SeqCst);
}

/// Kernel timing (`CMF_QWEN_NCU=<round>` of the qwen4 path): while on, every launch is bracketed by two
/// events and `kt_stop` prints the time per kernel. A profiler that cannot read the counters (a container)
/// still gets the time of each kernel this way.
static KT_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static KT_NAMES: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
// SAFETY: events are plain driver handles, used from the thread that owns the context lock.
struct KtEvent(usize, cs::CUevent, cs::CUevent);
unsafe impl Send for KtEvent {}
static KT_EVENTS: Mutex<Vec<KtEvent>> = Mutex::new(Vec::new());

pub(crate) fn kt_start() {
    KT_EVENTS.lock().unwrap().clear();
    KT_ON.store(true, Ordering::SeqCst);
}

pub(crate) fn kt_stop() {
    KT_ON.store(false, Ordering::SeqCst);
    let events = std::mem::take(&mut *KT_EVENTS.lock().unwrap());
    let names = KT_NAMES.lock().unwrap();
    let mut by: HashMap<String, (usize, f64)> = HashMap::new();
    let (mut total, mut n) = (0f64, 0usize);
    for KtEvent(f, a, b) in events {
        let ms = unsafe { cu::event::elapsed(a, b) }.unwrap_or(0.0) as f64;
        let name = names.as_ref().and_then(|m| m.get(&f)).cloned().unwrap_or_else(|| format!("{f:x}"));
        let e = by.entry(name).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
        total += ms;
        n += 1;
        unsafe {
            let _ = cu::event::destroy(a);
            let _ = cu::event::destroy(b);
        }
    }
    let mut rows: Vec<(String, (usize, f64))> = by.into_iter().collect();
    rows.sort_by(|x, y| y.1.1.total_cmp(&x.1.1));
    eprintln!("qwen4 cuda: kernel times of one round: {n} launches, {total:.2} ms between their events");
    for (name, (c, ms)) in rows.iter().take(40) {
        eprintln!("qwen4 cuda: kernel {name:<22} x{c:<5} {ms:8.3} ms  {:7.1} us each", ms * 1e3 / *c as f64);
    }
}

fn init() -> Res<Ctx> {
    unsafe {
        if !cs::is_culib_present() {
            return Err("the NVIDIA driver library (libcuda) was not found".into());
        }
        if !cudarc::nvrtc::sys::is_culib_present() {
            return Err("libnvrtc (CUDA toolkit) was not found: it compiles the kernels".into());
        }
        if !bl::is_culib_present() {
            return Err("libcublas (CUDA toolkit) was not found".into());
        }
    }
    cu::init().map_err(derr)?;
    let ordinal: i32 =
        std::env::var("CMF_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = cu::device::get(ordinal).map_err(derr)?;
    let cuctx = unsafe { cu::primary_ctx::retain(dev) }.map_err(derr)?;
    unsafe { cu::ctx::set_current(cuctx) }.map_err(derr)?;
    let name = cu::device::get_name(dev).unwrap_or_default();
    let total_mem = unsafe { cu::device::total_mem(dev) }.map_err(derr)? as u64;
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
    let stream = cu::stream::create(cu::stream::StreamKind::NonBlocking).map_err(derr)?;
    let module = compile(KERNELS_SRC, Vec::new(), arch)?;
    let f = Funcs {
        mv_multi: func(module, "mv_multi")?,
        rmsnorm_rows: func(module, "rmsnorm_rows")?,
        expand_q4tp: func(module, "expand_q4tp")?,
        expand_q8: func(module, "expand_q8")?,
        expand_q4tp_h: func(module, "expand_q4tp_h")?,
        expand_q8_h: func(module, "expand_q8_h")?,
        to_half: func(module, "to_half")?,
        act_half: func(module, "act_half")?,
        act_mul: func(module, "act_mul")?,
        rope_kv_rows: func(module, "rope_kv_rows")?,
        softmax_rows: func(module, "softmax_rows")?,
        attn_out_rows: func(module, "attn_out_rows")?,
        q_rows_strided: func(module, "q_rows_strided")?,
    };
    // More than the default 48 KB of shared memory a block where the card
    // allows it: `mv_multi` stages a whole input row there.
    let want = 96 << 10;
    let max_shared = if unsafe {
        cu::function::set_function_attribute(
            f.mv_multi,
            cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            want,
        )
    }
    .is_ok()
    {
        want as usize
    } else {
        48 << 10
    };
    let mut blas: bl::cublasHandle_t = std::ptr::null_mut();
    let s = unsafe { bl::cublasCreate_v2(&mut blas) };
    if s as u32 != 0 {
        return Err(format!("cuBLAS: create failed ({s:?})"));
    }
    let s = unsafe { bl::cublasSetStream_v2(blas, stream.cast()) };
    if s as u32 != 0 {
        return Err(format!("cuBLAS: set stream failed ({s:?})"));
    }
    tracing::info!(
        "cuda backend: {name} (compute {major}.{minor}, {:.1} GB), NVRTC kernels, cuBLAS",
        total_mem as f64 / 1e9
    );
    if std::env::var("CMF_CUDA_MEM").as_deref() == Ok("1") {
        let (free, total) = cu::mem_get_info().unwrap_or((0, 0));
        eprintln!("cuda mem after the context and cuBLAS: used {:.0} of {:.0} MiB", (total - free) as f64 / 1048576.0, total as f64 / 1048576.0);
    }
    Ok(Ctx {
        cuctx,
        stream,
        blas,
        f,
        name,
        arch,
        total_mem,
        max_shared,
        st: Mutex::new(State::default()),
        resident: AtomicU64::new(0),
        kv_bytes: AtomicU64::new(0),
        move_tmp: Mutex::new(None),
    })
}

fn ctx_or_reason() -> &'static Result<Ctx, String> {
    static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();
    CTX.get_or_init(|| {
        init().inspect_err(|e| {
            tracing::warn!("cuda backend unavailable — CPU fallback: {e}");
            if debug() {
                eprintln!("cuda backend unavailable: {e}");
            }
        })
    })
}

fn ctx() -> Option<&'static Ctx> {
    ctx_or_reason().as_ref().ok()
}

/// What `cortiq gpu` says about this backend: the card, or why there is none.
pub fn report() -> Vec<String> {
    match ctx_or_reason() {
        Ok(c) => vec![
            format!(
                "CUDA: {} ({}, {:.1} GB) — `CMF_GPU=cuda` runs dense attention stacks on it",
                c.name,
                c.arch.unwrap_or("default architecture"),
                c.total_mem as f64 / 1e9
            ),
            format!(
                "CUDA: kernels compiled by NVRTC, prompt GEMMs on cuBLAS{}",
                if selected() { " — selected" } else { "" }
            ),
        ],
        Err(e) => vec![format!("CUDA: unavailable — {e}")],
    }
}

/// Stage timers of `CMF_CUDA_PROF=1` (each stage is followed by a stream
/// sync, so the numbers are the card's, not the queue's).
const STAGES: [&str; 10] = [
    "norm", "expand", "gemm", "rope+kv", "attn scores", "attn softmax", "attn values", "act/attn out",
    "upload", "readback",
];
static STAGE_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
/// Decode stages of `CMF_CUDA_PROF=1`, a token's launches in order.
const DSTAGES: [&str; 10] = [
    "q/k/v + gate", "rope + append", "attn scores", "attn values", "attn finish", "o-proj", "gate + up",
    "down", "head", "readback",
];
static DSTAGE_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static TOKENS: AtomicU64 = AtomicU64::new(0);
/// Every decode token's wall time in microseconds (`CMF_CUDA_DEBUG=1`): a
/// mean hides the one token that waited on an allocation.
static TOKEN_US: Mutex<Vec<u32>> = Mutex::new(Vec::new());
static TOKEN_NS: AtomicU64 = AtomicU64::new(0);
static TOKEN_HOST_NS: AtomicU64 = AtomicU64::new(0);
static BATCH_ROWS: AtomicU64 = AtomicU64::new(0);
static BATCH_NS: AtomicU64 = AtomicU64::new(0);

impl Ctx {
    fn current(&self) -> Res<()> {
        unsafe { cu::ctx::set_current(self.cuctx) }.map_err(derr)
    }

    fn sync(&self) -> Res<()> {
        unsafe { cu::stream::synchronize(self.stream) }.map_err(derr)
    }

    /// Close a profiled stage: wait for the card and bill the time.
    fn stage(&self, s: usize, t0: &mut std::time::Instant) -> Res<()> {
        if prof_on() {
            self.sync()?;
            STAGE_NS[s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            *t0 = std::time::Instant::now();
        }
        Ok(())
    }

    /// Close a profiled decode stage.
    fn dstage(&self, s: usize, t0: &mut std::time::Instant) -> Res<()> {
        if prof_on() {
            self.sync()?;
            DSTAGE_NS[s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            *t0 = std::time::Instant::now();
        }
        Ok(())
    }

    unsafe fn launch(
        &self,
        f: cs::CUfunction,
        grid: (u32, u32),
        block: u32,
        shared: usize,
        params: &mut [*mut c_void],
    ) -> Res<()> {
        let timed = if KT_ON.load(Ordering::Relaxed) {
            let a = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT).ok();
            if let Some(a) = a {
                let _ = unsafe { cu::event::record(a, self.stream) };
            }
            a
        } else {
            None
        };
        let pdl = PDL_ON.load(Ordering::Relaxed)
            && PDL_FUNCS.lock().unwrap().as_ref().is_some_and(|set| set.contains(&(f as usize)));
        let r = if pdl {
            let mut attr: cs::CUlaunchAttribute = unsafe { std::mem::zeroed() };
            attr.id = cs::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attr.value.programmaticStreamSerializationAllowed = 1;
            let cfg = cs::CUlaunchConfig {
                gridDimX: grid.0.max(1),
                gridDimY: grid.1.max(1),
                gridDimZ: 1,
                blockDimX: block,
                blockDimY: 1,
                blockDimZ: 1,
                sharedMemBytes: shared as u32,
                hStream: self.stream,
                attrs: &mut attr,
                numAttrs: 1,
            };
            unsafe { cs::cuLaunchKernelEx(&cfg, f, params.as_mut_ptr(), std::ptr::null_mut()) }.result().map_err(derr)
        } else {
            unsafe {
                cu::launch_kernel(
                    f,
                    (grid.0.max(1), grid.1.max(1), 1),
                    (block, 1, 1),
                    shared as u32,
                    self.stream,
                    params,
                )
            }
            .map_err(derr)
        };
        let nop = NOP_F.load(Ordering::Relaxed);
        if nop != 0 && nop != f as usize && r.is_ok() {
            let (mut p, mut n) = (0u64, 0u64);
            let mut args = [std::ptr::from_mut(&mut p) as *mut c_void, std::ptr::from_mut(&mut n) as *mut c_void];
            let _ = unsafe { cu::launch_kernel(nop as cs::CUfunction, (1, 1, 1), (32, 1, 1), 0, self.stream, &mut args) };
        }
        if let Some(a) = timed {
            if let Ok(b) = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT) {
                let _ = unsafe { cu::event::record(b, self.stream) };
                KT_EVENTS.lock().unwrap().push(KtEvent(f as usize, a, b));
            }
        }
        r
    }

    unsafe fn launch_3d(
        &self,
        f: cs::CUfunction,
        grid: (u32, u32, u32),
        block: u32,
        shared: usize,
        params: &mut [*mut c_void],
    ) -> Res<()> {
        let timed = if KT_ON.load(Ordering::Relaxed) {
            let a = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT).ok();
            if let Some(a) = a {
                let _ = unsafe { cu::event::record(a, self.stream) };
            }
            a
        } else {
            None
        };
        let r = unsafe {
            cu::launch_kernel(
                f,
                (grid.0.max(1), grid.1.max(1), grid.2.max(1)),
                (block, 1, 1),
                shared as u32,
                self.stream,
                params,
            )
        }
        .map_err(derr);
        if let Some(a) = timed {
            if let Ok(b) = cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT) {
                let _ = unsafe { cu::event::record(b, self.stream) };
                KT_EVENTS.lock().unwrap().push(KtEvent(f as usize, a, b));
            }
        }
        r
    }

    /// C[m x n] = alpha * op(A) * op(B) + beta * C, column-major. C is f32;
    /// A and B are f32, or f16 under `Gemm::Half`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        &self,
        ta: bool,
        tb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: u64,
        lda: usize,
        b: u64,
        ldb: usize,
        beta: f32,
        cptr: u64,
        ldc: usize,
        mode: Gemm,
    ) -> Res<()> {
        let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
        let s = unsafe {
            if mode == Gemm::F32 {
                bl::cublasSgemm_v2(
                    self.blas,
                    op(ta),
                    op(tb),
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha,
                    a as *const f32,
                    lda as i32,
                    b as *const f32,
                    ldb as i32,
                    &beta,
                    cptr as *mut f32,
                    ldc as i32,
                )
            } else {
                let (ty, compute) = if mode == Gemm::Half {
                    (bl::cudaDataType::CUDA_R_16F, bl::cublasComputeType_t::CUBLAS_COMPUTE_32F)
                } else {
                    (bl::cudaDataType::CUDA_R_32F, bl::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32)
                };
                bl::cublasGemmEx(
                    self.blas,
                    op(ta),
                    op(tb),
                    m as i32,
                    n as i32,
                    k as i32,
                    std::ptr::from_ref(&alpha).cast(),
                    a as *const c_void,
                    ty,
                    lda as i32,
                    b as *const c_void,
                    ty,
                    ldb as i32,
                    std::ptr::from_ref(&beta).cast(),
                    cptr as *mut c_void,
                    bl::cudaDataType::CUDA_R_32F,
                    ldc as i32,
                    compute,
                    bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
        };
        if s as u32 != 0 {
            return Err(format!("cuBLAS gemm {m}x{n}x{k}: {s:?}"));
        }
        Ok(())
    }
}

impl Ctx {
    /// `count` GEMMs of one shape in a call: operand i sits `stride`
    /// elements after operand i - 1 (f32 operands; TF32 inside unless exact).
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_batched(
        &self,
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
        mode: Gemm,
    ) -> Res<()> {
        let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
        let beta = 0.0f32;
        let s = unsafe {
            if mode == Gemm::F32 {
                bl::cublasSgemmStridedBatched(
                    self.blas,
                    op(ta),
                    op(false),
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha,
                    a as *const f32,
                    lda as i32,
                    stride_a as i64,
                    b as *const f32,
                    ldb as i32,
                    stride_b as i64,
                    &beta,
                    cptr as *mut f32,
                    ldc as i32,
                    stride_c as i64,
                    count as i32,
                )
            } else {
                let ty = bl::cudaDataType::CUDA_R_32F;
                bl::cublasGemmStridedBatchedEx(
                    self.blas,
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
                    bl::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
        };
        if s as u32 != 0 {
            return Err(format!("cuBLAS batched gemm {count} x {m}x{n}x{k}: {s:?}"));
        }
        Ok(())
    }
}

impl State {
    /// A scratch buffer of at least `bytes` (contents not kept across growth).
    fn scr(&mut self, c: &Ctx, name: &'static str, bytes: usize) -> Res<u64> {
        if let Some(b) = self.scratch.get(name) {
            if b.bytes >= bytes {
                return Ok(b.ptr);
            }
            // queued work may still read the old one
            c.sync()?;
        }
        let b = Buf::new(bytes.next_multiple_of(1 << 20))?;
        let p = b.ptr;
        self.scratch.insert(name, b);
        Ok(p)
    }

    /// `bytes` (at most a quarter slab) of a slab, as a view: for buffers that
    /// live as long as the state.
    fn slab_buf(&mut self, bytes: usize) -> Res<Buf> {
        if bytes > SLAB / 4 {
            return Buf::new(bytes);
        }
        let at = self.slab_used.next_multiple_of(256);
        if self.slabs.is_empty() || at + bytes > SLAB {
            self.slabs.push(Buf::new(SLAB)?);
            self.slab_used = 0;
        }
        let at = self.slab_used.next_multiple_of(256);
        let v = self.slabs.last().unwrap().view(at, bytes);
        self.slab_used = at + bytes;
        Ok(v)
    }

    fn f32_buf(&mut self, data: &[f32]) -> Res<Arc<Buf>> {
        let key = (data.as_ptr() as usize, data.len());
        let fp = crate::gpu::fp_f32(data);
        if let Some((b, f)) = self.f32s.get(&key) {
            if *f == fp {
                return Ok(b.clone());
            }
        }
        // small constants (norms, gates, taps) share slabs: the card rounds
        // every allocation up to its page, which on a 64-layer model came to
        // several of them a layer
        let need = std::mem::size_of_val(data) + 64;
        let b = if need <= SLAB / 4 {
            let v = self.slab_buf(need)?;
            if !data.is_empty() {
                unsafe { cu::memcpy_htod_sync(v.ptr, data) }.map_err(derr)?;
            }
            Arc::new(v)
        } else {
            Arc::new(Buf::from_slice(data)?)
        };
        self.f32s.insert(key, (b.clone(), fp));
        Ok(b)
    }

    /// Resolve a graph weight to a resident device matrix.
    fn mat(
        &mut self,
        c: &Ctx,
        model: &Arc<CmfModel>,
        gw: &GraphW,
        rows: usize,
        cols: usize,
    ) -> Res<Arc<DevMat>> {
        if gw.prism != GraphPrismOp::None || gw.affine {
            return Err("Prism / affine weight".into());
        }
        if gw.kind == 4 {
            if gw.data.len() != rows * cols {
                return Err(format!("f32 weight of {} values, want {rows}x{cols}", gw.data.len()));
            }
            let w = self.f32_buf(gw.data)?;
            return Ok(Arc::new(DevMat { kind: 4, rows, cols, w, aux0: None, aux1: None, held: 0 }));
        }
        let key = (model.uid(), gw.idx);
        if let Some(m) = self.mats.get(&key) {
            if m.rows == rows && m.cols == cols && m.kind == gw.kind {
                return Ok(m.clone());
            }
        }
        let entry = model.tensors.get(gw.idx).ok_or("tensor index out of range")?;
        if entry.shape.first() != Some(&rows) || entry.shape.get(1) != Some(&cols) {
            return Err(format!(
                "tensor {} is {:?}, the graph wants {rows}x{cols}",
                entry.name, entry.shape
            ));
        }
        let bytes = model.entry_bytes(entry);
        let budget = vram_budget();
        let mut held = c.resident.load(Ordering::Relaxed);
        if held + bytes.len() as u64 > budget {
            // another model's weights are still resident: this one is the
            // one being run, so they go first
            c.sync()?;
            let uid = model.uid();
            self.plans.retain(|k, _| k.0 == uid);
            self.mats.retain(|k, m| {
                if k.0 != uid {
                    c.resident.fetch_sub(m.held, Ordering::Relaxed);
                }
                k.0 == uid
            });
            held = c.resident.load(Ordering::Relaxed);
        }
        if held + bytes.len() as u64 > budget {
            return Err(format!(
                "weights do not fit the card ({:.1} GB held, budget {:.1} GB)",
                held as f64 / 1e9,
                budget as f64 / 1e9
            ));
        }
        let m = dev_mat(gw.kind, bytes, rows, cols)?;
        c.resident.fetch_add(m.held, Ordering::Relaxed);
        let m = Arc::new(m);
        self.mats.insert(key, m.clone());
        Ok(m)
    }

    fn attn_funcs(&mut self, c: &Ctx, hpk: usize, hd: usize) -> Res<AttnFuncs> {
        if let Some(f) = self.attn.get(&(hpk, hd)) {
            return Ok(*f);
        }
        if hd % 32 != 0 || hd > 512 || hpk == 0 || 8 * hpk * hd * 4 > 44_000 {
            return Err(format!("attention geometry {hpk} heads per KV head x {hd} outside the kernels"));
        }
        let module = compile(ATTN_SRC, vec![format!("-DHPK={hpk}"), format!("-DHD={hd}")], c.arch)?;
        let f = AttnFuncs {
            scores: func(module, "attn_scores")?,
            values: func(module, "attn_values")?,
            finish: func(module, "attn_finish")?,
            far_insert: func(module, "o1_far_insert")?,
            far_prep: func(module, "o1_far_prep")?,
        };
        self.attn.insert((hpk, hd), f);
        Ok(f)
    }
}

/// What a call carries beside its layers.
struct CallGeom {
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    gemma: bool,
    scale: f32,
}

/// The identity of a call's descriptors: tensor indices, slice addresses and
/// geometry. Equal signatures reuse a plan.
fn signature(
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    g: &CallGeom,
    invf: &[f32],
) -> Vec<usize> {
    let mut s = Vec::with_capacity(layers.len() * 20 + 12);
    let w = |s: &mut Vec<usize>, gw: &GraphW| {
        s.extend([gw.idx, gw.kind as usize, gw.data.as_ptr() as usize, gw.data.len()]);
    };
    s.extend([model.uid() as usize, g.nh, g.nkv, g.hd, g.rd, g.hidden, g.gemma as usize]);
    s.extend([invf.as_ptr() as usize, invf.len(), final_norm.as_ptr() as usize, final_norm.len()]);
    if let Some((gw, rows)) = lm {
        w(&mut s, gw);
        s.push(rows);
    }
    for l in layers {
        s.extend([l.input_norm.as_ptr() as usize, l.post_norm.as_ptr() as usize]);
        if let GraphAttn::Full { wq, wk, wv, wo, geom, head_gate, .. } = &l.attn {
            w(&mut s, wq);
            w(&mut s, wk);
            w(&mut s, wv);
            w(&mut s, wo);
            if let Some(hg) = head_gate {
                w(&mut s, hg);
            }
            if let Some(g) = geom {
                s.extend([
                    g.nkv,
                    g.hd,
                    g.rd,
                    g.window.unwrap_or(0),
                    g.invf.as_ptr() as usize,
                    g.rope_scale.to_bits() as usize,
                ]);
            }
        }
        if let GraphFfn::Dense { gate, up, down, act, .. } = &l.ffn {
            w(&mut s, gate);
            w(&mut s, up);
            w(&mut s, down);
            s.push(act.code() as usize);
        }
    }
    s
}

/// Resolve a stack for the device. Err = this stack is not the shape the
/// backend runs (the caller declines; nothing was mutated).
#[allow(clippy::too_many_arguments)]
fn build_plan(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    g: &CallGeom,
    invf: &[f32],
    sig: Vec<usize>,
) -> Res<Plan> {
    // An empty stack with a head is the layer-less head pass (the first
    // token's logits after a prompt).
    if (layers.is_empty() && lm.is_none()) || g.nkv == 0 || g.nh % g.nkv != 0 {
        return Err("empty stack or head geometry".into());
    }
    let hpk = g.nh / g.nkv;
    let mut consts = Vec::new();
    let mut note = |d: &[f32]| consts.push((d.as_ptr() as usize, d.len(), crate::gpu::fp_f32(d)));
    let mut rope: Vec<Vec<f32>> = Vec::new();
    let mut rope_offs: Vec<usize> = Vec::new();
    let mut rope_row = 0usize;
    struct Pre {
        q: Arc<DevMat>,
        k: Arc<DevMat>,
        v: Arc<DevMat>,
        o: Arc<DevMat>,
        hg: Option<Arc<DevMat>>,
        gate: Arc<DevMat>,
        up: Arc<DevMat>,
        down: Arc<DevMat>,
        in_norm: Arc<Buf>,
        post_norm: Arc<Buf>,
        act: i32,
        inter: usize,
        nkv: usize,
        hd: usize,
        half: usize,
        window: usize,
        rope_scale: f32,
        rope_off: usize,
    }
    let mut pre: Vec<Pre> = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        if l.mhc.is_some() || l.attn_out_norm.is_some() || l.ffn_out_norm.is_some() || l.layer_scale.is_some() {
            return Err(format!("layer {li}: mHC streams, sandwich norms or a layer scale"));
        }
        if l.input_norm.len() != g.hidden || l.post_norm.len() != g.hidden {
            return Err(format!("layer {li}: norm width"));
        }
        let GraphAttn::Full {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            late_qk_norm: _,
            bias,
            output_gate,
            geom,
            head_gate,
            ..
        } = &l.attn
        else {
            return Err(format!("layer {li}: a token mixer other than full attention"));
        };
        if q_norm.is_some() || k_norm.is_some() || bias.is_some() || *output_gate {
            return Err(format!("layer {li}: Q/K norms, projection biases or an element-wise output gate"));
        }
        let (nkv, hd, rd, lf, window, rope_scale): (usize, usize, usize, &[f32], usize, f32) = match geom {
            Some(x) => {
                if x.dv != x.hd || x.sink.is_some() || x.v_norm || x.dash.is_some() || x.vres.is_some() {
                    return Err(format!("layer {li}: narrow V, sinks, V norm, DASH or a value residual"));
                }
                (x.nkv, x.hd, x.rd, x.invf, x.window.unwrap_or(0), x.rope_scale)
            }
            None => (g.nkv, g.hd, g.rd, invf, 0, 1.0),
        };
        if nkv != g.nkv || hd != g.hd {
            return Err(format!("layer {li}: its own head count or width"));
        }
        if rd % 2 != 0 || rd > hd || lf.len() < rd / 2 {
            return Err(format!("layer {li}: rotary width {rd}"));
        }
        let half = rd / 2;
        let table = &lf[..half];
        let rope_off = match rope.iter().position(|t| t.as_slice() == table) {
            Some(i) => rope_offs[i],
            None => {
                rope.push(table.to_vec());
                rope_offs.push(rope_row);
                rope_row += half * 2;
                rope_row - half * 2
            }
        };
        let GraphFfn::Dense { gate, up, down, act, limit, hot } = &l.ffn else {
            return Err(format!("layer {li}: an FFN other than a dense one"));
        };
        if *limit != 0.0 || hot.is_some() {
            return Err(format!("layer {li}: a clamped FFN or hot neurons"));
        }
        let act = match act {
            GraphAct::Silu => 0,
            GraphAct::GeluErf => 1,
            GraphAct::GeluTanh => 2,
            GraphAct::SwiGluOaiM3 => return Err(format!("layer {li}: SwiGLU-OAI activation")),
        };
        let inter = model
            .tensors
            .get(gate.idx)
            .and_then(|e| e.shape.first().copied())
            .ok_or("gate tensor")?;
        let q = st.mat(c, model, wq, g.nh * hd, g.hidden)?;
        let k = st.mat(c, model, wk, nkv * hd, g.hidden)?;
        let v = st.mat(c, model, wv, nkv * hd, g.hidden)?;
        let o = st.mat(c, model, wo, g.hidden, g.nh * hd)?;
        let hg = match head_gate {
            Some(hgw) => Some(st.mat(c, model, hgw, g.nh, g.hidden)?),
            None => None,
        };
        let gate = st.mat(c, model, gate, inter, g.hidden)?;
        let up = st.mat(c, model, up, inter, g.hidden)?;
        let down = st.mat(c, model, down, g.hidden, inter)?;
        note(l.input_norm);
        note(l.post_norm);
        let in_norm = st.f32_buf(l.input_norm)?;
        let post_norm = st.f32_buf(l.post_norm)?;
        pre.push(Pre {
            q,
            k,
            v,
            o,
            hg,
            gate,
            up,
            down,
            in_norm,
            post_norm,
            act,
            inter,
            nkv,
            hd,
            half,
            window,
            rope_scale,
            rope_off,
        });
    }
    // the widest input row any `mv_multi` launch stages
    let qmax = g.nh * g.hd;
    let kvmax = g.nkv * g.hd;
    let intermax = pre.iter().map(|p| p.inter).max().unwrap_or(0);
    let widest = g.hidden.max(qmax).max(intermax);
    if mv_shared_bytes(widest) > c.max_shared {
        return Err(format!("a {widest}-wide row does not fit a block's shared memory"));
    }
    st.attn_funcs(c, hpk, g.hd)?;
    let (lm_mat, lm_rows) = match lm {
        Some((gw, rows)) => (Some(st.mat(c, model, gw, rows, g.hidden)?), rows),
        None => (None, 0),
    };
    let final_norm_buf = if lm_mat.is_some() {
        if final_norm.len() != g.hidden {
            return Err("final norm width".into());
        }
        note(final_norm);
        Some(st.f32_buf(final_norm)?)
    } else {
        None
    };
    let h = Buf::new(g.hidden * 4)?;
    let q = Buf::new(qmax * 4)?;
    let qr = Buf::new(qmax * 4)?;
    let k = Buf::new(kvmax * 4)?;
    let v = Buf::new(kvmax * 4)?;
    let g16 = Buf::new(g.nh * 4)?;
    let ao = Buf::new(qmax * 4)?;
    let gg = Buf::new(intermax * 4)?;
    let uu = Buf::new(intermax * 4)?;
    let logits = Buf::new(lm_rows.max(1) * 4)?;
    let tab = Buf::new(rope_row.max(1) * 4)?;
    let mut out = Vec::with_capacity(pre.len());
    for p in pre {
        let (mut jobs, mut blk) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk, &p.q, q.ptr, 0);
        push_job(&mut jobs, &mut blk, &p.k, k.ptr, 0);
        push_job(&mut jobs, &mut blk, &p.v, v.ptr, 0);
        if let Some(hg) = &p.hg {
            push_job(&mut jobs, &mut blk, hg, g16.ptr, 0);
        }
        let (n_qkv, blk_qkv, jobs_qkv) = (jobs.len() as i32, blk, Buf::from_slice(&jobs)?);
        let (mut jobs, mut blk_o) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk_o, &p.o, h.ptr, h.ptr);
        let jobs_o = Buf::from_slice(&jobs)?;
        let (mut jobs, mut blk_gu) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk_gu, &p.gate, gg.ptr, 0);
        push_job(&mut jobs, &mut blk_gu, &p.up, uu.ptr, 0);
        let jobs_gu = Buf::from_slice(&jobs)?;
        let (mut jobs, mut blk_dn) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk_dn, &p.down, h.ptr, h.ptr);
        let jobs_dn = Buf::from_slice(&jobs)?;
        // a small f32 head gate: an f16 copy for the prompt's f16 GEMMs
        let hg_h = match &p.hg {
            Some(hg) if hg.kind == 4 && hg.rows * hg.cols % 4 == 0 => {
                let hb = Buf::new(hg.rows * hg.cols * 2)?;
                let quads = (hg.rows * hg.cols / 4) as u64;
                unsafe {
                    c.launch(c.f.to_half, (quads.div_ceil(256) as u32, 1), 256, 0, kargs!(hg.w.ptr, hb.ptr, quads))?;
                }
                Some(hb)
            }
            _ => None,
        };
        out.push(PlanLayer {
            q: p.q,
            k: p.k,
            v: p.v,
            o: p.o,
            hg: p.hg,
            hg_h,
            gate: p.gate,
            up: p.up,
            down: p.down,
            in_norm: p.in_norm,
            post_norm: p.post_norm,
            act: p.act,
            inter: p.inter,
            nkv: p.nkv,
            hd: p.hd,
            half: p.half,
            window: p.window,
            rope_scale: p.rope_scale,
            rope_off: p.rope_off,
            jobs_qkv,
            n_qkv,
            blk_qkv,
            jobs_o,
            blk_o,
            jobs_gu,
            blk_gu,
            jobs_dn,
            blk_dn,
        });
    }
    let lm = match lm_mat {
        Some(m) => {
            let (mut jobs, mut blk) = (Vec::new(), 0u32);
            push_job(&mut jobs, &mut blk, &m, logits.ptr, 0);
            Some((m, Buf::from_slice(&jobs)?, blk))
        }
        None => None,
    };
    Ok(Plan {
        layers: out,
        sig,
        consts,
        rope,
        rope_row,
        hpk,
        final_norm: final_norm_buf,
        lm,
        h,
        q,
        qr,
        k,
        v,
        g16,
        ao,
        gg,
        uu,
        logits,
        tab,
        qmax,
        kvmax,
        intermax,
    })
}

/// The plan of a call: the cached one while its descriptors (and, at a
/// sequence start, the bytes of its f32 constants) are unchanged.
#[allow(clippy::too_many_arguments)]
fn plan_for(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    layers: &[GraphLayer],
    layer_base: usize,
    lm: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    g: &CallGeom,
    invf: &[f32],
    first_pos: usize,
) -> Res<Arc<Plan>> {
    let key = (model.uid(), layer_base, layers.len(), lm.is_some());
    let sig = signature(model, layers, lm, final_norm, g, invf);
    if let Some(p) = st.plans.get(&key) {
        if p.sig == sig {
            // SAFETY: the signature holds these very addresses and lengths,
            // and the caller's slices are alive for the call.
            let same = first_pos != 0
                || p.consts.iter().all(|&(ptr, len, fp)| {
                    crate::gpu::fp_f32(unsafe { std::slice::from_raw_parts(ptr as *const f32, len) }) == fp
                });
            if same {
                return Ok(p.clone());
            }
        }
    }
    // a replaced plan's buffers may still be read by queued work
    c.sync()?;
    let t0 = std::time::Instant::now();
    let p = Arc::new(build_plan(c, st, model, layers, lm, final_norm, g, invf, sig)?);
    tracing::info!(
        "cuda graph: {} layers resident, {:.2} GB of weights, built in {:.1} s",
        p.layers.len(),
        c.resident.load(Ordering::Relaxed) as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    st.plans.insert(key, p.clone());
    Ok(p)
}

fn decline(site: &str, why: &str) {
    use std::sync::atomic::AtomicBool;
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!("cuda {site} graph declined: {why}");
    }
    if debug() {
        eprintln!("cuda {site} graph declined: {why}");
    }
}

/// (sin, cos) per position and frequency for every distinct table of the
/// plan, laid out `[table][row][pair][2]` — `attention::rope_table`'s values.
fn rope_tables(plan: &Plan, positions: impl Iterator<Item = usize> + Clone, rows: usize) -> Vec<f32> {
    let mut tab = vec![0f32; plan.rope_row * rows];
    let mut off = 0usize;
    for t in &plan.rope {
        for (b, p) in positions.clone().enumerate() {
            let dst = &mut tab[off * rows + b * t.len() * 2..][..t.len() * 2];
            for (i, &f) in t.iter().enumerate() {
                let (s, co) = (p as f32 * f).sin_cos();
                dst[2 * i] = s;
                dst[2 * i + 1] = co;
            }
        }
        off += t.len() * 2;
    }
    tab
}

/// A sealed layer's far-field state on the card: uploaded unless this seal
/// epoch's is already there. Err: the state is outside what the kernels take.
#[allow(clippy::too_many_arguments)]
fn o1_ensure(
    c: &Ctx,
    st: &mut State,
    kv_id: u64,
    layer: usize,
    views: &[crate::nystrom::O1DeviceView<'_>],
    epoch: u64,
    hpk: usize,
    nkv: usize,
    hd: usize,
) -> Res<()> {
    if st.o1.get(&(kv_id, layer)).is_some_and(|d| d.epoch == epoch) {
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
    let mut nk = vec![0f32; (ns + w) * row];
    let mut nv = vec![0f32; (ns + w) * row];
    let (mut kt, mut qt, mut mu, mut mz, mut th) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (g, v) in views.iter().enumerate() {
        let have = (v.win_k.len() / hd).min(v.win_v.len() / hd).min(w);
        if v.sink_k.len() != ns * hd || v.sink_v.len() != ns * hd || v.k_tilde.len() != m * hd || have < v.win_len {
            return Err("far field: sink, window or landmark rows of an unexpected length".into());
        }
        for r in 0..ns {
            nk[(r * nkv + g) * hd..][..hd].copy_from_slice(&v.sink_k[r * hd..(r + 1) * hd]);
            nv[(r * nkv + g) * hd..][..hd].copy_from_slice(&v.sink_v[r * hd..(r + 1) * hd]);
        }
        for r in 0..have {
            nk[((ns + r) * nkv + g) * hd..][..hd].copy_from_slice(&v.win_k[r * hd..(r + 1) * hd]);
            nv[((ns + r) * nkv + g) * hd..][..hd].copy_from_slice(&v.win_v[r * hd..(r + 1) * hd]);
        }
        kt.extend_from_slice(v.k_tilde);
        for h in &v.heads {
            if h.q_tilde.len() != m * hd
                || h.mu.len() != m * m
                || h.m_max.len() != m
                || h.z_hat.len() != m
                || h.t_hat.len() != m * hd
            {
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
    let bytes = ((nk.len() + nv.len() + kt.len() + qt.len() + mu.len() + mz.len() + th.len()) * 4) as u64;
    let dev = O1Dev {
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
    };
    if debug() {
        eprintln!(
            "cuda: far field of layer {layer} on the card (epoch {epoch}): {ns} exact rows, window {} of {w}, {m} landmarks, {} keys in the far field, {:.1} MB",
            dev.win_len,
            dev.far_len,
            bytes as f64 / 1e6
        );
    }
    st.o1.insert((kv_id, layer), dev);
    Ok(())
}

/// A sealed layer takes a row (`nystrom.rs` advance): the ring row it
/// replaces goes into the far accumulators first. The near row the new K and
/// V belong in.
unsafe fn o1_advance(c: &Ctx, af: &AttnFuncs, d: &mut O1Dev, nkv: usize, hpk: usize, position: usize) -> Res<i32> {
    let full = d.win_len == d.w;
    let row = (d.ns + if full { d.win_head } else { d.win_len }) as i32;
    if full {
        let (m_i, nkv_i) = (d.m as i32, nkv as i32);
        unsafe {
            c.launch(
                af.far_insert,
                (nkv as u32, 1),
                (hpk * d.m) as u32,
                0,
                kargs!(d.nk.ptr, d.nv.ptr, d.qt.ptr, d.mz.ptr, d.th.ptr, row, m_i, nkv_i, d.scale),
            )?;
        }
        d.far_len += 1;
        d.win_head = (d.win_head + 1) % d.w;
    } else {
        d.win_len += 1;
    }
    d.next_pos = Some(position + 1);
    Ok(row)
}

/// A row's attention on a sealed layer (`nystrom.rs` step): the near rows
/// and the far part under one shift. `q` is the row's rotated queries
/// [nh][hd], `ao` takes [nh][hd].
#[allow(clippy::too_many_arguments)]
unsafe fn o1_read(
    c: &Ctx,
    st: &mut State,
    af: &AttnFuncs,
    key: (u64, usize),
    q: u64,
    gate: u64,
    ao: u64,
    nh: usize,
    nkv: usize,
    hd: usize,
    t0: &mut std::time::Instant,
) -> Res<()> {
    let d = &st.o1[&key];
    let n = d.ns + d.win_len;
    let (m_i, scale, far_on, rect) = (d.m as i32, d.scale, (d.far_len > 0) as i32, d.rect_fm as i32);
    let (nkp, nvp, ktp, mup, mzp, thp) = (d.nk.ptr, d.nv.ptr, d.kt.ptr, d.mu.ptr, d.mz.ptr, d.th.ptr);
    // one block of P and E and one maximum more than the near rows' blocks
    let rb = 64usize;
    let ncap = (n + rb).next_multiple_of(4096);
    let nbcap = ncap / rb;
    let nblk = n.div_ceil(rb);
    let s_p = st.scr(c, "dS", nh * ncap * 4)?;
    let mx_p = st.scr(c, "dMx", nh * nbcap * 4)?;
    let p_p = st.scr(c, "dP", nbcap * nh * hd * 4)?;
    let e_p = st.scr(c, "dE", nbcap * nh * 4)?;
    let (n_i, nh_i, nkv_i, rb_i) = (n as i32, nh as i32, nkv as i32, rb as i32);
    let (ncap_i, nbcap_i, nblk_i) = (ncap as i32, nbcap as i32, nblk as i32);
    unsafe {
        c.launch(
            af.scores,
            (nblk as u32, nkv as u32),
            256,
            0,
            kargs!(q, nkp, s_p, mx_p, n_i, 0i32, nkv_i, scale, rb_i, ncap_i, nbcap_i),
        )?;
        c.launch(
            af.far_prep,
            (nh as u32, 1),
            64,
            0,
            kargs!(q, ktp, mup, mzp, thp, mx_p, p_p, e_p, nblk_i, nbcap_i, m_i, nh_i, scale, far_on, rect),
        )?;
        c.dstage(2, t0)?;
        c.launch(
            af.values,
            (nblk as u32, nkv as u32),
            256,
            0,
            kargs!(s_p, mx_p, nvp, p_p, e_p, n_i, 0i32, nkv_i, rb_i, ncap_i, nblk_i + 1, nbcap_i),
        )?;
        c.dstage(3, t0)?;
        c.launch(af.finish, (nh as u32, 1), 256, 0, kargs!(p_p, e_p, gate, ao, nblk_i + 1, nh_i))?;
        c.dstage(4, t0)?;
    }
    Ok(())
}

/// Whole-token decode graph: every layer and the head on the card, the
/// hidden resident, one readback. A sealed layer (`o1`) reads its far field
/// in place of its mirror.
#[allow(clippy::too_many_arguments)]
pub fn forward_token_graph(
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
    let Some(c) = ctx() else { return TokenGraphOutcome::Declined };
    if hyb::wants(layers, nh, nkv, hd) {
        return hyb::token(
            c, model, kv_id, layers, o1, o1_epoch, invf, h, nh, nkv, hd, attn_scale, rd, hidden, position, cap, gemma, eps,
            lm_head, final_norm, logits, loop_norm_at, steps, layers_run, layer_base, hidden_too,
        );
    }
    let mut st = c.st.lock().unwrap();
    // A far-field state the card has moved has no host copy to return to
    // (the host's is the seal's): from there a refusal is a failure, not a
    // hand-over to the host path.
    let moved = o1.iter().any(Option::is_some) && st.o1.iter().any(|(k, d)| k.0 == kv_id && d.next_pos.is_some());
    let refused = move || if moved { TokenGraphOutcome::Failed } else { TokenGraphOutcome::Declined };
    if steps != 1 || !loop_norm_at.is_empty() || (!o1.is_empty() && o1.len() < layers.len()) {
        decline("token", "multi-step burst, a mid-stack norm or far-field views short of the layers");
        return refused();
    }
    let sealed = |li: usize| o1.get(li).and_then(|v| v.as_ref());
    if position >= cap || h.len() != hidden {
        decline("token", "position past the context cap");
        return refused();
    }
    let t_tok = std::time::Instant::now();
    if c.current().is_err() {
        return refused();
    }
    let g = CallGeom { nh, nkv, hd, rd, hidden, gemma, scale: attn_scale };
    let plan = match plan_for(c, &mut st, model, layers, layer_base, lm_head, final_norm, &g, invf, position) {
        Ok(p) => p,
        Err(e) => {
            decline("token", &e);
            return refused();
        }
    };
    // Sealed layers: the far field goes up once per seal epoch. A state the
    // graph has already moved cannot go back to the host's stale copy.
    for (li, pl) in plan.layers.iter().enumerate() {
        let Some(views) = sealed(li) else { continue };
        let key = (kv_id, layer_base + li);
        let moved = st.o1.get(&key).filter(|d| d.epoch == o1_epoch).and_then(|d| d.next_pos);
        let res = match moved {
            Some(next) if next != position => Err(format!("far field: the card is at row {next}, the call at {position}")),
            _ => o1_ensure(c, &mut st, kv_id, layer_base + li, views, o1_epoch, plan.hpk, pl.nkv, pl.hd),
        };
        if let Err(e) = res {
            decline("token", &e);
            return refused();
        }
        // a sealed layer reads no mirror again
        if let Some(m) = st.kv.remove(&key) {
            let _ = c.sync();
            c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
        }
    }
    // Mirrors: exactly `position` rows each, before anything is launched.
    for (li, (pl, l)) in plan.layers.iter().zip(layers).enumerate() {
        if sealed(li).is_some() {
            continue;
        }
        let GraphAttn::Full { cpu_k, cpu_v, cpu_base, .. } = &l.attn else { unreachable!() };
        let row = pl.nkv * pl.hd;
        let m = st.kv.entry((kv_id, layer_base + li)).or_insert_with(|| KvMirror::new(row, pl.window));
        if m.row != row || m.window != pl.window {
            // another geometry under the same id: its rows mean nothing here
            let _ = c.sync();
            c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
            *m = KvMirror::new(row, pl.window);
        }
        match m.sync_to(c, cpu_k, cpu_v, *cpu_base, pl.nkv, pl.hd, position).and_then(|ok| {
            if ok {
                m.reserve(c, 1, 0)?;
            }
            Ok(ok)
        }) {
            Ok(true) => {}
            Ok(false) => {
                decline("token", "a layer's K/V rows are neither on the card nor on the host");
                return refused();
            }
            Err(e) => {
                decline("token", &e);
                return refused();
            }
        }
    }
    let mut run = || -> Res<()> {
        let f = &c.f;
        let af = st.attn[&(plan.hpk, hd)];
        let nullp = 0u64;
        let (eps_f, gemma_i) = (eps, gemma as i32);
        let no_act = -1i32;
        let shared = mv_shared_bytes;
        unsafe {
            cu::memcpy_htod_async(plan.h.ptr, h, c.stream).map_err(derr)?;
            let tab = rope_tables(&plan, std::iter::once(position), 1);
            if !tab.is_empty() {
                cu::memcpy_htod_async(plan.tab.ptr, &tab, c.stream).map_err(derr)?;
            }
        }
        let mut t0 = std::time::Instant::now();
        c.dstage(9, &mut t0)?;
        for (li, pl) in plan.layers.iter().enumerate() {
            let (hidden_i, q_cols, inter_i) = (hidden as i32, (nh * pl.hd) as i32, pl.inter as i32);
            // q, k, v (and the head gate) of the normed row
            unsafe {
                c.launch(
                    f.mv_multi,
                    (pl.blk_qkv, 1),
                    256,
                    shared(hidden),
                    kargs!(pl.jobs_qkv.ptr, pl.n_qkv, plan.h.ptr, nullp, no_act, pl.in_norm.ptr, eps_f, gemma_i, hidden_i),
                )?;
            }
            c.dstage(0, &mut t0)?;
            let (nh_i, nkv_i, hd_i, half_i) = (nh as i32, pl.nkv as i32, pl.hd as i32, pl.half as i32);
            let tabp = plan.tab.ptr + (pl.rope_off * 4) as u64;
            let gate_p = if pl.hg.is_some() { plan.g16.ptr } else { 0u64 };
            let rb = 64usize;
            let rb_i = rb as i32;
            if sealed(li).is_some() {
                // The far field (`nystrom.rs` step_group): the ring row this
                // token replaces goes into the far accumulators, the token's
                // K and V take its place, and the row reads the near rows
                // and the far part under one shift.
                let key = (kv_id, layer_base + li);
                let d = st.o1.get_mut(&key).unwrap();
                let (nkp, nvp) = (d.nk.ptr, d.nv.ptr);
                unsafe {
                    let row = o1_advance(c, &af, d, pl.nkv, plan.hpk, position)?;
                    c.launch(
                        f.rope_kv_rows,
                        (1, 1),
                        256,
                        0,
                        kargs!(plan.q.ptr, plan.qr.ptr, plan.k.ptr, plan.v.ptr, nkp, nvp, tabp, nh_i, nkv_i, hd_i, half_i, pl.rope_scale, row, 1i32),
                    )?;
                    c.dstage(1, &mut t0)?;
                    o1_read(c, &mut st, &af, key, plan.qr.ptr, gate_p, plan.ao.ptr, nh, pl.nkv, pl.hd, &mut t0)?;
                }
            } else {
                let m = st.kv.get_mut(&(kv_id, layer_base + li)).unwrap();
                let slot0 = (m.stored - m.base) as i32;
                let (kp, vp) = (m.kptr(), m.vptr());
                m.stored += 1;
                // the rows this query sees
                let first = if pl.window > 0 { m.stored.saturating_sub(pl.window) } else { 0 }.max(m.base);
                let n = m.stored - first;
                let r0 = (first - m.base) as i32;
                let n_i = n as i32;
                // Three launches a layer, not one: a single kernel that did the
                // rotary turn, the append and the whole attend for a 512-row
                // window with a block per KV head measured 55 us a layer on an
                // A40 against ~25 here — four blocks leave each warp 64 rows
                // deep in loads that wait on memory, where these spread the rows
                // over 32 blocks.
                unsafe {
                    c.launch(
                        f.rope_kv_rows,
                        (1, 1),
                        256,
                        0,
                        kargs!(plan.q.ptr, plan.qr.ptr, plan.k.ptr, plan.v.ptr, kp, vp, tabp, nh_i, nkv_i, hd_i, half_i, pl.rope_scale, slot0, 1i32),
                    )?;
                }
                c.dstage(1, &mut t0)?;
                let ncap = n.next_multiple_of(4096);
                let nbcap = ncap / rb;
                let nblk = n.div_ceil(rb);
                let s_p = st.scr(c, "dS", nh * ncap * 4)?;
                let mx_p = st.scr(c, "dMx", nh * nbcap * 4)?;
                let p_p = st.scr(c, "dP", nbcap * nh * pl.hd * 4)?;
                let e_p = st.scr(c, "dE", nbcap * nh * 4)?;
                let (ncap_i, nbcap_i, nblk_i) = (ncap as i32, nbcap as i32, nblk as i32);
                unsafe {
                    c.launch(
                        af.scores,
                        (nblk as u32, pl.nkv as u32),
                        256,
                        0,
                        kargs!(plan.qr.ptr, kp, s_p, mx_p, n_i, r0, nkv_i, g.scale, rb_i, ncap_i, nbcap_i),
                    )?;
                    c.dstage(2, &mut t0)?;
                    c.launch(
                        af.values,
                        (nblk as u32, pl.nkv as u32),
                        256,
                        0,
                        kargs!(s_p, mx_p, vp, p_p, e_p, n_i, r0, nkv_i, rb_i, ncap_i, nblk_i, nbcap_i),
                    )?;
                    c.dstage(3, &mut t0)?;
                    c.launch(af.finish, (nh as u32, 1), 256, 0, kargs!(p_p, e_p, gate_p, plan.ao.ptr, nblk_i, nh_i))?;
                    c.dstage(4, &mut t0)?;
                }
            }
            unsafe {
                // h += Wo ao
                c.launch(
                    f.mv_multi,
                    (pl.blk_o, 1),
                    256,
                    shared(nh * pl.hd),
                    kargs!(pl.jobs_o.ptr, 1i32, plan.ao.ptr, nullp, no_act, nullp, eps_f, gemma_i, q_cols),
                )?;
                c.dstage(5, &mut t0)?;
                // gate, up of the normed row
                c.launch(
                    f.mv_multi,
                    (pl.blk_gu, 1),
                    256,
                    shared(hidden),
                    kargs!(pl.jobs_gu.ptr, 2i32, plan.h.ptr, nullp, no_act, pl.post_norm.ptr, eps_f, gemma_i, hidden_i),
                )?;
                c.dstage(6, &mut t0)?;
                // h += Wd (act(gate) * up). The activation is its own small
                // launch: folded into `mv_multi`'s staging, every one of the
                // down projection's blocks ran it over the whole row (an A40:
                // 63 against 42 us a layer).
                let n_inter = pl.inter as u64;
                c.launch(f.act_mul, (n_inter.div_ceil(256) as u32, 1), 256, 0, kargs!(plan.gg.ptr, plan.uu.ptr, pl.act, n_inter))?;
                c.launch(
                    f.mv_multi,
                    (pl.blk_dn, 1),
                    256,
                    shared(pl.inter),
                    kargs!(pl.jobs_dn.ptr, 1i32, plan.gg.ptr, nullp, no_act, nullp, eps_f, gemma_i, inter_i),
                )?;
                c.dstage(7, &mut t0)?;
            }
        }
        let t_host = t_tok.elapsed();
        unsafe {
            if let Some((_, jobs, blk)) = &plan.lm {
                let hidden_i = hidden as i32;
                let fnorm = plan.final_norm.as_ref().unwrap().ptr;
                c.launch(
                    f.mv_multi,
                    (*blk, 1),
                    256,
                    shared(hidden),
                    kargs!(jobs.ptr, 1i32, plan.h.ptr, nullp, no_act, fnorm, eps_f, gemma_i, hidden_i),
                )?;
                c.dstage(8, &mut t0)?;
                logits.resize(plan.lm.as_ref().unwrap().0.rows, 0.0);
                cu::memcpy_dtoh_async(logits.as_mut_slice(), plan.logits.ptr, c.stream).map_err(derr)?;
                if hidden_too {
                    cu::memcpy_dtoh_async(h, plan.h.ptr, c.stream).map_err(derr)?;
                }
            } else {
                cu::memcpy_dtoh_async(h, plan.h.ptr, c.stream).map_err(derr)?;
            }
        }
        c.sync()?;
        c.dstage(9, &mut t0)?;
        if debug() {
            TOKEN_US.lock().unwrap().push(t_tok.elapsed().as_micros() as u32);
        }
        // steady tokens only: the first ones build the plan and warm the card
        if TOKENS.fetch_add(1, Ordering::Relaxed) >= 16 {
            TOKEN_NS.fetch_add(t_tok.elapsed().as_nanos() as u64, Ordering::Relaxed);
            TOKEN_HOST_NS.fetch_add(t_host.as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(())
    };
    match run() {
        Ok(()) => {
            if let Some(lr) = layers_run {
                *lr = layers.len();
            }
            TokenGraphOutcome::Completed
        }
        Err(e) => {
            tracing::error!("cuda token graph failed: {e}");
            let _ = c.sync();
            TokenGraphOutcome::Failed
        }
    }
}

thread_local! {
    static BATCH_DEFER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The next batch graph call on this thread leaves its rows on the card (a
/// prompt chunk nobody reads back).
pub fn set_batch_defer(on: bool) {
    BATCH_DEFER.with(|d| d.set(on));
}

/// Rows the prompt is about to append: full layers size their mirrors once.
pub fn set_batch_kv_hint(rows: usize) {
    if let Some(c) = ctx() {
        c.st.lock().unwrap().kv_hint = rows;
    }
}

/// Batched prompt graph: `k` contiguous positions through every layer, the
/// projections and FFNs as GEMMs over weights expanded once per chunk.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_graph_at(
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
    spec: Option<crate::gpu::SpecTail<'_>>,
    layers_run: Option<&mut usize>,
) -> BatchGraphOutcome {
    let Some(c) = ctx() else { return BatchGraphOutcome::Declined };
    if hyb::wants(layers, nh, nkv, hd) {
        let defer = BATCH_DEFER.with(|d| d.get());
        return hyb::batch(
            c, model, kv_id, layer_base, layers, invf, h, nh, nkv, hd, rd, hidden, positions, cap, gemma, eps,
            attn_scale, k, o1, o1_epoch, spec, layers_run, defer,
        );
    }
    let b = k;
    if spec.is_some() || o1.iter().any(|v| v.is_some()) {
        decline("batch", "a speculative verify tail or a sealed far field");
        return BatchGraphOutcome::Declined;
    }
    if b == 0
        || positions.len() != b
        || h.len() != b * hidden
        || positions.windows(2).any(|w| w[1] != w[0] + 1)
        || positions[b - 1] >= cap
    {
        decline("batch", "rows not contiguous or past the context cap");
        return BatchGraphOutcome::Declined;
    }
    let pos0 = positions[0];
    let defer = BATCH_DEFER.with(|d| d.get());
    let t_all = std::time::Instant::now();
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() {
        return BatchGraphOutcome::Declined;
    }
    let g = CallGeom { nh, nkv, hd, rd, hidden, gemma, scale: attn_scale };
    let plan = match plan_for(c, &mut st, model, layers, layer_base, None, &[], &g, invf, pos0) {
        Ok(p) => p,
        Err(e) => {
            decline("batch", &e);
            return BatchGraphOutcome::Declined;
        }
    };
    let hint = st.kv_hint;
    for (li, (pl, l)) in plan.layers.iter().zip(layers).enumerate() {
        let GraphAttn::Full { cpu_k, cpu_v, cpu_base, .. } = &l.attn else { unreachable!() };
        let row = pl.nkv * pl.hd;
        let m = st.kv.entry((kv_id, layer_base + li)).or_insert_with(|| KvMirror::new(row, pl.window));
        if m.row != row || m.window != pl.window {
            // another geometry under the same id: its rows mean nothing here
            let _ = c.sync();
            c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
            *m = KvMirror::new(row, pl.window);
        }
        match m.sync_to(c, cpu_k, cpu_v, *cpu_base, pl.nkv, pl.hd, pos0).and_then(|ok| {
            if ok {
                m.reserve(c, b, hint)?;
            }
            Ok(ok)
        }) {
            Ok(true) => {}
            Ok(false) => {
                decline("batch", "a layer's K/V rows are neither on the card nor on the host");
                return BatchGraphOutcome::Declined;
            }
            Err(e) => {
                decline("batch", &e);
                return BatchGraphOutcome::Declined;
            }
        }
    }
    let mut run = || -> Res<()> {
        let f = &c.f;
        let (gw, ga) = prec();
        let half = gw == Gemm::Half;
        let half_i = half as i32;
        let (qw, kvw, hpk) = (plan.qmax, plan.kvmax, plan.hpk);
        let mut t0 = std::time::Instant::now();
        // Row blocks. `bxn` (a layer's normed rows) and `bao` (its attention
        // output) are the operands of weight GEMMs: f16 on the f16 path, f32
        // otherwise, written in that type by the kernel that makes them.
        let op_bytes = if half { 2 } else { 4 };
        let bh = st.scr(c, "bh", b * hidden * 4)?;
        let bxn = st.scr(c, "bxn", b * hidden * op_bytes)?;
        let bq = st.scr(c, "bq", b * qw * 4)?;
        let bqg = st.scr(c, "bqg", b * qw * 4)?;
        let bk = st.scr(c, "bk", b * kvw * 4)?;
        let bv = st.scr(c, "bv", b * kvw * 4)?;
        let bg16 = st.scr(c, "bg16", b * nh * 4)?;
        let baog = st.scr(c, "baog", b * qw * 4)?;
        let bao = st.scr(c, "bao", b * qw * op_bytes)?;
        let bgg = st.scr(c, "bgg", b * plan.intermax * 4)?;
        let buu = st.scr(c, "buu", b * plan.intermax * 4)?;
        let bmh = if half { st.scr(c, "bmh", b * plan.intermax * 2)? } else { 0 };
        let bis = st.scr(c, "bis", b * nh * 4)?;
        let wmax = plan
            .layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down])
            .filter(|m| m.kind != 4)
            .map(|m| m.rows * m.cols)
            .max()
            .unwrap_or(0);
        let wd = st.scr(c, "wd", wmax * op_bytes)?;
        let tab = rope_tables(&plan, positions.iter().copied(), b);
        let btab = st.scr(c, "btab", tab.len().max(1) * 4)?;
        unsafe {
            cu::memcpy_htod_async(bh, h, c.stream).map_err(derr)?;
            if !tab.is_empty() {
                cu::memcpy_htod_async(btab, &tab, c.stream).map_err(derr)?;
            }
        }
        c.stage(8, &mut t0)?;
        let (hidden_i, eps_f, gemma_i) = (hidden as i32, eps, gemma as i32);
        // y [b][rows] = alpha * x [b][cols] * W^T + beta * y: the weight
        // expanded to a dense plane of the operand type, then one GEMM.
        let mul = |m: &DevMat, wh: Option<&Buf>, x: u64, y: u64, alpha: f32, beta: f32, t0: &mut std::time::Instant| -> Res<()> {
            let (rows_i, cols_i) = (m.rows as i32, m.cols as i32);
            let quads = (m.rows * m.cols / 4).div_ceil(256) as u32;
            let expand = match (m.kind, half) {
                (6, false) => Some(f.expand_q4tp),
                (6, true) => Some(f.expand_q4tp_h),
                (7, false) => Some(f.expand_q8),
                (7, true) => Some(f.expand_q8_h),
                _ => None,
            };
            let wp = match (expand, wh) {
                (Some(e), _) => {
                    unsafe { c.launch(e, (quads, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i))? };
                    wd
                }
                // an f32 weight: as it is, or its resident f16 copy
                (None, Some(hb)) if half => hb.ptr,
                (None, _) if half => return Err("f32 weight without an f16 copy".into()),
                (None, _) => m.w.ptr,
            };
            c.stage(1, t0)?;
            unsafe { c.gemm(true, false, m.rows, b, m.cols, alpha, wp, m.cols, x, m.cols, beta, y, m.rows, gw)? };
            c.stage(2, t0)
        };
        for (li, pl) in plan.layers.iter().enumerate() {
            let (nh_i, nkv_i, hd_i, rhalf_i, b_i) = (nh as i32, pl.nkv as i32, pl.hd as i32, pl.half as i32, b as i32);
            unsafe {
                c.launch(
                    f.rmsnorm_rows,
                    (b as u32, 1),
                    256,
                    0,
                    kargs!(bh, pl.in_norm.ptr, bxn, hidden_i, eps_f, gemma_i, half_i),
                )?;
            }
            c.stage(0, &mut t0)?;
            mul(&pl.q, None, bxn, bq, 1.0, 0.0, &mut t0)?;
            mul(&pl.k, None, bxn, bk, 1.0, 0.0, &mut t0)?;
            mul(&pl.v, None, bxn, bv, 1.0, 0.0, &mut t0)?;
            if let Some(hg) = &pl.hg {
                mul(hg, pl.hg_h.as_ref(), bxn, bg16, 1.0, 0.0, &mut t0)?;
            }
            let m = st.kv.get_mut(&(kv_id, layer_base + li)).unwrap();
            let slot0 = (m.stored - m.base) as i32;
            let tabp = btab + (pl.rope_off * b * 4) as u64;
            let (kp, vp) = (m.kptr(), m.vptr());
            unsafe {
                c.launch(
                    f.rope_kv_rows,
                    (b as u32, 1),
                    256,
                    0,
                    kargs!(bq, bqg, bk, bv, kp, vp, tabp, nh_i, nkv_i, hd_i, rhalf_i, pl.rope_scale, slot0, b_i),
                )?;
            }
            // A far-field layer of a prompt read for its seal: the rotated
            // queries the trace keeps wait in a buffer for `o1_qcap_drain`.
            let cap_stride = QCAP.with(|q| {
                q.borrow().as_ref().filter(|q| q.layers.contains(&(layer_base + li))).map(|q| q.stride.max(1))
            });
            if let Some(stride) = cap_stride {
                let first = (stride - pos0 % stride) % stride;
                let kept = if first < b { (b - first).div_ceil(stride) } else { 0 };
                if kept > 0 {
                    let floats = kept * nh * pl.hd;
                    let buf = Buf::new(floats * 4)?;
                    let (first_i, stride_i) = (first as i32, stride as i32);
                    unsafe {
                        c.launch(
                            f.q_rows_strided,
                            (kept as u32, 1),
                            256,
                            0,
                            kargs!(bqg, buf.ptr, nh_i, nkv_i, hd_i, b_i, first_i, stride_i),
                        )?;
                    }
                    QCAP.with(|q| {
                        if let Some(q) = q.borrow_mut().as_mut() {
                            q.bufs.push((layer_base + li, floats, buf));
                        }
                    });
                }
            }
            m.stored += b;
            c.stage(3, &mut t0)?;
            let (m_base, lda) = (m.base, pl.nkv * pl.hd);
            // A KV group's queries of the chunk are one GEMM's columns
            // ([group][row][head]), and the groups one batched call. The
            // score matrix is cut into runs of rows that keep it near 384 MB.
            let grow = hpk * pl.hd; // floats of one row's queries in a group
            let mut b0 = 0usize;
            while b0 < b {
                let p0 = pos0 + b0;
                let col_base = if pl.window > 0 { (p0 + 1).saturating_sub(pl.window) } else { 0 }.max(m_base);
                let fit = |n: usize| (p0 + n - col_base) * n * nh * 4 <= 384 << 20;
                let mut bn = b - b0;
                while bn > 64 && !fit(bn) {
                    bn = bn.div_ceil(2);
                }
                let l_rows = p0 + bn - col_base;
                let cols = bn * hpk;
                let koff = ((col_base - m_base) * lda * 4) as u64;
                // sized once for the prompt's longest run, not regrown a chunk
                // (a regrowth is a wait and two driver calls)
                let whole = (hint.min(cap).max(pos0 + b) * b * nh * 4).min(400 << 20);
                let bs = st.scr(c, "bs", (pl.nkv * l_rows * cols * 4).max(whole))?;
                let (l_i, p0_i, cb_i, win_i, hpk_i) = (l_rows as i32, p0 as i32, col_base as i32, pl.window as i32, hpk as i32);
                let (cols_i, istride) = (cols as i32, (b * hpk) as i32);
                let goff = (b0 * grow * 4) as u64;
                let isum = bis + (b0 * hpk * 4) as u64;
                unsafe {
                    // S [group][rows][L] = scale * q . k
                    c.gemm_batched(
                        true, l_rows, cols, pl.hd, g.scale, kp + koff, lda, pl.hd, bqg + goff, pl.hd, b * grow, bs,
                        l_rows, l_rows * cols, pl.nkv, ga,
                    )?;
                    c.stage(4, &mut t0)?;
                    c.launch(
                        f.softmax_rows,
                        ((pl.nkv * cols) as u32, 1),
                        256,
                        0,
                        kargs!(bs, isum, l_i, p0_i, cb_i, win_i, hpk_i, cols_i, istride),
                    )?;
                    c.stage(5, &mut t0)?;
                    // aog [group][rows][hd] = exp(S) . v
                    c.gemm_batched(
                        false, pl.hd, cols, l_rows, 1.0, vp + koff, lda, pl.hd, bs, l_rows, l_rows * cols, baog + goff,
                        pl.hd, b * grow, pl.nkv, ga,
                    )?;
                    c.stage(6, &mut t0)?;
                }
                b0 += bn;
            }
            let n_ao = (b * nh * pl.hd) as u64;
            let gate_p = if pl.hg.is_some() { bg16 } else { 0u64 };
            unsafe {
                c.launch(
                    f.attn_out_rows,
                    (n_ao.div_ceil(256) as u32, 1),
                    256,
                    0,
                    kargs!(baog, bis, gate_p, bao, nh_i, nkv_i, hd_i, b_i, half_i),
                )?;
            }
            c.stage(7, &mut t0)?;
            // h += ao * Wo^T
            mul(&pl.o, None, bao, bh, 1.0, 1.0, &mut t0)?;
            unsafe {
                c.launch(
                    f.rmsnorm_rows,
                    (b as u32, 1),
                    256,
                    0,
                    kargs!(bh, pl.post_norm.ptr, bxn, hidden_i, eps_f, gemma_i, half_i),
                )?;
            }
            c.stage(0, &mut t0)?;
            mul(&pl.gate, None, bxn, bgg, 1.0, 0.0, &mut t0)?;
            mul(&pl.up, None, bxn, buu, 1.0, 0.0, &mut t0)?;
            // h += (act(gate) * up) * Wd^T
            if half {
                // the middle goes to f16 a sixteenth its size (`act_half`),
                // the GEMM's alpha puts the sixteen back
                let quads = (b * pl.inter / 4) as u64;
                let inv = 1.0f32 / 16.0;
                unsafe {
                    c.launch(f.act_half, (quads.div_ceil(256) as u32, 1), 256, 0, kargs!(bgg, buu, bmh, pl.act, inv, quads))?;
                }
                c.stage(7, &mut t0)?;
                mul(&pl.down, None, bmh, bh, 16.0, 1.0, &mut t0)?;
            } else {
                let n = (b * pl.inter) as u64;
                unsafe {
                    c.launch(f.act_mul, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(bgg, buu, pl.act, n))?;
                }
                c.stage(7, &mut t0)?;
                mul(&pl.down, None, bgg, bh, 1.0, 1.0, &mut t0)?;
            }
        }
        if !defer {
            unsafe { cu::memcpy_dtoh_async(h, bh, c.stream).map_err(derr)? };
            c.sync()?;
            c.stage(9, &mut t0)?;
        }
        BATCH_ROWS.fetch_add(b as u64, Ordering::Relaxed);
        BATCH_NS.fetch_add(t_all.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    };
    match run() {
        Ok(()) => {
            if let Some(lr) = layers_run {
                *lr = layers.len();
            }
            BatchGraphOutcome::Completed
        }
        Err(e) => {
            tracing::error!("cuda batch graph failed: {e}");
            let _ = c.sync();
            BatchGraphOutcome::Failed
        }
    }
}

/// Drop a sequence's K/V mirrors.
pub fn kv_reset(kv_id: u64) {
    let Some(c) = ctx() else { return };
    let mut st = c.st.lock().unwrap();
    if c.current().is_err() || c.sync().is_err() {
        return;
    }
    hyb::reset(c, &mut st, kv_id);
    st.hyb.half.retain(|k| k.0 != kv_id);
    st.kv.retain(|&(id, _), m| {
        if id == kv_id {
            c.kv_bytes.fetch_sub((m.cap * m.row * 8) as u64, Ordering::Relaxed);
        }
        id != kv_id
    });
    st.o1.retain(|&(id, _), _| id != kv_id);
}

/// Sealed far-field layers of one sequence on the card, and their bytes.
pub fn o1_device_stats(kv_id: u64) -> (usize, u64) {
    let Some(c) = ctx() else { return (0, 0) };
    let st = c.st.lock().unwrap();
    st.o1.iter().filter(|(k, _)| k.0 == kv_id).fold((0, 0), |(n, b), (_, d)| (n + 1, b + d.bytes))
}

/// Keep the rotated queries of `layers` in the batch graph calls this thread
/// makes until `o1_qcap_take` (the seal's trace of a prompt read on the card).
pub fn o1_qcap_begin(layers: &[usize]) -> bool {
    if ctx().is_none() {
        return false;
    }
    let _ = o1_qcap_take();
    QCAP.with(|q| *q.borrow_mut() = Some(QCap { layers: layers.to_vec(), stride: 1, bufs: Vec::new() }));
    true
}

/// The armed capture keeps one row in `stride` by position (the rows a
/// seal's trace reads) instead of every row. False: nothing is armed.
pub fn o1_qcap_stride(stride: usize) -> bool {
    QCAP.with(|q| match q.borrow_mut().as_mut() {
        Some(q) => {
            q.stride = stride.max(1);
            true
        }
        None => false,
    })
}

/// What the capture holds so far, position-major `[rows][nh][hd]` a layer;
/// it stays armed.
pub fn o1_qcap_drain() -> Option<Vec<(usize, Vec<f32>)>> {
    let (layers, bufs) = QCAP.with(|q| {
        let mut q = q.borrow_mut();
        let q = q.as_mut()?;
        Some((q.layers.clone(), std::mem::take(&mut q.bufs)))
    })?;
    qcap_read(layers, bufs)
}

/// End the capture and read what is left.
pub fn o1_qcap_take() -> Option<Vec<(usize, Vec<f32>)>> {
    let q = QCAP.with(|q| q.borrow_mut().take())?;
    qcap_read(q.layers, q.bufs)
}

fn qcap_read(layers: Vec<usize>, bufs: Vec<(usize, usize, Buf)>) -> Option<Vec<(usize, Vec<f32>)>> {
    let c = ctx()?;
    // the buffers are freed under the lock, with the context current
    let _st = c.st.lock().unwrap();
    c.current().ok()?;
    let mut out: Vec<(usize, Vec<f32>)> = layers.iter().map(|&l| (l, Vec::new())).collect();
    let mut ok = c.sync().is_ok();
    for (layer, floats, buf) in &bufs {
        let Some((_, dst)) = out.iter_mut().find(|(l, _)| l == layer) else { continue };
        if !ok {
            break;
        }
        let at = dst.len();
        dst.resize(at + floats, 0.0);
        // waited for before `dst` may grow again
        ok = unsafe { cu::memcpy_dtoh_async(&mut dst[at..], buf.ptr, c.stream) }.is_ok() && c.sync().is_ok();
    }
    let _ = c.sync();
    drop(bufs);
    ok.then_some(out)
}

/// A sealed chunk's rows into the armed archive capture (nothing when none is):
/// `n` K/V rows of `row` values from row `n0` of the far field's mirror, and
/// the chunk's query rows at positions that are multiples of the stride.
#[allow(clippy::too_many_arguments)]
fn acap_note(c: &Ctx, layer: usize, kp: u64, vp: u64, n0: usize, n: usize, row: usize, qg: u64, nh: usize, nkv: usize, hd: usize, pos0: usize) -> Res<()> {
    let Some(stride) = ACAP.with(|a| a.borrow().as_ref().map(|a| a.stride.max(1))) else { return Ok(()) };
    let bytes = n * row * 2;
    let (k, v) = (Buf::new(bytes)?, Buf::new(bytes)?);
    let at = (n0 * row * 2) as u64;
    unsafe {
        cu::memcpy_dtod_async(k.ptr, kp + at, bytes, c.stream).map_err(derr)?;
        cu::memcpy_dtod_async(v.ptr, vp + at, bytes, c.stream).map_err(derr)?;
    }
    let first = (stride - pos0 % stride) % stride;
    let kept = if first < n { (n - first).div_ceil(stride) } else { 0 };
    let q = if kept > 0 {
        let buf = Buf::new(kept * nh * hd * 4)?;
        let (nh_i, nkv_i, hd_i, b_i, first_i, stride_i) = (nh as i32, nkv as i32, hd as i32, n as i32, first as i32, stride as i32);
        unsafe {
            c.launch(c.f.q_rows_strided, (kept as u32, 1), 256, 0, kargs!(qg, buf.ptr, nh_i, nkv_i, hd_i, b_i, first_i, stride_i))?;
        }
        Some(buf)
    } else {
        None
    };
    ACAP.with(|a| {
        if let Some(a) = a.borrow_mut().as_mut() {
            a.bufs.push((layer, n, k, v, kept * nh * hd, q));
        }
    });
    Ok(())
}

/// Rows a layer's mirror holds (absolute), None without a mirror.
/// y [n][rows] = W x [n][cols] on the card (a q4tp graph weight of the
/// hybrid arm): false when the backend cannot take it.
pub fn rows_matvec(model: &Arc<CmfModel>, gw: &GraphW, rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32]) -> bool {
    match hyb::rows_matvec(model, gw, rows, cols, x, n, y) {
        Ok(()) => true,
        Err(e) => {
            decline("rows matvec", &e);
            false
        }
    }
}

thread_local! {
    /// A DFlash drafter's taps (`layer_taps_arm`): the layers asked for, and
    /// what the next graph call of this thread kept, `[slot][row][hidden]`.
    static LAYER_TAPS: std::cell::RefCell<(Vec<usize>, Option<Vec<f32>>)> = const { std::cell::RefCell::new((Vec::new(), None)) };
}

/// Ask the next graph call of this thread to keep the residual stream after
/// each of `layers` (absolute indices); an empty list disarms.
pub fn layer_taps_arm(layers: &[usize]) {
    LAYER_TAPS.with(|t| *t.borrow_mut() = (layers.to_vec(), None));
}

/// What the armed call kept; disarms. False when it kept nothing.
pub fn layer_taps_take(out: &mut Vec<f32>) -> bool {
    let (_, got) = LAYER_TAPS.with(|t| std::mem::take(&mut *t.borrow_mut()));
    match got {
        Some(v) => {
            *out = v;
            true
        }
        None => false,
    }
}

fn taps_wanted() -> Vec<usize> {
    LAYER_TAPS.with(|t| t.borrow().0.clone())
}

fn taps_put(v: Vec<f32>) {
    LAYER_TAPS.with(|t| t.borrow_mut().1 = Some(v));
}

thread_local! {
    /// A greedy verify's row ids from the card (`verify_argmax_arm`):
    /// None unarmed, Some(empty) armed, Some(ids) after the verify.
    static VERIFY_IDS: std::cell::RefCell<Option<Vec<u32>>> = const { std::cell::RefCell::new(None) };
}

/// Arm (or disarm) the next verify window of this thread to pick each row's
/// argmax on the card and hand back the ids instead of the logits.
pub fn verify_argmax_arm(on: bool) {
    VERIFY_IDS.with(|v| *v.borrow_mut() = if on { Some(Vec::new()) } else { None });
}

fn verify_argmax_armed() -> bool {
    VERIFY_IDS.with(|v| v.borrow().is_some())
}

fn verify_argmax_put(ids: Vec<u32>) {
    VERIFY_IDS.with(|v| *v.borrow_mut() = Some(ids));
}

/// The ids the armed verify picked (None when it picked none); disarms.
pub fn verify_argmax_take() -> Option<Vec<u32>> {
    VERIFY_IDS.with(|v| v.borrow_mut().take()).filter(|ids| !ids.is_empty())
}

pub use hyb::dflash::{Dev as DflashDev, LayerIn as DflashLayerIn, SETTLE_MAX as DFLASH_SETTLE_MAX, SpecIn as DflashSpecIn};

/// A verify window over the far fields of `kv_id` kept `rows` rows: they
/// move into the window (`hyb::spec_commit`).
pub fn o1_spec_commit(kv_id: u64, pos0: usize, rows: usize) -> bool {
    hyb::spec_commit(kv_id, pos0, rows)
}

/// Drafts ride a far field on this backend.
pub fn o1_spec_on() -> bool {
    hyb::far_spec_on()
}

/// Dory's forward on the card (`hyb::dory`); `false` leaves the host to run it.
pub fn dory_forward(id: u64, m: &crate::dory::Dory, ids: &[u32], pos: usize, loops: usize, want: crate::dory::Want, logits: &mut Vec<f32>) -> bool {
    selected() && hyb::dory::forward(id, m, ids, pos, loops, want, logits)
}

/// A new Dory sequence on the card.
pub fn dory_reset(id: u64) {
    hyb::dory::reset(id);
}

/// Has this Dory sequence gone over to the far field on the card?
pub fn dory_sealed(id: u64) -> bool {
    hyb::dory::sealed(id)
}

/// The K/V bytes a Dory sequence holds on the card.
pub fn dory_kv_bytes(id: u64) -> Option<u64> {
    hyb::dory::kv_bytes(id)
}

/// The hybrid arm's kernels are built (a trunk plan ran): what the DFlash
/// block borrows.
pub fn hybrid_kernels_ready() -> bool {
    ctx().is_some_and(|c| hyb::kernels_ready(&c.st.lock().unwrap()))
}

/// The hybrid arm's MTP draft chain: `k` greedy drafts in one graph
/// (`hyb::mtp_chain`). Err(false) declined, Err(true) failed after admission.
#[allow(clippy::too_many_arguments)]
pub fn mtp_chain(
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
    let Some(c) = ctx() else { return Err(false) };
    hyb::mtp_chain(
        c, model, kv_id, layer, invf, h0, e0, t0, k, nh, nkv, hd, attn_scale, rd, hidden, position, cap, gemma, eps, lm_head,
        final_norm, eh, enorm, hnorm, embed, layer_base,
    )
}

/// After a partial speculative acceptance: the sequence's GDN states back to
/// the verify window's start, the accepted rows `..= slot` replayed.
pub fn gdn_spec_restore(kv_id: u64, slot: usize, base_pos: usize, expected_layers: usize) -> bool {
    hyb::spec_restore(kv_id, slot, base_pos, expected_layers)
}

/// Snapshot every GDN state of the sequence at prompt position `pos`.
pub fn gdn_snapshot(kv_id: u64, pos: usize) -> bool {
    hyb::gdn_snapshot(kv_id, pos)
}

/// Every GDN state of the sequence back to its snapshot at `pos`.
pub fn gdn_snapshot_restore(kv_id: u64, pos: usize) -> bool {
    hyb::gdn_snapshot_restore(kv_id, pos)
}

/// A recurrent state of this sequence's layer lives on the card.
pub fn state_resident(kv_id: u64, layer: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let st = c.st.lock().unwrap();
    hyb::state_resident(&st, kv_id, layer)
}

/// A GDN layer's state back to the host (`[ring | S]`).
pub fn gdn_state_read(kv_id: u64, layer: usize) -> Option<Vec<f32>> {
    let c = ctx()?;
    let st = c.st.lock().unwrap();
    c.current().ok()?;
    hyb::gdn_state_read(c, &st, kv_id, layer)
}

pub fn kv_stored(kv_id: u64, layer: usize) -> Option<usize> {
    let c = ctx()?;
    c.st.lock().unwrap().kv.get(&(kv_id, layer)).map(|m| m.stored)
}

/// Move a mirror's cursor back to `stored` rows.
pub fn kv_set_stored(kv_id: u64, layer: usize, stored: usize) -> bool {
    let Some(c) = ctx() else { return false };
    match c.st.lock().unwrap().kv.get_mut(&(kv_id, layer)) {
        Some(m) if stored <= m.stored && stored >= m.base => {
            m.stored = stored;
            true
        }
        // fewer rows than asked (an MTP mirror behind plain tokens): nothing
        // to step back from, and the next call zero-fills up to its row
        Some(m) if stored > m.stored => true,
        _ => false,
    }
}

/// Rows `[from, to)` of one mirror in the host layout; the third value is
/// the first position actually read (rows a sliding layer let go are zeros).
pub fn kv_pull_host(
    kv_id: u64,
    layer: usize,
    from: usize,
    to: usize,
    nkv: usize,
    hd: usize,
) -> Option<(Vec<f32>, Vec<f32>, usize)> {
    let c = ctx()?;
    let st = c.st.lock().unwrap();
    c.current().ok()?;
    let m = st.kv.get(&(kv_id, layer))?;
    let row = nkv * hd;
    // the hybrid arm keeps f16 rows (its mirror's `row` counts f32 slots)
    let half = st.hyb.half.contains(&(kv_id, layer));
    if m.row != if half { row / 2 } else { row } || to > m.stored || from > to {
        return None;
    }
    let first = from.max(m.base);
    let mut kb = vec![0f32; (to - from) * row];
    let mut vb = vec![0f32; (to - from) * row];
    if first < to {
        let dst = (first - from) * row;
        if half {
            let off = ((first - m.base) * row * 2) as u64;
            let mut kh = vec![0u16; (to - first) * row];
            let mut vh = vec![0u16; (to - first) * row];
            unsafe {
                cu::memcpy_dtoh_async(&mut kh, m.kptr() + off, c.stream).ok()?;
                cu::memcpy_dtoh_async(&mut vh, m.vptr() + off, c.stream).ok()?;
            }
            c.sync().ok()?;
            // a long prompt's mirror is gigabytes: its rows to f32 on threads
            let widen = |dst: &mut [f32], src: &[u16]| {
                let per = src.len().div_ceil(16).max(1 << 16);
                std::thread::scope(|sc| {
                    for (o, i) in dst.chunks_mut(per).zip(src.chunks(per)) {
                        sc.spawn(move || {
                            for (d, h) in o.iter_mut().zip(i) {
                                *d = cortiq_core::quant::f16_to_f32(*h);
                            }
                        });
                    }
                });
            };
            widen(&mut kb[dst..], &kh);
            widen(&mut vb[dst..], &vh);
        } else {
            let off = ((first - m.base) * row * 4) as u64;
            unsafe {
                cu::memcpy_dtoh_async(&mut kb[dst..], m.kptr() + off, c.stream).ok()?;
                cu::memcpy_dtoh_async(&mut vb[dst..], m.vptr() + off, c.stream).ok()?;
            }
            c.sync().ok()?;
        }
    }
    Some((kb, vb, first.min(to)))
}

/// Rows `[from, to)` of several mirrors, position-major; None unless every
/// one of them is still on the card.
pub fn kv_read_rows(
    kv_id: u64,
    reqs: &[(usize, usize, usize)],
    nkv: usize,
    hd: usize,
) -> Option<Vec<(Vec<f32>, Vec<f32>)>> {
    let mut out = Vec::with_capacity(reqs.len());
    for &(layer, from, to) in reqs {
        let (k, v, first) = kv_pull_host(kv_id, layer, from, to, nkv, hd)?;
        if first > from {
            return None;
        }
        out.push((k, v));
    }
    Some(out)
}

/// The backend came up (its context exists).
pub fn ctx_ready() -> bool {
    ctx().is_some()
}

/// A seal's far sums on the card (`gpu::o1_far_sums`).
#[allow(clippy::too_many_arguments)]
pub fn o1_far_sums(nkv: usize, hpk: usize, hd: usize, m: usize, scale: f32, qt: &[f32], idx: &[u32], ks: &[&[f32]], vs: &[&[f32]]) -> Option<(Vec<f32>, Vec<f32>)> {
    let c = ctx()?;
    match hyb::far_sums(c, nkv, hpk, hd, m, scale, qt, idx, ks, vs, hd) {
        Ok(r) => Some(r),
        Err(e) => {
            decline("far sums", &e);
            None
        }
    }
}

/// What the card may hold: its memory less a reserve for the K/V mirrors,
/// the prompt's scratch and cuBLAS (`CMF_GPU_VRAM_MB` caps it by hand).
pub fn vram_budget() -> u64 {
    let Some(c) = ctx() else { return 0 };
    if let Some(mb) = std::env::var("CMF_GPU_VRAM_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        return mb << 20;
    }
    c.total_mem.saturating_sub(2 << 30)
}

/// Bytes of resident weights (their size in the file).
pub fn resident_bytes() -> u64 {
    ctx().map_or(0, |c| c.resident.load(Ordering::Relaxed))
}

/// Bytes of K/V mirrors allocated right now.
pub fn kv_bytes() -> u64 {
    ctx().map_or(0, |c| c.kv_bytes.load(Ordering::Relaxed))
}

/// The card's name, for reports.
pub fn device_name() -> Option<String> {
    ctx().map(|c| c.name.clone())
}

/// `CMF_CUDA_PROF=1`: where the time went, on stderr.
pub fn perf_report() {
    if !prof_on() && !debug() {
        return;
    }
    hyb::report();
    let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
    let toks = TOKENS.load(Ordering::Relaxed).saturating_sub(16);
    if toks > 0 {
        eprintln!(
            "cuda decode: {toks} steady tokens, {:.2} ms a token ({:.2} ms of it host encode)",
            ms(&TOKEN_NS) / toks as f64,
            ms(&TOKEN_HOST_NS) / toks as f64
        );
        let us = TOKEN_US.lock().unwrap();
        if us.len() > 16 {
            let mut sorted: Vec<u32> = us[16..].to_vec();
            sorted.sort_unstable();
            let slow: Vec<String> = us
                .iter()
                .enumerate()
                .skip(16)
                .filter(|&(_, &t)| t > sorted[sorted.len() / 2] * 3 / 2)
                .map(|(i, t)| format!("#{i} {:.1} ms", *t as f64 / 1e3))
                .collect();
            eprintln!(
                "cuda decode tokens: median {:.2} ms, 90th {:.2} ms, slowest {:.2} ms; past 1.5x the median: {}",
                sorted[sorted.len() / 2] as f64 / 1e3,
                sorted[sorted.len() * 9 / 10] as f64 / 1e3,
                sorted[sorted.len() - 1] as f64 / 1e3,
                if slow.is_empty() { "none".to_string() } else { slow.join(", ") }
            );
        }
        drop(us);
        if prof_on() {
            for (name, ns) in DSTAGES.iter().zip(&DSTAGE_NS) {
                eprintln!("  {name:>13}: {:8.3} ms a token", ms(ns) / (toks + 16) as f64);
            }
        }
    }
    let rows = BATCH_ROWS.load(Ordering::Relaxed);
    if rows > 0 {
        eprintln!(
            "cuda prompt: {rows} rows in {:.1} ms ({:.0} rows/s)",
            ms(&BATCH_NS),
            rows as f64 / (ms(&BATCH_NS) / 1e3)
        );
        if prof_on() {
            for (name, ns) in STAGES.iter().zip(&STAGE_NS) {
                eprintln!("  {name:>13}: {:9.1} ms", ms(ns));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::quant;

    /// The job record is read by the kernels as `struct MvJob`: five
    /// pointers and four ints, 56 bytes.
    #[test]
    fn job_record_layout() {
        assert_eq!(std::mem::size_of::<MvJob>(), 56);
        assert_eq!(std::mem::align_of::<MvJob>(), 8);
    }

    #[test]
    fn rows_per_block_spreads_a_matrix() {
        // Spark-X2.6-4B's shapes: about 256 blocks each, eight rows at least.
        assert_eq!(rows_per_block(10240), 40);
        assert_eq!(rows_per_block(4096), 16);
        assert_eq!(rows_per_block(2560), 16);
        assert_eq!(rows_per_block(1024), 8);
        assert_eq!(rows_per_block(16), 8);
        assert_eq!(rows_per_block(131072), 64);
    }

    #[test]
    fn staged_row_fits_a_block() {
        // the widest Spark row (the FFN middle) under the default 48 KB
        assert!(mv_shared_bytes(10240) <= 48 << 10);
        assert!(mv_shared_bytes(2560) < mv_shared_bytes(4096));
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn rnd(seed: &mut u64) -> f32 {
        (lcg(seed) % 2001) as f32 / 1000.0 - 1.0
    }

    /// A random q4tp payload: nibbles, row ladders (lo, step) and 5-bit codes.
    fn q4tp_payload(rows: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
        let gpr = cols / quant::GROUP_SIZE;
        let (params_off, codes_off, stride) = quant::q4tp_sections(rows, cols);
        let mut bytes = vec![0u8; codes_off + rows * stride];
        for b in &mut bytes[..params_off] {
            *b = lcg(seed) as u8;
        }
        for r in 0..rows {
            let lo = quant::f32_to_f16(-9.0 + (lcg(seed) % 400) as f32 / 100.0).to_le_bytes();
            let st = quant::f32_to_f16(0.05 + (lcg(seed) % 20) as f32 / 100.0).to_le_bytes();
            bytes[params_off + r * 4..params_off + r * 4 + 2].copy_from_slice(&lo);
            bytes[params_off + r * 4 + 2..params_off + r * 4 + 4].copy_from_slice(&st);
            let cr = &mut bytes[codes_off + r * stride..codes_off + (r + 1) * stride];
            for g in 0..gpr {
                quant::q4tp_put_code(cr, g, (lcg(seed) % 32) as usize);
            }
        }
        bytes
    }

    /// y = W x on the card through `mv_multi` (with the RMS norm folded in
    /// when `nw` is given), read back.
    fn device_matvec(c: &Ctx, m: &DevMat, x: &[f32], nw: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let xb = Buf::from_slice(x).unwrap();
        let yb = Buf::new(m.rows * 4).unwrap();
        let nb = nw.map(|w| Buf::from_slice(w).unwrap());
        let (mut jobs, mut blk) = (Vec::new(), 0u32);
        push_job(&mut jobs, &mut blk, m, yb.ptr, 0);
        let jb = Buf::from_slice(&jobs).unwrap();
        let (nullp, no_act, cols_i) = (0u64, -1i32, m.cols as i32);
        let nwp = nb.as_ref().map_or(0u64, |b| b.ptr);
        let mut y = vec![0f32; m.rows];
        unsafe {
            c.launch(
                c.f.mv_multi,
                (blk, 1),
                256,
                mv_shared_bytes(m.cols),
                kargs!(jb.ptr, 1i32, xb.ptr, nullp, no_act, nwp, eps, 0i32, cols_i),
            )
            .unwrap();
            cu::memcpy_dtoh_async(&mut y, yb.ptr, c.stream).unwrap();
        }
        c.sync().unwrap();
        y
    }

    fn host_matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        (0..rows).map(|r| (0..cols).map(|i| w[r * cols + i] as f64 * x[i] as f64).sum()).collect()
    }

    fn worst_rel(dev: &[f32], host: &[f64]) -> f64 {
        let scale = host.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
        dev.iter().zip(host).fold(0f64, |m, (d, h)| m.max((*d as f64 - h).abs() / scale))
    }

    /// The decode matvec and the prompt's weight planes against the host's
    /// dequantization, on whatever card is here (skipped without one).
    /// The far-field token kernels against `nystrom.rs`: sealed groups'
    /// states go up, the kernels move them row by row past a whole turn of
    /// the ring, and each row's attention is compared with `step_group`'s.
    #[test]
    fn far_field_matches_the_host_step() {
        let Some(c) = ctx() else {
            eprintln!("cuda: no card, the far-field parity test is skipped");
            return;
        };
        let (nkv, hpk, hd, m, w, sink, t) = (2usize, 4usize, 64usize, 16usize, 96usize, 6usize, 400usize);
        let nh = nkv * hpk;
        let mut seed = 0x51ed_c0de_u64;
        let mut rows = |n: usize, amp: f32| -> Vec<f32> { (0..n).map(|_| rnd(&mut seed) * amp).collect() };
        let mut states = Vec::new();
        for _ in 0..nkv {
            let (ks, vs) = (rows(t * hd, 2.0), rows(t * hd, 1.0));
            let qs: Vec<Vec<f32>> = (0..hpk).map(|_| rows(t * hd, 2.0)).collect();
            let qr: Vec<&[f32]> = qs.iter().map(|q| &q[..]).collect();
            let mut s = crate::nystrom::NystromState::new_group(m, w, sink, hpk);
            s.prefill_group(&qr, &ks, &vs, t, hd, hd);
            states.push(s);
        }
        let mut st = c.st.lock().unwrap();
        c.current().unwrap();
        let af = st.attn_funcs(c, hpk, hd).unwrap();
        let key = (u64::MAX - 7, 0usize);
        {
            let views: Vec<_> = states.iter().map(|s| s.device_view()).collect();
            assert!(!views[0].exact_only && views[0].far_len > 0);
            o1_ensure(c, &mut st, key.0, key.1, &views, 1, hpk, nkv, hd).unwrap();
        }
        let (qb, ao) = (Buf::new(nh * hd * 4).unwrap(), Buf::new(nh * hd * 4).unwrap());
        let (mut worst, mut top) = (0f32, 0f32);
        for step in 0..3 * w / 2 {
            let (q, k, v) = (rows(nh * hd, 2.0), rows(nkv * hd, 2.0), rows(nkv * hd, 1.0));
            let mut host = vec![0f32; nh * hd];
            for (g, s) in states.iter_mut().enumerate() {
                let (lo, hi) = (g * hpk * hd, (g + 1) * hpk * hd);
                s.step_group(&q[lo..hi], &k[g * hd..(g + 1) * hd], &v[g * hd..(g + 1) * hd], &mut host[lo..hi]);
            }
            let mut dev = vec![0f32; nh * hd];
            unsafe {
                let d = st.o1.get_mut(&key).unwrap();
                let (nkp, nvp) = (d.nk.ptr, d.nv.ptr);
                let row = o1_advance(c, &af, d, nkv, hpk, step).unwrap() as usize;
                let off = (row * nkv * hd * 4) as u64;
                cu::memcpy_htod_async(nkp + off, &k, c.stream).unwrap();
                cu::memcpy_htod_async(nvp + off, &v, c.stream).unwrap();
                cu::memcpy_htod_async(qb.ptr, &q, c.stream).unwrap();
                let mut t0 = std::time::Instant::now();
                o1_read(c, &mut st, &af, key, qb.ptr, 0, ao.ptr, nh, nkv, hd, &mut t0).unwrap();
                cu::memcpy_dtoh_async(&mut dev, ao.ptr, c.stream).unwrap();
            }
            c.sync().unwrap();
            for (a, b) in dev.iter().zip(&host) {
                worst = worst.max((a - b).abs());
                top = top.max(b.abs());
            }
        }
        let d = st.o1.remove(&key).unwrap();
        eprintln!(
            "cuda: far field against the host over {} rows: largest difference {worst:.2e} (largest value {top:.2}), {} keys in the far field",
            3 * w / 2,
            d.far_len
        );
        assert_eq!(d.far_len, states[0].far_len(0));
        assert!(worst < 1e-3 * top.max(1.0), "far field off the host by {worst}");
    }

    #[test]
    fn kernels_match_the_host_codecs() {
        let Some(c) = ctx() else {
            eprintln!("cuda: no device or libraries here — kernel parity skipped");
            return;
        };
        let _st = c.st.lock().unwrap();
        c.current().unwrap();
        let mut seed = 0x5eed_u64;
        // shapes with a ragged last warp iteration (groups not a multiple of 32)
        for &(rows, cols) in &[(96usize, 160usize), (40, 2560), (17, 1056)] {
            let x: Vec<f32> = (0..cols).map(|_| rnd(&mut seed)).collect();
            // q4tp
            let bytes = q4tp_payload(rows, cols, &mut seed);
            let mut w = vec![0f32; rows * cols];
            quant::dequant_q4tp(&bytes, rows, cols, &mut w);
            let m = dev_mat(6, &bytes, rows, cols).unwrap();
            let e = worst_rel(&device_matvec(c, &m, &x, None, 0.0), &host_matvec(&w, &x, rows, cols));
            assert!(e < 2e-5, "q4tp {rows}x{cols}: matvec off by {e:e}");
            // its dense planes: f32 bit for bit, f16 the rounding of it
            let quads = (rows * cols / 4).div_ceil(256) as u32;
            let (rows_i, cols_i) = (rows as i32, cols as i32);
            let wd = Buf::new(rows * cols * 4).unwrap();
            let mut dense = vec![0f32; rows * cols];
            let mut dense_h = vec![0u16; rows * cols];
            unsafe {
                c.launch(c.f.expand_q4tp, (quads, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd.ptr, rows_i, cols_i)).unwrap();
                cu::memcpy_dtoh_async(&mut dense, wd.ptr, c.stream).unwrap();
                c.launch(c.f.expand_q4tp_h, (quads, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd.ptr, rows_i, cols_i)).unwrap();
                cu::memcpy_dtoh_async(&mut dense_h, wd.ptr, c.stream).unwrap();
            }
            c.sync().unwrap();
            assert!(dense.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()), "q4tp {rows}x{cols}: f32 plane differs");
            assert!(
                dense_h.iter().zip(&w).all(|(a, b)| *a == quant::f32_to_f16(*b)),
                "q4tp {rows}x{cols}: f16 plane is not the rounded f32 one"
            );
            // q8_2f (its width a multiple of 16)
            if cols % 16 == 0 {
                let vals: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut seed) * 0.05).collect();
                let bytes = quant::encode_q8_2f(&vals, rows, cols);
                let mut w = vec![0f32; rows * cols];
                quant::dequant_q8_2f(&bytes, rows, cols, &mut w);
                let m = dev_mat(7, &bytes, rows, cols).unwrap();
                let e = worst_rel(&device_matvec(c, &m, &x, None, 0.0), &host_matvec(&w, &x, rows, cols));
                assert!(e < 2e-5, "q8_2f {rows}x{cols}: matvec off by {e:e}");
                unsafe {
                    c.launch(c.f.expand_q8, (quads, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd.ptr, rows_i, cols_i)).unwrap();
                    cu::memcpy_dtoh_async(&mut dense, wd.ptr, c.stream).unwrap();
                }
                c.sync().unwrap();
                assert!(dense.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()), "q8_2f {rows}x{cols}: f32 plane differs");
            }
            // f32, with the RMS norm of the row folded into the staging
            let wf: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut seed)).collect();
            let nw: Vec<f32> = (0..cols).map(|_| 1.0 + 0.5 * rnd(&mut seed)).collect();
            let m = DevMat { kind: 4, rows, cols, w: Arc::new(Buf::from_slice(&wf).unwrap()), aux0: None, aux1: None, held: 0 };
            let xn = crate::inference::rms_norm(&x, &nw, 1e-6, cortiq_core::types::NormStyle::Qwen);
            let e = worst_rel(&device_matvec(c, &m, &x, Some(&nw), 1e-6), &host_matvec(&wf, &xn, rows, cols));
            assert!(e < 2e-5, "f32 {rows}x{cols} with the norm folded: off by {e:e}");
        }
    }
}
