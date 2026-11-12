//! MiniMax-H3 stacks on CUDA: the DiT of MiniMax-H3 / MiniMax-H3-Image and
//! the ViT decoder of their VAEs as resident chains, and a plain batched
//! product for the stages around them.
//!
//! **The chain** (`h3_chain`). The residual stream goes up once and comes
//! back once; between them every block runs on the card: an adaLN-modulated
//! RMS norm into f16, the qkv GEMM, per-head q/k norm and the split-half
//! rotation into head-major planes, attention as two batched GEMMs a chunk of
//! query rows with a softmax between them, the output projection, the gated
//! residual, then the SwiGLU MLP the same way. Weight products are cuBLAS on
//! the tensor cores (f16 operands, f32 accumulation and output).
//!
//! **Memory is a ceiling, not a consequence.** Activations that grow with
//! the sequence are the stream itself and the q / k / v planes; everything
//! else — a qkv panel, the MLP's middle, the score rows — is produced a chunk
//! of rows at a time, so the working set is set by the chunk and not by the
//! picture. Weights are kept on the card in their codec (a q4tp block of the
//! 20 B DiT is 222 MB) and expanded to a dense f16 plane only while their
//! GEMM runs; one plane buffer serves the whole stack. Blocks that do not fit
//! under the ceiling are not resident at all: their bytes pass through a
//! stream slot on the way to the plane, every step. `CMF_GPU_VRAM_MB` sets
//! the ceiling (the whole process's footprint on the card); without it the
//! chain takes what is free.
//!
//! **The batched product** (`h3_matmat`) serves the per-op callers — the
//! prompt encoder, a stack without a chain — with TF32 GEMMs over f32 planes
//! (their inputs are raw activations, which a half cannot be trusted to
//! hold), under a small plane cache.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cortiq_core::{CmfModel, TensorDtype};

use super::{Buf, Ctx, DevMat, Gemm, Res, State, bl, compile, cs, cu, ctx, debug, decline, derr, dev_mat, func};
use crate::gpu::H3Chain;

/// The Lumina Next-DiT block, one block a call, on this module's weight store.
#[path = "lumina.rs"]
mod lumina;
pub use lumina::lumina_block;

/// LTX-2.5's blocks as a resident chain, on this module's weight store.
#[path = "ltx.rs"]
mod ltx;
pub use ltx::{ltx_chain, ltx_vae_chain};

const H3_SRC: &str = include_str!("h3.cu");
const FLASH_SRC: &str = include_str!("h3_flash.cu");

/// The MLP's middle is stored divided by this before its GEMM (`h3_swiglu`).
const ACT_INV: f32 = 1.0 / 64.0;
/// Attention weights are stored multiplied by this (`h3_softmax`).
const P_SCALE: f32 = 4096.0;

#[derive(Clone, Copy)]
struct Funcs {
    norm_mod: cs::CUfunction,
    qkv_split: cs::CUfunction,
    softmax: cs::CUfunction,
    heads_join: cs::CUfunction,
    resid: cs::CUfunction,
    swiglu: cs::CUfunction,
    pack: cs::CUfunction,
    unpack: cs::CUfunction,
    part_split: cs::CUfunction,
    to_half: cs::CUfunction,
    sat_count: cs::CUfunction,
    abs_max: cs::CUfunction,
    cv_swap: cs::CUfunction,
    cv_gn_stats: cs::CUfunction,
    cv_gn_silu: cs::CUfunction,
    cv_im2col: cs::CUfunction,
    cv_im2col1d: cs::CUfunction,
    cv_add: cs::CUfunction,
}

#[derive(Default)]
pub(super) struct H3State {
    funcs: Option<Funcs>,
    /// The flash attention kernel by (head width, causal) (None: this card
    /// or width has none, and attention is two GEMMs a chunk).
    flash: HashMap<(usize, bool), Option<cs::CUfunction>>,
    /// Chain weights kept on the card in their codec, by (model, tensor).
    mats: HashMap<(u64, usize), Arc<DevMat>>,
    mats_bytes: u64,
    /// Working buffers by name; they only grow.
    bufs: HashMap<&'static str, Buf>,
    /// Dense f32 planes of the per-op product, with the tick of their last use.
    planes: HashMap<(u64, usize), (Buf, u64)>,
    plane_bytes: u64,
    tick: u64,
    /// Whether cuBLAS took the batched f16 -> f32 GEMM (None: untried).
    batched: Option<bool>,
    /// f16 planes of host f32 weights (a convolution's kernels), by the
    /// address and length of the host slice.
    host_w: HashMap<(usize, usize), Buf>,
    /// The kernels of the LTX-2.5 chain (`ltx.rs`), once compiled.
    ltx: Option<ltx::LxFuncs>,
    /// The kernels of the Lumina block (`lumina.rs`), once compiled.
    lumina: Option<lumina::LbFuncs>,
}

impl H3State {
    fn funcs(&mut self, c: &Ctx) -> Res<Funcs> {
        if let Some(f) = self.funcs {
            return Ok(f);
        }
        let module = compile(H3_SRC, Vec::new(), c.arch)?;
        let f = Funcs {
            norm_mod: func(module, "h3_norm_mod")?,
            qkv_split: func(module, "h3_qkv_split")?,
            softmax: func(module, "h3_softmax")?,
            heads_join: func(module, "h3_heads_join")?,
            resid: func(module, "h3_resid")?,
            swiglu: func(module, "h3_swiglu")?,
            pack: func(module, "h3_pack")?,
            unpack: func(module, "h3_unpack")?,
            part_split: func(module, "h3_part_split")?,
            to_half: func(module, "h3_to_half")?,
            sat_count: func(module, "h3_sat_count")?,
            abs_max: func(module, "h3_abs_max")?,
            cv_swap: func(module, "cv_swap")?,
            cv_gn_stats: func(module, "cv_gn_stats")?,
            cv_gn_silu: func(module, "cv_gn_silu")?,
            cv_im2col: func(module, "cv_im2col")?,
            cv_im2col1d: func(module, "cv_im2col1d")?,
            cv_add: func(module, "cv_add")?,
        };
        self.funcs = Some(f);
        Ok(f)
    }

    /// The attention kernel that materializes nothing, for this head width:
    /// tensor cores through `mma.sync` (compute 8.0 and up), a width that is
    /// a multiple of 32. `CMF_H3_FLASH=0` keeps the two-GEMM path.
    fn flash(&mut self, c: &Ctx, hd: usize, causal: bool) -> Option<cs::CUfunction> {
        if let Some(f) = self.flash.get(&(hd, causal)) {
            return *f;
        }
        let ok = std::env::var("CMF_H3_FLASH").as_deref() != Ok("0")
            && hd % 32 == 0
            && (32..=256).contains(&hd)
            && !matches!(c.arch, None | Some("compute_75"));
        let f = if ok {
            let defs = vec![format!("-DHD={hd}"), format!("-DCAUSAL={}", causal as u32)];
            match compile(FLASH_SRC, defs, c.arch).and_then(|m| func(m, "h3_flash")) {
                Ok(f) => Some(f),
                Err(e) => {
                    if debug() {
                        eprintln!("cuda h3: no flash attention for head width {hd}: {e}");
                    }
                    None
                }
            }
        } else {
            None
        };
        self.flash.insert((hd, causal), f);
        f
    }

    /// A working buffer of at least `bytes` (contents not kept across growth).
    fn buf(&mut self, c: &Ctx, name: &'static str, bytes: usize) -> Res<u64> {
        if let Some(b) = self.bufs.get(name) {
            if b.bytes >= bytes {
                return Ok(b.ptr);
            }
            // queued work may still read the old one
            c.sync()?;
            self.bufs.remove(name);
        }
        let b = Buf::new(bytes.next_multiple_of(2 << 20))?;
        let p = b.ptr;
        self.bufs.insert(name, b);
        Ok(p)
    }

    fn release(&mut self) {
        self.mats.clear();
        self.mats_bytes = 0;
        self.bufs.clear();
        self.planes.clear();
        self.plane_bytes = 0;
        self.host_w.clear();
    }
}

/// Stage timers of `CMF_CUDA_PROF=1` (each closed by a stream sync).
const STAGES: [&str; 13] = [
    "upload", "weights", "norm", "qkv gemm", "qk norm+rope", "attention", "softmax", "values", "out gemm", "fc1 gemm",
    "swiglu", "fc2 gemm", "download",
];
static STAGE_NS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static CHAIN_CALLS: AtomicU64 = AtomicU64::new(0);
static STREAMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// A streamed weight's way to the card: host preparation of its planes, the
/// copies, and (per-op product only) expansion, GEMM and the trip home.
const MM_STAGES: [&str; 6] = ["host planes", "copy up", "expand", "rows up", "gemm", "rows down"];
static MM_NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static MM_CALLS: AtomicU64 = AtomicU64::new(0);

fn mm_add(s: usize, t0: std::time::Instant) {
    if prof_on() {
        MM_NS[s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_CUDA_PROF").is_some())
}

fn stage(c: &Ctx, s: usize, t0: &mut std::time::Instant) -> Res<()> {
    if prof_on() {
        c.sync()?;
        STAGE_NS[s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        *t0 = std::time::Instant::now();
    }
    Ok(())
}

/// The chain's stage times and the per-op product's, most expensive first
/// (`CMF_CUDA_PROF=1`).
pub fn h3_prof_report() -> Option<String> {
    if !prof_on() || (CHAIN_CALLS.load(Ordering::Relaxed) == 0 && MM_CALLS.load(Ordering::Relaxed) == 0) {
        return None;
    }
    let table = |v: Vec<(u64, &str)>, s: &mut String| {
        let mut v = v;
        let total: u64 = v.iter().map(|x| x.0).sum();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        for (ns, name) in v {
            if ns > 0 {
                s.push_str(&format!("\n  {name:<14} {:>8.2} s  {:>5.1}%", ns as f64 / 1e9, 100.0 * ns as f64 / total.max(1) as f64));
            }
        }
    };
    let v: Vec<(u64, &str)> = STAGE_NS.iter().map(|a| a.load(Ordering::Relaxed)).zip(STAGES).collect();
    let mut s = format!(
        "cuda h3 chain: {} call(s), {:.2} s on the card, {:.1} GB of weights streamed",
        CHAIN_CALLS.load(Ordering::Relaxed),
        v.iter().map(|x| x.0).sum::<u64>() as f64 / 1e9,
        STREAMED_BYTES.load(Ordering::Relaxed) as f64 / 1e9
    );
    table(v, &mut s);
    let m: Vec<(u64, &str)> = MM_NS.iter().map(|a| a.load(Ordering::Relaxed)).zip(MM_STAGES).collect();
    s.push_str(&format!(
        "\ncuda h3 product: {} call(s), {:.2} s",
        MM_CALLS.load(Ordering::Relaxed),
        m.iter().map(|x| x.0).sum::<u64>() as f64 / 1e9
    ));
    table(m, &mut s);
    Some(s)
}

/// `CMF_GPU_VRAM_MB`: the ceiling on the process's footprint on the card.
fn ceiling() -> Option<u64> {
    std::env::var("CMF_GPU_VRAM_MB").ok().and_then(|v| v.parse::<u64>().ok()).map(|mb| mb << 20)
}

/// A tensor the chain can read: its codec kind, shape and bytes.
fn packed(model: &Arc<CmfModel>, idx: usize) -> Res<(u8, usize, usize, &[u8])> {
    let e = model.tensors.get(idx).ok_or("tensor index out of range")?;
    let kind = match e.dtype {
        TensorDtype::Q4TiledP => 6u8,
        TensorDtype::Q8_2f => 7u8,
        d => return Err(format!("{}: codec {d:?} has no CUDA plane", e.name)),
    };
    if e.shape.len() != 2 {
        return Err(format!("{}: not a matrix", e.name));
    }
    Ok((kind, e.shape[0], e.shape[1], model.entry_bytes(e)))
}

/// What a weight takes on the card in its codec (`dev_mat`'s buffers; the
/// card rounds each one up to its page).
fn packed_dev_bytes(kind: u8, rows: usize, cols: usize) -> u64 {
    let page = |b: usize| (b + 64).next_multiple_of(2 << 20) as u64;
    match kind {
        6 => page(rows * (cols / 32) * 16) + page(rows * (cols / 32)) + page(rows * 32 * 4),
        _ => page(rows * cols) + page(rows * 4) + page(cols * 4),
    }
}

fn dev_bytes(m: &DevMat) -> u64 {
    (m.w.bytes + m.aux0.as_ref().map_or(0, |b| b.bytes) + m.aux1.as_ref().map_or(0, |b| b.bytes)) as u64
}

/// The planes `dev_mat` uploads, prepared on the host: (body, aux0, aux1).
enum HostPlanes<'a> {
    Q4tp(&'a [u8], Vec<u8>, Vec<f32>),
    Q8(&'a [u8], Vec<f32>, Vec<f32>),
}

fn host_planes(kind: u8, bytes: &[u8], rows: usize, cols: usize) -> Res<HostPlanes<'_>> {
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
            // rows across the cores: a streamed weight pays this every time
            // it passes, and it was as long as the copy to the card
            let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, 8);
            let per = rows.div_ceil(threads).max(1);
            std::thread::scope(|sc| {
                for (t, (lc, cc)) in lad.chunks_mut(per * 32).zip(codes.chunks_mut(per * gpr)).enumerate() {
                    sc.spawn(move || {
                        let r0 = t * per;
                        for (i, (l, cd)) in lc.chunks_exact_mut(32).zip(cc.chunks_exact_mut(gpr)).enumerate() {
                            let r = r0 + i;
                            l.copy_from_slice(&q4tp_ladder(params, r));
                            let cr = &bytes[codes_off + r * stride..codes_off + (r + 1) * stride];
                            for (g, o) in cd.iter_mut().enumerate() {
                                *o = q4tp_code(cr, g) as u8;
                            }
                        }
                    });
                }
            });
            Ok(HostPlanes::Q4tp(&bytes[..params_off], codes, lad))
        }
        _ => {
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
            Ok(HostPlanes::Q8(
                &bytes[..body],
                f16s(&bytes[body..body + rows * 2]),
                f16s(&bytes[body + rows * 2..body + rows * 2 + cols * 2]),
            ))
        }
    }
}

/// Tensor `idx` as a dense plane at `dst`: f16 (`half`) or f32. A weight that
/// is resident, or that `keep` makes resident, expands from its own buffers;
/// any other passes through the stream slot.
unsafe fn expand(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    idx: usize,
    dst: u64,
    keep: bool,
    half: bool,
) -> Res<(usize, usize)> {
    let key = (model.uid(), idx);
    let (kind, rows, cols, bytes) = packed(model, idx)?;
    let quads = (rows * cols / 4).div_ceil(256) as u32;
    let (ri, ci) = (rows as i32, cols as i32);
    let f = match (kind, half) {
        (6, true) => c.f.expand_q4tp_h,
        (6, false) => c.f.expand_q4tp,
        (_, true) => c.f.expand_q8_h,
        (_, false) => c.f.expand_q8,
    };
    let m = match st.h3.mats.get(&key) {
        Some(m) => Some(m.clone()),
        None if keep => {
            let m = Arc::new(dev_mat(kind, bytes, rows, cols)?);
            // `dev_mat` copies from pageable memory on the default stream:
            // the call returns once the bytes are staged, not once they are
            // on the card, and the chain's stream does not wait for the
            // default one. Without this, the first call of a chain read
            // weights still in flight (two runs of one seed differed).
            cu::ctx::synchronize().map_err(derr)?;
            st.h3.mats_bytes += dev_bytes(&m);
            st.h3.mats.insert(key, m.clone());
            Some(m)
        }
        None => None,
    };
    unsafe {
        match m {
            Some(m) => c.launch(f, (quads, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), dst, ri, ci))?,
            None => {
                let tp = std::time::Instant::now();
                let (body, a0, a1): (&[u8], Vec<u8>, Vec<f32>) = match host_planes(kind, bytes, rows, cols)? {
                    HostPlanes::Q4tp(b, codes, lad) => (b, codes, lad),
                    HostPlanes::Q8(b, rs, col) => {
                        // the row scales ride the byte slot as raw f32 bytes
                        let raw: Vec<u8> = rs.iter().flat_map(|v| v.to_le_bytes()).collect();
                        (b, raw, col)
                    }
                };
                mm_add(0, tp);
                let tu = std::time::Instant::now();
                let pb = st.h3.buf(c, "slot body", body.len() + 64)?;
                let p0 = st.h3.buf(c, "slot aux0", a0.len() + 64)?;
                let p1 = st.h3.buf(c, "slot aux1", a1.len() * 4 + 64)?;
                cu::memcpy_htod_async(pb, body, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(p0, &a0, c.stream).map_err(derr)?;
                cu::memcpy_htod_async(p1, &a1, c.stream).map_err(derr)?;
                mm_add(1, tu);
                STREAMED_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                c.launch(f, (quads, 1), 256, 0, kargs!(pb, p0, p1, dst, ri, ci))?;
            }
        }
    }
    Ok((rows, cols))
}

/// `count` GEMMs of one shape over f16 operands into f32, operand i `stride`
/// elements after operand i - 1: one batched call where cuBLAS takes the
/// mixed types, a call a matrix where it does not.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_many(
    c: &Ctx,
    st: &mut State,
    ta: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: u64,
    lda: usize,
    sa: usize,
    b: u64,
    ldb: usize,
    sb: usize,
    cptr: u64,
    ldc: usize,
    sc: usize,
    count: usize,
) -> Res<()> {
    if st.h3.batched != Some(false) {
        let op = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
        let beta = 0.0f32;
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
                bl::cudaDataType::CUDA_R_16F,
                lda as i32,
                sa as i64,
                b as *const c_void,
                bl::cudaDataType::CUDA_R_16F,
                ldb as i32,
                sb as i64,
                std::ptr::from_ref(&beta).cast(),
                cptr as *mut c_void,
                bl::cudaDataType::CUDA_R_32F,
                ldc as i32,
                sc as i64,
                count as i32,
                bl::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        if s as u32 == 0 {
            st.h3.batched = Some(true);
            return Ok(());
        }
        if st.h3.batched == Some(true) {
            return Err(format!("cuBLAS batched gemm {count} x {m}x{n}x{k}: {s:?}"));
        }
        if debug() {
            eprintln!("cuda h3: the batched f16 GEMM is not taken ({s:?}); one call a head");
        }
        st.h3.batched = Some(false);
    }
    for i in 0..count {
        unsafe {
            c.gemm(
                ta,
                false,
                m,
                n,
                k,
                alpha,
                a + (i * sa * 2) as u64,
                lda,
                b + (i * sb * 2) as u64,
                ldb,
                0.0,
                cptr + (i * sc * 4) as u64,
                ldc,
                Gemm::Half,
            )?;
        }
    }
    Ok(())
}

/// Where each block's constants sit in the one buffer they are uploaded in
/// (offsets in floats; `usize::MAX`: the block has none).
#[derive(Clone, Copy)]
struct Consts {
    norm1: usize,
    norm2: usize,
    q_norm: usize,
    k_norm: usize,
    mods: usize,
    qkv_bias: usize,
    out_bias: usize,
    fc1_bias: usize,
    fc2_bias: usize,
    scale1: usize,
    scale2: usize,
}

pub fn h3_chain(job: &H3Chain<'_>, h: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { chain(c, &mut st, job, h) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 chain", &e);
            false
        }
    }
}

unsafe fn chain(c: &Ctx, st: &mut State, job: &H3Chain<'_>, h: &mut [f32]) -> Res<()> {
    let (n, hs, nh, hd, ffn) = (job.n, job.hidden, job.heads, job.head_dim, job.ffn);
    let g = job.groups.max(1);
    let (inner, rows_all) = (nh * hd, n * g);
    if n == 0 || job.blocks.is_empty() || h.len() != rows_all * hs {
        return Err("empty stack or a stream of the wrong size".into());
    }
    if hd % 32 != 0 || hd > 256 {
        return Err(format!("head width {hd} outside the kernels"));
    }
    let pairs = if job.angles.is_empty() { 0 } else { job.angles.len() / n };
    if pairs * n != job.angles.len() || 2 * pairs > hd {
        return Err("rotation table of the wrong shape".into());
    }
    let modulated = job.blocks.iter().any(|b| !b.mods.is_empty());
    if modulated && job.rows.len() != rows_all {
        return Err("modulation rows of the wrong length".into());
    }
    let f = st.h3.funcs(c)?;
    let cap = ceiling();

    // ── the working set ──
    // Panels are produced a chunk of rows at a time; the chunk is what a
    // panel budget buys, and the budget follows the ceiling.
    let panel: usize = match cap {
        Some(b) if b < (5 << 30) => 96 << 20,
        Some(b) if b < (9 << 30) => 192 << 20,
        _ => 384 << 20,
    };
    let per_row = (3 * inner).max(2 * ffn) * 4 + ffn * 2 + hs * 6;
    let cq = (panel / per_row).clamp(16, 4096).min(rows_all);
    let flash = st.h3.flash(c, hd, false);
    // Without the flash kernel, score rows: 4 B of scores and 2 B of
    // weights a key, for every head, a chunk of query rows at a time.
    let ca = if flash.is_some() {
        cq.min(n)
    } else if g > 1 {
        n
    } else {
        (panel / (nh * n * 6)).clamp(8, n)
    };
    if flash.is_none() && g * nh * ca * n * 6 > (2usize << 30) {
        return Err(format!("{g} groups of {n} rows: the score rows do not fit a chunk"));
    }
    // values are stored dim-major with the keys rounded up to 16
    let ns = n.next_multiple_of(16);
    let x = st.h3.buf(c, "x", rows_all * hs * 4)?;
    let xn = st.h3.buf(c, "xn", cq * hs * 2)?;
    let y = st.h3.buf(c, "y", cq * (3 * inner).max(2 * ffn) * 4)?;
    let y2 = st.h3.buf(c, "y2", cq.max(ca * g) * hs * 4)?;
    let act = st.h3.buf(c, "act", cq * ffn * 2)?;
    let qp = st.h3.buf(c, "q", rows_all * inner * 2)?;
    let kp = st.h3.buf(c, "k", rows_all * inner * 2)?;
    let vp = st.h3.buf(c, "v", g * inner * ns * 2 + 64)?;
    let (sp, pp, op, zp, ao) = if flash.is_some() {
        // the attention output of every row, as the output projection reads it
        (0, 0, 0, 0, st.h3.buf(c, "ao", rows_all * inner * 2)?)
    } else {
        (
            st.h3.buf(c, "s", g * nh * ca * n * 4)?,
            st.h3.buf(c, "p", g * nh * ca * n * 2)?,
            st.h3.buf(c, "o", g * nh * ca * hd * 4)?,
            st.h3.buf(c, "z", g * nh * ca * 4 + 64)?,
            st.h3.buf(c, "ao", g * ca * inner * 2)?,
        )
    };
    // one plane buffer: [fc1 | fc2] of the MLP half, [qkv | out] of the other
    let w = st.h3.buf(c, "w", (3 * ffn * hs).max(4 * inner * hs) * 2)?;

    // ── constants: one upload ──
    let mut consts: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        if v.is_empty() {
            return usize::MAX;
        }
        let at = consts.len();
        consts.extend_from_slice(v);
        at
    };
    let mut lay: Vec<Consts> = Vec::with_capacity(job.blocks.len());
    for b in job.blocks {
        if b.norm1.len() != hs || b.norm2.len() != hs {
            return Err("a norm of the wrong width".into());
        }
        if !b.mods.is_empty() && b.mods.len() % (6 * hs) != 0 {
            return Err("a modulation table of the wrong width".into());
        }
        lay.push(Consts {
            norm1: put(b.norm1),
            norm2: put(b.norm2),
            q_norm: put(b.q_norm),
            k_norm: put(b.k_norm),
            mods: put(b.mods),
            qkv_bias: put(b.qkv_bias),
            out_bias: put(b.out_bias),
            fc1_bias: put(b.fc1_bias),
            fc2_bias: put(b.fc2_bias),
            scale1: put(b.scale1),
            scale2: put(b.scale2),
        });
    }
    let cs_tab: Vec<f32> = if pairs == 0 {
        Vec::new()
    } else {
        // (cos, sin) per row and pair; one group's angles serve every group
        let mut t = Vec::with_capacity(rows_all * pairs * 2);
        for _ in 0..g {
            for a in job.angles {
                let (s, co) = a.sin_cos();
                t.push(co);
                t.push(s);
            }
        }
        t
    };
    let cb = st.h3.buf(c, "consts", consts.len() * 4 + 64)?;
    let csb = st.h3.buf(c, "cs", cs_tab.len() * 4 + 64)?;
    let mrow = st.h3.buf(c, "mrow", rows_all * 4 + 64)?;
    // `CMF_H3_SAT=1`: count the halves every f16 panel stores at the
    // saturation value, and the largest magnitude of every f32 panel — one
    // line a chain call. The panels are fixed-scale halves; a value past
    // their range is clamped, silently.
    let sat = std::env::var("CMF_H3_SAT").as_deref() == Ok("1");
    let satb = if sat { st.h3.buf(c, "sat", 256)? } else { 0 };

    // ── which blocks stay on the card ──
    let uid = job.model.uid();
    let (free, total) = cu::mem_get_info().map_err(derr)?;
    // what a streamed weight needs on its way through, and a margin for
    // cuBLAS's own workspace
    let reserve: i64 = (160 << 20) + (128 << 20);
    let mut room: i64 = match cap {
        Some(b) => b as i64 - (total - free) as i64,
        None => free as i64,
    } - reserve;
    let mut keep = vec![false; job.blocks.len()];
    let mut all_before = true;
    for (i, b) in job.blocks.iter().enumerate() {
        let mut need = 0u64;
        for ix in [b.qkv, b.out, b.fc1, b.fc2] {
            if !st.h3.mats.contains_key(&(uid, ix)) {
                let (kind, r, co, _) = packed(job.model, ix)?;
                need += packed_dev_bytes(kind, r, co);
            }
        }
        if need == 0 {
            keep[i] = true;
        } else if all_before && room >= need as i64 {
            keep[i] = true;
            room -= need as i64;
        } else {
            all_before = false;
        }
    }
    if debug() {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            let kept = keep.iter().filter(|k| **k).count();
            eprintln!(
                "cuda h3 chain: {} rows x {g}, chunks {cq} / {ca}, {kept} of {} blocks resident, card free {:.1} GB{}",
                n,
                job.blocks.len(),
                free as f64 / 1e9,
                cap.map(|b| format!(", ceiling {:.1} GB", b as f64 / 1e9)).unwrap_or_default()
            );
        }
    }

    CHAIN_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut t0 = std::time::Instant::now();
    unsafe {
        cu::memcpy_htod_async(x, h, c.stream).map_err(derr)?;
        if !consts.is_empty() {
            cu::memcpy_htod_async(cb, &consts, c.stream).map_err(derr)?;
        }
        if !cs_tab.is_empty() {
            cu::memcpy_htod_async(csb, &cs_tab, c.stream).map_err(derr)?;
        }
        if modulated {
            cu::memcpy_htod_async(mrow, job.rows, c.stream).map_err(derr)?;
        }
        if sat {
            cu::memcpy_htod_async(satb, &[0u32; 32], c.stream).map_err(derr)?;
        }
    }
    stage(c, 0, &mut t0)?;
    let count = |p: u64, len: usize, cell: usize| -> Res<()> {
        if sat {
            let (len, cellp) = (len as u64, satb + (cell * 4) as u64);
            unsafe { c.launch(f.sat_count, (len.div_ceil(256) as u32, 1), 256, 0, kargs!(p, len, cellp))? };
        }
        Ok(())
    };
    let amax = |p: u64, len: usize, cell: usize| -> Res<()> {
        if sat {
            let (len, cellp) = (len as u64, satb + (cell * 4) as u64);
            unsafe { c.launch(f.abs_max, (len.div_ceil(256) as u32, 1), 256, 0, kargs!(p, len, cellp))? };
        }
        Ok(())
    };

    let at = |off: usize| -> u64 { if off == usize::MAX { 0 } else { cb + (off * 4) as u64 } };
    let (hsi, nhi, hdi, ni, pi, ffni, gi) = (hs as i32, nh as i32, hd as i32, n as i32, pairs as i32, ffn as i32, g as i32);
    let nsi = ns as i32;
    let inter = job.qkv_interleaved as i32;
    // K and V in the flash kernel's lane order, or plain for the GEMMs
    let lanes = flash.is_some() as i32;
    let scale = 1.0 / (hd as f32).sqrt();
    let hd_block = hd as u32;
    for (bi, blk) in job.blocks.iter().enumerate() {
        let l = lay[bi];
        let (modp, mrowp) = if l.mods == usize::MAX { (0u64, 0u64) } else { (at(l.mods), mrow) };

        // ── attention ──
        let (qr, _) = unsafe { expand(c, st, job.model, blk.qkv, w, keep[bi], true)? };
        let wo = w + (3 * inner * hs * 2) as u64;
        let (or, oc) = unsafe { expand(c, st, job.model, blk.out, wo, keep[bi], true)? };
        if qr != 3 * inner || or != hs || oc != inner {
            return Err("an attention projection of the wrong shape".into());
        }
        stage(c, 1, &mut t0)?;
        let mut r0 = 0usize;
        while r0 < rows_all {
            let rows = cq.min(rows_all - r0);
            let (xr, mr) = (x + (r0 * hs * 4) as u64, if mrowp == 0 { 0 } else { mrowp + (r0 * 4) as u64 });
            let (e0, e1) = (0i32, 1i32);
            unsafe {
                amax(xr, rows * hs, 20)?;
                c.launch(f.norm_mod, (rows as u32, 1), 256, 0, kargs!(xr, at(l.norm1), modp, mr, xn, hsi, job.eps, e0, e1))?;
                count(xn, rows * hs, 0)?;
                stage(c, 2, &mut t0)?;
                c.gemm(true, false, 3 * inner, rows, hs, 1.0, w, hs, xn, hs, 0.0, y, 3 * inner, Gemm::Half)?;
                amax(y, rows * 3 * inner, 16)?;
                stage(c, 3, &mut t0)?;
                let csr = if pairs == 0 { 0 } else { csb + (r0 * pairs * 8) as u64 };
                let r0i = r0 as i32;
                c.launch(
                    f.qkv_split,
                    (rows as u32, nh as u32),
                    hd_block,
                    0,
                    kargs!(y, at(l.qkv_bias), at(l.q_norm), at(l.k_norm), csr, qp, kp, vp, r0i, ni, nsi, nhi, hdi, pi, job.qk_eps, inter, lanes),
                )?;
                stage(c, 4, &mut t0)?;
            }
            r0 += rows;
        }
        // `CMF_H3_DUMP=<dir>` (with `CMF_H3_FLASH=0`, which keeps the planes
        // plain): the attention operands of the blocks in
        // `CMF_H3_DUMP_BLOCKS` (default 0,12,25,37,49) of every modulated
        // call, as f16 — q, k `[nh][n][hd]` and v `[nh][hd][ns]` — for
        // pricing an approximation of this attention off line.
        if modulated && flash.is_none() && g == 1 {
            if let Ok(dir) = std::env::var("CMF_H3_DUMP") {
                let want: Vec<usize> = std::env::var("CMF_H3_DUMP_BLOCKS")
                    .unwrap_or_else(|_| "0,12,25,37,49".into())
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
                if want.contains(&bi) {
                    let call = CHAIN_CALLS.load(Ordering::Relaxed);
                    let _ = std::fs::create_dir_all(&dir);
                    for (name, ptr, bytes) in [("q", qp, n * inner * 2), ("k", kp, n * inner * 2), ("v", vp, inner * ns * 2)] {
                        let mut buf = vec![0u8; bytes];
                        unsafe { cu::memcpy_dtoh_async(&mut buf, ptr, c.stream).map_err(derr)? };
                        c.sync()?;
                        std::fs::write(format!("{dir}/c{call}_b{bi}_{name}.f16"), &buf).map_err(|e| e.to_string())?;
                    }
                    let meta = format!("{{\"n\": {n}, \"ns\": {ns}, \"nh\": {nh}, \"hd\": {hd}, \"call\": {call}, \"block\": {bi}}}");
                    std::fs::write(format!("{dir}/c{call}_b{bi}.json"), meta).map_err(|e| e.to_string())?;
                }
            }
        }
        let planes = g * nh;
        count(qp, rows_all * inner, 4)?;
        count(kp, rows_all * inner, 6)?;
        count(vp, g * inner * ns, 8)?;
        if let Some(fl) = flash {
            // every (group, head) plane in one launch: a warp takes 16
            // query rows through all the keys, nothing materialized
            unsafe {
                let one = 1i32;
                c.launch(fl, (n.div_ceil(64) as u32, planes as u32), 128, 0, kargs!(qp, kp, vp, ao, ni, nsi, nhi, scale, one))?;
            }
            count(ao, rows_all * inner, 10)?;
            stage(c, 5, &mut t0)?;
            let mut r0 = 0usize;
            while r0 < rows_all {
                let rows = cq.min(rows_all - r0);
                let xr = x + (r0 * hs * 4) as u64;
                let mr = if mrowp == 0 { 0 } else { mrowp + (r0 * 4) as u64 };
                let tot = (rows * hs) as u64;
                let e2 = 2i32;
                unsafe {
                    c.gemm(true, false, hs, rows, inner, 1.0, wo, inner, ao + (r0 * inner * 2) as u64, inner, 0.0, y2, hs, Gemm::Half)?;
                    amax(y2, rows * hs, 18)?;
                    c.launch(
                        f.resid,
                        (tot.div_ceil(256) as u32, 1),
                        256,
                        0,
                        kargs!(xr, y2, modp, mr, at(l.scale1), at(l.out_bias), hsi, e2, tot),
                    )?;
                }
                stage(c, 8, &mut t0)?;
                r0 += rows;
            }
        } else {
            // score rows a chunk of each group's queries at a time
            let mut a0 = 0usize;
            while a0 < n {
                let rows = ca.min(n - a0);
                unsafe {
                    gemm_many(
                        c,
                        st,
                        true,
                        n,
                        rows,
                        hd,
                        scale,
                        kp,
                        hd,
                        n * hd,
                        qp + (a0 * hd * 2) as u64,
                        hd,
                        n * hd,
                        sp,
                        n,
                        rows * n,
                        planes,
                    )?;
                    stage(c, 5, &mut t0)?;
                    c.launch(f.softmax, ((planes * rows) as u32, 1), 256, 0, kargs!(sp, pp, zp, ni, P_SCALE))?;
                    stage(c, 6, &mut t0)?;
                    // the values are dim-major: [hd][ns] a plane
                    gemm_many(c, st, true, hd, rows, n, 1.0 / P_SCALE, vp, ns, hd * ns, pp, n, rows * n, op, hd, rows * hd, planes)?;
                    stage(c, 7, &mut t0)?;
                    let total = g * rows * inner;
                    let ri = rows as i32;
                    c.launch(f.heads_join, (total.div_ceil(256) as u32, 1), 256, 0, kargs!(op, zp, ao, ri, nhi, hdi, gi))?;
                    // the chunk's rows of every group, as one batch
                    let brows = g * rows;
                    c.gemm(true, false, hs, brows, inner, 1.0, wo, inner, ao, inner, 0.0, y2, hs, Gemm::Half)?;
                    // a group's chunk is contiguous in the stream; with several
                    // groups a chunk is the whole group, so the batch is too
                    let xr = x + (a0 * hs * 4) as u64;
                    let mr = if mrowp == 0 { 0 } else { mrowp + (a0 * 4) as u64 };
                    let tot = (brows * hs) as u64;
                    let e2 = 2i32;
                    c.launch(
                        f.resid,
                        (tot.div_ceil(256) as u32, 1),
                        256,
                        0,
                        kargs!(xr, y2, modp, mr, at(l.scale1), at(l.out_bias), hsi, e2, tot),
                    )?;
                    stage(c, 8, &mut t0)?;
                }
                a0 += rows;
            }
        }

        // ── the MLP ──
        let (r1, c1) = unsafe { expand(c, st, job.model, blk.fc1, w, keep[bi], true)? };
        let w2 = w + (2 * ffn * hs * 2) as u64;
        let (r2, c2) = unsafe { expand(c, st, job.model, blk.fc2, w2, keep[bi], true)? };
        if r1 != 2 * ffn || c1 != hs || r2 != hs || c2 != ffn {
            return Err("an MLP projection of the wrong shape".into());
        }
        stage(c, 1, &mut t0)?;
        let mut r0 = 0usize;
        while r0 < rows_all {
            let rows = cq.min(rows_all - r0);
            let (xr, mr) = (x + (r0 * hs * 4) as u64, if mrowp == 0 { 0 } else { mrowp + (r0 * 4) as u64 });
            let (e3, e4, e5) = (3i32, 4i32, 5i32);
            unsafe {
                c.launch(f.norm_mod, (rows as u32, 1), 256, 0, kargs!(xr, at(l.norm2), modp, mr, xn, hsi, job.eps, e3, e4))?;
                count(xn, rows * hs, 2)?;
                stage(c, 2, &mut t0)?;
                c.gemm(true, false, 2 * ffn, rows, hs, 1.0, w, hs, xn, hs, 0.0, y, 2 * ffn, Gemm::Half)?;
                amax(y, rows * 2 * ffn, 17)?;
                stage(c, 9, &mut t0)?;
                let tot = (rows * ffn) as u64;
                c.launch(f.swiglu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(y, at(l.fc1_bias), act, ffni, ACT_INV, tot))?;
                count(act, rows * ffn, 12)?;
                stage(c, 10, &mut t0)?;
                c.gemm(true, false, hs, rows, ffn, 1.0 / ACT_INV, w2, ffn, act, ffn, 0.0, y2, hs, Gemm::Half)?;
                amax(y2, rows * hs, 19)?;
                let tot = (rows * hs) as u64;
                c.launch(
                    f.resid,
                    (tot.div_ceil(256) as u32, 1),
                    256,
                    0,
                    kargs!(xr, y2, modp, mr, at(l.scale2), at(l.fc2_bias), hsi, e5, tot),
                )?;
                stage(c, 11, &mut t0)?;
            }
            r0 += rows;
        }
    }
    unsafe { cu::memcpy_dtoh_async(h, x, c.stream).map_err(derr)? };
    c.sync()?;
    if sat {
        let mut cells = [0u32; 32];
        unsafe { cu::memcpy_dtoh_async(&mut cells, satb, c.stream).map_err(derr)? };
        c.sync()?;
        let mx = |i: usize| f32::from_bits(cells[i]);
        eprintln!(
            "h3 sat, chain call {} ({n} rows x {g}, {} blocks): clamped halves (not finite) — norm1 {} ({}), norm2 {} ({}), q {} ({}), k {} ({}), v {} ({}), attn out {} ({}), mlp mid {} ({}); max |f32| — stream {:.4e}, qkv {:.4e}, fc1 {:.4e}, out {:.4e}, fc2 {:.4e}",
            CHAIN_CALLS.load(Ordering::Relaxed),
            job.blocks.len(),
            cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6], cells[7], cells[8], cells[9],
            cells[10], cells[11], cells[12], cells[13],
            mx(20), mx(16), mx(17), mx(18), mx(19),
        );
    }
    stage(c, 12, &mut t0)?;
    Ok(())
}

/// What the plane cache of the per-op product may hold.
fn plane_cap() -> u64 {
    if let Some(mb) = std::env::var("CMF_CUDA_MM_CACHE_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        return mb << 20;
    }
    match ceiling() {
        Some(b) if b < (5 << 30) => 700 << 20,
        _ => 1400 << 20,
    }
}

/// `out[b][rows] = xs[b][cols] · Wᵀ` for a q4tp / q8_2f tensor: its dense f32
/// plane on the card (cached while it fits), TF32 on the tensor cores.
pub fn h3_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || xs.len() != b * cols || out.len() != b * rows {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { matmat(c, &mut st, model, idx, xs, b, rows, cols, out) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 matmat", &e);
            false
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn matmat(
    c: &Ctx,
    st: &mut State,
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Res<()> {
    let key = (model.uid(), idx);
    let (_, r, co, _) = packed(model, idx)?;
    if r != rows || co != cols {
        return Err(format!("tensor is {r}x{co}, the product wants {rows}x{cols}"));
    }
    st.h3.tick += 1;
    let tick = st.h3.tick;
    let need = (rows * cols * 4 + 64) as u64;
    let plane = match st.h3.planes.get_mut(&key) {
        Some((buf, t)) => {
            *t = tick;
            buf.ptr
        }
        None => {
            // make room: the least recently used planes go first
            let cap = plane_cap();
            if st.h3.plane_bytes + need > cap && !st.h3.planes.is_empty() {
                c.sync()?;
                while st.h3.plane_bytes + need > cap && !st.h3.planes.is_empty() {
                    let old = *st.h3.planes.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| k).unwrap();
                    if let Some((buf, _)) = st.h3.planes.remove(&old) {
                        st.h3.plane_bytes -= buf.bytes as u64;
                    }
                }
            }
            let buf = Buf::new(need as usize)?;
            let p = buf.ptr;
            st.h3.plane_bytes += buf.bytes as u64;
            st.h3.planes.insert(key, (buf, tick));
            unsafe { expand(c, st, model, idx, p, false, false)? };
            if prof_on() {
                let te = std::time::Instant::now();
                c.sync()?;
                mm_add(2, te);
            }
            p
        }
    };
    MM_CALLS.fetch_add(1, Ordering::Relaxed);
    // a chunk of rows at a time: the two activation panels stay bounded
    let cb = ((192usize << 20) / (rows.max(cols) * 4)).clamp(1, b);
    let xb = st.h3.buf(c, "mm x", cb * cols * 4)?;
    let yb = st.h3.buf(c, "mm y", cb * rows * 4)?;
    let mut b0 = 0usize;
    while b0 < b {
        let nb = cb.min(b - b0);
        unsafe {
            let t = std::time::Instant::now();
            cu::memcpy_htod_async(xb, &xs[b0 * cols..(b0 + nb) * cols], c.stream).map_err(derr)?;
            mm_add(3, t);
            let t = std::time::Instant::now();
            c.gemm(true, false, rows, nb, cols, 1.0, plane, cols, xb, cols, 0.0, yb, rows, Gemm::Tf32)?;
            if prof_on() {
                c.sync()?;
            }
            mm_add(4, t);
            let t = std::time::Instant::now();
            cu::memcpy_dtoh_async(&mut out[b0 * rows..(b0 + nb) * rows], yb, c.stream).map_err(derr)?;
            mm_add(5, t);
        }
        b0 += nb;
    }
    c.sync()
}

/// Full bidirectional attention over head-major f32 planes from the host
/// (`q`, `k`, `v` `[nh][n][hd]`), `out` `[n][nh · hd]`: the planes go up,
/// become the flash kernel's f16 operands (a head padded to a multiple of 32)
/// and the result comes home. For stacks that run per op — a vision tower.
#[allow(clippy::too_many_arguments)]
pub fn h3_attention(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    nh: usize,
    nkv: usize,
    n: usize,
    hd: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    // grouped-query attention: `nh / nkv` query heads share a key head
    if c.current().is_err() || nkv == 0 || !nh.is_multiple_of(nkv) || n == 0 {
        return false;
    }
    let (len, klen) = (nh * n * hd, nkv * n * hd);
    if qh.len() < len || kh.len() < klen || vh.len() < klen || out.len() < len {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    let mut run = |st: &mut State| -> Res<()> {
        let f = st.h3.funcs(c)?;
        let hdp = hd.next_multiple_of(32);
        let fl = st.h3.flash(c, hdp, false).ok_or("no flash attention kernel for this head width")?;
        let ns = n.next_multiple_of(16);
        let src = st.h3.buf(c, "at src", len * 4)?;
        let qp = st.h3.buf(c, "q", nh * n * hdp * 2)?;
        let kp = st.h3.buf(c, "k", nkv * n * hdp * 2)?;
        let vp = st.h3.buf(c, "v", nkv * ns * hdp * 2 + 64)?;
        let ao = st.h3.buf(c, "ao", n * nh * hdp * 2)?;
        let res = st.h3.buf(c, "at out", len * 4)?;
        let (ni, nsi, hsi, hdi) = (n as i32, ns as i32, hd as i32, hdp as i32);
        unsafe {
            for (mode, (plane, planes, dst)) in [(&qh[..len], nh, qp), (&kh[..klen], nkv, kp), (&vh[..klen], nkv, vp)].into_iter().enumerate() {
                let (mode, pi) = (mode as i32, planes as i32);
                let blocks = ((planes * ns * hdp).div_ceil(256)) as u32;
                cu::memcpy_htod_async(src, plane, c.stream).map_err(derr)?;
                c.launch(f.pack, (blocks, 1), 256, 0, kargs!(src, dst, ni, nsi, pi, hsi, hdi, mode))?;
            }
            let (nhi, hpk) = (nh as i32, (nh / nkv) as i32);
            c.launch(fl, (n.div_ceil(64) as u32, nh as u32), 128, 0, kargs!(qp, kp, vp, ao, ni, nsi, nhi, scale, hpk))?;
            c.launch(f.unpack, (len.div_ceil(256) as u32, 1), 256, 0, kargs!(ao, res, ni, nhi, hsi, hdi))?;
            cu::memcpy_dtoh_async(&mut out[..len], res, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run(&mut st) {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 attention", &e);
            false
        }
    }
}

/// A language-model stack over one causal sequence — the prompt encoder of
/// the H3 pipelines (Qwen3-VL: grouped-query attention with per-head q/k
/// norms and a rotation over the whole head, a SwiGLU MLP with separate gate
/// and up projections, plain residuals) — with the stream on the card from
/// the first layer to the last. Every weight is used once, so none is kept:
/// each passes through the stream slot into the plane buffer, and the stage
/// holds one layer's planes whatever the stack's size.
pub fn h3_te_chain(job: &crate::gpu::TeChain<'_>, h: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { te_chain(c, &mut st, job, h) } {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 encoder chain", &e);
            false
        }
    }
}

unsafe fn te_chain(c: &Ctx, st: &mut State, job: &crate::gpu::TeChain<'_>, h: &mut [f32]) -> Res<()> {
    let (n, hs, nh, nkv, hd, inter) = (job.n, job.hidden, job.heads, job.kv_heads, job.head_dim, job.inter);
    if n == 0 || job.layers.is_empty() || h.len() != n * hs {
        return Err("empty stack or a stream of the wrong size".into());
    }
    if hd % 32 != 0 || hd > 256 || nkv == 0 || nh % nkv != 0 {
        return Err(format!("attention geometry {nh}/{nkv} x {hd} outside the kernels"));
    }
    let pairs = hd / 2;
    if job.cs.len() != n * pairs * 2 {
        return Err("rotation table of the wrong shape".into());
    }
    let f = st.h3.funcs(c)?;
    let fl = st.h3.flash(c, hd, true).ok_or("no causal flash attention kernel")?;
    let (qd, kd) = (nh * hd, nkv * hd);
    let pw = qd + 2 * kd; // the [q | k | v] panel's row
    let ns = n.next_multiple_of(16);
    let panel: usize = match ceiling() {
        Some(b) if b < (5 << 30) => 96 << 20,
        Some(b) if b < (9 << 30) => 192 << 20,
        _ => 384 << 20,
    };
    let per_row = pw.max(2 * inter) * 4 + inter * 2 + hs * 6;
    let cq = (panel / per_row).clamp(16, 4096).min(n);
    let x = st.h3.buf(c, "x", n * hs * 4)?;
    let xn = st.h3.buf(c, "xn", cq * hs * 2)?;
    let y = st.h3.buf(c, "y", cq * pw.max(2 * inter) * 4)?;
    let y2 = st.h3.buf(c, "y2", cq * hs * 4)?;
    let act = st.h3.buf(c, "act", cq * inter * 2)?;
    let qp = st.h3.buf(c, "q", n * qd * 2)?;
    let kp = st.h3.buf(c, "k", n * kd * 2)?;
    let vp = st.h3.buf(c, "v", ns * kd * 2 + 64)?;
    let ao = st.h3.buf(c, "ao", n * qd * 2)?;
    // one plane buffer: [q | k | v], then o, then [gate | up | down]
    let w = st.h3.buf(c, "w", (pw * hs).max(3 * inter * hs) * 2)?;
    let mut consts: Vec<f32> = Vec::new();
    let mut lay: Vec<[usize; 4]> = Vec::with_capacity(job.layers.len());
    for l in job.layers {
        if l.input_norm.len() != hs || l.post_norm.len() != hs || l.q_norm.len() != hd || l.k_norm.len() != hd {
            return Err("a norm of the wrong width".into());
        }
        let mut put = |v: &[f32]| {
            let at = consts.len();
            consts.extend_from_slice(v);
            at
        };
        lay.push([put(l.input_norm), put(l.post_norm), put(l.q_norm), put(l.k_norm)]);
    }
    let cb = st.h3.buf(c, "consts", consts.len() * 4 + 64)?;
    let csb = st.h3.buf(c, "cs", job.cs.len() * 4 + 64)?;
    CHAIN_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut t0 = std::time::Instant::now();
    unsafe {
        cu::memcpy_htod_async(x, h, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(cb, &consts, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(csb, job.cs, c.stream).map_err(derr)?;
    }
    stage(c, 0, &mut t0)?;
    let at = |off: usize| -> u64 { cb + (off * 4) as u64 };
    let (hsi, ni, nsi, hdi, pi, nhi, interi) = (hs as i32, n as i32, ns as i32, hd as i32, pairs as i32, nh as i32, inter as i32);
    let (pwi, hpk) = (pw as i32, (nh / nkv) as i32);
    let scale = 1.0 / (hd as f32).sqrt();
    let (nul, e0) = (0u64, 0i32);
    for (li, l) in job.layers.iter().enumerate() {
        let [n1, n2, qn, kn] = lay[li];
        // ── attention ──
        let (wq, wk, wv) = (w, w + (qd * hs * 2) as u64, w + ((qd + kd) * hs * 2) as u64);
        let dq = unsafe { expand(c, st, job.model, l.q, wq, false, true)? };
        let dk = unsafe { expand(c, st, job.model, l.k, wk, false, true)? };
        let dv = unsafe { expand(c, st, job.model, l.v, wv, false, true)? };
        if dq != (qd, hs) || dk != (kd, hs) || dv != (kd, hs) {
            return Err("an attention projection of the wrong shape".into());
        }
        stage(c, 1, &mut t0)?;
        let mut r0 = 0usize;
        while r0 < n {
            let rows = cq.min(n - r0);
            let xr = x + (r0 * hs * 4) as u64;
            let csr = csb + (r0 * pairs * 8) as u64;
            let r0i = r0 as i32;
            unsafe {
                c.launch(f.norm_mod, (rows as u32, 1), 256, 0, kargs!(xr, at(n1), nul, nul, xn, hsi, job.eps, e0, e0))?;
                stage(c, 2, &mut t0)?;
                c.gemm(true, false, qd, rows, hs, 1.0, wq, hs, xn, hs, 0.0, y, pw, Gemm::Half)?;
                c.gemm(true, false, kd, rows, hs, 1.0, wk, hs, xn, hs, 0.0, y + (qd * 4) as u64, pw, Gemm::Half)?;
                c.gemm(true, false, kd, rows, hs, 1.0, wv, hs, xn, hs, 0.0, y + ((qd + kd) * 4) as u64, pw, Gemm::Half)?;
                stage(c, 3, &mut t0)?;
                let (m0, m1, m2) = (0i32, 1i32, 2i32);
                let (oq, ok, ov) = (0i32, qd as i32, (qd + kd) as i32);
                let hb = hd as u32;
                c.launch(f.part_split, (rows as u32, nh as u32), hb, 0, kargs!(y, pwi, oq, at(qn), csr, qp, r0i, ni, nsi, hdi, pi, job.eps, m0))?;
                c.launch(f.part_split, (rows as u32, nkv as u32), hb, 0, kargs!(y, pwi, ok, at(kn), csr, kp, r0i, ni, nsi, hdi, pi, job.eps, m1))?;
                c.launch(f.part_split, (rows as u32, nkv as u32), hb, 0, kargs!(y, pwi, ov, nul, csr, vp, r0i, ni, nsi, hdi, pi, job.eps, m2))?;
                stage(c, 4, &mut t0)?;
            }
            r0 += rows;
        }
        let (or, oc) = unsafe { expand(c, st, job.model, l.o, w, false, true)? };
        if or != hs || oc != qd {
            return Err("the output projection is of the wrong shape".into());
        }
        stage(c, 1, &mut t0)?;
        unsafe {
            c.launch(fl, (n.div_ceil(64) as u32, nh as u32), 128, 0, kargs!(qp, kp, vp, ao, ni, nsi, nhi, scale, hpk))?;
        }
        stage(c, 5, &mut t0)?;
        let mut r0 = 0usize;
        while r0 < n {
            let rows = cq.min(n - r0);
            let xr = x + (r0 * hs * 4) as u64;
            let tot = (rows * hs) as u64;
            unsafe {
                c.gemm(true, false, hs, rows, qd, 1.0, w, qd, ao + (r0 * qd * 2) as u64, qd, 0.0, y2, hs, Gemm::Half)?;
                c.launch(f.resid, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y2, nul, nul, nul, nul, hsi, e0, tot))?;
            }
            stage(c, 8, &mut t0)?;
            r0 += rows;
        }
        // ── the MLP ──
        let (wg, wu, wd) = (w, w + (inter * hs * 2) as u64, w + (2 * inter * hs * 2) as u64);
        let dg = unsafe { expand(c, st, job.model, l.gate, wg, false, true)? };
        let du = unsafe { expand(c, st, job.model, l.up, wu, false, true)? };
        let dd = unsafe { expand(c, st, job.model, l.down, wd, false, true)? };
        if dg != (inter, hs) || du != (inter, hs) || dd != (hs, inter) {
            return Err("an MLP projection of the wrong shape".into());
        }
        stage(c, 1, &mut t0)?;
        let mut r0 = 0usize;
        while r0 < n {
            let rows = cq.min(n - r0);
            let xr = x + (r0 * hs * 4) as u64;
            unsafe {
                c.launch(f.norm_mod, (rows as u32, 1), 256, 0, kargs!(xr, at(n2), nul, nul, xn, hsi, job.eps, e0, e0))?;
                stage(c, 2, &mut t0)?;
                // gate and up into the two halves of one [gate | up] panel
                c.gemm(true, false, inter, rows, hs, 1.0, wg, hs, xn, hs, 0.0, y, 2 * inter, Gemm::Half)?;
                c.gemm(true, false, inter, rows, hs, 1.0, wu, hs, xn, hs, 0.0, y + (inter * 4) as u64, 2 * inter, Gemm::Half)?;
                stage(c, 9, &mut t0)?;
                let tot = (rows * inter) as u64;
                c.launch(f.swiglu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(y, nul, act, interi, ACT_INV, tot))?;
                stage(c, 10, &mut t0)?;
                c.gemm(true, false, hs, rows, inter, 1.0 / ACT_INV, wd, inter, act, inter, 0.0, y2, hs, Gemm::Half)?;
                let tot = (rows * hs) as u64;
                c.launch(f.resid, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(xr, y2, nul, nul, nul, nul, hsi, e0, tot))?;
                stage(c, 11, &mut t0)?;
            }
            r0 += rows;
        }
    }
    unsafe { cu::memcpy_dtoh_async(h, x, c.stream).map_err(derr)? };
    c.sync()?;
    stage(c, 12, &mut t0)?;
    Ok(())
}

/// A chain of convolution stages over one activation `[c][h][w]` from the
/// host, on the card from the first stage to the last (the VAE encoder of an
/// edit's reference pictures): a convolution is an im2col panel in f16 and
/// one GEMM, GroupNorm + SiLU and the residual additions are kernels, and
/// only the result comes home, `([c'][h'][w'], c', h', w')`.
pub fn h3_conv_chain(
    steps: &[crate::gpu::ConvStep<'_>],
    x: &[f32],
    c0: usize,
    h0: usize,
    w0: usize,
) -> Option<(Vec<f32>, usize, usize, usize)> {
    let c = ctx()?;
    if c.current().is_err() || x.len() != c0 * h0 * w0 {
        return None;
    }
    let mut st = c.st.lock().unwrap();
    match unsafe { conv_chain(c, &mut st, steps, x, c0, h0, w0) } {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("h3 convolution chain", &e);
            None
        }
    }
}

unsafe fn conv_chain(
    c: &Ctx,
    st: &mut State,
    steps: &[crate::gpu::ConvStep<'_>],
    x: &[f32],
    c0: usize,
    h0: usize,
    w0: usize,
) -> Res<(Vec<f32>, usize, usize, usize)> {
    use crate::gpu::ConvStep;
    let f = st.h3.funcs(c)?;
    // the largest activation and im2col chunk the chain will need
    let (mut ch, mut hh, mut ww) = (c0, h0, w0);
    let mut act_max = c0 * h0 * w0;
    for s in steps {
        if let ConvStep::Conv { out_ch, in_ch, k, stride, pad, far, .. } = s {
            if *in_ch != ch {
                return Err(format!("a convolution takes {in_ch} channels, the activation has {ch}"));
            }
            let (ph, pw) = (hh + 2 * pad + *far as usize, ww + 2 * pad + *far as usize);
            hh = (ph - k) / stride + 1;
            ww = (pw - k) / stride + 1;
            ch = *out_ch;
            act_max = act_max.max(ch * hh * ww);
        }
    }
    // three activations (current, next, saved) and the staging of the input
    let a = st.h3.buf(c, "cv a", act_max * 4)?;
    let b = st.h3.buf(c, "cv b", act_max * 4)?;
    let sv = st.h3.buf(c, "cv s", act_max * 4)?;
    let stats = st.h3.buf(c, "cv st", 64 * 4)?;
    let (mut cur, mut nxt) = (a, b);
    let (mut ch, mut hh, mut ww) = (c0, h0, w0);
    let (mut sc, mut saved_px) = (0usize, 0usize);
    unsafe {
        cu::memcpy_htod_async(nxt, x, c.stream).map_err(derr)?;
        let (ci, hwi, one) = (ch as i32, (hh * ww) as i32, 1i32);
        c.launch(f.cv_swap, ((ch * hh * ww).div_ceil(256) as u32, 1), 256, 0, kargs!(nxt, cur, ci, hwi, one))?;
    }
    // a host f32 slice on the card: f32 as it is, or as an f16 plane
    let mut host_f32 = |st: &mut State, v: &[f32]| -> Res<u64> {
        let key = (v.as_ptr() as usize, v.len() | (1 << 62));
        if let Some(bf) = st.h3.host_w.get(&key) {
            return Ok(bf.ptr);
        }
        let bf = Buf::from_slice(v)?;
        // the copy is staged, not landed (see `expand`)
        cu::ctx::synchronize().map_err(derr)?;
        let p = bf.ptr;
        st.h3.host_w.insert(key, bf);
        Ok(p)
    };
    let mut host_f16 = |st: &mut State, v: &[f32]| -> Res<u64> {
        let key = (v.as_ptr() as usize, v.len());
        if let Some(bf) = st.h3.host_w.get(&key) {
            return Ok(bf.ptr);
        }
        let tmp = st.h3.buf(c, "cv wup", v.len() * 4)?;
        let bf = Buf::new(v.len() * 2 + 64)?;
        let p = bf.ptr;
        let n = v.len() as u64;
        unsafe {
            cu::memcpy_htod_async(tmp, v, c.stream).map_err(derr)?;
            c.launch(f.to_half, (v.len().div_ceil(256) as u32, 1), 256, 0, kargs!(tmp, p, n))?;
        }
        // the staging buffer is reused by the next weight
        c.sync()?;
        st.h3.host_w.insert(key, bf);
        Ok(p)
    };
    // one convolution of `src` [h * w][ci] into `dst` [oh * ow][co], bias added
    #[allow(clippy::too_many_arguments)]
    let mut conv = |st: &mut State,
                    src: u64,
                    dst: u64,
                    wt: &[f32],
                    bias: &[f32],
                    co: usize,
                    ci: usize,
                    k: usize,
                    stride: usize,
                    pad: usize,
                    far: bool,
                    h: usize,
                    w: usize|
     -> Res<(usize, usize)> {
        let kk = ci * k * k;
        if wt.len() != co * kk || bias.len() != co {
            return Err("a convolution's weight or bias is of the wrong size".into());
        }
        let (ph, pw) = (h + 2 * pad + far as usize, w + 2 * pad + far as usize);
        let (oh, ow) = ((ph - k) / stride + 1, (pw - k) / stride + 1);
        let wp = host_f16(st, wt)?;
        let bp = host_f32(st, bias)?;
        // an im2col chunk of at most 128 MB
        let chunk = ((128usize << 20) / (kk * 2)).clamp(1, oh * ow);
        let cols = st.h3.buf(c, "cv cols", chunk * kk * 2)?;
        let (cii, hi, wi, owi, ki, si, pi, fi) =
            (ci as i32, h as i32, w as i32, ow as i32, k as i32, stride as i32, pad as i32, far as i32);
        let mut p0 = 0usize;
        while p0 < oh * ow {
            let np = chunk.min(oh * ow - p0);
            let total = (np * ci) as u64;
            let p0i = p0 as i32;
            let out = dst + (p0 * co * 4) as u64;
            unsafe {
                c.launch(
                    f.cv_im2col,
                    (total.div_ceil(256) as u32, 1),
                    256,
                    0,
                    kargs!(src, cols, cii, hi, wi, owi, ki, si, pi, fi, p0i, total),
                )?;
                c.gemm(true, false, co, np, kk, 1.0, wp, kk, cols, kk, 0.0, out, co, Gemm::Half)?;
            }
            p0 += np;
        }
        let (coi, tot, nul) = (co as i32, (oh * ow * co) as u64, 0u64);
        unsafe { c.launch(f.cv_add, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(dst, bp, nul, coi, tot))? };
        Ok((oh, ow))
    };
    for s in steps {
        match s {
            ConvStep::GnSilu { w, b } => {
                if w.len() != ch || b.len() != ch || ch % 32 != 0 {
                    return Err("a group norm of the wrong width".into());
                }
                let (wp, bp) = (host_f32(st, w)?, host_f32(st, b)?);
                let (ci, hwi, tot, eps) = (ch as i32, (hh * ww) as i32, (ch * hh * ww) as u64, 1e-6f32);
                unsafe {
                    c.launch(f.cv_gn_stats, (32, 1), 256, 0, kargs!(cur, stats, ci, hwi, eps))?;
                    c.launch(f.cv_gn_silu, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, stats, wp, bp, ci, tot))?;
                }
            }
            ConvStep::Conv { w, b, out_ch, in_ch, k, stride, pad, far } => {
                let (oh, ow) = conv(st, cur, nxt, w, b, *out_ch, *in_ch, *k, *stride, *pad, *far, hh, ww)?;
                std::mem::swap(&mut cur, &mut nxt);
                (ch, hh, ww) = (*out_ch, oh, ow);
            }
            ConvStep::Save => {
                let bytes = ch * hh * ww * 4;
                unsafe { cu::memcpy_dtod_async(sv, cur, bytes, c.stream).map_err(derr)? };
                (sc, saved_px) = (ch, hh * ww);
            }
            ConvStep::AddSaved { via } => {
                if saved_px != hh * ww {
                    return Err("the saved activation is of another size".into());
                }
                let tot = (ch * hh * ww) as u64;
                let (ci, nul) = (ch as i32, 0u64);
                match via {
                    Some((w, b)) => {
                        // the saved branch through its 1x1 convolution
                        conv(st, sv, nxt, w, b, ch, sc, 1, 1, 0, false, hh, ww)?;
                        unsafe { c.launch(f.cv_add, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nul, nxt, ci, tot))? };
                    }
                    None => {
                        if sc != ch {
                            return Err("the saved activation has other channels and no shortcut".into());
                        }
                        unsafe { c.launch(f.cv_add, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nul, sv, ci, tot))? };
                    }
                }
            }
        }
    }
    let mut out = vec![0f32; ch * hh * ww];
    unsafe {
        let (ci, hwi, zero) = (ch as i32, (hh * ww) as i32, 0i32);
        c.launch(f.cv_swap, (out.len().div_ceil(256) as u32, 1), 256, 0, kargs!(cur, nxt, ci, hwi, zero))?;
        cu::memcpy_dtoh_async(&mut out, nxt, c.stream).map_err(derr)?;
    }
    c.sync()?;
    Ok((out, ch, hh, ww))
}

/// `y[n][m] = x[n][k] · w[m][k]ᵀ`, all f32 and all from the host: TF32 on the
/// tensor cores, a chunk of rows at a time. For products whose weight is not
/// a container tensor (the VAE encoder's convolutions as im2col panels).
pub fn h3_gemm_nt(x: &[f32], w: &[f32], y: &mut [f32], n: usize, k: usize, m: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() || x.len() < n * k || w.len() < m * k || y.len() < n * m {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    let mut run = |st: &mut State| -> Res<()> {
        let wb = st.h3.buf(c, "nt w", m * k * 4)?;
        let cb = ((256usize << 20) / (k.max(m) * 4)).clamp(1, n);
        let xb = st.h3.buf(c, "nt x", cb * k * 4)?;
        let yb = st.h3.buf(c, "nt y", cb * m * 4)?;
        unsafe {
            cu::memcpy_htod_async(wb, &w[..m * k], c.stream).map_err(derr)?;
            let mut r0 = 0usize;
            while r0 < n {
                let nb = cb.min(n - r0);
                cu::memcpy_htod_async(xb, &x[r0 * k..(r0 + nb) * k], c.stream).map_err(derr)?;
                c.gemm(true, false, m, nb, k, 1.0, wb, k, xb, k, 0.0, yb, m, Gemm::Tf32)?;
                cu::memcpy_dtoh_async(&mut y[r0 * m..(r0 + nb) * m], yb, c.stream).map_err(derr)?;
                r0 += nb;
            }
        }
        c.sync()
    };
    match run(&mut st) {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 gemm", &e);
            false
        }
    }
}

/// A 1-D convolution as a product whose column matrix is built on the card:
/// `x` `[ic][n]` (zero outside), `w` `[oc][ic * k]`, tap `j` reading sample
/// `t + j * dil - pad`; `yt` comes back `[out_n][oc]`, the layout of
/// `h3_gemm_nt`. Only `x` crosses the bus on the way in.
#[allow(clippy::too_many_arguments)]
pub fn h3_conv1d(
    x: &[f32],
    w: &[f32],
    ic: usize,
    oc: usize,
    n: usize,
    k: usize,
    pad: usize,
    dil: usize,
    out_n: usize,
    yt: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let kk = ic * k;
    if c.current().is_err() || out_n == 0 || x.len() < ic * n || w.len() < oc * kk || yt.len() < out_n * oc {
        return false;
    }
    // the last tap of the last sample stays within reach of an i32 index
    if n > i32::MAX as usize / 2 || (out_n + k * dil) > i32::MAX as usize / 2 {
        return false;
    }
    let mut st = c.st.lock().unwrap();
    let mut run = |st: &mut State| -> Res<()> {
        let f = st.h3.funcs(c)?;
        let xb = st.h3.buf(c, "c1 x", ic * n * 4)?;
        let wb = st.h3.buf(c, "c1 w", oc * kk * 4)?;
        // a column chunk of at most 256 MB
        let cb = ((256usize << 20) / (kk.max(oc) * 4)).clamp(1, out_n);
        let cols = st.h3.buf(c, "c1 cols", cb * kk * 4)?;
        let yb = st.h3.buf(c, "c1 y", cb * oc * 4)?;
        let (ci, ni, ki, pi, di) = (ic as i32, n as i32, k as i32, pad as i32, dil as i32);
        unsafe {
            cu::memcpy_htod_async(xb, &x[..ic * n], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(wb, &w[..oc * kk], c.stream).map_err(derr)?;
            let mut t0 = 0usize;
            while t0 < out_n {
                let nb = cb.min(out_n - t0);
                let (total, t0i) = ((nb * ic) as u64, t0 as i32);
                c.launch(
                    f.cv_im2col1d,
                    (total.div_ceil(256) as u32, 1),
                    256,
                    0,
                    kargs!(xb, cols, ci, ni, ki, pi, di, t0i, total),
                )?;
                c.gemm(true, false, oc, nb, kk, 1.0, wb, kk, cols, kk, 0.0, yb, oc, Gemm::Tf32)?;
                cu::memcpy_dtoh_async(&mut yt[t0 * oc..(t0 + nb) * oc], yb, c.stream).map_err(derr)?;
                t0 += nb;
            }
        }
        c.sync()
    };
    match run(&mut st) {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("h3 conv1d", &e);
            false
        }
    }
}

/// A pipeline stage is over: free the chain's resident weights, its working
/// buffers and the plane cache.
pub fn h3_release() {
    let Some(c) = ctx() else { return };
    if c.current().is_err() {
        return;
    }
    let mut st = c.st.lock().unwrap();
    let _ = c.sync();
    st.h3.release();
}

/// Bytes of chain weights resident on the card (in their codec).
pub fn h3_resident_bytes() -> u64 {
    ctx().map_or(0, |c| c.st.lock().unwrap().h3.mats_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's 1-D convolution against the loop it replaces: dilation,
    /// zero padding on both sides, and a column chunk boundary inside the
    /// run (TF32 products: three decimal digits).
    #[test]
    fn conv1d_matches_the_host() {
        if ctx().is_none() {
            eprintln!("cuda: no device or libraries here — conv1d parity skipped");
            return;
        }
        let mut seed = 0x1d_c0de_u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) % 2001) as f32 / 1000.0 - 1.0
        };
        for &(ic, oc, n, k, dil) in &[(24usize, 40usize, 3000usize, 7usize, 3usize), (16, 8, 513, 1, 1), (8, 12, 700, 3, 9)] {
            let pad = dil * (k - 1) / 2;
            let out_n = n + 2 * pad - dil * (k - 1);
            let x: Vec<f32> = (0..ic * n).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..oc * ic * k).map(|_| rnd() * 0.1).collect();
            let mut yt = vec![0f32; out_n * oc];
            assert!(h3_conv1d(&x, &w, ic, oc, n, k, pad, dil, out_n, &mut yt), "the card refused {ic}x{oc}x{n}");
            let (mut worst, mut top) = (0f32, 0f32);
            for t in 0..out_n {
                for o in 0..oc {
                    let mut acc = 0f64;
                    for i in 0..ic {
                        for j in 0..k {
                            let s = (t + j * dil) as isize - pad as isize;
                            if s >= 0 && (s as usize) < n {
                                acc += x[i * n + s as usize] as f64 * w[(o * ic + i) * k + j] as f64;
                            }
                        }
                    }
                    worst = worst.max((yt[t * oc + o] - acc as f32).abs());
                    top = top.max((acc as f32).abs());
                }
            }
            assert!(worst < 2e-3 * top.max(1.0), "conv1d {ic}x{oc}x{n} k{k} d{dil}: off by {worst:e} of {top:e}");
        }
    }
}
