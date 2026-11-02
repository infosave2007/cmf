//! Unlimited-OCR's DASH attention on the Metal graphs, the twin of
//! `gpu_wgpu/uocr.rs` (`uocr` has the semantics).
//!
//! The token graph and the chunk prefill project q/k/v, rotate them and append
//! K/V to the layer's mirror; these kernels replace their attend:
//!
//! * `dash_route`: a row's router in one threadgroup, the candidate blocks the
//!   host listed, the read set and its priors;
//! * `dash_close`: a closing block's summary, every 32 tokens;
//! * `dash_attend`: one threadgroup per (head, row) over those blocks and the
//!   local window.
//!
//! `dash_rope_rows` is the chunk's rotation, each row by its own angles, which
//! the graphs' `chunk_rope_kv` cannot do: it takes one position and a stride.

use super::{Ctx, set_bytes_t};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const DMSL: &str = include_str!("uocr_msl.metal");
const BLOCKS_MAX: usize = 4096;
const LIST_MAX: usize = 512;
pub(crate) const CHUNK_MAX: usize = crate::gpu::DASH_CHUNK_MAX;

pub(super) struct Pipes {
    route: ComputePipelineState,
    close: ComputePipelineState,
    attend: ComputePipelineState,
    rope: ComputePipelineState,
    pub(super) add: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

pub(super) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(DMSL, &opts)
            .map_err(|e| format!("dash MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            route: pso("dash_route")?,
            close: pso("dash_close")?,
            attend: pso("dash_attend")?,
            rope: pso("dash_rope_rows")?,
            add: pso("dash_add_rows")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("DASH Metal kernels unavailable ({e})");
            }
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DP {
    hidden: u32,
    d: u32,
    pos: u32,
    block: u32,
    nblk: u32,
    k: u32,
    m: u32,
    close: u32,
    eps: f32,
    inv_sqrt: f32,
    idx_scale: f32,
    floor: f32,
    sigma: f32,
    window: u32,
    nh: u32,
    hd: u32,
    cap: u32,
    scale: f32,
    rows: u32,
    pad: u32,
}

/// One layer's router on the device.
struct LayerDev {
    w: Buffer,
    qn: Buffer,
    kn: Buffer,
    hc: Buffer,
    open: Buffer,
    kbar: Buffer,
    sel: Buffer,
    bias: Buffer,
}

/// The rows' candidates and angles, shared by the layers of a frame.
struct Shared {
    cands: Buffer,
    cands_cap: usize,
    coff: Buffer,
    ang: Buffer,
    written: (u64, usize, usize),
}

static STATE: Mutex<Option<(HashMap<(u64, usize), LayerDev>, Option<Shared>)>> = Mutex::new(None);

fn buf(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
}

fn up(c: &Ctx, data: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// A geometry the kernels hold, checked before anything is encoded.
pub(crate) fn admits(c: &Ctx, d: &crate::gpu::GraphDash<'_>, hidden: usize, nh: usize, hd: usize) -> bool {
    let idim = d.q_norm.len();
    pipes(c).is_some()
        && hidden % 4 == 0
        && hd % 4 == 0
        && idim == 128
        && d.block <= 64
        && d.k <= 16
        && hd / 4 <= 32
        && d.block * d.k + d.block * d.local_blocks <= LIST_MAX
        && d.qk_proj.len() == 2 * idim * hidden
        && nh > 0
}

/// What an encode needs from a graph.
pub(super) struct DashEnc<'a> {
    pub d: &'a crate::gpu::GraphDash<'a>,
    pub key: (u64, usize),
    pub n1: &'a Buffer,
    pub q: &'a Buffer,
    pub kb: &'a Buffer,
    pub vb: &'a Buffer,
    pub attn: &'a Buffer,
    pub cap: usize,
    pub pos0: usize,
    pub k: usize,
    pub nh: usize,
    pub hd: usize,
    pub hidden: usize,
    pub scale: f32,
    pub eps: f32,
}

fn params(e: &DashEnc<'_>) -> DP {
    let d = e.d;
    let idim = d.q_norm.len();
    DP {
        hidden: e.hidden as u32,
        d: idim as u32,
        pos: e.pos0 as u32,
        block: d.block as u32,
        nblk: d.nblk as u32,
        k: d.k as u32,
        m: 16,
        close: 0,
        eps: e.eps,
        inv_sqrt: (idim as f32).sqrt().recip(),
        idx_scale: d.idx_scale,
        floor: d.floor,
        sigma: d.sigma,
        window: (d.block * d.local_blocks) as u32,
        nh: e.nh as u32,
        hd: e.hd as u32,
        cap: e.cap as u32,
        scale: e.scale,
        rows: e.k as u32,
        pad: 0,
    }
}

fn with_layer<R>(c: &Ctx, e: &DashEnc<'_>, f: impl FnOnce(&LayerDev, &Shared) -> R) -> Option<R> {
    let d = e.d;
    let idim = d.q_norm.len();
    let k = e.k;
    if k == 0 || k > CHUNK_MAX || (e.pos0 + k - 1) / d.block >= BLOCKS_MAX {
        return None;
    }
    let half = e.hd / 2;
    let mut g = STATE.lock().unwrap_or_else(|x| x.into_inner());
    let (layers, shared) = g.get_or_insert_with(|| (HashMap::new(), None));
    let sh = shared.get_or_insert_with(|| Shared {
        cands: buf(c, BLOCKS_MAX * 4),
        cands_cap: BLOCKS_MAX,
        coff: buf(c, (CHUNK_MAX + 1) * 4),
        ang: buf(c, CHUNK_MAX * half * 4),
        written: (u64::MAX, 0, 0),
    });
    if d.cands.len() > sh.cands_cap {
        sh.cands_cap = d.cands.len().next_power_of_two();
        sh.cands = buf(c, sh.cands_cap * 4);
    }
    let frame = (e.key.0, e.pos0, k);
    if sh.written != frame || e.key.1 == 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(d.cands.as_ptr(), sh.cands.contents() as *mut u32, d.cands.len());
            std::ptr::copy_nonoverlapping(d.coff.as_ptr(), sh.coff.contents() as *mut u32, d.coff.len());
            std::ptr::copy_nonoverlapping(d.angles.as_ptr(), sh.ang.contents() as *mut f32, k * half);
        }
        sh.written = frame;
    }
    let l = layers.entry(e.key).or_insert_with(|| LayerDev {
        w: up(c, d.qk_proj),
        qn: up(c, d.q_norm),
        kn: up(c, d.k_norm),
        hc: up(c, d.head_cls),
        open: buf(c, 3 * d.block * idim * 4),
        kbar: buf(c, BLOCKS_MAX * idim * 4),
        sel: buf(c, CHUNK_MAX * 17 * 4),
        bias: buf(c, CHUNK_MAX * 16 * 4),
    });
    Some(f(l, sh))
}

/// Encode the router and attend of `e.k` rows at `e.pos0` into `e.attn`.
/// `bar` orders dependent dispatches on a concurrent encoder.
pub(super) fn encode(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    e: &DashEnc<'_>,
    bar: &dyn Fn(&ComputeCommandEncoderRef),
) -> bool {
    let Some(p) = pipes(c) else { return false };
    let d = e.d;
    let prm = params(e);
    let idim = d.q_norm.len();
    with_layer(c, e, |l, sh| {
        enc.set_compute_pipeline_state(&p.route);
        enc.set_buffer(0, Some(e.n1), 0);
        enc.set_buffer(1, Some(&l.w), 0);
        enc.set_buffer(2, Some(&l.qn), 0);
        enc.set_buffer(3, Some(&l.kn), 0);
        enc.set_buffer(4, Some(&sh.ang), 0);
        enc.set_buffer(5, Some(&l.open), 0);
        enc.set_buffer(6, Some(&l.kbar), 0);
        enc.set_buffer(7, Some(&sh.coff), 0);
        enc.set_buffer(8, Some(&sh.cands), 0);
        enc.set_buffer(9, Some(&l.sel), 0);
        enc.set_buffer(10, Some(&l.bias), 0);
        set_bytes_t(enc, 11, &prm);
        enc.dispatch_thread_groups(MTLSize::new(e.k as u64, 1, 1), MTLSize::new(256, 1, 1));
        bar(enc);
        // Blocks closing in this chunk. The rows above read older blocks only:
        // a remote block is at least `local_blocks` behind, and a chunk spans
        // fewer than that.
        let end = e.pos0 + e.k;
        for b in e.pos0 / d.block..=(end - 1) / d.block {
            if (b + 1) * d.block > end {
                break;
            }
            let mut pc = prm;
            pc.close = b as u32;
            enc.set_compute_pipeline_state(&p.close);
            enc.set_buffer(0, Some(&l.open), 0);
            enc.set_buffer(1, Some(&l.hc), 0);
            enc.set_buffer(2, Some(&l.kbar), 0);
            set_bytes_t(enc, 3, &pc);
            let n = (idim / 4) as u64;
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(n.min(32), 1, 1));
            bar(enc);
        }
        enc.set_compute_pipeline_state(&p.attend);
        enc.set_buffer(0, Some(e.q), 0);
        enc.set_buffer(1, Some(e.kb), 0);
        enc.set_buffer(2, Some(e.vb), 0);
        enc.set_buffer(3, Some(e.attn), 0);
        enc.set_buffer(4, Some(&l.sel), 0);
        enc.set_buffer(5, Some(&l.bias), 0);
        set_bytes_t(enc, 6, &prm);
        enc.dispatch_thread_groups(MTLSize::new(e.nh as u64, e.k as u64, 1), MTLSize::new(256, 1, 1));
    })
    .is_some()
}

/// The chunk's rotation and mirror append, from the raw q/k/v rows.
pub(super) fn encode_rope(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    e: &DashEnc<'_>,
    qr: &Buffer,
    kr: &Buffer,
    vr: &Buffer,
) -> bool {
    let Some(p) = pipes(c) else { return false };
    let prm = params(e);
    with_layer(c, e, |_, sh| {
        enc.set_compute_pipeline_state(&p.rope);
        enc.set_buffer(0, Some(qr), 0);
        enc.set_buffer(1, Some(kr), 0);
        enc.set_buffer(2, Some(vr), 0);
        enc.set_buffer(3, Some(&sh.ang), 0);
        enc.set_buffer(4, Some(e.q), 0);
        enc.set_buffer(5, Some(e.kb), 0);
        enc.set_buffer(6, Some(e.vb), 0);
        set_bytes_t(enc, 7, &prm);
        enc.dispatch_threads(
            MTLSize::new((e.hd / 2) as u64, e.nh as u64, e.k as u64),
            MTLSize::new(32, 1, 1),
        );
    })
    .is_some()
}
