//! q8_2f routed experts on the Metal token graph and the MoE chunk prefill
//! (Mellum2.1 q8_2f: every expert, attention and the head in q8_2f).
//!
//! The graphs' MoE ladder addressed q4tp/q2tp expert payloads only, so a
//! q8_2f expert ended the decode plan at layer 0 and the chunk prefill's run
//! before it began: the file decoded through per-op Metal at 11 tok/s, below
//! its own CPU path (28). These kernels take the same job tables (the select
//! kernel's absolute bases, the chunk's packed expert panels) and read each
//! expert's row scales and input field straight from the blob behind its
//! int8 payload — nothing is staged per expert, so no job can read another
//! job's x·col (the 0.8.14 per-op bug, `q8_moe_jobs_keep_their_own_inputs`).
//!
//! The kernels live in their own library, compiled the first time a q8_2f
//! MoE layer asks for them: no other model pays for the compile.
//! `CMF_MOE_Q8_MSL=<file>` swaps the source without a rebuild (kernel work).

use super::{Ctx, WeightArena};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;
use std::sync::OnceLock;

const QMSL: &str = include_str!("moe_q8_msl.metal");

pub(super) struct Pipes {
    jobs: ComputePipelineState,
    gu: ComputePipelineState,
    down: ComputePipelineState,
    act: ComputePipelineState,
    mm: ComputePipelineState,
    att_nt: ComputePipelineState,
    att_nn: ComputePipelineState,
}

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key).as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => default,
    }
}

/// `CMF_MOE_Q8_PROJ` (default on, `=0` off): a q8_2f MoE layer's q8_2f
/// attention projections in the chunk prefill through `q8f_mul_mm_blob`
/// (f32 tiles) instead of `col_scale_rows` + the half-tile `q8_mul_mm`.
///
/// With `CMF_MOE_Q8_ATT` this is what holds the q8_2f file's Metal
/// perplexity on its CPU reference (Mellum2.1, 2048 tokens, CPU wiki 7.180 /
/// code 3.146): half tiles everywhere 7.206 / 3.152, f32 projections only
/// 7.178 / 3.153, f32 attention only 7.199 / 3.135, both 7.180 / 3.146 —
/// for ~10 % of the chunk's GPU time (M4, 3000-token prompt).
pub(super) fn proj_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("CMF_MOE_Q8_PROJ", true))
}

/// `CMF_MOE_Q8_ATT` (default on, `=0` off): a q8_2f MoE layer's chunk
/// attention GEMMs (scores, P·V) with f32 operand tiles (`att_mm_nt_f32` /
/// `att_mm_nn_f32`); see `proj_on`.
pub(super) fn att_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("CMF_MOE_Q8_ATT", true))
}

pub(super) fn set_att_nt(p: &Pipes, enc: &ComputeCommandEncoderRef) {
    enc.set_compute_pipeline_state(&p.att_nt);
}

pub(super) fn set_att_nn(p: &Pipes, enc: &ComputeCommandEncoderRef) {
    enc.set_compute_pipeline_state(&p.att_nn);
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

/// Whether q8_2f experts may take the graphs at all (`CMF_METAL_MOE_Q8=0`
/// keeps them on the per-op path / host walk, the 0.8.15 behaviour).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_METAL_MOE_Q8").as_deref() != Ok("0"))
}

/// `CMF_MOE_Q8_DN=1`: the decode graph's q8_2f down, mix and residual in
/// one pass (`q8f_moe_down_r4`) instead of jobs + reduce + axpy (M4: 1.044
/// of the jobs ladder's wall per token — off by default).
pub(super) fn down_fused() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("CMF_MOE_Q8_DN", false))
}

/// `CMF_MOE_Q8_GU=1`: the decode graph's q8_2f gate|up|SiLU in one pass
/// (`q8f_moe_gu_r4`, under the plan's `MOE_GU` lever) instead of the 2·k
/// jobs matvec + `moe_silu_jobs` (M4: 1.021 of the jobs pair — off by
/// default; `CMF_METAL_AB` arms with and without `g` compare the two).
pub(super) fn gu_fused() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("CMF_MOE_Q8_GU", false))
}

/// The compiled kernels, or None (said once) when the library failed.
pub(super) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let src = match std::env::var("CMF_MOE_Q8_MSL") {
            Ok(path) => std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?,
            Err(_) => QMSL.to_string(),
        };
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(&src, &opts)
            .map_err(|e| format!("moe_q8 MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            jobs: pso("q8f_jobs_r4")?,
            gu: pso("q8f_moe_gu_r4")?,
            down: pso("q8f_moe_down_r4")?,
            act: pso("moe_act_rows_col")?,
            mm: pso("q8f_mul_mm_blob")?,
            att_nt: pso("att_mm_nt_f32")?,
            att_nn: pso("att_mm_nn_f32")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("q8_2f MoE kernels unavailable ({e}) — experts stay off the graphs");
            }
            None
        }
    }
}

/// The byte size a q8_2f `[rows, cols]` tensor occupies in the blob
/// (payload, row scales, input field) when the kernels' layout holds:
/// `cols % 4 == 0` for the char4 rows; 64-byte tensor alignment covers the
/// rest.
pub(crate) fn tensor_bytes(rows: usize, cols: usize) -> Option<usize> {
    if cols % 4 != 0 || rows == 0 {
        return None;
    }
    rows.checked_mul(cols)?
        .checked_add(rows.checked_mul(2)?)?
        .checked_add(cols.checked_mul(2)?)
}

fn set_u32(enc: &ComputeCommandEncoderRef, slot: u64, v: u32) {
    enc.set_bytes(slot, 4, &v as *const u32 as *const c_void);
}

/// Jobs matvec over the select kernel's (or the host's) bases: `njob`
/// jobs of `rows × cols`, job j reading `x + j·xstride` floats.
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_jobs(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    bases: &Buffer,
    x: &Buffer,
    y: &Buffer,
    rows: usize,
    cols: usize,
    njob: usize,
    xstride: usize,
) {
    let sgs = 8u64;
    let tg_per = (rows as u64).div_ceil(sgs * 4);
    enc.set_compute_pipeline_state(&p.jobs);
    enc.set_buffer(0, Some(fbuf.window(0)), 0);
    fbuf.bind_jobs_windows(enc);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(y), 0);
    set_u32(enc, 3, (cols / 4) as u32);
    set_u32(enc, 4, rows as u32);
    enc.set_buffer(5, Some(bases), 0);
    set_u32(enc, 6, tg_per as u32);
    set_u32(enc, 7, (xstride / 4) as u32);
    enc.dispatch_thread_groups(
        MTLSize::new(tg_per * njob as u64, 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// gate|up|SiLU for the `ne` jobs in one pass (`bases`: gates, then ups).
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_gu(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    bases: &Buffer,
    x: &Buffer,
    act: &Buffer,
    inter: usize,
    hidden: usize,
    ne: usize,
) {
    let sgs = 8u64;
    let tg_per = (inter as u64).div_ceil(sgs * 4);
    enc.set_compute_pipeline_state(&p.gu);
    enc.set_buffer(0, Some(fbuf.window(0)), 0);
    fbuf.bind_jobs_windows(enc);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(act), 0);
    set_u32(enc, 3, (hidden / 4) as u32);
    set_u32(enc, 4, inter as u32);
    enc.set_buffer(5, Some(bases), 0);
    set_u32(enc, 6, tg_per as u32);
    set_u32(enc, 7, ne as u32);
    enc.dispatch_thread_groups(
        MTLSize::new(tg_per * ne as u64, 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// down + weighted mix + residual (`h += Σ w_e·down_e(a_e)`) in one pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_down_mix(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    bases: &Buffer,
    a: &Buffer,
    h: &Buffer,
    wmix: &Buffer,
    hidden: usize,
    inter: usize,
    ne: usize,
) {
    let sgs = 8u64;
    enc.set_compute_pipeline_state(&p.down);
    enc.set_buffer(0, Some(fbuf.window(0)), 0);
    fbuf.bind_jobs_windows(enc);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(h), 0);
    set_u32(enc, 3, (inter / 4) as u32);
    set_u32(enc, 4, hidden as u32);
    enc.set_buffer(5, Some(bases), 0);
    enc.set_buffer(6, Some(wmix), 0);
    set_u32(enc, 7, ne as u32);
    enc.dispatch_thread_groups(
        MTLSize::new((hidden as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// One q8_2f GEMM of the MoE chunk stage on a row slice (`q8f_mul_mm_blob`):
/// `nb` packed rows of `xs` from row `x_row` into `y` from row `y_row`; with
/// `xcol` the weight's input field multiplies X as it is staged.
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_mm_rows(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    abs: usize,
    xs: &Buffer,
    x_row: usize,
    y: &Buffer,
    y_row: usize,
    nb: usize,
    rows: usize,
    cols: usize,
    xcol: bool,
) {
    enc.set_compute_pipeline_state(&p.mm);
    fbuf.bind(enc, 0, abs);
    enc.set_buffer(1, Some(xs), (x_row * cols * 4) as u64);
    enc.set_buffer(2, Some(y), (y_row * rows * 4) as u64);
    set_u32(enc, 3, cols as u32);
    set_u32(enc, 4, rows as u32);
    set_u32(enc, 5, nb as u32);
    set_u32(enc, 6, xcol as u32);
    enc.dispatch_thread_groups(
        MTLSize::new((nb as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
        MTLSize::new(128, 1, 1),
    );
}

/// SiLU·up·(down's input field) for `n_e` packed rows of one expert panel
/// from row `row0`, with the half guard (`moe_act_rows_col`).
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_act_rows(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    down_abs: usize,
    g: &Buffer,
    u: &Buffer,
    a: &Buffer,
    up: &Buffer,
    row0: usize,
    n_e: usize,
    inter: usize,
    hidden: usize,
) {
    let off = (row0 * inter * 4) as u64;
    enc.set_compute_pipeline_state(&p.act);
    enc.set_buffer(0, Some(g), off);
    enc.set_buffer(1, Some(u), off);
    enc.set_buffer(2, Some(a), off);
    enc.set_buffer(3, Some(up), (row0 * 4) as u64);
    set_u32(enc, 4, inter as u32);
    fbuf.bind(enc, 5, down_abs);
    // down is [hidden, inter]: payload hidden·inter, row scales hidden·2.
    set_u32(enc, 6, (hidden * inter + hidden * 2) as u32);
    enc.dispatch_thread_groups(MTLSize::new(n_e as u64, 1, 1), MTLSize::new(256, 1, 1));
}

fn shared_buf(c: &Ctx, bytes: &[u8]) -> Buffer {
    c._device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len().max(4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(b: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
}

/// Test hook: one token through the decode graph's q8_2f expert stages with
/// the jobs given (no router): returns `Σ_j w_j · down_j(silu(g_j)·u_j)`,
/// each projection reading x through its own weight's input field. `trios`
/// are directory indices of q8_2f `[inter, hidden]` gate/up and
/// `[hidden, inter]` down tensors.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn decode_block_for_test(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    trios: &[(usize, usize, usize)],
    x: &[f32],
    w: &[f32],
    hidden: usize,
    inter: usize,
    fused_gu: bool,
    fused_dn: bool,
) -> Option<Vec<f32>> {
    let c = super::ctx()?;
    let (fbuf, _) = super::file_buffer(c, model)?;
    let p = pipes(c)?;
    let ne = trios.len();
    if w.len() != ne || x.len() != hidden {
        return None;
    }
    let abs = |i: usize| model.entry_abs_offset(&model.tensors[i]).map(|a| a as u64);
    let mut gu = vec![0u64; 2 * ne];
    let mut dn = vec![0u64; ne];
    for (j, &(g, u, d)) in trios.iter().enumerate() {
        gu[j] = abs(g)?;
        gu[ne + j] = abs(u)?;
        dn[j] = abs(d)?;
    }
    let as_bytes = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let bgu = shared_buf(c, &as_bytes(&gu));
    let bdn = shared_buf(c, &as_bytes(&dn));
    let xb = shared_buf(c, &f32_bytes(x));
    let wb = shared_buf(c, &f32_bytes(w));
    let gub = shared_buf(c, &vec![0u8; 2 * ne * inter * 4]);
    let ab = shared_buf(c, &vec![0u8; ne * inter * 4]);
    let eob = shared_buf(c, &vec![0u8; ne * hidden * 4]);
    let db = shared_buf(c, &vec![0u8; hidden * 4]);
    let hb = shared_buf(c, &vec![0u8; hidden * 4]);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let u32b = |enc: &ComputeCommandEncoderRef, slot: u64, v: usize| set_u32(enc, slot, v as u32);
    if fused_gu {
        enc_gu(p, enc, &fbuf, &bgu, &xb, &ab, inter, hidden, ne);
    } else {
        enc_jobs(p, enc, &fbuf, &bgu, &xb, &gub, inter, hidden, 2 * ne, 0);
        enc.set_compute_pipeline_state(&c.moesilu);
        enc.set_buffer(0, Some(&gub), 0);
        enc.set_buffer(1, Some(&ab), 0);
        u32b(enc, 2, inter);
        u32b(enc, 3, ne);
        enc.dispatch_threads(
            MTLSize::new((ne * inter) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    if fused_dn {
        enc_down_mix(p, enc, &fbuf, &bdn, &ab, &hb, &wb, hidden, inter, ne);
    } else {
        enc_jobs(p, enc, &fbuf, &bdn, &ab, &eob, hidden, inter, ne, inter);
        enc.set_compute_pipeline_state(&c.moered);
        enc.set_buffer(0, Some(&eob), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&db), 0);
        u32b(enc, 3, hidden);
        u32b(enc, 4, ne);
        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
        super::disp_axpy(c, enc, &db, &hb, 1.0, hidden);
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    (cmd.status() != metal::MTLCommandBufferStatus::Error).then(|| read_f32(&hb, hidden))
}

/// Test hook: the MoE chunk stage (`chunk_moe_ffn`) on q8_2f trios for `b`
/// rows of `xs`: router logits on the device, `route` on the host (top-k
/// ids and mixing weights per row), panels through `q8f_mul_mm_blob`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn chunk_block_for_test(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    router: &[f32],
    trios: &[(usize, usize, usize)],
    route: Box<dyn Fn(&[f32]) -> (Vec<usize>, Vec<f32>) + '_>,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
) -> Option<Vec<f32>> {
    let c = super::ctx()?;
    let (fbuf, _) = super::file_buffer(c, model)?;
    pipes(c)?;
    if xs.len() != b * hidden {
        return None;
    }
    let mabs = trios
        .iter()
        .map(|&(g, u, d)| {
            let a = |i: usize| model.entry_abs_offset(&model.tensors[i]);
            Some((a(g)?, a(u)?, a(d)?))
        })
        .collect::<Option<Vec<_>>>()?;
    let m = super::ChunkMoe {
        router,
        experts: trios.to_vec(),
        q8: true,
        route,
    };
    let nb = shared_buf(c, &f32_bytes(xs));
    let db = shared_buf(c, &vec![0u8; b * hidden * 4]);
    let mut cmd = c.queue.new_command_buffer().to_owned();
    if !super::chunk_moe_ffn(c, &fbuf, &mut cmd, &m, &mabs, &nb, &db, b, hidden, inter) {
        return None;
    }
    cmd.commit();
    cmd.wait_until_completed();
    (cmd.status() != metal::MTLCommandBufferStatus::Error).then(|| read_f32(&db, b * hidden))
}
