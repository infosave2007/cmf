//! Gemma-4 (the humanizer 12B) on the Metal token graph and chunk prefill.
//!
//! What the graphs did not know of Gemma-4 — and what kept every layer of
//! it on the CPU attend (4.5 tok/s on an M4, the CPU's own rate):
//! - tanh-GELU between gate and up (`gelu_tanh_mul_pre`, and the fused
//!   q4tp gate|up pass `q4tp_matvec_m_gu_gelut`);
//! - the scale-less V norm before V is cached (`vnorm_rows`);
//! - the sandwich norms around both mixers and the per-layer scalar
//!   (`norm_add_scale_rows`: h = (h + norm(d))·sc; the attention side
//!   norms O·ao in place with the main library's `rmsnorm_k`);
//! - the 512-wide global layers (1 KV head): the `_w16` attention kernels,
//!   the main library's kernels with 16 register slots a lane.
//!
//! The kernels live in their own library (`gemma4_msl.metal`), compiled the
//! first time a Gemma-4 layer asks for them. `CMF_METAL_GEMMA4=0` keeps
//! Gemma-4 off the Metal graphs (the 0.8.15 behaviour: CPU attention);
//! `CMF_GEMMA4_MSL=<file>` swaps the source without a rebuild.

use super::{Ctx, WeightArena};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;
use std::sync::OnceLock;

const GMSL: &str = include_str!("gemma4_msl.metal");

pub(super) struct Pipes {
    pub(super) gelut: ComputePipelineState,
    pub(super) mgu_gelut: ComputePipelineState,
    pub(super) vnorm: ComputePipelineState,
    pub(super) nas: ComputePipelineState,
    pub(super) rqkn16: ComputePipelineState,
    pub(super) cropekv16: ComputePipelineState,
    pub(super) gqat16: ComputePipelineState,
    pub(super) gqablk16: ComputePipelineState,
}

/// Whether Gemma-4 may take the Metal graphs at all (`CMF_METAL_GEMMA4=0`
/// keeps it on the CPU attention path).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_METAL_GEMMA4").as_deref() != Ok("0"))
}

/// `CMF_METAL_GEMMA4_W16` (default on, `=0` off): the 512-wide global
/// layers attend on the device too (`_w16` kernels); off, they take the
/// graph's sandwich (host attend between QKV and O, one sync each).
pub(crate) fn w16_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_METAL_GEMMA4_W16").as_deref() != Ok("0"))
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

/// The compiled kernels, or None (said once) when the library failed.
pub(super) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let src = match std::env::var("CMF_GEMMA4_MSL") {
            Ok(path) => std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?,
            Err(_) => GMSL.to_string(),
        };
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(&src, &opts)
            .map_err(|e| format!("gemma4 MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            gelut: pso("gelu_tanh_mul_pre")?,
            mgu_gelut: pso("q4tp_matvec_m_gu_gelut")?,
            vnorm: pso("vnorm_rows")?,
            nas: pso("norm_add_scale_rows")?,
            rqkn16: pso("attn_rope_qkn_w16")?,
            cropekv16: pso("chunk_rope_kv_w16")?,
            gqat16: pso("gqa_attend_w16")?,
            gqablk16: pso("gqa_attend_blk_w16")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("Gemma-4 Metal kernels unavailable ({e}) — the model stays off the graphs");
            }
            None
        }
    }
}

fn set_u32(enc: &ComputeCommandEncoderRef, slot: u64, v: u32) {
    enc.set_bytes(slot, 4, &v as *const u32 as *const c_void);
}

fn set_f32(enc: &ComputeCommandEncoderRef, slot: u64, v: f32) {
    enc.set_bytes(slot, 4, &v as *const f32 as *const c_void);
}

/// Scale-less RMS norm of the `nkv` V heads of `nb` rows (`[nb][nkv·hd]`).
pub(super) fn enc_vnorm(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    v: &Buffer,
    nkv: usize,
    hd: usize,
    nb: usize,
    eps: f32,
) {
    enc.set_compute_pipeline_state(&p.vnorm);
    enc.set_buffer(0, Some(v), 0);
    set_u32(enc, 1, nkv as u32);
    set_u32(enc, 2, hd as u32);
    set_f32(enc, 3, eps);
    enc.dispatch_thread_groups(
        MTLSize::new(nkv as u64, nb as u64, 1),
        MTLSize::new(32, 1, 1),
    );
}

/// h = (h + norm(d))·sc over `nb` rows of `n` (norm = rms(d)·w when `w`).
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_norm_add_scale(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    h: &Buffer,
    d: &Buffer,
    w: Option<&Buffer>,
    n: usize,
    gemma: bool,
    eps: f32,
    sc: f32,
    nb: usize,
) {
    enc.set_compute_pipeline_state(&p.nas);
    enc.set_buffer(0, Some(h), 0);
    enc.set_buffer(1, Some(d), 0);
    enc.set_buffer(2, Some(w.unwrap_or(d)), 0);
    set_u32(enc, 3, n as u32);
    set_u32(enc, 4, gemma as u32);
    set_f32(enc, 5, eps);
    set_u32(enc, 6, w.is_some() as u32);
    set_f32(enc, 7, sc);
    enc.dispatch_thread_groups(MTLSize::new(nb as u64, 1, 1), MTLSize::new(256, 1, 1));
}

/// act = gelu_tanh(g)·u over `n` elements (`act` may alias `g`).
pub(super) fn enc_gelut(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    g: &Buffer,
    u: &Buffer,
    act: &Buffer,
    n: usize,
) {
    enc.set_compute_pipeline_state(&p.gelut);
    enc.set_buffer(0, Some(g), 0);
    enc.set_buffer(1, Some(u), 0);
    enc.set_buffer(2, Some(g), 0); // dummy col (has_col = 0)
    enc.set_buffer(3, Some(act), 0);
    set_u32(enc, 4, n as u32);
    set_u32(enc, 5, 0);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
}

/// `encode_q4tp_matvec_m_gu` with the tanh GELU (same dispatch shape).
#[allow(clippy::too_many_arguments)]
pub(super) fn enc_mgu_gelut(
    p: &Pipes,
    enc: &ComputeCommandEncoderRef,
    fbuf: &WeightArena,
    abs_g: usize,
    abs_u: usize,
    xs: &Buffer,
    act: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&p.mgu_gelut);
    fbuf.bind(enc, 0, abs_g);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(act), 0);
    set_u32(enc, 3, gpr as u32);
    set_u32(enc, 4, rows as u32);
    fbuf.bind(enc, 5, abs_u);
    let sgs = 4u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}
