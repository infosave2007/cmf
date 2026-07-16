//! GPU path (D5, MVP): Metal on Apple Silicon.
//!
//! Architecture key: the CMF weights section is page-aligned in mmap → the GPU sees
//! THE SAME bytes via `newBufferWithBytesNoCopy` (unified memory), without
//! loading and without a second copy — cold weights stay cold.
//!
//! MVP scope: q8_row/q8_2f matvec for LARGE matrices (rows ≥ threshold —
//! in practice lm_head, the dominant decode matvec with a huge
//! vocabulary). Small matrices stay on the CPU: the dispatch cost (~50–100 µs)
//! eats the gain. Enable: `CMF_GPU=1`; any initialization failure —
//! an honest warning and CPU fallback (no silent accuracy degradations:
//! the kernel is mathematically identical to the CPU path, the same prescale trick).

use crate::gpu::{BatchJob, MoeJob};
use cortiq_core::quant::{Q1_TILE, GROUP_SIZE};
use cortiq_core::CmfModel;
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// y[o] = rs[o] * Σ_i q[o,i]·xs[i]; xs already prescaled by the col field (like CPU).
// SIMD group (32 lanes) per row: adjacent lanes read adjacent
// char4 → coalesced 128-byte reads; simd_sum reduction.
kernel void q8_matvec(
    device const char4*  q     [[buffer(0)]],
    device const float4* xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    constant uint&       cols4 [[buffer(4)]],
    constant uint&       rows  [[buffer(5)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tgpos * sgs + sg;
    if (row >= rows) return;
    ulong base = (ulong)row * cols4;
    float acc = 0.0f;
    for (uint i = lane; i < cols4; i += 32) {
        acc += dot(float4(q[base + i]), xs[i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc * rs[row];
}

// act[i] = silu(g[i])·u[i]·col[i] — down_proj input with the col field already
// applied (q8_2f prescale on the GPU, without returning to the CPU).
// GEMM prefill batch: y[bi, o] = rs[o]·Σ q[o,i]·xs[bi,i].
// SIMD group per (row, position); the row is hot in L2 across bi.
kernel void q8_matmat(
    device const char4*  q     [[buffer(0)]],
    device const float4* xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    constant uint&       cols4 [[buffer(4)]],
    constant uint&       rows  [[buffer(5)]],
    constant uint&       nb    [[buffer(6)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tg.x * sgs + sg;
    uint bi = tg.y;
    if (row >= rows || bi >= nb) return;
    ulong qb = (ulong)row * cols4;
    ulong xb = (ulong)bi * cols4;
    float acc = 0.0f;
    for (uint i = lane; i < cols4; i += 32) {
        acc += dot(float4(q[qb + i]), xs[xb + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(ulong)bi * rows + row] = acc * rs[row];
}

// q1: 6-byte tiles [f16 scale][4B sign bits] per 32-group; w = s*(2b-1).
// One SIMD group per row, each lane owns whole tiles; the scale lives
// inside the tile, so there is no separate row-scale buffer at all.
// One SIMD group per row, 8 rows per threadgroup; the activation is
// staged через threadgroup memory in 4096-float chunks — without it
// every row re-read the whole xs from L2 (rows × cols × 4B ≈ 350 MB of
// L2 traffic per FFN matvec, the measured wall). Tile PAIRS are 12
// bytes = three aligned u32 loads. gpr must be even (every real shape;
// the CPU handles odd-gpr tensors).
kernel void q1_matvec(
    device const uchar*  q    [[buffer(0)]],
    device const float4* xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       gpr  [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    threadgroup float4 shx[1024]; // 16 KB chunk of xs
    uint row = tgpos * sgs + sg;  // no early return: barriers below
    uint cols4 = gpr * 8u;        // cols / 4
    device const uint* qw =
        (device const uint*)(q + (ulong)min(row, rows - 1u) * gpr * 6u);
    float acc = 0.0f;
    uint nch = (cols4 + 1023u) / 1024u;
    for (uint ch = 0; ch < nch; ++ch) {
        uint cb4 = ch * 1024u;                    // chunk base in float4
        uint cl4 = min(1024u, cols4 - cb4);       // chunk length in float4
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < cl4; i += sgs * 32u) {
            shx[i] = xs[cb4 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < rows) {
            uint p0 = cb4 / 16u;                  // first tile PAIR of the chunk
            uint pn = cl4 / 16u;                  // pairs in the chunk
            for (uint pi = lane; pi < pn; pi += 32u) {
                uint pidx = p0 + pi;
                uint u0 = qw[pidx * 3u];
                uint u1 = qw[pidx * 3u + 1u];
                uint u2 = qw[pidx * 3u + 2u];
                float s0 = (float)as_type<half>((ushort)(u0 & 0xFFFFu));
                uint bits0 = (u0 >> 16) | (u1 << 16);
                float s1 = (float)as_type<half>((ushort)(u1 >> 16));
                uint bits1 = u2;
                uint xb = pi * 16u;               // pair base inside the chunk
                float4 sum0 = float4(0.0f);
                float4 sum1 = float4(0.0f);
                for (uint j = 0; j < 8; ++j) {
                    float4 x0 = shx[xb + j];
                    float4 x1 = shx[xb + 8u + j];
                    uint n0 = bits0 >> (j * 4u);
                    uint n1 = bits1 >> (j * 4u);
                    sum0 += select(-x0, x0, bool4(n0 & 1u, n0 & 2u, n0 & 4u, n0 & 8u));
                    sum1 += select(-x1, x1, bool4(n1 & 1u, n1 & 2u, n1 & 4u, n1 & 8u));
                }
                acc += s0 * (sum0.x + sum0.y + sum0.z + sum0.w)
                     + s1 * (sum1.x + sum1.y + sum1.z + sum1.w);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0 && row < rows) y[row] = acc;
}

kernel void silu_mul_pre(
    device const float* g   [[buffer(0)]],
    device const float* u   [[buffer(1)]],
    device const float* col [[buffer(2)]],
    device float*       act [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    constant uint&      has_col [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float gv = g[i];
    float cv = has_col != 0 ? col[i] : 1.0f;
    act[i] = (gv / (1.0f + exp(-gv))) * u[i] * cv;
}

kernel void axpy(
    device const float* d [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant float&     w [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] += w * d[i];
}

kernel void fill_zero(
    device float*  y [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] = 0.0f;
}

// Completion flag: the LAST encoder of every command buffer writes a
// monotone ticket into a shared buffer; the CPU spins on that word
// directly (UMA) instead of the driver's completion machinery, which
// costs ~1.3 ms per round trip. Reading every output buffer makes Metal
// order this pass after ALL producing passes (hazard tracking) —
// independent batch jobs may otherwise still be in flight when the
// flag lands. Unused slots are bound to y0.
kernel void write_flag(
    device const float* y0 [[buffer(0)]],
    device const float* y1 [[buffer(1)]],
    device const float* y2 [[buffer(2)]],
    device const float* y3 [[buffer(3)]],
    device atomic_uint* f  [[buffer(4)]],
    constant uint&      v  [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i == 0) {
        float probe = y0[0] + y1[0] + y2[0] + y3[0];
        uint bump = (probe == 123456789.0f) ? 1u : 0u; // never true: forces the reads
        atomic_store_explicit(f, v + bump, memory_order_relaxed);
    }
}
"#;

struct Ctx {
    _device: Device,
    queue: CommandQueue,
    q8: ComputePipelineState,
    q8mm: ComputePipelineState,
    q1: ComputePipelineState,
    flag: ComputePipelineState,
    silu: ComputePipelineState,
    axpy: ComputePipelineState,
    zero: ComputePipelineState,
    /// no-copy buffer per file (key — the base address of the mapping).
    file_bufs: Mutex<HashMap<usize, Buffer>>,
    /// row_scale buffer per tensor (key — (base, idx)).
    rs_bufs: Mutex<HashMap<(usize, usize), Buffer>>,
    /// Reusable xs/y buffers by size (no per-token allocations).
    io_bufs: Mutex<HashMap<usize, Buffer>>,
    /// Shared completion-flag word + monotone ticket (fast wait).
    flag_buf: Buffer,
    ticket: std::sync::atomic::AtomicU32,
}

// metal-rs objects — retained ObjC pointers; used under a Mutex
// or from a single decode thread.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

fn ctx() -> Option<&'static Ctx> {
    CTX.get_or_init(|| {
        if std::env::var("CMF_GPU").map(|v| v != "0").unwrap_or(false) {
            match init() {
                Ok(c) => {
                    tracing::info!("Metal GPU path: on ({})", c._device.name());
                    Some(c)
                }
                Err(e) => {
                    tracing::warn!("Metal init failed — CPU fallback: {e}");
                    None
                }
            }
        } else {
            None
        }
    })
    .as_ref()
}

fn init() -> Result<Ctx, String> {
    let device = Device::system_default().ok_or("no Metal device")?;
    // The zero-copy mmap buffers assume unified memory. On discrete-GPU
    // Macs (Intel-era) `newBufferWithBytesNoCopy` silently yields stale
    // data — measured max|Δ| ≈ 0.53 vs the f32 reference on a Radeon —
    // so refuse the device instead of returning wrong numbers.
    if !device.has_unified_memory() {
        return Err(format!(
            "device '{}' has no unified memory — no-copy mmap path needs UMA",
            device.name()
        ));
    }
    let lib = device
        .new_library_with_source(MSL, &metal::CompileOptions::new())
        .map_err(|e| format!("MSL compile: {e}"))?;
    let pso = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("kernel {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    let q8 = pso("q8_matvec")?;
    let q8mm = pso("q8_matmat")?;
    let q1 = pso("q1_matvec")?;
    let flag = pso("write_flag")?;
    let silu = pso("silu_mul_pre")?;
    let axpy = pso("axpy")?;
    let zero = pso("fill_zero")?;
    let queue = device.new_command_queue();
    let flag_buf = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
    unsafe { *(flag_buf.contents() as *mut u32) = 0 };
    Ok(Ctx {
        _device: device,
        queue,
        q8,
        q8mm,
        q1,
        flag,
        silu,
        axpy,
        zero,
        file_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
        io_bufs: Mutex::new(HashMap::new()),
        flag_buf,
        ticket: std::sync::atomic::AtomicU32::new(0),
    })
}

/// Is the GPU enabled and initialized?
pub fn enabled() -> bool {
    ctx().is_some()
}

/// Micro-bench hook: N empty command-buffer commit+wait round trips.
#[doc(hidden)]
pub fn empty_submit_bench(n: usize) -> f64 {
    let Some(c) = ctx() else { return f64::NAN };
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        wait_fast(cmd);
    }
    t0.elapsed().as_secs_f64()
}

/// Micro-bench hook: N empty command buffers committed back-to-back,
/// ONE wait at the end — separates pipeline latency from per-submit cost.
#[doc(hidden)]
pub fn pipelined_submit_bench(n: usize) -> f64 {
    let Some(c) = ctx() else { return f64::NAN };
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..n {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        last = Some(cmd.to_owned());
    }
    if let Some(cmd) = last {
        wait_fast(&cmd);
    }
    t0.elapsed().as_secs_f64()
}

/// Probe helper: weights are no-copy over the file mapping, so residency
/// is per FILE — true once the file buffer exists; otherwise create it
/// now (no dispatch, `may_upload` permitting) and report cold.
pub fn q8_resident_or_upload(model: &Arc<CmfModel>, _idx: usize, may_upload: bool) -> bool {
    let Some(c) = ctx() else { return false };
    let bytes = model.primary_bytes();
    if c.file_bufs.lock().unwrap().contains_key(&(bytes.as_ptr() as usize)) {
        return true;
    }
    if may_upload {
        let _ = file_buffer(c, bytes);
    }
    false
}

/// Commit with a fast completion path: append a flag-writing encoder
/// (ordered after `last_out` via a read hazard), commit, and spin on
/// the shared flag word — the driver's status/completion machinery
/// costs ~1.3 ms per round trip, the UMA flag lands in ~0.1 ms. Status
/// polling stays as the timeout fallback.
fn submit_and_wait(c: &Ctx, cmd: &metal::CommandBufferRef, outs: &[&Buffer]) {
    // NOTE: a "fast flag" variant (last encoder writes a ticket into a
    // shared buffer, CPU spins on the word) was tried here and REVERTED:
    // the flag becoming visible does not imply the earlier passes' output
    // lines have been written back — GPU cache write-back is not ordered
    // across buffers, and the readback raced (parity tests passed, the
    // real 27B decode corrupted). Only command-buffer completion gives
    // the system-scope guarantee, and its ~1.3 ms latency is exactly why
    // the road to 10+ tok/s is FEWER submissions per token, not faster
    // waits.
    let _ = (c, outs);
    cmd.commit();
    wait_fast(cmd);
}

/// Latency-critical wait: spin-poll the status instead of
/// waitUntilCompleted (sleeping/waking the thread costs ~1–3 ms —
/// across 40 MoE layers/token this canceled out the kernel's gain).
fn wait_fast(cmd: &metal::CommandBufferRef) {
    use metal::MTLCommandBufferStatus as S;
    let t0 = std::time::Instant::now();
    loop {
        match cmd.status() {
            S::Completed | S::Error => return,
            _ => {
                if t0.elapsed().as_millis() > 200 {
                    cmd.wait_until_completed(); // safeguard against an infinite spin
                    return;
                }
                std::hint::spin_loop();
            }
        }
    }
}

fn page_size() -> usize {
    // Apple Silicon: 16 KiB; taken from sysconf without a libc dependency.
    unsafe { getpagesize() as usize }
}

unsafe extern "C" {
    fn getpagesize() -> i32;
}

/// no-copy buffer over the file mapping (cached per file).
fn file_buffer(c: &Ctx, bytes: &[u8]) -> Option<(Buffer, usize)> {
    let base = bytes.as_ptr() as usize;
    let page = page_size();
    if base % page != 0 {
        return None; // mmap is always aligned, but we check honestly
    }
    let len = bytes.len() / page * page; // down to the page
    let mut cache = c.file_bufs.lock().unwrap();
    if let Some(b) = cache.get(&base) {
        return Some((b.clone(), len));
    }
    crate::gpu::probe_note_cold();
    let buf = c._device.new_buffer_with_bytes_no_copy(
        bytes.as_ptr() as *const std::ffi::c_void,
        len as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    cache.insert(base, buf.clone());
    Some((buf, len))
}

/// q8_row/q8_2f matvec on the GPU. `xs` — already prescaled activations (the same
/// math as the CPU path). false = could not (the caller falls back to CPU).
#[allow(clippy::too_many_arguments)]
pub fn q8_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q8_matvec_range(model, idx, 0, row_scale, xs, rows, cols, out)
}

/// Range variant (hybrid CPU∥GPU split): rows
/// [row0, row0+rows) of a large tensor.
#[allow(clippy::too_many_arguments)]
pub fn q8_matvec_range(
    model: &Arc<CmfModel>,
    idx: usize,
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(mut abs) = model.entry_abs_offset(entry) else {
        return false; // a neighboring shard — a different mapping; MVP: CPU
    };
    abs += row0 * cols; // offset into the sub-range (the GPU does not need 64-alignment)
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let qlen = rows * cols; // the int8 part of the blob (quants before scales)
    if abs + qlen > safe_len {
        return false; // the tail is past the buffer's page boundary
    }

    // row_scale — cached; xs/y — per call (small).
    let base = bytes.as_ptr() as usize;
    let rs_buf = {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx + row0 * 1_000_003))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    row_scale.as_ptr() as *const std::ffi::c_void,
                    (row_scale.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };
    let get_io = |nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(nbytes)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(
            xs.as_ptr(),
            xs_buf.contents() as *mut f32,
            xs.len(),
        );
    }
    let y_buf = get_io(rows * 4 + 4); // +4: does not share a key with xs of the same length

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.q8);
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&rs_buf), 0);
    enc.set_buffer(3, Some(&y_buf), 0);
    let cols4 = (cols / 4) as u32;
    let rows_u = rows as u32;
    enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    // 256 threads = 8 SIMD groups per threadgroup → 8 rows per group.
    let sgs = 8u64;
    let n_tg = (rows as u64).div_ceil(sgs);
    enc.dispatch_thread_groups(
        MTLSize::new(n_tg, 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32,
            out.as_mut_ptr(),
            rows,
        );
    }
    true
}

/// q1 matvec on the GPU: xs is the RAW f32 activation (the scale lives
/// inside the 6-byte tiles). GPU math is plain f32 — no A8 activation
/// quantization at all, so this path is if anything more accurate than
/// the CPU int8 kernel. false = CPU fallback.
pub fn q1_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    // The kernel stages xs through threadgroup memory in tile PAIRS —
    // odd group counts (unseen in real shapes) honestly stay on CPU.
    if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    if abs + rows * gpr * Q1_TILE > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(13_000_000_559 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(14_000_000_573 + rows, rows * 4);

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_q1_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// Encode one q1 matvec dispatch (shared by the single, batch and
/// MoE-chain paths).
#[allow(clippy::too_many_arguments)]
fn encode_q1_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q1);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// GEMM prefill batch: pre — prescaled inputs row-major [b, cols],
/// out — row-major [b, rows]. false = CPU path.
#[allow(clippy::too_many_arguments)]
pub fn q8_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    if abs + rows * cols > safe_len {
        return false;
    }
    let base = bytes.as_ptr() as usize;
    let rs_buf = {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    row_scale.as_ptr() as *const std::ffi::c_void,
                    (row_scale.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(11_000_000_453 + pre.len(), pre.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(pre.as_ptr(), xs_buf.contents() as *mut f32, pre.len());
    }
    let y_buf = get_io(12_000_000_469 + b * rows, b * rows * 4);

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.q8mm);
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&rs_buf), 0);
    enc.set_buffer(3, Some(&y_buf), 0);
    let cols4 = (cols / 4) as u32;
    let rows_u = rows as u32;
    let b_u = b as u32;
    enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &b_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs), b as u64, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    tracing::debug!("gpu matmat: {rows}x{cols} b={b}");
    true
}

/// Layer MoE-FFN in a single command buffer: for each selected expert
/// gate/up-matvec → silu·mul·prescale → down-matvec → axpy into y;
/// intermediate buffers are GPU-resident, one sync per layer. D5 design:
/// amortizing the dispatch cost over ~25 MB of work instead of a single matvec.
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() {
        return false;
    }
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let base = bytes.as_ptr() as usize;

    // Validate all tensors before encoding (fail → CPU without partial work).
    let mut abs3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let mut trio = [0usize; 3];
        for (slot, (idx, rows, cols, _)) in
            [(0, &j.gate), (1, &j.up), (2, &j.down)]
        {
            let entry = &model.tensors[*idx];
            let Some(abs) = model.entry_abs_offset(entry) else { return false };
            let qlen = if j.q1 {
                if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
                    return false;
                }
                rows * (cols / GROUP_SIZE) * Q1_TILE
            } else {
                if cols % 4 != 0 {
                    return false;
                }
                rows * cols
            };
            if abs + qlen > safe_len {
                return false;
            }
            trio[slot] = abs;
        }
        abs3.push(trio);
    }

    let inter = jobs[0].gate.1;
    let hidden = jobs[0].down.1;
    if out.len() != hidden {
        return false;
    }

    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    // Salted keys — sizes may coincide between assignments.
    let g_buf = get_io(1_000_000_007 + inter, inter * 4);
    let u_buf = get_io(2_000_000_011 + inter, inter * 4);
    let a_buf = get_io(3_000_000_019 + inter, inter * 4);
    let d_buf = get_io(4_000_000_021 + hidden, hidden * 4);
    let y_buf = get_io(5_000_000_033 + hidden, hidden * 4);

    let rs_or_col = |idx: usize, data: &[f32], salt: usize| -> Buffer {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base + salt, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };

    let cmd = c.queue.new_command_buffer();
    // Stage boundaries are ENCODER boundaries: Metal's automatic hazard
    // tracking fences tracked buffers between encoders, which on Apple
    // GPUs is far cheaper than memory_barrier_with_resources inside one
    // encoder (measured: the barrier variant cost ~2 ms extra per FFN
    // chain — more than all three matvecs together).
    let disp_elem = |enc: &metal::ComputeCommandEncoderRef,
                     pso: &ComputePipelineState,
                     n: usize| {
        enc.set_compute_pipeline_state(pso);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    };

    // y = 0
    let hid_u = hidden as u32;
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_buffer(0, Some(&y_buf), 0);
        enc.set_bytes(1, 4, &hid_u as *const u32 as *const std::ffi::c_void);
        disp_elem(enc, &c.zero, hidden);
        enc.end_encoding();
    }

    let matvec = |enc: &metal::ComputeCommandEncoderRef,
                  abs: usize, rows: usize, cols: usize, rs: Option<&Buffer>,
                  xs: &Buffer, y: &Buffer| {
        match rs {
            None => encode_q1_matvec(c, enc, &fbuf, abs, xs, y, rows, cols / GROUP_SIZE),
            Some(rs) => {
                enc.set_compute_pipeline_state(&c.q8);
                enc.set_buffer(0, Some(&fbuf), abs as u64);
                enc.set_buffer(1, Some(xs), 0);
                enc.set_buffer(2, Some(rs), 0);
                enc.set_buffer(3, Some(y), 0);
                let cols4 = (cols / 4) as u32;
                let rows_u = rows as u32;
                enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                let sgs = 8u64;
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(sgs), 1, 1),
                    MTLSize::new(sgs * 32, 1, 1),
                );
            }
        }
    };

    for (j, trio) in jobs.iter().zip(&abs3) {
        let (gi, grows, gcols, grs) = &j.gate;
        let (ui, urows, ucols, urs) = &j.up;
        let (di, drows, dcols, drs) = &j.down;
        // q1: scales live in the tiles — no rs buffers at all.
        let rs3 = if j.q1 {
            [None, None, None]
        } else {
            [
                Some(rs_or_col(*gi, grs, 0)),
                Some(rs_or_col(*ui, urs, 0)),
                Some(rs_or_col(*di, drs, 0)),
            ]
        };
        let has_col = !j.down_col.is_empty();
        let dcol_b = if has_col {
            rs_or_col(*di, j.down_col, 7_777_777)
        } else {
            g_buf.clone() // never read: silu has_col = 0
        };
        // gate/up xs — per call (small, via the size-keyed io cache).
        let xsg = get_io(6_000_000_087 + j.xs_gate.len(), j.xs_gate.len() * 4);
        let xsu = get_io(7_000_000_103 + j.xs_up.len(), j.xs_up.len() * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                j.xs_gate.as_ptr(), xsg.contents() as *mut f32, j.xs_gate.len());
            std::ptr::copy_nonoverlapping(
                j.xs_up.as_ptr(), xsu.contents() as *mut f32, j.xs_up.len());
        }

        {
            let enc = cmd.new_compute_command_encoder();
            matvec(enc, trio[0], *grows, *gcols, rs3[0].as_ref(), &xsg, &g_buf);
            matvec(enc, trio[1], *urows, *ucols, rs3[1].as_ref(), &xsu, &u_buf);
            enc.end_encoding();
        }
        {
            // act = silu(g)·u·col_down (col skipped when the job has none)
            let enc = cmd.new_compute_command_encoder();
            enc.set_buffer(0, Some(&g_buf), 0);
            enc.set_buffer(1, Some(&u_buf), 0);
            enc.set_buffer(2, Some(&dcol_b), 0);
            enc.set_buffer(3, Some(&a_buf), 0);
            let n_u = inter as u32;
            let hc_u = has_col as u32;
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hc_u as *const u32 as *const std::ffi::c_void);
            disp_elem(enc, &c.silu, inter);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            matvec(enc, trio[2], *drows, *dcols, rs3[2].as_ref(), &a_buf, &d_buf);
            enc.end_encoding();
        }
        {
            // y += w · d
            let enc = cmd.new_compute_command_encoder();
            enc.set_buffer(0, Some(&d_buf), 0);
            enc.set_buffer(1, Some(&y_buf), 0);
            enc.set_bytes(2, 4, &j.w as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &hid_u as *const u32 as *const std::ffi::c_void);
            disp_elem(enc, &c.axpy, hidden);
            enc.end_encoding();
        }
    }
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32, out.as_mut_ptr(), hidden);
    }
    true
}

/// Several independent q8-matvec in a single command buffer (one sync).
/// outs[i].len() == jobs[i].rows.
pub fn matvec_batch(
    model: &Arc<CmfModel>,
    jobs: &[BatchJob],
    outs: &mut [&mut [f32]],
) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() || jobs.len() != outs.len() {
        return false;
    }
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let base = bytes.as_ptr() as usize;

    let mut abss = Vec::with_capacity(jobs.len());
    for j in jobs {
        let entry = &model.tensors[j.idx];
        let Some(abs) = model.entry_abs_offset(entry) else { return false };
        let qlen = if j.q1 {
            if j.cols % GROUP_SIZE != 0 || (j.cols / GROUP_SIZE) % 2 != 0 {
                return false;
            }
            j.rows * (j.cols / GROUP_SIZE) * Q1_TILE
        } else {
            if j.cols % 4 != 0 {
                return false;
            }
            j.rows * j.cols
        };
        if abs + qlen > safe_len {
            return false;
        }
        abss.push(abs);
    }

    // Buffers: y per job (by size, via the io cache with a position salt),
    // xs per job, rs cached per-tensor.
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let rs_of = |idx: usize, data: &[f32]| -> Buffer {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };

    let mut y_bufs = Vec::with_capacity(jobs.len());
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for (slot, (j, abs)) in jobs.iter().zip(&abss).enumerate() {
        let xs_b = get_io(
            8_000_000_209 + slot * 131 + j.xs.len(),
            j.xs.len() * 4,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                j.xs.as_ptr(), xs_b.contents() as *mut f32, j.xs.len());
        }
        let y_b = get_io(9_000_000_341 + slot * 137 + j.rows, j.rows * 4);
        if j.q1 {
            encode_q1_matvec(c, enc, &fbuf, *abs, &xs_b, &y_b, j.rows, j.cols / GROUP_SIZE);
        } else {
            let rs_b = rs_of(j.idx, j.row_scale);
            enc.set_compute_pipeline_state(&c.q8);
            enc.set_buffer(0, Some(&fbuf), *abs as u64);
            enc.set_buffer(1, Some(&xs_b), 0);
            enc.set_buffer(2, Some(&rs_b), 0);
            enc.set_buffer(3, Some(&y_b), 0);
            let cols4 = (j.cols / 4) as u32;
            let rows_u = j.rows as u32;
            enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            let sgs = 8u64;
            enc.dispatch_thread_groups(
                MTLSize::new((j.rows as u64).div_ceil(sgs), 1, 1),
                MTLSize::new(sgs * 32, 1, 1),
            );
        }
        y_bufs.push(y_b);
    }
    enc.end_encoding();
    if y_bufs.len() <= 4 {
        let refs: Vec<&Buffer> = y_bufs.iter().collect();
        submit_and_wait(c, cmd, &refs);
    } else {
        cmd.commit();
        wait_fast(cmd);
    }

    for ((y_b, j), out) in y_bufs.iter().zip(jobs).zip(outs.iter_mut()) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                y_b.contents() as *const f32, out.as_mut_ptr(), j.rows);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::{
        CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype,
        TensorSpec, CMF_VERSION,
    };
    use crate::qtensor::QTensor;

    /// GPU kernel == CPU path on an lm_head-class q8_row tensor over
    /// a REAL mmap (no-copy buffer). Skipped without a Metal device.
    #[test]
    fn gpu_q8_matvec_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "1") };
        if !enabled() {
            eprintln!("gpu test skipped: no Metal device");
            return;
        }
        let (rows, cols) = (crate::gpu::GPU_MIN_ROWS, 64);
        // Reference q8_row encoder (like tests/roundtrip.rs).
        let mut w = vec![0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 197) as f32 / 197.0 - 0.5) * 0.3;
        }
        let mut q = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(rows * 2);
        for o in 0..rows {
            let row = &w[o * cols..(o + 1) * cols];
            let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if absmax == 0.0 { 1e-10 } else { absmax / 127.0 };
            let scale = {
                let h = cortiq_core::quant::f32_to_f16(scale);
                cortiq_core::quant::f16_to_f32(h)
            };
            for &v in row {
                q.push((v / scale).round().clamp(-128.0, 127.0) as i8 as u8);
            }
            scales.extend_from_slice(
                &cortiq_core::quant::f32_to_f16(scale).to_le_bytes());
        }
        q.extend_from_slice(&scales);

        let arch = ModelArch {
            arch_name: "tiny".into(),
            hidden_size: cols,
            intermediate_size: cols * 2,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: rows,
            layer_types: vec![LayerType::FullAttention],
            rms_norm_eps: 1e-6,
            norm_style: NormStyle::Qwen,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            partial_rotary_factor: 1.0,
            mtp: None,
            moe: None,
            linear_core: None,
            max_position_embeddings: 8,
            linear_conv_kernel_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Q8Row,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
        };
        let spec = TensorSpec {
            name: "lm_head.weight".into(),
            dtype: TensorDtype::Q8Row,
            shape: vec![rows, cols],
            data: q,
        };
        let dir = std::env::temp_dir().join(format!("cmf-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu.cmf");
        CmfModel::write(&path, &header, &[spec], None, None).unwrap();
        let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
        let t = QTensor::from_model(&model, "lm_head.weight").unwrap();

        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 3) % 89) as f32 / 89.0 - 0.5)
            .collect();
        let mut cpu = vec![0f32; rows];
        // CPU reference: matvec with the GPU disabled is impossible via env
        // (OnceLock) — compute manually from the source weights.
        for o in 0..rows {
            let mut acc = 0f32;
            for i in 0..cols {
                acc += w[o * cols + i] * x[i];
            }
            cpu[o] = acc;
        }
        let mut gpu = vec![0f32; rows];
        t.matvec(&x, &mut gpu, None); // rows ≥ threshold → GPU path
        let mut max_d = 0f32;
        for o in 0..rows {
            max_d = max_d.max((cpu[o] - gpu[o]).abs());
        }
        // q8 grid tolerance: |w|≤0.15, step ≈ absmax/127, dot over 64.
        assert!(max_d < 2e-2, "GPU vs f32 reference: max|Δ| = {max_d}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GPU q1 kernel == exact f32 reference over a real mmap. The GPU
    /// math is plain f32 (no A8 quantization), so the tolerance is pure
    /// float-summation noise. Skipped without a Metal device.
    #[test]
    fn gpu_q1_matvec_matches_reference() {
        // Two shapes: single-chunk (cols ≤ 4096) and the CHUNKED path
        // (cols 6144 → two threadgroup-memory chunks — the out_proj
        // shape that a small parity test would never touch).
        gpu_q1_case(512, 256);
        gpu_q1_case(256, 6144);
    }

    fn gpu_q1_case(rows: usize, cols: usize) {
        unsafe { std::env::set_var("CMF_GPU", "1") };
        if !enabled() {
            eprintln!("gpu test skipped: no Metal device");
            return;
        }
        let gpr = cols / GROUP_SIZE;
        // Binary weights ±s per group, packed as q1 tiles.
        let mut payload = Vec::with_capacity(rows * gpr * Q1_TILE);
        let mut w = vec![0f32; rows * cols];
        for o in 0..rows {
            for g in 0..gpr {
                let s = 0.004 + ((o * 7 + g) % 11) as f32 * 0.002;
                let s = cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(s));
                payload.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
                for j in 0..4 {
                    let mut byte = 0u8;
                    for k in 0..8 {
                        let i = g * GROUP_SIZE + j * 8 + k;
                        let bit = ((o * 37 + i * 13) % 5) < 2;
                        if bit {
                            byte |= 1 << k;
                        }
                        w[o * cols + i] = if bit { s } else { -s };
                    }
                    payload.push(byte);
                }
            }
        }
        let arch = ModelArch {
            arch_name: "tiny".into(),
            hidden_size: cols,
            intermediate_size: cols * 2,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: rows,
            layer_types: vec![LayerType::FullAttention],
            rms_norm_eps: 1e-6,
            norm_style: NormStyle::Qwen,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            partial_rotary_factor: 1.0,
            mtp: None,
            moe: None,
            linear_core: None,
            max_position_embeddings: 8,
            linear_conv_kernel_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Vbit,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
        };
        let spec = TensorSpec {
            name: "lm_head.weight".into(),
            dtype: TensorDtype::Q1,
            shape: vec![rows, cols],
            data: payload,
        };
        // The no-copy buffer is truncated to the last FULL page; a q1
        // payload has no trailing scales section, so pad the file past
        // the page boundary with a dummy tensor (in a real model some
        // other tensor plays this role; only the file's very last q1
        // tensor honestly falls back to CPU).
        let pad = TensorSpec {
            name: "pad.weight".into(),
            dtype: TensorDtype::F32,
            shape: vec![4096, 2],
            data: vec![0u8; 4096 * 2 * 4],
        };
        let dir = std::env::temp_dir().join(format!("cmf-gpu-q1-{}-{rows}x{cols}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu.cmf");
        CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
        let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
        let idx = model.tensor_index("lm_head.weight").unwrap();

        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 17 + 5) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut cpu = vec![0f32; rows];
        for o in 0..rows {
            cpu[o] = (0..cols).map(|i| w[o * cols + i] * x[i]).sum();
        }
        let mut gpu = vec![0f32; rows];
        assert!(
            q1_matvec(&model, idx, &x, rows, cols, &mut gpu),
            "metal q1_matvec refused"
        );
        let mut max_d = 0f32;
        for o in 0..rows {
            max_d = max_d.max((cpu[o] - gpu[o]).abs());
        }
        assert!(max_d < 1e-4, "GPU q1 vs f32 reference: max|Δ| = {max_d}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
