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
// One SIMD group per TWO rows: each activation float4 a lane loads is
// used against both rows' tile pairs, with no threadgroup staging (no
// barriers, xs hot in L1 across a core's simdgroups). Four rows per
// simdgroup was tried and REVERTED: the cached-x register block spills
// and occupancy drops (13.8 ms/block vs 8.8). Tile pairs are 12 bytes =
// three aligned u32 loads; gpr must be even (CPU handles the rest).
kernel void q1_matvec(
    device const uchar*  q    [[buffer(0)]],
    device const float4* xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       gpr  [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 2u;
    if (r0 >= rows) return;
    bool two = r0 + 1u < rows;
    uint np = gpr >> 1;
    device const uint* q0 = (device const uint*)(q + (ulong)r0 * gpr * 6u);
    device const uint* q1p = (device const uint*)(q + (ulong)(r0 + (two ? 1u : 0u)) * gpr * 6u);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint pidx = lane; pidx < np; pidx += 32u) {
        uint a0 = q0[pidx * 3u];
        uint a1 = q0[pidx * 3u + 1u];
        uint a2 = q0[pidx * 3u + 2u];
        uint b0 = q1p[pidx * 3u];
        uint b1 = q1p[pidx * 3u + 1u];
        uint b2 = q1p[pidx * 3u + 2u];
        float sa0 = (float)as_type<half>((ushort)(a0 & 0xFFFFu));
        float sa1 = (float)as_type<half>((ushort)(a1 >> 16));
        uint bitsa0 = (a0 >> 16) | (a1 << 16);
        uint bitsa1 = a2;
        float sb0 = (float)as_type<half>((ushort)(b0 & 0xFFFFu));
        float sb1 = (float)as_type<half>((ushort)(b1 >> 16));
        uint bitsb0 = (b0 >> 16) | (b1 << 16);
        uint bitsb1 = b2;
        ulong g = (ulong)pidx * 2u;
        float4 s0a = float4(0.0f), s1a = float4(0.0f);
        float4 s0b = float4(0.0f), s1b = float4(0.0f);
        for (uint j = 0; j < 8; ++j) {
            float4 x0 = xs[g * 8u + j];
            float4 x1 = xs[(g + 1u) * 8u + j];
            uint na0 = bitsa0 >> (j * 4u);
            uint na1 = bitsa1 >> (j * 4u);
            uint nb0 = bitsb0 >> (j * 4u);
            uint nb1 = bitsb1 >> (j * 4u);
            s0a += select(-x0, x0, bool4(na0 & 1u, na0 & 2u, na0 & 4u, na0 & 8u));
            s1a += select(-x1, x1, bool4(na1 & 1u, na1 & 2u, na1 & 4u, na1 & 8u));
            s0b += select(-x0, x0, bool4(nb0 & 1u, nb0 & 2u, nb0 & 4u, nb0 & 8u));
            s1b += select(-x1, x1, bool4(nb1 & 1u, nb1 & 2u, nb1 & 4u, nb1 & 8u));
        }
        acc0 += sa0 * (s0a.x + s0a.y + s0a.z + s0a.w)
              + sa1 * (s1a.x + s1a.y + s1a.z + s1a.w);
        acc1 += sb0 * (s0b.x + s0b.y + s0b.z + s0b.w)
              + sb1 * (s1b.x + s1b.y + s1b.z + s1b.w);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        y[r0] = acc0;
        if (two) y[r0 + 1u] = acc1;
    }
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

// ── Whole-block GDN kernels: an entire linear layer (norm → mixer →
// conv → recurrence → out_proj → norm → FFN) runs inside ONE command
// buffer, hidden state resident on device; the CPU sees one sync per
// BLOCK of consecutive GDN layers instead of ~12 per layer. ──

// Tiny f32 matvec (the GDN a/b gate projections live dequantized in
// RAM; they are uploaded once through the small-vector cache).
kernel void f32_matvec(
    device const float*  q    [[buffer(0)]],
    device const float*  xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       cols [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tgpos * sgs + sg;
    if (row >= rows) return;
    ulong base = (ulong)row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += q[base + i] * xs[i];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

kernel void rmsnorm_k(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float*       o [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    constant uint&  gemma [[buffer(4)]],
    constant float&   eps [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[8];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256u) { float v = x[i]; acc += v * v; }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float inv = rsqrt(tot / (float)n + eps);
    for (uint i = tid; i < n; i += 256u) {
        float wv = gemma != 0u ? (1.0f + w[i]) : w[i];
        o[i] = x[i] * inv * wv;
    }
}

// cq = silu(depthwise causal conv over [ring…, current qkv])
kernel void gdn_conv(
    device const float* qkv  [[buffer(0)]],
    device const float* ring [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    device float*       cq   [[buffer(3)]],
    constant uint&     c_dim [[buffer(4)]],
    constant uint&        kk [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c_dim) return;
    float acc = qkv[i] * taps[i * kk + kk - 1u];
    for (uint j = 0; j + 1u < kk; ++j) acc += ring[j * c_dim + i] * taps[i * kk + j];
    cq[i] = acc / (1.0f + exp(-acc));
}

// Ring shift: drop the oldest position, append the RAW current qkv.
kernel void gdn_ring_shift(
    device float*       ring [[buffer(0)]],
    device const float* qkv  [[buffer(1)]],
    constant uint&     c_dim [[buffer(2)]],
    constant uint&        kk [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c_dim) return;
    for (uint j = 0; j + 2u < kk; ++j) ring[j * c_dim + i] = ring[(j + 1u) * c_dim + i];
    ring[(kk - 2u) * c_dim + i] = qkv[i];
}

// Per-head decay g and write strength beta.
kernel void gdn_gates(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* a_log   [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float*       g       [[buffer(4)]],
    device float*       beta    [[buffer(5)]],
    constant uint&      nv      [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= nv) return;
    float x = a[i] + dt_bias[i];
    float sp = x > 20.0f ? x : log(1.0f + exp(x));
    g[i] = exp(-exp(a_log[i]) * sp);
    beta[i] = 1.0f / (1.0f + exp(-b[i]));
}

// l2-norm inverses of q/k per K head (one simdgroup per head).
kernel void gdn_qk_norms(
    device const float* cq   [[buffer(0)]],
    device float*       invq [[buffer(1)]],
    device float*       invk [[buffer(2)]],
    constant uint&      nk   [[buffer(3)]],
    constant uint&      dk   [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint h = tg * sgs + sg;
    if (h >= nk) return;
    uint kd = nk * dk;
    float nq = 0.0f, nkn = 0.0f;
    for (uint d = lane; d < dk; d += 32u) {
        float q = cq[h * dk + d];      nq  += q * q;
        float k = cq[kd + h * dk + d]; nkn += k * k;
    }
    nq = simd_sum(nq); nkn = simd_sum(nkn);
    if (lane == 0) {
        invq[h] = 1.0f / (sqrt(nq + 1e-6f) * sqrt((float)dk));
        invk[h] = 1.0f / sqrt(nkn + 1e-6f);
    }
}

// The GatedDeltaNet recurrence + gated RMSNorm, one threadgroup per V
// head (dv threads, thread dj owns one output column):
//   kv = k'ᵀ S_old;  Δ = β(v − g·kv);  S = g·S_old + k' ⊗ Δ;  o = q'ᵀ S
// S rows are read coalesced (threads span dj).
kernel void gdn_state_update(
    device float*       S     [[buffer(0)]],
    device const float* cq    [[buffer(1)]],
    device const float* z     [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const float* beta  [[buffer(4)]],
    device const float* invq  [[buffer(5)]],
    device const float* invk  [[buffer(6)]],
    device const float* gnorm [[buffer(7)]],
    device float*       of    [[buffer(8)]],
    constant uint&      nv    [[buffer(9)]],
    constant uint&      nk    [[buffer(10)]],
    constant uint&      dk    [[buffer(11)]],
    constant uint&      dv    [[buffer(12)]],
    constant float&     eps   [[buffer(13)]],
    uint h    [[threadgroup_position_in_grid]],
    uint dj   [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    uint rep = nv / nk;
    uint ko = h / rep;
    uint kd = nk * dk;
    device float* s = S + (ulong)h * dk * dv;
    float gh = g[h];
    float bh = beta[h];
    float iq = invq[ko];
    float ik = invk[ko];
    float vt = cq[2u * kd + h * dv + dj];
    float kv = 0.0f;
    for (uint di = 0; di < dk; ++di) {
        kv += cq[kd + ko * dk + di] * ik * s[di * dv + dj];
    }
    float delta = (vt - gh * kv) * bh;
    float o = 0.0f;
    for (uint di = 0; di < dk; ++di) {
        float kf = cq[kd + ko * dk + di] * ik;
        float qf = cq[ko * dk + di] * iq;
        float cell = gh * s[di * dv + dj] + kf * delta;
        s[di * dv + dj] = cell;
        o += qf * cell;
    }
    threadgroup float part[32];
    float ss = simd_sum(o * o);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k2 = 0; k2 < (dv + 31u) / 32u; ++k2) tot += part[k2];
    float inv = rsqrt(tot / (float)dv + eps);
    float zz = z[h * dv + dj];
    of[h * dv + dj] = o * inv * gnorm[dj] * (zz / (1.0f + exp(-zz)));
}
"#;

struct Ctx {
    _device: Device,
    queue: CommandQueue,
    q8: ComputePipelineState,
    q8mm: ComputePipelineState,
    q1: ComputePipelineState,
    flag: ComputePipelineState,
    rmsn: ComputePipelineState,
    f16mv: ComputePipelineState,
    conv: ComputePipelineState,
    ring: ComputePipelineState,
    gates: ComputePipelineState,
    qkn: ComputePipelineState,
    stateup: ComputePipelineState,
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
    let rmsn = pso("rmsnorm_k")?;
    let f16mv = pso("f32_matvec")?;
    let conv = pso("gdn_conv")?;
    let ring = pso("gdn_ring_shift")?;
    let gates = pso("gdn_gates")?;
    let qkn = pso("gdn_qk_norms")?;
    let stateup = pso("gdn_state_update")?;
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
        rmsn,
        f16mv,
        conv,
        ring,
        gates,
        qkn,
        stateup,
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
    let sgs = 8u64; // × 2 rows per simdgroup
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 2), 1, 1),
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


/// One GDN layer's worth of tensors/vectors for the whole-block GPU
/// path. Matvec tensors are (directory idx, rows, cols) of q1 weights.
pub struct GdnGpuLayer<'a> {
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub qkv: (usize, usize, usize),
    pub z: (usize, usize, usize),
    pub a: (&'a [f32], usize, usize),
    pub b: (&'a [f32], usize, usize),
    pub out: (usize, usize, usize),
    pub gate: (usize, usize, usize),
    pub up: (usize, usize, usize),
    pub down: (usize, usize, usize),
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub gnorm: &'a [f32],
}

/// Shared dims of the block (identical across GDN layers of a model).
#[derive(Clone, Copy)]
pub struct GdnGpuCfg {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub hidden: usize,
    pub inter: usize,
    pub c_dim: usize,
    pub eps: f32,
    /// Gemma-style norms: x̂·(1+w) (qwen3_5 family) vs Qwen x̂·w.
    pub gemma: bool,
}

/// Model-wide dims every token-graph layer agrees on.
#[derive(Clone, Copy)]
pub struct GraphDims {
    pub hidden: usize,
    pub eps: f32,
    /// Gemma-style norms: x̂·(1+w) (qwen3_5 family) vs Qwen x̂·w.
    pub gemma: bool,
}

/// One full-attention layer's q1 graph inputs: (directory idx, rows,
/// cols) triples; the qk-norms / RoPE / KV / attend stay on the CPU
/// between the graph's QKV prefix and O+FFN suffix.
pub struct AttnGpuLayer<'a> {
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub wq: (usize, usize, usize),
    pub wk: (usize, usize, usize),
    pub wv: (usize, usize, usize),
    pub wo: (usize, usize, usize),
    pub gate: (usize, usize, usize),
    pub up: (usize, usize, usize),
    pub down: (usize, usize, usize),
}

fn io_buf(c: &Ctx, key: usize, nbytes: usize) -> Buffer {
    let mut cache = c.io_bufs.lock().unwrap();
    cache
        .entry(key)
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            c._device.new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
        })
        .clone()
}

/// Small constant vectors cached by their (stable) data pointer.
fn const_buf(c: &Ctx, data: &[f32]) -> Buffer {
    let mut cache = c.rs_bufs.lock().unwrap();
    cache
        .entry((data.as_ptr() as usize, usize::MAX - 2))
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            c._device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .clone()
}

fn enc_simple(
    c_cmd: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    bufs: &[(&Buffer, u64)],
    words: &[u32],
    floats: &[f32],
    grid: (u64, u64),
) {
    let enc = c_cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    for (i, (b, off)) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), *off);
    }
    let base = bufs.len() as u64;
    for (i, w) in words.iter().enumerate() {
        enc.set_bytes(base + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
    }
    for (i, f) in floats.iter().enumerate() {
        enc.set_bytes(
            base + words.len() as u64 + i as u64,
            4,
            f as *const f32 as *const std::ffi::c_void,
        );
    }
    enc.dispatch_threads(MTLSize::new(grid.0, 1, 1), MTLSize::new(grid.1, 1, 1));
    enc.end_encoding();
}

/// A token's worth of layers as few command buffers: hidden lives in a
/// device buffer across GDN runs AND full-attention layers; the only
/// syncs are where the CPU genuinely needs data (q/k/v before the KV
/// attend, recurrent states, the final hidden). Contract: validate
/// every layer (`gdn_ok`/`attn_ok`) BEFORE encoding — after the first
/// `sync` a refused encode would leave the token half-executed.
pub struct TokenGraph {
    c: &'static Ctx,
    model: Arc<CmfModel>,
    fbuf: Buffer,
    safe_len: usize,
    dims: GraphDims,
    cmd: Option<metal::CommandBuffer>,
    h_b: Buffer,
    n_b: Buffer,
    d_b: Buffer,
    /// Recurrent-state buffers awaiting readback (buffer, f32 len).
    dirty: Vec<(Buffer, usize)>,
    /// Next state-buffer cache slot (reset when `dirty` drains).
    st_next: usize,
    /// q/k/v buffers of the last encoded attention prefix.
    qkv_bufs: Option<(Buffer, Buffer, Buffer)>,
}

impl TokenGraph {
    pub fn new(model: &Arc<CmfModel>, dims: GraphDims, h: &[f32]) -> Option<TokenGraph> {
        let c = ctx()?;
        if h.len() != dims.hidden {
            return None;
        }
        let (fbuf, safe_len) = file_buffer(c, model.primary_bytes())?;
        let h_b = io_buf(c, 20_000_000_003 + dims.hidden, dims.hidden * 4);
        let n_b = io_buf(c, 21_000_000_011 + dims.hidden, dims.hidden * 4);
        let d_b = io_buf(c, 32_000_000_207 + dims.hidden, dims.hidden * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(h.as_ptr(), h_b.contents() as *mut f32, dims.hidden);
        }
        Some(TokenGraph {
            c,
            model: model.clone(),
            fbuf,
            safe_len,
            dims,
            cmd: None,
            h_b,
            n_b,
            d_b,
            dirty: Vec::new(),
            st_next: 0,
            qkv_bufs: None,
        })
    }

    /// Validate one q1 tensor and resolve its absolute payload offset.
    fn q1_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, rows, cols) = t;
        if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        if abs + rows * (cols / GROUP_SIZE) * Q1_TILE > self.safe_len {
            return None;
        }
        Some(abs)
    }

    /// Pre-flight check for a GDN layer (call before any encode).
    pub fn gdn_ok(&self, l: &GdnGpuLayer, cfg: &GdnGpuCfg) -> bool {
        if cfg.kk < 2 || cfg.dv % 32 != 0 || cfg.dv > 1024 || cfg.hidden != self.dims.hidden {
            return false;
        }
        if l.a.0.len() != l.a.1 * l.a.2 || l.b.0.len() != l.b.1 * l.b.2 {
            return false;
        }
        [l.qkv, l.z, l.out, l.gate, l.up, l.down].iter().all(|t| self.q1_abs(*t).is_some())
    }

    /// Pre-flight check for a full-attention layer.
    pub fn attn_ok(&self, l: &AttnGpuLayer) -> bool {
        // The suffix reads the attention output back through ao (wo
        // cols) and writes hidden (wo rows) — both must match dims.
        if l.wo.1 != self.dims.hidden || l.down.1 != self.dims.hidden {
            return false;
        }
        [l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down].iter().all(|t| self.q1_abs(*t).is_some())
    }

    fn ensure_cmd(&mut self) -> metal::CommandBuffer {
        if self.cmd.is_none() {
            self.cmd = Some(self.c.queue.new_command_buffer().to_owned());
        }
        self.cmd.as_ref().unwrap().clone()
    }

    /// Submit everything encoded so far and wait for completion.
    pub fn sync(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            submit_and_wait(self.c, &cmd, &[]);
        }
    }

    /// Copy finished recurrent states back to their CPU owners (call
    /// after `sync`; order matches the `encode_gdn_run` calls).
    pub fn read_states(&mut self, outs: &mut [&mut [f32]]) {
        debug_assert_eq!(outs.len(), self.dirty.len());
        for ((buf, len), out) in self.dirty.drain(..).zip(outs.iter_mut()) {
            debug_assert_eq!(len, out.len());
            unsafe {
                std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), len);
            }
        }
        self.st_next = 0;
    }

    /// Final sync + hidden readback.
    pub fn finish(mut self, h: &mut [f32]) {
        self.sync();
        debug_assert!(self.dirty.is_empty(), "unread recurrent states at finish");
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.h_b.contents() as *const f32,
                h.as_mut_ptr(),
                self.dims.hidden,
            );
        }
    }

    /// norm(h) → n_b, then QKV projections n_b → q/k/v buffers. The
    /// caller must `sync` + `read_qkv` before using the values.
    pub fn encode_attn_prefix(&mut self, l: &AttnGpuLayer) {
        let cmd = self.ensure_cmd();
        let (aq, ak, av) =
            (self.q1_abs(l.wq).unwrap(), self.q1_abs(l.wk).unwrap(), self.q1_abs(l.wv).unwrap());
        enc_simple(
            &cmd,
            &self.c.rmsn,
            &[(&self.h_b, 0), (&const_buf(self.c, l.attn_norm), 0), (&self.n_b, 0)],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let q_b = io_buf(self.c, 40_000_000_003 + l.wq.1, l.wq.1 * 4);
        let k_b = io_buf(self.c, 41_000_000_019 + l.wk.1, l.wk.1 * 4);
        let v_b = io_buf(self.c, 42_000_000_037 + l.wv.1, l.wv.1 * 4);
        let enc = cmd.new_compute_command_encoder();
        encode_q1_matvec(self.c, enc, &self.fbuf, aq, &self.n_b, &q_b, l.wq.1, l.wq.2 / GROUP_SIZE);
        encode_q1_matvec(self.c, enc, &self.fbuf, ak, &self.n_b, &k_b, l.wk.1, l.wk.2 / GROUP_SIZE);
        encode_q1_matvec(self.c, enc, &self.fbuf, av, &self.n_b, &v_b, l.wv.1, l.wv.2 / GROUP_SIZE);
        enc.end_encoding();
        self.qkv_bufs = Some((q_b, k_b, v_b));
    }

    /// Read the prefix's q/k/v after `sync` (UMA memcpy).
    pub fn read_qkv(&mut self, q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        let (q_b, k_b, v_b) = self.qkv_bufs.take().expect("read_qkv without prefix");
        unsafe {
            std::ptr::copy_nonoverlapping(q_b.contents() as *const f32, q.as_mut_ptr(), q.len());
            std::ptr::copy_nonoverlapping(k_b.contents() as *const f32, k.as_mut_ptr(), k.len());
            std::ptr::copy_nonoverlapping(v_b.contents() as *const f32, v.as_mut_ptr(), v.len());
        }
    }

    /// Upload the CPU-attended output `ao`, then O-projection +
    /// residual + post-norm + FFN + residual on the device.
    pub fn encode_attn_suffix(&mut self, l: &AttnGpuLayer, ao: &[f32]) {
        debug_assert_eq!(ao.len(), l.wo.2);
        let cmd = self.ensure_cmd();
        let ao_b = io_buf(self.c, 43_000_000_057 + ao.len(), ao.len() * 4);
        // Safe to write: the previous command buffer completed at the
        // prefix sync, and the new one has not been committed yet.
        unsafe {
            std::ptr::copy_nonoverlapping(ao.as_ptr(), ao_b.contents() as *mut f32, ao.len());
        }
        {
            let enc = cmd.new_compute_command_encoder();
            let abs = self.q1_abs(l.wo).unwrap();
            encode_q1_matvec(
                self.c,
                enc,
                &self.fbuf,
                abs,
                &ao_b,
                &self.d_b,
                l.wo.1,
                l.wo.2 / GROUP_SIZE,
            );
            enc.end_encoding();
        }
        enc_axpy(self.c, &cmd, &self.d_b, &self.h_b, 1.0, self.dims.hidden);
        self.encode_post_ffn(&cmd, l.post_norm, l.gate, l.up, l.down);
    }

    /// post-norm(h) → n_b, gate/up, SiLU·mul, down, h += d — shared by
    /// the GDN layer tail and the attention suffix.
    fn encode_post_ffn(
        &self,
        cmd: &metal::CommandBufferRef,
        post_norm: &[f32],
        gate: (usize, usize, usize),
        up: (usize, usize, usize),
        down: (usize, usize, usize),
    ) {
        let inter = gate.1;
        let fg_b = io_buf(self.c, 33_000_000_209 + inter, inter * 4);
        let fu_b = io_buf(self.c, 34_000_000_213 + inter, inter * 4);
        let fa_b = io_buf(self.c, 35_000_000_221 + inter, inter * 4);
        enc_simple(
            cmd,
            &self.c.rmsn,
            &[(&self.h_b, 0), (&const_buf(self.c, post_norm), 0), (&self.n_b, 0)],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        {
            let enc = cmd.new_compute_command_encoder();
            let (ag, au) = (self.q1_abs(gate).unwrap(), self.q1_abs(up).unwrap());
            encode_q1_matvec(
                self.c,
                enc,
                &self.fbuf,
                ag,
                &self.n_b,
                &fg_b,
                gate.1,
                gate.2 / GROUP_SIZE,
            );
            encode_q1_matvec(
                self.c,
                enc,
                &self.fbuf,
                au,
                &self.n_b,
                &fu_b,
                up.1,
                up.2 / GROUP_SIZE,
            );
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.c.silu);
            enc.set_buffer(0, Some(&fg_b), 0);
            enc.set_buffer(1, Some(&fu_b), 0);
            enc.set_buffer(2, Some(&fg_b), 0); // dummy col (has_col = 0)
            enc.set_buffer(3, Some(&fa_b), 0);
            let (n_u, hc) = (inter as u32, 0u32);
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hc as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            let ad = self.q1_abs(down).unwrap();
            encode_q1_matvec(
                self.c,
                enc,
                &self.fbuf,
                ad,
                &fa_b,
                &self.d_b,
                down.1,
                down.2 / GROUP_SIZE,
            );
            enc.end_encoding();
        }
        enc_axpy(self.c, cmd, &self.d_b, &self.h_b, 1.0, self.dims.hidden);
    }

    /// Encode a run of consecutive GDN layers; recurrent states upload
    /// now and read back via `read_states` after the next `sync`.
    pub fn encode_gdn_run(
        &mut self,
        layers: &[GdnGpuLayer],
        states: &[&[f32]],
        cfg: &GdnGpuCfg,
    ) -> bool {
        if layers.is_empty() || layers.len() != states.len() {
            return false;
        }
        let c = self.c;
        let vd = cfg.nv * cfg.dv;
        let ring_len = (cfg.kk - 1) * cfg.c_dim;
        let s_len = cfg.nv * cfg.dk * cfg.dv;

        // Resolve and validate every q1 tensor before encoding anything.
        let mut abss: Vec<[usize; 6]> = Vec::with_capacity(layers.len());
        for (l, st) in layers.iter().zip(states) {
            if !self.gdn_ok(l, cfg) || st.len() != ring_len + s_len {
                return false;
            }
            let mut a8 = [0usize; 6];
            for (slot, t) in [l.qkv, l.z, l.out, l.gate, l.up, l.down].iter().enumerate() {
                a8[slot] = self.q1_abs(*t).unwrap();
            }
            abss.push(a8);
        }

        let qkv_b = io_buf(c, 22_000_000_017 + cfg.c_dim, cfg.c_dim * 4);
        let z_b = io_buf(c, 23_000_000_021 + vd, vd * 4);
        let a_b = io_buf(c, 24_000_000_047 + cfg.nv, cfg.nv * 4);
        let b_b = io_buf(c, 25_000_000_071 + cfg.nv, cfg.nv * 4);
        let cq_b = io_buf(c, 26_000_000_081 + cfg.c_dim, cfg.c_dim * 4);
        let g_b = io_buf(c, 27_000_000_093 + cfg.nv, cfg.nv * 4);
        let bt_b = io_buf(c, 28_000_000_129 + cfg.nv, cfg.nv * 4);
        let iq_b = io_buf(c, 29_000_000_131 + cfg.nk, cfg.nk * 4);
        let ik_b = io_buf(c, 30_000_000_133 + cfg.nk, cfg.nk * 4);
        let of_b = io_buf(c, 31_000_000_161 + vd, vd * 4);
        let st_bs: Vec<Buffer> = (0..layers.len())
            .map(|i| {
                io_buf(
                    c,
                    36_000_000_223 + (self.st_next + i) * 613 + ring_len + s_len,
                    (ring_len + s_len) * 4,
                )
            })
            .collect();
        self.st_next += layers.len();

        // Upload states (UMA memcpy into shared buffers) — safe: these
        // slots were read back before the previous sync window closed.
        unsafe {
            for (st, sb) in states.iter().zip(&st_bs) {
                std::ptr::copy_nonoverlapping(st.as_ptr(), sb.contents() as *mut f32, st.len());
            }
        }

        let cmd = self.ensure_cmd();
        let fbuf = self.fbuf.clone();
        let (h_b, n_b, d_b) = (self.h_b.clone(), self.n_b.clone(), self.d_b.clone());
        let enc_one = |pso: &ComputePipelineState,
                       bufs: &[(&Buffer, u64)],
                       words: &[u32],
                       floats: &[f32],
                       grid: (u64, u64)| {
            enc_simple(&cmd, pso, bufs, words, floats, grid);
        };
        let vec_buf = |data: &[f32]| -> Buffer { const_buf(c, data) };

        for (l, (a8, sb)) in layers.iter().zip(abss.iter().zip(&st_bs)) {
        let s_off = (ring_len * 4) as u64;
        // 1. attn rmsnorm h → n
        enc_one(
            &c.rmsn,
            &[(&h_b, 0), (&vec_buf(l.attn_norm), 0), (&n_b, 0)],
            &[cfg.hidden as u32, cfg.gemma as u32],
            &[cfg.eps],
            (256, 256),
        );
        // 2. mixer: qkv, z, a, b (independent — one encoder)
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q1_matvec(c, enc, &fbuf, a8[0], &n_b, &qkv_b, l.qkv.1, l.qkv.2 / GROUP_SIZE);
            encode_q1_matvec(c, enc, &fbuf, a8[1], &n_b, &z_b, l.z.1, l.z.2 / GROUP_SIZE);
            for (t, y) in [(&l.a, &a_b), (&l.b, &b_b)] {
                let (data, rows, cols) = *t;
                let wb = vec_buf(data);
                enc.set_compute_pipeline_state(&c.f16mv);
                enc.set_buffer(0, Some(&wb), 0);
                enc.set_buffer(1, Some(&n_b), 0);
                enc.set_buffer(2, Some(y), 0);
                let (cu, ru) = (cols as u32, rows as u32);
                enc.set_bytes(3, 4, &cu as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &ru as *const u32 as *const std::ffi::c_void);
                let sgs = 8u64;
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(sgs), 1, 1),
                    MTLSize::new(sgs * 32, 1, 1),
                );
            }
            enc.end_encoding();
        }
        // 3. conv + silu (reads ring BEFORE the shift)
        enc_one(
            &c.conv,
            &[(&qkv_b, 0), (sb, 0), (&vec_buf(l.conv1d), 0), (&cq_b, 0)],
            &[cfg.c_dim as u32, cfg.kk as u32],
            &[],
            (cfg.c_dim as u64, 256),
        );
        // 4. ring shift + gates + qk norms (one encoder, independent)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.ring);
            enc.set_buffer(0, Some(sb), 0);
            enc.set_buffer(1, Some(&qkv_b), 0);
            let (cd, kk) = (cfg.c_dim as u32, cfg.kk as u32);
            enc.set_bytes(2, 4, &cd as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &kk as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(cfg.c_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.set_compute_pipeline_state(&c.gates);
            enc.set_buffer(0, Some(&a_b), 0);
            enc.set_buffer(1, Some(&b_b), 0);
            enc.set_buffer(2, Some(&vec_buf(l.a_log)), 0);
            enc.set_buffer(3, Some(&vec_buf(l.dt_bias)), 0);
            enc.set_buffer(4, Some(&g_b), 0);
            enc.set_buffer(5, Some(&bt_b), 0);
            let nv = cfg.nv as u32;
            enc.set_bytes(6, 4, &nv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(cfg.nv as u64, 1, 1), MTLSize::new(64, 1, 1));
            enc.set_compute_pipeline_state(&c.qkn);
            enc.set_buffer(0, Some(&cq_b), 0);
            enc.set_buffer(1, Some(&iq_b), 0);
            enc.set_buffer(2, Some(&ik_b), 0);
            let (nk, dk) = (cfg.nk as u32, cfg.dk as u32);
            enc.set_bytes(3, 4, &nk as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &dk as *const u32 as *const std::ffi::c_void);
            let sgs = 8u64;
            enc.dispatch_thread_groups(
                MTLSize::new((cfg.nk as u64).div_ceil(sgs), 1, 1),
                MTLSize::new(sgs * 32, 1, 1),
            );
            enc.end_encoding();
        }
        // 5. recurrence + gated norm → of
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.stateup);
            enc.set_buffer(0, Some(sb), s_off);
            enc.set_buffer(1, Some(&cq_b), 0);
            enc.set_buffer(2, Some(&z_b), 0);
            enc.set_buffer(3, Some(&g_b), 0);
            enc.set_buffer(4, Some(&bt_b), 0);
            enc.set_buffer(5, Some(&iq_b), 0);
            enc.set_buffer(6, Some(&ik_b), 0);
            enc.set_buffer(7, Some(&vec_buf(l.gnorm)), 0);
            enc.set_buffer(8, Some(&of_b), 0);
            let w4 = [cfg.nv as u32, cfg.nk as u32, cfg.dk as u32, cfg.dv as u32];
            for (i, w) in w4.iter().enumerate() {
                enc.set_bytes(9 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
            }
            enc.set_bytes(13, 4, &cfg.eps as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(cfg.nv as u64, 1, 1),
                MTLSize::new(cfg.dv as u64, 1, 1),
            );
            enc.end_encoding();
        }
        // 6. out_proj of → d;  7. h += d
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q1_matvec(c, enc, &fbuf, a8[2], &of_b, &d_b, l.out.1, l.out.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        enc_axpy(c, &cmd, &d_b, &h_b, 1.0, cfg.hidden);
        // 8–12. post-norm + FFN + residual (shared with attn suffix)
        self.encode_post_ffn(&cmd, l.post_norm, l.gate, l.up, l.down);
        }

        for (sb, st) in st_bs.iter().zip(states) {
            self.dirty.push((sb.clone(), st.len()));
        }
        true
    }
}

/// A BLOCK of consecutive GDN layers in one command buffer: hidden
/// state stays device-resident across norm → mixer → conv → recurrence
/// → out_proj → norm → FFN → residuals of every layer; per-layer
/// recurrent states round-trip through shared memory (the CPU remains
/// their owner, so every other path stays coherent for free). One sync
/// per block instead of ~12 per layer.
pub fn gdn_block(
    model: &Arc<CmfModel>,
    layers: &[GdnGpuLayer],
    states: &mut [&mut [f32]],
    cfg: &GdnGpuCfg,
    h: &mut [f32],
) -> bool {
    let dims = GraphDims { hidden: cfg.hidden, eps: cfg.eps, gemma: cfg.gemma };
    let Some(mut g) = TokenGraph::new(model, dims, h) else { return false };
    let ro: Vec<&[f32]> = states.iter().map(|s| &**s).collect();
    if !g.encode_gdn_run(layers, &ro, cfg) {
        return false;
    }
    g.sync();
    g.read_states(states);
    g.finish(h);
    true
}

/// `y += w·d` as its own encoder.
fn enc_axpy(c: &Ctx, cmd: &metal::CommandBufferRef, d: &Buffer, y: &Buffer, w: f32, n: usize) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.axpy);
    enc.set_buffer(0, Some(d), 0);
    enc.set_buffer(1, Some(y), 0);
    let n_u = n as u32;
    enc.set_bytes(2, 4, &w as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
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
