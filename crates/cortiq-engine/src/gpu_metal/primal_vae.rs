//! Primal VAE on Metal, resident: a whole encode or decode
//! (`primal_vae::PvPlan`) in one command buffer — the input written once
//! into shared memory, every activation on the device, one wait.
//!
//! The kernels mirror the wgpu ones (`gpu_wgpu::primal_vae`), f32 NCHW:
//! `pv_conv` (3×3, dilation = padding, implicit GEMM of 64 pixels × 64
//! output channels per threadgroup, bias and an optional in-place residual
//! in the epilogue), `pv_norm` (channel RMSNorm × γ, SiLU), `pv_unshuf` /
//! `pv_shuf` (`pixel_unshuffle(2)` / `pixel_shuffle(2)` with an optional
//! in-place add).

use super::{Ctx, ctx};
use crate::primal_vae::{PvPlan, PvStep, PvWeights};
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct P { uint a; uint b; uint h; uint w; uint dil; uint flag; uint k; uint n; };

kernel void pv_conv(device const float* src [[buffer(0)]],
                    device const float* wt [[buffer(1)]],
                    device const float* bias [[buffer(2)]],
                    device float* dst [[buffer(3)]],
                    constant P& p [[buffer(4)]],
                    uint2 wg [[threadgroup_position_in_grid]],
                    uint li [[thread_index_in_threadgroup]]) {
    threadgroup float ta[1024];
    threadgroup float tb[1024];
    const uint hw = p.h * p.w;
    const uint m0 = wg.x * 64u;
    const uint n0 = wg.y * 64u;
    const uint lx = li % 16u;
    const uint ly = li / 16u;
    const int dil = int(p.dil);
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint k0 = 0u; k0 < p.k; k0 += 16u) {
        for (uint e = 0u; e < 4u; e++) {
            const uint idx = li + e * 256u;
            const uint kk = idx / 64u;
            const uint mm = idx % 64u;
            const uint k = k0 + kk;
            const uint pix = m0 + mm;
            float va = 0.0f;
            if (k < p.k && pix < hw) {
                const uint ci = k / 9u;
                const uint tap = k - ci * 9u;
                const int y = int(pix / p.w) + (int(tap / 3u) - 1) * dil;
                const int x = int(pix % p.w) + (int(tap % 3u) - 1) * dil;
                if (y >= 0 && x >= 0 && y < int(p.h) && x < int(p.w)) {
                    va = src[ci * hw + uint(y) * p.w + uint(x)];
                }
            }
            ta[idx] = va;
            const uint nn = idx / 16u;
            const uint kb = idx % 16u;
            float vb = 0.0f;
            if (n0 + nn < p.b && k0 + kb < p.k) { vb = wt[(n0 + nn) * p.k + k0 + kb]; }
            tb[kb * 64u + nn] = vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 16u; kk++) {
            const float4 a = *((threadgroup const float4*)&ta[kk * 64u + lx * 4u]);
            const float4 bv = *((threadgroup const float4*)&tb[kk * 64u + ly * 4u]);
            acc[0] = fma(float4(a.x), bv, acc[0]);
            acc[1] = fma(float4(a.y), bv, acc[1]);
            acc[2] = fma(float4(a.z), bv, acc[2]);
            acc[3] = fma(float4(a.w), bv, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0u; r < 4u; r++) {
        const uint pix = m0 + lx * 4u + r;
        if (pix >= hw) { continue; }
        for (uint c = 0u; c < 4u; c++) {
            const uint oc = n0 + ly * 4u + c;
            if (oc >= p.b) { continue; }
            const uint o = oc * hw + pix;
            float v = acc[r][c] + bias[oc];
            if (p.flag != 0u) { v += dst[o]; }
            dst[o] = v;
        }
    }
}

// The same convolution on simdgroup matrices: 64 pixels × 64 output
// channels per threadgroup of four simdgroups, each a 32×32 quarter as 4×4
// blocks of 8×8 f32 accumulators; 16 K values per staged step; the tile
// leaves through threadgroup memory so the epilogue (bias, residual, edge
// guards) writes rows of pixels.
kernel void pv_conv_sg(device const float* src [[buffer(0)]],
                       device const float* wt [[buffer(1)]],
                       device const float* bias [[buffer(2)]],
                       device float* dst [[buffer(3)]],
                       constant P& p [[buffer(4)]],
                       uint2 wg [[threadgroup_position_in_grid]],
                       uint li [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ta[16 * 64];
    threadgroup float tb[16 * 64];
    threadgroup float tc[64 * 64];
    const uint hw = p.h * p.w;
    const uint m0 = wg.x * 64u;
    const uint n0 = wg.y * 64u;
    const uint sm = (sg / 2u) * 32u;
    const uint sn = (sg % 2u) * 32u;
    const int dil = int(p.dil);
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0u; i < 4u; i++) {
        for (uint j = 0u; j < 4u; j++) { acc[i][j] = simdgroup_float8x8(0.0f); }
    }
    for (uint k0 = 0u; k0 < p.k; k0 += 16u) {
        for (uint e = 0u; e < 8u; e++) {
            const uint idx = li + e * 128u;
            const uint kk = idx / 64u;
            const uint mm = idx % 64u;
            const uint k = k0 + kk;
            const uint pix = m0 + mm;
            float va = 0.0f;
            if (k < p.k && pix < hw) {
                const uint ci = k / 9u;
                const uint tap = k - ci * 9u;
                const int y = int(pix / p.w) + (int(tap / 3u) - 1) * dil;
                const int x = int(pix % p.w) + (int(tap % 3u) - 1) * dil;
                if (y >= 0 && x >= 0 && y < int(p.h) && x < int(p.w)) {
                    va = src[ci * hw + uint(y) * p.w + uint(x)];
                }
            }
            ta[idx] = va;
            const uint nn = idx / 16u;
            const uint kb = idx % 16u;
            float vb = 0.0f;
            if (n0 + nn < p.b && k0 + kb < p.k) { vb = wt[(n0 + nn) * p.k + k0 + kb]; }
            tb[kb * 64u + nn] = vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kb = 0u; kb < 16u; kb += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            for (uint i = 0u; i < 4u; i++) {
                simdgroup_load(a[i], &ta[kb * 64u + sm + 8u * i], 64, ulong2(0, 0), true);
                simdgroup_load(b[i], &tb[kb * 64u + sn + 8u * i], 64, ulong2(0, 0), false);
            }
            for (uint i = 0u; i < 4u; i++) {
                for (uint j = 0u; j < 4u; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0u; i < 4u; i++) {
        for (uint j = 0u; j < 4u; j++) {
            simdgroup_store(acc[i][j], &tc[(sn + 8u * j) * 64u + sm + 8u * i], 64, ulong2(0, 0), true);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = li; idx < 4096u; idx += 128u) {
        const uint nn = idx / 64u;
        const uint mm = idx % 64u;
        const uint oc = n0 + nn;
        const uint pix = m0 + mm;
        if (oc < p.b && pix < hw) {
            const uint o = oc * hw + pix;
            float v = tc[idx] + bias[oc];
            if (p.flag != 0u) { v += dst[o]; }
            dst[o] = v;
        }
    }
}

kernel void pv_norm(device const float* src [[buffer(0)]],
                    device const float* g [[buffer(1)]],
                    device const float* unused [[buffer(2)]],
                    device float* dst [[buffer(3)]],
                    constant P& p [[buffer(4)]],
                    uint pix [[thread_position_in_grid]]) {
    const uint hw = p.h * p.w;
    if (pix >= hw) { return; }
    float ss = 0.0f;
    for (uint c = 0u; c < p.a; c++) { const float v = src[c * hw + pix]; ss += v * v; }
    const float inv = rsqrt(ss / float(p.a) + 1e-6f);
    for (uint c = 0u; c < p.a; c++) {
        const float v = src[c * hw + pix] * inv * g[c];
        dst[c * hw + pix] = v / (1.0f + exp(-v));
    }
}

kernel void pv_unshuf(device const float* src [[buffer(0)]],
                      device const float* u1 [[buffer(1)]],
                      device const float* u2 [[buffer(2)]],
                      device float* dst [[buffer(3)]],
                      constant P& p [[buffer(4)]],
                      uint o [[thread_position_in_grid]]) {
    if (o >= p.n) { return; }
    const uint oh = p.h / 2u;
    const uint ow = p.w / 2u;
    const uint x = o % ow;
    const uint y = (o / ow) % oh;
    const uint oc = o / (ow * oh);
    const uint c = oc / 4u;
    const uint i = (oc / 2u) % 2u;
    const uint j = oc % 2u;
    dst[o] = src[(c * p.h + 2u * y + i) * p.w + 2u * x + j];
}

kernel void pv_shuf(device const float* src [[buffer(0)]],
                    device const float* u1 [[buffer(1)]],
                    device const float* u2 [[buffer(2)]],
                    device float* dst [[buffer(3)]],
                    constant P& p [[buffer(4)]],
                    uint o [[thread_position_in_grid]]) {
    if (o >= p.n) { return; }
    const uint oh = 2u * p.h;
    const uint ow = 2u * p.w;
    const uint ox = o % ow;
    const uint oy = (o / ow) % oh;
    const uint c = o / (ow * oh);
    const uint ic = c * 4u + (oy % 2u) * 2u + (ox % 2u);
    float v = src[(ic * p.h + oy / 2u) * p.w + ox / 2u];
    if (p.flag != 0u) { v += dst[o]; }
    dst[o] = v;
}
"#;

struct Pipes {
    conv: ComputePipelineState,
    conv_sg: ComputePipelineState,
    norm: ComputePipelineState,
    unshuf: ComputePipelineState,
    shuf: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(MSL, &opts)
            .map_err(|e| format!("primal_vae MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            conv: pso("pv_conv")?,
            conv_sg: pso("pv_conv_sg")?,
            norm: pso("pv_norm")?,
            unshuf: pso("pv_unshuf")?,
            shuf: pso("pv_shuf")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("Primal VAE Metal kernels unavailable ({e}) — the host path runs");
            }
            None
        }
    }
}

struct DevWeights {
    key: u64,
    convs: Vec<(Buffer, Buffer)>,
    gammas: Vec<Buffer>,
    dummy: Buffer,
}

struct State {
    weights: Option<DevWeights>,
    bufs: Vec<(Buffer, usize)>,
}

// SAFETY: Metal buffers are thread-safe reference-counted objects; the
// mutex serializes every use.
unsafe impl Send for State {}

static STATE: Mutex<State> = Mutex::new(State { weights: None, bufs: Vec::new() });

fn shared(c: &Ctx, data: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// `CMF_PRIMAL_VAE_SG=1`: the simdgroup-matrix conv. Exact, but on an M4
/// it measured 2.7 s against the scalar tile's 1.27 s for a 1024² encode
/// (the gather, not the MMAs, bounds it), so the scalar tile is the default.
fn sg_on() -> bool {
    std::env::var("CMF_PRIMAL_VAE_SG").as_deref() == Ok("1")
}

/// Run `plan` on the device. `false` = refused; the caller runs the host.
pub fn run(key: u64, wts: &PvWeights, plan: &PvPlan, input: &[f32], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let max_buf = super::max_buffer_bytes() as usize;
    if max_buf != 0 && plan.buf_elems.iter().any(|&n| n * 4 > max_buf) {
        return false;
    }
    let mut st = STATE.lock().unwrap();
    if st.weights.as_ref().is_none_or(|w| w.key != key) {
        st.weights = Some(DevWeights {
            key,
            convs: wts.convs.iter().map(|cv| (shared(c, &cv.w), shared(c, &cv.b))).collect(),
            gammas: wts.gammas.iter().map(|g| shared(c, g)).collect(),
            dummy: shared(c, &[0.0]),
        });
    }
    while st.bufs.len() < plan.buf_elems.len() {
        st.bufs.push((c._device.new_buffer(16, MTLResourceOptions::StorageModePrivate), 4));
    }
    for (i, &n) in plan.buf_elems.iter().enumerate() {
        let need = n.max(1) * 4;
        if st.bufs[i].1 < need {
            // the input and output cross to the host; the rest stay private
            let mode = if i == plan.input as usize || i == plan.output as usize {
                MTLResourceOptions::StorageModeShared
            } else {
                MTLResourceOptions::StorageModePrivate
            };
            st.bufs[i] = (c._device.new_buffer(need as u64, mode), need);
        }
    }
    // A buffer that changed roles between plans may be private: give the
    // input/output slots shared storage whatever their history.
    for &i in &[plan.input as usize, plan.output as usize] {
        if st.bufs[i].0.storage_mode() != metal::MTLStorageMode::Shared {
            let need = st.bufs[i].1;
            st.bufs[i] = (c._device.new_buffer(need as u64, MTLResourceOptions::StorageModeShared), need);
        }
    }
    let st = &*st;
    let dw = st.weights.as_ref().expect("uploaded above");
    let buf = |i: u8| &st.bufs[i as usize].0;
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), buf(plan.input).contents() as *mut f32, input.len());
    }
    let t0 = std::time::Instant::now();
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for s in &plan.steps {
        let (pr, src, wb, bb, dst, pipe, tg, grid): ([u32; 8], u8, &Buffer, &Buffer, u8, &ComputePipelineState, MTLSize, MTLSize) = match *s {
            PvStep::Conv { wi, src, dst, res, dil, h, w } => {
                let cv = &wts.convs[wi];
                debug_assert!(res.is_none_or(|r| r == dst));
                let (wb, bb) = &dw.convs[wi];
                // The scalar tile (256 threads); `CMF_PRIMAL_VAE_SG=1` the
                // simdgroup-matrix one (128).
                let (pipe, threads) = if sg_on() { (&p.conv_sg, 128) } else { (&p.conv, 256) };
                (
                    [cv.cin as u32, cv.cout as u32, h as u32, w as u32, dil, res.is_some() as u32, (cv.cin * 9) as u32, 0],
                    src, wb, bb, dst, pipe,
                    MTLSize::new(threads, 1, 1),
                    MTLSize::new((h * w).div_ceil(64) as u64, cv.cout.div_ceil(64) as u64, 1),
                )
            }
            PvStep::NormSilu { gi, src, dst, c: ch, h, w } => (
                [ch as u32, 0, h as u32, w as u32, 0, 0, 0, 0],
                src, &dw.gammas[gi], &dw.dummy, dst, &p.norm,
                MTLSize::new(256, 1, 1),
                MTLSize::new((h * w).div_ceil(256) as u64, 1, 1),
            ),
            PvStep::Unshuffle { src, dst, c: ch, h, w } => (
                [ch as u32, 0, h as u32, w as u32, 0, 0, 0, (ch * h * w) as u32],
                src, &dw.dummy, &dw.dummy, dst, &p.unshuf,
                MTLSize::new(256, 1, 1),
                MTLSize::new((ch * h * w).div_ceil(256) as u64, 1, 1),
            ),
            PvStep::Shuffle { src, dst, add, c: ch, h, w } => {
                debug_assert!(add.is_none_or(|a| a == dst));
                (
                    [ch as u32, 0, h as u32, w as u32, 0, add.is_some() as u32, 0, (ch * h * w) as u32],
                    src, &dw.dummy, &dw.dummy, dst, &p.shuf,
                    MTLSize::new(256, 1, 1),
                    MTLSize::new((ch * h * w).div_ceil(256) as u64, 1, 1),
                )
            }
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf(src)), 0);
        enc.set_buffer(1, Some(wb), 0);
        enc.set_buffer(2, Some(bb), 0);
        enc.set_buffer(3, Some(buf(dst)), 0);
        enc.set_bytes(4, 32, pr.as_ptr() as *const c_void);
        enc.dispatch_thread_groups(grid, tg);
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != metal::MTLCommandBufferStatus::Completed {
        return false;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(buf(plan.output).contents() as *const f32, out.as_mut_ptr(), plan.out_elems);
    }
    if std::env::var_os("CMF_PRIMAL_VAE_PROF").is_some() {
        eprintln!(
            "primal_vae (Metal): {} steps, {:.3} s on the device",
            plan.steps.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    true
}
