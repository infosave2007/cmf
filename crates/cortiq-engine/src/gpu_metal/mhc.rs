//! SenseNova-Loop's mHC streams inside the Metal token graph — the MSL twin
//! of `gpu_wgpu::mhc` (two streams):
//!
//! - `mhc_route` (one threadgroup of 256): optionally copies the hidden into
//!   both streams (the range's first layer), the RMS of the flat `[2·d]`
//!   streams, the router's 8 logits, `pre = σ`, `post = 2σ`, the 2×2
//!   Sinkhorn in log space, the branch `Σ pre_s·r_s · pre_scale` into the
//!   hidden buffer, `post`/`mix` into the coefficient buffer;
//! - `mhc_merge` (one thread per lane): `r_i ← Σ_j mix_ij·r_j + post_i·y`,
//!   and after the range's last FFN the hidden becomes the streams' mean.
//!
//! Both take rows (the rows graph): `mhc_route` one threadgroup a row, `mhc_merge`
//! one thread per lane of every row, the streams `[rows][2][d]`, a row's
//! coefficients 8 floats.

use super::{Ctx, ctx};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct MP { uint d; uint s; uint outn; uint flags; uint iters; float pre_scale; uint pad0; uint pad1; };

static inline float sig(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float lse2(float a, float b) { float m = max(a, b); return m + log(exp(a - m) + exp(b - m)); }

kernel void mhc_route(device float* h [[buffer(0)]],
                      device float* r [[buffer(1)]],
                      device const float* w [[buffer(2)]],
                      device const float* bias [[buffer(3)]],
                      device float* coef [[buffer(4)]],
                      constant MP& p [[buffer(5)]],
                      uint t [[thread_index_in_threadgroup]],
                      uint row [[threadgroup_position_in_grid]]) {
    threadgroup float red[9 * 256];
    threadgroup float pre_s[2];
    const uint d = p.d;
    const uint n = 2u * d;
    h += row * d;
    r += row * n;
    coef += row * 8u;
    if ((p.flags & 1u) != 0u) {
        for (uint e = t; e < n; e += 256u) { r[e] = h[e % d]; }
    }
    threadgroup_barrier(mem_flags::mem_device);
    float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint e = t; e < n; e += 256u) {
        const float x = r[e];
        acc[0] += x * x;
        for (uint k = 0u; k < 8u; k++) { acc[k + 1u] += w[k * n + e] * x; }
    }
    for (uint k = 0u; k < 9u; k++) { red[k * 256u + t] = acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (t < st) {
            for (uint k = 0u; k < 9u; k++) { red[k * 256u + t] += red[k * 256u + t + st]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
        const float inv = rsqrt(red[0] / float(n) + 1e-6f);
        float l[8];
        for (uint k = 0u; k < 8u; k++) { l[k] = red[(k + 1u) * 256u] * inv + bias[k]; }
        pre_s[0] = sig(l[0]);
        pre_s[1] = sig(l[1]);
        coef[0] = 2.0f * sig(l[2]);
        coef[1] = 2.0f * sig(l[3]);
        float m00 = l[4], m01 = l[5], m10 = l[6], m11 = l[7];
        for (uint it = 0u; it < p.iters; it++) {
            const float r0 = lse2(m00, m01); m00 -= r0; m01 -= r0;
            const float r1 = lse2(m10, m11); m10 -= r1; m11 -= r1;
            const float c0 = lse2(m00, m10); m00 -= c0; m10 -= c0;
            const float c1 = lse2(m01, m11); m01 -= c1; m11 -= c1;
        }
        coef[2] = exp(m00); coef[3] = exp(m01); coef[4] = exp(m10); coef[5] = exp(m11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float p0 = pre_s[0] * p.pre_scale;
    const float p1 = pre_s[1] * p.pre_scale;
    for (uint e = t; e < d; e += 256u) { h[e] = p0 * r[e] + p1 * r[d + e]; }
}

kernel void mhc_merge(device float* h [[buffer(0)]],
                      device float* r [[buffer(1)]],
                      device const float* y [[buffer(2)]],
                      device const float* coef [[buffer(3)]],
                      constant MP& p [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.d * max(p.pad0, 1u)) { return; }
    const uint row = i / p.d;
    const uint e = i % p.d;
    device float* rr = r + row * 2u * p.d;
    device const float* cf = coef + row * 8u;
    const float yv = y[i];
    const float r0 = rr[e];
    const float r1 = rr[p.d + e];
    const float n0 = cf[2] * r0 + cf[3] * r1 + cf[0] * yv;
    const float n1 = cf[4] * r0 + cf[5] * r1 + cf[1] * yv;
    rr[e] = n0;
    rr[p.d + e] = n1;
    if ((p.flags & 2u) != 0u) { h[i] = 0.5f * (n0 + n1); }
}
"#;

struct Pipes {
    route: ComputePipelineState,
    merge: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = c._device.new_library_with_source(MSL, &opts).map_err(|e| format!("mHC MSL: {e}"))?;
            let pso = |name: &str| -> Result<ComputePipelineState, String> {
                let f = lib.get_function(name, None).map_err(|e| format!("{name}: {e}"))?;
                c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("{name}: {e}"))
            };
            Ok(Pipes { route: pso("mhc_route")?, merge: pso("mhc_merge")? })
        })
        .as_ref()
        .ok()
}

/// Whether the kernels compiled (the graph builder asks before planning
/// an mHC layer on the device).
pub fn available() -> bool {
    ctx().is_some_and(|c| pipes(c).is_some())
}

struct Bufs(usize, Buffer, Buffer);
// SAFETY: Metal buffers are thread-safe reference-counted objects.
unsafe impl Send for Bufs {}
static BUFS: Mutex<Option<Bufs>> = Mutex::new(None);

/// The streams (`[2·d]` f32) and coefficient buffers, kept.
pub(crate) fn bufs(c: &Ctx, d: usize) -> (Buffer, Buffer) {
    let mut g = BUFS.lock().unwrap();
    if let Some(Bufs(n, r, k)) = g.as_ref() {
        if *n == d {
            return (r.clone(), k.clone());
        }
    }
    let r = c._device.new_buffer((2 * d * 4) as u64, MTLResourceOptions::StorageModePrivate);
    let k = c._device.new_buffer(64, MTLResourceOptions::StorageModePrivate);
    *g = Some(Bufs(d, r.clone(), k.clone()));
    (r, k)
}

/// The uniform words: `[d, 2, 8, flags, iters, pre_scale, rows, 0]`, flags bit 0 =
/// enter (the hidden into both streams), bit 1 = leave (the hidden becomes their mean).
fn params(d: usize, rows: usize, enter: bool, leave: bool, iters: usize, pre_scale: f32) -> [u32; 8] {
    [d as u32, 2, 8, enter as u32 | (leave as u32) << 1, iters as u32, pre_scale.to_bits(), rows as u32, 0]
}

/// `mhc_route` into `h` with router `(w, b)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_route(c: &Ctx, enc: &ComputeCommandEncoderRef, h: &Buffer, w: &Buffer, b: &Buffer, d: usize, enter: bool, iters: usize, pre_scale: f32) {
    let (r, k) = bufs(c, d);
    encode_route_rows(c, enc, h, &r, &k, w, b, d, 1, enter, iters, pre_scale);
}

/// `mhc_merge` of the sub-layer output `y`.
pub(crate) fn encode_merge(c: &Ctx, enc: &ComputeCommandEncoderRef, h: &Buffer, y: &Buffer, d: usize, leave: bool) {
    let (r, k) = bufs(c, d);
    encode_merge_rows(c, enc, h, &r, &k, y, d, 1, leave);
}

/// The rows twin of `encode_route`: `rows` rows of `h` (`[rows][d]`) out of the streams
/// `r` (`[rows][2][d]`), their coefficients into `k` (`[rows][8]`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_route_rows(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    h: &Buffer,
    r: &Buffer,
    k: &Buffer,
    w: &Buffer,
    b: &Buffer,
    d: usize,
    rows: usize,
    enter: bool,
    iters: usize,
    pre_scale: f32,
) {
    let p = pipes(c).expect("mHC kernels (checked by the planner)");
    let pr = params(d, rows, enter, false, iters, pre_scale);
    enc.set_compute_pipeline_state(&p.route);
    enc.set_buffer(0, Some(h), 0);
    enc.set_buffer(1, Some(r), 0);
    enc.set_buffer(2, Some(w), 0);
    enc.set_buffer(3, Some(b), 0);
    enc.set_buffer(4, Some(k), 0);
    enc.set_bytes(5, 32, pr.as_ptr() as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(rows.max(1) as u64, 1, 1), MTLSize::new(256, 1, 1));
}

/// The rows twin of `encode_merge`: `y` is `[rows][d]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_merge_rows(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    h: &Buffer,
    r: &Buffer,
    k: &Buffer,
    y: &Buffer,
    d: usize,
    rows: usize,
    leave: bool,
) {
    let p = pipes(c).expect("mHC kernels (checked by the planner)");
    let pr = params(d, rows, false, leave, 0, 1.0);
    enc.set_compute_pipeline_state(&p.merge);
    enc.set_buffer(0, Some(h), 0);
    enc.set_buffer(1, Some(r), 0);
    enc.set_buffer(2, Some(y), 0);
    enc.set_buffer(3, Some(k), 0);
    enc.set_bytes(4, 32, pr.as_ptr() as *const c_void);
    enc.dispatch_threads(MTLSize::new((d * rows.max(1)) as u64, 1, 1), MTLSize::new(256, 1, 1));
}
