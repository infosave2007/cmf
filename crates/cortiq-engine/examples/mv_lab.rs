//! q4tp decode-walk lab (Metal): times candidate matvec kernels over the
//! exact sequence of weight dispatches one decode token walks
//! (per layer q, k, v, o, gate, up, down; then the head), serialized in one
//! compute encoder the way the token graph runs them, GPU-timestamped.
//!
//!   cargo run --release --example mv_lab -- <model.cmf> [rounds]
//!
//! Variants alternate inside one process (round-robin), so thermal drift
//! hits every arm alike; the per-arm median is the number. Each variant's
//! output is checked against `cur` (the production kernel) on three tensors.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mv_lab: macOS/Metal-only diagnostic");
}

#[cfg(target_os = "macos")]
fn main() {
    lab::main()
}

#[cfg(target_os = "macos")]
mod lab {
    use cortiq_core::CmfModel;
    use metal::*;
    use std::sync::Arc;

    const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q4_dot8_fast(uint b, float4 x_lo, float4 x_hi) {
    float4 w_lo = float4((float)(b & 0xFu) - 8.0f,
                         (float)((b >> 4u) & 0xFu) - 8.0f,
                         (float)((b >> 8u) & 0xFu) - 8.0f,
                         (float)((b >> 12u) & 0xFu) - 8.0f);
    float4 w_hi = float4((float)((b >> 16u) & 0xFu) - 8.0f,
                          (float)((b >> 20u) & 0xFu) - 8.0f,
                          (float)((b >> 24u) & 0xFu) - 8.0f,
                          (float)(b >> 28u) - 8.0f);
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}

// Production kernel, verbatim.
kernel void cur(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    threadgroup float lad[8u * 4u * 32u];
    uint r0 = (tgpos * sgs + sg) * 4u;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, 4u) : 0u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * 4u + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        uint xb = g * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        uint bit = g * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < nr; ++ri) {
            uint r = r0 + ri;
            device const uint* p32 = (device const uint*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            uint b0 = p32[0], b1 = p32[1], b2 = p32[2], b3 = p32[3];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[(sg * 4u + ri) * 32u + code];
            float gsum = q4_dot8_fast(b0, x0, x1)
                       + q4_dot8_fast(b1, x2, x3)
                       + q4_dot8_fast(b2, x4, x5)
                       + q4_dot8_fast(b3, x6, x7);
            float contrib = scale * gsum;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// Same math and order; the row's ladder lives in a register per lane and
// the rung is fetched with simd_shuffle — no threadgroup memory, no
// barrier. Bitwise equal to `cur`.
kernel void shf(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    {
        device const half* ph = (device const half*)(q + params_off + (ulong)r0 * 4ul);
        float fl = (float)lane;
        l0 = exp2((float)ph[0] + fl * (float)ph[1]);
        if (nr > 1u) l1 = exp2((float)ph[2] + fl * (float)ph[3]);
        if (nr > 2u) l2 = exp2((float)ph[4] + fl * (float)ph[5]);
        if (nr > 3u) l3 = exp2((float)ph[6] + fl * (float)ph[7]);
    }
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g0 = 0u; g0 < gpr; g0 += 32u) {
        uint g = g0 + lane;
        bool on = g < gpr;
        uint gg = on ? g : 0u;
        uint xb = gg * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        uint bit = gg * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < 4u; ++ri) {
            uint r = r0 + min(ri, nr - 1u);
            device const uint* p32 = (device const uint*)(q + ((ulong)r * gpr + (ulong)gg) * 16ul);
            uint b0 = p32[0], b1 = p32[1], b2 = p32[2], b3 = p32[3];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float lr = (ri == 0u) ? l0 : (ri == 1u) ? l1 : (ri == 2u) ? l2 : l3;
            float scale = simd_shuffle(lr, (ushort)code);
            float gsum = q4_dot8_fast(b0, x0, x1)
                       + q4_dot8_fast(b1, x2, x3)
                       + q4_dot8_fast(b2, x4, x5)
                       + q4_dot8_fast(b3, x6, x7);
            float contrib = on ? scale * gsum : 0.0f;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// Two lanes per group (16 weights, 8 B each), sixteen groups per step:
// more lanes busy when gpr is small (64 groups at K=2048 is only two
// steps of `cur`). Different summation order: not bitwise.
kernel void hlf(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    {
        device const half* ph = (device const half*)(q + params_off + (ulong)r0 * 4ul);
        float fl = (float)lane;
        l0 = exp2((float)ph[0] + fl * (float)ph[1]);
        if (nr > 1u) l1 = exp2((float)ph[2] + fl * (float)ph[3]);
        if (nr > 2u) l2 = exp2((float)ph[4] + fl * (float)ph[5]);
        if (nr > 3u) l3 = exp2((float)ph[6] + fl * (float)ph[7]);
    }
    uint hh = lane & 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g0 = 0u; g0 < gpr; g0 += 16u) {
        uint g = g0 + (lane >> 1u);
        bool on = g < gpr;
        uint gg = on ? g : 0u;
        device const float4* xv = (device const float4*)(x + gg * 32u + hh * 16u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        uint bit = gg * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < 4u; ++ri) {
            uint r = r0 + min(ri, nr - 1u);
            device const uint2* p = (device const uint2*)(q + ((ulong)r * gpr + (ulong)gg) * 16ul + hh * 8u);
            uint2 b = p[0];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float lr = (ri == 0u) ? l0 : (ri == 1u) ? l1 : (ri == 2u) ? l2 : l3;
            float scale = simd_shuffle(lr, (ushort)code);
            float gsum = q4_dot8_fast(b.x, x0, x1) + q4_dot8_fast(b.y, x2, x3);
            float contrib = on ? scale * gsum : 0.0f;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// Row body shared by the single and the merged (q|k|v, gate|up) kernels:
// `hlf` math for four rows starting at r0 of a `rows`-row tensor at q.
inline void hlf_body(device const uchar* q, device const float* x, device float* y,
                     uint gpr, uint rows, uint r0, uint lane) {
    uint nr = min(rows - r0, 4u);
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    {
        device const half* ph = (device const half*)(q + params_off + (ulong)r0 * 4ul);
        float fl = (float)lane;
        l0 = exp2((float)ph[0] + fl * (float)ph[1]);
        if (nr > 1u) l1 = exp2((float)ph[2] + fl * (float)ph[3]);
        if (nr > 2u) l2 = exp2((float)ph[4] + fl * (float)ph[5]);
        if (nr > 3u) l3 = exp2((float)ph[6] + fl * (float)ph[7]);
    }
    uint hh = lane & 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g0 = 0u; g0 < gpr; g0 += 16u) {
        uint g = g0 + (lane >> 1u);
        bool on = g < gpr;
        uint gg = on ? g : 0u;
        device const float4* xv = (device const float4*)(x + gg * 32u + hh * 16u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        uint bit = gg * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < 4u; ++ri) {
            uint r = r0 + min(ri, nr - 1u);
            device const uint2* p = (device const uint2*)(q + ((ulong)r * gpr + (ulong)gg) * 16ul + hh * 8u);
            uint2 b = p[0];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float lr = (ri == 0u) ? l0 : (ri == 1u) ? l1 : (ri == 2u) ? l2 : l3;
            float scale = simd_shuffle(lr, (ushort)code);
            float gsum = q4_dot8_fast(b.x, x0, x1) + q4_dot8_fast(b.y, x2, x3);
            float contrib = on ? scale * gsum : 0.0f;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// Up to three matrices sharing one input, one dispatch: rows [0,ra) of A,
// then B, then C. Outputs land at y[0..ra), y[ra..ra+rb), ...
kernel void hlfm(
    device const uchar* qa   [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint4&     seg  [[buffer(4)]],
    device const uchar* qb   [[buffer(5)]],
    device const uchar* qc   [[buffer(6)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint R = (tgpos * sgs + sg) * 4u;
    uint ra = seg.x, rb = seg.y, rc = seg.z;
    if (R < ra) { hlf_body(qa, x, y, gpr, ra, R, lane); return; }
    R -= ra;
    if (R < rb) { hlf_body(qb, x, y + ra, gpr, rb, R, lane); return; }
    R -= rb;
    if (R < rc) { hlf_body(qc, x, y + ra + rb, gpr, rc, R, lane); }
}

// Masked-nibble dot (the llama.cpp trick): x is pre-scaled by 16^-k so a
// nibble is used in place — AND + convert + FMA per weight, no shift, no
// "-8" (folded into -8·Σx per group). Reassociates: not bitwise.
inline float4 nib4(uint u) {
    return float4((float)(u & 0xFu), (float)(u & 0xF0u), (float)(u & 0xF00u), (float)(u & 0xF000u));
}
kernel void lma(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    threadgroup float lad[8u * 4u * 32u];
    uint r0 = (tgpos * sgs + sg) * 4u;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, 4u) : 0u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * 4u + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        device const float4* xv = (device const float4*)(x + g * 32u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        float4 s4 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
        float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);
        x0 *= sc; x1 *= sc; x2 *= sc; x3 *= sc; x4 *= sc; x5 *= sc; x6 *= sc; x7 *= sc;
        uint bit = g * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < nr; ++ri) {
            uint r = r0 + ri;
            device const uint4* p = (device const uint4*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            uint4 b = p[0];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[(sg * 4u + ri) * 32u + code];
            float gsum = dot(nib4(b.x), x0) + dot(nib4(b.x >> 16), x1)
                       + dot(nib4(b.y), x2) + dot(nib4(b.y >> 16), x3)
                       + dot(nib4(b.z), x4) + dot(nib4(b.z >> 16), x5)
                       + dot(nib4(b.w), x6) + dot(nib4(b.w >> 16), x7);
            float contrib = scale * (gsum + m8);
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// `lma` math in the `hlf` shape (two lanes per group, 16 weights each):
// the llama.cpp q4_0 layout. RPS rows per simdgroup.
template <uint RPS>
inline void hma_body(device const uchar* q, device const float* x, device float* y,
                     uint gpr, uint rows, uint r0, uint lane) {
    uint nr = min(rows - r0, RPS);
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    float lad[RPS];
    float acc[RPS];
    for (uint ri = 0u; ri < RPS; ++ri) {
        uint r = r0 + min(ri, nr - 1u);
        device const half* ph = (device const half*)(q + params_off + (ulong)r * 4ul);
        lad[ri] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
        acc[ri] = 0.0f;
    }
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    uint hh = lane & 1u;
    for (uint g0 = 0u; g0 < gpr; g0 += 16u) {
        uint g = g0 + (lane >> 1u);
        bool on = g < gpr;
        uint gg = on ? g : 0u;
        device const float4* xv = (device const float4*)(x + gg * 32u + hh * 16u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 s4 = x0 + x1 + x2 + x3;
        float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);
        x0 *= sc; x1 *= sc; x2 *= sc; x3 *= sc;
        uint bit = gg * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < RPS; ++ri) {
            uint r = r0 + min(ri, nr - 1u);
            device const uint2* p = (device const uint2*)(q + ((ulong)r * gpr + (ulong)gg) * 16ul + hh * 8u);
            uint2 b = p[0];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = simd_shuffle(lad[ri], (ushort)code);
            float gsum = dot(nib4(b.x), x0) + dot(nib4(b.x >> 16), x1)
                       + dot(nib4(b.y), x2) + dot(nib4(b.y >> 16), x3);
            acc[ri] += on ? scale * (gsum + m8) : 0.0f;
        }
    }
    for (uint ri = 0u; ri < RPS; ++ri) {
        float a = simd_sum(acc[ri]);
        if (lane == 0u && ri < nr) y[r0 + ri] = a;
    }
}
kernel void hma(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    hma_body<4>(q, x, y, gpr, rows, r0, lane);
}
// Same with lane-per-group (32 weights a lane), templated rows.
template <uint RPS>
inline void lmb_body(device const uchar* q, device const float* x, device float* y,
                     uint gpr, uint rows, uint r0, uint lane) {
    uint nr = min(rows - r0, RPS);
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    float lad[RPS];
    float acc[RPS];
    for (uint ri = 0u; ri < RPS; ++ri) {
        uint r = r0 + min(ri, nr - 1u);
        device const half* ph = (device const half*)(q + params_off + (ulong)r * 4ul);
        lad[ri] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
        acc[ri] = 0.0f;
    }
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    for (uint g0 = 0u; g0 < gpr; g0 += 32u) {
        uint g = g0 + lane;
        bool on = g < gpr;
        uint gg = on ? g : 0u;
        device const float4* xv = (device const float4*)(x + gg * 32u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        float4 s4 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
        float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);
        x0 *= sc; x1 *= sc; x2 *= sc; x3 *= sc; x4 *= sc; x5 *= sc; x6 *= sc; x7 *= sc;
        uint bit = gg * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < RPS; ++ri) {
            uint r = r0 + min(ri, nr - 1u);
            device const uint4* p = (device const uint4*)(q + ((ulong)r * gpr + (ulong)gg) * 16ul);
            uint4 b = p[0];
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = simd_shuffle(lad[ri], (ushort)code);
            float gsum = dot(nib4(b.x), x0) + dot(nib4(b.x >> 16), x1)
                       + dot(nib4(b.y), x2) + dot(nib4(b.y >> 16), x3)
                       + dot(nib4(b.z), x4) + dot(nib4(b.z >> 16), x5)
                       + dot(nib4(b.w), x6) + dot(nib4(b.w >> 16), x7);
            acc[ri] += on ? scale * (gsum + m8) : 0.0f;
        }
    }
    for (uint ri = 0u; ri < RPS; ++ri) {
        float a = simd_sum(acc[ri]);
        if (lane == 0u && ri < nr) y[r0 + ri] = a;
    }
}
kernel void lmb(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    lmb_body<4>(q, x, y, gpr, rows, r0, lane);
}
kernel void lm8(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 8u;
    if (r0 >= rows) return;
    lmb_body<8>(q, x, y, gpr, rows, r0, lane);
}

// Loads only: the same addresses as `cur`, no dot products — the ceiling
// of the access pattern at this dispatch granularity.
kernel void ld(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    uint acc = 0u;
    for (uint g = lane; g < gpr; g += 32u) {
        for (uint ri = 0u; ri < nr; ++ri) {
            device const uint4* p = (device const uint4*)(q + ((ulong)(r0 + ri) * gpr + (ulong)g) * 16ul);
            uint4 b = p[0];
            acc ^= b.x ^ b.y ^ b.z ^ b.w;
        }
    }
    acc = simd_xor(acc);
    if (lane == 0u) y[r0] = (float)(acc & 1u);
}
"#;

    fn gpu_ms(cmd: &CommandBufferRef) -> f64 {
        use metal::objc::{msg_send, sel, sel_impl};
        let p: *mut metal::objc::runtime::Object =
            cmd as *const _ as *mut metal::objc::runtime::Object;
        let s: f64 = unsafe { msg_send![p, GPUStartTime] };
        let e: f64 = unsafe { msg_send![p, GPUEndTime] };
        (e - s) * 1e3
    }

    fn barrier(enc: &ComputeCommandEncoderRef) {
        use metal::objc::{msg_send, sel, sel_impl};
        // MTLBarrierScopeBuffers = 1
        let scope: u64 = 1;
        let _: () = unsafe { msg_send![enc, memoryBarrierWithScope: scope] };
    }

    #[derive(Clone, Copy)]
    struct T {
        abs: usize,
        rows: usize,
        gpr: usize,
        bytes: usize,
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let path = args
            .next()
            .expect("usage: mv_lab <model.cmf> [rounds] [variants]");
        let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(9);
        let pick: Option<String> = args.next();
        let model = Arc::new(CmfModel::open(&path).expect("open"));
        let dev = Device::system_default().expect("metal device");
        let queue = dev.new_command_queue();
        let lib = dev
            .new_library_with_source(SRC, &CompileOptions::new())
            .expect("compile");
        // (kernel, simdgroups per threadgroup, mode): s = serial encoder,
        // one dispatch per tensor; c = concurrent encoder with barriers
        // only between dependent groups; m = q|k|v and gate|up merged into
        // one dispatch each (serial encoder).
        let all: Vec<(&str, u64, char)> = vec![
            ("cur", 8, 's'),
            ("shf", 4, 's'),
            ("hlf", 8, 's'),
            ("hlf", 4, 's'),
            ("hlf", 8, 'c'),
            ("hlf", 4, 'c'),
            ("hlfm", 8, 'm'),
            ("hlfm", 4, 'm'),
            ("ld", 8, 's'),
            ("ld", 8, 'c'),
            ("lma", 8, 's'),
            ("lma", 8, 'c'),
            ("lma", 4, 'c'),
            ("lmb", 4, 'c'),
            ("lmb", 2, 'c'),
            ("lm8", 4, 'c'),
            ("lm8", 2, 'c'),
            ("hma", 4, 'c'),
            ("hma", 2, 'c'),
        ];
        let variants: Vec<(&str, u64, char)> = match &pick {
            Some(p) => {
                let keep: Vec<&str> = p.split(',').collect();
                all.into_iter()
                    .filter(|(n, s, m)| keep.contains(&format!("{n}{s}{m}").as_str()))
                    .collect()
            }
            None => all,
        };
        let psos: Vec<ComputePipelineState> = variants
            .iter()
            .map(|(n, _, _)| {
                let f = lib.get_function(n, None).expect("fn");
                dev.new_compute_pipeline_state_with_function(&f)
                    .expect("pso")
            })
            .collect();

        let bytes = model.primary_bytes();
        let page = 16384usize;
        let len = bytes.len() / page * page;
        let wbuf = dev.new_buffer_with_bytes_no_copy(
            bytes.as_ptr() as *const std::ffi::c_void,
            len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // The decode walk: per layer [q,k,v] [o] [gate,up] [down], then head.
        let nl = model.header.arch.num_layers;
        let get = |name: String| -> Option<T> {
            let e = model.tensor(&name)?;
            if e.dtype != cortiq_core::TensorDtype::Q4TiledP {
                return None;
            }
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let abs = model.entry_abs_offset(e).unwrap();
            let need = cortiq_core::quant::expected_nbytes(e.dtype, &[rows, cols]).unwrap();
            assert!(abs + need <= len);
            Some(T {
                abs,
                rows,
                gpr: cols / 32,
                bytes: need,
            })
        };
        let mut groups: Vec<Vec<T>> = Vec::new();
        for l in 0..nl {
            let p = |s: &str| get(format!("model.layers.{l}.{s}.weight")).expect("tensor");
            groups.push(vec![
                p("self_attn.q_proj"),
                p("self_attn.k_proj"),
                p("self_attn.v_proj"),
            ]);
            groups.push(vec![p("self_attn.o_proj")]);
            groups.push(vec![p("mlp.gate_proj"), p("mlp.up_proj")]);
            groups.push(vec![p("mlp.down_proj")]);
        }
        if let Some(h) = get("lm_head.weight".to_string()) {
            groups.push(vec![h]);
        }
        let flat: Vec<T> = groups.iter().flatten().copied().collect();
        let total: usize = flat.iter().map(|t| t.bytes).sum();
        let maxc = flat.iter().map(|t| t.gpr * 32).max().unwrap();
        let maxr = flat.iter().map(|t| t.rows).max().unwrap() + 20000;
        println!(
            "walk: {} tensors in {} groups, {:.3} GB",
            flat.len(),
            groups.len(),
            total as f64 / 1e9
        );

        let xs: Vec<f32> = (0..maxc)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let xbuf = dev.new_buffer_with_data(
            xs.as_ptr() as *const std::ffi::c_void,
            (maxc * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ybuf = dev.new_buffer((maxr * 4) as u64, MTLResourceOptions::StorageModeShared);

        // One dispatch over up to three tensors sharing gpr (merged kernel)
        // or exactly one tensor (single kernels).
        let enc_grp = |enc: &ComputeCommandEncoderRef, vi: usize, ts: &[T]| {
            let (name, sgs, _) = variants[vi];
            enc.set_compute_pipeline_state(&psos[vi]);
            enc.set_buffer(0, Some(&wbuf), ts[0].abs as u64);
            enc.set_buffer(1, Some(&xbuf), 0);
            enc.set_buffer(2, Some(&ybuf), 0);
            let g = ts[0].gpr as u32;
            enc.set_bytes(3, 4, &g as *const u32 as *const std::ffi::c_void);
            let rows_total: usize;
            if name == "hlfm" {
                let seg: [u32; 4] = [
                    ts[0].rows as u32,
                    ts.get(1).map(|t| t.rows as u32).unwrap_or(0),
                    ts.get(2).map(|t| t.rows as u32).unwrap_or(0),
                    0,
                ];
                enc.set_bytes(4, 16, seg.as_ptr() as *const std::ffi::c_void);
                enc.set_buffer(5, Some(&wbuf), ts.get(1).unwrap_or(&ts[0]).abs as u64);
                enc.set_buffer(6, Some(&wbuf), ts.get(2).unwrap_or(&ts[0]).abs as u64);
                rows_total = ts.iter().map(|t| t.rows).sum();
            } else {
                assert_eq!(ts.len(), 1);
                let r = ts[0].rows as u32;
                enc.set_bytes(4, 4, &r as *const u32 as *const std::ffi::c_void);
                rows_total = ts[0].rows;
            }
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (rows_total as u64).div_ceil(sgs * if name == "lm8" { 8 } else { 4 }),
                    1,
                    1,
                ),
                MTLSize::new(sgs * 32, 1, 1),
            );
        };
        let enc_walk = |cmd: &CommandBufferRef, vi: usize, gs: &[Vec<T>]| {
            let mode = variants[vi].2;
            let enc = if mode == 'c' {
                cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
            } else {
                cmd.new_compute_command_encoder()
            };
            for (gi, grp) in gs.iter().enumerate() {
                match mode {
                    'm' => enc_grp(enc, vi, grp),
                    'c' => {
                        if gi > 0 {
                            barrier(enc);
                        }
                        for t in grp {
                            enc_grp(enc, vi, std::slice::from_ref(t));
                        }
                    }
                    _ => {
                        for t in grp {
                            enc_grp(enc, vi, std::slice::from_ref(t));
                        }
                    }
                }
            }
            enc.end_encoding();
        };

        // Correctness on layer 0's groups and the head, against `cur` and `hlf`.
        let probe: Vec<Vec<T>> = vec![
            groups[0].clone(),
            groups[2].clone(),
            groups[3].clone(),
            groups.last().unwrap().clone(),
        ];
        let run1 = |vi: usize, grp: &[T]| -> Vec<f32> {
            if variants[vi].0 == "hlfm" {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc_grp(enc, vi, grp);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let n: usize = grp.iter().map(|t| t.rows).sum();
                let mut o = vec![0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(ybuf.contents() as *const f32, o.as_mut_ptr(), n)
                };
                return o;
            }
            let mut outs = Vec::new();
            for t in grp {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc_grp(enc, vi, std::slice::from_ref(t));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let mut o = vec![0f32; t.rows];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ybuf.contents() as *const f32,
                        o.as_mut_ptr(),
                        t.rows,
                    )
                };
                outs.extend(o);
            }
            outs
        };
        let cur_i = variants.iter().position(|v| v.0 == "cur");
        let hlf_i = variants.iter().position(|v| v.0 == "hlf");
        for (vi, v) in variants.iter().enumerate() {
            if v.0 == "ld" {
                continue;
            }
            for (lbl, ri) in [("cur", cur_i), ("hlf", hlf_i)] {
                let Some(ri) = ri else { continue };
                let (mut worst, mut exact) = (0f64, true);
                for grp in &probe {
                    let a = run1(ri, grp);
                    let b = run1(vi, grp);
                    let rms = (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / a.len() as f64)
                        .sqrt();
                    for (x, y) in a.iter().zip(&b) {
                        exact &= x.to_bits() == y.to_bits();
                        worst = worst.max(((x - y).abs() as f64) / rms);
                    }
                }
                println!(
                    "{}{}{}: vs {lbl} worst |d|/rms {:.2e}{}",
                    v.0,
                    v.1,
                    v.2,
                    worst,
                    if exact { " (bitwise)" } else { "" }
                );
            }
        }

        let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        for _ in 0..rounds {
            for vi in 0..variants.len() {
                let cmd = queue.new_command_buffer();
                enc_walk(cmd, vi, &groups);
                cmd.commit();
                cmd.wait_until_completed();
                times[vi].push(gpu_ms(cmd));
            }
        }
        for (vi, v) in variants.iter().enumerate() {
            let mut t = times[vi].clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = t[t.len() / 2];
            // Ratio to arm 0 within the same round (adjacent in time), so
            // a drifting clock cancels; the median ratio is the verdict.
            let mut r: Vec<f64> = times[vi]
                .iter()
                .zip(&times[0])
                .map(|(a, b)| a / b)
                .collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{:>5}{}{} walk median {:6.2} ms (min {:6.2})  {:5.1} GB/s | vs arm0 {:.3} (q1 {:.3} q3 {:.3})",
                v.0,
                v.1,
                v.2,
                med,
                t[0],
                total as f64 / 1e6 / med,
                r[r.len() / 2],
                r[r.len() / 4],
                r[3 * r.len() / 4]
            );
        }
    }
}
