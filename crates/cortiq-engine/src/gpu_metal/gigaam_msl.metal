// The GigaAM encoder resident on Metal (`gpu_metal::gigaam`): the kernels
// between the encoders' GEMMs (`egemma2_msl.metal`) — biases and SiLU, the
// residual adds, the row norms, the rotation of q and k with the value and
// output gates, the convolution module's tail. f32 throughout; every row
// width is a multiple of 4 (float4 lanes).
#include <metal_stdlib>
using namespace metal;

struct GaP {
    uint rows;
    uint cols;
    uint flags;
    uint aux;
    float f0;
    float f1;
    float f2;
    float f3;
};

// the host's exp clamps the same way: no infinity under fast math
static inline float4 ga_sigm(float4 x) {
    return 1.0f / (1.0f + precise::exp(clamp(-x, -87.0f, 88.0f)));
}

// y += b a row (flags bit 0), then SiLU (bit 1).
kernel void ga_bias_act(
    device float* Y [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant GaP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x * 4u;
    if (i >= p.cols || gid.y >= p.rows) return;
    device float4* y = (device float4*)(Y + (ulong)gid.y * p.cols + i);
    float4 v = *y;
    if ((p.flags & 1u) != 0u) v += *(device const float4*)(b + i);
    if ((p.flags & 2u) != 0u) v = v * ga_sigm(v);
    *y = v;
}

// D = the sum of `aux` partial products, each [rows][cols], one after
// another in Y (a product of few rows runs its k in parts side by side).
kernel void ga_sum(
    device const float* Y [[buffer(0)]],
    device float* D [[buffer(1)]],
    constant GaP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x * 4u;
    if (i >= p.cols || gid.y >= p.rows) return;
    const ulong at = (ulong)gid.y * p.cols + i, plane = (ulong)p.rows * p.cols;
    float4 v = *(device const float4*)(Y + at);
    for (uint z = 1u; z < p.aux; ++z) v += *(device const float4*)(Y + z * plane + at);
    *(device float4*)(D + at) = v;
}

// D = A + Y (+ b a row, flags bit 0); D may be A.
kernel void ga_add(
    device const float* A [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* D [[buffer(3)]],
    constant GaP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x * 4u;
    if (i >= p.cols || gid.y >= p.rows) return;
    const ulong at = (ulong)gid.y * p.cols + i;
    float4 v = *(device const float4*)(A + at) + *(device const float4*)(Y + at);
    if ((p.flags & 1u) != 0u) v += *(device const float4*)(b + i);
    *(device float4*)(D + at) = v;
}

// B = f1·H, then H = f0·H: a stack's bypass and its scaled input.
kernel void ga_scale2(
    device float* H [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant GaP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x * 4u;
    if (i >= p.cols || gid.y >= p.rows) return;
    const ulong at = (ulong)gid.y * p.cols + i;
    const float4 v = *(device const float4*)(H + at);
    *(device float4*)(B + at) = p.f1 * v;
    *(device float4*)(H + at) = p.f0 * v;
}

// D = (R + f1·(S − R))·rsqrt(mean(.²) + f0)·w, one simdgroup a row (4 rows a
// group of 128); f1 = 1 with S = R is the plain row norm, and D may be R.
// flags bit 0: the scalar f2 for every lane instead of w (the output norm);
// bit 1: B += f3·D after (the stack's bypass).
kernel void ga_rms(
    device const float* R [[buffer(0)]],
    device const float* S [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device float* D [[buffer(3)]],
    device float* B [[buffer(4)]],
    constant GaP& p [[buffer(5)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.rows) return;
    const ulong at = (ulong)row * p.cols;
    const float mix = p.f1;
    float ss = 0.0f;
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        const float4 r = *(device const float4*)(R + at + i);
        const float4 v = r + mix * (*(device const float4*)(S + at + i) - r);
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    const bool scalar = (p.flags & 1u) != 0u, by = (p.flags & 2u) != 0u;
    float k = rsqrt(ss / (float)p.cols + p.f0);
    if (scalar) k *= p.f2;
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        const float4 r = *(device const float4*)(R + at + i);
        float4 v = (r + mix * (*(device const float4*)(S + at + i) - r)) * k;
        if (!scalar) v *= *(device const float4*)(w + i);
        *(device float4*)(D + at + i) = v;
        if (by) *(device float4*)(B + at + i) += p.f3 * v;
    }
}

// The attention operands in place. QK rows [n][2·hw] (q, then k; hw = cols):
// the bias (flags bit 0), then neighbour pairs rotated by the row's angles
// cs / sn [n][hd / 2] (hd = aux). VG rows [n][3·hw] (v, its gate, the output
// gate): the bias (bit 1), v ← v·σ(gate), and the output gate ← σ(output
// gate) for `ga_gate`.
kernel void ga_prep(
    device float* QK [[buffer(0)]],
    device float* VG [[buffer(1)]],
    device const float* bqk [[buffer(2)]],
    device const float* bvg [[buffer(3)]],
    device const float* cs [[buffer(4)]],
    device const float* sn [[buffer(5)]],
    constant GaP& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint hw = p.cols, hd = p.aux, hp = hd / 2u;
    const uint i = gid.x * 4u, row = gid.y;
    if (i >= hw || row >= p.rows) return;
    const uint pr = (i % hd) / 2u;
    const float2 c = *(device const float2*)(cs + (ulong)row * hp + pr);
    const float2 s = *(device const float2*)(sn + (ulong)row * hp + pr);
    const float4 cc = c.xxyy, sv = s.xxyy * float4(-1.0f, 1.0f, -1.0f, 1.0f);
    device float* q = QK + (ulong)row * 2u * hw + i;
    float4 qv = *(device const float4*)q, kv = *(device const float4*)(q + hw);
    if ((p.flags & 1u) != 0u) {
        qv += *(device const float4*)(bqk + i);
        kv += *(device const float4*)(bqk + hw + i);
    }
    // (x0, x1) → (x0·c − x1·s, x0·s + x1·c)
    *(device float4*)q = qv * cc + qv.yxwz * sv;
    *(device float4*)(q + hw) = kv * cc + kv.yxwz * sv;
    device float* v = VG + (ulong)row * 3u * hw + i;
    float4 vv = *(device const float4*)v, gi = *(device const float4*)(v + hw), go = *(device const float4*)(v + 2u * hw);
    if ((p.flags & 2u) != 0u) {
        vv += *(device const float4*)(bvg + i);
        gi += *(device const float4*)(bvg + hw + i);
        go += *(device const float4*)(bvg + 2u * hw + i);
    }
    *(device float4*)v = vv * ga_sigm(gi);
    *(device float4*)(v + 2u * hw) = ga_sigm(go);
}

// The attention context [n][hw] times the output gate `ga_prep` left in the
// last third of VG.
kernel void ga_gate(
    device float* C [[buffer(0)]],
    device const float* VG [[buffer(1)]],
    constant GaP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint hw = p.cols, i = gid.x * 4u, row = gid.y;
    if (i >= hw || row >= p.rows) return;
    *(device float4*)(C + (ulong)row * hw + i) *= *(device const float4*)(VG + (ulong)row * 3u * hw + 2u * hw + i);
}

// mean[c] = (Σ over frames of CG[frame][c] / n + b[c])·mean_w[c]; CG rows are
// 2·d wide (the stream, then its gate; d = cols), b the product's bias
// (flags bit 0), which `ga_conv` adds too.
kernel void ga_mean(
    device const float* CG [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* mean_w [[buffer(2)]],
    device float* mean [[buffer(3)]],
    constant GaP& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint c = gid * 4u, d = p.cols;
    if (c >= d) return;
    float4 acc = float4(0.0f);
    for (uint r = 0u; r < p.rows; ++r) acc += *(device const float4*)(CG + (ulong)r * 2u * d + c);
    acc /= (float)p.rows;
    if ((p.flags & 1u) != 0u) acc += *(device const float4*)(b + c);
    *(device float4*)(mean + c) = acc * *(device const float4*)(mean_w + c);
}

// Z[frame][c] = silu((conv + mean[c])·σ(gate)): the depthwise convolution
// over time of the stream half of CG, taps [k][d] (k = aux) with (k − 1) / 2
// zero frames on both sides.
kernel void ga_conv(
    device const float* CG [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const float* mean [[buffer(4)]],
    device float* Z [[buffer(5)]],
    constant GaP& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x * 4u, row = gid.y, d = p.cols, n = p.rows, k = p.aux;
    if (c >= d || row >= n) return;
    const int side = (int)((k - 1u) / 2u);
    float4 bs = float4(0.0f), bg = float4(0.0f);
    if ((p.flags & 1u) != 0u) {
        bs = *(device const float4*)(b + c);
        bg = *(device const float4*)(b + d + c);
    }
    float4 acc = *(device const float4*)(bias + c);
    for (uint j = 0u; j < k; ++j) {
        const int src = (int)row + (int)j - side;
        if (src < 0 || src >= (int)n) continue;
        acc += *(device const float4*)(taps + (ulong)j * d + c) * (*(device const float4*)(CG + (ulong)src * 2u * d + c) + bs);
    }
    const float4 v = (acc + *(device const float4*)(mean + c)) * ga_sigm(*(device const float4*)(CG + (ulong)row * 2u * d + d + c) + bg);
    *(device float4*)(Z + (ulong)row * d + c) = v * ga_sigm(v);
}
