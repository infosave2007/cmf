// The resident vision towers (`gpu_metal::vit`): the row ops around the
// encoders' GEMMs (`egemma2_msl.metal`) and SAM's window kernels.
#include <metal_stdlib>
using namespace metal;

struct VtRow {
    uint rows;
    uint cols;
    uint ldx;
    uint ldy;
    float eps;
    uint act;
    uint hd;
    uint hs;
};

// One simdgroup per row (4 rows per 128-thread group), float4 lanes.

// Y = LayerNorm(X)·w + b
kernel void vt_ln(
    device const float* X [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant VtRow& p [[buffer(4)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.rows) return;
    device const float* x = X + (ulong)row * p.ldx;
    device float* y = Y + (ulong)row * p.ldy;
    float s = 0.0f;
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        float4 v = *(device const float4*)(x + i);
        s += v.x + v.y + v.z + v.w;
    }
    const float mean = simd_sum(s) / (float)p.cols;
    float ss = 0.0f;
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        float4 v = *(device const float4*)(x + i) - mean;
        ss += dot(v, v);
    }
    const float inv = rsqrt(simd_sum(ss) / (float)p.cols + p.eps);
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        float4 v = (*(device const float4*)(x + i) - mean) * inv;
        *(device float4*)(y + i) = v * *(device const float4*)(w + i) + *(device const float4*)(b + i);
    }
}

// erf by Abramowitz–Stegun 7.1.26
static inline float4 vt_erf(float4 x) {
    float4 a = abs(x);
    float4 t = 1.0f / (1.0f + 0.3275911f * a);
    float4 poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign(x) * (1.0f - poly * exp(-a * a));
}

// y += b, then the activation (0 none, 1 exact GELU, 2 tanh GELU, 3 quick
// GELU). A row wider than `cols` has its tail zeroed: it is the k of the next
// GEMM, which takes multiples of 32.
kernel void vt_bias_act(
    device float* Y [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant VtRow& p [[buffer(2)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.rows) return;
    device float* y = Y + (ulong)row * p.ldy;
    for (uint i = lane * 4u; i < p.ldy; i += 128u) {
        if (i >= p.cols) {
            *(device float4*)(y + i) = float4(0.0f);
            continue;
        }
        float4 v = *(device const float4*)(y + i) + *(device const float4*)(b + i);
        if (p.act == 1u) {
            v = 0.5f * v * (1.0f + vt_erf(v * 0.70710678118654752f));
        } else if (p.act == 2u) {
            float4 u = clamp(0.7978845608f * (v + 0.044715f * v * v * v), -15.0f, 15.0f);
            v = 0.5f * v * (1.0f + tanh(u));
        } else if (p.act == 3u) {
            v = v / (1.0f + exp(-1.702f * v));
        }
        *(device float4*)(y + i) = v;
    }
}

// x += y + b
kernel void vt_add(
    device float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device const float* b [[buffer(2)]],
    constant VtRow& p [[buffer(3)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.rows) return;
    device float* x = X + (ulong)row * p.ldx;
    device const float* y = Y + (ulong)row * p.ldy;
    for (uint i = lane * 4u; i < p.cols; i += 128u) {
        *(device float4*)(x + i) += *(device const float4*)(y + i) + *(device const float4*)(b + i);
    }
}

// The q part of `[rows][3d]` into `[rows][nh·hs]` with zeroed head tails, for
// a head width that is not a multiple of 32 (ldx = 3d, ldy = nh·hs, cols = nh).
kernel void vt_pad_q(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant VtRow& p [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    device const float* x = X + (ulong)row * p.ldx;
    device float* y = Y + (ulong)row * p.ldy;
    for (uint h = 0u; h < p.cols; ++h) {
        for (uint j = 0u; j < p.hs; j += 4u) {
            *(device float4*)(y + h * p.hs + j) =
                j < p.hd ? *(device const float4*)(x + h * p.hd + j) : float4(0.0f);
        }
    }
}

struct SamP {
    uint rows;
    uint d;
    uint g;
    uint w;
    uint nh;
    uint hd;
    uint ld;
    float scale;
};

// Window partition: window rows from the grid, zero padded.
kernel void sam_part(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant SamP& p [[buffer(2)]],
    uint r [[thread_position_in_grid]])
{
    const uint wn = (p.g + p.w - 1u) / p.w;
    const uint win = r / (p.w * p.w);
    const uint local = r % (p.w * p.w);
    const uint yy = (win / wn) * p.w + local / p.w;
    const uint xx = (win % wn) * p.w + local % p.w;
    device float* y = Y + (ulong)r * p.d;
    if (yy >= p.g || xx >= p.g) {
        for (uint i = 0u; i < p.d; i += 4u) *(device float4*)(y + i) = float4(0.0f);
        return;
    }
    device const float* x = X + (ulong)(yy * p.g + xx) * p.d;
    for (uint i = 0u; i < p.d; i += 4u) *(device float4*)(y + i) = *(device const float4*)(x + i);
}

// The inverse, added to the grid with a bias.
kernel void sam_unpart(
    device float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device const float* b [[buffer(2)]],
    constant SamP& p [[buffer(3)]],
    uint r [[thread_position_in_grid]])
{
    const uint wn = (p.g + p.w - 1u) / p.w;
    const uint yy = r / p.g;
    const uint xx = r % p.g;
    const uint win = (yy / p.w) * wn + xx / p.w;
    const uint src = win * p.w * p.w + (yy % p.w) * p.w + xx % p.w;
    device float* x = X + (ulong)r * p.d;
    device const float* y = Y + (ulong)src * p.d;
    for (uint i = 0u; i < p.d; i += 4u) {
        *(device float4*)(x + i) += *(device const float4*)(y + i) + *(device const float4*)(b + i);
    }
}

// Windowed attention with decomposed relative positions, over the `[rows][q|k|v]`
// rows of `ld = 3d`: one thread per (head, window row), `[rows][nh·hd]` out.
kernel void sam_win(
    device const float* QKV [[buffer(0)]],
    device const float* rel [[buffer(1)]],
    device float* O [[buffer(2)]],
    constant SamP& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint h = gid.x;
    const uint r = gid.y;
    const uint s = p.w;
    const uint n = s * s;
    const uint win = r / n;
    const uint qi = r % n;
    const uint wn = (p.g + s - 1u) / s;
    if ((win / wn) * s + qi / s >= p.g || (win % wn) * s + qi % s >= p.g) return;
    const uint hd = p.hd;
    device const float* q = QKV + (ulong)r * p.ld + h * hd;
    device const float* kb = QKV + (ulong)(win * n) * p.ld + p.d + h * hd;
    float rh[32];
    float rw[32];
    const uint qy = qi / s;
    const uint qx = qi % s;
    const uint rl = 2u * s - 1u;
    for (uint k = 0u; k < s; ++k) {
        device const float* ry = rel + (qy + s - 1u - k) * hd;
        device const float* rx = rel + (rl + qx + s - 1u - k) * hd;
        float a = 0.0f;
        float b = 0.0f;
        for (uint e = 0u; e < hd; e += 4u) {
            float4 v = *(device const float4*)(q + e);
            a += dot(v, *(device const float4*)(ry + e));
            b += dot(v, *(device const float4*)(rx + e));
        }
        rh[k] = a;
        rw[k] = b;
    }
    float sc[256];
    float mx = -INFINITY;
    for (uint k = 0u; k < n; ++k) {
        device const float* kk = kb + (ulong)k * p.ld;
        float a = 0.0f;
        for (uint e = 0u; e < hd; e += 4u) {
            a += dot(*(device const float4*)(q + e), *(device const float4*)(kk + e));
        }
        const float v = a * p.scale + rh[k / s] + rw[k % s];
        sc[k] = v;
        mx = max(mx, v);
    }
    float den = 0.0f;
    for (uint k = 0u; k < n; ++k) {
        sc[k] = exp(sc[k] - mx);
        den += sc[k];
    }
    const float inv = 1.0f / den;
    device float* o = O + (ulong)r * p.d + h * hd;
    for (uint e = 0u; e < hd; e += 4u) {
        float4 a = float4(0.0f);
        for (uint k = 0u; k < n; ++k) {
            a += sc[k] * *(device const float4*)(kb + (ulong)k * p.ld + p.d + e);
        }
        *(device float4*)(o + e) = a * inv;
    }
}

struct SamBias {
    uint s_off;
    uint nq;
    uint lds;
    uint q0;
    uint h;
    uint pad0;
    uint pad1;
    uint pad2;
};

// The relative bias of a global attention's score rows, `S = Q·Kᵀ` scaled.
kernel void sam_bias(
    device const float* Q [[buffer(0)]],
    device const float* rel [[buffer(1)]],
    device float* S [[buffer(2)]],
    device const SamBias* items [[buffer(3)]],
    constant SamP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    device const SamBias& it = items[gid.y];
    if (gid.x >= it.nq) return;
    const uint s = p.g;
    const uint n = s * s;
    const uint hd = p.hd;
    const uint qi = it.q0 + gid.x;
    device const float* q = Q + (ulong)qi * p.ld + it.h * hd;
    float rh[128];
    float rw[128];
    const uint qy = qi / s;
    const uint qx = qi % s;
    const uint rl = 2u * s - 1u;
    for (uint k = 0u; k < s; ++k) {
        device const float* ry = rel + (qy + s - 1u - k) * hd;
        device const float* rx = rel + (rl + qx + s - 1u - k) * hd;
        float a = 0.0f;
        float b = 0.0f;
        for (uint e = 0u; e < hd; e += 4u) {
            float4 v = *(device const float4*)(q + e);
            a += dot(v, *(device const float4*)(ry + e));
            b += dot(v, *(device const float4*)(rx + e));
        }
        rh[k] = a;
        rw[k] = b;
    }
    device float* row = S + it.s_off + (ulong)gid.x * it.lds;
    for (uint k = 0u; k < n; ++k) row[k] += rh[k / s] + rw[k % s];
}

struct ConvP {
    uint g;
    uint ci;
    uint co;
    uint k;
    uint stride;
    uint pad;
    uint g2;
    uint _a;
};

// A bias-free conv over an HWC grid, weights `[co, k·k, ci]`: one thread per
// output channel of a cell.
kernel void sam_conv(
    device const float* w [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant ConvP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    const uint cell = idx / p.co;
    const uint c = idx % p.co;
    const uint oy = cell / p.g2;
    const uint ox = cell % p.g2;
    float acc = 0.0f;
    for (uint ky = 0u; ky < p.k; ++ky) {
        const uint iy = oy * p.stride + ky;
        if (iy < p.pad || iy - p.pad >= p.g) continue;
        for (uint kx = 0u; kx < p.k; ++kx) {
            const uint ix = ox * p.stride + kx;
            if (ix < p.pad || ix - p.pad >= p.g) continue;
            device const float* xr = X + (ulong)((iy - p.pad) * p.g + ix - p.pad) * p.ci;
            device const float* wr = w + (ulong)(c * p.k * p.k + ky * p.k + kx) * p.ci;
            for (uint e = 0u; e < p.ci; e += 4u) {
                acc += dot(*(device const float4*)(wr + e), *(device const float4*)(xr + e));
            }
        }
    }
    Y[idx] = acc;
}
