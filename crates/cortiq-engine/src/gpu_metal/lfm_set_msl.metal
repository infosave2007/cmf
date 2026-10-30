#include <metal_stdlib>
using namespace metal;

// d1's one-pass forward over a packed row (`lfm_set.rs`): row-wise norms, the
// short convolution walked over the positions, Q/K norm + RoPE per row, the
// `set` attention and the per-head output gate. The GEMMs between them are
// the main library's (q4tp) and `moe_q8`'s (q8_2f).

// o = rms(x)·w per row (x̂·(1+w) when `gemma`), one 256-thread group a row.
// In place is safe: every element is read and written by one thread.
kernel void ls_rms_rows(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       o   [[buffer(2)]],
    constant uint&  n       [[buffer(3)]],
    constant uint&  gemma   [[buffer(4)]],
    constant float& eps     [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    device const float* xr = x + (ulong)row * n;
    device float* orow = o + (ulong)row * n;
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256u) { float v = xr[i]; acc += v * v; }
    acc = simd_sum(acc);
    if (lane == 0u) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float inv = rsqrt(tot / (float)n + eps);
    for (uint i = tid; i < n; i += 256u) {
        float wv = w[i];
        if (gemma != 0u) wv += 1.0f;
        orow[i] = xr[i] * inv * wv;
    }
}

// h += rms(d)·w per row (the sandwich norm before the residual add), or
// h += d when `hasw` is 0.
kernel void ls_norm_add_rows(
    device float*       h    [[buffer(0)]],
    device const float* d    [[buffer(1)]],
    device const float* w    [[buffer(2)]],
    constant uint&  n        [[buffer(3)]],
    constant uint&  gemma    [[buffer(4)]],
    constant float& eps      [[buffer(5)]],
    constant uint&  hasw     [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    device float* hr = h + (ulong)row * n;
    device const float* dr = d + (ulong)row * n;
    float inv = 1.0f;
    if (hasw != 0u) {
        float acc = 0.0f;
        for (uint i = tid; i < n; i += 256u) { float v = dr[i]; acc += v * v; }
        acc = simd_sum(acc);
        if (lane == 0u) part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint k = 0; k < 8u; ++k) tot += part[k];
        inv = rsqrt(tot / (float)n + eps);
    }
    for (uint i = tid; i < n; i += 256u) {
        float f = dr[i];
        if (hasw != 0u) {
            float wv = w[i];
            if (gemma != 0u) wv += 1.0f;
            f = f * inv * wv;
        }
        hr[i] += f;
    }
}

// The gated short convolution over `t` positions from an empty ring, one
// thread a channel: bcx rows are [B | C | x] (3h), taps [h][k] with tap k-1
// on the current position; y = C·(tap[k-1]·B·x + Σ tap[k-2-s]·ring[s]).
kernel void ls_sconv(
    device const float* bcx  [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint& h         [[buffer(3)]],
    constant uint& k         [[buffer(4)]],
    constant uint& t         [[buffer(5)]],
    uint ch [[thread_position_in_grid]])
{
    if (ch >= h) return;
    float ring[15];
    for (uint s = 0; s < 15u; ++s) ring[s] = 0.0f;
    device const float* tp = taps + (ulong)ch * k;
    uint rl = k - 1u;
    for (uint r = 0; r < t; ++r) {
        device const float* row = bcx + (ulong)r * 3u * h;
        float bx = row[ch] * row[2u * h + ch];
        float acc = tp[k - 1u] * bx;
        for (uint s = 0; s < rl; ++s) acc += tp[k - 2u - s] * ring[s];
        y[(ulong)r * h + ch] = row[h + ch] * acc;
        for (uint s = rl; s > 1u; --s) ring[s - 1u] = ring[s - 2u];
        if (rl > 0u) ring[0] = bx;
    }
}

// Per (head, row): the head's RMS norm (when `hasw`), then RoPE at the row's
// own position over the first `rd` dims (pairs i, i + rd/2). In place on
// x [t][heads·hd]; one 32-lane group a head.
kernel void ls_qk_rope(
    device float*       x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* invf [[buffer(2)]],
    constant uint&  heads    [[buffer(3)]],
    constant uint&  hd       [[buffer(4)]],
    constant uint&  rd       [[buffer(5)]],
    constant uint&  hasw     [[buffer(6)]],
    constant uint&  gemma    [[buffer(7)]],
    constant float& eps      [[buffer(8)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup float a[512];
    uint head = tg.x, row = tg.y;
    device float* v = x + ((ulong)row * heads + head) * hd;
    float ss = 0.0f;
    for (uint d = lane; d < hd; d += 32u) { float e = v[d]; a[d] = e; ss += e * e; }
    ss = simd_sum(ss);
    if (hasw != 0u) {
        float inv = rsqrt(ss / (float)hd + eps);
        for (uint d = lane; d < hd; d += 32u) {
            float wv = w[d];
            if (gemma != 0u) wv += 1.0f;
            a[d] = a[d] * inv * wv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint hr = rd / 2u;
    float pos = (float)row;
    for (uint i = lane; i < hr; i += 32u) {
        float ang = pos * invf[i];
        float sn = precise::sin(ang);
        float cs = precise::cos(ang);
        float x0 = a[i], x1 = a[i + hr];
        a[i] = x0 * cs - x1 * sn;
        a[i + hr] = x0 * sn + x1 * cs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < hd; d += 32u) v[d] = a[d];
}

// Attention of row i, head hh over the packed row (online softmax in
// 128-key blocks, one key a thread for the scores, one dim a thread for
// P·V). `window` 0: d1's `set` mask (the state reads every row and is read
// by every row; a question row reads its own question); otherwise causal
// within the window. hd ≤ 256, hd % 4 == 0.
kernel void ls_attn(
    device const float* q     [[buffer(0)]],
    device const float* k     [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device float*       o     [[buffer(3)]],
    device const int*   grp   [[buffer(4)]],
    constant uint&  t         [[buffer(5)]],
    constant uint&  nh        [[buffer(6)]],
    constant uint&  nkv       [[buffer(7)]],
    constant uint&  hd        [[buffer(8)]],
    constant uint&  state     [[buffer(9)]],
    constant uint&  window    [[buffer(10)]],
    constant float& scale     [[buffer(11)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float qs[256];
    threadgroup float p[128];
    threadgroup float red[4];
    uint i = tg.x, hh = tg.y;
    uint g = hh / (nh / nkv);
    device const float* qr = q + ((ulong)i * nh + hh) * hd;
    for (uint d = tid; d < hd; d += 128u) qs[d] = qr[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int gi = grp[i];
    uint kvs = nkv * hd;
    uint j_lo = 0u, j_hi = t;
    if (window != 0u) {
        j_lo = (i + 1u > window) ? i + 1u - window : 0u;
        j_hi = i + 1u;
    }
    float m = -INFINITY, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    for (uint j0 = j_lo; j0 < j_hi; j0 += 128u) {
        uint j = j0 + tid;
        bool ok = j < j_hi;
        if (ok && window == 0u) ok = (i < state) || (j < state) || (grp[j] == gi);
        float s = -INFINITY;
        if (ok) {
            device const float4* kr = (device const float4*)(k + (ulong)j * kvs + (ulong)g * hd);
            float dot = 0.0f;
            for (uint d4 = 0; d4 < hd / 4u; ++d4) {
                float4 kk = kr[d4];
                uint d = d4 * 4u;
                dot += qs[d] * kk.x + qs[d + 1u] * kk.y + qs[d + 2u] * kk.z + qs[d + 3u] * kk.w;
            }
            s = dot * scale;
        }
        float bm = simd_max(s);
        if (lane == 0u) red[sg] = bm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bm = max(max(red[0], red[1]), max(red[2], red[3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mn = max(m, bm);
        if (mn == -INFINITY) continue;
        float pj = ok ? exp(s - mn) : 0.0f;
        p[tid] = pj;
        float bs = simd_sum(pj);
        if (lane == 0u) red[sg] = bs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bs = red[0] + red[1] + red[2] + red[3];
        float corr = (m == -INFINITY) ? 0.0f : exp(m - mn);
        l = l * corr + bs;
        acc0 *= corr;
        acc1 *= corr;
        uint nb = min(128u, j_hi - j0);
        for (uint jj = 0; jj < nb; ++jj) {
            float pv = p[jj];
            if (pv == 0.0f) continue;
            device const float* vr = v + (ulong)(j0 + jj) * kvs + (ulong)g * hd;
            if (tid < hd) acc0 += pv * vr[tid];
            if (tid + 128u < hd) acc1 += pv * vr[tid + 128u];
        }
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* orow = o + ((ulong)i * nh + hh) * hd;
    float il = (l > 0.0f) ? 1.0f / l : 0.0f;
    if (tid < hd) orow[tid] = acc0 * il;
    if (tid + 128u < hd) orow[tid + 128u] = acc1 * il;
}

// The per-head sigmoid gate: g = σ(x_row · gw[head]) scales the head's
// attention output, one 256-thread group per (head, row).
kernel void ls_head_gate(
    device float*       ao     [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device const float* gw     [[buffer(2)]],
    constant uint& nh          [[buffer(3)]],
    constant uint& hd          [[buffer(4)]],
    constant uint& hidden      [[buffer(5)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[8];
    uint hh = tg.x, row = tg.y;
    device const float* xr = x + (ulong)row * hidden;
    device const float* wr = gw + (ulong)hh * hidden;
    float acc = 0.0f;
    for (uint i = tid; i < hidden; i += 256u) acc += xr[i] * wr[i];
    acc = simd_sum(acc);
    if (lane == 0u) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float gate = 1.0f / (1.0f + exp(-tot));
    device float* a = ao + ((ulong)row * nh + hh) * hd;
    for (uint d = tid; d < hd; d += 256u) a[d] *= gate;
}

// a = silu(g)·u over n elements.
kernel void ls_silu_mul(
    device const float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device float*       a [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float x = g[i];
    a[i] = x / (1.0f + exp(-x)) * u[i];
}
