// Dory's token path on Metal, the twin of `gpu_wgpu/dory.rs` (`dory` has the
// semantics). Threadgroups of one simdgroup reduce with simd_sum; the row-sized
// stages after a projection go by short threads with no reduction, because a norm
// of a projection's input moves only through the residual's own norm.
#include <metal_stdlib>
using namespace metal;

struct P {
    uint t, pos0, cap, window;
    uint nh, nk, hd, hidden;
    uint ffn, qsize, ncap, ch;
    float scale, eps;
    uint pad0, pad1;
};

// Rotation, unit heads, the query scale, K and V into the ring: one simdgroup per
// head, one lane per rotated pair of vec4s.
kernel void dory_prep(
    device const float4* yq   [[buffer(0)]],
    device const float4* yk   [[buffer(1)]],
    device const float4* yv   [[buffer(2)]],
    device const float4* rope [[buffer(3)]],
    device const float4* sq   [[buffer(4)]],
    device float4*       qo   [[buffer(5)]],
    device float4*       ko   [[buffer(6)]],
    device float4*       vo   [[buffer(7)]],
    constant P&          p    [[buffer(8)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint a = tg.x, row = tg.y;
    uint n = p.hd / 4u, half_ = n / 2u, e = lane;
    uint rb = row * n;
    float4 c = rope[rb + e], sn = rope[rb + half_ + e];
    if (a < p.nh) {
        uint at = (row * p.nh + a) * n;
        float4 x = yq[at + e], y = yq[at + half_ + e];
        float4 u = x * c - y * sn, v = y * c + x * sn;
        float inv = 1.0f / sqrt(simd_sum(dot(u, u) + dot(v, v)) + p.eps);
        uint sb = a * n;
        qo[at + e] = u * inv * sq[sb + e];
        qo[at + half_ + e] = v * inv * sq[sb + half_ + e];
        return;
    }
    uint kh = a - p.nh;
    uint at = (row * p.nk + kh) * n;
    uint ring = (((p.pos0 + row) % p.cap) * p.nk + kh) * n;
    float4 x = yk[at + e], y = yk[at + half_ + e];
    float4 u = x * c - y * sn, v = y * c + x * sn;
    float inv = 1.0f / sqrt(simd_sum(dot(u, u) + dot(v, v)) + p.eps);
    ko[ring + e] = u * inv;
    ko[ring + half_ + e] = v * inv;
    vo[ring + e] = yv[at + e];
    vo[ring + half_ + e] = yv[at + half_ + e];
}

inline uint low_key(uint pos, uint window) { return pos + 1u > window ? pos + 1u - window : 0u; }

// One thread per key of a KV head and its four query heads at once.
kernel void dory_score(
    device const float4* q  [[buffer(0)]],
    device const float4* k  [[buffer(1)]],
    device float4*       sc [[buffer(2)]],
    constant P&          p  [[buffer(3)]],
    uint3 g [[thread_position_in_grid]])
{
    uint i = g.x, kh = g.y, row = g.z;
    uint pos = p.pos0 + row;
    uint lo = low_key(pos, p.window);
    if (i >= pos + 1u - lo) return;
    uint e4 = p.hd / 4u;
    uint q0 = (row * p.nh + kh * 4u) * e4;
    uint kb = (((lo + i) % p.cap) * p.nk + kh) * e4;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint e = 0u; e < e4; ++e) {
        float4 kv = k[kb + e];
        a0 += kv * q[q0 + e];
        a1 += kv * q[q0 + e4 + e];
        a2 += kv * q[q0 + 2u * e4 + e];
        a3 += kv * q[q0 + 3u * e4 + e];
    }
    float4 r = float4(a0.x + a0.y + a0.z + a0.w, a1.x + a1.y + a1.z + a1.w,
                      a2.x + a2.y + a2.z + a2.w, a3.x + a3.y + a3.z + a3.w);
    sc[(row * p.nk + kh) * p.ncap + i] = r * p.scale;
}

// The softmax of a KV head's four queries: each key's weight, and 1/Σ.
kernel void dory_soft(
    device float4*  sf  [[buffer(0)]],
    device float4*  sfi [[buffer(1)]],
    constant P&     p   [[buffer(2)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint kh = tg.x, row = tg.y;
    uint pos = p.pos0 + row;
    uint lo = low_key(pos, p.window);
    uint n = pos + 1u - lo;
    uint sb = (row * p.nk + kh) * p.ncap;
    float4 m = float4(-1.0e30f);
    for (uint i = lane; i < n; i += 32u) m = max(m, sf[sb + i]);
    float4 mx = float4(simd_max(m.x), simd_max(m.y), simd_max(m.z), simd_max(m.w));
    float4 den = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        float4 s = sf[sb + i] - mx;
        float4 e = float4(exp(s.x), exp(s.y), exp(s.z), exp(s.w));
        sf[sb + i] = e;
        den += e;
    }
    float4 total = float4(simd_sum(den.x), simd_sum(den.y), simd_sum(den.z), simd_sum(den.w));
    if (lane == 0u) sfi[row * p.nk + kh] = 1.0f / total;
}

// The value sum in pieces, one thread per (KV head, chunk of keys, vec4 of the head).
kernel void dory_sum(
    device const float4* v  [[buffer(0)]],
    device const float4* sc [[buffer(1)]],
    device float4*       pz [[buffer(2)]],
    constant P&          p  [[buffer(3)]],
    uint2 g [[thread_position_in_grid]])
{
    uint e4 = p.hd / 4u;
    uint t = g.x, row = g.y;
    uint d = t % e4;
    uint ch = (t / e4) % p.ch;
    uint kh = t / (e4 * p.ch);
    if (kh >= p.nk) return;
    uint pos = p.pos0 + row;
    uint lo = low_key(pos, p.window);
    uint n = pos + 1u - lo;
    uint per = (n + p.ch - 1u) / p.ch;
    uint sb = (row * p.nk + kh) * p.ncap;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    uint last = min((ch + 1u) * per, n);
    for (uint i = ch * per; i < last; ++i) {
        float4 a = sc[sb + i];
        float4 w = v[(((lo + i) % p.cap) * p.nk + kh) * e4 + d];
        o0 += a.x * w;
        o1 += a.y * w;
        o2 += a.z * w;
        o3 += a.w * w;
    }
    uint at = (((row * p.nk + kh) * 8u + ch) * 4u) * e4 + d;
    pz[at] = o0;
    pz[at + e4] = o1;
    pz[at + 2u * e4] = o2;
    pz[at + 3u * e4] = o3;
}

// The pieces added, 1/Σ and the gate.
kernel void dory_gate(
    device const float4* g_  [[buffer(0)]],
    device const float4* sg  [[buffer(1)]],
    device const float4* pz  [[buffer(2)]],
    device const float4* si  [[buffer(3)]],
    device float4*       z   [[buffer(4)]],
    constant P&          p   [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    uint row = id.y, e = id.x;
    uint e4 = p.hd / 4u;
    uint n = p.nh * e4;
    if (e >= n) return;
    uint base = row * n;
    uint d = e % e4, hi = e / e4, kh = hi / 4u, h = hi % 4u;
    float4 acc = 0.0f;
    uint pb = (row * p.nk + kh) * 8u;
    for (uint ch = 0u; ch < p.ch; ++ch) acc += pz[((pb + ch) * 4u + h) * e4 + d];
    float inv = si[row * p.nk + kh][h];
    float4 x = -(g_[base + e] * sg[e]);
    z[base + e] = acc * inv / (1.0f + float4(exp(x.x), exp(x.y), exp(x.z), exp(x.w)));
}

// The MLP activation, in place in the gate's output.
kernel void dory_silu(
    device float4*       g_ [[buffer(0)]],
    device const float4* u_ [[buffer(1)]],
    device const float4* a_ [[buffer(2)]],
    device const float4* b_ [[buffer(3)]],
    constant P&          p  [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    uint n = p.ffn / 4u, e = id.x;
    if (e >= n) return;
    uint base = id.y * n;
    float4 g = g_[base + e] * a_[e];
    float4 u = u_[base + e] * b_[e];
    float4 m = -g;
    g_[base + e] = g / (1.0f + float4(exp(m.x), exp(m.y), exp(m.z), exp(m.w))) * u;
}

// The nGPT residual.
kernel void dory_slerp(
    device float4*       h  [[buffer(0)]],
    device const float4* o  [[buffer(1)]],
    device const float4* al [[buffer(2)]],
    constant P&          p  [[buffer(3)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint n = p.hidden / 4u, base = tg * n;
    float sx = 0.0f, so = 0.0f;
    for (uint e = lane; e < n; e += 32u) {
        float4 x = h[base + e], y = o[base + e];
        sx += dot(x, x);
        so += dot(y, y);
    }
    float ix = 1.0f / sqrt(simd_sum(sx) + p.eps);
    float io = 1.0f / sqrt(simd_sum(so) + p.eps);
    float t = 0.0f;
    for (uint e = lane; e < n; e += 32u) {
        float4 a = h[base + e] * ix;
        float4 y = a + al[e] * (o[base + e] * io - a);
        h[base + e] = y;
        t += dot(y, y);
    }
    float r = 1.0f / sqrt(simd_sum(t) + p.eps);
    for (uint e = lane; e < n; e += 32u) h[base + e] = h[base + e] * r;
}

// The loop's edges: mode 0 enters the first pass (input = state = h), mode 1 goes
// between passes (state = ĥ), mode 2 leaves the last (h = ĥ). The early exit: mode 3
// reads out and keeps the state, mode 4 starts the next pass from the kept state.
kernel void dory_mix(
    device float4*  h    [[buffer(0)]],
    device float4*  in_  [[buffer(1)]],
    device float4*  st   [[buffer(2)]],
    constant uint&  mode [[buffer(3)]],
    constant P&     p    [[buffer(4)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint n = p.hidden / 4u, base = tg * n;
    if (mode == 0u) {
        float s = 0.0f;
        for (uint e = lane; e < n; e += 32u) {
            float4 v = h[base + e];
            in_[base + e] = v;
            st[base + e] = v;
            float4 y = v + v;
            h[base + e] = y;
            s += dot(y, y);
        }
        float r = 1.0f / sqrt(simd_sum(s) + p.eps);
        for (uint e = lane; e < n; e += 32u) h[base + e] = h[base + e] * r;
        return;
    }
    if (mode == 4u) {
        float t4 = 0.0f;
        for (uint e = lane; e < n; e += 32u) {
            float4 y = st[base + e] + in_[base + e];
            h[base + e] = y;
            t4 += dot(y, y);
        }
        float r4 = 1.0f / sqrt(simd_sum(t4) + p.eps);
        for (uint e = lane; e < n; e += 32u) h[base + e] = h[base + e] * r4;
        return;
    }
    float s = 0.0f;
    for (uint e = lane; e < n; e += 32u) {
        float4 v = h[base + e];
        s += dot(v, v);
    }
    float r = 1.0f / sqrt(simd_sum(s) + p.eps);
    if (mode == 3u) {
        for (uint e = lane; e < n; e += 32u) {
            float4 u = h[base + e] * r;
            h[base + e] = u;
            st[base + e] = u;
        }
        return;
    }
    if (mode == 2u) {
        for (uint e = lane; e < n; e += 32u) h[base + e] = h[base + e] * r;
        return;
    }
    float t = 0.0f;
    for (uint e = lane; e < n; e += 32u) {
        float4 u = h[base + e] * r;
        st[base + e] = u;
        float4 y = u + in_[base + e];
        h[base + e] = y;
        t += dot(y, y);
    }
    float r2 = 1.0f / sqrt(simd_sum(t) + p.eps);
    for (uint e = lane; e < n; e += 32u) h[base + e] = h[base + e] * r2;
}

// O(1) on a full layer (`nystrom`): the far, push and attend kernels are the
// engine's. These take the prompt's queries for the seal, a later pass's rewrite of
// this position, and the gate over the kernel's output.
kernel void dory_qcopy(
    device const float4* qi [[buffer(0)]],
    device float4*       qo [[buffer(1)]],
    constant P&          p  [[buffer(2)]],
    uint2 g [[thread_position_in_grid]])
{
    uint n = p.nh * (p.hd / 4u);
    uint at = p.pos0 + g.y;
    if (g.x >= n || at % 8u != 0u) return;
    qo[at / 8u * n + g.x] = qi[g.y * n + g.x];
}

kernel void dory_o1_back(
    device uint*   meta [[buffer(0)]],
    constant uint& w    [[buffer(1)]],
    uint g [[thread_position_in_grid]])
{
    uint at = g * 4u;
    if (meta[at] == w) {
        meta[at + 1u] = (meta[at + 1u] + w - 1u) % w;
        meta[at + 2u] = meta[at + 2u] - 1u;
    } else {
        meta[at] = meta[at] - 1u;
    }
}

// The far field's attention, one row, in the exact path's short threads.
struct OU { uint hpg, m, w, ns, d, dv; float scale; uint rect; };

static inline float4 dots4(device const float4* k, device const float4* q, uint q0, uint e4, float scale) {
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint e = 0u; e < e4; ++e) {
        float4 kv = k[e];
        a0 += kv * q[q0 + e];
        a1 += kv * q[q0 + e4 + e];
        a2 += kv * q[q0 + 2u * e4 + e];
        a3 += kv * q[q0 + 3u * e4 + e];
    }
    return float4(a0.x + a0.y + a0.z + a0.w, a1.x + a1.y + a1.z + a1.w,
                  a2.x + a2.y + a2.z + a2.w, a3.x + a3.y + a3.z + a3.w) * scale;
}

// Near scores: a thread per key of the sink and the window.
kernel void dory_os(
    device const uint*   om [[buffer(0)]],
    device const float4* q  [[buffer(1)]],
    device const float4* rk [[buffer(2)]],
    device const float4* sk [[buffer(3)]],
    device float4*       sc [[buffer(4)]],
    constant P&          p  [[buffer(5)]],
    constant OU&         ou [[buffer(6)]],
    uint2 g [[thread_position_in_grid]])
{
    uint s = g.x, kh = g.y;
    if (s >= ou.ns + om[kh * 4u]) return;
    uint e4 = p.hd / 4u;
    device const float4* k = s < ou.ns ? sk + (kh * ou.ns + s) * e4 : rk + (kh * ou.w + s - ou.ns) * e4;
    sc[kh * p.ncap + s] = dots4(k, q, kh * 4u * e4, e4, p.scale);
}

// Landmark scores, the same way.
kernel void dory_ol(
    device const float4* q  [[buffer(0)]],
    device const float4* kt [[buffer(1)]],
    device float4*       fs [[buffer(2)]],
    constant P&          p  [[buffer(3)]],
    constant OU&         ou [[buffer(4)]],
    uint2 g [[thread_position_in_grid]])
{
    uint a = g.x, kh = g.y;
    if (a >= ou.m) return;
    uint e4 = p.hd / 4u;
    fs[kh * ou.m + a] = dots4(kt + (kh * ou.m + a) * e4, q, kh * 4u * e4, e4, p.scale);
}

// The largest near score of a KV head's four queries.
kernel void dory_omax(
    device const uint*   om [[buffer(0)]],
    device const float4* sf [[buffer(1)]],
    device float4*       nm [[buffer(2)]],
    constant P&          p  [[buffer(3)]],
    constant OU&         ou [[buffer(4)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint kh = tg.x;
    uint n = ou.ns + om[kh * 4u];
    uint sb = kh * p.ncap;
    float4 m = float4(-1.0e30f);
    for (uint i = lane; i < n; i += 32u) m = max(m, sf[sb + i]);
    float4 mx = float4(simd_max(m.x), simd_max(m.y), simd_max(m.z), simd_max(m.w));
    if (lane == 0u) nm[kh] = mx;
}

// A query head's far part: the landmark weights through its inverse, one gain per
// landmark, the far mass, and the common exponent the near scores will use.
kernel void dory_ofar(
    device const uint*   om [[buffer(0)]],
    device const float4* fs [[buffer(1)]],
    device const float4* nm [[buffer(2)]],
    device const float*  mu [[buffer(3)]],
    device const float*  mz [[buffer(4)]],
    device float4*       hd [[buffer(5)]],
    device float*        gn [[buffer(6)]],
    constant P&          p  [[buffer(7)]],
    constant OU&         ou [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    uint gh = id;
    if (gh >= p.nh) return;
    uint kh = gh / 4u, h = gh % 4u, m = ou.m;
    float c_all = nm[kh][h];
    float far_den = 0.0f, have = 0.0f;
    if (om[kh * 4u + 2u] > 0u) {
        float f = -3.0e38f;
        float fe[32];
        float u[32];
        for (uint a = 0u; a < m; ++a) {
            fe[a] = fs[kh * m + a][h];
            f = max(f, fe[a]);
            u[a] = 0.0f;
        }
        for (uint a = 0u; a < m; ++a) {
            float e = exp(fe[a] - f);
            uint row = (gh * m + a) * m;
            for (uint b = 0u; b < m; ++b) u[b] += e * mu[row + b];
        }
        if (ou.rect != 0u) {
            for (uint b = 0u; b < m; ++b) u[b] = max(u[b], 0.0f);
        }
        uint mzb = gh * 2u * m;
        for (uint b = 0u; b < m; ++b) c_all = max(c_all, f + mz[mzb + b]);
        for (uint b = 0u; b < m; ++b) {
            float gain = u[b] * exp(f + mz[mzb + b] - c_all);
            gn[gh * m + b] = gain;
            far_den += gain * mz[mzb + m + b];
        }
        if (far_den >= 0.0f) { have = 1.0f; } else { far_den = 0.0f; }
    }
    hd[gh] = float4(c_all, far_den, have, 0.0f);
}

// The near weights under that exponent, and 1/Σ with the far mass in it.
kernel void dory_osoft(
    device const uint*   om  [[buffer(0)]],
    device float4*       sf  [[buffer(1)]],
    device const float4* hd  [[buffer(2)]],
    device float4*       sfi [[buffer(3)]],
    constant P&          p   [[buffer(4)]],
    constant OU&         ou  [[buffer(5)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint kh = tg.x;
    uint n = ou.ns + om[kh * 4u];
    uint sb = kh * p.ncap;
    float4 h0 = hd[kh * 4u], h1 = hd[kh * 4u + 1u], h2 = hd[kh * 4u + 2u], h3 = hd[kh * 4u + 3u];
    float4 c = float4(h0.x, h1.x, h2.x, h3.x);
    float4 far = float4(h0.y, h1.y, h2.y, h3.y);
    float4 den = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        float4 s = sf[sb + i] - c;
        float4 e = float4(exp(s.x), exp(s.y), exp(s.z), exp(s.w));
        sf[sb + i] = e;
        den += e;
    }
    float4 total = float4(simd_sum(den.x), simd_sum(den.y), simd_sum(den.z), simd_sum(den.w)) + far;
    if (lane == 0u) sfi[kh] = 1.0f / max(total, float4(1.0e-30f));
}

// The near value sum in eight pieces.
kernel void dory_osum(
    device const uint*   om [[buffer(0)]],
    device const float4* sc [[buffer(1)]],
    device const float4* rv [[buffer(2)]],
    device const float4* sv [[buffer(3)]],
    device float4*       pz [[buffer(4)]],
    constant P&          p  [[buffer(5)]],
    constant OU&         ou [[buffer(6)]],
    uint t [[thread_position_in_grid]])
{
    uint e4 = p.hd / 4u;
    uint d = t % e4;
    uint ch = (t / e4) % 8u;
    uint kh = t / (e4 * 8u);
    if (kh >= p.nk) return;
    uint n = ou.ns + om[kh * 4u];
    uint per = (n + 7u) / 8u;
    uint sb = kh * p.ncap;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    uint last = min((ch + 1u) * per, n);
    for (uint i = ch * per; i < last; ++i) {
        float4 a = sc[sb + i];
        float4 w = i < ou.ns ? sv[(kh * ou.ns + i) * e4 + d] : rv[(kh * ou.w + i - ou.ns) * e4 + d];
        o0 += a.x * w;
        o1 += a.y * w;
        o2 += a.z * w;
        o3 += a.w * w;
    }
    uint at = ((kh * 8u + ch) * 4u) * e4 + d;
    pz[at] = o0;
    pz[at + e4] = o1;
    pz[at + 2u * e4] = o2;
    pz[at + 3u * e4] = o3;
}

// The pieces, the far values, 1/Σ and the gate.
kernel void dory_ogate(
    device const float4* pz [[buffer(0)]],
    device const float4* hd [[buffer(1)]],
    device const float*  gn [[buffer(2)]],
    device const float4* th [[buffer(3)]],
    device const float4* si [[buffer(4)]],
    device const float4* g_ [[buffer(5)]],
    device const float4* sg [[buffer(6)]],
    device float4*       z  [[buffer(7)]],
    constant P&          p  [[buffer(8)]],
    constant OU&         ou [[buffer(9)]],
    uint e [[thread_position_in_grid]])
{
    uint e4 = p.hd / 4u;
    if (e >= p.nh * e4) return;
    uint d = e % e4, gh = e / e4, kh = gh / 4u, h = gh % 4u;
    float4 acc = 0.0f;
    uint pb = kh * 8u;
    for (uint ch = 0u; ch < 8u; ++ch) acc += pz[((pb + ch) * 4u + h) * e4 + d];
    if (hd[gh].z > 0.5f) {
        for (uint b = 0u; b < ou.m; ++b) acc += gn[gh * ou.m + b] * th[(gh * ou.m + b) * e4 + d];
    }
    float4 x = -(g_[e] * sg[e]);
    z[e] = acc * si[kh][h] / (1.0f + float4(exp(x.x), exp(x.y), exp(x.z), exp(x.w)));
}
