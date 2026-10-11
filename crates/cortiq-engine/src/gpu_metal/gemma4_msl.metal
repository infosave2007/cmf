// Gemma-4 (the humanizer 12B) kernels of the Metal token graph and chunk
// prefill — a library of its own, compiled the first time a Gemma-4 model
// asks for it (`gpu_metal::gemma4`), so no other model pays for it or sees
// its main-library kernels change.
//
// The `_w16` kernels are the main library's attention kernels with their
// per-lane register arrays widened from 8 to 16 slots: head width up to 512
// (Gemma-4's global layers), the code otherwise character for character
// (generated from gpu_metal.rs; regenerate rather than hand-edit them).
#include <metal_stdlib>
using namespace metal;

inline float4 q4_nib4_scaled(uint u) {
    return float4((float)(u & 0xFu), (float)(u & 0xF0u),
                  (float)(u & 0xF00u), (float)(u & 0xF000u));
}

// tanh-approximated GELU (HF `gelu_pytorch_tanh`): `inference::gelu_tanh`
// term for term. `precise::tanh` — the library compiles with fast math.
inline float gelu_tanh_f(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978846f * (x + 0.044715f * x * x * x)));
}

// `gelu_mul_pre` with the tanh GELU (same bindings).
kernel void gelu_tanh_mul_pre(
    device const float* g   [[buffer(0)]],
    device const float* u   [[buffer(1)]],
    device const float* col [[buffer(2)]],
    device float*       act [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    constant uint&      has_col [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float cv = has_col != 0 ? col[i] : 1.0f;
    act[i] = gelu_tanh_f(g[i]) * u[i] * cv;
}

// Scale-less RMS norm of every V head (Gemma-4, `attention::vnorm_head`):
// x <- x / sqrt(mean(x^2) + eps). One simdgroup per (head, row); rows are
// [nb][nkv*hd].
kernel void vnorm_rows(
    device float*   v   [[buffer(0)]],
    constant uint&  nkv [[buffer(1)]],
    constant uint&  hd  [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint h = tg.x, e = tg.y;
    if (h >= nkv) return;
    device float* x = v + ((ulong)e * nkv + h) * hd;
    float ss = 0.0f;
    for (uint d = lane; d < hd; d += 32u) { float a = x[d]; ss += a * a; }
    ss = simd_sum(ss);
    float inv = 1.0f / sqrt(ss / (float)hd + eps);
    for (uint d = lane; d < hd; d += 32u) x[d] *= inv;
}

// Gemma-4 layer tail: h = (h + norm(d)) * sc, norm(d) = rms(d)*w (Gemma
// style 1+w when `gemma`) when `hasn`, else d itself — the CPU's
// ffn_out_norm, residual add and layer scalar in that order. One
// threadgroup (256) per row.
kernel void norm_add_scale_rows(
    device float*       h    [[buffer(0)]],
    device const float* d    [[buffer(1)]],
    device const float* w    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    constant uint&      gemma [[buffer(4)]],
    constant float&     eps  [[buffer(5)]],
    constant uint&      hasn [[buffer(6)]],
    constant float&     sc   [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    device float* hr = h + (ulong)row * n;
    device const float* dr = d + (ulong)row * n;
    float inv = 1.0f;
    if (hasn != 0u) {
        float acc = 0.0f;
        for (uint i = tid; i < n; i += 256u) { float v = dr[i]; acc += v * v; }
        acc = simd_sum(acc);
        if (lane == 0) part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint k = 0; k < 8u; ++k) tot += part[k];
        inv = rsqrt(tot / (float)n + eps);
    }
    for (uint i = tid; i < n; i += 256u) {
        float f = dr[i];
        if (hasn != 0u) {
            float wv = gemma != 0u ? (1.0f + w[i]) : w[i];
            f = f * inv * wv;
        }
        hr[i] = (hr[i] + f) * sc;
    }
}

// `q4tp_matvec_m_gu_gelu` with the tanh GELU in its epilogue.
kernel void q4tp_matvec_m_gu_gelut(
    device const uchar* qg   [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       act  [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    device const uchar* qu   [[buffer(5)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    threadgroup float lad[8u * 8u * 32u];
    uint r0 = (tgpos * sgs + sg) * 4u;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, 4u) : 0u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* pg = (device const half*)(qg + params_off + (ulong)(r0 + ri) * 4ul);
        device const half* pu = (device const half*)(qu + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * 8u + ri) * 32u + lane] = exp2((float)pg[0] + (float)lane * (float)pg[1]);
        lad[(sg * 8u + 4u + ri) * 32u + lane] = exp2((float)pu[0] + (float)lane * (float)pu[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
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
        for (uint j = 0u; j < 2u * nr; ++j) {
            uint ri = j >> 1u;
            bool up = (j & 1u) != 0u;
            device const uchar* q = up ? qu : qg;
            uint r = r0 + ri;
            uint4 b = *(device const uint4*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[(sg * 8u + (up ? 4u : 0u) + ri) * 32u + code];
            float gsum = dot(q4_nib4_scaled(b.x), x0) + dot(q4_nib4_scaled(b.x >> 16), x1)
                       + dot(q4_nib4_scaled(b.y), x2) + dot(q4_nib4_scaled(b.y >> 16), x3)
                       + dot(q4_nib4_scaled(b.z), x4) + dot(q4_nib4_scaled(b.z >> 16), x5)
                       + dot(q4_nib4_scaled(b.w), x6) + dot(q4_nib4_scaled(b.w >> 16), x7);
            float contrib = scale * (gsum + m8);
            if (j == 0u) ag0 += contrib;
            else if (j == 1u) au0 += contrib;
            else if (j == 2u) ag1 += contrib;
            else if (j == 3u) au1 += contrib;
            else if (j == 4u) ag2 += contrib;
            else if (j == 5u) au2 += contrib;
            else if (j == 6u) ag3 += contrib;
            else au3 += contrib;
        }
    }
    ag0 = simd_sum(ag0); ag1 = simd_sum(ag1); ag2 = simd_sum(ag2); ag3 = simd_sum(ag3);
    au0 = simd_sum(au0); au1 = simd_sum(au1); au2 = simd_sum(au2); au3 = simd_sum(au3);
    if (lane == 0u) {
        float cv = 1.0f;
        float gv = ag0;
        act[r0] = gelu_tanh_f(gv) * au0 * cv;
        if (nr > 1u) { gv = ag1; act[r0 + 1u] = gelu_tanh_f(gv) * au1 * cv; }
        if (nr > 2u) { gv = ag2; act[r0 + 2u] = gelu_tanh_f(gv) * au2 * cv; }
        if (nr > 3u) { gv = ag3; act[r0 + 3u] = gelu_tanh_f(gv) * au3 * cv; }
    }
}

// `attn_rope_qkn` widened to 16 slots per lane (head width <= 512).
kernel void attn_rope_qkn_w16(
    device const float* qraw [[buffer(0)]],
    device float*       k    [[buffer(1)]],
    device float*       qout [[buffer(2)]],
    device float*       gout [[buffer(3)]],
    device const float* qnw  [[buffer(4)]],
    device const float* knw  [[buffer(5)]],
    device const float* invf [[buffer(6)]],
    constant uint&  nh    [[buffer(7)]],
    constant uint&  nkv   [[buffer(8)]],
    constant uint&  hd    [[buffer(9)]],
    constant uint&  rd    [[buffer(10)]],
    constant uint&  pos   [[buffer(11)]],
    constant uint&  flags [[buffer(12)]], // 1=gate 2=qnorm 4=knorm 8=gemma 32=norm-after-rope
    constant float& eps   [[buffer(13)]],
    constant float& rsc   [[buffer(14)]], // post-rotation RoPE scale (YaRN)
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Head = simdgroup index IN THE GRID. This kernel is launched with
    // dispatch_threads over (nh+nkv)·32 threads, and the last threadgroup
    // is partial whenever that is not a multiple of 256 — there
    // `simdgroups_per_threadgroup` reports the PARTIAL group's count, so
    // `tg·sgs + sg` re-derived heads 0..k for it instead of the tail
    // heads: the K heads of every model with (nh+nkv) % 8 != 0
    // (Qwen3.5-0.8B: 8+2, Qwen3.8-27B: 24+4) were never normed nor
    // rotated, the raw K went into the cache, and the device attend ran
    // 15-20% off the CPU's on every token.
    uint head = gid >> 5u;
    if (head >= nh + nkv) return;
    bool isq = head < nh;
    bool gate = (flags & 1u) != 0u;
    device const float* src = isq
        ? qraw + (ulong)head * (gate ? 2u : 1u) * hd
        : k + (ulong)(head - nh) * hd;
    uint nt = (hd + 31u) / 32u;
    float xv[16];
    float ss = 0.0f;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        xv[t] = d < hd ? src[d] : 0.0f;
        ss += xv[t] * xv[t];
    }
    ss = simd_sum(ss);
    bool normed = isq ? (flags & 2u) != 0u : (flags & 4u) != 0u;
    // HunYuan dense (flag 32): rotate FIRST, then norm. The rotation keeps
    // the head's sum of squares, so `ss` serves both orders; only the
    // elementwise norm weights must see the rotated vector.
    bool late = (flags & 32u) != 0u;
    // Partial RoPE: pair (i, i + rd/2); with (rd/2) % 32 == 0 both
    // halves live in the same lane, slots t and t + (rd/2)/32.
    uint hlf = rd / 2u;
    uint toff = hlf / 32u;
    if (late) {
        for (uint t = 0; t < toff; ++t) {
            uint i = t * 32u + lane;
            if (i < hlf) {
                float angle = (float)pos * invf[i];
                float c = cos(angle), s = sin(angle);
                float x0 = xv[t], x1 = xv[t + toff];
                xv[t] = x0 * c - x1 * s;
                xv[t + toff] = x0 * s + x1 * c;
                // YaRN attention amplitude (Mellum2.1 global layers): the
                // CPU's `rope_rotate_scaled`. 1.0 keeps the unscaled ops.
                if (rsc != 1.0f) { xv[t] *= rsc; xv[t + toff] *= rsc; }
            }
        }
    }
    if (normed) {
        float inv = 1.0f / sqrt(ss / (float)hd + eps);
        device const float* w = isq ? qnw : knw;
        bool gemma = (flags & 8u) != 0u;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float wd = w[d];
                xv[t] = xv[t] * inv * (gemma ? (1.0f + wd) : wd);
            }
        }
    }
    if (!late) {
        for (uint t = 0; t < toff; ++t) {
            uint i = t * 32u + lane;
            if (i < hlf) {
                float angle = (float)pos * invf[i];
                float c = cos(angle), s = sin(angle);
                float x0 = xv[t], x1 = xv[t + toff];
                xv[t] = x0 * c - x1 * s;
                xv[t + toff] = x0 * s + x1 * c;
                // YaRN attention amplitude (Mellum2.1 global layers): the
                // CPU's `rope_rotate_scaled`. 1.0 keeps the unscaled ops.
                if (rsc != 1.0f) { xv[t] *= rsc; xv[t + toff] *= rsc; }
            }
        }
    }
    device float* dst = isq ? qout + (ulong)head * hd : k + (ulong)(head - nh) * hd;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) dst[d] = xv[t];
    }
    if (isq && gate) {
        device const float* gsrc = qraw + (ulong)head * 2u * hd + hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) gout[(ulong)head * hd + d] = gsrc[d];
        }
    }
}

// `chunk_rope_kv` widened to 16 slots per lane (head width <= 512).
kernel void chunk_rope_kv_w16(
    device const float* qraw [[buffer(0)]],   // [nb, nh·hd]
    device const float* kraw [[buffer(1)]],   // [nb, nkv·hd]
    device const float* vraw [[buffer(2)]],   // [nb, nkv·hd]
    device float*       qout [[buffer(3)]],   // [nb, nh, hd]
    device float*       kbuf [[buffer(4)]],
    device float*       vbuf [[buffer(5)]],
    device const float* bq   [[buffer(6)]],
    device const float* bk   [[buffer(7)]],
    device const float* bv   [[buffer(8)]],
    device const float* qnw  [[buffer(9)]],
    device const float* knw  [[buffer(10)]],
    device const float* invf [[buffer(11)]],
    constant uint&  nh    [[buffer(12)]],
    constant uint&  nkv   [[buffer(13)]],
    constant uint&  hd    [[buffer(14)]],
    constant uint&  rd    [[buffer(15)]],
    constant uint&  pos0  [[buffer(16)]],
    constant uint&  st0   [[buffer(17)]],
    constant uint&  cap   [[buffer(18)]],
    constant uint&  flags [[buffer(19)]],
    constant float& eps   [[buffer(20)]],
    constant uint&  nb    [[buffer(21)]],
    constant float& rsc   [[buffer(22)]], // post-rotation RoPE scale (YaRN)
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint head = tg.x * sgs + sg;
    uint bi = tg.y;
    if (head >= nh + 2u * nkv || bi >= nb) return;
    bool isq = head < nh;
    bool isv = head >= nh + nkv;
    uint kvh = isv ? head - nh - nkv : head - nh;
    bool bias = (flags & 16u) != 0u;
    device const float* src = isq
        ? qraw + (ulong)bi * nh * hd + (ulong)head * hd
        : (isv ? vraw : kraw) + (ulong)bi * nkv * hd + (ulong)kvh * hd;
    device const float* brow = isq ? bq : (isv ? bv : bk);
    uint nt = (hd + 31u) / 32u;
    float xv[16];
    float ss = 0.0f;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        float v = d < hd ? src[d] : 0.0f;
        if (bias && d < hd) v += brow[(isq ? (ulong)head : (ulong)kvh) * hd + d];
        xv[t] = v;
        ss += v * v;
    }
    if (!isv) {
        ss = simd_sum(ss);
        bool normed = isq ? (flags & 2u) != 0u : (flags & 4u) != 0u;
        // HunYuan dense (flag 32): rotate first, then norm (see attn_rope_qkn).
        bool late = (flags & 32u) != 0u;
        uint hlf = rd / 2u;
        uint toff = hlf / 32u;
        uint pos = pos0 + bi;
        if (late) {
            for (uint t = 0; t < toff; ++t) {
                uint i = t * 32u + lane;
                if (i < hlf) {
                    float angle = (float)pos * invf[i];
                    float c = cos(angle), sn = sin(angle);
                    float x0 = xv[t], x1 = xv[t + toff];
                    xv[t] = x0 * c - x1 * sn;
                    xv[t + toff] = x0 * sn + x1 * c;
                    // YaRN attention amplitude (Mellum2.1 global layers): the
                    // CPU's `rope_rotate_scaled`. 1.0 keeps the unscaled ops.
                    if (rsc != 1.0f) { xv[t] *= rsc; xv[t + toff] *= rsc; }
                }
            }
        }
        if (normed) {
            float inv = 1.0f / sqrt(ss / (float)hd + eps);
            device const float* w = isq ? qnw : knw;
            bool gm = (flags & 8u) != 0u;
            for (uint t = 0; t < nt; ++t) {
                uint d = t * 32u + lane;
                if (d < hd) {
                    float wd = w[d];
                    xv[t] = xv[t] * inv * (gm ? (1.0f + wd) : wd);
                }
            }
        }
        if (!late) {
            for (uint t = 0; t < toff; ++t) {
                uint i = t * 32u + lane;
                if (i < hlf) {
                    float angle = (float)pos * invf[i];
                    float c = cos(angle), sn = sin(angle);
                    float x0 = xv[t], x1 = xv[t + toff];
                    xv[t] = x0 * c - x1 * sn;
                    xv[t + toff] = x0 * sn + x1 * c;
                    // YaRN attention amplitude (Mellum2.1 global layers): the
                    // CPU's `rope_rotate_scaled`. 1.0 keeps the unscaled ops.
                    if (rsc != 1.0f) { xv[t] *= rsc; xv[t + toff] *= rsc; }
                }
            }
        }
    }
    // Q lands head-major ([head][bi][hd]) — the group panel the scores
    // GEMM consumes without a gather.
    device float* dst = isq
        ? qout + ((ulong)head * nb + bi) * hd
        : (isv ? vbuf : kbuf) + ((ulong)kvh * cap + st0 + bi) * hd;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) dst[d] = xv[t];
    }
}

// `gqa_attend` widened to 16 slots per lane (head width <= 512).
kernel void gqa_attend_w16(
    device const float* q    [[buffer(0)]],
    device const float* kbuf [[buffer(1)]],
    device const float* vbuf [[buffer(2)]],
    device float*       outb [[buffer(3)]],
    device atomic_float* imp [[buffer(4)]],
    constant uint& nh  [[buffer(5)]],
    constant uint& hpk [[buffer(6)]],
    constant uint& hd  [[buffer(7)]],
    constant uint& cap [[buffer(8)]],
    constant uint& n   [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint h = tg;
    if (h >= nh) return;
    uint kh = h / hpk;
    device const float* kh0 = kbuf + (ulong)kh * cap * hd;
    device const float* vh0 = vbuf + (ulong)kh * cap * hd;
    uint nt = (hd + 31u) / 32u;
    float qv[16];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? q[(ulong)h * hd + d] * scale : 0.0f;
    }
    // This simdgroup's slice of the stored positions. Contiguous, so
    // the K/V walk stays sequential inside each slice.
    uint per = (n + sgs - 1u) / sgs;
    uint p0 = min(sg * per, n);
    uint p1 = min(p0 + per, n);
    float m = -INFINITY, l = 0.0f;
    float acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = p0; p < p1; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float partial = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) partial += qv[t] * kr[d];
        }
        float s = simd_sum(partial);
        float mp = max(m, s);
        float f = exp(m - mp), w = exp(s - mp);
        l = l * f + w;
        device const float* vr = vh0 + (ulong)p * hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) acc[t] = acc[t] * f + w * vr[d];
        }
        m = mp;
    }
    // Combine the slices: sh = [sgs × hd accumulators | sgs m | sgs l].
    // An empty slice contributes m = -INF, l = 0, acc = 0 — exp(-INF −
    // gm) = 0 kills it in both sums, and n ≥ 1 keeps gm finite.
    threadgroup float* sacc = sh;
    threadgroup float* sm = sh + sgs * hd;
    threadgroup float* sl = sm + sgs;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) sacc[sg * hd + d] = acc[t];
    }
    if (lane == 0) {
        sm[sg] = m;
        sl[sg] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -INFINITY;
    for (uint s = 0; s < sgs; ++s) gm = max(gm, sm[s]);
    float gl = 0.0f;
    for (uint s = 0; s < sgs; ++s) gl += sl[s] * exp(sm[s] - gm);
    float invl = gl > 0.0f ? 1.0f / gl : 0.0f;
    if (sg == 0) {
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float a = 0.0f;
                for (uint s = 0; s < sgs; ++s) a += sacc[s * hd + d] * exp(sm[s] - gm);
                outb[(ulong)h * hd + d] = a * invl;
            }
        }
    }
    m = gm; // the importance pass below wants the head's final max
    // Attention-importance pass: prob_p = exp(s_p − m)/l summed over heads.
    // The score is recomputed in the SAME lane-sliced layout as the main
    // loop. The obvious form — one position per lane, each lane walking
    // a whole K row — makes every lane touch a different 512 B row, so
    // the reads never coalesce: it cost more than the whole rest of the
    // kernel at decode depth (M4, 44 virtual layers: 0.145 ms/position
    // of context vs 0.003 ms bandwidth-bound). Four positions per step
    // so the simd_sum chains overlap; `qv` already carries `scale`.
    // Each simdgroup re-walks its own slice, now with the head's final
    // m/l — the slices tile [0, n), so every position is banked once.
    uint p = p0;
    for (; p + 4u <= p1; p += 4u) {
        device const float* r0 = kh0 + (ulong)p * hd;
        device const float* r1 = r0 + hd;
        device const float* r2 = r1 + hd;
        device const float* r3 = r2 + hd;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float qd = qv[t];
                a0 += qd * r0[d];
                a1 += qd * r1[d];
                a2 += qd * r2[d];
                a3 += qd * r3[d];
            }
        }
        float s0 = simd_sum(a0), s1 = simd_sum(a1);
        float s2 = simd_sum(a2), s3 = simd_sum(a3);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s0 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 1u], exp(s1 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 2u], exp(s2 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 3u], exp(s3 - m) * invl, memory_order_relaxed);
        }
    }
    for (; p < p1; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float a = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) a += qv[t] * kr[d];
        }
        float s = simd_sum(a);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s - m) * invl, memory_order_relaxed);
        }
    }
}

// `gqa_attend_blk` widened to 16 slots per lane (head width <= 512).
kernel void gqa_attend_blk_w16(
    device const float* q    [[buffer(0)]],   // [nb][nh][hd] rope'd
    device const float* kbuf [[buffer(1)]],
    device const float* vbuf [[buffer(2)]],
    device float*       part [[buffer(3)]],
    constant uint& nh   [[buffer(4)]],
    constant uint& hpk  [[buffer(5)]],
    constant uint& hd   [[buffer(6)]],
    constant uint& cap  [[buffer(7)]],
    constant uint& s0   [[buffer(8)]],
    constant uint& nb   [[buffer(9)]],
    constant uint& nblk [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint kh = tg.x, j = tg.y, e = tg.z;
    if (kh * hpk + sg >= nh || e >= nb || j >= nblk) return;
    uint h = kh * hpk + sg;
    uint n = s0 + e + 1u;
    uint p0 = j * 128u;
    uint p1 = min(p0 + 128u, n);
    uint nt = (hd + 31u) / 32u;
    device float* out = part + (((ulong)e * nh + h) * nblk + j) * (hd + 2u);
    if (p0 >= n) {
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) out[d] = 0.0f;
        }
        if (lane == 0) { out[hd] = -INFINITY; out[hd + 1u] = 0.0f; }
        return;
    }
    device const float* kh0 = kbuf + (ulong)kh * cap * hd;
    device const float* vh0 = vbuf + (ulong)kh * cap * hd;
    device const float* qh = q + ((ulong)e * nh + h) * hd;
    float qv[16];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? qh[d] * scale : 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Four positions a step: four independent dot/simd_sum chains in
    // flight instead of one dependent chain per position (the serial
    // loop was latency-bound at ~100 cycles a position).
    uint p = p0;
    for (; p + 4u <= p1; p += 4u) {
        device const float* k0 = kh0 + (ulong)p * hd;
        device const float* k1 = k0 + hd;
        device const float* k2 = k1 + hd;
        device const float* k3 = k2 + hd;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float qd = qv[t];
                a0 += qd * k0[d]; a1 += qd * k1[d]; a2 += qd * k2[d]; a3 += qd * k3[d];
            }
        }
        float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
        float mp = max(max(m, s0), max(max(s1, s2), s3));
        float f = exp(m - mp);
        float w0 = exp(s0 - mp), w1 = exp(s1 - mp), w2 = exp(s2 - mp), w3 = exp(s3 - mp);
        l = l * f + (w0 + w1) + (w2 + w3);
        device const float* v0 = vh0 + (ulong)p * hd;
        device const float* v1 = v0 + hd;
        device const float* v2 = v1 + hd;
        device const float* v3 = v2 + hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) acc[t] = acc[t] * f + (w0 * v0[d] + w1 * v1[d]) + (w2 * v2[d] + w3 * v3[d]);
        }
        m = mp;
    }
    for (; p < p1; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float partial = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) partial += qv[t] * kr[d];
        }
        float sv = simd_sum(partial);
        float mp = max(m, sv);
        float f = exp(m - mp), w = exp(sv - mp);
        l = l * f + w;
        device const float* vr = vh0 + (ulong)p * hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) acc[t] = acc[t] * f + w * vr[d];
        }
        m = mp;
    }
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) out[d] = acc[t];
    }
    if (lane == 0) { out[hd] = m; out[hd + 1u] = l; }
}

