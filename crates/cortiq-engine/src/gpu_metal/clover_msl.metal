// Block-diffusion (Clover) preview kernels on native Metal — the twins of
// `gpu_wgpu::clover`'s WGSL and of the host's `linear_core::gdn_preview_batch`
// / `attention::qwen_attention_preview`.
#include <metal_stdlib>
using namespace metal;

// ── preview conv: causal taps over [ring | block] + right taps, SiLU ──
// out[t] = Σ_j in[t + j]·taps[j] over in = [ring (kk−1 rows) | block] (tap
// kk−1 is row t itself) + Σ_d block[t + d]·right[d − 1]; the ring is read,
// never shifted. One thread a (channel, row).
kernel void bdp_conv(
    device const float* qkv   [[buffer(0)]],   // [kb][cdim]
    device const float* taps  [[buffer(1)]],   // [cdim][kk]
    device const float* right [[buffer(2)]],   // [cdim][kk-1]
    device const float* ring  [[buffer(3)]],   // [kk-1][cdim]
    device float*       cq    [[buffer(4)]],   // [kb][cdim]
    constant uint& cdim [[buffer(5)]],
    constant uint& kk   [[buffer(6)]],
    constant uint& kb   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x, t = gid.y;
    if (c >= cdim || t >= kb) return;
    float acc = 0.0f;
    for (uint j = 0; j < kk; ++j) {
        uint i = t + j;
        float x = (i + 1u < kk) ? ring[(ulong)i * cdim + c] : qkv[(ulong)(i + 1u - kk) * cdim + c];
        acc += x * taps[(ulong)c * kk + j];
    }
    for (uint d = 1; d < kk; ++d) {
        if (t + d < kb) acc += qkv[(ulong)(t + d) * cdim + c] * right[(ulong)c * (kk - 1u) + d - 1u];
    }
    cq[(ulong)t * cdim + c] = acc / (1.0f + exp(-acc));
}

// ── preview mixer: block softmax + history slot, gated RMSNorm ──
// Per (value head, four rows): the softmax over the block rows'
// l2-normalized keys (logits scaled per head, no β) plus one history slot
// whose value is the clean state's readout (q̂/√dk)ᵀS decayed by the row's
// gate exp(g), then the gated RMSNorm of the causal step. S is read only.
// The block's keys and the group's queries are staged in threadgroup memory
// once (coalesced); norms, logits and the softmax are simdgroup reductions;
// a lane owns one output column for every row of the group, so each S
// column is read once for all four rows. Needs kb·dk ≤ 4096, dk, dv ≤ 256,
// 256 % dv == 0.
#define BMR 4u
inline float bdp_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }
kernel void bdp_mix(
    device const float* cq   [[buffer(0)]],    // [kb][cdim] conv output
    device const float* z    [[buffer(1)]],    // [kb][nv·dv]
    device const float* a    [[buffer(2)]],    // [kb][nv]
    device const float* alog [[buffer(3)]],
    device const float* dtb  [[buffer(4)]],
    device const float* nrm  [[buffer(5)]],    // [dv]
    device const float* S    [[buffer(6)]],    // [nv][dk][dv]
    device float*       o    [[buffer(7)]],    // [kb][nv·dv]
    device const float* lsc  [[buffer(8)]],
    device const float* hb   [[buffer(9)]],
    device const float* hc   [[buffer(10)]],
    constant uint& nv   [[buffer(11)]],
    constant uint& dk   [[buffer(12)]],
    constant uint& dv   [[buffer(13)]],
    constant uint& kd   [[buffer(14)]],
    constant uint& rep  [[buffer(15)]],
    constant uint& cdim [[buffer(16)]],
    constant uint& kb   [[buffer(17)]],
    constant uint& gsig [[buffer(18)]],        // output gate: 1 sigmoid, 0 SiLU
    constant float& eps     [[buffer(19)]],
    constant float& len_log [[buffer(20)]],
    uint2 wid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sk[4096];          // the block's keys [kb][dk]
    threadgroup float sq[BMR * 256];     // the group's queries [BMR][dk]
    threadgroup float skn[256 + BMR];    // 1/|k| of the keys, then 1/|q| of the rows
    threadgroup float sw[BMR * 257];
    threadgroup float sov[BMR * 256];
    threadgroup float snr[BMR];
    uint l = lid.x;
    uint h = wid.x;
    uint t0 = wid.y * BMR;
    if (t0 >= kb) return;
    uint ko = h / rep;
    uint nr = min(BMR, kb - t0);
    for (uint i = l; i < kb * dk; i += 256u) {
        uint sx = i / dk, d = i % dk;
        sk[i] = cq[(ulong)sx * cdim + kd + ko * dk + d];
    }
    for (uint i = l; i < nr * dk; i += 256u) {
        uint rr = i / dk, d = i % dk;
        sq[i] = cq[(ulong)(t0 + rr) * cdim + ko * dk + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // norms: a simdgroup a vector — keys 0..kb, then the rows' queries
    for (uint j = sg; j < kb + nr; j += 8u) {
        threadgroup const float* v = j < kb ? sk + j * dk : sq + (j - kb) * dk;
        float acc = 0.0f;
        for (uint d = lane; d < dk; d += 32u) acc += v[d] * v[d];
        acc = simd_sum(acc);
        if (lane == 0) skn[j] = 1.0f / sqrt(acc + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = l; i < nr * dk; i += 256u) sq[i] *= skn[kb + i / dk];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // logits: a simdgroup a (row, key) pair; the history slot after each row
    uint w1 = kb + 1u;
    for (uint p = sg; p < nr * kb; p += 8u) {
        uint rr = p / kb, sx = p % kb;
        float acc = 0.0f;
        for (uint d = lane; d < dk; d += 32u) acc += sq[rr * dk + d] * sk[sx * dk + d];
        acc = simd_sum(acc);
        if (lane == 0) sw[rr * w1 + sx] = acc * skn[sx] * lsc[h];
    }
    if (l < nr) sw[l * w1 + kb] = hb[h] + hc[h] * len_log;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // softmax: a simdgroup a row
    if (sg < nr) {
        uint b0 = sg * w1;
        float mx = -INFINITY;
        for (uint sx = lane; sx <= kb; sx += 32u) mx = max(mx, sw[b0 + sx]);
        mx = simd_max(mx);
        float sum = 0.0f;
        for (uint sx = lane; sx <= kb; sx += 32u) {
            float e = exp(sw[b0 + sx] - mx);
            sw[b0 + sx] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        float inv = 1.0f / sum;
        for (uint sx = lane; sx <= kb; sx += 32u) sw[b0 + sx] *= inv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            // the history weight times the row's decay and 1/√dk
            float g = -exp(alog[h]) * bdp_softplus(a[(ulong)(t0 + sg) * nv + h] + dtb[h]);
            sw[b0 + kb] = sw[b0 + kb] * exp(g) / sqrt((float)dk);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // outputs: lane = (group gi, column c); rows gi, gi + G, … of the group
    uint G = 256u / dv;
    uint c = l % dv, gi = l / dv;
    if (gi < G) {
        ulong sb = (ulong)h * dk * dv;
        uint vo = 2u * kd + h * dv;
        float acc[BMR] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint di = 0; di < dk; ++di) {
            float sv = S[sb + (ulong)di * dv + c];
            for (uint j = 0, rr = gi; rr < nr; ++j, rr += G) acc[j] += sq[rr * dk + di] * sv;
        }
        for (uint j = 0, rr = gi; rr < nr; ++j, rr += G) {
            float ov = acc[j] * sw[rr * w1 + kb];
            for (uint sx = 0; sx < kb; ++sx) ov += sw[rr * w1 + sx] * cq[(ulong)sx * cdim + vo + c];
            sov[rr * dv + c] = ov;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // gated RMSNorm: a simdgroup a row
    if (sg < nr) {
        float ss = 0.0f;
        for (uint cc = lane; cc < dv; cc += 32u) ss += sov[sg * dv + cc] * sov[sg * dv + cc];
        ss = simd_sum(ss);
        if (lane == 0) snr[sg] = 1.0f / sqrt(ss / (float)dv + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = l; i < nr * dv; i += 256u) {
        uint rr = i / dv, cc = i % dv;
        ulong zo = (ulong)(t0 + rr) * nv * dv + h * dv + cc;
        float zz = z[zo];
        float gate = gsig != 0u ? 1.0f / (1.0f + exp(-zz)) : zz / (1.0f + exp(-zz));
        o[zo] = sov[rr * dv + cc] * snr[rr] * nrm[cc] * gate;
    }
}

// ── preview attend: `gqa_attend_blk` with one bound for every row ──
// One threadgroup per (kv head, 128-position block, row), its hpk
// simdgroups the kv head's query heads; every row attends the committed
// prefix AND the whole block (n positions, the block's rows appended to the
// mirror past the prefix). Partials to `part` ([nb][nh][nblk][hd+2]) for
// the main library's `gqa_combine`.
kernel void bdp_attend_blk(
    device const float* q    [[buffer(0)]],   // [nb][nh][hd] rope'd
    device const float* kbuf [[buffer(1)]],
    device const float* vbuf [[buffer(2)]],
    device float*       part [[buffer(3)]],
    constant uint& nh   [[buffer(4)]],
    constant uint& hpk  [[buffer(5)]],
    constant uint& hd   [[buffer(6)]],
    constant uint& cap  [[buffer(7)]],
    constant uint& n    [[buffer(8)]],
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
    float qv[8];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? qh[d] * scale : 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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


