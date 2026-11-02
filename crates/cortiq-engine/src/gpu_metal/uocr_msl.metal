// Unlimited-OCR's DASH attention on Metal (`gpu_metal::uocr`): the twin of the
// WGSL kernels in `gpu_wgpu/uocr.rs`, the semantics in `uocr.rs`.
#include <metal_stdlib>
using namespace metal;

struct DP {
    uint hidden;
    uint d;
    uint pos;
    uint block;
    uint nblk;
    uint k;
    uint m;
    uint close;
    float eps;
    float inv_sqrt;
    float idx_scale;
    float floor;
    float sigma;
    uint window;
    uint nh;
    uint hd;
    uint cap;
    float scale;
    uint rows;
    uint pad;
};

constant float ACT = 10000.0f;
constant float ENTMAX_NEG = -10000.0f;
constant uint NONE = 0xffffffffu;

// The router of a row, one 256-thread group a row: the q and k rows against n1,
// then on one thread RMSNorm and rotation, the key into its slot of the open
// block, the candidate scores, entmax-1.5 over the best 16, the read set and
// its centered log priors. index_dim 128.
kernel void dash_route(
    device const float4* n1 [[buffer(0)]],
    device const float4* w [[buffer(1)]],
    device const float* qn [[buffer(2)]],
    device const float* kn [[buffer(3)]],
    device const float* ang [[buffer(4)]],
    device float* open [[buffer(5)]],
    device const float4* kbar [[buffer(6)]],
    device const uint* coff [[buffer(7)]],
    device const uint* cands [[buffer(8)]],
    device uint* sel [[buffer(9)]],
    device float* sbias [[buffer(10)]],
    constant DP& p [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float rr[256];
    if (row >= p.rows) return;
    const uint lanes = p.hidden / 4u;
    device const float4* x = n1 + (ulong)row * lanes;
    device const float4* wr = w + (ulong)lid * lanes;
    float4 acc = float4(0.0f);
    for (uint c = 0u; c < lanes; ++c) acc += wr[c] * x[c];
    rr[lid] = acc.x + acc.y + acc.z + acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    const uint d = p.d;
    const uint half_d = d / 2u;
    const uint pos = p.pos + row;
    float ssq = 0.0f;
    float ssk = 0.0f;
    for (uint j = 0u; j < d; ++j) {
        ssq += rr[j] * rr[j];
        ssk += rr[d + j] * rr[d + j];
    }
    const float inv_q = rsqrt(ssq / (float)d + p.eps);
    const float inv_k = rsqrt(ssk / (float)d + p.eps);
    float qs[128];
    const uint at = (((pos / p.block) % 3u) * p.block + pos % p.block) * d;
    device const float* ar = ang + (ulong)row * half_d;
    for (uint j = 0u; j < half_d; ++j) {
        const float cc = precise::cos(ar[j]);
        const float ss = precise::sin(ar[j]);
        float a = clamp(clamp(rr[j] * inv_q, -ACT, ACT) * (1.0f + qn[j]), -ACT, ACT);
        float b = clamp(clamp(rr[half_d + j] * inv_q, -ACT, ACT) * (1.0f + qn[half_d + j]), -ACT, ACT);
        qs[j] = a * cc - b * ss;
        qs[half_d + j] = b * cc + a * ss;
        a = clamp(clamp(rr[d + j] * inv_k, -ACT, ACT) * (1.0f + kn[j]), -ACT, ACT);
        b = clamp(clamp(rr[d + half_d + j] * inv_k, -ACT, ACT) * (1.0f + kn[half_d + j]), -ACT, ACT);
        open[at + j] = a * cc - b * ss;
        open[at + half_d + j] = b * cc + a * ss;
    }
    const uint c0 = coff[row];
    const uint ncand = coff[row + 1u] - c0;
    const uint klanes = d / 4u;
    float vals[16];
    uint ids[16];
    const uint nblk = p.nblk == 0u ? (pos + p.block) / p.block : p.nblk;
    const uint m = clamp(min(p.m, nblk), 1u, 16u);
    for (uint t = 0u; t < m; ++t) {
        vals[t] = ENTMAX_NEG;
        ids[t] = NONE;
    }
    uint cnt = 0u;
    for (uint t = 0u; t < ncand; ++t) {
        device const float4* kb = kbar + (ulong)cands[c0 + t] * klanes;
        float s = 0.0f;
        for (uint e = 0u; e < klanes; ++e) {
            s += dot(float4(qs[e * 4u], qs[e * 4u + 1u], qs[e * 4u + 2u], qs[e * 4u + 3u]), kb[e]);
        }
        const float v = clamp(s * p.inv_sqrt, -ACT, ACT) * p.idx_scale;
        if (cnt == m && v <= vals[m - 1u]) continue;
        uint j = min(cnt, m - 1u);
        while (j != 0u && vals[j - 1u] < v) {
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
            --j;
        }
        vals[j] = v;
        ids[j] = cands[c0 + t];
        cnt = min(cnt + 1u, m);
    }
    uint count = 0u;
    if (cnt > 0u) {
        const float top = vals[0];
        float taus[16];
        float sum = 0.0f;
        float sum2 = 0.0f;
        uint support = 0u;
        for (uint t = 0u; t < m; ++t) {
            const float y = (vals[t] - top) / 2.0f;
            sum += y;
            sum2 += y * y;
            const float rho = (float)(t + 1u);
            const float mean = sum / rho;
            const float sq = rho * (sum2 / rho - mean * mean);
            const float delta = max((1.0f - sq) / rho, 0.0f);
            const float tau = mean - precise::sqrt(delta);
            taus[t] = tau;
            if (tau <= y) ++support;
        }
        const float tau_star = taus[max(support, 1u) - 1u];
        const uint k = clamp(min(p.k, m), 1u, 16u);
        float logp[16];
        float mu = 0.0f;
        uint valid = 0u;
        for (uint t = 0u; t < k; ++t) {
            const float y = (vals[t] - top) / 2.0f - tau_star;
            logp[t] = precise::log(max(y > 0.0f ? y * y : 0.0f, p.floor));
            if (ids[t] != NONE) {
                mu += logp[t];
                ++valid;
            }
        }
        mu /= (float)max(valid, 1u);
        for (uint t = 0u; t < k; ++t) {
            if (ids[t] == NONE) continue;
            sel[row * 17u + 1u + count] = ids[t];
            sbias[row * 16u + count] = (logp[t] - mu) / p.sigma;
            ++count;
        }
    }
    sel[row * 17u] = count;
}

// A closing block: the keys against head_cls, their softmax, the mean key.
kernel void dash_close(
    device const float4* open [[buffer(0)]],
    device const float4* hc [[buffer(1)]],
    device float4* kbar [[buffer(2)]],
    constant DP& p [[buffer(3)]],
    uint e [[thread_position_in_grid]])
{
    const uint n = p.d / 4u;
    if (e >= n) return;
    const uint at = (p.close % 3u) * p.block;
    float aw[64];
    float mx = -INFINITY;
    for (uint t = 0u; t < p.block; ++t) {
        float a = 0.0f;
        for (uint c = 0u; c < n; ++c) a += dot(open[(at + t) * n + c], hc[c]);
        aw[t] = clamp(a * p.inv_sqrt, -ACT, ACT);
        mx = max(mx, aw[t]);
    }
    float den = 0.0f;
    for (uint t = 0u; t < p.block; ++t) {
        aw[t] = precise::exp(aw[t] - mx);
        den += aw[t];
    }
    den = 1.0f / max(den, 1.0e-6f);
    float4 acc = float4(0.0f);
    for (uint t = 0u; t < p.block; ++t) acc += aw[t] * den * open[(at + t) * n + e];
    kbar[(ulong)p.close * n + e] = acc;
}

// Attention over the read set, one 256-thread group per (head, row): the score
// pass splits the rows over the threads, the value pass over eight groups of 32
// float4 columns.
kernel void dash_attend(
    device const float4* aq [[buffer(0)]],
    device const float4* ak [[buffer(1)]],
    device const float4* av [[buffer(2)]],
    device float4* ao [[buffer(3)]],
    device const uint* asel [[buffer(4)]],
    device const float* abias [[buffer(5)]],
    constant DP& p [[buffer(6)]],
    uint2 w [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort sgs [[simdgroups_per_threadgroup]])
{
    threadgroup float wsc[512];
    threadgroup uint wrow[512];
    threadgroup float red[32];
    threadgroup float4 vsum[256];
    const uint h = w.x;
    const uint row_q = w.y;
    if (h >= p.nh || row_q >= p.rows) return;
    const uint pos = p.pos + row_q;
    const uint lanes = p.hd / 4u;
    const uint sb = row_q * 17u;
    const uint remote = asel[sb] * p.block;
    const uint total = remote + p.window;
    const uint lo = (uint)max((int)pos + 1 - (int)p.window, 0);
    device const float4* q = aq + (ulong)(row_q * p.nh + h) * lanes;
    device const float4* kh = ak + (ulong)h * p.cap * lanes;
    device const float4* vh = av + (ulong)h * p.cap * lanes;
    float mym = -1.0e30f;
    for (uint part = 0u; part < 2u; ++part) {
        const uint i = lid + part * 256u;
        uint row = NONE;
        float bias = 0.0f;
        if (i < remote) {
            const uint s = i / p.block;
            const uint r = asel[sb + 1u + s] * p.block + i % p.block;
            // The local window wins on overlap.
            if (r < lo) {
                row = r;
                bias = abias[row_q * 16u + s];
            }
        } else if (i < total) {
            const uint r = lo + i - remote;
            if (r <= pos) row = r;
        }
        wrow[i] = row;
        if (row != NONE) {
            device const float4* kk = kh + (ulong)row * lanes;
            float4 acc = float4(0.0f);
            for (uint e = 0u; e < lanes; ++e) acc += q[e] * kk[e];
            const float v = (acc.x + acc.y + acc.z + acc.w) * p.scale + bias;
            wsc[i] = v;
            mym = max(mym, v);
        }
    }
    red[sg] = simd_max(mym);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = red[0];
    for (ushort t = 1; t < sgs; ++t) mx = max(mx, red[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint part = 0u; part < 2u; ++part) {
        const uint i = lid + part * 256u;
        if (wrow[i] != NONE) {
            const float e = precise::exp(wsc[i] - mx);
            wsc[i] = e;
            s += e;
        }
    }
    red[sg] = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float den = red[0];
    for (ushort t = 1; t < sgs; ++t) den += red[t];
    const float inv = 1.0f / den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint quad = lid & 31u;
    const uint warp = lid >> 5u;
    float4 acc = float4(0.0f);
    if (quad < lanes) {
        for (uint i = warp; i < total; i += 8u) {
            const uint r = wrow[i];
            if (r != NONE) acc += wsc[i] * vh[(ulong)r * lanes + quad];
        }
    }
    vsum[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0u && quad < lanes) {
        float4 t = float4(0.0f);
        for (uint g = 0u; g < 8u; ++g) t += vsum[g * 32u + quad];
        ao[(ulong)(row_q * p.nh + h) * lanes + quad] = t * inv;
    }
}

// A prompt chunk's rotation and mirror append: every row by its own angles.
kernel void dash_rope_rows(
    device const float* qr [[buffer(0)]],
    device const float* kr [[buffer(1)]],
    device const float* vr [[buffer(2)]],
    device const float* ang [[buffer(3)]],
    device float* qo [[buffer(4)]],
    device float* kc [[buffer(5)]],
    device float* vc [[buffer(6)]],
    constant DP& p [[buffer(7)]],
    uint3 g [[thread_position_in_grid]])
{
    const uint hd = p.hd;
    const uint half_d = hd / 2u;
    const uint j = g.x;
    const uint h = g.y;
    const uint row = g.z;
    const float a = ang[row * half_d + j];
    const float cc = precise::cos(a);
    const float ss = precise::sin(a);
    const ulong base = (ulong)(row * p.nh + h) * hd;
    const float q0 = qr[base + j];
    const float q1 = qr[base + half_d + j];
    qo[base + j] = q0 * cc - q1 * ss;
    qo[base + half_d + j] = q1 * cc + q0 * ss;
    const ulong mb = ((ulong)h * p.cap + p.pos + row) * hd;
    const float k0 = kr[base + j];
    const float k1 = kr[base + half_d + j];
    kc[mb + j] = k0 * cc - k1 * ss;
    kc[mb + half_d + j] = k1 * cc + k0 * ss;
    vc[mb + j] = vr[base + j];
    vc[mb + half_d + j] = vr[base + half_d + j];
}

// y[row] += x[row]·up[row]: a detached shared expert onto the routed mix.
kernel void dash_add_rows(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const float* up [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 g [[thread_position_in_grid]])
{
    y[(ulong)g.y * n + g.x] += x[(ulong)g.y * n + g.x] * up[g.y];
}
