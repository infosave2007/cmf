// Inkling-2-Mini on Metal (`gpu_metal/inkling.rs`): the stages of the looped
// stack and its routed experts. The semantics are `inkling.rs`'s host stack;
// every kernel takes the call's rows and the first position from `IP`.
#include <metal_stdlib>
using namespace metal;

struct IP {
    uint H, NH, NKV, HD, DR, KC;
    uint n, pos0, nb, bstride, ostride, ring;
    uint E, nq, full, rows, R, inter;
    uint V, bslot, topk, gpr, ostride2, win;
    float eps, qclip, rclip, alpha;
    float floor_, scale, cap, ucap;
    float rscale, pad1, pad2, pad3;
};

inline float tg_sum(float v, threadgroup float* sh, uint tid, uint nt) {
    v = simd_sum(v);
    uint lane = tid & 31u, w = tid >> 5u, nw = (nt + 31u) >> 5u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sh[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0u; i < nw; ++i) t += sh[i];
    return t;
}

inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// The block attention residual: grid (nq, n), 256 threads.
kernel void ink_mix(device const float* blk [[buffer(0)]], device const float* x [[buffer(1)]],
                    device const float* q [[buffer(2)]], device float* out [[buffer(3)]],
                    device const float* nrm [[buffer(4)]], device float* xo [[buffer(5)]],
                    constant IP& p [[buffer(15)]],
                    uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    threadgroup float sh2[32][16];
    threadgroup float lg[8], rn[8];
    threadgroup float shn[32];
    const uint qi = tg.x, r = tg.y, H = p.H, nb = p.nb, ns = nb + 1u;
    device const float* qv = q + (ulong)qi * H;
    // every state's squares and dot product in one pass, their sums through
    // one exchange (tg_sum's order: a simdgroup's sum, then the simdgroups')
    float ss[8], dt[8];
    for (uint s = 0u; s < ns; ++s) {
        ss[s] = 0.0f;
        dt[s] = 0.0f;
    }
    for (uint s = 0u; s < ns; ++s) {
        device const float* sv = s < nb ? blk + (ulong)r * p.bstride + (ulong)s * H : x + (ulong)r * H;
        for (uint i = tid; i < H; i += nt) {
            float v = sv[i];
            ss[s] = fma(v, v, ss[s]);
            dt[s] = fma(v, qv[i], dt[s]);
        }
    }
    const uint lane = tid & 31u, wi = tid >> 5u, nw = (nt + 31u) >> 5u;
    for (uint s = 0u; s < ns; ++s) {
        const float a = simd_sum(ss[s]), b = simd_sum(dt[s]);
        if (lane == 0u) {
            sh2[wi][2u * s] = a;
            sh2[wi][2u * s + 1u] = b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint s = 0u; s < ns; ++s) {
            float a = 0.0f, b = 0.0f;
            for (uint j = 0u; j < nw; ++j) {
                a += sh2[j][2u * s];
                b += sh2[j][2u * s + 1u];
            }
            float inv = rsqrt(a / (float)H + 1e-6f);
            rn[s] = inv;
            lg[s] = b * inv * rsqrt((float)H);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -3.0e38f;
    for (uint s = 0u; s < ns; ++s) mx = max(mx, lg[s]);
    float w[8], den = 0.0f;
    for (uint s = 0u; s < ns; ++s) {
        w[s] = exp(lg[s] - mx);
        den += w[s];
    }
    for (uint s = 0u; s < ns; ++s) w[s] = w[s] / den * rn[s];
    device float* o = out + (ulong)qi * p.ostride + (ulong)r * H;
    float sq = 0.0f;
    for (uint i = tid; i < H; i += nt) {
        float acc = 0.0f;
        for (uint s = 0u; s < ns; ++s) {
            device const float* sv = s < nb ? blk + (ulong)r * p.bstride + (ulong)s * H : x + (ulong)r * H;
            acc = fma(w[s], sv[i], acc);
        }
        o[i] = acc;
        sq = fma(acc, acc, sq);
    }
    // p.V: query 0's row RMS-normed into xo (`ink_rms`'s arithmetic on the
    // values this thread wrote), its own dispatch saved
    if (p.V == 0u || qi != 0u) return;
    sq = tg_sum(sq, shn, tid, nt);
    const float inv = rsqrt(sq / (float)H + p.eps);
    for (uint i = tid; i < H; i += nt) xo[(ulong)r * H + i] = nrm[i] * (o[i] * inv);
}

// Weighted RMS norm of each row: grid n, 256 threads.
kernel void ink_rms(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                    device float* xo [[buffer(2)]], constant IP& p [[buffer(15)]],
                    uint3 r_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint3 nt_v [[threads_per_threadgroup]])
{
    const uint r = r_v.x;
    const uint nt = nt_v.x;

    threadgroup float sh[32];
    const uint H = p.H;
    device const float* xr = x + (ulong)r * H;
    float ss = 0.0f;
    for (uint i = tid; i < H; i += nt) ss = fma(xr[i], xr[i], ss);
    ss = tg_sum(ss, sh, tid, nt);
    const float inv = rsqrt(ss / (float)H + p.eps);
    for (uint i = tid; i < H; i += nt) xo[(ulong)r * H + i] = w[i] * (xr[i] * inv);
}

// A prompt call's message gate as GEMMs: the input rows normed and side by
// side, xn[r] = [x[r] / rms x[r], m[r] / rms m[r]] (2H). Grid n, 256 threads.
kernel void ink_gate_norm(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                          device float* xn [[buffer(2)]], constant IP& p [[buffer(15)]],
                          uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint3 nt_v [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    const uint r = tg_v.x, H = p.H, nt = nt_v.x;
    device const float* xr = x + (ulong)r * H;
    device const float* mr = m + (ulong)r * H;
    float sx = 0.0f, sm = 0.0f;
    for (uint i = tid; i < H; i += nt) {
        sx = fma(xr[i], xr[i], sx);
        sm = fma(mr[i], mr[i], sm);
    }
    sx = tg_sum(sx, sh, tid, nt);
    sm = tg_sum(sm, sh, tid, nt);
    const float ix = rsqrt(sx / (float)H + p.eps), im = rsqrt(sm / (float)H + p.eps);
    device float* o = xn + (ulong)r * 2u * H;
    for (uint i = tid; i < H; i += nt) {
        o[i] = xr[i] * ix;
        o[H + i] = mr[i] * im;
    }
}

// out = (1 − tanh f)·x + tanh g·m over the call's n·H values. Grid (n·H/256), 256.
kernel void ink_gate_mix(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                         device const float* f [[buffer(2)]], device const float* g [[buffer(3)]],
                         device float* out [[buffer(4)]], constant IP& p [[buffer(15)]],
                         uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    const ulong i = (ulong)tg_v.x * 256u + tid;
    if (i < (ulong)p.n * p.H) out[i] = (1.0f - tanh(f[i])) * x[i] + tanh(g[i]) * m[i];
}

// Message gate, first half: grid (R, n), 256 threads.
kernel void ink_gate_down(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                          device const float* down [[buffer(2)]], device float* d [[buffer(3)]],
                          constant IP& p [[buffer(15)]],
                          uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    threadgroup float sh[32];
    const uint k = tg.x, r = tg.y, H = p.H;
    device const float* xr = x + (ulong)r * H;
    device const float* mr = m + (ulong)r * H;
    device const float* dw = down + (ulong)k * 2u * H;
    float sx = 0.0f, sm = 0.0f, ax = 0.0f, am = 0.0f;
    for (uint i = tid; i < H; i += nt) {
        float a = xr[i], b = mr[i];
        sx = fma(a, a, sx);
        sm = fma(b, b, sm);
        ax = fma(dw[i], a, ax);
        am = fma(dw[H + i], b, am);
    }
    // the four sums through one exchange (tg_sum's order)
    threadgroup float4 sh4[32];
    const float4 v4 = float4(simd_sum(sx), simd_sum(sm), simd_sum(ax), simd_sum(am));
    const uint lane = tid & 31u, w = tid >> 5u, nw = (nt + 31u) >> 5u;
    if (lane == 0u) sh4[w] = v4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float4 t = float4(0.0f);
        for (uint j = 0u; j < nw; ++j) t += sh4[j];
        d[(ulong)r * p.R + k] = t.z * rsqrt(t.x / (float)H + p.eps) + t.w * rsqrt(t.y / (float)H + p.eps);
    }
}

// Second half: a simdgroup an output channel. Grid (H/8, n), 256 threads.
kernel void ink_gate_out(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                         device const float* d [[buffer(2)]], device const float* F [[buffer(3)]],
                         device const float* G [[buffer(4)]], device float* out [[buffer(5)]],
                         constant IP& p [[buffer(15)]],
                         uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    const uint i = tg.x * 8u + sg, r = tg.y, H = p.H, R = p.R;
    if (i >= H) return;
    device const float* dr = d + (ulong)r * R;
    float f = 0.0f, g = 0.0f;
    for (uint k = lane; k < R; k += 32u) {
        f = fma(F[(ulong)i * R + k], dr[k], f);
        g = fma(G[(ulong)i * R + k], dr[k], g);
    }
    f = simd_sum(f);
    g = simd_sum(g);
    if (lane == 0u) {
        const ulong at = (ulong)r * H + i;
        out[at] = (1.0f - tanh(f)) * x[at] + tanh(g) * m[at];
    }
}

// y[r][o] = W[o]·x[r] for an f32 matrix [rows][H]: a simdgroup an output row.
// Grid (rows/8, n), 256 threads.
// A lane's share of an f32 row's dot product: float4 loads, two sums in
// flight (rows 16-byte aligned when H % 4 == 0, else scalar).
inline float row_dot(device const float* w, device const float* x, uint H, uint lane) {
    if ((H & 3u) != 0u) {
        float acc = 0.0f;
        for (uint i = lane; i < H; i += 32u) acc = fma(w[i], x[i], acc);
        return acc;
    }
    device const float4* w4 = (device const float4*)w;
    device const float4* x4 = (device const float4*)x;
    const uint H4 = H >> 2u;
    float4 a = float4(0.0f), b = float4(0.0f);
    for (uint i = lane; i < H4; i += 64u) {
        a = fma(w4[i], x4[i], a);
        if (i + 32u < H4) b = fma(w4[i + 32u], x4[i + 32u], b);
    }
    const float4 t = a + b;
    return (t.x + t.y) + (t.z + t.w);
}

kernel void ink_fmv(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                    device float* y [[buffer(2)]], constant IP& p [[buffer(15)]],
                    uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    const uint o = tg.x * 8u + sg, r = tg.y, H = p.H;
    if (o >= p.rows) return;
    const float acc = simd_sum(row_dot(W + (ulong)o * H, x + (ulong)r * H, H, lane));
    if (lane == 0u) y[(ulong)r * p.rows + o] = acc;
}

inline float tau_of(uint pos, float alpha, float floor_) {
    return 1.0f + alpha * log(max((float)(pos + 1u) / floor_, 1.0f));
}

// Queries and relative states in place: grid (NH, n), HD threads.
inline void qr_body(uint h, uint r, uint tid, uint nt, threadgroup float* sh, device float* yq, device float* yr,
                    device const float* qs, device const float* rs, constant IP& p)
{
    const uint HD = p.HD, DR = p.DR, NH = p.NH;
    const float tu = p.full ? tau_of(p.pos0 + r, p.alpha, p.floor_) : 1.0f;
    device float* q = yq + ((ulong)r * NH + h) * HD;
    const float v = q[tid] * qs[h * HD + tid];
    const float n2 = tg_sum(v * v, sh, tid, nt);
    q[tid] = v * min(p.qclip / (sqrt(n2) + 1e-6f), 1.0f) * tu;
    if (tid < 32u) {
        device float* rr = yr + ((ulong)r * NH + h) * DR;
        const float a = tid < DR ? rr[tid] * rs[h * DR + tid] : 0.0f;
        const float t = simd_sum(a * a);
        const float sr = min(p.rclip / (sqrt(t) + 1e-6f), 1.0f);
        if (tid < DR) rr[tid] = a * sr * tu;
    }
}

kernel void ink_qr(device float* yq [[buffer(0)]], device float* yr [[buffer(1)]],
                   device const float* qs [[buffer(2)]], device const float* rs [[buffer(3)]],
                   constant IP& p [[buffer(15)]],
                   uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint3 nt_v [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    qr_body(tg_v.x, tg_v.y, tid, nt_v.x, sh, yq, yr, qs, rs, p);
}

// Keys (y 0) or values (1) of the call's rows into the f16 cache, a KV head
// a threadgroup of HD threads (see `ink_kv` in inkling.cu). `dy` may be the
// zero-length dummy (`full` 0).
inline void kv_body(uint g, uint isv, uint tid, uint nt, threadgroup float* sh,
                    device const float* yk, device const float* yv, device const float* dk, device const float* dv,
                    device const float* tk, device const float* tv, device float* rk, device float* rv,
                    device half* kc, device half* vc, constant IP& p)
{
    const uint HD = p.HD, NKV = p.NKV, KC = p.KC;
    const uint C = NKV * HD, ch = g * HD + tid;
    device const float* y = isv ? yv : yk;
    device const float* dy = isv ? dv : dk;
    device const float* tp = (isv ? tv : tk) + (ulong)ch * KC;
    device float* rg = isv ? rv : rk;
    device half* cache = isv ? vc : kc;
    float hist[8];
    hist[0] = 0.0f;
    for (uint l = 1u; l < KC; ++l) hist[l] = rg[(ulong)(l - 1u) * C + ch];
    const bool dyn = p.full != 0u;
    const float scale = rsqrt((float)((dyn ? 2u : 1u) * KC + 1u));
    for (uint j = 0u; j < p.n; ++j) {
        const float x = y[(ulong)j * C + ch];
        hist[0] = x;
        float acc = 0.0f;
        for (uint l = 0u; l < KC; ++l) {
            float w = tp[l];
            if (dyn) w += dy[((ulong)j * NKV + g) * KC + l];
            acc = fma(w, hist[l], acc);
        }
        float v = (x + acc) * scale;
        for (uint l = KC - 1u; l > 0u; --l) hist[l] = hist[l - 1u];
        const float n2 = tg_sum(v * v, sh, tid, nt);
        if (isv) v *= rsqrt(n2 / (float)HD + p.eps);
        else v *= min(p.qclip / (sqrt(n2) + 1e-6f), 1.0f);
        const uint pos = p.pos0 + j;
        const uint row = p.ring > 0u ? pos % p.ring : pos;
        cache[(ulong)row * C + ch] = (half)v;
    }
    for (uint l = 1u; l < KC; ++l) rg[(ulong)(l - 1u) * C + ch] = hist[l];
}

kernel void ink_kv(device const float* yk [[buffer(0)]], device const float* yv [[buffer(1)]],
                   device const float* dk [[buffer(2)]], device const float* dv [[buffer(3)]],
                   device const float* tk [[buffer(4)]], device const float* tv [[buffer(5)]],
                   device float* rk [[buffer(6)]], device float* rv [[buffer(7)]],
                   device half* kc [[buffer(8)]], device half* vc [[buffer(9)]],
                   constant IP& p [[buffer(15)]],
                   uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint3 nt_v [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    kv_body(tg_v.x, tg_v.y, tid, nt_v.x, sh, yk, yv, dk, dv, tk, tv, rk, rv, kc, vc, p);
}

// `ink_qr` and `ink_kv` in one dispatch (a decode call): threadgroups [0, NH·n)
// the queries (head, row), the next 2·NKV the keys and values; `p.full` names
// a full layer for both (its log scale and its per-token taps). HD threads.
kernel void ink_qrkv(device float* yq [[buffer(0)]], device float* yr [[buffer(1)]],
                     device const float* qs [[buffer(2)]], device const float* rs [[buffer(3)]],
                     device const float* yk [[buffer(4)]], device const float* yv [[buffer(5)]],
                     device const float* dk [[buffer(6)]], device const float* dv [[buffer(7)]],
                     device const float* tk [[buffer(8)]], device const float* tv [[buffer(9)]],
                     device float* rk [[buffer(10)]], device float* rv [[buffer(11)]],
                     device half* kc [[buffer(12)]], device half* vc [[buffer(13)]],
                     constant IP& p [[buffer(15)]],
                     uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint3 nt_v [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    const uint b = tg_v.x, nq = p.NH * p.n;
    if (b < nq) qr_body(b % p.NH, b / p.NH, tid, nt_v.x, sh, yq, yr, qs, rs, p);
    else kv_body((b - nq) % p.NKV, (b - nq) / p.NKV, tid, nt_v.x, sh, yk, yv, dk, dv, tk, tv, rk, rv, kc, vc, p);
}

// The attention of row r and head h (see `ink_attn` in inkling.cu): grid
// (NH, n), 256 threads, HD 128.
kernel void ink_attn(device const float* yq [[buffer(0)]], device const float* yr [[buffer(1)]],
                     device const float* sinks [[buffer(2)]], device const float* table [[buffer(3)]],
                     device const half* kc [[buffer(4)]], device const half* vc [[buffer(5)]],
                     device float* out [[buffer(6)]], device const float* gt [[buffer(7)]],
                     constant IP& p [[buffer(15)]],
                     uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    threadgroup float bias[1024];
    threadgroup float qsh[128];
    threadgroup float wm[8], wl[8];
    threadgroup float wacc[8][128];
    const uint h = tg.x, r = tg.y, HD = p.HD, NKV = p.NKV, NH = p.NH, DR = p.DR, E = p.E;
    const uint C = NKV * HD, g = h / (NH / NKV);
    const uint pos = p.pos0 + r;
    // the window (`win` positions); the ring holds the window and a call's rows
    const uint first = p.win > 0u ? (pos + 1u > p.win ? pos + 1u - p.win : 0u) : 0u;
    const uint nk = pos + 1u - first;
    const uint near = min(E, nk);
    device const float* rr = yr + ((ulong)r * NH + h) * DR;
    for (uint d = tid; d < near; d += 256u) {
        float b = 0.0f;
        for (uint j = 0u; j < DR; ++j) b = fma(rr[j], table[(ulong)j * E + d], b);
        bias[d] = b;
    }
    for (uint i = tid; i < HD; i += 256u) qsh[i] = yq[((ulong)r * NH + h) * HD + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 qv = float4(qsh[lane * 4u], qsh[lane * 4u + 1u], qsh[lane * 4u + 2u], qsh[lane * 4u + 3u]);
    float4 acc = float4(0.0f);
    float m = -3.0e38f, l = 0.0f;
    for (uint i = sg; i < nk; i += 8u) {
        const uint tpos = first + i;
        const uint row = p.ring > 0u ? tpos % p.ring : tpos;
        device const half4* kr = (device const half4*)(kc + (ulong)row * C + g * HD) + lane;
        float s = simd_sum(dot(qv, float4(*kr))) * p.scale;
        const uint d = pos - tpos;
        if (d < near) s += bias[d];
        const float mn = max(m, s);
        const float c = exp(m - mn), e = exp(s - mn);
        l = l * c + e;
        device const half4* vr = (device const half4*)(vc + (ulong)row * C + g * HD) + lane;
        acc = acc * c + e * float4(*vr);
        m = mn;
    }
    if (lane == 0u) {
        wm[sg] = m;
        wl[sg] = l;
    }
    wacc[sg][lane * 4u] = acc.x;
    wacc[sg][lane * 4u + 1u] = acc.y;
    wacc[sg][lane * 4u + 2u] = acc.z;
    wacc[sg][lane * 4u + 3u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sk = sinks[(ulong)r * NH + h];
    float M = sk;
    for (uint i = 0u; i < 8u; ++i) M = max(M, wm[i]);
    float L = exp(sk - M);
    for (uint i = 0u; i < 8u; ++i) L += wl[i] * exp(wm[i] - M);
    for (uint i = tid; i < HD; i += 256u) {
        float a = 0.0f;
        for (uint k = 0u; k < 8u; ++k) a += wacc[k][i] * exp(wm[k] - M);
        const ulong at = ((ulong)r * NH + h) * HD + i;
        // p.V: `ink_ogate` here (σ(gate) on the output), its own dispatch saved
        out[at] = p.V != 0u ? (a / L) * sigm(gt[at]) : a / L;
    }
}

// o · σ(gate) in place: grid (NH*HD/256, n).
kernel void ink_ogate(device float* o [[buffer(0)]], device const float* gt [[buffer(1)]],
                      constant IP& p [[buffer(15)]],
                      uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    const uint W = p.NH * p.HD;
    const uint i = tg.x * nt + tid, r = tg.y;
    if (i >= W) return;
    const ulong at = (ulong)r * W + i;
    o[at] = o[at] * sigm(gt[at]);
}

// A sub-layer's output convolution over the call's rows (see `ink_post` in
// inkling.cu); `p.bslot` 0xFFFFFFFF: no block closes. Grid H/256.
kernel void ink_post(device const float* a [[buffer(0)]], device const float* taps [[buffer(1)]],
                     device float* ring [[buffer(2)]], device const float* res [[buffer(3)]],
                     device float* blk [[buffer(4)]], device float* out [[buffer(5)]],
                     constant IP& p [[buffer(15)]],
                     uint3 tgx_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint3 nt_v [[threads_per_threadgroup]])
{
    const uint tgx = tgx_v.x;
    const uint nt = nt_v.x;

    const uint i = tgx * nt + tid, H = p.H, KC = p.KC;
    if (i >= H) return;
    float hist[8];
    hist[0] = 0.0f;
    for (uint l = 1u; l < KC; ++l) hist[l] = ring[(ulong)(l - 1u) * H + i];
    device const float* tp = taps + (ulong)i * KC;
    const float scale = rsqrt((float)(KC + 1u));
    const bool close = p.bslot != 0xFFFFFFFFu;
    for (uint j = 0u; j < p.n; ++j) {
        const ulong at = (ulong)j * H + i;
        const float x = a[at];
        hist[0] = x;
        float acc = 0.0f;
        for (uint l = 0u; l < KC; ++l) acc = fma(tp[l], hist[l], acc);
        const float c = (x + acc) * scale;
        for (uint l = KC - 1u; l > 0u; --l) hist[l] = hist[l - 1u];
        const float hv = res[at];
        if (close) {
            blk[(ulong)j * p.bstride + (ulong)p.bslot * H + i] = hv;
            out[at] = c;
        } else {
            out[at] = hv + c;
        }
    }
    for (uint l = 1u; l < KC; ++l) ring[(ulong)(l - 1u) * H + i] = hist[l];
}

// dst[i] = src[i] for i < p.rows (in order with the dispatches around it).
kernel void ink_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                     constant IP& p [[buffer(15)]], uint3 i_v [[thread_position_in_grid]])
{
    const uint i = i_v.x;

    if (i < p.rows) dst[i] = src[i];
}

// cap · tanh(x / cap): grid (V/256, n).
kernel void ink_softcap(device float* lg [[buffer(0)]], constant IP& p [[buffer(15)]],
                        uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    const uint i = tg.x * nt + tid, r = tg.y;
    if (i >= p.V) return;
    const ulong at = (ulong)r * p.V + i;
    lg[at] = tanh(lg[at] / p.cap) * p.cap;
}

// ---------------------------------------------------------------- experts

// Router logits: a simdgroup an expert. Grid (E/8, n), 256 threads.
kernel void ink_router(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* lg [[buffer(2)]], constant IP& p [[buffer(15)]],
                       uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    const uint e = tg.x * 8u + sg, r = tg.y, H = p.H;
    if (e >= p.E) return;
    const float acc = simd_sum(row_dot(W + (ulong)e * H, x + (ulong)r * H, H, lane));
    if (lane == 0u) lg[(ulong)r * p.E + e] = acc;
}

// A row's top-k by σ(logit) + bias (ties to the lower expert), weights the
// chosen scores over their sum (+ 1e-20) times the route scale. A thread an
// expert (E ≤ 128), k argmax rounds over the threadgroup. Grid n, 128 threads.
kernel void ink_select(device const float* lg [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device uint* pick [[buffer(2)]], device float* wt [[buffer(3)]],
                       constant IP& p [[buffer(15)]], uint3 r_v [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    const uint r = r_v.x;

    threadgroup float ks[128];
    threadgroup uint is[128];
    threadgroup float chosen_sc[16];
    device const float* l = lg + (ulong)r * p.E;
    float key = tid < p.E ? sigm(l[tid]) + bias[tid] : -3.0e38f;
    for (uint k = 0u; k < p.topk; ++k) {
        ks[tid] = key;
        is[tid] = tid;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 64u; st > 0u; st >>= 1u) {
            if (tid < st) {
                const float ka = ks[tid], kb = ks[tid + st];
                const uint ia = is[tid], ib = is[tid + st];
                if (kb > ka || (kb == ka && ib < ia)) {
                    ks[tid] = kb;
                    is[tid] = ib;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint bi = is[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == bi) key = -3.0e38f;
        if (tid == 0u) {
            pick[r * p.topk + k] = bi;
            chosen_sc[k] = sigm(l[bi]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint k = 0u; k < p.topk; ++k) sum += chosen_sc[k];
        for (uint k = 0u; k < p.topk; ++k) wt[r * p.topk + k] = chosen_sc[k] / (sum + 1e-20f) * p.rscale;
    }
}

inline float4 q4_nib4_scaled(uint u) {
    return float4((float)(u & 0xFu), (float)(u & 0xF0u), (float)(u & 0xF00u), (float)(u & 0xF000u));
}

// The window holding absolute byte `ab` of the mapping: up to four windows,
// `wstride` apart (0: one).
inline device const uchar* at_abs(ulong ab, device const uchar* q0, device const uchar* q1,
                                  device const uchar* q2, device const uchar* q3, ulong wstride) {
    if (wstride == 0ul) return q0 + ab;
    uint wi = (uint)(ab / wstride);
    ulong rel = ab - (ulong)wi * wstride;
    return ((wi == 0u) ? q0 : (wi == 1u) ? q1 : (wi == 2u) ? q2 : q3) + rel;
}

// A q4tp row block's dot products with one input row (`q4tp_matvec_m`'s
// arithmetic): rows r0..r0+nr of the matrix at `w`, gpr groups, into acc.
inline void q4tp_rows(device const uchar* w, uint rows, uint gpr, uint r0, uint nr, device const float* x,
                      threadgroup float* lad, uint lane, thread float* acc)
{
    const ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    const ulong codes_off = params_off + (ulong)rows * 4ul;
    const uint stride = (gpr * 5u + 7u) / 8u;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    for (uint ri = 0u; ri < 4u; ++ri) acc[ri] = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        device const float4* xv = (device const float4*)(x + g * 32u);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        float4 s4 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
        float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);
        x0 *= sc; x1 *= sc; x2 *= sc; x3 *= sc; x4 *= sc; x5 *= sc; x6 *= sc; x7 *= sc;
        uint bit = g * 5u, cb = bit >> 3u, shf = bit & 7u;
        for (uint ri = 0u; ri < nr; ++ri) {
            uint r = r0 + ri;
            uint4 b = *(device const uint4*)(w + ((ulong)r * gpr + (ulong)g) * 16ul);
            device const uchar* cp = w + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[ri * 32u + code];
            float gsum = dot(q4_nib4_scaled(b.x), x0) + dot(q4_nib4_scaled(b.x >> 16), x1)
                       + dot(q4_nib4_scaled(b.y), x2) + dot(q4_nib4_scaled(b.y >> 16), x3)
                       + dot(q4_nib4_scaled(b.z), x4) + dot(q4_nib4_scaled(b.z >> 16), x5)
                       + dot(q4_nib4_scaled(b.w), x6) + dot(q4_nib4_scaled(b.w >> 16), x7);
            acc[ri] += scale * (gsum + m8);
        }
    }
    for (uint ri = 0u; ri < 4u; ++ri) acc[ri] = simd_sum(acc[ri]);
}

inline void ladder(device const uchar* w, uint rows, uint gpr, uint r0, uint nr, threadgroup float* lad, uint lane) {
    const ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(w + params_off + (ulong)(r0 + ri) * 4ul);
        lad[ri * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
}

// Every pick's gate and up rows, Inkling's softcapped SwiGLU into
// act[r][k][inter]. `bases` [E][3] absolute offsets of gate, up, down.
// Grid (inter/16, topk, n), 128 threads (4 simdgroups × 4 rows).
kernel void ink_moe_gu(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                       device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                       device const ulong* bases [[buffer(4)]], device const uint* pick [[buffer(5)]],
                       device const float* x [[buffer(6)]], device float* act [[buffer(7)]],
                       constant ulong& wstride [[buffer(8)]], constant IP& p [[buffer(15)]],
                       uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    const uint3 tg = tg_v;

    threadgroup float lad[4][2][4 * 32];
    const uint k = tg.y, r = tg.z, inter = p.inter, gpr = p.H / 32u;
    const uint e = pick[r * p.topk + k];
    device const uchar* wg = at_abs(bases[e * 3u], q0, q1, q2, q3, wstride);
    device const uchar* wu = at_abs(bases[e * 3u + 1u], q0, q1, q2, q3, wstride);
    const uint r0 = (tg.x * 4u + sg) * 4u;
    const bool active = r0 < inter;
    const uint nr = active ? min(inter - r0, 4u) : 0u;
    ladder(wg, inter, gpr, r0, nr, lad[sg][0], lane);
    ladder(wu, inter, gpr, r0, nr, lad[sg][1], lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    device const float* xr = x + (ulong)r * p.H;
    float ag[4], au[4];
    q4tp_rows(wg, inter, gpr, r0, nr, xr, lad[sg][0], lane, ag);
    q4tp_rows(wu, inter, gpr, r0, nr, xr, lad[sg][1], lane, au);
    if (lane == 0u) {
        device float* a = act + ((ulong)r * p.topk + k) * inter;
        for (uint ri = 0u; ri < nr; ++ri) {
            const float g = ag[ri], u = au[ri];
            a[r0 + ri] = (p.cap * tanh(g / p.cap)) * sigm(g) * (p.ucap * tanh(u / p.ucap));
        }
    }
}

// Every row's mixture of its picks' down projections: y[r] = Σ_k w[r][k]·down_e(act[r][k]).
// Grid (H/16, n), 128 threads.
kernel void ink_moe_down(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                         device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                         device const ulong* bases [[buffer(4)]], device const uint* pick [[buffer(5)]],
                         device const float* wt [[buffer(6)]], device const float* act [[buffer(7)]],
                         device float* y [[buffer(8)]], constant ulong& wstride [[buffer(9)]],
                         constant IP& p [[buffer(15)]],
                         uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    threadgroup float lad[4][4 * 32];
    const uint r = tg.y, H = p.H, inter = p.inter, gpr = inter / 32u;
    const uint r0 = (tg.x * 4u + sg) * 4u;
    const bool active = r0 < H;
    const uint nr = active ? min(H - r0, 4u) : 0u;
    float tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < p.topk; ++k) {
        const uint e = pick[r * p.topk + k];
        device const uchar* wd = at_abs(bases[e * 3u + 2u], q0, q1, q2, q3, wstride);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ladder(wd, H, gpr, r0, nr, lad[sg], lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            float acc[4];
            q4tp_rows(wd, H, gpr, r0, nr, act + ((ulong)r * p.topk + k) * inter, lad[sg], lane, acc);
            const float w = wt[r * p.topk + k];
            for (uint ri = 0u; ri < 4u; ++ri) tot[ri] += w * acc[ri];
        }
    }
    if (active && lane == 0u) {
        for (uint ri = 0u; ri < nr; ++ri) y[(ulong)r * H + r0 + ri] = tot[ri];
    }
}

// ---- the decode's experts with every lane busy ----
// `ink_moe_gu` / `ink_moe_down` give a simdgroup's 32 lanes a row block's
// groups: 72 groups (gate/up) leave a third pass to 8 lanes and 36 (down)
// a second to 4. Here gate/up splits the lanes into two halves of 16 over
// four rows each and down into four quarters of 8, a pick each: ~90 % of the
// lanes in every pass. A group's scale is exp2(lo + code·step) directly.

// One q4tp group of row `row` (`gpr` groups, `codes_off`/`stride` the codes)
// against x's 32 values pre-scaled by (1, 1/16, 1/256, 1/4096) (`m8` = −8·Σx).
inline float q4tp_group(device const uchar* w, ulong row, uint gpr, uint g, ulong params_off, ulong codes_off, uint stride,
                        float4 x0, float4 x1, float4 x2, float4 x3, float4 x4, float4 x5, float4 x6, float4 x7, float m8)
{
    const uint4 b = *(device const uint4*)(w + (row * gpr + (ulong)g) * 16ul);
    const uint bit = g * 5u, cb = bit >> 3u, shf = bit & 7u;
    device const uchar* cp = w + codes_off + row * (ulong)stride + cb;
    const uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
    device const half* ph = (device const half*)(w + params_off + row * 4ul);
    const float scale = exp2((float)ph[0] + (float)code * (float)ph[1]);
    const float gsum = dot(q4_nib4_scaled(b.x), x0) + dot(q4_nib4_scaled(b.x >> 16), x1)
                     + dot(q4_nib4_scaled(b.y), x2) + dot(q4_nib4_scaled(b.y >> 16), x3)
                     + dot(q4_nib4_scaled(b.z), x4) + dot(q4_nib4_scaled(b.z >> 16), x5)
                     + dot(q4_nib4_scaled(b.w), x6) + dot(q4_nib4_scaled(b.w >> 16), x7);
    return scale * (gsum + m8);
}

#define LOAD_X32(xp)                                                                    \
    device const float4* xv = (device const float4*)(xp);                               \
    float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];                              \
    float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];                              \
    const float4 s4 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;                            \
    const float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);                               \
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);         \
    x0 *= sc; x1 *= sc; x2 *= sc; x3 *= sc; x4 *= sc; x5 *= sc; x6 *= sc; x7 *= sc;

// Every pick's gate and up rows, eight rows a simdgroup (a half of 16 lanes
// on four), Inkling's softcapped SwiGLU into act[r][k][inter].
// Grid (inter/32, topk, n), 128 threads.
kernel void ink_moe_gu8(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                        device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                        device const ulong* bases [[buffer(4)]], device const uint* pick [[buffer(5)]],
                        device const float* x [[buffer(6)]], device float* act [[buffer(7)]],
                        constant ulong& wstride [[buffer(8)]], constant IP& p [[buffer(15)]],
                        uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    const uint k = tg_v.y, r = tg_v.z, inter = p.inter, gpr = p.H / 32u;
    const uint e = pick[r * p.topk + k];
    device const uchar* wg = at_abs(bases[e * 3u], q0, q1, q2, q3, wstride);
    device const uchar* wu = at_abs(bases[e * 3u + 1u], q0, q1, q2, q3, wstride);
    const uint hf = lane >> 4u, l16 = lane & 15u;
    const uint r0 = (tg_v.x * 4u + sg) * 8u + hf * 4u;
    const uint nr = r0 < inter ? min(inter - r0, 4u) : 0u;
    const ulong params_off = (ulong)inter * (ulong)gpr * 16ul;
    const ulong codes_off = params_off + (ulong)inter * 4ul;
    const uint stride = (gpr * 5u + 7u) / 8u;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* xr = x + (ulong)r * p.H;
    if (nr > 0u) {
        for (uint g = l16; g < gpr; g += 16u) {
            LOAD_X32(xr + g * 32u)
            for (uint ri = 0u; ri < nr; ++ri) {
                const ulong row = (ulong)(r0 + ri);
                ag[ri] += q4tp_group(wg, row, gpr, g, params_off, codes_off, stride, x0, x1, x2, x3, x4, x5, x6, x7, m8);
                au[ri] += q4tp_group(wu, row, gpr, g, params_off, codes_off, stride, x0, x1, x2, x3, x4, x5, x6, x7, m8);
            }
        }
    }
    for (uint ri = 0u; ri < 4u; ++ri) {
        for (uint o = 8u; o > 0u; o >>= 1u) {
            ag[ri] += simd_shuffle_xor(ag[ri], o);
            au[ri] += simd_shuffle_xor(au[ri], o);
        }
    }
    if (l16 == 0u && nr > 0u) {
        device float* a = act + ((ulong)r * p.topk + k) * inter;
        for (uint ri = 0u; ri < nr; ++ri) {
            const float g = ag[ri], u = au[ri];
            a[r0 + ri] = (p.cap * tanh(g / p.cap)) * sigm(g) * (p.ucap * tanh(u / p.ucap));
        }
    }
}

// Every row's mixture of its picks' downs, four rows a simdgroup and a pick a
// quarter of 8 lanes: y[r] = Σ_k w[r][k]·down_e(act[r][k]). Grid (H/16, n), 128.
kernel void ink_moe_dn8(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                        device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                        device const ulong* bases [[buffer(4)]], device const uint* pick [[buffer(5)]],
                        device const float* wt [[buffer(6)]], device const float* act [[buffer(7)]],
                        device float* y [[buffer(8)]], constant ulong& wstride [[buffer(9)]],
                        constant IP& p [[buffer(15)]],
                        uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    const uint r = tg_v.y, H = p.H, inter = p.inter, gpr = inter / 32u, K = p.topk;
    const uint qt = lane >> 3u, l8 = lane & 7u;
    const uint r0 = (tg_v.x * 4u + sg) * 4u;
    const uint nr = r0 < H ? min(H - r0, 4u) : 0u;
    const ulong params_off = (ulong)H * (ulong)gpr * 16ul;
    const ulong codes_off = params_off + (ulong)H * 4ul;
    const uint stride = (gpr * 5u + 7u) / 8u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (nr > 0u) {
        for (uint k = qt; k < K; k += 4u) {
            const uint e = pick[r * K + k];
            device const uchar* wd = at_abs(bases[e * 3u + 2u], q0, q1, q2, q3, wstride);
            device const float* xr = act + ((ulong)r * K + k) * inter;
            float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint g = l8; g < gpr; g += 8u) {
                LOAD_X32(xr + g * 32u)
                for (uint ri = 0u; ri < nr; ++ri)
                    t[ri] += q4tp_group(wd, (ulong)(r0 + ri), gpr, g, params_off, codes_off, stride, x0, x1, x2, x3, x4, x5, x6, x7, m8);
            }
            const float w = wt[r * K + k];
            for (uint ri = 0u; ri < 4u; ++ri) acc[ri] += w * t[ri];
        }
    }
    for (uint ri = 0u; ri < 4u; ++ri) acc[ri] = simd_sum(acc[ri]);
    if (lane == 0u && nr > 0u) {
        for (uint ri = 0u; ri < nr; ++ri) y[(ulong)r * H + r0 + ri] = acc[ri];
    }
}

// ---- a prompt call's routed experts, grouped by expert ----
// The decode kernels above read a pick's expert once a (row, pick); a prompt
// call groups its (row, pick) entries by expert, and an expert's rows read
// its weights once a call (the ladders once, the planes from the cache).

// The entries grouped: grp[0..E] the first entry of each expert in expert
// order, grp[E] the count, grp[E + 1 + q] the q-th entry (r·topk + k), and
// grp[E + 1 + nk + entry] its place q. One threadgroup, 256 threads, an
// expert a thread (E ≤ 1024 in turns).
kernel void ink_group(device const uint* pick [[buffer(0)]], device uint* grp [[buffer(1)]],
                      constant IP& p [[buffer(15)]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint cnt[1024];
    const uint E = p.E, nk = p.n * p.topk;
    for (uint e = tid; e < E; e += 256u) {
        uint c = 0u;
        for (uint i = 0u; i < nk; ++i) c += (pick[i] == e) ? 1u : 0u;
        cnt[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint o = 0u;
        for (uint e = 0u; e < E; ++e) {
            const uint c = cnt[e];
            cnt[e] = o;
            o += c;
        }
        grp[E] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < E; e += 256u) {
        uint q = cnt[e];
        grp[e] = q;
        for (uint i = 0u; i < nk; ++i)
            if (pick[i] == e) {
                grp[E + 1u + nk + i] = q;
                grp[E + 1u + q++] = i;
            }
    }
}

// An expert's gate and up rows over its entries, the softcapped SwiGLU into
// act[entry][inter]. Grid (inter/16, E), 128 threads (4 simdgroups × 4 rows).
kernel void ink_moe_gu_g(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                         device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                         device const ulong* bases [[buffer(4)]], device const uint* grp [[buffer(5)]],
                         device const float* x [[buffer(6)]], device float* act [[buffer(7)]],
                         constant ulong& wstride [[buffer(8)]], constant IP& p [[buffer(15)]],
                         uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]])
{
    const uint3 tg = tg_v;
    threadgroup float lad[4][2][4 * 32];
    const uint e = tg.y, inter = p.inter, gpr = p.H / 32u, E = p.E;
    const uint i0 = grp[e], i1 = (e + 1u < E) ? grp[e + 1u] : grp[E];
    if (i1 <= i0) return;
    device const uchar* wg = at_abs(bases[e * 3u], q0, q1, q2, q3, wstride);
    device const uchar* wu = at_abs(bases[e * 3u + 1u], q0, q1, q2, q3, wstride);
    const uint r0 = (tg.x * 4u + sg) * 4u;
    const bool active = r0 < inter;
    const uint nr = active ? min(inter - r0, 4u) : 0u;
    ladder(wg, inter, gpr, r0, nr, lad[sg][0], lane);
    ladder(wu, inter, gpr, r0, nr, lad[sg][1], lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    for (uint i = i0; i < i1; ++i) {
        const uint ent = grp[E + 1u + i];
        device const float* xr = x + (ulong)(ent / p.topk) * p.H;
        float ag[4], au[4];
        q4tp_rows(wg, inter, gpr, r0, nr, xr, lad[sg][0], lane, ag);
        q4tp_rows(wu, inter, gpr, r0, nr, xr, lad[sg][1], lane, au);
        if (lane == 0u) {
            device float* a = act + (ulong)ent * inter;
            for (uint ri = 0u; ri < nr; ++ri) {
                const float g = ag[ri], u = au[ri];
                a[r0 + ri] = (p.cap * tanh(g / p.cap)) * sigm(g) * (p.ucap * tanh(u / p.ucap));
            }
        }
    }
}

// An expert's down rows over its entries into yp[entry][H] (unweighted).
// Grid (H/16, E), 128 threads.
kernel void ink_moe_down_g(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                           device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                           device const ulong* bases [[buffer(4)]], device const uint* grp [[buffer(5)]],
                           device const float* act [[buffer(6)]], device float* yp [[buffer(7)]],
                           constant ulong& wstride [[buffer(8)]], constant IP& p [[buffer(15)]],
                           uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]])
{
    const uint3 tg = tg_v;
    threadgroup float lad[4][4 * 32];
    const uint e = tg.y, H = p.H, inter = p.inter, gpr = inter / 32u, E = p.E;
    const uint i0 = grp[e], i1 = (e + 1u < E) ? grp[e + 1u] : grp[E];
    if (i1 <= i0) return;
    device const uchar* wd = at_abs(bases[e * 3u + 2u], q0, q1, q2, q3, wstride);
    const uint r0 = (tg.x * 4u + sg) * 4u;
    const bool active = r0 < H;
    const uint nr = active ? min(H - r0, 4u) : 0u;
    ladder(wd, H, gpr, r0, nr, lad[sg], lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    for (uint i = i0; i < i1; ++i) {
        const uint ent = grp[E + 1u + i];
        float acc[4];
        q4tp_rows(wd, H, gpr, r0, nr, act + (ulong)ent * inter, lad[sg], lane, acc);
        if (lane == 0u) {
            device float* o = yp + (ulong)ent * H;
            for (uint ri = 0u; ri < nr; ++ri) o[r0 + ri] = acc[ri];
        }
    }
}

// Every row's mixture of its picks' downs: y[r] = Σ_k wt[r][k]·yp[r·topk + k].
// Grid (H/256, n), 256 threads.
kernel void ink_moe_mix(device const float* yp [[buffer(0)]], device const float* wt [[buffer(1)]],
                        device float* y [[buffer(2)]], constant IP& p [[buffer(15)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    const uint i = tg.x * 256u + tid, r = tg.y, H = p.H;
    if (i >= H) return;
    float acc = 0.0f;
    for (uint k = 0u; k < p.topk; ++k) acc += wt[r * p.topk + k] * yp[((ulong)r * p.topk + k) * H + i];
    y[(ulong)r * H + i] = acc;
}

// ---- the grouped experts as simdgroup GEMMs (`q4tp_mul_mm` of the backend,
// its inputs gathered by expert): Y[q][rows] = W_e · X[q] for the places q of
// expert e. which 0 / 1: the gate / up of e over the gathered rows
// x[entry / topk] (cols H); 2: the down of e over A[q] (cols inter). Grid
// (rows of a call / 32, out rows / 64, E), 128 threads; a tile past the
// expert's entries returns at once.
kernel void ink_moe_mm(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                       device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                       device const ulong* bases [[buffer(4)]], device const uint* grp [[buffer(5)]],
                       device const float* xs [[buffer(6)]], device float* y [[buffer(7)]],
                       constant ulong& wstride [[buffer(8)]], constant IP& p [[buffer(15)]],
                       uint tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint3 tg [[threadgroup_position_in_grid]])
{
    const uint which = p.V, e = tg.z, E = p.E, nk = p.n * p.topk;
    const uint i0 = grp[e], i1 = (e + 1u < E) ? grp[e + 1u] : grp[E];
    const uint nb = i1 - i0;
    const uint r1 = tg.x * 32u;
    if (r1 >= nb) return;
    const uint cols = which == 2u ? p.inter : p.H;
    const uint rows = which == 2u ? p.H : p.inter;
    device const uchar* q = at_abs(bases[e * 3u + which], q0, q1, q2, q3, wstride);
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    if (r0 >= rows) return;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);
    // this thread's input row: a gathered row of x, or the place's row of A
    const uint qp = i0 + r1 + lr1;
    const ulong xrow = which == 2u ? (ulong)qp : (ulong)(grp[E + 1u + qp] / p.topk);
    device const float* yrow = xs + xrow * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off = params_off + (ulong)rows * 4ul;
    uint cstride = (gpr * 5u + 7u) / 8u;
    device const half* prow = (device const half*)(q + params_off + (ulong)(r0 + lr0) * 4ul);
    float row_lo = (float)prow[0];
    float row_st = (float)prow[1];
    device const uchar* codes_row = q + codes_off + (ulong)(r0 + lr0) * (ulong)cstride;
    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint g = k0 >> 5u;
            uint bit = g * 5u;
            uint shf = bit & 7u;
            device const uchar* cp = codes_row + (bit >> 3u);
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = exp2(row_lo + (float)code * row_st);
            device const uchar* nib = q + ((ulong)(r0 + lr0) * gpr + (ulong)g) * 16ul + 8u * il0;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            uint2 nb2 = *(device const uint2*)nib;
            for (uint i = 0; i < 8u; ++i) {
                uint bb = ((i < 4u ? nb2.x : nb2.y) >> (8u * (i & 3u))) & 0xFFu;
                wv[2u * i]      = ((float)(bb & 0xFu) - 8.0f) * scale;
                wv[2u * i + 1u] = ((float)(bb >> 4u) - 8.0f) * scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)v0.x; dst[1] = (half)v0.y;
            dst[2] = (half)v0.z; dst[3] = (half)v0.w;
            dst[4] = (half)v1.x; dst[5] = (half)v1.y;
            dst[6] = (half)v1.z; dst[7] = (half)v1.w;
        }
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    // the tile's outputs at the expert's places i0 + r1 + j
    device float* yb = y + (ulong)(i0 + r1) * rows;
    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = yb + (r0 + 32u * (sgitg & 1u)) + (ulong)(16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4), rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem) + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4), 64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = yb + r0 + (ulong)j * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) D[i] = Cr[i];
            }
        }
    }
}

// The softcapped SwiGLU of the places' gate and up rows: A = f(G, U).
// Grid (inter/256, nk), 256 threads.
kernel void ink_moe_act(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],
                        device float* A [[buffer(2)]], constant IP& p [[buffer(15)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    const uint i = tg.x * 256u + tid, qp = tg.y, inter = p.inter;
    if (i >= inter || qp >= p.n * p.topk) return;
    const ulong at = (ulong)qp * inter + i;
    const float g = G[at], u = U[at];
    A[at] = (p.cap * tanh(g / p.cap)) * sigm(g) * (p.ucap * tanh(u / p.ucap));
}

// Every row's mixture of its picks' downs from the places: y[r] = Σ_k
// wt[r][k]·Yp[place(r·topk + k)]. Grid (H/256, n), 256 threads.
kernel void ink_moe_mixq(device const float* yp [[buffer(0)]], device const float* wt [[buffer(1)]],
                         device float* y [[buffer(2)]], device const uint* grp [[buffer(3)]],
                         constant IP& p [[buffer(15)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    const uint i = tg.x * 256u + tid, r = tg.y, H = p.H, nk = p.n * p.topk;
    if (i >= H) return;
    float acc = 0.0f;
    for (uint k = 0u; k < p.topk; ++k) {
        const uint qp = grp[p.E + 1u + nk + r * p.topk + k];
        acc += wt[r * p.topk + k] * yp[(ulong)qp * H + i];
    }
    y[(ulong)r * H + i] = acc;
}


// Three `ink_fmv`s in one dispatch (the same sums): rows [0, rows.x) of
// W0·x0, then rows.y of W1·x1, then rows.z of W2·x2. Grid (rows/8, n), 256.
kernel void ink_fmv3(device const float* W0 [[buffer(0)]], device const float* x0 [[buffer(1)]],
                     device float* y0 [[buffer(2)]], device const float* W1 [[buffer(3)]],
                     device const float* x1 [[buffer(4)]], device float* y1 [[buffer(5)]],
                     device const float* W2 [[buffer(6)]], device const float* x2 [[buffer(7)]],
                     device float* y2 [[buffer(8)]], constant uint3& rr [[buffer(9)]],
                     constant IP& p [[buffer(15)]],
                     uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{
    uint o = tg_v.x * 8u + sg;
    const uint r = tg_v.y, H = p.H;
    device const float* W;
    device const float* x;
    device float* y;
    uint rows;
    if (o < rr.x) { W = W0; x = x0; y = y0; rows = rr.x; }
    else if (o < rr.x + rr.y) { o -= rr.x; W = W1; x = x1; y = y1; rows = rr.y; }
    else if (o < rr.x + rr.y + rr.z) { o -= rr.x + rr.y; W = W2; x = x2; y = y2; rows = rr.z; }
    else return;
    const float acc = simd_sum(row_dot(W + (ulong)o * H, x + (ulong)r * H, H, lane));
    if (lane == 0u) y[(ulong)r * rows + o] = acc;
}

// A decode row through up to five q4tp matrices of the same width in one
// dispatch (`q4tp_matvec_m`'s arithmetic, `q4tp_rows`): job j's matrix at
// absolute byte `abs` of the mapping, `rows` rows, its input x`xs`, its output
// y`ys`, its first threadgroup `tg0` (16 rows a threadgroup, four a simdgroup).
struct MvJob {
    ulong abs;
    uint rows, xs, ys, tg0;
};
kernel void ink_mvq(device const uchar* q0 [[buffer(0)]], device const uchar* q1 [[buffer(1)]],
                    device const uchar* q2 [[buffer(2)]], device const uchar* q3 [[buffer(3)]],
                    device const MvJob* jobs [[buffer(4)]], constant uint& njobs [[buffer(5)]],
                    constant ulong& wstride [[buffer(6)]],
                    device const float* xa [[buffer(7)]], device const float* xb [[buffer(8)]],
                    device const float* xc [[buffer(9)]],
                    device float* ya [[buffer(10)]], device float* yb [[buffer(11)]], device float* yc [[buffer(12)]],
                    device float* yd [[buffer(13)]], device float* ye [[buffer(14)]],
                    constant IP& p [[buffer(15)]],
                    uint tgpos [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float lad[4u * 4u * 32u];
    uint j = 0u;
    for (uint i = 1u; i < njobs; ++i)
        if (jobs[i].tg0 <= tgpos) j = i;
    const MvJob jb = jobs[j];
    device const uchar* w = at_abs(jb.abs, q0, q1, q2, q3, wstride);
    const uint gpr = p.H / 32u;
    const uint r0 = ((tgpos - jb.tg0) * 4u + sg) * 4u;
    const uint nr = r0 < jb.rows ? min(jb.rows - r0, 4u) : 0u;
    ladder(w, jb.rows, gpr, r0, nr, lad + sg * 128u, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (nr == 0u) return;
    device const float* x = jb.xs == 0u ? xa : jb.xs == 1u ? xb : xc;
    float acc[4];
    q4tp_rows(w, jb.rows, gpr, r0, nr, x, lad + sg * 128u, lane, acc);
    if (lane == 0u) {
        device float* y = jb.ys == 0u ? ya : jb.ys == 1u ? yb : jb.ys == 2u ? yc : jb.ys == 3u ? yd : ye;
        for (uint ri = 0u; ri < nr; ++ri) y[r0 + ri] = acc[ri];
    }
}

// The three q, k, v message gates of a decode call in two launches
// (`ink_gate_down` / `ink_gate_out`'s arithmetic a gate, blockIdx z the gate):
// gate j's input at x + xo[j], its message at m + mo[j], its output at
// out + oo[j] (floats), its d at d + j·n·R.
struct G3Off {
    uint xo[3];
    uint mo[3];
    uint oo[3];
};
kernel void ink_gate_down3(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                           device const float* dw0 [[buffer(2)]], device const float* dw1 [[buffer(3)]],
                           device const float* dw2 [[buffer(4)]], device float* d [[buffer(5)]],
                           constant G3Off& go [[buffer(6)]], constant IP& p [[buffer(15)]],
                           uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint3 nt_v [[threads_per_threadgroup]])
{
    const uint nt = nt_v.x;
    const uint k = tg_v.x, r = tg_v.y, j = tg_v.z, H = p.H;
    device const float* xr = x + go.xo[j] + (ulong)r * H;
    device const float* mr = m + go.mo[j] + (ulong)r * H;
    device const float* dw = (j == 0u ? dw0 : j == 1u ? dw1 : dw2) + (ulong)k * 2u * H;
    float sx = 0.0f, sm = 0.0f, ax = 0.0f, am = 0.0f;
    for (uint i = tid; i < H; i += nt) {
        float a = xr[i], b = mr[i];
        sx = fma(a, a, sx);
        sm = fma(b, b, sm);
        ax = fma(dw[i], a, ax);
        am = fma(dw[H + i], b, am);
    }
    threadgroup float4 sh4[32];
    const float4 v4 = float4(simd_sum(sx), simd_sum(sm), simd_sum(ax), simd_sum(am));
    const uint lane = tid & 31u, w = tid >> 5u, nw = (nt + 31u) >> 5u;
    if (lane == 0u) sh4[w] = v4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float4 t = float4(0.0f);
        for (uint q = 0u; q < nw; ++q) t += sh4[q];
        d[((ulong)j * p.n + r) * p.R + k] = t.z * rsqrt(t.x / (float)H + p.eps) + t.w * rsqrt(t.y / (float)H + p.eps);
    }
}

kernel void ink_gate_out3(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                          device const float* d [[buffer(2)]], device const float* F0 [[buffer(3)]],
                          device const float* F1 [[buffer(4)]], device const float* F2 [[buffer(5)]],
                          device const float* G0 [[buffer(6)]], device const float* G1 [[buffer(7)]],
                          device const float* G2 [[buffer(8)]], device float* out [[buffer(9)]],
                          constant G3Off& go [[buffer(10)]], constant IP& p [[buffer(15)]],
                          uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    const uint i = tg_v.x * 8u + sg, r = tg_v.y, j = tg_v.z, H = p.H, R = p.R;
    if (i >= H) return;
    device const float* F = j == 0u ? F0 : j == 1u ? F1 : F2;
    device const float* G = j == 0u ? G0 : j == 1u ? G1 : G2;
    device const float* dr = d + ((ulong)j * p.n + r) * R;
    float f = 0.0f, g = 0.0f;
    for (uint k = lane; k < R; k += 32u) {
        f = fma(F[(ulong)i * R + k], dr[k], f);
        g = fma(G[(ulong)i * R + k], dr[k], g);
    }
    f = simd_sum(f);
    g = simd_sum(g);
    if (lane == 0u) {
        const ulong at = (ulong)r * H + i;
        out[go.oo[j] + at] = (1.0f - tanh(f)) * x[go.xo[j] + at] + tanh(g) * m[go.mo[j] + at];
    }
}

// A prompt call's attention on the simdgroup matrices (`ink_attn`'s
// arithmetic in flash form, the scores never stored): a threadgroup 32 query
// rows of one head, a simdgroup 8 of them; tiles of 32 key positions from the
// block's earliest window start to its last row; S = Q·Kᵀ and O += P·V as 8×8
// half products into float sums, the scale, relative bias (distances under
// E), window and causal mask and the running softmax a row through
// threadgroup memory, the sink at the end. Grid (ceil(n/32), NH), 128 threads.
kernel void ink_attn_fl(device const float* yq [[buffer(0)]], device const float* yr [[buffer(1)]],
                        device const float* sinks [[buffer(2)]], device const float* table [[buffer(3)]],
                        device const half* kc [[buffer(4)]], device const half* vc [[buffer(5)]],
                        device float* out [[buffer(6)]], constant IP& p [[buffer(15)]],
                        uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    // the queries first, then the same bytes hold a key and a value tile
    threadgroup half kv[2 * 32 * 128];
    threadgroup half* qh = kv;
    threadgroup half* ks = kv;
    threadgroup half* vs = kv + 32 * 128;
    threadgroup float ss[4][8 * 32];
    threadgroup half ph[4][8 * 32];
    threadgroup float ds[4][64];
    const uint h = tg_v.y, rb0 = tg_v.x * 32u, HD = p.HD, NKV = p.NKV, NH = p.NH, DR = p.DR, E = p.E;
    const uint C = NKV * HD, g = h / (NH / NKV);
    const uint n = p.n, rlast = min(n, rb0 + 32u) - 1u;
    // the block's queries as half rows (into the tile memory before the loop)
    for (uint i = tid; i < 32u * 128u; i += 128u) {
        const uint rr = i >> 7u, c = i & 127u;
        const uint row = min(rb0 + rr, n - 1u);
        qh[i] = (half)yq[((ulong)row * NH + h) * HD + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 mq[16];
    for (uint kk = 0u; kk < 16u; ++kk) simdgroup_load(mq[kk], qh + (sg * 8u) * 128u + kk * 8u, 128);
    simdgroup_float8x8 mo[16];
    for (uint i = 0u; i < 16u; ++i) mo[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // this lane's row (of the simdgroup's 8) and its quarter of a tile's 32 keys
    const uint lr = lane >> 2u, part = lane & 3u;
    const uint row = rb0 + sg * 8u + lr;
    const bool live = row < n;
    const uint pos = p.pos0 + min(row, n - 1u);
    const uint first = p.win > 0u ? (pos + 1u > p.win ? pos + 1u - p.win : 0u) : 0u;
    float rr[32];
    device const float* yrr = yr + ((ulong)min(row, n - 1u) * NH + h) * DR;
    for (uint j = 0u; j < 32u; ++j) rr[j] = j < DR ? yrr[j] : 0.0f;
    float m = -3.0e38f, l = 0.0f;
    const uint pos_lo = p.pos0 + rb0;
    const uint pfirst = p.win > 0u ? (pos_lo + 1u > p.win ? pos_lo + 1u - p.win : 0u) : 0u;
    const uint pend = p.pos0 + rlast;
    for (uint p0 = pfirst; p0 <= pend; p0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32u * 16u; i += 128u) {
            const uint key = i >> 4u, c8 = i & 15u, pk = p0 + key;
            half4 k0 = half4(0.0h), k1 = half4(0.0h), v0 = half4(0.0h), v1 = half4(0.0h);
            if (pk <= pend) {
                const uint cell = p.ring > 0u ? pk % p.ring : pk;
                device const half4* kr = (device const half4*)(kc + (ulong)cell * C + g * HD + c8 * 8u);
                device const half4* vr = (device const half4*)(vc + (ulong)cell * C + g * HD + c8 * 8u);
                k0 = kr[0]; k1 = kr[1]; v0 = vr[0]; v1 = vr[1];
            }
            threadgroup half4* kd = (threadgroup half4*)(ks + key * 128u + c8 * 8u);
            threadgroup half4* vd = (threadgroup half4*)(vs + key * 128u + c8 * 8u);
            kd[0] = k0; kd[1] = k1; vd[0] = v0; vd[1] = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q·Kᵀ: four 8×8 blocks of keys
        for (uint kb = 0u; kb < 4u; ++kb) {
            simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0u; kk < 16u; ++kk) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ks + (kb * 8u) * 128u + kk * 8u, 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, mq[kk], mk, sacc);
            }
            simdgroup_store(sacc, ss[sg] + kb * 8u, 32);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // the lane's eight scores: scale, bias, mask; the row's running max and sum
        float sv[8], tmax = -3.0e38f;
        for (uint q = 0u; q < 8u; ++q) {
            const uint kcol = part * 8u + q, pk = p0 + kcol;
            float s = ss[sg][lr * 32u + kcol] * p.scale;
            if (live && pk >= first && pk <= pos) {
                const uint d = pos - pk;
                if (d < E) {
                    float b = 0.0f;
                    for (uint j = 0u; j < DR; ++j) b = fma(rr[j], table[(ulong)j * E + d], b);
                    s += b;
                }
            } else {
                s = -3.0e38f;
            }
            sv[q] = s;
            tmax = max(tmax, s);
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1u));
        tmax = max(tmax, simd_shuffle_xor(tmax, 2u));
        const float mn = max(m, tmax);
        const float cf = mn > -1.0e38f ? exp(m - mn) : 1.0f;
        float ts = 0.0f;
        for (uint q = 0u; q < 8u; ++q) {
            const float e = sv[q] > -1.0e38f ? exp(sv[q] - mn) : 0.0f;
            ts += e;
            ph[sg][lr * 32u + part * 8u + q] = (half)e;
        }
        l = l * cf + ts;
        m = mn;
        if (part == 0u) {
            for (uint c = 0u; c < 8u; ++c) ds[sg][lr * 8u + c] = c == lr ? cf : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(c)·O + P·V
        simdgroup_float8x8 md;
        simdgroup_load(md, ds[sg], 8);
        simdgroup_half8x8 mp[4];
        for (uint kb = 0u; kb < 4u; ++kb) simdgroup_load(mp[kb], ph[sg] + kb * 8u, 32);
        for (uint cb = 0u; cb < 16u; ++cb) {
            simdgroup_multiply(mo[cb], md, mo[cb]);
            for (uint kb = 0u; kb < 4u; ++kb) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, vs + (kb * 8u) * 128u + cb * 8u, 128);
                simdgroup_multiply_accumulate(mo[cb], mp[kb], mv, mo[cb]);
            }
        }
    }
    // the row's sum over its four lanes, the sink, the output
    l += simd_shuffle_xor(l, 1u);
    l += simd_shuffle_xor(l, 2u);
    const float sk = sinks[(ulong)min(row, n - 1u) * NH + h];
    const float M = max(m, sk);
    const float f = m > -1.0e38f ? exp(m - M) : 0.0f;
    const float inv = f / (l * f + exp(sk - M));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0u) {
        for (uint c = 0u; c < 8u; ++c) ds[sg][lr * 8u + c] = c == lr ? inv : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 mi;
    simdgroup_load(mi, ds[sg], 8);
    for (uint cb = 0u; cb < 16u; ++cb) {
        simdgroup_multiply(mo[cb], mi, mo[cb]);
        simdgroup_store(mo[cb], ss[sg], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // lanes: row lr, columns part·2 .. part·2 + 1 of the block
        if (live) {
            for (uint c = part * 2u; c < part * 2u + 2u; ++c)
                out[((ulong)row * NH + h) * HD + cb * 8u + c] = ss[sg][lr * 8u + c];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
