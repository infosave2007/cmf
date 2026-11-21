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
                    constant IP& p [[buffer(15)]],
                    uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    threadgroup float sh[32];
    threadgroup float lg[8], rn[8];
    const uint qi = tg.x, r = tg.y, H = p.H, nb = p.nb, ns = nb + 1u;
    device const float* qv = q + (ulong)qi * H;
    for (uint s = 0u; s < ns; ++s) {
        device const float* sv = s < nb ? blk + (ulong)r * p.bstride + (ulong)s * H : x + (ulong)r * H;
        float ss = 0.0f, dt = 0.0f;
        for (uint i = tid; i < H; i += nt) {
            float v = sv[i];
            ss = fma(v, v, ss);
            dt = fma(v, qv[i], dt);
        }
        ss = tg_sum(ss, sh, tid, nt);
        dt = tg_sum(dt, sh, tid, nt);
        if (tid == 0u) {
            float inv = rsqrt(ss / (float)H + 1e-6f);
            rn[s] = inv;
            lg[s] = dt * inv * rsqrt((float)H);
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
    for (uint i = tid; i < H; i += nt) {
        float acc = 0.0f;
        for (uint s = 0u; s < ns; ++s) {
            device const float* sv = s < nb ? blk + (ulong)r * p.bstride + (ulong)s * H : x + (ulong)r * H;
            acc = fma(w[s], sv[i], acc);
        }
        o[i] = acc;
    }
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
    sx = tg_sum(sx, sh, tid, nt);
    sm = tg_sum(sm, sh, tid, nt);
    ax = tg_sum(ax, sh, tid, nt);
    am = tg_sum(am, sh, tid, nt);
    if (tid == 0u) d[(ulong)r * p.R + k] = ax * rsqrt(sx / (float)H + p.eps) + am * rsqrt(sm / (float)H + p.eps);
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
kernel void ink_fmv(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                    device float* y [[buffer(2)]], constant IP& p [[buffer(15)]],
                    uint3 tg_v [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]])
{
    const uint2 tg = tg_v.xy;

    const uint o = tg.x * 8u + sg, r = tg.y, H = p.H;
    if (o >= p.rows) return;
    device const float* w = W + (ulong)o * H;
    device const float* xr = x + (ulong)r * H;
    float acc = 0.0f;
    for (uint i = lane; i < H; i += 32u) acc = fma(w[i], xr[i], acc);
    acc = simd_sum(acc);
    if (lane == 0u) y[(ulong)r * p.rows + o] = acc;
}

inline float tau_of(uint pos, float alpha, float floor_) {
    return 1.0f + alpha * log(max((float)(pos + 1u) / floor_, 1.0f));
}

// Queries and relative states in place: grid (NH, n), HD threads.
kernel void ink_qr(device float* yq [[buffer(0)]], device float* yr [[buffer(1)]],
                   device const float* qs [[buffer(2)]], device const float* rs [[buffer(3)]],
                   constant IP& p [[buffer(15)]],
                   uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    threadgroup float sh[32];
    const uint h = tg.x, r = tg.y, HD = p.HD, DR = p.DR, NH = p.NH;
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

// Keys (y 0) or values (1) of the call's rows into the f16 cache, a KV head
// a threadgroup of HD threads (see `ink_kv` in inkling.cu). `dy` may be the
// zero-length dummy (`full` 0).
kernel void ink_kv(device const float* yk [[buffer(0)]], device const float* yv [[buffer(1)]],
                   device const float* dk [[buffer(2)]], device const float* dv [[buffer(3)]],
                   device const float* tk [[buffer(4)]], device const float* tv [[buffer(5)]],
                   device float* rk [[buffer(6)]], device float* rv [[buffer(7)]],
                   device half* kc [[buffer(8)]], device half* vc [[buffer(9)]],
                   constant IP& p [[buffer(15)]],
                   uint3 tg_v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint3 nt_v [[threads_per_threadgroup]])
{
    const uint2 tg = tg_v.xy;
    const uint nt = nt_v.x;

    threadgroup float sh[32];
    const uint g = tg.x, isv = tg.y, HD = p.HD, NKV = p.NKV, KC = p.KC;
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

// The attention of row r and head h (see `ink_attn` in inkling.cu): grid
// (NH, n), 256 threads, HD 128.
kernel void ink_attn(device const float* yq [[buffer(0)]], device const float* yr [[buffer(1)]],
                     device const float* sinks [[buffer(2)]], device const float* table [[buffer(3)]],
                     device const half* kc [[buffer(4)]], device const half* vc [[buffer(5)]],
                     device float* out [[buffer(6)]], constant IP& p [[buffer(15)]],
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
        out[((ulong)r * NH + h) * HD + i] = a / L;
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
    device const float* w = W + (ulong)e * H;
    device const float* xr = x + (ulong)r * H;
    float acc = 0.0f;
    for (uint i = lane; i < H; i += 32u) acc = fma(w[i], xr[i], acc);
    acc = simd_sum(acc);
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
