// The resident Embryo token on CUDA: `gpu_wgpu/embryo_wide.wgsl`'s stages over
// the same layout (the meta records, the scratch image's planes), the matrices
// in the file's q8_2f (`Q8` 1) or f16 form. A WGSL workgroup is a block, its
// invocation index the thread; a 32-wide stage's global id is
// blockIdx.x * 32 + threadIdx.x.

typedef unsigned int u32;
typedef unsigned short u16;

struct Params {
    u32 position;
    u32 max_seq;
    u32 vocab;
    u32 flags;
    float eps;
    float phase_mass;
    u32 layer;
    u32 chunk;
};

#define HEADER 32u
#define REC 64u
#define S_NORM 0u
#define S_GDN_Z 2048u
#define S_GDN_A 5120u
#define S_GDN_B 5632u
#define S_GDN_O 6144u
#define S_G 5120u
#define S_UP 8192u
#define S_GDN_QKV 9216u
#define S_GDN_CQ 11264u
#define S_SEL 13312u
#define S_PARTS 45056u
#define PART_STRIDE 1024u
#define CHUNK 128u
#define S_ROPE 53248u
#define CP_Q 106496u
#define CP_ATT 368640u
#define S_SC 303104u
#define S_PART 434176u
#define S_ROUTE_REDUCED 32768u
#define S_FOLD 40960u
#define UMAX 0xffffffffu
#define KEY_BLOCK 32u
#define COL_BLOCK 8u
#define WEIGHT_FLOOR (-16.0f)
#define DROPPED (-1e30f)
#define FMAX 3.402823e+38f

// Every stage's buffers, as the WGSL bindings.
#define EW_ARGS                                                                                                     \
    const float* __restrict__ small1, const u32* __restrict__ md, const uint4* __restrict__ head_mat,                \
        float* __restrict__ state1, float* __restrict__ l1, Params p, float* __restrict__ s1,                        \
        const uint4* __restrict__ mats, const float* __restrict__ clusters1, const float* __restrict__ final1,       \
        float* __restrict__ hb1, float* __restrict__ kv1

#define small4 ((const float4*)small1)
#define s4 ((float4*)s1)
#define kv4 ((float4*)kv1)
#define state4 ((float4*)state1)
#define hb4 ((float4*)hb1)
#define clusters4 ((const float4*)clusters1)
#define final4 ((const float4*)final1)

__device__ __forceinline__ u32 rb(const Params& p) { return HEADER + p.layer * REC; }
__device__ __forceinline__ float hsum(float4 v) { return v.x + v.y + v.z + v.w; }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ u32 chunks(u32 cols) { return (cols + CHUNK - 1u) / CHUNK; }
__device__ __forceinline__ float4 f4(float a) { return make_float4(a, a, a, a); }
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 sub4(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 scl4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ float4 max4(float4 a, float4 b) { return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w)); }
__device__ __forceinline__ float comp(float4 v, u32 i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }

__shared__ float wred[256];
__shared__ float winv;
__shared__ float4 wx[256];
__shared__ float4 wo[128];
__shared__ float4 wp[128];
__shared__ float wnq;
__shared__ float wnk;
__shared__ u32 wsel;

// ------------------------------------------------------------ the matrices
#if Q8
// q8_2f: w[o, i] = q[o, i] row[o] col[i]; the stage that makes a matrix's input
// multiplies in the column field once (`fold_*`), so a row reads only its codes.
__device__ __forceinline__ u32 x_in(u32 i) { return S_FOLD + i * 1024u; }
__device__ __forceinline__ float4 fold4(const float* small1, const u32* md, u32 e, u32 cell, float4 v) {
    return mul4(v, small4[(md[e + 2u] >> 2u) + cell]);
}
__device__ __forceinline__ float fold1(const float* small1, const u32* md, u32 e, u32 i, float v) {
    return v * small1[md[e + 2u] + i];
}
__device__ __forceinline__ void fold_lane(const float* small1, const u32* md, float* s1, u32 i, u32 e, u32 lid, float4 v) {
    s4[(x_in(i) >> 2u) + lid] = fold4(small1, md, e, lid, v);
}
__device__ __forceinline__ float4 u4(u32 v) {
    return make_float4((float)(signed char)(v & 0xFFu), (float)(signed char)((v >> 8) & 0xFFu),
                       (float)(signed char)((v >> 16) & 0xFFu), (float)(signed char)(v >> 24));
}
__device__ __forceinline__ float4 q16(uint4 w, const float* s1, u32 xb) {
    return add4(add4(mul4(u4(w.x), s4[xb]), mul4(u4(w.y), s4[xb + 1u])), add4(mul4(u4(w.z), s4[xb + 2u]), mul4(u4(w.w), s4[xb + 3u])));
}
__device__ float mv_part(const float* small1, const u32* md, const uint4* mats, const float* s1, u32 e, u32 row, u32 ch, u32 cols, u32 x) {
    const u32 first = ch * CHUNK;
    const u32 n = min(CHUNK, cols - first) >> 4u;
    const u32 o = (md[e] + row * cols + first) >> 4u;
    const u32 xb = (x + first) >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, q16(mats[o + j], s1, xb + 4u * j));
    return hsum(a) * small1[md[e + 1u] + row];
}
__device__ float mv(const float* small1, const u32* md, const uint4* mats, const float* s1, u32 e, u32 row, u32 cols, u32 x) {
    const u32 n = cols >> 4u;
    const u32 o = (md[e] + row * cols) >> 4u;
    const u32 xb = x >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, q16(mats[o + j], s1, xb + 4u * j));
    return hsum(a) * small1[md[e + 1u] + row];
}
__device__ float2 mv2r(const float* small1, const u32* md, const uint4* mats, const float* s1, u32 ea, u32 ra, u32 eb, u32 rbb, u32 cols, u32 xa, u32 xbb) {
    const u32 n = cols >> 4u;
    const u32 oa = (md[ea] + ra * cols) >> 4u, ob = (md[eb] + rbb * cols) >> 4u;
    const u32 va = xa >> 2u, vb = xbb >> 2u;
    float4 a = f4(0.0f), b = f4(0.0f);
    for (u32 j = 0; j < n; j++) {
        a = add4(a, q16(mats[oa + j], s1, va + 4u * j));
        b = add4(b, q16(mats[ob + j], s1, vb + 4u * j));
    }
    return make_float2(hsum(a) * small1[md[ea + 1u] + ra], hsum(b) * small1[md[eb + 1u] + rbb]);
}
__device__ float mv_head(const float* small1, const u32* md, const uint4* head_mat, const float* s1, u32 row) {
    const u32 e = md[30u], hidden = md[0u];
    const u32 n = hidden >> 4u;
    const u32 o = (row * hidden) >> 4u;
    const u32 xb = x_in(0u) >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, q16(head_mat[o + j], s1, xb + 4u * j));
    return hsum(a) * small1[md[e + 1u] + row];
}
#else
// f16: a matrix word is its offset in values (a multiple of 8); the vector as
// its stage left it.
__device__ __forceinline__ u32 x_in(u32) { return S_NORM; }
__device__ __forceinline__ float4 fold4(const float*, const u32*, u32, u32, float4 v) { return v; }
__device__ __forceinline__ float fold1(const float*, const u32*, u32, u32, float v) { return v; }
__device__ __forceinline__ void fold_lane(const float*, const u32*, float*, u32, u32, u32, float4) {}
__device__ __forceinline__ float h16(u32 w, int hi) {
    const u16 b = hi ? (u16)(w >> 16) : (u16)(w & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}
__device__ __forceinline__ float4 d8(uint4 q, float4 v0, float4 v1) {
    return add4(mul4(make_float4(h16(q.x, 0), h16(q.x, 1), h16(q.y, 0), h16(q.y, 1)), v0),
                mul4(make_float4(h16(q.z, 0), h16(q.z, 1), h16(q.w, 0), h16(q.w, 1)), v1));
}
__device__ float mv_part(const float*, const u32* md, const uint4* mats, const float* s1, u32 e, u32 row, u32 ch, u32 cols, u32 x) {
    const u32 first = ch * CHUNK;
    const u32 n = min(CHUNK, cols - first) >> 3u;
    const u32 o = (e + row * cols + first) >> 3u;
    const u32 xb = (x + first) >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, d8(mats[o + j], s4[xb + 2u * j], s4[xb + 2u * j + 1u]));
    return hsum(a);
}
__device__ float mv(const float*, const u32* md, const uint4* mats, const float* s1, u32 e, u32 row, u32 cols, u32 x) {
    const u32 n = cols >> 3u;
    const u32 o = (e + row * cols) >> 3u;
    const u32 xb = x >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, d8(mats[o + j], s4[xb + 2u * j], s4[xb + 2u * j + 1u]));
    return hsum(a);
}
__device__ float2 mv2r(const float*, const u32* md, const uint4* mats, const float* s1, u32 ea, u32 ra, u32 eb, u32 rbb, u32 cols, u32 xa, u32 xbb) {
    const u32 n = cols >> 3u;
    const u32 oa = (ea + ra * cols) >> 3u, ob = (eb + rbb * cols) >> 3u;
    const u32 va = xa >> 2u, vb = xbb >> 2u;
    float4 a = f4(0.0f), b = f4(0.0f);
    for (u32 j = 0; j < n; j++) {
        a = add4(a, d8(mats[oa + j], s4[va + 2u * j], s4[va + 2u * j + 1u]));
        b = add4(b, d8(mats[ob + j], s4[vb + 2u * j], s4[vb + 2u * j + 1u]));
    }
    return make_float2(hsum(a), hsum(b));
}
__device__ float mv_head(const float*, const u32* md, const uint4* head_mat, const float* s1, u32 row) {
    const u32 hidden = md[0u];
    const u32 n = hidden >> 3u;
    const u32 o = (row * hidden) >> 3u;
    const u32 xb = x_in(0u) >> 2u;
    float4 a = f4(0.0f);
    for (u32 j = 0; j < n; j++) a = add4(a, d8(head_mat[o + j], s4[xb + 2u * j], s4[xb + 2u * j + 1u]));
    return hsum(a);
}
#endif
#define MV(e, row, cols, x) mv(small1, md, mats, s1, e, row, cols, x)
#define MV2(ea, eb, row, cols, xa, xb) mv2r(small1, md, mats, s1, ea, row, eb, row, cols, xa, xb)
#define MV2R(ea, ra, eb, rbb, cols, xa, xb) mv2r(small1, md, mats, s1, ea, ra, eb, rbb, cols, xa, xb)
#define MV_PART(e, row, ch, cols, x) mv_part(small1, md, mats, s1, e, row, ch, cols, x)
#define FOLD_LANE(i, e, lid, v) fold_lane(small1, md, s1, i, e, lid, v)

// --------------------------------------------------------------- the norms
// RMSNorm into the scratch plane: a lane per four values.
__device__ float4 norm_lanes(EW_ARGS, u32 lid, bool from_final, u32 wbase, u32 parts) {
    const u32 hidden = md[0u];
    const u32 n = hidden >> 2u;
    float4 v = f4(0.0f);
    if (lid < n) {
        v = hb4[lid];
        for (u32 c = 0; c < parts; c++) v = add4(v, s4[((S_PARTS + c * PART_STRIDE) >> 2u) + lid]);
        hb4[lid] = v;
        wred[lid] = dot4(v, v);
    }
    __syncthreads();
    if (lid == 0) {
        float t = 0.0f;
        for (u32 j = 0; j < n; j++) t += wred[j];
        winv = rsqrtf(t / (float)hidden + p.eps);
    }
    __syncthreads();
    if (lid < n) {
        float4 w = from_final ? final4[lid] : small4[(wbase >> 2u) + lid];
        if (p.flags & 1u) w = add4(f4(1.0f), w);
        v = scl4(mul4(v, w), winv);
        s4[lid] = v;
    }
    return v;
}
#define NORM_ARGS small1, md, head_mat, state1, l1, p, s1, mats, clusters1, final1, hb1, kv1

extern "C" __global__ void __launch_bounds__(256) ew_norm(EW_ARGS) {
    const u32 lid = threadIdx.x, r = rb(p);
    // the first layer starts from the token embedding, the others from an FFN
    const float4 v = norm_lanes(NORM_ARGS, lid, false, md[r + 1u], p.layer == 0u ? 0u : chunks(md[1u]));
    if (lid >= (md[0u] >> 2u)) return;
    if (md[r] == 4u) {
        FOLD_LANE(0u, md[r + 56u], lid, v);
        FOLD_LANE(1u, md[r + 57u], lid, v);
        FOLD_LANE(2u, md[r + 58u], lid, v);
        FOLD_LANE(3u, md[r + 59u], lid, v);
    } else {
        FOLD_LANE(0u, md[r + 11u], lid, v);
        FOLD_LANE(1u, md[r + 12u], lid, v);
        FOLD_LANE(2u, md[r + 13u], lid, v);
    }
}

extern "C" __global__ void __launch_bounds__(256) ew_post_norm(EW_ARGS) {
    const u32 lid = threadIdx.x, r = rb(p);
    const u32 cols = md[r] == 4u ? md[24u] * md[27u] : md[7u] * md[9u];
    const float4 v = norm_lanes(NORM_ARGS, lid, false, md[r + 2u], chunks(cols));
    if (lid >= (md[0u] >> 2u)) return;
    FOLD_LANE(0u, md[r + 21u], lid, v);
    FOLD_LANE(1u, md[r + 22u], lid, v);
}

// The final norm and the cluster scores, as log probabilities, for the head's gate.
extern "C" __global__ void __launch_bounds__(256) ew_final_norm(EW_ARGS) {
    const u32 lid = threadIdx.x;
    const float4 v = norm_lanes(NORM_ARGS, lid, true, 0u, chunks(md[1u]));
    const u32 n = md[0u] >> 2u;
    if (lid < n) {
        wx[lid] = v;
        FOLD_LANE(0u, md[30u], lid, v);
    }
    __syncthreads();
    const u32 nc = md[11u];
    if (lid < nc) {
        float4 z = f4(0.0f);
        for (u32 j = 0; j < n; j++) z = add4(z, mul4(clusters4[lid * n + j], wx[j]));
        wred[lid] = hsum(z);
    }
    __syncthreads();
    if (lid == 0) {
        float mx = -FMAX;
        for (u32 j = 0; j < nc; j++) mx = fmaxf(mx, wred[j]);
        float se = 0.0f;
        for (u32 j = 0; j < nc; j++) se += expf(wred[j] - mx);
        winv = mx + logf(se);
    }
    __syncthreads();
    if (lid < nc) l1[p.vocab + lid] = wred[lid] - winv;
}

// ------------------------------------------------------------------- GDN
// The projections of the normed token; q/k/v through the depthwise causal conv
// and the ring shift in the same thread.
extern "C" __global__ void __launch_bounds__(32) ew_gdn_project(EW_ARGS) {
    const u32 r = rb(p), hidden = md[0u], nv = md[24u], vd = nv * md[27u], cdim = md[29u];
    u32 id = blockIdx.x * 32u + threadIdx.x;
    if (id < cdim) {
        const float x = MV(md[r + 56u], id, hidden, x_in(0u));
        const u32 kk = md[28u], ring = md[r + 25u], taps = md[r + 60u] + id * kk;
        float acc = x * small1[taps + kk - 1u];
        for (u32 j = 0; j + 1u < kk; j++) acc += state1[ring + j * cdim + id] * small1[taps + j];
        s1[S_GDN_CQ + id] = acc / (1.0f + expf(-acc));
        for (u32 j = 0; j + 2u < kk; j++) state1[ring + j * cdim + id] = state1[ring + (j + 1u) * cdim + id];
        if (kk > 1u) state1[ring + (kk - 2u) * cdim + id] = x;
        return;
    }
    id -= cdim;
    if (id < vd) {
        s1[S_GDN_Z + id] = MV(md[r + 57u], id, hidden, x_in(1u));
        return;
    }
    id -= vd;
    if (id < nv) {
        const float2 ab = MV2(md[r + 58u], md[r + 59u], id, hidden, x_in(2u), x_in(3u));
        s1[S_GDN_A + id] = ab.x;
        s1[S_GDN_B + id] = ab.y;
    }
}

extern "C" __global__ void __launch_bounds__(32) ew_gdn_out(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 vd = md[24u] * md[27u];
    const u32 k = chunks(vd), id = g / k, ch = g % k;
    if (id >= md[0u]) return;
    s1[S_PARTS + ch * PART_STRIDE + id] = MV_PART(md[rb(p) + 29u], id, ch, vd, S_GDN_O);
}

__device__ __forceinline__ float gdn_softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// The delta-rule state step and the gated RMSNorm of one head (a block of 128).
extern "C" __global__ void __launch_bounds__(128) ew_gdn_step(EW_ARGS) {
    const u32 lid = threadIdx.x, h = blockIdx.x;
    const u32 nv = md[24u];
    if (h >= nv) return;
    const u32 nk = md[25u], dk = md[26u], dv = md[27u], kk = md[28u], cdim = md[29u];
    const u32 dv4 = dv >> 2u, dk4 = dk >> 2u, r = rb(p);
    const u32 ko = h / (nv / nk), kd = nk * dk;
    const u32 qb = (S_GDN_CQ + ko * dk) >> 2u, kb = (S_GDN_CQ + kd + ko * dk) >> 2u;
    float2 a = make_float2(0.0f, 0.0f);
    if (lid < dk4) {
        const float4 q = s4[qb + lid], k = s4[kb + lid];
        a = make_float2(dot4(q, q), dot4(k, k));
    }
    wo[lid] = make_float4(a.x, a.y, 0.0f, 0.0f);
    __syncthreads();
    if (lid == 0) {
        float tx = 0.0f, ty = 0.0f;
        for (u32 j = 0; j < dk4; j++) {
            tx += wo[j].x;
            ty += wo[j].y;
        }
        wnq = tx;
        wnk = ty;
    }
    __syncthreads();
    const u32 blocks = 128u / dv4, col = lid % dv4, blk = lid / dv4;
    const bool live = blk < blocks;
    const u32 first = (blk * dk4) / blocks, end = ((blk + 1u) * dk4) / blocks;
    const u32 base = ((md[r + 25u] + (kk - 1u) * cdim + h * dk * dv) >> 2u) + col * dk;
    float4 kv = f4(0.0f);
    if (live) {
        for (u32 c4 = first; c4 < end; c4++) {
            const float4 kf = s4[kb + c4];
            const u32 b4 = base + 4u * c4;
            kv = add4(kv, add4(add4(scl4(state4[b4], kf.x), scl4(state4[b4 + 1u], kf.y)), add4(scl4(state4[b4 + 2u], kf.z), scl4(state4[b4 + 3u], kf.w))));
        }
    }
    wo[lid] = kv;
    __syncthreads();
    const float invq = 1.0f / (sqrtf(wnq + 1e-6f) * sqrtf((float)dk));
    const float invk = 1.0f / sqrtf(wnk + 1e-6f);
    float4 o = f4(0.0f);
    if (live) {
        kv = f4(0.0f);
        for (u32 b = 0; b < blocks; b++) kv = add4(kv, wo[col + b * dv4]);
        const float g = expf(-expf(small1[md[r + 61u] + h]) * gdn_softplus(s1[S_GDN_A + h] + small1[md[r + 62u] + h]));
        const float beta = 1.0f / (1.0f + expf(-s1[S_GDN_B + h]));
        const float4 vt = s4[((S_GDN_CQ + 2u * kd + h * dv) >> 2u) + col];
        const float4 delta = scl4(sub4(vt, scl4(kv, g * invk)), beta * invk);
        for (u32 c4 = first; c4 < end; c4++) {
            const float4 kf = s4[kb + c4], qf = s4[qb + c4];
            const u32 b4 = base + 4u * c4;
            const float4 s0 = add4(scl4(state4[b4], g), scl4(delta, kf.x));
            const float4 sa = add4(scl4(state4[b4 + 1u], g), scl4(delta, kf.y));
            const float4 s2 = add4(scl4(state4[b4 + 2u], g), scl4(delta, kf.z));
            const float4 s3 = add4(scl4(state4[b4 + 3u], g), scl4(delta, kf.w));
            state4[b4] = s0;
            state4[b4 + 1u] = sa;
            state4[b4 + 2u] = s2;
            state4[b4 + 3u] = s3;
            o = add4(o, add4(add4(scl4(s0, qf.x), scl4(sa, qf.y)), add4(scl4(s2, qf.z), scl4(s3, qf.w))));
        }
    }
    wp[lid] = o;
    __syncthreads();
    if (lid < dv4) {
        o = f4(0.0f);
        for (u32 b = 0; b < blocks; b++) o = add4(o, wp[col + b * dv4]);
        o = scl4(o, invq);
        wo[lid] = o;
        wred[lid] = dot4(o, o);
    }
    __syncthreads();
    if (lid == 0) {
        float t = 0.0f;
        for (u32 j = 0; j < dv4; j++) t += wred[j];
        winv = 1.0f / sqrtf(t / (float)dv + p.eps);
    }
    __syncthreads();
    if (lid < dv4) {
        const u32 cell = ((h * dv) >> 2u) + lid;
        const float4 z = s4[(S_GDN_Z >> 2u) + cell];
        const float4 gate = make_float4(silu(z.x), silu(z.y), silu(z.z), silu(z.w));
        s4[(S_GDN_O >> 2u) + cell] = fold4(small1, md, md[r + 29u], cell, mul4(mul4(scl4(wo[lid], winv), small4[(md[r + 63u] >> 2u) + lid]), gate));
    }
}

// ------------------------------------------------------------ the anchor
__device__ __forceinline__ u32 group_window(const float* small1, const u32* md, u32 kh) {
    if (md[22u] != UMAX) return (u32)small1[md[22u] + kh];
    return md[18u];
}
__device__ __forceinline__ u32 score_stride(const u32* md) { return (md[18u] + md[20u] + 3u) & ~3u; }
__device__ __forceinline__ u32 ring_slot(u32 pm, u32 d, u32 w) { return pm >= d ? pm - d : pm + w - d; }

// The rotation goes on the absolute position (the host writes its angles per
// token), so the ring holds rotated keys and a score is a plain dot product.
extern "C" __global__ void __launch_bounds__(32) ew_anchor_project(EW_ARGS) {
    const u32 r = rb(p), hidden = md[0u], hd = md[9u];
    const u32 qrows = md[7u] * hd, krows = md[8u] * hd, half = md[16u] >> 1u;
    u32 id = blockIdx.x * 32u + threadIdx.x;
    if (id < qrows) {
        const u32 rel = id % hd;
        if (rel < half) {
            const float2 q = MV2R(md[r + 11u], id, md[r + 11u], id + half, hidden, x_in(0u), x_in(0u));
            const float c = s1[S_ROPE + rel], sn = s1[S_ROPE + half + rel];
            s1[CP_Q + id] = q.x * c - q.y * sn;
            s1[CP_Q + id + half] = q.x * sn + q.y * c;
        } else if (rel >= 2u * half) {
            s1[CP_Q + id] = MV(md[r + 11u], id, hidden, x_in(0u));
        }
        return;
    }
    id -= qrows;
    const u32 w = md[18u];
    if (id < krows) {
        const u32 rel = id % hd;
        const u32 o = md[r + 26u] + ((id / hd) * w + p.position % w) * hd + rel;
        if (rel < half) {
            const float2 k = MV2R(md[r + 12u], id, md[r + 12u], id + half, hidden, x_in(1u), x_in(1u));
            const float c = s1[S_ROPE + rel], sn = s1[S_ROPE + half + rel];
            kv1[o] = k.x * c - k.y * sn;
            kv1[o + half] = k.x * sn + k.y * c;
        } else if (rel >= 2u * half) {
            kv1[o] = MV(md[r + 12u], id, hidden, x_in(1u));
        }
        return;
    }
    id -= krows;
    if (id >= krows) return;
    const u32 o = md[r + 26u] + ((id / hd) * w + p.position % w) * hd + id % hd;
    kv1[o + md[8u] * w * hd] = MV(md[r + 13u], id, hidden, x_in(2u));
}

__device__ float attend_score_one(EW_ARGS, u32 hq, u32 kh, u32 u) {
    const u32 hd = md[9u], w = md[18u], sink = md[20u], r = rb(p), n = hd >> 2u;
    const u32 qb = (CP_Q + hq * hd) >> 2u;
    float4 dv = f4(0.0f);
    if (u < sink) {
        // a sink key has no position: the query goes back to its plain form
        const u32 k = (md[r + 27u] + (kh * sink + u) * hd) >> 2u;
        const u32 half = md[16u] >> 1u, h4 = half >> 2u, co = S_ROPE >> 2u;
        for (u32 i = 0; i < h4; i++) {
            const float4 q0 = s4[qb + i], q1 = s4[qb + h4 + i], c = s4[co + i], sn = s4[co + h4 + i];
            dv = add4(dv, add4(mul4(add4(mul4(q0, c), mul4(q1, sn)), small4[k + i]), mul4(sub4(mul4(q1, c), mul4(q0, sn)), small4[k + h4 + i])));
        }
        for (u32 j = 2u * h4; j < n; j++) dv = add4(dv, mul4(s4[qb + j], small4[k + j]));
    } else {
        const u32 ko = (md[r + 26u] + (kh * w + ring_slot(p.position % w, u - sink, w)) * hd) >> 2u;
        for (u32 j = 0; j < n; j++) dv = add4(dv, mul4(s4[qb + j], kv4[ko + j]));
    }
    return hsum(dv) * rsqrtf((float)hd);
}

extern "C" __global__ void __launch_bounds__(32) ew_attend_score(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 qheads = md[7u], sink = md[20u], w = md[18u];
    const u32 per = (sink + min(p.position + 1u, w) + 3u) >> 2u;
    const u32 hq = g / per, u4i = g % per;
    if (hq >= qheads) return;
    const u32 kh = hq / (qheads / md[8u]);
    const u32 total = sink + min(p.position + 1u, group_window(small1, md, kh));
    if (u4i * 4u >= total) return;
    const u32 cell = ((S_SC + 256u + hq * score_stride(md)) >> 2u) + u4i;
    const u32 first = u4i * 4u;
    // a cell past the head's last key stays below every score
    if (first < sink || first + 4u > total) {
        float out[4] = {-FMAX, -FMAX, -FMAX, -FMAX};
        for (u32 c = 0; c < 4u; c++) {
            const u32 u = first + c;
            if (u < total) out[c] = attend_score_one(NORM_ARGS, hq, kh, u);
        }
        s4[cell] = make_float4(out[0], out[1], out[2], out[3]);
        return;
    }
    // four ring rows against one read of the query
    const u32 hd = md[9u], n = hd >> 2u, qb = (CP_Q + hq * hd) >> 2u;
    const u32 ring = md[rb(p) + 26u] + kh * w * hd, pm = p.position % w, d = first - sink;
    const u32 k0 = (ring + ring_slot(pm, d, w) * hd) >> 2u, k1 = (ring + ring_slot(pm, d + 1u, w) * hd) >> 2u;
    const u32 k2 = (ring + ring_slot(pm, d + 2u, w) * hd) >> 2u, k3 = (ring + ring_slot(pm, d + 3u, w) * hd) >> 2u;
    float4 z0 = f4(0.0f), z1 = f4(0.0f), z2 = f4(0.0f), z3 = f4(0.0f);
    for (u32 j = 0; j < n; j++) {
        const float4 q = s4[qb + j];
        z0 = add4(z0, mul4(q, kv4[k0 + j]));
        z1 = add4(z1, mul4(q, kv4[k1 + j]));
        z2 = add4(z2, mul4(q, kv4[k2 + j]));
        z3 = add4(z3, mul4(q, kv4[k3 + j]));
    }
    s4[cell] = scl4(make_float4(hsum(z0), hsum(z1), hsum(z2), hsum(z3)), rsqrtf((float)hd));
}

// The softmax of one head; a key below e^-16 of the largest leaves the list.
extern "C" __global__ void ew_attend_weights(EW_ARGS) {
    const u32 hq = blockIdx.x;
    const u32 qheads = md[7u];
    if (hq >= qheads || threadIdx.x != 0) return;
    const u32 kh = hq / (qheads / md[8u]);
    const u32 total = md[20u] + min(p.position + 1u, group_window(small1, md, kh));
    const u32 stride = score_stride(md);
    const u32 b4 = (S_SC + 256u + hq * stride) >> 2u;
    const u32 woo = S_SC + 256u + (qheads + hq) * stride, uo = S_SC + 256u + (2u * qheads + hq) * stride;
    const u32 n4 = (total + 3u) >> 2u;
    float4 mx4 = f4(-FMAX);
    for (u32 u = 0; u < n4; u++) mx4 = max4(mx4, s4[b4 + u]);
    const float mx = fmaxf(fmaxf(mx4.x, mx4.y), fmaxf(mx4.z, mx4.w));
    float l = 0.0f;
    u32 count = 0;
    for (u32 u = 0; u < n4; u++) {
        const float4 d = sub4(s4[b4 + u], f4(mx));
        for (u32 c = 0; c < 4u; c++) {
            const float dc = comp(d, c);
            if (dc >= WEIGHT_FLOOR) {
                const float v = expf(dc);
                l += v;
                s1[woo + count] = v;
                s1[uo + count] = (float)(u * 4u + c);
                count++;
            }
        }
    }
    s4[(S_SC >> 2u) + hq] = make_float4(1.0f / fmaxf(l, 1e-20f), (float)count, 0.0f, 0.0f);
}

__device__ __forceinline__ u32 part_at(const u32* md, u32 hq, u32 b) {
    return (S_PART + (hq * ((md[18u] + md[20u] + KEY_BLOCK - 1u) / KEY_BLOCK) + b) * md[9u]) >> 2u;
}

// The value sum by blocks: 32 keys of the list over 8 cells of a row a thread.
extern "C" __global__ void __launch_bounds__(32) ew_attend_part(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 qheads = md[7u], hd = md[9u], w = md[18u], sink = md[20u], n = hd >> 2u;
    const u32 cols = (n + COL_BLOCK - 1u) / COL_BLOCK;
    const u32 blocks = (sink + min(p.position + 1u, w) + KEY_BLOCK - 1u) / KEY_BLOCK;
    const u32 hq = g / (blocks * cols);
    if (hq >= qheads) return;
    const u32 rest = g % (blocks * cols), b = rest / cols, c = rest % cols;
    const float4 info = s4[(S_SC >> 2u) + hq];
    const u32 count = (u32)info.y, first = b * KEY_BLOCK;
    if (first >= count) return;
    const u32 r = rb(p), kh = hq / (qheads / md[8u]), stride = score_stride(md);
    const u32 woo = S_SC + 256u + (qheads + hq) * stride, uo = S_SC + 256u + (2u * qheads + hq) * stride;
    const u32 vb = md[r + 26u] + md[8u] * w * hd + kh * w * hd, pm = p.position % w;
    const u32 c0 = c * COL_BLOCK, m = min(COL_BLOCK, n - c0);
    float4 acc[8];
    for (u32 j = 0; j < 8u; j++) acc[j] = f4(0.0f);
    for (u32 i = first; i < min(first + KEY_BLOCK, count); i++) {
        const float t = s1[woo + i];
        const u32 u = (u32)s1[uo + i];
        if (u < sink) {
            const u32 row = ((md[r + 28u] + (kh * sink + u) * hd) >> 2u) + c0;
            for (u32 j = 0; j < m; j++) acc[j] = add4(acc[j], scl4(small4[row + j], t));
        } else {
            const u32 row = ((vb + ring_slot(pm, u - sink, w) * hd) >> 2u) + c0;
            for (u32 j = 0; j < m; j++) acc[j] = add4(acc[j], scl4(kv4[row + j], t));
        }
    }
    const u32 o = part_at(md, hq, b) + c0;
    for (u32 j = 0; j < m; j++) s4[o + j] = acc[j];
}

extern "C" __global__ void __launch_bounds__(32) ew_attend_sum(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 qheads = md[7u], hd = md[9u], n = hd >> 2u, hq = g / n, d4 = g % n;
    if (hq >= qheads) return;
    const float4 info = s4[(S_SC >> 2u) + hq];
    const u32 blocks = ((u32)info.y + KEY_BLOCK - 1u) / KEY_BLOCK;
    float4 acc = f4(0.0f);
    for (u32 b = 0; b < blocks; b++) acc = add4(acc, s4[part_at(md, hq, b) + d4]);
    s4[((CP_ATT + hq * hd) >> 2u) + d4] = fold4(small1, md, md[rb(p) + 14u], ((hq * hd) >> 2u) + d4, scl4(acc, info.x));
}

extern "C" __global__ void __launch_bounds__(32) ew_anchor_out(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 cols = md[7u] * md[9u], k = chunks(cols), id = g / k, ch = g % k;
    if (id >= md[0u]) return;
    s1[S_PARTS + ch * PART_STRIDE + id] = MV_PART(md[rb(p) + 14u], id, ch, cols, CP_ATT);
}

// --------------------------------------------------------- the router, FFN
// The resonance router's reconstruction error, a thread per (expert, row, chunk).
extern "C" __global__ void __launch_bounds__(32) ew_route_rows(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 r = rb(p), hidden = md[0u], rank = md[r + 20u], comps = rank + 1u, k = chunks(hidden);
    const u32 ec = g / k, ch = g % k;
    if (ec >= md[r + 16u] * comps) return;
    const u32 expert = ec / comps, component = ec % comps;
    const u32 first = ch * (CHUNK >> 2u), end = min(first + (CHUNK >> 2u), hidden >> 2u);
    const u32 mu = (md[r + 17u] + expert * hidden) >> 2u;
    float4 z = f4(0.0f);
    if (component == 0u) {
        for (u32 j = first; j < end; j++) {
            const float4 d = sub4(s4[j], small4[mu + j]);
            z = add4(z, mul4(d, d));
        }
    } else {
        const u32 u = (md[r + 18u] + (expert * rank + component - 1u) * hidden) >> 2u;
        for (u32 j = first; j < end; j++) z = add4(z, mul4(sub4(s4[j], small4[mu + j]), small4[u + j]));
    }
    s1[S_ROUTE_REDUCED + g] = hsum(z);
}

// Score and pick in one block (the lower id wins an exact tie), a growth
// shell's mask, then the chosen expert's gate and up inputs.
extern "C" __global__ void __launch_bounds__(256) ew_route_final(EW_ARGS) {
    const u32 lid = threadIdx.x, r = rb(p), ne = md[r + 16u], rank = md[r + 20u];
    if (lid < ne) {
        const u32 k = chunks(md[0u]);
        const u32 base = S_ROUTE_REDUCED + lid * (rank + 1u) * k;
        float d2 = 0.0f;
        for (u32 ch = 0; ch < k; ch++) d2 += s1[base + ch];
        float pn = 0.0f;
        for (u32 c = 1; c <= rank; c++) {
            float v = 0.0f;
            for (u32 ch = 0; ch < k; ch++) v += s1[base + c * k + ch];
            pn += v * v;
        }
        const float err = d2 - pn;
        float score = small1[md[r + 19u] + lid] - err;
        const u32 so = md[r + 30u];
        if (so != UMAX && err > small1[so + lid]) score = small1[so + ne];
        wred[lid] = score;
    }
    __syncthreads();
    if (lid == 0) {
        u32 best = 0;
        float top = wred[0];
        for (u32 e = 1; e < ne; e++) {
            if (wred[e] > top) {
                top = wred[e];
                best = e;
            }
        }
        wsel = best;
        s1[S_SEL] = (float)best;
    }
    __syncthreads();
    if (lid < (md[0u] >> 2u)) {
        const u32 sel = wsel;
        const float4 v = s4[lid];
        FOLD_LANE(2u, md[r + 32u + sel * 3u], lid, v);
        FOLD_LANE(3u, md[r + 33u + sel * 3u], lid, v);
    }
}

// Gate and up of the selected expert and of the shared FFN, with the activation.
extern "C" __global__ void __launch_bounds__(32) ew_ffn_up(EW_ARGS) {
    const u32 r = rb(p), hidden = md[0u], inter = md[1u];
    u32 id = blockIdx.x * 32u + threadIdx.x;
    if (md[r + 15u] == 1u) {
        if (id < inter) {
            const u32 sel = (u32)s1[S_SEL];
            const float2 gu = MV2(md[r + 32u + sel * 3u], md[r + 33u + sel * 3u], id, hidden, x_in(2u), x_in(3u));
            s1[S_G + id] = fold1(small1, md, md[r + 34u + sel * 3u], id, silu(gu.x) * gu.y);
            return;
        }
        id -= inter;
    }
    if (id < inter) {
        const float2 gu = MV2(md[r + 21u], md[r + 22u], id, hidden, x_in(0u), x_in(1u));
        s1[S_UP + id] = fold1(small1, md, md[r + 23u], id, silu(gu.x) * gu.y);
    }
}

// Down of both.
extern "C" __global__ void __launch_bounds__(32) ew_ffn_down(EW_ARGS) {
    const u32 g = blockIdx.x * 32u + threadIdx.x;
    const u32 inter = md[1u], k = chunks(inter), id = g / k, ch = g % k;
    if (id >= md[0u]) return;
    const u32 r = rb(p);
    float y = MV_PART(md[r + 23u], id, ch, inter, S_UP);
    if (md[r + 15u] == 1u) {
        const u32 sel = (u32)s1[S_SEL];
        y += MV_PART(md[r + 34u + sel * 3u], id, ch, inter, S_G);
    }
    s1[S_PARTS + ch * PART_STRIDE + id] = y;
}

// The hierarchical head: a block a cluster; a cluster below the gate reads no rows.
extern "C" __global__ void __launch_bounds__(256) ew_head(EW_ARGS) {
    const u32 lid = threadIdx.x, cl = blockIdx.x, cs = md[12u];
    if (cl >= md[11u]) return;
    const u32 start = cl * cs;
    const float g = l1[p.vocab + cl];
    if (g < p.phase_mass) {
        if (lid < cs) l1[start + lid] = DROPPED;
        return;
    }
    if (lid < cs) wred[lid] = mv_head(small1, md, head_mat, s1, start + lid);
    __syncthreads();
    if (lid == 0) {
        float m = -FMAX;
        for (u32 j = 0; j < cs; j++) m = fmaxf(m, wred[j]);
        float se = 0.0f;
        for (u32 j = 0; j < cs; j++) se += expf(wred[j] - m);
        winv = g - (m + logf(se));
    }
    __syncthreads();
    if (lid < cs) l1[start + lid] = wred[lid] + winv;
}
