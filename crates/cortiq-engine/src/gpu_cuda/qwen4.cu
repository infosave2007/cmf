// Qwen3.8-Flash-Next (qwen4_exp) on CUDA: the kernels between the cuBLAS
// GEMMs of a whole chunk of tokens. Every buffer is [token][...] row-major;
// a GEMM operand is "wt": f16 when `half` is set (tensor cores), f32 otherwise.
// Compiled by NVRTC as its own module, so it carries its own helpers.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
// The sparse-index top-k by radix select (q4_idx_topk; 0 = the one-bit search).
#ifndef TOPK_RADIX
#define TOPK_RADIX 1
#endif

// Programmatic dependent launch (CMF_QWEN_PDL=1): a kernel may start while its predecessor drains. It first waits for the
// predecessor to finish and flush (griddepcontrol.wait), then lets its own successor launch. Without the attribute on
// the launch both are no-ops.
#if __CUDA_ARCH__ >= 900
#define PDL_PRO() do { asm volatile("griddepcontrol.wait;" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;"); } while (0)
#else
#define PDL_PRO() do { } while (0)
#endif

// F64OPS=0 (the fast precision): the norms, gates and the conv of the kernels below in f32. A GeForce part runs f64
// at 1/64 of the f32 rate and these small kernels waited on chains of f64 exp/log/sqrt; the exact precision keeps
// f64 (the host's arithmetic).
#ifndef F64OPS
#define F64OPS 1
#endif
#if F64OPS
typedef double acc64;
#define EXP64(x) exp(x)
#define LOG1P64(x) log1p(x)
#define SQRT64(x) sqrt(x)
#define D64(x) (x)
#else
typedef float acc64;
#define EXP64(x) expf(x)
#define LOG1P64(x) log1pf(x)
#define SQRT64(x) sqrtf(x)
#define D64(x) ((float)(x))
#endif
__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ u16 f2h_sat(float f) { return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f)); }
__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
// Store / load one element of a wt buffer.
__device__ __forceinline__ void stf(void* p, u64 i, float v, int half) {
    if (half) ((u16*)p)[i] = f2h_sat(v); else ((float*)p)[i] = v;
}
__device__ __forceinline__ float ldf(const void* p, u64 i, int half) {
    return half ? h2f(((const u16*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x * sigm(x); }

// hyper [n][hc][hid] = the embedding row repeated over the streams.
extern "C" __global__ void q4_tile(const float* __restrict__ emb, float* __restrict__ hyper, int n, int hid, int hc)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    const float v = emb[i];
    for (int s = 0; s < hc; s++) hyper[(t * hc + s) * hid + d] = v;
}

// Zero-centred group RMS norm, a block per (token, group):
// y = x * rsqrt(mean(x^2) + eps) * (1 + w). f32 into y and/or wt into yh
// (either may be null).
extern "C" __global__ void q4_grms(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                   void* __restrict__ yh, int group, int groups, double eps, int half)
{ PDL_PRO();
    __shared__ acc64 red[256];
    const u64 b = blockIdx.x;
    const int g = (int)(b % groups);
    const float* xr = x + b * group;
    acc64 ss = D64(0.0);
    for (int i = threadIdx.x; i < group; i += blockDim.x) { const acc64 v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = (float)(D64(1.0) / SQRT64(red[0] / (acc64)group + ((acc64)eps)));
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = xr[i] * inv * (1.0f + w[g * group + i]);
        if (y) y[b * group + i] = v;
        if (yh) stf(yh, b * group + i, v, half);
    }
}

// q4_grms after the previous block's inject, in one launch: row (t, s) of hyper first takes
// hyper += iw[t][s] * block[t], is stored back, and is then normed as in q4_grms.
extern "C" __global__ void q4_grms_inj(float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                       void* __restrict__ yh, int group, int groups, double eps, int half,
                                       const float* __restrict__ block, const float* __restrict__ iw)
{ PDL_PRO();
    __shared__ acc64 red[256];
    const u64 b = blockIdx.x;
    const int g = (int)(b % groups);
    const u64 t = b / groups;
    float* xr = x + b * group;
    const float* bk = block + t * group;
    const float a = iw[b];
    acc64 ss = D64(0.0);
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = xr[i] + a * bk[i];
        xr[i] = v;
        ss += (acc64)v * (acc64)v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = (float)(D64(1.0) / SQRT64(red[0] / (acc64)group + ((acc64)eps)));
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = xr[i] * inv * (1.0f + w[g * group + i]);
        if (y) y[b * group + i] = v;
        if (yh) stf(yh, b * group + i, v, half);
    }
}

// q4_grms_inj with a float4 a thread (blockDim * 4 >= group) and the sum of squares by warp shuffles: the same row
// in one pass of loads and two reduction steps instead of eight barrier-separated ones.
extern "C" __global__ void q4_grms_inj4(float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                        void* __restrict__ yh, int group, int groups, double eps, int half,
                                        const float* __restrict__ block, const float* __restrict__ iw)
{ PDL_PRO();
    __shared__ acc64 red[32];
    const u64 b = blockIdx.x;
    const int g = (int)(b % groups);
    const u64 t = b / groups;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    float4* xr = (float4*)(x + b * group);
    const int n4 = group >> 2;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    acc64 ss = D64(0.0);
    if (tid < n4) {
        v = xr[tid];
        if (block) {
            const float a = iw[b];
            const float4 bk = ((const float4*)(block + t * group))[tid];
            v.x = v.x + a * bk.x; v.y = v.y + a * bk.y; v.z = v.z + a * bk.z; v.w = v.w + a * bk.w;
            xr[tid] = v;
        }
        ss = (acc64)v.x * (acc64)v.x + (acc64)v.y * (acc64)v.y + (acc64)v.z * (acc64)v.z + (acc64)v.w * (acc64)v.w;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        acc64 r = lane < NW ? red[lane] : D64(0.0);
        for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    const float inv = (float)(D64(1.0) / SQRT64(red[0] / (acc64)group + ((acc64)eps)));
    if (tid < n4) {
        const float4 wv = ((const float4*)(w + (u64)g * group))[tid];
        const float o0 = v.x * inv * (1.0f + wv.x), o1 = v.y * inv * (1.0f + wv.y);
        const float o2 = v.z * inv * (1.0f + wv.z), o3 = v.w * inv * (1.0f + wv.w);
        const u64 i0 = b * group + (u64)tid * 4;
        if (y) ((float4*)(y + b * group))[tid] = make_float4(o0, o1, o2, o3);
        if (yh) { stf(yh, i0, o0, half); stf(yh, i0 + 1, o1, half); stf(yh, i0 + 2, o2, half); stf(yh, i0 + 3, o3, half); }
    }
}

// The low-rank middle of a gated residual. low [n][R]: the first `lr` values
// become the wt operand silu(low / hc); the rest are the inject weights
// 2 * sigmoid(low / hc) -> iw [n][R - lr].
extern "C" __global__ void q4_hc_low(const float* __restrict__ low, void* __restrict__ lowh, float* __restrict__ iw,
                                     int n, int R, int lr, float hc, int half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * R) return;
    const u64 t = i / R;
    const int c = (int)(i - t * R);
    const float v = low[i] / hc;
    if (c < lr) stf(lowh, t * lr + c, silu_f(v), half);
    else iw[t * (R - lr) + (c - lr)] = 2.0f * sigm(v);
}

// folded [n][hid] = sum over streams of sigmoid(mix) * normed / hc.
extern "C" __global__ void q4_hc_fold(const float* __restrict__ mix, const float* __restrict__ normed,
                                      float* __restrict__ folded, void* __restrict__ foldedh,
                                      int n, int hid, int hc, int half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    float acc = 0.0f;
    for (int s = 0; s < hc; s++) {
        const u64 o = (t * hc + s) * hid + d;
        acc += sigm(mix[o]) * normed[o] / (float)hc;
    }
    folded[i] = acc;
    if (foldedh) stf(foldedh, i, acc, half);
}

// hyper [n][hc][hid] += iw [n][hc] * block [n][hid].
extern "C" __global__ void q4_inject(float* __restrict__ hyper, const float* __restrict__ block,
                                     const float* __restrict__ iw, int n, int hid, int hc)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hc * hid) return;
    const u64 t = i / ((u64)hc * hid);
    const u64 r = i - t * hc * hid;
    const int s = (int)(r / hid);
    hyper[i] += iw[t * hc + s] * block[t * hid + (r - (u64)s * hid)];
}

// A right-aligned history of `cap` rows after a chunk of n rows:
// out[r] = the row at chunk position n - cap + r, older ones from `hist`.
extern "C" __global__ void q4_hist(const float* __restrict__ hist, const float* __restrict__ chunk,
                                   float* __restrict__ out, int cap, int width, int n)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)cap * width) return;
    const int r = (int)(i / width);
    const u64 c = i - (u64)r * width;
    const int src = n - cap + r;
    out[i] = src >= 0 ? chunk[(u64)src * width + c] : hist[(u64)(cap + src) * width + c];
}

// ---------------------------------------------------------------- GDN

// Causal depthwise conv over the chunk, then SiLU, in f64 as the host does:
// cq[t][c] = silu(qkv[t][c] * taps[c][kk-1] + sum_j row(t - (kk-1-j))[c] * taps[c][j]),
// rows before the chunk coming from the ring [(kk-1)][c_dim] (newest last).
extern "C" __global__ void q4_gdn_conv(const float* __restrict__ qkv, const float* __restrict__ ring,
                                       const float* __restrict__ taps, float* __restrict__ cq,
                                       int n, int c_dim, int kk)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * c_dim) return;
    const int t = (int)(i / c_dim);
    const u64 c = i - (u64)t * c_dim;
    const float* tp = taps + c * kk;
    acc64 acc = (acc64)qkv[i] * (acc64)tp[kk - 1];
    for (int j = 0; j < kk - 1; j++) {
        const int src = t - (kk - 1 - j);
        const float v = src >= 0 ? qkv[(u64)src * c_dim + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        acc += (acc64)v * (acc64)tp[j];
    }
    cq[i] = (float)(acc / (D64(1.0) + EXP64(-acc)));
}

// L2 norm of the q and k heads in place, a block per (token, head of 2 * nk):
// q heads also divided by sqrt(dk).
extern "C" __global__ void q4_gdn_prep(float* __restrict__ cq, int c_dim, int nk, int dk)
{ PDL_PRO();
    __shared__ acc64 red[256];
    const u64 t = blockIdx.x / (2 * nk);
    const int g = (int)(blockIdx.x - t * (2 * nk));
    float* xr = cq + t * c_dim + (u64)g * dk;
    acc64 ss = D64(0.0);
    for (int i = threadIdx.x; i < dk; i += blockDim.x) { const acc64 v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const acc64 nrm = SQRT64(red[0] + D64(1e-6));
    const float inv = (float)(D64(1.0) / (g < nk ? nrm * SQRT64((acc64)dk) : nrm));
    for (int i = threadIdx.x; i < dk; i += blockDim.x) xr[i] *= inv;
}

// Decay and write gates of every (token, value head):
// gb[t][h] = exp(-exp(a_log) * softplus(a + dt_bias)), gb[t][nv + h] = sigmoid(b).
// ab [n][2 * nv] holds a then b.
extern "C" __global__ void q4_gdn_gb(const float* __restrict__ ab, const float* __restrict__ a_log,
                                     const float* __restrict__ dt_bias, float* __restrict__ gb, int n, int nv)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nv) return;
    const u64 t = i / nv;
    const int h = (int)(i - t * nv);
    const acc64 x = (acc64)ab[t * 2 * nv + h] + (acc64)dt_bias[h];
    const acc64 sp = x > D64(20.0) ? x : LOG1P64(EXP64(x));
    gb[t * 2 * nv + h] = (float)EXP64(-EXP64((acc64)a_log[h]) * sp);
    gb[t * 2 * nv + nv + h] = (float)(D64(1.0) / (D64(1.0) + EXP64(-(acc64)ab[t * 2 * nv + nv + h])));
}

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The delta-rule scan over the chunk. A warp owns one column (value head h,
// value dim dj) of the state S [nv][dk][dv]; a lane keeps 4 of its dk = 128
// cells in registers. Per token: kv = S . k, delta = (v - g kv) beta,
// S = g S + k delta, o = q . S. Blocks of 256 threads = 8 columns.
extern "C" __global__ void q4_gdn_scan(const float* __restrict__ cq, const float* __restrict__ gb,
                                       const float* S, float* So, float* __restrict__ o,
                                       int n, int c_dim, int nv, int nk, int dk, int dv)
{ PDL_PRO();
    const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (col >= nv * dv) return;
    const int lane = threadIdx.x & 31;
    const int h = col / dv, dj = col - h * dv;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk + lane * 4, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    const u64 so = ((u64)h * dk + lane * 4) * dv + dj;
    const float* sp = S + so;
    float s0 = sp[0], s1 = sp[dv], s2 = sp[2 * dv], s3 = sp[3 * dv];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (u64)t * c_dim;
        const float4 kf = *(const float4*)(row + kofs);
        const float4 qf = *(const float4*)(row + qo);
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        const float kv = wsum(s0 * kf.x + s1 * kf.y + s2 * kf.z + s3 * kf.w);
        const float delta = (row[vo] - g * kv) * beta;
        s0 = g * s0 + kf.x * delta;
        s1 = g * s1 + kf.y * delta;
        s2 = g * s2 + kf.z * delta;
        s3 = g * s3 + kf.w * delta;
        const float ov = wsum(qf.x * s0 + qf.y * s1 + qf.z * s2 + qf.w * s3);
        if (lane == 0) o[(u64)t * nv * dv + (u64)h * dv + dj] = ov;
    }
    float* wp = So + so;
    wp[0] = s0; wp[dv] = s1; wp[2 * dv] = s2; wp[3 * dv] = s3;
}

// of [n][nv][dv] (wt) = rms(o) * norm * sigmoid(z), a block per (token, head), f64.
// The gated delta rule with a state column per thread (dk == 128): thread dj of head h keeps S[h][*][dj] in
// registers, so a warp reads and writes 32 neighbouring floats of a state row at a time (q4_gdn_scan gave each lane
// four rows of one column: a 32-byte sector for every float). The key and query of a token are broadcast from
// shared memory; no warp sums. Grid (nv, dv / blockDim), block <= 128 columns.
extern "C" __global__ void q4_gdn_scan_c(const float* __restrict__ cq, const float* __restrict__ gb,
                                         const float* S, float* So, float* __restrict__ o,
                                         int n, int c_dim, int nv, int nk, int dk, int dv)
{ PDL_PRO();
    __shared__ float qs[128], ks[128];
    const int h = blockIdx.x, dj = blockIdx.y * blockDim.x + threadIdx.x;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    const float* sp = S + (u64)h * dk * dv + dj;
    float st[128];
#pragma unroll
    for (int k = 0; k < 128; k++) st[k] = sp[(u64)k * dv];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (u64)t * c_dim;
        __syncthreads();
        for (int k = threadIdx.x; k < 128; k += blockDim.x) { qs[k] = row[qo + k]; ks[k] = row[kofs + k]; }
        __syncthreads();
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        float kv = 0.0f;
#pragma unroll
        for (int k = 0; k < 128; k++) kv += st[k] * ks[k];
        const float delta = (row[vo] - g * kv) * beta;
        float ov = 0.0f;
#pragma unroll
        for (int k = 0; k < 128; k++) {
            st[k] = g * st[k] + ks[k] * delta;
            ov += qs[k] * st[k];
        }
        o[(u64)t * nv * dv + (u64)h * dv + dj] = ov;
    }
    float* wp = So + (u64)h * dk * dv + dj;
#pragma unroll
    for (int k = 0; k < 128; k++) wp[(u64)k * dv] = st[k];
}

#ifndef GDN_PARTS
#define GDN_PARTS 8
#endif
// `q4_gdn_scan_c` with each state column split over GDN_PARTS adjacent lanes (128 / GDN_PARTS key rows each):
// GDN_PARTS times the blocks of a head, no barrier between tokens (a lane reads its slice of q and k straight
// from the row, the block's lanes share the loads), the two dot products summed over the parts by shuffles.
// Grid (nv, dv * GDN_PARTS / blockDim.x).
extern "C" __global__ void q4_gdn_scan_p(const float* __restrict__ cq, const float* __restrict__ gb,
                                         const float* S, float* So, float* __restrict__ o,
                                         int n, int c_dim, int nv, int nk, int dk, int dv)
{ PDL_PRO();
    constexpr int P = GDN_PARTS, KP = 128 / GDN_PARTS;
    const int h = blockIdx.x, tid = threadIdx.x, p = tid % P;
    const int dj = blockIdx.y * (blockDim.x / P) + tid / P;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk + (u64)p * KP, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    const float* sp = S + (u64)h * dk * dv + dj;
    float st[KP];
#pragma unroll
    for (int k = 0; k < KP; k++) st[k] = sp[(u64)(p * KP + k) * dv];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (u64)t * c_dim;
        float qs[KP], ks[KP];
#pragma unroll
        for (int k = 0; k < KP; k += 4) {
            const float4 a = *(const float4*)(row + qo + k), b = *(const float4*)(row + kofs + k);
            qs[k] = a.x; qs[k + 1] = a.y; qs[k + 2] = a.z; qs[k + 3] = a.w;
            ks[k] = b.x; ks[k + 1] = b.y; ks[k + 2] = b.z; ks[k + 3] = b.w;
        }
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        const float vv = row[vo];
        float kv = 0.0f;
#pragma unroll
        for (int k = 0; k < KP; k++) kv += st[k] * ks[k];
#pragma unroll
        for (int m = 1; m < P; m <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, m);
        const float delta = (vv - g * kv) * beta;
        float ov = 0.0f;
#pragma unroll
        for (int k = 0; k < KP; k++) {
            st[k] = g * st[k] + ks[k] * delta;
            ov += qs[k] * st[k];
        }
#pragma unroll
        for (int m = 1; m < P; m <<= 1) ov += __shfl_xor_sync(0xffffffffu, ov, m);
        if (p == 0) o[(u64)t * nv * dv + (u64)h * dv + dj] = ov;
    }
    float* wp = So + (u64)h * dk * dv + dj;
#pragma unroll
    for (int k = 0; k < KP; k++) wp[(u64)(p * KP + k) * dv] = st[k];
}

extern "C" __global__ void q4_gdn_out(const float* __restrict__ o, const float* __restrict__ z,
                                      const float* __restrict__ norm, void* __restrict__ ofh,
                                      int dv, double eps, int half)
{ PDL_PRO();
    __shared__ acc64 red[256];
    const u64 b = blockIdx.x;
    const float* xr = o + b * dv;
    acc64 ss = D64(0.0);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) { const acc64 v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const acc64 inv = D64(1.0) / SQRT64(red[0] / (acc64)dv + ((acc64)eps));
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const acc64 gate = D64(1.0) / (D64(1.0) + EXP64(-(acc64)z[b * dv + i]));
        stf(ofh, b * dv + i, (float)(((acc64)xr[i] * inv) * (acc64)norm[i] * gate), half);
    }
}

// ---------------------------------------------------------------- QSA

// Head-wise zero-centred RMS norm, then rotary, a block per (item, head).
// The value is the mean of `nsum` consecutive source rows (1 = the row
// itself; the indexer's block keys average `compress_ratio` raw keys):
//   v[d] = sum_j src[(item * nsum + j) * sstride + head * shead + d] / nsum
//   v[d] *= rsqrt(mean(v^2) + eps) * (1 + w[d]);  rotate the first `rot` dims at
//   position pos0 + item * pstep;
//   dst[item * dstride + (head / dgroup) * dgstride + (head % dgroup) * dhead + d] = v[d].
extern "C" __global__ void q4_norm_rope(const float* __restrict__ src, float* __restrict__ dst,
                                        const float* __restrict__ w, const float* __restrict__ invf,
                                        int heads, int hd, u64 sstride, u64 shead, u64 dstride, u64 dhead,
                                        int dgroup, u64 dgstride, int rot, int nfreq, int pos0, int pstep,
                                        int nsum, double eps)
{ PDL_PRO();
    __shared__ float sv[256];
    __shared__ acc64 red[256];
    const u64 item = blockIdx.x / heads;
    const int head = (int)(blockIdx.x - item * heads);
    const int d = threadIdx.x;
    float v = 0.0f;
    if (d < hd) {
        for (int j = 0; j < nsum; j++) v += src[(item * nsum + j) * sstride + (u64)head * shead + d] / (float)nsum;
    }
    red[d] = (acc64)v * (acc64)v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    const float inv = (float)(D64(1.0) / SQRT64(red[0] / (acc64)hd + ((acc64)eps)));
    if (d < hd) sv[d] = v * (inv * (1.0f + w[d]));
    __syncthreads();
    if (d >= hd) return;
    int rd = rot < hd ? rot : hd;
    if (rd > 2 * nfreq) rd = 2 * nfreq;
    const int hf = rd >> 1;
    float out = sv[d];
    if (d < rd) {
        const int i = d < hf ? d : d - hf;
        const float ang = (float)(pos0 + (int)item * pstep) * invf[i];
        const float a = cosf(ang), b = sinf(ang);
        out = d < hf ? sv[d] * a - sv[d + hf] * b : sv[d] * a + sv[d - hf] * b;
    }
    dst[item * dstride + (u64)(head / dgroup) * dgstride + (u64)(head % dgroup) * dhead + d] = out;
}

// Indexer block scores: sc[t][b] = sum_h max(DS[t][h][b], 0) * inv.
extern "C" __global__ void q4_idx_reduce(const float* __restrict__ ds, float* __restrict__ sc,
                                         int nb, int ih, int nact, float inv)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)nact * nb) return;
    const u64 t = i / nb, b = i - t * nb;
    float s = 0.0f;
    for (int h = 0; h < ih; h++) s += fmaxf(ds[(t * ih + h) * nb + b], 0.0f);
    sc[i] = s * inv;
}

// The `keep` best blocks of every token (a block per token), ties to the
// lower index: bits of maskb[row][b / 32]. Token row r sits at position
// p0 + r and sees the blocks b < (p0 + r + 1) / cr. Scores are >= 0, so
// their bit patterns order as the values do: the threshold is found by
// bisection over the 32 bits. A thread owns `per` blocks (a multiple of 32).
extern "C" __global__ void q4_idx_topk(const float* __restrict__ sc, u32* __restrict__ maskb,
                                       int nb, int mwords, int keep, int p0, int cr, int per)
{ PDL_PRO();
    __shared__ int total;
    __shared__ int eqc[256];
    const int r = blockIdx.x;
    int complete = (p0 + r + 1) / cr;
    if (complete > nb) complete = nb;
    const u32* row = (const u32*)(sc + (u64)r * nb);
    const int b0 = threadIdx.x * per;
    int b1 = b0 + per;
    if (b1 > complete) b1 = complete;
    u32 T = 0;
#if TOPK_RADIX
    // the keep-th largest key by four 8-bit digits (a histogram and a scan each) instead of 32 one-bit passes of
    // three barriers: the same threshold, so the same mask
    {
        __shared__ int hist[256];
        __shared__ int dig, above, all;
        u32 prefix = 0, pmask = 0;
        int need = keep;
        for (int pass = 0; pass < 4; pass++) {
            const int sh = 24 - 8 * pass;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int b = b0; b < b1; b++) {
                const u32 v = row[b];
                if ((v & pmask) == prefix) atomicAdd(&hist[(v >> sh) & 255u], 1);
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                // lane l holds digits 255 - 8l .. 248 - 8l (from the top)
                const int l = threadIdx.x;
                int loc[8], ls = 0;
#pragma unroll
                for (int k = 0; k < 8; k++) { loc[k] = hist[255 - 8 * l - k]; ls += loc[k]; }
                int inc = ls;
                for (int o = 1; o < 32; o <<= 1) {
                    const int u = __shfl_up_sync(0xffffffffu, inc, o);
                    if (l >= o) inc += u;
                }
                const int exc = inc - ls;
                const int tot = __shfl_sync(0xffffffffu, inc, 31);
                if (l == 0) all = tot;
                if (exc < need && inc >= need) {
                    int run = exc;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        if (run + loc[k] >= need) { dig = 255 - 8 * l - k; above = run; break; }
                        run += loc[k];
                    }
                }
            }
            __syncthreads();
            if (all < need) break;   // fewer keys than wanted: the threshold stays where it is (all are kept)
            prefix |= (u32)dig << sh;
            pmask |= 255u << sh;
            need -= above;
            __syncthreads();
        }
        T = (all < need && pmask == 0) ? 0u : prefix;
    }
#else
    for (int bit = 31; bit >= 0; bit--) {
        const u32 cand = T | (1u << bit);
        if (threadIdx.x == 0) total = 0;
        __syncthreads();
        int c = 0;
        for (int b = b0; b < b1; b++) c += row[b] >= cand;
        if (c) atomicAdd(&total, c);
        __syncthreads();
        if (total >= keep) T = cand;
        __syncthreads();
    }
#endif
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    int gt = 0, eq = 0;
    for (int b = b0; b < b1; b++) { gt += row[b] > T; eq += row[b] == T; }
    eqc[threadIdx.x] = eq;
    if (gt) atomicAdd(&total, gt);
    __syncthreads();
    const int need_eq = keep - total;
    int rank = 0;
    for (int i = 0; i < (int)threadIdx.x; i++) rank += eqc[i];
    for (int wi = 0; wi < per / 32; wi++) {
        const int wb = b0 + wi * 32;
        if (wb / 32 >= mwords) break;
        u32 m = 0;
        for (int k = 0; k < 32; k++) {
            const int b = wb + k;
            if (b >= b1) break;
            const u32 v = row[b];
            if (v > T) m |= 1u << k;
            else if (v == T) { if (rank < need_eq) m |= 1u << k; rank++; }
        }
        maskb[(u64)r * mwords + wb / 32] = m;
    }
}

// Softmax down every column of S [L] x [cols] in place, a block per column.
// Column c belongs to the token at position ppos0 + c / hpt, which sees
// j < npos: all of them while the indexer is idle (complete <= keep), else
// the kept blocks plus the tail past the last complete block.
extern "C" __global__ void q4_qsa_softmax(float* __restrict__ S, const u32* __restrict__ maskb,
                                          int L, int hpt, int ppos0, int mask_p0, int mwords, int keep, int cr)
{ PDL_PRO();
    __shared__ float red[256];
    float* col = S + (u64)blockIdx.x * L;
    const int p = ppos0 + blockIdx.x / hpt;
    const int npos = p + 1;
    const int complete = npos / cr;
    const bool idx = complete > keep;
    const u32* mrow = idx ? maskb + (u64)(p - mask_p0) * mwords : (const u32*)0;
    const int tail = complete * cr;
    float mx = -__int_as_float(0x7f800000);
    for (int j = threadIdx.x; j < npos; j += blockDim.x) {
        bool vis = true;
        if (idx && j < tail) { const int b = j / cr; vis = (mrow[b >> 5] >> (b & 31)) & 1u; }
        if (vis) mx = fmaxf(mx, col[j]);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float z = 0.0f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        bool vis = j < npos;
        if (vis && idx && j < tail) { const int b = j / cr; vis = (mrow[b >> 5] >> (b & 31)) & 1u; }
        const float e = vis ? expf(col[j] - mx) : 0.0f;
        col[j] = e;
        z += e;
    }
    red[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float zi = 1.0f / fmaxf(red[0], 1.17549435e-38f);
    for (int j = threadIdx.x; j < L; j += blockDim.x) col[j] *= zi;
}

// merged [n][nh][hd] (wt) = O (group-major, as Q) * sigmoid(gate half of QG).
extern "C" __global__ void q4_qsa_merge(const float* __restrict__ O, const float* __restrict__ qg,
                                        void* __restrict__ mergedh, int n, int nh, int groups, int hd, int half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nh * hd) return;
    const u64 t = i / ((u64)nh * hd);
    const u64 r = i - t * nh * hd;
    const int qh = (int)(r / hd), d = (int)(r - (u64)qh * hd);
    const int kvh = qh / groups, g = qh - kvh * groups;
    const float v = O[(((u64)kvh * n + t) * groups + g) * hd + d];
    stf(mergedh, i, v * sigm(qg[t * (u64)nh * 2 * hd + (u64)qh * 2 * hd + hd + d]), half);
}

// ---------------------------------------------------------------- PLE

// gated [n][hc][hid] = sigmoid(signed sqrt of <key, query> / sqrt(hid)) * value,
// a block per (token, stream).
extern "C" __global__ void q4_ple_gate(const float* __restrict__ key, const float* __restrict__ query,
                                       const float* __restrict__ value, float* __restrict__ gated, int hid, int hc)
{ PDL_PRO();
    __shared__ float red[256];
    const u64 b = blockIdx.x;
    const u64 t = b / hc;
    const float* kr = key + b * hid;
    const float* qr = query + b * hid;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) acc += kr[i] * qr[i];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float dot = red[0] / sqrtf((float)hid);
    const float g = sigm(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    for (int i = threadIdx.x; i < hid; i += blockDim.x) gated[b * hid + i] = g * value[t * hid + i];
}

// hyper += silu(dilated causal depthwise conv of normed) + gated; rows before
// the chunk come from hist [(kernel-1) * dil][width] (newest last).
extern "C" __global__ void q4_ple_conv(const float* __restrict__ normed, const float* __restrict__ hist,
                                       const float* __restrict__ conv, const float* __restrict__ gated,
                                       float* __restrict__ hyper, int n, int width, int kernel, int dil)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * width) return;
    const int t = (int)(i / width);
    const u64 ch = i - (u64)t * width;
    const int cap = (kernel - 1) * dil;
    const float* tp = conv + ch * kernel;
    float sum = tp[kernel - 1] * normed[i];
    for (int tap = 0; tap < kernel - 1; tap++) {
        const int src = t - (kernel - 1 - tap) * dil;
        const float v = src >= 0 ? normed[(u64)src * width + ch] : hist[(u64)(cap + src) * width + ch];
        sum += tp[tap] * v;
    }
    hyper[i] += silu_f(sum) + gated[i];
}

// ---------------------------------------------------------------- MoE

// Softmax router: the top `topk` of the first `ne` logits of every token (a
// block per token), ties to the lower index, weights renormalised over the
// picks in rank order. logits [n][R], R >= ne.
extern "C" __global__ void q4_route(const float* __restrict__ logits, int* __restrict__ picks,
                                    float* __restrict__ wts, int ne, int R, int topk, float tail,
                                    const unsigned long long* __restrict__ tbl)
{ PDL_PRO();
    __shared__ float pe[1024];
    __shared__ float red[256];
    __shared__ int spick[32];
    __shared__ float sw[32];
    const float* l = logits + (u64)blockIdx.x * R;
    float mx = -__int_as_float(0x7f800000);
    for (int i = threadIdx.x; i < ne; i += blockDim.x) if (!tbl || tbl[i]) mx = fmaxf(mx, l[i]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    // the exponentials and their sum (a tree over the block: one thread summing 512 of them in a row was
    // the better part of the kernel)
    float ls = 0.0f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        const float v = (!tbl || tbl[i]) ? expf(l[i] - mx) : 0.0f;
        pe[i] = v;
        ls += v;
    }
    __syncthreads();
    red[threadIdx.x] = ls;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    const float s = red[0];
    __syncthreads();
    for (int i = threadIdx.x; i < ne; i += blockDim.x) pe[i] /= s;
    __syncthreads();
    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        const float p = pe[i];
        int rank = 0;
        for (int j = 0; j < ne; j++) rank += (pe[j] > p) || (pe[j] == p && j < i);
        if (rank < topk) { spick[rank] = i; sw[rank] = p; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ws = 0.0f;
        for (int r = 0; r < topk; r++) ws += sw[r];
        // the route tail: picks lighter than `tail` of the picked weight are dropped
        if (tail > 0.0f && ws > 0.0f) {
            const float cut = tail * ws;
            float keep = 0.0f;
            for (int r = 0; r < topk; r++) {
                if (sw[r] < cut) { sw[r] = 0.0f; spick[r] = -1; } else keep += sw[r];
            }
            if (keep > 0.0f) ws = keep;
        }
        for (int r = 0; r < topk; r++) {
            picks[(u64)blockIdx.x * topk + r] = spick[r];
            wts[(u64)blockIdx.x * topk + r] = sw[r] / ws;
        }
    }
}

// q4tp experts -> dense wt planes. A raw expert slot holds gate [inter][hid],
// up [inter][hid], down [hid][inter], each as [nibbles][row (f16 lo, f16
// step)][5-bit group codes]. Grid (words of a matrix / 256, 3 * B): a thread
// expands one u32 of nibbles = 8 weights of matrix blockIdx.y % 3 of expert
// blockIdx.y / 3 into plane j of Wg / Wu / Wd.
extern "C" __global__ void q4_expand_exp(const u64* __restrict__ slots, void* __restrict__ Wg, void* __restrict__ Wu,
                                         void* __restrict__ Wd, int inter, int hid, int half, int gbits)
{ PDL_PRO();
    const int m = blockIdx.y % 3, j = blockIdx.y / 3;
    const int rows = m == 2 ? hid : inter, cols = m == 2 ? inter : hid;
    const u32 wi = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 words = (u64)rows * cols / 8;
    if (wi >= words || !slots[j]) return;
    // gate and up are 2-bit (q2tp) or 4-bit (q4tp); the down matrix is 4-bit; section offsets of a matrix,
    // and of the three matrices in the slot
    const bool two = gbits == 2 && m < 2;
    const int gpr_gu = hid >> 5;
    const u64 size_gu = (u64)inter * gpr_gu * (gbits == 2 ? 8 : 16) + (u64)inter * 4 + (u64)inter * ((gpr_gu * 5 + 7) / 8);
    const int gpr = cols >> 5;
    const u64 params_off = (u64)rows * gpr * (two ? 8 : 16), codes_off = params_off + (u64)rows * 4;
    const int stride = (gpr * 5 + 7) / 8;
    // a slot with the top bit is a descriptor: the three matrices' addresses (pinned host pages the card reads in place)
    const u64 sj = slots[j];
    const u8* mat = (sj >> 63) ? (const u8*)((const u64*)(sj & 0x7fffffffffffffffull))[m] : (const u8*)sj + (u64)m * size_gu;
    const u64 base = (u64)wi * 8;
    const int r = (int)(base / cols);
    const int g = (int)(base - (u64)r * cols) >> 5;
    const u16* pp = (const u16*)(mat + params_off + (u64)r * 4);
    const u8* codes = mat + codes_off + (u64)r * stride;
    const int bit = g * 5, cb = bit >> 3, sh = bit & 7;
    const u32 lo = codes[cb];
    const u32 hi = (sh > 3 && cb + 1 < stride) ? codes[cb + 1] : 0u;
    const int code = (int)(((lo | (hi << 8)) >> sh) & 0x1Fu);
    float s = exp2f(h2f(pp[0]));
    const float ratio = exp2f(h2f(pp[1]));
    void* W = m == 0 ? Wg : (m == 1 ? Wu : Wd);
    const u64 o = (u64)j * rows * cols + base;
    if (two) {
        // rung 0 is zero, rung c is the q4tp rung c - 1
        for (int c2 = 1; c2 < code; c2++) s *= ratio;
        if (code == 0) s = 0.0f;
        const u32 w = *(const u16*)(mat + (u64)wi * 2);
        for (int k = 0; k < 8; k++) stf(W, o + k, ((float)((w >> (2 * k)) & 3u) - 1.5f) * s, half);
        return;
    }
    const u32 w = *(const u32*)(mat + (u64)wi * 4);
    for (int c2 = 0; c2 < code; c2++) s *= ratio;
    for (int k = 0; k < 4; k++) {
        const u32 by = (w >> (8 * k)) & 0xFFu;
        stf(W, o + 2 * k, ((float)(by & 15u) - 8.0f) * s, half);
        stf(W, o + 2 * k + 1, ((float)(by >> 4) - 8.0f) * s, half);
    }
}

// q8_2f -> dense wt [rows][cols]: body i8 * row scale * column field, a thread
// per 4 weights.
extern "C" __global__ void q4_expand_q8(const u32* __restrict__ body, const float* __restrict__ rs,
                                        const float* __restrict__ col, void* __restrict__ W,
                                        int rows, int cols, int half)
{ PDL_PRO();
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2;
    if (idx >= (u64)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (u64)r * qpr);
    const u32 w = body[idx];
    const float s = rs[r];
    const float* c = col + (u64)q * 4;
    stf(W, idx * 4, (float)((int)(w << 24) >> 24) * s * c[0], half);
    stf(W, idx * 4 + 1, (float)((int)(w << 16) >> 24) * s * c[1], half);
    stf(W, idx * 4 + 2, (float)((int)(w << 8) >> 24) * s * c[2], half);
    stf(W, idx * 4 + 3, (float)((int)w >> 24) * s * c[3], half);
}

// f32 -> wt.
extern "C" __global__ void q4_to_wt(const float* __restrict__ src, void* __restrict__ dst, u64 n, int half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(dst, i, src[i], half);
}

extern "C" __global__ void q4_zero(float* __restrict__ p, u64 n)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

// Xg [rows][hid] (wt) = the token rows of xh named by rowtok, zeros for -1.
extern "C" __global__ void q4_gather(const void* __restrict__ xh, const int* __restrict__ rowtok,
                                     void* __restrict__ xg, int rows, int hid, int half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * hid) return;
    const u64 r = i / hid;
    const int tok = rowtok[r];
    const u64 src = (u64)(tok < 0 ? 0 : tok) * hid + (i - r * hid);
    if (half) ((u16*)xg)[i] = tok < 0 ? (u16)0 : ((const u16*)xh)[src];
    else ((float*)xg)[i] = tok < 0 ? 0.0f : ((const float*)xh)[src];
}

// a = silu(g) * u * inv over n values; g, u in `in_half` type, a in `out_half`.
extern "C" __global__ void q4_act(const void* __restrict__ g, const void* __restrict__ u, void* __restrict__ a,
                                  u64 n, int in_half, int out_half, float inv)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(a, i, silu_f(ldf(g, i, in_half)) * ldf(u, i, in_half) * inv, out_half);
}

// out [n][hid] (+)= sum over the token's picks in this batch of wts * D[row].
// tokrow [n][topk]: the row of D for the pick, -1 = not in this batch.
extern "C" __global__ void q4_scatter(const void* __restrict__ D, const int* __restrict__ tokrow,
                                      const float* __restrict__ wts, float* __restrict__ out,
                                      int n, int hid, int topk, int d_half, int init)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    float acc = 0.0f;
    for (int k = 0; k < topk; k++) {
        const int row = tokrow[t * topk + k];
        if (row >= 0) acc += wts[t * topk + k] * ldf(D, (u64)row * hid + d, d_half);
    }
    out[i] = init ? acc : out[i] + acc;
}

// out [n][hid] += g * ds [n][hid], g = sigmoid(logits[t][gcol]) or 1 (gcol < 0).
extern "C" __global__ void q4_shared_add(float* __restrict__ out, const float* __restrict__ ds,
                                         const float* __restrict__ logits, int n, int hid, int R, int gcol)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid;
    const float g = gcol < 0 ? 1.0f : sigm(logits[t * R + gcol]);
    out[i] += g * ds[i];
}

// ---------------------------------------------------------------- small path
// A decode step or a verify round: a few token rows (NB <= 8) straight from
// the stored weights — int8 bodies, f32 planes, raw q4tp expert slots — in
// f32, with no dense planes and no cuBLAS call. A weight is read once and
// used for every row of the round.

// A matvec job of `q4_mvb*`: the layout of the backend's MvJob.
struct MvJob { u64 w, aux0, aux1, y, res; int rows, kind, blk0, rpb; };

// Eight 4-bit weights of a word against eight staged values `st` apart.
// An integer 0..2^23 as a float without I2F (a quarter-rate instruction): its bits under the exponent of 2^23.
// Exact, so the sums below are bit-identical to (float)n.
__device__ __forceinline__ float u2f_small(u32 n) { return __int_as_float(0x4B000000u | n) - 8388608.0f; }
// Four int8 codes (low byte first) in one word, the operand of __dp4a.
__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | (d << 24);
}

#ifndef NIB_FAST
#define NIB_FAST 0
#endif
__device__ __forceinline__ float nib8(u32 w, const float* p, int st, float a) {
#if NIB_FAST
#pragma unroll
    for (int i = 0; i < 8; i++) a += u2f_small((w >> (4 * i)) & 15u) * p[i * st];
#else
    a += (float)(w & 15u) * p[0];
    a += (float)((w >> 4) & 15u) * p[st];
    a += (float)((w >> 8) & 15u) * p[2 * st];
    a += (float)((w >> 12) & 15u) * p[3 * st];
    a += (float)((w >> 16) & 15u) * p[4 * st];
    a += (float)((w >> 20) & 15u) * p[5 * st];
    a += (float)((w >> 24) & 15u) * p[6 * st];
    a += (float)(w >> 28) * p[7 * st];
#endif
    return a;
}

// The 5-bit ladder code of group g of a row.
__device__ __forceinline__ int q4_code(const u8* codes, int g, int stride) {
    const int bit = g * 5, cb = bit >> 3, sh = bit & 7;
    const u32 lo = codes[cb];
    const u32 hi = (sh > 3 && cb + 1 < stride) ? codes[cb + 1] : 0u;
    return (int)(((lo | (hi << 8)) >> sh) & 0x1Fu);
}

// The 32 scales of a row: t[0] = 2^lo, t[c] = t[c-1] * 2^step, as the host.
__device__ __forceinline__ void q4_ladder(const u16* pp, float* l) {
    float s = exp2f(h2f(pp[0]));
    const float ratio = exp2f(h2f(pp[1]));
    l[0] = s;
    for (int c = 1; c < 32; c++) { s *= ratio; l[c] = s; }
}

// The 32 scales of a q2tp row: rung 0 is exactly zero, rungs 1..=31 are the q4tp ladder shifted by one.
__device__ __forceinline__ void q2_ladder(const u16* pp, float* l) {
    float s = exp2f(h2f(pp[0]));
    const float ratio = exp2f(h2f(pp[1]));
    l[0] = 0.0f;
    l[1] = s;
    for (int c = 2; c < 32; c++) { s *= ratio; l[c] = s; }
}

// Sixteen 2-bit weights of a word against sixteen staged values `st` apart.
__device__ __forceinline__ float crumb16(u32 w, const float* p, int st, float a) {
#pragma unroll
    for (int i = 0; i < 16; i++) a += (NIB_FAST ? u2f_small((w >> (2 * i)) & 3u) : (float)((w >> (2 * i)) & 3u)) * p[i * st];
    return a;
}

// 16-byte words of a q8_2f row a lane loads before using any of them.
// Ints of an int8-staged 16-column chunk of the matvec (four packed, the scale; 12 keeps a phase of 128-bit loads on
// distinct banks).
#define MQ_G 12
#ifndef MV_CONTIG
#define MV_CONTIG 1
#endif
#define MV_CS 20
// widest rows the side-by-side layout takes (its shared memory per input row is 5 bytes a column; the host sizes
// the launch the same way, see `mv_row_bytes`)
#define MV_CMAX 4096
#ifndef MV_SG
#define MV_SG 3
#endif
#ifndef MV_AH
#define MV_AH 8
#endif

// Up to five matvecs of one input over NB rows: y_j [NB][rows_j] = W_j x.
// x [NB] rows `xstride` apart, `cols` wide. Kind 7 = q8_2f (int8 body, row
// scale aux0, column field aux1), kind 4 = f32. A block serves `rpb` rows of
// the job its index falls into; shared memory holds the NB staged inputs.
template <int NB, bool Q>
__device__ __forceinline__ void mvb_t(MvJob j0, MvJob j1, MvJob j2, MvJob j3, MvJob j4, int njobs,
                                      const float* __restrict__ x, int xstride, int cols)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int bx = blockIdx.x;
    MvJob jb = j0;
    if (njobs > 1 && bx >= j1.blk0) jb = j1;
    if (njobs > 2 && bx >= j2.blk0) jb = j2;
    if (njobs > 3 && bx >= j3.blk0) jb = j3;
    if (njobs > 4 && bx >= j4.blk0) jb = j4;
    const int row0 = (bx - jb.blk0) * jb.rpb;
    const int row1 = min(row0 + jb.rpb, jb.rows);
    float* y = (float*)jb.y;
    PDL_PRO();
    if (Q && jb.kind == 7 && ((cols | xstride) & 3) == 0 && (((size_t)x | (size_t)jb.aux1) & 15) == 0) {
        // the int8 path (fast precision): the inputs times the column field are quantized per 16 columns (a lane's
        // word) to int8 with their scale, and a word of weights meets them in four __dp4a
        const int nch = cols >> 4;
        const float* col = (const float*)jb.aux1;
        const float* rs = (const float*)jb.aux0;
        int* xq = (int*)sh;                       // [NB][nch][MQ_G]: four packed ints, the scale
        const int c4 = cols >> 2, tot = NB * c4;
        for (int t = tid; t < tot; t += T) {
            const unsigned am = __activemask();
            const int b = t / c4, m = t - b * c4;
            const float4 cv = ((const float4*)col)[m];
            const float4 xv = ((const float4*)(x + (size_t)b * xstride))[m];
            const float v0 = xv.x * cv.x, v1 = xv.y * cv.y, v2 = xv.z * cv.z, v3 = xv.w * cv.w;
            float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            mx = fmaxf(mx, __shfl_xor_sync(am, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(am, mx, 2));
            const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
            int* dst = xq + ((size_t)b * nch + (m >> 2)) * MQ_G;
            dst[m & 3] = pack4(__float2int_rn(v0 * inv), __float2int_rn(v1 * inv), __float2int_rn(v2 * inv),
                               __float2int_rn(v3 * inv));
            if ((m & 3) == 0) dst[4] = __float_as_int(mx > 0.0f ? mx / 127.0f : 0.0f);
        }
        __syncthreads();
        for (int r = row0 + warp; r < row1; r += NW) {
            const uint4* W = (const uint4*)(jb.w + (size_t)r * cols);
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            for (int c0 = lane; c0 < nch; c0 += 32 * MV_AH) {
                uint4 w[MV_AH];
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) w[j] = W[c];
                }
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) {
                        for (int b = 0; b < NB; b++) {
                            const int* g = xq + ((size_t)b * nch + c) * MQ_G;
                            const int4 q = *(const int4*)g;
                            int sacc = __dp4a((int)w[j].x, q.x, 0);
                            sacc = __dp4a((int)w[j].y, q.y, sacc);
                            sacc = __dp4a((int)w[j].z, q.z, sacc);
                            sacc = __dp4a((int)w[j].w, q.w, sacc);
                            acc[b] += __int_as_float(g[4]) * (float)sacc;
                        }
                    }
                }
            }
            const float sc = rs[r];
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a * sc;
            }
        }
        return;
    }
#if MV_CONTIG
    if (jb.kind == 7 && cols <= MV_CMAX && ((cols | xstride) & 3) == 0 && (((size_t)x | (size_t)jb.aux1) & 15) == 0) {
        // a lane's 16 inputs side by side (stride MV_CS floats, a phase of 128-bit loads on distinct banks): four
        // vector loads a word and row instead of sixteen scalar ones with an address multiply each
        const int nch = cols >> 4, XS = nch * MV_CS;
        const float* col = (const float*)jb.aux1;
        const float* rs = (const float*)jb.aux0;
        const int c4 = cols >> 2;
        for (int m0 = tid; m0 < c4; m0 += T * MV_SG) {
            float4 cv[MV_SG], xv[MV_SG][NB];
#pragma unroll
            for (int k = 0; k < MV_SG; k++) {
                const int m = m0 + k * T;
                if (m < c4) {
                    cv[k] = ((const float4*)col)[m];
#pragma unroll
                    for (int b = 0; b < NB; b++) xv[k][b] = ((const float4*)(x + (size_t)b * xstride))[m];
                }
            }
#pragma unroll
            for (int k = 0; k < MV_SG; k++) {
                const int m = m0 + k * T;
                if (m < c4) {
                    const int c = m >> 2, q = (m & 3) * 4;
#pragma unroll
                    for (int b = 0; b < NB; b++)
                        *(float4*)(sh + b * XS + c * MV_CS + q) =
                            make_float4(xv[k][b].x * cv[k].x, xv[k][b].y * cv[k].y, xv[k][b].z * cv[k].z, xv[k][b].w * cv[k].w);
                }
            }
        }
        __syncthreads();
        for (int r = row0 + warp; r < row1; r += NW) {
            const uint4* W = (const uint4*)(jb.w + (size_t)r * cols);
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            for (int c0 = lane; c0 < nch; c0 += 32 * MV_AH) {
                uint4 w[MV_AH];
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) w[j] = W[c];
                }
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) {
                        float wf[16];
                        wf[0] = (float)((int)(w[j].x << 24) >> 24);  wf[1] = (float)((int)(w[j].x << 16) >> 24);
                        wf[2] = (float)((int)(w[j].x << 8) >> 24);   wf[3] = (float)((int)w[j].x >> 24);
                        wf[4] = (float)((int)(w[j].y << 24) >> 24);  wf[5] = (float)((int)(w[j].y << 16) >> 24);
                        wf[6] = (float)((int)(w[j].y << 8) >> 24);   wf[7] = (float)((int)w[j].y >> 24);
                        wf[8] = (float)((int)(w[j].z << 24) >> 24);  wf[9] = (float)((int)(w[j].z << 16) >> 24);
                        wf[10] = (float)((int)(w[j].z << 8) >> 24);  wf[11] = (float)((int)w[j].z >> 24);
                        wf[12] = (float)((int)(w[j].w << 24) >> 24); wf[13] = (float)((int)(w[j].w << 16) >> 24);
                        wf[14] = (float)((int)(w[j].w << 8) >> 24);  wf[15] = (float)((int)w[j].w >> 24);
                        for (int b = 0; b < NB; b++) {
                            const float4* p = (const float4*)(sh + b * XS + c * MV_CS);
                            float a = 0.0f;
#pragma unroll
                            for (int q = 0; q < 4; q++) {
                                const float4 v = p[q];
                                a += wf[4 * q] * v.x;
                                a += wf[4 * q + 1] * v.y;
                                a += wf[4 * q + 2] * v.z;
                                a += wf[4 * q + 3] * v.w;
                            }
                            acc[b] += a;
                        }
                    }
                }
            }
            const float sc = rs[r];
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a * sc;
            }
        }
        return;
    }
#endif
    if (jb.kind == 7) {
        const int nch = cols >> 4, CP = nch | 1, XS = 16 * CP;
        const float* col = (const float*)jb.aux1;
        // the inputs times the column field into shared memory, four columns a thread with the loads of every row
        // in flight together (a scalar loop waited out the L2 latency on each column)
        if (((cols | xstride) & 3) == 0 && (((size_t)x | (size_t)col) & 15) == 0) {
            const int c4 = cols >> 2;
            for (int m0 = tid; m0 < c4; m0 += T * MV_SG) {
                float4 cv[MV_SG], xv[MV_SG][NB];
#pragma unroll
                for (int k = 0; k < MV_SG; k++) {
                    const int m = m0 + k * T;
                    if (m < c4) {
                        cv[k] = ((const float4*)col)[m];
#pragma unroll
                        for (int b = 0; b < NB; b++) xv[k][b] = ((const float4*)(x + (size_t)b * xstride))[m];
                    }
                }
#pragma unroll
                for (int k = 0; k < MV_SG; k++) {
                    const int m = m0 + k * T;
                    if (m < c4) {
                        const int i = m * 4;
#pragma unroll
                        for (int b = 0; b < NB; b++) {
                            sh[b * XS + ((i) & 15) * CP + ((i) >> 4)] = xv[k][b].x * cv[k].x;
                            sh[b * XS + ((i + 1) & 15) * CP + ((i + 1) >> 4)] = xv[k][b].y * cv[k].y;
                            sh[b * XS + ((i + 2) & 15) * CP + ((i + 2) >> 4)] = xv[k][b].z * cv[k].z;
                            sh[b * XS + ((i + 3) & 15) * CP + ((i + 3) >> 4)] = xv[k][b].w * cv[k].w;
                        }
                    }
                }
            }
        } else {
            for (int i = tid; i < cols; i += T) {
                const float cv = col[i];
                const int idx = (i & 15) * CP + (i >> 4);
                for (int b = 0; b < NB; b++) sh[b * XS + idx] = x[(size_t)b * xstride + i] * cv;
            }
        }
        __syncthreads();
        const float* rs = (const float*)jb.aux0;
        for (int r = row0 + warp; r < row1; r += NW) {
            const uint4* W = (const uint4*)(jb.w + (size_t)r * cols);
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            // MV_AH words a lane in flight at once: one load at a time left the bus idle (each waited out the
            // memory latency before the next began)
            for (int c0 = lane; c0 < nch; c0 += 32 * MV_AH) {
                uint4 w[MV_AH];
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) w[j] = W[c];
                }
#pragma unroll
                for (int j = 0; j < MV_AH; j++) {
                    const int c = c0 + 32 * j;
                    if (c < nch) {
                        float wf[16];
                        wf[0] = (float)((int)(w[j].x << 24) >> 24);  wf[1] = (float)((int)(w[j].x << 16) >> 24);
                        wf[2] = (float)((int)(w[j].x << 8) >> 24);   wf[3] = (float)((int)w[j].x >> 24);
                        wf[4] = (float)((int)(w[j].y << 24) >> 24);  wf[5] = (float)((int)(w[j].y << 16) >> 24);
                        wf[6] = (float)((int)(w[j].y << 8) >> 24);   wf[7] = (float)((int)w[j].y >> 24);
                        wf[8] = (float)((int)(w[j].z << 24) >> 24);  wf[9] = (float)((int)(w[j].z << 16) >> 24);
                        wf[10] = (float)((int)(w[j].z << 8) >> 24);  wf[11] = (float)((int)w[j].z >> 24);
                        wf[12] = (float)((int)(w[j].w << 24) >> 24); wf[13] = (float)((int)(w[j].w << 16) >> 24);
                        wf[14] = (float)((int)(w[j].w << 8) >> 24);  wf[15] = (float)((int)w[j].w >> 24);
                        for (int b = 0; b < NB; b++) {
                            const float* p = sh + b * XS + c;
                            float a = 0.0f;
                            for (int k = 0; k < 16; k++) a += wf[k] * p[k * CP];
                            acc[b] += a;
                        }
                    }
                }
            }
            const float sc = rs[r];
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a * sc;
            }
        }
    } else {
        for (int i = tid; i < cols; i += T)
            for (int b = 0; b < NB; b++) sh[b * cols + i] = x[(size_t)b * xstride + i];
        __syncthreads();
        for (int r = row0 + warp; r < row1; r += NW) {
            const float* W = (const float*)jb.w + (size_t)r * cols;
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            for (int c = lane; c < cols; c += 32) {
                const float wv = W[c];
                for (int b = 0; b < NB; b++) acc[b] += wv * sh[b * cols + c];
            }
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a;
            }
        }
    }
}

// The 16 bytes of a word of a wt row as floats: eight f16 or four f32.
__device__ __forceinline__ void wt_dec(const uint4 w, int half, float* wf) {
    if (half) {
        wf[0] = h2f((u16)w.x); wf[1] = h2f((u16)(w.x >> 16));
        wf[2] = h2f((u16)w.y); wf[3] = h2f((u16)(w.y >> 16));
        wf[4] = h2f((u16)w.z); wf[5] = h2f((u16)(w.z >> 16));
        wf[6] = h2f((u16)w.w); wf[7] = h2f((u16)(w.w >> 16));
    } else {
        wf[0] = __int_as_float((int)w.x); wf[1] = __int_as_float((int)w.y);
        wf[2] = __int_as_float((int)w.z); wf[3] = __int_as_float((int)w.w);
    }
}
// Its dot with staged values `st` apart.
__device__ __forceinline__ float wt_acc(const float* wf, const float* p, int st, int half) {
    float a = wf[0] * p[0] + wf[1] * p[st] + wf[2] * p[2 * st] + wf[3] * p[3 * st];
    if (half) a += wf[4] * p[4 * st] + wf[5] * p[5 * st] + wf[6] * p[6 * st] + wf[7] * p[7 * st];
    return a;
}

// Words a lane reads ahead of their use: the loads are in flight together,
// where one load a step would wait out the memory latency each time.
#ifndef Q4_AHEAD
#define Q4_AHEAD 8
#endif

// The low-rank middle of a gated residual over NB rows, by column chunks:
// part [chunk][NB][R] = <W[r][chunk], x[b][chunk]>, W [R][Wd] (wt). Grid
// (ceil(R / warps), ceil(Wd / cw)): a block stages its chunk of the NB inputs
// and gives a warp to each of its rows, 16 bytes a lane a load. `q4_hc_up*`
// sums the chunks.
template <int NB>
__device__ __forceinline__ void hc_di_t(const void* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ part, int R, int Wd, int cw, int half,
                                        float* __restrict__ low, float* __restrict__ iw, u32* tick, int lr, int hc)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int c0 = blockIdx.y * cw, cn = min(cw, Wd - c0);
    const int ps = half ? 3 : 2, per = 1 << ps, nl = cn >> ps, CP = nl | 1, XS = per * CP;
    PDL_PRO();
    const int r = blockIdx.x * NW + warp;
    const uint4* W = (const uint4*)((const char*)Wp + ((u64)(r < R ? r : 0) * Wd + c0) * (half ? 2 : 4));
    // this row's first batch of words is requested before the inputs are staged (it does not depend on them)
    uint4 w[Q4_AHEAD];
    if (r < R) {
#pragma unroll
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int c = lane + 32 * k;
            if (c < nl) w[k] = W[c];
        }
    }
    if (((cn | Wd | c0) & 3) == 0) {
        // the inputs four at a time, every load of a thread in flight before its stores
        const int c4 = cn >> 2, tot = NB * c4;
        for (int m0 = tid; m0 < tot; m0 += T * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int m = m0 + k * T;
                if (m < tot) {
                    const int b = m / c4, q = m - b * c4;
                    v[k] = ((const float4*)(x + (u64)b * Wd + c0))[q];
                }
            }
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int m = m0 + k * T;
                if (m < tot) {
                    const int b = m / c4, i = (m - b * c4) * 4;
                    sh[b * XS + ((i) & (per - 1)) * CP + ((i) >> ps)] = v[k].x;
                    sh[b * XS + ((i + 1) & (per - 1)) * CP + ((i + 1) >> ps)] = v[k].y;
                    sh[b * XS + ((i + 2) & (per - 1)) * CP + ((i + 2) >> ps)] = v[k].z;
                    sh[b * XS + ((i + 3) & (per - 1)) * CP + ((i + 3) >> ps)] = v[k].w;
                }
            }
        }
    } else {
        for (int i = tid; i < cn; i += T) {
            const int idx = (i & (per - 1)) * CP + (i >> ps);
            for (int b = 0; b < NB; b++) sh[b * XS + idx] = x[(u64)b * Wd + c0 + i];
        }
    }
    __syncthreads();
    if (r < R) {
    float acc[NB];
    for (int b = 0; b < NB; b++) acc[b] = 0.0f;
    for (int c1 = lane; c1 < nl; c1 += 32 * Q4_AHEAD) {
        if (c1 != lane) {
            for (int k = 0; k < Q4_AHEAD; k++) {
                const int c = c1 + 32 * k;
                if (c < nl) w[k] = W[c];
            }
        }
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int c = c1 + 32 * k;
            if (c < nl) {
                float wf[8];
                wt_dec(w[k], half, wf);
                for (int b = 0; b < NB; b++) acc[b] += wt_acc(wf, sh + b * XS + c, CP, half);
            }
        }
    }
    for (int b = 0; b < NB; b++) {
        float a = acc[b];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) part[((u64)blockIdx.y * NB + b) * R + r] = a;
    }
    }
    // the last chunk block of these rows sums the chunks (in order) and applies the nonlinearity:
    // low [NB][R] = silu(sum / hc) for the first `lr` rows, iw [NB][R - lr] = 2 sigmoid(sum / hc) for the rest
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (int)(atomicAdd(tick + blockIdx.x, 1u) == gridDim.y - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (tid == 0) tick[blockIdx.x] = 0u;
    for (int i = tid; i < NB * NW; i += T) {
        const int b = i / NW, rr = blockIdx.x * NW + (i - b * NW);
        if (rr >= R) continue;
        float pv[32];
#pragma unroll
        for (int ch = 0; ch < 32; ch++) pv[ch] = ch < (int)gridDim.y ? __ldcg(part + ((u64)ch * NB + b) * R + rr) : 0.0f;
        float a = 0.0f;
#pragma unroll
        for (int ch = 0; ch < 32; ch++) if (ch < (int)gridDim.y) a += pv[ch];
        const float v = a / (float)hc;
        if (rr < lr) low[(u64)b * R + rr] = silu_f(v);
        else iw[(u64)b * (R - lr) + (rr - lr)] = 2.0f * sigm(v);
    }
}

// The fold of a gated residual over NB rows, a warp per output dim d. First
// the chunk sums of `q4_hc_di*`: v = sum / hc, the first `lr` rows as
// silu(v) into shared memory, the rest as iw [NB][R - lr] = 2 sigmoid(v)
// (block 0). Then folded[b][d] = sum over streams s of
// sigmoid(<W[s * hid + d], low[b]>) * normed[b][s][d] / hc, W [hc * hid][lr]
// (wt).
template <int NB>
__device__ __forceinline__ void hc_up_t(const void* __restrict__ Wp, const float* __restrict__ part,
                                        float* __restrict__ iw, const float* __restrict__ normed,
                                        float* __restrict__ folded, void* __restrict__ foldedh,
                                        int hid, int hc, int lr, int R, int nch, int half)
{ (void)iw; (void)nch;
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int ps = half ? 3 : 2, per = 1 << ps, nl = lr >> ps, CP = nl | 1, XS = per * CP;
    // `part` is the low vector now: silu(sum / hc), summed by the last block of q4_hc_di*
    for (int i = tid; i < NB * lr; i += T) {
        const int b = i / lr, c = i - b * lr;
        sh[b * XS + (c & (per - 1)) * CP + (c >> ps)] = part[(u64)b * R + c];
    }
    __syncthreads();
    const int d = blockIdx.x * NW + warp;
    if (d >= hid) return;
    const u64 rowb = (u64)lr * (half ? 2 : 4);
    // the words of the hc rows of this d in one sequence: stream s, word it
    const int nit = (nl + 31) >> 5, nq = hc * nit;
    float f[NB], acc[NB];
    for (int b = 0; b < NB; b++) { f[b] = 0.0f; acc[b] = 0.0f; }
    for (int q0 = 0; q0 < nq; q0 += Q4_AHEAD) {
        uint4 w[Q4_AHEAD];
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int q = q0 + k;
            if (q < nq) {
                const int s = q / nit, c = lane + 32 * (q - s * nit);
                if (c < nl) w[k] = ((const uint4*)((const char*)Wp + ((u64)s * hid + d) * rowb))[c];
            }
        }
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int q = q0 + k;
            if (q >= nq) break;
            const int s = q / nit, it = q - s * nit, c = lane + 32 * it;
            if (c < nl) {
                float wf[8];
                wt_dec(w[k], half, wf);
                for (int b = 0; b < NB; b++) acc[b] += wt_acc(wf, sh + b * XS + c, CP, half);
            }
            if (it == nit - 1) {
                for (int b = 0; b < NB; b++) {
                    float a = acc[b];
                    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                    if (lane == 0) f[b] += sigm(a) * normed[((u64)b * hc + s) * hid + d] / (float)hc;
                    acc[b] = 0.0f;
                }
            }
        }
    }
    if (lane == 0) {
        for (int b = 0; b < NB; b++) {
            folded[(u64)b * hid + d] = f[b];
            if (foldedh) stf(foldedh, (u64)b * hid + d, f[b], half);
        }
    }
}

// The fold of a gated residual with four streams, all four at once: lane group s = lane / 8 takes stream s, its
// eight lanes the row's words (nl <= 48), so a d is one load a lane, a three-step sum within the group and a
// two-step sum over the streams (q4_hc_up* walked the streams one after another, each with a full warp sum).
template <int NB>
__device__ __forceinline__ void hc_up4_t(const void* __restrict__ Wp, const float* __restrict__ low,
                                         const float* __restrict__ normed, float* __restrict__ folded,
                                         void* __restrict__ foldedh, int hid, int lr, int R, int half)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int ps = half ? 3 : 2, per = 1 << ps, nl = lr >> ps, CP = nl | 1, XS = per * CP;
    const int s = lane >> 3, l8 = lane & 7;
    const int d = blockIdx.x * NW + warp;
    const bool ok = d < hid;
    const u64 rowb = (u64)lr * (half ? 2 : 4);
    const uint4* W = (const uint4*)((const char*)Wp + ((u64)s * hid + (ok ? d : 0)) * rowb);
    uint4 w[6];
    float nv[NB];
    if (ok) {
#pragma unroll
        for (int k = 0; k < 6; k++)
            if (l8 + 8 * k < nl) w[k] = W[l8 + 8 * k];
        for (int b = 0; b < NB; b++) nv[b] = normed[((u64)b * 4 + s) * hid + d];
    }
    for (int i = tid; i < NB * lr; i += T) {
        const int b = i / lr, c = i - b * lr;
        sh[b * XS + (c & (per - 1)) * CP + (c >> ps)] = low[(u64)b * R + c];
    }
    __syncthreads();
    if (!ok) return;
    float acc[NB];
    for (int b = 0; b < NB; b++) acc[b] = 0.0f;
#pragma unroll
    for (int k = 0; k < 6; k++) {
        if (l8 + 8 * k < nl) {
            float wf[8];
            wt_dec(w[k], half, wf);
            for (int b = 0; b < NB; b++) acc[b] += wt_acc(wf, sh + b * XS + l8 + 8 * k, CP, half);
        }
    }
    for (int b = 0; b < NB; b++) {
        float a = acc[b];
        a += __shfl_xor_sync(0xffffffffu, a, 4);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        float f = sigm(a) * nv[b] / 4.0f;
        f += __shfl_xor_sync(0xffffffffu, f, 8);
        f += __shfl_xor_sync(0xffffffffu, f, 16);
        if (lane == 0) {
            folded[(u64)b * hid + d] = f;
            if (foldedh) stf(foldedh, (u64)b * hid + d, f, half);
        }
    }
}

#define Q4_SMALL(NB) \
    extern "C" __global__ void q4_mvb##NB(MvJob j0, MvJob j1, MvJob j2, MvJob j3, MvJob j4, int njobs, \
                                          const float* __restrict__ x, int xstride, int cols) \
    { mvb_t<NB, false>(j0, j1, j2, j3, j4, njobs, x, xstride, cols); } \
    extern "C" __global__ void q4_mvq##NB(MvJob j0, MvJob j1, MvJob j2, MvJob j3, MvJob j4, int njobs, \
                                          const float* __restrict__ x, int xstride, int cols) \
    { mvb_t<NB, true>(j0, j1, j2, j3, j4, njobs, x, xstride, cols); } \
    extern "C" __global__ void q4_hc_di##NB(const void* __restrict__ Wp, const float* __restrict__ x, \
                                            float* __restrict__ part, int R, int Wd, int cw, int half, \
                                            float* __restrict__ low, float* __restrict__ iw, u32* tick, int lr, int hc) \
    { hc_di_t<NB>(Wp, x, part, R, Wd, cw, half, low, iw, tick, lr, hc); } \
    extern "C" __global__ void q4_hc_up##NB(const void* __restrict__ Wp, const float* __restrict__ part, \
                                            float* __restrict__ iw, const float* __restrict__ normed, \
                                            float* __restrict__ folded, void* __restrict__ foldedh, \
                                            int hid, int hc, int lr, int R, int nch, int half) \
    { PDL_PRO(); hc_up_t<NB>(Wp, part, iw, normed, folded, foldedh, hid, hc, lr, R, nch, half); } \
    extern "C" __global__ void q4_hc_up4_##NB(const void* __restrict__ Wp, const float* __restrict__ low, \
                                              const float* __restrict__ normed, float* __restrict__ folded, \
                                              void* __restrict__ foldedh, int hid, int lr, int R, int half) \
    { PDL_PRO(); hc_up4_t<NB>(Wp, low, normed, folded, foldedh, hid, lr, R, half); }
Q4_SMALL(1) Q4_SMALL(2) Q4_SMALL(3) Q4_SMALL(4) Q4_SMALL(5) Q4_SMALL(6) Q4_SMALL(7) Q4_SMALL(8)

// A unit's expert: a raw slot (gate, up and down at 0, sgu, 2 sgu) or, with bit 63 set, the address of three
// pointers {gate, up, down} (an expert read in place from pinned host memory, whose matrices the file keeps apart).
__device__ __forceinline__ void unit_parts(u64 slot, u64 sgu, u64& g, u64& u, u64& dn) {
    if (slot >> 63) {
        const u64* ds = (const u64*)(slot & 0x7fffffffffffffffull);
        g = ds[0]; u = ds[1]; dn = ds[2];
    } else {
        g = slot; u = slot + sgu; dn = slot + 2 * sgu;
    }
}

// A unit of the expert kernels: one expert's raw slot and up to Q4_UR token
// rows that picked it, as 2 + Q4_UR words (slot lo, hi, then the rows, -1
// from the first free place on).
#define Q4_UR 4
// Words of a matrix row a lane reads ahead: gate and up together, down.
#ifndef EXP_STAGE2
#define EXP_STAGE2 0
#endif
#ifndef EXP_NOW
#define EXP_NOW 0
#endif
#ifndef Q4_GW
#define Q4_GW 3
#endif
#ifndef Q4_DW
#define Q4_DW 4
#endif

// One group of a row against the staged input of a unit row: the nibbles of
// the 16 bytes minus the bias of 8, times the group's scale.
__device__ __forceinline__ float q4_group(const uint4 w, float lc, const float* p, int GP, float gsum) {
    float a = nib8(w.x, p, GP, 0.0f);
    a = nib8(w.y, p + 8 * GP, GP, a);
    a = nib8(w.z, p + 16 * GP, GP, a);
    a = nib8(w.w, p + 24 * GP, GP, a);
    return lc * (a - gsum);
}

// The staged inputs of the expert kernels: XS_CONTIG lays a group's 32 values side by side (a row of XC_G floats,
// 16-byte aligned, so a lane reads them with eight 128-bit loads at fixed offsets; the stride of 36 keeps the
// eight lanes of a phase on distinct banks). The old layout (value k of group g at k * GP + g) cost a load and an
// address multiply per weight. Same values, same order of summation.
#ifndef XS_CONTIG
#define XS_CONTIG 0
#endif
#ifndef STG_N
#define STG_N 8
#endif
#define XC_G 36
__device__ __forceinline__ float q4_group_c(const uint4 w, float lc, const float* p, float gsum) {
    const float4* q = (const float4*)p;
    const u32 ws[4] = {w.x, w.y, w.z, w.w};
    float a = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const float4 A = q[2 * m], B = q[2 * m + 1];
        const u32 x = ws[m];
        a += u2f_small(x & 15u) * A.x;
        a += u2f_small((x >> 4) & 15u) * A.y;
        a += u2f_small((x >> 8) & 15u) * A.z;
        a += u2f_small((x >> 12) & 15u) * A.w;
        a += u2f_small((x >> 16) & 15u) * B.x;
        a += u2f_small((x >> 20) & 15u) * B.y;
        a += u2f_small((x >> 24) & 15u) * B.z;
        a += u2f_small(x >> 28) * B.w;
    }
    return lc * (a - gsum);
}
// The int8 path of the expert kernels (EXP_DP4A, the fast precision): a unit row's inputs are quantized per group of
// 32 to int8 with the group's scale amax/127, packed so that one __dp4a meets four nibbles of a weight word: word m
// of a group pairs (x[8m], x[8m+2], x[8m+4], x[8m+6]) with its low nibbles and (x[8m+1], ...) with its high ones.
// A group is XQ_G ints: E0 O0 E1 O1 E2 O2 E3 O3, the scale, the sum of the 32 codes (the bias of 8 comes off it).
#ifndef EXP_DP4A
#define EXP_DP4A 0
#endif
#define XQ_G 12
// Stage `cnt` rows of n floats (row j at base + row(j) * n) as int8 groups; 8 threads a group.
__device__ __forceinline__ void stage_q8(int* xq, const float* base, int n, const u32* idx, u64 row0, int cnt, int tid, int T)
{
    const int gpr = n >> 5, tot = cnt * gpr * 8;
    for (int t8 = tid; t8 < tot; t8 += T) {
        const unsigned am = __activemask();
        const int gq = t8 >> 3, sub = t8 & 7;
        const int j = gq / gpr, g = gq - j * gpr;
        const u64 row = idx ? (u64)idx[j] : row0 + j;
        const float4 v = ((const float4*)(base + row * n + g * 32))[sub];
        float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        m = fmaxf(m, __shfl_xor_sync(am, m, 1));
        m = fmaxf(m, __shfl_xor_sync(am, m, 2));
        m = fmaxf(m, __shfl_xor_sync(am, m, 4));
        const float sc = m > 0.0f ? m / 127.0f : 1.0f, inv = m > 0.0f ? 127.0f / m : 0.0f;
        const int q0 = __float2int_rn(v.x * inv), q1 = __float2int_rn(v.y * inv);
        const int q2 = __float2int_rn(v.z * inv), q3 = __float2int_rn(v.w * inv);
        int sum = q0 + q1 + q2 + q3;
        sum += __shfl_xor_sync(am, sum, 1);
        sum += __shfl_xor_sync(am, sum, 2);
        sum += __shfl_xor_sync(am, sum, 4);
        const int p0 = __shfl_xor_sync(am, q0, 1), p1 = __shfl_xor_sync(am, q1, 1);
        const int p2 = __shfl_xor_sync(am, q2, 1), p3 = __shfl_xor_sync(am, q3, 1);
        int* dst = xq + (u64)gq * XQ_G;
        if ((sub & 1) == 0) dst[sub] = pack4(q0, q2, p0, p2);
        else dst[sub] = pack4(p1, p3, q1, q3);
        if (sub == 0) { dst[8] = __float_as_int(sc); dst[9] = sum; }
    }
}
// One q4tp group of a row against an int8-staged group: lc * scale * (sum of nibble * code - 8 * sum of codes).
__device__ __forceinline__ float q4_group_q8(const uint4 w, float lc, const int* g) {
    const int4 a = *(const int4*)g, b = *(const int4*)(g + 4);
    const int M = 0x0F0F0F0F;
    int s = 0;
    s = __dp4a((int)(w.x & M), a.x, s);
    s = __dp4a((int)((w.x >> 4) & M), a.y, s);
    s = __dp4a((int)(w.y & M), a.z, s);
    s = __dp4a((int)((w.y >> 4) & M), a.w, s);
    s = __dp4a((int)(w.z & M), b.x, s);
    s = __dp4a((int)((w.z >> 4) & M), b.y, s);
    s = __dp4a((int)(w.w & M), b.z, s);
    s = __dp4a((int)((w.w >> 4) & M), b.w, s);
    return lc * __int_as_float(g[8]) * (float)(s - 8 * g[9]);
}

// Rows of a unit's input in the side-by-side layout: element i of row j at j * XS + (i >> 5) * XC_G + (i & 31).
__device__ __forceinline__ void stage_contig(float* xs, const float* base, int n, const u32* idx, u64 row0, int cnt,
                                             int XS, int tid, int T)
{
    const int n4 = n >> 2, tot = cnt * n4;
    for (int m0 = tid; m0 < tot; m0 += T * STG_N) {
        float4 v[STG_N];
#pragma unroll
        for (int k = 0; k < STG_N; k++) {
            const int m = m0 + k * T;
            if (m < tot) {
                const int j = m / n4;
                const u64 row = idx ? (u64)idx[j] : row0 + j;
                v[k] = ((const float4*)(base + row * n))[m - j * n4];
            }
        }
#pragma unroll
        for (int k = 0; k < STG_N; k++) {
            const int m = m0 + k * T;
            if (m < tot) {
                const int j = m / n4, i = (m - j * n4) * 4;
                *(float4*)(xs + (u64)j * XS + (i >> 5) * XC_G + (i & 31)) = v[k];
            }
        }
    }
}

// One 2-bit group of a row against the staged input of a unit row: the codes (0..3) minus 1.5, times the group's scale.
__device__ __forceinline__ float q2_group(const uint2 w, float lc, const float* p, int GP, float gsum) {
    float a = crumb16(w.x, p, GP, 0.0f);
    a = crumb16(w.y, p + 16 * GP, GP, a);
    return lc * (a - gsum);
}

#ifdef EXP_CLK
__device__ unsigned long long* g_clk;
#endif
// Rows of an expert unit's input into shared memory in the transposed layout xs[row][i & 31][i >> 5] (stride GP):
// float4 loads, STG_N of them in flight per thread before the first store (the old element-at-a-time loop waited
// out the L2 latency on each element and was most of an expert kernel's time).
#ifndef STG_N
#define STG_N 8
#endif
__device__ __forceinline__ void stage_scatter(float* xs, const float* base, int n, const u32* idx, u64 row0, int cnt,
                                              int XS, int GP, int tid, int T)
{
    const int n4 = n >> 2, tot = cnt * n4;
    for (int m0 = tid; m0 < tot; m0 += T * STG_N) {
        float4 v[STG_N];
#pragma unroll
        for (int k = 0; k < STG_N; k++) {
            const int m = m0 + k * T;
            if (m < tot) {
                const int j = m / n4;
                const u64 row = idx ? (u64)idx[j] : row0 + j;
                v[k] = ((const float4*)(base + row * n))[m - j * n4];
            }
        }
#pragma unroll
        for (int k = 0; k < STG_N; k++) {
            const int m = m0 + k * T;
            if (m < tot) {
                const int j = m / n4, i = (m - j * n4) * 4;
                float* xj = xs + j * XS;
                xj[(i & 31) * GP + (i >> 5)] = v[k].x;
                xj[((i + 1) & 31) * GP + ((i + 1) >> 5)] = v[k].y;
                xj[((i + 2) & 31) * GP + ((i + 2) >> 5)] = v[k].z;
                xj[((i + 3) & 31) * GP + ((i + 3) >> 5)] = v[k].w;
            }
        }
    }
}

// Gate and up of the round's units, straight from the raw slots:
// act [U * Q4_UR][inter] = silu(gate x) * (up x) for each row of a unit,
// folded [n][hid]. Grid (ceil(inter / rb), U): a block does `rb` rows of both
// matrices, its warps a row at a time; the rows' scale ladders sit in shared
// memory. A weight is read once for all the rows of the unit.
extern "C" __global__ void q4_exp_gu(const u32* __restrict__ units, const float* __restrict__ folded,
                                     float* __restrict__ act, int inter, int hid, int rb)
{ PDL_PRO();
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = hid >> 5, GP = gpr | 1, XS = 32 * GP, stride = (gpr * 5 + 7) / 8;
    const u64 params_off = (u64)inter * gpr * 16, codes_off = params_off + (u64)inter * 4;
    const u64 sgu = codes_off + (u64)inter * stride;
    u64 pg, pu, pd;
    unit_parts(slot, sgu, pg, pu, pd);
    (void)pd;
    float* xs = sh;                 // [cnt][32][GP]
    float* gs = xs + cnt * XS;      // [cnt][gpr]
    float* lad = gs + cnt * gpr;    // [2][rb][32]
    const int r0 = blockIdx.x * rb;
#if EXP_DP4A
    stage_q8((int*)xs, folded, hid, un + 2, 0, cnt, tid, T);
#elif !defined(EXP_NOSTAGE)
    stage_scatter(xs, folded, hid, un + 2, 0, cnt, XS, GP, tid, T);
#if EXP_STAGE2
    __syncthreads();
    stage_scatter(xs, folded, hid, un + 2, 0, cnt, XS, GP, tid, T);
#endif
#endif
    if (tid < 2 * rb) {
        const int m = tid / rb, r = r0 + tid - m * rb;
        if (r < inter) q4_ladder((const u16*)((m ? pu : pg) + params_off + (u64)r * 4), lad + tid * 32);
    }
    __syncthreads();
#if !defined(EXP_NOSTAGE) && !EXP_DP4A
    for (int i = tid; i < cnt * gpr; i += T) {
        const int j = i / gpr, g = i - j * gpr;
        const float* p = xs + j * XS + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
        gs[i] = 8.0f * s;
    }
#endif
    __syncthreads();
    // the words of this warp's rows in one sequence: row q, word it
    const int nit = (gpr + 31) >> 5;
    const int nz = warp < rb ? ((rb - warp + NW - 1) / NW) * nit : 0;
    const u8* mg = (const u8*)pg;
    const u8* mu = (const u8*)pu;
    float ag[Q4_UR], au[Q4_UR];
    for (int j = 0; j < Q4_UR; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
    for (int z0 = 0; z0 < nz; z0 += Q4_GW) {
        uint4 wg[Q4_GW], wu[Q4_GW];
        float lg[Q4_GW], lu[Q4_GW];
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z < nz) {
                const int q = z / nit, g = lane + 32 * (z - q * nit), i = warp + q * NW, r = r0 + i;
                if (r < inter && g < gpr && !EXP_NOW) {
                    wg[k] = ((const uint4*)(mg + (u64)r * gpr * 16))[g];
                    wu[k] = ((const uint4*)(mu + (u64)r * gpr * 16))[g];
                    lg[k] = lad[i * 32 + q4_code(mg + codes_off + (u64)r * stride, g, stride)];
                    lu[k] = lad[(rb + i) * 32 + q4_code(mu + codes_off + (u64)r * stride, g, stride)];
                }
            }
        }
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z >= nz) break;
            const int q = z / nit, it = z - q * nit, g = lane + 32 * it, i = warp + q * NW, r = r0 + i;
            if (r >= inter) break;
            if (g < gpr) {
                for (int j = 0; j < cnt; j++) {
#if EXP_DP4A
                    const int* qg = (const int*)xs + ((u64)j * gpr + g) * XQ_G;
                    ag[j] += q4_group_q8(wg[k], lg[k], qg);
                    au[j] += q4_group_q8(wu[k], lu[k], qg);
#else
                    const float* p = xs + j * XS + g;
                    const float gsum = gs[j * gpr + g];
                    ag[j] += q4_group(wg[k], lg[k], p, GP, gsum);
                    au[j] += q4_group(wu[k], lu[k], p, GP, gsum);
#endif
                }
            }
            if (it == nit - 1) {
                for (int j = 0; j < cnt; j++) {
                    float a = ag[j], b = au[j];
                    for (int o = 16; o > 0; o >>= 1) {
                        a += __shfl_xor_sync(0xffffffffu, a, o);
                        b += __shfl_xor_sync(0xffffffffu, b, o);
                    }
                    if (lane == 0) act[((u64)blockIdx.y * Q4_UR + j) * inter + r] = silu_f(a) * b;
                    ag[j] = 0.0f;
                    au[j] = 0.0f;
                }
            }
        }
    }
}

// Gate and up of q2tp experts (2-bit codes, rung 0 = zero): the same as q4_exp_gu with 8-byte groups.
extern "C" __global__ void q4_exp_gu2(const u32* __restrict__ units, const float* __restrict__ folded,
                                     float* __restrict__ act, int inter, int hid, int rb)
{ PDL_PRO();
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = hid >> 5, GP = gpr | 1, XS = 32 * GP, stride = (gpr * 5 + 7) / 8;
    const u64 params_off = (u64)inter * gpr * 8, codes_off = params_off + (u64)inter * 4;
    const u64 sgu = codes_off + (u64)inter * stride;
    u64 pg, pu, pd;
    unit_parts(slot, sgu, pg, pu, pd);
    (void)pd;
    float* xs = sh;                 // [cnt][32][GP]
    float* gs = xs + cnt * XS;      // [cnt][gpr]
    float* lad = gs + cnt * gpr;    // [2][rb][32]
    const int r0 = blockIdx.x * rb;
    stage_scatter(xs, folded, hid, un + 2, 0, cnt, XS, GP, tid, T);
#if EXP_STAGE2
    __syncthreads();
    stage_scatter(xs, folded, hid, un + 2, 0, cnt, XS, GP, tid, T);
#endif
    if (tid < 2 * rb) {
        const int m = tid / rb, r = r0 + tid - m * rb;
        if (r < inter) q2_ladder((const u16*)((m ? pu : pg) + params_off + (u64)r * 4), lad + tid * 32);
    }
    __syncthreads();
    for (int i = tid; i < cnt * gpr; i += T) {
        const int j = i / gpr, g = i - j * gpr;
        const float* p = xs + j * XS + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
        gs[i] = 1.5f * s;
    }
    __syncthreads();
    // the words of this warp's rows in one sequence: row q, word it
    const int nit = (gpr + 31) >> 5;
    const int nz = warp < rb ? ((rb - warp + NW - 1) / NW) * nit : 0;
    const u8* mg = (const u8*)pg;
    const u8* mu = (const u8*)pu;
    float ag[Q4_UR], au[Q4_UR];
    for (int j = 0; j < Q4_UR; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
    for (int z0 = 0; z0 < nz; z0 += Q4_GW) {
        uint2 wg[Q4_GW], wu[Q4_GW];
        float lg[Q4_GW], lu[Q4_GW];
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z < nz) {
                const int q = z / nit, g = lane + 32 * (z - q * nit), i = warp + q * NW, r = r0 + i;
                if (r < inter && g < gpr) {
                    wg[k] = ((const uint2*)(mg + (u64)r * gpr * 8))[g];
                    wu[k] = ((const uint2*)(mu + (u64)r * gpr * 8))[g];
                    lg[k] = lad[i * 32 + q4_code(mg + codes_off + (u64)r * stride, g, stride)];
                    lu[k] = lad[(rb + i) * 32 + q4_code(mu + codes_off + (u64)r * stride, g, stride)];
                }
            }
        }
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z >= nz) break;
            const int q = z / nit, it = z - q * nit, g = lane + 32 * it, i = warp + q * NW, r = r0 + i;
            if (r >= inter) break;
            if (g < gpr) {
                for (int j = 0; j < cnt; j++) {
                    const float* p = xs + j * XS + g;
                    const float gsum = gs[j * gpr + g];
                    ag[j] += q2_group(wg[k], lg[k], p, GP, gsum);
                    au[j] += q2_group(wu[k], lu[k], p, GP, gsum);
                }
            }
            if (it == nit - 1) {
                for (int j = 0; j < cnt; j++) {
                    float a = ag[j], b = au[j];
                    for (int o = 16; o > 0; o >>= 1) {
                        a += __shfl_xor_sync(0xffffffffu, a, o);
                        b += __shfl_xor_sync(0xffffffffu, b, o);
                    }
                    if (lane == 0) act[((u64)blockIdx.y * Q4_UR + j) * inter + r] = silu_f(a) * b;
                    ag[j] = 0.0f;
                    au[j] = 0.0f;
                }
            }
        }
    }
}

// Down of the round's units: part [U * Q4_UR][hid] = down act for each row of
// a unit. Grid (ceil(hid / rb), U): a block does `rb` output rows of one
// unit, its warps a row at a time.
extern "C" __global__ void q4_exp_d(const u32* __restrict__ units, const float* __restrict__ act,
                                    float* __restrict__ part, int inter, int hid, int rb, unsigned long long sgu)
{ PDL_PRO();
#ifdef EXP_CLK
    u64 tk[5];
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tk[0]));
#endif
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = inter >> 5, GP = gpr | 1, XS = XS_CONTIG ? gpr * XC_G : 32 * GP, stride = (gpr * 5 + 7) / 8;
    const u64 params_off = (u64)hid * gpr * 16, codes_off = params_off + (u64)hid * 4;
    u64 pg, pu, pd;
    unit_parts(slot, sgu, pg, pu, pd);
    const u8* mat = (const u8*)pd;
    float* xs = sh;                 // [cnt][32][GP]
    float* gs = xs + cnt * XS;      // [cnt][gpr]
    float* lad = gs + cnt * gpr;    // [rb][32]
    const int r0 = blockIdx.x * rb;
#if EXP_DP4A
    stage_q8((int*)xs, act, inter, nullptr, (u64)blockIdx.y * Q4_UR, cnt, tid, T);
#elif !defined(EXP_NOSTAGE)
#if XS_CONTIG
    stage_contig(xs, act, inter, nullptr, (u64)blockIdx.y * Q4_UR, cnt, XS, tid, T);
#else
    stage_scatter(xs, act, inter, nullptr, (u64)blockIdx.y * Q4_UR, cnt, XS, GP, tid, T);
#endif
#if EXP_STAGE2
    __syncthreads();
    stage_scatter(xs, act, inter, nullptr, (u64)blockIdx.y * Q4_UR, cnt, XS, GP, tid, T);
#endif
#endif
    if (tid < rb && r0 + tid < hid) q4_ladder((const u16*)(mat + params_off + (u64)(r0 + tid) * 4), lad + tid * 32);
    __syncthreads();
#ifdef EXP_CLK
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tk[1]));
#endif
#if !defined(EXP_NOSTAGE) && !EXP_DP4A
    for (int i = tid; i < cnt * gpr; i += T) {
        const int j = i / gpr, g = i - j * gpr;
#if XS_CONTIG
        const float* p = xs + j * XS + g * XC_G;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k];
#else
        const float* p = xs + j * XS + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
#endif
        gs[i] = 8.0f * s;
    }
#endif
    __syncthreads();
#ifdef EXP_CLK
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tk[2]));
#endif
    const int nit = (gpr + 31) >> 5;
    const int nz = warp < rb ? ((rb - warp + NW - 1) / NW) * nit : 0;
    float acc[Q4_UR];
    for (int j = 0; j < Q4_UR; j++) acc[j] = 0.0f;
    for (int z0 = 0; z0 < nz; z0 += Q4_DW) {
        uint4 w[Q4_DW];
        float lc[Q4_DW];
        for (int k = 0; k < Q4_DW; k++) {
            const int z = z0 + k;
            if (z < nz) {
                const int q = z / nit, g = lane + 32 * (z - q * nit), i = warp + q * NW, r = r0 + i;
                if (r < hid && g < gpr && !EXP_NOW) {
                    w[k] = ((const uint4*)(mat + (u64)r * gpr * 16))[g];
                    lc[k] = lad[i * 32 + q4_code(mat + codes_off + (u64)r * stride, g, stride)];
                }
            }
        }
        for (int k = 0; k < Q4_DW; k++) {
            const int z = z0 + k;
            if (z >= nz) break;
            const int q = z / nit, it = z - q * nit, g = lane + 32 * it, i = warp + q * NW, r = r0 + i;
            if (r >= hid) break;
            if (g < gpr) {
#ifdef EXP_NOMATH
                for (int j = 0; j < cnt; j++) acc[j] += lc[k] * __int_as_float((int)((w[k].x ^ w[k].y ^ w[k].z ^ w[k].w) & 0x3f7fffffu));
#elif EXP_DP4A
                for (int j = 0; j < cnt; j++) acc[j] += q4_group_q8(w[k], lc[k], (const int*)xs + ((u64)j * gpr + g) * XQ_G);
#elif XS_CONTIG
                for (int j = 0; j < cnt; j++) acc[j] += q4_group_c(w[k], lc[k], xs + j * XS + g * XC_G, gs[j * gpr + g]);
#else
                for (int j = 0; j < cnt; j++) acc[j] += q4_group(w[k], lc[k], xs + j * XS + g, GP, gs[j * gpr + g]);
#endif
            }
            if (it == nit - 1) {
                for (int j = 0; j < cnt; j++) {
                    float a = acc[j];
                    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                    if (lane == 0) part[((u64)blockIdx.y * Q4_UR + j) * hid + r] = a;
                    acc[j] = 0.0f;
                }
            }
        }
    }
#ifdef EXP_CLK
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tk[3]));
    __syncthreads();
    if (tid == 0 && g_clk) {
        u64* o = g_clk + ((u64)blockIdx.y * gridDim.x + blockIdx.x) * 4;
        o[0] = tk[0]; o[1] = tk[1]; o[2] = tk[2]; o[3] = tk[3];
    }
#endif
}


// The down projection of the int8 path with four lanes a row: lane (rr, sub) takes groups sub * GPL .. of row
// rr of its warp's eight, so all 32 lanes work (one group a lane left 12 of 32 idle with 640-wide rows) and a lane
// has GPL words in flight. `rb` rows a block (a multiple of 8 * warps for full use).
extern "C" __global__ void q4_exp_d4(const u32* __restrict__ units, const float* __restrict__ act,
                                     float* __restrict__ part, int inter, int hid, int rb, unsigned long long sgu)
{ PDL_PRO();
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = inter >> 5, stride = (gpr * 5 + 7) / 8;
    const u64 params_off = (u64)hid * gpr * 16, codes_off = params_off + (u64)hid * 4;
    u64 pg, pu, pd;
    unit_parts(slot, sgu, pg, pu, pd);
    const u8* mat = (const u8*)pd;
    int* xq = (int*)sh;                                  // [cnt][gpr][XQ_G]
    float* lad = sh + Q4_UR * gpr * XQ_G;                // [rb][32]
    const int r0 = blockIdx.x * rb;
    stage_q8(xq, act, inter, nullptr, (u64)blockIdx.y * Q4_UR, cnt, tid, T);
    for (int i = tid; i < rb; i += T)
        if (r0 + i < hid) q4_ladder((const u16*)(mat + params_off + (u64)(r0 + i) * 4), lad + i * 32);
    __syncthreads();
    const int rr = lane >> 2, sub = lane & 3, GPL = (gpr + 3) >> 2;
    for (int base = warp * 8; base < rb; base += NW * 8) {
        const int i = base + rr, r = r0 + i;
        const bool ok = i < rb && r < hid;
        uint4 w[8];
        float lc[8];
        if (ok) {
            const u8* codes = mat + codes_off + (u64)r * stride;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const int g = sub * GPL + q;
                if (q < GPL && g < gpr) {
                    w[q] = ((const uint4*)(mat + (u64)r * gpr * 16))[g];
                    lc[q] = lad[i * 32 + q4_code(codes, g, stride)];
                }
            }
        }
        float acc[Q4_UR];
        for (int j = 0; j < Q4_UR; j++) acc[j] = 0.0f;
        if (ok) {
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const int g = sub * GPL + q;
                if (q < GPL && g < gpr)
                    for (int j = 0; j < cnt; j++) acc[j] += q4_group_q8(w[q], lc[q], xq + ((u64)j * gpr + g) * XQ_G);
            }
        }
        for (int j = 0; j < cnt; j++) {
            float a = acc[j];
            a += __shfl_xor_sync(0xffffffffu, a, 1);
            a += __shfl_xor_sync(0xffffffffu, a, 2);
            if (ok && sub == 0) part[((u64)blockIdx.y * Q4_UR + j) * hid + r] = a;
        }
    }
}

// The weighted sum of a token's picks: blk [n][hid] = sum over its K places
// of w_k * part[ref[t][k]] (ref < 0: no pick), with w_k = wts[t][k] for a
// routed pick and, for place `topk` (the shared expert),
// sigmoid(logits[t][gcol]) or 1 (gcol < 0).
extern "C" __global__ void q4_exp_sum(const int* __restrict__ ref, const float* __restrict__ part,
                                      const float* __restrict__ wts, const float* __restrict__ logits,
                                      float* __restrict__ blk, int n, int K, int topk, int hid, int R, int gcol)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, r = i - t * hid;
    float a = 0.0f;
    for (int k = 0; k < K; k++) {
        const int p = ref[t * K + k];
        if (p < 0) continue;
        const float wk = k < topk ? wts[t * topk + k] : (gcol < 0 ? 1.0f : sigm(logits[t * R + gcol]));
        a += wk * part[(u64)p * hid + r];
    }
    blk[i] = a;
}

// The index of the largest value of each row [vocab] (a block per row), ties to the lower index.
extern "C" __global__ void q4_argmax(const float* __restrict__ rows, int* __restrict__ ids, int vocab)
{ PDL_PRO();
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* l = rows + (u64)blockIdx.x * vocab;
    float best = -__int_as_float(0x7f800000);
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = l[i];
        if (v > best) { best = v; at = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = at;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            const float v = bv[threadIdx.x + s];
            const int j = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x])) {
                bv[threadIdx.x] = v;
                bi[threadIdx.x] = j;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[blockIdx.x] = bi[0] == 0x7fffffff ? 0 : bi[0];
}

// The MTP draft head's input: hyper [hc][hid] = h [hc][hid] + e [hid] (the
// embedding's projection on every stream).
extern "C" __global__ void q4_mtp_fuse(const float* __restrict__ hs, const float* __restrict__ e,
                                       float* __restrict__ hyper, int hid, int hc)
{ PDL_PRO();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hid * hc) return;
    hyper[i] = hs[i] + e[i % hid];
}

// The lean path's units, built on the card from the route's picks, packed: one block
// counts the rows each expert in the table `tbl` (0: not in the arena, the pick is
// dropped) was picked by, scans the counts into the first unit of each expert, and
// writes the units (four rows to a unit, the expert's raw slot first) and, for every
// (token, place), the row of the partial outputs it takes (`refs [n][K]`, -1 for none).
// The shared expert's units (`nsh`, its slot `shared`) follow the experts'. The `umax`
// units of the launch that are not used stay empty (slot 0): the kernels skip them.
extern "C" __global__ void q4_units(const int* __restrict__ picks, const unsigned long long* __restrict__ tbl,
                                    unsigned long long shared, u32* __restrict__ units, int* __restrict__ refs,
                                    int n, int topk, int K, int ne, int umax, int nsh)
{ PDL_PRO();
    __shared__ int sp[8 * 32];
    __shared__ int cnt[512];
    __shared__ int off[512];
    __shared__ int scan[256];
    const int np = n * topk, W = 2 + Q4_UR, T = blockDim.x, tid = threadIdx.x;
    for (int i = tid; i < np; i += T) sp[i] = picks[i];
    for (int i = tid; i < umax * W; i += T) units[i] = (i % W) >= 2 ? 0xffffffffu : 0u;
    __syncthreads();
    for (int e = tid; e < ne; e += T) {
        int cn = 0;
        if (tbl[e]) for (int i = 0; i < np; i++) cn += (sp[i] == e);
        cnt[e] = cn;
    }
    __syncthreads();
    const int e0 = 2 * tid, e1 = e0 + 1;
    const int u0 = e0 < ne ? (cnt[e0] + Q4_UR - 1) / Q4_UR : 0;
    const int u1 = e1 < ne ? (cnt[e1] + Q4_UR - 1) / Q4_UR : 0;
    scan[tid] = u0 + u1;
    __syncthreads();
    for (int d = 1; d < T; d <<= 1) {
        const int x = tid >= d ? scan[tid - d] : 0;
        __syncthreads();
        scan[tid] += x;
        __syncthreads();
    }
    const int base = scan[tid] - (u0 + u1);
    if (e0 < ne) off[e0] = base;
    if (e1 < ne) off[e1] = base + u0;
    const int total = scan[T - 1];
    __syncthreads();
    for (int e = tid; e < ne; e += T) {
        if (!cnt[e]) continue;
        const unsigned long long a = tbl[e];
        int rank = 0;
        for (int i = 0; i < np; i++) {
            if (sp[i] != e) continue;
            const int u = off[e] + rank / Q4_UR, j = rank % Q4_UR;
            u32* w = units + (u64)u * W;
            w[0] = (u32)a; w[1] = (u32)(a >> 32); w[2 + j] = (u32)(i / topk);
            refs[(i / topk) * K + (i % topk)] = u * Q4_UR + j;
            rank++;
        }
    }
    for (int i = tid; i < np; i += T) {
        const int e = sp[i];
        if (e < 0 || e >= ne || !tbl[e]) refs[(i / topk) * K + (i % topk)] = -1;
    }
    for (int g = tid; g < nsh; g += T) {
        u32* w = units + (u64)(total + g) * W;
        w[0] = (u32)shared; w[1] = (u32)(shared >> 32);
        for (int j = 0; j < Q4_UR; j++) {
            const int t = g * Q4_UR + j;
            if (t < n) { w[2 + j] = (u32)t; refs[t * K + topk] = (total + g) * Q4_UR + j; }
        }
    }
}

// ---------------------------------------------------------------- 8-bit tubes
// A hot expert of a hot file is one q8_2f tube (gate, up, down, int8 body, row scale, column
// field), resident on the card. A unit names its tube by the address of a descriptor of nine
// words: for gate, up and down the body, the row scales and the column field.

// Sixteen int8 weights of a 16-byte word as floats.
__device__ __forceinline__ void dec_i8x16(const uint4 w, float* wf) {
    wf[0] = (float)((int)(w.x << 24) >> 24);  wf[1] = (float)((int)(w.x << 16) >> 24);
    wf[2] = (float)((int)(w.x << 8) >> 24);   wf[3] = (float)((int)w.x >> 24);
    wf[4] = (float)((int)(w.y << 24) >> 24);  wf[5] = (float)((int)(w.y << 16) >> 24);
    wf[6] = (float)((int)(w.y << 8) >> 24);   wf[7] = (float)((int)w.y >> 24);
    wf[8] = (float)((int)(w.z << 24) >> 24);  wf[9] = (float)((int)(w.z << 16) >> 24);
    wf[10] = (float)((int)(w.z << 8) >> 24);  wf[11] = (float)((int)w.z >> 24);
    wf[12] = (float)((int)(w.w << 24) >> 24); wf[13] = (float)((int)(w.w << 16) >> 24);
    wf[14] = (float)((int)(w.w << 8) >> 24);  wf[15] = (float)((int)w.w >> 24);
}

// act [(ub + unit) * Q4_UR + j][inter] = silu(gate x) * (up x) for the rows of the tube units.
// Grid (ceil(inter / rb), tube units); a warp does a row of both matrices.
extern "C" __global__ void q4_tube_gu(const u32* __restrict__ units, const float* __restrict__ folded,
                                      float* __restrict__ act, int inter, int hid, int rb, int ub)
{ PDL_PRO();
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64* dsc = (const u64*)((u64)un[0] | ((u64)un[1] << 32));
    if (!dsc) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const signed char* gw = (const signed char*)dsc[0];
    const float* grs = (const float*)dsc[1];
    const float* gcol = (const float*)dsc[2];
    const signed char* uw = (const signed char*)dsc[3];
    const float* urs = (const float*)dsc[4];
    const float* ucol = (const float*)dsc[5];
    const int nch = hid >> 4, CP = nch | 1, XS = 16 * CP;
    float* xg = sh;
    float* xu = sh + cnt * XS;
    for (int j = 0; j < cnt; j++) {
        const float* x = folded + (u64)un[2 + j] * hid;
        for (int i = tid; i < hid; i += T) {
            const int idx = (i & 15) * CP + (i >> 4);
            xg[j * XS + idx] = x[i] * gcol[i];
            xu[j * XS + idx] = x[i] * ucol[i];
        }
    }
    __syncthreads();
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, inter);
    for (int r = r0 + warp; r < r1; r += NW) {
        const uint4* Wg = (const uint4*)(gw + (u64)r * hid);
        const uint4* Wu = (const uint4*)(uw + (u64)r * hid);
        float ag[Q4_UR], au[Q4_UR];
        for (int j = 0; j < Q4_UR; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
        for (int c = lane; c < nch; c += 32) {
            float wa[16], wb[16];
            dec_i8x16(Wg[c], wa);
            dec_i8x16(Wu[c], wb);
            for (int j = 0; j < cnt; j++) {
                const float* pg = xg + j * XS + c;
                const float* pu = xu + j * XS + c;
                float sg = 0.0f, su = 0.0f;
                for (int k = 0; k < 16; k++) { sg += wa[k] * pg[k * CP]; su += wb[k] * pu[k * CP]; }
                ag[j] += sg;
                au[j] += su;
            }
        }
        for (int j = 0; j < cnt; j++) {
            float a = ag[j], b = au[j];
            for (int o = 16; o > 0; o >>= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, o);
                b += __shfl_xor_sync(0xffffffffu, b, o);
            }
            if (lane == 0) act[((u64)(ub + blockIdx.y) * Q4_UR + j) * inter + r] = silu_f(a * grs[r]) * (b * urs[r]);
        }
    }
}

// part [(ub + unit) * Q4_UR + j][hid] = down act for the rows of the tube units. Grid (ceil(hid / rb), tube units).
extern "C" __global__ void q4_tube_d(const u32* __restrict__ units, const float* __restrict__ act,
                                     float* __restrict__ part, int inter, int hid, int rb, int ub)
{ PDL_PRO();
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64* dsc = (const u64*)((u64)un[0] | ((u64)un[1] << 32));
    if (!dsc) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const signed char* dw = (const signed char*)dsc[6];
    const float* drs = (const float*)dsc[7];
    const float* dcol = (const float*)dsc[8];
    const int nch = inter >> 4, CP = nch | 1, XS = 16 * CP;
    for (int j = 0; j < cnt; j++) {
        const float* a = act + ((u64)(ub + blockIdx.y) * Q4_UR + j) * inter;
        for (int i = tid; i < inter; i += T) sh[j * XS + (i & 15) * CP + (i >> 4)] = a[i] * dcol[i];
    }
    __syncthreads();
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, hid);
    for (int r = r0 + warp; r < r1; r += NW) {
        const uint4* W = (const uint4*)(dw + (u64)r * inter);
        float acc[Q4_UR];
        for (int j = 0; j < Q4_UR; j++) acc[j] = 0.0f;
        for (int c = lane; c < nch; c += 32) {
            float wf[16];
            dec_i8x16(W[c], wf);
            for (int j = 0; j < cnt; j++) {
                const float* p = sh + j * XS + c;
                float a = 0.0f;
                for (int k = 0; k < 16; k++) a += wf[k] * p[k * CP];
                acc[j] += a;
            }
        }
        for (int j = 0; j < cnt; j++) {
            float a = acc[j];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) part[((u64)(ub + blockIdx.y) * Q4_UR + j) * hid + r] = a * drs[r];
        }
    }
}

// ---------------------------------------------------------------- latent NVFP4 experts (Edge0)
// A latent layer's routed experts work in an `L`-wide copy of the token: expert e of group
// labels[e] reads row (t * G + g) of xl [n * G][L] (D_g x) and its weighted output is summed into
// row (t * G + g) of acc [n * G][L], which U brings back to the hidden width. An expert slot holds
// gate [ff][L], up [ff][L], down [L][ff], each an `nvfp4` payload: E2M1 codes (rows * cols / 2,
// element 2j in the low nibble of byte j), UE4M3 scales (one per 16 elements), an f32 global.

// E2M1 code -> float, from the bits (sign bit 3, exponent bits 2..1, mantissa bit 0).
__device__ __forceinline__ float lat_e2m1(u32 c) {
    const u32 e = (c >> 1) & 3u, m = c & 1u;
    const u32 mag = e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
    return __int_as_float((int)(mag | ((c & 8u) << 28)));
}
// UE4M3 byte -> float (ggml's contract: bit 7 ignored, 0x7F reads as zero).
__device__ __forceinline__ float lat_ue4m3(u32 b) {
    b &= 0x7Fu;
    if (b == 0u || b == 0x7Fu) return 0.0f;
    const u32 e = (b >> 3) & 15u, m = b & 7u;
    return e ? __int_as_float((int)(((e + 120u) << 23) | (m << 20))) : (float)m * 0.001953125f;
}
// Bytes a payload of rows x cols takes in a slot: the payload, rounded up to 16 so every matrix's
// codes start 16-byte aligned.
__device__ __forceinline__ u64 lat_pbytes(int rows, int cols) {
    return ((u64)rows * cols / 2 + (u64)rows * cols / 16 + 4 + 15) & ~(u64)15;
}
// The f32 global of a payload.
__device__ __forceinline__ float lat_glob(const u8* p, int rows, int cols) {
    return *(const float*)(p + (u64)rows * cols / 2 + (u64)rows * cols / 16);
}
// Sixteen elements (one scale group) of a row against sixteen staged values.
__device__ __forceinline__ float lat_dot16(const u8* codes, u32 sc, const float* x) {
    const uint2 w = *(const uint2*)codes;
    float a = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const u32 by = (w.x >> (8 * k)) & 0xFFu;
        a += lat_e2m1(by & 15u) * x[2 * k] + lat_e2m1(by >> 4) * x[2 * k + 1];
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const u32 by = (w.y >> (8 * k)) & 0xFFu;
        a += lat_e2m1(by & 15u) * x[8 + 2 * k] + lat_e2m1(by >> 4) * x[8 + 2 * k + 1];
    }
    return a * lat_ue4m3(sc);
}

// Gate and up of every pick: act [n * topk][ff] = silu(sg * (G x)) * (su * (U x)), x the pick's latent
// row. Grid (ff / rb, n * topk), 256 threads; the row of x is staged once a block.
extern "C" __global__ void q4_lat_gu(const int* __restrict__ picks, const int* __restrict__ labels,
                                     const u64* __restrict__ tbl, const float* __restrict__ xl,
                                     float* __restrict__ act, int ff, int L, int G, int topk, int rb)
{ PDL_PRO();
    extern __shared__ float lx[];
    const int r = blockIdx.y, e = picks[r];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, NW = blockDim.x >> 5;
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, ff);
    if (e < 0) {
        for (int i = r0 + threadIdx.x; i < r1; i += blockDim.x) act[(u64)r * ff + i] = 0.0f;
        return;
    }
    // an expert outside the card's bank: the host computes its rows (`q4_lat_put` lays them in)
    const u8* gp = (const u8*)tbl[e];
    if (!gp) return;
    const float* x = xl + ((u64)(r / topk) * G + labels[e]) * L;
    for (int i = threadIdx.x; i < L; i += blockDim.x) lx[i] = x[i];
    __syncthreads();
    const u64 pb = lat_pbytes(ff, L);
    const u8* up = gp + pb;
    const int gpr = L >> 4;
    const float sg = lat_glob(gp, ff, L), su = lat_glob(up, ff, L);
    const u8* gsc = gp + (u64)ff * L / 2;
    const u8* usc = up + (u64)ff * L / 2;
    for (int row = r0 + warp; row < r1; row += NW) {
        float ag = 0.0f, au = 0.0f;
        for (int k = lane; k < gpr; k += 32) {
            const u64 o = (u64)row * gpr + k;
            ag += lat_dot16(gp + o * 8, gsc[o], lx + k * 16);
            au += lat_dot16(up + o * 8, usc[o], lx + k * 16);
        }
        for (int s = 16; s > 0; s >>= 1) {
            ag += __shfl_xor_sync(0xffffffffu, ag, s);
            au += __shfl_xor_sync(0xffffffffu, au, s);
        }
        if (lane == 0) act[(u64)r * ff + row] = silu_f(sg * ag) * (su * au);
    }
}

// Down of every pick: y [n * topk][L] = sd * (D act). Grid (L / rb, n * topk), 256 threads.
extern "C" __global__ void q4_lat_dn(const int* __restrict__ picks, const u64* __restrict__ tbl,
                                     const float* __restrict__ act, float* __restrict__ y, int ff, int L, int rb)
{ PDL_PRO();
    extern __shared__ float la[];
    const int r = blockIdx.y, e = picks[r];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, NW = blockDim.x >> 5;
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, L);
    if (e < 0) {
        for (int i = r0 + threadIdx.x; i < r1; i += blockDim.x) y[(u64)r * L + i] = 0.0f;
        return;
    }
    if (!tbl[e]) return;
    for (int i = threadIdx.x; i < ff; i += blockDim.x) la[i] = act[(u64)r * ff + i];
    __syncthreads();
    const u8* dp = (const u8*)tbl[e] + 2 * lat_pbytes(ff, L);
    const int gpr = ff >> 4;
    const float sd = lat_glob(dp, L, ff);
    const u8* dsc = dp + (u64)L * ff / 2;
    for (int row = r0 + warp; row < r1; row += NW) {
        float a = 0.0f;
        for (int k = lane; k < gpr; k += 32) {
            const u64 o = (u64)row * gpr + k;
            a += lat_dot16(dp + o * 8, dsc[o], la + k * 16);
        }
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0) y[(u64)r * L + row] = sd * a;
    }
}

#ifndef LAT_RPW
#define LAT_RPW 4
#endif
// 32 elements (two scale groups: codes w, scales s2 low and high byte) against the staged x chunk xv.
__device__ __forceinline__ float lat_dot32(uint4 w, u32 s2, const float* xv) {
    float a0 = 0.0f, a1 = 0.0f;
    const u32 ws[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float a = 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const u32 by = (ws[k] >> (8 * b)) & 0xFFu;
            a += lat_e2m1(by & 15u) * xv[8 * k + 2 * b] + lat_e2m1(by >> 4) * xv[8 * k + 2 * b + 1];
        }
        if (k < 2) a0 += a; else a1 += a;
    }
    return a0 * lat_ue4m3(s2 & 0xFFu) + a1 * lat_ue4m3((s2 >> 8) & 0xFFu);
}

// `q4_lat_gu` with LAT_RPW rows a warp in flight together and 16-byte loads (two scale groups a load);
// the latent row is staged transposed (element j of 32-chunk q at j * CP + q, CP odd: no bank conflicts).
// Grid (ff / rb, n * topk), rb = 8 warps x LAT_RPW.
extern "C" __global__ void q4_lat_gu2(const int* __restrict__ picks, const int* __restrict__ labels,
                                      const u64* __restrict__ tbl, const float* __restrict__ xl,
                                      float* __restrict__ act, int ff, int L, int G, int topk, int rb)
{ PDL_PRO();
    extern __shared__ float lx[];
    const int r = blockIdx.y, e = picks[r];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, NW = blockDim.x >> 5;
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, ff);
    if (e < 0) {
        for (int i = r0 + threadIdx.x; i < r1; i += blockDim.x) act[(u64)r * ff + i] = 0.0f;
        return;
    }
    const u8* gp = (const u8*)tbl[e];
    if (!gp) return;
    const int nq = L >> 5, CP = nq | 1;
    const float* x = xl + ((u64)(r / topk) * G + labels[e]) * L;
    for (int i = threadIdx.x; i < L; i += blockDim.x) lx[(i & 31) * CP + (i >> 5)] = x[i];
    __syncthreads();
    const u8* up = gp + lat_pbytes(ff, L);
    const float sg = lat_glob(gp, ff, L), su = lat_glob(up, ff, L);
    const u8* gsc = gp + (u64)ff * L / 2;
    const u8* usc = up + (u64)ff * L / 2;
    const int gpr = L >> 4;
    float ag[LAT_RPW], au[LAT_RPW];
    #pragma unroll
    for (int a = 0; a < LAT_RPW; a++) ag[a] = au[a] = 0.0f;
    for (int q = lane; q < nq; q += 32) {
        float xv[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) xv[j] = lx[j * CP + q];
        #pragma unroll
        for (int a = 0; a < LAT_RPW; a++) {
            const int row = r0 + warp + a * NW;
            if (row >= r1) break;
            const uint4 wg = *(const uint4*)(gp + (u64)row * (L / 2) + q * 16);
            const uint4 wu = *(const uint4*)(up + (u64)row * (L / 2) + q * 16);
            const u32 s2g = *(const unsigned short*)(gsc + (u64)row * gpr + 2 * q);
            const u32 s2u = *(const unsigned short*)(usc + (u64)row * gpr + 2 * q);
            ag[a] += lat_dot32(wg, s2g, xv);
            au[a] += lat_dot32(wu, s2u, xv);
        }
    }
    #pragma unroll
    for (int a = 0; a < LAT_RPW; a++) {
        float g = ag[a], u = au[a];
        for (int o = 16; o > 0; o >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, o);
            u += __shfl_xor_sync(0xffffffffu, u, o);
        }
        const int row = r0 + warp + a * NW;
        if (lane == 0 && row < r1) act[(u64)r * ff + row] = silu_f(sg * g) * (su * u);
    }
}

// `q4_lat_dn` in the same form: y [n * topk][L] = sd * (D act). Grid (L / rb, n * topk).
extern "C" __global__ void q4_lat_dn2(const int* __restrict__ picks, const u64* __restrict__ tbl,
                                      const float* __restrict__ act, float* __restrict__ y, int ff, int L, int rb)
{ PDL_PRO();
    extern __shared__ float la[];
    const int r = blockIdx.y, e = picks[r];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, NW = blockDim.x >> 5;
    const int r0 = blockIdx.x * rb, r1 = min(r0 + rb, L);
    if (e < 0) {
        for (int i = r0 + threadIdx.x; i < r1; i += blockDim.x) y[(u64)r * L + i] = 0.0f;
        return;
    }
    if (!tbl[e]) return;
    const int nq = ff >> 5, CP = nq | 1;
    for (int i = threadIdx.x; i < ff; i += blockDim.x) la[(i & 31) * CP + (i >> 5)] = act[(u64)r * ff + i];
    __syncthreads();
    const u8* dp = (const u8*)tbl[e] + 2 * lat_pbytes(ff, L);
    const float sd = lat_glob(dp, L, ff);
    const u8* dsc = dp + (u64)L * ff / 2;
    const int gpr = ff >> 4;
    float ad[LAT_RPW];
    #pragma unroll
    for (int a = 0; a < LAT_RPW; a++) ad[a] = 0.0f;
    for (int q = lane; q < nq; q += 32) {
        float xv[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) xv[j] = la[j * CP + q];
        #pragma unroll
        for (int a = 0; a < LAT_RPW; a++) {
            const int row = r0 + warp + a * NW;
            if (row >= r1) break;
            const uint4 w = *(const uint4*)(dp + (u64)row * (ff / 2) + q * 16);
            const u32 s2 = *(const unsigned short*)(dsc + (u64)row * gpr + 2 * q);
            ad[a] += lat_dot32(w, s2, xv);
        }
    }
    #pragma unroll
    for (int a = 0; a < LAT_RPW; a++) {
        float v = ad[a];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        const int row = r0 + warp + a * NW;
        if (lane == 0 && row < r1) y[(u64)r * L + row] = sd * v;
    }
}

// ---------------------------------------------------------------- latent prompt GEMMs straight from NVFP4
//
// A prompt chunk's routed experts as grouped GEMMs on the tensor cores with no dense copy of the weights:
// a block takes one unit (an expert's run of up to 64 rows of the chunk, `units` = {slot lo, slot hi,
// first row, rows}) and 64 output features; each warp 16 of them. The A fragments of `mma.m16n8k16` come
// from the slot's E2M1 codes and UE4M3 scales decoded in registers into exact f16 (a code's magnitude by
// two byte permutes, its sign by a shift, the group's scale by one multiply: a product of a 2-bit and a
// 4-bit significand), the B fragments from the rows' f16 inputs staged in shared memory 32 columns at a
// time; f32 accumulators, the tensor's global applied once at the end.

// A byte of two E2M1 codes -> their f16 pair (low code first), exact.
__device__ __forceinline__ u32 lat_h2(u32 b) {
    const u32 hb = __byte_perm(0x3E3C3800u, 0x46444240u, b & 0x77u);
    return __byte_perm(hb, 0u, 0x1404u) | ((b & 0x8u) << 12) | ((b & 0x80u) << 24);
}
__device__ __forceinline__ u32 lat_hmul2(u32 a, u32 s) {
    u32 r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(s));
    return r;
}
// a UE4M3 byte as an f16 pair (exact)
__device__ __forceinline__ u32 lat_s2(u32 b) {
    const u32 v = (u32)f2h(lat_ue4m3(b));
    return v | (v << 16);
}
__device__ __forceinline__ void lat_mma(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// The A fragment of k16 half `h` of a 32-column step: rows `wa` / `wb` (16 code bytes each), their two
// scale bytes `sa` / `sb`; lane (gid, tig) takes bytes tig and tig + 4 of the half's 8.
__device__ __forceinline__ void lat_afrag(uint4 wa, uint4 wb, u32 sa, u32 sb, int h, int tig, u32* a) {
    const u32 xa0 = h ? wa.z : wa.x, xa1 = h ? wa.w : wa.y, xb0 = h ? wb.z : wb.x, xb1 = h ? wb.w : wb.y;
    const u32 ssa = lat_s2((sa >> (8 * h)) & 0xFFu), ssb = lat_s2((sb >> (8 * h)) & 0xFFu);
    a[0] = lat_hmul2(lat_h2((xa0 >> (8 * tig)) & 0xFFu), ssa);
    a[1] = lat_hmul2(lat_h2((xb0 >> (8 * tig)) & 0xFFu), ssb);
    a[2] = lat_hmul2(lat_h2((xa1 >> (8 * tig)) & 0xFFu), ssa);
    a[3] = lat_hmul2(lat_h2((xb1 >> (8 * tig)) & 0xFFu), ssb);
}

#define LAT_XP 20
// gate and up of every unit's rows: act [row][ff] (f16) = silu(sg * (G x)) * (su * (U x)), x the row's latent
// input xl [rowtok[row]] (f16, L wide). Grid (units, ff / 64), 128 threads.
extern "C" __global__ void __launch_bounds__(128) q4_lat_ggu(const int4* __restrict__ units, const int* __restrict__ rowtok,
                                                             const unsigned short* __restrict__ xl, unsigned short* __restrict__ act,
                                                             int ff, int L)
{ PDL_PRO();
    __shared__ __align__(16) u32 xs[2][64 * LAT_XP];
    const int4 un = units[blockIdx.x];
    const u8* gp = (const u8*)(((u64)(u32)un.y << 32) | (u64)(u32)un.x);
    const int off = un.z, cnt = un.w;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const u8* upp = gp + lat_pbytes(ff, L);
    const u8* gsc = gp + (u64)ff * L / 2;
    const u8* usc = upp + (u64)ff * L / 2;
    const int gpr = L >> 4, half_l = L >> 1;
    const int ra = blockIdx.y * 64 + warp * 16 + gid, rb = ra + 8;
    float ag[8][4], au[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        #pragma unroll
        for (int i = 0; i < 4; i++) ag[n][i] = au[n][i] = 0.0f;
    const int nt = (cnt + 7) >> 3;
    auto load = [&](int buf, int k0) {
        for (int i = tid; i < 256; i += 128) {
            const int t = i >> 2, part = i & 3;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < cnt) v = *(const uint4*)(xl + (u64)rowtok[off + t] * L + k0 + part * 8);
            *(uint4*)&xs[buf][t * LAT_XP + part * 4] = v;
        }
    };
    const int nk = L >> 5;
    load(0, 0);
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
        const int buf = kk & 1;
        if (kk + 1 < nk) load(buf ^ 1, (kk + 1) * 32);
        const uint4 ga = *(const uint4*)(gp + (u64)ra * half_l + kk * 16);
        const uint4 gb = *(const uint4*)(gp + (u64)rb * half_l + kk * 16);
        const uint4 ua = *(const uint4*)(upp + (u64)ra * half_l + kk * 16);
        const uint4 ub = *(const uint4*)(upp + (u64)rb * half_l + kk * 16);
        const u32 sga = *(const unsigned short*)(gsc + (u64)ra * gpr + 2 * kk);
        const u32 sgb = *(const unsigned short*)(gsc + (u64)rb * gpr + 2 * kk);
        const u32 sua = *(const unsigned short*)(usc + (u64)ra * gpr + 2 * kk);
        const u32 sub = *(const unsigned short*)(usc + (u64)rb * gpr + 2 * kk);
        const u32* X = xs[buf];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            u32 a[4], b[4];
            lat_afrag(ga, gb, sga, sgb, h, tig, a);
            lat_afrag(ua, ub, sua, sub, h, tig, b);
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                if (n >= nt) break;
                const u32 b0 = X[(n * 8 + gid) * LAT_XP + h * 8 + tig];
                const u32 b1 = X[(n * 8 + gid) * LAT_XP + h * 8 + tig + 4];
                lat_mma(ag[n], a[0], a[1], a[2], a[3], b0, b1);
                lat_mma(au[n], b[0], b[1], b[2], b[3], b0, b1);
            }
        }
        __syncthreads();
    }
    const float sg = lat_glob(gp, ff, L), su = lat_glob(upp, ff, L);
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        if (n >= nt) break;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int t = n * 8 + 2 * tig + (i & 1), row = i < 2 ? ra : rb;
            if (t < cnt)
                act[(u64)(off + t) * ff + row] = f2h_sat(silu_f(sg * ag[n][i]) * (su * au[n][i]));
        }
    }
}

// down of every unit's rows: y [rowpick[row]][L] = sd * (D act[row]) (f32). Grid (units, L / 64), 128 threads.
extern "C" __global__ void __launch_bounds__(128) q4_lat_gdn(const int4* __restrict__ units, const int* __restrict__ rowpick,
                                                             const unsigned short* __restrict__ act, float* __restrict__ y,
                                                             int ff, int L)
{ PDL_PRO();
    __shared__ __align__(16) u32 xs[2][64 * LAT_XP];
    const int4 un = units[blockIdx.x];
    const u8* dp = (const u8*)(((u64)(u32)un.y << 32) | (u64)(u32)un.x) + 2 * lat_pbytes(ff, L);
    const int off = un.z, cnt = un.w;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const u8* dsc = dp + (u64)L * ff / 2;
    const int gpr = ff >> 4, half_f = ff >> 1;
    const int ra = blockIdx.y * 64 + warp * 16 + gid, rb = ra + 8;
    float ad[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        #pragma unroll
        for (int i = 0; i < 4; i++) ad[n][i] = 0.0f;
    const int nt = (cnt + 7) >> 3;
    auto load = [&](int buf, int k0) {
        for (int i = tid; i < 256; i += 128) {
            const int t = i >> 2, part = i & 3;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < cnt) v = *(const uint4*)(act + (u64)(off + t) * ff + k0 + part * 8);
            *(uint4*)&xs[buf][t * LAT_XP + part * 4] = v;
        }
    };
    const int nk = ff >> 5;
    load(0, 0);
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
        const int buf = kk & 1;
        if (kk + 1 < nk) load(buf ^ 1, (kk + 1) * 32);
        const uint4 wa = *(const uint4*)(dp + (u64)ra * half_f + kk * 16);
        const uint4 wb = *(const uint4*)(dp + (u64)rb * half_f + kk * 16);
        const u32 sa = *(const unsigned short*)(dsc + (u64)ra * gpr + 2 * kk);
        const u32 sb = *(const unsigned short*)(dsc + (u64)rb * gpr + 2 * kk);
        const u32* X = xs[buf];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            u32 a[4];
            lat_afrag(wa, wb, sa, sb, h, tig, a);
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                if (n >= nt) break;
                const u32 b0 = X[(n * 8 + gid) * LAT_XP + h * 8 + tig];
                const u32 b1 = X[(n * 8 + gid) * LAT_XP + h * 8 + tig + 4];
                lat_mma(ad[n], a[0], a[1], a[2], a[3], b0, b1);
            }
        }
        __syncthreads();
    }
    const float sd = lat_glob(dp, L, ff);
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        if (n >= nt) break;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int t = n * 8 + 2 * tig + (i & 1), row = i < 2 ? ra : rb;
            if (t < cnt) y[(u64)rowpick[off + t] * L + row] = sd * ad[n][i];
        }
    }
}

// ---------------------------------------------------------------- a round's picks by expert
//
// A verify round's picks grouped by expert (`q4_lat_units`, one block, a thread a pick): unit u = units[10 u ..]
// {expert, picks, pick 0..7} in order of the expert's first pick, their count at nunits[0]. The unit kernels
// then read each picked expert once for all its rows (a round of four rows picks some experts twice or more).
extern "C" __global__ void q4_lat_units(const int* __restrict__ picks, int routes, int* __restrict__ units,
                                        u32* __restrict__ nunits)
{ PDL_PRO();
    __shared__ int sp[256], lead[256], uid[256];
    const int i = threadIdx.x;
    const int e = i < routes ? picks[i] : -1;
    sp[i] = e;
    __syncthreads();
    int ld = -1, k = 0;
    if (e >= 0) {
        ld = i;
        for (int j = 0; j < i; j++)
            if (sp[j] == e) {
                if (ld == i) ld = j;
                k++;
            }
    }
    lead[i] = ld;
    __syncthreads();
    if (ld == i) {
        int u = 0;
        for (int j = 0; j < i; j++) u += lead[j] == j;
        uid[i] = u;
    }
    __syncthreads();
    if (ld >= 0) {
        const int u = uid[ld];
        if (ld == i) {
            int c = 1;
            for (int j = i + 1; j < routes; j++) c += sp[j] == e;
            units[u * 10] = e;
            units[u * 10 + 1] = min(c, 8);
        }
        if (k < 8) units[u * 10 + 2 + k] = i;
    }
    if (i == 0) {
        int nu = 0;
        for (int j = 0; j < routes; j++) nu += lead[j] == j;
        nunits[0] = (u32)nu;
    }
}

#ifndef LAT_U_RPW
#define LAT_U_RPW 2
#endif
#define LAT_U_CS 36
// Gate and up of a unit's rows: act [pick][ff] = silu(sg * (G x)) * (su * (U x)). Grid (ff / (8 LAT_U_RPW),
// routes), 256 threads; blocks past the round's units return, as do units of an expert outside the bank (the
// host's). The unit's latent rows are staged side by side (LAT_U_CS floats a 32-column chunk), each weight row
// decoded once for all of them.
extern "C" __global__ void q4_lat_gu3(const int* __restrict__ units, const u32* __restrict__ nunits,
                                      const int* __restrict__ labels, const u64* __restrict__ tbl,
                                      const float* __restrict__ xl, float* __restrict__ act,
                                      int ff, int L, int G, int topk)
{ PDL_PRO();
    extern __shared__ __align__(16) float ux[];
    if (blockIdx.y >= nunits[0]) return;
    const int* un = units + blockIdx.y * 10;
    const int e = un[0], cnt = un[1];
    const u8* gp = (const u8*)tbl[e];
    if (!gp) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nq = L >> 5, XS = nq * LAT_U_CS;
    const int grp = labels[e];
    for (int i = threadIdx.x; i < cnt * L; i += blockDim.x) {
        const int j = i / L, c = i - j * L;
        const int p = un[2 + j];
        ux[j * XS + (c >> 5) * LAT_U_CS + (c & 31)] = xl[((u64)(p / topk) * G + grp) * L + c];
    }
    __syncthreads();
    constexpr int RPW = LAT_U_RPW;
    const int r0 = blockIdx.x * 8 * RPW + warp * RPW;
    const u8* up = gp + lat_pbytes(ff, L);
    const u8* gsc = gp + (u64)ff * L / 2;
    const u8* usc = up + (u64)ff * L / 2;
    const int gpr = L >> 4;
    float ag[RPW][8], au[RPW][8];
    #pragma unroll
    for (int a = 0; a < RPW; a++)
        #pragma unroll
        for (int j = 0; j < 8; j++) ag[a][j] = au[a][j] = 0.0f;
    for (int q = lane; q < nq; q += 32) {
        uint4 wg[RPW], wu[RPW];
        u32 sg2[RPW], su2[RPW];
        #pragma unroll
        for (int a = 0; a < RPW; a++) {
            const int row = min(r0 + a, ff - 1);
            wg[a] = *(const uint4*)(gp + (u64)row * (L / 2) + q * 16);
            wu[a] = *(const uint4*)(up + (u64)row * (L / 2) + q * 16);
            sg2[a] = *(const unsigned short*)(gsc + (u64)row * gpr + 2 * q);
            su2[a] = *(const unsigned short*)(usc + (u64)row * gpr + 2 * q);
        }
        #pragma unroll
        for (int a = 0; a < RPW; a++) {
            float fg[32], fu[32];
            const u32 xg[4] = {wg[a].x, wg[a].y, wg[a].z, wg[a].w}, xu[4] = {wu[a].x, wu[a].y, wu[a].z, wu[a].w};
            #pragma unroll
            for (int k = 0; k < 4; k++)
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const u32 bg = (xg[k] >> (8 * b)) & 0xFFu, bu = (xu[k] >> (8 * b)) & 0xFFu;
                    fg[8 * k + 2 * b] = lat_e2m1(bg & 15u);
                    fg[8 * k + 2 * b + 1] = lat_e2m1(bg >> 4);
                    fu[8 * k + 2 * b] = lat_e2m1(bu & 15u);
                    fu[8 * k + 2 * b + 1] = lat_e2m1(bu >> 4);
                }
            const float g0 = lat_ue4m3(sg2[a] & 0xFFu), g1 = lat_ue4m3(sg2[a] >> 8);
            const float u0 = lat_ue4m3(su2[a] & 0xFFu), u1 = lat_ue4m3(su2[a] >> 8);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (j >= cnt) break;
                const float4* xp = (const float4*)(ux + j * XS + q * LAT_U_CS);
                float pg0 = 0.0f, pg1 = 0.0f, pu0 = 0.0f, pu1 = 0.0f;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const float4 x4 = xp[v];
                    const float sgv = fg[4 * v] * x4.x + fg[4 * v + 1] * x4.y + fg[4 * v + 2] * x4.z + fg[4 * v + 3] * x4.w;
                    const float suv = fu[4 * v] * x4.x + fu[4 * v + 1] * x4.y + fu[4 * v + 2] * x4.z + fu[4 * v + 3] * x4.w;
                    if (v < 4) { pg0 += sgv; pu0 += suv; } else { pg1 += sgv; pu1 += suv; }
                }
                ag[a][j] += pg0 * g0 + pg1 * g1;
                au[a][j] += pu0 * u0 + pu1 * u1;
            }
        }
    }
    const float sg = lat_glob(gp, ff, L), su = lat_glob(up, ff, L);
    #pragma unroll
    for (int a = 0; a < RPW; a++) {
        const int row = r0 + a;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= cnt) break;
            float g = ag[a][j], u = au[a][j];
            for (int o = 16; o > 0; o >>= 1) {
                g += __shfl_xor_sync(0xffffffffu, g, o);
                u += __shfl_xor_sync(0xffffffffu, u, o);
            }
            if (lane == 0 && row < ff) act[(u64)un[2 + j] * ff + row] = silu_f(sg * g) * (su * u);
        }
    }
}

// Down of a unit's rows: y [pick][L] = sd * (D act[pick]). Grid (L / (8 LAT_U_RPW), routes), 256 threads.
extern "C" __global__ void q4_lat_dn3(const int* __restrict__ units, const u32* __restrict__ nunits,
                                      const u64* __restrict__ tbl, const float* __restrict__ act,
                                      float* __restrict__ y, int ff, int L)
{ PDL_PRO();
    extern __shared__ __align__(16) float ua[];
    if (blockIdx.y >= nunits[0]) return;
    const int* un = units + blockIdx.y * 10;
    const int e = un[0], cnt = un[1];
    if (!tbl[e]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nq = ff >> 5, XS = nq * LAT_U_CS;
    for (int i = threadIdx.x; i < cnt * ff; i += blockDim.x) {
        const int j = i / ff, c = i - j * ff;
        ua[j * XS + (c >> 5) * LAT_U_CS + (c & 31)] = act[(u64)un[2 + j] * ff + c];
    }
    __syncthreads();
    constexpr int RPW = LAT_U_RPW;
    const int r0 = blockIdx.x * 8 * RPW + warp * RPW;
    const u8* dp = (const u8*)tbl[e] + 2 * lat_pbytes(ff, L);
    const u8* dsc = dp + (u64)L * ff / 2;
    const int gpr = ff >> 4;
    float ad[RPW][8];
    #pragma unroll
    for (int a = 0; a < RPW; a++)
        #pragma unroll
        for (int j = 0; j < 8; j++) ad[a][j] = 0.0f;
    for (int q = lane; q < nq; q += 32) {
        uint4 w[RPW];
        u32 s2[RPW];
        #pragma unroll
        for (int a = 0; a < RPW; a++) {
            const int row = min(r0 + a, L - 1);
            w[a] = *(const uint4*)(dp + (u64)row * (ff / 2) + q * 16);
            s2[a] = *(const unsigned short*)(dsc + (u64)row * gpr + 2 * q);
        }
        #pragma unroll
        for (int a = 0; a < RPW; a++) {
            float f[32];
            const u32 xw[4] = {w[a].x, w[a].y, w[a].z, w[a].w};
            #pragma unroll
            for (int k = 0; k < 4; k++)
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const u32 by = (xw[k] >> (8 * b)) & 0xFFu;
                    f[8 * k + 2 * b] = lat_e2m1(by & 15u);
                    f[8 * k + 2 * b + 1] = lat_e2m1(by >> 4);
                }
            const float d0 = lat_ue4m3(s2[a] & 0xFFu), d1 = lat_ue4m3(s2[a] >> 8);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (j >= cnt) break;
                const float4* xp = (const float4*)(ua + j * XS + q * LAT_U_CS);
                float p0 = 0.0f, p1 = 0.0f;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const float4 x4 = xp[v];
                    const float sv = f[4 * v] * x4.x + f[4 * v + 1] * x4.y + f[4 * v + 2] * x4.z + f[4 * v + 3] * x4.w;
                    if (v < 4) p0 += sv; else p1 += sv;
                }
                ad[a][j] += p0 * d0 + p1 * d1;
            }
        }
    }
    const float sd = lat_glob(dp, L, ff);
    #pragma unroll
    for (int a = 0; a < RPW; a++) {
        const int row = r0 + a;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= cnt) break;
            float v = ad[a][j];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row < L) y[(u64)un[2 + j] * L + row] = sd * v;
        }
    }
}

// acc [n * G][L]: row (t * G + g) = sum over the picks k of token t in group g of wts * y[t * topk + k].
extern "C" __global__ void q4_lat_sum(const int* __restrict__ picks, const int* __restrict__ labels,
                                      const float* __restrict__ wts, const float* __restrict__ y,
                                      float* __restrict__ acc, int n, int G, int L, int topk)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * G * L) return;
    const u64 tg = i / L, j = i - tg * L;
    const int t = (int)(tg / G), g = (int)(tg - (u64)t * G);
    float a = 0.0f;
    for (int k = 0; k < topk; k++) {
        const int e = picks[t * topk + k];
        if (e >= 0 && labels[e] == g) a += wts[t * topk + k] * y[((u64)t * topk + k) * L + j];
    }
    acc[i] = a;
}

// NVFP4 experts -> dense wt planes without their globals (E2M1 * UE4M3, exact in f16): grid
// (words of a matrix / 256, 3 * B), a thread expands one u32 of codes (8 values).
extern "C" __global__ void q4_lat_expand(const u64* __restrict__ slots, void* __restrict__ Wg, void* __restrict__ Wu,
                                         void* __restrict__ Wd, int ff, int L, int half)
{ PDL_PRO();
    const int m = blockIdx.y % 3, j = blockIdx.y / 3;
    const int rows = m == 2 ? L : ff, cols = m == 2 ? ff : L;
    const u64 wi = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= (u64)rows * cols / 8 || !slots[j]) return;
    const u8* p = (const u8*)slots[j] + (u64)m * lat_pbytes(ff, L);
    const u64 base = wi * 8;
    const float s = lat_ue4m3(p[(u64)rows * cols / 2 + base / 16]);
    const u32 w = *(const u32*)(p + wi * 4);
    void* W = m == 0 ? Wg : (m == 1 ? Wu : Wd);
    const u64 o = (u64)j * rows * cols + base;
    for (int k = 0; k < 4; k++) {
        const u32 by = (w >> (8 * k)) & 0xFFu;
        stf(W, o + 2 * k, lat_e2m1(by & 15u) * s, half);
        stf(W, o + 2 * k + 1, lat_e2m1(by >> 4) * s, half);
    }
}

// a = silu(fg * g) * (fu * u) * inv for the rows of a sub-batch: row r is expert j = r / per's, whose
// gate and up GEMMs ran with the sub-batch's alphas; f [2 * B] = (sg_j / alpha_g, su_j / alpha_u).
extern "C" __global__ void q4_lat_act(const void* __restrict__ g, const void* __restrict__ u, void* __restrict__ a,
                                      const float* __restrict__ f, int rows, int per, int ff, int in_half,
                                      int out_half, float inv)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * ff) return;
    const int j = (int)(i / ff) / per;
    stf(a, i, silu_f(f[2 * j] * ldf(g, i, in_half)) * (f[2 * j + 1] * ldf(u, i, in_half)) * inv, out_half);
}


// The groups any of the n rows picked: bit g of mask[0].
extern "C" __global__ void q4_lat_gmask(const int* __restrict__ picks, const int* __restrict__ labels,
                                        u32* __restrict__ mask, int n, int topk)
{ PDL_PRO();
    // one warp: the picks a lane apart, the groups' bits or-ed across the lanes
    if (blockIdx.x != 0 || threadIdx.x >= 32) return;
    u32 m = 0u;
    for (int i = threadIdx.x; i < n * topk; i += 32) {
        const int e = picks[i];
        if (e >= 0) m |= 1u << (labels[e] & 31);
    }
    m = __reduce_or_sync(0xffffffffu, m);
    if (threadIdx.x == 0) mask[0] = m;
}

#ifndef LAT_MV_RB
#define LAT_MV_RB 16
#endif
// y [n][rows] (rows ys apart) = W x for n <= 8 rows of x (xs apart), W q8_2f [rows][cols]: int8 body,
// f32 row scale `rs`, f32 column field `col`. The row is read once for all n rows. Group blocks no row
// picked are skipped: mode 0 (D) skips the row blocks of glen rows; mode 1 (U) gives every group of glen
// columns its own blocks (grid y = the group) and writes partial sums to `y` = part [G][n][rows], which
// `q4_lat_red` adds in group order (the inputs of an unpicked group are zero, its blocks return). The
// columns go in chunks of `ch` (a divisor of cols, of glen in mode 1), staged for all rows. 256 threads,
// LAT_MV_RB rows a block (a divisor of glen in mode 0).
extern "C" __global__ void q4_lat_mv(const u8* __restrict__ body, const float* __restrict__ rs,
                                     const float* __restrict__ col, const float* __restrict__ x, int xs,
                                     float* __restrict__ y, int ys, int rows, int cols, int n, int glen,
                                     const u32* __restrict__ gmask, int mode, int ch)
{ PDL_PRO();
    extern __shared__ float sx[];
    const u32 m = gmask[0];
    const int r0 = blockIdx.x * LAT_MV_RB;
    if (r0 >= rows) return;
    if (mode == 0 && !((m >> ((r0 / glen) & 31)) & 1u)) return;
    if (mode == 1 && !((m >> (blockIdx.y & 31)) & 1u)) return;
    // mode 1: this block's column range is its group's
    const int cb = mode == 1 ? (int)blockIdx.y * glen : 0;
    const int ce = mode == 1 ? cb + glen : cols;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int RPW = LAT_MV_RB / 8;
    float acc[RPW][8];
    #pragma unroll
    for (int a = 0; a < RPW; a++)
        #pragma unroll
        for (int t = 0; t < 8; t++) acc[a][t] = 0.0f;
    // the staged chunk is transposed: element j of 16-group q of row t at t * 16 * CP + j * CP + q, CP odd,
    // so the 32 lanes of one load (consecutive q) hit 32 different banks
    const int nq = ch >> 4, CP = nq | 1;
    for (int c0 = cb; c0 < ce; c0 += ch) {
        __syncthreads();
        for (int i = threadIdx.x; i < n * ch; i += blockDim.x) {
            const int t = i / ch, c = i - t * ch;
            sx[t * 16 * CP + (c & 15) * CP + (c >> 4)] = x[(u64)t * xs + c0 + c] * col[c0 + c];
        }
        __syncthreads();
        #pragma unroll
        for (int a = 0; a < RPW; a++) {
            const int r = r0 + warp * RPW + a;
            if (r >= rows) continue;
            const uint4* W = (const uint4*)(body + (u64)r * cols + c0);
            for (int q = lane; q < ch / 16; q += 32) {
                const uint4 w = W[q];
                const u32 ws[4] = {w.x, w.y, w.z, w.w};
                #pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t >= n) break;
                    const float* p = sx + t * 16 * CP + q;
                    float s = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        s += (float)((int)(ws[k] << 24) >> 24) * p[(4 * k) * CP];
                        s += (float)((int)(ws[k] << 16) >> 24) * p[(4 * k + 1) * CP];
                        s += (float)((int)(ws[k] << 8) >> 24) * p[(4 * k + 2) * CP];
                        s += (float)((int)ws[k] >> 24) * p[(4 * k + 3) * CP];
                    }
                    acc[a][t] += s;
                }
            }
        }
    }
    #pragma unroll
    for (int a = 0; a < RPW; a++) {
        const int r = r0 + warp * RPW + a;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            float v = acc[a][t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && r < rows && t < n) {
                if (mode == 1) y[((u64)blockIdx.y * n + t) * rows + r] = v * rs[r];
                else y[(u64)t * ys + r] = v * rs[r];
            }
        }
    }
}

#ifndef LAT_MV2_RPW
#define LAT_MV2_RPW 2
#endif
// `q4_lat_mv` in the form of the small path's matvec: the chunk's inputs (times the column field) side by side,
// MV_CS floats a 16-column word (four 128-bit loads a word and row), staged with 128-bit loads, and the
// LAT_MV2_RPW rows of a warp loaded together before any is used. Same grid, modes and results' layout;
// LAT_MV2_RPW * 8 rows a block.
template <int NB>
__device__ __forceinline__ void lat_mv2_body(const u8* __restrict__ body, const float* __restrict__ rs,
                                             const float* __restrict__ col, const float* __restrict__ x, int xs,
                                             float* __restrict__ y, int ys, int rows, int cols, int glen,
                                             int mode, int ch, float* sh)
{
    constexpr int RPW = LAT_MV2_RPW;
    const int r0 = blockIdx.x * (8 * RPW);
    const int cb = mode == 1 ? (int)blockIdx.y * glen : 0;
    const int ce = mode == 1 ? cb + glen : cols;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nch = ch >> 4, XS = nch * MV_CS, c4 = ch >> 2;
    float acc[RPW][NB];
    #pragma unroll
    for (int a = 0; a < RPW; a++)
        #pragma unroll
        for (int b = 0; b < NB; b++) acc[a][b] = 0.0f;
    for (int c0 = cb; c0 < ce; c0 += ch) {
        __syncthreads();
        for (int m4 = tid; m4 < c4; m4 += blockDim.x) {
            const float4 cv = ((const float4*)(col + c0))[m4];
            float4 xv[NB];
            #pragma unroll
            for (int b = 0; b < NB; b++) xv[b] = ((const float4*)(x + (u64)b * xs + c0))[m4];
            const int c = m4 >> 2, q = (m4 & 3) * 4;
            #pragma unroll
            for (int b = 0; b < NB; b++)
                *(float4*)(sh + b * XS + c * MV_CS + q) = make_float4(xv[b].x * cv.x, xv[b].y * cv.y, xv[b].z * cv.z, xv[b].w * cv.w);
        }
        __syncthreads();
        for (int c = lane; c < nch; c += 32) {
            uint4 w[RPW];
            #pragma unroll
            for (int a = 0; a < RPW; a++) {
                const int r = r0 + warp * RPW + a;
                w[a] = r < rows ? ((const uint4*)(body + (u64)r * cols + c0))[c] : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int a = 0; a < RPW; a++) {
                float wf[16];
                const u32 ws[4] = {w[a].x, w[a].y, w[a].z, w[a].w};
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    wf[4 * k] = (float)((int)(ws[k] << 24) >> 24);
                    wf[4 * k + 1] = (float)((int)(ws[k] << 16) >> 24);
                    wf[4 * k + 2] = (float)((int)(ws[k] << 8) >> 24);
                    wf[4 * k + 3] = (float)((int)ws[k] >> 24);
                }
                #pragma unroll
                for (int b = 0; b < NB; b++) {
                    const float4* p = (const float4*)(sh + b * XS + c * MV_CS);
                    float s = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float4 v = p[q];
                        s += wf[4 * q] * v.x + wf[4 * q + 1] * v.y + wf[4 * q + 2] * v.z + wf[4 * q + 3] * v.w;
                    }
                    acc[a][b] += s;
                }
            }
        }
    }
    #pragma unroll
    for (int a = 0; a < RPW; a++) {
        const int r = r0 + warp * RPW + a;
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            float v = acc[a][b];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && r < rows) {
                if (mode == 1) y[((u64)blockIdx.y * NB + b) * rows + r] = v * rs[r];
                else y[(u64)b * ys + r] = v * rs[r];
            }
        }
    }
}

extern "C" __global__ void q4_lat_mv2(const u8* __restrict__ body, const float* __restrict__ rs,
                                      const float* __restrict__ col, const float* __restrict__ x, int xs,
                                      float* __restrict__ y, int ys, int rows, int cols, int n, int glen,
                                      const u32* __restrict__ gmask, int mode, int ch)
{ PDL_PRO();
    extern __shared__ __align__(16) float sx[];
    const u32 m = gmask[0];
    const int r0 = blockIdx.x * (8 * LAT_MV2_RPW);
    if (r0 >= rows) return;
    if (mode == 0 && !((m >> ((r0 / glen) & 31)) & 1u)) return;
    if (mode == 1 && !((m >> (blockIdx.y & 31)) & 1u)) return;
    switch (n) {
        case 1: lat_mv2_body<1>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 2: lat_mv2_body<2>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 3: lat_mv2_body<3>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 4: lat_mv2_body<4>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 5: lat_mv2_body<5>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 6: lat_mv2_body<6>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        case 7: lat_mv2_body<7>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
        default: lat_mv2_body<8>(body, rs, col, x, xs, y, ys, rows, cols, glen, mode, ch, sx); break;
    }
}

// y [n][rows] (ys apart) = the sum over the picked groups of part [G][n][rows], in group order.
extern "C" __global__ void q4_lat_red(const float* __restrict__ part, const u32* __restrict__ gmask,
                                      float* __restrict__ y, int ys, int rows, int n, int G)
{ PDL_PRO();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    const int t = i / rows, r = i - t * rows;
    const u32 m = gmask[0];
    float a = 0.0f;
    for (int g = 0; g < G; g++)
        if ((m >> g) & 1u) a += part[((u64)g * n + t) * rows + r];
    y[(u64)t * ys + r] = a;
}

// y [picks][L]: the rows of this sub-batch's D output, each at its pick (rowpick[row]), times its
// expert's factor fd[row / per] (sd / alpha_d). Every pick's row is written once over the chunk; the
// groups' sums then take them in pick order (`q4_lat_sum`), so the result does not depend on the
// sub-batches.
extern "C" __global__ void q4_lat_put(const void* __restrict__ D, const int* __restrict__ rowpick,
                                      const float* __restrict__ fd, float* __restrict__ y, int rows, int L,
                                      int per, int d_half)
{ PDL_PRO();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * L) return;
    const int row = (int)(i / L), j = (int)(i - (u64)row * L);
    const int p = rowpick[row];
    if (p < 0) return;
    y[(u64)p * L + j] = fd[row / per] * ldf(D, i, d_half);
}
