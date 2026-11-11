// The Qwen3.5-family hybrid on CUDA (`gpu_cuda/hyb.rs`): GDN layers and gated
// full attention with Q/K norms, behind the token and batch graph contracts.
// Compiled by NVRTC per attention geometry: HPK query heads per KV head, HD
// head width (a multiple of 32), DPL = HD / 32 values per lane.
//
// The small path (a decode step, a verify window of up to 16 rows) reads the
// weights straight from their codec; every row of a window goes through the
// same arithmetic, in the same order, as a decode step of that row, so a
// verify and the plain decode it stands for agree to the bit.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;

#define DPL (HD / 32)
#define INF __int_as_float(0x7f800000)
#define FULL 0xffffffffu

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ u16 f2h_sat(float f) { return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f)); }
__device__ __forceinline__ void stf(void* p, u64 i, float v, int half) {
    if (half) ((u16*)p)[i] = f2h_sat(v); else ((float*)p)[i] = v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// bf16 planes of the tensor-core small path (see its section below).
__device__ __forceinline__ u16 f2bf(float f) {
    u16 h;
    asm("{ cvt.rn.bf16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float bf2f(u16 h) { return __uint_as_float((u32)h << 16); }

// Planes an input row of the tensor-core path is split into: 2 bf16 parts
// carry ~17 significant bits a value (a dot product of 5120 lands within
// ~1e-5 of its f32 sum, the f32 sum's own order error ~4e-6); 3 carry an
// f32's 24.
#ifndef NPL
#define NPL 2
#endif

// v as NPL bf16 parts at index i of planes p0, p1 (, p2).
__device__ __forceinline__ void put3(u16* __restrict__ p0, u16* __restrict__ p1, u16* __restrict__ p2, size_t i,
                                     float v)
{
    const u16 a = f2bf(v);
    const float r1 = __fsub_rn(v, bf2f(a));
    const u16 b = f2bf(r1);
    p0[i] = a;
    p1[i] = b;
    if (NPL > 2) p2[i] = f2bf(__fsub_rn(r1, bf2f(b)));
}


// Job record of the small-path matvec launches (`HyJob` in hyb.rs).
struct HyJob {
    u64 w;     // weight body
    u64 aux0;  // q4tp: codes
    u64 aux1;  // q4tp: ladder
    u64 y;     // output, row j at y + j * ystride
    u64 res;   // 0, or the rows added to the product
    int rows;
    int kind;  // 6 = q4tp, 4 = f32; bits 16.. rows a warp owns (q4tp)
    int blk0;  // first block of this job in the launch
    int ystride;
};

// RMS norm of every row of x [B][n] into f32 or f16 (a block of 256 threads
// a row): the very arithmetic `mv_multi` folds into its staging.
extern "C" __global__ void hy_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                                      void* __restrict__ out, int n, float eps, int gemma, int half,
                                      u16* __restrict__ p0, u16* __restrict__ p1, u16* __restrict__ p2, int pstride)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* xr = x + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float v = xr[i];
        s = fmaf(v, v, s);
    }
    for (int k = 16; k > 0; k >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, k));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = 1.0f / (float)sqrt(tot / (double)n + (double)eps);
    for (int i = tid; i < n; i += T) {
        const float v = __fmul_rn(__fmul_rn(xr[i], inv), gemma ? 1.0f + w[i] : w[i]);
        if (out) stf(out, (size_t)blockIdx.x * n + i, v, half);
        if (p0) put3(p0, p1, p2, (size_t)blockIdx.x * pstride + i, v);
    }
}

// silu(g) * u of the rows [n][ld] (the first `cols` of each) into the
// planes [n][pstride].
extern "C" __global__ void hy_silu_mul(const float* __restrict__ g, const float* __restrict__ u, int ld, int cols,
                                       int n, u16* __restrict__ p0, u16* __restrict__ p1, u16* __restrict__ p2,
                                       int pstride)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * cols) return;
    const int j = (int)(i / cols), c = (int)(i - (u64)j * cols);
    const float a = g[(size_t)j * ld + c];
    put3(p0, p1, p2, (size_t)j * pstride + c, __fmul_rn(__fdiv_rn(a, __fadd_rn(1.0f, expf(-a))), u[(size_t)j * ld + c]));
}

// The FFN middle as the f16 operand of the down GEMM: silu(g) * u / 16.
extern "C" __global__ void hy_silu_half(const float* __restrict__ g, const float* __restrict__ u,
                                        u16* __restrict__ xh, u64 n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        xh[i] = f2h_sat(__fmul_rn(__fdiv_rn(a, __fadd_rn(1.0f, expf(-a))), u[i]) * 0.0625f);
    }
}

// f32 -> f16 over n values (saturating).
extern "C" __global__ void hy_to_half(const float* __restrict__ x, u16* __restrict__ xh, u64 n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xh[i] = f2h_sat(x[i]);
}

// ------------------------------------------------------------------- GDN
//
// Rows are [token][...]. qkv [n][c_dim] the raw projections, cq the conv
// output, ab [n][2 nv] (a then b), gb the gates, o [n][nv dv].

// Causal depthwise conv + SiLU: rows before the call come from the ring
// [(kk-1)][c_dim] (oldest first).
extern "C" __global__ void hy_gdn_conv(const float* __restrict__ qkv, int qstride, const float* __restrict__ ring,
                                       const float* __restrict__ taps, float* __restrict__ cq, int n, int c_dim,
                                       int kk)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * c_dim) return;
    const int t = (int)(i / c_dim);
    const int c = (int)(i - (u64)t * c_dim);
    const float* tp = taps + (u64)c * kk;
    double acc = (double)qkv[(u64)t * qstride + c] * (double)tp[kk - 1];
    for (int j = 0; j < kk - 1; j++) {
        const int src = t - (kk - 1 - j);
        const float v = src >= 0 ? qkv[(u64)src * qstride + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        acc += (double)v * (double)tp[j];
    }
    cq[i] = (float)(acc / (1.0 + exp(-acc)));
}

// The ring after the first `n` rows of the call: the last kk-1 raw rows of
// [ring, qkv]. In place, a thread a channel walking the slots oldest first
// (a slot is read before any later slot is written).
extern "C" __global__ void hy_gdn_ring(float* __restrict__ ring, const float* __restrict__ qkv, int qstride, int n,
                                       int c_dim, int kk)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim || n <= 0) return;
    for (int j = 0; j < kk - 1; j++) {
        const int src = n - (kk - 1) + j;
        const float v = src >= 0 ? qkv[(u64)src * qstride + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        ring[(u64)j * c_dim + c] = v;
    }
}

// L2 norm of the q and k heads in place, a block per (token, head of 2 nk);
// q heads also divided by sqrt(dk).
extern "C" __global__ void hy_gdn_prep(float* __restrict__ cq, int c_dim, int nk, int dk)
{
    __shared__ double red[256];
    const u64 t = blockIdx.x / (2 * nk);
    const int g = (int)(blockIdx.x - t * (2 * nk));
    float* xr = cq + t * c_dim + (u64)g * dk;
    double ss = 0.0;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        const double v = xr[i];
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double nrm = sqrt(red[0] + 1e-6);
    const float inv = (float)(1.0 / (g < nk ? nrm * sqrt((double)dk) : nrm));
    for (int i = threadIdx.x; i < dk; i += blockDim.x) xr[i] = __fmul_rn(xr[i], inv);
}

// gb[t][h] = exp(-exp(a_log) softplus(a + dt_bias)), gb[t][nv + h] = sigmoid(b).
extern "C" __global__ void hy_gdn_gb(const float* __restrict__ ab, const float* __restrict__ a_log,
                                     const float* __restrict__ dt_bias, float* __restrict__ gb, int n, int nv)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nv) return;
    const u64 t = i / nv;
    const int h = (int)(i - t * nv);
    const double x = (double)ab[t * 2 * nv + h] + (double)dt_bias[h];
    const double sp = x > 20.0 ? x : log1p(exp(x));
    gb[t * 2 * nv + h] = (float)exp(-exp((double)a_log[h]) * sp);
    gb[t * 2 * nv + nv + h] = (float)(1.0 / (1.0 + exp(-(double)ab[t * 2 * nv + nv + h])));
}

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

// The delta-rule scan, in place on S [nv][dk][dv] (dk = 128). A warp owns a
// column (value head h, value dim dj), a lane 4 of its cells. Per token:
// kv = S . k, delta = (v - g kv) beta, S = g S + k delta, o = q . S.
extern "C" __global__ void hy_gdn_scan(const float* __restrict__ cq, const float* __restrict__ gb, float* S,
                                       float* __restrict__ o, int n, int c_dim, int nv, int nk, int dk, int dv,
                                       float* __restrict__ ring, const float* __restrict__ qkv, int qstride, int kk)
{
    // blocks past the state's columns move the conv ring (the front kernel
    // that read it has finished): its last kk-1 rows of [ring, qkv rows]
    const int sblocks = (nv * dv + (blockDim.x >> 5) - 1) / (blockDim.x >> 5);
    if ((int)blockIdx.x >= sblocks) {
        const int c = ((int)blockIdx.x - sblocks) * blockDim.x + threadIdx.x;
        if (!ring || c >= c_dim || n <= 0) return;
        float nr[8];
        for (int j = 0; j < kk - 1 && j < 8; j++) {
            const int src = n - (kk - 1) + j;
            nr[j] = src >= 0 ? qkv[(u64)src * qstride + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        }
        for (int j = 0; j < kk - 1 && j < 8; j++) ring[(u64)j * c_dim + c] = nr[j];
        return;
    }
    const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (col >= nv * dv) return;
    const int lane = threadIdx.x & 31;
    const int h = col / dv, dj = col - h * dv;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk + lane * 4, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    float* sp = S + ((u64)h * dk + lane * 4) * dv + dj;
    float s0 = sp[0], s1 = sp[dv], s2 = sp[2 * dv], s3 = sp[3 * dv];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (u64)t * c_dim;
        const float4 kf = *(const float4*)(row + kofs);
        const float4 qf = *(const float4*)(row + qo);
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        const float kv = wsum(fmaf(s3, kf.w, fmaf(s2, kf.z, fmaf(s1, kf.y, __fmul_rn(s0, kf.x)))));
        const float delta = __fmul_rn(__fsub_rn(row[vo], __fmul_rn(g, kv)), beta);
        s0 = fmaf(kf.x, delta, __fmul_rn(g, s0));
        s1 = fmaf(kf.y, delta, __fmul_rn(g, s1));
        s2 = fmaf(kf.z, delta, __fmul_rn(g, s2));
        s3 = fmaf(kf.w, delta, __fmul_rn(g, s3));
        const float ov = wsum(fmaf(qf.w, s3, fmaf(qf.z, s2, fmaf(qf.y, s1, __fmul_rn(qf.x, s0)))));
        if (lane == 0 && o) o[(u64)t * nv * dv + (u64)h * dv + dj] = ov;
    }
    sp[0] = s0;
    sp[dv] = s1;
    sp[2 * dv] = s2;
    sp[3 * dv] = s3;
}

// of [n][nv][dv] = rms(o) * norm * gate(z), a block per (token, head), f64;
// gate = SiLU (Qwen3.5) or sigmoid (`sigmoid` set). f32 or f16 out.
extern "C" __global__ void hy_gdn_out(const float* __restrict__ o, const float* __restrict__ z, int zstride,
                                      const float* __restrict__ norm, void* __restrict__ of, int nv, int dv,
                                      double eps, int sigmoid, int half, u16* __restrict__ p0, u16* __restrict__ p1,
                                      u16* __restrict__ p2, int pstride)
{
    __shared__ double red[256];
    const u64 b = blockIdx.x;
    const u64 t = b / nv;
    const int hh = (int)(b - t * nv);
    const float* xr = o + b * dv;
    const float* zr = z + t * zstride + (u64)hh * dv;
    double ss = 0.0;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const double v = xr[i];
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double inv = 1.0 / sqrt(red[0] / (double)dv + eps);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const double zz = (double)zr[i];
        const double gate = sigmoid ? 1.0 / (1.0 + exp(-zz)) : zz / (1.0 + exp(-zz));
        const float v = (float)(((double)xr[i] * inv) * (double)norm[i] * gate);
        if (p0) put3(p0, p1, p2, t * pstride + (u64)hh * dv + i, v);
        else stf(of, b * dv + i, v, half);
    }
}

// ------------------------------------------------------------- attention
//
// qraw [B][nh][2 HD] (each head's query, then its gate) or [B][nh][HD]
// (`gated` = 0); k, v [B][nkv][HD]. The query and key heads are RMS-normed
// (when a weight is given), the first `half` x 2 dims rotated (NeoX halves),
// the queries written GROUP-major qg [kv head][B][head of the group][HD], the
// gates [B][nh][HD], the keys and values into mirror rows slot0 + b. A block
// a (head, row): blockIdx.x < nh a query head, else a key head.
extern "C" __global__ void hy_qk(const float* __restrict__ qraw, int qstride, const float* __restrict__ k,
                                 const float* __restrict__ v, int kvstride, const float* __restrict__ qn,
                                 const float* __restrict__ kn, float* __restrict__ qg, float* __restrict__ gate,
                                 float* __restrict__ Km, float* __restrict__ Vm, const float* __restrict__ tab,
                                 int nh, int nkv, int half, float eps, int gemma, int gated, int slot0_, int B,
                                 const int* __restrict__ ci)
{
    // the mirror row of the call's first row: from the call record when
    // given (a replayed graph), else the argument
    const int slot0 = ci ? ci[1] : slot0_;
    __shared__ float xs[HD];
    __shared__ float red[32];
    const int u = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;
    const bool isq = u < nh;
    const float* src;
    const float* nw;
    if (isq) {
        src = qraw + (size_t)b * qstride + (size_t)u * HD * (gated ? 2 : 1);
        nw = qn;
    } else {
        src = k + (size_t)b * kvstride + (size_t)(u - nh) * HD;
        nw = kn;
    }
    float s = 0.0f;
    for (int i = tid; i < HD; i += blockDim.x) {
        const float x = src[i];
        xs[i] = x;
        s = fmaf(x, x, s);
    }
    float inv = 1.0f;
    if (nw) {
        for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, o));
        if (lane == 0) red[warp] = s;
        __syncthreads();
        double tot = 0.0;
        for (int w = 0; w < NW; w++) tot += (double)red[w];
        inv = 1.0f / (float)sqrt(tot / (double)HD + (double)eps);
    }
    __syncthreads();
    if (nw)
        for (int i = tid; i < HD; i += blockDim.x) xs[i] = __fmul_rn(__fmul_rn(xs[i], inv), gemma ? 1.0f + nw[i] : nw[i]);
    __syncthreads();
    const float* t = tab + (size_t)b * half * 2;
    for (int d = tid; d < HD; d += blockDim.x) {
        float val;
        if (d < half) {
            const float sn = t[2 * d], c = t[2 * d + 1];
            val = __fsub_rn(__fmul_rn(xs[d], c), __fmul_rn(xs[d + half], sn));
        } else if (d < 2 * half) {
            const int i = d - half;
            const float sn = t[2 * i], c = t[2 * i + 1];
            val = __fadd_rn(__fmul_rn(xs[i], sn), __fmul_rn(xs[d], c));
        } else {
            val = xs[d];
        }
        if (isq) {
            const int g = u / HPK, j = u - g * HPK;
            qg[(((size_t)g * B + b) * HPK + j) * HD + d] = val;
            if (gated) gate[((size_t)b * nh + u) * HD + d] = src[HD + d];
        } else {
            const int g = u - nh;
            Km[((size_t)(slot0 + b) * nkv + g) * HD + d] = val;
            Vm[((size_t)(slot0 + b) * nkv + g) * HD + d] = v[(size_t)b * kvstride + (size_t)g * HD + d];
        }
    }
}

// Decode attention of B rows at once, row b seeing the mirror's rows
// [r0, r0 + n0 + b + 1) (the rows before the call, the call's rows up to its
// own). S [B][nh][ncap], Mx [B][nh][nbcap], P [B][nbcap][nh][HD],
// E [B][nbcap][nh]; a block `rb` key rows of one KV group of one row.
// The call record ci (when given) carries n0, the mirror row of the first
// row, rb, ncap and nbcap; the grid covers the largest key count a record
// may give, blocks past this call's keys leave at once.
extern "C" __global__ void hy_scores(const float* __restrict__ qg, const float* __restrict__ Km,
                                     float* __restrict__ S, float* __restrict__ Mx, int n0, int r0, int nkv,
                                     float scale, int rb, int ncap, int nbcap, int B, const int* __restrict__ ci)
{
    __shared__ float sm[32][HPK];
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        ncap = ci[3];
        nbcap = ci[4];
    }
    const int g = blockIdx.y, blk = blockIdx.x, b = blockIdx.z, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int n = n0 + b + 1;
    if (blk * rb >= n) return;
    float qv[HPK][DPL];
#pragma unroll
    for (int j = 0; j < HPK; j++)
#pragma unroll
        for (int t = 0; t < DPL; t++) qv[j][t] = qg[(((size_t)g * B + b) * HPK + j) * HD + lane * DPL + t];
    float mx[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) mx[j] = -INF;
    const int row0 = blk * rb, row1 = min(row0 + rb, n);
    float* Sb = S + (size_t)b * nh * ncap;
    for (int r = row0 + warp; r < row1; r += NW) {
        const float* kr = Km + ((size_t)(r0 + r) * nkv + g) * HD + lane * DPL;
        float kv[DPL];
#pragma unroll
        for (int t = 0; t < DPL; t++) kv[t] = kr[t];
#pragma unroll
        for (int j = 0; j < HPK; j++) {
            float a = 0.0f;
#pragma unroll
            for (int t = 0; t < DPL; t++) a = fmaf(qv[j][t], kv[t], a);
            for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(FULL, a, o));
            a = __fmul_rn(a, scale);
            if (lane == 0) Sb[(size_t)(g * HPK + j) * ncap + r] = a;
            mx[j] = fmaxf(mx[j], a);
        }
    }
    if (lane == 0)
        for (int j = 0; j < HPK; j++) sm[warp][j] = mx[j];
    __syncthreads();
    if (tid < HPK) {
        float m = -INF;
        for (int w = 0; w < NW; w++) m = fmaxf(m, sm[w][tid]);
        Mx[((size_t)b * nh + g * HPK + tid) * nbcap + blk] = m;
    }
}

extern "C" __global__ void hy_values(const float* __restrict__ S, const float* __restrict__ Mx,
                                     const float* __restrict__ Vm, float* __restrict__ P, float* __restrict__ E,
                                     int n0, int r0, int nkv, int rb, int ncap, int nbcap, int B,
                                     const int* __restrict__ ci)
{
    __shared__ float sp[8][HD];
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        ncap = ci[3];
        nbcap = ci[4];
    }
    __shared__ float se[8][HPK];
    const int g = blockIdx.y, blk = blockIdx.x, b = blockIdx.z, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int n = n0 + b + 1;
    const int nblk = (n + rb - 1) / rb;
    if (blk >= nblk) return;
    const float* Sb = S + (size_t)b * nh * ncap;
    float mx[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        float m = -INF;
        for (int q = lane; q < nblk; q += 32) m = fmaxf(m, Mx[((size_t)b * nh + g * HPK + j) * nbcap + q]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        mx[j] = m;
    }
    float acc[HPK][DPL];
    float es[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        es[j] = 0.0f;
#pragma unroll
        for (int t = 0; t < DPL; t++) acc[j][t] = 0.0f;
    }
    const int row0 = blk * rb, row1 = min(row0 + rb, n);
    for (int r = row0 + warp; r < row1; r += NW) {
        const float* vr = Vm + ((size_t)(r0 + r) * nkv + g) * HD + lane * DPL;
        float vv[DPL];
#pragma unroll
        for (int t = 0; t < DPL; t++) vv[t] = vr[t];
#pragma unroll
        for (int j = 0; j < HPK; j++) {
            const float e = expf(__fsub_rn(Sb[(size_t)(g * HPK + j) * ncap + r], mx[j]));
            es[j] = __fadd_rn(es[j], e);
#pragma unroll
            for (int t = 0; t < DPL; t++) acc[j][t] = fmaf(e, vv[t], acc[j][t]);
        }
    }
    float* Pb = P + (size_t)b * nbcap * nh * HD;
    float* Eb = E + (size_t)b * nbcap * nh;
    if (lane == 0)
        for (int j = 0; j < HPK; j++) se[warp][j] = es[j];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        __syncthreads();
#pragma unroll
        for (int t = 0; t < DPL; t++) sp[warp][lane * DPL + t] = acc[j][t];
        __syncthreads();
        for (int d = tid; d < HD; d += blockDim.x) {
            float s = 0.0f;
            for (int w = 0; w < NW; w++) s = __fadd_rn(s, sp[w][d]);
            Pb[((size_t)blk * nh + g * HPK + j) * HD + d] = s;
        }
    }
    if (tid < HPK) {
        float s = 0.0f;
        for (int w = 0; w < NW; w++) s = __fadd_rn(s, se[w][tid]);
        Eb[(size_t)blk * nh + g * HPK + tid] = s;
    }
}

// ao [B][nh][HD] = sum of the blocks' partials / sum of their weights, times
// sigmoid(gate) where the layer is gated. A block a (head, row).
extern "C" __global__ void hy_finish(const float* __restrict__ P, const float* __restrict__ E,
                                     const float* __restrict__ gate, u16* __restrict__ p0, u16* __restrict__ p1,
                                     u16* __restrict__ p2, int pstride, int n0, int rb, int nh, int nbcap,
                                     const int* __restrict__ ci)
{
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        nbcap = ci[4];
    }
    const int h = blockIdx.x, b = blockIdx.y;
    const int n = n0 + b + 1;
    const int nblk = (n + rb - 1) / rb;
    const float* Pb = P + (size_t)b * nbcap * nh * HD;
    const float* Eb = E + (size_t)b * nbcap * nh;
    float tot = 0.0f;
    for (int q = 0; q < nblk; q++) tot = __fadd_rn(tot, Eb[(size_t)q * nh + h]);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float s = 0.0f;
        for (int q = 0; q < nblk; q++) s = __fadd_rn(s, Pb[((size_t)q * nh + h) * HD + d]);
        float a = tot > 0.0f ? __fdiv_rn(s, tot) : s;
        if (gate) a = __fmul_rn(a, sigm(gate[((size_t)b * nh + h) * HD + d]));
        put3(p0, p1, p2, (size_t)b * pstride + (size_t)h * HD + d, a);
    }
}

// The prompt path's attention output back in position-major order with the
// element-wise gate: ao [B][nh][HD] = aog [kv head][B][head of the group][HD]
// * isum * sigmoid(gate); f32, or f16 for the output projection's GEMM.
extern "C" __global__ void hy_attn_out(const float* __restrict__ aog, const float* __restrict__ isum,
                                       const float* __restrict__ gate, void* __restrict__ ao, int nh, int B, int half)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * nh * HD) return;
    const int d = (int)(i % HD);
    const size_t bh = i / HD;
    const int h = (int)(bh % nh), b = (int)(bh / nh);
    const int g = h / HPK, j = h - g * HPK;
    float f = isum[(size_t)g * B * HPK + (size_t)b * HPK + j];
    float v = aog[(((size_t)g * B + b) * HPK + j) * HD + d] * f;
    if (gate) v *= sigm(gate[i]);
    stf(ao, i, v, half);
}

// Greedy pick of every row of logits [B][rows]: the first maximum.
extern "C" __global__ void hy_argmax(const float* __restrict__ lg, int rows, int* __restrict__ out)
{
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* r = lg + (size_t)blockIdx.x * rows;
    float m = -INF;
    int mi = 0x7fffffff;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        const float v = r[i];
        if (v > m || (v == m && i < mi)) {
            m = v;
            mi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(FULL, m, o);
        const int oi = __shfl_xor_sync(FULL, mi, o);
        if (ov > m || (ov == m && oi < mi)) {
            m = ov;
            mi = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        bv[warp] = m;
        bi[warp] = mi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++)
            if (bv[w] > m || (bv[w] == m && bi[w] < mi)) {
                m = bv[w];
                mi = bi[w];
            }
        out[blockIdx.x] = mi;
    }
}

// ------------------------------------------------- tensor-core small path
//
// The q4tp weights of this arm are repacked for `mma.sync.m16n8k16` (see
// `repack_q4tp` in hyb.rs): for 16-row tile t and 32-column group g, 32 lanes
// x 8 bytes; lane l = 4 gid + tig holds X0 (k 0..15) and X1 (k 16..31), word
// X_s's nibbles m and m + 4 the low and high element of the A register a_m
// of step s (a0: row gid, k 16s + 2tig; a1: row gid + 8; a2: row gid,
// k 16s + 2tig + 8; a3: row gid + 8). Codes [t][g][16 rows]; the ladder as
// stored, [row][32].
//
// The input rows come as NPL bf16 planes (`put3`): x = p0 + p1 (+ p2). A
// weight (nibble - 8) is exact in bf16, a product exact in the f32
// accumulator.

// A register a_m of a repacked word: (nibble - 8) as bf16x2. 128 + n is
// 0x4300 | n; one fma takes 136 off both halves.
__device__ __forceinline__ u32 a_reg(u32 X, int m) {
    u32 h;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(h) : "r"(X >> (4 * m)), "r"(0x000F000Fu), "r"(0x43004300u));
    u32 r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h), "r"(0x3F803F80u), "r"(0xC308C308u));
    return r;
}

__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A block of 32 KS threads takes TB tiles; its KS warps take contiguous
// ranges of the groups, every warp all TB tiles of its range, and the KS
// partials are summed in order. NT n tiles of 8 input rows (rows past
// `nrows` are not read). The arithmetic of an output depends on KS alone,
// never on TB, NT or the window's row count.
template <int NT, int TB, int KS>
__device__ __forceinline__ void mmp_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ x0,
                                         const u16* __restrict__ x1, const u16* __restrict__ x2, int xstride,
                                         int nrows, int cols)
{
    __shared__ float2 par_s[16 * TB];
    __shared__ float part[KS][TB][NT][32][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int tile0 = ((int)blockIdx.x - jb.blk0) * TB;
    const int rows = jb.rows;
    const int ntile = (rows + 15) >> 4;
    // a row's ladder is exp2(lo + code * step): its two numbers, not its 32 rungs
    for (int i = tid; i < 16 * TB; i += blockDim.x) {
        const int rr = tile0 * 16 + i;
        par_s[i] = rr < rows ? ((const float2*)jb.aux1)[rr] : make_float2(0.0f, 0.0f);
    }
    __syncthreads();
    const int gid = lane >> 2, tig = lane & 3;
    const int gpr = cols >> 5;
    const int per = (gpr + KS - 1) / KS;
    const int g0 = min(gpr, warp * per), g1 = min(gpr, g0 + per);
    float acc[TB][NT][4];
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[b][t][i] = 0.0f;
    const uint2* T = (const uint2*)jb.w;
    const u8* Cc = (const u8*)jb.aux0;
    for (int g = g0; g < g1; g++) {
        // B fragments: [n tile][plane][4 words]
        u32 bx[NT][NPL][4];
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const int j = t * 8 + gid;
            if (j < nrows) {
                const size_t xo = (size_t)j * xstride + k0;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    bx[t][0][q] = *(const u32*)(x0 + xo + 8 * q);
                    bx[t][1][q] = *(const u32*)(x1 + xo + 8 * q);
                    if (NPL > 2) bx[t][NPL - 1][q] = *(const u32*)(x2 + xo + 8 * q);
                }
            } else {
#pragma unroll
                for (int pl = 0; pl < NPL; pl++)
#pragma unroll
                    for (int q = 0; q < 4; q++) bx[t][pl][q] = 0u;
            }
        }
#pragma unroll
        for (int b = 0; b < TB; b++) {
            const int tile = tile0 + b;
            if (tile >= ntile) break;
            const size_t tg = (size_t)tile * gpr + g;
            const uint2 X = T[tg * 32 + lane];
            const float2 pa = par_s[b * 16 + gid], pb = par_s[b * 16 + gid + 8];
            const float sa = exp2f(fmaf((float)Cc[tg * 16 + gid], pa.y, pa.x));
            const float sb = exp2f(fmaf((float)Cc[tg * 16 + gid + 8], pb.y, pb.x));
            const u32 a00 = a_reg(X.x, 0), a01 = a_reg(X.x, 1), a02 = a_reg(X.x, 2), a03 = a_reg(X.x, 3);
            const u32 a10 = a_reg(X.y, 0), a11 = a_reg(X.y, 1), a12 = a_reg(X.y, 2), a13 = a_reg(X.y, 3);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t * 8 >= nrows) break;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int pl = 0; pl < NPL; pl++) {
                    mma16816(c, a00, a01, a02, a03, bx[t][pl][0], bx[t][pl][1]);
                    mma16816(c, a10, a11, a12, a13, bx[t][pl][2], bx[t][pl][3]);
                }
                acc[b][t][0] = fmaf(sa, c[0], acc[b][t][0]);
                acc[b][t][1] = fmaf(sa, c[1], acc[b][t][1]);
                acc[b][t][2] = fmaf(sb, c[2], acc[b][t][2]);
                acc[b][t][3] = fmaf(sb, c[3], acc[b][t][3]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) part[warp][b][t][lane][i] = acc[b][t][i];
    __syncthreads();
    if (jb.kind & 0x100) {
        // a gate/up pair (rows of a tile: 8 of gate, then the same 8 of up):
        // silu(gate) * up of output row (tile, gid) into the planes y, res
        u16* q0 = (u16*)jb.y;
        u16* q1 = (u16*)jb.res;
        for (int o = tid; o < TB * NT * 32 * 2; o += blockDim.x) {
            const int i = o & 1, ln = (o >> 1) & 31, t = (o >> 6) % NT, b = (o >> 6) / NT;
            const int gd = ln >> 2, tg = ln & 3;
            const int j = t * 8 + 2 * tg + i;
            const int r = (tile0 + b) * 8 + gd;
            if (j >= nrows || 2 * r >= rows) continue;
            float g = part[0][b][t][ln][i], u = part[0][b][t][ln][i + 2];
#pragma unroll
            for (int k = 1; k < KS; k++) {
                g = __fadd_rn(g, part[k][b][t][ln][i]);
                u = __fadd_rn(u, part[k][b][t][ln][i + 2]);
            }
            const float v = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u);
            put3(q0, q1, q1, (size_t)j * jb.ystride + r, v);
        }
        return;
    }
    float* y = (float*)jb.y;
    const float* res = (const float*)jb.res;
    for (int o = tid; o < TB * NT * 32 * 4; o += blockDim.x) {
        const int i = o & 3, ln = (o >> 2) & 31, t = (o >> 7) % NT, b = (o >> 7) / NT;
        const int gd = ln >> 2, tg = ln & 3;
        const int j = t * 8 + 2 * tg + (i & 1);
        const int r = (tile0 + b) * 16 + gd + (i < 2 ? 0 : 8);
        if (j >= nrows || r >= rows) continue;
        float v = part[0][b][t][ln][i];
#pragma unroll
        for (int k = 1; k < KS; k++) v = __fadd_rn(v, part[k][b][t][ln][i]);
        const size_t at = (size_t)j * jb.ystride + r;
        y[at] = res ? __fadd_rn(res[at], v) : v;
    }
}

#define MMP(NT, TB, KS)                                                                                         \
    extern "C" __global__ void __launch_bounds__(32 * KS) hy_mmp_##NT##_##TB##_##KS(                            \
        const HyJob* __restrict__ j, int nj, const u16* __restrict__ x0, const u16* __restrict__ x1,              \
        const u16* __restrict__ x2, int xstride, int n, int cols)                                               \
    {                                                                                                           \
        mmp_body<NT, TB, KS>(j, nj, x0, x1, x2, xstride, n, cols);                                              \
    }
MMP(1, 1, 8)
MMP(1, 2, 8)
MMP(2, 1, 8)
MMP(2, 2, 8)
MMP(1, 1, 16)
MMP(1, 2, 16)
MMP(2, 1, 16)
MMP(2, 2, 16)

// f32 rows [n][xstride] into the three planes [n][ostride] (what the
// producers below do in place; for inputs no kernel of this arm makes).
extern "C" __global__ void hy_split(const float* __restrict__ x, int xstride, int cols, u16* __restrict__ p0,
                                    u16* __restrict__ p1, u16* __restrict__ p2, int ostride, int n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * cols) return;
    const int j = (int)(i / cols), c = (int)(i - (size_t)j * cols);
    put3(p0, p1, p2, (size_t)j * ostride + c, x[(size_t)j * xstride + c]);
}

// A repacked q4tp matrix back to a dense f16 plane [rows][cols] (the prompt
// path's GEMM operand). A thread a (tile, group, lane): its 16 weights.
extern "C" __global__ void hy_expand_p(const uint2* __restrict__ T, const u8* __restrict__ Cc,
                                       const float* __restrict__ lad, u16* __restrict__ W, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const uint2 X = T[idx];
    const int ra = tile * 16 + gid, rb = ra + 8;
    const float2* par = (const float2*)lad;
    const float sa = ra < rows ? exp2f(fmaf((float)Cc[tg * 16 + gid], par[ra].y, par[ra].x)) : 0.0f;
    const float sb = rb < rows ? exp2f(fmaf((float)Cc[tg * 16 + gid + 8], par[rb].y, par[rb].x)) : 0.0f;
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const u32 w = s ? X.y : X.x;
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = (m & 1) ? rb : ra;
            if (r >= rows) continue;
            const float sc = (m & 1) ? sb : sa;
            const int k = g * 32 + 16 * s + 2 * tig + ((m & 2) ? 8 : 0);
            const float lo = (float)((w >> (4 * m)) & 15u) - 8.0f;
            const float hi = (float)((w >> (4 * m + 16)) & 15u) - 8.0f;
            const u32 pk = (u32)f2h(lo * sc) | ((u32)f2h(hi * sc) << 16);
            *(u32*)(W + (size_t)r * cols + k) = pk;
        }
    }
}

// ---------------------------------------------------- GDN, one front launch
//
// Conv + SiLU of every channel, the L2 norm of the q and k heads (q also over
// sqrt(dk)), the ring after the rows, and the gates, in f32: a block of DH
// threads a head-wide unit of channels (units 0..2nk-1 the q and k heads,
// then the v heads), its thread a channel for all n rows; block `units` the
// gates of every row. The rows of a window run as the rows of one-row steps.
extern "C" __global__ void hy_gdn_front(const float* __restrict__ qkv, int qstride, float* __restrict__ ring,
                                        const float* __restrict__ taps, float* __restrict__ cq, int n, int c_dim,
                                        int kk, int nk, int dk, const float* __restrict__ xn, int hidden,
                                        const float* __restrict__ wa, const float* __restrict__ wb,
                                        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                        float* __restrict__ gb, int nv)
{
    // a block a (unit, row): blockIdx.y is the row; the ring is read here and
    // moved by `hy_gdn_ring` after this launch
    __shared__ float red[2][32];
    const int u = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;
    const int units = c_dim / dk;
    if (u >= units) {
        // the gates of head h = u - units for row t: the a and b projections
        // of the normed row (f32 weights), then
        // gb[t][h] = exp(-exp(a_log) softplus(a + dt_bias)), gb[t][nv + h] = sigmoid(b)
        const int h = u - units;
        const float* ra = wa + (size_t)h * hidden;
        const float* rb = wb + (size_t)h * hidden;
        const float* x = xn + (size_t)t * hidden;
        float sa = 0.0f, sb = 0.0f;
        for (int c = tid; c < hidden; c += blockDim.x) {
            const float xv = x[c];
            sa = fmaf(ra[c], xv, sa);
            sb = fmaf(rb[c], xv, sb);
        }
        for (int o = 16; o > 0; o >>= 1) {
            sa = __fadd_rn(sa, __shfl_xor_sync(FULL, sa, o));
            sb = __fadd_rn(sb, __shfl_xor_sync(FULL, sb, o));
        }
        if (lane == 0) {
            red[0][warp] = sa;
            red[1][warp] = sb;
        }
        __syncthreads();
        if (tid == 0) {
            float a = 0.0f, b = 0.0f;
            for (int w = 0; w < NW; w++) {
                a = __fadd_rn(a, red[0][w]);
                b = __fadd_rn(b, red[1][w]);
            }
            const float x2 = __fadd_rn(a, dt_bias[h]);
            const float sp = x2 > 20.0f ? x2 : log1pf(expf(x2));
            gb[(size_t)t * 2 * nv + h] = expf(-__fmul_rn(expf(a_log[h]), sp));
            gb[(size_t)t * 2 * nv + nv + h] = 1.0f / (1.0f + expf(-b));
        }
        return;
    }
    const int c = u * dk + tid;
    const bool live = tid < dk;
    const bool qk = u < 2 * nk;
    float v = 0.0f;
    if (live) {
        const float* tp = taps + (size_t)c * kk;
        float acc = __fmul_rn(qkv[(size_t)t * qstride + c], tp[kk - 1]);
        for (int j = 0; j < kk - 1; j++) {
            const int src = t - (kk - 1 - j);
            const float r = src >= 0 ? qkv[(size_t)src * qstride + c] : ring[(size_t)(kk - 1 + src) * c_dim + c];
            acc = fmaf(r, tp[j], acc);
        }
        v = __fdiv_rn(acc, __fadd_rn(1.0f, expf(-acc)));
    }
    if (qk) {
        float s = v * v;
        for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, o));
        if (lane == 0) red[0][warp] = s;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < NW; w++) tot = __fadd_rn(tot, red[0][w]);
        float inv = 1.0f / sqrtf(__fadd_rn(tot, 1e-6f));
        if (u < nk) inv = __fdiv_rn(inv, sqrtf((float)dk));
        v = __fmul_rn(v, inv);
    }
    if (live) cq[(size_t)t * c_dim + c] = v;
}

// of [n][nv][dv] = rms(o) * norm * silu(z) (sigmoid with `sigmoid`), f32, a
// block of dv threads a (row, head); into the planes, or f16 / f32 `of`.
extern "C" __global__ void hy_gdn_out32(const float* __restrict__ o, const float* __restrict__ z, int zstride,
                                        const float* __restrict__ norm, void* __restrict__ of, int nv, int dv,
                                        float eps, int sigmoid, int half, u16* __restrict__ p0, u16* __restrict__ p1,
                                        u16* __restrict__ p2, int pstride)
{
    __shared__ float red[32];
    const size_t b = blockIdx.x;
    const size_t t = b / nv;
    const int hh = (int)(b - t * nv);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const float* xr = o + b * dv;
    const float* zr = z + t * zstride + (size_t)hh * dv;
    float s = 0.0f;
    for (int i = tid; i < dv; i += blockDim.x) s = fmaf(xr[i], xr[i], s);
    for (int k = 16; k > 0; k >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, k));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < NW; w++) tot = __fadd_rn(tot, red[w]);
    const float inv = 1.0f / sqrtf(__fadd_rn(__fdiv_rn(tot, (float)dv), eps));
    for (int i = tid; i < dv; i += blockDim.x) {
        const float zz = zr[i];
        const float gate = sigmoid ? 1.0f / (1.0f + expf(-zz)) : __fdiv_rn(zz, __fadd_rn(1.0f, expf(-zz)));
        const float v = __fmul_rn(__fmul_rn(__fmul_rn(xr[i], inv), norm[i]), gate);
        if (p0) put3(p0, p1, p2, t * pstride + (size_t)hh * dv + i, v);
        else stf(of, b * dv + i, v, half);
    }
}

// RMS norm of the small path's rows, spread over the card: block (j, chunk)
// sums the squares of the whole row j (every block the same sum, in the
// same order) and writes its 1024 columns, f32 into `out` (when given) and
// into the planes.
extern "C" __global__ void hy_rmsnorm2(const float* __restrict__ x, const float* __restrict__ w,
                                       float* __restrict__ out, int n, float eps, int gemma, u16* __restrict__ p0,
                                       u16* __restrict__ p1, u16* __restrict__ p2, int pstride)
{
    __shared__ float red[32];
    const int j = blockIdx.y, chunk = blockIdx.x;
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* xr = x + (size_t)j * n;
    float s = 0.0f;
    for (int i = tid * 4; i < n; i += T * 4) {
        const float4 v = *(const float4*)(xr + i);
        s = fmaf(v.x, v.x, s);
        s = fmaf(v.y, v.y, s);
        s = fmaf(v.z, v.z, s);
        s = fmaf(v.w, v.w, s);
    }
    for (int k = 16; k > 0; k >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, k));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = 1.0f / (float)sqrt(tot / (double)n + (double)eps);
    const int c1 = min(n, (chunk + 1) * 1024);
    for (int i = chunk * 1024 + tid; i < c1; i += T) {
        const float v = __fmul_rn(__fmul_rn(xr[i], inv), gemma ? 1.0f + w[i] : w[i]);
        if (out) out[(size_t)j * n + i] = v;
        if (p0) put3(p0, p1, p2, (size_t)j * pstride + i, v);
    }
}

// f32 matrices of few rows (the GDN gates): a block a row of a job, its
// threads over the columns, the window's rows j < nrows in registers, the
// block's partial sums added in a fixed order. y_j[r] = W[r] . x_j.
extern "C" __global__ void hy_mv32(const HyJob* __restrict__ jobs, int njobs, const float* __restrict__ x,
                                   int xstride, int nrows, int cols)
{
    __shared__ float red[8][16];
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int r = (int)blockIdx.x - jb.blk0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* W = (const float*)jb.w + (size_t)r * cols;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float wv = W[c];
#pragma unroll
        for (int j = 0; j < 16; j++)
            if (j < nrows) acc[j] = fmaf(wv, x[(size_t)j * xstride + c], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j >= nrows) break;
        float v = acc[j];
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
        if (lane == 0) red[warp][j] = v;
    }
    __syncthreads();
    if (tid < nrows) {
        float v = 0.0f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) v = __fadd_rn(v, red[w][tid]);
        float* y = (float*)jb.y;
        const size_t at = (size_t)tid * jb.ystride + r;
        y[at] = jb.res ? __fadd_rn(((const float*)jb.res)[at], v) : v;
    }
}

// ---------------------------------------- GDN glue of the prompt path, f32
// (a chunk of many rows: element- and head-parallel, unlike the front
// kernel, which walks a short window's rows in order).

extern "C" __global__ void hy_gdn_conv32(const float* __restrict__ qkv, int qstride, const float* __restrict__ ring,
                                         const float* __restrict__ taps, float* __restrict__ cq, int n, int c_dim,
                                         int kk)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * c_dim) return;
    const int t = (int)(i / c_dim);
    const int c = (int)(i - (u64)t * c_dim);
    const float* tp = taps + (u64)c * kk;
    float acc = __fmul_rn(qkv[(u64)t * qstride + c], tp[kk - 1]);
    for (int j = 0; j < kk - 1; j++) {
        const int src = t - (kk - 1 - j);
        const float v = src >= 0 ? qkv[(u64)src * qstride + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        acc = fmaf(v, tp[j], acc);
    }
    cq[i] = __fdiv_rn(acc, __fadd_rn(1.0f, expf(-acc)));
}

// L2 norm of the q and k heads in place, a block of dk threads a (row, head).
extern "C" __global__ void hy_gdn_prep32(float* __restrict__ cq, int c_dim, int nk, int dk)
{
    __shared__ float red[32];
    const u64 t = blockIdx.x / (2 * nk);
    const int g = (int)(blockIdx.x - t * (2 * nk));
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    float* xr = cq + t * c_dim + (u64)g * dk;
    float s = 0.0f;
    for (int i = tid; i < dk; i += blockDim.x) s = fmaf(xr[i], xr[i], s);
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, o));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < NW; w++) tot = __fadd_rn(tot, red[w]);
    float inv = 1.0f / sqrtf(__fadd_rn(tot, 1e-6f));
    if (g < nk) inv = __fdiv_rn(inv, sqrtf((float)dk));
    for (int i = tid; i < dk; i += blockDim.x) xr[i] = __fmul_rn(xr[i], inv);
}

extern "C" __global__ void hy_gdn_gb32(const float* __restrict__ ab, const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias, float* __restrict__ gb, int n, int nv)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nv) return;
    const u64 t = i / nv;
    const int h = (int)(i - t * nv);
    const float x = __fadd_rn(ab[t * 2 * nv + h], dt_bias[h]);
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    gb[t * 2 * nv + h] = expf(-__fmul_rn(expf(a_log[h]), sp));
    gb[t * 2 * nv + nv + h] = 1.0f / (1.0f + expf(-ab[t * 2 * nv + nv + h]));
}

// ------------------------------------------- GDN restore, every layer at once
//
// After a verify window kept rows 0..keep-1: every GDN layer's state from
// the copy the window started from, the kept rows replayed. tab[layer] =
// {S saved, S, conv rows, gates, ring saved, ring, raw rows}; grid.y the
// layer. The scan is `hy_gdn_scan`'s step, reading the saved state and
// writing the current one; the ring is the last kk-1 rows of
// [ring saved, raw rows 0..keep-1].
extern "C" __global__ void hy_gdn_restore_scan(const u64* __restrict__ tab, int keep, int c_dim, int nv, int nk,
                                               int dk, int dv)
{
    const u64* tb = tab + (size_t)blockIdx.y * 7;
    const float* Sin = (const float*)tb[0];
    float* Sout = (float*)tb[1];
    const float* cq = (const float*)tb[2];
    const float* gb = (const float*)tb[3];
    const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (col >= nv * dv) return;
    const int lane = threadIdx.x & 31;
    const int h = col / dv, dj = col - h * dv;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk + lane * 4, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    const u64 so = ((u64)h * dk + lane * 4) * dv + dj;
    const float* sp = Sin + so;
    float s0 = sp[0], s1 = sp[dv], s2 = sp[2 * dv], s3 = sp[3 * dv];
    for (int t = 0; t < keep; t++) {
        const float* row = cq + (u64)t * c_dim;
        const float4 kf = *(const float4*)(row + kofs);
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        const float kv = wsum(fmaf(s3, kf.w, fmaf(s2, kf.z, fmaf(s1, kf.y, __fmul_rn(s0, kf.x)))));
        const float delta = __fmul_rn(__fsub_rn(row[vo], __fmul_rn(g, kv)), beta);
        s0 = fmaf(kf.x, delta, __fmul_rn(g, s0));
        s1 = fmaf(kf.y, delta, __fmul_rn(g, s1));
        s2 = fmaf(kf.z, delta, __fmul_rn(g, s2));
        s3 = fmaf(kf.w, delta, __fmul_rn(g, s3));
    }
    float* wp = Sout + so;
    wp[0] = s0;
    wp[dv] = s1;
    wp[2 * dv] = s2;
    wp[3 * dv] = s3;
}

extern "C" __global__ void hy_gdn_restore_ring(const u64* __restrict__ tab, int keep, int qstride, int c_dim, int kk)
{
    const u64* tb = tab + (size_t)blockIdx.y * 7;
    const float* rs = (const float*)tb[4];
    float* ring = (float*)tb[5];
    const float* qkv = (const float*)tb[6];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim) return;
    for (int j = 0; j < kk - 1; j++) {
        const int src = keep - (kk - 1) + j;
        ring[(u64)j * c_dim + c] = src >= 0 ? qkv[(u64)src * qstride + c] : rs[(u64)(kk - 1 + src) * c_dim + c];
    }
}


// The prompt path's FFN middle from a gate/up pair GEMM: y [B][2 inter] in
// the pair's row order (tile t: 8 gate rows, then the same 8 up rows) into
// f16 xh [B][inter] = silu(gate) * up / 16.
extern "C" __global__ void hy_silu_pair_half(const float* __restrict__ y, u16* __restrict__ xh, int inter, int B)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)B * inter) return;
    const u64 b = i / inter;
    const int r = (int)(i - b * inter);
    const float* yr = y + b * 2 * inter;
    const int base = (r >> 3) * 16 + (r & 7);
    const float g = yr[base], u = yr[base + 8];
    xh[i] = f2h_sat(__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u) * 0.0625f);
}

// ---------------------------------------------------------- repack on card
//
// A q4tp matrix (or a gate/up pair, `pair`) into the tensor-core layout: a
// thread a (tile, group, lane). Logical row r is row r of A, or of a pair
// row (r & 15) < 8 ? A : B at (r >> 4) * 8 + (r & 7). nib*: the payload's
// nibble words [row][group][4]; codes*: its 5-bit code rows, `stride` bytes
// each. Rows past `rows`: nibbles 8 (a zero weight), code 0.
__device__ __forceinline__ void rp_src(int r, int pair, const u32* nA, const u8* cA, const u32* nB, const u8* cB,
                                       const u32** n, const u8** cc, int* sr)
{
    if (pair && (r & 15) >= 8) {
        *n = nB;
        *cc = cB;
    } else {
        *n = nA;
        *cc = cA;
    }
    *sr = pair ? (r >> 4) * 8 + (r & 7) : r;
}

extern "C" __global__ void hy_repack(const u32* __restrict__ nibA, const u8* __restrict__ codesA,
                                     const u32* __restrict__ nibB, const u8* __restrict__ codesB, int stride,
                                     int pair, uint2* __restrict__ T, u8* __restrict__ Cc, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 wa[4], wb[4];
    for (int h = 0; h < 2; h++) {
        const int r = tile * 16 + gid + 8 * h;
        u32* w = h ? wb : wa;
        if (r < rows) {
            const u32* n;
            const u8* cc;
            int sr;
            rp_src(r, pair, nibA, codesA, nibB, codesB, &n, &cc, &sr);
            const u32* p = n + ((size_t)sr * gpr + g) * 4;
            for (int q = 0; q < 4; q++) w[q] = p[q];
        } else {
            for (int q = 0; q < 4; q++) w[q] = 0x88888888u;
        }
    }
    uint2 X;
    for (int s = 0; s < 2; s++) {
        u32 x = 0;
        for (int m = 0; m < 4; m++) {
            const u32 word = (m & 1) ? wb[2 * s + (m >> 1)] : wa[2 * s + (m >> 1)];
            const u32 byte = (word >> (8 * tig)) & 0xFFu;
            x |= (byte & 0xFu) << (4 * m);
            x |= (byte >> 4) << (4 * m + 16);
        }
        if (s) X.y = x;
        else X.x = x;
    }
    T[idx] = X;
    if (lane < 16) {
        const int r = tile * 16 + lane;
        u8 code = 0;
        if (r < rows) {
            const u32* n;
            const u8* cc;
            int sr;
            rp_src(r, pair, nibA, codesA, nibB, codesB, &n, &cc, &sr);
            const u8* cr = cc + (size_t)sr * stride;
            const int bit = g * 5, b = bit >> 3, sh = bit & 7;
            u32 v = cr[b];
            if (sh > 3) v |= (u32)cr[b + 1] << 8;
            code = (u8)((v >> sh) & 31u);
        }
        Cc[tg * 16 + lane] = code;
    }
}
