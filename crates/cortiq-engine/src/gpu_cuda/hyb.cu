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
// rows of a call one gate block of `hy_gdn_front` takes
#define GR 4
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
__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
// 8 f16 of a 16-byte aligned row slice as f32
__device__ __forceinline__ void h8(const u16* p, float* o) {
    const uint4 w = *(const uint4*)p;
    const u32 a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        o[2 * i] = h2f((u16)(a[i] & 0xFFFFu));
        o[2 * i + 1] = h2f((u16)(a[i] >> 16));
    }
}

// A lane's DPL halves of a mirror row as floats: eight a 16-byte load when
// DPL is a multiple of 8 (heads of 256 and 512), else two a 4-byte one (MLA's
// heads of 192: DPL 6).
__device__ __forceinline__ void hdpl(const u16* p, float* o) {
#if DPL % 8 == 0
#pragma unroll
    for (int t = 0; t < DPL; t += 8) h8(p + t, o + t);
#else
#pragma unroll
    for (int t = 0; t < DPL; t += 2) {
        const u32 w = *(const u32*)(p + t);
        o[t] = h2f((u16)(w & 0xFFFFu));
        o[t + 1] = h2f((u16)(w >> 16));
    }
#endif
}

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
    int xoff;  // elements into the input planes this job's rows start at (a MoE slot's own input)
    int nr;    // > 0: this job's input rows (fewer than the launch's), a grouped expert's
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

// The delta-rule step with a lane a state COLUMN (value head h, value dim
// j) for dk = 128: a block of 4 warps a (head, 32 columns), warp w the key
// rows [32 w, 32 w + 32), so a warp reads 32 consecutive floats of a key
// row. kv and o are summed in a fixed order (two halves in the warp, then
// the four warps), the same in the scan and in the restore: a restored state
// is the bits the plain decode would hold. Rows `n` from `Sin` into `Sout`
// (the same buffer for a scan; n <= 16); `ssave` (if given) gets the state before
// the rows, `o` (if given) the outputs.
__device__ __forceinline__ void gdn_scan4_body(const float* __restrict__ cq, const float* __restrict__ gb,
                                               const float* Sin, float* Sout, float* __restrict__ o, int n,
                                               int c_dim, int nv, int nk, int dv, float* __restrict__ ssave)
{
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int wpr = dv >> 5;
    const int h = blockIdx.x / wpr, j = (blockIdx.x - h * wpr) * 32 + lane;
    const int ko = h / (nv / nk), kd = nk * 128;
    const size_t base = ((size_t)h * 128 + w * 32) * dv + j;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = Sin[base + (size_t)i * dv];
    if (ssave) {
#pragma unroll
        for (int i = 0; i < 32; i++) ssave[base + (size_t)i * dv] = s[i];
    }
    // every row's k and q heads, gates and this column's v first (n <= 16):
    // the row loop then reads only shared memory and registers
    __shared__ float ks[16][128], qs[16][128];
    __shared__ float gbs[16][2];
    __shared__ float part[2][4][32];
    const int nn = min(n, 16);
    for (int t = w; t < nn; t += 4) {
        const float* row = cq + (size_t)t * c_dim;
        *(float4*)(&ks[t][lane * 4]) = *(const float4*)(row + kd + (size_t)ko * 128 + lane * 4);
        *(float4*)(&qs[t][lane * 4]) = *(const float4*)(row + (size_t)ko * 128 + lane * 4);
        if (lane == 0) {
            gbs[t][0] = gb[(size_t)t * 2 * nv + h];
            gbs[t][1] = gb[(size_t)t * 2 * nv + nv + h];
        }
    }
    float vv[16];
#pragma unroll
    for (int t = 0; t < 16; t++) vv[t] = t < nn ? cq[(size_t)t * c_dim + 2 * kd + (size_t)h * dv + j] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t >= nn) break;
        const float g = gbs[t][0], beta = gbs[t][1];
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
            a0 = fmaf(s[i], ks[t][w * 32 + i], a0);
            a1 = fmaf(s[i + 1], ks[t][w * 32 + i + 1], a1);
        }
        part[0][w][lane] = __fadd_rn(a0, a1);
        __syncthreads();
        const float kv = __fadd_rn(__fadd_rn(__fadd_rn(part[0][0][lane], part[0][1][lane]), part[0][2][lane]), part[0][3][lane]);
        const float delta = __fmul_rn(__fsub_rn(vv[t], __fmul_rn(g, kv)), beta);
        float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
            s[i] = fmaf(ks[t][w * 32 + i], delta, __fmul_rn(g, s[i]));
            s[i + 1] = fmaf(ks[t][w * 32 + i + 1], delta, __fmul_rn(g, s[i + 1]));
            b0 = fmaf(qs[t][w * 32 + i], s[i], b0);
            b1 = fmaf(qs[t][w * 32 + i + 1], s[i + 1], b1);
        }
        part[1][w][lane] = __fadd_rn(b0, b1);
        // part[0] was read before this barrier, so the next row may write it
        // after; part[1] is read by warp 0 before it reaches the next one
        __syncthreads();
        if (w == 0 && o)
            o[(size_t)t * nv * dv + (size_t)h * dv + j] =
                __fadd_rn(__fadd_rn(__fadd_rn(part[1][0][lane], part[1][1][lane]), part[1][2][lane]), part[1][3][lane]);
    }
#pragma unroll
    for (int i = 0; i < 32; i++) Sout[base + (size_t)i * dv] = s[i];
}

// The small path's scan for dk = 128 (`gdn_scan4_body`), 128 threads a
// block; blocks past the state's move the conv ring as `hy_gdn_scan`'s do.
extern "C" __global__ void __launch_bounds__(128) hy_gdn_scan4(const float* __restrict__ cq, const float* __restrict__ gb,
                                                              float* S, float* __restrict__ o, int n, int c_dim, int nv,
                                                              int nk, int dv, float* __restrict__ ring,
                                                              const float* __restrict__ qkv, int qstride, int kk,
                                                              float* __restrict__ ssave)
{
    const int sblocks = nv * (dv >> 5);
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
    gdn_scan4_body(cq, gb, S, S, o, n, c_dim, nv, nk, dv, ssave);
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
                                 u16* __restrict__ Km, u16* __restrict__ Vm, const float* __restrict__ tab,
                                 int nh, int nkv, int half, float eps, int gemma, int gated, int slot0_, int B,
                                 const int* __restrict__ ci, u16* __restrict__ qgh, const float* __restrict__ bq,
                                 const float* __restrict__ bk, const float* __restrict__ bv)
{
    // gemma bit 1: the head's RMS norm after the rotation (HunYuan), its
    // weight then applied to the rotated dims
    const bool late = (gemma & 2) != 0;
    // bit 2: Gemma-4's scale-less V norm (`vnorm_head`): every V head
    // RMS-normalized before the mirror keeps it
    const bool vnorm = (gemma & 4) != 0;
    gemma &= 1;
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
    const float* bias;
    if (isq) {
        src = qraw + (size_t)b * qstride + (size_t)u * HD * (gated ? 2 : 1);
        nw = qn;
        bias = bq ? bq + (size_t)u * HD : 0;
    } else {
        src = k + (size_t)b * kvstride + (size_t)(u - nh) * HD;
        nw = kn;
        bias = bk ? bk + (size_t)(u - nh) * HD : 0;
    }
    float s = 0.0f;
    for (int i = tid; i < HD; i += blockDim.x) {
        const float x = bias ? __fadd_rn(src[i], bias[i]) : src[i];
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
        for (int i = tid; i < HD; i += blockDim.x)
            xs[i] = late ? __fmul_rn(xs[i], inv) : __fmul_rn(__fmul_rn(xs[i], inv), gemma ? 1.0f + nw[i] : nw[i]);
    __syncthreads();
    float vinv = 1.0f;
    if (!isq && vnorm) {
        const int g = u - nh;
        float s2 = 0.0f;
        for (int i = tid; i < HD; i += blockDim.x) {
            float vv = v[(size_t)b * kvstride + (size_t)g * HD + i];
            if (bv) vv = __fadd_rn(vv, bv[(size_t)g * HD + i]);
            s2 = fmaf(vv, vv, s2);
        }
        for (int o = 16; o > 0; o >>= 1) s2 = __fadd_rn(s2, __shfl_xor_sync(FULL, s2, o));
        if (lane == 0) red[warp] = s2;
        __syncthreads();
        double tot = 0.0;
        for (int w = 0; w < NW; w++) tot += (double)red[w];
        vinv = (float)(1.0 / sqrt(tot / (double)HD + (double)eps));
        __syncthreads();
    }
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
        if (late && nw) val = __fmul_rn(val, gemma ? 1.0f + nw[d] : nw[d]);
        if (isq) {
            const int g = u / HPK, j = u - g * HPK;
            const size_t qi = (((size_t)g * B + b) * HPK + j) * HD + d;
            qg[qi] = val;
            if (qgh) qgh[qi] = f2h_sat(val);
            if (gated) gate[((size_t)b * nh + u) * HD + d] = src[HD + d];
        } else {
            // the mirrors keep f16 rows (half the bytes a decode step reads)
            const int g = u - nh;
            Km[((size_t)(slot0 + b) * nkv + g) * HD + d] = f2h_sat(val);
            const float vv = v[(size_t)b * kvstride + (size_t)g * HD + d];
            const float vb = bv ? __fadd_rn(vv, bv[(size_t)g * HD + d]) : vv;
            Vm[((size_t)(slot0 + b) * nkv + g) * HD + d] = f2h_sat(vnorm ? __fmul_rn(vb, vinv) : vb);
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
extern "C" __global__ void hy_scores(const float* __restrict__ qg, const u16* __restrict__ Km,
                                     float* __restrict__ S, float* __restrict__ Mx, int n0, int r0, int nkv,
                                     float scale, int rb, int ncap, int nbcap, int B, const int* __restrict__ ci)
{
    __shared__ float sm[32][HPK];
    // an exact mirror's record carries a sliding window in word 7 (word 5 is
    // the far-field flag of the other records)
    int win = 0;
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        ncap = ci[3];
        nbcap = ci[4];
        if (ci[5] == 0) win = ci[7];
    }
    const int g = blockIdx.y, blk = blockIdx.x, b = blockIdx.z, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int n = n0 + b + 1;
    if (blk * rb >= n) return;
    // keys left of the window take no weight
    const int lo = win > 0 ? max(0, n - win) : 0;
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
        if (r < lo) {
            if (lane == 0)
                for (int j = 0; j < HPK; j++) Sb[(size_t)(g * HPK + j) * ncap + r] = -INF;
            continue;
        }
        const u16* kr = Km + ((size_t)(r0 + r) * nkv + g) * HD + lane * DPL;
        float kv[DPL];
        hdpl(kr, kv);
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
                                     const u16* __restrict__ Vm, float* __restrict__ P, float* __restrict__ E,
                                     int n0, int r0, int nkv, int rb, int ncap, int nbcap, int B,
                                     const int* __restrict__ ci, const float* __restrict__ sink)
{
    __shared__ float sp[8][HD];
    // `extra`: a far field's block after the near ones (ci[5])
    int extra = 0;
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        ncap = ci[3];
        nbcap = ci[4];
        extra = ci[5];
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
        for (int q = lane; q < nblk + extra; q += 32) m = fmaxf(m, Mx[((size_t)b * nh + g * HPK + j) * nbcap + q]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        // a learned sink logit joins the max (and, block 0, the weights) with no value
        if (sink) m = fmaxf(m, sink[g * HPK + j]);
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
        const u16* vr = Vm + ((size_t)(r0 + r) * nkv + g) * HD + lane * DPL;
        float vv[DPL];
        hdpl(vr, vv);
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
        if (sink && blk == 0) {
            float mj = mx[0];
#pragma unroll
            for (int j = 1; j < HPK; j++)
                if (j == tid) mj = mx[j];
            s = __fadd_rn(s, expf(__fsub_rn(sink[g * HPK + tid], mj)));
        }
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
    int extra = 0;
    if (ci) {
        n0 = ci[0];
        rb = ci[2];
        nbcap = ci[4];
        extra = ci[5];
    }
    const int h = blockIdx.x, b = blockIdx.y;
    const int n = n0 + b + 1;
    const int nblk = (n + rb - 1) / rb + extra;
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

// The A register of half h of an int8 tile word: its two signed bytes as
// bf16x2, exactly. A byte's unsigned twin u = b + 128 makes the float
// 2^23 + u (0x4B0000uu), one subtraction leaves b; an integer of 8 bits is
// a bf16, so its float's upper half is it.
__device__ __forceinline__ u32 a_i8(u32 W, int h) {
    const u32 u = W ^ 0x80808080u;
    const float lo = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7440 | (2 * h))) - 8388736.0f;
    const float hi = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7440 | (2 * h + 1))) - 8388736.0f;
    return __byte_perm(__float_as_uint(lo), __float_as_uint(hi), 0x7632);
}

// A q1 tile lane's 16 sign bits (register r's two at bits r and r + 8)
// spread: the high byte to bits 16..23, so register r's pair sits at r, 16 + r.
__device__ __forceinline__ u32 q1_spread(u32 B) { return __byte_perm(B, 0u, 0x4140); }

// The A register r of a q1 tile lane from its spread bits D: bf16x2 +-1 (a
// set bit +1, a clear one -1; the sign bit is the clear bit's). A shift and a
// lop3.
__device__ __forceinline__ u32 a_q1(u32 D, int r) {
    return 0x3F803F80u | (~(D << (15 - r)) & 0x80008000u);
}

// The A register r of a ternary tile lane: two-bit codes c at bits 2r and
// 16 + 2r as bf16x2 c - 1 (128 + c is 0x4300 | c; one fma takes 129 off).
__device__ __forceinline__ u32 a_q1t(u32 W, int r) {
    const u32 h = ((W >> (2 * r)) & 0x00030003u) | 0x43004300u;
    u32 v;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(v) : "r"(h), "r"(0x3F803F80u), "r"(0xC301C301u));
    return v;
}

// The A register r of a q2tp tile lane: two-bit codes c at bits 2r and
// 16 + 2r as bf16x2 2c - 3 (the mid-rise (c - 1.5) s with the half in the
// scale) or, `aff`, c - 1; both exact.
__device__ __forceinline__ u32 a_q2(u32 W, int r, bool aff) {
    const u32 h = ((W >> (2 * r)) & 0x00030003u) | 0x43004300u;
    u32 v;
    if (aff) {
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(v) : "r"(h), "r"(0x3F803F80u), "r"(0xC301C301u));
    } else {
        u32 c;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(c) : "r"(h), "r"(0x3F803F80u), "r"(0xC300C300u));
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(v) : "r"(c), "r"(0x40004000u), "r"(0xC040C040u));
    }
    return v;
}

// The A register r (0..3 of a word) of a q4_tiled tile lane: nibbles at bits
// 8r, 8r + 4 as bf16x2 n - 8 (128 + n is 0x4300 | n; one fma takes 136 off).
__device__ __forceinline__ u32 a_q4g(u32 W, int r) {
    const u32 t = W >> (8 * r);
    const u32 h = (t & 0xFu) | ((t & 0xF0u) << 12) | 0x43004300u;
    u32 v;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(v) : "r"(h), "r"(0x3F803F80u), "r"(0xC308C308u));
    return v;
}

// tanh-approximated GELU (HF `gelu_pytorch_tanh`), `inference::gelu_tanh`
// term for term: GeGLU's activation in the pair epilogue and the prompt path.
__device__ __forceinline__ float gelu_t(float x) {
    const float c = 0.7978846f;
    const float x3 = __fmul_rn(__fmul_rn(__fmul_rn(0.044715f, x), x), x);
    const float t = tanhf(__fmul_rn(c, __fadd_rn(x, x3)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, t));
}

// Plane `pl` of an input pair (x = w0 + w1 as bf16 planes, two columns)
// under a column field f: (x * f) split again into the planes.
__device__ __forceinline__ u32 field_plane(u32 w0, u32 w1, float2 f, int pl) {
    float lo = __fadd_rn(__fadd_rn(0.0f, bf2f((u16)(w0 & 0xFFFFu))), bf2f((u16)(w1 & 0xFFFFu)));
    float hi = __fadd_rn(__fadd_rn(0.0f, bf2f((u16)(w0 >> 16))), bf2f((u16)(w1 >> 16)));
    lo = __fmul_rn(lo, f.x);
    hi = __fmul_rn(hi, f.y);
    u16 a = f2bf(lo), b = f2bf(hi);
    if (pl) {
        a = f2bf(__fsub_rn(lo, bf2f(a)));
        b = f2bf(__fsub_rn(hi, bf2f(b)));
    }
    return (u32)a | ((u32)b << 16);
}

__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A block of 32 KS threads takes TB tiles; its KS warps take contiguous
// ranges of the groups, every warp all TB tiles of its range, and the KS
// partials are summed in order. NT n tiles of 4 input rows: an n tile's
// column 2i is plane 0 of its row i, column 2i + 1 plane 1, so a decode row
// costs two mma a group, not four (rows past `nrows`, or past the job's own
// `nr`, are not read). The arithmetic of an output depends on KS alone,
// never on TB, NT, KD or the window's row count. KD 0 reads each job's
// codec from its kind; KD > 0 is a launch of that codec alone (1 = q2tp,
// either form), whose loop carries none of the others' branches (a decode
// row's q1 matvec: 156 against 86 us on an A5000).
#if NPL != 2
#error "the tensor-core tiles take two bf16 planes"
#endif
template <int NT, int TB, int KS, int KD>
__device__ __forceinline__ void mmp_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ x0,
                                         const u16* __restrict__ x1, const u16* __restrict__ x2, int xstride,
                                         int nrows_in, int cols)
{
    __shared__ float2 par_s[16 * TB];
    __shared__ float part[KS][TB][NT][32][2];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // the block's job: a binary search of the first blocks (a MoE launch has
    // a hundred slots and more, where a scan was a hundred dependent loads)
    int jx = 0, jhi = njobs - 1;
    while (jx < jhi) {
        const int mid = (jx + jhi + 1) >> 1;
        if ((int)blockIdx.x >= jobs[mid].blk0) jx = mid;
        else jhi = mid - 1;
    }
    const HyJob jb = jobs[jx];
    // an empty job (a dead MoE slot, a grouped launch's spare) has no blocks' work
    if (jb.rows <= 0) return;
    const int nrows = jb.nr > 0 ? min(nrows_in, jb.nr) : nrows_in;
    const int tile0 = ((int)blockIdx.x - jb.blk0) * TB;
    const int rows = jb.rows;
    const int ntile = (rows + 15) >> 4;
    // kind 5: int8 tiles, a row's scale its one number (aux1), the column
    // field (aux0, or none) taken into the input rows
    const int kk = jb.kind & 0xFF;
    const bool i8 = KD ? KD == 5 : (kk == 5 || kk == 15);
    // bit 0x800: the launch's rows come under the field already
    // (`hy_field_planes`), the job's at its `xoff`
    const float* cf = (i8 && !(jb.kind & 0x800)) ? (const float*)jb.aux0 : nullptr;
    // kind 15: a q8_2f gate/up pair, two column fields (gate's, then up's
    // `cols` later): a tile's up rows (gid + 8) take the input under up's
    const bool dual = i8 && kk == 15;
    // kind 3: q1 tiles (a lane's 16 sign bits a group), a (row, group)'s f16 scale in aux0;
    // kind 2: ternary tiles (a lane's 16 two-bit codes, value + 1), the same scales
    const bool q1 = KD ? KD == 3 : kk == 3;
    const bool q1t = KD ? KD == 2 : kk == 2;
    // kind 1 / 9: q2tp tiles (a lane's 16 two-bit codes), the q4tp ladder and
    // rung codes (rung 0 = zero); 1 = (c - 1.5) s, 9 = the affine (c - 1) s
    const bool q2 = KD ? KD == 1 : (kk == 1 || kk == 9);
    const bool q2a = q2 && kk == 9;
    // kind 10: q4_tiled tiles (a lane's 16 nibbles), a (row, group)'s f16 scale in aux0
    const bool q4g = KD ? KD == 10 : kk == 10;
    // a row's ladder is exp2(lo + code * step): its two numbers, not its 32 rungs
    for (int i = tid; i < 16 * TB; i += blockDim.x) {
        const int rr = tile0 * 16 + i;
        par_s[i] = (rr >= rows || q1 || q1t || q4g) ? make_float2(0.0f, 0.0f)
                   : i8               ? make_float2(((const float*)jb.aux1)[rr], 0.0f)
                                      : ((const float2*)jb.aux1)[rr];
    }
    __syncthreads();
    const int gid = lane >> 2, tig = lane & 3;
    const int gpr = cols >> 5;
    const int per = (gpr + KS - 1) / KS;
    const int g0 = min(gpr, warp * per), g1 = min(gpr, g0 + per);
    float acc[TB][NT][2];
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[b][t][0] = acc[b][t][1] = 0.0f;
    const uint2* T = (const uint2*)jb.w;
    const u8* Cc = (const u8*)jb.aux0;
    // UG groups' weight words and scales are read before any of them is
    // used (one codec's decode-row loop: q1 86 -> 80 us; the all-codec loop
    // ran slower with it); the arithmetic is the one-group loop's.
    constexpr int UG = (KD && NT == 1) ? 2 : 1;
    for (int gb = g0; gb < g1; gb += UG) {
        u32 wx[UG][TB][4];
        float wsa[UG][TB], wsb[UG][TB];
#pragma unroll
        for (int u = 0; u < UG; u++)
#pragma unroll
            for (int b = 0; b < TB; b++) {
                const int g = gb + u, tile = tile0 + b;
                wx[u][b][0] = wx[u][b][1] = wx[u][b][2] = wx[u][b][3] = 0u;
                wsa[u][b] = wsb[u][b] = 0.0f;
                if (g >= g1 || tile >= ntile) continue;
                const size_t tg = (size_t)tile * gpr + g;
                if (q4g) {
                    const uint2 X4 = ((const uint2*)jb.w)[tg * 32 + lane];
                    const u16* S1 = (const u16*)jb.aux0 + tg * 16;
                    wx[u][b][0] = X4.x;
                    wx[u][b][1] = X4.y;
                    wsa[u][b] = h2f(S1[gid]);
                    wsb[u][b] = h2f(S1[gid + 8]);
                } else if (q2) {
                    wx[u][b][0] = ((const u32*)jb.w)[tg * 32 + lane];
                    const float2 pa = par_s[b * 16 + gid], pb = par_s[b * 16 + gid + 8];
                    const int ca = Cc[tg * 16 + gid], cb = Cc[tg * 16 + gid + 8];
                    const float half = q2a ? 1.0f : 0.5f;
                    wsa[u][b] = ca ? half * exp2f(fmaf((float)(ca - 1), pa.y, pa.x)) : 0.0f;
                    wsb[u][b] = cb ? half * exp2f(fmaf((float)(cb - 1), pb.y, pb.x)) : 0.0f;
                } else if (q1t || q1) {
                    wx[u][b][0] = q1 ? q1_spread(((const u16*)jb.w)[tg * 32 + lane]) : ((const u32*)jb.w)[tg * 32 + lane];
                    const u16* S1 = (const u16*)jb.aux0 + tg * 16;
                    wsa[u][b] = h2f(S1[gid]);
                    wsb[u][b] = h2f(S1[gid + 8]);
                } else if (i8) {
                    const uint4 X8 = ((const uint4*)jb.w)[tg * 32 + lane];
                    wx[u][b][0] = X8.x;
                    wx[u][b][1] = X8.y;
                    wx[u][b][2] = X8.z;
                    wx[u][b][3] = X8.w;
                    wsa[u][b] = par_s[b * 16 + gid].x;
                    wsb[u][b] = par_s[b * 16 + gid + 8].x;
                } else {
                    const uint2 X = T[tg * 32 + lane];
                    const float2 pa = par_s[b * 16 + gid], pb = par_s[b * 16 + gid + 8];
                    wx[u][b][0] = X.x;
                    wx[u][b][1] = X.y;
                    wsa[u][b] = exp2f(fmaf((float)Cc[tg * 16 + gid], pa.y, pa.x));
                    wsb[u][b] = exp2f(fmaf((float)Cc[tg * 16 + gid + 8], pb.y, pb.x));
                }
            }
#pragma unroll
        for (int u = 0; u < UG; u++) {
            const int g = gb + u;
            if (g >= g1) break;
            // B fragments: [n tile][4 words], lane gid's column plane (gid & 1)
            // of the tile's input row gid >> 1 (bx2: under the up field)
            u32 bx[NT][4], bx2[NT][4];
            const int k0 = g * 32 + 2 * tig;
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const int j = t * 4 + (gid >> 1);
                if (j < nrows) {
                    const size_t xo = (size_t)jb.xoff + (size_t)j * xstride + k0;
                    if (!cf) {
                        const u16* xp = (gid & 1) ? x1 : x0;
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            bx[t][q] = *(const u32*)(xp + xo + 8 * q);
                            // a prescaled two-field pair: up's rows `cols` on
                            if (dual) bx2[t][q] = *(const u32*)(xp + xo + cols + 8 * q);
                        }
                    } else {
                        // x * field, split again into the planes: W x with W = int8 * rs * cf
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            const u32 w0 = *(const u32*)(x0 + xo + 8 * q), w1 = *(const u32*)(x1 + xo + 8 * q);
                            bx[t][q] = field_plane(w0, w1, *(const float2*)(cf + k0 + 8 * q), gid & 1);
                            if (dual) bx2[t][q] = field_plane(w0, w1, *(const float2*)(cf + cols + k0 + 8 * q), gid & 1);
                        }
                    }
                } else {
#pragma unroll
                    for (int q = 0; q < 4; q++) bx[t][q] = bx2[t][q] = 0u;
                }
            }
#pragma unroll
            for (int b = 0; b < TB; b++) {
                const int tile = tile0 + b;
                if (tile >= ntile) break;
                u32 a00, a01, a02, a03, a10, a11, a12, a13;
                const float sa = wsa[u][b], sb = wsb[u][b];
                const u32* W = wx[u][b];
                if (q4g) {
                    a00 = a_q4g(W[0], 0), a01 = a_q4g(W[0], 1), a02 = a_q4g(W[0], 2), a03 = a_q4g(W[0], 3);
                    a10 = a_q4g(W[1], 0), a11 = a_q4g(W[1], 1), a12 = a_q4g(W[1], 2), a13 = a_q4g(W[1], 3);
                } else if (q2) {
                    a00 = a_q2(W[0], 0, q2a), a01 = a_q2(W[0], 1, q2a), a02 = a_q2(W[0], 2, q2a), a03 = a_q2(W[0], 3, q2a);
                    a10 = a_q2(W[0], 4, q2a), a11 = a_q2(W[0], 5, q2a), a12 = a_q2(W[0], 6, q2a), a13 = a_q2(W[0], 7, q2a);
                } else if (q1t) {
                    a00 = a_q1t(W[0], 0), a01 = a_q1t(W[0], 1), a02 = a_q1t(W[0], 2), a03 = a_q1t(W[0], 3);
                    a10 = a_q1t(W[0], 4), a11 = a_q1t(W[0], 5), a12 = a_q1t(W[0], 6), a13 = a_q1t(W[0], 7);
                } else if (q1) {
                    a00 = a_q1(W[0], 0), a01 = a_q1(W[0], 1), a02 = a_q1(W[0], 2), a03 = a_q1(W[0], 3);
                    a10 = a_q1(W[0], 4), a11 = a_q1(W[0], 5), a12 = a_q1(W[0], 6), a13 = a_q1(W[0], 7);
                } else if (i8) {
                    a00 = a_i8(W[0], 0), a01 = a_i8(W[0], 1), a02 = a_i8(W[1], 0), a03 = a_i8(W[1], 1);
                    a10 = a_i8(W[2], 0), a11 = a_i8(W[2], 1), a12 = a_i8(W[3], 0), a13 = a_i8(W[3], 1);
                } else {
                    a00 = a_reg(W[0], 0), a01 = a_reg(W[0], 1), a02 = a_reg(W[0], 2), a03 = a_reg(W[0], 3);
                    a10 = a_reg(W[1], 0), a11 = a_reg(W[1], 1), a12 = a_reg(W[1], 2), a13 = a_reg(W[1], 3);
                }
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    if (t * 4 >= nrows) break;
                    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma16816(c, a00, a01, a02, a03, bx[t][0], bx[t][1]);
                    mma16816(c, a10, a11, a12, a13, bx[t][2], bx[t][3]);
                    if (dual) {
                        // the up rows (gid + 8) again, under their own field
                        float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma16816(c2, a00, a01, a02, a03, bx2[t][0], bx2[t][1]);
                        mma16816(c2, a10, a11, a12, a13, bx2[t][2], bx2[t][3]);
                        c[2] = c2[2];
                        c[3] = c2[3];
                    }
                    // the planes of input row 4t + tig: columns 2 tig, 2 tig + 1
                    acc[b][t][0] = fmaf(sa, __fadd_rn(c[0], c[1]), acc[b][t][0]);
                    acc[b][t][1] = fmaf(sb, __fadd_rn(c[2], c[3]), acc[b][t][1]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++) {
            part[warp][b][t][lane][0] = acc[b][t][0];
            part[warp][b][t][lane][1] = acc[b][t][1];
        }
    __syncthreads();
    if (jb.kind & 0x100) {
        // a gate/up pair (rows of a tile: 8 of gate, then the same 8 of up):
        // silu(gate) * up of output row (tile, gid) into the planes y, res
        u16* q0 = (u16*)jb.y;
        u16* q1 = (u16*)jb.res;
        for (int o = tid; o < TB * NT * 32; o += blockDim.x) {
            const int ln = o & 31, t = (o >> 5) % NT, b = (o >> 5) / NT;
            const int gd = ln >> 2, tg = ln & 3;
            const int j = t * 4 + tg;
            const int r = (tile0 + b) * 8 + gd;
            if (j >= nrows || 2 * r >= rows) continue;
            float g = part[0][b][t][ln][0], u = part[0][b][t][ln][1];
#pragma unroll
            for (int k = 1; k < KS; k++) {
                g = __fadd_rn(g, part[k][b][t][ln][0]);
                u = __fadd_rn(u, part[k][b][t][ln][1]);
            }
            // kind bit 0x200: GeGLU (tanh GELU), else SwiGLU
            const float act = (jb.kind & 0x200) ? gelu_t(g) : __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
            const float v = __fmul_rn(act, u);
            put3(q0, q1, q1, (size_t)j * jb.ystride + r, v);
        }
        return;
    }
    float* y = (float*)jb.y;
    const float* res = (const float*)jb.res;
    for (int o = tid; o < TB * NT * 32 * 2; o += blockDim.x) {
        const int i = o & 1, ln = (o >> 1) & 31, t = (o >> 6) % NT, b = (o >> 6) / NT;
        const int gd = ln >> 2, tg = ln & 3;
        const int j = t * 4 + tg;
        const int r = (tile0 + b) * 16 + gd + 8 * i;
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
        mmp_body<NT, TB, KS, 0>(j, nj, x0, x1, x2, xstride, n, cols);                                           \
    }
// one codec's window (KD as above)
#define MMPK(NT, TB, KS, KD)                                                                                    \
    extern "C" __global__ void __launch_bounds__(32 * KS) hy_mmpk_##NT##_##TB##_##KS##_##KD(                    \
        const HyJob* __restrict__ j, int nj, const u16* __restrict__ x0, const u16* __restrict__ x1,              \
        const u16* __restrict__ x2, int xstride, int n, int cols)                                               \
    {                                                                                                           \
        mmp_body<NT, TB, KS, KD>(j, nj, x0, x1, x2, xstride, n, cols);                                          \
    }
#define MMPK3(TB, KS, KD) MMPK(1, TB, KS, KD) MMPK(2, TB, KS, KD) MMPK(4, TB, KS, KD)
#define MMPK4(KD) MMPK3(1, 8, KD) MMPK3(2, 8, KD) MMPK3(1, 16, KD) MMPK3(2, 16, KD)
MMPK4(6)
MMPK4(5)
MMPK4(3)
MMPK4(2)
MMPK4(1)
MMPK4(10)
// windows of up to 4, 8 and 16 rows
MMP(1, 1, 8)
MMP(1, 2, 8)
MMP(2, 1, 8)
MMP(2, 2, 8)
MMP(4, 1, 8)
MMP(4, 2, 8)
MMP(1, 1, 16)
MMP(1, 2, 16)
MMP(2, 1, 16)
MMP(2, 2, 16)
MMP(4, 1, 16)
MMP(4, 2, 16)
// a prompt chunk's experts, grouped (up to 32 or 64 rows a job)
MMP(8, 2, 8)
MMP(16, 1, 8)

// A launch's input rows under its jobs' column fields, once a call: rows
// [n] of planes p0 + p1 (`pstride` a row) into o0, o1 at column `ooff` of
// rows `ostride` long, `fields` fields of `cols` (cf, one after the other)
// side by side, `field_plane`'s arithmetic element for element; no field
// (`fields` 0): the planes as they are.
extern "C" __global__ void hy_field_planes(const u16* __restrict__ p0, const u16* __restrict__ p1, int pstride,
                                           const float* __restrict__ cf, int fields, int cols, int n, u16* __restrict__ o0,
                                           u16* __restrict__ o1, int ostride, int ooff)
{
    const int nf = fields > 0 ? fields : 1;
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nf * cols) return;
    const int k = (int)(i % cols);
    const u64 jf = i / cols;
    const int fi = (int)(jf % nf), j = (int)(jf / nf);
    const u16 a0 = p0[(size_t)j * pstride + k], a1 = p1[(size_t)j * pstride + k];
    const size_t at = (size_t)j * ostride + ooff + (size_t)fi * cols + k;
    if (fields == 0) {
        o0[at] = a0;
        o1[at] = a1;
        return;
    }
    const float v = __fmul_rn(__fadd_rn(__fadd_rn(0.0f, bf2f(a0)), bf2f(a1)), cf[(size_t)fi * cols + k]);
    const u16 b0 = f2bf(v);
    o0[at] = b0;
    o1[at] = f2bf(__fsub_rn(v, bf2f(b0)));
}

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

// `hy_expand_p` over row tiles [tile0, tile0 + nt) of a matrix of `rows`:
// W holds those rows only (row tile0 * 16 first), so a large weight expands a
// band at a time into a bounded buffer.
extern "C" __global__ void hy_expand_pr(const uint2* __restrict__ T, const u8* __restrict__ Cc,
                                        const float* __restrict__ lad, u16* __restrict__ W, int rows, int cols,
                                        int tile0, int nt)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const uint2 X = T[idx];
    const int ra = tile * 16 + gid, rb = ra + 8;
    const float2* par = (const float2*)lad;
    const float sa = ra < rows ? exp2f(fmaf((float)Cc[tg * 16 + gid], par[ra].y, par[ra].x)) : 0.0f;
    const float sb = rb < rows ? exp2f(fmaf((float)Cc[tg * 16 + gid + 8], par[rb].y, par[rb].x)) : 0.0f;
    const int r0 = tile0 * 16;
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
            *(u32*)(W + (size_t)(r - r0) * cols + k) = pk;
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
        // the gates: block (units + r, y) is projection row r (a rows, then
        // b rows) for the call's rows [GR y, GR y + GR), its weights read in
        // float4 (hidden a multiple of 4); blocks past the rows have no work.
        //   gb[t][h] = exp(-exp(a_log) softplus(a + dt_bias)), gb[t][nv + h] = sigmoid(b)
        const int r0 = t * GR;
        if (r0 >= n) return;
        const int nr = min(GR, n - r0);
        __shared__ float gred[GR][8];
        const int r = u - units;
        const bool isb = r >= nv;
        const int h = isb ? r - nv : r;
        const float4* w4 = (const float4*)((isb ? wb : wa) + (size_t)h * hidden);
        float acc[GR];
#pragma unroll
        for (int i = 0; i < GR; i++) acc[i] = 0.0f;
        for (int c = tid; c < (hidden >> 2); c += blockDim.x) {
            const float4 w = w4[c];
#pragma unroll
            for (int i = 0; i < GR; i++) {
                if (i >= nr) break;
                const float4 x = ((const float4*)(xn + (size_t)(r0 + i) * hidden))[c];
                acc[i] = fmaf(w.w, x.w, fmaf(w.z, x.z, fmaf(w.y, x.y, fmaf(w.x, x.x, acc[i]))));
            }
        }
#pragma unroll
        for (int i = 0; i < GR; i++) {
            if (i >= nr) break;
            float v = acc[i];
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
            if (lane == 0 && warp < 8) gred[i][warp] = v;
        }
        __syncthreads();
        if (tid < nr) {
            float v = 0.0f;
            for (int w = 0; w < NW && w < 8; w++) v = __fadd_rn(v, gred[tid][w]);
            const size_t row = (size_t)(r0 + tid) * 2 * nv;
            if (isb) {
                gb[row + nv + h] = 1.0f / (1.0f + expf(-v));
            } else {
                const float x2 = __fadd_rn(v, dt_bias[h]);
                const float sp = x2 > 20.0f ? x2 : log1pf(expf(x2));
                gb[row + h] = expf(-__fmul_rn(expf(a_log[h]), sp));
            }
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

// `hy_gdn_restore_scan` for layers the small path scanned with
// `hy_gdn_scan4`: the same step, so the replayed rows give its bits.
extern "C" __global__ void __launch_bounds__(128) hy_gdn_restore_scan4(const u64* __restrict__ tab, int keep, int c_dim,
                                                                      int nv, int nk, int dv)
{
    const u64* tb = tab + (size_t)blockIdx.y * 7;
    gdn_scan4_body((const float*)tb[2], (const float*)tb[3], (const float*)tb[0], (float*)tb[1], nullptr, keep, c_dim,
                   nv, nk, dv, nullptr);
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


// ------------------------------------------------- the MTP draft chain
//
// Step j's token embedding: row tok[j] of a q4tp matrix in the tensor-core
// layout (`hy_repack`'s: the shortlist rows of the embedding), f32 into e,
// a thread a column; a token past the rows gives a zero row (a draft no
// verify keeps). The weight is (nibble - 8) exp2(lo + code step), as the
// tensor-core kernel reads it.
extern "C" __global__ void hy_embed_gather(const uint2* __restrict__ T, const u8* __restrict__ Cc,
                                           const float2* __restrict__ par, int rows, int cols,
                                           const int* __restrict__ tok, int j, float* __restrict__ e)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const int r = tok[j];
    if (r < 0 || r >= rows) {
        e[c] = 0.0f;
        return;
    }
    const int gpr = cols >> 5;
    const int tile = r >> 4, rr = r & 15, g = c >> 5, kk = c & 31;
    // column kk of the group: step s (k 0..15 or 16..31), register a_m of
    // the row half and k half, the low or high element of the pair
    const int s = kk >> 4, q = kk & 15, m = ((q >> 3) << 1) + (rr >> 3), tig = (q & 7) >> 1, lh = q & 1;
    const size_t tg = (size_t)tile * gpr + g;
    const uint2 X = T[tg * 32 + (rr & 7) * 4 + tig];
    const u32 w = s ? X.y : X.x;
    const int nib = (int)((w >> (4 * (m + 4 * lh))) & 15u);
    const float2 p = par[r];
    e[c] = __fmul_rn((float)(nib - 8), exp2f(fmaf((float)Cc[tg * 16 + rr], p.y, p.x)));
}

// The argmax of row blockIdx.x of x [rows][stride], its first n values, into
// out[j + row], a block a row: the LAST of equal maxima, as the host's
// `sampler::argmax` picks it (NaN never wins), so a verify on the card makes
// the plain decode's choice.
extern "C" __global__ void hy_argmax(const float* __restrict__ x, int n, int stride, int* __restrict__ out, int j)
{
    __shared__ float sv[32];
    __shared__ int si[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const float* xr = x + (size_t)blockIdx.x * stride;
    float bv = -INF;
    int bi = -1;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v >= bv) {
            bv = v;
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(FULL, bv, o);
        const int oi = __shfl_xor_sync(FULL, bi, o);
        if (ov > bv || (ov == bv && oi > bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < NW; w++)
            if (sv[w] > bv || (sv[w] == bv && si[w] > bi)) {
                bv = sv[w];
                bi = si[w];
            }
        out[j + blockIdx.x] = bi < 0 ? 0 : bi;
    }
}

// The prompt path's FFN middle from a gate/up pair GEMM: y [B][2 inter] in
// the pair's row order (tile t: 8 gate rows, then the same 8 up rows) into
// f16 xh [B][inter] = silu(gate) * up / 16.
extern "C" __global__ void hy_silu_pair_half(const float* __restrict__ y, u16* __restrict__ xh, int inter, int B, int act)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)B * inter) return;
    const u64 b = i / inter;
    const int r = (int)(i - b * inter);
    const float* yr = y + b * 2 * inter;
    const int base = (r >> 3) * 16 + (r & 7);
    const float g = yr[base], u = yr[base + 8];
    const float a = act ? gelu_t(g) : __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    xh[i] = f2h_sat(__fmul_rn(a, u) * 0.0625f);
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


// The prompt path's causal softmax over f32 scores S [rows of a KV group][L]
// (see `softmax_rows` of kernels.cu: row r is head r % hpk of the query at
// position pos0 + r / hpk, column c the key at col_base + c), the
// exponentials written as f16 into P (the value GEMM's operand) and each
// row's 1 / sum into isum.
extern "C" __global__ void hy_softmax_h(const float* __restrict__ S, u16* __restrict__ P, float* __restrict__ isum,
                                        int L, int pos0, int col_base, int hpk, int cols, int istride, int win,
                                        const float* __restrict__ sink)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* s = S + (size_t)blockIdx.x * L;
    u16* p = P + (size_t)blockIdx.x * L;
    const int gk = (int)blockIdx.x / cols, r = (int)blockIdx.x - gk * cols;
    const int pq = pos0 + r / hpk;
    const int hi = pq - col_base;
    // a sliding window: the keys left of its `win` (`hy_scores`' rule) weigh nothing
    const int lo = win > 0 ? hi - win + 1 : 0;
    float m = -INF;
    for (int c = tid; c <= hi && c < L; c += T)
        if (c >= lo) m = fmaxf(m, s[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INF;
    for (int k = 0; k < NW; k++) m = fmaxf(m, red[k]);
    // a learned sink logit (head gk * hpk + r % hpk) joins the max and the sum
    const float sk = sink ? sink[gk * hpk + r % hpk] : -INF;
    m = fmaxf(m, sk);
    __syncthreads();
    float sum = 0.0f;
    for (int c = tid; c < L; c += T) {
        float e = 0.0f;
        if (c <= hi && c >= lo) {
            e = expf(s[c] - m);
            sum += e;
        }
        p[c] = f2h(e);
    }
    for (int k = 16; k > 0; k >>= 1) sum += __shfl_xor_sync(FULL, sum, k);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int k = 0; k < NW; k++) sum += red[k];
        if (sink) sum += expf(sk - m);
        isum[(size_t)gk * istride + r] = sum > 0.0f ? 1.0f / sum : 1.0f;
    }
}


// ------------------------------------------------- the far field of a turbo file
//
// A sealed layer (`nystrom.rs`) keeps its near rows (the sinks with the rows
// kept exact, then the ring of the window) as an f16 mirror [row][kv head][HD]
// that the decode attention reads, and each head's far accumulators: QT
// [nh][m][HD] query landmarks, KT [nkv][m][HD] key landmarks, MU [nh][m][m]
// the landmark inverse, MZ [nh][2m] the running maxima then the far mass at
// that scale, TH [nh][m][HD] the far numerator at that scale (attn.cu's
// kernels, reading the call record: ci[6] the ring row a new key replaces,
// ci[7] whether the far field holds keys yet).

// The ring row a new key is about to replace goes into each head's far
// accumulators. A block a KV group, a thread per (head of the group, landmark).
extern "C" __global__ void hy_far_insert(const u16* __restrict__ NK, const u16* __restrict__ NV,
                                         const float* __restrict__ QT, float* __restrict__ MZ, float* __restrict__ TH,
                                         const int* __restrict__ ci, int m, int nkv, float scale)
{
    // a warp per (KV head g, its head j, landmark i): the key's dot product
    // across the lanes, then the head's far numerator row by the lanes
    const int wg = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (wg >= nkv * HPK * m) return;
    const int g = wg / (HPK * m), r = wg - g * (HPK * m), j = r / m, i = r - j * m;
    const int row = ci[6];
    const int h = g * HPK + j;
    const u16* k = NK + ((size_t)row * nkv + g) * HD;
    const u16* v = NV + ((size_t)row * nkv + g) * HD;
    const float* qt = QT + ((size_t)h * m + i) * HD;
    float l = 0.0f;
    for (int d = lane; d < HD; d += 32) l = fmaf(qt[d], h2f(k[d]), l);
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
    l *= scale;
    float* mz = MZ + (size_t)h * 2 * m;
    float mm = mz[i], z = mz[m + i], rs = 1.0f;
    if (l > mm) {
        rs = expf(mm - l);
        z *= rs;
        mm = l;
    }
    const float e = expf(l - mm);
    __syncwarp();
    if (lane == 0) {
        mz[i] = mm;
        mz[m + i] = z + e;
    }
    float* th = TH + ((size_t)h * m + i) * HD;
    for (int d = lane; d < HD; d += 32) th[d] = th[d] * rs + e * h2f(v[d]);
}

// The far part of one row's attention, a block a query head: block `nblk`
// (after the near rows' blocks) of P and E and maximum `nblk` of Mx
// (`o1_far_prep` of attn.cu, over the call record's geometry).
// The far part of a row's attention over a sealed layer, a block a query head
// (256 threads): block `nblk` (after the near rows' blocks) of P and E and
// maximum `nblk` of Mx (`o1_far_prep` of attn.cu, over the call record's
// geometry). Row b of B (a verify window's rows: b sees ci[0] + 1 + b near
// keys, the window's earlier rows among them, and the far field as it stood
// before the window). One body for both, so a window's row and a decode step
// of it agree to the bit.
__device__ __forceinline__ void far_prep_body(const float* __restrict__ q, const float* __restrict__ KT,
                                              const float* __restrict__ MU, const float* __restrict__ MZ,
                                              const float* __restrict__ TH, float* __restrict__ Mx,
                                              float* __restrict__ P, float* __restrict__ E,
                                              const int* __restrict__ ci, int m, int nh, float scale, int rect_fm,
                                              int B, int b)
{
    __shared__ float sf[64];
    __shared__ float su[64];
    __shared__ float sc[2];
    const int keys = ci[0] + 1 + b, rb = ci[2], nbcap = ci[4], far_on = ci[7];
    const int nblk = (keys + rb - 1) / rb;
    const int h = blockIdx.x, g = h / HPK, j = h - g * HPK, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;
    const float* mz = MZ + (size_t)h * 2 * m;
    float* Mxb = Mx + (size_t)b * nh * nbcap;
    float* Pb = P + (size_t)b * nbcap * nh * HD;
    float* Eb = E + (size_t)b * nbcap * nh;
    if (far_on) {
        const float* qh = q + (((size_t)g * B + b) * HPK + j) * HD;
        for (int a = warp; a < m; a += NW) {
            const float* kt = KT + ((size_t)g * m + a) * HD;
            float s = 0.0f;
            for (int d = lane; d < HD; d += 32) s = fmaf(qh[d], kt[d], s);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
            if (lane == 0) sf[a] = s * scale;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float c = -INF, f = -INF;
        for (int bb = 0; bb < nblk; bb++) c = fmaxf(c, Mxb[(size_t)h * nbcap + bb]);
        if (far_on) {
            for (int a = 0; a < m; a++) f = fmaxf(f, sf[a]);
            for (int bb = 0; bb < m; bb++) c = fmaxf(c, f + mz[bb]);
        }
        sc[0] = f;
        sc[1] = c;
    }
    __syncthreads();
    const float f = sc[0], call = sc[1];
    if (far_on && tid < m) sf[tid] = expf(sf[tid] - f);
    __syncthreads();
    if (far_on && tid < m) {
        const float* mu = MU + (size_t)h * m * m;
        float s = 0.0f;
        for (int a = 0; a < m; a++) s += sf[a] * mu[a * m + tid];
        if (rect_fm) s = fmaxf(s, 0.0f);
        su[tid] = s * expf(f + mz[tid] - call);
    }
    __syncthreads();
    if (tid == 0) {
        float den = 0.0f;
        if (far_on)
            for (int bb = 0; bb < m; bb++) den += su[bb] * mz[m + bb];
        sc[0] = den;
    }
    __syncthreads();
    const float den = sc[0];
    const bool have = far_on && den >= 0.0f;
    for (int d = tid; d < HD; d += blockDim.x) {
        float s = 0.0f;
        if (have) {
            const float* th = TH + (size_t)h * m * HD + d;
            for (int bb = 0; bb < m; bb++) s += su[bb] * th[(size_t)bb * HD];
        }
        Pb[((size_t)nblk * nh + h) * HD + d] = s;
    }
    if (tid == 0) {
        Eb[(size_t)nblk * nh + h] = have ? den : 0.0f;
        Mxb[(size_t)h * nbcap + nblk] = call;
    }
}

extern "C" __global__ void hy_far_prep_b(const float* __restrict__ q, const float* __restrict__ KT,
                                         const float* __restrict__ MU, const float* __restrict__ MZ,
                                         const float* __restrict__ TH, float* __restrict__ Mx,
                                         float* __restrict__ P, float* __restrict__ E,
                                         const int* __restrict__ ci, int m, int nh, float scale, int rect_fm, int B)
{
    far_prep_body(q, KT, MU, MZ, TH, Mx, P, E, ci, m, nh, scale, rect_fm, B, blockIdx.y);
}

extern "C" __global__ void hy_far_prep(const float* __restrict__ q, const float* __restrict__ KT,
                                       const float* __restrict__ MU, const float* __restrict__ MZ,
                                       const float* __restrict__ TH, float* __restrict__ Mx,
                                       float* __restrict__ P, float* __restrict__ E,
                                       const int* __restrict__ ci, int m, int nh, float scale, int rect_fm)
{
    far_prep_body(q, KT, MU, MZ, TH, Mx, P, E, ci, m, nh, scale, rect_fm, 1, 0);
}

// ------------------------------------------------- a sealed prompt chunk on the GEMMs
//
// A prompt read past a seal (`big` over a sealed layer): the chunk's rows sit
// after the near rows of the far-field mirror, the near part of their
// attention goes by the same batched GEMMs as an exact layer's, and the far
// part comes from the landmarks, a row at a time:

// Row b's far part for head h at its own scale: FM [B][nh] the scale (the
// largest far logit bound, as `far_prep_body`'s), FZ [B][nh] the far mass and
// FN [B][nh][HD] the far numerator at that scale. A block (head, row), 256
// threads; the arithmetic of `far_prep_body` without the near blocks.
extern "C" __global__ void hy_far_rows(const float* __restrict__ q, const float* __restrict__ KT,
                                       const float* __restrict__ MU, const float* __restrict__ MZ,
                                       const float* __restrict__ TH, float* __restrict__ FM,
                                       float* __restrict__ FZ, float* __restrict__ FN,
                                       int m, int nh, float scale, int rect_fm, int B)
{
    __shared__ float sf[64];
    __shared__ float su[64];
    __shared__ float sc[2];
    const int h = blockIdx.x, b = blockIdx.y, g = h / HPK, j = h - g * HPK, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;
    const float* mz = MZ + (size_t)h * 2 * m;
    const float* qh = q + (((size_t)g * B + b) * HPK + j) * HD;
    for (int a = warp; a < m; a += NW) {
        const float* kt = KT + ((size_t)g * m + a) * HD;
        float s = 0.0f;
        for (int d = lane; d < HD; d += 32) s = fmaf(qh[d], kt[d], s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
        if (lane == 0) sf[a] = s * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float f = -INF, c = -INF;
        for (int a = 0; a < m; a++) f = fmaxf(f, sf[a]);
        for (int bb = 0; bb < m; bb++) c = fmaxf(c, f + mz[bb]);
        sc[0] = f;
        sc[1] = c;
    }
    __syncthreads();
    const float f = sc[0], call = sc[1];
    if (tid < m) sf[tid] = expf(sf[tid] - f);
    __syncthreads();
    if (tid < m) {
        const float* mu = MU + (size_t)h * m * m;
        float s = 0.0f;
        for (int a = 0; a < m; a++) s += sf[a] * mu[a * m + tid];
        if (rect_fm) s = fmaxf(s, 0.0f);
        su[tid] = s * expf(f + mz[tid] - call);
    }
    __syncthreads();
    if (tid == 0) {
        float den = 0.0f;
        for (int bb = 0; bb < m; bb++) den += su[bb] * mz[m + bb];
        sc[0] = den;
    }
    __syncthreads();
    const float den = sc[0];
    const bool have = den >= 0.0f;
    for (int d = tid; d < HD; d += blockDim.x) {
        float s = 0.0f;
        if (have) {
            const float* th = TH + (size_t)h * m * HD + d;
            for (int bb = 0; bb < m; bb++) s += su[bb] * th[(size_t)bb * HD];
        }
        FN[((size_t)b * nh + h) * HD + d] = s;
    }
    if (tid == 0) {
        FM[(size_t)b * nh + h] = call;
        FZ[(size_t)b * nh + h] = have ? den : 0.0f;
    }
}

// `hy_softmax_h` with a far part. The weights stay at the near maximum (f16
// keeps its precision), and the row's two factors go beside the mass: fa the
// near part's (exp(near max - max)), fc the far numerator's (exp(FM - max)),
// isum one over the whole mass. FM/FZ point at the sub-chunk's first row.
extern "C" __global__ void hy_softmax_hf(const float* __restrict__ S, u16* __restrict__ P, float* __restrict__ isum,
                                         float* __restrict__ fa, float* __restrict__ fc, int L, int pos0, int col_base,
                                         int hpk, int cols, int istride, const float* __restrict__ FM,
                                         const float* __restrict__ FZ, int nh)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* s = S + (size_t)blockIdx.x * L;
    u16* p = P + (size_t)blockIdx.x * L;
    const int gk = (int)blockIdx.x / cols, r = (int)blockIdx.x - gk * cols;
    const int b = r / hpk, h = gk * hpk + (r - b * hpk);
    const int pq = pos0 + b;
    const int hi = pq - col_base;
    float m = -INF;
    for (int c = tid; c <= hi && c < L; c += T) m = fmaxf(m, s[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INF;
    for (int k = 0; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float sum = 0.0f;
    for (int c = tid; c < L; c += T) {
        float e = 0.0f;
        if (c <= hi) {
            e = expf(s[c] - m);
            sum += e;
        }
        p[c] = f2h(e);
    }
    for (int k = 16; k > 0; k >>= 1) sum += __shfl_xor_sync(FULL, sum, k);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int k = 0; k < NW; k++) sum += red[k];
        float a = 1.0f, cf = 0.0f, fz = 0.0f;
        if (FM) {
            const float fm = FM[(size_t)b * nh + h], mm = fmaxf(m, fm);
            a = expf(m - mm);
            cf = expf(fm - mm);
            fz = FZ[(size_t)b * nh + h];
        }
        const float tot = a * sum + cf * fz;
        const size_t ix = (size_t)gk * istride + r;
        isum[ix] = tot > 0.0f ? 1.0f / tot : 1.0f;
        fa[ix] = a;
        fc[ix] = cf;
    }
}

// `hy_attn_out` with the far numerator: (fa * near + fc * far) * isum.
extern "C" __global__ void hy_attn_out_f(const float* __restrict__ aog, const float* __restrict__ isum,
                                         const float* __restrict__ fa, const float* __restrict__ fc,
                                         const float* __restrict__ FN, const float* __restrict__ gate,
                                         void* __restrict__ ao, int nh, int B, int half)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * nh * HD) return;
    const int d = (int)(i % HD);
    const size_t bh = i / HD;
    const int h = (int)(bh % nh), b = (int)(bh / nh);
    const int g = h / HPK, j = h - g * HPK;
    const size_t ix = (size_t)g * B * HPK + (size_t)b * HPK + j;
    float v = aog[(((size_t)g * B + b) * HPK + j) * HD + d] * fa[ix];
    if (FN) v = fmaf(fc[ix], FN[((size_t)b * nh + h) * HD + d], v);
    v *= isum[ix];
    if (gate) v *= sigm(gate[i]);
    stf(ao, i, v, half);
}

// The chunk's commit: E ring rows, the oldest first (slots ns + (head0 + t) % w),
// into each head's far accumulators, row after row as `hy_far_insert` takes
// one. A warp per (KV head, its head, landmark), the landmark and its
// numerator row in registers.
extern "C" __global__ void hy_far_insert_n(const u16* __restrict__ NK, const u16* __restrict__ NV,
                                           const float* __restrict__ QT, float* __restrict__ MZ, float* __restrict__ TH,
                                           int m, int nkv, float scale, int ns, int w, int head0, int E)
{
    const int wg = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (wg >= nkv * HPK * m) return;
    const int g = wg / (HPK * m), r = wg - g * (HPK * m), j = r / m, i = r - j * m;
    const int h = g * HPK + j;
    const float* qt = QT + ((size_t)h * m + i) * HD;
    float* th = TH + ((size_t)h * m + i) * HD;
    float qr[DPL], acc[DPL];
#pragma unroll
    for (int k = 0; k < DPL; k++) {
        qr[k] = qt[lane + 32 * k];
        acc[k] = th[lane + 32 * k];
    }
    float* mz = MZ + (size_t)h * 2 * m;
    float mm = mz[i], z = mz[m + i];
    for (int t = 0; t < E; t++) {
        const int row = ns + (head0 + t) % w;
        const u16* k = NK + ((size_t)row * nkv + g) * HD;
        const u16* v = NV + ((size_t)row * nkv + g) * HD;
        float l = 0.0f;
#pragma unroll
        for (int kk = 0; kk < DPL; kk++) l = fmaf(qr[kk], h2f(k[lane + 32 * kk]), l);
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
        l *= scale;
        float rs = 1.0f;
        if (l > mm) {
            rs = expf(mm - l);
            z *= rs;
            mm = l;
        }
        const float e = expf(l - mm);
        z = z + e;
#pragma unroll
        for (int kk = 0; kk < DPL; kk++) acc[kk] = acc[kk] * rs + e * h2f(v[lane + 32 * kk]);
    }
    if (lane == 0) {
        mz[i] = mm;
        mz[m + i] = z;
    }
#pragma unroll
    for (int k = 0; k < DPL; k++) th[lane + 32 * k] = acc[k];
}


// ------------------------------------------------------------- MoE
//
// A routed FFN inside the token graph. The router's logits and the shared
// expert's gate come from `hy_moe_router`; `hy_moe_select` picks every row's
// top k and lays the union of the picked experts out as the launch's slots:
// per slot the gate/up and the down job of its expert (the layer's job
// templates, outputs at slot offsets), and the mixing weight of every (row,
// slot), 0 where the row did not pick that expert. Two tensor-core launches
// then run every slot (`hy_mmp_*`, the down jobs reading their own slot's
// planes through `xoff`), and `hy_moe_combine` adds the mixture (and the
// shared expert) to the stream. Nothing leaves the card, so the layer is
// captured with the rest of the token.

// logits[r][e] = Wr[e] . xn[r] for e < E; with `sgw` the shared gate's
// pre-sigmoid score sg[r] (as row E). A warp a router row.
extern "C" __global__ void hy_moe_router(const float* __restrict__ xn, const float* __restrict__ Wr,
                                         const float* __restrict__ sgw, float* __restrict__ logits,
                                         float* __restrict__ sg, int n, int E, int hidden)
{
    const int lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (e > E || (e == E && !sgw)) return;
    const float* w = e < E ? Wr + (size_t)e * hidden : sgw;
    // a row a grid row
    {
        const int r = blockIdx.y;
        const float* x = xn + (size_t)r * hidden;
        float acc = 0.0f;
        for (int i = lane; i < hidden; i += 32) acc = fmaf(w[i], x[i], acc);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
        if (lane == 0) {
            if (e < E) logits[(size_t)r * E + e] = acc;
            else sg[r] = acc;
        }
    }
}

__device__ __forceinline__ void moe_argmax(float& v, int& i)
{
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(FULL, v, o);
        const int i2 = __shfl_xor_sync(FULL, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
}

// One block, a warp a row (n <= 16 warps); E <= 1024 experts, k <= 64,
// S <= 1024 slots. A row's scores sit in its lanes' registers and its top k
// are k warp-wide argmax shuffles; the union is laid out by one thread.
// flags: bit0 renormalize the picked weights, bit1 sigmoid scores (else a
// softmax), bit2 a selection bias on the sigmoid ranking key, bit5 the
// sigmoid renormalization floor is 1e-20 (else 1e-6).
extern "C" __global__ void hy_moe_select(const float* __restrict__ logits, const float* __restrict__ bias, int n, int E,
                                         int k, int flags, float route_scale, const HyJob* __restrict__ gu_t,
                                         const HyJob* __restrict__ dn_t, HyJob* __restrict__ gu_a,
                                         HyJob* __restrict__ dn_a, float* __restrict__ wsel, int S, u64 qq0, u64 qq1,
                                         int pstride, u64 dnS, int hidden, int bpj_gu, int bpj_dn)
{
    __shared__ int slot_of[1024];
    __shared__ int U[1024];
    __shared__ int pick[16][64];
    __shared__ float pw[16][64];
    __shared__ int used;
    const int tid = threadIdx.x, lane = tid & 31, r = tid >> 5;
    const bool sigm = (flags & 2) != 0;
    for (int e = tid; e < E; e += blockDim.x) slot_of[e] = -1;
    for (int i = tid; i < n * S; i += blockDim.x) wsel[i] = 0.0f;
    if (r < n) {
        const float* L = logits + (size_t)r * E;
        float key[32], lg[32];
#pragma unroll
        for (int q = 0; q < 32; q++) {
            const int e = lane + 32 * q;
            lg[q] = e < E ? L[e] : -3.0e38f;
        }
        float mx = -3.0e38f, sum = 0.0f;
        if (!sigm) {
#pragma unroll
            for (int q = 0; q < 32; q++) mx = fmaxf(mx, lg[q]);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
#pragma unroll
            for (int q = 0; q < 32; q++)
                if (lane + 32 * q < E) sum += expf(lg[q] - mx);
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL, sum, o);
        }
#pragma unroll
        for (int q = 0; q < 32; q++) {
            const int e = lane + 32 * q;
            float kk = lg[q];
            if (sigm && e < E) {
                kk = 1.0f / (1.0f + expf(-kk));
                if ((flags & 4) && bias) kk += bias[e];
            }
            key[q] = e < E ? kk : -3.0e38f;
        }
        for (int j = 0; j < k; j++) {
            float best = -3.0e38f;
            int bi = 0x7fffffff;
#pragma unroll
            for (int q = 0; q < 32; q++) {
                const int e = lane + 32 * q;
                if (key[q] > best || (key[q] == best && e < bi)) {
                    best = key[q];
                    bi = e;
                }
            }
            moe_argmax(best, bi);
            // the owner lane takes the pick out of the running
#pragma unroll
            for (int q = 0; q < 32; q++)
                if (lane + 32 * q == bi) key[q] = -3.0e38f;
            if (lane == 0) {
                const float lb = L[bi];
                pick[r][j] = bi;
                pw[r][j] = sigm ? 1.0f / (1.0f + expf(-lb)) : expf(lb - mx) / sum;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        used = 0;
        for (int rr = 0; rr < n; rr++) {
            float tot = 0.0f;
            for (int j = 0; j < k; j++) tot += pw[rr][j];
            const float dn = sigm ? tot + ((flags & 32) ? 1e-20f : 1e-6f) : tot;
            for (int j = 0; j < k; j++) {
                float w = pw[rr][j];
                if (flags & 1) w = w / dn;
                w *= route_scale;
                const int e = pick[rr][j];
                int s = slot_of[e];
                if (s < 0 && used < S) {
                    s = used++;
                    slot_of[e] = s;
                    U[s] = e;
                }
                if (s >= 0) wsel[(size_t)rr * S + s] = w;
            }
        }
    }
    __syncthreads();
    for (int s = tid; s < S; s += blockDim.x) {
        const bool live = s < used;
        HyJob g = gu_t[live ? U[s] : 0];
        HyJob d = dn_t[live ? U[s] : 0];
        if (!live) {
            g.rows = 0;
            d.rows = 0;
        }
        g.y = qq0 + (u64)s * n * pstride * 2;
        g.res = qq1 + (u64)s * n * pstride * 2;
        g.ystride = pstride;
        g.blk0 = s * bpj_gu;
        g.xoff = 0;
        d.y = dnS + (u64)s * n * hidden * 4;
        d.res = 0;
        d.ystride = hidden;
        d.blk0 = s * bpj_dn;
        d.xoff = s * n * pstride;
        gu_a[s] = g;
        dn_a[s] = d;
    }
}

// h[r] += sum over slots of wsel[r][s] * dnS[s][r] (+ the shared expert's
// output times its weight: sigmoid(sg[r]) under `gated`, else 1).
extern "C" __global__ void hy_moe_combine(float* __restrict__ h, const float* __restrict__ dnS,
                                          const float* __restrict__ wsel, const float* __restrict__ dsh,
                                          const float* __restrict__ sg, int gated, int n, int hidden, int S)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= hidden || r >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < S; s++) {
        const float w = wsel[(size_t)r * S + s];
        if (w != 0.0f) acc = fmaf(w, dnS[((size_t)s * n + r) * hidden + i], acc);
    }
    if (dsh) {
        const float w = gated ? 1.0f / (1.0f + expf(-sg[r])) : 1.0f;
        acc = fmaf(w, dsh[(size_t)r * hidden + i], acc);
    }
    h[(size_t)r * hidden + i] += acc;
}

// ---------------------------------------------- MoE on the prompt path
//
// A prompt chunk's routed FFN grouped by expert. `hy_moe_topk` picks every
// row's k experts and their mixing weights (`hy_moe_select`'s arithmetic);
// `hy_moe_group` (one block) sorts the (row, pick) entries by expert, in
// entry order, and writes a gate/up and a down job for every `rmax` rows of
// an expert (the job's row count in `nr`, its rows' planes at `xoff`);
// `hy_moe_gather` lays the picked rows out in that order as input planes;
// two tensor-core launches run the jobs; `hy_moe_gcombine` adds the weighted
// outputs (and the shared expert) to the stream.

// A warp a row: pidx[r][j], pwt[r][j] for its k picks.
extern "C" __global__ void hy_moe_topk(const float* __restrict__ logits, const float* __restrict__ bias, int n, int E,
                                       int k, int flags, float route_scale, int* __restrict__ pidx, float* __restrict__ pwt)
{
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= n) return;
    const bool sigm = (flags & 2) != 0;
    const float* L = logits + (size_t)r * E;
    float key[32], lg[32];
#pragma unroll
    for (int q = 0; q < 32; q++) {
        const int e = lane + 32 * q;
        lg[q] = e < E ? L[e] : -3.0e38f;
    }
    float mx = -3.0e38f, sum = 0.0f;
    if (!sigm) {
#pragma unroll
        for (int q = 0; q < 32; q++) mx = fmaxf(mx, lg[q]);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
#pragma unroll
        for (int q = 0; q < 32; q++)
            if (lane + 32 * q < E) sum += expf(lg[q] - mx);
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL, sum, o);
    }
#pragma unroll
    for (int q = 0; q < 32; q++) {
        const int e = lane + 32 * q;
        float kk = lg[q];
        if (sigm && e < E) {
            kk = 1.0f / (1.0f + expf(-kk));
            if ((flags & 4) && bias) kk += bias[e];
        }
        key[q] = e < E ? kk : -3.0e38f;
    }
    float tot = 0.0f;
    for (int j = 0; j < k; j++) {
        float best = -3.0e38f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int q = 0; q < 32; q++) {
            const int e = lane + 32 * q;
            if (key[q] > best || (key[q] == best && e < bi)) {
                best = key[q];
                bi = e;
            }
        }
        moe_argmax(best, bi);
#pragma unroll
        for (int q = 0; q < 32; q++)
            if (lane + 32 * q == bi) key[q] = -3.0e38f;
        if (lane == 0) {
            const float lb = L[bi];
            const float w = sigm ? 1.0f / (1.0f + expf(-lb)) : expf(lb - mx) / sum;
            pidx[(size_t)r * k + j] = bi;
            pwt[(size_t)r * k + j] = w;
            tot += w;
        }
    }
    if (lane == 0) {
        const float dn = sigm ? tot + ((flags & 32) ? 1e-20f : 1e-6f) : tot;
        for (int j = 0; j < k; j++) {
            float w = pwt[(size_t)r * k + j];
            if (flags & 1) w = w / dn;
            pwt[(size_t)r * k + j] = w * route_scale;
        }
    }
}

// One block (E <= 1024): pos[i] = entry i's place in the expert-sorted order
// (an expert's entries in entry order), src[pos] = its row (i / k), and the
// jobs: per `rmax` rows of an expert a gate/up job from gu_t (input planes
// at its first row's place of the gathered rows, `hidden` a row; output
// planes qq0, qq1, `inter_s` a row) and a down job from dn_t (input the
// gate/up's planes, output ys, `hidden` floats a row); jobs past the count
// up to jmax are empty, their first blocks still ascending.
extern "C" __global__ void hy_moe_group(const int* __restrict__ pidx, int nk, int k, int E, int rmax, int jmax,
                                        const HyJob* __restrict__ gu_t, const HyJob* __restrict__ dn_t,
                                        HyJob* __restrict__ gu_a, HyJob* __restrict__ dn_a, int* __restrict__ pos,
                                        int* __restrict__ src, u64 qq0, u64 qq1, int hidden, int inter_s, u64 ys,
                                        int bpj_gu, int bpj_dn)
{
    __shared__ int cnt[1024], off[1024], jb0[1024];
    __shared__ int njobs;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    // a warp an expert: its entries by ballots, in entry order
    for (int e = warp; e < E; e += NW) {
        int c = 0;
        for (int base = 0; base < nk; base += 32) {
            const int i = base + lane;
            c += __popc(__ballot_sync(FULL, i < nk && pidx[i] == e));
        }
        if (lane == 0) cnt[e] = c;
    }
    __syncthreads();
    if (tid == 0) {
        int o = 0, j = 0;
        for (int e = 0; e < E; e++) {
            off[e] = o;
            jb0[e] = j;
            o += cnt[e];
            j += (cnt[e] + rmax - 1) / rmax;
        }
        njobs = min(j, jmax);
    }
    __syncthreads();
    for (int e = warp; e < E; e += NW) {
        int p = off[e];
        for (int base = 0; base < nk; base += 32) {
            const int i = base + lane;
            const bool m = i < nk && pidx[i] == e;
            const unsigned b = __ballot_sync(FULL, m);
            if (m) {
                const int q = p + __popc(b & ((1u << lane) - 1u));
                pos[i] = q;
                src[q] = i / k;
            }
            p += __popc(b);
        }
    }
    for (int e = tid; e < E; e += blockDim.x) {
        const int c = cnt[e];
        for (int ps = 0; ps * rmax < c; ps++) {
            const int j = jb0[e] + ps;
            if (j >= jmax) break;
            const int r0 = off[e] + ps * rmax, nr = min(rmax, c - ps * rmax);
            HyJob g = gu_t[e];
            g.y = qq0 + (u64)r0 * inter_s * 2;
            g.res = qq1 + (u64)r0 * inter_s * 2;
            g.ystride = inter_s;
            g.blk0 = j * bpj_gu;
            g.xoff = r0 * hidden;
            g.nr = nr;
            HyJob d = dn_t[e];
            d.y = ys + (u64)r0 * hidden * 4;
            d.res = 0;
            d.ystride = hidden;
            d.blk0 = j * bpj_dn;
            d.xoff = r0 * inter_s;
            d.nr = nr;
            gu_a[j] = g;
            dn_a[j] = d;
        }
    }
    for (int j = njobs + tid; j < jmax; j += blockDim.x) {
        HyJob g = gu_t[0], d = dn_t[0];
        g.rows = 0;
        d.rows = 0;
        g.blk0 = j * bpj_gu;
        d.blk0 = j * bpj_dn;
        g.nr = 0;
        d.nr = 0;
        gu_a[j] = g;
        dn_a[j] = d;
    }
}

// The picked rows in sorted order as input planes: row q of the planes is
// row src[q] of xn [n][hidden] f32.
extern "C" __global__ void hy_moe_gather(const float* __restrict__ xn, const int* __restrict__ src, int total, int hidden,
                                         u16* __restrict__ p0, u16* __restrict__ p1, u16* __restrict__ p2)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)total * hidden) return;
    const int q = (int)(idx / hidden), i = (int)(idx - (size_t)q * hidden);
    put3(p0, p1, p2, idx, xn[(size_t)src[q] * hidden + i]);
}

// h[r] += sum over its picks of pwt[r][j] * ys[pos[r][j]] (+ the shared
// expert's rows times sigmoid(sg[r]) under `gated`, else 1).
extern "C" __global__ void hy_moe_gcombine(float* __restrict__ h, const float* __restrict__ ys, const int* __restrict__ pos,
                                           const float* __restrict__ pwt, const float* __restrict__ dsh,
                                           const float* __restrict__ sg, int gated, int n, int k, int hidden)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= hidden || r >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc = fmaf(pwt[(size_t)r * k + j], ys[(size_t)pos[(size_t)r * k + j] * hidden + i], acc);
    if (dsh) {
        const float w = gated ? 1.0f / (1.0f + expf(-sg[r])) : 1.0f;
        acc = fmaf(w, dsh[(size_t)r * hidden + i], acc);
    }
    h[(size_t)r * hidden + i] += acc;
}


// Gemma's sandwich norm: h[r] += rmsnorm(y[r]) * w (1 + w under `gemma`), a
// block a row; then, with `ls` given, the layer's scale on the whole row.
extern "C" __global__ void hy_norm_add(float* __restrict__ h, const float* __restrict__ y, const float* __restrict__ w,
                                       int hidden, float eps, int gemma)
{
    __shared__ float red[32];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const float* yr = y + (size_t)r * hidden;
    float s = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) s = fmaf(yr[i], yr[i], s);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = 1.0f / (float)sqrt(tot / (double)hidden + (double)eps);
    float* hr = h + (size_t)r * hidden;
    for (int i = tid; i < hidden; i += blockDim.x)
        hr[i] = __fadd_rn(hr[i], __fmul_rn(__fmul_rn(yr[i], inv), gemma ? 1.0f + w[i] : w[i]));
}

// h *= s over `total` values (Gemma-4's layer scalar).
extern "C" __global__ void hy_scale(float* __restrict__ h, float s, unsigned long long total)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) h[i] *= s;
}


// LFM2's gated short convolution over n rows in order: y [n][ystride] holds
// each row's B | C | x (hidden each), the ring [channel][k - 1] (slot 0 the
// newest) the conv's history; row r's C * (taps . (B x, ring)) goes into the
// planes the out projection reads, and the ring moves row by row. A thread a
// channel; k <= 9.
extern "C" __global__ void hy_sconv(const float* __restrict__ y, int ystride, const float* __restrict__ taps,
                                    float* __restrict__ ring, int hidden, int k, int n, u16* __restrict__ p0,
                                    u16* __restrict__ p1, u16* __restrict__ p2, int pstride)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= hidden) return;
    const int rl = k - 1;
    float rg[8];
#pragma unroll
    for (int s = 0; s < 8; s++) rg[s] = s < rl ? ring[(size_t)ch * rl + s] : 0.0f;
    const float* t = taps + (size_t)ch * k;
    for (int r = 0; r < n; r++) {
        const float* yr = y + (size_t)r * ystride;
        const float bx = yr[ch] * yr[2 * hidden + ch];
        float acc = t[k - 1] * bx;
#pragma unroll
        for (int s = 0; s < 8; s++)
            if (s < rl) acc += t[k - 2 - s] * rg[s];
        put3(p0, p1, p2, (size_t)r * pstride + ch, yr[hidden + ch] * acc);
#pragma unroll
        for (int s = 7; s > 0; s--)
            if (s < rl) rg[s] = rg[s - 1];
        if (rl > 0) rg[0] = bx;
    }
#pragma unroll
    for (int s = 0; s < 8; s++)
        if (s < rl) ring[(size_t)ch * rl + s] = rg[s];
}


// Spark-X2.5 / LFM3's head-wise output gate on the attention's output planes
// (before the O projection): head h of row r times sigmoid(g[r][h]).
extern "C" __global__ void hy_head_gate(u16* __restrict__ p0, u16* __restrict__ p1, int pstride, const float* __restrict__ g,
                                        int nh, int hd, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= nh * hd || r >= n) return;
    const size_t at = (size_t)r * pstride + i;
    const float v = __fadd_rn(bf2f(p0[at]), bf2f(p1[at]));
    const float s = 1.0f / (1.0f + expf(-g[(size_t)r * nh + i / hd]));
    put3(p0, p1, p1, at, __fmul_rn(v, s));
}


// An int8 matrix (q8_2f / q8_row) as the tensor-core kernel's int8 tiles
// (kind 5): for tile t and group g, 32 lanes x 16 bytes; lane l = 4 gid + tig
// holds the elements of its eight A registers (s, m) — row gid + 8 (m & 1),
// columns 32 g + 16 s + 8 (m >> 1) + 2 tig and the next — register 2q + h at
// bytes 4q + 2h, 4q + 2h + 1. `pair`: a gate/up pair's rows as `hy_repack`
// maps them (8 rows of A, then the same 8 of B). The row scales and the
// column field stay apart (f32, hyb.rs).
extern "C" __global__ void hy_repack8(const signed char* __restrict__ bA, const signed char* __restrict__ bB, int pair,
                                      uint4* __restrict__ T, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 wd[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int col = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const bool second = pair && (r & 15) >= 8;
            const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
            const signed char* b = (second ? bB : bA) + (size_t)sr * cols + col;
            const int reg = s * 4 + m;
            const u32 two = (u32)(u8)b[0] | ((u32)(u8)b[1] << 8);
            wd[reg >> 1] |= two << (16 * (reg & 1));
        }
    T[idx] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
}

// `hy_expand_pr` for kind 5 (int8 tiles, f32 row scales `rs`, f32 column
// field `cf` or null): row tiles [tile0, tile0 + nt) as f16 rows, W holding
// those rows only.
extern "C" __global__ void hy_expand_i8r(const uint4* __restrict__ T, const float* __restrict__ rs, const float* __restrict__ cf,
                                         u16* __restrict__ W, int rows, int cols, int tile0, int nt, int dual)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const uint4 X = T[idx];
    const u32 wd[4] = {X.x, X.y, X.z, X.w};
    const int r0 = tile0 * 16;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int k = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            const u32 two = wd[reg >> 1] >> (16 * (reg & 1));
            const float sc = rs[r];
            // `dual` (kind 15): a pair's up rows under the second field
            const float* f = (cf && dual && (r & 15) >= 8) ? cf + cols : cf;
            const float lo = (float)(signed char)(two & 0xFF) * sc * (f ? f[k] : 1.0f);
            const float hi = (float)(signed char)((two >> 8) & 0xFF) * sc * (f ? f[k + 1] : 1.0f);
            *(u32*)(W + (size_t)(r - r0) * cols + k) = (u32)f2h(lo) | ((u32)f2h(hi) << 16);
        }
}

// A q1 matrix (per 32-column group [f16 scale][4 bytes of sign bits, LSB
// first]) as the tensor-core kernel's q1 tiles (kind 3): for tile t and group
// g, lane l = 4 gid + tig holds in bits r, r + 8 the signs of its A register
// r = 4 s + m (row gid + 8 (m & 1), columns 32 g + 16 s + 8 (m >> 1) + 2 tig and
// the next); S[t][g][16] the rows' f16 scales. `pair`: a gate/up pair's rows
// as `hy_repack` maps them.
extern "C" __global__ void hy_repack1(const u8* __restrict__ A, const u8* __restrict__ Bm, int pair, u16* __restrict__ T,
                                      u16* __restrict__ S, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 bits = 0u;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const bool second = pair && (r & 15) >= 8;
            const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
            const u8* t6 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 6 + 2;
            const int c = 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            bits |= (u32)((t6[c >> 3] >> (c & 7)) & 1) << reg;
            bits |= (u32)((t6[(c + 1) >> 3] >> ((c + 1) & 7)) & 1) << (reg + 8);
        }
    T[idx] = (u16)bits;
    if (tig == 0) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = tile * 16 + gid + 8 * h;
            u16 sc = 0;
            if (r < rows) {
                const bool second = pair && (r & 15) >= 8;
                const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
                const u8* t6 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 6;
                sc = (u16)(t6[0] | (t6[1] << 8));
            }
            S[tg * 16 + gid + 8 * h] = sc;
        }
    }
}

// `hy_expand_pr` for kind 3 (q1 tiles): row tiles [tile0, tile0 + nt) as f16
// rows, W holding those rows only.
extern "C" __global__ void hy_expand_q1r(const u16* __restrict__ T, const u16* __restrict__ S, u16* __restrict__ W, int rows,
                                         int cols, int tile0, int nt)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const u32 B = T[idx];
    const int r0 = tile0 * 16;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int k = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            const float sc = h2f(S[tg * 16 + gid + 8 * (m & 1)]);
            const float lo = ((B >> reg) & 1u) ? sc : -sc;
            const float hi = ((B >> (reg + 8)) & 1u) ? sc : -sc;
            *(u32*)(W + (size_t)(r - r0) * cols + k) = (u32)f2h(lo) | ((u32)f2h(hi) << 16);
        }
}

// A q1t matrix (per 32-column group [f16 scale][7 bytes of base-3 codes, five
// a byte: 0 -> 0, 1 -> +s, 2 -> -s]; no outlier rows) as ternary tiles (kind
// 2): lane l holds at bits 2r, 16 + 2r the codes (value + 1) of its A register
// r's two elements (`hy_repack1`'s order); S[t][g][16] the rows' f16 scales.
__device__ __forceinline__ u32 q1t_val1(const u8* c7, int k) {
    const u32 p3[5] = {1u, 3u, 9u, 27u, 81u};
    const u32 d = ((u32)c7[k / 5] / p3[k % 5]) % 3u;
    return d == 1u ? 2u : (d == 2u ? 0u : 1u);
}

extern "C" __global__ void hy_repack1t(const u8* __restrict__ A, const u8* __restrict__ Bm, int pair, u32* __restrict__ T,
                                       u16* __restrict__ S, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 w = 0u;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            const int reg = s * 4 + m;
            u32 lo = 1u, hi = 1u;
            if (r < rows) {
                const bool second = pair && (r & 15) >= 8;
                const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
                const u8* c7 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 9 + 2;
                const int c = 16 * s + 8 * (m >> 1) + 2 * tig;
                lo = q1t_val1(c7, c);
                hi = q1t_val1(c7, c + 1);
            }
            w |= (lo << (2 * reg)) | (hi << (16 + 2 * reg));
        }
    T[idx] = w;
    if (tig == 0) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = tile * 16 + gid + 8 * h;
            u16 sc = 0;
            if (r < rows) {
                const bool second = pair && (r & 15) >= 8;
                const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
                const u8* t9 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 9;
                sc = (u16)(t9[0] | (t9[1] << 8));
            }
            S[tg * 16 + gid + 8 * h] = sc;
        }
    }
}

// `hy_expand_pr` for kind 2 (ternary tiles): row tiles [tile0, tile0 + nt)
// as f16 rows, W holding those rows only.
extern "C" __global__ void hy_expand_q1tr(const u32* __restrict__ T, const u16* __restrict__ S, u16* __restrict__ W, int rows,
                                          int cols, int tile0, int nt)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const u32 w = T[idx];
    const int r0 = tile0 * 16;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int k = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            const float sc = h2f(S[tg * 16 + gid + 8 * (m & 1)]);
            const float lo = ((float)((w >> (2 * reg)) & 3u) - 1.0f) * sc;
            const float hi = ((float)((w >> (16 + 2 * reg)) & 3u) - 1.0f) * sc;
            *(u32*)(W + (size_t)(r - r0) * cols + k) = (u32)f2h(lo) | ((u32)f2h(hi) << 16);
        }
}

// A q2tp matrix ([2-bit chunks: 8 bytes a (row, group), weight k at byte k/4,
// bits 2 (k % 4)][row params][5-bit rung codes]) as q2tp tiles (kinds 1 / 9):
// lane l holds at bits 2r, 16 + 2r the codes of its A register r's two
// elements (`hy_repack1t`'s order); Cc[t][g][16] the rows' rung codes as
// `hy_repack` lays them. `pair`: a gate/up pair's rows as `hy_repack` maps them.
extern "C" __global__ void hy_repack2(const u8* __restrict__ chA, const u8* __restrict__ codesA, const u8* __restrict__ chB,
                                      const u8* __restrict__ codesB, int stride, int pair, u32* __restrict__ T,
                                      u8* __restrict__ Cc, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 w = 0u;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const bool second = pair && (r & 15) >= 8;
            const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
            const u8* ch = (second ? chB : chA) + ((size_t)sr * gpr + g) * 8;
            const int c = 16 * s + 8 * (m >> 1) + 2 * tig;
            const u32 lo = (ch[c >> 2] >> (2 * (c & 3))) & 3u;
            const u32 hi = (ch[(c + 1) >> 2] >> (2 * ((c + 1) & 3))) & 3u;
            w |= (lo << (2 * (s * 4 + m))) | (hi << (16 + 2 * (s * 4 + m)));
        }
    T[idx] = w;
    if (lane < 16) {
        const int r = tile * 16 + lane;
        u8 code = 0;
        if (r < rows) {
            const bool second = pair && (r & 15) >= 8;
            const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
            const u8* cr = (second ? codesB : codesA) + (size_t)sr * stride;
            const int bit = g * 5, b = bit >> 3, sh = bit & 7;
            u32 v = cr[b];
            if (sh > 3) v |= (u32)cr[b + 1] << 8;
            code = (u8)((v >> sh) & 31u);
        }
        Cc[tg * 16 + lane] = code;
    }
}

// `hy_expand_pr` for kinds 1 / 9 (q2tp tiles): row tiles [tile0, tile0 + nt)
// as f16 rows, W holding those rows only.
extern "C" __global__ void hy_expand_q2r(const u32* __restrict__ T, const u8* __restrict__ Cc, const float* __restrict__ lad,
                                         u16* __restrict__ W, int rows, int cols, int tile0, int nt, int aff)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const u32 w = T[idx];
    const int r0 = tile0 * 16;
    const float2* par = (const float2*)lad;
    const float off = aff ? 1.0f : 1.5f;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int k = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            const int code = Cc[tg * 16 + gid + 8 * (m & 1)];
            const float sc = code ? exp2f(fmaf((float)(code - 1), par[r].y, par[r].x)) : 0.0f;
            const float lo = ((float)((w >> (2 * reg)) & 3u) - off) * sc;
            const float hi = ((float)((w >> (16 + 2 * reg)) & 3u) - off) * sc;
            *(u32*)(W + (size_t)(r - r0) * cols + k) = (u32)f2h(lo) | ((u32)f2h(hi) << 16);
        }
}

// Prism's activation boundary (`prism::forward`): every block of `blk` (a
// power of two) of a row becomes FWHT(x * signs) / sqrt(blk), rounded to f16
// under `f16`, the butterflies in the host's order. A CUDA block a (row,
// block), the block in shared memory. `hy_prism_p`: the row's planes in
// place; `hy_prism_h`: f16 rows `src` into `dst` (in place allowed).
__device__ __forceinline__ void prism_fwht(float* sv, int blk) {
    for (int w = 1; w < blk; w <<= 1) {
        for (int i = threadIdx.x; i < blk / 2; i += blockDim.x) {
            const int a = (i / w) * 2 * w + (i % w), b = a + w;
            const float x = sv[a], y = sv[b];
            sv[a] = __fadd_rn(x, y);
            sv[b] = __fsub_rn(x, y);
        }
        __syncthreads();
    }
}

extern "C" __global__ void hy_prism_p(u16* __restrict__ p0, u16* __restrict__ p1, int pstride, const float* __restrict__ signs,
                                      int blk, int f16)
{
    extern __shared__ float sv[];
    const int r = blockIdx.y, b0 = blockIdx.x * blk;
    const size_t at = (size_t)r * pstride + b0;
    for (int i = threadIdx.x; i < blk; i += blockDim.x) sv[i] = __fmul_rn(__fadd_rn(bf2f(p0[at + i]), bf2f(p1[at + i])), signs[b0 + i]);
    __syncthreads();
    prism_fwht(sv, blk);
    const float inv = 1.0f / sqrtf((float)blk);
    for (int i = threadIdx.x; i < blk; i += blockDim.x) {
        float v = __fmul_rn(sv[i], inv);
        if (f16) v = h2f(f2h(v));
        put3(p0, p1, p1, at + i, v);
    }
}

extern "C" __global__ void hy_prism_h(const u16* src, u16* dst, int stride, const float* __restrict__ signs,
                                      int blk, int f16)
{
    extern __shared__ float sv[];
    const int r = blockIdx.y, b0 = blockIdx.x * blk;
    const size_t at = (size_t)r * stride + b0;
    for (int i = threadIdx.x; i < blk; i += blockDim.x) sv[i] = __fmul_rn(h2f(src[at + i]), signs[b0 + i]);
    __syncthreads();
    prism_fwht(sv, blk);
    const float inv = 1.0f / sqrtf((float)blk);
    for (int i = threadIdx.x; i < blk; i += blockDim.x) {
        float v = __fmul_rn(sv[i], inv);
        if (f16) v = h2f(f2h(v));
        dst[at + i] = f2h(v);
    }
}

// A q4_tiled matrix (per 32-column group [f16 scale][16 bytes: byte k's low
// nibble weight 2k, its high one 2k + 1], value (n - 8) s) as q4_tiled tiles
// (kind 10): lane l holds its A register r = 4 s + m's two nibbles at bits
// 8 (r & 3), + 4 of word r >> 2; S[t][g][16] the rows' f16 scales.
extern "C" __global__ void hy_repack4t(const u8* __restrict__ A, const u8* __restrict__ Bm, int pair, uint2* __restrict__ T,
                                       u16* __restrict__ S, int rows, int cols)
{
    const int gpr = cols >> 5, ntile = (rows + 15) >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ntile * gpr * 32) return;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    u32 w[2] = {0x88888888u, 0x88888888u};
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const bool second = pair && (r & 15) >= 8;
            const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
            const u8* t18 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 18 + 2;
            const int c = 16 * s + 8 * (m >> 1) + 2 * tig;
            const u32 byte = t18[c >> 1];
            const int reg = s * 4 + m;
            w[reg >> 2] = (w[reg >> 2] & ~(0xFFu << (8 * (reg & 3)))) | (byte << (8 * (reg & 3)));
        }
    T[idx] = make_uint2(w[0], w[1]);
    if (tig == 0) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = tile * 16 + gid + 8 * h;
            u16 sc = 0;
            if (r < rows) {
                const bool second = pair && (r & 15) >= 8;
                const int sr = pair ? (r >> 4) * 8 + (r & 7) : r;
                const u8* t18 = (second ? Bm : A) + ((size_t)sr * gpr + g) * 18;
                sc = (u16)(t18[0] | (t18[1] << 8));
            }
            S[tg * 16 + gid + 8 * h] = sc;
        }
    }
}

// `hy_expand_pr` for kind 10 (q4_tiled tiles).
extern "C" __global__ void hy_expand_q4gr(const uint2* __restrict__ T, const u16* __restrict__ S, u16* __restrict__ W, int rows,
                                          int cols, int tile0, int nt)
{
    const int gpr = cols >> 5;
    const size_t idx0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx0 >= (size_t)nt * gpr * 32) return;
    const size_t idx = idx0 + (size_t)tile0 * gpr * 32;
    const int lane = (int)(idx & 31);
    const size_t tg = idx >> 5;
    const int tile = (int)(tg / gpr), g = (int)(tg - (size_t)tile * gpr);
    const int gid = lane >> 2, tig = lane & 3;
    const uint2 X = T[idx];
    const u32 w[2] = {X.x, X.y};
    const int r0 = tile0 * 16;
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int r = tile * 16 + gid + 8 * (m & 1);
            if (r >= rows) continue;
            const int k = g * 32 + 16 * s + 8 * (m >> 1) + 2 * tig;
            const int reg = s * 4 + m;
            const u32 b = (w[reg >> 2] >> (8 * (reg & 3))) & 0xFFu;
            const float sc = h2f(S[tg * 16 + gid + 8 * (m & 1)]);
            const float lo = ((float)(b & 0xFu) - 8.0f) * sc;
            const float hi = ((float)(b >> 4) - 8.0f) * sc;
            *(u32*)(W + (size_t)(r - r0) * cols + k) = (u32)f2h(lo) | ((u32)f2h(hi) << 16);
        }
}

// ---------------------------------------------------------- mHC streams
//
// SenseNova-Loop's manifold hyper-connections (`sensenova::Mhc`): inside the
// range the residual is `ns` streams R [row][ns][dim]. Before a sub-layer
// `hy_mhc_route` RMS-normalizes a row's streams, takes the router's `out`
// logits, and writes the sub-layer's input h = pre_scale * sum_s pre_s R_s
// and the merge coefficients (post_i, then mix_ij, Sinkhorn-normalized);
// after it `hy_mhc_merge` makes R_i = sum_j mix_ij R_j + post_i y.
// `hy_mhc_enter` copies h into every stream, `hy_mhc_leave` takes their mean.

extern "C" __global__ void hy_mhc_enter(const float* __restrict__ h, float* __restrict__ R, int ns, int dim)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (e >= dim) return;
    const float v = h[(size_t)r * dim + e];
    for (int s = 0; s < ns; s++) R[((size_t)r * ns + s) * dim + e] = v;
}

extern "C" __global__ void hy_mhc_leave(const float* __restrict__ R, float* __restrict__ h, int ns, int dim)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (e >= dim) return;
    float v = 0.0f;
    for (int s = 0; s < ns; s++) v = __fadd_rn(v, R[((size_t)r * ns + s) * dim + e]);
    h[(size_t)r * dim + e] = __fdiv_rn(v, (float)ns);
}

// The router's logits of a row's streams: a block a (logit k, row), the
// streams' RMS and the k-th row of W against them; L[row][24].
extern "C" __global__ void hy_mhc_logits(const float* __restrict__ R, const float* __restrict__ W, float* __restrict__ L, int ns,
                                         int dim)
{
    __shared__ float red[2][32];
    const int k = blockIdx.x, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int n = ns * dim;
    const float* Rr = R + (size_t)r * n;
    const float* Wk = W + (size_t)k * n;
    float s2 = 0.0f, d = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float x = Rr[i];
        s2 = fmaf(x, x, s2);
        d = fmaf(Wk[i], x, d);
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(FULL, s2, o);
        d += __shfl_xor_sync(FULL, d, o);
    }
    if (lane == 0) {
        red[0][warp] = s2;
        red[1][warp] = d;
    }
    __syncthreads();
    if (tid == 0) {
        double ss = 0.0, dd = 0.0;
        for (int w = 0; w < NW; w++) {
            ss += (double)red[0][w];
            dd += (double)red[1][w];
        }
        L[(size_t)r * 24 + k] = (float)(dd / sqrt(ss / (double)n + 1e-6));
    }
}

// A block a row: the logits (`hy_mhc_logits`) plus the bias into pre, post
// and the Sinkhorn mix; the sub-layer's input rows into h.
extern "C" __global__ void hy_mhc_route(const float* __restrict__ R, const float* __restrict__ L, const float* __restrict__ bvec,
                                        float* __restrict__ h, float* __restrict__ coef, int ns, int dim, int out, int iters,
                                        int fixed, float pre_scale)
{
    __shared__ float pre_s[4];
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* Rr = R + (size_t)r * ns * dim;
    float* cr = coef + (size_t)r * 20;
    if (tid == 0) {
        float l[24];
        for (int k = 0; k < out; k++) l[k] = L[(size_t)r * 24 + k] + bvec[k];
        for (int s = 0; s < ns; s++) {
            pre_s[s] = 1.0f / (1.0f + expf(-l[s]));
            cr[s] = 2.0f / (1.0f + expf(-l[ns + s]));
        }
        // Sinkhorn in the log domain, f32 (a 4 x 4: the f64 exp / log of the
        // host's took 159 us a call on the card)
        float m[4][4];
        for (int i = 0; i < ns; i++)
            for (int j = 0; j < ns; j++) m[i][j] = fixed ? 0.0f : l[2 * ns + i * ns + j];
        if (!fixed) {
            for (int it = 0; it < iters; it++) {
                for (int i = 0; i < ns; i++) {
                    float mx = -3.0e38f;
                    for (int j = 0; j < ns; j++) mx = fmaxf(mx, m[i][j]);
                    float se = 0.0f;
                    for (int j = 0; j < ns; j++) se += expf(m[i][j] - mx);
                    const float lse = mx + logf(se);
                    for (int j = 0; j < ns; j++) m[i][j] -= lse;
                }
                for (int j = 0; j < ns; j++) {
                    float mx = -3.0e38f;
                    for (int i = 0; i < ns; i++) mx = fmaxf(mx, m[i][j]);
                    float se = 0.0f;
                    for (int i = 0; i < ns; i++) se += expf(m[i][j] - mx);
                    const float lse = mx + logf(se);
                    for (int i = 0; i < ns; i++) m[i][j] -= lse;
                }
            }
        }
        for (int i = 0; i < ns; i++)
            for (int j = 0; j < ns; j++) cr[4 + i * 4 + j] = fixed ? (i == j ? 1.0f : 0.0f) : expf(m[i][j]);
    }
    __syncthreads();
    for (int e = tid; e < dim; e += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < ns; s++) v = fmaf(pre_s[s], Rr[s * dim + e], v);
        h[(size_t)r * dim + e] = __fmul_rn(v, pre_scale);
    }
}

extern "C" __global__ void hy_mhc_merge(float* __restrict__ R, const float* __restrict__ y, const float* __restrict__ coef, int ns,
                                        int dim)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (e >= dim) return;
    const float* cr = coef + (size_t)r * 20;
    float* Rr = R + (size_t)r * ns * dim;
    float t[4];
    const float yv = y[(size_t)r * dim + e];
    for (int i = 0; i < ns; i++) {
        float v = __fmul_rn(cr[i], yv);
        for (int j = 0; j < ns; j++) v = fmaf(cr[4 + i * 4 + j], Rr[j * dim + e], v);
        t[i] = v;
    }
    for (int i = 0; i < ns; i++) Rr[i * dim + e] = t[i];
}

// ------------------------------------------------------------------ DASH
//
// Unlimited-OCR's block-sparse attend (`gpu_wgpu/uocr.rs` has the semantics):
// a router a row (its q and k index rows against the normed row, RMSNorm and
// rotation, the key into its slot of the open block, the candidate blocks'
// scores, entmax-1.5 over the best 16, the read set and its centered log
// priors), a closing block's summary key, and the attend over the read set
// and the local window from the mirror's f16 rows. Index width 128, blocks
// of at most 64 rows, at most 16 blocks read, 512 rows a query. `rec`: the
// call's first position and the router's block count (0: every block so far);
// `tab`: the rows' rotation table (sin, cos a pair, `hy_qk`'s), 64 pairs a row.

#define DASH_ACT 10000.0f
#define DASH_NONE 0xffffffffu

__device__ __forceinline__ float dash_clamp(float x) { return fminf(fmaxf(x, -DASH_ACT), DASH_ACT); }

extern "C" __global__ void __launch_bounds__(256) hy_dash_route(
    const float* __restrict__ xn, int hidden, const float* __restrict__ W, const float* __restrict__ qn,
    const float* __restrict__ kn, const float* __restrict__ tab, float* __restrict__ open, const float* __restrict__ kbar,
    const unsigned* __restrict__ coff, const unsigned* __restrict__ cands, float* __restrict__ sc,
    unsigned* __restrict__ sel, float* __restrict__ sbias, const int* __restrict__ rec, int n, int block, int kk,
    float eps, float idx_scale, float floor_, float sigma)
{
    const int row = blockIdx.x, lid = threadIdx.x, lane = lid & 31, warp = lid >> 5;
    if (row >= n) return;
    const int d = 128, half = 64;
    const unsigned pos = (unsigned)(rec[0] + row);
    __shared__ float rr[256];
    __shared__ float qs[128];
    __shared__ float red[8];
    // lane lid: row lid of the router (the q rows, then the k rows) against x
    {
        const float4* x4 = (const float4*)(xn + (size_t)row * hidden);
        const float4* w4 = (const float4*)(W + (size_t)lid * hidden);
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int c = 0; c < hidden / 4; c++) {
            const float4 a = w4[c], b = x4[c];
            acc.x = fmaf(a.x, b.x, acc.x);
            acc.y = fmaf(a.y, b.y, acc.y);
            acc.z = fmaf(a.z, b.z, acc.z);
            acc.w = fmaf(a.w, b.w, acc.w);
        }
        rr[lid] = acc.x + acc.y + acc.z + acc.w;
    }
    // the two sums of squares: warps 0..3 (q), 4..7 (k)
    float s2 = rr[lid] * rr[lid];
    for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(FULL, s2, o);
    if (lane == 0) red[warp] = s2;
    __syncthreads();
    const float inv_q = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)d + eps);
    const float inv_k = rsqrtf((red[4] + red[5] + red[6] + red[7]) / (float)d + eps);
    if (lid < half) {
        const int j = lid;
        const float s = tab[((size_t)row * half + j) * 2], c = tab[((size_t)row * half + j) * 2 + 1];
        const float a = dash_clamp(dash_clamp(rr[j] * inv_q) * (1.0f + qn[j]));
        const float b = dash_clamp(dash_clamp(rr[half + j] * inv_q) * (1.0f + qn[half + j]));
        qs[j] = a * c - b * s;
        qs[half + j] = b * c + a * s;
    } else if (lid >= d && lid < d + half) {
        const int j = lid - d;
        const float s = tab[((size_t)row * half + j) * 2], c = tab[((size_t)row * half + j) * 2 + 1];
        const float a = dash_clamp(dash_clamp(rr[d + j] * inv_k) * (1.0f + kn[j]));
        const float b = dash_clamp(dash_clamp(rr[d + half + j] * inv_k) * (1.0f + kn[half + j]));
        const size_t at = ((size_t)((pos / block) % 3u) * block + pos % block) * d;
        open[at + j] = a * c - b * s;
        open[at + half + j] = b * c + a * s;
    }
    __syncthreads();
    const unsigned c0 = coff[row], ncand = coff[row + 1] - c0;
    const float inv_sqrt = rsqrtf((float)d);
    for (unsigned i = lid; i < ncand; i += 256) {
        const float4* k4 = (const float4*)(kbar + (size_t)cands[c0 + i] * d);
        const float4* q4 = (const float4*)qs;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int e = 0; e < d / 4; e++) {
            const float4 a = q4[e], b = k4[e];
            acc.x = fmaf(a.x, b.x, acc.x);
            acc.y = fmaf(a.y, b.y, acc.y);
            acc.z = fmaf(a.z, b.z, acc.z);
            acc.w = fmaf(a.w, b.w, acc.w);
        }
        sc[c0 + i] = dash_clamp((acc.x + acc.y + acc.z + acc.w) * inv_sqrt);
    }
    __syncthreads();
    if (lid != 0) return;
    float vals[16], taus[16], logp[16];
    unsigned ids[16];
    const unsigned nblk_ = rec[1] != 0 ? (unsigned)rec[1] : (pos + block) / block;
    const unsigned m = min(max(min(16u, nblk_), 1u), 16u);
    for (unsigned t = 0; t < m; t++) {
        vals[t] = -10000.0f;
        ids[t] = DASH_NONE;
    }
    unsigned cnt = 0;
    for (unsigned t = 0; t < ncand; t++) {
        const float x = sc[c0 + t] * idx_scale;
        if (cnt == m && x <= vals[m - 1]) continue;
        unsigned j = min(cnt, m - 1);
        while (j != 0 && vals[j - 1] < x) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = x;
        ids[j] = cands[c0 + t];
        cnt = min(cnt + 1, m);
    }
    unsigned count = 0;
    if (cnt > 0) {
        const float top = vals[0];
        float sum = 0.0f, sum2 = 0.0f;
        unsigned support = 0;
        for (unsigned t = 0; t < m; t++) {
            const float y = (vals[t] - top) / 2.0f;
            sum += y;
            sum2 += y * y;
            const float rho = (float)(t + 1);
            const float mean = sum / rho;
            const float ss = rho * (sum2 / rho - mean * mean);
            const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
            const float tau = mean - sqrtf(delta);
            taus[t] = tau;
            if (tau <= y) support++;
        }
        const float tau_star = taus[max(support, 1u) - 1];
        const unsigned k = min(max(min((unsigned)kk, m), 1u), 16u);
        float mu = 0.0f;
        unsigned valid = 0;
        for (unsigned t = 0; t < k; t++) {
            const float y = (vals[t] - top) / 2.0f - tau_star;
            logp[t] = logf(fmaxf(y > 0.0f ? y * y : 0.0f, floor_));
            if (ids[t] != DASH_NONE) {
                mu += logp[t];
                valid++;
            }
        }
        mu /= (float)max(valid, 1u);
        for (unsigned t = 0; t < k; t++) {
            if (ids[t] == DASH_NONE) continue;
            sel[row * 17 + 1 + count] = ids[t];
            sbias[row * 16 + count] = (logp[t] - mu) / sigma;
            count++;
        }
    }
    sel[row * 17] = count;
}

// The blocks closing in the call's rows: a block's keys against head_cls,
// their softmax, the weighted mean key into kbar[block] (one block of 128).
extern "C" __global__ void __launch_bounds__(128) hy_dash_close(const float* __restrict__ open, const float* __restrict__ hc,
                                                                float* __restrict__ kbar, const int* __restrict__ rec, int n,
                                                                int block)
{
    __shared__ float aw[64];
    const int tid = threadIdx.x, d = 128;
    const int pos0 = rec[0], end = pos0 + n;
    const float inv_sqrt = rsqrtf((float)d);
    for (int b = pos0 / block; b <= (end - 1) / block; b++) {
        if ((b + 1) * block > end) break;
        const size_t at = (size_t)(b % 3) * block;
        if (tid < block) {
            const float4* k4 = (const float4*)(open + (at + tid) * d);
            const float4* h4 = (const float4*)hc;
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int e = 0; e < d / 4; e++) {
                const float4 a = k4[e], c = h4[e];
                acc.x = fmaf(a.x, c.x, acc.x);
                acc.y = fmaf(a.y, c.y, acc.y);
                acc.z = fmaf(a.z, c.z, acc.z);
                acc.w = fmaf(a.w, c.w, acc.w);
            }
            aw[tid] = dash_clamp((acc.x + acc.y + acc.z + acc.w) * inv_sqrt);
        }
        __syncthreads();
        if (tid == 0) {
            float mx = -1.0e30f;
            for (int t = 0; t < block; t++) mx = fmaxf(mx, aw[t]);
            float den = 0.0f;
            for (int t = 0; t < block; t++) {
                const float v = expf(aw[t] - mx);
                aw[t] = v;
                den += v;
            }
            den = 1.0f / fmaxf(den, 1.0e-6f);
            for (int t = 0; t < block; t++) aw[t] *= den;
        }
        __syncthreads();
        float acc = 0.0f;
        for (int t = 0; t < block; t++) acc = fmaf(aw[t], open[(at + t) * d + tid], acc);
        kbar[(size_t)b * d + tid] = acc;
        __syncthreads();
    }
}

// The attend of head h of row b (grid (nh, n), 256 threads) over the read set
// and the local window: q from `hy_qk`'s layout [kv head][row][head of the
// group][HD], K / V the mirror's f16 rows [slot][nkv][HD] (position r at slot
// slot0 + r - pos0); the output (gated) into the planes, or f16 rows [n][nh][HD].
extern "C" __global__ void __launch_bounds__(256) hy_dash_attend(
    const float* __restrict__ qg, const u16* __restrict__ Km, const u16* __restrict__ Vm, const unsigned* __restrict__ sel,
    const float* __restrict__ sbias, const float* __restrict__ gate, u16* __restrict__ p0, u16* __restrict__ p1, int pstride,
    u16* __restrict__ oh, const int* __restrict__ rec, const int* __restrict__ ci, int slot0_, int nh, int nkv, int n, int block,
    int window, float scale)
{
    const int h = blockIdx.x, row = blockIdx.y, lid = threadIdx.x;
    if (h >= nh || row >= n) return;
    const int hpk = nh / nkv, g = h / hpk, jh = h - g * hpk;
    const int pos0 = rec[0], pos = pos0 + row;
    const int slot0 = ci ? ci[1] : slot0_;
    const unsigned* s = sel + row * 17;
    const int remote = (int)s[0] * block, total = remote + window;
    const int lo = max(pos + 1 - window, 0);
    __shared__ float wsc[512];
    __shared__ int wrow[512];
    __shared__ float red[256];
    __shared__ float4 vsum[256];
    __shared__ float qsh[HD];
    const float* q = qg + (((size_t)g * n + row) * hpk + jh) * HD;
    for (int i = lid; i < HD; i += 256) qsh[i] = q[i];
    __syncthreads();
    float mym = -1.0e30f;
    for (int part = 0; part < 2; part++) {
        const int i = lid + part * 256;
        int r = -1;
        float bias = 0.0f;
        if (i < remote) {
            const int sb = i / block;
            const int rr = (int)s[1 + sb] * block + i % block;
            // the local window wins on overlap
            if (rr < lo) {
                r = rr;
                bias = sbias[row * 16 + sb];
            }
        } else if (i < total) {
            const int rr = lo + i - remote;
            if (rr <= pos) r = rr;
        }
        wrow[i] = r;
        if (r >= 0) {
            const u16* kr = Km + ((size_t)(slot0 + r - pos0) * nkv + g) * HD;
            float acc = 0.0f;
            for (int e = 0; e < HD; e++) acc = fmaf(qsh[e], h2f(kr[e]), acc);
            const float v = acc * scale + bias;
            wsc[i] = v;
            mym = fmaxf(mym, v);
        }
    }
    red[lid] = mym;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) red[lid] = fmaxf(red[lid], red[lid + st]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int part = 0; part < 2; part++) {
        const int i = lid + part * 256;
        if (wrow[i] >= 0) {
            const float e = expf(wsc[i] - mx);
            wsc[i] = e;
            sum += e;
        }
    }
    red[lid] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) red[lid] += red[lid + st];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    // the rows split over the eight warps: (warp lid >> 5, four values lid & 31)
    const int quad = lid & 31, warp = lid >> 5;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (quad < HD / 4) {
        for (int i = warp; i < total; i += 8) {
            const int r = wrow[i];
            if (r < 0) continue;
            const u16* vr = Vm + ((size_t)(slot0 + r - pos0) * nkv + g) * HD + quad * 4;
            const float w = wsc[i];
            acc.x = fmaf(w, h2f(vr[0]), acc.x);
            acc.y = fmaf(w, h2f(vr[1]), acc.y);
            acc.z = fmaf(w, h2f(vr[2]), acc.z);
            acc.w = fmaf(w, h2f(vr[3]), acc.w);
        }
    }
    vsum[lid] = acc;
    __syncthreads();
    for (int wst = 4; wst > 0; wst >>= 1) {
        if (warp < wst) {
            float4 a = vsum[lid], b = vsum[lid + wst * 32];
            vsum[lid] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
        __syncthreads();
    }
    if (warp == 0 && quad < HD / 4) {
        const float4 v = vsum[lid];
        const float o[4] = {v.x * inv, v.y * inv, v.z * inv, v.w * inv};
        for (int c = 0; c < 4; c++) {
            const int dd = quad * 4 + c;
            float a = o[c];
            if (gate) a = __fmul_rn(a, sigm(gate[((size_t)row * nh + h) * HD + dd]));
            if (p0) put3(p0, p1, p1, (size_t)row * pstride + (size_t)h * HD + dd, a);
            else oh[((size_t)row * nh + h) * HD + dd] = f2h_sat(a);
        }
    }
}

// ------------------------------------------------------------------- KDA
//
// Kimi Delta Attention (`linear_core::kda_step`): the GDN front's conv,
// SiLU and q / k L2 norms (`hy_gdn_front` over the q | k | v channels), then
// a decay per key channel and the delta rule with it.

// gd [n][nh dk] = exp(log-decay(f + dt_bias)), beta [n][nh] = sigmoid(b):
// log-decay -exp(A) softplus(x), or lb sigmoid(exp(A) x) (`lb_on`); A by
// head, by key dim or by channel as `a_len` says.
extern "C" __global__ void hy_kda_gates(const float* __restrict__ f, int fstride, const float* __restrict__ braw, int bstride,
                                        const float* __restrict__ a_log, int a_len, const float* __restrict__ dt_bias,
                                        float* __restrict__ gd, float* __restrict__ beta, int n, int nh, int dk, int lb_on,
                                        float lb)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int kd = nh * dk;
    if (i >= (size_t)n * kd) return;
    const int t = (int)(i / kd), c = (int)(i % kd), h = c / dk, d = c - h * dk;
    const float a = a_len == nh ? a_log[h] : (a_len == dk ? a_log[d] : a_log[c]);
    const float x = __fadd_rn(f[(size_t)t * fstride + c], dt_bias[c]);
    float ld;
    if (lb_on) ld = lb / (1.0f + expf(-__fmul_rn(expf(a), x)));
    else ld = -__fmul_rn(expf(a), x > 20.0f ? x : log1pf(expf(x)));
    gd[i] = expf(ld);
    if (d == 0) beta[(size_t)t * nh + h] = 1.0f / (1.0f + expf(-braw[(size_t)t * bstride + h]));
}

// The KDA step with a lane a state column for dk = 128, `gdn_scan4_body`'s
// layout: the key channels' decay folds into the k readout (kv = (k gd)ᵀ S),
// delta = beta (v - kv), S = gd S + k deltaᵀ, o = qᵀ S. cq rows [c_dim]
// (q | k | v, normed), gd [n][nh 128], beta [n][nh]; n <= 16.
extern "C" __global__ void __launch_bounds__(128) hy_kda_scan4(const float* __restrict__ cq, const float* __restrict__ gdr,
                                                              const float* __restrict__ betar, float* S, float* __restrict__ o,
                                                              int n, int c_dim, int nh, int dv, float* __restrict__ ring,
                                                              const float* __restrict__ qkv, int qstride, int kk)
{
    const int sblocks = nh * (dv >> 5);
    if ((int)blockIdx.x >= sblocks) {
        // blocks past the state's move the conv ring, as `hy_gdn_scan4`'s do
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
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int wpr = dv >> 5;
    const int h = blockIdx.x / wpr, j = (blockIdx.x - h * wpr) * 32 + lane;
    const int kd = nh * 128;
    const size_t base = ((size_t)h * 128 + w * 32) * dv + j;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = S[base + (size_t)i * dv];
    __shared__ float ks[16][128], qs[16][128], gs[16][128];
    __shared__ float bs[16];
    __shared__ float part[2][4][32];
    const int nn = min(n, 16);
    for (int t = w; t < nn; t += 4) {
        const float* row = cq + (size_t)t * c_dim;
        *(float4*)(&ks[t][lane * 4]) = *(const float4*)(row + kd + (size_t)h * 128 + lane * 4);
        *(float4*)(&qs[t][lane * 4]) = *(const float4*)(row + (size_t)h * 128 + lane * 4);
        *(float4*)(&gs[t][lane * 4]) = *(const float4*)(gdr + (size_t)t * kd + (size_t)h * 128 + lane * 4);
        if (lane == 0) bs[t] = betar[(size_t)t * nh + h];
    }
    float vv[16];
#pragma unroll
    for (int t = 0; t < 16; t++) vv[t] = t < nn ? cq[(size_t)t * c_dim + 2 * kd + (size_t)h * dv + j] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t >= nn) break;
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
            a0 = fmaf(s[i], __fmul_rn(ks[t][w * 32 + i], gs[t][w * 32 + i]), a0);
            a1 = fmaf(s[i + 1], __fmul_rn(ks[t][w * 32 + i + 1], gs[t][w * 32 + i + 1]), a1);
        }
        part[0][w][lane] = __fadd_rn(a0, a1);
        __syncthreads();
        const float kv = __fadd_rn(__fadd_rn(__fadd_rn(part[0][0][lane], part[0][1][lane]), part[0][2][lane]), part[0][3][lane]);
        const float delta = __fmul_rn(__fsub_rn(vv[t], kv), bs[t]);
        float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
            s[i] = fmaf(ks[t][w * 32 + i], delta, __fmul_rn(gs[t][w * 32 + i], s[i]));
            s[i + 1] = fmaf(ks[t][w * 32 + i + 1], delta, __fmul_rn(gs[t][w * 32 + i + 1], s[i + 1]));
            b0 = fmaf(qs[t][w * 32 + i], s[i], b0);
            b1 = fmaf(qs[t][w * 32 + i + 1], s[i + 1], b1);
        }
        part[1][w][lane] = __fadd_rn(b0, b1);
        __syncthreads();
        if (w == 0 && o)
            o[(size_t)t * nh * dv + (size_t)h * dv + j] =
                __fadd_rn(__fadd_rn(__fadd_rn(part[1][0][lane], part[1][1][lane]), part[1][2][lane]), part[1][3][lane]);
    }
#pragma unroll
    for (int i = 0; i < 32; i++) S[base + (size_t)i * dv] = s[i];
}

// ------------------------------------------------------------------- MLA
//
// DeepSeek-V2 Multi-head Latent Attention as multi-head attention
// (`pipeline::mla_attention`): the latent's RMS norm into the planes for
// kv_b, the heads' keys [k_rope | k_nope] and values [v | 0] (as wide as the
// keys), and the attention's output heads cut back to v wide.

// The latent (the first `lora` of a row `stride` long) RMS-normed with w, a
// block a row, into the planes (`pstride` a row).
extern "C" __global__ void hy_mla_lat(const float* __restrict__ kva, int stride, const float* __restrict__ w, int lora,
                                      float eps, u16* __restrict__ p0, u16* __restrict__ p1, int pstride)
{
    __shared__ float red[32];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const float* x = kva + (size_t)r * stride;
    float s = 0.0f;
    for (int i = tid; i < lora; i += blockDim.x) s = fmaf(x[i], x[i], s);
    for (int k = 16; k > 0; k >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, k));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = 1.0f / (float)sqrt(tot / (double)lora + (double)eps);
    for (int i = tid; i < lora; i += blockDim.x) put3(p0, p1, p1, (size_t)r * pstride + i, __fmul_rn(__fmul_rn(x[i], inv), w[i]));
}

// k [n][nh][rope + nope] = [k_rope (the row's shared rope key) | k_nope],
// v [n][nh][rope + nope] = [v | 0]; kva rows `kstride` long (the latent,
// then the rope key), kvb [n][nh][nope + v].
extern "C" __global__ void hy_mla_kv(const float* __restrict__ kva, int kstride, const float* __restrict__ kvb, float* __restrict__ k,
                                     float* __restrict__ v, int n, int nh, int rope, int nope, int vd, int lora)
{
    const int hd = rope + nope;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * nh * hd) return;
    const int d = (int)(i % hd);
    const size_t th = i / hd;
    const int h = (int)(th % nh), t = (int)(th / nh);
    const float* b = kvb + ((size_t)t * nh + h) * (nope + vd);
    k[i] = d < rope ? kva[(size_t)t * kstride + lora + d] : b[d - rope];
    v[i] = d < vd ? b[nope + d] : 0.0f;
}

// Planes [n][nh][hd] (the attention's heads) into planes [n][nh][vd] (the
// first vd of each head), for the output projection.
extern "C" __global__ void hy_mla_cut(const u16* __restrict__ s0, const u16* __restrict__ s1, int sstride, u16* __restrict__ d0,
                                      u16* __restrict__ d1, int dstride, int n, int nh, int hd, int vd)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * nh * vd) return;
    const int e = (int)(i % vd);
    const size_t th = i / vd;
    const int h = (int)(th % nh), t = (int)(th / nh);
    const size_t src = (size_t)t * sstride + (size_t)h * hd + e, dst = (size_t)t * dstride + (size_t)h * vd + e;
    d0[dst] = s0[src];
    d1[dst] = s1[src];
}
