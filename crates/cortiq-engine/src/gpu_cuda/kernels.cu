// CUDA kernels of the native backend (`gpu_cuda.rs`), compiled by NVRTC on
// first use. Activations are f32 everywhere; weights stay in their codec
// (q4tp nibbles, q8_2f int8) and are unpacked by the thread that reads them.
//
// Layouts:
//   q4tp   nib   u32[rows * gpr * 4]   16 B of nibbles per 32-weight group
//          codes u8 [rows * gpr]       the group's rung of the row ladder
//          lad   f32[rows * 32]        the row's 32 scales (host-expanded)
//   q8_2f  body  i8 [rows * cols]
//          rs    f32[rows]             row scales
//          col   f32[cols]             column field
//   f32    w     f32[rows * cols]
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;

struct MvJob {
    u64 w;     // weight body
    u64 aux0;  // q4tp: codes, q8_2f: row scales
    u64 aux1;  // q4tp: ladder, q8_2f: column field
    u64 y;     // output [rows]
    u64 res;   // 0, or the rows added to the product (a residual stream)
    int rows;
    int kind;  // 6 = q4tp, 7 = q8_2f, 4 = f32
    int blk0;  // first block of this job in the launch
    int rpb;   // rows per block
};

#define INF __int_as_float(0x7f800000)

// act(gate) of a dense FFN: 0 SiLU, 1 exact GELU (erf by Abramowitz-Stegun
// 7.1.26, the host's `inference::gelu_erf`), 2 tanh GELU.
__device__ __forceinline__ float act_f(float g, int act) {
    if (act == 0) return g / (1.0f + expf(-g));
    if (act == 1) {
        float x = g * 0.70710678118654752440f;
        float a = fabsf(x);
        float t = 1.0f / (1.0f + 0.3275911f * a);
        float y = 1.0f - (((((1.0614054f * t - 1.4531521f) * t + 1.4214138f) * t - 0.28449674f) * t + 0.2548296f) * t) * expf(-a * a);
        float e = x < 0.0f ? -y : y;
        return 0.5f * g * (1.0f + e);
    }
    return 0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * g * g * g)));
}

// Eight nibbles of one word against eight staged inputs `st` floats apart.
__device__ __forceinline__ float nib8(u32 w, const float* p, int st, float a) {
    a += (float)(w & 15u) * p[0];
    a += (float)((w >> 4) & 15u) * p[st];
    a += (float)((w >> 8) & 15u) * p[2 * st];
    a += (float)((w >> 12) & 15u) * p[3 * st];
    a += (float)((w >> 16) & 15u) * p[4 * st];
    a += (float)((w >> 20) & 15u) * p[5 * st];
    a += (float)((w >> 24) & 15u) * p[6 * st];
    a += (float)(w >> 28) * p[7 * st];
    return a;
}

// Four int8 of one word against four staged inputs `st` floats apart.
__device__ __forceinline__ float i8x4(u32 w, const float* p, int st, float a) {
    a += (float)((int)(w << 24) >> 24) * p[0];
    a += (float)((int)(w << 16) >> 24) * p[st];
    a += (float)((int)(w << 8) >> 24) * p[2 * st];
    a += (float)((int)w >> 24) * p[3 * st];
    return a;
}

// One launch, several matrices against one input row: y_j = (res_j +) W_j x.
// The input is staged into shared memory once per block, with what precedes
// the product folded in: the RMS norm of the row (`nw`), a dense FFN's
// act(x)*x2, a q8_2f job's column field. A warp owns a row; its lanes own
// the row's groups.
//
// The staged row is TRANSPOSED: lane l of a warp reads element k of group
// l + 32i, and in the natural order (group-major) all 32 lanes would hit the
// same shared-memory bank on every load. Element-major with an odd stride
// puts the lanes of one load, and the threads of one staging store, on
// different banks:
//   q4tp   xs[k * GP + g] = x[32 g + k]   GP = groups | 1
//   q8_2f  xs[k * CP + c] = x[16 c + k]   CP = 16-byte chunks | 1
//   f32    xs[c] = x[c]
extern "C" __global__ void mv_multi(const MvJob* __restrict__ jobs, int njobs,
                                    const float* __restrict__ x, const float* __restrict__ x2, int act,
                                    const float* __restrict__ nw, float eps, int gemma, int cols)
{
    extern __shared__ float sh[];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    int j = 0;
    while (j + 1 < njobs && (int)blockIdx.x >= jobs[j + 1].blk0) j++;
    const MvJob jb = jobs[j];
    const int cpad = (cols + 31) & ~31;
    float* red = sh;             // [16]
    float* xs = sh + 16;         // [cpad + 64]
    float* gs = xs + cpad + 64;  // [cpad / 32 + 8]
    const int gpr = cols >> 5, nch = cols >> 4;
    const int GP = gpr | 1, CP = nch | 1;

    float inv = 1.0f;
    if (nw) {
        // float partials (a thread's few terms, then the warp, then the
        // warps in double): pairwise in effect, and no f64 arithmetic in
        // the loop, which this class of card runs at a small fraction of f32
        float s = 0.0f;
        for (int i = tid; i < cols; i += T) {
            float v = x[i];
            s += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[warp] = s;
        __syncthreads();
        double tot = 0.0;
        for (int w = 0; w < NW; w++) tot += (double)red[w];
        inv = 1.0f / (float)sqrt(tot / (double)cols + (double)eps);
    }
    for (int i = tid; i < cols; i += T) {
        float v = x[i];
        if (nw) v = __fmul_rn(__fmul_rn(v, inv), gemma ? 1.0f + nw[i] : nw[i]);
        if (act >= 0) v = act_f(v, act) * x2[i];
        int idx = i;
        if (jb.kind == 6) {
            idx = (i & 31) * GP + (i >> 5);
        } else if (jb.kind == 7) {
            v *= ((const float*)jb.aux1)[i];
            idx = (i & 15) * CP + (i >> 4);
        }
        xs[idx] = v;
    }
    __syncthreads();
    if (jb.kind == 6) {
        for (int g = tid; g < gpr; g += T) {
            const float* p = xs + g;
            float s = 0.0f;
            for (int k = 0; k < 32; k++) s += p[k * GP];
            gs[g] = 8.0f * s;
        }
        __syncthreads();
    }
    const int row0 = ((int)blockIdx.x - jb.blk0) * jb.rpb;
    const int row1 = min(row0 + jb.rpb, jb.rows);
    float* y = (float*)jb.y;
    const float* res = (const float*)jb.res;
    for (int r = row0 + warp; r < row1; r += NW) {
        float acc = 0.0f;
        if (jb.kind == 6) {
            const uint4* W = (const uint4*)jb.w + (size_t)r * gpr;
            const u8* C = (const u8*)jb.aux0 + (size_t)r * gpr;
            // The row's 32 scales sit in the lanes' registers: a group's
            // scale is a shuffle, not a second load that waits on the code
            // (measured on an A40: down 46 -> 37 us, o 16.5 -> 14 us). A
            // shuffle needs the whole warp at one program point, so every
            // lane runs every iteration; a lane past the row's last group
            // loads nothing and adds nothing.
            const float lad = ((const float*)jb.aux1)[(size_t)r * 32 + lane];
            const int iters = (gpr + 31) >> 5;
            for (int it = 0; it < iters; it++) {
                const int g = lane + (it << 5);
                float a = 0.0f, s0 = 0.0f;
                int code = 0;
                if (g < gpr) {
                    const uint4 w = W[g];
                    code = C[g];
                    const float* p = xs + g;
                    a = nib8(w.x, p, GP, a);
                    a = nib8(w.y, p + 8 * GP, GP, a);
                    a = nib8(w.z, p + 16 * GP, GP, a);
                    a = nib8(w.w, p + 24 * GP, GP, a);
                    s0 = gs[g];
                }
                acc += __shfl_sync(0xffffffffu, lad, code) * (a - s0);
            }
        } else if (jb.kind == 7) {
            const uint4* W = (const uint4*)(jb.w + (size_t)r * cols);
            for (int c = lane; c < nch; c += 32) {
                const uint4 w = W[c];
                const float* p = xs + c;
                float a = 0.0f;
                a = i8x4(w.x, p, CP, a);
                a = i8x4(w.y, p + 4 * CP, CP, a);
                a = i8x4(w.z, p + 8 * CP, CP, a);
                a = i8x4(w.w, p + 12 * CP, CP, a);
                acc += a;
            }
            acc *= ((const float*)jb.aux0)[r];
        } else {
            const float* W = (const float*)jb.w + (size_t)r * cols;
            for (int c = lane; c < cols; c += 32) acc += W[c] * xs[c];
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[r] = res ? res[r] + acc : acc;
    }
}

// f32 -> IEEE half, round to nearest (what `__float2half` of cuda_fp16.h is;
// spelled out because NVRTC is given no include path).
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

// The same with values past the half range saturated instead of made infinite.
__device__ __forceinline__ unsigned short f2h_sat(float f) {
    return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

__device__ __forceinline__ u64 pack4h(float a, float b, float c, float d) {
    return (u64)f2h(a) | ((u64)f2h(b) << 16) | ((u64)f2h(c) << 32) | ((u64)f2h(d) << 48);
}

// RMS norm of every row of x [B][n] (a block per row): f32 into `out`, or —
// `half` set — f16 into it, the operand of a tensor-core GEMM.
extern "C" __global__ void rmsnorm_rows(const float* __restrict__ x, const float* __restrict__ w,
                                        void* __restrict__ out, int n, float eps, int gemma, int half)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* xr = x + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = tid; i < n; i += T) {
        float v = xr[i];
        s += v * v;
    }
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = 1.0f / (float)sqrt(tot / (double)n + (double)eps);
    if (half) {
        unsigned short* o = (unsigned short*)out + (size_t)blockIdx.x * n;
        for (int i = tid; i < n; i += T) o[i] = f2h_sat(__fmul_rn(__fmul_rn(xr[i], inv), gemma ? 1.0f + w[i] : w[i]));
    } else {
        float* o = (float*)out + (size_t)blockIdx.x * n;
        for (int i = tid; i < n; i += T) o[i] = __fmul_rn(__fmul_rn(xr[i], inv), gemma ? 1.0f + w[i] : w[i]);
    }
}

// q4tp -> dense f32 [rows][cols], a thread per four weights.
extern "C" __global__ void expand_q4tp(const u32* __restrict__ nib, const u8* __restrict__ codes,
                                       const float* __restrict__ lad, float4* __restrict__ W, int rows, int cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2, gpr = cols >> 5;
    if (idx >= (size_t)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (size_t)r * qpr);
    const int g = q >> 3;
    const u32 w = nib[((size_t)r * gpr + g) * 4 + ((q & 7) >> 1)];
    const u32 h = (q & 1) ? (w >> 16) : w;
    const float s = lad[(size_t)r * 32 + codes[(size_t)r * gpr + g]];
    W[idx] = make_float4(__fmul_rn((float)(h & 15u) - 8.0f, s), __fmul_rn((float)((h >> 4) & 15u) - 8.0f, s),
                         __fmul_rn((float)((h >> 8) & 15u) - 8.0f, s), __fmul_rn((float)((h >> 12) & 15u) - 8.0f, s));
}

// q8_2f -> dense f32 [rows][cols], a thread per four weights (cols % 4 == 0).
extern "C" __global__ void expand_q8(const u32* __restrict__ body, const float* __restrict__ rs,
                                     const float4* __restrict__ col, float4* __restrict__ W, int rows, int cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2;
    if (idx >= (size_t)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (size_t)r * qpr);
    const u32 w = body[idx];
    const float s = rs[r];
    const float4 c = col[q];
    W[idx] = make_float4(__fmul_rn(__fmul_rn((float)((int)(w << 24) >> 24), s), c.x),
                         __fmul_rn(__fmul_rn((float)((int)(w << 16) >> 24), s), c.y),
                         __fmul_rn(__fmul_rn((float)((int)(w << 8) >> 24), s), c.z),
                         __fmul_rn(__fmul_rn((float)((int)w >> 24), s), c.w));
}

// q4tp -> dense f16 [rows][cols] (the tensor-core GEMM's weight plane).
extern "C" __global__ void expand_q4tp_h(const u32* __restrict__ nib, const u8* __restrict__ codes,
                                         const float* __restrict__ lad, u64* __restrict__ W, int rows, int cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2, gpr = cols >> 5;
    if (idx >= (size_t)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (size_t)r * qpr);
    const int g = q >> 3;
    const u32 w = nib[((size_t)r * gpr + g) * 4 + ((q & 7) >> 1)];
    const u32 h = (q & 1) ? (w >> 16) : w;
    const float s = lad[(size_t)r * 32 + codes[(size_t)r * gpr + g]];
    W[idx] = pack4h(((float)(h & 15u) - 8.0f) * s, ((float)((h >> 4) & 15u) - 8.0f) * s,
                    ((float)((h >> 8) & 15u) - 8.0f) * s, ((float)((h >> 12) & 15u) - 8.0f) * s);
}

// q8_2f -> dense f16 [rows][cols].
extern "C" __global__ void expand_q8_h(const u32* __restrict__ body, const float* __restrict__ rs,
                                       const float4* __restrict__ col, u64* __restrict__ W, int rows, int cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2;
    if (idx >= (size_t)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (size_t)r * qpr);
    const u32 w = body[idx];
    const float s = rs[r];
    const float4 c = col[q];
    W[idx] = pack4h((float)((int)(w << 24) >> 24) * s * c.x, (float)((int)(w << 16) >> 24) * s * c.y,
                    (float)((int)(w << 8) >> 24) * s * c.z, (float)((int)w >> 24) * s * c.w);
}

// f32 -> f16 over n / 4 quads, saturating (the resident f16 copy of a small
// f32 weight).
extern "C" __global__ void to_half(const float4* __restrict__ x, u64* __restrict__ xh, u64 nquads)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nquads) return;
    const float4 v = x[i];
    xh[i] = (u64)f2h_sat(v.x) | ((u64)f2h_sat(v.y) << 16) | ((u64)f2h_sat(v.z) << 32) | ((u64)f2h_sat(v.w) << 48);
}

// A dense FFN's middle as the f16 operand of its down GEMM, in one pass:
// xh = (act(g) * u) * inv. The product has no bound a half could rely on, so
// it is stored divided by a fixed power of two (the GEMM's alpha multiplies
// it back): that moves the half range up to magnitudes of a million, and
// what falls into the subnormals instead is below a thousandth.
extern "C" __global__ void act_half(const float4* __restrict__ g, const float4* __restrict__ u,
                                    u64* __restrict__ xh, int act, float inv, u64 nquads)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nquads) return;
    const float4 a = g[i], b = u[i];
    xh[i] = (u64)f2h_sat(act_f(a.x, act) * b.x * inv) | ((u64)f2h_sat(act_f(a.y, act) * b.y * inv) << 16) |
            ((u64)f2h_sat(act_f(a.z, act) * b.z * inv) << 32) | ((u64)f2h_sat(act_f(a.w, act) * b.w * inv) << 48);
}

// g = act(g) * u over n values.
extern "C" __global__ void act_mul(float* __restrict__ g, const float* __restrict__ u, int act, u64 n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = act_f(g[i], act) * u[i];
}

// Rotary rows. q [B][nh][hd] is rotated into qg, GROUP-major:
// [kv head][row][head of the group][hd], so one KV group's queries of a whole
// chunk are the contiguous columns of one GEMM (with B = 1 that is [nh][hd]).
// k [B][nkv][hd] is rotated into the mirror rows slot0.., v copied beside it.
// `tab` holds (sin, cos) per row and frequency, computed on the host exactly
// as `attention::rope_table` does.
extern "C" __global__ void rope_kv_rows(const float* __restrict__ q, float* __restrict__ qg,
                                        const float* __restrict__ k, const float* __restrict__ v,
                                        float* __restrict__ Km, float* __restrict__ Vm, const float* __restrict__ tab,
                                        int nh, int nkv, int hd, int half, float scale, int slot0, int B)
{
    const int b = blockIdx.x, T = blockDim.x, tid = threadIdx.x, hpk = nh / nkv;
    const float* t = tab + (size_t)b * half * 2;
    const float* qb = q + (size_t)b * nh * hd;
    for (int it = tid; it < nh * hd; it += T) {
        const int h = it / hd, d = it - h * hd;
        float val;
        if (d < half) {
            const float s = t[2 * d], c = t[2 * d + 1];
            val = __fmul_rn(__fsub_rn(__fmul_rn(qb[h * hd + d], c), __fmul_rn(qb[h * hd + d + half], s)), scale);
        } else if (d < 2 * half) {
            const int i = d - half;
            const float s = t[2 * i], c = t[2 * i + 1];
            val = __fmul_rn(__fadd_rn(__fmul_rn(qb[h * hd + i], s), __fmul_rn(qb[h * hd + d], c)), scale);
        } else {
            val = qb[it];
        }
        const int g = h / hpk, j = h - g * hpk;
        qg[(((size_t)g * B + b) * hpk + j) * hd + d] = val;
    }
    const float* kb = k + (size_t)b * nkv * hd;
    const float* vb = v + (size_t)b * nkv * hd;
    float* km = Km + (size_t)(slot0 + b) * nkv * hd;
    float* vm = Vm + (size_t)(slot0 + b) * nkv * hd;
    for (int it = tid; it < nkv * hd; it += T) {
        const int g = it / hd, d = it - g * hd;
        float val;
        if (d < half) {
            const float s = t[2 * d], c = t[2 * d + 1];
            val = __fmul_rn(__fsub_rn(__fmul_rn(kb[g * hd + d], c), __fmul_rn(kb[g * hd + d + half], s)), scale);
        } else if (d < 2 * half) {
            const int i = d - half;
            const float s = t[2 * i], c = t[2 * i + 1];
            val = __fmul_rn(__fadd_rn(__fmul_rn(kb[g * hd + i], s), __fmul_rn(kb[g * hd + d], c)), scale);
        } else {
            val = kb[it];
        }
        km[it] = val;
        vm[it] = vb[it];
    }
}

// Every `stride`-th row of a chunk's rotated queries (row `first` on), out of
// the group-major layout into [row][head][hd]: what a far-field seal keeps of
// a prompt's queries. A block a kept row.
extern "C" __global__ void q_rows_strided(const float* __restrict__ qg, float* __restrict__ out,
                                          int nh, int nkv, int hd, int B, int first, int stride)
{
    const int r = blockIdx.x, b = first + r * stride, hpk = nh / nkv;
    for (int it = threadIdx.x; it < nh * hd; it += blockDim.x) {
        const int h = it / hd, d = it - h * hd, g = h / hpk, j = h - g * hpk;
        out[((size_t)r * nh + h) * hd + d] = qg[(((size_t)g * B + b) * hpk + j) * hd + d];
    }
}

// Causal (and windowed) softmax of the score rows of a run of queries, left
// unnormalized. S is [kv head][row][L], `cols` rows a KV head: row r is head
// r % hpk of the query at position pos0 + r / hpk, column c the key at
// position col_base + c. Each score becomes exp(s - max) (0 outside the
// row's span) and the row's 1 / sum goes to isum [kv head][...] (`istride`
// floats from one KV head's rows to the next): the value GEMM runs on the
// exponentials and its output row takes the factor, one pass over S less.
extern "C" __global__ void softmax_rows(float* __restrict__ S, float* __restrict__ isum, int L, int pos0,
                                        int col_base, int window, int hpk, int cols, int istride)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    float* s = S + (size_t)blockIdx.x * L;
    const int gk = (int)blockIdx.x / cols, r = (int)blockIdx.x - gk * cols;
    const int p = pos0 + r / hpk;
    int lo = 0;
    if (window > 0 && p + 1 - window > col_base) lo = p + 1 - window - col_base;
    const int hi = p - col_base;
    float m = -INF;
    for (int c = lo + tid; c <= hi; c += T) m = fmaxf(m, s[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INF;
    for (int k = 0; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float sum = 0.0f;
    for (int c = tid; c < L; c += T) {
        float e = 0.0f;
        if (c >= lo && c <= hi) {
            e = expf(s[c] - m);
            sum += e;
        }
        s[c] = e;
    }
    for (int k = 16; k > 0; k >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, k);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int k = 0; k < NW; k++) sum += red[k];
        isum[(size_t)gk * istride + r] = sum > 0.0f ? 1.0f / sum : 1.0f;
    }
}

// The chunk's attention output back in position-major order:
// ao [B][nh][hd] = aog [kv head][B][head of the group][hd] * isum * sigmoid(gate)
// (isum [kv head][B * hpk]: the softmax's 1 / sum; gate [B][nh]: Spark's
// head-wise gate, null where the layer has none). f32, or — `half` set —
// f16, the operand of the output projection's tensor-core GEMM.
extern "C" __global__ void attn_out_rows(const float* __restrict__ aog, const float* __restrict__ isum,
                                         const float* __restrict__ gate, void* __restrict__ ao,
                                         int nh, int nkv, int hd, int B, int half)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * nh * hd) return;
    const int hpk = nh / nkv;
    const int d = (int)(i % hd);
    const size_t bh = i / hd;
    const int h = (int)(bh % nh), b = (int)(bh / nh);
    const int g = h / hpk, j = h - g * hpk;
    float f = isum[(size_t)g * B * hpk + (size_t)b * hpk + j];
    if (gate) f *= 1.0f / (1.0f + expf(-gate[bh]));
    const float v = aog[(((size_t)g * B + b) * hpk + j) * hd + d] * f;
    if (half) ((unsigned short*)ao)[i] = f2h_sat(v);
    else ((float*)ao)[i] = v;
}
