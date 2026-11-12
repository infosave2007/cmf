// Kernels of the Qwen-Image-2.1 denoiser on CUDA (`gpu_cuda/qi21.rs`): the
// affine-free LayerNorm times a per-channel factor, per-head RMS norm of q
// and k with the interleaved rotation into head-major planes, those planes
// into the flash kernel's layouts at a row offset of a longer sequence (the
// prefix cache, then a step's own rows), the gated residual, the SwiGLU and
// the prefill's softmax over each row's visible keys. Products are cuBLAS
// GEMMs, attention the H3 chain's `h3_flash` (steps) or TF32 products (the
// block-causal prefill).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ u16 f2h_sat(float f) {
    return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

__device__ float block_sum(float v) {
    __shared__ float red[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = (blockDim.x + 31) >> 5;
    for (int k = 16; k > 0; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < NW; k++) tot += red[k];
    __syncthreads();
    return tot;
}

// LN(x) times (1 + s[i]) under `plus1`, else times s[i], into f16 (`oh`,
// times `oscale`) or f32 (`of`). A block a row.
extern "C" __global__ void q21_lnmod(const float* __restrict__ x, u16* __restrict__ oh, float* __restrict__ of,
                                     const float* __restrict__ s, int plus1, int width, float eps, float oscale)
{
    const size_t r = blockIdx.x;
    const float* xr = x + r * width;
    float a = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) a += xr[i];
    const float mean = block_sum(a) / width;
    float q = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = (xr[i] - mean) * inv * (plus1 ? 1.0f + s[i] : s[i]);
        if (oh) oh[r * width + i] = f2h_sat(v * oscale);
        else of[r * width + i] = v;
    }
}

// Rows r of a projection chunk `y` [rows][heads * hd] into head-major f32
// [heads][total][hd] at rows r0 + r; with `w` the head's RMS norm (eps) and
// weight first and the interleaved rotation of pairs (2 j, 2 j + 1) by `cs` /
// `sn` (the chunk's [rows][hd / 2]). A block of 32 a (row, head).
extern "C" __global__ void q21_heads(const float* __restrict__ y, float* __restrict__ out, const float* __restrict__ w,
                                     const float* __restrict__ cs, const float* __restrict__ sn, int total, int r0,
                                     int heads, int hd, float eps)
{
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, hp = hd / 2;
    const float* v = y + ((size_t)r * heads + h) * hd;
    float* o = out + ((size_t)h * total + r0 + r) * hd;
    if (!w) {
        for (int i = lane; i < hd; i += 32) o[i] = v[i];
        return;
    }
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) ss += v[i] * v[i];
    for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
    const float inv = rsqrtf(ss / hd + eps);
    for (int j = lane; j < hp; j += 32) {
        const float a = v[2 * j] * inv * w[2 * j], b = v[2 * j + 1] * inv * w[2 * j + 1];
        const float c = cs[(size_t)r * hp + j], s = sn[(size_t)r * hp + j];
        o[2 * j] = a * c - b * s;
        o[2 * j + 1] = a * s + b * c;
    }
}

// Head-major f32 planes [planes][rows][hd] into the flash kernel's layouts of
// a sequence of `n` rows (`ns` padded) at rows r0.. : mode 0 the queries, 1
// the keys in lane order, 2 the values in lane order (`h3_pack`'s layouts,
// with a row offset). A thread a source element.
extern "C" __global__ void q21_pack(const float* __restrict__ src, u16* __restrict__ dst, int rows, int r0, int n, int ns,
                                    int planes, int hd, int mode)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)planes * rows * hd) return;
    const int d = (int)(idx % hd);
    const size_t t = idx / hd;
    const int rr = (int)(t % rows);
    const size_t plane = t / rows;
    const int row = r0 + rr;
    const u16 v = f2h_sat(src[idx]);
    if (mode == 2) {
        const int j = d >> 3, gid = d & 7;
        dst[plane * ns * hd + ((size_t)(row >> 4) * 32 + gid * 4 + ((row & 7) >> 1)) * (hd >> 1) + j * 4 +
            ((row & 15) >> 3) * 2 + (row & 1)] = v;
    } else if (mode == 0) {
        dst[(plane * n + row) * hd + d] = v;
    } else {
        const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
        dst[(plane * n + row) * hd + (size_t)tig * (hd >> 2) + 2 * m + e] = v;
    }
}

// x += gate * y over `total` values of rows `width` wide.
extern "C" __global__ void q21_addg(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ gate,
                                    int width, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] += gate[i % width] * y[i];
}

// act = f16(silu(g) * u * inv) over `total` values.
extern "C" __global__ void q21_swiglu(const float* __restrict__ g, const float* __restrict__ u, u16* __restrict__ act,
                                      float inv, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gv = g[i];
    act[i] = f2h_sat(gv / (1.0f + __expf(-fminf(fmaxf(gv, -80.0f), 80.0f))) * u[i] * inv);
}

// Rows q0.. of scores S [rows][n] into softmax weights over the keys each
// row sees (`vis[q0 + row]`, a leading run); the rest become zero. A block a
// row.
extern "C" __global__ void q21_softmax_vis(float* __restrict__ S, int n, const u32* __restrict__ vis, int q0)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    float* s = S + (size_t)blockIdx.x * n;
    const int lim = min((int)vis[q0 + blockIdx.x], n);
    float m = -3.0e38f;
    for (int i = tid; i < lim; i += T) m = fmaxf(m, s[i]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float e = i < lim ? expf(s[i] - m) : 0.0f;
        s[i] = e;
        z += e;
    }
    for (int k = 16; k > 0; k >>= 1) z += __shfl_xor_sync(0xffffffffu, z, k);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < NW; k++) tot += red[k];
    const float inv = tot > 0.0f ? 1.0f / tot : 0.0f;
    for (int i = tid; i < n; i += T) s[i] *= inv;
}
