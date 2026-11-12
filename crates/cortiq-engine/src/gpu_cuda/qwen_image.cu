// Kernels of the Qwen-Image transformer chain on CUDA (`gpu_cuda/qwen_image.rs`):
// the affine-free LayerNorm with each stream's shift and scale, the per-head
// RMS norm of q and k with the interleaved rotation into one head-major
// buffer for the joint attention ([text | image] rows), the gated residual
// and the tanh-GELU of the MLP. Products are cuBLAS GEMMs over f16 operands,
// attention the H3 chain's `h3_flash`.
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

// The sum of `v` across the block (every thread gets it).
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

// out = f16((layer_norm(x) * (1 + scale) + shift) * oscale), the shift and
// scale `width` apart in `mod`. A block a row.
extern "C" __global__ void qi_lnmod(const float* __restrict__ x, u16* __restrict__ out, const float* __restrict__ shift,
                                    const float* __restrict__ scale, int width, float eps, float oscale)
{
    const size_t r = blockIdx.x;
    const float* xr = x + r * width;
    float s = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s) / width;
    float q = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = (xr[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
        out[r * width + i] = f2h_sat(v * oscale);
    }
}

// Rows r of a projection `y` [rows][heads * hd] plus its bias into the
// head-major f32 buffer `out` [heads][total][hd] at rows r0 + r; with `w` the
// head's RMS norm (eps) and weight first, with `cs` / `sn` ([rows][hd / 2])
// the interleaved rotation of pairs (2 j, 2 j + 1). A block of 32 a
// (row, head).
extern "C" __global__ void qi_heads(const float* __restrict__ y, const float* __restrict__ bias,
                                    float* __restrict__ out, const float* __restrict__ w, const float* __restrict__ cs,
                                    const float* __restrict__ sn, int heads, int hd, int total, int r0, float eps)
{
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const float* v = y + ((size_t)r * heads + h) * hd;
    const float* b = bias ? bias + (size_t)h * hd : 0;
    float* o = out + ((size_t)h * total + r0 + r) * hd;
    float inv = 1.0f;
    if (w) {
        float ss = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            const float t = v[i] + (b ? b[i] : 0.0f);
            ss += t * t;
        }
        for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
        inv = rsqrtf(ss / hd + eps);
    }
    for (int j = lane; j < hd / 2; j += 32) {
        float a = (v[2 * j] + (b ? b[2 * j] : 0.0f)) * inv, c2 = (v[2 * j + 1] + (b ? b[2 * j + 1] : 0.0f)) * inv;
        if (w) {
            a *= w[2 * j];
            c2 *= w[2 * j + 1];
        }
        if (cs) {
            const float c = cs[(size_t)r * (hd / 2) + j], s = sn[(size_t)r * (hd / 2) + j];
            const float a2 = a * c - c2 * s, b2 = a * s + c2 * c;
            a = a2;
            c2 = b2;
        }
        o[2 * j] = a;
        o[2 * j + 1] = c2;
    }
}

// x += gate * (y + bias) over `total` values of rows `width` wide.
extern "C" __global__ void qi_addg(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ bias,
                                   const float* __restrict__ gate, int width, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = (int)(i % width);
    x[i] += gate[e] * (y[i] + (bias ? bias[e] : 0.0f));
}

// act = f16(gelu_tanh(y + bias) * inv) over `total` values of rows `width` wide.
extern "C" __global__ void qi_gelu(const float* __restrict__ y, const float* __restrict__ bias, u16* __restrict__ act,
                                   int width, float inv, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] + (bias ? bias[i % width] : 0.0f);
    const float t = tanhf(fminf(fmaxf(0.7978845608f * (v + 0.044715f * v * v * v), -15.0f), 15.0f));
    act[i] = f2h_sat(0.5f * v * (1.0f + t) * inv);
}
