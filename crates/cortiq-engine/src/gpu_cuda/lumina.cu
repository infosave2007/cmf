// Kernels of the Lumina Next-DiT block on CUDA (`gpu_cuda/lumina.rs`): the
// sandwich RMS norms with their timestep scale, the per-head q/k norm with
// the interleaved rotation into head-major planes, the gated post-normed
// residual and the SwiGLU of the feed-forward. The products are cuBLAS GEMMs
// over f16 operands, attention the H3 chain's `h3_flash`.
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

// The sum of squares of row `r` of `x` (width w) across the block.
__device__ float row_sumsq(const float* __restrict__ x, int w) {
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = (T + 31) >> 5;
    float ss = 0.0f;
    for (int i = tid; i < w; i += T) {
        const float v = x[i];
        ss += v * v;
    }
    for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < NW; k++) tot += red[k];
    __syncthreads();
    return tot;
}

// out = f16(rms(x) * w * (1 + s) * oscale), s optional. A block a row.
extern "C" __global__ void lb_norm(const float* __restrict__ x, u16* __restrict__ out, const float* __restrict__ w,
                                   const float* __restrict__ s, int width, float eps, float oscale)
{
    const size_t r = blockIdx.x;
    const float* xr = x + r * width;
    const float inv = rsqrtf(row_sumsq(xr, width) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = xr[i] * inv * w[i];
        if (s) v *= 1.0f + s[i];
        out[r * width + i] = f2h_sat(v * oscale);
    }
}

// x += g * rms(y) * w (g optional). A block a row.
extern "C" __global__ void lb_res(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w,
                                  const float* __restrict__ g, int width, float eps)
{
    const size_t r = blockIdx.x;
    const float* yr = y + r * width;
    const float inv = rsqrtf(row_sumsq(yr, width) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = yr[i] * inv * w[i];
        x[r * width + i] += g ? g[i] * v : v;
    }
}

// Rows of a projection `y` [rows][heads * hd] into head-major f32
// [heads][rows][hd]; with `w` the head's RMS norm (eps) and weight first,
// with `cs` / `sn` ([rows][hd / 2]) the interleaved rotation of pairs
// (2 j, 2 j + 1). A block of 32 a (row, head).
extern "C" __global__ void lb_heads(const float* __restrict__ y, float* __restrict__ out, const float* __restrict__ w,
                                    const float* __restrict__ cs, const float* __restrict__ sn, int rows, int heads,
                                    int hd, float eps)
{
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const float* v = y + ((size_t)r * heads + h) * hd;
    float* o = out + ((size_t)h * rows + r) * hd;
    float inv = 1.0f;
    if (w) {
        float ss = 0.0f;
        for (int i = lane; i < hd; i += 32) ss += v[i] * v[i];
        for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
        inv = rsqrtf(ss / hd + eps);
    }
    for (int j = lane; j < hd / 2; j += 32) {
        float a = v[2 * j] * inv, b = v[2 * j + 1] * inv;
        if (w) {
            a *= w[2 * j];
            b *= w[2 * j + 1];
        }
        if (cs) {
            const float c = cs[(size_t)r * (hd / 2) + j], s = sn[(size_t)r * (hd / 2) + j];
            const float a2 = a * c - b * s, b2 = a * s + b * c;
            a = a2;
            b = b2;
        }
        o[2 * j] = a;
        o[2 * j + 1] = b;
    }
}

// act = f16(silu(g) * u * inv) over `total` values.
extern "C" __global__ void lb_swiglu(const float* __restrict__ g, const float* __restrict__ u, u16* __restrict__ act,
                                     float inv, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gv = g[i];
    act[i] = f2h_sat(gv / (1.0f + expf(-gv)) * u[i] * inv);
}
