// Kernels of the MOSS-Image transformer chain on CUDA (`gpu_cuda/moss.rs`):
// the RMS norm with its weight and the step's shift and scale, the per-head
// RMS norm of q and k with the split-half rotation into one head-major
// buffer, the SiTU of the feed-forward and the gated residual. Products are
// cuBLAS GEMMs over f16 operands, attention the H3 chain's `h3_flash`.
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

// out = f16(rms(x) * w * (1 + scale) + shift). A block a row.
extern "C" __global__ void mo_norm(const float* __restrict__ x, u16* __restrict__ out, const float* __restrict__ w,
                                   const float* __restrict__ shift, const float* __restrict__ scale, int width, float eps)
{
    __shared__ float red[32];
    const size_t r = blockIdx.x;
    const float* xr = x + r * width;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = (blockDim.x + 31) >> 5;
    float ss = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) ss += xr[i] * xr[i];
    for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < NW; k++) tot += red[k];
    const float inv = rsqrtf(tot / width + eps);
    for (int i = tid; i < width; i += blockDim.x)
        out[r * width + i] = f2h_sat(xr[i] * inv * w[i] * (1.0f + scale[i]) + shift[i]);
}

// Rows r of a projection `y` [rows][heads * hd] into the head-major f32
// buffer `out` [heads][total][hd] at rows r0 + r; with `w` the head's RMS
// norm (eps) and weight first and the split-half rotation of pairs
// (j, j + hd / 2) by `cs` / `sn` ([rows][hd / 2]). A block of 32 a (row, head).
extern "C" __global__ void mo_heads(const float* __restrict__ y, float* __restrict__ out, const float* __restrict__ w,
                                    const float* __restrict__ cs, const float* __restrict__ sn, int heads, int hd,
                                    int total, int r0, float eps)
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
        const float a = v[j] * inv * w[j], b = v[j + hp] * inv * w[j + hp];
        const float c = cs[(size_t)r * hp + j], s = sn[(size_t)r * hp + j];
        o[j] = a * c - b * s;
        o[j + hp] = b * c + a * s;
    }
}

// act = f16(situ(hidden, gate) * inv) over `total` values of rows `inter`
// wide; `y` holds [rows][hidden (inter) | gate (inter)].
extern "C" __global__ void mo_situ(const float* __restrict__ y, u16* __restrict__ act, int inter, float inv, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / inter, e = i - r * inter;
    const float hv = y[r * 2 * inter + e], g = y[r * 2 * inter + inter + e];
    const float sg = 1.0f / (1.0f + __expf(-fminf(fmaxf(g, -60.0f), 60.0f)));
    act[i] = f2h_sat(4.0f * tanhf(g * 0.25f) * sg * 25.0f * tanhf(hv * 0.04f) * inv);
}

// x += gate * y over `total` values of rows `width` wide.
extern "C" __global__ void mo_addg(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ gate,
                                   int width, u64 total)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] += gate[i % width] * y[i];
}
