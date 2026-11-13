// Kernels of the Primal VAE on CUDA (`gpu_cuda/primal.rs`): the 3×3
// convolution's im2col with a dilation (= its padding), the RMS norm over a
// pixel's channels with SiLU, and pixel (un)shuffle by 2. Products are f16
// cuBLAS GEMMs.
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

// cols [np][c·9] (f16) of output positions p0.. of a 3×3 conv with dilation
// and padding `dil` over x [c][h][w]. A thread a (position, channel).
extern "C" __global__ void pv_im2col(const float* __restrict__ x, u16* __restrict__ cols, int c, int h, int w, int dil,
                                     u64 p0, u64 total)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const u64 i = idx / c, p = p0 + i;
    const int oy = (int)(p / w), ox = (int)(p % w);
    u16* dst = cols + i * (u64)c * 9 + (u64)ch * 9;
    const float* xc = x + (u64)ch * h * w;
    for (int t = 0; t < 9; t++) {
        const int sy = oy + (t / 3) * dil - dil, sx = ox + (t % 3) * dil - dil;
        float v = 0.0f;
        if (sy >= 0 && sx >= 0 && sy < h && sx < w) v = xc[(u64)sy * w + sx];
        dst[t] = f2h_sat(v);
    }
}

// dst = silu(x · rsqrt(mean_c(x²) + eps) · g) per pixel of [c][hw]. A thread
// a pixel.
extern "C" __global__ void pv_norm_silu(const float* __restrict__ x, float* __restrict__ dst, const float* __restrict__ g,
                                        int c, u64 hw, float eps)
{
    const u64 p = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hw) return;
    float ss = 0.0f;
    for (int ch = 0; ch < c; ch++) {
        const float v = x[(u64)ch * hw + p];
        ss += v * v;
    }
    const float inv = rsqrtf(ss / c + eps);
    for (int ch = 0; ch < c; ch++) {
        const float v = x[(u64)ch * hw + p] * inv * g[ch];
        dst[(u64)ch * hw + p] = v / (1.0f + __expf(-v));
    }
}

// dst [4c][h/2][w/2] = pixel_unshuffle(src [c][h][w], 2). A thread a value.
extern "C" __global__ void pv_unshuffle(const float* __restrict__ src, float* __restrict__ dst, int c, int h, int w, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int oh = h / 2, ow = w / 2;
    const int x = (int)(i % ow);
    const u64 r = i / ow;
    const int y = (int)(r % oh);
    const int oc = (int)(r / oh);
    const int ch = oc / 4, ij = oc % 4, ii = ij / 2, jj = ij % 2;
    dst[i] = src[((u64)ch * h + 2 * y + ii) * w + 2 * x + jj];
}

// dst [c/4][2h][2w] = pixel_shuffle(src [c][h][w], 2) (+ add). A thread a value.
extern "C" __global__ void pv_shuffle(const float* __restrict__ src, float* __restrict__ dst, const float* __restrict__ add,
                                      int c, int h, int w, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int oh = 2 * h, ow = 2 * w;
    const int ox = (int)(i % ow);
    const u64 r = i / ow;
    const int oy = (int)(r % oh);
    const int ch = (int)(r / oh);
    const int ic = ch * 4 + (oy & 1) * 2 + (ox & 1);
    float v = src[((u64)ic * h + (oy >> 1)) * w + (ox >> 1)];
    if (add) v += add[i];
    dst[i] = v;
}
