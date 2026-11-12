// Kernels of the Z-Image step on CUDA (`gpu_cuda/zimage.rs`) around the
// Lumina block's (`lumina.cu`): the embed's bias and pad rows, the final
// affine-free LayerNorm times the step's scale, and a bias over rows.
typedef unsigned long long u64;

// Rows < n_img: x += bias; rows in [n_img, n_pad): x = pad. A thread a value.
extern "C" __global__ void zi_bias_pad(float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ pad,
                                       int width, int n_img, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = (int)(i % width);
    const u64 r = i / width;
    if (r < (u64)n_img) x[i] += bias[e];
    else x[i] = pad[e];
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

// out = LayerNorm(x) (no affine, eps) * s. A block a row.
extern "C" __global__ void zi_lnscale(const float* __restrict__ x, float* __restrict__ out, const float* __restrict__ s,
                                      int width, float eps)
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
    for (int i = threadIdx.x; i < width; i += blockDim.x) out[r * width + i] = (xr[i] - mean) * inv * s[i];
}

// y[i] += b[i % width].
extern "C" __global__ void zi_bias(float* __restrict__ y, const float* __restrict__ b, int width, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    y[i] += b[i % width];
}
