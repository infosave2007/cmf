// Kernels of the FLUX-class VAE decoder chain on CUDA (`gpu_cuda/flux_vae.rs`):
// GroupNorm in three passes (partial sums of each group's span over many
// blocks, the group's mean and inverse deviation, then the affine norm with an
// optional SiLU) and a bias over token-major rows. The convolutions are the
// LTX chain's im2col with cuBLAS products, attention TF32 products around its
// softmax.
typedef unsigned long long u64;

// Sum and sum of squares of chunk `blockIdx.x` of group `blockIdx.y`'s span
// (`span` values from `g · span`) into `part[(g · nchunk + b) · 2 ..]`.
extern "C" __global__ void fv_gn_part(const float* __restrict__ x, double* __restrict__ part, u64 span, u64 chunk, int nchunk)
{
    __shared__ double rs[32], rq[32];
    const int g = blockIdx.y, b = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = (blockDim.x + 31) >> 5;
    const float* xs = x + (u64)g * span;
    const u64 lo = (u64)b * chunk, hi = lo + chunk < span ? lo + chunk : span;
    float s = 0.0f, q = 0.0f;
    for (u64 i = lo + tid; i < hi; i += blockDim.x) {
        const float v = xs[i];
        s += v;
        q += v * v;
    }
    double ds = s, dq = q;
    for (int k = 16; k > 0; k >>= 1) {
        ds += __shfl_xor_sync(0xffffffffu, ds, k);
        dq += __shfl_xor_sync(0xffffffffu, dq, k);
    }
    if (lane == 0) {
        rs[warp] = ds;
        rq[warp] = dq;
    }
    __syncthreads();
    if (tid == 0) {
        double a = 0.0, c = 0.0;
        for (int k = 0; k < NW; k++) {
            a += rs[k];
            c += rq[k];
        }
        part[((u64)g * nchunk + b) * 2] = a;
        part[((u64)g * nchunk + b) * 2 + 1] = c;
    }
}

// stats[g] = (mean, 1 / sqrt(var + eps)) of group g from its partial sums.
// A warp a group.
extern "C" __global__ void fv_gn_fin(const double* __restrict__ part, float* __restrict__ stats, int nchunk, double count, float eps)
{
    const int g = blockIdx.x, lane = threadIdx.x;
    double s = 0.0, q = 0.0;
    for (int b = lane; b < nchunk; b += 32) {
        s += part[((u64)g * nchunk + b) * 2];
        q += part[((u64)g * nchunk + b) * 2 + 1];
    }
    for (int k = 16; k > 0; k >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, k);
        q += __shfl_xor_sync(0xffffffffu, q, k);
    }
    if (lane == 0) {
        const double mean = s / count;
        double var = q / count - mean * mean;
        if (var < 0.0) var = 0.0;
        stats[2 * g] = (float)mean;
        stats[2 * g + 1] = (float)(1.0 / sqrt(var + (double)eps));
    }
}

// out = (x - mean) · inv · w[c] + b[c] over NCHW [c][hw], SiLU under `silu`.
// `out` may be `x`.
extern "C" __global__ void fv_gn_apply(const float* x, float* out, const float* __restrict__ stats,
                                       const float* __restrict__ w, const float* __restrict__ b, u64 hw, int per,
                                       int silu, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i / hw), g = c / per;
    float v = (x[i] - stats[2 * g]) * stats[2 * g + 1] * w[c] + b[c];
    if (silu) v = v / (1.0f + __expf(-v));
    out[i] = v;
}

// y[i] += b[i % c] over token-major rows of `c`.
extern "C" __global__ void fv_bias_rows(float* __restrict__ y, const float* __restrict__ b, int c, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    y[i] += b[i % c];
}
