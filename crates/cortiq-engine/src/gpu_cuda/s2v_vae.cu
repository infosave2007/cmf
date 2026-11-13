// Kernels of the HunyuanVideo VAE decoder on CUDA (`gpu_cuda/s2v_vae.rs`),
// the stream f16 channel-major [C][F·H·W]: the 3-D causal conv's im2col with
// replicate padding (kt−1 frames in front, k/2 in space), the store of a
// product chunk with bias and residual, GroupNorm (32 groups) in three passes
// with SiLU, the nearest upsample that keeps the first frame, and f16 ↔ f32.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ u16 f2h_sat(float f) { return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f)); }
__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}

// cols [c·kt·kh·kw][np] (f16, K-major: consecutive threads read and write
// consecutive positions) of output positions p0.. (an output Ho × Wo a frame)
// of a conv of strides (st, sh, sw) over x [c][F][H][W] (f16),
// replicate-clamped. A thread a (channel, position).
extern "C" __global__ void hv_im2col(const u16* __restrict__ x, u16* __restrict__ cols, int c, int F, int H, int W, int kt,
                                     int kh, int kw, int Ho, int Wo, int st, int sh, int sw, u64 p0, u64 np)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (u64)c * np) return;
    const u64 i = idx % np;
    const int ch = (int)(idx / np);
    const u64 p = p0 + i;
    const int ow = (int)(p % Wo) * sw;
    const u64 r = p / Wo;
    const int oh = (int)(r % Ho) * sh, of = (int)(r / Ho) * st;
    const int pt = kt - 1, ph = kh / 2, pw = kw / 2;
    const int taps = kt * kh * kw;
    u16* dst = cols + (u64)ch * taps * np + i;
    const u16* xc = x + (u64)ch * F * H * W;
    int o = 0;
    for (int dt = 0; dt < kt; dt++) {
        const int f = min(max(of + dt - pt, 0), F - 1);
        for (int dh = 0; dh < kh; dh++) {
            const int hh = min(max(oh + dh - ph, 0), H - 1);
            for (int dw = 0; dw < kw; dw++) {
                const int ww = min(max(ow + dw - pw, 0), W - 1);
                dst[(u64)(o++) * np] = xc[((u64)f * H + hh) * W + ww];
            }
        }
    }
}

// dst[co][p0 + i] = f16(y[co][i] + bias[co] (+ res[co][p0 + i])) for a product
// chunk y [cout][np]; `res` may be `dst`. A thread a value.
extern "C" __global__ void hv_store(const float* __restrict__ y, u16* dst, const float* __restrict__ bias, const u16* res,
                                    int cout, u64 np, u64 P, u64 p0)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (u64)cout * np) return;
    const int co = (int)(idx / np);
    const u64 i = idx % np;
    float v = y[(u64)co * np + i] + bias[co];
    const u64 d = (u64)co * P + p0 + i;
    if (res) v += h2f(res[d]);
    dst[d] = f2h_sat(v);
}

// GroupNorm pass 1/2: partial sums (and sums of squares) of chunk b of group
// g's span (`span` values from g·span) of an f16 stream into part[(g·nchunk+b)·2].
extern "C" __global__ void hv_gn_part(const u16* __restrict__ x, double* __restrict__ part, u64 span, u64 chunk, int nchunk)
{
    __shared__ double rs[32], rq[32];
    const int g = blockIdx.y, b = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = (blockDim.x + 31) >> 5;
    const u16* xs = x + (u64)g * span;
    const u64 lo = (u64)b * chunk, hi = lo + chunk < span ? lo + chunk : span;
    float s = 0.0f, q = 0.0f;
    for (u64 i = lo + tid; i < hi; i += blockDim.x) {
        const float v = h2f(xs[i]);
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

extern "C" __global__ void hv_gn_fin(const double* __restrict__ part, float* __restrict__ stats, int nchunk, double count, float eps)
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

// out = (x − mean)·inv·w[c] + b[c] (then SiLU) over f16 [c][P]; out may be x.
extern "C" __global__ void hv_gn_apply(const u16* x, u16* out, const float* __restrict__ stats, const float* __restrict__ w,
                                       const float* __restrict__ b, u64 P, int per, int silu, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i / P), g = c / per;
    float v = (h2f(x[i]) - stats[2 * g]) * stats[2 * g + 1] * w[c] + b[c];
    if (silu) v = v / (1.0f + __expf(-v));
    out[i] = f2h_sat(v);
}

// Nearest upsample by (ft, fs, fs) of f16 [c][F][H][W]; with ft = 2 the first
// frame stays one frame (F' = 1 + 2(F−1)). A thread an output value.
extern "C" __global__ void hv_up(const u16* __restrict__ x, u16* __restrict__ out, int c, int F, int H, int W, int ft, int fs,
                                 u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int Fo = ft == 2 ? 1 + (F - 1) * 2 : F, Ho = H * fs, Wo = W * fs;
    const int ow = (int)(i % Wo);
    u64 r = i / Wo;
    const int oh = (int)(r % Ho);
    r /= Ho;
    const int of = (int)(r % Fo), ch = (int)(r / Fo);
    const int sf = (ft == 2 && of > 0) ? 1 + (of - 1) / 2 : of;
    out[i] = x[(((u64)ch * F + sf) * H + oh / fs) * W + ow / fs];
}

extern "C" __global__ void hv_to_f32(const u16* __restrict__ x, float* __restrict__ y, u64 n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = h2f(x[i]);
}

extern "C" __global__ void hv_to_f16(const float* __restrict__ x, u16* __restrict__ y, u64 n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h_sat(x[i]);
}

// Rows of scores S [rows][n] into softmax weights over their first `live`
// columns (the rest zero). A block a row.
extern "C" __global__ void hv_softmax(float* __restrict__ S, int n, int live)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    float* s = S + (size_t)blockIdx.x * n;
    float m = -3.0e38f;
    for (int i = tid; i < live; i += T) m = fmaxf(m, s[i]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float e = i < live ? expf(s[i] - m) : 0.0f;
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
