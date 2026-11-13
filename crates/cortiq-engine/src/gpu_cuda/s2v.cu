// Kernels of the Kandinsky S2V denoiser on CUDA (`gpu_cuda/s2v.rs`): the
// LayerNorm with each row's modulation (memory and reference rows at t = 0,
// the noisy tail at t) or an affine, a projection's bias with the per-head RMS
// norm and rotation, the gated residual, the erf GELU, the speech attention
// (a frame's rows over its own audio tokens) and a bias over rows.
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

// xn = f16((LN(x) · (1 + scale) + shift) · oscale) for rows row0.. of x: a
// row below `nclean` takes (shc, scc), the rest (shn, scn); with `aff` the
// affine LN(x) · scn + shn instead. A block a row.
extern "C" __global__ void s2v_ln(const float* __restrict__ x, u16* __restrict__ xn, const float* __restrict__ shc,
                                  const float* __restrict__ scc, const float* __restrict__ shn, const float* __restrict__ scn,
                                  int nclean, int row0, int width, float eps, float oscale, int aff)
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
    const bool clean = (int)r + row0 < nclean;
    const float* sh = clean ? shc : shn;
    const float* sc = clean ? scc : scn;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float n = (xr[i] - mean) * inv;
        const float v = aff ? n * sc[i] + sh[i] : n * (1.0f + sc[i]) + sh[i];
        xn[r * width + i] = f2h_sat(v * oscale);
    }
}

// Rows of a projection chunk y [rows][heads·hd] plus the bias, with `w` the
// head's RMS norm (eps) and weight, with `cs` the interleaved rotation of
// pairs (2 j, 2 j + 1) by cs / sn ([rows][hd/2]); into `out` at rows r0 + r,
// head-major [heads][total][hd] under `hmaj`, else token-major. A block of 32
// a (row, head).
extern "C" __global__ void s2v_heads(const float* __restrict__ y, const float* __restrict__ bias, float* __restrict__ out,
                                     const float* __restrict__ w, const float* __restrict__ cs, const float* __restrict__ sn,
                                     int total, int r0, int heads, int hd, float eps, int hmaj)
{
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, hp = hd / 2;
    const float* v = y + ((size_t)r * heads + h) * hd;
    const float* b = bias ? bias + (size_t)h * hd : 0;
    float* o = hmaj ? out + ((size_t)h * total + r0 + r) * hd : out + ((size_t)(r0 + r) * heads + h) * hd;
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
    for (int j = lane; j < hp; j += 32) {
        float a = (v[2 * j] + (b ? b[2 * j] : 0.0f)) * inv, c2 = (v[2 * j + 1] + (b ? b[2 * j + 1] : 0.0f)) * inv;
        if (w) {
            a *= w[2 * j];
            c2 *= w[2 * j + 1];
        }
        if (cs) {
            const float c = cs[(size_t)r * hp + j], s = sn[(size_t)r * hp + j];
            const float a2 = a * c - c2 * s, b2 = a * s + c2 * c;
            a = a2;
            c2 = b2;
        }
        o[2 * j] = a;
        o[2 * j + 1] = c2;
    }
}

// x[i] += g · (y[i] + bias[e]) over rows row0.. of width `width`: g from gc
// (rows below nclean) or gn, 1 without gates. A thread a value.
extern "C" __global__ void s2v_addg(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ bias,
                                    const float* __restrict__ gc, const float* __restrict__ gn, int nclean, int row0,
                                    int width, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = (int)(i % width);
    const int r = (int)(i / width) + row0;
    float v = y[i] + (bias ? bias[e] : 0.0f);
    if (gc) v *= (r < nclean ? gc : gn)[e];
    x[i] += v;
}

// act = f16(gelu_erf(y + bias) · inv) over `total` values of rows `width` wide.
extern "C" __global__ void s2v_gelu(const float* __restrict__ y, const float* __restrict__ bias, u16* __restrict__ act,
                                    float inv, int width, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] + bias[i % width];
    act[i] = f2h_sat(0.5f * v * (1.0f + erff(v * 0.70710678118f)) * inv);
}

// Each query row's head over its frame's audio keys: frame f = row / per,
// keys [grp[2f], grp[2f] + grp[2f+1]) of k, v (token-major [na][heads·hd]);
// q token-major [rows][heads·hd]; out f16 token-major. A warp a (row, head),
// at most 64 keys a frame.
extern "C" __global__ void s2v_gattn(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                     u16* __restrict__ out, const u32* __restrict__ grp, int per, int heads, int hd,
                                     float scale)
{
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int f = r / per;
    const int k0 = (int)grp[2 * f], kn = min((int)grp[2 * f + 1], 64);
    const size_t ld = (size_t)heads * hd;
    const float* qr = q + (size_t)r * ld + (size_t)h * hd;
    float s[64];
    float mx = -3.0e38f;
    for (int j = 0; j < kn; j++) {
        const float* kr = k + (size_t)(k0 + j) * ld + (size_t)h * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += 32) d += qr[i] * kr[i];
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        s[j] = d * scale;
        mx = fmaxf(mx, s[j]);
    }
    float z = 0.0f;
    for (int j = 0; j < kn; j++) {
        s[j] = expf(s[j] - mx);
        z += s[j];
    }
    const float iz = z > 0.0f ? 1.0f / z : 0.0f;
    for (int i = lane; i < hd; i += 32) {
        float acc = 0.0f;
        for (int j = 0; j < kn; j++) acc += s[j] * v[(size_t)(k0 + j) * ld + (size_t)h * hd + i];
        out[(size_t)r * ld + (size_t)h * hd + i] = f2h_sat(acc * iz);
    }
}

// x[i] += b1[e] (+ b2[e]) over rows of `width`.
extern "C" __global__ void s2v_addvec(float* __restrict__ x, const float* __restrict__ b1, const float* __restrict__ b2,
                                      int width, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = (int)(i % width);
    x[i] += b1[e] + (b2 ? b2[e] : 0.0f);
}
