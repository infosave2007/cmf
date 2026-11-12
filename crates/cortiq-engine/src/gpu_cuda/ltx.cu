// Kernels of the LTX-2.5 resident chain (`gpu_cuda/ltx.rs`): everything
// between the weight products of a block — the timestep-modulated norms, the
// q/k norm with the split rotation into head-major planes, the softmax of a
// chunk of score rows, the per-head gate, the gated residuals and the GELU
// of the feed-forward. The products themselves are cuBLAS GEMMs over f16
// operands with f32 accumulation.
//
// Layouts:
//   x    f32 [n][dim]             a residual stream
//   xn   f16 [rows][dim]          normed, modulated rows: a GEMM's operand
//   y    f32 [rows][out]          a GEMM's result
//   Q K  f16 [heads][n][dh]       head-major planes (one GEMM operand a head)
//   VT   f16 [heads][dh][m]       the values dim-major
//   S    f32 [heads][c][m]        a chunk of score rows, P f16 the same shape
//   tab  f32 [R][stride]          modulation rows; a token reads row rowidx[t]
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

// Values past the half range saturate instead of becoming infinite.
__device__ __forceinline__ u16 f2h_sat(float f) {
    return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

// The sum of squares of a row across the block's threads (a block a row).
#define ROW_SUMSQ(expr)                                                       \
    __shared__ float red[32];                                                  \
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31;              \
    const int warp = tid >> 5, NW = T >> 5;                                    \
    float ss = 0.0f;                                                           \
    for (int i = tid; i < width; i += T) {                                     \
        const float v = (expr);                                                \
        ss += v * v;                                                           \
    }                                                                          \
    for (int k = 16; k > 0; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k); \
    if (lane == 0) red[warp] = ss;                                             \
    __syncthreads();                                                           \
    float tot = 0.0f;                                                          \
    for (int k = 0; k < NW; k++) tot += red[k];

// out = (rms ? x / rms(x) : x) * (1 + scale) + shift, as f16. `tab` set: the
// shift and scale are chunks `sh` and `sc` of the row's table row (row 0
// without `rowidx`); unset: the norm alone. A block a row.
extern "C" __global__ void lx_ada(const float* __restrict__ x, u16* __restrict__ out,
                                  const float* __restrict__ tab, const u32* __restrict__ rowidx,
                                  int width, int stride, int sh, int sc, int rms, float eps)
{
    const size_t r = blockIdx.x;
    const float* xr = x + r * width;
    ROW_SUMSQ(rms ? xr[i] : 0.0f)
    const float inv = rms ? rsqrtf(tot / width + eps) : 1.0f;
    const float* t = tab ? tab + (size_t)(rowidx ? rowidx[r] : 0) * stride : 0;
    u16* o = out + r * width;
    for (int i = tid; i < width; i += T) {
        float v = xr[i] * inv;
        if (t) v = v * (1.0f + t[sc + i]) + t[sh + i];
        o[i] = f2h_sat(v);
    }
}

// A chunk of a query or key panel: (y + bias), the RMS norm over the whole
// row with its weight, the split rotation, and out as head-major f16 planes
// P [heads][n][dh] at rows r0... The rotation tables are [n][width / 2]; a
// rotation group is 2 * half wide and pairs element i with i + half. A block
// a row.
extern "C" __global__ void lx_qk(const float* __restrict__ y, const float* __restrict__ bias,
                                 const float* __restrict__ nw, const float* __restrict__ cs,
                                 const float* __restrict__ sn, u16* __restrict__ P, int width, int half,
                                 int dh, int r0, int n, float eps)
{
    const size_t r = blockIdx.x;
    const float* yr = y + r * width;
    ROW_SUMSQ(yr[i] + (bias ? bias[i] : 0.0f))
    const float inv = rsqrtf(tot / width + eps);
    const size_t row = (size_t)r0 + r;
    const float* c = cs ? cs + row * (width / 2) : 0;
    const float* s = sn ? sn + row * (width / 2) : 0;
    #define NV(j) ((yr[j] + (bias ? bias[j] : 0.0f)) * inv * nw[j])
    for (int e = tid; e < width; e += T) {
        float v;
        if (c) {
            const int g = e / (2 * half), i = e - g * 2 * half;
            const int first = i < half;
            const int lo = first ? e : e - half;
            const int ci = g * half + (first ? i : i - half);
            const float a = NV(lo), b = NV(lo + half);
            v = first ? a * c[ci] - b * s[ci] : b * c[ci] + a * s[ci];
        } else {
            v = NV(e);
        }
        const int h = e / dh, d = e - h * dh;
        P[((size_t)h * n + row) * dh + d] = f2h_sat(v);
    }
    #undef NV
}

// A chunk of a value panel, (y + bias), into the dim-major f16 planes
// VT [heads][dh][m] at keys r0...
extern "C" __global__ void lx_v(const float* __restrict__ y, const float* __restrict__ bias,
                                u16* __restrict__ VT, int width, int dh, int r0, int m, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / width, e = idx - r * width;
    const float v = y[idx] + (bias ? bias[e] : 0.0f);
    // e = h * dh + d, and the plane index is the same number
    VT[e * (size_t)m + r0 + r] = f2h_sat(v);
}

// Score rows into attention weights (a block a row): P = exp(S + mask - max)
// * pscale as f16, and the row's sum of exp into Z. The division by the sum
// is left to `lx_join`, after the value GEMM (whose alpha takes pscale back
// out). `mask` is an additive bias a key.
extern "C" __global__ void lx_softmax(const float* __restrict__ S, u16* __restrict__ P, float* __restrict__ Z,
                                      const float* __restrict__ mask, int n, float pscale)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* s = S + (size_t)blockIdx.x * n;
    u16* p = P + (size_t)blockIdx.x * n;
    float m = -3.0e38f;
    for (int i = tid; i < n; i += T) m = fmaxf(m, s[i] + (mask ? mask[i] : 0.0f));
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float e = expf(s[i] + (mask ? mask[i] : 0.0f) - m);
        z += e;
        p[i] = f2h(e * pscale);
    }
    for (int k = 16; k > 0; k >>= 1) z += __shfl_xor_sync(0xffffffffu, z, k);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int k = 0; k < NW; k++) tot += red[k];
        Z[blockIdx.x] = tot;
    }
}

// A chunk's attention output, head-major f32 O [heads][c][dh] still carrying
// each score row's sum Z [heads][c], as the f16 rows AO [c][heads * dh] of
// the output projection's operand; `gl` set: every head scaled by
// 2 * sigmoid(gl[row][head] + gb[head]).
extern "C" __global__ void lx_join(const float* __restrict__ O, const float* __restrict__ Z,
                                   const float* __restrict__ gl, const float* __restrict__ gb,
                                   u16* __restrict__ AO, int c, int heads, int dh, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t inner = (size_t)heads * dh;
    const size_t r = idx / inner, e = idx - r * inner;
    const size_t h = e / dh, d = e - h * dh;
    const size_t pr = h * c + r;
    float v = O[pr * dh + d] / Z[pr];
    if (gl) {
        const float l = gl[r * heads + h] + (gb ? gb[h] : 0.0f);
        v *= 2.0f / (1.0f + expf(-l));
    }
    AO[idx] = f2h_sat(v);
}

// x += g * (y + bias) over a chunk of rows: g is chunk `go` of the row's
// table row (row 0 without `rowidx`), or one without a table.
extern "C" __global__ void lx_addg(float* __restrict__ x, const float* __restrict__ y,
                                   const float* __restrict__ bias, const float* __restrict__ tab,
                                   const u32* __restrict__ rowidx, int width, int stride, int go, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / width, i = idx - r * width;
    const float v = y[idx] + (bias ? bias[i] : 0.0f);
    if (tab) x[idx] += tab[(size_t)(rowidx ? rowidx[r] : 0) * stride + go + i] * v;
    else x[idx] += v;
}

// gelu(y + bias, approximate = "tanh") * inv as f16 (the feed-forward's
// middle, stored divided by a power of two the second GEMM's alpha takes
// back).
extern "C" __global__ void lx_gelu(const float* __restrict__ y, const float* __restrict__ bias,
                                   u16* __restrict__ a, int width, float inv, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float v = y[idx] + (bias ? bias[idx % width] : 0.0f);
    const float g = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
    a[idx] = f2h_sat(g * inv);
}

// dst = src over n floats (a stream's state before the fusion).
extern "C" __global__ void lx_copy(const float* __restrict__ src, float* __restrict__ dst, u64 n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ── the video VAE: volumes [c][f][h][w], channel-major ──

// The RMS over the channels at every position of x [c][p], then SiLU, into
// out [c][p]. A thread a position.
extern "C" __global__ void vx_norm_silu(const float* __restrict__ x, float* __restrict__ out, int c, u64 p, float eps)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    float s = 0.0f;
    for (int ch = 0; ch < c; ch++) {
        const float v = x[(size_t)ch * p + i];
        s += v * v;
    }
    const float inv = rsqrtf(s / c + eps);
    for (int ch = 0; ch < c; ch++) {
        const float v = x[(size_t)ch * p + i] * inv;
        out[(size_t)ch * p + i] = v / (1.0f + expf(-v));
    }
}

// im2col of positions p0.. of a 3x3x3 convolution over x [c][f][h][w]: cols
// f16 [positions][c * 27] in the weight's (channel, kf, kh, kw) order, zero
// padding in space and replicated frames in time. A thread a (position,
// channel).
extern "C" __global__ void vx_im2col3d(const float* __restrict__ x, u16* __restrict__ cols, int c, int f, int h,
                                       int w, u64 p0, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const size_t i = idx / c, p = p0 + i;
    const int pw = (int)(p % w);
    const size_t rest = p / w;
    const int ph = (int)(rest % h), pf = (int)(rest / h);
    u16* dst = cols + i * (size_t)c * 27 + (size_t)ch * 27;
    const float* xc = x + (size_t)ch * f * h * w;
    for (int kf = 0; kf < 3; kf++) {
        int sf = pf + kf - 1;
        sf = sf < 0 ? 0 : (sf >= f ? f - 1 : sf);
        for (int kh = 0; kh < 3; kh++) {
            const int sh = ph + kh - 1;
            for (int kw = 0; kw < 3; kw++) {
                const int sw = pw + kw - 1;
                const float v = (sh < 0 || sh >= h || sw < 0 || sw >= w) ? 0.0f : xc[((size_t)sf * h + sh) * w + sw];
                dst[kf * 9 + kh * 3 + kw] = f2h_sat(v);
            }
        }
    }
}

// y [c][p] += bias [c] (when given) + add [c][p] (when given).
extern "C" __global__ void vx_bias(float* __restrict__ y, const float* __restrict__ bias,
                                   const float* __restrict__ add, u64 p, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = y[idx];
    if (bias) v += bias[idx / p];
    if (add) v += add[idx];
    y[idx] = v;
}

// Depth to space of y [c * p1 * p2 * p3][f][h][w] into out
// [c][f * p1 - drop][h * p2][w * p3]; the channels are laid out
// (c p1 p2 p3), and `drop` leading frames are left out. A thread a value.
extern "C" __global__ void vx_d2s(const float* __restrict__ y, float* __restrict__ out, int f, int h, int w, int p1,
                                  int p2, int p3, int drop, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t w2 = (size_t)w * p3, h2 = (size_t)h * p2, f3 = (size_t)f * p1 - drop;
    const size_t ow = idx % w2;
    size_t r = idx / w2;
    const size_t oh = r % h2;
    r /= h2;
    const size_t of = r % f3 + drop, cc = r / f3;
    const size_t fr = of / p1, a = of % p1, hh = oh / p2, b = oh % p2, ww = ow / p3, d = ow % p3;
    const size_t src = ((cc * p1 + a) * p2 + b) * p3 + d;
    out[idx] = y[((src * f + fr) * h + hh) * w + ww];
}
