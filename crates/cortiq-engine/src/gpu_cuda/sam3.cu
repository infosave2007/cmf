// Kernels of SAM3's ViT trunk on CUDA (`gpu_cuda/sam3.rs`): LayerNorm of
// rows into an f16 product input (or in place), the fused q|k|v rows into the
// flash kernel's planes with the bias and the axial rotation (a 24 x 24
// window a group under `win`, or the whole 72 x 72 grid), a product added
// back onto the stream in raster order, and the erf GELU.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#define GRID 72
#define WIN 24
#define NWIN 3
#define HD 64
#define HEADS 16
#define DIM (HD * HEADS)

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ u16 f2h_sat(float f) { return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f)); }

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

// LayerNorm of each row of x [rows][width] with weight w and bias b: into
// xn (f16) or, without xn, back into x. A block a row.
extern "C" __global__ void sm_ln(float* __restrict__ x, u16* __restrict__ xn, const float* __restrict__ w,
                                 const float* __restrict__ b, int width, float eps)
{
    const size_t r = blockIdx.x;
    float* xr = x + r * width;
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
        const float v = (xr[i] - mean) * inv * w[i] + b[i];
        if (xn) xn[r * width + i] = f2h_sat(v);
        else xr[i] = v;
    }
}

// The raster token t of the 72 x 72 grid as (group, row in group): its 24 x 24
// window and the row inside it under `win`, else (0, t).
__device__ __forceinline__ void place(int t, int win, int& g, int& r) {
    if (win) {
        const int y = t / GRID, x = t - y * GRID;
        g = (y / WIN) * NWIN + x / WIN;
        r = (y % WIN) * WIN + x % WIN;
    } else {
        g = 0;
        r = t;
    }
}

// qkv [tokens][3 DIM] (raster rows) plus its bias into the flash planes
// (plane = group·HEADS + head, n rows a plane): Q [plane][n][HD], K and V in
// the lane orders of `h3_flash` (h3_pack modes 1 and 2), q and k rotated by
// the pair table fc [n][32][2] (pair j of a head = dims 2j, 2j+1, at the
// token's row in its group). A thread a (token, head, pair).
extern "C" __global__ void sm_pack(const float* __restrict__ qkv, const float* __restrict__ bias,
                                   const float* __restrict__ fc, u16* __restrict__ Q, u16* __restrict__ K,
                                   u16* __restrict__ V, int tokens, int win)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (u64)tokens * HEADS * (HD / 2)) return;
    const int j = (int)(idx % (HD / 2));
    const u64 rest = idx / (HD / 2);
    const int h = (int)(rest % HEADS), t = (int)(rest / HEADS);
    int g, r;
    place(t, win, g, r);
    const int n = win ? WIN * WIN : tokens;
    const size_t plane = (size_t)g * HEADS + h;
    const float* row = qkv + (size_t)t * 3 * DIM;
    const int d0 = h * HD + 2 * j;
    const float c = fc[((size_t)r * 32 + j) * 2], s = fc[((size_t)r * 32 + j) * 2 + 1];
    const float qa = row[d0] + bias[d0], qb = row[d0 + 1] + bias[d0 + 1];
    const float ka = row[DIM + d0] + bias[DIM + d0], kb = row[DIM + d0 + 1] + bias[DIM + d0 + 1];
    const float va = row[2 * DIM + d0] + bias[2 * DIM + d0], vb = row[2 * DIM + d0 + 1] + bias[2 * DIM + d0 + 1];
    const float q0 = qa * c - qb * s, q1 = qb * c + qa * s;
    const float k0 = ka * c - kb * s, k1 = kb * c + ka * s;
    // Q: plain rows
    Q[(plane * n + r) * HD + 2 * j] = f2h_sat(q0);
    Q[(plane * n + r) * HD + 2 * j + 1] = f2h_sat(q1);
    // K: dim d at tig·HD/4 + 2m + e (m = d >> 3, tig = (d & 7) >> 1, e = d & 1)
    for (int e = 0; e < 2; e++) {
        const int d = 2 * j + e;
        const int m = d >> 3, tig = (d & 7) >> 1;
        K[(plane * n + r) * HD + (size_t)tig * (HD >> 2) + 2 * m + e] = f2h_sat(e ? k1 : k0);
        const int jj = d >> 3, gid = d & 7;
        V[plane * n * HD + ((size_t)(r >> 4) * 32 + gid * 4 + ((r & 7) >> 1)) * (HD >> 1) + jj * 4 + ((r & 15) >> 3) * 2 + (r & 1)] =
            f2h_sat(e ? vb : va);
    }
}

// x[t] += y[row of t] + bias over the stream [tokens][width] in raster order;
// y's rows are in group order under `win` (the flash output's), else raster.
// A thread a value.
extern "C" __global__ void sm_addrows(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ bias,
                                      int tokens, int width, int win)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)tokens * width) return;
    const int e = (int)(i % width), t = (int)(i / width);
    int g, r;
    place(t, win, g, r);
    const size_t src = win ? ((size_t)g * WIN * WIN + r) : (size_t)t;
    x[i] += y[src * width + e] + bias[e];
}

// act = f16(gelu_erf(y + bias)) over rows `width` wide.
extern "C" __global__ void sm_gelu(const float* __restrict__ y, const float* __restrict__ bias, u16* __restrict__ act, int width,
                                   u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] + bias[i % width];
    act[i] = f2h_sat(0.5f * v * (1.0f + erff(v * 0.70710678118f)));
}

// x += a (elementwise).
extern "C" __global__ void sm_add(float* __restrict__ x, const float* __restrict__ a, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += a[i];
}

// cols [ci·kh·kw][np] (f32, K-major) of output positions p0.. of a conv of
// stride `stride` and zero padding `pad` over x [ci][h][w]; the output ow
// wide. A thread a (column, position).
extern "C" __global__ void sm_im2col(const float* __restrict__ x, float* __restrict__ cols, int ci, int h, int w, int kh, int kw,
                                     int stride, int pad, int ow, u64 p0, u64 np)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const int taps = kh * kw;
    if (idx >= (u64)ci * taps * np) return;
    const u64 i = idx % np;
    const int kidx = (int)(idx / np);
    const int c = kidx / taps, t = kidx - c * taps, dy = t / kw, dx = t - dy * kw;
    const u64 p = p0 + i;
    const int ox = (int)(p % ow), oy = (int)(p / ow);
    const int iy = oy * stride + dy - pad, ix = ox * stride + dx - pad;
    cols[idx] = (iy >= 0 && iy < h && ix >= 0 && ix < w) ? x[((size_t)c * h + iy) * w + ix] : 0.0f;
}

// y[c][p] += b[c] over [channels][P].
extern "C" __global__ void sm_bias_ch(float* __restrict__ y, const float* __restrict__ b, u64 P, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i / P];
}

// The 2 x 2 stride-2 transposed conv's taps yt [co·4][h·w] (tap = dy·2 + dx)
// into out [co][2h][2w] with the bias. A thread an output value.
extern "C" __global__ void sm_convt_scatter(const float* __restrict__ yt, const float* __restrict__ b, float* __restrict__ out,
                                            int co, int h, int w)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const int ow = 2 * w, oh = 2 * h;
    if (i >= (u64)co * oh * ow) return;
    const int ox = (int)(i % ow);
    const u64 r = i / ow;
    const int oy = (int)(r % oh), c = (int)(r / oh);
    const int tap = (oy & 1) * 2 + (ox & 1);
    out[i] = yt[((size_t)c * 4 + tap) * h * w + (size_t)(oy >> 1) * w + (ox >> 1)] + b[c];
}

// Rows of raw scores S [rows][nk] into softmax weights after the scale and
// the mask: mode 0 none, 1 a key row [nk], 2 an additive [rows][nk] (row r0
// + r), 3 the same per head (already offset). A block a row.
extern "C" __global__ void sm_softmax(float* __restrict__ S, const float* __restrict__ mask, int nk, int mode, float scale, int r0)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const size_t r = blockIdx.x;
    float* s = S + r * nk;
    const float* mr = mode == 1 ? mask : (mode >= 2 ? mask + (r0 + r) * nk : 0);
    float m = -3.0e38f;
    for (int i = tid; i < nk; i += T) {
        const float v = s[i] * scale + (mr ? mr[i] : 0.0f);
        s[i] = v;
        m = fmaxf(m, v);
    }
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < nk; i += T) {
        const float e = expf(s[i] - m);
        s[i] = e;
        z += e;
    }
    for (int k = 16; k > 0; k >>= 1) z += __shfl_xor_sync(0xffffffffu, z, k);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < NW; k++) tot += red[k];
    const float inv = 1.0f / tot;
    for (int i = tid; i < nk; i += T) s[i] *= inv;
}

// y[r][e] += b[e] over rows `width` wide.
extern "C" __global__ void sm_bias_rows(float* __restrict__ y, const float* __restrict__ b, int width, u64 total)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i % width];
}
