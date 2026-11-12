// Kernels of the MiniMax-H3 resident chain (`gpu_cuda/h3.rs`): a stack of
// single-stream DiT blocks (adaLN-modulated RMS norms, one fused qkv
// projection, full bidirectional attention with per-head q/k norm and a
// partial split-half rotation, a SwiGLU MLP with a row-packed [gate | up]
// first layer) with the residual stream on the card from the first block to
// the last. The weight products are cuBLAS GEMMs over f16 operands with f32
// accumulation; these kernels are everything between them.
//
// Layouts:
//   x     f32 [n][hs]            the residual stream
//   xn    f16 [rows][hs]         a chunk of normed, modulated rows
//   y     f32 [rows][3 * inner]  a chunk's qkv panel, row = [q | k | v], each
//                                [nh][hd]
//   Q K   f16 [nh][n][hd]        head-major planes (one GEMM operand a head)
//   VT    f16 [nh][hd][ns]       the values dim-major, ns = n rounded up to 16
//   S     f32 [nh][c][n]         a chunk of score rows, P f16 the same shape
//                                (only where attention is two GEMMs and not
//                                the flash kernel of h3_flash.cu)
//   mods  f32 [R][6 * hs]        shift, scale, gate for attention, then MLP
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

// RMS norm of a chunk of rows, its weight, and — `mods` set — the adaLN
// scale and shift of each row's modulation row; f16 out. A block a row.
extern "C" __global__ void h3_norm_mod(const float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ mods, const u32* __restrict__ mrow,
                                       u16* __restrict__ out, int hs, float eps, int shift_e, int scale_e)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* xr = x + (size_t)blockIdx.x * hs;
    float s = 0.0f;
    for (int i = tid; i < hs; i += T) {
        float v = xr[i];
        s += v * v;
    }
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    double tot = 0.0;
    for (int k = 0; k < NW; k++) tot += (double)red[k];
    const float inv = rsqrtf((float)(tot / (double)hs) + eps);
    u16* o = out + (size_t)blockIdx.x * hs;
    if (mods) {
        const float* m = mods + (size_t)mrow[blockIdx.x] * 6 * hs;
        const float* sh = m + (size_t)shift_e * hs;
        const float* sc = m + (size_t)scale_e * hs;
        for (int i = tid; i < hs; i += T) o[i] = f2h_sat(xr[i] * inv * w[i] * (1.0f + sc[i]) + sh[i]);
    } else {
        for (int i = tid; i < hs; i += T) o[i] = f2h_sat(xr[i] * inv * w[i]);
    }
}

// A chunk's qkv panel into the head-major planes: an optional bias, per-head
// RMS norm of q and k (with their weights, or none), the split-half rotation
// of the first 2 * pairs dims (pair (j, j + pairs) turns by the row's angle
// j; `cs` holds (cos, sin) per row and pair), v as it is. `inter` = 0: a row
// is [q | k | v], each [nh][hd]; 1: [nh][q k v] (the ViT decoder's panel).
// The sequence is `n` rows a group; row `row0 + r` of the launch lands in
// plane (group * nh + head). Q is [planes][n][hd]. K and the values come in
// two layouts:
//   lanes = 0 (attention as two GEMMs): K [planes][n][hd], the values
//     dim-major, VT [planes][hd][ns] with ns = n rounded up to 16;
//   lanes = 1 (the flash kernel of h3_flash.cu): both in the order its lanes
//     read them, so that a lane's operands of a whole tile are one contiguous
//     run — K [planes][n][4 tig][hd / 8][2] (dims 8m + 2 tig + e), the
//     values [planes][ns / 16 tiles][8 gid][4 tig][hd / 8][4] (dim 8j + gid,
//     keys 2 tig + e and 2 tig + 8 + e of the tile).
// Keys past n are zeros in either. A block a (row, head), a thread a dim.
extern "C" __global__ void h3_qkv_split(const float* __restrict__ y, const float* __restrict__ bias,
                                        const float* __restrict__ qn, const float* __restrict__ kn,
                                        const float* __restrict__ cs, u16* __restrict__ Q, u16* __restrict__ K,
                                        u16* __restrict__ VT, int row0, int n, int ns, int nh, int hd, int pairs,
                                        float eps, int inter, int lanes)
{
    __shared__ float red[2][8];
    __shared__ float qs[256], ks[256];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, lane = d & 31, warp = d >> 5, NW = (hd + 31) >> 5;
    const size_t inner = (size_t)nh * hd;
    const size_t oq = inter ? (size_t)h * 3 * hd : (size_t)h * hd;
    const size_t ok = inter ? oq + hd : oq + inner;
    const size_t ov = inter ? oq + 2 * hd : oq + 2 * inner;
    const float* yr = y + (size_t)r * 3 * inner;
    float q = 0.0f, k = 0.0f, v = 0.0f;
    if (d < hd) {
        q = yr[oq + d];
        k = yr[ok + d];
        v = yr[ov + d];
        if (bias) {
            q += bias[oq + d];
            k += bias[ok + d];
            v += bias[ov + d];
        }
    }
    float sq = q * q, sk = k * k;
    for (int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
    if (lane == 0) {
        red[0][warp] = sq;
        red[1][warp] = sk;
    }
    __syncthreads();
    float tq = 0.0f, tk = 0.0f;
    for (int i = 0; i < NW; i++) {
        tq += red[0][i];
        tk += red[1][i];
    }
    const float iq = rsqrtf(tq / (float)hd + eps);
    const float ik = rsqrtf(tk / (float)hd + eps);
    if (d < hd) {
        q = q * iq * (qn ? qn[d] : 1.0f);
        k = k * ik * (kn ? kn[d] : 1.0f);
        qs[d] = q;
        ks[d] = k;
    }
    __syncthreads();
    if (d >= hd) return;
    if (d < 2 * pairs) {
        const int j = d < pairs ? d : d - pairs;
        const float c = cs[((size_t)r * pairs + j) * 2], s = cs[((size_t)r * pairs + j) * 2 + 1];
        if (d < pairs) {
            q = qs[j] * c - qs[j + pairs] * s;
            k = ks[j] * c - ks[j + pairs] * s;
        } else {
            q = qs[j] * s + qs[j + pairs] * c;
            k = ks[j] * s + ks[j + pairs] * c;
        }
    }
    const int row = row0 + r, g = row / n, local = row - g * n;
    const size_t plane = (size_t)g * nh + h;
    const size_t o = (plane * n + local) * hd + d;
    Q[o] = f2h_sat(q);
    if (!lanes) {
        K[o] = f2h_sat(k);
        u16* vt = VT + (plane * hd + d) * ns;
        vt[local] = f2h_sat(v);
        if (local == n - 1)
            for (int i = n; i < ns; i++) vt[i] = 0;
        return;
    }
    // dim d = 8 m + 2 tig + e
    const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
    K[(plane * n + local) * hd + (size_t)tig * (hd >> 2) + 2 * m + e] = f2h_sat(k);
    // key kk of tile kt: fragment half = kk / 8, tig = (kk % 8) / 2, e = kk % 2;
    // dim d = 8 j + gid
    const int j = d >> 3, gid = d & 7;
    u16* vt = VT + (plane * (ns >> 4) * 8 + gid) * (size_t)(hd >> 1) * 4;
    #define VPOS(key) ((((size_t)((key) >> 4) * 8) * 4 + (((key) & 7) >> 1)) * (hd >> 1) + (size_t)j * 4 + (((key) & 15) >> 3) * 2 + ((key) & 1))
    vt[VPOS(local)] = f2h_sat(v);
    if (local == n - 1)
        for (int i = n; i < ns; i++) vt[VPOS(i)] = 0;
    #undef VPOS
}

// Score rows into attention weights (a block a row): P = exp(S - max) *
// pscale as f16, and the row's sum of exp(S - max) into Z. The division by
// the sum is left to `h3_heads_join`, after the value GEMM (whose alpha takes
// pscale back out): the row is read twice and exponentiated once. The factor
// keeps a weight a millionth of the row's largest clear of the half
// subnormals.
extern "C" __global__ void h3_softmax(const float* __restrict__ S, u16* __restrict__ P, float* __restrict__ Z,
                                      int n, float pscale)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* s = S + (size_t)blockIdx.x * n;
    u16* p = P + (size_t)blockIdx.x * n;
    float m = -3.0e38f;
    for (int i = tid; i < n; i += T) m = fmaxf(m, s[i]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float e = expf(s[i] - m);
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

// A chunk's attention output, head-major f32 [groups * nh][c][hd] and still
// carrying each score row's sum Z [groups * nh][c], as the f16 rows
// [groups * c][nh * hd] of the output projection's operand.
extern "C" __global__ void h3_heads_join(const float* __restrict__ O, const float* __restrict__ Z,
                                         u16* __restrict__ AO, int c, int nh, int hd, int groups)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t inner = (size_t)nh * hd;
    if (idx >= (size_t)groups * c * inner) return;
    const size_t row = idx / inner, rem = idx - row * inner;
    const size_t g = row / c, r = row - g * c;
    const size_t h = rem / hd, d = rem - h * hd;
    const size_t pr = (g * nh + h) * c + r;
    AO[idx] = f2h_sat(O[pr * hd + d] / Z[pr]);
}

// x += gate * (y + bias) over a chunk of rows. The gate is chunk `gate_e` of
// each row's modulation row, else a per-channel scale `chan`, else one; the
// bias [hs] is optional.
extern "C" __global__ void h3_resid(float* __restrict__ x, const float* __restrict__ y,
                                    const float* __restrict__ mods, const u32* __restrict__ mrow,
                                    const float* __restrict__ chan, const float* __restrict__ bias, int hs,
                                    int gate_e, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / hs, i = idx - r * hs;
    const float v = bias ? y[idx] + bias[i] : y[idx];
    if (mods) x[idx] += mods[((size_t)mrow[r] * 6 + gate_e) * hs + i] * v;
    else if (chan) x[idx] += chan[i] * v;
    else x[idx] += v;
}

// SwiGLU over a chunk: gu f32 [rows][2 * ffn] = [gate | up] a row, out f16
// [rows][ffn] = silu(gate) * up * inv. The product has no bound a half could
// rely on, so it is stored divided by a power of two that the second GEMM's
// alpha multiplies back.
extern "C" __global__ void h3_swiglu(const float* __restrict__ gu, const float* __restrict__ bias,
                                     u16* __restrict__ a, int ffn, float inv, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / ffn, i = idx - r * ffn;
    float g = gu[r * 2 * ffn + i], u = gu[r * 2 * ffn + ffn + i];
    if (bias) {
        g += bias[i];
        u += bias[ffn + i];
    }
    a[idx] = f2h_sat(g / (1.0f + expf(-g)) * u * inv);
}

// f32 -> f16 over n values, saturating (a batch of rows on its way to a GEMM).
extern "C" __global__ void h3_to_half(const float* __restrict__ x, u16* __restrict__ xh, u64 n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xh[i] = f2h_sat(x[i]);
}

// Diagnostics of CMF_H3_SAT=1. How many halves of a panel sit at the
// saturation value (cell[0]) or are not finite (cell[1]) ...
extern "C" __global__ void h3_sat_count(const u16* __restrict__ p, u64 n, u32* __restrict__ cell)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u16 a = p[i] & 0x7fff;
    if (a == 0x7bff) atomicAdd(cell, 1u);
    else if (a >= 0x7c00) atomicAdd(cell + 1, 1u);
}

// ... and the largest magnitude of an f32 panel, as the bits of a float (the
// order of non-negative floats is the order of their bits).
extern "C" __global__ void h3_abs_max(const float* __restrict__ x, u64 n, int* __restrict__ cell)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = __float_as_int(fabsf(x[i]));
    if (b > *cell) atomicMax(cell, b);
}

// Head-major f32 planes [planes][n][hds] from the host as the f16 operands of
// the flash kernel, the head padded with zeros to hd: mode 0 the queries
// [planes][n][hd], 1 the keys in lane order, 2 the values in lane order
// (zeros past n). A thread a destination element.
extern "C" __global__ void h3_pack(const float* __restrict__ src, u16* __restrict__ dst, int n, int ns, int planes,
                                   int hds, int hd, int mode)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)planes * ns * hd) return;
    const int d = (int)(idx % hd);
    const size_t t = idx / hd;
    const int row = (int)(t % ns);
    const size_t plane = t / ns;
    const u16 v = (row < n && d < hds) ? f2h_sat(src[(plane * n + row) * hds + d]) : (u16)0;
    if (mode == 2) {
        const int j = d >> 3, gid = d & 7;
        dst[plane * ns * hd + ((size_t)(row >> 4) * 32 + gid * 4 + ((row & 7) >> 1)) * (hd >> 1) + j * 4 +
            ((row & 15) >> 3) * 2 + (row & 1)] = v;
    } else if (row < n) {
        if (mode == 0) {
            dst[(plane * n + row) * hd + d] = v;
        } else {
            const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
            dst[(plane * n + row) * hd + (size_t)tig * (hd >> 2) + 2 * m + e] = v;
        }
    }
}

// The flash kernel's output f16 [n][nh * hd] back as f32 [n][nh * hds] (the
// padding of each head dropped).
extern "C" __global__ void h3_unpack(const u16* __restrict__ ao, float* __restrict__ out, int n, int nh, int hds, int hd)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * nh * hds) return;
    const int d = (int)(idx % hds);
    const size_t t = idx / hds;
    const int h = (int)(t % nh);
    const size_t row = t / nh;
    float f;
    const u16 hv = ao[(row * nh + h) * hd + d];
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(hv));
    out[idx] = f;
}

// One part of a projection panel into its attention planes, for stacks whose
// q, k and v are separate projections with their own head counts (a language
// model's grouped-query attention). The panel is f32 [rows][stride]; this
// part's head h sits at off + h * hd. Per-head RMS norm with weights `w` and
// the split-half rotation (modes 0 and 1; `cs` holds (cos, sin) per row and
// pair), then: mode 0 the queries [heads][n][hd]; 1 the keys in the flash
// kernel's lane order; 2 the values in lane order, as they are (zeros past
// n). A block a (row, head), a thread a dim.
extern "C" __global__ void h3_part_split(const float* __restrict__ y, int stride, int off,
                                         const float* __restrict__ w, const float* __restrict__ cs,
                                         u16* __restrict__ OUT, int row0, int n, int ns, int hd, int pairs,
                                         float eps, int mode)
{
    __shared__ float red[8];
    __shared__ float xs[256];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, lane = d & 31, warp = d >> 5, NW = (hd + 31) >> 5;
    float x = d < hd ? y[(size_t)r * stride + off + (size_t)h * hd + d] : 0.0f;
    if (mode != 2) {
        float sq = x * x;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) red[warp] = sq;
        __syncthreads();
        float tq = 0.0f;
        for (int i = 0; i < NW; i++) tq += red[i];
        if (d < hd) {
            x = x * rsqrtf(tq / (float)hd + eps) * (w ? w[d] : 1.0f);
            xs[d] = x;
        }
        __syncthreads();
        if (d < 2 * pairs) {
            const int j = d < pairs ? d : d - pairs;
            const float c = cs[((size_t)r * pairs + j) * 2], s = cs[((size_t)r * pairs + j) * 2 + 1];
            x = d < pairs ? xs[j] * c - xs[j + pairs] * s : xs[j] * s + xs[j + pairs] * c;
        }
    }
    if (d >= hd) return;
    const int row = row0 + r;
    const size_t plane = h;
    const u16 v = f2h_sat(x);
    if (mode == 0) {
        OUT[(plane * n + row) * hd + d] = v;
    } else if (mode == 1) {
        const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
        OUT[(plane * n + row) * hd + (size_t)tig * (hd >> 2) + 2 * m + e] = v;
    } else {
        const int j = d >> 3, gid = d & 7;
        u16* vt = OUT + (plane * (ns >> 4) * 8 + gid) * (size_t)(hd >> 1) * 4;
        #define VPOS(key) ((((size_t)((key) >> 4) * 8) * 4 + (((key) & 7) >> 1)) * (hd >> 1) + (size_t)j * 4 + (((key) & 15) >> 3) * 2 + ((key) & 1))
        vt[VPOS(row)] = v;
        if (row == n - 1)
            for (int i = n; i < ns; i++) vt[VPOS(i)] = 0;
        #undef VPOS
    }
}

// ---------------------------------------------------- a convolution chain
//
// The VAE encoder's stages on the card, over one activation kept
// pixel-major — x f32 [h * w][c] — so that a convolution is an im2col panel
// and one GEMM whose output is already the next stage's layout.

// Channel-major [c][hw] <-> pixel-major [hw][c].
extern "C" __global__ void cv_swap(const float* __restrict__ src, float* __restrict__ dst, int c, int hw, int to_pixel)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)c * hw) return;
    const size_t ch = idx / hw, px = idx - ch * hw;
    if (to_pixel) dst[px * c + ch] = src[idx];
    else dst[idx] = src[px * c + ch];
}

// GroupNorm statistics of x [hw][c]: (mean, 1 / sqrt(var + eps)) of each of
// the 32 groups. A block a group.
extern "C" __global__ void cv_gn_stats(const float* __restrict__ x, float* __restrict__ st, int c, int hw, float eps)
{
    __shared__ double red[2][8];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int g = blockIdx.x, per = c / 32;
    const size_t total = (size_t)hw * per;
    double s = 0.0, q = 0.0;
    for (size_t i = tid; i < total; i += T) {
        const size_t px = i / per;
        const float v = x[px * c + g * per + (i - px * per)];
        s += v;
        q += (double)v * v;
    }
    for (int k = 16; k > 0; k >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, k);
        q += __shfl_xor_sync(0xffffffffu, q, k);
    }
    if (lane == 0) {
        red[0][warp] = s;
        red[1][warp] = q;
    }
    __syncthreads();
    if (tid == 0) {
        double ts = 0.0, tq = 0.0;
        for (int k = 0; k < NW; k++) {
            ts += red[0][k];
            tq += red[1][k];
        }
        const double mean = ts / (double)total;
        const double var = tq / (double)total - mean * mean;
        st[2 * g] = (float)mean;
        st[2 * g + 1] = (float)(1.0 / sqrt(fmax(var, 0.0) + (double)eps));
    }
}

// GroupNorm with its affine weights, then SiLU, in place.
extern "C" __global__ void cv_gn_silu(float* __restrict__ x, const float* __restrict__ st, const float* __restrict__ w,
                                      const float* __restrict__ b, int c, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ch = (int)(idx % c), g = ch / (c / 32);
    const float v = (x[idx] - st[2 * g]) * st[2 * g + 1] * w[ch] + b[ch];
    x[idx] = v / (1.0f + expf(-v));
}

// im2col of output pixels p0.. of a k x k convolution over x [h * w][c]:
// cols f16 [pixels][c * k * k] in the weight's (channel, ky, kx) order.
// `pad` rows and columns of reflect padding on every side; `far`: one
// reflected row and column on the far side instead (index h reads h - 2),
// the downsample's padding. A thread a (pixel, channel).
extern "C" __global__ void cv_im2col(const float* __restrict__ x, u16* __restrict__ cols, int c, int h, int w, int ow,
                                     int k, int stride, int pad, int far, int p0, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const int p = p0 + (int)(idx / c);
    const int oy = p / ow, ox = p - oy * ow;
    u16* dst = cols + (idx / c) * (size_t)c * k * k + (size_t)ch * k * k;
    for (int ky = 0; ky < k; ky++) {
        int sy = oy * stride + ky - pad;
        if (sy < 0) sy = -sy;
        if (sy >= h) sy = far ? h - 2 : 2 * (h - 1) - sy;
        for (int kx = 0; kx < k; kx++) {
            int sx = ox * stride + kx - pad;
            if (sx < 0) sx = -sx;
            if (sx >= w) sx = far ? w - 2 : 2 * (w - 1) - sx;
            dst[ky * k + kx] = f2h_sat(x[((size_t)sy * w + sx) * c + ch]);
        }
    }
}

// y [rows][c] += bias [c] (and `add` [rows][c] when given).
extern "C" __global__ void cv_add(float* __restrict__ y, const float* __restrict__ bias, const float* __restrict__ add,
                                  int c, u64 total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = y[idx];
    if (bias) v += bias[idx % c];
    if (add) v += add[idx];
    y[idx] = v;
}
