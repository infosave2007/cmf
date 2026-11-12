// Nemotron Labs Dory on CUDA (`gpu_cuda/hyb/dory.rs`): the stages between
// the hybrid arm's tensor-core matvecs (`hyb.cu`), the twin of
// `gpu_metal/dory_msl.metal` (`dory.rs` has the semantics). Compiled with
// HPK query heads per KV head and HD the head width.
//
// Every kernel that feeds a matvec writes its rows as the bf16 planes the
// tensor-core kernel reads (`put3`, as in `hyb.cu`). K and V rows are f16:
// a sliding layer's in a ring of `cap` rows by position, a full layer's at
// their position. Row b of a call sits at position pos0 + b and reads the
// keys from max(0, pos + 1 - window) (window 0: all of them).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#define DPL (HD / 32)
#define INF __int_as_float(0x7f800000)
#define FULL 0xffffffffu
#ifndef NPL
#define NPL 2
#endif

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
__device__ __forceinline__ void h8(const u16* p, float* o) {
    const uint4 w = *(const uint4*)p;
    const u32 a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        o[2 * i] = h2f((u16)(a[i] & 0xFFFFu));
        o[2 * i + 1] = h2f((u16)(a[i] >> 16));
    }
}
__device__ __forceinline__ u16 f2bf(float f) {
    u16 h;
    asm("{ cvt.rn.bf16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float bf2f(u16 h) { return __uint_as_float((u32)h << 16); }
__device__ __forceinline__ void put3(u16* __restrict__ p0, u16* __restrict__ p1, size_t i, float v)
{
    const u16 a = f2bf(v);
    p0[i] = a;
    p1[i] = f2bf(__fsub_rn(v, bf2f(a)));
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// The sum of v over the block (blockDim.x a multiple of 32, at most 1024).
__device__ __forceinline__ float bsum(float v, float* red)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < nw; w++) t += red[w];
    return t;
}

__device__ __forceinline__ int low_key(int pos, int window) { return window > 0 && pos + 1 > window ? pos + 1 - window : 0; }

// The call record a replayed graph reads its positions from: [0] pos0.
__device__ __forceinline__ int rec_pos(const int* cr, int pos0) { return cr ? cr[0] : pos0; }

// Rotation, unit heads, the query scale and the gate's scale; K and V into
// their rows. A block a (head, row), HD threads. yq/yg [B][nh][HD], yk/yv
// [B][nkv][HD]; rope [B][HD] (cos of the pairs, then their sin). qg
// [nkv][B][HPK][HD] (the decode attention's layout), gate [B][nh][HD].
// `qtr`: every QT-th position's queries [pos / QT][nh][HD] for a far
// field's seal. `fci`: a sealed layer's call record, row b's K/V at fci[1] + b
// (a prompt's last rows past the seal sit after the near rows, as a verify
// window's do on the hybrid arm).
extern "C" __global__ void dk_qk(const float* __restrict__ yq, const float* __restrict__ yk, const float* __restrict__ yv,
                                 const float* __restrict__ yg, const float* __restrict__ rope, const float* __restrict__ sq,
                                 const float* __restrict__ sg, float* __restrict__ qg, float* __restrict__ gate,
                                 u16* __restrict__ Km, u16* __restrict__ Vm, float* __restrict__ qtr, int pos0, int cap,
                                 int nh, int nkv, float eps, int B, const int* __restrict__ cr, const int* __restrict__ fci,
                                 u16* __restrict__ qgh)
{
    __shared__ float xs[HD];
    __shared__ float red[32];
    const int a = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const int half = HD / 2, i = t < half ? t : t - half;
    const int pos = rec_pos(cr, pos0) + b;
    const float c = rope[(size_t)b * HD + i], sn = rope[(size_t)b * HD + half + i];
    const bool isq = a < nh;
    const float* src = isq ? yq + ((size_t)b * nh + a) * HD : yk + ((size_t)b * nkv + (a - nh)) * HD;
    xs[t] = src[t];
    __syncthreads();
    const float u = t < half ? xs[t] * c - xs[t + half] * sn : xs[t] * c + xs[t - half] * sn;
    const float inv = 1.0f / sqrtf(bsum(u * u, red) + eps);
    if (isq) {
        const float q = u * inv * sq[(size_t)a * HD + t];
        const int g = a / HPK, j = a - g * HPK;
        const size_t qi = (((size_t)g * B + b) * HPK + j) * HD + t;
        qg[qi] = q;
        if (qgh) qgh[qi] = f2h_sat(q);
        if (qtr && pos % 8 == 0) qtr[((size_t)(pos / 8) * nh + a) * HD + t] = q;
        const size_t gi = ((size_t)b * nh + a) * HD + t;
        gate[gi] = yg[gi] * sg[(size_t)a * HD + t];
        return;
    }
    const int kh = a - nh;
    const int row = fci ? fci[1] + b : (cap > 0 ? pos % cap : pos);
    const size_t at = ((size_t)row * nkv + kh) * HD + t;
    Km[at] = f2h_sat(u * inv);
    Vm[at] = f2h_sat(yv[((size_t)b * nkv + kh) * HD + t]);
}

// The decode attention of B rows, split over key blocks (`hy_scores` with the
// keys of a ring): S [B][nh][ncap], Mx [B][nh][nbcap]. A block `rb` keys of
// one KV head of one row; the rows of a key block are neighbours in the grid
// (x the row, y the block, z the KV head), so they read its keys from L2.
extern "C" __global__ void dk_scores(const float* __restrict__ qg, const u16* __restrict__ Km, float* __restrict__ S,
                                     float* __restrict__ Mx, int pos0, int cap, int window, int nkv, float scale, int rb,
                                     int ncap, int nbcap, int B, const int* __restrict__ cr)
{
    __shared__ float sm[32][HPK];
    const int g = blockIdx.z, blk = blockIdx.y, b = blockIdx.x, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int pos = rec_pos(cr, pos0) + b, lo = low_key(pos, window), n = pos + 1 - lo;
    if (blk * rb >= n) return;
    float qv[HPK][DPL];
#pragma unroll
    for (int j = 0; j < HPK; j++)
#pragma unroll
        for (int t = 0; t < DPL; t++) qv[j][t] = qg[(((size_t)g * B + b) * HPK + j) * HD + lane * DPL + t];
    float mx[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) mx[j] = -INF;
    const int row0 = blk * rb, row1 = min(row0 + rb, n);
    float* Sb = S + (size_t)b * nh * ncap;
    for (int r = row0 + warp; r < row1; r += NW) {
        const int kr_row = cap > 0 ? (lo + r) % cap : lo + r;
        const u16* kr = Km + ((size_t)kr_row * nkv + g) * HD + lane * DPL;
        float kv[DPL];
#pragma unroll
        for (int t = 0; t < DPL; t += 8) h8(kr + t, kv + t);
#pragma unroll
        for (int j = 0; j < HPK; j++) {
            float s = 0.0f;
#pragma unroll
            for (int t = 0; t < DPL; t++) s = fmaf(qv[j][t], kv[t], s);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
            s *= scale;
            if (lane == 0) Sb[(size_t)(g * HPK + j) * ncap + r] = s;
            mx[j] = fmaxf(mx[j], s);
        }
    }
    if (lane == 0)
        for (int j = 0; j < HPK; j++) sm[warp][j] = mx[j];
    __syncthreads();
    if (tid < HPK) {
        float m = -INF;
        for (int w = 0; w < NW; w++) m = fmaxf(m, sm[w][tid]);
        Mx[((size_t)b * nh + g * HPK + tid) * nbcap + blk] = m;
    }
}

// Each key block's exponentials (under the row's largest score) times its
// values: P [B][nbcap][nh][HD], E [B][nbcap][nh].
extern "C" __global__ void dk_values(const float* __restrict__ S, const float* __restrict__ Mx, const u16* __restrict__ Vm,
                                     float* __restrict__ P, float* __restrict__ E, int pos0, int cap, int window, int nkv,
                                     int rb, int ncap, int nbcap, int B, const int* __restrict__ cr)
{
    __shared__ float sp[8][HD];
    __shared__ float se[8][HPK];
    const int g = blockIdx.z, blk = blockIdx.y, b = blockIdx.x, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int pos = rec_pos(cr, pos0) + b, lo = low_key(pos, window), n = pos + 1 - lo;
    const int nblk = (n + rb - 1) / rb;
    if (blk >= nblk) return;
    const float* Sb = S + (size_t)b * nh * ncap;
    float mx[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        float m = -INF;
        for (int q = lane; q < nblk; q += 32) m = fmaxf(m, Mx[((size_t)b * nh + g * HPK + j) * nbcap + q]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        mx[j] = m;
    }
    float acc[HPK][DPL];
    float es[HPK];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        es[j] = 0.0f;
#pragma unroll
        for (int t = 0; t < DPL; t++) acc[j][t] = 0.0f;
    }
    const int row0 = blk * rb, row1 = min(row0 + rb, n);
    for (int r = row0 + warp; r < row1; r += NW) {
        const int vr_row = cap > 0 ? (lo + r) % cap : lo + r;
        const u16* vr = Vm + ((size_t)vr_row * nkv + g) * HD + lane * DPL;
        float vv[DPL];
#pragma unroll
        for (int t = 0; t < DPL; t += 8) h8(vr + t, vv + t);
#pragma unroll
        for (int j = 0; j < HPK; j++) {
            const float e = expf(Sb[(size_t)(g * HPK + j) * ncap + r] - mx[j]);
            es[j] += e;
#pragma unroll
            for (int t = 0; t < DPL; t++) acc[j][t] = fmaf(e, vv[t], acc[j][t]);
        }
    }
    float* Pb = P + (size_t)b * nbcap * nh * HD;
    float* Eb = E + (size_t)b * nbcap * nh;
    if (lane == 0)
        for (int j = 0; j < HPK; j++) se[warp][j] = es[j];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        __syncthreads();
#pragma unroll
        for (int t = 0; t < DPL; t++) sp[warp][lane * DPL + t] = acc[j][t];
        __syncthreads();
        for (int d = tid; d < HD; d += blockDim.x) {
            float s = 0.0f;
            for (int w = 0; w < NW; w++) s += sp[w][d];
            Pb[((size_t)blk * nh + g * HPK + j) * HD + d] = s;
        }
    }
    if (tid < HPK) {
        float s = 0.0f;
        for (int w = 0; w < NW; w++) s += se[w][tid];
        Eb[(size_t)blk * nh + g * HPK + tid] = s;
    }
}

// The blocks' parts over their weights, times the gate: z [B][nh][HD] as the
// output projection's planes. A block a (head, row).
extern "C" __global__ void dk_finish(const float* __restrict__ P, const float* __restrict__ E, const float* __restrict__ gate,
                                     u16* __restrict__ p0, u16* __restrict__ p1, int pstride, int pos0, int window, int rb,
                                     int nh, int nbcap, const int* __restrict__ cr, int half)
{
    const int h = blockIdx.x, b = blockIdx.y;
    const int pos = rec_pos(cr, pos0) + b, lo = low_key(pos, window), n = pos + 1 - lo;
    const int nblk = (n + rb - 1) / rb;
    const float* Pb = P + (size_t)b * nbcap * nh * HD;
    const float* Eb = E + (size_t)b * nbcap * nh;
    float tot = 0.0f;
    for (int q = 0; q < nblk; q++) tot += Eb[(size_t)q * nh + h];
    const float inv = tot > 0.0f ? 1.0f / tot : 1.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float s = 0.0f;
        for (int q = 0; q < nblk; q++) s += Pb[((size_t)q * nh + h) * HD + d];
        const size_t gi = ((size_t)b * nh + h) * HD + d;
        const float v = s * inv * sigm(gate[gi]);
        const size_t oi = (size_t)b * pstride + (size_t)h * HD + d;
        if (half) p0[oi] = f2h_sat(v); else put3(p0, p1, oi, v);
    }
}

// The MLP's activation: silu(g * a) * (u * b) as the down projection's planes.
extern "C" __global__ void dk_silu(const float* __restrict__ fg, const float* __restrict__ fu, const float* __restrict__ sa,
                                   const float* __restrict__ sb, u16* __restrict__ p0, u16* __restrict__ p1, int ffn, int B, int half)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * ffn) return;
    const int e = (int)(i % ffn);
    const float g = fg[i] * sa[e], u = fu[i] * sb[e];
    const float v = g / (1.0f + expf(-g)) * u;
    if (half) p0[i] = f2h_sat(v); else put3(p0, p1, i, v);
}

// The nGPT residual h = (ĥ + α (ô − ĥ))^, and h's planes. A block a row,
// its values in registers (EPT a thread: hidden <= EPT * blockDim).
#define EPT 8
extern "C" __global__ void dk_slerp(float* __restrict__ h, const float* __restrict__ o, const float* __restrict__ al,
                                    u16* __restrict__ p0, u16* __restrict__ p1, int hidden, float eps, u16* __restrict__ xh)
{
    __shared__ float red[32];
    float* x = h + (size_t)blockIdx.x * hidden;
    const float* y = o + (size_t)blockIdx.x * hidden;
    float xv[EPT], yv[EPT];
    float sx = 0.0f, so = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        const int e = threadIdx.x + i * blockDim.x;
        xv[i] = e < hidden ? x[e] : 0.0f;
        yv[i] = e < hidden ? y[e] : 0.0f;
        sx += xv[i] * xv[i];
        so += yv[i] * yv[i];
    }
    const float ix = 1.0f / sqrtf(bsum(sx, red) + eps);
    const float io = 1.0f / sqrtf(bsum(so, red) + eps);
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        const int e = threadIdx.x + i * blockDim.x;
        const float a = xv[i] * ix;
        xv[i] = e < hidden ? a + al[e] * (yv[i] * io - a) : 0.0f;
        t += xv[i] * xv[i];
    }
    const float r = 1.0f / sqrtf(bsum(t, red) + eps);
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        const int e = threadIdx.x + i * blockDim.x;
        if (e < hidden) {
            const float v = xv[i] * r;
            x[e] = v;
            put3(p0, p1, (size_t)blockIdx.x * hidden + e, v);
            if (xh) xh[(size_t)blockIdx.x * hidden + e] = f2h_sat(v);
        }
    }
}

// The loop's edges (`dory_mix` on Metal): mode 0 enters the first pass (input
// = state = h, h = (h + h)^), 1 goes between passes (state = ĥ, h = (ĥ +
// input)^), 2 leaves the last (h = ĥ), 3 reads out and keeps the state (h =
// state = ĥ), 4 starts the next pass from the kept state (h = (state +
// input)^). h's planes after. A block a row.
extern "C" __global__ void dk_mix(float* __restrict__ h, float* __restrict__ in_, float* __restrict__ st, int mode,
                                  u16* __restrict__ p0, u16* __restrict__ p1, int hidden, float eps, u16* __restrict__ xh)
{
    __shared__ float red[32];
    const size_t base = (size_t)blockIdx.x * hidden;
    float* x = h + base;
    float* inp = in_ + base;
    float* s = st + base;
    float acc = 0.0f;
    if (mode == 0) {
        for (int e = threadIdx.x; e < hidden; e += blockDim.x) {
            const float v = x[e];
            inp[e] = v;
            s[e] = v;
            const float y = v + v;
            x[e] = y;
            acc += y * y;
        }
    } else if (mode == 4) {
        for (int e = threadIdx.x; e < hidden; e += blockDim.x) {
            const float y = s[e] + inp[e];
            x[e] = y;
            acc += y * y;
        }
    } else {
        for (int e = threadIdx.x; e < hidden; e += blockDim.x) acc += x[e] * x[e];
        const float r = 1.0f / sqrtf(bsum(acc, red) + eps);
        acc = 0.0f;
        for (int e = threadIdx.x; e < hidden; e += blockDim.x) {
            const float u = x[e] * r;
            if (mode == 1) {
                s[e] = u;
                const float y = u + inp[e];
                x[e] = y;
                acc += y * y;
            } else {
                if (mode == 3) s[e] = u;
                x[e] = u;
                put3(p0, p1, base + e, u);
                if (xh) xh[base + e] = f2h_sat(u);
            }
        }
        if (mode != 1) return;
    }
    const float r = 1.0f / sqrtf(bsum(acc, red) + eps);
    for (int e = threadIdx.x; e < hidden; e += blockDim.x) {
        const float v = x[e] * r;
        x[e] = v;
        put3(p0, p1, base + e, v);
        if (xh) xh[base + e] = f2h_sat(v);
    }
}

// A windowed layer's scores of a prompt chunk in place, exponentials and
// 1/Σ (`hy_softmax_h` with a lower bound): S and P [nkv][cols][L], column
// (group gk, row r / HPK of the chunk) sees the keys of positions
// [pq + 1 - window, pq], key c at position col_base + c.
extern "C" __global__ void dk_softmax_w(const float* __restrict__ S, u16* __restrict__ P, float* __restrict__ isum, int L,
                                        int pos0, int col_base, int cols, int istride, int window)
{
    __shared__ float red[32];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const float* s = S + (size_t)blockIdx.x * L;
    u16* p = P + (size_t)blockIdx.x * L;
    const int gk = (int)blockIdx.x / cols, r = (int)blockIdx.x - gk * cols;
    const int pq = pos0 + r / HPK;
    const int hi = pq - col_base, lo = max(0, pq + 1 - window - col_base);
    float m = -INF;
    for (int c = lo + tid; c <= hi && c < L; c += T) m = fmaxf(m, s[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, k));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INF;
    for (int k = 0; k < NW; k++) m = fmaxf(m, red[k]);
    __syncthreads();
    float sum = 0.0f;
    for (int c = tid; c < L; c += T) {
        float e = 0.0f;
        if (c >= lo && c <= hi) {
            e = expf(s[c] - m);
            sum += e;
        }
        p[c] = f2h(e);
    }
    for (int k = 16; k > 0; k >>= 1) sum += __shfl_xor_sync(FULL, sum, k);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int k = 0; k < NW; k++) sum += red[k];
        isum[(size_t)gk * istride + r] = sum > 0.0f ? 1.0f / sum : 1.0f;
    }
}

// The decode attention in one pass over the keys (flash decoding): a block
// takes `rb` keys of one KV head of one row, each warp its keys with a running
// maximum, and leaves its block's maximum Mx, weight sum E and weighted values
// P under that maximum; `dk_finish_m` puts the blocks together. Grid (row,
// key block, KV head).
extern "C" __global__ void dk_attn(const float* __restrict__ qg, const u16* __restrict__ Km, const u16* __restrict__ Vm,
                                   float* __restrict__ P, float* __restrict__ E, float* __restrict__ Mx, int pos0, int cap,
                                   int window, int nkv, float scale, int rb, int nbcap, int B, const int* __restrict__ cr)
{
    __shared__ float sp[8][HD];
    __shared__ float sm[8][HPK];
    __shared__ float se[8][HPK];
    const int g = blockIdx.z, blk = blockIdx.y, b = blockIdx.x, nh = nkv * HPK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int pos = rec_pos(cr, pos0) + b, lo = low_key(pos, window), n = pos + 1 - lo;
    if (blk * rb >= n) return;
    float qv[HPK][DPL];
#pragma unroll
    for (int j = 0; j < HPK; j++)
#pragma unroll
        for (int t = 0; t < DPL; t++) qv[j][t] = qg[(((size_t)g * B + b) * HPK + j) * HD + lane * DPL + t];
    float m[HPK], es[HPK], acc[HPK][DPL];
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        m[j] = -INF;
        es[j] = 0.0f;
#pragma unroll
        for (int t = 0; t < DPL; t++) acc[j][t] = 0.0f;
    }
    const int row0 = blk * rb, row1 = min(row0 + rb, n);
    for (int r = row0 + warp; r < row1; r += NW) {
        const int kr_row = cap > 0 ? (lo + r) % cap : lo + r;
        const size_t at = ((size_t)kr_row * nkv + g) * HD + lane * DPL;
        float kv[DPL], vv[DPL];
#pragma unroll
        for (int t = 0; t < DPL; t += 8) {
            h8(Km + at + t, kv + t);
            h8(Vm + at + t, vv + t);
        }
#pragma unroll
        for (int j = 0; j < HPK; j++) {
            float s = 0.0f;
#pragma unroll
            for (int t = 0; t < DPL; t++) s = fmaf(qv[j][t], kv[t], s);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
            s *= scale;
            const float mn = fmaxf(m[j], s);
            const float a = expf(m[j] - mn), e = expf(s - mn);
            m[j] = mn;
            es[j] = es[j] * a + e;
#pragma unroll
            for (int t = 0; t < DPL; t++) acc[j][t] = fmaf(e, vv[t], acc[j][t] * a);
        }
    }
    if (lane == 0)
        for (int j = 0; j < HPK; j++) sm[warp][j] = m[j];
    __syncthreads();
    float* Pb = P + (size_t)b * nbcap * nh * HD;
    float* Eb = E + (size_t)b * nbcap * nh;
#pragma unroll
    for (int j = 0; j < HPK; j++) {
        float M = -INF;
        for (int w = 0; w < NW; w++) M = fmaxf(M, sm[w][j]);
        const float f = m[j] == -INF ? 0.0f : expf(m[j] - M);
        __syncthreads();
#pragma unroll
        for (int t = 0; t < DPL; t++) sp[warp][lane * DPL + t] = acc[j][t] * f;
        if (lane == 0) se[warp][j] = es[j] * f;
        __syncthreads();
        for (int d = tid; d < HD; d += blockDim.x) {
            float s = 0.0f;
            for (int w = 0; w < NW; w++) s += sp[w][d];
            Pb[((size_t)blk * nh + g * HPK + j) * HD + d] = s;
        }
        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NW; w++) s += se[w][j];
            Eb[(size_t)blk * nh + g * HPK + j] = s;
            Mx[((size_t)b * nh + g * HPK + j) * nbcap + blk] = M;
        }
    }
}

// `dk_attn`'s blocks put together under their common maximum, times the gate:
// the output projection's planes. A block a (head, row).
extern "C" __global__ void dk_finish_m(const float* __restrict__ P, const float* __restrict__ E, const float* __restrict__ Mx,
                                       const float* __restrict__ gate, u16* __restrict__ p0, u16* __restrict__ p1, int pstride,
                                       int pos0, int window, int rb, int nh, int nbcap, const int* __restrict__ cr)
{
    __shared__ float wq[256];
    const int h = blockIdx.x, b = blockIdx.y;
    const int pos = rec_pos(cr, pos0) + b, lo = low_key(pos, window), n = pos + 1 - lo;
    const int nblk = (n + rb - 1) / rb;
    const float* Pb = P + (size_t)b * nbcap * nh * HD;
    const float* Eb = E + (size_t)b * nbcap * nh;
    const float* Mb = Mx + ((size_t)b * nh + h) * nbcap;
    float M = -INF;
    for (int q = 0; q < nblk; q++) M = fmaxf(M, Mb[q]);
    for (int q = threadIdx.x; q < nblk; q += blockDim.x) wq[q] = expf(Mb[q] - M);
    __syncthreads();
    float tot = 0.0f;
    for (int q = 0; q < nblk; q++) tot += Eb[(size_t)q * nh + h] * wq[q];
    const float inv = tot > 0.0f ? 1.0f / tot : 1.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float s = 0.0f;
        for (int q = 0; q < nblk; q++) s += Pb[((size_t)q * nh + h) * HD + d] * wq[q];
        const size_t gi = ((size_t)b * nh + h) * HD + d;
        put3(p0, p1, (size_t)b * pstride + (size_t)h * HD + d, s * inv * sigm(gate[gi]));
    }
}
