// Inkling-2-Mini on CUDA (`gpu_cuda/hyb/inkling.rs`): the stages of the looped
// stack between the hybrid arm's tensor-core matvecs and routed experts. The
// semantics are `inkling.rs`'s host stack; every kernel takes the call's rows
// (a decode or verify call at most 16, a prompt call up to the arm's row cap)
// and reads the first position from the call record `cr[0]`.
//
// Compiled by NVRTC with H (hidden), HD (head width 128), NKV (KV heads), NH
// (query heads), DR (relative-state width), KC (convolution taps).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
#define FULL 0xffffffffu

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u16 f2bf(float f) {
    u16 h;
    asm("{ cvt.rn.bf16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float bf2f(u16 h) { return __uint_as_float((u32)h << 16); }
// the hybrid arm's two bf16 planes of an f32 (`put3` with NPL 2)
__device__ __forceinline__ void put2(u16* __restrict__ p0, u16* __restrict__ p1, size_t i, float v) {
    const u16 a = f2bf(v);
    p0[i] = a;
    p1[i] = f2bf(__fsub_rn(v, bf2f(a)));
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// A block's sum of v (all threads get it). `sh` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; i++) t += sh[i];
    return t;
}

// The block attention residual (`mix`): for row r and query qi, the softmax
// over the states [blk[r][0..nb), x[r]] (each unit-RMS normed, eps 1e-6) of
// q·s / √H, and the weighted sum of the normed states. Grid (nq, n), 256
// threads. out[qi * ostride + r * H + i].
#define MAXS 8
extern "C" __global__ void ink_mix(const float* __restrict__ blk, int nb, int bstride, const float* __restrict__ x,
                                   const float* __restrict__ q, float* __restrict__ out, int ostride)
{
    __shared__ float sh[32];
    __shared__ float lg[MAXS], rn[MAXS];
    const int qi = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const float* qv = q + (size_t)qi * H;
    const int ns = nb + 1;
    for (int s = 0; s < ns; s++) {
        const float* sv = s < nb ? blk + (size_t)r * bstride + (size_t)s * H : x + (size_t)r * H;
        float ss = 0.0f, dt = 0.0f;
        for (int i = tid; i < H; i += blockDim.x) {
            const float v = sv[i];
            ss = fmaf(v, v, ss);
            dt = fmaf(v, qv[i], dt);
        }
        ss = block_sum(ss, sh);
        dt = block_sum(dt, sh);
        if (tid == 0) {
            const float inv = rsqrtf(ss / (float)H + 1e-6f);
            rn[s] = inv;
            lg[s] = dt * inv * rsqrtf((float)H);
        }
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int s = 0; s < ns; s++) mx = fmaxf(mx, lg[s]);
    float w[MAXS], den = 0.0f;
    for (int s = 0; s < ns; s++) {
        w[s] = expf(lg[s] - mx);
        den += w[s];
    }
    for (int s = 0; s < ns; s++) w[s] = w[s] / den * rn[s];
    float* o = out + (size_t)qi * ostride + (size_t)r * H;
    for (int i = tid; i < H; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < ns; s++) {
            const float* sv = s < nb ? blk + (size_t)r * bstride + (size_t)s * H : x + (size_t)r * H;
            acc = fmaf(w[s], sv[i], acc);
        }
        o[i] = acc;
    }
}

// Weighted RMS norm of row r of x into xo (f32) and, when p0 is set, the
// tensor-core planes of the row. Grid n, 256 threads.
extern "C" __global__ void ink_rms(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                   float* __restrict__ xo, u16* __restrict__ p0, u16* __restrict__ p1, int pstride)
{
    __shared__ float sh[32];
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)r * H;
    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x) {
        const float v = w[i] * (xr[i] * inv);
        if (xo) xo[(size_t)r * H + i] = v;
        if (p0) put2(p0, p1, (size_t)r * pstride + i, v);
    }
}

// A loop layer's message gate, first half: d[r][k] = down[k] · [x/rms(x), m/rms(m)].
// Grid (R, n), 256 threads.
extern "C" __global__ void ink_gate_down(const float* __restrict__ x, const float* __restrict__ m,
                                         const float* __restrict__ down, int R, float eps, float* __restrict__ d)
{
    __shared__ float sh[32];
    const int k = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const float* xr = x + (size_t)r * H;
    const float* mr = m + (size_t)r * H;
    const float* dw = down + (size_t)k * 2 * H;
    float sx = 0.0f, sm = 0.0f, ax = 0.0f, am = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        const float a = xr[i], b = mr[i];
        sx = fmaf(a, a, sx);
        sm = fmaf(b, b, sm);
        ax = fmaf(dw[i], a, ax);
        am = fmaf(dw[H + i], b, am);
    }
    sx = block_sum(sx, sh);
    sm = block_sum(sm, sh);
    ax = block_sum(ax, sh);
    am = block_sum(am, sh);
    if (tid == 0) d[(size_t)r * R + k] = ax * rsqrtf(sx / (float)H + eps) + am * rsqrtf(sm / (float)H + eps);
}

// Second half: out = (1 − tanh(F·d))·x + tanh(G·d)·m, a warp an output
// channel (the lanes split the rank, a coalesced read of F's and G's rows).
// Grid (H/8, n), 256 threads.
extern "C" __global__ void ink_gate_out(const float* __restrict__ x, const float* __restrict__ m,
                                        const float* __restrict__ d, const float* __restrict__ F,
                                        const float* __restrict__ G, int R, float* __restrict__ out)
{
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), r = blockIdx.y;
    if (i >= H) return;
    const float* dr = d + (size_t)r * R;
    float f = 0.0f, g = 0.0f;
    for (int k = lane; k < R; k += 32) {
        f = fmaf(F[(size_t)i * R + k], dr[k], f);
        g = fmaf(G[(size_t)i * R + k], dr[k], g);
    }
    for (int o = 16; o > 0; o >>= 1) {
        f += __shfl_xor_sync(FULL, f, o);
        g += __shfl_xor_sync(FULL, g, o);
    }
    if (lane == 0) {
        const size_t at = (size_t)r * H + i;
        out[at] = (1.0f - tanhf(f)) * x[at] + tanhf(g) * m[at];
    }
}

// y[r][o] = W[o] · x[r] for an f32 matrix [rows][H]: a warp an output row.
// Grid (rows/8, n), 256 threads.
extern "C" __global__ void ink_fmv(const float* __restrict__ W, int rows, const float* __restrict__ x,
                                   float* __restrict__ y)
{
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), r = blockIdx.y;
    if (o >= rows) return;
    const float* w = W + (size_t)o * H;
    const float* xr = x + (size_t)r * H;
    float acc = 0.0f;
    for (int i = lane; i < H; i += 32) acc = fmaf(w[i], xr[i], acc);
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(FULL, acc, s);
    if (lane == 0) y[(size_t)r * rows + o] = acc;
}

__device__ __forceinline__ float tau_of(int pos, float alpha, float floor_) {
    return 1.0f + alpha * logf(fmaxf((float)(pos + 1) / floor_, 1.0f));
}

// Queries and relative states of the call's rows, in place: per head the
// learned scale, the L2 clip, and on the full layers the log scale of the
// position. Grid (NH, n), HD threads (HD >= 32).
extern "C" __global__ void ink_qr(float* __restrict__ yq, float* __restrict__ yr, const float* __restrict__ qs,
                                  const float* __restrict__ rs, float qclip, float rclip, int full, float alpha,
                                  float floor_, const int* __restrict__ cr)
{
    __shared__ float sh[32];
    const int h = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const float tu = full ? tau_of(cr[0] + r, alpha, floor_) : 1.0f;
    float* q = yq + ((size_t)r * NH + h) * HD;
    const float v = q[tid] * qs[h * HD + tid];
    const float n2 = block_sum(v * v, sh);
    const float s = fminf(qclip / (sqrtf(n2) + 1e-6f), 1.0f);
    q[tid] = v * s * tu;
    if (tid < 32) {
        float* rr = yr + ((size_t)r * NH + h) * DR;
        const float a = tid < DR ? rr[tid] * rs[h * DR + tid] : 0.0f;
        float t = a * a;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(FULL, t, o);
        const float sr = fminf(rclip / (sqrtf(t) + 1e-6f), 1.0f);
        if (tid < DR) rr[tid] = a * sr * tu;
    }
}

// The keys (blockIdx.y 0) or values (1) of the call's rows, a KV head a block
// (HD threads, a channel each): the short convolution over the ring (taps
// `taps` [C][KC], tap 0 the current row; `dyn` [n][NKV][KC] adds per-token
// taps on the full layers), `(x + conv) * scale`, then per head the keys'
// L2 clip or the values' unit RMS, into the f16 cache at the row's slot
// (`ring` > 0: position mod ring). The ring [KC-1][C] (newest first) moves on.
extern "C" __global__ void ink_kv(const float* __restrict__ yk, const float* __restrict__ yv,
                                  const float* __restrict__ dk, const float* __restrict__ dv,
                                  const float* __restrict__ tk, const float* __restrict__ tv, float* __restrict__ rk,
                                  float* __restrict__ rv, u16* __restrict__ kc, u16* __restrict__ vc, int ring, int n,
                                  float qclip, float eps, const int* __restrict__ cr, float* __restrict__ hk,
                                  float* __restrict__ hv)
{
    __shared__ float sh[32];
    const int g = blockIdx.x, isv = blockIdx.y, tid = threadIdx.x;
    // the call's input rows kept (a draft verify rolls the ring back to a prefix)
    float* ho = isv ? hv : hk;
    const int C = NKV * HD, ch = g * HD + tid;
    const float* y = isv ? yv : yk;
    const float* dy = isv ? dv : dk;
    const float* tp = (isv ? tv : tk) + (size_t)ch * KC;
    float* rg = isv ? rv : rk;
    u16* cache = isv ? vc : kc;
    float hist[KC];
    hist[0] = 0.0f;
#pragma unroll
    for (int l = 1; l < KC; l++) hist[l] = rg[(size_t)(l - 1) * C + ch];
    const float scale = rsqrtf((float)((dy ? 2 : 1) * KC + 1));
    const int pos0 = cr[0];
    for (int j = 0; j < n; j++) {
        const float x = y[(size_t)j * C + ch];
        if (ho) ho[(size_t)j * C + ch] = x;
        hist[0] = x;
        float acc = 0.0f;
#pragma unroll
        for (int l = 0; l < KC; l++) {
            float w = tp[l];
            if (dy) w += dy[((size_t)j * NKV + g) * KC + l];
            acc = fmaf(w, hist[l], acc);
        }
        float v = (x + acc) * scale;
#pragma unroll
        for (int l = KC - 1; l > 0; l--) hist[l] = hist[l - 1];
        const float n2 = block_sum(v * v, sh);
        if (isv) v *= rsqrtf(n2 / (float)HD + eps);
        else v *= fminf(qclip / (sqrtf(n2) + 1e-6f), 1.0f);
        const int pos = pos0 + j;
        const int row = ring > 0 ? pos % ring : pos;
        cache[(size_t)row * C + ch] = f2h(v);
    }
#pragma unroll
    for (int l = 1; l < KC; l++) rg[(size_t)(l - 1) * C + ch] = hist[l];
}

// The attention of row r (position cr[0] + r) and head h: every cached key
// of its window (win > 0: the last `win` positions) or its whole past,
// q·k·scale + the relative bias r·table[d] for distances d < E, the sink
// logit, the weighted values. Grid (NH, n), 256 threads (8 warps over the
// keys, online softmax a warp, merged at the end).
extern "C" __global__ void ink_attn(const float* __restrict__ yq, const float* __restrict__ yr,
                                    const float* __restrict__ sinks, const float* __restrict__ table, int E,
                                    const u16* __restrict__ kc, const u16* __restrict__ vc, int ring, int win,
                                    float scale, float* __restrict__ out, const int* __restrict__ cr)
{
    __shared__ float bias[1024];
    __shared__ float qs[HD];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int h = blockIdx.x, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int C = NKV * HD, g = h / (NH / NKV);
    const int pos = cr[0] + r;
    // the window (`win` positions, the row's own included); the ring holds
    // `win` + the call's rows, so the call's later rows never overwrite a key
    // an earlier row still reads
    const int first = win > 0 ? max(0, pos + 1 - win) : 0;
    const int nk = pos + 1 - first;
    const int near = min(E, nk);
    const float* rr = yr + ((size_t)r * NH + h) * DR;
    for (int d = tid; d < near; d += blockDim.x) {
        float b = 0.0f;
#pragma unroll
        for (int j = 0; j < DR; j++) b = fmaf(rr[j], table[(size_t)j * E + d], b);
        bias[d] = b;
    }
    for (int i = tid; i < HD; i += blockDim.x) qs[i] = yq[((size_t)r * NH + h) * HD + i];
    __syncthreads();
    constexpr int PL = HD / 32;
    float qv[PL], acc[PL];
#pragma unroll
    for (int t = 0; t < PL; t++) {
        qv[t] = qs[lane * PL + t];
        acc[t] = 0.0f;
    }
    float m = -3.0e38f, l = 0.0f;
    for (int i = w; i < nk; i += 8) {
        const int tpos = first + i;
        const int row = ring > 0 ? tpos % ring : tpos;
        const u16* kr = kc + (size_t)row * C + g * HD + lane * PL;
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < PL; t++) s = fmaf(qv[t], h2f(kr[t]), s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
        s *= scale;
        const int d = pos - tpos;
        if (d < near) s += bias[d];
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        l = l * c + e;
        const u16* vr = vc + (size_t)row * C + g * HD + lane * PL;
#pragma unroll
        for (int t = 0; t < PL; t++) acc[t] = fmaf(e, h2f(vr[t]), acc[t] * c);
        m = mn;
    }
    if (lane == 0) {
        wm[w] = m;
        wl[w] = l;
    }
#pragma unroll
    for (int t = 0; t < PL; t++) wacc[w][lane * PL + t] = acc[t];
    __syncthreads();
    // the warps and the sink (one more logit with a zero value)
    const float sk = sinks[(size_t)r * NH + h];
    float M = sk;
    for (int i = 0; i < 8; i++) M = fmaxf(M, wm[i]);
    float L = expf(sk - M);
    for (int i = 0; i < 8; i++) L += wl[i] * expf(wm[i] - M);
    for (int i = tid; i < HD; i += blockDim.x) {
        float a = 0.0f;
        for (int k = 0; k < 8; k++) a += wacc[k][i] * expf(wm[k] - M);
        out[((size_t)r * NH + h) * HD + i] = a / L;
    }
}

// `ink_attn` split over the keys, for a decode or verify call (few rows): a
// block a tile of AT keys of row r and head h, its online-softmax partial
// (max, sum, weighted values) into part[r][h][sp][HD + 2]; `ink_attn_merge`
// adds the sink and merges the tiles. Grid (NH, n, splits) with splits the
// tiles of the longest span (the window, or the context cap): a tile past the
// row's keys only marks itself empty, so a recorded call holds at any position.
#define AT 256
extern "C" __global__ void ink_attn_split(const float* __restrict__ yq, const float* __restrict__ yr,
                                          const float* __restrict__ table, int E, const u16* __restrict__ kc,
                                          const u16* __restrict__ vc, int ring, int win, float scale,
                                          float* __restrict__ part, int splits, const int* __restrict__ cr)
{
    __shared__ float bias[AT];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int h = blockIdx.x, r = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int C = NKV * HD, g = h / (NH / NKV);
    const int pos = cr[0] + r;
    const int first = win > 0 ? max(0, pos + 1 - win) : 0;
    const int nk = pos + 1 - first;
    const int k0 = sp * AT, k1 = min(nk, k0 + AT);
    float* pp = part + (((size_t)r * NH + h) * splits + sp) * (HD + 2);
    if (k0 >= nk) {
        if (tid == 0) {
            pp[0] = -3.0e38f;
            pp[1] = 0.0f;
        }
        return;
    }
    const float* rr = yr + ((size_t)r * NH + h) * DR;
    for (int i = k0 + tid; i < k1; i += blockDim.x) {
        const int d = pos - first - i;
        float b = 0.0f;
        if (d < E) {
#pragma unroll
            for (int j = 0; j < DR; j++) b = fmaf(rr[j], table[(size_t)j * E + d], b);
        }
        bias[i - k0] = b;
    }
    constexpr int PL = HD / 32;
    float qv[PL], acc[PL];
    const float* qr = yq + ((size_t)r * NH + h) * HD + lane * PL;
#pragma unroll
    for (int t = 0; t < PL; t++) {
        qv[t] = qr[t];
        acc[t] = 0.0f;
    }
    __syncthreads();
    float m = -3.0e38f, l = 0.0f;
    for (int i = k0 + w; i < k1; i += 8) {
        const int tpos = first + i;
        const int row = ring > 0 ? tpos % ring : tpos;
        const u16* kr = kc + (size_t)row * C + g * HD + lane * PL;
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < PL; t++) s = fmaf(qv[t], h2f(kr[t]), s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
        s = s * scale + bias[i - k0];
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        l = l * c + e;
        const u16* vr = vc + (size_t)row * C + g * HD + lane * PL;
#pragma unroll
        for (int t = 0; t < PL; t++) acc[t] = fmaf(e, h2f(vr[t]), acc[t] * c);
        m = mn;
    }
    if (lane == 0) {
        wm[w] = m;
        wl[w] = l;
    }
#pragma unroll
    for (int t = 0; t < PL; t++) wacc[w][lane * PL + t] = acc[t];
    __syncthreads();
    float M = -3.0e38f;
    for (int i = 0; i < 8; i++) M = fmaxf(M, wm[i]);
    if (tid == 0) {
        float L = 0.0f;
        for (int i = 0; i < 8; i++) L += wl[i] * expf(wm[i] - M);
        pp[0] = M;
        pp[1] = L;
    }
    for (int i = tid; i < HD; i += blockDim.x) {
        float a = 0.0f;
        for (int k = 0; k < 8; k++) a += wacc[k][i] * expf(wm[k] - M);
        pp[2 + i] = a;
    }
}

// The tiles of `ink_attn_split` and the sink (a logit with a zero value)
// merged into the attention output. Grid (NH, n), HD threads.
extern "C" __global__ void ink_attn_merge(const float* __restrict__ part, int splits, const float* __restrict__ sinks,
                                          int win, float* __restrict__ out, const int* __restrict__ cr)
{
    const int h = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const int pos = cr[0] + r;
    const int first = win > 0 ? max(0, pos + 1 - win) : 0;
    const int ns = min(splits, (pos + 1 - first + AT - 1) / AT);
    const float* pp = part + ((size_t)r * NH + h) * splits * (HD + 2);
    const float sk = sinks[(size_t)r * NH + h];
    float M = sk;
    for (int s = 0; s < ns; s++) M = fmaxf(M, pp[(size_t)s * (HD + 2)]);
    float L = expf(sk - M);
    for (int s = 0; s < ns; s++) L += pp[(size_t)s * (HD + 2) + 1] * expf(pp[(size_t)s * (HD + 2)] - M);
    for (int i = tid; i < HD; i += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < ns; s++) a += pp[(size_t)s * (HD + 2) + 2 + i] * expf(pp[(size_t)s * (HD + 2)] - M);
        out[((size_t)r * NH + h) * HD + i] = a / L;
    }
}

// The attention output gated by σ(gate) into the planes of the O projection.
// Grid (NH*HD/256, n).
extern "C" __global__ void ink_ogate(const float* __restrict__ o, const float* __restrict__ gt, u16* __restrict__ p0,
                                     u16* __restrict__ p1, int pstride)
{
    const int W = NH * HD;
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= W) return;
    const size_t at = (size_t)r * W + i;
    put2(p0, p1, (size_t)r * pstride + i, o[at] * sigm(gt[at]));
}

// A sub-layer's output convolution over the call's rows (a channel a
// thread, the rows in order): c = (a + conv(a)) / √(KC + 1). After the
// attention (`res` the running sum h): a block boundary closes h into
// blk[r][bslot] and the new running sum is c, else h + c. After the MoE:
// h + c. The ring [KC-1][H] moves on. Grid H/256.
extern "C" __global__ void ink_post(const float* __restrict__ a, const float* __restrict__ taps,
                                    float* __restrict__ ring, const float* __restrict__ res, float* __restrict__ blk,
                                    int bslot, int bstride, float* __restrict__ out, int n, float* __restrict__ ho)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float hist[KC];
    hist[0] = 0.0f;
#pragma unroll
    for (int l = 1; l < KC; l++) hist[l] = ring[(size_t)(l - 1) * H + i];
    const float* tp = taps + (size_t)i * KC;
    const float scale = rsqrtf((float)(KC + 1));
    for (int j = 0; j < n; j++) {
        const size_t at = (size_t)j * H + i;
        const float x = a[at];
        if (ho) ho[at] = x;
        hist[0] = x;
        float acc = 0.0f;
#pragma unroll
        for (int l = 0; l < KC; l++) acc = fmaf(tp[l], hist[l], acc);
        const float c = (x + acc) * scale;
#pragma unroll
        for (int l = KC - 1; l > 0; l--) hist[l] = hist[l - 1];
        const float hv = res[at];
        if (bslot >= 0) {
            blk[(size_t)j * bstride + (size_t)bslot * H + i] = hv;
            out[at] = c;
        } else {
            out[at] = hv + c;
        }
    }
#pragma unroll
    for (int l = 1; l < KC; l++) ring[(size_t)(l - 1) * H + i] = hist[l];
}

// ---- the prompt path (a call of more than 16 rows) ----
// The rows go in parallel: a convolution's history is its input rows, so a
// row needs no earlier row's output; the rings move on after the call.

// f16 of the two bf16 planes (the cuBLAS input of a weight GEMM). total
// elements, grid total/256.
extern "C" __global__ void ink_h16(const u16* __restrict__ p0, const u16* __restrict__ p1, u16* __restrict__ out,
                                   int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = f2h(__fadd_rn(bf2f(p0[i]), bf2f(p1[i])));
}

// `ink_kv` a row a block: grid (2·NKV, n) (keys, then values), HD threads.
// Row j's history is rows j-1 … j-KC+1 of the call, then the ring (newest
// first).
extern "C" __global__ void ink_kv_rows(const float* __restrict__ yk, const float* __restrict__ yv,
                                       const float* __restrict__ dk, const float* __restrict__ dv,
                                       const float* __restrict__ tk, const float* __restrict__ tv,
                                       const float* __restrict__ rk, const float* __restrict__ rv,
                                       u16* __restrict__ kc, u16* __restrict__ vc, int ring, float qclip, float eps,
                                       const int* __restrict__ cr)
{
    __shared__ float sh[32];
    const int g = blockIdx.x % NKV, isv = blockIdx.x / NKV, j = blockIdx.y, tid = threadIdx.x;
    const int C = NKV * HD, ch = g * HD + tid;
    const float* y = isv ? yv : yk;
    const float* dy = isv ? dv : dk;
    const float* tp = (isv ? tv : tk) + (size_t)ch * KC;
    const float* rg = isv ? rv : rk;
    u16* cache = isv ? vc : kc;
    const float scale = rsqrtf((float)((dy ? 2 : 1) * KC + 1));
    const float x = y[(size_t)j * C + ch];
    float acc = 0.0f;
#pragma unroll
    for (int l = 0; l < KC; l++) {
        const float hv = l == 0 ? x : (j - l >= 0 ? y[(size_t)(j - l) * C + ch] : rg[(size_t)(l - 1 - j) * C + ch]);
        float w = tp[l];
        if (dy) w += dy[((size_t)j * NKV + g) * KC + l];
        acc = fmaf(w, hv, acc);
    }
    float v = (x + acc) * scale;
    const float n2 = block_sum(v * v, sh);
    if (isv) v *= rsqrtf(n2 / (float)HD + eps);
    else v *= fminf(qclip / (sqrtf(n2) + 1e-6f), 1.0f);
    const int pos = cr[0] + j;
    const int row = ring > 0 ? pos % ring : pos;
    cache[(size_t)row * C + ch] = f2h(v);
}

// A convolution ring [KC-1][C] (newest first) after a call of n input rows
// y [n][C]. Grid C/256.
extern "C" __global__ void ink_ring_rows(const float* __restrict__ y, float* __restrict__ ring, int C, int n)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;
    float nv[KC];
#pragma unroll
    for (int l = 1; l < KC; l++) nv[l] = n - l >= 0 ? y[(size_t)(n - l) * C + ch] : ring[(size_t)(l - 1 - n) * C + ch];
#pragma unroll
    for (int l = 1; l < KC; l++) ring[(size_t)(l - 1) * C + ch] = nv[l];
}

// `ink_post` a row a block row: grid (H/256, n). The ring moves on after it
// (`ink_ring_rows` over `a`).
extern "C" __global__ void ink_post_rows(const float* __restrict__ a, const float* __restrict__ taps,
                                         const float* __restrict__ ring, const float* __restrict__ res,
                                         float* __restrict__ blk, int bslot, int bstride, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (i >= H) return;
    const float* tp = taps + (size_t)i * KC;
    const float scale = rsqrtf((float)(KC + 1));
    const size_t at = (size_t)j * H + i;
    const float x = a[at];
    float acc = 0.0f;
#pragma unroll
    for (int l = 0; l < KC; l++) {
        const float hv = l == 0 ? x : (j - l >= 0 ? a[(size_t)(j - l) * H + i] : ring[(size_t)(l - 1 - j) * H + i]);
        acc = fmaf(tp[l], hv, acc);
    }
    const float c = (x + acc) * scale;
    const float hv = res[at];
    if (bslot >= 0) {
        blk[(size_t)j * bstride + (size_t)bslot * H + i] = hv;
        out[at] = c;
    } else {
        out[at] = hv + c;
    }
}

// A prompt call's message gate in GEMMs (`gate_big`): the rows' normed
// halves [x / rms x, m / rms m] into xn [n][2H] (grid n, 256 threads), the
// rank-R products through cuBLAS, then the mix below.
extern "C" __global__ void ink_gate_norm(const float* __restrict__ x, const float* __restrict__ m, float eps,
                                         float* __restrict__ xn)
{
    __shared__ float sh[32];
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)r * H;
    const float* mr = m + (size_t)r * H;
    float sx = 0.0f, sm = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        sx = fmaf(xr[i], xr[i], sx);
        sm = fmaf(mr[i], mr[i], sm);
    }
    sx = block_sum(sx, sh);
    sm = block_sum(sm, sh);
    const float ix = rsqrtf(sx / (float)H + eps), im = rsqrtf(sm / (float)H + eps);
    float* o = xn + (size_t)r * 2 * H;
    for (int i = tid; i < H; i += blockDim.x) {
        o[i] = xr[i] * ix;
        o[H + i] = mr[i] * im;
    }
}

// out = (1 − tanh f)·x + tanh g·m over n·H values (f, g the F and G products).
extern "C" __global__ void ink_gate_mix(const float* __restrict__ x, const float* __restrict__ m,
                                        const float* __restrict__ f, const float* __restrict__ g,
                                        float* __restrict__ out, int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
}

// A prompt call's attention through cuBLAS (`attn_gemm`): the queries as f16
// rows, S = Q·Kᵀ and O = P·V as batched GEMMs a KV group, and between them
// `ink_attn_soft`: per (row, head) the scaled scores, the relative bias for
// distances under E, the window and the ring's positions, the sink, the
// softmax into P (f16). `ink_attn`'s arithmetic with the products on the
// tensor cores.
extern "C" __global__ void ink_f16(const float* __restrict__ x, u16* __restrict__ out, int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = f2h(x[i]);
}

// S [NH][rows][nk] f32 (rows r0 .. r0 + rows of the call), in place; P the
// same layout f16. Key slot s holds position p = s (ring 0) or the latest
// p ≤ the call's last position with p ≡ s (mod ring). Grid (rows, NH), 256.
extern "C" __global__ void ink_attn_soft(float* __restrict__ S, u16* __restrict__ P, const float* __restrict__ yr,
                                         const float* __restrict__ sinks, const float* __restrict__ table, int E,
                                         int nk, int ring, int win, float scale, int rows, int r0, int nall,
                                         const int* __restrict__ cr)
{
    __shared__ float sh[32];
    const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int row = r0 + r;
    const int pos = cr[0] + row, plast = cr[0] + nall - 1;
    const int first = win > 0 ? max(0, pos + 1 - win) : 0;
    float rr[DR];
    const float* rp = yr + ((size_t)row * NH + h) * DR;
#pragma unroll
    for (int j = 0; j < DR; j++) rr[j] = rp[j];
    float* srow = S + ((size_t)h * rows + r) * nk;
    u16* prow = P + ((size_t)h * rows + r) * nk;
    float mx = -3.0e38f;
    for (int s = tid; s < nk; s += blockDim.x) {
        int p = s;
        if (ring > 0) {
            int q = (plast - s) % ring;
            if (q < 0) q += ring;
            p = plast - q;
        }
        float v = -3.0e38f;
        if (p >= first && p <= pos) {
            v = srow[s] * scale;
            const int d = pos - p;
            if (d < E) {
                float b = 0.0f;
#pragma unroll
                for (int j = 0; j < DR; j++) b = fmaf(rr[j], table[(size_t)j * E + d], b);
                v += b;
            }
        }
        srow[s] = v;
        mx = fmaxf(mx, v);
    }
    // the block's max (a max, so the order of the reduction does not matter)
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
    __syncthreads();
    if ((tid & 31) == 0) sh[tid >> 5] = mx;
    __syncthreads();
    const float sk = sinks[(size_t)row * NH + h];
    float M = sk;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) M = fmaxf(M, sh[i]);
    __syncthreads();
    float sum = 0.0f;
    for (int s = tid; s < nk; s += blockDim.x) {
        const float v = srow[s];
        const float e = v > -1.0e38f ? expf(v - M) : 0.0f;
        srow[s] = e;
        sum += e;
    }
    const float L = block_sum(sum, sh) + expf(sk - M);
    const float il = 1.0f / L;
    for (int s = tid; s < nk; s += blockDim.x) prow[s] = f2h(srow[s] * il);
}

// ---- a prompt call's routed experts as tensor-core GEMMs over their rows ----
// The hybrid arm's job of a grouped expert (`struct HyJob` of hyb.cu, the
// same layout): the expert's repacked q4tp matrix and the places of its rows.
struct HyJob {
    u64 w;
    u64 aux0;
    u64 aux1;
    u64 y;
    u64 res;
    int rows;
    int kind;
    int blk0;
    int ystride;
    int xoff;
    int nr;
    float lim;
    int pad0, pad1, pad2;
};

// The f16 rows the expert GEMMs read, in their places: out[q] = x[src[q]].
// Grid (total / 256), total = places · H.
extern "C" __global__ void ink_gather16(const float* __restrict__ x, const int* __restrict__ src, int total,
                                        u16* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int q = i / H, c = i - q * H;
    out[i] = f2h(x[(size_t)src[q] * H + c]);
}

// A register a_m of a repacked q4tp word as f16x2 (nibble − 8): 1024 + n is
// 0x6400 | n; one fma takes 1032 off both halves.
__device__ __forceinline__ u32 a_h16(u32 X, int m) {
    u32 h;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(h) : "r"(X >> (4 * m)), "r"(0x000F000Fu), "r"(0x64006400u));
    u32 r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h), "r"(0x3C003C00u), "r"(0xE408E408u));
    return r;
}

__device__ __forceinline__ void mma16816h(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One expert job (`hy_moe_group`'s, up to 64 rows at places xoff / cols + j)
// against 128 output rows of its q4tp matrix: 4 warps × MT 16-row tiles, the
// job's rows NTL n tiles of 8. A group of 32 columns is two mma steps into a
// fresh sum, scaled by the row's ladder rung and added (the codec's
// arithmetic; the inputs f16, the sums f32). pair: the gate/up pair (a tile's
// rows 8 of gate, then the same 8 of up), the softcapped SwiGLU into act
// [place][cols_out] f16; else the down product into ys [place][rows] f32.
// Grid (ceil(tiles / (4 MT)), jobs), 128 threads.
#define MT 2
#define NTL 8
extern "C" __global__ void ink_moe_mma(const HyJob* __restrict__ jobs, const u16* __restrict__ x, int cols, int pair,
                                       u16* __restrict__ act, int act_stride, float* __restrict__ ys, int ys_stride)
{
    const HyJob jb = jobs[blockIdx.y];
    if (jb.rows <= 0 || jb.nr <= 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
    const int ntile = (jb.rows + 15) >> 4;
    const int tbase = ((int)blockIdx.x * 4 + warp) * MT;
    if (tbase >= ntile) return;
    const int gpr = cols >> 5;
    const int place0 = jb.xoff / cols, nr = jb.nr;
    const uint2* T = (const uint2*)jb.w;
    const unsigned char* Cc = (const unsigned char*)jb.aux0;
    const float2* par = (const float2*)jb.aux1;
    float2 pa[MT], pb[MT];
#pragma unroll
    for (int b = 0; b < MT; b++) {
        const int tile = min(tbase + b, ntile - 1);
        pa[b] = par[tile * 16 + gid];
        pb[b] = par[tile * 16 + gid + 8];
    }
    float acc[MT][NTL][4];
#pragma unroll
    for (int b = 0; b < MT; b++)
#pragma unroll
        for (int t = 0; t < NTL; t++) acc[b][t][0] = acc[b][t][1] = acc[b][t][2] = acc[b][t][3] = 0.0f;
    for (int g = 0; g < gpr; g++) {
        u32 bx[NTL][4];
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NTL; t++) {
            const int j = t * 8 + gid;
            if (j < nr) {
                const u16* xp = x + (size_t)(place0 + j) * cols + k0;
#pragma unroll
                for (int q = 0; q < 4; q++) bx[t][q] = *(const u32*)(xp + 8 * q);
            } else {
#pragma unroll
                for (int q = 0; q < 4; q++) bx[t][q] = 0u;
            }
        }
#pragma unroll
        for (int b = 0; b < MT; b++) {
            const int tile = tbase + b;
            if (tile >= ntile) break;
            const size_t tg = (size_t)tile * gpr + g;
            const uint2 X = T[tg * 32 + lane];
            const float sa = exp2f(fmaf((float)Cc[tg * 16 + gid], pa[b].y, pa[b].x));
            const float sb = exp2f(fmaf((float)Cc[tg * 16 + gid + 8], pb[b].y, pb[b].x));
            const u32 a00 = a_h16(X.x, 0), a01 = a_h16(X.x, 1), a02 = a_h16(X.x, 2), a03 = a_h16(X.x, 3);
            const u32 a10 = a_h16(X.y, 0), a11 = a_h16(X.y, 1), a12 = a_h16(X.y, 2), a13 = a_h16(X.y, 3);
#pragma unroll
            for (int t = 0; t < NTL; t++) {
                if (t * 8 >= nr) break;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma16816h(c, a00, a01, a02, a03, bx[t][0], bx[t][1]);
                mma16816h(c, a10, a11, a12, a13, bx[t][2], bx[t][3]);
                acc[b][t][0] = fmaf(sa, c[0], acc[b][t][0]);
                acc[b][t][1] = fmaf(sa, c[1], acc[b][t][1]);
                acc[b][t][2] = fmaf(sb, c[2], acc[b][t][2]);
                acc[b][t][3] = fmaf(sb, c[3], acc[b][t][3]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < MT; b++) {
        const int tile = tbase + b;
        if (tile >= ntile) break;
#pragma unroll
        for (int t = 0; t < NTL; t++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int j = t * 8 + 2 * tig + e;
                if (j >= nr) continue;
                const size_t pl = (size_t)(place0 + j);
                if (pair) {
                    // Inkling's softcapped SwiGLU (the cap in `lim`, the up branch's in pad0)
                    const float gv = acc[b][t][e], uv = acc[b][t][2 + e];
                    const float cu = __int_as_float(jb.pad0);
                    const float v = (jb.lim * tanhf(gv / jb.lim)) * (1.0f / (1.0f + expf(-gv))) * (cu * tanhf(uv / cu));
                    act[pl * act_stride + tile * 8 + gid] = f2h(v);
                } else {
                    ys[pl * ys_stride + tile * 16 + gid] = acc[b][t][e];
                    ys[pl * ys_stride + tile * 16 + gid + 8] = acc[b][t][2 + e];
                }
            }
        }
    }
}

// `ink_moe_mma` with the job's rows staged once a block: the 32-column slice of
// its (up to 64) f16 rows goes into shared memory through a three-stage
// cp.async pipeline (one copy for the four warps, not one a warp, and the
// next slices in flight while this one multiplies), each warp's B fragments
// read from there (rows padded to 40 halves: conflict-free), its weights one
// group ahead in registers. Same jobs, grid, arithmetic and outputs.
#define MS 3
#define XROW 40
__device__ __forceinline__ void cp16(void* dst, const void* src, bool full)
{
#if __CUDA_ARCH__ >= 800
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = full ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(n));
#else
    *(uint4*)dst = full ? *(const uint4*)src : make_uint4(0u, 0u, 0u, 0u);
#endif
}
__device__ __forceinline__ void cp_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
template <int N> __device__ __forceinline__ void cp_wait()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}
extern "C" __global__ void __launch_bounds__(128) ink_moe_mma2(const HyJob* __restrict__ jobs, const u16* __restrict__ x, int cols, int pair,
                                                             u16* __restrict__ act, int act_stride, float* __restrict__ ys, int ys_stride)
{
    __shared__ __align__(16) u16 xs[MS][NTL * 8][XROW];
    const HyJob jb = jobs[blockIdx.y];
    if (jb.rows <= 0 || jb.nr <= 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
    const int ntile = (jb.rows + 15) >> 4;
    const int tbase = ((int)blockIdx.x * 4 + warp) * MT;
    // the block's first tile past the matrix: nothing to do (every warp's tiles beyond)
    if ((int)blockIdx.x * 4 * MT >= ntile) return;
    const int gpr = cols >> 5;
    const int place0 = jb.xoff / cols, nr = jb.nr;
    const uint2* T = (const uint2*)jb.w;
    const unsigned char* Cc = (const unsigned char*)jb.aux0;
    const float2* par = (const float2*)jb.aux1;
    // the slice copy: 64 rows × 4 chunks of 16 bytes, two a thread
    auto stage = [&](int g, int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = threadIdx.x + i * 128, row = c >> 2, ch = c & 3;
            const bool in = row < nr;
            const u16* src = x + (size_t)(place0 + (in ? row : 0)) * cols + g * 32 + ch * 8;
            cp16(&xs[buf][row][ch * 8], src, in);
        }
    };
#pragma unroll
    for (int s = 0; s < MS - 1; s++) {
        if (s < gpr) stage(s, s);
        cp_commit();
    }
    const bool live = tbase < ntile;
    float2 pa[MT], pb[MT];
#pragma unroll
    for (int b = 0; b < MT; b++) {
        const int tile = min(tbase + b, ntile - 1);
        pa[b] = par[tile * 16 + gid];
        pb[b] = par[tile * 16 + gid + 8];
    }
    float acc[MT][NTL][4];
#pragma unroll
    for (int b = 0; b < MT; b++)
#pragma unroll
        for (int t = 0; t < NTL; t++) acc[b][t][0] = acc[b][t][1] = acc[b][t][2] = acc[b][t][3] = 0.0f;
    // this warp's weights and codes one group ahead
    uint2 Xn[MT];
    unsigned char ca[MT], cb[MT];
#pragma unroll
    for (int b = 0; b < MT; b++) {
        const int tile = min(tbase + b, ntile - 1);
        const size_t tg = (size_t)tile * gpr;
        Xn[b] = T[tg * 32 + lane];
        ca[b] = Cc[tg * 16 + gid];
        cb[b] = Cc[tg * 16 + gid + 8];
    }
    const int ntl = min(NTL, (nr + 7) >> 3);
    for (int g = 0; g < gpr; g++) {
        if (g + MS - 1 < gpr) stage(g + MS - 1, (g + MS - 1) % MS);
        cp_commit();
        cp_wait<MS - 1>();
        __syncthreads();
        uint2 X[MT];
        float sa[MT], sb[MT];
#pragma unroll
        for (int b = 0; b < MT; b++) {
            X[b] = Xn[b];
            sa[b] = exp2f(fmaf((float)ca[b], pa[b].y, pa[b].x));
            sb[b] = exp2f(fmaf((float)cb[b], pb[b].y, pb[b].x));
        }
        if (g + 1 < gpr) {
#pragma unroll
            for (int b = 0; b < MT; b++) {
                const int tile = min(tbase + b, ntile - 1);
                const size_t tg = (size_t)tile * gpr + g + 1;
                Xn[b] = T[tg * 32 + lane];
                ca[b] = Cc[tg * 16 + gid];
                cb[b] = Cc[tg * 16 + gid + 8];
            }
        }
        if (live) {
            const int buf = g % MS;
#pragma unroll
            for (int t = 0; t < NTL; t++) {
                if (t >= ntl) break;
                const u16* xr = &xs[buf][t * 8 + gid][2 * tig];
                const u32 b0 = *(const u32*)(xr), b1 = *(const u32*)(xr + 8), b2 = *(const u32*)(xr + 16), b3 = *(const u32*)(xr + 24);
#pragma unroll
                for (int b = 0; b < MT; b++) {
                    if (tbase + b >= ntile) break;
                    const u32 a00 = a_h16(X[b].x, 0), a01 = a_h16(X[b].x, 1), a02 = a_h16(X[b].x, 2), a03 = a_h16(X[b].x, 3);
                    const u32 a10 = a_h16(X[b].y, 0), a11 = a_h16(X[b].y, 1), a12 = a_h16(X[b].y, 2), a13 = a_h16(X[b].y, 3);
                    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma16816h(c, a00, a01, a02, a03, b0, b1);
                    mma16816h(c, a10, a11, a12, a13, b2, b3);
                    acc[b][t][0] = fmaf(sa[b], c[0], acc[b][t][0]);
                    acc[b][t][1] = fmaf(sa[b], c[1], acc[b][t][1]);
                    acc[b][t][2] = fmaf(sb[b], c[2], acc[b][t][2]);
                    acc[b][t][3] = fmaf(sb[b], c[3], acc[b][t][3]);
                }
            }
        }
        __syncthreads();
    }
    cp_wait<0>();
    if (!live) return;
#pragma unroll
    for (int b = 0; b < MT; b++) {
        const int tile = tbase + b;
        if (tile >= ntile) break;
#pragma unroll
        for (int t = 0; t < NTL; t++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int j = t * 8 + 2 * tig + e;
                if (j >= nr) continue;
                const size_t pl = (size_t)(place0 + j);
                if (pair) {
                    const float gv = acc[b][t][e], uv = acc[b][t][2 + e];
                    const float cu = __int_as_float(jb.pad0);
                    const float v = (jb.lim * tanhf(gv / jb.lim)) * (1.0f / (1.0f + expf(-gv))) * (cu * tanhf(uv / cu));
                    act[pl * act_stride + tile * 8 + gid] = f2h(v);
                } else {
                    ys[pl * ys_stride + tile * 16 + gid] = acc[b][t][e];
                    ys[pl * ys_stride + tile * 16 + gid + 8] = acc[b][t][2 + e];
                }
            }
        }
    }
}

// ---- a draft verify's rollback: the convolution rings of every application ----
// A table entry per ring: {the live ring, its snapshot, the call's input rows
// (the small-path kernels' history), the width}. Snapshot before a verify;
// after it, the ring as it stands after the first r rows: ring[l-1] = the row
// r-l when r-l >= 0, else the snapshot's l-1-r. Grid (ceil(maxw/256), rings).
struct RingRef {
    u64 live, snap, hist;
    int width, pad;
};
extern "C" __global__ void ink_ring_snap(const RingRef* __restrict__ t)
{
    const RingRef e = t[blockIdx.y];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e.width) return;
#pragma unroll
    for (int l = 0; l < KC - 1; l++) ((float*)e.snap)[(size_t)l * e.width + i] = ((const float*)e.live)[(size_t)l * e.width + i];
}
extern "C" __global__ void ink_ring_back(const RingRef* __restrict__ t, int r)
{
    const RingRef e = t[blockIdx.y];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e.width) return;
    const float* hs = (const float*)e.hist;
    const float* sn = (const float*)e.snap;
    float* lv = (float*)e.live;
#pragma unroll
    for (int l = 1; l < KC; l++)
        lv[(size_t)(l - 1) * e.width + i] = r - l >= 0 ? hs[(size_t)(r - l) * e.width + i] : sn[(size_t)(l - 1 - r) * e.width + i];
}

// ---- the DSpark drafter ----
// NeoX rotation of a head vector held one value a thread (HD threads): the
// pairs (i, i + HD/2) at angle pos · theta^(−2i/HD).
__device__ __forceinline__ float rope_neox(float v, float* sh_v, int i, int pos, float theta)
{
    __syncthreads();
    sh_v[i] = v;
    __syncthreads();
    const int half = HD / 2, j = i < half ? i : i - half;
    const float ang = (float)pos * powf(theta, -2.0f * (float)j / (float)HD);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    return i < half ? v * cs - sh_v[i + half] * sn : v * cs + sh_v[i - half] * sn;
}

// The mixes of a call's taps: mix j of row r = rms(Σ_k alpha[j][k] tap[k][r]) ·
// norms[j]; mixes j < nmix − 1 into planes (p0, p1 [j][rows][H]) and f16 rows
// (h16 [j][rows][H]), the last (the Markov head's context) into the ring row
// fr[(pos % fr_cap)]. Grid (n, nmix), 256 threads.
extern "C" __global__ void ink_ds_mix(const float* __restrict__ tap, int ntap, int rows, const float* __restrict__ alpha,
                                      const float* __restrict__ norms, float eps, int nmix, u16* __restrict__ p0,
                                      u16* __restrict__ p1, u16* __restrict__ h16, float* __restrict__ fr, int fr_cap,
                                      const int* __restrict__ cr)
{
    __shared__ float sh[32];
    __shared__ float mixv[H];
    const int r = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < ntap; k++) v = fmaf(alpha[j * ntap + k], tap[((size_t)k * rows + r) * H + i], v);
        mixv[i] = v;
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)H + eps);
    const float* nw = norms + (size_t)j * H;
    if (j == nmix - 1) {
        // a call of more rows than the ring: only its last fr_cap rows land
        // (rows fr_cap apart would race for one cell)
        if (r + fr_cap < (int)gridDim.x) return;
        float* o = fr + (size_t)((cr[0] + r) % fr_cap) * H;
        for (int i = tid; i < H; i += blockDim.x) o[i] = mixv[i] * inv * nw[i];
        return;
    }
    const size_t base = ((size_t)j * rows + r) * H;
    for (int i = tid; i < H; i += blockDim.x) {
        const float v = mixv[i] * inv * nw[i];
        put2(p0, p1, base + i, v);
        h16[base + i] = f2h(v);
    }
}

// A call's context keys and values into a drafter layer's cache: k = rope(rms_head(k_ctx · mix) ·
// knorm, pos), v = v_ctx · mix, f16 at cell pos (a ring: pos % cap). Grid (n, NKV), HD threads.
extern "C" __global__ void ink_ds_ctx(const float* __restrict__ yk, const float* __restrict__ yv,
                                      const float* __restrict__ knorm, float eps, float theta, u16* __restrict__ kc,
                                      u16* __restrict__ vc, int cap, int ring, const int* __restrict__ cr)
{
    __shared__ float sh[32];
    __shared__ float sv[HD];
    const int r = blockIdx.x, g = blockIdx.y, i = threadIdx.x;
    const int C = NKV * HD;
    const int pos = cr[0] + r;
    const size_t at = (size_t)r * C + g * HD + i;
    const float k = yk[at];
    const float n2 = block_sum(k * k, sh);
    const float kn = k * rsqrtf(n2 / (float)HD + eps) * knorm[i];
    const float kr = rope_neox(kn, sv, i, pos, theta);
    const int cell = ring ? pos % cap : pos;
    kc[(size_t)cell * C + g * HD + i] = f2h(kr);
    vc[(size_t)cell * C + g * HD + i] = f2h(yv[at]);
}

// The block's queries (heads < NH) normed and rotated in place, its keys
// normed, rotated and with the values written at cells pos0 + r. Grid (B, NH + NKV), HD threads.
extern "C" __global__ void ink_ds_qk(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                     const float* __restrict__ qnorm, const float* __restrict__ knorm, float eps,
                                     float theta, u16* __restrict__ kc, u16* __restrict__ vc, int cap, int ring,
                                     const int* __restrict__ dcr)
{
    __shared__ float sh[32];
    __shared__ float sv[HD];
    const int r = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int pos = dcr[0] + r;
    if (h < NH) {
        const size_t at = ((size_t)r * NH + h) * HD + i;
        const float x = q[at];
        const float n2 = block_sum(x * x, sh);
        q[at] = rope_neox(x * rsqrtf(n2 / (float)HD + eps) * qnorm[i], sv, i, pos, theta);
        return;
    }
    const int g = h - NH, C = NKV * HD;
    const size_t at = (size_t)r * C + g * HD + i;
    const float x = k[at];
    const float n2 = block_sum(x * x, sh);
    const float kr = rope_neox(x * rsqrtf(n2 / (float)HD + eps) * knorm[i], sv, i, pos, theta);
    const int cell = ring ? pos % cap : pos;
    kc[(size_t)cell * C + g * HD + i] = f2h(kr);
    vc[(size_t)cell * C + g * HD + i] = f2h(v[at]);
}

// The block's attention: row r (position pos0 + r) and head h over the cells of
// positions [lo, pos0 + B) (lo = pos0 − window for a sliding layer, else 0):
// the context and every block row; q·k / √HD, softmax, the weighted values.
// Grid (NH, B), 256 threads (8 warps over the keys, online softmax a warp).
extern "C" __global__ void ink_ds_attn(const float* __restrict__ q, const u16* __restrict__ kc,
                                       const u16* __restrict__ vc, int cap, int ring, int window, int B,
                                       float* __restrict__ out, const int* __restrict__ dcr)
{
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int h = blockIdx.x, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int C = NKV * HD, g = h / (NH / NKV);
    const int pos0 = dcr[0];
    const int lo = window > 0 ? max(0, pos0 - window) : 0, hi = pos0 + B;
    constexpr int PL = HD / 32;
    float qv[PL], acc[PL];
    const float* qr = q + ((size_t)r * NH + h) * HD + lane * PL;
#pragma unroll
    for (int t = 0; t < PL; t++) {
        qv[t] = qr[t];
        acc[t] = 0.0f;
    }
    const float scale = rsqrtf((float)HD);
    float m = -3.0e38f, l = 0.0f;
    for (int p = lo + w; p < hi; p += 8) {
        const int cell = ring ? p % cap : p;
        const u16* kr = kc + (size_t)cell * C + g * HD + lane * PL;
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < PL; t++) s = fmaf(qv[t], h2f(kr[t]), s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
        s *= scale;
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        l = l * c + e;
        const u16* vr = vc + (size_t)cell * C + g * HD + lane * PL;
#pragma unroll
        for (int t = 0; t < PL; t++) acc[t] = fmaf(e, h2f(vr[t]), acc[t] * c);
        m = mn;
    }
    if (lane == 0) {
        wm[w] = m;
        wl[w] = l;
    }
#pragma unroll
    for (int t = 0; t < PL; t++) wacc[w][lane * PL + t] = acc[t];
    __syncthreads();
    float M = -3.0e38f;
    for (int i = 0; i < 8; i++) M = fmaxf(M, wm[i]);
    float L = 0.0f;
    for (int i = 0; i < 8; i++) L += wl[i] * expf(wm[i] - M);
    for (int i = tid; i < HD; i += blockDim.x) {
        float a = 0.0f;
        for (int k = 0; k < 8; k++) a += wacc[k][i] * expf(wm[k] - M);
        out[((size_t)r * NH + h) * HD + i] = a / L;
    }
}

// f32 rows into the tensor-core planes. Grid (total / 256).
extern "C" __global__ void ink_planes(const float* __restrict__ x, u16* __restrict__ p0, u16* __restrict__ p1, int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) put2(p0, p1, i, x[i]);
}

// One step of the Markov head's recurrence (one block, 3R threads): raw = c_h + c_f + b + W [state ; w1[prev]]
// (W the first 2R columns of the rnn projection, row stride ws); gate = σ(raw[0:R]), state = gate·state +
// (1 − gate)·tanh(raw[R:2R]), latent = tanh(raw[2R:3R]) as planes.
extern "C" __global__ void ink_ds_markov(const float* __restrict__ ch, const float* __restrict__ cf,
                                         const float* __restrict__ b, const float* __restrict__ W, int ws, int R,
                                         float* __restrict__ state, const int* __restrict__ prev,
                                         const u16* __restrict__ w1, u16* __restrict__ lp0, u16* __restrict__ lp1)
{
    extern __shared__ float smem[];
    float* in = smem;          // [2R]: state, w1[prev]
    float* raw = smem + 2 * R; // [3R]
    const int o = threadIdx.x;
    const int pv = prev[0];
    for (int j = o; j < R; j += blockDim.x) {
        in[j] = state[j];
        in[R + j] = h2f(w1[(size_t)pv * R + j]);
    }
    __syncthreads();
    if (o < 3 * R) {
        const float* wr = W + (size_t)o * ws;
        float a = ch[o] + cf[o] + b[o];
        for (int j = 0; j < 2 * R; j++) a = fmaf(wr[j], in[j], a);
        raw[o] = a;
    }
    __syncthreads();
    if (o < R) {
        const float g = 1.0f / (1.0f + expf(-raw[o]));
        state[o] = g * in[o] + (1.0f - g) * tanhf(raw[R + o]);
        put2(lp0, lp1, o, tanhf(raw[2 * R + o]));
    }
}

// A draft's pick: the first argmax of logits + fold_b (the softcap keeps the
// order) into ids[i] and *prev. One block, 1024 threads.
extern "C" __global__ void ink_ds_pick(const float* __restrict__ lg, const float* __restrict__ fb, int V, int i,
                                       int* __restrict__ ids, int* __restrict__ prev)
{
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    float best = -3.0e38f;
    int idx = 0x7fffffff;
    for (int v = tid; v < V; v += blockDim.x) {
        const float x = lg[v] + fb[v];
        if (x > best) {
            best = x;
            idx = v;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(FULL, best, o);
        const int oi = __shfl_xor_sync(FULL, idx, o);
        if (ob > best || (ob == best && oi < idx)) {
            best = ob;
            idx = oi;
        }
    }
    if (lane == 0) {
        bv[w] = best;
        bi[w] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        float b = bv[0];
        int x = bi[0];
        for (int k = 1; k < (int)(blockDim.x >> 5); k++)
            if (bv[k] > b || (bv[k] == b && bi[k] < x)) {
                b = bv[k];
                x = bi[k];
            }
        ids[i] = x;
        prev[0] = x;
    }
}

// The head's logits capped: cap · tanh(x / cap). Grid (V/256, n).
extern "C" __global__ void ink_softcap(float* __restrict__ lg, int V, float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= V) return;
    const size_t at = (size_t)r * V + i;
    lg[at] = tanhf(lg[at] / cap) * cap;
}

// The greedy pick of each row of `lg` ([rows][V], a block a row): the first
// index of the largest value, as the host's argmax takes it.
extern "C" __global__ void ink_argmax_rows(const float* __restrict__ lg, int V, int* __restrict__ ids)
{
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* r = lg + (size_t)blockIdx.x * V;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    float best = -3.0e38f;
    int idx = 0x7fffffff;
    for (int v = tid; v < V; v += blockDim.x) {
        const float x = r[v];
        if (x > best) {
            best = x;
            idx = v;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(FULL, best, o);
        const int oi = __shfl_xor_sync(FULL, idx, o);
        if (ob > best || (ob == best && oi < idx)) {
            best = ob;
            idx = oi;
        }
    }
    if (lane == 0) {
        bv[w] = best;
        bi[w] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        float b = bv[0];
        int x = bi[0];
        for (int j = 1; j < (int)(blockDim.x >> 5); j++) {
            if (bv[j] > b || (bv[j] == b && bi[j] < x)) {
                b = bv[j];
                x = bi[j];
            }
        }
        ids[blockIdx.x] = x == 0x7fffffff ? 0 : x;
    }
}
