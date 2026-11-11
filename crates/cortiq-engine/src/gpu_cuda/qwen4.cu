// Qwen3.8-Flash-Next (qwen4_exp) on CUDA: the kernels between the cuBLAS
// GEMMs of a whole chunk of tokens. Every buffer is [token][...] row-major;
// a GEMM operand is "wt": f16 when `half` is set (tensor cores), f32 otherwise.
// Compiled by NVRTC as its own module, so it carries its own helpers.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;

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
// Store / load one element of a wt buffer.
__device__ __forceinline__ void stf(void* p, u64 i, float v, int half) {
    if (half) ((u16*)p)[i] = f2h_sat(v); else ((float*)p)[i] = v;
}
__device__ __forceinline__ float ldf(const void* p, u64 i, int half) {
    return half ? h2f(((const u16*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x * sigm(x); }

// hyper [n][hc][hid] = the embedding row repeated over the streams.
extern "C" __global__ void q4_tile(const float* __restrict__ emb, float* __restrict__ hyper, int n, int hid, int hc)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    const float v = emb[i];
    for (int s = 0; s < hc; s++) hyper[(t * hc + s) * hid + d] = v;
}

// Zero-centred group RMS norm, a block per (token, group):
// y = x * rsqrt(mean(x^2) + eps) * (1 + w). f32 into y and/or wt into yh
// (either may be null).
extern "C" __global__ void q4_grms(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                   void* __restrict__ yh, int group, int groups, double eps, int half)
{
    __shared__ double red[256];
    const u64 b = blockIdx.x;
    const int g = (int)(b % groups);
    const float* xr = x + b * group;
    double ss = 0.0;
    for (int i = threadIdx.x; i < group; i += blockDim.x) { const double v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(red[0] / (double)group + eps));
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = xr[i] * inv * (1.0f + w[g * group + i]);
        if (y) y[b * group + i] = v;
        if (yh) stf(yh, b * group + i, v, half);
    }
}

// The low-rank middle of a gated residual. low [n][R]: the first `lr` values
// become the wt operand silu(low / hc); the rest are the inject weights
// 2 * sigmoid(low / hc) -> iw [n][R - lr].
extern "C" __global__ void q4_hc_low(const float* __restrict__ low, void* __restrict__ lowh, float* __restrict__ iw,
                                     int n, int R, int lr, float hc, int half)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * R) return;
    const u64 t = i / R;
    const int c = (int)(i - t * R);
    const float v = low[i] / hc;
    if (c < lr) stf(lowh, t * lr + c, silu_f(v), half);
    else iw[t * (R - lr) + (c - lr)] = 2.0f * sigm(v);
}

// folded [n][hid] = sum over streams of sigmoid(mix) * normed / hc.
extern "C" __global__ void q4_hc_fold(const float* __restrict__ mix, const float* __restrict__ normed,
                                      float* __restrict__ folded, void* __restrict__ foldedh,
                                      int n, int hid, int hc, int half)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    float acc = 0.0f;
    for (int s = 0; s < hc; s++) {
        const u64 o = (t * hc + s) * hid + d;
        acc += sigm(mix[o]) * normed[o] / (float)hc;
    }
    folded[i] = acc;
    if (foldedh) stf(foldedh, i, acc, half);
}

// hyper [n][hc][hid] += iw [n][hc] * block [n][hid].
extern "C" __global__ void q4_inject(float* __restrict__ hyper, const float* __restrict__ block,
                                     const float* __restrict__ iw, int n, int hid, int hc)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hc * hid) return;
    const u64 t = i / ((u64)hc * hid);
    const u64 r = i - t * hc * hid;
    const int s = (int)(r / hid);
    hyper[i] += iw[t * hc + s] * block[t * hid + (r - (u64)s * hid)];
}

// A right-aligned history of `cap` rows after a chunk of n rows:
// out[r] = the row at chunk position n - cap + r, older ones from `hist`.
extern "C" __global__ void q4_hist(const float* __restrict__ hist, const float* __restrict__ chunk,
                                   float* __restrict__ out, int cap, int width, int n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)cap * width) return;
    const int r = (int)(i / width);
    const u64 c = i - (u64)r * width;
    const int src = n - cap + r;
    out[i] = src >= 0 ? chunk[(u64)src * width + c] : hist[(u64)(cap + src) * width + c];
}

// ---------------------------------------------------------------- GDN

// Causal depthwise conv over the chunk, then SiLU, in f64 as the host does:
// cq[t][c] = silu(qkv[t][c] * taps[c][kk-1] + sum_j row(t - (kk-1-j))[c] * taps[c][j]),
// rows before the chunk coming from the ring [(kk-1)][c_dim] (newest last).
extern "C" __global__ void q4_gdn_conv(const float* __restrict__ qkv, const float* __restrict__ ring,
                                       const float* __restrict__ taps, float* __restrict__ cq,
                                       int n, int c_dim, int kk)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * c_dim) return;
    const int t = (int)(i / c_dim);
    const u64 c = i - (u64)t * c_dim;
    const float* tp = taps + c * kk;
    double acc = (double)qkv[i] * (double)tp[kk - 1];
    for (int j = 0; j < kk - 1; j++) {
        const int src = t - (kk - 1 - j);
        const float v = src >= 0 ? qkv[(u64)src * c_dim + c] : ring[(u64)(kk - 1 + src) * c_dim + c];
        acc += (double)v * (double)tp[j];
    }
    cq[i] = (float)(acc / (1.0 + exp(-acc)));
}

// L2 norm of the q and k heads in place, a block per (token, head of 2 * nk):
// q heads also divided by sqrt(dk).
extern "C" __global__ void q4_gdn_prep(float* __restrict__ cq, int c_dim, int nk, int dk)
{
    __shared__ double red[256];
    const u64 t = blockIdx.x / (2 * nk);
    const int g = (int)(blockIdx.x - t * (2 * nk));
    float* xr = cq + t * c_dim + (u64)g * dk;
    double ss = 0.0;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) { const double v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double nrm = sqrt(red[0] + 1e-6);
    const float inv = (float)(1.0 / (g < nk ? nrm * sqrt((double)dk) : nrm));
    for (int i = threadIdx.x; i < dk; i += blockDim.x) xr[i] *= inv;
}

// Decay and write gates of every (token, value head):
// gb[t][h] = exp(-exp(a_log) * softplus(a + dt_bias)), gb[t][nv + h] = sigmoid(b).
// ab [n][2 * nv] holds a then b.
extern "C" __global__ void q4_gdn_gb(const float* __restrict__ ab, const float* __restrict__ a_log,
                                     const float* __restrict__ dt_bias, float* __restrict__ gb, int n, int nv)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nv) return;
    const u64 t = i / nv;
    const int h = (int)(i - t * nv);
    const double x = (double)ab[t * 2 * nv + h] + (double)dt_bias[h];
    const double sp = x > 20.0 ? x : log1p(exp(x));
    gb[t * 2 * nv + h] = (float)exp(-exp((double)a_log[h]) * sp);
    gb[t * 2 * nv + nv + h] = (float)(1.0 / (1.0 + exp(-(double)ab[t * 2 * nv + nv + h])));
}

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The delta-rule scan over the chunk. A warp owns one column (value head h,
// value dim dj) of the state S [nv][dk][dv]; a lane keeps 4 of its dk = 128
// cells in registers. Per token: kv = S . k, delta = (v - g kv) beta,
// S = g S + k delta, o = q . S. Blocks of 256 threads = 8 columns.
extern "C" __global__ void q4_gdn_scan(const float* __restrict__ cq, const float* __restrict__ gb,
                                       const float* S, float* So, float* __restrict__ o,
                                       int n, int c_dim, int nv, int nk, int dk, int dv)
{
    const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (col >= nv * dv) return;
    const int lane = threadIdx.x & 31;
    const int h = col / dv, dj = col - h * dv;
    const int ko = h / (nv / nk);
    const int kd = nk * dk;
    const u64 qo = (u64)ko * dk + lane * 4, kofs = (u64)kd + qo, vo = (u64)2 * kd + (u64)h * dv + dj;
    const u64 so = ((u64)h * dk + lane * 4) * dv + dj;
    const float* sp = S + so;
    float s0 = sp[0], s1 = sp[dv], s2 = sp[2 * dv], s3 = sp[3 * dv];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (u64)t * c_dim;
        const float4 kf = *(const float4*)(row + kofs);
        const float4 qf = *(const float4*)(row + qo);
        const float g = gb[(u64)t * 2 * nv + h], beta = gb[(u64)t * 2 * nv + nv + h];
        const float kv = wsum(s0 * kf.x + s1 * kf.y + s2 * kf.z + s3 * kf.w);
        const float delta = (row[vo] - g * kv) * beta;
        s0 = g * s0 + kf.x * delta;
        s1 = g * s1 + kf.y * delta;
        s2 = g * s2 + kf.z * delta;
        s3 = g * s3 + kf.w * delta;
        const float ov = wsum(qf.x * s0 + qf.y * s1 + qf.z * s2 + qf.w * s3);
        if (lane == 0) o[(u64)t * nv * dv + (u64)h * dv + dj] = ov;
    }
    float* wp = So + so;
    wp[0] = s0; wp[dv] = s1; wp[2 * dv] = s2; wp[3 * dv] = s3;
}

// of [n][nv][dv] (wt) = rms(o) * norm * sigmoid(z), a block per (token, head), f64.
extern "C" __global__ void q4_gdn_out(const float* __restrict__ o, const float* __restrict__ z,
                                      const float* __restrict__ norm, void* __restrict__ ofh,
                                      int dv, double eps, int half)
{
    __shared__ double red[256];
    const u64 b = blockIdx.x;
    const float* xr = o + b * dv;
    double ss = 0.0;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) { const double v = xr[i]; ss += v * v; }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double inv = 1.0 / sqrt(red[0] / (double)dv + eps);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const double gate = 1.0 / (1.0 + exp(-(double)z[b * dv + i]));
        stf(ofh, b * dv + i, (float)(((double)xr[i] * inv) * (double)norm[i] * gate), half);
    }
}

// ---------------------------------------------------------------- QSA

// Head-wise zero-centred RMS norm, then rotary, a block per (item, head).
// The value is the mean of `nsum` consecutive source rows (1 = the row
// itself; the indexer's block keys average `compress_ratio` raw keys):
//   v[d] = sum_j src[(item * nsum + j) * sstride + head * shead + d] / nsum
//   v[d] *= rsqrt(mean(v^2) + eps) * (1 + w[d]);  rotate the first `rot` dims at
//   position pos0 + item * pstep;
//   dst[item * dstride + (head / dgroup) * dgstride + (head % dgroup) * dhead + d] = v[d].
extern "C" __global__ void q4_norm_rope(const float* __restrict__ src, float* __restrict__ dst,
                                        const float* __restrict__ w, const float* __restrict__ invf,
                                        int heads, int hd, u64 sstride, u64 shead, u64 dstride, u64 dhead,
                                        int dgroup, u64 dgstride, int rot, int nfreq, int pos0, int pstep,
                                        int nsum, double eps)
{
    __shared__ float sv[256];
    __shared__ double red[256];
    const u64 item = blockIdx.x / heads;
    const int head = (int)(blockIdx.x - item * heads);
    const int d = threadIdx.x;
    float v = 0.0f;
    if (d < hd) {
        for (int j = 0; j < nsum; j++) v += src[(item * nsum + j) * sstride + (u64)head * shead + d] / (float)nsum;
    }
    red[d] = (double)v * (double)v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(red[0] / (double)hd + eps));
    if (d < hd) sv[d] = v * (inv * (1.0f + w[d]));
    __syncthreads();
    if (d >= hd) return;
    int rd = rot < hd ? rot : hd;
    if (rd > 2 * nfreq) rd = 2 * nfreq;
    const int hf = rd >> 1;
    float out = sv[d];
    if (d < rd) {
        const int i = d < hf ? d : d - hf;
        const float ang = (float)(pos0 + (int)item * pstep) * invf[i];
        const float a = cosf(ang), b = sinf(ang);
        out = d < hf ? sv[d] * a - sv[d + hf] * b : sv[d] * a + sv[d - hf] * b;
    }
    dst[item * dstride + (u64)(head / dgroup) * dgstride + (u64)(head % dgroup) * dhead + d] = out;
}

// Indexer block scores: sc[t][b] = sum_h max(DS[t][h][b], 0) * inv.
extern "C" __global__ void q4_idx_reduce(const float* __restrict__ ds, float* __restrict__ sc,
                                         int nb, int ih, int nact, float inv)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)nact * nb) return;
    const u64 t = i / nb, b = i - t * nb;
    float s = 0.0f;
    for (int h = 0; h < ih; h++) s += fmaxf(ds[(t * ih + h) * nb + b], 0.0f);
    sc[i] = s * inv;
}

// The `keep` best blocks of every token (a block per token), ties to the
// lower index: bits of maskb[row][b / 32]. Token row r sits at position
// p0 + r and sees the blocks b < (p0 + r + 1) / cr. Scores are >= 0, so
// their bit patterns order as the values do: the threshold is found by
// bisection over the 32 bits. A thread owns `per` blocks (a multiple of 32).
extern "C" __global__ void q4_idx_topk(const float* __restrict__ sc, u32* __restrict__ maskb,
                                       int nb, int mwords, int keep, int p0, int cr, int per)
{
    __shared__ int total;
    __shared__ int eqc[256];
    const int r = blockIdx.x;
    int complete = (p0 + r + 1) / cr;
    if (complete > nb) complete = nb;
    const u32* row = (const u32*)(sc + (u64)r * nb);
    const int b0 = threadIdx.x * per;
    int b1 = b0 + per;
    if (b1 > complete) b1 = complete;
    u32 T = 0;
    for (int bit = 31; bit >= 0; bit--) {
        const u32 cand = T | (1u << bit);
        if (threadIdx.x == 0) total = 0;
        __syncthreads();
        int c = 0;
        for (int b = b0; b < b1; b++) c += row[b] >= cand;
        if (c) atomicAdd(&total, c);
        __syncthreads();
        if (total >= keep) T = cand;
        __syncthreads();
    }
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    int gt = 0, eq = 0;
    for (int b = b0; b < b1; b++) { gt += row[b] > T; eq += row[b] == T; }
    eqc[threadIdx.x] = eq;
    if (gt) atomicAdd(&total, gt);
    __syncthreads();
    const int need_eq = keep - total;
    int rank = 0;
    for (int i = 0; i < (int)threadIdx.x; i++) rank += eqc[i];
    for (int wi = 0; wi < per / 32; wi++) {
        const int wb = b0 + wi * 32;
        if (wb / 32 >= mwords) break;
        u32 m = 0;
        for (int k = 0; k < 32; k++) {
            const int b = wb + k;
            if (b >= b1) break;
            const u32 v = row[b];
            if (v > T) m |= 1u << k;
            else if (v == T) { if (rank < need_eq) m |= 1u << k; rank++; }
        }
        maskb[(u64)r * mwords + wb / 32] = m;
    }
}

// Softmax down every column of S [L] x [cols] in place, a block per column.
// Column c belongs to the token at position ppos0 + c / hpt, which sees
// j < npos: all of them while the indexer is idle (complete <= keep), else
// the kept blocks plus the tail past the last complete block.
extern "C" __global__ void q4_qsa_softmax(float* __restrict__ S, const u32* __restrict__ maskb,
                                          int L, int hpt, int ppos0, int mask_p0, int mwords, int keep, int cr)
{
    __shared__ float red[256];
    float* col = S + (u64)blockIdx.x * L;
    const int p = ppos0 + blockIdx.x / hpt;
    const int npos = p + 1;
    const int complete = npos / cr;
    const bool idx = complete > keep;
    const u32* mrow = idx ? maskb + (u64)(p - mask_p0) * mwords : (const u32*)0;
    const int tail = complete * cr;
    float mx = -__int_as_float(0x7f800000);
    for (int j = threadIdx.x; j < npos; j += blockDim.x) {
        bool vis = true;
        if (idx && j < tail) { const int b = j / cr; vis = (mrow[b >> 5] >> (b & 31)) & 1u; }
        if (vis) mx = fmaxf(mx, col[j]);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float z = 0.0f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        bool vis = j < npos;
        if (vis && idx && j < tail) { const int b = j / cr; vis = (mrow[b >> 5] >> (b & 31)) & 1u; }
        const float e = vis ? expf(col[j] - mx) : 0.0f;
        col[j] = e;
        z += e;
    }
    red[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float zi = 1.0f / fmaxf(red[0], 1.17549435e-38f);
    for (int j = threadIdx.x; j < L; j += blockDim.x) col[j] *= zi;
}

// merged [n][nh][hd] (wt) = O (group-major, as Q) * sigmoid(gate half of QG).
extern "C" __global__ void q4_qsa_merge(const float* __restrict__ O, const float* __restrict__ qg,
                                        void* __restrict__ mergedh, int n, int nh, int groups, int hd, int half)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * nh * hd) return;
    const u64 t = i / ((u64)nh * hd);
    const u64 r = i - t * nh * hd;
    const int qh = (int)(r / hd), d = (int)(r - (u64)qh * hd);
    const int kvh = qh / groups, g = qh - kvh * groups;
    const float v = O[(((u64)kvh * n + t) * groups + g) * hd + d];
    stf(mergedh, i, v * sigm(qg[t * (u64)nh * 2 * hd + (u64)qh * 2 * hd + hd + d]), half);
}

// ---------------------------------------------------------------- PLE

// gated [n][hc][hid] = sigmoid(signed sqrt of <key, query> / sqrt(hid)) * value,
// a block per (token, stream).
extern "C" __global__ void q4_ple_gate(const float* __restrict__ key, const float* __restrict__ query,
                                       const float* __restrict__ value, float* __restrict__ gated, int hid, int hc)
{
    __shared__ float red[256];
    const u64 b = blockIdx.x;
    const u64 t = b / hc;
    const float* kr = key + b * hid;
    const float* qr = query + b * hid;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) acc += kr[i] * qr[i];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float dot = red[0] / sqrtf((float)hid);
    const float g = sigm(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    for (int i = threadIdx.x; i < hid; i += blockDim.x) gated[b * hid + i] = g * value[t * hid + i];
}

// hyper += silu(dilated causal depthwise conv of normed) + gated; rows before
// the chunk come from hist [(kernel-1) * dil][width] (newest last).
extern "C" __global__ void q4_ple_conv(const float* __restrict__ normed, const float* __restrict__ hist,
                                       const float* __restrict__ conv, const float* __restrict__ gated,
                                       float* __restrict__ hyper, int n, int width, int kernel, int dil)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * width) return;
    const int t = (int)(i / width);
    const u64 ch = i - (u64)t * width;
    const int cap = (kernel - 1) * dil;
    const float* tp = conv + ch * kernel;
    float sum = tp[kernel - 1] * normed[i];
    for (int tap = 0; tap < kernel - 1; tap++) {
        const int src = t - (kernel - 1 - tap) * dil;
        const float v = src >= 0 ? normed[(u64)src * width + ch] : hist[(u64)(cap + src) * width + ch];
        sum += tp[tap] * v;
    }
    hyper[i] += silu_f(sum) + gated[i];
}

// ---------------------------------------------------------------- MoE

// Softmax router: the top `topk` of the first `ne` logits of every token (a
// block per token), ties to the lower index, weights renormalised over the
// picks in rank order. logits [n][R], R >= ne.
extern "C" __global__ void q4_route(const float* __restrict__ logits, int* __restrict__ picks,
                                    float* __restrict__ wts, int ne, int R, int topk, float tail,
                                    const unsigned long long* __restrict__ tbl)
{
    __shared__ float pe[1024];
    __shared__ float red[256];
    __shared__ int spick[32];
    __shared__ float sw[32];
    const float* l = logits + (u64)blockIdx.x * R;
    float mx = -__int_as_float(0x7f800000);
    for (int i = threadIdx.x; i < ne; i += blockDim.x) if (!tbl || tbl[i]) mx = fmaxf(mx, l[i]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    for (int i = threadIdx.x; i < ne; i += blockDim.x) pe[i] = (!tbl || tbl[i]) ? expf(l[i] - mx) : 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < ne; i++) s += pe[i];
        red[0] = s;
    }
    __syncthreads();
    const float s = red[0];
    __syncthreads();
    for (int i = threadIdx.x; i < ne; i += blockDim.x) pe[i] /= s;
    __syncthreads();
    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        const float p = pe[i];
        int rank = 0;
        for (int j = 0; j < ne; j++) rank += (pe[j] > p) || (pe[j] == p && j < i);
        if (rank < topk) { spick[rank] = i; sw[rank] = p; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ws = 0.0f;
        for (int r = 0; r < topk; r++) ws += sw[r];
        // the route tail: picks lighter than `tail` of the picked weight are dropped
        if (tail > 0.0f && ws > 0.0f) {
            const float cut = tail * ws;
            float keep = 0.0f;
            for (int r = 0; r < topk; r++) {
                if (sw[r] < cut) { sw[r] = 0.0f; spick[r] = -1; } else keep += sw[r];
            }
            if (keep > 0.0f) ws = keep;
        }
        for (int r = 0; r < topk; r++) {
            picks[(u64)blockIdx.x * topk + r] = spick[r];
            wts[(u64)blockIdx.x * topk + r] = sw[r] / ws;
        }
    }
}

// q4tp experts -> dense wt planes. A raw expert slot holds gate [inter][hid],
// up [inter][hid], down [hid][inter], each as [nibbles][row (f16 lo, f16
// step)][5-bit group codes]. Grid (words of a matrix / 256, 3 * B): a thread
// expands one u32 of nibbles = 8 weights of matrix blockIdx.y % 3 of expert
// blockIdx.y / 3 into plane j of Wg / Wu / Wd.
extern "C" __global__ void q4_expand_exp(const u64* __restrict__ slots, void* __restrict__ Wg, void* __restrict__ Wu,
                                         void* __restrict__ Wd, int inter, int hid, int half)
{
    const int m = blockIdx.y % 3, j = blockIdx.y / 3;
    const int rows = m == 2 ? hid : inter, cols = m == 2 ? inter : hid;
    const u32 wi = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 words = (u64)rows * cols / 8;
    if (wi >= words) return;
    // section offsets of a matrix, and of the three matrices in the slot
    const int gpr_gu = hid >> 5;
    const u64 size_gu = (u64)inter * gpr_gu * 16 + (u64)inter * 4 + (u64)inter * ((gpr_gu * 5 + 7) / 8);
    const int gpr = cols >> 5;
    const u64 params_off = (u64)rows * gpr * 16, codes_off = params_off + (u64)rows * 4;
    const int stride = (gpr * 5 + 7) / 8;
    const u8* mat = (const u8*)slots[j] + (u64)m * size_gu;
    const u64 base = (u64)wi * 8;
    const int r = (int)(base / cols);
    const int g = (int)(base - (u64)r * cols) >> 5;
    const u32 w = *(const u32*)(mat + (u64)wi * 4);
    const u16* pp = (const u16*)(mat + params_off + (u64)r * 4);
    const u8* codes = mat + codes_off + (u64)r * stride;
    const int bit = g * 5, cb = bit >> 3, sh = bit & 7;
    const u32 lo = codes[cb];
    const u32 hi = (sh > 3 && cb + 1 < stride) ? codes[cb + 1] : 0u;
    const int code = (int)(((lo | (hi << 8)) >> sh) & 0x1Fu);
    float s = exp2f(h2f(pp[0]));
    const float ratio = exp2f(h2f(pp[1]));
    for (int c = 0; c < code; c++) s *= ratio;
    void* W = m == 0 ? Wg : (m == 1 ? Wu : Wd);
    const u64 o = (u64)j * rows * cols + base;
    for (int k = 0; k < 4; k++) {
        const u32 by = (w >> (8 * k)) & 0xFFu;
        stf(W, o + 2 * k, ((float)(by & 15u) - 8.0f) * s, half);
        stf(W, o + 2 * k + 1, ((float)(by >> 4) - 8.0f) * s, half);
    }
}

// q8_2f -> dense wt [rows][cols]: body i8 * row scale * column field, a thread
// per 4 weights.
extern "C" __global__ void q4_expand_q8(const u32* __restrict__ body, const float* __restrict__ rs,
                                        const float* __restrict__ col, void* __restrict__ W,
                                        int rows, int cols, int half)
{
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const int qpr = cols >> 2;
    if (idx >= (u64)rows * qpr) return;
    const int r = (int)(idx / qpr);
    const int q = (int)(idx - (u64)r * qpr);
    const u32 w = body[idx];
    const float s = rs[r];
    const float* c = col + (u64)q * 4;
    stf(W, idx * 4, (float)((int)(w << 24) >> 24) * s * c[0], half);
    stf(W, idx * 4 + 1, (float)((int)(w << 16) >> 24) * s * c[1], half);
    stf(W, idx * 4 + 2, (float)((int)(w << 8) >> 24) * s * c[2], half);
    stf(W, idx * 4 + 3, (float)((int)w >> 24) * s * c[3], half);
}

// f32 -> wt.
extern "C" __global__ void q4_to_wt(const float* __restrict__ src, void* __restrict__ dst, u64 n, int half)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(dst, i, src[i], half);
}

extern "C" __global__ void q4_zero(float* __restrict__ p, u64 n)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

// Xg [rows][hid] (wt) = the token rows of xh named by rowtok, zeros for -1.
extern "C" __global__ void q4_gather(const void* __restrict__ xh, const int* __restrict__ rowtok,
                                     void* __restrict__ xg, int rows, int hid, int half)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * hid) return;
    const u64 r = i / hid;
    const int tok = rowtok[r];
    const u64 src = (u64)(tok < 0 ? 0 : tok) * hid + (i - r * hid);
    if (half) ((u16*)xg)[i] = tok < 0 ? (u16)0 : ((const u16*)xh)[src];
    else ((float*)xg)[i] = tok < 0 ? 0.0f : ((const float*)xh)[src];
}

// a = silu(g) * u * inv over n values; g, u in `in_half` type, a in `out_half`.
extern "C" __global__ void q4_act(const void* __restrict__ g, const void* __restrict__ u, void* __restrict__ a,
                                  u64 n, int in_half, int out_half, float inv)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(a, i, silu_f(ldf(g, i, in_half)) * ldf(u, i, in_half) * inv, out_half);
}

// out [n][hid] (+)= sum over the token's picks in this batch of wts * D[row].
// tokrow [n][topk]: the row of D for the pick, -1 = not in this batch.
extern "C" __global__ void q4_scatter(const void* __restrict__ D, const int* __restrict__ tokrow,
                                      const float* __restrict__ wts, float* __restrict__ out,
                                      int n, int hid, int topk, int d_half, int init)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, d = i - t * hid;
    float acc = 0.0f;
    for (int k = 0; k < topk; k++) {
        const int row = tokrow[t * topk + k];
        if (row >= 0) acc += wts[t * topk + k] * ldf(D, (u64)row * hid + d, d_half);
    }
    out[i] = init ? acc : out[i] + acc;
}

// out [n][hid] += g * ds [n][hid], g = sigmoid(logits[t][gcol]) or 1 (gcol < 0).
extern "C" __global__ void q4_shared_add(float* __restrict__ out, const float* __restrict__ ds,
                                         const float* __restrict__ logits, int n, int hid, int R, int gcol)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid;
    const float g = gcol < 0 ? 1.0f : sigm(logits[t * R + gcol]);
    out[i] += g * ds[i];
}

// ---------------------------------------------------------------- small path
// A decode step or a verify round: a few token rows (NB <= 8) straight from
// the stored weights — int8 bodies, f32 planes, raw q4tp expert slots — in
// f32, with no dense planes and no cuBLAS call. A weight is read once and
// used for every row of the round.

// A matvec job of `q4_mvb*`: the layout of the backend's MvJob.
struct MvJob { u64 w, aux0, aux1, y, res; int rows, kind, blk0, rpb; };

// Eight 4-bit weights of a word against eight staged values `st` apart.
__device__ __forceinline__ float nib8(u32 w, const float* p, int st, float a) {
    a += (float)(w & 15u) * p[0];
    a += (float)((w >> 4) & 15u) * p[st];
    a += (float)((w >> 8) & 15u) * p[2 * st];
    a += (float)((w >> 12) & 15u) * p[3 * st];
    a += (float)((w >> 16) & 15u) * p[4 * st];
    a += (float)((w >> 20) & 15u) * p[5 * st];
    a += (float)((w >> 24) & 15u) * p[6 * st];
    a += (float)(w >> 28) * p[7 * st];
    return a;
}

// The 5-bit ladder code of group g of a row.
__device__ __forceinline__ int q4_code(const u8* codes, int g, int stride) {
    const int bit = g * 5, cb = bit >> 3, sh = bit & 7;
    const u32 lo = codes[cb];
    const u32 hi = (sh > 3 && cb + 1 < stride) ? codes[cb + 1] : 0u;
    return (int)(((lo | (hi << 8)) >> sh) & 0x1Fu);
}

// The 32 scales of a row: t[0] = 2^lo, t[c] = t[c-1] * 2^step, as the host.
__device__ __forceinline__ void q4_ladder(const u16* pp, float* l) {
    float s = exp2f(h2f(pp[0]));
    const float ratio = exp2f(h2f(pp[1]));
    l[0] = s;
    for (int c = 1; c < 32; c++) { s *= ratio; l[c] = s; }
}

// Up to five matvecs of one input over NB rows: y_j [NB][rows_j] = W_j x.
// x [NB] rows `xstride` apart, `cols` wide. Kind 7 = q8_2f (int8 body, row
// scale aux0, column field aux1), kind 4 = f32. A block serves `rpb` rows of
// the job its index falls into; shared memory holds the NB staged inputs.
template <int NB>
__device__ __forceinline__ void mvb_t(MvJob j0, MvJob j1, MvJob j2, MvJob j3, MvJob j4, int njobs,
                                      const float* __restrict__ x, int xstride, int cols)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int bx = blockIdx.x;
    MvJob jb = j0;
    if (njobs > 1 && bx >= j1.blk0) jb = j1;
    if (njobs > 2 && bx >= j2.blk0) jb = j2;
    if (njobs > 3 && bx >= j3.blk0) jb = j3;
    if (njobs > 4 && bx >= j4.blk0) jb = j4;
    const int row0 = (bx - jb.blk0) * jb.rpb;
    const int row1 = min(row0 + jb.rpb, jb.rows);
    float* y = (float*)jb.y;
    if (jb.kind == 7) {
        const int nch = cols >> 4, CP = nch | 1, XS = 16 * CP;
        const float* col = (const float*)jb.aux1;
        for (int i = tid; i < cols; i += T) {
            const float cv = col[i];
            const int idx = (i & 15) * CP + (i >> 4);
            for (int b = 0; b < NB; b++) sh[b * XS + idx] = x[(size_t)b * xstride + i] * cv;
        }
        __syncthreads();
        const float* rs = (const float*)jb.aux0;
        for (int r = row0 + warp; r < row1; r += NW) {
            const uint4* W = (const uint4*)(jb.w + (size_t)r * cols);
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            for (int c = lane; c < nch; c += 32) {
                const uint4 w = W[c];
                float wf[16];
                wf[0] = (float)((int)(w.x << 24) >> 24);  wf[1] = (float)((int)(w.x << 16) >> 24);
                wf[2] = (float)((int)(w.x << 8) >> 24);   wf[3] = (float)((int)w.x >> 24);
                wf[4] = (float)((int)(w.y << 24) >> 24);  wf[5] = (float)((int)(w.y << 16) >> 24);
                wf[6] = (float)((int)(w.y << 8) >> 24);   wf[7] = (float)((int)w.y >> 24);
                wf[8] = (float)((int)(w.z << 24) >> 24);  wf[9] = (float)((int)(w.z << 16) >> 24);
                wf[10] = (float)((int)(w.z << 8) >> 24);  wf[11] = (float)((int)w.z >> 24);
                wf[12] = (float)((int)(w.w << 24) >> 24); wf[13] = (float)((int)(w.w << 16) >> 24);
                wf[14] = (float)((int)(w.w << 8) >> 24);  wf[15] = (float)((int)w.w >> 24);
                for (int b = 0; b < NB; b++) {
                    const float* p = sh + b * XS + c;
                    float a = 0.0f;
                    for (int k = 0; k < 16; k++) a += wf[k] * p[k * CP];
                    acc[b] += a;
                }
            }
            const float sc = rs[r];
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a * sc;
            }
        }
    } else {
        for (int i = tid; i < cols; i += T)
            for (int b = 0; b < NB; b++) sh[b * cols + i] = x[(size_t)b * xstride + i];
        __syncthreads();
        for (int r = row0 + warp; r < row1; r += NW) {
            const float* W = (const float*)jb.w + (size_t)r * cols;
            float acc[NB];
            for (int b = 0; b < NB; b++) acc[b] = 0.0f;
            for (int c = lane; c < cols; c += 32) {
                const float wv = W[c];
                for (int b = 0; b < NB; b++) acc[b] += wv * sh[b * cols + c];
            }
            for (int b = 0; b < NB; b++) {
                float a = acc[b];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) y[(size_t)b * jb.rows + r] = a;
            }
        }
    }
}

// The 16 bytes of a word of a wt row as floats: eight f16 or four f32.
__device__ __forceinline__ void wt_dec(const uint4 w, int half, float* wf) {
    if (half) {
        wf[0] = h2f((u16)w.x); wf[1] = h2f((u16)(w.x >> 16));
        wf[2] = h2f((u16)w.y); wf[3] = h2f((u16)(w.y >> 16));
        wf[4] = h2f((u16)w.z); wf[5] = h2f((u16)(w.z >> 16));
        wf[6] = h2f((u16)w.w); wf[7] = h2f((u16)(w.w >> 16));
    } else {
        wf[0] = __int_as_float((int)w.x); wf[1] = __int_as_float((int)w.y);
        wf[2] = __int_as_float((int)w.z); wf[3] = __int_as_float((int)w.w);
    }
}
// Its dot with staged values `st` apart.
__device__ __forceinline__ float wt_acc(const float* wf, const float* p, int st, int half) {
    float a = wf[0] * p[0] + wf[1] * p[st] + wf[2] * p[2 * st] + wf[3] * p[3 * st];
    if (half) a += wf[4] * p[4 * st] + wf[5] * p[5 * st] + wf[6] * p[6 * st] + wf[7] * p[7 * st];
    return a;
}

// Words a lane reads ahead of their use: the loads are in flight together,
// where one load a step would wait out the memory latency each time.
#define Q4_AHEAD 8

// The low-rank middle of a gated residual over NB rows, by column chunks:
// part [chunk][NB][R] = <W[r][chunk], x[b][chunk]>, W [R][Wd] (wt). Grid
// (ceil(R / warps), ceil(Wd / cw)): a block stages its chunk of the NB inputs
// and gives a warp to each of its rows, 16 bytes a lane a load. `q4_hc_up*`
// sums the chunks.
template <int NB>
__device__ __forceinline__ void hc_di_t(const void* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ part, int R, int Wd, int cw, int half)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int c0 = blockIdx.y * cw, cn = min(cw, Wd - c0);
    const int ps = half ? 3 : 2, per = 1 << ps, nl = cn >> ps, CP = nl | 1, XS = per * CP;
    for (int i = tid; i < cn; i += T) {
        const int idx = (i & (per - 1)) * CP + (i >> ps);
        for (int b = 0; b < NB; b++) sh[b * XS + idx] = x[(u64)b * Wd + c0 + i];
    }
    __syncthreads();
    const int r = blockIdx.x * NW + warp;
    if (r >= R) return;
    const uint4* W = (const uint4*)((const char*)Wp + ((u64)r * Wd + c0) * (half ? 2 : 4));
    float acc[NB];
    for (int b = 0; b < NB; b++) acc[b] = 0.0f;
    for (int c1 = lane; c1 < nl; c1 += 32 * Q4_AHEAD) {
        uint4 w[Q4_AHEAD];
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int c = c1 + 32 * k;
            if (c < nl) w[k] = W[c];
        }
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int c = c1 + 32 * k;
            if (c < nl) {
                float wf[8];
                wt_dec(w[k], half, wf);
                for (int b = 0; b < NB; b++) acc[b] += wt_acc(wf, sh + b * XS + c, CP, half);
            }
        }
    }
    for (int b = 0; b < NB; b++) {
        float a = acc[b];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) part[((u64)blockIdx.y * NB + b) * R + r] = a;
    }
}

// The fold of a gated residual over NB rows, a warp per output dim d. First
// the chunk sums of `q4_hc_di*`: v = sum / hc, the first `lr` rows as
// silu(v) into shared memory, the rest as iw [NB][R - lr] = 2 sigmoid(v)
// (block 0). Then folded[b][d] = sum over streams s of
// sigmoid(<W[s * hid + d], low[b]>) * normed[b][s][d] / hc, W [hc * hid][lr]
// (wt).
template <int NB>
__device__ __forceinline__ void hc_up_t(const void* __restrict__ Wp, const float* __restrict__ part,
                                        float* __restrict__ iw, const float* __restrict__ normed,
                                        float* __restrict__ folded, void* __restrict__ foldedh,
                                        int hid, int hc, int lr, int R, int nch, int half)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const int ps = half ? 3 : 2, per = 1 << ps, nl = lr >> ps, CP = nl | 1, XS = per * CP;
    for (int i = tid; i < NB * lr; i += T) {
        const int b = i / lr, c = i - b * lr;
        float a = 0.0f;
        for (int ch = 0; ch < nch; ch++) a += part[((u64)ch * NB + b) * R + c];
        sh[b * XS + (c & (per - 1)) * CP + (c >> ps)] = silu_f(a / (float)hc);
    }
    if (blockIdx.x == 0) {
        const int ni = R - lr;
        for (int i = tid; i < NB * ni; i += T) {
            const int b = i / ni, j = i - b * ni;
            float a = 0.0f;
            for (int ch = 0; ch < nch; ch++) a += part[((u64)ch * NB + b) * R + lr + j];
            iw[i] = 2.0f * sigm(a / (float)hc);
        }
    }
    __syncthreads();
    const int d = blockIdx.x * NW + warp;
    if (d >= hid) return;
    const u64 rowb = (u64)lr * (half ? 2 : 4);
    // the words of the hc rows of this d in one sequence: stream s, word it
    const int nit = (nl + 31) >> 5, nq = hc * nit;
    float f[NB], acc[NB];
    for (int b = 0; b < NB; b++) { f[b] = 0.0f; acc[b] = 0.0f; }
    for (int q0 = 0; q0 < nq; q0 += Q4_AHEAD) {
        uint4 w[Q4_AHEAD];
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int q = q0 + k;
            if (q < nq) {
                const int s = q / nit, c = lane + 32 * (q - s * nit);
                if (c < nl) w[k] = ((const uint4*)((const char*)Wp + ((u64)s * hid + d) * rowb))[c];
            }
        }
        for (int k = 0; k < Q4_AHEAD; k++) {
            const int q = q0 + k;
            if (q >= nq) break;
            const int s = q / nit, it = q - s * nit, c = lane + 32 * it;
            if (c < nl) {
                float wf[8];
                wt_dec(w[k], half, wf);
                for (int b = 0; b < NB; b++) acc[b] += wt_acc(wf, sh + b * XS + c, CP, half);
            }
            if (it == nit - 1) {
                for (int b = 0; b < NB; b++) {
                    float a = acc[b];
                    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                    if (lane == 0) f[b] += sigm(a) * normed[((u64)b * hc + s) * hid + d] / (float)hc;
                    acc[b] = 0.0f;
                }
            }
        }
    }
    if (lane == 0) {
        for (int b = 0; b < NB; b++) {
            folded[(u64)b * hid + d] = f[b];
            if (foldedh) stf(foldedh, (u64)b * hid + d, f[b], half);
        }
    }
}

#define Q4_SMALL(NB) \
    extern "C" __global__ void q4_mvb##NB(MvJob j0, MvJob j1, MvJob j2, MvJob j3, MvJob j4, int njobs, \
                                          const float* __restrict__ x, int xstride, int cols) \
    { mvb_t<NB>(j0, j1, j2, j3, j4, njobs, x, xstride, cols); } \
    extern "C" __global__ void q4_hc_di##NB(const void* __restrict__ Wp, const float* __restrict__ x, \
                                            float* __restrict__ part, int R, int Wd, int cw, int half) \
    { hc_di_t<NB>(Wp, x, part, R, Wd, cw, half); } \
    extern "C" __global__ void q4_hc_up##NB(const void* __restrict__ Wp, const float* __restrict__ part, \
                                            float* __restrict__ iw, const float* __restrict__ normed, \
                                            float* __restrict__ folded, void* __restrict__ foldedh, \
                                            int hid, int hc, int lr, int R, int nch, int half) \
    { hc_up_t<NB>(Wp, part, iw, normed, folded, foldedh, hid, hc, lr, R, nch, half); }
Q4_SMALL(1) Q4_SMALL(2) Q4_SMALL(3) Q4_SMALL(4) Q4_SMALL(5) Q4_SMALL(6) Q4_SMALL(7) Q4_SMALL(8)

// A unit of the expert kernels: one expert's raw slot and up to Q4_UR token
// rows that picked it, as 2 + Q4_UR words (slot lo, hi, then the rows, -1
// from the first free place on).
#define Q4_UR 4
// Words of a matrix row a lane reads ahead: gate and up together, down.
#define Q4_GW 3
#define Q4_DW 4

// One group of a row against the staged input of a unit row: the nibbles of
// the 16 bytes minus the bias of 8, times the group's scale.
__device__ __forceinline__ float q4_group(const uint4 w, float lc, const float* p, int GP, float gsum) {
    float a = nib8(w.x, p, GP, 0.0f);
    a = nib8(w.y, p + 8 * GP, GP, a);
    a = nib8(w.z, p + 16 * GP, GP, a);
    a = nib8(w.w, p + 24 * GP, GP, a);
    return lc * (a - gsum);
}

// Gate and up of the round's units, straight from the raw slots:
// act [U * Q4_UR][inter] = silu(gate x) * (up x) for each row of a unit,
// folded [n][hid]. Grid (ceil(inter / rb), U): a block does `rb` rows of both
// matrices, its warps a row at a time; the rows' scale ladders sit in shared
// memory. A weight is read once for all the rows of the unit.
extern "C" __global__ void q4_exp_gu(const u32* __restrict__ units, const float* __restrict__ folded,
                                     float* __restrict__ act, int inter, int hid, int rb)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = hid >> 5, GP = gpr | 1, XS = 32 * GP, stride = (gpr * 5 + 7) / 8;
    const u64 params_off = (u64)inter * gpr * 16, codes_off = params_off + (u64)inter * 4;
    const u64 sgu = codes_off + (u64)inter * stride;
    float* xs = sh;                 // [cnt][32][GP]
    float* gs = xs + cnt * XS;      // [cnt][gpr]
    float* lad = gs + cnt * gpr;    // [2][rb][32]
    const int r0 = blockIdx.x * rb;
    for (int j = 0; j < cnt; j++) {
        const float* x = folded + (u64)un[2 + j] * hid;
        float* xj = xs + j * XS;
        for (int i = tid; i < hid; i += T) xj[(i & 31) * GP + (i >> 5)] = x[i];
    }
    if (tid < 2 * rb) {
        const int m = tid / rb, r = r0 + tid - m * rb;
        if (r < inter) q4_ladder((const u16*)(slot + (u64)m * sgu + params_off + (u64)r * 4), lad + tid * 32);
    }
    __syncthreads();
    for (int i = tid; i < cnt * gpr; i += T) {
        const int j = i / gpr, g = i - j * gpr;
        const float* p = xs + j * XS + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
        gs[i] = 8.0f * s;
    }
    __syncthreads();
    // the words of this warp's rows in one sequence: row q, word it
    const int nit = (gpr + 31) >> 5;
    const int nz = warp < rb ? ((rb - warp + NW - 1) / NW) * nit : 0;
    const u8* mg = (const u8*)slot;
    const u8* mu = mg + sgu;
    float ag[Q4_UR], au[Q4_UR];
    for (int j = 0; j < Q4_UR; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
    for (int z0 = 0; z0 < nz; z0 += Q4_GW) {
        uint4 wg[Q4_GW], wu[Q4_GW];
        float lg[Q4_GW], lu[Q4_GW];
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z < nz) {
                const int q = z / nit, g = lane + 32 * (z - q * nit), i = warp + q * NW, r = r0 + i;
                if (r < inter && g < gpr) {
                    wg[k] = ((const uint4*)(mg + (u64)r * gpr * 16))[g];
                    wu[k] = ((const uint4*)(mu + (u64)r * gpr * 16))[g];
                    lg[k] = lad[i * 32 + q4_code(mg + codes_off + (u64)r * stride, g, stride)];
                    lu[k] = lad[(rb + i) * 32 + q4_code(mu + codes_off + (u64)r * stride, g, stride)];
                }
            }
        }
        for (int k = 0; k < Q4_GW; k++) {
            const int z = z0 + k;
            if (z >= nz) break;
            const int q = z / nit, it = z - q * nit, g = lane + 32 * it, i = warp + q * NW, r = r0 + i;
            if (r >= inter) break;
            if (g < gpr) {
                for (int j = 0; j < cnt; j++) {
                    const float* p = xs + j * XS + g;
                    const float gsum = gs[j * gpr + g];
                    ag[j] += q4_group(wg[k], lg[k], p, GP, gsum);
                    au[j] += q4_group(wu[k], lu[k], p, GP, gsum);
                }
            }
            if (it == nit - 1) {
                for (int j = 0; j < cnt; j++) {
                    float a = ag[j], b = au[j];
                    for (int o = 16; o > 0; o >>= 1) {
                        a += __shfl_xor_sync(0xffffffffu, a, o);
                        b += __shfl_xor_sync(0xffffffffu, b, o);
                    }
                    if (lane == 0) act[((u64)blockIdx.y * Q4_UR + j) * inter + r] = silu_f(a) * b;
                    ag[j] = 0.0f;
                    au[j] = 0.0f;
                }
            }
        }
    }
}

// Down of the round's units: part [U * Q4_UR][hid] = down act for each row of
// a unit. Grid (ceil(hid / rb), U): a block does `rb` output rows of one
// unit, its warps a row at a time.
extern "C" __global__ void q4_exp_d(const u32* __restrict__ units, const float* __restrict__ act,
                                    float* __restrict__ part, int inter, int hid, int rb)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, T = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const u32* un = units + (u64)blockIdx.y * (2 + Q4_UR);
    const u64 slot = (u64)un[0] | ((u64)un[1] << 32);
    if (!slot) return;
    int cnt = 0;
    while (cnt < Q4_UR && (int)un[2 + cnt] >= 0) cnt++;
    const int gpr = inter >> 5, GP = gpr | 1, XS = 32 * GP, stride = (gpr * 5 + 7) / 8;
    const int gpr_gu = hid >> 5;
    const u64 sgu = (u64)inter * gpr_gu * 16 + (u64)inter * 4 + (u64)inter * ((gpr_gu * 5 + 7) / 8);
    const u64 params_off = (u64)hid * gpr * 16, codes_off = params_off + (u64)hid * 4;
    const u8* mat = (const u8*)(slot + 2 * sgu);
    float* xs = sh;                 // [cnt][32][GP]
    float* gs = xs + cnt * XS;      // [cnt][gpr]
    float* lad = gs + cnt * gpr;    // [rb][32]
    const int r0 = blockIdx.x * rb;
    for (int j = 0; j < cnt; j++) {
        const float* a = act + ((u64)blockIdx.y * Q4_UR + j) * inter;
        float* xj = xs + j * XS;
        for (int i = tid; i < inter; i += T) xj[(i & 31) * GP + (i >> 5)] = a[i];
    }
    if (tid < rb && r0 + tid < hid) q4_ladder((const u16*)(mat + params_off + (u64)(r0 + tid) * 4), lad + tid * 32);
    __syncthreads();
    for (int i = tid; i < cnt * gpr; i += T) {
        const int j = i / gpr, g = i - j * gpr;
        const float* p = xs + j * XS + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
        gs[i] = 8.0f * s;
    }
    __syncthreads();
    const int nit = (gpr + 31) >> 5;
    const int nz = warp < rb ? ((rb - warp + NW - 1) / NW) * nit : 0;
    float acc[Q4_UR];
    for (int j = 0; j < Q4_UR; j++) acc[j] = 0.0f;
    for (int z0 = 0; z0 < nz; z0 += Q4_DW) {
        uint4 w[Q4_DW];
        float lc[Q4_DW];
        for (int k = 0; k < Q4_DW; k++) {
            const int z = z0 + k;
            if (z < nz) {
                const int q = z / nit, g = lane + 32 * (z - q * nit), i = warp + q * NW, r = r0 + i;
                if (r < hid && g < gpr) {
                    w[k] = ((const uint4*)(mat + (u64)r * gpr * 16))[g];
                    lc[k] = lad[i * 32 + q4_code(mat + codes_off + (u64)r * stride, g, stride)];
                }
            }
        }
        for (int k = 0; k < Q4_DW; k++) {
            const int z = z0 + k;
            if (z >= nz) break;
            const int q = z / nit, it = z - q * nit, g = lane + 32 * it, i = warp + q * NW, r = r0 + i;
            if (r >= hid) break;
            if (g < gpr) {
                for (int j = 0; j < cnt; j++) acc[j] += q4_group(w[k], lc[k], xs + j * XS + g, GP, gs[j * gpr + g]);
            }
            if (it == nit - 1) {
                for (int j = 0; j < cnt; j++) {
                    float a = acc[j];
                    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                    if (lane == 0) part[((u64)blockIdx.y * Q4_UR + j) * hid + r] = a;
                    acc[j] = 0.0f;
                }
            }
        }
    }
}

// The weighted sum of a token's picks: blk [n][hid] = sum over its K places
// of w_k * part[ref[t][k]] (ref < 0: no pick), with w_k = wts[t][k] for a
// routed pick and, for place `topk` (the shared expert),
// sigmoid(logits[t][gcol]) or 1 (gcol < 0).
extern "C" __global__ void q4_exp_sum(const int* __restrict__ ref, const float* __restrict__ part,
                                      const float* __restrict__ wts, const float* __restrict__ logits,
                                      float* __restrict__ blk, int n, int K, int topk, int hid, int R, int gcol)
{
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * hid) return;
    const u64 t = i / hid, r = i - t * hid;
    float a = 0.0f;
    for (int k = 0; k < K; k++) {
        const int p = ref[t * K + k];
        if (p < 0) continue;
        const float wk = k < topk ? wts[t * topk + k] : (gcol < 0 ? 1.0f : sigm(logits[t * R + gcol]));
        a += wk * part[(u64)p * hid + r];
    }
    blk[i] = a;
}

// The index of the largest value of each row [vocab] (a block per row), ties to the lower index.
extern "C" __global__ void q4_argmax(const float* __restrict__ rows, int* __restrict__ ids, int vocab)
{
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* l = rows + (u64)blockIdx.x * vocab;
    float best = -__int_as_float(0x7f800000);
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = l[i];
        if (v > best) { best = v; at = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = at;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            const float v = bv[threadIdx.x + s];
            const int j = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x])) {
                bv[threadIdx.x] = v;
                bi[threadIdx.x] = j;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[blockIdx.x] = bi[0] == 0x7fffffff ? 0 : bi[0];
}

// The MTP draft head's input: hyper [hc][hid] = h [hc][hid] + e [hid] (the
// embedding's projection on every stream).
extern "C" __global__ void q4_mtp_fuse(const float* __restrict__ hs, const float* __restrict__ e,
                                       float* __restrict__ hyper, int hid, int hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hid * hc) return;
    hyper[i] = hs[i] + e[i % hid];
}

// The lean path's units, built on the card from the route's picks, packed: one block
// counts the rows each expert in the table `tbl` (0: not in the arena, the pick is
// dropped) was picked by, scans the counts into the first unit of each expert, and
// writes the units (four rows to a unit, the expert's raw slot first) and, for every
// (token, place), the row of the partial outputs it takes (`refs [n][K]`, -1 for none).
// The shared expert's units (`nsh`, its slot `shared`) follow the experts'. The `umax`
// units of the launch that are not used stay empty (slot 0): the kernels skip them.
extern "C" __global__ void q4_units(const int* __restrict__ picks, const unsigned long long* __restrict__ tbl,
                                    unsigned long long shared, u32* __restrict__ units, int* __restrict__ refs,
                                    int n, int topk, int K, int ne, int umax, int nsh)
{
    __shared__ int sp[8 * 32];
    __shared__ int cnt[512];
    __shared__ int off[512];
    __shared__ int scan[256];
    const int np = n * topk, W = 2 + Q4_UR, T = blockDim.x, tid = threadIdx.x;
    for (int i = tid; i < np; i += T) sp[i] = picks[i];
    for (int i = tid; i < umax * W; i += T) units[i] = (i % W) >= 2 ? 0xffffffffu : 0u;
    __syncthreads();
    for (int e = tid; e < ne; e += T) {
        int cn = 0;
        if (tbl[e]) for (int i = 0; i < np; i++) cn += (sp[i] == e);
        cnt[e] = cn;
    }
    __syncthreads();
    const int e0 = 2 * tid, e1 = e0 + 1;
    const int u0 = e0 < ne ? (cnt[e0] + Q4_UR - 1) / Q4_UR : 0;
    const int u1 = e1 < ne ? (cnt[e1] + Q4_UR - 1) / Q4_UR : 0;
    scan[tid] = u0 + u1;
    __syncthreads();
    for (int d = 1; d < T; d <<= 1) {
        const int x = tid >= d ? scan[tid - d] : 0;
        __syncthreads();
        scan[tid] += x;
        __syncthreads();
    }
    const int base = scan[tid] - (u0 + u1);
    if (e0 < ne) off[e0] = base;
    if (e1 < ne) off[e1] = base + u0;
    const int total = scan[T - 1];
    __syncthreads();
    for (int e = tid; e < ne; e += T) {
        if (!cnt[e]) continue;
        const unsigned long long a = tbl[e];
        int rank = 0;
        for (int i = 0; i < np; i++) {
            if (sp[i] != e) continue;
            const int u = off[e] + rank / Q4_UR, j = rank % Q4_UR;
            u32* w = units + (u64)u * W;
            w[0] = (u32)a; w[1] = (u32)(a >> 32); w[2 + j] = (u32)(i / topk);
            refs[(i / topk) * K + (i % topk)] = u * Q4_UR + j;
            rank++;
        }
    }
    for (int i = tid; i < np; i += T) {
        const int e = sp[i];
        if (e < 0 || e >= ne || !tbl[e]) refs[(i / topk) * K + (i % topk)] = -1;
    }
    for (int g = tid; g < nsh; g += T) {
        u32* w = units + (u64)(total + g) * W;
        w[0] = (u32)shared; w[1] = (u32)(shared >> 32);
        for (int j = 0; j < Q4_UR; j++) {
            const int t = g * Q4_UR + j;
            if (t < n) { w[2 + j] = (u32)t; refs[t * K + topk] = (total + g) * Q4_UR + j; }
        }
    }
}
